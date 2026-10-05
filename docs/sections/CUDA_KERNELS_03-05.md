<!-- 分割元: docs/CUDA_KERNELS.md | このファイルは参照用です。原本（docs/CUDA_KERNELS.md）が権威です。 -->
## 3. ALE カーネル群

> **適用条件**: ALE rezone/remap は **2D_RZ のみ** で使用する。1D_SPH では ALE は無効（Lagrangian 固定。NUMERICS §3.3 参照）。
> 実装時は `if (cfg.main.geometry == "2D_RZ" && cfg.mesh.motion == "ale" && cfg.mesh.rezoning.enabled)` ガードで ALE Phase 全体を囲むこと。

### 3.1 A1: mesh_quality_check

```cpp
__global__ void mesh_quality_check(
    double* __restrict__ q_cell,     // [n_cells] out: セル品質指標
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    DeviceErrorFlags* __restrict__ error_flags,  // mesh_tangle: J_min < 0 検出（§0.6）
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 4 Gauss点でヤコビアン評価 → `q_c = J_min/J_max`（NUMERICS §3.3.2）
  - **退化ガード**: `J_max_eff = max(J_max, J_floor)` (J_floor = 1e-30)。`J_max < J_floor` の場合 `q_c = 0.0` とし `mesh_tangle` フラグ設定
  - `J_min < 0` 検出時: `atomicExch(&error_flags->mesh_tangle, 1)`（ARCHITECTURE §10.1: 即ERROR）
- 後段で CUB `DeviceReduce::Min` → `q_min < q_threshold` で rezone 発動判定
- **レジスタ**: ~15（4 Gauss点のヤコビアン値 + min/max 一時変数）
- **メモリ**: 4頂点座標読み込み（構造格子固定ストライド）、coalesced write（q_cell）

Related Hydro2D pre-commit diagnostics in
`src/hydro/corner_jacobian_quality.cu` reuse the same multiblock
RZ-volume cubic/root logic as the production mesh-quality dt limiter.
`mesh_quality_rz_volume_cell_margin_kernel` is default-off and writes only
per-requested-cell diagnostic `sigma`/safe-eta buffers for the
`[ring7_seam]` scaffold and the Ring7 seam optimizer B_V
acceptance check; it does not update dt, mesh, or state.  The active Ring7
path may pass candidate start coordinates and candidate cell volumes to the
same kernel via
`compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume()` so the
floor volume matches the proposed zero-time seam geometry.  The reactive
driver retry path only requests the StepStart Ring7 transaction after the
production dt-CFL reports `mesh_quality_rz_volume`; the candidate acceptance
still uses this same diagnostic-margin wrapper around the production kernel.
Increment 3a then builds the dedicated `[ring7_seam_packet]` geometry ledger on
the host and commits no CUDA state or coordinate writes.  Increment 4b keeps
the driven-pole cap remap host-side as well: it reuses the production
`compute_mesh_quality_dt_limit()` CUDA path as the proxy and recomputed
validation oracle, but the conservative pole-cap packet ledger itself is a
deterministic host transaction and launches no new CUDA kernels.

### 3.2 A2: winslow_jacobi_step

```cpp
__global__ void winslow_jacobi_step(
    double* __restrict__ x_r_new,    // [n_nodes] out
    double* __restrict__ x_z_new,    // [n_nodes] out
    const double* __restrict__ x_r,  // [n_nodes] in (前反復)
    const double* __restrict__ x_z,  // [n_nodes] in
    const uint8_t* __restrict__ node_flags, // 境界フラグ
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_nodes + 255) / 256`（境界ノードはカーネル内で早期リターン）
- **反復**: ホスト側で最大 `max_iterations`（既定20）回ループ（ダブルバッファで swap、NUMERICS §3.3.3 準拠）
- **境界ノード**: `NODE_BOUNDARY | NODE_AXIS | NODE_CENTER` フラグ付きノードは位置固定
- **node_flags ビットマスク定義**: `NODE_BOUNDARY=0x01, NODE_AXIS=0x02, NODE_CENTER=0x04`。
  `(node_flags[n] & (NODE_BOUNDARY|NODE_AXIS|NODE_CENTER)) != 0` のノードは rezone で位置固定。
  NODE_BOUNDARY: 計算領域外周ノード、NODE_AXIS: r=0 軸上ノード（2D_RZ のみ）、
  NODE_CENTER: 球対称中心ノード（1D_SPH のみ）。`node_flags` は `[n_nodes]` サイズ
- **計量係数 α,β**: 隣接ノード座標から on-the-fly 計算（NUMERICS §3.3.3 Winslow 計量係数、共有メモリ不要）
- **レジスタ**: ~20
- **メモリ**: 隣接ノード 8 点（十字+対角）の座標読み込み。構造格子固定ストライドで参照
- **収束判定**: ホスト側で毎反復ノードあたりユークリッド変位の最大値 `δ_rezone = max_n sqrt(Δr_n² + Δz_n²)` を CUB `DeviceReduce::Max` で評価。`δ_rezone < convergence_tol × Δl_min_global`（既定 convergence_tol=1e-6）で早期終了（NUMERICS §3.3.3、SPECIFICATION §6.4.2）。**MPI注意**: `Δl_min_global = MPI_Allreduce(MIN, Δl_min_local)` でグローバル化した値を使用すること（§9 Phase 5 参照）。ローカル Δl_min を使うとランク間で break 判定が不一致し halo_exchange デッドロックが発生する

### 3.3 A3: conservative_remap

```cpp
__global__ __launch_bounds__(256, 4) void conservative_remap(
    double* __restrict__ field_new,      // [n_cells] out
    const double* __restrict__ field_old, // [n_cells] in
    const double* __restrict__ vol_old,
    const double* __restrict__ vol_new,
    const double* __restrict__ x_r_old,  // old mesh
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,  // new mesh
    const double* __restrict__ x_z_new,
    int sweep_direction,                 // 0=r方向, 1=z方向（Strang-type交替スイープ、§3.3.4）
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 1方向 flux-based remap + Van Leerスロープリミッタ（NUMERICS §3.3.4）
- **方向分離**（NUMERICS §3.3.4 必須）: 2Dのremapをr方向・z方向に分離し逐次実行。
  Strang-type で交替スイープ: 偶数ステップ r→z、奇数ステップ z→r（O(Δt²) 精度維持）。
  **偶奇判定**: グローバルな hydro ステップ番号 `step_number` で決定する（A3 の呼び出し回数やフィールドインデックスではない）。
  `first_dir = (step_number % 2 == 0) ? 0 : 1; second_dir = 1 - first_dir;` として、
  全保存量を同一ステップ内で同一方向順序 (first_dir → second_dir) で remap する。
  各保存量（mass, momentum_r, momentum_z, e_i, e_e, volFrac）に対し、
  1方向ずつ計2回起動（sweep_direction パラメータで r/z を指定）
- **state_supply z-face flux**: z_bottom/z_top が active `state_supply` の場合、boundary face speed は projected node `v_z` ではなく `supply_u_z_cm_per_s` を直接使う。これにより mesh clamping（`x_z=z_min/z_max`, mesh `v_z=0`）と open-flow material flux（\(\rho_s u_{z,s} A\Delta t\)）を分離する。
- **レジスタ**: ~30
- **メモリ**: old/new mesh 座標 + フィールド値の読み込み（構造格子固定ストライド）
- **ワープ発散**: Van Leer リミッタの min/max 分岐があるが、全スレッドが同一パスを実行するケースが大半

### 3.4 A4: project_cell_velocity_to_nodes

```cpp
__global__ void project_cell_velocity_to_nodes(
    double* __restrict__ v_r_node,          // [n_nodes] out: ノード速度R成分
    double* __restrict__ v_z_node,          // [n_nodes] out: ノード速度Z成分
    const double* __restrict__ v_r_cell,    // [n_cells] in: セル中心速度R成分
    const double* __restrict__ v_z_cell,    // [n_cells] in: セル中心速度Z成分
    const double* __restrict__ rho,         // [n_cells] in: セル密度 [g/cm³]
    const double* __restrict__ vol,         // [n_cells] in: セル体積 [cm³]（質量重み m_c = rho[c]*vol[c]）
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_nodes+255)/256`
- **処理**: remap 後のセル中心量からノード速度を再構築（NUMERICS §3.3.4 節点速度投影）。
  各ノードに隣接するセル（最大4セル for 2D_RZ）の速度を**質量重み平均**で投影:
  `v_node = Σ_c (m_c × v_c) / Σ_c m_c`  （m_c = ρ_c × V_c、NUMERICS §3.3.4 準拠）
  - 1D_SPH: 左右2セルの質量重み平均（境界は1セル）
  - 2D_RZ: 4隣接セルの質量重み平均。軸・境界では隣接セル数に応じて動的に重み算出
- **境界ノード**: 隣接セル数が少ない（角: 1, 辺: 2）→ 実在セルのみで加重平均。投影後の velocity boundary mode は 0=free, 1=reflect, 2=fixed, 3=state_supply。mode 1 は reflect-style に法線成分をゼロ化し、mode 2 は全成分を固定する。mode 3 は state_supply z-boundary の material `v_z` を拘束せず、supplied/restored `u_z_cm_per_s` を保持する。
- **レジスタ**: ~10
- **メモリ**: 隣接セル値の読み込み（構造格子固定ストライド）。ノードへのcoalesced write

### 3.5 A5: normalize_volFrac

```cpp
__global__ void normalize_volFrac(
    double* __restrict__ volFrac,      // [n_cells × n_mat] in/out: 体積分率
    DeviceErrorFlags* __restrict__ error_flags,  // volfrac_degenerate 報告用
    int n_cells, int n_mat
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: ALE remap 後に `Σ_mat volFrac[c, mat] = 1` を強制する正規化カーネル（NUMERICS §3.3.4, ARCHITECTURE §4.3）。
  1. **非負クランプ**: `volFrac[c, m] = max(volFrac[c, m], 0.0)` （Van Leerスロープリミッタ由来の微小負値を除去。NUMERICS §3.3.4 準拠）
  2. `sum = Σ_{m=0}^{n_mat-1} volFrac[c, m]`（クランプ後の非負値の和）
  3. `sum ≥ ε_vf` (1e-30) の場合: `volFrac[c, m] /= sum`（全材料を正規化）
  4. `sum < ε_vf` の場合（退化）: `α* = argmax_m volFrac[c, m]`（クランプ前の値で判定）に対し `volFrac[c, α*] = 1.0`、他を `0.0` に設定。`error_flags->volfrac_degenerate` 設定（NUMERICS §3.3.4 退化ガード準拠）
- **呼び出し位置**: §9 Phase 5 ALE remap 直後（A3 × 2回 の後）
- **レジスタ**: ~8
- **メモリ**: `volFrac` を1パスで読み書き。coalesced access（材料次元が内側）
- **注意**: 単材料（`n_mat == 1`）の場合は `volFrac[c, 0] = 1.0` を直接設定（正規化不要）

---

## 4. Conduction カーネル群

### 4.1 C1: compute_spitzer_deff

```cpp
__global__ void compute_spitzer_deff(
    double* __restrict__ D_eff,      // [n_cells] out
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ Zbar,
    const double* __restrict__ Cv_e,  // [n_cells] 質量比熱 c_v,e [erg/(g·eV)]（H13出力）
    const double* __restrict__ x_r,  // [n_nodes] ノードR座標（|∇T|近似のセル間距離計算に必要）
    const double* __restrict__ x_z,  // [n_nodes] ノードZ座標（2D_RZ用。1D_SPHではNULL可）
    const double* __restrict__ A_eff, // [n_cells] 有効原子量（ne = rho * Zbar / (A_eff[c] * m_p) をカーネル内で計算。
                                      //   単材料: A_mat[0]、多材料: §1.1.5(c) 調和平均。U8 出力を使用）
    double f_lim,                    // flux limiter係数
    int n_cells, int nr, int nz     // 格子次元（隣接セル特定に必要。1D_SPH: nz=1）
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: κ_SH → q_SH → q_max → q_limited → D_eff = |q|/(ρ c_v |∇T|)（NUMERICS §4.1-4.2。c_v は質量比熱 [erg/(g·eV)]）
- 勾配 |∇T|_c の計算（NUMERICS §4.3 準拠）:
  - **2D_RZ**: Kershaw B-演算子（Appendix A.4）を用いてノード (i,j) での勾配を構成。
    各ノードは周囲4セルの T_e からB-演算子で ∇T|_n を計算。
    セル中心では4コーナーノード勾配の算術平均: |∇T|_c = (1/4)Σ_{k=1}^{4} |∇T|_{n_k}
  - **1D_SPH**: 隣接セル温度差 / セル間距離（直接差分）
- **ε_grad ガード**（NUMERICS §4.3 必須）: |∇T|_c < ε_grad の場合 D_eff = D_SH（Spitzer無制限）。
  ε_grad = max(1e-10 × T_{e,c}/ℓ_c, 1e-30 [eV/cm])、ℓ_c = V_c^{1/3}。
  この相対的閾値により、ほぼ均一温度領域で D_eff の非物理的発散を防ぐ
- **レジスタ**: ~20
- **メモリ**: 隣接セルの Te 読み込み（2D: 4近傍 → 構造格子固定ストライド）、`__ldg()` でセルデータ参照
- **1D の実装**（`compute_spitzer_deff_1d_kernel`、`conduction_bodies.cuh`）：出力はセルの Spitzer–Härm 伝導率 \(\kappa_{SH}\)
  （と伝導率の床・上限）で、\(D_\mathrm{eff}\) は陽的な安定刻み \(\Delta t_\mathrm{exp}\) の見積もりにだけ使う。流束制限は
  `compute_1d_flux_limiter_faces_kernel` が面ごとに与え、面の係数は `compute_1d_face_kappa_kernel` がステップの始めの
  \(T_e^n\) で一度だけ評価する（既定 `face_kappa_policy="kirchhoff_same_material"`、NUMERICS §4.2.1）

### 4.2 C2: kershaw_stencil_build（2D RZ専用、最重要）

```cpp
__global__ __launch_bounds__(256, 2) void kershaw_stencil_build(
    double* __restrict__ stencil,    // [N × 9] out: 9点係数。N=n_cells（伝導）or n_cells+n_ghost（輻射R3用）
    const double* __restrict__ D_eff, // [N]（伝導: §4.3 の D_eff、輻射: 1/(3σ_{R,g})）
    const double* __restrict__ x_r,  // [n_nodes]（ゴーストノード座標含む）
    const double* __restrict__ x_z,  // [n_nodes]（同上）
    const uint8_t* __restrict__ face_bc_type, // [4] 物理境界タイプ: {r_lo, r_hi, z_lo, z_hi}。0=内部/MPI, 1=reflect, 2=vacuum
    bool apply_mmatrix_repair,       // True=伝導用（修復後出力）、False=輻射R3用（修復前出力）
    int* __restrict__ mmatrix_fix_count, // [1] out: M-matrix修復適用セル数（atomicAdd、apply_mmatrix_repair=True時のみ使用。False時はnullptr可）
    int nr, int nz,
    int n_ghost                      // 0=伝導用（owned cells のみ）、>0=輻射R3用（ghost cells 含む）
);
```

- **block**: 256, **grid**: `((n_cells+n_ghost)+255)/256`（伝導用: n_ghost=0、輻射R3用: n_ghost>0）
- **メモリレイアウト**: `stencil[c*9 + k]` — セルmajor。k=0:C, 1:E, 2:W, 3:N, 4:S, 5:NE, 6:NW, 7:SE, 8:SW
- **処理**（Appendix A準拠）:
  1. セル(i,j)の4コーナーノード座標を読む
  2. 4ノードの辺中点・セル中心座標を計算（A.2）
  3. 4ノードの A, B ベクトル・ヤコビアン J を計算（A.3）
  4. σ, λ（面の調和平均拡散係数）を計算（A.5）
  5. **RZ幾何因子の適用**（A.7、係数計算の**前**に実行）:
     σ_{i,j+1/2} → σ_{i,j+1/2} × r_{i,j+1/2}（i-面中点のR座標で重み付け）
     λ_{i+1/2,j} → λ_{i+1/2,j} × r_{i+1/2,j}（j-面中点のR座標で重み付け）
     **注**: r=0 軸上ではσ項がゼロ化され、軸対称反射BCが自然に実現（A.7）。
     この置換をA.6の係数計算の**前**に行うことが必須（後処理ではなく入力の重み付け）
  6. a_E, a_W, a_N, a_S（直接隣接）をRZ重み付きσ,λから計算（A.6）
  7. ρ^{1-4}（交差項）→ a_NE, a_NW, a_SE, a_SW を計算（A.6）
  8. a_C = -(a_E+a_W+a_N+a_S+a_NE+a_NW+a_SE+a_SW)
  9. **物理境界係数折り込み**（NUMERICS Appendix A.8 準拠）:
     境界セル（i=0, i=nr-1, j=0, j=nz-1）で、ドメイン外を向く隣接係数を処理:
     - **Reflect/Neumann**（face_bc_type=1）: φ_ghost = φ_interior に相当 → a_C += a_boundary; a_boundary = 0。
       コーナー係数（a_NE等）もドメイン外ならば同様に折り込む。行和ゼロ維持
     - **Vacuum/Robin**（face_bc_type=2、設計時は退役した DDMC の R3 用）: φ_ghost = -φ_interior × (d_ext-Δx/2)/(d_ext+Δx/2)
       に相当 → a_C += a_boundary × (-(d_ext-Δx/2)/(d_ext+Δx/2)); a_boundary = 0。d_ext = 0.7104/σ_tr
     - **内部/MPI**（face_bc_type=0）: 折り込みなし（MPI ghost cell を C3 が読む）
     - r=0軸は RZ重み付け（step 5）で σ=0 となり、反射BCが自然に実現済み（追加処理不要）
- **レジスタ**: ~48（A,B ベクトル 4組、σ,λ 4面、ρ^{1-4} 4点、係数9個、BC一時変数3個）
  - `__launch_bounds__(256, 2)` で最低25% occupancy（512 threads/SM）を保証。
    実行時は ~48 reg × 256 = 12288 reg/block → 最大4 blocks/SM → ~50% occupancy が期待されるが、
    コンパイラの最適化余地を確保するため `min_blocks=2` を指定
  - 計算量が支配的（compute-bound）であり、occupancy の低下は許容
- **M-matrix修復**（`apply_mmatrix_repair=True` 時のみ実行、NUMERICS Appendix A.6 準拠）:
  1. 各セルで a_{off-diag} ≤ 0 を検証
  2. **正のオフ対角クランプ**: a_corner > 0 の場合、a_corner ← 0 に設定（対角隣接のみ: a_NE, a_NW, a_SE, a_SW）
  3. **対角再計算**: a_C = -(a_E+a_W+a_N+a_S+a_NE+a_NW+a_SE+a_SW) を再計算し行和ゼロを維持
  4. 修復適用セル数を `atomicAdd(&fix_count, 1)` で集計 → diagnostics/kershaw_mmatrix_fix_count に記録
  5. fix_count > 0.1 × n_cells の場合、ホスト側で WARNING 出力（メッシュ品質劣化の兆候）
  - **2つの呼び出しモード**:
    - `apply_mmatrix_repair=True`（伝導用、Phase 2）: 修復後の係数を出力し安定な explicit 更新を行う
    - `apply_mmatrix_repair=False`（設計時は退役した DDMC の R3 用、Phase 4）: 修復前の raw 係数を出力。R3 `ddmc_leak_coeff` が M-matrix 違反を検出し、違反セル×群を IMC にフォールバックしていた（NUMERICS §7.3.3。DDMC とともに退役）

### 4.3 C3: kershaw_apply

```cpp
__global__ __launch_bounds__(256, 4) void kershaw_apply(
    double* __restrict__ Te_new,     // [n_cells] out
    const double* __restrict__ Te,   // [n_cells] in
    const double* __restrict__ stencil, // [n_cells × 9]
    const double* __restrict__ rho,
    const double* __restrict__ Cv_e,  // [n_cells] 質量比熱 c_v,e [erg/(g·eV)]（H13出力）
    const double* __restrict__ vol,
    double dt_sub,                   // サブステップ幅
    double T_floor,
    int nr, int nz,
    int* __restrict__ clamp_count    // 温度フロア適用回数（atomic）
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 1スレッド=1セル。9近傍の Te を読み、ステンシル積 → ΔTe → Te_new（NUMERICS Appendix A.9）
- **Super-Time-Stepping (STS)**: ホスト側で `s` ステージ分ループ。各ステージ後にダブルバッファ swap
  - `s = min(max(1, ceil(sqrt(2 * dt / dt_exp))), config.sts_max_stages)` — NUMERICS §4.2.1 準拠。`sts_max_stages`（既定40）でクランプ
  - 典型 s = 1–5（爆縮初期）、16–27（コロナ発達時）。素朴法の N_sub=130–340 を大幅に削減
  - 各ステージのサブステップ幅 τ_j は Chebyshev 根分布（不均一）:
    `τ_j = dt_exp / (ν² + (1-ν²) cos²(π(2j-1)/(4s+2)))`, ν=0.01（NUMERICS §4.2.1）
  - **タイムステップスケーリング**（NUMERICS §4.2.1 必須）: `dt_sts = Σ τ_j; τ_j ← τ_j × Δt/dt_sts`。
    Chebyshev 根分布はΣτ_j ≠ Δt となるため、全τ_jを一様スケーリングして Στ_j = Δt を保証。
    この正規化を省略すると伝導の実効積分時間が hydro ステップと不一致になり、温度更新が系統的にズレる
  - D_eff と Kershaw 係数はスーパーステップ開始時に凍結（C1→C2 は1回のみ、C3 を s 回実行）
  - ダブルバッファ swap はホスト側のポインタ交換のみ（カーネル内操作なし）
- **負温度防止**: `Te_new = max(Te_new, T_floor)` + clamp_count 加算
- **局所安全策**（NUMERICS §4.2.2、実装は `conduction_alpha_pass*_kernel`）: セルの床より上のエネルギー
  \(\rho c_v (T - T_\mathrm{floor})\) を、正味の流出（`sts_floor_limiter="net"`、既定）または流出の和（`"donor"`）で割った
  時間を \(\tau_j\) と比べて \(\alpha_c = \min(1, \cdot/\tau_j)\) を作り、面の流束を両側で同じ係数（net は
  \(\min(\alpha_c,\alpha_{nb})\)、donor は流出側のセルの \(\alpha\)）でスケーリングする。対の反対称が保たれるので
  スケーリング自体はエネルギーを作らない。旧設計の `Δt_safe = Δl²/(2 D_eff)` と、非対称分を `E_safety` に計上する方式は
  無い。エネルギーの計上が生じるのは最後の床クランプ（`E_floor_injected`）だけ
- **レジスタ**: ~15
- **同期**: STS の全ステージは同一 compute ストリーム上で起動されるため、CUDA ストリーム順序保証によりカーネル間は自動的に逐次実行される。ダブルバッファ swap（ホスト側ポインタ交換）に明示的 `cudaStreamSynchronize` は不要。ただし MPI halo exchange が介在する場合は、P1(pack) カーネル完了後に MPI 送信が必要なため、halo exchange ルーチン内部で `cudaStreamSynchronize(compute)` を実行する

### 4.4 C4: conduction_1d_tridiag（1D_SPH用）

> **実装**：このカーネルは存在しない。1D の電子伝導は次の 2 経路（`conduction.cu`）：
> - **STS**（既定）：`conduction_1d_sts_stage_kernel`（調和平均の面）、`_kirchhoff_kernel`・`_secant_kernel`（割線の面）、
>   `conduction_1d_sts_fused_kirchhoff_kernel`。面の係数はステップの始めの値で固定し、ステージは `kappa_face` を読む。
>   面積は形状ごと（球 \(4\pi r^2\)、円筒・平板は `geometry_1d_face_area`）
> - **陰解法**（`Numerics.conduction.solver="implicit"`）：`build_1d_implicit_system{,_kirchhoff}_kernel` で三重対角を組み、
>   cuSPARSE `gtsv2` で温度 \(T^*\) を解く。エネルギーは `implicit_conduction_flux_form_kernel` が面の流束
>   \(G_f = -u_f\,(T^*_{f+1}-T^*_f)\)（\(u_f\) は組んだ行列の上対角）の形で記帳する
>   （\(\Delta T_i = \Delta t\,(G_i-G_{i-1})/(\rho c_v V)_i\)、NUMERICS §4.2.3）
>
> イオン伝導は `ion_conduction_*_kernel`（`ion_conduction_pcr_solve_kernel` の並列巡回縮約）、非局所伝導（SNB）は
> `conduction_snb_1d.cu` の `snb_*_kernel`。以下は設計時の記述。

```cpp
__global__ void conduction_1d_tridiag(
    double* __restrict__ Te_new,
    const double* __restrict__ Te,
    const double* __restrict__ kappa_eff, // [n_nodes] 面の伝導率
    const double* __restrict__ r_node,    // [n_nodes]
    const double* __restrict__ rho,
    const double* __restrict__ Cv_e,  // [n_cells] 質量比熱 c_v,e [erg/(g·eV)]（H13出力）
    const double* __restrict__ vol,
    double dt_sub,
    double T_floor,
    int n_cells,
    int* __restrict__ clamp_count
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: NUMERICS §3.1.7 の離散化。面積 A_j = 4πr_j² を含む
- **レジスタ**: ~15
- **メモリ**: 隣接ノード座標 + Te 読み込み（1Dなので左右2近傍のみ、coalesced）
- **Super-Time-Stepping (STS)**: C3 と同様に、ホスト側で `s` ステージ分ループ。各ステージ後にダブルバッファ swap（ホスト側のポインタ交換のみ、カーネル内操作なし）。STS パラメータ（ステージ数 s、サブステップ幅 τ_j、Στ_j=Δt 正規化含む）は C3 と同一式（NUMERICS §4.2.1 準拠）。CUDA 同一ストリーム上のカーネル起動は順序保証されるため、ポインタ交換に明示的 `cudaStreamSynchronize` は不要

### 4.5 Hypre 陰的ソルバ統合（オプション、`-DTENRYU_ENABLE_HYPRE=ON`）

`conduction.solver="hypre"` 時の処理フロー（NUMERICS §4.2.3 参照）。
**2D_RZ 専用**: C2（Kershaw 9点ステンシル構築）が前提のため、1D_SPH では使用不可。
`geometry="1D_SPH"` かつ `conduction.solver="hypre"` の場合、STS（C4）に自動フォールバックし WARNING を出力する：

1. **C1 + C2 カーネル**：STS パスと共通（`D_eff` 計算 → Kershaw 9点ステンシル構築）
2. **行列変換**（ホスト側コード、デバイスメモリ上で動作）：
   - C2 出力のステンシル係数 `stencil[n_cells × 9]` を `HYPRE_IJMatrixSetValues` で ParCSR 行列に転写
   - 対角に \(C_v / \Delta t\) を加算（\(C_v = \rho c_v\) [erg/(cm³·eV)]、質量行列項。NUMERICS §4.2.3 準拠）
   - 初回のみスパーシティパターン構築（`HYPRE_IJMatrixSetRowSizes`、9 entries/row 固定）
   - 2回目以降は `HYPRE_IJMatrixSetValues` で値のみ更新
3. **Te^n 退避**：Hypre solve が State.Te を上書きするため、solve 前に `Te_old[n_cells]`（Scratch バッファ）へコピーする。
   Step 6 の E_solver 計算で \(T_{e,c}^{n+1} - T_{e,c}^n\) を算出するために必須。
   STS パスではダブルバッファで Te^n が自然に保持されるため本ステップは不要。
4. **Hypre solve**：`HYPRE_ParCSRPCGSolve`（BoomerAMG前処理、デバイスメモリ上で完結）
5. **解の書き戻し**：`HYPRE_IJVectorGetValues` → State.Te 配列 → EOS forward (H13: Te→ee, Pe, Cv_e) で再同期（NUMERICS §4.2.1 準拠。EOS inverse ではなく forward を使用する — Hypre は Te を直接解くため、Te→ee の順方向変換が必要）
6. **E_solver 計算**（NUMERICS §4.2.3）：
   \(E_{solver} = \sum_c C_{v,c}(T_{e,c}^{n+1}-T_{e,c}^n) V_c - \Delta t \sum_c (\nabla\cdot q)_c V_c\)
   ここで \(C_{v,c} = \rho_c \cdot c_{v,e,c}\) [erg/(cm³·eV)]、\(T_{e,c}^n\) は Step 3 の `Te_old` から取得。
   CUB `DeviceReduce::Sum` で集約し、Phase 6 D2H で `State.E_solver += step_E_solver` に累積する。
   STS パスでは本ステップは省略（E_solver=0）

> **カーネル起動の観点**：Hypre パスでは C3 カーネル（`kershaw_apply`）は **起動しない**。
> 代わりに Hypre 内部が AMG V-cycle と PCG 反復を実行する。
> C1, C2 は両パスで共通であり、ステンシル構築コードの重複はない。
>
> **Hypre のメモリ管理**：Hypre 2.25+ は `HYPRE_SetMemoryPoolAllocator` で
> TENRYU の Scratch バッファ（§0.4）を共有できる。
> ただし v1.0 では Hypre 独自のメモリプールを使用し、Scratch 共有は将来最適化とする。

---

## 5. Laser カーネル群

### 5.1 レーザー格子への写像（現行）

> **設計時の L1 `laser_mesh_map`・L2 `compute_density_gradient`・L5 `deposit_lm_to_hydro` は実装に存在しない。**
> 現行の写像と沈着の経路は次のとおり（ARCHITECTURE §4.6、NUMERICS §5.7–§5.8）。

**1D**（`laser_mesh.cu` の `map_from_hydro_1d` → `map_trace_profile_1d`、写像のたびに節点を作り直す）：

```cpp
__global__ void map_hydro_to_laser_1d_kernel(
    const double* __restrict__ node_R,       // 節点の R（2 回目の起動では径方向プロファイル節点の r）
    const double* __restrict__ node_Z,       // 節点の Z（2 回目は 0 を指す 1 要素）
    const double* __restrict__ rho, const double* __restrict__ Te, const double* __restrict__ zbar,
    const double* __restrict__ A_eff,        // [n_cells] device scratch
    const uint8_t* __restrict__ cell_is_void,// [n_cells] device scratch
    const double* __restrict__ r_edges,      // [n_cells+1] 流体の節点の半径
    double* __restrict__ n_hat, double* __restrict__ n_hat_raw,
    double* __restrict__ Te_LM, double* __restrict__ Zbar_LM,
    int n_nodes_total, int n_nodes_z, int n_cells,
    /* n_crit, ghost corona, critical reconstruction, clip scalars */);
```

- **block**: 256。1 スレッド = 1 節点。\(r=\sqrt{R^2+Z^2}\) で流体セルを device 上の二分探索で特定し、密度の正規化、
  臨界に隣接するセル対の対数線形プロファイル（NUMERICS §5.7.3(a)）、ゴーストコロナ、critical clip を評価する
- **2 回起動する**：(1) 2D レーザー格子の節点（`n_nodes_r × n_nodes_z`）、(2) `place_trace_profile_nodes_1d`
  （判定・区分ごとの節点数・最後の区分の尾部の節点数・CUB の前置和・区分の書き込み・尾部の書き込み・仕上げ。区分の
  数え上げと書き込みと尾部の書き込みは多数のブロック、判定と尾部の数え上げは 1 ブロック、順の配置に戻るときは 1 スレッド）
  が置いた径方向プロファイル節点（面・臨界の対・プロファイルの節点）。節点数は host へ
  1 回だけ返す。(2) の後に `map_node_material_1d_kernel`（同じ材料の \(n_e\) 補間用の節点の材料）、
  `compute_radial_gradient_kernel`（\(d\hat n/dr\)）、`compute_smooth_kappa{,_ext}_kernel`（逆制動輻射の smooth 係数）
- **step 間平滑化なし**: 1D の clipped \(\hat n\) は写像した値そのもの（`ema_smooth_n_hat_radial_kernel` は 2026-09-24 に撤去、NUMERICS §5.7.4）

**2D_RZ**（`map_from_hydro_2d`）：流体セルからレーザー格子の節点への補間は host で行い（`map_from_hydro_2d_impl`）、
device へ写したあと `compute_gradient_kernel`（中心差分の \(\partial\hat n/\partial R\)、\(\partial\hat n/\partial Z\)、軸で
\(\partial\hat n/\partial R=0\)、端は片側差分）と smooth 係数のカーネルを起動する。

**沈着の転写**：
- 1D の光線追跡は吸収したパワーを流体セルへ直接積む（`deposit_power_cell`、レーザー格子の節点を経由しない）。
  臨界付近の再配分（`apply_deposit_redistribution_1d`、`deposit_transfer.cu`）は host で行い、\(\Delta t\) を掛けて
  `laser_dep` [erg] にする。`transfer_to_1d` は試験だけが呼ぶ
- 2D_RZ は `transfer_2d_kernel`（1 スレッド = 1 流体セル、セル中心でのレーザー格子の双線形補間 × \(\Delta t\)）

### 5.1d L6: ray_skip_check

```cpp
__global__ void ray_skip_check_kernel(      // src/laser/raytrace_skip.cu
    double* __restrict__ delta_max_cell,    // [n_cells] out: セルごとの変化量（max_relative は 3 指標の最大、l2_relative は 3 指標の二乗和）
    int* __restrict__ crit_hit,             // [1] out: 臨界帯を横断したセルがあれば 1
    const double* __restrict__ rho,         // [n_cells] in: 現ステップ ρ（HydroMesh）
    const double* __restrict__ Te,          // [n_cells] in: 現ステップ Te（HydroMesh）
    const double* __restrict__ Zbar,        // [n_cells] in: 現ステップ Z̄（HydroMesh）
    const double* __restrict__ volFrac,     // [n_cells × n_mat] in: 体積分率（多材料の A_eff 用）
    const double* __restrict__ A_mat,       // [n_mat] in: 材料の原子量
    const int n_mat,
    const double* __restrict__ rho_cached,  // [n_cells] in: 前回レイトレース時 ρ
    const double* __restrict__ Te_cached,   // [n_cells] in: 前回レイトレース時 Te
    const double* __restrict__ Zbar_cached, // [n_cells] in: 前回レイトレース時 Z̄
    const double rho_floor, const double Te_floor, const double Zbar_floor,  // 分母のフロア（Zbar_floor = 1e-2）
    const double n_crit,                    // 臨界密度 n_crit(λ) [1/cm³]
    const double n_hat_margin,              // Laser.lasermesh.critical_margin
    const double crit_guard,                // 臨界帯の幅（既定 0.01）
    const double A_eff_uniform,             // A_mat が無いときの原子量
    const int use_l2_relative,              // 1 = l2_relative、0 = max_relative
    const int n_cells);
// HydroMesh の全セルを処理する（キャッシュ配列も HydroMesh のセル数）。
// max_relative: δ = max(|Δρ|/max(ρ_cached, ρ_floor), |ΔTe|/max(Te_cached, Te_floor), |ΔZ̄|/max(Z̄_cached, Z̄_floor))
// l2_relative:  分母は現在値 max(|x|, x_floor) で、3 指標の二乗和を書く
// 臨界帯: A_eff は単一材料なら A_mat[0]、多材料は体積分率で重み付けした 1/A の和の逆数。
//   n_e = ρ Z̄ / (A_eff m_p) を現在値とキャッシュ値の両方で求め、n_e/n_crit が帯域 n_hat_margin − crit_guard を
//   横断した（一方だけが帯域を超える）セルがあれば crit_hit = 1（NUMERICS §5.9.4）
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: HydroMesh の各セルで 3 指標 δ_ρ, δ_Te, δ_Z̄（NUMERICS §5.9.2）と臨界帯の横断（§5.9.4）を 1 回の起動で求める
- **後段**: `crit_hit` が 1 なら再計算する。そうでなければ δ をホストへ転送し、ホストで max_relative は最大値、l2_relative は \(\sqrt{\sum/(3N)}\) を求め、`Laser.raytrace_skip.threshold`（既定 0.01）未満ならレイトレースを省略する
- **レジスタ**: ~8

### 5.1e レイトレース省略のキャッシュ更新

> **実装**：設計時の `laser_cache_update` カーネルは無い。キャッシュの更新は host の `RaytraceSkipCache::update_cache`
> （`raytrace_skip.cu`）が行う：\(\rho, T_e, \bar Z\) を device 間コピーで保存し、群ごとの沈着の比 \(\hat f_g\)（host で
> セルの吸収パワー ÷ 群の入射パワー \(P_g\) として作る）を device へ写す（NUMERICS §5.9.3）。

### 5.1f helper: reconstruct_laser_dep

```cpp
__global__ void reconstruct_laser_dep_kernel(     // src/laser/raytrace_skip.cu
    double* __restrict__ laser_dep,               // [n_cells] out: 再構成沈着エネルギー [erg]
    const double* __restrict__ f_hat,             // [n_groups * n_cells] in: f̂_g_cached [無次元]
    const double* __restrict__ P_g_dt,            // [n_groups] in: グループ g の P_g(t_now)×Δt [erg]
    int n_groups, int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: per-cell 変換カーネル。`laser_dep[c] = Σ_g f̂_g[g * n_cells + c] × P_g_dt[g]` を計算し、skip path 用の沈着配列を再構成する（NUMERICS §5.9.3）
- **初期化**: カーネルが各セルの和を書き込む（加算ではない）ので、呼び出し前のゼロ初期化は不要
- **n_groups=1**: `laser_dep[c] = f̂_cached[c] × P_total_dt` に退化（従来動作と同一）

### 5.2 L3/L4: ray_trace（最重要レーザーカーネル）

> **現行の 1D（2026-09-29）**：1D の既定の積分法は特性線法（`Laser.raytrace.integrator="auto"` → `"characteristic"`、
> `ray_trace_1d_characteristic`、NUMERICS §5.3.6）。球対称場では角運動量 \(B=|x\times v|\) が保存されるので、全光線で共有する
> 区切り（径方向節点・流体の面・臨界に隣接する分割半径）と光線ごとの事象半径の間の区間ごとに、弧長・極角の増分・光学的厚さを
> 適応 Gauss–Kronrod (3,7) 求積で積分する（転回点の近くでは \(u=\sqrt{r-r_0}\) に変数変換）。`kTile` 本のレーン（1、32、
> 64〜256、光線数と GPU からランチャが選ぶ）で 1 本の光線の区間を並列に評価し、前置和・前置積で各区間の入口の極角とパワーを得る。
> 臨界半径に達した光線は既定（`critical_handling.terminate` を指定しないとき）で反射して外向きに追跡する。1D の円筒・平板は
> この積分法だけが扱う。`integrator="leapfrog"` の `ray_trace_1d_sph` は刻み幅可変の Verlet（leapfrog）で球だけを扱い、既定で臨界で
> 終了する。どちらの 1D カーネルも吸収したパワーを流体セルへ直接積む。下の擬似コードのレーザー格子の 4 節点への沈着と固定刻みは、
> 2D_RZ の `ray_trace_3d`（と検証用の `ray_trace_2d`）のもの。

```cpp
// 2D version (1D_SPH用)
__global__ __launch_bounds__(64, 16)
void ray_trace_2d(
    double* __restrict__ deposit,        // [N_LM_nodes] LaserMesh沈着 [erg/s]（atomicAdd）
    const double* __restrict__ nhat,     // [N_LM_nodes] n_e/n_crit（クリップ済み）
    const double* __restrict__ nhat_raw, // [N_LM_nodes] n_e/n_crit 生値（>1 許容 — 臨界層ハンドオフ判定用、dual-field）
    const double* __restrict__ grad_nhat_r, // [N_LM_nodes]
    const double* __restrict__ grad_nhat_z, // [N_LM_nodes]
    const double* __restrict__ Te_LM,    // [N_LM_nodes]
    const double* __restrict__ Zbar_LM,  // [N_LM_nodes]
    const double* __restrict__ r_LM,     // [nr_LM+1]
    const double* __restrict__ z_LM,     // [nz_LM+1]
    const double* __restrict__ ray_R0,   // [n_rays] 初期R位置
    const double* __restrict__ ray_power, // [n_rays] パワー重み
    double ds_ray_max,                   // 空間ステップサイズ Δs_ray [cm]
    double eps_n, double eps_crit,       // 臨界パラメータ
    double lambda_L,                     // レーザー波長 [cm]
    double intensity_cutoff,             // 最小強度カットオフ [無次元]（NUMERICS §5.2、既定 1e-6）
    int nr_LM, int nz_LM, int n_rays,
    double* __restrict__ P_unabsorbed,   // [1] 未吸収パワー [erg/s]（atomicAdd）
    DeviceErrorFlags* error_flags        // MAX_RAY_STEPS到達時に infinite_loop フラグを設定（§0.6 準拠）
);
```

- **block**: 64, **grid**: `(n_rays+63)/64`
  - 1D_SPH の `ray_trace_1d_sph` は `__launch_bounds__(64)` のまま 32 スレッド（1 warp）/block で起動する（2026-09-24）。レイは前回のトレースの step 数の降順に並ぶので長いレイが先頭の warp に集まり、1 warp/block にするとそれらが別々の SM に載って、FP64 演算器の少ない GPU で同じ SM の演算器を奪い合わない（GXII 300 step の kernel 時間 36.2 → 34.8 ms/step、RTX 4090）。レイ別集約も同じ block で起動する。
- **n_rays の決定**（NUMERICS §5.6.3、SPECIFICATION §6.4.6 `rays_per_beam` 参照）：
  - **1D_SPH**（L3 `ray_trace_2d`）：`n_rays = Σ_beams rays_per_beam`。各ビームの `rays_per_beam` 本のレイを R 方向に等間隔配置（NUMERICS §5.6.3(a)）
  - **2D_RZ**（L4 `ray_trace_3d`）：`n_rays = Σ_beams N_eff(beam)`。各ビームの `rays_per_beam = N` は断面2D格子の1辺あたりの本数。円形アパーチャにより実効本数 `N_eff ≈ π/4 × N²`（NUMERICS §5.6.3(b)）
  - 既定 `rays_per_beam`：1D_SPH=1000、2D_RZ=128（2D実効本数 ~12,800本/beam）
- **1スレッド=1レイ**、内部ループ:
  ```
  // Δs_ray = C_ray_max × Δx_LM_min（LaserMesh 最小セル幅 × CFL 係数）
  // C_ray = c × Δt_ray / Δx ≤ cfl_ray（デフォルト 0.8）
  // 関係: Δt_ray = Δs_ray / c = C_ray_max × Δx_LM_min / c
  // レイトレース呼び出しごとに1回計算（LaserMesh 最小セル幅から）
  // L3/L4 の leapfrog: vR -= (Δs_ray/2) × gR

  initialize: (R,Z,vR,vZ) from ray parameters
  ds = ds_ray_max  // 空間ステップサイズ Δs_ray [cm]（記号 κ は opacity 系に予約）
  half-step velocity correction (§5.3.2)
  int n_steps = 0;
  const int MAX_RAY_STEPS = 100000;  // 無限ループ防止ガード
  while (ray is active && n_steps++ < MAX_RAY_STEPS):
      1. Find LaserMesh cell for current (R,Z)
      2. Bilinear interpolate grad_nhat at (R,Z) → (gR, gZ)
      3. Velocity update: vR -= (ds/2) * gR,  vZ -= (ds/2) * gZ  // ds = Δs_ray [cm]
      4. Position update: R += ds * vR,  Z += ds * vZ              // ds = Δs_ray [cm]
      5. Check LaserMesh bounds → if outside: atomicAdd(P_unabsorbed, I_current); terminate
      6. Check critical density（**dual-field**：ステップ間で raw n̂ を carried_nh_raw として持ち回る）:
         - nh_old >= 1-eps_crit（クリップ値）→ atomicAdd(P_unabsorbed, I_current); terminate
         - nh_old_raw >= kCritLayerHandoffNhatRaw（生値）→ **臨界層 tail-closure ハンドオフ**（`try_tail_closure`）:
           亜臨界ノードのみから A_entry を再構成して近臨界帯へ沈着、残余は P_unabsorbed へ計上して terminate
           （NUMERICS §5.2 dual-field、SPECIFICATION §5.4 ハンドオフ規約）
      7. IB absorption (§5.4, **桁落ち回避形**で実装、NUMERICS §5.4.2 準拠):
         - Interpolate nhat, Te, Zbar at old and new positions
         - n_refr2 = max(eps_n, 1 - nhat)  // 臨界面近傍での 1/sqrt(1-nhat) 発散を防止（NUMERICS §5.4）
         - Compute κ_IB at both points (η = sqrt(n_refr2) を分母に使用)
         - Δs = sqrt((R_new-R_old)² + (Z_new-Z_old)²)  // 実変位（NUMERICS §5.4.2: Δs_ray=|r^{n+1}-r^n|、ds パラメータではない）
         - Optical depth S = (κ_IB_old + κ_IB_new)/2 * Δs
         - ΔP = -I_old * expm1(-S)              // 吸収パワー [erg/s]（expm1形式で桁落ち回避）
         - I_new = I_old - ΔP                    // 差分更新（I_old*exp(-S) は使用しない — テレスコーピング和保存性）
      8. Deposit ΔP to 4 neighboring LaserMesh nodes:
         - atomicAdd(deposit[node], w * ΔP)  ← 4回  // deposit は [erg/s]
      9. Intensity cutoff (NUMERICS §5.2):
         - if (I_new < intensity_cutoff * I_0) → terminate
         - atomicAdd(P_unabsorbed, I_new)        // 残存パワーを未吸収に計上
  ```
- **レジスタ**: ~40（位置2, 速度2, 強度1, LaserMeshセル座標2, 補間重み4, κ_IB 2, 一時10）
- **ワープ発散**: レイ長のばらつきが大きい（内側レイは長い光路、外側レイは短い）
  - **緩和策**: レイをR座標でソートし、同ワープ内のレイ長が近くなるよう配置
- **Atomic競合**: deposit配列（~32K nodes）への atomicAdd
  - 同一ノードへの同時書き込みは稀（レイは空間的に分散）
  - v1.0では追加の最適化は不要と判断
- **エラー処理**: MAX_RAY_STEPS 到達時は `atomicExch(&error_flags->infinite_loop, 1)` で記録し、未吸収パワーを `P_unabsorbed` に加算

#### Diagnostic per-ray step counts

The `ray_trace_*` launcher wrappers accept an optional `d_step_count` device
array for Phase 0 diagnostics (see the internal design note laser_kernel_rewrite_plan.md
v2 §2.1). The parameter is null-by-default; passing `nullptr` is the production
path. When present, the kernel writes each ray's final loop count at
termination. This is diagnostic-only and does not affect deposition, ray
integration, or any physics result.

Verbose profiling runs may emit one stdout line per profiled beam:

```text
[laser_per_ray_steps] step=N beam=B n_rays=K max=... p90=... p50=... mean=... mean_per_warp_max=... mean_per_warp_mean=...
```

`step`, `beam`, and `n_rays` identify the hydro step, beam index, and rays in
the beam; `max`, `p90`, `p50`, and `mean` summarize per-ray step counts; and
`mean_per_warp_max` / `mean_per_warp_mean` average the per-warp maximum and
mean step counts. This is verbose-only stdout instrumentation, not HDF5 schema,
and is not a downstream-consumer contract. The first verbose call can include
`LaserMesh::ensure_per_ray_step_scratch` scratch-growth allocation cost, so
treat it as a warm-up sample for profiling.

### 5.3 3Dレイトレース（L4、2D_RZ用）

L3と同様だが、位置・速度が3Dベクトル `(x,y,z)` に拡張される（NUMERICS §5.3.4）。
LaserMesh参照は `R = sqrt(x²+y²), Z = z` で2Dに投影。
3D勾配は2D勾配から NUMERICS §5.3.4 (b) の変換で計算。
臨界面横断判定は L3 と同一の dual-field ロジック（クリップ n̂ + 生値 n̂_raw、上記 step 6）を適用する。

```cpp
__global__ __launch_bounds__(64, 16)
void ray_trace_3d(
    double* __restrict__ deposit,        // [N_LM_nodes] LaserMesh沈着 [erg/s]（atomicAdd）
    const double* __restrict__ nhat,     // [N_LM_nodes] n_e/n_crit（クリップ済み）
    const double* __restrict__ nhat_raw, // [N_LM_nodes] n_e/n_crit 生値（>1 許容 — dual-field）
    const double* __restrict__ grad_nhat_r, // [N_LM_nodes]
    const double* __restrict__ grad_nhat_z, // [N_LM_nodes]
    const double* __restrict__ Te_LM,    // [N_LM_nodes]
    const double* __restrict__ Zbar_LM,  // [N_LM_nodes]
    const double* __restrict__ r_LM,     // [nr_LM+1]
    const double* __restrict__ z_LM,     // [nz_LM+1]
    const double* __restrict__ ray_x0,   // [n_rays] 初期3D位置 x
    const double* __restrict__ ray_y0,   // [n_rays] 初期3D位置 y
    const double* __restrict__ ray_z0,   // [n_rays] 初期3D位置 z
    const double* __restrict__ ray_vx0,  // [n_rays] 初期3D方向 vx
    const double* __restrict__ ray_vy0,  // [n_rays] 初期3D方向 vy
    const double* __restrict__ ray_vz0,  // [n_rays] 初期3D方向 vz
    const double* __restrict__ ray_power, // [n_rays] パワー重み
    double ds_ray_max,                   // 空間ステップサイズ [cm]
    double eps_n, double eps_crit,       // 臨界パラメータ
    double lambda_L,                     // レーザー波長 [cm]
    double intensity_cutoff,             // 最小強度カットオフ [無次元]（既定 1e-6）
    int nr_LM, int nz_LM, int n_rays,
    double* __restrict__ P_unabsorbed,   // [1] 未吸収パワー [erg/s]（atomicAdd）
    DeviceErrorFlags* error_flags        // MAX_RAY_STEPS到達時に infinite_loop フラグ設定
);
```

- **block**: 64, **grid**: `(n_rays+63)/64`

- `__launch_bounds__(64, 16)` — L3 と同一設定
- **追加レジスタ**: +6（3D位置・速度の追加次元）→ 計 ~46
- **R=0特異性**: `R < R_floor` で横方向勾配をゼロ化（NUMERICS §5.5 R=0 軸特異性処理）
- **error_flags**: L3 と同じく `DeviceErrorFlags*` を引数に取る（MAX_RAY_STEPS到達時に infinite_loop フラグ設定）

### 5.4 L7: radial_absorption_1d_kernel

```cpp
template <bool kHotECapture>
__global__ __launch_bounds__(256, 1)
void radial_absorption_1d_kernel(
    double P_total,                              // 入射総パワー Σ_b P_b(t) [erg/s]
    const double* __restrict__ hydro_r_edges,    // [n_hydro_cells+1] Hydro 1D cell edges [cm]
    const double* __restrict__ radial_node_r,    // [n_radial_nodes] radial lookup nodes [cm]
    const double* __restrict__ radial_n_hat,     // [n_radial_nodes] clipped n̂
    const double* __restrict__ radial_n_hat_raw, // [n_radial_nodes] raw n̂
    const double* __restrict__ radial_smooth_kappa, // [n_radial_nodes] smooth κ factor
    double eps_n, double eps_crit,               // IB/critical parameters
    double test_kappa_cm_inv,                    // >0 のとき検証用固定κ [cm^-1]
    double intensity_cutoff,                     // P < cutoff*P_total で終了
    int n_hydro_cells,
    int n_radial_nodes,
    double* __restrict__ deposit_power_cell,     // [n_hydro_cells] out: 吸収パワー [erg/s]
    double* __restrict__ P_unabsorbed,           // [1] out: 未吸収パワー [erg/s]
    unsigned long long* __restrict__ critical_surface_hit_count,
    DeviceErrorFlags* __restrict__ error_flags,
    const HotECaptureParams hot_e_params,        // 熱電子の捕捉（kHotECapture のとき）
    double* __restrict__ hot_e_capture
);
```

- **launch**: `<<<1, 256, 0, stream>>>`（`kRadialAbsorptionBlock = 256`）。1D の `Laser.mode="radial_absorption_1d"` で、
  全 rank が同じ入力で起動する（沈着は rank 間で同一。2026-09-29 以前は rank 0 だけが起動し、他の rank は沈着が 0 だった）
- **処理**（NUMERICS §5.4a）：外側の流体セルから内側へ、1 本の内向きの流束として積分する
  - 1024 セルずつ（`kRadialAbsorptionChunk`）、256 スレッドが各セルの入力（中点の `n_hat`・`n_hat_raw`、逆制動輻射の κ、
    光学的厚さ \(\tau\)、セルの状態）を並列に評価して共有メモリへ置き、スレッド 0 がセルの順にパワーを運ぶ
    （以前の 1 スレッドのループと同じ算術で、結果はビット一致）
  - `n_hat_raw >= 1 - eps_crit` で臨界到達、残りの \(P\) を `P_unabsorbed` へ
  - `ΔP = absorbed_power_expm1(P, τ, P_next)`、`deposit_power_cell[c] += ΔP`
  - 最内セルに達するか `P < intensity_cutoff * P_total` で残りを `P_unabsorbed` へ
- **出力**: `deposit_power_cell` は [erg/s] のまま 1D の沈着経路へ渡り、後段で \(\Delta t\) を掛けて `laser_dep` [erg] になる
- **error_flags**: 不正な入力・半径・κ・τ で `invalid_cell`、非有限のパワーで `nan_particle` を立て、残りのパワーを未吸収へ戻す

---
