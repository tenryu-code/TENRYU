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
     - **Vacuum/Robin**（face_bc_type=2、DDMC R3用）: φ_ghost = -φ_interior × (d_ext-Δx/2)/(d_ext+Δx/2)
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
    - `apply_mmatrix_repair=False`（輻射R3用、Phase 4）: 修復前の raw 係数を出力。R3 `ddmc_leak_coeff` が M-matrix 違反を検出し、違反セル×群を IMC にフォールバックする（NUMERICS §7.3.3）

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
    int* __restrict__ clamp_count,   // 温度フロア適用回数（atomic）
    double* __restrict__ E_safety    // [1] out (atomicAdd): §4.2.2 安全スケーリングによるエネルギー注入 [erg]
                                     // α<1 適用時の非対称性 ΔE = Σ_c ρc_v(T_new-T_old)V - Δt·Σ_c(∇·q)V
                                     // NUMERICS §10.2 E_{safety} に計上。Hypre使用時は0
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
- **局所安全策**: フラックスが過大な場合のスケーリング `α = min(1, Δt_safe/τ_j) — Δt_safe = Δl²/(2 D_eff) はセル固有の明示的安定タイムステップ（NUMERICS §4.2.2）` でフラックスを抑制。α < 1 適用時のエネルギー非対称性は `E_safety` として計上
- **レジスタ**: ~15
- **同期**: STS の全ステージは同一 compute ストリーム上で起動されるため、CUDA ストリーム順序保証によりカーネル間は自動的に逐次実行される。ダブルバッファ swap（ホスト側ポインタ交換）に明示的 `cudaStreamSynchronize` は不要。ただし MPI halo exchange が介在する場合は、P1(pack) カーネル完了後に MPI 送信が必要なため、halo exchange ルーチン内部で `cudaStreamSynchronize(compute)` を実行する

### 4.4 C4: conduction_1d_tridiag（1D_SPH用）

1D球対称では Kershaw が3点トリダイアゴナルに退化する。
cyclic reduction または Thomas 法で直接解法も可能だが、v1.0は STS（NUMERICS §4.2.1）で統一する。

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

### 5.1 L1: laser_mesh_map

```cpp
__global__ void laser_mesh_map(
    double* __restrict__ nhat_LM,    // [N_LM] out: n_e/n_crit（クリップ済み ≤1）
    double* __restrict__ nhat_raw_LM,// [N_LM] out: n_e/n_crit 生値（クリップなし、>1 許容 — dual-field、NUMERICS §5.2）
    double* __restrict__ Te_LM,      // [N_LM] out
    double* __restrict__ Zbar_LM,    // [N_LM] out
    const double* __restrict__ rho_hydro,  // [n_cells]
    const double* __restrict__ Te_hydro,
    const double* __restrict__ Zbar_hydro,
    const double* __restrict__ x_r_hydro,  // [n_nodes]
    const double* __restrict__ x_z_hydro,
    /* LaserMesh geometry */
    const double* __restrict__ r_LM,       // [nr_LM+1] LaserMeshノードR座標
    const double* __restrict__ z_LM,       // [nz_LM+1] LaserMeshノードZ座標
    int nr_LM, int nz_LM,
    int nr_hydro, int nz_hydro
);
```

1D_SPH の実装は専用カーネルで HydroMesh device field を直接読む：

```cpp
__global__ void map_hydro_to_laser_1d_kernel(
    const double* __restrict__ node_R,       // [n_nodes_r] LaserMesh R nodes
    const double* __restrict__ node_Z,       // [n_nodes_z] LaserMesh Z nodes
    const double* __restrict__ rho,          // [n_cells] HydroMesh device rho
    const double* __restrict__ Te,           // [n_cells] HydroMesh device Te
    const double* __restrict__ zbar,         // [n_cells] HydroMesh device Zbar
    const double* __restrict__ A_eff,        // [n_cells] device scratch
    const uint8_t* __restrict__ cell_is_void,// [n_cells] device scratch
    const double* __restrict__ r_edges,      // [n_cells+1] HydroMesh node radii
    double* __restrict__ n_hat,              // [N_LM] out: clipped n_e/n_crit
    double* __restrict__ n_hat_raw,          // [N_LM] out: unclipped n_e/n_crit
    double* __restrict__ Te_LM,              // [N_LM] out
    double* __restrict__ Zbar_LM,            // [N_LM] out
    int n_nodes_total, int n_nodes_z, int n_cells,
    /* n_crit, ghost corona, critical reconstruction, clip scalars */
);

__global__ void ema_smooth_n_hat_kernel(
    double* __restrict__ n_hat,              // [N_LM] in/out clipped n_e/n_crit
    const double* __restrict__ prev_n_hat,   // [N_LM] previous clipped n_e/n_crit
    int n_nodes_total
);
```

- **block**: 256, **grid**: `(N_LM_nodes+255)/256`
- **処理**: 各LaserMeshノードの(R,Z)座標に対応するHydroMeshセルを特定し、双線形補間（NUMERICS §5.2）
  - 1D_SPH: `idx = blockIdx.x * blockDim.x + threadIdx.x`、`i = idx / n_nodes_z`、`j = idx % n_nodes_z`。`r = sqrt(R² + Z²)` で1D球メッシュの対応セルを device-side 二分探索し、密度正規化、edge-anchored log reconstruction、ghost corona、critical clip を各ノードで評価する。出力は LaserMesh device arrays に直接書き込む（host側の節点フィールド生成とH2D転送は行わない）
  - 2D_RZ: LaserMeshとHydroMeshが同じRZ座標系 → 直接的な双線形補間
- **EMA**: 1D_SPH near-critical smoothing は `ema_smooth_n_hat_kernel` で実行する。条件は `cur > 0.3 && prev > 0.3 && |cur-prev| < 0.01`、係数は `α=0.05`。`prev_n_hat` は LaserMesh device buffer に保持し、mesh size change/release で無効化する
- **レジスタ**: ~15（補間重み4、HydroMeshセルインデックス、一時変数）
- **メモリ**: HydroMesh フィールドへの読み込みは scattered（各 LaserMesh ノードが異なる HydroMesh セルを参照）。`__ldg()` で L2 キャッシュ活用

### 5.1b L2: compute_density_gradient

```cpp
__global__ void compute_density_gradient(
    double* __restrict__ grad_nhat_r,       // [N_LM_nodes] out: ∂(n̂)/∂R
    double* __restrict__ grad_nhat_z,       // [N_LM_nodes] out: ∂(n̂)/∂Z
    const double* __restrict__ nhat,        // [N_LM_nodes] in: n_e/n_crit（L1出力）
    const double* __restrict__ r_LM,        // [nr_LM+1] LaserMeshノードR座標
    const double* __restrict__ z_LM,        // [nz_LM+1] LaserMeshノードZ座標
    int nr_LM, int nz_LM
);
// Central difference: ∂n/∂R = (n[i+1,j] - n[i-1,j]) / (2*ΔR_i)   // ΔR_i = 局所格子間隔（ストレッチ格子対応、NUMERICS §5.5(a)）
//                     ∂n/∂Z = (n[i,j+1] - n[i,j-1]) / (2*ΔZ_j)   // ΔZ_j = 局所格子間隔
// 境界:
//   R=0軸 (i=0): ∂n/∂R = 0 (対称性、ミラー拡張と等価)
//   i=nr_LM-1:   one-sided difference (backward)
//   Z境界: one-sided difference (forward/backward)
```

- **block**: 256, **grid**: `(N_LM_nodes+255)/256`
- **レジスタ**: ~10
- **メモリ**: coalesced read（nhat配列）、coalesced write（grad_nhat_r, grad_nhat_z）

### 5.1c L5: deposit_lm_to_hydro

```cpp
__global__ void deposit_lm_to_hydro(
    double* __restrict__ laser_dep,     // [n_cells] out: セル沈着エネルギー [erg]（NUMERICS §5.8.1）
    const double* __restrict__ deposit, // [N_LM_nodes] in: LaserMesh沈着パワー [erg/s]
    const double* __restrict__ r_LM,    // [nr_LM+1] LaserMesh R座標
    const double* __restrict__ z_LM,    // [nz_LM+1] LaserMesh Z座標
    const double* __restrict__ r_cell,  // [n_cells] HydroMeshセル中心 R座標
    const double* __restrict__ z_cell,  // [n_cells] HydroMeshセル中心 Z座標
    double dt,                          // 流体タイムステップ幅 [s]（パワー→エネルギー変換に使用）
    int nr_LM, int nz_LM, int n_cells
);
// block=256, grid=次元依存: 2D_RZ=(n_cells+255)/256, 1D_SPH=(N_LM_nodes+255)/256
// 2D_RZ: 各 HydroMesh セルの中心 (r_c, z_c) を LaserMesh 上で逆引き
// bilinear interpolation で LaserMesh ノード deposit から HydroMesh セル沈着を計算
// パワー→エネルギー変換: laser_dep[c] = (Σ w_ij deposit_ij) × dt  [erg]（NUMERICS §5.8.1）
// パワー保存: |Σ_c P_c - Σ_{ij} deposit_{ij}| / Σ_{ij} deposit_{ij} ≤ 1e-10（相対誤差、NUMERICS §5.8.2 準拠）
// **MPI注意**: Σ_c P_c は各rankのローカル合計。マルチGPU時は MPI_Allreduce(SUM) でグローバル合算後に
// LaserMesh deposit 総和（replicated→全rank同一値）と比較すること（Phase 3 Allreduce 参照）
```

- **block**: 256, **grid**: **次元依存**（下記参照）
- **処理**（**次元依存**、NUMERICS §5.8.1 準拠）:
  - **2D_RZ**: grid = `(n_cells+255)/256`（1スレッド=1 HydroMeshセル）。各HydroMeshセルの中心 `(r_c, z_c)` をLaserMeshのノード座標上に射影し、4隣接ノードの deposit 値から双線形補間でセル沈着パワー `P_c` [erg/s] を算出。その後 `laser_dep[c] = P_c × dt` [erg] としてエネルギーに変換（Δt乗算はここで1回のみ）
  - **1D_SPH**: grid = `(N_LM_nodes+255)/256`（1スレッド=1 LaserMeshノード）。各LaserMeshノード `(R_i, Z_j)` に対し半径 `r = √(R_i² + Z_j²)` を計算し、1D球座標メッシュのセル `k`（`r_{k-1/2} ≤ r < r_{k+1/2}`）に `deposit_{i,j}` を `atomicAdd` で集約。転写後 `laser_dep[k] = (Σ deposit) × dt` [erg]。双線形補間ではなくr-bin sum方式（NUMERICS §5.8.1 (a)）。**注**: 2D_RZ とはスレッディングモデルが異なる（セルベース→ノードベース）
- **レジスタ**: ~15（補間重み4、LaserMeshインデックス2、deposit一時変数）
- **パワー保存検証**: ×Δt変換 **前** のパワー値 `P_c` の総和とLaserMesh deposit 総和の差分を後段で CUB `DeviceReduce::Sum` により確認（単位: [erg/s]）。検証はパワー段で行い、エネルギー変換は最終ステップで実施。**MPI**: `Σ_c P_c` はローカル合計 → `MPI_Allreduce(SUM)` でグローバル合算後に LaserMesh 総和と比較（deposit は replicated で全 rank 同一値）

### 5.1d L6: ray_skip_check

```cpp
__global__ void ray_skip_check(
    double* __restrict__ delta_max_cell,    // [n_LM_cells] out: LMセル最大相対変化量
    const double* __restrict__ rho,         // [n_cells] in: 現ステップ ρ（HydroMesh 全体）
    const double* __restrict__ Te,          // [n_cells] in: 現ステップ Te（HydroMesh 全体）
    const double* __restrict__ Zbar,        // [n_cells] in: 現ステップ Z̄（HydroMesh 全体）
    const double* __restrict__ rho_cached,  // [n_LM_cells] in: 前回レイトレース時 ρ（LM対応セルのみ）
    const double* __restrict__ Te_cached,   // [n_LM_cells] in: 前回レイトレース時 Te（LM対応セルのみ）
    const double* __restrict__ Zbar_cached, // [n_LM_cells] in: 前回レイトレース時 Z̄（LM対応セルのみ）
    const int32_t* __restrict__ lm_to_hydro,// [n_LM_cells] in: LMセル→HydroMeshセルのマッピング
    double rho_floor, double Te_floor, double Zbar_floor,  // フロア値（分母保護用）
    double n_crit,                          // 臨界密度 n_crit(λ) [1/cm³]（crit_guard判定用）
    double crit_guard,                      // 臨界面近傍再計算ガード（既定 0.01）
    const double* __restrict__ A_eff,       // [n_cells] in: 有効原子量（多材料: §1.1.6 調和平均、H15出力と同一。n_e = ρ × Z̄ × N_A / A_eff[hydro_c]）
    int n_LM_cells
);
// **重要: L6 は HydroMesh 上の ρ, Te, Z̄ を読み取る（LaserMesh ではない）**。
// 理由: §9 Phase 3 の実行順序で L6 は L1（laser_mesh_map）の前に起動されるため、
// LaserMesh の値は前ステップのもの（stale）。HydroMesh は Phase 1/2 で最新に更新済み。
// L6 は n_LM_cells 個のセルのみ処理する。lm_to_hydro[lm_c] で HydroMesh セル index を取得し、
// rho[hydro_c] vs rho_cached[lm_c] を比較する。キャッシュ配列は n_LM_cells サイズ。
// δ_ρ = |ρ[hydro_c] - ρ_cached[lm_c]| / max(ρ_cached[lm_c], rho_floor)（+ Te, Z̄ 同様）
// crit_guard 判定: n_e = ρ × Z̄ × N_A / A_eff[hydro_c] を局所計算し、n_e/n_crit > 1 - crit_guard なら強制再計算
// 注: 分母は **cached** 値を使用する（current ではない）。NUMERICS §5.9.2 準拠
// 出力: δ_max = max(δ_ρ, δ_Te, δ_Z̄) per cell → CUB Max → host check < raytrace_skip → skip
```

- **block**: 256, **grid**: `(n_LM_cells+255)/256`
- **処理**: 各LaserMesh対応HydroMeshセルの3指標 δ_ρ, δ_Te, δ_Z̄ を計算（NUMERICS §5.9.2）。lm_to_hydro マッピングで HydroMesh 値を間接参照。分母は max(x_cached, x_floor) でフロア保護。さらに n_e/n_crit > 1 - crit_guard のセルがあれば強制再計算（§5.9.4、n_e = ρ × Z̄ × N_A / A_eff で局所計算）
- **後段**: CUB `DeviceReduce::Max` → `δ_max` をホストに転送。`δ_max < raytrace_skip` (default 0.01) ならレイトレースを省略
- **レジスタ**: ~8

### 5.1e helper: laser_cache_update

```cpp
__global__ void laser_cache_update(
    double* __restrict__ rho_cached,         // [n_LM_cells] out: L6用 ρ キャッシュ
    double* __restrict__ Te_cached,          // [n_LM_cells] out: L6用 Te キャッシュ
    double* __restrict__ Zbar_cached,        // [n_LM_cells] out: L6用 Z̄ キャッシュ
    double* __restrict__ laser_dep,          // [n_cells] inout: 全グループ累積沈着 [erg]（+= laser_dep_g）
    double* __restrict__ laser_dep_frac,     // [n_groups * n_cells] out: f̂_g_cached（skip再構成用、グループ毎）
    const double* __restrict__ laser_dep_g,  // [n_cells] in: 当該グループ g の沈着 [erg]
    const double* __restrict__ rho,          // [n_cells] in
    const double* __restrict__ Te,           // [n_cells] in
    const double* __restrict__ Zbar,         // [n_cells] in
    const int32_t* __restrict__ lm_to_hydro, // [n_LM_cells] in: LMセル→HydroMeshセル
    double P_g_dt,                           // グループ g のビームパワー合計 × Δt [erg]
    int group_idx,                           // グループインデックス g（f̂ 配列オフセット用）
    int n_LM_cells, int n_cells
);
```

- **block**: 256, **grid**: `(max(n_cells, n_LM_cells)+255)/256`（2パスの最大要素数でグリッドサイズを決定。n_LM_cells > n_cells の場合に pass 1 のキャッシュ更新漏れを防止）
- **処理**（2パス構成、各パスで独立にガード）:
  1. **LMセル部分** (`if (tid < n_LM_cells)`): `hydro_c = lm_to_hydro[tid]` → `rho/Te/Z̄` キャッシュ更新
  2. **全セル** (`if (tid < n_cells)`): `laser_dep_frac[group_idx * n_cells + tid] = laser_dep_g[tid] / P_g_dt`（`P_g_dt<=0` は 0）、`laser_dep[tid] += laser_dep_g[tid]`（累積）
- **ρ/Te/Z̄ キャッシュ**: 全グループ共通（最終グループで上書きされるが、値は同一ステップ内で不変のため問題なし）
- **per-group f̂ + 累積**: f̂ 計算と laser_dep 累積をフュージングし、グループループ内でカーネル起動を1回に抑える。NUMERICS §5.9.3「グループ毎にキャッシュ」準拠。n_groups=1（GXII等）は従来と同一動作

### 5.1f helper: reconstruct_laser_dep

```cpp
__global__ void reconstruct_laser_dep(
    double* __restrict__ laser_dep,               // [n_cells] out: 再構成沈着エネルギー [erg]（呼び出し前にゼロ初期化）
    const double* __restrict__ laser_dep_frac,    // [n_groups * n_cells] in: f̂_g_cached [無次元]
    const double* __restrict__ P_g_dt,            // [n_groups] in: グループ g の P_g(t_now)×Δt [erg]
    int n_groups, int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: per-cell 変換カーネル。`laser_dep[c] = Σ_g f̂_g[g * n_cells + c] × P_g_dt[g]` を計算し、skip path 用の沈着配列を再構成する（NUMERICS §5.9.3）
- **呼び出し前**: ホスト側で `cudaMemsetAsync(laser_dep, 0)` を実行すること
- **n_groups=1**: `laser_dep[c] = f̂_cached[c] × P_total_dt` に退化（従来動作と同一）

### 5.2 L3/L4: ray_trace（最重要レーザーカーネル）

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
array for Phase 0 diagnostics (see `docs/design/laser_kernel_rewrite_plan.md`
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
__global__ __launch_bounds__(1, 1)
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
    double* __restrict__ P_unabsorbed,           // [1] out: 未吸収パワー [erg/s]（atomicAdd）
    unsigned long long* __restrict__ critical_surface_hit_count,
    DeviceErrorFlags* __restrict__ error_flags
);
```

- **launch**: `<<<1, 1, 0, stream>>>`。1D_SPH `radial_absorption_1d` の rank0 のみが起動する
- **処理**: 単一スレッドで外側 Hydro cell から内側 cell へ serial 積分する（NUMERICS §5.4a）
  - `P_total = Σ_b P_b(t)` を1本の inward radial flux として扱う
  - セル幅 `dr = hydro_r_edges[c+1] - hydro_r_edges[c]`
  - radial lookup で `n_hat`, `n_hat_raw`, `radial_smooth_kappa` を線形補間
  - `n_hat_raw >= 1 - eps_crit` で臨界到達、残存 `P` を `P_unabsorbed` へ加算
  - `τ = compute_optical_depth(κ, κ, dr)`、`ΔP = absorbed_power_expm1(P, τ, P_next)`
  - `deposit_power_cell[c] += ΔP`、`P = P_next`
  - 最内セル到達後または `P < intensity_cutoff * P_total` で残存 `P` を `P_unabsorbed` へ加算
- **出力**: `deposit_power_cell` は [erg/s] のまま既存 1D deposit path へ渡され、後段で `Δt` を掛けて `laser_dep` [erg] へ変換する
- **並行性**: 単一スレッド serial 積分。`P_unabsorbed` は既存 tally と同じ `atomicAdd` を使う
- **理由**: 1D球対称1本積分の計算量は \(O(n_{\mathrm{cells}})\) で GPU 並列化の利益が小さい。
  既存 device-side インフラ（`LaserMesh` radial arrays, IB helper, `deposit_power_cell`）を再利用するため CUDA kernel として保持する
- **error_flags**: 不正な入力/半径/κ/τ で `invalid_cell`、非有限パワーで `nan_particle` を設定し、残存パワーを未吸収へ戻す

---

