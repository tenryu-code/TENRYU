<!-- 分割元: docs/CUDA_KERNELS.md | このファイルは参照用です。原本（docs/CUDA_KERNELS.md）が権威です。 -->
## 7. Coupling/Utility カーネル群

### 7.1 U1: source_injection

現行の注入はレーザーと燃焼の沈着だけ（`coupling/source_terms.cu`）：`prepare_laser_injection_kernel` がセルの
\(c_{v,e}\) の閉包を用意し、`inject_laser_source_cells_kernel` が `laser_dep[c]` を \(e_e\) へ加え、退化セル
（\(\rho V\) が小さすぎるセル）の分は `E_numerical_loss` に回す。`fold_laser_injection_ledger_kernel` が台帳を畳む。
解析的 \(T^4\) 閉包（`cv_e_override` と `eos_T_ref_eV`、tabular 2T でないデッキ）のセルは host の `pow(x, 1/4)`
（`core::glibc_libm`）で閉じる（`inject_laser_source_cells_kernel<true>`。2026-10-02 まで host で注入していた）。
燃焼は `inject_burn_source_cells_kernel`。輻射の沈着 `rad_dep` は注入しない — FLD・\(S_N\) は物質の更新を自分の
Newton の中で行う。本節にあった設計（`laser_dep` と `rad_dep` を合わせて注入する `source_injection` と、その
二重計上防止プロトコル）は、モンテカルロ輻射の沈着の注入とともに退役し、
`retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md` へ移した。

### 7.2 U3: energy_budget

```cpp
__global__ void energy_budget(
    double* __restrict__ partial_sums,    // [grid_size × 3] out（E_kin, E_int_e, E_int_i）
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ vr,
    const double* __restrict__ vz,
    const double* __restrict__ vol,
    const double* __restrict__ E_numerical_loss, // [1] 数値損失（退化セルへの未注入分など）
    int n_cells, int n_groups
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: セルベース3成分をブロック内 shared memory で部分和 → global（NUMERICS §10.2 準拠）:
- **同期**: ブロック内 reduction に `__syncthreads()` が必要（CUB `BlockReduce` 使用時は自動挿入）
  1. **E_kin**: 運動エネルギー = Σ ½ρu² V
  2. **E_int_e**: 電子内部エネルギー = Σ ρ e_e V
  3. **E_int_i**: イオン内部エネルギー = Σ ρ e_i V
  4. **E_rad**: 放射場エネルギー = \(\sum_c \sum_g \texttt{rad\_E}_{c,g}\,V_c\)（FLD・\(S_N\) の場。負の値は切り捨てず、その個数を
     数える。GPU で block 部分和を作り host で決定論的に足す — `coupling/driver_fld_energy.cu`）。境界からの流出 `E_rad_esc` は各ソルバーの境界の台帳から得る
     （退役したモンテカルロ輻射では census 粒子のエネルギー和と群別の `E_escape[g]` を使っていた）
  - E_laser_absorbed は別途 LaserMesh から取得（Laser 演算子内で計算済み）
  - **保存則チェック**（NUMERICS §10.2 準拠）:
    ΔE_total = ΔE_kin + ΔE_int_e + ΔE_int_i + ΔE_rad
    = E_laser_in - E_laser_esc + E_Marshak_in - E_rad_esc - E_pdV_bdry - E_numerical_loss + E_floor + E_safety + E_solver
    ε_budget = |ΔE_total - (E_source - E_sink)| / E_denom（NUMERICS §10.2 Eq.）
    E_source = E_laser_in + E_Marshak_in, E_sink = E_laser_esc + E_rad_esc + E_numerical_loss + E_pdV_bdry
    **項の生成元**:
    - E_laser_in: ホスト計算 Σ_b P_b(t) × Δt（Phase 3 で算出済み）
    - E_Marshak_in: ホスト解析計算 Σ_f (a_eV c/4) T_{r,f}⁴ A_f dt（§10.2、CUB Sum 不要。§9 Phase 4 R13 後参照）
    - E_pdV_bdry: **ホスト側計算**。Phase 1/5 Corrector 完了後に境界面の PdV 仕事を集計:
      `E_pdV_bdry += Σ_{f∈∂Ω} P_f × A_f × v_{n,f} × Δt_half` （NUMERICS §10.2）。
      H11/H12 は per-cell PdV を計算するが境界寄与を分離しないため、
      境界面の圧力・面積・法線速度から直接算出する（NUMERICS §3.2.14 境界面定義参照）。
      ステップ合計 = Phase 1 寄与 + Phase 5 寄与。Phase 6 D2H 時に累積
    - E_solver: STS では 0（陽的スキームはエネルギー的に閉じるため追加項不要）。
      Hypre 有効時（`conduction.solver="hypre"`）は非ゼロ：反復ソルバ残差による
      \(E_{solver} = \sum_c C_{v,c} (T_{e,c}^{n+1} - T_{e,c}^n) V_c - \Delta t \sum_c (\nabla\cdot q)_c V_c\)
      を Hypre solve 後（§4.5 step 6）に計算する（NUMERICS §4.2.3）。\(C_{v,c} = \rho_c c_{v,e,c}\)。v1.0 既定は STS のため通常 0
- 後段で CUB `DeviceReduce::Sum` を3回（E_kin/E_int_e/E_int_i）

### 7.2b U6: qei_exchange

```
// U6: qei_exchange — Q_ei = C_{v,e}(T_e - T_i) / τ_eq の明示的評価（per-cell、NUMERICS §1.1.3）
// block=256, grid=(n_cells+255)/256, 1 thread = 1 cell, ~12 reg
// H11/H12 の前に実行し、Q_ei を計算。H11/H12 で e_i, e_e に反映する
```

```cpp
__global__ void qei_exchange(
    double* __restrict__ Q_ei,           // [n_cells] out: Q_ei [erg/cm³/s]
    const double* __restrict__ Te,       // [n_cells] 電子温度 [eV]
    const double* __restrict__ Ti,       // [n_cells] イオン温度 [eV]
    const double* __restrict__ rho,      // [n_cells]
    const double* __restrict__ Zbar,     // [n_cells] 有効電荷 Z̄_eff（多材料時は §1.1.6 の質量加重平均）
    const double* __restrict__ Cv_e,     // [n_cells] 電子質量比熱 c_v,e [erg/(g·eV)]（H13出力）
    const double* __restrict__ A_eff,    // [n_cells] 有効原子量（多材料: §1.1.6 調和平均、単一材料: A_ion）
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: NUMERICS §1.1.3 準拠。体積比熱 C_{v,e} = ρ × c_v,e をカーネル内で構成し、Q_ei = C_{v,e} × (T_e - T_i) / τ_eq を計算
  - τ_eq = 3.16e8 × A_eff × Te^(3/2) / (Z̄² × n_i × ln Λ_ei)（NUMERICS §1.1.3 数値形式）
  - **安全策**: `τ_eq < τ_eq_floor`（τ_eq_floor = 1e-30 s）の場合、`Q_ei = 0`（極端条件での Inf/NaN 防止）。`Te < Te_floor` or `Z̄ < Z̄_floor` の場合も `Q_ei = 0` にクランプ
  - n_i = ρ / (A_eff × m_p)、n_e = Z̄ × n_i
  - ln Λ_ei: §1.1.4 の Coulomb 対数（NRL Plasma Formulary、下限 lnΛ_min = 2、SPECIFICATION §6.4.7）
  - 符号規約: Q_ei > 0 → 電子→イオンへエネルギー移動（NUMERICS §1.1.3 符号規約）
  - **多材料セル**: A_eff（§1.1.6 調和平均）と Z̄_eff（§1.1.6 質量加重平均）をホスト/前段カーネルで算出し入力
- **レジスタ**: ~12（解析式のみ、テーブル参照なし）
- **メモリ**: coalesced read/write

### 7.2c U2: floor_clamp

> **実装**：このカーネルは存在しない。床の処理は各演算子のカーネルの中で行う — 流体の閉包
> （`enforce_1t/2t_closure_kernel`、1T の表セルは符号付きのエネルギーを許す、NUMERICS §3.1.5）、伝導の
> `clamp_1d_conduction_solution_kernel`、FLD・S\(_N\) の物質更新、レーザーの閉包。それぞれが注入エネルギーを
> `E_floor_injected` に、回数をステップのクランプ数（`clamp_warn_threshold`・`clamp_fatal_threshold` の判定に使う）に積む。
> 以下は設計時の記述。

```
// U2: floor_clamp — ρ >= ρ_floor, T_e >= T_e,floor, T_i >= T_i,floor をクランプ
// エネルギー補正: Δee = c_v × max(0, T_floor - T) [erg/g]（ee に直接加算）
// clamp_count を atomicAdd で加算
// block=256, grid=(n_cells+255)/256, ~8 reg
```

```cpp
__global__ void floor_clamp(
    double* __restrict__ rho,            // [n_cells] in/out
    double* __restrict__ Te,             // [n_cells] in/out
    double* __restrict__ Ti,             // [n_cells] in/out
    double* __restrict__ ee,             // [n_cells] in/out: エネルギー補正
    double* __restrict__ ei,             // [n_cells] in/out: エネルギー補正
    const double* __restrict__ Cv_e,     // [n_cells] 質量比熱 c_v,e [erg/(g·eV)]（H13出力）
    const double* __restrict__ Cv_i,     // [n_cells] 質量比熱 c_v,i [erg/(g·eV)]（H13出力）
    const double* __restrict__ velocity_r, // [n_nodes] 速度R成分（密度フロア運動エネルギー会計用）
    const double* __restrict__ velocity_z, // [n_nodes] 速度Z成分（1D_SPHではnullptr）
    const double* __restrict__ vol,      // [n_cells] セル体積
    double* __restrict__ E_floor_injected, // [1] atomicAdd: フロア補正の総注入エネルギー [erg]（NUMERICS §10.2）
    double rho_floor, double Te_floor, double Ti_floor,
    double Te_ceiling, double Ti_ceiling,  // 上限クランプ（SPECIFICATION §6.4.7: T_ceiling_eV、既定 1e6 eV）
    int* __restrict__ clamp_count,       // [1] atomicAdd で加算
    DeviceErrorFlags* error_flags,       // §0.6 準拠：大量クランプ時の WARNING
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 各セルで ρ, T_e, T_i, |u| を下限/上限と比較し、違反時にクランプ
  - **温度フロア**（NUMERICS §11.2）: `Δee = c_v × max(0, T_floor - T)` [erg/g] を比内部エネルギー ee に加算。E_floor_injected へは `ρ × c_v × ΔT × V` [erg] を atomicAdd
  - **密度フロア**（NUMERICS §11.1）: `ΔE_int = (ρ_floor - ρ_old) × (ee[c] + ei[c]) × V`（e_specific = ee + ei、電子+イオン比内部エネルギーの合計）、`ΔE_kin = 0.5 × (ρ_floor - ρ_old) × |u|² × V`
  - **速度リミッター**: U2 はセルベースカーネルのためノード速度は変更しない。速度上限 `|u| ≤ c` の強制は **H16 apply_hydro_bc**（ノードベース）で実施する（§2.6 参照）
  - 全 ΔE を `E_floor_injected` に `atomicAdd` で累積（NUMERICS §10.2 エネルギー収支）
  - clamp 発生時に atomicAdd で clamp_count をインクリメント
- **レジスタ**: ~8
- **メモリ**: coalesced read/write（全配列がセルインデックスでアクセス）

### 7.3 U4: cfl_reduction

```cpp
// Step 1: compute per-cell dt candidates
__global__ void compute_dt_candidates(
    double* __restrict__ dt_hydro_cell,   // [n_cells] out
    const double* __restrict__ c_s,
    const double* __restrict__ u_mag,
    const double* __restrict__ delta_l,
    const int8_t* __restrict__ hydro_active,
    double cfl,
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理（1D の実装、`cfl_1d_kernel`、NUMERICS §3.1.9）**: Lagrange 格子は流体と動くので分母に \(|u|\) を含めず、
  人工粘性の項を加える：`dt = C_CFL × Δr / (c_s + g_i C_1 c_s + C_2 Δr χ_i)`（\(g_i\) は圧縮セルで 1、`av_type="csw"` は
  `csw_C1`・`csw_C2`、riemann 系は面の圧縮ジャンプを加える）。分母が 0 のセルは制約しない。別に節点の交差の上限
  `crossing_dt_safety × Δr / (u_i − u_{i+1})`（既定 0.5、閉じる面だけ）と人工熱の上限を min で合わせる。
  2D は `cfl_2d_kernel`。以下の \(|u|+c_s\) の式は設計時のもの：`dt_hydro_cell[c] = cfl × Δl_c / (|u|_c + c_s[c])`
- **レジスタ**: ~10（c_s, u_mag, delta_l, dt_candidate）
- **メモリ**: coalesced read（全配列がセルインデックスでアクセス）
- 後段: CUB `DeviceReduce::Min` → dt_hydro scalar
- `hydro_active[c]==0` のセルは `dt_hydro_cell[c] = DBL_MAX` として CFL 計算から除外

**Step 2: dt_cond（伝導 CFL、NUMERICS §2.2 (b)）**

```cpp
__global__ void compute_dt_cond(
    double* __restrict__ dt_cond_cell,      // [n_cells] out
    const double* __restrict__ D_eff,       // [n_cells] 実効拡散係数 [cm²/s]（C1 出力）
    const double* __restrict__ delta_l,     // [n_cells] セル代表長 [cm]
    double cfl_cond,                        // C_cond 既定 0.25
    int sts_max_stages,                     // s_max 既定 40
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`

- STS: `dt_cond = s_max*(s_max+1)/2 * dt_exp`、`dt_exp = C_cond * min_c(Δl_c² / D_eff_c)`（Kirchhoff の面では
  Gershgorin 型の \(2C_iV_i/\sum_f G_f\) との min、NUMERICS §4.2.1）。この上限は STS のサブステップ分割の安全係数
  \(\eta\)（`sts_subcycle_eta`）を含まないので、伝導で刻みが決まるステップは 2 サブステップで進む（NUMERICS §2.2(b)）
- 陰解法（1D `solver="implicit"`、2D `"hypre"`）: `dt_cond_cell[c] = DBL_MAX`（伝導 CFL 制約なし）
- 後段: CUB `DeviceReduce::Min` → dt_cond scalar

**Step 3: dt_rad — 退役**。モンテカルロ輻射の Fleck 因子の下限による制約（`compute_dt_rad`、NUMERICS §2.2 (c)）で、FLD・\(S_N\) では
常に \(+\infty\) だった。2026-09-29 に Δt の候補と history の `dt_breakdown_history/dt_rad` から外した（設計は
`retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md`）。

**Step 4: host 側グローバル Δt 決定**

- `dt = min(dt_hydro, dt_cond, dt_user, dt_output, growth_factor * dt_old)`（NUMERICS §2.2）
- **dt_laser 省略**: NUMERICS §2.2(d) により `dt_laser := dt_hydro`（レーザーはサブステップを持つため独立制約なし）。U4 での明示的計算は不要
- `dt_output`: 時間間隔ベース出力の出力時刻整合（NUMERICS §2.2 (f)）。host側で計算（GPU不要）
- マルチGPU: 各 rank がローカル min 後に `MPI_Allreduce(MPI_MIN)` で 1 回の集約（NUMERICS §2.2）

### 7.4 U7: cell_search_after_rezone

退役（ALE rezone 後の光子粒子のセル再同定）。記述は `retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md` へ移した。

### 7.5 U5: nan_check

```cpp
__global__ void nan_check(
    const double* __restrict__ rho,         // [n_cells]
    const double* __restrict__ Te,          // [n_cells]
    const double* __restrict__ Ti,          // [n_cells]
    const double* __restrict__ ee,          // [n_cells]
    const double* __restrict__ ei,          // [n_cells]
    const double* __restrict__ Pe,          // [n_cells]
    const double* __restrict__ Pi,          // [n_cells]
    const double* __restrict__ v_r,         // [n_nodes]
    const double* __restrict__ v_z,         // [n_nodes]（1D_SPHではnullptr可）
    const double* __restrict__ vol,         // [n_cells] セル体積（vol≤0 は mesh_tangle、NaN は nan_detected）
    DeviceErrorFlags* error_flags,          // out: NaN/Inf検出フラグ
    int n_cells, int n_nodes
);
```

- **block**: 256, **grid**: `(max(n_cells, n_nodes)+255)/256`（セル・ノード両方を1パスで検査。`tid < n_cells` でセルフィールド、`tid < n_nodes` でノードフィールドをそれぞれ検査）
- **処理**: 前ステップの全State主要フィールドを走査し、NaN/Inf を検出。
  `isnan(x) || isinf(x)` で各値を検査。検出時に `error_flags` の対応フラグを `atomicExch` で設定
- **呼び出しタイミング**: Phase 0 先頭（§9）。前ステップの数値異常を早期検出し、診断情報をホストに返す
- **レジスタ**: ~8
- **メモリ**: coalesced read（全配列がセル/ノードインデックスでアクセス）。書き込みは error_flags の atomic のみ

### 7.6 U8: compute_zbar

```cpp
__global__ void compute_zbar(
    double* __restrict__ Zbar,              // [n_cells] out: 有効電荷数 Z̄_eff
    double* __restrict__ A_eff_out,         // [n_cells] out: 有効原子量 A_eff（NUMERICS §1.1.5(c)）
    const double* __restrict__ Te,          // [n_cells] in: 電子温度 [eV]
    const double* __restrict__ rho,         // [n_cells] in: セル平均密度 [g/cm³]
    const double* __restrict__ volFrac,     // [n_cells × n_mat] in: 体積分率 f_α（多材料時。n_mat==1 → nullptr 可）
    const int* __restrict__ material_id,    // [n_mat] 材料ID（テーブル選択に使用）
    const double* __restrict__ A_mat,       // [n_mat] 材料ごとの原子量 A_α [amu]
    const double* __restrict__ Z_mat,       // [n_mat] 材料ごとの原子番号 Z_α（常に必要: fixed+n_mat>1 は Z_mat[α] を Z̄_α として使用、thomas_fermi/tabular はテーブル入力に使用）
    int zbar_model,                         // 0=fixed, 1=thomas_fermi, 2=tabular（IONMIX Z̄(ρ,Te) 補間）
    double Zbar_fixed,                      // zbar_model==0 時の固定値
    const void* __restrict__ zbar_table,    // Z̄ テーブル（zbar_model==1: Thomas-Fermi, zbar_model==2: IONMIX。zbar_model==0 かつ n_mat==1 → nullptr 可）
    DeviceErrorFlags* error_flags,          // volfrac_degenerate フラグ（§0.6 準拠）
    int n_cells, int n_mat
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
> **1D の実装**：このカーネルは無い。初期値は host（`geometry_eval.cpp`）、ステップの更新は `update_zbar_fields_kernel`
> （`zbar_device.cu`、thomas_fermi・tabular）で作る。多材料セルの混合はどの経路も `ZbarMixAccumulator`（`zbar_math.hpp`）の
> イオン数の重み \(f_\alpha/A_\alpha\)（void を除く）。`model="fixed"` は `fixed_value ≥ 0` なら全材料・全セルにその値を使い、
> 負（未指定）なら材料の \(Z_\alpha\) を同じ重みで混ぜる。fixed の \(\bar Z\) はステップごとに更新しない。
> `volfrac_degenerate` のフラグは無い。

- **処理**（NUMERICS §1.1.4 + §1.1.5(c) 準拠）:
  - **fixed, n_mat==1**: `Zbar[c] = Zbar_fixed`, `A_eff_out[c] = A_mat[0]`
  - **thomas_fermi, n_mat==1**: Z̄ = TF_table(ρ, Te, Z_mat[0]), `A_eff_out[c] = A_mat[0]`
  - **tabular, n_mat==1**: Z̄ = IONMIX_table(ρ, Te, material_id[0]), `A_eff_out[c] = A_mat[0]`
  - **多材料セル** (n_mat > 1): 各材料 α の Z̄_α を個別に評価:
    - **fixed**: Z̄_α = Z_mat[α]（完全電離仮定、テーブル不参照）
    - **thomas_fermi**: Z̄_α = TF_table(ρ, Te, Z_mat[α])
    - **tabular**: Z̄_α = IONMIX_table(ρ, Te, material_id[α])
    single-state仮定（全材料同一 ρ, Te）のもと:
    - f_{m,α} = volFrac[c,α]（single-state: f_m = f_vol、NUMERICS §1.1.5(c)）
    - Z̄_eff = Σ_α (f_{m,α} Z̄_α / A_mat[α]) / Σ_α (f_{m,α} / A_mat[α])（NUMERICS §1.1.5(c)）
    - A_eff = 1 / Σ_α (f_{m,α} / A_mat[α])（調和平均、NUMERICS §1.1.5(c)）
  - **退化ガード**: volFrac 合計が 1±ε から大きく外れる場合は `error_flags->volfrac_degenerate = 1`
- **呼び出しタイミング**: Phase 0（§9）。C1, L1, R1 が Z̄ を参照するため最新化が必要
- **レジスタ**: ~12（テーブル補間 + 材料ループ変数）
- **メモリ**: Z̄ テーブル（Thomas-Fermi / IONMIX）は `__ldg()` で L2 キャッシュ経由参照

### 7.7 U9: compute_opacities

退役（IMC の不透明度の前計算）。現行は各輻射ソルバーが不透明度を評価する（ARCHITECTURE §4.3.4）。記述は
`retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md` へ移した。

---

## 8. Parallel カーネル群

### 8.1 P1-P4: halo_pack/unpack_cell/node

```cpp
__global__ void halo_pack_cell(
    double* __restrict__ send_buf,          // [n_halo × n_fields] out
    const double* __restrict__* field_ptrs, // [n_fields] 各フィールドのポインタ配列
    const int* __restrict__ halo_indices,   // [n_halo] ゴーストセルインデックス
    int n_halo, int n_fields
);
```

- **block**: 256, **grid**: `(n_halo+255)/256`
- **処理**: 各スレッドが1ゴーストセルの全フィールドをパック
  - `send_buf[tid * n_fields + f] = field_ptrs[f][halo_indices[tid]]`
- n_fields は典型 5-10（ρ, Te, Ti, P, Q, ...）
- **レジスタ**: ~10（フィールドループ変数 + ポインタ）
- **メモリ**: `halo_indices` による間接参照でフィールド読み込みは scattered。`send_buf` への書き込みは cell-major（`tid * n_fields + f`）。ゴーストセル数が小さい（~数百）ため帯域は問題にならない
- **型制約**: 上記シグネチャは double フィールド専用。int8 フィールド（hydro_active、NUMERICS §12.2.2）はバッファ末尾に別途 int8→int8 コピーで処理する（double キャストは行わない）。P2/P3/P4 も同様

```cpp
__global__ void halo_unpack_cell(
    double* __restrict__* field_ptrs,       // [n_fields] 各フィールドのポインタ配列（out）
    const double* __restrict__ recv_buf,    // [n_halo × n_fields] in
    const int* __restrict__ halo_indices,   // [n_halo] ゴーストセルインデックス
    int n_halo, int n_fields
);
```

- **block**: 256, **grid**: `(n_halo+255)/256`
- **処理**: P1 の逆操作。各スレッドが1ゴーストセルの全フィールドを受信バッファから展開し、`field_ptrs[f][halo_indices[tid]]` に書き戻す

```cpp
__global__ void halo_pack_node(
    double* __restrict__ send_buf,               // [n_halo_nodes × n_fields] out
    const double* __restrict__* field_ptrs,      // [n_fields] ノードフィールドのポインタ配列
    const int* __restrict__ halo_node_indices,   // [n_halo_nodes] ゴーストノードインデックス
    int n_halo_nodes, int n_fields
);
```

- **block**: 256, **grid**: `(n_halo_nodes+255)/256`
- **処理**: P1 のノード版。各スレッドが1ゴーストノードの全フィールドを `field_ptrs[f][halo_node_indices[tid]]` から読み取り、`send_buf` にパック

```cpp
__global__ void halo_unpack_node(
    double* __restrict__* field_ptrs,            // [n_fields] ノードフィールドのポインタ配列（out）
    const double* __restrict__ recv_buf,         // [n_halo_nodes × n_fields] in
    const int* __restrict__ halo_node_indices,   // [n_halo_nodes] ゴーストノードインデックス
    int n_halo_nodes, int n_fields
);
```

- **block**: 256, **grid**: `(n_halo_nodes+255)/256`
- **処理**: P3 の逆操作。各スレッドが1ゴーストノード分を `recv_buf` から復元し、`field_ptrs[f][halo_node_indices[tid]]` へ書き込む

### 8.2 P5: emigrant_detect_pack

退役（光子粒子の rank 間移動。§8.3 P6 も同じ）。記述は `retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md` へ移した。

### 8.3 P6: immigrant_unpack_merge

退役（§8.2 を参照）。

---
