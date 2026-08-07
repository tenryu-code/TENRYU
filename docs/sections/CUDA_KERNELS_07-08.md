<!-- 分割元: docs/CUDA_KERNELS.md | このファイルは参照用です。原本（docs/CUDA_KERNELS.md）が権威です。 -->
## 7. Coupling/Utility カーネル群

### 7.1 U1: source_injection

```cpp
__global__ void source_injection(
    double* __restrict__ ee,              // [n_cells] in/out: 電子比内部エネルギー
    const double* __restrict__ laser_dep, // [n_cells] レーザー沈着 [erg]
    const double* __restrict__ rad_dep,   // [n_cells × G] 輻射沈着
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    double* __restrict__ E_numerical_loss, // [1] atomicAdd: 退化セル（ρV<1e-30）の未注入エネルギー [erg]（ARCHITECTURE §5.2）
    DeviceErrorFlags* error_flags,        // §0.6 準拠：負 rad_dep/laser_dep 検出
    int n_cells, int n_groups
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 退化セルガード + 比エネルギー注入（NUMERICS §10.2 エネルギー収支）:
  - `ρV = rho[c] × vol[c]`
  - **退化ガード**: `ρV < 1e-30` の場合、`ee[c]` は変更せず、未注入エネルギー `(laser_dep[c] + Σ_g rad_dep[c,g])` を `atomicAdd(&E_numerical_loss[0], ...)` で計上して return（ARCHITECTURE §5.2 E_numerical_loss 規約）
  - **負 dep 検出**（§0.6 negative_source_dep）: `dep = laser_dep[c] + Σ_g rad_dep[c,g]` を計算後、`dep < 0` の場合は `atomicExch(&error_flags->negative_source_dep, 1)` + `atomicAdd(&E_numerical_loss[0], -dep)` + `dep = 0`（ゼロクランプ）
  - 通常パス: `ee[c] += dep / ρV`
  - rad_dep は [erg]（NUMERICS §10.1 の規約に従い、ρV で除算して比内部エネルギーに変換）
- **二重計上防止プロトコル**（NUMERICS §2.1 準拠）:
  NUMERICS は `nullptr` ゲーティング（Phase 3: `rad_dep=nullptr`、Phase 4: `laser_dep=nullptr`）を規約化するが、
  GPU 実装では **分岐回避のためゼロ初期化方式** を採用する（結果は同一）:
  - Phase 3 の U1 呼び出し時: `laser_dep` は L5 出力（有値）、`rad_dep` は Phase 0 の `cudaMemsetAsync(rad_dep, 0)` でゼロ保証済み
  - Phase 4 の U1 呼び出し時: `rad_dep` は R8/R9/R12 の atomicAdd 累積出力（有値。R10 は rad_E_tally の正規化のみで rad_dep には書き込まない）、`laser_dep` は Phase 4 冒頭で `cudaMemsetAsync(laser_dep, 0)` → ゼロ化
  - **不変量**: 各ソース（laser_dep, rad_dep）は e_e に **正確に1回** 注入される
  - **実装者注意**: カーネルは常に両項を加算する。呼び出し側で「使用しない」ソースバッファがゼロであることを保証すること。ゼロ保証が崩れると二重計上/未計上が発生する
- 保存性検証は別カーネル（Kahan sum reduction）で行う
- **レジスタ**: ~10（群ループの部分和 + rho/vol 一時変数）
- **メモリ**: coalesced read/write。rad_dep[n_cells×G] は群ループで stride-G アクセス

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
    const double* __restrict__ E_escape,  // [G] 境界脱出エネルギー（R8/R9 atomicAdd 出力）
    const double* __restrict__ E_numerical_loss, // [1] 数値損失（R8 MAX_EVENTS超過 / P6粒子喪失。R12 roulette消滅はrad_depに沈着、§6.3.4）
    int n_cells, int n_groups
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: セルベース3成分をブロック内 shared memory で部分和 → global（NUMERICS §10.2 準拠）:
- **同期**: ブロック内 reduction に `__syncthreads()` が必要（CUB `BlockReduce` 使用時は自動挿入）
  1. **E_kin**: 運動エネルギー = Σ ½ρu² V
  2. **E_int_e**: 電子内部エネルギー = Σ ρ e_e V
  3. **E_int_i**: イオン内部エネルギー = Σ ρ e_i V
  4. **E_rad**: 放射場エネルギー = **E_census = Σ_p E_p**（alive粒子のエネルギー合計、**R12 Russian roulette 適用後**）。difference path では `E_census = Σ U_ref + Σ_p sign[p]E_p`。
     **注意**: track-length推定量 `rad_E[c,g]×V_c` ではなくcensus energy を使用する（NUMERICS §10.2: ΔE_rad = E_census^{n+1} - E_census^{n}）。
     legacy path の E_census は **別途 CUB `DeviceReduce::Sum`** で粒子配列 `energy[0..n_alive-1]` を **alive==1 フィルタ付き**で集約して算出する（粒子レベル reduction、U3 カーネル外で実行。n_alive は R12 後の第2 R7 ソートで確定。ALE U7 が alive=0 にした粒子を除外するためフィルタ必須 — §9 Phase 6 参照）。difference path は deterministic reference reservoir と signed residual 粒子和を加える。
     `rad_E` は inject_sources（U1）の物質結合にのみ使用し、エネルギー収支には使用しない
  5. **E_escape**: 境界脱出累積エネルギー（群別 `E_escape[g]` を集約して使用）
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
- 後段で CUB `DeviceReduce::Sum` を3回（E_kin/E_int_e/E_int_i）+ E_census用に1回（粒子）

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
- **処理**: `dt_hydro_cell[c] = cfl × Δl_c / (|u|_c + c_s[c])`
  - **分母ゼロ安全策**: `denom = |u|_c + c_s[c]`。`denom < 1e-30` の場合は `dt_hydro_cell[c] = DBL_MAX`（無拘束）。
    物理的に |u|=0 かつ c_s=0 は静止かつ圧力なしのセルであり、CFL 制約なしが正しい
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

- STS: `dt_cond = s_max*(s_max+1)/2 * C_cond * min_c(Δl_c² / D_eff_c)`
- Hypre: `dt_cond_cell[c] = DBL_MAX`（伝導 CFL 制約なし）
- 後段: CUB `DeviceReduce::Min` → dt_cond scalar

**Step 3: dt_rad（輻射 CFL、NUMERICS §2.2 (c)）**

```cpp
__global__ void compute_dt_rad(
    double* __restrict__ dt_rad_cell,       // [n_cells] out
    const double* __restrict__ sigma_P,     // [n_cells] Planck 吸収係数 [cm⁻¹]（U9 出力）
    const double* __restrict__ Te,          // [n_cells] 電子温度 [eV]
    const double* __restrict__ Cv_e,        // [n_cells] 電子比熱 [erg/(g·eV)]
    const double* __restrict__ rho,         // [n_cells] 密度 [g/cm³]
    double f_min,                           // Fleck factor 下限 既定 0.01
    double alpha,                           // IMC α パラメータ 既定 1.0
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`

- β_c = 4 a_eV T_e³ / (ρ_c × Cv_e[c])（NUMERICS §6.1。Cv_e は質量比熱 [erg/(g·eV)] のため、体積比熱 C_v = ρ×c_v に変換が必要。§0.1.1 規約準拠）
- σ_P,c = ρ_c κ_P,c（U9 で計算済み）
- **分母ゼロ安全策**: `denom = f_min * α * c_light * β_c * σ_P,c`。`denom < ε_dt_rad`（ε_dt_rad = 1e-30 [s⁻¹]）の場合は `dt_rad_cell[c] = DBL_MAX`（無拘束）。
  これにより σ_P ≈ 0（透明領域）や β ≈ 0（低温領域）で Inf/NaN が CUB Min に伝播することを防止する。
  物理的に σ_P=0 は radiation CFL 制約なし（Fleck因子 f→1）に対応するため、DBL_MAX は正しい
- 通常パス: `dt_rad_cell[c] = (1-f_min) / denom`
- 後段: CUB `DeviceReduce::Min` → dt_rad scalar

**Step 4: host 側グローバル Δt 決定**

- `dt = min(dt_hydro, dt_cond, dt_rad, dt_user, dt_output, growth_factor * dt_old)`（NUMERICS §2.2）
- **dt_laser 省略**: NUMERICS §2.2(d) により `dt_laser := dt_hydro`（レーザーはサブステップを持つため独立制約なし）。U4 での明示的計算は不要
- `dt_output`: 時間間隔ベース出力の出力時刻整合（NUMERICS §2.2 (f)）。host側で計算（GPU不要）
- マルチGPU: 各 rank がローカル min 後に `MPI_Allreduce(MPI_MIN)` で 1 回の集約（NUMERICS §2.2）

### 7.4 U7: cell_search_after_rezone

```cpp
__global__ void cell_search_after_rezone(
    int32_t* __restrict__ cell_id,          // [n_alive] in/out: 粒子セルID（更新対象）
    const double* __restrict__ pos_r,       // [n_alive] in: 粒子位置R
    const double* __restrict__ pos_z,       // [n_alive] in: 粒子位置Z
    const double* __restrict__ energy,      // [n_alive] in: 粒子エネルギー（未発見粒子の E_numerical_loss 会計に必要）
    uint8_t* __restrict__ alive,             // [n_alive] in/out: alive==1 のみ処理。未発見粒子は alive=0 に設定（§7.4 最終フォールバック）
    const uint8_t* __restrict__ mode,       // [n_alive] in: **DDMC粒子(mode==DDMC, 即ち mode==1)はスキップ**（pos=NaN sentinel のため位置探索不可。
                                            //   DDMC は cell_id を直接保持し、ALE rezone はセルIDを変えないため再探索不要）
    const double* __restrict__ x_r,         // [n_nodes] 新メッシュ節点R座標
    const double* __restrict__ x_z,         // [n_nodes] 新メッシュ節点Z座標
    const int* __restrict__ hash_grid,      // [M_R × M_Z × max_per_bin] ハッシュグリッド（§9.5）
    int M_R, int M_Z, int max_per_bin,      // ハッシュグリッドパラメータ
    double* __restrict__ E_numerical_loss,  // [1] atomicAdd: 未発見粒子のエネルギー損失
    DeviceErrorFlags* error_flags,          // cell_search_fail フラグ
    bool cell_search_fatal,                 // True=未発見で fatal、False=消滅+会計
    int max_walk,                           // stencil walk 最大ステップ数（既定 20、ARCHITECTURE §4.1.2 CellSearchConfig）
    int max_rings,                          // リング拡張の最大リング数（既定 3、§9.4）
    int n_alive, int nr, int nz
);
```

- **block**: 128, **grid**: `(n_alive+127)/128`
- **処理**: ALE rezone 後、**IMC alive粒子のみ** cell_id を再計算（DDMC粒子は `mode==DDMC`（=1）で早期リターン — pos=NaN のため位置探索不可。DDMC は cell_id をそのまま維持する。ALE rezone はセル番号を変えないため、cell_id の更新は不要）。3段階フォールバック（NUMERICS §9.3-9.5）:
  1. **Stencil walk**（§9.3）: 現在の cell_id から開始し、隣接セルを探索（point-in-quad 判定）
     - rezone 後のメッシュ変位が小さければ 1-2 ステップで収束
  2. **リング拡張**（§9.4）: stencil walk が max_walk（既定 20、ARCHITECTURE §4.1.2 CellSearchConfig）で収束しない場合、
     チェビシェフ距離 k=2,3,...,max_rings(既定3) のリング状近傍を探索
  3. **背景 hash grid**（§9.5）: リング拡張でも失敗した場合、空間ハッシュ構造（M_R×M_Z 一様格子）
     から候補セルを取得し point-in-quad 判定。hash grid でも未発見の場合は brute-force 全セル走査（O(N_cells)）
  4. **最終フォールバック**: 全探索で未発見の場合は**領域外逸脱**と判定。
     `cell_search.fatal=True`（既定、ARCHITECTURE §4.1.2 CellSearchConfig::fatal）なら fatal error で停止。
     `False` なら粒子エネルギーを `E_numerical_loss` に加算し粒子を消滅（error_flags->invalid_cell に記録。
       領域外逸脱は物理的脱出ではなく数値的損失のため `E_rad_escaped` ではなく `E_numerical_loss` に計上する）
- **レジスタ**: ~20（セル頂点座標 4×2 + 粒子位置 2 + walk カウンタ + hash 一時変数）
- **呼び出しタイミング**: Phase 1/5 の ALE rezone 後（§9 カーネル起動シーケンス参照）

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

```cpp
__global__ void compute_opacities(
    double* __restrict__ sigma_a,           // [(n_cells+n_ghost) × G] out: 吸収係数 [cm⁻¹]
    double* __restrict__ sigma_s,           // [(n_cells+n_ghost) × G] out: 散乱係数 [cm⁻¹]
    double* __restrict__ sigma_t,           // [(n_cells+n_ghost) × G] out: 全係数 [cm⁻¹]（σ_a + σ_s）
    double* __restrict__ sigma_R,           // [(n_cells+n_ghost) × G] out: Rosseland平均 [cm⁻¹]
    double* __restrict__ sigma_P,           // [(n_cells+n_ghost)] out: Planck平均 [cm⁻¹]（群積分済み、dt_rad用）
    const double* __restrict__ rho,         // [(n_cells+n_ghost)]
    const double* __restrict__ Te,          // [(n_cells+n_ghost)]
    const double* __restrict__ Zbar,        // [(n_cells+n_ghost)]
    const double* __restrict__ volFrac,     // [(n_cells+n_ghost) × n_mat]（多材料時）
    const void* __restrict__ opacity_table, // IONMIX/constant テーブル
    int opacity_model,                      // 0=constant, 1=ionmix
    double kappa_a_const, double kappa_s_const, // opacity_model==0 時の質量不透明度定数値 [cm²/g]（SPEC §5.2: kappa_a, kappa_s）
    double kappa_floor,                     // 質量不透明度κの安全下限 [cm²/g]（既定 1e-20。SPEC §6.4.7 opacity_floor）。
                                            //   カーネル内で σ_floor = ρ[cell] × kappa_floor [cm⁻¹] を per-cell 計算し、
                                            //   max(σ, σ_floor) でフロア適用する。密度依存にすることで、
                                            //   低密度セル（corona等）で過剰クランプを防止
    int* __restrict__ opacity_clamp_count,  // [1] out: フロアクランプ回数（atomicAdd）
    DeviceErrorFlags* __restrict__ error_flags,  // opacity_out_of_range: テーブル範囲外検出（§0.6）
    int n_cells, int n_ghost, int n_mat, int G
);
```

- **block**: 256, **grid**: `((n_cells+n_ghost)+255)/256`
- **処理**（ARCHITECTURE §4.7、NUMERICS §6.1 準拠）:
  - **constant**: `σ_a = ρ × κ_a_const`, `σ_s = ρ × κ_s_const`（質量吸収係数 κ [cm²/g] → 線吸収係数 σ [cm⁻¹]）
- **ionmix**: IONMIX テーブルから (ρ, T_e) → κ_a(g), κ_R(g) を双対数補間 → σ = ρ × κ。**散乱**: v1.0 では物理散乱を実装しない（NUMERICS §6.1）。σ_{s,g} = 0（全セル・全群）。将来版で Thomson 散乱等を導入する場合はテーブルに κ_s 列を追加する
  - **多材料セル**（NUMERICS §1.1.6 不透明度混合則 準拠）: 各材料 α の κ_α を個別に評価し、**質量分率** f_{m,α} で混合:
    - **Planck/吸収 (κ_a, κ_P)**: 質量加重線形平均 κ_{P,mix} = Σ_α f_{m,α} κ_{P,α} → σ_{P,mix} = ρ × κ_{P,mix}
    - **Rosseland (κ_R)**: 質量加重**調和平均** 1/κ_{R,mix} = Σ_α f_{m,α}/κ_{R,α} → σ_{R,mix} = ρ × κ_{R,mix}
      調和平均前に各成分にフロア適用: κ_{R,α} ≥ κ_floor（NUMERICS §1.1.6）
    - **散乱 (κ_s)**: 質量加重線形平均（吸収と同一方式）
    - f_{m,α} は volFrac[c,α] から算出（single-state: f_{m,α} = f_α / Σ_β f_β ≡ f_α、NUMERICS §1.1.5(c)）
  - **フロア適用**: `σ_floor = ρ × kappa_floor`（per-cell計算）、`σ_a = max(σ_a, σ_floor)`, `σ_R = max(σ_R, σ_floor)`。適用時に `opacity_clamp_count` をインクリメント
  - **ゴーストセル**: R3 が隣接セルの opacity を参照するため、ゴーストセルを含めて計算（n_ghost > 0）。ゴーストセルの ρ,Te 等は事前の halo_exchange で充填済み
- **呼び出しタイミング**: Phase 4 先頭（§9）。R1, R3 が不透明度を参照する
- **レジスタ**: ~15（テーブル補間 + 群ループ変数）
- **メモリ**: IONMIX テーブルは `__ldg()` で L2 キャッシュ経由。σ_a/σ_s/σ_t/σ_R は coalesced write（(n_cells+n_ghost)×G の1Dレイアウト）

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

```cpp
__global__ void emigrant_detect_pack(
    void* __restrict__ emigrant_buf,        // AoS packed buffer (104B/particle, ARCHITECTURE §7.1.3)
    int* __restrict__ emigrant_count,       // [1] atomic counter
    int* __restrict__ per_dest_count,       // [8] 宛先毎のカウント
    int emigrant_capacity,                 // emigrant_buf の最大粒子数（オーバーフロー防止）
    // PhotonPool SoA — 104B ParticleEmigrant パックに全フィールド必要
    const double* __restrict__ pos_r,
    const double* __restrict__ pos_z,
    const double* __restrict__ dir_r,
    const double* __restrict__ dir_z,
    const double* __restrict__ dir_phi,
    const double* __restrict__ energy,
    const double* __restrict__ weight,
    const double* __restrict__ time_remain,
    const double* __restrict__ birth_energy,
    const int8_t* __restrict__ sign,
    const uint64_t* __restrict__ global_id,
    const uint32_t* __restrict__ rng_counter,
    const int32_t* __restrict__ cell_id,
    const uint16_t* __restrict__ group_id,
    const uint8_t* __restrict__ mode,
    uint8_t* __restrict__ alive,           // in/out: パック後 alive=2（OVERFLOW）に設定（NUMERICS §12.3.1）
    // Partition info
    const int* __restrict__ neighbor_ranks,  // [n_faces] 面方向の隣接rank（face→rank マッピング）
    int n_faces,                             // 面数（2D_RZ=4, 1D_SPH=2）
    int ir_start, int ir_end,
    int jz_start, int jz_end,
    int nr_local, int nz_local,
    int n_alive,
    // Safety
    double* __restrict__ E_numerical_loss, // [1] オーバーフロー時のエネルギー保存
    DeviceErrorFlags* error_flags          // emigrant_overflow フラグ（EmigrantBuffer容量超過、NUMERICS §12.3.1）
);
```

- **block**: 128（粒子カーネル標準；NUMERICS §12.5, ARCHITECTURE §5.6.1 準拠）, **grid**: `(n_alive+127)/128`
- **処理**: 各粒子の `alive == 1 && cell_id < 0`（R8/R9 のパーティション境界マーク。ARCHITECTURE §7.1.3 統一契約: `cell_id = -(100+face)`）をチェック → emigrant_bufに104Bパック。alive=0（dead/escaped）かつcell_id<0の粒子は vacuum/marshak 脱出済みのためスキップ（二重パック防止）
- **宛先rank決定**（ARCHITECTURE §7.1.3 統一契約、IMC/DDMC共通）:
  `face = -(cell_id) - 100`（R8/R9 がともに `cell_id = -(100 + crossing_face)` で統一エンコード）。
  `dest_rank = neighbor_ranks[face]`（face=0:R_left, 1:R_right, 2:Z_bottom, 3:Z_top）。
  IMC は粒子位置での検証も可能だが、face 規約が正規の判定手段。
  DDMC は位置が NaN（ソースセルのグローバル座標を一時格納）のため face 規約が唯一の手段
- **face デコード**: `face = -(cell_id) - 100`（ARCHITECTURE §7.1.3。face=0:R_left, 1:R_right, 2:Z_bottom, 3:Z_top）
  → `dest = neighbor_ranks[face]`。IMC/DDMC共通。
  **face 範囲検証**: `face < 0 || face >= n_faces`（2D_RZ: n_faces=4, 1D_SPH: n_faces=2）の場合、パック**しない** → `error_flags->emigrant_invalid_face = 1` → `alive = 0` → エネルギーを `E_numerical_loss` に加算。R8/R9 の cell_id エンコーディングバグによる OOB `neighbor_ranks[face]` アクセスを防止
- **カウント＆パック手順**:
  1. `slot = atomicAdd(emigrant_count, 1)` でスロット取得
  2. `slot < emigrant_capacity` の場合: バッファに 104B パック（mode/sign 含む）→ `atomicAdd(per_dest_count[dest], 1)` → `alive = 0`
  3. `slot >= emigrant_capacity` の場合: パック**しない** → `per_dest_count` も更新**しない** → `error_flags->emigrant_overflow = 1` → `alive = 2`（OVERFLOW） → エネルギーを `E_numerical_loss` に加算（保存則維持、NUMERICS §12.3.1 準拠）
  - **per_dest_count 整合性**: `per_dest_count[dest]` は成功パック時のみインクリメントする。オーバーフロー粒子をカウントすると `sum(per_dest_count) > n_send` となり MPI で OOB 読み込みが発生する
  - OVERFLOW粒子は R7 で `alive != 1` により comp_key = 0xFFFFFFFF → dead として除去される
- **レジスタ**: ~15（cell_id → (i,j) 変換、範囲判定、パック一時変数）
- **ワープ発散**: emigrant は全粒子の少数（典型 <1%）のため、大半のスレッドは early return。ワープ発散のコストは emigrant パック処理の回数が少ないため影響軽微
- **メモリ**: emigrant_buf への書き込みは atomicAdd で取得したオフセットに基づく scattered write。104B/particle の AoS パック（ARCHITECTURE §7.1.3 ParticleEmigrant 準拠）

### 8.3 P6: immigrant_unpack_merge

```cpp
__global__ void immigrant_unpack_merge(
    // PhotonPool SoA — 受信粒子をマージ
    double* __restrict__ pos_r,
    double* __restrict__ pos_z,
    double* __restrict__ dir_r,
    double* __restrict__ dir_z,
    double* __restrict__ dir_phi,
    double* __restrict__ energy,
    double* __restrict__ weight,
    double* __restrict__ time_remain,
    double* __restrict__ birth_energy,
    int8_t* __restrict__ sign,
    uint64_t* __restrict__ global_id,
    uint32_t* __restrict__ rng_counter,
    int32_t* __restrict__ cell_id,
    uint16_t* __restrict__ group_id,
    uint8_t* __restrict__ mode,
    uint8_t* __restrict__ alive,
    // Receive buffer
    const void* __restrict__ recv_buf,  // AoS packed (104B/particle, ARCHITECTURE §7.1.3)
    int n_recv,                         // 受信粒子数（MPI_Irecv で取得）
    int pool_offset,                    // SoA 書き込み開始位置 = n_alive（ホスト計算）
    int pool_capacity,                  // プール容量上限
    // Cell search (§9)
    const double* __restrict__ node_r,  // [n_nodes] 受信側メッシュ節点R座標（§9.1 point-in-cell用）
    const double* __restrict__ node_z,  // [n_nodes] 受信側メッシュ節点Z座標
    int nr_local, int nz_local,         // ローカルセル数（stencil walk範囲制限）
    int ir_start, int jz_start,         // ローカル領域オフセット（DDMC宛先セル算出に必要）
    int n_faces,                        // 面数（2D_RZ=4, 1D_SPH=2）。DDMC face 再検証用（P5ガードの防御的二重チェック）
    const int8_t* __restrict__ face_bc_type, // [n_faces_boundary] 境界面タイプ（§9.3 stencil walk 停止判定）
    // Safety
    double* __restrict__ E_numerical_loss,
    int* __restrict__ n_recv_accepted,      // [1] out: 実際にマージ成功した粒子数（atomicAdd。pool_capacity超過分を除外。
                                            //   ホスト側で N_alive_post_mpi = n_alive_pre_mpi - n_emigrant + n_recv_accepted とする）
    DeviceErrorFlags* error_flags
);
```

- **block**: 128, **grid**: `(n_recv+127)/128`
- **処理**:
  1. `recv_buf` から 104B ParticleEmigrant を読み出し、SoA フィールドに展開
  2. **セル再同定**（NUMERICS §12.3.2 + §9 準拠、**モード依存**）:
     - **IMC（mode==0）**: 粒子の実位置 `(pos_r, pos_z)` から §9 stencil walk でローカル cell_id を決定。
       初期推定セル: 受信方向の境界セルから開始。stencil walk 失敗時は §9.4 リング拡張 → §9.5 ハッシュグリッドの順でフォールバック。
       全探索失敗時は粒子を消滅させ `E_numerical_loss` に計上（§9.6）
     - **DDMC（mode==1）**: 位置が NaN sentinel ではなくソースセルのグローバル座標が格納済み（R9 emigrant 規約、ARCHITECTURE §7.1.3）。
       **int cast 安全策**: `!isfinite(pos_r) || !isfinite(pos_z)` の場合、粒子を消滅させ `E_numerical_loss` に計上（NaN/Inf → int cast は C++ UB）。
       `i_src = (int)pos_r`, `j_src = (int)pos_z`, `face = leak_face`（ParticleEmigrant から取得）から宛先セルを算術決定。
       **face 範囲再検証（防御的）**: `face < 0 || face >= n_faces` の場合は粒子消滅 + `E_numerical_loss` 計上（P5 が送信側でガード済みだが、MPI 転送後の防御策。1D_SPH で face=2/3 が到着した場合、cell_local が別方向の有効セルにエイリアスしうるためサイレント誤配置を防止）。P6 に `n_faces` 引数を追加すること:
       face=0(R_left): `cell_local=(i_src-1-ir_start)*nz_local+(j_src-jz_start)`,
       face=1(R_right): `cell_local=(i_src+1-ir_start)*nz_local+(j_src-jz_start)`,
       face=2(Z_bottom): `cell_local=(i_src-ir_start)*nz_local+(j_src-1-jz_start)`,
       face=3(Z_top): `cell_local=(i_src-ir_start)*nz_local+(j_src+1-jz_start)`.
       0 <= cell_local < n_local_cells を検証。超過時は `E_numerical_loss` に計上。
       マージ後 `pos_r = pos_z = NaN`（DDMC NaN sentinel を復元）
  3. `cell_id` をローカルセルIDに設定（step 2 で算出）。`group_id` は変更不要（ARCH統一契約: リーク面は `cell_id`/`leak_face` にエンコード、group_id は汚染されない）
  4. `alive = 1` に設定
  5. **容量チェック + 連続配置**: `write_slot = pool_offset + atomicAdd(n_recv_accepted, 1)` で書き込みスロットを予約。`write_slot >= pool_capacity` の場合は書き込みをスキップし `error_flags->particle_overflow = 1`、エネルギーを `E_numerical_loss` に加算、`atomicSub(n_recv_accepted, 1)` でカウントを戻す。成功時のみ SoA に書き込み
- **連続配置の保証**: `atomicAdd` による順序付けにより、受理粒子は `[pool_offset, pool_offset + n_recv_accepted - 1]` に隙間なく連続配置される。これにより第2R7のスキャン範囲 `[0, N_alive_post_mpi)` に未初期化スロットが混入することを防止する（従来の `pool_offset + thread_idx` 方式では容量超過時に非連続な穴が生じ、R7が未初期化メモリを走査するリスクがあった）
- **global_id 不変**: 受信粒子の `global_id` は送信元で割り当て済み（RNG 独立性を維持するため変更不可）
- **レジスタ**: ~15（AoS→SoA 展開 + cell_id 変換）
- **メモリ**: recv_buf は coalesced read（104B aligned）、SoA は scattered write（pool_offset 以降、atomicAdd 順序で連続配置）

---

