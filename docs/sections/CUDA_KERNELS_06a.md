<!-- 分割元: docs/CUDA_KERNELS.md | このファイルは参照用です。原本（docs/CUDA_KERNELS.md）が権威です。 -->
## 6. Radiation カーネル群（詳細設計）

> **【状態注記 2026-07-10】** 本章の R2–R14 カーネル群（mode judge、粒子ソース、composite sort、IMC persistent-warp 輸送、DDMC event loop、tally 等）は**退役 `mode="imc_ddmc"` の粒子輸送カーネル**であり歴史的仕様（SPECIFICATION 冒頭ステータス参照）。例外: **fleck.cu は dt_rad 制限（`compute_dt_rad_limit`）として現行経路からも利用される**。ただし現行 FLD の Fleck 因子は fleck.cu の R1 ではなく各ソルバ側（`compute_fleck_for_fld_kernel` / nlte_coeffs.cu）が生成する — R1 は退役 IMC 経路の生成器。**現行の決定論放射カーネルの設計章は本章 §6.7（FLD）/ §6.8（S_N）** — 2026-07-10 の doc 監査で新設（数理は NUMERICS §6.7/§6.8 が正、実装 file:line 引用付き）。

> 本節では全Radiationカーネルの詳細仕様を記載する。

### 6.0a R3: ddmc_leak_coeff

v1.0 既定（`leak_stencil="9_kershaw"`）では、本カーネルはKershaw行列 \(A_{ij}\) から
リーク係数を抽出する。`leak_stencil="4"` の場合は面別 Densmore 近似を使用する。

**パス A（既定 `9_kershaw`）**：

```cpp
__global__ void ddmc_leak_coeff_kershaw(
    const double* __restrict__ stencil,    // [(n_cells+n_ghost) × 9 × G] C2出力（9点Kershaw係数、ゴーストセル含む）
                                           // R3 は indices 1-8（off-diagonal）のみ使用。index 0（center a_C）は無視
    const uint8_t* __restrict__ ddmc_candidate, // [(n_cells+n_ghost) × G] R2出力（ω,τ,P制約を満たす候補、ゴーストセル含む）
    bool m_matrix_check,                   // 既定True: 違反セルをIMCへ
    const double* __restrict__ vol,        // [(n_cells+n_ghost)] cell volume
    const double* __restrict__ face_area,  // [(n_cells+n_ghost) × n_faces]
    uint8_t* __restrict__ ddmc_mode,       // [(n_cells+n_ghost) × G] out: 最終mode（0=IMC,1=DDMC、ゴーストセル含む）
    uint8_t* __restrict__ mmatrix_violation, // [n_cells × G] out（ローカルセルのみ）
    double* __restrict__ leak_coeff_face,  // [n_cells × n_faces × G] output: Σ^leak [1/cm]（ローカルセルのみ）
    double* __restrict__ leak_total_int,   // [n_cells × G] output: Σ^out（内部面、ローカルセルのみ）
    double* __restrict__ leak_total_bdry,  // [n_cells × G] output: Σ^bnd（境界面、ローカルセルのみ）
    const int8_t* __restrict__ face_bc_type, // [n_faces_boundary] 境界面タイプ（§6.4.3 bc_code準拠）
                                             // 0=VACUUM, 1=REFLECT, 2=MARSHAK, 3=AXIS
                                             // 構造格子: face_bc_type[4]（R_left,R_right,Z_bottom,Z_top）
                                             // 1D_SPH: face_bc_type[2]（inner,outer）
                                             // DDMCリーク係数の観点では AXIS(3)=REFLECT(1) と同一処理（leak=0）。
                                             // PARTITION 面は n_ghost 領域で処理し、face_bc_type には含まない
    int n_cells, int n_ghost, int n_neighbors, int n_faces, int G, // n_neighbors=8 (2D RZ), =2 (1D SPH)
    int nr, int nz                   // 格子次元（境界面判定: c_row=c/nz, c_col=c%nz に必要）
);  // block=256, 1 thread = 1 cell, loops over neighbors × groups
```

- **block**: 256, **grid**: `((n_cells+n_ghost)+255)/256`
- **MPI注意**: R3b が隣接セルの ddmc_mode を参照するため、パスAもパスBと同様にゴーストセルを含めて処理する。C2 Kershaw カーネルもゴーストセル含みで事前呼出しが必要
- **ゴーストセルガード**: `c >= n_cells` の場合は `ddmc_mode` のみ設定（`ddmc_candidate` をコピー）し、リーク係数（leak_coeff_face, leak_total_int, leak_total_bdry, mmatrix_violation）の書き込みはスキップ（出力配列が [n_cells] 次元のため）
- **処理**: 各セルの全neighbors×groupsをループ:
  1. 候補セルのみ処理（`ddmc_candidate[c,g]==1`）
  2. `m_matrix_check=True` の場合、`stencil` の off-diagonal（k=1-8）に正値を検出したセル×群は `mmatrix_violation=1` とし `ddmc_mode=0`（IMC）へ降格、リーク係数は0
  3. `m_matrix_check=False` の場合は安全クランプ `Σ^leak_raw = max(0, -A_{c,j,g}) / V_c`
  4. 2Dの角近傍リーク（NE/NW/SE/SW）は隣接2面へ面積重みで射影:
     **退化ガード**（NUMERICS §7.3.3）: `A_f1 + A_f2 < ε_area`（ε_area = 1e-30 cm²）の場合は等分配 `w_f1=w_f2=0.5`。
     通常: `w_f = A_f/(A_f1 + A_f2)`、`Σ_corner -> (w_f1 Σ_corner, w_f2 Σ_corner)`（総和保存）
  5. `leak_coeff_face[c,f,g]` を生成し、`leak_total_int`（内部面和）/`leak_total_bdry`（境界面和）を `face_bc_type` に基づいて分離:
     - 面の内部/境界判定: セル(i,j)の各面が境界に接するかをi,j,nr,nzから算術判定し、境界面は `face_bc_type[face]` でタイプを参照
     - REFLECT/AXIS 境界面: `leak_coeff_face=0`（反射、リークなし）
     - VACUUM/MARSHAK 境界面: リーク値を `leak_total_bdry` に加算後、`leak_coeff_face[c,f,g]=0` に設定
       （R9 CDF が内部面のみ走査する前提。境界リークは Σ_leak_bdry 経由で R9 の else 分岐で処理。§6.5 R9 参照）
     - PARTITION 境界面: `leak_total_int` に加算（R9 でリーク先rank判定→emigrant、NUMERICS §12.3.3）
     - 内部面（隣接セルが存在）: `leak_total_int` に加算
  6. `mmatrix_violation==0` の候補のみ `ddmc_mode=1`、それ以外は `ddmc_mode=0`
     > **注**: ステップ6でddmc_modeを確定してから、§9 の R3b（下記）でインターフェース修正を行う。
     > 単一カーネル内でddmc_mode書き込みと隣接セルddmc_mode読み取りを同時に行うと、
     > GPUスレッドスケジューリング依存の競合が生じるため、2パス分離が必須。
- **前提**: Phase 4 の §9 R3前処理ブロックで C2 カーネルを `D_g = 1/(3σ_{R,g})`, `apply_mmatrix_repair=False` で事前呼出し（**ゴーストセル含む**：grid を `((n_cells+n_ghost)+255)/256` で起動し、stencil は `[(n_cells+n_ghost) × 9 × G]` で確保）。
  多群の場合は群ごとに C2 を G 回呼び出し、各群の出力先を `stencil + g * (n_cells+n_ghost) * 9` にオフセットする。
  **線形化レイアウト**: `stencil[(g * N + c) * 9 + k]`（N = n_cells+n_ghost、g=群、c=セル、k=ステンシル位置 0-8）。
  R3 内で `stencil[(g * N + c) * 9 + k]` として off-diagonal (k=1-8) を参照する。
  **1D_SPH**: C2 は2D_RZ専用のため、`leak_stencil="9_kershaw"` は1D_SPHでは使用不可。
  1D_SPHでは自動的に `leak_stencil="4"`（面別 Densmore 近似パスB）にフォールバックし WARNING を出力する
- **レジスタ**: ~15
- **メモリ**: stencil は C2 の出力をそのまま参照（off-diagonal k=1-8 のみ使用）。`__ldg()` 推奨

**パス B（`leak_stencil="4"`、面別近似）**：

```cpp
__global__ void ddmc_leak_coeff_face(
    const uint8_t* __restrict__ ddmc_candidate, // [(n_cells+n_ghost) × G] R2出力（ω,τ,P制約を満たす候補、ゴーストセル含む）
    uint8_t* __restrict__ ddmc_mode,       // [(n_cells+n_ghost) × G] out: 最終mode（0=IMC,1=DDMC、ゴーストセル含む）
    const double* __restrict__ vol,        // [(n_cells+n_ghost)] cell volume（H7出力 + halo_exchange）
    const double* __restrict__ face_area,  // [(n_cells+n_ghost) × n_faces] face area（同上）
    const double* __restrict__ face_sigma_R, // [(n_cells+n_ghost) × n_faces × G] face-evaluated σ_R（ゴーストセル含む）
    double* __restrict__ leak_coeff_face,  // [n_cells × n_faces × G] output: Σ^leak [1/cm]
    double* __restrict__ leak_total_int,   // [n_cells × G] output: Σ^out（内部面）
    double* __restrict__ leak_total_bdry,  // [n_cells × G] output: Σ^bnd（境界面）
    const int8_t* __restrict__ face_bc_type, // [n_faces_boundary] 境界面タイプ（パスAと共通、§6.4.3 bc_code準拠）
    int n_cells, int n_ghost, int n_faces, int G,
    int nr, int nz                   // 格子次元（境界面判定に必要。パスAと同一）
);  // block=256, 1 thread = 1 cell, loops over faces × groups
```

- **block**: 256, **grid**: `((n_cells+n_ghost)+255)/256`
- **MPI注意**: R3b が隣接セルの ddmc_mode を参照するため、R3 はゴーストセルも含めて処理する必要がある（grid に n_ghost を加算）。U9 が既にゴーストセルの opacity を計算済みのため追加通信は不要。R2 も同様にゴーストセルを含める
- **処理**: 各セルの全faces×groupsをループ:
  1. 候補セルのみ処理（`ddmc_candidate[c,g]==1`）
  2. r=0 軸に接する面: A_f → 0 のため `Σ^leak = 0` として以降スキップ（軸方向リークなし、NUMERICS §7.3.2 注記）
  3. 不透明度フロア: `σ_{R,f}^{face} = max(σ_{R,f}^{face}, σ_floor)`（σ_floor = 10^{-20}、§11.3。リーク係数のゼロ除算防止のため式評価前に適用）
  4. 面毎のリーク係数（NUMERICS §7.3.2, 面別 Densmore 近似）:
     `Σ^leak_{c,f,g} = 2 A_f / (3 V_c σ_{R,f,g}^{face} (Δx_f + 2λ_{mfp,f}))`
     ここで `Δx_f = V_c / A_f` [cm]（面垂直有効セル幅）、`λ_{mfp,f} = 1/σ_{R,f,g}^{face}` [cm]
  5. パスAと同一分類規則で `leak_total_int`/`leak_total_bdry` へ分離格納。
     VACUUM/MARSHAK 境界面: `leak_total_bdry` 加算後 `leak_coeff_face=0`（R9 CDF 整合性、パスA同様）
  6. `ddmc_mode=ddmc_candidate`（`leak_stencil="4"` ではM-matrix追加判定なし）
- **レジスタ**: ~20
- **ゴーストセルガード**: `c >= n_cells` の場合は `ddmc_mode` のみ設定（`ddmc_candidate` をコピー）し、リーク係数（leak_coeff_face, leak_total_int, leak_total_bdry）の書き込みはスキップ（出力配列が [n_cells] 次元のため）
- **メモリ**: face_area は `[(n_cells+n_ghost) × n_faces]`、face_sigma_R は `[(n_cells+n_ghost) × n_faces × G]` レイアウトで stride アクセス → `__ldg()` 推奨
- **ワープ発散**: DDMCセルのみ処理するため、IMC/DDMC境界付近のワープで発散あり。ICF問題ではDDMCセルが空間的に集中するため、大半のワープは全スレッド同一パス

### 6.0a1 helper: compute_face_sigma_R

```cpp
__global__ void compute_face_sigma_R(
    double* __restrict__ face_sigma_R,         // [(n_cells+n_ghost) × n_faces × G] out
    const double* __restrict__ Te,             // [(n_cells+n_ghost)] 電子温度 [eV]（T⁴面温度平均に使用）
    const double* __restrict__ rho,            // [(n_cells+n_ghost)] 密度 [g/cm³]（面σ_R再評価に使用）
    const double* __restrict__ kappa_R_table,  // opacity テーブル（FrozenTable2D: κ_R(ρ,T) [cm²/g]）
    int n_cells, int n_ghost, int n_faces, int G,
    int nr, int nz
);
```

- **block**: 256, **grid**: `((n_cells+n_ghost)+255)/256`
- **処理**: 各セルが `faces × groups` を処理。NUMERICS §7.3.2 準拠の T⁴ 面温度平均を使用:
  1. 面温度: `T_{face} = ((Te[owner]^4 + Te[neighbor]^4) / 2)^{1/4}`（NUMERICS §7.3.2 Eq. T_{n,j+1/2}）
  2. 面 σ_R: `face_sigma_R[c,f,g] = rho[owner] × κ_R(rho[owner], T_{face}, g)`（owner 側密度で評価、NUMERICS §7.3.2 σ^-_{R,j+1/2}）
  3. 境界面は owner 側セルの Te/rho をそのまま使用（neighbor 不在）
  - **注**: R3b のリーク係数公式は V_c/A_f ベースの多次元一般化形式を使用するため、σ_R^+ と σ_R^- の分離は不要。owner 側密度での面温度評価で十分（NUMERICS §7.3.2 の 1D 左右分離形式は §7.3.5 ddmc_interface_correct で補正）

### 6.0a2 R3b: ddmc_interface_correct（インターフェース修正パス）

R3 で `ddmc_mode` が確定した後に起動する**第2パス**。
DDMC-IMCインターフェースセルのリーク不透明度を修正する（NUMERICS §7.3.5 必須）。
R3 内で ddmc_mode の書き込みと隣接セル ddmc_mode の読み取りを同時に行うと
GPU スレッドスケジューリング依存の競合が生じるため、別カーネルとして分離する。

```cpp
__global__ void ddmc_interface_correct(
    const uint8_t* __restrict__ ddmc_mode,        // [(n_cells+n_ghost) × G] R3確定済み（read-only、ゴーストセル含む）
    double* __restrict__ leak_coeff_face,          // [n_cells × n_faces × G] in/out: R3出力を修正
    double* __restrict__ leak_total_int,           // [n_cells × G] in/out: 修正分の差分を反映
    const double* __restrict__ Te,                 // [(n_cells+n_ghost)] 電子温度 [eV]（隣接 ghost セルの Te を面温度計算に参照）
    const double* __restrict__ rho,                // [(n_cells+n_ghost)] 密度 [g/cm³]（同上）
    const double* __restrict__ vol,                // [(n_cells+n_ghost)] セル体積（halo_exchange 済み）
    const double* __restrict__ face_area,          // [(n_cells+n_ghost) × n_faces] 面面積（halo_exchange 済み）
    const void* opacity_table,                     // κ_R(ρ, T) テーブル（面温度での σ_R 再評価に使用）
    int n_cells, int n_faces, int G,
    int nr, int nz
);  // block=256, 1 thread = 1 cell, loops over faces × groups
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: DDMCセル（`ddmc_mode[c,g]==1`）の全面をループ:
  0. **境界面ガード**: 面(c,k)が物理境界面（i,j,nr,nzから算術判定）の場合は skip（interface修正は内部DDMC-IMC境界のみに適用）。n_cells=1 の場合は全面が境界のため R3b は実質空操作。MPI パーティション面は ghost セル側に neighbor が存在するため内部面として処理する
  1. 隣接セルの `ddmc_mode[neighbor,g]` を読み取り（R3 で確定済みのため安全。neighbor = get_neighbor(c, k, nr, nz) は内部面ガード通過後のため常に有効範囲 [0, n_cells+n_ghost)）
  2. `ddmc_mode[neighbor,g]==0`（IMC側面）の場合、リーク係数を修正版 σ_{L,1} に置換:
     `σ_{L,1} = (1/Δx_1) × 2/(3σ_{R,1}Δx_1 + 6λ)`（λ≈0.7104 = Milne外挿距離）
     σ_{R,1} は DDMCセル密度 ρ_1 と **面温度** T_{n,face} で評価（§7.3.5, §7.3.4 の T^4 平均）
  3. `leak_total_int` を修正量の差分で更新（旧 Σ^leak を引き、新 σ_{L,1} を足す）
  4. 反対側（内部側）のσ_R は標準の §7.3.4 Eq.21 をそのまま使用
     > **重要**: 標準のσ_L（Eq.20）を境界セルに使うと P(μ) 導出と不整合になり
     > インターフェースでのエネルギー保存が崩れる（Densmore 2007 Eqs.32-33 参照）
- **前提**: R3 が完了し `ddmc_mode` が確定していること（§9 シーケンスで R3 → R3b の順に起動）
- **レジスタ**: ~12
- **メモリ**: ddmc_mode は read-only（R3 で write 完了後）。leak_coeff_face, leak_total_int は read-modify-write

---

### 6.0b R4: compute_source_energy

```cpp
__global__ void compute_source_energy(
    const double* __restrict__ sigma_a_eff,  // [n_cells × G] effective absorption
    const double* __restrict__ Te,           // [n_cells] electron temperature [eV]
    const double* __restrict__ vol,          // [n_cells] cell volume
    const double* __restrict__ E_ref,        // [n_cells × G] optional difference reference density; nullptr for legacy
    const PlanckTable* __restrict__ planck_table, // PlanckTable 構造体（ARCHITECTURE §4.5: 200点テーブル）。カーネル内で b_g = planck_fraction(g, Te[c], planck_table) を評価
    double* __restrict__ source_E,           // [n_cells × G] output: transported source energy; signed when E_ref!=nullptr
    double* __restrict__ rad_emit_E,         // [n_cells × G] output: physical emission diagnostic
    double* __restrict__ source_total,       // output: sum(abs(source_E)) (atomicAdd)
    double dt, int n_cells, int G
);  // block=256, 1 thread = 1 cell
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 各セルで全G群をループ:
  1. 放射率密度 S^{emit}_{c,g} = c·σ_{a,eff,c,g}·a_{eV}·T_e⁴·b_g [erg/cm³/s]（NUMERICS §6.2）
  2. `rad_emit_E[c,g] = S^{emit}_{c,g} × V_c × Δt` [erg]（物理 emission diagnostic）
  3. legacy: `source_E[c,g] = rad_emit_E[c,g]`
  4. difference PR4: `source_E[c,g] = c σ_{a,eff,c,g} (B_{c,g}-E_ref[c,g]) V_c Δt`
  5. `source_total += abs(source_E[c,g])`（atomicAdd(double*) — 全スレッドから同一アドレスへの競合あり）
- **レジスタ**: ~15
- **Atomic競合**: source_total は単一アドレスへの全スレッド atomicAdd。競合が多いが 1 回/スレッド で頻度は低い。精度保証: double atomicAdd は加算順序非決定的だが、source_total は N_p 配分の重みに使用するため、相対精度 ~1e-10 で十分

### 6.0b1 R4b: preseed_reference_absorption

```cpp
__global__ void preseed_reference_absorption(
    double* __restrict__ rad_dep,
    const double* __restrict__ sigma_a_eff,
    const double* __restrict__ E_ref,
    const double* __restrict__ vol,
    int n_cells, int G, double dt
);
```

- **block**: 256, **grid**: `(n_cells*G+255)/256`
- **処理**: difference PR4 の LTE nonlinear source residualization が有効な場合のみ、
  Phase 4 init の tally zero 後に
  `rad_dep[c,g] += c_light * sigma_a_eff[c,g] * E_ref[c,g] * vol[c] * dt`
  を加算する。reference face transport は R4c の別bufferで扱い、`rad_dep` には加えない。

### 6.0b2 R4c: reference_face_transport_1d

```cpp
__global__ void reference_face_transport_1d(
    double* U_ref_end,             // [n_cells × G] deterministic reference reservoir [erg]
    double* ref_face_delta_U,      // [n_cells × G] face divergence [erg]
    double* E_ref_avg,             // [n_cells × G] time-average reference density
    const double* E_ref_start,     // [n_cells × G] start reference density
    const double* sigma_R,         // [n_cells × G] Rosseland opacity proxy [1/cm]
    const double* node_r,          // [n_cells + 1] 1D_SPH face radii [cm]
    const double* vol,             // [n_cells] cell volume [cm^3]
    int n_cells, int n_groups, double dt,
    int bc_inner, int bc_outer);
```

- **適用対象**: `Radiation.imc.difference.enabled=true`、LTE nonlinear source path、
  `difference.face_transport=true`、1D_SPH のみ。
- **処理**: face \(f\) の向きを小さい \(r\) から大きい \(r\) へ取り、
  `Q_ref[f,g] = A_f * c/4 * psi(tau_f,g) * (E_L - E_R) * dt` を使う。
  `psi(tau)=tanh(3*tau/4)/(3*tau/4)`、`psi(0)=1`。内部 face の
  `tau_f,g` は harmonic pair の `sigma_R` と隣接 cell-center 間距離から作る。
  cell divergence は `ref_face_delta_U[c,g] = Q_left - Q_right`。
- **出力**:
  `U_ref_end[c,g] = E_ref_start[c,g] * vol[c] + ref_face_delta_U[c,g]`、
  `E_ref_avg[c,g] = E_ref_start[c,g] + 0.5 * ref_face_delta_U[c,g] / vol[c]`。
  `U_ref_end` は次 step の previous-reference reservoir に保存し、
  `E_ref_avg` は PR7 の `rad_E` reconstruction に使う。
- **境界 accounting**: reflect face は zero flux。Vacuum/Marshak face から外向きに出る
  deterministic reference leakage は `IMC::escaped_energy_total()` に加算し、
  driver の `E_rad_escaped` accounting に入る。PR6 の 1D_SPH path は既存 Marshak
  incident source 粒子を保持するため外部 reference density は 0 とする。
- **不変条件**: reference face transport は `rad_dep` に加算しない。

### 6.0b3 PR5 census residualization

- **処理**: difference PR5 の LTE nonlinear path が有効な場合のみ、tally zero と
  source emission の前に host orchestration で census 粒子を cell-group bin ごとに
  residual 化する。
  - 旧物理エネルギー: `U_phys_old[c,g] = U_ref_old[c,g] + Σ sign[p]*energy[p]`
  - 新 residual target: `R_new[c,g] = U_phys_old[c,g] - E_ref_start[c,g]*vol[c]`
  - existing nonempty bin: conditioned な `R_old` では magnitude scaling と sign flip で
    `Σ sign*energy = R_new` に合わせる。
  - ill-conditioned nonempty bin: 既存粒子1個を template として `abs(R_new)` と
    `sign(R_new)` に置換し、余分な粒子を kill する。
  - empty nonzero bin: residual 粒子を1個作る。位置と方向は thermal source の
    1D_SPH cell-volume 一様 / isotropic sampling kernel を再利用する。
- **global_id**: empty-bin residual 粒子は step local-id `[2^38,2^39)` を予約する。
  通常 source emission の低 local-id 範囲および diffusion の `[2^39,2^40)` と重ならない。
- **出力**: `U_ref_start = E_ref_start * V` を previous-reference reservoir として保存し、
  `face_transport=false` ではそのまま次 step に使う。`face_transport=true` では
  R4c の `U_ref_end` で reservoir を置き換える。`IMC::census_energy()` は
  reservoir と signed residual census の和を返す。

### 6.0c R5: source_particle_count

```cpp
__global__ void source_particle_count(
    const double* __restrict__ source_E,     // [n_cells × G] source energy; signed residual allowed
    double source_total,                     // total abs(source_E)
    int N_p_total,                           // total particles to generate
    int* __restrict__ count,                 // [n_cells × G] output: particle count per cell per group
    int* __restrict__ offset,                // [n_cells × G + 1] output: prefix sum (CUB ExclusiveSum、末尾要素=総粒子数)
    int n_cells, int G
);  // block=256, 1 thread = 1 cell×group
```

- **block**: 256, **grid**: `(n_cells*G+255)/256`
- **処理**: `source_total == 0` の場合は `count[c*G+g] = 0`（全セルで放射源なし → 粒子生成なし）。それ以外:
  - `abs(source_E[c*G+g]) > 0` の場合: `count[c*G+g] = max(1, round(N_p_total × abs(source_E[c*G+g]) / source_total))`（NUMERICS §6.2 粒子数配分）
  - `source_E[c*G+g] == 0` の場合: `count[c*G+g] = 0`（ゼロソースビンに粒子を生成しない）
  - `max(1,...)` により nonzero source_E の全 (cell,group) bin で最低1粒子を保証（NUMERICS §6.2 規約）
  - 実際の生成総数 Σ count は N_p_total と正確には一致しないが、`E_p = source_E / N_p` が自動補償するため物理バイアスなし（NUMERICS §6.2 注記）
- 後段で CUB `DeviceScan::ExclusiveSum` を適用し offset を生成。`n_new_particles = offset[n_cells×G]`（prefix sum末尾値、実際の生成総数を反映）
- **レジスタ**: ~10
- **メモリ**: coalesced read/write（セル×群の1D配列）

### 6.0d R7: composite_sort_and_partition（合成キーソート + fused gather）

R7 は従来の R7（セルソート）+ R11（dead compaction）+ R14（モード分離）を
**単一パイプラインに融合**した操作である（§0.5参照）。3つのサブステップで構成される。

#### サブステップ 1: 合成キー生成（カスタムカーネル）

```cpp
__global__ void build_composite_key(
    const int32_t* __restrict__ cell_id,     // [N_total] 粒子のセルID
    const uint16_t* __restrict__ group_id,   // [N_total] 粒子の群ID
    uint8_t* __restrict__ mode,              // [N_total] in/out: 0=IMC, 1=DDMC
    const uint8_t* __restrict__ ddmc_mode,   // [(n_cells+n_ghost) × G] R3で確定したセルモード（確保サイズはゴースト含む。alive粒子のcell_id<n_cellsのみ参照するためOOBなし）
    const uint8_t* __restrict__ alive,       // [N_total] 0=dead, 1=alive
    double* __restrict__ pos_r,              // [N_total] in/out: IMC→DDMC遷移時にNaN化
    double* __restrict__ pos_z,              // [N_total] in/out: 同上
    double* __restrict__ dir_r,              // [N_total] in/out: 同上
    double* __restrict__ dir_z,              // [N_total] in/out: 同上
    double* __restrict__ dir_phi,            // [N_total] in/out: 同上
    uint32_t* __restrict__ comp_key,         // [N_total] output: 合成キー
    int* __restrict__ perm,                  // [N_total] output: 初期順列 [0..N-1]
    int* __restrict__ count_imc,             // [1] output: IMC alive count (atomicAdd)
    int* __restrict__ count_ddmc,            // [1] output: DDMC alive count (atomicAdd)
    int N_total, int G, int n_cells
);  // block=256, grid=(N_total+255)/256
```

- **処理**（1 thread = 1 particle）：
  ```
  if alive[tid] != 1 || cell_id[tid] < 0 || cell_id[tid] >= n_cells || group_id[tid] >= G:
      comp_key[tid] = 0xFFFFFFFF    // dead(0) / OVERFLOW(2) / emigrant残留（cell_id<0）/ 不正セルID / 不正群ID → 末尾にソート
      // alive=2（OVERFLOW）: P5 でバッファ超過時に設定（NUMERICS §12.3.1）
      // cell_id < 0 の alive 粒子は P5 で alive=2 設定済みのはずだが、
      // 防御的に cell_id<0 もガードし ddmc_mode の OOB read を防止する
      // cell_id >= n_cells: R8/R9 のバグで不正セルIDが書かれた場合の OOB 防止（防御的ガード）
      // group_id >= G: R8 scatter等のバグで不正群IDが書かれた場合の OOB 防止（防御的ガード）
  else:
      // セルモードを粒子へ同期（census粒子を含む）
      mode_eff = ddmc_mode[cell_id[tid] * G + group_id[tid]]
      old_mode = mode[tid]
      mode[tid] = mode_eff
      if mode_eff == 0:              // IMC
          comp_key[tid] = (0u << 30) | cell_id[tid]  // [0x00000000, 0x3FFFFFFF]
          atomicAdd(count_imc, 1)
      else:                          // DDMC
          comp_key[tid] = (1u << 30) | cell_id[tid]  // [0x40000000, 0x7FFFFFFF]
          atomicAdd(count_ddmc, 1)
          // **IMC→DDMC 遷移時: pos/dir を NaN sentinel に設定**
          // 条件: old_mode==IMC(0) かつ mode_eff==DDMC(1) → ステップ境界で IMC→DDMC に遷移した粒子
          // NaN 化が必要な理由:
          //   (1) U7 は mode==DDMC でスキップするが、NaN は防御的不変条件（mode 破損時に NaN が point-in-cell を安全に失敗させる）
          //   (2) 将来のステップで DDMC→IMC に再遷移した場合、R7b が isnan(pos_r) で検出
          //   (3) NaN 化しないと stale 位置が残り R7b が見逃す
          // 既に DDMC だった census 粒子は前ステップの R8 変換時または R6 生成時に
          // NaN 化済みなので二重書き込みは冗長だが無害
          if old_mode == 0:  // IMC→DDMC 遷移
              pos_r[tid] = NaN; pos_z[tid] = NaN
              dir_r[tid] = NaN; dir_z[tid] = NaN; dir_phi[tid] = NaN
  perm[tid] = tid
  ```
- **レジスタ**: ~8
- **atomicAdd**: `count_imc`/`count_ddmc` は CUB BlockReduce + 1回の global atomic で実装可能（warp-level reduction推奨）

#### サブステップ 2: CUB RadixSort

```cpp
cub::DeviceRadixSort::SortPairs(
    scratch, scratch_bytes,
    comp_key_in, comp_key_out,     // uint32_t[N_total]
    perm_in, perm_out,             // int[N_total]
    N_total,
    0, 32                          // begin_bit=0, end_bit=32 (CUB end_bit は排他的: [0,32) = 全32ビット、bit31=dead flag を含む)
);
```

- `end_bit=32`（CUB の end_bit は排他的 [begin_bit, end_bit) なので、bit31=dead flag を含むには end_bit=32 が必要）
- メッシュサイズに応じて `end_bit` を動的に縮小可能：500×250メッシュでは cell_id ≤ 125,000 → 17ビット → `end_bit = 32` で mode+dead flag 含む全32ビットをソート。CUB は内部で上位ゼロビットをスキップするため性能影響は軽微
- 補助メモリ: ~24 × N_total bytes（CUB内部バッファ）

#### サブステップ 3: Fused SoA Gather（カスタムカーネル）

```cpp
__global__ void fused_soa_gather(
    const PhotonPool src,           // 入力 SoA（ソート前）
    PhotonPool dst,                 // 出力 SoA（ソート後）
    const int* __restrict__ perm,   // [N_alive] ソート済み順列
    int N_alive,                    // alive粒子数 = count_imc + count_ddmc
    double dt,                      // 現ステップΔt（census re-arm用、NUMERICS §6.3.1）
    bool enable_rearm               // true=census re-arm実行（第1R7）、false=スキップ（第2R7）
);  // block=256, grid=(N_alive+255)/256
// 前提: src と dst は非重複（ダブルバッファ保証。ARCHITECTURE §5.3 active_pool_index 参照）
```

- **処理**（1 thread = 1 alive particle）：
  ```
  int s = perm[tid];  // 元の粒子インデックス（1回ロード、15回再利用）

  // 15 scattered loads + alive read（gather は全16フィールドを dst に書き込む）
  double r  = src.pos_r[s];
  double z  = src.pos_z[s];
  double dr = src.dir_r[s];
  double dz = src.dir_z[s];
  double dp = src.dir_phi[s];
  double e  = src.energy[s];
  double w  = src.weight[s];
  double t  = src.time_remain[s];
  double b  = src.birth_energy[s];
  int8_t sgn = src.sign[s];
  uint64_t gid = src.global_id[s];
  uint32_t rng = src.rng_counter[s];
  int32_t  cid = src.cell_id[s];
  uint16_t grp = src.group_id[s];
  uint8_t  mod = src.mode[s];

  // Census粒子の time_remain 再装填（NUMERICS §6.3.1 re-arm）:
  // gather と同時に実行し、追加パスを回避する。
  // **enable_rearm=true**（第1R7）の場合のみ実行。第2R7（P6後）では
  // enable_rearm=false とし、当該ステップのcensus粒子にstale dtを設定することを防止する。
  if (enable_rearm && t <= 0.0 && src.alive[s] == 1) t = dt;  // dt はカーネル引数

  // 16 coalesced stores（全 SoA フィールド。alive 含む）
  dst.pos_r[tid] = r;
  dst.pos_z[tid] = z;
  ... // 残り13フィールド同様
  dst.alive[tid] = 1;  // **必須**: gather 対象は全て alive（composite key bit31=0）。
                         // alive を書かないと dst バッファに前ステップの stale 値（0）が残り、
                         // R9 の `while (alive && ...)` や R12 が粒子をスキップして
                         // サイレントデータ損失が発生する。
  ```
- **census re-arm**: `time_remain ≤ 0` かつ `alive == 1` の粒子（前ステップの census）に `time_remain = dt` を設定（NUMERICS §6.3.1 準拠）。R8/R9 にも冗長な re-arm があるが、fused_soa_gather が正規の実施箇所
- **レジスタ**: ~32（16フィールド保持 + perm + tid）
- **block**: 256, **grid**: `(N_alive+255)/256`
- **メモリ帯域**: N_alive × 93B（read、scattered） + N_alive × 93B（write、coalesced）= 186B/particle
  - 1000万粒子: ~1.86 GB、A100 2TB/s → 理論下限 0.93ms、scatter read penalty 2-3× → 実測 ~1.5ms
- **ILP 効果**: 15本の独立 load（alive以外の可変フィールド）が同時発行されL2ミスレイテンシを隠蔽。CUB の逐次15回 gather（各回が独立カーネル起動 + 全粒子走査）に対し、カーネル起動14回分（~70μs）+ メモリ往復14回分を節約

#### R7 パイプライン全体のコスト

| サブステップ | 100万粒子 | 1000万粒子 | メモリ |
|------------|----------|-----------|--------|
| 合成キー生成 | ~0.1 ms | ~0.3 ms | 8B/particle (key+perm) |
| CUB RadixSort | ~1.0 ms | ~3.0 ms | ~24 × N bytes (CUB temp) |
| Fused SoA Gather | ~1.5 ms | ~6.0 ms | 93B × N (double buffer) |
| **合計** | **~2.6 ms** | **~9.3 ms** | |

**従来比較**（100万粒子）：
| 従来 | コスト | → | Composite Key | コスト |
|------|--------|---|--------------|--------|
| R7 sort + 15可変配列×gather | ~3.5 ms | → | (上記に含む) | — |
| R11 compact + 15可変配列×gather | ~2.0 ms | → | (R7に吸収) | 0 |
| R14 partition + 15可変配列×gather | ~1.5 ms | → | (R7に吸収) | 0 |
| **合計** | **~7.0 ms** | → | **合計** | **~2.6 ms (63% 削減)** |

### 6.0d1 R7b: ddmc_to_imc_resample（DDMC→IMC モード遷移位置再サンプル）

R7 の `build_composite_key` は `ddmc_mode[cell,g]` に基づいて粒子の mode を上書きする（§6.0d）。
前ステップで DDMC だった census 粒子のセルが IMC に遷移した場合（不透明度変化で τ < τ_DDMC）、
粒子は `mode=IMC` に再分類されるが、位置・方向は **NaN sentinel のまま** である。
NaN 位置の粒子を R8（幾何光学追跡）に渡すと、距離計算が NaN に伝播し致命的な破綻が生じる。
R7b はこの遷移粒子を検出し、セル内一様位置 + 等方方向を再サンプルする。

```cpp
__global__ void ddmc_to_imc_resample(
    double* __restrict__ pos_r,              // [n_imc] in/out: NaN → セル内一様位置
    double* __restrict__ pos_z,              // [n_imc] in/out
    double* __restrict__ dir_r,              // [n_imc] in/out: NaN → 等方方向
    double* __restrict__ dir_z,              // [n_imc] in/out
    double* __restrict__ dir_phi,            // [n_imc] in/out
    uint32_t* __restrict__ rng_counter,      // [n_imc] in/out: RNG 消費カウンタ更新
    const uint64_t* __restrict__ global_id,  // [n_imc] in: RNG key 導出用
    const int32_t* __restrict__ cell_id,     // [n_imc] in: 所属セル（R7 ソート済み）
    const double* __restrict__ x_r,          // [n_nodes] メッシュ節点 R 座標
    const double* __restrict__ x_z,          // [n_nodes] メッシュ節点 Z 座標
    uint64_t user_seed,                      // Main.seed
    uint64_t step,                           // 現在のステップ番号（RNG subsequence）
    int n_imc, int nr, int nz
);
```

- **block**: 128, **grid**: `(n_imc+127)/128`
- **処理**:
  ```
  if (tid >= n_imc) return;
  if (!isnan(pos_r[tid])) return;    // NaN でなければ遷移粒子ではない → スキップ
  // --- 遷移粒子検出: pos_r が NaN sentinel ---
  // RNG 復元
  curandStatePhilox4_32_10_t rng;
  curand_init(global_id[tid] ^ user_seed, step, rng_counter[tid], &rng);
  // セル内一様位置サンプル（R6 §6.2 と同一ロジック）
  int c = cell_id[tid];
  // 1D_SPH: r = (r_lo³ + ξ(r_hi³-r_lo³))^{1/3}
  // 2D_RZ: 双線形写像 + R重み棄却法
  pos_r[tid] = sampled_r;
  pos_z[tid] = sampled_z;
  // 等方方向サンプル: μ = 2ξ-1, φ = 2πξ → (Ω_r, Ω_z, Ω_φ)
  dir_r[tid] = ...;
  dir_z[tid] = ...;
  dir_phi[tid] = ...;
  rng_counter[tid] += N_draws;  // 消費した乱数の数を記録
  ```
- **コスト**: 遷移粒子のみ処理（大半は `isnan` チェックで早期リターン）。
  τ ≈ τ_DDMC の境界領域でのみ遷移が発生するため、典型的には全 IMC 粒子の 1% 未満。
  ワープ発散は無視可能。カーネル起動オーバーヘッド ~5μs が支配的
- **レジスタ**: ~20（RNG state + セル頂点座標 + サンプリング変数）
- **メモリ**: 遷移粒子のみ書き込み。非遷移粒子は `pos_r` の読み込み（8B）のみ
- **呼び出しタイミング**: Phase 4、R7 fused_soa_gather 直後、R8 の前（§9 参照）

### 6.0e R10: tally_finalize

```cpp
__global__ void tally_finalize(
    const double* __restrict__ rad_E_tally,    // [n_cells × G] raw track-length estimator [erg·cm]（ARCHITECTURE §5.2 State.rad_E_tally と同一バッファ）
    const double* __restrict__ vol,            // [n_cells] cell volume
    double dt,                                 // timestep [s]
    const double* __restrict__ E_ref_avg,      // [n_cells × G] optional difference reference average; nullptr for legacy
    double* __restrict__ residual_E,           // [n_cells × G] optional signed residual density diagnostic
    double* __restrict__ rad_E,                // [n_cells × G] output: energy density [erg/cm³]
    int n_cells, int G
);  // block=256, 1 thread = 1 cell×group.
```

- **block**: 256, **grid**: `(n_cells*G+255)/256`
- **処理**: 各セル×群で（NUMERICS §10.2）:
  1. **退化セルガード**: `vol[c] < 1e-30` の場合 `rad_E[c,g] = 0`（ゼロ除算防止。退化セルに粒子が存在する可能性は極めて低いが、ALE rezone 直後に体積がほぼゼロのセルが生じうる）
  2. legacy: `rad_E[c,g] = rad_E_tally[c,g] / (vol[c] × c_light × dt)`（track-length推定量の正規化、NUMERICS §10.1）
  3. difference: `residual = signed_rad_E_tally[c,g] / (vol[c] × c_light × dt)` を作り、`rad_E[c,g] = E_ref_avg[c,g] + residual` とする。clamp はこの final physical `rad_E` のみに適用し、signed residual 単体は clamp しない。
- **注**: `rad_dep` は R10 の入出力ではない。R8/R9/R12 が `rad_dep[n_cells×G]` に直接 `atomicAdd` し、
  U1 が同配列を消費する。R10 は track-length 推定量 `rad_E_tally` の正規化のみを行う。
  Phase 4 init で `rad_dep[n_cells×G]=0` をゼロ初期化し、difference PR4 が有効な
  LTE nonlinear path では R4b で deterministic reference absorption を preseed する。
  その後 R8→R9→R12→U1 の一貫したパイプラインを保証する
- **レジスタ**: ~10
- **メモリ**: coalesced read/write（セル×群の1D配列、連続アクセス）

### 6.0f ~~R11: photon_compaction~~ → R7 に吸収

> **v1.0設計変更**: R11（dead粒子compaction）は R7 Composite Key Sort に吸収された（§0.5）。
> 合成キーの bit 31 = dead flag により、ソート後に dead 粒子が自動的に末尾に配置され、
> `n_alive = count_imc + count_ddmc` で切り捨てることで compaction と等価の効果を得る。
> 独立の CUB `DeviceSelect::Flagged` 呼び出しと15可変配列（16 SoA中）の個別 gather は不要となった。
> **例外**: `particle_sort_by_cell=False` フォールバック時は CUB `DeviceSelect::Flagged` を使用する
> （NUMERICS §6.5 フォールバックパス参照。mode sync + NaN化 + R7b resample も必須）。

### 6.0g R12: russian_roulette

```cpp
__global__ void russian_roulette(
    double* __restrict__ energy,           // [N_p] particle energy
    uint8_t* __restrict__ alive,            // [N_p] alive flag
    uint32_t* __restrict__ rng_counter,    // [N_p] RNG draw index
    const uint64_t* __restrict__ global_id, // [N_p] for RNG key
    const int32_t* __restrict__ cell_id,   // [N_p] cell index（消滅粒子のエネルギー沈着先）
    const uint16_t* __restrict__ group_id, // [N_p] group index（消滅粒子のエネルギー沈着先）
    const uint8_t* __restrict__ mode,      // [N_p] 粒子モード（IMC/DDMC判定用。if (mode==IMC && time_remain>0) return）
    const double* __restrict__ time_remain, // [N_p] 残り時間（census判定用）
    double* __restrict__ rad_dep,          // [n_cells × G] radiation energy deposition [erg]
    double w_cutoff,                       // weight cutoff fraction (default 1e-10, SPECIFICATION §6.4.5)
    double p_survival,                     // roulette survival probability (default 0.1, SPECIFICATION §6.4.5)
    double E_avg,                          // average source energy this step
    double* __restrict__ E_numerical_loss, // [1] cell_id >= n_cells の粒子エネルギーを数値損失に計上
    DeviceErrorFlags* error_flags,        // invalid_cell_id フラグ（cell_id >= n_cells 時に設定、§0.6 準拠）
    int N_p, int n_cells, int n_groups, uint64_t step, uint64_t user_seed
);  // block=128, 1 thread = 1 particle. E < w_cutoff * E_avg → roulette
```

- **block**: 128, **grid**: `(N_p+127)/128`
- **適用対象**: census粒子（`time_remain == 0`）および DDMC粒子（`mode == DDMC`）。IMC輸送中の粒子は R8 内のインライン roulette（§6.4 Phase 2 step 6）で処理済みのため、R12 では **IMC active 粒子をスキップ** する（二重適用防止）。具体的には `if (mode == IMC && time_remain > 0) return;` で早期リターン
- **emigrantガード**: R12 は R8/R9 後 P5 前に実行されるため、emigrant粒子（`cell_id < 0`）がプールに残存しうる。`cell_id < 0` の粒子は `return;` でスキップする（emigrant は MPI 転送待ちであり roulette 対象外。`rad_dep[cell_id<0]` への OOB atomicAdd を防止）。`cell_id >= n_cells` の場合は R8/R9 のバグによる不正セルIDであるため、`alive = 0` で安全に殺し `error_flags->invalid_cell_id = 1` を設定する（エネルギーは `E_numerical_loss` に沈着）
- **処理**: 対象粒子で（NUMERICS §6.3.4 Russian roulette）:
  1. `E < w_cutoff × E_avg` → ルーレット判定
  2. curand_init(seed=global_id[p] ^ user_seed, subsequence=step, offset=rng_counter[p])（NUMERICS §12.7.1 準拠）
  3. ξ < p_survival → `energy /= p_survival`、それ以外 → `atomicAdd(&rad_dep[cell*G+group], energy); alive = 0`（NUMERICS §6.3.4: 消滅粒子のエネルギーをrad_depに沈着し保存）
- **レジスタ**: ~15
- **ワープ発散**: ルーレット対象の粒子は全体の少数（低エネルギー粒子のみ）。大半のスレッドは条件不成立で early return → 発散は限定的

### 6.0h R13: marshak_source

```cpp
__global__ void marshak_source(
    const double* __restrict__ face_area,     // [n_boundary_faces] boundary face areas [cm²]（H7出力。1D_SPH: 4πr²、2D_RZ: 面長×2πr_face）
    const double* __restrict__ T_boundary,    // [n_boundary_faces] 各Marshak面の放射温度 T_{r,f} [eV]（面ごとに独立）
    const double* __restrict__ planck_b,      // [n_boundary_faces × G] 各面の Planck 分率 b_g(T_{r,f})（面ごとに異なるスペクトル）
    const int32_t* __restrict__ boundary_cell_id, // [n_boundary_faces] boundary face → cell mapping
    const int* __restrict__ face_offset,      // [n_boundary_faces + 1] prefix sum（面別粒子数配分、ホスト事前計算）
    const double* __restrict__ x_r,           // [n_nodes] 節点R座標（面上位置サンプルに使用。2D_RZ: 面端点特定）
    const double* __restrict__ x_z,           // [n_nodes] 節点Z座標
    const double* __restrict__ face_normal_r, // [n_boundary_faces] 境界面法線R成分（半球方向サンプルの基準）
    const double* __restrict__ face_normal_z, // [n_boundary_faces] 境界面法線Z成分
    double* __restrict__ pos_r, double* __restrict__ pos_z, // output: new particle positions
    double* __restrict__ dir_r, double* __restrict__ dir_z, double* __restrict__ dir_phi, // output: directions
    double* __restrict__ energy,              // output: particle energies
    double* __restrict__ weight,              // output: statistical weights (= 1.0)
    double* __restrict__ time_remain,         // output: remaining time = dt × (1 - ξ)（ステップ内一様サンプル、NUMERICS §8.2 step 7）
    double* __restrict__ birth_energy,        // output: birth energy (= energy, for Russian roulette §6.3.4)
    int8_t* __restrict__ sign,                // output: particle sign (= +1 for legacy source paths)
    uint32_t* __restrict__ rng_counter,       // output: RNG counters
    uint64_t* __restrict__ global_id,         // output: global IDs
    int32_t* __restrict__ cell_id,            // output: cell index (from boundary_cell_id)
    uint16_t* __restrict__ group_id,          // output: group indices
    uint8_t* __restrict__ mode,                // output: transport mode (= 0, IMC)
    uint8_t* __restrict__ alive,               // output: alive flag (= 1)
    int N_marshak,                            // total Marshak particles to generate
    int n_boundary_faces,                     // number of boundary faces
    int n_groups,                             // number of energy groups G（planck_b 群ループに使用）
    int nr, int nz,                           // 格子次元（face端点→node index算出に使用）
    double dt, uint64_t step, uint64_t user_seed, uint64_t id_offset
);  // block=128, 1 thread = 1 particle
```

- **block**: 128, **grid**: `(N_marshak+127)/128`
- **処理**: 各粒子で（NUMERICS §8.2 Marshak BC）:
  0. **面別粒子配分**（ホスト側で事前計算、NUMERICS §8.2 step 2）:
     N_f = round(N_total × A_f / ΣA)。最低 1 粒子/面を保証（端数調整は最大面積の面に加減）。
     prefix sum → face_offset[n_boundary_faces+1] を構築し、各スレッドが担当面を特定
  1. スレッドIDから担当面 f を prefix sum から特定（tid ∈ [face_offset[f], face_offset[f+1])）
  2. 面 f 上の位置をランダムサンプル（2D_RZ: R重み付き棄却法、1D_SPH: 等方球面）
  3. コサイン重み方向サンプル（半球内向き）: P(μ) = 2μ。`sample_isotropic_half_space(-face_normal)` で呼び出す（`face_normal` は**外向き**法線のため符号反転して内向き半空間を指定。R8 IMC→DDMC 棄却のサンプリングと同一規約）
  4. エネルギー = (a_eV × c / 4) × T_{r,f}⁴ × A_f × dt / N_f  (NUMERICS §8.2)。
     **面ごとの T_{r,f}** を使用（全面同一温度とは限らない）。
     群は面fの b_g(T_{r,f})（planck_b[f*G + g]）に比例してサンプル。
     per-particle エネルギーには b_g を乗じない（群選択の重みのみ、全群合計で E_Marshak,f を保存）
  5. Philox RNG初期化: curand_init(seed=global_id ^ user_seed, subsequence=step_number, offset=0)（新規粒子のため offset=0。R6 と同一。NUMERICS §12.7.1 準拠）。カーネル終了時に rng_counter を消費済み draw 数に更新して出力
- **レジスタ**: ~25
- **メモリ**: 出力のみ（新規粒子をSoAに書き込み）。書き込みは particle_offset ベースで coalesced

### 6.1 R1: compute_fleck_factor

```cpp
__global__ void compute_fleck_factor(
    double* __restrict__ f_fleck,        // [n_cells] out
    double* __restrict__ sigma_a_eff,    // [n_cells × G] out
    double* __restrict__ sigma_s_eff,    // [n_cells × G] out
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ Cv_e,      // [n_cells] 電子比熱 c_v,e [erg/(g·eV)]
    const double* __restrict__ sigma_a,  // [n_cells × G] Planck吸収
    const PlanckTable* __restrict__ planck_table, // PlanckTable（ARCHITECTURE §4.5）。b_g = planck_fraction(g, Te[c], table) をカーネル内評価
    double alpha, double dt, double f_max,
    int n_cells, int n_groups
);
```

- **block**: 256, 1スレッド=1セル, **grid**: `(n_cells+255)/256`
- **処理**: 各セルで β = 4aT³/(ρ×Cv_e)（= 4aT³/C_{v,e}）→ σ_{a,P} (Planck重み平均) → f = min(1/(1+αβcΔtσ_{a,P}), f_max)（NUMERICS §6.1。f_max クランプ必須）
- **C_v 防御ガード**: `C_{v,e} = ρ × max(Cv_e, Cv_floor)` (Cv_floor = 1e-30 erg/(g·eV))。Cv_e ≤ 0 の場合 β=0（f=1: 完全暗黙化）として処理。テーブルEOSの外挿エラーによる Cv_e < 0 を安全に吸収する
- **単位規約**: 入力 `Cv_e` は質量比熱 \(c_{v,e}\) [erg/(g·eV)]。体積比熱 \(C_{v,e}=\rho c_{v,e}\) はカーネル内で構成して β を評価する。
- **群ループ**: 1スレッドが全G群をループ（G=16は小さいのでループ展開なしでOK）
  - σ_{a,eff,g} = f × σ_{a,g}
  - σ_{s,eff,g} = (1-f) × σ_{a,g}
- **レジスタ**: ~15（β, f, Planck平均 + 群ループ一時変数）
- **メモリ**: セルフィールド coalesced read、σ_a[n_cells×G] は群ループで G 回 stride-G アクセス → `__ldg()` で L2 活用
- **MPI注意**: R2 がゴーストセルの f_fleck を参照するため（R2 §6.2 注記参照）、ゴーストセルの f_fleck が必要。**v1.0 正規方式**: R1 は `(n_cells+255)/256` のみで実行し、R1 後に `halo_exchange(f_fleck)` でゴーストセルの値を取得する（§9 シーケンス準拠）。代替案として R1 の grid を `((n_cells+n_ghost)+255)/256` に拡張し U9 のゴーストセル σ_a を利用してローカル計算も可能だが、Cv_e のゴーストセル値も必要になるため halo_exchange 方式の方がシンプル

### 6.2 R2: ddmc_mode_judge

```cpp
__global__ void ddmc_mode_judge(
    uint8_t* __restrict__ ddmc_candidate, // [(n_cells+n_ghost) × G] out: 0=IMC, 1=DDMC候補（ゴーストセル含む、R3注記参照）
    const double* __restrict__ sigma_R,  // [(n_cells+n_ghost) × G] Rosseland（U9 がゴーストセル含めて計算）
    const double* __restrict__ sigma_a,  // [(n_cells+n_ghost) × G] Planck（同上）
    const double* __restrict__ f_fleck,  // [(n_cells+n_ghost)] R1出力 + halo_exchange で取得（R1 は n_cells のみ計算）
    const double* __restrict__ ell_ddmc,   // [(n_cells+n_ghost)] ℓ_i（H7出力 + halo_exchange。ARCHITECTURE §5.2 State.ell_ddmc）
    const double* __restrict__ vol,        // [(n_cells+n_ghost)] セル体積（H7出力 + halo_exchange）
    const double* __restrict__ face_area,  // [(n_cells+n_ghost) × n_faces] 面面積（H7出力 + halo_exchange）
    double tau_ddmc, double omega_ddmc,
    int n_cells, int n_ghost, int n_groups
);
```

- **block**: 256, 1スレッド=1セル, **grid**: `((n_cells+n_ghost)+255)/256`
- **MPI注意**: R3 がゴーストセルの ddmc_mode を参照するため、R2 もゴーストセルを含めて処理する（R3 注記参照）
- **処理**: 各セルの全G群をループ（NUMERICS §7.1）:
  1. τ_{i,g} = σ_{R,i,g} × ℓ_i → τ ≥ τ_DDMC?
  2. ω_{i,g} = 1 - f_i (v1.0) → ω ≥ ω_DDMC?
  3. 0 ≤ P̂(μ) ≤ 1? (NUMERICS §7.7.3: 全μで確率制約。μ=1で上限、μ=0近傍で下限チェック)
  4. ω・τ・P制約を満たすセル×群を `ddmc_candidate=1` に設定
  5. M-matrix条件（§7.3.3）は R3 `ddmc_leak_coeff` で最終判定し、`ddmc_mode` を確定
- **レジスタ**: ~15（τ, ω, P̂ + 群ループ変数）
- **ワープ発散**: 4条件の組み合わせで分岐するが、各条件は単純な比較演算のみで処理コストが均一 → 影響軽微

### 6.3 R6: source_particle_fill

```cpp
__global__ __launch_bounds__(128, 8) void source_particle_fill(
    // PhotonPool SoA output arrays
    double* __restrict__ pos_r,
    double* __restrict__ pos_z,
    double* __restrict__ dir_r,
    double* __restrict__ dir_z,
    double* __restrict__ dir_phi,
    double* __restrict__ energy,
    double* __restrict__ weight,
    double* __restrict__ time_remain,
    uint64_t* __restrict__ global_id,
    uint32_t* __restrict__ rng_counter,
    int32_t* __restrict__ cell_id,
    uint16_t* __restrict__ group_id,
    uint8_t* __restrict__ mode,
    uint8_t* __restrict__ alive,
    double* __restrict__ birth_energy,       // [n_particles] out: 誕生時エネルギー [erg]（= energy、R8 の f_cutoff 判定 §6.3.4 が参照）
    int8_t* __restrict__ sign,               // [n_particles] out: 粒子符号（legacy source は +1）
    // Source parameters
    const double* __restrict__ source_E,     // [n_cells × G] ソースエネルギー [erg]（R4出力）
    const int* __restrict__ particle_offsets, // [n_cells × G + 1] prefix sum
    const double* __restrict__ x_r,          // [n_nodes]（位置サンプルに使用: 1D_SPH r_lo/r_hi, 2D_RZ 双線形写像端点）
    const double* __restrict__ x_z,          // [n_nodes]
    const uint8_t* __restrict__ ddmc_mode,   // [(n_cells+n_ghost) × G]（ローカルセルのみ参照: cell_id < n_cells）
    double dt,
    uint64_t global_id_base,                 // step_base + rank_offset (§12.7.1): step×2^40 + MPI_Exscan
    uint64_t step,                           // RNG seed component (curand_init subsequence)
    uint64_t user_seed,                      // Main.seed（curand_init: global_id ^ user_seed, NUMERICS §12.7.1）
    int n_new_particles,                    // 生成粒子数 = particle_offsets[n_cells×G]（ホスト側でD2H済み）
    int n_cells, int n_groups, int nr, int nz
);
```

- **block**: 128, **grid**: `(n_new_particles+127)/128`（n_new_particles==0 の場合はカーネル起動をスキップ）
- **1スレッド=1粒子**。`if (tid >= n_new_particles) return;`（末尾ブロックの余剰スレッドガード、必須）
- **プールオフセット規約**: ホスト側で SoA ポインタを `&pos_r[n_alive_prev]` 等にオフセットして渡す。R6 はインデックス 0 から n_new_particles-1 に書き込み、プール全体では `[n_alive_prev .. n_alive_prev + n_new_particles - 1]` に格納される。R13（Marshak）は R6 の後に起動し、ポインタを `n_alive_prev + n_new_particles` でさらにオフセットする
- **処理**:
  1. スレッドID → (cell, group) をparticle_offsetsから逆引き:
     ```
     // スレッドID → (cell, group) 逆引き: particle_offsets[0..n_cells*G] に対する
     // upper_bound 二分探索で O(log(n_cells*G)) で決定
     int bin = upper_bound(particle_offsets, n_cells * G + 1, thread_particle_idx) - 1;
     int cell = bin / G;
     int group = bin % G;
     ```
  2. RNG初期化: `curand_init(seed=global_id ^ user_seed, subsequence=step, offset=0)`（§6.4.0、NUMERICS §12.7.1）。内部 Philox key/counter は NVIDIA 実装に委譲
  3. mode: ddmc_mode[c,g] から初期モード決定（位置・方向サンプルの前に実行）
  4. 位置・方向サンプル（**モード依存**、NUMERICS §7.5.2 準拠）:
     - **IMC（mode==0）**: 位置サンプル + 方向サンプル
       - 1D_SPH: `r = (r_lo³ + ξ(r_hi³-r_lo³))^{1/3}` (§6.2 (a))
       - 2D_RZ: 双線形写像 + R重み棄却法 (§6.2 (b))
       - 方向: 等方 `(μ,φ)` → `(Ω_r, Ω_z, Ω_φ)` (§6.2)
     - **DDMC（mode==1）**: 位置・方向フィールドを **NaN sentinel**（`0x7FF8000000000000`）に設定。
       DDMCはセル・群・エネルギー・時刻のみで追跡するため空間情報は不要。
       DDMC→IMCリーク（§7.7.2）時に初めて位置・方向をサンプルする
  5. エネルギー: `E_p = abs(source_E[c,g]) / N_p[c,g]` [erg]（source_E は R4 で V×Δt を含むため、ここでは除算のみ。NUMERICS §6.2）
  6. `weight = 1.0`（v1.0 不変。Russian roulette（R8 step 7 / R12）は `energy` を直接変更し、`weight` は常に 1.0 のまま。将来のバリアンスリダクション拡張用に予約）
  7. `birth_energy = energy`（誕生時エネルギーを記録。R8 の f_cutoff 判定 §6.3.4 が参照）
  8. `sign = sign(source_E[c,g])`（legacy source path は常に +1、difference residual path は ±1）
  9. alive = 1, time_remain = dt
- **ワープ発散**: 1D_SPH / 2D_RZ の位置サンプル分岐はコンパイル時に決定（`#ifdef TENRYU_2D`）のため実行時の発散なし。二分探索（upper_bound）はスレッド間でループ回数が同一（O(log(n_cells×G))固定）のため発散なし

