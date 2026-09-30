# 退役した輻射 Monte Carlo — CUDA_KERNELS の記述

2026-09-29 にビルドから外したモンテカルロ輻射（IMC・DDMC・ランダムウォーク・HOLO・difference 定式化）について、
`docs/CUDA_KERNELS.md` にあった設計記述をここへ移した。本文は移設時点（コミット 5bc8f6ce3 の文書）のままで、以後は更新しない。
節番号・節への参照（「§5.3」「CUDA_KERNELS §6.4」など）は移設前の `docs/CUDA_KERNELS.md` のもの。コードの所在と復元の手順は
`../README.md` を参照。

---

## 1. 元 `docs/CUDA_KERNELS.md` の §0.5 Composite Key Sort 戦略（R7+R11+R14 融合）と §0.5.1 Census Combing GPU パイプライン

### 0.5 Composite Key Sort 戦略（R7+R11+R14 融合）

輻射演算子の冒頭で、全粒子（alive + 前ステップの dead）に対し **合成キーソート** をステップ内2回（pre-transport + post-MPI）実行し、
従来の3操作（R7 セルソート + R11 dead compaction + R14 モード分離）を単一パスに融合する。

**合成キー（64ビット）**：
```
bits[63:49] = bucket_hash（15ビット）
              mode * (n_cells * G) + cell_id * G + group_id
              dead粒子は最大値（0x7FFF...F）でソート末尾化
bits[48:0]  = global_id の下位49ビット（粒子RNGストリーム識別子）
              **前提条件**: n_cells * G <= 2^14。初期化時に assert で検証する。
              bucket_hash によりモード→セル→群の3段階ソートが1回の RadixSort で実現される。
              dead粒子のエネルギーは dropped_energy に atomicAdd で蓄積し、
              E_numerical_loss に計上する（エネルギー保存）。
```

ソート後のメモリレイアウト（昇順）：
```
[0 .. n_imc-1]            : IMC alive粒子（cell_id昇順）
[n_imc .. n_alive-1]      : DDMC alive粒子（cell_id昇順）
[n_alive .. N_total-1]    : dead粒子 → 切り捨て（n_alive 以降は無視）
```

**効果**：
- セルデータ（opacity, Te等）への読み込みがL2キャッシュでヒット（セルソート効果）
- 同一セルへのタリーatomic競合がwarp内で完結（warp-level集約の前提）
- dead粒子が自動的に末尾に排除され、compaction不要
- IMC/DDMC分離が同時完了し、mode_partition不要
- 3操作×（16 SoA中15可変配列）gather → 1操作×1 fused gather で**メモリ帯域を ~60% 削減**
- 純IMCベンチ（P2）でスループット **20-40%** の改善が期待される

**コスト**（A100基準、\(N = N_{total}\)：alive + dead の合計粒子数）：
- 合成キー生成カーネル：~0.1 ms/100万粒子（1 thread = 1 particle、atomic count で n_imc/n_ddmc 算出）
- CUB RadixSort（key 8B + index 4B）：~1 ms/100万粒子
- Fused SoA Gather（16 SoA中15可変フィールド一括、alive除く）：~1.5 ms/100万粒子
- **合計 ~2.6 ms/100万粒子**（従来の R7+R11+R14 合計 ~5-7 ms/100万粒子 に対し **50-60% 削減**）
- 補助メモリ：CUB RadixSort temp + SoA double buffer ≈ \(\sim (24 + 93) \times N_{particles}\) bytes

**compact_alive_only() 最適化パス**：全粒子がIMCモード（`n_ddmc == 0`）の場合、
RadixSort をスキップし O(N) の atomic compaction を実行する。
alive粒子が `atomicAdd(counter)` で出力インデックスを取得し、`compact_indices[out]=src`
を記録する。後続の `gather_compacted_soa_kernel` が persistent scratch pool へ16フィールドを
一括gatherし、pool swapで反映する。drop粒子の signed energy は同一パス内の device counter に
集約し、gather kernel 内で `E_numerical_loss += fabs(sum)` として反映する。
~4.5 ms/100万粒子（RadixSort の ~12 ms に対し ~63% 削減、実装は per-step SoA再確保と
個別D2D copyを避ける）。
セル順序は保持されないが、post-transport sort ではタリー不要のため問題にならない。

#### 0.5.1 Census Combing GPU パイプライン

Census Combing（NUMERICS section 6.4.1）のGPU実装は、以下の3カーネル + CPU選択ロジックの
hybrid パイプラインとして構成される。

**R15: detect_bins_kernel**（block=256, grid=ceil(N/256)）
- ソート済み粒子配列のbucket hash（合成キーの上位15ビット）の変化点を検出
- 出力：`bin_flag[i]`（ビン開始位置で1、それ以外は0）
- CUB `DeviceExclusiveSum` でビンインデックスを算出

**CPU選択ロジック**（D2H転送後）：
- Kahan summation でビンエネルギー E_b を決定的に計算
- 重要度重み付き選択：`n_copy_b = floor(target * E_b / E_total)`
- 残余配分：systematic residual resampling
- gather_idx 配列を生成し H2D 転送

**R16: gather_equalized_kernel**（block=256, grid=ceil(N_selected/256)）
- gather_idx に基づき選択された粒子を連続配置にコピー
- エネルギー均等化：`E_new = E_bin / n_selected_bin`
- RNG counter を bin.key ベースで再初期化（決定性保証）

**パイプライン実行コスト**（A100基準、100万粒子、1000ビン）：
- detect_bins: ~0.1 ms
- CUB scan: ~0.2 ms
- D2H (energy+cell+group): ~0.5 ms
- CPU selection: ~1 ms
- H2D (gather_idx): ~0.1 ms
- gather_equalized: ~0.3 ms
- **合計: ~2.2 ms**（CPU kernel の ~300 ms に対し大幅改善だが、CPU選択がボトルネック）

---

## 2. 元 `docs/CUDA_KERNELS.md` の §1.5 Radiation のカーネル表（R1〜R16）

### 1.5 Radiation（16カーネル — 退役 imc_ddmc 系。現行 FLD/S_N カーネルの一覧は §6.7/§6.8）

| ID | カーネル名 | 種別 | N | 主要入出力 |
|----|-----------|------|---|-----------|
| R1 | `compute_fleck_factor` | cell | n_cells | Te,Cv_e,σ_a,Δt → f,σ_eff（G群ループ内包） |
| R2 | `ddmc_mode_judge` | cell | n_cells+n_ghost | τ,ω,P制約 → DDMC候補（ghost含む、G群ループ内包） |
| R3 | `ddmc_leak_coeff_kershaw` / `ddmc_leak_coeff_face` | cell | n_cells+n_ghost | Kershaw(2D)/face(1D) leak + M-matrix判定 → 最終mode（ghost含む、近傍×G群ループ内包） |
| R3b | `ddmc_interface_correct` | cell | n_cells | R3後: DDMC-IMCインターフェースリーク修正（§7.3.5、面×G群ループ内包） |
| R4 | `compute_source_energy` | cell | n_cells | f,σ_a,Te → source_E [erg]（difference有効時はsigned residual Q'、G群ループ内包） |
| R4b | `preseed_reference_absorption` | cell | n_cells×G | difference有効時: cσ_a,eff E_ref VΔt → rad_dep |
| R4c | `reference_face_transport_1d` | cell | n_cells×G | difference face_transport有効時: AP reference face divergence → deterministic U_ref_end/E_ref_avg buffers |
| R5 | `source_particle_count` | — | CUB prefix-sum | abs(source_E) → N_p offsets |
| R6 | `source_particle_fill` | particle | n_new | RNG → pos,dir,|E|,sign,group |
| R7 | `composite_sort_and_partition` | particle+CUB | N_total | 合成キーソート+fused gather（§0.5） |
| R7b | `ddmc_to_imc_resample` | particle | n_imc | DDMC→IMCモード遷移粒子の位置・方向再サンプル（§6.0d1） |
| R8 | `imc_transport_persistent` | particle | n_imc | **主要カーネル**：追跡ループ（Persistent Warp） |
| R9 | `ddmc_event_loop` | particle | n_ddmc | DDMCイベント処理 |
| R10 | `tally_finalize` | cell | n_cells×G | rad_E_tally → rad_E（legacy 正規化、difference では E_ref_avg + signed residual） |
| ~~R11~~ | ~~`photon_compaction`~~ | — | — | **R7に吸収**（§0.5 Composite Key Sort） |
| R12 | `russian_roulette` | particle | n_alive | 低重み粒子間引き |
| R13 | `marshak_source` | particle | n_marshak | 境界ソース粒子生成 |
| ~~R14~~ | ~~`mode_partition`~~ | — | — | **R7に吸収**（§0.5 Composite Key Sort） |
| R15 | `census_comb_detect_bins` | particle | N_alive | ビン境界検出（隣接キー比較） |
| R16 | `census_comb_gather_equalized` | particle | N_selected | 選択結果適用+エネルギー均等化 |

---

## 3. 元 `docs/CUDA_KERNELS.md` の §6 Radiation カーネル群の状態注記と §6.0a〜§6.6（R1〜R16 の詳細設計）

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

### 6.4 R8: imc_transport_persistent（旧・プロジェクト最重要カーネル — Persistent Warp）[RETIRED — legacy]

```cpp
__global__ __launch_bounds__(128, 8)
void imc_transport_persistent(
    // PhotonPool SoA (in/out) — IMCモード粒子のみ（R7 composite_sort_and_partition後）
    double* __restrict__ pos_r,
    double* __restrict__ pos_z,
    double* __restrict__ dir_r,
    double* __restrict__ dir_z,
    double* __restrict__ dir_phi,
    double* __restrict__ energy,
    double* __restrict__ weight,           // [n_particles] 粒子重み
    double* __restrict__ birth_energy,     // [n_particles] 誕生時エネルギー [erg]
    double* __restrict__ time_remain,
    uint64_t* __restrict__ global_id,
    uint32_t* __restrict__ rng_counter,
    int32_t* __restrict__ cell_id,
    uint16_t* __restrict__ group_id,
    uint8_t* __restrict__ mode,
    uint8_t* __restrict__ alive,
    // Cell data (read-only)
    const double* __restrict__ sigma_a_eff,  // [n_cells × G]
    const double* __restrict__ sigma_s_eff,  // [n_cells × G]
    const double* __restrict__ Te,           // [n_cells]
    const double* __restrict__ x_r,          // [n_nodes]
    const double* __restrict__ x_z,          // [n_nodes]
    const double* __restrict__ vol,          // [n_cells]
    const PlanckTable* __restrict__ planck_table,
    const uint8_t* __restrict__ ddmc_mode,   // [(n_cells+n_ghost) × G]（R8は隣接セルのddmc_modeを参照: IMC→DDMC変換判定 §7.7）
    const double* __restrict__ sigma_R,      // [n_cells × G] Rosseland不透明度（compute_P_hat用、NUMERICS §7.7.3）
    const double* __restrict__ face_area,    // [n_cells × n_faces] H7出力の面面積 [cm²]。compute_P_hat で面別 Δx_m = vol[new_cell]/face_area[new_cell*n_faces+entry_face] を算出（NUMERICS §7.7.4。entry_face = crossing_face ^ 1。面法線方向の代表長は面依存であり、セル平均は不可）
    const double* __restrict__ f_fleck,      // [n_cells] Fleck factor（R1出力）。compute_P_hat の omega = 1-f に使用（§7.7.3）
    const int8_t* __restrict__ face_bc_type, // [n_faces_boundary] 境界面タイプ（R3と同一。get_neighbor_cell のbc_code判定に使用）
    // Tally output (atomic)
    double* __restrict__ rad_dep,             // [n_cells × G]
    double* __restrict__ rad_E_tally,         // [n_cells × G]
    double* __restrict__ face_current_step,   // [(n_cells+1) × G] signed crossing current [erg] for diffusion reduced-flux classification
    const uint8_t* __restrict__ diff_cell,    // [n_cells] deterministic diffusion mask
    double* __restrict__ diff_face_current_in,// [(n_cells+1) × G] positive IMC→diffusion source [erg]
    double* __restrict__ rad_mom_dep,         // [n_cells × dim] 運動量沈着 [dyne·s/cm³]（NUMERICS §7.8）。
                                              // 2D_RZ: [n_cells×2]（R,Z成分）、1D_SPH: [n_cells×1]。
                                              // 各吸収イベントで atomicAdd: Δp = (ΔE_dep/c) × Ω̂（NUMERICS §10.1.1）
    double* __restrict__ E_escape,            // [n_groups] 群別脱出エネルギー（atomicAdd(&E_escape[group_id], ...)）
    double* __restrict__ E_numerical_loss,    // [1] 数値的エネルギー喪失（MAX_EVENTS超過粒子の残余エネルギー、§10.2 保存）
    // Work queue
    int* __restrict__ global_work_counter,    // [1] atomicカウンタ（0初期化済み）
    int n_imc,                                // IMCモード粒子数
    // Parameters
    double dt,
    double E_avg,                             // スカラー: S_total / N_p_total (ホスト計算)
    double w_cutoff, double p_survival,       // Russian roulette (defaults: w_cutoff=1e-10, p_survival=0.1, spec §6.4.5)
    double f_cutoff,                          // Cutoff fraction (default 0.0=無効、§6.3.4。E < f_cutoff × birth_energy で粒子終了)
    int nr, int nz, int n_groups, int n_faces, // n_faces: 1D_SPH=2, 2D_RZ=4（face_area 配列の stride に必要。R9 と統一）
    int dim,                                  // 空間次元数: 1D_SPH=1, 2D_RZ=2。rad_mom_dep[cell*dim+d] のストライドに使用
    int interface_method,                     // IMC→DDMC変換方式: 0=ASYMPTOTIC_DIFFUSION_LIMIT（既定, P̂(μ)判定）, 1=MARSHAK（常に変換）
    bool emissivity_preserving,               // True=P̂(μ)（既定、Densmore 2006 Eq.48）, False=標準P(μ)（Densmore 2007 Eq.34）。SPECIFICATION §6.4.5, NUMERICS §7.7.3
    uint64_t step,                            // 現在ステップ番号（curand_init subsequence、NUMERICS §12.7.1）
    uint64_t user_seed,                       // Main.seed（curand_init: global_id ^ user_seed、NUMERICS §12.7.1）
    // Error flags
    DeviceErrorFlags* error_flags
);
```

- **block**: 128, **grid**: `n_sm × 8`（SM数依存、固定）

**v1.0 追加引数（イベントカウンタ）**：
```cpp
    unsigned long long* __restrict__ cnt_boundary,       // [1] 境界交差カウンタ
    unsigned long long* __restrict__ cnt_scatter,        // [1] 散乱カウンタ
    unsigned long long* __restrict__ cnt_census,         // [1] census終了カウンタ
    unsigned long long* __restrict__ cnt_absorb_kill,    // [1] 吸収消滅カウンタ
    unsigned long long* __restrict__ cnt_absorb_survive, // [1] 吸収生存カウンタ
    unsigned long long* __restrict__ cnt_roulette_kill,  // [1] roulette消滅カウンタ
```
全カウンタはオプショナル（nullptr で無効化可能）。`verbosity="verbose"` 時のみ非 null。
スレッドローカル変数（`uint32_t`）として蓄積し、粒子追跡終了時に1回の `atomicAdd` で
グローバルカウンタに集約する（atomic 圧力の最小化）。

**v1.0 追加レジスタ（Scatter Carry）**：
```cpp
    double tau_scatter_remain;  // 残留散乱光学厚
```
- 粒子ロード時に `-log(xi)` で初期化（NUMERICS section 6.3.2 Scatter Carry参照）
- 各ステップで `tau_scatter_remain -= sigma_s * s_min` を減算
- `tau_scatter_remain <= 0` で散乱イベント発生
- 散乱後に `tau_scatter_remain = -log(xi_new)` で再サンプル
- 丸め誤差ガード：`0 < tau_scatter_remain <= 1e-14` の場合は `s_scatter = 0`（即時散乱）
- **Persistent Warp**: 1スレッドが複数粒子を順次処理

**Persistent Warp 疑似コード**:
```
const int lane = threadIdx.x & 31;

// === Phase 1: 初回粒子取得（ワープ単位一括） ===
int my_idx;
if (lane == 0) my_idx = atomicAdd(global_work_counter, 32);
my_idx = __shfl_sync(0xFFFFFFFF, my_idx, 0) + lane;

bool active = (my_idx < n_imc);
// レジスタに粒子状態をロード（coalesced）
if (active) load_particle_from_SoA(my_idx, ...);
// Census粒子の time_remain 再装填: 前ステップでcensus（time_remain=0）した粒子に新ステップの時間予算を付与
if (active && time_remain <= 0.0) time_remain = dt;

// === Phase 2: Persistent ループ ===
int n_events = 0;
const int MAX_EVENTS = 10000;  // 無限ループ防止ガード

// **ワープ同期制約（Volta+ ITS 必須）**:
// Phase 2 ループ内の `if (active):` ブロックで `continue` を使用してはならない。
// `continue` は Phase 3 の `__ballot_sync(0xFFFFFFFF, ...)` をバイパスし、
// 一部のレーンが Phase 2 先頭の `__any_sync` に到達する一方で
// 他のレーンが Phase 3 の `__ballot_sync` にいる状態を生み出す。
// full-mask sync primitive の異なるインスタンスへの分岐は
// Volta+ Independent Thread Scheduling で未定義動作（デッドロック/ハング）。
// **解決策**: active=false 後は `if (active):` ガードで後続処理をスキップし、
// 全レーンが毎イテレーション Phase 3 に到達することを保証する。
// ※ Phase 3 内の `continue`（need_refill==0, base_idx>=n_imc）は全レーンが
//   同一条件を評価するため非分岐であり安全。

while (true):
    if (!__any_sync(0xFFFFFFFF, active)) break;  // ワープ全体終了

    if (active):
        // 無限ループ検出
        n_events++;
        if (n_events >= MAX_EVENTS):
            atomicExch(&error_flags->infinite_loop, 1);
            atomicAdd(&E_numerical_loss[0], energy);  // 残余エネルギーを数値的喪失に計上（§10.2 保存）
            alive = 0;  // 粒子を強制終了
            store_particle_to_SoA(my_idx, ...);
            active = false;
            // ※ continue 禁止（ワープ同期制約）— 全レーンが Phase 3 へ到達する必要がある

    if (active):
        c = cell_id;  g = group_id;
        σ_a = __ldg(&sigma_a_eff[c*G+g]);
        σ_s = __ldg(&sigma_s_eff[c*G+g]);

        // 1. イベント距離計算（§6.3.2 と同一）
        s_bdry = compute_boundary_distance(pos, dir, cell c);
        s_cen  = c_light * time_remain;
        s_scat = (σ_s > 0) ? -log(rng()) / σ_s : INFINITY;
        s_min  = min(s_bdry, s_cen, s_scat);

        // 2. 連続吸収（§6.3.3 と同一、expm1で桁落ち回避；L5レーザーカーネル §5.6.6 準拠）
        E_old = energy;
        ΔE_dep = -E_old * expm1(-σ_a * s_min);
        energy = E_old - ΔE_dep;
        // **即時終了ガード**: exp underflow で energy==0 の場合、以降のイベント処理は無意味。
        // ΔE_dep = E_old（全エネルギー沈着済み）のため、タリー蓄積後に即座に終了する。
        // Russian roulette ではなく決定論的終了（NUMERICS §6.3.3 エネルギー枯渇条件）

        // 3. タリー蓄積（warp-level集約、§10.3）
        warp_tally_accumulate(rad_dep, rad_E_tally, c, g, ΔE_dep, ...);

        // 3b. エネルギー枯渇即時終了
        if (energy <= 0.0):
            alive = 0; store_particle_to_SoA(my_idx, ...); active = false;
            // ※ continue 禁止（ワープ同期制約）— 全レーンが Phase 3 へ到達する必要がある

    if (active):
        // 4. 位置・時間更新
        pos += dir * s_min;
        time_remain -= s_min / c_light;
        // **Census近傍スナップ**（浮動小数点桁落ち防止）:
        // s_min/c ≈ time_remain の場合、減算で time_remain が微小負値や非物理的微小正値に
        // なりうる。ε_cen = 1e-10 × dt（NUMERICS §6.4 準拠） で判定し、強制census化する。
        if (time_remain < 1e-10 * dt):
            time_remain = 0.0;  // → 後段の census パスで処理

        // 5. イベント処理
        // **優先順位**（NUMERICS §6.3.2 準拠）: Census > Boundary > Scatter
        // 同一距離の場合、Census を最優先とする。これにより、ステップ境界と同時に
        // セル境界に到達した粒子は確実にcensus化され、次ステップで処理される。
        // Census > Boundary の理由: セル遷移よりも時間管理を優先し、
        // 次ステップでの正確な物理状態更新を保証する。
        // --- イベント優先順位: Census > Boundary > Scatter ---
        // 浮動小数点等値比較を回避し、状態ベースの優先判定を使用する。
        // Census: time_remain ≤ 0（ε_cen スナップ済み）で判定。s_min 比較不要。
        // Boundary/Scatter: s_min == s_bdry は min() がビット同一値を返すため安全だが、
        //   Census チェックを先行させることで同時到達時の優先が保証される。
        if (time_remain <= 0.0):  // Census（最優先）
            time_remain = 0;
            store_particle_to_SoA(my_idx, ...);
            active = false;
        elif (s_min == s_bdry):   // Boundary crossing
            new_cell = get_neighbor_cell(c, crossing_face);  // §6.4.3 参照
            if (new_cell < 0):
                // === 境界条件処理 (§6.3.2) ===
                // new_cell の負値で境界タイプを判定:
                //   new_cell = -1: VACUUM (脱出)
                //   new_cell = -2: REFLECT (鏡面反射)
                //   new_cell = -3: AXIS (R=0 軸対称)
                //   new_cell = -4: MARSHAK (外部放射源 — 脱出として処理)
                //   new_cell = -5: パーティション境界 → emigrant buffer
                if (new_cell == -1 || new_cell == -4):
                    // VACUUM / MARSHAK: 粒子脱出
                    atomicAdd(&E_escape[group_id], energy);
                    alive = 0;
                    store_particle_to_SoA(my_idx, ...);
                    active = false;
                elif (new_cell == -2):
                    // REFLECT: 鏡面反射
                    // crossing_face: 0=R_left, 1=R_right, 2=Z_bottom, 3=Z_top
                    if (crossing_face == 0 || crossing_face == 1):
                        dir_r = -dir_r;  // R 方向反転
                    else:
                        dir_z = -dir_z;  // Z 方向反転
                    // cell_id は変更しない (同一セルに留まる)
                elif (new_cell == -3):
                    // AXIS (R=0): R 方向反転 + φ → φ + π
                    dir_r = -dir_r;
                    dir_phi = -dir_phi;
                    pos_r = abs(pos_r);  // R >= 0 保証
                elif (new_cell == -5):
                    // パーティション境界: emigrant マーク（ARCHITECTURE §7.1.3 統一契約準拠）
                    cell_id = -(100 + crossing_face);  // face エンコード: face=0→-100, 1→-101, 2→-102, 3→-103
                    store_particle_to_SoA(my_idx, ...);
                    active = false;  // 次の MPI exchange で処理（alive=1 維持。P5 が cell_id<0 で検出）
            elif (new_cell >= 0):
                cell_id = new_cell;
                if (ddmc_mode[new_cell*G+g]):
                    // IMC→DDMC変換 (§7.7.1)
                    // interface_method 分岐（SPECIFICATION §6.4.5, NUMERICS §7.7）:
                    //   "asymptotic_diffusion_limit"（既定）: P̂(μ)確率判定（Densmore 2006 Eq.48）
                    //   "marshak": P=1（常に変換、精度低・安定）
                    if (interface_method == MARSHAK):
                        convert = true;  // 無条件変換
                    else:  // asymptotic_diffusion_limit
                        // **退化面ガード（検査を除算より先に実行）**: face_area ≤ 0 は構造格子では非物理的だが、
                        // 極端なメッシュ歪みで face_area ≈ 0 になりうる → 除算前に検査してNaN/Inf伝播を防止
                        // **entry_face**: 粒子が new_cell に入る面 = crossing_face の対面
                        // 物理面規約: R_left(0)↔R_right(1), Z_bottom(2)↔Z_top(3)
                        int entry_face = crossing_face ^ 1;  // 0↔1, 2↔3
                        if (face_area[new_cell*n_faces + entry_face] < 1e-30):
                            convert = false;  // 退化面: IMC 継続
                        else:
                            Δx_m = vol[new_cell] / face_area[new_cell*n_faces + entry_face];  // 面別代表長（§7.7.4）
                            // **注意**: crossing_face は旧セルの出口面。new_cell の入口面は対面。
                            // crossing_face を直接使うと new_cell の反対側の面積が参照され、
                            // 変形メッシュで P̂ 計算に誤差が生じる。
                            μ = max(0, -dot(Ω, n̂_face))  // 面法線に対する入射余弦（n̂_face は面外向き法線、NUMERICS §7.7.3）
                            if (emissivity_preserving):
                                P_hat = compute_P_hat(σ_R, Δx_m, ω, μ);  // Densmore 2006 Eq.48（既定）
                            else:
                                P_hat = compute_P_standard(σ_R, Δx_m, μ); // Densmore 2007 Eq.34（比較用）
                            convert = (rng() < P_hat);
                    if (convert):
                        mode = DDMC;
                        // **位置・方向を NaN sentinel に設定**（DDMC粒子の不変条件を保証）：
                        // DDMCは位置を使用しないため安全。NaN化が必要な理由:
                        // (1) U7 は mode==DDMC でスキップするが、NaN は防御的不変条件（mode 破損時の安全策）
                        // (2) R7b が isnan で DDMC→IMC 遷移を検出する判定に依存
                        // (3) NaN化しないと ALE rezone 後に stale 位置が残り、
                        //     セルモード遷移時に R7b が見逃して R8 が壊れる
                        pos_r = NaN;  pos_z = NaN;
                        dir_r = NaN;  dir_z = NaN;  dir_phi = NaN;
                        // 粒子をSoAに書き戻し、DDMCカーネルで処理
                        store_particle_to_SoA(my_idx, ...);
                        active = false;
                    else:
                        // IMC→DDMC 変換棄却: IMC 側に反射
                        cell_id = c;  // **必須**: cell_id を元のIMCセルに復元。
                                      // line 1951 で cell_id=new_cell（DDMCセル）に更新済みのため、
                                      // 復元しないと粒子が DDMCセルに IMC モードで留まり、
                                      // 次イテレーションで誤ったジオメトリを参照する。
                        // IMC 側半空間へ**等方的に**方向を再サンプル（NUMERICS §7.7.1）
                        // P(μ) = 1（等方）: cos θ = ξ（NOT √ξ）
                        // ※ DDMC→IMC 変換（R9）の cosine 分布 P(μ)=2μ とは異なる
                        sample_isotropic_half_space_uniform(dir, -face_normal);
            else:
                // 未定義の境界コード（new_cell が -4 等の未使用値）
                // R8/R9 のバグまたはメッシュ破損 → DeviceErrorFlags 設定 + 粒子殺害
                error_flags->invalid_boundary_code = 1;
                atomicAdd(&E_numerical_loss, energy);  // 殺害粒子のエネルギーを数値損失に計上（保存則維持）
                alive = 0; store_particle_to_SoA(my_idx, ...); active = false;
        else:  // Scatter（最低優先）
            sample_isotropic(dir);
            // IMC の Fleck 因子による実効散乱は v1.0 では常に inelastic（グループ再サンプリングあり）。
            // SPECIFICATION §6.4.5 の imc.inelastic_scatter パラメータは v1.0 では常に True として扱い、
            // False が指定されても無視する（WARNING 出力）。将来バージョンで elastic 散乱に対応予定。
            if (inelastic):  // v1.0: always true（SPECIFICATION §6.4.5 参照）
                int g_new = sample_emission_group(planck_frac, sigma_a_eff, c, G, rng_state, rng_counter);
                if (g_new >= 0) group_id = g_new;  // -1 = 退化ケース（§6.3.4）、群変更なし

        // 6. Cutoff fraction termination（§6.3.4、既定 f_cutoff=0.0 で無効）
        if (active && f_cutoff > 0.0 && energy < f_cutoff * birth_energy):
            warp_tally_accumulate(rad_dep, ..., c, g, energy, ...);
            alive = 0; store_particle_to_SoA(my_idx, ...); active = false;

        // 7. Russian roulette（§6.3.4）
        if (active && energy < w_cutoff * E_avg):
            if (rng() < p_survival):
                energy /= p_survival;
            else:
                warp_tally_accumulate(rad_dep, ..., c, g, energy, ...);
                alive = 0;
                store_particle_to_SoA(my_idx, ...);
                active = false;

    // === Phase 3: Ballot Refill — 終了スレッドに新粒子を補充 ===
    uint32_t need_refill = __ballot_sync(0xFFFFFFFF, !active);
    if (need_refill == 0) continue;  // 全レーンアクティブ

    int n_needed = __popc(need_refill);
    int base_idx;
    if (lane == 0) base_idx = atomicAdd(global_work_counter, n_needed);
    base_idx = __shfl_sync(0xFFFFFFFF, base_idx, 0);

    if (base_idx >= n_imc):
        if (__all_sync(0xFFFFFFFF, !active)) break;
        continue;

    if (!active):
        int my_offset = __popc(need_refill & ((1u << lane) - 1));
        int new_idx = base_idx + my_offset;
        if (new_idx < n_imc):
            my_idx = new_idx;
            load_particle_from_SoA(my_idx, ...);
            // Census粒子の time_remain 再装填（Phase 1 と同一、NUMERICS §6.3.1）
            // Ballot Refill で取得した粒子が census 由来（time_remain<=0）の場合、
            // 新ステップの時間予算 dt を付与する。この処理を省略すると census 粒子が
            // 即座に再 census 化され、放射輸送が遅延する
            if (time_remain <= 0.0) time_remain = dt;
            n_events = 0;  // **必須**: 新粒子のイベントカウンタをリセット。
                           // リセットしないと前の粒子の n_events が蓄積され、
                           // MAX_EVENTS 判定で新粒子が即座に kill される。
            active = true;
```

**性能特性分析**:

| 項目 | 値 | 影響 |
|-----|---|------|
| レジスタ/スレッド | ~60 (32-bit換算) | block=128 → 8 blocks/SM → 50% occupancy |
| グリッドサイズ | n_sm × 8（固定） | A100: 864 blocks = 110,592 threads |
| SIMT効率 | 90-95%（典型） | Ballot Refill により idle lane を最小化 |
| Work Queue atomic | ~1回/warp/refill | 粒子寿命（5-50 iter）あたり → 全体 atomic 数は微小 |
| メモリ帯域 | 粒子load/store は history-based と同一 | 93B/particle × 2 回（生涯1往復 + refill時） |
| タリー集約 | warp-level（`__match_any_sync`） | セルソート済みで peers ~28-32 → atomic 削減 |

**タリー最適化（NUMERICS §10.3 / ARCHITECTURE §4.5 準拠）**:

Persistent Warp 内のタリー蓄積は `warp_tally_accumulate` で実装する。
セルソート済み粒子に対し、`tally_mode` 設定に応じた集約を適用する。

**Stage 1: Warp-level集約（v1.0既定）**（`tally_mode="warp"`、CC 7.0+ 必須）

セルソート済み粒子（§0.5）に対し、warp内のピアグループ集約を行う。

```cuda
// --- Warp-level tally reduction（NUMERICS §10.3 準拠）---
uint32_t active = __activemask();
int      lane   = threadIdx.x & 31;
int      key    = cell_id * n_groups + group_id;

// 同一keyのレーンを検出
uint32_t peers  = __match_any_sync(active, key);
int      leader = __ffs(peers) - 1;

// ピアグループ内 segmented reduction（全ピアレーンが __shfl_down_sync に参加 — 必須）
// 注意：リーダーのみが __shfl_sync を呼ぶパターンは CUDA 仕様上の未定義動作（§B.15）。
// 全ピアレーンが同一 mask で __shfl_down_sync に参加する以下のパターンを使用する。
double sum_dep = delta_E_dep;
double sum_tl  = delta_E_tl;
double sum_mom_r = delta_mom_r;  // 運動量沈着 R成分（momentum_deposition有効時のみ非ゼロ）
double sum_mom_z = delta_mom_z;  // 運動量沈着 Z成分（1D_SPHではゼロ）
for (int offset = 16; offset >= 1; offset >>= 1) {
    double tmp_dep = __shfl_down_sync(peers, sum_dep, offset);
    double tmp_tl  = __shfl_down_sync(peers, sum_tl,  offset);
    double tmp_mr  = __shfl_down_sync(peers, sum_mom_r, offset);
    double tmp_mz  = __shfl_down_sync(peers, sum_mom_z, offset);
    int src_lane = lane + offset;
    if (src_lane < 32 && ((peers >> src_lane) & 1)) {
        sum_dep += tmp_dep;
        sum_tl  += tmp_tl;
        sum_mom_r += tmp_mr;
        sum_mom_z += tmp_mz;
    }
}
// リーダーのみが集約結果を書き出す
if (lane == leader) {
    if (USE_BLOCK_TALLY)
        block_tally_accumulate(key, sum_dep, sum_tl);  // Stage 2
    else {
        atomicAdd(&rad_dep[key],     sum_dep);
        atomicAdd(&rad_E_tally[key], sum_tl);
    }
    // 運動量沈着も warp-level 集約（rad_dep/rad_E_tally と同じピアグループ）
    // rad_mom_dep は [n_cells × dim] インデックスのため key ではなく cell_id で書き出す
    if (sum_mom_r != 0.0) atomicAdd(&rad_mom_dep[cell_id * dim + 0], sum_mom_r);
    if (dim > 1 && sum_mom_z != 0.0) atomicAdd(&rad_mom_dep[cell_id * dim + 1], sum_mom_z);  // 1D_SPH: dim=1, z成分なし
}
```

- レジスタ増加：~8（peers, leader, offset, tmp×4）
- 削減率：global atomicAdd 回数を最大32分の1に削減
- **同期**: `__match_any_sync` と `__shfl_down_sync` は暗黙のワープ同期を含むため、追加の `__syncwarp()` は不要
- **重要**: `__shfl_down_sync(peers, ...)` は全ピアレーンが参加する必要がある（CUDA §B.15）。リーダーのみの `__shfl_sync` は未定義動作

**Stage 2: Block-level共有メモリ集約（将来拡張）**（`tally_mode="warp_block"`、atomicCAS open-addressing）

> **v1.0では実装しない**：Persistent Warp（NUMERICS §6.6）ではブロックがカーネル全生存期間にわたって存続し、
> N\_BINS=128のヒストグラムが約8ユニークセルで飽和するため、大半がglobal atomicAddにフォールバックする。
> 定期的フラッシュに必要な `__syncthreads()` はワープ独立進行と矛盾する。

Stage 1 のリーダー出力を共有メモリ上のビンヒストグラムに蓄積する。
スロット割り当てには atomicCAS open-addressing を使用し、競合状態を排除する（NUMERICS §10.3 準拠）。

```cuda
// --- Block-level tally histogram (atomicCAS open-addressing) ---
// Stage 2: ブロックレベル集約 (atomicCAS open-addressing)
__shared__ int    smem_keys[N_BINS];     // flat tally key = cell_id * G + group_id per bin (-1 = empty)
__shared__ double smem_dep[N_BINS];      // rad_dep accumulator
__shared__ double smem_tl[N_BINS];       // rad_E_tally accumulator
// N_BINS = 128 (compile-time constant)

// 初期化（ブロック先頭）
for (int i = threadIdx.x; i < N_BINS; i += blockDim.x) {
    smem_keys[i] = -1;
    smem_dep[i]  = 0.0;
    smem_tl[i]   = 0.0;
}
__syncthreads();

__device__ void block_tally_accumulate(int key, double dep, double tl) {
    // key = cell_id * G + group_id（Stage 1 リーダーが渡す flat tally key）
    int slot = key % N_BINS;             // initial probe
    for (int probe = 0; probe < N_BINS; ++probe) {
        int old = atomicCAS(&smem_keys[slot], -1, key);
        if (old == -1 || old == key) {
            // スロット確保成功 or 既存キー一致
            atomicAdd(&smem_dep[slot], dep);
            atomicAdd(&smem_tl[slot], tl);
            return;
        }
        slot = (slot + 1) % N_BINS;  // linear probing
    }
    // フォールバック: 全スロット使用済み → グローバルメモリに直接書き込み (Stage 3)
    atomicAdd(&rad_dep[key], dep);
    atomicAdd(&rad_E_tally[key], tl);
}

// ブロック末尾で一括flush
__syncthreads();
for (int i = threadIdx.x; i < N_BINS; i += blockDim.x) {
    if (smem_keys[i] >= 0) {
        atomicAdd(&rad_dep[smem_keys[i]],     smem_dep[i]);
        atomicAdd(&rad_E_tally[smem_keys[i]], smem_tl[i]);
    }
}
```

- 共有メモリ：128 × (8+8+4) = 2.5 KB/block
- 8 blocks/SM で 20 KB（A100 164 KB の 12%）→ occupancy影響なし

**Stage 3: Global atomicAdd**（全モード共通）

```cuda
atomicAdd(&rad_dep[cell_id * G + group_id],     delta_E_dep);
atomicAdd(&rad_E_tally[cell_id * G + group_id], delta_E_tl);
```

CC 6.0+（Pascal以降）でハードウェアサポート。`atomicAdd(double*)` は relaxed ordering で十分（タリー値の読み取りは R10 tally_finalize で行い、R8/R9 完了後に `cudaStreamSynchronize` が介在するため、明示的な `__threadfence()` は不要）。

### 6.4.0 サンプリング __device__ 関数

R8/R9/R6/R13 から呼び出されるサンプリングヘルパー関数群。
**cuRAND device API**（`curand_kernel.h`）を使用（NUMERICS §12.7.1 準拠）:

```cuda
#include <curand_kernel.h>

// --- cuRAND state 初期化（カーネル冒頭で1回、レジスタ常駐）---
__device__ __forceinline__ curandStatePhilox4_32_10_t init_rng(
    uint64_t global_id, uint64_t user_seed, uint64_t step_number, uint32_t rng_counter) {
    curandStatePhilox4_32_10_t state;
    curand_init(global_id ^ user_seed, step_number, (unsigned long long)rng_counter, &state);
    return state;  // 44B、レジスタ常駐。user_seed = Main.seed（NUMERICS §12.7.1）
}

// --- 等方サンプリング (4π全方向) ---
__device__ __forceinline__ void sample_isotropic(
    double& dir_r, double& dir_z, double& dir_phi,
    curandStatePhilox4_32_10_t& rng_state, uint32_t& rng_counter) {
    // 2 RNG draws: cos_theta = 2*xi1 - 1, phi = 2*pi*xi2
    double xi1 = curand_uniform_double(&rng_state); rng_counter++;
    double xi2 = curand_uniform_double(&rng_state); rng_counter++;
    double cos_theta = 2.0 * xi1 - 1.0;
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    double phi = 2.0 * M_PI * xi2;
    dir_r   = sin_theta * cos(phi);
    dir_z   = cos_theta;
    dir_phi = sin_theta * sin(phi);
}

// --- 半球等方サンプリング (P(μ)=1, cos θ = ξ) ---
// IMC→DDMC非変換時の反射に使用（NUMERICS §7.7.1「等方的に反射」）
__device__ __forceinline__ void sample_isotropic_half_space_uniform(
    double n_r, double n_z,
    double& dir_r, double& dir_z, double& dir_phi,
    curandStatePhilox4_32_10_t& rng_state, uint32_t& rng_counter) {
    // 2 RNG draws: cos_theta = xi1 (等方: P(μ)=1), phi = 2*pi*xi2
    double xi1 = curand_uniform_double(&rng_state); rng_counter++;
    double xi2 = curand_uniform_double(&rng_state); rng_counter++;
    double cos_theta = xi1;  // 等方: P(μ) = 1（NOT sqrt(xi) の cosine 分布）
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    double phi = 2.0 * M_PI * xi2;
    dir_r   = cos_theta * n_r + sin_theta * cos(phi) * (-n_z);
    dir_z   = cos_theta * n_z + sin_theta * cos(phi) * n_r;
    dir_phi = sin_theta * sin(phi);
}

// --- 半球サンプリング (コサイン重み: P(μ) = 2μ, cos θ = √ξ) ---
// DDMC→IMC変換時の面法線サンプリングに使用（NUMERICS §7.7.2, R9 参照）
__device__ __forceinline__ void sample_isotropic_half_space(
    double n_r, double n_z,
    double& dir_r, double& dir_z, double& dir_phi,
    curandStatePhilox4_32_10_t& rng_state, uint32_t& rng_counter) {
    // 2 RNG draws: cos_theta = sqrt(xi1), phi = 2*pi*xi2
    double xi1 = curand_uniform_double(&rng_state); rng_counter++;
    double xi2 = curand_uniform_double(&rng_state); rng_counter++;
    double cos_theta = sqrt(xi1);  // コサイン重み: P(μ) = 2μ
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    double phi = 2.0 * M_PI * xi2;
    // ローカル座標系 (法線 n 基準) からグローバル座標系に変換
    // 面法線 n̂ = (n_r, n_z) に対する正規直交基底:
    //   û = (-n_z, n_r)  （面接線方向、RZ平面内）
    //   ŵ = n̂ × û        （方位角方向、RZ平面外）
    // ローカル方向ベクトル:
    //   Ω_local = cos_theta * n̂ + sin_theta * (cos(phi) * û + sin(phi) * ŵ)
    // グローバル (dir_r, dir_z, dir_phi) への変換:
    //   dir_r   = cos_theta * n_r + sin_theta * cos(phi) * (-n_z);
    //   dir_z   = cos_theta * n_z + sin_theta * cos(phi) * n_r;
    //   dir_phi = sin_theta * sin(phi);
    // （NUMERICS §7.7.2 の面法線座標系構築を参照）
    dir_r   = cos_theta * n_r + sin_theta * cos(phi) * (-n_z);
    dir_z   = cos_theta * n_z + sin_theta * cos(phi) * n_r;
    dir_phi = sin_theta * sin(phi);
}

// --- 面上位置サンプリング (DDMC→IMC 変換 + R13 Marshak 用) ---
// NUMERICS §7.7.2（DDMC→IMC位置）および §8.2 step 4（Marshak位置）準拠
__device__ __forceinline__ void sample_position_on_face(
    int face_id,                 // 面ID（0=R_left,1=R_right,2=Z_bottom,3=Z_top）
    int cell_id, int nr, int nz, // セル → 面端点特定用
    const double* x_r, const double* x_z, // [n_nodes] 節点座標
    double& pos_r, double& pos_z,         // out: サンプル位置 [cm]
    curandStatePhilox4_32_10_t& rng_state, uint32_t& rng_counter) {
    // **1D_SPH**: 球面 r=r_f 上で等方サンプル（NUMERICS §7.7.2）
    //   mu = 2*xi1 - 1, phi = 2*pi*xi2, r = (r_f, 0)（1D内部表現）
    // **2D_RZ**: 辺 (V_k, V_{k+1}) 上で R 重み付きサンプル（NUMERICS §7.7.2）
    //   面端点: n00/n01/n10/n11 から face_id に応じて (P1, P2) を特定
    //   t = sample_R_weighted(P1.r, P2.r, xi)（NUMERICS §8.2 step 4）
    //   pos_r = P1.r + t*(P2.r - P1.r), pos_z = P1.z + t*(P2.z - P1.z)
    //   R 重み付き: CDF(t) = (r1*t + (r2-r1)*t²/2) / (r1 + (r2-r1)/2)
    //   逆CDF は二次方程式の解。r1≈r2 の場合は一様サンプルにフォールバック
    double xi = curand_uniform_double(&rng_state); rng_counter++;
    // [実装]: face_id → 面端点(P1, P2)の特定は §6.4.3 の面規約に従う
    // 面端点の特定後、上記の R 重み付き逆 CDF で t を算出し pos_r/pos_z を設定
}

// --- 放出群サンプリング (σ_a,eff × Planck 重み付き CDF) ---
// NUMERICS §6.3.4: P_g ∝ σ_{a,eff,i,g} × b_g(T_{e,i})
__device__ __forceinline__ int sample_emission_group(
    const double* planck_frac, // [G] Planck分率 b_g(T_e)（PlanckTable から事前評価。ARCHITECTURE §4.5）
    const double* sigma_a_eff, // [n_cells × G] 実効吸収不透明度（R1出力）
    int c, int G,              // セルインデックス、群数
    curandStatePhilox4_32_10_t& rng_state, uint32_t& rng_counter) {
    // 2-pass CDF サンプリング（スタック配列不使用、レジスタ圧力最小化）
    // **設計根拠**: 1-pass 方式の `double cdf[48]` は 384B/thread のローカルメモリを消費し、
    // __forceinline__ 展開時に R8 全体のレジスタ圧力を悪化させる。
    // 2-pass 方式は sigma_a_eff × planck_frac を 2回走査するが、2回目は L1 キャッシュに載るため
    // 実効コストは O(1) スタック + ~10% の追加メモリ帯域（スカラー4個のみ使用）。
    //
    // Pass 1: 合計 sum を計算
    double sum = 0.0;
    for (int g = 0; g < G; g++) {
        sum += sigma_a_eff[c * G + g] * planck_frac[g];
    }
    // 退化ガード: sum == 0（全群 σ_a,eff=0、真空セル）→ 群変更なし
    if (sum <= 0.0) return -1;  // 呼び出し側で g_old を維持（NUMERICS §6.3.4）
    // Pass 2: running CDF でサンプリング（線形探索、G ≤ 48 では二分探索不要、NUMERICS §6.3.4）
    double xi = curand_uniform_double(&rng_state); rng_counter++;
    double target = xi * sum;
    double running = 0.0;
    for (int g = 0; g < G; g++) {
        running += sigma_a_eff[c * G + g] * planck_frac[g];
        if (running >= target) return g;
    }
    return G - 1;  // 丸め誤差のフォールバック
}
```

- **レジスタ**: 各関数 ~5-8 + cuRAND state 44B（レジスタ常駐）。インライン展開されるため呼び出し元のレジスタ予算に含まれる。
  `sample_emission_group` は2-pass方式で **ローカルメモリ（スタック配列）不使用**（スカラー4個のみ）
- **インライン**: `__forceinline__` 属性を付与。R8 の内部ループから高頻度で呼び出されるため、関数呼び出しオーバーヘッド（レジスタ退避・復帰）を排除する。
  **注意**: `sample_emission_group` は **R8専用**。R9（DDMC）ではステップ内の群変更は行わない（NUMERICS §7.5: DDMCイベントは群固定）。
  `sample_isotropic_direction`, `init_rng` 等は R8/R9 共通
- **RNG**: cuRAND device API（Philox4x32-10）。`curand_uniform_double()` は Philox 実装で1語（uint32）消費の \(U(0,1]\) を返す。`rng_counter` は呼び出し回数を追跡し、カーネル終了時にPhototonPool へ書き戻す（NUMERICS §12.7.1）
- **cuRAND state ライフサイクル**: カーネル冒頭で `init_rng()` → レジスタ常駐 → カーネル終了時に `rng_counter` のみ保存。cuRAND state 自体はグローバルメモリに書き出さない

### 6.4.0a1 IMC→DDMC 変換確率 compute_P_hat __device__ 関数

R8（imc_transport）の IMC→DDMC 変換判定で使用する修正変換確率 P_hat(mu) を計算する（NUMERICS §7.7.3, Densmore 2006 Eq.48）。

```cuda
// --- IMC→DDMC 変換確率（emissivity-preserving, NUMERICS §7.7.3）---
__device__ __forceinline__ double compute_P_hat(
    double sigma_R,   // DDMC側セルの Rosseland 不透明度 [cm^-1]
    double delta_x,   // 面法線方向の代表長: V_cell / A_face [cm] (§7.7.4)。面依存: R8 で crossing_face から算出
    double omega,     // 散乱比: 1 - f_fleck (§7.1.1)
    double mu         // 入射方向余弦 (面法線に対する cos, mu > 0)
) {
    // 光学厚
    double tau = sigma_R * delta_x;

    // 完全散乱の特殊ケース（ω→1: ε'→0, β→0 で P̂→0。§7.7.3 数値安定性）
    // 純散乱媒質では emissivity がゼロのため変換確率もゼロ（P̂→1 ではない）
    if (omega > 0.999999) return 0.0;

    // Milne 外挿長
    double lambda = 0.7104;  // [無次元] (§7.4 vacuum 外挿長 d_ext = lambda / sigma_tr)

    // 解析拡散 emissivity（Densmore 2006 Eq.19、NUMERICS §7.7.3）
    double sqrt_arg = 3.0 * (1.0 - omega);
    double eps_prime = (4.0 / 3.0) * sqrt(sqrt_arg)
                     / (1.0 + lambda * sqrt(sqrt_arg));

    // β（Densmore 2006 Eq.48 の分母補助量）
    double one_minus_omega = 1.0 - omega;
    double beta = 1.5 * one_minus_omega * tau * tau
                + sqrt(3.0 * one_minus_omega * tau * tau
                     + 2.25 * one_minus_omega * one_minus_omega
                           * tau * tau * tau * tau);

    // 安全策1: 分母が非正の場合は標準 P にフォールバック（§7.7.3）
    double denom = beta - (4.0 / 3.0) * eps_prime * tau;
    if (denom <= 0.0) {
        // 標準 P（§7.7.1）: P(μ) = 4/(3τ + 6λ) × (1 + 3μ/2)
        double P_std = 4.0 / (3.0 * tau + 6.0 * lambda) * (1.0 + 1.5 * mu);
        return fmin(P_std, 1.0);
    }

    // 修正変換確率 P_hat（Densmore 2006 Eq.48）
    double P_hat = eps_prime * beta / denom;

    // 安全策3: P̂ < 0 の場合は標準 P にフォールバック（§7.7.3）
    if (P_hat < 0.0) {
        double P_std = 4.0 / (3.0 * tau + 6.0 * lambda) * (1.0 + 1.5 * mu);
        return fmin(P_std, 1.0);
    }

    // 安全策2: P̂ > 4/5 の場合はクランプ（P̂(1) = 5/4 × P̂ ≤ 1 を保証、§7.7.3）
    if (P_hat > 0.8) P_hat = 0.8;

    // 方向依存変換確率 P_hat(mu)（NUMERICS §7.7.3）
    double P_hat_mu = 0.5 * P_hat * (1.0 + 1.5 * mu);

    return P_hat_mu;
}
```

- **レジスタ**: ~8（tau, omega, eps_prime, beta, P_hat, P_hat_mu + 一時変数）
- **インライン**: `__forceinline__`。R8 の境界交差処理内で条件付き呼び出し（DDMCセルへの遷移時のみ）

### 6.4.0a1 compute_P_standard（標準変換確率、Densmore 2007 Eq.34）

```cuda
__device__ __forceinline__ double compute_P_standard(
    double sigma_R,    // Rosseland不透明度 [cm⁻¹]（新セル）
    double delta_x,    // 面法線方向代表長 [cm] = vol[new_cell] / face_area[new_cell*n_faces+entry_face]
    double mu          // 入射角コサイン |Ω̂·n̂|
) {
    double tau = sigma_R * delta_x;
    double lambda = 0.7104;  // Milne 外挿長 [無次元]
    if (tau < 1e-30) return fmin(1.0, 1.0 + 1.5 * mu);  // τ→0: 分母→0、P→1
    double P_std = 4.0 / (3.0 * tau + 6.0 * lambda) * (1.0 + 1.5 * mu);
    return fmin(P_std, 1.0);
}
```

- **レジスタ**: ~4（tau, lambda, P_std + 一時変数）
- **インライン**: `__forceinline__`。`emissivity_preserving=False` 時にR8が呼び出す（NUMERICS §7.7.1）

### 6.4.0b SoA ロード/ストア __device__ 関数

```cuda
__device__ void load_particle_from_SoA(int idx, const PhotonPool& pool,
    double& pos_r, double& pos_z, double& dir_r, double& dir_z, double& dir_phi,
    double& energy, double& weight, double& time_remain, double& birth_energy,
    int8_t& sign, uint64_t& global_id, uint32_t& rng_counter, int& cell_id, int& group_id, int& mode) {
    pos_r = pool.pos_r[idx];  // __ldg for read-only arrays
    pos_z = pool.pos_z[idx];
    dir_r = pool.dir_r[idx];
    dir_z = pool.dir_z[idx];
    dir_phi = pool.dir_phi[idx];
    energy = pool.energy[idx];
    weight = pool.weight[idx];
    time_remain = pool.time_remain[idx];
    birth_energy = pool.birth_energy[idx];
    sign = pool.sign[idx];
    global_id = pool.global_id[idx];
    rng_counter = pool.rng_counter[idx];
    cell_id = pool.cell_id[idx];
    group_id = pool.group_id[idx];
    mode = pool.mode[idx];
    // 16 SoA中、aliveを除く15可変フィールドをレジスタにロード
    // Ballot Refill 後は非連続アクセスとなるが、cell sort により
    // 初期状態では隣接スレッドが隣接粒子を処理するため coalesced
}

// store_particle_to_SoA は load の逆操作（同一15可変フィールド）
// alive フラグは Persistent Warp の active 変数で管理するため、
// load/store 対象には含まない。store 時に alive=0/1 を別途書き込む。
```

- **`__forceinline__`**: load/store 関数にも `__forceinline__` を付与し、関数呼び出しオーバーヘッドを排除する

### 6.4.1 境界条件処理

R8 疑似コード内の境界条件処理の完全仕様（上記 Phase 2 内にインライン記載済み）。

境界タイプは `get_neighbor_cell()` の戻り値（負値）で判定する:
- `new_cell = -1`: **VACUUM** — 粒子脱出。`E_escape[group_id]` に atomicAdd
- `new_cell = -2`: **REFLECT** — 鏡面反射。crossing_face の法線 \(\hat{n}\) に対し \(\hat\Omega \leftarrow \hat\Omega - 2(\hat\Omega\cdot\hat{n})\hat{n}\)。
    一般辺法線を使用（`face_geometry_2d.cuh`）。矩形メッシュでは dir_r or dir_z 反転と等価
- `new_cell = -3`: **AXIS** (R=0) — R方向反転 + φ方向反転。`pos_r = abs(pos_r)` で R≥0 保証
- `new_cell = -4`: **MARSHAK** — 外部放射源境界。脱出として処理（Marshak入射粒子は R13 で別途生成）
- `new_cell = -5`: **パーティション境界** — emigrant buffer に追加し、MPI exchange で移動

### 6.4.2 セル境界距離計算（2D RZ 四辺形セル）

> **設計方針**：ALE rezone 後の歪んだ四辺形セルを正しく扱うため、
> 辺端点座標をそのまま使用する**一般辺交差**を実装する。
> ノード座標の平均による軸揃え近似（定数R面/定数Z面）は使用しない。
> 辺端点からの法線 \(\hat{n} = (\Delta Z, -\Delta R)/L\) は `face_geometry_2d.cuh` の
> `FaceGeom2D` 構造体で計算し、境界距離・mu計算・push-off・反射で共有する。

```
// === compute_boundary_distance ===
// __device__ 関数: imc_transport (R8) および ddmc_event_loop (R9) から呼び出し
//
// 入力: pos=(R,Z), dir=(dR,dZ), cell c の4頂点 (R_k, Z_k), k=0,1,2,3 (反時計回り)
// 出力: (s_min, crossing_face_id)  — 最小正距離と交差面ID
//
// **面インデックス規約（物理面）**: 0=R_left, 1=R_right, 2=Z_bottom, 3=Z_top
// 辺→物理面マッピング（反時計回り頂点 0,1,2,3 に対応）:
//   f=0 (R_left):   edge 3→0  — P1=vertex3(R_lo,Z_hi), P2=vertex0(R_lo,Z_lo)
//   f=1 (R_right):  edge 1→2  — P1=vertex1(R_hi,Z_lo), P2=vertex2(R_hi,Z_hi)
//   f=2 (Z_bottom): edge 0→1  — P1=vertex0(R_lo,Z_lo), P2=vertex1(R_hi,Z_lo)
//   f=3 (Z_top):    edge 2→3  — P1=vertex2(R_hi,Z_hi), P2=vertex3(R_lo,Z_hi)
// **重要**: 辺の反時計回り順序（0→1→2→3→0）と物理面順序は異なる。
// crossing_face_id は物理面規約（0=R_left,...）で返す。get_neighbor_cell と同一規約。
//
// 各面 f で:
//   面方程式: r(t) = P1 + t*(P2 - P1), t ∈ [0, 1]
//   光線方程式: p(s) = pos + s * dir, s > 0
//
//   連立方程式:
//     pos_R + s * dir_R = P1_R + t * (P2_R - P1_R)
//     pos_Z + s * dir_Z = P1_Z + t * (P2_Z - P1_Z)
//
//   det = dir_R * (P1_Z - P2_Z) - dir_Z * (P1_R - P2_R)
//   if |det| < 1e-30: 光線は面に平行 → skip
//   s = ((P1_R - pos_R)*(P1_Z - P2_Z) - (P1_Z - pos_Z)*(P1_R - P2_R)) / det
//   t = ((P1_R - pos_R)*dir_Z - (P1_Z - pos_Z)*dir_R) / det
//
//   if s > eps_geom AND 0 <= t <= 1: 有効な交差
//     s_candidates[f] = s
//
// s_min = min(s_candidates)
// crossing_face_id = argmin(s_candidates)
//
// eps_geom = 1e-12 [cm]（NUMERICS §6.3.2 準拠。絶対値定数。メッシュサイズ非依存）
// 光線がコーナー点を通過する場合 (複数面で t=0 or t=1):
//   最小 s の面を採用（タイブレイク: 面 ID が小さい方）
```

- **レジスタ**: ~12（4面の s_candidates + P1/P2 座標 + det/s/t 一時変数）
- **分岐**: 4面ループ、各面で parallel/skip 判定。構造格子では全面が有効な場合がほとんど

**1D_SPH 版（球殻セル）**:
```
// === compute_boundary_distance (1D_SPH) ===
// セル c = [r_lo, r_hi] の球殻。粒子位置 r、方向余弦 μ = cos(θ)
// 球面 r=R との交差距離: s² + 2rμs + (r² - R²) = 0
// 判別式: D = (rμ)² - (r² - R²) = r²(μ²-1) + R²
//
// **判別式ガード**（NUMERICS §6.3.2 準拠）:
//   D < -eps_geom² → 交差なし（接線近傍の浮動小数点誤差）
//   D < 0 かつ D ≥ -eps_geom² → D = max(D, 0) にクランプ
//
// **安定二次解法（q-form）**（NUMERICS §6.3.2、桁落ち回避）:
//   q = -(rμ + sign(rμ) × sqrt(D))
//   s1 = q,  s2 = (r² - R²) / q
//   q == 0 の場合: s = sqrt(D)
//   正の実数解のうち最小の s > eps_geom を s_bdry とする
//
// 外側面 (r_hi): 常に正の解が存在（r < r_hi）
// 内側面 (r_lo): μ < 0（内向き）の場合のみ正の解が存在
// 最内セル (r_lo=0): 内側面交差なし
//
// s_min = min(valid s_lo, s_hi)
// crossing_face_id: 0=inner, 1=outer
```
- **レジスタ**: ~10（r, μ, r_lo, r_hi, D, q, s1, s2, s_lo, s_hi）
- **分岐**: 2面のみ。内側面交差は μ < 0 の場合のみ有効

### 6.4.3 隣接セル探索

```
// === get_neighbor_cell ===
// __device__ 関数: imc_transport (R8) から呼び出し
//
// 構造格子の場合（v1.0）:
//   面インデックス規約: 0=R_left, 1=R_right, 2=Z_bottom, 3=Z_top
//   セル (i,j) のインデックス: c = i * nz + j  (i: R方向, j: Z方向)
//
//   get_neighbor_cell(c, face):
//     i = c / nz;  j = c % nz;
//     switch (face):
//       case 0 (R_left):   if (i == 0)      return bc_code(face); else return (i-1)*nz + j;
//       case 1 (R_right):  if (i == nr-1)    return bc_code(face); else return (i+1)*nz + j;
//       case 2 (Z_bottom): if (j == 0)       return bc_code(face); else return i*nz + (j-1);
//       case 3 (Z_top):    if (j == nz-1)    return bc_code(face); else return i*nz + (j+1);
//
//   bc_code(face): face_bc_type[face]（0=VACUUM,1=REFLECT,2=MARSHAK,3=AXIS）から負の戻り値を生成
//     face_bc_type==0 (VACUUM)  → return -1
//     face_bc_type==1 (REFLECT) → return -2
//     face_bc_type==2 (MARSHAK) → return -4
//     face_bc_type==3 (AXIS)    → return -3（R=0軸: dir_phi反転+pos_r=abs、§6.3.2）
//     パーティション境界（MPI隣接）→ emigrant buffer に追加、return -5
//   注: DDMC R3はリーク係数の観点でAXIS=REFLECTだが、IMC R8ではAXIS固有のφ反転が必要
```

- **レジスタ**: ~5（i, j, face, result + 一時変数）
- **分岐**: switch文は4分岐だが、crossing_face は常に1つのみ

**1D_SPH 版**:
```
// === get_neighbor_cell (1D_SPH) ===
// 面インデックス規約: 0=inner, 1=outer
// セルインデックス: c = 0, 1, ..., nr-1（r昇順）
//
// get_neighbor_cell(c, face):
//   switch (face):
//     case 0 (inner): if (c == 0)     return bc_code(0); else return c - 1;
//     case 1 (outer): if (c == nr-1)  return bc_code(1); else return c + 1;
//
// bc_code(face): face_bc_type[face]（0=VACUUM,1=REFLECT,2=MARSHAK,3=AXIS）から負の戻り値を生成（§6.4.3 2D版と同一規約）
//   典型: inner(c=0) → face_bc_type=1(REFLECT) → return -2、outer(c=nr-1) → face_bc_type=0(VACUUM) → return -1 or face_bc_type=2(MARSHAK) → return -4
//   1D_SPHでは AXIS(3) は使用しない（inner は常に REFLECT）
```
- **レジスタ**: ~3（c, face, result）

### 6.5 R9: ddmc_event_loop

```cpp
__global__ __launch_bounds__(128, 16)
void ddmc_event_loop(
    // PhotonPool SoA (in/out) — DDMCモード粒子のみ
    double* __restrict__ pos_r,              // [n_particles] DDMC→IMC変換時に書き込み（§7.7.2 位置サンプル）
    double* __restrict__ pos_z,              // [n_particles] DDMC→IMC変換時に書き込み
    double* __restrict__ dir_r,              // [n_particles] DDMC→IMC変換時に書き込み（§7.7.2 方向サンプル）
    double* __restrict__ dir_z,              // [n_particles] DDMC→IMC変換時に書き込み
    double* __restrict__ dir_phi,            // [n_particles] DDMC→IMC変換時に書き込み
    double* __restrict__ energy,
    double* __restrict__ time_remain,
    uint32_t* __restrict__ rng_counter,
    int32_t* __restrict__ cell_id,
    uint16_t* __restrict__ group_id,
    uint8_t* __restrict__ mode,
    uint8_t* __restrict__ alive,
    // DDMC coefficients (read-only)
    const double* __restrict__ sigma_a_eff,  // [n_cells × G]
    const double* __restrict__ Sigma_leak,   // [n_cells × n_faces × G]
    const double* __restrict__ Sigma_out,    // [n_cells × G]
    const double* __restrict__ Sigma_leak_bdry, // [n_cells × G] 境界リーク
    const uint8_t* __restrict__ ddmc_mode,   // [(n_cells+n_ghost) × G]（R9は隣接セルのddmc_modeを参照: DDMC→IMC変換判定 §7.7）
    const int8_t* __restrict__ face_bc_type, // [n_faces_boundary] 境界面タイプ（R3と同一規約、ARCHITECTURE §5.2 bc_type_rad 準拠）
                                              // 0=VACUUM, 1=REFLECT, 2=MARSHAK, 3=AXIS（§6.4.3）。DDMCではAXIS=REFLECTと同一処理（リーク=0）。PARTITION面はface_bc_typeに含まない
    // Cell geometry
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    // Tally
    double* __restrict__ rad_dep,
    double* __restrict__ rad_E_tally,
    // 注: rad_mom_dep は R9 シグネチャに含めない。DDMC 運動量沈着は R10 後のポストプロセスカーネル
    // ddmc_momentum_postprocess で算出する（NUMERICS §7.8: residence estimator → φ → 面フラックス → p_i）。
    // R9 イベントループ中には rad_mom_dep に書き込まない。§9 Phase 4 のシーケンスを参照。
    double* __restrict__ E_escape,
    double* __restrict__ E_numerical_loss,   // [1] MAX_EVENTS到達/退化セル等での粒子消滅エネルギー累積（atomicAdd、Phase 0 で初期化済み）
    // Parameters
    double dt, int n_ddmc,
    int nr, int nz, int n_groups, int n_faces,
    int dim,                                  // 空間次元数: 1D_SPH=1, 2D_RZ=2。rad_mom_dep[cell*dim+d] のストライドに使用
    int ir_start, int jz_start,              // ローカル領域オフセット（DDMC emigrant のグローバル座標計算に必要）
    uint64_t* __restrict__ global_id,        // [n_particles] RNG seed復元用（NUMERICS §12.7.1）
    uint64_t step,                            // 現在ステップ番号（curand_init subsequence、NUMERICS §12.7.1）
    uint64_t user_seed,                       // Main.seed（curand_init: global_id ^ user_seed, NUMERICS §12.7.1）
    int interface_exit_distribution,          // DDMC→IMCリーク方向分布: 0=COSINE（既定, P(μ)=2μ）, 1=HALF_ISOTROPIC（P(μ)=1）。SPECIFICATION §6.4.5, NUMERICS §7.7
    DeviceErrorFlags* error_flags
);
```

- **block**: 128, **grid**: `(n_ddmc+127)/128`
- **SoA オフセット規約**: ホスト側は全SoAポインタを `+n_imc` でオフセットして R9 に渡す（例：`pos_r + n_imc`）。R9 内では `thread_idx = blockIdx.x * blockDim.x + threadIdx.x` を直接添字として使用する。これにより R8 と R9 が同一 SoA 配列の異なるスライスを独立に処理する
- **処理**: IMCより単純（位置・方向の追跡不要）
  ```
  // Census粒子の time_remain 再装填（R8 と同一ロジック）
  if (time_remain <= 0.0) time_remain = dt;  // census粒子の time_remain 再装填
  int n_events = 0;
  const int MAX_EVENTS_DDMC = 100000;  // R8(10000)より大きい：DDMCはイベント処理が軽量で多数のイベントが正常
  while (alive && time_remain > 0):
      // 無限ループ検出（R8 と同一パターン: ループ内冒頭で判定）
      n_events++;
      if (n_events >= MAX_EVENTS_DDMC):
          atomicExch(&error_flags->infinite_loop, 1)
          atomicAdd(&E_numerical_loss[0], E)  // 残余エネルギーを数値的喪失に計上
          alive = 0; break
      Σ_tot = σ_{a,eff} + Σ_out + Σ_leak_bdry
      if (Σ_tot <= 1e-30):
          // ゼロ/subnormal 率セル: 即census（数値安全策）
          // Σ_tot が subnormal（~1e-310）の場合、-log(ξ)/(c×Σ_tot) が overflow → inf
          // となりうるため、Σ_tot ≤ 1e-30 でガード（NUMERICS §11.3 不透明度フロア相当）
          atomicExch(&error_flags->ddmc_sigma_tot_zero, 1)  // §0.6 準拠：host 側で WARNING 出力
          warp_tally_accumulate(rad_E_tally, cell, g, E × (c_light × time_remain))  // residence estimator
          time_remain = 0  // census: alive=1 のまま（R12 roulette 対象、次ステップで time_remain 再装填）
          break
      Δt_evt = -log(ξ) / (c_light × Σ_tot)

      if (Δt_evt >= time_remain):
          // Census: イベント発生前にステップ終了
          warp_tally_accumulate(rad_E_tally, cell, g, E × (c_light × time_remain))  // residence estimator（×c_light で [erg·cm] 単位）
          time_remain = 0  // census: alive=1 のまま（R12 roulette 対象、次ステップで time_remain 再装填）
          break

      time_remain -= Δt_evt
      // **Census近傍スナップ**（R8 と同一、浮動小数点桁落ち防止）:
      // time_remain ≈ Δt_evt の場合、減算で微小正値が残りイベントループが空転する。
      // ε_cen = 1e-10 × dt（NUMERICS §6.4 準拠） で判定し、強制census化する。
      if (time_remain < 1e-10 * dt):
          time_remain = 0.0;  // → while条件で脱出 → census
      warp_tally_accumulate(rad_E_tally, cell, g, E × (c_light × Δt_evt))  // residence estimator（×c_light で [erg·cm] 単位）
      // v1.0: tally_mode="warp"（既定）では DDMC にも Stage 1（warp-level __match_any_sync）を適用
      // NUMERICS §10.3.3「適用カーネル：imc_transport（R8）、ddmc_event_loop（R9）の両方に適用」に準拠
      // DDMC の 1 スレッド 1 粒子モデルでは衝突頻度が IMC より低いが、
      // セルソートと Stage 1 は不可分（NUMERICS §6.5「不可分性」参照）
      // Stage 2（ブロック集約）は Phase B で追加予定
      // IMC と DDMC で同一の tally infrastructure を共有する設計

      // Event selection
      r = ξ × Σ_tot
      if (r < σ_{a,eff}):
          // 吸収
          warp_tally_accumulate(rad_dep, cell, g, E);  alive = 0; break
      elif (r < σ_{a,eff} + Σ_out):
          // 近傍リーク → CDF サンプリングでリーク面を選択（NUMERICS §7.5）
          // 残余 r' = r - σ_{a,eff} を内部面リーク係数の累積和で走査:
          //   f* = min{f ∈ F_int : Σ_{f'≤f} Σ^leak[cell,f',g] ≥ r'}
          //   **フォールバック**: 浮動小数点丸め誤差で全面走査後も条件未達の場合、
          //   最後の正リーク係数を持つ面を選択する（到達不能の防御ガード）
          double r_residual = r - σ_{a,eff};
          double cdf = 0.0;
          int selected_face = -1;
          int last_positive_face = 0;  // フォールバック用: 最後の正リーク係数を持つ面
          for (int f = 0; f < n_faces; f++):
              double Sf = Sigma_leak[c * n_faces * G + f * G + g]
              if (Sf > 0.0): last_positive_face = f
              cdf += Sf
              if (cdf >= r_residual):
                  selected_face = f; break
          if (selected_face < 0):
              selected_face = last_positive_face  // 浮動小数点丸め誤差で条件未達の場合
          new_cell = neighbor[c, selected_face]
          if (neighbor is on different rank):
              // rank境界越え → モード判定を受信rankに延期（NUMERICS §12.3.3）
              // alive=1 維持（ARCHITECTURE §7.1.3 統一契約準拠）
              cell_id = -(100 + selected_face);  // face エンコード: R8 IMC と同一規約。P5 が face=-(cell_id)-100 で復元
              // DDMC粒子は pos_r/pos_z が NaN のため、P6 の位置ベースセル再同定が使えない。
              // **解決策**: pos_r/pos_z にソースセルのグローバル座標を一時格納する。
              // DDMC粒子は輸送に位置を使用しないため、この転用は安全。
              // P6（受信側）で宛先セルを算術決定後、pos_r/pos_z = NaN に復元する。
              int i_local = c / nz;  int j_local = c % nz;
              pos_r = (double)(i_local + ir_start);  // グローバル R インデックス
              pos_z = (double)(j_local + jz_start);  // グローバル Z インデックス
              // group_id は変更しない（ARCH統一契約: リーク面は cell_id にエンコード済み）
              break  // alive=1 のまま（§9 notes: "alive=1, cell_id<0 で R8/R9 を終了"）
          if (ddmc_mode[new_cell,g]):
              cell_id = new_cell  // DDMC継続（同一rank内）
          else:
              // DDMC→IMC変換 (NUMERICS §7.7.2)（同一rank内）
              // 位置: リーク面上でサンプル
              //   2D_RZ: R-重み付きサンプル
              //     辺端点 V_k, V_{k+1} に対し t = (-R_k + sqrt(R_k² + ξ(R_{k+1}² - R_k²))) / (R_{k+1} - R_k)
              //     |R_{k+1}-R_k| < ε_R の場合は t = ξ（一様退化）
              //   1D_SPH: 球面 r=r_f 上で等方位置サンプル（cos(θ)=2ξ-1, φ=2πξ）
              //   位置 r = V_k + t × (V_{k+1} - V_k)
              sample_position_on_face(selected_face, ...)
              // 方向: IMC側半空間へリーク角度分布でサンプル
              if (interface_exit_distribution == COSINE):
                  // cosine 分布 P(μ)=2μ（既定、物理的に正しい拡散流束分布）
                  //   μ = sqrt(ξ_1), φ = 2πξ_2
                  //   Ω = μ n̂ + sqrt(1-μ²)(cos(φ) û + sin(φ) ŵ)
                  sample_isotropic_half_space(face_normal, ...)
              else:  // HALF_ISOTROPIC
                  // 半等方分布 P(μ)=1（簡略化。μ = ξ_1）
                  //   Ω = μ n̂ + sqrt(1-μ²)(cos(φ) û + sin(φ) ŵ)
                  sample_isotropic_half_space_uniform(face_normal, ...)
              //   n̂ = 面外向き法線, û = 面接線, ŵ = n̂ × û
              //   方位角 φ_P は [0,2π) から一様サンプル（DDMC粒子は方位角を持たないため）
              cell_id = new_cell  // **必須**: 粒子をIMCセル（リーク先）に配置。
                                  // cell_id を更新しないと、粒子が DDMCセルに IMC モードで残り、
                                  // R8 が DDMCセルのジオメトリで境界距離を計算する。
                                  // 粒子位置は面上にサンプル済み → new_cell 内。
              mode = IMC; break  // 次ステップの R8 で処理（v1.0: 同一ステップ内再処理なし）
      else:
          // 境界リーク（Σ_leak_bdry イベント）
          // **境界面選択**（NUMERICS §7.4 + §7.5 準拠）:
          //   Σ_leak_bdry はセルの全境界面リーク係数の合計。
          //   コーナーセル（複数の境界面を持つ）では面選択が必要:
          //   残余 r' = r - σ_a_eff - Σ_out を境界面リーク係数の累積和で走査し、リーク面を特定。
          //   **v1.0簡略化**: 境界面のリーク係数はR3で Σ_leak_bdry に合算され個別値は保持されない。
          //   VACUUM境界脱出は面に依存しない（E_escape計上のみ）ため、v1.0ではコーナーセルでも
          //   最初のVACUUM/MARSHAK境界面を選択する。多面Marshak BCの正確な面選択は将来版で対応。
          // 面タイプ判定（§11.4 準拠）:
          //   VACUUM/MARSHAK: atomicAdd(&E_escape[group_id], E); alive = 0; break
          //   REFLECT/AXIS: 定義上到達不能（Σ_leak=0 のため CDF で選択されない）。
          //     万一到達した場合は error_flags->ddmc_reflect_leak = 1 を設定し、
          //     粒子を吸収扱い（rad_dep 加算、alive=0）で安全に処理する
          //     （NUMERICS §11.4「DeviceErrorFlags::ddmc_reflect_leak」参照）
          if (face_is_vacuum_or_marshak):
              atomicAdd(&E_escape[group_id], E);  alive = 0; break
          else:  // REFLECT/AXIS — 到達不能パス
              atomicExch(&error_flags->ddmc_reflect_leak, 1)  // §0.6 準拠: atomicExch で書き込み
              warp_tally_accumulate(rad_dep, cell, g, E);  alive = 0; break
  // MAX_EVENTS_DDMC 超過ガード: ループ内冒頭に移動済み（R8 と同一パターン）
  // ポストループガードを廃止: while 内の break で正常終了した粒子
  // （census, DDMC→IMC変換, emigrant）が n_events == MAX_EVENTS_DDMC の場合に
  // 誤って kill されるバグを防止
  ```
- **レジスタ**: ~30（位置・方向不要で低い）。`__launch_bounds__(128, 16)` → max 32 reg → 100% occupancy
- **ワープ発散**: IMCより低い（イベント処理が単純）。3分岐（吸収/リーク/境界リーク）のうち、リーク後の DDMC→IMC 変換パスのみ追加処理あり
- **DDMC→IMC変換**: `sample_position_on_face` + `sample_isotropic_half_space` で方向を初期化（NUMERICS §7.7.2）。変換された粒子は `mode=IMC` に書き戻し、次の R8 呼び出しで処理される

### 6.6 ~~R14: mode_partition~~ → R7 に吸収

> **v1.0設計変更**: R14（IMC/DDMC分離）は R7 Composite Key Sort に吸収された（§0.5）。
> 合成キーの bit 30 = mode flag により、ソート後に IMC粒子が先頭、DDMC粒子が後方に
> 自動配置される。`n_imc` と `n_ddmc` は合成キー生成カーネル内で atomic count により算出。
> 独立の CUB `DevicePartition::Flagged` 呼び出しと15可変配列（16 SoA中）の個別 gather は不要となった。
> **例外**: `particle_sort_by_cell=False` フォールバック時は CUB `DevicePartition::Flagged` を使用する
> （NUMERICS §6.5 フォールバックパス参照。mode sync + NaN化 + R7b resample も必須）。

**IMC/DDMCモード分離の必須性**（根拠は不変）：
- IMC Persistent Warp（§6.4）はIMC粒子のみのwork queueを必要とする
- IMC（~60 reg）と DDMC（~30 reg）のレジスタ要件が大きく異なり、
  分離によりDDMCは `__launch_bounds__(128, 16)` で 100% occupancy を達成
- ICF問題ではDDMCセルが空間的にまとまっている（中心の高密度領域）ため、
  合成キーソート後のメモリレイアウトは局所的（セル順でさらにモード分離済み）

---

---

## 4. 元 `docs/CUDA_KERNELS.md` の §7.1 U1: source_injection の設計（輻射の沈着の注入と二重計上防止プロトコルを含む）


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

---

## 5. 元 `docs/CUDA_KERNELS.md` の §7.4 U7: cell_search_after_rezone


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

---

## 6. 元 `docs/CUDA_KERNELS.md` の §7.7 U9: compute_opacities


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
- **1D の実装**（`multimat_opacity_1d.cuh`、FLD と S\(_N\) が共有）：各材料を部分密度で評価し、`Materials.opacity_mix_rule`
  （既定 `"linear_mass"`：Planck も Rosseland も質量の重みの線形平均。`"harmonic_mass_R"` で Rosseland を調和平均（各 \(\kappa_{R,\alpha}\ge 10^{-20}\) cm²/g）、
  `"max"` で材料の最大値）で混ぜる。床は \(\sigma \ge \rho\,\kappa_\mathrm{floor}\) で、\(\kappa_\mathrm{floor}\) は
  FLD が `Radiation.multigroup_diffusion.opacity_floor`（既定 1e-6）、S\(_N\) が `Radiation.sn_transport.opacity_floor`
  （既定 1e-100）。FLD の同じ値は面の拡散係数の組み立てでは \(\sigma_R\) の床 [cm⁻¹] として使う（空の極限で \(D\) が
  発散しないため）。FLD は物理散乱を使わない。以下は設計時の記述
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

---

## 7. 元 `docs/CUDA_KERNELS.md` の §8.2 P5: emigrant_detect_pack と §8.3 P6: immigrant_unpack_merge


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

---

## 8. 元 `docs/CUDA_KERNELS.md` の §7.2 U3 energy_budget の E_rad（census エネルギー）と E_escape の項

  4. **E_rad**: 放射場エネルギー = **E_census = Σ_p E_p**（alive粒子のエネルギー合計、**R12 Russian roulette 適用後**）。difference path では `E_census = Σ U_ref + Σ_p sign[p]E_p`。
     **注意**: track-length推定量 `rad_E[c,g]×V_c` ではなくcensus energy を使用する（NUMERICS §10.2: ΔE_rad = E_census^{n+1} - E_census^{n}）。
     legacy path の E_census は **別途 CUB `DeviceReduce::Sum`** で粒子配列 `energy[0..n_alive-1]` を **alive==1 フィルタ付き**で集約して算出する（粒子レベル reduction、U3 カーネル外で実行。n_alive は R12 後の第2 R7 ソートで確定。ALE U7 が alive=0 にした粒子を除外するためフィルタ必須 — §9 Phase 6 参照）。difference path は deterministic reference reservoir と signed residual 粒子和を加える。
     `rad_E` は inject_sources（U1）の物質結合にのみ使用し、エネルギー収支には使用しない
  5. **E_escape**: 境界脱出累積エネルギー（群別 `E_escape[g]` を集約して使用）

---

## 9. 元 `docs/CUDA_KERNELS.md` の §7.3 U4 cfl_reduction の Step 3: dt_rad（`compute_dt_rad`）

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

---

## 10. 元 `docs/CUDA_KERNELS.md` の §9 カーネル起動シーケンスの Phase 4: Radiation R(Δt)（IMC/DDMC の起動列）

```text
═══════════════════════════════════════════════════════
 Phase 4: Radiation R(Δt) — 最も計算量が大きい
                               // radiation.enabled=False の場合は Phase 4 全体をスキップ（rad_dep=0, dt_rad=∞）
═══════════════════════════════════════════════════════
  if (!radiation.enabled): goto Phase 5  // PhotonPool/opacity テーブル未確保時のアクセス防止
  --- pre-radiation 準備 ---
  [MPI] halo_exchange(Te, rho, zbar, vol, ell_ddmc)      // double scalar stride=1
  [MPI] halo_exchange(volFrac[n_mat])                    // double stride=n_mat（多材料混合則用）
  [MPI] halo_exchange(face_area[n_faces])                // double stride=n_faces（R2/R3用）
                                                // NUMERICS §12.2.2 行7：U9 がゴーストセルの opacity を計算するために必要
                                                // volFrac: 多材料時に U9 の混合則計算でゴーストセル体積分率が必要
                                                // vol, face_area, ell_ddmc: R2/R3 がゴーストセルを処理するため必要（H7 出力は n_cells のみ）
  U9:  compute_opacities               // ARCHITECTURE §4.7：σ_a,σ_R,σ_t 前計算（R1 の入力）
  // 注: R8 の compute_P_hat は面別代表長 Δx_m = vol[c] / face_area[c,f] を
  //     crossing_face から **インラインで** 算出する（§6.4 疑似コード参照）。
  //     セル平均 delta_x の事前計算は不要。R2 は ell_ddmc（H7出力）を使用する。
  --- per-step カウンタ/タリー初期化 ---
  cudaMemsetAsync: source_total=0, count_imc=0, count_ddmc=0,
    global_work_counter=0, emigrant_count=0, per_dest_count[8]=0,
    laser_dep[n_cells]=0,           // U1 二重計上防止（Phase 3 の laser_dep が Phase 4 U1 で再加算されないよう）
    rad_dep[n_cells×G]=0, rad_E_tally[n_cells×G]=0,   // rad_dep: R4b/R8/R9/R12 がatomicAdd/writeする沈着先（R10は使用しない）
    rad_mom_dep[n_cells×dim]=0,                       // R8/R9 が atomicAdd する運動量沈着先（NUMERICS §7.8、ARCHITECTURE §5.2）
    E_escape[G]=0,
    ddmc_candidate[(n_cells+n_ghost)×G]=0,  // R2 は candidate=1 のみ書き込み、非候補に 0 を明示書きしないためゼロ初期化必須
    ddmc_mode[(n_cells+n_ghost)×G]=0,  // R3が書く前にゼロ初期化（R2は ddmc_candidate を出力、R3が ddmc_mode を最終確定。ゴーストセル含む。純IMC時も安全）
    leak_coeff_face[n_cells×n_faces×G]=0,  // R3 は ddmc_candidate==1 セルのみ書き込み。非候補セルの stale 値を防御的にゼロ初期化
    leak_total_int[n_cells×G]=0,            // 同上。R3 が非候補セルに書き込まないため
    leak_total_bdry[n_cells×G]=0            // 同上
    // 注: E_floor_injected, clamp_count, E_numerical_loss は Phase 0 で初期化済み（Phase 1-3 の U2/U1 が累積するため、Phase 4 では初期化しない）
  R1:  compute_fleck_factor
  if (cfg.radiation.ddmc_enabled):          // 純IMC構成（ddmc_enabled=False）ではR2/R3/R3bをスキップ
    [MPI] halo_exchange(f_fleck)         // R2 がゴーストセルの f_fleck を参照するため必要（R1 は n_cells のみ計算）
    R2:  ddmc_mode_judge (ω, τ, P制約でDDMC候補を抽出)
    --- R3 前処理: リーク係数の入力準備 ---
  if (leak_stencil == "9_kershaw"):
    // C2 を D_g = 1/(3σ_{R,g}) で G 回呼び出し → stencil[(n_cells+n_ghost) × 9 × G]
    // **ゴーストセル含む**: R3 パスA がゴーストセルを処理するため、C2 もゴーストセル含みで起動
    // apply_mmatrix_repair=False（R3 が修復前 raw 係数で M-matrix 判定を行うため）
    // **注**: C2 出力は 9点係数（k=0:C, k=1-8:off-diag）。R3 は indices 1-8 のみ使用（center は無視）
    for g in 0..G-1:
      C2: kershaw_stencil_build(D_g=1/(3σ_{R,g}), apply_mmatrix_repair=False, grid=((n_cells+n_ghost)+255)/256) → stencil[...,g]
  elif (leak_stencil == "4"):
    // face_sigma_R 生成（R3 "4" 前処理）:
    //   内部面: owner/neighbor セルの σ_R_cell から face-average を構築
    //   境界面: owner セルの σ_R_cell をそのまま使用
    //   入力: sigma_R[(n_cells+n_ghost) × G]（U9 出力）
    //   出力: face_sigma_R[(n_cells+n_ghost) × n_faces × G]
    //   block=256, grid=((n_cells+n_ghost)+255)/256（ゴーストセル含む）
    compute_face_sigma_R → face_sigma_R[(n_cells+n_ghost) × n_faces × G]
  R3:  ddmc_leak_coeff_kershaw（"9_kershaw"）/ ddmc_leak_coeff_face（"4"）（リーク係数計算 + M-matrix判定 + ddmc_mode最終確定）
       // "9_kershaw" は geometry==2D_RZ のみ（NUMERICS §7.3.3, Appendix A）。1D_SPH は "4" を使用。
       // leak_stencil の妥当性は namelist validation で保証済み（SPECIFICATION §6.4）。
  R3b: ddmc_interface_correct (DDMC-IMCインターフェースセルのリーク修正。R3でddmc_modeが確定した後に起動。§7.3.5)
  if (cfg.radiation.imc.difference.enabled && LTE nonlinear source path):
    [Host/GPU] compute/load E_ref_start = W * a_eV * Te^4 * b_g(Te)
    [Host] PR5 census residualization:
      U_phys_old = U_ref_old + Σ_p sign_p E_p
      target residual = U_phys_old - E_ref_start * V
      scale/rebuild existing census bins exactly; create one residual particle for empty nonzero bins
    R4b: preseed_reference_absorption   // rad_dep += c * sigma_a_eff * E_ref_start * V * dt
    if (difference.face_transport):
      R4c: reference_face_transport_1d   // deterministic ΔU_ref_face → U_ref_end/E_ref_avg; no rad_dep write
      [D2H] U_ref_end → previous-reference reservoir
  R4:  compute_source_energy            // legacy: source_E=cσB Vdt; difference PR4: source_E=cσ(B-E_ref)Vdt; source_total=Σ|source_E| (device atomicAdd)
  [SYNC] → [D2H] source_total          // R5 のホスト起動引数に必要
  [MPI] MPI_Allreduce(SUM, source_total) → source_total_global  // E_avg = source_total_global / N_p_global（R8/R12 のRussian roulette閾値がランク間で一致するため必要）
  R5:  source_particle_count(source_total) → CUB ExclusiveSum → offset[n_cells×G+1]
  [SYNC] → [D2H] n_new_particles       // = offset[n_cells×G]（prefix sum末尾値。R6/R13 のグリッドサイズに必要）
  [MPI] MPI_Allreduce(SUM, n_new_particles) → N_p_global  // E_avg = source_total_global / N_p_global
  [Host: E_avg = (N_p_global > 0) ? source_total_global / N_p_global : T_floor * eV_to_erg]
  // 分母ゼロガード（NUMERICS §6.3.4）: ソース粒子なし → E_avg = T_floor×eV_to_erg
  // この場合 Russian roulette は事実上不活性（census由来粒子は E ≫ w_cutoff × E_avg）
  [Host: n_marshak_total = cfg.radiation.boundary.marshak_particles]  // namelist 由来の定数（NUMERICS §8.2, ARCHITECTURE §4.5.2）。MPI分配前の全ランク合計値
  [Host: pool capacity check]          // n_alive + n_new_particles + n_marshak_total > pool_capacity の場合、
                                        // cudaMalloc でプール拡張（NUMERICS §6.3.1）。
                                        // n_marshak_total（全ランク合計）で保守的に検査。per-rank 分配は後段で算出。
                                        // R6/R13 が OOB 書き込みしないことをホスト側で保証する。
                                        // pool_capacity は 1.5 × 初期容量 で確保し、不足時は 2倍拡張
  // --- Marshak 粒子数配分（並列時：NUMERICS §8.2 step 2 + §12.5 準拠）---
  // **重要**: n_marshak_local は MPI_Exscan の入力に必要なため、Exscan より前に算出すること。
  // [Host: A_local = Σ_{owned faces f} face_area[f] for Marshak BC faces]
  // [MPI] MPI_Allreduce(SUM, A_local) → A_global  // 全Marshak面の面積合計（並列時に必須）
  // [Host: if (A_global <= 0.0) { n_marshak_local = 0; skip N_f calculation below }]
  // [Host: N_f = round(N_total × A_f / A_global) for each owned face f]
  // [Host: n_marshak_local = Σ_f N_f]  // R13 グリッドサイズ＋MPI_Exscan の入力
  // 単一GPU（n_ranks==1）の場合は MPI_Allreduce をスキップし A_global = A_local。
  // Marshak BC が存在しない場合（全面が vacuum/reflect）は A_global=0, n_marshak_local=0。
  [MPI] MPI_Exscan((int64_t)(n_new_particles + n_marshak_local), MPI_INT64_T, SUM) → rank_offset  // int64_t 必須（2K+ GPU × 1M粒子/GPU で int32 オーバーフロー）
  // **MPI_Exscan 注意**: rank 0 の recvbuf は MPI 規格で未定義。
  // 実装必須: `if (rank == 0) rank_offset = 0;`。
  // 単一GPU（n_ranks==1）の場合は MPI_Exscan をスキップし rank_offset=0。
  [Host: step_base = (uint64_t)step * N_max_per_step]  // N_max_per_step = 2^40。Census粒子との global_id 衝突回避（NUMERICS §12.7.1）
  [Host: global_id_base = step_base + rank_offset]      // global_id = global_id_base + local_index（local_index = 0..N_emit-1）
  if (n_new_particles > 0):                              // ゼロ粒子時はカーネル起動をスキップ
    R6:  source_particle_fill(grid=(n_new_particles+127)/128)  // global_id_base を引数に渡す
  if (n_marshak_local > 0):                              // Marshak BC 非適用ランクではスキップ
    R13: marshak_source (id_offset = global_id_base + n_new_particles)
       // R13 の global_id = id_offset + local_thread_idx（R6 と ID 空間が重複しない）
       // RNG: curand_init offset=0 で初期化。カーネル終了時に rng_counter を消費済み draw 数に更新（§6.0h 準拠）
  // --- E_Marshak_in 診断（U3 §7.2 エネルギー収支に必要）---
  // 方式: **解析計算**（ホスト側）。R13 粒子は (a_eV c/4) T_{r,f}⁴ A_f dt / N_f のエネルギーで生成され、
  //        Σ_p E_p = Σ_f (a_eV c/4) T_{r,f}⁴ A_f dt が保証されるため（NUMERICS §8.2）、
  //        CUB Sum は不要。ホストが namelist 定数から直接計算する:
  //        E_Marshak_in = Σ_f (a_eV × c / 4) × T_{r,f}⁴ × A_f × dt（NUMERICS §10.2）
  注: M5の純IMC構成では R2/R3/R3b は無効化し、R4-R6 はIMCソース生成として使用する。DDMC拡張はM6で有効化。
       純IMC時は ddmc_mode[(n_cells+n_ghost)×G] を全ゼロ初期化（cudaMemsetAsync、上記初期化ブロック）し、
       R7 の composite key 生成で全粒子が mode=IMC として分類されることを保証する。

  --- Composite Key Sort（§0.5：R7 = 旧R7+R11+R14 融合）---
  [Host: N_total = n_census + n_new_particles + n_marshak_local]  // n_census = 前ステップから生存した粒子数（R7 ソート結果の n_alive、step 0 では 0）
  // **N_total==0 ガード**: census=0 かつ n_new_particles=0 かつ n_marshak=0 の場合、
  // R7 全サブステップをスキップし count_imc=count_ddmc=0 を設定する。
  // CUB RadixSort は size=0 で呼び出すと未定義動作の可能性があるため、ホスト側でガードすること。
  if (N_total > 0):
    R7:  composite_sort_and_partition
         サブステップ1: build_composite_key (合成キー生成 + count_imc/count_ddmc atomicAdd)
         [SYNC] → [D2H] count_imc, count_ddmc  // サブステップ2/3 と R8/R9 のグリッドサイズに必要
         N_alive = count_imc + count_ddmc（ホスト計算）
         サブステップ2: CUB RadixSort (comp_key, perm)
         サブステップ3: fused_soa_gather(N_alive)
         → 結果: SoA[0..n_imc-1]=IMC(cell順), SoA[n_imc..n_alive-1]=DDMC(cell順)
         → 前ステップのdead粒子は自動除去（ソート末尾→n_alive以降を無視）
  else:
    [Host: count_imc=0, count_ddmc=0, N_alive=0]

  --- DDMC→IMC 遷移粒子再サンプル ---
  if (n_imc > 0):
    R7b: ddmc_to_imc_resample (IMC粒子[0..n_imc-1]、§6.0d1)
    // 前ステップで DDMC だった census 粒子のセルが IMC に遷移した場合、
    // R7 build_composite_key が mode=IMC に上書きしたが pos/dir は NaN sentinel のまま。
    // R7b は isnan(pos_r) で遷移粒子を検出し、セル内一様位置 + 等方方向を再サンプルする。
    // 非遷移粒子（大半）は isnan チェックで即座に return → コストは ~5μs（起動オーバーヘッドのみ）

  --- 輸送 ---
  if (n_imc > 0):
    R8:  imc_transport_persistent (Persistent Warp, IMC粒子[0..n_imc-1], grid=n_sm×8)
  if (n_ddmc > 0):
    R9:  ddmc_event_loop (History-based, DDMC粒子[n_imc..n_alive-1])
  [SYNC]

  > **ステップ間 DDMC→IMC モード遷移**：
  > セルの不透明度変化（τ < τ_DDMC）により ddmc_mode が DDMC→IMC に遷移する場合、
  > R7 build_composite_key が census 粒子の mode を IMC に上書きするが、
  > 位置・方向は NaN sentinel のまま残る。R7b がこの遷移を検出し再サンプルする。
  > 逆（IMC→DDMC）は **2経路** で発生し、いずれも **pos/dir を NaN sentinel に書き換え必須**：
  > (a) **ステップ境界（R7）**: build_composite_key が ddmc_mode テーブルから mode=DDMC に上書きする際、
  >     old_mode==IMC なら pos/dir を NaN 化（§6.0d サブステップ1 参照）。
  > (b) **ステップ内（R8）**: IMC 輸送中に τ≥τ_DDMC セルへ移動し mode=DDMC に変換する際、
  >     pos/dir を NaN 化（§6.0c R8 処理フロー参照）。
  > NaN 化が必要な理由：(1) U7 は mode==DDMC でスキップするが、NaN は防御的不変条件（mode 破損時の安全策）、
  > (2) 次ステップ R7b が isnan(pos_r) で DDMC→IMC 遷移検出に依存、(3) NaN 化しないと stale 位置が残り
  > セルモード再遷移時に R7b が見逃す。R9 は pos を参照しないため動作上は安全だが、
  > NaN 不変条件の一貫性のために変換時点で設定する。

  > **ステップ内 R8/R9 モード遷移の扱い（v1.0設計方針）**：
  > R8 内で IMC→DDMC に変換された粒子（time_remain > 0, mode=DDMC）、および
  > R9 内で DDMC→IMC に変換された粒子（time_remain > 0, mode=IMC）は、
  > 当該ステップ内では変換先カーネルで再処理**されない**。
  > 次ステップの R7（composite_sort_and_partition）で合成キーにより正しく分離され、
  > R7b（遷移リサンプル）→ R8/R9 で処理される。
  > これは O(Δt) の誤差を含むが、Strang splitting の分割誤差と同等であり、
  > 統計的再現性に影響しない（NUMERICS §6.6.3 物理的等価性参照）。

  > **Dead粒子の遅延除去**：R8/R9 内で死亡（census, escape, absorption）した粒子、
  > および R12 で kill された粒子は `alive=0` に設定されるが、当該ステップ内では
  > compaction されない。次ステップの R7 composite sort で自動的に末尾に排除される。
  > これにより1ステップあたりの SoA全体permutation を1回に削減する。
  > dead粒子が混在する間の余分なソートコスト（~10-20% の粒子数増）は、
  > 独立compactionパス削減（15可変配列×gather）のコストを大きく下回る。

  R10: tally_finalize                    // difference: rad_E=E_ref_avg+signed_residual/(V c dt)
  // --- DDMC 運動量沈着ポストプロセス（NUMERICS §7.8 準拠）---
  // DDMC の rad_mom_dep は R9 イベントループ中ではなく**全イベント完了後**にポストプロセスとして算出する（NUMERICS §7.8: §7.8.2 "イベントループ完了後に算出"）。
  // R10 が rad_E_tally を正規化して rad_E[c,g] を確定した後、以下の手順で DDMC 運動量沈着を計算:
  //   1. φ_{i,g} = c × rad_E[i,g] / (4π)（scalar intensity、§7.8.1 residence estimator 由来）
  //   2. 面フラックス F_f = σ_{R,f,g} × φ_i × Δx_i - σ_{L,f+1,g} × φ_{i+1} × Δx_{i+1}（§7.8.1）
  //   3. p_i = (1/(2c V_i)) Σ_f σ_{R,f,g} × F_f × A_f × n̂_f（§7.8.2。R/Z成分分離）
  //   4. atomicAdd(&rad_mom_dep[i*dim + d], p_i_d × V_i × dt)（IMC寄与と同一配列に加算）
  // **v1.0 実装**: 上記を専用カーネル `ddmc_momentum_postprocess` として実装（block=256, grid=(n_cells+255)/256）。
  //   入力: rad_E[n_cells×G]（R10出力）, sigma_R[(n_cells+n_ghost)×G], Sigma_leak[n_cells×n_faces×G], vol, face_area, face_normal
  //   出力: rad_mom_dep[n_cells×dim]（atomicAdd で IMC 寄与に加算）
  //   DDMC 無効時（ddmc_enabled=False）はスキップ。DDMC セルがゼロの場合もカーネル起動は安全（全セル rad_E=0 で寄与なし）
  if (cfg.radiation.ddmc_enabled):
    ddmc_momentum_postprocess(rad_E, sigma_R, Sigma_leak, vol, face_area → rad_mom_dep)
  if (N_alive > 0):                              // grid=0 起動防止（N_alive=0 のとき R12 は処理対象なし）
    R12: russian_roulette (census粒子 + DDMC粒子に適用。IMC粒子はR8内でインラインrouletteを受けるため
         R12の対象外。R12は n_alive 粒子中 mode==DDMC || time_remain==0 のみを処理。
         条件不成立の粒子は early return)
  U1:  source_injection(rad_dep → ee)
  H14: eos_inverse(species=0, ee → Te)  // re-closure: Phase 5 (Hydro) が最新Te/Peを必要とする
  // --- Temperature maximum-principle monitoring（NUMERICS §11.8）---
  // T_max_n = max(max_i(Te_i^n), T_boundary) [eV]
  //   T_boundary = Marshak BC 駆動温度（§8.2）。境界が真空/反射のみの場合は T_boundary=0
  //   Phase 3 U2 後に CUB Max(Te) → [D2H] → Host 保持。T_boundary は namelist 由来の定数
  // H14 出力の Te^{n+1} に対し overshoot 検出:
  //   CUB DeviceReduce::Max(Te) → Te_max_new
  //   [SYNC] → [D2H] Te_max_new
  //   [Host: overshoot_max = (Te_max_new - T_max_n) / T_max_n]
  //   CUB DeviceReduce::Sum(TransformInputIterator(Te, [T_max_n](Te_i){ return Te_i > T_max_n ? 1 : 0; }), overshoot_count, n_cells)
  //   [SYNC] → [D2H] overshoot_count
  //   [Host: if overshoot_count > 0 && overshoot_max > ε_warn(0.01): WARNING]
  //   [Host: if safety.overshoot_fatal_enabled && overshoot_max > ε_fatal(0.10): FATAL]
  H13: eos_forward(Te → Pe, Cv)         // 圧力・比熱更新
  U2:  floor_clamp(rho, Te, Ti)
  // **U2後のPe/ee stale window 注記**: U2 が Te/Ti/ρ をクランプした場合、
  // Pe/Pi/Cv_e/Cv_i/ee/ei は H13（U2 前）の値のまま stale となる。
  // Phase 5 Hydro predictor の **H4**（compute_corner_force）が stale P^n を使用するが、
  // フロアクランプ対象セルは ρ≈ρ_floor, Te≈T_floor であり
  // 圧力寄与は ΔP ~ ρ_floor × kB × T_floor ≈ 10^{-22} dyne/cm² — 完全に無視可能。
  // **イオン側も同様**: Pi/Cv_i の staleness による H11 への影響も同程度に無視可能。
  // 厳密を期す場合は U2→H13 の順序に入れ替え可能だが、v1.0 では現状順序を維持する。

  --- 並列粒子移動 (MPI)（NUMERICS §12.3.2 per-substep 同期プロトコル準拠）---
  > **v1.0設計**：Persistent Warp (R8) は1ステップ=1サブステップ（time_remain 消費で完結）。
  > 領域外に脱出した粒子は alive=1, cell_id<0 で R8/R9 を終了し、下記の P5/P6 で移送する。
  > NUMERICS §12.3.2 の「各トラッキングサブステップ終了後に交換」は、
  > v1.0 では1回の R8/R9 完了後に1回の P5→MPI→P6 として実現される。
  > 将来版でサブステップ分割を導入する場合は、R8/R9 内にサブステップ境界を設け、
  > 各境界で P5→MPI→P6 を挿入する設計に拡張する。
  P5:  emigrant_detect_pack
  [SYNC] → [D2H] emigrant_count, per_dest_count  // MPI Isend/Irecv のバッファサイズに必要
  [Host: n_send = min(emigrant_count, emigrant_capacity)]  // オーバーフロー時のバッファ超過防止
  [MPI] exchange_emigrants (Isend/Irecv/Waitall, n_send使用)  // n_ranks==1 の場合は P5/MPI/P6 全体をスキップ（emigrant は存在しない）
  [H2D] cudaMemcpyAsync(device_recv_buf, host_recv_buf, n_recv×104B)  // Waitall完了後に recv_buf をデバイスに転送（P6 がデバイスポインタとして読むため必須）
  // **P6 容量注記**: R6/R13 前の pool capacity check は n_alive+n_new+n_marshak を対象とし、
  // n_recv は事前予測不可。P6 時点の pool_capacity 超過は P6 カーネル内で処理する
  // （pool_offset+tid >= pool_capacity → 書き込みスキップ, particle_overflow=1,
  // E_numerical_loss 計上。§8.3 参照）。R8/R9 で死亡・emigrant化した粒子分の余裕があるため稀。
  cudaMemsetAsync: n_recv_accepted=0             // P6 の atomicAdd 先をゼロ初期化（前ステップの残留値防止）
  if (n_recv > 0):                              // grid=0 起動防止（受信粒子なし時は P6 スキップ）
    P6:  immigrant_unpack_merge
  [SYNC] → [D2H] n_recv_accepted  // P6 の atomicAdd 結果を取得（容量超過分を除外した実受理数）
  [Host: n_emigrant = emigrant_count]  // 全emigrant試行数（alive=0化+alive=2化）。alive=2（overflow）粒子も R7 では dead 扱い（alive!=1）
  [Host: N_alive_post_mpi = n_alive_pre_mpi - n_emigrant + n_recv_accepted]  // n_emigrant=emigrant_count（overflow含む）。第2R7 のガード用
  if (N_alive_post_mpi > 0):
    cudaMemsetAsync: count_imc=0, count_ddmc=0  // 第1R7の値をクリア（第2R7のatomicAddが正しく動作するため）
    R7:  composite_sort_and_partition (受信粒子含む再ソート+compact+partition, **re-arm無効**)
  // **重要**: 第2R7の fused_soa_gather では census re-arm を**実行しない**（dt引数=0 または re-arm フラグ=false）。
  // 理由: R8/R9 が当該ステップで生成した census 粒子（time_remain=0）を、当該ステップの dt で
  // re-arm すると、次ステップの第1R7 で re-arm が不要と判定され（time_remain>0）、
  // 次ステップの Δt ではなく当該ステップの Δt でトランスポートされる。
  // Census 粒子の正規の re-arm 箇所は次ステップの第1R7 の fused_soa_gather のみ。
    [SYNC] → [D2H] count_imc, count_ddmc  // 受信粒子込みの最終 n_alive を取得
    [Host: n_alive = count_imc + count_ddmc]  // 次ステップの R5/R6 オフセットと pool 容量管理に必要
  else:
    [Host: count_imc=0, count_ddmc=0, n_alive=0]  // 全粒子消滅時（稀だが1D低密度問題で発生しうる）
```

---

## 11. 元 `docs/CUDA_KERNELS.md` の §9 冒頭の multi-stream 化の RAW 依存のうち Phase 4 の粒子輸送に関するもの（R7/R7b/R8、SoA ダブルバッファ）

- Phase 4 init (`cudaMemsetAsync`) → R2（ddmc_mode ゼロ初期化）, R8/R9/R12（rad_dep, rad_E_tally はゼロ状態を前提に atomicAdd）
- R7 `fused_soa_gather` → **R7b** `ddmc_to_imc_resample` → R8/R9（SoA ダブルバッファ所有権遷移。R7b は R7 が gather 完了した pos/dir を読み書きするため、R7→R7b→R8 の順序が必須。active_pool_index のフリップは R7 完了イベント後のみ許可）
  **ダブルバッファ状態遷移**（ステップ内）:
  1. 第1R7: gather(src→dst), flip(active=dst) → R8/R9 は dst を読み書き
  2. P5: dst を読み取り emigrant 抽出
  3. P6: dst に immigrant 追記
  4. 第2R7: gather(dst→src), flip(active=src) → 次ステップの第1R7 は src を読み取り
  **注**: N_alive_post_mpi==0 で第2R7がスキップされた場合、active_pool_index は dst のまま残る。
  次ステップの第1R7 は `active_pool_index` 変数が指すバッファから gather する（固定の "src" ではない）。
  実装は `src = pool[active_pool_index], dst = pool[1 - active_pool_index]` とし、各 R7 完了後に flip する。

---

## 12. 元 `docs/CUDA_KERNELS.md` の §9 冒頭の multi-stream 化の RAW 依存（R8/R9/R12 → U1、P6）

- R8/R9/R12 → U1（rad_dep）
- MPI `Waitall` → P6 `immigrant_unpack_merge`（recv_buf H2D 完了保証）

---

## 13. 元 `docs/CUDA_KERNELS.md` の §10.1 アクセスパターン表の粒子行（Random cell read: IMC/DDMC、Atomic scatter: タリー）

```text
| **Coalesced SoA** | 粒子load/store | 100% | SoAレイアウト |
| **Stencil** | Kershaw, AV, corner force | 80-90% | 構造格子の固定ストライド |
| **Random cell read** | IMC/DDMC（セルデータ参照） | 30-60% | `__ldg()` + セルソート |
| **Atomic scatter** | Tally (rad_dep, rad_E_tally) | 20-50% | セルソート + warp集約（§6.4、v1.0既定） |
```

---

## 14. 元 `docs/CUDA_KERNELS.md` の §10.3 `__launch_bounds__` 一覧の R8/R9/R6 行

```text
| `imc_transport` (R8) | 128 | 8 | 64 | 50% (1024 threads/SM) | ~60 reg使用、IMC主ループ |
| `ddmc_event_loop` (R9) | 128 | 16 | 32 | 100% | ~30 reg、mode partition で R8 と分離 |
| `source_particle_fill` (R6) | 128 | 8 | 64 | 50% | Philox RNG + position sampling |
```

---

## 15. 元 `docs/CUDA_KERNELS.md` の §10.4 共有メモリ表の R8/R9 行と将来拡張タリーの内訳

```text
| `energy_budget` (U3) | 256 × 3 × 8B = 6 KB | 部分和accumulation（E_kin, E_int_e, E_int_i）。E_census/E_escape は CUB Sum で別途算出 | v1.0 |
| `cfl_reduction` (U4) | 256 × 8B = 2 KB | min reduction | v1.0 |
| `imc_transport` (R8) | 128 × 20B = 2.5 KB | タリー Stage 2 ビンヒストグラム | 将来拡張 |
| `ddmc_event_loop` (R9) | 128 × 20B = 2.5 KB | タリー Stage 2 ビンヒストグラム | 将来拡張 |

**将来拡張 タリー共有メモリ内訳**（`tally_mode="warp_block"` 時、v1.0では未使用）:
- `smem_dep[128]`：double × 128 = 1024 B（吸収沈着集約）
- `smem_tl[128]`：double × 128 = 1024 B（track-length推定量集約）
- `smem_keys[128]`：int × 128 = 512 B（セル×群キー）
- `smem_n_bins`：int × 1 = 4 B（使用中ビン数）
- **合計**：~2.5 KB/block

**occupancy への影響**（A100: 164 KB shared/SM）:
- R8/R9 を 8 blocks/SM で起動する場合：2.5 KB × 8 = 20 KB（12%）→ 影響なし
- U3 (energy_budget) は 1 block/SM あたり 10 KB だが、grid_size が小さいため制約なし
```

---

## 16. 元 `docs/CUDA_KERNELS.md` の §12 CUB 使用一覧の粒子の行と Scratch 最大必要量

```text
| Prefix sum | `DeviceScan::ExclusiveSum` | Source particle offsets | ~4 × N_cells × G B |
| ~~Flagged select~~ | ~~`DeviceSelect::Flagged`~~ | ~~PhotonPool compaction~~ | R7に吸収 |
| Radix sort | `DeviceRadixSort::SortPairs` | Composite Key Sort（R7、§0.5） | ~24 × N_particles B |
| ~~Partition~~ | ~~`DevicePartition::Flagged`~~ | ~~IMC/DDMC分離~~ | R7に吸収 |

**Scratch buffer最大必要量**: RadixSort (~24 × N) + Fused Gather double buffer (~92 × N)。
100万粒子で ~116MB。ただし Fused Gather の double buffer は Scratch とは別に
PhotonPool の `src`/`dst` として確保される（§5.3 の PhotonPool 容量に含まれる）。
Scratch 単体では RadixSort が支配的 → ~24 × N_particles bytes。
```

---

## 17. 元 `docs/CUDA_KERNELS.md` の §12.5 カーネル→マイルストーン対応表の R 行

```text
| R1 | Fleck factor | M5 |
| R2-R3 | DDMC mode judge, DDMC leak coeff | M6（DDMC統合） |
| R4-R6 | Source energy/count/fill（IMC実装 + DDMC拡張） | M5（IMC）, M6（DDMC拡張） |
| R7 | Composite sort and partition（旧R7+R11+R14融合、§0.5） | M5（IMC）, M6（DDMC統合） |
| R8-R9 | IMC transport (Persistent Warp), DDMC event loop | M5（IMC）, M6（DDMC統合） |
| R10,R12,R13 | Tally finalize, russian roulette, marshak source | M5 |
```

---

## 18. 元 `docs/CUDA_KERNELS.md` の §13 最適化ロードマップの 1. Composite Key Sort と 2. Warp-level タリー集約

1. **Composite Key Sort**（§0.5、NUMERICS §6.5）: 合成キーソートによるセルソート + dead compaction + モード分離の融合。
   従来の R7+R11+R14（3操作 × 15可変配列個別gather）を R7 単一パイプライン（合成キー生成 → RadixSort → fused gather）に統合し、
   粒子管理オーバーヘッドを **~60% 削減**
2. **Warp-level タリー集約**: `__match_any_sync` + `__shfl_down_sync` によるwarp内reduction
   （§6.4、NUMERICS §10.3.3）。セルソートと不可分で同時有効化される
   - namelist: `Parallel.gpu_optimization.tally_mode="warp"`（v1.0既定）
   - 効果：global atomicAdd 回数を最大32分の1に削減

---

## 19. 元 `docs/CUDA_KERNELS.md` の §13 最適化ロードマップの 6〜8（タリー集約・Persistent Warp の拡張）

6. **Block-level タリー集約（将来拡張）**: 共有メモリビンヒストグラム（§6.4、§10.4、NUMERICS §10.3.4）
   - namelist: `Parallel.gpu_optimization.tally_mode="warp_block"`
   - 共有メモリ：2.5 KB/block（R8, R9）
   - Persistent Warpとの整合性課題あり（§6.4 Stage 2 注記参照）
7. **DDMCへのPersistent Warp拡張**: v1.0ではIMCのみPersistent Warp（§6.4）、DDMCはHistory-based（§6.5）。DDMCにもPersistent Warpを適用し負荷分散を改善
8. **IMC/DDMC統合Persistent Warp**: IMC/DDMCを単一カーネルに統合（Composite Key Sortによりmode_partitionは既にR7に吸収済みだが、レジスタ要件の差を解消するにはカーネル統合が必要）
