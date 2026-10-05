# 退役した輻射 Monte Carlo — ARCHITECTURE の記述

2026-09-29 にビルドから外したモンテカルロ輻射（IMC・DDMC・ランダムウォーク・HOLO・difference 定式化）について、
`docs/ARCHITECTURE.md` にあった設計記述をここへ移した。本文は移設時点（コミット 5bc8f6ce3 の文書）のままで、以後は更新しない。
節番号・節への参照（「§5.3」「CUDA_KERNELS §6.4」など）は移設前の `docs/ARCHITECTURE.md` のもの。コードの所在と復元の手順は
`../README.md` を参照。

---

## 1. 元 `docs/ARCHITECTURE.md` の §4.5 radiation/ の退役部分（`CellRadiationCoeffs` 構造体から §4.5.4 の終わりまで）

**CellRadiationCoeffs 構造体**（M17: f 整合の構造的保証）：

```cpp
// セル毎の放射係数バンドル — NUMERICS §6.1.1 準拠
// R演算子冒頭で全セルに対して1回だけ生成（one-shot）
// 同一ステップ内ではこの構造体のみを参照し、f/σ を個別に再計算してはならない
struct CellRadiationCoeffs {
    // deviceメモリ上のフラット配列（SoA）
    double* f;                  // [n_cells] Fleck factor
    double* sigma_pa;           // [n_cells × G] σ^PA_g [1/cm]（Planck absorption）
    double* sigma_pe;           // [n_cells × G] σ^PE_g [1/cm]（Planck emission; LTE時は σ^PA と同一）
    double* sigma_a_eff;        // [n_cells × G] f × σ^PA_g
    double* sigma_s_eff;        // [n_cells × G] (1-f) × σ^PA_g
    double* eta;                // [n_cells × G] η_g = σ^PE_g c a_eV T^4 b_g [erg/(cm³·s)]
    double* eta_tot;            // [n_cells] Σ_g η_g
    double* eta_cdf;            // [n_cells × G] 群再サンプル用 CDF
    double* emission_bias_cdf;  // [n_cells × G] thermal emission 用 spectral-bias CDF（任意）
    int     n_cells;
    int     G;                  // 群数

    // index helper: cell c, group g → c * G + g
};
```

> **生成タイミング**：R演算子冒頭で1カーネル（R2相当）として全セルを並列計算。
> LTE モードでは既存の Fleck factor 計算を CellRadiationCoeffs 形式に拡張。
> Non-LTE モードでは NLTETable からの補間結果を使用する。
> `emission_bias_cdf` は `sigma_R` と Planck table から host 側で組み立てて
> `PersistentCoeffBuffers` へ転送する補助配列であり、Phase-1 では thermal source の count allocation のみが参照する。
> **メモリ**: (8×G+2) × n_cells × 8 bytes。G=100, n_cells=10000 で ~64 MB。Scratch 領域に配置可。

- `Rad::IMC` **[RETIRED — legacy IMC Monte Carlo; 現行輻射は §4.5 冒頭 banner の FLD/\(S_N\)]**
  - 粒子プール（SoA）、census管理
  - Fleck factor計算（LTE: Planck重み平均のσ_a,P、Non-LTE: Λベース — NUMERICS §6.1.1）
  - difference reference の scalar helper（`difference_reference_weight`,
    `difference_reference_cell_sigma`）は PR9 unit test から直接呼べる pure function とし、
    transport state や GPU buffer を変更しない
  - implicit capture（連続吸収）と実効散乱
  - 追跡カーネル（境界交差/衝突/散乱/census）
  - diffusion 分類マスク（current/previous/hold、\(\tau_R\)、reduced flux）、deterministic \(E^D_{i,g}\) buffer、前ステップ signed face-current buffer、IMC↔diffusion interface face-current buffers を保持する
  - **FREEZE-1D-RAD (2026-04-26)**: IMC/DDMC/HOLO/difference 経路は
    1D_SPH では namelist validation で `ConfigError` とし、2D_RZ 用コードとして保持する。
    1D_SPH production radiation は `multigroup_diffusion` と `sn_transport` のみ。
- `Rad::DifferenceResidualization`（`src/radiation/difference_residualization.cu`, `difference_residualization.cuh`）
  - difference formulation の census 残差化を GPU 上で実行し、cell×group bin の signed/absolute energy、scale/rebuild/kill/empty 判定、残差 PhotonPool 再構築を担当する
  - `previous_reference_U` は device-resident reservoir として維持し、empty bin 残差粒子も host roundtrip なしで生成する
- `Rad::DiffusionConversion`（`src/radiation/diffusion_conversion.cu`, `diffusion_conversion.cuh`）
  - diffusion entry セルの alive 粒子を cell×group accumulator へ fold し、`IMC::diff_E_` [erg/cm³] へ変換する
  - diffusion exit セルの `diff_E_` を IMC 粒子へ戻し、予約済み high local-id range の exit subrange から `global_id` を割り当てる
- `Rad::DiffusionInterface`（`src/radiation/diffusion_interface.cu`, `diffusion_interface.cuh`）
  - IMC packet が diffusion cell へ入る boundary crossing を positive `face_current_in` source と signed `face_current_step` に分離して tally する
  - RKL2 後の diffusion-IMC interface leakage を `face_current_out` に保存し、`diff_E_` から差し引いた energy と同量の IMC packet を adjacent IMC cell に生成する
  - interface spawn 粒子は exit subrange と disjoint な high local-id subrange から `global_id` を割り当てる
  - tail IMC pass 後に diffusion へ戻った interface packet energy を `diff_E_` へ直接加算する
- `Rad::DiffusionSourceSolve`（`src/radiation/diffusion_source_solve.cu`, `diffusion_source_solve.cuh`）
  - diffusion セルだけを1 thread/cell で処理し、cell-local Newton solve により `diff_E_`, `Te`, `ee`, `Pe` を Radiation step 内で更新する（NUMERICS §7.1.2c）
  - `rad_dep` / `rad_emit` には gross absorption/emission 診断を加算するが、後段 `Coupling::SourceTerms` では diffusion セルに再適用しない
- `Num::RKL2STS`（`src/numerics/rkl2_sts.cu`, `rkl2_sts.hpp`）
  - Legendre recurrence から RKL2 stage 係数を host 側で生成し、stage 数を \(\sqrt{\Delta t/\Delta t_{exp}}\) scaling で見積もる（NUMERICS §7.1.2d）
- `Rad::DeterministicDiffusion1D`（`src/radiation/deterministic_diffusion_1d.cu`, `deterministic_diffusion_1d.cuh`）
  - 1D_SPH diffusion セルの `diff_E_` を frozen Rosseland face stencil と RKL2 super-time-stepping で更新する
  - diffusion-diffusion 内部面は antisymmetric finite-volume flux、RKL2 内の diffusion-IMC interface は zero-current、outer vacuum は \(cE/4\) leakage を使う。PR5 の diffusion-IMC outward leakage は RKL2 後に `Rad::DiffusionInterface` が処理する
- `Rad::DDMC` **[RETIRED — legacy DDMC/HOLO; 現行輻射は §4.5 冒頭 banner の FLD/\(S_N\)]**
  - セル×群モード判定（τ **かつ** ω、NUMERICS §7.1.2準拠。σ_R ベースで f/η/σ^PA とは独立）
  - diffusion離散（Kershaw係数）→リーク係数生成
  - **M‑matrix診断**（オフ対角 ≤0、正値確率を保証できないセルはDDMC無効化）
  - DDMCイベント（リーク/吸収/census）
- `Rad::Interface`
  - IMC⇄DDMC変換（位置・方向サンプル）
  - interface bc の選択（cosine / half_isotropic）
- `Rad::Tally`
  - 沈着（rad_dep[cell,g]）
  - 推定量（rad_E[cell,g]：track‑length/residence estimator）
  - 境界流出、エネルギー収支
  - difference formulation の deterministic reference face transport buffers
    （`diff_ref_face_delta_U_`, `diff_U_ref_end_`, `diff_E_ref_avg_`）。これらは
    `rad_dep`/`rad_E_tally` とは別に保持し、`diff_U_ref_end_` を次 step の
    previous-reference reservoir、`diff_E_ref_avg_` を PR7 の `rad_E`
    reconstruction に使う
  - **タリーは本モジュールに集約**し、他モジュール（Hydro/Laser等）が直接atomic操作しない
  - **GPU集約戦略（3段階、NUMERICS §10.3 準拠）**：
    1. **Stage 1: warp-level**（`tally_mode="warp"` or `"warp_block"`）：
       - セルソート済み（NUMERICS §6.5）の粒子に対し、`__match_any_sync`（CC 7.0+）で
         warp内の同一セル×群ピアグループを検出
       - 全ピアレーンが `__shfl_down_sync(peers, ...)` で segmented reduction を実行し、リーダーが1回の atomicAdd で書き出す
         （**注意**: リーダーのみが `__shfl_sync` を呼ぶパターンはCUDA仕様§B.15で未定義動作。CUDA_KERNELS §6.4参照）
       - 削減率：最大32倍（warpサイズ分）。セルソート済みで典型的に 28–32倍
       - レジスタ増加：~4（peers, leader, others, src）→ occupancy影響なし
    2. **Stage 2: block-level**（`tally_mode="warp_block"`）：
       - shared memory上のビンヒストグラム（`smem_dep[N_BINS]`, `smem_tl[N_BINS]`, `smem_keys[N_BINS]`）
       - Stage 1のワープリーダー出力を `atomicAdd_block`（ブロックスコープ atomic）で共有メモリに蓄積
       - ブロック末尾で `__syncthreads()` 後、一括 flush（`atomicAdd` → global）
       - 共有メモリ：128 × 20B = 2.5 KB/block（`smem_dep` 8B + `smem_tl` 8B + `smem_keys` 4B）。A100で8 blocks/SM → 20 KB（12%）
       - ビンオーバーフロー時は global atomicAdd にフォールバック
    3. **Stage 3: global atomicAdd**（全モード共通の最終書き出し）：
       - `atomicAdd(double*, ...)` で `rad_dep[cell*G+g]` へ直接加算
       - CC 6.0+（Pascal）でハードウェアサポート。v1.0最低要件（CUDA 12.0+）で充足
  - **v1.0既定**：`tally_mode="warp"`（Stage 1+3）。セルソート（NUMERICS §6.5）と不可分で同時有効化される。
    `"warp_block"`（Stage 1+2+3）はPersistent Warp（NUMERICS §6.6）との設計上の緊張があるため将来拡張とする
  - **namelist制御**：`Parallel.gpu_optimization.tally_mode`（既定 `"warp"`）
  - **セルソート**：輻射演算子冒頭で粒子を `cell_id` でRadixSort（NUMERICS §6.5）。
    Stage 1/2 の前提条件。`Parallel.gpu_optimization.particle_sort_by_cell`（既定 True）で制御
  - IMC / DDMC / PGRW は共通の `rad_E_tally` を共有する。PGRW は `imc_transport_persistent` 内の IMC branch で処理され、吸収減衰は通常 IMC と同じ `rad_dep` / `rad_E_tally` へ加算する
- `Rad::FaceGeometry`（`src/radiation/face_geometry_2d.cuh`）
  - 2D_RZ セルの辺端点からの幾何計算を一元化するヘッダ
  - `FaceGeom2D` 構造体：面端点 \((R_1,Z_1), (R_2,Z_2)\)、法線 \(\hat{n}\)、
    接線 \(\hat{t}\)、面長 \(L\)、中点R座標 \(\bar{R}\) を保持
  - `compute_face_geom()`：4頂点と面ID (0-3) から FaceGeom2D をon-the-flyで計算
    - 法線方向は CCW トポロジ（CUDA_KERNELS §6.4.2 の辺→物理面マッピング）から決定
    - 外向き法線 \(\hat{n} = (\Delta Z, -\Delta R) / L\)
  - IMC transport（`imc_transport_2d.cu`）、DDMC transport（`ddmc_transport_2d.cpp`）、
    境界距離計算（`boundary_distance_2d.cuh`）の3箇所で共有される
  - Hydro の Svec（NUMERICS §3.2.6）とは独立。Svec はコーナー力用、FaceGeom2D は輻射面幾何用
- `Rad::BC`（`src/radiation/boundary.cuh`, `boundary.cu`）
  - vacuum：境界到達で粒子消滅、`E_escape` に計上
  - reflect：鏡面反射（方向ベクトルの**面法線**成分反転。一般法線 \(\hat{n}\) を使用）
  - Marshak：境界放射源の粒子生成（NUMERICS §8.2）
- `Rad::CompositeSort`（`src/radiation/composite_sort.cu`, `composite_sort.cuh`）
  - 64ビット合成キーによるソート＋compaction＋モード分離の融合（NUMERICS §6.5）
  - `composite_sort_and_partition()`：フルRadixSortパス
  - `compact_alive_only()`：全IMC時の O(N) atomic compaction 最適化パス。persistent scratch pool と
    device counter を再利用し、mapping gather + pool swap でSoA copy-backを一括化
  - dead粒子のエネルギー回収（`E_numerical_loss` 計上）
- `Rad::CensusComb`（`src/radiation/census_comb.cu`, `census_comb.cuh`）
  - Census粒子の個体数制御（NUMERICS §6.4.1）
  - `census_comb_gpu()`：hybrid CPU/GPU パイプライン
  - detect_bins -> CUB scan -> D2H -> CPU importance-weighted selection -> H2D -> gather
- `Rad::ModeSelector`（`src/radiation/mode_selector.cpp`, `mode_selector.hpp`）
  - セル*群のIMC/DDMCモード判定（NUMERICS §7.1.2）
  - ヒステリシスモード選択器（NUMERICS §7.1.3）
  - `apply_hysteresis()`：状態機械による遷移制御
  - diffusion mask は selector 後の force-IMC mask として適用し、diffusion/guard セルを DDMC から除外する
- `Rad::DDMCDiffusion1D`（`src/radiation/ddmc_diffusion_1d.cu`, `ddmc_diffusion_1d.cuh`）
  - HIMCD Phase-1 の 1D implicit radiation diffusion solve（NUMERICS §7.4.1）
  - DDMC セル×群だけを抽出し、host-side tridiagonal solve で `rad_E_tally` / `rad_dep` / `E_escape` を更新
  - DDMC-IMC 界面は zero-flux、vacuum 境界は既存 DDMC leak coefficient を sink として再利用
- `Rad::RWTransport1D`（`src/radiation/rw_transport_gpu.cu`, `rw_transport_gpu.cuh`）
  - legacy external RW path。`TransportMode::RW` を生成しないため現行フローでは dead code
  - active な PGRW 実装は `src/radiation/imc_transport_persistent.cu` の internal branch（NUMERICS §7.4.2）
- `Rad::RadLiteMesh`（`src/radiation/rad_lite_mesh.cu`, `rad_lite_mesh.hpp`）
  - 放射メッシュ粗視化（1D_SPH専用、未アクティブ）
  - 隣接セルの不透明度比に基づくセル結合

#### 4.5.1 Radiation トップレベル関数シグネチャ

```cpp
struct RadiationResult {
    double E_absorbed;          // [erg] 総吸収エネルギー（このステップ）
    double E_emitted;           // [erg] 総放出エネルギー（ソース粒子）
    double E_escaped;           // [erg] 境界脱出エネルギー（物理的境界流出のみ）
    double E_numerical_loss;    // [erg] 数値的喪失エネルギー（粒子移送失敗等のアルゴリズム限界。E_escapedとは別計上）
    double E_census;            // [erg] census粒子のエネルギー合計
    int n_particles_end;        // ステップ終了時の生存粒子数
    int n_roulette_kills;       // Russian roulette で除去された粒子数
    double dt_rad;              // [s] 放射CFL制約から算出された推奨Δt
};
```

```cpp
// Radiationフルステップ（Strang splitting R(Δt)）— NUMERICS §6, §7
RadiationResult radiation_step(
    State& state,                       // 流体場・PhotonPool・タリー配列
    const OpacityTable* opacity,        // 不透明度テーブル（device）
    const EOSTable* eos_e,              // 電子EOS（Fleck factor計算用）
    const PlanckTable* planck,          // Planck分率テーブル（device）
    double dt,                          // フルステップ幅 Δt [s]
    const Config::RadiationConfig& rad, // 群境界、粒子数、DDMC閾値
    const PartitionInfo& part,          // 並列情報
    CommBuffers& comm,                  // 粒子移動バッファ
    cudaStream_t stream
);
```

#### 4.5.2 Radiation GPU カーネル起動仕様

放射輸送モジュールの主要カーネルと起動設定：

| カーネル名 | 粒度 | block_size | grid_size | shared memory | 備考 |
|-----------|------|-----------|-----------|--------------|------|
| `compute_opacities` (U9) | 1スレッド=1セル（群ループ内） | **256** | `((n_cells+n_ghost)+255)/256` | なし | σ_a,σ_s,σ_R,σ_P,σ_t 一括前計算（ARCHITECTURE §4.7、CUDA_KERNELS §7.7） |
| `compute_fleck_factor` (R1) | 1スレッド=1セル（群ループ内） | **256** | `(n_cells+255)/256` | なし | R1: Fleck因子。ゴーストセルの f_fleck は halo_exchange で取得（CUDA_KERNELS §6.1 注記） |
| `ddmc_mode_judge` (R2) + `ddmc_leak_coeff` (R3) | 1スレッド=1セル（群ループ内） | **256** | `((n_cells+n_ghost)+255)/256` | なし | R2: DDMCモード判定, R3: リーク係数+M-matrix。ゴーストセル含む（R3b が隣接 ddmc_mode 参照。M08ではR2/R3無効） |
| `compute_source_energy` (R4) + `source_particle_count` (R5) + `source_particle_fill` (R6) + `marshak_source` (R13) | R4/R5: 1スレッド=1セル（群ループ内）, R6/R13: 1スレッド=1粒子 | R4/R5: **256**, R6/R13: **128** | 標準 | なし | R4: source_E [erg] 計算, R5: CUB prefix-sum, R6: 体積ソース生成, R13: Marshak境界ソース生成（BC適用時のみ） |
| `imc_transport_persistent` | Persistent Warp（1warp=1粒子ストリーム） | **128** | `n_sm × blocks_per_sm`（SM数依存、固定） | なし（Stage 1はレジスタのみ） | IMC追跡ループ。`__launch_bounds__(128, 8)` 指定。§5.6.1参照 |
| `ddmc_event_loop` | 1スレッド = 1粒子 | **128** | `(n_ddmc + 127)/128` | なし（Stage 1はレジスタのみ） | DDMCイベントループ。`__launch_bounds__(128, 16)` 指定（CUDA_KERNELS §6.5準拠） |
| `tally_finalize` (R10) | 1スレッド = 1セル×1群 | **256** | `(n_cells*G + 255)/256` | なし | 1スレッド=1(cell,group): legacy は rad_E_tally/(V×c×dt)、difference は E_ref_avg + signed residual を `rad_E` へ正規化（CUDA_KERNELS §6.0e） |

**`imc_transport_persistent` 詳細**（NUMERICS §6.3 準拠、CUDA_KERNELS §6.4）：
> **注**: 以下のシグネチャは主要引数を示す。完全な引数リスト（ddmc_mode, sigma_R, mesh node座標, PlanckTable, birth_energy, global_id, step, user_seed, boundary_type 等）は CUDA_KERNELS §6.4 を参照。R8 は per-face Δx_m をメッシュノード座標からインライン計算する。

```cpp
// IMC粒子追跡カーネル（Persistent Warp）
// 1warp = 1粒子ストリーム、各粒子が独立にイベントループを実行
__global__ void imc_transport_persistent(
    // --- 粒子データ（SoA、PhotonPool）---
    double* pos_r, double* pos_z,
    double* dir_r, double* dir_z, double* dir_phi,
    double* energy, double* weight, double* time_remain,
    int32_t* cell_id, uint16_t* group_id,
    uint8_t* mode, uint8_t* alive,
    uint32_t* rng_counter,
    int n_particles,
    // --- メッシュ・物性（read-only）---
    const Mesh* mesh,
    const double* sigma_a_eff,      // [n_cells × G] 実効吸収 [1/cm]（Fleck factor適用済み）
    const double* sigma_s_eff,      // [n_cells × G] 実効散乱 [1/cm]（Fleck factor適用済み）
    // --- タリー出力（atomic書き込み、thread-unsafe: atomicAddで排他）---
    double* rad_dep,                // [n_cells × G] 吸収沈着 [erg]
    double* rad_E_tally,            // [n_cells × G] track-length推定量の蓄積 [erg·cm]（NUMERICS §10.1）
                                    // カーネル内で Σ E_mid × Δs を atomicAdd で蓄積。
                                    // ステップ末に rad_E[i,g] = rad_E_tally[i,g] / (V_i^{*} × c × Δt) [erg/cm³] へ変換
    double* face_current_step,      // [(n_cells+1) × G] 1D diffusion reduced-flux用の signed face current [erg]
    uint8_t* diff_cell,             // [n_cells] deterministic diffusion cell mask
    double* diff_face_current_in,   // [(n_cells+1) × G] IMC→diffusion positive source [erg]
    double* E_escape,               // [n_groups] 群別境界流出エネルギー [erg]（CUDA_KERNELS §6.4 参照）
    double* rad_mom_dep,            // [n_cells × D_mom] 運動量沈着（診断のみ、momentum_deposition=true時）。Δp = ΔE/c × Ω̂
    // --- 制御パラメータ ---
    double dt,                      // [s] タイムステップ幅
    double t_end,                   // [s] ステップ終了時刻
    int interface_method,           // IMC→DDMC変換方式: 0=asymptotic_diffusion_limit, 1=marshak（CUDA_KERNELS §6.4）
    int dim,                        // 1=1D_SPH, 2=2D_RZ
    DeviceErrorFlags* error_flags,  // §10.1 準拠
    double* E_numerical_loss_dev    // [1] MAX_EVENTS強制終了時の残余エネルギー計上先（atomicAdd）
);
```

- **タリー集約**（§4.5 Rad::Tally、NUMERICS §10.3 準拠）：
  - v1.0（`tally_mode="warp"`、既定）：warp-level `__match_any_sync` + `__shfl_sync` 集約 → global atomicAdd。
    セルソート済み粒子に対しwarp内同一セル×群のピアを検出し、リーダーが1回の atomicAdd で書き出す。
    shared memory 不使用（レジスタ ~4個追加のみ）。global atomicAdd 回数を最大 32分の1 に削減
  - フォールバック（`tally_mode="global"`）：各スレッドが直接 `atomicAdd` で global `rad_dep[]` へ書き込む。CC 7.0未満用
- **イベントループ**：各スレッドが `while(alive && time_remain > 0)` で
  境界交差/散乱/census の最短イベントを逐次処理
- **DDMC遷移**：IMC粒子がDDMCセルに入った場合、`mode` を1に変更し
  `imc_transport_persistent` を終了。当該ステップでは再処理せず、次ステップの `composite_sort_and_partition`（R7）で再分配後に `ddmc_event_loop` で処理される

**`ddmc_event_loop` 詳細**（NUMERICS §7.5 準拠、CUDA_KERNELS §6.5 参照）：

> **注**: 以下のシグネチャは主要引数を示す。完全な引数リスト（rad_mom_dep, Sigma_out, Sigma_leak_bdry, ddmc_mode, global_id, step, user_seed, boundary_type, DeviceErrorFlags, E_numerical_loss_dev 等）は CUDA_KERNELS §6.5 を参照。DDMC運動量沈着（rad_mom_dep）は R9 内のリークイベントで隣接面法線方向に atomicAdd で蓄積する（NUMERICS §7.8 DDMC寄与、CUDA_KERNELS §6.5）。

```cpp
// DDMCイベント処理カーネル
// 1スレッド = 1粒子（DDMCモードのみ）、History-based（Persistent Warp不使用）
__global__ void ddmc_event_loop(
    // 粒子データ（SoA）+ タリー出力（imc_transport_persistent と同一レイアウト）
    // ...省略（imc_transport_persistent と同一引数群）...
    double* pos_r, double* pos_z,       // [n_particles] 位置（DDMC→IMC変換時に書き戻し）
    double* dir_r, double* dir_z, double* dir_phi,
    double* energy, double* time_remain,
    uint32_t* rng_counter,
    int32_t* cell_id, uint16_t* group_id,
    uint8_t* mode, uint8_t* alive,
    int n_ddmc,
    // --- タリー出力（atomic書き込み、thread-unsafe: atomicAddで排他）---
    double* rad_dep,                    // [n_cells × G] 吸収沈着 [erg]
    double* rad_E_tally,                // [n_cells × G] track-length推定量の蓄積 [erg·cm]（NUMERICS §7.6）
                                        // カーネル内で Σ c × E × Δt_res を atomicAdd で蓄積。
                                        // ステップ末に rad_E[i,g] = rad_E_tally[i,g] / (V_i^{*} × c × Δt) [erg/cm³] へ変換
    double* E_escape,                   // [n_groups] 群別境界流出エネルギー [erg]（CUDA_KERNELS §6.5 参照）
    // --- DDMC固有 ---
    const double* leak_coeff,       // [n_cells × n_faces × G] リーク係数 Σ^leak [1/cm]（NUMERICS §7.3）
                                    // イベント時間: Δt = -ln(ξ)/(c × Σ^tot) で c を乗じて [1/s] へ変換
                                    // メモリレイアウト：cell-major → face → group
                                    // 1D: [n_cells×2×G]（左/右）、idx = cell*2*G + face*G + g
                                    // 2D: [n_cells×4×G]（R_left/R_right/Z_bottom/Z_top の4面、idx = cell*4*G + face*G + g — §6.4.3 面規約）
                                    // 注：Kershaw 9-point ステンシルは8近傍だが、DDMCリーク面はトポロジカル面(4面)のみ。
                                    //   角近傍のリーク寄与は隣接する2面に分配される（NUMERICS §7.3.5）
    const double* sigma_a_eff,      // [n_cells × G] 実効吸収 [1/cm]
    // --- 制御パラメータ ---
    double dt,                      // [s] タイムステップ幅
    int nr, int nz, int n_groups, int n_faces  // メッシュ次元（CUDA_KERNELS §6.5 準拠）
);
```

- **イベント処理**：総イベント率 \(\Sigma^{tot}\) から指数分布で時刻を進め、
  リーク/吸収/census を確率的に選択（NUMERICS §7.5）
- **IMC遷移**：DDMCセルからIMCセルへリークした場合、
  `cell_id` をリーク先IMCセルに更新し（CUDA_KERNELS §6.5 参照）、出射方向をサンプルし
  `mode` を0に変更。次ステップの `imc_transport_persistent` で追跡継続

**ソース粒子生成パイプライン詳細**（NUMERICS §6.2 準拠、CUDA_KERNELS §6.0b-§6.3 参照）：

v1.0 では4カーネルパイプラインで構成する：
1. **R4** `compute_source_energy`: 各セル×群の source_E [erg] = S^{emit} × V × Δt を計算
2. **R5** `source_particle_count`: 既定は `max(1, round(N_p_total × source_E / source_total))`。`spectral_bias_eta>0` のときは cell-local `emission_bias_cdf` を参照し、`source_cell_total × q_g / source_total` へ粒子数配分のみを切り替える → CUB prefix-sum
3. **R6** `source_particle_fill`: `E_p = source_E[c,g] / N_p[c,g]` で粒子エネルギーを決定し、位置・方向をサンプル
4. **R13** `marshak_source`（Marshak BC適用時のみ）: 境界面ごとに入射粒子を生成（NUMERICS §8.2、CUDA_KERNELS §6.0h）。`global_id_base` は R6 の末尾ID+1 から割り当て（RNGストリーム独立性を保証）

```cpp
// R6: source_particle_fill（CUDA_KERNELS §6.3）
// 1スレッド = 1粒子：particle_offsets から (cell, group) を逆引きし、
// セル内の位置・方向をサンプルしPhotonPoolに書き込み
__global__ void source_particle_fill(
    PhotonPool pool,                // 書き込み先（SoA）
    int pool_offset,                // census粒子の後に配置
    const double* source_E,         // [n_cells × G] ソースエネルギー [erg]（R4出力、V×Δtを含む）
    const int* particle_offsets,    // [n_cells × G + 1] prefix sum（R5出力）
    const Mesh* mesh,               // 位置サンプリング用
    uint64_t step,                  // RNGシード用ステップ番号
    uint64_t global_id_base,        // MPI offset = step_base + rank_offset（NUMERICS §12.7.1）
    int n_cells, int n_groups, int dim
);
```

`global_id[k] = global_id_base + k`。`global_id_base = step × N_max_per_step + MPI_Exscan(N_emit, SUM)`（N_max_per_step = 2^40、NUMERICS §12.7.1）。

#### 4.5.3 放射輸送の分散低減・加速機能

| 機能 | ファイル | Config | 本番状態 |
|------|---------|--------|---------|
| **差分定式化（DF）** | `imc.cpp`, `difference_residualization.cu`, `tally.cu` | `difference.enabled` | **ON** (本番) |
| **正味電子ソーススムージング** | `source_terms.cu` | `net_e_source_smoothing.*` | **ON** (α=0.25, τ=6) |
| **スペクトルバイアス** | `imc.cpp`, `source.cu` | `spectral_bias_eta` | ON (η=0.3) |
| **ソースティルティング** | `source.cu` | `source_tilting` | ON |
| **PGRW** | `imc_transport_persistent.cu` | `tau_rw` | ON (τ=5) |
| **Census ESS フロア** | `census_comb.cu` | `ess_floor_enabled` | OFF |
| **ソース局在化** | `source.cu`, `imc.cpp` | `source_localization` | OFF |
| **勾配適応フィルタ** | `source_terms.cu` | `gradient_adaptive` | OFF |
| **HOLO global LO solver/coupling mask** | `config.hpp`, `builder.cpp`, `freeze.cpp`, `imc_transport_persistent.cu`, `holo_geometry.hpp`, `holo_selector.cpp`, `holo_lo_state.cu`, `holo_lo_solver.cpp`, `sn_transport_1d.cpp`, `sn_transport_gpu.cu` | `holo.enabled` | OFF |

**差分定式化（DF）**: 一般化参照場 W = W_max × τ²/(τ²+τ0²) × 1/(1+(χ/χ0)⁴) で近平衡殻の MCノイズを根本低減。signed particles（`PhotonPool::sign`）、census 残差化（ビンレベル再利用/スケーリング）、AP制限面輸送 ψ(τ) = tanh(3τ/4)/(3τ/4)。W≥0.5 セルではソーススムージングを自動無効化。設計: 社内の設計メモ difference_formulation_full.md。

**正味電子ソーススムージング**: H = Σ_g(rad_dep - rad_emit) を保存的フェイス交換で平滑化。IMC 沈着ノイズが電子圧力に入る前にフィルタリング。xRAGE のカプセルデポジションスムーザーに類似。マルチパス対応（ping-pong GPU バッファ）、勾配適応α対応。

**Census ESS フロア**: Window群（Rosseland 重要度高）のセンサスESS を閾値以上に維持するためにスプリット。ただし偽証テストで starvation仮説が棄却されたため本番では無効。

**HOLO LO solver**: `holo_lo_solver.cpp` は v1 の swappable backend であり、
1D_SPH 全 cell に対して CPU Thomas 法で global physical-frame LO diffusion/source solve を
毎 radiation step 実行する。内部 HOLO 境界と high-order face-current boundary condition は
使わず、inner reflect と outer vacuum の物理境界だけを適用する。API は geometry dimension
付き host pointer view（`HoloLOInputs`/`HoloLOResult`）で、ideal-gas と electron table
EOS（TMAT 由来 table を含む）closure を扱う。`holo_selector.cpp` は
Rosseland optical depth 閾値と guard cell 膨張で LO material-coupling mask を作るだけで、
solver domain は制限しない。`imc.cpp` は transport 後に global LO solve を呼び、
`source_terms.cu` が mask cell の material source を
`State.holo_rad_dep - State.holo_rad_emit` の LO 診断へ切り替える。particle
`rad_dep/rad_emit` は出力用 raw diagnostic として保持し、mask cell では material
energy へ再適用しない。非 mask cell は従来通り particle source で material update する。
LO solve 失敗時は `State.holo_lo_source_valid=False` のまま warning を出して継続し、
その step の mask cell material source は通常の particle `rad_dep/rad_emit` へ fallback する。
`holo_geometry.hpp` は selector/solver に共通の geometry dimension を定義する。v1 runtime
の LO solver は `Spherical1D` のみを有効化し、2D_RZ は namelist validation で無効化する。
`sn_transport_1d.cpp` は QD closure 専用の standalone CPU backend であり、
`holo.solver="quasidiffusion_1d"` かつ `holo.sn_closure=True` のとき、
LO solve 直前に IMC と同じ host opacity/Fleck/Planck source data から
noise-free \(P_{rr}/E\) closure を作る。MC transport、DDMC、PGRW の particle path とは
独立で、CUDA kernel は持たない。`sn_transport_gpu.cu` は 1D_SPH と 2D_RZ の
group-parallel CUDA \(S_N\) backend である。When
`holo.sn_material_coupling=True`, 1D_SPH QD runs use it as a chi-only HO closure.
`solve_holo_sn_material_coupling`
launches the GPU \(S_N\) sweep with material update disabled, copies
`State.holo_chi` to host, and calls `solve_holo_lo_source_ownership` with that
precomputed closure. `solve_holo_lo_source_ownership` applies the common QD
closure regularizer (`closure_smooth_passes`, `closure_smooth_alpha`,
`closure_relax`) for MC tally closure, CPU \(S_N\) closure, and GPU \(S_N\)
precomputed closure before assembling `HoloLOInputs` and invoking
`solve_holo_lo_1d_cpu(..., true)`.  The LO solver then writes
`State.holo_E_LO`, `State.holo_F_LO`, `State.Te`, `State.ee`, and `State.Pe`.
2D_RZ still bypasses LO/QD and generates the deterministic material source
`State.holo_rad_dep - State.holo_rad_emit` for all cells.
`source_terms.cu` treats the 1D SN+QD LO path as direct material ownership, so
the published LO source is recorded in `delta_E_rad_prev` for diagnostics but
is not applied to `ee` a second time.  The 2D_RZ \(S_N\) path remains
source-injection owned.

#### 4.5.4 ハイブリッド輸送（研究ブランチ、本番 OFF）

3モード構成: IMC（薄い領域）+ DDMC/PGRW（中間）+ RKL2 決定論的拡散（厚い殻）。

| コンポーネント | ファイル |
|---|---|
| セル分類 | `imc.cpp`（τ_R, reduced flux, ヒステリシス） |
| エントリ/エグジット変換 | `diffusion_conversion.cu` |
| セルローカルソース解法 | `diffusion_source_solve.cu` |
| RKL2 STS 拡散 | `deterministic_diffusion_1d.cu`, `rkl2_sts.cu` |
| IMC↔拡散界面 | `diffusion_interface.cu` |

**無効化理由**: 移動界面、モードチャタリング（5,900 入退出）、境界ソースクロージャが振動を 50-80% 悪化。設計: 社内の設計メモ hybrid_transport_plan.md。

---

---

## 2. 元 `docs/ARCHITECTURE.md` の §4.2.3 Mesh::Search — ハッシュグリッドセル探索（本文）

粒子のセル同定（NUMERICS §9）に使用するハッシュグリッド構造体。
背景グリッド（NUMERICS §9.5）として機能し、ALE rezone 後に再構築する。

```cpp
struct HashGrid {
    double cell_size;           // hash cell size = max(Δr, Δz) × 1.1 [cm]（NUMERICS §9.5）
    int* table;                 // OWNED [n_buckets]: hash table: cell_idx per hash bucket (device)
    int n_buckets;              // prime number >= hash_table_factor * n_cells（既定 factor=4）
};
__device__ int find_cell(double r, double z, int cell_hint,
                         const Mesh& mesh, const HashGrid& hash);
```

- `find_cell` はまず `cell_hint`（前回のセルID）から局所探索を試み、
  失敗時にハッシュグリッドフォールバックを使用する（NUMERICS §9.3–§9.5）
- `HashGrid::table` は初期化時に構築し、ALE rezone 後に再構築する
- ハッシュ関数：`hash(i_r, i_z) = (i_r * 73856093 ^ i_z * 19349669) % n_buckets`
- 衝突解決：線形探索（open addressing、stride=1）
  - 空バケット sentinel：`table[bucket] = -1`（初期化時に `cudaMemset(-1)` で設定）
  - 挿入：`bucket = hash(i_r, i_z)` から stride=1 で空き（`-1`）を探し `table[bucket] = cell_idx` を格納
  - 探索：`bucket = hash(i_r, i_z)` から stride=1 で巡回し、格納セルの包含判定を実行。
    空バケット（`-1`）に到達したら探索失敗。最大探索回数 = `n_buckets`（full scan防止、NUMERICS §9.5）
  - 探索失敗時：`DeviceErrorFlags::invalid_cell` を設定し、`cell_id = -1` を返す（§10.1 で host 側が処理）
  - 負荷率：`n_cells / n_buckets ≤ 1/hash_table_factor = 0.25`（既定）で衝突確率を低減

---

## 3. 元 `docs/ARCHITECTURE.md` の §4.2.4 の段落「remap後の粒子再配置」

**remap後の粒子再配置**：
rezone でセル形状が変わるため、**IMC粒子のみ**の `cellId` を再同定する（U7: `cell_search_after_rezone`）。
DDMC粒子は pos=NaN sentinel のため空間探索不可であり、cell_id をそのまま維持する
（ALE rezone はセル番号を変えないため再同定不要。NUMERICS §9.6、CUDA_KERNELS §9 Phase 5 参照）。
Mesh::Search（NUMERICS §9.3–§9.5）を使用。粒子の物理座標は変化しない。

---

## 4. 元 `docs/ARCHITECTURE.md` の §4.3.4 の「不透明度の前計算」`compute_opacities`（U9）と出力バッファの説明

```cpp
// 不透明度の前計算：各セルの σ_a,g, σ_R,g を State から一括計算
// 各ステップの Radiation 呼び出し前に実行
// **M17 Note**: opacity.model="table_nlte" の場合、σ_a,g と σ_R,g は
//   NLTEOpacityTable から補間される（OpacityTable からではない）。
//   DDMCリーク係数は σ_R ベースで変更なし（アルゴリズム不変）だが、
//   データ供給源が NLTEOpacityTable に切り替わるためコードパスは変更される。
//   CellRadiationCoeffs 生成時に σ^PA_g, σ^PE_g の分離も同時に行う。
__global__ void compute_opacities(
    const double* rho,              // [n_cells] 密度 [g/cm³]
    const double* Te,               // [n_cells] 電子温度 [eV]
    const double* volFrac,          // [n_cells × n_mat] 体積分率 [dimensionless]
    const OpacityTable* tables,     // [n_materials] LTE不透明度テーブル
    const NLTEOpacityTable* nlte_tables,  // [n_materials] Non-LTE不透明度（NULLならLTEパス）
    double* sigma_a,                // [n_cells × G] 出力：ρ κ^PA_g [1/cm]（LTE時: ρ κ_P,g）
    double* sigma_R,                // [n_cells × G] 出力：ρ κ_R,g [1/cm]
    double* sigma_pe,               // [n_cells × G] 出力：ρ κ^PE_g [1/cm]（LTE時: = sigma_a）
    double* sigma_t,                // [n_cells × G] 出力：σ_a + σ_s [1/cm]（v1.0: σ_t=σ_a）
    int n_cells, int n_groups, int n_materials,
    MixingRule mixing_rule          // enum（GPU上での文字列比較を回避）
);
```

出力4バッファ：`sigma_a[n_cells×G] = ρκ^PA`、`sigma_R[n_cells×G] = ρκ_R`、`sigma_pe[n_cells×G] = ρκ^PE`（LTE時: = sigma_a）、`sigma_t[n_cells×G] = σ_a + σ_s`（v1.0: σ_t=σ_a）。

---

## 5. 元 `docs/ARCHITECTURE.md` の §4.7 `Coupling::SourceTerms` の輻射の注入（`inject_radiation_source_terms` と正味電子ソースの平滑化）

  - `inject_laser_source_terms` / `inject_radiation_source_terms` は
    `source_injection` 後の \(e_e \leftrightarrow T_e\) クロージャで
    セルごとの \(A_{eff},\gamma_{eff}\) から構成した \(c_{v,e}\) を用いる（NUMERICS §1.1.5a）
  - `inject_laser_source_terms` は cell-local な `laser_dep[c]` をそのまま
    \(e_e\) へ注入し、退化セルでは `E_numerical_loss` へ退避する。
  - `inject_radiation_source_terms` は host 側で
    \(H_c^{raw}=\sum_g(\texttt{rad\_dep}_{c,g}-\texttt{rad\_emit}_{c,g})\) を構成し、
    `TransportMode::Diffusion` セルではこれを smoothing 前に 0 として barrier 扱いし、
    difference reference weight \(W_c\ge0.5\) のセルも difference 併用時の smoothing
    barrier に加え、
    `Radiation.imc.net_e_source_smoothing.enabled` のときだけ
    IMC が保持した `sigma_R_max[c]` を受け取って 1D GPU kernel を起動し、
    \[
    F_{c+1/2}=\alpha\,\lambda_{c+1/2}\,m_{c+1/2}
    \left(\frac{H_c^{raw}}{m_c}-\frac{H_{c+1}^{raw}}{m_{c+1}}\right),\qquad
    H_c^{apply}=H_c^{raw}+F_{c-1/2}-F_{c+1/2}
    \]
    を計算する。`lambda` は optical-depth gate と void/material interface で決まり、
    `delta_E_rad_prev` には raw tally ではなく \(H^{apply}\) を保存する。
  - radiation source の electron update は `ee[c] += H_apply[c] / mass[c]` を用い、
    laser source は従来どおり `rho[c] * vol[c]` ベースの direct deposition を使う。
  - per-group `rad_dep[c,g]` / `rad_emit[c,g]` は raw tally のまま保持し、群別診断は温存する

---

## 6. 元 `docs/ARCHITECTURE.md` の §5.3 PhotonPool（SoA粒子プール）の本文


> **【RETIRED】** PhotonPool・ParticleEmigrant・ParticleMode（`0=IMC, 1=DDMC, 2=RW`）は IMC/DDMC Monte Carlo 粒子輸送専用のデータ構造であり **RETIRED**（FREEZE-1D-RAD・D1 以降）。現行の決定論 **FLD（NUMERICS §6.7, `mode="multigroup_diffusion"`）** / **\(S_N\)（§6.8, `mode="sn_transport"`）** は粒子プールを使わず、cell×group の `rad_E` 場を GPU 上で直接 solve する（ARCHITECTURE §4.5 の `Rad::FLD*`/`Rad::SNTransport*` 参照）。以下は歴史的参照。

IMC/DDMC粒子をStructure-of-Arrays（SoA）で管理する。
全フィールドは NUMERICS §12.3.4 の `ParticleEmigrant` と整合する。

**粒子状態 enum**：

```cpp
// 粒子の輸送モード（IMC連続追跡 vs DDMC離散イベント）
enum class ParticleMode : uint8_t {
    IMC  = 0,   // IMCモード：連続的な幾何光学追跡（NUMERICS §6）
    DDMC = 1    // DDMCモード：離散イベント処理（NUMERICS §7）
};

// 粒子の生死状態
enum class ParticleStatus : uint8_t {
    DEAD     = 0,   // 消滅済み（吸収、境界脱出、Russian roulette で除去）
    ALIVE    = 1,   // 生存中（輸送継続）またはcensus保持（time_remain=0、次ステップで再処理）
    OVERFLOW = 2    // PhotonPool 容量超過で処理できず（エラーフラグ §10.1 に報告）
};
```

```cpp
struct PhotonPool {
    // --- 位置・方向（NUMERICS §0.4 準拠）---
    double* pos_r;          // [capacity] R座標 [cm]
    double* pos_z;          // [capacity] Z座標 [cm]（1Dでも3D方向追跡で使用、NUMERICS §0.4）
    double* dir_r;          // [capacity] 方向ベクトル R成分 [dimensionless]
    double* dir_z;          // [capacity] 方向ベクトル Z成分 [dimensionless]
    double* dir_phi;        // [capacity] 方向ベクトル φ成分 [dimensionless]（RZ内部表現）

    // --- スカラー量 ---
    double* energy;         // [capacity] 粒子エネルギー [erg]
    double* weight;         // [capacity] 統計重み
    double* time_remain;    // [capacity] 残存時間 [s]
    double* birth_energy;   // [capacity] 生成時エネルギー [erg]
                            // Russian roulette/cutoff判定用（NUMERICS §6.3.4）
                            // census粒子は前ステップの値を引き継ぐ
    int8_t*  sign;          // [capacity] 粒子符号（+1 or -1）。legacy path は +1

    // --- 識別子・状態 ---
    uint64_t* global_id;    // [capacity] グローバル粒子ID（RNGストリーム識別用）
    uint32_t* rng_counter;  // [capacity] cuRAND rng_counter（消費済み乱数数）
                            // cuRAND state は curand_init(global_id ^ user_seed, step_number, rng_counter)
                            // でカーネル冒頭に O(1) 復元（NUMERICS §12.7.1 準拠）。
                            // 内部の Philox key/counter 写像は NVIDIA 実装に委ねる。
                            // PhotonPool に保存するのは rng_counter（uint32）のみ
    int32_t*  cell_id;      // [capacity] 所属セルID（**localインデックス**、0 ≤ cell_id < n_cells_local）
                            // **チェックポイント注意**: 書き込み時は local → global 変換が必要。
                            // **1D_SPH**: global = cell_id + cell_offset（PartitionInfo::cell_offset）
                            // **2D_RZ**: local(i,j)=(cell_id/nz_local, cell_id%nz_local) →
                            //   global = (i + ir_start) * nz_global + (j + jz_start)
                            //   （単純な +offset では nz_local ≠ nz_global 時にストライド不整合）
                            // 読み込み時は global → (i_g,j_g) → rank 特定 → local 変換。
                            // rank数変更リスタートで local ID をそのまま使うと粒子が誤セルに配置される
    uint16_t* group_id;     // [capacity] 群番号
    uint8_t*  mode;         // [capacity] ParticleMode enum（上記参照）
    uint8_t*  alive;        // [capacity] ParticleStatus enum（上記参照）

    // --- プール管理（host側で管理、カーネル起動前に設定）---
    int capacity;           // 確保済み要素数（全SoA配列の共通サイズ）
    int n_alive;            // 現在のalive粒子数（compaction後に更新）
    int n_census;           // census粒子数（前ステップからの引き継ぎ）

    // compaction: Composite Key Sort（§5.4）で alive/dead 分離 + セルソートを一括実行
    // （旧仕様の CUB DeviceSelect::Flagged は R7 に吸収済み、CUDA_KERNELS §6.0d 参照）
    void* cub_temp;         // CUB一時バッファ（RadixSort + gather用）
    size_t cub_temp_bytes;

    // --- SoA ダブルバッファ（Composite Key Sort R7 用）---
    // fused_soa_gather は src SoA → dst SoA へ permutation 付きコピーを行う。
    // active_pool_index（0 or 1）が現在の読み取り元を示す。
    // R7 完了後にフリップ: active_pool_index ^= 1。
    // 全16フィールドの第2バッファ: pos_r_alt, pos_z_alt, ..., alive_alt
    // 確保サイズ: 93 bytes × capacity
    // CUDA_KERNELS §9 注記: active_pool_index のフリップは R7 完了イベント後のみ許可。
    // R7b (ddmc_to_imc_resample) は gather 完了済みの dst 側を読み書きするため、
    // R7→flip→R7b→R8/R9 の順序が必須。
    int active_pool_index = 0;  // 0: primary SoA が active、1: alt SoA が active
    // alt SoA ポインタ（primary と同一サイズ、State::allocate() で確保）
    double* pos_r_alt;
    double* pos_z_alt;
    double* dir_r_alt;
    double* dir_z_alt;
    double* dir_phi_alt;
    double* energy_alt;
    double* weight_alt;
    double* time_remain_alt;
    double* birth_energy_alt;
    int8_t*  sign_alt;
    uint64_t* global_id_alt;
    uint32_t* rng_counter_alt;
    int32_t*  cell_id_alt;
    uint16_t* group_id_alt;
    uint8_t*  mode_alt;
    uint8_t*  alive_alt;
};
```

**容量管理**：
- **初期容量**：`initial_capacity = min(particles_per_cell_group * n_local_cells * n_groups * 1.5, max_pool_size)`
  - `×1.5` は census 粒子 + 新規 source 粒子の同時存在を見込む安全係数（NUMERICS §6.4 準拠）
  - `max_pool_size` で上限制約（既定 10⁸。SPECIFICATION §6.4.5 参照）
- **成長戦略**：census + 新規 source が capacity を超える場合、**2倍に拡張**
  （全SoA配列を新領域に `cudaMemcpyAsync` でコピー）
- **拡張手順**：`cudaStreamSynchronize` → 16本 `cudaMalloc(2×cap)` → `cudaMemcpyAsync` → sync → `cudaFree(old)` → ポインタ更新。カーネル間ギャップでのみ実行
- **最大容量**：GPU メモリ予算（§5.6）の 60% を上限とする。
  超過時は以下の段階的回復手順を実行（§5.6.4 メモリ不足対応と同一プロトコル）：
    1. 緊急 Russian roulette：weight_cutoff を一時的に max(weight_cutoff×10³, 10⁻⁴) に引き上げ、全 alive 粒子に間引き判定を再実行
    2. 1回で解消しない場合：w_survive を2倍、N_p を50%削減。最大3ステップまで繰り返し
    3. 3ステップで未解消 → ERROR 停止
- **1粒子あたりメモリ**：`5×8(pos/dir) + 4×8(energy/weight/time/birth) + 1(sign) + 8(global_id) + 4(rng_counter,uint32) + 4(cell_id) + 2(group_id) + 1(mode) + 1(alive) = 93 bytes`
  （+ CUB temp per particle ≈ 4 bytes → 約 97 bytes/particle）
- **チェックポイント時**（SPECIFICATION §7.4 準拠）：
  `rng/rng_counter: uint32[N_p]` + `rng/global_id: uint64[N_p]` を保存。
  リスタート時は `curand_init(global_id ^ user_seed, step_number, rng_counter)` で
  RNG state を O(1) 復元する（NUMERICS §12.7.1）。

---

## 7. 元 `docs/ARCHITECTURE.md` の §5.6.3 Persistent Warp 実行モデルの本文


> **[RETIRED — legacy IMC 輸送の persistent-warp 実行モデル（NUMERICS §6.6 と同系）。現行輻射 FLD/S_N はこのモデルを使用しない]**
IMC輸送カーネル（R8）は **Persistent Warp** モデルで実行する（NUMERICS §6.6）。

**グリッドサイズ決定**：
```cpp
int n_sm;
cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, device_id);
int grid_size = n_sm * 8;  // __launch_bounds__(128, 8) → 8 blocks/SM
// A100: 108 SM × 8 = 864 blocks × 128 threads = 110,592 persistent threads
```

**Work Queue**：
```cpp
struct PersistentWorkQueue {
    int* global_counter;    // [1] atomicカウンタ（device memory）
    int  n_total;           // IMC粒子総数（R7 composite_sort_and_partition後、CUDA_KERNELS §0.5）
    // 終了判定: acquired_index >= n_total のとき、当該レーンは inactive
    // 全レーンが inactive (ballot == 0) でワープ終了
    // 全ワープ終了で grid 終了 (cooperative groups 不要)
};
```
- `global_counter` は `imc_transport_persistent` 起動前に `cudaMemsetAsync(..., 0)` で初期化
- ワープ単位で32粒子ずつ取得（atomicAdd頻度を最小化）
- 粒子補充時は `__ballot_sync` + `__popc` で空きレーン数を計算し、一括取得
- **終了判定**：leader lane（lane 0）が `atomicAdd(global_counter, n_needed)` で取得した base index を `__shfl_sync` で全レーンにブロードキャスト。各レーンは `base + lane_offset` が `n_total` 以上になったら inactive（CUDA_KERNELS §6.4 Ballot Refill 参照）。
  `__ballot_sync(0xFFFFFFFF, active)` でワープ内の active レーン数を監視し、
  全レーンが inactive（ballot == 0）になったらワープ終了。grid 全体の同期は不要

**PhotonPool SoA との統合**：
- Persistent Warp はパーティクルプールの **IMC部分** のみを処理する
- Composite Key Sort（NUMERICS §6.5、CUDA_KERNELS §0.5）で IMC粒子を SoA 先頭に配置済み
- 粒子のload/storeは通常のSoAアクセスと同一（追加バッファ不要）
- 粒子終了時にSoAに書き戻し、新粒子をSoAからロード

**IMC/DDMCモード分離**：
Persistent Warp の導入により、active transport mode は IMC / DDMC の2値である。
`TransportMode::RW` は後方互換/予約のため残し、`TransportMode::Diffusion` は
post-radiation coupling と diagnostics のための cell-map 値として使うが、
現行 PGRW 実装は `imc_transport_persistent` 内の internal branch であり、
hybrid diffusion 分類セルも transport kernel では IMC/guard 扱いまたは deterministic `diff_E_` 扱いのため、粒子を独立 RW/Diffusion スライスへ分離しない。実行順序は：
1. `Radiation.diffusion.enabled=True` かつ 1D_SPH では、分類・entry/exit 表現変換後に `diffusion_source_solve_cuda(dt/2)` を実行し、thermal emission は diffusion cell を skip する
2. `composite_sort_and_partition`（R7）で dead除去 + セルソート + セルモード→粒子モード同期 + IMC/DDMC分離を一括実行
3. `imc_transport_persistent`（Persistent Warp）で IMC 粒子 [0..n\_imc-1] を処理
   ここで `tau_rw>0` かつ 1D_SPH 条件を満たす粒子だけ PGRW branch に入る。destination cell が diffusion cell の boundary crossing は packet を kill し、`face_current_in` と `face_current_step` に tally する
4. `ddmc_event_loop`（History-based）で DDMC 粒子 [n\_imc..n\_imc+n\_ddmc-1] を処理
5. `Radiation.diffusion.enabled=True` かつ 1D_SPH では、`deterministic_diffusion_step_1d()` で `diff_E_` を RKL2 更新し、`face_current_in` を source として取り込む。続けて `spawn_imc_from_diffusion_faces()` が diffusion-IMC interface の `face_current_out` を計算し、同量の IMC packet を adjacent IMC cell に生成する
6. DDMC/RW→IMC または diffusion-interface spawn が発生した場合のみ tail `imc_transport_persistent` を追加実行する。tail 中に diffusion へ戻った packet energy は tail 後に `diff_E_` へ直接加算する
   ただし `ddmc.implicit_diffusion=True` かつ Phase-1 対応条件では、
   この段階を `solve_ddmc_diffusion_1d()` に置き換え、DDMC 粒子スライスは破棄する
7. diffusion 有効時は後段の `diffusion_source_solve_cuda(dt/2)` を実行し、finalization では diffusion cell の `rad_E` に `diff_E_` を書き込む

**根拠**：
- IMC（~60 reg）と DDMC（~30 reg）のレジスタ要件が大きく異なる
- 分離により DDMC は `__launch_bounds__(128, 16)` で 100% occupancy を達成
- Persistent Warp のwork queueはIMC粒子数のみを対象とし、DDMC粒子は含まない
- ICF問題ではDDMCセルが空間的に集中するため、Composite Key Sortのモード分離は効率的

---

## 8. 元 `docs/ARCHITECTURE.md` の §5.6.4 GPUメモリ予算の粒子プールの注記

> **粒子数がメモリ支配的**：PhotonPool が全体の 95%+ を占める。
> `particles_per_cell_group` の設定がメモリ使用量を決定する。
> **ピーク時メモリ**（Composite Key Sort 中）：pool (92N) + double buffer (92N) + comp_key/perm (8N) + CUB temp (24N) = **216 bytes/particle**。
> 定常時は pool (92N) + CUB temp 分の Scratch ≈ **96 bytes/particle**。
> A100 (80GB) では定常 ~500M、**ピーク ~350M** 粒子が上限目安。
> V100 (32GB) では定常 ~200M、**ピーク ~140M** 粒子が上限目安。
> `max_pool_size` は定常値ではなくピーク値で設定すること。

---

## 9. 元 `docs/ARCHITECTURE.md` の §7.1.3 `Parallel::ParticleMigration` の本文

- EMIGRANT粒子の検出・パック・交換・展開・ソート
- IMC越境（NUMERICS §12.3.2）とDDMC越境リーク（NUMERICS §12.3.3）の統一処理
- SoA→AoSパック→MPI送受信→AoS→SoA展開
- 受信後のglobal_idソート（オプション、デバッグ用。既定OFF。§12.7.2）
- 移動先でのDDMC/IMCモード判定はRadiationモジュールに委譲

**ファイル**：`src/parallel/particle_migration.cuh`, `particle_migration.cu`

**ParticleEmigrant パック構造体**（NUMERICS §12.3.4 準拠、104 bytes/particle）：

```cpp
struct alignas(8) ParticleEmigrant {
    double position[3];     // 24 B  (R, Z, phi) [cm]
    double direction[3];    // 24 B  (dir_r, dir_z, dir_phi) [dimensionless]
    double energy;          // 8 B   [erg]
    double weight;          // 8 B   [dimensionless]
    double time_remain;     // 8 B   [s]
    double birth_energy;    // 8 B   [erg] 生成時エネルギー（R12 Russian roulette 参照、§6.3.4）
    uint64_t global_id;     // 8 B   RNG key derivation用
    uint32_t rng_counter;   // 4 B   rng_counter（Philox counter[0]）
    int32_t  cell_id_src;   // 4 B   送信元ローカルセルID
    uint16_t group;         // 2 B   群番号
    int8_t   sign;          // 1 B   粒子符号（+1 or -1）
    int8_t   leak_face;     // 1 B   0-3 (R_left,R_right,Z_bottom,Z_top) — §6.4.3 面規約。IMC/DDMC共通（P5がcell_id=-(100+face)からデコード）
    uint8_t  mode;          // 1 B   ParticleMode: 0=IMC, 1=DDMC, 2=RW（P6 セル再同定で mode 依存分岐に必須。CUDA_KERNELS §8.3 参照）
    int8_t   padding[3];    // 3 B   8-byte alignment用パディング
    // Total: 104 B = 13×8, alignas(8) で 8-byte aligned
    // MPI転送時は MPI_BYTE × 104 で送受信（MPI派生型不使用）
    // Note: dest_rank は別配列 int32_t dest_rank[capacity] で管理
    //       (rank 数 > 127 に対応するため本体に含めない)
};
```

**EmigrantBuffer 構造体**：

```cpp
struct EmigrantBuffer {
    ParticleEmigrant* data; // deviceメモリ：パック済みAoS粒子データ
    int32_t* dest_rank;     // deviceメモリ：宛先rank配列 [capacity]
    int     count;              // 現在の粒子数
    int     capacity;           // 確保済み容量
    int     per_dest_count[8];  // 宛先rank毎の粒子数（8方向、§7.1.1 neighbor_ranks準拠）
    int     per_dest_offset[8]; // 宛先rank毎のオフセット（ソート後）

    static constexpr int BYTES_PER_PARTICLE = sizeof(ParticleEmigrant);  // 104 bytes

    // 初期容量：parallel.migration.initial_capacity（既定 10000 粒子）
    // 不足時：growth_factor（既定 1.5）倍に動的拡張（cudaMalloc + cudaMemcpy + cudaFree）
    void resize_if_needed(int required);
};
```

**ParticleEmigrant (NUMERICS §12.3.4) と PhotonPool (§5.3) のフィールドマッピング**：

| ParticleEmigrant フィールド | PhotonPool フィールド | 備考 |
|---|---|---|
| `position[3]` | `pos_r`, `pos_z` | `position[2]=0` が通常（DDMC リークのみ非ゼロ） |
| `direction[3]` | `dir_r`, `dir_z`, `dir_phi` | 内部表現 (R,Z,φ) でパック |
| `energy` | `energy` | [erg] |
| `weight` | `weight` | 統計重み |
| `time_remain` | `time_remain` | 残存時間 [s] |
| `birth_energy` | `birth_energy` | [erg] 生成時エネルギー（R12 Russian roulette参照） |
| `global_id` | `global_id` | uint64 グローバル粒子ID |
| `rng_counter` | `rng_counter` | uint32 rng_counter（NUMERICS §12.3.4 準拠、curand_init 呼び出し時に uint64 へ暗黙昇格） |
| `group` | `group_id` | uint16 群番号 |
| `sign` | `sign` | int8 粒子符号（legacy path は +1） |
| `leak_face` | ― | リーク面方向（int8, 0-3。IMC/DDMC共通）。P5 が `cell_id=-(100+face)` からデコード |
| `mode` | `mode` | uint8 ParticleMode: 0=IMC, 1=DDMC, 2=RW。P6 セル再同定で mode 依存分岐に必須 |
| `cell_id_src` | `cell_id` | パック時設定（送信元 cellId） |
| `dest_rank` | ― | パック時に PartitionInfo から決定 |

**emigrant検出基準**（CUDA_KERNELS §9 越境粒子契約に準拠）：
- **統一契約**：R8（IMC）/ R9（DDMC）ともに、パーティション境界を越えた粒子は
  `cell_id = -(100 + leak_face)`, `alive = 1` で SoA に書き戻す。
  - `leak_face` の復元：`face = -(cell_id) - 100`（2D_RZ: 0=R_left, 1=R_right, 2=Z_bottom, 3=Z_top — CUDA_KERNELS §6.4.3 準拠）
  - 宛先 rank：`dest_rank = part.neighbor_ranks[face]`
  - 物理境界方向（`neighbor_ranks[face] < 0`）は R8/R9 内で境界条件処理済み（vacuum脱出/reflect）であり、emigrant にはならない
- **検出カーネル P5**（CUDA_KERNELS §8.2）：全 alive 粒子をスキャンし、
  `alive == 1 && cell_id < 0` の粒子を検出して ParticleEmigrant にパック。
  パック成功後に `alive = 0` に設定（以降の R7 で dead として除去）。
  1スレッド=1粒子、atomicAdd カウンタによる scatter write

**主要API**：
```cpp
// 戻り値：検出されたemigrant粒子数
int detect_emigrants(
    const PhotonPool& pool,         // 入力：粒子プール（read-only）
    const PartitionInfo& part,      // 入力：領域情報
    EmigrantBuffer& emigrants,      // 出力：パック済みemigrantデータ
    cudaStream_t stream
);

void exchange_emigrants(
    const PartitionInfo& part,
    CommBuffers& buffers,             // 送受信バッファ（GPU-aware MPI or host staging）
    const EmigrantBuffer& send,       // パック済み送信データ
    EmigrantBuffer& recv              // 受信データ（countが更新される）
);

void merge_immigrants(
    PhotonPool& pool,
    const EmigrantBuffer& recv,
    bool sort_by_global_id,
    cudaStream_t stream
);
```
