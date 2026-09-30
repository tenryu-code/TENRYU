<!-- 分割元: docs/ARCHITECTURE.md | このファイルは参照用です。原本（docs/ARCHITECTURE.md）が権威です。 -->
## 6. 依存方向（Dependency Direction）
循環禁止。依存は“矢印の方向”のみ。

```
core       ->  (none)
mesh       ->  (none)
parallel   ->  (core, mesh)
materials  ->  (core)
hydro      ->  (mesh, materials, core, parallel)
radiation  ->  (mesh, materials, core, parallel)
laser      ->  (mesh, materials, core, parallel)
coupling   ->  (hydro, radiation, laser, materials, mesh, core, parallel, verification)
diagnostics -> (state views including mesh geometry, core, verification)
verification -> (core, parallel)
io         ->  (state views only, core, parallel)
drivers    ->  (coupling, io, diagnostics, materials, core, parallel)
```

### 6.1 Hydro::ALE retry repair modes

`Hydro::ALE` owns the 2D_RZ rezone/remap path and the driver-requested local
repair ladder. `AleMode::AxisVariationalProjection` is the
axis-band escalation mode, default-off via
`Numerics.ale.axis_variational_projection_enabled` (SPECIFICATION §6.4.2).
Its ladder position is:

AxisSpinePlusLocal (first rung) → AxisVariationalProjection (the
axis-band escalation rung) → InteriorMultiNodeProjection (the interior
multi-node projection rung) → FullWinslow (terminal).

`AxisVariationalProjection` is implemented as a deterministic projection-style
half-space feasibility operator in `src/hydro/local_rezone.{cuh,cu}`. The
algorithm, constraint set, Picard schedule, telemetry
`record_kind="axis_projection_attempt"`, and documented carry-over items are
specified in NUMERICS §3.3.5 (Rezone制約 — axis variational projection 系小節).

（注記 2026-07-10: 本節は doc 監査で docs/sections/ARCHITECTURE_06-10.md split
にのみ存在していた孤児記述を正典へ移植したもの。）

### 6.1 Hydro::ALE retry repair modes

`Hydro::ALE` owns the 2D_RZ rezone/remap path and the driver-requested local
repair ladder. `AleMode::AxisVariationalProjection` is the
axis-band escalation mode, default-off via
`Numerics.ale.axis_variational_projection_enabled` (SPECIFICATION §6.4.2).
Its ladder position is:

AxisSpinePlusLocal (first rung) → AxisVariationalProjection (the
axis-band escalation rung) → InteriorMultiNodeProjection (the interior
multi-node projection rung) → FullWinslow (terminal).

`AxisVariationalProjection` is implemented as a deterministic projection-style
half-space feasibility operator in `src/hydro/local_rezone.{cuh,cu}`. The
algorithm, constraint set, Picard schedule, telemetry
`record_kind="axis_projection_attempt"`, and documented carry-over items are
specified in NUMERICS_03 §3.3.5a.

---

## 7. 並列モデル（MPI + CUDA）

**基本方針**：1 MPI rank = 1 GPU。空間メッシュの領域分割に基づく並列化。
数理詳細は NUMERICS.md §12 に定義。

**§7.0 v1 実装正規（Option C、M18 2026-07）**：v1 実装は NUMERICS §12.1.4a の
Option C（global-size 配列 + 所有窓 + 自然位置 ghost 帯）である。本節の
local 配列・local↔global 写像の記述は M18 前の設計案（写像は恒等）。
実装 API の対応：分割 = `parallel::PartitionInfo`（`split_axis` 均等分割）、
交換 = `src/parallel/halo_exchange.cu`（`exchange_cell_fields` /
`exchange_node_fields`（owner-overwrite）/ `exchange_cell_strips_scaled`
（per-cell 多要素）/ `sendrecv_add_planes(_asym)`（SN 界面 face 和完成））、
縮約 = `parallel::Reduction`（`allreduce_sum` / `allgatherv`）、
rank→GPU binding = local rank による `cudaSetDevice`。ghost_layers = 2
（NUMERICS §12.2.1）。光子粒子の移送モジュール（`particle_migration`）は退役したモンテカルロ輻射とともに 2026-09-29 に
`retired/radiation_monte_carlo/` へ移した。
1D の出力の集約（2026-09-26）：1D はスナップショット・チェックポイント・履歴をランク 0 が全長の配列から書くので、step 0 の出力の前と各ステップの終わり（エネルギーの集計の前）に、所有窓で更新される場（流体・閉包・セルとともに動く放射エネルギー・節点・流体の開始のフラグ）を全ランクで持ち主の値に集める（driver の `consolidate_1d_owned_lines`）。1D の熱伝導の陰解法とイオン熱伝導は、全ランクで全線を解く前に物質の線を集める（`conduction_replicate_1d_matter_lines`、係数は全線で計算）。

### 7.1 並列モジュール（`src/parallel/`）

並列化機能は `src/parallel/` に集約する。以下の4モジュールで構成（§7.1.3 は退役）：

#### 7.1.1 `Parallel::Partition`
- 領域分割の計算と分割メタデータ（`PartitionInfo`）の管理
- 1D_SPH：動径スラブ分割（NUMERICS §12.1.1）
- 2D_RZ：2Dデカルト分割（NUMERICS §12.1.2）、`MPI_Cart_create`
- 入力パラメータ：`parallel.decomposition.*`（SPECIFICATION §6.4.10）
- 最小セル数制約の検証（`min_cells_per_rank`）
- local↔global ID写像の提供

**PartitionInfo 構造体**（NUMERICS §12.1.4 準拠）：

```cpp
struct PartitionInfo {
    int rank;                       // MPI rank番号
    int n_ranks;                    // 総rank数
    int cart_coords[2];             // 2Dカート座標 [p_r, p_z]（1Dでは[p, 0]）
    int cart_dims[2];               // カートトポロジ [P_r, P_z]（1Dでは[P, 1]）

    // ローカルセル範囲（global indexing）
    int local_cell_range[2][2];     // [[ir_start, ir_end), [jz_start, jz_end)]
    int local_node_range[2][2];     // セル範囲+1（節点範囲）

    int ghost_layers;               // ゴーストレイヤー数（= 2、NUMERICS §12.2.1）
    int n_ghost_cells;              // ゴーストセル総数（derived: local_array_nr*local_array_nz - nr_local*nz_local）
                                    // 2D_RZ 1層: 2*(nr_local+nz_local)+4。1D_SPH 1層: 2。
                                    // **重要**: カーネルシグネチャ・バッファサイズ注記の `n_ghost` は
                                    // この `n_ghost_cells` のエイリアスであり、`ghost_layers` ではない。
    int nr_local, nz_local;         // ローカルセル数（ゴースト除く）

    // --- セルゾーン分類（compute-comm overlap 用）---
    // Phase B（§5.6.2）で使用。v1.0既定では逐次実行のため参照されないが、
    // Parallel.gpu_optimization.compute_comm_overlap=True で有効化される。
    uint8_t* cell_zone;             // [n_cells_local] deviceメモリ
                                    //   0 = 内部セル（ハロー非依存）
                                    //   1 = 境界セル（ハロー依存）
    // Parallel::Partition::init() で計算し、以後不変。
    // ALE はトポロジー変更なし → cell_zone の再計算不要。
    // 境界セル: Kershaw 9点ステンシルの近接1層要件に基づき、外周 ghost_layers 層のセル。
    // 1D_SPH: 左端/右端の ghost_layers セルが境界、残りが内部。
    // 2D_RZ: 四辺の外周 ghost_layers 層が境界、残りが内部。

    // 近傍rank（-1 = 物理境界で隣接rankなし）
    // 2D: 8方向（face 4 + corner 4）
    int neighbor_ranks[8];          // [left, right, bottom, top, NE, NW, SE, SW]

    // local↔global写像
    int global_offset_r, global_offset_z;  // local_i = global_i - global_offset
    int local_array_nr, local_array_nz;    // nr_local + 2*ghost_layers, nz_local + 2*ghost_layers

    MPI_Comm cart_comm;             // MPI_Cart_create で生成したコミュニケータ

    // 境界判定ヘルパー
    bool has_left_boundary() const { return neighbor_ranks[0] < 0; }
    bool has_right_boundary() const { return neighbor_ranks[1] < 0; }
    bool has_axis() const { return cart_coords[0] == 0; }  // R=0軸を持つか
};
```

**ファイル**：`src/parallel/partition.cuh`, `partition.cu`

**MPIタグ規約**（NUMERICS §12.2.5 準拠）：`tag = phase_id * 1000 + direction * 100 + field_id`。field_id は v1.0 ではパック交換のため常に 0。face方向は direction=0-3（LEFT, RIGHT, BOTTOM, TOP）、corner方向は direction=4-7（NE, NW, SE, SW）。phase_id はセルハロー=1, ノードハロー=2, cell_conduction=3, cell_radiation=4, cell_radiation_f_fleck=5, cell_ALE=6, emigrant=7, DDMCリーク=8。コーナー名との対応は `TR→NE(4)`, `TL→NW(5)`, `BR→SE(6)`, `BL→SW(7)` を固定する。

#### 7.1.2 `Parallel::HaloExchange`
- ゴーストセル/ゴースト節点のハロー交換管理
- Pack/Isend/Irecv/Waitall/Unpackの一連のパイプライン
- cell-centered交換とnode-centered交換の両方に対応
- フィールド選択的な交換（フェーズ毎に異なるフィールド集合を指定）
- GPU-aware MPI / host-staging フォールバックの自動切替

**ファイル**：`src/parallel/halo_exchange.cuh`, `halo_exchange.cu`

**主要API**：
```cpp
// パッキングレイアウト: cell-major（1スレッドが1ゴーストセルの全フィールドをパック）
// send_buf[i * n_fields + f] = field_ptrs[f][ghost_cell_index[i]]
// ここで ghost_cell_index は PartitionInfo のローカル配列範囲から計算される。
// ゴーストセルインデックス: face方向の最初の ghost_layers 行/列のセルID。
// field_ptrs は pinned host memory に配置された device ポインタ配列。
// pack カーネルが gather（デバイス→送信バッファ）、
// unpack カーネルが scatter（受信バッファ→デバイス）を実行する。
// cell-major レイアウトの理由: 1スレッドが1ゴーストセルの全フィールドを連続パック
// することでコアレスドアクセスを実現し、MPI 送受信は方向ごとに1回で転送できる。
void exchange_cell_fields(
    const PartitionInfo& part,
    CommBuffers& buffers,
    const double* const* field_ptrs,  // n_fields 個のデバイスポインタ配列（ホスト側 pinned memory）
    int n_fields,                     // フィールド数
    int field_size,                   // 各フィールドの要素数（ghost含む）
    cudaStream_t stream
);

// int8 フィールド（hydro_active 等）は exchange_cell_fields (double) とは別経路で交換する。
// hydro_active は各ステップ冒頭で int8 → double 昇格してバッファに含めるか、
// 専用の exchange_int8_fields を用いる（NUMERICS §12.2.2 hydro フェーズ参照）。
void exchange_int8_fields(
    const PartitionInfo& part,
    CommBuffers& buffers,
    const int8_t* const* field_ptrs,
    int n_fields,
    int field_size,
    cudaStream_t stream
);

void exchange_node_fields(
    const PartitionInfo& part,
    CommBuffers& buffers,
    const double* const* field_ptrs,  // n_fields 個のデバイスポインタ配列（ホスト側 pinned memory）
    int n_fields,                     // フィールド数
    int field_size,                   // 各フィールドの要素数（ghost含む）
    cudaStream_t stream
);
```

#### 7.1.3 `Parallel::ParticleMigration` — 退役

光子粒子（IMC・DDMC）の rank 間移動（EMIGRANT の検出・パック・交換・展開、`ParticleEmigrant`・`EmigrantBuffer`）。
2026-09-29 にコード（`particle_migration.{hpp,cu}`）とともに退役し、本節の記述を
`retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。

#### 7.1.4 `Parallel::CommBuffers`
- 送受信バッファの事前確保と動的拡張
- cell/nodeハロー用バッファを管理
- GPU-aware MPI時はデバイスメモリ、フォールバック時はpinned hostメモリ

**ファイル**：`src/parallel/comm_buffers.hpp`, `comm_buffers.cpp`

**基本型定義**：

```cpp
// RAIIラッパー：cudaMalloc 管理
struct DeviceArray {
    void*   ptr;            // デバイスメモリポインタ（cudaMalloc）
    size_t  size;           // 使用中バイト数 [bytes]
    size_t  capacity;       // 確保済みバイト数 [bytes]

    void resize(size_t new_capacity);   // 不足時にcudaFree+cudaMalloc（非同期不可）
    template<typename T> T* as() { return static_cast<T*>(ptr); }
    ~DeviceArray();         // cudaFree（RAIIで自動解放）
};

// RAIIラッパー：cudaMallocHost（page-locked）管理
struct PinnedArray {
    void*   ptr;            // ホストpinnedメモリポインタ（cudaMallocHost）
    size_t  size;           // 使用中バイト数 [bytes]
    size_t  capacity;       // 確保済みバイト数 [bytes]

    void resize(size_t new_capacity);   // 不足時にcudaFreeHost+cudaMallocHost
    template<typename T> T* as() { return static_cast<T*>(ptr); }
    ~PinnedArray();         // cudaFreeHost（RAIIで自動解放）
};
```

**CommBuffers 構造体**：

```cpp
// 近傍方向定数（PartitionInfo::neighbor_ranks と同一順序）
enum Direction : int {
    LEFT = 0, RIGHT = 1, BOTTOM = 2, TOP = 3,
    NE = 4, NW = 5, SE = 6, SW = 7
};
static constexpr int MAX_NEIGHBORS = 8;  // face 4 + corner 4

struct CommBuffers {
    // ハロー交換用（8方向：face 4 + corner 4）
    DeviceArray send_halo[MAX_NEIGHBORS];
    DeviceArray recv_halo[MAX_NEIGHBORS];
    // host staging（GPU-aware MPI非対応時のフォールバック）
    PinnedArray host_send[MAX_NEIGHBORS];
    PinnedArray host_recv[MAX_NEIGHBORS];

    void resize_if_needed(size_t required);
    bool gpu_aware_mpi;         // CMake検出結果

    // 1Dではdirection 0,1 のみ使用。2Dでは0-7 全使用。
    // neighbor_ranks[dir] < 0 の方向はバッファを確保しない（物理境界）。
};
```

> **8方向の根拠**：Kershaw 9点ステンシル（Appendix A）は対角隣接セルを参照するため、
> 2Dデカルト分割ではコーナーゴーストセルが必要。face方向（4）のハロー交換後に
> コーナー方向（4）を交換する2段階方式とする（NUMERICS §12.2.5）。
> 1D_SPHでは LEFT/RIGHT の2方向のみ使用。

#### 7.1.5 `Parallel::Reduction`
- MPI_Allreduce（エネルギー収支、粒子統計）
- MPI_Exscan（global_id offset計算、§12.7.1）
- 全rank一致チェック（分割メタデータ検証）

**MPI Reduction方式**：

標準 `MPI_Allreduce`（`MPI_SUM`）を使用する。浮動小数点加算の結合順序は
MPI実装依存であり最下位ビットの変動があり得るが、モンテカルロ法の統計的再現に影響しない。

> **旧設計からの変更**：bitwise再現のための Gather + Root逐次加算プロトコルは廃止。
> 標準 `MPI_Allreduce` の O(log P) レイテンシを活用する。

**適用箇所**：
- エネルギー収支の全ランク合計
- LaserMesh の Allreduce
- 粒子統計の全ランク集約

**ファイル**：`src/parallel/reduction.cuh`, `reduction.cu`

### 7.2 依存方向の更新

`parallel` モジュールの追加により、依存グラフ（§6）に以下の変更が入る：

- `parallel` は `core` と `mesh` に依存（分割にメッシュ情報が必要）
- `hydro`, `radiation`, `laser` は `parallel` に依存（ハロー交換・粒子移動を呼ぶ）
- `coupling` は `parallel` に依存（Strang splitting内の交換タイミング制御）
- `io` は `parallel` に依存（出力を書く rank 0 の判定に rank 情報を使う）
- `drivers` は `parallel` に依存（MPI初期化/終了）

> 注：`parallel` を含む最新の依存グラフ全体は §6 を参照。

### 7.3 GPU-aware MPI要件

- **検出（2段階）**：
  1. **コンパイル時**：CMake `try_compile` で `MPI_Send` にデバイスポインタを渡す小テスト
     → `TENRYU_GPU_AWARE_MPI_COMPILE` マクロ定義
  2. **ランタイム時**：`MPI_Init` 後に `MPIX_Query_cuda_support()`（OpenMPI）または
     環境変数 `MPICH_GPU_SUPPORT_ENABLED`（MPICH/Cray）をチェック
     → コンパイル時検出がTRUEでもランタイム検出がFALSEなら host-staging フォールバック
  - 最終結果を `CommBuffers::gpu_aware_mpi` フラグに格納
- **推奨環境**：
  - NVIDIA HPC SDK（OpenMPI + CUDA-aware）
  - Spectrum MPI（POWER系）
  - MVAPICH2-GDR
- **フォールバック**：GPU-aware MPI が利用不可の場合は host-staging
  （cudaMemcpy Device→Host→MPI→Host→Device）で動作
  - 性能低下は想定されるが、正確性には影響しない

### 7.4 バッファ管理

`CommBuffers`（§7.1.4）のメモリ管理方針：

- **初期確保**：分割メタデータから必要なハローバッファサイズを計算し確保
  - cell halo: \(n_{ghost} \times \max(n_r^{local}, n_z^{local}) \times n_{fields} \times 8\) bytes
  - node halo: 同上（節点数はセル数+1）
- **再利用**：バッファはステップ間で再利用（毎ステップ確保/解放しない）

### 7.5 ディレクトリ構造

```
src/parallel/
├── partition.hpp          # PartitionInfo, 分割計算API
├── partition.cpp          # 分割計算実装
├── halo_exchange.hpp      # ハロー交換API
├── halo_exchange.cu       # pack/exchange/unpack実装
├── comm_buffers.hpp       # バッファ管理API
├── comm_buffers.cpp       # バッファ確保/拡張実装
├── reduction.hpp          # Allreduce/Exscan API
└── reduction.cpp          # MPI Allreduce/Exscan実装
```

---

## 8. Namelist→Config→Run のフロー（実行時シーケンス）

```
 0. MPI_Init(&argc, &argv)
    └── CUDA device選択: cudaSetDevice(local_rank % n_devices)
 1. tenryu run namelist.py  → コマンドライン引数解析
 2. Core::Namelist が CPython を起動 (Py_Initialize) し namelist を実行
 3. Builder が全ブロックを検証し Config を構築（§4.1.2）
    3a. Python callable を評価し FrozenTable1D を構築（laser波形、Marshak温度、初期条件）
    3b. FrozenTable1D のデバイスメモリを確保し、データをコピー
    3c. Config 構造体を構築（PlanckTable は `cmd_run` の初期放射場設定、および Radiation driver の step-local cache で構築）
 4. Py_Finalize（以後Pythonは呼ばれない — この時点で全 callable は凍結済み）
 5. Freeze が namelist原文コピー + frozen config JSON を生成
 6. Parallel::Partition が Config.parallel から PartitionInfo を構築
    └── MPI_Cart_create、近傍rank特定、最小セル数検証
 7. Mesh 初期化：Config.mesh + PartitionInfo からローカルメッシュ構築
    └── ゴーストセル/ノードの確保、境界フラグ設定
 8. Materials 初期化：EOS/opacity テーブルロード → deviceメモリへ転送
    └── SESAME: xSESAME ASCII パース → 単位変換（K→eV, GPa→dyne/cm², MJ/kg→erg/g）→ EOSTable構築
    └── IONMIX: IONMIX v4 パース → EOSTable構築
    └── 推奨構成: SESAME EOS + IONMIX opacity（混合ロード）
 8a. State::allocate(cfg, part) で全フィールド確保（§5.2 ファクトリ関数）
    └── CellField/NodeField/CellFieldG/CellFieldMat、hydro_active、Scratch、DeviceErrorFlags
    └── 全 CellField/NodeField/CellFieldG を **cudaMemset ゼロ初期化**（Qvisc=0, c_s=0, D_eff=0 等を保証。
        未初期化メモリの読み出しを防止。Step 9b の H13→H15 で c_s を正しい値に上書きする）
    └── laser_cache_valid=false（レーザーキャッシュ無効状態で開始）
 8b. [条件分岐] Main.restart_from が非空の場合 → リスタートモード:
    └── IO::load_checkpoint(restart_from) でチェックポイントHDF5を読み込み
    └── State フィールド（全 CellField/NodeField/CellFieldG）を復元
    └── schema 1 のチェックポイント（2026-09-29 より前）は、光子粒子 `particles/` が空であれば読み、粒子があれば
        退役した `imc_ddmc` の run として再開を拒否する（`/holo`・`/difference`・`radiation/ddmc_flag`・
        `radiation/delta_E_rad_prev` は読まない。SPECIFICATION §7.5。光子粒子プールと RNG 状態の復元の手順は
        モンテカルロ輻射とともに退役した）
    └── hydro_active フラグを復元
    └── 時間管理状態を復元（t, step, dt, t_next_plot, t_next_history, t_next_checkpoint）
    └── 累積診断値を復元（E_safety, E_numerical_loss, E_laser_deposited, E_laser_escaped, E_rad_escaped, E_floor_injected, E_pdV_bdry, E_Marshak_in, E_solver）
    └── Config の frozen パラメータを検証（メッシュサイズ、群数、材料数、**Main.seed** が一致することを確認）
    └── Main.seed 不一致の場合 ConfigError を送出（燃焼の α 粒子 Monte Carlo の RNG ストリームの連続性を守る。NUMERICS §12.7.1）
    └── laser_cache_valid = false, laser_dep_frac をゼロクリア（リスタート時は必ず初回 full raytrace を実行。
        laser_dep_frac は stale キャッシュであり再構成が必要。SPECIFICATION §7.4 step 7、v1.0 必須ルール）
    └── Steps 9-10 をスキップ（チェックポイント値を使用）
        ただしこれは初期条件の幾何/場設定を省略する意味であり、幾何導出量（face_area/delta_l）は
        Step 14b で node 座標から再計算する
    └── Step 11 へ進む（reclosure は Step 14b で実行 — CommBuffers/halo が必要なため）
    └── 詳細手順は SPECIFICATION §7.4 restart 8-step プロトコル参照
    [非リスタート（新規実行）の場合 → Steps 9-10 で初期化:]
 9. Geometry関数の結果（Config.geometry の配列）を State フィールドへ格納
    └── ρ, Te, Ti, velocity, volFrac → 対応する CellField / NodeField / CellFieldMat へコピー
    └── H7(compute_cell_geometry) → vol, face_area, delta_l を計算（14b でゴースト充填後に再実行。ここでは owned セルの初期化）
    └── **mass 初期化**: mass[c] = rho[c] × vol[c]（Lagrangian保存量。Phase 1 H2 が読むため必須。
        block=256, grid=(n_cells+255)/256 の単純カーネル）
 9a. hydro_active フラグ初期化（NUMERICS §2.1.1）：
    └── T_start_eV == 0.0 → 全セル hydro_active = 1（常時有効）
    └── T_start_eV > 0.0  → 全セル hydro_active = 0（初期非活性）
    └── int8_t* を cudaMalloc で確保し、cudaMemset で初期化
 9b. Z̄ / A_eff 初期化 + EOS順方向評価：
    └── U8 compute_zbar<<<grid,256>>>: Z̄ と A_eff を初期化（fixed モード: n_mat==1 は Zbar_fixed を全セルに書込、
        n_mat>1 は材料別 Z̄_α = Z_α を混合平均。thomas_fermi/tabular モード: テーブル補間。
        H13 が Z̄ を参照するため **H13 より前に必須**。CUDA_KERNELS §7.6 参照）
    └── eos_forward<<<grid,256>>>（H13）：Te,Ti → ee,ei,Pe,Pi,Cv_e,Cv_i
    └── compute_sound_speed<<<grid,256>>>（H15）：Pe, Pi, ρ → c_s
    └── floor_clamp<<<grid,256>>>（U2）：ρ, Te, Ti のフロアクランプ（防御的安全ネット）
        Builder が初期条件 Te/Ti ≥ floor を検証するため（SPECIFICATION §6.4.2）通常はクランプ不要だが、
        浮動小数点変換誤差や callable の数値ノイズに対する安全策として実行する
    └── 初期温度から内部エネルギー・圧力・比熱・音速・フロアクランプを設定
    └── **注意**: C1(compute_spitzer_deff) は Step 14b で実行する（ghost Te が必要なため、
        Step 14 halo exchange の後でなければならない）
10. 初期放射場の設定：
    └── `cmd_run` が `evaluate_geometry(...)` の直後、最初の `Driver::run(...)` / 初期snapshot書き込み前に実行する
    └── "equilibrium"：PlanckTable を構築し、`rad_E[i,g] = b_g(Te[i]) × a_eV × Te[i]⁴`、`rad_E_old[i,g] = rad_E[i,g]`（熱平衡）
    └── "zero"：allocation のゼロ値を保持する（真空初期化）
    └── restart 時は checkpoint の `rad_E`/`rad_E_old` を使用し、この初期化をスキップする
11. Laser/BC 初期化（FrozenTable1D は Step 3 で凍結済み）
    └── Config 内の FrozenTable1D（既にデバイス上）を Laser/BC モジュールに参照渡し
    └── LaserMesh のグリッド構築
    └── 初期時刻 t=0 での波形評価（FrozenTable1D::eval(0.0)）
12. CommBuffers の初期確保（§7.4）
13. Scratch の確保（§5.5：全モジュールの最大必要量）
14. 初回ハロー交換（State フィールドの全交換。リスタート・非リスタート共通で実行）
14b. Post-halo 初期設定（halo exchange 後に ghost セルが充填された状態で実行）：
    └── [restart only] U8(compute_zbar)：A_eff 再計算（A_eff はチェックポイント非保存。
        Z̄ はチェックポイントから復元済みだが、U8 は A_eff も出力するため実行必須）
    └── [restart only] H13(eos_forward) → H15(compute_sound_speed)：
        Cv_e/Cv_i/c_s はチェックポイント非保存のため再計算が必要（ee/Pe は HDF5 hydro/ から復元済みだが、
        H13 で EOS 整合性を保証し Cv_e を取得する。C1 が Cv_e を参照するため H13 は C1 より前に必須）
    └── [ALL paths] H7(compute_cell_geometry)：x_r, x_z → vol, face_area, delta_l を再計算。
        チェックポイントは node 座標と vol のみ保存し、face_area/delta_l は保存しない（導出量のため）。
        リスタート時は復元した node 座標から再計算が必須。新規実行時は Step 9 で実行済みだが、
        Step 14 halo exchange でゴースト node が充填された後に再実行することで境界セルの幾何量も正確になる
    └── [ALL paths] C1(compute_spitzer_deff)：Te, ρ, Z̄, Cv_e, A_eff → D_eff。
        C1 は隣接セル Te から |∇T| を計算するため ghost データが必要（Step 14 後に実行必須）。
        D_eff=0 のまま Step 15 に進むと dt_cond=∞ となり伝導支配問題の初回ステップが過大になる
15. 初期 dt 計算：
    └── [新規実行] dt = min(dt_initial, CFL constraints)（NUMERICS §2.2(e)、dt_initial = dt.initial_s）
    └── [restart]  dt = min(checkpoint_dt, CFL constraints)（NUMERICS §2.2(e)、dt.initial_s は適用しない）
16. 初期 diagnostics 出力（step=0 の状態）
17. 初期 HDF5 出力（snapshot + frozen config + namelist copy）
18. main time loop（Coupling::Driver）
    └── §4.7 の Strang splitting を time step 毎に繰り返す
19. 最終出力 + checkpoint
20. MPI_Finalize
```

> **MPI_Init と CPython の順序**：MPI_Init は CPython 起動前に実行する。
> CPython が MPI を内部的に使用する可能性は低いが、
> rank番号に基づく出力制御（rank 0 のみ stdout）を namelist 実行前に確立するため。

---

## 9. 互換性ポリシー（入力API）
- `tenryu_namelist` のブロックAPIは **破壊的変更禁止**
- 既存引数の意味変更・削除は不可。追加のみ許可。
- 互換性は `examples/verification/namelist_api_smoke.py` でCIに組み込み、破壊を即検出する。

---

## 10. エラーハンドリングアーキテクチャ

### 10.1 GPUカーネルからのエラー報告

GPUカーネル内でのエラー（非有限の粒子・光線状態、不正セル等）は device-side assert ではなく
**エラーフラグ方式** で host に報告する（`src/core/device_error_flags.cuh`）：

```cpp
// 光線追跡と不透明度評価のカーネルが書き込む。
// フラグ項目は atomicExch で 1 を立て、infinite_loop と unresolved_quadrature は atomicAdd で数える。
// 呼び出し側がカーネル段の前に cudaMemset で 0 クリアする。
struct DeviceErrorFlags {
  int32_t nan_particle = 0;          // 非有限の粒子・光線状態
  int32_t invalid_cell = 0;          // 不正なセル番号・補間状態
  int32_t invalid_boundary = 0;      // 未知の境界コード
  int32_t opacity_out_of_range = 0;  // 不透明度評価で ρ ≤ 0 をクランプ
  int32_t infinite_loop = 0;         // MAX_RAY_STEPS の上限に達した回数
  int32_t unresolved_quadrature = 0; // 1D 特性線の光線追跡で、求積のパネル上限で受理した区間の数（誤差ではない）
};
// モンテカルロ輻射の pool_overflow・ddmc_sigma_tot_zero・roulette_kill は 2026-09-29 に外した
// （persistent loop の error_code のビット 3・6・7 は欠番。他のビットの位置は変えていない）。
```

**プロトコル**：
1. 呼び出し側（レーザーの光線追跡、不透明度評価）が段の前に `cudaMemset` でクリアする
2. カーネルは異常を見つけてもスレッドを中断せず、フラグを立てて処理を続ける
3. 段の後に host へ転送し、呼び出し側が判定する：
   - レーザー光線追跡（`log_laser_flags`、`laser.cu`）：`infinite_loop` は WARNING、`unresolved_quadrature` は WARNING
     （最初の 10 回と 1000 回ごと）、`invalid_cell` は WARNING（該当光線は未吸収として扱う）、`nan_particle` は
     WARNING のあと停止（`TENRYU_ASSERT`）。persistent loop ではフラグのビットを `error_code`（理由 2）に詰めて
     chunk を止める
   - 不透明度評価（`copy_and_check_flags`、`opacity.cu`）：`opacity_out_of_range` は WARNING

旧設計の表にあった他の検査はフラグ構造体ではなく、次の仕組みで行う：
- **状態量の非有限値と温度の行き過ぎ**：`driver_safety_audit.cu` が各物理段の前後で \(T_e, T_i, \rho, e_e, e_i\) の
  非有限値と \(\max T_e\) を GPU 上で集約する。`Numerics.safety.nan_fatal`（既定 True）なら非有限値で停止、
  `overshoot_fatal_enabled`（既定 False）なら \(\delta = (\max T_e^{n+1} - T_{\max}^n)/T_{\max}^n\) が
  `overshoot_fatal` を超えたとき停止する（NUMERICS §11.8）。輻射段の行き過ぎ（セル数と最大値）は history の
  `radiation/overshoot_count`・`radiation/overshoot_max`（2026-09-29 まで `mc/overshoot_*`）に記録し、`overshoot_warn` を超えたステップは WARNING
  （最初と 100 回ごと。persistent loop はこの量を計算しない）
- **エネルギー収支**：各ステップの `epsilon_budget` を history に記録し、`energy_fatal`（既定 False）なら
  `energy_budget_tol` を超えたとき停止する（NUMERICS §11.1）
- **床クランプ**：ステップのクランプ数が `clamp_warn_threshold` を超えると WARNING（1 回）、
  `clamp_fatal_threshold` を超えると停止する
- **1D の体積が非正（節点の交差）**：流体段が失敗を返し、driver がステップ前の状態へ戻して \(\Delta t/2\) で
  やり直す（`driver_full_step_retry_*`、上限回数または `dt.min_s` で停止）。persistent loop では理由 3 で止まる
- **EOS の逆変換の非収束**：閉包へフラグで返し、閉包が回数を数える（§4.3、`eos_newton_nonconverge` の
  フラグは無い）
- **SESAME のイオン圧の差が負**：クランプせずに保持し、節点の数を 1 回警告する（`eos_ion_negative` と
  `max(0, …)` は無い）
- 旧設計の `mesh_tangle`・`energy_violation`・`temperature_overshoot`・`sound_speed_negative`・
  `volfrac_degenerate`・`negative_source_dep`・`emigrant_*`・`ddmc_reflect_leak`・`invalid_cell_id`・
  `invalid_boundary_code` はフラグとして存在しない

> **device-side assert を使わない理由**：`__assert_fail` はデバイス全体を停止させるため、
> マルチGPU実行でデッドロックを引き起こす。エラーフラグ方式は graceful degradation を実現する。

### 10.2 Host側エラー階層

```
FATAL   → MPI_Abort（全rank停止）。回復不能エラー（MPI通信失敗、メモリ確保失敗）
ERROR   → 現ステップを中断し checkpoint 書き出し後に終了
WARNING → spdlog + diagnostics 記録。実行継続
INFO    → 通常ログ出力（rank 0 のみ stdout、全rank ファイル）
```

**ERROR手順**：flag設定 → Allreduce伝播 → `cudaDeviceSync` → checkpoint書出 → `Barrier` → ログ → `MPI_Finalize` → `exit(1)`。
**FATAL**：checkpoint省略 → `MPI_Abort`。

### 10.3 CUDA APIエラーチェック

全CUDA API呼び出しを `CUDA_CHECK` マクロで保護する。非同期エラーは `cudaGetLastError()` で捕捉。CUDAエラー → FATAL。

---
