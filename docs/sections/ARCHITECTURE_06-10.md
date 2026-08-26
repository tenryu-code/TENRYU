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
（NUMERICS §12.2.1）。粒子移送モジュールは LEGACY-inactive（IMC 退役）。

### 7.1 並列モジュール（`src/parallel/`）

並列化機能は `src/parallel/` に集約する。以下の5モジュールで構成：

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

#### 7.1.3 `Parallel::ParticleMigration`
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

#### 7.1.4 `Parallel::CommBuffers`
- 送受信バッファの事前確保と動的拡張
- cell/nodeハロー用バッファとparticle migration用バッファを統合管理
- GPU-aware MPI時はデバイスメモリ、フォールバック時はpinned hostメモリ

**ファイル**：`src/parallel/comm_buffers.cuh`, `comm_buffers.cu`

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

    // 粒子移動用（宛先rank毎に動的拡張）
    DeviceArray emigrant_send;
    DeviceArray emigrant_recv;

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
- `io` は `parallel` に依存（並列HDF5出力のrank情報）
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
- **動的拡張**：粒子移動バッファは emigrant 数に応じて動的拡張
  - 初期サイズ：`parallel.migration.initial_capacity`（既定 10000）
  - 拡張：不足時に `ceil(capacity × parallel.migration.growth_factor)`（既定 1.5）で再確保
  - 粒子移動閾値 `emigrant_threshold`（既定1000）を超えた場合に警告
- **再利用**：バッファはステップ間で再利用（毎ステップ確保/解放しない）

### 7.5 ディレクトリ構造

```
src/parallel/
├── partition.cuh          # PartitionInfo, 分割計算API
├── partition.cu           # 分割計算実装
├── halo_exchange.cuh      # ハロー交換API
├── halo_exchange.cu       # pack/exchange/unpack実装
├── particle_migration.cuh # 粒子移動API
├── particle_migration.cu  # EMIGRANT検出/交換/merge実装
├── comm_buffers.cuh       # バッファ管理API
├── comm_buffers.cu        # バッファ確保/拡張実装
├── reduction.cuh          # Allreduce/Exscan API
└── reduction.cu           # MPI Allreduce/Exscan実装
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
    └── CellField/NodeField/CellFieldG/CellFieldMat、hydro_active、PhotonPool、Scratch、DeviceErrorFlags
    └── 全 CellField/NodeField/CellFieldG を **cudaMemset ゼロ初期化**（Qvisc=0, c_s=0, D_eff=0 等を保証。
        未初期化メモリの読み出しを防止。Step 9b の H13→H15 で c_s を正しい値に上書きする）
    └── laser_cache_valid=false（レーザーキャッシュ無効状態で開始）
 8b. [条件分岐] Main.restart_from が非空の場合 → リスタートモード:
    └── IO::load_checkpoint(restart_from) でチェックポイントHDF5を読み込み
    └── State フィールド（全 CellField/NodeField/CellFieldG）を復元
    └── PhotonPool を復元（SoA全フィールド + RNG state: curand_init(global_id ^ user_seed, **step+1**, rng_counter)
        ここで step はチェックポイントの time_state/step（最後に完了したステップ）。再開ステップ = step+1 のストリームを開始する）
    └── DDMC NaN sentinel 検証：mode==DDMC の粒子に対し pos_r/pos_z/dir_r/dir_z/dir_phi が NaN であることを
        アサート。NaN でない場合は強制的に NaN 設定し WARNING（旧チェックポイント後方互換。SPECIFICATION §7.4 step 3）
    └── PhotonPool メタデータ復元: N_total（配列長）はHDF5データセット次元から取得。
        チェックポイントは **alive 粒子のみ** を保存するため（SPECIFICATION §7.4 particles/ 参照）、
        復元後に alive[i]=1 を全粒子に設定し、n_alive = N_total とする。
        n_census = n_alive（再開直後は全alive粒子がcensus扱い。R5/R6がn_censusから新粒子数を決定）。
        capacity はチェックポイントの particles/pool_capacity から復元
    └── hydro_active フラグを復元
    └── 時間管理状態を復元（t, step, dt, t_next_plot, t_next_history, t_next_checkpoint）
    └── 累積診断値を復元（E_safety, E_numerical_loss, E_laser_deposited, E_laser_escaped, E_rad_escaped, E_floor_injected, E_pdV_bdry, E_Marshak_in, E_solver）
    └── Config の frozen パラメータを検証（メッシュサイズ、群数、材料数、**Main.seed** が一致することを確認）
    └── Main.seed 不一致の場合 ConfigError を送出（RNG ストリーム連続性の破壊を防止。NUMERICS §12.7.1）
    └── laser_cache_valid = false, laser_dep_frac をゼロクリア（リスタート時は必ず初回 full raytrace を実行。
        laser_dep_frac は stale キャッシュであり再構成が必要。SPECIFICATION §7.4 step 7、v1.0 必須ルール）
    └── Steps 9-10 をスキップ（チェックポイント値を使用）
        ただしこれは初期条件の幾何/場設定を省略する意味であり、幾何導出量（face_area/delta_l/ell_ddmc）は
        Step 14b で node 座標から再計算する
    └── Step 11 へ進む（reclosure は Step 14b で実行 — CommBuffers/halo が必要なため）
    └── 詳細手順は SPECIFICATION §7.4 restart 8-step プロトコル参照
    [非リスタート（新規実行）の場合 → Steps 9-10 で初期化:]
 9. Geometry関数の結果（Config.geometry の配列）を State フィールドへ格納
    └── ρ, Te, Ti, velocity, volFrac → 対応する CellField / NodeField / CellFieldMat へコピー
    └── H7(compute_cell_geometry) → vol, face_area, delta_l, ell_ddmc を計算（14b でゴースト充填後に再実行。ここでは owned セルの初期化）
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
    └── [ALL paths] H7(compute_cell_geometry)：x_r, x_z → vol, face_area, delta_l, ell_ddmc を再計算。
        チェックポイントは node 座標と vol のみ保存し、face_area/delta_l/ell_ddmc は保存しない（導出量のため）。
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

GPUカーネル内でのエラー（NaN検出、不正セルID等）は device-side assert ではなく
**エラーフラグ方式** で host に報告する：

```cpp
// グローバルメモリ上のエラーフラグ（State に含める）
// フラグ項目は atomicExch、カウンタ項目は atomicAdd で書き込み（thread-safe）。
// ステップ開始時に cudaMemset で 0 クリア。
struct DeviceErrorFlags {
    int32_t nan_detected;         // [flag: 0/1] NaN値を検出（atomicExch で設定）
    int32_t invalid_cell;         // [flag: 0/1] 不正セルIDを検出
    int32_t particle_overflow;    // [flag: 0/1] PhotonPool容量超過
    int32_t opacity_out_of_range; // [flag: 0/1] テーブル範囲外
    int32_t energy_violation;     // [flag: 0/1] エネルギー保存違反（|ΔE/E| > safety.energy_budget_tol）
    int32_t mesh_tangle;          // [flag: 0/1] メッシュ交差（ヤコビアン J < 0 検出）
    int32_t emigrant_overflow;    // [flag: 0/1] EmigrantBuffer容量超過（NUMERICS §12.3.1）
    int32_t infinite_loop;        // [flag: 0/1] 粒子イベント数がMAX_EVENTS超過（CUDA_KERNELS §6.4, §5.2）
    int32_t ddmc_reflect_leak;    // [flag: 0/1] DDMC R9でREFLECT/AXIS面が選択された（リーク係数異常、CUDA_KERNELS §6.5）
    int32_t eos_newton_nonconverge; // [flag: 0/1] EOS逆変換Newton法がMAX_ITER(20)到達（CUDA_KERNELS §2.4）
    int32_t eos_ion_negative;      // [flag: 0/1] SESAME イオンEOS差分(P_total-P_e)が負 → max(0)クランプ発生（NUMERICS §1.1.5）
    uint32_t temperature_overshoot; // [count] 最大原理違反セル数（atomicAdd で集計、NUMERICS §11.8）
    int32_t sound_speed_negative;   // [flag: 0/1] テーブルEOS由来 c_s² < 0 を max(0) クランプ（NUMERICS §1.1.6）
    int32_t ddmc_sigma_tot_zero;    // [flag: 0/1] DDMC Σ^tot ≤ Σ_floor → census化（NUMERICS §7.5）
    int32_t emigrant_invalid_face;  // [flag: 0/1] P5 face デコードが範囲外（CUDA_KERNELS §8.2）
    int32_t volfrac_degenerate;     // [flag: 0/1] 体積分率縮退（Σ volFrac ≈ 0 or 1成分 > 1、CUDA_KERNELS §0.6、NUMERICS §1.1.6）
    int32_t negative_source_dep;    // [flag: 0/1] source_injection で rad_dep or laser_dep < 0 検出（CUDA_KERNELS §0.6、NUMERICS §11.5）
    int32_t invalid_cell_id;        // [flag: 0/1] cell_id >= n_cells の粒子を検出（R12 russian_roulette での OOB 防止、CUDA_KERNELS §6.6）。R8/R9 のセル追跡バグを示唆
    int32_t invalid_boundary_code;  // [flag: 0/1] R8 IMC transport で未知の境界コード検出（CUDA_KERNELS §6.4）。R8 の new_cell エンコーディングバグを示唆
    // sizeof(DeviceErrorFlags) = 76 bytes, 4-byte aligned
};
```

**プロトコル**：
1. 各ステップの開始前に `cudaMemsetAsync(flags, 0, ...)` でクリア
2. カーネル内でエラー検出時に `atomicExch(&flags->nan_detected, 1)`
   （カーネルは中断せず、残りスレッドは正常完了を試みる）
3. カーネル完了後に `cudaMemcpy` でフラグを host へ転送
4. host 側で `Core::Error` がフラグをチェック：
   - `nan_detected`：spdlog で警告出力。`safety.nan_fatal=True`（既定）なら fatal。`energy_fatal` とは独立制御
   - `particle_overflow`：回復手順 — w_survive 2倍、N_p 50%削減、最大3ステップ。未解消→ERROR
   - `emigrant_overflow`：超過分の粒子は `alive=2` に設定し、エネルギーを `E_numerical_loss` に計上して消滅（cell_id<0 のため再追跡不能）。ステップ末に `capacity = ceil(capacity × parallel.migration.growth_factor)` で拡張し、WARNING を出力（NUMERICS §12.3.1、CUDA_KERNELS §8.2 P5 参照）
   - `energy_violation`：`safety.energy_fatal=True`時は即ERROR、False時はWARNING出力+diagnostics記録（NUMERICS §11.1準拠）
   - `temperature_overshoot`：`overshoot_count` として diagnostics/history に記録（MPI_Allreduce(MAX) によるランク最大値。カウント値のためグローバル合計ではなくランク最大値で閾値判定する）。`overshoot_max > safety.overshoot_warn`（既定 0.01）で WARNING、`safety.overshoot_fatal_enabled`（既定 False）かつ `overshoot_max > safety.overshoot_fatal`（既定 0.10）で FATAL（NUMERICS §11.8）
   - `mesh_tangle`：即ERROR（回復不能）
   - `sound_speed_negative`：WARNING出力 + diagnostics記録。c_s²<0セルは c_s=0 にクランプ済み（NUMERICS §1.1.6）
   - `ddmc_sigma_tot_zero`：WARNING出力 + diagnostics記録。Σ^tot≤0の粒子は即census化済み（NUMERICS §7.5）
   - `emigrant_invalid_face`：WARNING出力 + diagnostics記録。不正face粒子は消滅+E_numerical_loss計上済み（CUDA_KERNELS §8.2 P5）。R8/R9のcell_idエンコーディングバグを示唆
   - `invalid_cell`：`cell_search.fatal=True`（既定）なら ERROR（回復不能なセル探索失敗）。`False` なら WARNING + 粒子消滅 + E_numerical_loss 計上（NUMERICS §9.5, SPECIFICATION §6.4.7 cell_search.fatal 参照）
   - `infinite_loop`：WARNING出力 + diagnostics記録。MAX_EVENTS到達粒子は census 化済み（CUDA_KERNELS §6.4, §6.5）。多発時はΔtが大きすぎる可能性を示唆
   - `ddmc_reflect_leak`：WARNING出力 + diagnostics記録。反射/軸面でのリーク選択はリーク係数計算バグを示唆（CUDA_KERNELS §6.5、NUMERICS §11.4）
   - `eos_newton_nonconverge`：WARNING出力 + diagnostics記録。最終反復値を採用済み。多発時はEOS外挿による Newton 不安定を示唆（CUDA_KERNELS §2.4）
   - `eos_ion_negative`：WARNING出力 + diagnostics記録。P_ion=max(0, P_total-P_e) クランプ済み（NUMERICS §1.1.5）
   - `opacity_out_of_range`：WARNING出力 + diagnostics記録。テーブル端値でクランプ済み（NUMERICS §11.3）
   - `invalid_cell_id`：WARNING出力 + diagnostics記録。粒子は消滅+E_numerical_loss計上済み（CUDA_KERNELS §6.6 R12）。cell_id >= n_cells はR8/R9のセル追跡バグを示唆
   - `invalid_boundary_code`：WARNING出力 + diagnostics記録。粒子は消滅+E_numerical_loss計上済み（CUDA_KERNELS §6.4 R8）。未知の境界コードはR8のnew_cell処理バグを示唆
   - `volfrac_degenerate`：WARNING出力 + diagnostics記録。退化セルの混合則は volFrac をフロア値にクランプして計算済み（NUMERICS §1.1.6）。多発時は材料界面の解像度不足を示唆
   - `negative_source_dep`：WARNING出力 + diagnostics記録。負の rad_dep/laser_dep は source_injection でゼロクランプ済み + E_numerical_loss に計上。IMCタリーバグまたはレーザー吸収計算の数値誤差を示唆（NUMERICS §11.5）

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
