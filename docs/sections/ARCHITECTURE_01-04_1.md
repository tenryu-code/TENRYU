<!-- 分割元: docs/ARCHITECTURE.md | このファイルは参照用です。原本（docs/ARCHITECTURE.md）が権威です。 -->
# TENRYU — ARCHITECTURE.md
本書はTENRYUの全体設計（モジュール境界、依存方向、データ所有権、並列/ GPU実行モデル）を定義する。

---

## 1. 設計原則（Design Principles）
1. **GPU-first（CUDA-only）**  
   主要計算（Hydro更新、FLD/S_N 輻射ソルバ（tridiagonal/CG solve・S_N sweep）、レーザーレイトレース）はNVIDIA GPU上で実行する。  
   公式サポートは **CUDAのみ**（他GPU/他バックエンドは対象外）。（旧原則の「IMC/DDMC粒子追跡」は退役 — §4.5 の現行輻射モデル注記参照。）
2. **局所性（Locality）**  
   大域通信（MPI）を減らす。（旧原則「DDMCは疎行列ソルバを避け厚領域を局所イベント処理へ落とす」は退役 — 現行 FLD/S_N は cuSPARSE tridiagonal / CG（AMGXオプション）等の線形解法を意図的に採用する。）
3. **明確なモジュール境界**  
   Hydro / Radiation / Laser / Materials / Mesh / Coupling / Diagnostics / IO / Driver を分離し、依存方向を固定（循環禁止）。
4. **再現性（Reproducibility）**
   現行の決定論輸送（FLD/S_N）+ 1D Lagrangian 経路は同一GPU・同一構成で run-to-run bit 恒等を検証 gate で確認する（既知例外は文書化: 1D の一部 host 集計 ledger ~1e-15 帯、2D_RZ の atomicAdd 順序由来 LSB 帯 — 社内の検証記録 の noise-band gate）。
   （旧原則: 退役 imc_ddmc モードでは MC の性質上 bitwise 再現を要求せず、同一seed・同一GPU構成での統計的再現（平均・分散一致）のみを保証していた。）
5. **入力は単一Python namelist（Smilei方式）**  
   すべてのシミュレーション条件は1つの `.py` に書く。  
   C++側はCPythonを埋め込み、namelistを実行して設定を構築する（実行中にPythonを呼ばない）。
6. **段階的拡張**  
   3D・LPI 波動レベル計算・核燃焼等は将来拡張。v1.0の境界/抽象化は将来追加を阻害しないように設計する。（FLD は現行の既定輻射モデルとして、CBET・ホット電子プリヒートは opt-in 機能として実装済み — 「将来拡張」リストから卒業。）

---

## 2. 実装言語・ビルド
### 2.1 言語
- **C++20**（必須）
- GPU：**CUDA C++**（必須）
- Python：**入力namelistの実行（埋め込みCPython）**、後処理・最適化で使用

### 2.2 主要依存

| ライブラリ | 最低バージョン | 備考 |
|-----------|--------------|------|
| CUDA Toolkit | **12.0+** | `atomicAdd(double*)` は compute capability 6.0+ で必須。12.0以降の CUDA driver API を想定 |
| C++ compiler | **C++20対応**（GCC 12+, Clang 15+, NVCC host compiler） | concepts, `<format>` は使用しない（fmt代替） |
| MPI | **MPI-3.1+** | `MPI_Iallreduce`, `MPI_Neighbor_alltoallv` を使用。GPU-aware MPI 推奨 |
| HDF5 | **1.12+** | 逐次版・並列版のどちらでもよい（出力は rank 0 だけが書き、MPI-IO・collective I/O は使わない。並列版を検出すると MPI をリンクする） |
| Python3 + 開発ヘッダ | **3.10+** | namelist埋め込み用 |
| **pybind11** | **≥ 2.11** | Python namelist → C++ Config 変換。ヘッダオンリー |
| fmt | 9.0+ | ログフォーマット |
| spdlog | 1.12+ | ログバックエンド |
| Catch2 | 3.0+ | 単体テスト |
| **CLI11** | **2.4+** | サブコマンドCLIパーサー。ヘッダオンリー。MITライセンス。FetchContent で取得 |
| NVTX | CUDA Toolkit同梱 | プロファイル用 |
| **cuRAND device API** | CUDA Toolkit同梱 | `curand_kernel.h` ヘッダオンリー。Philox4x32-10 デバイスRNG。`libcurand.so` リンク不要 |
| **Hypre**（オプション） | **2.25+** | 陰的拡散ソルバ（BoomerAMG + PCG）。`-DTENRYU_ENABLE_HYPRE=ON` で有効化。`--with-cuda --with-gpu-aware-mpi` ビルド必須。MIT/Apache-2.0 デュアルライセンス |

> **cuRAND device API 採用の理由**：Philox4x32-10 RNG をNVIDIA最適化実装で提供する。
> `curand_kernel.h` はヘッダオンリーであり、`libcurand.so` のリンクは不要（デバイスAPI のみ使用）。
> カスタムPhilox実装と比較して：(1) NVIDIA による統計品質検証済み、
> (2) `curand_uniform_double()` は Philox で1語（uint32）消費の \(U(0,1]\) を返す（退役した輻射 Monte Carlo の CPU 参照実装
> `philox_cpu.hpp` と一致していた — 同実装は `retired/radiation_monte_carlo/` へ移した）、
> (3) PTX intrinsics による最適化された乗算ラウンド実装。
> CUDA Toolkit に同梱されるため追加依存なし。ライセンスはNVIDIA EULA（ヘッダ使用、LICENSE.md参照）。

> **Hypre 採用の理由**：Kershaw 9点拡散方程式の陰的解法を提供する。
> STS（明示的、§4.4参照）ではコロナ領域の極端な剛性（\(N_{sub} > s_{max}^2/2\)）で
> ステージ数が上限に達する場合がある。Hypre の BoomerAMG（代数的マルチグリッド）+
> PCG は \(O(N)\) ソルバであり、Δtに対する伝導CFL制約を完全に除去する。
> GPU対応（HYPRE_MEMORY_DEVICE、`--with-cuda` ビルド）によりデバイスメモリ上で直接解ける。
> MIT/Apache-2.0 デュアルライセンスであり、BSD-3-Clause（TENRYU）と完全互換。
> LLNL による exascale 実証済み（Frontier, Summit）。

> **pybind11 採用の理由**：namelistのPython→C++橋渡しにpybind11を使用する。
> ヘッダオンリーライブラリであり、ビルド依存は最小限。
> `PyObject*` の直接操作（`PyDict_GetItemString` 等）と比較して、
> 型安全な変換とエラーハンドリングが自動化され、Namelist::Builder のコード量を大幅に削減できる。
> Python callable の評価は初期化時1回のみであるため、実行時オーバーヘッドは問題にならない。

> 注：Kokkos等の抽象化は **公式には採用しない**（CUDA-only方針を明確化するため）。  
> もし導入する場合も “CUDA backend固定” とし、他バックエンドはビルド無効にする。

### 2.3 ビルド（CMake）
- CMake + Ninja
- 代表：
  - `-DTENRYU_ENABLE_MPI=ON`
  - `-DTENRYU_ENABLE_HDF5=ON`
  - `-DTENRYU_ENABLE_PYTHON=ON`
  - `-DTENRYU_ENABLE_NVTX=ON`
  - `-DTENRYU_ENABLE_HYPRE=ON`（オプション、既定OFF。Hypre陰的拡散ソルバを有効化。FindHypre.cmake でパス検出。`HYPRE_DIR` 環境変数で手動指定可）
  - `-DTENRYU_RFA_V2_MODE={OFF,STUB,DUMMY_BUFFER,FULL}`（既定 `FULL`。radial Fourier audit v2 Heisenbug isolation builds: compiled out, no-op, dummy GPU buffer, or normal HDF5 output）

#### 2.3.0 Source revision for `tenryu --version`
- `cmake/SourceRevision.cmake` writes `${build}/src/drivers/generated/tenryu_source_revision.hpp` (the macro `TENRYU_SOURCE_REVISION`) at every build through the custom target `tenryu_source_revision`, a dependency of the `tenryu` executable; it rewrites the header only when the revision changes. Only the `tenryu` executable compiles `src/drivers/version_banner.cpp`, the one file that includes the header, so a new revision recompiles that file and relinks the executable; the libraries and the test executables do not change.
- The revision is the first line of a `SOURCE_REVISION` file at the top of the source tree (`tools/beta_export.sh` writes the exported commit there), else the commit of the git work tree whose top is the source tree (`+modified` when tracked files differ), else `unknown`. `version_banner()` (`drivers/version_banner.hpp`) appends it to the version for `--version`; `tenryu_version_string()` (`core/version.hpp`), written into the frozen configuration, does not carry it.

#### 2.3.1 Config/State ABI dependency policy
- `src/core/config.hpp` and `src/core/state.hpp` define host-side ABI-sensitive structs that are consumed by several C++ and CUDA translation units.
- The HydroConfig/AleConfig stale-object regression scope attaches explicit object dependencies through `cmake/ConfigAbiDeps.cmake` using `tenryu_attach_config_state_abi_deps(<target>)`.
- The helper appends CMake `OBJECT_DEPENDS` edges from each selected target source object to both headers. This supplements compiler depfiles so CMake/Ninja rebuilds affected objects when Config or State layout changes, without requiring manual object deletion.
- Extend the helper to any additional target where `ninja -d explain` or `ninja -t deps` shows Config/State ABI users are not rebuilt by compiler depfiles.

---

## 3. トップレベル構成（提案）
```
tenryu/
  CMakeLists.txt
  src/
    core/
      config_validate.hpp # pybind11-free Config invariant checks shared by Builder/tests
      units/
      profiler/
      namelist/          # CPython埋め込み + Namelist API (Smilei風)
      error/
    mesh/
    materials/
    hydro/
    radiation/
    laser/
    parallel/           # MPI領域分割・ハロー交換 (§7)
    coupling/
    diagnostics/
    verification/
    io/
    drivers/
  examples/
    implosion/
    verification/
    perf/
  tools/                 # Python後処理・最適化
    validation/
  docs/
  tests/
  retired/               # ビルドしない退役コードの保管（CMake は参照しない）
    radiation_monte_carlo/ # 退役した輻射 Monte Carlo 一式（IMC・DDMC・ランダムウォーク・HOLO・difference 定式化）の
                           # ソース・試験・デッキ。README に最後にビルドと試験が通った状態と復元手順
```

**ファイル拡張子規約**：
- CUDA ソースファイル：`.cu`
- CUDA ヘッダファイル：`.cuh`
- 純C++ヘッダ（CUDA非依存）：`.hpp`（例：Config構造体、ユーティリティ）
- 理由：NVIDIA標準規約に準拠。nvcc、IDE、Nsight全てが `.cuh` を正しく認識する

---

### 3.1 Studio deck import

`gui/src/core/deck/loadDeck.ts` は、ファイル読込と貼付けによる取込を共通化する。
v1 の状態ヘッダがある場合は従来の復元経路を使い、ヘッダがない場合は同梱の
`tools/assist/assist.py import-deck` をタイムアウト付きの Python 子プロセスで実行する。
`gui/src/ui/DeckLoadDialog.tsx` で作業ディレクトリー・環境変数・実行先を指定し、
ローカルまたは選択中のサーバーで評価する。サーバーでは同梱ハーネスを一時転送する。
評価設定は取込状態と保存ヘッダに記録する。`deckImport.ts` は記録した kwargs をフォームへ対応付け、
対応付け・標本点での検証・元ソースでの保持・省略の対象を管理する。
`deck_import_runtime.py` は元ソースの保持または省略規則が必要な場合だけ
生成 deck に埋め込み、入力の初期化時に実際の namelist ブロックを呼び出す。
フォームの値は取込時に固定し、元ソースで保持する設定は solver 実行時の
作業ディレクトリーと環境に従う。再実行ヘルパーはこれらを変更しない。
setter の初期化や入れ子の更新が呼出し順序に依存する場合は、反復呼出しの
順序も保持する。solver の時間ループと C++ namelist API は変更しない。
プロトコルと忠実度の検証範囲は `docs/gui/DECK_IMPORT.md` を参照。

## 4. モジュール一覧と責務
### 4.1 core/
**責務**：共通ユーティリティ（RNG、単位、プロファイル、エラー）

- 乱数：**cuRAND device API** の Philox4x32-10（counter-based）を、現行では燃焼の α 粒子 Monte Carlo 輸送
  （`src/burn/mc_transport.cu`）だけが使う。粒子ごとに `curand_init(Main.seed ^ global_id, step_index, 0, &state)` で
  初期化し、粒子固有の `global_id` とタイムステップ番号でストリームを分ける（並列度に依存しない再現性、NUMERICS §12.7.1）。
  - 退役した輻射 Monte Carlo 用の共通部品（`src/core/rng/rng_init.cuh` の `init_philox`・`rng_init_kernel`、
    CPU 参照実装 `philox_cpu.hpp`、PhotonPool の `rng_counter` から cuRAND state を O(1) で復元する規約）は
    2026-09-29 に `retired/radiation_monte_carlo/src/core/rng/` へ移した（ビルドしない）。
- `Core::Units`：cgs+eV単位変換（NUMERICS §0.1 準拠。入力は人間に優しい単位も許容）
- `Core::Profiler`：NVTXレンジ、CUDA event timer
- `Core::Error`：NaN検出、assert、fatal、例外境界
- `Core::RadiationGroupStructure`（`src/core/radiation_group_structure.hpp`）：
  namelist validation and restart-safe config derivation for pure radiation
  group-boundary math. It contains no runtime radiation state and has no
  dependency on `radiation/`.
- `Core::AutoZone`：1D_SPH用自動等質量ゾーニング（NUMERICS §3.1.12）
  - `auto_zone.hpp`：`AutoZoneRegion`, `AutoZoneConfig`, `AutoZoneDiagnostics` 構造体、`compute_auto_zone_nodes()` API
  - `auto_zone.cpp`：等質量球殻分割、非対称幾何級数ブリッジ、二分法 \(q\) 求解、制約調整、ファイナライゼーション
  - 初期化時に `Namelist::Builder` から呼び出され、生成されたノード配列を `MeshConfig::explicit_nodes` に格納。ランタイムでは使用されない
- `Core::MeshRequirement`（`src/core/mesh_requirement.{hpp,cpp}`、NUMERICS §3.1.0c）: 1D 初期メッシュの物理由来分解能要求（レーザー波形・波長・材料層・幾何 → アブレート帯の面密度質量天井プロファイル・衝撃波分離天井・層則・推奨帯）。ホスト専用・Python 非依存。
  - `build_mesh_requirement()`：メッシュ非依存の見積もり（`FrozenTable1D` の出力波形と piecewise 一定の ρ₀(r)/材料(r) を入力）
  - `check_mesh_requirement()`：ノード列とセル別 ρ₀/材料に対する判定、`mesh_requirement_json()`：決定論 JSON
  - `tools/assist/mesh_recommendation.py` fits the shipped convergence table; `recommend_mesh.py` constructs and validates Mesh blocks. Optional `MeshEmpiricalRequirement` provenance is carried through config, enforce, freeze and JSON; `deck_lint.py` independently recomputes the recommendation. No solver-runtime Python dependency.
  - 呼び出し元: `Namelist::Builder`（`apply="enforce"` の `zoning_intent` 帯注入 — `zoning_intent::measure_fraction_at` で測度分数へ換算）、`drivers/cmd_validate`（`--mesh-preview` の `mesh_requirement` 項と `[mesh-requirement]` 判定）、`drivers/cmd_run`（run 開始時の `mesh_requirement.json`）。ランタイム物理では使用されない。1D ノード列は `mesh::build_1d_radial_nodes()`（`src/mesh/radial_nodes_1d.cuh`、`create_mesh` と同一実装）で得る。
- `Core::DeviceScratch`（`src/core/device_scratch.{hpp,cu}`、per-call cudaMalloc/cudaFree を scratch pool に置換する host オーバーヘッド削減）：
  プロセス寿命・タグ指名・grow-only のデバイス／ピン止めホスト scratch プール
  （`device_scratch_acquire(tag, bytes)` / `host_pinned_scratch_acquire`、内容はゼロ化されない
  = cudaMalloc と同一契約、単一ホストスレッド前提、解放は `device_scratch_shutdown()` のみ）。
  1D は step 経路の生 cudaMalloc/cudaFree 対を直接置換（最初の scratch-pool 適用）。その 2D 拡張（2026-07-07）は
  RAII ラッパ三種（`core::Field1D<Tag>` / `core::DeviceArray<T>` / `parallel::DeviceArray`）に
  **opt-in の pool-tag コンストラクタ**を追加する形で行う：`core::CellField1D f{"mod:purpose"};`
  は reset()/operator= がプールから取得し（ゼロ初期化契約は明示 memset で維持）、デストラクタは
  no-op（プール解放禁止）、move はタグごと移譲、pooled Field1D の `resize()`
  （prefix 保存 grow）は grow-only プールと両立しないため assert 禁止。タグは呼び出し点毎に一意
  （同時生存バッファは別タグ必須）、既定コンストラクタの非 pooled 経路はビット恒等で不変。
- `core/device_ordered_sum.cuh`（2026-10-02 に laser/ から移動）：host のループの順序を再現する device の部品 —
  ブロック内の排他的接頭和、ゼロを飛ばした添字順の和（+0 から始まる和は不変）、x86-64 の glibc と同じ規則の
  fmax/fmin（NaN は無視、等しい 2 値は後者）。1D の laser・burn・diagnostics・hydro（粘性の history 集約）・
  S_N（体積源）が使う。
- `core/glibc_libm_device.cuh`・`core/glibc_libm_host.hpp`（2026-10-02）：x86-64 の glibc（2.28 以降。ifunc が FMA と
  AVX2 の CPU で選ぶ FMA 版）の exp・log・pow と同じ結果を返す device 関数（Arm optimized-routines v19.11 の
  アルゴリズムと表、MIT。表の 931 値は glibc 2.39 の libm と一致を確認、積和の融合は glibc 2.39 の機械語に合わせる）。
  27 億点の引数（全ビットパターン・端点・指数ごとの範囲）で NaN の中身までビット一致（RTX 4090、glibc 2.39）。host の
  計算を device へ移して結果を変えないために使う（TMAT 以外の材料の Thomas–Fermi \(\bar Z\)、レーザー注入の解析的 \(T^4\)
  閉包、高波数速度ダンパーの front mask、S\(_N\) 1D の Marshak 境界。1D ALE の再配置 candidate も使っていたが、1D ALE は
  同日に退役した）。
  `host_has_reproduced_build()` はその host（glibc ≥ 2.28 の x86-64、FMA と AVX2）かを返し、host と比べる試験が使う。
- `Core::FieldMeasure`（`src/core/field_measure.{hpp,cpp}`、ALE P0A F2 — 設計
  社内の設計メモ ale_asymmetric_robust_design_20260727.md §2 F2）：場ごとの転送契約
  （support／測度／保存則／bounds／再構成次数／転送種別／epoch 依存）を宣言する
  fail-loud レジストリと 11 エントリの中核 seed table。二重質量分離
  （`subcell_mass`=overlay 積分保存 vs `kinematic_node_mass`=基底再構築）と FIX-2 測度
  教訓（物理 RZ 体積と平面面積は交換不能）をコード上の契約として固定する。宣言のみ
  （P0A）— transaction 転送層への enforcement 接続は P0B。
- `Core::MeshTransaction`（`src/core/mesh_transaction.{hpp,cu}`、ALE P0A F3 — Layer-T
  scaffold、社内の設計メモ q10_shadow_transaction_layerT_20260727.md）：typed mesh event
  （7 種 `MeshEventKind` + client kind 対応表 + per-kind 契約 C_e）と `ShadowTransaction`
  （単一 256B 整列 device arena への byte-exact D2D capture／commit、discard=rollback、
  fail-closed gate 台帳、transaction-scoped telemetry、failure-injection plumbing、
  非 support 領域の FNV-1a device hash）。単独基盤のみ — reference-barrier の移行は
  T-v1a（別コミット）。

#### 4.1.1 core/namelist（最重要）
**責務**：単一 `.py` namelist を実行し、C++側の `Config` を構築する。

- `Namelist::Runtime`
  - CPython初期化（`Py_Initialize`）
  - `sys.path` 設定（namelistのディレクトリ、TENRYUのpythonモジュール）
  - namelist実行（例外を捕捉し、ユーザ向けに整形して出す）
- `Namelist::API`（Pythonへ公開する関数群）
  - `Main(...)`, `Mesh(...)`, `Materials(...)`, `Geometry(...)`, `Radiation(...)`, `Laser(...)`, `Numerics(...)`, `Output(...)`, `Diagnostics(...)`, `Parallel(...)`
  - これらは呼び出されるとC++側のBuilderへ値を格納する（Smileiのブロック方式）
- `Namelist::Builder`
  - バリデーション（型・必須引数・単位・範囲）
  - pybind11 非依存の cross-field invariant は `core/config_validate.hpp` の helper を呼ぶ
  - 既定値の適用
  - python callable（密度/温度/波形）の “凍結”
- `Namelist::Freeze`
  - 実行したnamelistの原文コピー
  - すべての設定を"純データ（JSON）"に落とした frozen config を生成
  - 出力HDF5へ保存（再現性）
- `Namelist::GeometryVolumeCut`
  (`src/core/namelist/geometry_eval_volume_cut.{hpp,cpp}`)
  - The PLIC-enabled t0 material volume-cut sampler.  It is called only
    from initial geometry evaluation, samples already-frozen geometry
    callables, and writes volume-averaged rho/Te/Ti/material fractions into
    `State`; it has no runtime Python dependency.

**重要方針：実行中にPythonを呼ばない**

Python callable は初期化時に **一括評価しテーブル化** する。3種類の凍結パターン：

| callable種別 | 凍結方法 | テーブル型 | 補間方法 |
|-------------|---------|----------|---------|
| geometry関数（`density(r)`, `temperature(r,z)` 等） | メッシュ座標配列を渡し一括評価 | **直接State配列** | 補間なし（セル値として直接格納） |
| laser波形（`power(t)`） | 時間グリッドでサンプル → テーブル化 | `FrozenTable1D` | piecewise linear |
| 境界温度（`T_{r,f}(t)` Marshak用、面別） | 時間グリッドでサンプル → テーブル化 | `FrozenTable1D` | piecewise linear |

> **開発マイルストーン注記**：M01 では callable 評価をまだ行わず、識別メタデータのみを freeze 出力する。
> 本表の「一括評価しテーブル化」は M02 以降の挙動を示す。

```cpp
// 1D piecewise linear テーブル（device上で使用可能）
struct FrozenTable1D {
    double* x;       // 独立変数（時刻等）[n_points]、deviceメモリ [s]
    double* y;       // 関数値            [n_points]、deviceメモリ [erg/s] or [eV] etc.
    int     n_points;
    double  x_min, x_max;  // clamp用範囲 [s]（x[0], x[n_points-1] と一致）

    // device function: piecewise linear interpolation
    __device__ double eval(double xi) const;
};
```

**サンプリングパラメータ**：
- laser波形：時刻 \(k \cdot 2^{-40}\) s（\(k = 0..\lceil t_{end}/2^{-40}\rceil\)）に局所細分（弦と中点の差 > 局所値の 1e-6、最小 \(2^{-47}\) s）を加えた標本（`core/namelist/frozen_table.cpp` `create_frozen_time_table`）
- 境界温度：同上。検証用途のため精度よりもシンプルさを優先
- geometry関数：メッシュ座標数 = セル数（or ノード数）の一括評価。テーブル化不要

→ これにより「性能」と「決定性」を守る。

#### 4.1.2 Config 構造体

`Config` は namelist のパース結果を保持する中心データ構造であり、全モジュールの初期化入力となる。
各namelist block（SPECIFICATION.md §6.4）に対応するサブ構造体を持つ。

```cpp
struct Config {
    // --- SPECIFICATION.md §6.4 の各ブロックに対応 ---
    struct MainConfig {
        std::string name;        // シミュレーション名
        int    dim;              // 1 or 2
        std::string geometry;    // "1D_SPH" or "2D_RZ"
        double t_end;            // 終了時刻 [s]
        int    max_steps = 10000000; // 最大ステップ数（既定 10^7、SPECIFICATION §9.1）
        uint64_t seed = 12345;       // RNG グローバルシード（既定 12345、SPECIFICATION §9.1）
        std::string restart_from; // リスタートファイルパス（空=新規実行）
        std::string units = "cgs_eV";       // 単位系（v1.0固定、ドキュメント用。SPECIFICATION §9.1）
        std::string verbosity = "normal";   // "quiet" | "normal" | "verbose" | "debug"（SPECIFICATION §9.1）
    } main;

    struct MeshConfig {
        int    nr, nz;           // セル数（1Dではnzは無視）
        double r_min, r_max;     // 動径範囲 [cm]
        double z_min, z_max;     // 軸方向範囲（2Dのみ）[cm]
        std::string grid_type_r = "graded";   // 1D_SPH は常に graded、2D_RZ は uniform 固定
        std::string grid_type_z = "uniform";  // 2D RZ only
        std::vector<GridSegment> grid_segments;
        GradingConfig grading;
        std::string motion = "lagrangian";   // "lagrangian" | "ale"（SPECIFICATION §6.4.2）
        std::string logical_mesh_2d = "rectangular_rz"; // "rectangular_rz" | "spherical_polar_halfplane"
        std::string polar_center_treatment = "annular"; // "annular" | "tri_fan" for spherical_polar_halfplane
        // 注: SPECIFICATION §9.1 の次元依存既定: 1D_SPH="lagrangian", 2D_RZ="ale"
        // init時に Config::apply_dimension_defaults(dim) で上書きされる
        struct RezoningConfig {
            bool enabled = false;            // 2D_RZ ALE rezoning有効化（motion="ale"時のみ使用。既定は無効）
            int every_n_steps = 5;           // rezoning頻度 [cycles]（SPECIFICATION §6.4.2 既定 5）
            int warmup_steps = 0;            // reserved guard [cycles]（SPECIFICATION §6.4.2 既定 0）
            double relaxation = 0.2;         // reserved relaxation factor
            double spacing_ratio_threshold = 1.5; // reserved mesh-spacing threshold
            int max_iterations = 20;         // Winslow Jacobi最大反復数（SPECIFICATION §6.4.2 既定 20）
            double quality_threshold = 0.2;  // [dimensionless] メッシュ品質閾値（SPECIFICATION §6.4.2 既定 0.2）
            double max_displacement_fraction = 0.5; // [dimensionless] 最大変位率（SPECIFICATION §6.4.2 既定 0.5）
            std::string remap_limiter = "van_leer";
            bool remap_ms_midpoint = false;
            bool remap_ms_post_check = false;
            int remap_ms_post_max_iter = 3;
            double remap_ms_rescale_floor = 0.01;
            bool conservative_remap_enabled = false;
            std::string conservative_remap_target = "reference";
            bool conservative_remap_radiation_enabled = true;
            bool multiblock_cross_seam_rezone_enabled = false;
            bool ke_fixup = true;
            int shock_sensor_guard_cells = 2;
            double density_jump_threshold = 0.1;
            double Te_jump_threshold = 0.2;
            double convergence_tol = 1e-6;   // [dimensionless] 収束判定（SPECIFICATION §6.4.2 既定 1e-6、NUMERICS §3.3.3）
        } rezoning;
    } mesh;

    struct MaterialsConfig {
        struct MatDef {
            std::string name;               // 材料名（例 "CH", "DT"）。一意必須（SPECIFICATION §6.4.3）
            double A, Z;                    // 質量数 [amu]、原子番号 [dimensionless]
            std::string eos_model = "ideal_gas";       // "ideal_gas"（既定） / "sesame" / "ionmix" / "tmat" / "power_law_te"
            std::string opacity_model = "constant";    // "constant"（既定） / "table_nlte" / "tmat" / "power_law" / "freq_dep_marshak" /
                                                       // "ionmix"（LTE の IONMIX 表。builder が "table_nlte" の経路へ変換）。
                                                       // "sesame" は受理されない（502/505 は読めるが実行時の不透明度に使わない）。
                                                       // 旧名 "none" は "constant" + kappa=0 に変換、WARNING出力
            double ideal_gas_gamma = 5.0/3.0; // [dimensionless] eos_model="ideal_gas"時のγ（SPECIFICATION §6.4.3）
            double cv_e_override = -1.0;    // [erg/(cm³·eV)] 電子比熱オーバーライド（-1=テーブル使用。SPECIFICATION §6.4.3）
            double kappa_a_constant = 0.0;  // [cm²/g] opacity_model="constant" 時の吸収不透明度
            double kappa_s_constant = 0.0;  // [cm²/g] opacity_model="constant" 時の散乱不透明度
            std::string eos_file;           // テーブルファイルパス（SESAME xSESAME ASCII / IONMIX v4/v6 .cn4 バイナリ）
            std::string opacity_file;       // 不透明度テーブルファイルパス
            // SESAME 固有パラメータ（eos_model="sesame" 時のみ使用）
            int sesame_material_id = -1;    // SESAME 材料番号（例: CH=7593, DT=5265）
            int sesame_cold_curve_rows = 12; // T=0 の cold curve を保つ合成行の数（0 は従来どおり捨てる。NUMERICS §1.1.5(b)）
            // 形式（xSESAME ASCII）と表番号（301 total / 304 electron）は固定で、namelist のキーではない
            // （sesame_format / sesame_table_total / sesame_table_electron は ConfigError）。304 が無い材料は、
            // 材料の電離モデルの Z̄ で全表を節点ごとに Z̄/(1+Z̄) に分割して電子表を作る（2026-09-29）
            bool is_void = false;           // true = 真空（void）材料。EOS/opacity テーブル不要（SPECIFICATION §6.4.3）
        };
        std::vector<MatDef> materials;      // 材料リスト（最大 MAX_MATERIALS=8）
        MixingRule opacity_mix_rule = MixingRule::LINEAR_MASS; // spec §6.4.3; enum定義は§4.3参照
        // Materials.mixture は opacity_mix_rule だけを保持する（MaterialsConfig::opacity_mix_rule）。fractions と
        // eos_mix_rule は受け付けるが警告を出して無視する — Geometry.volfrac は常に体積分率で、1D の EOS は各セルの
        // 支配材料で閉じる（NUMERICS §1.1.5(c)）
        struct ZbarConfig {
            std::string model = "fixed";    // "fixed" / "thomas_fermi" / "tabular"（既定 "fixed"、SPECIFICATION §9.1）
            double fixed_value = -1.0;      // \>= 0 なら全材料・全非 void セルの Z̄ をこの 1 値で上書き（fixed 以外のモデルの初期値にも。既定 -1 は無効）
            std::string table_file;         // model="tabular" 時のテーブルファイルパス
        } zbar;
        struct VoidConfig {
            double rho = 1e-10;             // [g/cm³] void セル密度下限（SPECIFICATION §6.4.3）
            double Te  = 1e-3;              // [eV] void セル電子温度下限
            double Ti  = 1e-3;              // [eV] void セルイオン温度下限
        } void_config;
    } materials;
    // --- Void helper ---
    // int first_nonvoid_material_index() const;
    //   materials.materials[] で最初の !is_void な材料のインデックスを返す。
    //   全材料がvoidの場合は -1 を返す。
    //   EOS/opacity テーブル参照が単一材料前提のコールサイトで使用（SPECIFICATION §6.4.3）。

    struct GeometryConfig {
        // 凍結済み初期条件（evaluate後にState配列へ直接格納）
        // Builder がPython callable を評価し、結果の配列をここに保持
        std::vector<double> density;        // [n_cells] セル毎の初期密度 [g/cm³]
        std::vector<double> Te, Ti;         // [n_cells] セル毎の初期温度 [eV]
        std::vector<double> velocity_r, velocity_z;  // [n_nodes] ノード毎の初期速度 [cm/s]
        std::string radiation_field = "equilibrium";  // "equilibrium" / "zero"（SPECIFICATION §6.4.4 既定 "equilibrium"）
        // --- 多材料初期条件（SPECIFICATION §6.4.4、NUMERICS §1.1.5 (c)）---
        bool enforce_sum_to_one = true;     // 体積分率 sum=1 制約を強制するか（SPECIFICATION §6.4.4 既定 True）
        std::vector<std::vector<double>> volfrac; // [n_mat][n_cells] 材料体積分率。enforce_sum_to_one=True で正規化（SPECIFICATION §6.4.4 volfrac）
                                                  // Builder で Python callable を評価し、State.volFrac へ投入（Step 9）
    } geometry;

    struct RadiationConfig {
        bool enabled = true;                  // 輻射輸送有効化（SPECIFICATION §6.4.5 既定 True）
        RadiationMode mode = RadiationMode::MultigroupDiffusion; // "multigroup_diffusion" / "sn_transport"（"imc_ddmc" は ConfigError）
        bool origin_parity_only = false;      // 1D_SPH S_N origin parity investigation flag; no-op when legacy parity sweep is used
        bool group_repack_hard_xray = false;  // optional 80-group hard-X-ray boundary redistribution
        bool diagnose_hard_xray_opacity = false; // startup-only CD kappa_PA audit log
        int groups = 16;
        std::vector<double> group_bounds_eV;   // [eV] 要素数 groups+1; table, user, or hard-X-ray repacked bounds
        // --- Radiation.imc（SPECIFICATION §6.4.5）: 決定論モードに効くのは two_stage だけ ---
        struct ImcConfig {
            bool two_stage = false;           // 輻射演算子を半ステップ 2 回で進め、その間で EOS を閉じ直す
        } imc;
        // 退役した輻射 Monte Carlo（IMC・DDMC・ランダムウォーク・HOLO・difference 定式化）の設定 — imc の他のキーと
        // ddmc / diffusion / holo の各 dict — は 2026-09-29 に Config から外した。Builder は受理して無視し（節ごとに
        // WARNING 1 回）、各手法の enabled=True は ConfigError とする（SPECIFICATION §6.4.5）。
        struct MultigroupDiffusionConfig {
            std::string flux_limiter = "levermore_pomraning";
            int max_outer_iterations = 20;
            double outer_tol = 1e-5;
            std::string linear_solver_1d = "cusparse_tridiag";
            std::string linear_solver_2d = "amgx_cg"; // "amgx_cg" | "cusparse_cg_jacobi" | "cusparse_cg_zline"
            struct AmgxConfig {
                std::string preset = "AGGREGATION_JACOBI";
            } amgx_config;
            double opacity_floor = 1e-100;
            double opacity_cap = 1e20;
            std::string state_supply_boundary_policy = "local_D_current"; // "local_D_current" | diagnostic-only "harmonic_ghost_D_test" | "radial_mean_D_test"
            bool diagnostic_radial_fourier_substage_enabled = false; // FLD substage audit; default-off
            double cg_inner_tol = 1e-10; // 2D_RZ FLD CG inner tolerance
            struct BoundaryConfig {
                std::string inner_r = "reflect";
                std::string outer_r = "vacuum"; // "vacuum" | "reflect"
                std::string z = "vacuum"; // 2D_RZ common: "vacuum" | "reflect" | "marshak"
                std::string z_bottom = "vacuum";
                std::string z_top = "vacuum";
            } boundary;
            struct MarshakConfig {
                double flux_erg_per_cm2_s = 0.0;
                double flux_pulse_duration_s = -1.0;
            } marshak;
            std::string z_boundary = "vacuum"; // alias for boundary.z
        } multigroup_diffusion;
        struct SnTransportConfig {
            int n_angles = 16;
            std::string angular_quadrature = "level_symmetric_16";
            int max_outer_iterations = 20;
            int max_inner_iterations = 100;
            double outer_tol = 1e-5;
            double inner_tol = 1e-6;
            bool dsa_enabled = true;
            std::string diffusion_fallback_mode = "none";
            double tau_diffusion_on = 10.0;
            double tau_diffusion_off = 5.0;
            double opacity_floor = 1e-100;
            double opacity_cap = 1e20;
            bool timing_enabled = false;
            struct BoundaryConfig {
                std::string inner_r = "reflect_parity";
                std::string outer_r = "vacuum";
                std::string z = "vacuum"; // 2D_RZ common: "vacuum" | "reflect" | "marshak"
                std::string z_bottom = "vacuum";
                std::string z_top = "vacuum";
            } boundary;
            struct MarshakConfig {
                double flux_erg_per_cm2_s = 0.0;
            } marshak;
            std::string z_boundary = "vacuum"; // alias for boundary.z
        } sn_transport;
        // --- Planck fraction table (SPECIFICATION §6.4.5 planck_fraction) ---
        struct PlanckFractionConfig {
            std::string method = "compute";     // "compute" | "tabulate"（SPECIFICATION §6.4.5 既定 "compute"）
            int compute_N_T = 200;              // Planckテーブル温度格子点数（SPECIFICATION §6.4.5 既定 200）
            double compute_T_range_eV[2] = {0.01, 100.0}; // [eV] 温度範囲（SPECIFICATION §6.4.5 既定 [0.01,100]）
        } planck_fraction;
        struct BoundaryConfig {
            // namelist名: r_inner, r_outer, z_bottom, z_top（SPECIFICATION §6.4.5）
            std::string inner_r = "reflect";   // 1D: inner; 2D RZ: R内側（r=0対称軸、変更不可）
            std::string outer_r = "vacuum";    // 1D: outer; 2D RZ: R外側
            std::string bottom_z = "vacuum";   // 2D RZ only: Z下面（SPECIFICATION §6.4.5 既定 "vacuum"）
            std::string top_z = "vacuum";      // 2D RZ only: Z上面
            // Marshak 放射温度 T_r(t) [eV]：1D_SPH は単一 FrozenTable1D、2D_RZ は面別 map
            // 1D_SPH: marshak_Tr = FrozenTable1D（callable → 時刻表に凍結、SPECIFICATION §6.4.5）
            // 2D_RZ:  marshak_Tr_map["r_outer"] / ["z_bottom"] / ["z_top"] = FrozenTable1D（SPECIFICATION §6.4.5）
            FrozenTable1D marshak_Tr;                          // 1D_SPH 用（2D_RZ では未使用）
            std::map<std::string, FrozenTable1D> marshak_Tr_map; // 2D_RZ 用（面名 → 温度テーブル）
        } boundary;
    } radiation;

    struct LaserConfig {                     // 実体は src/core/config.hpp（全キーは SPECIFICATION §6.4.6）。主なものだけを示す
        bool   enabled = false;            // レーザー有効化（既定 False）
        double wavelength_nm = 351.0;      // laser wavelength [nm] (GXII: 3ω Nd:glass)
        // n_crit = π m_e c² / (e² λ²) [cm⁻³]（NUMERICS §5.1 参照）
        std::string mode;                   // "raytrace_2d" | "radial_absorption_1d"（1D）/ "raytrace_3d"（2D_RZ）。Builder が次元で既定を置く
        // radial_absorption_1d では rays/profile/f_number/focus/defocus は吸収分布に影響しない。
        int    rays_per_beam = 1000;       // ビームあたりレイ（環）数（\>= 10。2D_RZ 未指定時は Builder で 128）
        struct AbsorptionConfig {
            std::string model = "inverse_bremsstrahlung";
            double eps_n = 1e-4;           // 屈折率下限（(0, 0.1]、NUMERICS §5.3）
            bool   terminate = true;       // 臨界で終了（True）/ 反射（False、特性曲線積分のみ）。未指定なら特性曲線積分では
                                           // Builder が False にする（NUMERICS §5.2）
            std::string terminate_mode = "escape"; // "escape" | "deposit"（臨界に達した残りを臨界隣接セルへ沈着）
            double coulomb_log_floor = 2.0;// IB の lnΛ 下限（[1, 30]、NUMERICS §5.4.3）
        } absorption;
        struct LaserMeshConfig {
            int    nr = 128, nz = 256;     // 2D の格子（\>= 4）。1D は §5.7.2 の規則で毎回作り直す（上限 nr_max = 4096）
            double r_max_factor = 1.5;     // 1D の外半径の係数（NUMERICS §5.7.2）
            double mesh_factor = 0.5, rmax_n_hat_threshold = 0.001;
            bool   critical_clip = true;   // 節点の n̂ を critical_margin で頭打ち（格子の境界ではない、NUMERICS §5.7.1）
            double critical_margin = NaN;  // 未指定 = 1 − eps_crit
            std::string stretch_method;    // 受け付けるがどのメッシュも読まない（警告）。min_ratio も同じ
            GhostCoronaConfig ghost_corona;// 1D のゴーストコロナ（NUMERICS §5.7.5）
        } lasermesh;
        struct RaytraceConfig {
            std::string integrator = "auto"; // "auto" → 1D 球の raytrace_2d は "characteristic"（NUMERICS §5.3.6）、それ以外は "leapfrog"
            double cfl_ray = 0.8, intensity_cutoff = 1e-6, eps_crit = 1e-4; // eps_crit は (0, 0.1]
            int    max_steps = 100000, azimuthal_rays = 16;               // azimuthal_rays は 1D の円筒・平板
        } raytrace;
        struct RaytraceSkipConfig { bool enabled = false; double threshold = 0.01; int max_consecutive = 10; } raytrace_skip;
        struct DepositConfig {
            double conservation_tol = 1e-10;
            int    deposit_smooth_passes = 0;      // 既定は平滑化なし（診断用。1D の例題デッキの一部は 3 を与える）
            double deposit_smooth_alpha = 0.25;
        } deposit;
        std::string profile_model = "gaussian";    // "gaussian" | "super_gaussian" | "flat_top" | "table" | "custom"（custom は凍結後 table）
        double profile_w0_um = -1.0;               // 未指定なら R_target / (2 max(F, 1))
        int    profile_m = 2;
        // ib（Langdon・Zeff・Coulomb log の拡張）、ra（共鳴吸収）、cbet、port_configuration、hot_electron は §4.6 と NUMERICS §5.4.5/§5.10/§5.11
        struct BeamDef {
            std::vector<double> direction;  // ビーム方向（3 成分）。theta/phi（ラジアン）でも与えられる
            double f_number = 8.0;          // F 値（任意、既定 8.0、\> 0）
            std::vector<double> focus;      // 焦点の lab 座標（3 成分）。defocus_DR との変換は行わず、1D はビーム軸へ射影（NUMERICS §5.6.3）
            double defocus_DR = 0.0;        // focus 未指定時の D/R（NUMERICS §5.6.5）
            double delta_lambda_nm = 0.0;   // CBET の離調
            std::string profile_model;      // "" = Laser.profile_model を継承。profile_w0_um / profile_m / 表も同様
            CallableInfo power;             // パワー波形（凍結して表）
            double energy_J = -1.0;         // \> 0 なら波形を [0, t_end] の積分がこの値になるよう拡大・縮小
        };
        std::vector<BeamDef> beams;        // ビームリスト（enabled=True時は≥1本が必須）
    } laser;

    struct NumericsConfig {
        // splitting_order / splitting / coulomb_log_floor / cell_search は namelist で受け付けるが警告を出して無視する:
        // 演算子の順序は固定（L→B→H/2→C→R→H/2、NUMERICS §2.1）、クーロン対数の下限は 2 に固定（NUMERICS §1.1.4）
        double T_start_eV = 0.0;        // Hydro開始温度 [eV]（既定 0.0）
        // hydro.T_start_inactive_cells = "passive_fill" | "rigid_wall" | "cold_equilibrium"（既定 "passive_fill"）
        struct DtConfig {
            double initial_s = 1e-15;       // [s] 初期Δt（SPECIFICATION §6.4.7 既定 1e-15）。
                                            // Python API で None 指定時は Builder が -1.0 に変換し、
                                            // auto (0.1 * min(Δl/c_s))（NUMERICS §2.2）を Step 1 で計算
            double cfl_hydro = 0.3;         // [dimensionless] CFL係数（SPECIFICATION §6.4.7 既定 0.3）
            double cfl_cond = 0.25;         // [dimensionless] 伝導CFL数（SPECIFICATION §6.4.7 既定 0.25、NUMERICS §2.2(b)）。STS Δt_exp の係数として使用
            double f_min_fleck = 0.01;      // [dimensionless] Fleck factor下限によるΔt_rad制約（SPECIFICATION §6.4.7 既定 0.01、NUMERICS §2.2(c)）
            // cfl_ray は LaserConfig::RaytraceConfig に配置（SPECIFICATION §6.4.6）。DtConfig では管理しない
            double growth_factor = 1.2;     // [dimensionless] dt成長倍率制限
            double max_s = 1e-9;            // [s] dt上限（SPECIFICATION §6.4.7 dt.max_s 既定 1e-9）
            double min_s = 1e-20;           // [s] dt下限（SPECIFICATION §6.4.7 dt.min_s 既定 1e-20）。Δt < min_s で FATAL 停止（ストーリング防止、NUMERICS §2.2(e)）
        } dt;
        struct HydroConfig {
            // 1D: string; 2D RZ: per-face struct
            bool rho_e_linear_grid = false;  // rho_e_table diagnostic: false -> (log rho, log e), true -> (rho, e)
            bool eos_writeback = false;      // table EOS closure: false -> keep hydro-updated e, true -> legacy e(rho,T) re-projection
            std::string exact_override = "none"; // "none" | "pressure" | "sound_speed" | "temperature"（1D table-backend diagnostic）
            std::string boundary_1d = "free"; // "free" | "fixed" | "reflect" | "pressure"（SPECIFICATION §6.4.7）
            struct Boundary2D {
                std::string r_inner = "axis";        // "axis"（既定、変更不可：R=0対称軸、v_r=0強制）。SPECIFICATION §6.4.7
                std::string r_outer = "free";
                struct ZFaceConfig {
                    std::string type = "free";       // "free" | "fixed" | "reflect" | "pressure" | "state_supply"
                    double supply_rho_g_per_cc = 0.0;
                    double supply_u_z_cm_per_s = 0.0;
                    double supply_T_eV = 0.0;
                } z_bottom_cfg, z_top_cfg;
                std::string z_bottom = "free";       // legacy mirror of z_bottom_cfg.type
                std::string z_top = "free";          // legacy mirror of z_top_cfg.type
                std::string mesh_tangential_target = "lagrangian"; // "lagrangian" | "reference"（SPECIFICATION §6.4.7）
                std::string state_supply_donor_mode = "interior_per_i"; // "interior_per_i" | "interior_radial_average"
                hydro::BC2DRZConfig bc_config;       // explicit normal/tangential material/mesh semantics
                FrozenTable1D pressure_drive;       // [dyne/cm²] 時間依存駆動圧力 P_drive(t)（"pressure" type 時に使用。NUMERICS §8.1、SPECIFICATION §6.4.7）
                // namelist の boundary_pressure callable から初期化時に FrozenTable1D 化（時刻表、線形補間）
            } boundary_2d;
            std::string av_type = "vnr";    // "vnr" | "riemann"（"riemann" は 1D_SPH 限定。SPECIFICATION §6.4.7）
            // namelist名: av_C1, av_C2, av_limiter_J, av_heat_C（SPECIFICATION §6.4.7）
            double av_linear = 0.1;          // [dimensionless] 線形人工粘性係数 C₁（= av_C1、NUMERICS §3.1.6；既定0.1）
            double av_quadratic = 1.5;       // [dimensionless] 二次人工粘性係数 C₂（= av_C2、NUMERICS §3.1.6；既定1.5、式中はC₂²で使用）
            double av_limiter_J = 1.0;       // [dimensionless] 1D Christensen速度リミタ係数 J（NUMERICS §3.1.6, §3.1.9；既定1.0、2Dでは未使用）
            double av_heat_C = 0.0;          // [dimensionless] 1D人工熱流束係数 C_H（NUMERICS §3.1.6；既定0.0、2Dでは未使用）
            double hk_velocity_damper_C = 0.0;              // [dimensionless] 1D high-k nodal velocity damper strength（NUMERICS §3.1.4；0で無効）
            double hk_velocity_damper_tau_min = 8.0;        // [dimensionless] damper optical-depth gate τ_min
            double hk_velocity_damper_grad_Te_max = 0.2;    // [dimensionless] front mask / secondary max adjacent |Δln Te|
            double hk_velocity_damper_grad_rho_max = 0.3;   // [dimensionless] front mask / secondary max adjacent |Δln ρ|
            int hk_velocity_damper_guard_cells = 25;        // [cells] front mask expansion half-width
            std::string av_heat_to = "ion";  // "ion" | "electron" | "split"（v1.0: "split"→ConfigError。SPECIFICATION §6.4.7）
            bool compatible_energy = false;  // [flag] 1D_SPH ideal-gas exact compatible force-work energy update（NUMERICS §3.1.5；既定false）
            // 1D shock sensor の閾値（pressure jump / density jump / RH整合性 / odd-even / support floor）
            // は artificial_viscosity.cu 内の内部定数で保持し、v1.0 では namelist 非公開
            FrozenTable1D pressure_drive_1d; // [dyne/cm²] 1D_SPH pressure BC用 P_drive(t)（SPECIFICATION §6.4.7。boundary_1d="pressure"時使用）
        } hydro;
        struct ConductionConfig {
            bool enabled = true;             // 電子熱伝導の有効/無効
            enum class Solver : uint8_t { STS = 0, IMPLICIT = 1, HYPRE = 2 };
            Solver solver = Solver::STS;     // "sts"（既定）| "implicit"（1D三重対角陰解法）| "hypre"（NUMERICS §4.2.1/§4.2.3）
            bool ion_conduction = false;     // Braginskii イオン熱伝導、1D・2T（SPECIFICATION §6.4.7 既定 False、NUMERICS §4.6）
            double ion_f_lim = 1.0;          // [dimensionless] イオン熱流束の制限係数（イオン自由流束の倍数、NUMERICS §4.6）
            double f_lim = 0.06;             // [dimensionless] flux limiter（NUMERICS §4.1）
            double mfp_limiter_C = 0.0;      // [dimensionless] mean-free-path limiter係数（NUMERICS §4.1）
            // STS パラメータ（solver=STS 時のみ使用）
            double sts_damping = 0.01;       // [dimensionless] STSダンピングパラメータ ν（NUMERICS §4.2.1）
            int sts_max_stages = 40;         // STS最大ステージ数 s_max（NUMERICS §4.2.1）
            std::string halo_strategy = "every"; // "every"（既定）| "adaptive"（NUMERICS §12.2.3）
            // Hypre パラメータ（solver=HYPRE 時のみ使用、§4.2.3）
            double hypre_rtol = 1e-8;        // [dimensionless] PCG相対収束判定
            int hypre_max_iter = 50;         // PCG最大反復数
            int hypre_amg_coarsen = 10;      // BoomerAMG粗視化タイプ（HMIS=10）
            int hypre_amg_relax = 18;        // BoomerAMG緩和タイプ（l1-Jacobi=18、GPU向き）
            int hypre_amg_interp = 6;        // BoomerAMG補間タイプ（ext+i=6）
            int hypre_amg_levels = 25;       // BoomerAMG最大レベル数
            // SNB 非局所電子熱輸送（1D opt-in、NUMERICS §4.4、SPECIFICATION §6.4.7）
            std::string nonlocal_model = "none";       // "none"（既定、bit 恒等）| "snb"
            int snb_n_groups = 24;                     // SNB エネルギー群数
            double snb_E_max_over_Te = 20.0;           // 群構造上限 E_max/(k_B max T_e)
            std::string snb_mfp = "geometric_r2";      // "geometric_r2" | "original"
            std::string snb_efield = "none";           // "none" | "local"
            int snb_picard_max_iters = 8;              // iSNB Picard 反復上限
            double snb_picard_rtol = 0.01;             // Picard 収束判定（max-norm）
        } conduction;
        struct FloorsConfig {
            // namelist path: Mesh.floors.rho_floor_gcc / Te_floor_eV / Ti_floor_eV（SPECIFICATION §6.4.2）
            // Builder が Mesh.floors → NumericsConfig.floors にマッピングする
            double rho = 1e-10;             // 密度下限 [g/cm³]（NUMERICS §1.1.7）
            double Te = 1e-3;               // 電子温度下限 [eV]（NUMERICS §11.2）
            double Ti = 1e-3;               // イオン温度下限 [eV]（NUMERICS §11.2）
        } floors;
        bool radiation_thermal_subcycle = false; // [flag] Radiation callback thermal microcycling（SPECIFICATION §6.4.7、NUMERICS §2.1）
        bool positivity_clamp = true;        // [flag] 温度・密度フロアへのクランプ有効化（SPECIFICATION §6.4.7 positivity.clamp 既定 True）
        // False の場合は負温度・負密度が発生しうる（デバッグ用のみ推奨）
        // 注：旧 PositivityConfig は FloorsConfig に統合済み。
        // 下位互換のため namelist で positivity.rho_floor_gcc 等が指定された場合は
        // Builder が floors.rho/Te/Ti にマッピングし WARNING を出力する。
        struct SafetyConfig {
            bool energy_fatal = false;        // [flag] エネルギー保存違反時に fatal 停止するか（SPECIFICATION §6.4.7 既定 False）
            bool nan_fatal = true;            // [flag] NaN/Inf検出時に fatal 停止するか（既定 True。energy_fatal とは独立制御）
            double energy_budget_tol = 1e-3;    // [dimensionless] 相対許容誤差 |ΔE/E|
                                                // Python API名: safety.energy_threshold（SPECIFICATION §6.4.7 既定 1e-3）
                                                // Builder が energy_threshold → energy_budget_tol にマッピング
            // opacity_floor / opacity_cap（退役したモンテカルロ輻射の不透明度の下限・上限）は 2026-09-29 に外した。
            // Builder は受理して無視する。FLD・S_N の下限・上限は Radiation.multigroup_diffusion / sn_transport にある。
            int clamp_warn_threshold = 100;
            int clamp_fatal_threshold = 10000;
            double overshoot_warn = 0.01;       // [dimensionless] 最大原理違反率 WARNING 閾値（SPECIFICATION §6.4.7、NUMERICS §11.8）
            double overshoot_fatal = 0.10;      // [dimensionless] 最大原理違反率 FATAL 閾値（overshoot_fatal_enabled=true 時のみ有効）
            bool overshoot_fatal_enabled = false; // [flag] overshoot_fatal 超過時に FATAL 停止するか（SPECIFICATION §6.4.7 既定 False）
            // cell_search_fatal は CellSearchConfig::fatal に移動済み（SPECIFICATION §6.4.7 cell_search.fatal）。
            // Builder が safety.cell_search_fatal を cell_search.fatal にマッピングし WARNING を出力する（後方互換）。
        } safety;
        struct CellSearchConfig {
            int hash_table_factor = 4;      // ハッシュテーブルサイズ係数（NUMERICS §9.5）
            int max_walk = 20;              // 最大歩行数（NUMERICS §9.3）
            int max_rings = 3;              // リング拡張探索の最大半径（NUMERICS §9.4、SPECIFICATION §6.4.7 既定 3）
            bool fatal = true;              // 全探索失敗時にfatal停止するか（SPECIFICATION §6.4.7 cell_search.fatal、NUMERICS §9.5）
        } cell_search;
        int diagnostics_every = 1;           // [cycles] 内部安全チェック頻度（SPECIFICATION §6.4.7 既定 1）
        // Diagnostics.every（出力頻度）とは独立に設定可能。
        // diagnostics_every は内部安全チェック（フロアカウンタ等）の頻度を制御する。
        // STSConfig は不要（cfl_cond は DtConfig に統合、STS固有パラメータは ConductionConfig に配置）
    } numerics;

    struct OutputConfig {
        std::string directory = "./output";   // 出力先（存在しなければ作成）
        std::string format = "hdf5";          // 出力形式（SPECIFICATION §6.4.8 既定 "hdf5"、v1.0唯一）
        int plot_every = 100;
        int history_every = 1;
        int checkpoint_every = 1000;           // SPECIFICATION §6.4.8 既定 1000；有効範囲 ≥ 1
        double plot_every_s = -1.0;            // [s] 時間間隔ベース出力（-1.0=無効、>0.0 で有効、0.0 は ConfigError）
        double history_every_s = -1.0;         // [s] SPECIFICATION §6.4.8、NUMERICS §2.2 (f)
        double checkpoint_every_s = -1.0;      // [s] SPECIFICATION §6.4.8
        int checkpoint_keep_last = 2;          // 保持するチェックポイント数（SPECIFICATION §6.4.8 既定 2）
        std::string compression = "gzip";      // HDF5圧縮方式 "none" / "gzip"（SPECIFICATION §6.4.8 既定 "gzip"）
        int compression_level = 4;             // gzip圧縮レベル [0-9]（SPECIFICATION §6.4.8 既定 4）
        bool save_namelist_copy = true;        // namelist源ファイルのコピー保存（SPECIFICATION §6.4.8 既定 True）
        bool save_frozen_config = true;        // 凍結設定（JSON）の保存（SPECIFICATION §6.4.8 既定 True）
        std::vector<std::string> plot_fields = {
            "rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "Qvisc",
            "mass", "vol", "zbar", "energy_density"
        };
        // HDF5 group structure: /hydro/rho, /mesh/x_r, /radiation/energy_density, etc.
        // データセット名は State フィールド名と 1:1 対応（SPECIFICATION §7.2 準拠）
    } output;

    struct DiagnosticsConfig {
        bool enabled = true;                  // 全診断無効化スイッチ（SPECIFICATION §6.4.9 既定 True）
        int every = 1;
        struct EnergyBudget {
            bool enabled = true;
            std::vector<std::string> components = {
                "kinetic", "internal_electron", "internal_ion", "radiation_field",
                "laser_incident", "laser_deposited", "laser_escaped",
                "marshak_in", "radiation_escaped",
                "pdv_boundary", "numerical_loss",
                "floor_injected", "safety_injected", "solver_residual"
            };  // NUMERICS §10.2 恒等式の全成分。SPECIFICATION §6.4.9 + §7.2 HDF5スキーマ準拠
            double warn_threshold = 1e-3;  // 保存誤差の警告閾値（SPECIFICATION §6.4.9 既定 1e-3）
        } energy_budget;
        struct ArealDensity {
            bool enabled = true;              // ρR面密度診断の有効化（SPECIFICATION §6.4.9 既定 True）
            std::string r_range = "shell";    // "full" | "shell"（SPECIFICATION §6.4.9 既定 "shell"）
            std::vector<double> angles_deg = {0, 45, 90};  // 対称軸からの角度 [deg]（SPECIFICATION §6.4.9 既定 [0,45,90]）
        } areal_density;
        struct Sphericity {
            bool enabled = true;             // 球面性診断の有効化（SPECIFICATION §6.4.9 既定 True）
            std::string surface = "isodensity"; // "isodensity" | "material_interface"（SPECIFICATION §6.4.9 既定 "isodensity"）
            double rho_threshold = 10.0;     // [g/cm³]（SPECIFICATION §6.4.9 既定 10.0。-1.0 = auto: 0.1×ρ_max を使用）
            std::vector<int> modes = {0, 2, 4};  // Legendre モード次数（SPECIFICATION §6.4.9 既定 [0,2,4]）
        } sphericity;
        struct ShellTracking {
            double rho_threshold_factor = 0.1; // cells with ρ > factor * ρ_max
        } shell;
        struct LaserPatternDiag {
            bool enabled = true;               // laser.enabled=True時のみ有効（SPECIFICATION §6.4.9）
            bool absorbed_power_profile = true; // 臨界面近傍吸収パワー密度分布
            bool critical_surface = true;       // 臨界面位置 R_crit(θ)
            bool per_beam = false;              // ビーム別吸収分率
        } laser_pattern;
        // mc_stats / fleck_diag（退役したモンテカルロ輻射の粒子統計と Fleck 係数のログ）は Config に無い。Builder は受理して無視する。
        bool per_operator_radial_fourier_enabled = false; // 2D_RZ per-Strang-stage radial Fourier audit（SPECIFICATION §6.4.9; default-off）
        double radial_fourier_window_t_start_s = 1.35e-5; // [s] audit start time, inclusive
        double radial_fourier_window_t_end_s = 1.70e-5;   // [s] audit end time, exclusive
        int radial_fourier_max_mode = -1;                 // -1 = all radial modes through Nyquist; otherwise max mode index audited
        bool per_operator_radial_fourier_complex_enabled = false; // PR G2-A fixed-mode complex coefficient audit; emits radial_fourier_audit_v2/v1
        vector<int> per_operator_radial_fourier_complex_m_targets = {14,15,16};
        vector<int> per_operator_radial_fourier_complex_j_targets = {507,508,509,510,511};
        vector<string> per_operator_radial_fourier_complex_fields; // hidden-variable field list; unavailable fields skipped
        bool overshoot_monitor = true;     // 放射演算子後の温度最大原理違反を監視（SPECIFICATION §6.4.9 既定 True、NUMERICS §11.8）。
                                            // false 時は Phase 4 の CUB Max + overshoot 検出をスキップ
    } diagnostics;

    struct ParallelConfig {
        // --- decomposition（SPECIFICATION §6.4.10）---
        struct Decomposition {
            std::string method = "slab";     // "slab" | "cartesian"
            // 注: SPECIFICATION §9.1 の次元依存既定: 1D_SPH="slab", 2D_RZ="cartesian"
            // init時に Config::apply_dimension_defaults(dim) で上書きされる
            std::vector<int> dims = {};      // [P_r, P_z]; empty = auto
            int min_cells_per_rank = 8;      // Kershaw stencil + ghost 安全余裕
        } decomposition;

        // --- halo ---
        struct Halo {
            std::string gpu_aware_mpi = "auto"; // "auto" | "force" | "disable"
            int ghost_layers = 1;            // ghost cell layers（Kershaw 9点ステンシルに必要な最小値、Appendix A参照）
        } halo;

        // migration・laser_parallel・particle_balance・reproducibility・gpu_optimization は Config に無い
        // （Builder は受理して無視する。migration は退役したモンテカルロ輻射の光子粒子の rank 間移動で、2026-09-29 に外した）。
    } parallel;
};
```

> **設計方針**：
> - `Config` はhost側のPOD的構造体。GPU側へは必要なフィールドのみ個別にコピーする
> - `Config` の生成後にPython関連リソースは全て解放される（`Py_Finalize`）
> - `Config` は frozen JSON として HDF5 に保存され、再現性に使用される
> - 各モジュールの `init()` 関数は `const Config&` を受け取り、自身の内部状態を構築する

---
