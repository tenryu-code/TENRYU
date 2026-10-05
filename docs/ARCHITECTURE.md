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

### 4.2 mesh/
**責務**：計算格子（ALE/Lagrangian）と幾何演算、近傍構造

- `Mesh::Topology`：論理構造格子のインデックスとフラグ管理
- `Mesh::Geometry`：体積、面積、法線、中心、RZ回転体積（2πr）など
- `Mesh::Search`：退役（光子粒子のセル同定。§4.2.3）
- `Mesh::Remap`：rezoning後の保存的remap（v1.0で必須、NUMERICS §3.3.4）

`Mesh::VoronoiRZ` emits a global `node_r`/`node_z` table and per-cell CCW `node_indices`, while retaining compatibility coordinate cycles populated from that table. Interior generator triples, boundary generator-pair/segment tuples, and input domain-corner vertex indices are canonical keys whose coordinates are computed once (domain corners are copied verbatim), so shared topology is bitwise deterministic and reciprocity and valence collapse operate on node ids. The v1 cocircular policy accepts roundoff-scale sliver edges from distinct generator triples, merges only bitwise zero-length edges before collapse, and retains unreferenced nodes without compaction.

- `Mesh::PolygonMesh`（`src/mesh/polygon_mesh.{hpp,cpp}`、ALE P0A F1）：トポロジ可変
  多角形 mesh の canonical host 状態 — 次数上限付き CSR（`kMaxPolygonDegree`=8）、
  stable cell/node ID + 単調 allocator、AMR-ready lineage（parent id／lineage epoch）、
  reference epoch と別立ての topology epoch、FNV-1a 表現契約 hash。検証は
  result-struct イディオム（`PolygonMeshValidation`、`MeshGeometryResult` と同型）+
  構築時 fail-loud。structured quad import が現行 mesh を特殊例として再現。P0A では
  mutation API・device mirror・幾何モーメントを持たない（幾何は F4 が権威）。
- `Mesh::PentagonBeltShell`（`src/mesh/pentagon_belt_shell.cu`）：五角形
  belt 遷移リング付き polar-shell の生産 builder。ring-major 節点（θ は最細 ladder の
  サブサンプリング＝2:1 入れ子 bitwise・南半球は鏡映）、block-major セル（2K+1 block、
  role PENTAGON_BELT、orientation −1）、stride-8 CSR＋nverts-aware bilateral 検証、
  解析殻体積ゲート、内縁 node flag（NODE_INNER_PHYSICAL_BOUNDARY）。幾何は multiblock
  CSR kernel（容量 8）＋汎用 ring Svec tangent balance（外環・内環）。checkpoint は
  topology v3＋v4（corner_stride・cell_nverts）。離散化契約は NUMERICS §3.2.y。
- `Mesh::Moments`（`src/mesh/rz_moments.cuh`、ALE P0A F4）：厳密 RZ 重み付き幾何
  モーメント庫（単一幾何権威）— 三角形／多角形の閉形式 signed r-モーメント
  （area, ∫r, ∫z, ∫r², ∫rz、λ 重み一次）、fan／star 三角形分割、線形再構成転送
  モーメント、回転面測度、決定論的 orient 述語、overlay partition gate。全演算は
  contraction-immune 2 層規約（関数内明示 fma + 公開関数の最終積 fma(a,b,0.0)）で
  host≡device bitwise（sm_89 実測、memcmp ctest 常設）。既存
  `hydro::ale::detail::rz_signed_quad_volume` の置換統合は後続 wave。
- `Mesh::ReferenceFlatPath` (`src/mesh/reference_flat_path.{hpp,cpp}`):
  construction-coordinate corner typing and host exact predicates shared by
  the Hydro G1 predictor/corrector path guard and max-min repair. It owns the
  adaptive exact reference orientation certificate, exact-dyadic quadratic
  interval/equality-order tests, and continuous nonincident-edge collision
  root isolation for cells containing structural flat corners. CUDA receives
  only the resulting immutable flat-corner byte mask; regular-corner strict
  evaluation remains in `Hydro::CornerJacobianQuality`.

#### 4.2.1 Mesh::Topology — 論理構造格子

v1.0の既定メッシュは **論理構造格子**（structured quad mesh）である。
ALE rezone 後も論理トポロジ（i,j）は変わらず、物理座標のみ変化する。
S1の `topology_scheme="multiblock_cart_core_polar_shell"` は default-off の
mesh topology extension で、single-block では CSR 等の非構造格子データ構造を
確保しない。

```cpp
struct MeshTopology {
    int nr, nz;                 // single_block dimensions; multiblock stores shell Nr and 4*Nc
    int n_cells;                // single_block: nr * nz; multiblock: block-total cell count
    int n_nodes;                // single_block structured count; multiblock: shared-node total

    // linear indexing: cell(i,j) = i * nz + j （row-major, i=r方向, j=z方向）
    // node(i,j) = i * (nz+1) + j

    // 境界フラグ（device配列、初期化後は read-only）
    uint8_t* node_flags;        // OWNED [n_nodes] bit flags: BOUNDARY, AXIS, CENTER, POLE_AXIS

    optional<MultiBlockTopology> multiblock; // nullopt for single_block
};

struct MultiBlockTopology {
    vector<int> cell_block_id;          // [n_cells_total], dense block id
    vector<int> cell_id_stable;         // [n_cells_total], block-major stable ids
    vector<uint8_t> cell_nverts;        // [n_cells_total], active corners (3 for cap triangles, 4 otherwise)
    vector<int> cell_orientation_sign;  // [n_cells_total], +1/-1 canonical geometry sign
    vector<int> cell_node_csr_offsets;  // [n_cells_total+1]
    vector<int> cell_node_csr_indices;  // [total_corners]
    vector<int> face_adj_csr_offsets;   // [n_cells_total+1]
    vector<int> face_adj_csr_indices;   // [total_faces]
    vector<int> face_bc_tags;           // [total_faces], BoundaryKind as int
    int block_count;                    // 3 for gamma MVP, 5 for half-butterfly
    vector<int> block_role;             // semantic role enum per block
    vector<int> block_cell_counts;      // [block_count]
    vector<int> block_node_counts;      // [block_count], owner-written nodes
    int n_cells_core, n_cells_bridge, n_cells_shell;
    int n_nodes_core, n_nodes_bridge_interior, n_nodes_shell;
};

struct ReverseCellNodeCSR {
    vector<int> node_offsets; // [n_nodes+1]
    vector<int> node_cells;   // incident cell ids, sorted by (cell_id_stable, corner)
    vector<int> node_corners; // incident corner slot in {0,1,2,3}
};

struct Mesh {
    DeviceArray<int> multiblock_cell_node_csr_offsets; // device mirror, multiblock only
    DeviceArray<int> multiblock_cell_node_csr_indices; // device mirror, multiblock only
    DeviceArray<int> multiblock_reverse_csr_node_offsets; // [n_nodes+1], multiblock only
    DeviceArray<int> multiblock_reverse_csr_node_cells;   // incident cells, multiblock only
    DeviceArray<int> multiblock_reverse_csr_node_corners; // incident corners, multiblock only
};

// フラグ定数
enum NodeFlag : uint8_t {
    NODE_NONE      = 0,
    NODE_BOUNDARY  = 1 << 0,    // 外側境界上
    NODE_AXIS      = 1 << 1,    // R=0 軸上（2D_RZ）
    NODE_CENTER    = 1 << 2,    // r=0 center (1D_SPH or tri_fan origin row)
    NODE_POLE_AXIS = 1 << 3,    // spherical-polar theta=0/pi pole axis
};
```

**single-block の面（Face）はデータ構造を持たない**：構造格子では面は暗黙的に定義される。
セル `(i,j)` の4面は隣接セル `(i±1,j)`, `(i,j±1)` との間の面であり、
面積・法線はジオメトリ計算時にon-the-flyで算出する。
Multiblock では `MultiBlockTopology` の CSR vectors が cell-to-node と
face-adjacency の source of truth になる。`cell_block_id` と
`cell_id_stable` は block-major order の stable cell identity を保持し、
`block_count` と per-block counts は allocation、serialization、diagnostics の
shape contract になる。

**Runtime dispatch (single_block vs multiblock):**

When `mesh.topology_scheme = "single_block"` (default), the hydro stack uses
structured `c = i*nz + j` cell indexing and the structured `(nr+1)*(nz+1)`
node grid throughout. This path is byte-identical to TENRYU's pre-S2 behavior.
Tri_fan center treatment is similarly unchanged for
`polar_center_treatment = "tri_fan"` decks.

When `mesh.topology_scheme` selects a multiblock mode, hydro
consumers branch on `mesh_topo_is_multiblock(cfg)` and dispatch CSR-aware
kernel variants. These read connectivity from
`state.mesh.multiblock_cell_node_csr_offsets/indices` (cell-to-node CSR) and
`state.mesh.multiblock_reverse_csr_node_*` (reverse CSR for node-loop
force/mass accumulation). Determinism on GPU is guaranteed via fixed
incident-corner sort order `(cell_id_stable ASC, corner_index ASC)` and no
`atomicAdd`. The numerical equations and units are documented in NUMERICS
§3.2; the corresponding namelist keys and planned S2 feature gate are in
SPECIFICATION §6.4.2.
The S5 `rounded_half_butterfly` transition scheme is a geometry generator
variant inside the same three-block multiblock architecture: it changes only
bridge interior node coordinates via rounded-superellipse-cap TFI and optional
mesh-local elliptic smoothing.  Core/shell seam nodes, block counts, CSR
connectivity, reverse CSR, face adjacency, seam tags, HDF5 layout, and remap
dispatch remain identical to the legacy `hermite_bridge` topology.
`topology_scheme="multiblock_half_butterfly_5block"` uses the same multiblock
CSR containers with five block table entries: central core, north fan, east
fan, south fan, and polar shell. Shared central-fan, fan-fan diagonal, and
fan-shell seams resolve to one owner-written node id, so all incident blocks
read one coordinate value. The shell keeps the existing polar-shell node
ordering. The finite-valence multiblock vertices at the half-plane axis and
fan diagonals replace the former smooth square-core diagonal corner; no
single logical cell corner is required to carry both edge tangents of a rounded
seam. `cell_orientation_sign` records the canonical signed-volume convention
(central core `+1`, fans and shell `-1`) so geometry kernels can use one
positive-volume predicate across the mixed orientation. B-S1 validated this
mesh/schema topology; B-S2 makes it hydro-runnable by retargeting ALE/remap,
scaled-reference orientation, polar-shell pressure BC, path-admissibility
diagnostics, radial/Fourier diagnostics, committed mesh-quality observation,
and GCL audit code to block-role metadata, CSR connectivity, and
`cell_orientation_sign` instead of three-block structured offsets. The B-S2
smoke acceptance is intentionally gentle: a closed, uniform, at-rest 2T
Lagrangian plus ALE/remap run. B-S3 still owns seam flux under gradients and
B-S4 still owns the decisive compression gate. Single-block and three-block
multiblock paths are unchanged and remain byte-identical in their default/off
configurations.
`topology_scheme="multiblock_half_butterfly_trifan_cap_5block"` replaces the
central core block with a pinned tri-fan cap. The cap owns one apex
\(O=(R,Z)=(0,0)\) flagged `NODE_CENTER|NODE_AXIS|NODE_BOUNDARY`, \(4N_c\)
first-ring triangular cells with `cell_nverts=3`, and outer cap quad rings.
The cap outer ring reuses the fan inner-seam node ids, so cap/fan exchange is
ordinary bilateral CSR over `unique_internal_faces`; there is no special
one-sided cap seam. The fan and shell blocks keep the half-butterfly ownership,
counts, and CSR containers.
The active-slot topology contract is centralized in
`mesh.hpp::mesh_topo_cell_active_nverts` and
`mesh.hpp::mesh_topo_active_local_face_corners`: storage remains four slots for
every cell, but a cap triangle uses active vertex slots `{0,1,2}` and active
local faces `{1,2,3}`. Storage slot 3 and local face 0 are inactive and must
not contribute geometry, flux, mass, velocity projection, or quality samples.
Boundary dispatch follows the topology, not `logical_mesh_2d`: multiblock
`apply_boundary_2d` always uses the node-flag boundary path and never the
structured `i*stride+j` path, even when the deck uses the polar logical mesh
setting for gamma-MVP construction.

S3 verifies this runtime dispatch at seam-conservation level with four gates:
constant-state seam GCL, uniform-pressure force balance, homothetic three-ring
symmetry, and dynamic A4 spherical smoke convergence. These gates add no new
runtime architecture or namelist keys; they constrain the existing CSR-aware
multiblock path and its boundary-projection ordering. Thresholds and empirical
γ MVP floors are documented in NUMERICS §3.2.x.

**Multiblock ALE dispatch (S4-T1-next)**: When `mesh_topo_is_multiblock` AND
`numerics.ale.enabled`, `apply_ale` (`src/hydro/ale_driver.cu`) enters the CSR
production path. Under `mesh.motion="ale"`, the production driver reaches this
path from the scheduled post-step ALE block so `apply_multiblock_csr_ale_step`
controls the per-step cadence. The default
`numerics.ale.multiblock_cross_seam_rezone_enabled=false` runs T5a per-block
CSR Winslow smoothing and T3/T4 CSR conservative remap; setting the flag true
uses the T5b cross-seam smoother before the same CSR remap. Reference-barrier
ALE uses CSR admissibility and the CSR remap path for multiblock states. When
`numerics.ale.multiblock_scaled_reference_enabled=true`, multiblock reference
targets are rebuilt through `hydro/ale_scaled_reference.cuh`: the driver/remap
path computes the current outer-ring scale \(\alpha(t)\), installs
\(\alpha X^0\) plus \(\alpha^3 V^0\) as the conservative target, and the
reference-barrier target builder uses the same scaled coordinates. The target
volume orientation comes from `cell_orientation_sign`, so the five-block
central/fan/shell winding no longer depends on `cell < n_cells_core`
inference. With the flag false, the static IC reference arrays are unchanged.
When `Numerics.ale.multiblock_differential_reference_enabled=true`, the CSR
remap path calls
`prepare_multiblock_differential_reference_if_enabled(state, cfg)` before the
scaled γ-MVP installer. A true return means
`install_multiblock_differential_reference` has installed the Lagrangian-close
reference and recomputed `State.cell_vol_initial`; a false return falls back to
`prepare_scaled_gamma_mvp_reference_if_enabled`. The differential path is
therefore an opt-in gate-replacement for multiblock 2D_RZ, while γ-MVP remains
the legacy default. Its helper functions in `src/hydro/ale_tracking_reference.cuh`
are `build_multiblock_reference_xi_initial`,
`build_multiblock_differential_band_scales`,
`install_multiblock_differential_reference`, and
`prepare_multiblock_differential_reference_if_enabled`. These build the fixed
radial ξ/director fields, compute ξ-band median/smoothed/cap-limited/monotone
corrections, run the orientation-aware CSR reference line search, write
`State.x_r_reference/x_z_reference`, and recompute reference volumes from the
accepted coordinates. `src/hydro/ale_reference_diagnostics.cuh` provides
pre-remap sampling for this experiment and does not affect remap state.
`Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled` is a
second replacement reference mode, mutually exclusive with both scaled and
differential multiblock reference modes. The CSR ALE driver calls the
center/quality patch builder, uploads `node_rezone_active` as a masked CSR
Winslow seed mask plus `node_patch_boundary` for frozen patch edges, then uses
the ALE motion trigger to run the local Phi-barrier optimizer only when the seed
patch corner-J quality crosses the center-patch on-demand threshold. The seed or
barrier output then goes through the existing CSR admissibility line search.
Bulk, patch-boundary, axis, pole-axis, center, and cap-apex nodes stay at the
post-hydro Lagrangian coordinates. The driver installs the accepted reference
and exact RZ reference volumes for remap, then restores the Lagrangian donor
coordinates before CSR conservative remap. The remap installer path skips the
differential/scaled reference installers under this flag so they cannot clobber
the installed center-patch target. The host-only patch/mask builder lives in
`src/hydro/multiblock_center_patch_reference.cuh`; the device-side local
barrier optimizer lives in `src/hydro/center_patch_barrier_optimizer.cuh`. The
builder constructs the permanent and quality-seeded cell patch, dilates it
through multiblock face adjacency, and returns host `cell_in_patch`,
`node_rezone_active`, and `node_patch_boundary` masks while persisting the
runtime-only `State.center_patch_latch` hysteresis bitmask.
With `TENRYU_I1B_DIFFREF_DIAG`, CSR remap fills
`AleRemap2DRZResult::eta_contact_step` and emits `[eta_contact_diag]`; the ALE
driver forwards it through `AleStepResult::eta_contact_step` and
`eta_contact_cumulative`. The center-patch branch also exposes the CSR
line-search `sigma_accepted` as `AleStepResult::center_patch_sigma_accepted`.
The CSR conservative remap has two hydro-energy branches. With
`Numerics.hydro.total_energy_remap_2d_rz=false`, it uses the legacy separate
electron/ion internal-energy remap and legacy cell-to-node velocity projection.
With `total_energy_remap_2d_rz=true` on
`topology_scheme="multiblock_half_butterfly_trifan_cap_5block"`, it builds and
remaps the extensive material total energy, remaps \(mY_e^{int}\), uses the
corrected swept-volume convention, applies the CSR hydro mass-positivity
face-flux limiter, projects cell velocity to nodes with
RZ corner-mass weights, applies the KE-realizability nodal velocity limiter, and
recovers \(e_e/e_i\) from remapped total energy minus the actual post-limiter
corner kinetic energy. Floor energy from the total-energy recovery is reported
through `AleRemap2DRZResult::E_floor_injected`, then
`AleStepResult::E_floor_injected`, and finally the driver energy budget; it is
distinct from the CSR mass-floor closure term `E_redistribution_unresolved`.
`Numerics.ale.swept_volume_sign_fixed` is corrected-only since epoch 2
(2026-08-05); the legacy convention has been removed.
The default/off path is kept byte-identical. Stage 4b adds a default-off CSR
Option B corner-velocity remap component in
`hydro::ale::csr_optionb_corner_velocity_remap_component`
(`src/hydro/ale_remap_2d_rz.hpp`, `src/hydro/ale_remap_2d_rz.cu`).  It is
enabled only by the process environment flag
`TENRYU_OPTIONB_CSR_CORNER_VELOCITY_REMAP=1` and writes to
`CsrOptionBCornerVelocityRemapBuffers`, not to production `State::v_r/v_z`.
The component builds deterministic colors for `unique_internal_faces`, launches
Option B FCT corner-momentum packets color by color, applies the Option B
affine-orthogonal hourglass filter, and scatters through
`multiblock_reverse_csr_node_*`.  It shares the existing CSR swept-volume and
mass-flux-scale kernels with scalar remap.
Stage 4c wires that component into `conservative_remap_csr` behind the separate
default-off process environment flag `TENRYU_I1B_OPTIONB_VELREMAP=1`.  When the
flag is unset, the driver does not allocate Option B buffers or launch Option B
kernels.  When set, the scalar path's effective swept-volume convention includes
the env flag, the driver completes scalar CSR remap first, passes the finished
`d_mass_new` to the Option B component so the corner-mass cell sums match the
scalar mass state, copies the reverse-CSR scattered velocity to `State::v_r/v_z`,
and skips both `csr_project_cell_velocity_to_nodes*` kernels. Boundary
conditions are applied after the copy. If total-energy remap is also enabled,
the subsequent KE-realizability scale and `E-K` recovery consume that Option B
nodal velocity.
The 5-block half-butterfly axis ALE path is an opt-in, target-only extension:
`mesh::build_full_axis_node_chain` derives the full shared-node physical
\(R=0\) chain and lumped incident corner mass. The production driver sanitizes
that mass for the gated path by zeroing void/inactive-cell corners, preserving
strict finite non-negative checks for active-cell corners, and compacting the
PAVA input to positive-mass axis nodes only. All-dormant zero-mass axis nodes
are not active degrees of freedom. `hydro/axis_ale_rezone.{cuh,cu}` computes
the weighted lower-bound PAVA target \(Z^*\) and first off-axis ring
diagnostics. `src/hydro/ale_driver.cu` checks
`numerics.ale.axis_rezone_enabled` after the Lagrangian update inside the
multiblock CSR ALE path, gates it to
`mesh.topology_scheme="multiblock_half_butterfly_5block"` or
`mesh.topology_scheme="multiblock_half_butterfly_trifan_cap_5block"` with five
blocks,
applies the edge/altitude trigger at most once per hydro step, installs the
axis target into the existing conservative-reference target, and immediately
uses the existing CSR conservative remap. The module does not directly write
hydro state, force, velocity, pressure, or energy arrays; state changes come
only from the remap transaction. The feature is default-off and inert for
single-block and legacy three-block topologies.

The active-vertex contract is consumed throughout the cap runtime path:
CSR remap/GCL, reverse cell-node CSR rebuilds, Hydro2D corner mass, node mass,
\(\dot V\), pressure force and compatible force-work, CSW edge AV, subzonal
pressure/mass, CFL, ALE reference barriers, axis rezone, and the full-patch
driver all read `cell_nverts` rather than assuming four live corners. Triangle
anti-hourglass is intentionally a no-op because a true triangle has no
bilinear keystone/hourglass mode, while its three subzones still participate in
compatible bookkeeping. The cap apex \(O\) is fixed by PAVA and full-patch
target construction, and velocity/position projection pins it at
\((R,Z)=(0,0)\). HDF5 `/mesh/topology/v3` dispatch carries the mixed active
vertex counts additively; default/off decks retain legacy shapes and behavior.

`numerics.ale.force_rezone_every_n_steps>0` is a driver-only diagnostic policy
that passes `force_rezone=true` into the same 2D ALE dispatch on matching steps.
When `numerics.ale.multiblock_path_admissibility_enabled=true`, Hydro2D calls
`mesh/path_admissibility.cuh` after the Lagrangian corrector commit and before
geometry refresh.  The call is controlled by the path-admissibility toggle and
multiblock topology, not by `mesh.motion`.  A path failure returns a
`ReduceDtOnly` soft failure to the
coupling driver, whose existing full-step snapshot/restore path recomputes the
entire split step at the smaller dt.
`src/hydro/pole_angular_coarsen.{cuh,cu}` provides the default-off I1-B Q2
pole pilot overlay.  Hydro2D builds it under
`TENRYU_I1B_POLE_COARSEN_PILOT=1` for the topology-only path quotient pilot,
or under `TENRYU_I1B_POLE_MOTION_PILOT=1` for the coherent mesh-position
velocity pilot; ALE rezone callers use the default null overlay.  The helper
describes a POLAR_SHELL radial band and dyadic angular quotient macros.
`mesh/path_admissibility.cuh` validates candidate macro loops at old and trial
positions, escalates simple-loop failures to the next dyadic span, builds a
separate accepted-macro fine-cell mask, and evaluates the accepted macro
boundaries on host.  The coarsen pilot merges that mask with the existing
inactive-member mask for the CUDA cell scan; the motion pilot leaves fine-cell
paths enabled and uses the accepted macros only to reconstruct \(w_r,w_z\) on
the q-band plus the configured inward smoothstep transition rows.  Material
velocity, CCH force/work, `State` masks, and remap ownership are not modified
by the motion pilot.
Structured local repair entry points in `src/hydro/local_rezone.cu` retain an
early multiblock guard.

For `logical_mesh_2d="spherical_polar_halfplane"` with
`polar_center_treatment="tri_fan"`, TENRYU overlays a derived triangle-fan
center topology without changing the structured storage. The full
`(nr+1)*(nz+1)` node array remains allocated, `node_index(i,j)=i*(nz+1)+j`
is unchanged, HDF5/restart shapes are unchanged, and no new HDF5 datasets are
introduced. The `i=0` row is initialized at the origin and marked
`NODE_CENTER`. `Mesh::cell_nverts` is derived at mesh creation: center cells
use three active slots `{0,1,2}` and all other cells use four. This derived
vector is runtime-only. Stage 1 wires it into geometry, candidate
admissibility, and ALE post-rezone quality predicates. Stage 2 also threads
optional `cell_nverts` into hydro corner mass, node-mass gather,
ALE mass/kinetic/projection accounting, output node-mass recompute, and energy
accounting. Stage 3 threads optional `cell_nverts` and `node_flags` into the
2D_RZ conservative reference remap for tri-aware centroids, swept volumes, and
post-remap `NODE_CENTER` velocity pinning. Null device pointers keep
rectangular and non-polar quad paths on the legacy code path.

Spherical-polar half-plane meshes also mark theta-pole boundary nodes with
`NODE_POLE_AXIS` for `i>0` (excluding the tri_fan `NODE_CENTER` origin row). Hydro and
ALE velocity-projection kernels upload `node_flags` when either `NODE_CENTER`
or `NODE_POLE_AXIS` is present. At `NODE_POLE_AXIS` nodes, reflect/state-supply
pole constraints zero only the cylindrical radial component; fixed pole
constraints zero both components. `NODE_CENTER` remains the final constraint
and pins both velocity components and the origin position.

Hydro receives `MeshTopology::node_flags` as an optional device pointer when
`NODE_CENTER` or `NODE_POLE_AXIS` nodes exist. Stage 2 pins center nodes in the acceleration,
velocity, and position-update kernels: acceleration is zero, predictor and
corrector velocity writes are zero, and predictor/corrector position commits
write exactly `(R,Z)=(0,0)`. The force and dVdt kernels still consume `Svec`;
tri_fan slot 3 is zero by the geometry contract, while active apex slot 0 is
nonzero and requires the pin.

Config validation rejects tri_fan with anti-hourglass, HLLC z-flux, precise
RZ geometric CFL, or `total_energy_remap_2d_rz=true` until their center-cell
designs are implemented. Stage 3 supports first-order conservative reference
remap for tri_fan; center-aware rezone policy remains Stage 4.

#### 4.2.2 Mesh 構造体

`Mesh` は1D_SPHと2D_RZを統一的に扱う計算格子データ構造であり、
`MeshTopology`（§4.2.1）とジオメトリデータ（座標・体積・面積）を統合する。

```cpp
// 計算格子（1D_SPH / 2D_RZ 共通）
// NUMERICS §3.1（1D）、§3.2（2D RZ）準拠
struct Mesh {
    MeshTopology topo;              // 論理トポロジ（§4.2.1）

    // --- ノード座標（deviceメモリ、BORROWED from State.x_r/x_z）---
    // 1D_SPH：node_r[i]（i = 0..nr）、node_z 未使用
    // 2D_RZ ：row-major node_r[i*(nz+1)+j], node_z[i*(nz+1)+j]
    double* node_r;                 // BORROWED [n_nodes] R座標 [cm]（所有権: State.x_r）
    double* node_z;                 // BORROWED [n_nodes] Z座標 [cm]（所有権: State.x_z、1Dでは nullptr）

    // --- ノード速度（deviceメモリ、BORROWED from State.v_r/v_z）---
    double* node_vr;                // BORROWED [n_nodes] R方向速度 [cm/s]（所有権: State.v_r）
    double* node_vz;                // BORROWED [n_nodes] Z方向速度 [cm/s]（所有権: State.v_z、1Dでは nullptr）

    // --- セル量（deviceメモリ、OWNED: Mesh が確保・解放）---
    double* cell_vol;               // OWNED [n_cells] セル体積 [cm³]
    double* cell_centroid_r;        // OWNED [n_cells] セル重心R座標 [cm]
    DeviceArray<double> cell_centroid_r_device; // device mirror for hydro predicates
    double* cell_centroid_z;        // OWNED [n_cells] セル重心Z座標 [cm]（1Dでは nullptr）

    // --- セル↔ノード接続（構造格子のため暗黙的）---
    // cell(i,j) の4頂点：node(i,j), node(i+1,j), node(i+1,j+1), node(i,j+1)
    // 1D_SPH：cell(i) の2端点：node(i), node(i+1)
    // spherical_polar tri_fan: cell_nverts[c]==3 for i=0 center cells and
    // slots {0,1,2} are active; slot 3 remains allocated but inactive.
    std::vector<uint8_t> cell_nverts; // derived runtime topology, not persisted

    // --- ジオメトリ次元 ---
    int dim;                        // 1 = 1D_SPH, 2 = 2D_RZ

    // --- ジオメトリ再計算 ---
    // Lagrangian移動またはALE rezone後に体積・重心・面積を再計算
    // cell_vol, cell_centroid_r, cell_centroid_r_device, cell_centroid_z を更新する
    // recompute_geometry_checked() は host-side MeshGeometryResult を返し、
    // 既存 recompute_geometry() は同じ検査を HardAssert policy で呼ぶ互換wrapperである。
    // **同期契約**：recompute_geometry() 完了直後に State.vol = Mesh.cell_vol の
    //   コピーを実行すること（Coupling::step 内で実施）。State.vol は物理カーネル
    //   （source_injection, EOS, conduction等）が参照する正規の体積フィールドである。
    //   Mesh.cell_vol はジオメトリ計算の一次ソースとして扱い、State.vol との
    //   同期は Coupling レイヤーの責務とする。
    void recompute_geometry(cudaStream_t stream);  // NUMERICS §3.2.2, §3.2.3

    // 1D_SPH hydro fast path: State.vol の device buffer へ直接体積を書き、
    // Mesh の host geometry cache は更新しない。host consumer の前には
    // sync_device_geometry_to_host() または recompute_geometry() による同期を行う。
    // 1D の幾何検査（検査窓のセルで最初に体積が下限を超えないセルと、そこまでの最小の
    // 有限体積）は device のカーネルで行い、その値は host への写しと同じ readback で
    // 戻る（2026-10-01、それまでは host の写しをセル順に走査していた）。
    void recompute_geometry_device_only(double* state_vol);
    void sync_device_geometry_to_host(const double* state_vol);

    // --- セル体積計算（device function）---
    // 1D_SPH: V = (4π/3)(r_{i+1}³ - r_i³)（NUMERICS §3.1.2）
    // 2D_RZ : V = 2π × 四辺形面積 × R_centroid（回転体積）（NUMERICS §3.2.2）
    // 戻り値：セル体積 [cm³]
    __device__ static double compute_cell_volume(
        int dim, int i, int j,
        const double* __restrict__ node_r,
        const double* __restrict__ node_z,
        int nz_plus1
    );
};
```

> **1D/2Dの統一**：`dim` フラグで分岐する。1D_SPHでは `node_z`, `node_vz`, `cell_centroid_z`
> は `nullptr` であり、Z方向の処理はスキップされる。
> 構造格子のため接続情報は `(i,j)` インデックス演算で暗黙的に解決され、
> 明示的な接続配列は不要である。

> **Mesh/State エイリアス**：`Mesh.node_r`/`node_z` と `State.x_r`/`x_z` は同一デバイスメモリのエイリアス。所有権は State。Mesh は借用ポインタ（non-owning）。

#### 4.2.3 Mesh::Search — ハッシュグリッドセル探索

退役：モンテカルロ輻射の光子粒子のセル同定（NUMERICS §9）のためのハッシュグリッド。粒子の再同定（`particle_reid.cu`）とともに
2026-09-29 に退役し、記述を `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。

#### 4.2.4 Mesh::Remap — ALE rezone後の保存的転写

```cpp
// ALE rezone後の保存的remap（NUMERICS §3.3.4）
void remap_conservative_fields(
    const Mesh& old_mesh,           // rezone前のメッシュ
    const Mesh& new_mesh,           // rezone後のメッシュ
    CellField* fields,              // remap対象フィールド配列（§5.1 CellField エイリアス）
    int n_fields,                   // フィールド数
    cudaStream_t stream
);
```

Shock-frame 2D_RZ reference remap is exposed separately as
`hydro::ale::ale_remap_2d_rz` (`src/hydro/ale_remap_2d_rz.hpp`,
`src/hydro/ale_remap_2d_rz.cu`).  It runs immediately after a Lagrangian
Hydro2D update when `Numerics.ale.conservative_remap_enabled=true`, maps from
the post-Lagrange mesh to `State.x_*_reference`, and then replaces
`State.x_r/x_z` and `State.vol` with the configured reference geometry. For
multiblock states, `Numerics.ale.multiblock_scaled_reference_enabled=true`
refreshes that geometry to the scaled γ-MVP target immediately before CSR
remap. `Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled=true`
takes precedence over both remap-path installers: the scheduled CSR ALE driver
has already installed a Lagrangian-bulk center/quality-patch target, and the
remap path preserves it. For this opt-in path the driver forces the protected
CSR swept-volume convention in the temporary remap config, activating the CSR
outgoing-mass scale without changing default legacy remap decks. Otherwise
`Numerics.ale.multiblock_differential_reference_enabled=true` takes precedence
over the scaled γ-MVP installer for multiblock 2D_RZ and refreshes the target
to the Lagrangian-close differential reference instead. Otherwise the scaled or
IC reference geometry is used. This path is
default-off and bypasses the scheduled Winslow ALE block for that step.  Its
state-supply z-face boundary flux helper uses the adjacent interior cell
velocity for the signed face speed and selects supply vs interior donor by
upwind direction before the first- or second-order remap kernel consumes the
per-column boundary flux arrays.

For `polar_center_treatment="tri_fan"`, this remap uploads optional
`cell_nverts` for tri cells and uploads `node_flags` whenever center or
pole-axis velocity constraints are present. The uploaded topology
selects three-node center-cell centroids, applies the polar signed-volume
orientation to swept volumes, falls back to first-order donor flux on faces
whose donor pair includes a tri cell, re-pins `NODE_CENTER` node velocities
to zero after projection, and applies `NODE_POLE_AXIS` radial-only pole
projection constraints. `total_energy_remap_2d_rz=true` remains guarded off
for the structured `polar_center_treatment="tri_fan"` single-block path; the
supported CSR total-energy branch is the multiblock
`multiblock_half_butterfly_trifan_cap_5block` path described in §4.2.1 above.

The experimental I1 z-HLLC path is exposed as
`hydro::apply_hllc_z_flux_2d_rz` (`src/hydro/hllc_z_flux_2d_rz.hpp`,
`src/hydro/hllc_z_flux_2d_rz.cu`).  It is selected only by
`Numerics.hydro.hllc_z_flux_2d_rz=true` and requires
`total_energy_remap_2d_rz=true`.  In that mode the Hydro2D step uses a fixed
reference mesh Eulerian z-face HLLC update for quasi-1D shocks and the driver
skips the conservative reference remap for that hydro substep, preventing double
z transport.  The path stores authoritative cell-centered z momentum in
`State.hllc_mom_z_cell`; projected nodal `v_z` is compatibility/output state and
is not used to reconstruct the next HLLC primitive state.
The I1 closure evidence and scope caveats for this experimental path are in
`docs/validation/2d_rz/I1/closure_summary.md`.

**remap対象フィールド**（保存量 = 密度 × 体積）：
- `ρ`（質量密度）→ 質量 `m = ρV`
- `ρu_r`, `ρu_z`（運動量密度）
- `ρe_e`, `ρe_i`（内部エネルギー密度）
- `volFrac[mat]`（材料体積分率）— 保存的 remap: `f_new[c] = Σ_{c'∈overlap} f_old[c'] × V_overlap(c,c') / V_new[c]`。
  Σ_mat f_new[c,mat] = 1 の正規化を remap 後に強制し、丸め誤差を修正する。
  単一材料セル（f=1.0）は remap 不要（最適化対象）

**remap後の reclosure**（CUDA_KERNELS §9 Phase 5 準拠）：
remap は保存量（ρe_e, ρe_i, ρu_r, ρu_z 等）を転写するが、原始変数（Te, Ti, Pe, Pi, c_s）および節点速度は更新しない。
remap 直後に以下の reclosure シーケンスを実行する：
1. `compute_cell_geometry`（H7）：新メッシュの体積・面積・特性長を再計算
2. `compute_density`（H8）：mass / V_new → ρ_new
3. `project_cell_velocity_to_nodes`（A4）：セル中心速度 u_r,u_z → 節点速度 v_r,v_z（NUMERICS §3.3.4 質量重み投影）
4. `eos_inverse`（H14）：ρ_new, ee → Te; ρ_new, ei → Ti
5. `eos_forward`（H13）：ρ_new, Te, Ti → Pe, Pi, Cv_e, Cv_i
6. `compute_sound_speed`（H15）：Pe, Pi, ρ_new → c_s
7. `floor_clamp`（U2）：安全策適用

**データ所有**
- Node：`x_r, x_z`（1Dは `x_r` のみ）、`v_r, v_z`、`node_flags`
- Cell：centroid（キャッシュ）、volume、surface metrics
- Face：暗黙的（on-the-fly計算）

---


### 4.3 materials/
**責務**：EOS/opacity/導電率/緩和/Zbar など物性を提供（状態→係数）

- `Materials::EOS`：P_e,P_i,e_e,e_i,Cv など（SESAME/IONMIX/ideal gas）
- `Materials::Opacity`：
  - LTE（`"ionmix"`）：入力 κ^PA_g, κ_R,g [cm²/g] → 出力 σ_a,g = ρ κ^PA_g, σ_R,g = ρ κ_R,g [1/cm]
  - Non-LTE（`"table_nlte"`）：入力 κ^PA_g, κ^PE_g, κ_R,g [cm²/g]（IONMIX 3種分離）→ 出力 σ^PA_g, σ^PE_g, σ_R,g [1/cm]
  - **Planck absorption/emission/Rosseland の使い分けはNUMERICSと一致させる**
- `Materials::Transport`：κ_e（Spitzer+flux limiter）、ν_ei、Zbar
- `Materials::Mixture`：多材料セルの混合則（EOS混合は NUMERICS §1.1.5 (c)、伝導/ソース結合のセル実効量は NUMERICS §1.1.5a、SPECIFICATION §6.4.3）
- `Materials::Tables`：SESAME/IONMIXロード、単位変換、範囲外clamp、診断
- `Materials::DeviceEOSTable`（`src/materials/eos_device_table.hpp`, `eos_device_table.cuh`, `eos_device_table.cu`）：
  table EOS data/view upload and device-side interpolation/inversion helpers.
  `eos_device_table_warp.cuh` (2026-09-25) holds warp-cooperative forms of the
  inversion (`device_inverse_reclose_warp`, `..._with_high_t_tail_warp`,
  `device_eos_T_from_e_monotone_warp`, `cold_inverse_Te_warp`): every lane of a
  warp calls them with the same arguments, the bisections over table rows and
  temperature nodes evaluate the next five levels' 31 midpoints in parallel and
  descend with the sequential loop's decisions, so the decisions are the
  sequential ones and the values agree to rounding (the compiler may contract a
  multiply-add differently; `tests/materials/test_eos_inverse_warp.cu`). The 1D 2T
  closure kernel (`enforce_2t_closure_split_kernel`, hydro_1d.cu) runs the ion and
  electron inversions of a cell on two such warps
- `Materials::ZbarDeviceContext` (`src/materials/zbar_device.hpp`, `zbar_device.cu`,
  `zbar_math.hpp`): driver-owned cache for immutable per-material ionization tables,
  material descriptors, and the host-owned void mask. The 1D updater reads resident
  rho/Te/volFrac and writes resident zbar; tabular interpolation and Thomas–Fermi
  evaluation preserve ordered material mixing. Tabular clamp warnings share the
  host reader's ordered warning budget through a bounded device summary. The
  fixed-ionization path remains a no-op, and `TENRYU_ZBAR_HOST=1` selects the
  original per-step host update for bisection. Decks with a non-TMAT material
  evaluate the Thomas–Fermi fit in the host's rounding (`zbar_tf_host_rounding.cuh`:
  glibc's pow and exp from `core::glibc_libm`, each operation rounded separately),
  which gives the host values bit for bit (they ran the host update until
  2026-10-02); TMAT-only decks keep CUDA's pow and exp.
- `Materials::EOSRhoETable`（`src/materials/eos_rho_e_table.hpp`, `eos_rho_e_table.cpp`, `eos_rho_e_device.hpp`, `eos_rho_e_device.cuh`, `eos_rho_e_device.cu`）：
  hydro-only total-EOS direct table on either \((\log\rho,\log e)\) or \((\rho,e)\). CPU
  initialization resamples raw `total` `P(\rho,T), e(\rho,T)` into `P(\rho,e), T(\rho,e)` on
  the original density grid and a 200-point energy grid (log-uniform by default; linear when
  `Numerics.hydro.rho_e_linear_grid=True`), precomputes natural-cubic second-derivative fields
  \((P_{xx},P_{yy},P_{xxyy})\), \((T_{xx},T_{yy},T_{xxyy})\), and then device evaluators expose
  total \(P,T,c_v,c_s\) from the resulting C² tensor-product spline without runtime total-EOS
  `e→T` inversion.
- `Materials::HelmholtzSpline`（`src/materials/helmholtz_spline.hpp`, `helmholtz_spline.cpp`, `helmholtz_spline_device.hpp`, `helmholtz_spline_device.cuh`, `helmholtz_spline_device.cu`）：
  hydro-only smooth EOS surrogate. Despite the compatibility name, the implementation builds a shape-preserving C¹ bicubic Hermite surrogate for the raw `total` EOS `P(\log\rho,\log T)` and `e(\log\rho,\log T)` on CPU at initialization, stores node data for both fields \((P, P_x, P_y, P_{xy})\), \((e, e_x, e_y, e_{xy})\), and exposes device evaluators for \(T(\rho,e), P, e, c_v, c_s\).
- `Materials::HelmholtzJet`（`src/materials/helmholtz_jet.hpp`, `helmholtz_jet.cpp`, `helmholtz_jet_device.hpp`, `helmholtz_jet_device.cuh`, `helmholtz_jet_device.cu`）：
  hydro-only local projected-jet surrogate for raw `total` EOS tables. CPU initialization builds nodal jets
  \((\phi,\phi_x,\phi_y,\phi_{xx},\phi_{yy},\phi_{xy})\) on \((\log\rho,\log T)\), applies local positivity-preserving clamps for \(c_v\) and \(\partial P/\partial\rho|_T\), then packs per-cell tensor-product biquintic Hermite coefficients. Device evaluators expose \(T(\rho,e), P, e, c_v, c_s\).
- `Materials::HelmholtzBSpline`（`src/materials/helmholtz_bspline.hpp`, `helmholtz_bspline.cpp`）：
  CPU-only Phase 1 fitter for a future thermodynamically consistent hydro EOS backend. It solves a weighted least-squares projection of the raw `total` table onto a quintic tensor-product B-spline representation of \(\phi(\ln\rho,\ln T)=F/T\) using reduced knot density, monotone-Hermite reference fields for \(c_v\) and \(\partial P/\partial\rho|_T\), and an active-set KKT solve that enforces positivity constraints at selected data nodes. It logs nodal reconstruction / positivity / \(T(\rho,e)\) inversion diagnostics and is not yet wired into runtime hydro kernels.
- `Materials::IONMIXBinaryEOS`（M17+）：IONMIX .cn4 バイナリからの EOS データロード（`load_ionmix_binary_eos`）。12 EOS ブロック（Zbar, P_i, P_e, e_i, e_e）を単位変換して `IonmixEOSData` に格納。`ionmix_eos_to_table_pair` で EOSTablePair（total + electron）に変換、`ionmix_eos_to_zbar_table` で `IonmixZbarTable` に変換。
- `Materials::NLTEOpacity`（M17）：IONMIX .cn4 の3種不透明度テーブル（κ^PA_g, κ^PE_g, κ_R,g）のロード・デバイス転送・補間（§4.3.2a）
- `Materials::OpacityDiagnostics`（`src/materials/opacity_diagnostics.hpp`, `opacity_diagnostics.cu`）：
  startup-only host diagnostic for hard-X-ray \(\kappa^{PA}\) audit logging. It reads
  the configured material opacity through the Materials readers and has no dependency on
  `radiation/`.

**禁止**：Hydro/Radiation/Laserがテーブルを直接読むこと（単位事故を防ぐ）。

#### 4.3.1 EOS GPU実行モデル

EOS 評価はセル毎の独立計算で、表 EOS は `__device__` 関数（`materials/eos_device_table.cuh`）を各演算子のカーネルが
インラインで呼ぶ。独立の EOS カーネル（旧設計の `eos_forward` / `eos_inverse`、CUDA_KERNELS H13/H14）は存在しない。

- **表の device 配置**：host の `EOSTable`（`materials/eos_table.hpp`）を `DeviceEOSTable::upload` が device へ写し、
  カーネルは `DeviceEOSTableView`（格子・\(P, e, c_v\) の表・任意の cold branch）を受け取る。多材料のデッキでは材料別の
  view 配列とセルの支配材料の添字（`CellEOSTableSelector`）でセルごとに表を選ぶ（NUMERICS §1.1.5）。
- **順方向**（\(T\to e, P, c_v\)）：`device_eos_energy` / `device_eos_pressure` / `device_eos_cv`（\((\log\rho,\log T)\) の双線形、
  表の端でクランプ。低密度の延長と energy_authoritative の高温側の延長は別関数）。
- **逆方向**（\(e\to T\)）：`device_inverse_reclose`。Newton 反復ではなく、温度の節点の上で根を含む区間を単調に探して
  区間内の線形の式で解く（反復なし）。床・表の端でのクランプと区間を作れない場合（bracket failure）は結果の構造体の
  フラグで返し、閉包が回数を数える（旧設計の `eos_newton_nonconverge` などのエラーフラグは無い）。
- **音速**：`device_eos_sound_speed`（補間関数の解析導関数からの \(\Gamma_1\) 形。\(c_s^2\le0\) なら \(\sqrt{(5/3)P/\rho}\)、
  NUMERICS §1.1.6）。
- **呼び出し元**：hydro の閉包（`enforce_1t/2t_closure_kernel`）・音速・エネルギー更新、伝導後の再閉包、FLD/S\(_N\) の物質更新、
  レーザー・燃焼・輻射の入射の閉包が、それぞれのカーネルの中で上の関数を呼ぶ。
- 理想気体EOS（NUMERICS §1.1.5 (a)）は解析式のためテーブル不要。

#### 4.3.2 EOSTable / OpacityTable 構造体

表は host の構造体で読み込み、device へ写して view でカーネルへ渡す。補間は `(log ρ, log T)` の双線形
（NUMERICS §1.1.5 (b)）。

```cpp
// materials/eos_table.hpp（host）。1 本の表 = 1 つの成分（ion / electron / total）
struct EOSTable {
    std::vector<double> rho_grid, T_grid_eV;       // 密度 [g/cm³]・温度 [eV] の格子（log 格子も保持）
    std::vector<double> P_table, e_table, cv_table; // T-major [j_T * n_rho + i_rho]。cv は e の差分（TMAT の cv 欄があればそれ）、下限 1e-3
    const ColdEquilibriumTable* cold = nullptr;     // 低温平衡構成則の補正（電子の表だけ。NUMERICS §1 (b)）
    void finalize();                                // 導出量の作成
};
struct EOSTableTriplet { EOSTable ion, electron, total; };   // 材料ごと（MatDef::eos_tables、shared_ptr）
// device: DeviceEOSTable::upload → DeviceEOSTableView（materials/eos_device_table.cuh、§4.3.1）
// 断熱指数 γ の表は無い（表 EOS の音速は補間関数の導関数から、NUMERICS §1.1.6）

// materials/ionmix_reader.hpp（host）/ ionmix_reader.cuh（device view）。群別の不透明度表（IONMIX・TMAT 共通）
struct IonmixOpacityData {
    std::vector<double> numdens_cm3, temps_eV;      // 密度軸はイオン数密度 n_i（TMAT も同じ）
    int ngroups; std::vector<double> bounds_eV;     // 群の数と境界（最初の表が Radiation.groups を決める、NUMERICS §6.7）
    std::vector<double> kappa_PA, kappa_PE, kappa_R; // [g][n_i][T]、質量不透明度 [cm²/g]
};
// 補間は log κ の双線形（隅が 1e-30 以下なら κ の線形へ切替）、表の範囲でクランプ（NUMERICS §1.1.5 (b)・§11.3）
```

> **EOS テーブル管理（2T モデル）**：
>
> **SESAME**：テーブル 301（total EOS）とテーブル 304（electron EOS）を**独立グリッド**で読む（NUMERICS §1.1.5 (b)）。
>   - `total`：301 のグリッド。`electron`：304 のグリッド（304 が無い材料は、材料の電離モデルの \(\bar Z\) で 301 を節点ごとに
>     \(\bar Z/(1+\bar Z)\) に分けた表 — `split_sesame_electron_table`、2026-09-29）
>   - `ion`：読み込み時に 301 のグリッド上で `build_sesame_ion_table` が作る（304 を実行時と同じ双線形・対数補間で 301 の節点へ
>     写して差を取る）。負の節点はクランプせずに保持し、数を 1 回警告する（旧設計の `max(0, …)` と `eos_ion_negative` は無い）
>   - 不透明度表 502/505 は `sesame_reader` が読めるが、実行時の不透明度には使わない（`opacity.model="sesame"` は `ConfigError`）
>
> **IONMIX**：IONMIX v4/v6 .cn4 バイナリファイルは `e_i, P_i`（イオン）と `e_e, P_e`（電子）を1ファイルに格納する。
> 密度軸はイオン数密度 \(n_i\) [cm⁻³]（SPECIFICATION §6.4.3 参照）。EOS 単位は J/g, J/cm³ で、cgs 変換（×10⁷）が必要。
> 読み込み時に \(\rho=n_iAm_p\) へ変換して `ion`・`electron`・`total` の 3 表を作る（比熱は数値微分）。
>
> **TMAT-H5**：`/eos/fields` の \(P_i, P_e, e_i, e_e, \bar Z\)（任意で `cv_i`・`cv_e`）から同じ 3 表を作る（§4.3.2b）。
>
> **共通**：Zbar テーブルは材料ごとの別の表（`MaterialsConfig::zbar_tables`）。混合セルは各セルの支配材料の表で閉じる
> （質量分率での合成はしない、NUMERICS §1.1.5 (c)）。

#### 4.3.2a NLTEOpacityTable 構造体（M17: IONMIX 3種不透明度 Non-LTE）

Non-LTE Phase 1（M17）で導入される不透明度テーブル構造体。
IONMIX4/6 `.cn4` バイナリファイルからロードし、デバイスメモリに配置する。
LTE の OpacityTable（§4.3.2）とは独立に管理される。

LTE の OpacityTable は κ^PA と κ_R の2種のみを格納するが、NLTEOpacityTable は
IONMIX4/6 の**3種不透明度**（κ^PA, κ^PE, κ_R）をすべて保持し、
\(\kappa^{PA} \neq \kappa^{PE}\) による非LTE放射輸送を可能にする。

```cpp
// Non-LTE 3種不透明度テーブル — IONMIX .cn4 由来、deviceグローバルメモリに配置
// NUMERICS §6.1.1、SPECIFICATION §6.4.3 準拠
struct NLTEOpacityTable {
    // --- 独立変数グリッド（log空間）---
    double* log_ni_grid;        // [n_dens] log10(n_i [cm⁻³])（イオン数密度）
    double* log_T_grid;         // [n_temp] log10(T_e [eV])
    int     n_dens;
    int     n_temp;
    int     n_groups;           // 群数 G

    // --- 材料パラメータ（密度変換用）---
    double  A;                  // 平均原子量 [amu]（ρ → n_i 変換: n_i = ρ/(A·m_p)）

    // --- 群別不透明度テーブル（3種分離）---
    // メモリレイアウト：group-major — kappa[g * n_dens * n_temp + id * n_temp + it]
    //   群が最外（outermost）、密度が中間、温度が最速（innermost）
    //   IONMIX バイナリのストリーム順序と一致
    double* kappa_R;            // [G × n_dens × n_temp] κ_R,g [cm²/g]（Rosseland）
    double* kappa_pa;           // [G × n_dens × n_temp] κ^PA_g [cm²/g]（Planck absorption）
    double* kappa_pe;           // [G × n_dens × n_temp] κ^PE_g [cm²/g]（Planck emission）

    // --- 群境界 ---
    double* bounds_eV;          // [G+1] 群境界 [eV]（単調増加）

    // --- テーブル範囲（clamp用）---
    double  ni_min, ni_max;     // [cm⁻³]（イオン数密度）
    double  T_min, T_max;       // [eV]

    // --- LTE/Non-LTE 判定フラグ ---
    bool    is_lte;             // |κ^PA - κ^PE| / max(κ^PA, ε) ≤ 1e-6 for all entries

    // --- device function: log(n_i)-log(T) bilinear 補間 ---
    // ρ [g/cm³] → n_i = ρ/(A·m_p) 変換はホスト側 or カーネル内で実施
    __device__ double interp_kappa_R(int g, double log_ni, double log_T) const;
    __device__ double interp_kappa_pa(int g, double log_ni, double log_T) const;
    __device__ double interp_kappa_pe(int g, double log_ni, double log_T) const;
};
```

> **設計方針**：
> - NLTEOpacityTable は材料種毎に1インスタンス（OpacityTable と同様）
> - `opacity.model = "table_nlte"` の場合のみロードされる
> - IONMIX .cn4 のヘッダ・EOS ブロック・不透明度テーブルを順次読み込み。
>   EOS ブロックは既存の IONMIX EOS ローダと共有し、不透明度テーブルのみ NLTEOpacityTable が管理する
> - **密度軸変換**: IONMIX の密度軸はイオン数密度 \(n_i\) [cm⁻³]。
>   TENRYU 内部の質量密度 \(\rho\) [g/cm³] からの変換: \(n_i = \rho / (A \cdot m_p)\)
> - **κ^PE 不在時の LTE フォールバック**: IONMIX ファイルに Planck emission テーブル（3番目）が
>   存在しない場合、\(\kappa^{PE} = \kappa^{PA}\) と仮定する（WARNING 出力、`is_lte = true` に設定）
> - 負の κ^PA, κ^PE, κ_R はロード時の全要素検査で拒否する（クランプしない — `ionmix_reader.cpp` がエラーで止める）
> - Fortran レコードマーカーの整合性チェックを実施（不整合時は `ConfigError`）
> - **η_g はテーブルに格納しない**: η_g = σ^PE_g × c × a_eV × T^4 × b_g(T) として
>   実行時に構成する（1D FLD の不透明度・放出の評価、NUMERICS §6.7。`CellRadiationCoeffs` は退役した IMC の構造体）

#### 4.3.2b TMAT-H5 Material Table

TMAT-H5 は IONMIX4/SESAME の代替として導入する self-describing HDF5 物性テーブル形式であり、
EOS と多群不透明度を 1 ファイルで保持できる。

- **Format identifier**：`tenryu.material_table.hdf5`
- **ファイル拡張子**：`.tmat.h5`
- **単位系**：`cgs_eV`（TENRYU ネイティブ。ロード時の単位変換は行わない）
- **密度軸**：`n_i` [cm⁻³]（IONMIX と同じイオン数密度軸）
- **グリッド**：EOS と opacity は独立グリッドを許容（共有を要求しない）
- **内部変換**：EOS/Zbar は `A_amu` を用いて \(\rho = n_i A m_p\) [g/cm³] に変換して保持し、
  opacity は `n_i` 軸のまま `IonmixOpacityData` に受け渡す

**HDF5 グループ階層（v1.0）**：
- `/`：`format_id`, `schema_version`, `units_system` などのルート属性
- `/material`：組成情報（`Z`, `A_amu`, `mass_fraction`, `Abar_ion_amu`）
- `/eos`：EOS 軸・場（`/grid/ni_cm3`, `/grid/temperature_eV`, `/fields/*`）
- `/opacity`：不透明度軸・場（`/grid/ni_cm3`, `/grid/group_bounds_eV`, `/fields/kappa_*`）
- `/provenance`：生成元・変換履歴（推奨）
- `/extensions`：将来拡張（任意）

**メモリレイアウト**（C row-major）：
- EOS：`[D,T]`（index = `d * nT + t`、`T` fastest）
- Opacity：`[G,D,T]`（index = `g * nD * nT + d * nT + t`、`T` fastest）

**Reader API**（`Materials::Tables`）：
- `load_tmat()`：`.tmat.h5` を読み込み `TmatFile` を構築
- `tmat_eos_to_table_pair()`：`TmatFile.eos` を `EOSTablePair` に変換（`[D,T] -> [T,D]`）
- `tmat_to_ionmix_opacity()`：`TmatFile.opacity` を `IonmixOpacityData` 互換データへ変換（`[G,D,T]` は保持、`ni_cm3` はそのまま転写）

**データフロー**：

```text
material.tmat.h5
   -> load_tmat()
   -> TmatFile {material, eos, opacity}
      -> tmat_eos_to_table_pair() -> EOSTablePair
      -> tmat_to_ionmix_opacity() -> IonmixOpacityData
```

| 項目 | IONMIX4 (`.cn4`) | SESAME (xSESAME ASCII) | TMAT-H5 (`.tmat.h5`) |
|---|---|---|---|
| 形式 | Fortran unformatted binary | 固定幅 ASCII | HDF5 |
| Self-describing metadata | なし | なし | あり |
| 密度軸 | `n_i` [cm⁻³] | `rho` [g/cm³] | `n_i` [cm⁻³] |
| 多群 opacity | あり | grey のみ | あり |
| EOS/opacity 独立グリッド | 不可 | 実質不可 | 可能 |
| TENRYUロード時単位変換 | 必要 | 必要 | 不要（`cgs_eV`） |

#### 4.3.3 Mixture mixing rules

多材料セルの不透明度の混合則（`Materials.mixture.opacity_mix_rule`、NUMERICS §1.1.5 (c)）：
- `linear_mass`（既定）：Planck・Rosseland とも \(\kappa_{mix} = \sum_m Y_m \kappa_m\)（\(Y_m\) = 質量分率、未追跡なら体積分率）
- `harmonic_mass_R`：Planck は線形、Rosseland は調和 \(1/\kappa_{R,mix} = \sum_m Y_m / \kappa_{R,m}\)
- `max`：両方とも存在する材料の最大値
- 表の材料は各材料の部分密度で評価して合成する（1D FLD の `eval_opacity_multimat_kernel`）

**Non-LTE 不透明度の混合則**（M17）：
- \(\kappa^{PA}_{mix,g} = \sum_m Y_m \kappa^{PA}_{m,g}\)（質量分率線形、吸収は additive）
- \(\kappa^{PE}_{mix,g} = \sum_m Y_m \kappa^{PE}_{m,g}\)（質量分率線形、ソース η_g に直結するため additive が物理的に正しい）
- \(\kappa_{R,mix,g}\)：既存の `harmonic_mass_R` を適用（Rosseland 平均は調和平均）
- 上の Non-LTE の記述は退役した IMC の設計で、`compute_opacities` カーネルと `CellRadiationCoeffs` は存在しない（1D FLD の非 LTE の
  混合は NUMERICS §1.1.5 (c) の「第 2 段」「第 3 段」）

EOS は混ぜない：1D の閉包は各セルを支配材料（体積分率が最大の非 void 材料）の EOS だけで閉じる（NUMERICS §1.1.5 (c)。
`Materials.mixture.eos_mix_rule` は受け付けるが無視する）。

伝導・ソース結合向けのセル実効量は非 void 材料の体積分率 \(f_m\)（その和で割り直す）で評価する（NUMERICS §1.1.5a）：
- \(A_{eff} = (\sum_m f_m/A_m)^{-1}\)（調和平均）
- \(\gamma_{eff} = \sum_m f_m \gamma_m\)（線形平均）
- \(n_e = \rho \bar{Z}/(A_{eff} m_p)\)
- \(c_{v,e} = \bar{Z}k_B/(A_{eff}m_p(\gamma_{eff}-1))\)、\(c_{v,i} = k_B/(A_{eff}m_p(\gamma_{eff}-1))\)
- `n_mat == 1` では \(A_{eff}=A_0,\ \gamma_{eff}=\gamma_0\) に退化し単一材料式と一致

#### 4.3.4 Materials トップレベル関数シグネチャ

Materials モジュールは「ステップ関数」を持たず、他モジュールのカーネル内で
`__device__` 関数として呼び出される（§4.3.1）。ホスト側APIは初期化を提供する。

```cpp
// Materials 初期化：テーブルファイル読込 → deviceメモリへ転送
// SESAME: xSESAME ASCII パース → 単位変換 → eos_total + eos_e 構築
// IONMIX: IONMIX v4/v6 .cn4 バイナリパース → eos_e + eos_i 構築
//   opacity.model="table_nlte" 時は NLTEOpacityTable に3種不透明度をロード
void materials_init(
    const Config::MaterialsConfig& mat_cfg,
    EOSTable* eos_e_out,            // [n_materials] 電子EOS（device確保済み）
    EOSTable* eos_i_out,            // [n_materials] イオンEOS（IONMIX時）/ total EOS（SESAME時）
    OpacityTable* opacity_out,      // [n_materials] LTE不透明度テーブル（κ^PA, κ_R）
    NLTEOpacityTable* nlte_out,     // [n_materials] Non-LTE不透明度テーブル（κ^PA, κ^PE, κ_R）
                                    // model="table_nlte" の材料のみ非NULL、それ以外はNULL
    int n_groups                    // 放射群数
);
```

不透明度の評価は輻射ソルバーの中で行う（IMC 時代の前計算カーネル `compute_opacities`（U9）と `CellRadiationCoeffs` は
退役し、記述を `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した）：1D は `fld_1d_gpu.cu` の
`evaluate_fld_opacity_and_emission`・`sn_transport_1d_gpu.cu` の `evaluate_opacity_and_emission`（材料ごとの記述と部分密度で
混合する `eval_opacity_multimat_kernel`、表の非 LTE 係数は `nlte_coeffs.cu` の `compute_nlte_coefficients_cuda_with_pe`・
`_pure_sn`）、2D_RZ は `fld_2d_rz_gpu.cu` の `evaluate_fld_opacity_and_emission`・`sn_transport_2d_gpu.cu` の `evaluate_opacity`。

**MixingRule enum**（`const char*` はGPUカーネルに渡せないため enum を使用）：

```cpp
enum class MixingRule : uint8_t {
    LINEAR_MASS = 0,
    HARMONIC_MASS_R = 1,
    MAX = 2
};
```
`const char*` はデバイスメモリ上の文字列比較が非効率かつ非推奨のため、
Config パース時に文字列→enum変換を行う。

---

### 4.4 hydro/
**責務**：流体更新（運動量・位置・密度・内部エネルギー）、伝導、境界条件

- `Hydro::LagrangianStep`
  - 圧力 + 人工粘性でノード加速 → 速度/位置更新
- `Hydro::EnergyStep`
  - PdV、人工粘性加熱（既定はイオンへ）、e‑i緩和、外部源（laser/rad）を反映
- `Hydro::ALE`（NUMERICS §3.3）
  - rezoning（メッシュ品質、NUMERICS §3.3.3 Winslow equipotential）
  - remap（保存的：質量/運動量/エネルギー/材料分率、NUMERICS §3.3.4）Mesh::Remap を呼び出す
  - `ale_align_monitor.{hpp,cpp}`：Stage 0 の default-off host 診断。
    post-Lagrange の `rho`, `Pe+Pi`, 節点座標、`volFrac`, vacuum mask から
    WLS gradient、structure tensor、coherence、i-face alignment angle を計算し、
    `[ale-align-diag]` へ log-only 出力する。simulation state は変更しない
  - `ale_align_rezone.{hpp,cpp}`：Stage 1--2 の host-only planar/RZ prototype。
    Stage-0 の cell director/coherence を frozen monitor として再利用し、
    family-specific direction-control energy を Q1 \(2\times2\) symmetric
    quadrature で評価する。default planar mode は Stage-1 path を維持し、RZ
    mode は quadrature-point の exact \(2\pi r\) weight、cell-center RZ scale
    付き preconditioner、axis-node projection、および exact signed RZ cell
    volume gate/exposure を追加する。deterministic central-FD gradient、
    fixed-iteration damped update、initial boundary line 上の tangential slip、
    global fixed-ladder line search、および全 Q1 corner Jacobian の relative
    floor は両 mode で共通。reference/HR/remap は含まず、ALE driver からは
    呼ばれない
  - post-remap reclosure は `HydroEOSContext` を受け取り、table-backed electron/ion EOS では共有 inverse reclosure + low-density policy を用いる。context がない成分は従来の理想気体 reclosure に戻る。
  - The env-gated shell protected rezone probe/commit path is owned by
    `src/hydro/ale_driver.cu`.  `Hydro2D::lagrangian_step` returns
    `HydroStepResult::shell_subcycle_committed` when that path commits at
    `post_corrector_commit`; `Coupling::Driver` uses that flag only to apply
    accounting and skip the scheduled post-step multiblock ALE for that step.
- `Hydro::ConservationAudit` (`src/hydro/conservation_audit.{hpp,cu}`)
  - Default-off env-gated (`TENRYU_I1B_CONS_AUDIT`) hydro/ALE
    conservation ledger. It records raw and owner-accounted mass/energy at
    Lagrangian, macro aggregate, BBSW, compatible-work, and ALE remap stage
    boundaries, and reports macro ownership count defects without changing
    state when disabled.
- `Hydro::PerMaterialEOS` (`src/hydro/per_material_eos_accessors.cuh`,
  `src/hydro/per_material_eos_project.{cuh,cu}`)
  - The per-material EOS refresh module. Raw-view
    `TENRYU_DEVICE` accessors reuse the shared materials inverse-reclosure and
    sound-speed helpers. The projection kernel maps per-material
    thermodynamic state back to cell means (mass-weighted
    \(T_e,T_i,\bar Z,c_v\), reduced `ee`/`ei`,
    volume-weighted \(P_e,P_i\), max-over-present
    \(c_s\)). The host launcher owns temporary lazy-cache valid-flag device
    mirrors and local dispatch counters, and exposes cache invalidation
    helpers for the per-material mutation sites.
- `Hydro::ALE1D`（1D_SPH の solution-adaptive ALE V3、`src/hydro/ale_1d_*`）は 2026-10-02 に退役した。コードはビルドから外し `retired/ale_1d/` に保管し、この項の記述は `retired/ale_1d/docs/ARCHITECTURE_ale_1d.md` へ移した。1D_SPH は pure Lagrangian で、1D にメッシュの再配置と状態量の remap は無い。
- `Hydro::PLIC` (`src/hydro/plic_geometry.{cuh,cu}`,
  `src/hydro/plic_normal.{cuh,cu}`, `src/hydro/plic_fast_path.{cuh,cu}`,
  `src/hydro/plic_remap.{cuh,cu}`)
  - The PLIC material-interface reconstruction core.  Geometry helpers
    provide RZ Pappus polygon clipping and bracketed alpha bisection; normal
    helpers provide Youngs-seeded LVIRA and deterministic degradation
    fallbacks; fast-path helpers build host-side interface masks.
  - Default-disabled PLIC material-volume remap entry points were added
    in `plic_remap`.  The ALE driver branches once at the material
    volume-fraction remap loop boundary: rho, momentum, electron/ion internal
    energy, diagnostic kinetic energy, and radiation energy stay on the scalar
    remapper, while `f_m V` may use PLIC face material-volume fluxes.
  - Runtime scratch is owned by `core::State` as non-checkpointed buffers:
    interface/active masks, reconstruction validity, normals, alpha,
    interface centroids, face material-volume fluxes, and per-cell residuals.
    The buffers are allocated lazily only when `numerics.plic.enabled` is true,
    `in_run_disabled` is false, and sticky fallback is not engaged.
  - The PLIC material-volume remap is serial-only.  `part.n_ranks > 1` with PLIC enabled is rejected at
    ALE entry; MPI validation is deferred to future work.  PLIC scalar fallback
    and class-(c) ALE escape valves report independently.
- 初期の ALE 整理で削除された旧 1D ALE path は復活させない。2D_RZ の `Hydro::ALE` / Winslow path は2D専用実装として維持する。
  - 2D_RZ ALE backtracking first evaluates the legacy 4-Gauss Jacobian
    `post_tangle` gate, then, when
    `Numerics.ale.corner_jacobian_post_tangle_enabled=True`, a parallel
    active-cell signed corner-J gate. Corner-J failures are reported separately
    in ALE backtrack telemetry and reject the candidate before remap-damage,
    remap-admissibility, or predictive-acceptance gates.
  - If all backtrack candidates fail the corner-J gate and
    `Numerics.ale.local_boundary_repair_enabled=True`, the driver may invoke a
    narrow boundary fallback that moves one non-corner top/bottom or outer-r
    boundary node along its boundary line. Accepted local repairs re-enter the
    normal ALE acceptance gates before remap.
  - If the local boundary interval is empty and
    `Numerics.ale.multi_node_boundary_repair_enabled=True`, the same emergency
    path may expand to a capped 1-ring coordinated repair solved by a small
    host-side linearized half-space projection. The candidate is globally
    revalidated before it can re-enter the normal ALE acceptance gates.
  - If the capped multi-node repair is infeasible and
    `Numerics.ale.emergency_cell_deactivation_enabled=True`, the driver may
    roll the mesh back, transfer the failing cell's mass/internal energy to a
    face-neighbor active cell, mark the failing cell void/inactive, and return
    without an ALE remap. This fallback is opt-in and globally revalidates
    active-cell Gauss/corner-J admissibility before it commits.
  - Driver retry may pass a host-only `AleRequest` (`src/hydro/ale_mode.hpp`)
    into `apply_ale_with_request` without changing the legacy `apply_ale`
    call surface.  The request dispatches local repair operators in
    `src/hydro/local_rezone.{cuh,cu}` and `src/hydro/cd_local_rezone.{cuh,cu}`;
    when no request is present, the scheduled 2D ALE path is unchanged.
- `Hydro::CFL`
  - dt_hydro の計算（NUMERICS §3.1.9, §3.2.13）
  - `src/hydro/rz_geometric_cfl.cu` provides the default-off 2D_RZ predictive
    geometric hydro CFL used by `compute_dt_hydro` to clamp the Lagrangian
    half-step before projected RZ cell volume drops below the configured
    fraction of the current volume or, when enabled, a fixed fraction of
    `State.cell_vol_initial`.  `compute_dt_hydro` can optionally provide a
    force-predicted \(\mathbf{u}^{1/2}\) instead of the current state velocity.
- `Hydro::CornerJacobianQuality`（`src/hydro/corner_jacobian_quality.{cuh,cu}`）
  - 2D_RZ cell corner の signed Jacobian を評価し、default-off の pre-hydro ALE trigger と Hydro2D pre-commit diagnostic に共有 predicate を提供する（NUMERICS §3.2.13）
  - G1 multiblock path guard は `Mesh::ReferenceFlatPath` の immutable typed
    mask を受け、regular corner の既存 strict quadratic を維持しながら
    reference-flat corner とその cell embedding を host exact predicate で合成する。
  - retry active mesh repair 用に、current corner-J の same-cell balance \(q_{bal}=\min J/\max J\) を GPU reduction で評価し、非正・非有限 corner と 100:1 既定 imbalance を dt-independent に検出する（NUMERICS §3.2.13b）
- `Hydro::Ring7SeamQuotientRemap` (`src/hydro/ring7_seam_quotient_remap.{cuh,cu}`)
  - I1-B `Ring7OuterSeamQuotientRemap` の default-off module. Increment 1 defines RZ face/swept-volume primitives, fixed-order compensated packet accumulation, ring-7 seam patch discovery, and diagnostic-only q/eta scanning.
  - Increment 2 adds a Hydro2D StepStart seam transaction under `Numerics.hydro.ring7_quotient_enabled` (or `TENRYU_I1B_RING7_QUOTIENT`). It fires only from a one-shot request armed proactively by an accepted-step multiblock path-margin guard band or reactively when the driver retries a production `mesh_quality_rz_volume` or `multiblock_path_admissibility` failure on the Ring7 seam patch. The no-request path returns before seam allocation, metric evaluation, dt reduction, or state mutation. The reactive path restores the full-step retry snapshot, carries the failing cell in a single-use runtime request, and reruns Hydro2D at the same `dt`; driven pole-cap boundary failures are routed away from the seam remap by the existing candidate-cell predicate.
  - The transaction symmetry-pairs tangential ring-7 seam-node motion, checks exact path admissibility through `Mesh::PathAdmissibility`, and checks B_V through the production multiblock RZ-volume root kernel.
  - Increment 3a does not call the global ALE remap path. The accepted target is consumed only by a dedicated seam packet-geometry scaffold that promotes every cell whose multiblock `cell_node_csr` node list references any moved seam node, enumerates internal/boundary faces from `MultiBlockTopology::unique_internal_faces`, splits signed RZ swept volumes by the existing `State.corner_volume` edge-corner weights, and logs `[ring7_seam_packet]` geometry/patch/core-isolation diagnostics.
  - Increment 3b applies rebuilt geometric submove packet ledgers to host working copies, validates donor-corner positivity, whole-domain mass/total-energy and paired-momentum deltas, and central-core hash/sum isolation, then commits `State.x_r/x_z`, conservative cell fields, node velocities, transported corner masses, and refreshed mesh geometry only after all gates pass. It still does not call the global ALE remap/reference-volume/node-kinematics machinery.
  - Increment 4a keeps the seam packet remap as an interior-seam repair and adds a diagnostic-only driven-pole cap oracle for cell-608-class domain-boundary RZ-volume failures. Hydro2D retains the exact failed production limiter velocity field, the driver requests a single StepStart oracle on the restored retry, and the oracle sweeps small cap sizes and normal-velocity taper options through `compute_mesh_quality_dt_limit()` without committing coordinates or state.
  - Increment 4b commits the selected pole-cap ALE candidate through a separate host packet transaction from the retained failed Lagrangian geometry to the cap mesh geometry. The relative target equals the retained Lagrangian geometry outside the cap, and the coordinate commit writes only cap nodes so the normal hydro step still owns the off-cap Lagrangian motion. The transaction uses deterministic internal-face packets, a driven-boundary volume ledger, donor-corner positivity and central-core/conservation gates, then marks a one-shot validation so the next Hydro2D production limiter call logs the recomputed retry `u_half` pass/fail at the requested pole cell.
- `Mesh::PathAdmissibility` (`src/mesh/path_admissibility.cuh`)
  - Multiblock CSR quad paths \(X^n\rightarrow X^{n+1}\) の signed corner-J,
    2x2 Gauss-J, and polygon-area quadratics are evaluated on CUDA before
    Hydro2D accepts the Lagrangian corrector.  The module is header-only so the
    Hydro2D caller can launch the check without adding a mesh library source.
    Default-off anatomy (`TENRYU_I1B_PATH_ADMIS_ANATOMY`) annotates the winning
    rejection metric kind and block-local cell coordinates.  The separate
    default-off hardening gate (`TENRYU_I1B_PATH_PREDICATE_HARDEN`) switches the
    live predicate to `cell_orientation_sign` orientation, scaled/canonical loop
    guards, and repair-only lambda-zero old-geometry classification; unset uses
    the legacy live predicate path.  The optional `Hydro::PoleAngularCoarsen`
    overlay is consumed here as accepted quotient macro boundaries plus a
    separate fine-child skip mask when `skip_fine_child_paths=true`; the motion
    pilot sets that flag false so covered fine cells are still scanned.
- `Hydro::PoleAngularCoarsen` (`src/hydro/pole_angular_coarsen.{cuh,cu}`)
  - Default-off I1-B Q2 geometry/path pilot.  It constructs the POLAR_SHELL
    radial-band metadata and true quotient macro perimeters for Hydro2D path
    checks.  Macro angular arcs contain only interval endpoints; radial sides
    retain intermediate nodes when the selected q-band spans multiple rows.
    Under `TENRYU_I1B_POLE_MOTION_PILOT=1`, Hydro2D also asks the helper to
    reconstruct candidate mesh position velocities from accepted macro endpoint
    motion for the q-band node rows plus a default-four-row inward smoothstep
    taper controlled by `TENRYU_I1B_POLE_MOTION_TRANSITION_ROWS` and
    `TENRYU_I1B_POLE_MOTION_PROFILE`.  It does not mutate mesh topology,
    hydrodynamic state, pseudo-core
    masks, or ALE/remap data structures.
- `Hydro::PoleAxisBBSW` (`src/hydro/pole_axis_bbsw.hpp`)
  - Header-only host helper for the default-off
    `TENRYU_I1B_POLE_AXIS_BBSW` Hydro2D pole-column closure.  It owns the
    planar corner/edge geometry primitives, the roundoff-scale hard-gap
    function, and the weighted PAVA projection used by `hydro_2d.cu`; it does
    not add state arrays, HDF5 schema, or namelist parameters.
- `Hydro::AntiHourglass`（`src/hydro/anti_hourglass.{cuh,cu}`）
  - 2D_RZ default-off Caramana-Shashkov subzonal pressure anti-hourglass
    force（NUMERICS §3.2.9b）。`Hydro2D::lagrangian_step` が pressure/AV
    acceleration 後、predictor/corrector velocity update 前に呼び出す。
    Compatible work が有効な場合は同じ force work を cell internal energy
    increment へ加える。Pure-Lagrange Hydro2D step では runtime
    `state.corner_mass` and `state.subzonal_mass_corner{0,1,2,3}` は
    IC-time initialization 後に Lagrangian-invariant として保持し、
    `state.corner_volume` だけを current mesh から更新する。ALE remap
    accepted compatible multiblock ALE remap 後は
    `ale_remap_2d_rz.cu` の subzonal-aware remap path が corner-mass
    fractions \(m_{c,k}/M_c\) を passive mass-weighted scalars として
    CSR conservative remap で transport し、\(\sum_k m_{c,k}=M_c\) に
    closure する。compatible-off path は従来通り warning 付き
    post-remap geometry 再初期化を行う。
  - Phase 3 multiblock path は cell-node CSR と reverse node CSR を使い、
    `state.corner_mass[c*4+k]` / `state.corner_volume[c*4+k]` を exact
    centroid+midpoint subpolygon partition で管理する。Nodal force and
    nodal mass assembly は incident corner sum で行い、single-block /
    tri_fan の default-off paths は変更しない。
- `Hydro::CompatibleForceWork2D`
  (`src/hydro/compatible_force_work_2d.{cuh,cu}`)
  - Compatible Lagrangian force/work scratch。`av_model=csw_edge`
    and `subzonal_pressure_enabled=true` の時、Hydro2D は pressure corner
    force, subzonal corner force, and edge AV force を separate in-memory
    buffers に zero/assemble し、corner-to-node and edge-to-node force
    accumulation plus per-cell work scratch を計算する（NUMERICS §3.2.9b）。
  - Legacy `scalar_vnr_legacy + subzonal_pressure_enabled=false` は従来の
    scalar \(p_q\) force assembly を呼び続ける。Compatible work scratch は
    HDF5/checkpoint schema には含めない。T5 以降の compatible path では
    corrector 後に half-state force を再計算し、同じ force と
    \(\bar u=(u^n+u^{n+1})/2\) から pressure/subzonal work を
    \(P_e/(P_e+P_i)\), \(P_i/(P_e+P_i)\)（cold limit 50/50）で 2T split
    し、CSW AV work は ion energy へ全量 deposit する。
- `Hydro::CompatibleAvCsw`
  (`src/hydro/compatible_av_csw.{cuh,cu}`)
  - T3 2D_RZ `av_model=csw_edge` force package。current corner positions
    から pressure operator と同じ full-\(2\pi\) RZ convention の edge median
    vector を作り、CSW edge force, per-cell AV work, compressive-edge count,
    and edge-relative AV CFL component を計算する（NUMERICS §3.2.9b）。
  - Multiblock limiter neighbor lookup は `face_adj_csr_offsets/indices` を
    device temporary として使い、sentinel `-1` seam/boundary は missing
    neighbor ratio 1 として扱う。HDF5/checkpoint schema は変更しない。
- `Hydro::CompatibleSubzonalPressure`
  (`src/hydro/compatible_subzonal_pressure.{cuh,cu}`)
  - `subzonal_pressure_enabled=true` path の canonical
    Caramana-Shashkov subzonal pressure force package。runtime
    `state.corner_mass` / `state.corner_volume` から corner density and
    subzonal pressure perturbation を評価し、cell-corner force and
    per-cell work scratch を compatible force buffer に assemble する
    （NUMERICS §3.2.9b）。HDF5/checkpoint schema は変更しない。
- `Hydro::MeshMotionTrace` (`src/hydro/mesh_motion_trace.hpp`)
  - Phase 3 mesh-freeze diagnosis 用の default-off host trace helper.
    `Numerics.debug.trace_mesh_motion=True` の場合だけ Hydro2D/ALE から
    device fields を host にコピーし、mesh-motion stages を stderr に出力する。
    Kernel force assembly, remap logic, and HDF5 schema are unchanged.
- `Hydro::MeshRegime` (`src/hydro/mesh_regime.{hpp,cuh,cu}`)
  - Default-off regime metadata for the 2D_RZ corner-J guard.  The
    host-readable POD types live in `mesh_regime.hpp`; CUDA declarations and
    the classifier live in `mesh_regime.cuh/.cu`.
  - The driver owns `MeshRegimeDeviceCache` and passes it to Hydro2D, avoiding
    `core::State` schema churn.  The cache holds the current `CellRegime` array
    plus one previous-primary-regime byte per cell for hysteresis.  At
    512x1024 cells this is about 13 MiB with the current compiler padding; no
    allocation occurs while
    `Numerics.hydro.regime_aware_corner_j_guard_enabled=False`.
- `Hydro::LocalRezone` (`src/hydro/ale_mode.hpp`,
  `src/hydro/local_rezone.{cuh,cu}`, `src/hydro/cd_local_rezone.{cuh,cu}`)
  - Host-side retry repair selector and local projection operators for
    driver-requested 2D_RZ ALE retries.  `AleMode` / `AleRequest` /
    `RepairPlan` are host-only data contracts; CUDA kernels are not launched
    from the header.  The legacy Winslow, axis-spine, and boundary repair
    operator bodies remain in their original modules.
- `Hydro::ALEGCL` (`src/hydro/ale_gcl.{hpp,cu}`)
  - The Geometric Conservation Law residual hook for `apply_ale_with_request`
    tail coverage of all ALE invocation paths.  It is a pure diagnostic and
    does not change ALE acceptance, remap state, or physics behavior.  The
    multiblock audit path is CSR-native for cell-node lookup and velocity
    averaging, covering both the legacy three-block mesh and the B-S2
    five-block hydro smoke.
- `Hydro::AxisALERezone` (`src/hydro/axis_ale_rezone.{cuh,cu}`)
  - Target-only primitive for the default-off 5-block half-butterfly axis ALE
    path.  It applies exact \(O(N)\) weighted lower-bound PAVA to the
    positive-mass subset of the ordered physical \(R=0\) chain derived from
    `mesh::build_full_axis_node_chain` and returns target \(Z^*\) plus first
    off-axis ring min-edge/min-altitude diagnostics. `src/hydro/ale_driver.cu`
    owns dormant-cell corner-mass zeroing, active-DOF compaction, the
    post-Lagrangian trigger, target installation, and existing CSR conservative
    remap. The primitive does not mutate state, mesh coordinates, velocity,
    force, pressure, or energy arrays.
- `src/hydro/axis_band_guard.{hpp,cu}` — 2026-07-27
  で transaction 基盤へ移行: band 行 prefix を core::ShadowTransaction の device arena に
  byte-exact capture する AxisBandGuard（旧 axis_band_snapshot.{hpp,cu} の D2H/H2D 実装を
  置換・削除。検証 assert 契約は旧実装から逐語維持）。
- `src/hydro/axis_band_margin.{cuh,cu}` — managed axis-band controller:
  row-K margin diagnostics and K-selection for managed axis-band remap.
- `src/hydro/axis_band_remap.{cuh,cu}` — managed axis-band controller:
  band-only swept-volume equal-volume remap with positivity hard gates and
  conservation diagnostics.
- `src/hydro/ale_axis_band_controller.{cuh,cu}` — managed axis-band controller:
  controller orchestration for margin evaluation, snapshot/restore K fallback,
  band remap commit, and post-remap EOS reclosure.
- `Hydro::ArtificialViscosity`（`src/hydro/artificial_viscosity.hpp`, `artificial_viscosity.cu`）
  - 1D_SPH: Christensen 速度リミタからノード勾配 \(\sigma_j\) とセル圧縮センサ \(\chi_i\) を構成し、
    \(Q_i = \phi_i \rho_i (C_2^2 \Delta r_i^2 \chi_i^2 + C_1 \Delta r_i c_{s,i} \chi_i)\) を評価
  - \(\phi_i = W_{shock,i}\max(0.25,\;W_{comp,i}W_{osc,i})\) とし、
    total pressure \(P=P_e+P_i\) の jump・密度 jump・RH整合性・圧縮Mach数・odd-even 指標で
    source-heated front と実 shock を分離する
  - `W_shock` は developed shock branch と pressure-dominated precursor branch の2分岐を持ち、
    Sedov blast launch のような極端な圧力駆動 shock-support を維持する
  - `W_osc` は Quirk 型の pressure sign-flip を抑制するが、両側 interface が developed shock のときは解除する
  - `av_type="riemann"` では 1D_SPH face で midpoint cell velocity の minmod reconstruction と
    nonlinear impedance \(Z^{eff}=\rho(c_s+\alpha\Delta u^+)\), \(\alpha=(\Gamma_1+1)/4\)
    による acoustic Riemann pressure correction を計算し、
    \(Q_i=0.5(Q_{i-1/2}+Q_{i+1/2})\) として既存の force/work 経路へ渡す。
    この branch は VNR の shock-support gate、mild-compression branch、`av_C1`、`av_C2`、
    `av_limiter_J`、`av_eos_aware` を使用しない
  - 同じ \(\chi_i\) を 1D人工熱流束 `av_heat_C` にも用いる。
  - The per-material physics operators add `compute_q_per_material_2d` for 2D_RZ
    per-material AV pressure scratch. The 2D momentum path still consumes the
    scalar aggregate `state.Qvisc = sum_m volfrac_m Q_m`; per-material energy
    deposition in `hydro_2d.cu` consumes only the scratch to avoid double
    counting. 1D_SPH per-material AV remains future work.
- `Hydro::Conduction`（`src/hydro/conduction.cuh`, `conduction.cu`,
  `conduction_snb_2d.cuh`, `conduction_snb_2d.cu`）
  - 電子熱伝導：Spitzer-Härm + flux limiter（NUMERICS §4）
  - イオン伝導（オプション、既定OFF、1D・2T、2026-09-26）：`ion_conduction_step_1d`
    （`conduction.cu`）が電子の伝導と記帳の後に Braginskii の \(\kappa_i\) で後退 Euler の
    三重対角系（電子の陰解法の組み立てカーネルを共用し、640 セル以下は 1 ブロックの並列巡回縮約、それより長い線は cuSPARSE の gtsv2 で）を解き、
    \(e_i\) に記帳してから `Hydro1D::close_eos` で閉じ直す（NUMERICS §4.6）。呼び出しは
    Driver の伝導段（`run_conduction_phase`）の末尾
  - 1D_SPH：3点トリダイアゴナル離散化（NUMERICS §3.1.7）
  - 2D_RZ：Kershaw 9点ステンシル（NUMERICS Appendix A、§4.3）
  - SNB 非局所電子熱輸送（`nonlocal_model="snb"`、既定OFF）：2D_RZ port は
    `conduction_snb_2d.{cu,cuh}`（群バッチ Kershaw-CSR Jacobi-PCG + iSNB
    Picard、NUMERICS §4.5）。dispatch は `conduction_step_2d_sts` 内
    else-wrap、既存 kernel byte 不変。probe API は verify 専用
    （`snb2d::snb2d_probe`）。1D 実装は feature/1d-brushup ブランチ（merge で合流）
  - 多材料セルでは `Materials::Mixture` の \(A_{eff},\gamma_{eff}\) を使って
    `C1 compute_spitzer_deff` が \(n_e\), \(q_{max}\), \(c_{v,e}\), \(D_{eff}\) を評価
  - **2つのソルバパス**（`conduction.solver` で選択、NUMERICS §4.2.1/§4.2.3）：
    - **`"sts"`（既定）**：Super-Time-Stepping（Chebyshev加速明示的サブサイクリング）
      - ステージ数 \(s = O(\sqrt{N_{sub}})\) で \(O(s^2)\) 倍の安定領域を実現
      - コロナ領域での \(N_{sub}=130\text{–}340\) を \(s=16\text{–}27\) に削減
      - \(D_{eff}\) と Kershaw 係数はスーパーステップ開始時に凍結
    - **`"hypre"`（オプション、`-DTENRYU_ENABLE_HYPRE=ON` 必須）**：Hypre 陰的ソルバ
      - BoomerAMG 前処理付き PCG（Kershaw 行列は SPD）
      - 陰的定式化：\((diag(C_v/\Delta t) + M_{Kershaw})\, T_e^{n+1} = diag(C_v/\Delta t)\, T_e^n\)（C_v = ρc_v）
      - 伝導CFL制約なし（\(\Delta t_{cond} = \infty\)）→ 他演算子のΔtのみがグローバルΔtを決定
      - Kershaw係数（C2カーネル出力）→ `HYPRE_IJMatrix` 変換（デバイスメモリ上）
      - スパーシティパターン（9点固定）は初回のみ構築、以後は値のみ更新
      - Hypre 未ビルド時に `solver="hypre"` 指定 → `ConfigError`
  - 負温度防止clamp（NUMERICS §11.2、§4.2.2）
  - **SNB 非局所電子熱輸送**（`src/hydro/conduction_snb_1d.cuh`, `conduction_snb_1d.cu`;
    NUMERICS §4.4、opt-in `nonlocal_model="snb"`、1D 専用）：群別 H_g 拡散方程式を
    `cusparseDgtsv2StridedBatch` 群バッチで解き（radiation FLD A-3 と同パターン）、
    面補正流束 δq を外側 f_lim cap（θ 形）と合成して STS 超ステップへ注入する
    iSNB Picard 反復（Cao 2015）。stage kernel は歴史 body の byte 不変 clone
    （`snb_stage_kernel<GEOM, KIRCHHOFF>`）。verify 専用 probe API
    `conduction::snb_probe_fluxes`（Te 不変で面流束を返す）を公開
    （社内の検証記録 §7.9 の測定基盤）。既定 OFF は歴史経路 bit 恒等
    （GXII golden rel=0 ×6 で gate）。診断は ConductionResult snb_* +
    history `/diagnostics/conduction/snb/v1/*`。
- `Hydro::EOSContext`（`src/hydro/eos_context.hpp`, `eos_context.cu`）
  - per-material table-EOS context for hydro kernels (`HydroEOSContext`)
  - owns raw `DeviceEOSTable` vectors (`ion`, `electron`, `total`), hydro-only `DeviceEOSRhoETable` vectors (`total_rho_e`), hydro-only `DeviceHelmholtzSpline` vectors (`ion_helmholtz`, `electron_helmholtz`, `total_helmholtz`), hydro-only `DeviceHelmholtzJet` vectors (`total_helmholtz_jet`), and hydro-only `DeviceMieGruneisen` vectors (`mie_gruneisen`)
  - stores per-material backend selection (`hydro_backend_kind`) resolved from `Materials.materials[].eos.hydro_backend` (`0=legacy`, `1=helmholtz_spline`, `2=helmholtz_jet`, `3=exact_ideal_gas`, `4=rho_e_table`, `5=mie_gruneisen`)
  - legacy compatibility device view arrays (`d_ion_views`, `d_electron_views`, `d_total_views`) are retained for the raw table path
- `Hydro::BC`（`src/hydro/boundary.cuh`, `boundary.cu`; 2D_RZ semantic types: `src/hydro/bc_2d_rz_semantics.hpp`）
  - Lagrangianメッシュの流体境界条件（NUMERICS §8.1）
  - ゴーストセル/ゴーストノードの値更新
  - 5種別（SPECIFICATION §6.4.7）：free（P_ext=0）、fixed（v=0固定壁）、reflect（スリップ壁 v_n=0、v_t自由）、pressure（P_drive(t)、2D_RZ では r_outer のみ。z_bottom/z_top で pressure 指定は ConfigError）、state_supply（2D_RZ z-face のみ。境界 row の rho/mass/Te/Ti と material v_z を reservoir 供給値へ戻す）
  - 2D_RZ は `BC2DRZConfig` に normal/tangential material 条件、normal/tangential mesh 条件、PR B 用 open-flow remap eligibility、state-supply donor mode を展開する。`boundary_2d.mesh_tangential_target="reference"` では clamped r-face の z 座標と clamped z-face の r 座標を IC reference mesh に戻す。
  - 2D_RZ state_supply z-face の mesh anchoring は `boundary_2d.cu::apply_state_supply_z_bottom_node_kernel` / `apply_state_supply_z_top_node_kernel` が担当し、boundary node `x_z` を \(z_{min}/z_{max}\) に clamped し、mesh node `v_z` をゼロ化する。この constraint は material velocity ownership を持たない。
  - 中心境界（1D r=0）：速度反射、ゼロ流束
  - RZ軸（2D R=0）：v_R=0 強制
- `Hydro::StateSupplyBC`（`src/hydro/state_supply_bc.hpp`, `state_supply_bc.cu`）
  - 2D_RZ z-face state-supply 用の cell mask、zonal override、material velocity restoration、reservoir tally を担当する。`override_state_supply_kernel` は境界 row cell の `rho/mass/Te/Ti` を供給値へ戻し、material `v_z = supply_u_z_cm_per_s` を復元する。`restore_state_supply_material_velocity` は Hydro2D predictor/corrector boundary application 後にも同じ material `v_z` contract を再適用する。EOS 一貫性は Hydro2D 側の既存 EOS closure を再利用する。
  - Hydro2D の position update は state-supply z-boundary node 用の temporary `predictor_pos_z` / `corrector_pos_z` buffers だけをゼロ化して clamped mesh node を動かさない。一方、material `state.v_z` は supply velocity を保持し、hydro flux、diagnostics、reservoir mass/momentum/energy tally に使われる。
  - ALE projection/remap は state_supply を reflect と分離する。`ale_remap_2d_rz.cu::velocity_bc_mode_local` と `ale_driver.cu` は `STATE_SUPPLY` を mode 3 として渡し、`ale_velocity_project.cuh` は mode 3 の z-boundary `v_z` を拘束しない。Conservative remap の active state-supply z-face flux は projected node velocity ではなく `supply_u_z_cm_per_s` を boundary face speed として使う。
  - Reservoir tallies capture mass, z-momentum, and material energy deltas using the restored supplied state. Snap-0001 RH audit checks these fluxes to relative error \(\le 10^{-3}\) and asserts positive `supply_u_z_cm_per_s` on both z_bottom and z_top.

> **呼び出し関係**：Coupling::Driver → Hydro::LagrangianStep → Hydro::BC（ゴースト更新）
> → Hydro::Conduction（Strang splitting 内で独立ステップ）

**HydroEOSContext lifecycle (current implementation)**:
- Created in `coupling::Driver::run` as a stack object: `HydroEOSContext eos_ctx; eos_ctx.initialize(cfg);`
- `initialize(cfg)` uploads per-material raw table EOS (`mat.eos_tables`) into owned `DeviceEOSTable` containers and, when requested by `mat.hydro_eos_backend == "rho_e_table"`, builds the hydro-side direct `total P/T(\rho,e)` table using either the default log grid or the diagnostic linear grid selected by `cfg.numerics.hydro.rho_e_linear_grid`, then uploads it into `total_rho_e`; when requested by `mat.hydro_eos_backend == "helmholtz_spline"`, it builds the hydro-side bicubic `total P/e` surrogate and uploads it into `total_helmholtz`; when requested by `mat.hydro_eos_backend == "helmholtz_jet"`, it builds the hydro-side projected-jet biquintic `total` surrogate and uploads it into `total_helmholtz_jet`; when requested by `mat.hydro_eos_backend == "mie_gruneisen"`, it builds the branchwise affine `P_{ref}(\rho), e_{ref}(\rho), \Gamma(\rho)` fit and uploads it into `mie_gruneisen`; when requested by `mat.hydro_eos_backend == "exact_ideal_gas"`, it keeps the raw tables uploaded but marks the material for analytic ideal-gas closure inside the 1D hydro table-kernel path.
- Hydro entry points query `HydroEOSContext` once per step to select either the legacy raw-table path, the `exact_ideal_gas` path, the `rho_e_table` path, the `helmholtz_spline` path, the `helmholtz_jet` path, or the `mie_gruneisen` path; ALE post-remap reclosure also consumes the context for table-backed electron/ion inverse reclosure and otherwise keeps the ideal-gas fallback. Table backends write the closed energy back by default (`cfg.numerics.hydro.eos_writeback=true`; under the default `eos_closure_mode="energy_authoritative"` a closure that clamps at a table edge or floor keeps the hydro-updated energy instead), and `eos_writeback=false` keeps the hydro-updated `ee/ei` and only repairs NaN / Inf / negative energies with table/surrogate-clamped values. The `mie_gruneisen` path is 1D/2T-only and differs in one important way: predictor/corrector hydro uses only the uploaded affine closure for `Pe/Pi/cs`, while `Driver` refreshes raw-table `Te/Ti/cv_e/cv_i` outside hydro after hydro/source phases. In 1D, `cfg.numerics.hydro.exact_override` can further replace one post-closure quantity (`pressure`, `sound_speed`, or `temperature`) with a diagnostic ideal-gas value on table backends, and `exact_override="no_writeback"` forces writeback off for compatibility. Radiation/opacity modules continue to use the raw table data directly.
- Passed by pointer to hydro entry points (`prepare_initial_sound_speed`, `lagrangian_step`) for both 1D and 2D hydro paths, and to `Hydro::ALE` for 2D_RZ post-remap reclosure.
- `HydroEOSContext` owns GPU memory via RAII (`destroy()` + destructor + move support). Memory is released automatically when `Driver::run` exits.

#### 4.4.1 Hydro トップレベル関数シグネチャ

```cpp
struct HydroResult {
    double dt_cfl;              // [s] CFL制約から算出された推奨Δt
    int n_floor_applied;        // フロア適用回数（このステップ）
    double E_floor_injected;    // [erg] フロア注入エネルギー（このステップ合計）
};
```

`hydro_step` は `HydroResult` を返す。

```cpp
// Hydro半ステップ（Strang splitting H(Δt/2)）— NUMERICS §3.1, §3.2
HydroResult hydro_step(
    State& state,                   // メッシュ・流体場・hydro_active
    const EOSTable* eos_e,          // 電子EOS（device）
    const EOSTable* eos_i,          // イオンEOS（device）
    double dt,                      // 半ステップ幅 Δt/2 [s]
    const Config::NumericsConfig& num,  // AV係数、境界種別
    const PartitionInfo& part,      // 並列情報
    CommBuffers& comm,              // ハロー交換バッファ
    cudaStream_t stream
);

// 電子熱伝導フルステップ（Strang splitting C(Δt)）— NUMERICS §4, §4.2.1/§4.2.3
// ConductionConfig::solver が "sts" → STS明示的、
// "implicit" → 1D_SPH backward Euler + 三重対角直接解法、
// "hypre" → 2D_RZ Hypre陰的 を分岐
void conduction_step(
    State& state,
    const EOSTable* eos_e,
    double dt,                      // フルステップ幅 Δt [s]
    const Config::NumericsConfig::ConductionConfig& cond,
    const PartitionInfo& part,
    CommBuffers& comm,
    cudaStream_t stream,
    HypreSolver* hypre = nullptr   // solver="hypre" 時のみ非null。Hypre未ビルド時は常にnullptr
);

// イオン熱伝導（1D・2T、ion_conduction=True のとき）— NUMERICS §4.6。電子の伝導と
// その記帳の後に Driver が呼ぶ。e_i を記帳し、Hydro1D::close_eos で T_i, P_i を閉じ直す
// ところまでを自身で行う（下の契約は電子側のみ）。
IonConductionResult ion_conduction_step_1d(
    State& state, double dt, const Config& cfg,
    const PartitionInfo& part, cudaStream_t stream,
    const HydroEOSContext* eos_ctx);
```

**Post-conduction EOS sync 契約**：
STS/Hypre はいずれも `Te` を直接更新するが、`ee`（電子内部エネルギー密度）は更新しない。
`conduction_step` 完了後、呼び出し元（Driver）は以下を実行する義務がある：
1. `floor_clamp`（U2）：Te 安全策適用
2. `eos_forward`（H13）：更新後の Te から ee, Pe, Cv_e を再計算

これにより、後続の Radiation 演算子と2回目の Hydro 半ステップが
Te/ee/Pe/Cv_e の整合した状態を参照できる。
これがないと、伝導更新後の状態が放射係数や後続エネルギー更新に反映されない。CUDA_KERNELS §9 参照。

#### 4.4.2 Hypre 陰的拡散ソルバ（オプション）

`-DTENRYU_ENABLE_HYPRE=ON` 時のみコンパイルされる。

```cpp
#ifdef TENRYU_ENABLE_HYPRE
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>

// Hypre ソルバの永続コンテキスト（初回構築、以後再利用）
struct HypreSolver {
    HYPRE_IJMatrix    A_ij;       // Kershaw行列 + 質量対角（HYPRE_MEMORY_DEVICE）
    HYPRE_ParCSRMatrix A_parcsr;  // 内部ParCSR形式へのビュー
    HYPRE_IJVector    b_ij;       // RHS: diag(C_v/Δt) × T_e^n
    HYPRE_IJVector    x_ij;       // 解: T_e^{n+1}（初期推定 = T_e^n）
    HYPRE_Solver      solver;     // PCG ソルバ
    HYPRE_Solver      precond;    // BoomerAMG 前処理
    bool              structure_built;  // スパーシティパターン構築済みフラグ

    // 設定パラメータ
    double rtol;         // 相対収束判定（既定 1e-8）
    int    max_iter;     // PCG最大反復数（既定 50）
    int    amg_coarsen;  // BoomerAMG粗視化タイプ（既定 HMIS=10）
    int    amg_relax;    // BoomerAMG緩和タイプ（既定 l1-Jacobi=18、GPU向き）
    int    amg_interp;   // BoomerAMG補間タイプ（既定 ext+i=6）
    int    amg_levels;   // 最大AMGレベル数（既定 25）

    void init(const PartitionInfo& part, int n_local_cells, int stencil_width);
    void update_matrix(const double* stencil_9pt, const double* rho_Cv, double dt,
                       int n_cells, cudaStream_t stream);
    int  solve(double* Te_out, const double* Te_in, cudaStream_t stream);
    void destroy();
};
#endif
```

> **データフロー（Hypre パス）**：
> 1. C1 カーネル：`D_eff` 計算（STS パスと共通）
> 2. C2 カーネル：Kershaw 9点ステンシル係数計算（STS パスと共通）
> 3. `HypreSolver::update_matrix`：ステンシル係数 → `HYPRE_IJMatrixSetValues`（デバイスメモリ上）
>    - 対角に \(C_v / \Delta t\) を加算（質量行列項、C_v = ρc_v）
>    - スパーシティパターンは初回のみ `HYPRE_IJMatrixSetRowSizes` → 以後は値のみ更新
> 4. `HypreSolver::solve`：BoomerAMG + PCG（デバイスメモリ上で完結）
> 5. 解 \(T_e^{n+1}\) を State に書き戻し → U2 + EOS forward（H13）で ee, Pe, Cv を再同期（§4.4 Post-conduction 契約）
>
> **性能特性**：
> - AMG setup：\(O(N)\) work、大きな定数（粗視化・補間構築）。毎ステップ実行（\(D_{eff}\) 変化のため）
> - PCG solve：\(O(N)\) per iteration × 5–20 iterations（典型）
> - STS との損益分岐点：STS ステージ数 \(s \gtrsim 15\text{–}20\) で Hypre が有利
> - 主な利点は Δt 制約の除去（伝導CFL free）による総ステップ数の削減

#### 4.4.3 Hydro 安定化・エネルギー制御機能

以下の機能はすべて `src/hydro/hydro_1d.cu` に実装。各機能は Config フラグで独立に有効/無効化可能。「既定」は namelist で
指定しないときの値（2026-09-29 の監査で実装に合わせた。GXII の例題デッキ `gxii_solid_1D_fld.py` などは
`av_type="csw"`・`compatible_energy=True`・`odd_even_damping_C=1.0`・`ee_odd_even_C=0.5`・`deposit_smooth_passes=3` を与える）。

| 機能 | ファイル | Config パラメータ | 既定 |
|------|---------|-----------------|---------|
| **CSW 人工粘性** | `artificial_viscosity.cu` | `av_type="csw"` | 1D の既定（`av_type` 未指定なら builder が `"csw"`、2026-08-03） |
| **VNR 人工粘性** | `artificial_viscosity.cu` | `av_type="vnr"`, `av_linear`, `av_quadratic` | 明示すれば使える |
| **Adaptive AV gate** | `shock_tracker.cu`, `adaptive_av_gate.cu`, `artificial_viscosity.cu` | `adaptive_av.enabled` | OFF (診断/改善用、VNR 専用) |
| **Riemann 人工粘性** | `artificial_viscosity.cu` | `av_type="riemann"` / `"riemann_compatible"` | OFF |
| **1D mesh motion** | `hydro_1d.cu` | `mesh.motion="lagrangian"` | ON: pure Lagrangian (the solution-adaptive 1D ALE, `numerics.ale1d`, was retired on 2026-10-02; `retired/ale_1d/`) |
| **Odd-even 圧力フィルタ** | `hydro_1d.cu` | `odd_even_damping_C` | OFF（既定 0。GXII デッキは 1.0） |
| **電子 odd-even ダンピング** | `hydro_1d.cu` | `ee_odd_even_C` | OFF（既定 0。GXII デッキは 0.5） |
| **Compatible-energy 2T** | `hydro_1d.cu` | `compatible_energy` | OFF（既定。GXII デッキは True） |
| **高波数速度ダンパー** | `hydro_1d.cu` | `hk_velocity_damper_C` | OFF (フロント近接) |
| **イオン人工熱伝導** | `hydro_1d.cu` | `ion_art_heat_C` | OFF (Pe支配で無効) |

**Compatible-energy**: 正確な ΔIE = -ΔKE をノード力仕事分解で計算。Odd-even ダンピング力を compatible work に含み、separate heat_oe を除去。legacy PdV 散逸を除去するため、単独では振動が悪化。明示的安定化と組み合わせが必要。

**Riemann AV**: 非線形インピーダンス Z_eff = ρ(cs + α Δu⁺) による音響リーマン解。minmod 再構成。VNR のセンサー/リミッター不使用。現状では ICF 爆縮に対して散逸不足。

**Adaptive AV gate**: 1D_SPH + VNR 専用。hydro step 先頭で base VNR probe から leading shock cluster を同定し、`State.adaptive_av_gate` の履歴付き gate で `C1/C2/heat_C/Cpsv/cbulk` をセルごとに補間する。tracker は `State.adaptive_av_r0`、前回 shock 半径/速度、bounce latch を保持し、predictor/corrector 内では同じ gate field を固定して使う。`Cpsv` は Stage 1 では既存 post-shock nodal damping の per-cell 係数として渡し、compatible-energy rewrite は別段階の対象。

**高波数速度ダンパー**: 3ノード線形フィット → 残差 dv → 保存的ペアインパルス → KE→ion heat。25セル前方バッファガード付き。アブレーション面近接で多段衝撃波を誘発するため無効化。

**イオン人工熱伝導**: 衝撃波制限 κ = C_H × S_comp × S_contact × β × cv_i。二次AVに連動。Pe が全圧力の 73% を占め、Te≈Ti のため、イオンのみの熱伝導は間違った圧力成分に作用。

#### 4.4.4 レーザーデポジットスムージング

`src/laser/deposit_transfer.cu` に実装。保存的質量重み付き Laplacian フィルタを laser 沈着パワーに適用。レイの離散的サンプリングによるステアケース状沈着パターンを平滑化し、多段衝撃波の発生を防止。

| Config | 値 |
|--------|-----|
| `deposit_smooth_passes` | 0（既定 = 平滑化なし。1D の例題デッキの一部は 3） |
| `deposit_smooth_alpha` | 0.25 |

---

### 4.5 radiation/

> **【現行の輻射モデル】** 採用モデルは決定論の **FLD（`mode="multigroup_diffusion"`, NUMERICS §6.7）** と **\(S_N\)（`mode="sn_transport"`, NUMERICS §6.8）** の 2 つ。構成：`Rad::FLD1D`/`Rad::FLD2DRZ`（`fld_1d_gpu`/`fld_2d_rz_gpu`, `driver_fld_energy`；線形 solver は cuSPARSE tridiag (1D) / AmgX-CG・cuSPARSE CG variants (2D)）と `Rad::SNTransport1D`/`Rad::SNTransport2DRZ`（`sn_transport_1d_gpu`/`sn_transport_2d_gpu`, `sn_dsa`, `sn_material_newton`；加速は DSA + RKL2/AMGX）。driver からの入口は `Rad::RadiationStep`。退役したモンテカルロ輻射（IMC・DDMC・ランダムウォーク・HOLO・difference 定式化：`Rad::IMC`・`Rad::DDMC`・`Rad::Diffusion*`・`Rad::Tally`・PhotonPool 等）は 2026-09-29 にビルドから外し、コード・試験・デッキを `retired/radiation_monte_carlo/` へ、本節にあったその設計記述（モジュール・`CellRadiationCoeffs`・旧トップレベル関数・カーネル起動仕様・分散低減・ハイブリッド輸送）を `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。

**責務**：1D_SPH/2D_RZ multigroup **FLD**（既定）と 1D_SPH/2D_RZ pure **\(S_N\)**。群構造と Planck 分率、不透明度の評価、物質との結合（Newton）、沈着 `rad_dep`・放出 `rad_emit` の publish。

- `Rad::RadiationStep`（`src/radiation/radiation_step.hpp`, `radiation_step.cpp`）：driver が輻射演算子 \(\mathcal{R}(\Delta t)\) として呼ぶ入口
  （2026-09-29 まで `radiation::IMC::transport_step` がこの役を兼ねていた）。輻射が無効・\(\Delta t \le 0\)・セルや材料が無いときは
  何もしない。2D_RZ では境界の組（内側 reflect、外側 vacuum/reflect、z 面 vacuum/reflect/marshak）を検査する。群境界は
  `group_bounds_eV`、無ければ `compute_T_range_eV` の対数等分。`PlanckTable` は群境界・温度範囲・`compute_N_T` が変わったときだけ
  作り直す（run の最初の輻射ステップで作る）。mode と次元で `advance_radiation_step_fld_1d` / `_fld_2d_rz` / `_sn_1d` /
  `_sn_2d_rz` を呼ぶ。最大値原理の超過（NUMERICS §11.8）は driver が輻射の後に測って `set_last_overshoot_metrics` で渡し、
  history の `radiation/overshoot_count`・`radiation/overshoot_max` と `Numerics.safety.overshoot_warn` の判定に使う。

- `Rad::Groups`：群境界（eV）、代表値、Planck fraction b_g(T)（計算/テーブル）
  - GPU側問い合わせ：`__device__ double planck_fraction(int g, double T, const PlanckTable* table)`
  - テーブルは温度グリッド（既定 200 点、log-uniform、0.01–100.0 eV）で計算し device メモリに保持（SPECIFICATION §6.4.5 既定）
  - 実行中は \(\ln T\) の 3 次 Hermite 多項式で累積分率 \(C_g\)（または裾の和 \(D_g=1-C_g\)）の対数を補間し、\(b_g\) を隣り合う値の差として得る
    （和は構成上 1。2026-09-23 まで \(b_g\) を \(T\) の線形で補間していた。NUMERICS §6.1）
- `Rad::GroupStructure`（`src/radiation/group_structure.hpp`, `group_structure.cu`）：
  optional hard-X-ray group-boundary repacking and host-side opacity-table resampling.
  It depends on `materials::IonmixOpacityData` and preserves the existing group count.
- `Rad::FLD1D`（`src/radiation/fld_1d_gpu.cuh`, `fld_1d_gpu.cu`,
  `nlte_coeffs.cu`）：
  `Radiation.mode="multigroup_diffusion"` 専用の 1D_SPH multigroup flux-limited diffusion 経路。
  cell×group の `rad_E` と persistent
  `rad_E_old` を有限体積 backward Euler で更新する。群ごとの三重対角系は
  cuSPARSE `gtsvStridedBatch` で GPU 上に solve し、HYDRA-aligned Fleck linearization
  (`f·σ^PA` total removal diagonal、`f·η + (1-f)·c·σ^PA·E^n` RHS) で stiff な
  物質-放射結合を放射線形系へ implicit に組み込む（NUMERICS §6.7、FLD-FIX-1）。
  物質結合は GPU Newton で `Te`, `ee`, `Pe`, `rad_dep`, `rad_emit` を更新し、
  σ^PA を吸収・σ^PE を放射に分離する PA/PE-consistent residual を用いる。
  TMAT/table EOS が利用可能な場合は `materials::DeviceEOSTableView` 経由で
  `e_e(ρ,T)` と `c_{v,e}(ρ,T)` を Newton iteration 内で評価し、収束後の
  `ee`/`Pe` を同じテーブルから書き戻す（電子 EOS device view が無い場合は
  定比熱 + ideal-gas 圧力へ fallback）。
  Persistent state は `rad_E_old`, `fld_sigma_a`, `fld_sigma_pe`,
  `fld_sigma_R`, `fld_eta`, `fld_nlte_f_work`, `fld_nlte_sigma_eff_work`,
  `fld_D_cell`, `fld_lower/diag/upper/rhs`, `fld_Te_old` を使う。
- 1D の多材料の係数（2026-09-24、FLD と \(S_N\) が共有）：
  `src/radiation/multimat_opacity_1d.cuh`（材料ごとの不透明度の記述
  `MatOpacityDesc` と、セルの材料から部分密度で不透明度を混合する
  `eval_opacity_multimat_kernel`、\(S_N\) の物理散乱 `eval_scattering_multimat_kernel`。
  無名名前空間なので取り込む翻訳単位ごとに実体を持つ）と
  `src/radiation/material_electron_eos_1d.hpp`/`.cu`（材料ごとの電子 EOS 表の device
  配列とセルごとの選択 `cell_electron_table_selector_1d`、物質結合の Newton が使う）。
- `Rad::FLD2DRZ`（`src/radiation/fld_2d_rz_gpu.cuh`, `fld_2d_rz_gpu.cu`,
  `src/radiation/amgx_solver.hpp`, `amgx_solver.cpp`）：
  `Radiation.mode="multigroup_diffusion"` かつ `Main.dimension="2D_RZ"` の
  multigroup FLD 経路。R/Z 4-face finite-volume CSR system を群ごとに組み立て、
  AmgX が link されている場合は `linear_solver_2d="amgx_cg"`、未検出 build では
  WARNING を出して `cusparse_cg_jacobi` debug fallback で solve する。
  `cusparse_cg_zline` は radial line ごとの z-tridiagonal を cuSPARSE `cusparseDgtsv2`
  で解く z-line block-Jacobi preconditioned CG option である。R軸は reflect、
  outer R は vacuum、Z端は `z_boundary` / `boundary.z` で vacuum、reflect、
  Marshak、または grey one-group state_supply Dirichlet を選ぶ。state_supply
  Dirichlet の供給温度は hydro z-face state-supply config から取得し、step/cumulative
  reservoir tally は `State::fld_state_supply_*` に保持する。
  HYDRA-aligned Fleck linearization、PA/PE-consistent matter Newton、TMAT-aware
  electron EOS 連携は 1D_SPH と同一仕様（NUMERICS §6.7、FLD-FIX-1）。
  `State.fld_fleck[n_cells]` は output-only の per-cell Fleck factor diagnostic として
  matter update で publish する。
- `Rad::SNTransport1D`（`src/radiation/sn_transport_1d_gpu.cuh`,
  `sn_transport_1d_gpu.cu`, `sn_dsa_1d_gpu.cu`,
  `sn_material_newton_gpu.cu`; 線形不連続法 `sn_ld_1d_gpu.{cuh,cu}`,
  共有部 `sn_transport_1d_internal.hpp`（不透明度評価・角度求積・状態配列）と
  `sn_electron_eos.cuh`（電子 EOS の尾部・逆算・比熱）、吸収率密度の GMRES の Givens 回転の
  `rounded_hypot.cuh`（device の正しく丸めた \(\sqrt{a^2+b^2}\)））：
  `Radiation.mode="sn_transport"` 専用の 1D_SPH pure \(S_N\) production 経路。
  `spatial_scheme="linear_discontinuous"`（1D の既定、2026-09-25）は
  `advance_radiation_step_sn_1d` が Marshak 入射・体積源の設定後に
  `sn_ld::advance_step` へ渡し、節点強度の掃引（群ごとに 1 warp。セル行列の逆行列は
  Newton 反復ごとに前計算）、整合 P1 系の block Thomas（Newton 反復ごとに 1 回の組み立てと
  ブロック消去、対角ブロックは部分ピボットの Gauss–Jordan で逆行列にして保持、右辺ごとの代入）、節点温度の Newton と
  吸収率密度の GMRES、節点物質更新（線形化と物質更新は節点ごとに 1 warp、群の Planck 項を
  レーンで並列、EOS の逆算は warp 並列の二分法）をすべて GPU 上で行う（host は GMRES の
  Hessenberg 小行列と収束判定のみ。NUMERICS §6.8.4）。以下は `"linear_characteristic"` の記述。
  group-parallel/angle-serial
  spherical sweep、group-batched DSA tridiagonal solve、GPU Newton material
  coupling を CUDA 上で実行する。
  Pure SN は Fleck linearization を bypass し（\(f=1\)、Fleck-derived effective
  scattering = 0）、raw \(\sigma^{PA}\) を sweep 吸収・raw \(\sigma^{PE}\) を
  emission に渡す（NUMERICS §6.8、Cut-FIX-4）。
  1D_SPH sweep は Morel-Adams angular-edge redistribution のため group 内で
  ordinate order を保持し、pair-parallel angle sweep は debug diagnostic のみ。
  各 cell では Cut-FIX-3 の metric scaling \(a_{c,n+1/2}=(\Delta A/w_n)\alpha_{n+1/2}\)
  により球面 LTE fixed-point cancellation identity を保つ（NUMERICS §6.8.1）。
  1D_SPH DSA は outer-vacuum Robin leakage と \(r=0\) scalar-parity no-flux
  boundary terms を transport boundary convention と一貫させて組み立てる。
  Material Newton は conservative active-set + face-flux + donor-theta +
  AP face-blend closure に固定され、σ^PA 吸収・σ^PE 放射の PA/PE split residual を用い、
  TMAT/table EOS が利用可能な場合は `materials::DeviceEOSTableView` 経由で
  `e_e(ρ,T)` と `c_{v,e}(ρ,T)` を Newton iteration 内で評価する（Cut-FIX-5；
  EOS view が null のときは定比熱 + ideal-gas fallback）。
  Persistent state は `rad_E_old`, `sn_psi_scratch`, `sn_dsa_*`,
  `sn_phi_*`, `sn_eta`, `sn_sigma_a` (= raw σ^PA), `sn_sigma_pe` (= raw σ^PE),
  `sn_sigma_s` (Fleck bypass scattering; AP face_blend では \(\sigma_R\) として再利用),
  face-flux mode 用の
  `sn_face_flux_raw[(n_cells+1)×G]`, donor-theta limiter 用の
  `sn_face_flux_limited[(n_cells+1)×G]`, `sn_stream_theta[n_cells×G]`,
  AP face_blend 用の `sn_face_flux_diff[(n_cells+1)×G]`,
  `sn_face_alpha[(n_cells+1)×G]`, `sn_E_star_flux[n_cells×G]` を使い、deterministic
  `rad_dep`/`rad_emit` を publish する。
- `Rad::SNTransport2DRZ`（`src/radiation/sn_transport_2d_gpu.cuh`,
  `sn_transport_2d_gpu.cu`, `sn_transport_gpu.cu`）：
  `Radiation.mode="sn_transport"` かつ `Main.dimension="2D_RZ"` の pure \(S_N\)
  経路。product level-symmetric quadrature、R軸 reflect parity、outer R vacuum、
  configurable Z vacuum/reflect boundary、および GPU 2D DSA Jacobi correction を使う。
  Fleck bypass と TMAT-aware electron EOS 連携は 1D_SPH と共通だが、material
  coupling closure は 2D Concern 1-4 実装まで legacy non-implicit closure を内部で使う。
  AP face-blend 後に `State.sn_tau_R`, `State.sn_reduced_flux`, `State.sn_ap_alpha`
  へ per-cell transition diagnostics を publish する。
  Cut-2 では機能検証優先であり、GXII-scale 2D \(S_N\) 性能最適化は別タスクである。

**PlanckTable**（`src/radiation/planck_table.cuh`）：host のクラス `PlanckTable`（`build(groups, n_T, T_min_eV, T_max_eV)`、
picket-fence 用の `build_constant_fractions`）が温度格子とその対数、群ごとの \(b_g\)・累積分率 \(C_g\)・裾の和 \(D_g\)、
\(\ln C_g\)・\(\ln D_g\) とその \(\ln T\) 微分を device に置き、kernel へは `PlanckTableDeviceView`（`interpolate_b(g, T)`、
多群を同じ温度で引く `locate_b(T)`）として渡す。表の範囲外の温度は端の値に丸める（host 側の評価は最初の 1 回だけ WARNING を出す）。
`planck_host_rounding.cuh` の `planck_fraction_host_rounding` は host の `interpolate_b_host` を device で同じ丸めで評価する
（log・exp は `core::glibc_libm`、ビット一致。S\(_N\) 1D の Marshak 境界が使い、範囲外の WARNING は host の
`warn_if_outside_range` が出す。2026-10-02）。
`planck_fraction.method="tabulate"` は受理して保存するが使われず、計算した分率に戻す（WARNING。SPECIFICATION §6.4.5）。


#### 4.5.1 Radiation トップレベル関数シグネチャ

```cpp
// 輻射演算子 R(Δt)（NUMERICS §2.1、§6.7、§6.8）。driver が 1 ステップに 1 回（Radiation.imc.two_stage では
// 半ステップずつ 2 回）呼ぶ。物質の更新（Te, ee, Pe）と rad_dep / rad_emit の publish は各ソルバーの中で行う。
class RadiationStep {
 public:
  void step(core::State& state, const core::Config& cfg, double dt,
            const parallel::PartitionInfo& part = {},    // MPI の分割（2D_RZ の r-slab）
            parallel::CommBuffers* bufs = nullptr,       // n_ranks > 1 では必須
            double drive_time_s = NaN);                  // 1D の時間依存の境界駆動を引く時刻（NaN = state.t）
  void set_last_overshoot_metrics(std::int64_t count, double max_ratio);  // step() が 0 に戻す
  std::int64_t last_overshoot_count() const;             // history radiation/overshoot_count
  double last_overshoot_max() const;                     // history radiation/overshoot_max
};
```

#### 4.5.2 Radiation GPU カーネル起動仕様

現行の FLD・\(S_N\) のカーネルと起動設定は CUDA_KERNELS §6.7（FLD）・§6.8（\(S_N\)）を参照。
（本節にあった IMC/DDMC のカーネル起動仕様は `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。）

#### 4.5.3 放射輸送の分散低減・加速機能

退役（モンテカルロ輻射の分散低減・加速機能。記述は `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した）。

#### 4.5.4 ハイブリッド輸送（研究ブランチ、本番 OFF）

退役（IMC + DDMC/PGRW + 決定論拡散の 3 モード構成。記述は `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した）。

---

### 4.6 laser/
**責務**：レーザービーム追跡と沈着

- `Laser::Beams`：ビーム定義（方向、焦点、F値、D/R、プロファイルテーブル、波形テーブル）
  - 初期化時にプロファイル関数をサンプリングしてテーブル化（Python呼び出し禁止方針に準拠）
  - `spot`（非推奨）が指定された場合は内部で `profile` に自動変換
  - `radial_absorption_1d` では各ビームの `power(t)` のみを合算し、方向・F値・焦点・D/R・プロファイル・レイ本数は吸収分布に使わない
- `Laser::CoordinateTransform`：座標変換
  - 1D_SPH：1D球座標 \(r\) → ビームローカルRZ座標 \((R,Z)\) への射影（球対称仮定 \(Q(\sqrt{R^2+Z^2})\)）
  - 1D_SPH：ビームローカルRZ → 1D球座標 \(r\) への逆射影（吸収エネルギーの転写用）
  - ビーム方向ベクトルからRZ座標系の原点・軸を構築
  - 2D_RZ：3D Lab座標 → LaserMesh座標の変換 \((R,Z)=(\sqrt{x^2+y^2},z)\)（NUMERICS §5.3.4）
  - 2D_RZ：ビーム軸直交平面の正規直交基底 \((\hat{\mathbf{u}},\hat{\mathbf{w}},\hat{\mathbf{d}})\) 構築
- `Laser::Mesh`：**1つの**LaserMesh（2D RZ構造格子）を生成
  - 1D_SPH：代表ビーム軸をZ軸とするビームローカル2D RZ格子
  - 2D_RZ：**流体対称軸に沿う** 2D RZ格子（ビームローカルではない、NUMERICS §5.7.1）
  - 1D_SPH：\(\rho(r), T_e(r), \bar Z(r)\) をRZメッシュ上に \(\rho(\sqrt{R^2+Z^2})\) としてマッピング
  - 1D_SPH：ray trace 用に midplane から radial side arrays
    \((r,\hat n,\hat n_{raw},A_{smooth},d\hat n/dr)\) も併せて保持
  - 2D_RZ：2D_RZ HydroMeshの \(\rho(R,Z), T_e(R,Z), \bar Z(R,Z)\) を直接マッピング（NUMERICS §5.7.3 (b)）
  - `critical_clip`：節点の \(\hat n\) を `critical_margin` で頭打ちにする（値の上限。1D の格子は臨界面より奥も含む）
  - 1D は写像のたびに節点を作り直す：臨界面の近くを細かく、内外へ幾何級数で粗くする graded 配置（NUMERICS §5.7.2）。
    ゴーストコロナが外半径を超えると外半径を広げて作り直す（2026-09-29）。2D は初期化時の一様格子。
    （密度勾配で伸縮する格子は無い — `stretch_method`・`min_ratio` はどのメッシュも読まず、指定すると警告）
  - 節点（node-centered）に \(\hat n = n_e/n_{crit}\), \(T_e\), \(\bar Z\), \(\nabla\hat n\) を保持
  - 節点での中心差分による密度勾配 \(\nabla(n_e/n_{crit})\) の計算
- `Laser::Cbet`（`cbet.cu/.cuh`、`cbet_stage_gpu.cu/.cuh`、`cbet_stage_host.cpp/.hpp`、v1 = 1D_SPH `raytrace_2d` + 2D_RZ `raytrace_3d` opt-in）：Marozas 型保存的 pairwise CBET
  - trace kernel の record モード（`template<bool kCbetRecord>`、OFF 実体化は従来と構造同一）が ray 毎のセル横断記録を生成
  - 1D の毎 step の前処理と後処理は device 上（`cbet_stage_gpu.cu`、FMA 縮約なしで旧 host ループと同じ演算順）: セルの有効質量数とプラズマ量（chi 前因子・音速・流速・波数・体積・有効セルの印）を device の State から作り、解の後の交換量の地図（セルごとの |dQ| 総和と内向き群の dQ）と作用量の閉じの検査、port_section のポートごとの出射パワーと高速電子の捕捉の集計を device で行う。step 中に host へ戻るのは検査の旗と捕捉の和だけで、地図とポートごとの量は snapshot・checkpoint の直前に `sync_laser_snapshot_fields` で State へ写す。旧 host ループは試験の参照として `cbet_stage_host.cpp` と `cbet_stage_cell_fields` に残る
  - CbetWorkspace（grow-only device 常駐）上で決定論的固定点反復（tally → 反対称交換+donor cap → IB/2·CBET·IB/2 propagate）
  - 沈着・未吸収は per-ray 行 + 固定順 reduction でビーム毎に集計し、既存の deposit 再配分・skip cache 経路へ接続（NUMERICS §5.10）
  - 2D_RZ CBET は theta-group ごとの record-mode trace → joint exchange solve → per-group LaserMesh node deposit を生成し、既存の 2D transfer path に接続する。Workspace singleton は 1D と共有し、recorder template 実体化は defining TU に閉じる；nvcc+RDC では header 側 template declaration を増やすと OFF path まで再実体化されるため、新規宣言は wrapper/header isolation で分離する。
- `Laser::HotElectron1D`（`hot_electron_1d.cuh/.cpp`、`hot_electron_1d_gpu.cu`、`hot_e_transport_1d_gpu.cu/.cuh`）：1D hot-electron preheat: capture reduction, cone quadrature, chord walkers, CSDA pipelines。本番の経路は device: `hot_e_transport_1d_gpu.cu` がトレースの捕獲行の収集・源への集約・cone の chord の列挙と `radial` の行進・セルごとの診断と dt 上限を行い（NUMERICS §5.11）、cone の chord は `hot_electron_1d_gpu.cu` の `cone_chords_device` が進める。step 中に host へ戻るのはチャンネルごとの数値とセルの \(Q\)・\(\varepsilon_{cum}\)。host の実装と `deposit_hot_electrons_cone_1d_device` は試験の参照
- `Laser::HotEEtaModel`（`hot_e_eta_model.cpp`、`hot_e_inputs_gpu.cu/.cuh`）：\(\eta(t)\) の閾値モデルと緩和（NUMERICS §5.11.3）
  - モデルの入力は 1D の毎 step device 上で作る（`hot_e_inputs_gpu.cu`、FMA 縮約なしで旧 host 計算と同じ演算順）: セルの \(n_e\)・\(n_e/n_c\)・中心半径、チャンネルごとの評価面（\(n_e\) が \(n_c\) の指定割合を横切る最も外側の位置）とそこでの \(T_e\)・最寄りの殻・重み付き当てはめの密度尺度、port_section では前 step の device の位相空間表から参照ビームの角度分布、ポート配置の照度指標と共通波駆動（snapshot 用の天球の地図を含む）。step 中に host へ戻るのはチャンネルごとの数値だけで（\(\eta\) の緩和の更新は host のスカラー計算）、天球の地図は snapshot・checkpoint の直前に `sync_laser_snapshot_fields` で State へ写す。device の超越関数（exp・log・acos・atan2・sin・cos）の末位の差の分だけ旧 host 計算と異なりうる
- `Laser::PortSection`（`port_geometry`、`port_section_chi`、`port_section_overlap`、`port_section_s1_gpu`、`port_section_s1_host`、`sector_phase_space`、`sector_adapter`）：
  実ポート配置の多ビーム CBET（`cbet.geometry_mode="port_section"`、NUMERICS §5.10.8）
  - 位相空間表（参照ビームの ray path と殻の交差、NUMERICS §5.10.8 S1）は毎 step `port_section_s1_gpu.cu` が CbetWorkspace の ray 記録から device 上で組み、chi の構築（`port_section_chi.cu`、`ChiBuildInput::device_table`）は device の表をそのまま読む。step 中に host へ戻るのは監査と除外台帳の数値だけで、snapshot 用の強度地図（`State::ps_ray_map`）は driver が snapshot・checkpoint を書く直前に `sync_laser_snapshot_fields` で State へ写す
  - host の `sector_adapter::build_ray_paths` と `sector_ps::build_table` は device の表と比べる試験の参照（`port_section_s1_host.cpp`）に、1 殻分の表の host への写し（`host_table_for_shell`）は高速電子の照度指標と共通波駆動の host 関数（`port_section_overlap`）を device の入力と比べる試験に使う
- `Laser::HotElectron2D`（`hot_electron_2d.cuh/.cpp`）：2D RZ hot-electron transport: topology-agnostic MeshView2D, revolved-face chord walker, 3D band quadrature, capture reduction, host cone pipeline（reference path）
- `Laser::HotElectron2DGpu`（`hot_electron_2d_gpu.cuh/.cu`）：device chord pipeline（1 thread/chord, deterministic host fold）+ scratch-pooled staging

**LaserMesh 構造体**（NUMERICS §5.7 準拠）：

```cpp
// レーザーレイトレース用 2D RZ 構造格子（Laser モジュール内部管理）
// NUMERICS §5.7.1–§5.7.5 準拠
//
// LaserMesh はリクティリニア（テンソル積）グリッドである。
// ノード座標は 1D 配列に因数分解される: node_R[nr+1], node_Z[nz+1]
// 2D ノード (i,j) の物理位置は (node_R[i], node_Z[j]) で決定される。
// ALE による一般四辺形メッシュ（HydroMesh / Mesh §4.2.2）とは異なり、
// LaserMesh は常に直交格子を維持する（2D_RZ の双線形補間と
// 1D_SPH の radial side-array 抽出の基盤）。
struct LaserMesh {
    // --- 格子構造 ---
    int     nr, nz;                 // セル数（既定 128×256）
    int     n_nodes_r, n_nodes_z;   // 節点数 = (nr+1), (nz+1)

    // --- 節点座標（deviceメモリ、ストレッチ格子対応）---
    // テンソル積構造：1D配列 node_R × node_Z で2Dグリッドを定義
    double* node_R;                 // [n_nodes_r] R方向節点座標 [cm]
    double* node_Z;                 // [n_nodes_z] Z方向節点座標 [cm]

    // --- 節点物理量（node-centered、deviceメモリ）---
    // メモリレイアウト：row-major [i * n_nodes_z + j]（i=R方向, j=Z方向）
    double* n_e_hat;                // [(nr+1)×(nz+1)] 正規化電子密度 n_e/n_crit（臨界でクリップ）[dimensionless]
    double* n_e_hat_raw;            // [(nr+1)×(nz+1)] クリップ前の n_e/n_crit [dimensionless]
    double* T_e;                    // [(nr+1)×(nz+1)] 電子温度 [eV]
    double* Zbar;                   // [(nr+1)×(nz+1)] 平均電荷数 [dimensionless]
    double* smooth_kappa_factor;    // [(nr+1)×(nz+1)] 逆制動輻射の smooth 係数（事前計算）

    // --- 密度勾配（node-centered、中心差分で計算）---
    double* grad_n_hat_R;           // [(nr+1)×(nz+1)] ∂(n_e/n_crit)/∂R [1/cm]
    double* grad_n_hat_Z;           // [(nr+1)×(nz+1)] ∂(n_e/n_crit)/∂Z [1/cm]

    // --- 1D の光線追跡の径方向プロファイル [radial_n_nodes ≤ radial_capacity] ---
    // 2026-09-24 以降は map_from_hydro_1d が hydro セルに結び付けて作る節点
    // （面・臨界の対・プロファイルの節点、laser_mesh_bodies::build_trace_profile_nodes_1d）。
    // それ以前は 2D レーザー格子の軸の列（Z=0 上の [nr+1]）だった。
    double* radial_node_r;          // 節点の半径 r [cm]
    double* radial_n_hat;           // クリップ後の n̂
    double* radial_n_hat_raw;       // クリップ前の n̂
    double* radial_smooth_kappa;    // smooth 係数（compute_smooth_kappa が埋める）
    double* radial_T_e;             // 電子温度 [eV]
    double* radial_Zbar;            // 平均電荷数
    double* radial_dn_dr;           // d(n̂)/dr [1/cm]
    int     radial_n_nodes;         // 使用中の節点数
    int     radial_capacity;        // 確保済みの節点数
    bool    trace_profile_1d;       // map_from_hydro_1d が設定（プロファイルを使う）
    TraceProfileMap1D trace_profile_map; // 直近の 1D 写像の入力（外表面・臨界のセル、ゴーストの幅・密度など）
    int     geometry_code;          // 写像した流体の Mesh.geometry_1d（0 球、1 円筒、2 平板）

    // --- 沈着配列（node-centered）---
    double* deposit;                // [(nr+1)×(nz+1)] 吸収パワー [erg/s]（NUMERICS §5.5）

    // --- メタ情報 ---
    double  R_max;                  // R方向上限 [cm]
    double  Z_min, Z_max;           // Z方向範囲 [cm]
    double  n_crit;                 // 臨界密度 [1/cm³]
    double  n_hat_margin;           // [dimensionless] 臨界密度クリップ閾値（既定 1-eps_crit=0.9999）（NUMERICS §5.7.1）

    // --- 1D_SPH map device buffers ---
    double* hydro_A_eff_device;     // [hydro_cell_capacity] 1D_SPH map用 A_eff scratch
    uint8_t* hydro_cell_is_void_device; // [hydro_cell_capacity] 1D_SPH map用 void flag scratch
    int     hydro_cell_capacity;    // hydro scratch capacity [cells]

    // （1D の写像は step 間平滑化をしないので状態を持たない。2026-09-24 に
    //   prev_n_hat_device と State::laser_nhat_smoothing_{r,n_hat} を撤去、NUMERICS §5.7.4）

    // --- ゴーストコロナ設定（1D_SPH、NUMERICS §5.7.5）---
    bool    ghost_corona_enabled;           // ゴーストコロナ有効フラグ
    int     ghost_n_out;                    // ゴーストセル数（既定 12）
    double  ghost_ne_min_frac;              // ゴースト密度下限比 n̂_min [dimensionless]
    double  ghost_ne_max_frac;              // ゴースト密度上限比 n̂_max [dimensionless]
    double  ghost_Te_min_eV;                // ゴースト電子温度下限 [eV]
    double  ghost_zbar_min;                 // ゴースト Z̄ 下限 [dimensionless]
    double  ghost_zbar_max;                 // ゴースト Z̄ 上限 [dimensionless]
    int     ghost_handoff_cells;            // ハンドオフステンシル深さ（既定 4）
    double  ghost_handoff_decay;            // ハンドオフ指数関数減衰長（既定 1.5）

    // --- ブローオフ遷移モデル（NUMERICS §5.7.5.4）---
    bool    ghost_transition_enabled;       // 遷移モデル有効フラグ
    double  ghost_transition_resolved_nhat; // resolved-corona 判定閾値 n̂ [dimensionless]（既定 0.9。ゴーストの減衰にも使う）
    int     ghost_transition_resolved_cells;// ゴーストが消え、ハンドオフが復帰する亜臨界セル数（既定 3）
    double  ghost_transition_density_exponent; // 密度バイアス指数 α_ρ（既定 1.0）
    double  last_ghost_transition_blend;    // 直近のブレンド係数 β（診断用、毎ステップ更新）
    int     last_ghost_transition_resolved_cells; // 直近の resolved セル数（診断用、毎ステップ更新）
    double  last_ghost_width;               // 直近の 1D マッピングのゴースト幅 W [cm]（0＝なし。診断用、NUMERICS §5.7.5.2）
    double  last_trace_unabsorbed_power;    // raytrace/skip 直後の未吸収パワー [erg/s]（診断用）
    double  last_transfer_blocked_power;    // transfer で受け皿がなく捨てたパワー [erg/s]（診断用）
    double  last_unabsorbed_power;          // transfer 後の最終未吸収パワー [erg/s]（診断用）
    double  last_commanded_energy;          // 当該 step の入射エネルギー [erg]（診断用）
    int64_t last_tail_closure_count;        // tail closure で終了した ray 数 [count]（診断用）
    double  last_tail_closure_absorbed_power; // tail closure で吸収されたパワー [erg/s]（診断用）
    int64_t last_critical_surface_hit_count; // 臨界面で打ち切られた ray 数 [count]（診断用）

    // device function: 位置 (R,Z) [cm] から補間済み物理量を取得（双線形補間）
    __device__ double interp_n_hat(double R, double Z) const;  // 戻り値: n̂ [dimensionless]
    __device__ void   interp_grad(double R, double Z,          // 出力: ∂n̂/∂R, ∂n̂/∂Z [1/cm]
                                  double& dndR, double& dndZ) const;

    // 格子外判定
    __device__ bool is_outside(double R, double Z) const;
};
```

> **所有権**：`LaserMesh` は Laser モジュールが `init()` 時に確保し、ステップ毎に
> 物理量を HydroMesh から再マッピングする（NUMERICS §5.7.4）。
> State のメンバーではない（§5.2 の注記参照）。
> 2D の格子点配置は初期化時に固定。1D は写像のたびに節点を作り直し（`map_from_hydro_1d`、NUMERICS §5.7.2）、
> 径方向の配列は hydro セルに結び付いた節点（面・臨界の対・プロファイルの節点）を持つ。上の構造体は主要な欄だけを示し、
> 容量の管理（`*_capacity`）、ステップごとの作業領域（`scratch_*`）、CBET の診断（`last_cbet_*`）などは
> `src/laser/laser_mesh.cuh` を参照。

- `Laser::RayInit`：レイ初期条件の生成
  - 1D_SPH：F値と集光位置からレイの初期位置（R方向1D配列）・方向を計算（NUMERICS §5.6.3 (a)）。輪の重み・輪のパワー・球の光線は device（`ray_init_1d_gpu.cu/.cuh`）、host の計算は試験の参照（`initialize_rays_1d_host_reference`）
  - 2D_RZ：ビーム軸直交平面上の2D断面配列としてレイを初期化（NUMERICS §5.6.3 (b)）
    - 正規直交基底 \((\hat{\mathbf{u}},\hat{\mathbf{w}})\) 上の格子点座標 → 3D Lab座標への変換
    - 初期方向は3D焦点座標に向かうベクトル
  - ビームプロファイルからレイのパワー重みを計算（1D_SPH：環状面積 \(2\pi R\Delta R\)、2D_RZ：断面面積 \(\Delta u\Delta w\)）
  - 2D_RZ：極角θとパラメータによるビームグループ化（NUMERICS §5.6.4）
- `Laser::RayTrace`：幾何光学（屈折）+ IB吸収
  - 1D_SPH `raytrace_2d`：既定の `integrator="auto"` は特性曲線積分（`ray_trace_characteristic.cuh`、NUMERICS §5.3.6）で、
    臨界半径で反射する（`terminate=False` が既定）。`integrator="leapfrog"` は 2D ベクトル \((R,Z)\) の刻み幅可変の Verlet 行進
    （NUMERICS §5.3.2、臨界で終了。前の trace の各レイのステップ数による最長順の並べ替えと、ステップ上限
    \(\max(20000, 10\,p_{90})\) の \(p_{90}\) は device で求め（CUB の安定な基数ソート、2026-10-02）、
    ステップ数はビームごとの device 配列で次の step へ持ち越す）。円筒・平板の 1D は特性曲線積分だけ
    - 場参照は 2D bilinear ではなく radial side array への 1D linear lookup
    - 勾配は \(d\hat n/dr\) から \((\partial\hat n/\partial R,\partial\hat n/\partial Z)\) を再構成
    - 吸収パワーは Hydro の 1Dセル配列へ直接蓄積（決定論の固定順）
  - 1D_SPH `radial_absorption_1d`：レイ初期化を行わず、全ビームパワー合計を `launch_radial_absorption_1d` へ渡して外側セルから
    内側セルへ動径積分する（256 スレッドで 1024 セルずつ並列に評価。MPI では全 rank が同じ全線を計算し、所有窓で適用 — 2026-09-29）
  - 2D_RZ：**3Dベクトル \((x,y,z)\)** でLeapfrog追跡（NUMERICS §5.3.4）
    - 2D勾配→3D変換：\(\partial\hat n/\partial x = (\partial\hat n/\partial R)(x/R)\) 等
    - R=0特異性処理：\(R < R_{floor}\) で勾配の横方向成分を零とする
  - 任意点での場参照は次元依存（1D_SPH: radial 1D linear、2D_RZ: 2D bilinear）
  - IB吸収は台形公式による光学厚 \(S\) 計算（NUMERICS §5.4準拠）
  - 臨界処理は2段階（`terminate=True` の経路。1D の特性曲線積分の既定は反射）
    - 通常は \(\varepsilon_{crit}\) 面までセグメントを切り詰めて terminate する（NUMERICS §5.2）
    - 臨界近傍では `critical-layer mode` に切り替え、carried \(\kappa\) から再構成した \(A_{entry}\) と \(|\nabla \hat n|\) で解析 tail 光学厚 \(\tau_{tail}\) を計算し、1D_SPH は entry 半径を含む Hydro 1Dセル、2D_RZ は entry 点の bilinear nodes に沈着して終了する（NUMERICS §5.4.4）
- `Laser::DepositMap`：LaserMesh → HydroMesh の写像
  - 吸収パワーの空間分配は次元依存（1D_SPH: 1D cell direct、2D_RZ: 4-node bilinear）
  - 2D_RZ：3D中間位置 \((x,y,z)\) → \((R,Z)=(\sqrt{x^2+y^2},z)\) でLaserMeshセルを特定して分配
  - 1D_SPH：ray trace または `radial_absorption_1d` 中に Hydro の 1Dセル配列へ直接沈着し、ビームの付着を device 上で合算して blocked/ghost handoff・平滑化を device で適用（`deposit_1d_gpu.cu/.cuh`、NUMERICS §5.8.1 (a)。host の `apply_deposit_redistribution_1d` は試験の参照）
  - 1D_SPH の写像のスカラー部（節点配置・臨界面・ゴーストコロナ・例外受け皿・共鳴吸収の入力、NUMERICS §5.7.2 (e)）も device（`laser_map_1d_gpu.cu/.cuh`、`map_from_hydro_1d_device`）。host への流体量の写し（`build_hydro_mirror_1d`）は光線密度の診断だけが使う
  - 2D_RZ：LaserMesh沈着を **直接** 2D_RZ HydroMeshへ双線形補間で分配（NUMERICS §5.8.1 (b)、1D球座標転写は不要）
  - 多ビーム：1D は各ビームをそれぞれのパワーで追跡し、全ビームのキーが一致するときだけ 1 回の追跡を使い回す。
    2D は極角グループごとに追跡してグループ内パワー合計でスケーリング（NUMERICS §5.6.4）
  - エネルギー保存検証（転写前後の差分 ≤ \(10^{-10}\)）
  - `laser_dep` の `ee` 注入は Coupling 側 `inject_laser_source_terms` で実施し、
    \(e_e \leftrightarrow T_e\) クロージャにはセルごとの \(A_{eff},\gamma_{eff}\)（§4.3.3）を用いる

#### 4.6.1 Laser トップレベル関数シグネチャ

```cpp
// Laserフルステップ（Strang splitting L(Δt)）— NUMERICS §5
void laser_step(
    State& state,                       // 流体場（ρ,Te,Zbar読取）+ laser_dep 書込
    LaserMesh& lmesh,                   // レーザーメッシュ（内部管理、物理量を再マッピング）
    const Config::LaserConfig& laser,   // ビーム定義、レイ数
    const CellField& zbar,               // 平均電離度 Z̄（IB吸収計算用、1D=CellField1D / 2D=CellField2D）
    double dt,                          // フルステップ幅 Δt [s]
    double t,                           // 現在時刻 [s]（波形評価用）
    const PartitionInfo& part,          // 並列情報（LaserMesh全rank複製）
    cudaStream_t stream
);
```

---

### 4.6a burn/
**責務**：核燃焼（1D_SPH v1 / 2D_RZ port）— Bosch-Hale 反応率、per-cell 種ネットワーク、
2D 局所沈着および Corman 多群荷電粒子拡散、Li-Petrasso e/i 分配表（NUMERICS §14）

- `tenryu_burn`（STATIC、依存は `tenryu_core` のみ；State/Config 非依存の純関数層）
  - `burn_constants.hpp`：種/反応 enum、質量（proton-mass 単位）、エネルギー分配表、
    Bosch-Hale Table VII 係数 — すべて `__host__ __device__` アクセサ関数
    （RDC下で runtime-index の namespace-scope constexpr 配列は device 不可視のため）
  - `reactivity.cuh`：`bosch_hale_sv(reaction, T_keV)`（床・天井クランプ込み）
  - `network.cuh`：`burn_network_step()` — 凍結温度 RK2 subcycle、決定論 scale-back
    正値性、counts 一次主義の台帳恒等（host/device 共有単一実装）
  - `network_gpu.cu/.cuh`：セル並列 device kernel（host-device identity ctest 済み）
  - `deposition.cuh`：`alpha_rho_lambda()`（Fraley 3d×δ_log、媒質係数 `FraleyRangeMedium`）、
    種 range スケール、`point_sphere_deposited_fraction(u,τ)` 閉形式
  - `field_ions.hpp`：減速の場イオン `FieldIons`（イオンあたりの平均 A・Z・Z²・Z²/A）、
    燃焼在庫と材料の体積分率からのセル毎の組成 `cell_field_ions()`（host）、
    局所 range の媒質係数 `fraley_range_medium()`、Corman/MC カーネル用のセル配列 `FieldIonCells`
  - `partition.hpp/.cpp`（host）/ `partition_device.cuh`：LP 減速積分の初期化時タブレーション
    （\(T_e\) 64 × \(T_i\) 16 × \(n_e\) 16 の log 格子 × 10 slot — 荷電生成物 6 と中性子の弾性反跳 4）、Fraley Eq.4 knob。表の構築は、減速項を
    エネルギーごとに 1 回評価して各軸で使い回し（電子の減速は Ti に、イオンの速度項は
    ne によらない）、生成物 slot × Te の作業を優先度の低いスレッド群で並列に計算する。
    driver は初期化時に `std::shared_future` で背景構築を始め、燃焼の段が最初に表を
    要するときに待つ（局所沈着は燃料が反応しうるまで段を飛ばす）。1 項ずつの定義と
    ビット一致を `tests/burn/test_burn_partition_table.cpp` で検査（2026-09-25）
  - `burn_stage.hpp/.cpp`：`compute_burn_step_1d()` — 燃料域/column 幾何、
    ネットワーク呼び出し、per-cell 沈着/分配、台帳（プレーン配列 in/out、単体テスト可能）。既定では
    `burn_stage_gpu.cu/.cuh` の `compute_burn_step_1d_device_stage` が同じ段を GPU で実行し、host の実装は
    `TENRYU_BURN_HOST_STAGE=1` のときと参照・試験用（host/device の一致は `test_burn_stage_gpu_parity`、rel 1e-12）。
    1D では `compute_burn_step_1d_resident` が device の場と device 常駐の燃焼配列で同じカーネルを実行する
    （局所沈着と拡散・MC の誕生源、2026-10-02、旧経路とのビット一致も `test_burn_stage_gpu_parity`）
  - `burn_inputs_1d_gpu.cu/.cuh`：1D の段と輸送の前後で driver が host で計算していたセルごとの量（セル速度、燃料イオン
    組成と Fraley 係数、\(\epsilon_{cum}\) と中性子数の更新、輸送の電子密度、粒子のあるスロット、輸送後の付与・加熱率・
    陽的源の dt 上限）を device で計算する（-fmad=false、`test_burn_inputs_1d_gpu`）
  - `corman_diffusion.cu/.cuh`：1D の多群荷電粒子拡散（`Burn.scheme="diffusion"`、NUMERICS §14.7）
  - `mc_transport.cu/.cuh`：直線 CSDA の MC α 輸送（`Burn.scheme="mc"`、NUMERICS §14.9。粒子の詰め直しの添字は
    device の排他的接頭和、2026-10-02）
  - `neutron_heating.hpp/.cpp`・`neutron_heating_device.cuh`：中性子の最初の衝突による加熱（`Burn.neutron_heating`、NUMERICS §14.11）
  - `neutron_moments.hpp/.cpp`：Brysk の中性子スペクトルのモーメント（診断、NUMERICS §14.8）
  - `screening.hpp/.cpp`・`screening_device.cuh`：Salpeter / Chugunov–DeWitt の遮蔽（NUMERICS §14.1）
- driver 結線（coupling/ 所有）：`callbacks.burn`（laser 直後・radiation 前）、
  比在庫 `State::burn_n_host` [1/g]（1D では device の写し `State::burn_n_dev` などが作業用で、host の写しは出力・再開用。
  `sync_burn_arrays_to_host/to_device` と同期の状態、2026-10-02）、`inject_burn_source_terms`（source_terms.cu、
  2T 再閉包）、dt lineage "burn"、budget `E_burn_in`
- テスト：`tests/burn/`（reactivity anchors / network closed-forms / deposition
  kernel / stage synthetic spheres / rung-2 6-run 実 run ctest）
  - `burn_stage_2d.hpp/.cpp`（host）：`compute_burn_step_2d()` — 2D_RZ per-cell
    ネットワーク、局所沈着/分配、台帳（プレーン配列 in/out、単体テスト可能）
  - `corman_diffusion.cuh`：次元共通 Corman 係数・群・出生 binning
  - `corman_diffusion_2d.cu/.cuh`：2D RZ 5 点 FV assembly、Jacobi-CG、
    Post-Wilson/Milne 境界、群 cascade と逃逸台帳
- driver 結線（coupling/ 所有）：`callbacks.burn`（laser 直後・radiation 前）、
  比在庫 `State::burn_n_host` [1/g] と `burn_Yg` [1/g]、`inject_burn_source_terms`
  （source_terms.cu、2T/per-material 再閉包）、dt lineage "burn"、budget `E_burn_in`
- テスト：`tests/burn/`（reactivity / network / screening / deposition / 2D stage /
  `burn_net0_2d` / `burn_remap_2d` / Corman 2D）

### 4.7 coupling/
**責務**：演算子分割、dt制御、ソース項統合

- `Coupling::Driver`
  - `DriverRecloseContext` (`driver_reclose.hpp/.cu`) also owns device snapshots
    of conduction-entry Te/cv. Its 1D temperature and energy-increment helpers
    reuse `HydroEOSContext` raw EOS views and preserve each old helper's field
    write set and high-temperature tail. Device table availability is checked
    before selecting snapshots; unavailable tables retain the old host path.
    `TENRYU_CONDUCTION_EOS_HOST=1` selects the original host EOS handoff.
  - dt = min(hydro, cond, rad, user, output)（NUMERICS §2.2）。成長制限 ≤1.2×dt^n は別途適用。レーザーは独立Δt制約を持たない（hydroサブステップに包含、NUMERICS §2.2(d)）
  - **マルチrank同期**：各rank がローカル dt を計算後、`MPI_Allreduce(MPI_MIN)` でグローバル最小 dt を全rankで共有（NUMERICS §2.2）。成長制限 1.2×dt^n はグローバル dt に対して適用する
  - **Hydro開始温度チェック**（NUMERICS §2.1.1）：
    - State に `int8_t* hydro_active`（セル単位の一方向フラグ、`Field<>`はdouble専用のため生ポインタ管理）を保持
    - 初期化時に `hydro_active[c] = (T_start_eV == 0.0)` で設定（§8 ステップ9a）
    - 各ステップ冒頭で非活性セルのみ `T_e[c] >= T_start_eV` を判定（活性セルはスキップ）
    - 非活性セルは圧力・人工粘性の力寄与をゼロ化、ノードは隣接セルのOR論理で移動判定
    - 全セル非活性時：Δt_hydro を除外
    - MPI並列時：ハロー交換でゴーストセルの `hydro_active` を交換（NUMERICS §12.2.2）
  - **Strang splitting 演算子順序**（NUMERICS §2.1）：
    ```
    L(Δt) → H(Δt/2) → C(Δt) → R(Δt) → H(Δt/2)
    ```
    演算子間同期：同一 compute_stream 上で逐次起動。ハロー交換は comm_stream、cudaEvent で依存管理。
    - H = Hydro（Lagrangian step + BC）— セル単位で `hydro_active[c]` に基づき力を計算。ALE は **2回目の H(Δt/2) 後にのみ** 条件付きで実行（2D_RZ: NUMERICS §3.3、1D_SPH: NUMERICS §3.4）
    - C = Conduction（Spitzer-Härm 電子/イオン熱伝導。Q_ei e-i緩和はHydro Corrector内で適用、NUMERICS §1.1.3）
    - L = Laser（ray trace + deposition。ステップ先頭で full-step を1回だけ適用）
    - R = Radiation（`Rad::RadiationStep::step`：FLD または \(S_N\) が輻射場と物質（`Te`, `ee`, `Pe`）を同時に進め、
      `rad_dep`・`rad_emit` を診断として publish する。沈着を後から物質へ注入する段は無い）
  - 1D Hydro の high-k velocity damper の光学的厚さのゲートはセルごとの Rosseland 不透明度の最大値を受け取るが、
    それを与えていたのは退役したモンテカルロ輻射（`IMC::last_sigma_R_max()`）だけで、2026-09-29 から damper は
    このゲートなしで動く（NUMERICS §3.1.4）。
  - 各演算子の前後でハロー交換を挿入（NUMERICS §12.2.3）
  - **演算子間 EOS 再クロージャ**（NUMERICS §2.1、CUDA_KERNELS §9 参照）：各演算子が Te/ee を更新した後、後続演算子向けに EOS 同期を実行する。C(Δt) 後：U2→H13(Te→ee,Pe,Cv)。L(Δt) 後：H14(ee→Te)→H13→U2。R(Δt) 後：H14→H13→U2。
  - `src/coupling/driver_safety_audit.{hpp,cu}` provides the device-side
    post-operator safety audit used by `Driver::run`: Te/Ti/rho/ee/ei
    non-finite flags and finite-Te maximum are reduced on the GPU, then a
    small result is copied to host for the existing `nan_fatal` and overshoot
    decision logic.
  - `src/coupling/driver_fld_energy.{hpp,cu}` provides the device-side FLD/SN
    radiation energy diagnostic used by `Driver::run`: `max(rad_E,0)*vol`
    contributions are formed on the GPU, block-reduced deterministically, and
    accumulated on host from block partials for epsilon-budget inputs.
  - `src/coupling/thermal_subcycle_scan.{hpp,cu}` provides the two device
    scans of the thermal subcycle (NUMERICS §2.1): the compressed floor-hit
    test after each substep and the initial substep-count prediction (minimum
    Te margin above the floor guard). Each returns one value to the host
    instead of copying Te and rho. The attempt backups and substep tallies are
    driver-owned buffers kept across steps.
  - `src/coupling/driver_retry_snapshot.{hpp,cu}` provides the State
    snapshot/restore primitive for driver-level full-step retry on an
    inadmissible hydro corrector.
    Scope is the deterministic radiation modes (FLD/SN).
  - `src/coupling/dispatcher_decision.{hpp,cpp}` provides the pure free-function
    retry dispatcher classifier consumed by `Driver::run` when
    `Numerics.hydro.dispatcher_state_sensitive_bypass_enabled=True`.
  - `src/coupling/profile_observability.{hpp,cpp}` provides the driver-owned
    ICF standard ALE provenance counters/classifier. `Driver::run` resets it at
    run start when `Numerics.profile.icf_standard_ale.enabled=True`, threads a
    nullable pointer through Hydro2D/ALE geometry soft-fail sites, and logs
    `[ale_provenance]` state at run start, fatal-abort, and run end.
    The V22 restart/output contract also defines the production_comparable
    gate structure for per-material conservation: seven criteria, residual-aware status enum, and the
    PASS/PARTIAL-A/PARTIAL-B/INCONCLUSIVE/FAIL/DISABLED classifier used by the
    planned follow-up empirical rerun.
  - When `Numerics.hydro.driver_full_step_retry_enabled=True`, the main driver
    loop wraps each outer step in a retry epoch: capture State at step entry,
    run split operators, accept only if Hydro reports an admissible corrector,
    otherwise restore, halve dt, and retry up to
    `driver_full_step_retry_max_attempts`.  History, snapshot, checkpoint, and
    cumulative energy accounting are below the acceptance boundary, so rejected
    attempts do not publish output.
  - When `Numerics.hydro.driver_retry_active_mesh_repair_enabled=True`, the
    retry epoch also evaluates current corner-J balance before hydro.  Attempt
    0 records diagnostics only; retry attempts with failed balance force the
    existing 2D ALE path via `apply_ale(..., force_rezone=true)`, then
    recompute dt and continue through the ordinary retry acceptance boundary.
    ALE strategy remains the configured 2D ALE mode (`axis_spine_only`,
    `full_winslow`, etc.); the driver adds only the retry-time invocation
    policy.
  - Production-audit infrastructure in `src/coupling/driver.cpp`
    initializes at run start when
    `Numerics.diagnostics.production_audit.enabled=True`, consumes per-step
    escape-valve events, launches the positivity scan, enforces the Tier-A
    termination gate, and emits `audit_summary` output through
    `tools/validation/audit_summary.py`.  The `tenryu_coupling` library now
    PUBLIC links `tenryu_verification` for the shared audit data model.

- `Coupling::SourceTerms`
  - レーザー・燃焼の沈着を e_e へ加える（保存性を保証）
  - `inject_laser_source_terms` は `source_injection` 後の \(e_e \leftrightarrow T_e\) クロージャで
    セルごとの \(A_{eff},\gamma_{eff}\) から構成した \(c_{v,e}\) を用い（NUMERICS §1.1.5a）、cell-local な `laser_dep[c]` を
    そのまま \(e_e\) へ注入し、退化セルでは `E_numerical_loss` へ退避する。燃焼の沈着は `inject_burn_source_terms`。
  - 輻射の沈着は注入しない：FLD・\(S_N\) は物質の更新を自分の Newton の中で行う。モンテカルロ輻射の沈着を注入していた
    `inject_radiation_source_terms`（正味電子ソースの平滑化を含む）は 2026-09-29 に退役し、
    `retired/radiation_monte_carlo/src/coupling/source_terms_radiation.cu` に保管した。
  - **保存性検証**：injection前後の `Σ(ρ·ee·V)` の差と注入量（`Σ laser_dep` 等）の一致を
    Kahan summation で計算し、`|差| / |入力| < 1e-14` を assert

> **注**：電子熱伝導の物理実装（Spitzer-Härm + flux limiter + STS + 負温度防止）は
> `hydro/conduction.*` に配置する（Hydro::Conduction、§4.4）。
> Coupling::Driver が Strang splitting の中で Hydro::Conduction を呼び出す。

---

### 4.8 diagnostics/
**責務**：ICF向け診断量

- `Diag::ArealDensity`：ρR（角度指定の線積分）
- `Diag::Sphericity`：RZのPℓモード、殻半径R(θ)抽出
- `Diag::EnergyBudget`：入射/吸収/流出/系内の収支
- `Diag::LaserPattern`：吸収分布、臨界終了統計、入射角
- `Diag::MCStats`：分散推定、粒子数統計、CI計算
- `Diag::TemperatureMaximumPrinciple`：放射演算子後の温度最大原理違反（`overshoot_count`, `overshoot_max`）の検出・記録
- `Diag::History1D`（`src/diagnostics/history_1d_gpu.{cuh,cu}`）：1D の history 行のセル走査を device で行う（レーザーのエネルギーと吸収重み付き半径、質量重み平均 Zbar と最大 Zbar、ピーク密度・殻平均半径・中心の電子温度、ρR と殻半径）。最大・最小・中心のセルは厳密、ρR の double の和はセルの順に 1 スレッドでビット一致、host が long double で足した和は double-double（末位で異なりうる。inf と NaN は host と同じ）。host へ戻るのは数値だけ
- `corner_collapse_ledger.{cu,hpp}`：`TENRYU_I1B_COLLAPSE_LEDGER` で有効化する read-only の geometry-collapse 診断。
- `Diag::MeshDeformAttribution`（`src/diagnostics/mesh_deform_attribution.{hpp,cuh,cu}`）：default-off の 2D_RZ mesh failure root-cause diagnostics。`Hydro2D::lagrangian_step` invocation ごとに opt-in workspace が start node positions と per-source displacement buffers を所有し、failure 時だけ `mesh_failure_attribution.jsonl` に per-source corner-J degradation を書く。HDF5 schema と `dt_lineage.jsonl` format は変更しない。
- `Diag::MeshDegeneracyForensics`（`src/diagnostics/mesh_degeneracy_forensics.{hpp,cu}`）：default-off の repeated pre-commit `mesh_quality_*` / `in_hydro_*` failure diagnostics。`Hydro2D` は opt-in 時だけ failing cell の4 node position/velocity/acceleration sample を `HydroStepResult` に載せ、`Coupling::Driver` retry path が同一 `(cell, corner, stage)` count と `sigma_safe` threshold を評価して `mesh_degeneracy_forensics.jsonl` へ J(σ), nodal velocity, hourglass amplitude, material/work context を追記する。HDF5 schema と physics state は変更しない。
- `Diag::IcfShellDiagnostics`, `Diag::HotspotGasDiagnostics`, と `Diag::OperatorEnergyResiduals`（`src/diagnostics/diagnostics.{hpp,cu}`, `operator_energy_residuals.{hpp,cu}`）：default-off の ICF shell IFAR/CR、inert gas-hotspot tracer compression metrics、per-operator energy residual。`Coupling::Driver` が history cadence で初期 shell 半径、hotspot tracer state、operator 境界、明示的 `delta_E_ext` を渡し、`HistoryWriter` が `/diagnostics/icf/v1/`, `/diagnostics/hotspot_gas/v1/`, `/diagnostics/conservation/v1/`, `/diagnostics/ale_provenance/v1/` に path-versioned HDF5 series を追記する。HDF5 root `schema_version` は変更しない。
- `Diag::HistoryAppendFile` (`src/diagnostics/history_writer.cpp`): transaction-local
  1D history buffers group already computed rows by dataset and append one
  hyperslab per dataset within the existing flush/copy/rename transaction.
  Dataset handles and owned variable-length strings live until flush; pending
  lengths and last times preserve monotonic rejection and consistency checks.
  Numeric values, attributes, datatypes, shapes and cadence are unchanged.
  Public raw-HDF5 writers and 2D retain immediate appends;
  `TENRYU_HISTORY_BATCH_WRITES=0` selects that path for 1D bisection.
  Since 2026-09-25 the batches are written by one worker thread of the
  writer: the first row of a run is written synchronously (it creates the
  file), later batches (64 rows or 30 s) are queued to the worker, and
  `flush_pending()` (outputs, run end, restart) queues the pending rows and
  waits for the queue to drain; the destructor drains and joins. HDF5 may be
  built without thread safety, so every HDF5 call of the time loop (history
  worker and appenders, snapshot and checkpoint writers, checkpoint and TMAT
  readers, escape-valve and positivity histories) runs under the
  process-wide `core::hdf5_mutex()` (`src/core/hdf5_mutex.hpp`, a recursive
  mutex).
- `Diag::CornerBCAudit`（`src/diagnostics/history_writer.cpp`）：`dt_breakdown_history_enabled=True` の history writer が、CFL winner が r_outer-reflect ∩ z_top-state_supply corner halo に入った step だけ `/diagnostics/corner_bc_audit/v1/` へ interior/ghost state と local dt/cs/Qvisc を追記する diagnostic-only HDF5 group。physics state と HDF5 root `schema_version` は変更しない。
- `Diag::EscapeValveHistory` (`src/diagnostics/escape_valve_history.{hpp,cpp}`):
  Fixed-column HDF5 writer for
  `/diagnostics/escape_valve_audit/v1`.
- `Diag::PositivityHistory` (`src/diagnostics/positivity_history.{hpp,cpp}`):
  Fixed-field HDF5 writer for `/diagnostics/positivity/v1`.
- `Diag::RadialFourierAudit` (`src/diagnostics/radial_fourier_audit.{hpp,cu}`):
  default-off 2D_RZ per-operator radial-null-mode audit. `Coupling::Driver`
  emits before/after stage samples inside the configured time window. The v1
  CUDA kernel computes direct radial DFT amplitudes for `rho`, `Te`, `Ti`,
  cell `u_r`, cell `u_z`, and total `E_rad`, and `HistoryWriter` appends
  `/diagnostics/radial_fourier_audit/v1/` rows without modifying physics state.
  PR G2-A adds an independent v2 fixed-mode complex-coefficient path gated by
  `per_operator_radial_fourier_complex_enabled`; it records selected hidden
  variables (`M`, `V`, momenta, internal/radiation energies, mesh centers/areas,
  `Q_visc`, `f_Fleck` where available) under
  `/diagnostics/radial_fourier_audit_v2/v1/`. The v2 path allocates scratch and
  launches kernels only when that flag is enabled. `TENRYU_RFA_V2_MODE` provides
  a build-time Heisenbug verification matrix: `OFF` compiles the v2 compute path
  out, `STUB` keeps the API but suppresses v2 launches, `DUMMY_BUFFER` launches
  the fixed-mode kernel into temporary device records while suppressing host
  capture/HDF5 append, and `FULL` preserves PR G2-A output. The
  `audit_heisenbug_4config_smoke` ctest builds those four variants, runs the
  same small I1 configuration, and compares final HDF5 state datasets for
  bit-exact or roundoff-bounded drift.
- `Diag::FldSubstageAudit` (`src/radiation/fld_substage_audit_drain.cpp`,
  `src/radiation/fld_2d_rz_gpu.cu`, `src/diagnostics/history_writer.cpp`):
  default-off FLD-internal radial Fourier substage audit gated by
  `Radiation.multigroup_diffusion.diagnostic_radial_fourier_substage_enabled`.
  FLD appends records to a thread-local batch during the radiation solve,
  `src/coupling/driver.cpp` drains the batch immediately after the radiation
  stage, stamps cycle/time metadata, and `HistoryWriter` appends
  `/diagnostics/fld_substage_audit/v1/` on rank 0 only. The path is additive
  and does not modify physics state or root HDF5 `schema_version`.
- Production-audit diagnostics library invariant: `tenryu_diagnostics` now PUBLIC links
  `tenryu_verification` for the shared audit summary data model.  The schema is
  additive-only: existing `/diagnostics/ale_provenance/v1` output is unchanged,
  and new audit output uses new `/diagnostics/*/v1` paths.

```cpp
// DiagOutput は各ステップ終了時の診断スナップショット。
// 瞬時量はメッシュ/粒子から計算し、累積量は State のメンバーから読み出す（§5.2 State 参照）。
struct DiagOutput {
    // --- 瞬時量（現ステップのメッシュ/粒子状態から計算）---
    double E_kinetic;            // 運動エネルギー [erg]
    double E_internal_e;         // 電子内部エネルギー [erg]
    double E_internal_i;         // イオン内部エネルギー [erg]
    double E_radiation;          // 放射エネルギー [erg] = Σ_c Σ_g rad_E[c,g] V_c（FLD・S_N の場。NUMERICS §10.2）
    double rhoR_avg;             // 面密度 [g/cm²]
    double shell_radius_mean;    // 殻平均半径 [cm]
    double shell_radius_min;     // 殻最小半径 [cm] (ρ > 0.1*ρ_max の最小 r_c)
    double a_2;                  // Legendre mode 2 振幅 [dimensionless]（Pℓ分解、§4.8 Diag::Sphericity）
    double Zbar_mean;            // 質量重み平均電離度 [dimensionless]
    double Zbar_max;             // 最大電離度 [dimensionless]
    double T_e_max;              // 最大電子温度 [eV]
    double T_i_max;              // 最大イオン温度 [eV]
    double rho_max;              // 最大密度 [g/cm³]
    int clamp_count;             // フロアクランプ回数 (per step, reset each step)
    int overshoot_count;         // 最大原理違反セル数 (per step, Radiation直後。history radiation/overshoot_count)
    double overshoot_max;        // 最大超過率 δ_max (per step, Radiation直後。history radiation/overshoot_max)
    // --- 累積量（State のメンバーから読み出し — §5.2 Cumulative diagnostics）---
    double E_laser_deposited;    // = State.E_laser_deposited [erg]
    double E_laser_escaped;      // = State.E_laser_escaped [erg]
    double E_rad_escaped;        // = State.E_rad_escaped [erg]
    double E_floor_injected;     // = State.E_floor_injected [erg]
    double E_safety;             // = State.E_safety [erg]
    double E_numerical_loss;     // = State.E_numerical_loss [erg]
};
DiagOutput compute_diagnostics(
    const Mesh& mesh,
    const State& state,
    const Config& cfg,
    const LaserMesh* laser_mesh,     // nullable（laser無効時は nullptr）
    const EOSTable* eos_e,           // [n_materials] 電子EOS（C_v計算等に使用）
    const double* zbar,              // [n_cells] 現ステップの Z̄ [dimensionless]
    const RadiationResult& rad_result,
    cudaStream_t stream
);
```

---

### 4.9 io/
**責務**：入出力、再始動、メタデータ

- `IO::HDF5Writer`：HDF5 出力（rank 0 だけが書き、MPI-IO は使わない）。snapshot と checkpoint の deflate 付き dataset のうち 64 KiB 以上は、プロセス共通の圧縮スレッド群（`DeflatePool`。スレッド数は使える CPU 数 − 2、最低 2。使える CPU 数は Linux ではプロセスの affinity と cgroup の CPU 上限で制限する — RunPod の pod はハードウェアスレッド 128・上限 13.6 CPU）が圧縮（zlib `compress2`、HDF5 の deflate フィルタと同じ形式・同じレベル）し、ファイルを閉じる前に `H5Dwrite_chunk` で書く（`DeferredDeflateChunks`、2026-09-25）。512 KiB を超える dataset は第 1 次元に沿って約 256 KiB のチャンクに分けて保存する（端のチャンクは 0 で埋める）。時間ループの snapshot（`OutputManager::write_snapshot`）は `HDF5Writer::write_snapshot_in_background` で書く: ファイルの作成・群・属性・小さい dataset の書き込みと大きい dataset のチャンクの圧縮依頼までを行って返り（データはこの時点で複製済み）、完了（圧縮の待ち・チャンクの書き込み・close・`.tmp` からの rename による公開）は 1 本の完了スレッド（`SnapshotFinisher`）が書いた順に行う（2026-09-25。完了待ちは 2 個まで、超えると書き込み側が待つ。完了スレッドのエラーは次の書き込みか `HDF5Writer::wait_for_snapshot_writes` で再送出され、driver は run の最後の snapshot の後で待つ）。`HDF5Writer::write_snapshot` は同じ書き込みの後、公開まで待ってから返る（戻った時点でファイルがある）。HDF5 の呼び出しはプロセス共通の再帰ロック（`core::hdf5_mutex`）で直列化する（完了スレッドも同じロックの下で書く）。スナップショットの各群は NVTX 区間 `io.snapshot.*`（完了スレッドの close は `io.snapshot.close`）で計測できる
- `IO::Checkpoint`：State + Mesh（schema 2。schema 1 にあった光子粒子 `particles/` と RNG 状態 `rng/` は 2026-09-29 に廃止）
- `IO::Restart`
- `IO::Schema`：互換性ルール（スキーマ破壊禁止）

**IO方式**：
- `write_snapshot`：rank 0 だけが 1 つのファイルに書く（MPI-IO は使わない。1D の MPI 実行では書く前に各 rank の担当区間を rank 0 に集める）
- `write_checkpoint`：rank 0 だけが 1 つのファイル `checkpoints/<case>_ckpt_NNNN.h5` に書く（旧形式のランク別ファイル `_rNNNN.h5` は読み込める）
- `read_checkpoint`：各 rank が同じチェックポイントファイルを全体読み込む
  - **ランク数変更可**：各 rank がファイル全体を読むので、保存時と rank 数が違ってもよい（SPECIFICATION §7.4）
  - 粒子の再配布：`cell_id` から新パーティションの所属 rank を判定し MPI 送信
  - `config_hash` 不一致時：WARNING 出力（凍結パラメータ変更は `ConfigError`、SPECIFICATION §7.4 参照）
  - RNG復元：`curand_init(global_id ^ user_seed, step_number, rng_counter)` で O(1) 復元（rank非依存、NUMERICS §12.7.1 準拠）

```cpp
// === IO モジュール (§4.9) ===
namespace IO {
    // --- HDF5 スキーマバージョニング ---
    // ルートグループ attrs: schema_version = 1（v1.0 初版、SPECIFICATION §7.5準拠）
    // 読込時のバージョン互換ルール：
    //   - schema_version == 現行 → そのまま読込
    //   - schema_version < 現行  → 後方互換リーダーが欠落フィールドに既定値を補完し WARNING 出力
    //   - schema_version > 現行  → ConfigError("checkpoint schema version {v} is newer than code version {current}")
    //   - schema_version 属性が存在しない（v0 以前）→ version=0 として後方互換リーダーを適用
    // バージョン変更条件：HDF5 group/dataset の追加・削除・型変更時にインクリメント
    static constexpr int SCHEMA_VERSION = 1;

    // Snapshot: HDF5 ファイル構造（SPECIFICATION §7.2 準拠）
    // データセット名は State フィールド名と 1:1 対応（名前マッピング不要）
    // 例外: mesh/cell_material_id は State 直接フィールドではなく IO 時に派生生成（SPECIFICATION §7.2）
    // /metadata/          : {namelist_source, frozen_config, group_bounds_eV, schema_version, attrs...}
    // /mesh/x_r           : double[n_nodes]   — 節点R座標 [cm] (State.x_r)
    // /mesh/x_z           : double[n_nodes]   — 節点Z座標 [cm] (2D only, State.x_z)
    // /mesh/v_r           : double[n_nodes]   — 節点R速度 [cm/s] (State.v_r)
    // /mesh/v_z           : double[n_nodes]   — 節点Z速度 [cm/s] (2D only, State.v_z)
    // /hydro/rho          : double[n_cells]   — 質量密度 [g/cm³] (State.rho)
    // /hydro/Te           : double[n_cells]   — 電子温度 [eV] (State.Te)
    // /hydro/Ti           : double[n_cells]   — イオン温度 [eV] (State.Ti)
    // /hydro/ee           : double[n_cells]   — 電子比内部エネルギー [erg/g] (State.ee)
    // /hydro/ei           : double[n_cells]   — イオン比内部エネルギー [erg/g] (State.ei)
    // /hydro/Pe           : double[n_cells]   — 電子圧力 [dyne/cm²] (State.Pe)
    // /hydro/Pi           : double[n_cells]   — イオン圧力 [dyne/cm²] (State.Pi)
    // /hydro/Qvisc        : double[n_cells]   — 人工粘性圧 [dyne/cm²] (State.Qvisc)
    // /hydro/mass         : double[n_cells]   — セル質量 [g] (State.mass)
    // /hydro/vol          : double[n_cells]   — セル体積 [cm³] (State.vol)
    // /hydro/zbar         : double[n_cells]   — 平均電離度 Z̄ [-] (State.zbar)
    // /hydro/per_material/v1/{mass,Ee,Ei}
    //                      : double[n_cells,n_materials] — per-material conservation
    //                        authoritative extensive per-material state.
    // /diagnostics/conservation/v1/per_material_*_residual
    //                      : scalar residual diagnostics for Σ_m conserved
    //                        state against cell-mean projections.
    // /diagnostics/per_material/v1/*
    //                      : cumulative per-material event counters.
    // /metadata/dispatch_counters/*
    //                      : dispatch counter regression-hash inputs; disabled
    //                        per-material conservation mode persists all-zero per-material counts.
    // /mesh/topology/v2/*  : optional group written only when
    //                        topology_scheme="multiblock_cart_core_polar_shell";
    //                        path-versioned topology extension, no root
    //                        schema_version bump. Sub-datasets:
    //                        cell_block_id[int32,n_cells_total],
    //                        cell_id_stable[int32,n_cells_total],
    //                        cell_node_csr_offsets[int32,n_cells_total+1],
    //                        cell_node_csr_indices[int32,total_corners],
    //                        face_adj_csr_offsets[int32,n_cells_total+1],
    //                        face_adj_csr_indices[int32,total_faces],
    //                        face_bc_tags[int32,total_faces],
    //                        block_counts[int32,7] =
    //                        {block_count,n_cells_core,n_cells_bridge,
    //                         n_cells_shell,n_nodes_core,
    //                         n_nodes_bridge_interior,n_nodes_shell}.
    //                        Readers first probe /mesh/topology/v2. If absent,
    //                        they fall back to v1/single-block reconstruction.
    //                        Unknown newer path versions hard-fail rather than
    //                        changing root schema_version semantics.
    // /mesh/topology/v3/*  : optional group written only when
    //                        topology_scheme="multiblock_half_butterfly_5block";
    //                        variable-block topology extension for the B-S1
    //                        five-block half-butterfly. Readers probe v3,
    //                        then v2, then v1/single-block. Sub-datasets:
    //                        block_count[int32]=5,
    //                        block_id[int32,block_count],
    //                        block_role[int32,block_count],
    //                        block_n_i_cells[int32,block_count],
    //                        block_n_j_cells[int32,block_count],
    //                        block_cell_begin[int32,block_count],
    //                        block_cell_count[int32,block_count],
    //                        block_owned_node_begin[int32,block_count],
    //                        block_owned_node_count[int32,block_count],
    //                        seam_* tables with orientation/index ranges,
    //                        cell_block_id[int32,n_cells_total],
    //                        cell_id_stable[int32,n_cells_total],
    //                        cell_orientation_sign[int32,n_cells_total],
    //                        cell_node_csr_offsets/indices,
    //                        face_adj_csr_offsets/indices,
    //                        face_bc_tags[int32,total_faces].
    //                        The root schema_version remains unchanged.
    // /hydro/eta_compatible : double[n_cells] — optional legacy compatible-volume mismatch [cm³] (State.eta_compatible)
    // /hydro/volFrac      : double[n_cells x n_mat] — 体積分率 (State.volFrac)
    // /radiation/energy_density    : double[n_cells x G] — E_g [erg/cm³]
    // /radiation/rad_dep           : double[n_cells x G] — [erg] (当該ステップ累積)
    // /radiation/rad_emit          : double[n_cells x G] — emission diagnostic [erg]（当該ステップ累積）
    // /radiation/deposited_power   : double[n_cells x G] — [erg/cm³/s] = rad_dep / (V × dt)
    // /radiation/fleck_factor      : double[n_cells] — 2D_RZ FLD Fleck factor f_i [-]
    // /radiation/sn_tau_R, sn_reduced_flux, sn_ap_alpha
    //     : double[n_cells] — 2D_RZ S_N AP transition diagnostics [-]
    // /radiation/diag_rad_E_pre, diag_rad_E_post,
    // /radiation/diag_rad_emission_at_Tn, diag_rad_emission_at_Tnp1,
    // /radiation/diag_rad_absorption, diag_clip_energy, diag_clip_full_deficit,
    // /radiation/diag_chi_opacity, diag_F_first_moment, diag_E_star_flux, diag_stream_theta
    //     : double[n_cells x G] — 1D S_N plateau investigation diagnostics (output-only)
    // /radiation/diag_ap_alpha_face : double[n_faces x G] — AP face_blend weight (output-only)
    // （schema 2（2026-09-29）で /radiation/ddmc_flag・/radiation/delta_E_rad_prev・/holo/*・/difference/* を廃止。
    //  いずれも退役したモンテカルロ輻射の出力。OUTPUT_SCHEMA・SPECIFICATION §7.5）
    //
    // Checkpoint: 上記 + 以下を追加（SPECIFICATION §7.4 準拠）
    // hydro_flags/hydro_active  : int8[n_cells]（SPECIFICATION §7.4 準拠、State は int8_t*）
    // config_hash               : uint64 attrs (frozen config の hash — restart 時検証)
    // time_state/*              : E_laser_deposited, E_laser_escaped, E_rad_escaped,
    //                              E_floor_injected, E_safety, E_numerical_loss,
    //                              E_pdV_bdry, E_Marshak_in, E_solver
    //                            （State フィールド名と一致。SPECIFICATION §7.4 time_state/* 準拠）
    // output_state/t_next_*     : double × 3（SPECIFICATION §7.4 出力タイミング状態）
    // time_state/t              : float64（現在時刻 [s]、State.t と対応）
    // time_state/step           : int32（**最後に完了したステップ番号**、State.step と対応。
    //                              リスタート時は step+1 から実行再開する。燃焼の α 粒子 Monte Carlo は
    //                              step を Philox の subsequence に使うので、off-by-one はストリーム重複を
    //                              引き起こす — この契約を厳守すること）
    // time_state/dt             : float64（タイムステップ幅 [s]、NUMERICS §2.2 Δt成長制限復元用）
    // time_state/ale_last_applied_step : int32（ALE が最後に rezone を commit した step。2D_RZ の
    //                              ALE が書く。判定に読んでいたのは 1D V3 ALE の cadence・最小間隔
    //                              だけで、1D ALE は 2026-10-02 に退役。旧checkpointでは -1 で補完）
    // time_state/E_safety            : float64（伝導安全補正累積、State.E_safety と対応）
    // time_state/E_numerical_loss    : float64（退化セル損失累積、State.E_numerical_loss と対応）
    // time_state/E_laser_deposited   : float64（レーザー沈着累積、State.E_laser_deposited と対応）
    // time_state/E_laser_escaped     : float64（レーザー脱出累積、State.E_laser_escaped と対応）
    // time_state/E_rad_escaped       : float64（放射脱出累積、State.E_rad_escaped と対応）
    // time_state/E_floor_injected    : float64（フロア注入累積、State.E_floor_injected と対応）
    // time_state/E_pdV_bdry          : float64（境界PdV仕事累積、State.E_pdV_bdry と対応）
    // time_state/E_Marshak_in        : float64（Marshak入射累積、State.E_Marshak_in と対応）
    // time_state/E_solver            : float64（Hypre残差累積、State.E_solver と対応。v1.0=0）

    void write_snapshot(const std::string& path, const State& state,
                        const Mesh& mesh, const Config& cfg, int step, double time);
    void write_checkpoint(const std::string& path, const State& state,
                          const Mesh& mesh, const Config& cfg, int step, double time);
    State load_checkpoint(const std::string& path, const Config& cfg,
                          const PartitionInfo& part);
    // 出力は rank 0 だけが書く（MPI-IO・collective write は使わない）
    // 大きい dataset のチャンク分割と圧縮は上の IO::HDF5Writer の説明を参照
}
```

---

### 4.10 verification/
**責務**：verification references, audit policy data models, and validation-facing summaries

- Existing analytic/reference modules remain in `src/verification/`:
  `marshak.cu`, `sedov_analytic.cpp`, `noh_analytic.cpp`, and `laser_analytic.cu`
  (`diffusion_ref.cpp`, the diffusion reference of the retired DDMC gates, moved to
  `retired/radiation_monte_carlo/` on 2026-09-29).
- `Verification::TierThreshold` (`src/verification/tier_threshold.{hpp,cu}`):
  The 9-tier threshold framework.
- `Verification::AuditSummary` (`src/verification/audit_summary.{hpp,cpp}`):
  shared audit summary data model and JSON serialization.
- `Verification::EscapeValveAudit`
  (`src/verification/escape_valve_audit.{hpp,cpp}`): 6-flag escape-valve
  counter and Tier-A/Tier-B policy.
- `Verification::PositivityTracker`
  (`src/verification/positivity_tracker.{hpp,cu}`): per-step positivity scanner
  over six fields.
- The `tenryu_verification` library now PUBLIC links `tenryu_parallel` for the
  Reduction primitive used by production-audit scanners.
- Architectural invariants from commits `cc62bad5`, `f56f8612`,
  `547d894c`, and `f08646b8`:
  - all infrastructure is default OFF, preserving the bit-exact baseline when
    `production_audit.enabled=false`;
  - frozen-config schema advances V22 to V23 additively;
  - Tier-A verification requires zero escape-valve firings, while Tier-B
    engineering uses the 6-condition escape-valve policy.

### 4.11 tools/validation/
**責務**：production-audit validation postprocessing

- `tools/validation/audit_summary.py` assembles `<case>.audit.json` from
  `run_info.json`, history HDF5, and reproducibility metadata.
- `tools/validation/reproducibility_check.py` evaluates same-architecture byte
  reproducibility, cross-architecture tolerance, and emits
  `cross_arch_metadata.json`.

---

## 5. データモデル（State）
計算状態は `State` が一括保持し、各モジュールは必要ビューを受け取って更新する。

### 5.1 Field\<\> テンプレート

GPUデバイスメモリ上のフィールドデータを管理する薄いRAIIラッパー群。
タグ型でインデックス次元を静的に区別する。

> **旧 `Field<Tag>` / `Field<Tag1,Tag2>` からの変更**：
> C++ では同名クラステンプレートのパラメータ数違い（1引数 vs 2引数）は
> 部分特殊化ではなく **再定義** となり、コンパイルエラーになる。
> `Field1D` / `Field2D` / `FieldG` に分離することで衝突を回避し、
> 1D/2D切替も明示的になる。

```cpp
// 1Dフィールド（1D_SPH: セルまたはノード単位）
// メモリ確保：cudaMalloc でGPUデバイスメモリに配置。ホストミラーは持たない。
// I/O（HDF5出力）時は copy_to_host() でピン留めホストバッファへ転送し、
// HDF5 への書き込み後にホストバッファを解放する。
template<typename Tag>
struct Field1D {
    double* data;     // [n] deviceメモリ（cudaMalloc で確保）
    int     n;

    // host↔device転送
    void copy_to_host(double* host_dst) const;
    void copy_from_host(const double* host_src);

    // size変更（破壊的realloc + ゼロ初期化）
    void reset(int new_size);
    // size変更（データ保持）
    void resize(int new_size);

    // ムーブセマンティクス（コピー禁止）
    Field1D(Field1D&& other) noexcept;
    Field1D& operator=(Field1D&& other) noexcept;
    Field1D(const Field1D&) = delete;
    Field1D& operator=(const Field1D&) = delete;

    // デストラクタで cudaFree
    ~Field1D();
};

// 2Dフィールド（2D_RZ: セルまたはノード単位）
// メモリ確保：cudaMalloc でGPUデバイスメモリに配置（Field1D と同一方針）。
template<typename Tag>
struct Field2D {
    double* data;     // [nr × nz] deviceメモリ（cudaMalloc）（row-major: data[i*nz + j], i=r方向, j=z方向）
    int     nr, nz;

    // host↔device転送
    void copy_to_host(double* host_dst) const;
    void copy_from_host(const double* host_src);

    // size変更（realloc）
    void resize(int new_nr, int new_nz);

    // ムーブセマンティクス（コピー禁止）
    Field2D(Field2D&& other) noexcept;
    Field2D& operator=(Field2D&& other) noexcept;
    Field2D(const Field2D&) = delete;
    Field2D& operator=(const Field2D&) = delete;

    // デストラクタで cudaFree
    ~Field2D();
};
// **ゴーストセル対応**：MPI並列時、ghost付きフィールド（§5.2 で [n_cells+n_ghost] と注記されるもの）は
// data = cudaMalloc(sizeof(double) * (nr*nz + n_ghost)) で確保する。Field2D.nr/nz は **interior 次元** であり
// 確保サイズは nr*nz + n_ghost。resize(nr, nz) もこの拡張サイズを考慮すること。
// ghost アクセス: data[n_cells + ghost_offset]（CUDA_KERNELS §2.2 H4 ゴーストセル契約参照）。
// single-GPU (n_ghost=0) では nr*nz == n_cells。

// 群依存フィールド（放射エネルギー密度など）
// メモリ確保：cudaMalloc でGPUデバイスメモリに配置（Field1D と同一方針）。
template<typename Tag>
struct FieldG {
    double* data;     // [n_spatial × G] deviceメモリ（cudaMalloc）（1D: n_spatial=n_cells, 2D: n_spatial=nr*nz）
    int     n_spatial; // セル数: n_cells (1D) or nr*nz (2D)
    int     G;         // 群数

    // メモリレイアウト：spatial-major（cell-major）
    // data[i * G + g] = spatial index i, group g の値
    // → 同一セルの全群が連続 → セル単位で群を回すカーネルでキャッシュ効率が良い

    int total_size() const { return n_spatial * G; }
};

// タグ型（ゼロコスト識別）
struct CellTag {};
struct NodeTag {};

// エイリアス
using CellField1D = Field1D<CellTag>;
using NodeField1D = Field1D<NodeTag>;
using CellField2D = Field2D<CellTag>;
using NodeField2D = Field2D<NodeTag>;
using CellFieldG  = FieldG<CellTag>;  // rad_dep, rad_E 等
```
dim=1 の場合は `Field1D` を使用し、dim=2 の場合は `Field2D` を使用する。
コンパイル時ディスパッチは以下の `#ifdef` で処理：

```cpp
// --- v1.0 設計方針: コンパイル時ディスパッチ ---
// TENRYU_2D マクロは CMake オプション -DTENRYU_DIM=2 で設定される。
// 1D バイナリと 2D バイナリを別々にビルドする（dim はコンパイル時定数）。
// Config::MainConfig::dim は Python 入力からの読み込み値であり、
// コンパイル時定数 TENRYU_DIM と一致しない場合は ConfigError を送出する。
// if constexpr (TENRYU_DIM == 2) による分岐で 1D/2D コードを選択する。
// これにより未使用次元のコードがコンパイル時に除去され、
// レジスタ圧力の低減と分岐コストの排除が実現できる。

#ifdef TENRYU_2D
  using CellField = CellField2D;
  using NodeField = NodeField2D;
#else
  using CellField = CellField1D;
  using NodeField = NodeField1D;
#endif

// 多材料フィールド：cells × n_mat の2D配列。群依存フィールド (FieldG) と同構造だが
// 第2軸が群数ではなく材料数。FieldG<CellTag> を再利用し n_group の代わりに n_mat を渡す。
using CellFieldMat = FieldG<CellTag>;  // data[cell * n_mat + mat]

// --- 型安全に関する注記（v1.0 設計判断）---
// v1.0: n_mat は最大 MAX_MATERIALS=8 で固定、G は最大 80
// CellFieldG は group 用、CellFieldMat は material 用として別名定義：
//   using CellFieldG   = FieldG<CellTag>;  // [n_cells × G] group-indexed
//   using CellFieldMat = FieldG<CellTag>;  // [n_cells × n_mat] material-indexed
// 注意: CellFieldG と CellFieldMat は同じ型。意味的区別はコンテキストとコメントで管理。
// 将来の型安全強化では GroupTag/MatTag テンプレート引数を追加し、
//   FieldG<CellTag, GroupTag> / FieldG<CellTag, MatTag> のように区別する予定。
```

**設計判断**：
- `thrust::device_vector` は不使用。薄いラッパーでメモリ管理を明示化し、
  不要なhost-device同期を回避する
- **所有権**：`State` が全 Field を所有する。モジュールは `double*` ポインタを受け取って操作する
- **群依存フィールドはcell-major**：`data[cell * G + group]`。
  粒子カーネルは1粒子が1セルの全群を参照するため、群方向が連続だとキャッシュヒット率が高い

> **導入タイミング**：Field テンプレート群は **M03（1D SPH基本実装）から導入**する。
> 理由：コンパイル時型安全はゼロランタイムコスト。後からの導入は全ファイルのリファクタが必要になる。
> M03の時点で CellTag / NodeTag タグ型を定義し、以降のマイルストーンで一貫して使用する。
>
> **メモリモードのフェーズ展開**：
> - **M03（Hydro検証）**：`std::vector<double>` ベースのホストメモリモード。CUDAカーネル不使用のため `data()` で生ポインタ取得。
> - **M04以降（GPU化）**：`cudaMalloc` デバイスメモリへ移行。Field API（`operator[]`, `data()`, `size()`）は共通のため、呼び出し側コード変更は不要。
> - 移行時の変更点：`std::vector` → デバイスポインタ + `cudaMemcpy` ヘルパー。State のアロケータを切り替えるだけで完了。

### 5.2 State 構造体

State はホスト側配置。メンバポインタは全てデバイスメモリ。カーネルへは個別ポインタ引数展開で渡す。

> **ゴーストセル込みバッファサイズ規約**：
> MPI並列時、ハロー交換で更新するフィールド（rho, Te, Ti, Pe, Pi, Qvisc, adaptive_av_gate, zbar, vol, face_area,
> volFrac, hydro_active）は `n_cells + n_ghost` 要素で確保する。
> ここで `n_ghost` は `PartitionInfo.n_ghost_cells`（ゴーストセル総数）のエイリアスである。
> `PartitionInfo.ghost_layers`（レイヤー数 = 1）とは異なるので注意。
> 2D_RZ 1層ハロー: n_ghost_cells = 2×(nr_local+nz_local)+4。単一GPU時は `n_ghost_cells=0`。
> カーネルシグネチャで `[n_cells + n_ghost]` と注記されたフィールドは ghost 込みバッファを前提とする。
> `[n_cells]` のみ注記されたフィールド（mass, ee, ei, Cv_e, Cv_i, delta_l 等）は owned セル専用で
> ghost 領域は確保しない。FieldG 型（rad_dep, rad_E 等）は owned セル `[n_cells × G]` のみ。

```cpp
struct State {
    Mesh mesh;              // 計算格子（Topology + Geometry）

    // --- 時間・ステップ管理 ---
    // allocate() で初期化。main loop で毎ステップ更新。
    // checkpoint 保存・復元対象。ただしhistory専用カウンタは復元必須ではない。
    double t = 0.0;         // 現在時刻 [s]
    int    step = 0;        // 現在ステップ番号
    double dt = 0.0;        // 現在タイムステップ幅 [s]
    bool   ale_rezoned = false;       // 直近ステップでALE rezoneが発動したか
    int    ale_rezone_invocations = 0; // 累積history診断用（checkpoint必須ではない）
    int    ale_remaps_applied = 0;     // accepted 2D ALE remap counter（checkpoint必須ではない）
    int    ale_last_applied_step = -1; // ALE が最後に rezone を commit した step（2D_RZ が書く。1D V3 ALE の cadence/min-step gate が読んでいた — 2026-10-02 に退役）

    // --- 出力タイミング状態（時間間隔ベース出力用）---
    // checkpoint 保存・復元対象（SPECIFICATION §7.4 output_state/）。
    // -1.0 = 対応する X_every_s が無効。
    double t_next_plot = -1.0;       // [s] 次回plot出力時刻
    double t_next_history = -1.0;    // [s] 次回history出力時刻
    double t_next_checkpoint = -1.0; // [s] 次回checkpoint出力時刻

    // --- ファクトリ関数 ---
    // Config と PartitionInfo から全フィールドを確保し初期化する。
    // 確保対象：全 CellField/NodeField/CellFieldG/CellFieldMat、
    //           hydro_active、Scratch、DeviceErrorFlags
    static State allocate(const Config& cfg, const PartitionInfo& part);

    // --- Hydro cell fields ---
    // CellField / NodeField は §5.1 の #ifdef TENRYU_2D で
    // CellField1D/CellField2D, NodeField1D/NodeField2D に解決される。
    CellField rho;          // 質量密度 [g/cm³]
    CellField zbar;         // 平均電離度 Z̄ [dimensionless]（NUMERICS §1.1.4）。Hydro ステップ冒頭で更新
    CellField A_eff;        // 有効原子量 [dimensionless]（NUMERICS §1.1.5a）。U8 compute_zbar 出力。
                            // 多材料: 体積分率加重調和平均 1/A_eff = Σ f_α/A_α。単一材料: A_ion。
                            // H15(compute_sound_speed), U6(qei_exchange), C1(compute_spitzer_deff) が参照。
                            // チェックポイント保存推奨（リスタート時は post-restart U8 で再計算可能）
    CellField mass;         // セル質量 [g] = ρ×V
    CellField vol;          // セル体積 [cm³]（Mesh.recompute_geometry() 後に Mesh.cell_vol からコピー。§4.2.2 同期契約参照）
    CellField cell_vol_initial; // IC 時セル体積 [cm³]。2D_RZ geometric CFL cumulative floor の固定基準
    std::vector<uint8_t> center_patch_latch; // HOST runtime-only center/quality-patch hysteresis bitmask; restart re-seeds from geometry
    CellField Te, Ti;       // 電子/イオン温度 [eV]
    int8_t* hydro_active;   // OWNED [n_cells + n_ghost] deviceメモリ。セル単位Hydro活性フラグ（一方向、NUMERICS §2.1.1）。
                            // n_ghost込み: ハロー交換対象（§5.2 ゴーストセル込みバッファサイズ規約、CUDA_KERNELS §9 Phase 1 halo_exchange 参照）
    std::vector<int8_t> state_supply_mask; // HOST [n_cells] z-face state_supply row mask
    CellField state_supply_pre_rho, state_supply_pre_mass, state_supply_pre_ee,
              state_supply_pre_ei, state_supply_pre_uz; // reservoir tally pre-state snapshots
    double state_supply_dM_cumulative, state_supply_dE_cumulative, state_supply_dPz_cumulative;
    double state_supply_dM_step, state_supply_dE_step, state_supply_dPz_step;
    CellField ee, ei;       // 電子/イオン比内部エネルギー [erg/g]
    CellField Pe, Pi;       // 電子/イオン圧力 [dyne/cm²]
    CellField Cv_e, Cv_i;   // 電子/イオン比熱 [erg/(g·eV)]（H13 eos_forward 出力。R1 Fleck因子、C1 D_eff で使用）
    CellField Qvisc;        // 人工粘性圧 [dyne/cm²]（NUMERICS §3.1.6, §3.2.9）
    CellField hllc_mom_z_cell; // optional experimental z-HLLC authoritative cell z momentum [g cm/s]
    CellField corner_mass, corner_volume, corner_pressure; // optional 2D_RZ Phase 3 flat corner/subzone state [g, cm³, dyne/cm²]（runtime-only、checkpoint対象外）
    CentralPseudoCoreState central_pseudo_core; // runtime-only virtual center macro overlay
    PoleAngularDerefineState pole_angular_derefine; // runtime-only I1-B polar-shell angular macro overlay
    CellField subzonal_mass_corner0, subzonal_mass_corner1,
              subzonal_mass_corner2, subzonal_mass_corner3; // 2D_RZ hourglass runtime subzonal masses [g]（default-off、checkpoint対象外、NUMERICS §3.2.9b）
    CellField adaptive_av_gate; // optional adaptive AV gate 履歴 g_i [-]（adaptive_av.enabled=True）
    CellField eta_compatible; // optional legacy compatible-volume mismatch [cm³]（exact force-work path では保存機構に未使用）
    CellFieldMat volFrac;   // [dimensionless] 体積分率 [(n_cells+n_ghost) × n_mat]（§5.1 CellFieldMat、NUMERICS §1.1.5 (c)）。
                            // n_ghost込み: U9 がゴーストセルの opacity 混合則に volFrac を参照（§5.2 規約準拠）
    std::vector<std::uint8_t> cell_is_void; // [n_cells] void セルマスク（1=void, 0=通常）。
                            // geometry_eval で体積分率から導出: nonvoid_sum ≤ 1e-12 → void。
                            // Fleck (f=1), 伝導 (κ=0), カップリング (deposit skip) で参照。
                            // ALE remap 後に再計算（driver.cpp）。
    double adaptive_av_r0, adaptive_av_last_rs, adaptive_av_last_us, adaptive_av_rs_min; // adaptive AV tracker scalars
    int adaptive_av_tracker_steps, adaptive_av_mode;                 // tracker warm-up and mode trace
    bool adaptive_av_tracker_valid, adaptive_av_bounce_seen;         // adaptive AV state machine latches

    // --- Node fields ---
    // エイリアス契約：Mesh.node_r/node_z は State.x_r.data/x_z.data の
    // 借用ポインタ（non-owning）である（§4.2.2 参照）。
    // State が x_r/x_z の所有権を持ち、Mesh は参照のみ。
    // Lagrangian 移動や ALE rezone で x_r/x_z を更新した後は、
    // Mesh.recompute_geometry() を呼び出して体積・重心・面積を再計算すること。
    // Mesh.node_r ポインタの再設定は不要（同一バッファを指し続ける）。
    NodeField x_r, x_z;    // ノード座標 [cm]（1Dでは x_z 未使用）
    NodeField x_r_reference, x_z_reference; // IC reference mesh coordinates for 2D_RZ BC/ALE targeting
    NodeField v_r, v_z;     // ノード速度 [cm/s]

    // --- Radiation ---
    CellFieldG rad_E;       // 群ごとの輻射エネルギー密度 [erg/cm³]（FLD・S_N の解。NUMERICS §6.7、§6.8）
    CellFieldG rad_E_old;   // ステップ初めの rad_E [erg/cm³]（FLD・S_N の時間差分と γ_r=4/3 圧縮の基準）
    CellFieldG rad_dep;     // 物質が吸収した輻射エネルギー [erg]（セル×群、ステップ内の累積。FLD・S_N が publish）
    CellFieldG rad_emit;    // 物質が放出した輻射エネルギー [erg]（同上）
    CellFieldG sn_diag_*;   // 1D S_N output-only plateau diagnostics（E pre/post, emission/absorption,
                            // clip rad/full deficit, opacity lag, angular first moment, face-flux E*, stream theta）[N_cell × G]
    FaceFieldG sn_face_flux_raw; // 1D S_N signed radial face flux [N_face × G], positive outward
    FaceFieldG sn_face_flux_limited; // donor-theta limited radial face flux [N_face × G]
    FaceFieldG sn_face_flux_diff;    // AP face_blend FLD-style diffusion face flux [N_face × G]
    FaceFieldG sn_face_alpha;        // AP face_blend weight [N_face × G]
    CellFieldG sn_stream_theta;      // donor streaming limiter theta [N_cell × G]
    CellFieldG sn_E_star_flux;   // 1D S_N face-flux streaming state [N_cell × G]
    CellField fld_fleck;         // 2D_RZ FLD output-only Fleck factor diagnostic [N_cell]
    double fld_state_supply_in_cumulative, fld_state_supply_out_cumulative,
           fld_state_supply_net_cumulative; // 2D_RZ FLD state_supply Dirichlet boundary tally [erg]
    double fld_state_supply_in_step, fld_state_supply_out_step,
           fld_state_supply_net_step;       // current FLD stage state_supply tally [erg]
    CellField sn_tau_R;          // 2D_RZ S_N AP optical-depth diagnostic [N_cell]
    CellField sn_reduced_flux;   // 2D_RZ S_N AP reduced-flux diagnostic [N_cell]
    CellField sn_ap_alpha;       // 2D_RZ S_N AP alpha diagnostic [N_cell]
    bool holo_ale_invalidated; // ALE の再マップが輻射場の下の格子を動かしたとき（1D・2D）に立ち、FLD・S_N が格子に依存する
                               // キャッシュを作り直して時間履歴を始め直してから下ろす（名前は退役した HOLO の名残）
    // 退役したモンテカルロ輻射の状態（PhotonPool、DDMC のモード判定 ddmc_candidate/ddmc_mode、界面の face current、
    // HOLO の holo_*、difference 定式化の difference_*、rad_E_tally、rad_mom_dep、bc_type_rad、ell_ddmc）は State に無い
    // （2026-09-29 に最後の holo_*・difference_* を外した）。

    // --- Geometry cache（H7 compute_cell_geometry 出力、ステップ間保持）---
    double* face_area;      // OWNED [(n_cells+n_ghost) × n_faces] 面面積 A_m [cm²]
    CellField delta_l;      // 特性長 Δl = sqrt(A_cell) [cm]（H10 人工粘性、U4 CFL で使用）

    // --- Derived step-lifetime fields ---
    // Phase 間で保持が必要な中間量。State::allocate() で確保、チェックポイントでの保存は任意
    // （リスタート時は初期化プロローグで再計算可能）。
    CellField c_s;          // 音速 [cm/s]（H15 出力。H10 人工粘性、U4 CFL で使用。
                            //   初期化: State::allocate() 後に H13→H15 で算出。チェックポイント保存推奨）
    CellField D_eff;        // 実効拡散係数 [cm²/s]（C1 出力。U4 dt_cond で使用。
                            //   Scratch ではなく State に保持する理由: C1(Phase 2) → U4(Phase 6) で
                            //   Phase 3-5 のオペレータが Scratch をリセットするため）

    // --- Laser ---
    CellField laser_dep;    // 全ビーム合算後の沈着 [erg]
    double* laser_dep_frac; // OWNED [n_laser_groups * n_cells] per-group レーザー沈着パワー分率 f̂_g [無次元]
                             // （skip mode用、CUDA_KERNELS §5.1e、ステップ間保持。NUMERICS §5.9.3 グループ毎キャッシュ）
                             // n_laser_groups = Builder がビームパラメータ同一性から導出（NUMERICS §5.4）。GXII等の全ビーム同一パラメータ構成では 1
                             // LaserConfig.beams から (θ, F#, profile) の一致で自動グループ化（§8.2 init Step 2）
    bool laser_cache_valid = false; // レーザーキャッシュ有効フラグ。step=0/リスタート直後は false。
                                     // laser_cache_update 後に true。Phase 3 で false の場合 L6 バイパス（CUDA_KERNELS §9 Phase 0 注記）。
                                     // チェックポイント保存対象: laser_dep_frac とともに保存/復元するか、
                                     // リスタート時に false で初期化し初回ステップで full raytrace を強制するか、
                                     // いずれかの方式を選択する。v1.0推奨: リスタート時 false 初期化（実装が単純）

    // --- Cumulative diagnostics（ステップ間で累積、チェックポイントに保存 §4.9 checkpoint）---
    // **命名規約（per-step vs cumulative の二重ライフサイクル）**:
    // デバイス側には同名の per-step アキュムレータ（double* [1] deviceメモリ）が存在し、
    // Phase 0 で cudaMemsetAsync ゼロ初期化される。Phase 1-6 の U2/U1/R8 等が atomicAdd で累積する。
    // Phase 6 で D2H 転送後、ホスト側で以下の累積フィールドに加算する:
    //   State.E_floor_injected += step_E_floor_injected_dev;  // etc.
    // **実装注意**: デバイスper-stepバッファとホスト累積変数は別物。リセットタイミングを混同しないこと。
    // デバイスバッファ名の推奨: `dev_step_E_floor_injected` 等で接頭辞区別。
    //
    // **MPI セマンティクス**: 累積値は **global-per-rank**（全 rank 同一値）で保持する。
    // Phase 6 で per-step デバイス値を D2H した後、MPI_Allreduce(SUM) でグローバル合計を算出し、
    // 全 rank が同一の step 値を State の累積フィールドに加算する（CUDA_KERNELS §9 Phase 6 準拠）。
    // これにより追加の MPI_Reduce なしでエネルギー収支出力が可能。
    // **checkpoint**: rank 0 が time_state/ グループにスカラー値を書き込む（SPECIFICATION §7.4）。
    // 全 rank が同一値を保持するため、rank 数変更時の再合算は不要 — 全 rank が同じ値を復元する。
    //
    // **スレッド安全性**: ホスト累積変数は main time loop の単一スレッドからのみ更新される。
    // HDF5 出力は同一ステップの Phase 6 完了後に実行され、累積変数の read/write が重複しない。
    // 将来非同期出力を導入する場合は、出力前にスナップショットコピーを作成すること。
    double E_safety = 0.0;           // [erg] 伝導安全補正の累積値（NUMERICS §4.2.2, §10.2）。
                                     // 負温度クランプ時のフラックススケーリング非対称性エネルギー ΔE_scaling を加算。
    double E_numerical_loss = 0.0;   // [erg] 退化セル（ρV < 1e-30）で注入できなかったエネルギーの累積値。
                                     // source_injection（§4.7）でゼロ除算ガードに引っかかった場合に加算。
    double E_laser_deposited = 0.0;  // [erg] レーザー沈着エネルギーの累積値（各ステップで laser_dep の総和を加算）
    double E_laser_escaped = 0.0;    // [erg] レーザー脱出エネルギーの累積値（臨界面未到達レイのエネルギー）
    double E_rad_escaped = 0.0;      // [erg] 放射脱出エネルギーの累積値（VACUUM/MARSHAK 境界から脱出した輻射エネルギー）
    double E_floor_injected = 0.0;   // [erg] フロア注入エネルギーの累積値（floor_clamp で追加されたエネルギー）
    double E_pdV_bdry = 0.0;         // [erg] 境界PdV仕事の累積値（NUMERICS §10.2: Σ P_f A_f v_{n,f} Δt）
                                      // Phase 1/5 H(Δt/2) Corrector 後にホスト側で計算・加算
    double E_Marshak_in = 0.0;       // [erg] Marshak境界入射エネルギーの累積値（NUMERICS §10.2: Σ (a_eV c/4) T_{r,f}⁴ A_f Δt）
    double E_solver = 0.0;           // [erg] Hypre残差エネルギーの累積値（v1.0=0、Hypre有効時のみ非ゼロ。NUMERICS §10.2）

    // --- Error ---
    DeviceErrorFlags* error_flags;  // OWNED [1] deviceメモリ。GPUカーネルからのエラー報告（§10.1）。
                                    // State::allocate() で cudaMalloc、各ステップ開始時に cudaMemsetAsync で 0 クリア。
                                    // カーネル完了後に cudaMemcpy D2H でホスト側コピーを取得しチェック

    // --- Scratch ---
    Scratch scratch;        // 一時ワークスペース（§5.5）。各Strangオペレータが排他的に使用
};
```

Runtime macro overlays for 2D_RZ multiblock hydro are runtime `State`
subobjects, not checkpoint/HDF5 schema.  `src/hydro/pole_angular_derefine.{cuh,cu}`
owns the I1-B polar-shell angular overlay: dyadic one-row macro descriptors,
proactive criterion-driven span maintenance, reactive same-pole active-child
span extension, the shared member/inactive cell masks, boundary-node masks,
flattened macro boundary loops, and pressure-work impulse buffers.  CSR mesh
topology remains unchanged.  Hydro, CFL, path admissibility, compatible
force/work, CSW edge AV, ALE remap, and remap audit/fixup paths consume the
module's inactive mask or effective active mask instead of discovering members
independently.

> **LaserMesh の所有権**：LaserMesh は **Laser モジュールが管理**する内部データ構造であり、
> State のメンバーではない。Laser::Mesh が `init()` 時に確保し、ステップ毎に更新する。
> LaserMesh のフィールド（`n_hat`, `Te`, `Zbar`, `grad_n_hat`, `deposit`）は
> State.laser_dep への転写後に参照されない（NUMERICS §5.7.1）。

### 5.3 PhotonPool（SoA粒子プール）— 退役

モンテカルロ輻射の光子粒子プール（`PhotonPool`・`ParticleMode`・`ParticleStatus`・容量と成長の戦略）。2026-09-29 にコードとともに
退役し、本節の記述を `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。State に粒子プールは無い。

### 5.4 GPUレイアウト方針

- 原則 **SoA**（coalesced access）：32スレッドが連続アドレスを読む
- （退役したモンテカルロ輻射の粒子のソート・compaction・再同定・タリー集約の方針は §5.3 と同じく
  `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した）

> **SoAスコープの明確化**：
> - Field<> は各物理量が独立した連続配列として格納されるため、実質的に SoA 相当
> - したがって AoS→SoA 変換は不要（設計時点で SoA を採用済み）
> - M15以降の性能最適化ではカーネルチューニング・メモリアクセスパターン最適化に注力

### 5.5 Scratch（一時ワークスペース）

```cpp
struct Scratch {
    void*   buffer;         // 汎用一時バッファ（deviceメモリ）
    size_t  buffer_size;    // 確保済みバイト数

    // 初期化時に全演算子の最大必要量を調査し、maxで確保
    // 主な使用者：
    //   - CUB reduction (エネルギー収支)            : ~256 bytes
    //   - Kershaw 9点ステンシル係数一時配列          : ~9 * n_cells * 8 bytes
    //   - ALE Jacobi反復の中間ノード座標            : ~2 * n_nodes * 8 bytes
    //   - vol_old (PdV work用体積スナップショット)  : ~n_cells * 8 bytes
    //   - v_r_old, v_z_old (速度スナップショット)  : ~2 * n_nodes * 8 bytes
    //   - x_r_old, x_z_old (座標スナップショット)  : ~2 * n_nodes * 8 bytes
    //   - P_i_old, P_e_old, Q_old (圧力スナップショット) : ~3 * n_cells * 8 bytes
    //     Predictor前に vol/v/x/P/Q → *_old をコピーし、
    //     Corrector H5 が v^n + Δt·a^{pred}、H6 が r^n + Δt·v、
    //     H11/H12 が ΔV = V^{corrector} - V^n、P_mid = (P^n+P^{pred})/2 を計算
};
```

**確保戦略**：`State::init()` 時に全モジュールの `scratch_requirement()` を呼び出し、
最大値で一括確保する。ステップ中は再確保しない（固定サイズ）。
各演算子は `scratch.buffer` を自身のデータ型にキャストして使用する。
同一ステップ内で複数演算子が同時に scratch を使うことはない（逐次実行のため）。

**サブアロケーション・アリーナ**：

```cpp
struct ScratchArena {
    void*  base;       // Scratch::buffer の先頭（deviceメモリ、256-byte aligned）
    size_t offset;     // 現在のオフセット [bytes]（alloc() で前進）
    size_t capacity;   // 全容量 [bytes]（初期化時に確定、以後不変）

    // アライメント付きサブアロケーション
    template<typename T>
    T* alloc(int count) {
        size_t align = alignof(T);
        offset = (offset + align - 1) & ~(align - 1);  // アライメント
        T* ptr = reinterpret_cast<T*>(static_cast<char*>(base) + offset);
        offset += sizeof(T) * count;
        assert(offset <= capacity);  // オーバーフローチェック
        return ptr;
    }

    void reset() { offset = 0; }  // オペレータ開始時にリセット
};
```

各 Strang オペレータの開始時に `arena.reset()` を呼び、オペレータ内では
`arena.alloc<double>(n)` で一時バッファを確保する。
オペレータ間でスクラッチメモリは共有されない（排他使用）。
`ScratchArena` は `Scratch::buffer` のサブ領域を返すだけであり、
`cudaMalloc` / `cudaFree` は発生しない（ゼロオーバーヘッド）。

**scratch メモリ要求インターフェース**：

```cpp
// 各モジュールが必要とする scratch メモリの申告
struct ScratchRequirement {
    size_t bytes;           // 必要バイト数
    size_t alignment = 256; // アライメント (CUDA 推奨)
    const char* label;      // デバッグ用ラベル ("CUB_prefix_sum", "Kershaw_temp", etc.)
};
// 各モジュールは static メソッドで要求量を申告:
//   static ScratchRequirement Radiation::scratch_requirement(const Config& cfg);
//   static ScratchRequirement Hydro::scratch_requirement(const Config& cfg);
//   static ScratchRequirement Conduction::scratch_requirement(const Config& cfg);
//   static ScratchRequirement ALE::scratch_requirement(const Config& cfg);
// 初期化時に全モジュールの max(bytes) を確保し、Scratch::buffer に割り当て
// 典型値 (80K cells, G=16): CUB prefix_sum ≈ 10 MB, Kershaw ≈ 5.8 MB, ALE ≈ 1.3 MB
// （モンテカルロ輻射の粒子ソート用の CUB RadixSort temp ≈ 24×N_p と SoA double buffer ≈ 92×N_p は 2026-09-29 に退役）
```

### 5.6 GPU実行モデル

#### 5.6.1 カーネル起動設定

| カーネル種別 | block_size | `__launch_bounds__` | grid_size | 備考 |
|------------|-----------|-------------------|-----------|------|
| cell-based（Hydro, EOS, Tally集約） | **256** | `(256, 4)` | `(n_cells + 255) / 256` | レジスタ <32、occupancy ≥75%。CUDA_KERNELS §2 |
| node-based（座標更新, 加速度） | **256** | `(256, 4)` | `(n_nodes + 255) / 256` | CUDA_KERNELS §2.2 |
| ray-based（Laser ray trace） | **64** | `(64, 16)` | `(n_rays + 63) / 64` | ~40-46 reg、warp発散対策でblock小。CUDA_KERNELS §5.2 |
| Kershaw stencil build | **256** | `(256, 2)` | `(n_cells + 255) / 256` | ~45 reg、compute-bound で occupancy 低下を許容。CUDA_KERNELS §4.2 |
| pack/unpack（ハロー交換） | **256** | — | `(n_halo + 255) / 256` | メモリバウンド、レジスタ少。CUDA_KERNELS §1.7 |

> **`__launch_bounds__` の役割**：コンパイラにレジスタ割り当て上限を伝え、指定した
> `min_blocks` 数を保証する。例：`__launch_bounds__(128, 8)` は1SMあたり最低8ブロック
> （= 1024スレッド = 50% occupancy）を保証し、レジスタ数を 65536/1024 = 64 以下に制約する。
> 全主要カーネルに `__launch_bounds__` を付与し、コンパイラによるレジスタスピルを防止する
> （CUDA_KERNELS §10.3 参照）。

#### 5.6.2 CUDAストリーム管理

```cpp
struct StreamManager {
    cudaStream_t compute;       // 主計算ストリーム（全演算子のカーネル実行）
    cudaStream_t comm;          // 通信用ストリーム（halo pack/unpack + MPI）
    cudaStream_t utility;       // ユーティリティ（diagnostics, I/O staging）

    cudaEvent_t evt_compute_done, evt_comm_done, evt_utility_done;
    // cudaEventCreateWithFlags(cudaEventDisableTiming) — timing不要でオーバーヘッド最小化
    // cudaStreamWaitEvent() でストリーム間依存を設定する（§5.6.2 compute-comm overlap）
};
```

**方針**：
- v1.0では **3ストリーム**：compute, comm, utility
- **計算-通信オーバーラップ**（NUMERICS §12.5.5 準拠）：
  - **v1.0既定**：逐次実行（overlap なし）。`cudaStreamSynchronize(compute)` 後に comm 開始
  - **Phase B**（設計のみ。有効化の設定だった `Parallel.gpu_optimization.compute_comm_overlap` は受理して無視される）：
    1. 内部セル（ゴースト非依存）のカーネルを compute ストリームで起動
    2. 同時に境界セルのハローパック → MPI通信 → ハローアンパックを comm ストリームで実行
    3. 両ストリーム同期後、境界セル（ゴースト依存）のカーネルを compute ストリームで起動
  - **適用可能演算子**：Hydro（コーナー力）、Conduction（Kershaw/tridiag）、Radiation（セルベースのカーネル）
  - **適用不可**：Laser（全rank複製でハロー交換なし）
  - **セル分類**：初期化時に `uint8_t cell_zone[n_cells]` を生成（0=内部、1=境界）。
    Kershaw 9点ステンシルの近接1層要件に基づき、MPI区画境界から1層以内の所有セルを境界セルとする（ghost_layers=1 前提）
  - **性能見積もり**：4 GPU時、ステップあたり ~1.2 ms の隠蔽（2–3% 改善）。
    GPU数増加で通信時間が支配的になるため、効果は相対的に増大する

#### 5.6.3 Persistent Warp 実行モデル — 退役

IMC 輸送カーネル（R8）の実行モデル（NUMERICS §6.6）。2026-09-29 に退役し、本節の記述を
`retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。

#### 5.6.4 GPUメモリ予算

初期化時にデバイスメモリの使用量を推算し、空きメモリの **85%** を上限とする。

```
メモリ予算の内訳（代表的な 2D_RZ、nr=200, nz=400, G=16群, 2材料）：

State fields:
  cell fields (ρ,m,V,Te,Ti,ee,ei,Pe,Pi,Q) : 10 × 80K × 8B  =   6.4 MB
  cell×group fields (rad_E, rad_dep)       :  2 × 80K × 16 × 8B = 20.5 MB
  cell×mat fields (volFrac)                :  1 × 80K × 2 × 8B  =  1.3 MB
  node fields (x_r, x_z, v_r, v_z)        :  4 × 80.6K × 8B    =  2.6 MB
  laser_dep [N_cell]                        :  1 × 80K × 8B      =  0.6 MB
  --- State subtotal                                              ~ 31 MB

LaserMesh (128×256):
  5 fields × 129 × 257 × 8B                                     ~  1.3 MB

EOS/Opacity tables (device):
  SESAME: 2材料 × (301+304) × (n_ρ×n_T) × 8B  (73×41=24K)     ~  3 MB
  IONMIX opacity: 2材料 × G × (n_ρ×n_T) × 8B                   ~  6 MB
  Planck fraction table: 200 pts × 8B                            ~  0 MB
  --- EOS/Opacity subtotal                                       ~  9 MB

CommBuffers + Scratch:
  halo + scratch                                                 ~ 50 MB

--- 典型合計（この表の項目）                                       ~ 0.1 GB
```

（FLD・\(S_N\) の作業配列（群ごとの係数・三重対角系・\(S_N\) の角度束など）はこの表に含まない。退役したモンテカルロ輻射では
光子粒子プールが全体の 95% 以上を占めていた（100 粒子/セル/群で ~12 GB）— その見積もりは
`retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。）

**メモリ不足時の対応**：
1. `State::init()` で `cudaMemGetInfo` により空きメモリを取得
2. 推定使用量が空きの 85% を超える場合、警告を出力
（退役したモンテカルロ輻射には、粒子プールの初期容量の自動縮小・ピーク推定による停止・緊急 Russian roulette の手順もあった。）

---

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
