<!-- 分割元: docs/ARCHITECTURE.md | このファイルは参照用です。原本（docs/ARCHITECTURE.md）が権威です。 -->
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
// HDF5 collective write 後にホストバッファを解放する。
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
    // → 同一セルの全群が連続 → IMC/DDMCカーネルでキャッシュ効率が良い

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
> ell_ddmc, volFrac, hydro_active）は `n_cells + n_ghost` 要素で確保する。
> ここで `n_ghost` は `PartitionInfo.n_ghost_cells`（ゴーストセル総数）のエイリアスである。
> `PartitionInfo.ghost_layers`（レイヤー数 = 1）とは異なるので注意。
> 2D_RZ 1層ハロー: n_ghost_cells = 2×(nr_local+nz_local)+4。単一GPU時は `n_ghost_cells=0`。
> カーネルシグネチャで `[n_cells + n_ghost]` と注記されたフィールドは ghost 込みバッファを前提とする。
> `[n_cells]` のみ注記されたフィールド（mass, ee, ei, Cv_e, Cv_i, delta_l 等）は owned セル専用で
> ghost 領域は確保しない。FieldG 型（rad_dep, rad_E 等）は owned セル `[n_cells × G]` のみ。
> ただし ddmc_candidate, ddmc_mode は `[(n_cells+n_ghost) × G]` で確保する（R2/R3 がゴーストセルを処理するため）。

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
    int    ale_last_applied_step = -1; // 1D V3 ALE cadence/min-step gate 用

    // --- 出力タイミング状態（時間間隔ベース出力用）---
    // checkpoint 保存・復元対象（SPECIFICATION §7.4 output_state/）。
    // -1.0 = 対応する X_every_s が無効。
    double t_next_plot = -1.0;       // [s] 次回plot出力時刻
    double t_next_history = -1.0;    // [s] 次回history出力時刻
    double t_next_checkpoint = -1.0; // [s] 次回checkpoint出力時刻

    // --- ファクトリ関数 ---
    // Config と PartitionInfo から全フィールドを確保し初期化する。
    // 確保対象：全 CellField/NodeField/CellFieldG/CellFieldMat、
    //           hydro_active、PhotonPool、Scratch、DeviceErrorFlags
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
    CellFieldG rad_E;       // エネルギー密度推定量 [erg/cm³]（NUMERICS §10.1: IMC track-length、§7.6: DDMC residence）
    CellFieldG rad_E_tally; // track-length/residence推定量の生タリー [erg·cm]（R8/R9で累積、R10で rad_E へ変換）
                            // ステップ冒頭ゼロクリア必須。legacy変換: rad_E[i,g] = rad_E_tally[i,g]/(V_i*×c×Δt)
                            // difference変換: rad_E[i,g] = difference_E_ref[i,g] + rad_E_tally[i,g]/(V_i*×c×Δt)
    CellFieldG rad_dep;     // セル→物質の放射交換エネルギー [erg]（セルあたり、ステップ中の累積）
                            // IMC/DDMC の gross absorption tally。PGRW は IMC branch として同じ tally に書き込む
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
    CellField difference_W; // optional difference reference weight W_i [dimensionless]
    CellFieldG difference_E_ref;      // optional time-average reference density used by PR7 rad_E reconstruction [erg/cm³]
    CellFieldG difference_residual_E; // optional signed residual density diagnostic [erg/cm³]
    double* rad_mom_dep;    // 運動量沈着（診断、NUMERICS §7.8）[dyne·s/cm³]
                            // 2D_RZ: [N_cell × 2]（R,Z成分）、1D_SPH: [N_cell × 1]（径方向のみ）
                            // SPECIFICATION §7.2: radiation/momentum_dep float64[N_cell, 2]
    int8_t* bc_type_rad;    // OWNED [4] deviceメモリ。輻射BC種別（面順: R_low,R_high,Z_low,Z_high）
                            // 0=vacuum, 1=reflect, 2=marshak。BoundaryConfig 文字列から init() で変換（CUDA_KERNELS §6.0a R3 参照）
    int8_t* ddmc_candidate; // OWNED [(n_cells+n_ghost) × G] deviceメモリ。DDMC候補フラグ（R2出力、§5.2規約準拠）
    int8_t* ddmc_mode;      // OWNED [(n_cells+n_ghost) × G] deviceメモリ。transport mode 最終判定（R3出力）。
                            // 0=IMC, 1=DDMC, 2=RW（legacy enum値。現行 PGRW 実装は 2 を生成しない）, 3=Diffusion（予約。PR1では生成しない）
    double* face_current_prev; // IMC-owned [(n_cells+1) × G] 前ステップ signed face current [erg]。diffusion用
    double* face_current_step; // IMC-owned [(n_cells+1) × G] 現ステップ signed face current [erg]。HOLO consistency residual でも使用
    double* face_current_in;   // IMC-owned [(n_cells+1) × G] 現ステップ IMC→diffusion positive source [erg]
    double* face_current_out;  // IMC-owned [(n_cells+1) × G] diffusion→IMC outgoing source [erg]
    std::vector<uint8_t> holo_core_mask, holo_core_prev_mask; // LO material-coupling mask [N_cell]（legacy名）
    CellFieldG holo_E_LO; // global HOLO low-order physical radiation energy density [erg/cm³];
                          // persistent across steps; initialized to zero on allocation/resize
    CellFieldG holo_consistency_source; // same-step HOLO corrector RHS source [erg/s]
    CellFieldG holo_Prr;  // passive HOLO high-order radial pressure moment [erg/cm³]
    CellFieldG holo_chi;  // passive HOLO Prr/E closure diagnostic [dimensionless]
    CellFieldG holo_chi_filtered; // runtime filtered QD closure history [dimensionless]
    CellFieldG holo_Prr_coverage; // passive HOLO Prr covered track-length fraction [dimensionless]
    bool holo_lo_source_valid; // 現 radiation stage の LO source ownership が commit 済みなら true
    bool holo_ale_invalidated; // ALE remap が LO closure を無効化済みなら true。次 HOLO solve で rad_E/LTE から再初期化
    bool particle_sort_cache_invalidated; // ALE/mesh re-ID 後、次 census sort で cell_id order を再構築する
    PhotonPool photons;     // IMC + DDMC 粒子プール（legacy enum として RW 値は保持）

    // --- Geometry cache（H7 compute_cell_geometry 出力、ステップ間保持）---
    double* face_area;      // OWNED [(n_cells+n_ghost) × n_faces] 面面積 A_m [cm²]（§7.7.4, §7.3）。
                            // n_ghost込み: R2/R3 がゴーストセルの face_area を参照（§5.2 規約準拠。H7 は n_cells のみ計算、ゴーストはハロー交換で充填）
    CellField delta_l;      // 特性長 Δl = sqrt(A_cell) [cm]（H10 人工粘性、U4 CFL で使用）
    CellField ell_ddmc;     // DDMC代表長 ℓ_i [cm]（NUMERICS §7.1.1: 2×min_f d_center→face。R2 DDMC判定で使用。delta_l とは異なる量）

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

### 5.3 PhotonPool（SoA粒子プール） [RETIRED — legacy IMC/DDMC particle pool]

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

### 5.4 GPUレイアウト方針

- 原則 **SoA**（coalesced access）：32スレッドが連続アドレスを読む
- PhotonPool の alive compaction・セルソート・モード分離は **Composite Key Sort**（NUMERICS §6.5、CUDA_KERNELS §0.5）で一括実行：
  合成キー生成 → CUB RadixSort → Fused SoA Gather の3サブステップで、
  dead粒子除去 + cell\_id順ソート + IMC/DDMC分離を単一パスに融合
- 1D Lagrangian mesh 後の finite-position 粒子 cellId 再同定は
  `radiation/particle_reid.cu` が担当する。kernel は `PhotonPool::cell_id` のみを更新し、
  DDMC/NaN sentinel 粒子を保持する。coupling driver は re-ID 後に
  `particle_sort_cache_invalidated` を立て、次の radiation operator の
  sort/partition が stale cell order を使わないようにする
- **実行タイミング**：Radiation演算子冒頭、ソース粒子投入後。dead粒子は前ステップから遅延除去される
- タリー集約は §4.5 `Rad::Tally` 準拠。v1.0 では Stage 1（warp-level `__match_any_sync` 集約）
  + Stage 3（global atomicAdd）を使用。shared memory 不使用（レジスタのみ）

> **SoAスコープの明確化**：
> - PhotonPool は SoA レイアウト（§5.3で定義済み）
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
    //   - CUB RadixSort (Composite Key Sort)        : ~24 * n_particles bytes
    //   - CUB reduction (エネルギー収支)            : ~256 bytes
    //   - Kershaw 9点ステンシル係数一時配列          : ~9 * n_cells * 8 bytes
    //   - ALE Jacobi反復の中間ノード座標            : ~2 * n_nodes * 8 bytes
    //   - 粒子ソート用一時配列 (CUB RadixSort)     : ~key_size * n_particles bytes
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
// 典型値 (80K cells, 10M particles, G=16):
//   CUB RadixSort temp ≈ 24×N_p ≈ 240 MB (dominant), SoA double buffer ≈ 92×N_p ≈ 920 MB,
//   CUB prefix_sum ≈ 10 MB, Kershaw ≈ 5.8 MB, ALE ≈ 1.3 MB
```

### 5.6 GPU実行モデル

#### 5.6.1 カーネル起動設定

| カーネル種別 | block_size | `__launch_bounds__` | grid_size | 備考 |
|------------|-----------|-------------------|-----------|------|
| cell-based（Hydro, EOS, Tally集約） | **256** | `(256, 4)` | `(n_cells + 255) / 256` | レジスタ <32、occupancy ≥75%。CUDA_KERNELS §2 |
| node-based（座標更新, 加速度） | **256** | `(256, 4)` | `(n_nodes + 255) / 256` | CUDA_KERNELS §2.2 |
| particle-based IMC（Persistent Warp） | **128** | `(128, 8)` | `n_sm × 8`（固定、SM数依存） | ~60 reg/thread、50% occupancy。NUMERICS §6.6、CUDA_KERNELS §6.4 |
| particle-based DDMC（History-based） | **128** | `(128, 16)` | `(n_ddmc + 127) / 128` | ~30 reg/thread、100% occupancy。CUDA_KERNELS §6.5 |
| ray-based（Laser ray trace） | **64** | `(64, 16)` | `(n_rays + 63) / 64` | ~40-46 reg、warp発散対策でblock小。CUDA_KERNELS §5.2 |
| Kershaw stencil build | **256** | `(256, 2)` | `(n_cells + 255) / 256` | ~45 reg、compute-bound で occupancy 低下を許容。CUDA_KERNELS §4.2 |
| source R4/R5（エネルギー計算+prefix-sum） | **256** | `(256, 4)` | `(n_cells * G + 255) / 256` | cell-based、NUMERICS §6.2、CUDA_KERNELS §6.2-§6.3 |
| source R6（体積ソース fill） | **128** | `(128, 8)` | `(n_new_particles + 127) / 128` | particle-based、CUDA_KERNELS §6.3 |
| source R13（Marshak境界ソース） | **128** | `(128, 8)` | `(n_marshak + 127) / 128` | particle-based、Marshak BC適用時のみ。CUDA_KERNELS §6.0h |
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
  - **Phase B**（`Parallel.gpu_optimization.compute_comm_overlap=True` で有効化）：
    1. 内部セル（ゴースト非依存）のカーネルを compute ストリームで起動
    2. 同時に境界セルのハローパック → MPI通信 → ハローアンパックを comm ストリームで実行
    3. 両ストリーム同期後、境界セル（ゴースト依存）のカーネルを compute ストリームで起動
  - **適用可能演算子**：Hydro（コーナー力）、Conduction（Kershaw/tridiag）、Radiation（Fleck/mode judge等のセルベースカーネル）
  - **適用不可**：粒子輸送カーネル（任意セル横断のため内部/境界分割が困難）、Laser（全rank複製でハロー交換なし）
  - **セル分類**：初期化時に `uint8_t cell_zone[n_cells]` を生成（0=内部、1=境界）。
    Kershaw 9点ステンシルの近接1層要件に基づき、MPI区画境界から1層以内の所有セルを境界セルとする（ghost_layers=1 前提）
  - **性能見積もり**：4 GPU時、ステップあたり ~1.2 ms の隠蔽（2–3% 改善）。
    GPU数増加で通信時間が支配的になるため、効果は相対的に増大する

#### 5.6.3 Persistent Warp 実行モデル

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
  rad_mom_dep [N_cell × 2]                  :  1 × 80K × 2 × 8B  =  1.3 MB
  --- State subtotal                                              ~ 32 MB

PhotonPool (100 particles/cell/group):
  80K × 16 × 100 × 93B                                          ~11.9 GB

LaserMesh (128×256):
  5 fields × 129 × 257 × 8B                                     ~  1.3 MB

EOS/Opacity tables (device):
  SESAME: 2材料 × (301+304) × (n_ρ×n_T) × 8B  (73×41=24K)     ~  3 MB
  IONMIX opacity: 2材料 × G × (n_ρ×n_T) × 8B                   ~  6 MB
  Planck fraction table: 200 pts × 8B                            ~  0 MB
  --- EOS/Opacity subtotal                                       ~  9 MB

CommBuffers + Scratch:
  halo + emigrant + scratch                                      ~ 50 MB

--- 典型合計                                                      ~12.4 GB
```

> **粒子数がメモリ支配的**：PhotonPool が全体の 95%+ を占める。
> `particles_per_cell_group` の設定がメモリ使用量を決定する。
> **ピーク時メモリ**（Composite Key Sort 中）：pool (92N) + double buffer (92N) + comp_key/perm (8N) + CUB temp (24N) = **216 bytes/particle**。
> 定常時は pool (92N) + CUB temp 分の Scratch ≈ **96 bytes/particle**。
> A100 (80GB) では定常 ~500M、**ピーク ~350M** 粒子が上限目安。
> V100 (32GB) では定常 ~200M、**ピーク ~140M** 粒子が上限目安。
> `max_pool_size` は定常値ではなくピーク値で設定すること。

**メモリ不足時の対応**：
1. `State::init()` で `cudaMemGetInfo` により空きメモリを取得
2. 推定使用量が空きの 85% を超える場合、警告を出力
3. PhotonPool の initial_capacity を空きメモリに収まるよう自動縮小
4. **縮小後もピーク推定（216 B/particle × capacity + 固定フィールド + Scratch）が空きの 95% を超える場合は ERROR 停止**（初期化時に OOM を検出。ランタイムのnondeterministic abortを防止）
5. 実行中に PhotonPool 拡張が不可能な場合、緊急 Russian roulette を発動する（NUMERICS §6.4）：
   \(w_{cutoff}\) を一時的に \(\max(w_{cutoff} \times 10^3,\; 10^{-4})\) に引き上げ、全 alive 粒子に対して間引き判定を再実行して粒子数を抑制する

---

