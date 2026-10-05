# TENRYU — CUDA_KERNELS.md
CUDAカーネルの設計仕様書。各カーネルのスレッド/ブロック構成、メモリアクセスパターン、
レジスタ圧力、最適化戦略、および1タイムステップ内のカーネル起動シーケンスを定義する。

**参照GPU**: NVIDIA A100 80GB SXM4（108 SM、65536 reg/SM、164 KB shared/SM、2048 threads/SM）
**最低要件**: Compute Capability 6.0+（`atomicAdd(double*)`）。

> 退役したモンテカルロ輻射（IMC・DDMC・ランダムウォーク・HOLO・difference 定式化）のカーネル（R1〜R16、Composite Key Sort、
> census combing、粒子のセル再同定 U7、不透明度の前計算 U9、粒子の rank 間移動 P5/P6）は 2026-09-29 にコードとともにビルドから外し、
> 本書にあったその設計を `retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md` へ移した（節番号は残し、注記だけを置く）。

---

## 0. 設計原則

### 0.1 カーネル分類と起動パラメータ

| カーネル種別 | block_size | 根拠 | 典型 occupancy |
|------------|-----------|------|---------------|
| **cell-based** | 256 | メモリバウンド、低レジスタ（<32） | ≥75% |
| **node-based** | 256 | 同上 | ≥75% |
| **particle-based** | 128 | 燃焼の α 粒子 Monte Carlo（`burn/mc_transport.cu`）。レジスタ圧力高、50%で十分 | ≥50% |
| **ray-based** | 64 | 分岐多、レイ長の分散大 | ≥25% |
| **reduction** | 256 | CUB標準 | — |
| **pack/unpack** | 256 | メモリバウンド | ≥75% |

grid_size は原則 `(N + block_size - 1) / block_size` で算出する。
**末尾スレッドガード必須**: ceil-grid 起動する全カーネルは先頭で `int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= N) return;` を実行すること。以下の疑似コードでは省略する場合があるが、実装時には必ず挿入する。
**例外**:
- **CUB ライブラリ呼び出し**（RadixSort, Partition, Reduce 等）: CUB 内部がグリッドサイズを自動決定

### 0.2 メモリアクセス方針

- **SoA（Structure of Arrays）**が原則。32スレッドが連続アドレスを読む（coalesced access）
- セルデータへの読み込み（opacity等）は `__ldg()` intrinsic で L2 キャッシュ経由のread-only アクセス
- 粒子→セルデータ参照はランダムアクセスだが、**セルID順ソート**（§0.5）で局所性を改善
- タリー（`rad_dep`, `rad_E_tally`）は `atomicAdd(double*)` で集約

### 0.2b 型安全・オーバーフロー制約

- **粒子数**: `N_p_total`, `N_total` は `int`（int32）。v1.0 の最大粒子数は 2^31-1 ≈ 21.5億。社内の性能記録 P1-P3 の上限 ~10M 粒子では十分。将来 >2B 粒子が必要な場合は `int64_t` 移行を検討
- **セルインデックス算術**: `cell_id * G + group_id` は int32 算術。有効範囲: n_cells × G < 2^31。v1.0 想定の n_cells=125K, G=48 では ~6M（十分）
- **step_base**: `(uint64_t)step << 40`。step < 2^24 = 16,777,216 で有効（ICF標準: 10K-100K steps）
- **rng_counter**: `uint32_t`。1ステップあたりの最大 RNG 描画数 ~10^3/粒子で 2^32 には到達しない
- **atomicAdd(double*)**: CC 6.0+ 必須（§0.1）。全物理タリーは double 精度
- **face_bc_type エンコーディング**: 伝導（C2）と輻射（R3/R8/R9）で異なるエンコーディングを使用する。
  - **C2（Kershaw伝導）**: `uint8_t face_bc_type[4]` — `0=内部/MPI, 1=reflect, 2=vacuum`
  - **R3/R8/R9（輻射）**: `int8_t face_bc_type[4]` — `0=VACUUM, 1=REFLECT, 2=MARSHAK, 3=AXIS`（§6.4.3 参照）
  - 実装時に混同しないこと。C2 と R3/R8/R9 は異なる配列を参照する
  - **面インデックス規約（全カーネル共通）**:
    - **2D_RZ**: `k=0: R_left (r_lo面), k=1: R_right (r_hi面), k=2: Z_bottom (z_lo面), k=3: Z_top (z_hi面)`
    - **1D_SPH**: `k=0: inner (r_lo面), k=1: outer (r_hi面)`
    - **[n_cells × n_faces]** 配列のストライド: `idx = cell_id * n_faces + k`（cell-major）
    - この規約は H3, H7, H9, R3, R8, R9, C2, L4 等、面配列を使用する全カーネルで共通
  - **H16（Hydro境界）**: `int8_t bc_type[4]` — `0=FREE, 1=FIXED, 2=REFLECT, 3=PRESSURE, 4=AXIS`
    - 面順序は面インデックス規約に従う: `bc_type[0..3] = {R_left, R_right, Z_bottom, Z_top}`
    - 1D_SPH: `bc_type[0..1] = {inner, outer}`。inner は常に REFLECT（球対称中心）

### 0.3 CUDAストリーム方針（ARCHITECTURE §5.6.2 準拠）

v1.0では3ストリーム：`compute`, `comm`, `utility`。
ただし **v1.0 では全物理カーネルを単一の `compute` ストリーム上で逐次起動する**（§9 参照）。
`comm`/`utility` ストリームは MPI 通信前後の D2H/H2D 転送およびユーティリティ操作用に予約されるが、
v1.0 では計算-通信オーバーラップは無効であり、実質的に `compute` ストリームのみが使用される。
各演算子の開始前に `cudaStreamSynchronize(compute)` で前演算子の完了を保証する。
multi-stream 化の際に必要なイベント依存は §9 の「将来 multi-stream 化する場合」注記に列挙。

### 0.4 Scratch バッファ共有（ARCHITECTURE §5.5 準拠）

全カーネルの一時メモリは `Scratch::buffer` を共有する。
Strang splitting 内の演算子は逐次実行のため、同時使用は発生しない。
各モジュールの `scratch_requirement()` で最大量を報告し、初期化時に一括確保する。
主な消費先: Hydro F_r/F_z (2 × n_nodes × 8B)、
Conduction Te_old (n_cells × 8B、Hypre パスの E_solver 用、§4.5 step 3)。
（退役したモンテカルロ輻射の RadixSort (~24 × N_particles) と DDMC の Kershaw stencil (9 × (n_cells+n_ghost) × G × 8B) は
2026-09-29 に無くなった。）
Strang splitting 内は逐次のため同時使用なし。

> **排他規約**：Scratch を使用するカーネル・CUB 呼び出しは **compute stream** 上でのみ実行する。
> utility/comm stream から Scratch を参照すると、逐次実行の保証が崩れ競合が発生する。

### 0.5 Composite Key Sort 戦略（R7+R11+R14 融合）— 退役

光子粒子のソート・compaction・モード分離と census combing の GPU パイプライン。記述は
`retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md` へ移した。

### 0.6 エラーフラグプロトコル（ARCHITECTURE §10.1 準拠）

`DeviceErrorFlags`（`src/core/device_error_flags.cuh`、欄の一覧は ARCHITECTURE §10.1）を引数に取るのは、
次のカーネルだけである。異常を見つけてもスレッドは中断せず、フラグを立てて（`infinite_loop` と
`unresolved_quadrature` は数えて）処理を続け、段の後に host が判定する。
- レーザーの光線追跡：`ray_trace_1d_characteristic`・`ray_trace_1d_sph`・`fast_trace_1d_kernel`・
  `radial_absorption_1d_kernel`（1D）、`ray_trace_2d`・`ray_trace_3d`（2D_RZ）。上限回数に達した光線
  （`infinite_loop`）、非有限の光線状態（`nan_particle`）、不正な補間状態（`invalid_cell`、光線は未吸収として扱う）、
  求積のパネル上限での受理（`unresolved_quadrature`、特性線法のみ）
- 不透明度の評価：`opacity.cu` の評価カーネル（`opacity_out_of_range`：ρ ≤ 0 をクランプ）

設計時にここへ並べていた他の検査（EOS の `eos_ion_negative`・`eos_newton_nonconverge`、`sound_speed_negative`、
`volfrac_degenerate`、`negative_source_dep`、`mesh_tangle`、`emigrant_*`）はフラグとして存在しない。
代わりの仕組み（段の前後の状態量の非有限値と温度の行き過ぎの GPU 集約、エネルギー収支、床クランプの回数、
1D の体積が非正になったときの \(\Delta t/2\) でのやり直し、EOS 逆変換の結果構造体のフラグ）は ARCHITECTURE §10.1 を参照。

---

## 1. カーネル一覧（モジュール別）

> **N列の規約**: N = CUDA grid サイズ計算に使用するスレッド数（`grid = (N + block-1) / block`）。
> 「G群ループ内包」と記載のカーネルは、1スレッドが全G群を内部ループで処理する（N ≠ N×G）。

### 1.0 現行 1D 実装のカーネル（2026-09-29 のソースから）

§1.1〜§1.7 の表と §2〜§8 の詳細は設計時の ID（H1〜H16、A1〜A5、C1〜C4、L1〜L7、U1〜U9、P1〜P6）で書かれており、
名前が実装と違うもの、実装に存在しないもの（本文の各節に注記）がある。1D の本番経路で実際に起動されるカーネルは
次のファイルにある（`grep __global__` で一覧できる）：

| モジュール | ファイル | 主なカーネル |
|---|---|---|
| 流体（Lagrange） | `hydro/hydro_1d.cu` | `predictor_update_kernel`・`corrector_update_kernel`・`compute_acceleration_1d_kernel`・`compute_density_kernel`・`energy_update_with_old_volume{,_2t}_kernel`・`compatible_energy_update_1d_kernel`・`enforce_1t/2t_closure_kernel`・`compute_sound_speed_1t/2t{,_split}_kernel`・`apply_qei_transfer_2t_kernel`・`find_nonpositive_volume_1d_kernel`・`follow_void_nodes_1d_kernel`・`signed_energy_cells_kernel`・`hk_dominant_material_1d_kernel`・`hk_velocity_front_cells_1d_kernel`・`hk_velocity_near_front_1d_kernel` ほか |
| 人工粘性 | `hydro/artificial_viscosity.cu` | `compute_node_sigma_1d_kernel`・`compute_q_1d_kernel`（VNR）・`compute_q_csw_1d_kernel`（既定）・`compute_q_riemann{,_compatible}_1d_kernel`・`compute_artificial_heat_1d_kernel`・`add_bulk_viscosity_1d_kernel` |
| 時間刻み | `hydro/cfl.cu` | `cfl_1d_kernel`・`cfl_1d_lineage_argmin_kernel`・`volume_rate_cfl_*_kernel` |
| 電子・イオン熱伝導 | `hydro/conduction.cu`、`hydro/conduction_snb_1d.cu` | `compute_spitzer_deff_1d_kernel`・`compute_1d_face_kappa_kernel`・`conduction_1d_sts_stage{,_kirchhoff,_secant}_kernel`・`conduction_1d_sts_fused_kirchhoff_kernel`・`build_1d_implicit_system{,_kirchhoff}_kernel`・`implicit_conduction_flux_form_kernel`・`ion_conduction_*_kernel`・`snb_*_kernel` |
| 平均電荷 | `materials/zbar_device.cu` | `update_zbar_fields_kernel`（TMAT 以外の材料を含む Thomas–Fermi は host の丸め: `zbar_tf_host_rounding.cuh`） |
| FLD | `radiation/fld_1d_gpu.cu` | `assemble_fld_tridiag_kernel`・`update_matter_kernel`・`fld_grey_*_kernel`（灰色加速）・`compute_fleck_for_fld_kernel`・`compute_marshak_finc_kernel` ほか |
| S\(_N\) | `radiation/sn_ld_1d_gpu.cu`（LD、既定）、`radiation/sn_transport_1d_gpu.cu`（LC 系）、`radiation/sn_dsa_1d_gpu.cu` | `scan_sweep_kernel`・`sweep_kernel`・`moments_kernel`・`p1_*_kernel`（前処理）・`matter_update_kernel`／`sn_sweep_spherical_lc_kernel`・`sn_sweep_spherical_serial_kernel`・`sn_marshak_inflow_1d_kernel` ほか／`assemble_sn_dsa_consistent_kernel` |
| レーザー | `laser/*.cu` | §1.4 の表 |
| 燃焼 | `burn/burn_stage_gpu.cu`、`burn/network_gpu.cu`、`burn/mc_transport.cu`、`burn/corman_diffusion.cu` | `burn_stage_main_1d_kernel`・`burn_network_apply_1d_kernel`・`burn_neutron_*_1d_kernel`・MC α の `spawn/transport/compact_kernel`・多群拡散の `assemble_group_kernel` ほか |
| 安全検査 | `coupling/driver_safety_audit.cu` | `temperature_audit_kernel`・`phase_safety_violation_kernel`・`overshoot_metrics_kernel` |

### 1.1 Hydro（16カーネル）

| ID | カーネル名 | 種別 | N | 主要入出力 |
|----|-----------|------|---|-----------|
| H1 | `hydro_active_update` | cell | n_cells | Te → hydro_active |
| H2 | `compute_node_mass` | cell gather | n_cells | cell mass, node R → R-weighted node mass（§3.2.4）|
| H3 | `compute_area_vectors` | cell | n_cells | node coords → S_{c,k} |
| H4 | `compute_corner_force` | node | n_nodes | x,P,Q → F_r,F_z |
| H5 | `velocity_update` | node | n_nodes | v,F,m → v_new |
| H6 | `position_update` | node | n_nodes | r,v → r_new |
| H7 | `compute_cell_geometry` | cell | n_cells | node coords → V,A_cell,centroid,Δl,face_area |
| H8 | `compute_density` | cell | n_cells | mass,V → rho |
| H9 | `compute_divergence` | cell | n_cells | v,S → dV/dt, div_u |
| H10 | `compute_artificial_viscosity` | cell | n_cells | rho,node_r,node_u,c_s,J → Q,(chi) |
| H11 | `energy_update_ion` | cell | n_cells | P_i,Q,dV/dt,Q_ei → de_i |
| H12 | `energy_update_electron` | cell | n_cells | P_e,dV/dt,Q_ei → de_e |
| H13 | `eos_forward` | cell | n_cells | rho,Te,Ti → ee,ei,Pe,Pi,Cv_e,Cv_i |
| H14 | `eos_inverse` | cell | n_cells | rho,e,species → T（Newton反復、species=0:電子/1:イオン） |
| H15 | `compute_sound_speed` | cell | n_cells | P_e,P_i,rho,Te,Ti,Cv_e,Cv_i,Zbar,A_eff,eos_model → c_s |
| H16 | `apply_hydro_bc` | node/cell | n_boundary | 境界条件適用 |

### 1.2 ALE（5カーネル）

| ID | カーネル名 | 種別 | N | 主要入出力 |
|----|-----------|------|---|-----------|
| A1 | `mesh_quality_check` | cell | n_cells | node coords → q_c |
| A2 | `winslow_jacobi_step` | node | n_nodes | neighbor coords → new coords |
| A3 | `conservative_remap` | cell | n_cells | old/new mesh, fields → remapped fields |
| A4 | `project_cell_velocity_to_nodes` | node | n_nodes | セル中心速度→節点速度（NUMERICS §3.3.4 質量重み投影）|
| A5 | `normalize_volFrac` | cell | n_cells | remap後 volFrac正規化（Σ=1強制 + 退化ガード） |

### 1.3 Conduction（4カーネル）

| ID | カーネル名 | 種別 | N | 主要入出力 |
|----|-----------|------|---|-----------|
| C1 | `compute_spitzer_deff` | cell | n_cells | Te,rho,Zbar,Cv_e,A_eff → D_eff |
| C2 | `kershaw_stencil_build` | cell | n_cells（伝導）/ n_cells+n_ghost（輻射R3用） | coords,D_eff → 9係数/cell |
| C3 | `kershaw_apply` | cell | n_cells | 9点ステンシル×Te → ΔTe |
| C4 | `conduction_1d_tridiag` | cell | n_cells | 3点ステンシル×Te → ΔTe（1D用） |

> **1D の実装**：C1 に当たる `compute_spitzer_deff_1d_kernel` はセルの \(\kappa_{SH}\) を出力し、\(D_\mathrm{eff}\) は時間刻みの
> 見積もりにだけ使う。面の係数は `compute_1d_face_kappa_kernel`（ステップの始めの \(T_e^n\) で評価して固定、NUMERICS §4.2.1）。
> C4 `conduction_1d_tridiag` は存在しない — 1D は STS（`conduction_1d_sts_stage*_kernel`）と、`Numerics.conduction.solver`
> の陰解法（`build_1d_implicit_system*_kernel` で三重対角を組み cuSPARSE `gtsv2` で解き、`implicit_conduction_flux_form_kernel`
> で面の流束の形でエネルギーを記帳する、`conduction.cu`）。

### 1.4 Laser

設計時の L1 `laser_mesh_map`・L2 `compute_density_gradient`・L5 `deposit_lm_to_hydro`・`laser_cache_update` は
実装に存在しない（§5.1 の注記）。現行のカーネル：

| ファイル | カーネル | 役割 |
|---|---|---|
| `laser_mesh.cu` | `map_hydro_to_laser_1d_kernel`・`map_node_material_1d_kernel`・`trace_profile_{plan,count,tail_count,write,tail_write,finish}_1d_kernel` | 1D の写像（2D レーザー格子の節点と、光線追跡用の径方向プロファイル節点） |
| `laser_mesh.cu` | `compute_radial_gradient_kernel`・`compute_gradient_kernel`・`compute_smooth_kappa{,_ext}_kernel`・`extract_radial_profile_kernel`・`extract_radial_te_kernel` | 勾配、逆制動輻射の smooth 係数、軸の列の取り出し |
| `ray_init.cu` | `initialize_rays_1d_geometry_kernel` | 1D の円筒・平板の光線の初期化 |
| `ray_trace.cu` | `ray_trace_1d_characteristic`（1D 既定）・`build_characteristic_breakpoints_kernel`・`build_characteristic_piece_info_kernel` | 特性線に沿った区間ごとの求積（NUMERICS §5.3.6）。復路は往路の区間積分をレイごとの表から使う（`Laser.raytrace.reuse_inward_integrals`、§5.3.6 (c'')） |
| `ray_trace.cu` | `ray_trace_1d_sph`（`integrator="leapfrog"`）・`reduce_per_ray_tallies_1d_kernel`・`sum_absorbed_power_per_ray_1d_kernel` | 刻み幅可変の Verlet（leapfrog）積分と光線ごとの集計 |
| `ray_trace.cu` | `ray_trace_3d`（2D_RZ の光線追跡、レーザー格子の節点へ沈着）・`ray_trace_2d`（検証ハーネス `tenryu verify` だけが呼ぶ） | |
| `ray_trace.cu` | `radial_absorption_1d_kernel` | `mode="radial_absorption_1d"`（§5.4） |
| `fast_trace_1d.cu` | `fast_trace_1d_kernel` ほか | 環境変数 `TENRYU_FAST_TRACE=1` のときだけの試験的な近道 |
| `laser.cu` | `gather_ray_output_1d_kernel`・`per_warp_step_reduce_kernel`・`replay_fold_tallies_kernel` | 光線の出力・診断 |
| `raytrace_skip.cu` | `ray_skip_check_kernel`・`reconstruct_laser_dep_kernel` | レイトレースの省略判定と沈着の再構成 |
| `deposit_transfer.cu` | `transfer_2d_kernel` | 2D_RZ のレーザー格子 → 流体セル |
| `cbet.cu` | `cbet_*_kernel`（19 本） | CBET の記録・交換・伝播（1D と 2D_RZ） |
| `port_section_chi.cu` | `build_chi_ps_warp_kernel`（参照: `build_chi_ps_pair_kernel`）・`audit_chi_kernel` | 断面ポートの CBET 相空間 |
| `hot_electron_1d_gpu.cu`・`hot_electron_2d_gpu.cu` | `hot_e_cone_chord_kernel`・`hot_e_reduce_rows_kernel`・`hot_e_cone_2d_chord_kernel` | 熱電子の円錐の弦に沿った沈着 |

### 1.5 Radiation

現行の FLD・\(S_N\) のカーネルは §1.0 の表と §6.7（FLD）・§6.8（\(S_N\)）。設計時の ID R1〜R16 で表にしていた
モンテカルロ輻射のカーネル（`compute_fleck_factor`・`ddmc_*`・`source_particle_*`・`composite_sort_and_partition`・
`imc_transport_persistent`・`ddmc_event_loop`・`tally_finalize`・`russian_roulette`・`marshak_source`・`census_comb_*`）は
2026-09-29 に退役した（表は `retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md`）。

#### 1.5.1 S_N 2D RZ Sweep Kernels

The production 2D_RZ \(S_N\) sweep uses a direction-parallel linear
characteristic kernel launched as
`<<<dim3(n_dirs_in_octant, n_groups), threads>>>`. Each block owns one
`(group, direction)` pair. A full inner iteration launches four octants
sequentially in the fixed order `(mr<0, mz<0)`, `(mr<0, mz>0)`,
`(mr>0, mz<0)`, `(mr>0, mz>0)`.

Per-direction `r_face`, `z_face`, `D_avg`, and `D_face_psi` workspaces are
allocated once per solve and zeroed each inner iteration before the sweep. The
reflection buffers persist across inner iterations, preserving the existing
z-top one-iteration-lag source-iteration semantic. After the octant launches,
`sn_reduce_cell_outputs_2d_kernel` runs one thread per cell-group and sums
direction contributions in increasing d order into the scalar-flux moment
`phi` and radial pressure tensor component `P_rr`.
`sn_reduce_face_flux_2d_kernel` runs one thread per face-group and applies the
same deterministic d-order reduction to the unique-face `face_flux_raw` layout.

### 1.6 Coupling/Utility（9カーネル）

| ID | カーネル名 | 種別 | N | 主要入出力 |
|----|-----------|------|---|-----------|
| U1 | `source_injection` | cell | n_cells | laser_dep → ee（輻射の沈着は FLD・S_N が物質の更新の中で扱う。rad_dep の注入はモンテカルロ輻射とともに退役） |
| U2 | `floor_clamp` | cell | n_cells | rho,Te,Ti → clamped |
| U3 | `energy_budget` | cell→scalar | n_cells | fields → E_kin,E_int,E_rad |
| U4 | `cfl_reduction` | cell→scalar | n_cells | c_s,u,Δl → dt_min（sub-kernels: dt_hydro/dt_cond。dt_rad はモンテカルロ輻射とともに退役）|
| U5 | `nan_check` | cell | n_cells | all fields → error flags |
| U6 | `qei_exchange` | cell | n_cells | Te,Ti,rho,Zbar,A_eff → Q_ei |
| U7 | ~~`cell_search_after_rezone`~~ | — | — | 退役（光子粒子のセル再同定、§7.4）|
| U8 | `compute_zbar` | cell | n_cells | Te,rho,volFrac → Z̄,A_eff（fixed/thomas_fermi/tabular、§7.6）|
| U9 | ~~`compute_opacities`~~ | — | — | 退役（IMC の不透明度の前計算、§7.7。現行は各輻射ソルバーが評価する — ARCHITECTURE §4.3.4）|

### 1.7 Parallel（4カーネル）

| ID | カーネル名 | 種別 | N | 主要入出力 |
|----|-----------|------|---|-----------|
| P1 | `halo_pack_cell` | cell | n_halo | fields → send_buf |
| P2 | `halo_unpack_cell` | cell | n_halo | recv_buf → ghost fields |
| P3 | `halo_pack_node` | node | n_halo | fields → send_buf |
| P4 | `halo_unpack_node` | node | n_halo | recv_buf → ghost fields |
| ~~P5~~ | ~~`emigrant_detect_pack`~~ | — | — | 退役（光子粒子の rank 間移動、§8.2）|
| ~~P6~~ | ~~`immigrant_unpack_merge`~~ | — | — | 退役（同上、§8.3）|

---

## 2. Hydro カーネル群（詳細設計）

### 2.1 H1: hydro_active_update

> **実装**：このカーネルは無い。`hydro_active` の更新は host のループ（`driver.cpp` の `update_hydro_active`）で行う。

```cpp
__global__ void hydro_active_update(
    int8_t* __restrict__ hydro_active,  // [n_cells] in/out
    const double* __restrict__ Te,       // [n_cells]
    double T_start_eV,
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 非活性セルのみ `Te >= T_start` を判定。活性セルは即座に return
- **分岐**: 非活性セルは時間経過と共に減少 → ワープ発散は限定的
- **メモリ**: coalesced read (Te), coalesced write (hydro_active)
- **レジスタ**: <10

### 2.1b H2: compute_node_mass

```cpp
__global__ void compute_node_mass(
    double* __restrict__ node_mass,         // [n_nodes] out: ノード質量 m_n [g]
    const double* __restrict__ cell_mass,   // [n_cells] in: セル質量 ΔM_c = ρ_c × V_c [g]
    const double* __restrict__ x_r,         // [n_nodes] in: ノードR座標 [cm]
    int nr, int nz                          // 構造格子次元（n_nodes=(nr+1)*(nz+1)、n_cells=nr*nz）
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 各セルが4コーナーへ R-weighted corner mass を `atomicAdd` で集約（NUMERICS §3.2.4）:
  `R_L=(R_00+R_01)/2`, `R_R=(R_10+R_11)/2`,
  `w_L=(2R_L+R_R)/(6(R_L+R_R))`, `w_R=(R_L+2R_R)/(6(R_L+R_R))`
  - 内側コーナー `n00,n01` に `w_L ΔM_c`、外側コーナー `n10,n11` に `w_R ΔM_c`
  - `2w_L+2w_R=1` のためセル質量は保存され、R-weighted `Svec_z` コーナー力と整合
  - 1D_SPH: `m_n = (mass[c-1] + mass[c]) / 2`（1Dでは2セル隣接、1/2重み）
- **レジスタ**: ~8
- **メモリ**: 各セルが `mass` と4コーナーの `x_r` を参照し、4コーナーの `node_mass` へ atomic 加算

### 2.1c H3: compute_area_vectors（2D_RZ専用呼び出し、1D_SPH対応コード含む）

```cpp
__global__ void compute_area_vectors(
    double* __restrict__ S_r,               // [n_cells × n_faces] out: 面積ベクトルR成分 [cm²]
    double* __restrict__ S_z,               // [n_cells × n_faces] out: 面積ベクトルZ成分 [cm²]
    const double* __restrict__ x_r,         // [n_nodes]
    const double* __restrict__ x_z,         // [n_nodes]
    int nr, int nz,
    int n_faces                             // 2D_RZ=4, 1D_SPH=2
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: NUMERICS §3.2.6 の面積ベクトル S_{c,k} をセルごとに計算
  - 2D_RZ: 各面の2ノード間の辺ベクトルに RZ 幾何因子 r̄_{c,k} を乗算
  - RZ幾何因子: r̄_{c,k} = (r_{k-1} + 2r_k + r_{k+1})/4
  - 1D_SPH: S_r[c,0] = 4πr_lo², S_r[c,1] = 4πr_hi²（面積ベクトルは半径方向のみ）
- **レジスタ**: ~15（4頂点座標 + 面積ベクトル4本×2成分）
- **メモリ**: 4隣接ノード座標を構造格子固定ストライドで参照
- **呼び出し条件**: §9 の Phase 1/Phase 5 では `if (geometry == GEOM_2D_RZ)` ガード付き。1D_SPH では H4/H9 が球面幾何を直接計算するため H3 はスキップされる。カーネル自体は 1D_SPH コードパスを含むが、v1.0 フローでは使用されない

### 2.2 H4: compute_corner_force（代表例：ノード型カーネル）

```cpp
__global__ void compute_corner_force(
    double* __restrict__ F_r,        // [n_nodes] out: Scratchバッファ（H5 が入力として使用）
    double* __restrict__ F_z,        // [n_nodes] out: Scratchバッファ（H5 が入力として使用）
    const double* __restrict__ x_r,  // [n_nodes]
    const double* __restrict__ x_z,  // [n_nodes]
    const double* __restrict__ P_e,  // [n_cells + n_ghost] 電子圧力（MPI境界ノードがゴーストセルP参照）
    const double* __restrict__ P_i,  // [n_cells + n_ghost] イオン圧力
    const double* __restrict__ Q,    // [n_cells + n_ghost] 人工粘性（H16がゴーストQ設定済み）
    const int8_t* __restrict__ hydro_active, // [n_cells + n_ghost]
    int nr, int nz, int n_ghost      // n_ghost=0 for single GPU
);
```

- **block**: 256, **grid**: `(n_nodes+255)/256`
- **ゴーストセル契約**: P_e/P_i/Q/hydro_active はゴーストセル含み。MPI境界ノードの隣接ゴーストセル圧力を正しく反映するため必須。出力 F_r/F_z は [n_nodes] であり所有ノードのみ書き込み
  - ゴーストセルは `[n_cells, n_cells+n_ghost)` に配置。構造格子ハロー幅=1: R_left面ゴーストが先頭、R_right, Z_bottom, Z_top, 4コーナーの順で連続配置（ARCHITECTURE §7.1 + NUMERICS §12.1 参照）。n_ghost = 2*(nr+nz)+4（2D_RZ）
  - **ゴーストセル索引公式（2D_RZ）**: `G = n_cells + offset`。各面/コーナーのoffset:
    - R_left (k=0): offset = j, j∈[0,nz) → 対応内部セル (0, j)
    - R_right (k=1): offset = nz + j, j∈[0,nz) → 対応内部セル (nr-1, j)
    - Z_bottom (k=2): offset = 2*nz + i, i∈[0,nr) → 対応内部セル (i, 0)
    - Z_top (k=3): offset = 2*nz + nr + i, i∈[0,nr) → 対応内部セル (i, nz-1)
    - Corner(R_left,Z_bottom): offset = 2*(nz+nr) → 対応内部セル (0, 0)
    - Corner(R_left,Z_top): offset = 2*(nz+nr)+1 → 対応内部セル (0, nz-1)
    - Corner(R_right,Z_bottom): offset = 2*(nz+nr)+2 → 対応内部セル (nr-1, 0)
    - Corner(R_right,Z_top): offset = 2*(nz+nr)+3 → 対応内部セル (nr-1, nz-1)
    - **1D_SPH**: n_ghost=2。offset=0 → inner (cell 0)、offset=1 → outer (cell nr-1)
    - **逆引き**: ghost_idx - n_cells からface/face内位置を算術的に復元可能。H16 Pass 2 / P1 pack / P4 unpack で使用
- **処理**: ノード n に隣接する最大4セルからコーナー力を集約（NUMERICS §3.2.5）
  - コーナー力: F_{c→n_k} = -(P_c + Q_c) × S_{c,k}（NUMERICS §3.2.5）。P_c = P_{i,c} + P_{e,c}（総圧力）
  - 面積ベクトル S_{c,k} の計算（NUMERICS §3.2.6）: 隣接ノード3点の座標を参照
  - RZ幾何因子 r̄_{c,k} = (r_{k-1} + 2r_k + r_{k+1})/4
  - 非活性セル: 力の寄与をゼロ化
  - **r=0軸ノード**（2D_RZ）: 全コーナー力の半径成分を強制ゼロ化 F_{r,k} = 0（NUMERICS §3.2.14(a)）。
    v_r=0 は H16 が強制するが、F_r≠0 のまま H5 で加速度を計算すると PdV 仕事が不整合になる
- **メモリ**: 各ノードが4セルのP,Q + 周辺ノード座標を参照 → ランダム寄りだが構造格子なのでストライドは固定
  - ストライド: nz+1（ノード）、nz（セル）で計算可能
- **レジスタ**: ~30（面積ベクトル4本×2成分 + 一時変数）
- **最適化**: ノード番号の2D→1D写像が既知のため、隣接セルのインデックスは算術計算で取得（間接参照不要）

### 2.2b H5: velocity_update

```cpp
__global__ void velocity_update(
    double* __restrict__ v_r,               // [n_nodes] in/out
    double* __restrict__ v_z,               // [n_nodes] in/out（1D_SPHではnullptr可）
    const double* __restrict__ F_r,         // [n_nodes] in: H4 出力（Scratchバッファ）
    const double* __restrict__ F_z,         // [n_nodes] in: H4 出力（1D_SPHではnullptr可）
    const double* __restrict__ node_mass,   // [n_nodes] in: H2 出力
    double dt,                              // サブステップ幅（Predictor: Δt/4、Corrector: Δt/2）
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_nodes+255)/256`
- **処理**: `v_new = v_old + dt × F / m_node`（NUMERICS §3.2.5）
  - **r=0軸ノード**（2D_RZ）: `v_r = 0` を強制（NUMERICS §3.2.14(a)）
  - ゼロ質量ガード: `m_node < m_floor` → 速度更新スキップ（数値安全策）
- **レジスタ**: ~8
- **メモリ**: coalesced read/write

### 2.2c H6: position_update

```cpp
__global__ void position_update(
    double* __restrict__ x_r,               // [n_nodes] in/out
    double* __restrict__ x_z,               // [n_nodes] in/out（1D_SPHではnullptr可）
    const double* __restrict__ v_r,         // [n_nodes] in
    const double* __restrict__ v_z,         // [n_nodes] in（1D_SPHではnullptr可）
    double dt,                              // サブステップ幅（Predictor: Δt/4、Corrector: Δt/2）
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_nodes+255)/256`
- **処理**: `x_new = x_old + dt × v`（NUMERICS §3.2.5）
  - **r=0ガード**（2D_RZ）: `x_r = max(0, x_r)`（NUMERICS §3.2.14(a)：非負保証）
- **レジスタ**: ~6
- **メモリ**: coalesced read/write

### 2.3 H7: compute_cell_geometry（融合カーネル）

体積V、断面積A_cell、セル中心(centroid)、特性長Δl を1カーネルで計算する。

```cpp
__global__ void compute_cell_geometry(
    double* __restrict__ vol,       // [n_cells] out
    double* __restrict__ area,      // [n_cells] out: 断面積 A_cell
    double* __restrict__ cent_r,    // [n_cells] out
    double* __restrict__ cent_z,    // [n_cells] out
    double* __restrict__ delta_l,   // [n_cells] out: 特性長 Δl = sqrt(A_cell) [cm]
    double* __restrict__ face_area, // [n_cells × n_faces] out: 面面積 A_m [cm²]（2D: 4面、1D: 2面）
                                    // 2D_RZ: 各面 = 2ノード間の辺長 × r̄_face × 2π（回転体の側面積）
                                    // 1D_SPH: face_area[c,0]=4πr_lo², face_area[c,1]=4πr_hi²
                                    // H7 の座標参照と融合して帯域を節約（設計時の参照先は退役した DDMC の R2/R3/R3b と Marshak 粒子源 R13）
    const double* __restrict__ x_r, // [n_nodes]
    const double* __restrict__ x_z, // [n_nodes]
    int nr, int nz,
    int n_faces                     // 2D_RZ=4, 1D_SPH=2（face_area の列数。H9 と同一引数）
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 4頂点座標を読み、§3.2.2 (RZ体積) + §3.2.3 (Shoelace面積) + centroid + 特性長 Δl = sqrt(A_cell) 計算（NUMERICS §3.2.2-3.2.3）。Δl は H10 (人工粘性) および U4 (CFL) で使用される
  - **face_area**: 各面の面積を算出（2D_RZ: 辺長×r̄_face×2π、1D_SPH: 4πr²）
  - （DDMC の代表長 `ell_ddmc` もここで作る設計だったが、DDMC とともに退役した）
- **融合理由**: 全て同じ4頂点座標を入力とし、出力が異なるだけ。分離すると座標の再読み込みが発生。face_area も同一座標から算出するため融合が自然
- **レジスタ**: ~20（4頂点座標 4×2 + 体積/面積/centroid/Δl 一時変数）
- **メモリ**: 隣接ノード座標は構造格子の固定ストライドで参照（coalesced ではないが L1 キャッシュで吸収）

### 2.3b H8: compute_density

```cpp
__global__ void compute_density(
    double* __restrict__ rho,               // [n_cells] out: 質量密度 [g/cm³]
    const double* __restrict__ mass,        // [n_cells] in: セル質量 [g]
    const double* __restrict__ vol,         // [n_cells] in: セル体積 [cm³]（H7出力）
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: `rho[c] = mass[c] / vol[c]`（NUMERICS §3.2.2）
- **退化セルガード**: `vol[c] < vol_floor`（vol_floor = 1e-30 cm³）の場合、`rho[c] = rho_floor`（NUMERICS §11.7 item 5）
- **レジスタ**: ~4
- **メモリ**: coalesced read/write

### 2.3c H9: compute_divergence

```cpp
__global__ void compute_divergence(
    double* __restrict__ div_u,             // [n_cells] out: 速度発散 [1/s]
    double* __restrict__ dVdt,              // [n_cells] out: 体積変化率 [cm³/s]
    const double* __restrict__ v_r,         // [n_nodes] in
    const double* __restrict__ v_z,         // [n_nodes] in（1D_SPHではnullptr可）
    const double* __restrict__ x_r,         // [n_nodes] in: ノードR座標（1D_SPH: dVdt=4π(r²v) 計算に必要。2D_RZではS経由のため参照しないがnullptr不可）
    const double* __restrict__ S_r,         // [n_cells × n_faces] in: H3出力の面積ベクトルR成分（1D_SPHではnullptr可：dVdt直接計算）
    const double* __restrict__ S_z,         // [n_cells × n_faces] in: H3出力の面積ベクトルZ成分（1D_SPHではnullptr可）
    const double* __restrict__ vol,         // [n_cells] in: セル体積（H7出力）
    int nr, int nz,
    int n_faces                             // 2D_RZ=4, 1D_SPH=2
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: `dVdt = Σ_k (v_k · S_k)`、`div_u = dVdt / V`（NUMERICS §3.2.8）
- **退化セルガード**: `vol[c] < vol_floor`（vol_floor = 1e-30 cm³）の場合、`div_u[c] = 0`, `dVdt[c] = 0`（NUMERICS §11.7 item 5）
  - 各セルの面を走査し、面ノード速度と面積ベクトルの内積を累積
  - 1D_SPH: `dVdt = 4π(r_hi² v_hi - r_lo² v_lo)`（r_hi = x_r[i+1], r_lo = x_r[i] で参照）
- **レジスタ**: ~12（面ループ変数 + 部分和）
- **メモリ**: ノード速度は構造格子固定ストライドで参照

### 2.3d H15: compute_sound_speed

```cpp
__global__ void compute_sound_speed(
    double* __restrict__ c_s,               // [n_cells] out: 音速 [cm/s]
    const double* __restrict__ Pe,          // [n_cells] in
    const double* __restrict__ Pi,          // [n_cells] in
    const double* __restrict__ rho,         // [n_cells] in
    const double* __restrict__ Te,          // [n_cells] in
    const double* __restrict__ Ti,          // [n_cells] in
    const double* __restrict__ Cv_e,        // [n_cells] in: 電子比熱 [erg/(g·eV)]
    const double* __restrict__ Cv_i,        // [n_cells] in: イオン比熱 [erg/(g·eV)]
    const double* __restrict__ Zbar,        // [n_cells] in
    const double* __restrict__ A_eff,       // [n_cells] in: 有効原子量（多材料: §1.1.6 調和平均）
    int eos_model,                          // 0=ideal, 1=ionmix, 2=sesame
    const EOSTable* __restrict__ eos_table, // テーブルEOS時に偏微分で使用
    DeviceErrorFlags* error_flags,          // §0.6 準拠：c_s²<0 時に sound_speed_negative フラグ設定
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**（NUMERICS §1.1.6）:
  - **ideal_gas**: `c_s = sqrt((γ_e P_e + γ_i P_i) / ρ)`、γ=5/3
  - **table_eos**（1D の実装、`hydro_1d_bodies.cuh` の `compute_sound_speed_2t_kernel_body`）：イオン表と電子表の
    それぞれで、自分の温度（\(T_i\)、\(T_e\)）と自分の比熱で `device_eos_sound_speed` を評価し、
    \(c_s = \sqrt{c_{s,i}^2 + c_{s,e}^2}\) とする（設計時の混合温度 \(T_\mathrm{eff}=(T_i+\bar ZT_e)/(1+\bar Z)\) は使わない）。
    cold branch を持つ電子表では電子の寄与を符号付きの \(c_{s,e}^2\) とし、\(c_s=\sqrt{\max(c_{s,i}^2+c_{s,e}^2,0)}\)
  - **c_s² ≤ 0 の扱い**：`device_eos_sound_speed` は \(c_s^2\le0\) のとき \(\sqrt{(5/3)P/\rho}\) を返す（NUMERICS §1.1.6）。
    エラーフラグ（`sound_speed_negative`）は無い
- **レジスタ**: ~15（ideal）/ ~25（table: 偏微分計算含む）
- **メモリ**: coalesced read（セルフィールド）。テーブルEOS時は `__ldg()` でテーブル参照

### 2.4 H13/H14: EOS カーネル

> **現行の実装との違い（2026-09-29）**：以下の `eos_forward`・`eos_inverse`（Newton 反復）は設計時のもので、実装に無い。
> 表 EOS は §2.4x の `__device__` 関数（`eos_device_table.cuh`）を流体・伝導・輻射のカーネルが直接呼ぶ。実装の要点：
> - 逆変換 \(e\to T\) は Newton 反復ではなく `device_inverse_reclose`：温度の節点の上で根を含む区間を単調に探し、
>   区間内の線形の式で解く（反復なし、`MAX_ITER` も無い）。床・表の端でのクランプと区間を作れない場合は結果の構造体の
>   フラグで返し、閉包が回数を数える
> - SESAME のイオン表は、読み込み時に 301（全体）と 304（電子）の差を 301 の格子の上で一度だけ作る。差が負の節点は
>   クランプせずに保持し、数を 1 回警告する（クエリ時の `max(·, 0)` と `eos_ion_negative` は無い、NUMERICS §1.1.5）
> - 多材料のセルは質量分率で混合せず、支配材料（void を除く体積分率が最大の材料）の表で閉じる
>   （`CellEOSTableSelector`、NUMERICS §1.1.5）。`Materials.mixture.eos_mix_rule` と `fractions` は読み込まれるが使われない
>
> 以下は設計時の記述として残す。

```cpp
// Forward: T → (e, P, Cv) — __global__ wrapper calling __device__ eos_forward_impl
__global__ void eos_forward(
    double* __restrict__ ee,     // [n_cells] out
    double* __restrict__ ei,     // [n_cells] out
    double* __restrict__ Pe,     // [n_cells] out
    double* __restrict__ Pi,     // [n_cells] out
    double* __restrict__ Cv_e,   // [n_cells] out: 質量比熱 c_v,e [erg/(g·eV)]
    double* __restrict__ Cv_i,   // [n_cells] out: 質量比熱 c_v,i [erg/(g·eV)]
    const double* __restrict__ rho,  // [n_cells]
    const double* __restrict__ Te,   // [n_cells]
    const double* __restrict__ Ti,   // [n_cells]
    const double* __restrict__ Zbar, // [n_cells]
    const EOSTable* eos_e,           // 電子EOS テーブル（SESAME 304 / IONMIX electron）（ARCHITECTURE §4.3.1 準拠）
    const EOSTable* eos_i,           // イオンEOS テーブル（SESAME 301=total / IONMIX ion）
    const double* __restrict__ volFrac, // [n_cells × n_mat]（多材料時。n_mat==1 では nullptr 可）
    const double* __restrict__ A_mat,   // [n_mat] 材料ごとの原子量
    const EOSTable** eos_e_per_mat,     // [n_mat] 材料ごとの電子EOS テーブル配列（多材料+テーブルEOS時。n_mat==1 では eos_e と同一）
    const EOSTable** eos_i_per_mat,     // [n_mat] 材料ごとのイオンEOS テーブル配列（同上）
    int eos_model,                   // 0=ideal, 1=ionmix, 2=sesame。GPU上では SESAME/IONMIX 同一 EOSTable 構造体で補間コード共通
    int n_mat,                       // 材料数（単一材料: 1）
    DeviceErrorFlags* error_flags,   // §0.6 準拠：C_v≤0 クランプ等の WARNING 記録
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **多材料EOS** (NUMERICS §1.1.5(c)): n_mat > 1 の場合、各セルで材料ループを実行:
  1. 各材料 α の EOS を個別に評価: e_α, P_α, Cv_α = eos_forward_impl(ρ, T, Z̄_α, eos_e_per_mat[α])
  2. 質量分率 f_{m,α} で混合: e = Σ f_{m,α} e_α, P = Σ f_{m,α} P_α, Cv = Σ f_{m,α} Cv_α
  3. single-state仮定: 全材料が同一 (ρ, Te, Ti) を共有（NUMERICS §1.1.5(c)）
  4. n_mat==1 の場合は材料ループをスキップし直接評価（既存コードパスと同一）
- **理想気体**: 解析式で A_eff, Z̄_eff を使用すれば材料ループ不要（H13 内で直接計算、NUMERICS §1.1.5(c)）。レジスタ ~15
- **テーブルEOS**: `(log ρ, log T)` 空間の双線形補間（NUMERICS §1.1.5）。テーブルは `__ldg()` で参照
  - **SESAME 2テーブル規約**（NUMERICS §1.1.5 必須）:
    - `eos_total[mat]`（テーブル301）と `eos_e[mat]`（テーブル304）は**独立グリッド**で構築。
      要素ごとの減算は不可 → クエリ時に各テーブルを個別補間し差分算出:
      `P_i(ρ,T) = max(interp(eos_total,ρ,T) - interp(eos_e,ρ,T), 0)` [非負ガード]
    - **positivity guard**: 高Z材料で P_total ≈ P_e の場合、桁落ちで P_i<0 になりうる → `max(...,0)` 必須。
      クランプ発生時は `error_flags->eos_ion_negative = 1` を設定
    - **304不在時 1Tフォールバック**: P_e = P_total × Z̄/(1+Z̄), P_i = P_total - P_e
  - **テーブル範囲外処理**: クエリが `(log ρ, log T)` テーブル範囲外の場合、最近傍のテーブル境界値にクランプして補間する（NUMERICS §1.1.5 「テーブル範囲外はフロア値にクランプ」）。外挿は行わない。`c_v ≤ 0` の場合は `c_v_floor = 1e-3 erg/(g·eV)` にクランプし WARNING を `error_flags` に記録する（NUMERICS §1.1.5）
- **融合**: forward では e,P,Cv を同時に計算（テーブル参照が共通）
- **メモリ**: coalesced read（rho, Te, Ti, Zbar）。テーブルEOS時のテーブル参照は `__ldg()` 経由で L2 キャッシュに収まる（テーブルサイズ ~数十KB）

```cpp
// Inverse: e → T (Newton反復) — __global__ wrapper calling __device__ eos_inverse_impl
__global__ __launch_bounds__(256, 4) void eos_inverse(
    double* __restrict__ T_out,   // [n_cells] out
    const double* __restrict__ rho,
    const double* __restrict__ e_target,
    const double* __restrict__ T_guess, // 前ステップのTを初期推定
    const double* __restrict__ Zbar,
    const EOSTable* eos_primary,         // 主テーブル（ARCHITECTURE §4.3.1 準拠: IONMIX→eos_e/eos_i、SESAME電子→eos_e）
    const EOSTable* eos_secondary,       // SESAMEイオン時のみ非null: eos_e（差分評価用。IONMIX/理想気体→nullptr）
    const double* __restrict__ volFrac,  // [n_cells × n_mat]（多材料時。n_mat==1 → nullptr）
    const double* __restrict__ A_mat,    // [n_mat] 材料ごとの原子量
    const EOSTable** eos_per_mat,        // [n_mat] 材料ごとのEOSテーブル配列（多材料+テーブルEOS時）
    int eos_model,
    int species,                     // 0=electron, 1=ion（SESAME 2テーブル分岐に必須）
    int n_mat,                       // 材料数
    DeviceErrorFlags* error_flags,   // §0.6 準拠：MAX_ITER 到達時の WARNING 記録
    int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`

- **species パラメータ**（SESAME 2テーブル必須）:
  - `species=0`（電子）: テーブル304（eos_e）で Newton 反復。残差 = eos_e(ρ,T).ee - e_target
  - `species=1`（イオン）: テーブル301−304 で Newton 反復。残差 = [eos_total(ρ,T).e - eos_e(ρ,T).ee] - e_target。
    両テーブルを毎反復で評価するため電子よりコスト約2倍。
    **安全策**: e_i_target < 0 → T_ion = T_floor フォールバック + `eos_ion_negative` 設定。Cv_i = Cv_total - Cv_e ≤ 0 時も同様
  - IONMIX/理想気体: species は無視（テーブル構造が同一のため分岐不要）
- **多材料EOS逆変換** (NUMERICS §1.1.5(c)): n_mat > 1 の場合、混合EOS関数 e_mix(T) = Σ f_{m,α} e_α(ρ,T) の逆変換をNewton反復で解く。各反復の Cv_mix = Σ f_{m,α} Cv_α をJacobian として使用。n_mat==1 は既存パスと同一
- **反復**: Newton法 ≤MAX_ITER(20)回、収束判定 `|T^{(m+1)}-T^{(m)}|/max(T^{(m)},T_floor) < 1e-8`（NUMERICS §3.1.5 EOS逆変換）。各反復後にフロアクランプ適用。打ち切り時は最終値を採用し WARNING
- **分岐**: 反復回数がセルにより異なる → ワープ発散あり
  - 典型: ほとんどのセルが3-5回で収束 → 限定的
  - 最悪ケース: ショック近傍で大きな温度変化があり ~15 回反復。MAX_ITER(20) 到達時は error_flags に記録
- **レジスタ**: ~25（理想気体は反復不要で解析逆変換）
- **メモリ**: テーブルEOS時は `__ldg()` でテーブル参照。coalesced read（rho, e_target, T_guess）

### 2.4x Table-EOS Device Helpers (`src/materials/eos_device_table.cuh`)

Current hydro table-EOS kernels use the following device-side data view and helpers:

```cpp
struct DeviceEOSTableView {
    const double* log_rho_grid;   // [n_rho]
    const double* log_T_grid;     // [n_T]
    const double* P_table;        // [n_T * n_rho]
    const double* e_table;        // [n_T * n_rho]
    const double* cv_table;       // [n_T * n_rho]
    int n_rho;                    // n_rho==0 => no table (caller uses ideal-gas fallback)
    int n_T;
    double log_rho_min, log_rho_max;
    double log_T_min,  log_T_max;
    double d_log_rho_inv;         // >0: uniform-grid fast path, 0: binary search
    double d_log_T_inv;           // >0: uniform-grid fast path, 0: binary search
};

struct RhoBracket {
    int i0;
    int i1;
    double w;                     // interpolation weight in rho direction
};
```

Required device functions (implementation-aligned):

```cpp
__device__ RhoBracket find_rho_bracket(const DeviceEOSTableView& tab, double rho);
__device__ double interp_at(const DeviceEOSTableView& tab,
                            const double* field,
                            const RhoBracket& rb,
                            double logT);
__device__ double device_eos_pressure(const DeviceEOSTableView& tab,
                                      const RhoBracket& rb,
                                      double logT);
__device__ double device_eos_energy(const DeviceEOSTableView& tab,
                                    const RhoBracket& rb,
                                    double logT);
__device__ double device_eos_cv(const DeviceEOSTableView& tab,
                                const RhoBracket& rb,
                                double logT);
__device__ double device_eos_T_from_e_monotone(const DeviceEOSTableView& tab,
                                                const RhoBracket& rb,
                                                double e_target);
__device__ double device_eos_sound_speed(const DeviceEOSTableView& tab,
                                         const RhoBracket& rb,
                                         double logT,
                                         double rho,
                                         double cv);
```

- `find_rho_bracket` is intentionally called once per cell and reused for `T_from_e`, `P`, `e`, `cv`, and `c_s` evaluations.
- `interp_at` performs bilinear interpolation in \((\log\rho,\log T)\) with axis clamping and supports uniform-grid fast index mapping.
- `device_eos_T_from_e_monotone` performs monotone binary search on the mixed row \(e(T)\) (no Newton loop).
- `device_eos_sound_speed` uses a \(\Gamma_1\)-based evaluation with the analytic derivatives of the bilinear interpolant (not finite differences) and returns \(\sqrt{(5/3)P/\rho}\) when \(c_s^2\le 0\).
- The inverse used by the closures is `device_inverse_reclose` (and its `_with_low_density_extrap` / `_with_high_t_tail` variants), which calls `device_eos_T_from_e_monotone` and reports clamps and bracket failures in its result struct.

### 2.4a H10: compute_artificial_viscosity

```cpp
__global__ void compute_node_sigma_1d_kernel(
    double* __restrict__ sigma,             // [n_nodes] out: Christensen limiter slope
    const double* __restrict__ node_r,      // [n_nodes] in
    const double* __restrict__ node_u,      // [n_nodes] in
    int n_nodes,
    double J                                // limiter parameter（既定 1.0）
);
__global__ void compute_q_1d_kernel(
    double* __restrict__ Qvisc,             // [n_cells] out: 人工粘性 [dyne/cm²]
    const double* __restrict__ rho,         // [n_cells] in
    const double* __restrict__ vol,         // [n_cells] in
    const double* __restrict__ node_r,      // [n_nodes] in
    const double* __restrict__ node_u,      // [n_nodes] in
    const double* __restrict__ c_s,         // [n_cells] in: H15出力 [cm/s]
    const double* __restrict__ sigma,       // [n_nodes] in: limiter slopes
    double* __restrict__ chi_out,           // [n_cells] out: shock sensor（任意、nullptr可）
    const int8_t* __restrict__ hydro_active,
    int n_cells,
    double C1,                              // 線形項係数（既定 0.1）
    double C2                               // 二次項係数（既定 1.5）。注: 式中は C2² を使用（Wilkins型）
);
__global__ void compute_artificial_heat_1d_kernel(
    double* __restrict__ heat_rate,         // [n_cells] out: セルへ流入する人工熱 [erg/s]
    const double* __restrict__ rho,         // [n_cells] in
    const double* __restrict__ e,           // [n_cells] in: ion or total specific energy [erg/g]
    const double* __restrict__ chi,         // [n_cells] in: Christensen shock sensor
    const double* __restrict__ node_r,      // [n_nodes] in
    const int8_t* __restrict__ hydro_active,
    int n_cells,
    double C_H                              // 人工熱流束係数（`Numerics.hydro.av_heat_C`、既定 0 = 無効）
);
__global__ void apply_artificial_heat_kernel(
    double* __restrict__ e,                 // [n_cells] in/out: 比内部エネルギー [erg/g]
    const double* __restrict__ heat_rate,   // [n_cells] in: net heat power [erg/s]
    const double* __restrict__ mass,        // [n_cells] in: セル質量 [g]
    const int8_t* __restrict__ hydro_active,
    int n_cells,
    double dt,
    int clamp_to_zero,
    double* __restrict__ E_floor_injected,  // [1] atomicAdd
    int* __restrict__ clamp_count           // [1] atomicAdd
);
// 1D_SPH:
//   Pass 1: Christensen limiter slope sigma_j を構成
//     SL = J (u_j-u_{j-1})/(r_j-r_{j-1}), SR = J (u_{j+1}-u_j)/(r_{j+1}-r_j)
//     sigma_j = sign(SL) min(|SL|,|SC|,|SR|) if SL*SR>0 else 0
//   Center ghost: r_{-1}=2r_0-r_1（中心 r_0=0 または剛体の内壁での鏡像、r_0=0 なら -r_1 とビット一致）, u_{-1}=-u_1
//   Outer ghost : r_{N+1}=2r_N-r_{N-1}, u_{N+1}=2u_N-u_{N-1}
//   Pass 2: u_L*=u_j+0.5Δr sigma_j, u_R*=u_{j+1}-0.5Δr sigma_{j+1}
//     chi = max(0, -(u_R*-u_L*)/Δr)
//     div_u < 0 を満たすセルのみ
//     Q = ρ × (C2² × Δr² × chi² + C1 × Δr × c_s × chi)
//   Pass 3: artificial heat (1D_SPH only, av_heat_C > 0)
//     H_j = -C_H ρ_j l_j² chi_H,j (e_i-e_{i-1}) / (r_c,i-r_c,i-1)
//     heat_rate_i = -(A_{i+1}H_{i+1} - A_i H_i)
//     inactive interfaces and physical boundaries use H = 0
//   Pass 4: e += dt * heat_rate / M
// 2D_RZ は従来どおり scalar path（div_u ベース）のみを使用
// NUMERICS §3.1.6（1D_SPH）、§3.2.9（2D_RZ）
```

- **block**: 256
- **grid**: `compute_node_sigma_1d_kernel` は `(n_nodes+255)/256`、
  `compute_q_1d_kernel` / `compute_artificial_heat_1d_kernel` /
  `apply_artificial_heat_kernel` は `(n_cells+255)/256`
- **レジスタ**: `compute_node_sigma_1d_kernel` ~16, `compute_q_1d_kernel` ~14,
  `compute_artificial_heat_1d_kernel` ~24, `apply_artificial_heat_kernel` ~12
- **メモリ**: `sigma[n_nodes]` と、`av_heat_C > 0` の場合は `chi[n_cells]`,
  `heat_rate[n_cells]` の一時バッファを追加。read/write はいずれも coalesced

### 2.4b H11/H12: energy_update カーネル

```cpp
// H11: ion energy update
__global__ void energy_update_ion(
    double* __restrict__ ei,                // [n_cells] in/out: イオン比内部エネルギー [erg/g]
    const double* __restrict__ Pi,          // [n_cells] in: イオン圧力（時間中心化 P_i^{n+1/2}、NUMERICS §3.2.12）
    const double* __restrict__ Qvisc,       // [n_cells] in: 人工粘性 Q^{pred}（NUMERICS §3.2.12）
    const double* __restrict__ Q_ei,        // [n_cells] in: 電子-イオン交換 [erg/cm³/s]（U6出力）
    const double* __restrict__ vol_old,     // [n_cells] in: 旧体積
    const double* __restrict__ vol_new,     // [n_cells] in: 新体積
    const double* __restrict__ mass,        // [n_cells] in: セル質量 [g]
    double dt,                              // タイムステップ幅
    int n_cells
);
// e_i += -(P_i + Q) × (V_new - V_old) / M + Q_ei × V_new × dt / M
// P_i は時間中心化 (P_i^n + P_i^{pred})/2、Q は Q^{pred}（NUMERICS §3.2.12）
// Q_ei > 0 → 電子→イオンへエネルギー移動（NUMERICS §1.1.3 符号規約）

// H12: electron energy update
__global__ void energy_update_electron(
    double* __restrict__ ee,                // [n_cells] in/out: 電子比内部エネルギー [erg/g]
    const double* __restrict__ Pe,          // [n_cells] in: 電子圧力（時間中心化 P_e^{n+1/2}、NUMERICS §3.2.12）
    const double* __restrict__ Q_ei,        // [n_cells] in: 電子-イオン交換 [erg/cm³/s]（U6出力）
    const double* __restrict__ vol_old,     // [n_cells] in
    const double* __restrict__ vol_new,     // [n_cells] in
    const double* __restrict__ mass,        // [n_cells] in
    double dt,
    int n_cells
);
// e_e += -P_e × (V_new - V_old) / M - Q_ei × V_new × dt / M
// 人工粘性 Q は電子には適用しない（Q加熱は全量イオンへ、NUMERICS §3.2.10）
// 1D_SPH の人工熱流束 H は H11/H12 の後段で apply_artificial_heat_kernel により
// ion energy（1Tでは total energy）へ別途加算する
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **レジスタ**: H11 ~20, H12 ~20（Q_ei×V×dt/M 項を含む）
- **質量フロアガード**: `mass[c] < M_floor`（M_floor = 1e-30 g）の場合、エネルギー更新をスキップ（ΔE = 0）
- **メモリ**: coalesced read/write（全配列がセルインデックスでアクセス）
- **NUMERICS参照**: H11/H12 は NUMERICS §3.2.10（2Tエネルギー方程式）の離散化、Predictor-Corrector時間積分は §3.2.12

### 2.5 Predictor-Corrector シーケンス

> **速度更新のdt係数**:
> H5 `velocity_update(dt_half)` のカーネルパラメータ `dt_half = Δt/2`（Strang 半ステップ）。
> Predictor 内部で `a^n × dt_half/2 = a^n × Δt/4` を適用。
> Corrector 内部で `a^{pred} × dt_half = a^{pred} × Δt/2` を適用。

1つの hydro half-step は以下のカーネルチェーンで構成される：

> **基準状態の保持（実装必須）**: Predictor 実行**前**に以下の6配列を Scratch バッファに退避する：
> `v_r_save = v_r`, `v_z_save = v_z`, `x_r_save = x_r`, `x_z_save = x_z`,
> `Pe_save = Pe`, `Pi_save = Pi`, `vol_save = vol`
>
> - **v^n, r^n**: Corrector の H5/H6 が基準状態として使用（NUMERICS §3.2.12: v^{n+1} = v^n + Δt·a^{n+1/2}）
> - **Pe^n, Pi^n**: Corrector の時間中心化 P^{n+1/2} = (P^n + P^{pred})/2 に使用（NUMERICS §3.2.12 Eq.）
> - **V^n**: H11/H12 の PdV 仕事 (V^{n+1} - V^n) に使用（vol_old 引数）
>
> **Corrector 内のカーネル順序（重要）**: NUMERICS §3.2.12 は Leapfrog 形式の位置更新
> r^{n+1} = r^n + Δt·v^{n+1/2}（Predictor 速度を使用）を規定する。
> Corrector H5 が v を上書きすると v^{n+1/4} が消失するため、**H6 を H5 より先に実行**し、
> v^{n+1/4} が v バッファに残存している間に位置更新を完了させる。

```
// dt_half = Δt_hydro / 2 （Strang半ステップ幅）
// 【実装注意】v^n, r^n, P^n, V^n は Predictor 前に退避。Corrector で復元して使用する
// Scratch バッファ: v_r_save, v_z_save, x_r_save, x_z_save, Pe_save, Pi_save, vol_save
PREDICTOR:
  H3: compute_area_vectors        ← node coords
  H4: compute_corner_force        ← P^n, Q^n, S
  H5: velocity_update(dt_half/2)  ← v^n + (Δt/4)*a^n → v^{n+1/4}
  H6: position_update(dt_half/2)  ← r^n + (Δt/4)*v^{n+1/4}
  H7: compute_cell_geometry       ← new coords → V, A, Δl
  H8: compute_density             ← mass/V → ρ
  H9: compute_divergence          ← v,S → div_u
  H10: compute_artificial_viscosity
  H13: eos_forward                ← ρ, T → P^{pred}（Pe, Pi を上書き）
  H15: compute_sound_speed

CORRECTOR:
  H3: compute_area_vectors        // r^{pred}（Predictor 座標）を使用
  H4: compute_corner_force        ← P^{pred}, Q^{pred}
  // --- 位置更新を速度更新より先に実行（v^{n+1/4} 保護） ---
  [D2D: x_r ← x_r_save, x_z ← x_z_save]   // r^n 復元
  H6: position_update(dt_half)    ← r^n + (Δt/2)*v^{n+1/4}（v はまだ v^{n+1/4}）
  [D2D: v_r ← v_r_save, v_z ← v_z_save]   // v^n 復元（v^{n+1/4} を上書き）
  H5: velocity_update(dt_half)    ← v^n + (Δt/2)*a^{pred} → v^{n+1/2}
  // --- 幾何・密度・発散 ---
  H7: compute_cell_geometry       // r^{n+1/2} → V^{n+1/2}
  H8: compute_density
  H9: compute_divergence
  // --- エネルギー更新 ---
  [Kernel: Pe = (Pe_save + Pe)/2, Pi = (Pi_save + Pi)/2]  // P^{n+1/2} 時間中心化（NUMERICS §3.2.12）
  // Q^{n+1/2} = Q^{pred}（Predictor divergence で評価済み、NUMERICS §3.2.12 注）
  U6: qei_exchange                ← T_e, T_i → Q_ei
  H11: energy_update_ion(vol_old=vol_save, vol_new=vol)    // V^n, V^{n+1/2}
  H12: energy_update_electron(vol_old=vol_save, vol_new=vol)
  H14: eos_inverse(species=0)     ← ee → Te
  H14: eos_inverse(species=1)     ← ei → Ti
  H13: eos_forward                ← T → P, Cv
  H15: compute_sound_speed
```

**各半ステップのカーネル起動数**: 約13-16回

> **注**: 上記は2D_RZ用。1D_SPHではH3（面積ベクトル）をスキップし、H4/H9 が球面幾何公式を直接使用する。ALE（A1-A5）は2D_RZ専用で1D_SPHでは無効。
> **NUMERICS参照**: Predictor-Corrector の時間積分方式は NUMERICS §2.2 および §3.2.5 (2D RZ hydro)。

### 2.6 H16: apply_hydro_bc

```cpp
__global__ void apply_hydro_bc(
    double* __restrict__ v_r,        // [n_nodes] in/out: 節点速度R成分
    double* __restrict__ v_z,        // [n_nodes] in/out: 節点速度Z成分
    double* __restrict__ Te,         // [n_cells + n_ghost] in/out: 電子温度
    double* __restrict__ Ti,         // [n_cells + n_ghost] in/out: イオン温度
    double* __restrict__ Pe,         // [n_cells + n_ghost] in/out: 電子圧力
    double* __restrict__ Pi,         // [n_cells + n_ghost] in/out: イオン圧力
    double* __restrict__ Qvisc,      // [n_cells + n_ghost] in/out: 人工粘性（H4 がゴーストQ参照）
    double* __restrict__ rho,        // [n_cells + n_ghost] in/out: 密度（free: ρ_ghost=ρ_interior, pressure: EOS入力）
    const int8_t* __restrict__ bc_type, // [4 or 2] 面別BC種別（free/fixed/reflect/pressure/axis）
    double P_ext,                    // 外部圧力 [dyne/cm²]（bc_type="pressure" 時に使用、§8.1）
    const EOSTable* __restrict__ eos_table, // EOS テーブル（bc_type="pressure" 時に EOS_forward 呼び出しに必要）
    const double* __restrict__ Zbar, // [n_cells + n_ghost] 有効電荷（pressure BC の EOS_forward 呼び出しに必要）
    int eos_model,                   // 0=ideal, 1=ionmix, 2=sesame（EOS_forward 分岐に必要）
    int* __restrict__ clamp_count,   // [1] atomicAdd: 速度リミッター発動時にインクリメント（NUMERICS §11.7）
    int n_boundary_nodes, int n_boundary_cells,
    int nr, int nz                   // 格子次元
);
```

- **block**: 256, **grid**: 下記パス方式に依存
- **ディスパッチ戦略**: ノード処理とゴーストセル処理は2パスで実行する:
  - **Pass 1**（ノードベース、grid=`(n_boundary_nodes+255)/256`）: 境界ノードの速度BC適用（fixed/reflect/axis/速度上限）
  - **Pass 2**（セルベース、grid=`(n_boundary_cells+255)/256`）: ゴーストセルの流体変数設定（ρ,Te,Ti,Pe,Pi,Qvisc の外挿/EOS評価）
  - **実装方式**: 同一カーネル内で `tid < n_boundary_nodes` / `tid < n_boundary_cells` で分岐するか、別カーネルに分離してもよい（実装者判断）。
    **注意**: 同一カーネル方式の場合、grid = `(max(n_boundary_nodes, n_boundary_cells)+255)/256` とすること。
    n_boundary_cells = 2*(nr+nz)+4 > n_boundary_nodes = 2*(nr+nz) のため、grid を n_boundary_nodes に合わせると 4 つのコーナーゴーストが未処理になり、H4 がstale値を読む
  - **境界ノード列挙**: 構造格子のため、境界ノードは (nr, nz) から算術的に列挙可能。tid→(面, 面内位置) の写像: tid ∈ [0, nz+1) → R_left面ノード j=tid、tid ∈ [nz+1, 2*(nz+1)) → R_right面ノード j=tid-(nz+1)、以降 Z_bottom, Z_top。コーナーノードは最初に出現する面で処理（重複排除は面チェックで保証）
  - **境界セル列挙**: 構造格子から算術的に列挙。`n_boundary_cells = n_ghost = 2*(nr+nz)+4`。
    tid→(面, 面内位置): R_left面 i=0,j=0..nz-1、R_right面 i=nr-1,j=0..nz-1、Z_bottom面 i=0..nr-1,j=0、Z_top面 i=0..nr-1,j=nz-1（4隅は重複排除）
    4つの隅ゴーストも明示的に充填:  
    - (0,0) = R_left × Z_bottom  
    - (0,nz-1) = R_left × Z_top  
    - (nr-1,0) = R_right × Z_bottom  
    - (nr-1,nz-1) = R_right × Z_top
    コーナーゴーストは、対角隣接内側セル（例: R_left×Z_bottom→(0,0)）からのゼロ勾配外挿で与える。
- **処理**（NUMERICS §8.1 準拠、物理境界面にのみ適用。パーティション境界はハロー交換で処理）:
  - **free**: ゴーストセルの圧力 P_ghost = 0（P_ext=0、ICF標準の真空外側条件）、密度 ρ_ghost = ρ_interior（最近接内部セルコピー）、温度はゼロ勾配外挿（Te_ghost=Te_interior, Ti_ghost=Ti_interior）。速度は運動方程式で自由決定（NUMERICS §8.1(a)）。1D の実装（`hydro_1d.cu` の外側節点の力）はゴーストの \(P+Q\) を最外セルの \(Q_{N-1}\) とする（fixed・reflect は \(P_e+P_i+Q\)、pressure は \(P_\mathrm{ext}+Q_{N-1}\)）
  - **fixed**: 境界ノード速度 = 0（v_r = v_z = 0）。ゴーストセル値は内部値コピー
  - **reflect**: 境界ノードの法線方向速度成分 = 0（R面: v_r=0、Z面: v_z=0）。ゴーストセルの T,P は内部値コピー、Qvisc_ghost = Qvisc_boundary
  - **state_supply**: 2D_RZ z_bottom/z_top の dict-form 専用。`apply_state_supply_z_bottom_node_kernel` / `apply_state_supply_z_top_node_kernel` は境界 node の `x_z` を `z_min` / `z_max` に固定し、mesh node `v_z` をゼロにする。Material `v_z` はここでゼロ化せず、`state_supply_bc.cu::override_state_supply_kernel` と `restore_state_supply_material_velocity` が境界 row cell を supplied `rho_g_per_cc`, `u_z_cm_per_s`, `T_eV` へ復元する。
  - **pressure**: ゴーストセル総圧力 = P_ext を保証する。
    Te_ghost = Te_interior, Ti_ghost = Ti_interior, ρ_ghost = ρ_interior（NUMERICS §8.1(a)）。
    Pe_ghost = EOS_forward(ρ_ghost, Te_ghost).Pe, Pi_ghost = P_ext - Pe_ghost（総圧力 = P_ext を保証）。
    **注意**: Pe_ghost > P_ext の場合 Pi_ghost < 0 になるが、**クランプしてはならない**。
    H4 は Pe_ghost + Pi_ghost = P_ext のみ使用するため、数学的に正しい。クランプすると境界圧が P_ext から乖離する。
    H4（force計算）はゴーストセルの Pe_ghost + Pi_ghost = P_ext を読み取り、境界力を自然に計算する。
    H4 に P_ext 引数は不要（ゴーストセル状態駆動）。T,Qvisc はゼロ勾配外挿
  - **axis**（r=0、2D_RZ専用）: v_r = 0（R=0 上の全ノード）。ゴーストセルの T,P は reflect と同一
  - **速度上限**（NUMERICS §11.7、全BC共通）: 全境界ノードで `|u| = sqrt(v_r² + v_z²)` を検査。`|u| > c`（光速）の場合、`v_r *= c/|u|`, `v_z *= c/|u|` でスケーリング。clamp_count をインクリメントし WARNING を出力。v1.0 は非相対論的コードのため物理的に発生しないはずだが、数値的安全策として設ける
- **1D_SPH**: 2面（inner, outer）。inner は常に reflect（軸対称中心）
- **2D_RZ**: 4面（R_left, R_right, Z_bottom, Z_top）。R_left が r=0 軸の場合は axis 扱い
- **制約**: `pressure` BC は R_right 面のみサポート（NUMERICS §8.1）。他面への指定は namelist validation でエラー
- **コーナーノード**: 2面が交わるコーナーノードは axis(v_r=0) が最優先、次に各面の BC を順次適用
- **レジスタ**: ~10
- **メモリ**: 境界ノード/セルのみアクセス。ゴーストセルは n_cells 以降に配置

---

## 3. ALE カーネル群

> **適用条件**: ALE rezone/remap は **2D_RZ のみ** で使用する。1D_SPH では ALE は無効（Lagrangian 固定。NUMERICS §3.3 参照）。
> 実装時は `if (cfg.main.geometry == "2D_RZ" && cfg.mesh.motion == "ale" && cfg.mesh.rezoning.enabled)` ガードで ALE Phase 全体を囲むこと。

### 3.1 A1: mesh_quality_check

```cpp
__global__ void mesh_quality_check(
    double* __restrict__ q_cell,     // [n_cells] out: セル品質指標
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    DeviceErrorFlags* __restrict__ error_flags,  // mesh_tangle: J_min < 0 検出（§0.6）
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 4 Gauss点でヤコビアン評価 → `q_c = J_min/J_max`（NUMERICS §3.3.2）
  - **退化ガード**: `J_max_eff = max(J_max, J_floor)` (J_floor = 1e-30)。`J_max < J_floor` の場合 `q_c = 0.0` とし `mesh_tangle` フラグ設定
  - `J_min < 0` 検出時: `atomicExch(&error_flags->mesh_tangle, 1)`（ARCHITECTURE §10.1: 即ERROR）
- 後段で CUB `DeviceReduce::Min` → `q_min < q_threshold` で rezone 発動判定
- **レジスタ**: ~15（4 Gauss点のヤコビアン値 + min/max 一時変数）
- **メモリ**: 4頂点座標読み込み（構造格子固定ストライド）、coalesced write（q_cell）

Related Hydro2D pre-commit diagnostics in
`src/hydro/corner_jacobian_quality.cu` reuse the same multiblock
RZ-volume cubic/root logic as the production mesh-quality dt limiter.
`mesh_quality_rz_volume_cell_margin_kernel` is default-off and writes only
per-requested-cell diagnostic `sigma`/safe-eta buffers for the
`[ring7_seam]` scaffold and the Ring7 seam optimizer B_V
acceptance check; it does not update dt, mesh, or state.  The active Ring7
path may pass candidate start coordinates and candidate cell volumes to the
same kernel via
`compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume()` so the
floor volume matches the proposed zero-time seam geometry.  The reactive
driver retry path only requests the StepStart Ring7 transaction after the
production dt-CFL reports `mesh_quality_rz_volume`; the candidate acceptance
still uses this same diagnostic-margin wrapper around the production kernel.
Increment 3a then builds the dedicated `[ring7_seam_packet]` geometry ledger on
the host and commits no CUDA state or coordinate writes.  Increment 4b keeps
the driven-pole cap remap host-side as well: it reuses the production
`compute_mesh_quality_dt_limit()` CUDA path as the proxy and recomputed
validation oracle, but the conservative pole-cap packet ledger itself is a
deterministic host transaction and launches no new CUDA kernels.

### 3.2 A2: winslow_jacobi_step

```cpp
__global__ void winslow_jacobi_step(
    double* __restrict__ x_r_new,    // [n_nodes] out
    double* __restrict__ x_z_new,    // [n_nodes] out
    const double* __restrict__ x_r,  // [n_nodes] in (前反復)
    const double* __restrict__ x_z,  // [n_nodes] in
    const uint8_t* __restrict__ node_flags, // 境界フラグ
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_nodes + 255) / 256`（境界ノードはカーネル内で早期リターン）
- **反復**: ホスト側で最大 `max_iterations`（既定20）回ループ（ダブルバッファで swap、NUMERICS §3.3.3 準拠）
- **境界ノード**: `NODE_BOUNDARY | NODE_AXIS | NODE_CENTER` フラグ付きノードは位置固定
- **node_flags ビットマスク定義**: `NODE_BOUNDARY=0x01, NODE_AXIS=0x02, NODE_CENTER=0x04`。
  `(node_flags[n] & (NODE_BOUNDARY|NODE_AXIS|NODE_CENTER)) != 0` のノードは rezone で位置固定。
  NODE_BOUNDARY: 計算領域外周ノード、NODE_AXIS: r=0 軸上ノード（2D_RZ のみ）、
  NODE_CENTER: 球対称中心ノード（1D_SPH のみ）。`node_flags` は `[n_nodes]` サイズ
- **計量係数 α,β**: 隣接ノード座標から on-the-fly 計算（NUMERICS §3.3.3 Winslow 計量係数、共有メモリ不要）
- **レジスタ**: ~20
- **メモリ**: 隣接ノード 8 点（十字+対角）の座標読み込み。構造格子固定ストライドで参照
- **収束判定**: ホスト側で毎反復ノードあたりユークリッド変位の最大値 `δ_rezone = max_n sqrt(Δr_n² + Δz_n²)` を CUB `DeviceReduce::Max` で評価。`δ_rezone < convergence_tol × Δl_min_global`（既定 convergence_tol=1e-6）で早期終了（NUMERICS §3.3.3、SPECIFICATION §6.4.2）。**MPI注意**: `Δl_min_global = MPI_Allreduce(MIN, Δl_min_local)` でグローバル化した値を使用すること（§9 Phase 5 参照）。ローカル Δl_min を使うとランク間で break 判定が不一致し halo_exchange デッドロックが発生する

### 3.3 A3: conservative_remap

```cpp
__global__ __launch_bounds__(256, 4) void conservative_remap(
    double* __restrict__ field_new,      // [n_cells] out
    const double* __restrict__ field_old, // [n_cells] in
    const double* __restrict__ vol_old,
    const double* __restrict__ vol_new,
    const double* __restrict__ x_r_old,  // old mesh
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,  // new mesh
    const double* __restrict__ x_z_new,
    int sweep_direction,                 // 0=r方向, 1=z方向（Strang-type交替スイープ、§3.3.4）
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 1方向 flux-based remap + Van Leerスロープリミッタ（NUMERICS §3.3.4）
- **方向分離**（NUMERICS §3.3.4 必須）: 2Dのremapをr方向・z方向に分離し逐次実行。
  Strang-type で交替スイープ: 偶数ステップ r→z、奇数ステップ z→r（O(Δt²) 精度維持）。
  **偶奇判定**: グローバルな hydro ステップ番号 `step_number` で決定する（A3 の呼び出し回数やフィールドインデックスではない）。
  `first_dir = (step_number % 2 == 0) ? 0 : 1; second_dir = 1 - first_dir;` として、
  全保存量を同一ステップ内で同一方向順序 (first_dir → second_dir) で remap する。
  各保存量（mass, momentum_r, momentum_z, e_i, e_e, volFrac）に対し、
  1方向ずつ計2回起動（sweep_direction パラメータで r/z を指定）
- **state_supply z-face flux**: z_bottom/z_top が active `state_supply` の場合、boundary face speed は projected node `v_z` ではなく `supply_u_z_cm_per_s` を直接使う。これにより mesh clamping（`x_z=z_min/z_max`, mesh `v_z=0`）と open-flow material flux（\(\rho_s u_{z,s} A\Delta t\)）を分離する。
- **レジスタ**: ~30
- **メモリ**: old/new mesh 座標 + フィールド値の読み込み（構造格子固定ストライド）
- **ワープ発散**: Van Leer リミッタの min/max 分岐があるが、全スレッドが同一パスを実行するケースが大半

### 3.4 A4: project_cell_velocity_to_nodes

```cpp
__global__ void project_cell_velocity_to_nodes(
    double* __restrict__ v_r_node,          // [n_nodes] out: ノード速度R成分
    double* __restrict__ v_z_node,          // [n_nodes] out: ノード速度Z成分
    const double* __restrict__ v_r_cell,    // [n_cells] in: セル中心速度R成分
    const double* __restrict__ v_z_cell,    // [n_cells] in: セル中心速度Z成分
    const double* __restrict__ rho,         // [n_cells] in: セル密度 [g/cm³]
    const double* __restrict__ vol,         // [n_cells] in: セル体積 [cm³]（質量重み m_c = rho[c]*vol[c]）
    int nr, int nz
);
```

- **block**: 256, **grid**: `(n_nodes+255)/256`
- **処理**: remap 後のセル中心量からノード速度を再構築（NUMERICS §3.3.4 節点速度投影）。
  各ノードに隣接するセル（最大4セル for 2D_RZ）の速度を**質量重み平均**で投影:
  `v_node = Σ_c (m_c × v_c) / Σ_c m_c`  （m_c = ρ_c × V_c、NUMERICS §3.3.4 準拠）
  - 1D_SPH: 左右2セルの質量重み平均（境界は1セル）
  - 2D_RZ: 4隣接セルの質量重み平均。軸・境界では隣接セル数に応じて動的に重み算出
- **境界ノード**: 隣接セル数が少ない（角: 1, 辺: 2）→ 実在セルのみで加重平均。投影後の velocity boundary mode は 0=free, 1=reflect, 2=fixed, 3=state_supply。mode 1 は reflect-style に法線成分をゼロ化し、mode 2 は全成分を固定する。mode 3 は state_supply z-boundary の material `v_z` を拘束せず、supplied/restored `u_z_cm_per_s` を保持する。
- **レジスタ**: ~10
- **メモリ**: 隣接セル値の読み込み（構造格子固定ストライド）。ノードへのcoalesced write

### 3.5 A5: normalize_volFrac

```cpp
__global__ void normalize_volFrac(
    double* __restrict__ volFrac,      // [n_cells × n_mat] in/out: 体積分率
    DeviceErrorFlags* __restrict__ error_flags,  // volfrac_degenerate 報告用
    int n_cells, int n_mat
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: ALE remap 後に `Σ_mat volFrac[c, mat] = 1` を強制する正規化カーネル（NUMERICS §3.3.4, ARCHITECTURE §4.3）。
  1. **非負クランプ**: `volFrac[c, m] = max(volFrac[c, m], 0.0)` （Van Leerスロープリミッタ由来の微小負値を除去。NUMERICS §3.3.4 準拠）
  2. `sum = Σ_{m=0}^{n_mat-1} volFrac[c, m]`（クランプ後の非負値の和）
  3. `sum ≥ ε_vf` (1e-30) の場合: `volFrac[c, m] /= sum`（全材料を正規化）
  4. `sum < ε_vf` の場合（退化）: `α* = argmax_m volFrac[c, m]`（クランプ前の値で判定）に対し `volFrac[c, α*] = 1.0`、他を `0.0` に設定。`error_flags->volfrac_degenerate` 設定（NUMERICS §3.3.4 退化ガード準拠）
- **呼び出し位置**: §9 Phase 5 ALE remap 直後（A3 × 2回 の後）
- **レジスタ**: ~8
- **メモリ**: `volFrac` を1パスで読み書き。coalesced access（材料次元が内側）
- **注意**: 単材料（`n_mat == 1`）の場合は `volFrac[c, 0] = 1.0` を直接設定（正規化不要）

---

## 4. Conduction カーネル群

### 4.1 C1: compute_spitzer_deff

```cpp
__global__ void compute_spitzer_deff(
    double* __restrict__ D_eff,      // [n_cells] out
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ Zbar,
    const double* __restrict__ Cv_e,  // [n_cells] 質量比熱 c_v,e [erg/(g·eV)]（H13出力）
    const double* __restrict__ x_r,  // [n_nodes] ノードR座標（|∇T|近似のセル間距離計算に必要）
    const double* __restrict__ x_z,  // [n_nodes] ノードZ座標（2D_RZ用。1D_SPHではNULL可）
    const double* __restrict__ A_eff, // [n_cells] 有効原子量（ne = rho * Zbar / (A_eff[c] * m_p) をカーネル内で計算。
                                      //   単材料: A_mat[0]、多材料: §1.1.5(c) 調和平均。U8 出力を使用）
    double f_lim,                    // flux limiter係数
    int n_cells, int nr, int nz     // 格子次元（隣接セル特定に必要。1D_SPH: nz=1）
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: κ_SH → q_SH → q_max → q_limited → D_eff = |q|/(ρ c_v |∇T|)（NUMERICS §4.1-4.2。c_v は質量比熱 [erg/(g·eV)]）
- 勾配 |∇T|_c の計算（NUMERICS §4.3 準拠）:
  - **2D_RZ**: Kershaw B-演算子（Appendix A.4）を用いてノード (i,j) での勾配を構成。
    各ノードは周囲4セルの T_e からB-演算子で ∇T|_n を計算。
    セル中心では4コーナーノード勾配の算術平均: |∇T|_c = (1/4)Σ_{k=1}^{4} |∇T|_{n_k}
  - **1D_SPH**: 隣接セル温度差 / セル間距離（直接差分）
- **ε_grad ガード**（NUMERICS §4.3 必須）: |∇T|_c < ε_grad の場合 D_eff = D_SH（Spitzer無制限）。
  ε_grad = max(1e-10 × T_{e,c}/ℓ_c, 1e-30 [eV/cm])、ℓ_c = V_c^{1/3}。
  この相対的閾値により、ほぼ均一温度領域で D_eff の非物理的発散を防ぐ
- **レジスタ**: ~20
- **メモリ**: 隣接セルの Te 読み込み（2D: 4近傍 → 構造格子固定ストライド）、`__ldg()` でセルデータ参照
- **1D の実装**（`compute_spitzer_deff_1d_kernel`、`conduction_bodies.cuh`）：出力はセルの Spitzer–Härm 伝導率 \(\kappa_{SH}\)
  （と伝導率の床・上限）で、\(D_\mathrm{eff}\) は陽的な安定刻み \(\Delta t_\mathrm{exp}\) の見積もりにだけ使う。流束制限は
  `compute_1d_flux_limiter_faces_kernel` が面ごとに与え、面の係数は `compute_1d_face_kappa_kernel` がステップの始めの
  \(T_e^n\) で一度だけ評価する（既定 `face_kappa_policy="kirchhoff_same_material"`、NUMERICS §4.2.1）

### 4.2 C2: kershaw_stencil_build（2D RZ専用、最重要）

```cpp
__global__ __launch_bounds__(256, 2) void kershaw_stencil_build(
    double* __restrict__ stencil,    // [N × 9] out: 9点係数。N=n_cells（伝導）or n_cells+n_ghost（輻射R3用）
    const double* __restrict__ D_eff, // [N]（伝導: §4.3 の D_eff、輻射: 1/(3σ_{R,g})）
    const double* __restrict__ x_r,  // [n_nodes]（ゴーストノード座標含む）
    const double* __restrict__ x_z,  // [n_nodes]（同上）
    const uint8_t* __restrict__ face_bc_type, // [4] 物理境界タイプ: {r_lo, r_hi, z_lo, z_hi}。0=内部/MPI, 1=reflect, 2=vacuum
    bool apply_mmatrix_repair,       // True=伝導用（修復後出力）、False=輻射R3用（修復前出力）
    int* __restrict__ mmatrix_fix_count, // [1] out: M-matrix修復適用セル数（atomicAdd、apply_mmatrix_repair=True時のみ使用。False時はnullptr可）
    int nr, int nz,
    int n_ghost                      // 0=伝導用（owned cells のみ）、>0=輻射R3用（ghost cells 含む）
);
```

- **block**: 256, **grid**: `((n_cells+n_ghost)+255)/256`（伝導用: n_ghost=0、輻射R3用: n_ghost>0）
- **メモリレイアウト**: `stencil[c*9 + k]` — セルmajor。k=0:C, 1:E, 2:W, 3:N, 4:S, 5:NE, 6:NW, 7:SE, 8:SW
- **処理**（Appendix A準拠）:
  1. セル(i,j)の4コーナーノード座標を読む
  2. 4ノードの辺中点・セル中心座標を計算（A.2）
  3. 4ノードの A, B ベクトル・ヤコビアン J を計算（A.3）
  4. σ, λ（面の調和平均拡散係数）を計算（A.5）
  5. **RZ幾何因子の適用**（A.7、係数計算の**前**に実行）:
     σ_{i,j+1/2} → σ_{i,j+1/2} × r_{i,j+1/2}（i-面中点のR座標で重み付け）
     λ_{i+1/2,j} → λ_{i+1/2,j} × r_{i+1/2,j}（j-面中点のR座標で重み付け）
     **注**: r=0 軸上ではσ項がゼロ化され、軸対称反射BCが自然に実現（A.7）。
     この置換をA.6の係数計算の**前**に行うことが必須（後処理ではなく入力の重み付け）
  6. a_E, a_W, a_N, a_S（直接隣接）をRZ重み付きσ,λから計算（A.6）
  7. ρ^{1-4}（交差項）→ a_NE, a_NW, a_SE, a_SW を計算（A.6）
  8. a_C = -(a_E+a_W+a_N+a_S+a_NE+a_NW+a_SE+a_SW)
  9. **物理境界係数折り込み**（NUMERICS Appendix A.8 準拠）:
     境界セル（i=0, i=nr-1, j=0, j=nz-1）で、ドメイン外を向く隣接係数を処理:
     - **Reflect/Neumann**（face_bc_type=1）: φ_ghost = φ_interior に相当 → a_C += a_boundary; a_boundary = 0。
       コーナー係数（a_NE等）もドメイン外ならば同様に折り込む。行和ゼロ維持
     - **Vacuum/Robin**（face_bc_type=2、設計時は退役した DDMC の R3 用）: φ_ghost = -φ_interior × (d_ext-Δx/2)/(d_ext+Δx/2)
       に相当 → a_C += a_boundary × (-(d_ext-Δx/2)/(d_ext+Δx/2)); a_boundary = 0。d_ext = 0.7104/σ_tr
     - **内部/MPI**（face_bc_type=0）: 折り込みなし（MPI ghost cell を C3 が読む）
     - r=0軸は RZ重み付け（step 5）で σ=0 となり、反射BCが自然に実現済み（追加処理不要）
- **レジスタ**: ~48（A,B ベクトル 4組、σ,λ 4面、ρ^{1-4} 4点、係数9個、BC一時変数3個）
  - `__launch_bounds__(256, 2)` で最低25% occupancy（512 threads/SM）を保証。
    実行時は ~48 reg × 256 = 12288 reg/block → 最大4 blocks/SM → ~50% occupancy が期待されるが、
    コンパイラの最適化余地を確保するため `min_blocks=2` を指定
  - 計算量が支配的（compute-bound）であり、occupancy の低下は許容
- **M-matrix修復**（`apply_mmatrix_repair=True` 時のみ実行、NUMERICS Appendix A.6 準拠）:
  1. 各セルで a_{off-diag} ≤ 0 を検証
  2. **正のオフ対角クランプ**: a_corner > 0 の場合、a_corner ← 0 に設定（対角隣接のみ: a_NE, a_NW, a_SE, a_SW）
  3. **対角再計算**: a_C = -(a_E+a_W+a_N+a_S+a_NE+a_NW+a_SE+a_SW) を再計算し行和ゼロを維持
  4. 修復適用セル数を `atomicAdd(&fix_count, 1)` で集計 → diagnostics/kershaw_mmatrix_fix_count に記録
  5. fix_count > 0.1 × n_cells の場合、ホスト側で WARNING 出力（メッシュ品質劣化の兆候）
  - **2つの呼び出しモード**:
    - `apply_mmatrix_repair=True`（伝導用、Phase 2）: 修復後の係数を出力し安定な explicit 更新を行う
    - `apply_mmatrix_repair=False`（設計時は退役した DDMC の R3 用、Phase 4）: 修復前の raw 係数を出力。R3 `ddmc_leak_coeff` が M-matrix 違反を検出し、違反セル×群を IMC にフォールバックしていた（NUMERICS §7.3.3。DDMC とともに退役）

### 4.3 C3: kershaw_apply

```cpp
__global__ __launch_bounds__(256, 4) void kershaw_apply(
    double* __restrict__ Te_new,     // [n_cells] out
    const double* __restrict__ Te,   // [n_cells] in
    const double* __restrict__ stencil, // [n_cells × 9]
    const double* __restrict__ rho,
    const double* __restrict__ Cv_e,  // [n_cells] 質量比熱 c_v,e [erg/(g·eV)]（H13出力）
    const double* __restrict__ vol,
    double dt_sub,                   // サブステップ幅
    double T_floor,
    int nr, int nz,
    int* __restrict__ clamp_count    // 温度フロア適用回数（atomic）
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 1スレッド=1セル。9近傍の Te を読み、ステンシル積 → ΔTe → Te_new（NUMERICS Appendix A.9）
- **Super-Time-Stepping (STS)**: ホスト側で `s` ステージ分ループ。各ステージ後にダブルバッファ swap
  - `s = min(max(1, ceil(sqrt(2 * dt / dt_exp))), config.sts_max_stages)` — NUMERICS §4.2.1 準拠。`sts_max_stages`（既定40）でクランプ
  - 典型 s = 1–5（爆縮初期）、16–27（コロナ発達時）。素朴法の N_sub=130–340 を大幅に削減
  - 各ステージのサブステップ幅 τ_j は Chebyshev 根分布（不均一）:
    `τ_j = dt_exp / (ν² + (1-ν²) cos²(π(2j-1)/(4s+2)))`, ν=0.01（NUMERICS §4.2.1）
  - **タイムステップスケーリング**（NUMERICS §4.2.1 必須）: `dt_sts = Σ τ_j; τ_j ← τ_j × Δt/dt_sts`。
    Chebyshev 根分布はΣτ_j ≠ Δt となるため、全τ_jを一様スケーリングして Στ_j = Δt を保証。
    この正規化を省略すると伝導の実効積分時間が hydro ステップと不一致になり、温度更新が系統的にズレる
  - D_eff と Kershaw 係数はスーパーステップ開始時に凍結（C1→C2 は1回のみ、C3 を s 回実行）
  - ダブルバッファ swap はホスト側のポインタ交換のみ（カーネル内操作なし）
- **負温度防止**: `Te_new = max(Te_new, T_floor)` + clamp_count 加算
- **局所安全策**（NUMERICS §4.2.2、実装は `conduction_alpha_pass*_kernel`）: セルの床より上のエネルギー
  \(\rho c_v (T - T_\mathrm{floor})\) を、正味の流出（`sts_floor_limiter="net"`、既定）または流出の和（`"donor"`）で割った
  時間を \(\tau_j\) と比べて \(\alpha_c = \min(1, \cdot/\tau_j)\) を作り、面の流束を両側で同じ係数（net は
  \(\min(\alpha_c,\alpha_{nb})\)、donor は流出側のセルの \(\alpha\)）でスケーリングする。対の反対称が保たれるので
  スケーリング自体はエネルギーを作らない。旧設計の `Δt_safe = Δl²/(2 D_eff)` と、非対称分を `E_safety` に計上する方式は
  無い。エネルギーの計上が生じるのは最後の床クランプ（`E_floor_injected`）だけ
- **レジスタ**: ~15
- **同期**: STS の全ステージは同一 compute ストリーム上で起動されるため、CUDA ストリーム順序保証によりカーネル間は自動的に逐次実行される。ダブルバッファ swap（ホスト側ポインタ交換）に明示的 `cudaStreamSynchronize` は不要。ただし MPI halo exchange が介在する場合は、P1(pack) カーネル完了後に MPI 送信が必要なため、halo exchange ルーチン内部で `cudaStreamSynchronize(compute)` を実行する

### 4.4 C4: conduction_1d_tridiag（1D_SPH用）

> **実装**：このカーネルは存在しない。1D の電子伝導は次の 2 経路（`conduction.cu`）：
> - **STS**（既定）：`conduction_1d_sts_stage_kernel`（調和平均の面）、`_kirchhoff_kernel`・`_secant_kernel`（割線の面）、
>   `conduction_1d_sts_fused_kirchhoff_kernel`。面の係数はステップの始めの値で固定し、ステージは `kappa_face` を読む。
>   面積は形状ごと（球 \(4\pi r^2\)、円筒・平板は `geometry_1d_face_area`）
> - **陰解法**（`Numerics.conduction.solver="implicit"`）：`build_1d_implicit_system{,_kirchhoff}_kernel` で三重対角を組み、
>   cuSPARSE `gtsv2` で温度 \(T^*\) を解く。エネルギーは `implicit_conduction_flux_form_kernel` が面の流束
>   \(G_f = -u_f\,(T^*_{f+1}-T^*_f)\)（\(u_f\) は組んだ行列の上対角）の形で記帳する
>   （\(\Delta T_i = \Delta t\,(G_i-G_{i-1})/(\rho c_v V)_i\)、NUMERICS §4.2.3）
>
> イオン伝導は `ion_conduction_*_kernel`（`ion_conduction_pcr_solve_kernel` の並列巡回縮約）、非局所伝導（SNB）は
> `conduction_snb_1d.cu` の `snb_*_kernel`。以下は設計時の記述。

```cpp
__global__ void conduction_1d_tridiag(
    double* __restrict__ Te_new,
    const double* __restrict__ Te,
    const double* __restrict__ kappa_eff, // [n_nodes] 面の伝導率
    const double* __restrict__ r_node,    // [n_nodes]
    const double* __restrict__ rho,
    const double* __restrict__ Cv_e,  // [n_cells] 質量比熱 c_v,e [erg/(g·eV)]（H13出力）
    const double* __restrict__ vol,
    double dt_sub,
    double T_floor,
    int n_cells,
    int* __restrict__ clamp_count
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: NUMERICS §3.1.7 の離散化。面積 A_j = 4πr_j² を含む
- **レジスタ**: ~15
- **メモリ**: 隣接ノード座標 + Te 読み込み（1Dなので左右2近傍のみ、coalesced）
- **Super-Time-Stepping (STS)**: C3 と同様に、ホスト側で `s` ステージ分ループ。各ステージ後にダブルバッファ swap（ホスト側のポインタ交換のみ、カーネル内操作なし）。STS パラメータ（ステージ数 s、サブステップ幅 τ_j、Στ_j=Δt 正規化含む）は C3 と同一式（NUMERICS §4.2.1 準拠）。CUDA 同一ストリーム上のカーネル起動は順序保証されるため、ポインタ交換に明示的 `cudaStreamSynchronize` は不要

### 4.5 Hypre 陰的ソルバ統合（オプション、`-DTENRYU_ENABLE_HYPRE=ON`）

`conduction.solver="hypre"` 時の処理フロー（NUMERICS §4.2.3 参照）。
**2D_RZ 専用**: C2（Kershaw 9点ステンシル構築）が前提のため、1D_SPH では使用不可。
`geometry="1D_SPH"` かつ `conduction.solver="hypre"` の場合、STS（C4）に自動フォールバックし WARNING を出力する：

1. **C1 + C2 カーネル**：STS パスと共通（`D_eff` 計算 → Kershaw 9点ステンシル構築）
2. **行列変換**（ホスト側コード、デバイスメモリ上で動作）：
   - C2 出力のステンシル係数 `stencil[n_cells × 9]` を `HYPRE_IJMatrixSetValues` で ParCSR 行列に転写
   - 対角に \(C_v / \Delta t\) を加算（\(C_v = \rho c_v\) [erg/(cm³·eV)]、質量行列項。NUMERICS §4.2.3 準拠）
   - 初回のみスパーシティパターン構築（`HYPRE_IJMatrixSetRowSizes`、9 entries/row 固定）
   - 2回目以降は `HYPRE_IJMatrixSetValues` で値のみ更新
3. **Te^n 退避**：Hypre solve が State.Te を上書きするため、solve 前に `Te_old[n_cells]`（Scratch バッファ）へコピーする。
   Step 6 の E_solver 計算で \(T_{e,c}^{n+1} - T_{e,c}^n\) を算出するために必須。
   STS パスではダブルバッファで Te^n が自然に保持されるため本ステップは不要。
4. **Hypre solve**：`HYPRE_ParCSRPCGSolve`（BoomerAMG前処理、デバイスメモリ上で完結）
5. **解の書き戻し**：`HYPRE_IJVectorGetValues` → State.Te 配列 → EOS forward (H13: Te→ee, Pe, Cv_e) で再同期（NUMERICS §4.2.1 準拠。EOS inverse ではなく forward を使用する — Hypre は Te を直接解くため、Te→ee の順方向変換が必要）
6. **E_solver 計算**（NUMERICS §4.2.3）：
   \(E_{solver} = \sum_c C_{v,c}(T_{e,c}^{n+1}-T_{e,c}^n) V_c - \Delta t \sum_c (\nabla\cdot q)_c V_c\)
   ここで \(C_{v,c} = \rho_c \cdot c_{v,e,c}\) [erg/(cm³·eV)]、\(T_{e,c}^n\) は Step 3 の `Te_old` から取得。
   CUB `DeviceReduce::Sum` で集約し、Phase 6 D2H で `State.E_solver += step_E_solver` に累積する。
   STS パスでは本ステップは省略（E_solver=0）

> **カーネル起動の観点**：Hypre パスでは C3 カーネル（`kershaw_apply`）は **起動しない**。
> 代わりに Hypre 内部が AMG V-cycle と PCG 反復を実行する。
> C1, C2 は両パスで共通であり、ステンシル構築コードの重複はない。
>
> **Hypre のメモリ管理**：Hypre 2.25+ は `HYPRE_SetMemoryPoolAllocator` で
> TENRYU の Scratch バッファ（§0.4）を共有できる。
> ただし v1.0 では Hypre 独自のメモリプールを使用し、Scratch 共有は将来最適化とする。

---

## 5. Laser カーネル群

### 5.1 レーザー格子への写像（現行）

> **設計時の L1 `laser_mesh_map`・L2 `compute_density_gradient`・L5 `deposit_lm_to_hydro` は実装に存在しない。**
> 現行の写像と沈着の経路は次のとおり（ARCHITECTURE §4.6、NUMERICS §5.7–§5.8）。

**1D**（`laser_mesh.cu` の `map_from_hydro_1d` → `map_trace_profile_1d`、写像のたびに節点を作り直す）：

```cpp
__global__ void map_hydro_to_laser_1d_kernel(
    const double* __restrict__ node_R,       // 節点の R（2 回目の起動では径方向プロファイル節点の r）
    const double* __restrict__ node_Z,       // 節点の Z（2 回目は 0 を指す 1 要素）
    const double* __restrict__ rho, const double* __restrict__ Te, const double* __restrict__ zbar,
    const double* __restrict__ A_eff,        // [n_cells] device scratch
    const uint8_t* __restrict__ cell_is_void,// [n_cells] device scratch
    const double* __restrict__ r_edges,      // [n_cells+1] 流体の節点の半径
    double* __restrict__ n_hat, double* __restrict__ n_hat_raw,
    double* __restrict__ Te_LM, double* __restrict__ Zbar_LM,
    int n_nodes_total, int n_nodes_z, int n_cells,
    /* n_crit, ghost corona, critical reconstruction, clip scalars */);
```

- **block**: 256。1 スレッド = 1 節点。\(r=\sqrt{R^2+Z^2}\) で流体セルを device 上の二分探索で特定し、密度の正規化、
  臨界に隣接するセル対の対数線形プロファイル（NUMERICS §5.7.3(a)）、ゴーストコロナ、critical clip を評価する
- **2 回起動する**：(1) 2D レーザー格子の節点（`n_nodes_r × n_nodes_z`）、(2) `place_trace_profile_nodes_1d`
  （判定・区分ごとの節点数・最後の区分の尾部の節点数・CUB の前置和・区分の書き込み・尾部の書き込み・仕上げ。区分の
  数え上げと書き込みと尾部の書き込みは多数のブロック、判定と尾部の数え上げは 1 ブロック、順の配置に戻るときは 1 スレッド）
  が置いた径方向プロファイル節点（面・臨界の対・プロファイルの節点）。節点数は host へ
  1 回だけ返す。(2) の後に `map_node_material_1d_kernel`（同じ材料の \(n_e\) 補間用の節点の材料）、
  `compute_radial_gradient_kernel`（\(d\hat n/dr\)）、`compute_smooth_kappa{,_ext}_kernel`（逆制動輻射の smooth 係数）
- **step 間平滑化なし**: 1D の clipped \(\hat n\) は写像した値そのもの（`ema_smooth_n_hat_radial_kernel` は 2026-09-24 に撤去、NUMERICS §5.7.4）

**2D_RZ**（`map_from_hydro_2d`）：流体セルからレーザー格子の節点への補間は host で行い（`map_from_hydro_2d_impl`）、
device へ写したあと `compute_gradient_kernel`（中心差分の \(\partial\hat n/\partial R\)、\(\partial\hat n/\partial Z\)、軸で
\(\partial\hat n/\partial R=0\)、端は片側差分）と smooth 係数のカーネルを起動する。

**沈着の転写**：
- 1D の光線追跡は吸収したパワーを流体セルへ直接積む（`deposit_power_cell`、レーザー格子の節点を経由しない）。
  臨界付近の再配分（`apply_deposit_redistribution_1d`、`deposit_transfer.cu`）は host で行い、\(\Delta t\) を掛けて
  `laser_dep` [erg] にする。`transfer_to_1d` は試験だけが呼ぶ
- 2D_RZ は `transfer_2d_kernel`（1 スレッド = 1 流体セル、セル中心でのレーザー格子の双線形補間 × \(\Delta t\)）

### 5.1d L6: ray_skip_check

```cpp
__global__ void ray_skip_check_kernel(      // src/laser/raytrace_skip.cu
    double* __restrict__ delta_max_cell,    // [n_cells] out: セルごとの変化量（max_relative は 3 指標の最大、l2_relative は 3 指標の二乗和）
    int* __restrict__ crit_hit,             // [1] out: 臨界帯を横断したセルがあれば 1
    const double* __restrict__ rho,         // [n_cells] in: 現ステップ ρ（HydroMesh）
    const double* __restrict__ Te,          // [n_cells] in: 現ステップ Te（HydroMesh）
    const double* __restrict__ Zbar,        // [n_cells] in: 現ステップ Z̄（HydroMesh）
    const double* __restrict__ volFrac,     // [n_cells × n_mat] in: 体積分率（多材料の A_eff 用）
    const double* __restrict__ A_mat,       // [n_mat] in: 材料の原子量
    const int n_mat,
    const double* __restrict__ rho_cached,  // [n_cells] in: 前回レイトレース時 ρ
    const double* __restrict__ Te_cached,   // [n_cells] in: 前回レイトレース時 Te
    const double* __restrict__ Zbar_cached, // [n_cells] in: 前回レイトレース時 Z̄
    const double rho_floor, const double Te_floor, const double Zbar_floor,  // 分母のフロア（Zbar_floor = 1e-2）
    const double n_crit,                    // 臨界密度 n_crit(λ) [1/cm³]
    const double n_hat_margin,              // Laser.lasermesh.critical_margin
    const double crit_guard,                // 臨界帯の幅（既定 0.01）
    const double A_eff_uniform,             // A_mat が無いときの原子量
    const int use_l2_relative,              // 1 = l2_relative、0 = max_relative
    const int n_cells);
// HydroMesh の全セルを処理する（キャッシュ配列も HydroMesh のセル数）。
// max_relative: δ = max(|Δρ|/max(ρ_cached, ρ_floor), |ΔTe|/max(Te_cached, Te_floor), |ΔZ̄|/max(Z̄_cached, Z̄_floor))
// l2_relative:  分母は現在値 max(|x|, x_floor) で、3 指標の二乗和を書く
// 臨界帯: A_eff は単一材料なら A_mat[0]、多材料は体積分率で重み付けした 1/A の和の逆数。
//   n_e = ρ Z̄ / (A_eff m_p) を現在値とキャッシュ値の両方で求め、n_e/n_crit が帯域 n_hat_margin − crit_guard を
//   横断した（一方だけが帯域を超える）セルがあれば crit_hit = 1（NUMERICS §5.9.4）
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: HydroMesh の各セルで 3 指標 δ_ρ, δ_Te, δ_Z̄（NUMERICS §5.9.2）と臨界帯の横断（§5.9.4）を 1 回の起動で求める
- **後段**: `crit_hit` が 1 なら再計算する。そうでなければ δ をホストへ転送し、ホストで max_relative は最大値、l2_relative は \(\sqrt{\sum/(3N)}\) を求め、`Laser.raytrace_skip.threshold`（既定 0.01）未満ならレイトレースを省略する
- **レジスタ**: ~8

### 5.1e レイトレース省略のキャッシュ更新

> **実装**：設計時の `laser_cache_update` カーネルは無い。キャッシュの更新は host の `RaytraceSkipCache::update_cache`
> （`raytrace_skip.cu`）が行う：\(\rho, T_e, \bar Z\) を device 間コピーで保存し、群ごとの沈着の比 \(\hat f_g\)（host で
> セルの吸収パワー ÷ 群の入射パワー \(P_g\) として作る）を device へ写す（NUMERICS §5.9.3）。

### 5.1f helper: reconstruct_laser_dep

```cpp
__global__ void reconstruct_laser_dep_kernel(     // src/laser/raytrace_skip.cu
    double* __restrict__ laser_dep,               // [n_cells] out: 再構成沈着エネルギー [erg]
    const double* __restrict__ f_hat,             // [n_groups * n_cells] in: f̂_g_cached [無次元]
    const double* __restrict__ P_g_dt,            // [n_groups] in: グループ g の P_g(t_now)×Δt [erg]
    int n_groups, int n_cells
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: per-cell 変換カーネル。`laser_dep[c] = Σ_g f̂_g[g * n_cells + c] × P_g_dt[g]` を計算し、skip path 用の沈着配列を再構成する（NUMERICS §5.9.3）
- **初期化**: カーネルが各セルの和を書き込む（加算ではない）ので、呼び出し前のゼロ初期化は不要
- **n_groups=1**: `laser_dep[c] = f̂_cached[c] × P_total_dt` に退化（従来動作と同一）

### 5.2 L3/L4: ray_trace（最重要レーザーカーネル）

> **現行の 1D（2026-09-29）**：1D の既定の積分法は特性線法（`Laser.raytrace.integrator="auto"` → `"characteristic"`、
> `ray_trace_1d_characteristic`、NUMERICS §5.3.6）。球対称場では角運動量 \(B=|x\times v|\) が保存されるので、全光線で共有する
> 区切り（径方向節点・流体の面・臨界に隣接する分割半径）と光線ごとの事象半径の間の区間ごとに、弧長・極角の増分・光学的厚さを
> 適応 Gauss–Kronrod (3,7) 求積で積分する（転回点の近くでは \(u=\sqrt{r-r_0}\) に変数変換）。`kTile` 本のレーン（1、32、
> 64〜256、光線数と GPU からランチャが選ぶ）で 1 本の光線の区間を並列に評価し、前置和・前置積で各区間の入口の極角とパワーを得る。
> 臨界半径に達した光線は既定（`critical_handling.terminate` を指定しないとき）で反射して外向きに追跡する。1D の円筒・平板は
> この積分法だけが扱う。`integrator="leapfrog"` の `ray_trace_1d_sph` は刻み幅可変の Verlet（leapfrog）で球だけを扱い、既定で臨界で
> 終了する。どちらの 1D カーネルも吸収したパワーを流体セルへ直接積む。下の擬似コードのレーザー格子の 4 節点への沈着と固定刻みは、
> 2D_RZ の `ray_trace_3d`（と検証用の `ray_trace_2d`）のもの。

```cpp
// 2D version (1D_SPH用)
__global__ __launch_bounds__(64, 16)
void ray_trace_2d(
    double* __restrict__ deposit,        // [N_LM_nodes] LaserMesh沈着 [erg/s]（atomicAdd）
    const double* __restrict__ nhat,     // [N_LM_nodes] n_e/n_crit（クリップ済み）
    const double* __restrict__ nhat_raw, // [N_LM_nodes] n_e/n_crit 生値（>1 許容 — 臨界層ハンドオフ判定用、dual-field）
    const double* __restrict__ grad_nhat_r, // [N_LM_nodes]
    const double* __restrict__ grad_nhat_z, // [N_LM_nodes]
    const double* __restrict__ Te_LM,    // [N_LM_nodes]
    const double* __restrict__ Zbar_LM,  // [N_LM_nodes]
    const double* __restrict__ r_LM,     // [nr_LM+1]
    const double* __restrict__ z_LM,     // [nz_LM+1]
    const double* __restrict__ ray_R0,   // [n_rays] 初期R位置
    const double* __restrict__ ray_power, // [n_rays] パワー重み
    double ds_ray_max,                   // 空間ステップサイズ Δs_ray [cm]
    double eps_n, double eps_crit,       // 臨界パラメータ
    double lambda_L,                     // レーザー波長 [cm]
    double intensity_cutoff,             // 最小強度カットオフ [無次元]（NUMERICS §5.2、既定 1e-6）
    int nr_LM, int nz_LM, int n_rays,
    double* __restrict__ P_unabsorbed,   // [1] 未吸収パワー [erg/s]（atomicAdd）
    DeviceErrorFlags* error_flags        // MAX_RAY_STEPS到達時に infinite_loop フラグを設定（§0.6 準拠）
);
```

- **block**: 64, **grid**: `(n_rays+63)/64`
  - 1D_SPH の `ray_trace_1d_sph` は `__launch_bounds__(64)` のまま 32 スレッド（1 warp）/block で起動する（2026-09-24）。レイは前回のトレースの step 数の降順に並ぶので長いレイが先頭の warp に集まり、1 warp/block にするとそれらが別々の SM に載って、FP64 演算器の少ない GPU で同じ SM の演算器を奪い合わない（GXII 300 step の kernel 時間 36.2 → 34.8 ms/step、RTX 4090）。レイ別集約も同じ block で起動する。
- **n_rays の決定**（NUMERICS §5.6.3、SPECIFICATION §6.4.6 `rays_per_beam` 参照）：
  - **1D_SPH**（L3 `ray_trace_2d`）：`n_rays = Σ_beams rays_per_beam`。各ビームの `rays_per_beam` 本のレイを R 方向に等間隔配置（NUMERICS §5.6.3(a)）
  - **2D_RZ**（L4 `ray_trace_3d`）：`n_rays = Σ_beams N_eff(beam)`。各ビームの `rays_per_beam = N` は断面2D格子の1辺あたりの本数。円形アパーチャにより実効本数 `N_eff ≈ π/4 × N²`（NUMERICS §5.6.3(b)）
  - 既定 `rays_per_beam`：1D_SPH=1000、2D_RZ=128（2D実効本数 ~12,800本/beam）
- **1スレッド=1レイ**、内部ループ:
  ```
  // Δs_ray = C_ray_max × Δx_LM_min（LaserMesh 最小セル幅 × CFL 係数）
  // C_ray = c × Δt_ray / Δx ≤ cfl_ray（デフォルト 0.8）
  // 関係: Δt_ray = Δs_ray / c = C_ray_max × Δx_LM_min / c
  // レイトレース呼び出しごとに1回計算（LaserMesh 最小セル幅から）
  // L3/L4 の leapfrog: vR -= (Δs_ray/2) × gR

  initialize: (R,Z,vR,vZ) from ray parameters
  ds = ds_ray_max  // 空間ステップサイズ Δs_ray [cm]（記号 κ は opacity 系に予約）
  half-step velocity correction (§5.3.2)
  int n_steps = 0;
  const int MAX_RAY_STEPS = 100000;  // 無限ループ防止ガード
  while (ray is active && n_steps++ < MAX_RAY_STEPS):
      1. Find LaserMesh cell for current (R,Z)
      2. Bilinear interpolate grad_nhat at (R,Z) → (gR, gZ)
      3. Velocity update: vR -= (ds/2) * gR,  vZ -= (ds/2) * gZ  // ds = Δs_ray [cm]
      4. Position update: R += ds * vR,  Z += ds * vZ              // ds = Δs_ray [cm]
      5. Check LaserMesh bounds → if outside: atomicAdd(P_unabsorbed, I_current); terminate
      6. Check critical density（**dual-field**：ステップ間で raw n̂ を carried_nh_raw として持ち回る）:
         - nh_old >= 1-eps_crit（クリップ値）→ atomicAdd(P_unabsorbed, I_current); terminate
         - nh_old_raw >= kCritLayerHandoffNhatRaw（生値）→ **臨界層 tail-closure ハンドオフ**（`try_tail_closure`）:
           亜臨界ノードのみから A_entry を再構成して近臨界帯へ沈着、残余は P_unabsorbed へ計上して terminate
           （NUMERICS §5.2 dual-field、SPECIFICATION §5.4 ハンドオフ規約）
      7. IB absorption (§5.4, **桁落ち回避形**で実装、NUMERICS §5.4.2 準拠):
         - Interpolate nhat, Te, Zbar at old and new positions
         - n_refr2 = max(eps_n, 1 - nhat)  // 臨界面近傍での 1/sqrt(1-nhat) 発散を防止（NUMERICS §5.4）
         - Compute κ_IB at both points (η = sqrt(n_refr2) を分母に使用)
         - Δs = sqrt((R_new-R_old)² + (Z_new-Z_old)²)  // 実変位（NUMERICS §5.4.2: Δs_ray=|r^{n+1}-r^n|、ds パラメータではない）
         - Optical depth S = (κ_IB_old + κ_IB_new)/2 * Δs
         - ΔP = -I_old * expm1(-S)              // 吸収パワー [erg/s]（expm1形式で桁落ち回避）
         - I_new = I_old - ΔP                    // 差分更新（I_old*exp(-S) は使用しない — テレスコーピング和保存性）
      8. Deposit ΔP to 4 neighboring LaserMesh nodes:
         - atomicAdd(deposit[node], w * ΔP)  ← 4回  // deposit は [erg/s]
      9. Intensity cutoff (NUMERICS §5.2):
         - if (I_new < intensity_cutoff * I_0) → terminate
         - atomicAdd(P_unabsorbed, I_new)        // 残存パワーを未吸収に計上
  ```
- **レジスタ**: ~40（位置2, 速度2, 強度1, LaserMeshセル座標2, 補間重み4, κ_IB 2, 一時10）
- **ワープ発散**: レイ長のばらつきが大きい（内側レイは長い光路、外側レイは短い）
  - **緩和策**: レイをR座標でソートし、同ワープ内のレイ長が近くなるよう配置
- **Atomic競合**: deposit配列（~32K nodes）への atomicAdd
  - 同一ノードへの同時書き込みは稀（レイは空間的に分散）
  - v1.0では追加の最適化は不要と判断
- **エラー処理**: MAX_RAY_STEPS 到達時は `atomicExch(&error_flags->infinite_loop, 1)` で記録し、未吸収パワーを `P_unabsorbed` に加算

#### Diagnostic per-ray step counts

The `ray_trace_*` launcher wrappers accept an optional `d_step_count` device
array for Phase 0 diagnostics (see the internal design note laser_kernel_rewrite_plan.md
v2 §2.1). The parameter is null-by-default; passing `nullptr` is the production
path. When present, the kernel writes each ray's final loop count at
termination. This is diagnostic-only and does not affect deposition, ray
integration, or any physics result.

Verbose profiling runs may emit one stdout line per profiled beam:

```text
[laser_per_ray_steps] step=N beam=B n_rays=K max=... p90=... p50=... mean=... mean_per_warp_max=... mean_per_warp_mean=...
```

`step`, `beam`, and `n_rays` identify the hydro step, beam index, and rays in
the beam; `max`, `p90`, `p50`, and `mean` summarize per-ray step counts; and
`mean_per_warp_max` / `mean_per_warp_mean` average the per-warp maximum and
mean step counts. This is verbose-only stdout instrumentation, not HDF5 schema,
and is not a downstream-consumer contract. The first verbose call can include
`LaserMesh::ensure_per_ray_step_scratch` scratch-growth allocation cost, so
treat it as a warm-up sample for profiling.

### 5.3 3Dレイトレース（L4、2D_RZ用）

L3と同様だが、位置・速度が3Dベクトル `(x,y,z)` に拡張される（NUMERICS §5.3.4）。
LaserMesh参照は `R = sqrt(x²+y²), Z = z` で2Dに投影。
3D勾配は2D勾配から NUMERICS §5.3.4 (b) の変換で計算。
臨界面横断判定は L3 と同一の dual-field ロジック（クリップ n̂ + 生値 n̂_raw、上記 step 6）を適用する。

```cpp
__global__ __launch_bounds__(64, 16)
void ray_trace_3d(
    double* __restrict__ deposit,        // [N_LM_nodes] LaserMesh沈着 [erg/s]（atomicAdd）
    const double* __restrict__ nhat,     // [N_LM_nodes] n_e/n_crit（クリップ済み）
    const double* __restrict__ nhat_raw, // [N_LM_nodes] n_e/n_crit 生値（>1 許容 — dual-field）
    const double* __restrict__ grad_nhat_r, // [N_LM_nodes]
    const double* __restrict__ grad_nhat_z, // [N_LM_nodes]
    const double* __restrict__ Te_LM,    // [N_LM_nodes]
    const double* __restrict__ Zbar_LM,  // [N_LM_nodes]
    const double* __restrict__ r_LM,     // [nr_LM+1]
    const double* __restrict__ z_LM,     // [nz_LM+1]
    const double* __restrict__ ray_x0,   // [n_rays] 初期3D位置 x
    const double* __restrict__ ray_y0,   // [n_rays] 初期3D位置 y
    const double* __restrict__ ray_z0,   // [n_rays] 初期3D位置 z
    const double* __restrict__ ray_vx0,  // [n_rays] 初期3D方向 vx
    const double* __restrict__ ray_vy0,  // [n_rays] 初期3D方向 vy
    const double* __restrict__ ray_vz0,  // [n_rays] 初期3D方向 vz
    const double* __restrict__ ray_power, // [n_rays] パワー重み
    double ds_ray_max,                   // 空間ステップサイズ [cm]
    double eps_n, double eps_crit,       // 臨界パラメータ
    double lambda_L,                     // レーザー波長 [cm]
    double intensity_cutoff,             // 最小強度カットオフ [無次元]（既定 1e-6）
    int nr_LM, int nz_LM, int n_rays,
    double* __restrict__ P_unabsorbed,   // [1] 未吸収パワー [erg/s]（atomicAdd）
    DeviceErrorFlags* error_flags        // MAX_RAY_STEPS到達時に infinite_loop フラグ設定
);
```

- **block**: 64, **grid**: `(n_rays+63)/64`

- `__launch_bounds__(64, 16)` — L3 と同一設定
- **追加レジスタ**: +6（3D位置・速度の追加次元）→ 計 ~46
- **R=0特異性**: `R < R_floor` で横方向勾配をゼロ化（NUMERICS §5.5 R=0 軸特異性処理）
- **error_flags**: L3 と同じく `DeviceErrorFlags*` を引数に取る（MAX_RAY_STEPS到達時に infinite_loop フラグ設定）

### 5.4 L7: radial_absorption_1d_kernel

```cpp
template <bool kHotECapture>
__global__ __launch_bounds__(256, 1)
void radial_absorption_1d_kernel(
    double P_total,                              // 入射総パワー Σ_b P_b(t) [erg/s]
    const double* __restrict__ hydro_r_edges,    // [n_hydro_cells+1] Hydro 1D cell edges [cm]
    const double* __restrict__ radial_node_r,    // [n_radial_nodes] radial lookup nodes [cm]
    const double* __restrict__ radial_n_hat,     // [n_radial_nodes] clipped n̂
    const double* __restrict__ radial_n_hat_raw, // [n_radial_nodes] raw n̂
    const double* __restrict__ radial_smooth_kappa, // [n_radial_nodes] smooth κ factor
    double eps_n, double eps_crit,               // IB/critical parameters
    double test_kappa_cm_inv,                    // >0 のとき検証用固定κ [cm^-1]
    double intensity_cutoff,                     // P < cutoff*P_total で終了
    int n_hydro_cells,
    int n_radial_nodes,
    double* __restrict__ deposit_power_cell,     // [n_hydro_cells] out: 吸収パワー [erg/s]
    double* __restrict__ P_unabsorbed,           // [1] out: 未吸収パワー [erg/s]
    unsigned long long* __restrict__ critical_surface_hit_count,
    DeviceErrorFlags* __restrict__ error_flags,
    const HotECaptureParams hot_e_params,        // 熱電子の捕捉（kHotECapture のとき）
    double* __restrict__ hot_e_capture
);
```

- **launch**: `<<<1, 256, 0, stream>>>`（`kRadialAbsorptionBlock = 256`）。1D の `Laser.mode="radial_absorption_1d"` で、
  全 rank が同じ入力で起動する（沈着は rank 間で同一。2026-09-29 以前は rank 0 だけが起動し、他の rank は沈着が 0 だった）
- **処理**（NUMERICS §5.4a）：外側の流体セルから内側へ、1 本の内向きの流束として積分する
  - 1024 セルずつ（`kRadialAbsorptionChunk`）、256 スレッドが各セルの入力（中点の `n_hat`・`n_hat_raw`、逆制動輻射の κ、
    光学的厚さ \(\tau\)、セルの状態）を並列に評価して共有メモリへ置き、スレッド 0 がセルの順にパワーを運ぶ
    （以前の 1 スレッドのループと同じ算術で、結果はビット一致）
  - `n_hat_raw >= 1 - eps_crit` で臨界到達、残りの \(P\) を `P_unabsorbed` へ
  - `ΔP = absorbed_power_expm1(P, τ, P_next)`、`deposit_power_cell[c] += ΔP`
  - 最内セルに達するか `P < intensity_cutoff * P_total` で残りを `P_unabsorbed` へ
- **出力**: `deposit_power_cell` は [erg/s] のまま 1D の沈着経路へ渡り、後段で \(\Delta t\) を掛けて `laser_dep` [erg] になる
- **error_flags**: 不正な入力・半径・κ・τ で `invalid_cell`、非有限のパワーで `nan_particle` を立て、残りのパワーを未吸収へ戻す

---

## 6. Radiation カーネル群（詳細設計）

> **【状態注記 2026-09-29】** 現行の放射カーネルの設計は §6.7（FLD）と §6.8（\(S_N\)）。数理は NUMERICS §6.7/§6.8 が正。
> 本章の §6.0a〜§6.6 にあったモンテカルロ輻射のカーネル（R1 `compute_fleck_factor`、R2/R3 の DDMC モード判定とリーク係数、
> R4〜R6 の粒子ソース、R7 の合成キーソート、R8 `imc_transport_persistent`、R9 `ddmc_event_loop`、R10 `tally_finalize`、
> R12 `russian_roulette`、R13 `marshak_source`）は 2026-09-29 にコードとともに退役し、その設計を
> `retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md` へ移した。`fleck.cu`（R1 と退役した \(\Delta t_{rad}\) 制限
> `compute_dt_rad_limit`）も退役した。現行 FLD の Fleck 因子は各ソルバ側（`compute_fleck_for_fld_kernel` / `nlte_coeffs.cu`）が作る。

### 6.0a〜6.6 — 退役（モンテカルロ輻射のカーネル。`retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md`）

### 6.7 現行 FLD カーネル群（`mode="multigroup_diffusion"` — 決定論、1D/2D_RZ）【CURRENT】

2026-07-10 新設（doc 監査 open item の解消）。数理は NUMERICS §6.7、実装は
`src/radiation/fld_1d_gpu.cu` / `fld_2d_rz_gpu.cu` / `nlte_coeffs.cu`。
ブロック定数は両者 `kBlock=256`、grid は `(N+kBlock-1)/kBlock` 形式。
（退役した IMC-DDMC ハイブリッドの拡散部品 `deterministic_diffusion_1d.cu` / `diffusion_{conversion,interface,source_solve}.cu` は
FLD とは別物で、2026-09-29 に `retired/radiation_monte_carlo/` へ移した。）

#### 6.7.1 FLD 1D（`fld_1d_gpu.cu`、エントリ `advance_radiation_step_fld_1d`）

行番号は変わるので書かない（`grep -n __global__ src/radiation/fld_1d_gpu.cu`）。

| kernel | 目的 | thread mapping | 備考 |
|---|---|---|---|
| `build_eta_from_planck_kernel` | η_g = c·σ_a·a·T⁴·b_g | 1 thread/(cell×group) | constant-opacity 経路 |
| `fill_log_kappa_kernel` | 表の不透明度の対数補間の準備 | per element | |
| `compute_fleck_for_fld_kernel` | Fleck f=1/(1+z) | 1 thread/cell（群は thread 内 for） | NLTE 時は nlte_coeffs.cu が代替 |
| `compute_marshak_finc_kernel` | Marshak 群別入射流束 | 1 thread/group | |
| `assemble_fld_tridiag_kernel<GEOM>` | 三重対角 + RHS 組立 | 1 thread/(cell×group) | 面 D は面で評価した flux limiter（`fld_face_diffusion_coeff`）。GEOM ∈ {球, 円筒, 平板} |
| `publish_solution_kernel` | 解 → rad_E 公開 | 1 thread/(cell×group) | |
| `snapshot_Te_kernel` | Te→Te_old 退避 | 1 thread/cell | Fleck ブレンド基準 |
| `update_matter_kernel` | 物質側 Newton（Te/ee/Pe/rad_dep/rad_emit） | 1 block/cell、32 thread、動的 shared 2·G·8B | 群を warp 内分担。反復上限と許容値は `max_outer_iterations`・`outer_tol` を流用し、1 回の温度変化を ±T/2 に制限 |
| `fld_grey_spectrum_kernel`・`fld_grey_matrix_kernel`・`fld_grey_rhs_kernel`・`fld_grey_solve_pcr_kernel`・`fld_grey_apply_kernel` | 外側反復の灰色加速（`outer_accel="grey"`、1D の既定、`fld_1d_grey_accel.cuh`） | spectrum は warp/cell、灰色の三重対角は並列巡回縮約（1 thread/row） | 収束解は加速なしの反復と同じ |
| `build_grey_fleck_cell_mask_kernel`・`build_nlte_cell_mask_kernel` | 灰色加速・NLTE の対象セルの印 | 1 thread/cell | |
| `exp_source_transfer_kernel`・`exp_mg_source_transfer_kernel` | `source_integrator="exp_rosenbrock"` の源の移送 | per element | |
| `zero_void_radiation_exchange_kernel` | VOID セルの交換を 0 に | 1 thread/cell | |
| `copy_outer_snapshot_kernel` | 外側反復の先行実行のための状態の退避・復元 | span × block | 下の「外反復」 |
| `max_reduce_kernel`・`reduce_outer_check_kernel`・`fld_outer_converged_kernel`・`max_reduced_flux_kernel` | 外反復の収束判定の縮約 | shared/atomic | |
| `escaped_energy_kernel<GEOM>`・`volume_source_energy_kernel`・`fv_interior_flux_residual_kernel` | 台帳集計・検査 | 1 thread/group / cell + atomicAdd | |

- **線形解法**: 群バッチ三重対角を `cusparseDgtsv2StridedBatch`（m=n_cells、batchCount=n_groups、batchStride=n_cells）で直接一括求解。
- **多群**: host 群ループなし — 行列 layout は group-major（idx = g·n_cells + c）、cuSPARSE batch が群を畳み込む。
- **外反復**: 不透明度・放出の評価（NLTE は `compute_nlte_coefficients_cuda_with_pe`）→ 組立 → gtsv2 → publish →
  `update_matter_kernel` → 収束判定。host が反復 k の収束判定（2 double の D2H）を待つ間に、状態を退避して反復 k+1 を先に
  積んでおき、反復 k が収束していれば退避した状態へ戻す（先行実行と巻き戻し、`kPipelineSlots` 個のスロット）。
  灰色加速は反復 k の組立の直後に灰色の行列を作り、補正を反復 k+1 の始めに当てる。
- **flux limiter の評価場**: 既定 `Radiation.multigroup_diffusion.limiter_evaluation="predictor"` は最初の外反復の場で limiter を固定する（NUMERICS §6.7）。
- **Fleck**: 生成先 `state.fld_nlte_f_work`（cg layout）、消費は組立の擬似散乱源 `f·η+(1-f)·c·σ_pa·E_old` と物質更新の 2 点。AFI モード（`fld.fleck_mode="afi"`）は両消費点を無効化。
- **散乱**: FLD は物理散乱 \(\kappa_s\) を使わない（`κ_s > 0` と FLD の組み合わせは ConfigError）。
- scratch pool tags: `"fld_1d_gpu:*"` / `"fld_1d:outer_check_pack"` / `"fld1d:outer_pipeline_snapshots"`。恒久場は `state.fld_*`。

#### 6.7.2 FLD 2D RZ（`fld_2d_rz_gpu.cu`、エントリ `advance_radiation_step_fld_2d_rz` :6750）

| kernel 群 | 内容 |
|---|---|
| 物理系 | `build_eta_from_planck_kernel` :1117 / `compute_d_cell_2d_kernel` :1160（**セル中心** limiter D — 1D の face-centered と非対称） / `compute_fleck_for_fld_kernel` :1236（1/(1+z)↔exp(−z) smoothstep ブレンド） / `assemble_fld_2d_csr_kernel` :1348（5-point CSR、1 thread/row、境界 vacuum/reflect/Marshak/state-supply） / `publish_with_projection_kernel` :2493（正値クランプ + 計数 atomic） / `update_matter_kernel` :3479（1 block/cell、32 thread、shared — 1D と同型） |
| CG 系 | 自作 CG: `cusparseSpMV`（Ap）+ **決定論 dot**（`dot_single_block_kernel` :2333 単一 block 固定順 / 2 段 `dot_partials`+`dot_finalize`）+ `cg_update_x_r_kernel` :2401（breakdown を atomicCAS 検出）+ `cg_apply_preconditioner_kernel` :2453 + `cg_update_p_kernel` :2463 |
| 前処理 | Jacobi（diag_inv）/ **z-line**（CSR→三重対角化 :1912 + `cusparseDgtsv2StridedBatch`、batch=n_groups·nr）/ **RGMG**（r 方向 pairwise Galerkin 多重格子 :1963、V-cycle 平滑化は z-line 解、nr は 2 冪必須）/ **AMGX**（`amgx_solver.cpp`、`AMGX_mode_dDDI`、ビルドオプション） |

- **線形系**: 全群を 1 本の **group-major ブロック対角 5-point CSR**（n_rows = n_cells·n_groups、群間結合なし）にまとめ CG を 1 回呼ぶ。`linear_solver_2d` で amgx_cg / cusparse_cg_zline / cusparse_cg_rgmg / 既定 Jacobi-CG を選択 (:7070)。
- **D2H 同期点**: CG 内残差チェック（初期 4 回 + 4 回毎 + 最終）+ 外反復末の max\|ΔT\|。
- 作業領域は恒久 `Fld2DWorkspace`（`DeviceArray` 群、resize 再利用；cuSPARSE handle/descriptor キャッシュ）。

### 6.8 現行 S_N カーネル群（`mode="sn_transport"` — 決定論、1D/2D_RZ）【CURRENT】

数理は NUMERICS §6.8/§8。実装は `sn_transport_1d_gpu.cu`（1D 自己完結）、
`sn_transport_2d_gpu.cu`（2D ドライバ）+ `sn_transport_gpu.cu`（2D sweep エンジン）、
`sn_dsa_1d_gpu.cu`、`sn_material_newton_gpu.cu`、`sn_cyl_quadrature_1d.cpp`（host 求積）。
（`sn_transport_gpu.cu` の 1D 経路は HOLO/QD-LO の \(S_N\) 閉包で、CPU 参照実装 `sn_transport_1d.cpp` とともに
2026-09-29 に退役した — `retired/radiation_monte_carlo/`。）

#### 6.8.1 S_N 1D（エントリ `advance_radiation_step_sn_1d`）

**空間スキーム**：1D の既定は線形不連続（LD、`Radiation.sn_transport.spatial_scheme="linear_discontinuous"`、
2026-09-25 以降、`sn_ld_1d_gpu.cu`、NUMERICS §6.8.4）。`"linear_characteristic"`（2D_RZ の既定）と `"diamond"` は
`sn_transport_1d_gpu.cu` の旧経路。行番号は書かない（`grep -n __global__`）。

**LD（既定）**（`sn_ld_1d_gpu.cu`、`advance_step`）：

| kernel | 目的 | thread mapping |
|---|---|---|
| `compute_cells_kernel` | セルの幾何（h、集中質量 \(M_{L,R}\)、角度再配分の重み \(N_{L,R}\)、面積） | 1 thread/cell |
| `scan_sweep_kernel<kOneCell>` | 全群のスイープ（既定、2026-10-02）。出発方向・内向き・外向きの順序方向を順に、各順序方向のセル方向の漸化式をスキャンで解く：セルの流出の値は流入の値のアフィン写像（係数はセルの演算量と前の順序方向が残した角度の端の値から）で、各スレッドが自分のセル（1 セル版のブロック（レジスタで決まり、CUDA 12.6 の sm_89 のビルドでは 512 スレッド）に収まるまで 1 セル、それを超えると連続する k セル）の写像を合成し、ブロックがスレッドの写像をスイープの順に走査し、各スレッドは自分の最初のセルの流入から逐次版と同じ式でセルを解く。逐次版とは丸めで異なる（NUMERICS §6.8.4） | **1 block=1 群**、`⌈n/32⌉×32` threads（`kOneCell`、1 セル版の上限まで）、それを超えると複数セル版の上限の threads（`maxThreadsPerBlock` を実行時に問い合わせる） |
| `sweep_kernel` | 逐次版のスイープ（`TENRYU_SN_LD_SEQUENTIAL_SWEEP=1`）。セルごとの 2×2 の LD 行列、曲がった形状は重み付きダイヤモンドの角度閉包と出発方向、半分の順序方向をセル方向の対角の波面で | 1 block=1 群、1 warp（`<<<n_groups, 32, shared>>>`） |
| `sweep_matrix_kernel` | 同じ断面積で何度もスイープするときの逆行列の前計算 | per element |
| `moments_kernel` | 節点のスカラー流束、面の流れ、\(P_{rr}\) | per element |
| `p1_assemble_kernel`・`p1_factor_kernel`・`p1_apply_kernel` | スイープと整合する P1 低次系（ブロック三重対角）：散乱の拡散合成加速と放出結合の灰色前処理 | per system |
| `linearize_kernel`・`matter_setup_kernel`・`matter_update_kernel`・`cell_temperature_kernel` | 節点の電子温度のまわりの放出の線形化と、輸送自身の吸収・放出からの保存的な物質更新 | 1 warp/節点（群ごとの項は群のレーンで、群についての和は群の順に全レーンで。`matter_update_kernel` の Planck 分率とその温度微分は、群の境界の累積側・裾側の値を境界ごとに 1 回評価して群の差に組み合わせる（2026-10-02、同じ値）） |
| `dots_kernel`・`orth_update_kernel`・`residual_kernel`・`solution_update_kernel` ほか | 放出結合の GMRES（節点の吸収率密度 \(\sum_g\sigma_{a,g}\phi_g\) の上） | per element |
| `outer_current_kernel`・`history_kernel` | 外面の流束、角度の履歴の次ステップへの持ち越し | per element |

- **反復**：温度の Newton の各反復で、放出結合を GMRES で解く（作用素の 1 回の適用が全群のスイープ、散乱は source iteration と
  P1 加速で収束させる）。灰色前処理は `grey_preconditioner="auto"` で局所の利得が閾値を超える Newton 反復だけで使う。
  収束しない場合は `sn_material_retry_flag` のビットでステップの棄却を要求する。

**LC・diamond（旧経路）**（`sn_transport_1d_gpu.cu`）：

| kernel | 目的 | thread mapping |
|---|---|---|
| `build_sweep_inputs_kernel` | σ_t と IMEX 等方源（0.5η + 0.5σ_s·φ_old + 時間項） | 1 thread/(cell×group) |
| `sn_sweep_spherical_serial_kernel` | 球面 diamond sweep（`spatial_scheme="diamond"`） | 1 block=1 群、thread0 のみ（全角度×全セルを逐次） |
| `sn_sweep_spherical_lc_kernel` | 球面 linear-characteristic | block=群、32 lane の warp が (cell,angle) の対角波面 |
| `sn_sweep_cylindrical_{serial,lc}_kernel` | 円筒 product-quadrature 版 | 同構造 |
| `precompute_lc_weights_kernel` | LC 指数閉包 θ(τ)=(1−A)/(τA)∈[0.5,1] | 1 thread/(cell×group×angle) |
| `sn_moments_reduction_k2_kernel`・`sn_face_flux_reduction_k2_kernel` | φ/F/Prr、面の流束の還元 | 1 thread/(cell or face ×group) |
| `phi_to_rad_E_kernel`・`sn_chi_kernel`・`apply_face_flux_boundary_kernel` | 後処理（外面の流束は掃引の離散的な流出の集計から入射 \(S^-\psi_\mathrm{in}\) を引く） | per element |
| スイープ後の連鎖 | E* flux → Fick 拡散面流束 → AP ブレンド（τ_lo=10/τ_hi=20）→ donor/θ リミッタ | per (cell/face×group) |

- **DSA 1D**（`sn_dsa_1d_gpu.cu`、全形状）：スイープの空間閉包と整合する P1 の面モーメント \((\Phi_f, J_f)\) の系
  （未知数の並びで五重対角）を `assemble_sn_dsa_consistent_kernel` で組み、**`cusparseDgpsvInterleavedBatch`** で解き、
  `apply_sn_dsa_consistent_correction_kernel` が補正を加える（2026-09-24。以前のセル中心の調和平均の拡散作用素は、平均自由行程
  約 3 を超える厚いセルで反復が発散した）。内側の反射面で \(J=0\)、外面は入射補正なし。
- **求積**: 球面/平面 = GL(n_angles) + Carlson α 漸化 + Morel&Montry weighted-diamond τ; 円筒 = `build_sn_cyl_quadrature_1d`
  （n_angles=2L²、level-major、per-level Carlson α、M&M A1-A4、開始方向 sd_μ=sinθ_l）。
- **物質結合 Newton**（旧経路、`sn_material_newton_gpu.cu`、1D/2D 共通）: 1 block=1 cell、128 thread が群を stride 分担、
  energy-variable bracketed Newton。床/ブラケット拡張は `atomicOr(retry_flag)` で global step 棄却を要求。
- **Marshak/Tr(t) 駆動**: host が外面の入射強度を作ってアップロードする。黒体（温度）駆動は \(\psi_\mathrm{in}=2F_\mathrm{inc}\)
  （\(F_\mathrm{inc}=\tfrac14 c a T_r^4 b_g\)）、LD の流束駆動は \(\psi_\mathrm{in}=F_\mathrm{inc}/S^-\)
  （\(S^-=\sum_{\mu<0}w|\mu|\)、離散の入射流束が指定値になる）。旧経路の流束駆動は \(2F_\mathrm{inc}\) のまま。
- **体積線源**：LC・diamond の経路では外部体積線源は群 0 にだけ入る。

#### 6.8.2 S_N 2D RZ（エントリ `advance_radiation_step_sn_2d_rz` :1443 → `solve_sn_transport_2d_rz_gpu`）

- **スイープ並列化**: `sn_sweep_2d_kernel`（sn_transport_gpu.cu :1193）は **grid=dim3(n_polar, n_groups)**（block=(polar level, 群)）、**方位半角 m は host 逐次ループ** (:2420)、block 内は **KBA 反対角波面**（stage=0..nr+nz−2、threads が反対角セルを分担、threads=clamp(128,[32,256])）。LC 版 `sn_sweep_2d_rz_lc_kernel` :1459 同構造。
- 還元: `sn_reduce_cell_outputs_2d_kernel` :1695 / `sn_reduce_face_flux_2d_kernel` :1731。
- **DSA 2D**: `sn_dsa_{setup,jacobi,apply}_2d_kernel`（:1759/:1777/:1920）— 5 点 RZ 拡散ステンシルを**固定 50 回の点 Jacobi 反復**で解く（1D の cuSPARSE 直接解と実装が根本的に異なる）。
- **Newton は coupling solve の外**でドライバが呼ぶ（1D は outer 内）— `update_material=false` で sweep し、スイープ後連鎖（拡散/AP/リミッタ/E*、各段後に `zero_axis_faces_checked_kernel` :308 で r=0 軸面ゼロ）→ Newton :1712。
- 境界: 真空/反射/Marshak（`kSNBoundary*`）、軸対称は軸面フラックスゼロ化。

#### 6.8.3 設計上の注意（host オーバーヘッド削減/perf 文脈）

1. **1D の sweep の並列度は群数 × 1 warp**（LD の既定と LC は 1 warp、diamond は 1 群=1 thread の逐次）— 1D S_N の GPU 占有率は原理的に低く、host 律速と併せて 1D が local-GPU tier に留まる一因。
2. **2D は (polar×群) block × KBA 波面**で並列度が立つ（方位 m と波面 stage は逐次）。
3. **DSA 実装の 1D/2D 非対称**（1D はスイープと整合する五重対角の直接解、2D は 5 点拡散の 50 回 Jacobi）と **FLD の D 評価の 1D/2D 非対称**（face-centered vs cell-centered）は将来の統一候補として明示しておく。

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

## 9. カーネル起動シーケンス（1タイムステップ）

> **【状態注記 2026-07-10、2026-09-29 更新】** 本節は設計時の起動列で、実装の正は `src/coupling/driver.cpp` の
> ステップと各演算子のソース（カーネル名は §1.0）。とくに次の点が実装と違う：
> - Radiation phase（Phase 4）は現行の入口だけを書いた。以前ここにあったモンテカルロ輻射の起動列（R2/R6/R7/R7b/R8/R9/R12、
>   粒子 MPI P5/P6、ダブルバッファ遷移）は 2026-09-29 に `retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md` へ移した。
>   現行 FLD/S_N の起動列は §6.7/§6.8 と各ソルバ（`fld_*_gpu.cu`・`sn_*_gpu.cu`）。Phase 0・3-post・5・6 に残る
>   `rad_E_tally`・`E_escape[G]`・`rad_mom_dep`・`mmatrix_fix_count`、R1・U9 のための EOS 再クロージャ、U7（粒子のセル再同定と
>   hash grid）、`E_census`、`dt_rad` はモンテカルロ輻射の設計で、実装に無い
> - Laser phase：L1 `laser_mesh_map`・L2 `compute_density_gradient`・L5 `deposit_lm_to_hydro`・`laser_cache_update` は無い。
>   1D は `map_from_hydro_1d`（§5.1 の写像カーネル）→ 光線追跡（既定は特性線法、§5.2）が流体セルへ直接沈着 →
>   host の再配分（`apply_deposit_redistribution_1d`）→ host の省略キャッシュ更新（`RaytraceSkipCache::update_cache`）
> - H1 `hydro_active_update` は host のループ、U2 `floor_clamp`・U5 `nan_check`・U8 `compute_zbar` は存在しない
>   （床は各演算子の中、非有限値の検査は `driver_safety_audit.cu` の段の前後の集約、\(\bar Z\) は §7.6 の注記）
> - fixed の \(\bar Z\) はステップごとに更新しない（下の U8 の「fixed + n_mat>1: 毎ステップ実行」は設計時の記述）
> - H13/H14 `eos_forward`/`eos_inverse` は無く、表 EOS は各カーネルが §2.4x の device 関数を直接呼ぶ
>
> 単一 compute_stream の方針と phase の順序（流体 → 伝導 → レーザー → 輻射 → …）は現行。

1つの Strang splitting ステップ `t^n → t^{n+1}` における全カーネルの実行順序。
`[SYNC]` はストリーム同期、`[MPI]` はMPI通信を示す。

**ストリーム方針（v1.0）**: 全カーネルは単一の `compute_stream` 上で逐次起動する。
CUDA ストリームのFIFO保証により、同一ストリーム内のカーネル間 RAW/WAR/WAW 依存は
暗黙に満たされる。`[SYNC]` は D2H 転送やホスト判定が必要な箇所にのみ挿入する。
**将来 multi-stream 化する場合**、以下の重要 RAW 依存に `cudaEventRecord/WaitEvent` が必要：
- L5 `deposit_lm_to_hydro` → U1 `source_injection`（laser_dep）

```
═══════════════════════════════════════════════════════
 Phase 0: ステップ前処理
═══════════════════════════════════════════════════════
  H1:  hydro_active_update
  U8:  compute_zbar                    // Z̄/A_eff 更新。実行条件:
                                       //   thomas_fermi / tabular: 毎ステップ実行（Z̄ が ρ, Te に依存）
                                       //   fixed + n_mat>1: （設計時）毎ステップ実行。実装は初期化時だけで、ステップごとには更新しない
                                       //   fixed + n_mat==1: Phase 0 ではスキップ（定数、ARCHITECTURE §8 Step 9b で初期化済み）
                                       // Conduction(C1), Laser(L1), Radiation(R1) が Z̄ を参照するため
                                       // ステップ冒頭で最新化する（NUMERICS §1.1.4, ARCHITECTURE §5.2）
  if (step > 0):                      // step=0 では前ステップが存在しないためスキップ
    U5:  nan_check (前ステップのエラーチェック)
    // 検査対象フィールド（ホワイトリスト）: rho, Te, Ti, ee, ei, Pe, Pi, v_r, v_z, vol
    // アキュムレータ配列（rad_dep, rad_E_tally等）は毎ステップゼロ初期化されるため検査対象外
  // error_flags の D2H とチェックは **無条件**（step=0 でも U8 がフラグを立てうるため）
  [SYNC] → [D2H] error_flags        // DeviceErrorFlags をホストに転送
  Host: エラーフラグ確認、DeviceErrorFlags クリア（cudaMemsetAsync）
  cudaMemsetAsync: E_floor_injected=0, E_safety=0, clamp_count=0, opacity_clamp_count=0, E_numerical_loss=0, step_E_solver=0,
    E_escape[G]=0, rad_mom_dep[n_cells×dim]=0, mmatrix_fix_count=0
    // 注: step_E_pdV_bdry / step_E_Marshak_in はホスト側スカラー（下記 Host: ブロックで初期化）。デバイスメモリではない
    // ステップ先頭で**無条件**ゼロ初期化（Phase 4 がスキップされても Phase 6 U3/D2H が参照するため）。
    // E_floor_injected 等は Phase 1-5 の U2/U1/R8 が累積。opacity_clamp_count: NUMERICS §11.3 準拠。
    // E_numerical_loss: Phase 3 U1 の退化セル損失 + Phase 4 R8 MAX_EVENTS 損失を全て捕捉するためステップ先頭で1回。
    // step_E_solver: Hypre 有効時のみ非ゼロになるが、常にゼロ初期化（STS パスで残留値が累積するのを防止）。
    // E_escape/rad_mom_dep: radiation.enabled=False 時に Phase 4 がスキップされても Phase 6 でゼロ値が保証される
  cudaMemsetAsync: rad_dep[n_cells×G]=0  // **毎ステップ**ゼロ初期化（Phase 3 U1 が rad_dep=0 を前提、NUMERICS §2.1 準拠）。
    // Phase 4 冒頭でも rad_dep をゼロ初期化するが、Phase 4 の R8/R9 が atomicAdd した後
    // Phase 4 U1 が読んだ後もゼロ化されないため、Phase 3 U1 前に再度ゼロ保証が必要
  // ホスト側 per-step スカラーのゼロ初期化（Phase 6 Allreduce で参照されるため、
  // 当該 Phase がスキップされても stale 値が累積しないよう毎ステップ無条件で初期化する）:
  Host: step_laser_dep_total = 0.0   // Phase 3 スキップ時（laser.enabled=false）に stale 防止
  Host: step_laser_escaped   = 0.0   // Phase 6 で E_incident - step_laser_dep_total から算出。laser.enabled=false 時はゼロ保証
  Host: step_E_pdV_bdry      = 0.0   // Phase 1/5 ホスト計算結果（スキップ時のゼロ保証）
  Host: step_E_Marshak_in    = 0.0   // Phase 4 ホスト計算結果（radiation.enabled=false 時のゼロ保証）
  // レーザーキャッシュ初期化:
  // step=0 またはリスタート直後は laser_cache_valid=false（State.laser_cache_valid で管理）。
  // Phase 3 で laser_cache_valid==false の場合、L6 をバイパスし強制的に full raytrace を実行する。
  // laser_cache_update 完了後に laser_cache_valid=true を設定する。
  // **実装必須**: Phase 3 の L6 呼び出し前に `if (!laser_cache_valid) goto full_raytrace;` ガードを挿入すること。

═══════════════════════════════════════════════════════
 Phase 1: Hydro H(Δt/2) — Predictor-Corrector
═══════════════════════════════════════════════════════
  [MPI] halo_exchange(rho, Te, Ti, Pe, Pi, Q)  // double scalar stride=1（NUMERICS §12.2.2）
  [MPI] halo_exchange_int8(hydro_active)       // int8_t stride=1（exchange_int8_fields）
  // --- Predictor 前退避（§2.4b 実装注意 参照）---
  [D2D] v_r_save ← v_r, v_z_save ← v_z       // v^n 退避
  [D2D] x_r_save ← x_r, x_z_save ← x_z       // r^n 退避
  [D2D] Pe_save ← Pe, Pi_save ← Pi             // P^n 退避（Corrector P^{n+1/2} 時間中心化に使用）
  [D2D] vol_save ← vol                         // V^n 退避（H11/H12 vol_old に使用）
  --- Predictor ---
  if (geometry == GEOM_2D_RZ):
    H3:  compute_area_vectors        // 2D_RZ専用。1D_SPHではH4/H9が球面幾何を直接計算するためスキップ（§2.1c）
  H2:  compute_node_mass
  H4:  compute_corner_force         ← P^n, Q^n
  H5:  velocity_update(Δt/4)        // v → v^{n+1/4}
  H6:  position_update(Δt/4)        // r → r^{pred}
  H7:  compute_cell_geometry
  H8:  compute_density
  H9:  compute_divergence
  H10: compute_artificial_viscosity
  H13: eos_forward                  ← ρ^{pred}, T → P^{pred}（Pe, Pi を上書き）
  H15: compute_sound_speed
  --- Corrector ---
  if (geometry == GEOM_2D_RZ):
    H3:  compute_area_vectors        // r^{pred} を使用。1D_SPHではスキップ（§2.1c）
  H4:  compute_corner_force         ← P^{pred}, Q^{pred}
  // --- 位置更新を速度更新より先に実行（Leapfrog: v^{n+1/4} 保護）---
  [D2D] x_r ← x_r_save, x_z ← x_z_save       // r^n 復元
  H6:  position_update(Δt/2)       ← r^n + (Δt/2)·v^{n+1/4}（v^{n+1/4} はまだ v バッファに存在）
  [D2D] v_r ← v_r_save, v_z ← v_z_save         // v^n 復元（v^{n+1/4} を上書き）
  H5:  velocity_update(Δt/2)       ← v^n + (Δt/2)·a^{pred} → v^{n+1/2}
  // --- 幾何・密度・発散 ---
  H7:  compute_cell_geometry        // r^{n+1/2} → V^{n+1/2}
  H8:  compute_density
  H9:  compute_divergence
  // --- エネルギー更新 ---
  [Kernel: Pe = (Pe_save + Pe)/2, Pi = (Pi_save + Pi)/2]   // P^{n+1/2} 時間中心化（NUMERICS §3.2.12）
  // Q^{n+1/2} = Q^{pred}（Predictor divergence で評価済み、算術平均ではない。NUMERICS §3.2.12 注）
  U6:  qei_exchange                 ← T_e, T_i → Q_ei
  H11: energy_update_ion(vol_old=vol_save, vol_new=vol)     ← P_i^{n+1/2}, Q^{n+1/2}, V^n, V^{n+1/2}
  H12: energy_update_electron(vol_old=vol_save, vol_new=vol) ← P_e^{n+1/2}, V^n, V^{n+1/2}
  // --- E_pdV_bdry 計算（Phase 1 寄与、NUMERICS §10.2）---
  // [Host: step_E_pdV_bdry += Σ_{f∈∂Ω} P_f^{n+1/2} × A_f × v_{n,f} × (Δt/2)]
  // P_f = 境界セルの (Pe+Pi+Q)、v_{n,f} = 境界ノードの法線速度、A_f = 境界面面積
  H14: eos_inverse(species=0)       ← ee → Te
  H14: eos_inverse(species=1)       ← ei → Ti
  H13: eos_forward                  ← T → P, Cv
  H15: compute_sound_speed
  H16: apply_hydro_bc
  U2:  floor_clamp(rho, Te, Ti)
  // **U2後のPe/Cv stale window 注記**（設計判断）:
  // U2 が Te をフロアにクランプした場合、直前の H13 で算出した Pe/Cv_e は stale になる。
  // Phase 2 C1 が Cv_e を参照するが、フロアクランプは T_floor=1e-3 eV のみ適用されるため、
  // 影響セルの κ_SH ∝ Te^{5/2} ≈ 3e-8 は極小。D_eff 誤差の物理的影響は無視可能。
  // 必要なら C1 前に H13(subset) を挿入可能だが、v1.0 では省略する。
  [MPI] halo_exchange(x_r, x_z, v_r, v_z)
  // 注: ALE rezone/remap は Phase 5（2回目の H(Δt/2) 後）にのみ実行する
  //     （NUMERICS §3.3, ARCHITECTURE §4.4）。Phase 1 では ALE を行わない。

═══════════════════════════════════════════════════════
 Phase 2: Conduction C(Δt)    // conduction.enabled=False の場合は Phase 2 全体をスキップ（dt_cond=∞）
═══════════════════════════════════════════════════════
  if (!conduction.enabled): goto Phase 3
  [MPI] halo_exchange(Te)
  C1:  compute_spitzer_deff          // D_eff 凍結（§4.2.1: STS 開始時に1回のみ）
  if (2D_RZ):
    cudaMemsetAsync(mmatrix_fix_count, 0, sizeof(int), compute_stream)  // C2 が atomicAdd で使用
    C2: kershaw_stencil_build        // Kershaw 係数凍結
    if (solver == "hypre"):          // §4.5 Hypre パス（2D_RZ専用、オプション）
      Hypre: 行列構築（C2 出力 + C_v/Δt 対角、§4.5 step 2）→ Te_old 退避（step 3）→ PCG+AMG solve（step 4）→ Te 直接更新+H13（step 5）→ E_solver（step 6）
      // C3 は起動しない。Hypre 内部が AMG V-cycle + PCG 反復を実行
      // E_solver 計算（§4.5 step 6）: CUB Reduce で Σ ρc_v(T^{n+1}-T^n)V - Δt Σ (∇·q)V を算出
      // Te^n は step 3 で退避した Te_old から取得。C_v = ρc_v [erg/(cm³·eV)]
      // → step_E_solver をPhase 6 D2H に含めて State.E_solver に累積（NUMERICS §4.2.3）
    else:                            // STS パス（既定）
      for j = 1 to s:               // STS ステージ（s = O(√N_naive)、NUMERICS §4.2.1）
        [MPI] halo_exchange(Te)      // ← ghost cell の Te を更新（ステージ毎に必要）
        C3: kershaw_apply(tau[j]) → Te更新 → ダブルバッファswap
  else (1D_SPH):
    // 注: Hypre は2D_RZ専用。1D_SPH + solver="hypre" → STS フォールバック + WARNING
    for j = 1 to s:                  // STS ステージ
      [MPI] halo_exchange(Te)        // ← ghost cell の Te を更新
      C4: conduction_1d_tridiag(tau[j]) → Te更新 → ダブルバッファswap
  U2:  floor_clamp(rho, Te, Ti)
  H13: eos_forward(Te → ee, Pe, Cv)       // Post-conduction EOS sync（必須：NUMERICS §4.2.1 step 4）
                                            // STS/Hypre は Te のみ直接更新、ee は未更新 → H13 で再同期
                                            // これがないと Phase 3 U1 が古い ee に沈着を加算し、
                                            // 後続 H14 が古い ee から旧 Te を復元して伝導更新が無効化される
  [MPI] halo_exchange(Te)

═══════════════════════════════════════════════════════
 Phase 3: Laser L(Δt)         // laser.enabled=False の場合は Phase 3 全体をスキップ（LaserMesh 未確保、SPECIFICATION §6.4.6）
═══════════════════════════════════════════════════════
  if (!laser.enabled): goto Phase 3-post  // Phase 3 のスキップ（laser_dep=0 が保証されるため U1 は空振り）
  // **前提条件**: laser.enabled=True ならば n_LM_cells > 0 かつ n_LM_nodes > 0（Builder §6.4.6 が検証）。
  // この前提が崩れると L6 CUB Max が空配列で未定義値を返す。防御的に assert(n_LM_cells > 0) を推奨
  if (laser.mode == "radial_absorption_1d"):
    invalidate raytrace skip cache
    skip = false                       // radial mode は毎回 full 1D integral
  else:
    L6:  ray_skip_check(ρ, Te, Z̄ vs cached) → CUB Max reduction
                                          // L6 は HydroMesh 上の ρ, Te, Z̄ を読む（LaserMesh ではない）。
                                          // L1 (laser_mesh_map) はまだ起動前のため LaserMesh は stale。
                                          // キャッシュ値も HydroMesh ρ, Te, Z̄（下記 cache update 参照）。
    [SYNC] → [D2H] δ_max                // CUB 出力をホストに転送
    [MPI] MPI_Allreduce(MAX, δ_max)     // 全rankでスキップ判定を統一（NUMERICS §12.2.2 行5）
                                         // ローカル判定のみだと集団通信(laser_mesh_sync)でデッドロック
    [Host: δ_max < raytrace_skip? → skip]
  if (not skip):
    L1: laser_mesh_map(HydroMesh → LaserMesh)
    [SYNC] → [D2H] LaserMesh fields     // L1出力をホストに転送（MPI通信の前提）
    [MPI] laser_mesh_sync
      if (1D_SPH): MPI_Allgatherv(ρ, T_e, n_e profile)
      else:        MPI_Allreduce(SUM) (ρ, T_e, n_e, Z* on LaserMesh)
    [H2D] LaserMesh fields             // MPI 結果をデバイスに転送（L2 の入力前提）
                                        // GPU-aware MPI 使用時はデバイスバッファ上で直接 allreduce し、
                                        // D2H/H2D を省略可能。その場合は L1 後の [D2H] も不要
    L2: compute_density_gradient
    if (1D_SPH && laser.mode == "radial_absorption_1d"):
      cudaMemsetAsync: deposit_power_cell[n_cells]=0, P_unabsorbed=0
      if (rank == 0):
        L7: radial_absorption_1d_kernel(P_total, radial arrays → deposit_power_cell)
      [MPI] Allgatherv deposit_power_cell  // 既存 1D deposit 集約経路
      apply existing 1D deposit path (deposit_power_cell × Δt → laser_dep)
      // L5 と laser_cache_update は起動しない
    else:
      cudaMemsetAsync: laser_dep[n_cells]=0, P_unabsorbed=0  // 全グループの累積先をゼロ初期化
      // --- ビームグループループ（NUMERICS §5.4）---
      // 同一パラメータ（F値・プロファイル・極角）のビームをグループ化
      // GXII等の全ビーム同一パラメータ構成では n_groups=1、1回のレイトレースで完結
      for g in range(n_groups):
        cudaMemsetAsync: deposit[N_LM_nodes]=0  // グループ毎にゼロ初期化
        cudaMemsetAsync: laser_dep_g[n_cells]=0 // L5(1D_SPH)がatomicAddするためゼロ初期化必須
        if (1D_SPH):
          L3: ray_trace_2d(beam_group_g_params, P_g)
        else:
          L4: ray_trace_3d(beam_group_g_params, P_g)
        L5: deposit_lm_to_hydro(deposit → laser_dep_g)  // グループ g の沈着を一時配列へ
        // 明示的累積: laser_dep[c] += laser_dep_g[c]（block=256, grid=(n_cells+255)/256）
        // laser_cache_update カーネル内でフュージング可能（f̂ 計算と同時に +=）
        laser_cache_update(ρ, Te, Z̄, laser_dep_g, P_g_dt[g], group_idx=g → cached, laser_dep += laser_dep_g)
      // end beam-group loop
    // レーザー診断 D2H（step_laser_dep_total 累積用）:
    //   CUB Sum(laser_dep[n_cells]) → step_laser_dep_total [erg]（ローカルセル合計）
    //   [SYNC] → [D2H] step_laser_dep_total
    //   注: step_laser_escaped は Phase 6 Allreduce 後にホストで算出する
    //   （replicated strategy では P_unabsorbed が既にグローバル値のため、
    //    Phase 3 時点で step_laser_escaped を計算すると Allreduce(SUM) で N_ranks 倍になる。
    //    Phase 6 で step_laser_dep_total がグローバル化された後に
    //    step_laser_escaped = E_incident - step_laser_dep_total_global とする）
    U1: source_injection(laser_dep → ee, rad_dep=0)
  else:
    // skip path: per-group キャッシュ済み f̂_g から laser_dep を再構成（NUMERICS §5.9.3）
    // laser_dep[c] = Σ_g f̂_g[g*n_cells+c] × P_g(t_now) × Δt [erg]
    // P_g_dt[n_groups] 配列はホストで計算し D2H 転送
    cudaMemsetAsync: laser_dep[n_cells]=0
    reconstruct_laser_dep(laser_dep_frac[n_groups*n_cells], P_g_dt[n_groups] → laser_dep)
    // skip 分岐のレーザー診断 D2H:
    //   CUB Sum(laser_dep[n_cells]) → step_laser_dep_total [erg]（ローカルセル合計）
    //   [SYNC] → [D2H] step_laser_dep_total
    //   注: step_laser_escaped は Phase 6 で算出（full raytrace と同一プロトコル）
    U1: source_injection (laser_dep → ee, rad_dep=0)
  H14: eos_inverse(species=0, ee → Te)  // re-closure: R1(fleck_factor) が最新Teを必要とする
  H13: eos_forward(Te → Pe, Cv)         // R1 が最新 Cv_e を必要とする（Phase 3 での ee 変更を反映）
  U2:  floor_clamp(rho, Te, Ti)
  // **U2後のPe/Cv stale window 注記**: Phase 4 は U9 compute_opacities から開始し、
  // σ = ρ×κ(ρ, Te) で Te を直接使用する（Cv_e/Pe 不使用）。R1 fleck_factor は Cv_e を使用するが、
  // フロアクランプ影響セルの Cv_e 誤差は f_fleck ≈ 1/(1+β) の β に入り、β ∝ T³/Cv → 極小域で自動抑制。

  // --- T_max_n キャプチャ（NUMERICS §11.8: overshoot 基準温度の保持）---
  CUB DeviceReduce::Max(Te) → Te_max_device
  [SYNC] → [D2H] Te_max_device
  [Host: T_max_n = max(Te_max_device, T_boundary)]  // Phase 4 overshoot 検出の基準値

═══════════════════════════════════════════════════════
 Phase 4: Radiation R(Δt)
═══════════════════════════════════════════════════════
  if (!radiation.enabled): skip
  RadiationStep::step → advance_radiation_step_{fld,sn}_{1d,2d_rz}   // §6.7 / §6.8。カーネル列は各ソルバ。
                                       // 物質の更新（Te, ee, Pe）と rad_dep / rad_emit の publish を含む
                                       // （Radiation.imc.two_stage では半ステップずつ 2 回、間で EOS を閉じ直す）
  overshoot_metrics_kernel             // driver_safety_audit.cu。最大値原理の超過 → history radiation/overshoot_*
  // 2026-09-29 まで本フェーズに書いていたモンテカルロ輻射の起動列（R2〜R13、粒子の MPI P5/P6、SoA ダブルバッファ遷移）は
  // retired/radiation_monte_carlo/docs/CUDA_KERNELS_monte_carlo.md へ移した。

═══════════════════════════════════════════════════════
 Phase 5: Hydro H(Δt/2) — Phase 1と同一
═══════════════════════════════════════════════════════
  // Phase 1 と同一の Hydro H(Δt/2) シーケンスを実行（Predictor-Corrector 全体）
  // **注**: H3 は Phase 1 と同様に `if (geometry == GEOM_2D_RZ)` ガード付き（§2.1c）。1D_SPH ではスキップ。
  // **退避・復元**: Phase 1 と同一（v^n, r^n, P^n, V^n を Predictor 前に退避、Corrector で復元）
  [MPI] halo_exchange(rho, Te, Ti, Pe, Pi, Q)  // double scalar stride=1（NUMERICS §12.2.2）
  [MPI] halo_exchange_int8(hydro_active)       // int8_t stride=1
  [D2D] 退避: v_save, x_save, Pe_save, Pi_save, vol_save
  --- Predictor ---
  [H3]→H2→H4→H5(Δt/4)→H6(Δt/4)→H7→H8→H9→H10→H13→H15
  --- Corrector ---
  [H3]→H4→[restore x]→H6(Δt/2)→[restore v]→H5(Δt/2)→H7→H8→H9→[P centering]→U6→H11(vol_save)→H12(vol_save)→H14→H13→H15→H16→U2
  // E_pdV_bdry 計算（Phase 5 寄与）: Phase 1 と同一（NUMERICS §10.2）
  // [Host: step_E_pdV_bdry += Σ_{f∈∂Ω} P_f^{n+1/2} × A_f × v_{n,f} × (Δt/2)]
  [MPI] halo_exchange(x_r, x_z, v_r, v_z)

  --- ALE (条件付き — 2D_RZ かつ ALE有効の場合のみ。2回目の H(Δt/2) 後にのみ実行。NUMERICS §3.3, ARCHITECTURE §4.4) ---
  if (cfg.main.geometry != "2D_RZ" || cfg.mesh.motion != "ale" || !cfg.mesh.rezoning.enabled): goto Phase 6
  A1:  mesh_quality_check → CUB Min reduction
  [SYNC] → [D2H] q_min                // CUB 出力をホストに転送
  [MPI] MPI_Allreduce(MIN, q_min)     // **必須**: 全rankでrezone判定を統一（ローカル判定ではhalo_exchangeデッドロック）
  [Host: q_min < threshold?]
  if (rezone needed):
    // **pre-rezone スナップショット（必須）**: A3 conservative_remap が x_r_old/x_z_old（Lagrangian メッシュ）を
    // 入力として必要とするため、Winslow 反復ループ**開始前**に退避する。ダブルバッファ swap chain で
    // 2反復目以降に元の座標が上書きされるため、明示的な退避がないと A3 に渡す old 座標が不正になる。
    [Host/Device: x_r_old ← x_r, x_z_old ← x_z]  // cudaMemcpy D2D（Scratchバッファ使用可、ノード配列のため小容量）
    // Δl_min グローバル化（収束判定用）:
    CUB DeviceReduce::Min(char_length[n_cells]) → Δl_min_local
    [SYNC] → [D2H] Δl_min_local
    [MPI] MPI_Allreduce(MIN, Δl_min_local) → Δl_min_global
    for iter = 1 to max_iterations:  // 既定20、convergence_tol達成で早期終了。NUMERICS §3.3.3 準拠
      [MPI] halo_exchange(x_r, x_z)  // 各Jacobi反復前にゴーストノード座標を交換
      A2: winslow_jacobi_step        // → x_r_new, x_z_new（ダブルバッファ書き込み）
      CUB DeviceReduce::Max(displacement) → δ_rezone  // ノード変位最大値
      [SYNC] → [D2H] δ_rezone       // ホスト側で収束判定
      [MPI] MPI_Allreduce(MAX, δ_rezone)  // **必須**: 全rankで収束判定を統一（ローカルbreakではhalo_exchangeデッドロック）
      swap(x_r, x_r_new); swap(x_z, x_z_new)  // ホスト側ポインタ交換
      [Host: δ_rezone < convergence_tol × Δl_min_global? → break]  // 全rank同一条件でbreak（Δl_min_global はループ前で算出済み）
    // **最終ゴースト座標同期（必須）**: Winslow 反復ループ内の halo_exchange は各反復の
    // 開始時に実行されるため、最終反復のスワップ後のゴーストノード座標は 1 反復前の値。
    // remap (A3) がゴーストセルの面座標を使用するため、最終座標を交換しないと
    // ドメイン境界でのフラックス計算が O(convergence_tol) だけ不整合になる。
    [MPI] halo_exchange(x_r, x_z)
    // **vol_new/座標計算（必須）**: A3 の入力 vol_old/vol_new, x_r_old/x_z_old/x_r/x_z が必要。
    // Winslow 反復で x_r/x_z が更新された後、remap 前に新メッシュの体積を再計算する。
    // vol_old は Winslow 前の値を保持しておく（ホスト側で vol_old = vol をコピー）。
    // x_r_old/x_z_old は上記の pre-rezone スナップショットで退避済み。
    [Host: vol_old ← vol]  // rezone 前の体積を退避
    H7:  compute_cell_geometry  // x_r, x_z (rezoned) → vol (= vol_new), face_area, char_length
    // 方向分離 remap（NUMERICS §3.3.4）: Strang-type 交替スイープ
    // 偶数ステップ: A3(r-sweep) → A3(z-sweep)、奇数ステップ: A3(z-sweep) → A3(r-sweep)
    // A3 の座標引数: x_r_old/x_z_old = pre-rezone スナップショット、x_r/x_z = rezoned（現在値）
    A3: conservative_remap(x_r_old, x_z_old, x_r, x_z, vol_old, vol, sweep=first_dir)  × (5+n_mat) ← mass,mom_r,mom_z,e_i,e_e,volFrac
    A3: conservative_remap(x_r_old, x_z_old, x_r, x_z, vol_old, vol, sweep=second_dir) × (5+n_mat) ← mass,mom_r,mom_z,e_i,e_e,volFrac
    // volFrac 正規化（§3.5, ARCHITECTURE §4.3）: remap 後に Σ_mat volFrac[c,mat] = 1 を強制
    A5: normalize_volFrac(volFrac, error_flags, n_cells, n_mat)
    // 退化ガード: Σ < ε_vf (1e-30) のセルは volfrac_degenerate フラグ設定、argmax成分を1/他0に設定（NUMERICS §3.3.4）
    --- Post-remap reclosure（ARCHITECTURE §5.1 必須シーケンス）---
    H7:  compute_cell_geometry         // 新メッシュの体積・面積・特性長を再計算（remap前のH7と同一結果だが reclosure の自己完結性のため再実行）
    H8:  compute_density               // mass / V_new → ρ_new
    A4:  project_cell_velocity_to_nodes  // セル中心速度→節点速度（NUMERICS §3.3.4 質量重み投影）。velocity_bc_mode: 0=free, 1=reflect, 2=fixed, 3=state_supply; mode 3 は z-boundary v_z をゼロ化せず supplied/restored material velocity を保持
    H14: eos_inverse(species=0)        // ρ_new, ee → Te
    H14: eos_inverse(species=1)        // ρ_new, ei → Ti
    H13: eos_forward                   // ρ_new, Te, Ti → Pe, Pi, Cv_e, Cv_i
    H15: compute_sound_speed           // NUMERICS §1.1.6:
         // ideal_gas: c_s = sqrt((γ_e P_e + γ_i P_i)/ρ), γ=5/3
         // table_eos: c_s² = (∂P/∂ρ)|_T + T/(ρ Cv)(∂P/∂T|_ρ)², EOS テーブル偏微分使用
         //            P=Pe+Pi, c_v=c_v,e+c_v,i [erg/(g·eV)], C_v=ρ c_v, T=T_eff=(Ti+Z̄Te)/(1+Z̄)
    U2:  floor_clamp                   // 安全策適用
    // **hash grid 再構築（必須）**: U7 のフォールバック探索（§9.5）が使用する hash grid は
    // rezone 前のメッシュ座標で構築されているため、rezone 後に再構築が必要。
    // 再構築しないと stale なビン割り当てでフォールバック探索が失敗し、
    // 粒子が不必要に numerical_loss に計上される（NUMERICS §9.5 構築タイミング準拠）。
    // コスト: O(N_cells)（全セル AABB 再計算 + CSR リスト再生成）。125K セルで ~0.1ms。
    build_hash_grid(x_r, x_z, nr, nz, M_R, M_Z, max_per_bin → hash_grid)  // rezone 後メッシュで再構築
    U7:  cell_search_after_rezone      // stencil walk + hash grid fallback (NUMERICS §9)
    [MPI] halo_exchange(rho, Te, Ti, Pe, Pi)  // remap + reclosure 後の原始変数を交換（NUMERICS §12.2.2 行12）
    [MPI] halo_exchange(x_r, x_z, v_r, v_z)  // rezone 後のノード座標を交換

═══════════════════════════════════════════════════════
 Phase 6: ステップ後処理
═══════════════════════════════════════════════════════
  U4:  cfl_reduction → CUB Min (dt_hydro, dt_cond, dt_rad)
  // **モジュール無効時ガード**: dt_cond は conduction.enabled=True 時のみ compute_dt_cond を起動。
  //   conduction.enabled=False → dt_cond = DBL_MAX（D_eff 未計算のためカーネル呼び出しを省略）。
  //   同様に dt_rad は radiation.enabled=True 時のみ compute_dt_rad を起動。
  //   radiation.enabled=False → dt_rad = DBL_MAX。ホスト側で設定し CUB Min は dt_hydro のみ実行。
  // **U4→U2 順序の根拠**: U4 は Phase 5 の H15 出力 c_s を使用して dt を計算する。
  // U2 がこの後に Te/ρ をクランプしても、クランプ対象セルは T≈T_floor, ρ≈ρ_floor であり
  // c_s ∝ sqrt(T_floor) → dt_hydro = dx/c_s が大きい（CFL非制約）。
  // dt_rad: β ∝ T³ → 0, σ_P ∝ ρκ_P → 小 → denom ≈ 0 → dt_rad_cell = DBL_MAX（非制約）。
  // dt_cond: D_eff ∝ κ_SH/ρ ∝ T^{5/2}/ρ → T_floor^{5/2} ≈ 3e-8 → dt_cond_cell 極大（非制約）。
  // 3制約全てについて、クランプ対象セルは CFL を制約しない。
  // したがって U4 の dt は保守的（dt_pre_clamp ≤ dt_post_clamp）であり安全。
  U2:  floor_clamp
  // **Phase 6 U2 のステップ境界 stale window 注記**（Phase 1/4 と同パターン）:
  // U2 後に H13 は起動しない。次ステップの Phase 1 Predictor H4 が P^n を使用するが、
  // U2 がクランプしたセルの圧力寄与は ΔP ~ ρ_floor × kB × T_floor ≈ 10^{-22} dyne/cm²。
  // Phase 1 Predictor H13（line ~3258）が H4 後に P を再計算するため、
  // stale P^n は Predictor 内のみで使用される（Corrector には影響しない）。
  // E_rad（飛行中輻射エネルギー）は粒子プール上の量 → U3 の前に CUB 集約が必要:
  // **alive フィルタ必須**: ALE U7（Phase 5）が cell_search_fatal=False 時に粒子を kill（alive=0）するため、
  // [0..n_alive-1] 範囲に dead 粒子が混在しうる。フィルタなしの素朴 Sum では dead 粒子エネルギーが
  // E_census と E_numerical_loss の両方に計上され、エネルギー収支が破綻する。
  if (n_alive > 0):
    CUB DeviceReduce::Sum(
      TransformInputIterator(pool.energy, pool.alive,
        [](double e, uint8_t a) -> double { return (a == 1) ? e : 0.0; }),
      n_alive) → E_census  // [erg]。alive==1 粒子のみ集約
  else:
    E_census = 0.0  // n_alive==0: CUB Reduce は num_items=0 で出力未定義のためホスト側で明示設定
  U3:  energy_budget → CUB Sum (×3: E_kin, E_int_e, E_int_i)  // E_rad=E_census は上記 CUB Sum で算出済み。E_escape[G] は atomicAdd 累積済み（D2H のみ）
  U5:  nan_check
  [SYNC] → [D2H] dt_hydro, dt_cond, dt_rad, E_kin, E_int_e, E_int_i, E_rad, E_escape,
                  E_numerical_loss, E_floor_injected, E_safety, step_E_solver,
                  error_flags, clamp_count,
                  opacity_clamp_count, mmatrix_fix_count, rad_mom_dep  // NUMERICS §11.3 + §7.8 + Kershaw §4.3
  [MPI] MPI_Allreduce(SUM): E_kin, E_int_e, E_int_i, E_rad, E_escape[G],
                            E_numerical_loss, E_floor_injected, E_safety,
                            step_E_pdV_bdry, step_E_Marshak_in, step_E_solver,
                            step_laser_dep_total  // エネルギー収支は全ランク合算が必要（NUMERICS §10.2, ARCHITECTURE §5.2 MPI セマンティクス）
  // **step_laser_escaped は Allreduce 対象外**（v1.0 replicated strategy では全 rank が同一レイトレースを
  // 実行するため P_unabsorbed は既にグローバル値。SUM すると N_ranks 倍に過大計上される）。
  // Allreduce 後にホストで算出:
  //   step_laser_escaped = E_laser_incident - step_laser_dep_total
  //   E_laser_incident = Σ_g P_g(t_now) × Δt（全rank同一値、MPI不要）
  // 将来 distributed strategy では step_laser_escaped を Allreduce(SUM) に追加する
  [MPI] MPI_Allreduce(MIN): dt_hydro, dt_cond, dt_rad  // CFL は全ランクの最小値（NUMERICS §2.2, §12.2.2 行1）
  [MPI] MPI_Allreduce(MAX): error_flags（フラグ部分: 0/1 → MAX≡OR で正しい）
  //   注: temperature_overshoot（カウント値, atomicAdd）は MAX ではランク最大値のみ取得。
  //   グローバル合計が必要な場合は別途 SUM reduction するか、
  //   MAX 値を閾値判定に使う（保守的: いずれかのランクが閾値超過→全ランクで検出）。
  //   v1.0 では MAX を採用（閾値判定は保守的に機能し、diagnostics 記録は per-rank-max として扱う）
  // --- Host-side cumulative diagnostics accumulation（ARCHITECTURE §5.2 準拠）---
  // State の累積フィールドは HOST 側で毎ステップ加算する（device 側は per-step リセット）:
  //   State.E_floor_injected   += step_E_floor_injected     // Phase 0 でデバイス側ゼロ初期化済み
  //   State.E_safety           += step_E_safety
  //   State.E_numerical_loss   += step_E_numerical_loss
  //   State.E_rad_escaped      += sum(E_escape[0..G-1])     // E_escape は群別→全群合算で累積
  //   State.E_laser_deposited  += step_laser_dep_total      // Allreduce(SUM) 後のグローバル値。Phase 3 CUB Sum → D2H → Allreduce
  //   step_laser_escaped = E_laser_incident - step_laser_dep_total  // Allreduce 後に算出（replicated: E_incident は全rank同一）
  //   State.E_laser_escaped    += step_laser_escaped        // Allreduce 後のホスト計算結果を累積
  //   State.E_pdV_bdry         += step_E_pdV_bdry           // Phase 1/5 のホスト計算を累積（NUMERICS §10.2）
  //   State.E_Marshak_in       += step_E_Marshak_in         // Phase 4 R13 後のホスト解析計算を累積
  //   State.E_solver           += step_E_solver             // v1.0=0（Hypre有効時のみ非ゼロ）
  // --- Host-side Δt 決定擬似コード（NUMERICS §2.2 + U4 §7.3 Step 4）---
  // dt_cfl = min(dt_hydro, dt_cond, dt_rad)        // U4 デバイス出力 → Allreduce(MIN) 後のグローバル値
  // dt_laser = dt_hydro                             // NUMERICS §2.2(d): hydro 追従（dt_laser は独立制約なし）
  // dt_phys = min(dt_cfl, dt.max_s)                 // ユーザ上限（SPECIFICATION §6.4.7）
  // if (step == 0 && !is_restart):
  //   dt = min(dt_phys, dt.initial_s)               // step 0 のみ（NUMERICS §2.2(e)）
  // elif (is_restart && step == restart_step):
  //   dt = min(dt_phys, checkpoint_dt)              // リスタート直後（NUMERICS §2.2(e)）
  // else:
  //   dt = min(dt_phys, growth_factor * dt_old)     // 成長制限（g_dt=1.2、NUMERICS §2.2）
  // dt = min(dt, dt_output)                         // 出力時刻整合（NUMERICS §2.2(f)）
  // if (dt < dt.min_s): FATAL("dt stalling")        // 下限チェック（NUMERICS §2.2(e)）
  [Host: エラーチェック、diagnostics]
  [Host: opacity_clamp_count 判定: >10 → per-step最大10件WARNING表示 + "N times this step" 集約メッセージ（NUMERICS §11.3）]
  [Host: 出力トリガー判定（step%X_every==0 OR t>=t_next_X-ε のOR論理）]
  // **出力時の熱力学的整合性**: HDF5 出力が Pe/Pi/Cv を含む場合、
  // U2 後の stale Pe/Pi がスナップショットに混入する。出力頻度が低い（~100-1000 step に1回）ため、
  // 出力トリガー成立時のみ H13 を追加実行して Pe/Pi/Cv を最新化する。
  // checkpoint 出力では Te/Ti/ee/ei を保存するため Pe/Pi の stale は restart に影響しない（restart 時に H13 再実行）
  [条件付き: if (output_triggered) { H13: eos_forward → Pe,Pi,Cv 最新化 }]
  [条件付き: HDF5出力、checkpoint、t_next_X更新]
```

### 9.1 1ステップあたりのカーネル起動回数

| Phase | カーネル起動数 | 支配的カーネル |
|-------|-------------|-------------|
| Hydro H(Δt/2) × 2 | ~30 × 2 = ~60 | H4 (corner force), H14 (EOS inverse) |
| Conduction | ~3 + s（STS） | C2 (Kershaw build ×1) + C3 (apply ×s、s=1–27) |
| Laser | ~5 | L3/L4 (ray trace) |
| Radiation | ソルバーと反復回数による | FLD: 群の三重対角（1D）/ CG（2D）と物質の Newton、S\(_N\): sweep・DSA・物質の Newton（§6.7/§6.8） |
| Utility | ~10 | — |
| **合計** | **~100** | |

> **注**: カーネル起動オーバーヘッド ~5μs/launch × 100 = ~500μs ≪ 計算時間（~50ms/step）

---

## 10. メモリアクセスパターンと最適化

### 10.1 アクセスパターン分類

| パターン | 該当カーネル | 帯域効率 | 最適化手法 |
|---------|------------|---------|-----------|
| **Coalesced SoA** | セル・節点の場、燃焼の α 粒子 | 100% | SoAレイアウト |
| **Stencil** | Kershaw, AV, corner force | 80-90% | 構造格子の固定ストライド |
| **Random cell read** | レーザー光線（セルデータ参照） | 30-60% | `__ldg()` |
| **Reduction** | CFL, energy budget | 90%+ | CUB ライブラリ |

### 10.2 L2キャッシュ戦略

A100: L2キャッシュ 40MB。

**キャッシュに収まるデータ**:
- セルフィールド（125Kセル × 10フィールド × 8B = 10MB）→ 収まる（500×250メッシュ、社内の性能記録 P1-P3準拠）
- LaserMeshフィールド（32Kノード × 5フィールド × 8B = 1.3MB）→ 収まる

**方針**: セルデータは`__ldg()`で明示的にL2キャッシュを活用。（退役したモンテカルロ輻射の ddmc_mode 配列は収まり、
光子粒子の SoA（100万粒子 × 93B = 93MB）は収まらないので streaming としていた。）

### 10.3 レジスタスピル防止と `__launch_bounds__` 仕様

全主要カーネルに `__launch_bounds__(block_size, min_blocks)` を付与し、
コンパイラのレジスタ割り当てを制御する。

**`__launch_bounds__` 一覧**:

| カーネル | block_size | min_blocks | 最大reg/thread | occupancy保証 | 根拠 |
|---------|-----------|-----------|---------------|-------------|------|
| `kershaw_stencil_build` (C2) | 256 | 2 | 128 | 25% (512 threads/SM) | ~45 reg、compute-bound |
| `kershaw_apply` (C3) | 256 | 4 | 64 | 50% | ~15 reg、STSステージ内 |
| `ray_trace_2d` (L3) | 64 | 16 | 64 | 50% | ~40 reg、warp発散対策 |
| `ray_trace_3d` (L4) | 64 | 16 | 64 | 50% | ~46 reg、3D拡張 |
| `eos_inverse` (H14) | 256 | 4 | 64 | 50% | Newton反復~20 iter |
| `conservative_remap` (A3) | 256 | 4 | 64 | 50% | Van Leer limiter |

**算出方法**：A100基準（65536 reg/SM, 2048 threads/SM）
- `max_reg = 65536 / (block_size × min_blocks)`
- 例：`__launch_bounds__(128, 8)` → 65536 / 1024 = 64 reg/thread

**効果**：
- レジスタスピル（local memory fallback）を防止。local memory アクセスは L1 経由だが latency ~100 cycles
- コンパイラが制約に合わせてレジスタプレッシャーを自動調整（変数の再計算 vs spill のトレードオフ）
- 全主要カーネルに適用することで、occupancy が予測可能になりプロファイル時の解釈が容易になる

### 10.4 共有メモリ使用

| カーネル | 共有メモリ/block | 用途 | Phase |
|---------|----------------|------|-------|
| `energy_budget` (U3) | 256 × 3 × 8B = 6 KB | 部分和accumulation（E_kin, E_int_e, E_int_i） | v1.0 |
| `cfl_reduction` (U4) | 256 × 8B = 2 KB | min reduction | v1.0 |

**occupancy への影響**（A100: 164 KB shared/SM）:
- U3 (energy_budget) は 1 block/SM あたり 10 KB だが、grid_size が小さいため制約なし

---

## 11. パフォーマンス推定

> **【状態注記 2026-07-10、2026-09-29 更新】** 本節の見積りは退役したモンテカルロ輻射（粒子輸送、alive 粒子数前提）の歴史的推定で、
> Radiation の行と粒子数スケーリング・ボトルネック特定フローの粒子の項はそのコードとともに退役した（コードは
> `retired/radiation_monte_carlo/`）。現行の性能実測は 社内の性能記録（host オーバーヘッド削減系列の wall/steps + host API 呼数）と
> `ops/runpod/bench/CALIBRATION.md` を正とする。

### 11.1 Phase別時間内訳推定

**前提条件**: 2D_RZ 500×250メッシュ（125Kセル）、16群、alive粒子数 N_p = 100万、A100。

> **注**: alive粒子数はソース投入＋census残存の合計であり、`particles_per_cell_group`（既定50）と
> メッシュサイズから一意には決まらない。100万粒子は中規模テストケースの典型値。
> 本番計算（200+/cell/group、SPECIFICATION §8.2.1）では N_p ≫ 10⁶ となり、
> Radiation phase の時間が粒子数に線形に増加する（§11.2参照）。

| Phase | 推定時間/step | 根拠 |
|-------|-------------|------|
| Hydro × 2 | ~3 ms | 125Kセル（500×250）、~30 kernels × 2、メモリバウンド |
| Conduction | ~1–3 ms | Kershaw build ×1 + STS apply ×s（s=1–27、NUMERICS §4.2.1） |
| Laser | ~5 ms | 5000レイ × ~200 substeps、block=64 |
| **Radiation** | **~40 ms** | **100万粒子 × ~20イベント/粒子** |
| Utility | ~0.5 ms | reduction + floor |
| MPI通信 | ~1 ms | halo + particle migration |
| **合計** | **~50 ms/step** | |

> Radiation が全体の ~80% を占める。KPI目標（社内の性能記録 参照: ≥2×10⁹ events/s）に対し:
> 100万粒子 × 20イベント = 2×10⁷ events、40ms → 5×10⁸ events/s。
> 目標達成には粒子数増加（10⁶→10⁷）または最適化が必要。

### 11.2 スケーリング特性

**粒子数スケーリング**: Radiation phaseは粒子数に線形。
10⁷粒子 → ~400ms/step（Radiation支配）。

**セル数スケーリング**: Hydro + Conduction + Laserはセル数に線形。
1M cells → Hydro ~25ms、Conduction ~12ms、Laser ~5ms。

**群数スケーリング**: G=16→32でFleck計算・モード判定が2×、粒子数も概ね2×。

### 11.3 ボトルネック特定フロー

```
1. Nsight Systems で全体プロファイル → Phase別内訳
2. Radiation が支配的 →
   2a. Nsight Compute で imc_transport の詳細:
       - warp divergence → 30%超なら mode partition を実装
       - atomic 競合 → L2 sector conflict が高ければ warp集約を実装
       - register spill → __launch_bounds__ 調整
   2b. 粒子ソートの効果確認（ON/OFF比較）
3. Laser が想定以上 →
   - レイ長の分散確認 → R座標ソートの効果
   - LaserMesh deposit の atomic 競合確認
4. Hydro が想定以上 →
   - EOS inverse のNewton収束回数確認
   - メモリ帯域利用率確認（roofline model）
```

---

## 12. CUBライブラリ使用一覧

| 操作 | CUB API | 使用箇所 | 一時メモリ(概算) |
|------|---------|---------|-----------------|
| Min reduction | `DeviceReduce::Min` | CFL dt, mesh quality | ~256 B |
| Sum reduction | `DeviceReduce::Sum` | Energy budget (×5) | ~256 B |

（モンテカルロ輻射が使っていた Prefix sum（source particle offsets）と Radix sort（Composite Key Sort、~24 × N_particles B）と
その Scratch の見積りは 2026-09-29 に退役した。）

---

## 12.5 カーネル→マイルストーン対応表

各カーネルを実装するマイルストーンの対応関係を以下に示す。

| Kernel ID | Name | Milestone |
|-----------|------|-----------|
| H1-H12 | Hydro kernels (hydro_active, node_mass, area_vectors, corner_force, velocity/position/geometry/density/divergence update, artificial_viscosity, energy_update_ion/electron) | M3（1D）, M4（2D RZ拡張） |
| H13-H16 | EOS forward/inverse, sound speed, hydro BC | M3（1D）, M4（テーブルEOS） |
| A1-A5 | ALE (mesh_quality_check, winslow_jacobi_step, conservative_remap, project_cell_velocity_to_nodes, normalize_volFrac) | M4 |
| C1-C4 | Conduction (spitzer_deff, kershaw_stencil_build/apply, 1d_tridiag) | M4 |
| L1-L7 | Laser (laser_mesh_map, density_gradient, ray_trace_2d/3d, deposit, ray_skip_check, radial_absorption_1d) | M7 |
| R1-R16 | モンテカルロ輻射（M5 IMC、M6 DDMC 統合）— 2026-09-29 に退役 | — |
| U1 | Source injection | M5 |
| U2 | floor clamp | M4（Hydro/Conduction） |
| U7 | cell_search_after_rezone — 退役 | M4（ALE） |
| U3 | Energy budget | M5（放射）, M8（統合拡張） |
| U4-U5 | CFL reduction, NaN check | M8（統合） |
| U6 | Q_ei exchange | M4 |
| U8 | compute_zbar | M05（Materials） |
| U9 | compute_opacities — 退役 | M05（放射前準備） |
| P1-P4 | Parallel (halo pack/unpack cell/node)。P5/P6（粒子の rank 間移動）は退役 | M9 |

---

## 13. 最適化ロードマップと実装段階

### v1.0 baseline — 設計済み
以下は v1.0 初版の仕様として本文書内で定義済みである。

（1. Composite Key Sort と 2. warp-level タリー集約はモンテカルロ輻射の最適化で、2026-09-29 に退役した。）
3. **`__launch_bounds__`**: 全主要カーネルに適用（§10.3）
4. **`__ldg()`**: セルデータの read-only アクセス（§10.2）

### Phase B（性能最適化）— 本文書で仕様定義済み、実装はM5以降
以下は本文書内でアルゴリズムとカーネル仕様を完全に定義しており、実装可能な状態である。

5. **計算-通信オーバーラップ**: 内部セル計算とハロー交換の非同期並列実行（NUMERICS §12.5.5、ARCHITECTURE §5.6.2）
   - namelist: 未実装（有効化の設定だった `Parallel.gpu_optimization.compute_comm_overlap` は受理して無視される）
   - 適用: Hydro, Conduction のセルベースカーネル
   - 効果：4 GPU時 ~1.2 ms/step 隠蔽（2-3%改善）、GPU数増加で効果増大

### 高度な最適化 — 将来検討
以下は将来の検討事項として記録する。

9. **Kernel fusion**: 連続する小カーネルの統合（Hydro predictor-corrector 内）
10. **Multi-stream execution**: 独立カーネルの並列起動
11. **FP16 テーブル補間**: EOS/Opacity テーブルをFP16化しメモリ帯域削減

---
