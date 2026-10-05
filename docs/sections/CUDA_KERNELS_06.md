<!-- 分割元: docs/CUDA_KERNELS.md | このファイルは参照用です。原本（docs/CUDA_KERNELS.md）が権威です。 -->
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
