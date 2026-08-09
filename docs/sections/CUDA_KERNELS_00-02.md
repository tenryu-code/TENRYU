<!-- 分割元: docs/CUDA_KERNELS.md | このファイルは参照用です。原本（docs/CUDA_KERNELS.md）が権威です。 -->
# TENRYU — CUDA_KERNELS.md
CUDAカーネルの設計仕様書。各カーネルのスレッド/ブロック構成、メモリアクセスパターン、
レジスタ圧力、最適化戦略、および1タイムステップ内のカーネル起動シーケンスを定義する。

**参照GPU**: NVIDIA A100 80GB SXM4（108 SM、65536 reg/SM、164 KB shared/SM、2048 threads/SM）
**最低要件**: Compute Capability 6.0+（`atomicAdd(double*)`）。CC 7.0+（Volta以降）推奨（warp-level `__match_any_sync` タリー集約に必要、§6.4）。CC < 7.0 では `tally_mode` が自動的に `"global"` にフォールバック（NUMERICS §10.3.5）。

---

## 0. 設計原則

### 0.1 カーネル分類と起動パラメータ

| カーネル種別 | block_size | 根拠 | 典型 occupancy |
|------------|-----------|------|---------------|
| **cell-based** | 256 | メモリバウンド、低レジスタ（<32） | ≥75% |
| **node-based** | 256 | 同上 | ≥75% |
| **particle-based** | 128 | レジスタ圧力高（~60 reg）、50%で十分 | ≥50% |
| **ray-based** | 64 | 分岐多、レイ長の分散大 | ≥25% |
| **reduction** | 256 | CUB標準 | — |
| **pack/unpack** | 256 | メモリバウンド | ≥75% |

grid_size は原則 `(N + block_size - 1) / block_size` で算出する。
**末尾スレッドガード必須**: ceil-grid 起動する全カーネルは先頭で `int tid = blockIdx.x * blockDim.x + threadIdx.x; if (tid >= N) return;` を実行すること。以下の疑似コードでは省略する場合があるが、実装時には必ず挿入する。
**例外**:
- **Persistent Warp**（R8 `imc_transport`）: `grid = n_sm × blocks_per_sm`（SM数依存、固定、粒子数非依存。NUMERICS §6.6.2）
- **CUB ライブラリ呼び出し**（RadixSort, Partition, Reduce 等）: CUB 内部がグリッドサイズを自動決定

### 0.2 メモリアクセス方針

- **SoA（Structure of Arrays）**が原則。32スレッドが連続アドレスを読む（coalesced access）
- セルデータへの読み込み（opacity等）は `__ldg()` intrinsic で L2 キャッシュ経由のread-only アクセス
- 粒子→セルデータ参照はランダムアクセスだが、**セルID順ソート**（§0.5）で局所性を改善
- タリー（`rad_dep`, `rad_E_tally`）は `atomicAdd(double*)` で集約

### 0.2b 型安全・オーバーフロー制約

- **粒子数**: `N_p_total`, `N_total` は `int`（int32）。v1.0 の最大粒子数は 2^31-1 ≈ 21.5億。PERFORMANCE P1-P3 の上限 ~10M 粒子では十分。将来 >2B 粒子が必要な場合は `int64_t` 移行を検討
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
主な消費先: Radiation RadixSort (~24 × N_particles)、Hydro F_r/F_z (2 × n_nodes × 8B)、
Conduction Te_old (n_cells × 8B、Hypre パスの E_solver 用、§4.5 step 3)、
DDMC Kershaw stencil (9 × (n_cells+n_ghost) × G × 8B、R3 パスA用。n_cells=40K, G=16 で ~46MB)。
Strang splitting 内は逐次のため同時使用なし。

> **排他規約**：Scratch を使用するカーネル・CUB 呼び出しは **compute stream** 上でのみ実行する。
> utility/comm stream から Scratch を参照すると、逐次実行の保証が崩れ競合が発生する。

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

### 0.6 エラーフラグプロトコル（ARCHITECTURE §10.1 準拠）

**物理判定を含むカーネル**は `DeviceErrorFlags*` を引数に取り、異常検出時に `atomicExch` で書き込む。
カーネルは中断せず実行を完了し、host側が事後チェックする。

**error_flags 必須カーネル**（物理的異常を検出しうるもの）：
- EOS: `eos_forward`（C_v≤0 クランプ、eos_ion_negative: SESAME 2-table差分で P_i or e_i が負→max(0)クランプ）, `eos_inverse`（eos_newton_nonconverge: Newton MAX_ITER到達）
- Hydro: `compute_sound_speed`（sound_speed_negative: テーブルEOS c_s²<0 クランプ、NUMERICS §1.1.6）
- Transport: `imc_transport_persistent`（NaN/負エネルギー/セル逸脱）, `ddmc_event_loop`（同上 + ddmc_sigma_tot_zero: Σ^tot≤Σ_floor、NUMERICS §7.5）
- Laser: `ray_trace_2d`, `ray_trace_3d`（MAX_RAY_STEPS到達 → infinite_loop フラグ）
- Safety: `floor_clamp`（フロアカウンタ）, `source_injection`（negative_source_dep: 負 rad_dep/laser_dep 検出、NUMERICS §11.5）
- Material: `compute_zbar` / 混合則カーネル（volfrac_degenerate: 体積分率縮退時フラグ、NUMERICS §1.1.6）
- Opacity: `compute_opacities`（opacity_out_of_range: テーブル範囲外クランプ、NUMERICS §11.3）
- MPI: `emigrant_detect_pack`（P5、emigrant_bufオーバーフロー検出 + emigrant_invalid_face: faceデコード範囲外検出）
- ALE: `mesh_quality_check`（mesh_tangle: ヤコビアン J < 0 検出、ARCHITECTURE §10.1 即ERROR）
- ALE: `normalize_volFrac`（volfrac_degenerate: 体積分率縮退、NUMERICS §1.1.6）

**error_flags 不要カーネル**（純粋な配列操作・幾何計算・MPI補助）：
halo_pack/unpack, SoA gather, composite_key build, tally_finalize, compute_node_mass,
compute_area_vectors, compute_cell_geometry 等。
これらのカーネルでは error_flags 引数を省略してよい。

---

## 1. カーネル一覧（モジュール別）

> **N列の規約**: N = CUDA grid サイズ計算に使用するスレッド数（`grid = (N + block-1) / block`）。
> 「G群ループ内包」と記載のカーネルは、1スレッドが全G群を内部ループで処理する（N ≠ N×G）。
> R10（tally_finalize）のみ例外で N=n_cells×G（1スレッド=1セル×群ペア）。

### 1.1 Hydro（16カーネル）

| ID | カーネル名 | 種別 | N | 主要入出力 |
|----|-----------|------|---|-----------|
| H1 | `hydro_active_update` | cell | n_cells | Te → hydro_active |
| H2 | `compute_node_mass` | cell gather | n_cells | cell mass, node R → R-weighted node mass（§3.2.4）|
| H3 | `compute_area_vectors` | cell | n_cells | node coords → S_{c,k} |
| H4 | `compute_corner_force` | node | n_nodes | x,P,Q → F_r,F_z |
| H5 | `velocity_update` | node | n_nodes | v,F,m → v_new |
| H6 | `position_update` | node | n_nodes | r,v → r_new |
| H7 | `compute_cell_geometry` | cell | n_cells | node coords → V,A_cell,centroid,Δl,face_area,ell_ddmc |
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

### 1.4 Laser（7カーネル）

| ID | カーネル名 | 種別 | N | 主要入出力 |
|----|-----------|------|---|-----------|
| L1 | `laser_mesh_map` | LM-node | N_LM | HydroMesh → LaserMesh物理量 |
| L2 | `compute_density_gradient` | LM-node | N_LM | n_hat → grad_n_hat |
| L3 | `ray_trace_2d` | ray | n_rays | Leapfrog + IB吸収（1D_SPH用） |
| L4 | `ray_trace_3d` | ray | n_rays | 3D Leapfrog + IB吸収（2D_RZ用） |
| L5 | `deposit_lm_to_hydro` | cell/LM | n_cells | LaserMesh deposit → laser_dep |
| L6 | `ray_skip_check` | LM-cell | n_LM_cells | δ変化量計算（LaserMesh対応セルのみ） |
| L7 | `radial_absorption_1d_kernel` | serial | n_cells | 1D radial flux → `deposit_power_cell` |

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
| U1 | `source_injection` | cell | n_cells | laser_dep,rad_dep → ee |
| U2 | `floor_clamp` | cell | n_cells | rho,Te,Ti → clamped |
| U3 | `energy_budget` | cell→scalar | n_cells | fields → E_kin,E_int,E_rad |
| U4 | `cfl_reduction` | cell→scalar | n_cells | c_s,u,Δl → dt_min（3 sub-kernels: dt_hydro/dt_cond/dt_rad）|
| U5 | `nan_check` | cell | n_cells | all fields → error flags |
| U6 | `qei_exchange` | cell | n_cells | Te,Ti,rho,Zbar,A_eff → Q_ei |
| U7 | `cell_search_after_rezone` | particle | n_alive | ALE rezone後の粒子セル再同定（IMCのみ、DDMC pos=NaNスキップ）|
| U8 | `compute_zbar` | cell | n_cells | Te,rho,volFrac → Z̄,A_eff（fixed/thomas_fermi/tabular、§7.6）|
| U9 | `compute_opacities` | cell | n_cells+n_ghost | rho,Te,Zbar,volFrac,OpacityTable → σ_a,σ_s,σ_R,σ_P,σ_t（ghost含む、G群ループ内包、ARCHITECTURE §4.7）|

### 1.7 Parallel（6カーネル）

| ID | カーネル名 | 種別 | N | 主要入出力 |
|----|-----------|------|---|-----------|
| P1 | `halo_pack_cell` | cell | n_halo | fields → send_buf |
| P2 | `halo_unpack_cell` | cell | n_halo | recv_buf → ghost fields |
| P3 | `halo_pack_node` | node | n_halo | fields → send_buf |
| P4 | `halo_unpack_node` | node | n_halo | recv_buf → ghost fields |
| P5 | `emigrant_detect_pack` | particle | n_alive | pool → emigrant_buf |
| P6 | `immigrant_unpack_merge` | particle | n_recv | recv_buf → pool |

---

## 2. Hydro カーネル群（詳細設計）

### 2.1 H1: hydro_active_update

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
                                    // DDMC R2/R3/R3b、Marshak R13 が参照。H7 の座標参照と融合して帯域を節約
    double* __restrict__ ell_ddmc,  // [n_cells] out: DDMC代表長 ℓ_i = 2 × min_f(d_{center→face}) [cm]
                                    // NUMERICS §7.1.1 準拠。R2 ddmc_mode_judge が τ = σ_R × ℓ_i で参照
    const double* __restrict__ x_r, // [n_nodes]
    const double* __restrict__ x_z, // [n_nodes]
    int nr, int nz,
    int n_faces                     // 2D_RZ=4, 1D_SPH=2（face_area の列数。H9 と同一引数）
);
```

- **block**: 256, **grid**: `(n_cells+255)/256`
- **処理**: 4頂点座標を読み、§3.2.2 (RZ体積) + §3.2.3 (Shoelace面積) + centroid + 特性長 Δl = sqrt(A_cell) 計算（NUMERICS §3.2.2-3.2.3）。Δl は H10 (人工粘性) および U4 (CFL) で使用される
  - **face_area**: 各面の面積を算出（2D_RZ: 辺長×r̄_face×2π、1D_SPH: 4πr²）。DDMC leak (R2/R3/R3b)、Marshak (R13) が参照
  - **ell_ddmc**: セル中心→各面中点距離の最小値 × 2（NUMERICS §7.1.1）。R2 が τ = σ_R × ℓ_i で DDMC 判定に使用
- **融合理由**: 全て同じ4頂点座標を入力とし、出力が異なるだけ。分離すると座標の再読み込みが発生。face_area と ell_ddmc も同一座標から算出するため融合が自然
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
  - **table_eos**: `c_s² = (∂P/∂ρ)|_T + T/(ρ Cv)(∂P/∂T|_ρ)²`。EOSテーブル偏微分使用。
    P=Pe+Pi、c_v=c_v,e+c_v,i [erg/(g·eV)]、T=T_eff=(Ti+Z̄Te)/(1+Z̄)
  - **c_s² < 0 ガード**: `c_s = sqrt(max(c_s², 0))`。クランプ発生時に `atomicExch(&error_flags->sound_speed_negative, 1)` 設定
- **レジスタ**: ~15（ideal）/ ~25（table: 偏微分計算含む）
- **メモリ**: coalesced read（セルフィールド）。テーブルEOS時は `__ldg()` でテーブル参照

### 2.4 H13/H14: EOS カーネル

> **実装構造**（ARCHITECTURE §4.3.1 準拠）: 各EOS操作は `__device__` コア関数
> （`eos_forward_impl`, `eos_inverse_impl`）として実装され、他カーネル（H11, H12等）から
> インライン呼び出し可能。以下の `__global__` カーネルはスタンドアロン起動用のラッパーである。

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
- `device_eos_sound_speed` uses \(\Gamma_1\)-based evaluation with local finite differences and returns a guarded/fallback sound speed.

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
    double C_H                              // 人工熱流束係数（既定 1.0）
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
//   Center ghost: r_{-1}=-r_1, u_{-1}=-u_1
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
  - **free**: ゴーストセルの圧力 P_ghost = 0（P_ext=0、ICF標準の真空外側条件）、密度 ρ_ghost = ρ_interior（最近接内部セルコピー）、温度はゼロ勾配外挿（Te_ghost=Te_interior, Ti_ghost=Ti_interior）。Qvisc_ghost = 0。速度は運動方程式で自由決定（NUMERICS §8.1(a)）
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

