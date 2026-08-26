<!-- 分割元: docs/CUDA_KERNELS.md | このファイルは参照用です。原本（docs/CUDA_KERNELS.md）が権威です。 -->
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

### 6.7 現行 FLD カーネル群（`mode="multigroup_diffusion"` — 決定論、1D/2D_RZ）【CURRENT】

2026-07-10 新設（doc 監査 open item の解消）。数理は NUMERICS §6.7、実装は
`src/radiation/fld_1d_gpu.cu` / `fld_2d_rz_gpu.cu` / `nlte_coeffs.cu`。
ブロック定数は両者 `kBlock=256`、grid は `(N+kBlock-1)/kBlock` 形式。
注意: `deterministic_diffusion_1d.cu` / `diffusion_{conversion,interface,source_solve}.cu` は
**FLD ではなく退役 IMC-DDMC ハイブリッド側**の拡散部品（`imc.cpp` からのみ参照）。

#### 6.7.1 FLD 1D（`fld_1d_gpu.cu`、エントリ `advance_radiation_step_fld_1d` :1672）

| kernel | 目的 | thread mapping | 備考 |
|---|---|---|---|
| `build_eta_from_planck_kernel` (:358) | η_g = c·σ_a·a·T⁴·b_g | 1 thread/(cell×group) | constant-opacity 経路 |
| `compute_fleck_for_fld_kernel` (:489) | Fleck f=1/(1+z) | 1 thread/cell（群は thread 内 for） | NLTE 時は nlte_coeffs.cu が代替 |
| `compute_marshak_finc_kernel` (:570) | Marshak 群別入射流束 | 1 thread/group | 一時 DeviceArray |
| `assemble_fld_tridiag_kernel<GEOM>` (:698) | 三重対角 + RHS 組立 | 1 thread/(cell×group) | **面 D は face-centered flux limiter**（`fld_face_diffusion_coeff` :393 を両面評価）。GEOM∈{球,円筒,平面} template |
| `publish_solution_kernel` (:744) | 解 → rad_E 公開 | 1 thread/(cell×group) | |
| `snapshot_Te_kernel` (:763) | Te→Te_old 退避 | 1 thread/cell | Fleck ブレンド基準 |
| `update_matter_kernel` (:980) | 物質側 Newton（Te/ee/Pe/rad_dep/rad_emit） | **1 block/cell、32 thread、動的 shared 2·G·8B** | 群を warp 内分担、shared 縮約 |
| `max_reduce_kernel` (:1043) | max\|ΔT\| 縮約 | 1 thread/cell + shared/atomic | 外反復収束用 |
| `escaped_energy_kernel<GEOM>` (:1114) / `volume_source_energy_kernel` (:1134) | 台帳集計 | 1 thread/group / cell + atomicAdd | ループ後 |

- **線形解法**: 群バッチ三重対角を **`cusparseDgtsv2StridedBatch`**（m=n_cells、batchCount=n_groups、batchStride=n_cells、:1544）で直接一括求解。CG/Thomas 自作は不使用。
- **多群**: host 群ループなし — 行列 layout は group-major（idx = g·n_cells + c）、cuSPARSE batch が群を畳み込む。
- **外反復シーケンス** (:1783): opacity/emission 評価（NLTE は `compute_nlte_coefficients_cuda_with_pe`）→ 組立 → gtsv2 → publish → `update_matter_kernel` → max-reduce → **収束判定の 2-double D2H（反復内唯一の host 同期点）**。
- **Fleck**: 生成先 `state.fld_nlte_f_work`（cg layout）、消費は組立の擬似散乱源 `f·η+(1-f)·c·σ_pa·E_old` と物質更新の 2 点。AFI モード（`fld.fleck_mode="afi"`）は両消費点を無効化。
- scratch pool tags: `"fld_1d_gpu:*"` / `"fld_1d:outer_check_pack"`。恒久場は `state.fld_*`。

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
`sn_transport_2d_gpu.cu`（2D ドライバ）+ `sn_transport_gpu.cu`（2D sweep エンジン兼 HOLO/QD-LO 1D）、
`sn_dsa_1d_gpu.cu`、`sn_material_newton_gpu.cu`、`sn_cyl_quadrature_1d.cpp`（host 求積）。
`sn_transport_1d.cpp` は CPU 参照実装（検証/フォールバック）。

#### 6.8.1 S_N 1D（エントリ `advance_radiation_step_sn_1d` :3650）

| kernel | 目的 | thread mapping |
|---|---|---|
| `build_sweep_inputs_kernel` (:664) | σ_t と IMEX 等方源（0.5η + 0.5σ_s·φ_old + 時間項） | 1 thread/(cell×group) |
| **`sn_sweep_spherical_serial_kernel`** (:691) | 球面 diamond sweep（既定） | **1 block=1 群、thread0 のみ** — 全角度×全セルを逐次。起動 `<<<n_groups,1,shared>>>`、shared=(n_cells+n_angles)·8B |
| **`sn_sweep_spherical_lc_kernel`** (:917) | linear-characteristic 版 | block=群、**32 lane warp が (cell,angle) 対角波面**。開始方向は Mark–Miller weighted-diamond seed |
| `sn_sweep_cylindrical_{serial,lc}_kernel` (:1179/:1417) | 円筒 product-quadrature 版 | 同構造（level 内 α=0 で chain 区切り、level ≤32） |
| `precompute_lc_weights_kernel` (:1666) | LC 指数閉包 θ(τ)=(1−A)/(τA)∈[0.5,1] | 1 thread/(cell×group×angle) |
| `sn_moments_reduction_k2_kernel` (:1855) / `sn_face_flux_reduction_k2_kernel` (:1889) | φ/F/Prr、面フラックス（donor 半レンジ）還元 | 1 thread/(cell or face ×group) |
| `phi_to_rad_E` :1930 / `sn_chi` :1939（Eddington χ=Prr/E clamp[0,1]）/ `apply_face_flux_boundary` :2065（真空=Milne 0.5cE、Marshak=離散 outgoing−S_neg） | 後処理 | per element |
| スイープ後連鎖 | E* flux :2101 → Fick 拡散面流束 :2139 → **AP ブレンド**（τ_lo=10/τ_hi=20 で SN↔拡散混合）:2182 → donor/θ リミッタ :2267/:2301 | per (cell/face×group) |

- **反復構造**: OUTER（opacity/emission 再評価 + Newton + Te 残差）× INNER source iteration（sweep→moments→**DSA**→残差 D2H、CUDA-graph unroll 対応 :3876）。
- **DSA 1D**（`sn_dsa_1d_gpu.cu`、球面専用）: 群内三重対角拡散補正 δφ を `assemble_sn_dsa_tridiag_kernel` :62（面 D 調和平均、外面 Milne β=0.5）+ **`cusparseDgtsv2StridedBatch`** :221 で直接解、`apply_sn_dsa_correction_kernel` :141 が φ に加算（clamp≥0）。source iteration 毎 1 回。
- **求積**: 球面/平面 = GL(n_angles) + Carlson α 漸化 + Morel&Montry weighted-diamond τ; 円筒 = `build_sn_cyl_quadrature_1d`（n_angles=2L²、level-major、per-level Carlson α、M&M A1-A4、開始方向 sd_μ=sinθ_l）。device 側 `SnQuadratureDeviceCache` :2774。
- **物質結合 Newton**（`sn_material_newton_gpu.cu` :281、1D/2D 共通）: **1 block=1 cell、128 thread が群を stride 分担**、energy-variable bracketed Newton（E*_override = スイープ後連鎖の E*）。床/ブラケット拡張は `atomicOr(retry_flag)` で global step 棄却を要求。
- **Marshak/Tr(t) 駆動**: 入口で host が `psi_in = 2·F_inc`（F_inc = ¼·c·a·T_r⁴·b_g、T_r は定数 or 凍結テーブル `state.marshak_Tr_1d`）を構築・アップロード (:3712-3773)。

#### 6.8.2 S_N 2D RZ（エントリ `advance_radiation_step_sn_2d_rz` :1443 → `solve_sn_transport_2d_rz_gpu`）

- **スイープ並列化**: `sn_sweep_2d_kernel`（sn_transport_gpu.cu :1193）は **grid=dim3(n_polar, n_groups)**（block=(polar level, 群)）、**方位半角 m は host 逐次ループ** (:2420)、block 内は **KBA 反対角波面**（stage=0..nr+nz−2、threads が反対角セルを分担、threads=clamp(128,[32,256])）。LC 版 `sn_sweep_2d_rz_lc_kernel` :1459 同構造。
- 還元: `sn_reduce_cell_outputs_2d_kernel` :1695 / `sn_reduce_face_flux_2d_kernel` :1731。
- **DSA 2D**: `sn_dsa_{setup,jacobi,apply}_2d_kernel`（:1759/:1777/:1920）— 5 点 RZ 拡散ステンシルを**固定 50 回の点 Jacobi 反復**で解く（1D の cuSPARSE 直接解と実装が根本的に異なる）。
- **Newton は coupling solve の外**でドライバが呼ぶ（1D は outer 内）— `update_material=false` で sweep し、スイープ後連鎖（拡散/AP/リミッタ/E*、各段後に `zero_axis_faces_checked_kernel` :308 で r=0 軸面ゼロ）→ Newton :1712。
- 境界: 真空/反射/Marshak（`kSNBoundary*`）、軸対称は軸面フラックスゼロ化。

#### 6.8.3 設計上の注意（host オーバーヘッド削減/perf 文脈）

1. **1D 既定 sweep は 1 群=1 thread の逐次**（並列度=群数のみ；LC のみ warp 波面）— 1D S_N の GPU 占有率は原理的に低く、host 律速（ホスト側オーバーヘッド削減系列で追跡）と併せて 1D が local-GPU tier に留まる一因。
2. **2D は (polar×群) block × KBA 波面**で並列度が立つ（方位 m と波面 stage は逐次）。
3. **DSA 実装の 1D/2D 非対称**（直接 tridiag vs 50 回 Jacobi）と **FLD の D 評価の 1D/2D 非対称**（face-centered vs cell-centered）は将来の統一候補として明示しておく。
