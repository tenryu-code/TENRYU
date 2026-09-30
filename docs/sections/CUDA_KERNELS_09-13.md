<!-- 分割元: docs/CUDA_KERNELS.md | このファイルは参照用です。原本（docs/CUDA_KERNELS.md）が権威です。 -->
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
- セルフィールド（125Kセル × 10フィールド × 8B = 10MB）→ 収まる（500×250メッシュ、PERFORMANCE P1-P3準拠）
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
> `retired/radiation_monte_carlo/`）。現行の性能実測は `PERFORMANCE.md`（host オーバーヘッド削減系列の wall/steps + host API 呼数）と
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

> Radiation が全体の ~80% を占める。KPI目標（PERFORMANCE.md 参照: ≥2×10⁹ events/s）に対し:
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
