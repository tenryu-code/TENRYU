# OUTPUT_SCHEMA.md

この文書は TENRYU の HDF5 出力の実用的な参照です。  
**authoritative source は `docs/SPECIFICATION.md` §7.2 / §7.3 / §7.4** であり、ここは要約と確認手順を提供します。

## 1. 出力ファイル
- snapshot: `<case>_NNNNNN.h5`
- history: `<case>_history.h5`
- checkpoint: `<case>_ckpt_NNNNNN_rNNNN.h5`
- run 開始時の付帯ファイル: `run_info.json`、`config/<case>_frozen.json`、`mesh_requirement.json`（1D かつ Laser 有効のとき。物理由来の初期メッシュ分解能要求の見積もりと判定 — NUMERICS §3.1.0c、schema `tenryu.mesh_requirement.v1`: `applicable`/`reason`、`params`、`inputs`（波長・幾何・R0・面積・尖頭強度・fluence・アブレータの ρ_c）、`ablation`（アブレート面密度質量・質量割合・形成時刻・天井プロファイル）、`scale_length_track`、`shocks`、`layers`、`bands_recommended`、`requirement_check`（則ごとの違反数・最悪セル））

`NNNNNN` は cycle 番号（6桁ゼロ埋め）です。

## 2. Snapshot (`<case>_NNNNNN.h5`)

主な構成:
- `/` attrs: `t`, `cycle`, `geometry`, `n_cells`, `n_nodes`, `n_groups`, `n_materials`, `schema_version`; 1D files also carry `geometry_1d` (`"spherical"` | `"cylindrical"` | `"planar"`, 2026-09-24): `geometry` is `Main.dimension`, which reads `1D_SPH` for every `Mesh.geometry_1d`. Readers that do not know the attribute are unaffected (additive, no `schema_version` change).
- `/metadata`: `namelist_source`, `frozen_config`, `group_bounds_eV`
- `/mesh`: `x_r`, `x_z(2Dのみ)`, `v_r`, `v_z(2Dのみ)`, `cell_material_id`; multiblock files additionally use `/mesh/topology/v2` for the 3-block scheme or `/mesh/topology/v3` for the half-butterfly 5-block scheme
- `/hydro`: `rho`, `Te`, `Ti`, `ee`, `ei`, `Pe`, `Pi`, `Qvisc`, `mass`, `vol`, `zbar`, `volFrac`; burn-enabled runs add `burn_rate`, `burn_Q_e`, `burn_Q_i`, `burn_eps_cum`, `burn_n_{D,T,He3,He4,p}` and, in 1D, `burn_neutron_cum` (cumulative number of neutrons born in the cell, DD and DT neutron branches, 2026-09-26 additive; the birth distribution of the yield, while `burn_eps_cum` follows where the charged products deposit)
- `/diagnostics/areal_density/v1` (areal density有効時): `angles_deg`, `rhoR`, optional `rhoR_hotspot_tracer`, reserved optional `rhoR_fuel_tracer`
- `/diagnostics/hotspot_gas/v1` (hotspot gas有効時): `hotspot_Te_*`, `hotspot_Ti_valid`, 2T-only `hotspot_Ti_*`, `hotspot_energy_*`, `hotspot_work_proxy_*`, `hotspot_work_definition`
- `/radiation`: `energy_density`, `rad_dep`, `rad_emit`, `deposited_power`, `diag_rad_E_pre`, `diag_rad_E_post`, `diag_rad_emission_at_Tn`, `diag_rad_emission_at_Tnp1`, `diag_rad_absorption`, `diag_clip_energy`, `diag_clip_full_deficit`, `diag_chi_opacity`, `diag_F_first_moment`, `diag_E_star_flux`, `diag_stream_theta`, `diag_ap_alpha_face`, `ddmc_flag`（退役 imc_ddmc 経路のみ有意）, `boundary_flux`, `momentum_dep`
- `/holo` (HOLO selector有効時 — 退役 imc_ddmc 系経路): `E_LO`, `consistency_source`, `rad_dep_LO`, `rad_emit_LO`, `Prr_HO`, `chi`, `Prr_coverage`, `core_mask`, `prev_core_mask`, `hold_count`, `dwell_count`, `tau_R`, `reduced_flux`, `mass_q`
- `/difference` (difference有効時 — 退役 imc_ddmc 系経路): `W`, `E_ref`, `residual_energy_density`
- `/laser` (laser有効時):
  - `deposited_power`, `absorption_fraction`
  - `ray_density`（レイ初期配置密度: セルに配置されたレイ数 / セル体積）
  - `mesh/`（`ray_output_trajectory=True` かつ非skipステップ時）:
    - `n_nodes_r`, `n_nodes_z`, `n_crit`
    - `node_R`, `node_Z`
    - `n_e_hat`, `T_e`, `Zbar`
    - `grad_n_hat_R`, `grad_n_hat_Z`
  - `rays/`（`ray_output_count > 0` かつ非skipステップ時）:
    - `n_rays`, `beam_id`, `power0`
    - 1D_SPH: `R0`, `Z0`, `vR0`, `vZ0`
    - 2D_RZ: `x0`, `y0`, `z0`, `vx0`, `vy0`, `vz0`
    - `trajectory/`（`ray_output_trajectory=True` かつ非skipステップ時）:
      - `n_rays`, `offsets`, `step_count`, `beam_id`
      - `pos_R/pos_x`, `pos_Z/pos_y`, `pos_z`（3Dのみ）, `power`

全数値データセットに `units` 属性を付与します。

## 3. History (`<case>_history.h5`)

時系列（append）形式で、以下を追記:
- `/t`, `/cycle`, `/dt`
- `/energy/*`
- `/implosion/*` (`rho_R`, optional `rho_R_hotspot_tracer`, per-angle scalar aliases)
- `/diagnostics/hotspot_gas/v1/*` (hotspot gas有効時; same stagnation Te/Ti and work-proxy fields as snapshot, plus existing hotspot compression summaries)
- `/modes/*` (2D)
- `/mc/*`
- `/holo/*` (HOLO diagnostics: `n_core_cells`, `E_LO_total`, LO boundary/source balance terms, `particle_net_source_core`, `lo_particle_source_mismatch`, `Prr_coverage`, `chi_min`, `chi_mean`, `chi_max`)
- `/laser/cbet_*`（CBET v1 診断スカラー、2026-07-07 追加 — additive・後方互換）:
  `cbet_exchanged_power_total` [erg/s]、`cbet_ledger_residual_rel` [-]、
  `cbet_iterations` [count]、`cbet_clamp_count` [count]。`Laser.cbet.enable=false`
  （既定）でも 0 値で出力される（列の有無は設定に依存しない）。

## 4. Checkpoint (`<case>_ckpt_NNNNNN_rNNNN.h5`)

snapshot内容に加えて以下を保存:
- `/hydro_flags/hydro_active`
- `/particles/*` (PhotonPool SoA)
- `/rng/*` (`rng_counter`, `global_id`)
- `/time_state/*` (`t`, `step`, `dt`, 累積エネルギー項, `user_seed`)
- `/output_state/*` (`t_next_plot`, `t_next_history`, `t_next_checkpoint`)
- 1D: `/hydro/cv_e`, `/hydro/cv_i`, `/hydro/cs` (閉包の比熱と音速。あれば再開時に閉包をかけ直さない), `/hydro_flags/cell_is_void`
- `/conduction_state/*` (熱伝導ソルバーの run 累積統計)
- 1D S_N: `/radiation_sn/psi_prev`, `/radiation_sn/psi_sd_prev`, `/radiation_sn/ee_node_offset`
- 燃焼: `/burn_state/specific_inventory` (比インベントリ Y_s [1/g]、cell-major [n_cells×5]、D・T・He3・He4・p。再開はこれを `/hydro/burn_n_*` から作る Y_s より優先する)

## 5. frozen_config.json 形式

`frozen_config` は namelist 凍結JSON（`tenryu freeze` と同系）です。  
`metadata/frozen_config` には補助attrsを持ちます:
- `git_hash`
- `build_type`
- `cuda_arch`
- `gpu_name`
- `n_ranks`
- `rng_seed`

## 6. CLI確認例

ヘッダ構造:
```bash
h5dump -H <file>.h5
```

グループ一覧:
```bash
h5ls <file>.h5
```

チェックポイントの粒子配列確認:
```bash
h5ls <case>_ckpt_000100_r0000.h5/particles
```

## 7. リスタート時の互換規約

- `schema_version == current`: 通常読込
- `schema_version < current` または欠落: 後方互換読込（不足項目は既定値）
- `schema_version > current`: 読込拒否
- 旧checkpointに `/holo/E_LO`、`/holo/consistency_source` がない場合はゼロで補完します。旧 `/holo/gamma` は same-step HOLO では restart 入力に使わず、consistency source は次 step 内で再生成します。
- 旧checkpointに `/radiation/rad_emit`、`/holo/rad_dep_LO`、`/holo/rad_emit_LO`、
  `/holo/Prr_HO`、`/holo/chi`、`/holo/Prr_coverage`
  がない場合はゼロで補完します。
- `/radiation/diag_*` は output-only diagnostics です。`diag_E_star_flux` は SN face-flux \(E^*\) 診断、`diag_stream_theta` は donor-theta streaming limiter 診断、`diag_ap_alpha_face` は AP face_blend の face weight 診断です。旧snapshot/checkpointで欠損してもrestartの物理状態復元には使いません。
- Topology compatibility is path-versioned: single-block files omit `/mesh/topology`, 3-block multiblock files use `/mesh/topology/v2`, and half-butterfly 5-block files use `/mesh/topology/v3`. Readers check v3 first, then v2, then the v1 single-block fallback.

凍結項目（dimension/mesh/materials/groups/seed/group_bounds）が不一致なら再開不可です。

## 8. V21 schema (Stage 30 additive)

Stage 30 adds a path-versioned material-interface diagnostics group without
bumping root `kSchemaVersion`; it remains `schema_version = 1`.

New history group:
- `/diagnostics/material_interface/v1/`
- Wave A attributes: `plic_reconstruction_engine_version`,
  `plic_normal_estimator`, `t0_volume_cut_method`, `plic_enabled`,
  `plic_schema_version`, `plic_reconstruction_method`
- Wave D per-sample datasets: `time_s`, `step`,
  `interface_cells_observed`, `interface_reconstruction_attempt_count`,
  `interface_reconstruction_success_count`, `plic_max_eta_E_observed`,
  `plic_max_volume_fraction_residual_observed`, and
  `plic_min_grad_F_observed`.
- Wave D matrix dataset: `class_d_runtime_fires_matrix` with shape
  `[N,3,3]`, indexed by `[sample, case_id-1, severity]`.
- Wave D event datasets under `/diagnostics/material_interface/v1/plic_events/`:
  `case_id`, `severity`, `cell_idx`, `i`, `j`, `eta_E`,
  `grad_F_magnitude`, `severity_metric_kind`, `severity_metric_value`,
  `fallback_used`, `prev_normal_invalidated`, `step`, and `time`.
- If more than 10000 new PLIC events are emitted in one history call, event
  rows are replaced for that call by aggregate rows under
  `/diagnostics/material_interface/v1/plic_events_summary/` with `case_id`,
  `severity`, `count`, `step`, and `time`.
- Wave D final attributes: `final_class_d_aggregate` and
  `plic_remap_fallback_engaged`.
- Optional `/per_cell_state/` is written when
  `material_interface_per_cell_state` requests it.

Migration rules:
- V20 reader handles V21 file: ignore `/diagnostics/material_interface/v1/`.
- V21 reader handles V20 file: missing group means PLIC disabled.
- V21 reader handles V21 file with PLIC disabled: group is omitted and treated
  the same as a V20 file.
- V21 reader handles a `production_comparable` final claim with missing
  material-interface group as inconsistent.
