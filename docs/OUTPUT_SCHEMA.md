# OUTPUT_SCHEMA.md

この文書は TENRYU の HDF5 出力の実用的な参照です。  
**authoritative source は `docs/SPECIFICATION.md` §7.2 / §7.3 / §7.4** であり、ここは要約と確認手順を提供します。

## 1. 出力ファイル
- snapshot: `results/<case>_NNNN.h5`
- history: `results/<case>_history.h5`
- checkpoint: `checkpoints/<case>_ckpt_NNNN.h5`（MPI 実行でも rank 0 が 1 つのファイルに書く。旧形式のランク別ファイル `<case>_ckpt_NNNNNN_rNNNN.h5` もリスタートで読み込める）
- run 開始時の付帯ファイル: `run_info.json`、`config/<case>_frozen.json`、`mesh_requirement.json`（1D かつ Laser 有効のとき。物理由来の初期メッシュ分解能要求の見積もりと判定 — NUMERICS §3.1.0c、schema `tenryu.mesh_requirement.v1`: `applicable`/`reason`、`params`、`inputs`（波長・幾何・R0・面積・尖頭強度・fluence・アブレータの ρ_c）、`ablation`（アブレート面密度質量・質量割合・形成時刻・天井プロファイル）、`scale_length_track`、`shocks`、`layers`、`bands_recommended`、`requirement_check`（則ごとの違反数・最悪セル））

パスは `Output.directory` からの相対です（`run_info.json` と `mesh_requirement.json` は `Output.directory` 直下、`<case>_frozen.json` は `config/`）。`NNNN` はスナップショットとチェックポイントそれぞれの 4 桁ゼロ埋めの通し番号で（0 から数え、出力先に既存のファイルがあれば、再開かどうかによらずその最大番号の次から続ける）、cycle 番号ではありません。cycle 番号はルート属性 `cycle` にあります。停止理由は `run_info.json` の `termination_reason` に記録されます。

## 2. Snapshot (`<case>_NNNN.h5`)

主な構成:
- `/` attrs: `t`, `cycle`, `geometry`, `n_cells`, `n_nodes`, `n_groups`, `n_materials`, `schema_version`; 1D files also carry `geometry_1d` (`"spherical"` | `"cylindrical"` | `"planar"`, 2026-09-24): `geometry` is `Main.dimension`, which reads `1D_SPH` for every `Mesh.geometry_1d`. Readers that do not know the attribute are unaffected (additive, no `schema_version` change).
- `/metadata`: `namelist_source`, `frozen_config`, `group_bounds_eV`
- `/mesh`: `x_r`, `x_z(2Dのみ)`, `v_r`, `v_z(2Dのみ)`, `cell_material_id`; multiblock files additionally use `/mesh/topology/v2` for the 3-block scheme or `/mesh/topology/v3` for the half-butterfly 5-block scheme
- `/hydro`: `rho`, `Te`, `Ti`, `ee`, `ei`, `Pe`, `Pi`, `Qvisc`, `mass`, `vol`, `zbar`, `volFrac`; burn-enabled runs add `burn_rate`, `burn_Q_e`, `burn_Q_i`, `burn_eps_cum`, `burn_n_{D,T,He3,He4,p}` and, in 1D, `burn_neutron_cum` (cumulative number of neutrons born in the cell, DD and DT neutron branches, 2026-09-26 additive; the birth distribution of the yield, while `burn_eps_cum` follows where the charged products deposit)
- `/diagnostics/areal_density/v1` (areal density有効時): `angles_deg`, `rhoR`, optional `rhoR_hotspot_tracer`, reserved optional `rhoR_fuel_tracer`
- `/diagnostics/hotspot_gas/v1` (hotspot gas有効時): `hotspot_Te_*`, `hotspot_Ti_valid`, 2T-only `hotspot_Ti_*`, `hotspot_energy_*`, `hotspot_work_proxy_*`, `hotspot_work_definition`
- `/radiation`: `energy_density`, `rad_dep`, `rad_emit`, `deposited_power`, `diag_rad_E_pre`, `diag_rad_E_post`, `diag_rad_emission_at_Tn`, `diag_rad_emission_at_Tnp1`, `diag_rad_absorption`, `diag_clip_energy`, `diag_clip_full_deficit`, `diag_chi_opacity`, `diag_F_first_moment`, `diag_E_star_flux`, `diag_stream_theta`, `diag_ap_alpha_face`
- `schema_version` 2（2026-09-29）: モンテカルロ輻射（`Radiation.mode="imc_ddmc"`: IMC・DDMC・HOLO・difference 定式化）がビルドから外れ（`retired/radiation_monte_carlo/`）、その出力 `/radiation/ddmc_flag`、`/radiation/delta_E_rad_prev`、`/holo`、`/difference` は書かなくなった。いずれも FLD・S_N の run では書かれていなかった（HOLO・difference・DDMC が動いた run だけの出力）。schema 1 のファイルには残っていることがある。
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
- `/radiation/*`: `fld_outer_iterations`, `fld_outer_residual`, `fld_outer_converged`（FLD）、`sn_outer_iterations`, `sn_inner_iterations`, `sn_outer_residual`, `sn_converged`（S_N の run）、`overshoot_count`, `overshoot_max`（輻射相の最大値原理の超過: 相の後の電子温度が T_max = max(相の前の最大電子温度, Marshak 駆動温度) を超えたセル数と最大の (T_e − T_max)/T_max、NUMERICS §11.8。`Numerics.safety.overshoot_warn` はこの値で判定する）
- 2026-09-29（モンテカルロ輻射の退役）で `/mc/*`（粒子数・重み・DDMC の統計など 23 列）、`/holo/*`（19 列）、`/difference/*`（18 列）、`/diagnostics/dt_breakdown_history/dt_rad`（IMC 専用の Δt 制約。2026-09-15 から FLD・S_N では常に +∞）を書かなくなった。`dt_winner_code` の 2（`rad`）は欠番。`overshoot_count`・`overshoot_max` は以前は `/mc/` の下にあり、それより前に始めた history ファイルへ追記するときは `/mc/overshoot_count`・`/mc/overshoot_max` に書き続ける（列の長さを揃えるため）。`/mc/` の下にあった間は `Diagnostics.mc_stats.enabled=False` のデッキで 0 を書いていたが、`/radiation/` の 2 列は常に計測した値を書く（`Diagnostics.mc_stats` は効果を持たないキーになった）。
- `/laser/cbet_*`（CBET v1 診断スカラー、2026-07-07 追加 — additive・後方互換）:
  `cbet_exchanged_power_total` [erg/s]、`cbet_ledger_residual_rel` [-]、
  `cbet_iterations` [count]、`cbet_clamp_count` [count]。`Laser.cbet.enable=false`
  （既定）でも 0 値で出力される（列の有無は設定に依存しない）。
  2026-09-29 追加（additive・後方互換）: `cbet_converged` [flag]（最後の solve が収束したら 1、しなければ 0。
  CBET なしの step は 1）、`cbet_convergence_residual` [-]（最後の反復の収束指標）、`cbet_overflow_rays` [count]
  （レイごとの記録容量をあふれたレイの本数。あふれた後半は未吸収に数えて交換から外す、NUMERICS §5.10.6）。
  それ以前の history にはこの 3 列が無く、再開で追記した history では再開以降の行だけに入る。

## 4. Checkpoint (`<case>_ckpt_NNNN.h5`)

snapshot内容に加えて以下を保存:
- `/hydro_flags/hydro_active`
- `/time_state/*` (`t`, `step`, `dt`, 累積エネルギー項, `user_seed`)
- `/output_state/*` (`t_next_plot`, `t_next_history`, `t_next_checkpoint`)
- 1D: `/hydro/cv_e`, `/hydro/cv_i`, `/hydro/cs` (閉包の比熱と音速。あれば再開時に閉包をかけ直さない), `/hydro_flags/cell_is_void`
- `/conduction_state/*` (熱伝導ソルバーの run 累積統計)
- 1D S_N: `/radiation_sn/psi_prev`, `/radiation_sn/psi_sd_prev`, `/radiation_sn/ee_node_offset`
- 燃焼: `/burn_state/specific_inventory` (比インベントリ Y_s [1/g]、cell-major [n_cells×5]、D・T・He3・He4・p。再開はこれを `/hydro/burn_n_*` から作る Y_s より優先する)
- schema 1 のチェックポイントにあった `/particles/*`（モンテカルロ輻射の光子プール。FLD・S_N の run では 0 粒子）と `/rng/*` は schema 2 で無くなった。

## 5. frozen_config.json 形式

`frozen_config` は namelist 凍結JSON（`tenryu freeze` と同系）です。  
`metadata/frozen_config` には補助attrsを持ちます:
- `git_hash`（現在は常に `"unknown"`）
- `build_type`
- `cuda_arch`（現在は常に `"unknown"`）
- `gpu_name`
- `n_ranks`（現在は常に 1）
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

チェックポイントの S_N 角度強度の確認:
```bash
h5ls checkpoints/<case>_ckpt_0001.h5/radiation_sn
```

## 7. リスタート時の互換規約

- 現行は `schema_version` 2（2026-09-29）。
- `schema_version == current`: 通常読込
- `schema_version < current` または欠落: 後方互換読込（不足項目は既定値）
- `schema_version > current`: 読込拒否
- schema 1 のチェックポイント: `/particles` は空でなければならない（粒子を持つのは退役した `imc_ddmc` の run のもので、再開を拒否する）。`/holo`、`/difference`、`/radiation/ddmc_flag`、`/radiation/delta_E_rad_prev` は読まない。`metadata/frozen_config` の比較は両側から退役したキー（`Radiation.imc` の `two_stage` 以外、`Radiation.ddmc`・`.diffusion`・`.holo`・`.origin_parity_only`、`Radiation.boundary.marshak_particles`、材料の `opacity.lambda_method`・`f_min`、`Numerics.dt.f_min_fleck`、`Numerics.safety.opacity_floor`・`opacity_cap`、`Diagnostics.mc_stats`・`.fleck_diag`、`Parallel.migration`）を除いて行う。再開には従来どおり、そのチェックポイントを書いたデッキを内容を変えずに使う（`metadata/frozen_config` にはデッキのハッシュ `_namelist_source_hash` と `Output.directory` も入っていて比較される。デッキから退役したキーを消すと再開は拒否される）。
- 旧checkpointに `/radiation/rad_emit` がない場合はゼロで補完します。
- `/radiation/diag_*` は output-only diagnostics です。`diag_E_star_flux` は SN face-flux \(E^*\) 診断、`diag_stream_theta` は donor-theta streaming limiter 診断、`diag_ap_alpha_face` は AP face_blend の face weight 診断です。旧snapshot/checkpointで欠損してもrestartの物理状態復元には使いません。
- Topology compatibility is path-versioned: single-block files omit `/mesh/topology`, 3-block multiblock files use `/mesh/topology/v2`, and half-butterfly 5-block files use `/mesh/topology/v3`. Readers check v3 first, then v2, then the v1 single-block fallback.

凍結項目（dimension/mesh/materials/groups/seed/group_bounds）が不一致なら再開不可です。

## 8. Material-interface diagnostics group (additive)

The history file carries a path-versioned material-interface diagnostics group
added without bumping root `kSchemaVersion` (then 1; the root version is 2 since 2026-09-29, see §7).

History group:
- `/diagnostics/material_interface/v1/`
- Attributes: `plic_reconstruction_engine_version`,
  `plic_normal_estimator`, `t0_volume_cut_method`, `plic_enabled`,
  `plic_schema_version`, `plic_reconstruction_method`
- Per-sample datasets: `time_s`, `step`,
  `interface_cells_observed`, `interface_reconstruction_attempt_count`,
  `interface_reconstruction_success_count`, `plic_max_eta_E_observed`,
  `plic_max_volume_fraction_residual_observed`, and
  `plic_min_grad_F_observed`.
- Matrix dataset: `class_d_runtime_fires_matrix` with shape
  `[N,3,3]`, indexed by `[sample, case_id-1, severity]`.
- Event datasets under `/diagnostics/material_interface/v1/plic_events/`:
  `case_id`, `severity`, `cell_idx`, `i`, `j`, `eta_E`,
  `grad_F_magnitude`, `severity_metric_kind`, `severity_metric_value`,
  `fallback_used`, `prev_normal_invalidated`, `step`, and `time`.
- If more than 10000 new PLIC events are emitted in one history call, event
  rows are replaced for that call by aggregate rows under
  `/diagnostics/material_interface/v1/plic_events_summary/` with `case_id`,
  `severity`, `count`, `step`, and `time`.
- Final attributes: `final_class_d_aggregate` and
  `plic_remap_fallback_engaged`.
- Optional `/per_cell_state/` is written when
  `material_interface_per_cell_state` requests it.

Migration rules:
- A reader that predates the group ignores `/diagnostics/material_interface/v1/`.
- A reader that knows the group treats a missing group as PLIC disabled.
- With PLIC disabled the group is omitted, so the file reads like one written
  before the group existed.
- A reader that knows the group treats a `production_comparable` final claim
  without the material-interface group as inconsistent.
