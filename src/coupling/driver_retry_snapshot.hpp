#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "coupling/hydro_step_result.hpp"

namespace tenryu::core {
struct State;
struct Config;
}  // namespace tenryu::core

namespace tenryu::coupling {

// Bit-exact rollback container for tenryu::core::State, scoped to deterministic
// radiation/conduction modes (FLD / SN / HOLO). This intentionally excludes the
// IMC particle pool and RNG counters; Commit 2 will reject IMC + retry.
struct DriverRetrySnapshot {
  ~DriverRetrySnapshot();

  bool valid = false;
  std::size_t n_cells = 0;
  std::size_t n_nodes = 0;
  std::size_t n_groups = 0;

  // Device-resident snapshot arena (2026-07-31 perf lane C): capture and
  // restore are device-side table copies; the host never stages the data.
  void* d_arena = nullptr;
  std::size_t arena_bytes = 0;
  struct Entry {
    const char* name;
    void* field_ptr;
    std::size_t offset;
    std::size_t bytes;
  };
  std::vector<Entry> entries;
  void* d_table = nullptr;
  std::size_t table_capacity = 0;
  bool table_dirty = true;

  // Burn host-primary state (mutated by callbacks.burn before the hydro
  // halves under STRANG; must roll back with the step).
  std::vector<double> burn_n_host, burn_rate_host, burn_Q_e_host;
  std::vector<double> burn_Q_i_host, burn_eps_cum_host;
  double E_burn_released = 0.0;
  double E_burn_dep_e = 0.0;
  double E_burn_dep_i = 0.0;
  double E_burn_esc_charged = 0.0;
  double E_burn_esc_neutron = 0.0;
  double E_burn_inflight = 0.0;
  double N_burn_neutrons_dt = 0.0;
  double N_burn_neutrons_dd = 0.0;

  // MC alpha-transport pool (device particle state mutated by
  // callbacks.burn under scheme="mc"; must roll back with the step;
  // Philox streams replay deterministically, so no RNG state is needed).
  int burn_mc_live = 0;

  // Hot-electron preheat host-primary state (mutated by callbacks.laser
  // inside the retried region; the cumulative specific-preheat accumulator
  // must roll back with the step — burn_eps_cum precedent).
  std::vector<double> hot_e_eps_cum_host;
  double E_hot_e_deposited = 0.0;
  double E_hot_e_escaped = 0.0;
  bool hot_e_enabled_any = false;

  // HOLO host vectors and DDMC mode map.
  std::vector<std::int8_t> ddmc_mode_map;
  std::vector<std::uint8_t> holo_core_mask, holo_patch_mask;
  std::vector<std::uint8_t> holo_core_prev_mask;
  std::vector<std::int32_t> holo_hold_count, holo_dwell_count;
  std::vector<double> holo_tau_R, holo_reduced_flux, holo_mass_q, holo_lo_weight;
  std::vector<std::int8_t> hydro_active;
  std::vector<std::int8_t> state_supply_mask;
  std::vector<std::uint8_t> cell_is_void;

  // Cumulative + adaptive scalars.
  double E_safety = 0.0, E_numerical_loss = 0.0;
  double E_laser_deposited = 0.0, E_laser_escaped = 0.0;
  double E_laser_incident = 0.0;
  double E_ra_deposited = 0.0;
  double E_cbet_iaw_step = 0.0, E_cbet_iaw = 0.0;
  double E_rad_escaped = 0.0, E_floor_injected = 0.0;
  double E_pdV_bdry = 0.0, E_Marshak_in = 0.0, E_solver = 0.0;
  double state_supply_dM_cumulative = 0.0, state_supply_dE_cumulative = 0.0;
  double state_supply_dPz_cumulative = 0.0;
  double state_supply_dM_step = 0.0, state_supply_dE_step = 0.0;
  double state_supply_dPz_step = 0.0;
  double fld_state_supply_in_cumulative = 0.0;
  double fld_state_supply_out_cumulative = 0.0;
  double fld_state_supply_net_cumulative = 0.0;
  double fld_state_supply_in_step = 0.0;
  double fld_state_supply_out_step = 0.0;
  double fld_state_supply_net_step = 0.0;
  double t = 0.0, dt = 0.0, dt_prev_hydro = -1.0;
  int step = 0;
  bool corner_mass_initialized = false;
  bool corner_mass_is_lagrangian_invariant = false;
  bool gas_tracer_initialized = false;
  bool hllc_mom_z_cell_initialized = false;
  bool ale_rezoned = false;
  int ale_rezone_invocations = 0, ale_remaps_applied = 0;
  int ale_last_applied_step = -1;
  bool plic_remap_sticky_fallback = false;
  int plic_consecutive_drift_triggers = 0;
  int plic_last_reconstruction_step = -1;
  double axis_margin_initial = -1.0;
  double hydro_t_start_eV = 0.0;
  double adaptive_av_r0 = 0.0, adaptive_av_last_rs = 0.0;
  double adaptive_av_last_us = 0.0, adaptive_av_rs_min = 0.0;
  int adaptive_av_tracker_steps = 0, adaptive_av_mode = 0;
  bool adaptive_av_tracker_valid = false, adaptive_av_bounce_seen = false;

  // Axis budget host vectors.
  std::vector<double> axis_mass_initial, axis_inflow_budget;

  // HOLO valid flags + DDMC valid + ALE flags.
  bool ddmc_mode_map_valid = false;
  bool holo_core_mask_valid = false;
  bool holo_lo_source_valid = false;
  bool holo_ale_invalidated = false;
  bool particle_sort_cache_invalidated = false;

  // _step accumulators (FLD diagnostics).
  int fld_clamp_hits_step = 0;
  double fld_clamp_energy_delta_step = 0.0;
  double fld_min_x_raw_step = 0.0;
  double fld_cg_true_residual_l2_rel_step = 0.0;
  double fld_cg_true_residual_max_step = 0.0;
  double fld_E_solver_step = 0.0;
  double fld_cg_true_residual_l2_rel_RAW_step = 0.0;
  double fld_cg_true_residual_max_RAW_step = 0.0;
  double fld_E_solver_RAW_step = 0.0;
  double fld_csr_diag_min_pos_step = 0.0;
  double fld_csr_diag_max_step = 0.0;
  int fld_csr_weak_diag_dom_count_step = 0;
  int fld_csr_nonfinite_count_step = 0;
  double fld_gershgorin_lower_min_step = 0.0;
  double fld_gershgorin_upper_max_step = 0.0;
  double fld_cg_pAp_min_step = 0.0;
  int fld_cg_nonpos_pAp_count_step = 0;
  int fld_cg_nonfinite_count_step = 0;
  double fld_cg_recurrent_resid_last_check_max_step = 0.0;
  double fld_Dcell_min_step = 0.0;
  double fld_Dcell_max_step = 0.0;
  int fld_Dcell_zero_count_step = 0;
  int fld_Dcell_nonfinite_count_step = 0;
  int fld_face_skip_D_count_step = 0;
  int fld_face_skip_nonfinite_count_step = 0;
  int fld_face_skip_dist_count_step = 0;
  int fld_diag_fallback_count_step = 0;
  double fld_pair_symmetry_max_diff_step = 0.0;
  int fld_pair_symmetry_violation_count_step = 0;
  int fld_newton_converged_count_step = 0;
  int fld_newton_cap_hit_count_step = 0;
  int fld_newton_invalid_count_step = 0;
  double fld_newton_resid_abs_max_step = 0.0;
  double fld_newton_resid_rel_max_step = 0.0;
  int fld_newton_reject_count_step = 0;
  double fld_newton_reject_resid_rel_max_step = 0.0;
  double fld_rogue_max_abs_x_raw_step = 0.0;
  int fld_rogue_max_abs_x_raw_cell_step = -1;
  int fld_rogue_max_abs_x_raw_group_step = -1;
  double fld_rogue_min_x_raw_step = 0.0;
  int fld_rogue_min_x_raw_cell_step = -1;
  int fld_rogue_min_x_raw_group_step = -1;
  double fld_rogue_max_rad_E_step = 0.0;
  int fld_rogue_max_rad_E_cell_step = -1;
  int fld_rogue_max_rad_E_group_step = -1;
  double fld_rogue_max_r_true_step = 0.0;
  int fld_rogue_max_r_true_cell_step = -1;
  int fld_rogue_max_r_true_group_step = -1;
  double fld_rogue_max_E_solver_row_step = 0.0;
  int fld_rogue_max_E_solver_row_cell_step = -1;
  int fld_rogue_max_E_solver_row_group_step = -1;
  double fld_escaped_step = 0.0, fld_marshak_in_step = 0.0;

  // FLD outer iteration state.
  int fld_outer_iterations = 0;
  bool fld_converged = false;
  double fld_outer_residual = 0.0;

  // _step accumulators (SN).
  int sn_outer_iterations = 0, sn_inner_iterations = 0;
  bool sn_converged = false, sn_outer_stagnated = false;
  double sn_outer_residual = 0.0, sn_inner_residual = 0.0;
  double sn_escaped_step = 0.0, sn_marshak_in_step = 0.0;
  double sn_void_anchor_dE_step = 0.0, sn_void_anchor_dE_abs_step = 0.0;
};

// Test/diagnostic access to the device-resident snapshot payloads.
struct SnapshotEntryView {
  const char* name;
  std::size_t bytes;
  std::vector<unsigned char> host_copy;
};

std::vector<SnapshotEntryView> driver_retry_snapshot_readback(
    const DriverRetrySnapshot& snap);
bool driver_retry_snapshot_entry_readback(
    const DriverRetrySnapshot& snap,
    const char* name,
    std::vector<unsigned char>& out_bytes);

void capture_driver_retry_snapshot(DriverRetrySnapshot& snap,
                                   const tenryu::core::State& state,
                                   const tenryu::core::Config& cfg,
                                   void* cuda_stream = nullptr);

void restore_driver_retry_snapshot(tenryu::core::State& state,
                                   const tenryu::core::Config& cfg,
                                   DriverRetrySnapshot& snap,
                                   bool restore_transient_topology_masks = false,
                                   bool invalidate_particle_sort_cache = true,
                                   void* cuda_stream = nullptr);

}  // namespace tenryu::coupling
