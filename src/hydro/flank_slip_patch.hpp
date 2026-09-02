#pragma once

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro {

struct FlankSlipPatchResult {
  bool attempted = false;
  bool applied = false;
  const char* reject_reason = "";
  int reject_cell = -1;
  int reject_node = -1;
  const char* reject_check = "";
  double reject_quad_r[4] = {0, 0, 0, 0};
  double reject_quad_z[4] = {0, 0, 0, 0};
  int direction = 0;
  double q_before = 0.0;
  double q_after = 0.0;
  double slip_before = 0.0;
  double slip_after = 0.0;
  double mass_defect_rel = 0.0;
  double energy_defect_rel = 0.0;
  double momentum_defect_rel = 0.0;
  double partition_row_defect = 0.0;
  double partition_col_defect = 0.0;
};

// Host-side atomic flank rezone/remap transaction. The caller is responsible
// for refreshing auxiliary device geometry/material mirrors and EOS closure
// before continuing the hydro driver.
FlankSlipPatchResult flank_slip_patch_execute(
    core::State& state,
    const core::Config& cfg,
    int target_cell,
    int direction);

}  // namespace tenryu::hydro
