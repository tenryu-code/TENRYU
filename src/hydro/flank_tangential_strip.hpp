#pragma once

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro {

double flank_strip_corner_angle_quality(const double* r,
                                        const double* z,
                                        int corner);

void flank_strip_capture_reference(core::State& state,
                                   const core::Config& cfg);

struct FlankStripEvaluation {
  bool armed_now = false;
  bool would_fire = false;
  double worst_ratio = 1.0;
  int worst_cell = -1;
  int worst_corner = -1;
  double n_sliver_pred = 1.0e300;
};

FlankStripEvaluation flank_strip_update_and_evaluate(
    core::State& state, const core::Config& cfg, double dt_step);

struct FlankStripRezoneResult {
  bool triggered = false;
  bool converged = false;
  double q_before = 0.0;
  double q_after = 0.0;
  double max_displacement = 0.0;
  double slip_mismatch_max_ratio = 0.0;
  double slip_mismatch_signed_ratio = 0.0;
  int moved_nodes_outside_band = 0;
  int alpha_halvings = 0;
  int used_untangler = 0;
  int target_cell = -1;
  int band_argmin_cell_before = -1;
  int band_argmin_cell_after = -1;
  double target_corner_j_before[4] = {0.0, 0.0, 0.0, 0.0};
  double target_corner_j_after[4] = {0.0, 0.0, 0.0, 0.0};
};

FlankStripRezoneResult run_flank_tangential_strip_rezone(
    core::State& state, const core::Config& cfg, int focus_cell = -1);

}  // namespace tenryu::hydro
