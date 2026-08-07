#pragma once

#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::button_morph {

// C2 quintic smoothstep schedule: 0 for t <= t_start, 1 for t >= t_end,
// s(u) = 6u^5 - 15u^4 + 10u^3 with u = (t - t_start)/(t_end - t_start) between.
// Asserts t_end > t_start.
double button_morph_alpha(double t, double t_start, double t_end);

// Build absolute per-node morph targets for blend alpha in [0,1] on the
// MULTIBLOCK_CART_CORE_POLAR_SHELL topology. Source frame x_init is the
// state's reference mesh (x_r_reference/x_z_reference = the initial mesh;
// the caller must not use this with reference-mutating ALE modes). Targets:
//   - shell nodes (including the seam ring at r_match): target == current
//     (bitwise copy).
//   - core nodes (i,j): ring a = max(r_init, |z_init|); the aligned position
//     is the Shirley-Chiu equal-volume image from src/mesh/button_align_target.hpp
//     (radius button_align_radius(a), angle from the perimeter fraction of the
//     init position); axis nodes must land bitwise on r = 0; the center node
//     maps to the origin.
//   - bridge interior nodes (layer l in (0, n_b), k): the S-B bridge target
//     with mu = l / n_b between the aligned core boundary circle
//     (S_c = button_align_radius(r_c)) and the seam circle (s_b = r_match).
// Core and bridge-interior target = init + s_alpha * (aligned - init), where
// s_alpha is the caller-supplied blend (already scheduled); alpha = 0 must
// reproduce current bitwise for every node (enforce by early-out copy, not by
// trusting arithmetic).
// current_r/current_z: the CURRENT node coordinates (host, size n_nodes). Nodes outside the
// morph mandate — every node that is not a core node or a bridge-interior node (i.e. the
// whole shell family, including the seam ring) — receive target == current (bitwise), so the
// barrier transaction cannot move them. Core and bridge-interior targets remain anchored to
// the init/reference frame (the interior is static in the intended pre-shock window).
// tr/tz are resized to n_nodes.
void build_button_morph_targets(const tenryu::core::State& state,
                                const tenryu::core::Config& cfg,
                                double alpha,
                                const std::vector<double>& current_r,
                                const std::vector<double>& current_z,
                                std::vector<double>& tr,
                                std::vector<double>& tz);

struct ButtonMorphStepResult {
  bool attempted = false;
  bool applied = false;
  double alpha = 0.0;
  double lambda_cap = 1.0;       // uniform pre-scale from max_step_fraction
  double lambda_accepted = 0.0;  // from the barrier transaction
  double max_step_over_h = 0.0;  // pre-cap max |dx|/h_local
  double residual_max_over_h = 0.0;  // post-transaction max |target-x|/h_local
};

// Runs the morph transaction when due: enabled, topology is
// MULTIBLOCK_CART_CORE_POLAR_SHELL, t inside [t_start_s, t_end_s], and
// step % every_n_steps == 0. Computes alpha from the *end-of-step* time
// passed by the caller, builds capped targets, and applies them through
// apply_reference_barrier_rezone (transactional rezone + conservative CSR
// remap + quality line-search). Logs one "[button_morph]" info line per
// attempt. Returns attempted=false when not due.
ButtonMorphStepResult apply_button_morph_if_due(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    double t_end_of_step,
    int step);

}  // namespace tenryu::hydro::button_morph
