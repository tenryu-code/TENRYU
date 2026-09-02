#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace tenryu::core {
struct State;
struct Config;
}

namespace tenryu::hydro {

struct AutopilotFireRecord {
  bool pending = false;
  long long step = 0;
  double t = 0.0;
  double s50_cm = 0.0;
  double s_lead_cm = 0.0;
  double d_clear_h = 0.0;
  double a_old = 0.0;
  double dt_arrival_early_s = 0.0;
  double q99_e = 0.0;
};

// §18.1 W3a: per-column front tracker + autopilot decision engine.
// Host-side observer: consumes state.refine_error (already
// filled by refinement_estimator_step this step), stitches per-angular-
// column radial profiles across structured blocks, locates the
// innermost strong feature per column via the physical pressure ridge,
// aggregates robust front radii, runs the SEARCH/TRACKED/ARMED state
// machine, and logs [refine_trk] / [refine_apilot] lines. Shadow mode
// never modifies state; arm_exit may request a same-step checkpoint.
// No-op unless refinement_estimator ran this step and
// refinement_autopilot.enabled.
void refinement_tracker_step(core::State& state, const core::Config& cfg);

// arm_exit: pending WOULD_FIRE record for the driver to consume
// (nullptr when none). The record refers to the CURRENT step's
// state; the checkpoint written at this same step is the commit
// state.
const AutopilotFireRecord* refinement_tracker_pending_fire();
void refinement_tracker_clear_fire();

// §18.5 v5: most recent tracked front for the estimator maintenance band's
// front hold. Returns true ONLY when the last refinement_tracker_step has a
// TRACKED or ARMED front whose median location lies inside the POLAR_SHELL
// block; outputs that median front expressed as a shell-block radial row
// index (i-index of the shell block, ascending outward) and the column
// coverage. SEARCH state, or a front that has left the shell block inward,
// returns false.
bool refinement_tracker_front_shell_row(int* shell_row, double* coverage);
// test seam: override/clear the reported front (test-only).
void refinement_tracker_set_front_for_test(int shell_row, double coverage);
void refinement_tracker_clear_front_for_test();

// --- pure test seams ---
// arm_exit checkpoint-request predicate (pure).
bool tracker_should_request_checkpoint(double d_clear_h,
                                       double window_hi_h,
                                       double ckpt_lead_h);

// Connected components of flag[i] = (e[i] >= cut) with gaps of up to
// gap_bridge unflagged cells bridged; returns (begin,end) inclusive
// index pairs in order.
std::vector<std::pair<int, int>> tracker_components(
    const std::vector<double>& e, double cut, int gap_bridge);

// Least-squares fit s(t) over the ring buffer (t centered internally):
// linear [a0,a1] and quadratic [b0,b1,b2] coefficients about t_ref =
// last observation time; outputs per-model rms residual. Deterministic
// closed-form normal equations. Returns false if n < 3 or the system
// is singular.
bool tracker_fit_models(const std::vector<double>& t,
                        const std::vector<double>& s,
                        double t_ref,
                        double linear_out[2],
                        double& linear_rms,
                        double quad_out[3],
                        double& quad_rms);

// Earliest arrival time (relative to t_ref) at radius s_target for the
// fitted models, each reduced by 3*sigma_t where sigma_t =
// rms / max(|ds/dt at t_ref|, floor). Inward motion only: returns
// false if neither model predicts reaching s_target with ds/dt < 0.
bool tracker_earliest_arrival(const double linear_coeffs[2],
                              double linear_rms,
                              const double quad_coeffs[3],
                              double quad_rms,
                              double s_target,
                              double& dt_arrival_early);

}  // namespace tenryu::hydro
