#pragma once

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro {

struct AxisEdgeCollapsePredicateResult {
  bool eligible = false;
  const char* ineligible_reason = "";
  bool would_fire = false;      // full predicate
  bool terminal_waiver = false; // terminal-retry waiver path
  double h0 = 0.0;
  double hdot = 0.0;
  double h_retire = 0.0;
  double h_min_pred = 0.0;      // predicted turning gap; -inf when not decelerating
  int closing_count = 0;        // closing samples in the window
  int window = 0;               // valid samples
};

struct AxisEdgeCollapseTransactionResult {
  bool attempted = false;
  bool committed = false;
  const char* reject_reason = "";
  double q_node = 0.0;          // nodal-merge kinetic-energy loss [erg]
  double tri_volume = 0.0;      // post-collapse RZ volume of the triangle [cc]
  double min_offaxis_corner_j = 0.0;
  double mass_residual = 0.0;
  double mom_r_residual = 0.0;
  double mom_z_residual = 0.0;
  double energy_residual = 0.0;
};

// Update the monitor with one sample (call at most once per step) and
// evaluate the collapse predicate for the failing axis cell.
AxisEdgeCollapsePredicateResult axis_edge_collapse_update_and_evaluate(
    core::State& state,
    const core::Config& cfg,
    int failing_cell,          // flat cell id, must be an axis-row cell (i==0)
    double retry_dt_next,      // the dt the retry ladder is about to use [s]
    double dt_min_s);          // Numerics.dt.min_s

void rebuild_axis_edge_collapse_device_state(core::State& state);
void rebuild_geometry_policy_exempt(core::State& state);

AxisEdgeCollapseTransactionResult axis_edge_collapse_execute(
    core::State& state, const core::Config& cfg, int failing_cell);

}  // namespace tenryu::hydro
