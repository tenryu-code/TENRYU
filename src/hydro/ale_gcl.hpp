#pragma once

#include "core/state.hpp"

namespace tenryu::parallel {
class Reduction;
}  // namespace tenryu::parallel

namespace tenryu::hydro::ale {

struct AleVolClosureReference {
  double total_mass = 0.0;
  double total_internal_energy = 0.0;
  double total_r_momentum = 0.0;
  double total_z_momentum = 0.0;
  double Te = 0.0;
  double Ti = 0.0;
  double v_r = 0.0;
  double v_z = 0.0;
  int n_cells = 0;
  int n_nodes = 0;
  bool valid = false;
};

// ALE volume-closure residuals are total conservative-invariant drifts across
// rezone+remap. Density is intentionally not a per-cell invariant under ALE:
// mass is computed as rho*V, internal energy as (ee+ei)*mass, and momenta as
// cell-average velocity times mass. Te/Ti/velocity linf remain uniform-state
// diagnostics only.
struct AleVolClosureResidual {
  double total_mass_drift_rel = 0.0;
  double total_internal_energy_drift_rel = 0.0;
  double total_r_momentum_drift_rel = 0.0;
  double total_z_momentum_drift_rel = 0.0;
  double Te_linf = 0.0;
  double Ti_linf = 0.0;
  double velocity_linf = 0.0;
  bool valid = false;

  [[nodiscard]] double max_total_drift_rel() const noexcept {
    double value = total_mass_drift_rel;
    value = total_internal_energy_drift_rel > value
                ? total_internal_energy_drift_rel
                : value;
    value = total_r_momentum_drift_rel > value ? total_r_momentum_drift_rel : value;
    value = total_z_momentum_drift_rel > value ? total_z_momentum_drift_rel : value;
    return value;
  }

  [[nodiscard]] double max_linf() const noexcept {
    double value = Te_linf;
    value = Ti_linf > value ? Ti_linf : value;
    value = velocity_linf > value ? velocity_linf : value;
    return value;
  }
};

[[nodiscard]] AleVolClosureReference capture_ale_vol_closure_reference(
    const core::State& state);

[[nodiscard]] AleVolClosureResidual compute_ale_vol_closure_residual(
    const core::State& state,
    const AleVolClosureReference& reference,
    const parallel::Reduction* reduction = nullptr);

void capture_ale_vol_closure_reference_if_needed(const core::State& state);

[[nodiscard]] AleVolClosureResidual log_ale_vol_closure_residual(
    const core::State& state,
    const parallel::Reduction* reduction = nullptr);

void reset_ale_vol_closure_monitor();

}  // namespace tenryu::hydro::ale
