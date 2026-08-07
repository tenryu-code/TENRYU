#include "diagnostics/operator_energy_residuals.hpp"

#include <algorithm>
#include <cmath>

#include "diagnostics/energy_budget.hpp"

namespace tenryu::diagnostics {

OperatorResidualSnapshot capture_total_energy(const core::State& state) {
  OperatorResidualSnapshot out{};
  EnergyTotals totals{};
  if (state.mesh.dim == 2) {
    totals = compute_energy_totals_2d(state);
  } else {
    totals = compute_energy_totals_1d(state);
  }
  out.E_thermal_total = totals.E_int_e + totals.E_int_i;
  out.E_kinetic_total = totals.E_kin;
  out.E_kinetic_total_nodal = totals.E_kin_nodal;
  out.timestamp = state.t;
  out.valid = std::isfinite(out.E_thermal_total) &&
              std::isfinite(out.E_kinetic_total);
  return out;
}

void OperatorEnergyTracker::record(
    const std::string& op_name,
    const OperatorResidualSnapshot& before,
    const OperatorResidualSnapshot& after,
    const double delta_E_ext,
    const int step,
    const double time) {
  if (!enabled_) {
    return;
  }

  OperatorResidualEntry entry{};
  entry.operator_name = op_name;
  entry.step = step;
  entry.time = time;
  entry.delta_E_ext = delta_E_ext;
  entry.valid = before.valid && after.valid && std::isfinite(delta_E_ext);
  entry.E_before = before.E_thermal_total + before.E_kinetic_total;
  entry.E_after = after.E_thermal_total + after.E_kinetic_total;

  if (entry.valid && std::isfinite(entry.E_before) &&
      std::isfinite(entry.E_after)) {
    const double denom = std::max({std::abs(entry.E_before),
                                   std::abs(entry.E_after),
                                   std::abs(delta_E_ext),
                                   1.0});
    entry.residual =
        (entry.E_after - entry.E_before - delta_E_ext) / denom;
    entry.valid = std::isfinite(entry.residual);
  }

  entries_.push_back(entry);
}

}  // namespace tenryu::diagnostics
