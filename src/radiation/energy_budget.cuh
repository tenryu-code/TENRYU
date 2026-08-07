#pragma once

#include "core/state.hpp"

namespace tenryu::radiation {

// Legacy radiation-only energy budget path.
// Deprecated: the active driver path uses diagnostics::compute_step_energy_budget
// (src/diagnostics/energy_budget.cu) for full coupled accounting.
struct EnergyBudget {
  double dE_int_e = 0.0;
  double dE_int_i = 0.0;
  double dE_kin = 0.0;
  double dE_rad = 0.0;
  double E_laser_in = 0.0;
  double E_laser_esc = 0.0;
  double E_marshak_in = 0.0;
  double E_rad_esc = 0.0;
  double E_pdV_boundary = 0.0;
  double E_floor = 0.0;
  double E_safety = 0.0;
  double E_solver = 0.0;
  double epsilon_budget = 0.0;
};

// Deprecated legacy helper retained for compatibility with older tooling.
EnergyBudget compute_energy_budget(const core::State& before,
                                   const core::State& after,
                                   double E_marshak_in,
                                   double E_rad_esc,
                                   double E_floor,
                                   double E_safety);

}  // namespace tenryu::radiation
