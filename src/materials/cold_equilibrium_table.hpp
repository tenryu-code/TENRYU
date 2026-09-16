#pragma once

#include <vector>

#include "materials/cold_equilibrium.hpp"

namespace tenryu::materials {

struct EOSTable;

// Host owner of the cold-equilibrium primitive table of one material.
struct ColdEquilibriumTable {
  ColdEquilibriumParams params;
  std::vector<double> v_knots;   // ascending [cm^3/g]
  std::vector<double> PN_knots;  // [dyn/cm^2]
  std::vector<double> C0_knots;  // [erg/g]
  [[nodiscard]] bool empty() const noexcept { return v_knots.size() < 2; }
  [[nodiscard]] ColdEquilibriumView view() const;  // host pointers
};

// Builds the primitive from the runtime ion and electron tables of the
// material (P_N = ion.pressure(rho, Ti0) + electron.pressure(rho, Te0)).
// Knots: rho0 plus every ion-table density knot inside
// [rho0 / (density_outer_ratio * 1.05), rho0 * density_outer_ratio * 1.05]
// plus one bracketing knot beyond each end when available. Asserts on
// invalid parameters (rho0 > 0, K0 > 0, T_star > 0, 0 < T_begin_fraction < 1,
// 1 < density_core_ratio < density_outer_ratio, non-empty tables).
ColdEquilibriumTable build_cold_equilibrium_table(const EOSTable& ion,
                                                  const EOSTable& electron,
                                                  const ColdEquilibriumParams& params);

}  // namespace tenryu::materials
