#pragma once

#include <cstdint>
#include <vector>

#include "radiation/ddmc_coefficients.hpp"
#include "radiation/mode_selector.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

struct DDMCDiffusionSolveResult {
  std::int64_t solved_cell_groups = 0;
  double escaped_energy = 0.0;
};

DDMCDiffusionSolveResult solve_ddmc_diffusion_1d(
    const ModeSelector& mode_selector,
    const DDMCCoefficients& coefficients,
    const PlanckTable& planck,
    const std::vector<double>& node_r,
    const std::vector<double>& cell_vol,
    const std::vector<double>& sigma_a_eff,
    const std::vector<double>& Te,
    const std::vector<double>& cell_heat_capacity,
    const std::vector<double>& rad_E_prev,
    double* d_rad_dep,
    double* d_rad_emit,
    double* d_rad_E_tally,
    double* d_E_escape,
    double dt);

}  // namespace tenryu::radiation
