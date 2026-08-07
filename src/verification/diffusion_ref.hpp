#pragma once

#include <vector>

namespace tenryu::verification {

struct DiffusionReferenceConfig {
  int n_cells = 0;
  double length_cm = 1.0;
  double diffusion_coeff_cm = 0.0;
  double sigma_abs_cm = 0.0;
  double dt_s = 0.0;
  int n_steps = 0;
  double initial_rad_E = 0.0;
  double left_boundary_rad_E = 0.0;
};

[[nodiscard]] std::vector<double> solve_diffusion_reference_1d(
    const DiffusionReferenceConfig& cfg);

}  // namespace tenryu::verification
