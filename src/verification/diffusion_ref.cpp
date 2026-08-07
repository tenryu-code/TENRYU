#include "verification/diffusion_ref.hpp"

#include <algorithm>
#include <vector>

#include "core/constants.hpp"

namespace tenryu::verification {

std::vector<double> solve_diffusion_reference_1d(const DiffusionReferenceConfig& cfg) {
  if (cfg.n_cells <= 0 || cfg.length_cm <= 0.0 || cfg.dt_s <= 0.0 || cfg.n_steps <= 0) {
    return {};
  }

  const double dx = cfg.length_cm / static_cast<double>(cfg.n_cells);
  const double inv_dx2 = 1.0 / (dx * dx);

  std::vector<double> E(static_cast<std::size_t>(cfg.n_cells),
                        std::max(cfg.initial_rad_E, 0.0));
  std::vector<double> E_next(E.size(), 0.0);

  for (int step = 0; step < cfg.n_steps; ++step) {
    for (int i = 0; i < cfg.n_cells; ++i) {
      const double Ei = E[static_cast<std::size_t>(i)];
      const double E_left =
          (i == 0) ? std::max(cfg.left_boundary_rad_E, 0.0)
                   : E[static_cast<std::size_t>(i - 1)];
      // Vacuum-like sink at the outer boundary.
      const double E_right =
          (i == cfg.n_cells - 1) ? 0.0 : E[static_cast<std::size_t>(i + 1)];

      const double lap = (E_left - 2.0 * Ei + E_right) * inv_dx2;
      const double rhs = tenryu::core::constants::c_light *
                         (cfg.diffusion_coeff_cm * lap - cfg.sigma_abs_cm * Ei);
      E_next[static_cast<std::size_t>(i)] = std::max(Ei + cfg.dt_s * rhs, 0.0);
    }
    E.swap(E_next);
  }

  return E;
}

}  // namespace tenryu::verification
