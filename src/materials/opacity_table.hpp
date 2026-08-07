#pragma once

#include <cstddef>
#include <vector>

namespace tenryu::materials {

// Placeholder container for parsed tabular opacity payloads.
// Runtime interpolation is not wired in v1.0; namelist validation emits a
// warning and then throws ConfigError for tabular opacity models to avoid
// silent fallback.
struct OpacityTable {
  std::vector<double> rho_grid;
  std::vector<double> T_grid_eV;
  std::vector<double> kappa_planck;
  std::vector<double> kappa_rosseland;

  [[nodiscard]] std::size_t n_rho() const noexcept {
    return rho_grid.size();
  }

  [[nodiscard]] std::size_t n_T() const noexcept {
    return T_grid_eV.size();
  }

  [[nodiscard]] bool empty() const noexcept {
    return rho_grid.empty() || T_grid_eV.empty();
  }
};

}  // namespace tenryu::materials
