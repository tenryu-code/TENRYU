#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "materials/eos_table.hpp"

namespace tenryu::materials {

struct HelmholtzSpline1T {
  std::vector<double> log_rho_grid;
  std::vector<double> log_T_grid;

  // Compatibility note: the backend key remains "helmholtz_spline", but the runtime
  // implementation uses a shape-preserving C1 tensor-product bicubic Hermite spline for
  // the hydro total-EOS surrogate P(log rho, log T) and e(log rho, log T) directly.
  std::vector<double> P_nodes;
  std::vector<double> P_dx_nodes;
  std::vector<double> P_dy_nodes;
  std::vector<double> P_dxdy_nodes;

  std::vector<double> e_nodes;
  std::vector<double> e_dx_nodes;
  std::vector<double> e_dy_nodes;
  std::vector<double> e_dxdy_nodes;

  [[nodiscard]] std::size_t n_rho() const noexcept {
    return log_rho_grid.size();
  }

  [[nodiscard]] std::size_t n_T() const noexcept {
    return log_T_grid.size();
  }

  [[nodiscard]] std::size_t flat_index(const std::size_t i_rho,
                                       const std::size_t j_T) const noexcept {
    return j_T * n_rho() + i_rho;
  }

  [[nodiscard]] bool empty() const noexcept {
    return log_rho_grid.empty() || log_T_grid.empty();
  }

  void clear();
  void build_from_table(const EOSTable& table, std::string_view label = "unnamed");
};

struct HelmholtzSplineTriplet {
  HelmholtzSpline1T ion;
  HelmholtzSpline1T electron;
  HelmholtzSpline1T total;
};

[[nodiscard]] HelmholtzSplineTriplet build_helmholtz_spline_triplet(
    const EOSTableTriplet& tables);

}  // namespace tenryu::materials
