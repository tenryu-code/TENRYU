#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "materials/eos_table.hpp"

namespace tenryu::materials {

struct HelmholtzBSplineFitOptions {
  int degree = 5;
  double relative_floor_fraction = 1.0e-3;
  double lambda_second_diff = 1.0e-5;
  double lambda_ref_cv = 1.0e-2;
  double lambda_ref_drho = 1.0e-2;
  double lambda_constraint_initial = 1.0;
  double positivity_eps_fraction = 1.0e-6;
  int knot_stride_rho = 1;
  int knot_stride_T = 1;
  int max_constraint_iters = 200;
  double gauge_weight = 1.0;
  double diagonal_jitter = 1.0e-12;
};

struct HelmholtzBSplineEval {
  double phi = 0.0;
  double phi_x = 0.0;
  double phi_y = 0.0;
  double phi_xx = 0.0;
  double phi_yy = 0.0;
  double phi_xy = 0.0;
};

struct HelmholtzBSplineThermo {
  double pressure = 0.0;
  double energy = 0.0;
  double cv = 0.0;
  double dP_drho_T = 0.0;
  double dP_dT_rho = 0.0;
  double sound_speed = 0.0;
};

class HelmholtzBSpline1T {
 public:
  int degree = 5;
  std::vector<double> log_rho_grid;
  std::vector<double> log_T_grid;
  std::vector<double> knot_x;
  std::vector<double> knot_y;
  std::size_t n_basis_x = 0;
  std::size_t n_basis_y = 0;
  double phi_scale = 1.0;
  double effective_lambda_constraint = 0.0;
  std::size_t final_active_set_size = 0;
  int total_qp_iterations = 0;
  std::vector<double> coeffs;

  [[nodiscard]] bool empty() const noexcept {
    return coeffs.empty();
  }

  [[nodiscard]] std::size_t n_coeffs() const noexcept {
    return coeffs.size();
  }

  void clear();
  void build_from_table(const EOSTable& table,
                        const HelmholtzBSplineFitOptions& options = {},
                        std::string_view label = "unnamed");
  [[nodiscard]] static bool self_test();

  [[nodiscard]] HelmholtzBSplineEval eval(double log_rho, double log_T) const;
  [[nodiscard]] HelmholtzBSplineThermo eval_thermo(double rho, double T_eV) const;
  [[nodiscard]] double temperature_from_energy(double rho, double e_target) const;
};

}  // namespace tenryu::materials
