#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

#include "materials/eos_table.hpp"

namespace tenryu::materials {

struct HelmholtzJetNode {
  double phi = 0.0;
  double phi_x = 0.0;
  double phi_y = 0.0;
  double phi_xx = 0.0;
  double phi_yy = 0.0;
  double phi_xy = 0.0;
};

struct HelmholtzJetEval {
  double phi = 0.0;
  double phi_x = 0.0;
  double phi_y = 0.0;
  double phi_xx = 0.0;
  double phi_yy = 0.0;
  double phi_xy = 0.0;
};

struct HelmholtzJetThermo {
  double pressure = 0.0;
  double energy = 0.0;
  double cv = 0.0;
  double dP_drho_T = 0.0;
  double dP_dT_rho = 0.0;
  double sound_speed = 0.0;
};

class HelmholtzJetEOS {
 public:
  std::vector<double> log_rho_grid;
  std::vector<double> log_T_grid;
  std::vector<HelmholtzJetNode> nodes;
  std::vector<double> cell_coeffs;  // [(n_rho-1)*(n_T-1)*36]
  std::size_t n_rho = 0;
  std::size_t n_T = 0;

  [[nodiscard]] bool empty() const noexcept {
    return n_rho < 2 || n_T < 2 || cell_coeffs.empty();
  }

  [[nodiscard]] std::size_t flat_index(const std::size_t i_rho,
                                       const std::size_t j_T) const noexcept {
    return j_T * n_rho + i_rho;
  }

  void clear();
  void build_from_table(const EOSTable& table, std::string_view label = "unnamed");

  [[nodiscard]] HelmholtzJetEval eval(double log_rho, double log_T) const;
  [[nodiscard]] HelmholtzJetThermo eval_thermo(double rho, double T_eV) const;
  [[nodiscard]] double temperature_from_energy(double rho, double e_target) const;
};

}  // namespace tenryu::materials
