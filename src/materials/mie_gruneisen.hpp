#pragma once

#include <string>
#include <vector>

namespace tenryu::materials {

struct EOSTable;

class MieGruneisenEOS {
 public:
  void build_from_tables(const EOSTable& ion,
                         const EOSTable& electron,
                         double T_ref_eV,
                         double dT_rel,
                         const std::string& label = "");

  [[nodiscard]] bool empty() const noexcept {
    return log_rho_grid_.empty();
  }

  [[nodiscard]] std::size_t n_rho() const noexcept {
    return log_rho_grid_.size();
  }

  [[nodiscard]] const std::vector<double>& log_rho_grid() const noexcept {
    return log_rho_grid_;
  }

  [[nodiscard]] const std::vector<double>& gamma_ion() const noexcept {
    return gamma_ion_;
  }

  [[nodiscard]] const std::vector<double>& gamma_ele() const noexcept {
    return gamma_ele_;
  }

  [[nodiscard]] const std::vector<double>& p_ref_ion() const noexcept {
    return p_ref_ion_;
  }

  [[nodiscard]] const std::vector<double>& p_ref_ele() const noexcept {
    return p_ref_ele_;
  }

  [[nodiscard]] const std::vector<double>& e_ref_ion() const noexcept {
    return e_ref_ion_;
  }

  [[nodiscard]] const std::vector<double>& e_ref_ele() const noexcept {
    return e_ref_ele_;
  }

  [[nodiscard]] const std::vector<double>& dgamma_ion_dlogrho() const noexcept {
    return dgamma_ion_dlogrho_;
  }

  [[nodiscard]] const std::vector<double>& dgamma_ele_dlogrho() const noexcept {
    return dgamma_ele_dlogrho_;
  }

  [[nodiscard]] const std::vector<double>& dp_ref_ion_dlogrho() const noexcept {
    return dp_ref_ion_dlogrho_;
  }

  [[nodiscard]] const std::vector<double>& dp_ref_ele_dlogrho() const noexcept {
    return dp_ref_ele_dlogrho_;
  }

  [[nodiscard]] const std::vector<double>& de_ref_ion_dlogrho() const noexcept {
    return de_ref_ion_dlogrho_;
  }

  [[nodiscard]] const std::vector<double>& de_ref_ele_dlogrho() const noexcept {
    return de_ref_ele_dlogrho_;
  }

 private:
  std::vector<double> log_rho_grid_;
  std::vector<double> gamma_ion_;
  std::vector<double> gamma_ele_;
  std::vector<double> p_ref_ion_;
  std::vector<double> p_ref_ele_;
  std::vector<double> e_ref_ion_;
  std::vector<double> e_ref_ele_;
  std::vector<double> dgamma_ion_dlogrho_;
  std::vector<double> dgamma_ele_dlogrho_;
  std::vector<double> dp_ref_ion_dlogrho_;
  std::vector<double> dp_ref_ele_dlogrho_;
  std::vector<double> de_ref_ion_dlogrho_;
  std::vector<double> de_ref_ele_dlogrho_;
};

}  // namespace tenryu::materials
