#pragma once

#include <vector>

namespace tenryu::radiation {

class Groups {
 public:
  Groups();
  explicit Groups(std::vector<double> group_bounds_eV);

  [[nodiscard]] static std::vector<double> make_log_uniform_bounds(int n_groups,
                                                                   double E_min_eV,
                                                                   double E_max_eV);

  [[nodiscard]] int num_groups() const noexcept;
  [[nodiscard]] double energy_lo(int g) const;
  [[nodiscard]] double energy_hi(int g) const;
  [[nodiscard]] double energy_rep(int g) const;
  [[nodiscard]] double planck_fraction(int g, double T_eV) const;
  [[nodiscard]] const std::vector<double>& bounds_eV() const noexcept;

  [[nodiscard]] double planck_fraction_raw(int g, double T_eV) const;

 private:

  std::vector<double> group_bounds_eV_;
};

}  // namespace tenryu::radiation
