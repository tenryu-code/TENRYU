#pragma once

#include <cstdint>
#include <vector>

#include "radiation/ddmc_coefficients.hpp"

namespace tenryu::radiation {

class DDMCMomentumEstimator {
 public:
  DDMCMomentumEstimator() = default;
  DDMCMomentumEstimator(std::int64_t n_cells, int n_groups);

  void reset();

  void tally_face_flux(std::int64_t cell,
                       int group,
                       int direction,
                       double energy);

  [[nodiscard]] std::vector<double> compute_cell_momentum_deposition(
      const DDMCCoefficients& coefficients) const;

  [[nodiscard]] double get_right_flux(std::int64_t face, int group) const;
  [[nodiscard]] double get_left_flux(std::int64_t face, int group) const;
  [[nodiscard]] double get_net_face_flux(std::int64_t face, int group) const;

  [[nodiscard]] std::int64_t n_cells() const { return n_cells_; }
  [[nodiscard]] int n_groups() const { return n_groups_; }

 private:
  [[nodiscard]] std::size_t index(std::int64_t face, int group) const;

  std::int64_t n_cells_ = 0;
  int n_groups_ = 0;
  std::vector<double> face_flux_right_;
  std::vector<double> face_flux_left_;
};

}  // namespace tenryu::radiation
