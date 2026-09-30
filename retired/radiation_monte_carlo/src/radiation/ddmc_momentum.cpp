#include "radiation/ddmc_momentum.hpp"

#include <algorithm>

#include "core/constants.hpp"

namespace tenryu::radiation {

DDMCMomentumEstimator::DDMCMomentumEstimator(const std::int64_t n_cells,
                                             const int n_groups)
    : n_cells_(n_cells),
      n_groups_(n_groups),
      face_flux_right_(static_cast<std::size_t>(n_cells + 1) *
                       static_cast<std::size_t>(n_groups),
                       0.0),
      face_flux_left_(static_cast<std::size_t>(n_cells + 1) *
                      static_cast<std::size_t>(n_groups),
                      0.0) {}

std::size_t DDMCMomentumEstimator::index(const std::int64_t face,
                                         const int group) const {
  return static_cast<std::size_t>(face) * static_cast<std::size_t>(n_groups_) +
         static_cast<std::size_t>(group);
}

void DDMCMomentumEstimator::reset() {
  std::fill(face_flux_right_.begin(), face_flux_right_.end(), 0.0);
  std::fill(face_flux_left_.begin(), face_flux_left_.end(), 0.0);
}

void DDMCMomentumEstimator::tally_face_flux(const std::int64_t cell,
                                            const int group,
                                            const int direction,
                                            const double energy) {
  // TODO(M16): include dt/face-area normalization in the stored face flux if
  // momentum coupling is promoted beyond the current diagnostic path.
  if (group < 0 || group >= n_groups_) {
    return;
  }
  if (cell < 0 || cell >= n_cells_) {
    return;
  }

  if (direction > 0) {
    const std::int64_t face = cell + 1;
    face_flux_right_[index(face, group)] += energy;
  } else {
    const std::int64_t face = cell;
    face_flux_left_[index(face, group)] += energy;
  }
}

std::vector<double> DDMCMomentumEstimator::compute_cell_momentum_deposition(
    const DDMCCoefficients& coefficients) const {
  std::vector<double> momentum(static_cast<std::size_t>(n_cells_), 0.0);

  for (std::int64_t c = 0; c < n_cells_; ++c) {
    double cell_momentum = 0.0;
    for (int g = 0; g < n_groups_; ++g) {
      const std::int64_t left_face = c;
      const std::int64_t right_face = c + 1;

      const double flux_left = get_net_face_flux(left_face, g);
      const double flux_right = get_net_face_flux(right_face, g);
      const double sigma_plus_left = coefficients.get_face_sigma_plus(left_face, g);
      const double sigma_minus_right = coefficients.get_face_sigma_minus(right_face, g);

      // TODO(M16): this diagnostic currently uses unnormalized tallies; align
      // with the fully normalized DDMC momentum estimator before enabling force
      // coupling in production runs.
      cell_momentum += sigma_plus_left * flux_left + sigma_minus_right * flux_right;
    }
    momentum[static_cast<std::size_t>(c)] =
        cell_momentum / (2.0 * tenryu::core::constants::c_light);
  }

  return momentum;
}

double DDMCMomentumEstimator::get_right_flux(const std::int64_t face,
                                             const int group) const {
  if (face < 0 || face > n_cells_ || group < 0 || group >= n_groups_) {
    return 0.0;
  }
  return face_flux_right_[index(face, group)];
}

double DDMCMomentumEstimator::get_left_flux(const std::int64_t face,
                                            const int group) const {
  if (face < 0 || face > n_cells_ || group < 0 || group >= n_groups_) {
    return 0.0;
  }
  return face_flux_left_[index(face, group)];
}

double DDMCMomentumEstimator::get_net_face_flux(const std::int64_t face,
                                                const int group) const {
  return get_right_flux(face, group) - get_left_flux(face, group);
}

}  // namespace tenryu::radiation
