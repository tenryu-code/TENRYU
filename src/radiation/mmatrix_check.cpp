#include "radiation/mmatrix_check.hpp"

namespace tenryu::radiation {

bool check_mmatrix_single(const double sigma_leak_left,
                          const double sigma_leak_right,
                          const double sigma_leak_bnd,
                          const double sigma_a_eff,
                          const double tolerance,
                          const double sigma_leak_face2,
                          const double sigma_leak_face3) {
  if (sigma_leak_left < -tolerance || sigma_leak_right < -tolerance ||
      sigma_leak_bnd < -tolerance || sigma_leak_face2 < -tolerance ||
      sigma_leak_face3 < -tolerance) {
    return false;
  }

  const double diagonal =
      sigma_a_eff + sigma_leak_left + sigma_leak_right + sigma_leak_face2 +
      sigma_leak_face3 + sigma_leak_bnd;
  if (diagonal <= tolerance) {
    return false;
  }

  if (sigma_a_eff < -tolerance) {
    return false;
  }

  return true;
}

MMatrixDiagnostics check_mmatrix_condition(const DDMCCoefficients& coefficients,
                                           ModeSelector& mode_selector,
                                           const std::vector<double>& sigma_a_eff,
                                           const double tolerance) {
  MMatrixDiagnostics diagnostics{};
  const std::int64_t n_cells = mode_selector.n_cells();
  const int n_groups = mode_selector.n_groups();

  for (std::int64_t c = 0; c < n_cells; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      if (mode_selector.get_mode(c, g) != TransportMode::DDMC) {
        continue;
      }

      const std::size_t idx =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(g);
      const auto& data = coefficients.get_cell_data(c, g);
      const double sigma_a = sigma_a_eff[idx];
      const double leak_f0 = data.sigma_leak_left;
      const double leak_f1 = data.sigma_leak_right;
      const double leak_f2 = data.sigma_leak_face[2];
      const double leak_f3 = data.sigma_leak_face[3];

      if (leak_f0 < -tolerance || leak_f1 < -tolerance || leak_f2 < -tolerance ||
          leak_f3 < -tolerance || data.sigma_leak_bnd < -tolerance) {
        ++diagnostics.off_diagonal_violations;
      }

      const double diagonal = sigma_a + data.sigma_leak_out + data.sigma_leak_bnd;
      if (diagonal <= tolerance) {
        ++diagnostics.diagonal_violations;
      }

      if (sigma_a < -tolerance) {
        ++diagnostics.row_sum_violations;
      }

      if (!check_mmatrix_single(data.sigma_leak_left,
                                data.sigma_leak_right,
                                data.sigma_leak_bnd,
                                sigma_a,
                                tolerance,
                                leak_f2,
                                leak_f3)) {
        mode_selector.force_imc(c, g);
        ++diagnostics.total_violations;
      }
    }
  }

  return diagnostics;
}

}  // namespace tenryu::radiation
