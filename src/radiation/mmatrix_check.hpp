#pragma once

#include <cstdint>
#include <vector>

#include "radiation/ddmc_coefficients.hpp"
#include "radiation/mode_selector.hpp"

namespace tenryu::radiation {

struct MMatrixDiagnostics {
  std::int64_t total_violations = 0;
  std::int64_t off_diagonal_violations = 0;
  std::int64_t diagonal_violations = 0;
  std::int64_t row_sum_violations = 0;
};

[[nodiscard]] bool check_mmatrix_single(double sigma_leak_left,
                                        double sigma_leak_right,
                                        double sigma_leak_bnd,
                                        double sigma_a_eff,
                                        double tolerance = 1.0e-12,
                                        double sigma_leak_face2 = 0.0,
                                        double sigma_leak_face3 = 0.0);

[[nodiscard]] MMatrixDiagnostics check_mmatrix_condition(
    const DDMCCoefficients& coefficients,
    ModeSelector& mode_selector,
    const std::vector<double>& sigma_a_eff,
    double tolerance = 1.0e-12);

}  // namespace tenryu::radiation
