#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "radiation/ddmc_coefficients.hpp"
#include "radiation/mmatrix_check.hpp"
#include "radiation/mode_selector.hpp"

using tenryu::radiation::DDMCBoundaryType;
using tenryu::radiation::DDMCCoefficients;
using tenryu::radiation::ModeSelector;
using tenryu::radiation::ModeSelectorConfig;
using tenryu::radiation::TransportMode;

TEST_CASE("M-matrix single-cell predicate", "[radiation][ddmc][mmatrix]") {
  REQUIRE(tenryu::radiation::check_mmatrix_single(0.1, 0.1, 0.0, 0.5));
  REQUIRE_FALSE(tenryu::radiation::check_mmatrix_single(-0.1, 0.1, 0.0, 0.5));
  REQUIRE_FALSE(tenryu::radiation::check_mmatrix_single(0.1, 0.1, 0.0, -0.1));
  REQUIRE(tenryu::radiation::check_mmatrix_single(-1.0e-15, 0.1, 0.0, 0.5, 1.0e-12));
  REQUIRE_FALSE(
      tenryu::radiation::check_mmatrix_single(-1.0e-8, 0.1, 0.0, 0.5, 1.0e-12));
}

TEST_CASE("M-matrix fallback forces IMC on violated cell",
          "[radiation][ddmc][mmatrix]") {
  constexpr int n_cells = 3;
  constexpr int n_groups = 1;
  const std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0};
  const std::vector<double> rho(n_cells, 1.0);
  const std::vector<double> Te(n_cells, 10.0);
  const std::vector<double> sigma(n_cells, 20.0);
  const std::vector<double> fleck_f(n_cells, 0.05);

  ModeSelectorConfig cfg{};
  cfg.tau_ddmc = 4.0;
  cfg.omega_ddmc = 0.9;
  ModeSelector mode(n_cells, n_groups, cfg);
  mode.compute_modes(node_r, sigma, fleck_f, sigma);
  REQUIRE(mode.count_ddmc() == n_cells);

  DDMCCoefficients coeff(n_cells, n_groups);
  coeff.compute_1d(node_r,
                   rho,
                   Te,
                   sigma,
                   mode,
                   DDMCBoundaryType::Vacuum,
                   DDMCBoundaryType::Vacuum,
                   false,
                   nullptr);

  auto& bad = const_cast<tenryu::radiation::CellDDMCData&>(coeff.get_cell_data(1, 0));
  bad.sigma_leak_left = -std::abs(bad.sigma_leak_left);
  bad.sigma_leak_out = bad.sigma_leak_left + bad.sigma_leak_right;

  const std::vector<double> sigma_a_eff(n_cells, 1.0);
  const auto diag =
      tenryu::radiation::check_mmatrix_condition(coeff, mode, sigma_a_eff);

  REQUIRE(diag.total_violations >= 1);
  REQUIRE(diag.off_diagonal_violations >= 1);
  REQUIRE(mode.get_mode(1, 0) == TransportMode::IMC);
  REQUIRE(mode.get_mode(0, 0) == TransportMode::DDMC);
  REQUIRE(mode.get_mode(2, 0) == TransportMode::DDMC);
}
