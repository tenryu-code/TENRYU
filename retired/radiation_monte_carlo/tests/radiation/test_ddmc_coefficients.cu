#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/ddmc_coefficients.hpp"
#include "radiation/mode_selector.hpp"

using tenryu::radiation::CellDDMCData;
using tenryu::radiation::ConstantOpacityProvider;
using tenryu::radiation::DDMCBoundaryType;
using tenryu::radiation::DDMCCoefficients;
using tenryu::radiation::ModeSelector;
using tenryu::radiation::ModeSelectorConfig;
using tenryu::radiation::TransportMode;

TEST_CASE("DDMC coefficients basic formulas", "[radiation][ddmc][coefficients]") {
  REQUIRE(DDMCCoefficients::compute_diffusion_coeff(1.0) ==
          Catch::Approx(1.0 / 3.0).epsilon(1.0e-14));
  REQUIRE(DDMCCoefficients::compute_diffusion_coeff(10.0) ==
          Catch::Approx(1.0 / 30.0).epsilon(1.0e-14));

  const double tf = DDMCCoefficients::compute_face_temperature(1.0, 2.0);
  const double tf_ref = std::pow(0.5 * (1.0 + 16.0), 0.25);
  REQUIRE(tf == Catch::Approx(tf_ref).epsilon(1.0e-14));
}

TEST_CASE("DDMC coefficients uniform slab leak rates",
          "[radiation][ddmc][coefficients]") {
  constexpr int n_cells = 5;
  constexpr int n_groups = 1;
  const std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  const std::vector<double> rho(n_cells, 1.0);
  const std::vector<double> Te(n_cells, 10.0);
  const std::vector<double> sigma_R(n_cells, 10.0);
  const std::vector<double> sigma_a = sigma_R;
  const std::vector<double> fleck_f(n_cells, 0.05);

  ModeSelectorConfig mode_cfg{};
  mode_cfg.tau_ddmc = 4.0;
  mode_cfg.omega_ddmc = 0.9;
  ModeSelector mode(n_cells, n_groups, mode_cfg);
  mode.compute_modes(node_r, sigma_R, fleck_f, sigma_a);

  for (int c = 0; c < n_cells; ++c) {
    REQUIRE(mode.get_mode(c, 0) == TransportMode::DDMC);
  }

  DDMCCoefficients coeff(n_cells, n_groups);
  coeff.compute_1d(node_r,
                   rho,
                   Te,
                   sigma_R,
                   mode,
                   DDMCBoundaryType::Vacuum,
                   DDMCBoundaryType::Vacuum,
                   true,
                   nullptr);

  const double sigma_expected = 2.0 / (3.0 * 1.0 * (10.0 + 10.0));
  const CellDDMCData mid = coeff.get_cell_data(2, 0);
  REQUIRE(mid.sigma_leak_left == Catch::Approx(sigma_expected).epsilon(1.0e-10));
  REQUIRE(mid.sigma_leak_right == Catch::Approx(sigma_expected).epsilon(1.0e-10));

  const CellDDMCData left = coeff.get_cell_data(0, 0);
  REQUIRE(left.bc_left == DDMCBoundaryType::Vacuum);
  REQUIRE(left.sigma_leak_left > 0.0);

  const CellDDMCData right = coeff.get_cell_data(n_cells - 1, 0);
  REQUIRE(right.bc_right == DDMCBoundaryType::Vacuum);
  REQUIRE(right.sigma_leak_right > 0.0);
}

TEST_CASE("DDMC coefficients reflective boundary and face opacities",
          "[radiation][ddmc][coefficients]") {
  constexpr int n_cells = 2;
  constexpr int n_groups = 1;
  const std::vector<double> node_r = {0.0, 1.0, 2.0};
  const std::vector<double> rho = {1.0, 2.0};
  const std::vector<double> Te = {4.0, 8.0};
  const std::vector<double> sigma_center = {10.0, 10.0};
  const std::vector<double> fleck_f = {0.0, 0.0};

  ModeSelectorConfig mode_cfg{};
  mode_cfg.tau_ddmc = 3.0;
  mode_cfg.omega_ddmc = 0.0;
  ModeSelector mode(n_cells, n_groups, mode_cfg);
  mode.compute_modes(node_r, sigma_center, fleck_f, sigma_center);
  REQUIRE(mode.get_mode(0, 0) == TransportMode::DDMC);
  REQUIRE(mode.get_mode(1, 0) == TransportMode::DDMC);

  ConstantOpacityProvider provider({10.0});
  DDMCCoefficients coeff(n_cells, n_groups, 1.0e-20);
  coeff.compute_1d(node_r,
                   rho,
                   Te,
                   sigma_center,
                   mode,
                   DDMCBoundaryType::Reflective,
                   DDMCBoundaryType::Reflective,
                   true,
                   &provider);

  const auto c0 = coeff.get_cell_data(0, 0);
  const auto c1 = coeff.get_cell_data(1, 0);
  REQUIRE(c0.sigma_leak_left == Catch::Approx(0.0).margin(1.0e-30));
  REQUIRE(c1.sigma_leak_right == Catch::Approx(0.0).margin(1.0e-30));
  REQUIRE(coeff.get_face_sigma_minus(1, 0) ==
          Catch::Approx(10.0).epsilon(1.0e-14));
  REQUIRE(coeff.get_face_sigma_plus(1, 0) ==
          Catch::Approx(20.0).epsilon(1.0e-14));

  // Independent floor check with tiny center sigma.
  const std::vector<double> sigma_tiny = {1.0e-30, 1.0e-30};
  ModeSelectorConfig tiny_mode_cfg{};
  tiny_mode_cfg.tau_ddmc = 0.0;
  tiny_mode_cfg.omega_ddmc = 0.0;
  tiny_mode_cfg.sigma_floor = 0.0;
  ModeSelector tiny_mode(n_cells, n_groups, tiny_mode_cfg);
  tiny_mode.compute_modes(node_r, sigma_tiny, fleck_f, sigma_tiny);

  DDMCCoefficients coeff_floor(n_cells, n_groups, 1.0e-20);
  coeff_floor.compute_1d(node_r,
                         rho,
                         Te,
                         sigma_tiny,
                         tiny_mode,
                         DDMCBoundaryType::Reflective,
                         DDMCBoundaryType::Reflective,
                         true,
                         nullptr);

  // Boundary face with tiny center sigma must be clamped by floor.
  REQUIRE(coeff_floor.get_face_sigma_minus(0, 0) >= 1.0e-20);
  REQUIRE(coeff_floor.get_face_sigma_plus(2, 0) >= 1.0e-20);
}
