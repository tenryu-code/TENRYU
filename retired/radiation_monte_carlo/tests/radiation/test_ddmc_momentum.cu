#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/ddmc_coefficients.hpp"
#include "radiation/ddmc_momentum.hpp"
#include "radiation/mode_selector.hpp"

using tenryu::radiation::DDMCBoundaryType;
using tenryu::radiation::DDMCCoefficients;
using tenryu::radiation::DDMCMomentumEstimator;
using tenryu::radiation::ModeSelector;
using tenryu::radiation::ModeSelectorConfig;

namespace {

DDMCCoefficients build_uniform_coefficients() {
  constexpr int n_cells = 3;
  constexpr int n_groups = 1;

  const std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0};
  const std::vector<double> rho = {1.0, 1.0, 1.0};
  const std::vector<double> Te = {10.0, 10.0, 10.0};
  const std::vector<double> sigma = {10.0, 10.0, 10.0};
  const std::vector<double> fleck_f = {0.05, 0.05, 0.05};

  ModeSelectorConfig mode_cfg{};
  mode_cfg.tau_ddmc = 4.0;
  mode_cfg.omega_ddmc = 0.9;
  ModeSelector mode(n_cells, n_groups, mode_cfg);
  mode.compute_modes(node_r, sigma, fleck_f, sigma);

  DDMCCoefficients coeff(n_cells, n_groups);
  coeff.compute_1d(node_r,
                   rho,
                   Te,
                   sigma,
                   mode,
                   DDMCBoundaryType::Reflective,
                   DDMCBoundaryType::Reflective,
                   true,
                   nullptr);
  return coeff;
}

}  // namespace

TEST_CASE("DDMC momentum face flux tally", "[radiation][ddmc][momentum]") {
  DDMCMomentumEstimator estimator(3, 1);

  estimator.tally_face_flux(0, 0, +1, 100.0);
  estimator.tally_face_flux(1, 0, -1, 40.0);

  REQUIRE(estimator.get_right_flux(1, 0) == Catch::Approx(100.0).epsilon(1.0e-14));
  REQUIRE(estimator.get_left_flux(1, 0) == Catch::Approx(40.0).epsilon(1.0e-14));
  REQUIRE(estimator.get_net_face_flux(1, 0) == Catch::Approx(60.0).epsilon(1.0e-14));
}

TEST_CASE("DDMC momentum reset", "[radiation][ddmc][momentum]") {
  DDMCMomentumEstimator estimator(3, 1);
  estimator.tally_face_flux(0, 0, +1, 100.0);
  estimator.reset();
  REQUIRE(estimator.get_right_flux(1, 0) == Catch::Approx(0.0).margin(1.0e-30));
  REQUIRE(estimator.get_left_flux(1, 0) == Catch::Approx(0.0).margin(1.0e-30));
}

TEST_CASE("DDMC momentum deposition symmetry", "[radiation][ddmc][momentum]") {
  const auto coeff = build_uniform_coefficients();
  DDMCMomentumEstimator estimator(3, 1);

  estimator.tally_face_flux(1, 0, -1, 50.0);
  estimator.tally_face_flux(1, 0, +1, 50.0);
  const auto momentum = estimator.compute_cell_momentum_deposition(coeff);

  REQUIRE(momentum.size() == 3);
  REQUIRE(std::abs(momentum[1]) < 1.0e-20);
}
