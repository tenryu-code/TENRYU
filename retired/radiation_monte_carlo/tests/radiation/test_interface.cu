#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/interface.hpp"

using tenryu::radiation::compute_delta_x_m_1d;
using tenryu::radiation::compute_delta_x_m_2d_rz;
using tenryu::radiation::compute_P_hat_emissivity;
using tenryu::radiation::compute_standard_p_mu;

TEST_CASE("Interface P-hat known value", "[radiation][ddmc][interface]") {
  const auto result = compute_P_hat_emissivity(10.0, 1.0, 0.9, 0.25);
  REQUIRE_FALSE(result.used_standard_fallback);
  REQUIRE_FALSE(result.clamped_high);
  REQUIRE(result.c_hat ==
          Catch::Approx(0.6795489078720981).epsilon(1.0e-12));
  REQUIRE(result.probability ==
          Catch::Approx(0.4671898741620674).epsilon(1.0e-12));
}

TEST_CASE("Interface P-hat safety fallback when denominator is non-positive",
          "[radiation][ddmc][interface]") {
  const double sigma_R = 1.0;
  const double delta_x = 10.0;
  const double omega = 0.999;
  const double mu = 0.3;
  const auto result = compute_P_hat_emissivity(sigma_R, delta_x, omega, mu);
  const double p_standard = compute_standard_p_mu(sigma_R, delta_x, mu);
  REQUIRE(result.used_standard_fallback);
  REQUIRE(result.probability == Catch::Approx(p_standard).epsilon(1.0e-14));
}

TEST_CASE("Interface P-hat clamps scalar C-hat to 4/5",
          "[radiation][ddmc][interface]") {
  const auto result = compute_P_hat_emissivity(4.0, 0.8, 0.85, 0.5);
  REQUIRE(result.clamped_high);
  REQUIRE(result.c_hat == Catch::Approx(0.8).epsilon(1.0e-14));
  REQUIRE(result.probability == Catch::Approx(0.7).epsilon(1.0e-14));
}

TEST_CASE("Interface omega->1 limit gives zero emissivity conversion probability",
          "[radiation][ddmc][interface]") {
  const auto result = compute_P_hat_emissivity(10.0, 1.0, 1.0, 0.75);
  REQUIRE(result.c_hat == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(result.probability == Catch::Approx(0.0).margin(1.0e-14));
}

TEST_CASE("Interface directional probability respects C-hat(mu=1)<=1",
          "[radiation][ddmc][interface]") {
  const std::vector<double> omegas = {0.7, 0.85, 0.9, 0.95};
  const std::vector<double> taus = {2.0, 4.0, 8.0, 12.0};

  for (const double omega : omegas) {
    for (const double tau : taus) {
      const auto result = compute_P_hat_emissivity(tau, 1.0, omega, 1.0);
      REQUIRE(result.probability <= 1.0 + 1.0e-14);
    }
  }
}

TEST_CASE("Interface standard P(mu) formula and clamping",
          "[radiation][ddmc][interface]") {
  const double p_ref = compute_standard_p_mu(10.0, 1.0, 0.25);
  REQUIRE(p_ref == Catch::Approx(0.16052582422714112).epsilon(1.0e-12));

  const double p_clamped = compute_standard_p_mu(0.1, 0.1, 1.0);
  REQUIRE(p_clamped == Catch::Approx(1.0).margin(1.0e-14));
}

TEST_CASE("Interface delta_x_m definitions for 1D and 2D_RZ",
          "[radiation][ddmc][interface]") {
  const double dx_1d = compute_delta_x_m_1d(1.0, 1.5);
  REQUIRE(dx_1d == Catch::Approx(0.5).epsilon(1.0e-14));

  const double dx_2d = compute_delta_x_m_2d_rz(2.0, 1.0, 0.5, 1.2);
  REQUIRE(dx_2d ==
          Catch::Approx(2.0 / 3.14159265358979323846).epsilon(1.0e-14));

  const double dx_axis = compute_delta_x_m_2d_rz(2.0, 0.0, 0.5, 1.2);
  REQUIRE(dx_axis ==
          Catch::Approx(2.0 / (3.14159265358979323846 * 1.2 * 0.5))
              .epsilon(1.0e-14));
}
