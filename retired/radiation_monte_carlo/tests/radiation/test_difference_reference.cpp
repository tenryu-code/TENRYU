#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/imc.hpp"

TEST_CASE("Difference reference weight is monotone and continuous",
          "[radiation][difference]") {
  constexpr double W_max = 1.0;
  constexpr double tau0 = 3.0;
  constexpr double chi0 = 0.25;

  const double W_tau_lo =
      tenryu::radiation::difference_reference_weight(W_max, 0.5, tau0, 0.0, chi0);
  const double W_tau_mid =
      tenryu::radiation::difference_reference_weight(W_max, 3.0, tau0, 0.0, chi0);
  const double W_tau_hi =
      tenryu::radiation::difference_reference_weight(W_max, 30.0, tau0, 0.0, chi0);

  REQUIRE(W_tau_lo < W_tau_mid);
  REQUIRE(W_tau_mid < W_tau_hi);
  REQUIRE(W_tau_hi < W_max);

  const double W_chi_lo =
      tenryu::radiation::difference_reference_weight(W_max, 30.0, tau0, 0.05, chi0);
  const double W_chi_hi =
      tenryu::radiation::difference_reference_weight(W_max, 30.0, tau0, 1.0, chi0);
  REQUIRE(W_chi_lo > W_chi_hi);
  REQUIRE(tenryu::radiation::difference_reference_weight(W_max, 0.0, tau0, 0.0, chi0) ==
          Catch::Approx(0.0).margin(1.0e-15));

  const double eps = 1.0e-6;
  const double W_left = tenryu::radiation::difference_reference_weight(
      W_max, tau0 * (1.0 - eps), tau0, 0.1, chi0);
  const double W_right = tenryu::radiation::difference_reference_weight(
      W_max, tau0 * (1.0 + eps), tau0, 0.1, chi0);
  REQUIRE(std::abs(W_right - W_left) < 1.0e-6);
}

TEST_CASE("Difference reference cell opacity uses harmonic Rosseland mean",
          "[radiation][difference]") {
  const double sigma_low = 1.0;
  const double sigma_high = 100.0;
  const double weight_sum = 2.0;
  const double inverse_sigma_weight_sum = 1.0 / sigma_low + 1.0 / sigma_high;
  const double sigma_max = sigma_high;
  const double sigma_cell = tenryu::radiation::difference_reference_cell_sigma(
      weight_sum, inverse_sigma_weight_sum, sigma_max);

  REQUIRE(sigma_cell == Catch::Approx(2.0 / 1.01).epsilon(1.0e-14));
  REQUIRE(sigma_cell < 0.05 * sigma_max);

  const double W_harmonic =
      tenryu::radiation::difference_reference_weight(1.0, sigma_cell, 3.0, 0.0, 0.25);
  const double W_max_group =
      tenryu::radiation::difference_reference_weight(1.0, sigma_max, 3.0, 0.0, 0.25);
  REQUIRE(W_harmonic < 0.5 * W_max_group);
}

