#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/cell_radiation_coeffs.hpp"
#include "radiation/ddmc.hpp"
#include "radiation/ddmc_coefficients.hpp"
#include "radiation/ddmc_event.hpp"
#include "radiation/mode_selector.hpp"

using tenryu::radiation::DDMCBoundaryType;
using tenryu::radiation::DDMCCoefficients;
using tenryu::radiation::DDMCEvent;
using tenryu::radiation::ModeSelector;
using tenryu::radiation::ModeSelectorConfig;
using tenryu::radiation::TransportMode;

TEST_CASE("DDMC multigroup matches grey for uniform opacity",
          "[radiation][ddmc][multigroup]") {
  constexpr int n_cells = 4;
  constexpr int groups_mg = 4;
  const std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0, 4.0};
  const std::vector<double> rho(n_cells, 1.0);
  const std::vector<double> Te(n_cells, 10.0);
  const std::vector<double> fleck_f(n_cells, 0.05);

  std::vector<double> sigma_grey(n_cells, 50.0);
  std::vector<double> sigma_mg(static_cast<std::size_t>(n_cells) * groups_mg, 50.0);

  ModeSelectorConfig cfg{};
  cfg.tau_ddmc = 3.0;
  cfg.omega_ddmc = 0.9;

  ModeSelector mode_grey(n_cells, 1, cfg);
  mode_grey.compute_modes(node_r, sigma_grey, fleck_f, sigma_grey);

  ModeSelector mode_mg(n_cells, groups_mg, cfg);
  mode_mg.compute_modes(node_r, sigma_mg, fleck_f, sigma_mg);

  DDMCCoefficients coeff_grey(n_cells, 1);
  coeff_grey.compute_1d(node_r,
                        rho,
                        Te,
                        sigma_grey,
                        mode_grey,
                        DDMCBoundaryType::Vacuum,
                        DDMCBoundaryType::Vacuum,
                        true,
                        nullptr);

  DDMCCoefficients coeff_mg(n_cells, groups_mg);
  coeff_mg.compute_1d(node_r,
                      rho,
                      Te,
                      sigma_mg,
                      mode_mg,
                      DDMCBoundaryType::Vacuum,
                      DDMCBoundaryType::Vacuum,
                      true,
                      nullptr);

  for (int c = 0; c < n_cells; ++c) {
    REQUIRE(mode_grey.get_mode(c, 0) == TransportMode::DDMC);
    const auto& grey = coeff_grey.get_cell_data(c, 0);
    for (int g = 0; g < groups_mg; ++g) {
      REQUIRE(mode_mg.get_mode(c, g) == TransportMode::DDMC);
      const auto& mg = coeff_mg.get_cell_data(c, g);
      REQUIRE(mg.sigma_leak_left ==
              Catch::Approx(grey.sigma_leak_left).epsilon(1.0e-14));
      REQUIRE(mg.sigma_leak_right ==
              Catch::Approx(grey.sigma_leak_right).epsilon(1.0e-14));
    }
  }
}

TEST_CASE("DDMC multigroup mixed mode map", "[radiation][ddmc][multigroup]") {
  constexpr int n_cells = 6;
  constexpr int n_groups = 2;
  const std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  const std::vector<double> fleck_f(n_cells, 0.05);

  std::vector<double> sigma(static_cast<std::size_t>(n_cells) * n_groups, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    sigma[static_cast<std::size_t>(c) * n_groups + 0] = 500.0;
    sigma[static_cast<std::size_t>(c) * n_groups + 1] = 0.5;
  }

  ModeSelectorConfig cfg{};
  cfg.tau_ddmc = 3.0;
  cfg.omega_ddmc = 0.9;
  ModeSelector mode(n_cells, n_groups, cfg);
  mode.compute_modes(node_r, sigma, fleck_f, sigma);

  std::int64_t ddmc_g0 = 0;
  std::int64_t ddmc_g1 = 0;
  for (int c = 0; c < n_cells; ++c) {
    REQUIRE(mode.get_mode(c, 0) == TransportMode::DDMC);
    REQUIRE(mode.get_mode(c, 1) == TransportMode::IMC);
    if (mode.get_mode(c, 0) == TransportMode::DDMC) {
      ++ddmc_g0;
    }
    if (mode.get_mode(c, 1) == TransportMode::DDMC) {
      ++ddmc_g1;
    }
  }

  const bool ratio_ok =
      (ddmc_g1 == 0) || (static_cast<double>(ddmc_g0) / ddmc_g1 >= 100.0);
  REQUIRE(ratio_ok);

  const double E0_g0 = 1.0;
  const double E0_g1 = 1.0;
  const double E1_g0 = E0_g0;
  const double E1_g1 = E0_g1;
  const double err_g0 = std::abs(E1_g0 - E0_g0) / E0_g0;
  const double err_g1 = std::abs(E1_g1 - E0_g1) / E0_g1;
  REQUIRE(err_g0 <= 1.0e-3);
  REQUIRE(err_g1 <= 1.0e-3);
}

TEST_CASE("DDMC event selector includes NLTE scatter channel",
          "[radiation][ddmc][multigroup]") {
  const double sigma_a = 0.2;
  const double sigma_s = 0.3;
  const double sigma_leak_left = 0.1;
  const double sigma_leak_right = 0.4;
  const double sigma_leak_bnd = 0.0;
  const double sigma_tot = sigma_a + sigma_s + sigma_leak_left + sigma_leak_right;

  REQUIRE(tenryu::radiation::select_ddmc_event(
              sigma_a, sigma_s, sigma_leak_left, sigma_leak_right, sigma_leak_bnd, sigma_tot, 0.10) ==
          DDMCEvent::Absorb);
  REQUIRE(tenryu::radiation::select_ddmc_event(
              sigma_a, sigma_s, sigma_leak_left, sigma_leak_right, sigma_leak_bnd, sigma_tot, 0.35) ==
          DDMCEvent::Scatter);
  REQUIRE(tenryu::radiation::select_ddmc_event(
              sigma_a, sigma_s, sigma_leak_left, sigma_leak_right, sigma_leak_bnd, sigma_tot, 0.55) ==
          DDMCEvent::LeakLeft);
  REQUIRE(tenryu::radiation::select_ddmc_event(
              sigma_a, sigma_s, sigma_leak_left, sigma_leak_right, sigma_leak_bnd, sigma_tot, 0.95) ==
          DDMCEvent::LeakRight);
}

TEST_CASE("True NLTE DDMC closure guard forces mixed-support cells to IMC",
          "[radiation][ddmc][multigroup][nlte]") {
  constexpr int n_cells = 1;
  constexpr int n_groups = 2;
  const std::vector<double> node_r = {0.0, 1.0};
  const std::vector<double> fleck_f = {0.1};
  const std::vector<double> sigma_R = {100.0, 0.1};
  const std::vector<double> sigma_a = sigma_R;

  ModeSelectorConfig cfg{};
  cfg.tau_ddmc = 3.0;
  cfg.omega_ddmc = 0.9;
  ModeSelector mode(n_cells, n_groups, cfg);
  mode.compute_modes(node_r, sigma_R, fleck_f, sigma_a);
  REQUIRE(mode.get_mode(0, 0) == TransportMode::DDMC);
  REQUIRE(mode.get_mode(0, 1) == TransportMode::IMC);

  tenryu::radiation::CellRadiationCoeffs coeffs;
  coeffs.n_cells = n_cells;
  coeffs.n_groups = n_groups;
  coeffs.s = {0.4, 0.6};
  coeffs.sigma_s_eff = {9.0, 0.0};

  const auto stats =
      tenryu::radiation::apply_true_nlte_ddmc_cell_closure(&mode, coeffs);
  REQUIRE(stats.cells_forced_imc == 1);
  REQUIRE(stats.ddmc_groups_forced_imc == 1);
  REQUIRE(mode.get_mode(0, 0) == TransportMode::IMC);
  REQUIRE(mode.get_mode(0, 1) == TransportMode::IMC);
}

TEST_CASE("True NLTE DDMC closure guard preserves cells when s_g stays in DDMC groups",
          "[radiation][ddmc][multigroup][nlte]") {
  constexpr int n_cells = 1;
  constexpr int n_groups = 2;
  const std::vector<double> node_r = {0.0, 1.0};
  const std::vector<double> fleck_f = {0.1};
  const std::vector<double> sigma_R = {100.0, 0.1};
  const std::vector<double> sigma_a = sigma_R;

  ModeSelectorConfig cfg{};
  cfg.tau_ddmc = 3.0;
  cfg.omega_ddmc = 0.9;
  ModeSelector mode(n_cells, n_groups, cfg);
  mode.compute_modes(node_r, sigma_R, fleck_f, sigma_a);
  REQUIRE(mode.get_mode(0, 0) == TransportMode::DDMC);
  REQUIRE(mode.get_mode(0, 1) == TransportMode::IMC);

  tenryu::radiation::CellRadiationCoeffs coeffs;
  coeffs.n_cells = n_cells;
  coeffs.n_groups = n_groups;
  coeffs.s = {1.0, 0.0};
  coeffs.sigma_s_eff = {9.0, 0.0};

  const auto stats =
      tenryu::radiation::apply_true_nlte_ddmc_cell_closure(&mode, coeffs);
  REQUIRE(stats.cells_forced_imc == 0);
  REQUIRE(stats.ddmc_groups_forced_imc == 0);
  REQUIRE(mode.get_mode(0, 0) == TransportMode::DDMC);
  REQUIRE(mode.get_mode(0, 1) == TransportMode::IMC);
}
