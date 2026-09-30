#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/mode_selector.hpp"

using tenryu::radiation::ModeSelector;
using tenryu::radiation::ModeSelectorConfig;
using tenryu::radiation::TransportMode;

TEST_CASE("ModeSelector basic formulas", "[radiation][ddmc][mode_selector]") {
  REQUIRE(ModeSelector::compute_optical_depth(10.0, 0.5) ==
          Catch::Approx(5.0).epsilon(1.0e-14));
  REQUIRE(ModeSelector::compute_scattering_ratio(0.1) ==
          Catch::Approx(0.9).epsilon(1.0e-14));
  REQUIRE(ModeSelector::compute_scattering_ratio(0.1, 1.0, 0.0) ==
          Catch::Approx(0.9).epsilon(1.0e-14));
  REQUIRE(ModeSelector::compute_scattering_ratio(0.2, 4.0, 0.0) ==
          Catch::Approx(0.8).epsilon(1.0e-14));
  REQUIRE(ModeSelector::compute_cell_length(1.0, 2.5) ==
          Catch::Approx(1.5).epsilon(1.0e-14));
}

TEST_CASE("ModeSelector conversion probability constraint",
          "[radiation][ddmc][mode_selector]") {
  ModeSelectorConfig cfg{};
  cfg.emissivity_preserving = false;
  ModeSelector selector(1, 1, cfg);

  REQUIRE(selector.check_conversion_probability_constraint(10.0, 1.0, 0.95));
  REQUIRE(selector.check_conversion_probability_constraint(2.0, 1.0, 0.95));
  REQUIRE_FALSE(selector.check_conversion_probability_constraint(0.5, 1.0, 0.95));
}

TEST_CASE("ModeSelector mode thresholds and counters",
          "[radiation][ddmc][mode_selector]") {
  ModeSelectorConfig cfg{};
  cfg.tau_ddmc = 4.0;
  cfg.omega_ddmc = 0.9;
  ModeSelector selector(4, 1, cfg);

  const std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0, 4.0};
  const std::vector<double> sigma_R = {10.0, 10.0, 0.5, 10.0};
  const std::vector<double> sigma_a = sigma_R;
  const std::vector<double> fleck_f = {0.05, 0.50, 0.05, 0.05};

  selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);

  REQUIRE(selector.get_mode(0, 0) == TransportMode::DDMC);
  REQUIRE(selector.get_mode(1, 0) == TransportMode::IMC);   // omega too low
  REQUIRE(selector.get_mode(2, 0) == TransportMode::IMC);   // tau too low
  REQUIRE(selector.get_mode(3, 0) == TransportMode::DDMC);
  REQUIRE(selector.count_ddmc() == 2);
  REQUIRE(selector.count_imc() == 2);
  REQUIRE(selector.count_omega_below_threshold() >= 1);
}

TEST_CASE("ModeSelector sigma floor and degenerate cell handling",
          "[radiation][ddmc][mode_selector]") {
  SECTION("sigma below floor stays IMC") {
    ModeSelectorConfig cfg{};
    cfg.tau_ddmc = 0.0;
    cfg.omega_ddmc = 0.0;
    cfg.sigma_floor = 1.0e-20;
    ModeSelector selector(1, 1, cfg);

    const std::vector<double> node_r = {0.0, 1.0};
    const std::vector<double> sigma_R = {1.0e-30};
    const std::vector<double> sigma_a = {1.0e-30};
    const std::vector<double> fleck_f = {0.0};
    selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);

    REQUIRE(selector.get_mode(0, 0) == TransportMode::IMC);
  }

  SECTION("degenerate cell length falls back to IMC") {
    ModeSelectorConfig cfg{};
    cfg.tau_ddmc = 1.0;
    cfg.omega_ddmc = 0.0;
    ModeSelector selector(2, 1, cfg);

    const std::vector<double> node_r = {0.0, 0.0, 1.0};
    const std::vector<double> sigma_R = {10.0, 10.0};
    const std::vector<double> sigma_a = sigma_R;
    const std::vector<double> fleck_f = {0.0, 0.0};
    selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);

    REQUIRE(selector.cell_lengths()[0] == Catch::Approx(0.0).margin(1.0e-30));
    REQUIRE(selector.get_mode(0, 0) == TransportMode::IMC);
    REQUIRE(selector.get_mode(1, 0) == TransportMode::DDMC);
  }
}

TEST_CASE("ModeSelector hysteresis prevents chattering",
          "[radiation][ddmc][mode_selector]") {
  ModeSelectorConfig cfg{};
  cfg.tau_ddmc = 3.0;
  cfg.omega_ddmc = 0.9;
  cfg.tau_ddmc_off = 1.5;
  cfg.omega_ddmc_off = 0.9;

  ModeSelector selector(4, 1, cfg);
  const std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0, 4.0};

  std::vector<double> sigma_R = {4.0, 4.0, 4.0, 4.0};
  const std::vector<double> sigma_a = sigma_R;
  const std::vector<double> fleck_f = {0.05, 0.05, 0.05, 0.05};
  selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);

  std::vector<TransportMode> prev_mode(4, TransportMode::IMC);
  std::vector<double> prev_tau(4, 0.0);
  std::vector<std::uint8_t> hold_count(4, 0U);
  const auto first = selector.apply_hysteresis(prev_mode, prev_tau, hold_count);
  REQUIRE(first.switches_imc_to_ddmc == 4);
  for (std::int64_t c = 0; c < 4; ++c) {
    REQUIRE(selector.get_mode(c, 0) == TransportMode::DDMC);
  }

  prev_mode.assign(selector.modes().begin(), selector.modes().end());
  for (std::int64_t c = 0; c < 4; ++c) {
    prev_tau[static_cast<std::size_t>(c)] = selector.get_tau(c, 0);
  }

  sigma_R[1] = 2.0;  // between tau_ddmc_off and tau_ddmc
  selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);
  REQUIRE(selector.get_mode(1, 0) == TransportMode::IMC);
  selector.apply_hysteresis(prev_mode, prev_tau, hold_count);
  REQUIRE(selector.get_mode(1, 0) == TransportMode::DDMC);
}

TEST_CASE("ModeSelector hysteresis respects hold time",
          "[radiation][ddmc][mode_selector]") {
  ModeSelectorConfig cfg{};
  cfg.tau_ddmc = 3.0;
  cfg.omega_ddmc = 0.0;
  cfg.mode_hold = 2;

  ModeSelector selector(2, 1, cfg);
  const std::vector<double> node_r = {0.0, 1.0, 2.0};
  const std::vector<double> sigma_R = {4.0, 4.0};
  const std::vector<double> sigma_a = sigma_R;
  const std::vector<double> fleck_f = {0.0, 0.0};

  selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);
  std::vector<TransportMode> prev_mode = {TransportMode::IMC, TransportMode::DDMC};
  std::vector<double> prev_tau = {4.0, 4.0};
  std::vector<std::uint8_t> hold_count = {0U, 0U};

  selector.apply_hysteresis(prev_mode, prev_tau, hold_count);
  REQUIRE(selector.get_mode(0, 0) == TransportMode::IMC);

  hold_count[0] = 2U;
  selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);
  selector.apply_hysteresis(prev_mode, prev_tau, hold_count);
  REQUIRE(selector.get_mode(0, 0) == TransportMode::DDMC);
}

TEST_CASE("ModeSelector hysteresis exits on safety",
          "[radiation][ddmc][mode_selector]") {
  ModeSelectorConfig cfg{};
  cfg.tau_ddmc = 0.0;
  cfg.omega_ddmc = 0.0;
  cfg.sigma_floor = 1.0e-3;

  ModeSelector selector(1, 1, cfg);
  const std::vector<double> node_r = {0.0, 1.0};
  std::vector<double> sigma_R = {10.0};
  const std::vector<double> sigma_a = sigma_R;
  const std::vector<double> fleck_f = {0.0};

  selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);
  std::vector<TransportMode> prev_mode = {TransportMode::IMC};
  std::vector<double> prev_tau = {0.0};
  std::vector<std::uint8_t> hold_count = {0U};
  selector.apply_hysteresis(prev_mode, prev_tau, hold_count);
  REQUIRE(selector.get_mode(0, 0) == TransportMode::DDMC);

  prev_mode[0] = TransportMode::DDMC;
  prev_tau[0] = selector.get_tau(0, 0);
  sigma_R[0] = 1.0e-6;
  selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);
  const auto out = selector.apply_hysteresis(prev_mode, prev_tau, hold_count);
  REQUIRE(selector.get_mode(0, 0) == TransportMode::IMC);
  REQUIRE(out.switches_ddmc_to_imc == 1);
}

TEST_CASE("ModeSelector no hysteresis at defaults",
          "[radiation][ddmc][mode_selector]") {
  ModeSelectorConfig cfg{};
  ModeSelector baseline(4, 1, cfg);
  ModeSelector selector(4, 1, cfg);

  const std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0, 4.0};
  const std::vector<double> sigma_R = {6.0, 2.0, 8.0, 1.0};
  const std::vector<double> sigma_a = sigma_R;
  const std::vector<double> fleck_f = {0.05, 0.20, 0.05, 0.05};

  baseline.compute_modes(node_r, sigma_R, fleck_f, sigma_a);
  selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);

  std::vector<TransportMode> prev_mode = {TransportMode::DDMC,
                                          TransportMode::IMC,
                                          TransportMode::IMC,
                                          TransportMode::DDMC};
  std::vector<double> prev_tau(4, 0.0);
  for (std::int64_t c = 0; c < 4; ++c) {
    prev_tau[static_cast<std::size_t>(c)] = baseline.get_tau(c, 0);
  }
  std::vector<std::uint8_t> hold_count(4, 0U);

  selector.apply_hysteresis(prev_mode, prev_tau, hold_count);
  for (std::int64_t c = 0; c < 4; ++c) {
    REQUIRE(selector.get_mode(c, 0) == baseline.get_mode(c, 0));
  }
}
