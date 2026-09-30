// Retired with the Monte Carlo radiation on 2026-09-29: the test cases of tests/coupling/test_source_terms.cpp (as it
// was at 5bc8f6ce3) that exercised inject_radiation_source_terms (the Monte Carlo radiation's matter update, now in
// retired/radiation_monte_carlo/src/coupling/source_terms_radiation.cu), with the helper only they used. The
// file's other helpers (make_config, make_state, copy_field) stayed there. Not built.

namespace {

void set_rad_source(tenryu::core::State& state, const std::vector<double>& raw_delta_E) {
  REQUIRE(state.rad_dep.size() == raw_delta_E.size());
  const std::vector<double> rad_emit(raw_delta_E.size(), 0.0);
  state.rad_dep.copy_from_host(raw_delta_E.data());
  state.rad_emit.copy_from_host(rad_emit.data());
}

}  // namespace

TEST_CASE("Radiation source smoothing conserves and diffuses applied net source",
          "[coupling][source_terms]") {
  auto cfg = make_config(3);
  cfg.radiation.imc.net_e_source_smoothing.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.alpha = 0.2;
  cfg.radiation.imc.net_e_source_smoothing.tau_threshold = 4.0;

  auto state = make_state(cfg);
  const auto ee_before = copy_field(state.ee);
  const std::vector<double> raw_delta_E{0.0, 1.0, 0.0};
  const std::vector<double> sigma_R_max{5.0, 5.0, 5.0};
  set_rad_source(state, raw_delta_E);

  const double skipped =
      tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12,
                                                      nullptr, nullptr, &sigma_R_max);

  const auto ee_after = copy_field(state.ee);
  const auto applied = copy_field(state.delta_E_rad_prev);
  const std::vector<double> expected{0.2, 0.6, 0.2};

  REQUIRE(skipped == Catch::Approx(0.0));
  REQUIRE(applied.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(applied[i] == Catch::Approx(expected[i]).epsilon(1.0e-12));
    REQUIRE(ee_after[i] == Catch::Approx(ee_before[i] + expected[i]).epsilon(1.0e-12));
  }
  const double raw_sum = std::accumulate(raw_delta_E.begin(), raw_delta_E.end(), 0.0);
  const double applied_sum = std::accumulate(applied.begin(), applied.end(), 0.0);
  REQUIRE(applied_sum == Catch::Approx(raw_sum).epsilon(1.0e-12));
}

TEST_CASE("Radiation source smoothing applies multiple Jacobi passes",
          "[coupling][source_terms]") {
  auto cfg = make_config(3);
  cfg.radiation.imc.net_e_source_smoothing.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.alpha = 0.2;
  cfg.radiation.imc.net_e_source_smoothing.tau_threshold = 4.0;
  cfg.radiation.imc.net_e_source_smoothing.passes = 2;

  auto state = make_state(cfg);
  const std::vector<double> raw_delta_E{0.0, 1.0, 0.0};
  const std::vector<double> sigma_R_max{5.0, 5.0, 5.0};
  set_rad_source(state, raw_delta_E);

  tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12,
                                                  nullptr, nullptr, &sigma_R_max);

  const auto applied = copy_field(state.delta_E_rad_prev);
  const std::vector<double> expected{0.28, 0.44, 0.28};

  REQUIRE(applied.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(applied[i] == Catch::Approx(expected[i]).epsilon(1.0e-12));
  }
  const double raw_sum = std::accumulate(raw_delta_E.begin(), raw_delta_E.end(), 0.0);
  const double applied_sum = std::accumulate(applied.begin(), applied.end(), 0.0);
  REQUIRE(applied_sum == Catch::Approx(raw_sum).epsilon(1.0e-12));
}

TEST_CASE("Radiation source smoothing is blocked in active difference cells",
          "[coupling][source_terms][difference]") {
  auto cfg = make_config(3);
  cfg.radiation.imc.difference.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.alpha = 0.2;
  cfg.radiation.imc.net_e_source_smoothing.tau_threshold = 4.0;

  auto state = make_state(cfg);
  const std::vector<double> raw_delta_E{0.0, 1.0, 0.0};
  const std::vector<double> sigma_R_max{5.0, 5.0, 5.0};
  const std::vector<double> difference_W{0.0, 0.5, 0.0};
  state.difference_W.reset(difference_W.size());
  state.difference_W.copy_from_host(difference_W.data());
  set_rad_source(state, raw_delta_E);

  tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12,
                                                  nullptr, nullptr, &sigma_R_max);

  const auto applied = copy_field(state.delta_E_rad_prev);
  REQUIRE(applied == raw_delta_E);
}

TEST_CASE("Radiation source smoothing ignores gradients unless adaptive smoothing is enabled",
          "[coupling][source_terms]") {
  auto cfg = make_config(3);
  cfg.radiation.imc.net_e_source_smoothing.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.alpha = 0.2;
  cfg.radiation.imc.net_e_source_smoothing.tau_threshold = 4.0;

  auto state = make_state(cfg);
  const std::vector<double> Te{10.0, 20.0, 10.0};
  const std::vector<double> rho{1.0, 2.0, 1.0};
  state.Te.copy_from_host(Te.data());
  state.rho.copy_from_host(rho.data());
  const std::vector<double> raw_delta_E{0.0, 1.0, 0.0};
  const std::vector<double> sigma_R_max{5.0, 5.0, 5.0};
  set_rad_source(state, raw_delta_E);

  tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12,
                                                  nullptr, nullptr, &sigma_R_max);

  const auto applied = copy_field(state.delta_E_rad_prev);
  const std::vector<double> expected{0.2, 0.6, 0.2};

  REQUIRE(applied.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(applied[i] == Catch::Approx(expected[i]).epsilon(1.0e-12));
  }
}

TEST_CASE("Radiation source smoothing reuses adaptive alpha across Jacobi passes",
          "[coupling][source_terms]") {
  auto cfg = make_config(3);
  auto& smoothing = cfg.radiation.imc.net_e_source_smoothing;
  smoothing.enabled = true;
  smoothing.alpha = 0.2;
  smoothing.tau_threshold = 4.0;
  smoothing.passes = 2;
  smoothing.gradient_adaptive = true;
  smoothing.grad_Te_scale = 0.3;
  smoothing.grad_rho_scale = 0.5;

  auto state = make_state(cfg);
  const std::vector<double> Te{10.0, 10.0 * std::exp(smoothing.grad_Te_scale), 10.0};
  const std::vector<double> rho{1.0, std::exp(smoothing.grad_rho_scale), 1.0};
  state.Te.copy_from_host(Te.data());
  state.rho.copy_from_host(rho.data());
  const std::vector<double> raw_delta_E{0.0, 1.0, 0.0};
  const std::vector<double> sigma_R_max{5.0, 5.0, 5.0};
  set_rad_source(state, raw_delta_E);

  tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12,
                                                  nullptr, nullptr, &sigma_R_max);

  const double alpha_face = smoothing.alpha * std::exp(-2.0);
  const std::vector<double> expected{
      2.0 * alpha_face - 3.0 * alpha_face * alpha_face,
      1.0 - 4.0 * alpha_face + 6.0 * alpha_face * alpha_face,
      2.0 * alpha_face - 3.0 * alpha_face * alpha_face};
  const auto applied = copy_field(state.delta_E_rad_prev);

  REQUIRE(applied.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    REQUIRE(applied[i] == Catch::Approx(expected[i]).epsilon(1.0e-12));
  }
  const double raw_sum = std::accumulate(raw_delta_E.begin(), raw_delta_E.end(), 0.0);
  const double applied_sum = std::accumulate(applied.begin(), applied.end(), 0.0);
  REQUIRE(applied_sum == Catch::Approx(raw_sum).epsilon(1.0e-12));
}

TEST_CASE("Radiation source smoothing is disabled by tau gate in optically thin cells",
          "[coupling][source_terms]") {
  auto cfg = make_config(3);
  cfg.radiation.imc.net_e_source_smoothing.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.alpha = 0.2;
  cfg.radiation.imc.net_e_source_smoothing.tau_threshold = 4.0;

  auto state = make_state(cfg);
  const std::vector<double> raw_delta_E{0.0, 1.0, 0.0};
  const std::vector<double> sigma_R_max{1.0, 1.0, 1.0};
  set_rad_source(state, raw_delta_E);

  tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12, nullptr,
                                                  nullptr, &sigma_R_max);

  const auto applied = copy_field(state.delta_E_rad_prev);
  REQUIRE(applied == raw_delta_E);
}

TEST_CASE("Radiation source smoothing does not cross material interfaces",
          "[coupling][source_terms]") {
  auto cfg = make_config(3, 2);
  cfg.radiation.imc.net_e_source_smoothing.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.alpha = 0.2;
  cfg.radiation.imc.net_e_source_smoothing.tau_threshold = 4.0;

  auto state = make_state(cfg);
  std::vector<double> volfrac(static_cast<std::size_t>(cfg.mesh.nr) * 2, 0.0);
  volfrac[0] = 1.0;
  volfrac[3] = 1.0;
  volfrac[4] = 1.0;
  state.volFrac.copy_from_host(volfrac.data());

  const std::vector<double> raw_delta_E{0.0, 1.0, 0.0};
  const std::vector<double> sigma_R_max{5.0, 5.0, 5.0};
  set_rad_source(state, raw_delta_E);

  tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12, nullptr,
                                                  nullptr, &sigma_R_max);

  const auto applied = copy_field(state.delta_E_rad_prev);
  REQUIRE(applied == raw_delta_E);
}

TEST_CASE("Radiation source smoothing disabled path preserves raw source history",
          "[coupling][source_terms]") {
  auto cfg = make_config(3);
  cfg.radiation.imc.net_e_source_smoothing.enabled = false;

  auto state = make_state(cfg);
  const auto ee_before = copy_field(state.ee);
  const std::vector<double> raw_delta_E{0.0, 1.0, 0.0};
  set_rad_source(state, raw_delta_E);

  tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12);

  const auto ee_after = copy_field(state.ee);
  const auto applied = copy_field(state.delta_E_rad_prev);
  REQUIRE(applied == raw_delta_E);
  for (std::size_t i = 0; i < raw_delta_E.size(); ++i) {
    REQUIRE(ee_after[i] == Catch::Approx(ee_before[i] + raw_delta_E[i]).epsilon(1.0e-12));
  }
}

TEST_CASE("Radiation source injection skips diffusion cells before smoothing",
          "[coupling][source_terms][diffusion]") {
  auto cfg = make_config(3);
  cfg.radiation.imc.net_e_source_smoothing.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.alpha = 0.2;
  cfg.radiation.imc.net_e_source_smoothing.tau_threshold = 4.0;

  auto state = make_state(cfg);
  const auto ee_before = copy_field(state.ee);
  set_rad_source(state, {0.0, 1.0, 0.0});
  state.ddmc_mode_map = {0, 3, 0};
  state.ddmc_mode_map_valid = true;
  const std::vector<double> sigma_R_max{5.0, 5.0, 5.0};

  const double skipped =
      tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12,
                                                      nullptr, nullptr, &sigma_R_max);

  const auto ee_after = copy_field(state.ee);
  const auto applied = copy_field(state.delta_E_rad_prev);
  REQUIRE(skipped == Catch::Approx(0.0));
  REQUIRE(applied == std::vector<double>({0.0, 0.0, 0.0}));
  REQUIRE(ee_after == ee_before);
}

TEST_CASE("HOLO core source ownership keeps particle tallies diagnostic",
          "[coupling][source_terms][holo]") {
  auto cfg = make_config(3);
  cfg.radiation.holo.enabled = true;

  auto state = make_state(cfg);
  const auto ee_before = copy_field(state.ee);
  set_rad_source(state, {1.0, 100.0, 3.0});
  const std::vector<double> holo_dep{0.0, 7.0, 0.0};
  const std::vector<double> holo_emit{0.0, 2.0, 0.0};
  state.holo_rad_dep.copy_from_host(holo_dep.data());
  state.holo_rad_emit.copy_from_host(holo_emit.data());
  state.holo_core_mask = {0U, 1U, 0U};
  state.holo_core_mask_valid = true;

  const double skipped =
      tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12);

  const auto ee_after = copy_field(state.ee);
  const auto applied = copy_field(state.delta_E_rad_prev);
  REQUIRE(skipped == Catch::Approx(0.0));
  REQUIRE(applied == std::vector<double>({1.0, 5.0, 3.0}));
  REQUIRE(ee_after[0] == Catch::Approx(ee_before[0] + 1.0).epsilon(1.0e-12));
  REQUIRE(ee_after[1] == Catch::Approx(ee_before[1]).epsilon(1.0e-12));
  REQUIRE(ee_after[2] == Catch::Approx(ee_before[2] + 3.0).epsilon(1.0e-12));
}

TEST_CASE("HOLO blend source applies weighted LO and particle source without smoothing",
          "[coupling][source_terms][holo]") {
  auto cfg = make_config(3);
  cfg.radiation.holo.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.enabled = true;
  cfg.radiation.imc.net_e_source_smoothing.alpha = 0.2;
  cfg.radiation.imc.net_e_source_smoothing.tau_threshold = 4.0;

  auto state = make_state(cfg);
  const auto ee_before = copy_field(state.ee);
  set_rad_source(state, {0.0, 100.0, 0.0});
  const std::vector<double> holo_dep{0.0, 20.0, 0.0};
  const std::vector<double> holo_emit{0.0, 0.0, 0.0};
  state.holo_rad_dep.copy_from_host(holo_dep.data());
  state.holo_rad_emit.copy_from_host(holo_emit.data());
  state.holo_core_mask = {0U, 0U, 0U};
  state.holo_patch_mask = {0U, 1U, 0U};
  state.holo_lo_weight = {0.0, 0.25, 0.0};
  state.holo_core_mask_valid = true;
  state.holo_lo_source_valid = true;
  const std::vector<double> sigma_R_max{5.0, 5.0, 5.0};

  const double skipped =
      tenryu::coupling::inject_radiation_source_terms(state, cfg, 1.0e-12,
                                                      nullptr, nullptr, &sigma_R_max);

  const auto ee_after = copy_field(state.ee);
  const auto applied = copy_field(state.delta_E_rad_prev);
  REQUIRE(skipped == Catch::Approx(0.0));
  REQUIRE(applied == std::vector<double>({0.0, 80.0, 0.0}));
  REQUIRE(ee_after[0] == Catch::Approx(ee_before[0]).epsilon(1.0e-12));
  REQUIRE(ee_after[1] == Catch::Approx(ee_before[1] + 80.0).epsilon(1.0e-12));
  REQUIRE(ee_after[2] == Catch::Approx(ee_before[2]).epsilon(1.0e-12));
}
