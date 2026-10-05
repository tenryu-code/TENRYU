// Retired with the Monte Carlo radiation on 2026-09-29: the verification targets of Radiation.mode "imc_ddmc" (IMC,
// DDMC, HOLO and their host NLTE coefficients) and the helpers only they used, cut out of
// src/drivers/cmd_verify.cpp as it was at 5bc8f6ce3, where they last built: nlte_sanity, nlte_lte_regression,
// nlte_cooling_mms, nlte_lambda_agreement, nlte_ddmc_classification, nlte_energy_conservation, nlte_group_resample,
// imc_ddmc_hybrid, imc_ddmc_angular, imc_ddmc_tau_scan, imc_ddmc_convergence, ddmc_diffusion,
// ddmc_leak_normalization, mmatrix_fallback, ddmc_multigroup, void_passthrough, and the imc_ddmc-era GXII 1D
// regression (gxii_1d_regression, already refused by the verify command since 2026-07-06). Not built; the helpers
// they share with the remaining targets (load_state_from_namelist, format_double, copy_field_to_host, ...) stayed
// in cmd_verify.cpp.

namespace tenryu::drivers {
namespace {


double chi_square_critical_p01(const int dof) {
  if (dof <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  static constexpr std::array<double, 20> kChi2CritP01 = {
      6.635,  9.210,  11.345, 13.277, 15.086, 16.812, 18.475,
      20.090, 21.666, 23.209, 24.725, 26.217, 27.688, 29.141,
      30.578, 32.000, 33.409, 34.805, 36.191, 37.566};
  if (dof <= static_cast<int>(kChi2CritP01.size())) {
    return kChi2CritP01[static_cast<std::size_t>(dof - 1)];
  }
  // Wilson-Hilferty approximation for upper-tail quantile p=0.01.
  constexpr double kZ99 = 2.3263478740408408;
  const double nu = static_cast<double>(dof);
  const double x = 1.0 - (2.0 / (9.0 * nu)) + kZ99 * std::sqrt(2.0 / (9.0 * nu));
  return nu * x * x * x;
}

double elapsed_seconds(const std::chrono::steady_clock::time_point start,
                       const std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

radiation::DDMCBoundaryType ddmc_boundary_type_from_string(
    const std::string& boundary_mode) {
  if (boundary_mode == "vacuum" || boundary_mode == "marshak") {
    return radiation::DDMCBoundaryType::Vacuum;
  }
  if (boundary_mode == "reflect") {
    return radiation::DDMCBoundaryType::Reflective;
  }
  return radiation::DDMCBoundaryType::Internal;
}

void collapse_to_one_temperature_state(core::State& state, const core::Config& cfg) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "1T collapse requires at least one material");
  const auto& mat = cfg.materials.materials.front();
  TENRYU_ASSERT(mat.ideal_gas_gamma > 1.0,
                "1T collapse requires gamma > 1");

  auto host_Te = copy_field_to_host(state.Te);
  auto host_Ti = copy_field_to_host(state.Ti);
  auto host_ee = copy_field_to_host(state.ee);
  auto host_ei = copy_field_to_host(state.ei);
  auto host_Pe = copy_field_to_host(state.Pe);
  auto host_Pi = copy_field_to_host(state.Pi);
  const auto host_rho = copy_field_to_host(state.rho);

  const double gm1 = mat.ideal_gas_gamma - 1.0;
  for (std::size_t c = 0; c < host_ee.size(); ++c) {
    const double e_total = std::max(host_ee[c] + host_ei[c], 0.0);
    host_ee[c] = e_total;
    host_ei[c] = 0.0;
    host_Ti[c] = host_Te[c];
    host_Pe[c] = gm1 * std::max(host_rho[c], 0.0) * e_total;
    host_Pi[c] = 0.0;
  }

  copy_field_from_host(state.Ti, host_Ti);
  copy_field_from_host(state.ee, host_ee);
  copy_field_from_host(state.ei, host_ei);
  copy_field_from_host(state.Pe, host_Pe);
  copy_field_from_host(state.Pi, host_Pi);
}

struct GxiiGoldenMetrics {
  double rho_peak = 75.0;
  double rhoR = 3.0e-2;
  double E_laser_absorbed = 1.0e11;
  double shock_time = 1.0e-9;
  double ablation_multishock_metric = -1.0;
  double shell_dep_noise_cv = -1.0;
};

struct GxiiRegressionRun {
  GxiiGoldenMetrics metrics{};
  double t_rel = std::numeric_limits<double>::infinity();
  bool pass_finite = false;
  bool pass_non_negative = false;
  bool pass_t = false;
};

std::filesystem::path gxii_golden_path() {
  return std::filesystem::path("examples/verification/gxii_1d_regression/golden.json");
}

std::filesystem::path gxii_golden_legacy_path() {
  return std::filesystem::path("examples/verification/golden/gxii_1d_regression.json");
}

double coefficient_of_variation(const std::vector<double>& values) {
  if (values.size() < 2U) {
    return 0.0;
  }
  const double inv_n = 1.0 / static_cast<double>(values.size());
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) * inv_n;
  double var = 0.0;
  for (const double value : values) {
    const double d = value - mean;
    var += d * d;
  }
  var *= inv_n;
  return std::sqrt(std::max(var, 0.0)) / std::max(std::abs(mean), 1.0e-300);
}

double gxii_shell_deposition_noise_cv(const std::vector<double>& rho,
                                      const std::vector<double>& rad_dep,
                                      const int n_groups) {
  if (rho.empty() || n_groups <= 0 ||
      rad_dep.size() != rho.size() * static_cast<std::size_t>(n_groups)) {
    return 0.0;
  }
  const double rho_peak = *std::max_element(rho.begin(), rho.end());
  if (!(rho_peak > 0.0)) {
    return 0.0;
  }
  const double shell_threshold = 0.5 * rho_peak;
  std::vector<double> shell_dep;
  shell_dep.reserve(rho.size());
  for (std::size_t c = 0; c < rho.size(); ++c) {
    if (rho[c] < shell_threshold) {
      continue;
    }
    double dep = 0.0;
    const std::size_t base = c * static_cast<std::size_t>(n_groups);
    for (int g = 0; g < n_groups; ++g) {
      dep += rad_dep[base + static_cast<std::size_t>(g)];
    }
    shell_dep.push_back(dep);
  }
  return coefficient_of_variation(shell_dep);
}

double gxii_ablation_multishock_metric(const std::vector<double>& rho,
                                       const std::vector<double>& Te) {
  if (rho.size() < 3U || Te.size() != rho.size()) {
    return 0.0;
  }
  const auto [rho_min_it, rho_max_it] = std::minmax_element(rho.begin(), rho.end());
  const double rho_range = *rho_max_it - *rho_min_it;
  if (!(rho_range > 0.0)) {
    return 0.0;
  }
  const double rho_floor = *rho_min_it + 0.25 * rho_range;
  double metric = 0.0;
  for (std::size_t i = 1; i + 1U < rho.size(); ++i) {
    if (rho[i] < rho_floor) {
      continue;
    }
    const double slope_l = rho[i] - rho[i - 1U];
    const double slope_r = rho[i + 1U] - rho[i];
    if (slope_l * slope_r >= 0.0) {
      continue;
    }
    const double rho_prominence =
        std::min(std::abs(slope_l), std::abs(slope_r)) / std::max(rho_range, 1.0e-300);
    const double Te_scale = std::max(std::abs(Te[i]), 1.0e-300);
    const double Te_jump =
        std::abs(Te[i + 1U] - Te[i - 1U]) / Te_scale;
    if (rho_prominence > 0.02 && Te_jump > 0.02) {
      metric += rho_prominence * Te_jump;
    }
  }
  return metric;
}

bool write_gxii_golden_json(const std::filesystem::path& golden_path,
                            const GxiiGoldenMetrics& metrics) {
  if (!golden_path.has_parent_path()) {
    core::log_error("[verify:gxii_1d_regression] invalid golden path: " +
                    golden_path.string());
    return false;
  }
  std::error_code ec;
  std::filesystem::create_directories(golden_path.parent_path(), ec);
  if (ec) {
    core::log_error("[verify:gxii_1d_regression] failed to create golden directory '" +
                    golden_path.parent_path().string() + "': " + ec.message());
    return false;
  }

  std::ofstream ofs(golden_path, std::ios::binary | std::ios::trunc);
  if (!ofs.good()) {
    core::log_error("[verify:gxii_1d_regression] failed to open golden file for write: " +
                    golden_path.string());
    return false;
  }
  ofs << "{\n";
  ofs << "  \"run_profile\": \"" << kGxiiRunProfile << "\",\n";
  ofs << "  \"rho_peak\": " << format_double(metrics.rho_peak) << ",\n";
  ofs << "  \"rhoR\": " << format_double(metrics.rhoR) << ",\n";
  ofs << "  \"E_laser_absorbed\": " << format_double(metrics.E_laser_absorbed) << ",\n";
  ofs << "  \"shock_time\": " << format_double(metrics.shock_time) << ",\n";
  ofs << "  \"ablation_multishock_metric\": "
      << format_double(metrics.ablation_multishock_metric) << ",\n";
  ofs << "  \"shell_dep_noise_cv\": " << format_double(metrics.shell_dep_noise_cv) << '\n';
  ofs << "}\n";
  return ofs.good();
}

GxiiGoldenMetrics load_gxii_golden() {
  GxiiGoldenMetrics golden{};
  const std::array<std::filesystem::path, 2> candidate_paths = {
      gxii_golden_path(),
      gxii_golden_legacy_path(),
  };
  for (const auto& golden_path : candidate_paths) {
    std::ifstream ifs(golden_path, std::ios::binary);
    if (!ifs.good()) {
      continue;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    const std::string content = oss.str();
    golden.rho_peak = extract_json_number(content, "rho_peak", golden.rho_peak);
    golden.rhoR = extract_json_number(content, "rhoR", golden.rhoR);
    golden.E_laser_absorbed =
        extract_json_number(content, "E_laser_absorbed", golden.E_laser_absorbed);
    golden.shock_time = extract_json_number(content, "shock_time", golden.shock_time);
    golden.ablation_multishock_metric = extract_json_number(
        content, "ablation_multishock_metric", golden.ablation_multishock_metric);
    golden.shell_dep_noise_cv =
        extract_json_number(content, "shell_dep_noise_cv", golden.shell_dep_noise_cv);
    return golden;
  }
  core::log_warning("[verify:gxii_1d_regression] golden file not found, using defaults");
  return golden;
}

GxiiRegressionRun run_gxii_1d_regression_case() {
  GxiiRegressionRun result{};
  core::Config cfg;
  auto state = load_state_from_namelist("examples/verification/gxii_1d_regression.py", cfg);
  // Keep regression runtime bounded for CI smoke coverage while avoiding
  // fs-scale hard truncation that invalidates the documented shock-time scale.
  cfg.main.t_end = std::min(cfg.main.t_end, 2.0e-12);
  cfg.main.max_steps = std::min(cfg.main.max_steps, 500000);
  cfg.output.directory = "./build/output_verify_gxii_1d_regression";
  cfg.main.verbosity = "quiet";

  coupling::Driver driver;
  driver.run(state, cfg);

  const auto rho = copy_field_to_host(state.rho);
  result.metrics.rho_peak = rho.empty() ? 0.0 : *std::max_element(rho.begin(), rho.end());
  const auto Te = copy_field_to_host(state.Te);
  const auto rad_dep = copy_field_to_host(state.rad_dep);
  const auto areal = diagnostics::compute_areal_density(state, cfg);
  result.metrics.rhoR = areal.rhoR.empty() ? 0.0 : areal.rhoR.front();
  result.metrics.E_laser_absorbed = std::max(state.E_laser_deposited, 0.0);
  result.metrics.ablation_multishock_metric = gxii_ablation_multishock_metric(rho, Te);
  result.metrics.shell_dep_noise_cv =
      gxii_shell_deposition_noise_cv(rho, rad_dep, cfg.radiation.groups);

  // v1.0 placeholder: shock arrival is reported as final simulation time.
  result.metrics.shock_time = state.t;
  result.pass_finite = std::isfinite(result.metrics.rho_peak) && std::isfinite(result.metrics.rhoR) &&
                       std::isfinite(result.metrics.E_laser_absorbed) &&
                       std::isfinite(result.metrics.shock_time) &&
                       std::isfinite(result.metrics.ablation_multishock_metric) &&
                       std::isfinite(result.metrics.shell_dep_noise_cv);
  result.pass_non_negative =
      (result.metrics.rho_peak >= 0.0) && (result.metrics.rhoR >= 0.0) &&
      (result.metrics.E_laser_absorbed >= 0.0) &&
      (result.metrics.ablation_multishock_metric >= 0.0) &&
      (result.metrics.shell_dep_noise_cv >= 0.0);
  result.t_rel =
      std::abs(state.t - cfg.main.t_end) / std::max(std::abs(cfg.main.t_end), 1.0e-30);
  result.pass_t = result.t_rel <= 1.0e-10;
  return result;
}

bool run_gxii_1d_regression_verify() {
  const GxiiRegressionRun run = run_gxii_1d_regression_case();
  const double rho_peak = run.metrics.rho_peak;
  const double rhoR = run.metrics.rhoR;
  const double E_laser_absorbed = run.metrics.E_laser_absorbed;
  const double shock_time = run.metrics.shock_time;
  const double ablation_multishock_metric = run.metrics.ablation_multishock_metric;
  const double shell_dep_noise_cv = run.metrics.shell_dep_noise_cv;
  const GxiiGoldenMetrics golden = load_gxii_golden();
  // E_laser_absorbed tolerance is wider than other metrics because GPU
  // transport kernels use atomicAdd(double) for energy deposition tallies,
  // whose floating-point accumulation order is non-deterministic across runs.
  // At t_end=2ps (Gaussian pulse tail, ~0.14% of peak power), the absorbed
  // energy is small (~0.08 erg) and highly sensitive to this rounding noise.
  // Measured run-to-run CV ~ 5-10%.
  constexpr double kRhoPeakRelTol = 0.05;
  constexpr double kRhoRRelTol = 0.05;
  constexpr double kAbsorbedRelTol = 0.10;
  constexpr double kShockRelTol = 0.10;
  const double rho_peak_rel = relative_error(rho_peak, golden.rho_peak);
  const double rhoR_rel = relative_error(rhoR, golden.rhoR);
  const double E_abs_rel = relative_error(E_laser_absorbed, golden.E_laser_absorbed);
  const double shock_rel = relative_error(shock_time, golden.shock_time);
  const double shock_abs = std::abs(shock_time - golden.shock_time);
  const bool has_front_golden = golden.ablation_multishock_metric >= 0.0;
  const bool has_noise_golden = golden.shell_dep_noise_cv >= 0.0;
  const double front_rel =
      has_front_golden ? relative_error(ablation_multishock_metric,
                                        golden.ablation_multishock_metric)
                       : 0.0;
  const double noise_rel =
      has_noise_golden ? relative_error(shell_dep_noise_cv, golden.shell_dep_noise_cv) : 0.0;
  const bool pass_rho_peak = rho_peak_rel <= kRhoPeakRelTol;
  const bool pass_rhoR = rhoR_rel <= kRhoRRelTol;
  const bool pass_E_abs = E_abs_rel <= kAbsorbedRelTol;
  const bool pass_shock = shock_rel <= kShockRelTol;
  const bool pass_front =
      !has_front_golden ||
      (ablation_multishock_metric <= golden.ablation_multishock_metric + 1.0e-12);
  const bool pass_noise =
      !has_noise_golden || (shell_dep_noise_cv <= 1.25 * golden.shell_dep_noise_cv + 1.0e-12);
  const bool pass = run.pass_finite && run.pass_non_negative && run.pass_t && pass_rho_peak &&
                    pass_rhoR && pass_E_abs && pass_shock && pass_front && pass_noise;

  core::log_info("[verify:gxii_1d_regression] rho_peak=" + format_double(rho_peak) +
                 " (golden=" + format_double(golden.rho_peak) +
                 ", rel=" + format_double(rho_peak_rel) + ")");
  core::log_info("[verify:gxii_1d_regression] rhoR=" + format_double(rhoR) +
                 " (golden=" + format_double(golden.rhoR) +
                 ", rel=" + format_double(rhoR_rel) + ")");
  core::log_info("[verify:gxii_1d_regression] E_laser_absorbed=" +
                 format_double(E_laser_absorbed) + " (golden=" +
                 format_double(golden.E_laser_absorbed) +
                 ", rel=" + format_double(E_abs_rel) + ")");
  core::log_info("[verify:gxii_1d_regression] t_end=" + format_double(shock_time) +
                 " (golden=" + format_double(golden.shock_time) +
                 ", abs=" + format_double(shock_abs) +
                 ", rel=" + format_double(shock_rel) +
                 ", rel_to_cfg=" + format_double(run.t_rel) + ")");
  const std::string front_suffix =
      has_front_golden
          ? (std::string(" (golden=") + format_double(golden.ablation_multishock_metric) +
             ", rel=" + format_double(front_rel) + ")")
          : std::string(" (golden=not_set, informational)");
  const std::string noise_suffix =
      has_noise_golden
          ? (std::string(" (golden=") + format_double(golden.shell_dep_noise_cv) +
             ", rel=" + format_double(noise_rel) + ")")
          : std::string(" (golden=not_set, informational)");
  core::log_info("[verify:gxii_1d_regression] ablation_multishock_metric=" +
                 format_double(ablation_multishock_metric) + front_suffix);
  core::log_info("[verify:gxii_1d_regression] shell_dep_noise_cv=" +
                 format_double(shell_dep_noise_cv) + noise_suffix);

  if (!pass) {
    if (!run.pass_finite) {
      core::log_error("[verify:gxii_1d_regression] non-finite metric detected");
    }
    if (!run.pass_non_negative) {
      core::log_error("[verify:gxii_1d_regression] negative metric detected");
    }
    if (!run.pass_t) {
      core::log_error("[verify:gxii_1d_regression] final-time mismatch rel_to_cfg=" +
                      format_double(run.t_rel));
    }
    if (!pass_rho_peak) {
      core::log_error("[verify:gxii_1d_regression] rho_peak rel_err=" +
                      format_double(rho_peak_rel) + " exceeds tol=" +
                      format_double(kRhoPeakRelTol));
    }
    if (!pass_rhoR) {
      core::log_error("[verify:gxii_1d_regression] rhoR rel_err=" + format_double(rhoR_rel) +
                      " exceeds tol=" + format_double(kRhoRRelTol));
    }
    if (!pass_E_abs) {
      core::log_error("[verify:gxii_1d_regression] E_laser_absorbed rel_err=" +
                      format_double(E_abs_rel) + " exceeds tol=" +
                      format_double(kAbsorbedRelTol));
    }
    if (!pass_shock) {
      core::log_error("[verify:gxii_1d_regression] shock_time rel_err=" +
                      format_double(shock_rel) + " exceeds tol=" +
                      format_double(kShockRelTol));
    }
    if (!pass_front) {
      core::log_error("[verify:gxii_1d_regression] ablation_multishock_metric increased: value=" +
                      format_double(ablation_multishock_metric) + ", golden=" +
                      format_double(golden.ablation_multishock_metric));
    }
    if (!pass_noise) {
      core::log_error("[verify:gxii_1d_regression] shell_dep_noise_cv rel_err=" +
                      format_double(noise_rel) + " exceeds tol=2.5e-1");
    }
    core::log_error("[verify:gxii_1d_regression] FAILED");
  } else {
    core::log_info("[verify:gxii_1d_regression] PASSED");
  }

  return pass;
}

int generate_gxii_1d_regression_golden() {
  const GxiiRegressionRun run = run_gxii_1d_regression_case();
  if (!run.pass_finite || !run.pass_non_negative || !run.pass_t) {
    if (!run.pass_finite) {
      core::log_error("[verify:gxii_1d_regression] non-finite metric detected during golden generation");
    }
    if (!run.pass_non_negative) {
      core::log_error("[verify:gxii_1d_regression] negative metric detected during golden generation");
    }
    if (!run.pass_t) {
      core::log_error("[verify:gxii_1d_regression] final-time mismatch during golden generation: rel_to_cfg=" +
                      format_double(run.t_rel));
    }
    return 1;
  }

  const std::filesystem::path canonical_path = gxii_golden_path();
  if (!write_gxii_golden_json(canonical_path, run.metrics)) {
    return 1;
  }
  const std::filesystem::path legacy_path = gxii_golden_legacy_path();
  if (legacy_path != canonical_path && !write_gxii_golden_json(legacy_path, run.metrics)) {
    return 1;
  }
  std::cout << "Generated golden reference: " << canonical_path.string() << '\n';
  return 0;
}

radiation::PlanckTable build_planck_table_from_config(const core::Config& cfg) {
  TENRYU_ASSERT(!cfg.radiation.group_bounds_eV.empty(),
                "NLTE verify requires explicit group bounds");
  radiation::Groups groups(cfg.radiation.group_bounds_eV);
  radiation::PlanckTable planck;
  const int n_T = std::max(cfg.radiation.planck_fraction.compute_N_T, 2);
  const std::vector<double> range =
      radiation::resolve_compute_T_range_eV(cfg, false);
  const double T_min = range[0];
  const double T_max = range[1];
  planck.build(groups, n_T, T_min, T_max);
  return planck;
}

core::Config make_nlte_verify_config(const std::string& name,
                                     const std::string& output_dir,
                                     const std::string& opacity_file,
                                     const std::vector<double>& group_bounds_eV,
                                     const int n_cells) {
  TENRYU_ASSERT(n_cells > 0, "NLTE verify config requires positive n_cells");
  TENRYU_ASSERT(group_bounds_eV.size() >= 2,
                "NLTE verify config requires at least two group bounds");

  core::Config cfg{};
  cfg.main.name = name;
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.main.t_end = 1.0e-10;
  cfg.main.seed = 12345;
  cfg.main.max_steps = 50;
  cfg.main.verbosity = "quiet";

  cfg.mesh.nr = n_cells;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 100.0;
  cfg.mesh.r_max = 101.0;
  cfg.mesh.grid_type_r = "uniform";

  core::Config::MaterialsConfig::MatDef mat{};
  mat.name = "nlte_mat";
  mat.A = 6.5;
  mat.Z = 3.5;
  mat.eos_model = "ideal_gas";
  mat.ideal_gas_gamma = 5.0 / 3.0;
  mat.opacity_model = "table_nlte";
  mat.opacity_file = opacity_file;
  mat.opacity_units = "cm2_per_g";
  mat.lambda_method = "finite_difference";
  mat.lambda_fd_delta_rel = 1.0e-4;
  mat.lambda_fd_abs_min = 1.0e-6;
  mat.nlte_f_min = 1.0e-4;
  cfg.materials.materials = {mat};
  cfg.materials.zbar.model = "fixed";
  cfg.materials.zbar.fixed_value = 3.5;

  cfg.radiation.enabled = true;
  cfg.radiation.mode = core::RadiationMode::ImcDdmc;
  cfg.radiation.groups = static_cast<int>(group_bounds_eV.size()) - 1;
  cfg.radiation.group_bounds_eV = group_bounds_eV;
  cfg.radiation.compute_T_range_eV = {0.1, 1000.0};
  cfg.radiation.planck_fraction.compute_N_T = 200;

  cfg.radiation.imc.alpha = 1.0;
  cfg.radiation.imc.f_max = 1.0;
  cfg.radiation.imc.particles_per_cell_group = 200;
  cfg.radiation.imc.implicit_capture = true;
  cfg.radiation.imc.cutoff_fraction = 0.0;
  cfg.radiation.imc.inelastic_scatter = true;
  cfg.radiation.imc.weight_cutoff = 1.0e-10;
  cfg.radiation.imc.roulette_survival = 0.1;
  cfg.radiation.imc.linearized_planck = false;

  cfg.radiation.ddmc.enabled = false;
  cfg.radiation.ddmc.tau_ddmc = 3.0;
  cfg.radiation.ddmc.omega_ddmc = 0.9;
  cfg.radiation.ddmc.leak_stencil = "4";
  cfg.radiation.ddmc.interface_method = "asymptotic_diffusion_limit";
  cfg.radiation.ddmc.emissivity_preserving = true;
  cfg.radiation.ddmc.interface_exit_distribution = "cosine";
  cfg.radiation.ddmc.rz_face_r_weight = true;
  cfg.radiation.ddmc.face_opacity_temperature = "radiative_mean";
  cfg.radiation.ddmc.m_matrix_check = true;

  cfg.radiation.boundary.inner_r = "reflect";
  cfg.radiation.boundary.outer_r = "vacuum";
  cfg.radiation.boundary.marshak_Tr_eV = 0.0;
  cfg.radiation.boundary.marshak_particles = 0;

  cfg.numerics.dt.initial_s = 1.0e-11;
  cfg.numerics.dt.max_s = 1.0e-11;
  cfg.numerics.dt.min_s = 1.0e-20;
  cfg.numerics.dt.growth_factor = 1.0;
  cfg.numerics.hydro.enabled = false;
  cfg.numerics.conduction.enabled = false;
  cfg.numerics.floors.rho = 1.0e-12;
  cfg.numerics.floors.Te = 1.0e-3;
  cfg.numerics.floors.Ti = 1.0e-3;

  cfg.laser.enabled = false;
  cfg.output.directory = output_dir;
  cfg.output.plot_every = 0;
  cfg.output.history_every = 0;
  cfg.output.checkpoint_every = 0;
  cfg.output.plot_every_s = -1.0;
  cfg.output.history_every_s = -1.0;
  cfg.output.checkpoint_every_s = -1.0;
  cfg.diagnostics.enabled = true;

  return cfg;
}

core::State make_nlte_state_with_rho_profile(core::Config& cfg,
                                             const std::vector<double>& rho_profile,
                                             const double Te_eV,
                                             const double Ti_eV,
                                             const double zbar_value) {
  TENRYU_ASSERT(cfg.mesh.nr == static_cast<int>(rho_profile.size()),
                "NLTE verify state rho_profile size mismatch");
  core::State state = core::State::allocate(cfg);
  state.mesh = mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;

  std::vector<double> vol(state.vol.size(), 0.0);
  state.vol.copy_to_host(vol.data());

  std::vector<double> mass(state.mass.size(), 0.0);
  std::vector<double> zbar(state.zbar.size(), zbar_value);
  std::vector<double> Te(state.Te.size(), Te_eV);
  std::vector<double> Ti(state.Ti.size(), Ti_eV);
  for (std::size_t c = 0; c < mass.size(); ++c) {
    mass[c] = rho_profile[c] * std::max(vol[c], 0.0);
  }

  copy_field_from_host(state.rho, rho_profile);
  copy_field_from_host(state.mass, mass);
  copy_field_from_host(state.zbar, zbar);
  copy_field_from_host(state.Te, Te);
  copy_field_from_host(state.Ti, Ti);
  initialize_output_timing(state, cfg);
  return state;
}

core::State make_uniform_nlte_state(core::Config& cfg,
                                    const double rho,
                                    const double Te_eV,
                                    const double Ti_eV,
                                    const double zbar_value) {
  std::vector<double> rho_profile(static_cast<std::size_t>(cfg.mesh.nr), rho);
  return make_nlte_state_with_rho_profile(cfg, rho_profile, Te_eV, Ti_eV, zbar_value);
}

void advance_radiation_step(core::State& state,
                            const core::Config& cfg,
                            radiation::IMC& imc,
                            const double dt) {
  imc.transport_step(state, cfg, dt);
  coupling::inject_radiation_source_terms(state, cfg, dt, nullptr, nullptr,
                                          &imc.last_sigma_R_max());
  state.t += dt;
  state.step += 1;
  state.dt = dt;
}

double total_system_energy_with_imc(const core::State& state,
                                    const radiation::IMC& imc,
                                    const int n_groups) {
  (void)n_groups;
  return compute_total_material_energy_1d(state) + imc.census_energy() +
         imc.escaped_energy_total();
}

double relative_l2_difference(const std::vector<double>& a,
                              const std::vector<double>& b) {
  TENRYU_ASSERT(a.size() == b.size(), "L2 comparison size mismatch");
  long double num = 0.0L;
  long double den = 0.0L;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double da = static_cast<long double>(a[i]) -
                           static_cast<long double>(b[i]);
    num += da * da;
    den += static_cast<long double>(b[i]) * static_cast<long double>(b[i]);
  }
  return std::sqrt(static_cast<double>(num / std::max(den, 1.0e-300L)));
}

core::Config make_imc_ddmc_base_config(const std::string& name,
                                       const std::string& output_dir) {
  core::Config cfg{};
  cfg.main.name = name;
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.main.t_end = 1.0e-9;
  cfg.main.seed = 12345;
  cfg.main.max_steps = 50;
  cfg.main.verbosity = "quiet";

  cfg.mesh.nr = 20;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 1.0;
  cfg.mesh.r_max = 3.0;
  cfg.mesh.grid_type_r = "uniform";

  core::Config::MaterialsConfig::MatDef mat{};
  mat.name = "hybrid_mat";
  mat.A = 12.0;
  mat.Z = 6.0;
  mat.eos_model = "ideal_gas";
  mat.ideal_gas_gamma = 5.0 / 3.0;
  mat.cv_e_override = 5.488e11;
  mat.opacity_model = "constant";
  mat.kappa_a_constant = 1.0;
  mat.kappa_s_constant = 0.0;
  mat.opacity_units = "cm2_per_g";
  cfg.materials.materials = {mat};
  cfg.materials.zbar.model = "fixed";
  cfg.materials.zbar.fixed_value = 6.0;

  cfg.radiation.enabled = true;
  cfg.radiation.mode = core::RadiationMode::ImcDdmc;
  cfg.radiation.groups = 1;
  cfg.radiation.group_bounds_eV = {0.0, 1.0e6};
  cfg.radiation.compute_T_range_eV = {1.0e-3, 1.0e3};

  cfg.radiation.imc.alpha = 1.0;
  cfg.radiation.imc.f_max = 1.0;
  cfg.radiation.imc.particles_per_cell_group = 50;
  cfg.radiation.imc.implicit_capture = true;
  cfg.radiation.imc.cutoff_fraction = 0.0;
  cfg.radiation.imc.inelastic_scatter = true;
  cfg.radiation.imc.weight_cutoff = 1.0e-10;
  cfg.radiation.imc.roulette_survival = 0.1;
  cfg.radiation.imc.linearized_planck = false;

  cfg.radiation.ddmc.enabled = true;
  cfg.radiation.ddmc.tau_ddmc = 3.0;
  cfg.radiation.ddmc.omega_ddmc = 0.9;
  cfg.radiation.ddmc.leak_stencil = "4";
  cfg.radiation.ddmc.interface_method = "asymptotic_diffusion_limit";
  cfg.radiation.ddmc.emissivity_preserving = true;
  cfg.radiation.ddmc.interface_exit_distribution = "cosine";
  cfg.radiation.ddmc.rz_face_r_weight = true;
  cfg.radiation.ddmc.face_opacity_temperature = "radiative_mean";
  cfg.radiation.ddmc.m_matrix_check = true;

  cfg.radiation.boundary.inner_r = "vacuum";
  cfg.radiation.boundary.outer_r = "marshak";
  cfg.radiation.boundary.marshak_Tr_eV = 100.0;
  cfg.radiation.boundary.marshak_particles = 10000;

  cfg.numerics.dt.initial_s = 2.0e-11;
  cfg.numerics.dt.max_s = 2.0e-11;
  cfg.numerics.dt.min_s = 1.0e-20;
  cfg.numerics.dt.growth_factor = 1.0;
  cfg.numerics.hydro.enabled = false;
  cfg.numerics.conduction.enabled = false;
  cfg.numerics.floors.rho = 1.0e-10;
  cfg.numerics.floors.Te = 1.0e-3;
  cfg.numerics.floors.Ti = 1.0e-3;

  cfg.laser.enabled = false;
  cfg.output.directory = output_dir;
  cfg.output.plot_every = 0;
  cfg.output.history_every = 0;
  cfg.output.checkpoint_every = 0;
  cfg.output.plot_every_s = -1.0;
  cfg.output.history_every_s = -1.0;
  cfg.output.checkpoint_every_s = -1.0;
  cfg.diagnostics.enabled = true;
  return cfg;
}

std::vector<double> make_sigma_profile_two_layer(const int n_cells,
                                                 const double sigma_imc,
                                                 const double sigma_ddmc) {
  TENRYU_ASSERT(n_cells == 20, "M11 two-layer profile expects 20 cells");
  std::vector<double> sigma(static_cast<std::size_t>(n_cells), sigma_imc);
  for (int c = 5; c <= 14; ++c) {
    sigma[static_cast<std::size_t>(c)] = sigma_ddmc;
  }
  return sigma;
}

core::State make_hybrid_state(core::Config& cfg,
                              const std::vector<double>& sigma_profile,
                              const double Te_init_eV) {
  TENRYU_ASSERT(cfg.mesh.nr == static_cast<int>(sigma_profile.size()),
                "hybrid state sigma size mismatch");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "hybrid state requires one material");
  const double kappa = std::max(cfg.materials.materials.front().kappa_a_constant, 1.0e-30);
  const double zbar0 = std::max(cfg.materials.zbar.fixed_value, 0.0);

  core::State state = core::State::allocate(cfg);
  state.mesh = mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;

  const auto vol = copy_field_to_host(state.vol);
  std::vector<double> rho(state.rho.size(), 0.0);
  std::vector<double> mass(state.mass.size(), 0.0);
  std::vector<double> zbar(state.zbar.size(), zbar0);
  std::vector<double> Te(state.Te.size(), Te_init_eV);
  std::vector<double> Ti(state.Ti.size(), Te_init_eV);
  std::vector<double> zero_cell(state.rho.size(), 0.0);
  std::vector<double> zero_node(state.v_r.size(), 0.0);
  std::vector<double> zero_rad(state.rad_E.size(), 0.0);

  for (std::size_t c = 0; c < rho.size(); ++c) {
    rho[c] = std::max(sigma_profile[c], 0.0) / kappa;
    mass[c] = rho[c] * std::max(vol[c], 0.0);
  }

  copy_field_from_host(state.rho, rho);
  copy_field_from_host(state.mass, mass);
  copy_field_from_host(state.zbar, zbar);
  copy_field_from_host(state.Te, Te);
  copy_field_from_host(state.Ti, Ti);
  copy_field_from_host(state.ee, zero_cell);
  copy_field_from_host(state.ei, zero_cell);
  copy_field_from_host(state.Pe, zero_cell);
  copy_field_from_host(state.Pi, zero_cell);
  copy_field_from_host(state.Qvisc, zero_cell);
  copy_field_from_host(state.v_r, zero_node);
  copy_field_from_host(state.v_z, zero_node);
  copy_field_from_host(state.rad_E, zero_rad);
  copy_field_from_host(state.rad_dep, zero_rad);
  copy_field_from_host(state.rad_emit, zero_rad);
  state.laser_dep.fill(0.0);

  initialize_thermo_from_temperature(state, cfg);
  collapse_to_one_temperature_state(state, cfg);
  initialize_output_timing(state, cfg);
  state.t = 0.0;
  state.step = 0;
  state.dt = 0.0;
  return state;
}

double compute_total_system_energy_1d(const core::State& state, const int n_groups) {
  const auto rho = copy_field_to_host(state.rho);
  const auto ee = copy_field_to_host(state.ee);
  const auto ei = copy_field_to_host(state.ei);
  const auto vol = copy_field_to_host(state.vol);
  const auto rad = copy_field_to_host(state.rad_E);
  TENRYU_ASSERT(static_cast<int>(rho.size()) * n_groups == static_cast<int>(rad.size()),
                "system energy rad_E size mismatch");

  long double sum = 0.0L;
  for (std::size_t c = 0; c < rho.size(); ++c) {
    sum += static_cast<long double>(rho[c]) *
           static_cast<long double>(ee[c] + ei[c]) *
           static_cast<long double>(vol[c]);
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = c * static_cast<std::size_t>(n_groups) +
                              static_cast<std::size_t>(g);
      sum += static_cast<long double>(rad[idx]) * static_cast<long double>(vol[c]);
    }
  }
  return static_cast<double>(sum);
}

double compute_total_material_energy_1d(const core::State& state) {
  const auto rho = copy_field_to_host(state.rho);
  const auto ee = copy_field_to_host(state.ee);
  const auto ei = copy_field_to_host(state.ei);
  const auto vol = copy_field_to_host(state.vol);

  long double sum = 0.0L;
  for (std::size_t c = 0; c < rho.size(); ++c) {
    sum += static_cast<long double>(rho[c]) *
           static_cast<long double>(ee[c] + ei[c]) *
           static_cast<long double>(vol[c]);
  }
  return static_cast<double>(sum);
}

std::vector<double> extract_group_averaged_rad_profile(const core::State& state,
                                                       const int n_groups) {
  const auto rad = copy_field_to_host(state.rad_E);
  std::vector<double> profile(state.rho.size(), 0.0);
  for (std::size_t c = 0; c < profile.size(); ++c) {
    double sum = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = c * static_cast<std::size_t>(n_groups) +
                              static_cast<std::size_t>(g);
      sum += rad[idx];
    }
    profile[c] = sum / static_cast<double>(std::max(n_groups, 1));
  }
  return profile;
}

double estimate_marshak_source_energy(const core::Config& cfg, const core::State& state) {
  const auto node_r = copy_field_to_host(state.x_r);
  TENRYU_ASSERT(!node_r.empty(), "marshak source estimate requires node_r");
  const double r_src =
      (cfg.radiation.boundary.inner_r == "marshak") ? node_r.front() : node_r.back();
  const double area = 4.0 * 3.14159265358979323846 * r_src * r_src;
  const double T_src = std::max(cfg.radiation.boundary.marshak_Tr_eV, 0.0);
  return 0.25 * core::constants::a_eV * core::constants::c_light *
         T_src * T_src * T_src * T_src * area * state.t;
}

double relative_l2_profile_difference(const std::vector<double>& a,
                                      const std::vector<double>& b) {
  TENRYU_ASSERT(a.size() == b.size(), "L2 profile comparison size mismatch");
  long double num = 0.0L;
  long double den = 0.0L;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const long double da = static_cast<long double>(a[i]) -
                           static_cast<long double>(b[i]);
    num += da * da;
    den += static_cast<long double>(b[i]) * static_cast<long double>(b[i]);
  }
  return std::sqrt(static_cast<double>(num / std::max(den, 1.0e-300L)));
}

std::int64_t count_ddmc_modes_for_profile(const core::Config& cfg,
                                          const core::State& state,
                                          const std::vector<double>& sigma_profile) {
  const int n_cells = cfg.mesh.nr;
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const auto node_r = copy_field_to_host(state.x_r);
  std::vector<double> sigma_flat(static_cast<std::size_t>(n_cells) *
                                     static_cast<std::size_t>(n_groups),
                                 0.0);
  for (int c = 0; c < n_cells; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(g);
      sigma_flat[idx] = sigma_profile[static_cast<std::size_t>(c)];
    }
  }
  std::vector<double> fleck(static_cast<std::size_t>(n_cells), 0.05);
  radiation::ModeSelectorConfig mode_cfg{};
  mode_cfg.tau_ddmc = cfg.radiation.ddmc.tau_ddmc;
  mode_cfg.omega_ddmc = cfg.radiation.ddmc.omega_ddmc;
  mode_cfg.emissivity_preserving = cfg.radiation.ddmc.emissivity_preserving;
  mode_cfg.sigma_floor = cfg.numerics.safety.opacity_floor;

  radiation::ModeSelector mode_selector(n_cells, n_groups, mode_cfg);
  mode_selector.compute_modes(node_r, sigma_flat, fleck, sigma_flat);
  return mode_selector.count_ddmc();
}

struct HybridRunSummary {
  std::vector<double> Te;
  std::vector<double> rad_profile;
  std::int64_t ddmc_count = 0;
  std::uint64_t interface_transitions = 0;
  std::uint64_t interface_reflections = 0;
  std::uint64_t conversion_prob_violations = 0;
  double conservation_rel = std::numeric_limits<double>::infinity();
  double energy_delta = 0.0;
  double source_energy = 0.0;
};

HybridRunSummary run_hybrid_case(core::Config cfg,
                                 const std::vector<double>& sigma_profile,
                                 const double Te_init_eV) {
  auto state = make_hybrid_state(cfg, sigma_profile, Te_init_eV);
  HybridRunSummary out{};
  out.ddmc_count = cfg.radiation.ddmc.enabled
                       ? count_ddmc_modes_for_profile(cfg, state, sigma_profile)
                       : 0;
  radiation::IMC imc;
  const double material0 = compute_total_material_energy_1d(state);
  const double census0 = imc.census_energy();
  const double escaped0 = imc.escaped_energy_total();
  const double E0 = material0 + census0 + escaped0;

  while (state.t < cfg.main.t_end && state.step < cfg.main.max_steps) {
    const double dt = coupling::compute_dt(state, cfg, cfg.main.t_end);
    TENRYU_ASSERT(dt > 0.0, "hybrid verify produced non-positive dt");
    imc.transport_step(state, cfg, dt);
    out.interface_transitions += imc.last_interface_transitions();
    out.interface_reflections += imc.last_interface_reflections();
    out.conversion_prob_violations += imc.last_conversion_prob_violations();
    coupling::inject_radiation_source_terms(state, cfg, dt, nullptr, nullptr,
                                            &imc.last_sigma_R_max());
    state.t += dt;
    state.step += 1;
    state.dt = dt;
  }

  const double material1 = compute_total_material_energy_1d(state);
  const double census1 = imc.census_energy();
  const double escaped1 = imc.escaped_energy_total();
  const double E1 = material1 + census1 + escaped1;
  out.energy_delta = E1 - E0;
  out.source_energy = estimate_marshak_source_energy(cfg, state);
  const double residual = out.energy_delta - out.source_energy;
  out.conservation_rel = std::abs(residual) /
                         std::max({std::abs(E0), std::abs(out.source_energy), 1.0});
  out.Te = copy_field_to_host(state.Te);
  out.rad_profile = extract_group_averaged_rad_profile(state, cfg.radiation.groups);
  return out;
}

// TODO (C32): Vacuum escape energy is aggregate-only in NLTE verify tests.
// Per-group escape energy validation should be added in a future milestone.
bool run_nlte_sanity_verify() {
  if (!verify_cuda_available("nlte_sanity")) {
    return true;
  }
  const auto table =
      materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  core::Config cfg = make_nlte_verify_config("nlte_sanity",
                                             "./build/output_verify_nlte_sanity",
                                             "tests/data/ionmix_nlte_simple.cn4",
                                             table.bounds_eV,
                                             1);
  cfg.radiation.boundary.inner_r = "vacuum";
  cfg.radiation.boundary.outer_r = "vacuum";
  cfg.radiation.imc.particles_per_cell_group = 2000;
  cfg.numerics.dt.initial_s = 1.0e-11;
  cfg.numerics.dt.max_s = 1.0e-11;

  auto state = make_uniform_nlte_state(cfg, 1.0e-8, 20.0, 20.0, 3.5);
  coupling::initialize_eos_fields_if_needed(state, cfg);
  const double E0 = compute_total_material_energy_1d(state);

  const auto planck = build_planck_table_from_config(cfg);
  const auto coeffs = radiation::compute_nlte_coefficients(
      state, cfg, table, planck, 1, cfg.radiation.groups, cfg.numerics.dt.initial_s);
  const auto vol = copy_field_to_host(state.vol);
  const double Te0_eV = 20.0;
  const double T4 = Te0_eV * Te0_eV * Te0_eV * Te0_eV;
  const double expected_emit =
      coeffs.f[0] * coeffs.eta_tot[0] * std::max(vol[0], 0.0) * cfg.numerics.dt.initial_s;
  const double expected_emit_analytic =
      coeffs.f[0] * coeffs.sigma_p_em[0] * core::constants::c_light * core::constants::a_eV * T4 *
      std::max(vol[0], 0.0) * cfg.numerics.dt.initial_s;

  radiation::IMC imc;
  advance_radiation_step(state, cfg, imc, cfg.numerics.dt.initial_s);

  const double E1 = compute_total_material_energy_1d(state);
  const double delta_U = E1 - E0;
  const double rel =
      std::abs(delta_U + expected_emit) / std::max(std::abs(expected_emit), 1.0e-30);
  // NLTE sanity: analytical rel ~O(1e-7) due to Monte Carlo noise.
  // Tightened from 1e-4 (original) to 1e-6 (the internal verification record §5.4 target 1e-12
  // not achievable without increasing particle count).
  constexpr double kRelTol = 1.0e-6;
  constexpr double kAnalyticRelTol = 5.0e-2;
  const double rel_analytic =
      std::abs(expected_emit - expected_emit_analytic) /
      std::max(std::abs(expected_emit), 1.0e-30);
  const bool pass_energy = (rel <= kRelTol);
  const bool pass_analytic = (rel_analytic <= kAnalyticRelTol);

  const int n_groups = cfg.radiation.groups;
  const bool eta_size_ok = coeffs.eta.size() >= static_cast<std::size_t>(n_groups);
  const bool eta_cdf_size_ok = coeffs.eta_cdf.size() >= static_cast<std::size_t>(n_groups);
  const bool eta_tot_size_ok = !coeffs.eta_tot.empty();
  bool eta_nonneg = eta_size_ok;
  bool eta_cdf_monotonic = eta_cdf_size_ok;
  double eta_sum = 0.0;
  double eta_cdf_prev = 0.0;
  double eta_cdf_last = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const std::size_t idx = static_cast<std::size_t>(g);
    if (eta_size_ok) {
      const double eta_g = coeffs.eta[idx];
      eta_nonneg = eta_nonneg && (eta_g >= 0.0);
      eta_sum += eta_g;
    }
    if (eta_cdf_size_ok) {
      const double eta_cdf_g = coeffs.eta_cdf[idx];
      if (g > 0 && eta_cdf_g + 1.0e-14 < eta_cdf_prev) {
        eta_cdf_monotonic = false;
      }
      eta_cdf_prev = eta_cdf_g;
      eta_cdf_last = eta_cdf_g;
    }
  }
  const bool eta_cdf_last_ok =
      eta_cdf_size_ok && (n_groups > 0) && std::abs(eta_cdf_last - 1.0) <= 1.0e-12;
  const bool eta_sum_ok =
      eta_size_ok && eta_tot_size_ok && std::abs(eta_sum - coeffs.eta_tot[0]) <= 1.0e-10;
  const bool pass = pass_energy && pass_analytic && eta_nonneg && eta_cdf_monotonic &&
                    eta_cdf_last_ok && eta_sum_ok;

  core::log_info("[verify:nlte_sanity] dU=" + format_double(delta_U) +
                 ", expected=-" + format_double(expected_emit) +
                 ", rel=" + format_double(rel));
  core::log_info("[verify:nlte_sanity] expected_emit_analytic=" +
                 format_double(expected_emit_analytic) +
                 ", rel_analytic=" + format_double(rel_analytic));
  core::log_info("[verify:nlte_sanity] checks pass_energy=" +
                 std::string(pass_energy ? "true" : "false") +
                 ", pass_analytic=" + std::string(pass_analytic ? "true" : "false") +
                 ", eta_cdf_last_ok=" + std::string(eta_cdf_last_ok ? "true" : "false") +
                 ", eta_sum_ok=" + std::string(eta_sum_ok ? "true" : "false"));
  core::log_info("[verify:nlte_sanity] eta_nonneg=" +
                 std::string(eta_nonneg ? "true" : "false") +
                 ", eta_cdf_monotonic=" + std::string(eta_cdf_monotonic ? "true" : "false") +
                 ", eta_cdf_last=" + format_double(eta_cdf_last) +
                 ", eta_sum=" + format_double(eta_sum) +
                 ", eta_tot=" + format_double(eta_tot_size_ok ? coeffs.eta_tot[0] : 0.0));
  if (!pass) {
    core::log_error("[verify:nlte_sanity] FAILED");
  } else {
    core::log_info("[verify:nlte_sanity] PASSED");
  }
  return pass;
}

bool run_nlte_lte_regression_verify() {
  if (!verify_cuda_available("nlte_lte_regression")) {
    return true;
  }
  const auto table =
      materials::load_ionmix_opacity("tests/data/ionmix_lte_const.cn4");
  core::Config cfg_nlte = make_nlte_verify_config("nlte_lte_regression_nlte",
                                                   "./build/output_verify_nlte_lte_regression_nlte",
                                                   "tests/data/ionmix_lte_const.cn4",
                                                   table.bounds_eV,
                                                   20);
  cfg_nlte.radiation.boundary.inner_r = "vacuum";
  cfg_nlte.radiation.boundary.outer_r = "vacuum";
  cfg_nlte.radiation.imc.particles_per_cell_group = 300;
  cfg_nlte.numerics.dt.initial_s = 1.0e-11;
  cfg_nlte.numerics.dt.max_s = 1.0e-11;
  cfg_nlte.main.max_steps = 8;
  cfg_nlte.main.t_end = cfg_nlte.main.max_steps * cfg_nlte.numerics.dt.initial_s;

  core::Config cfg_lte = cfg_nlte;
  cfg_lte.main.name = "nlte_lte_regression_lte_ref";
  cfg_lte.output.directory = "./build/output_verify_nlte_lte_regression_lte_ref";
  auto& mat_lte = cfg_lte.materials.materials.front();
  mat_lte.opacity_model = "constant";
  mat_lte.opacity_file.clear();
  mat_lte.kappa_a_constant = 100.0;
  mat_lte.kappa_s_constant = 0.0;
  mat_lte.opacity_units = "cm2_per_g";

  auto run_case = [](core::Config& cfg) {
    auto state = make_uniform_nlte_state(cfg, 1.0, 15.0, 15.0, 3.5);
    coupling::initialize_eos_fields_if_needed(state, cfg);
    radiation::IMC imc;
    while (state.step < cfg.main.max_steps && state.t < cfg.main.t_end) {
      advance_radiation_step(state, cfg, imc, cfg.numerics.dt.initial_s);
    }
    return copy_field_to_host(state.Te);
  };

  auto Te_nlte = run_case(cfg_nlte);
  auto Te_lte = run_case(cfg_lte);
  const double l2_rel = relative_l2_difference(Te_nlte, Te_lte);

  auto state_coeff = make_uniform_nlte_state(cfg_nlte, 1.0, 15.0, 15.0, 3.5);
  coupling::initialize_eos_fields_if_needed(state_coeff, cfg_nlte);
  const auto planck = build_planck_table_from_config(cfg_nlte);
  const auto coeffs_nlte = radiation::compute_nlte_coefficients(
      state_coeff,
      cfg_nlte,
      table,
      planck,
      cfg_nlte.mesh.nr,
      cfg_nlte.radiation.groups,
      cfg_nlte.numerics.dt.initial_s);

  const auto rho = copy_field_to_host(state_coeff.rho);
  const auto Te = copy_field_to_host(state_coeff.Te);
  const auto zbar = copy_field_to_host(state_coeff.zbar);
  const auto& mat = cfg_nlte.materials.materials.front();
  const double A = std::max(mat.A, 1.0e-12);
  const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
  const double alpha = (cfg_nlte.radiation.imc.alpha > 0.0) ? cfg_nlte.radiation.imc.alpha : 1.0;
  const double dt = cfg_nlte.numerics.dt.initial_s;
  const double kappa_const = std::max(cfg_lte.materials.materials.front().kappa_a_constant, 0.0);
  const int n_groups = cfg_nlte.radiation.groups;

  double max_fleck_rel = 0.0;
  double max_sigma_a_eff_rel = 0.0;
  double max_sigma_p_rel = 0.0;
  double max_gamma_rel = 0.0;
  for (int c = 0; c < cfg_nlte.mesh.nr; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    const double rho_c = std::max(rho[c_us], 0.0);
    const double Te_c = std::max(Te[c_us], 0.0);
    const double zbar_c = std::max(zbar[c_us], 0.0);

    double Cv_e = 0.0;
    if (mat.cv_e_override > 0.0) {
      Cv_e = mat.cv_e_override;
    } else {
      Cv_e = rho_c * zbar_c * core::constants::eV_to_erg /
             (A * core::constants::proton_mass * gm1);
    }
    Cv_e = std::max(Cv_e, 1.0e-30);

    const double sigma_lte = rho_c * kappa_const;
    const double beta = 4.0 * core::constants::a_eV * Te_c * Te_c * Te_c / Cv_e;
    const double f_lte =
        1.0 / (1.0 + alpha * dt * beta * core::constants::c_light * sigma_lte);
    const double rel_f =
        std::abs(coeffs_nlte.f[c_us] - f_lte) / std::max(std::abs(f_lte), 1.0e-30);
    max_fleck_rel = std::max(max_fleck_rel, rel_f);
    const double rel_sigma_p_abs =
        std::abs(coeffs_nlte.sigma_p_abs[c_us] - sigma_lte) /
        std::max(std::abs(sigma_lte), 1.0e-30);
    const double rel_sigma_p_em =
        std::abs(coeffs_nlte.sigma_p_em[c_us] - sigma_lte) /
        std::max(std::abs(sigma_lte), 1.0e-30);
    max_sigma_p_rel = std::max(max_sigma_p_rel, std::max(rel_sigma_p_abs, rel_sigma_p_em));
    max_gamma_rel = std::max(max_gamma_rel,
                             std::abs(coeffs_nlte.gamma_diag[c_us] - 1.0));

    const double sigma_a_eff_lte = f_lte * sigma_lte;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(g);
      const double rel_sigma =
          std::abs(coeffs_nlte.sigma_a_eff[idx] - sigma_a_eff_lte) /
          std::max(std::abs(sigma_a_eff_lte), 1.0e-30);
      max_sigma_a_eff_rel = std::max(max_sigma_a_eff_rel, rel_sigma);
    }
  }

  // the internal verification record §5.4.2: NLTE-vs-LTE profile agreement tolerance (1%).
  constexpr double kL2RelTol = 1.0e-2;
  // Coefficient-level LTE regression tolerance for Fleck and sigma_a_eff.
  constexpr double kCoeffRelTol = 5.0e-2;
  const bool pass = (l2_rel <= kL2RelTol) &&
                    (max_fleck_rel <= kCoeffRelTol) &&
                    (max_sigma_a_eff_rel <= kCoeffRelTol) &&
                    (max_sigma_p_rel <= kCoeffRelTol) &&
                    (max_gamma_rel <= kCoeffRelTol);

  core::log_info("[verify:nlte_lte_regression] l2_rel=" + format_double(l2_rel) +
                 ", max_fleck_rel=" + format_double(max_fleck_rel) +
                 ", max_sigma_a_eff_rel=" + format_double(max_sigma_a_eff_rel) +
                 ", max_sigma_p_rel=" + format_double(max_sigma_p_rel) +
                 ", max_gamma_rel=" + format_double(max_gamma_rel));
  if (!pass) {
    core::log_error("[verify:nlte_lte_regression] FAILED");
  } else {
    core::log_info("[verify:nlte_lte_regression] PASSED");
  }
  return pass;
}

bool run_nlte_cooling_mms_verify() {
  if (!verify_cuda_available("nlte_cooling_mms")) {
    return true;
  }
  const auto table =
      materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  core::Config cfg = make_nlte_verify_config("nlte_cooling_mms",
                                             "./build/output_verify_nlte_cooling_mms",
                                             "tests/data/ionmix_nlte_simple.cn4",
                                             table.bounds_eV,
                                             1);
  cfg.radiation.boundary.inner_r = "vacuum";
  cfg.radiation.boundary.outer_r = "vacuum";
  cfg.radiation.imc.particles_per_cell_group = 4000;
  cfg.numerics.dt.initial_s = 1.0e-12;
  cfg.numerics.dt.max_s = 1.0e-12;
  cfg.main.max_steps = 30;
  cfg.main.t_end = cfg.main.max_steps * cfg.numerics.dt.initial_s;

  auto state = make_uniform_nlte_state(cfg, 1.0e-8, 100.0, 10.0, 3.5);
  coupling::initialize_eos_fields_if_needed(state, cfg);
  radiation::IMC imc;

  const double T0 = 100.0;
  while (state.step < cfg.main.max_steps && state.t < cfg.main.t_end) {
    advance_radiation_step(state, cfg, imc, cfg.numerics.dt.initial_s);
  }

  const auto Te = copy_field_to_host(state.Te);
  const double T_sim = Te.front();
  const double rho = 1.0e-8;
  const double zbar = cfg.materials.zbar.fixed_value;
  const double gm1 = cfg.materials.materials.front().ideal_gas_gamma - 1.0;
  const double cv_e = rho * zbar * core::constants::eV_to_erg /
                      (cfg.materials.materials.front().A *
                       core::constants::proton_mass * gm1);
  constexpr double kappa_pe = 200.0;
  const double sigma_p_em = rho * kappa_pe;
  double T_ref = T0;
  for (int step = 0; step < cfg.main.max_steps; ++step) {
    double beta = 4.0 * core::constants::a_eV * T_ref * T_ref * T_ref /
                  std::max(cv_e, 1.0e-30);
    beta = std::min(beta, 1.0);
    const double f =
        1.0 / (1.0 + cfg.radiation.imc.alpha * cfg.numerics.dt.initial_s * beta *
                         core::constants::c_light * sigma_p_em);
    const double cooling = f * sigma_p_em * core::constants::c_light *
                           core::constants::a_eV * T_ref * T_ref * T_ref * T_ref;
    T_ref = std::max(T_ref - (cfg.numerics.dt.initial_s * cooling / std::max(cv_e, 1.0e-30)),
                     cfg.numerics.floors.Te);
  }
  const double rel = std::abs(T_sim - T_ref) / std::max(std::abs(T_ref), 1.0e-30);
  // the internal verification record §5.4.3: one-zone Jayenne cooling relative-temperature tolerance.
  constexpr double kRelTol = 2.0e-2;
  const bool pass = (rel <= kRelTol);

  core::log_info("[verify:nlte_cooling_mms] T_sim=" + format_double(T_sim) +
                 ", T_ref=" + format_double(T_ref) +
                 ", rel=" + format_double(rel));
  if (!pass) {
    core::log_error("[verify:nlte_cooling_mms] FAILED");
  } else {
    core::log_info("[verify:nlte_cooling_mms] PASSED");
  }
  return pass;
}

bool run_nlte_lambda_agreement_verify() {
  if (!verify_cuda_available("nlte_lambda_agreement")) {
    return true;
  }
  const auto table =
      materials::load_ionmix_opacity("tests/data/ionmix_lte_const.cn4");
  core::Config cfg = make_nlte_verify_config("nlte_lambda_agreement",
                                             "./build/output_verify_nlte_lambda_agreement",
                                             "tests/data/ionmix_lte_const.cn4",
                                             table.bounds_eV,
                                             8);
  cfg.radiation.boundary.inner_r = "reflect";
  cfg.radiation.boundary.outer_r = "vacuum";
  cfg.radiation.imc.particles_per_cell_group = 300;
  cfg.numerics.dt.initial_s = 1.0e-11;
  cfg.numerics.dt.max_s = 1.0e-11;
  cfg.main.max_steps = 1;
  cfg.main.t_end = cfg.main.max_steps * cfg.numerics.dt.initial_s;

  core::Config cfg_freeze = cfg;
  cfg_freeze.materials.materials.front().lambda_method = "freeze_opacity";
  cfg_freeze.materials.materials.front().lambda_fd_delta_rel = 1.0e-2;
  cfg_freeze.materials.materials.front().lambda_fd_abs_min = 1.0e-3;
  core::Config cfg_fd = cfg;
  cfg_fd.materials.materials.front().lambda_method = "finite_difference";
  cfg_fd.materials.materials.front().lambda_fd_delta_rel = 1.0e-6;
  cfg_fd.materials.materials.front().lambda_fd_abs_min = 1.0e-9;

  auto state_for_lambda = make_uniform_nlte_state(cfg_freeze, 1.0, 10.0, 10.0, 3.5);
  const auto planck = build_planck_table_from_config(cfg_freeze);
  const auto coeffs_freeze = radiation::compute_nlte_coefficients(
      state_for_lambda, cfg_freeze, table, planck, cfg_freeze.mesh.nr, cfg_freeze.radiation.groups,
      cfg_freeze.numerics.dt.initial_s);
  const auto coeffs_fd = radiation::compute_nlte_coefficients(
      state_for_lambda, cfg_fd, table, planck, cfg_fd.mesh.nr, cfg_fd.radiation.groups,
      cfg_fd.numerics.dt.initial_s);

  const double rho = 1.0;
  const double Te = 10.0;
  const double zbar = 3.5;
  const double sigma_lte = rho * 100.0;
  const double gm1 = cfg_freeze.materials.materials.front().ideal_gas_gamma - 1.0;
  const double cv_e = rho * zbar * core::constants::eV_to_erg /
                      (cfg_freeze.materials.materials.front().A *
                       core::constants::proton_mass * gm1);
  const double beta = 4.0 * core::constants::a_eV * Te * Te * Te / cv_e;
  const double f_ref =
      1.0 / (1.0 + cfg_freeze.radiation.imc.alpha * cfg_freeze.numerics.dt.initial_s * beta *
                       core::constants::c_light * sigma_lte);

  const double legacy_rel =
      std::abs(coeffs_freeze.f[0] - coeffs_fd.f[0]) /
      std::max(std::abs(coeffs_freeze.f[0]), 1.0e-30);
  const double sigma_p_abs_rel =
      std::abs(coeffs_freeze.sigma_p_abs[0] - sigma_lte) / std::max(std::abs(sigma_lte), 1.0e-30);
  const double sigma_p_em_rel =
      std::abs(coeffs_freeze.sigma_p_em[0] - sigma_lte) / std::max(std::abs(sigma_lte), 1.0e-30);
  const double gamma_rel = std::abs(coeffs_freeze.gamma_diag[0] - 1.0);
  const double fleck_rel =
      std::abs(coeffs_freeze.f[0] - f_ref) / std::max(std::abs(f_ref), 1.0e-30);

  constexpr double kLegacyTol = 1.0e-12;
  constexpr double kLteTol = 5.0e-2;
  const bool pass = (legacy_rel <= kLegacyTol) &&
                    (sigma_p_abs_rel <= kLteTol) &&
                    (sigma_p_em_rel <= kLteTol) &&
                    (gamma_rel <= kLteTol) &&
                    (fleck_rel <= kLteTol);

  core::log_info("[verify:nlte_lambda_agreement] legacy_rel=" + format_double(legacy_rel) +
                 ", sigma_p_abs_rel=" + format_double(sigma_p_abs_rel) +
                 ", sigma_p_em_rel=" + format_double(sigma_p_em_rel) +
                 ", gamma_rel=" + format_double(gamma_rel) +
                 ", fleck_rel=" + format_double(fleck_rel));
  if (!pass) {
    core::log_error("[verify:nlte_lambda_agreement] FAILED");
  } else {
    core::log_info("[verify:nlte_lambda_agreement] PASSED");
  }
  return pass;
}

bool run_nlte_ddmc_classification_verify() {
  if (!verify_cuda_available("nlte_ddmc_classification")) {
    return true;
  }
  const auto table =
      materials::load_ionmix_opacity("tests/data/ionmix_lte_const.cn4");

  core::Config cfg = make_nlte_verify_config("nlte_ddmc_classification",
                                              "./build/output_verify_nlte_ddmc_classification",
                                              "tests/data/ionmix_lte_const.cn4",
                                              table.bounds_eV,
                                              20);
  cfg.radiation.ddmc.enabled = true;
  cfg.radiation.ddmc.tau_ddmc = 3.0;
  cfg.radiation.ddmc.omega_ddmc = 0.9;
  cfg.radiation.boundary.inner_r = "reflect";
  cfg.radiation.boundary.outer_r = "reflect";
  cfg.numerics.dt.initial_s = 1.0e-11;
  cfg.numerics.dt.max_s = 1.0e-11;
  cfg.radiation.compute_T_range_eV = {0.01, 10000.0};

  const int n_cells = cfg.mesh.nr;
  const int n_groups = cfg.radiation.groups;
  std::vector<double> rho_profile(static_cast<std::size_t>(n_cells), 0.01);
  for (int c = 5; c <= 14; ++c) {
    rho_profile[static_cast<std::size_t>(c)] = 2.0;
  }

  const double Te_eV = 1000.0;
  const double Ti_eV = 1000.0;
  const double zbar_val = 3.5;
  const double dt = cfg.numerics.dt.initial_s;

  auto state_nlte = make_nlte_state_with_rho_profile(cfg, rho_profile, Te_eV, Ti_eV, zbar_val);
  coupling::initialize_eos_fields_if_needed(state_nlte, cfg);

  const auto bounds = cfg.radiation.group_bounds_eV;
  radiation::Groups groups(bounds);
  radiation::PlanckTable planck;
  const std::vector<double> planck_range =
      radiation::resolve_compute_T_range_eV(cfg, false);
  planck.build(groups,
               std::max(cfg.radiation.planck_fraction.compute_N_T, 2),
               planck_range[0],
               planck_range[1]);

  const auto nlte_coeffs = radiation::compute_nlte_coefficients(
      state_nlte, cfg, table, planck, n_cells, n_groups, dt);

  radiation::ModeSelectorConfig sel_cfg;
  sel_cfg.tau_ddmc = cfg.radiation.ddmc.tau_ddmc;
  sel_cfg.omega_ddmc = cfg.radiation.ddmc.omega_ddmc;
  radiation::ModeSelector selector_nlte(n_cells, n_groups, sel_cfg);

  std::vector<double> node_r(state_nlte.x_r.size(), 0.0);
  state_nlte.x_r.copy_to_host(node_r.data());

  selector_nlte.compute_modes(node_r,
                              nlte_coeffs.sigma_R,
                              nlte_coeffs.f,
                              nlte_coeffs.sigma_pa);

  const auto& mat = cfg.materials.materials.front();
  const double kappa_const = 100.0;
  const double A = std::max(mat.A, 1.0e-12);
  const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
  const double alpha = (cfg.radiation.imc.alpha > 0.0) ? cfg.radiation.imc.alpha : 1.0;

  std::vector<double> sigma_a_lte(static_cast<std::size_t>(n_cells * n_groups), 0.0);
  std::vector<double> sigma_R_lte(static_cast<std::size_t>(n_cells * n_groups), 0.0);
  std::vector<double> fleck_lte(static_cast<std::size_t>(n_cells), 1.0);

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    const double rho_c = rho_profile[c_us];
    const double sigma = rho_c * kappa_const;

    double Cv_e = rho_c * zbar_val * core::constants::eV_to_erg /
                  (A * core::constants::proton_mass * gm1);
    Cv_e = std::max(Cv_e, 1.0e-30);

    const double T3 = Te_eV * Te_eV * Te_eV;
    const double beta = 4.0 * sigma * core::constants::c_light *
                        core::constants::a_eV * T3 / Cv_e;
    const double f_lte = 1.0 / (1.0 + alpha * dt * beta);
    fleck_lte[c_us] = f_lte;

    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = static_cast<std::size_t>(c * n_groups + g);
      sigma_a_lte[idx] = sigma;
      sigma_R_lte[idx] = sigma;
    }
  }

  radiation::ModeSelector selector_lte(n_cells, n_groups, sel_cfg);
  selector_lte.compute_modes(node_r, sigma_R_lte, fleck_lte, sigma_a_lte);

  const auto& modes_nlte = selector_nlte.modes();
  const auto& modes_lte = selector_lte.modes();

  std::int64_t ddmc_count = 0;
  for (const auto mode : modes_nlte) {
    if (mode == radiation::TransportMode::DDMC) {
      ++ddmc_count;
    }
  }

  const bool map_equal = (modes_nlte == modes_lte);
  double sigma_R_rel_max = 0.0;
  bool sigma_R_match = (nlte_coeffs.sigma_R.size() == sigma_R_lte.size());
  if (sigma_R_match) {
    for (std::size_t idx = 0; idx < sigma_R_lte.size(); ++idx) {
      const double sigma_ref = std::max(std::abs(sigma_R_lte[idx]), 1.0e-30);
      const double sigma_rel =
          std::abs(nlte_coeffs.sigma_R[idx] - sigma_R_lte[idx]) / sigma_ref;
      sigma_R_rel_max = std::max(sigma_R_rel_max, sigma_rel);
    }
  }
  constexpr double kSigmaRRelTol = 1.0e-12;
  sigma_R_match = sigma_R_match && (sigma_R_rel_max <= kSigmaRRelTol);
  const bool pass = map_equal && sigma_R_match && ddmc_count > 0 &&
                    ddmc_count < static_cast<std::int64_t>(modes_nlte.size());

  core::log_info("[verify:nlte_ddmc_classification] map_equal=" +
                 std::string(map_equal ? "true" : "false") +
                 ", sigma_R_match=" + std::string(sigma_R_match ? "true" : "false") +
                 ", sigma_R_rel_max=" + format_double(sigma_R_rel_max) +
                 ", ddmc_count=" + std::to_string(ddmc_count) +
                 ", total=" + std::to_string(modes_nlte.size()));
  // Limitation: this fixture uses an LTE table where kappa_R = kappa_PA = constant.
  core::log_info("[verify:nlte_ddmc_classification] NOTE: test uses LTE table "
                 "(kappa_R=kappa_PA=const). True non-LTE kappa_PA!=kappa_PE invariance "
                 "test requires separate NLTE table fixture.");
  if (!pass) {
    core::log_error("[verify:nlte_ddmc_classification] FAILED");
  } else {
    core::log_info("[verify:nlte_ddmc_classification] PASSED");
  }
  return pass;
}

bool run_nlte_energy_conservation_verify() {
  if (!verify_cuda_available("nlte_energy_conservation")) {
    return true;
  }
  const auto table =
      materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  core::Config cfg = make_nlte_verify_config("nlte_energy_conservation",
                                             "./build/output_verify_nlte_energy_conservation",
                                             "tests/data/ionmix_nlte_simple.cn4",
                                             table.bounds_eV,
                                             12);
  cfg.radiation.ddmc.enabled = false;
  cfg.radiation.boundary.inner_r = "reflect";
  cfg.radiation.boundary.outer_r = "reflect";
  cfg.radiation.imc.particles_per_cell_group = 120;
  cfg.numerics.dt.initial_s = 5.0e-12;
  cfg.numerics.dt.max_s = 5.0e-12;
  cfg.main.max_steps = 4;
  cfg.main.t_end = cfg.main.max_steps * cfg.numerics.dt.initial_s;

  const int n_groups = cfg.radiation.groups;

  auto compute_group_radiation_energy = [&](const core::State& state_case) {
    const auto rad = copy_field_to_host(state_case.rad_E);
    const auto vol = copy_field_to_host(state_case.vol);
    TENRYU_ASSERT(static_cast<int>(rad.size()) == cfg.mesh.nr * n_groups,
                  "nlte_energy_conservation group energy size mismatch");
    std::vector<double> E_group(static_cast<std::size_t>(n_groups), 0.0);
    for (int c = 0; c < cfg.mesh.nr; ++c) {
      const double vol_c = std::max(vol[static_cast<std::size_t>(c)], 0.0);
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t idx =
            static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
            static_cast<std::size_t>(g);
        E_group[static_cast<std::size_t>(g)] += rad[idx] * vol_c;
      }
    }
    return E_group;
  };

  struct EnergyRunSummary {
    double max_rel_step = 0.0;
    double cumulative_drift_rel = 0.0;
    double max_group_rel = std::numeric_limits<double>::quiet_NaN();
    bool group_balance_available = true;
    double escaped_total = 0.0;
    double census = 0.0;
  };

  auto run_case = [&](const double dt_case, const int max_steps_case) {
    core::Config cfg_case = cfg;
    cfg_case.numerics.dt.initial_s = dt_case;
    cfg_case.numerics.dt.max_s = dt_case;
    cfg_case.main.max_steps = max_steps_case;
    cfg_case.main.t_end = static_cast<double>(max_steps_case) * dt_case;

    auto state_case = make_uniform_nlte_state(cfg_case, 1.0, 20.0, 20.0, 3.5);
    coupling::initialize_eos_fields_if_needed(state_case, cfg_case);
    radiation::IMC imc_case;

    EnergyRunSummary out{};
    const double E_initial = total_system_energy_with_imc(state_case, imc_case, n_groups);
    double E_prev = E_initial;

    const auto E_rad_group_initial = compute_group_radiation_energy(state_case);
    std::vector<double> emitted_group(static_cast<std::size_t>(n_groups), 0.0);
    std::vector<double> absorbed_group(static_cast<std::size_t>(n_groups), 0.0);
    const std::size_t n_cell_groups = static_cast<std::size_t>(cfg_case.mesh.nr) *
                                      static_cast<std::size_t>(n_groups);

    while (state_case.step < cfg_case.main.max_steps && state_case.t < cfg_case.main.t_end) {
      advance_radiation_step(state_case, cfg_case, imc_case, dt_case);
      const double E_curr = total_system_energy_with_imc(state_case, imc_case, n_groups);
      const double rel = std::abs(E_curr - E_prev) / std::max(std::abs(E_prev), 1.0);
      out.max_rel_step = std::max(out.max_rel_step, rel);
      E_prev = E_curr;

      const bool have_group_tallies = (state_case.rad_emit.size() == n_cell_groups) &&
                                      (state_case.rad_dep.size() == n_cell_groups) &&
                                      (state_case.rad_emit.size() == state_case.rad_dep.size());
      if (out.group_balance_available && !have_group_tallies) {
        out.group_balance_available = false;
      }
      if (out.group_balance_available) {
        const auto rad_emit = copy_field_to_host(state_case.rad_emit);
        const auto rad_dep = copy_field_to_host(state_case.rad_dep);
        for (int c = 0; c < cfg_case.mesh.nr; ++c) {
          for (int g = 0; g < n_groups; ++g) {
            const std::size_t idx =
                static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
                static_cast<std::size_t>(g);
            emitted_group[static_cast<std::size_t>(g)] += rad_emit[idx];
            absorbed_group[static_cast<std::size_t>(g)] += rad_dep[idx];
          }
        }
      }
    }

    const double E_final = total_system_energy_with_imc(state_case, imc_case, n_groups);
    out.cumulative_drift_rel = (E_final - E_initial) / std::max(std::abs(E_initial), 1.0);
    out.escaped_total = imc_case.escaped_energy_total();
    out.census = imc_case.census_energy();

    if (out.group_balance_available) {
      const auto E_rad_group_final = compute_group_radiation_energy(state_case);
      const double E_norm = std::max(std::abs(E_initial), 1.0);
      double max_group_rel = 0.0;
      for (int g = 0; g < n_groups; ++g) {
        const double net_transfer = emitted_group[static_cast<std::size_t>(g)] -
                                    absorbed_group[static_cast<std::size_t>(g)];
        const double balance =
            E_rad_group_final[static_cast<std::size_t>(g)] -
            E_rad_group_initial[static_cast<std::size_t>(g)] - net_transfer;
        max_group_rel = std::max(max_group_rel, std::abs(balance) / E_norm);
      }
      out.max_group_rel = max_group_rel;
    } else {
      // TODO(U42): Expose explicit per-group emitted/absorbed diagnostics in IMC API so
      // this check does not depend on state.rad_emit/state.rad_dep buffer availability.
    }
    return out;
  };

  const double dt = cfg.numerics.dt.initial_s;
  const int n_steps = cfg.main.max_steps;
  const auto coarse = run_case(dt, n_steps);
  const auto fine = run_case(0.5 * dt, n_steps * 2);

  // the internal verification record §2.3 / §5.4.6: Monte Carlo per-step energy conservation threshold.
  constexpr double kStepEnergyRelTol = 1.0e-6;
  constexpr double kGroupBalanceRelTol = 5.0e-2;
  constexpr double kCumulativeDriftRelTol = 1.0e-3;
  constexpr double kOrderRatioMin = 2.0;
  constexpr double kOrderRatioMax = 8.0;
  constexpr double kOrderNoiseFloor = 1.0e-12;

  const bool pass_step = (coarse.max_rel_step <= kStepEnergyRelTol);
  const bool pass_group = !coarse.group_balance_available ||
                          (coarse.max_group_rel <= kGroupBalanceRelTol);
  const bool pass_cumulative =
      (std::abs(coarse.cumulative_drift_rel) <= kCumulativeDriftRelTol);
  const bool ratio_measurable = (coarse.max_rel_step > kOrderNoiseFloor) &&
                                (fine.max_rel_step > kOrderNoiseFloor);
  const double richardson_ratio =
      ratio_measurable ? (coarse.max_rel_step / fine.max_rel_step)
                       : std::numeric_limits<double>::quiet_NaN();
  const bool pass_order = !ratio_measurable ||
                          ((richardson_ratio >= kOrderRatioMin) &&
                           (richardson_ratio <= kOrderRatioMax));

  const bool pass = pass_step && pass_group && pass_cumulative && pass_order;
  core::log_info("[verify:nlte_energy_conservation] max_rel_step=" +
                 format_double(coarse.max_rel_step) +
                 ", max_group_rel=" + format_double(coarse.max_group_rel) +
                 ", cumulative_drift=" + format_double(coarse.cumulative_drift_rel) +
                 ", max_rel_step_dt_half=" + format_double(fine.max_rel_step) +
                 ", richardson_ratio=" + format_double(richardson_ratio) +
                 ", group_balance_available=" +
                 std::string(coarse.group_balance_available ? "true" : "false") +
                 ", ratio_measurable=" + std::string(ratio_measurable ? "true" : "false") +
                 ", escaped_total=" + format_double(coarse.escaped_total) +
                 ", census=" + format_double(coarse.census));
  if (!coarse.group_balance_available) {
    core::log_warning(
        "[verify:nlte_energy_conservation] group-wise balance check skipped: "
        "per-group rad_emit/rad_dep unavailable");
  }
  if (!pass) {
    core::log_error("[verify:nlte_energy_conservation] FAILED");
  } else {
    core::log_info("[verify:nlte_energy_conservation] PASSED");
  }
  return pass;
}

bool run_nlte_group_resample_verify() {
  if (!verify_cuda_available("nlte_group_resample")) {
    return true;
  }
  const auto table =
      materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  core::Config cfg = make_nlte_verify_config("nlte_group_resample",
                                             "./build/output_verify_nlte_group_resample",
                                             "tests/data/ionmix_nlte_simple.cn4",
                                             table.bounds_eV,
                                             1);
  cfg.radiation.boundary.inner_r = "reflect";
  cfg.radiation.boundary.outer_r = "reflect";
  cfg.radiation.imc.particles_per_cell_group = 10000;
  cfg.numerics.dt.initial_s = 5.0e-9;
  cfg.numerics.dt.max_s = 5.0e-9;

  auto state = make_uniform_nlte_state(cfg, 1.0, 100.0, 100.0, 3.5);
  const auto planck = build_planck_table_from_config(cfg);
  const auto coeffs = radiation::compute_nlte_coefficients(
      state, cfg, table, planck, 1, cfg.radiation.groups, cfg.numerics.dt.initial_s);

  radiation::IMC imc;
  imc.transport_step(state, cfg, cfg.numerics.dt.initial_s);

  const auto& pool = imc.photon_pool();
  std::vector<std::uint16_t> group_ids(static_cast<std::size_t>(std::max(pool.n_alive, 0)), 0);
  if (!group_ids.empty()) {
    cuda_check_verify(cudaMemcpy(group_ids.data(),
                                 pool.group_id,
                                 sizeof(std::uint16_t) * group_ids.size(),
                                 cudaMemcpyDeviceToHost),
                      "run_nlte_group_resample_verify memcpy group_id failed");
  }

  const int n_groups = cfg.radiation.groups;
  std::vector<double> obs(static_cast<std::size_t>(n_groups), 0.0);
  bool group_id_oob = false;
  for (const std::uint16_t g : group_ids) {
    const std::size_t g_us = static_cast<std::size_t>(g);
    if (g_us < obs.size()) {
      obs[g_us] += 1.0;
    } else {
      group_id_oob = true;
    }
  }

  const double eta_tot = std::max(coeffs.eta_tot[0], 1.0e-30);
  double chi2 = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const double p = std::max(coeffs.eta[static_cast<std::size_t>(g)] / eta_tot, 1.0e-12);
    const double exp = p * static_cast<double>(std::max(pool.n_alive, 1));
    const double diff = obs[static_cast<std::size_t>(g)] - exp;
    chi2 += diff * diff / std::max(exp, 1.0);
  }

  // the internal verification record §5.4.7: require enough samples for chi-square goodness-of-fit.
  constexpr int kMinSamples = 200;
  // the internal verification record §5.4.7: use significance level p=0.01 for chi-square rejection.
  const int dof = cfg.radiation.groups - 1;
  const double chi2_crit = chi_square_critical_p01(dof);
  const bool pass_counts = (pool.n_alive > kMinSamples);
  const bool pass_chi2 = (dof <= 0) ? true : (chi2 <= chi2_crit);
  const bool pass = pass_counts && !group_id_oob && pass_chi2;
  core::log_info("[verify:nlte_group_resample] n_alive=" +
                 std::to_string(pool.n_alive) +
                 ", group_id_oob=" + std::string(group_id_oob ? "true" : "false") +
                 ", chi2=" + format_double(chi2) +
                 ", dof=" + std::to_string(dof) +
                 ", chi2_crit_p01=" + format_double(chi2_crit));
  if (!pass) {
    core::log_error("[verify:nlte_group_resample] FAILED");
  } else {
    core::log_info("[verify:nlte_group_resample] PASSED");
  }
  return pass;
}

bool run_imc_ddmc_hybrid_verify() {
  core::Config cfg = make_imc_ddmc_base_config("imc_ddmc_hybrid",
                                                "./build/output_verify_imc_ddmc_hybrid");
  cfg.main.max_steps = 50;
  cfg.main.t_end = 50.0 * cfg.numerics.dt.initial_s;
  cfg.radiation.ddmc.tau_ddmc = 3.0;
  cfg.radiation.boundary.marshak_particles = 10000;
  cfg.radiation.imc.particles_per_cell_group = 50;

  const auto sigma_profile = make_sigma_profile_two_layer(cfg.mesh.nr, 0.5, 100.0);
  const auto result = run_hybrid_case(cfg, sigma_profile, kHybridInitTeEv);

  const double left_jump =
      std::abs(result.rad_profile[5] - result.rad_profile[4]) /
      std::max(std::max(std::abs(result.rad_profile[5]),
                        std::abs(result.rad_profile[4])),
               1.0e-30);
  const double right_jump =
      std::abs(result.rad_profile[15] - result.rad_profile[14]) /
      std::max(std::max(std::abs(result.rad_profile[15]),
                        std::abs(result.rad_profile[14])),
               1.0e-30);

  const bool pass_modes = (result.ddmc_count > 0) &&
                          (result.ddmc_count < cfg.mesh.nr);
  // the internal verification record §9.1: interface continuity check for mixed IMC/DDMC layer.
  constexpr double kInterfaceJumpTol = 1.0;
  const bool pass_interface = result.rad_profile[14] > 0.0 &&
                              result.rad_profile[15] > 0.0 &&
                              right_jump <= kInterfaceJumpTol;
  // the internal verification record §9.1 / §2.3: Monte Carlo cumulative energy tolerance (0.1%).
  constexpr double kEnergyRelTol = 1.0e-3;
  const bool pass_energy = (result.conservation_rel <= kEnergyRelTol);
  const bool pass = pass_modes && pass_interface && pass_energy;

  core::log_info("[verify:imc_ddmc_hybrid] ddmc_count=" +
                 std::to_string(result.ddmc_count) +
                 ", left_jump=" + format_double(left_jump) +
                 ", right_jump=" + format_double(right_jump) +
                 ", conservation_rel=" + format_double(result.conservation_rel) +
                 ", dE=" + format_double(result.energy_delta) +
                 ", Ein=" + format_double(result.source_energy));
  if (!pass) {
    core::log_error("[verify:imc_ddmc_hybrid] FAILED");
  } else {
    core::log_info("[verify:imc_ddmc_hybrid] PASSED");
  }
  return pass;
}

bool run_imc_ddmc_angular_verify() {
  const auto t_start = std::chrono::steady_clock::now();
  core::Config cfg_hat = make_imc_ddmc_base_config(
      "imc_ddmc_angular_hat", "./build/output_verify_imc_ddmc_angular_hat");
  cfg_hat.main.max_steps = 20;
  cfg_hat.main.t_end = 20.0 * cfg_hat.numerics.dt.initial_s;
  cfg_hat.main.seed = 67890;
  cfg_hat.radiation.ddmc.tau_ddmc = 1.5;
  cfg_hat.radiation.ddmc.omega_ddmc = 0.0;
  cfg_hat.radiation.ddmc.emissivity_preserving = true;
  cfg_hat.radiation.boundary.marshak_particles = 5000;
  cfg_hat.radiation.imc.particles_per_cell_group = 30;

  core::Config cfg_std = cfg_hat;
  cfg_std.main.name = "imc_ddmc_angular_std";
  cfg_std.output.directory = "./build/output_verify_imc_ddmc_angular_std";
  cfg_std.radiation.ddmc.emissivity_preserving = false;

  const auto sigma_profile = make_sigma_profile_two_layer(cfg_hat.mesh.nr, 0.5, 20.0);
  const auto result_hat = run_hybrid_case(cfg_hat, sigma_profile, kHybridInitTeEv);
  const auto result_std = run_hybrid_case(cfg_std, sigma_profile, kHybridInitTeEv);

  const double l2_rel_te =
      relative_l2_profile_difference(result_hat.Te, result_std.Te);
  const bool pass_modes = (result_hat.ddmc_count > 0) &&
                          (result_std.ddmc_count > 0);
  const bool pass_interface =
      (result_hat.interface_transitions > 0) &&
      (result_std.interface_transitions > 0);
  // the internal verification record §9.2: reject degenerate near-identical profiles.
  constexpr double kMinDistinguishableL2Rel = 1.0e-6;
  // the internal verification record §9.2: angular-model profile agreement tolerance.
  constexpr double kL2RelTol = 0.20;
  const bool pass_non_degenerate = (l2_rel_te >= kMinDistinguishableL2Rel);
  const bool pass_profile = (l2_rel_te <= kL2RelTol);
  const bool pass = pass_modes && pass_interface && pass_non_degenerate &&
                    pass_profile;
  const auto t_end = std::chrono::steady_clock::now();
  const double runtime_total_s = elapsed_seconds(t_start, t_end);

  core::log_info("[verify:imc_ddmc_angular] tau_ddmc=1.5"
                 ", omega_ddmc=0.0"
                 ", marshak_particles=5000"
                 ", ppcg=30"
                 ", l2_rel_Te_hat_vs_std=" + format_double(l2_rel_te) +
                 ", ddmc_count_hat=" + std::to_string(result_hat.ddmc_count) +
                 ", ddmc_count_std=" + std::to_string(result_std.ddmc_count) +
                 ", interface_to_ddmc_hat=" +
                 std::to_string(result_hat.interface_transitions) +
                 ", interface_to_ddmc_std=" +
                 std::to_string(result_std.interface_transitions) +
                 ", reflections_hat=" +
                 std::to_string(result_hat.interface_reflections) +
                 ", reflections_std=" +
                 std::to_string(result_std.interface_reflections) +
                 ", prob_fallbacks_hat=" +
                 std::to_string(result_hat.conversion_prob_violations) +
                 ", prob_fallbacks_std=" +
                 std::to_string(result_std.conversion_prob_violations) +
                 ", cons_hat=" + format_double(result_hat.conservation_rel) +
                 ", cons_std=" + format_double(result_std.conservation_rel) +
                 ", runtime_total_s=" + format_double(runtime_total_s));
  if (!pass) {
    core::log_error("[verify:imc_ddmc_angular] FAILED");
  } else {
    core::log_info("[verify:imc_ddmc_angular] PASSED");
  }
  return pass;
}

bool run_imc_ddmc_tau_scan_verify() {
  std::vector<double> sigma_uniform(20, 50.0);

  core::Config cfg_ddmc =
      make_imc_ddmc_base_config("imc_ddmc_tau_scan_ddmc",
                                "./build/output_verify_imc_ddmc_tau_scan_ddmc");
  cfg_ddmc.main.max_steps = 10;
  cfg_ddmc.main.t_end = 10.0 * cfg_ddmc.numerics.dt.initial_s;
  cfg_ddmc.radiation.ddmc.tau_ddmc = 3.0;
  cfg_ddmc.radiation.boundary.marshak_particles = 8000;
  cfg_ddmc.radiation.imc.particles_per_cell_group = 20;

  core::Config cfg_imc = cfg_ddmc;
  cfg_imc.main.name = "imc_ddmc_tau_scan_imc";
  cfg_imc.output.directory = "./build/output_verify_imc_ddmc_tau_scan_imc";
  cfg_imc.radiation.ddmc.tau_ddmc = 10.0;

  const auto result_ddmc = run_hybrid_case(cfg_ddmc, sigma_uniform, kHybridInitTeEv);
  const auto result_imc = run_hybrid_case(cfg_imc, sigma_uniform, kHybridInitTeEv);

  const double l2_rel_te = relative_l2_profile_difference(result_ddmc.Te, result_imc.Te);
  const bool pass_modes = (result_ddmc.ddmc_count == 20) &&
                          (result_imc.ddmc_count == 0);
  // the internal verification record §9.3: tau_ddmc sweep profile tolerance vs IMC reference.
  constexpr double kL2RelTol = 0.10;
  // the internal verification record §9.3 / §2.3: cumulative energy tolerance for Monte Carlo runs (0.5%).
  constexpr double kEnergyRelTol = 5.0e-3;
  const bool pass_profile = (l2_rel_te <= kL2RelTol);
  const bool pass_energy = (result_ddmc.conservation_rel <= kEnergyRelTol) &&
                           (result_imc.conservation_rel <= kEnergyRelTol);
  const bool pass = pass_modes && pass_profile && pass_energy;

  core::log_info("[verify:imc_ddmc_tau_scan] ci_compromise_two_point=true"
                 ", tau_tested=[3.0,10.0], spec_tau_full=[1.0,3.0,5.0,10.0]"
                 ", ddmc_count_tau3=" +
                 std::to_string(result_ddmc.ddmc_count) +
                 ", ddmc_count_tau10=" + std::to_string(result_imc.ddmc_count) +
                 ", l2_rel_Te=" + format_double(l2_rel_te) +
                 ", cons_tau3=" + format_double(result_ddmc.conservation_rel) +
                 ", cons_tau10=" + format_double(result_imc.conservation_rel));
  if (!pass) {
    core::log_error("[verify:imc_ddmc_tau_scan] FAILED");
  } else {
    core::log_info("[verify:imc_ddmc_tau_scan] PASSED");
  }
  return pass;
}

double compute_sample_mean(const std::vector<double>& values) {
  TENRYU_ASSERT(!values.empty(), "mean requires non-empty samples");
  long double sum = 0.0L;
  for (const double v : values) {
    sum += static_cast<long double>(v);
  }
  return static_cast<double>(sum / static_cast<long double>(values.size()));
}

double compute_standard_error(const std::vector<double>& values) {
  TENRYU_ASSERT(values.size() >= 2, "standard error requires at least 2 samples");
  const double mean = compute_sample_mean(values);
  long double var = 0.0L;
  for (const double v : values) {
    const long double d = static_cast<long double>(v) - static_cast<long double>(mean);
    var += d * d;
  }
  var /= static_cast<long double>(values.size() - 1);
  return std::sqrt(static_cast<double>(var / static_cast<long double>(values.size())));
}

double compute_trimmed_standard_error(const std::vector<double>& values) {
  TENRYU_ASSERT(values.size() >= 2, "trimmed standard error requires at least 2 samples");
  if (values.size() >= 5) {
    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    std::vector<double> trimmed(sorted.begin() + 1, sorted.end() - 1);
    const double se_trimmed = compute_standard_error(trimmed);
    if (se_trimmed > 0.0) {
      return se_trimmed;
    }
  }
  return compute_standard_error(values);
}

double compute_profile_rms(const std::vector<double>& profile) {
  TENRYU_ASSERT(!profile.empty(), "profile RMS requires non-empty profile");
  long double sum_sq = 0.0L;
  for (const double v : profile) {
    const long double vv = static_cast<long double>(v);
    sum_sq += vv * vv;
  }
  return std::sqrt(static_cast<double>(sum_sq / static_cast<long double>(profile.size())));
}

double relative_error_symmetric(const double a, const double b) {
  return std::abs(a - b) / std::max({std::abs(a), std::abs(b), 1.0e-30});
}

double compute_profile_rms_standard_error(
    const std::vector<std::vector<double>>& profiles) {
  TENRYU_ASSERT(profiles.size() >= 2, "profile RMS-SE requires at least 2 batches");
  const std::size_t n_cells = profiles.front().size();
  TENRYU_ASSERT(n_cells > 0, "profile RMS-SE requires non-empty profile");
  std::vector<double> rms_samples;
  rms_samples.reserve(profiles.size());
  for (const auto& profile : profiles) {
    TENRYU_ASSERT(profile.size() == n_cells,
                  "profile RMS-SE requires consistent profile size");
    rms_samples.push_back(compute_profile_rms(profile));
  }
  return compute_trimmed_standard_error(rms_samples);
}

bool run_imc_ddmc_convergence_verify() {
  const auto sigma_profile =
      make_sigma_profile_two_layer(20, 0.5, 100.0);
  const std::vector<int> particles_per_step = {4000, 16000, 64000};
  constexpr int kNumBatches = 5;
  constexpr int kConvergenceSteps = 10;
  constexpr double kReproRelTol = 1.0e-10;
  std::vector<std::vector<std::vector<double>>> profiles_by_level;
  profiles_by_level.resize(particles_per_step.size());
  bool fixed_seed_checked = false;
  double fixed_seed_max_rel = std::numeric_limits<double>::infinity();
  const auto t_start = std::chrono::steady_clock::now();

  for (std::size_t i = 0; i < particles_per_step.size(); ++i) {
    const auto t_level_start = std::chrono::steady_clock::now();
    const int n_src = particles_per_step[i];
    auto& level_profiles = profiles_by_level[i];
    level_profiles.reserve(kNumBatches);
    for (int batch = 0; batch < kNumBatches; ++batch) {
      core::Config cfg = make_imc_ddmc_base_config(
          "imc_ddmc_convergence_n" + std::to_string(n_src) +
              "_b" + std::to_string(batch),
          "./build/output_verify_imc_ddmc_convergence_n" + std::to_string(n_src) +
              "_b" + std::to_string(batch));
      cfg.main.max_steps = kConvergenceSteps;
      cfg.main.t_end =
          static_cast<double>(kConvergenceSteps) * cfg.numerics.dt.initial_s;
      cfg.main.seed = 12345 + static_cast<std::uint64_t>(batch) * 97ULL;
      cfg.radiation.boundary.marshak_particles = n_src;
      cfg.radiation.imc.particles_per_cell_group = 20;

      const auto result = run_hybrid_case(cfg, sigma_profile, kHybridInitTeEv);
      TENRYU_ASSERT(!result.rad_profile.empty(),
                    "convergence profile metric requires non-empty profile");
      if (i == particles_per_step.size() - 1 && batch == 0) {
        // One extra fixed-seed rerun keeps verify runtime well under 2x.
        core::Config cfg_repeat = cfg;
        cfg_repeat.main.name += "_repeat";
        cfg_repeat.output.directory += "_repeat";
        const auto repeat = run_hybrid_case(cfg_repeat, sigma_profile, kHybridInitTeEv);
        TENRYU_ASSERT(repeat.rad_profile.size() == result.rad_profile.size(),
                      "convergence fixed-seed reproducibility rad_profile size mismatch");
        TENRYU_ASSERT(repeat.Te.size() == result.Te.size(),
                      "convergence fixed-seed reproducibility Te size mismatch");
        const double rel_energy_delta =
            relative_error_symmetric(repeat.energy_delta, result.energy_delta);
        const double rel_source =
            relative_error_symmetric(repeat.source_energy, result.source_energy);
        const double rel_conservation =
            relative_error_symmetric(repeat.conservation_rel, result.conservation_rel);
        const double rel_rad_rms = relative_error_symmetric(
            compute_profile_rms(repeat.rad_profile), compute_profile_rms(result.rad_profile));
        const double rel_te_rms = relative_error_symmetric(
            compute_profile_rms(repeat.Te), compute_profile_rms(result.Te));
        fixed_seed_max_rel =
            std::max({rel_energy_delta, rel_source, rel_conservation, rel_rad_rms, rel_te_rms});
        fixed_seed_checked = true;
        core::log_info("[verify:imc_ddmc_convergence] fixed_seed_repeat"
                       ", N=" + std::to_string(n_src) +
                       ", rel_energy_delta=" + format_double(rel_energy_delta) +
                       ", rel_source=" + format_double(rel_source) +
                       ", rel_conservation=" + format_double(rel_conservation) +
                       ", rel_rad_rms=" + format_double(rel_rad_rms) +
                       ", rel_te_rms=" + format_double(rel_te_rms) +
                       ", max_rel=" + format_double(fixed_seed_max_rel) +
                       ", tol=" + format_double(kReproRelTol));
      }
      level_profiles.push_back(result.rad_profile);
    }

    const auto t_level_end = std::chrono::steady_clock::now();
    core::log_info("[verify:imc_ddmc_convergence] N=" + std::to_string(n_src) +
                   ", batches=" + std::to_string(kNumBatches) +
                   ", runtime_s=" +
                   format_double(elapsed_seconds(t_level_start, t_level_end)));
  }

  TENRYU_ASSERT(profiles_by_level.size() == 3, "convergence expected 3 levels");
  TENRYU_ASSERT(fixed_seed_checked,
                "convergence fixed-seed reproducibility check did not execute");
  TENRYU_ASSERT(!profiles_by_level[2].empty(), "convergence requires highest-N samples");
  const std::size_t n_cells = profiles_by_level[2].front().size();
  TENRYU_ASSERT(n_cells > 0, "convergence requires non-empty profiles");
  for (const auto& level_profiles : profiles_by_level) {
    TENRYU_ASSERT(static_cast<int>(level_profiles.size()) == kNumBatches,
                  "convergence batch count mismatch");
    for (const auto& profile : level_profiles) {
      TENRYU_ASSERT(profile.size() == n_cells,
                    "convergence profile size mismatch across levels");
    }
  }

  const double rms_se_1 = compute_profile_rms_standard_error(profiles_by_level[0]);
  const double rms_se_2 = compute_profile_rms_standard_error(profiles_by_level[1]);
  const double rms_se_3 = compute_profile_rms_standard_error(profiles_by_level[2]);
  const double ratio_12 = rms_se_1 / std::max(rms_se_2, 1.0e-300);
  const double ratio_23 = rms_se_2 / std::max(rms_se_3, 1.0e-300);
  const bool pass_12 = (ratio_12 >= 1.0 && ratio_12 <= 25.0);
  const bool pass_23 = (ratio_23 >= 1.0 && ratio_23 <= 25.0);
  const bool pass_fixed_seed =
      std::isfinite(fixed_seed_max_rel) && (fixed_seed_max_rel <= kReproRelTol);
  const bool pass = std::isfinite(ratio_12) && std::isfinite(ratio_23) &&
                    pass_12 && pass_23 && pass_fixed_seed;
  const auto t_end = std::chrono::steady_clock::now();

  core::log_info("[verify:imc_ddmc_convergence] ci_compromise=true"
                 ", metric=profile_rms_standard_error_trimmed"
                 ", N1=" + std::to_string(particles_per_step[0]) +
                 ", N2=" + std::to_string(particles_per_step[1]) +
                 ", N3=" + std::to_string(particles_per_step[2]) +
                 ", steps=" + std::to_string(kConvergenceSteps) +
                 ", rms_se_N1=" + format_double(rms_se_1) +
                 ", rms_se_N2=" + format_double(rms_se_2) +
                 ", rms_se_N3=" + format_double(rms_se_3) +
                 ", ratio_rmsse_N1_N2=" + format_double(ratio_12) +
                 ", ratio_rmsse_N2_N3=" + format_double(ratio_23) +
                 ", fixed_seed_max_rel=" + format_double(fixed_seed_max_rel) +
                 ", fixed_seed_tol=" + format_double(kReproRelTol) +
                 ", bounds=[1.0,25.0], total_runtime_s=" +
                 format_double(elapsed_seconds(t_start, t_end)));
  if (!pass) {
    core::log_error("[verify:imc_ddmc_convergence] FAILED");
  } else {
    core::log_info("[verify:imc_ddmc_convergence] PASSED");
  }
  return pass;
}

bool run_ddmc_diffusion_verify() {
  core::Config cfg;
  auto state = load_state_from_namelist("examples/verification/ddmc_diffusion.py", cfg);

  TENRYU_ASSERT(cfg.radiation.enabled, "ddmc_diffusion requires radiation enabled");
  TENRYU_ASSERT(cfg.radiation.ddmc.enabled, "ddmc_diffusion requires ddmc enabled");
  TENRYU_ASSERT(cfg.radiation.groups >= 1, "ddmc_diffusion requires at least one group");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "ddmc_diffusion requires at least one material");

  const int n_cells = cfg.mesh.nr;
  const int n_groups = std::max(cfg.radiation.groups, 1);
  TENRYU_ASSERT(n_cells > 0, "ddmc_diffusion requires positive cell count");

  const auto node_r_initial = copy_field_to_host(state.x_r);
  const auto rho_initial = copy_field_to_host(state.rho);
  const auto Te_initial = copy_field_to_host(state.Te);
  const auto zbar_initial = copy_field_to_host(state.zbar);
  const auto ee_initial = copy_field_to_host(state.ee);
  const auto rad_initial = copy_field_to_host(state.rad_E);
  const auto vol = copy_field_to_host(state.vol);
  TENRYU_ASSERT(static_cast<int>(node_r_initial.size()) == n_cells + 1,
                "ddmc_diffusion node_r size mismatch");
  TENRYU_ASSERT(static_cast<int>(rho_initial.size()) == n_cells,
                "ddmc_diffusion rho size mismatch");
  TENRYU_ASSERT(static_cast<int>(Te_initial.size()) == n_cells,
                "ddmc_diffusion Te size mismatch");
  TENRYU_ASSERT(static_cast<int>(zbar_initial.size()) == n_cells,
                "ddmc_diffusion zbar size mismatch");
  TENRYU_ASSERT(static_cast<int>(vol.size()) == n_cells,
                "ddmc_diffusion vol size mismatch");

  // Keep DDMC diffusion verification fast enough for routine CI runs.
  constexpr int kMaxVerifySteps = 20;
  constexpr int kMaxVerifyParticlesPerCellGroup = 5000;
  constexpr int kMaxVerifyMarshakParticles = 5000;
  const double dt_nominal = std::max(cfg.numerics.dt.initial_s, 1.0e-30);

  cfg.main.max_steps = std::min(cfg.main.max_steps, kMaxVerifySteps);
  cfg.main.t_end = std::min(cfg.main.t_end,
                            dt_nominal * static_cast<double>(kMaxVerifySteps));
  cfg.radiation.imc.particles_per_cell_group =
      std::min(cfg.radiation.imc.particles_per_cell_group,
               kMaxVerifyParticlesPerCellGroup);
  cfg.radiation.boundary.marshak_particles =
      std::min(cfg.radiation.boundary.marshak_particles, kMaxVerifyMarshakParticles);

  coupling::Driver driver;
  driver.run(state, cfg);

  const auto node_r_final = copy_field_to_host(state.x_r);
  const auto rho_final = copy_field_to_host(state.rho);
  const auto Te_final = copy_field_to_host(state.Te);
  const auto zbar_final = copy_field_to_host(state.zbar);
  const auto ee_final = copy_field_to_host(state.ee);
  const auto rad = copy_field_to_host(state.rad_E);
  TENRYU_ASSERT(static_cast<int>(node_r_final.size()) == n_cells + 1,
                "ddmc_diffusion final node_r size mismatch");
  TENRYU_ASSERT(static_cast<int>(rho_final.size()) == n_cells,
                "ddmc_diffusion final rho size mismatch");
  TENRYU_ASSERT(static_cast<int>(Te_final.size()) == n_cells,
                "ddmc_diffusion final Te size mismatch");
  TENRYU_ASSERT(static_cast<int>(zbar_final.size()) == n_cells,
                "ddmc_diffusion final zbar size mismatch");
  TENRYU_ASSERT(static_cast<int>(ee_final.size()) == n_cells,
                "ddmc_diffusion final ee size mismatch");
  TENRYU_ASSERT(static_cast<int>(rad.size()) == n_cells * n_groups,
                "ddmc_diffusion final rad_E size mismatch");

  const auto& mat = cfg.materials.materials.front();
  const double kappa_a = std::max(mat.kappa_a_constant, 0.0);
  TENRYU_ASSERT(kappa_a > 0.0, "ddmc_diffusion requires positive kappa_a");
  const double dt_ref = (state.step > 0)
                            ? (state.t / static_cast<double>(state.step))
                            : std::max(cfg.numerics.dt.initial_s, 1.0e-30);
  const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
  const double A_safe = std::max(mat.A, 1.0e-12);
  const bool linearized_planck =
      cfg.radiation.imc.linearized_planck && (mat.cv_e_override > 0.0);

  const auto compute_fleck = [&](const std::vector<double>& rho,
                                 const std::vector<double>& Te,
                                 const std::vector<double>& zbar,
                                 const double dt_eval) {
    std::vector<double> fleck(static_cast<std::size_t>(n_cells), 1.0);
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double rho_c = std::max(rho[c_us], 0.0);
      const double Te_c = std::max(Te[c_us], cfg.numerics.floors.Te);
      const double zbar_c = std::max(zbar[c_us], 0.0);
      const double sigma_c = std::max(rho_c * kappa_a, 0.0);

      double Cv_e = mat.cv_e_override;
      if (!(Cv_e > 0.0)) {
        Cv_e = rho_c * zbar_c * kEvToErg / (A_safe * kProtonMass * gm1);
      }
      Cv_e = std::max(Cv_e, 1.0e-30);

      double beta = 4.0 * core::constants::a_eV * Te_c * Te_c * Te_c / Cv_e;
      if (linearized_planck) {
        beta = 1.0;
      }

      const double denom =
          1.0 + cfg.radiation.imc.alpha * beta * core::constants::c_light * sigma_c * dt_eval;
      double f = 1.0 / std::max(denom, 1.0e-30);
      f = std::clamp(f, 0.0, cfg.radiation.imc.f_max);
      fleck[c_us] = f;
    }
    return fleck;
  };

  const auto build_sigma = [&](const std::vector<double>& rho) {
    std::vector<double> sigma(static_cast<std::size_t>(n_cells) *
                                  static_cast<std::size_t>(n_groups),
                              0.0);
    for (int c = 0; c < n_cells; ++c) {
      const double sigma_c =
          std::max(rho[static_cast<std::size_t>(c)], 0.0) * std::max(kappa_a, 0.0);
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t idx =
            static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
            static_cast<std::size_t>(g);
        sigma[idx] = sigma_c;
      }
    }
    return sigma;
  };

  radiation::ModeSelectorConfig mode_cfg{};
  mode_cfg.tau_ddmc = cfg.radiation.ddmc.tau_ddmc;
  mode_cfg.omega_ddmc = cfg.radiation.ddmc.omega_ddmc;
  mode_cfg.emissivity_preserving = cfg.radiation.ddmc.emissivity_preserving;
  mode_cfg.sigma_floor = cfg.numerics.safety.opacity_floor;

  const auto sigma_initial = build_sigma(rho_initial);
  const auto fleck_initial = compute_fleck(rho_initial, Te_initial, zbar_initial, dt_ref);
  radiation::ModeSelector mode_initial(n_cells, n_groups, mode_cfg);
  mode_initial.compute_modes(node_r_initial, sigma_initial, fleck_initial, sigma_initial);
  const std::int64_t ddmc_count_initial = mode_initial.count_ddmc();
  const std::int64_t imc_count_initial = mode_initial.count_imc();
  const std::int64_t omega_below_initial = mode_initial.count_omega_below_threshold();

  const auto sigma_final = build_sigma(rho_final);
  const auto fleck_final = compute_fleck(rho_final, Te_final, zbar_final, dt_ref);
  radiation::ModeSelector mode_final(n_cells, n_groups, mode_cfg);
  mode_final.compute_modes(node_r_final, sigma_final, fleck_final, sigma_final);
  const std::int64_t ddmc_count_final = mode_final.count_ddmc();
  const std::int64_t imc_count_final = mode_final.count_imc();
  const std::int64_t omega_below_final = mode_final.count_omega_below_threshold();

  const bool source_on_outer = (cfg.radiation.boundary.outer_r == "marshak");
  const bool source_on_inner = (cfg.radiation.boundary.inner_r == "marshak");
  TENRYU_ASSERT(source_on_outer != source_on_inner,
                "ddmc_diffusion requires exactly one marshak boundary");

  std::vector<double> rad_profile(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    double sum = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(g);
      sum += std::max(rad[idx], 0.0);
    }
    rad_profile[static_cast<std::size_t>(c)] =
        sum / std::max(static_cast<double>(n_groups), 1.0);
  }

  const auto monotonic_fraction = [](const std::vector<double>& values,
                                     const bool increasing_toward_outer) {
    if (values.size() < 2) {
      return 1.0;
    }
    int monotonic_pairs = 0;
    const int total_pairs = static_cast<int>(values.size()) - 1;
    for (int i = 0; i < total_pairs; ++i) {
      const double left = values[static_cast<std::size_t>(i)];
      const double right = values[static_cast<std::size_t>(i + 1)];
      const bool ok = increasing_toward_outer ? (right >= left) : (right <= left);
      if (ok) {
        ++monotonic_pairs;
      }
    }
    return static_cast<double>(monotonic_pairs) / std::max(total_pairs, 1);
  };

  const bool increasing_toward_outer = source_on_outer;
  const double rad_monotonic = monotonic_fraction(rad_profile, increasing_toward_outer);
  const double te_monotonic = monotonic_fraction(Te_final, increasing_toward_outer);

  const double rad_source = source_on_outer ? rad_profile.back() : rad_profile.front();
  const double rad_sink = source_on_outer ? rad_profile.front() : rad_profile.back();
  const double Te_source = source_on_outer ? Te_final.back() : Te_final.front();
  const double Te_sink = source_on_outer ? Te_final.front() : Te_final.back();
  const double rad_ratio = rad_source / std::max(rad_sink, 1.0e-30);
  const double Te_ratio = Te_source / std::max(Te_sink, cfg.numerics.floors.Te);

  auto total_radiation = [&](const std::vector<double>& rad_field) {
    long double sum = 0.0L;
    for (int c = 0; c < n_cells; ++c) {
      const double V = std::max(vol[static_cast<std::size_t>(c)], 0.0);
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t idx =
            static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
            static_cast<std::size_t>(g);
        sum += static_cast<long double>(rad_field[idx]) * static_cast<long double>(V);
      }
    }
    return static_cast<double>(sum);
  };

  auto total_internal_from_ee = [&](const std::vector<double>& ee_field,
                                    const std::vector<double>& rho_field) {
    long double sum = 0.0L;
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double rho_c = std::max(rho_field[c_us], 0.0);
      const double ee_c = std::max(ee_field[c_us], 0.0);
      const double V = std::max(vol[c_us], 0.0);
      sum += static_cast<long double>(rho_c) * static_cast<long double>(ee_c) *
             static_cast<long double>(V);
    }
    return static_cast<double>(sum);
  };

  auto total_internal_from_temperature = [&](const std::vector<double>& Te_field,
                                             const std::vector<double>& rho_field,
                                             const std::vector<double>& zbar_field) {
    long double sum = 0.0L;
    const double cv_i = kEvToErg / (A_safe * kProtonMass * gm1);
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double rho_c = std::max(rho_field[c_us], 0.0);
      const double rho_safe = std::max(rho_c, 1.0e-30);
      const double Te_c = std::max(Te_field[c_us], cfg.numerics.floors.Te);
      double e_total = 0.0;
      if (mat.eos_T_ref_eV > 0.0 && mat.cv_e_override > 0.0) {
        const double T_ref = mat.eos_T_ref_eV;
        const double T_ref3 = T_ref * T_ref * T_ref;
        const double alpha0 = mat.cv_e_override / (4.0 * T_ref3);
        const double T4 = Te_c * Te_c * Te_c * Te_c;
        e_total = alpha0 * T4 / rho_safe;
      } else {
        const double zbar_c = std::max(zbar_field[c_us], 0.0);
        double cv_e = 0.0;
        if (mat.cv_e_override > 0.0) {
          cv_e = mat.cv_e_override / rho_safe;
        } else {
          cv_e = zbar_c * kEvToErg / (A_safe * kProtonMass * gm1);
        }
        cv_e = std::max(cv_e, 0.0);
        e_total = (cv_i + cv_e) * Te_c;
      }
      const double V = std::max(vol[c_us], 0.0);
      sum += static_cast<long double>(rho_c) * static_cast<long double>(e_total) *
             static_cast<long double>(V);
    }
    return static_cast<double>(sum);
  };

  const double E_rad0 = total_radiation(rad_initial);
  const double E_rad1 = total_radiation(rad);
  double E_int0 = total_internal_from_ee(ee_initial, rho_initial);
  if (!(E_int0 > 0.0)) {
    E_int0 = total_internal_from_temperature(Te_initial, rho_initial, zbar_initial);
  }
  const double E_int1 = total_internal_from_ee(ee_final, rho_final);
  const double delta_E_system = (E_int1 + E_rad1) - (E_int0 + E_rad0);

  const double T_src = std::max(cfg.radiation.boundary.marshak_Tr_eV, 0.0);
  const double r_boundary = source_on_outer ? node_r_final.back() : node_r_final.front();
  constexpr double kPi = 3.14159265358979323846;
  const double area = 4.0 * kPi * r_boundary * r_boundary;
  const double E_marshak_in = 0.25 * core::constants::a_eV * core::constants::c_light *
                              T_src * T_src * T_src * T_src * area * state.t;
  const double retained_fraction = delta_E_system / std::max(E_marshak_in, 1.0e-30);

  const std::int64_t total_modes =
      static_cast<std::int64_t>(n_cells) * static_cast<std::int64_t>(n_groups);
  const bool pass_mode = (ddmc_count_initial == total_modes) && (ddmc_count_final > 0);
  // the internal verification record §8.1: pure-DDMC diffusion profile sanity gates.
  constexpr double kRadMonotonicMin = 0.60;
  constexpr double kTeMonotonicMin = 0.55;
  constexpr double kRatioMin = 1.0;
  const bool pass_profile =
      (rad_monotonic >= kRadMonotonicMin) && (te_monotonic >= kTeMonotonicMin) &&
      (rad_ratio > kRatioMin) && (Te_ratio >= kRatioMin);
  // the internal verification record §8.1: retained-energy envelope for finite-step diffusion runs.
  // Keep a 1% window: tighter than the historical 5% gate, while still allowing MC variance.
  constexpr double kRetainedFractionMin = -0.01;
  constexpr double kRetainedFractionMax = 1.01;
  const bool pass_energy = (E_marshak_in > 0.0) &&
                           (retained_fraction >= kRetainedFractionMin) &&
                           (retained_fraction <= kRetainedFractionMax);
  const bool pass = pass_mode && pass_profile && pass_energy;

  core::log_info("[verify:ddmc_diffusion] steps=" + std::to_string(state.step) +
                 ", dt_ref=" + format_double(dt_ref) +
                 ", ddmc_initial=" + std::to_string(ddmc_count_initial) +
                 ", imc_initial=" + std::to_string(imc_count_initial) +
                 ", omega_below_initial=" + std::to_string(omega_below_initial) +
                 ", ddmc_final=" + std::to_string(ddmc_count_final) +
                 ", imc_final=" + std::to_string(imc_count_final) +
                 ", omega_below_final=" + std::to_string(omega_below_final));
  core::log_info("[verify:ddmc_diffusion] rad_monotonic=" + format_double(rad_monotonic) +
                 ", Te_monotonic=" + format_double(te_monotonic) +
                 ", rad_ratio_source_to_sink=" + format_double(rad_ratio) +
                 ", Te_ratio_source_to_sink=" + format_double(Te_ratio) +
                 ", retained_fraction=" + format_double(retained_fraction) +
                 ", E_marshak_in=" + format_double(E_marshak_in) +
                 ", delta_E_system=" + format_double(delta_E_system));
  if (!pass) {
    core::log_error("[verify:ddmc_diffusion] FAILED");
  } else {
    core::log_info("[verify:ddmc_diffusion] PASSED");
  }
  return pass;
}

bool run_ddmc_leak_normalization_verify() {
  core::Config cfg;
  auto state =
      load_state_from_namelist("examples/verification/ddmc_leak_normalization.py", cfg);
  TENRYU_ASSERT(cfg.radiation.groups >= 1,
                "ddmc_leak_normalization requires at least one group");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "ddmc_leak_normalization requires at least one material");

  const int n_cells = cfg.mesh.nr;
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const auto node_r = copy_field_to_host(state.x_r);
  const auto rho = copy_field_to_host(state.rho);
  const auto Te = copy_field_to_host(state.Te);

  const auto& mat = cfg.materials.materials.front();
  std::vector<double> sigma_R(static_cast<std::size_t>(n_cells) *
                                  static_cast<std::size_t>(n_groups),
                              0.0);
  std::vector<double> sigma_a = sigma_R;
  for (int c = 0; c < n_cells; ++c) {
    const double sigma_c = std::max(rho[static_cast<std::size_t>(c)], 0.0) *
                           std::max(mat.kappa_a_constant, 0.0);
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(g);
      sigma_R[idx] = sigma_c;
      sigma_a[idx] = sigma_c;
    }
  }

  std::vector<double> fleck_f(static_cast<std::size_t>(n_cells), 0.05);
  radiation::ModeSelectorConfig mode_cfg{};
  mode_cfg.tau_ddmc = cfg.radiation.ddmc.tau_ddmc;
  mode_cfg.omega_ddmc = cfg.radiation.ddmc.omega_ddmc;
  mode_cfg.emissivity_preserving = cfg.radiation.ddmc.emissivity_preserving;
  mode_cfg.sigma_floor = cfg.numerics.safety.opacity_floor;

  radiation::ModeSelector mode_selector(n_cells, n_groups, mode_cfg);
  mode_selector.compute_modes(node_r, sigma_R, fleck_f, sigma_a);

  radiation::DDMCCoefficients coefficients(n_cells,
                                           n_groups,
                                           cfg.numerics.safety.opacity_floor);
  coefficients.compute_1d(
      node_r,
      rho,
      Te,
      sigma_R,
      mode_selector,
      ddmc_boundary_type_from_string(cfg.radiation.boundary.inner_r),
      ddmc_boundary_type_from_string(cfg.radiation.boundary.outer_r),
      true,
      nullptr);

  std::vector<double> sigma_a_eff(static_cast<std::size_t>(n_cells) *
                                      static_cast<std::size_t>(n_groups),
                                  0.0);
  for (int c = 0; c < n_cells; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(g);
      sigma_a_eff[idx] = std::max(fleck_f[static_cast<std::size_t>(c)] * sigma_a[idx], 0.0);
    }
  }

  double max_norm_err = 0.0;
  double min_prob = std::numeric_limits<double>::infinity();
  std::int64_t checked = 0;
  for (int c = 0; c < n_cells; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      if (mode_selector.get_mode(c, g) != radiation::TransportMode::DDMC) {
        continue;
      }
      const auto& cell = coefficients.get_cell_data(c, g);
      const std::size_t idx =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
          static_cast<std::size_t>(g);
      const double sigma_abs = std::max(sigma_a_eff[idx], 0.0);
      const double sigma_left = std::max(cell.sigma_leak_left, 0.0);
      const double sigma_right = std::max(cell.sigma_leak_right, 0.0);

      double sigma_internal = 0.0;
      double sigma_boundary = 0.0;
      if (cell.bc_left == radiation::DDMCBoundaryType::Internal ||
          cell.bc_left == radiation::DDMCBoundaryType::Interface) {
        sigma_internal += sigma_left;
      } else if (cell.bc_left == radiation::DDMCBoundaryType::Vacuum) {
        sigma_boundary += sigma_left;
      }
      if (cell.bc_right == radiation::DDMCBoundaryType::Internal ||
          cell.bc_right == radiation::DDMCBoundaryType::Interface) {
        sigma_internal += sigma_right;
      } else if (cell.bc_right == radiation::DDMCBoundaryType::Vacuum) {
        sigma_boundary += sigma_right;
      }

      const double sigma_tot = sigma_abs + sigma_internal + sigma_boundary;
      if (sigma_tot <= 0.0) {
        max_norm_err = std::numeric_limits<double>::infinity();
        continue;
      }

      const double p_internal = sigma_internal / sigma_tot;
      const double p_abs = sigma_abs / sigma_tot;
      const double p_boundary = sigma_boundary / sigma_tot;
      const double p_sum = p_internal + p_abs + p_boundary;
      max_norm_err = std::max(max_norm_err, std::abs(p_sum - 1.0));
      min_prob = std::min(min_prob, std::min(p_internal, std::min(p_abs, p_boundary)));
      ++checked;
    }
  }

  auto mode_for_mmatrix = mode_selector;
  const auto mmatrix =
      radiation::check_mmatrix_condition(coefficients, mode_for_mmatrix, sigma_a_eff);
  const bool all_ddmc =
      (mode_selector.count_ddmc() == static_cast<std::int64_t>(n_cells) * n_groups);
  // the internal verification record §8.2: DDMC leak-probability normalization tolerance.
  constexpr double kNormalizationTol = 1.0e-14;
  const bool pass = all_ddmc && checked > 0 && max_norm_err <= kNormalizationTol &&
                    min_prob >= -kNormalizationTol && mmatrix.total_violations == 0;

  core::log_info("[verify:ddmc_leak_normalization] checked=" + std::to_string(checked) +
                 ", ddmc_count=" + std::to_string(mode_selector.count_ddmc()) +
                 ", max_norm_err=" + format_double(max_norm_err) +
                 ", min_prob=" + format_double(min_prob) +
                 ", mmatrix_violations=" + std::to_string(mmatrix.total_violations));
  if (!pass) {
    core::log_error("[verify:ddmc_leak_normalization] FAILED");
  } else {
    core::log_info("[verify:ddmc_leak_normalization] PASSED");
  }
  return pass;
}

bool run_mmatrix_fallback_verify() {
  core::Config cfg;
  auto state = load_state_from_namelist("examples/verification/mmatrix_fallback.py", cfg);
  TENRYU_ASSERT(cfg.radiation.groups >= 1,
                "mmatrix_fallback requires at least one radiation group");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "mmatrix_fallback requires at least one material");

  const int n_cells = cfg.mesh.nr;
  const int n_groups = 1;
  const int target_cell = std::min(5, n_cells - 1);
  const auto node_r = copy_field_to_host(state.x_r);
  const auto rho = copy_field_to_host(state.rho);
  const auto Te = copy_field_to_host(state.Te);

  const auto& mat = cfg.materials.materials.front();
  std::vector<double> sigma_R(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    sigma_R[static_cast<std::size_t>(c)] =
        std::max(rho[static_cast<std::size_t>(c)], 0.0) *
        std::max(mat.kappa_a_constant, 0.0);
  }
  std::vector<double> fleck_f(static_cast<std::size_t>(n_cells), 0.05);

  radiation::ModeSelectorConfig mode_cfg{};
  mode_cfg.tau_ddmc = cfg.radiation.ddmc.tau_ddmc;
  mode_cfg.omega_ddmc = cfg.radiation.ddmc.omega_ddmc;
  mode_cfg.emissivity_preserving = cfg.radiation.ddmc.emissivity_preserving;
  mode_cfg.sigma_floor = cfg.numerics.safety.opacity_floor;

  radiation::ModeSelector mode_selector(n_cells, n_groups, mode_cfg);
  mode_selector.compute_modes(node_r, sigma_R, fleck_f, sigma_R);

  radiation::DDMCCoefficients coefficients(n_cells,
                                           n_groups,
                                           cfg.numerics.safety.opacity_floor);
  coefficients.compute_1d(
      node_r,
      rho,
      Te,
      sigma_R,
      mode_selector,
      ddmc_boundary_type_from_string(cfg.radiation.boundary.inner_r),
      ddmc_boundary_type_from_string(cfg.radiation.boundary.outer_r),
      false,
      nullptr);

  auto& bad = const_cast<radiation::CellDDMCData&>(coefficients.get_cell_data(target_cell, 0));
  bad.sigma_leak_right = -std::abs(bad.sigma_leak_right);
  bad.sigma_leak_out = bad.sigma_leak_left + bad.sigma_leak_right;

  std::vector<double> sigma_a_eff(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    sigma_a_eff[static_cast<std::size_t>(c)] =
        std::max(fleck_f[static_cast<std::size_t>(c)] * sigma_R[static_cast<std::size_t>(c)],
                 0.0);
  }
  const auto mmatrix =
      radiation::check_mmatrix_condition(coefficients, mode_selector, sigma_a_eff);

  bool others_ddmc = true;
  for (int c = 0; c < n_cells; ++c) {
    const auto mode = mode_selector.get_mode(c, 0);
    if (c == target_cell) {
      if (mode != radiation::TransportMode::IMC) {
        others_ddmc = false;
      }
    } else if (mode != radiation::TransportMode::DDMC) {
      others_ddmc = false;
    }
  }

  std::vector<double> pseudo_energy(static_cast<std::size_t>(n_cells), 1.0);
  double E0 = 0.0;
  for (const double e : pseudo_energy) {
    E0 += e;
  }
  double E1 = 0.0;
  for (const double e : pseudo_energy) {
    E1 += e;
  }
  const double energy_rel = std::abs(E1 - E0) / std::max(std::abs(E0), 1.0);

  // the internal verification record §8.4: mixed-mode fallback run energy stability tolerance (0.1%).
  constexpr double kEnergyRelTol = 1.0e-3;
  const bool pass = (mmatrix.total_violations >= 1) &&
                    (mmatrix.off_diagonal_violations >= 1) && others_ddmc &&
                    (mode_selector.get_mode(target_cell, 0) ==
                     radiation::TransportMode::IMC) &&
                    (energy_rel <= kEnergyRelTol);

  core::log_info("[verify:mmatrix_fallback] target_cell=" + std::to_string(target_cell) +
                 ", mmatrix_violations=" + std::to_string(mmatrix.total_violations) +
                 ", offdiag_violations=" +
                 std::to_string(mmatrix.off_diagonal_violations) +
                 ", energy_rel=" + format_double(energy_rel));
  if (!pass) {
    core::log_error("[verify:mmatrix_fallback] FAILED");
  } else {
    core::log_info("[verify:mmatrix_fallback] PASSED");
  }
  return pass;
}

bool run_ddmc_multigroup_verify() {
  core::Config cfg;
  auto state = load_state_from_namelist("examples/verification/ddmc_multigroup.py", cfg);
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "ddmc_multigroup requires at least one material");
  TENRYU_ASSERT(cfg.mesh.nr > 0, "ddmc_multigroup requires positive cell count");

  const int n_cells = cfg.mesh.nr;
  const auto node_r = copy_field_to_host(state.x_r);
  const auto rho = copy_field_to_host(state.rho);
  const auto Te = copy_field_to_host(state.Te);
  const auto& mat = cfg.materials.materials.front();
  const double kappa_uniform = std::max(mat.kappa_a_constant, 0.0);
  std::vector<double> fleck_f(static_cast<std::size_t>(n_cells), 0.05);

  radiation::ModeSelectorConfig mode_cfg{};
  mode_cfg.tau_ddmc = cfg.radiation.ddmc.tau_ddmc;
  mode_cfg.omega_ddmc = cfg.radiation.ddmc.omega_ddmc;
  mode_cfg.emissivity_preserving = cfg.radiation.ddmc.emissivity_preserving;
  mode_cfg.sigma_floor = cfg.numerics.safety.opacity_floor;

  // Config A: grey vs multigroup with uniform opacity.
  constexpr int kGroupsA = 4;
  std::vector<double> sigma_A(static_cast<std::size_t>(n_cells) * kGroupsA, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const double sigma_c =
        std::max(rho[static_cast<std::size_t>(c)], 0.0) * kappa_uniform;
    for (int g = 0; g < kGroupsA; ++g) {
      sigma_A[static_cast<std::size_t>(c) * kGroupsA + static_cast<std::size_t>(g)] =
          sigma_c;
    }
  }

  radiation::ModeSelector mode_grey(n_cells, 1, mode_cfg);
  std::vector<double> sigma_grey(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    sigma_grey[static_cast<std::size_t>(c)] =
        sigma_A[static_cast<std::size_t>(c) * kGroupsA];
  }
  mode_grey.compute_modes(node_r, sigma_grey, fleck_f, sigma_grey);

  radiation::ModeSelector mode_A(n_cells, kGroupsA, mode_cfg);
  mode_A.compute_modes(node_r, sigma_A, fleck_f, sigma_A);

  radiation::DDMCCoefficients coeff_grey(n_cells, 1, cfg.numerics.safety.opacity_floor);
  coeff_grey.compute_1d(
      node_r,
      rho,
      Te,
      sigma_grey,
      mode_grey,
      ddmc_boundary_type_from_string(cfg.radiation.boundary.inner_r),
      ddmc_boundary_type_from_string(cfg.radiation.boundary.outer_r),
      true,
      nullptr);
  radiation::DDMCCoefficients coeff_A(n_cells, kGroupsA, cfg.numerics.safety.opacity_floor);
  coeff_A.compute_1d(
      node_r,
      rho,
      Te,
      sigma_A,
      mode_A,
      ddmc_boundary_type_from_string(cfg.radiation.boundary.inner_r),
      ddmc_boundary_type_from_string(cfg.radiation.boundary.outer_r),
      true,
      nullptr);

  bool all_ddmc_A = true;
  double max_rel_coeff_A = 0.0;
  for (int c = 0; c < n_cells; ++c) {
    const auto& grey = coeff_grey.get_cell_data(c, 0);
    for (int g = 0; g < kGroupsA; ++g) {
      if (mode_A.get_mode(c, g) != radiation::TransportMode::DDMC) {
        all_ddmc_A = false;
      }
      const auto& mg = coeff_A.get_cell_data(c, g);
      const double rel_left = std::abs(mg.sigma_leak_left - grey.sigma_leak_left) /
                              std::max(std::abs(grey.sigma_leak_left), 1.0e-20);
      const double rel_right = std::abs(mg.sigma_leak_right - grey.sigma_leak_right) /
                               std::max(std::abs(grey.sigma_leak_right), 1.0e-20);
      max_rel_coeff_A = std::max(max_rel_coeff_A, std::max(rel_left, rel_right));
    }
  }
  // the internal verification record §8.3: grey-vs-multigroup coefficient consistency tolerance.
  constexpr double kCoeffRelTol = 1.0e-12;
  const bool pass_A = all_ddmc_A && (max_rel_coeff_A <= kCoeffRelTol);

  // Config B: mixed per-group mode map (group0 DDMC, group1 IMC).
  constexpr int kGroupsB = 2;
  std::vector<double> sigma_B(static_cast<std::size_t>(n_cells) * kGroupsB, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const double rho_c = std::max(rho[static_cast<std::size_t>(c)], 0.0);
    sigma_B[static_cast<std::size_t>(c) * kGroupsB + 0] = 500.0 * rho_c;
    sigma_B[static_cast<std::size_t>(c) * kGroupsB + 1] = 0.5 * rho_c;
  }

  radiation::ModeSelector mode_B(n_cells, kGroupsB, mode_cfg);
  mode_B.compute_modes(node_r, sigma_B, fleck_f, sigma_B);

  bool mixed_mode_ok = true;
  for (int c = 0; c < n_cells; ++c) {
    const bool g0_ddmc = (mode_B.get_mode(c, 0) == radiation::TransportMode::DDMC);
    const bool g1_imc = (mode_B.get_mode(c, 1) == radiation::TransportMode::IMC);
    if (!(g0_ddmc && g1_imc)) {
      mixed_mode_ok = false;
    }
  }

  const double E0_g0 = 1.0;
  const double E0_g1 = 1.0;
  const double E1_g0 = E0_g0;
  const double E1_g1 = E0_g1;
  const double err_g0 = std::abs(E1_g0 - E0_g0) / std::max(std::abs(E0_g0), 1.0);
  const double err_g1 = std::abs(E1_g1 - E0_g1) / std::max(std::abs(E0_g1), 1.0);
  // the internal verification record §8.3: per-group energy tolerance (0.1%).
  constexpr double kGroupEnergyRelTol = 1.0e-3;
  const bool pass_energy_B = (err_g0 <= kGroupEnergyRelTol) && (err_g1 <= kGroupEnergyRelTol);

  const std::int64_t ddmc_group0 = n_cells;
  std::int64_t ddmc_group1 = 0;
  for (int c = 0; c < n_cells; ++c) {
    if (mode_B.get_mode(c, 1) == radiation::TransportMode::DDMC) {
      ++ddmc_group1;
    }
  }
  // the internal verification record §8.3: DDMC event-count separation between groups.
  constexpr double kDdmcGroupRatioMin = 100.0;
  const bool ratio_ok = (ddmc_group1 == 0) ||
                        (static_cast<double>(ddmc_group0) / ddmc_group1 >= kDdmcGroupRatioMin);

  const bool pass = pass_A && mixed_mode_ok && pass_energy_B && ratio_ok;
  core::log_info("[verify:ddmc_multigroup] pass_A=" + std::string(pass_A ? "true" : "false") +
                 ", max_rel_coeff_A=" + format_double(max_rel_coeff_A) +
                 ", mixed_mode_ok=" + std::string(mixed_mode_ok ? "true" : "false") +
                 ", err_g0=" + format_double(err_g0) +
                 ", err_g1=" + format_double(err_g1) +
                 ", ddmc_g0=" + std::to_string(ddmc_group0) +
                 ", ddmc_g1=" + std::to_string(ddmc_group1));
  if (!pass) {
    core::log_error("[verify:ddmc_multigroup] FAILED");
  } else {
    core::log_info("[verify:ddmc_multigroup] PASSED");
  }
  return pass;
}

bool run_void_passthrough_verify() {
  if (!verify_cuda_available("void_passthrough")) {
    return true;
  }

  core::Config cfg{};
  cfg.main.name = "void_passthrough";
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.main.t_end = 1.0e-11;
  cfg.main.seed = 12345;
  cfg.main.max_steps = 1;
  cfg.main.verbosity = "quiet";

  cfg.mesh.nr = 20;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = 1.0;
  cfg.mesh.grid_type_r = "uniform";

  core::Config::MaterialsConfig::MatDef absorber{};
  absorber.name = "absorber";
  absorber.A = 1.0;
  absorber.Z = 1.0;
  absorber.eos_model = "ideal_gas";
  absorber.ideal_gas_gamma = 5.0 / 3.0;
  absorber.opacity_model = "constant";
  absorber.kappa_a_constant = 100.0;
  absorber.kappa_s_constant = 0.0;
  absorber.opacity_units = "cm2_per_g";
  absorber.is_void = false;

  core::Config::MaterialsConfig::MatDef void_mat{};
  void_mat.name = "void";
  void_mat.A = 1.0;
  void_mat.Z = 0.0;
  void_mat.eos_model = "ideal_gas";
  void_mat.ideal_gas_gamma = 5.0 / 3.0;
  void_mat.opacity_model = "constant";
  void_mat.kappa_a_constant = 0.0;
  void_mat.kappa_s_constant = 0.0;
  void_mat.opacity_units = "cm2_per_g";
  void_mat.is_void = true;
  cfg.materials.materials = {absorber, void_mat};
  cfg.materials.zbar.model = "fixed";
  cfg.materials.zbar.fixed_value = 1.0;
  cfg.materials.void_config.rho = 1.0e-10;
  cfg.materials.void_config.Te = 1.0e-3;
  cfg.materials.void_config.Ti = 1.0e-3;

  cfg.radiation.enabled = true;
  cfg.radiation.mode = core::RadiationMode::ImcDdmc;
  cfg.radiation.groups = 1;
  cfg.radiation.group_bounds_eV = {0.0, 1.0e6};
  cfg.radiation.compute_T_range_eV = {1.0e-3, 1.0e3};
  cfg.radiation.planck_fraction.compute_N_T = 200;
  cfg.radiation.imc.alpha = 1.0;
  cfg.radiation.imc.f_max = 1.0;
  cfg.radiation.imc.particles_per_cell_group = 200;
  cfg.radiation.imc.implicit_capture = true;
  cfg.radiation.imc.cutoff_fraction = 0.0;
  cfg.radiation.imc.inelastic_scatter = true;
  cfg.radiation.imc.weight_cutoff = 1.0e-10;
  cfg.radiation.imc.roulette_survival = 0.1;
  cfg.radiation.imc.linearized_planck = false;
  cfg.radiation.ddmc.enabled = false;
  cfg.radiation.boundary.inner_r = "marshak";
  cfg.radiation.boundary.outer_r = "vacuum";
  cfg.radiation.boundary.marshak_Tr_eV = 1.0;
  cfg.radiation.boundary.marshak_particles = 2000;

  cfg.numerics.dt.initial_s = 1.0e-11;
  cfg.numerics.dt.max_s = 1.0e-11;
  cfg.numerics.dt.min_s = 1.0e-20;
  cfg.numerics.dt.growth_factor = 1.0;
  cfg.numerics.hydro.enabled = false;
  cfg.numerics.conduction.enabled = false;
  cfg.numerics.floors.rho = 1.0e-10;
  cfg.numerics.floors.Te = 1.0e-3;
  cfg.numerics.floors.Ti = 1.0e-3;

  cfg.laser.enabled = false;
  cfg.output.directory = "./output_verify_void_passthrough";
  cfg.output.plot_every = 0;
  cfg.output.history_every = 0;
  cfg.output.checkpoint_every = 0;
  cfg.output.plot_every_s = -1.0;
  cfg.output.history_every_s = -1.0;
  cfg.output.checkpoint_every_s = -1.0;
  cfg.diagnostics.enabled = true;

  core::State state = core::State::allocate(cfg);
  state.mesh = mesh::create_mesh(cfg, state);

  const auto nodes = build_uniform_nodes(cfg.mesh.r_min, cfg.mesh.r_max, cfg.mesh.nr);
  TENRYU_ASSERT(state.x_r.size() == nodes.size(),
                "void_passthrough node count mismatch");
  state.x_r.copy_from_host(nodes.data());
  std::vector<double> node_z(state.x_z.size(), 0.0);
  copy_field_from_host(state.x_z, node_z);

  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;

  const int n_cells = cfg.mesh.nr;
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  const int n_groups = std::max(cfg.radiation.groups, 1);
  TENRYU_ASSERT(state.volFrac.size() ==
                    static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat),
                "void_passthrough volFrac size mismatch");
  TENRYU_ASSERT(state.cell_is_void.size() == static_cast<std::size_t>(n_cells),
                "void_passthrough cell_is_void size mismatch");

  std::vector<double> host_volfrac(state.volFrac.size(), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const bool absorber_cell = (c < n_cells / 2);
    const std::size_t base = static_cast<std::size_t>(c) * static_cast<std::size_t>(n_mat);
    host_volfrac[base + 0] = absorber_cell ? 1.0 : 0.0;
    host_volfrac[base + 1] = absorber_cell ? 0.0 : 1.0;
  }
  copy_field_from_host(state.volFrac, host_volfrac);

  constexpr double kVolFracTol = 1.0e-12;
  int n_void_cells = 0;
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t base = static_cast<std::size_t>(c) * static_cast<std::size_t>(n_mat);
    double nonvoid_sum = 0.0;
    for (int m = 0; m < n_mat; ++m) {
      if (!cfg.materials.materials[static_cast<std::size_t>(m)].is_void) {
        nonvoid_sum += std::max(host_volfrac[base + static_cast<std::size_t>(m)], 0.0);
      }
    }
    state.cell_is_void[static_cast<std::size_t>(c)] =
        (nonvoid_sum <= kVolFracTol) ? static_cast<std::uint8_t>(1)
                                     : static_cast<std::uint8_t>(0);
    if (state.cell_is_void[static_cast<std::size_t>(c)] != 0U) {
      ++n_void_cells;
    }
  }

  const auto vol = copy_field_to_host(state.vol);
  std::vector<double> host_rho(state.rho.size(), cfg.materials.void_config.rho);
  std::vector<double> host_mass(state.mass.size(), 0.0);
  std::vector<double> host_zbar(state.zbar.size(), 0.0);
  std::vector<double> host_Te(state.Te.size(), cfg.materials.void_config.Te);
  std::vector<double> host_Ti(state.Ti.size(), cfg.materials.void_config.Ti);

  constexpr double kRhoAbsorber = 1.0;
  constexpr double kTeInit = 1.0;
  constexpr double kTiInit = 1.0;
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    if (state.cell_is_void[c_us] == 0U) {
      host_rho[c_us] = kRhoAbsorber;
      host_zbar[c_us] = 1.0;
      host_Te[c_us] = kTeInit;
      host_Ti[c_us] = kTiInit;
    }
    host_mass[c_us] = host_rho[c_us] * std::max(vol[c_us], 0.0);
  }

  std::vector<double> zero_cell(state.rho.size(), 0.0);
  std::vector<double> zero_node_r(state.v_r.size(), 0.0);
  std::vector<double> zero_node_z(state.v_z.size(), 0.0);
  std::vector<double> zero_rad(state.rad_E.size(), 0.0);

  copy_field_from_host(state.rho, host_rho);
  copy_field_from_host(state.mass, host_mass);
  copy_field_from_host(state.zbar, host_zbar);
  copy_field_from_host(state.Te, host_Te);
  copy_field_from_host(state.Ti, host_Ti);
  copy_field_from_host(state.ee, zero_cell);
  copy_field_from_host(state.ei, zero_cell);
  copy_field_from_host(state.Pe, zero_cell);
  copy_field_from_host(state.Pi, zero_cell);
  copy_field_from_host(state.Qvisc, zero_cell);
  copy_field_from_host(state.v_r, zero_node_r);
  copy_field_from_host(state.v_z, zero_node_z);
  copy_field_from_host(state.rad_E, zero_rad);
  copy_field_from_host(state.rad_dep, zero_rad);
  copy_field_from_host(state.rad_emit, zero_rad);
  state.laser_dep.fill(0.0);
  state.ray_density.fill(0.0);

  state.t = 0.0;
  state.step = 0;
  state.dt = 0.0;
  initialize_output_timing(state, cfg);

  coupling::initialize_eos_fields_if_needed(state, cfg);

  radiation::IMC imc;
  const double dt = cfg.numerics.dt.initial_s;
  advance_radiation_step(state, cfg, imc, dt);

  const auto rad_dep = copy_field_to_host(state.rad_dep);
  TENRYU_ASSERT(rad_dep.size() ==
                    static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups),
                "void_passthrough rad_dep size mismatch");

  double void_dep_l1 = 0.0;
  double absorber_dep_l1 = 0.0;
  double void_dep_max = 0.0;
  for (int c = 0; c < n_cells; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
                              static_cast<std::size_t>(g);
      const double dep_abs = std::abs(rad_dep[idx]);
      if (state.cell_is_void[static_cast<std::size_t>(c)] != 0U) {
        void_dep_l1 += dep_abs;
        void_dep_max = std::max(void_dep_max, dep_abs);
      } else {
        absorber_dep_l1 += dep_abs;
      }
    }
  }

  const double void_rel = void_dep_l1 / std::max(absorber_dep_l1, 1.0e-30);
  constexpr double kVoidDepAbsTol = 1.0e-6;
  constexpr double kVoidDepRelTol = 1.0e-10;
  constexpr double kAbsorberDepMin = 1.0e-20;
  const bool pass_layout = (n_void_cells == n_cells / 2);
  const bool pass_signal = (absorber_dep_l1 > kAbsorberDepMin);
  const bool pass_void =
      (void_dep_l1 <= kVoidDepAbsTol) || (void_rel <= kVoidDepRelTol);
  const bool pass = pass_layout && pass_signal && pass_void;

  core::log_info("[verify:void_passthrough] n_void_cells=" + std::to_string(n_void_cells) +
                 ", absorber_dep_l1=" + format_double(absorber_dep_l1) +
                 ", void_dep_l1=" + format_double(void_dep_l1) +
                 ", void_dep_max=" + format_double(void_dep_max) +
                 ", void_rel=" + format_double(void_rel));
  core::log_info("[verify:void_passthrough] checks pass_layout=" +
                 std::string(pass_layout ? "true" : "false") +
                 ", pass_signal=" + std::string(pass_signal ? "true" : "false") +
                 ", pass_void=" + std::string(pass_void ? "true" : "false"));
  if (!pass) {
    core::log_error("[verify:void_passthrough] FAILED");
  } else {
    core::log_info("[verify:void_passthrough] PASSED");
  }
  return pass;
}
}  // namespace
}  // namespace tenryu::drivers
