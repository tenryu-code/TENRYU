// Extract (a reference copy, not a compilable unit): the configuration front end of the retired 1D ALE
// (Numerics.ale1d), cut out of files that stayed in the build, as they were at 6d62bf929.
//
//   - src/core/config.hpp: Config::NumericsConfig::Ale1dConfig and the NumericsConfig member `ale1d`.
//   - src/core/config_validate.hpp: validate_ale1d_config, called in Builder::validate
//     (builder.cpp line 19706: `tenryu::core::validate_ale1d_config(config);`).
//   - src/core/namelist/builder.cpp: the Numerics.ale1d block of Builder::set_numerics, and the
//     Radiation.multigroup_diffusion.hydro_coupling="conservative_advection" check that refused the 1D ALE.
//   - src/core/namelist/freeze.cpp: the frozen-configuration serialization of Numerics.ale1d (inside
//     serialize_numerics, which also set `out["ale1d"] = ale1d;`), and the defaults a frozen configuration
//     written before a key existed took (inside apply_legacy_numerics_defaults).

// ---- src/core/config.hpp, lines 2595-2746 at 6d62bf929 ----
    struct Ale1dConfig {
      struct LaserSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.060;
        int sigma_min_cells = 4;
        int sigma_max_cells = 16;
        double peak_fraction = 0.35;
        double conf_low = 0.10;
        double conf_high = 0.40;
      };

      struct AblationSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.080;
        int sigma_min_cells = 3;
        int sigma_max_cells = 14;
        double peak_fraction = 0.40;
        double reference_density_gcc = 1.05;
        double rho_gate_frac = 0.07;
        double rho_gate_width = 0.02;
        double te_gate_low_eV = 0.5;
        double te_gate_high_eV = 2.0;
        double conf_low = 0.10;
        double conf_high = 0.35;
      };

      struct ShockSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.040;
        int sigma_min_cells = 2;
        int sigma_max_cells = 8;
        double peak_fraction = 0.35;
        double qvisc_conf_low = 0.03;
        double qvisc_conf_high = 0.10;
        double du_cs_conf_low = 0.03;
        double du_cs_conf_high = 0.15;
      };

      struct InterfaceSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.033;
        double target_cells_cap_fraction = 0.067;
        int max_features = 8;
        int min_separation_cells = 4;
        double jump_low = 0.05;
        double jump_high = 0.25;
        int sigma_min_cells = 2;
        int sigma_max_cells = 4;
        bool pin_interfaces = true;
      };

      struct CenterSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.053;
        int sigma_min_cells = 6;
        int sigma_max_cells = 20;
        double search_x = 0.12;
      };

      struct RezoneConfig {
        double monitor_floor = 1.0;
        double monitor_wmax_ratio = 50.0;
        int monitor_smoothing_iterations = 2;
        bool monitor_smooth_across_protected_faces = false;
        double min_floor_fraction = 0.55;
        double gaussian_truncation_sigma = 3.0;

        bool spatial_monitor_enabled = true;
        double spatial_target_cells_fraction = 0.067;
        double spatial_power = 2.0;
        double laser_spatial_dr_min_cm = 2.5e-5;
        double laser_spatial_dr_max_cm = 2.0e-4;
        double ablation_spatial_dr_min_cm = 1.5e-5;
        double ablation_spatial_dr_max_cm = 1.2e-4;
        double shock_spatial_dr_min_cm = 1.0e-5;
        double shock_spatial_dr_max_cm = 8.0e-5;
      };

      struct MinWidthFloorConfig {
        bool enabled = false;
        double floor_cm = 0.0;          // trigger + guarantee: no cell below this after rezone
        double target_factor = 1.25;    // respace target = target_factor * floor_cm
        int relief_halfwidth_cells = 3;  // half-width of the minimum-cell relief neighborhood
        double max_growth_factor = 1.8;  // per-application cap: no cell grows more than this per rezone
        int retrigger_cooldown_steps = 0;  // after a floor-triggered attempt is not applied, skip the floor-trigger evaluation for this many steps (0 = evaluate every step)
      };

      struct RemapConfig {
        bool reject_multicell_sweeps = true;
        bool high_order_enabled = true;
        double limiter_theta = 1.5;
        int high_order_ramp_cells = 2;
        int radiation_high_order_ramp_cells = 2;
        bool fallback_to_first_order_on_bounds_fail = true;
        bool reject_strict_zero_flux_on_moving_protected_face = true;
      };

      bool enabled = false;  // EXPERIMENTAL: opt-in only

      // Trigger
      int every_n_steps = 100;
      int min_steps_between_ale = 50;
      // Candidate gates on the acoustic time-step bound min_i dr_i / c_s,i
      // (NUMERICS §3.4.1): an attempt triggered by the cadence alone must
      // raise it by benefit_min_dt_gain when enable_benefit_gate, and no
      // candidate may lower it by more than candidate_dt_penalty_max.
      bool enable_benefit_gate = true;
      double benefit_min_dt_gain = 1.5;
      double candidate_dt_penalty_max = 1.25;
      // Mesh-quality trigger: the largest adjacent cell-width ratio above
      // emergency_max_dr_ratio (2026-09-23; this used
      // candidate_dt_penalty_max as the threshold).
      bool emergency_enabled = true;
      double emergency_max_dr_ratio = 1.25;

      // Eligibility guards
      int min_cells = 256;
      double protected_fraction_max = 0.25;
      int min_movable_segment_warn = 24;
      int min_movable_segment_hard = 8;
      double max_node_displacement_fraction_mu = 0.35;
      double max_node_displacement_fraction_r = 0.35;
      bool ke_conservation_closure = false;

      // Conservation tolerances
      struct Tol {
        double soft = 0.0;
        double hard = 0.0;
      };
      Tol total_mass_tol{1e-12, 1e-9};
      Tol material_mass_tol{1e-11, 1e-8};
      Tol radiation_group_energy_tol{1e-8, 1e-5};
      Tol material_internal_energy_tol{1e-8, 1e-5};
      Tol total_material_energy_tol{1e-7, 1e-5};
      Tol global_total_energy_tol{1e-6, 1e-4};
      Tol kinetic_energy_drift_tol{1e-7, 1e-5};

      // Diagnostics
      bool diagnostics_enabled = true;
      int diagnostics_log_every_n_steps = 100;
      bool diagnostics_collect_step_result = true;
      bool diagnostics_fail_on_unexpected_apply = false;

      LaserSensorConfig laser_sensor;
      AblationSensorConfig ablation_sensor;
      ShockSensorConfig shock_sensor;
      InterfaceSensorConfig interface_sensor;
      CenterSensorConfig center_sensor;
      RezoneConfig rezone;
      MinWidthFloorConfig min_width_floor;
      RemapConfig remap;
    };

// ---- src/core/config.hpp, line 2814 at 6d62bf929 ----
    Ale1dConfig ale1d;

// ---- src/core/config_validate.hpp, lines 1222-1412 at 6d62bf929 ----
inline void validate_ale1d_config(const Config& config) {
  const auto& main = config.main;
  const auto& radiation = config.radiation;
  const auto& ale = config.numerics.ale1d;
  const auto& rezone = ale.rezone;
  const auto& min_width_floor = ale.min_width_floor;
  const auto& remap = ale.remap;
  const auto validate_tol = [](const auto& tol, const char* path) {
    if (tol.soft > tol.hard) {
      throw namelist::ConfigError(std::string(path) + ".soft must be <= hard");
    }
  };
  const auto validate_range = [](const double lo,
                                 const double hi,
                                 const char* lo_path,
                                 const char* hi_path) {
    if (!(lo > 0.0) || !(hi > 0.0) || lo > hi) {
      throw namelist::ConfigError(std::string(lo_path) + "/" + hi_path +
                                  " must be positive with min <= max");
    }
  };
  if (!(ale.emergency_max_dr_ratio >= 1.0)) {
    throw namelist::ConfigError("Numerics.ale1d.emergency_max_dr_ratio must be >= 1");
  }
  if (!(ale.candidate_dt_penalty_max >= 1.0)) {
    throw namelist::ConfigError("Numerics.ale1d.candidate_dt_penalty_max must be >= 1");
  }
  if (!(ale.benefit_min_dt_gain >= 1.0)) {
    throw namelist::ConfigError("Numerics.ale1d.benefit_min_dt_gain must be >= 1");
  }
  validate_tol(ale.total_mass_tol, "Numerics.ale1d.total_mass_tol");
  validate_tol(ale.material_mass_tol, "Numerics.ale1d.material_mass_tol");
  validate_tol(ale.radiation_group_energy_tol,
               "Numerics.ale1d.radiation_group_energy_tol");
  validate_tol(ale.material_internal_energy_tol,
               "Numerics.ale1d.material_internal_energy_tol");
  validate_tol(ale.total_material_energy_tol,
               "Numerics.ale1d.total_material_energy_tol");
  validate_tol(ale.global_total_energy_tol,
               "Numerics.ale1d.global_total_energy_tol");
  validate_tol(ale.kinetic_energy_drift_tol,
               "Numerics.ale1d.kinetic_energy_drift_tol");

  if (!(ale.max_node_displacement_fraction_mu > 0.0 &&
        ale.max_node_displacement_fraction_mu < 0.5)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.max_node_displacement_fraction_mu must be in (0, 0.5)");
  }
  if (!(ale.max_node_displacement_fraction_r > 0.0 &&
        ale.max_node_displacement_fraction_r < 0.5)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.max_node_displacement_fraction_r must be in (0, 0.5)");
  }
  if (!(ale.protected_fraction_max > 0.0 &&
        ale.protected_fraction_max < 0.5)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.protected_fraction_max must be in (0, 0.5)");
  }
  if (ale.min_movable_segment_hard < 4) {
    throw namelist::ConfigError(
        "Numerics.ale1d.min_movable_segment_hard must be >= 4");
  }
  if (ale.min_movable_segment_warn < ale.min_movable_segment_hard) {
    throw namelist::ConfigError(
        "Numerics.ale1d.min_movable_segment_warn must be >= min_movable_segment_hard");
  }
  if (ale.every_n_steps < 1) {
    throw namelist::ConfigError("Numerics.ale1d.every_n_steps must be >= 1");
  }
  if (!(rezone.monitor_floor > 0.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.monitor_floor must be positive");
  }
  if (!(rezone.monitor_wmax_ratio >= 1.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.monitor_wmax_ratio must be >= 1");
  }
  if (rezone.monitor_smoothing_iterations < 0) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.monitor_smoothing_iterations must be >= 0");
  }
  if (!(rezone.min_floor_fraction > 0.0 && rezone.min_floor_fraction < 1.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.min_floor_fraction must be in (0, 1)");
  }
  if (!(rezone.gaussian_truncation_sigma > 0.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.gaussian_truncation_sigma must be positive");
  }
  if (!(rezone.spatial_target_cells_fraction >= 0.0 &&
        rezone.spatial_target_cells_fraction < 1.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.spatial_target_cells_fraction must be in [0, 1)");
  }
  if (!(rezone.spatial_power > 0.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.spatial_power must be positive");
  }
  validate_range(rezone.laser_spatial_dr_min_cm,
                 rezone.laser_spatial_dr_max_cm,
                 "Numerics.ale1d.rezone.laser_spatial_dr_min_cm",
                 "Numerics.ale1d.rezone.laser_spatial_dr_max_cm");
  validate_range(rezone.ablation_spatial_dr_min_cm,
                 rezone.ablation_spatial_dr_max_cm,
                 "Numerics.ale1d.rezone.ablation_spatial_dr_min_cm",
                 "Numerics.ale1d.rezone.ablation_spatial_dr_max_cm");
  validate_range(rezone.shock_spatial_dr_min_cm,
                 rezone.shock_spatial_dr_max_cm,
                 "Numerics.ale1d.rezone.shock_spatial_dr_min_cm",
                 "Numerics.ale1d.rezone.shock_spatial_dr_max_cm");
  if (min_width_floor.enabled) {
    if (!(min_width_floor.floor_cm > 0.0)) {
      throw namelist::ConfigError(
          "Numerics.ale1d.min_width_floor.floor_cm must be positive");
    }
    if (!(min_width_floor.target_factor > 1.0)) {
      throw namelist::ConfigError(
          "Numerics.ale1d.min_width_floor.target_factor must be > 1");
    }
    if (min_width_floor.relief_halfwidth_cells < 1) {
      throw namelist::ConfigError(
          "Numerics.ale1d.min_width_floor.relief_halfwidth_cells must be >= 1");
    }
    if (!(min_width_floor.max_growth_factor > 1.0)) {
      throw namelist::ConfigError(
          "Numerics.ale1d.min_width_floor.max_growth_factor must be > 1");
    }
    if (!(min_width_floor.max_growth_factor <= 2.0)) {
      throw namelist::ConfigError(
          "Numerics.ale1d.min_width_floor.max_growth_factor must be <= 2");
    }
    if (min_width_floor.retrigger_cooldown_steps < 0) {
      throw namelist::ConfigError(
          "Numerics.ale1d.min_width_floor.retrigger_cooldown_steps must be >= 0");
    }
  }
  if (!(remap.limiter_theta > 0.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.remap.limiter_theta must be positive");
  }
  if (remap.high_order_ramp_cells < 0) {
    throw namelist::ConfigError(
        "Numerics.ale1d.remap.high_order_ramp_cells must be >= 0");
  }
  if (remap.radiation_high_order_ramp_cells < 0) {
    throw namelist::ConfigError(
        "Numerics.ale1d.remap.radiation_high_order_ramp_cells must be >= 0");
  }

  if (!ale.enabled) {
    return;
  }

  if (main.dimension != "1D_SPH") {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True is supported only in 1D_SPH");
  }
  if (main.dim != 1) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True requires Main.dim=1");
  }
  // 2026-07-26 review: fail-closed operating boundary for
  // the experimental V3 ALE prototype: the remap has no companion-field
  // registry (S_N angular state and burn inventories are left on the old mesh),
  // and post-remap EOS closure remains single-material. Reject configurations
  // outside the validated single-material + (hydro|FLD) envelope instead of
  // silently corrupting them.
  if (radiation.enabled && radiation.mode == RadiationMode::SnTransport) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True does not support mode=\"sn_transport\" "
        "(S_N angular state is not remapped; use multigroup_diffusion or "
        "disable ALE1D)");
  }
  if (config.burn.enabled) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True does not support Burn.enabled=True "
        "(burn inventories are not remapped)");
  }
  if (config.materials.materials.size() != 1) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True supports exactly one material "
        "(multi-material volFrac/companion-field remap is not certified)");
  }
  const auto& ale_material = config.materials.materials.front();
  if (ale_material.eos_model != "ideal_gas" && !ale_material.eos_tables) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True requires eos_model=\"ideal_gas\" or a "
        "table EOS (table backend rho-e reclosure capability is checked at "
        "the ALE1D driver entry)");
  }
}

// ---- src/core/namelist/builder.cpp, lines 13567-14089 at 6d62bf929 ----
  if (has_key(kwargs, "ale1d")) {
    const py::handle ale1d_obj = kwargs["ale1d"];
    if (!py::isinstance<py::dict>(ale1d_obj)) {
      throw_value_type_error("Numerics.ale1d", "dict", ale1d_obj);
    }
    const py::dict ale1d = py::reinterpret_borrow<py::dict>(ale1d_obj);
    enforce_known_keys(ale1d, "Numerics.ale1d",
                       {"enabled", "every_n_steps", "min_steps_between_ale",
                        "enable_benefit_gate", "benefit_min_dt_gain",
                        "candidate_dt_penalty_max", "emergency_enabled",
                        "emergency_max_dr_ratio",
                        "min_cells", "protected_fraction_max",
                        "min_movable_segment_warn", "min_movable_segment_hard",
                        "max_node_displacement_fraction_mu",
                        "max_node_displacement_fraction_r",
                        "ke_conservation_closure",
                        "total_mass_tol", "material_mass_tol",
                        "radiation_group_energy_tol",
                        "material_internal_energy_tol",
                        "total_material_energy_tol", "global_total_energy_tol",
                        "kinetic_energy_drift_tol",
                        "diagnostics_enabled",
                        "diagnostics_log_every_n_steps",
                        "diagnostics_collect_step_result",
                        "diagnostics_fail_on_unexpected_apply",
                        "laser_sensor", "ablation_sensor", "shock_sensor",
                        "interface_sensor", "center_sensor", "rezone",
                        "min_width_floor", "remap"});
    auto& ale1d_cfg = numerics.ale1d;
    const auto parse_tol = [&](const char* key, auto& tol) {
      if (!has_key(ale1d, key)) {
        return;
      }
      const py::handle tol_obj = ale1d[key];
      const std::string path = std::string("Numerics.ale1d.") + key;
      if (!py::isinstance<py::dict>(tol_obj)) {
        throw_value_type_error(path, "dict", tol_obj);
      }
      const py::dict tol_dict = py::reinterpret_borrow<py::dict>(tol_obj);
      enforce_known_keys(tol_dict, path, {"soft", "hard"});
      if (has_key(tol_dict, "soft")) {
        tol.soft = numeric_as_double(tol_dict["soft"], path + ".soft");
      }
      if (has_key(tol_dict, "hard")) {
        tol.hard = numeric_as_double(tol_dict["hard"], path + ".hard");
      }
    };
    const auto parse_sensor_dict = [&](const char* key, const auto& parse_contents) {
      if (!has_key(ale1d, key)) {
        return;
      }
      const py::handle sensor_obj = ale1d[key];
      const std::string path = std::string("Numerics.ale1d.") + key;
      if (!py::isinstance<py::dict>(sensor_obj)) {
        throw_value_type_error(path, "dict", sensor_obj);
      }
      const py::dict sensor = py::reinterpret_borrow<py::dict>(sensor_obj);
      parse_contents(sensor, path);
    };
    const auto parse_rezone_dict = [&]() {
      if (!has_key(ale1d, "rezone")) {
        return;
      }
      const py::handle rezone_obj = ale1d["rezone"];
      const std::string path = "Numerics.ale1d.rezone";
      if (!py::isinstance<py::dict>(rezone_obj)) {
        throw_value_type_error(path, "dict", rezone_obj);
      }
      const py::dict rezone = py::reinterpret_borrow<py::dict>(rezone_obj);
      enforce_known_keys(
          rezone, path,
          {"monitor_floor", "monitor_wmax_ratio",
           "monitor_smoothing_iterations",
           "monitor_smooth_across_protected_faces", "min_floor_fraction",
           "gaussian_truncation_sigma", "spatial_monitor_enabled",
           "spatial_target_cells_fraction", "spatial_power",
           "laser_spatial_dr_min_cm", "laser_spatial_dr_max_cm",
           "ablation_spatial_dr_min_cm", "ablation_spatial_dr_max_cm",
           "shock_spatial_dr_min_cm", "shock_spatial_dr_max_cm"});
      auto& cfg = ale1d_cfg.rezone;
      if (has_key(rezone, "monitor_floor")) {
        cfg.monitor_floor =
            numeric_as_double(rezone["monitor_floor"], path + ".monitor_floor");
      }
      if (has_key(rezone, "monitor_wmax_ratio")) {
        cfg.monitor_wmax_ratio =
            numeric_as_double(rezone["monitor_wmax_ratio"],
                              path + ".monitor_wmax_ratio");
      }
      if (has_key(rezone, "monitor_smoothing_iterations")) {
        cfg.monitor_smoothing_iterations = strict_int32(
            rezone["monitor_smoothing_iterations"],
            path + ".monitor_smoothing_iterations");
      }
      if (has_key(rezone, "monitor_smooth_across_protected_faces")) {
        cfg.monitor_smooth_across_protected_faces = strict_bool(
            rezone["monitor_smooth_across_protected_faces"],
            path + ".monitor_smooth_across_protected_faces");
      }
      if (has_key(rezone, "min_floor_fraction")) {
        cfg.min_floor_fraction =
            numeric_as_double(rezone["min_floor_fraction"],
                              path + ".min_floor_fraction");
      }
      if (has_key(rezone, "gaussian_truncation_sigma")) {
        cfg.gaussian_truncation_sigma = numeric_as_double(
            rezone["gaussian_truncation_sigma"],
            path + ".gaussian_truncation_sigma");
      }
      if (has_key(rezone, "spatial_monitor_enabled")) {
        cfg.spatial_monitor_enabled = strict_bool(
            rezone["spatial_monitor_enabled"],
            path + ".spatial_monitor_enabled");
      }
      if (has_key(rezone, "spatial_target_cells_fraction")) {
        cfg.spatial_target_cells_fraction = numeric_as_double(
            rezone["spatial_target_cells_fraction"],
            path + ".spatial_target_cells_fraction");
      }
      if (has_key(rezone, "spatial_power")) {
        cfg.spatial_power =
            numeric_as_double(rezone["spatial_power"], path + ".spatial_power");
      }
      if (has_key(rezone, "laser_spatial_dr_min_cm")) {
        cfg.laser_spatial_dr_min_cm = numeric_as_double(
            rezone["laser_spatial_dr_min_cm"],
            path + ".laser_spatial_dr_min_cm");
      }
      if (has_key(rezone, "laser_spatial_dr_max_cm")) {
        cfg.laser_spatial_dr_max_cm = numeric_as_double(
            rezone["laser_spatial_dr_max_cm"],
            path + ".laser_spatial_dr_max_cm");
      }
      if (has_key(rezone, "ablation_spatial_dr_min_cm")) {
        cfg.ablation_spatial_dr_min_cm = numeric_as_double(
            rezone["ablation_spatial_dr_min_cm"],
            path + ".ablation_spatial_dr_min_cm");
      }
      if (has_key(rezone, "ablation_spatial_dr_max_cm")) {
        cfg.ablation_spatial_dr_max_cm = numeric_as_double(
            rezone["ablation_spatial_dr_max_cm"],
            path + ".ablation_spatial_dr_max_cm");
      }
      if (has_key(rezone, "shock_spatial_dr_min_cm")) {
        cfg.shock_spatial_dr_min_cm = numeric_as_double(
            rezone["shock_spatial_dr_min_cm"],
            path + ".shock_spatial_dr_min_cm");
      }
      if (has_key(rezone, "shock_spatial_dr_max_cm")) {
        cfg.shock_spatial_dr_max_cm = numeric_as_double(
            rezone["shock_spatial_dr_max_cm"],
            path + ".shock_spatial_dr_max_cm");
      }
    };
    const auto parse_min_width_floor_dict = [&]() {
      if (!has_key(ale1d, "min_width_floor")) {
        return;
      }
      const py::handle floor_obj = ale1d["min_width_floor"];
      const std::string path = "Numerics.ale1d.min_width_floor";
      if (!py::isinstance<py::dict>(floor_obj)) {
        throw_value_type_error(path, "dict", floor_obj);
      }
      const py::dict floor = py::reinterpret_borrow<py::dict>(floor_obj);
      enforce_known_keys(
          floor, path,
          {"enabled", "floor_cm", "target_factor", "relief_halfwidth_cells",
           "max_growth_factor", "retrigger_cooldown_steps"});
      auto& cfg = ale1d_cfg.min_width_floor;
      if (has_key(floor, "enabled")) {
        cfg.enabled = strict_bool(floor["enabled"], path + ".enabled");
      }
      if (has_key(floor, "floor_cm")) {
        cfg.floor_cm = numeric_as_double(floor["floor_cm"], path + ".floor_cm");
      }
      if (has_key(floor, "target_factor")) {
        cfg.target_factor = numeric_as_double(
            floor["target_factor"], path + ".target_factor");
      }
      if (has_key(floor, "relief_halfwidth_cells")) {
        cfg.relief_halfwidth_cells = strict_int32(
            floor["relief_halfwidth_cells"],
            path + ".relief_halfwidth_cells");
      }
      if (has_key(floor, "max_growth_factor")) {
        cfg.max_growth_factor = numeric_as_double(
            floor["max_growth_factor"], path + ".max_growth_factor");
      }
      if (has_key(floor, "retrigger_cooldown_steps")) {
        cfg.retrigger_cooldown_steps = strict_int32(
            floor["retrigger_cooldown_steps"],
            path + ".retrigger_cooldown_steps");
      }
    };
    const auto parse_remap_dict = [&]() {
      if (!has_key(ale1d, "remap")) {
        return;
      }
      const py::handle remap_obj = ale1d["remap"];
      const std::string path = "Numerics.ale1d.remap";
      if (!py::isinstance<py::dict>(remap_obj)) {
        throw_value_type_error(path, "dict", remap_obj);
      }
      const py::dict remap = py::reinterpret_borrow<py::dict>(remap_obj);
      enforce_known_keys(
          remap, path,
          {"reject_multicell_sweeps", "high_order_enabled",
           "limiter_theta", "high_order_ramp_cells",
           "radiation_high_order_ramp_cells",
           "fallback_to_first_order_on_bounds_fail",
           "reject_strict_zero_flux_on_moving_protected_face"});
      auto& cfg = ale1d_cfg.remap;
      if (has_key(remap, "reject_multicell_sweeps")) {
        cfg.reject_multicell_sweeps = strict_bool(
            remap["reject_multicell_sweeps"],
            path + ".reject_multicell_sweeps");
      }
      if (has_key(remap, "high_order_enabled")) {
        cfg.high_order_enabled =
            strict_bool(remap["high_order_enabled"],
                        path + ".high_order_enabled");
      }
      if (has_key(remap, "limiter_theta")) {
        cfg.limiter_theta =
            numeric_as_double(remap["limiter_theta"], path + ".limiter_theta");
      }
      if (has_key(remap, "high_order_ramp_cells")) {
        cfg.high_order_ramp_cells = strict_int32(
            remap["high_order_ramp_cells"], path + ".high_order_ramp_cells");
      }
      if (has_key(remap, "radiation_high_order_ramp_cells")) {
        cfg.radiation_high_order_ramp_cells =
            strict_int32(remap["radiation_high_order_ramp_cells"],
                         path + ".radiation_high_order_ramp_cells");
      }
      if (has_key(remap, "fallback_to_first_order_on_bounds_fail")) {
        cfg.fallback_to_first_order_on_bounds_fail = strict_bool(
            remap["fallback_to_first_order_on_bounds_fail"],
            path + ".fallback_to_first_order_on_bounds_fail");
      }
      if (has_key(remap, "reject_strict_zero_flux_on_moving_protected_face")) {
        cfg.reject_strict_zero_flux_on_moving_protected_face = strict_bool(
            remap["reject_strict_zero_flux_on_moving_protected_face"],
            path + ".reject_strict_zero_flux_on_moving_protected_face");
      }
    };

    if (has_key(ale1d, "enabled")) {
      ale1d_cfg.enabled = strict_bool(ale1d["enabled"], "Numerics.ale1d.enabled");
    }
    if (has_key(ale1d, "every_n_steps")) {
      ale1d_cfg.every_n_steps =
          strict_int32(ale1d["every_n_steps"], "Numerics.ale1d.every_n_steps");
    }
    if (has_key(ale1d, "min_steps_between_ale")) {
      ale1d_cfg.min_steps_between_ale = strict_int32(
          ale1d["min_steps_between_ale"], "Numerics.ale1d.min_steps_between_ale");
    }
    if (has_key(ale1d, "enable_benefit_gate")) {
      ale1d_cfg.enable_benefit_gate = strict_bool(
          ale1d["enable_benefit_gate"], "Numerics.ale1d.enable_benefit_gate");
    }
    if (has_key(ale1d, "benefit_min_dt_gain")) {
      ale1d_cfg.benefit_min_dt_gain = numeric_as_double(
          ale1d["benefit_min_dt_gain"], "Numerics.ale1d.benefit_min_dt_gain");
    }
    if (has_key(ale1d, "candidate_dt_penalty_max")) {
      ale1d_cfg.candidate_dt_penalty_max = numeric_as_double(
          ale1d["candidate_dt_penalty_max"], "Numerics.ale1d.candidate_dt_penalty_max");
    }
    if (has_key(ale1d, "emergency_enabled")) {
      ale1d_cfg.emergency_enabled = strict_bool(
          ale1d["emergency_enabled"], "Numerics.ale1d.emergency_enabled");
    }
    if (has_key(ale1d, "emergency_max_dr_ratio")) {
      ale1d_cfg.emergency_max_dr_ratio = numeric_as_double(
          ale1d["emergency_max_dr_ratio"], "Numerics.ale1d.emergency_max_dr_ratio");
    }
    if (has_key(ale1d, "min_cells")) {
      ale1d_cfg.min_cells =
          strict_int32(ale1d["min_cells"], "Numerics.ale1d.min_cells");
    }
    if (has_key(ale1d, "protected_fraction_max")) {
      ale1d_cfg.protected_fraction_max = numeric_as_double(
          ale1d["protected_fraction_max"], "Numerics.ale1d.protected_fraction_max");
    }
    if (has_key(ale1d, "min_movable_segment_warn")) {
      ale1d_cfg.min_movable_segment_warn = strict_int32(
          ale1d["min_movable_segment_warn"], "Numerics.ale1d.min_movable_segment_warn");
    }
    if (has_key(ale1d, "min_movable_segment_hard")) {
      ale1d_cfg.min_movable_segment_hard = strict_int32(
          ale1d["min_movable_segment_hard"], "Numerics.ale1d.min_movable_segment_hard");
    }
    if (has_key(ale1d, "max_node_displacement_fraction_mu")) {
      ale1d_cfg.max_node_displacement_fraction_mu = numeric_as_double(
          ale1d["max_node_displacement_fraction_mu"],
          "Numerics.ale1d.max_node_displacement_fraction_mu");
    }
    if (has_key(ale1d, "max_node_displacement_fraction_r")) {
      ale1d_cfg.max_node_displacement_fraction_r = numeric_as_double(
          ale1d["max_node_displacement_fraction_r"],
          "Numerics.ale1d.max_node_displacement_fraction_r");
    }
    if (has_key(ale1d, "ke_conservation_closure")) {
      ale1d_cfg.ke_conservation_closure = strict_bool(
          ale1d["ke_conservation_closure"],
          "Numerics.ale1d.ke_conservation_closure");
    }
    parse_tol("total_mass_tol", ale1d_cfg.total_mass_tol);
    parse_tol("material_mass_tol", ale1d_cfg.material_mass_tol);
    parse_tol("radiation_group_energy_tol", ale1d_cfg.radiation_group_energy_tol);
    parse_tol("material_internal_energy_tol", ale1d_cfg.material_internal_energy_tol);
    parse_tol("total_material_energy_tol", ale1d_cfg.total_material_energy_tol);
    parse_tol("global_total_energy_tol", ale1d_cfg.global_total_energy_tol);
    parse_tol("kinetic_energy_drift_tol", ale1d_cfg.kinetic_energy_drift_tol);
    if (has_key(ale1d, "diagnostics_enabled")) {
      ale1d_cfg.diagnostics_enabled = strict_bool(
          ale1d["diagnostics_enabled"], "Numerics.ale1d.diagnostics_enabled");
    }
    if (has_key(ale1d, "diagnostics_log_every_n_steps")) {
      ale1d_cfg.diagnostics_log_every_n_steps = strict_int32(
          ale1d["diagnostics_log_every_n_steps"],
          "Numerics.ale1d.diagnostics_log_every_n_steps");
    }
    if (has_key(ale1d, "diagnostics_collect_step_result")) {
      ale1d_cfg.diagnostics_collect_step_result = strict_bool(
          ale1d["diagnostics_collect_step_result"],
          "Numerics.ale1d.diagnostics_collect_step_result");
    }
    if (has_key(ale1d, "diagnostics_fail_on_unexpected_apply")) {
      ale1d_cfg.diagnostics_fail_on_unexpected_apply = strict_bool(
          ale1d["diagnostics_fail_on_unexpected_apply"],
          "Numerics.ale1d.diagnostics_fail_on_unexpected_apply");
    }
    parse_sensor_dict("laser_sensor", [&](const py::dict& sensor,
                                           const std::string& path) {
      auto& cfg = ale1d_cfg.laser_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "sigma_min_cells",
                          "sigma_max_cells", "peak_fraction", "conf_low", "conf_high"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "peak_fraction")) {
        cfg.peak_fraction = numeric_as_double(sensor["peak_fraction"], path + ".peak_fraction");
      }
      if (has_key(sensor, "conf_low")) {
        cfg.conf_low = numeric_as_double(sensor["conf_low"], path + ".conf_low");
      }
      if (has_key(sensor, "conf_high")) {
        cfg.conf_high = numeric_as_double(sensor["conf_high"], path + ".conf_high");
      }
    });
    parse_sensor_dict("ablation_sensor", [&](const py::dict& sensor,
                                              const std::string& path) {
      auto& cfg = ale1d_cfg.ablation_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "sigma_min_cells",
                          "sigma_max_cells", "peak_fraction", "reference_density_gcc",
                          "rho_gate_frac", "rho_gate_width", "te_gate_low_eV",
                          "te_gate_high_eV", "conf_low", "conf_high"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "peak_fraction")) {
        cfg.peak_fraction = numeric_as_double(sensor["peak_fraction"], path + ".peak_fraction");
      }
      if (has_key(sensor, "reference_density_gcc")) {
        cfg.reference_density_gcc =
            numeric_as_double(sensor["reference_density_gcc"], path + ".reference_density_gcc");
      }
      if (has_key(sensor, "rho_gate_frac")) {
        cfg.rho_gate_frac = numeric_as_double(sensor["rho_gate_frac"], path + ".rho_gate_frac");
      }
      if (has_key(sensor, "rho_gate_width")) {
        cfg.rho_gate_width =
            numeric_as_double(sensor["rho_gate_width"], path + ".rho_gate_width");
      }
      if (has_key(sensor, "te_gate_low_eV")) {
        cfg.te_gate_low_eV =
            numeric_as_double(sensor["te_gate_low_eV"], path + ".te_gate_low_eV");
      }
      if (has_key(sensor, "te_gate_high_eV")) {
        cfg.te_gate_high_eV =
            numeric_as_double(sensor["te_gate_high_eV"], path + ".te_gate_high_eV");
      }
      if (has_key(sensor, "conf_low")) {
        cfg.conf_low = numeric_as_double(sensor["conf_low"], path + ".conf_low");
      }
      if (has_key(sensor, "conf_high")) {
        cfg.conf_high = numeric_as_double(sensor["conf_high"], path + ".conf_high");
      }
    });
    parse_sensor_dict("shock_sensor", [&](const py::dict& sensor,
                                           const std::string& path) {
      auto& cfg = ale1d_cfg.shock_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "sigma_min_cells",
                          "sigma_max_cells", "peak_fraction", "qvisc_conf_low",
                          "qvisc_conf_high", "du_cs_conf_low", "du_cs_conf_high"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "peak_fraction")) {
        cfg.peak_fraction = numeric_as_double(sensor["peak_fraction"], path + ".peak_fraction");
      }
      if (has_key(sensor, "qvisc_conf_low")) {
        cfg.qvisc_conf_low =
            numeric_as_double(sensor["qvisc_conf_low"], path + ".qvisc_conf_low");
      }
      if (has_key(sensor, "qvisc_conf_high")) {
        cfg.qvisc_conf_high =
            numeric_as_double(sensor["qvisc_conf_high"], path + ".qvisc_conf_high");
      }
      if (has_key(sensor, "du_cs_conf_low")) {
        cfg.du_cs_conf_low =
            numeric_as_double(sensor["du_cs_conf_low"], path + ".du_cs_conf_low");
      }
      if (has_key(sensor, "du_cs_conf_high")) {
        cfg.du_cs_conf_high =
            numeric_as_double(sensor["du_cs_conf_high"], path + ".du_cs_conf_high");
      }
    });
    parse_sensor_dict("interface_sensor", [&](const py::dict& sensor,
                                               const std::string& path) {
      auto& cfg = ale1d_cfg.interface_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "target_cells_cap_fraction",
                          "max_features", "min_separation_cells", "jump_low",
                          "jump_high", "sigma_min_cells", "sigma_max_cells",
                          "pin_interfaces"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "target_cells_cap_fraction")) {
        cfg.target_cells_cap_fraction = numeric_as_double(
            sensor["target_cells_cap_fraction"], path + ".target_cells_cap_fraction");
      }
      if (has_key(sensor, "max_features")) {
        cfg.max_features = strict_int32(sensor["max_features"], path + ".max_features");
      }
      if (has_key(sensor, "min_separation_cells")) {
        cfg.min_separation_cells =
            strict_int32(sensor["min_separation_cells"], path + ".min_separation_cells");
      }
      if (has_key(sensor, "jump_low")) {
        cfg.jump_low = numeric_as_double(sensor["jump_low"], path + ".jump_low");
      }
      if (has_key(sensor, "jump_high")) {
        cfg.jump_high = numeric_as_double(sensor["jump_high"], path + ".jump_high");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "pin_interfaces")) {
        cfg.pin_interfaces = strict_bool(sensor["pin_interfaces"], path + ".pin_interfaces");
      }
    });
    parse_sensor_dict("center_sensor", [&](const py::dict& sensor,
                                            const std::string& path) {
      auto& cfg = ale1d_cfg.center_sensor;
      enforce_known_keys(sensor, path,
                         {"enabled", "target_cells_fraction", "sigma_min_cells",
                          "sigma_max_cells", "search_x"});
      if (has_key(sensor, "enabled")) {
        cfg.enabled = strict_bool(sensor["enabled"], path + ".enabled");
      }
      if (has_key(sensor, "target_cells_fraction")) {
        cfg.target_cells_fraction =
            numeric_as_double(sensor["target_cells_fraction"], path + ".target_cells_fraction");
      }
      if (has_key(sensor, "sigma_min_cells")) {
        cfg.sigma_min_cells = strict_int32(sensor["sigma_min_cells"], path + ".sigma_min_cells");
      }
      if (has_key(sensor, "sigma_max_cells")) {
        cfg.sigma_max_cells = strict_int32(sensor["sigma_max_cells"], path + ".sigma_max_cells");
      }
      if (has_key(sensor, "search_x")) {
        cfg.search_x = numeric_as_double(sensor["search_x"], path + ".search_x");
      }
    });
    parse_rezone_dict();
    parse_min_width_floor_dict();
    parse_remap_dict();
  }

// ---- src/core/namelist/builder.cpp, lines 18129-18135 at 6d62bf929 ----
  if (radiation.multigroup_diffusion.hydro_coupling == "conservative_advection" &&
      (main.dim != 1 || radiation.mode != RadiationMode::MultigroupDiffusion ||
       mesh.motion != "lagrangian" || numerics.ale1d.enabled ||
       numerics.persistent_loop.enabled)) {
    throw ConfigError(
        "conservative_advection requires 1D Lagrangian FLD without ALE1D or persistent loop");
  }

// ---- src/core/namelist/freeze.cpp, lines 2725-2909 at 6d62bf929 ----
  auto serialize_tol = [](const auto& tol) {
    py::dict out;
    out["soft"] = tol.soft;
    out["hard"] = tol.hard;
    return out;
  };

  py::dict ale1d;
  ale1d["enabled"] = numerics.ale1d.enabled;
  ale1d["every_n_steps"] = numerics.ale1d.every_n_steps;
  ale1d["min_steps_between_ale"] = numerics.ale1d.min_steps_between_ale;
  ale1d["enable_benefit_gate"] = numerics.ale1d.enable_benefit_gate;
  ale1d["benefit_min_dt_gain"] = numerics.ale1d.benefit_min_dt_gain;
  ale1d["candidate_dt_penalty_max"] =
      numerics.ale1d.candidate_dt_penalty_max;
  ale1d["emergency_enabled"] = numerics.ale1d.emergency_enabled;
  ale1d["emergency_max_dr_ratio"] = numerics.ale1d.emergency_max_dr_ratio;
  ale1d["min_cells"] = numerics.ale1d.min_cells;
  ale1d["protected_fraction_max"] = numerics.ale1d.protected_fraction_max;
  ale1d["min_movable_segment_warn"] =
      numerics.ale1d.min_movable_segment_warn;
  ale1d["min_movable_segment_hard"] =
      numerics.ale1d.min_movable_segment_hard;
  ale1d["max_node_displacement_fraction_mu"] =
      numerics.ale1d.max_node_displacement_fraction_mu;
  ale1d["max_node_displacement_fraction_r"] =
      numerics.ale1d.max_node_displacement_fraction_r;
  ale1d["ke_conservation_closure"] =
      numerics.ale1d.ke_conservation_closure;
  ale1d["total_mass_tol"] = serialize_tol(numerics.ale1d.total_mass_tol);
  ale1d["material_mass_tol"] = serialize_tol(numerics.ale1d.material_mass_tol);
  ale1d["radiation_group_energy_tol"] =
      serialize_tol(numerics.ale1d.radiation_group_energy_tol);
  ale1d["material_internal_energy_tol"] =
      serialize_tol(numerics.ale1d.material_internal_energy_tol);
  ale1d["total_material_energy_tol"] =
      serialize_tol(numerics.ale1d.total_material_energy_tol);
  ale1d["global_total_energy_tol"] =
      serialize_tol(numerics.ale1d.global_total_energy_tol);
  ale1d["kinetic_energy_drift_tol"] =
      serialize_tol(numerics.ale1d.kinetic_energy_drift_tol);
  ale1d["diagnostics_enabled"] = numerics.ale1d.diagnostics_enabled;
  ale1d["diagnostics_log_every_n_steps"] =
      numerics.ale1d.diagnostics_log_every_n_steps;
  ale1d["diagnostics_collect_step_result"] =
      numerics.ale1d.diagnostics_collect_step_result;
  ale1d["diagnostics_fail_on_unexpected_apply"] =
      numerics.ale1d.diagnostics_fail_on_unexpected_apply;
  py::dict laser_sensor;
  laser_sensor["enabled"] = numerics.ale1d.laser_sensor.enabled;
  laser_sensor["target_cells_fraction"] =
      numerics.ale1d.laser_sensor.target_cells_fraction;
  laser_sensor["sigma_min_cells"] = numerics.ale1d.laser_sensor.sigma_min_cells;
  laser_sensor["sigma_max_cells"] = numerics.ale1d.laser_sensor.sigma_max_cells;
  laser_sensor["peak_fraction"] = numerics.ale1d.laser_sensor.peak_fraction;
  laser_sensor["conf_low"] = numerics.ale1d.laser_sensor.conf_low;
  laser_sensor["conf_high"] = numerics.ale1d.laser_sensor.conf_high;
  ale1d["laser_sensor"] = laser_sensor;

  py::dict ablation_sensor;
  ablation_sensor["enabled"] = numerics.ale1d.ablation_sensor.enabled;
  ablation_sensor["target_cells_fraction"] =
      numerics.ale1d.ablation_sensor.target_cells_fraction;
  ablation_sensor["sigma_min_cells"] =
      numerics.ale1d.ablation_sensor.sigma_min_cells;
  ablation_sensor["sigma_max_cells"] =
      numerics.ale1d.ablation_sensor.sigma_max_cells;
  ablation_sensor["peak_fraction"] =
      numerics.ale1d.ablation_sensor.peak_fraction;
  ablation_sensor["reference_density_gcc"] =
      numerics.ale1d.ablation_sensor.reference_density_gcc;
  ablation_sensor["rho_gate_frac"] =
      numerics.ale1d.ablation_sensor.rho_gate_frac;
  ablation_sensor["rho_gate_width"] =
      numerics.ale1d.ablation_sensor.rho_gate_width;
  ablation_sensor["te_gate_low_eV"] =
      numerics.ale1d.ablation_sensor.te_gate_low_eV;
  ablation_sensor["te_gate_high_eV"] =
      numerics.ale1d.ablation_sensor.te_gate_high_eV;
  ablation_sensor["conf_low"] = numerics.ale1d.ablation_sensor.conf_low;
  ablation_sensor["conf_high"] = numerics.ale1d.ablation_sensor.conf_high;
  ale1d["ablation_sensor"] = ablation_sensor;

  py::dict shock_sensor;
  shock_sensor["enabled"] = numerics.ale1d.shock_sensor.enabled;
  shock_sensor["target_cells_fraction"] =
      numerics.ale1d.shock_sensor.target_cells_fraction;
  shock_sensor["sigma_min_cells"] = numerics.ale1d.shock_sensor.sigma_min_cells;
  shock_sensor["sigma_max_cells"] = numerics.ale1d.shock_sensor.sigma_max_cells;
  shock_sensor["peak_fraction"] = numerics.ale1d.shock_sensor.peak_fraction;
  shock_sensor["qvisc_conf_low"] = numerics.ale1d.shock_sensor.qvisc_conf_low;
  shock_sensor["qvisc_conf_high"] = numerics.ale1d.shock_sensor.qvisc_conf_high;
  shock_sensor["du_cs_conf_low"] = numerics.ale1d.shock_sensor.du_cs_conf_low;
  shock_sensor["du_cs_conf_high"] = numerics.ale1d.shock_sensor.du_cs_conf_high;
  ale1d["shock_sensor"] = shock_sensor;

  py::dict interface_sensor;
  interface_sensor["enabled"] = numerics.ale1d.interface_sensor.enabled;
  interface_sensor["target_cells_fraction"] =
      numerics.ale1d.interface_sensor.target_cells_fraction;
  interface_sensor["target_cells_cap_fraction"] =
      numerics.ale1d.interface_sensor.target_cells_cap_fraction;
  interface_sensor["max_features"] = numerics.ale1d.interface_sensor.max_features;
  interface_sensor["min_separation_cells"] =
      numerics.ale1d.interface_sensor.min_separation_cells;
  interface_sensor["jump_low"] = numerics.ale1d.interface_sensor.jump_low;
  interface_sensor["jump_high"] = numerics.ale1d.interface_sensor.jump_high;
  interface_sensor["sigma_min_cells"] =
      numerics.ale1d.interface_sensor.sigma_min_cells;
  interface_sensor["sigma_max_cells"] =
      numerics.ale1d.interface_sensor.sigma_max_cells;
  interface_sensor["pin_interfaces"] =
      numerics.ale1d.interface_sensor.pin_interfaces;
  ale1d["interface_sensor"] = interface_sensor;

  py::dict center_sensor;
  center_sensor["enabled"] = numerics.ale1d.center_sensor.enabled;
  center_sensor["target_cells_fraction"] =
      numerics.ale1d.center_sensor.target_cells_fraction;
  center_sensor["sigma_min_cells"] = numerics.ale1d.center_sensor.sigma_min_cells;
  center_sensor["sigma_max_cells"] = numerics.ale1d.center_sensor.sigma_max_cells;
  center_sensor["search_x"] = numerics.ale1d.center_sensor.search_x;
  ale1d["center_sensor"] = center_sensor;

  py::dict ale1d_rezone;
  ale1d_rezone["monitor_floor"] = numerics.ale1d.rezone.monitor_floor;
  ale1d_rezone["monitor_wmax_ratio"] =
      numerics.ale1d.rezone.monitor_wmax_ratio;
  ale1d_rezone["monitor_smoothing_iterations"] =
      numerics.ale1d.rezone.monitor_smoothing_iterations;
  ale1d_rezone["monitor_smooth_across_protected_faces"] =
      numerics.ale1d.rezone.monitor_smooth_across_protected_faces;
  ale1d_rezone["min_floor_fraction"] =
      numerics.ale1d.rezone.min_floor_fraction;
  ale1d_rezone["gaussian_truncation_sigma"] =
      numerics.ale1d.rezone.gaussian_truncation_sigma;
  ale1d_rezone["spatial_monitor_enabled"] =
      numerics.ale1d.rezone.spatial_monitor_enabled;
  ale1d_rezone["spatial_target_cells_fraction"] =
      numerics.ale1d.rezone.spatial_target_cells_fraction;
  ale1d_rezone["spatial_power"] = numerics.ale1d.rezone.spatial_power;
  ale1d_rezone["laser_spatial_dr_min_cm"] =
      numerics.ale1d.rezone.laser_spatial_dr_min_cm;
  ale1d_rezone["laser_spatial_dr_max_cm"] =
      numerics.ale1d.rezone.laser_spatial_dr_max_cm;
  ale1d_rezone["ablation_spatial_dr_min_cm"] =
      numerics.ale1d.rezone.ablation_spatial_dr_min_cm;
  ale1d_rezone["ablation_spatial_dr_max_cm"] =
      numerics.ale1d.rezone.ablation_spatial_dr_max_cm;
  ale1d_rezone["shock_spatial_dr_min_cm"] =
      numerics.ale1d.rezone.shock_spatial_dr_min_cm;
  ale1d_rezone["shock_spatial_dr_max_cm"] =
      numerics.ale1d.rezone.shock_spatial_dr_max_cm;
  ale1d["rezone"] = ale1d_rezone;

  py::dict ale1d_min_width_floor;
  ale1d_min_width_floor["enabled"] =
      numerics.ale1d.min_width_floor.enabled;
  ale1d_min_width_floor["floor_cm"] =
      numerics.ale1d.min_width_floor.floor_cm;
  ale1d_min_width_floor["target_factor"] =
      numerics.ale1d.min_width_floor.target_factor;
  ale1d_min_width_floor["relief_halfwidth_cells"] =
      numerics.ale1d.min_width_floor.relief_halfwidth_cells;
  ale1d_min_width_floor["max_growth_factor"] =
      numerics.ale1d.min_width_floor.max_growth_factor;
  ale1d_min_width_floor["retrigger_cooldown_steps"] =
      numerics.ale1d.min_width_floor.retrigger_cooldown_steps;
  ale1d["min_width_floor"] = ale1d_min_width_floor;

  py::dict ale1d_remap;
  ale1d_remap["reject_multicell_sweeps"] =
      numerics.ale1d.remap.reject_multicell_sweeps;
  ale1d_remap["high_order_enabled"] =
      numerics.ale1d.remap.high_order_enabled;
  ale1d_remap["limiter_theta"] = numerics.ale1d.remap.limiter_theta;
  ale1d_remap["high_order_ramp_cells"] =
      numerics.ale1d.remap.high_order_ramp_cells;
  ale1d_remap["radiation_high_order_ramp_cells"] =
      numerics.ale1d.remap.radiation_high_order_ramp_cells;
  ale1d_remap["fallback_to_first_order_on_bounds_fail"] =
      numerics.ale1d.remap.fallback_to_first_order_on_bounds_fail;
  ale1d_remap["reject_strict_zero_flux_on_moving_protected_face"] =
      numerics.ale1d.remap.reject_strict_zero_flux_on_moving_protected_face;
  ale1d["remap"] = ale1d_remap;

// ---- src/core/namelist/freeze.cpp, line 3215 at 6d62bf929 ----
  out["ale1d"] = ale1d;

// ---- src/core/namelist/freeze.cpp, lines 4156-4182 at 6d62bf929 ----
  // Numerics.ale1d keys added after the V3 introduction: a legacy frozen
  // config takes the struct defaults (min_width_floor 2026-08-07 and its
  // retrigger_cooldown_steps 2026-08-10, emergency_max_dr_ratio 2026-09-23),
  // so a restart from an older checkpoint compares equal.
  py::dict ale1d;
  if (try_get_child_dict(numerics, "ale1d", &ale1d)) {
    const Config::NumericsConfig::Ale1dConfig ale1d_defaults;
    set_default_if_missing(ale1d, "emergency_max_dr_ratio",
                           py::cast(ale1d_defaults.emergency_max_dr_ratio));
    if (!dict_contains(ale1d, "min_width_floor")) {
      ale1d[py::str("min_width_floor")] = py::dict();
    }
    py::dict min_width_floor;
    if (try_get_child_dict(ale1d, "min_width_floor", &min_width_floor)) {
      const auto& floor_defaults = ale1d_defaults.min_width_floor;
      set_default_if_missing(min_width_floor, "enabled", py::cast(floor_defaults.enabled));
      set_default_if_missing(min_width_floor, "floor_cm", py::cast(floor_defaults.floor_cm));
      set_default_if_missing(min_width_floor, "target_factor",
                             py::cast(floor_defaults.target_factor));
      set_default_if_missing(min_width_floor, "relief_halfwidth_cells",
                             py::cast(floor_defaults.relief_halfwidth_cells));
      set_default_if_missing(min_width_floor, "max_growth_factor",
                             py::cast(floor_defaults.max_growth_factor));
      set_default_if_missing(min_width_floor, "retrigger_cooldown_steps",
                             py::cast(floor_defaults.retrigger_cooldown_steps));
    }
  }
