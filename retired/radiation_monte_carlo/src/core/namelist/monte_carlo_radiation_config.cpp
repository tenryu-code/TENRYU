// Retired with the Monte Carlo radiation on 2026-09-29: the configuration front end of Radiation.mode "imc_ddmc", cut
// out of src/core/config.hpp, src/core/namelist/builder.cpp and src/core/namelist/freeze.cpp as they were at
// 5bc8f6ce3 (reference copy, not built). The builder now accepts these keys and ignores them (Radiation.imc keeps
// two_stage); enabled=True of Radiation.imc, .ddmc, .holo or .imc.difference is an error.

// ---- config.hpp: Config::RadiationConfig members

    struct CensusCombConfig {
      bool enabled = false;
      int max_particles = 1000000;
      int min_per_bin = 1;
      double trigger_ratio = 1.0;
      double target_fraction = 0.8;
      double mode_weight_imc = 1.0;
      double mode_weight_ddmc = 0.5;
      bool adaptive_trigger = true;
      double adaptive_util_start = 0.70;
      double adaptive_util_end = 0.95;
      double trigger_ratio_floor = 0.85;
      double trigger_hysteresis = 0.05;
      bool ess_floor_enabled = false;
      double ess_min_tier0 = 16.0;
      double ess_min_tier1 = 8.0;
      int max_split_factor = 4;
    };

    struct RadLiteMeshConfig {
      bool enabled = false;
      double sigma_ratio_max = 2.0;
      bool nlte_auto = false;
    };

    struct ImcConfig {
      struct DifferenceConfig {
        bool enabled = false;
        double W_max = 1.0;
        double tau0 = 3.0;
        double chi0 = 1.0;
        bool face_transport = true;
      };

      struct NetElectronSourceSmoothingConfig {
        bool enabled = false;
        double alpha = 0.2;
        double tau_threshold = 4.0;
        int passes = 1;
        double grad_Te_scale = 0.3;
        double grad_rho_scale = 0.5;
        bool gradient_adaptive = false;
      };

      struct ConservativeSmootherConfig {
        bool enabled = false;
        int passes = 10;
        double alpha = 0.5;
      };

      bool enabled = false;
      double alpha = 1.0;
      double f_max = 1.0;
      bool corrected_fleck = false;
      int particles_per_cell_group = 50;
      // Parsed for input compatibility; v1.0 transport always uses continuous
      // absorption (implicit capture effectively hardcoded on).
      bool implicit_capture = true;
      double cutoff_fraction = 0.0;
      bool inelastic_scatter = true;
      double weight_cutoff = 1e-10;
      double roulette_survival = 0.1;
      double weight_split = 1e2;
      int max_split = 8;
      bool linearized_planck = false;
      bool source_tilting = false;
      bool source_localization = false;
      double sloc_ema_beta = 0.4;
      double sloc_sigma_floor = 0.1;
      double sloc_sigma_cap = 0.5;
      double sloc_tau_ref = 1.0;
      double spectral_bias_eta = 0.0;
      bool opacity_predictor = false;
      bool two_stage = false;
      DifferenceConfig difference;
      NetElectronSourceSmoothingConfig net_e_source_smoothing;
      ConservativeSmootherConfig conservative_smoother;
      // -1 = disabled (backward compatible), >0 = total particle cap
      int particle_budget = -1;
      CensusCombConfig census_comb;
      RadLiteMeshConfig rad_lite_mesh;
    };

    struct DdmcConfig {
      bool enabled = false;
      bool implicit_diffusion = false;
      double tau_ddmc = 4.0;
      double tau_rw = 0.0;
      double omega_ddmc = 0.9;
      double tau_ddmc_off = -1.0;
      double omega_ddmc_off = -1.0;
      int mode_hold = 0;
      double rate_max = 1.0e30;
      std::string leak_stencil = "9_kershaw";
      // Parsed for compatibility; v1.0 transport implements asymptotic_diffusion_limit only.
      std::string interface_method = "asymptotic_diffusion_limit";
      bool emissivity_preserving = true;
      std::string interface_exit_distribution = "cosine";
      bool rz_face_r_weight = true;
      std::string face_opacity_temperature = "radiative_mean";
      bool m_matrix_check = true;
    };

    struct DiffusionConfig {
      bool enabled = false;
      double tau_on = 5.0;
      double tau_off = 3.0;
      double reduced_flux_on = 0.15;
      double reduced_flux_off = 0.25;
      int mode_hold = 0;
      double rate_max = 1.0e30;
      int mode_update_interval = 10;
      int min_diffusion_island_cells = 5;
      int imc_guard_cells = 1;
      int sts_max_stages = 0;
      double sts_damping = 0.05;
      double sts_subcycle_eta = 0.8;
      int interface_particles_per_face_group = 32;
      int exit_particles_per_cell_group = 32;
      bool lte_entry_initialization = false;
      double lte_entry_energy_fraction_cap = 0.01;
    };

    // EXPERIMENTAL: HOLO (High-Order Low-Order) radiation acceleration.
    // Not validated for production use. Use Radiation.imc.difference instead.
    // Enabling HOLO with difference formulation is not supported.
    struct HoloConfig {
      bool enabled = false;  // default OFF; must be explicitly enabled
      std::string region = "shell";
      std::string material_group = "shell";
      double coupling_tau = 5.0;
      int guard_cells = 3;
      int blend_cells = 3;
      int min_lo_cells = 20;
      double q_min = 0.0;
      double q_max = 1.0;
      double tau_on = 5.0;
      double tau_off = 3.0;
      double reduced_flux_on = 0.15;
      double reduced_flux_off = 0.25;
      int update_interval = 10;
      int hold_on = 0;
      int min_dwell_steps = 20;
      int min_island_cells = 5;
      int core_margin_cells = 3;
      std::string solver = "implicit_1d";  // "implicit_1d" or "quasidiffusion_1d"
      std::string closure = "diffusion";
      double closure_relax = 0.2;
      int closure_smooth_passes = 1;
      double closure_smooth_alpha = 0.5;
      double consistency_alpha = 1.0;
      std::string boundary_flux = "physical";
      bool p_rr_tally = true;
      bool sn_closure = true;
      int sn_n_angles = 8;
      bool sn_material_coupling = false;
      int residual_particles_per_cell_group = 4;
    };


// ---- config.hpp: Config::DiagnosticsConfig members

    struct McStats {
      bool enabled = true;
      bool particle_counts = true;
      bool weight_stats = true;
      bool cell_particle_density = false;
      bool ddmc_fraction = true;
    };
    struct FleckDiag {
      bool enabled = false;
      int every = 10;
      std::vector<int> cells;
      double r_min_cm = -1.0;
      double r_max_cm = -1.0;
    };


// ---- builder.cpp: value predicates

bool is_ddmc_leak_stencil(const std::string& value) {
  return value == "4" || value == "9_kershaw";
}

bool is_ddmc_interface_method(const std::string& value) {
  return value == "asymptotic_diffusion_limit" || value == "marshak" ||
         value == "cleveland_gentile";
}

bool is_ddmc_interface_exit_distribution(const std::string& value) {
  return value == "cosine" || value == "half_isotropic";
}

bool is_ddmc_face_opacity_temperature(const std::string& value) {
  return value == "radiative_mean";
}


// ---- builder.cpp: Builder::set_radiation, the Radiation.imc, .ddmc and .diffusion dicts

  if (has_key(kwargs, "imc")) {
    const py::handle imc_obj = kwargs["imc"];
    if (!py::isinstance<py::dict>(imc_obj)) {
      throw_value_type_error("Radiation.imc", "dict", imc_obj);
    }
    const py::dict imc = py::reinterpret_borrow<py::dict>(imc_obj);
    enforce_known_keys(imc, "Radiation.imc",
                       {"enabled",
                        "alpha", "f_max", "corrected_fleck", "particles_per_cell_group",
                        "implicit_capture",
                        "cutoff_fraction", "inelastic_scatter", "weight_cutoff",
                        "roulette_survival", "weight_split", "max_split",
                        "linearized_planck", "source_tilting", "source_localization",
                        "sloc_ema_beta", "sloc_sigma_floor", "sloc_sigma_cap",
                        "sloc_tau_ref",
                        "spectral_bias_eta",
                        "opacity_predictor", "two_stage",
                        "difference",
                        "net_e_source_smoothing",
                        "conservative_smoother",
                        "particle_budget",
                        "census_comb", "rad_lite_mesh"});
    if (has_key(imc, "enabled")) {
      radiation.imc.enabled = strict_bool(imc["enabled"], "Radiation.imc.enabled");
    }
    if (has_key(imc, "alpha")) {
      radiation.imc.alpha = numeric_as_double(imc["alpha"], "Radiation.imc.alpha");
      if (!(radiation.imc.alpha > 0.0)) {
        throw ValueError("Radiation.imc.alpha must be > 0");
      }
    }
    if (has_key(imc, "f_max")) {
      radiation.imc.f_max = numeric_as_double(imc["f_max"], "Radiation.imc.f_max");
    }
    if (has_key(imc, "corrected_fleck")) {
      radiation.imc.corrected_fleck =
          strict_bool(imc["corrected_fleck"], "Radiation.imc.corrected_fleck");
    }
    if (has_key(imc, "particles_per_cell_group")) {
      radiation.imc.particles_per_cell_group = strict_int32(
          imc["particles_per_cell_group"],
          "Radiation.imc.particles_per_cell_group");
    }
    if (has_key(imc, "implicit_capture")) {
      radiation.imc.implicit_capture =
          strict_bool(imc["implicit_capture"], "Radiation.imc.implicit_capture");
    }
    if (has_key(imc, "cutoff_fraction")) {
      radiation.imc.cutoff_fraction = numeric_as_double(
          imc["cutoff_fraction"], "Radiation.imc.cutoff_fraction");
    }
    if (has_key(imc, "inelastic_scatter")) {
      radiation.imc.inelastic_scatter =
          strict_bool(imc["inelastic_scatter"], "Radiation.imc.inelastic_scatter");
    }
    if (has_key(imc, "weight_cutoff")) {
      radiation.imc.weight_cutoff =
          numeric_as_double(imc["weight_cutoff"], "Radiation.imc.weight_cutoff");
    }
    if (has_key(imc, "roulette_survival")) {
      radiation.imc.roulette_survival = numeric_as_double(
          imc["roulette_survival"], "Radiation.imc.roulette_survival");
    }
    if (has_key(imc, "weight_split")) {
      radiation.imc.weight_split =
          numeric_as_double(imc["weight_split"], "Radiation.imc.weight_split");
    }
    if (has_key(imc, "max_split")) {
      radiation.imc.max_split =
          strict_int32(imc["max_split"], "Radiation.imc.max_split");
    }
    if (has_key(imc, "linearized_planck")) {
      radiation.imc.linearized_planck =
          strict_bool(imc["linearized_planck"], "Radiation.imc.linearized_planck");
    }
    if (has_key(imc, "source_tilting")) {
      radiation.imc.source_tilting =
          strict_bool(imc["source_tilting"], "Radiation.imc.source_tilting");
    }
    if (has_key(imc, "source_localization")) {
      radiation.imc.source_localization = strict_bool(
          imc["source_localization"], "Radiation.imc.source_localization");
    }
    if (has_key(imc, "sloc_ema_beta")) {
      radiation.imc.sloc_ema_beta = numeric_as_double(
          imc["sloc_ema_beta"], "Radiation.imc.sloc_ema_beta");
    }
    if (has_key(imc, "sloc_sigma_floor")) {
      radiation.imc.sloc_sigma_floor = numeric_as_double(
          imc["sloc_sigma_floor"], "Radiation.imc.sloc_sigma_floor");
    }
    if (has_key(imc, "sloc_sigma_cap")) {
      radiation.imc.sloc_sigma_cap = numeric_as_double(
          imc["sloc_sigma_cap"], "Radiation.imc.sloc_sigma_cap");
    }
    if (has_key(imc, "sloc_tau_ref")) {
      radiation.imc.sloc_tau_ref = numeric_as_double(
          imc["sloc_tau_ref"], "Radiation.imc.sloc_tau_ref");
    }
    if (has_key(imc, "spectral_bias_eta")) {
      radiation.imc.spectral_bias_eta = numeric_as_double(
          imc["spectral_bias_eta"], "Radiation.imc.spectral_bias_eta");
    }
    if (has_key(imc, "opacity_predictor")) {
      radiation.imc.opacity_predictor =
          strict_bool(imc["opacity_predictor"], "Radiation.imc.opacity_predictor");
    }
    if (has_key(imc, "two_stage")) {
      radiation.imc.two_stage =
          strict_bool(imc["two_stage"], "Radiation.imc.two_stage");
    }
    if (has_key(imc, "difference")) {
      const py::handle difference_obj = imc["difference"];
      if (!py::isinstance<py::dict>(difference_obj)) {
        throw_value_type_error("Radiation.imc.difference", "dict", difference_obj);
      }
      const py::dict difference = py::reinterpret_borrow<py::dict>(difference_obj);
      enforce_known_keys(difference, "Radiation.imc.difference",
                         {"enabled", "W_max", "tau0", "chi0", "face_transport"});
      if (has_key(difference, "enabled")) {
        radiation.imc.difference.enabled =
            strict_bool(difference["enabled"], "Radiation.imc.difference.enabled");
      }
      if (has_key(difference, "W_max")) {
        radiation.imc.difference.W_max =
            numeric_as_double(difference["W_max"], "Radiation.imc.difference.W_max");
      }
      if (has_key(difference, "tau0")) {
        radiation.imc.difference.tau0 =
            numeric_as_double(difference["tau0"], "Radiation.imc.difference.tau0");
      }
      if (has_key(difference, "chi0")) {
        radiation.imc.difference.chi0 =
            numeric_as_double(difference["chi0"], "Radiation.imc.difference.chi0");
      }
      if (has_key(difference, "face_transport")) {
        radiation.imc.difference.face_transport = strict_bool(
            difference["face_transport"], "Radiation.imc.difference.face_transport");
      }
    }
    if (has_key(imc, "net_e_source_smoothing")) {
      const py::handle smoothing_obj = imc["net_e_source_smoothing"];
      if (!py::isinstance<py::dict>(smoothing_obj)) {
        throw_value_type_error("Radiation.imc.net_e_source_smoothing", "dict",
                               smoothing_obj);
      }
      const py::dict smoothing = py::reinterpret_borrow<py::dict>(smoothing_obj);
      enforce_known_keys(smoothing, "Radiation.imc.net_e_source_smoothing",
                         {"enabled", "alpha", "tau_threshold", "passes",
                          "grad_Te_scale", "grad_rho_scale", "gradient_adaptive"});
      if (has_key(smoothing, "enabled")) {
        radiation.imc.net_e_source_smoothing.enabled =
            strict_bool(smoothing["enabled"],
                        "Radiation.imc.net_e_source_smoothing.enabled");
      }
      if (has_key(smoothing, "alpha")) {
        radiation.imc.net_e_source_smoothing.alpha =
            numeric_as_double(smoothing["alpha"],
                              "Radiation.imc.net_e_source_smoothing.alpha");
      }
      if (has_key(smoothing, "tau_threshold")) {
        radiation.imc.net_e_source_smoothing.tau_threshold =
            numeric_as_double(
                smoothing["tau_threshold"],
                "Radiation.imc.net_e_source_smoothing.tau_threshold");
      }
      if (has_key(smoothing, "passes")) {
        radiation.imc.net_e_source_smoothing.passes =
            strict_int32(smoothing["passes"],
                         "Radiation.imc.net_e_source_smoothing.passes");
      }
      if (has_key(smoothing, "grad_Te_scale")) {
        radiation.imc.net_e_source_smoothing.grad_Te_scale =
            numeric_as_double(
                smoothing["grad_Te_scale"],
                "Radiation.imc.net_e_source_smoothing.grad_Te_scale");
      }
      if (has_key(smoothing, "grad_rho_scale")) {
        radiation.imc.net_e_source_smoothing.grad_rho_scale =
            numeric_as_double(
                smoothing["grad_rho_scale"],
                "Radiation.imc.net_e_source_smoothing.grad_rho_scale");
      }
      if (has_key(smoothing, "gradient_adaptive")) {
        radiation.imc.net_e_source_smoothing.gradient_adaptive =
            strict_bool(
                smoothing["gradient_adaptive"],
                "Radiation.imc.net_e_source_smoothing.gradient_adaptive");
      }
    }
    if (has_key(imc, "conservative_smoother")) {
      const py::handle smoother_obj = imc["conservative_smoother"];
      if (!py::isinstance<py::dict>(smoother_obj)) {
        throw_value_type_error("Radiation.imc.conservative_smoother", "dict",
                               smoother_obj);
      }
      const py::dict smoother = py::reinterpret_borrow<py::dict>(smoother_obj);
      enforce_known_keys(smoother, "Radiation.imc.conservative_smoother",
                         {"enabled", "passes", "alpha"});
      if (has_key(smoother, "enabled")) {
        radiation.imc.conservative_smoother.enabled =
            strict_bool(smoother["enabled"],
                        "Radiation.imc.conservative_smoother.enabled");
      }
      if (has_key(smoother, "passes")) {
        radiation.imc.conservative_smoother.passes =
            strict_int32(smoother["passes"],
                         "Radiation.imc.conservative_smoother.passes");
      }
      if (has_key(smoother, "alpha")) {
        radiation.imc.conservative_smoother.alpha =
            numeric_as_double(smoother["alpha"],
                              "Radiation.imc.conservative_smoother.alpha");
      }
    }
    if (has_key(imc, "particle_budget")) {
      radiation.imc.particle_budget = strict_int32(
          imc["particle_budget"], "Radiation.imc.particle_budget");
    }
    if (has_key(imc, "census_comb")) {
      const py::handle census_comb_obj = imc["census_comb"];
      if (!py::isinstance<py::dict>(census_comb_obj)) {
        throw_value_type_error("Radiation.imc.census_comb", "dict", census_comb_obj);
      }
      const py::dict census_comb = py::reinterpret_borrow<py::dict>(census_comb_obj);
      enforce_known_keys(census_comb, "Radiation.imc.census_comb",
                         {"enabled", "max_particles", "min_per_bin", "trigger_ratio",
                          "target_fraction", "mode_weight_imc", "mode_weight_ddmc",
                          "adaptive_trigger", "adaptive_util_start", "adaptive_util_end",
                          "trigger_ratio_floor", "trigger_hysteresis",
                          "ess_floor_enabled", "ess_min_tier0",
                          "ess_min_tier1", "max_split_factor"});
      if (has_key(census_comb, "enabled")) {
        radiation.imc.census_comb.enabled =
            strict_bool(census_comb["enabled"], "Radiation.imc.census_comb.enabled");
      }
      if (has_key(census_comb, "max_particles")) {
        radiation.imc.census_comb.max_particles = strict_int32(
            census_comb["max_particles"], "Radiation.imc.census_comb.max_particles");
      }
      if (has_key(census_comb, "min_per_bin")) {
        radiation.imc.census_comb.min_per_bin = strict_int32(
            census_comb["min_per_bin"], "Radiation.imc.census_comb.min_per_bin");
      }
      if (has_key(census_comb, "trigger_ratio")) {
        radiation.imc.census_comb.trigger_ratio = numeric_as_double(
            census_comb["trigger_ratio"], "Radiation.imc.census_comb.trigger_ratio");
      }
      if (has_key(census_comb, "target_fraction")) {
        radiation.imc.census_comb.target_fraction = numeric_as_double(
            census_comb["target_fraction"], "Radiation.imc.census_comb.target_fraction");
      }
      if (has_key(census_comb, "mode_weight_imc")) {
        radiation.imc.census_comb.mode_weight_imc = numeric_as_double(
            census_comb["mode_weight_imc"], "Radiation.imc.census_comb.mode_weight_imc");
      }
      if (has_key(census_comb, "mode_weight_ddmc")) {
        radiation.imc.census_comb.mode_weight_ddmc = numeric_as_double(
            census_comb["mode_weight_ddmc"], "Radiation.imc.census_comb.mode_weight_ddmc");
      }
      if (has_key(census_comb, "adaptive_trigger")) {
        radiation.imc.census_comb.adaptive_trigger = strict_bool(
            census_comb["adaptive_trigger"], "Radiation.imc.census_comb.adaptive_trigger");
      }
      if (has_key(census_comb, "adaptive_util_start")) {
        radiation.imc.census_comb.adaptive_util_start = numeric_as_double(
            census_comb["adaptive_util_start"], "Radiation.imc.census_comb.adaptive_util_start");
      }
      if (has_key(census_comb, "adaptive_util_end")) {
        radiation.imc.census_comb.adaptive_util_end = numeric_as_double(
            census_comb["adaptive_util_end"], "Radiation.imc.census_comb.adaptive_util_end");
      }
      if (has_key(census_comb, "trigger_ratio_floor")) {
        radiation.imc.census_comb.trigger_ratio_floor = numeric_as_double(
            census_comb["trigger_ratio_floor"], "Radiation.imc.census_comb.trigger_ratio_floor");
      }
      if (has_key(census_comb, "trigger_hysteresis")) {
        radiation.imc.census_comb.trigger_hysteresis = numeric_as_double(
            census_comb["trigger_hysteresis"], "Radiation.imc.census_comb.trigger_hysteresis");
      }
      if (has_key(census_comb, "ess_floor_enabled")) {
        radiation.imc.census_comb.ess_floor_enabled = strict_bool(
            census_comb["ess_floor_enabled"], "Radiation.imc.census_comb.ess_floor_enabled");
      }
      if (has_key(census_comb, "ess_min_tier0")) {
        radiation.imc.census_comb.ess_min_tier0 = numeric_as_double(
            census_comb["ess_min_tier0"], "Radiation.imc.census_comb.ess_min_tier0");
      }
      if (has_key(census_comb, "ess_min_tier1")) {
        radiation.imc.census_comb.ess_min_tier1 = numeric_as_double(
            census_comb["ess_min_tier1"], "Radiation.imc.census_comb.ess_min_tier1");
      }
      if (has_key(census_comb, "max_split_factor")) {
        radiation.imc.census_comb.max_split_factor = strict_int32(
            census_comb["max_split_factor"], "Radiation.imc.census_comb.max_split_factor");
      }
    }
    if (has_key(imc, "rad_lite_mesh")) {
      const py::handle rlm_obj = imc["rad_lite_mesh"];
      if (!py::isinstance<py::dict>(rlm_obj)) {
        throw_value_type_error("Radiation.imc.rad_lite_mesh", "dict", rlm_obj);
      }
      const py::dict rlm = py::reinterpret_borrow<py::dict>(rlm_obj);
      enforce_known_keys(rlm, "Radiation.imc.rad_lite_mesh",
                         {"enabled", "sigma_ratio_max", "nlte_auto"});
      if (has_key(rlm, "enabled")) {
        radiation.imc.rad_lite_mesh.enabled =
            strict_bool(rlm["enabled"], "Radiation.imc.rad_lite_mesh.enabled");
      }
      if (has_key(rlm, "sigma_ratio_max")) {
        radiation.imc.rad_lite_mesh.sigma_ratio_max = numeric_as_double(
            rlm["sigma_ratio_max"], "Radiation.imc.rad_lite_mesh.sigma_ratio_max");
        if (!(radiation.imc.rad_lite_mesh.sigma_ratio_max > 1.0)) {
          throw ValueError("Radiation.imc.rad_lite_mesh.sigma_ratio_max must be > 1.0");
        }
      }
      if (has_key(rlm, "nlte_auto")) {
        radiation.imc.rad_lite_mesh.nlte_auto =
            strict_bool(rlm["nlte_auto"], "Radiation.imc.rad_lite_mesh.nlte_auto");
      }
    }
  }

  if (has_key(kwargs, "ddmc")) {
    const py::handle ddmc_obj = kwargs["ddmc"];
    if (!py::isinstance<py::dict>(ddmc_obj)) {
      throw_value_type_error("Radiation.ddmc", "dict", ddmc_obj);
    }
    const py::dict ddmc = py::reinterpret_borrow<py::dict>(ddmc_obj);
    enforce_known_keys(ddmc, "Radiation.ddmc",
                       {"enabled", "implicit_diffusion", "tau_ddmc", "tau_rw", "omega_ddmc", "leak_stencil",
                        "tau_ddmc_off", "omega_ddmc_off", "mode_hold", "rate_max",
                        "interface_method", "emissivity_preserving",
                        "interface_exit_distribution", "rz_face_r_weight",
                        "face_opacity_temperature", "m_matrix_check"});
    if (has_key(ddmc, "enabled")) {
      radiation.ddmc.enabled = strict_bool(ddmc["enabled"], "Radiation.ddmc.enabled");
    }
    if (has_key(ddmc, "implicit_diffusion")) {
      radiation.ddmc.implicit_diffusion = strict_bool(
          ddmc["implicit_diffusion"], "Radiation.ddmc.implicit_diffusion");
    }
    if (has_key(ddmc, "tau_ddmc")) {
      radiation.ddmc.tau_ddmc =
          numeric_as_double(ddmc["tau_ddmc"], "Radiation.ddmc.tau_ddmc");
    }
    if (has_key(ddmc, "tau_rw")) {
      radiation.ddmc.tau_rw =
          numeric_as_double(ddmc["tau_rw"], "Radiation.ddmc.tau_rw");
    }
    if (has_key(ddmc, "omega_ddmc")) {
      radiation.ddmc.omega_ddmc =
          numeric_as_double(ddmc["omega_ddmc"], "Radiation.ddmc.omega_ddmc");
    }
    if (has_key(ddmc, "tau_ddmc_off")) {
      radiation.ddmc.tau_ddmc_off =
          numeric_as_double(ddmc["tau_ddmc_off"], "Radiation.ddmc.tau_ddmc_off");
    }
    if (has_key(ddmc, "omega_ddmc_off")) {
      radiation.ddmc.omega_ddmc_off = numeric_as_double(
          ddmc["omega_ddmc_off"], "Radiation.ddmc.omega_ddmc_off");
    }
    if (has_key(ddmc, "mode_hold")) {
      radiation.ddmc.mode_hold = strict_int32(ddmc["mode_hold"], "Radiation.ddmc.mode_hold");
    }
    if (has_key(ddmc, "rate_max")) {
      radiation.ddmc.rate_max =
          numeric_as_double(ddmc["rate_max"], "Radiation.ddmc.rate_max");
    }
    if (has_key(ddmc, "leak_stencil")) {
      radiation.ddmc.leak_stencil =
          strict_string(ddmc["leak_stencil"], "Radiation.ddmc.leak_stencil");
    }
    if (has_key(ddmc, "interface_method")) {
      radiation.ddmc.interface_method =
          strict_string(ddmc["interface_method"],
                        "Radiation.ddmc.interface_method");
    }
    if (has_key(ddmc, "emissivity_preserving")) {
      radiation.ddmc.emissivity_preserving = strict_bool(
          ddmc["emissivity_preserving"], "Radiation.ddmc.emissivity_preserving");
    }
    if (has_key(ddmc, "interface_exit_distribution")) {
      radiation.ddmc.interface_exit_distribution =
          strict_string(ddmc["interface_exit_distribution"],
                        "Radiation.ddmc.interface_exit_distribution");
    }
    if (has_key(ddmc, "rz_face_r_weight")) {
      radiation.ddmc.rz_face_r_weight =
          strict_bool(ddmc["rz_face_r_weight"], "Radiation.ddmc.rz_face_r_weight");
    }
    if (has_key(ddmc, "face_opacity_temperature")) {
      radiation.ddmc.face_opacity_temperature =
          strict_string(ddmc["face_opacity_temperature"],
                        "Radiation.ddmc.face_opacity_temperature");
    }
    if (has_key(ddmc, "m_matrix_check")) {
      radiation.ddmc.m_matrix_check = strict_bool(ddmc["m_matrix_check"],
                                                  "Radiation.ddmc.m_matrix_check");
    }
  }

  if (has_key(kwargs, "diffusion")) {
    const py::handle diffusion_obj = kwargs["diffusion"];
    if (!py::isinstance<py::dict>(diffusion_obj)) {
      throw_value_type_error("Radiation.diffusion", "dict", diffusion_obj);
    }
    const py::dict diffusion = py::reinterpret_borrow<py::dict>(diffusion_obj);
    enforce_known_keys(diffusion, "Radiation.diffusion",
                       {"enabled", "tau_on", "tau_off",
                        "reduced_flux_on", "reduced_flux_off",
                        "mode_hold", "rate_max",
                        "mode_update_interval", "min_diffusion_island_cells",
                        "imc_guard_cells",
                        "sts_max_stages", "sts_damping", "sts_subcycle_eta",
                        "interface_particles_per_face_group",
                        "exit_particles_per_cell_group",
                        "lte_entry_initialization",
                        "lte_entry_energy_fraction_cap"});
    if (has_key(diffusion, "enabled")) {
      radiation.diffusion.enabled =
          strict_bool(diffusion["enabled"], "Radiation.diffusion.enabled");
    }
    if (has_key(diffusion, "tau_on")) {
      radiation.diffusion.tau_on =
          numeric_as_double(diffusion["tau_on"], "Radiation.diffusion.tau_on");
    }
    if (has_key(diffusion, "tau_off")) {
      radiation.diffusion.tau_off =
          numeric_as_double(diffusion["tau_off"], "Radiation.diffusion.tau_off");
    }
    if (has_key(diffusion, "reduced_flux_on")) {
      radiation.diffusion.reduced_flux_on = numeric_as_double(
          diffusion["reduced_flux_on"], "Radiation.diffusion.reduced_flux_on");
    }
    if (has_key(diffusion, "reduced_flux_off")) {
      radiation.diffusion.reduced_flux_off = numeric_as_double(
          diffusion["reduced_flux_off"], "Radiation.diffusion.reduced_flux_off");
    }
    if (has_key(diffusion, "mode_hold")) {
      radiation.diffusion.mode_hold =
          strict_int32(diffusion["mode_hold"], "Radiation.diffusion.mode_hold");
    }
    if (has_key(diffusion, "rate_max")) {
      radiation.diffusion.rate_max =
          numeric_as_double(diffusion["rate_max"], "Radiation.diffusion.rate_max");
    }
    if (has_key(diffusion, "mode_update_interval")) {
      radiation.diffusion.mode_update_interval = strict_int32(
          diffusion["mode_update_interval"],
          "Radiation.diffusion.mode_update_interval");
    }
    if (has_key(diffusion, "min_diffusion_island_cells")) {
      radiation.diffusion.min_diffusion_island_cells = strict_int32(
          diffusion["min_diffusion_island_cells"],
          "Radiation.diffusion.min_diffusion_island_cells");
    }
    if (has_key(diffusion, "imc_guard_cells")) {
      radiation.diffusion.imc_guard_cells = strict_int32(
          diffusion["imc_guard_cells"], "Radiation.diffusion.imc_guard_cells");
    }
    if (has_key(diffusion, "sts_max_stages")) {
      radiation.diffusion.sts_max_stages = strict_int32(
          diffusion["sts_max_stages"], "Radiation.diffusion.sts_max_stages");
    }
    if (has_key(diffusion, "sts_damping")) {
      radiation.diffusion.sts_damping = numeric_as_double(
          diffusion["sts_damping"], "Radiation.diffusion.sts_damping");
    }
    if (has_key(diffusion, "sts_subcycle_eta")) {
      radiation.diffusion.sts_subcycle_eta = numeric_as_double(
          diffusion["sts_subcycle_eta"], "Radiation.diffusion.sts_subcycle_eta");
    }
    if (has_key(diffusion, "interface_particles_per_face_group")) {
      radiation.diffusion.interface_particles_per_face_group = strict_int32(
          diffusion["interface_particles_per_face_group"],
          "Radiation.diffusion.interface_particles_per_face_group");
    }
    if (has_key(diffusion, "exit_particles_per_cell_group")) {
      radiation.diffusion.exit_particles_per_cell_group = strict_int32(
          diffusion["exit_particles_per_cell_group"],
          "Radiation.diffusion.exit_particles_per_cell_group");
    }
    if (has_key(diffusion, "lte_entry_initialization")) {
      radiation.diffusion.lte_entry_initialization = strict_bool(
          diffusion["lte_entry_initialization"],
          "Radiation.diffusion.lte_entry_initialization");
    }
    if (has_key(diffusion, "lte_entry_energy_fraction_cap")) {
      radiation.diffusion.lte_entry_energy_fraction_cap = numeric_as_double(
          diffusion["lte_entry_energy_fraction_cap"],
          "Radiation.diffusion.lte_entry_energy_fraction_cap");
    }
  }


// ---- builder.cpp: Builder::set_radiation, the Radiation.holo dict

  if (has_key(kwargs, "holo")) {
    const py::handle holo_obj = kwargs["holo"];
    if (!py::isinstance<py::dict>(holo_obj)) {
      throw_value_type_error("Radiation.holo", "dict", holo_obj);
    }
    const py::dict holo = py::reinterpret_borrow<py::dict>(holo_obj);
    const bool holo_has_tau_on = has_key(holo, "tau_on");
    const bool holo_has_tau_off = has_key(holo, "tau_off");
    enforce_known_keys(holo, "Radiation.holo",
                       {"enabled", "region", "material_group", "q_min", "q_max",
                        "coupling_tau", "guard_cells", "blend_cells",
                        "min_lo_cells",
                        "tau_on", "tau_off",
                        "reduced_flux_on", "reduced_flux_off",
                        "update_interval", "hold_on", "min_dwell_steps",
                        "min_island_cells", "core_margin_cells",
                        "solver", "closure", "closure_relax",
                        "closure_smooth_passes", "closure_smooth_alpha",
                        "consistency_alpha", "gamma_alpha", "boundary_flux", "p_rr_tally",
                        "sn_closure", "sn_n_angles",
                        "sn_material_coupling",
                        "residual_particles_per_cell_group"});
    if (has_key(holo, "enabled")) {
      radiation.holo.enabled = strict_bool(holo["enabled"], "Radiation.holo.enabled");
    }
    if (has_key(holo, "region")) {
      radiation.holo.region = strict_string(holo["region"], "Radiation.holo.region");
    }
    if (has_key(holo, "material_group")) {
      radiation.holo.material_group =
          strict_string(holo["material_group"], "Radiation.holo.material_group");
    }
    if (has_key(holo, "coupling_tau")) {
      radiation.holo.coupling_tau =
          numeric_as_double(holo["coupling_tau"], "Radiation.holo.coupling_tau");
    }
    if (has_key(holo, "guard_cells")) {
      radiation.holo.guard_cells =
          strict_int32(holo["guard_cells"], "Radiation.holo.guard_cells");
    }
    if (has_key(holo, "blend_cells")) {
      radiation.holo.blend_cells =
          strict_int32(holo["blend_cells"], "Radiation.holo.blend_cells");
    }
    if (has_key(holo, "min_lo_cells")) {
      radiation.holo.min_lo_cells =
          strict_int32(holo["min_lo_cells"], "Radiation.holo.min_lo_cells");
    }
    if (has_key(holo, "q_min")) {
      radiation.holo.q_min =
          numeric_as_double(holo["q_min"], "Radiation.holo.q_min");
    }
    if (has_key(holo, "q_max")) {
      radiation.holo.q_max =
          numeric_as_double(holo["q_max"], "Radiation.holo.q_max");
    }
    if (holo_has_tau_on) {
      radiation.holo.tau_on =
          numeric_as_double(holo["tau_on"], "Radiation.holo.tau_on");
    }
    if (holo_has_tau_off) {
      radiation.holo.tau_off =
          numeric_as_double(holo["tau_off"], "Radiation.holo.tau_off");
    }
    if (!holo_has_tau_on) {
      radiation.holo.tau_on = radiation.holo.coupling_tau;
    }
    if (has_key(holo, "reduced_flux_on")) {
      radiation.holo.reduced_flux_on = numeric_as_double(
          holo["reduced_flux_on"], "Radiation.holo.reduced_flux_on");
    }
    if (has_key(holo, "reduced_flux_off")) {
      radiation.holo.reduced_flux_off = numeric_as_double(
          holo["reduced_flux_off"], "Radiation.holo.reduced_flux_off");
    }
    if (has_key(holo, "update_interval")) {
      radiation.holo.update_interval = strict_int32(
          holo["update_interval"], "Radiation.holo.update_interval");
    }
    if (has_key(holo, "hold_on")) {
      radiation.holo.hold_on = strict_int32(
          holo["hold_on"], "Radiation.holo.hold_on");
    }
    if (has_key(holo, "min_dwell_steps")) {
      radiation.holo.min_dwell_steps = strict_int32(
          holo["min_dwell_steps"], "Radiation.holo.min_dwell_steps");
    }
    if (has_key(holo, "min_island_cells")) {
      radiation.holo.min_island_cells = strict_int32(
          holo["min_island_cells"], "Radiation.holo.min_island_cells");
    }
    if (has_key(holo, "core_margin_cells")) {
      radiation.holo.core_margin_cells = strict_int32(
          holo["core_margin_cells"], "Radiation.holo.core_margin_cells");
    }
    if (has_key(holo, "solver")) {
      radiation.holo.solver = strict_string(holo["solver"], "Radiation.holo.solver");
    }
    if (has_key(holo, "closure")) {
      radiation.holo.closure = strict_string(holo["closure"], "Radiation.holo.closure");
    }
    if (has_key(holo, "closure_relax")) {
      radiation.holo.closure_relax =
          numeric_as_double(holo["closure_relax"], "Radiation.holo.closure_relax");
    }
    if (has_key(holo, "closure_smooth_passes")) {
      radiation.holo.closure_smooth_passes = strict_int32(
          holo["closure_smooth_passes"], "Radiation.holo.closure_smooth_passes");
    }
    if (has_key(holo, "closure_smooth_alpha")) {
      radiation.holo.closure_smooth_alpha = numeric_as_double(
          holo["closure_smooth_alpha"], "Radiation.holo.closure_smooth_alpha");
    }
    if (has_key(holo, "gamma_alpha")) {
      radiation.holo.consistency_alpha =
          numeric_as_double(holo["gamma_alpha"], "Radiation.holo.gamma_alpha");
    }
    if (has_key(holo, "consistency_alpha")) {
      radiation.holo.consistency_alpha = numeric_as_double(
          holo["consistency_alpha"], "Radiation.holo.consistency_alpha");
    }
    if (has_key(holo, "boundary_flux")) {
      radiation.holo.boundary_flux =
          strict_string(holo["boundary_flux"], "Radiation.holo.boundary_flux");
    }
    if (has_key(holo, "p_rr_tally")) {
      radiation.holo.p_rr_tally =
          strict_bool(holo["p_rr_tally"], "Radiation.holo.p_rr_tally");
    }
    if (has_key(holo, "sn_closure")) {
      radiation.holo.sn_closure =
          strict_bool(holo["sn_closure"], "Radiation.holo.sn_closure");
    }
    if (has_key(holo, "sn_n_angles")) {
      radiation.holo.sn_n_angles =
          strict_int32(holo["sn_n_angles"], "Radiation.holo.sn_n_angles");
    }
    if (has_key(holo, "sn_material_coupling")) {
      radiation.holo.sn_material_coupling = strict_bool(
          holo["sn_material_coupling"], "Radiation.holo.sn_material_coupling");
    }
    if (has_key(holo, "residual_particles_per_cell_group")) {
      radiation.holo.residual_particles_per_cell_group = strict_int32(
          holo["residual_particles_per_cell_group"],
          "Radiation.holo.residual_particles_per_cell_group");
    }
  }


// ---- builder.cpp: Builder::set_diagnostics, Diagnostics.mc_stats and Diagnostics.fleck_diag

  if (has_key(kwargs, "mc_stats")) {
    const py::handle mc_obj = kwargs["mc_stats"];
    if (!py::isinstance<py::dict>(mc_obj)) {
      throw_value_type_error("Diagnostics.mc_stats", "dict", mc_obj);
    }
    const py::dict mc = py::reinterpret_borrow<py::dict>(mc_obj);
    enforce_known_keys(mc, "Diagnostics.mc_stats",
                       {"enabled", "particle_counts", "weight_stats",
                        "cell_particle_density", "ddmc_fraction"});
    if (has_key(mc, "enabled")) {
      diagnostics.mc_stats.enabled =
          strict_bool(mc["enabled"], "Diagnostics.mc_stats.enabled");
    }
    if (has_key(mc, "particle_counts")) {
      diagnostics.mc_stats.particle_counts =
          strict_bool(mc["particle_counts"], "Diagnostics.mc_stats.particle_counts");
    }
    if (has_key(mc, "weight_stats")) {
      diagnostics.mc_stats.weight_stats =
          strict_bool(mc["weight_stats"], "Diagnostics.mc_stats.weight_stats");
    }
    if (has_key(mc, "cell_particle_density")) {
      diagnostics.mc_stats.cell_particle_density = strict_bool(
          mc["cell_particle_density"], "Diagnostics.mc_stats.cell_particle_density");
    }
    if (has_key(mc, "ddmc_fraction")) {
      diagnostics.mc_stats.ddmc_fraction =
          strict_bool(mc["ddmc_fraction"], "Diagnostics.mc_stats.ddmc_fraction");
    }
  }
  if (has_key(kwargs, "fleck_diag")) {
    const py::handle fleck_obj = kwargs["fleck_diag"];
    if (!py::isinstance<py::dict>(fleck_obj)) {
      throw_value_type_error("Diagnostics.fleck_diag", "dict", fleck_obj);
    }
    const py::dict fleck = py::reinterpret_borrow<py::dict>(fleck_obj);
    enforce_known_keys(fleck, "Diagnostics.fleck_diag",
                       {"enabled", "every", "cells", "r_min_cm", "r_max_cm"});
    if (has_key(fleck, "enabled")) {
      diagnostics.fleck_diag.enabled =
          strict_bool(fleck["enabled"], "Diagnostics.fleck_diag.enabled");
    }
    if (has_key(fleck, "every")) {
      diagnostics.fleck_diag.every =
          strict_int32(fleck["every"], "Diagnostics.fleck_diag.every");
      ensure_int_ge(diagnostics.fleck_diag.every, 1, "Diagnostics.fleck_diag.every");
    }
    if (has_key(fleck, "cells")) {
      diagnostics.fleck_diag.cells =
          strict_int_vector(fleck["cells"], "Diagnostics.fleck_diag.cells");
      for (const int cell : diagnostics.fleck_diag.cells) {
        if (cell < 0) {
          throw ValueError("Diagnostics.fleck_diag.cells entries must be >= 0");
        }
      }
    }
    if (has_key(fleck, "r_min_cm")) {
      diagnostics.fleck_diag.r_min_cm = numeric_as_double(
          fleck["r_min_cm"], "Diagnostics.fleck_diag.r_min_cm");
    }
    if (has_key(fleck, "r_max_cm")) {
      diagnostics.fleck_diag.r_max_cm = numeric_as_double(
          fleck["r_max_cm"], "Diagnostics.fleck_diag.r_max_cm");
    }

    const bool has_r_min = diagnostics.fleck_diag.r_min_cm >= 0.0;
    const bool has_r_max = diagnostics.fleck_diag.r_max_cm >= 0.0;
    if (has_r_min != has_r_max) {
      throw ConfigError(
          "Diagnostics.fleck_diag requires both r_min_cm and r_max_cm when using radius selection");
    }
    if (has_r_min && diagnostics.fleck_diag.r_max_cm < diagnostics.fleck_diag.r_min_cm) {
      throw ValueError("Diagnostics.fleck_diag.r_max_cm must be >= r_min_cm");
    }
  }

// ---- builder.cpp: value checks of the build step

  if (!(radiation.imc.alpha > 0.0)) {
    throw ValueError("Radiation.imc.alpha must be > 0");
  }
  if (radiation.imc.particles_per_cell_group < 1) {
    throw ConfigError("Radiation.imc.particles_per_cell_group must be >= 1");
  }
  if (radiation.imc.particle_budget > 0 &&
      radiation.imc.particle_budget < radiation.imc.particles_per_cell_group) {
    throw ConfigError(
        "Radiation.imc.particle_budget must be >= particles_per_cell_group or -1 (disabled)");
  }
  if (!(radiation.imc.difference.W_max >= 0.0 &&
        radiation.imc.difference.W_max <= 1.0)) {
    throw ValueError("Radiation.imc.difference.W_max must be in [0, 1]");
  }
  if (!(radiation.imc.difference.tau0 > 0.0)) {
    throw ValueError("Radiation.imc.difference.tau0 must be > 0");
  }
  if (!(radiation.imc.difference.chi0 > 0.0)) {
    throw ValueError("Radiation.imc.difference.chi0 must be > 0");
  }
  if (radiation.imc.difference.enabled) {
    if (main.dimension == "2D_RZ" && radiation.imc.difference.face_transport) {
      throw ConfigError(
          "Radiation.imc.difference.face_transport=True is not yet supported for "
          "Main.dimension=\"2D_RZ\". Use face_transport=False.");
    }
    if (main.dimension != "1D_SPH" && main.dimension != "2D_RZ") {
      throw ConfigError(
          "Radiation.imc.difference.enabled currently requires "
          "Main.dimension=\"1D_SPH\" or \"2D_RZ\"");
    }
  }
  if (!(radiation.imc.spectral_bias_eta >= 0.0 &&
        radiation.imc.spectral_bias_eta <= 1.0)) {
    throw ValueError("Radiation.imc.spectral_bias_eta must be in [0, 1]");
  }
  if (!(radiation.imc.sloc_ema_beta >= 0.0 &&
        radiation.imc.sloc_ema_beta <= 1.0)) {
    throw ValueError("Radiation.imc.sloc_ema_beta must be in [0, 1]");
  }
  if (!(radiation.imc.sloc_sigma_floor > 0.0)) {
    throw ValueError("Radiation.imc.sloc_sigma_floor must be > 0");
  }
  if (!(radiation.imc.sloc_sigma_cap > 0.0)) {
    throw ValueError("Radiation.imc.sloc_sigma_cap must be > 0");
  }
  if (radiation.imc.sloc_sigma_floor > radiation.imc.sloc_sigma_cap) {
    throw ValueError(
        "Radiation.imc.sloc_sigma_floor must be <= Radiation.imc.sloc_sigma_cap");
  }
  if (!(radiation.imc.sloc_tau_ref > 0.0)) {
    throw ValueError("Radiation.imc.sloc_tau_ref must be > 0");
  }
  const double net_e_source_smoothing_alpha_max =
      (main.dimension == "2D_RZ" &&
       radiation.imc.net_e_source_smoothing.enabled)
          ? 0.125
          : 0.25;
  if (!(radiation.imc.net_e_source_smoothing.alpha >= 0.0 &&
        radiation.imc.net_e_source_smoothing.alpha <=
            net_e_source_smoothing_alpha_max)) {
    throw ValueError(
        "Radiation.imc.net_e_source_smoothing.alpha must be in [0, " +
        std::to_string(net_e_source_smoothing_alpha_max) + "]");
  }
  if (!(radiation.imc.net_e_source_smoothing.tau_threshold > 0.0)) {
    throw ValueError(
        "Radiation.imc.net_e_source_smoothing.tau_threshold must be > 0");
  }
  if (radiation.imc.net_e_source_smoothing.passes < 0) {
    throw ValueError("Radiation.imc.net_e_source_smoothing.passes must be >= 0");
  }
  if (!(radiation.imc.net_e_source_smoothing.grad_Te_scale > 0.0)) {
    throw ValueError(
        "Radiation.imc.net_e_source_smoothing.grad_Te_scale must be > 0");
  }
  if (!(radiation.imc.net_e_source_smoothing.grad_rho_scale > 0.0)) {
    throw ValueError(
        "Radiation.imc.net_e_source_smoothing.grad_rho_scale must be > 0");
  }
  if (radiation.imc.conservative_smoother.passes < 0) {
    throw ValueError("Radiation.imc.conservative_smoother.passes must be >= 0");
  }
  if (!(radiation.imc.conservative_smoother.alpha > 0.0)) {
    throw ValueError("Radiation.imc.conservative_smoother.alpha must be > 0");
  }
  if (radiation.boundary.marshak_particles < 1) {
    tenryu::core::log_warning(
        "Radiation.boundary.marshak_particles < 1; clamping to 1");
    radiation.boundary.marshak_particles = 1;
  }
  if (!(radiation.ddmc.tau_ddmc >= 1.0)) {
    throw ValueError("Radiation.ddmc.tau_ddmc must be >= 1");
  }
  if (!(radiation.ddmc.tau_rw >= 0.0)) {
    throw ValueError("Radiation.ddmc.tau_rw must be >= 0");
  }
  if (radiation.ddmc.tau_ddmc_off >= 0.0 &&
      !(radiation.ddmc.tau_ddmc_off >= 0.5 &&
        radiation.ddmc.tau_ddmc_off <= radiation.ddmc.tau_ddmc)) {
    throw ValueError("Radiation.ddmc.tau_ddmc_off must satisfy "
                     "tau_ddmc_off < 0 or (0.5 <= tau_ddmc_off <= tau_ddmc)");
  }
  if (radiation.ddmc.omega_ddmc_off >= 0.0 &&
      !(radiation.ddmc.omega_ddmc_off >= 0.0 &&
        radiation.ddmc.omega_ddmc_off <= radiation.ddmc.omega_ddmc)) {
    throw ValueError("Radiation.ddmc.omega_ddmc_off must satisfy "
                     "omega_ddmc_off < 0 or (0 <= omega_ddmc_off <= omega_ddmc)");
  }
  if (radiation.ddmc.mode_hold < 0 || radiation.ddmc.mode_hold > 100) {
    throw ValueError("Radiation.ddmc.mode_hold must satisfy 0 <= mode_hold <= 100");
  }
  if (!(radiation.ddmc.rate_max > 0.0)) {
    throw ValueError("Radiation.ddmc.rate_max must be > 0");
  }
  if (!(radiation.diffusion.tau_on >= radiation.diffusion.tau_off &&
        radiation.diffusion.tau_off > 0.0)) {
    throw ValueError(
        "Radiation.diffusion must satisfy tau_on >= tau_off > 0");
  }
  if (!(radiation.diffusion.reduced_flux_on >= 0.0 &&
        radiation.diffusion.reduced_flux_on <= radiation.diffusion.reduced_flux_off &&
        radiation.diffusion.reduced_flux_off <= 1.0)) {
    throw ValueError(
        "Radiation.diffusion must satisfy 0 <= reduced_flux_on <= reduced_flux_off <= 1");
  }
  if (radiation.diffusion.mode_update_interval < 1) {
    throw ValueError("Radiation.diffusion.mode_update_interval must be >= 1");
  }
  if (radiation.diffusion.min_diffusion_island_cells < 1) {
    throw ValueError("Radiation.diffusion.min_diffusion_island_cells must be >= 1");
  }
  if (radiation.diffusion.imc_guard_cells < 1) {
    throw ValueError("Radiation.diffusion.imc_guard_cells must be >= 1");
  }
  if (radiation.diffusion.sts_max_stages < 0) {
    throw ValueError("Radiation.diffusion.sts_max_stages must be >= 0");
  }
  if (!(radiation.diffusion.sts_damping > 0.0 &&
        radiation.diffusion.sts_damping < 1.0)) {
    throw ValueError("Radiation.diffusion.sts_damping must be in (0, 1)");
  }
  if (!(radiation.diffusion.sts_subcycle_eta > 0.0 &&
        radiation.diffusion.sts_subcycle_eta <= 1.0)) {
    throw ValueError("Radiation.diffusion.sts_subcycle_eta must be in (0, 1]");
  }
  if (radiation.diffusion.interface_particles_per_face_group < 1) {
    throw ValueError(
        "Radiation.diffusion.interface_particles_per_face_group must be >= 1");
  }
  if (radiation.diffusion.exit_particles_per_cell_group < 1) {
    throw ValueError(
        "Radiation.diffusion.exit_particles_per_cell_group must be >= 1");
  }
  if (!(radiation.diffusion.lte_entry_energy_fraction_cap >= 0.0)) {
    throw ValueError(
        "Radiation.diffusion.lte_entry_energy_fraction_cap must be >= 0");
  }
  if (!(radiation.holo.coupling_tau >= 0.0)) {
    throw ValueError("Radiation.holo.coupling_tau must be >= 0");
  }
  if (radiation.holo.guard_cells < 0) {
    throw ValueError("Radiation.holo.guard_cells must be >= 0");
  }
  if (radiation.holo.blend_cells < 0) {
    throw ValueError("Radiation.holo.blend_cells must be >= 0");
  }
  if (radiation.holo.min_lo_cells < 0) {
    throw ValueError("Radiation.holo.min_lo_cells must be >= 0");
  }
  if (!(radiation.holo.q_min >= 0.0 &&
        radiation.holo.q_min <= radiation.holo.q_max &&
        radiation.holo.q_max <= 1.0)) {
    throw ValueError("Radiation.holo must satisfy 0 <= q_min <= q_max <= 1");
  }
  const bool holo_legacy_tau =
      radiation.holo.tau_on == 0.0 && radiation.holo.tau_off == 0.0;
  if (!holo_legacy_tau &&
      !(radiation.holo.tau_on >= radiation.holo.tau_off &&
        radiation.holo.tau_off > 0.0)) {
    throw ValueError(
        "Radiation.holo must satisfy tau_on >= tau_off > 0, or tau_on=tau_off=0");
  }
  if (!(radiation.holo.reduced_flux_on >= 0.0 &&
        radiation.holo.reduced_flux_on <= radiation.holo.reduced_flux_off &&
        radiation.holo.reduced_flux_off <= 1.0)) {
    throw ValueError(
        "Radiation.holo must satisfy 0 <= reduced_flux_on <= reduced_flux_off <= 1");
  }
  if (radiation.holo.update_interval < 1) {
    throw ValueError("Radiation.holo.update_interval must be >= 1");
  }
  if (radiation.holo.hold_on < 0) {
    throw ValueError("Radiation.holo.hold_on must be >= 0");
  }
  if (radiation.holo.min_dwell_steps < 0) {
    throw ValueError("Radiation.holo.min_dwell_steps must be >= 0");
  }
  if (radiation.holo.min_island_cells < 1) {
    throw ValueError("Radiation.holo.min_island_cells must be >= 1");
  }
  if (radiation.holo.core_margin_cells < 0) {
    throw ValueError("Radiation.holo.core_margin_cells must be >= 0");
  }
  if (radiation.holo.region != "shell") {
    throw ConfigError("Radiation.holo.region must be \"shell\" in v1");
  }
  if (radiation.holo.material_group != "shell") {
    throw ConfigError("Radiation.holo.material_group must be \"shell\" in v1");
  }
  if (radiation.holo.solver != "implicit_1d" &&
      radiation.holo.solver != "quasidiffusion_1d") {
    throw ConfigError(
        "Radiation.holo.solver must be \"implicit_1d\" or \"quasidiffusion_1d\"");
  }
  if (radiation.holo.closure != "diffusion") {
    throw ConfigError("Radiation.holo.closure must be \"diffusion\" in v1");
  }
  if (!(radiation.holo.closure_relax >= 0.0 &&
        radiation.holo.closure_relax <= 1.0)) {
    throw ValueError("Radiation.holo.closure_relax must be in [0, 1]");
  }
  if (radiation.holo.closure_smooth_passes < 0) {
    throw ValueError("Radiation.holo.closure_smooth_passes must be >= 0");
  }
  if (!(radiation.holo.closure_smooth_alpha >= 0.0 &&
        radiation.holo.closure_smooth_alpha <= 1.0)) {
    throw ValueError("Radiation.holo.closure_smooth_alpha must be in [0, 1]");
  }
  if (!(radiation.holo.consistency_alpha >= 0.0 &&
        radiation.holo.consistency_alpha <= 1.0)) {
    throw ValueError("Radiation.holo.consistency_alpha must be in [0, 1]");
  }
  if (radiation.holo.boundary_flux != "physical") {
    throw ConfigError(
        "Radiation.holo.boundary_flux must be \"physical\" in v1");
  }
  if (radiation.holo.sn_n_angles < 2 || (radiation.holo.sn_n_angles % 2) != 0) {
    throw ValueError("Radiation.holo.sn_n_angles must be an even integer >= 2");
  }
  if (radiation.holo.sn_material_coupling && !radiation.holo.sn_closure) {
    throw ValueError(
        "Radiation.holo.sn_material_coupling requires Radiation.holo.sn_closure=true");
  }
  if (radiation.holo.residual_particles_per_cell_group < 1) {
    throw ValueError(
        "Radiation.holo.residual_particles_per_cell_group must be >= 1");
  }
  if (radiation.holo.enabled) {
    tenryu::core::log_warning(
        "Radiation.holo.enabled=true: HOLO is experimental and not validated "
        "for production use. Use Radiation.imc.difference for production "
        "oscillation reduction.");
    if (radiation.imc.difference.enabled) {
      throw ValueError("Radiation.holo.enabled and Radiation.imc.difference.enabled "
                       "cannot both be true. HOLO+DF simultaneous operation is not "
                       "supported. Use difference formulation (DF) only for production.");
    }
  }
  if (radiation.holo.enabled && main.dimension == "2D_RZ" &&
      !radiation.holo.sn_material_coupling) {
    tenryu::core::log_warning(
        "Radiation.holo.enabled=True is only supported for Main.dimension=\"1D_SPH\" "
        "in v1; disabling HOLO for 2D_RZ");
    radiation.holo.enabled = false;
  }

  if (!is_ddmc_leak_stencil(radiation.ddmc.leak_stencil)) {
    throw ConfigError(
        "Radiation.ddmc.leak_stencil must be \"4\" or \"9_kershaw\"");
  }
  if (!is_ddmc_interface_method(radiation.ddmc.interface_method)) {
    throw ConfigError("Radiation.ddmc.interface_method must be one of "
                      "{\"asymptotic_diffusion_limit\", \"marshak\", "
                      "\"cleveland_gentile\"}");
  }
  if (!is_ddmc_interface_exit_distribution(radiation.ddmc.interface_exit_distribution)) {
    throw ConfigError(
        "Radiation.ddmc.interface_exit_distribution must be \"cosine\" or \"half_isotropic\"");
  }
  if (!is_ddmc_face_opacity_temperature(radiation.ddmc.face_opacity_temperature)) {
    throw ConfigError(
        "Radiation.ddmc.face_opacity_temperature must be \"radiative_mean\"");
  }
  if (!radiation.enabled && radiation.ddmc.enabled) {
    tenryu::core::log_warning(
        "Radiation.enabled=False with ddmc.enabled=True; DDMC settings will be ignored");
  }
  if (radiation.ddmc.interface_method != "asymptotic_diffusion_limit") {
    tenryu::core::log_warning(
        "Radiation.ddmc.interface_method=\"" + radiation.ddmc.interface_method +
        "\" is parsed but not implemented in v1.0; using asymptotic_diffusion_limit");
  }
  if (radiation.imc.census_comb.enabled) {
    if (radiation.imc.census_comb.max_particles < 1) {
      throw ConfigError("Radiation.imc.census_comb.max_particles must be >= 1");
    }
    if (radiation.imc.census_comb.min_per_bin < 1) {
      throw ConfigError("Radiation.imc.census_comb.min_per_bin must be >= 1");
    }
    if (!(radiation.imc.census_comb.trigger_ratio > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.trigger_ratio must be > 0");
    }
    if (!(radiation.imc.census_comb.target_fraction > 0.0 &&
          radiation.imc.census_comb.target_fraction <= 1.0)) {
      throw ValueError("Radiation.imc.census_comb.target_fraction must be in (0, 1]");
    }
    if (!(radiation.imc.census_comb.mode_weight_imc > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.mode_weight_imc must be > 0");
    }
    if (!(radiation.imc.census_comb.mode_weight_ddmc > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.mode_weight_ddmc must be > 0");
    }
    if (!(radiation.imc.census_comb.adaptive_util_start >= 0.0 &&
          radiation.imc.census_comb.adaptive_util_start <
              radiation.imc.census_comb.adaptive_util_end &&
          radiation.imc.census_comb.adaptive_util_end <= 1.0)) {
      throw ValueError(
          "Radiation.imc.census_comb adaptive util must satisfy 0 <= start < end <= 1");
    }
    if (!(radiation.imc.census_comb.trigger_ratio >=
          radiation.imc.census_comb.trigger_ratio_floor)) {
      throw ValueError("Radiation.imc.census_comb.trigger_ratio must be >= trigger_ratio_floor");
    }
    if (!(radiation.imc.census_comb.trigger_ratio_floor >=
          radiation.imc.census_comb.target_fraction +
              radiation.imc.census_comb.trigger_hysteresis)) {
      throw ValueError(
          "Radiation.imc.census_comb.trigger_ratio_floor must be >= "
          "target_fraction + trigger_hysteresis");
    }
    if (!(radiation.imc.census_comb.trigger_hysteresis >= 0.0)) {
      throw ValueError("Radiation.imc.census_comb.trigger_hysteresis must be >= 0");
    }
    if (!(radiation.imc.census_comb.ess_min_tier0 > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.ess_min_tier0 must be > 0");
    }
    if (!(radiation.imc.census_comb.ess_min_tier1 > 0.0)) {
      throw ValueError("Radiation.imc.census_comb.ess_min_tier1 must be > 0");
    }
    if (!(radiation.imc.census_comb.max_split_factor >= 1)) {
      throw ConfigError("Radiation.imc.census_comb.max_split_factor must be >= 1");
    }
  }

// ---- freeze.cpp: serialize_radiation (the Radiation.imc, .ddmc, .diffusion dicts)

py::dict serialize_radiation(const Config::RadiationConfig& radiation) {
  py::dict imc;
  py::dict census_comb;
  census_comb["enabled"] = radiation.imc.census_comb.enabled;
  census_comb["max_particles"] = radiation.imc.census_comb.max_particles;
  census_comb["min_per_bin"] = radiation.imc.census_comb.min_per_bin;
  census_comb["trigger_ratio"] = radiation.imc.census_comb.trigger_ratio;
  census_comb["target_fraction"] = radiation.imc.census_comb.target_fraction;
  census_comb["mode_weight_imc"] = radiation.imc.census_comb.mode_weight_imc;
  census_comb["mode_weight_ddmc"] = radiation.imc.census_comb.mode_weight_ddmc;
  census_comb["adaptive_trigger"] = radiation.imc.census_comb.adaptive_trigger;
  census_comb["adaptive_util_start"] = radiation.imc.census_comb.adaptive_util_start;
  census_comb["adaptive_util_end"] = radiation.imc.census_comb.adaptive_util_end;
  census_comb["trigger_ratio_floor"] = radiation.imc.census_comb.trigger_ratio_floor;
  census_comb["trigger_hysteresis"] = radiation.imc.census_comb.trigger_hysteresis;
  census_comb["ess_floor_enabled"] = radiation.imc.census_comb.ess_floor_enabled;
  census_comb["ess_min_tier0"] = radiation.imc.census_comb.ess_min_tier0;
  census_comb["ess_min_tier1"] = radiation.imc.census_comb.ess_min_tier1;
  census_comb["max_split_factor"] = radiation.imc.census_comb.max_split_factor;
  imc["enabled"] = radiation.imc.enabled;
  imc["alpha"] = radiation.imc.alpha;
  imc["f_max"] = radiation.imc.f_max;
  imc["corrected_fleck"] = radiation.imc.corrected_fleck;
  imc["particles_per_cell_group"] = radiation.imc.particles_per_cell_group;
  imc["implicit_capture"] = radiation.imc.implicit_capture;
  imc["cutoff_fraction"] = radiation.imc.cutoff_fraction;
  imc["inelastic_scatter"] = radiation.imc.inelastic_scatter;
  imc["weight_cutoff"] = radiation.imc.weight_cutoff;
  imc["roulette_survival"] = radiation.imc.roulette_survival;
  imc["weight_split"] = radiation.imc.weight_split;
  imc["max_split"] = radiation.imc.max_split;
  imc["linearized_planck"] = radiation.imc.linearized_planck;
  imc["source_tilting"] = radiation.imc.source_tilting;
  imc["source_localization"] = radiation.imc.source_localization;
  imc["sloc_ema_beta"] = radiation.imc.sloc_ema_beta;
  imc["sloc_sigma_floor"] = radiation.imc.sloc_sigma_floor;
  imc["sloc_sigma_cap"] = radiation.imc.sloc_sigma_cap;
  imc["sloc_tau_ref"] = radiation.imc.sloc_tau_ref;
  imc["spectral_bias_eta"] = radiation.imc.spectral_bias_eta;
  imc["opacity_predictor"] = radiation.imc.opacity_predictor;
  imc["two_stage"] = radiation.imc.two_stage;
  py::dict difference;
  difference["enabled"] = radiation.imc.difference.enabled;
  difference["W_max"] = radiation.imc.difference.W_max;
  difference["tau0"] = radiation.imc.difference.tau0;
  difference["chi0"] = radiation.imc.difference.chi0;
  difference["face_transport"] = radiation.imc.difference.face_transport;
  imc["difference"] = difference;
  py::dict net_e_source_smoothing;
  net_e_source_smoothing["enabled"] = radiation.imc.net_e_source_smoothing.enabled;
  net_e_source_smoothing["alpha"] = radiation.imc.net_e_source_smoothing.alpha;
  net_e_source_smoothing["tau_threshold"] =
      radiation.imc.net_e_source_smoothing.tau_threshold;
  net_e_source_smoothing["passes"] = radiation.imc.net_e_source_smoothing.passes;
  net_e_source_smoothing["grad_Te_scale"] =
      radiation.imc.net_e_source_smoothing.grad_Te_scale;
  net_e_source_smoothing["grad_rho_scale"] =
      radiation.imc.net_e_source_smoothing.grad_rho_scale;
  net_e_source_smoothing["gradient_adaptive"] =
      radiation.imc.net_e_source_smoothing.gradient_adaptive;
  imc["net_e_source_smoothing"] = net_e_source_smoothing;
  imc["particle_budget"] = radiation.imc.particle_budget;
  imc["census_comb"] = census_comb;
  py::dict rad_lite_mesh;
  rad_lite_mesh["enabled"] = radiation.imc.rad_lite_mesh.enabled;
  rad_lite_mesh["sigma_ratio_max"] = radiation.imc.rad_lite_mesh.sigma_ratio_max;
  rad_lite_mesh["nlte_auto"] = radiation.imc.rad_lite_mesh.nlte_auto;
  imc["rad_lite_mesh"] = rad_lite_mesh;

  py::dict ddmc;
  ddmc["enabled"] = radiation.ddmc.enabled;
  ddmc["implicit_diffusion"] = radiation.ddmc.implicit_diffusion;
  ddmc["tau_ddmc"] = radiation.ddmc.tau_ddmc;
  ddmc["tau_rw"] = radiation.ddmc.tau_rw;
  ddmc["omega_ddmc"] = radiation.ddmc.omega_ddmc;
  ddmc["tau_ddmc_off"] = radiation.ddmc.tau_ddmc_off;
  ddmc["omega_ddmc_off"] = radiation.ddmc.omega_ddmc_off;
  ddmc["mode_hold"] = radiation.ddmc.mode_hold;
  ddmc["rate_max"] = radiation.ddmc.rate_max;
  ddmc["leak_stencil"] = radiation.ddmc.leak_stencil;
  ddmc["interface_method"] = radiation.ddmc.interface_method;
  ddmc["emissivity_preserving"] = radiation.ddmc.emissivity_preserving;
  ddmc["interface_exit_distribution"] = radiation.ddmc.interface_exit_distribution;
  ddmc["rz_face_r_weight"] = radiation.ddmc.rz_face_r_weight;
  ddmc["face_opacity_temperature"] = radiation.ddmc.face_opacity_temperature;
  ddmc["m_matrix_check"] = radiation.ddmc.m_matrix_check;

  py::dict diffusion;
  diffusion["enabled"] = radiation.diffusion.enabled;
  diffusion["tau_on"] = radiation.diffusion.tau_on;
  diffusion["tau_off"] = radiation.diffusion.tau_off;
  diffusion["reduced_flux_on"] = radiation.diffusion.reduced_flux_on;
  diffusion["reduced_flux_off"] = radiation.diffusion.reduced_flux_off;
  diffusion["mode_hold"] = radiation.diffusion.mode_hold;
  diffusion["rate_max"] = radiation.diffusion.rate_max;
  diffusion["mode_update_interval"] = radiation.diffusion.mode_update_interval;
  diffusion["min_diffusion_island_cells"] =
      radiation.diffusion.min_diffusion_island_cells;
  diffusion["imc_guard_cells"] = radiation.diffusion.imc_guard_cells;
  diffusion["sts_max_stages"] = radiation.diffusion.sts_max_stages;
  diffusion["sts_damping"] = radiation.diffusion.sts_damping;
  diffusion["sts_subcycle_eta"] = radiation.diffusion.sts_subcycle_eta;
  diffusion["interface_particles_per_face_group"] =
      radiation.diffusion.interface_particles_per_face_group;
  diffusion["exit_particles_per_cell_group"] =
      radiation.diffusion.exit_particles_per_cell_group;
  diffusion["lte_entry_initialization"] =
      radiation.diffusion.lte_entry_initialization;
  diffusion["lte_entry_energy_fraction_cap"] =
      radiation.diffusion.lte_entry_energy_fraction_cap;


// ---- freeze.cpp: serialize_radiation (the Radiation.holo dict)

  py::dict holo;
  holo["enabled"] = radiation.holo.enabled;
  holo["region"] = radiation.holo.region;
  holo["material_group"] = radiation.holo.material_group;
  holo["coupling_tau"] = radiation.holo.coupling_tau;
  holo["guard_cells"] = radiation.holo.guard_cells;
  holo["blend_cells"] = radiation.holo.blend_cells;
  holo["min_lo_cells"] = radiation.holo.min_lo_cells;
  holo["q_min"] = radiation.holo.q_min;
  holo["q_max"] = radiation.holo.q_max;
  holo["tau_on"] = radiation.holo.tau_on;
  holo["tau_off"] = radiation.holo.tau_off;
  holo["reduced_flux_on"] = radiation.holo.reduced_flux_on;
  holo["reduced_flux_off"] = radiation.holo.reduced_flux_off;
  holo["update_interval"] = radiation.holo.update_interval;
  holo["hold_on"] = radiation.holo.hold_on;
  holo["min_dwell_steps"] = radiation.holo.min_dwell_steps;
  holo["min_island_cells"] = radiation.holo.min_island_cells;
  holo["core_margin_cells"] = radiation.holo.core_margin_cells;
  holo["solver"] = radiation.holo.solver;
  holo["closure"] = radiation.holo.closure;
  holo["closure_relax"] = radiation.holo.closure_relax;
  holo["closure_smooth_passes"] = radiation.holo.closure_smooth_passes;
  holo["closure_smooth_alpha"] = radiation.holo.closure_smooth_alpha;
  holo["consistency_alpha"] = radiation.holo.consistency_alpha;
  holo["gamma_alpha"] = radiation.holo.consistency_alpha;
  holo["boundary_flux"] = radiation.holo.boundary_flux;
  holo["p_rr_tally"] = radiation.holo.p_rr_tally;
  holo["sn_closure"] = radiation.holo.sn_closure;
  holo["sn_n_angles"] = radiation.holo.sn_n_angles;
  holo["sn_material_coupling"] = radiation.holo.sn_material_coupling;
  holo["residual_particles_per_cell_group"] =
      radiation.holo.residual_particles_per_cell_group;

// ============================================================================================================
// Keys that only the Monte Carlo radiation read, retired with it on 2026-09-29 as well (5bc8f6ce3): Materials opacity
// lambda_method and f_min, Radiation.origin_parity_only, Numerics.safety.opacity_floor / opacity_cap and
// Parallel.migration. The builder accepts them and ignores them (a warning; lambda_method and f_min at their former
// defaults pass silently), and they are not in the frozen configuration any more.
// ============================================================================================================

// ---- config.hpp: MatDef Non-LTE Fleck controls (lambda_method, nlte_f_min kept only by the Monte Carlo radiation)
      // Legacy Non-LTE Fleck controls kept for namelist compatibility.
      // Jayenne separate-emissivity reformulation ignores these at runtime.
      std::string lambda_method = "finite_difference";
      double lambda_fd_delta_rel = 1.0e-4;
      double lambda_fd_abs_min = 1.0e-6;
      double nlte_f_min = 1.0e-4;

// ---- config.hpp: RadiationConfig::origin_parity_only (read only by the HOLO S_N closure)
    bool origin_parity_only = false;

// ---- config.hpp: SafetyConfig opacity clamp (read only by the Monte Carlo radiation)
      double opacity_floor = 1e-20;
      double opacity_cap = 1e20;

// ---- config.hpp: ParallelConfig::Migration (photon-packet migration settings)
    struct Migration {
      std::string method = "batch";
      int max_substeps = 32;
      int emigrant_threshold = 1000;
      int initial_capacity = 10000;
      double growth_factor = 1.5;
    };

// ---- builder.cpp: is_lambda_method
bool is_lambda_method(const std::string& value) {
  return value == "finite_difference" || value == "freeze_opacity";
}

// ---- builder.cpp: Materials opacity lambda_method / f_min parsing
      if (has_key(opacity, "lambda_method")) {
        def.lambda_method = strict_string(
            opacity["lambda_method"],
            "Materials.materials[" + std::to_string(i) + "].opacity.lambda_method");
        if (!is_lambda_method(def.lambda_method)) {
          throw ValueError(
              "Materials.materials[" + std::to_string(i) +
              "].opacity.lambda_method must be one of {\"finite_difference\", \"freeze_opacity\"}, got " +
              def.lambda_method);
        }
      if (has_key(opacity, "f_min")) {
        def.nlte_f_min = numeric_as_double(
            opacity["f_min"],
            "Materials.materials[" + std::to_string(i) + "].opacity.f_min");
      }

// ---- builder.cpp: Radiation.origin_parity_only parsing
  if (has_key(kwargs, "origin_parity_only")) {
    radiation.origin_parity_only =
        strict_bool(kwargs["origin_parity_only"], "Radiation.origin_parity_only");
  }

// ---- builder.cpp: Numerics.safety opacity_floor / opacity_cap parsing
    if (has_key(safety, "opacity_floor")) {
      numerics.safety.opacity_floor =
          numeric_as_double(safety["opacity_floor"], "Numerics.safety.opacity_floor");
    }
    if (has_key(safety, "opacity_cap")) {
      numerics.safety.opacity_cap =
          numeric_as_double(safety["opacity_cap"], "Numerics.safety.opacity_cap");
    }

// ---- builder.cpp: Parallel.migration parsing
  if (has_key(kwargs, "migration")) {
    const py::handle migration_obj = kwargs["migration"];
    if (!py::isinstance<py::dict>(migration_obj)) {
      throw_value_type_error("Parallel.migration", "dict", migration_obj);
    }
    const py::dict migration = py::reinterpret_borrow<py::dict>(migration_obj);
    enforce_known_keys(migration, "Parallel.migration",
                       {"method", "max_substeps", "emigrant_threshold",
                        "initial_capacity", "growth_factor"});
    if (has_key(migration, "method")) {
      parallel.migration.method =
          strict_string(migration["method"], "Parallel.migration.method");
    }
    if (has_key(migration, "max_substeps")) {
      parallel.migration.max_substeps =
          strict_int32(migration["max_substeps"], "Parallel.migration.max_substeps");
    }
    if (has_key(migration, "emigrant_threshold")) {
      parallel.migration.emigrant_threshold = strict_int32(
          migration["emigrant_threshold"], "Parallel.migration.emigrant_threshold");
    }
    if (has_key(migration, "initial_capacity")) {
      parallel.migration.initial_capacity = strict_int32(
          migration["initial_capacity"], "Parallel.migration.initial_capacity");
    }
    if (has_key(migration, "growth_factor")) {
      parallel.migration.growth_factor = numeric_as_double(
          migration["growth_factor"], "Parallel.migration.growth_factor");
    }
  }

// ---- builder.cpp: validation of the above
      if (!is_lambda_method(mat.lambda_method)) {
        throw ConfigError("Materials.materials[\"" + mat.name +
                          "\"].opacity.lambda_method must be one of "
                          "{\"finite_difference\", \"freeze_opacity\"}");
      }
      if (!(mat.nlte_f_min > 0.0 && mat.nlte_f_min <= 1.0)) {
        throw ValueError("Materials.materials[\"" + mat.name +
                         "\"].opacity.f_min must be in (0, 1]");
      }
  if (parallel.migration.max_substeps < 1) {
    throw ConfigError("Parallel.migration.max_substeps must be >= 1");
  }
  if (parallel.migration.emigrant_threshold < 1) {
    throw ConfigError("Parallel.migration.emigrant_threshold must be >= 1");
  }
  if (parallel.migration.initial_capacity < 1) {
    throw ConfigError("Parallel.migration.initial_capacity must be >= 1");
  }
  // opacity clamp consistency
  if (numerics.safety.opacity_floor > numerics.safety.opacity_cap) {
    throw ConfigError("Numerics.safety.opacity_floor must be <= opacity_cap");
  }

// ---- freeze.cpp: serialization of the above
    m["lambda_method"] = mat.lambda_method;
    m["f_min"] = mat.nlte_f_min;
  if (radiation.origin_parity_only) {
    out["origin_parity_only"] = radiation.origin_parity_only;
  }
  safety["opacity_floor"] = numerics.safety.opacity_floor;
  safety["opacity_cap"] = numerics.safety.opacity_cap;
  py::dict migration;
  migration["method"] = parallel.migration.method;
  migration["max_substeps"] = parallel.migration.max_substeps;
  migration["emigrant_threshold"] = parallel.migration.emigrant_threshold;
  migration["initial_capacity"] = parallel.migration.initial_capacity;
  migration["growth_factor"] = parallel.migration.growth_factor;
