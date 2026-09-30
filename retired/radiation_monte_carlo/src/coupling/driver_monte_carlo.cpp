// Retired with the Monte Carlo radiation on 2026-09-29: the parts of src/coupling/driver.cpp (as it was at 5bc8f6ce3)
// that served only Radiation.mode "imc_ddmc" (IMC, DDMC, random walk, HOLO, difference formulation). Not built.
// The rest of that code was inline in Driver::run and is best read in place at 5bc8f6ce3 (git show
// 5bc8f6ce3:src/coupling/driver.cpp): the radiation::IMC object and its restart photon pool, the photon
// migration between MPI ranks, the census and escaped-energy bookkeeping of the energy budget, the HOLO step
// tallies, the difference-formulation reference invalidation after the hydro, the time-step limit
// IMC::compute_dt_rad (Fleck-factor floor, NUMERICS §2.2 (c)) and its lineage and history columns, and the
// history fill of the mc/*, holo/* and difference/* groups.

namespace tenryu::coupling {
namespace {

// ---- assert_driver_retry_supported

void assert_driver_retry_supported(const core::Config& cfg) {
  if (cfg.numerics.hydro.driver_full_step_retry_enabled &&
      cfg.radiation.mode == core::RadiationMode::ImcDdmc) {
    TENRYU_ASSERT(false,
                  "Numerics.hydro.driver_full_step_retry_enabled does not support "
                  "RadiationMode::ImcDdmc (v1 scope: deterministic FLD/SN only). "
                  "Set retry_enabled=false or change radiation.mode.");
  }
  if (hydro::i1b_path_guard_enabled() &&
      cfg.radiation.mode == core::RadiationMode::ImcDdmc) {
    TENRYU_ASSERT(false,
                  "TENRYU_I1B_PATH_GUARD does not support RadiationMode::ImcDdmc "
                  "because its full-step snapshot retry is deterministic FLD/SN only. "
                  "Disable the path guard or change radiation.mode.");
  }
}

// ---- fleck diagnostics log

bool fleck_diag_has_radius_window(const core::Config::DiagnosticsConfig::FleckDiag& cfg) {
  return cfg.r_min_cm >= 0.0 && cfg.r_max_cm >= 0.0;
}

bool fleck_diag_requested(const core::Config& cfg) {
  return cfg.main.verbosity == "verbose" ||
         (cfg.diagnostics.enabled && cfg.diagnostics.fleck_diag.enabled);
}

void log_fleck_diagnostics_if_needed(const core::State& state,
                                     const core::Config& cfg,
                                     const radiation::IMC& imc) {
  const auto& diag = cfg.diagnostics.fleck_diag;
  if (!fleck_diag_requested(cfg)) {
    return;
  }

  const bool use_radius = fleck_diag_has_radius_window(diag);
  const bool has_selection = !diag.cells.empty() || use_radius;
  if (!has_selection) {
    if (diag.enabled) {
      static int warn_count = 0;
      ++warn_count;
      if (warn_count == 1 || warn_count % 100 == 0) {
        core::log_warning(
            "Diagnostics.fleck_diag is enabled but no cells or radius window were configured");
      }
    }
    return;
  }

  const int step_number = state.step + 1;
  if (step_number <= 0 || (step_number % std::max(diag.every, 1)) != 0) {
    return;
  }

  if (state.mesh.dim != 1) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning("Diagnostics.fleck_diag currently supports 1D_SPH only; skipping");
    }
    return;
  }

  const auto* coeffs = imc.last_cell_radiation_coeffs();
  if (coeffs == nullptr) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning(
          "Diagnostics.fleck_diag requested but no NLTE/TMAT cell-radiation coefficients are "
          "available for this step");
    }
    return;
  }

  const std::size_t n_cells = state.rho.size();
  TENRYU_ASSERT(coeffs->n_cells == static_cast<int>(n_cells),
                "fleck_diag coefficient/state cell count mismatch");
  TENRYU_ASSERT(coeffs->rho_eval.size() == n_cells,
                "fleck_diag requires rho_eval size == n_cells");
  TENRYU_ASSERT(coeffs->Te_eval.size() == n_cells,
                "fleck_diag requires Te_eval size == n_cells");
  TENRYU_ASSERT(coeffs->cv_e.size() == n_cells,
                "fleck_diag requires cv_e size == n_cells");
  TENRYU_ASSERT(coeffs->beta.size() == n_cells,
                "fleck_diag requires beta size == n_cells");
  TENRYU_ASSERT(coeffs->sigma_p_em.size() == n_cells,
                "fleck_diag requires sigma_p_em size == n_cells");
  TENRYU_ASSERT(coeffs->f.size() == n_cells,
                "fleck_diag requires f size == n_cells");
  TENRYU_ASSERT(coeffs->eta_tot.size() == n_cells,
                "fleck_diag requires eta_tot size == n_cells");
  TENRYU_ASSERT(state.rad_dep.size() % std::max<std::size_t>(n_cells, 1) == 0,
                "fleck_diag requires rad_dep size divisible by n_cells");
  TENRYU_ASSERT(state.rad_emit.empty() || state.rad_emit.size() == state.rad_dep.size(),
                "fleck_diag requires rad_emit size == rad_dep size when present");

  std::vector<char> selected(n_cells, 0);
  std::vector<int> cells;
  cells.reserve(diag.cells.size() + 8);

  int invalid_fixed = 0;
  for (const int cell : diag.cells) {
    if (cell < 0 || static_cast<std::size_t>(cell) >= n_cells) {
      ++invalid_fixed;
      continue;
    }
    if (!selected[static_cast<std::size_t>(cell)]) {
      selected[static_cast<std::size_t>(cell)] = 1;
      cells.push_back(cell);
    }
  }
  if (invalid_fixed > 0) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning("Diagnostics.fleck_diag ignored " + std::to_string(invalid_fixed) +
                        " out-of-range local cell indices");
    }
  }

  if (use_radius) {
    TENRYU_ASSERT(state.x_r.size() == n_cells + 1,
                  "fleck_diag requires x_r node count = n_cells + 1 in 1D");
    std::vector<double> node_r(state.x_r.size(), 0.0);
    state.x_r.copy_to_host(node_r.data());
    for (std::size_t c = 0; c < n_cells; ++c) {
      const double rc = 0.5 * (node_r[c] + node_r[c + 1]);
      if (rc < diag.r_min_cm || rc > diag.r_max_cm) {
        continue;
      }
      if (!selected[c]) {
        selected[c] = 1;
        cells.push_back(static_cast<int>(c));
      }
    }
  }

  if (cells.empty()) {
    return;
  }

  std::vector<double> rad_dep(state.rad_dep.size(), 0.0);
  std::vector<double> rad_emit(state.rad_dep.size(), 0.0);
  state.rad_dep.copy_to_host(rad_dep.data());
  if (!state.rad_emit.empty()) {
    state.rad_emit.copy_to_host(rad_emit.data());
  }
  const int n_groups =
      (n_cells > 0) ? static_cast<int>(state.rad_dep.size() / n_cells) : 0;

  for (const int cell : cells) {
    const std::size_t c = static_cast<std::size_t>(cell);
    double dep_sum = 0.0;
    double E_emit_cell = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx =
          c * static_cast<std::size_t>(n_groups) + static_cast<std::size_t>(g);
      dep_sum += rad_dep[idx];
      E_emit_cell += rad_emit[idx];
    }
    const double delta_E = dep_sum - E_emit_cell;

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(16);
    oss << "[fleck_diag] step=" << step_number
        << " cell=" << cell
        << " Te=" << coeffs->Te_eval[c]
        << " rho=" << coeffs->rho_eval[c]
        << " cv_e=" << coeffs->cv_e[c]
        << " sigma_p_em=" << coeffs->sigma_p_em[c]
        << " beta=" << coeffs->beta[c]
        << " f=" << coeffs->f[c]
        << " eta_tot=" << coeffs->eta_tot[c]
        << " E_emit=" << E_emit_cell
        << " dep_sum=" << dep_sum
        << " delta_E=" << delta_E;
    core::log_info(oss.str());
  }
}

// ---- output_if_needed (unused)

void output_if_needed(core::State& state,
                      const core::Config& cfg,
                      io::OutputManager& out,
                      const radiation::IMC& imc,
                      const std::string& case_name,
                      const int rank) {
  if (out.should_plot(state.step, state.t, state, cfg)) {
    if (rank == 0) {
      out.write_snapshot(state, cfg, state.step, state.t, case_name, rank);
      out.write_run_info(state, cfg);
      core::log_info("[output] wrote snapshot (step=" + std::to_string(state.step) +
                     ", t=" + format_sci(state.t) + ")");
    }
    update_next_output_time(state.t_next_plot, cfg.output.plot_every_s, state.t);
  }

  if (out.should_history(state.step, state.t, state, cfg)) {
    update_next_output_time(state.t_next_history, cfg.output.history_every_s, state.t);
  }

  if (out.should_checkpoint(state.step, state.t, state, cfg)) {
    update_next_output_time(state.t_next_checkpoint, cfg.output.checkpoint_every_s,
                            state.t);
    if (rank == 0) {
      out.write_checkpoint(state,
                           cfg,
                           imc.photon_pool(),
                           state.step,
                           state.t,
                           case_name,
                           rank);
      core::log_info("[output] wrote checkpoint (step=" + std::to_string(state.step) +
                     ", t=" + format_sci(state.t) + ")");
    }
  }
}

// ---- compute_holo_E_LO_total

double compute_holo_E_LO_total(const core::State& state, const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  if (!state.holo_core_mask_valid || state.holo_core_mask.size() != n_cells ||
      state.holo_E_LO.size() != n_cell_groups || state.vol.size() != n_cells) {
    return 0.0;
  }

  const auto E_LO = copy_field_to_host(state.holo_E_LO);
  const auto vol = copy_field_to_host(state.vol);
  long double total = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const double vol_c = vol[c];
    if (!std::isfinite(vol_c) || vol_c <= 0.0) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const double E = E_LO[c * n_groups + g];
      if (std::isfinite(E)) {
        total += static_cast<long double>(E) * static_cast<long double>(vol_c);
      }
    }
  }
  return static_cast<double>(total);
}

// ---- HOLO particle source and P_rr summaries

double compute_holo_particle_net_source_core(const core::State& state,
                                             const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  if (!state.holo_core_mask_valid || state.holo_core_mask.size() != n_cells ||
      state.rad_dep.size() != n_cell_groups || state.rad_emit.size() != n_cell_groups) {
    return 0.0;
  }

  const auto rad_dep = copy_field_to_host(state.rad_dep);
  const auto rad_emit = copy_field_to_host(state.rad_emit);
  long double total = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (state.holo_core_mask[c] == 0U) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const std::size_t key = c * n_groups + g;
      const double dep = rad_dep[key];
      const double emit = rad_emit[key];
      if (std::isfinite(dep) && std::isfinite(emit)) {
        total += static_cast<long double>(dep - emit);
      }
    }
  }
  return static_cast<double>(total);
}

struct HoloPrrSummary {
  double coverage = 0.0;
  double chi_min = 0.0;
  double chi_mean = 0.0;
  double chi_max = 0.0;
};

HoloPrrSummary compute_holo_prr_summary(const core::State& state,
                                        const core::Config& cfg) {
  HoloPrrSummary out{};
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  if (!state.holo_core_mask_valid || state.holo_core_mask.size() != n_cells ||
      state.holo_chi.size() != n_cell_groups ||
      state.holo_Prr_coverage.size() != n_cell_groups) {
    return out;
  }

  const auto chi = copy_field_to_host(state.holo_chi);
  const auto coverage = copy_field_to_host(state.holo_Prr_coverage);
  long double coverage_sum = 0.0L;
  long double chi_sum = 0.0L;
  double chi_min = std::numeric_limits<double>::infinity();
  double chi_max = -std::numeric_limits<double>::infinity();
  std::size_t count = 0;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (state.holo_core_mask[c] == 0U) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const std::size_t key = c * n_groups + g;
      const double cov = coverage[key];
      const double chi_value = chi[key];
      if (!std::isfinite(cov) || !std::isfinite(chi_value)) {
        continue;
      }
      coverage_sum += static_cast<long double>(std::clamp(cov, 0.0, 1.0));
      chi_sum += static_cast<long double>(chi_value);
      chi_min = std::min(chi_min, chi_value);
      chi_max = std::max(chi_max, chi_value);
      ++count;
    }
  }
  if (count > 0U) {
    const double inv = 1.0 / static_cast<double>(count);
    out.coverage = static_cast<double>(coverage_sum) * inv;
    out.chi_min = chi_min;
    out.chi_mean = static_cast<double>(chi_sum) * inv;
    out.chi_max = chi_max;
  }
  return out;
}

// ---- run_radiation_stage (with the Monte Carlo branches) and the photon migration

      const auto migrate_radiation_particles = [&]() {
        if (part_info.n_ranks > 1) {
          cudaStream_t mig_stream = nullptr;
          parallel::detect_emigrants(imc.photon_pool(), part_info, emigrants, mig_stream);
          parallel::exchange_emigrants(
              part_info, comm_buffers, emigrants, immigrants, mig_stream);
          parallel::merge_immigrants(imc.photon_pool(), immigrants, false, mig_stream);
        }
      };
      // drive_time_s: evaluation time of the 1D FLD / S_N boundary drive
      // (Marshak T_r(t), pulsed flux) = midpoint of the interval this stage
      // advances; NaN keeps the solver's historic state.t.
      const auto run_radiation_stage = [&](const double dt_stage,
                                           const double t_stage,
                                           const double drive_time_s) {
        const bool deterministic_stage =
            cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion ||
            cfg.radiation.mode == core::RadiationMode::SnTransport;
        const double escaped_before = imc.escaped_energy_total();
        mark_rad_subphase(rad_other_ms);
        if (replicated_radiation_1d) {
          allgather_1d_radiation_inputs();
        } else {
          exchange_radiation_halo();
        }
        mark_rad_subphase(rad_exchange_ms);
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::FldSolve,
            diagnostics::RadialFourierStagePhase::Before,
            t_stage);
        imc.transport_step(state, cfg, dt_stage, part_info, &comm_buffers,
                           drive_time_s);
        if (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion) {
          ++step_fld_solves;
          step_fld_outer_iterations +=
              static_cast<std::int64_t>(state.fld_outer_iterations);
          // A NaN exit residual stays visible in the history.
          const double solve_residual = state.fld_outer_residual;
          if (!std::isnan(step_fld_outer_residual) &&
              (std::isnan(solve_residual) ||
               solve_residual > step_fld_outer_residual)) {
            step_fld_outer_residual = solve_residual;
          }
          step_fld_converged = step_fld_converged && state.fld_converged;
        }
        if (cfg.radiation.mode == core::RadiationMode::SnTransport) {
          ++step_sn_solves;
          step_sn_outer_iterations += static_cast<std::int64_t>(state.sn_outer_iterations);
          step_sn_inner_iterations += static_cast<std::int64_t>(state.sn_inner_iterations);
          const double solve_residual = state.sn_outer_residual;
          if (!std::isnan(step_sn_outer_residual) &&
              (std::isnan(solve_residual) || solve_residual > step_sn_outer_residual)) {
            step_sn_outer_residual = solve_residual;
          }
          step_sn_converged = step_sn_converged && state.sn_converged;
        }
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::FldSolve,
            diagnostics::RadialFourierStagePhase::After,
            t_stage + dt_stage);
        auto fld_substage_audit_records =
            radiation::drain_fld_substage_audit_records();
        if (!fld_substage_audit_records.empty()) {
          const auto audit_cycle = static_cast<std::uint64_t>(state.step + 1);
          for (auto& record : fld_substage_audit_records) {
            record.cycle = audit_cycle;
            record.t_s = t_stage + dt_stage;
            record.dt_cycle = dt;
          }
          if (part_info.rank == 0 && history_writer.enabled()) {
            history_writer.append_fld_substage_audit_batch(
                fld_substage_audit_records);
          }
        }
        if (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion) {
          record_mesh_attr_zero(
              diagnostics::mesh_attribution::MeshDeformSource::FLDEnergyCoupling);
          record_mesh_attr_zero(
              diagnostics::mesh_attribution::MeshDeformSource::FLDRadiationPressure);
        } else if (cfg.radiation.mode == core::RadiationMode::SnTransport) {
          record_mesh_attr_zero(
              diagnostics::mesh_attribution::MeshDeformSource::SNTransport);
        }
        mark_rad_subphase(rad_imc_ms);
        const auto& holo_lo = imc.last_holo_lo_result();
        step_holo_boundary_in += holo_lo.boundary_E_in;
        step_holo_boundary_out += holo_lo.boundary_E_out;
        step_holo_matter_delta += holo_lo.matter_delta;
        step_holo_source_balance_error += holo_lo.conservation_error;
        mark_rad_subphase(rad_other_ms);
        if (!deterministic_stage) {
          migrate_radiation_particles();
        }
        mark_rad_subphase(rad_migrate_ms);
        accumulate_device_flags(state.radiation_device_flags,
                                imc.last_device_error_flags());
        mark_rad_subphase(rad_dflags_ms);
        const double escaped_after = imc.escaped_energy_total();
        const double deterministic_escaped =
            (cfg.radiation.mode == core::RadiationMode::SnTransport)
                ? state.sn_escaped_step
                : state.fld_escaped_step;
        const double deterministic_marshak =
            (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion)
                ? state.fld_marshak_in_step
                : ((cfg.radiation.mode == core::RadiationMode::SnTransport)
                       ? state.sn_marshak_in_step
                       : 0.0);
        step_E_rad_esc += deterministic_stage
                              ? replicated_tally_share *
                                    std::max(deterministic_escaped, 0.0)
                              : std::max(escaped_after - escaped_before, 0.0);
        step_E_marshak_in += deterministic_stage
                                  ? replicated_tally_share *
                                        std::max(deterministic_marshak, 0.0)
                                  : std::max(imc.last_marshak_in_step(), 0.0);
        const double deterministic_volume_in =
            (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion)
                ? state.fld_volume_source_in_step
                : ((cfg.radiation.mode == core::RadiationMode::SnTransport)
                       ? state.sn_volume_source_in_step
                       : 0.0);
        step_E_volume_in += deterministic_stage
                                ? replicated_tally_share *
                                      std::max(deterministic_volume_in, 0.0)
                                : std::max(imc.last_volume_source_step(), 0.0);
        step_E_numerical_loss += std::max(imc.last_numerical_loss_step(), 0.0);
        double source_E_floor = 0.0;
        int source_clamp_count = 0;
        mark_rad_subphase(rad_other_ms);
        // Radiation source-term closure reads state.zbar (single shared Zbar field).
        if (!deterministic_stage) {
          emit_radial_fourier_audit(
              diagnostics::RadialFourierStageId::NewtonSource,
              diagnostics::RadialFourierStagePhase::Before,
              t_stage + dt_stage);
          step_E_numerical_loss += inject_radiation_source_terms(
              state, cfg, dt_stage, &source_E_floor, &source_clamp_count,
              &imc.last_sigma_R_max(), &eos_ctx);
          emit_radial_fourier_audit(
              diagnostics::RadialFourierStageId::NewtonSource,
              diagnostics::RadialFourierStagePhase::After,
              t_stage + dt_stage);
        }
        mark_rad_subphase(rad_inject_ms);
        step_E_floor += std::max(source_E_floor, 0.0);
        step_clamp_count += std::max(source_clamp_count, 0);
        mark_rad_subphase(rad_other_ms);
      };

// ---- history snapshot fill of the Monte Carlo, difference and HOLO diagnostics

      if (cfg.radiation.enabled) {
        snapshot.mc.ddmc_mode_count = imc.last_ddmc_mode_count();
        snapshot.mc.imc_mode_count = imc.last_imc_mode_count();
        snapshot.mc.n_total = imc.last_n_total();
        snapshot.mc.n_imc_particles = imc.last_n_imc_particles();
        snapshot.mc.n_ddmc_particles = imc.last_n_ddmc_particles();
        snapshot.mc.n_census = imc.last_n_census();
        snapshot.mc.n_absorbed = imc.last_n_absorbed();
        snapshot.mc.n_escaped = imc.last_n_escaped();
        snapshot.mc.n_leaked = imc.last_ddmc_to_imc_conversions();
        snapshot.mc.ddmc_fraction = imc.last_ddmc_fraction();
        snapshot.mc.weight_min = imc.last_weight_min();
        snapshot.mc.weight_mean = imc.last_weight_mean();
        snapshot.mc.weight_max = imc.last_weight_max();
        snapshot.mc.overshoot_count = imc.last_overshoot_count();
        snapshot.mc.overshoot_max = imc.last_overshoot_max();
        snapshot.mc.mmatrix_violations = imc.last_mmatrix_violations();
        snapshot.mc.mmatrix_fallback_count = imc.last_mmatrix_fallback_count();
        snapshot.mc.omega_below_threshold = imc.last_omega_below_threshold();
        snapshot.mc.interface_transitions =
            saturating_i64(imc.last_interface_transitions());
        snapshot.mc.interface_reflections =
            saturating_i64(imc.last_interface_reflections());
        snapshot.mc.conversion_prob_violations =
            saturating_i64(imc.last_conversion_prob_violations());
        snapshot.mc.ddmc_to_imc_conversions = imc.last_ddmc_to_imc_conversions();
        snapshot.mc.rad_momentum_deposition = imc.last_rad_momentum_deposition();
        const auto& difference_ref = imc.last_reference_field_diagnostics();
        snapshot.mc.difference_reference_valid = difference_ref.valid ? 1 : 0;
        snapshot.mc.difference_eligible_cells = difference_ref.eligible_cells;
        snapshot.mc.difference_active_cells = difference_ref.active_cells;
        snapshot.mc.difference_strong_cells = difference_ref.strong_cells;
        snapshot.mc.difference_hybrid_suppressed_cells =
            difference_ref.hybrid_suppressed_cells;
        snapshot.mc.difference_W_min = difference_ref.W_min;
        snapshot.mc.difference_W_mean = difference_ref.W_mean;
        snapshot.mc.difference_W_max = difference_ref.W_max;
        snapshot.mc.difference_tau_min = difference_ref.tau_min;
        snapshot.mc.difference_tau_mean = difference_ref.tau_mean;
        snapshot.mc.difference_tau_max = difference_ref.tau_max;
        snapshot.mc.difference_chi_mean = difference_ref.chi_mean;
        snapshot.mc.difference_chi_max = difference_ref.chi_max;
        snapshot.mc.difference_reduced_flux_max = difference_ref.reduced_flux_max;
        snapshot.mc.difference_knudsen_max = difference_ref.knudsen_max;
        snapshot.mc.difference_front_grad_Te_max = difference_ref.front_grad_Te_max;
        snapshot.mc.difference_front_grad_rho_max = difference_ref.front_grad_rho_max;
        snapshot.mc.difference_E_ref_total = difference_ref.E_ref_total;
        const auto& holo = imc.last_holo_selector_diagnostics();
        snapshot.mc.holo_n_core_cells = holo.n_core_cells;
        snapshot.mc.holo_n_entered = holo.n_entered;
        snapshot.mc.holo_n_exited = holo.n_exited;
        snapshot.mc.holo_n_hard_exited = holo.n_hard_exited;
        snapshot.mc.holo_n_island_rejected = holo.n_island_rejected;
        snapshot.mc.holo_tau_R_min = holo.tau_R_min;
        snapshot.mc.holo_tau_R_max = holo.tau_R_max;
        snapshot.mc.holo_reduced_flux_max = holo.reduced_flux_max;
        snapshot.mc.holo_E_LO_total = compute_holo_E_LO_total(state, cfg);
        snapshot.mc.holo_E_LO_boundary_in = step_holo_boundary_in;
        snapshot.mc.holo_E_LO_boundary_out = step_holo_boundary_out;
        snapshot.mc.holo_matter_delta = step_holo_matter_delta;
        snapshot.mc.holo_source_balance_error = step_holo_source_balance_error;
        snapshot.mc.holo_particle_net_source_core =
            compute_holo_particle_net_source_core(state, cfg);
        snapshot.mc.holo_lo_particle_source_mismatch =
            step_holo_matter_delta - snapshot.mc.holo_particle_net_source_core;
        const HoloPrrSummary holo_prr = compute_holo_prr_summary(state, cfg);
        snapshot.mc.holo_Prr_coverage = holo_prr.coverage;
        snapshot.mc.holo_chi_min = holo_prr.chi_min;
        snapshot.mc.holo_chi_mean = holo_prr.chi_mean;
        snapshot.mc.holo_chi_max = holo_prr.chi_max;
      }

// ---- accumulate_device_flags (the Monte Carlo transport device flags)

void accumulate_device_flags(core::DeviceErrorFlags& acc,
                             const core::DeviceErrorFlags& step) {
  acc.nan_particle = std::max(acc.nan_particle, step.nan_particle);
  acc.invalid_cell = std::max(acc.invalid_cell, step.invalid_cell);
  acc.invalid_boundary = std::max(acc.invalid_boundary, step.invalid_boundary);
  acc.pool_overflow = std::max(acc.pool_overflow, step.pool_overflow);
  acc.opacity_out_of_range = std::max(acc.opacity_out_of_range,
                                      step.opacity_out_of_range);
  if (step.infinite_loop > 0) {
    const auto acc_i64 = static_cast<std::int64_t>(acc.infinite_loop);
    const auto step_i64 = static_cast<std::int64_t>(step.infinite_loop);
    const auto sum_i64 = acc_i64 + step_i64;
    const auto cap_i64 = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
    if (sum_i64 > cap_i64) {
      static bool warned = false;
      if (!warned) {
        core::log_warning("DeviceErrorFlags: infinite_loop counter saturated at INT32_MAX");
        warned = true;
      }
    }
    acc.infinite_loop = static_cast<std::int32_t>(std::min(sum_i64, cap_i64));
  }
  acc.ddmc_sigma_tot_zero = std::max(acc.ddmc_sigma_tot_zero,
                                     step.ddmc_sigma_tot_zero);
  acc.roulette_kill = std::max(acc.roulette_kill, step.roulette_kill);
}

}  // namespace
}  // namespace tenryu::coupling
