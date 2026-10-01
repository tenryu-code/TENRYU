#include "drivers/cli.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>

#if TENRYU_ENABLE_MPI
#include <mpi.h>
#endif
#include <stdexcept>
#include <string>
#include <utility>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "coupling/driver.hpp"
#include "io/hdf5_reader.hpp"
#include "io/output_manager.hpp"
#include "laser/laser.cuh"
#include "materials/opacity_diagnostics.hpp"
#include "mesh/mesh.hpp"
#include "radiation/radiation_init.cuh"
#if TENRYU_ENABLE_PYTHON
#include "core/namelist/freeze.hpp"
#include "core/namelist/frozen_table.hpp"
#include "core/namelist/geometry_eval.hpp"
#include "core/namelist/runtime.hpp"
#include "core/namelist/errors.hpp"
#endif

namespace tenryu::drivers {

void validate_s2_multiblock_runtime_features(const tenryu::core::Config& cfg) {
  if (cfg.mesh.topology_scheme !=
      tenryu::core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL) {
    return;
  }

  const auto reject = [](const std::string& reason) {
    throw std::runtime_error(
        "multiblock_cart_core_polar_shell topology requires the hydro/ALE "
        "feature subset for S2/S4 runtime; " +
        reason +
        ". See SPECIFICATION §6.4.2 S2 runtime feature gate. "
        "Use topology_scheme=single_block for full feature set.");
  };

  if (cfg.main.dimension != "2D_RZ") {
    reject("only main.dimension=\"2D_RZ\" is supported in S2");
  }
  if (!cfg.numerics.hydro.enabled) {
    reject("numerics.hydro.enabled=false is not supported in S2");
  }
  if (cfg.mesh.motion != "lagrangian" && cfg.mesh.motion != "ale") {
    reject("only mesh.motion=\"lagrangian\" or \"ale\" is supported in S2/S4");
  }
  if (cfg.numerics.conduction.enabled) {
    reject(
        "numerics.conduction.enabled=true is not supported in S2 — pending S3+");
  }
  if (cfg.radiation.enabled) {
    reject("radiation.enabled=true is not supported in S2 — pending S3+");
  }
  if (cfg.laser.enabled) {
    reject("laser.enabled=true is not supported in S2 — pending S3+");
  }
  if (cfg.numerics.plic.enabled) {
    reject("numerics.plic.enabled=true is not supported in S2 — pending S3+");
  }
  if (cfg.numerics.hydro.boundary_2d.r_outer == "state_supply") {
    reject(
        "state-supply BC on the outer shell is not supported in S4-T6 "
        "— current state_supply support is z-face only");
  }
}

namespace {

std::string format_cli_error(const std::string& message) {
  return "TENRYU ERROR [namelist]: " + message;
}

void initialize_output_timing(tenryu::core::State& state,
                              const tenryu::core::Config& cfg) {
  state.t_next_plot = (cfg.output.plot_every_s > 0.0)
                          ? state.t
                          : -1.0;
  state.t_next_history = (cfg.output.history_every_s > 0.0)
                             ? (state.t + cfg.output.history_every_s)
                             : -1.0;
  state.t_next_checkpoint = (cfg.output.checkpoint_every_s > 0.0)
                                ? (state.t + cfg.output.checkpoint_every_s)
                                : -1.0;
}

#if TENRYU_ENABLE_PYTHON

using TableMap = std::map<std::string, tenryu::core::namelist::FrozenTable1D>;
namespace py = pybind11;

bool cast_numeric(const py::handle value, double* out) {
  if (py::isinstance<py::bool_>(value) || value.is_none()) {
    return false;
  }
  try {
    *out = py::cast<double>(value);
  } catch (...) {
    return false;
  }
  return true;
}

// The declared pulse energy in J (the waveform is power in W, so its
// integral is in J): total_energy_erg is converted, total_energy is taken
// as J. The erg attribute used to be compared with the integral in J.
std::optional<double> extract_declared_total_energy(const py::object& callable_obj,
                                                    const std::string& callable_path) {
  constexpr std::array<const char*, 2> kEnergyAttrs = {"total_energy_erg", "total_energy"};
  constexpr std::array<double, 2> kEnergyAttrToJoule = {1.0e-7, 1.0};
  for (std::size_t a = 0; a < kEnergyAttrs.size(); ++a) {
    const char* attr = kEnergyAttrs[a];
    if (!py::hasattr(callable_obj, attr)) {
      continue;
    }
    const py::object value = callable_obj.attr(attr);
    double energy = 0.0;
    if (!cast_numeric(value, &energy) || !std::isfinite(energy) || !(energy > 0.0)) {
      tenryu::core::log_warning("[TENRYU][laser] callable '" + callable_path + "' attribute '" +
                                std::string(attr) +
                                "' is not a finite positive number; skipping energy consistency check");
      return std::nullopt;
    }
    return energy * kEnergyAttrToJoule[a];
  }
  return std::nullopt;
}

void validate_laser_waveform_integral(const std::string& callable_path,
                                      const tenryu::core::namelist::FrozenTable1D& table,
                                      const std::optional<double> declared_total_energy) {
  const auto summary = tenryu::core::namelist::summarize_frozen_table(table);
  const double integral = summary.integrated_value;
  if (!std::isfinite(integral)) {
    tenryu::core::log_warning("[TENRYU][laser] callable '" + callable_path +
                              "' waveform integral is non-finite");
    return;
  }

  if (declared_total_energy.has_value()) {
    const double declared = *declared_total_energy;
    const double rel_err =
        std::abs(integral - declared) / std::max(std::abs(declared), 1.0e-30);
    if (rel_err > 0.10) {
      tenryu::core::log_warning("[TENRYU][laser] callable '" + callable_path +
                                "' waveform integral mismatch: integral=" +
                                std::to_string(integral) +
                                " J, declared_total_energy=" + std::to_string(declared) +
                                " J" +
                                ", rel_err=" + std::to_string(rel_err) +
                                " (> 0.10)");
    }
    return;
  }

  if (!(integral > 0.0)) {
    tenryu::core::log_warning("[TENRYU][laser] callable '" + callable_path +
                              "' waveform integral is not positive: integral=" +
                              std::to_string(integral));
  }
}

std::string format_time(const double t_s) {
  std::ostringstream oss;
  oss << std::setprecision(12) << t_s;
  return oss.str();
}

// --t-end / --max-steps (restarts only): Main.t_end and Main.max_steps of the continued run. The deck stays unchanged,
// and cfg.meta.frozen_config_json was frozen from it before this point, so the checkpoint comparison and the frozen
// configuration of the checkpoints this run writes keep the deck's values (as for --output-dir); the frozen JSON file
// in config/ and run_info.json record the values in effect.
void apply_run_control_overrides(tenryu::core::Config& cfg,
                                 const RunControlOverrides& run_control,
                                 const bool is_restart) {
  if (!run_control.t_end.has_value() && !run_control.max_steps.has_value()) {
    return;
  }
  if (!is_restart) {
    throw tenryu::core::namelist::ConfigError(
        "--t-end and --max-steps apply to restarts only (--restart or Main.restart_from): a fresh run takes "
        "Main.t_end and Main.max_steps from the deck");
  }
  if (run_control.t_end.has_value()) {
    const double t_end = *run_control.t_end;
    if (!(std::isfinite(t_end) && t_end > 0.0)) {
      throw tenryu::core::namelist::ConfigError("--t-end must be a finite time > 0 s, got " + format_time(t_end));
    }
    tenryu::core::log_info("[TENRYU] Main.t_end of the continued run set by --t-end: " +
                           format_time(cfg.main.t_end) + " s in the deck -> " + format_time(t_end) + " s");
    cfg.main.t_end = t_end;
  }
  if (run_control.max_steps.has_value()) {
    const int max_steps = *run_control.max_steps;
    constexpr int kUpper = tenryu::core::Config::MainConfig::kMaxStepsUpperBound;
    if (max_steps < 1 || max_steps > kUpper) {
      throw tenryu::core::namelist::ConfigError("--max-steps must be in [1, " + std::to_string(kUpper) +
                                                "], got " + std::to_string(max_steps));
    }
    tenryu::core::log_info("[TENRYU] Main.max_steps of the continued run set by --max-steps: " +
                           std::to_string(cfg.main.max_steps) + " in the deck -> " + std::to_string(max_steps));
    cfg.main.max_steps = max_steps;
  }
}

// A restart that would stop before its first step (the checkpoint at or after the end time, or at the step limit) is
// refused with the option that continues it, instead of ending without a step.
void require_steps_left(const tenryu::core::Config& cfg,
                        const RunControlOverrides& run_control,
                        const tenryu::core::State& checkpoint_state) {
  const double t = checkpoint_state.t;
  const double t_end = cfg.main.t_end;
  // The driver's own end-of-run test (driver.cpp): within 1e-14 relative of t_end counts as reached.
  if ((t_end - t) <= 1.0e-14 * std::max(std::abs(t), std::abs(t_end))) {
    throw tenryu::core::namelist::ConfigError(
        "the checkpoint is at t=" + format_time(t) + " s, at or after the end time " + format_time(t_end) + " s (" +
        (run_control.t_end.has_value() ? std::string("--t-end") : std::string("Main.t_end of the deck")) +
        "): pass --t-end <later time> to continue the run");
  }
  if (checkpoint_state.step >= cfg.main.max_steps) {
    throw tenryu::core::namelist::ConfigError(
        "the checkpoint is at step " + std::to_string(checkpoint_state.step) + ", at or beyond the step limit " +
        std::to_string(cfg.main.max_steps) + " (" +
        (run_control.max_steps.has_value() ? std::string("--max-steps") : std::string("Main.max_steps of the deck")) +
        "): pass --max-steps <larger limit> to continue the run");
  }
}

std::optional<tenryu::core::namelist::FrozenTable1D> maybe_make_time_table(
    const tenryu::core::namelist::Builder& builder,
    const std::string& callable_path,
    const double t_end,
    const bool zero_outside,
    const bool require_non_negative = false) {
  const auto it = builder.callable_objects.find(callable_path);
  if (it == builder.callable_objects.end()) {
    return std::nullopt;
  }

  auto table = tenryu::core::namelist::create_frozen_time_table(
      it->second, t_end, callable_path, require_non_negative);
  table.zero_outside = zero_outside;
  return table;
}

// The time tables span the run: [0, Main.t_end]. energy_window_t_end is the end of the window over which
// LaserBeam.energy_J normalizes the beam power: the deck's Main.t_end, also for a run whose end --t-end moved, so that
// run keeps the power history of the run that wrote the checkpoint (SPECIFICATION §7.4). The tables then span the
// longer of the two, so the window is always inside them (an earlier --t-end would otherwise cut the window short and
// change the normalization), and their samples on the shared range are those of the original run's tables while the
// table step is unchanged.
TableMap build_frozen_tables(const tenryu::core::Config& cfg,
                             const tenryu::core::namelist::Builder& builder,
                             const double energy_window_t_end) {
  TableMap tables;
  const double table_t_end = std::max(cfg.main.t_end, energy_window_t_end);
  const bool extended = cfg.main.t_end > energy_window_t_end;
  if (tenryu::core::namelist::frozen_time_table_step(table_t_end) !=
      tenryu::core::namelist::frozen_time_table_step(energy_window_t_end)) {
    tenryu::core::log_warning(
        "[TENRYU] --t-end " + format_time(cfg.main.t_end) +
        " s makes the time tables (beam power, Marshak drive, boundary pressure, hot-electron eta) sample every " +
        format_time(tenryu::core::namelist::frozen_time_table_step(table_t_end)) + " s instead of " +
        format_time(tenryu::core::namelist::frozen_time_table_step(energy_window_t_end)) +
        " s (runs longer than about 0.95 us): the waveforms up to the deck's Main.t_end are sampled more coarsely "
        "than in the run that wrote the checkpoint");
  }

  for (std::size_t i = 0; i < cfg.laser.beams.size(); ++i) {
    const std::string path = "Laser.beams[" + std::to_string(i) + "].power";
    auto table = maybe_make_time_table(builder, path, table_t_end, true,
                                       /*require_non_negative=*/true);
    if (table.has_value()) {
      tenryu::core::namelist::normalize_beam_power_table(
          *table, energy_window_t_end, cfg.laser.beams[i].energy_J,
          ("Laser.beams[" + std::to_string(i) + "]").c_str());
      if (extended && cfg.laser.beams[i].energy_J > 0.0) {
        const double extra_J = tenryu::core::namelist::integrate_frozen_table(
            *table, energy_window_t_end, cfg.main.t_end);
        if (extra_J > 0.0) {
          std::ostringstream oss;
          oss << std::setprecision(6) << "[TENRYU][laser] Laser.beams[" << i
              << "]: the waveform is still on after the deck's Main.t_end=" << format_time(energy_window_t_end)
              << " s; energy_J=" << cfg.laser.beams[i].energy_J
              << " J normalizes it over [0, Main.t_end] only, so the power before that time is unchanged and the "
                 "continued run delivers another "
              << extra_J << " J up to --t-end=" << format_time(cfg.main.t_end) << " s";
          tenryu::core::log_warning(oss.str());
        }
      }
      std::optional<double> declared_total_energy;
      const auto callable_it = builder.callable_objects.find(path);
      if (callable_it != builder.callable_objects.end()) {
        declared_total_energy = extract_declared_total_energy(callable_it->second, path);
      }
      validate_laser_waveform_integral(path, *table, declared_total_energy);
      const std::string name =
          "laser.beams[" + std::to_string(i) + "].waveform";
      tables.emplace(name, *table);
    }
  }

  if (cfg.radiation.boundary.marshak_Tr.detected) {
    const auto table = maybe_make_time_table(
        builder, "Radiation.boundary.marshak_Tr", table_t_end, true,
        /*require_non_negative=*/true);
    if (table.has_value()) {
      tables.emplace("radiation.marshak_Tr", *table);
    }
  }

  if (cfg.laser.hot_electron.eta_hot_table.detected) {
    const auto table = maybe_make_time_table(
        builder, "Laser.hot_electron.eta_hot_table", table_t_end, true);
    if (table.has_value()) {
      tables.emplace("laser.hot_e_eta", *table);
    }
  }
  if (cfg.laser.hot_electron.sources_specified) {
    for (std::size_t si = 0; si < cfg.laser.hot_electron.sources.size(); ++si) {
      if (!cfg.laser.hot_electron.sources[si].eta_table.detected) {
        continue;
      }
      const auto channel_table = maybe_make_time_table(
          builder,
          "Laser.hot_electron.sources[" + std::to_string(si) + "].eta_table",
          table_t_end, true);
      if (channel_table.has_value()) {
        tables.emplace("laser.hot_e_eta_ch" + std::to_string(si), *channel_table);
      }
    }
  }

  for (const auto& [face, _] : cfg.radiation.boundary.marshak_Tr_map) {
    const std::string path = "Radiation.boundary.marshak_Tr_map." + face;
    const auto table = maybe_make_time_table(builder, path, table_t_end, true,
                                             /*require_non_negative=*/true);
    if (table.has_value()) {
      tables.emplace("radiation.marshak_Tr_map." + face, *table);
    }
  }

  if (cfg.numerics.hydro.pressure_drive_1d.detected) {
    const auto table = maybe_make_time_table(
        builder, "Numerics.hydro.boundary_pressure", table_t_end, true);
    if (table.has_value()) {
      tables.emplace("hydro.boundary_pressure", *table);
    }
  }

  return tables;
}

tenryu::core::namelist::FreezeExtras make_freeze_extras(
    const tenryu::core::namelist::GeometrySummary& geometry_summary,
    const TableMap& tables) {
  tenryu::core::namelist::FreezeExtras extras;

  tenryu::core::namelist::FreezeGeometrySummary geo;
  geo.has_rho = geometry_summary.rho.valid;
  geo.rho_min = geometry_summary.rho.min;
  geo.rho_max = geometry_summary.rho.max;
  geo.rho_mean = geometry_summary.rho.mean;

  geo.has_Te = geometry_summary.Te.valid;
  geo.Te_min = geometry_summary.Te.min;
  geo.Te_max = geometry_summary.Te.max;

  geo.has_Ti = geometry_summary.Ti.valid;
  geo.Ti_min = geometry_summary.Ti.min;
  geo.Ti_max = geometry_summary.Ti.max;

  geo.material_volume = geometry_summary.material_volume;
  extras.geometry = geo;

  for (const auto& [name, table] : tables) {
    const auto summary = tenryu::core::namelist::summarize_frozen_table(table);
    tenryu::core::namelist::FreezeTableSummary entry;
    entry.t_min = summary.t_min;
    entry.t_max = summary.t_max;
    entry.n_points = summary.n_points;
    entry.peak_value = summary.peak_value;
    entry.integrated_value = summary.integrated_value;
    extras.tables[name] = entry;
  }

  return extras;
}

#endif

}  // namespace

int cmd_run(const std::string& namelist_path,
            const std::string& restart_prefix,
            const std::string& output_dir_override,
            const RunControlOverrides& run_control) {
#if TENRYU_ENABLE_PYTHON
  try {
    tenryu::core::Config cfg;
    tenryu::core::State state;
    bool restarted_from_checkpoint = false;
    std::filesystem::path resolved_namelist_path;
    std::string case_name;
    std::string effective_restart_for_driver;
    std::string frozen_json;
    std::string mesh_requirement_json;
    std::optional<tenryu::core::namelist::FrozenTable1D> pressure_drive_1d;
    tenryu::io::PerMaterialCheckpointReadStatus per_material_checkpoint_status =
        tenryu::io::PerMaterialCheckpointReadStatus::MissingGroupDisabled;
    int initial_plic_interface_cells_observed = 0;

    {
      tenryu::core::namelist::PythonGuard python_guard;
      tenryu::core::namelist::Runtime runtime;
      runtime.execute(namelist_path);

      cfg = runtime.config();
      validate_s2_multiblock_runtime_features(cfg);
      if (!output_dir_override.empty()) {
        if (!restart_prefix.empty() || !cfg.main.restart_from.empty()) {
          // A restart writes to the deck's Output.directory and continues the file numbering found there (the
          // output manager resumes after the highest existing index). The message used to say that a restart
          // "continues its original output layout", which is true only when the original run also used the deck's
          // directory (2026-09-29).
          // The checkpoint's frozen configuration holds the deck's Output.directory (the override is applied
          // after it is frozen), so a run started with --output-dir restarts with the unchanged deck and writes to
          // the deck's directory; editing Output.directory in the deck would change the deck and refuse the restart
          // (the advice of this message until 2026-09-30).
          throw tenryu::core::namelist::ConfigError(
              "--output-dir applies to fresh runs only: a restarted run "
              "(--restart or Main.restart_from) writes to the deck's "
              "Output.directory (numbered _001, _002, ... when it exists), also "
              "when the run that wrote the checkpoint used --output-dir; restart "
              "with the unchanged deck");
        }
        cfg.output.directory = output_dir_override;
        core::log_info(
            "[TENRYU] Output.directory overridden by --output-dir: " +
            output_dir_override);
      }
      resolved_namelist_path = runtime.resolved_namelist_path();
      case_name = (cfg.main.name.empty() || cfg.main.name == "unnamed")
                      ? resolved_namelist_path.stem().string()
                      : cfg.main.name;

      const std::string effective_restart =
          restart_prefix.empty() ? cfg.main.restart_from : restart_prefix;
      const bool is_restart = !effective_restart.empty();
      const double deck_t_end = cfg.main.t_end;
      apply_run_control_overrides(cfg, run_control, is_restart);

      tenryu::core::namelist::GeometrySummary geometry_summary;
      if (is_restart) {
        tenryu::io::HDF5Reader reader;
        auto checkpoint = reader.read_checkpoint(cfg, effective_restart);
        require_steps_left(cfg, run_control, checkpoint.state);
        state = std::move(checkpoint.state);
        restarted_from_checkpoint = true;
        per_material_checkpoint_status = checkpoint.per_material_checkpoint_status;
        state.hydro_t_start_eV = runtime.builder().hydro_t_start_eV;
        effective_restart_for_driver = effective_restart;
        core::log_info("Restart loaded from: " + effective_restart);
      } else {
        per_material_checkpoint_status =
            cfg.numerics.materials.per_material_conservation_enabled
                ? tenryu::io::PerMaterialCheckpointReadStatus::MissingGroupEnabled
                : tenryu::io::PerMaterialCheckpointReadStatus::MissingGroupDisabled;
        state = tenryu::core::State::allocate(cfg, runtime.builder().hydro_t_start_eV);
        state.mesh = tenryu::mesh::create_mesh(cfg, state);
        state.vol = state.mesh.cell_vol;

        geometry_summary =
            tenryu::core::namelist::evaluate_geometry(cfg, runtime.builder(), state);
        initial_plic_interface_cells_observed =
            geometry_summary.interface_cells_observed;
        if (cfg.main.dim == 1) {
          bool mesh_requirement_violated = false;
          std::string mesh_requirement_violation;
          mesh_requirement_json = build_mesh_requirement_json_for_config(
              cfg, runtime.builder(), &mesh_requirement_violated,
              &mesh_requirement_violation);
          if (mesh_requirement_violated) {
            throw tenryu::core::namelist::ConfigError(
                mesh_requirement_violation);
          }
        }
        initialize_output_timing(state, cfg);
      }

      if (!is_restart) {
        tenryu::radiation::apply_initial_radiation_field(state, cfg);
      }

      const auto tables = build_frozen_tables(cfg, runtime.builder(), deck_t_end);
      state.laser_waveforms.assign(cfg.laser.beams.size(),
                                   tenryu::core::namelist::FrozenTable1D{});
      for (std::size_t i = 0; i < cfg.laser.beams.size(); ++i) {
        const std::string name = "laser.beams[" + std::to_string(i) + "].waveform";
        const auto it = tables.find(name);
        if (it != tables.end()) {
          state.laser_waveforms[i] = it->second;
          continue;
        }
        tenryu::core::namelist::FrozenTable1D fallback;
        fallback.x = {0.0, cfg.main.t_end};
        fallback.y = {0.0, 0.0};
        fallback.n_points = 2;
        fallback.x_min = 0.0;
        fallback.x_max = cfg.main.t_end;
        fallback.zero_outside = true;
        state.laser_waveforms[i] = std::move(fallback);
      }

      const auto pressure_table_it = tables.find("hydro.boundary_pressure");
      if (pressure_table_it != tables.end()) {
        pressure_drive_1d = pressure_table_it->second;
      }
      const auto marshak_table_it = tables.find("radiation.marshak_Tr");
      if (marshak_table_it != tables.end()) {
        state.marshak_Tr_1d = marshak_table_it->second;
      }
      const auto hot_e_eta_it = tables.find("laser.hot_e_eta");
      if (hot_e_eta_it != tables.end()) {
        state.hot_e_eta_1d = hot_e_eta_it->second;
      }
      if (cfg.laser.hot_electron.sources_specified) {
        state.hot_e_eta_ch_1d.assign(cfg.laser.hot_electron.sources.size(), std::nullopt);
        for (std::size_t si = 0; si < cfg.laser.hot_electron.sources.size(); ++si) {
          const auto channel_it = tables.find("laser.hot_e_eta_ch" + std::to_string(si));
          if (channel_it != tables.end()) {
            state.hot_e_eta_ch_1d[si] = channel_it->second;
          }
        }
      }
      state.marshak_Tr_face_tables.clear();
      for (const auto& [face, _] : cfg.radiation.boundary.marshak_Tr_map) {
        const std::string key = "radiation.marshak_Tr_map." + face;
        const auto face_it = tables.find(key);
        if (face_it != tables.end()) {
          state.marshak_Tr_face_tables[face] = face_it->second;
        }
      }

      const auto extras = make_freeze_extras(geometry_summary, tables);
      frozen_json = tenryu::core::namelist::Freeze::to_json(cfg, &extras);
    }

    state.pressure_drive_1d = pressure_drive_1d;

    // Under mpirun every rank executes this path: only rank 0 claims the
    // output tree and writes the run artifacts (M18e I/O consolidation —
    // the per-rank directory-index race produced run_XXX_001 siblings).
    int io_rank = 0;
#if TENRYU_ENABLE_MPI
    {
      int mpi_up = 0;
      MPI_Initialized(&mpi_up);
      if (mpi_up != 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &io_rank);
      }
    }
#endif
    tenryu::io::OutputManager out;
    out.init(cfg, io_rank);
    if (io_rank == 0) {
      tenryu::drivers::setup_file_logging(out.log_dir);
    }

    if (io_rank == 0 && cfg.output.save_namelist_copy) {
      const std::filesystem::path namelist_copy =
          std::filesystem::path(out.config_dir) / (case_name + "_namelist.py");
      std::filesystem::copy_file(resolved_namelist_path, namelist_copy,
                                 std::filesystem::copy_options::overwrite_existing);
    }

    if (io_rank == 0 && cfg.output.save_frozen_config) {
      out.write_frozen_config(case_name, frozen_json);
    }

    if (restarted_from_checkpoint) {
      laser::invalidate_global_skip_cache();
      state.laser_dep.fill(0.0);
    }

    if (io_rank == 0) {
      out.write_run_info(state, cfg);
      if (!mesh_requirement_json.empty()) {
        out.write_mesh_requirement(mesh_requirement_json);
      }
    }
    tenryu::materials::log_hard_xray_opacity_diagnostic(cfg);

    tenryu::coupling::Driver driver;
    driver.set_initial_plic_interface_cells_observed(
        initial_plic_interface_cells_observed);
    driver.set_checkpoint_per_material_status(per_material_checkpoint_status);
    if (!effective_restart_for_driver.empty()) {
      driver.set_restart_checkpoint_prefix(effective_restart_for_driver);
    }
    driver.run(state, cfg, out);

    std::cout << "[TENRYU] Run completed.\n";
    std::cout << "[TENRYU] Output directory: " << out.output_dir << "/\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << format_cli_error(e.what()) << '\n';
    return 1;
  }
#else
  (void)namelist_path;
  (void)restart_prefix;
  (void)output_dir_override;
  (void)run_control;
  tenryu::core::log_error("TENRYU was built without Python support (TENRYU_ENABLE_PYTHON=OFF)");
  return 1;
#endif
}

}  // namespace tenryu::drivers
