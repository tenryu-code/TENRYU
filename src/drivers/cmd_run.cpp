#include "drivers/cli.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <optional>

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

std::optional<double> extract_declared_total_energy(const py::object& callable_obj,
                                                    const std::string& callable_path) {
  constexpr std::array<const char*, 2> kEnergyAttrs = {"total_energy_erg", "total_energy"};
  for (const char* attr : kEnergyAttrs) {
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
    return energy;
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
                                ", declared_total_energy=" + std::to_string(declared) +
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

std::optional<tenryu::core::namelist::FrozenTable1D> maybe_make_time_table(
    const tenryu::core::namelist::Builder& builder,
    const std::string& callable_path,
    const double t_end,
    const bool zero_outside) {
  const auto it = builder.callable_objects.find(callable_path);
  if (it == builder.callable_objects.end()) {
    return std::nullopt;
  }

  constexpr int kSamples = 10000;
  auto table = tenryu::core::namelist::create_frozen_table(
      it->second, 0.0, t_end, kSamples);
  table.zero_outside = zero_outside;
  return table;
}

TableMap build_frozen_tables(const tenryu::core::Config& cfg,
                             const tenryu::core::namelist::Builder& builder) {
  TableMap tables;

  for (std::size_t i = 0; i < cfg.laser.beams.size(); ++i) {
    const std::string path = "Laser.beams[" + std::to_string(i) + "].power";
    const auto table = maybe_make_time_table(builder, path, cfg.main.t_end, true);
    if (table.has_value()) {
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
        builder, "Radiation.boundary.marshak_Tr", cfg.main.t_end, true);
    if (table.has_value()) {
      tables.emplace("radiation.marshak_Tr", *table);
    }
  }

  if (cfg.laser.hot_electron.eta_hot_table.detected) {
    const auto table = maybe_make_time_table(
        builder, "Laser.hot_electron.eta_hot_table", cfg.main.t_end, true);
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
          cfg.main.t_end, true);
      if (channel_table.has_value()) {
        tables.emplace("laser.hot_e_eta_ch" + std::to_string(si), *channel_table);
      }
    }
  }

  for (const auto& [face, _] : cfg.radiation.boundary.marshak_Tr_map) {
    const std::string path = "Radiation.boundary.marshak_Tr_map." + face;
    const auto table = maybe_make_time_table(builder, path, cfg.main.t_end, true);
    if (table.has_value()) {
      tables.emplace("radiation.marshak_Tr_map." + face, *table);
    }
  }

  if (cfg.laser.hot_electron.eta_hot_table.detected) {
    const auto table = maybe_make_time_table(
        builder, "Laser.hot_electron.eta_hot_table", cfg.main.t_end, true);
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
          cfg.main.t_end, true);
      if (channel_table.has_value()) {
        tables.emplace("laser.hot_e_eta_ch" + std::to_string(si), *channel_table);
      }
    }
  }

  if (cfg.numerics.hydro.pressure_drive_1d.detected) {
    const auto table = maybe_make_time_table(
        builder, "Numerics.hydro.boundary_pressure", cfg.main.t_end, true);
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
            const std::string& output_dir_override) {
#if TENRYU_ENABLE_PYTHON
  try {
    tenryu::core::Config cfg;
    tenryu::core::State state;
    std::optional<tenryu::radiation::PhotonPool> restart_pool;
    std::filesystem::path resolved_namelist_path;
    std::string case_name;
    std::string effective_restart_for_driver;
    std::string frozen_json;
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
          throw tenryu::core::namelist::ConfigError(
              "--output-dir applies to fresh runs only: a restarted run "
              "(--restart or Main.restart_from) continues its original "
              "output layout");
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

      tenryu::core::namelist::GeometrySummary geometry_summary;
      if (is_restart) {
        tenryu::io::HDF5Reader reader;
        auto checkpoint = reader.read_checkpoint(cfg, effective_restart);
        state = std::move(checkpoint.state);
        restart_pool = std::move(checkpoint.photon_pool);
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
        initialize_output_timing(state, cfg);
      }

      if (!is_restart) {
        tenryu::radiation::apply_initial_radiation_field(state, cfg);
      }

      const auto tables = build_frozen_tables(cfg, runtime.builder());
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

    if (restart_pool.has_value()) {
      laser::invalidate_global_skip_cache();
      state.laser_dep.fill(0.0);
    }

    if (io_rank == 0) {
      out.write_run_info(state, cfg);
    }
    tenryu::materials::log_hard_xray_opacity_diagnostic(cfg);

    tenryu::coupling::Driver driver;
    driver.set_initial_plic_interface_cells_observed(
        initial_plic_interface_cells_observed);
    driver.set_checkpoint_per_material_status(per_material_checkpoint_status);
    if (!effective_restart_for_driver.empty()) {
      driver.set_restart_checkpoint_prefix(effective_restart_for_driver);
    }
    if (restart_pool.has_value()) {
      driver.set_restart_photon_pool(std::move(*restart_pool));
      restart_pool.reset();
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
  tenryu::core::log_error("TENRYU was built without Python support (TENRYU_ENABLE_PYTHON=OFF)");
  return 1;
#endif
}

}  // namespace tenryu::drivers
