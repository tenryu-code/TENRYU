#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/field.hpp"
#include "materials/eos_device_table.cuh"
#include "materials/eos_cell_table_selector.cuh"
#include "materials/eos_table.hpp"

namespace tenryu::core {
struct Config;
struct State;
}
namespace tenryu::hydro {
struct HydroEOSContext;
}

namespace tenryu::coupling {

struct DriverRecloseContext {
  ~DriverRecloseContext();
  DriverRecloseContext() = default;
  DriverRecloseContext(const DriverRecloseContext&) = delete;
  DriverRecloseContext& operator=(const DriverRecloseContext&) = delete;

  std::uint8_t* d_cell_is_void = nullptr;
  std::size_t cell_is_void_capacity = 0;
  std::vector<std::uint8_t> cached_cell_is_void;
  core::DeviceArray<double> conduction_Te_old;
  core::DeviceArray<double> conduction_cv_old;
  // Per-material table temperature ceilings (T_grid_eV.back(); 0 when the
  // material has no table), indexed by the Config material slot. Uploaded
  // once from Config so the per-cell tail extension uses each cell's own
  // table ceiling (multi-material closure, 2026-09-14).
  core::DeviceArray<double> electron_T_top_by_material;
  core::DeviceArray<double> ion_T_top_by_material;
  core::DeviceArray<double> total_T_top_by_material;
  // Identity of the Config material tables the ceilings were uploaded for
  // (one entry per material slot; nullptr for materials without tables).
  std::vector<const materials::EOSTableTriplet*> material_T_top_keys;
};

void capture_conduction_fields_device(DriverRecloseContext& context, const core::State& state);

bool sync_ee_from_Te_device(core::State& state, const core::Config& cfg,
                            const hydro::HydroEOSContext& eos_context,
                            DriverRecloseContext& context);

bool apply_conduction_energy_increment_device(
    core::State& state, const core::Config& cfg,
    const hydro::HydroEOSContext& eos_context, DriverRecloseContext& context);

void launch_tabular_eos_reclose(
    DriverRecloseContext& context,
    materials::DeviceEOSTableView electron_table,
    materials::DeviceEOSTableView ion_table,
    const std::vector<std::uint8_t>& cell_is_void,
    std::size_t n_cells,
    const double* rho,
    double* ee,
    double* ei,
    double* Te,
    double* Ti,
    double* Pe,
    double* Pi,
    double* cv_e,
    double* cv_i,
    double te_floor,
    double ti_floor);

}  // namespace tenryu::coupling
