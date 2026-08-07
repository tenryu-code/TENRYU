#pragma once

#include <cstdint>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/cap_energy_audit.hpp"

namespace tenryu::hydro {

inline bool corner_mass_lagrangian_invariant_enabled(const core::Config& cfg) {
  return cfg.numerics.hydro.subzonal_pressure_enabled ||
         cfg.numerics.hydro.subzonal_mass_lagrangian_invariant_enabled ||
         cfg.numerics.hydro.subzonal_mass_enabled ||
         cfg.numerics.hydro.hourglass.enabled;
}

inline bool subzonal_hourglass_enabled(const core::Config& cfg) {
  return !cfg.numerics.hydro.subzonal_pressure_enabled &&
         (cfg.numerics.hydro.subzonal_mass_enabled ||
          cfg.numerics.hydro.hourglass.enabled);
}

inline double anti_hourglass_scale(const core::Config& cfg) {
  return cfg.numerics.hydro.subzonal_mass_enabled
             ? cfg.numerics.hydro.anti_hourglass_kappa
             : cfg.numerics.hydro.hourglass.scale;
}

void ensure_hourglass_subzonal_masses_2d(core::State& state,
                                         const core::Config& cfg,
                                         bool force_recompute);

const double* ensure_bridge_band_subzonal_weights_2d(
    core::State& state,
    const core::Config& cfg);

void compute_anti_hourglass_acceleration_2d(
    core::NodeField1D& accel_r_hg,
    core::NodeField1D& accel_z_hg,
    core::CellField1D* compatible_work_erg,
    const core::State& state,
    const core::Config& cfg,
    const core::CellField1D& cell_pressure,
    const core::NodeField1D& node_mass,
    const core::NodeField1D& pressure_accel_r,
    const core::NodeField1D& pressure_accel_z,
    const std::int8_t* d_hydro_active,
    double dt,
    bool compute_compatible_work,
    const std::uint8_t* d_node_flags = nullptr,
    const double* work_velocity_r = nullptr,
    const double* work_velocity_z = nullptr,
    double compatible_work_dt = -1.0);

void add_hourglass_acceleration_2d(core::NodeField1D& accel_r,
                                   core::NodeField1D& accel_z,
                                   const core::NodeField1D& accel_r_hg,
                                   const core::NodeField1D& accel_z_hg);

void apply_hourglass_work_2d(core::State& state,
                             const core::Config& cfg,
                             const core::CellField1D& compatible_work_erg,
                             bool use_two_temp,
                             const std::int8_t* d_hydro_active,
                             double* E_floor_injected,
                             int* clamp_count);

I1BSpuriousSensorSummary compute_hydro_nonaffine_ke_sensor_2d(
    const core::State& state,
    const std::int8_t* d_hydro_active,
    int top_k);

}  // namespace tenryu::hydro
