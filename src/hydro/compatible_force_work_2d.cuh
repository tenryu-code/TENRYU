#pragma once

#include <cstddef>
#include <cstdint>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::compatible {

struct CompatibleForceBuffers {
  double* corner_p_r = nullptr;
  double* corner_p_z = nullptr;
  double* corner_sub_r = nullptr;
  double* corner_sub_z = nullptr;
  double* corner_q_r = nullptr;
  double* corner_q_z = nullptr;
  double* edge_av_r = nullptr;
  double* edge_av_z = nullptr;
  int n_cells = 0;
  int n_edges = 0;
};

struct CompatibleWorkBuffers {
  double* work_p = nullptr;
  double* work_sub = nullptr;
  double* work_av = nullptr;
  int n_cells = 0;
};

bool compatible_force_work_enabled(const core::Config& cfg);
bool compatible_force_work_buffers_requested(const core::Config& cfg);
std::size_t compatible_edge_force_count(const core::State& state,
                                        const core::Config& cfg);

void ensure_compatible_force_work_buffers(core::State& state,
                                          const core::Config& cfg,
                                          bool force_requested = false);
void zero_compatible_force_work_buffers(core::State& state);

void launch_assemble_pressure_corner_forces_2d(
    core::State& state,
    const core::CellField1D& cell_pressure,
    const core::CellField1D& svec_r,
    const core::CellField1D& svec_z,
    const std::int8_t* hydro_active,
    const core::Config& cfg);

void launch_assemble_scalar_av_corner_forces_2d(
    core::State& state,
    const core::CellField1D& cell_q,
    const core::CellField1D& svec_r,
    const core::CellField1D& svec_z,
    const std::int8_t* hydro_active);

void launch_sum_compatible_forces_to_nodes_2d(double* force_r,
                                              double* force_z,
                                              const std::int8_t* hydro_active,
                                              const core::State& state,
                                              const core::Config& cfg);

void launch_sum_scalar_av_corner_forces_to_nodes_2d(
    double* force_r,
    double* force_z,
    const std::int8_t* hydro_active,
    const core::State& state);

void launch_compute_compatible_work_2d(core::State& state,
                                       const core::Config& cfg,
                                       const double* v_r,
                                       const double* v_z,
                                       const std::int8_t* hydro_active);

void launch_compute_scalar_av_compatible_work_2d(
    core::State& state,
    const double* v_r,
    const double* v_z,
    const std::int8_t* hydro_active);

}  // namespace tenryu::hydro::compatible
