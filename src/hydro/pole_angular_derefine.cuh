#pragma once

#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/pole_angular_coarsen.cuh"

namespace tenryu::hydro::pole_angular_derefine {

bool configured(const core::Config& cfg);
bool assertions_enabled();
bool active(const core::State& state);

void ensure_built(core::State& state, const core::Config& cfg);
void aggregate_state(core::State& state,
                     const core::Config& cfg,
                     const char* phase,
                     bool emit_audit);
bool maintain_pole_spans(core::State& state,
                         const core::Config& cfg,
                         const char* phase);

const std::uint8_t* member_mask_device(core::State& state,
                                       const core::Config& cfg);
const std::uint8_t* inactive_member_mask_device(core::State& state,
                                                const core::Config& cfg);
const std::uint8_t* boundary_node_mask_device(core::State& state,
                                              const core::Config& cfg);
const std::uint8_t* combined_inactive_mask_device(
    core::State& state,
    const core::Config& cfg,
    core::DeviceArray<std::uint8_t>& scratch);
const std::uint8_t* combined_inactive_mask_device(
    core::State& state,
    core::DeviceArray<std::uint8_t>& scratch);
void refresh_geometry_exempt_cells(core::State& state);

std::vector<std::int8_t> apply_inactive_to_active_mask(
    const core::State& state,
    const std::vector<std::int8_t>* base_active = nullptr);

tenryu::hydro::pole_angular_coarsen::Overlay path_overlay(
    core::State& state,
    const core::Config& cfg);
bool repair_macro_boundary(core::State& state,
                           const core::Config& cfg,
                           int local_i_begin,
                           int local_j_begin,
                           int local_j_end);
bool extend_for_adjacent_active_child(core::State& state,
                                      const core::Config& cfg,
                                      int failing_cell,
                                      bool allow_next_span);

double acoustic_dt(core::State& state, const core::Config& cfg);
void assert_active_cell(const core::State& state, int cell, const char* op);

void zero_member_compatible_cell_buffers(core::State& state);
void zero_member_edge_av_forces(core::State& state);

void add_boundary_pressure_force(core::State& state,
                                 const core::Config& cfg,
                                 const core::CellField1D& cell_pressure,
                                 double* force_r,
                                 double* force_z,
                                 double impulse_dt);

void set_boundary_pressure_work(core::State& state,
                                const core::Config& cfg,
                                double dt,
                                const double* old_velocity_r,
                                const double* old_velocity_z,
                                const double* new_velocity_r,
                                const double* new_velocity_z);

void apply_boundary_pressure_work(core::State& state,
                                  const core::Config& cfg,
                                  bool use_two_temp);

}  // namespace tenryu::hydro::pole_angular_derefine
