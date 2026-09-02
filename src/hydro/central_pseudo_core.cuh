#pragma once

#include <cstdint>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::central_pseudo_core {

namespace detail {

struct PlanarBoundaryPressureSegmentAreaVector {
  double r = 0.0;
  double z = 0.0;
};

__host__ __device__ inline PlanarBoundaryPressureSegmentAreaVector
planar_boundary_pressure_segment_area_vector(const double r_a,
                                             const double z_a,
                                             const double r_b,
                                             const double z_b) {
  PlanarBoundaryPressureSegmentAreaVector area{
      0.5 * (z_b - z_a), -0.5 * (r_b - r_a)};
  const double r_mid = 0.5 * (r_a + r_b);
  const double z_mid = 0.5 * (z_a + z_b);
  if (area.r * r_mid + area.z * z_mid < 0.0) {
    area.r = -area.r;
    area.z = -area.z;
  }
  return area;
}

}  // namespace detail

bool configured(const core::Config& cfg);
bool active(const core::State& state);

void ensure_built(core::State& state, const core::Config& cfg);
void maybe_absorb_first_active_ring(core::State& state,
                                    const core::Config& cfg);
// Arm a pending ring-absorption request for a failing active block-0 cell
// (ALE center-patch inadmissibility site). Returns true when the request was
// armed; the caller must then abort the step so the driver full-step retry
// restores the pre-step state and maybe_absorb_first_active_ring executes
// the absorption at the start of the retried attempt.
bool request_ring_absorption(core::State& state,
                             const core::Config& cfg,
                             int failing_cell);
// Arm absorption of the FIRST ACTIVE unit when the macro boundary itself is
// path-inadmissible (kCentralMacroCoreSentinelCell failure): the boundary
// moves outward onto healthier mesh. Same guards and retry contract as
// request_ring_absorption.
bool request_macro_boundary_absorption(core::State& state,
                                       const core::Config& cfg);
// Detect-gated stale-reference repair (env TENRYU_I1B_CORE_REF_REPAIR): if a
// core-halo cell's exact RZ polygon volume on the persistent reference has
// flipped sign or vanished vs the current mesh, sync the core-halo reference
// to current (prevents CSR swept-closure mass creation). Safe inside the
// axis-rezone temporary reference override: restore_persistent_reference()
// simply undoes the sync there (worst case: that one rezone becomes a no-op).
void detect_and_repair_stale_core_reference(core::State& state);
void aggregate_state(core::State& state,
                     const core::Config& cfg,
                     const char* phase,
                     bool emit_audit);
void rebuild_virtual_member_geometry(core::State& state,
                                     const core::Config& cfg,
                                     const char* phase);
void emit_phase_ledger(core::State& state,
                       const core::Config& cfg,
                       const char* phase,
                       const char* source_phase);

const std::uint8_t* member_mask_device(core::State& state,
                                       const core::Config& cfg);
const std::uint8_t* inactive_member_mask_device(core::State& state,
                                                const core::Config& cfg);
const std::uint8_t* active_cell_class_device(core::State& state,
                                             const core::Config& cfg);
const std::uint8_t* boundary_node_mask_device(core::State& state,
                                              const core::Config& cfg);
double acoustic_dt(core::State& state, const core::Config& cfg);

void zero_member_compatible_cell_buffers(core::State& state);
void zero_member_edge_av_forces(core::State& state);

void add_boundary_pressure_force(core::State& state,
                                 const core::Config& cfg,
                                 const core::CellField1D& cell_pressure,
                                 double* force_r,
                                 double* force_z,
                                 double impulse_dt,
                                 const char* diag_tag = "untagged");

void set_boundary_pressure_work(core::State& state,
                                const core::Config& cfg,
                                double dt,
                                const core::NodeField1D& node_mass,
                                const double* old_velocity_r,
                                const double* old_velocity_z,
                                const double* new_velocity_r,
                                const double* new_velocity_z);

void apply_boundary_pressure_work(core::State& state,
                                  const core::Config& cfg,
                                  bool use_two_temp);

}  // namespace tenryu::hydro::central_pseudo_core
