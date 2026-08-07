#pragma once

#include <cstdint>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::central_pseudo_core {

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
// Terminal absorption (I1-B-R endgame, env TENRYU_I1B_TERMINAL_ABSORB):
// armed by request_ring_absorption when the rebound-phase emergency walk
// requests the structurally unabsorbable LAST shell row. The driver checks
// the pending flag at the top of every step attempt, executes the terminal
// absorption on the restored pre-step state (ALL remaining active cells
// become stratified 1D shells; the 2D mesh state is frozen), then integrates
// the core1d-only tail to t_end via run_terminal_core1d_tail.
bool terminal_absorb_pending(const core::State& state);
void execute_terminal_absorption(core::State& state, const core::Config& cfg);
// Integrate the post-terminal core1d rebound to t_end. p_drive evaluates the
// external drive pressure [dyn/cm^2] at a given time (zero after drive-off).
// Advances state.t / state.step; returns false only if the sub-model stalls
// (no substep progress), which the caller must treat as a hard failure.
bool run_terminal_core1d_tail(core::State& state,
                              const core::Config& cfg,
                              double (*p_drive)(const core::State&, double));

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

}  // namespace tenryu::hydro::central_pseudo_core
