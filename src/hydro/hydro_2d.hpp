#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "coupling/hydro_step_result.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::coupling {
struct ProfileObservability;
}

namespace tenryu::hydro {

struct HydroEOSContext;
class MeshRegimeDeviceCache;

struct TrialVolumeCflResult {
  bool admissible = true;
  double min_trial_vol_ratio = 1.0;
  double suggested_dt = 0.0;
  int first_failing_cell = -1;
};

TrialVolumeCflResult check_trial_lagrangian_volume_cfl(
    const core::State& state,
    double dt,
    double floor_fraction,
    double shrink_fraction,
    const parallel::Reduction* reduction);

tenryu::coupling::HydroStepResult refresh_geometry_and_density(
    core::State& state,
    const core::Config& cfg,
    tenryu::coupling::HydroFailureStage stage,
    int* rho_clamp_count = nullptr,
    tenryu::coupling::ProfileObservability* observability = nullptr);

void validate_geometry_after_retry_restore(core::State& state,
                                            const core::Config& cfg);

double compute_node_r_momentum_2d(const core::State& state,
                                  const core::Config& cfg);

namespace detail {

void launch_compute_node_activity_2d_multiblock(
    std::uint8_t* node_active,
    const int* reverse_csr_node_offsets,
    int n_nodes_total);

void launch_compute_node_mass_2d_multiblock(
    double* node_mass,
    const double* corner_mass,
    const double* mass_cell,
    const double* x_r,
    const int* reverse_csr_node_offsets,
    const int* reverse_csr_node_cells,
    const int* reverse_csr_node_corners,
    int n_nodes_total,
    int equal_split_node_mass_mode);

void launch_apply_boundary_accel_constraints_multiblock(
    double* accel_r,
    double* accel_z,
    const double* x_r,
    const double* x_z,
    const std::uint8_t* node_flags,
    int n_nodes_total,
    int r_outer_type,
    int z_bottom_type,
    int z_top_type);

void launch_compute_cell_dVdt_2d(
    double* dVdt,
    const double* v_r,
    const double* v_z,
    const double* Svec_r,
    const double* Svec_z,
    const core::State& state,
    const core::Config& cfg);

void launch_compute_force_from_cells_2d(
    double* force_r,
    double* force_z,
    const double* cell_pq,
    const double* Svec_r,
    const double* Svec_z,
    const std::int8_t* hydro_active,
    const core::State& state,
    const core::Config& cfg);

}  // namespace detail

// Drive-work ledger (env TENRYU_I1B_DRIVE_WORK_LEDGER, default off):
// per COMMITTED step, accumulate the outer-boundary drive's work on the
// mesh, W_step = sum_n f_n(P_drive(t_mid)) . v_n * dt, using the same
// force assembly as the shell_drive_sym diagnostic. Prints [drive_work]
// every TENRYU_I1B_DRIVE_WORK_LEDGER_EVERY steps (default 200). Call ONLY
// from the driver's committed-step epilogue (retries would double-count).
void run_drive_geometry_audit(const core::State& state,
                              const core::Config& cfg);
void suppress_tangential_velocity(core::State& state, double t);
void accumulate_drive_work_ledger(const core::State& state,
                                  const core::Config& cfg,
                                  double dt,
                                  double t_mid);

class Hydro2D {
 public:
  void prepare_initial_sound_speed(core::State& state,
                                   const core::Config& cfg,
                                   const HydroEOSContext* eos_ctx = nullptr) const;

  // Initialize the subzonal corner masses (and thus the compatible nodal
  // mass basis) BEFORE the initial snapshot/history record. Without this
  // the step-0 outputs use a different nodal-mass basis than every
  // subsequent step (corner masses initialize inside the first Lagrangian
  // step), which writes no hydro/node_mass at step 0 and bakes a fake
  // one-time offset into the history energy series.
  void prepare_initial_corner_mass(core::State& state,
                                   const core::Config& cfg) const;

  void apply_state_supply_boundary(core::State& state,
                                   const core::Config& cfg,
                                   const HydroEOSContext* eos_ctx = nullptr,
                                   bool accumulate_tally = true) const;

  tenryu::coupling::HydroStepResult lagrangian_step(
      core::State& state,
      double dt,
      const core::Config& cfg,
      const HydroEOSContext* eos_ctx = nullptr) const {
    return lagrangian_step(state, dt, cfg, state.t, nullptr, nullptr, nullptr,
                           parallel::PartitionInfo{}, nullptr, nullptr, "unknown",
                           eos_ctx, nullptr, nullptr);
  }

  tenryu::coupling::HydroStepResult lagrangian_step(
      core::State& state,
      double dt,
      const core::Config& cfg,
      double t_op,
      double* E_floor_injected = nullptr,
      int* clamp_count = nullptr,
      int* rho_clamp_count = nullptr,
      const parallel::PartitionInfo& part = parallel::PartitionInfo{},
      parallel::CommBuffers* bufs = nullptr,
      cudaStream_t stream = nullptr,
      const char* hydro_half = "unknown",
      const HydroEOSContext* eos_ctx = nullptr,
      const parallel::Reduction* reduction = nullptr,
      MeshRegimeDeviceCache* mesh_regime_cache = nullptr,
      tenryu::coupling::ProfileObservability* observability = nullptr,
      double* r_momentum_source_impulse = nullptr) const;
};

}  // namespace tenryu::hydro
