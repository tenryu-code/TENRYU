#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "coupling/hydro_step_result.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"

namespace tenryu::hydro {

struct HydroEOSContext;

void filter_pq_checkerboard_1d(core::CellField1D& pq_filtered,
                               const core::CellField1D& pq_raw,
                               const core::CellField1D& rho,
                               const core::CellField1D& Qvisc);

void compute_odd_even_weight_1d(core::CellField1D& W_oe,
                                const core::CellField1D& rho);

class Hydro1D {
 public:
  void prepare_initial_sound_speed(core::State& state,
                                   const core::Config& cfg,
                                   const HydroEOSContext* eos_ctx = nullptr) const;

  tenryu::coupling::HydroStepResult lagrangian_step(
      core::State& state,
      double dt,
      const core::Config& cfg,
      const HydroEOSContext* eos_ctx = nullptr) const {
    return lagrangian_step(state, dt, cfg, state.t, nullptr, nullptr, nullptr,
                           parallel::PartitionInfo{}, nullptr, nullptr, eos_ctx);
  }

  // Returns a soft-failure result (retry_required, reason
  // "non_positive_volume_1d") when the Lagrangian motion produced a
  // non-positive cell volume and Numerics.hydro.driver_full_step_retry_enabled
  // is set; the driver restores the step snapshot and retries at dt/2.
  // Without the retry opt-in the historic hard assert fires here (earlier and
  // with better context than the downstream laser/diagnostics geometry
  // recompute that used to catch it).
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
      const HydroEOSContext* eos_ctx = nullptr,
      const std::vector<double>* sigma_R_max = nullptr,
      const core::CellField1D* p_extra = nullptr,
      core::CellField1D* p_extra_work = nullptr) const;

  // Persistent per-step scratch (perf lane B 2026-07-31): avoids per-step cudaMalloc/free churn.
  mutable core::CellField1D cs_pingpong_;
  mutable core::CellField1D h_half_pingpong_;
  mutable core::CellField1D heat_oe_half_pingpong_;
  mutable core::CellField1D q_probe_persist_;
  mutable core::CellField1D div_u_probe_persist_;
};

}  // namespace tenryu::hydro
