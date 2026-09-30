#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "parallel/partition.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::parallel {
struct CommBuffers;
}

namespace tenryu::radiation {

// The radiation operator of a run: Radiation.mode = "multigroup_diffusion" (FLD, NUMERICS §6.7) or "sn_transport"
// (S_N, §6.8), in 1D or 2D_RZ. It keeps the Planck-fraction table of the run's groups and calls the solver of the
// mode and dimension. Until 2026-09-29 this entry was IMC::transport_step, the step of the Monte Carlo radiation
// class, which dispatched the deterministic modes before its own; the Monte Carlo radiation (IMC, DDMC, random walk,
// HOLO, difference formulation) was frozen and is kept under retired/, outside the build.
class RadiationStep {
 public:
  // drive_time_s: evaluation time of time-dependent boundary drives in the 1D FLD / S_N solves (midpoint of the
  // advanced interval); NaN = state.t.
  void step(core::State& state,
            const core::Config& cfg,
            double dt,
            const parallel::PartitionInfo& part = parallel::PartitionInfo{},
            parallel::CommBuffers* bufs = nullptr,
            double drive_time_s = std::numeric_limits<double>::quiet_NaN());

  // The maximum-principle overshoot of the radiation phase (NUMERICS §11.8), computed by the driver after the phase
  // and kept here for the history (radiation/overshoot_count, radiation/overshoot_max) and
  // Numerics.safety.overshoot_warn. Every step() call resets it to zero.
  void set_last_overshoot_metrics(std::int64_t count, double max_ratio);
  [[nodiscard]] std::int64_t last_overshoot_count() const { return last_overshoot_count_; }
  [[nodiscard]] double last_overshoot_max() const { return last_overshoot_max_; }

 private:
  PlanckTable planck_cache_;
  std::vector<double> planck_cache_bounds_eV_;
  int planck_cache_n_groups_ = 0;
  int planck_cache_n_T_ = 0;
  double planck_cache_T_min_eV_ = 0.0;
  double planck_cache_T_max_eV_ = 0.0;
  bool planck_cache_valid_ = false;
  std::int64_t last_overshoot_count_ = 0;
  double last_overshoot_max_ = 0.0;
};

}  // namespace tenryu::radiation
