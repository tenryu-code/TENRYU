#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/mesh_transaction.hpp"

namespace tenryu::core {
struct State;
}

namespace tenryu::hydro {

struct HostReplayState {
  std::vector<double> hot_e_eps_cum_host;
  std::vector<double> burn_eps_cum_host;
  std::vector<double> mesh_cell_vol;
  std::vector<double> mesh_cell_area;
  std::vector<double> mesh_cell_centroid_r;
  std::vector<double> mesh_cell_centroid_z;
  std::vector<double> mesh_cell_Svec_r;
  std::vector<double> mesh_cell_Svec_z;
  bool corner_mass_initialized = false;
  bool corner_mass_is_lagrangian_invariant = false;
  bool gas_tracer_initialized = false;
  bool hllc_mom_z_cell_initialized = false;
  bool holo_ale_invalidated = false;
  bool ale_rezoned = false;
  int ale_rezone_invocations = 0;
  int ale_remaps_applied = 0;
  int ale_last_applied_step = -1;
  double dt_prev_hydro = -1.0;
  double state_supply_dM_cumulative = 0.0;
  double state_supply_dE_cumulative = 0.0;
  double state_supply_dPz_cumulative = 0.0;
  double state_supply_dM_step = 0.0;
  double state_supply_dE_step = 0.0;
  double state_supply_dPz_step = 0.0;
  bool plic_remap_sticky_fallback = false;
  int plic_consecutive_drift_triggers = 0;
  int plic_last_reconstruction_step = -1;
  std::vector<std::uint8_t> cell_is_void;
  std::vector<std::int8_t> hydro_active;
};

// Rollback adapter over core::ShadowTransaction used in snapshot mode:
// capture() = save the pre-attempt state; restore() = copy the saved bytes back onto the
// live device fields (the transaction's commit, with the shadow holding OLD data);
// accept() = drop the snapshot (attempt kept). Single-use per attempt, like the legacy
// detail::ReplayStateSnapshot in ale_driver.cu, which this adapter reproduces
// byte-for-byte on the device side (D2D arena instead of D2H/H2D round trips).
class RollbackGuard {
 public:
  void capture(core::State& state, cudaStream_t stream);
  // Re-queries every field pointer from `state` at call time (fields may have been
  // reallocated during the attempt); TENRYU_ASSERTs that every field's byte-size equals
  // the captured size (the shell replay path never resizes; fail loudly if it ever does).
  void restore(core::State& state, cudaStream_t stream);
  void accept();
  void telemetry_increment(const char* counter_name, std::uint64_t delta = 1);
  bool captured() const;

 private:
  core::ShadowTransaction tx_;
  std::vector<std::size_t> captured_bytes_;
  std::vector<void*> captured_ptrs_;
  HostReplayState host_;
};

}  // namespace tenryu::hydro
