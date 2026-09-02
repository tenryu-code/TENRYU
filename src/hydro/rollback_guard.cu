#include "hydro/rollback_guard.hpp"

#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {
namespace {

std::vector<core::TransactionBufferDesc> replay_device_descs(core::State& s) {
#define TENRYU_REPLAY_DESC(member) \
  {#member, s.member.data(), s.member.size() * sizeof(double)}
  return {
      TENRYU_REPLAY_DESC(rho),
      TENRYU_REPLAY_DESC(mass),
      TENRYU_REPLAY_DESC(corner_mass),
      TENRYU_REPLAY_DESC(corner_volume),
      TENRYU_REPLAY_DESC(corner_pressure),
      TENRYU_REPLAY_DESC(subzonal_mass_corner0),
      TENRYU_REPLAY_DESC(subzonal_mass_corner1),
      TENRYU_REPLAY_DESC(subzonal_mass_corner2),
      TENRYU_REPLAY_DESC(subzonal_mass_corner3),
      TENRYU_REPLAY_DESC(vol),
      TENRYU_REPLAY_DESC(cell_vol_initial),
      TENRYU_REPLAY_DESC(vol_prev_hydro),
      TENRYU_REPLAY_DESC(zbar),
      TENRYU_REPLAY_DESC(Te),
      TENRYU_REPLAY_DESC(Ti),
      TENRYU_REPLAY_DESC(ee),
      TENRYU_REPLAY_DESC(ei),
      TENRYU_REPLAY_DESC(Pe),
      TENRYU_REPLAY_DESC(Pi),
      TENRYU_REPLAY_DESC(Qvisc),
      TENRYU_REPLAY_DESC(hllc_mom_z_cell),
      TENRYU_REPLAY_DESC(shock_time),
      TENRYU_REPLAY_DESC(adaptive_av_gate),
      TENRYU_REPLAY_DESC(eta_compatible),
      TENRYU_REPLAY_DESC(volFrac),
      TENRYU_REPLAY_DESC(gas_tracer_Y),
      TENRYU_REPLAY_DESC(cv_e),
      TENRYU_REPLAY_DESC(cv_i),
      TENRYU_REPLAY_DESC(cs),
      TENRYU_REPLAY_DESC(mass_per_material),
      TENRYU_REPLAY_DESC(Ee_per_material),
      TENRYU_REPLAY_DESC(Ei_per_material),
      TENRYU_REPLAY_DESC(Te_per_material),
      TENRYU_REPLAY_DESC(Ti_per_material),
      TENRYU_REPLAY_DESC(x_r),
      TENRYU_REPLAY_DESC(x_z),
      TENRYU_REPLAY_DESC(x_r_reference),
      TENRYU_REPLAY_DESC(x_z_reference),
      TENRYU_REPLAY_DESC(mesh.cell_centroid_r_device),
      TENRYU_REPLAY_DESC(mesh.cell_vol_device),
      TENRYU_REPLAY_DESC(mesh.cell_area_device),
      TENRYU_REPLAY_DESC(mesh.cell_centroid_z_device),
      TENRYU_REPLAY_DESC(mesh.cell_Svec_r_device),
      TENRYU_REPLAY_DESC(mesh.cell_Svec_z_device),
      TENRYU_REPLAY_DESC(v_r),
      TENRYU_REPLAY_DESC(v_z),
      TENRYU_REPLAY_DESC(rad_E),
      TENRYU_REPLAY_DESC(state_supply_pre_rho),
      TENRYU_REPLAY_DESC(state_supply_pre_mass),
      TENRYU_REPLAY_DESC(state_supply_pre_ee),
      TENRYU_REPLAY_DESC(state_supply_pre_ei),
      TENRYU_REPLAY_DESC(state_supply_pre_uz),
      TENRYU_REPLAY_DESC(plic_interface_mask),
      TENRYU_REPLAY_DESC(plic_active_mask),
      TENRYU_REPLAY_DESC(plic_reconstruction_valid),
      TENRYU_REPLAY_DESC(plic_normal_r),
      TENRYU_REPLAY_DESC(plic_normal_z),
      TENRYU_REPLAY_DESC(plic_alpha),
      TENRYU_REPLAY_DESC(plic_centroid_r),
      TENRYU_REPLAY_DESC(plic_centroid_z),
      TENRYU_REPLAY_DESC(plic_last_centroid_r),
      TENRYU_REPLAY_DESC(plic_last_centroid_z),
      TENRYU_REPLAY_DESC(plic_face_flux_r),
      TENRYU_REPLAY_DESC(plic_face_flux_z),
      TENRYU_REPLAY_DESC(plic_cell_residual),
  };
#undef TENRYU_REPLAY_DESC
}

}  // namespace

void RollbackGuard::capture(core::State& state, const cudaStream_t stream) {
  const auto descs = replay_device_descs(state);
  const core::TransactionError error = tx_.capture(descs, stream);
  TENRYU_ASSERT(error == core::TransactionError::kNone,
                "RollbackGuard capture failed");

  captured_bytes_.clear();
  captured_ptrs_.clear();
  captured_bytes_.reserve(descs.size());
  captured_ptrs_.reserve(descs.size());
  for (const auto& desc : descs) {
    captured_bytes_.push_back(desc.bytes);
    captured_ptrs_.push_back(desc.live_ptr);
  }

  host_.hot_e_eps_cum_host = state.hot_e_eps_cum_host;
  host_.burn_eps_cum_host = state.burn_eps_cum_host;
  host_.mesh_cell_vol = state.mesh.cell_vol;
  host_.mesh_cell_area = state.mesh.cell_area;
  host_.mesh_cell_centroid_r = state.mesh.cell_centroid_r;
  host_.mesh_cell_centroid_z = state.mesh.cell_centroid_z;
  state.mesh.materialize_host_svec();
  host_.mesh_cell_Svec_r = state.mesh.cell_Svec_r;
  host_.mesh_cell_Svec_z = state.mesh.cell_Svec_z;
  host_.corner_mass_initialized = state.corner_mass_initialized;
  host_.corner_mass_is_lagrangian_invariant =
      state.corner_mass_is_lagrangian_invariant;
  host_.gas_tracer_initialized = state.gas_tracer_initialized;
  host_.hllc_mom_z_cell_initialized = state.hllc_mom_z_cell_initialized;
  host_.holo_ale_invalidated = state.holo_ale_invalidated;
  host_.ale_rezoned = state.ale_rezoned;
  host_.ale_rezone_invocations = state.ale_rezone_invocations;
  host_.ale_remaps_applied = state.ale_remaps_applied;
  host_.ale_last_applied_step = state.ale_last_applied_step;
  host_.dt_prev_hydro = state.dt_prev_hydro;
  host_.state_supply_dM_cumulative = state.state_supply_dM_cumulative;
  host_.state_supply_dE_cumulative = state.state_supply_dE_cumulative;
  host_.state_supply_dPz_cumulative = state.state_supply_dPz_cumulative;
  host_.state_supply_dM_step = state.state_supply_dM_step;
  host_.state_supply_dE_step = state.state_supply_dE_step;
  host_.state_supply_dPz_step = state.state_supply_dPz_step;
  host_.plic_remap_sticky_fallback = state.plic_remap_sticky_fallback;
  host_.plic_consecutive_drift_triggers =
      state.plic_consecutive_drift_triggers;
  host_.plic_last_reconstruction_step = state.plic_last_reconstruction_step;
  host_.cell_is_void = state.cell_is_void;
  host_.hydro_active = state.hydro_active;
}

void RollbackGuard::restore(core::State& state, const cudaStream_t stream) {
  TENRYU_ASSERT(tx_.captured(),
                "RollbackGuard restore requires a captured transaction");

  const auto descs = replay_device_descs(state);
  TENRYU_ASSERT(descs.size() == captured_bytes_.size(),
                "RollbackGuard descriptor count changed after capture");
  TENRYU_ASSERT(descs.size() == captured_ptrs_.size(),
                "RollbackGuard captured pointer count mismatch");

  bool pointers_unchanged = true;
  for (std::size_t i = 0; i < descs.size(); ++i) {
    TENRYU_ASSERT(
        descs[i].bytes == captured_bytes_[i],
        std::string("RollbackGuard byte-size mismatch for field ") + descs[i].name);
    pointers_unchanged =
        pointers_unchanged && descs[i].live_ptr == captured_ptrs_[i];
  }

  if (pointers_unchanged) {
    tx_.record_gate("rollback", true);
    TENRYU_ASSERT(tx_.commit(stream), "RollbackGuard rollback commit failed");
  } else {
    for (const auto& desc : descs) {
      if (desc.bytes == 0) {
        continue;
      }
      void* const shadow = tx_.shadow_ptr(desc.name);
      TENRYU_ASSERT(
          shadow != nullptr,
          std::string("RollbackGuard missing shadow slot for field ") + desc.name);
      CUDA_CHECK(cudaMemcpyAsync(desc.live_ptr, shadow, desc.bytes,
                                 cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    tx_.discard();
  }

  state.hot_e_eps_cum_host = host_.hot_e_eps_cum_host;
  state.burn_eps_cum_host = host_.burn_eps_cum_host;
  state.mesh.cell_vol = host_.mesh_cell_vol;
  state.mesh.cell_area = host_.mesh_cell_area;
  state.mesh.cell_centroid_r = host_.mesh_cell_centroid_r;
  state.mesh.cell_centroid_z = host_.mesh_cell_centroid_z;
  state.mesh.cell_Svec_r = host_.mesh_cell_Svec_r;
  state.mesh.cell_Svec_z = host_.mesh_cell_Svec_z;
  state.corner_mass_initialized = host_.corner_mass_initialized;
  state.corner_mass_is_lagrangian_invariant =
      host_.corner_mass_is_lagrangian_invariant;
  state.gas_tracer_initialized = host_.gas_tracer_initialized;
  state.hllc_mom_z_cell_initialized = host_.hllc_mom_z_cell_initialized;
  state.holo_ale_invalidated = host_.holo_ale_invalidated;
  state.ale_rezoned = host_.ale_rezoned;
  state.ale_rezone_invocations = host_.ale_rezone_invocations;
  state.ale_remaps_applied = host_.ale_remaps_applied;
  state.ale_last_applied_step = host_.ale_last_applied_step;
  state.dt_prev_hydro = host_.dt_prev_hydro;
  state.state_supply_dM_cumulative = host_.state_supply_dM_cumulative;
  state.state_supply_dE_cumulative = host_.state_supply_dE_cumulative;
  state.state_supply_dPz_cumulative = host_.state_supply_dPz_cumulative;
  state.state_supply_dM_step = host_.state_supply_dM_step;
  state.state_supply_dE_step = host_.state_supply_dE_step;
  state.state_supply_dPz_step = host_.state_supply_dPz_step;
  state.plic_remap_sticky_fallback = host_.plic_remap_sticky_fallback;
  state.plic_consecutive_drift_triggers =
      host_.plic_consecutive_drift_triggers;
  state.plic_last_reconstruction_step = host_.plic_last_reconstruction_step;
  state.cell_is_void = host_.cell_is_void;
  state.hydro_active = host_.hydro_active;
}

void RollbackGuard::accept() {
  if (tx_.captured()) {
    tx_.discard();
  }
}

void RollbackGuard::telemetry_increment(const char* counter_name,
                                        const std::uint64_t delta) {
  tx_.telemetry_increment(counter_name, delta);
}

bool RollbackGuard::captured() const {
  return tx_.captured();
}

}  // namespace tenryu::hydro
