// Driver retry snapshot/restore primitive for deterministic full-step rollback.

#include "coupling/driver_retry_snapshot.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/hydro_2d.hpp"

namespace tenryu::coupling {
namespace {

std::size_t radiation_group_count(const tenryu::core::Config& cfg) {
  TENRYU_ASSERT(cfg.radiation.groups >= 0,
                "driver retry snapshot requires non-negative radiation group count");
  return static_cast<std::size_t>(cfg.radiation.groups);
}

void cuda_check(const cudaError_t err, const char* const message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

struct SnapshotCopyDesc {
  const char* field;
  unsigned long long offset;
  unsigned long long bytes;
};

__global__ void snapshot_table_copy_kernel(
    const SnapshotCopyDesc* __restrict__ table,
    const int n_entries,
    const bool to_arena,
    char* __restrict__ arena) {
  const int entry_index = static_cast<int>(blockIdx.x);
  if (entry_index >= n_entries) {
    return;
  }

  const SnapshotCopyDesc desc = table[entry_index];
  if (desc.bytes == 0ULL) {
    return;
  }

  const char* const arena_entry = arena + desc.offset;
  const char* const src = to_arena ? desc.field : arena_entry;
  char* const dst =
      to_arena ? arena + desc.offset : const_cast<char*>(desc.field);
  const bool word_aligned =
      (desc.bytes % sizeof(unsigned long long) == 0ULL) &&
      ((reinterpret_cast<std::uintptr_t>(src) &
        (alignof(unsigned long long) - 1U)) == 0U) &&
      ((reinterpret_cast<std::uintptr_t>(dst) &
        (alignof(unsigned long long) - 1U)) == 0U);

  if (word_aligned) {
    const auto* const src_words =
        reinterpret_cast<const unsigned long long*>(src);
    auto* const dst_words = reinterpret_cast<unsigned long long*>(dst);
    const unsigned long long n_words =
        desc.bytes / sizeof(unsigned long long);
    for (unsigned long long i = threadIdx.x; i < n_words;
         i += blockDim.x) {
      dst_words[i] = src_words[i];
    }
  } else {
    const auto* const src_bytes =
        reinterpret_cast<const unsigned char*>(src);
    auto* const dst_bytes = reinterpret_cast<unsigned char*>(dst);
    for (unsigned long long i = threadIdx.x; i < desc.bytes;
         i += blockDim.x) {
      dst_bytes[i] = src_bytes[i];
    }
  }
}

constexpr std::size_t kArenaAlignment = alignof(unsigned long long);
constexpr int kSnapshotCopyBlockSize = 256;

std::size_t align_arena_offset(const std::size_t offset) {
  return (offset + kArenaAlignment - 1U) &
         ~(kArenaAlignment - 1U);
}

template <typename Field>
void append_snapshot_entry(
    std::vector<DriverRetrySnapshot::Entry>& entries,
    const char* const name,
    const Field& field,
    std::size_t& arena_bytes) {
  using Value =
      std::remove_cv_t<std::remove_pointer_t<decltype(field.data())>>;
  const std::size_t bytes = field.size() * sizeof(Value);
  const std::size_t offset = align_arena_offset(arena_bytes);
  entries.push_back(
      {name, const_cast<Value*>(field.data()), offset, bytes});
  arena_bytes = offset + bytes;
}

void enumerate_snapshot_fields(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    std::vector<DriverRetrySnapshot::Entry>& out_entries) {
  static_cast<void>(cfg);
  out_entries.clear();
  std::size_t arena_bytes = 0;

#define TENRYU_SNAPSHOT_FIELD(name)                                       \
  append_snapshot_entry(out_entries, #name, state.name, arena_bytes)

  TENRYU_SNAPSHOT_FIELD(rho);
  TENRYU_SNAPSHOT_FIELD(mass);
  TENRYU_SNAPSHOT_FIELD(corner_mass);
  TENRYU_SNAPSHOT_FIELD(corner_volume);
  TENRYU_SNAPSHOT_FIELD(corner_pressure);
  TENRYU_SNAPSHOT_FIELD(subzonal_mass_corner0);
  TENRYU_SNAPSHOT_FIELD(subzonal_mass_corner1);
  TENRYU_SNAPSHOT_FIELD(subzonal_mass_corner2);
  TENRYU_SNAPSHOT_FIELD(subzonal_mass_corner3);
  TENRYU_SNAPSHOT_FIELD(vol);
  TENRYU_SNAPSHOT_FIELD(cell_vol_initial);
  TENRYU_SNAPSHOT_FIELD(vol_prev_hydro);
  TENRYU_SNAPSHOT_FIELD(zbar);
  TENRYU_SNAPSHOT_FIELD(Te);
  TENRYU_SNAPSHOT_FIELD(Ti);
  TENRYU_SNAPSHOT_FIELD(ee);
  TENRYU_SNAPSHOT_FIELD(ei);
  TENRYU_SNAPSHOT_FIELD(Pe);
  TENRYU_SNAPSHOT_FIELD(Pi);
  TENRYU_SNAPSHOT_FIELD(Qvisc);
  TENRYU_SNAPSHOT_FIELD(hllc_mom_z_cell);
  TENRYU_SNAPSHOT_FIELD(shock_time);
  TENRYU_SNAPSHOT_FIELD(adaptive_av_gate);
  TENRYU_SNAPSHOT_FIELD(wake_heat_flux_eta);
  TENRYU_SNAPSHOT_FIELD(wake_heat_flux_zeta);
  TENRYU_SNAPSHOT_FIELD(eta_compatible);
  TENRYU_SNAPSHOT_FIELD(volFrac);
  TENRYU_SNAPSHOT_FIELD(gas_tracer_Y);
  TENRYU_SNAPSHOT_FIELD(cv_e);
  TENRYU_SNAPSHOT_FIELD(cv_i);
  TENRYU_SNAPSHOT_FIELD(cs);
  TENRYU_SNAPSHOT_FIELD(mass_per_material);
  TENRYU_SNAPSHOT_FIELD(Ee_per_material);
  TENRYU_SNAPSHOT_FIELD(Ei_per_material);
  TENRYU_SNAPSHOT_FIELD(Te_per_material);
  TENRYU_SNAPSHOT_FIELD(Ti_per_material);
  if (!state.burn_Ng.empty()) {
    TENRYU_SNAPSHOT_FIELD(burn_Ng);
  }
  TENRYU_SNAPSHOT_FIELD(burn_mc_r);
  TENRYU_SNAPSHOT_FIELD(burn_mc_mu);
  TENRYU_SNAPSHOT_FIELD(burn_mc_E);
  TENRYU_SNAPSHOT_FIELD(burn_mc_w);
  TENRYU_SNAPSHOT_FIELD(burn_mc_slot);
  TENRYU_SNAPSHOT_FIELD(burn_mc_alive);
  TENRYU_SNAPSHOT_FIELD(x_r);
  TENRYU_SNAPSHOT_FIELD(x_z);
  TENRYU_SNAPSHOT_FIELD(x_r_reference);
  TENRYU_SNAPSHOT_FIELD(x_z_reference);
  TENRYU_SNAPSHOT_FIELD(v_r);
  TENRYU_SNAPSHOT_FIELD(v_z);
  TENRYU_SNAPSHOT_FIELD(node_accel_r);
  TENRYU_SNAPSHOT_FIELD(node_accel_z);
  TENRYU_SNAPSHOT_FIELD(rad_E);
  TENRYU_SNAPSHOT_FIELD(rad_E_old);
  TENRYU_SNAPSHOT_FIELD(rad_dep);
  TENRYU_SNAPSHOT_FIELD(rad_emit);
  TENRYU_SNAPSHOT_FIELD(fld_sigma_a);
  TENRYU_SNAPSHOT_FIELD(fld_sigma_pe);
  TENRYU_SNAPSHOT_FIELD(fld_sigma_R);
  TENRYU_SNAPSHOT_FIELD(fld_eta);
  TENRYU_SNAPSHOT_FIELD(fld_D_cell);
  TENRYU_SNAPSHOT_FIELD(fld_cell_rc);
  TENRYU_SNAPSHOT_FIELD(fld_cell_zc);
  TENRYU_SNAPSHOT_FIELD(fld_lower);
  TENRYU_SNAPSHOT_FIELD(fld_diag);
  TENRYU_SNAPSHOT_FIELD(fld_upper);
  TENRYU_SNAPSHOT_FIELD(fld_rhs);
  TENRYU_SNAPSHOT_FIELD(fld_Te_old);
  TENRYU_SNAPSHOT_FIELD(fld_delta_T);
  TENRYU_SNAPSHOT_FIELD(fld_fleck);
  TENRYU_SNAPSHOT_FIELD(fld_reduction_work);
  TENRYU_SNAPSHOT_FIELD(fld_cusparse_buffer);
  TENRYU_SNAPSHOT_FIELD(fld_group_bounds_work);
  TENRYU_SNAPSHOT_FIELD(fld_nlte_f_work);
  TENRYU_SNAPSHOT_FIELD(fld_nlte_sigma_eff_work);
  TENRYU_SNAPSHOT_FIELD(fld_nlte_sigma_s_eff_work);
  TENRYU_SNAPSHOT_FIELD(fld_nlte_eta_cdf_work);
  TENRYU_SNAPSHOT_FIELD(fld_nlte_lambda_work);
  TENRYU_SNAPSHOT_FIELD(fld_trace_records);
  TENRYU_SNAPSHOT_FIELD(sn_sigma_a);
  TENRYU_SNAPSHOT_FIELD(sn_rad_E_pre_newton);
  TENRYU_SNAPSHOT_FIELD(sn_sigma_pe);
  TENRYU_SNAPSHOT_FIELD(sn_sigma_s);
  TENRYU_SNAPSHOT_FIELD(sn_eta);
  TENRYU_SNAPSHOT_FIELD(sn_phi_old);
  TENRYU_SNAPSHOT_FIELD(sn_phi_new);
  TENRYU_SNAPSHOT_FIELD(sn_phi_sweep);
  TENRYU_SNAPSHOT_FIELD(sn_diag_rad_E_pre);
  TENRYU_SNAPSHOT_FIELD(sn_diag_rad_E_post);
  TENRYU_SNAPSHOT_FIELD(sn_diag_rad_emission_at_Tn);
  TENRYU_SNAPSHOT_FIELD(sn_diag_rad_emission_at_Tnp1);
  TENRYU_SNAPSHOT_FIELD(sn_diag_rad_absorption);
  TENRYU_SNAPSHOT_FIELD(sn_diag_clip_energy);
  TENRYU_SNAPSHOT_FIELD(sn_diag_clip_full_deficit);
  TENRYU_SNAPSHOT_FIELD(sn_diag_chi_opacity);
  TENRYU_SNAPSHOT_FIELD(sn_diag_F_first_moment);
  TENRYU_SNAPSHOT_FIELD(sn_face_flux_raw);
  TENRYU_SNAPSHOT_FIELD(sn_face_flux_diff);
  TENRYU_SNAPSHOT_FIELD(sn_face_flux_blended);
  TENRYU_SNAPSHOT_FIELD(sn_face_flux_limited);
  TENRYU_SNAPSHOT_FIELD(sn_face_alpha);
  TENRYU_SNAPSHOT_FIELD(sn_stream_theta);
  TENRYU_SNAPSHOT_FIELD(sn_E_star_flux);
  TENRYU_SNAPSHOT_FIELD(sn_diag_E_star_flux);
  TENRYU_SNAPSHOT_FIELD(sn_psi_scratch);
  TENRYU_SNAPSHOT_FIELD(sn_psi_outgoing_scratch);
  TENRYU_SNAPSHOT_FIELD(sn_lc_E_scratch);
  TENRYU_SNAPSHOT_FIELD(sn_lc_A_scratch);
  TENRYU_SNAPSHOT_FIELD(sn_origin_boundary);
  TENRYU_SNAPSHOT_FIELD(sn_radial_fixup_count);
  TENRYU_SNAPSHOT_FIELD(sn_radial_fixup_artificial_abs);
  TENRYU_SNAPSHOT_FIELD(sn_angular_fixup_count);
  TENRYU_SNAPSHOT_FIELD(sn_angular_fixup_artificial_abs);
  TENRYU_SNAPSHOT_FIELD(sn_Prr);
  TENRYU_SNAPSHOT_FIELD(sn_chi);
  TENRYU_SNAPSHOT_FIELD(sn_F_z);
  TENRYU_SNAPSHOT_FIELD(sn_P_zz);
  TENRYU_SNAPSHOT_FIELD(sn_chi_z);
  TENRYU_SNAPSHOT_FIELD(sn_dsa_lower);
  TENRYU_SNAPSHOT_FIELD(sn_dsa_diag);
  TENRYU_SNAPSHOT_FIELD(sn_dsa_upper);
  TENRYU_SNAPSHOT_FIELD(sn_dsa_rhs);
  TENRYU_SNAPSHOT_FIELD(sn_dsa_cusparse_buffer);
  TENRYU_SNAPSHOT_FIELD(sn_group_bounds_work);
  TENRYU_SNAPSHOT_FIELD(sn_nlte_f_work);
  TENRYU_SNAPSHOT_FIELD(sn_nlte_sigma_eff_work);
  TENRYU_SNAPSHOT_FIELD(sn_nlte_sigma_s_eff_work);
  TENRYU_SNAPSHOT_FIELD(sn_nlte_eta_cdf_work);
  TENRYU_SNAPSHOT_FIELD(sn_nlte_lambda_work);
  TENRYU_SNAPSHOT_FIELD(sn_Te_old);
  TENRYU_SNAPSHOT_FIELD(sn_delta_T);
  TENRYU_SNAPSHOT_FIELD(sn_reduction_work);
  TENRYU_SNAPSHOT_FIELD(sn_tau_R);
  TENRYU_SNAPSHOT_FIELD(sn_reduced_flux);
  TENRYU_SNAPSHOT_FIELD(sn_ap_alpha);
  TENRYU_SNAPSHOT_FIELD(holo_E_LO);
  TENRYU_SNAPSHOT_FIELD(holo_F_LO);
  TENRYU_SNAPSHOT_FIELD(holo_consistency_source);
  TENRYU_SNAPSHOT_FIELD(holo_rad_dep);
  TENRYU_SNAPSHOT_FIELD(holo_rad_emit);
  TENRYU_SNAPSHOT_FIELD(holo_Prr);
  TENRYU_SNAPSHOT_FIELD(holo_chi);
  TENRYU_SNAPSHOT_FIELD(holo_chi_filtered);
  TENRYU_SNAPSHOT_FIELD(holo_Prr_coverage);
  TENRYU_SNAPSHOT_FIELD(difference_W);
  TENRYU_SNAPSHOT_FIELD(difference_E_ref);
  TENRYU_SNAPSHOT_FIELD(difference_residual_E);
  TENRYU_SNAPSHOT_FIELD(delta_E_rad_prev);
  TENRYU_SNAPSHOT_FIELD(state_supply_pre_rho);
  TENRYU_SNAPSHOT_FIELD(state_supply_pre_mass);
  TENRYU_SNAPSHOT_FIELD(state_supply_pre_ee);
  TENRYU_SNAPSHOT_FIELD(state_supply_pre_ei);
  TENRYU_SNAPSHOT_FIELD(state_supply_pre_uz);
  TENRYU_SNAPSHOT_FIELD(plic_interface_mask);
  TENRYU_SNAPSHOT_FIELD(plic_active_mask);
  TENRYU_SNAPSHOT_FIELD(plic_reconstruction_valid);
  TENRYU_SNAPSHOT_FIELD(plic_normal_r);
  TENRYU_SNAPSHOT_FIELD(plic_normal_z);
  TENRYU_SNAPSHOT_FIELD(plic_alpha);
  TENRYU_SNAPSHOT_FIELD(plic_centroid_r);
  TENRYU_SNAPSHOT_FIELD(plic_centroid_z);
  TENRYU_SNAPSHOT_FIELD(plic_last_centroid_r);
  TENRYU_SNAPSHOT_FIELD(plic_last_centroid_z);
  TENRYU_SNAPSHOT_FIELD(plic_face_flux_r);
  TENRYU_SNAPSHOT_FIELD(plic_face_flux_z);
  TENRYU_SNAPSHOT_FIELD(plic_cell_residual);

#undef TENRYU_SNAPSHOT_FIELD
}

bool registry_matches(
    const std::vector<DriverRetrySnapshot::Entry>& lhs,
    const std::vector<DriverRetrySnapshot::Entry>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (std::size_t i = 0; i < lhs.size(); ++i) {
    if (std::strcmp(lhs[i].name, rhs[i].name) != 0 ||
        lhs[i].field_ptr != rhs[i].field_ptr ||
        lhs[i].offset != rhs[i].offset ||
        lhs[i].bytes != rhs[i].bytes) {
      return false;
    }
  }
  return true;
}

void upload_snapshot_table(DriverRetrySnapshot& snap) {
  if (!snap.table_dirty) {
    return;
  }

  std::vector<SnapshotCopyDesc> copy_table;
  copy_table.reserve(snap.entries.size());
  for (const auto& entry : snap.entries) {
    copy_table.push_back(
        {static_cast<const char*>(entry.field_ptr),
         static_cast<unsigned long long>(entry.offset),
         static_cast<unsigned long long>(entry.bytes)});
  }
  if (!copy_table.empty()) {
    cuda_check(cudaMemcpy(snap.d_table,
                          copy_table.data(),
                          copy_table.size() * sizeof(SnapshotCopyDesc),
                          cudaMemcpyHostToDevice),
               "driver retry snapshot table upload failed");
  }
  snap.table_dirty = false;
}

void rebind_mesh_coordinate_fields(tenryu::core::State& state) {
  state.mesh.node_r = state.x_r.data();
  state.mesh.node_z = (state.mesh.dim == 2) ? state.x_z.data() : nullptr;
}

}  // namespace

DriverRetrySnapshot::~DriverRetrySnapshot() {
  // Best-effort: the CUDA context may already be torn down at destruction time.
  if (d_table != nullptr) {
    static_cast<void>(cudaFree(d_table));
  }
  if (d_arena != nullptr) {
    static_cast<void>(cudaFree(d_arena));
  }
}

std::vector<SnapshotEntryView> driver_retry_snapshot_readback(
    const DriverRetrySnapshot& snap) {
  std::vector<SnapshotEntryView> views;
  views.reserve(snap.entries.size());
  const auto* const arena =
      static_cast<const unsigned char*>(snap.d_arena);
  for (const auto& entry : snap.entries) {
    TENRYU_ASSERT(
        entry.offset <= snap.arena_bytes &&
            entry.bytes <= snap.arena_bytes - entry.offset,
        "driver retry snapshot readback entry exceeds arena");
    SnapshotEntryView view{entry.name, entry.bytes, {}};
    view.host_copy.resize(entry.bytes);
    if (entry.bytes > 0) {
      TENRYU_ASSERT(arena != nullptr,
                    "driver retry snapshot readback requires an arena");
      cuda_check(cudaMemcpy(view.host_copy.data(),
                            arena + entry.offset,
                            entry.bytes,
                            cudaMemcpyDeviceToHost),
                 "driver retry snapshot readback D2H failed");
    }
    views.push_back(std::move(view));
  }
  return views;
}

bool driver_retry_snapshot_entry_readback(
    const DriverRetrySnapshot& snap,
    const char* const name,
    std::vector<unsigned char>& out_bytes) {
  out_bytes.clear();
  if (name == nullptr) {
    return false;
  }
  for (const auto& entry : snap.entries) {
    if (std::strcmp(entry.name, name) != 0) {
      continue;
    }
    TENRYU_ASSERT(
        entry.offset <= snap.arena_bytes &&
            entry.bytes <= snap.arena_bytes - entry.offset,
        "driver retry snapshot entry readback exceeds arena");
    out_bytes.resize(entry.bytes);
    if (entry.bytes > 0) {
      TENRYU_ASSERT(snap.d_arena != nullptr,
                    "driver retry snapshot entry readback requires an arena");
      const auto* const arena =
          static_cast<const unsigned char*>(snap.d_arena);
      cuda_check(cudaMemcpy(out_bytes.data(),
                            arena + entry.offset,
                            entry.bytes,
                            cudaMemcpyDeviceToHost),
                 "driver retry snapshot entry readback D2H failed");
    }
    return true;
  }
  return false;
}

void capture_driver_retry_snapshot(DriverRetrySnapshot& snap,
                                   const tenryu::core::State& state,
                                   const tenryu::core::Config& cfg,
                                   void* cuda_stream) {
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  snap.n_cells = state.rho.size();
  snap.n_nodes = state.x_r.size();
  snap.n_groups = radiation_group_count(cfg);

  std::vector<DriverRetrySnapshot::Entry> new_entries;
  new_entries.reserve(snap.entries.size());
  enumerate_snapshot_fields(state, cfg, new_entries);
  const std::size_t required_arena_bytes =
      new_entries.empty()
          ? 0
          : new_entries.back().offset + new_entries.back().bytes;

  snap.burn_n_host = state.burn_n_host;
  snap.burn_rate_host = state.burn_rate_host;
  snap.burn_Q_e_host = state.burn_Q_e_host;
  snap.burn_Q_i_host = state.burn_Q_i_host;
  snap.burn_eps_cum_host = state.burn_eps_cum_host;
  snap.E_burn_released = state.E_burn_released;
  snap.E_burn_dep_e = state.E_burn_dep_e;
  snap.E_burn_dep_i = state.E_burn_dep_i;
  snap.E_burn_esc_charged = state.E_burn_esc_charged;
  snap.E_burn_esc_neutron = state.E_burn_esc_neutron;
  snap.E_burn_inflight = state.E_burn_inflight;
  snap.N_burn_neutrons_dt = state.N_burn_neutrons_dt;
  snap.N_burn_neutrons_dd = state.N_burn_neutrons_dd;
  snap.burn_mc_live = state.burn_mc_live;
  snap.hot_e_eps_cum_host = state.hot_e_eps_cum_host;
  snap.E_hot_e_deposited = state.E_hot_e_deposited;
  snap.E_hot_e_escaped = state.E_hot_e_escaped;
  snap.hot_e_enabled_any = state.hot_e_enabled_any;

  if (!registry_matches(snap.entries, new_entries)) {
    snap.entries = std::move(new_entries);
    snap.table_dirty = true;
  }

  if (required_arena_bytes > snap.arena_bytes) {
    if (snap.d_arena != nullptr) {
      cuda_check(cudaFree(snap.d_arena),
                 "driver retry snapshot arena cudaFree failed");
    }
    cuda_check(cudaMalloc(&snap.d_arena, required_arena_bytes),
               "driver retry snapshot arena cudaMalloc failed");
    snap.arena_bytes = required_arena_bytes;
  }

  if (snap.entries.size() > snap.table_capacity) {
    if (snap.d_table != nullptr) {
      cuda_check(cudaFree(snap.d_table),
                 "driver retry snapshot table cudaFree failed");
    }
    cuda_check(cudaMalloc(
                   &snap.d_table,
                   snap.entries.size() * sizeof(SnapshotCopyDesc)),
               "driver retry snapshot table cudaMalloc failed");
    snap.table_capacity = snap.entries.size();
    snap.table_dirty = true;
  }

  upload_snapshot_table(snap);

  TENRYU_ASSERT(
      snap.entries.size() <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "driver retry snapshot entry count exceeds kernel limit");
  if (!snap.entries.empty()) {
    snapshot_table_copy_kernel<<<static_cast<int>(snap.entries.size()),
                                 kSnapshotCopyBlockSize,
                                 0,
                                 stream>>>(
        static_cast<const SnapshotCopyDesc*>(snap.d_table),
        static_cast<int>(snap.entries.size()),
        true,
        static_cast<char*>(snap.d_arena));
    cuda_check(cudaGetLastError(),
               "driver retry snapshot capture kernel launch failed");
  }
  // No capture synchronization is needed: the table upload, arena copy, and
  // subsequent step kernels are ordered on the same (normally default) stream.

  snap.ddmc_mode_map = state.ddmc_mode_map;
  snap.holo_core_mask = state.holo_core_mask;
  snap.holo_patch_mask = state.holo_patch_mask;
  snap.holo_core_prev_mask = state.holo_core_prev_mask;
  snap.holo_hold_count = state.holo_hold_count;
  snap.holo_dwell_count = state.holo_dwell_count;
  snap.holo_tau_R = state.holo_tau_R;
  snap.holo_reduced_flux = state.holo_reduced_flux;
  snap.holo_mass_q = state.holo_mass_q;
  snap.holo_lo_weight = state.holo_lo_weight;
  snap.hydro_active = state.hydro_active;
  snap.state_supply_mask = state.state_supply_mask;
  snap.cell_is_void = state.cell_is_void;
  // Active/void masks encode topology decisions. Emergency ALE cell
  // deactivation (emergency-cell-deactivation feature) must persist across hydro retries
  // instead of being rewound with transient hydro state.
  // state_supply_mask (RH1 state-supply BC) is also re-applied each step from
  // the BC config, so excluding it from snapshot is safe.

  snap.E_safety = state.E_safety;
  snap.E_numerical_loss = state.E_numerical_loss;
  snap.E_laser_deposited = state.E_laser_deposited;
  snap.E_laser_escaped = state.E_laser_escaped;
  snap.E_laser_incident = state.E_laser_incident;
  snap.E_ra_deposited = state.E_ra_deposited;
  snap.E_cbet_iaw_step = state.E_cbet_iaw_step;
  snap.E_cbet_iaw = state.E_cbet_iaw;
  snap.E_rad_escaped = state.E_rad_escaped;
  snap.E_floor_injected = state.E_floor_injected;
  snap.E_pdV_bdry = state.E_pdV_bdry;
  snap.E_Marshak_in = state.E_Marshak_in;
  snap.E_solver = state.E_solver;
  snap.state_supply_dM_cumulative = state.state_supply_dM_cumulative;
  snap.state_supply_dE_cumulative = state.state_supply_dE_cumulative;
  snap.state_supply_dPz_cumulative = state.state_supply_dPz_cumulative;
  snap.state_supply_dM_step = state.state_supply_dM_step;
  snap.state_supply_dE_step = state.state_supply_dE_step;
  snap.state_supply_dPz_step = state.state_supply_dPz_step;
  snap.fld_state_supply_in_cumulative =
      state.fld_state_supply_in_cumulative;
  snap.fld_state_supply_out_cumulative =
      state.fld_state_supply_out_cumulative;
  snap.fld_state_supply_net_cumulative =
      state.fld_state_supply_net_cumulative;
  snap.fld_state_supply_in_step = state.fld_state_supply_in_step;
  snap.fld_state_supply_out_step = state.fld_state_supply_out_step;
  snap.fld_state_supply_net_step = state.fld_state_supply_net_step;
  snap.t = state.t;
  snap.dt = state.dt;
  snap.dt_prev_hydro = state.dt_prev_hydro;
  snap.step = state.step;
  snap.corner_mass_initialized = state.corner_mass_initialized;
  snap.corner_mass_is_lagrangian_invariant =
      state.corner_mass_is_lagrangian_invariant;
  snap.gas_tracer_initialized = state.gas_tracer_initialized;
  snap.hllc_mom_z_cell_initialized = state.hllc_mom_z_cell_initialized;
  snap.ale_rezoned = state.ale_rezoned;
  snap.ale_rezone_invocations = state.ale_rezone_invocations;
  snap.ale_remaps_applied = state.ale_remaps_applied;
  snap.ale_last_applied_step = state.ale_last_applied_step;
  snap.plic_remap_sticky_fallback = state.plic_remap_sticky_fallback;
  snap.plic_consecutive_drift_triggers = state.plic_consecutive_drift_triggers;
  snap.plic_last_reconstruction_step = state.plic_last_reconstruction_step;
  snap.axis_margin_initial = state.axis_margin_initial;
  snap.hydro_t_start_eV = state.hydro_t_start_eV;
  snap.adaptive_av_r0 = state.adaptive_av_r0;
  snap.adaptive_av_last_rs = state.adaptive_av_last_rs;
  snap.adaptive_av_last_us = state.adaptive_av_last_us;
  snap.adaptive_av_rs_min = state.adaptive_av_rs_min;
  snap.adaptive_av_tracker_steps = state.adaptive_av_tracker_steps;
  snap.adaptive_av_mode = state.adaptive_av_mode;
  snap.adaptive_av_tracker_valid = state.adaptive_av_tracker_valid;
  snap.adaptive_av_bounce_seen = state.adaptive_av_bounce_seen;
  snap.axis_mass_initial = state.axis_mass_initial;
  snap.axis_inflow_budget = state.axis_inflow_budget;
  snap.ddmc_mode_map_valid = state.ddmc_mode_map_valid;
  snap.holo_core_mask_valid = state.holo_core_mask_valid;
  snap.holo_lo_source_valid = state.holo_lo_source_valid;
  snap.holo_ale_invalidated = state.holo_ale_invalidated;
  snap.particle_sort_cache_invalidated = state.particle_sort_cache_invalidated;

  snap.fld_clamp_hits_step = state.fld_clamp_hits_step;
  snap.fld_clamp_energy_delta_step = state.fld_clamp_energy_delta_step;
  snap.fld_min_x_raw_step = state.fld_min_x_raw_step;
  snap.fld_cg_true_residual_l2_rel_step = state.fld_cg_true_residual_l2_rel_step;
  snap.fld_cg_true_residual_max_step = state.fld_cg_true_residual_max_step;
  snap.fld_E_solver_step = state.fld_E_solver_step;
  snap.fld_cg_true_residual_l2_rel_RAW_step =
      state.fld_cg_true_residual_l2_rel_RAW_step;
  snap.fld_cg_true_residual_max_RAW_step = state.fld_cg_true_residual_max_RAW_step;
  snap.fld_E_solver_RAW_step = state.fld_E_solver_RAW_step;
  snap.fld_csr_diag_min_pos_step = state.fld_csr_diag_min_pos_step;
  snap.fld_csr_diag_max_step = state.fld_csr_diag_max_step;
  snap.fld_csr_weak_diag_dom_count_step = state.fld_csr_weak_diag_dom_count_step;
  snap.fld_csr_nonfinite_count_step = state.fld_csr_nonfinite_count_step;
  snap.fld_gershgorin_lower_min_step = state.fld_gershgorin_lower_min_step;
  snap.fld_gershgorin_upper_max_step = state.fld_gershgorin_upper_max_step;
  snap.fld_cg_pAp_min_step = state.fld_cg_pAp_min_step;
  snap.fld_cg_nonpos_pAp_count_step = state.fld_cg_nonpos_pAp_count_step;
  snap.fld_cg_nonfinite_count_step = state.fld_cg_nonfinite_count_step;
  snap.fld_cg_recurrent_resid_last_check_max_step =
      state.fld_cg_recurrent_resid_last_check_max_step;
  snap.fld_Dcell_min_step = state.fld_Dcell_min_step;
  snap.fld_Dcell_max_step = state.fld_Dcell_max_step;
  snap.fld_Dcell_zero_count_step = state.fld_Dcell_zero_count_step;
  snap.fld_Dcell_nonfinite_count_step = state.fld_Dcell_nonfinite_count_step;
  snap.fld_face_skip_D_count_step = state.fld_face_skip_D_count_step;
  snap.fld_face_skip_nonfinite_count_step = state.fld_face_skip_nonfinite_count_step;
  snap.fld_face_skip_dist_count_step = state.fld_face_skip_dist_count_step;
  snap.fld_diag_fallback_count_step = state.fld_diag_fallback_count_step;
  snap.fld_pair_symmetry_max_diff_step = state.fld_pair_symmetry_max_diff_step;
  snap.fld_pair_symmetry_violation_count_step =
      state.fld_pair_symmetry_violation_count_step;
  snap.fld_newton_converged_count_step = state.fld_newton_converged_count_step;
  snap.fld_newton_cap_hit_count_step = state.fld_newton_cap_hit_count_step;
  snap.fld_newton_invalid_count_step = state.fld_newton_invalid_count_step;
  snap.fld_newton_resid_abs_max_step = state.fld_newton_resid_abs_max_step;
  snap.fld_newton_resid_rel_max_step = state.fld_newton_resid_rel_max_step;
  snap.fld_newton_reject_count_step = state.fld_newton_reject_count_step;
  snap.fld_newton_reject_resid_rel_max_step =
      state.fld_newton_reject_resid_rel_max_step;
  snap.fld_rogue_max_abs_x_raw_step = state.fld_rogue_max_abs_x_raw_step;
  snap.fld_rogue_max_abs_x_raw_cell_step =
      state.fld_rogue_max_abs_x_raw_cell_step;
  snap.fld_rogue_max_abs_x_raw_group_step =
      state.fld_rogue_max_abs_x_raw_group_step;
  snap.fld_rogue_min_x_raw_step = state.fld_rogue_min_x_raw_step;
  snap.fld_rogue_min_x_raw_cell_step = state.fld_rogue_min_x_raw_cell_step;
  snap.fld_rogue_min_x_raw_group_step = state.fld_rogue_min_x_raw_group_step;
  snap.fld_rogue_max_rad_E_step = state.fld_rogue_max_rad_E_step;
  snap.fld_rogue_max_rad_E_cell_step = state.fld_rogue_max_rad_E_cell_step;
  snap.fld_rogue_max_rad_E_group_step = state.fld_rogue_max_rad_E_group_step;
  snap.fld_rogue_max_r_true_step = state.fld_rogue_max_r_true_step;
  snap.fld_rogue_max_r_true_cell_step = state.fld_rogue_max_r_true_cell_step;
  snap.fld_rogue_max_r_true_group_step = state.fld_rogue_max_r_true_group_step;
  snap.fld_rogue_max_E_solver_row_step = state.fld_rogue_max_E_solver_row_step;
  snap.fld_rogue_max_E_solver_row_cell_step =
      state.fld_rogue_max_E_solver_row_cell_step;
  snap.fld_rogue_max_E_solver_row_group_step =
      state.fld_rogue_max_E_solver_row_group_step;
  snap.fld_escaped_step = state.fld_escaped_step;
  snap.fld_marshak_in_step = state.fld_marshak_in_step;
  snap.fld_outer_iterations = state.fld_outer_iterations;
  snap.fld_converged = state.fld_converged;
  snap.fld_outer_residual = state.fld_outer_residual;
  snap.sn_outer_iterations = state.sn_outer_iterations;
  snap.sn_inner_iterations = state.sn_inner_iterations;
  snap.sn_converged = state.sn_converged;
  snap.sn_outer_stagnated = state.sn_outer_stagnated;
  snap.sn_outer_residual = state.sn_outer_residual;
  snap.sn_inner_residual = state.sn_inner_residual;
  snap.sn_escaped_step = state.sn_escaped_step;
  snap.sn_marshak_in_step = state.sn_marshak_in_step;
  snap.sn_void_anchor_dE_step = state.sn_void_anchor_dE_step;
  snap.sn_void_anchor_dE_abs_step = state.sn_void_anchor_dE_abs_step;

  snap.valid = true;
}

void restore_driver_retry_snapshot(tenryu::core::State& state,
                                   const tenryu::core::Config& cfg,
                                   DriverRetrySnapshot& snap,
                                   const bool restore_transient_topology_masks,
                                   const bool invalidate_particle_sort_cache,
                                   void* cuda_stream) {
  const cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);
  TENRYU_ASSERT(snap.valid, "restore_driver_retry_snapshot requires a valid snapshot");
  TENRYU_ASSERT(snap.n_cells == state.rho.size(),
                "restore_driver_retry_snapshot n_cells mismatch");
  TENRYU_ASSERT(snap.n_nodes == state.x_r.size(),
                "restore_driver_retry_snapshot n_nodes mismatch");
  TENRYU_ASSERT(snap.n_groups == radiation_group_count(cfg),
                "restore_driver_retry_snapshot radiation group count mismatch");

  std::vector<DriverRetrySnapshot::Entry> current_entries;
  current_entries.reserve(snap.entries.size());
  enumerate_snapshot_fields(state, cfg, current_entries);
  TENRYU_ASSERT(
      current_entries.size() == snap.entries.size(),
      "restore_driver_retry_snapshot entry count mismatch");
  for (std::size_t i = 0; i < snap.entries.size(); ++i) {
    TENRYU_ASSERT(
        std::strcmp(current_entries[i].name, snap.entries[i].name) == 0,
        "restore_driver_retry_snapshot entry name mismatch");
    TENRYU_ASSERT(
        current_entries[i].bytes == snap.entries[i].bytes,
        "restore_driver_retry_snapshot entry size mismatch");
    if (current_entries[i].field_ptr != snap.entries[i].field_ptr) {
      snap.entries[i].field_ptr = current_entries[i].field_ptr;
      snap.table_dirty = true;
    }
  }
  upload_snapshot_table(snap);

  if (!snap.entries.empty()) {
    snapshot_table_copy_kernel<<<static_cast<int>(snap.entries.size()),
                                 kSnapshotCopyBlockSize,
                                 0,
                                 stream>>>(
        static_cast<const SnapshotCopyDesc*>(snap.d_table),
        static_cast<int>(snap.entries.size()),
        false,
        static_cast<char*>(snap.d_arena));
    cuda_check(cudaGetLastError(),
               "driver retry snapshot restore kernel launch failed");
  }
  cuda_check(cudaDeviceSynchronize(),
             "driver retry snapshot restore device sync failed");

  state.burn_n_host = snap.burn_n_host;
  state.burn_rate_host = snap.burn_rate_host;
  state.burn_Q_e_host = snap.burn_Q_e_host;
  state.burn_Q_i_host = snap.burn_Q_i_host;
  state.burn_eps_cum_host = snap.burn_eps_cum_host;
  state.E_burn_released = snap.E_burn_released;
  state.E_burn_dep_e = snap.E_burn_dep_e;
  state.E_burn_dep_i = snap.E_burn_dep_i;
  state.E_burn_esc_charged = snap.E_burn_esc_charged;
  state.E_burn_esc_neutron = snap.E_burn_esc_neutron;
  state.E_burn_inflight = snap.E_burn_inflight;
  state.N_burn_neutrons_dt = snap.N_burn_neutrons_dt;
  state.N_burn_neutrons_dd = snap.N_burn_neutrons_dd;
  state.burn_mc_live = snap.burn_mc_live;
  state.hot_e_eps_cum_host = snap.hot_e_eps_cum_host;
  state.E_hot_e_deposited = snap.E_hot_e_deposited;
  state.E_hot_e_escaped = snap.E_hot_e_escaped;
  state.hot_e_enabled_any = snap.hot_e_enabled_any;
  state.ring7_seam_rezone_requested = false;
  state.ring7_seam_rezone_request_cell = -1;
  state.ring7_pole_cap_oracle_requested = false;
  state.ring7_pole_cap_request_cell = -1;
  state.ring7_pole_cap_validation_pending = false;
  state.ring7_pole_cap_validation_cell = -1;
  state.ring7_pole_cap_validation_dt = 0.0;
  state.ring7_pole_cap_eta_prod_proxy = 0.0;

  state.ddmc_mode_map = snap.ddmc_mode_map;
  state.holo_core_mask = snap.holo_core_mask;
  state.holo_patch_mask = snap.holo_patch_mask;
  state.holo_core_prev_mask = snap.holo_core_prev_mask;
  state.holo_hold_count = snap.holo_hold_count;
  state.holo_dwell_count = snap.holo_dwell_count;
  state.holo_tau_R = snap.holo_tau_R;
  state.holo_reduced_flux = snap.holo_reduced_flux;
  state.holo_mass_q = snap.holo_mass_q;
  state.holo_lo_weight = snap.holo_lo_weight;
  // Keep active/void masks as they stand after any retry-local topology update
  // (Emergency-cell-deactivation persistence.)
  if (restore_transient_topology_masks) {
    if (snap.hydro_active.size() != state.hydro_active.size()) {
      std::ostringstream log;
      log << "[snapshot] hydro_active size change on restore: snap="
          << snap.hydro_active.size()
          << " state=" << state.hydro_active.size();
      core::log_warning(log.str());
    }
    state.hydro_active = snap.hydro_active;
    state.state_supply_mask = snap.state_supply_mask;
    state.cell_is_void = snap.cell_is_void;
  }

  state.E_safety = snap.E_safety;
  state.E_numerical_loss = snap.E_numerical_loss;
  state.E_laser_deposited = snap.E_laser_deposited;
  state.E_laser_escaped = snap.E_laser_escaped;
  state.E_laser_incident = snap.E_laser_incident;
  state.E_ra_deposited = snap.E_ra_deposited;
  state.E_cbet_iaw_step = snap.E_cbet_iaw_step;
  state.E_cbet_iaw = snap.E_cbet_iaw;
  state.E_rad_escaped = snap.E_rad_escaped;
  state.E_floor_injected = snap.E_floor_injected;
  state.E_pdV_bdry = snap.E_pdV_bdry;
  state.E_Marshak_in = snap.E_Marshak_in;
  state.E_solver = snap.E_solver;
  state.state_supply_dM_cumulative = snap.state_supply_dM_cumulative;
  state.state_supply_dE_cumulative = snap.state_supply_dE_cumulative;
  state.state_supply_dPz_cumulative = snap.state_supply_dPz_cumulative;
  state.state_supply_dM_step = snap.state_supply_dM_step;
  state.state_supply_dE_step = snap.state_supply_dE_step;
  state.state_supply_dPz_step = snap.state_supply_dPz_step;
  state.fld_state_supply_in_cumulative =
      snap.fld_state_supply_in_cumulative;
  state.fld_state_supply_out_cumulative =
      snap.fld_state_supply_out_cumulative;
  state.fld_state_supply_net_cumulative =
      snap.fld_state_supply_net_cumulative;
  state.fld_state_supply_in_step = snap.fld_state_supply_in_step;
  state.fld_state_supply_out_step = snap.fld_state_supply_out_step;
  state.fld_state_supply_net_step = snap.fld_state_supply_net_step;
  state.t = snap.t;
  state.dt = snap.dt;
  state.dt_prev_hydro = snap.dt_prev_hydro;
  state.step = snap.step;
  state.corner_mass_initialized = snap.corner_mass_initialized;
  state.corner_mass_is_lagrangian_invariant =
      snap.corner_mass_is_lagrangian_invariant;
  state.gas_tracer_initialized = snap.gas_tracer_initialized;
  state.hllc_mom_z_cell_initialized = snap.hllc_mom_z_cell_initialized;
  state.ale_rezoned = snap.ale_rezoned;
  state.ale_rezone_invocations = snap.ale_rezone_invocations;
  state.ale_remaps_applied = snap.ale_remaps_applied;
  state.ale_last_applied_step = snap.ale_last_applied_step;
  state.plic_remap_sticky_fallback = snap.plic_remap_sticky_fallback;
  state.plic_consecutive_drift_triggers = snap.plic_consecutive_drift_triggers;
  state.plic_last_reconstruction_step = snap.plic_last_reconstruction_step;
  state.axis_margin_initial = snap.axis_margin_initial;
  state.hydro_t_start_eV = snap.hydro_t_start_eV;
  state.adaptive_av_r0 = snap.adaptive_av_r0;
  state.adaptive_av_last_rs = snap.adaptive_av_last_rs;
  state.adaptive_av_last_us = snap.adaptive_av_last_us;
  state.adaptive_av_rs_min = snap.adaptive_av_rs_min;
  state.adaptive_av_tracker_steps = snap.adaptive_av_tracker_steps;
  state.adaptive_av_mode = snap.adaptive_av_mode;
  state.adaptive_av_tracker_valid = snap.adaptive_av_tracker_valid;
  state.adaptive_av_bounce_seen = snap.adaptive_av_bounce_seen;
  state.axis_mass_initial = snap.axis_mass_initial;
  state.axis_inflow_budget = snap.axis_inflow_budget;
  state.ddmc_mode_map_valid = snap.ddmc_mode_map_valid;
  state.holo_core_mask_valid = snap.holo_core_mask_valid;
  state.holo_lo_source_valid = snap.holo_lo_source_valid;
  state.holo_ale_invalidated = snap.holo_ale_invalidated;
  state.particle_sort_cache_invalidated = snap.particle_sort_cache_invalidated;

  state.fld_clamp_hits_step = snap.fld_clamp_hits_step;
  state.fld_clamp_energy_delta_step = snap.fld_clamp_energy_delta_step;
  state.fld_min_x_raw_step = snap.fld_min_x_raw_step;
  state.fld_cg_true_residual_l2_rel_step = snap.fld_cg_true_residual_l2_rel_step;
  state.fld_cg_true_residual_max_step = snap.fld_cg_true_residual_max_step;
  state.fld_E_solver_step = snap.fld_E_solver_step;
  state.fld_cg_true_residual_l2_rel_RAW_step =
      snap.fld_cg_true_residual_l2_rel_RAW_step;
  state.fld_cg_true_residual_max_RAW_step = snap.fld_cg_true_residual_max_RAW_step;
  state.fld_E_solver_RAW_step = snap.fld_E_solver_RAW_step;
  state.fld_csr_diag_min_pos_step = snap.fld_csr_diag_min_pos_step;
  state.fld_csr_diag_max_step = snap.fld_csr_diag_max_step;
  state.fld_csr_weak_diag_dom_count_step = snap.fld_csr_weak_diag_dom_count_step;
  state.fld_csr_nonfinite_count_step = snap.fld_csr_nonfinite_count_step;
  state.fld_gershgorin_lower_min_step = snap.fld_gershgorin_lower_min_step;
  state.fld_gershgorin_upper_max_step = snap.fld_gershgorin_upper_max_step;
  state.fld_cg_pAp_min_step = snap.fld_cg_pAp_min_step;
  state.fld_cg_nonpos_pAp_count_step = snap.fld_cg_nonpos_pAp_count_step;
  state.fld_cg_nonfinite_count_step = snap.fld_cg_nonfinite_count_step;
  state.fld_cg_recurrent_resid_last_check_max_step =
      snap.fld_cg_recurrent_resid_last_check_max_step;
  state.fld_Dcell_min_step = snap.fld_Dcell_min_step;
  state.fld_Dcell_max_step = snap.fld_Dcell_max_step;
  state.fld_Dcell_zero_count_step = snap.fld_Dcell_zero_count_step;
  state.fld_Dcell_nonfinite_count_step = snap.fld_Dcell_nonfinite_count_step;
  state.fld_face_skip_D_count_step = snap.fld_face_skip_D_count_step;
  state.fld_face_skip_nonfinite_count_step = snap.fld_face_skip_nonfinite_count_step;
  state.fld_face_skip_dist_count_step = snap.fld_face_skip_dist_count_step;
  state.fld_diag_fallback_count_step = snap.fld_diag_fallback_count_step;
  state.fld_pair_symmetry_max_diff_step = snap.fld_pair_symmetry_max_diff_step;
  state.fld_pair_symmetry_violation_count_step =
      snap.fld_pair_symmetry_violation_count_step;
  state.fld_newton_converged_count_step = snap.fld_newton_converged_count_step;
  state.fld_newton_cap_hit_count_step = snap.fld_newton_cap_hit_count_step;
  state.fld_newton_invalid_count_step = snap.fld_newton_invalid_count_step;
  state.fld_newton_resid_abs_max_step = snap.fld_newton_resid_abs_max_step;
  state.fld_newton_resid_rel_max_step = snap.fld_newton_resid_rel_max_step;
  state.fld_newton_reject_count_step = snap.fld_newton_reject_count_step;
  state.fld_newton_reject_resid_rel_max_step =
      snap.fld_newton_reject_resid_rel_max_step;
  state.fld_rogue_max_abs_x_raw_step = snap.fld_rogue_max_abs_x_raw_step;
  state.fld_rogue_max_abs_x_raw_cell_step =
      snap.fld_rogue_max_abs_x_raw_cell_step;
  state.fld_rogue_max_abs_x_raw_group_step =
      snap.fld_rogue_max_abs_x_raw_group_step;
  state.fld_rogue_min_x_raw_step = snap.fld_rogue_min_x_raw_step;
  state.fld_rogue_min_x_raw_cell_step = snap.fld_rogue_min_x_raw_cell_step;
  state.fld_rogue_min_x_raw_group_step = snap.fld_rogue_min_x_raw_group_step;
  state.fld_rogue_max_rad_E_step = snap.fld_rogue_max_rad_E_step;
  state.fld_rogue_max_rad_E_cell_step = snap.fld_rogue_max_rad_E_cell_step;
  state.fld_rogue_max_rad_E_group_step = snap.fld_rogue_max_rad_E_group_step;
  state.fld_rogue_max_r_true_step = snap.fld_rogue_max_r_true_step;
  state.fld_rogue_max_r_true_cell_step = snap.fld_rogue_max_r_true_cell_step;
  state.fld_rogue_max_r_true_group_step = snap.fld_rogue_max_r_true_group_step;
  state.fld_rogue_max_E_solver_row_step = snap.fld_rogue_max_E_solver_row_step;
  state.fld_rogue_max_E_solver_row_cell_step =
      snap.fld_rogue_max_E_solver_row_cell_step;
  state.fld_rogue_max_E_solver_row_group_step =
      snap.fld_rogue_max_E_solver_row_group_step;
  state.fld_escaped_step = snap.fld_escaped_step;
  state.fld_marshak_in_step = snap.fld_marshak_in_step;
  state.fld_outer_iterations = snap.fld_outer_iterations;
  state.fld_converged = snap.fld_converged;
  state.fld_outer_residual = snap.fld_outer_residual;
  state.sn_outer_iterations = snap.sn_outer_iterations;
  state.sn_inner_iterations = snap.sn_inner_iterations;
  state.sn_converged = snap.sn_converged;
  state.sn_outer_stagnated = snap.sn_outer_stagnated;
  state.sn_outer_residual = snap.sn_outer_residual;
  state.sn_inner_residual = snap.sn_inner_residual;
  state.sn_escaped_step = snap.sn_escaped_step;
  state.sn_marshak_in_step = snap.sn_marshak_in_step;
  state.sn_void_anchor_dE_step = snap.sn_void_anchor_dE_step;
  state.sn_void_anchor_dE_abs_step = snap.sn_void_anchor_dE_abs_step;

  rebind_mesh_coordinate_fields(state);
  tenryu::hydro::validate_geometry_after_retry_restore(state, cfg);
  if (invalidate_particle_sort_cache) {
    state.particle_sort_cache_invalidated = true;
  }
}

}  // namespace tenryu::coupling
