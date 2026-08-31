#include "core/state.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "core/error.hpp"

namespace tenryu::core {
namespace {

constexpr double kShockTimeInactive = -1.0e300;

bool compatible_force_work_buffers_requested(const Config& cfg) {
  return cfg.main.dim == 2 &&
         (cfg.numerics.hydro.subzonal_mass_enabled ||
          cfg.numerics.hydro.hourglass.enabled ||
          cfg.numerics.hydro.av_model != AvModel::ScalarVnrLegacy ||
          cfg.numerics.hydro.time_integration ==
              HydroTimeIntegration::MidpointV1);
}

std::size_t compatible_edge_buffer_size(const Config& cfg,
                                        const std::size_t n_cells,
                                        const std::size_t n_nodes) {
  if (cfg.main.dim != 2) {
    return 0U;
  }
  if (cfg.mesh.topology_scheme == TopologyScheme::SINGLE_BLOCK) {
    return static_cast<std::size_t>(cfg.mesh.nr) *
               static_cast<std::size_t>(cfg.mesh.nz + 1) +
           static_cast<std::size_t>(cfg.mesh.nr + 1) *
               static_cast<std::size_t>(cfg.mesh.nz);
  }
  return (n_cells + n_nodes > 0U) ? (n_cells + n_nodes - 1U) : 0U;
}

template <typename Tag>
void zero_field_cells(Field1D<Tag>& field,
                      const std::vector<std::size_t>& cells,
                      const std::size_t stride = 1U) {
  if (field.empty() || cells.empty() || stride == 0U ||
      field.size() % stride != 0U) {
    return;
  }
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  const std::size_t n_cells = field.size() / stride;
  for (const std::size_t c : cells) {
    if (c >= n_cells) {
      continue;
    }
    for (std::size_t k = 0; k < stride; ++k) {
      host[c * stride + k] = 0.0;
    }
  }
  field.copy_from_host(host.data());
}

void zero_button_dormant_cell_state(State& state,
                                    const std::vector<std::size_t>& cells) {
  if (cells.empty()) {
    return;
  }

  zero_field_cells(state.rho, cells);
  zero_field_cells(state.mass, cells);
  zero_field_cells(state.ee, cells);
  zero_field_cells(state.ei, cells);
  zero_field_cells(state.Pe, cells);
  zero_field_cells(state.Pi, cells);
  zero_field_cells(state.Qvisc, cells);
  zero_field_cells(state.zbar, cells);
  zero_field_cells(state.Te, cells);
  zero_field_cells(state.Ti, cells);
  zero_field_cells(state.hllc_mom_z_cell, cells);

  zero_field_cells(state.corner_mass, cells,
                   static_cast<std::size_t>(state.corner_stride));
  zero_field_cells(state.subzonal_mass_corner0, cells);
  zero_field_cells(state.subzonal_mass_corner1, cells);
  zero_field_cells(state.subzonal_mass_corner2, cells);
  zero_field_cells(state.subzonal_mass_corner3, cells);
  zero_field_cells(state.work_p_per_cell, cells);
  zero_field_cells(state.work_sub_per_cell, cells);
  zero_field_cells(state.work_av_per_cell, cells);
  zero_field_cells(state.state_supply_pre_rho, cells);
  zero_field_cells(state.state_supply_pre_mass, cells);
  zero_field_cells(state.state_supply_pre_ee, cells);
  zero_field_cells(state.state_supply_pre_ei, cells);
  zero_field_cells(state.state_supply_pre_uz, cells);

  const std::size_t n_cells = state.rho.size();
  if (n_cells > 0U) {
    if (state.rad_E.size() % n_cells == 0U) {
      zero_field_cells(state.rad_E, cells, state.rad_E.size() / n_cells);
    }
    if (state.mass_per_material.size() % n_cells == 0U) {
      zero_field_cells(state.mass_per_material, cells,
                       state.mass_per_material.size() / n_cells);
    }
    if (state.Ee_per_material.size() % n_cells == 0U) {
      zero_field_cells(state.Ee_per_material, cells,
                       state.Ee_per_material.size() / n_cells);
    }
    if (state.Ei_per_material.size() % n_cells == 0U) {
      zero_field_cells(state.Ei_per_material, cells,
                       state.Ei_per_material.size() / n_cells);
    }
  }
}

void apply_button_dormant_storage_mask(State& state,
                                       const tenryu::mesh::Mesh& mesh) {
  if (!mesh.button_center || !mesh.button_center->enabled) {
    return;
  }
  std::vector<std::size_t> dormant_cells;
  bool hydro_active_wrote = false;
  for (int c = 0; c < mesh.topo.n_cells; ++c) {
    if (!mesh.is_dormant_cell(c)) {
      continue;
    }
    const std::size_t c_u = static_cast<std::size_t>(c);
    dormant_cells.push_back(c_u);
    if (c_u < state.cell_is_void.size()) {
      state.cell_is_void[c_u] = static_cast<std::uint8_t>(1);
    }
    if (c_u < state.hydro_active.size()) {
      state.hydro_active[c_u] = static_cast<std::int8_t>(0);
      hydro_active_wrote = true;
    }
  }
  if (hydro_active_wrote) {
    state.note_hydro_active_host_write();
  }
  zero_button_dormant_cell_state(state, dormant_cells);
}

void apply_button_dormant_storage_mask(State& state,
                                       const Config& cfg,
                                       const std::size_t n_cells) {
  if (cfg.main.dim != 2 || cfg.mesh.polar_center_treatment != "button") {
    return;
  }
  tenryu::mesh::Mesh mesh;
  mesh.topo.nr = cfg.mesh.nr;
  mesh.topo.nz = cfg.mesh.nz;
  mesh.topo.n_cells = static_cast<int>(n_cells);
  mesh.button_center = tenryu::mesh::ButtonCenterTopology{
      true,
      cfg.mesh.center_button_outer_node_ring,
  };
  apply_button_dormant_storage_mask(state, mesh);
}

}  // namespace

const std::int8_t* State::hydro_active_device_ptr() const {
  if (hydro_active.empty()) {
    return nullptr;
  }
  const std::size_t n = hydro_active.size();
  if (hydro_active_device.size() != n) {
    hydro_active_device = DeviceArray<std::int8_t>(n);
    hydro_active_device_version = hydro_active_host_version - 1;
  }
  if (hydro_active_device_version != hydro_active_host_version) {
    const cudaError_t err =
        cudaMemcpy(hydro_active_device.data(), hydro_active.data(),
                   n * sizeof(std::int8_t), cudaMemcpyHostToDevice);
    TENRYU_ASSERT(err == cudaSuccess, "State: hydro_active mirror upload failed");
    hydro_active_device_version = hydro_active_host_version;
  }
  return hydro_active_device.data();
}

State State::allocate(const Config& cfg) {
  return allocate(cfg, cfg.numerics.T_start_eV);
}

State State::allocate(const Config& cfg, const double hydro_t_start_eV) {
  State state;
  state.corner_stride = corner_stride_for_scheme(cfg.mesh.topology_scheme);

  TENRYU_ASSERT(cfg.mesh.nr > 0, "Mesh.nr must be > 0");
  TENRYU_ASSERT(cfg.main.dim == 1 || cfg.main.dim == 2,
                "Main.dim must be 1 or 2");

  const std::size_t nr = static_cast<std::size_t>(cfg.mesh.nr);
  const std::size_t nz = static_cast<std::size_t>(cfg.main.dim == 2 ? cfg.mesh.nz : 1);
  const std::size_t n_cells =
      (cfg.main.dim == 2)
          ? static_cast<std::size_t>(
                ::tenryu::mesh::mesh_topo_n_cells_total(cfg.mesh))
          : nr;
  const std::size_t n_nodes =
      (cfg.main.dim == 2)
          ? static_cast<std::size_t>(
                ::tenryu::mesh::mesh_topo_n_nodes_total(cfg.mesh))
          : (nr + 1U);
  // S_N face-flux sizing is structured-only; config validation rejects pentagon-belt S_N.
  const std::size_t n_face_flux_total =
      (cfg.main.dim == 1) ? (nr + 1U) : ((nr + 1U) * nz + nr * (nz + 1U));
  const std::size_t n_mat = cfg.materials.materials.size();
  const std::size_t n_groups = static_cast<std::size_t>(cfg.radiation.groups);

  if (cfg.main.dim == 2 &&
      cfg.mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK) {
    state.mesh.topo.nr = cfg.mesh.nr;
    state.mesh.topo.nz = cfg.mesh.nz;
    state.mesh.topo.n_cells = static_cast<int>(n_cells);
    state.mesh.topo.n_nodes = static_cast<int>(n_nodes);
    state.mesh.topo.multiblock =
        ::tenryu::mesh::mesh_topo_make_empty_multiblock_topology(cfg.mesh);
  }

  // W-G: bind the 1D coordinate geometry here as well as at the full mesh
  // construction sites (mesh.cu keeps the same ternary) so states built
  // without a Mesh pass (verify gates, unit tests) do not silently run
  // spherical when the config asks for cylindrical/planar. The legacy B1
  // selector Main.dimension="1D_CYL" must also map to cylindrical here,
  // matching mesh.cu — otherwise a 1D_CYL config that never passes through
  // create_mesh() runs spherical geometry.
  const bool legacy_cylindrical_1d =
      cfg.main.dim == 1 && cfg.main.dimension == "1D_CYL";
  state.mesh.geometry_code =
      (legacy_cylindrical_1d || cfg.mesh.geometry_1d == "cylindrical")
          ? 1
          : ((cfg.mesh.geometry_1d == "planar") ? 2 : 0);

  state.rho.reset(n_cells);
  state.mass.reset(n_cells);
  if (cfg.main.dim == 2 &&
      (cfg.numerics.hydro.hourglass.enabled ||
       cfg.numerics.hydro.subzonal_mass_enabled)) {
    state.subzonal_mass_corner0.reset(n_cells);
    state.subzonal_mass_corner1.reset(n_cells);
    state.subzonal_mass_corner2.reset(n_cells);
    state.subzonal_mass_corner3.reset(n_cells);
  }
  state.vol.reset(n_cells);
  state.cell_vol_initial.reset(n_cells);
  state.center_patch_latch.assign(n_cells, static_cast<std::uint8_t>(0));
  state.vol_prev_hydro.reset(n_cells);
  state.zbar.reset(n_cells);
  state.Te.reset(n_cells);
  state.Ti.reset(n_cells);
  state.ee.reset(n_cells);
  state.ei.reset(n_cells);
  state.Pe.reset(n_cells);
  state.Pi.reset(n_cells);
  state.Qvisc.reset(n_cells);
  state.zmom_r2.reset(n_cells);
  state.zmom_r4.reset(n_cells);
  state.zmom_r2.fill(1.0);
  state.zmom_r4.fill(1.0);
  if (compatible_force_work_buffers_requested(cfg)) {
    const std::size_t n_corners =
        n_cells * static_cast<std::size_t>(state.corner_stride);
    const std::size_t n_edges =
        compatible_edge_buffer_size(cfg, n_cells, n_nodes);
    state.corner_force_p_r.reset(n_corners);
    state.corner_force_p_z.reset(n_corners);
    state.corner_force_sub_r.reset(n_corners);
    state.corner_force_sub_z.reset(n_corners);
    if (cfg.numerics.hydro.time_integration ==
        HydroTimeIntegration::MidpointV1) {
      state.corner_force_q_r.reset(n_corners);
      state.corner_force_q_z.reset(n_corners);
    }
    state.edge_force_av_r.reset(n_edges);
    state.edge_force_av_z.reset(n_edges);
    state.work_p_per_cell.reset(n_cells);
    state.work_sub_per_cell.reset(n_cells);
    state.work_av_per_cell.reset(n_cells);
  }
  if (cfg.numerics.hydro.hllc_z_flux_2d_rz) {
    state.hllc_mom_z_cell.reset(n_cells);
    state.hllc_mom_z_cell_initialized = false;
  }
  if (cfg.numerics.hydro.adaptive_av.enabled) {
    state.adaptive_av_gate.reset(n_cells);
    state.adaptive_av_gate.fill(0.0);
  }
  if (cfg.numerics.hydro.post_shock_heat ||
      cfg.numerics.hydro.post_shock_velocity_damping_C > 0.0 ||
      (cfg.numerics.hydro.adaptive_av.enabled &&
       (cfg.numerics.hydro.adaptive_av.primary.Cpsv > 0.0 ||
        cfg.numerics.hydro.adaptive_av.rebound.Cpsv > 0.0))) {
    state.shock_time.reset(n_cells);
    state.shock_time.fill(kShockTimeInactive);
  }
  if (cfg.numerics.hydro.compatible_energy) {
    state.eta_compatible.reset(n_cells);
  }

  state.volFrac.reset(n_cells * n_mat);
  if (cfg.numerics.diagnostics.hotspot_gas.enabled) {
    state.gas_tracer_Y.reset(n_cells);
  }
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    const std::size_t n_cell_mat = n_cells * n_mat;
    state.mass_per_material.reset(n_cell_mat);
    state.Ee_per_material.reset(n_cell_mat);
    state.Ei_per_material.reset(n_cell_mat);
    if (cfg.numerics.materials.lazy_cache_te_m_enabled) {
      state.Te_per_material.reset(n_cell_mat);
      state.Ti_per_material.reset(n_cell_mat);
      state.Te_per_material_valid.assign(n_cell_mat, static_cast<std::uint8_t>(0));
      state.Ti_per_material_valid.assign(n_cell_mat, static_cast<std::uint8_t>(0));
    }
  }

  state.x_r.reset(n_nodes);
  state.x_z.reset(n_nodes);
  state.x_r_initial.reset(n_nodes);
  state.x_z_initial.reset(n_nodes);
  state.x_r_reference.reset(n_nodes);
  state.x_z_reference.reset(n_nodes);
  state.x_r_shock_target.reset(n_nodes);
  state.x_z_shock_target.reset(n_nodes);
  state.ref_xi_node.reset(0);
  state.ref_xi_cell.reset(0);
  state.ref_dir0_r.reset(0);
  state.ref_dir0_z.reset(0);
  state.v_r.reset(n_nodes);
  state.v_z.reset(n_nodes);

  state.rad_E.reset(n_cells * n_groups);
  state.rad_E_old.reset(n_cells * n_groups);
  state.rad_dep.reset(n_cells * n_groups);
  state.rad_emit.reset(n_cells * n_groups);
  state.fld_sigma_a.reset(n_cells * n_groups);
  state.fld_sigma_pe.reset(n_cells * n_groups);
  state.fld_sigma_R.reset(n_cells * n_groups);
  state.fld_eta.reset(n_cells * n_groups);
  state.fld_D_cell.reset(n_cells * n_groups);
  state.fld_lower.reset(n_cells * n_groups);
  state.fld_diag.reset(n_cells * n_groups);
  state.fld_upper.reset(n_cells * n_groups);
  state.fld_rhs.reset(n_cells * n_groups);
  state.fld_Te_old.reset(n_cells);
  state.fld_delta_T.reset(n_cells);
  state.fld_fleck.reset(n_cells);
  state.sn_sigma_a.reset(n_cells * n_groups);
  state.sn_rad_E_pre_newton.reset(n_cells * n_groups);
  state.sn_sigma_pe.reset(n_cells * n_groups);
  state.sn_sigma_s.reset(n_cells * n_groups);
  state.sn_eta.reset(n_cells * n_groups);
  state.sn_phi_old.reset(n_cells * n_groups);
  state.sn_phi_new.reset(n_cells * n_groups);
  state.sn_phi_sweep.reset(n_cells * n_groups);
  state.sn_diag_rad_E_pre.reset(n_cells * n_groups);
  state.sn_diag_rad_E_post.reset(n_cells * n_groups);
  state.sn_diag_rad_emission_at_Tn.reset(n_cells * n_groups);
  state.sn_diag_rad_emission_at_Tnp1.reset(n_cells * n_groups);
  state.sn_diag_rad_absorption.reset(n_cells * n_groups);
  state.sn_diag_clip_energy.reset(n_cells * n_groups);
  state.sn_diag_clip_full_deficit.reset(n_cells * n_groups);
  state.sn_diag_chi_opacity.reset(n_cells * n_groups);
  state.sn_diag_F_first_moment.reset(n_cells * n_groups);
  state.sn_face_flux_raw.reset(n_face_flux_total * n_groups);
  state.sn_face_flux_diff.reset(n_face_flux_total * n_groups);
  state.sn_face_flux_blended.reset(n_face_flux_total * n_groups);
  state.sn_face_flux_limited.reset(n_face_flux_total * n_groups);
  state.sn_face_alpha.reset(n_face_flux_total * n_groups);
  state.sn_stream_theta.reset(n_cells * n_groups);
  state.sn_E_star_flux.reset(n_cells * n_groups);
  state.sn_diag_E_star_flux.reset(n_cells * n_groups);
  state.sn_origin_boundary.reset(n_cells * n_groups);
  state.sn_radial_fixup_count.reset(n_cells * n_groups);
  state.sn_radial_fixup_artificial_abs.reset(n_cells * n_groups);
  state.sn_angular_fixup_count.reset(n_cells * n_groups);
  state.sn_angular_fixup_artificial_abs.reset(n_cells * n_groups);
  state.sn_Prr.reset(n_cells * n_groups);
  state.sn_chi.reset(n_cells * n_groups);
  state.sn_F_z.reset(n_cells * n_groups);
  state.sn_P_zz.reset(n_cells * n_groups);
  state.sn_chi_z.reset(n_cells * n_groups);
  state.sn_dsa_lower.reset(n_cells * n_groups);
  state.sn_dsa_diag.reset(n_cells * n_groups);
  state.sn_dsa_upper.reset(n_cells * n_groups);
  state.sn_dsa_rhs.reset(n_cells * n_groups);
  state.sn_Te_old.reset(n_cells);
  state.sn_delta_T.reset(n_cells);
  state.sn_tau_R.reset(n_cells);
  state.sn_reduced_flux.reset(n_cells);
  state.sn_ap_alpha.reset(n_cells);
  state.ddmc_mode_map.assign(n_cells * n_groups, static_cast<std::int8_t>(0));
  state.ddmc_mode_map_valid = false;
  state.particle_sort_cache_invalidated = false;
  state.holo_core_mask.assign(n_cells, static_cast<std::uint8_t>(0));
  state.holo_patch_mask.assign(n_cells, static_cast<std::uint8_t>(0));
  state.holo_core_prev_mask.assign(n_cells, static_cast<std::uint8_t>(0));
  state.holo_hold_count.assign(n_cells, 0);
  state.holo_dwell_count.assign(n_cells, 0);
  state.holo_tau_R.assign(n_cells, 0.0);
  state.holo_reduced_flux.assign(n_cells, 0.0);
  state.holo_mass_q.assign(n_cells, 0.0);
  state.holo_lo_weight.assign(n_cells, 0.0);
  state.holo_E_LO.reset(n_cells * n_groups);
  state.holo_consistency_source.reset(n_cells * n_groups);
  state.holo_rad_dep.reset(n_cells * n_groups);
  state.holo_rad_emit.reset(n_cells * n_groups);
  state.holo_Prr.reset(n_cells * n_groups);
  state.holo_chi.reset(n_cells * n_groups);
  state.holo_chi_filtered.reset(n_cells * n_groups);
  state.holo_Prr_coverage.reset(n_cells * n_groups);
  state.holo_core_mask_valid = false;
  state.holo_lo_source_valid = false;
  state.holo_ale_invalidated = false;
  state.fld_outer_iterations = 0;
  state.fld_converged = false;
  state.fld_outer_residual = 0.0;
  state.cg_cap_exit_unconverged = 0;
  state.newton_cap_exit_unconverged = 0;
  state.fld_escaped_step = 0.0;
  state.fld_marshak_in_step = 0.0;
  state.fld_state_supply_in_cumulative = 0.0;
  state.fld_state_supply_out_cumulative = 0.0;
  state.fld_state_supply_net_cumulative = 0.0;
  state.fld_state_supply_in_step = 0.0;
  state.fld_state_supply_out_step = 0.0;
  state.fld_state_supply_net_step = 0.0;
  state.sn_outer_iterations = 0;
  state.sn_inner_iterations = 0;
  state.sn_converged = false;
  state.sn_outer_stagnated = false;
  state.sn_outer_residual = 0.0;
  state.sn_inner_residual = 0.0;
  state.sn_escaped_step = 0.0;
  state.sn_marshak_in_step = 0.0;
  state.sn_void_anchor_dE_step = 0.0;
  state.sn_void_anchor_dE_abs_step = 0.0;
  state.sn_ap_alpha_max = 0.0;
  state.sn_ap_alpha_active_faces = 0.0;
  state.laser_dep.reset(n_cells);
  state.ray_density.reset(n_cells);
  state.laser_waveforms.resize(cfg.laser.beams.size());

  const std::int8_t hydro_init = (hydro_t_start_eV == 0.0) ? 1 : 0;
  state.hydro_active.resize(n_cells, hydro_init);
  state.note_hydro_active_host_write();
  state.state_supply_mask.assign(n_cells, static_cast<std::int8_t>(0));
  state.state_supply_pre_rho.reset(n_cells);
  state.state_supply_pre_mass.reset(n_cells);
  state.state_supply_pre_ee.reset(n_cells);
  state.state_supply_pre_ei.reset(n_cells);
  state.state_supply_pre_uz.reset(n_cells);
  state.state_supply_dM_cumulative = 0.0;
  state.state_supply_dE_cumulative = 0.0;
  state.state_supply_dPz_cumulative = 0.0;
  state.state_supply_dM_step = 0.0;
  state.state_supply_dE_step = 0.0;
  state.state_supply_dPz_step = 0.0;
  state.cell_is_void.assign(n_cells, static_cast<std::uint8_t>(0));
  apply_button_dormant_storage_mask(state, cfg, n_cells);
  state.hydro_t_start_eV = hydro_t_start_eV;

  state.t = 0.0;
  state.step = 0;
  state.dt = 0.0;
  state.dt_prev_hydro = -1.0;
  state.trace_mesh_outer_node = -1;
  state.trace_mesh_outer_cell = -1;
  state.ale_rezoned = false;
  state.ale_rezone_invocations = 0;
  state.ale_remaps_applied = 0;
  state.ale_last_applied_step = -1;
  state.ale1d_floor_cooldown_remaining = 0;
  state.diff_ref_diag_baseline_initialized = false;
  state.diff_ref_diag_gas_mesh_volume_initial = 0.0;
  state.diff_ref_diag_gas_rho_p50_initial = 0.0;
  state.multiblock_path_dt_rejection_count = 0;
  state.multiblock_path_dt_rejection_exhausted_count = 0;
  state.multiblock_path_dt_min_accepted =
      std::numeric_limits<double>::infinity();
  state.plic_remap_sticky_fallback = false;
  state.plic_consecutive_drift_triggers = 0;
  state.plic_last_reconstruction_step = -1;
  state.E_safety = 0.0;
  state.E_numerical_loss = 0.0;
  state.E_laser_deposited = 0.0;
  state.E_laser_escaped = 0.0;
  state.E_laser_incident = 0.0;
  state.E_ra_deposited = 0.0;
  state.E_cbet_iaw_step = 0.0;
  state.E_cbet_iaw = 0.0;
  state.E_rad_escaped = 0.0;
  state.E_floor_injected = 0.0;
  state.E_pdV_bdry = 0.0;
  state.E_Marshak_in = 0.0;
  state.E_solver = 0.0;
  state.radiation_device_flags = DeviceErrorFlags{};
  state.corner_mass_initialized = false;
  state.corner_mass_is_lagrangian_invariant = false;
  state.dispatch_counters.reset();
  state.tri_fan_center_perturbation_diag.reset();
  state.count_edge_compressive_edges_step = 0;
  state.max_corner_density_spread_step = 0.0;
  state.max_corner_density_spread_run = 0.0;
  state.max_subzonal_merit_step = 0.0;
  state.subzonal_nonzero_force_cells_step = 0;
  state.max_abs_subzonal_force_step = 0.0;
  state.max_abs_subzonal_work_step = 0.0;
  state.max_corner_density_spread_step_idx = -1;
  state.max_corner_density_spread_cell_id = -1;
  state.max_corner_density_spread_corner_id = -1;
  state.t_next_plot = -1.0;
  state.t_next_history = -1.0;
  state.t_next_checkpoint = -1.0;
  state.pressure_drive_1d.reset();
  state.marshak_Tr_1d.reset();
  state.hot_e_eta_1d.reset();
  state.hot_e_eta_ch_1d.clear();
  state.hot_e_eta_state_eta.clear();
  state.hot_e_eta_state_kappa_bar.clear();
  state.hot_e_eta_prev_Pcross.clear();
  state.hot_e_eta_prev_rbar.clear();
  state.hot_e_eta_prev_valid.clear();
  state.hot_e_eta_diag_g.clear();
  state.hot_e_eta_diag_eta_eq.clear();
  state.hot_e_eta_diag_tau_s.clear();
  state.hot_e_eta_diag_I14.clear();
  state.hot_e_eta_diag_I14_lower.clear();
  state.hot_e_eta_diag_I14_upper.clear();
  state.hot_e_eta_diag_n_sigma.clear();
  state.hot_e_eta_diag_Te_keV.clear();
  state.hot_e_eta_diag_Ln_um.clear();
  state.hot_e_eta_diag_clamped.clear();
  state.cbet_gross_exchange.clear();
  state.cbet_net_to_inbound.clear();
  state.ps_sky_mu.clear();
  state.ps_sky_phi.clear();
  state.ps_sky_I_tot.clear();
  state.ps_sky_I_cw.clear();
  state.ps_sky_n_sigma.clear();
  state.ps_ray_map.clear();
  state.ps_ray_map_shell_r.clear();
  state.ps_port_outgoing_power.clear();
  state.ps_port_capture_pcross.clear();
  state.ps_f_illum2 = 0.0;
  state.ps_f_union = 0.0;
  state.hot_e_enabled_any = false;
  state.hot_e_in_step = 0.0;
  state.hot_e_deposited_step = 0.0;
  state.hot_e_residual_step = 0.0;
  state.hot_e_escaped_step = 0.0;
  state.hot_e_source_r = 0.0;
  state.hot_e_conservation_resid = 0.0;
  state.hot_e_dt_limit_s = std::numeric_limits<double>::infinity();
  state.hot_e_Q_host.clear();
  state.hot_e_eps_cum_host.clear();
  state.hot_e_ch_in_step.clear();
  state.hot_e_ch_deposited_step.clear();
  state.hot_e_ch_escaped_step.clear();
  state.E_hot_e_deposited = 0.0;
  state.E_hot_e_escaped = 0.0;
  state.E_burn_inflight = 0.0;
  state.burn_diffusion_any = false;
  state.burn_mc_any = false;
  state.burn_mc_live = 0;
  state.marshak_Tr_face_tables.clear();

  return state;
}

void State::invalidate_cell_material_props() {
  A_eff.reset(0);
  gamma_eff.reset(0);
  kappa_planck_eff.reset(0);
  kappa_rosseland_eff.reset(0);
  cell_material_index.reset(0);
}

void State::ensure_cell_material_props(const Config& cfg) {
  const std::size_t n_cells = rho.size();
  if (n_cells == 0U) {
    return;
  }
  if (A_eff.size() == n_cells && gamma_eff.size() == n_cells &&
      kappa_planck_eff.size() == n_cells &&
      kappa_rosseland_eff.size() == n_cells &&
      cell_material_index.size() == n_cells) {
    return;
  }
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "ensure_cell_material_props requires at least one material");
  // Mixing rule and floors are kept numerically identical to the historic
  // conduction-local implementation (harmonic A / linear gamma over positive
  // volfrac weights) so single-material and legacy paths reproduce the same
  // values bit-for-bit.
  constexpr double kMinA = 1.0e-12;
  constexpr double kMinGamma = 1.0 + 1.0e-12;
  const auto& materials = cfg.materials.materials;
  const auto& mat0 = materials.front();
  const double A0 = std::max(mat0.A, kMinA);
  const double gamma0 = std::max(mat0.ideal_gas_gamma, kMinGamma);

  std::vector<double> host_A(n_cells, A0);
  std::vector<double> host_gamma(n_cells, gamma0);

  const std::size_t n_mat = materials.size();
  const std::size_t expected = n_cells * n_mat;
  std::size_t d0 = 0U;
  for (std::size_t m = 0; m < n_mat; ++m) {
    if (!materials[m].is_void) {
      d0 = m;
      break;
    }
  }
  std::vector<int> host_mat_index(n_cells, static_cast<int>(d0));
  if (n_mat > 1U && volFrac.size() == expected) {
    std::vector<double> host_volfrac(expected, 0.0);
    volFrac.copy_to_host(host_volfrac.data());

    std::vector<double> inv_A_m(n_mat, 0.0);
    std::vector<double> gamma_m(n_mat, gamma0);
    for (std::size_t m = 0; m < n_mat; ++m) {
      inv_A_m[m] = 1.0 / std::max(materials[m].A, kMinA);
      gamma_m[m] = std::max(materials[m].ideal_gas_gamma, kMinGamma);
    }
    for (std::size_t c = 0; c < n_cells; ++c) {
      const std::size_t base = c * n_mat;
      double frac_sum = 0.0;
      double inv_A_c = 0.0;
      double gamma_c = 0.0;
      double max_non_void_frac = 0.0;
      for (std::size_t m = 0; m < n_mat; ++m) {
        const double frac_raw = host_volfrac[base + m];
        const double frac =
            (std::isfinite(frac_raw) && frac_raw > 0.0) ? frac_raw : 0.0;
        if (!materials[m].is_void && frac > max_non_void_frac) {
          max_non_void_frac = frac;
          host_mat_index[c] = static_cast<int>(m);
        }
        frac_sum += frac;
        inv_A_c += frac * inv_A_m[m];
        gamma_c += frac * gamma_m[m];
      }
      if (frac_sum > 1.0e-30) {
        inv_A_c /= frac_sum;
        gamma_c /= frac_sum;
      }
      if (std::isfinite(inv_A_c) && inv_A_c > 1.0e-30) {
        host_A[c] = std::max(1.0 / inv_A_c, kMinA);
      }
      if (std::isfinite(gamma_c) && gamma_c > 0.0) {
        host_gamma[c] = std::max(gamma_c, kMinGamma);
      }
    }
  }

  const double kappa0 = mat0.kappa_a_constant;
  std::vector<double> host_kappa_planck(n_cells, kappa0);
  std::vector<double> host_kappa_rosseland(n_cells, kappa0);
  // Diagnostic-only harmonic opacity mixing override, spec §5b A1.
  static const bool use_harmonic_opacity_mix = [] {
    const char* mix = std::getenv("TENRYU_MM_OPACITY_MIX");
    return mix != nullptr && std::strcmp(mix, "harmonic") == 0;
  }();
  constexpr double kMinOpacityDenom = 1.0e-300;
  std::vector<double> kappa_m(n_mat, kappa0);
  for (std::size_t m = 0; m < n_mat; ++m) {
    kappa_m[m] = materials[m].kappa_a_constant;
  }
  if (mass_per_material.size() == expected) {
    std::vector<double> host_mass(expected, 0.0);
    mass_per_material.copy_to_host(host_mass.data());
    for (std::size_t c = 0; c < n_cells; ++c) {
      const std::size_t base = c * n_mat;
      double mass_sum = 0.0;
      for (std::size_t m = 0; m < n_mat; ++m) {
        mass_sum += host_mass[base + m];
      }
      const double mass_denom = std::max(mass_sum, kMinOpacityDenom);
      double kappa_c = 0.0;
      double inv_kappa_c = 0.0;
      for (std::size_t m = 0; m < n_mat; ++m) {
        const double w = host_mass[base + m] / mass_denom;
        if (use_harmonic_opacity_mix) {
          inv_kappa_c += w / std::max(kappa_m[m], kMinOpacityDenom);
        } else {
          kappa_c += w * kappa_m[m];
        }
      }
      if (use_harmonic_opacity_mix) {
        kappa_c =
            (std::isfinite(inv_kappa_c) && inv_kappa_c > 0.0) ? 1.0 / inv_kappa_c : 0.0;
      }
      host_kappa_planck[c] = kappa_c;
      host_kappa_rosseland[c] = kappa_c;
    }
  } else if (n_mat > 1U && volFrac.size() == expected) {
    std::vector<double> host_volfrac(expected, 0.0);
    volFrac.copy_to_host(host_volfrac.data());
    for (std::size_t c = 0; c < n_cells; ++c) {
      const std::size_t base = c * n_mat;
      double frac_sum = 0.0;
      double kappa_c = 0.0;
      double inv_kappa_c = 0.0;
      for (std::size_t m = 0; m < n_mat; ++m) {
        if (materials[m].is_void) {
          continue;
        }
        const double frac_raw = host_volfrac[base + m];
        const double frac =
            (std::isfinite(frac_raw) && frac_raw > 0.0) ? frac_raw : 0.0;
        frac_sum += frac;
        if (use_harmonic_opacity_mix) {
          inv_kappa_c += frac / std::max(kappa_m[m], kMinOpacityDenom);
        } else {
          kappa_c += frac * kappa_m[m];
        }
      }
      if (frac_sum > 1.0e-30) {
        if (use_harmonic_opacity_mix) {
          inv_kappa_c /= frac_sum;
          kappa_c = (std::isfinite(inv_kappa_c) && inv_kappa_c > 0.0)
                        ? 1.0 / inv_kappa_c
                        : 0.0;
        } else {
          kappa_c /= frac_sum;
        }
        host_kappa_planck[c] = kappa_c;
        host_kappa_rosseland[c] = kappa_c;
      }
    }
  }

  A_eff.reset(n_cells);
  gamma_eff.reset(n_cells);
  kappa_planck_eff.reset(n_cells);
  kappa_rosseland_eff.reset(n_cells);
  cell_material_index.reset(n_cells);
  A_eff.copy_from_host(host_A.data());
  gamma_eff.copy_from_host(host_gamma.data());
  kappa_planck_eff.copy_from_host(host_kappa_planck.data());
  kappa_rosseland_eff.copy_from_host(host_kappa_rosseland.data());
  cell_material_index.copy_from_host(host_mat_index.data());
}

void State::reset() {
  rho.fill(0.0);
  mass.fill(0.0);
  corner_mass.fill(0.0);
  corner_mass_initialized = false;
  corner_mass_is_lagrangian_invariant = false;
  corner_volume.fill(0.0);
  corner_pressure.fill(0.0);
  subzonal_mass_corner0.fill(0.0);
  subzonal_mass_corner1.fill(0.0);
  subzonal_mass_corner2.fill(0.0);
  subzonal_mass_corner3.fill(0.0);
  vol.fill(0.0);
  cell_vol_initial.fill(0.0);
  std::fill(center_patch_latch.begin(), center_patch_latch.end(),
            static_cast<std::uint8_t>(0));
  vol_prev_hydro.fill(0.0);
  zbar.fill(0.0);
  Te.fill(0.0);
  Ti.fill(0.0);
  ee.fill(0.0);
  ei.fill(0.0);
  Pe.fill(0.0);
  Pi.fill(0.0);
  Qvisc.fill(0.0);
  zmom_r2.fill(1.0);
  zmom_r4.fill(1.0);
  corner_force_p_r.fill(0.0);
  corner_force_p_z.fill(0.0);
  corner_force_sub_r.fill(0.0);
  corner_force_sub_z.fill(0.0);
  corner_force_q_r.fill(0.0);
  corner_force_q_z.fill(0.0);
  edge_force_av_r.fill(0.0);
  edge_force_av_z.fill(0.0);
  work_p_per_cell.fill(0.0);
  work_sub_per_cell.fill(0.0);
  work_av_per_cell.fill(0.0);
  tri_fan_center_perturbation_diag.reset();
  pole_angular_derefine = PoleAngularDerefineState{};
  ring7_seam_rezone_requested = false;
  ring7_seam_rezone_request_cell = -1;
  ring7_pole_cap_oracle_requested = false;
  ring7_pole_cap_request_cell = -1;
  ring7_last_failed_velocity_valid = false;
  ring7_last_failed_cell = -1;
  ring7_last_failed_sigma_safe = 1.0;
  ring7_last_failed_dt = 0.0;
  ring7_last_failed_v_r.fill(0.0);
  ring7_last_failed_v_z.fill(0.0);
  ring7_pole_cap_validation_pending = false;
  ring7_pole_cap_validation_cell = -1;
  ring7_pole_cap_validation_dt = 0.0;
  ring7_pole_cap_eta_prod_proxy = 0.0;
  count_edge_compressive_edges_step = 0;
  max_corner_density_spread_step = 0.0;
  max_corner_density_spread_run = 0.0;
  max_subzonal_merit_step = 0.0;
  subzonal_nonzero_force_cells_step = 0;
  max_abs_subzonal_force_step = 0.0;
  max_abs_subzonal_work_step = 0.0;
  max_corner_density_spread_step_idx = -1;
  max_corner_density_spread_cell_id = -1;
  max_corner_density_spread_corner_id = -1;
  hllc_mom_z_cell.fill(0.0);
  hllc_mom_z_cell_initialized = false;
  shock_time.fill(kShockTimeInactive);
  adaptive_av_gate.fill(0.0);
  gas_tracer_Y.fill(0.0);
  gas_tracer_initialized = false;
  gas_tracer_mass_initial = 0.0;
  hotspot_rho_bar_initial = std::numeric_limits<double>::quiet_NaN();
  hotspot_cr50_initial = std::numeric_limits<double>::quiet_NaN();
  hotspot_rho50_initial = std::numeric_limits<double>::quiet_NaN();
  hotspot_internal_energy_initial_erg = std::numeric_limits<double>::quiet_NaN();
  hotspot_kinetic_energy_initial_erg = std::numeric_limits<double>::quiet_NaN();
  diff_ref_diag_baseline_initialized = false;
  diff_ref_diag_gas_mesh_volume_initial = 0.0;
  diff_ref_diag_gas_rho_p50_initial = 0.0;
  adaptive_av_r0 = 0.0;
  adaptive_av_last_rs = 0.0;
  adaptive_av_last_us = 0.0;
  adaptive_av_rs_min = std::numeric_limits<double>::infinity();
  adaptive_av_tracker_steps = 0;
  adaptive_av_mode = 0;
  adaptive_av_tracker_valid = false;
  adaptive_av_bounce_seen = false;
  eta_compatible.fill(0.0);
  volFrac.fill(0.0);
  mass_per_material.fill(0.0);
  Ee_per_material.fill(0.0);
  Ei_per_material.fill(0.0);
  Te_per_material.fill(0.0);
  Ti_per_material.fill(0.0);
  std::fill(Te_per_material_valid.begin(), Te_per_material_valid.end(),
            static_cast<std::uint8_t>(0));
  std::fill(Ti_per_material_valid.begin(), Ti_per_material_valid.end(),
            static_cast<std::uint8_t>(0));
  x_r.fill(0.0);
  x_z.fill(0.0);
  x_r_initial.fill(0.0);
  x_z_initial.fill(0.0);
  x_r_reference.fill(0.0);
  x_z_reference.fill(0.0);
  x_r_shock_target.fill(0.0);
  x_z_shock_target.fill(0.0);
  reference_epoch = 0;
  ref_xi_node.reset(0);
  ref_xi_cell.reset(0);
  ref_dir0_r.reset(0);
  ref_dir0_z.reset(0);
  v_r.fill(0.0);
  v_z.fill(0.0);
  axis_lagrangian_tangential_engaged_count = 0;
  axis_lagrangian_tangential_ineffective_count = 0;
  axis_lagrangian_tangential_last_sigma = 1.0;
  trace_mesh_outer_node = -1;
  trace_mesh_outer_cell = -1;
  c1_solver_steps_total = 0;
  c1_solver_residual_max = 0.0;
  c1_solver_residual_last = 0.0;
  c1_solver_iter_max = 0;
  c1_solver_iter_last = 0;
  c1_solver_cond_number_max = 0.0;
  c1_solver_cond_number_last = 0.0;
  snb_steps_total = 0;
  snb_picard_iters_last = 0;
  snb_picard_iters_max = 0;
  snb_nonconverged_steps = 0;
  snb_picard_resid_last = 0.0;
  snb_cap_faces_99_last = 0;
  snb_cap_faces_50_last = 0;
  snb_cap_theta_min_last = 1.0;
  snb_cap_theta_min_run = 1.0;
  snb_dq_over_qsh_max_last = 0.0;
  snb_dq_over_qsh_max_run = 0.0;
  snb_solver_iters = 0;
  snb_solver_resid = 0.0;
  c1_bc_heat_flux_integrated.fill(0.0);
  c1_bc_heat_flux_last.fill(0.0);
  reference_barrier_engaged_count = 0;
  reference_barrier_succeeded_count = 0;
  reference_barrier_last_lambda = 1.0;
  driver_retry_reference_barrier_attempts = 0;
  driver_retry_reference_barrier_successes = 0;
  driver_retry_reference_barrier_lambda_collapse_events = 0;
  driver_retry_reference_barrier_same_sig_aborts = 0;
  driver_retry_reference_barrier_dt_collapse_aborts = 0;
  driver_retry_reference_barrier_max_attempts_aborts = 0;
  driver_retry_reference_barrier_quality_stagnation_aborts = 0;
  driver_retry_reference_barrier_rezone_recent_steps.fill(0);
  driver_retry_reference_barrier_recent_idx = 0;
  rad_E.fill(0.0);
  rad_E_old.fill(0.0);
  rad_dep.fill(0.0);
  rad_emit.fill(0.0);
  fld_sigma_a.fill(0.0);
  fld_sigma_pe.fill(0.0);
  fld_sigma_R.fill(0.0);
  fld_eta.fill(0.0);
  fld_D_cell.fill(0.0);
  fld_lower.fill(0.0);
  fld_diag.fill(0.0);
  fld_upper.fill(0.0);
  fld_rhs.fill(0.0);
  fld_Te_old.fill(0.0);
  fld_delta_T.fill(0.0);
  fld_reduction_work.fill(0.0);
  fld_cusparse_buffer.fill(0.0);
  fld_group_bounds_work.fill(0.0);
  fld_nlte_f_work.fill(0.0);
  fld_nlte_sigma_eff_work.fill(0.0);
  fld_nlte_sigma_s_eff_work.fill(0.0);
  fld_nlte_eta_cdf_work.fill(0.0);
  fld_nlte_lambda_work.fill(0.0);
  fld_fleck.fill(0.0);
  sn_sigma_a.fill(0.0);
  sn_rad_E_pre_newton.fill(0.0);
  sn_sigma_pe.fill(0.0);
  sn_sigma_s.fill(0.0);
  sn_eta.fill(0.0);
  sn_phi_old.fill(0.0);
  sn_phi_new.fill(0.0);
  sn_phi_sweep.fill(0.0);
  sn_diag_rad_E_pre.fill(0.0);
  sn_diag_rad_E_post.fill(0.0);
  sn_diag_rad_emission_at_Tn.fill(0.0);
  sn_diag_rad_emission_at_Tnp1.fill(0.0);
  sn_diag_rad_absorption.fill(0.0);
  sn_diag_clip_energy.fill(0.0);
  sn_diag_clip_full_deficit.fill(0.0);
  sn_diag_chi_opacity.fill(0.0);
  sn_diag_F_first_moment.fill(0.0);
  sn_face_flux_raw.fill(0.0);
  sn_face_flux_diff.fill(0.0);
  sn_face_flux_blended.fill(0.0);
  sn_face_flux_limited.fill(0.0);
  sn_face_alpha.fill(0.0);
  sn_stream_theta.fill(0.0);
  sn_E_star_flux.fill(0.0);
  sn_diag_E_star_flux.fill(0.0);
  sn_psi_scratch.fill(0.0);
  sn_psi_outgoing_scratch.fill(0.0);
  sn_lc_E_scratch.fill(0.0);
  sn_lc_A_scratch.fill(0.0);
  sn_origin_boundary.fill(0.0);
  sn_radial_fixup_count.fill(0.0);
  sn_radial_fixup_artificial_abs.fill(0.0);
  sn_angular_fixup_count.fill(0.0);
  sn_angular_fixup_artificial_abs.fill(0.0);
  sn_Prr.fill(0.0);
  sn_chi.fill(0.0);
  sn_F_z.fill(0.0);
  sn_P_zz.fill(0.0);
  sn_chi_z.fill(0.0);
  sn_dsa_lower.fill(0.0);
  sn_dsa_diag.fill(0.0);
  sn_dsa_upper.fill(0.0);
  sn_dsa_rhs.fill(0.0);
  sn_dsa_cusparse_buffer.fill(0.0);
  sn_group_bounds_work.fill(0.0);
  sn_nlte_f_work.fill(0.0);
  sn_nlte_sigma_eff_work.fill(0.0);
  sn_nlte_sigma_s_eff_work.fill(0.0);
  sn_nlte_eta_cdf_work.fill(0.0);
  sn_nlte_lambda_work.fill(0.0);
  sn_Te_old.fill(0.0);
  sn_delta_T.fill(0.0);
  sn_reduction_work.fill(0.0);
  sn_tau_R.fill(0.0);
  sn_reduced_flux.fill(0.0);
  sn_ap_alpha.fill(0.0);
  difference_W.reset(0);
  difference_E_ref.reset(0);
  difference_residual_E.reset(0);
  delta_E_rad_prev.reset(0);
  std::fill(ddmc_mode_map.begin(), ddmc_mode_map.end(), static_cast<std::int8_t>(0));
  ddmc_mode_map_valid = false;
  particle_sort_cache_invalidated = false;
  std::fill(holo_core_mask.begin(), holo_core_mask.end(), static_cast<std::uint8_t>(0));
  std::fill(holo_patch_mask.begin(), holo_patch_mask.end(), static_cast<std::uint8_t>(0));
  std::fill(holo_core_prev_mask.begin(), holo_core_prev_mask.end(), static_cast<std::uint8_t>(0));
  std::fill(holo_hold_count.begin(), holo_hold_count.end(), 0);
  std::fill(holo_dwell_count.begin(), holo_dwell_count.end(), 0);
  std::fill(holo_tau_R.begin(), holo_tau_R.end(), 0.0);
  std::fill(holo_reduced_flux.begin(), holo_reduced_flux.end(), 0.0);
  std::fill(holo_mass_q.begin(), holo_mass_q.end(), 0.0);
  std::fill(holo_lo_weight.begin(), holo_lo_weight.end(), 0.0);
  holo_E_LO.fill(0.0);
  holo_consistency_source.fill(0.0);
  holo_rad_dep.fill(0.0);
  holo_rad_emit.fill(0.0);
  holo_Prr.fill(0.0);
  holo_chi.fill(0.0);
  holo_chi_filtered.fill(0.0);
  holo_Prr_coverage.fill(0.0);
  holo_core_mask_valid = false;
  holo_lo_source_valid = false;
  holo_ale_invalidated = false;
  fld_outer_iterations = 0;
  fld_converged = false;
  fld_outer_residual = 0.0;
  cg_cap_exit_unconverged = 0;
  newton_cap_exit_unconverged = 0;
  fld_escaped_step = 0.0;
  fld_marshak_in_step = 0.0;
  fld_state_supply_in_cumulative = 0.0;
  fld_state_supply_out_cumulative = 0.0;
  fld_state_supply_net_cumulative = 0.0;
  fld_state_supply_in_step = 0.0;
  fld_state_supply_out_step = 0.0;
  fld_state_supply_net_step = 0.0;
  sn_outer_iterations = 0;
  sn_inner_iterations = 0;
  sn_converged = false;
  sn_outer_stagnated = false;
  sn_outer_residual = 0.0;
  sn_inner_residual = 0.0;
  sn_escaped_step = 0.0;
  sn_marshak_in_step = 0.0;
  sn_void_anchor_dE_step = 0.0;
  sn_void_anchor_dE_abs_step = 0.0;
  sn_ap_alpha_max = 0.0;
  sn_ap_alpha_active_faces = 0.0;
  laser_dep.fill(0.0);
  ray_density.fill(0.0);
  laser_ray_R0.clear();
  laser_ray_Z0.clear();
  laser_ray_vR0.clear();
  laser_ray_vZ0.clear();
  laser_ray_x0.clear();
  laser_ray_y0.clear();
  laser_ray_z0.clear();
  laser_ray_vx0.clear();
  laser_ray_vy0.clear();
  laser_ray_vz0.clear();
  laser_ray_power0.clear();
  laser_ray_beam_id.clear();
  laser_ray_is_3d = false;
  std::fill(hydro_active.begin(), hydro_active.end(), static_cast<std::int8_t>(0));
  note_hydro_active_host_write();
  std::fill(cell_is_void.begin(), cell_is_void.end(), static_cast<std::uint8_t>(0));
  apply_button_dormant_storage_mask(*this, mesh);

  t = 0.0;
  step = 0;
  dt = 0.0;
  dt_prev_hydro = -1.0;
  ale_rezoned = false;
  ale_rezone_invocations = 0;
  ale_remaps_applied = 0;
  ale_last_applied_step = -1;
  ale1d_floor_cooldown_remaining = 0;
  diff_ref_diag_baseline_initialized = false;
  diff_ref_diag_gas_mesh_volume_initial = 0.0;
  diff_ref_diag_gas_rho_p50_initial = 0.0;
  multiblock_path_dt_rejection_count = 0;
  multiblock_path_dt_rejection_exhausted_count = 0;
  multiblock_path_dt_min_accepted = std::numeric_limits<double>::infinity();
  axis_margin_initial = -1.0;
  plic_remap_sticky_fallback = false;
  plic_consecutive_drift_triggers = 0;
  plic_last_reconstruction_step = -1;
  plic_interface_mask.reset(0);
  plic_active_mask.reset(0);
  plic_reconstruction_valid.reset(0);
  plic_normal_r.reset(0);
  plic_normal_z.reset(0);
  plic_alpha.reset(0);
  plic_centroid_r.reset(0);
  plic_centroid_z.reset(0);
  plic_last_centroid_r.reset(0);
  plic_last_centroid_z.reset(0);
  plic_face_flux_r.reset(0);
  plic_face_flux_z.reset(0);
  plic_cell_residual.reset(0);
  axis_mass_initial.clear();
  axis_inflow_budget.clear();
  ale_lambda_sweep_target_cell_c = -1;
  ale_lambda_sweep_target_cell_i = -1;
  ale_lambda_sweep_target_cell_j = -1;
  ale_lambda_sweep_classification.clear();
  ale_lambda_sweep_lambda.clear();
  ale_lambda_sweep_min_gauss_j.clear();
  ale_lambda_sweep_min_corner_j.clear();
  ale_lambda_sweep_min_v_rz.clear();
  ale_lambda_sweep_admissible.clear();
  E_safety = 0.0;
  E_numerical_loss = 0.0;
  E_laser_deposited = 0.0;
  E_laser_escaped = 0.0;
  E_laser_incident = 0.0;
  E_ra_deposited = 0.0;
  E_cbet_iaw_step = 0.0;
  E_cbet_iaw = 0.0;
  E_rad_escaped = 0.0;
  E_floor_injected = 0.0;
  E_pdV_bdry = 0.0;
  E_Marshak_in = 0.0;
  E_solver = 0.0;
  radiation_device_flags = DeviceErrorFlags{};
  dispatch_counters.reset();
  t_next_plot = -1.0;
  t_next_history = -1.0;
  t_next_checkpoint = -1.0;
  hydro_t_start_eV = 0.0;
  pressure_drive_1d.reset();
  marshak_Tr_1d.reset();
  hot_e_eta_1d.reset();
  hot_e_eta_ch_1d.clear();
  hot_e_eta_state_eta.clear();
  hot_e_eta_state_kappa_bar.clear();
  hot_e_eta_prev_Pcross.clear();
  hot_e_eta_prev_rbar.clear();
  hot_e_eta_prev_valid.clear();
  hot_e_eta_diag_g.clear();
  hot_e_eta_diag_eta_eq.clear();
  hot_e_eta_diag_tau_s.clear();
  hot_e_eta_diag_I14.clear();
  hot_e_eta_diag_I14_lower.clear();
  hot_e_eta_diag_I14_upper.clear();
  hot_e_eta_diag_n_sigma.clear();
  hot_e_eta_diag_Te_keV.clear();
  hot_e_eta_diag_Ln_um.clear();
  hot_e_eta_diag_clamped.clear();
  cbet_gross_exchange.clear();
  cbet_net_to_inbound.clear();
  ps_sky_mu.clear();
  ps_sky_phi.clear();
  ps_sky_I_tot.clear();
  ps_sky_I_cw.clear();
  ps_sky_n_sigma.clear();
  ps_ray_map.clear();
  ps_ray_map_shell_r.clear();
  ps_port_outgoing_power.clear();
  ps_port_capture_pcross.clear();
  ps_f_illum2 = 0.0;
  ps_f_union = 0.0;
  hot_e_enabled_any = false;
  hot_e_in_step = 0.0;
  hot_e_deposited_step = 0.0;
  hot_e_residual_step = 0.0;
  hot_e_escaped_step = 0.0;
  hot_e_source_r = 0.0;
  hot_e_conservation_resid = 0.0;
  hot_e_dt_limit_s = std::numeric_limits<double>::infinity();
  hot_e_Q_host.clear();
  hot_e_eps_cum_host.clear();
  hot_e_ch_in_step.clear();
  hot_e_ch_deposited_step.clear();
  hot_e_ch_escaped_step.clear();
  E_hot_e_deposited = 0.0;
  E_hot_e_escaped = 0.0;
  E_burn_inflight = 0.0;
  burn_diffusion_any = false;
  burn_mc_any = false;
  burn_mc_live = 0;
  burn_Ng.reset(0);
  burn_Ng_work.reset(0);
  burn_dep_e_dev.reset(0);
  burn_dep_i_dev.reset(0);
  burn_corman_scratch.reset(0);
  burn_mc_r.reset(0);
  burn_mc_mu.reset(0);
  burn_mc_E.reset(0);
  burn_mc_w.reset(0);
  burn_mc_slot.reset(0);
  burn_mc_alive.reset(0);
  marshak_Tr_face_tables.clear();
  laser_waveforms.clear();
}

}  // namespace tenryu::core
