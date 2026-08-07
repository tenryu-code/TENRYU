#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/namelist/errors.hpp"

namespace tenryu::coupling {
struct ProfileObservability;
}

namespace tenryu::core {

inline bool effective_mesh_geometry_soft_fail(const tenryu::core::Config& cfg) {
  return cfg.numerics.hydro.mesh_geometry_soft_fail_enabled ||
         cfg.numerics.profile.icf_standard_ale.enabled;
}

inline bool effective_diagnostics_icf_enabled(const Config& cfg) {
  return cfg.numerics.diagnostics.icf.enabled ||
         cfg.numerics.profile.icf_standard_ale.enabled;
}

inline bool effective_diagnostics_conservation_enabled(const Config& cfg) {
  return cfg.numerics.diagnostics.conservation.enabled ||
         cfg.numerics.profile.icf_standard_ale.enabled;
}

inline bool effective_diagnostics_hotspot_gas_enabled(const Config& cfg) {
  return cfg.numerics.diagnostics.hotspot_gas.enabled;
}

inline bool effective_diagnostics_ale_provenance_emission_enabled(
    const Config& cfg) {
  return cfg.numerics.diagnostics.ale_provenance_emission.enabled ||
         cfg.numerics.profile.icf_standard_ale.enabled;
}

inline bool is_hydro_av_type_config(const std::string& value) {
  return value == "vnr" || value == "riemann" ||
         value == "riemann_compatible" || value == "csw";
}

inline bool is_csw_limiter_config(const std::string& value) {
  return value == "van_leer" || value == "bj";
}

inline bool is_subzonal_merit_mode_config(const std::string& value) {
  return value == "caramana_auto" || value == "constant" || value == "off";
}

inline bool is_icf_standard_ale_claim_level(const std::string& value) {
  return value == "characterization" || value == "pre_plic_smoke" ||
         value == "production_comparable";
}

inline bool is_polar_family(const std::string& logical_mesh_2d) {
  return logical_mesh_2d == "spherical_polar_halfplane" ||
         logical_mesh_2d == "polar_in_box";
}

inline void validate_multigroup_diffusion_config(const Config& config) {
  if (config.radiation.multigroup_diffusion.cg_max_iter < 1) {
    throw namelist::ValueError(
        "Radiation.multigroup_diffusion.cg_max_iter must be >= 1");
  }
  if (config.radiation.multigroup_diffusion.cap_exit_policy != "warn" &&
      config.radiation.multigroup_diffusion.cap_exit_policy != "fail") {
    throw namelist::ConfigError(
        "Radiation.multigroup_diffusion.cap_exit_policy must be one of "
        "{\"warn\", \"fail\"}");
  }
}

inline void validate_hydro_av_config(const Config& config) {
  const auto& main = config.main;
  const auto& numerics = config.numerics;

  if (numerics.hydro.av_heat_to != "ion" &&
      numerics.hydro.av_heat_to != "electron") {
    throw namelist::ConfigError(
        "Numerics.hydro.av_heat_to must be \"ion\" or \"electron\"");
  }
  if (!is_hydro_av_type_config(numerics.hydro.av_type)) {
    throw namelist::ConfigError(
        "Numerics.hydro.av_type must be \"vnr\", \"riemann\", "
        "\"riemann_compatible\", or \"csw\"");
  }
  if (numerics.hydro.av_type == "riemann" ||
      numerics.hydro.av_type == "riemann_compatible") {
    if (main.dimension != "1D_SPH") {
      throw namelist::ConfigError(
          "Numerics.hydro.av_type=\"riemann\" or \"riemann_compatible\" is "
          "supported only in 1D_SPH");
    }
    if (numerics.hydro.av_heat_C > 0.0) {
      throw namelist::ConfigError(
          "Numerics.hydro.av_heat_C > 0 requires Numerics.hydro.av_type=\"vnr\"");
    }
  }
  if (numerics.hydro.av_type == "csw" && main.dimension != "1D_SPH") {
    throw namelist::ConfigError(
        "Numerics.hydro.av_type=\"csw\" is supported only in 1D_SPH");
  }
  if (!is_csw_limiter_config(numerics.hydro.csw_limiter)) {
    throw namelist::ConfigError(
        "Numerics.hydro.csw_limiter must be one of {\"van_leer\", \"bj\"}");
  }
  if (!(numerics.hydro.csw_C1 >= 0.0)) {
    throw namelist::ValueError("Numerics.hydro.csw_C1 must be >= 0");
  }
  if (!(numerics.hydro.csw_C2 >= 0.0)) {
    throw namelist::ValueError("Numerics.hydro.csw_C2 must be >= 0");
  }
  if (!(numerics.hydro.av_cfl_coefficient > 0.0)) {
    throw namelist::ValueError(
        "Numerics.hydro.av_cfl_coefficient must be > 0");
  }
  if (!(std::isfinite(numerics.hydro.qei_multiplier) &&
        numerics.hydro.qei_multiplier > 0.0)) {
    throw namelist::ValueError("Numerics.hydro.qei_multiplier must be > 0");
  }
  if (!(numerics.hydro.csw_shock_limiter_floor >= 0.0 &&
        numerics.hydro.csw_shock_limiter_floor <= 1.0)) {
    throw namelist::ValueError(
        "Numerics.hydro.csw_shock_limiter_floor must be in [0, 1]");
  }
  if (!(numerics.hydro.anti_hourglass_kappa > 0.0)) {
    throw namelist::ValueError(
        "Numerics.hydro.anti_hourglass_kappa must be > 0");
  }
  if (numerics.hydro.av_model == AvModel::CswEdgePlusTensorLimited) {
    throw namelist::ConfigError(
        "Stage G tensor AV is not yet implemented; use csw_edge");
  }
  if (numerics.hydro.av_model == AvModel::CswEdge &&
      !numerics.hydro.subzonal_pressure_enabled) {
    throw namelist::ConfigError(
        "Phase 4 csw_edge AV requires subzonal_pressure_enabled=true "
        "(DRACO/HYDRA pair)");
  }
  if (numerics.hydro.av_model == AvModel::ScalarVnrLegacy &&
      numerics.hydro.subzonal_pressure_enabled) {
    throw namelist::ConfigError(
        "Subzonal pressure requires av_model=csw_edge (DRACO/HYDRA pair)");
  }
  if (numerics.hydro.rz_momentum_scheme != "volume_weighted" &&
      numerics.hydro.rz_momentum_scheme != "area_weighted_symmetric") {
    throw namelist::ConfigError(
        "Numerics.hydro.rz_momentum_scheme must be one of "
        "{\"volume_weighted\", \"area_weighted_symmetric\"}");
  }
  if (numerics.hydro.rz_momentum_scheme == "area_weighted_symmetric") {
    if (main.dim != 2) {
      throw namelist::ConfigError(
          "Numerics.hydro.rz_momentum_scheme=\"area_weighted_symmetric\" "
          "is supported only in 2D_RZ");
    }
    if (numerics.hydro.av_model != AvModel::ScalarVnrLegacy) {
      throw namelist::ConfigError(
          "Numerics.hydro.rz_momentum_scheme=\"area_weighted_symmetric\" "
          "requires av_model=\"scalar_vnr_legacy\"");
    }
    if (numerics.hydro.subzonal_pressure_enabled) {
      throw namelist::ConfigError(
          "Numerics.hydro.rz_momentum_scheme=\"area_weighted_symmetric\" "
          "requires subzonal_pressure_enabled=false");
    }
    if (config.mesh.topology_scheme ==
        TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK) {
      throw namelist::ConfigError(
          "Numerics.hydro.rz_momentum_scheme=\"area_weighted_symmetric\" "
          "rejects multiblock tri-fan-cap meshes in v1; all CSR cells must "
          "be quads");
    }
  }
  if (config.mesh.topology_scheme ==
          TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL &&
      numerics.hydro.av_model == AvModel::ScalarVnrLegacy &&
      !numerics.hydro.subzonal_pressure_enabled) {
    log_warning(
        "multiblock topology with legacy scalar VNR and no subzonal pressure "
        "is not DRACO/HYDRA-class; mesh may tangle under strong drive.");
  }
  if (numerics.hydro.subzonal_pressure_mode != "uniform_cell") {
    throw namelist::ConfigError(
        "Numerics.hydro.subzonal_pressure_mode must be \"uniform_cell\" "
        "in the current implementation");
  }
  if (!is_subzonal_merit_mode_config(numerics.hydro.subzonal_merit_mode)) {
    throw namelist::ConfigError(
        "Numerics.hydro.subzonal_merit_mode must be one of "
        "{\"caramana_auto\", \"constant\", \"off\"}");
  }
  if (!(numerics.hydro.subzonal_alpha1 > 0.0)) {
    throw namelist::ValueError("Numerics.hydro.subzonal_alpha1 must be > 0");
  }
  if (!(numerics.hydro.subzonal_alpha2 >= 0.0)) {
    throw namelist::ValueError("Numerics.hydro.subzonal_alpha2 must be >= 0");
  }
  if (numerics.hydro.subzonal_merit_power < 0) {
    throw namelist::ValueError(
        "Numerics.hydro.subzonal_merit_power must be >= 0");
  }
  if (!(numerics.hydro.subzonal_merit_constant >= 0.0)) {
    throw namelist::ValueError(
        "Numerics.hydro.subzonal_merit_constant must be >= 0");
  }
  if (numerics.hydro.subzonal_mass_enabled && main.dimension != "2D_RZ") {
    throw namelist::ConfigError(
        "Numerics.hydro.subzonal_mass_enabled=true is supported only in 2D_RZ");
  }
  if (numerics.hydro.bbs_axis_policy_enabled && main.dimension != "2D_RZ") {
    throw namelist::ConfigError(
        "Numerics.hydro.bbs_axis_policy_enabled=true is supported only in 2D_RZ");
  }
  if (numerics.hydro.subzonal_mass_lagrangian_invariant_enabled &&
      main.dimension != "2D_RZ") {
    throw namelist::ConfigError(
        "Numerics.hydro.subzonal_mass_lagrangian_invariant_enabled=true is supported only in 2D_RZ");
  }
  if (!(numerics.hydro.hourglass.scale > 0.0)) {
    throw namelist::ValueError("Numerics.hydro.hourglass.scale must be > 0");
  }
  if (!(numerics.hydro.hourglass.activation_corner_j_ratio_threshold > 0.0 &&
        numerics.hydro.hourglass.activation_corner_j_ratio_threshold <= 1.0)) {
    throw namelist::ValueError(
        "Numerics.hydro.hourglass.activation_corner_j_ratio_threshold must be in (0, 1]");
  }
  if (!(numerics.hydro.hourglass.activation_hourglass_amplitude_threshold > 0.0 &&
        numerics.hydro.hourglass.activation_hourglass_amplitude_threshold <= 1.0)) {
    throw namelist::ValueError(
        "Numerics.hydro.hourglass.activation_hourglass_amplitude_threshold must be in (0, 1]");
  }
  if (numerics.hydro.hourglass.subzonal_pressure_model != "linearized") {
    throw namelist::ConfigError(
        "Numerics.hydro.hourglass.subzonal_pressure_model must be \"linearized\" "
        "in the current implementation");
  }
  if (!(numerics.hydro.hourglass.max_force_per_node_fraction > 0.0)) {
    throw namelist::ValueError(
        "Numerics.hydro.hourglass.max_force_per_node_fraction must be > 0");
  }
  if (numerics.hydro.hourglass.enabled && main.dimension != "2D_RZ") {
    throw namelist::ConfigError(
        "Numerics.hydro.hourglass.enabled=true is supported only in 2D_RZ");
  }
  if (numerics.hydro.adaptive_av.enabled) {
    if (main.dimension != "1D_SPH") {
      throw namelist::ConfigError(
          "Numerics.hydro.adaptive_av.enabled=True is supported only in 1D_SPH");
    }
    if (numerics.hydro.av_type != "vnr") {
      throw namelist::ConfigError(
          "Numerics.hydro.adaptive_av.enabled=True requires Numerics.hydro.av_type=\"vnr\"");
    }
  }
}

inline bool is_tri_fan_stage2_mesh(const Config& config) {
  return config.main.dimension == "2D_RZ" &&
         config.mesh.polar_center_treatment == "tri_fan";
}

inline void validate_tri_fan_stage2_config(const Config& config) {
  if (!is_tri_fan_stage2_mesh(config)) {
    return;
  }
  if (config.numerics.hydro.hourglass.enabled) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='tri_fan' does not support "
        "Numerics.hydro.hourglass.enabled=true in Stage 2");
  }
  if (config.numerics.hydro.subzonal_mass_enabled) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='tri_fan' does not support "
        "Numerics.hydro.subzonal_mass_enabled=true in Stage 2");
  }
  if (config.numerics.hydro.hllc_z_flux_2d_rz) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='tri_fan' does not support "
        "Numerics.hydro.hllc_z_flux_2d_rz=true in Stage 2");
  }
  if (config.numerics.hydro.rz_geometric_cfl_precise_u_half_enabled) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='tri_fan' does not support "
        "Numerics.hydro.rz_geometric_cfl_precise_u_half_enabled=true in Stage 2");
  }
  if (config.numerics.hydro.total_energy_remap_2d_rz &&
      config.mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='tri_fan' does not support "
        "Numerics.hydro.total_energy_remap_2d_rz=true in Stage 3");
  }
}

inline bool is_button_stage1_mesh(const Config& config) {
  return config.mesh.polar_center_treatment == "button";
}

inline void validate_button_stage1_config(const Config& config) {
  if (!is_button_stage1_mesh(config)) {
    return;
  }
  if (config.main.dimension != "2D_RZ") {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='button' requires Main.dimension='2D_RZ' "
        "in Stage 1");
  }
  if (config.mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='button' requires "
        "Mesh.topology_scheme='single_block' in Stage 1");
  }
  if (!is_polar_family(config.mesh.logical_mesh_2d)) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='button' requires "
        "Mesh.logical_mesh_2d in {'spherical_polar_halfplane', "
        "'polar_in_box'} in Stage 1");
  }
  if (!(config.mesh.center_button_outer_node_ring >= 1 &&
        config.mesh.center_button_outer_node_ring < config.mesh.nr)) {
    throw namelist::ConfigError(
        "Mesh.center_button_outer_node_ring must satisfy 1 <= I_btn < Mesh.nr");
  }
  if (config.numerics.hydro.hourglass.enabled) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='button' does not support "
        "Numerics.hydro.hourglass.enabled=true in Stage 1");
  }
  if (config.numerics.hydro.subzonal_mass_enabled) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='button' does not support "
        "Numerics.hydro.subzonal_mass_enabled=true in Stage 1");
  }
  if (config.numerics.hydro.hllc_z_flux_2d_rz) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='button' does not support "
        "Numerics.hydro.hllc_z_flux_2d_rz=true in Stage 1");
  }
  if (config.numerics.hydro.rz_geometric_cfl_precise_u_half_enabled) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='button' does not support "
        "Numerics.hydro.rz_geometric_cfl_precise_u_half_enabled=true in Stage 1");
  }
  if (config.numerics.hydro.total_energy_remap_2d_rz) {
    throw namelist::ConfigError(
        "Mesh.polar_center_treatment='button' does not support "
        "Numerics.hydro.total_energy_remap_2d_rz=true in Stage 1");
  }
}

inline void validate_central_pseudo_core_config(const Config& config) {
  if (!config.numerics.ale.central_pseudo_core_enabled) {
    return;
  }
  if (config.main.dimension != "2D_RZ") {
    throw namelist::ConfigError(
        "Numerics.ale.central_pseudo_core_enabled=true is supported only in 2D_RZ");
  }
  if (config.mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK &&
      config.mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK) {
    throw namelist::ConfigError(
        "Numerics.ale.central_pseudo_core_enabled=true requires a 5-block "
        "multiblock center topology");
  }
  if (config.numerics.materials.per_material_conservation_enabled) {
    throw namelist::ConfigError(
        "Numerics.ale.central_pseudo_core_enabled=true does not support "
        "per-material conservation in Exp1");
  }
  if ((config.numerics.hydro.av_model != AvModel::CswEdge &&
       config.numerics.hydro.av_model != AvModel::CswEdgeCsw98) ||
      !config.numerics.hydro.subzonal_pressure_enabled) {
    throw namelist::ConfigError(
        "Numerics.ale.central_pseudo_core_enabled=true requires the 2D "
        "compatible pressure-force path: Numerics.hydro.av_model='csw_edge' "
        "or 'csw_edge_csw98', and subzonal_pressure_enabled=true");
  }
  if (!(config.numerics.ale.central_pseudo_core_s_c > 0.0)) {
    throw namelist::ValueError(
        "Numerics.ale.central_pseudo_core_s_c must be > 0 when "
        "central_pseudo_core_enabled=true");
  }
}

inline void validate_multiblock_s2_runtime_features(const Config& config);

inline bool is_cart_core_parameterized_topology(const TopologyScheme scheme) {
  return scheme == TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL ||
         scheme == TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK ||
         scheme ==
             TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK;
}

inline void validate_multiblock_topology_config(const Config& config) {
  const auto& mesh = config.mesh;
  if (!(std::isfinite(mesh.multiblock_cap_p) &&
        mesh.multiblock_cap_p > 2.0)) {
    throw namelist::ConfigError(
        "Mesh.multiblock_cap_p must be finite and > 2");
  }
  if (mesh.multiblock_bridge_elliptic_sweeps < 0) {
    throw namelist::ConfigError(
        "Mesh.multiblock_bridge_elliptic_sweeps must be >= 0");
  }
  if (!(std::isfinite(mesh.multiblock_bridge_elliptic_omega) &&
        mesh.multiblock_bridge_elliptic_omega > 0.0 &&
        mesh.multiblock_bridge_elliptic_omega < 2.0)) {
    throw namelist::ConfigError(
        "Mesh.multiblock_bridge_elliptic_omega must be finite and in (0, 2)");
  }
  if ((mesh.multiblock_transition_scheme ==
           MultiblockTransitionScheme::ROUNDED_HALF_BUTTERFLY ||
       mesh.multiblock_transition_scheme ==
           MultiblockTransitionScheme::ROUNDED_CORE_SEAM) &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL) {
    throw namelist::ConfigError(
        "Mesh.multiblock_transition_scheme=\"rounded_half_butterfly\" or "
        "\"rounded_core_seam\" "
        "requires Mesh.topology_scheme=\"multiblock_cart_core_polar_shell\"");
  }
  if (mesh.topology_scheme != TopologyScheme::PENTAGON_BELT_SHELL &&
      !mesh.pentagon_belt_layers.empty()) {
    throw namelist::ConfigError(
        "Mesh.pentagon_belt_layers requires "
        "Mesh.topology_scheme=\"pentagon_belt_shell\"");
  }
  if (mesh.topology_scheme == TopologyScheme::PENTAGON_BELT_SHELL &&
      (config.main.dimension != "2D_RZ" || config.main.dim != 2)) {
    throw namelist::ConfigError(
        "Mesh.topology_scheme multiblock modes require Main.dimension=\"2D_RZ\" "
        "and Main.dim=2");
  }
  if (mesh.topology_scheme == TopologyScheme::PENTAGON_BELT_SHELL) {
    if (config.numerics.hydro.av_model == AvModel::CswEdge) {
      throw namelist::ConfigError(
          "pentagon-belt: legacy CSW AV has no polygon face pairing; use csw98");
    }
    if (config.numerics.hydro.subzonal_mass_enabled) {
      throw namelist::ConfigError(
          "pentagon-belt: subzonal corner masses are staged to ALE P2-1c");
    }
    if (config.numerics.hydro.hllc_z_flux_2d_rz) {
      throw namelist::ConfigError("pentagon-belt: HLLC z-flux staged");
    }
    if (mesh.motion == "ale" && config.numerics.ale.enabled &&
        !(config.numerics.ale.rezone_solver == "m1_tmop" &&
          config.numerics.ale.conservative_remap_enabled)) {
      throw namelist::ConfigError(
          "pentagon-belt: runtime ALE requires rezone_solver=m1_tmop with "
          "conservative remap (ALE P4); other modes remain staged");
    }
    if (mesh.pentagon_belt_layers.empty()) {
      throw namelist::ConfigError(
          "Mesh.pentagon_belt_layers must be non-empty for pentagon-belt");
    }
    if (mesh.pentagon_belt_layers.size() > 4U) {
      throw namelist::ConfigError(
          "Mesh.pentagon_belt_layers may contain at most 4 layers");
    }
    for (std::size_t j = 0; j < mesh.pentagon_belt_layers.size(); ++j) {
      const int belt_layer = mesh.pentagon_belt_layers[j];
      if (belt_layer < 1 || belt_layer > mesh.nr - 2) {
        throw namelist::ConfigError(
            "Mesh.pentagon_belt_layers entries must be in [1, Mesh.nr-2]");
      }
      if (j > 0U && belt_layer <= mesh.pentagon_belt_layers[j - 1U]) {
        throw namelist::ConfigError(
            "Mesh.pentagon_belt_layers must be strictly increasing");
      }
      if (j > 0U &&
          belt_layer < mesh.pentagon_belt_layers[j - 1U] + 2) {
        throw namelist::ConfigError(
            "pentagon-belt: belt layers must be separated by at least one "
            "quad layer");
      }
    }
    const int belt_count =
        static_cast<int>(mesh.pentagon_belt_layers.size());
    const int refinement = 1 << belt_count;
    if (mesh.nz % refinement != 0) {
      throw namelist::ConfigError(
          "Mesh.nz must be divisible by 2^K for K pentagon-belt layers");
    }
    if ((mesh.nz >> belt_count) < 4) {
      throw namelist::ConfigError(
          "Mesh.nz >> K must be >= 4 for K pentagon-belt layers");
    }
    if (mesh.logical_mesh_2d != "spherical_polar_halfplane") {
      throw namelist::ConfigError(
          "pentagon-belt requires "
          "Mesh.logical_mesh_2d=\"spherical_polar_halfplane\"");
    }
    if (mesh.polar_center_treatment != "annular") {
      throw namelist::ConfigError(
          "pentagon-belt: center treatments are staged; use annular");
    }
    if (config.numerics.hydro.boundary_2d.r_inner != "axis" &&
        config.numerics.hydro.boundary_2d.r_inner != "pinned") {
      throw namelist::ConfigError(
          "pentagon-belt: annular inner ring requires r_inner=\"axis\" "
          "(spherical-normal projection) or \"pinned\" (rigid no-slip wall)");
    }
    if (!mesh.explicit_nodes_theta.empty() ||
        !mesh.grid_segments_theta.empty()) {
      throw namelist::ConfigError(
          "pentagon-belt: only uniform or equal-mu theta ladders nest 2:1 "
          "across belts");
    }
    if (config.radiation.mode == RadiationMode::SnTransport) {
      throw namelist::ConfigError(
          "pentagon-belt: S_N face-flux arrays are structured-index only; "
          "staged");
    }
  }
  if (!is_cart_core_parameterized_topology(mesh.topology_scheme)) {
    return;
  }
  if (config.numerics.plic.enabled) {
    throw namelist::ConfigError(
        "Numerics.plic.enabled is not supported on multiblock topologies: "
        "PLIC interface reconstruction is structured-index only "
        "(AI-review k02 F-13 fail-loud guard)");
  }
  if (config.main.dimension != "2D_RZ" || config.main.dim != 2) {
    throw namelist::ConfigError(
        "Mesh.topology_scheme multiblock modes require Main.dimension=\"2D_RZ\" "
        "and Main.dim=2");
  }
  if (config.numerics.hydro.total_energy_remap_2d_rz &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK) {
    throw namelist::ConfigError(
        "Numerics.hydro.total_energy_remap_2d_rz=true is supported for CSR "
        "multiblock only with "
        "Mesh.topology_scheme=\"multiblock_half_butterfly_trifan_cap_5block\"");
  }
  if (mesh.multiblock_cart_core_n_c < 4) {
    throw namelist::ConfigError(
        "Mesh.multiblock_cart_core_n_c must be >= 4");
  }
  if (static_cast<long long>(mesh.nz) !=
      4LL * static_cast<long long>(mesh.multiblock_cart_core_n_c)) {
    throw namelist::ConfigError(
        "Mesh.nz must equal 4*Mesh.multiblock_cart_core_n_c for "
        "Mesh.topology_scheme=\"multiblock_cart_core_polar_shell\" or "
        "\"multiblock_half_butterfly_5block\"");
  }
  if (!(std::isfinite(mesh.multiblock_cart_core_r_c) &&
        mesh.multiblock_cart_core_r_c > 0.0)) {
    throw namelist::ConfigError(
        "Mesh.multiblock_cart_core_r_c must be finite and > 0");
  }
  if (!(std::isfinite(mesh.multiblock_cart_core_r_match) &&
        std::sqrt(2.0) * mesh.multiblock_cart_core_r_c <
            mesh.multiblock_cart_core_r_match)) {
    throw namelist::ConfigError(
        "Mesh.multiblock_cart_core_r_match must satisfy "
        "sqrt(2)*Mesh.multiblock_cart_core_r_c < r_match");
  }
  if (!(std::isfinite(mesh.spherical_polar_s_max) &&
        mesh.spherical_polar_s_max > 0.0)) {
    throw namelist::ConfigError(
        "Mesh.spherical_polar_s_max must be finite and > 0");
  }
  if (!(mesh.multiblock_cart_core_r_match < mesh.spherical_polar_s_max)) {
    throw namelist::ConfigError(
        "Mesh.multiblock_cart_core_r_match must be < "
        "Mesh.spherical_polar_s_max");
  }
  if (mesh.multiblock_cart_core_bridge_layers < 1) {
    throw namelist::ConfigError(
        "Mesh.multiblock_cart_core_bridge_layers must be >= 1");
  }
  validate_multiblock_s2_runtime_features(config);
}

inline void validate_multiblock_s2_runtime_features(const Config& config) {
  const auto& hydro = config.numerics.hydro;
  if (hydro.center_perturbation_diag_radial_bins < 1) {
    throw namelist::ConfigError(
        "center_perturbation_diag_radial_bins must be >= 1");
  }
}

inline void validate_ale1d_config(const Config& config) {
  const auto& main = config.main;
  const auto& radiation = config.radiation;
  const auto& ale = config.numerics.ale1d;
  const auto& rezone = ale.rezone;
  const auto& remap = ale.remap;
  const auto validate_tol = [](const auto& tol, const char* path) {
    if (tol.soft > tol.hard) {
      throw namelist::ConfigError(std::string(path) + ".soft must be <= hard");
    }
  };
  const auto validate_range = [](const double lo,
                                 const double hi,
                                 const char* lo_path,
                                 const char* hi_path) {
    if (!(lo > 0.0) || !(hi > 0.0) || lo > hi) {
      throw namelist::ConfigError(std::string(lo_path) + "/" + hi_path +
                                  " must be positive with min <= max");
    }
  };
  validate_tol(ale.total_mass_tol, "Numerics.ale1d.total_mass_tol");
  validate_tol(ale.material_mass_tol, "Numerics.ale1d.material_mass_tol");
  validate_tol(ale.radiation_group_energy_tol,
               "Numerics.ale1d.radiation_group_energy_tol");
  validate_tol(ale.material_internal_energy_tol,
               "Numerics.ale1d.material_internal_energy_tol");
  validate_tol(ale.total_material_energy_tol,
               "Numerics.ale1d.total_material_energy_tol");
  validate_tol(ale.global_total_energy_tol,
               "Numerics.ale1d.global_total_energy_tol");
  validate_tol(ale.kinetic_energy_drift_tol,
               "Numerics.ale1d.kinetic_energy_drift_tol");

  if (!(ale.max_node_displacement_fraction_mu > 0.0 &&
        ale.max_node_displacement_fraction_mu < 0.5)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.max_node_displacement_fraction_mu must be in (0, 0.5)");
  }
  if (!(ale.max_node_displacement_fraction_r > 0.0 &&
        ale.max_node_displacement_fraction_r < 0.5)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.max_node_displacement_fraction_r must be in (0, 0.5)");
  }
  if (!(ale.protected_fraction_max > 0.0 &&
        ale.protected_fraction_max < 0.5)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.protected_fraction_max must be in (0, 0.5)");
  }
  if (ale.min_movable_segment_hard < 4) {
    throw namelist::ConfigError(
        "Numerics.ale1d.min_movable_segment_hard must be >= 4");
  }
  if (ale.min_movable_segment_warn < ale.min_movable_segment_hard) {
    throw namelist::ConfigError(
        "Numerics.ale1d.min_movable_segment_warn must be >= min_movable_segment_hard");
  }
  if (ale.every_n_steps < 1) {
    throw namelist::ConfigError("Numerics.ale1d.every_n_steps must be >= 1");
  }
  if (!(rezone.monitor_floor > 0.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.monitor_floor must be positive");
  }
  if (!(rezone.monitor_wmax_ratio >= 1.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.monitor_wmax_ratio must be >= 1");
  }
  if (rezone.monitor_smoothing_iterations < 0) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.monitor_smoothing_iterations must be >= 0");
  }
  if (!(rezone.min_floor_fraction > 0.0 && rezone.min_floor_fraction < 1.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.min_floor_fraction must be in (0, 1)");
  }
  if (!(rezone.gaussian_truncation_sigma > 0.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.gaussian_truncation_sigma must be positive");
  }
  if (!(rezone.spatial_target_cells_fraction >= 0.0 &&
        rezone.spatial_target_cells_fraction < 1.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.spatial_target_cells_fraction must be in [0, 1)");
  }
  if (!(rezone.spatial_power > 0.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.rezone.spatial_power must be positive");
  }
  validate_range(rezone.laser_spatial_dr_min_cm,
                 rezone.laser_spatial_dr_max_cm,
                 "Numerics.ale1d.rezone.laser_spatial_dr_min_cm",
                 "Numerics.ale1d.rezone.laser_spatial_dr_max_cm");
  validate_range(rezone.ablation_spatial_dr_min_cm,
                 rezone.ablation_spatial_dr_max_cm,
                 "Numerics.ale1d.rezone.ablation_spatial_dr_min_cm",
                 "Numerics.ale1d.rezone.ablation_spatial_dr_max_cm");
  validate_range(rezone.shock_spatial_dr_min_cm,
                 rezone.shock_spatial_dr_max_cm,
                 "Numerics.ale1d.rezone.shock_spatial_dr_min_cm",
                 "Numerics.ale1d.rezone.shock_spatial_dr_max_cm");
  if (!(remap.limiter_theta > 0.0)) {
    throw namelist::ConfigError(
        "Numerics.ale1d.remap.limiter_theta must be positive");
  }
  if (remap.high_order_ramp_cells < 0) {
    throw namelist::ConfigError(
        "Numerics.ale1d.remap.high_order_ramp_cells must be >= 0");
  }
  if (remap.radiation_high_order_ramp_cells < 0) {
    throw namelist::ConfigError(
        "Numerics.ale1d.remap.radiation_high_order_ramp_cells must be >= 0");
  }

  if (!ale.enabled) {
    return;
  }

  if (main.dimension != "1D_SPH") {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True is supported only in 1D_SPH");
  }
  if (main.dim != 1) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True requires Main.dim=1");
  }
  if (radiation.mode == RadiationMode::ImcDdmc ||
      radiation.imc.enabled ||
      radiation.ddmc.enabled ||
      radiation.imc.difference.enabled ||
      radiation.holo.enabled) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True requires deterministic radiation "
        "(multigroup_diffusion) or radiation off, not IMC/DDMC/HOLO");
  }
  // k16 C3/C7/C8 (AI review 2026-07-26) fail-closed operating boundary for
  // the experimental V3 ALE prototype: the remap has no companion-field
  // registry (S_N angular state and burn inventories are left on the old mesh),
  // and post-remap EOS closure remains single-material. Reject configurations
  // outside the validated single-material + (hydro|FLD) envelope instead of
  // silently corrupting them.
  if (radiation.enabled && radiation.mode == RadiationMode::SnTransport) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True does not support mode=\"sn_transport\" "
        "(S_N angular state is not remapped; use multigroup_diffusion or "
        "disable ALE1D)");
  }
  if (config.burn.enabled) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True does not support Burn.enabled=True "
        "(burn inventories are not remapped)");
  }
  if (config.materials.materials.size() != 1) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True supports exactly one material "
        "(multi-material volFrac/companion-field remap is not certified)");
  }
  const auto& ale_material = config.materials.materials.front();
  if (ale_material.eos_model != "ideal_gas" && !ale_material.eos_tables) {
    throw namelist::ConfigError(
        "Numerics.ale1d.enabled=True requires eos_model=\"ideal_gas\" or a "
        "table EOS (table backend rho-e reclosure capability is checked at "
        "the ALE1D driver entry)");
  }
}

inline void validate_ale_identity_diag_config(const Config& config) {
  if ((config.numerics.ale.ale_identity_mode ||
       config.numerics.ale.ale_mover_diag ||
       config.numerics.ale.ale_preserve_lagrangian_velocity_carry) &&
      config.main.dimension != "2D_RZ") {
    throw namelist::ConfigError(
        "Numerics.ale.ale_identity_mode, Numerics.ale.ale_mover_diag, and "
        "Numerics.ale.ale_preserve_lagrangian_velocity_carry are supported "
        "only for Main.dimension=\"2D_RZ\"");
  }
}

inline void validate_transaction_failure_inject_config(const Config& config) {
  const int point = config.numerics.ale.transaction_failure_inject_point;
  if (!(point >= 0 && point <= 10)) {
    throw namelist::ValueError(
        "Numerics.ale.transaction_failure_inject_point must be in [0, 10]");
  }
}

inline bool is_multiblock_differential_reference_topology(const Config& config) {
  return config.main.dimension == "2D_RZ" &&
         config.mesh.topology_scheme != TopologyScheme::SINGLE_BLOCK;
}

inline void validate_m1_tmop_config(const Config& config) {
  const auto& ale = config.numerics.ale;
  if (ale.rezone_solver != "legacy_winslow" &&
      ale.rezone_solver != "rz_full_metric_winslow" &&
      ale.rezone_solver != "m1_tmop") {
    throw namelist::ValueError(
        "Numerics.ale.rezone_solver must be one of "
        "{\"legacy_winslow\", \"rz_full_metric_winslow\", \"m1_tmop\"}");
  }
  if (!(ale.m1_gamma_align >= 0.0)) {
    throw namelist::ValueError(
        "Numerics.ale.m1_gamma_align must be >= 0");
  }
  if (!(ale.m1_lambda_tether >= 0.0)) {
    throw namelist::ValueError(
        "Numerics.ale.m1_lambda_tether must be >= 0");
  }
  if (!(ale.m1_theta_reg >= 0.0)) {
    throw namelist::ValueError(
        "Numerics.ale.m1_theta_reg must be >= 0");
  }
  if (ale.m1_sweeps < 1) {
    throw namelist::ValueError(
        "Numerics.ale.m1_sweeps must be >= 1");
  }
  if (!std::isfinite(ale.m1_min_j_dec_rel) ||
      ale.m1_min_j_dec_rel < 0.0) {
    throw namelist::ValueError(
        "Numerics.ale.m1_min_j_dec_rel must be finite and >= 0");
  }
  if (!(ale.m1_barrier_beta >= 0.0)) {
    throw namelist::ValueError(
        "Numerics.ale.m1_barrier_beta must be >= 0");
  }
  if (ale.rezone_solver == "m1_tmop" &&
      (config.mesh.logical_mesh_2d == "rectangular_rz" ||
       config.mesh.logical_mesh_2d == "cone_shell")) {
    throw namelist::ConfigError(
        "Numerics.ale.rezone_solver=\"m1_tmop\" is staged for polar-family "
        "logical meshes; rectangular/cone logical meshes are not supported "
        "in M1 v1");
  }
}

inline void validate_multiblock_differential_reference_config(
    const Config& config) {
  const auto& ale = config.numerics.ale;
  const int reference_mode_count =
      (ale.multiblock_scaled_reference_enabled ? 1 : 0) +
      (ale.multiblock_differential_reference_enabled ? 1 : 0) +
      (ale.multiblock_lagrangian_bulk_center_patch_reference_enabled ? 1 : 0);
  if (reference_mode_count > 1) {
    throw namelist::ConfigError(
        "At most one ALE multiblock reference mode may be enabled: "
        "Numerics.ale.multiblock_scaled_reference_enabled, "
        "Numerics.ale.multiblock_differential_reference_enabled, "
        "Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled");
  }
  if (ale.multiblock_differential_reference_enabled &&
      !is_multiblock_differential_reference_topology(config)) {
    throw namelist::ConfigError(
        "Numerics.ale.multiblock_differential_reference_enabled=true requires "
        "Main.dimension=\"2D_RZ\" and multiblock Mesh.topology_scheme");
  }
  if (ale.multiblock_lagrangian_bulk_center_patch_reference_enabled) {
    if (!is_multiblock_differential_reference_topology(config)) {
      throw namelist::ConfigError(
          "Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled=true "
          "requires Main.dimension=\"2D_RZ\" and multiblock Mesh.topology_scheme");
    }
    if (!ale.conservative_remap_enabled ||
        ale.conservative_remap_target != "reference") {
      throw namelist::ConfigError(
          "Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled=true "
          "requires Numerics.ale.conservative_remap_enabled=true and "
          "Numerics.ale.conservative_remap_target=\"reference\"");
    }
  }
  if (ale.multiblock_center_patch_ring_max < 0) {
    throw namelist::ValueError(
        "Numerics.ale.multiblock_center_patch_ring_max must be >= 0");
  }
  if (ale.multiblock_center_patch_halo_layers < 0) {
    throw namelist::ValueError(
        "Numerics.ale.multiblock_center_patch_halo_layers must be >= 0");
  }
  if (!(ale.multiblock_center_patch_vol_on > 0.0 &&
        ale.multiblock_center_patch_vol_on < 1.0 &&
        ale.multiblock_center_patch_vol_off > 0.0 &&
        ale.multiblock_center_patch_vol_off < 1.0 &&
        ale.multiblock_center_patch_vol_on <
            ale.multiblock_center_patch_vol_off)) {
    throw namelist::ValueError(
        "Numerics.ale.multiblock_center_patch_vol_on/off must be in (0, 1) "
        "with on < off");
  }
  if (!(ale.multiblock_center_patch_cornerj_on > 0.0 &&
        ale.multiblock_center_patch_cornerj_on < 1.0 &&
        ale.multiblock_center_patch_cornerj_off > 0.0 &&
        ale.multiblock_center_patch_cornerj_off < 1.0 &&
        ale.multiblock_center_patch_cornerj_on <
            ale.multiblock_center_patch_cornerj_off)) {
    throw namelist::ValueError(
        "Numerics.ale.multiblock_center_patch_cornerj_on/off must be in (0, 1) "
        "with on < off");
  }
  if (!(ale.multiblock_center_patch_gaussj_on > 0.0 &&
        ale.multiblock_center_patch_gaussj_on < 1.0 &&
        ale.multiblock_center_patch_gaussj_off > 0.0 &&
        ale.multiblock_center_patch_gaussj_off < 1.0 &&
        ale.multiblock_center_patch_gaussj_on <
            ale.multiblock_center_patch_gaussj_off)) {
    throw namelist::ValueError(
        "Numerics.ale.multiblock_center_patch_gaussj_on/off must be in (0, 1) "
        "with on < off");
  }
}

struct IcfStandardAleProfileValidationCounters {
  int forbidden_config_violations = 0;
  int escape_valve_activations = 0;
};

inline IcfStandardAleProfileValidationCounters
validate_icf_standard_ale_profile_config_counted(
    const Config& config,
    const bool emit_warnings = true) {
  IcfStandardAleProfileValidationCounters counters{};
  const auto& profile = config.numerics.profile.icf_standard_ale;
  if (!profile.enabled) {
    return counters;
  }
  if (!is_icf_standard_ale_claim_level(profile.claim_level)) {
    throw namelist::ConfigError(
        "Numerics.profile.icf_standard_ale.claim_level must be one of "
        "{\"characterization\", \"pre_plic_smoke\", "
        "\"production_comparable\"}");
  }

  const auto bool_to_string = [](const bool value) -> std::string {
    return value ? "true" : "false";
  };
  const auto join_values = [](const std::vector<std::string>& values) {
    std::string joined;
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        joined += ",";
      }
      joined += values[i];
    }
    return joined;
  };

  struct ForbiddenCheck {
    bool forbidden_value;
    bool current_value;
    const char* namelist_path;
  };

  const auto& fwe = profile.forbidden_when_enabled;
  const ForbiddenCheck forbidden[] = {
      {fwe.hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value,
       config.numerics.hydro.dispatcher_state_sensitive_bypass_enabled,
       "Numerics.hydro.dispatcher_state_sensitive_bypass_enabled"},
      {fwe.ale_local_boundary_repair_enabled_forbidden_value,
       config.numerics.ale.local_boundary_repair_enabled,
       "Numerics.ale.local_boundary_repair_enabled"},
      {fwe.ale_multi_node_boundary_repair_enabled_forbidden_value,
       config.numerics.ale.multi_node_boundary_repair_enabled,
       "Numerics.ale.multi_node_boundary_repair_enabled"},
      {fwe.ale_multi_node_interior_repair_enabled_forbidden_value,
       config.numerics.ale.multi_node_interior_repair_enabled,
       "Numerics.ale.multi_node_interior_repair_enabled"},
      {fwe.ale_axis_variational_projection_enabled_forbidden_value,
       config.numerics.ale.axis_variational_projection_enabled,
       "Numerics.ale.axis_variational_projection_enabled"},
      {fwe.ale_emergency_cell_deactivation_enabled_forbidden_value,
       config.numerics.ale.emergency_cell_deactivation_enabled,
       "Numerics.ale.emergency_cell_deactivation_enabled"},
      {fwe.hydro_driver_retry_active_mesh_repair_enabled_forbidden_value,
       config.numerics.hydro.driver_retry_active_mesh_repair_enabled,
       "Numerics.hydro.driver_retry_active_mesh_repair_enabled"},
  };

  for (const auto& check : forbidden) {
    if (!check.forbidden_value || !check.current_value) {
      continue;
    }
    const std::string path = check.namelist_path;
    ++counters.forbidden_config_violations;
    if (profile.escape_valves.allow_nonstandard_mesh_rescue) {
      ++counters.escape_valve_activations;
      if (emit_warnings) {
        log_warning("[icf_standard_ale_profile] escape valve fires: " + path +
                    "=true is forbidden under public-baseline but "
                    "escape_valves.allow_nonstandard_mesh_rescue=true. "
                    "Run will be classified TENRYU_EXTENDED_ALE.");
      }
    } else if (profile.enforce) {
      throw namelist::ConfigError(
          "Numerics.profile.icf_standard_ale.enabled=true forbids " + path +
          "=true (forbidden_when_enabled). Set "
          "escape_valves.allow_nonstandard_mesh_rescue=true to override "
          "(run becomes TENRYU_EXTENDED_ALE), or set " +
          path + "=false, or set enabled=false.");
    } else {
      if (emit_warnings) {
        log_warning("[icf_standard_ale_profile] forbidden flag " + path +
                    "=true with enforce=false; documented violation, run will be "
                    "classified TENRYU_EXTENDED_ALE.");
      }
    }
  }

  // Mesh-rescue escape valves apply only to forbidden rescue/repair mechanisms.
  const auto emit_allowed_violation = [&](const std::string& path,
                                          const std::string& current_value,
                                          const std::string& expected_value) {
    ++counters.forbidden_config_violations;
    if (profile.enforce) {
      throw namelist::ConfigError(
          "Numerics.profile.icf_standard_ale.enabled=true requires " + path +
          "=" + expected_value + " (allowed_when_enabled), got " + current_value +
          ". Set " + path + "=" + expected_value +
          ", set enabled=false, or set enforce=false to document a nonstandard run.");
    }
    if (emit_warnings) {
      log_warning("[icf_standard_ale_profile] allowed_when_enabled violation: " +
                  path + "=" + current_value + " but expected " + expected_value +
                  " with enforce=false; documented violation, run will be classified "
                  "TENRYU_EXTENDED_ALE.");
    }
  };

  const auto& awe = profile.allowed_when_enabled;
  if (config.numerics.ale.enabled != awe.ale_enabled_required_value) {
    emit_allowed_violation(
        "Numerics.ale.enabled",
        bool_to_string(config.numerics.ale.enabled),
        bool_to_string(awe.ale_enabled_required_value));
  }
  if (config.numerics.ale.axis_repair_mode !=
      awe.ale_axis_repair_mode_required_value) {
    emit_allowed_violation("Numerics.ale.axis_repair_mode",
                           config.numerics.ale.axis_repair_mode,
                           awe.ale_axis_repair_mode_required_value);
  }
  const auto& allowed_remap_schemes = awe.ale_remap_scheme_allowed_values;
  if (std::find(allowed_remap_schemes.begin(),
                allowed_remap_schemes.end(),
                config.numerics.ale.remap_scheme) ==
      allowed_remap_schemes.end()) {
    emit_allowed_violation("Numerics.ale.remap_scheme",
                           config.numerics.ale.remap_scheme,
                           "one of {" + join_values(allowed_remap_schemes) + "}");
  }
  const auto& allowed_donor_sign_fixed =
      awe.ale_donor_sign_fixed_allowed_values;
  if (!allowed_donor_sign_fixed.empty() &&
      std::find(allowed_donor_sign_fixed.begin(),
                allowed_donor_sign_fixed.end(),
                config.numerics.ale.swept_volume_sign_fixed) ==
          allowed_donor_sign_fixed.end()) {
    std::vector<std::string> values;
    values.reserve(allowed_donor_sign_fixed.size());
    for (const bool value : allowed_donor_sign_fixed) {
      values.push_back(bool_to_string(value));
    }
    emit_allowed_violation("Numerics.ale.swept_volume_sign_fixed",
                           bool_to_string(config.numerics.ale.swept_volume_sign_fixed),
                           "one of {" + join_values(values) + "}");
  }
  if (config.numerics.hydro.driver_full_step_retry_enabled !=
      awe.hydro_driver_full_step_retry_enabled_required_value) {
    emit_allowed_violation(
        "Numerics.hydro.driver_full_step_retry_enabled",
        bool_to_string(config.numerics.hydro.driver_full_step_retry_enabled),
        bool_to_string(awe.hydro_driver_full_step_retry_enabled_required_value));
  }
  return counters;
}

inline void validate_icf_standard_ale_profile_config(
    const Config& config,
    std::nullptr_t = nullptr) {
  (void)validate_icf_standard_ale_profile_config_counted(config, true);
}

inline void validate_legacy_regression_profile(const Config& config) {
  const auto& profile = config.numerics.profile.legacy_regression;
  if (!profile.enabled) {
    return;
  }
  if (profile.revision != "2026-07-27") {
    throw namelist::ConfigError(
        "Numerics.profile.legacy_regression@" + profile.revision +
        ": unknown legacy_regression revision " + profile.revision +
        "; known: 2026-07-27");
  }
  if (config.numerics.ale.swept_volume_sign_fixed) {
    throw namelist::ConfigError(
        "Numerics.profile.legacy_regression@2026-07-27 requires "
        "Numerics.ale.swept_volume_sign_fixed=false (legacy donor convention); "
        "remove the override or leave the profile.");
  }
  if (config.numerics.profile.icf_standard_ale.enabled) {
    throw namelist::ConfigError(
        "Numerics.profile.legacy_regression@2026-07-27 cannot coexist with "
        "Numerics.profile.icf_standard_ale; a certified-production profile and "
        "the legacy regression profile cannot coexist.");
  }
  log_warning(
      "Numerics.profile.legacy_regression@2026-07-27 active — CI/regression use "
      "only; not a production science profile");
}

void validate_icf_standard_ale_profile_config(
    const Config& config,
    tenryu::coupling::ProfileObservability* observability);

}  // namespace tenryu::core
