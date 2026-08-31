#include "core/namelist/freeze.hpp"
#include "core/error.hpp"
#include "core/namelist/errors.hpp"

#if TENRYU_ENABLE_PYTHON

#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include "core/version.hpp"

namespace tenryu::core::namelist {
namespace {

namespace py = pybind11;
constexpr int kCheckpointJsonSchemaV1 = 1;
constexpr int kCheckpointJsonSchemaV2 = 2;
constexpr int kCheckpointJsonSchemaV3 = 3;
constexpr int kCheckpointJsonSchemaV4 = 4;
constexpr int kCheckpointJsonSchemaV5 = 5;
constexpr int kCheckpointJsonSchemaV6 = 6;
constexpr int kCheckpointJsonSchemaV7 = 7;
constexpr int kCheckpointJsonSchemaV8 = 8;
constexpr int kCheckpointJsonSchemaV9 = 9;
constexpr int kCheckpointJsonSchemaV10 = 10;
constexpr int kCheckpointJsonSchemaV11 = 11;
constexpr int kCheckpointJsonSchemaV12 = 12;
constexpr int kCheckpointJsonSchemaV13 = 13;
constexpr int kCheckpointJsonSchemaV14 = 14;
constexpr int kCheckpointJsonSchemaV15 = 15;
constexpr int kCheckpointJsonSchemaV16 = 16;
constexpr int kCheckpointJsonSchemaV17 = 17;
constexpr int kCheckpointJsonSchemaV18 = 18;
constexpr int kCheckpointJsonSchemaV19 = 19;
constexpr int kCheckpointJsonSchemaV20 = 20;
constexpr int kCheckpointJsonSchemaV21 = 21;
constexpr int kCheckpointJsonSchemaV22 = 22;
constexpr int kCheckpointJsonSchemaV23 = 23;
constexpr int kCheckpointJsonSchemaV24 = 24;
constexpr int kCheckpointJsonSchemaV25 = 25;

std::string now_utc_iso8601() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  std::tm utc_tm{};
#if defined(_WIN32)
  gmtime_s(&utc_tm, &now_time);
#else
  gmtime_r(&now_time, &utc_tm);
#endif
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_tm);
  return std::string(buffer);
}

py::object nan_safe(double value) {
  if (std::isnan(value) || std::isinf(value)) {
    return py::none();
  }
  return py::cast(value);
}

py::list serialize_double_list_17g(const std::vector<double>& values) {
  py::list out;
  py::object format_fn = py::module_::import("builtins").attr("format");
  for (const double value : values) {
    const py::str formatted = py::str(format_fn(value, ".17g"));
    out.append(py::float_(formatted));
  }
  return out;
}

bool auto_config_has_non_default(const Config::MeshConfig::AutoZoneConfig& cfg) {
  const Config::MeshConfig::AutoZoneConfig def;
  return cfg.mass_ratio_max != def.mass_ratio_max ||
         cfg.n_bridge_min != def.n_bridge_min ||
         cfg.n_bridge_max != def.n_bridge_max ||
         cfg.bridge_frac_max != def.bridge_frac_max ||
         cfg.rho_void_cut != def.rho_void_cut ||
         cfg.dr_min != def.dr_min ||
         cfg.mass_ratio_hard_max != def.mass_ratio_hard_max ||
         cfg.max_iter != def.max_iter ||
         cfg.bulk_mass_tol != def.bulk_mass_tol;
}

py::object serialize_callable(const Config::CallableInfo& callable) {
  if (!callable.detected) {
    return py::none();
  }
  py::dict out;
  out["_type"] = "callable";
  out["name"] = callable.name;
  out["repr"] = callable.repr;
  out["source_hash"] = callable.source_hash;
  return std::move(out);
}

const char* topology_scheme_to_string(const TopologyScheme topology_scheme) {
  switch (topology_scheme) {
    case TopologyScheme::SINGLE_BLOCK:
      return "single_block";
    case TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL:
      return "multiblock_cart_core_polar_shell";
    case TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK:
      return "multiblock_half_butterfly_5block";
    case TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK:
      return "multiblock_half_butterfly_trifan_cap_5block";
    case TopologyScheme::CONE_SHELL_SPINE:
      return "cone_shell_spine";
    case TopologyScheme::PENTAGON_BELT_SHELL:
      return "pentagon_belt_shell";
  }
  return "single_block";
}

const char* multiblock_transition_scheme_to_string(
    const MultiblockTransitionScheme transition_scheme) {
  switch (transition_scheme) {
    case MultiblockTransitionScheme::HERMITE_BRIDGE:
      return "hermite_bridge";
    case MultiblockTransitionScheme::ROUNDED_HALF_BUTTERFLY:
      return "rounded_half_butterfly";
    case MultiblockTransitionScheme::ROUNDED_CORE_SEAM:
      return "rounded_core_seam";
  }
  return "hermite_bridge";
}

const char* av_qcap_scope_to_string(const AvQcapScope scope) {
  switch (scope) {
    case AvQcapScope::GLOBAL:
      return "global";
    case AvQcapScope::TRI_FAN_RADIAL_INDEX:
      return "tri_fan_radial_index";
    case AvQcapScope::CENTROID_R_LE_R_MATCH:
      return "centroid_r_le_r_match";
  }
  return "global";
}

const char* center_cfl_scope_to_string(const CenterCflScope scope) {
  switch (scope) {
    case CenterCflScope::DISABLED:
      return "disabled";
    case CenterCflScope::TRI_FAN_RADIAL_INDEX:
      return "tri_fan_radial_index";
    case CenterCflScope::CENTROID_R_LE_R_MATCH:
      return "centroid_r_le_r_match";
  }
  return "disabled";
}

const char* center_perturbation_diag_scope_to_string(
    const CenterPerturbationDiagScope scope) {
  switch (scope) {
    case CenterPerturbationDiagScope::DISABLED:
      return "disabled";
    case CenterPerturbationDiagScope::TRI_FAN_FIRST_RING:
      return "tri_fan_first_ring";
    case CenterPerturbationDiagScope::CENTROID_R_INNERMOST_BINS:
      return "centroid_r_innermost_bins";
  }
  return "disabled";
}

/*
 * Phase-1 limitation:
 * - Freeze serialization emits flattened material keys.
 * - No corresponding thaw/deserialize path exists for this JSON shape.
 * - Restart compatibility compares canonical frozen_config JSON by string equality,
 *   which is sufficient for same-version restarts.
 *
 * Phase-2 should implement version-aware semantic comparison or a proper
 * deserialize path.
 */
py::dict serialize_main(const Config::MainConfig& main) {
  py::dict out;
  out["name"] = main.name;
  out["dimension"] = main.dimension;
  out["temperature_model"] = main.temperature_model;
  out["two_temperature"] = main.two_temperature;
  out["t_end"] = main.t_end;
  out["seed"] = main.seed;
  out["max_steps"] = main.max_steps;
  out["verbosity"] = main.verbosity;
  out["restart_from"] = main.restart_from;
  return out;
}

py::dict serialize_mesh(const Config::MeshConfig& mesh) {
  py::dict floors;
  floors["rho_floor_gcc"] = mesh.floors.rho_floor_gcc;
  floors["Te_floor_eV"] = mesh.floors.Te_floor_eV;
  floors["Ti_floor_eV"] = mesh.floors.Ti_floor_eV;

  py::dict grading;
  grading["edge_ratio"] = mesh.grading.edge_ratio;
  grading["sg_order"] = mesh.grading.sg_order;
  grading["sg_sigma"] = mesh.grading.sg_sigma;
  grading["mapping"] = mesh.grading.mapping;

  py::dict out;
  out["nr"] = mesh.nr;
  out["nz"] = mesh.nz;
  out["r_min"] = mesh.r_min;
  out["r_max"] = mesh.r_max;
  out["z_min"] = nan_safe(mesh.z_min);
  out["z_max"] = nan_safe(mesh.z_max);
  out["grid_type_r"] = mesh.grid_type_r;
  out["geometry_1d"] = mesh.geometry_1d;
  out["grid_type_z"] = mesh.grid_type_z;
  out["logical_mesh_2d"] = mesh.logical_mesh_2d;
  out["polar_center_treatment"] = mesh.polar_center_treatment;
  if (mesh.polar_center_treatment == "button") {
    out["center_button_outer_node_ring"] =
        mesh.center_button_outer_node_ring;
  }
  out["polar_equal_mu_zoning"] = mesh.polar_equal_mu_zoning;
  out["spherical_polar_s_max"] = mesh.spherical_polar_s_max;
  out["polar_theta_min"] = mesh.polar_theta_min;
  if (mesh.logical_mesh_2d == "polar_in_box") {
    out["box_r_max"] = mesh.box_r_max;
    out["box_z_min"] = mesh.box_z_min;
    out["box_z_max"] = mesh.box_z_max;
    out["polar_prefix_nr"] = mesh.polar_prefix_nr;
  }
  if (mesh.logical_mesh_2d == "cone_shell") {
    out["box_r_max"] = mesh.box_r_max;
    out["box_z_min"] = mesh.box_z_min;
    out["box_z_max"] = mesh.box_z_max;
    out["cone_shell_alpha"] = nan_safe(mesh.cone_shell_alpha);
    out["cone_shell_wall_thickness"] =
        nan_safe(mesh.cone_shell_wall_thickness);
    out["cone_shell_tip_radius"] = nan_safe(mesh.cone_shell_tip_radius);
    out["cone_shell_tip_radius_kind"] = mesh.cone_shell_tip_radius_kind;
    out["cone_shell_tip_z"] = nan_safe(mesh.cone_shell_tip_z);
    out["cone_shell_wall_length"] = nan_safe(mesh.cone_shell_wall_length);
    out["cone_shell_axis_sign"] = mesh.cone_shell_axis_sign;
    out["cone_shell_n_cells"] = mesh.cone_shell_n_cells;
    out["cone_shell_n_growth"] = mesh.cone_shell_n_growth;
    out["cone_shell_tip_size_factor"] = mesh.cone_shell_tip_size_factor;
    out["cone_shell_base_size_factor"] = mesh.cone_shell_base_size_factor;
    out["cone_shell_tip_hold"] = nan_safe(mesh.cone_shell_tip_hold);
    out["cone_shell_grading_length"] =
        nan_safe(mesh.cone_shell_grading_length);
    out["cone_shell_l_ratio_max"] = mesh.cone_shell_l_ratio_max;
    out["cone_shell_tip_rotation_length"] =
        nan_safe(mesh.cone_shell_tip_rotation_length);
    out["cone_shell_base_cut"] = mesh.cone_shell_base_cut;
    out["cone_shell_base_rotation_length"] =
        nan_safe(mesh.cone_shell_base_rotation_length);
    out["cone_shell_farfield_target_measure"] =
        mesh.cone_shell_farfield_target_measure;
    out["cone_shell_outer_vac_first_factor"] =
        mesh.cone_shell_outer_vac_first_factor;
    out["cone_shell_outer_vac_layers"] = mesh.cone_shell_outer_vac_layers;
    out["cone_shell_outer_vac_growth"] = mesh.cone_shell_outer_vac_growth;
    out["cone_shell_inner_vac_first_factor"] =
        mesh.cone_shell_inner_vac_first_factor;
    out["cone_shell_inner_vac_layers"] = mesh.cone_shell_inner_vac_layers;
    out["cone_shell_inner_vac_growth"] = mesh.cone_shell_inner_vac_growth;
    out["cone_shell_end_vac_first_factor"] =
        mesh.cone_shell_end_vac_first_factor;
    out["cone_shell_end_vac_layers"] = mesh.cone_shell_end_vac_layers;
    out["cone_shell_end_vac_growth"] = mesh.cone_shell_end_vac_growth;
  }
  if (mesh.logical_mesh_2d != "cone_shell") {
    out["box_center_z"] = mesh.box_center_z;
  }
  out["cone_theta_wall"] = nan_safe(mesh.cone_theta_wall);
  out["cone_tip_radius"] = nan_safe(mesh.cone_tip_radius);
  out["cone_activation_radius"] = nan_safe(mesh.cone_activation_radius);
  out["cone_fine_cells_minus"] = mesh.cone_fine_cells_minus;
  out["cone_fine_cells_plus"] = mesh.cone_fine_cells_plus;
  out["cone_angular_growth_max"] = mesh.cone_angular_growth_max;
  out["cone_tip_style"] = mesh.cone_tip_style;
  out["morph_rings"] = mesh.morph_rings;
  out["collar_rings"] = mesh.collar_rings;
  out["morph_growth_max"] = mesh.morph_growth_max;
  out["spherical_polar_kappa"] = mesh.spherical_polar_kappa;
  const Config::MeshConfig defaults;
  if (mesh.topology_scheme == TopologyScheme::SINGLE_BLOCK) {
    if (mesh.topology_scheme_explicit) {
      out["topology_scheme"] =
          topology_scheme_to_string(mesh.topology_scheme);
    }
  } else if (mesh.topology_scheme == TopologyScheme::CONE_SHELL_SPINE) {
    out["topology_scheme"] = topology_scheme_to_string(mesh.topology_scheme);
  } else if (mesh.topology_scheme == TopologyScheme::PENTAGON_BELT_SHELL) {
    out["topology_scheme"] = topology_scheme_to_string(mesh.topology_scheme);
    out["pentagon_belt_layers"] = py::cast(mesh.pentagon_belt_layers);
  } else {
    out["topology_scheme"] = topology_scheme_to_string(mesh.topology_scheme);
    out["multiblock_cart_core_r_c"] = mesh.multiblock_cart_core_r_c;
    out["multiblock_cart_core_r_match"] = mesh.multiblock_cart_core_r_match;
    out["multiblock_cart_core_n_c"] = mesh.multiblock_cart_core_n_c;
    out["multiblock_cart_core_bridge_layers"] =
        mesh.multiblock_cart_core_bridge_layers;
    out["multiblock_cart_core_bridge_grading"] =
        mesh.multiblock_cart_core_bridge_grading;
    if (mesh.multiblock_transition_scheme !=
        defaults.multiblock_transition_scheme) {
      out["multiblock_transition_scheme"] =
          multiblock_transition_scheme_to_string(
              mesh.multiblock_transition_scheme);
    }
    if (mesh.multiblock_cap_p != defaults.multiblock_cap_p) {
      out["multiblock_cap_p"] = mesh.multiblock_cap_p;
    }
    if (mesh.multiblock_bridge_elliptic_sweeps !=
        defaults.multiblock_bridge_elliptic_sweeps) {
      out["multiblock_bridge_elliptic_sweeps"] =
          mesh.multiblock_bridge_elliptic_sweeps;
    }
    if (mesh.multiblock_bridge_elliptic_omega !=
        defaults.multiblock_bridge_elliptic_omega) {
      out["multiblock_bridge_elliptic_omega"] =
          mesh.multiblock_bridge_elliptic_omega;
    }
    if (mesh.multiblock_outer_svec_tangent_balance !=
        defaults.multiblock_outer_svec_tangent_balance) {
      out["multiblock_outer_svec_tangent_balance"] =
          mesh.multiblock_outer_svec_tangent_balance;
    }
  }
  if (!mesh.grid_segments.empty()) {
    py::list segs;
    for (const auto& seg : mesh.grid_segments) {
      py::dict s;
      s["r_start"] = seg.r_start;
      s["r_end"] = seg.r_end;
      s["nr"] = seg.nr;
      segs.append(std::move(s));
    }
    out["grid_segments"] = segs;
  }
  if (!mesh.grid_segments_z.empty()) {
    py::list segs;
    for (const auto& seg : mesh.grid_segments_z) {
      py::dict s;
      s["r_start"] = seg.r_start;
      s["r_end"] = seg.r_end;
      s["nr"] = seg.nr;
      segs.append(std::move(s));
    }
    out["grid_segments_z"] = segs;
  }
  if (!mesh.grid_segments_theta.empty()) {
    py::list segs;
    for (const auto& seg : mesh.grid_segments_theta) {
      py::dict s;
      s["r_start"] = seg.r_start;
      s["r_end"] = seg.r_end;
      s["nr"] = seg.nr;
      segs.append(std::move(s));
    }
    out["grid_segments_theta"] = segs;
  }
  if (!mesh.auto_regions.empty()) {
    py::list regions;
    for (const auto& region : mesh.auto_regions) {
      py::dict r;
      r["r_end"] = region.r_end;
      r["nz"] = region.nz;
      r["rho_ref"] = region.rho_ref;
      r["is_void"] = region.is_void;
      r["material_group"] = region.material_group;
      regions.append(std::move(r));
    }
    out["auto_regions"] = regions;
  }
  if (mesh.zoning_intent.enabled) {
    const auto& zoning = mesh.zoning_intent;
    py::dict intent;
    intent["n_cells"] = zoning.n_cells;
    intent["measure"] = zoning.measure;
    intent["dr_min"] = zoning.dr_min;
    intent["cell_measure_min"] = zoning.cell_measure_min;
    intent["cell_measure_max"] = zoning.cell_measure_max;
    intent["preferred_ratio"] = zoning.preferred_ratio;
    intent["ratio_hard_max"] = zoning.ratio_hard_max;
    intent["min_cells_per_segment"] = zoning.min_cells_per_segment;

    py::list pins;
    for (const auto& pin : zoning.pins) {
      py::dict p;
      p["r"] = pin.r;
      p["ratio_jump_allowed"] = pin.ratio_jump_allowed;
      pins.append(std::move(p));
    }
    intent["pins"] = pins;

    py::list profile;
    for (const auto& point : zoning.profile) {
      py::dict p;
      p["r"] = point.r;
      p["w"] = point.w;
      profile.append(std::move(p));
    }
    intent["profile"] = profile;

    py::list anchors;
    for (const auto& anchor : zoning.anchors) {
      py::dict a;
      a["r"] = anchor.r;
      a["half_width"] = anchor.half_width;
      a["log_amplitude"] = anchor.log_amplitude;
      anchors.append(std::move(a));
    }
    intent["anchors"] = anchors;

    py::list bands;
    for (const auto& band : zoning.bands) {
      py::dict b;
      b["measure_frac_begin"] = band.measure_frac_begin;
      b["measure_frac_end"] = band.measure_frac_end;
      b["cell_measure_min"] = band.cell_measure_min;
      b["cell_measure_max"] = band.cell_measure_max;
      bands.append(std::move(b));
    }
    intent["bands"] = bands;

    py::list density_regions;
    for (const auto& region : zoning.density_regions) {
      py::dict r;
      r["r_end"] = region.r_end;
      r["rho"] = region.rho;
      density_regions.append(std::move(r));
    }
    intent["density_regions"] = density_regions;
    intent["extra_events"] =
        serialize_double_list_17g(zoning.extra_events);
    out["zoning_intent"] = intent;
  }
  if (mesh.auto_regions_axis != "r") {
    out["auto_regions_axis"] = mesh.auto_regions_axis;
  }
  if (auto_config_has_non_default(mesh.auto_config)) {
    py::dict auto_cfg;
    auto_cfg["mass_ratio_max"] = mesh.auto_config.mass_ratio_max;
    auto_cfg["n_bridge_min"] = mesh.auto_config.n_bridge_min;
    auto_cfg["n_bridge_max"] = mesh.auto_config.n_bridge_max;
    auto_cfg["bridge_frac_max"] = mesh.auto_config.bridge_frac_max;
    auto_cfg["rho_void_cut"] = mesh.auto_config.rho_void_cut;
    auto_cfg["dr_min"] = mesh.auto_config.dr_min;
    auto_cfg["mass_ratio_hard_max"] = mesh.auto_config.mass_ratio_hard_max;
    auto_cfg["max_iter"] = mesh.auto_config.max_iter;
    auto_cfg["bulk_mass_tol"] = mesh.auto_config.bulk_mass_tol;
    out["auto_config"] = auto_cfg;
  }
  out["grading"] = grading;
  out["explicit_nodes"] = serialize_double_list_17g(mesh.explicit_nodes);
  out["explicit_nodes_z"] =
      serialize_double_list_17g(mesh.explicit_nodes_z);
  out["explicit_nodes_theta"] =
      serialize_double_list_17g(mesh.explicit_nodes_theta);
  out["motion"] = mesh.motion;
  out["floors"] = floors;
  return out;
}

py::dict serialize_materials(const Config::MaterialsConfig& materials) {
  py::list material_list;
  for (const auto& mat : materials.materials) {
    py::dict m;
    m["name"] = mat.name;
    m["A"] = mat.A;
    m["Z"] = mat.Z;
    m["eos_model"] = mat.eos_model;
    m["eos_file"] = mat.eos_file;
    m["sesame_material_id"] = mat.sesame_material_id;
    m["ideal_gas_gamma"] = mat.ideal_gas_gamma;
    m["cv_e_override"] = mat.cv_e_override;
    m["eos_T_ref_eV"] = mat.eos_T_ref_eV;
    if (mat.eos_model == "power_law_te") {
      m["eos_power_law_f_erg_g"] = mat.eos_power_law_f_erg_g;
      m["eos_power_law_beta"] = mat.eos_power_law_beta;
      m["eos_power_law_mu_rho"] = mat.eos_power_law_mu_rho;
      m["eos_power_law_gamma_p"] = mat.eos_power_law_gamma_p;
      m["eos_power_law_step_D_erg_g_eV"] = mat.eos_power_law_step_D_erg_g_eV;
      m["eos_power_law_step_Tc_eV"] = mat.eos_power_law_step_Tc_eV;
      m["eos_power_law_step_w_eV"] = mat.eos_power_law_step_w_eV;
    }
    m["hydro_eos_backend"] = mat.hydro_eos_backend;
    m["mg_T_ref_eV"] = mat.mg_T_ref_eV;
    m["mg_dT_rel"] = mat.mg_dT_rel;
    m["opacity_model"] = mat.opacity_model;
    m["opacity_file"] = mat.opacity_file;
    m["kappa_a_constant"] = mat.kappa_a_constant;
    if (mat.kappa_planck_override >= 0.0) {
      m["kappa_planck_override"] = mat.kappa_planck_override;
    }
    m["kappa_s_constant"] = mat.kappa_s_constant;
    if (mat.opacity_model == "power_law") {
      m["opacity_power_law_kappa0_cm2_g"] = mat.opacity_power_law_kappa0_cm2_g;
      m["opacity_power_law_alpha_T"] = mat.opacity_power_law_alpha_T;
      m["opacity_power_law_lambda_rho"] = mat.opacity_power_law_lambda_rho;
      m["opacity_power_law_T_ref_eV"] = mat.opacity_power_law_T_ref_eV;
      m["opacity_power_law_rho_ref_g_cc"] = mat.opacity_power_law_rho_ref_g_cc;
    }
    m["opacity_units"] = mat.opacity_units;
    m["lambda_method"] = mat.lambda_method;
    m["lambda_fd_delta_rel"] = mat.lambda_fd_delta_rel;
    m["lambda_fd_abs_min"] = mat.lambda_fd_abs_min;
    m["f_min"] = mat.nlte_f_min;
    material_list.append(std::move(m));
  }

  py::dict zbar;
  zbar["model"] = materials.zbar.model;
  zbar["fixed_value"] = materials.zbar.fixed_value;
  zbar["table_file"] = materials.zbar.table_file;

  py::dict out;
  out["materials"] = material_list;
  out["opacity_mix_rule"] = materials.opacity_mix_rule;
  out["low_density_extrapolation"] = materials.low_density_extrapolation;
  out["zbar"] = zbar;
  return out;
}

py::dict serialize_geometry(const Config::GeometryConfig& geometry) {
  py::dict volfrac;
  for (const auto& [name, callable] : geometry.volfrac) {
    volfrac[py::str(name)] = serialize_callable(callable);
  }

  py::dict out;
  out["rho"] = serialize_callable(geometry.rho);
  out["Te"] = serialize_callable(geometry.Te);
  out["Ti"] = serialize_callable(geometry.Ti);
  out["velocity"] = serialize_callable(geometry.velocity);
  out["volfrac"] = volfrac;
  out["radiation_field"] = geometry.radiation_field;
  out["radiation_field_Tr_eV"] = geometry.radiation_field_Tr_eV;
  out["enforce_sum_to_one"] = geometry.enforce_sum_to_one;
  return out;
}

py::dict serialize_radiation(const Config::RadiationConfig& radiation) {
  py::dict imc;
  py::dict census_comb;
  census_comb["enabled"] = radiation.imc.census_comb.enabled;
  census_comb["max_particles"] = radiation.imc.census_comb.max_particles;
  census_comb["min_per_bin"] = radiation.imc.census_comb.min_per_bin;
  census_comb["trigger_ratio"] = radiation.imc.census_comb.trigger_ratio;
  census_comb["target_fraction"] = radiation.imc.census_comb.target_fraction;
  census_comb["mode_weight_imc"] = radiation.imc.census_comb.mode_weight_imc;
  census_comb["mode_weight_ddmc"] = radiation.imc.census_comb.mode_weight_ddmc;
  census_comb["adaptive_trigger"] = radiation.imc.census_comb.adaptive_trigger;
  census_comb["adaptive_util_start"] = radiation.imc.census_comb.adaptive_util_start;
  census_comb["adaptive_util_end"] = radiation.imc.census_comb.adaptive_util_end;
  census_comb["trigger_ratio_floor"] = radiation.imc.census_comb.trigger_ratio_floor;
  census_comb["trigger_hysteresis"] = radiation.imc.census_comb.trigger_hysteresis;
  census_comb["ess_floor_enabled"] = radiation.imc.census_comb.ess_floor_enabled;
  census_comb["ess_min_tier0"] = radiation.imc.census_comb.ess_min_tier0;
  census_comb["ess_min_tier1"] = radiation.imc.census_comb.ess_min_tier1;
  census_comb["max_split_factor"] = radiation.imc.census_comb.max_split_factor;
  imc["enabled"] = radiation.imc.enabled;
  imc["alpha"] = radiation.imc.alpha;
  imc["f_max"] = radiation.imc.f_max;
  imc["corrected_fleck"] = radiation.imc.corrected_fleck;
  imc["particles_per_cell_group"] = radiation.imc.particles_per_cell_group;
  imc["implicit_capture"] = radiation.imc.implicit_capture;
  imc["cutoff_fraction"] = radiation.imc.cutoff_fraction;
  imc["inelastic_scatter"] = radiation.imc.inelastic_scatter;
  imc["weight_cutoff"] = radiation.imc.weight_cutoff;
  imc["roulette_survival"] = radiation.imc.roulette_survival;
  imc["weight_split"] = radiation.imc.weight_split;
  imc["max_split"] = radiation.imc.max_split;
  imc["linearized_planck"] = radiation.imc.linearized_planck;
  imc["source_tilting"] = radiation.imc.source_tilting;
  imc["source_localization"] = radiation.imc.source_localization;
  imc["sloc_ema_beta"] = radiation.imc.sloc_ema_beta;
  imc["sloc_sigma_floor"] = radiation.imc.sloc_sigma_floor;
  imc["sloc_sigma_cap"] = radiation.imc.sloc_sigma_cap;
  imc["sloc_tau_ref"] = radiation.imc.sloc_tau_ref;
  imc["spectral_bias_eta"] = radiation.imc.spectral_bias_eta;
  imc["opacity_predictor"] = radiation.imc.opacity_predictor;
  imc["two_stage"] = radiation.imc.two_stage;
  py::dict difference;
  difference["enabled"] = radiation.imc.difference.enabled;
  difference["W_max"] = radiation.imc.difference.W_max;
  difference["tau0"] = radiation.imc.difference.tau0;
  difference["chi0"] = radiation.imc.difference.chi0;
  difference["face_transport"] = radiation.imc.difference.face_transport;
  imc["difference"] = difference;
  py::dict net_e_source_smoothing;
  net_e_source_smoothing["enabled"] = radiation.imc.net_e_source_smoothing.enabled;
  net_e_source_smoothing["alpha"] = radiation.imc.net_e_source_smoothing.alpha;
  net_e_source_smoothing["tau_threshold"] =
      radiation.imc.net_e_source_smoothing.tau_threshold;
  net_e_source_smoothing["passes"] = radiation.imc.net_e_source_smoothing.passes;
  net_e_source_smoothing["grad_Te_scale"] =
      radiation.imc.net_e_source_smoothing.grad_Te_scale;
  net_e_source_smoothing["grad_rho_scale"] =
      radiation.imc.net_e_source_smoothing.grad_rho_scale;
  net_e_source_smoothing["gradient_adaptive"] =
      radiation.imc.net_e_source_smoothing.gradient_adaptive;
  imc["net_e_source_smoothing"] = net_e_source_smoothing;
  imc["particle_budget"] = radiation.imc.particle_budget;
  imc["census_comb"] = census_comb;
  py::dict rad_lite_mesh;
  rad_lite_mesh["enabled"] = radiation.imc.rad_lite_mesh.enabled;
  rad_lite_mesh["sigma_ratio_max"] = radiation.imc.rad_lite_mesh.sigma_ratio_max;
  rad_lite_mesh["nlte_auto"] = radiation.imc.rad_lite_mesh.nlte_auto;
  imc["rad_lite_mesh"] = rad_lite_mesh;

  py::dict ddmc;
  ddmc["enabled"] = radiation.ddmc.enabled;
  ddmc["implicit_diffusion"] = radiation.ddmc.implicit_diffusion;
  ddmc["tau_ddmc"] = radiation.ddmc.tau_ddmc;
  ddmc["tau_rw"] = radiation.ddmc.tau_rw;
  ddmc["omega_ddmc"] = radiation.ddmc.omega_ddmc;
  ddmc["tau_ddmc_off"] = radiation.ddmc.tau_ddmc_off;
  ddmc["omega_ddmc_off"] = radiation.ddmc.omega_ddmc_off;
  ddmc["mode_hold"] = radiation.ddmc.mode_hold;
  ddmc["rate_max"] = radiation.ddmc.rate_max;
  ddmc["leak_stencil"] = radiation.ddmc.leak_stencil;
  ddmc["interface_method"] = radiation.ddmc.interface_method;
  ddmc["emissivity_preserving"] = radiation.ddmc.emissivity_preserving;
  ddmc["interface_exit_distribution"] = radiation.ddmc.interface_exit_distribution;
  ddmc["rz_face_r_weight"] = radiation.ddmc.rz_face_r_weight;
  ddmc["face_opacity_temperature"] = radiation.ddmc.face_opacity_temperature;
  ddmc["m_matrix_check"] = radiation.ddmc.m_matrix_check;

  py::dict diffusion;
  diffusion["enabled"] = radiation.diffusion.enabled;
  diffusion["tau_on"] = radiation.diffusion.tau_on;
  diffusion["tau_off"] = radiation.diffusion.tau_off;
  diffusion["reduced_flux_on"] = radiation.diffusion.reduced_flux_on;
  diffusion["reduced_flux_off"] = radiation.diffusion.reduced_flux_off;
  diffusion["mode_hold"] = radiation.diffusion.mode_hold;
  diffusion["rate_max"] = radiation.diffusion.rate_max;
  diffusion["mode_update_interval"] = radiation.diffusion.mode_update_interval;
  diffusion["min_diffusion_island_cells"] =
      radiation.diffusion.min_diffusion_island_cells;
  diffusion["imc_guard_cells"] = radiation.diffusion.imc_guard_cells;
  diffusion["sts_max_stages"] = radiation.diffusion.sts_max_stages;
  diffusion["sts_damping"] = radiation.diffusion.sts_damping;
  diffusion["sts_subcycle_eta"] = radiation.diffusion.sts_subcycle_eta;
  diffusion["interface_particles_per_face_group"] =
      radiation.diffusion.interface_particles_per_face_group;
  diffusion["exit_particles_per_cell_group"] =
      radiation.diffusion.exit_particles_per_cell_group;
  diffusion["lte_entry_initialization"] =
      radiation.diffusion.lte_entry_initialization;
  diffusion["lte_entry_energy_fraction_cap"] =
      radiation.diffusion.lte_entry_energy_fraction_cap;

  py::dict fld_boundary;
  fld_boundary["inner_r"] = radiation.multigroup_diffusion.boundary.inner_r;
  fld_boundary["outer_r"] = radiation.multigroup_diffusion.boundary.outer_r;
  fld_boundary["z"] = radiation.multigroup_diffusion.boundary.z;
  fld_boundary["z_bottom"] =
      radiation.multigroup_diffusion.boundary.z_bottom;
  fld_boundary["z_top"] = radiation.multigroup_diffusion.boundary.z_top;
  py::dict fld_marshak;
  fld_marshak["flux_erg_per_cm2_s"] =
      radiation.multigroup_diffusion.marshak.flux_erg_per_cm2_s;
  fld_marshak["flux_pulse_duration_s"] =
      radiation.multigroup_diffusion.marshak.flux_pulse_duration_s;

  py::dict multigroup_diffusion;
  multigroup_diffusion["flux_limiter"] =
      radiation.multigroup_diffusion.flux_limiter;
  multigroup_diffusion["max_outer_iterations"] =
      radiation.multigroup_diffusion.max_outer_iterations;
  multigroup_diffusion["fleck_mode"] =
      radiation.multigroup_diffusion.fleck_mode;
  multigroup_diffusion["fleck_cv_source"] =
      radiation.multigroup_diffusion.fleck_cv_source;
  multigroup_diffusion["fleck_beta"] =
      radiation.multigroup_diffusion.fleck_beta;
  multigroup_diffusion["fleck_form"] =
      radiation.multigroup_diffusion.fleck_form;
  multigroup_diffusion["source_integrator"] =
      radiation.multigroup_diffusion.source_integrator;
  multigroup_diffusion["hydro_coupling"] =
      radiation.multigroup_diffusion.hydro_coupling;
  multigroup_diffusion["outer_tol"] = radiation.multigroup_diffusion.outer_tol;
  multigroup_diffusion["state_supply_boundary_policy"] =
      radiation.multigroup_diffusion.state_supply_boundary_policy;
  multigroup_diffusion["diagnostic_radial_fourier_substage_enabled"] =
      radiation.multigroup_diffusion.diagnostic_radial_fourier_substage_enabled;
  multigroup_diffusion["cg_inner_tol"] =
      radiation.multigroup_diffusion.cg_inner_tol;
  multigroup_diffusion["cg_tol_norm"] =
      radiation.multigroup_diffusion.cg_tol_norm;
  multigroup_diffusion["outer_accel"] =
      radiation.multigroup_diffusion.outer_accel;
  multigroup_diffusion["anderson_m"] =
      radiation.multigroup_diffusion.anderson_m;
  multigroup_diffusion["anderson_beta"] =
      radiation.multigroup_diffusion.anderson_beta;
  multigroup_diffusion["cg_max_iter"] =
      radiation.multigroup_diffusion.cg_max_iter;
  multigroup_diffusion["cap_exit_policy"] =
      radiation.multigroup_diffusion.cap_exit_policy;
  multigroup_diffusion["linear_solver_1d"] =
      radiation.multigroup_diffusion.linear_solver_1d;
  multigroup_diffusion["linear_solver_2d"] =
      radiation.multigroup_diffusion.linear_solver_2d;
  multigroup_diffusion["rgmg_smoother_omega"] =
      radiation.multigroup_diffusion.rgmg_smoother_omega;
  py::dict amgx_config;
  amgx_config["preset"] = radiation.multigroup_diffusion.amgx_config.preset;
  multigroup_diffusion["amgx_config"] = amgx_config;
  multigroup_diffusion["z_boundary"] =
      radiation.multigroup_diffusion.z_boundary;
  multigroup_diffusion["opacity_floor"] =
      radiation.multigroup_diffusion.opacity_floor;
  multigroup_diffusion["opacity_cap"] =
      radiation.multigroup_diffusion.opacity_cap;
  multigroup_diffusion["boundary"] = fld_boundary;
  multigroup_diffusion["marshak"] = fld_marshak;

  py::dict sn_boundary;
  sn_boundary["inner_r"] = radiation.sn_transport.boundary.inner_r;
  sn_boundary["outer_r"] = radiation.sn_transport.boundary.outer_r;
  sn_boundary["z"] = radiation.sn_transport.boundary.z;
  sn_boundary["z_bottom"] = radiation.sn_transport.boundary.z_bottom;
  sn_boundary["z_top"] = radiation.sn_transport.boundary.z_top;

  py::dict sn_marshak;
  sn_marshak["flux_erg_per_cm2_s"] =
      radiation.sn_transport.marshak.flux_erg_per_cm2_s;
  sn_marshak["flux_pulse_duration_s"] =
      radiation.sn_transport.marshak.flux_pulse_duration_s;

  py::dict sn_transport;
  sn_transport["n_angles"] = radiation.sn_transport.n_angles;
  sn_transport["angular_quadrature"] = radiation.sn_transport.angular_quadrature;
  sn_transport["spatial_scheme"] = radiation.sn_transport.spatial_scheme;
  sn_transport["max_outer_iterations"] =
      radiation.sn_transport.max_outer_iterations;
  sn_transport["max_inner_iterations"] =
      radiation.sn_transport.max_inner_iterations;
  sn_transport["outer_tol"] = radiation.sn_transport.outer_tol;
  sn_transport["outer_tol_stagnation_factor"] =
      radiation.sn_transport.outer_tol_stagnation_factor;
  sn_transport["outer_tol_hydro_error_scale"] =
      radiation.sn_transport.outer_tol_hydro_error_scale;
  sn_transport["inner_tol"] = radiation.sn_transport.inner_tol;
  sn_transport["inner_graph_unroll"] = radiation.sn_transport.inner_graph_unroll;
  sn_transport["dsa_enabled"] = radiation.sn_transport.dsa_enabled;
  sn_transport["z_boundary"] = radiation.sn_transport.z_boundary;
  sn_transport["diffusion_fallback_mode"] =
      radiation.sn_transport.diffusion_fallback_mode;
  sn_transport["tau_diffusion_on"] = radiation.sn_transport.tau_diffusion_on;
  sn_transport["tau_diffusion_off"] = radiation.sn_transport.tau_diffusion_off;
  sn_transport["opacity_floor"] = radiation.sn_transport.opacity_floor;
  sn_transport["opacity_cap"] = radiation.sn_transport.opacity_cap;
  sn_transport["timing_enabled"] = radiation.sn_transport.timing_enabled;
  sn_transport["boundary"] = sn_boundary;
  sn_transport["marshak"] = sn_marshak;

  py::dict holo;
  holo["enabled"] = radiation.holo.enabled;
  holo["region"] = radiation.holo.region;
  holo["material_group"] = radiation.holo.material_group;
  holo["coupling_tau"] = radiation.holo.coupling_tau;
  holo["guard_cells"] = radiation.holo.guard_cells;
  holo["blend_cells"] = radiation.holo.blend_cells;
  holo["min_lo_cells"] = radiation.holo.min_lo_cells;
  holo["q_min"] = radiation.holo.q_min;
  holo["q_max"] = radiation.holo.q_max;
  holo["tau_on"] = radiation.holo.tau_on;
  holo["tau_off"] = radiation.holo.tau_off;
  holo["reduced_flux_on"] = radiation.holo.reduced_flux_on;
  holo["reduced_flux_off"] = radiation.holo.reduced_flux_off;
  holo["update_interval"] = radiation.holo.update_interval;
  holo["hold_on"] = radiation.holo.hold_on;
  holo["min_dwell_steps"] = radiation.holo.min_dwell_steps;
  holo["min_island_cells"] = radiation.holo.min_island_cells;
  holo["core_margin_cells"] = radiation.holo.core_margin_cells;
  holo["solver"] = radiation.holo.solver;
  holo["closure"] = radiation.holo.closure;
  holo["closure_relax"] = radiation.holo.closure_relax;
  holo["closure_smooth_passes"] = radiation.holo.closure_smooth_passes;
  holo["closure_smooth_alpha"] = radiation.holo.closure_smooth_alpha;
  holo["consistency_alpha"] = radiation.holo.consistency_alpha;
  holo["gamma_alpha"] = radiation.holo.consistency_alpha;
  holo["boundary_flux"] = radiation.holo.boundary_flux;
  holo["p_rr_tally"] = radiation.holo.p_rr_tally;
  holo["sn_closure"] = radiation.holo.sn_closure;
  holo["sn_n_angles"] = radiation.holo.sn_n_angles;
  holo["sn_material_coupling"] = radiation.holo.sn_material_coupling;
  holo["residual_particles_per_cell_group"] =
      radiation.holo.residual_particles_per_cell_group;

  py::dict marshak_map;
  for (const auto& [face, callable] : radiation.boundary.marshak_Tr_map) {
    marshak_map[py::str(face)] = serialize_callable(callable);
  }

  py::dict boundary;
  boundary["type"] = radiation.boundary.type;
  boundary["inner_r"] = radiation.boundary.inner_r;
  boundary["outer_r"] = radiation.boundary.outer_r;
  boundary["bottom_z"] = radiation.boundary.bottom_z;
  boundary["top_z"] = radiation.boundary.top_z;
  boundary["marshak_particles"] = radiation.boundary.marshak_particles;
  boundary["marshak_Tr_eV"] = nan_safe(radiation.boundary.marshak_Tr_eV);
  boundary["marshak_Tr"] = serialize_callable(radiation.boundary.marshak_Tr);
  boundary["marshak_Tr_map"] = marshak_map;

  py::dict out;
  out["enabled"] = radiation.enabled;
  out["mode"] = (radiation.mode == RadiationMode::MultigroupDiffusion)
                    ? "multigroup_diffusion"
                    : ((radiation.mode == RadiationMode::SnTransport)
                           ? "sn_transport"
                           : "imc_ddmc");
  if (radiation.origin_parity_only) {
    out["origin_parity_only"] = radiation.origin_parity_only;
  }
  if (radiation.group_repack_hard_xray) {
    out["group_repack_hard_xray"] = radiation.group_repack_hard_xray;
  }
  if (radiation.diagnose_hard_xray_opacity) {
    out["diagnose_hard_xray_opacity"] = radiation.diagnose_hard_xray_opacity;
  }
  out["groups"] = radiation.groups;
  out["group_bounds_eV"] = radiation.group_bounds_eV;
  out["compute_T_range_eV"] = radiation.compute_T_range_eV;
  out["volume_source_rate"] = radiation.volume_source_rate;
  out["volume_source_x_max"] = radiation.volume_source_x_max;
  out["imc"] = imc;
  out["ddmc"] = ddmc;
  out["diffusion"] = diffusion;
  out["multigroup_diffusion"] = multigroup_diffusion;
  out["sn_transport"] = sn_transport;
  out["holo"] = holo;
  out["boundary"] = boundary;
  return out;
}

py::dict serialize_laser(const Config::LaserConfig& laser) {
  py::list beams;
  for (const auto& beam : laser.beams) {
    py::dict b;
    b["name"] = beam.name;
    b["direction"] = beam.direction;
    b["theta"] = nan_safe(beam.theta);
    b["phi"] = nan_safe(beam.phi);
    b["f_number"] = beam.f_number;
    b["focus"] = beam.focus;
    b["defocus_DR"] = beam.defocus_DR;
    b["delta_lambda_nm"] = beam.delta_lambda_nm;
    b["profile_model"] = beam.profile_model;
    b["profile_w0_um"] = beam.profile_w0_um;
    b["profile_m"] = beam.profile_m;
    if (!beam.profile_r_cm.empty() && !beam.profile_I.empty()) {
      std::vector<double> profile_r_um;
      profile_r_um.reserve(beam.profile_r_cm.size());
      for (const double r : beam.profile_r_cm) {
        profile_r_um.push_back(r * 1.0e4);
      }
      b["profile_r_um"] = profile_r_um;
      b["profile_I_rel"] = beam.profile_I;
    }
    b["power"] = serialize_callable(beam.power);
    beams.append(std::move(b));
  }

  py::dict profile;
  profile["model"] = laser.profile_model;
  profile["w0_um"] = laser.profile_w0_um;
  profile["m"] = laser.profile_m;
  if (!laser.profile_r_cm.empty() && !laser.profile_I.empty()) {
    std::vector<double> profile_r_um;
    profile_r_um.reserve(laser.profile_r_cm.size());
    for (const double r : laser.profile_r_cm) {
      profile_r_um.push_back(r * 1.0e4);
    }
    profile["profile_r_um"] = profile_r_um;
    profile["profile_I_rel"] = laser.profile_I;
  }

  py::dict out;
  out["enabled"] = laser.enabled;
  out["wavelength_nm"] = laser.wavelength_nm;
  out["mode"] = laser.mode;
  out["rays_per_beam"] = laser.rays_per_beam;
  out["ray_output_count"] = laser.ray_output_count;
  out["ray_output_trajectory"] = laser.ray_output_trajectory;
  out["ray_output_max_steps"] = laser.ray_output_max_steps;
  py::dict absorption;
  absorption["eps_n"] = laser.absorption.eps_n;
  absorption["coulomb_log_floor"] = laser.absorption.coulomb_log_floor;
  absorption["debug_dump_lasermesh"] = laser.absorption.debug_dump_lasermesh;
  out["absorption"] = absorption;
  py::dict lasermesh;
  lasermesh["nr"] = laser.lasermesh.nr;
  lasermesh["nz"] = laser.lasermesh.nz;
  lasermesh["r_max_factor"] = laser.lasermesh.r_max_factor;
  lasermesh["z_span_factor"] = laser.lasermesh.z_span_factor;
  lasermesh["critical_clip"] = laser.lasermesh.critical_clip;
  lasermesh["critical_margin"] = nan_safe(laser.lasermesh.critical_margin);
  lasermesh["stretch_method"] = laser.lasermesh.stretch_method;
  lasermesh["min_ratio"] = laser.lasermesh.min_ratio;
  lasermesh["mesh_factor"] = laser.lasermesh.mesh_factor;
  lasermesh["rmax_n_hat_threshold"] = laser.lasermesh.rmax_n_hat_threshold;
  lasermesh["nr_max"] = laser.lasermesh.nr_max;
  py::dict ghost_corona;
  ghost_corona["enabled"] = laser.lasermesh.ghost_corona.enabled;
  ghost_corona["n_out"] = laser.lasermesh.ghost_corona.n_out;
  ghost_corona["ne_min_frac"] = laser.lasermesh.ghost_corona.ne_min_frac;
  ghost_corona["ne_max_frac"] = laser.lasermesh.ghost_corona.ne_max_frac;
  ghost_corona["Te_min_eV"] = laser.lasermesh.ghost_corona.Te_min_eV;
  ghost_corona["zbar_min"] = laser.lasermesh.ghost_corona.zbar_min;
  ghost_corona["zbar_max"] = laser.lasermesh.ghost_corona.zbar_max;
  ghost_corona["handoff_cells"] = laser.lasermesh.ghost_corona.handoff_cells;
  ghost_corona["handoff_decay"] = laser.lasermesh.ghost_corona.handoff_decay;
  ghost_corona["transition_enabled"] = laser.lasermesh.ghost_corona.transition_enabled;
  ghost_corona["transition_resolved_nhat"] =
      laser.lasermesh.ghost_corona.transition_resolved_nhat;
  ghost_corona["transition_resolved_cells"] =
      laser.lasermesh.ghost_corona.transition_resolved_cells;
  ghost_corona["transition_density_exponent"] =
      laser.lasermesh.ghost_corona.transition_density_exponent;
  lasermesh["ghost_corona"] = ghost_corona;
  out["lasermesh"] = lasermesh;
  py::dict ib;
  ib["zeff_model"] = laser.ib.zeff_model;
  py::list species;
  for (std::size_t i = 0; i < laser.ib.species_z.size(); ++i) {
    py::list entry;
    entry.append(laser.ib.species_z[i]);
    entry.append(laser.ib.species_x[i]);
    species.append(std::move(entry));
  }
  ib["species"] = species;
  ib["coulomb_log_model"] = laser.ib.coulomb_log_model;
  ib["langdon_model"] = laser.ib.langdon_model;
  ib["langdon_te_min_eV"] = laser.ib.langdon_te_min_eV;
  ib["zeff_table_loaded"] = laser.ib.zeff_table.ndens > 0;
  out["ib"] = ib;
  py::dict ra;
  ra["enable"] = laser.ra.enable;
  ra["chi_p"] = laser.ra.chi_p;
  ra["c_ra"] = laser.ra.c_ra;
  out["ra"] = ra;
  py::dict raytrace;
  raytrace["cfl_ray"] = laser.raytrace.cfl_ray;
  raytrace["intensity_cutoff"] = laser.raytrace.intensity_cutoff;
  raytrace["eps_crit"] = laser.raytrace.eps_crit;
  raytrace["max_steps"] = laser.raytrace.max_steps;
  raytrace["integrator"] = laser.raytrace.integrator;
  raytrace["test_kappa"] = laser.raytrace.test_kappa;
  raytrace["ds_adapt_g_target"] = laser.raytrace.ds_adapt_g_target;
  raytrace["ds_adapt_tau_target"] = laser.raytrace.ds_adapt_tau_target;
  raytrace["ds_adapt_max_factor"] = laser.raytrace.ds_adapt_max_factor;
  raytrace["debug_one_ray"] = laser.raytrace.debug_one_ray;
  out["raytrace"] = raytrace;
  out["raytrace_skip"] = laser.raytrace_skip;
  py::dict raytrace_skip_config;
  raytrace_skip_config["enabled"] = laser.raytrace_skip_config.enabled;
  raytrace_skip_config["threshold"] = laser.raytrace_skip_config.threshold;
  raytrace_skip_config["max_consecutive"] = laser.raytrace_skip_config.max_consecutive;
  raytrace_skip_config["norm"] = laser.raytrace_skip_config.norm;
  raytrace_skip_config["crit_guard"] = laser.raytrace_skip_config.crit_guard;
  out["raytrace_skip_config"] = raytrace_skip_config;
  py::dict deposit;
  deposit["conservation_tol"] = laser.deposit.conservation_tol;
  deposit["deposit_smooth_passes"] = laser.deposit.deposit_smooth_passes;
  deposit["deposit_smooth_alpha"] = laser.deposit.deposit_smooth_alpha;
  out["deposit"] = deposit;
  if (!laser.port_configuration.ports.empty()) {
    py::dict port_configuration;
    port_configuration["normalization"] =
        laser.port_configuration.normalization;
    py::list ports;
    for (const auto& port : laser.port_configuration.ports) {
      py::dict p;
      p["port_id"] = port.port_id;
      p["direction"] = port.direction;
      p["roll_deg"] = port.roll_deg;
      p["power_weight"] = port.power_weight;
      p["delta_lambda_nm"] = port.delta_lambda_nm;
      p["beam_class"] = port.beam_class;
      p["polarization"] = port.polarization;
      ports.append(std::move(p));
    }
    port_configuration["ports"] = std::move(ports);
    out["port_configuration"] = std::move(port_configuration);
  }
  py::dict cbet;
  cbet["enable"] = laser.cbet.enable;
  cbet["f_cbet"] = laser.cbet.f_cbet;
  cbet["alpha_iaw"] = laser.cbet.alpha_iaw;
  cbet["theta_cap"] = laser.cbet.theta_cap;
  cbet["tol"] = laser.cbet.tol;
  cbet["max_iters"] = laser.cbet.max_iters;
  cbet["n_impact_bins"] = laser.cbet.n_impact_bins;
  cbet["n_phi"] = laser.cbet.n_phi;
  cbet["ne_frac_cutoff"] = laser.cbet.ne_frac_cutoff;
  cbet["k_a_floor"] = laser.cbet.k_a_floor;
  cbet["max_segments_per_ray"] = laser.cbet.max_segments_per_ray;
  cbet["test_chi"] = laser.cbet.test_chi;
  if (laser.cbet.geometry_mode != "legacy") {
    cbet["geometry_mode"] = laser.cbet.geometry_mode;
  }
  if (laser.cbet.geometry_mode == "port_section") {
    cbet["n_section_phi"] = laser.cbet.n_section_phi;
  }
  out["cbet"] = std::move(cbet);
  py::dict hot_electron;
  hot_electron["enable"] = laser.hot_electron.enable;
  hot_electron["source_nc_fraction"] = laser.hot_electron.source_nc_fraction;
  hot_electron["eta_hot"] = laser.hot_electron.eta_hot;
  hot_electron["eta_hot_table"] = serialize_callable(laser.hot_electron.eta_hot_table);
  if (laser.hot_electron.eta_mode != "legacy") {
    hot_electron["eta_mode"] = laser.hot_electron.eta_mode;
  }
  if (laser.hot_electron.eta_mode == "model") {
    py::dict eta_model;
    eta_model["ln_filter_tau_s"] = laser.hot_electron.eta_model.ln_filter_tau_s;
    eta_model["eta_total_cap"] = laser.hot_electron.eta_model.eta_total_cap;
    hot_electron["eta_model"] = std::move(eta_model);
  }
  if (laser.hot_electron.tpd_overlap_mode != "single_beam") {
    hot_electron["tpd_overlap_mode"] =
        laser.hot_electron.tpd_overlap_mode;
  }
  if (laser.hot_electron.srs_overlap_mode != "per_beam_class") {
    hot_electron["srs_overlap_mode"] =
        laser.hot_electron.srs_overlap_mode;
  }
  if (laser.hot_electron.illumination_metric != "fixed") {
    hot_electron["illumination_metric"] =
        laser.hot_electron.illumination_metric;
  }
  if (laser.hot_electron.tpd_overlap_mode == "common_wave_cluster") {
    hot_electron["common_wave_delta_theta_deg"] =
        laser.hot_electron.common_wave_delta_theta_deg;
  }
  hot_electron["T_hot_eV"] = laser.hot_electron.T_hot_eV;
  hot_electron["n_energy_groups"] = laser.hot_electron.n_energy_groups;
  hot_electron["E_min_over_Th"] = laser.hot_electron.E_min_over_Th;
  hot_electron["E_max_over_Th"] = laser.hot_electron.E_max_over_Th;
  hot_electron["angular_model"] = laser.hot_electron.angular_model;
  hot_electron["theta_div_deg"] = laser.hot_electron.theta_div_deg;
  hot_electron["n_mu"] = laser.hot_electron.n_mu;
  hot_electron["n_phi"] = laser.hot_electron.n_phi;
  hot_electron["subtract_from_laser"] = laser.hot_electron.subtract_from_laser;
  hot_electron["inner_bc"] = laser.hot_electron.inner_bc;
  hot_electron["explicit_source_limit"] = laser.hot_electron.explicit_source_limit;
  if (laser.hot_electron.sources_specified) {
    py::list sources;
    for (const auto& channel : laser.hot_electron.sources) {
      py::dict c;
      c["mechanism"] = channel.mechanism;
      c["capture_nc_fraction"] = channel.capture_nc_fraction;
      c["eta"] = channel.eta;
      c["eta_table"] = serialize_callable(channel.eta_table);
      if (laser.hot_electron.eta_mode == "model") {
        c["eval_nc_fraction"] = channel.eval_nc_fraction;
        c["threshold_multiplier"] = channel.threshold_multiplier;
        c["eta_inf"] = channel.eta_inf;
        c["eta_hard_cap"] = channel.eta_hard_cap;
        c["shape_coefficient"] = channel.shape_coefficient;
        c["relaxation_model"] = channel.relaxation_model;
        c["relaxation_tau_s"] = channel.relaxation_tau_s;
        c["relaxation_tau_min_s"] = channel.relaxation_tau_min_s;
        c["relaxation_tau_max_s"] = channel.relaxation_tau_max_s;
      }
      c["T_hot_eV"] = channel.T_hot_eV;
      c["n_energy_groups"] = channel.n_energy_groups;
      c["E_min_over_Th"] = channel.E_min_over_Th;
      c["E_max_over_Th"] = channel.E_max_over_Th;
      if (channel.mechanism == "tpd") {
        c["tpd_theta_deg"] = channel.tpd_theta_deg;
        c["tpd_delta_deg"] = channel.tpd_delta_deg;
      } else {
        c["theta_div_deg"] = channel.theta_div_deg;
      }
      c["n_mu"] = channel.n_mu;
      c["n_phi"] = channel.n_phi;
      sources.append(std::move(c));
    }
    hot_electron["sources"] = std::move(sources);
  }
  out["hot_electron"] = std::move(hot_electron);
  out["profile"] = profile;
  out["beams"] = beams;
  return out;
}

py::dict serialize_numerics(const Config::NumericsConfig& numerics) {
  py::dict dt;
  dt["initial_s"] = numerics.dt.initial_s;
  dt["cfl_hydro"] = numerics.dt.cfl_hydro;
  dt["cfl_cond"] = numerics.dt.cfl_cond;
  dt["f_min_fleck"] = numerics.dt.f_min_fleck;
  dt["growth_factor"] = numerics.dt.growth_factor;
  dt["max_s"] = numerics.dt.max_s;
  dt["min_s"] = numerics.dt.min_s;
  dt["floor_stall_max_consecutive_steps"] =
      numerics.dt.floor_stall_max_consecutive_steps;

  py::dict persistent_loop;
  persistent_loop["enabled"] = numerics.persistent_loop.enabled;
  persistent_loop["chunk_steps"] = numerics.persistent_loop.chunk_steps;

  py::dict hydro_boundary_2d;
  hydro_boundary_2d["r_inner"] = numerics.hydro.boundary_2d.r_inner;
  hydro_boundary_2d["r_outer"] = numerics.hydro.boundary_2d.r_outer;
  hydro_boundary_2d["mesh_tangential_target"] =
      numerics.hydro.boundary_2d.mesh_tangential_target;
  hydro_boundary_2d["state_supply_donor_mode"] =
      numerics.hydro.boundary_2d.state_supply_donor_mode;
  auto serialize_hydro_z_boundary =
      [](const Config::NumericsConfig::HydroConfig::Boundary2D::ZFaceConfig& face) {
        if (!face.is_state_supply()) {
          return py::object(py::str(face.type));
        }
        py::dict out;
        out["type"] = face.type;
        out["rho_g_per_cc"] = face.supply_rho_g_per_cc;
        out["u_z_cm_per_s"] = face.supply_u_z_cm_per_s;
        out["T_eV"] = face.supply_T_eV;
        return py::object(out);
      };
  hydro_boundary_2d["z_bottom"] =
      serialize_hydro_z_boundary(numerics.hydro.boundary_2d.z_bottom_cfg);
  hydro_boundary_2d["z_top"] =
      serialize_hydro_z_boundary(numerics.hydro.boundary_2d.z_top_cfg);

  auto serialize_adaptive_av_coeff =
      [](const Config::NumericsConfig::HydroConfig::AdaptiveAVCoeff& coeff) {
        py::dict out;
        out["c1"] = coeff.c1;
        out["c2"] = coeff.c2;
        out["heat_C"] = coeff.heat_C;
        out["Cpsv"] = coeff.Cpsv;
        out["cbulk"] = coeff.cbulk;
        return out;
      };

  py::dict adaptive_av;
  adaptive_av["enabled"] = numerics.hydro.adaptive_av.enabled;
  adaptive_av["base"] =
      serialize_adaptive_av_coeff(numerics.hydro.adaptive_av.base);
  adaptive_av["primary"] =
      serialize_adaptive_av_coeff(numerics.hydro.adaptive_av.primary);
  adaptive_av["rebound"] =
      serialize_adaptive_av_coeff(numerics.hydro.adaptive_av.rebound);
  adaptive_av["taper_r_start"] = numerics.hydro.adaptive_av.taper_r_start;
  adaptive_av["taper_r_end"] = numerics.hydro.adaptive_av.taper_r_end;
  adaptive_av["hysteresis_w"] = numerics.hydro.adaptive_av.hysteresis_w;
  adaptive_av["hysteresis_tau"] = numerics.hydro.adaptive_av.hysteresis_tau;
  adaptive_av["support_ahead"] = numerics.hydro.adaptive_av.support_ahead;
  adaptive_av["support_behind"] = numerics.hydro.adaptive_av.support_behind;

  py::dict hourglass;
  hourglass["enabled"] = numerics.hydro.hourglass.enabled;
  hourglass["scale"] = numerics.hydro.hourglass.scale;
  hourglass["compatible_work_enabled"] =
      numerics.hydro.hourglass.compatible_work_enabled;
  hourglass["activation_corner_j_ratio_threshold"] =
      numerics.hydro.hourglass.activation_corner_j_ratio_threshold;
  hourglass["activation_hourglass_amplitude_threshold"] =
      numerics.hydro.hourglass.activation_hourglass_amplitude_threshold;
  hourglass["subzonal_pressure_model"] =
      numerics.hydro.hourglass.subzonal_pressure_model;
  hourglass["max_force_per_node_fraction"] =
      numerics.hydro.hourglass.max_force_per_node_fraction;

  py::dict plasma_viscosity;
  plasma_viscosity["enabled"] = numerics.hydro.plasma_viscosity.enabled;
  plasma_viscosity["model"] = numerics.hydro.plasma_viscosity.model;
  plasma_viscosity["species"] = numerics.hydro.plasma_viscosity.species;
  plasma_viscosity["eta_const"] = numerics.hydro.plasma_viscosity.eta_const;
  plasma_viscosity["eta0_scale"] = numerics.hydro.plasma_viscosity.eta0_scale;
  plasma_viscosity["mfp_cap_cells"] =
      numerics.hydro.plasma_viscosity.mfp_cap_cells;
  plasma_viscosity["lnlambda_fixed"] =
      numerics.hydro.plasma_viscosity.lnlambda_fixed;
  plasma_viscosity["dt_safety"] = numerics.hydro.plasma_viscosity.dt_safety;

  py::dict hydro;
  hydro["enabled"] = numerics.hydro.enabled;
  hydro["compatible_energy"] = numerics.hydro.compatible_energy;
  hydro["rho_e_linear_grid"] = numerics.hydro.rho_e_linear_grid;
  hydro["eos_writeback"] = numerics.hydro.eos_writeback;
  hydro["eos_closure_mode"] = numerics.hydro.eos_closure_mode;
  hydro["qei_evaluate_at_t_n"] = numerics.hydro.qei_evaluate_at_t_n;
  hydro["qei_multiplier"] = numerics.hydro.qei_multiplier;
  hydro["exact_override"] = numerics.hydro.exact_override;
  hydro["total_energy_remap_2d_rz"] =
      numerics.hydro.total_energy_remap_2d_rz;
  hydro["work_split_audit_2d_rz"] =
      numerics.hydro.work_split_audit_2d_rz;
  hydro["work_split_audit_cell_every_n_steps"] =
      numerics.hydro.work_split_audit_cell_every_n_steps;
  hydro["work_split_audit_all_rows"] =
      numerics.hydro.work_split_audit_all_rows;
  hydro["hllc_z_flux_2d_rz"] = numerics.hydro.hllc_z_flux_2d_rz;
  hydro["hllc_z_flux_audit_2d_rz"] =
      numerics.hydro.hllc_z_flux_audit_2d_rz;
  hydro["bbs_axis_policy_enabled"] =
      numerics.hydro.bbs_axis_policy_enabled;
  hydro["subzonal_mass_enabled"] =
      numerics.hydro.subzonal_mass_enabled;
  hydro["subzonal_mass_lagrangian_invariant_enabled"] =
      numerics.hydro.subzonal_mass_lagrangian_invariant_enabled;
  hydro["anti_hourglass_kappa"] =
      numerics.hydro.anti_hourglass_kappa;
  hydro["subzonal_pressure_enabled"] =
      numerics.hydro.subzonal_pressure_enabled;
  hydro["subzonal_dt_limiter_enabled"] =
      numerics.hydro.subzonal_dt_limiter_enabled;
  hydro["subzonal_pressure_mode"] =
      numerics.hydro.subzonal_pressure_mode;
  hydro["subzonal_band_mode"] =
      numerics.hydro.subzonal_band_mode;
  hydro["subzonal_band_feather_layers"] =
      numerics.hydro.subzonal_band_feather_layers;
  hydro["subzonal_merit_mode"] =
      numerics.hydro.subzonal_merit_mode;
  hydro["subzonal_alpha1"] = numerics.hydro.subzonal_alpha1;
  hydro["subzonal_alpha2"] = numerics.hydro.subzonal_alpha2;
  hydro["subzonal_merit_power"] =
      numerics.hydro.subzonal_merit_power;
  hydro["subzonal_merit_constant"] =
      numerics.hydro.subzonal_merit_constant;
  hydro["hllc_z_flux_hlle_fallback"] =
      numerics.hydro.hllc_z_flux_hlle_fallback;
  hydro["hllc_z_flux_strict_quasi_1d"] =
      numerics.hydro.hllc_z_flux_strict_quasi_1d;
  hydro["axis_motion_floor_fraction"] = numerics.hydro.axis_motion_floor_fraction;
  hydro["axis_margin_dt_floor_fraction"] =
      numerics.hydro.axis_margin_dt_floor_fraction;
  hydro["volume_rate_cfl_enabled"] = numerics.hydro.volume_rate_cfl_enabled;
  hydro["volume_rate_cfl_threshold"] = numerics.hydro.volume_rate_cfl_threshold;
  hydro["tri_fan_center_cfl_enabled"] =
      numerics.hydro.tri_fan_center_cfl_enabled;
  hydro["tri_fan_center_cfl_safety"] = numerics.hydro.tri_fan_center_cfl_safety;
  hydro["tri_fan_center_cfl_band_radial_index"] =
      numerics.hydro.tri_fan_center_cfl_band_radial_index;
  hydro["corner_j_predict_cfl_enabled"] =
      numerics.hydro.corner_j_predict_cfl_enabled;
  hydro["corner_j_predict_cfl_safety"] =
      numerics.hydro.corner_j_predict_cfl_safety;
  hydro["corner_j_predict_floor_frac"] =
      numerics.hydro.corner_j_predict_floor_frac;
  hydro["corner_j_predict_max_shrink"] =
      numerics.hydro.corner_j_predict_max_shrink;
  hydro["corner_j_predict_shell_rings"] =
      numerics.hydro.corner_j_predict_shell_rings;
  hydro["tri_fan_center_perturbation_diag_enabled"] =
      numerics.hydro.tri_fan_center_perturbation_diag_enabled;
  if (numerics.hydro.av_qcap_scope != AvQcapScope::GLOBAL) {
    hydro["av_qcap_scope"] =
        av_qcap_scope_to_string(numerics.hydro.av_qcap_scope);
  }
  if (numerics.hydro.center_cfl_scope != CenterCflScope::DISABLED) {
    hydro["center_cfl_scope"] =
        center_cfl_scope_to_string(numerics.hydro.center_cfl_scope);
  }
  if (numerics.hydro.center_perturbation_diag_scope !=
          CenterPerturbationDiagScope::DISABLED ||
      numerics.hydro.center_perturbation_diag_radial_bins != 2) {
    hydro["center_perturbation_diag_scope"] =
        center_perturbation_diag_scope_to_string(
            numerics.hydro.center_perturbation_diag_scope);
    hydro["center_perturbation_diag_radial_bins"] =
        numerics.hydro.center_perturbation_diag_radial_bins;
  }
  hydro["rz_geometric_cfl_enabled"] =
      numerics.hydro.rz_geometric_cfl_enabled;
  hydro["rz_geometric_cfl_etaV"] = numerics.hydro.rz_geometric_cfl_etaV;
  hydro["rz_geometric_cfl_r_floor"] =
      numerics.hydro.rz_geometric_cfl_r_floor;
  hydro["rz_geometric_cfl_cumulative_protection_enabled"] =
      numerics.hydro.rz_geometric_cfl_cumulative_protection_enabled;
  hydro["rz_geometric_cfl_v_initial_floor"] =
      numerics.hydro.rz_geometric_cfl_v_initial_floor;
  hydro["rz_geometric_cfl_precise_u_half_enabled"] =
      numerics.hydro.rz_geometric_cfl_precise_u_half_enabled;
  hydro["trial_volume_cfl_enabled"] = numerics.hydro.trial_volume_cfl_enabled;
  hydro["trial_volume_cfl_floor_fraction"] =
      numerics.hydro.trial_volume_cfl_floor_fraction;
  hydro["trial_volume_cfl_shrink_fraction"] =
      numerics.hydro.trial_volume_cfl_shrink_fraction;
  hydro["corner_jacobian_ale_trigger_enabled"] =
      numerics.hydro.corner_jacobian_ale_trigger_enabled;
  hydro["corner_jacobian_floor_eps"] = numerics.hydro.corner_jacobian_floor_eps;
  hydro["corner_jacobian_ale_trigger_scale"] =
      numerics.hydro.corner_jacobian_ale_trigger_scale;
  hydro["in_hydro_corner_j_guard_enabled"] =
      numerics.hydro.in_hydro_corner_j_guard_enabled;
  hydro["in_hydro_gauss_j_guard_enabled"] =
      numerics.hydro.in_hydro_gauss_j_guard_enabled;
  hydro["in_hydro_rz_volume_guard_enabled"] =
      numerics.hydro.in_hydro_rz_volume_guard_enabled;
  hydro["in_hydro_gauss_j_floor_rel"] =
      numerics.hydro.in_hydro_gauss_j_floor_rel;
  hydro["in_hydro_rz_volume_floor_rel"] =
      numerics.hydro.in_hydro_rz_volume_floor_rel;
  hydro["mesh_quality_dt_cfl_enabled"] =
      numerics.hydro.mesh_quality_dt_cfl_enabled;
  hydro["mesh_quality_dt_safety_alpha"] =
      numerics.hydro.mesh_quality_dt_safety_alpha;
  hydro["mesh_quality_dt_corner_j_enabled"] =
      numerics.hydro.mesh_quality_dt_corner_j_enabled;
  hydro["mesh_quality_dt_gauss_j_enabled"] =
      numerics.hydro.mesh_quality_dt_gauss_j_enabled;
  hydro["mesh_quality_dt_rz_volume_enabled"] =
      numerics.hydro.mesh_quality_dt_rz_volume_enabled;
  hydro["mesh_quality_dt_axis_margin_additive"] =
      numerics.hydro.mesh_quality_dt_axis_margin_additive;
  hydro["mesh_quality_dt_corner_j_floor_rel"] =
      numerics.hydro.mesh_quality_dt_corner_j_floor_rel;
  hydro["mesh_quality_dt_gauss_j_floor_rel"] =
      numerics.hydro.mesh_quality_dt_gauss_j_floor_rel;
  hydro["mesh_quality_dt_rz_volume_floor_rel"] =
      numerics.hydro.mesh_quality_dt_rz_volume_floor_rel;
  hydro["ring7_quotient_enabled"] =
      numerics.hydro.ring7_quotient_enabled;
  hydro["regime_aware_corner_j_guard_enabled"] =
      numerics.hydro.regime_aware_corner_j_guard_enabled;
  hydro["axis_margin_guard_enabled"] =
      numerics.hydro.axis_margin_guard_enabled;
  hydro["axis_margin_additive_in_action8_enabled"] =
      numerics.hydro.axis_margin_additive_in_action8_enabled;
  hydro["axis_guard_band_cells"] = numerics.hydro.axis_guard_band_cells;
  hydro["driver_full_step_retry_enabled"] =
      numerics.hydro.driver_full_step_retry_enabled;
  hydro["driver_full_step_retry_max_attempts"] =
      numerics.hydro.driver_full_step_retry_max_attempts;
  hydro["dispatcher_state_sensitive_bypass_enabled"] =
      numerics.hydro.dispatcher_state_sensitive_bypass_enabled;
  hydro["dispatcher_state_sensitive_repair_cap_per_step"] =
      numerics.hydro.dispatcher_state_sensitive_repair_cap_per_step;
  hydro["strategy_first_retry_enabled"] =
      numerics.hydro.strategy_first_retry_enabled;
  hydro["strategy_first_max_same_dt_attempts"] =
      numerics.hydro.strategy_first_max_same_dt_attempts;
  hydro["driver_retry_active_mesh_repair_enabled"] =
      numerics.hydro.driver_retry_active_mesh_repair_enabled;
  hydro["driver_retry_corner_balance_threshold"] =
      numerics.hydro.driver_retry_corner_balance_threshold;
  hydro["cascade_on_hydro_retry_enabled"] =
      numerics.hydro.cascade_on_hydro_retry_enabled;
  hydro["driver_retry_use_suggested_dt_enabled"] =
      numerics.hydro.driver_retry_use_suggested_dt_enabled;
  py::dict geometric_retry_stagnation;
  geometric_retry_stagnation["enabled"] =
      numerics.hydro.geometric_retry_stagnation.enabled;
  geometric_retry_stagnation["same_cell_count_threshold"] =
      numerics.hydro.geometric_retry_stagnation.same_cell_count_threshold;
  geometric_retry_stagnation["sigma_rel_tol"] =
      numerics.hydro.geometric_retry_stagnation.sigma_rel_tol;
  geometric_retry_stagnation["dt_drop_factor"] =
      numerics.hydro.geometric_retry_stagnation.dt_drop_factor;
  geometric_retry_stagnation["force_diagnostic_dump"] =
      numerics.hydro.geometric_retry_stagnation.force_diagnostic_dump;
  hydro["geometric_retry_stagnation"] = geometric_retry_stagnation;
  hydro["mesh_geometry_soft_fail_enabled"] =
      numerics.hydro.mesh_geometry_soft_fail_enabled;
  hydro["av_type"] = numerics.hydro.av_type;
  hydro["av_model"] = av_model_to_string(numerics.hydro.av_model);
  hydro["corner_mass_convention"] =
      corner_mass_convention_to_string(
          numerics.hydro.corner_mass_convention);
  hydro["time_integration"] =
      hydro_time_integration_to_string(numerics.hydro.time_integration);
  hydro["total_energy_identity_check"] =
      numerics.hydro.total_energy_identity_check;
  hydro["rz_momentum_scheme"] = numerics.hydro.rz_momentum_scheme;
  hydro["axis_node_mass_convention"] =
      numerics.hydro.axis_node_mass_convention;
  hydro["boundary_1d"] = numerics.hydro.boundary_1d;
  hydro["boundary_2d"] = hydro_boundary_2d;
  hydro["av_linear"] = numerics.hydro.av_linear;
  hydro["av_quadratic"] = numerics.hydro.av_quadratic;
  hydro["av_qcap_over_p"] = numerics.hydro.av_qcap_over_p;
  hydro["av_qcap_center_band_only"] = numerics.hydro.av_qcap_center_band_only;
  hydro["av_cfl_coefficient"] = numerics.hydro.av_cfl_coefficient;
  hydro["csw_C1"] = numerics.hydro.csw_C1;
  hydro["csw_C2"] = numerics.hydro.csw_C2;
  hydro["csw_limiter"] = numerics.hydro.csw_limiter;
  hydro["csw_limiter_enabled"] = numerics.hydro.csw_limiter_enabled;
  hydro["csw_shock_limiter_floor"] = numerics.hydro.csw_shock_limiter_floor;
  hydro["csw_zero_uniform_compression"] =
      numerics.hydro.csw_zero_uniform_compression;
  hydro["csw_diagnostics"] = numerics.hydro.csw_diagnostics;
  hydro["av_limiter_J"] = numerics.hydro.av_limiter_J;
  hydro["av_heat_C"] = numerics.hydro.av_heat_C;
  hydro["post_shock_heat"] = numerics.hydro.post_shock_heat;
  hydro["post_shock_heat_C"] = numerics.hydro.post_shock_heat_C;
  hydro["post_shock_heat_decay"] = numerics.hydro.post_shock_heat_decay;
  hydro["post_shock_velocity_damping_C"] =
      numerics.hydro.post_shock_velocity_damping_C;
  hydro["bulk_viscosity_C"] = numerics.hydro.bulk_viscosity_C;
  hydro["ion_art_heat_C"] = numerics.hydro.ion_art_heat_C;
  hydro["crossing_dt_safety"] = numerics.hydro.crossing_dt_safety;
  hydro["time_integrator"] = numerics.hydro.time_integrator;
  hydro["hourglass"] = hourglass;
  hydro["adaptive_av"] = adaptive_av;
  hydro["plasma_viscosity"] = plasma_viscosity;
  hydro["av_eos_aware"] = numerics.hydro.av_eos_aware;
  hydro["av_eos_gamma1_ref"] = numerics.hydro.av_eos_gamma1_ref;
  hydro["av_eos_boost_max"] = numerics.hydro.av_eos_boost_max;
  hydro["odd_even_damping_C"] = numerics.hydro.odd_even_damping_C;
  hydro["ee_odd_even_C"] = numerics.hydro.ee_odd_even_C;
  hydro["hk_velocity_damper_C"] = numerics.hydro.hk_velocity_damper_C;
  hydro["hk_velocity_damper_tau_min"] = numerics.hydro.hk_velocity_damper_tau_min;
  hydro["hk_velocity_damper_grad_Te_max"] =
      numerics.hydro.hk_velocity_damper_grad_Te_max;
  hydro["hk_velocity_damper_grad_rho_max"] =
      numerics.hydro.hk_velocity_damper_grad_rho_max;
  hydro["hk_velocity_damper_guard_cells"] =
      numerics.hydro.hk_velocity_damper_guard_cells;
  hydro["av_heat_to"] = numerics.hydro.av_heat_to;
  hydro["boundary_pressure"] = serialize_callable(numerics.hydro.pressure_drive_1d);

  py::dict conduction;
  conduction["enabled"] = numerics.conduction.enabled;
  conduction["solver"] = numerics.conduction.solver;
  conduction["sts_floor_limiter"] = numerics.conduction.sts_floor_limiter;
  conduction["ion_conduction"] = numerics.conduction.ion_conduction;
  conduction["f_lim"] = numerics.conduction.f_lim;
  conduction["mfp_limiter_C"] = numerics.conduction.mfp_limiter_C;
  conduction["sts_damping"] = numerics.conduction.sts_damping;
  conduction["sts_max_stages"] = numerics.conduction.sts_max_stages;
  conduction["sts_subcycle_eta"] = numerics.conduction.sts_subcycle_eta;
  conduction["halo_strategy"] = numerics.conduction.halo_strategy;
  conduction["hypre_rtol"] = numerics.conduction.hypre_rtol;
  conduction["hypre_max_iter"] = numerics.conduction.hypre_max_iter;
  conduction["test_kappa"] = numerics.conduction.test_kappa;
  conduction["test_planar"] = numerics.conduction.test_planar;
  conduction["face_kappa_policy"] = numerics.conduction.face_kappa_policy;
  conduction["nonlocal_model"] = numerics.conduction.nonlocal_model;
  conduction["snb_n_groups"] = numerics.conduction.snb_n_groups;
  conduction["snb_E_max_over_Te"] = numerics.conduction.snb_E_max_over_Te;
  conduction["snb_mfp"] = numerics.conduction.snb_mfp;
  conduction["snb_efield"] = numerics.conduction.snb_efield;
  conduction["snb_picard_max_iters"] = numerics.conduction.snb_picard_max_iters;
  conduction["snb_picard_rtol"] = numerics.conduction.snb_picard_rtol;

  py::dict ale;
  const Config::NumericsConfig::AleConfig ale_defaults;
  ale["enabled"] = numerics.ale.enabled;
  ale["every_n_steps"] = numerics.ale.every_n_steps;
  ale["warmup_steps"] = numerics.ale.warmup_steps;
  ale["relaxation"] = numerics.ale.relaxation;
  ale["spacing_ratio_threshold"] = numerics.ale.spacing_ratio_threshold;
  ale["quality_threshold"] = numerics.ale.quality_threshold;
  ale["force_rezone_every_n_steps"] = numerics.ale.force_rezone_every_n_steps;
  ale["max_iterations"] = numerics.ale.max_iterations;
  ale["convergence_tol"] = numerics.ale.convergence_tol;
  ale["max_displacement_fraction"] = numerics.ale.max_displacement_fraction;
  ale["remap_limiter"] = numerics.ale.remap_limiter;
  ale["swept_volume_sign_fixed"] = numerics.ale.swept_volume_sign_fixed;
  ale["remap_ms_midpoint"] = numerics.ale.remap_ms_midpoint;
  ale["remap_ms_post_check"] = numerics.ale.remap_ms_post_check;
  ale["remap_ms_post_max_iter"] = numerics.ale.remap_ms_post_max_iter;
  ale["remap_ms_rescale_floor"] = numerics.ale.remap_ms_rescale_floor;
  ale["ke_fixup"] = numerics.ale.ke_fixup;
  ale["ke_conservation_closure"] = numerics.ale.ke_conservation_closure;
  ale["ke_conservation_closure_audit"] =
      numerics.ale.ke_conservation_closure_audit;
  ale["ke_closure_redistribute_floor"] =
      numerics.ale.ke_closure_redistribute_floor;
  ale["debug_per_remap_log"] = numerics.ale.debug_per_remap_log;
  ale["shock_sensor_guard_cells"] = numerics.ale.shock_sensor_guard_cells;
  ale["density_jump_threshold"] = numerics.ale.density_jump_threshold;
  ale["Te_jump_threshold"] = numerics.ale.Te_jump_threshold;
  ale["preventive_axis_guard_fraction"] = numerics.ale.preventive_axis_guard_fraction;
  ale["axis_z_motion"] = numerics.ale.axis_z_motion;
  ale["winslow_axis_kappa"] = numerics.ale.winslow_axis_kappa;
  py::dict button_morph;
  button_morph["enabled"] = numerics.ale.button_morph.enabled;
  button_morph["t_start_s"] = numerics.ale.button_morph.t_start_s;
  button_morph["t_end_s"] = numerics.ale.button_morph.t_end_s;
  button_morph["max_step_fraction"] =
      numerics.ale.button_morph.max_step_fraction;
  button_morph["every_n_steps"] =
      numerics.ale.button_morph.every_n_steps;
  ale["button_morph"] = button_morph;
  ale["reference_barrier_enabled"] = numerics.ale.reference_barrier_enabled;
  ale["reference_target"] = numerics.ale.reference_target;
  ale["reference_blend_default"] = numerics.ale.reference_blend_default;
  ale["reference_volume_floor_rel"] = numerics.ale.reference_volume_floor_rel;
  ale["reference_corner_j_floor_rel"] = numerics.ale.reference_corner_j_floor_rel;
  ale["reference_gauss_j_floor_rel"] = numerics.ale.reference_gauss_j_floor_rel;
  ale["reference_linesearch_max_iters"] =
      numerics.ale.reference_linesearch_max_iters;
  ale["reference_force_engage_every_step"] =
      numerics.ale.reference_force_engage_every_step;
  ale["reference_trigger_axis_margin_enabled"] =
      numerics.ale.reference_trigger_axis_margin_enabled;
  ale["reference_trigger_axis_margin_threshold"] =
      numerics.ale.reference_trigger_axis_margin_threshold;
  ale["reference_trigger_corner_j_ratio_enabled"] =
      numerics.ale.reference_trigger_corner_j_ratio_enabled;
  ale["reference_trigger_corner_j_ratio_threshold"] =
      numerics.ale.reference_trigger_corner_j_ratio_threshold;
  if (numerics.ale.dgcl_commit_gate ||
      numerics.ale.dgcl_commit_rtol != ale_defaults.dgcl_commit_rtol) {
    ale["dgcl_commit_gate"] = numerics.ale.dgcl_commit_gate;
    ale["dgcl_commit_rtol"] = numerics.ale.dgcl_commit_rtol;
  }
  if (numerics.ale.transaction_failure_inject_point !=
      ale_defaults.transaction_failure_inject_point) {
    ale["transaction_failure_inject_point"] =
        numerics.ale.transaction_failure_inject_point;
  }
  ale["driver_retry_reference_barrier_enabled"] =
      numerics.ale.driver_retry_reference_barrier_enabled;
  ale["driver_retry_reference_barrier_K_axis"] =
      numerics.ale.driver_retry_reference_barrier_K_axis;
  ale["driver_retry_reference_barrier_eta_axis"] =
      numerics.ale.driver_retry_reference_barrier_eta_axis;
  ale["driver_retry_reference_barrier_max_attempts"] =
      numerics.ale.driver_retry_reference_barrier_max_attempts;
  ale["driver_retry_reference_barrier_same_sig_max"] =
      numerics.ale.driver_retry_reference_barrier_same_sig_max;
  ale["driver_retry_reference_barrier_cell_window"] =
      numerics.ale.driver_retry_reference_barrier_cell_window;
  ale["driver_retry_reference_barrier_dt_collapse_rel"] =
      numerics.ale.driver_retry_reference_barrier_dt_collapse_rel;
  ale["driver_retry_reference_barrier_lambda_collapse_threshold"] =
      numerics.ale.driver_retry_reference_barrier_lambda_collapse_threshold;
  ale["driver_retry_reference_barrier_lambda_collapse_count"] =
      numerics.ale.driver_retry_reference_barrier_lambda_collapse_count;
  ale["driver_retry_reference_barrier_quality_progress_factor"] =
      numerics.ale.driver_retry_reference_barrier_quality_progress_factor;
  ale["driver_retry_reference_barrier_quality_progress_count"] =
      numerics.ale.driver_retry_reference_barrier_quality_progress_count;
  ale["driver_retry_reference_barrier_rezone_freq_warn_fraction"] =
      numerics.ale.driver_retry_reference_barrier_rezone_freq_warn_fraction;
  ale["driver_retry_reference_barrier_rezone_freq_window"] =
      numerics.ale.driver_retry_reference_barrier_rezone_freq_window;
  ale["driver_retry_reference_barrier_chi"] =
      numerics.ale.driver_retry_reference_barrier_chi;
  ale["driver_retry_reference_barrier_q_retry"] =
      numerics.ale.driver_retry_reference_barrier_q_retry;
  ale["remap_damage_gate_enabled"] = numerics.ale.remap_damage_gate_enabled;
  ale["remap_damage_dmax"] = numerics.ale.remap_damage_dmax;
  ale["remap_damage_axis_eta"] = numerics.ale.remap_damage_axis_eta;
  ale["remap_damage_axis_budget_enabled"] =
      numerics.ale.remap_damage_axis_budget_enabled;
  ale["remap_damage_axis_budget_factor"] =
      numerics.ale.remap_damage_axis_budget_factor;
  ale["predictive_acceptance_enabled"] =
      numerics.ale.predictive_acceptance_enabled;
  ale["predictive_acceptance_axis_floor_fraction"] =
      numerics.ale.predictive_acceptance_axis_floor_fraction;
  ale["predictive_acceptance_cell_vol_floor_fraction"] =
      numerics.ale.predictive_acceptance_cell_vol_floor_fraction;
  ale["safe_backtrack_enabled"] =
      numerics.ale.safe_backtrack_enabled;
  ale["safe_backtrack_min_exp"] =
      numerics.ale.safe_backtrack_min_exp;
  ale["safe_backtrack_binary_iters"] =
      numerics.ale.safe_backtrack_binary_iters;
  ale["corner_cell_aspect_protection_enabled"] =
      numerics.ale.corner_cell_aspect_protection_enabled;
  ale["corner_cell_aspect_eta"] =
      numerics.ale.corner_cell_aspect_eta;
  ale["rezone_solver"] = numerics.ale.rezone_solver;
  if (numerics.ale.rezone_solver == "m1_tmop") {
    ale["m1_gamma_align"] = numerics.ale.m1_gamma_align;
    ale["m1_lambda_tether"] = numerics.ale.m1_lambda_tether;
    ale["m1_theta_reg"] = numerics.ale.m1_theta_reg;
    ale["m1_sweeps"] = numerics.ale.m1_sweeps;
    ale["m1_min_j_dec_rel"] = numerics.ale.m1_min_j_dec_rel;
    ale["m1_barrier_beta"] = numerics.ale.m1_barrier_beta;
  }
  if (numerics.ale.euler_window.enabled) {
    py::dict euler_window;
    euler_window["enabled"] = numerics.ale.euler_window.enabled;
    euler_window["shape"] = numerics.ale.euler_window.shape;
    euler_window["r0"] = numerics.ale.euler_window.r0;
    euler_window["r1"] = numerics.ale.euler_window.r1;
    euler_window["z0"] = numerics.ale.euler_window.z0;
    euler_window["z1"] = numerics.ale.euler_window.z1;
    euler_window["cr"] = numerics.ale.euler_window.cr;
    euler_window["cz"] = numerics.ale.euler_window.cz;
    euler_window["rad_in"] = numerics.ale.euler_window.rad_in;
    euler_window["rad_out"] = numerics.ale.euler_window.rad_out;
    euler_window["transition_width"] =
        numerics.ale.euler_window.transition_width;
    ale["euler_window"] = euler_window;
  }
  ale["rezone_local_admissibility_linesearch"] =
      numerics.ale.rezone_local_admissibility_linesearch;
  ale["rezone_local_j_floor_rel"] =
      numerics.ale.rezone_local_j_floor_rel;
  ale["rezone_local_linesearch_max_halves"] =
      numerics.ale.rezone_local_linesearch_max_halves;
  ale["reject_zero_gauss_j"] =
      numerics.ale.reject_zero_gauss_j;
  ale["zero_gauss_j_floor_rel"] =
      numerics.ale.zero_gauss_j_floor_rel;
  ale["lambda_sweep_diagnostic_enabled"] =
      numerics.ale.lambda_sweep_diagnostic_enabled;
  ale["lambda_sweep_target_cell_c"] =
      numerics.ale.lambda_sweep_target_cell_c;
  ale["lambda_sweep_target_cell_i"] =
      numerics.ale.lambda_sweep_target_cell_i;
  ale["lambda_sweep_target_cell_j"] =
      numerics.ale.lambda_sweep_target_cell_j;
  ale["lambda_sweep_max_exp"] =
      numerics.ale.lambda_sweep_max_exp;
  ale["corner_jacobian_post_tangle_enabled"] =
      numerics.ale.corner_jacobian_post_tangle_enabled;
  ale["corner_post_tangle_strict_floor_enabled"] =
      numerics.ale.corner_post_tangle_strict_floor_enabled;
  ale["local_boundary_repair_enabled"] =
      numerics.ale.local_boundary_repair_enabled;
  ale["multi_node_boundary_repair_enabled"] =
      numerics.ale.multi_node_boundary_repair_enabled;
  ale["multi_node_interior_repair_enabled"] =
      numerics.ale.multi_node_interior_repair_enabled;
  ale["axis_variational_projection_enabled"] =
      numerics.ale.axis_variational_projection_enabled;
  ale["emergency_cell_deactivation_enabled"] =
      numerics.ale.emergency_cell_deactivation_enabled;
  ale["multiblock_cross_seam_rezone_enabled"] =
      numerics.ale.multiblock_cross_seam_rezone_enabled;
  ale["multiblock_scaled_reference_enabled"] =
      numerics.ale.multiblock_scaled_reference_enabled;
  if (numerics.ale.multiblock_differential_reference_enabled ||
      numerics.ale.multiblock_differential_reference_band_count !=
          ale_defaults.multiblock_differential_reference_band_count ||
      numerics.ale.multiblock_differential_reference_smoothing_g0 !=
          ale_defaults.multiblock_differential_reference_smoothing_g0 ||
      numerics.ale.multiblock_differential_reference_nu !=
          ale_defaults.multiblock_differential_reference_nu ||
      numerics.ale.multiblock_differential_reference_eps_v !=
          ale_defaults.multiblock_differential_reference_eps_v ||
      numerics.ale.multiblock_differential_reference_s_cap_min_rel !=
          ale_defaults.multiblock_differential_reference_s_cap_min_rel ||
      numerics.ale.multiblock_differential_reference_xi_seam_tol !=
          ale_defaults.multiblock_differential_reference_xi_seam_tol ||
      numerics.ale.multiblock_differential_reference_sigma_warn_floor !=
          ale_defaults.multiblock_differential_reference_sigma_warn_floor) {
    ale["multiblock_differential_reference_enabled"] =
        numerics.ale.multiblock_differential_reference_enabled;
    ale["multiblock_differential_reference_band_count"] =
        numerics.ale.multiblock_differential_reference_band_count;
    ale["multiblock_differential_reference_smoothing_g0"] =
        numerics.ale.multiblock_differential_reference_smoothing_g0;
    ale["multiblock_differential_reference_nu"] =
        numerics.ale.multiblock_differential_reference_nu;
    ale["multiblock_differential_reference_eps_v"] =
        numerics.ale.multiblock_differential_reference_eps_v;
    ale["multiblock_differential_reference_s_cap_min_rel"] =
        numerics.ale.multiblock_differential_reference_s_cap_min_rel;
    ale["multiblock_differential_reference_xi_seam_tol"] =
        numerics.ale.multiblock_differential_reference_xi_seam_tol;
    ale["multiblock_differential_reference_sigma_warn_floor"] =
        numerics.ale.multiblock_differential_reference_sigma_warn_floor;
  }
  if (numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled ||
      numerics.ale.multiblock_center_patch_ring_max !=
          ale_defaults.multiblock_center_patch_ring_max ||
      numerics.ale.multiblock_center_patch_xi_center !=
          ale_defaults.multiblock_center_patch_xi_center ||
      numerics.ale.multiblock_center_patch_halo_layers !=
          ale_defaults.multiblock_center_patch_halo_layers ||
      numerics.ale.multiblock_center_patch_vol_on !=
          ale_defaults.multiblock_center_patch_vol_on ||
      numerics.ale.multiblock_center_patch_vol_off !=
          ale_defaults.multiblock_center_patch_vol_off ||
      numerics.ale.multiblock_center_patch_cornerj_on !=
          ale_defaults.multiblock_center_patch_cornerj_on ||
      numerics.ale.multiblock_center_patch_cornerj_off !=
          ale_defaults.multiblock_center_patch_cornerj_off ||
      numerics.ale.multiblock_center_patch_gaussj_on !=
          ale_defaults.multiblock_center_patch_gaussj_on ||
      numerics.ale.multiblock_center_patch_gaussj_off !=
          ale_defaults.multiblock_center_patch_gaussj_off) {
    ale["multiblock_lagrangian_bulk_center_patch_reference_enabled"] =
        numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled;
    ale["multiblock_center_patch_ring_max"] =
        numerics.ale.multiblock_center_patch_ring_max;
    ale["multiblock_center_patch_xi_center"] =
        numerics.ale.multiblock_center_patch_xi_center;
    ale["multiblock_center_patch_halo_layers"] =
        numerics.ale.multiblock_center_patch_halo_layers;
    ale["multiblock_center_patch_vol_on"] =
        numerics.ale.multiblock_center_patch_vol_on;
    ale["multiblock_center_patch_vol_off"] =
        numerics.ale.multiblock_center_patch_vol_off;
    ale["multiblock_center_patch_cornerj_on"] =
        numerics.ale.multiblock_center_patch_cornerj_on;
    ale["multiblock_center_patch_cornerj_off"] =
        numerics.ale.multiblock_center_patch_cornerj_off;
    ale["multiblock_center_patch_gaussj_on"] =
        numerics.ale.multiblock_center_patch_gaussj_on;
    ale["multiblock_center_patch_gaussj_off"] =
        numerics.ale.multiblock_center_patch_gaussj_off;
  }
  if (numerics.ale.ale_reference_diagnostics_enabled) {
    ale["ale_reference_diagnostics_enabled"] =
        numerics.ale.ale_reference_diagnostics_enabled;
  }
  ale["multiblock_path_admissibility_enabled"] =
      numerics.ale.multiblock_path_admissibility_enabled;
  ale["path_admissibility_floor"] =
      numerics.ale.path_admissibility_floor;
  ale["dt_rejection_factor"] =
      numerics.ale.dt_rejection_factor;
  ale["max_dt_rejections"] =
      numerics.ale.max_dt_rejections;
  ale["axis_band_managed_remap_enabled"] =
      numerics.ale.axis_band_managed_remap_enabled;
  ale["axis_band_managed_remap_width"] =
      numerics.ale.axis_band_managed_remap_width;
  ale["axis_band_managed_remap_max_width"] =
      numerics.ale.axis_band_managed_remap_max_width;
  ale["axis_band_managed_remap_every_hydro_half_step"] =
      numerics.ale.axis_band_managed_remap_every_hydro_half_step;
  ale["axis_band_managed_remap_margin_trigger"] =
      numerics.ale.axis_band_managed_remap_margin_trigger;
  ale["axis_band_managed_remap_equal_volume"] =
      numerics.ale.axis_band_managed_remap_equal_volume;
  ale["axis_band_managed_remap_include_radiation_groups"] =
      numerics.ale.axis_band_managed_remap_include_radiation_groups;
  ale["axis_rezone_enabled"] =
      numerics.ale.axis_rezone_enabled;
  ale["axis_rezone_trigger_edge_fraction"] =
      numerics.ale.axis_rezone_trigger_edge_fraction;
  ale["axis_rezone_trigger_min_altitude_fraction"] =
      numerics.ale.axis_rezone_trigger_min_altitude_fraction;
  ale["axis_rezone_eta_floor"] =
      numerics.ale.axis_rezone_eta_floor;
  if (numerics.ale.ale_identity_mode) {
    ale["ale_identity_mode"] = numerics.ale.ale_identity_mode;
  }
  if (numerics.ale.ale_mover_diag) {
    ale["ale_mover_diag"] = numerics.ale.ale_mover_diag;
  }
  if (numerics.ale.ale_preserve_lagrangian_velocity_carry) {
    ale["ale_preserve_lagrangian_velocity_carry"] =
        numerics.ale.ale_preserve_lagrangian_velocity_carry;
  }
  const Config::NumericsConfig::AleConfig::AlignDiagnosticsConfig
      align_diagnostics_defaults;
  const auto& align_diagnostics = numerics.ale.align_diagnostics;
  if (align_diagnostics.enabled ||
      align_diagnostics.every_n_steps !=
          align_diagnostics_defaults.every_n_steps ||
      align_diagnostics.c_q_threshold !=
          align_diagnostics_defaults.c_q_threshold ||
      align_diagnostics.w_rho != align_diagnostics_defaults.w_rho ||
      align_diagnostics.w_p != align_diagnostics_defaults.w_p ||
      align_diagnostics.floor_rel != align_diagnostics_defaults.floor_rel) {
    py::dict align;
    align["enabled"] = align_diagnostics.enabled;
    align["every_n_steps"] = align_diagnostics.every_n_steps;
    align["c_q_threshold"] = align_diagnostics.c_q_threshold;
    align["w_rho"] = align_diagnostics.w_rho;
    align["w_p"] = align_diagnostics.w_p;
    align["floor_rel"] = align_diagnostics.floor_rel;
    ale["align_diagnostics"] = align;
  }
  if (numerics.ale.core_freeze_enabled) {
    ale["core_freeze_enabled"] = numerics.ale.core_freeze_enabled;
    ale["core_freeze_source"] = numerics.ale.core_freeze_source;
    ale["core_freeze_tracer_cut"] = numerics.ale.core_freeze_tracer_cut;
    ale["core_freeze_halo_layers"] = numerics.ale.core_freeze_halo_layers;
    ale["core_freeze_apply_to_axis_rezone"] =
        numerics.ale.core_freeze_apply_to_axis_rezone;
    ale["core_freeze_skip_velocity_projection"] =
        numerics.ale.core_freeze_skip_velocity_projection;
  }
  ale["axis_repair_mode"] = numerics.ale.axis_repair_mode;
  ale["remap_scheme"] = numerics.ale.remap_scheme;
  ale["remap_ms2_limiter"] = numerics.ale.remap_ms2_limiter;
  ale["conservative_remap_enabled"] =
      numerics.ale.conservative_remap_enabled;
  ale["conservative_remap_target"] =
      numerics.ale.conservative_remap_target;
  ale["conservative_remap_radiation_enabled"] =
      numerics.ale.conservative_remap_radiation_enabled;
  ale["conservative_remap_order"] =
      numerics.ale.conservative_remap_order;
  ale["tri_fan_tracking_reference_enabled"] =
      numerics.ale.tri_fan_tracking_reference_enabled;
  ale["tri_fan_tracking_reference_mode"] =
      numerics.ale.tri_fan_tracking_reference_mode;
  ale["tri_fan_tracking_reference_omega"] =
      numerics.ale.tri_fan_tracking_reference_omega;
  ale["tri_fan_tracking_reference_beta"] =
      numerics.ale.tri_fan_tracking_reference_beta;
  ale["tri_fan_tracking_reference_g0"] =
      numerics.ale.tri_fan_tracking_reference_g0;
  ale["tri_fan_tracking_reference_nu"] =
      numerics.ale.tri_fan_tracking_reference_nu;
  ale["tri_fan_tracking_reference_eps_v"] =
      numerics.ale.tri_fan_tracking_reference_eps_v;
  ale["conservative_remap_lagrangian_bulk_enabled"] =
      numerics.ale.conservative_remap_lagrangian_bulk_enabled;
  ale["conservative_remap_lagrangian_bulk_center_node_ring_max"] =
      numerics.ale.conservative_remap_lagrangian_bulk_center_node_ring_max;
  if (numerics.ale.central_pseudo_core_enabled ||
      numerics.ale.central_pseudo_core_s_c !=
          ale_defaults.central_pseudo_core_s_c) {
    ale["central_pseudo_core_enabled"] =
        numerics.ale.central_pseudo_core_enabled;
    ale["central_pseudo_core_s_c"] =
        numerics.ale.central_pseudo_core_s_c;
  }
  if (numerics.ale.central_pseudo_core_ring_absorption_enabled ||
      numerics.ale.central_pseudo_core_ring_absorption_tau !=
          ale_defaults.central_pseudo_core_ring_absorption_tau ||
      numerics.ale.central_pseudo_core_ring_absorption_max_rings !=
          ale_defaults.central_pseudo_core_ring_absorption_max_rings ||
      numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_min !=
          ale_defaults.central_pseudo_core_ring_absorption_gas_tracer_min ||
      numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_cell_min !=
          ale_defaults
              .central_pseudo_core_ring_absorption_gas_tracer_cell_min) {
    ale["central_pseudo_core_ring_absorption_enabled"] =
        numerics.ale.central_pseudo_core_ring_absorption_enabled;
    ale["central_pseudo_core_ring_absorption_tau"] =
        numerics.ale.central_pseudo_core_ring_absorption_tau;
    ale["central_pseudo_core_ring_absorption_max_rings"] =
        numerics.ale.central_pseudo_core_ring_absorption_max_rings;
    ale["central_pseudo_core_ring_absorption_gas_tracer_min"] =
        numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_min;
    ale["central_pseudo_core_ring_absorption_gas_tracer_cell_min"] =
        numerics.ale.central_pseudo_core_ring_absorption_gas_tracer_cell_min;
  }
  if (numerics.ale.conv_rezone_enabled !=
      ale_defaults.conv_rezone_enabled) {
    ale["conv_rezone_enabled"] = numerics.ale.conv_rezone_enabled;
  }
  if (numerics.ale.central_pseudo_core_core1d_enabled ||
      numerics.ale.central_pseudo_core_core1d_build_shells !=
          ale_defaults.central_pseudo_core_core1d_build_shells ||
      numerics.ale.central_pseudo_core_core1d_split_append !=
          ale_defaults.central_pseudo_core_core1d_split_append ||
      numerics.ale.central_pseudo_core_core1d_av_c1 !=
          ale_defaults.central_pseudo_core_core1d_av_c1 ||
      numerics.ale.central_pseudo_core_core1d_av_c2 !=
          ale_defaults.central_pseudo_core_core1d_av_c2 ||
      numerics.ale.central_pseudo_core_core1d_cfl !=
          ale_defaults.central_pseudo_core_core1d_cfl ||
      numerics.ale.central_pseudo_core_core1d_piston_cap !=
          ale_defaults.central_pseudo_core_core1d_piston_cap ||
      numerics.ale.central_pseudo_core_core1d_max_substeps !=
          ale_defaults.central_pseudo_core_core1d_max_substeps ||
      numerics.ale.central_pseudo_core_core1d_dist_append !=
          ale_defaults.central_pseudo_core_core1d_dist_append) {
    ale["central_pseudo_core_core1d_enabled"] =
        numerics.ale.central_pseudo_core_core1d_enabled;
    ale["central_pseudo_core_core1d_build_shells"] =
        numerics.ale.central_pseudo_core_core1d_build_shells;
    ale["central_pseudo_core_core1d_split_append"] =
        numerics.ale.central_pseudo_core_core1d_split_append;
    ale["central_pseudo_core_core1d_av_c1"] =
        numerics.ale.central_pseudo_core_core1d_av_c1;
    ale["central_pseudo_core_core1d_av_c2"] =
        numerics.ale.central_pseudo_core_core1d_av_c2;
    ale["central_pseudo_core_core1d_cfl"] =
        numerics.ale.central_pseudo_core_core1d_cfl;
    ale["central_pseudo_core_core1d_piston_cap"] =
        numerics.ale.central_pseudo_core_core1d_piston_cap;
    ale["central_pseudo_core_core1d_max_substeps"] =
        numerics.ale.central_pseudo_core_core1d_max_substeps;
    ale["central_pseudo_core_core1d_dist_append"] =
        numerics.ale.central_pseudo_core_core1d_dist_append;
  }
  if (numerics.ale.central_pseudo_core_spherical_absorb_gasfront ||
      numerics.ale.central_pseudo_core_spherical_absorb_alpha !=
          ale_defaults.central_pseudo_core_spherical_absorb_alpha ||
      numerics.ale.central_pseudo_core_spherical_absorb_pjump !=
          ale_defaults.central_pseudo_core_spherical_absorb_pjump ||
      numerics.ale.central_pseudo_core_mixed_absorb_enabled ||
      numerics.ale.central_pseudo_core_absorb_watch_rows !=
          ale_defaults.central_pseudo_core_absorb_watch_rows ||
      numerics.ale.central_pseudo_core_terminal_absorb_enabled ||
      numerics.ale.central_pseudo_core_terminal_rebound_factor !=
          ale_defaults.central_pseudo_core_terminal_rebound_factor ||
      numerics.ale.central_pseudo_core_terminal_tail_dt_s !=
          ale_defaults.central_pseudo_core_terminal_tail_dt_s ||
      numerics.ale.remap_mass_closure_reject_tol !=
          ale_defaults.remap_mass_closure_reject_tol ||
      numerics.ale.rezone_closure_cooldown_steps !=
          ale_defaults.rezone_closure_cooldown_steps) {
    ale["central_pseudo_core_spherical_absorb_gasfront"] =
        numerics.ale.central_pseudo_core_spherical_absorb_gasfront;
    ale["central_pseudo_core_spherical_absorb_alpha"] =
        numerics.ale.central_pseudo_core_spherical_absorb_alpha;
    ale["central_pseudo_core_spherical_absorb_pjump"] =
        numerics.ale.central_pseudo_core_spherical_absorb_pjump;
    ale["central_pseudo_core_mixed_absorb_enabled"] =
        numerics.ale.central_pseudo_core_mixed_absorb_enabled;
    ale["central_pseudo_core_absorb_watch_rows"] =
        numerics.ale.central_pseudo_core_absorb_watch_rows;
    ale["central_pseudo_core_terminal_absorb_enabled"] =
        numerics.ale.central_pseudo_core_terminal_absorb_enabled;
    ale["central_pseudo_core_terminal_rebound_factor"] =
        numerics.ale.central_pseudo_core_terminal_rebound_factor;
    ale["central_pseudo_core_terminal_tail_dt_s"] =
        numerics.ale.central_pseudo_core_terminal_tail_dt_s;
    ale["remap_mass_closure_reject_tol"] =
        numerics.ale.remap_mass_closure_reject_tol;
    ale["rezone_closure_cooldown_steps"] =
        numerics.ale.rezone_closure_cooldown_steps;
  }
  if (numerics.ale.csr_optionb_coherent_enabled ||
      numerics.ale.csr_optionb_velocity_remap_enabled ||
      numerics.ale.pole_axis_bbsw_enabled ||
      numerics.ale.axis_contact_guard_enabled ||
      numerics.ale.mass_floor_absorb_enabled ||
      numerics.ale.interior_patch_remap_enabled) {
    ale["csr_optionb_coherent_enabled"] =
        numerics.ale.csr_optionb_coherent_enabled;
    ale["csr_optionb_velocity_remap_enabled"] =
        numerics.ale.csr_optionb_velocity_remap_enabled;
    ale["pole_axis_bbsw_enabled"] =
        numerics.ale.pole_axis_bbsw_enabled;
    ale["axis_contact_guard_enabled"] =
        numerics.ale.axis_contact_guard_enabled;
    ale["mass_floor_absorb_enabled"] =
        numerics.ale.mass_floor_absorb_enabled;
    ale["interior_patch_remap_enabled"] =
        numerics.ale.interior_patch_remap_enabled;
  }
  if (numerics.ale.pole_sector_rezone_enabled ||
      numerics.ale.pole_sector_rezone_m_theta !=
          ale_defaults.pole_sector_rezone_m_theta ||
      numerics.ale.pole_sector_rezone_lambda !=
          ale_defaults.pole_sector_rezone_lambda ||
      numerics.ale.pole_sector_rezone_mode !=
          ale_defaults.pole_sector_rezone_mode ||
      numerics.ale.pole_sector_rezone_deadband_frac !=
          ale_defaults.pole_sector_rezone_deadband_frac) {
    ale["pole_sector_rezone_enabled"] =
        numerics.ale.pole_sector_rezone_enabled;
    ale["pole_sector_rezone_m_theta"] =
        numerics.ale.pole_sector_rezone_m_theta;
    ale["pole_sector_rezone_lambda"] =
        numerics.ale.pole_sector_rezone_lambda;
    ale["pole_sector_rezone_mode"] = numerics.ale.pole_sector_rezone_mode;
    ale["pole_sector_rezone_deadband_frac"] =
        numerics.ale.pole_sector_rezone_deadband_frac;
  }

  py::dict plic;
  plic["enabled"] = numerics.plic.enabled;
  plic["normal_estimator"] = numerics.plic.normal_estimator;
  plic["t0_volume_cut_method"] = numerics.plic.t0_volume_cut_method;
  plic["t0_volume_cut_max_depth"] = numerics.plic.t0_volume_cut_max_depth;
  plic["t0_volume_cut_volfrac_tol"] =
      numerics.plic.t0_volume_cut_volfrac_tol;
  plic["fast_path_threshold_min"] = numerics.plic.fast_path_threshold_min;
  plic["fast_path_threshold_max"] = numerics.plic.fast_path_threshold_max;
  plic["fast_path_halo_radius_cells"] =
      numerics.plic.fast_path_halo_radius_cells;
  plic["alpha_solver_max_iter"] = numerics.plic.alpha_solver_max_iter;
  plic["alpha_tolerance_rel"] = numerics.plic.alpha_tolerance_rel;
  plic["thermodynamic_error_soft_threshold"] =
      numerics.plic.thermodynamic_error_soft_threshold;
  plic["thermodynamic_error_hard_threshold"] =
      numerics.plic.thermodynamic_error_hard_threshold;
  plic["class_d_dense_fraction_threshold"] =
      numerics.plic.class_d_dense_fraction_threshold;
  plic["material_interface_per_cell_state"] =
      numerics.plic.material_interface_per_cell_state;
  plic["production_comparable_gate_strict"] =
      numerics.plic.production_comparable_gate_strict;
  plic["drift_sensor_max_relative"] = numerics.plic.drift_sensor_max_relative;
  plic["drift_sensor_max_swept_fraction"] =
      numerics.plic.drift_sensor_max_swept_fraction;
  plic["prev_normal_freshness_volfrac_threshold"] =
      numerics.plic.prev_normal_freshness_volfrac_threshold;
  plic["plic_per_step_cost_target_fraction"] =
      numerics.plic.plic_per_step_cost_target_fraction;
  plic["in_run_disabled"] = numerics.plic.in_run_disabled;
  plic["rho_material_aware_donor"] =
      numerics.plic.rho_material_aware_donor;

  py::dict materials;
  materials["per_material_conservation_enabled"] =
      numerics.materials.per_material_conservation_enabled;
  materials["presence_threshold_volfrac"] =
      numerics.materials.presence_threshold_volfrac;
  materials["presence_threshold_mass_density_g_per_cc"] =
      numerics.materials.presence_threshold_mass_density_g_per_cc;
  py::dict eos_bounds;
  for (const auto& [name, lower] :
       numerics.materials.eos_table_validity_lower_bound_g_per_cc) {
    eos_bounds[py::str(name)] = lower;
  }
  materials["eos_table_validity_lower_bound_g_per_cc"] = eos_bounds;
  materials["lazy_cache_te_m_enabled"] =
      numerics.materials.lazy_cache_te_m_enabled;
  materials["hdf5_emit_derived_per_material"] =
      numerics.materials.hdf5_emit_derived_per_material;
  materials["deposit_redistribute_fallback_enabled"] =
      numerics.materials.deposit_redistribute_fallback_enabled;
  materials["deposit_redistribute_provenance_label"] =
      numerics.materials.deposit_redistribute_provenance_label;
  materials["conservation_residual_warn_threshold_rel"] =
      numerics.materials.conservation_residual_warn_threshold_rel;
  materials["conservation_residual_hard_warning_threshold_rel"] =
      numerics.materials.conservation_residual_hard_warning_threshold_rel;

  auto serialize_tol = [](const auto& tol) {
    py::dict out;
    out["soft"] = tol.soft;
    out["hard"] = tol.hard;
    return out;
  };

  py::dict ale1d;
  ale1d["enabled"] = numerics.ale1d.enabled;
  ale1d["every_n_steps"] = numerics.ale1d.every_n_steps;
  ale1d["min_steps_between_ale"] = numerics.ale1d.min_steps_between_ale;
  ale1d["enable_benefit_gate"] = numerics.ale1d.enable_benefit_gate;
  ale1d["benefit_min_dt_gain"] = numerics.ale1d.benefit_min_dt_gain;
  ale1d["candidate_dt_penalty_max"] =
      numerics.ale1d.candidate_dt_penalty_max;
  ale1d["emergency_enabled"] = numerics.ale1d.emergency_enabled;
  ale1d["min_cells"] = numerics.ale1d.min_cells;
  ale1d["protected_fraction_max"] = numerics.ale1d.protected_fraction_max;
  ale1d["min_movable_segment_warn"] =
      numerics.ale1d.min_movable_segment_warn;
  ale1d["min_movable_segment_hard"] =
      numerics.ale1d.min_movable_segment_hard;
  ale1d["max_node_displacement_fraction_mu"] =
      numerics.ale1d.max_node_displacement_fraction_mu;
  ale1d["max_node_displacement_fraction_r"] =
      numerics.ale1d.max_node_displacement_fraction_r;
  ale1d["ke_conservation_closure"] =
      numerics.ale1d.ke_conservation_closure;
  ale1d["total_mass_tol"] = serialize_tol(numerics.ale1d.total_mass_tol);
  ale1d["material_mass_tol"] = serialize_tol(numerics.ale1d.material_mass_tol);
  ale1d["radiation_group_energy_tol"] =
      serialize_tol(numerics.ale1d.radiation_group_energy_tol);
  ale1d["material_internal_energy_tol"] =
      serialize_tol(numerics.ale1d.material_internal_energy_tol);
  ale1d["total_material_energy_tol"] =
      serialize_tol(numerics.ale1d.total_material_energy_tol);
  ale1d["global_total_energy_tol"] =
      serialize_tol(numerics.ale1d.global_total_energy_tol);
  ale1d["kinetic_energy_drift_tol"] =
      serialize_tol(numerics.ale1d.kinetic_energy_drift_tol);
  ale1d["diagnostics_enabled"] = numerics.ale1d.diagnostics_enabled;
  ale1d["diagnostics_log_every_n_steps"] =
      numerics.ale1d.diagnostics_log_every_n_steps;
  ale1d["diagnostics_collect_step_result"] =
      numerics.ale1d.diagnostics_collect_step_result;
  ale1d["diagnostics_fail_on_unexpected_apply"] =
      numerics.ale1d.diagnostics_fail_on_unexpected_apply;
  py::dict laser_sensor;
  laser_sensor["enabled"] = numerics.ale1d.laser_sensor.enabled;
  laser_sensor["target_cells_fraction"] =
      numerics.ale1d.laser_sensor.target_cells_fraction;
  laser_sensor["sigma_min_cells"] = numerics.ale1d.laser_sensor.sigma_min_cells;
  laser_sensor["sigma_max_cells"] = numerics.ale1d.laser_sensor.sigma_max_cells;
  laser_sensor["peak_fraction"] = numerics.ale1d.laser_sensor.peak_fraction;
  laser_sensor["conf_low"] = numerics.ale1d.laser_sensor.conf_low;
  laser_sensor["conf_high"] = numerics.ale1d.laser_sensor.conf_high;
  ale1d["laser_sensor"] = laser_sensor;

  py::dict ablation_sensor;
  ablation_sensor["enabled"] = numerics.ale1d.ablation_sensor.enabled;
  ablation_sensor["target_cells_fraction"] =
      numerics.ale1d.ablation_sensor.target_cells_fraction;
  ablation_sensor["sigma_min_cells"] =
      numerics.ale1d.ablation_sensor.sigma_min_cells;
  ablation_sensor["sigma_max_cells"] =
      numerics.ale1d.ablation_sensor.sigma_max_cells;
  ablation_sensor["peak_fraction"] =
      numerics.ale1d.ablation_sensor.peak_fraction;
  ablation_sensor["reference_density_gcc"] =
      numerics.ale1d.ablation_sensor.reference_density_gcc;
  ablation_sensor["rho_gate_frac"] =
      numerics.ale1d.ablation_sensor.rho_gate_frac;
  ablation_sensor["rho_gate_width"] =
      numerics.ale1d.ablation_sensor.rho_gate_width;
  ablation_sensor["te_gate_low_eV"] =
      numerics.ale1d.ablation_sensor.te_gate_low_eV;
  ablation_sensor["te_gate_high_eV"] =
      numerics.ale1d.ablation_sensor.te_gate_high_eV;
  ablation_sensor["conf_low"] = numerics.ale1d.ablation_sensor.conf_low;
  ablation_sensor["conf_high"] = numerics.ale1d.ablation_sensor.conf_high;
  ale1d["ablation_sensor"] = ablation_sensor;

  py::dict shock_sensor;
  shock_sensor["enabled"] = numerics.ale1d.shock_sensor.enabled;
  shock_sensor["target_cells_fraction"] =
      numerics.ale1d.shock_sensor.target_cells_fraction;
  shock_sensor["sigma_min_cells"] = numerics.ale1d.shock_sensor.sigma_min_cells;
  shock_sensor["sigma_max_cells"] = numerics.ale1d.shock_sensor.sigma_max_cells;
  shock_sensor["peak_fraction"] = numerics.ale1d.shock_sensor.peak_fraction;
  shock_sensor["qvisc_conf_low"] = numerics.ale1d.shock_sensor.qvisc_conf_low;
  shock_sensor["qvisc_conf_high"] = numerics.ale1d.shock_sensor.qvisc_conf_high;
  shock_sensor["du_cs_conf_low"] = numerics.ale1d.shock_sensor.du_cs_conf_low;
  shock_sensor["du_cs_conf_high"] = numerics.ale1d.shock_sensor.du_cs_conf_high;
  ale1d["shock_sensor"] = shock_sensor;

  py::dict interface_sensor;
  interface_sensor["enabled"] = numerics.ale1d.interface_sensor.enabled;
  interface_sensor["target_cells_fraction"] =
      numerics.ale1d.interface_sensor.target_cells_fraction;
  interface_sensor["target_cells_cap_fraction"] =
      numerics.ale1d.interface_sensor.target_cells_cap_fraction;
  interface_sensor["max_features"] = numerics.ale1d.interface_sensor.max_features;
  interface_sensor["min_separation_cells"] =
      numerics.ale1d.interface_sensor.min_separation_cells;
  interface_sensor["jump_low"] = numerics.ale1d.interface_sensor.jump_low;
  interface_sensor["jump_high"] = numerics.ale1d.interface_sensor.jump_high;
  interface_sensor["sigma_min_cells"] =
      numerics.ale1d.interface_sensor.sigma_min_cells;
  interface_sensor["sigma_max_cells"] =
      numerics.ale1d.interface_sensor.sigma_max_cells;
  interface_sensor["pin_interfaces"] =
      numerics.ale1d.interface_sensor.pin_interfaces;
  ale1d["interface_sensor"] = interface_sensor;

  py::dict center_sensor;
  center_sensor["enabled"] = numerics.ale1d.center_sensor.enabled;
  center_sensor["target_cells_fraction"] =
      numerics.ale1d.center_sensor.target_cells_fraction;
  center_sensor["sigma_min_cells"] = numerics.ale1d.center_sensor.sigma_min_cells;
  center_sensor["sigma_max_cells"] = numerics.ale1d.center_sensor.sigma_max_cells;
  center_sensor["search_x"] = numerics.ale1d.center_sensor.search_x;
  ale1d["center_sensor"] = center_sensor;

  py::dict ale1d_rezone;
  ale1d_rezone["monitor_floor"] = numerics.ale1d.rezone.monitor_floor;
  ale1d_rezone["monitor_wmax_ratio"] =
      numerics.ale1d.rezone.monitor_wmax_ratio;
  ale1d_rezone["monitor_smoothing_iterations"] =
      numerics.ale1d.rezone.monitor_smoothing_iterations;
  ale1d_rezone["monitor_smooth_across_protected_faces"] =
      numerics.ale1d.rezone.monitor_smooth_across_protected_faces;
  ale1d_rezone["min_floor_fraction"] =
      numerics.ale1d.rezone.min_floor_fraction;
  ale1d_rezone["gaussian_truncation_sigma"] =
      numerics.ale1d.rezone.gaussian_truncation_sigma;
  ale1d_rezone["spatial_monitor_enabled"] =
      numerics.ale1d.rezone.spatial_monitor_enabled;
  ale1d_rezone["spatial_target_cells_fraction"] =
      numerics.ale1d.rezone.spatial_target_cells_fraction;
  ale1d_rezone["spatial_power"] = numerics.ale1d.rezone.spatial_power;
  ale1d_rezone["laser_spatial_dr_min_cm"] =
      numerics.ale1d.rezone.laser_spatial_dr_min_cm;
  ale1d_rezone["laser_spatial_dr_max_cm"] =
      numerics.ale1d.rezone.laser_spatial_dr_max_cm;
  ale1d_rezone["ablation_spatial_dr_min_cm"] =
      numerics.ale1d.rezone.ablation_spatial_dr_min_cm;
  ale1d_rezone["ablation_spatial_dr_max_cm"] =
      numerics.ale1d.rezone.ablation_spatial_dr_max_cm;
  ale1d_rezone["shock_spatial_dr_min_cm"] =
      numerics.ale1d.rezone.shock_spatial_dr_min_cm;
  ale1d_rezone["shock_spatial_dr_max_cm"] =
      numerics.ale1d.rezone.shock_spatial_dr_max_cm;
  ale1d["rezone"] = ale1d_rezone;

  py::dict ale1d_min_width_floor;
  ale1d_min_width_floor["enabled"] =
      numerics.ale1d.min_width_floor.enabled;
  ale1d_min_width_floor["floor_cm"] =
      numerics.ale1d.min_width_floor.floor_cm;
  ale1d_min_width_floor["target_factor"] =
      numerics.ale1d.min_width_floor.target_factor;
  ale1d_min_width_floor["relief_halfwidth_cells"] =
      numerics.ale1d.min_width_floor.relief_halfwidth_cells;
  ale1d_min_width_floor["max_growth_factor"] =
      numerics.ale1d.min_width_floor.max_growth_factor;
  ale1d_min_width_floor["retrigger_cooldown_steps"] =
      numerics.ale1d.min_width_floor.retrigger_cooldown_steps;
  ale1d["min_width_floor"] = ale1d_min_width_floor;

  py::dict ale1d_remap;
  ale1d_remap["reject_multicell_sweeps"] =
      numerics.ale1d.remap.reject_multicell_sweeps;
  ale1d_remap["high_order_enabled"] =
      numerics.ale1d.remap.high_order_enabled;
  ale1d_remap["limiter_theta"] = numerics.ale1d.remap.limiter_theta;
  ale1d_remap["high_order_ramp_cells"] =
      numerics.ale1d.remap.high_order_ramp_cells;
  ale1d_remap["radiation_high_order_ramp_cells"] =
      numerics.ale1d.remap.radiation_high_order_ramp_cells;
  ale1d_remap["fallback_to_first_order_on_bounds_fail"] =
      numerics.ale1d.remap.fallback_to_first_order_on_bounds_fail;
  ale1d_remap["reject_strict_zero_flux_on_moving_protected_face"] =
      numerics.ale1d.remap.reject_strict_zero_flux_on_moving_protected_face;
  ale1d["remap"] = ale1d_remap;

  py::dict floors;
  floors["rho"] = numerics.floors.rho;
  floors["Te"] = numerics.floors.Te;
  floors["Ti"] = numerics.floors.Ti;

  py::dict safety;
  safety["energy_fatal"] = numerics.safety.energy_fatal;
  safety["nan_fatal"] = numerics.safety.nan_fatal;
  safety["energy_budget_tol"] = numerics.safety.energy_budget_tol;
  safety["opacity_floor"] = numerics.safety.opacity_floor;
  safety["opacity_cap"] = numerics.safety.opacity_cap;
  safety["clamp_warn_threshold"] = numerics.safety.clamp_warn_threshold;
  safety["clamp_fatal_threshold"] = numerics.safety.clamp_fatal_threshold;
  safety["overshoot_warn"] = numerics.safety.overshoot_warn;
  safety["overshoot_fatal"] = numerics.safety.overshoot_fatal;
  safety["overshoot_fatal_enabled"] = numerics.safety.overshoot_fatal_enabled;

  py::dict debug;
  debug["trace_mesh_motion"] = numerics.debug.trace_mesh_motion;
  debug["trace_mesh_node_selector"] =
      numerics.debug.trace_mesh_node_selector;
  debug["trace_mesh_cell"] = numerics.debug.trace_mesh_cell;
  debug["trace_max_steps"] = numerics.debug.trace_max_steps;

  py::dict diagnostics;
  diagnostics["phase_resolved_energy"] =
      numerics.diagnostics.phase_resolved_energy;
  diagnostics["r_momentum_source_audit"] =
      numerics.diagnostics.r_momentum_source_audit;
  diagnostics["dt_breakdown_history_enabled"] =
      numerics.diagnostics.dt_breakdown_history_enabled;
  py::dict mesh_attribution;
  mesh_attribution["enabled"] =
      numerics.diagnostics.mesh_attribution.enabled;
  mesh_attribution["record_node_displacements"] =
      numerics.diagnostics.mesh_attribution.record_node_displacements;
  mesh_attribution["dump_on_failure_only"] =
      numerics.diagnostics.mesh_attribution.dump_on_failure_only;
  mesh_attribution["enable_leave_one_out_replay"] =
      numerics.diagnostics.mesh_attribution.enable_leave_one_out_replay;
  diagnostics["mesh_attribution"] = mesh_attribution;
  py::dict icf;
  icf["enabled"] = numerics.diagnostics.icf.enabled;
  icf["rho_inner_threshold_g_per_cc"] =
      numerics.diagnostics.icf.rho_inner_threshold_g_per_cc;
  icf["rho_outer_threshold_g_per_cc"] =
      numerics.diagnostics.icf.rho_outer_threshold_g_per_cc;
  diagnostics["icf"] = icf;
  py::dict hotspot_gas;
  hotspot_gas["enabled"] = numerics.diagnostics.hotspot_gas.enabled;
  hotspot_gas["R_g_cm"] = numerics.diagnostics.hotspot_gas.R_g_cm;
  hotspot_gas["mass_drift_warn_rel"] =
      numerics.diagnostics.hotspot_gas.mass_drift_warn_rel;
  diagnostics["hotspot_gas"] = hotspot_gas;
  py::dict conservation;
  conservation["enabled"] = numerics.diagnostics.conservation.enabled;
  diagnostics["conservation"] = conservation;
  py::dict ale_provenance_emission;
  ale_provenance_emission["enabled"] =
      numerics.diagnostics.ale_provenance_emission.enabled;
  diagnostics["ale_provenance_emission"] = ale_provenance_emission;
  if (numerics.diagnostics.ale_velcoherence.enabled ||
      numerics.diagnostics.ale_velcoherence.every_n_steps != 1) {
    py::dict ale_velcoherence;
    ale_velcoherence["enabled"] =
        numerics.diagnostics.ale_velcoherence.enabled;
    ale_velcoherence["every_n_steps"] =
        numerics.diagnostics.ale_velcoherence.every_n_steps;
    diagnostics["ale_velcoherence"] = ale_velcoherence;
  }
  py::dict mesh_quality_min;
  mesh_quality_min["enabled"] =
      numerics.diagnostics.mesh_quality_min.enabled;
  diagnostics["mesh_quality_min"] = mesh_quality_min;
  py::dict shock_approach;
  shock_approach["enabled"] =
      numerics.diagnostics.shock_approach.enabled;
  shock_approach["every"] =
      numerics.diagnostics.shock_approach.every;
  shock_approach["target_radius_cm"] =
      numerics.diagnostics.shock_approach.target_radius_cm;
  shock_approach["bins"] =
      numerics.diagnostics.shock_approach.bins;
  shock_approach["h_cell_cm"] =
      numerics.diagnostics.shock_approach.h_cell_cm;
  diagnostics["shock_approach"] = shock_approach;
  py::dict production_audit;
  production_audit["enabled"] =
      numerics.diagnostics.production_audit.enabled;
  production_audit["tier"] = numerics.diagnostics.production_audit.tier;
  production_audit["audit_json_path"] =
      numerics.diagnostics.production_audit.audit_json_path;
  py::dict escape_valve_budget;
  escape_valve_budget["mass_max"] =
      numerics.diagnostics.production_audit.escape_valve_budget.mass_max;
  escape_valve_budget["energy_max"] =
      numerics.diagnostics.production_audit.escape_valve_budget.energy_max;
  production_audit["escape_valve_budget"] = escape_valve_budget;
  py::list region_of_interest;
  for (const auto& region :
       numerics.diagnostics.production_audit.region_of_interest) {
    py::dict out_region;
    out_region["i_min"] = region.i_min;
    out_region["i_max"] = region.i_max;
    out_region["j_min"] = region.j_min;
    out_region["j_max"] = region.j_max;
    region_of_interest.append(out_region);
  }
  production_audit["region_of_interest"] = region_of_interest;
  py::dict gcl;
  gcl["enabled"] = numerics.diagnostics.production_audit.gcl.enabled;
  production_audit["gcl"] = gcl;
  py::dict audit_positivity;
  audit_positivity["enabled"] =
      numerics.diagnostics.production_audit.positivity.enabled;
  audit_positivity["fatal_on_neg"] =
      numerics.diagnostics.production_audit.positivity.fatal_on_neg;
  production_audit["positivity"] = audit_positivity;
  diagnostics["production_audit"] = production_audit;
  py::dict mesh_degeneracy_forensics;
  mesh_degeneracy_forensics["enabled"] =
      numerics.diagnostics.mesh_degeneracy_forensics.enabled;
  mesh_degeneracy_forensics["corner_j_source_budget_enabled"] =
      numerics.diagnostics.mesh_degeneracy_forensics
          .corner_j_source_budget_enabled;
  mesh_degeneracy_forensics["corner_j_source_budget_include_1_ring"] =
      numerics.diagnostics.mesh_degeneracy_forensics
          .corner_j_source_budget_include_1_ring;
  mesh_degeneracy_forensics["velocity_history_enabled"] =
      numerics.diagnostics.mesh_degeneracy_forensics.velocity_history_enabled;
  mesh_degeneracy_forensics["velocity_history_target_cell_c"] =
      numerics.diagnostics.mesh_degeneracy_forensics
          .velocity_history_target_cell_c;
  mesh_degeneracy_forensics["velocity_history_sample_every_n_steps"] =
      numerics.diagnostics.mesh_degeneracy_forensics
          .velocity_history_sample_every_n_steps;
  mesh_degeneracy_forensics["velocity_history_include_1_ring"] =
      numerics.diagnostics.mesh_degeneracy_forensics
          .velocity_history_include_1_ring;
  mesh_degeneracy_forensics["velocity_history_max_records"] =
      numerics.diagnostics.mesh_degeneracy_forensics
          .velocity_history_max_records;
  mesh_degeneracy_forensics["same_cell_count"] =
      numerics.diagnostics.mesh_degeneracy_forensics.same_cell_count;
  mesh_degeneracy_forensics["sigma_threshold"] =
      numerics.diagnostics.mesh_degeneracy_forensics.sigma_threshold;
  mesh_degeneracy_forensics["max_dumps_per_run"] =
      numerics.diagnostics.mesh_degeneracy_forensics.max_dumps_per_run;
  mesh_degeneracy_forensics["output_dir"] =
      numerics.diagnostics.mesh_degeneracy_forensics.output_dir;
  diagnostics["mesh_degeneracy_forensics"] = mesh_degeneracy_forensics;

  py::dict profile;
  py::dict icf_standard_ale;
  icf_standard_ale["enabled"] = numerics.profile.icf_standard_ale.enabled;
  icf_standard_ale["enforce"] = numerics.profile.icf_standard_ale.enforce;
  icf_standard_ale["claim_level"] =
      numerics.profile.icf_standard_ale.claim_level;
  py::dict allowed_when_enabled;
  allowed_when_enabled["ale_enabled_required_value"] =
      numerics.profile.icf_standard_ale.allowed_when_enabled
          .ale_enabled_required_value;
  allowed_when_enabled["ale_axis_repair_mode_required_value"] =
      numerics.profile.icf_standard_ale.allowed_when_enabled
          .ale_axis_repair_mode_required_value;
  allowed_when_enabled["ale_remap_scheme_allowed_values"] =
      numerics.profile.icf_standard_ale.allowed_when_enabled
          .ale_remap_scheme_allowed_values;
  allowed_when_enabled["ale_donor_sign_fixed"] =
      numerics.profile.icf_standard_ale.allowed_when_enabled
          .ale_donor_sign_fixed_allowed_values;
  allowed_when_enabled["hydro_driver_full_step_retry_enabled_required_value"] =
      numerics.profile.icf_standard_ale.allowed_when_enabled
          .hydro_driver_full_step_retry_enabled_required_value;
  icf_standard_ale["allowed_when_enabled"] = allowed_when_enabled;
  py::dict forbidden_when_enabled;
  forbidden_when_enabled[
      "hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value"] =
      numerics.profile.icf_standard_ale.forbidden_when_enabled
          .hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value;
  forbidden_when_enabled["ale_local_boundary_repair_enabled_forbidden_value"] =
      numerics.profile.icf_standard_ale.forbidden_when_enabled
          .ale_local_boundary_repair_enabled_forbidden_value;
  forbidden_when_enabled[
      "ale_multi_node_boundary_repair_enabled_forbidden_value"] =
      numerics.profile.icf_standard_ale.forbidden_when_enabled
          .ale_multi_node_boundary_repair_enabled_forbidden_value;
  forbidden_when_enabled[
      "ale_multi_node_interior_repair_enabled_forbidden_value"] =
      numerics.profile.icf_standard_ale.forbidden_when_enabled
          .ale_multi_node_interior_repair_enabled_forbidden_value;
  forbidden_when_enabled[
      "ale_axis_variational_projection_enabled_forbidden_value"] =
      numerics.profile.icf_standard_ale.forbidden_when_enabled
          .ale_axis_variational_projection_enabled_forbidden_value;
  forbidden_when_enabled[
      "ale_emergency_cell_deactivation_enabled_forbidden_value"] =
      numerics.profile.icf_standard_ale.forbidden_when_enabled
          .ale_emergency_cell_deactivation_enabled_forbidden_value;
  forbidden_when_enabled[
      "hydro_driver_retry_active_mesh_repair_enabled_forbidden_value"] =
      numerics.profile.icf_standard_ale.forbidden_when_enabled
          .hydro_driver_retry_active_mesh_repair_enabled_forbidden_value;
  icf_standard_ale["forbidden_when_enabled"] = forbidden_when_enabled;
  py::dict escape_valves;
  escape_valves["allow_nonstandard_mesh_rescue"] =
      numerics.profile.icf_standard_ale.escape_valves
          .allow_nonstandard_mesh_rescue;
  escape_valves["require_deck_reason"] =
      numerics.profile.icf_standard_ale.escape_valves.require_deck_reason;
  escape_valves["mark_run_nonstandard"] =
      numerics.profile.icf_standard_ale.escape_valves.mark_run_nonstandard;
  icf_standard_ale["escape_valves"] = escape_valves;
  profile["icf_standard_ale"] = icf_standard_ale;
  py::dict legacy_regression;
  legacy_regression["enabled"] = numerics.profile.legacy_regression.enabled;
  legacy_regression["revision"] = numerics.profile.legacy_regression.revision;
  profile["legacy_regression"] = legacy_regression;

  py::dict out;
  out["T_start_eV"] = numerics.T_start_eV;
  out["radiation_thermal_subcycle"] = numerics.radiation_thermal_subcycle;
  out["has_physical_rz_axis"] = numerics.has_physical_rz_axis;
  out["persistent_loop"] = persistent_loop;
  out["dt"] = dt;
  out["hydro"] = hydro;
  out["conduction"] = conduction;
  out["ale"] = ale;
  out["plic"] = plic;
  out["materials"] = materials;
  out["ale1d"] = ale1d;
  out["floors"] = floors;
  out["positivity_clamp"] = numerics.positivity_clamp;
  out["safety"] = safety;
  out["debug"] = debug;
  out["diagnostics"] = diagnostics;
  out["profile"] = profile;
  out["diagnostics_every"] = numerics.diagnostics_every;
  return out;
}

py::dict serialize_output(const Config::OutputConfig& output) {
  py::dict out;
  out["directory"] = output.directory;
  out["format"] = output.format;
  out["plot_every"] = output.plot_every;
  out["history_every"] = output.history_every;
  out["checkpoint_every"] = output.checkpoint_every;
  out["plot_every_s"] = output.plot_every_s;
  out["history_every_s"] = output.history_every_s;
  out["checkpoint_every_s"] = output.checkpoint_every_s;
  out["checkpoint_keep_last"] = output.checkpoint_keep_last;
  out["compression"] = output.compression;
  out["compression_level"] = output.compression_level;
  out["save_namelist_copy"] = output.save_namelist_copy;
  out["save_frozen_config"] = output.save_frozen_config;
  out["plot_fields"] = output.plot_fields;
  return out;
}

py::dict serialize_diagnostics(const Config::DiagnosticsConfig& diagnostics) {
  py::dict energy_budget;
  energy_budget["enabled"] = diagnostics.energy_budget.enabled;
  energy_budget["warn_threshold"] = diagnostics.energy_budget.warn_threshold;

  py::dict areal_density;
  areal_density["enabled"] = diagnostics.areal_density.enabled;
  areal_density["r_range"] = diagnostics.areal_density.r_range;
  areal_density["angles_deg"] = diagnostics.areal_density.angles_deg;

  py::dict sphericity;
  sphericity["enabled"] = diagnostics.sphericity.enabled;
  sphericity["surface"] = diagnostics.sphericity.surface;
  sphericity["rho_threshold"] = diagnostics.sphericity.rho_threshold;
  sphericity["modes"] = diagnostics.sphericity.modes;

  py::dict laser_pattern;
  laser_pattern["enabled"] = diagnostics.laser_pattern.enabled;
  laser_pattern["absorbed_power_profile"] =
      diagnostics.laser_pattern.absorbed_power_profile;
  laser_pattern["critical_surface"] = diagnostics.laser_pattern.critical_surface;
  laser_pattern["per_beam"] = diagnostics.laser_pattern.per_beam;

  py::dict mc_stats;
  mc_stats["enabled"] = diagnostics.mc_stats.enabled;
  mc_stats["particle_counts"] = diagnostics.mc_stats.particle_counts;
  mc_stats["weight_stats"] = diagnostics.mc_stats.weight_stats;
  mc_stats["cell_particle_density"] = diagnostics.mc_stats.cell_particle_density;
  mc_stats["ddmc_fraction"] = diagnostics.mc_stats.ddmc_fraction;

  py::dict fleck_diag;
  fleck_diag["enabled"] = diagnostics.fleck_diag.enabled;
  fleck_diag["every"] = diagnostics.fleck_diag.every;
  fleck_diag["cells"] = diagnostics.fleck_diag.cells;
  fleck_diag["r_min_cm"] = diagnostics.fleck_diag.r_min_cm;
  fleck_diag["r_max_cm"] = diagnostics.fleck_diag.r_max_cm;

  py::dict out;
  out["enabled"] = diagnostics.enabled;
  out["every"] = diagnostics.every;
  out["per_operator_radial_fourier_enabled"] =
      diagnostics.per_operator_radial_fourier_enabled;
  out["radial_fourier_window_t_start_s"] =
      diagnostics.radial_fourier_window_t_start_s;
  out["radial_fourier_window_t_end_s"] =
      diagnostics.radial_fourier_window_t_end_s;
  out["radial_fourier_max_mode"] = diagnostics.radial_fourier_max_mode;
  out["per_operator_radial_fourier_complex_enabled"] =
      diagnostics.per_operator_radial_fourier_complex_enabled;
  out["per_operator_radial_fourier_complex_m_targets"] =
      diagnostics.per_operator_radial_fourier_complex_m_targets;
  out["per_operator_radial_fourier_complex_j_targets"] =
      diagnostics.per_operator_radial_fourier_complex_j_targets;
  out["per_operator_radial_fourier_complex_fields"] =
      diagnostics.per_operator_radial_fourier_complex_fields;
  out["energy_budget"] = energy_budget;
  out["areal_density"] = areal_density;
  out["sphericity"] = sphericity;
  out["laser_pattern"] = laser_pattern;
  out["mc_stats"] = mc_stats;
  out["fleck_diag"] = fleck_diag;
  out["overshoot_monitor"] = diagnostics.overshoot_monitor;
  return out;
}

py::dict serialize_parallel(const Config::ParallelConfig& parallel) {
  py::dict decomposition;
  decomposition["method"] = parallel.decomposition.method;
  decomposition["dims"] = parallel.decomposition.dims;
  decomposition["min_cells_per_rank"] = parallel.decomposition.min_cells_per_rank;

  py::dict halo;
  halo["gpu_aware_mpi"] = parallel.halo.gpu_aware_mpi;
  halo["ghost_layers"] = parallel.halo.ghost_layers;

  py::dict migration;
  migration["method"] = parallel.migration.method;
  migration["max_substeps"] = parallel.migration.max_substeps;
  migration["emigrant_threshold"] = parallel.migration.emigrant_threshold;
  migration["initial_capacity"] = parallel.migration.initial_capacity;
  migration["growth_factor"] = parallel.migration.growth_factor;

  py::dict out;
  out["decomposition"] = decomposition;
  out["halo"] = halo;
  out["migration"] = migration;
  return out;
}

py::dict serialize_burn(const Config::BurnConfig& burn) {
  const auto serialize_string_vector = [](const std::vector<std::string>& values) {
    py::list out;
    for (const auto& value : values) {
      out.append(py::str(value));
    }
    return out;
  };

  py::dict out;
  out["enabled"] = burn.enabled;
  out["fuels"] = serialize_string_vector(burn.fuels);
  out["scheme"] = burn.scheme;
  out["diffusion_groups"] = burn.diffusion_groups;
  out["diffusion_E_min_keV"] = burn.diffusion_E_min_keV;
  if (burn.scheme == "mc" || burn.mc_particles_per_cell !=
                              Config::BurnConfig{}.mc_particles_per_cell) {
    out["mc_particles_per_cell"] = burn.mc_particles_per_cell;
  }
  out["partition"] = burn.partition;
  out["screening"] = burn.screening;
  out["fuel_materials"] = serialize_string_vector(burn.fuel_materials);
  out["x_D"] = burn.x_D;
  out["x_T"] = burn.x_T;
  out["x_He3"] = burn.x_He3;
  out["T_floor_keV"] = burn.T_floor_keV;
  out["explicit_source_limit"] = burn.explicit_source_limit;
  out["eps_deplete"] = burn.eps_deplete;
  out["subcycle_max"] = burn.subcycle_max;
  out["vf_threshold"] = burn.vf_threshold;
  out["neutron_heating"] = burn.neutron_heating;
  out["neutron_heating_n_mu"] = burn.neutron_heating_n_mu;
  return out;
}

py::dict serialize_geometry_summary(const FreezeGeometrySummary& summary) {
  py::dict out;

  if (summary.has_rho) {
    py::dict rho;
    rho["min"] = summary.rho_min;
    rho["max"] = summary.rho_max;
    rho["mean"] = summary.rho_mean;
    out["rho"] = rho;
  }

  if (summary.has_Te) {
    py::dict Te;
    Te["min"] = summary.Te_min;
    Te["max"] = summary.Te_max;
    out["Te"] = Te;
  }

  if (summary.has_Ti) {
    py::dict Ti;
    Ti["min"] = summary.Ti_min;
    Ti["max"] = summary.Ti_max;
    out["Ti"] = Ti;
  }

  py::dict volfrac;
  for (const auto& [name, volume] : summary.material_volume) {
    volfrac[py::str(name)] = volume;
  }
  out["volFrac_volume"] = volfrac;
  return out;
}

py::dict serialize_table_summary(const FreezeTableSummary& table) {
  py::dict out;
  out["t_min"] = table.t_min;
  out["t_max"] = table.t_max;
  out["n_points"] = table.n_points;
  out["peak_value"] = table.peak_value;
  out["integrated_value"] = table.integrated_value;
  return out;
}

py::dict serialize_extras(const FreezeExtras& extras) {
  py::dict out;

  if (extras.geometry.has_value()) {
    out["geometry"] = serialize_geometry_summary(*extras.geometry);
  }

  py::dict tables;
  for (const auto& [name, summary] : extras.tables) {
    tables[py::str(name)] = serialize_table_summary(summary);
  }
  out["tables"] = tables;
  return out;
}

void validate_plic_config_for_freeze(const Config::NumericsConfig::PlicConfig& plic) {
  if (!plic.enabled) {
    return;
  }
  if (!(plic.alpha_tolerance_rel > 0.0)) {
    throw ConfigError("Numerics.plic.alpha_tolerance_rel must be > 0 when PLIC is enabled");
  }
  if (!(plic.t0_volume_cut_volfrac_tol > 0.0)) {
    throw ConfigError(
        "Numerics.plic.t0_volume_cut_volfrac_tol must be > 0 when PLIC is enabled");
  }
  if (!(plic.fast_path_threshold_max < 1.0)) {
    throw ConfigError(
        "Numerics.plic.fast_path_threshold_max must be < 1 when PLIC is enabled");
  }
  if (!(plic.fast_path_threshold_min > 0.0)) {
    throw ConfigError(
        "Numerics.plic.fast_path_threshold_min must be > 0 when PLIC is enabled");
  }
  if (plic.material_interface_per_cell_state == "dense_debug") {
    tenryu::core::log_warning(
        "Numerics.plic.material_interface_per_cell_state=\"dense_debug\" may substantially increase HDF5 output size");
  }
}

py::dict serialize_root(const NamelistConfig& config,
                        const FreezeExtras* extras,
                        const bool include_frozen_at,
                        const bool include_summary,
                        const bool include_tenryu_version) {
  validate_plic_config_for_freeze(config.numerics.plic);
  py::dict root;
  root["_schema_version"] = config.meta.schema_version;
  if (include_tenryu_version) {
    root["_tenryu_version"] = tenryu_version_string();
  }
  if (include_frozen_at) {
    root["_frozen_at"] = now_utc_iso8601();
  }
  root["_namelist_source_hash"] = config.meta.namelist_source_hash;
  root["main"] = serialize_main(config.main);
  root["mesh"] = serialize_mesh(config.mesh);
  root["materials"] = serialize_materials(config.materials);
  root["geometry"] = serialize_geometry(config.geometry);
  root["radiation"] = serialize_radiation(config.radiation);
  root["laser"] = serialize_laser(config.laser);
  root["numerics"] = serialize_numerics(config.numerics);
  root["output"] = serialize_output(config.output);
  root["diagnostics"] = serialize_diagnostics(config.diagnostics);
  root["parallel"] = serialize_parallel(config.parallel);
  root["burn"] = serialize_burn(config.burn);
  if (include_summary && extras != nullptr) {
    root["frozen_summary"] = serialize_extras(*extras);
  }
  return root;
}

void ensure_parent_directory(const std::filesystem::path& output_path) {
  const auto parent = output_path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
}

std::string dumps(const py::dict& obj, const bool pretty) {
  py::object json = py::module_::import("json");
  if (pretty) {
    return py::str(json.attr("dumps")(obj, py::arg("indent") = 2,
                                       py::arg("allow_nan") = false))
        .cast<std::string>();
  }
  return py::str(json.attr("dumps")(obj, py::arg("allow_nan") = false))
      .cast<std::string>();
}

bool dict_contains(const py::dict& dict, const char* key) {
  return dict.contains(py::str(key));
}

void set_default_if_missing(py::dict& dict, const char* key, const py::object& value) {
  if (!dict_contains(dict, key)) {
    dict[py::str(key)] = value;
  }
}

void pop_if_equal(py::dict& dict, const char* key, const py::object& value) {
  const py::str py_key(key);
  if (!dict.contains(py_key)) {
    return;
  }
  const int result = PyObject_RichCompareBool(dict[py_key].ptr(), value.ptr(), Py_EQ);
  if (result < 0) {
    throw py::error_already_set();
  }
  if (result == 1) {
    dict.attr("pop")(py_key);
  }
}

bool try_get_child_dict(const py::dict& parent, const char* key, py::dict* out) {
  if (!dict_contains(parent, key)) {
    return false;
  }
  const py::object child_obj = parent[py::str(key)];
  if (!py::isinstance<py::dict>(child_obj)) {
    return false;
  }
  *out = child_obj.cast<py::dict>();
  return true;
}

int read_schema_version_or_default(const py::dict& root, const int fallback) {
  if (!dict_contains(root, "_schema_version")) {
    return fallback;
  }
  try {
    return root[py::str("_schema_version")].cast<int>();
  } catch (const py::cast_error&) {
    return fallback;
  }
}

void apply_legacy_nlte_defaults(py::dict& root) {
  if (!dict_contains(root, "materials")) {
    return;
  }
  const py::object materials_obj = root[py::str("materials")];
  if (!py::isinstance<py::dict>(materials_obj)) {
    return;
  }
  py::dict materials = materials_obj.cast<py::dict>();
  if (!dict_contains(materials, "materials")) {
    return;
  }
  const py::object mats_obj = materials[py::str("materials")];
  if (!py::isinstance<py::list>(mats_obj)) {
    return;
  }
  py::list mats = mats_obj.cast<py::list>();
  const Config::MaterialsConfig::MatDef defaults;
  for (const py::handle mat_obj : mats) {
    if (!py::isinstance<py::dict>(mat_obj)) {
      continue;
    }
    py::dict mat = py::reinterpret_borrow<py::dict>(mat_obj);
    set_default_if_missing(mat, "lambda_method", py::str(defaults.lambda_method));
    set_default_if_missing(mat, "lambda_fd_delta_rel", py::cast(defaults.lambda_fd_delta_rel));
    set_default_if_missing(mat, "lambda_fd_abs_min", py::cast(defaults.lambda_fd_abs_min));
    set_default_if_missing(mat, "f_min", py::cast(defaults.nlte_f_min));
  }
}

void apply_legacy_mesh_defaults(py::dict& root) {
  py::dict mesh;
  if (!try_get_child_dict(root, "mesh", &mesh)) {
    return;
  }
  const Config::MeshConfig defaults;
  py::dict grading;
  grading["edge_ratio"] = defaults.grading.edge_ratio;
  grading["sg_order"] = defaults.grading.sg_order;
  grading["sg_sigma"] = defaults.grading.sg_sigma;
  grading["mapping"] = defaults.grading.mapping;
  set_default_if_missing(mesh, "grading", grading);
  // Per-key completion for checkpoints that predate grading.mapping
  // (2026-07-26 review): their grading dict exists but lacks
  // the key, so the whole-dict default above does not fill it.
  if (py::isinstance<py::dict>(mesh[py::str("grading")])) {
    py::dict grading_existing = mesh[py::str("grading")].cast<py::dict>();
    set_default_if_missing(grading_existing, "mapping",
                           py::str(defaults.grading.mapping));
  }
  set_default_if_missing(mesh, "explicit_nodes", py::cast(defaults.explicit_nodes));
  set_default_if_missing(mesh, "explicit_nodes_z",
                         py::cast(defaults.explicit_nodes_z));
  set_default_if_missing(mesh, "explicit_nodes_theta",
                         py::cast(defaults.explicit_nodes_theta));
  set_default_if_missing(mesh, "auto_regions_axis",
                         py::str(defaults.auto_regions_axis));
  set_default_if_missing(mesh, "logical_mesh_2d", py::str(defaults.logical_mesh_2d));
  set_default_if_missing(mesh, "polar_center_treatment",
                         py::str(defaults.polar_center_treatment));
  set_default_if_missing(mesh, "polar_equal_mu_zoning",
                         py::cast(defaults.polar_equal_mu_zoning));
  set_default_if_missing(mesh, "spherical_polar_s_max", py::cast(defaults.spherical_polar_s_max));
  set_default_if_missing(mesh, "polar_theta_min",
                         py::cast(defaults.polar_theta_min));
  set_default_if_missing(mesh, "box_center_z", py::cast(defaults.box_center_z));
  set_default_if_missing(mesh, "cone_theta_wall",
                         py::cast(defaults.cone_theta_wall));
  set_default_if_missing(mesh, "cone_tip_radius",
                         py::cast(defaults.cone_tip_radius));
  set_default_if_missing(mesh, "cone_activation_radius",
                         py::cast(defaults.cone_activation_radius));
  set_default_if_missing(mesh, "cone_fine_cells_minus",
                         py::cast(defaults.cone_fine_cells_minus));
  set_default_if_missing(mesh, "cone_fine_cells_plus",
                         py::cast(defaults.cone_fine_cells_plus));
  set_default_if_missing(mesh, "cone_angular_growth_max",
                         py::cast(defaults.cone_angular_growth_max));
  set_default_if_missing(mesh, "cone_tip_style",
                         py::str(defaults.cone_tip_style));
  if (dict_contains(mesh, "logical_mesh_2d") &&
      mesh[py::str("logical_mesh_2d")].cast<std::string>() == "polar_in_box") {
    set_default_if_missing(mesh, "polar_prefix_nr",
                           py::cast(defaults.polar_prefix_nr));
  }
  set_default_if_missing(mesh, "morph_rings", py::cast(defaults.morph_rings));
  set_default_if_missing(mesh, "collar_rings", py::cast(defaults.collar_rings));
  set_default_if_missing(mesh, "morph_growth_max",
                         py::cast(defaults.morph_growth_max));
  set_default_if_missing(mesh, "spherical_polar_kappa", py::cast(defaults.spherical_polar_kappa));
  set_default_if_missing(mesh, "multiblock_transition_scheme",
                         py::str(multiblock_transition_scheme_to_string(
                             defaults.multiblock_transition_scheme)));
  set_default_if_missing(mesh, "multiblock_cap_p", py::cast(defaults.multiblock_cap_p));
  set_default_if_missing(mesh, "multiblock_bridge_elliptic_sweeps",
                         py::cast(defaults.multiblock_bridge_elliptic_sweeps));
  set_default_if_missing(mesh, "multiblock_bridge_elliptic_omega",
                         py::cast(defaults.multiblock_bridge_elliptic_omega));
  set_default_if_missing(
      mesh, "multiblock_outer_svec_tangent_balance",
      py::cast(defaults.multiblock_outer_svec_tangent_balance));
}

void normalize_mesh_default_elision(py::dict& root) {
  py::dict mesh;
  if (!try_get_child_dict(root, "mesh", &mesh)) {
    return;
  }
  const Config::MeshConfig defaults;
  pop_if_equal(mesh, "multiblock_transition_scheme",
               py::str(multiblock_transition_scheme_to_string(
                   defaults.multiblock_transition_scheme)));
  pop_if_equal(mesh, "multiblock_cap_p", py::cast(defaults.multiblock_cap_p));
  pop_if_equal(mesh, "multiblock_bridge_elliptic_sweeps",
               py::cast(defaults.multiblock_bridge_elliptic_sweeps));
  pop_if_equal(mesh, "multiblock_bridge_elliptic_omega",
               py::cast(defaults.multiblock_bridge_elliptic_omega));
  // 2026-07-26 review: grading.mapping is compared by elision —
  // checkpoints that predate the key and current configs at the default value
  // both end up without it. apply_legacy_mesh_defaults only runs for
  // schema<=V1 checkpoints, so per-key completion there cannot equalize
  // modern checkpoints; elision here runs unconditionally on both sides.
  py::dict grading;
  if (try_get_child_dict(mesh, "grading", &grading)) {
    pop_if_equal(grading, "mapping", py::str(defaults.grading.mapping));
  }
  pop_if_equal(mesh, "multiblock_outer_svec_tangent_balance",
               py::cast(defaults.multiblock_outer_svec_tangent_balance));
}

void apply_legacy_laser_defaults(py::dict& root) {
  py::dict laser;
  if (!try_get_child_dict(root, "laser", &laser)) {
    return;
  }

  const Config::LaserConfig defaults;
  set_default_if_missing(laser, "ray_output_trajectory", py::cast(defaults.ray_output_trajectory));
  set_default_if_missing(laser, "ray_output_max_steps", py::cast(defaults.ray_output_max_steps));

  py::dict absorption;
  if (try_get_child_dict(laser, "absorption", &absorption)) {
    const Config::LaserConfig::AbsorptionConfig absorption_defaults;
    set_default_if_missing(absorption, "debug_dump_lasermesh",
                           py::cast(absorption_defaults.debug_dump_lasermesh));
  }

  py::dict lasermesh;
  if (try_get_child_dict(laser, "lasermesh", &lasermesh)) {
    const Config::LaserConfig::LaserMeshConfig lasermesh_defaults;
    set_default_if_missing(lasermesh, "mesh_factor", py::cast(lasermesh_defaults.mesh_factor));
    set_default_if_missing(
        lasermesh, "rmax_n_hat_threshold", py::cast(lasermesh_defaults.rmax_n_hat_threshold));
    set_default_if_missing(lasermesh, "nr_max", py::cast(lasermesh_defaults.nr_max));
  }

  py::dict raytrace;
  if (try_get_child_dict(laser, "raytrace", &raytrace)) {
    const Config::LaserConfig::RaytraceConfig raytrace_defaults;
    set_default_if_missing(
        raytrace, "ds_adapt_g_target", py::cast(raytrace_defaults.ds_adapt_g_target));
    set_default_if_missing(
        raytrace, "ds_adapt_tau_target", py::cast(raytrace_defaults.ds_adapt_tau_target));
    set_default_if_missing(
        raytrace, "ds_adapt_max_factor", py::cast(raytrace_defaults.ds_adapt_max_factor));
    set_default_if_missing(raytrace, "debug_one_ray",
                           py::cast(raytrace_defaults.debug_one_ray));
  }
}

void apply_legacy_radiation_defaults(py::dict& root) {
  py::dict radiation;
  if (!try_get_child_dict(root, "radiation", &radiation)) {
    return;
  }
  const Config::RadiationConfig radiation_defaults;
  set_default_if_missing(radiation, "mode", py::str("multigroup_diffusion"));
  set_default_if_missing(radiation,
                         "origin_parity_only",
                         py::cast(radiation_defaults.origin_parity_only));
  set_default_if_missing(radiation,
                         "group_repack_hard_xray",
                         py::cast(radiation_defaults.group_repack_hard_xray));
  set_default_if_missing(radiation,
                         "diagnose_hard_xray_opacity",
                         py::cast(radiation_defaults.diagnose_hard_xray_opacity));

  py::dict imc;
  if (try_get_child_dict(radiation, "imc", &imc)) {
    const Config::RadiationConfig::ImcConfig imc_defaults;
    set_default_if_missing(imc, "enabled", py::cast(imc_defaults.enabled));
    set_default_if_missing(
        imc, "corrected_fleck", py::cast(imc_defaults.corrected_fleck));
    set_default_if_missing(imc, "source_tilting", py::cast(imc_defaults.source_tilting));
    set_default_if_missing(
        imc, "source_localization", py::cast(imc_defaults.source_localization));
    set_default_if_missing(imc, "sloc_ema_beta", py::cast(imc_defaults.sloc_ema_beta));
    set_default_if_missing(
        imc, "sloc_sigma_floor", py::cast(imc_defaults.sloc_sigma_floor));
    set_default_if_missing(
        imc, "sloc_sigma_cap", py::cast(imc_defaults.sloc_sigma_cap));
    set_default_if_missing(imc, "sloc_tau_ref", py::cast(imc_defaults.sloc_tau_ref));
    set_default_if_missing(
        imc, "spectral_bias_eta", py::cast(imc_defaults.spectral_bias_eta));
    set_default_if_missing(
        imc, "opacity_predictor", py::cast(imc_defaults.opacity_predictor));
    set_default_if_missing(imc, "two_stage", py::cast(imc_defaults.two_stage));
    py::dict difference;
    if (!dict_contains(imc, "difference")) {
      difference = py::dict();
      imc[py::str("difference")] = difference;
    } else if (try_get_child_dict(imc, "difference", &difference)) {
      // Existing child dict reused below.
    }
    if (py::isinstance<py::dict>(imc[py::str("difference")])) {
      difference = imc[py::str("difference")].cast<py::dict>();
      const auto difference_defaults = imc_defaults.difference;
      set_default_if_missing(difference, "enabled",
                             py::cast(difference_defaults.enabled));
      set_default_if_missing(difference, "W_max",
                             py::cast(difference_defaults.W_max));
      set_default_if_missing(difference, "tau0",
                             py::cast(difference_defaults.tau0));
      set_default_if_missing(difference, "chi0",
                             py::cast(difference_defaults.chi0));
      set_default_if_missing(difference, "face_transport",
                             py::cast(difference_defaults.face_transport));
    }
    py::dict net_e_source_smoothing;
    if (!dict_contains(imc, "net_e_source_smoothing")) {
      net_e_source_smoothing = py::dict();
      imc[py::str("net_e_source_smoothing")] = net_e_source_smoothing;
    } else if (try_get_child_dict(imc, "net_e_source_smoothing",
                                  &net_e_source_smoothing)) {
      // Existing child dict reused below.
    }
    const auto smoothing_defaults = imc_defaults.net_e_source_smoothing;
    set_default_if_missing(net_e_source_smoothing, "enabled",
                           py::cast(smoothing_defaults.enabled));
    set_default_if_missing(net_e_source_smoothing, "alpha",
                           py::cast(smoothing_defaults.alpha));
    set_default_if_missing(net_e_source_smoothing, "tau_threshold",
                           py::cast(smoothing_defaults.tau_threshold));
    set_default_if_missing(net_e_source_smoothing, "passes",
                           py::cast(smoothing_defaults.passes));
    set_default_if_missing(net_e_source_smoothing, "grad_Te_scale",
                           py::cast(smoothing_defaults.grad_Te_scale));
    set_default_if_missing(net_e_source_smoothing, "grad_rho_scale",
                           py::cast(smoothing_defaults.grad_rho_scale));
    set_default_if_missing(net_e_source_smoothing, "gradient_adaptive",
                           py::cast(smoothing_defaults.gradient_adaptive));
    set_default_if_missing(imc, "particle_budget", py::cast(imc_defaults.particle_budget));

    py::dict census_comb;
    if (!dict_contains(imc, "census_comb")) {
      census_comb = py::dict();
      imc[py::str("census_comb")] = census_comb;
    } else if (try_get_child_dict(imc, "census_comb", &census_comb)) {
      // Existing dict is updated in-place below.
    }
    if (py::isinstance<py::dict>(imc[py::str("census_comb")])) {
      census_comb = imc[py::str("census_comb")].cast<py::dict>();
      const Config::RadiationConfig::CensusCombConfig census_defaults;
      set_default_if_missing(census_comb, "enabled", py::cast(census_defaults.enabled));
      set_default_if_missing(census_comb, "max_particles", py::cast(census_defaults.max_particles));
      set_default_if_missing(census_comb, "min_per_bin", py::cast(census_defaults.min_per_bin));
      set_default_if_missing(census_comb, "trigger_ratio", py::cast(census_defaults.trigger_ratio));
      set_default_if_missing(
          census_comb, "target_fraction", py::cast(census_defaults.target_fraction));
      set_default_if_missing(
          census_comb, "mode_weight_imc", py::cast(census_defaults.mode_weight_imc));
      set_default_if_missing(
          census_comb, "mode_weight_ddmc", py::cast(census_defaults.mode_weight_ddmc));
      set_default_if_missing(
          census_comb, "adaptive_trigger", py::cast(census_defaults.adaptive_trigger));
      set_default_if_missing(
          census_comb, "adaptive_util_start", py::cast(census_defaults.adaptive_util_start));
      set_default_if_missing(
          census_comb, "adaptive_util_end", py::cast(census_defaults.adaptive_util_end));
      set_default_if_missing(
          census_comb, "trigger_ratio_floor", py::cast(census_defaults.trigger_ratio_floor));
      set_default_if_missing(
          census_comb, "trigger_hysteresis", py::cast(census_defaults.trigger_hysteresis));
      set_default_if_missing(
          census_comb, "ess_floor_enabled", py::cast(census_defaults.ess_floor_enabled));
      set_default_if_missing(
          census_comb, "ess_min_tier0", py::cast(census_defaults.ess_min_tier0));
      set_default_if_missing(
          census_comb, "ess_min_tier1", py::cast(census_defaults.ess_min_tier1));
      set_default_if_missing(
          census_comb, "max_split_factor", py::cast(census_defaults.max_split_factor));
    }

    py::dict rad_lite_mesh;
    if (!dict_contains(imc, "rad_lite_mesh")) {
      rad_lite_mesh = py::dict();
      imc[py::str("rad_lite_mesh")] = rad_lite_mesh;
    } else if (try_get_child_dict(imc, "rad_lite_mesh", &rad_lite_mesh)) {
      // Existing dict is updated in-place below.
    }
    if (py::isinstance<py::dict>(imc[py::str("rad_lite_mesh")])) {
      rad_lite_mesh = imc[py::str("rad_lite_mesh")].cast<py::dict>();
      const Config::RadiationConfig::RadLiteMeshConfig rlm_defaults;
      set_default_if_missing(rad_lite_mesh, "enabled", py::cast(rlm_defaults.enabled));
      set_default_if_missing(
          rad_lite_mesh, "sigma_ratio_max", py::cast(rlm_defaults.sigma_ratio_max));
      set_default_if_missing(rad_lite_mesh, "nlte_auto", py::cast(rlm_defaults.nlte_auto));
    }
  }

  py::dict ddmc;
  if (try_get_child_dict(radiation, "ddmc", &ddmc)) {
    const Config::RadiationConfig::DdmcConfig ddmc_defaults;
    set_default_if_missing(ddmc, "tau_rw", py::cast(ddmc_defaults.tau_rw));
    set_default_if_missing(ddmc, "tau_ddmc_off", py::cast(ddmc_defaults.tau_ddmc_off));
    set_default_if_missing(ddmc, "omega_ddmc_off", py::cast(ddmc_defaults.omega_ddmc_off));
    set_default_if_missing(ddmc, "mode_hold", py::cast(ddmc_defaults.mode_hold));
    set_default_if_missing(ddmc, "rate_max", py::cast(ddmc_defaults.rate_max));
  }

  py::dict diffusion;
  if (!dict_contains(radiation, "diffusion")) {
    diffusion = py::dict();
    radiation[py::str("diffusion")] = diffusion;
  } else if (try_get_child_dict(radiation, "diffusion", &diffusion)) {
    // Existing dict is updated in-place below.
  }
  if (py::isinstance<py::dict>(radiation[py::str("diffusion")])) {
    diffusion = radiation[py::str("diffusion")].cast<py::dict>();
    const Config::RadiationConfig::DiffusionConfig diffusion_defaults;
    set_default_if_missing(diffusion, "enabled", py::cast(diffusion_defaults.enabled));
    set_default_if_missing(diffusion, "tau_on", py::cast(diffusion_defaults.tau_on));
    set_default_if_missing(diffusion, "tau_off", py::cast(diffusion_defaults.tau_off));
    set_default_if_missing(
        diffusion, "reduced_flux_on", py::cast(diffusion_defaults.reduced_flux_on));
    set_default_if_missing(
        diffusion, "reduced_flux_off", py::cast(diffusion_defaults.reduced_flux_off));
    set_default_if_missing(diffusion, "mode_hold", py::cast(diffusion_defaults.mode_hold));
    set_default_if_missing(diffusion, "rate_max", py::cast(diffusion_defaults.rate_max));
    set_default_if_missing(diffusion,
                           "mode_update_interval",
                           py::cast(diffusion_defaults.mode_update_interval));
    set_default_if_missing(diffusion,
                           "min_diffusion_island_cells",
                           py::cast(diffusion_defaults.min_diffusion_island_cells));
    set_default_if_missing(
        diffusion, "imc_guard_cells", py::cast(diffusion_defaults.imc_guard_cells));
    set_default_if_missing(
        diffusion, "sts_max_stages", py::cast(diffusion_defaults.sts_max_stages));
    set_default_if_missing(
        diffusion, "sts_damping", py::cast(diffusion_defaults.sts_damping));
    set_default_if_missing(
        diffusion, "sts_subcycle_eta", py::cast(diffusion_defaults.sts_subcycle_eta));
    set_default_if_missing(diffusion,
                           "interface_particles_per_face_group",
                           py::cast(diffusion_defaults.interface_particles_per_face_group));
    set_default_if_missing(diffusion,
                           "exit_particles_per_cell_group",
                           py::cast(diffusion_defaults.exit_particles_per_cell_group));
    set_default_if_missing(diffusion,
                           "lte_entry_initialization",
                           py::cast(diffusion_defaults.lte_entry_initialization));
    set_default_if_missing(diffusion,
                           "lte_entry_energy_fraction_cap",
                           py::cast(diffusion_defaults.lte_entry_energy_fraction_cap));
  }

  py::dict multigroup_diffusion;
  if (!dict_contains(radiation, "multigroup_diffusion")) {
    multigroup_diffusion = py::dict();
    radiation[py::str("multigroup_diffusion")] = multigroup_diffusion;
  } else if (try_get_child_dict(radiation, "multigroup_diffusion",
                                &multigroup_diffusion)) {
    // Existing child dict reused below.
  }
  if (py::isinstance<py::dict>(radiation[py::str("multigroup_diffusion")])) {
    multigroup_diffusion =
        radiation[py::str("multigroup_diffusion")].cast<py::dict>();
    const auto fld_defaults = radiation_defaults.multigroup_diffusion;
    set_default_if_missing(multigroup_diffusion,
                           "flux_limiter",
                           py::cast(fld_defaults.flux_limiter));
    set_default_if_missing(multigroup_diffusion,
                           "max_outer_iterations",
                           py::cast(fld_defaults.max_outer_iterations));
    set_default_if_missing(multigroup_diffusion,
                           "outer_tol",
                           py::cast(fld_defaults.outer_tol));
    set_default_if_missing(multigroup_diffusion,
                           "state_supply_boundary_policy",
                           py::cast(fld_defaults.state_supply_boundary_policy));
    set_default_if_missing(
        multigroup_diffusion,
        "diagnostic_radial_fourier_substage_enabled",
        py::cast(fld_defaults.diagnostic_radial_fourier_substage_enabled));
    set_default_if_missing(multigroup_diffusion,
                           "cg_inner_tol",
                           py::cast(fld_defaults.cg_inner_tol));
    set_default_if_missing(multigroup_diffusion,
                           "cg_max_iter",
                           py::cast(fld_defaults.cg_max_iter));
    set_default_if_missing(multigroup_diffusion,
                           "cap_exit_policy",
                           py::cast(fld_defaults.cap_exit_policy));
    set_default_if_missing(multigroup_diffusion,
                           "linear_solver_1d",
                           py::cast(fld_defaults.linear_solver_1d));
    set_default_if_missing(multigroup_diffusion,
                           "linear_solver_2d",
                           py::cast(fld_defaults.linear_solver_2d));
    set_default_if_missing(multigroup_diffusion,
                           "rgmg_smoother_omega",
                           py::cast(fld_defaults.rgmg_smoother_omega));
    set_default_if_missing(multigroup_diffusion,
                           "z_boundary",
                           py::cast(fld_defaults.z_boundary));
    py::dict amgx_config;
    if (!dict_contains(multigroup_diffusion, "amgx_config")) {
      amgx_config = py::dict();
      multigroup_diffusion[py::str("amgx_config")] = amgx_config;
    } else if (try_get_child_dict(multigroup_diffusion, "amgx_config", &amgx_config)) {
      // Existing child dict reused below.
    }
    if (py::isinstance<py::dict>(multigroup_diffusion[py::str("amgx_config")])) {
      amgx_config = multigroup_diffusion[py::str("amgx_config")].cast<py::dict>();
      set_default_if_missing(amgx_config,
                             "preset",
                             py::cast(fld_defaults.amgx_config.preset));
    }
    set_default_if_missing(multigroup_diffusion,
                           "opacity_floor",
                           py::cast(fld_defaults.opacity_floor));
    set_default_if_missing(multigroup_diffusion,
                           "opacity_cap",
                           py::cast(fld_defaults.opacity_cap));
    py::dict marshak;
    if (!dict_contains(multigroup_diffusion, "marshak")) {
      marshak = py::dict();
      multigroup_diffusion[py::str("marshak")] = marshak;
    } else if (try_get_child_dict(multigroup_diffusion, "marshak", &marshak)) {
      // Existing child dict reused below.
    }
    if (py::isinstance<py::dict>(multigroup_diffusion[py::str("marshak")])) {
      marshak = multigroup_diffusion[py::str("marshak")].cast<py::dict>();
      set_default_if_missing(
          marshak,
          "flux_erg_per_cm2_s",
          py::cast(fld_defaults.marshak.flux_erg_per_cm2_s));
      set_default_if_missing(
          marshak,
          "flux_pulse_duration_s",
          py::cast(fld_defaults.marshak.flux_pulse_duration_s));
    }
    py::dict boundary;
    if (!dict_contains(multigroup_diffusion, "boundary")) {
      boundary = py::dict();
      multigroup_diffusion[py::str("boundary")] = boundary;
    } else if (try_get_child_dict(multigroup_diffusion, "boundary", &boundary)) {
      // Existing child dict reused below.
    }
    if (py::isinstance<py::dict>(multigroup_diffusion[py::str("boundary")])) {
      boundary = multigroup_diffusion[py::str("boundary")].cast<py::dict>();
      set_default_if_missing(boundary,
                             "inner_r",
                             py::cast(fld_defaults.boundary.inner_r));
      set_default_if_missing(boundary,
                             "outer_r",
                             py::cast(fld_defaults.boundary.outer_r));
      set_default_if_missing(boundary,
                             "z",
                             py::cast(fld_defaults.boundary.z));
      const py::object fld_z_default =
          py::reinterpret_borrow<py::object>(boundary[py::str("z")]);
      set_default_if_missing(boundary,
                             "z_bottom",
                             fld_z_default);
      set_default_if_missing(boundary,
                             "z_top",
                             fld_z_default);
    }
  }

  py::dict sn_transport;
  if (!dict_contains(radiation, "sn_transport")) {
    sn_transport = py::dict();
    radiation[py::str("sn_transport")] = sn_transport;
  } else if (try_get_child_dict(radiation, "sn_transport",
                                &sn_transport)) {
    // Existing child dict reused below.
  }
  if (py::isinstance<py::dict>(radiation[py::str("sn_transport")])) {
    sn_transport = radiation[py::str("sn_transport")].cast<py::dict>();
    const auto sn_defaults = radiation_defaults.sn_transport;
    set_default_if_missing(sn_transport,
                           "n_angles",
                           py::cast(sn_defaults.n_angles));
    set_default_if_missing(sn_transport,
                           "angular_quadrature",
                           py::cast(sn_defaults.angular_quadrature));
    set_default_if_missing(sn_transport,
                           "spatial_scheme",
                           py::cast(sn_defaults.spatial_scheme));
    set_default_if_missing(sn_transport,
                           "max_outer_iterations",
                           py::cast(sn_defaults.max_outer_iterations));
    set_default_if_missing(sn_transport,
                           "max_inner_iterations",
                           py::cast(sn_defaults.max_inner_iterations));
    set_default_if_missing(sn_transport,
                           "outer_tol",
                           py::cast(sn_defaults.outer_tol));
    set_default_if_missing(sn_transport,
                           "outer_tol_stagnation_factor",
                           py::cast(sn_defaults.outer_tol_stagnation_factor));
    set_default_if_missing(sn_transport,
                           "outer_tol_hydro_error_scale",
                           py::cast(sn_defaults.outer_tol_hydro_error_scale));
    set_default_if_missing(sn_transport,
                           "inner_tol",
                           py::cast(sn_defaults.inner_tol));
    set_default_if_missing(sn_transport,
                           "inner_graph_unroll",
                           py::cast(sn_defaults.inner_graph_unroll));
    set_default_if_missing(sn_transport,
                           "dsa_enabled",
                           py::cast(sn_defaults.dsa_enabled));
    set_default_if_missing(sn_transport,
                           "z_boundary",
                           py::cast(sn_defaults.z_boundary));
    set_default_if_missing(sn_transport,
                           "diffusion_fallback_mode",
                           py::cast(sn_defaults.diffusion_fallback_mode));
    set_default_if_missing(sn_transport,
                           "tau_diffusion_on",
                           py::cast(sn_defaults.tau_diffusion_on));
    set_default_if_missing(sn_transport,
                           "tau_diffusion_off",
                           py::cast(sn_defaults.tau_diffusion_off));
    set_default_if_missing(sn_transport,
                           "opacity_floor",
                           py::cast(sn_defaults.opacity_floor));
    set_default_if_missing(sn_transport,
                           "opacity_cap",
                           py::cast(sn_defaults.opacity_cap));
    if (!dict_contains(sn_transport, "marshak")) {
      py::dict marshak;
      marshak["flux_erg_per_cm2_s"] =
          sn_defaults.marshak.flux_erg_per_cm2_s;
      sn_transport[py::str("marshak")] = marshak;
    } else if (py::isinstance<py::dict>(sn_transport[py::str("marshak")])) {
      py::dict marshak = sn_transport[py::str("marshak")].cast<py::dict>();
      set_default_if_missing(marshak,
                             "flux_erg_per_cm2_s",
                             py::cast(sn_defaults.marshak.flux_erg_per_cm2_s));
    }
    set_default_if_missing(sn_transport,
                           "timing_enabled",
                           py::cast(sn_defaults.timing_enabled));
    py::dict boundary;
    if (!dict_contains(sn_transport, "boundary")) {
      boundary = py::dict();
      sn_transport[py::str("boundary")] = boundary;
    } else if (try_get_child_dict(sn_transport, "boundary", &boundary)) {
      // Existing child dict reused below.
    }
    if (py::isinstance<py::dict>(sn_transport[py::str("boundary")])) {
      boundary = sn_transport[py::str("boundary")].cast<py::dict>();
      set_default_if_missing(boundary,
                             "inner_r",
                             py::cast(sn_defaults.boundary.inner_r));
      set_default_if_missing(boundary,
                             "outer_r",
                             py::cast(sn_defaults.boundary.outer_r));
      set_default_if_missing(boundary,
                             "z",
                             py::cast(sn_defaults.boundary.z));
      const py::object z_default = boundary[py::str("z")];
      set_default_if_missing(boundary,
                             "z_bottom",
                             z_default);
      set_default_if_missing(boundary,
                             "z_top",
                             z_default);
    }
  }

  py::dict holo;
  if (!dict_contains(radiation, "holo")) {
    holo = py::dict();
    radiation[py::str("holo")] = holo;
  } else if (try_get_child_dict(radiation, "holo", &holo)) {
    // Existing dict is updated in-place below.
  }
  if (py::isinstance<py::dict>(radiation[py::str("holo")])) {
    holo = radiation[py::str("holo")].cast<py::dict>();
    const Config::RadiationConfig::HoloConfig holo_defaults;
    set_default_if_missing(holo, "enabled", py::cast(holo_defaults.enabled));
    set_default_if_missing(holo, "region", py::cast(holo_defaults.region));
    set_default_if_missing(
        holo, "material_group", py::cast(holo_defaults.material_group));
    set_default_if_missing(holo, "coupling_tau", py::cast(holo_defaults.coupling_tau));
    set_default_if_missing(holo, "guard_cells", py::cast(holo_defaults.guard_cells));
    set_default_if_missing(holo, "blend_cells", py::cast(holo_defaults.blend_cells));
    set_default_if_missing(holo, "min_lo_cells", py::cast(holo_defaults.min_lo_cells));
    set_default_if_missing(holo, "q_min", py::cast(holo_defaults.q_min));
    set_default_if_missing(holo, "q_max", py::cast(holo_defaults.q_max));
    set_default_if_missing(holo, "tau_on", py::cast(holo_defaults.tau_on));
    set_default_if_missing(holo, "tau_off", py::cast(holo_defaults.tau_off));
    set_default_if_missing(
        holo, "reduced_flux_on", py::cast(holo_defaults.reduced_flux_on));
    set_default_if_missing(
        holo, "reduced_flux_off", py::cast(holo_defaults.reduced_flux_off));
    set_default_if_missing(
        holo, "update_interval", py::cast(holo_defaults.update_interval));
    set_default_if_missing(holo, "hold_on", py::cast(holo_defaults.hold_on));
    set_default_if_missing(
        holo, "min_dwell_steps", py::cast(holo_defaults.min_dwell_steps));
    set_default_if_missing(
        holo, "min_island_cells", py::cast(holo_defaults.min_island_cells));
    set_default_if_missing(
        holo, "core_margin_cells", py::cast(holo_defaults.core_margin_cells));
    set_default_if_missing(holo, "solver", py::cast(holo_defaults.solver));
    set_default_if_missing(holo, "closure", py::cast(holo_defaults.closure));
    set_default_if_missing(
        holo, "closure_relax", py::cast(holo_defaults.closure_relax));
    set_default_if_missing(
        holo, "closure_smooth_passes", py::cast(holo_defaults.closure_smooth_passes));
    set_default_if_missing(
        holo, "closure_smooth_alpha", py::cast(holo_defaults.closure_smooth_alpha));
    if (!dict_contains(holo, "consistency_alpha") && dict_contains(holo, "gamma_alpha")) {
      holo[py::str("consistency_alpha")] = holo[py::str("gamma_alpha")];
    }
    set_default_if_missing(
        holo, "consistency_alpha", py::cast(holo_defaults.consistency_alpha));
    set_default_if_missing(
        holo, "gamma_alpha", py::cast(holo_defaults.consistency_alpha));
    set_default_if_missing(
        holo, "boundary_flux", py::cast(holo_defaults.boundary_flux));
    set_default_if_missing(
        holo, "p_rr_tally", py::cast(holo_defaults.p_rr_tally));
    set_default_if_missing(
        holo, "sn_closure", py::cast(holo_defaults.sn_closure));
    set_default_if_missing(
        holo, "sn_n_angles", py::cast(holo_defaults.sn_n_angles));
    set_default_if_missing(
        holo, "sn_material_coupling", py::cast(holo_defaults.sn_material_coupling));
    set_default_if_missing(holo,
                           "residual_particles_per_cell_group",
                           py::cast(holo_defaults.residual_particles_per_cell_group));
  }
}

void apply_legacy_diagnostics_defaults(py::dict& root) {
  py::dict diagnostics;
  if (!try_get_child_dict(root, "diagnostics", &diagnostics)) {
    return;
  }

  py::dict fleck_diag;
  if (!dict_contains(diagnostics, "fleck_diag")) {
    fleck_diag = py::dict();
    diagnostics[py::str("fleck_diag")] = fleck_diag;
  } else if (try_get_child_dict(diagnostics, "fleck_diag", &fleck_diag)) {
    // Existing dict is updated in-place below.
  }
  if (py::isinstance<py::dict>(diagnostics[py::str("fleck_diag")])) {
    fleck_diag = diagnostics[py::str("fleck_diag")].cast<py::dict>();
    const Config::DiagnosticsConfig::FleckDiag fleck_defaults;
    set_default_if_missing(fleck_diag, "enabled", py::cast(fleck_defaults.enabled));
    set_default_if_missing(fleck_diag, "every", py::cast(fleck_defaults.every));
    set_default_if_missing(fleck_diag, "cells", py::cast(fleck_defaults.cells));
    set_default_if_missing(fleck_diag, "r_min_cm", py::cast(fleck_defaults.r_min_cm));
    set_default_if_missing(fleck_diag, "r_max_cm", py::cast(fleck_defaults.r_max_cm));
  }

  const Config::DiagnosticsConfig diagnostics_defaults;
  set_default_if_missing(diagnostics,
                         "per_operator_radial_fourier_enabled",
                         py::cast(diagnostics_defaults
                                      .per_operator_radial_fourier_enabled));
  set_default_if_missing(diagnostics,
                         "radial_fourier_window_t_start_s",
                         py::cast(diagnostics_defaults
                                      .radial_fourier_window_t_start_s));
  set_default_if_missing(diagnostics,
                         "radial_fourier_window_t_end_s",
                         py::cast(diagnostics_defaults
                                      .radial_fourier_window_t_end_s));
  set_default_if_missing(diagnostics,
                         "radial_fourier_max_mode",
                         py::cast(diagnostics_defaults.radial_fourier_max_mode));
  set_default_if_missing(diagnostics,
                         "per_operator_radial_fourier_complex_enabled",
                         py::cast(diagnostics_defaults
                                      .per_operator_radial_fourier_complex_enabled));
  set_default_if_missing(
      diagnostics,
      "per_operator_radial_fourier_complex_m_targets",
      py::cast(diagnostics_defaults
                   .per_operator_radial_fourier_complex_m_targets));
  set_default_if_missing(
      diagnostics,
      "per_operator_radial_fourier_complex_j_targets",
      py::cast(diagnostics_defaults
                   .per_operator_radial_fourier_complex_j_targets));
  set_default_if_missing(diagnostics,
                         "per_operator_radial_fourier_complex_fields",
                         py::cast(diagnostics_defaults
                                      .per_operator_radial_fourier_complex_fields));
}

void apply_legacy_numerics_defaults(py::dict& root) {
  py::dict numerics;
  if (!try_get_child_dict(root, "numerics", &numerics)) {
    return;
  }

  py::dict dt;
  if (!dict_contains(numerics, "dt")) {
    dt = py::dict();
    numerics[py::str("dt")] = dt;
  } else if (try_get_child_dict(numerics, "dt", &dt)) {
    // Existing dict is updated below.
  }
  if (py::isinstance<py::dict>(numerics[py::str("dt")])) {
    dt = numerics[py::str("dt")].cast<py::dict>();
    const Config::NumericsConfig::DtConfig dt_defaults;
    set_default_if_missing(
        dt,
        "floor_stall_max_consecutive_steps",
        py::cast(dt_defaults.floor_stall_max_consecutive_steps));
  }

  py::dict persistent_loop;
  if (!dict_contains(numerics, "persistent_loop")) {
    persistent_loop = py::dict();
    numerics[py::str("persistent_loop")] = persistent_loop;
  } else if (try_get_child_dict(numerics, "persistent_loop", &persistent_loop)) {
    // Existing dict is updated below.
  }
  if (py::isinstance<py::dict>(numerics[py::str("persistent_loop")])) {
    persistent_loop = numerics[py::str("persistent_loop")].cast<py::dict>();
    const Config::NumericsConfig::PersistentLoopConfig persistent_loop_defaults;
    set_default_if_missing(
        persistent_loop, "enabled", py::cast(persistent_loop_defaults.enabled));
    set_default_if_missing(
        persistent_loop,
        "chunk_steps",
        py::cast(persistent_loop_defaults.chunk_steps));
  }

  py::dict ale;
  if (!dict_contains(numerics, "ale")) {
    ale = py::dict();
    numerics[py::str("ale")] = ale;
  } else if (try_get_child_dict(numerics, "ale", &ale)) {
    // Existing dict is updated below.
  }
  if (py::isinstance<py::dict>(numerics[py::str("ale")])) {
    ale = numerics[py::str("ale")].cast<py::dict>();
    const Config::NumericsConfig::AleConfig ale_defaults;
    py::dict align_diagnostics;
    if (!dict_contains(ale, "align_diagnostics")) {
      align_diagnostics = py::dict();
      ale[py::str("align_diagnostics")] = align_diagnostics;
    } else if (try_get_child_dict(
                   ale, "align_diagnostics", &align_diagnostics)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(ale[py::str("align_diagnostics")])) {
      align_diagnostics =
          ale[py::str("align_diagnostics")].cast<py::dict>();
      const auto& defaults = ale_defaults.align_diagnostics;
      set_default_if_missing(align_diagnostics, "enabled",
                             py::cast(defaults.enabled));
      set_default_if_missing(align_diagnostics, "every_n_steps",
                             py::cast(defaults.every_n_steps));
      set_default_if_missing(align_diagnostics, "c_q_threshold",
                             py::cast(defaults.c_q_threshold));
      set_default_if_missing(align_diagnostics, "w_rho",
                             py::cast(defaults.w_rho));
      set_default_if_missing(align_diagnostics, "w_p",
                             py::cast(defaults.w_p));
      set_default_if_missing(align_diagnostics, "floor_rel",
                             py::cast(defaults.floor_rel));
    }
    if (dict_contains(ale, "donor_sign_fixed") &&
        !dict_contains(ale, "swept_volume_sign_fixed")) {
      ale[py::str("swept_volume_sign_fixed")] = ale[py::str("donor_sign_fixed")];
    }
    set_default_if_missing(
        ale,
        "swept_volume_sign_fixed",
        py::cast(ale_defaults.swept_volume_sign_fixed));
    if (dict_contains(ale, "donor_sign_fixed")) {
      ale.attr("pop")(py::str("donor_sign_fixed"));
    }
    set_default_if_missing(ale,
                           "ke_conservation_closure",
                           py::cast(ale_defaults.ke_conservation_closure));
    set_default_if_missing(ale,
                           "ke_conservation_closure_audit",
                           py::cast(ale_defaults.ke_conservation_closure_audit));
    set_default_if_missing(ale,
                           "ke_closure_redistribute_floor",
                           py::cast(ale_defaults.ke_closure_redistribute_floor));
    set_default_if_missing(ale,
                           "debug_per_remap_log",
                           py::cast(ale_defaults.debug_per_remap_log));
    set_default_if_missing(ale,
                           "reference_barrier_enabled",
                           py::cast(ale_defaults.reference_barrier_enabled));
    set_default_if_missing(ale,
                           "reference_target",
                           py::cast(ale_defaults.reference_target));
    set_default_if_missing(ale,
                           "reference_blend_default",
                           py::cast(ale_defaults.reference_blend_default));
    set_default_if_missing(ale,
                           "reference_volume_floor_rel",
                           py::cast(ale_defaults.reference_volume_floor_rel));
    set_default_if_missing(ale,
                           "reference_corner_j_floor_rel",
                           py::cast(ale_defaults.reference_corner_j_floor_rel));
    set_default_if_missing(ale,
                           "reference_gauss_j_floor_rel",
                           py::cast(ale_defaults.reference_gauss_j_floor_rel));
    set_default_if_missing(ale,
                           "reference_linesearch_max_iters",
                           py::cast(ale_defaults.reference_linesearch_max_iters));
    set_default_if_missing(
        ale,
        "reference_force_engage_every_step",
        py::cast(ale_defaults.reference_force_engage_every_step));
    set_default_if_missing(
        ale,
        "force_rezone_every_n_steps",
        py::cast(ale_defaults.force_rezone_every_n_steps));
    set_default_if_missing(
        ale,
        "reference_trigger_axis_margin_enabled",
        py::cast(ale_defaults.reference_trigger_axis_margin_enabled));
    set_default_if_missing(
        ale,
        "reference_trigger_axis_margin_threshold",
        py::cast(ale_defaults.reference_trigger_axis_margin_threshold));
    set_default_if_missing(
        ale,
        "reference_trigger_corner_j_ratio_enabled",
        py::cast(ale_defaults.reference_trigger_corner_j_ratio_enabled));
    set_default_if_missing(
        ale,
        "reference_trigger_corner_j_ratio_threshold",
        py::cast(ale_defaults.reference_trigger_corner_j_ratio_threshold));
    set_default_if_missing(ale,
                           "dgcl_commit_gate",
                           py::cast(ale_defaults.dgcl_commit_gate));
    set_default_if_missing(
        ale,
        "transaction_failure_inject_point",
        py::cast(ale_defaults.transaction_failure_inject_point));
    set_default_if_missing(ale,
                           "dgcl_commit_rtol",
                           py::cast(ale_defaults.dgcl_commit_rtol));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_enabled",
        py::cast(ale_defaults.driver_retry_reference_barrier_enabled));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_K_axis",
        py::cast(ale_defaults.driver_retry_reference_barrier_K_axis));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_eta_axis",
        py::cast(ale_defaults.driver_retry_reference_barrier_eta_axis));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_max_attempts",
        py::cast(ale_defaults.driver_retry_reference_barrier_max_attempts));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_same_sig_max",
        py::cast(ale_defaults.driver_retry_reference_barrier_same_sig_max));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_cell_window",
        py::cast(ale_defaults.driver_retry_reference_barrier_cell_window));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_dt_collapse_rel",
        py::cast(ale_defaults.driver_retry_reference_barrier_dt_collapse_rel));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_lambda_collapse_threshold",
        py::cast(
            ale_defaults
                .driver_retry_reference_barrier_lambda_collapse_threshold));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_lambda_collapse_count",
        py::cast(ale_defaults
                     .driver_retry_reference_barrier_lambda_collapse_count));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_quality_progress_factor",
        py::cast(ale_defaults
                     .driver_retry_reference_barrier_quality_progress_factor));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_quality_progress_count",
        py::cast(ale_defaults
                     .driver_retry_reference_barrier_quality_progress_count));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_rezone_freq_warn_fraction",
        py::cast(ale_defaults
                     .driver_retry_reference_barrier_rezone_freq_warn_fraction));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_rezone_freq_window",
        py::cast(ale_defaults
                     .driver_retry_reference_barrier_rezone_freq_window));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_chi",
        py::cast(ale_defaults.driver_retry_reference_barrier_chi));
    set_default_if_missing(
        ale,
        "driver_retry_reference_barrier_q_retry",
        py::cast(ale_defaults.driver_retry_reference_barrier_q_retry));
    set_default_if_missing(ale,
                           "predictive_acceptance_enabled",
                           py::cast(ale_defaults.predictive_acceptance_enabled));
    set_default_if_missing(
        ale,
        "predictive_acceptance_axis_floor_fraction",
        py::cast(ale_defaults.predictive_acceptance_axis_floor_fraction));
    set_default_if_missing(
        ale,
        "predictive_acceptance_cell_vol_floor_fraction",
        py::cast(ale_defaults.predictive_acceptance_cell_vol_floor_fraction));
    set_default_if_missing(ale,
                           "safe_backtrack_enabled",
                           py::cast(ale_defaults.safe_backtrack_enabled));
    set_default_if_missing(ale,
                           "safe_backtrack_min_exp",
                           py::cast(ale_defaults.safe_backtrack_min_exp));
    set_default_if_missing(ale,
                           "safe_backtrack_binary_iters",
                           py::cast(ale_defaults.safe_backtrack_binary_iters));
    set_default_if_missing(
        ale,
        "corner_cell_aspect_protection_enabled",
        py::cast(ale_defaults.corner_cell_aspect_protection_enabled));
    set_default_if_missing(ale,
                           "corner_cell_aspect_eta",
                           py::cast(ale_defaults.corner_cell_aspect_eta));
    set_default_if_missing(ale,
                           "rezone_solver",
                           py::cast(ale_defaults.rezone_solver));
    set_default_if_missing(ale,
                           "m1_gamma_align",
                           py::cast(ale_defaults.m1_gamma_align));
    set_default_if_missing(ale,
                           "m1_lambda_tether",
                           py::cast(ale_defaults.m1_lambda_tether));
    set_default_if_missing(ale,
                           "m1_theta_reg",
                           py::cast(ale_defaults.m1_theta_reg));
    set_default_if_missing(ale,
                           "m1_sweeps",
                           py::cast(ale_defaults.m1_sweeps));
    set_default_if_missing(ale,
                           "m1_min_j_dec_rel",
                           py::cast(ale_defaults.m1_min_j_dec_rel));
    set_default_if_missing(ale,
                           "m1_barrier_beta",
                           py::cast(ale_defaults.m1_barrier_beta));
    py::dict euler_window_defaults;
    euler_window_defaults["enabled"] = ale_defaults.euler_window.enabled;
    euler_window_defaults["shape"] = ale_defaults.euler_window.shape;
    euler_window_defaults["r0"] = ale_defaults.euler_window.r0;
    euler_window_defaults["r1"] = ale_defaults.euler_window.r1;
    euler_window_defaults["z0"] = ale_defaults.euler_window.z0;
    euler_window_defaults["z1"] = ale_defaults.euler_window.z1;
    euler_window_defaults["cr"] = ale_defaults.euler_window.cr;
    euler_window_defaults["cz"] = ale_defaults.euler_window.cz;
    euler_window_defaults["rad_in"] = ale_defaults.euler_window.rad_in;
    euler_window_defaults["rad_out"] = ale_defaults.euler_window.rad_out;
    euler_window_defaults["transition_width"] =
        ale_defaults.euler_window.transition_width;
    set_default_if_missing(
        ale, "euler_window", euler_window_defaults);
    if (py::isinstance<py::dict>(ale[py::str("euler_window")])) {
      py::dict euler_window =
          ale[py::str("euler_window")].cast<py::dict>();
      set_default_if_missing(
          euler_window,
          "enabled",
          py::cast(ale_defaults.euler_window.enabled));
      set_default_if_missing(
          euler_window,
          "shape",
          py::cast(ale_defaults.euler_window.shape));
      set_default_if_missing(
          euler_window, "r0", py::cast(ale_defaults.euler_window.r0));
      set_default_if_missing(
          euler_window, "r1", py::cast(ale_defaults.euler_window.r1));
      set_default_if_missing(
          euler_window, "z0", py::cast(ale_defaults.euler_window.z0));
      set_default_if_missing(
          euler_window, "z1", py::cast(ale_defaults.euler_window.z1));
      set_default_if_missing(
          euler_window, "cr", py::cast(ale_defaults.euler_window.cr));
      set_default_if_missing(
          euler_window, "cz", py::cast(ale_defaults.euler_window.cz));
      set_default_if_missing(
          euler_window,
          "rad_in",
          py::cast(ale_defaults.euler_window.rad_in));
      set_default_if_missing(
          euler_window,
          "rad_out",
          py::cast(ale_defaults.euler_window.rad_out));
      set_default_if_missing(
          euler_window,
          "transition_width",
          py::cast(ale_defaults.euler_window.transition_width));
    }
    set_default_if_missing(
        ale,
        "rezone_local_admissibility_linesearch",
        py::cast(ale_defaults.rezone_local_admissibility_linesearch));
    set_default_if_missing(ale,
                           "rezone_local_j_floor_rel",
                           py::cast(ale_defaults.rezone_local_j_floor_rel));
    set_default_if_missing(
        ale,
        "rezone_local_linesearch_max_halves",
        py::cast(ale_defaults.rezone_local_linesearch_max_halves));
    set_default_if_missing(ale,
                           "reject_zero_gauss_j",
                           py::cast(ale_defaults.reject_zero_gauss_j));
    set_default_if_missing(ale,
                           "zero_gauss_j_floor_rel",
                           py::cast(ale_defaults.zero_gauss_j_floor_rel));
    set_default_if_missing(
        ale,
        "conservative_remap_order",
        py::cast(ale_defaults.conservative_remap_order));
    set_default_if_missing(
        ale,
        "tri_fan_tracking_reference_mode",
        py::cast(ale_defaults.tri_fan_tracking_reference_mode));
    set_default_if_missing(
        ale,
        "tri_fan_tracking_reference_beta",
        py::cast(ale_defaults.tri_fan_tracking_reference_beta));
    set_default_if_missing(
        ale,
        "tri_fan_tracking_reference_g0",
        py::cast(ale_defaults.tri_fan_tracking_reference_g0));
    set_default_if_missing(
        ale,
        "tri_fan_tracking_reference_eps_v",
        py::cast(ale_defaults.tri_fan_tracking_reference_eps_v));
    set_default_if_missing(
        ale,
        "conservative_remap_lagrangian_bulk_enabled",
        py::cast(ale_defaults.conservative_remap_lagrangian_bulk_enabled));
    set_default_if_missing(
        ale,
        "conservative_remap_lagrangian_bulk_center_node_ring_max",
        py::cast(
            ale_defaults.conservative_remap_lagrangian_bulk_center_node_ring_max));
    set_default_if_missing(
        ale,
        "central_pseudo_core_enabled",
        py::cast(ale_defaults.central_pseudo_core_enabled));
    set_default_if_missing(
        ale,
        "central_pseudo_core_s_c",
        py::cast(ale_defaults.central_pseudo_core_s_c));
    set_default_if_missing(ale,
                           "lambda_sweep_diagnostic_enabled",
                           py::cast(ale_defaults.lambda_sweep_diagnostic_enabled));
    set_default_if_missing(ale,
                           "lambda_sweep_target_cell_c",
                           py::cast(ale_defaults.lambda_sweep_target_cell_c));
    set_default_if_missing(ale,
                           "lambda_sweep_target_cell_i",
                           py::cast(ale_defaults.lambda_sweep_target_cell_i));
    set_default_if_missing(ale,
                           "lambda_sweep_target_cell_j",
                           py::cast(ale_defaults.lambda_sweep_target_cell_j));
    set_default_if_missing(ale,
                           "lambda_sweep_max_exp",
                           py::cast(ale_defaults.lambda_sweep_max_exp));
    set_default_if_missing(ale,
                           "corner_jacobian_post_tangle_enabled",
                           py::cast(ale_defaults.corner_jacobian_post_tangle_enabled));
    set_default_if_missing(
        ale,
        "corner_post_tangle_strict_floor_enabled",
        py::cast(ale_defaults.corner_post_tangle_strict_floor_enabled));
    set_default_if_missing(ale,
                           "local_boundary_repair_enabled",
                           py::cast(ale_defaults.local_boundary_repair_enabled));
    set_default_if_missing(
        ale,
        "multi_node_boundary_repair_enabled",
        py::cast(ale_defaults.multi_node_boundary_repair_enabled));
    set_default_if_missing(
        ale,
        "multi_node_interior_repair_enabled",
        py::cast(ale_defaults.multi_node_interior_repair_enabled));
    set_default_if_missing(
        ale,
        "axis_variational_projection_enabled",
        py::cast(ale_defaults.axis_variational_projection_enabled));
    set_default_if_missing(
        ale,
        "emergency_cell_deactivation_enabled",
        py::cast(ale_defaults.emergency_cell_deactivation_enabled));
    set_default_if_missing(
        ale,
        "multiblock_cross_seam_rezone_enabled",
        py::cast(ale_defaults.multiblock_cross_seam_rezone_enabled));
    set_default_if_missing(
        ale,
        "multiblock_scaled_reference_enabled",
        py::cast(ale_defaults.multiblock_scaled_reference_enabled));
    set_default_if_missing(
        ale,
        "multiblock_differential_reference_enabled",
        py::cast(ale_defaults.multiblock_differential_reference_enabled));
    set_default_if_missing(
        ale,
        "multiblock_differential_reference_band_count",
        py::cast(ale_defaults.multiblock_differential_reference_band_count));
    set_default_if_missing(
        ale,
        "multiblock_differential_reference_smoothing_g0",
        py::cast(ale_defaults.multiblock_differential_reference_smoothing_g0));
    set_default_if_missing(
        ale,
        "multiblock_differential_reference_nu",
        py::cast(ale_defaults.multiblock_differential_reference_nu));
    set_default_if_missing(
        ale,
        "multiblock_differential_reference_eps_v",
        py::cast(ale_defaults.multiblock_differential_reference_eps_v));
    set_default_if_missing(
        ale,
        "multiblock_differential_reference_s_cap_min_rel",
        py::cast(ale_defaults.multiblock_differential_reference_s_cap_min_rel));
    set_default_if_missing(
        ale,
        "multiblock_differential_reference_xi_seam_tol",
        py::cast(ale_defaults.multiblock_differential_reference_xi_seam_tol));
    set_default_if_missing(
        ale,
        "multiblock_differential_reference_sigma_warn_floor",
        py::cast(ale_defaults.multiblock_differential_reference_sigma_warn_floor));
    set_default_if_missing(
        ale,
        "multiblock_lagrangian_bulk_center_patch_reference_enabled",
        py::cast(
            ale_defaults.multiblock_lagrangian_bulk_center_patch_reference_enabled));
    set_default_if_missing(
        ale,
        "multiblock_center_patch_ring_max",
        py::cast(ale_defaults.multiblock_center_patch_ring_max));
    set_default_if_missing(
        ale,
        "multiblock_center_patch_xi_center",
        py::cast(ale_defaults.multiblock_center_patch_xi_center));
    set_default_if_missing(
        ale,
        "multiblock_center_patch_halo_layers",
        py::cast(ale_defaults.multiblock_center_patch_halo_layers));
    set_default_if_missing(
        ale,
        "multiblock_center_patch_vol_on",
        py::cast(ale_defaults.multiblock_center_patch_vol_on));
    set_default_if_missing(
        ale,
        "multiblock_center_patch_vol_off",
        py::cast(ale_defaults.multiblock_center_patch_vol_off));
    set_default_if_missing(
        ale,
        "multiblock_center_patch_cornerj_on",
        py::cast(ale_defaults.multiblock_center_patch_cornerj_on));
    set_default_if_missing(
        ale,
        "multiblock_center_patch_cornerj_off",
        py::cast(ale_defaults.multiblock_center_patch_cornerj_off));
    set_default_if_missing(
        ale,
        "multiblock_center_patch_gaussj_on",
        py::cast(ale_defaults.multiblock_center_patch_gaussj_on));
    set_default_if_missing(
        ale,
        "multiblock_center_patch_gaussj_off",
        py::cast(ale_defaults.multiblock_center_patch_gaussj_off));
    set_default_if_missing(
        ale,
        "ale_reference_diagnostics_enabled",
        py::cast(ale_defaults.ale_reference_diagnostics_enabled));
    set_default_if_missing(
        ale,
        "multiblock_path_admissibility_enabled",
        py::cast(ale_defaults.multiblock_path_admissibility_enabled));
    set_default_if_missing(ale,
                           "path_admissibility_floor",
                           py::cast(ale_defaults.path_admissibility_floor));
    set_default_if_missing(ale,
                           "dt_rejection_factor",
                           py::cast(ale_defaults.dt_rejection_factor));
    set_default_if_missing(ale,
                           "max_dt_rejections",
                           py::cast(ale_defaults.max_dt_rejections));
    set_default_if_missing(
        ale,
        "axis_band_managed_remap_enabled",
        py::cast(ale_defaults.axis_band_managed_remap_enabled));
    set_default_if_missing(ale,
                           "axis_band_managed_remap_width",
                           py::cast(ale_defaults.axis_band_managed_remap_width));
    set_default_if_missing(
        ale,
        "axis_band_managed_remap_max_width",
        py::cast(ale_defaults.axis_band_managed_remap_max_width));
    set_default_if_missing(
        ale,
        "axis_band_managed_remap_every_hydro_half_step",
        py::cast(ale_defaults.axis_band_managed_remap_every_hydro_half_step));
    set_default_if_missing(
        ale,
        "axis_band_managed_remap_margin_trigger",
        py::cast(ale_defaults.axis_band_managed_remap_margin_trigger));
    set_default_if_missing(
        ale,
        "axis_band_managed_remap_equal_volume",
        py::cast(ale_defaults.axis_band_managed_remap_equal_volume));
    set_default_if_missing(
        ale,
        "axis_band_managed_remap_include_radiation_groups",
        py::cast(ale_defaults.axis_band_managed_remap_include_radiation_groups));
    set_default_if_missing(ale,
                           "axis_rezone_enabled",
                           py::cast(ale_defaults.axis_rezone_enabled));
    set_default_if_missing(
        ale,
        "axis_rezone_trigger_edge_fraction",
        py::cast(ale_defaults.axis_rezone_trigger_edge_fraction));
    set_default_if_missing(
        ale,
        "axis_rezone_trigger_min_altitude_fraction",
        py::cast(ale_defaults.axis_rezone_trigger_min_altitude_fraction));
    set_default_if_missing(ale,
                           "axis_rezone_eta_floor",
                           py::cast(ale_defaults.axis_rezone_eta_floor));
  }

  py::dict plic;
  if (!dict_contains(numerics, "plic")) {
    plic = py::dict();
    numerics[py::str("plic")] = plic;
  } else if (try_get_child_dict(numerics, "plic", &plic)) {
    // Existing dict is updated below.
  }
  if (py::isinstance<py::dict>(numerics[py::str("plic")])) {
    plic = numerics[py::str("plic")].cast<py::dict>();
    const Config::NumericsConfig::PlicConfig plic_defaults;
    set_default_if_missing(plic, "enabled", py::cast(plic_defaults.enabled));
    set_default_if_missing(
        plic, "normal_estimator", py::cast(plic_defaults.normal_estimator));
    set_default_if_missing(
        plic, "t0_volume_cut_method", py::cast(plic_defaults.t0_volume_cut_method));
    set_default_if_missing(
        plic, "t0_volume_cut_max_depth", py::cast(plic_defaults.t0_volume_cut_max_depth));
    set_default_if_missing(plic,
                           "t0_volume_cut_volfrac_tol",
                           py::cast(plic_defaults.t0_volume_cut_volfrac_tol));
    set_default_if_missing(plic,
                           "fast_path_threshold_min",
                           py::cast(plic_defaults.fast_path_threshold_min));
    set_default_if_missing(plic,
                           "fast_path_threshold_max",
                           py::cast(plic_defaults.fast_path_threshold_max));
    set_default_if_missing(plic,
                           "fast_path_halo_radius_cells",
                           py::cast(plic_defaults.fast_path_halo_radius_cells));
    set_default_if_missing(
        plic, "alpha_solver_max_iter", py::cast(plic_defaults.alpha_solver_max_iter));
    set_default_if_missing(
        plic, "alpha_tolerance_rel", py::cast(plic_defaults.alpha_tolerance_rel));
    set_default_if_missing(plic,
                           "thermodynamic_error_soft_threshold",
                           py::cast(plic_defaults.thermodynamic_error_soft_threshold));
    set_default_if_missing(plic,
                           "thermodynamic_error_hard_threshold",
                           py::cast(plic_defaults.thermodynamic_error_hard_threshold));
    set_default_if_missing(plic,
                           "class_d_dense_fraction_threshold",
                           py::cast(plic_defaults.class_d_dense_fraction_threshold));
    set_default_if_missing(
        plic,
        "material_interface_per_cell_state",
        py::cast(plic_defaults.material_interface_per_cell_state));
    set_default_if_missing(plic,
                           "production_comparable_gate_strict",
                           py::cast(plic_defaults.production_comparable_gate_strict));
    set_default_if_missing(plic,
                           "drift_sensor_max_relative",
                           py::cast(plic_defaults.drift_sensor_max_relative));
    set_default_if_missing(plic,
                           "drift_sensor_max_swept_fraction",
                           py::cast(plic_defaults.drift_sensor_max_swept_fraction));
    set_default_if_missing(
        plic,
        "prev_normal_freshness_volfrac_threshold",
        py::cast(plic_defaults.prev_normal_freshness_volfrac_threshold));
    set_default_if_missing(plic,
                           "plic_per_step_cost_target_fraction",
                           py::cast(plic_defaults.plic_per_step_cost_target_fraction));
    set_default_if_missing(
        plic, "in_run_disabled", py::cast(plic_defaults.in_run_disabled));
    set_default_if_missing(plic,
                           "rho_material_aware_donor",
                           py::cast(plic_defaults.rho_material_aware_donor));
  }

  py::dict materials;
  if (!dict_contains(numerics, "materials")) {
    materials = py::dict();
    numerics[py::str("materials")] = materials;
  } else if (try_get_child_dict(numerics, "materials", &materials)) {
    // Existing dict is updated below.
  }
  if (py::isinstance<py::dict>(numerics[py::str("materials")])) {
    materials = numerics[py::str("materials")].cast<py::dict>();
    const Config::NumericsConfig::MaterialsSubConfig materials_defaults;
    set_default_if_missing(
        materials,
        "per_material_conservation_enabled",
        py::cast(materials_defaults.per_material_conservation_enabled));
    set_default_if_missing(materials,
                           "presence_threshold_volfrac",
                           py::cast(materials_defaults.presence_threshold_volfrac));
    set_default_if_missing(
        materials,
        "presence_threshold_mass_density_g_per_cc",
        py::cast(materials_defaults.presence_threshold_mass_density_g_per_cc));
    py::dict eos_bounds;
    if (!dict_contains(materials, "eos_table_validity_lower_bound_g_per_cc")) {
      eos_bounds = py::dict();
      materials[py::str("eos_table_validity_lower_bound_g_per_cc")] = eos_bounds;
    } else if (try_get_child_dict(materials,
                                  "eos_table_validity_lower_bound_g_per_cc",
                                  &eos_bounds)) {
      // Existing dict is preserved.
    }
    set_default_if_missing(
        materials,
        "lazy_cache_te_m_enabled",
        py::cast(materials_defaults.lazy_cache_te_m_enabled));
    set_default_if_missing(
        materials,
        "hdf5_emit_derived_per_material",
        py::cast(materials_defaults.hdf5_emit_derived_per_material));
    set_default_if_missing(
        materials,
        "deposit_redistribute_fallback_enabled",
        py::cast(materials_defaults.deposit_redistribute_fallback_enabled));
    set_default_if_missing(
        materials,
        "deposit_redistribute_provenance_label",
        py::cast(materials_defaults.deposit_redistribute_provenance_label));
    set_default_if_missing(
        materials,
        "conservation_residual_warn_threshold_rel",
        py::cast(materials_defaults.conservation_residual_warn_threshold_rel));
    set_default_if_missing(
        materials,
        "conservation_residual_hard_warning_threshold_rel",
        py::cast(materials_defaults.conservation_residual_hard_warning_threshold_rel));
  }

  py::dict hydro;
  if (!dict_contains(numerics, "hydro")) {
    hydro = py::dict();
    numerics[py::str("hydro")] = hydro;
  } else if (try_get_child_dict(numerics, "hydro", &hydro)) {
    // Existing dict is updated below.
  }
  if (py::isinstance<py::dict>(numerics[py::str("hydro")])) {
    hydro = numerics[py::str("hydro")].cast<py::dict>();
    const Config::NumericsConfig::HydroConfig hydro_defaults;
    // 2026-07-26 review: legacy frozen-config
    // default completion for the new 1D keys — no schema bump needed.
    set_default_if_missing(hydro, "crossing_dt_safety",
                           py::cast(hydro_defaults.crossing_dt_safety));
    set_default_if_missing(hydro, "time_integrator",
                           py::str(hydro_defaults.time_integrator));
    if (dict_contains(hydro, "adaptive_av") &&
        py::isinstance<py::dict>(hydro[py::str("adaptive_av")])) {
      py::dict adaptive_av_existing =
          hydro[py::str("adaptive_av")].cast<py::dict>();
      set_default_if_missing(
          adaptive_av_existing, "hysteresis_tau",
          py::cast(hydro_defaults.adaptive_av.hysteresis_tau));
    }
    py::dict hourglass;
    if (!dict_contains(hydro, "hourglass")) {
      hourglass = py::dict();
      hydro[py::str("hourglass")] = hourglass;
    } else if (try_get_child_dict(hydro, "hourglass", &hourglass)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(hydro[py::str("hourglass")])) {
      hourglass = hydro[py::str("hourglass")].cast<py::dict>();
      set_default_if_missing(
          hourglass,
          "enabled",
          py::cast(hydro_defaults.hourglass.enabled));
      set_default_if_missing(
          hourglass,
          "scale",
          py::cast(hydro_defaults.hourglass.scale));
      set_default_if_missing(
          hourglass,
          "compatible_work_enabled",
          py::cast(hydro_defaults.hourglass.compatible_work_enabled));
      set_default_if_missing(
          hourglass,
          "activation_corner_j_ratio_threshold",
          py::cast(hydro_defaults.hourglass.activation_corner_j_ratio_threshold));
      set_default_if_missing(
          hourglass,
          "activation_hourglass_amplitude_threshold",
          py::cast(hydro_defaults.hourglass.activation_hourglass_amplitude_threshold));
      set_default_if_missing(
          hourglass,
          "subzonal_pressure_model",
          py::cast(hydro_defaults.hourglass.subzonal_pressure_model));
      set_default_if_missing(
          hourglass,
          "max_force_per_node_fraction",
          py::cast(hydro_defaults.hourglass.max_force_per_node_fraction));
    }
    set_default_if_missing(
        hydro,
        "volume_rate_cfl_enabled",
        py::cast(hydro_defaults.volume_rate_cfl_enabled));
    set_default_if_missing(
        hydro,
        "volume_rate_cfl_threshold",
        py::cast(hydro_defaults.volume_rate_cfl_threshold));
    set_default_if_missing(
        hydro,
        "tri_fan_center_cfl_enabled",
        py::cast(hydro_defaults.tri_fan_center_cfl_enabled));
    set_default_if_missing(
        hydro,
        "tri_fan_center_cfl_safety",
        py::cast(hydro_defaults.tri_fan_center_cfl_safety));
    set_default_if_missing(
        hydro,
        "tri_fan_center_cfl_band_radial_index",
        py::cast(hydro_defaults.tri_fan_center_cfl_band_radial_index));
    set_default_if_missing(
        hydro,
        "corner_j_predict_cfl_enabled",
        py::cast(hydro_defaults.corner_j_predict_cfl_enabled));
    set_default_if_missing(
        hydro,
        "corner_j_predict_cfl_safety",
        py::cast(hydro_defaults.corner_j_predict_cfl_safety));
    set_default_if_missing(
        hydro,
        "corner_j_predict_floor_frac",
        py::cast(hydro_defaults.corner_j_predict_floor_frac));
    set_default_if_missing(
        hydro,
        "corner_j_predict_max_shrink",
        py::cast(hydro_defaults.corner_j_predict_max_shrink));
    set_default_if_missing(
        hydro,
        "corner_j_predict_shell_rings",
        py::cast(hydro_defaults.corner_j_predict_shell_rings));
    set_default_if_missing(
        hydro,
        "tri_fan_center_perturbation_diag_enabled",
        py::cast(hydro_defaults.tri_fan_center_perturbation_diag_enabled));
    set_default_if_missing(
        hydro,
        "rz_geometric_cfl_enabled",
        py::cast(hydro_defaults.rz_geometric_cfl_enabled));
    set_default_if_missing(
        hydro,
        "rz_geometric_cfl_etaV",
        py::cast(hydro_defaults.rz_geometric_cfl_etaV));
    set_default_if_missing(
        hydro,
        "rz_geometric_cfl_r_floor",
        py::cast(hydro_defaults.rz_geometric_cfl_r_floor));
    set_default_if_missing(
        hydro,
        "rz_geometric_cfl_cumulative_protection_enabled",
        py::cast(hydro_defaults.rz_geometric_cfl_cumulative_protection_enabled));
    set_default_if_missing(
        hydro,
        "rz_geometric_cfl_v_initial_floor",
        py::cast(hydro_defaults.rz_geometric_cfl_v_initial_floor));
    set_default_if_missing(
        hydro,
        "rz_geometric_cfl_precise_u_half_enabled",
        py::cast(hydro_defaults.rz_geometric_cfl_precise_u_half_enabled));
    set_default_if_missing(
        hydro,
        "trial_volume_cfl_enabled",
        py::cast(hydro_defaults.trial_volume_cfl_enabled));
    set_default_if_missing(
        hydro,
        "trial_volume_cfl_floor_fraction",
        py::cast(hydro_defaults.trial_volume_cfl_floor_fraction));
    set_default_if_missing(
        hydro,
        "trial_volume_cfl_shrink_fraction",
        py::cast(hydro_defaults.trial_volume_cfl_shrink_fraction));
    set_default_if_missing(
        hydro,
        "corner_jacobian_ale_trigger_enabled",
        py::cast(hydro_defaults.corner_jacobian_ale_trigger_enabled));
    set_default_if_missing(
        hydro,
        "corner_jacobian_floor_eps",
        py::cast(hydro_defaults.corner_jacobian_floor_eps));
    set_default_if_missing(
        hydro,
        "corner_jacobian_ale_trigger_scale",
        py::cast(hydro_defaults.corner_jacobian_ale_trigger_scale));
    set_default_if_missing(
        hydro,
        "in_hydro_corner_j_guard_enabled",
        py::cast(hydro_defaults.in_hydro_corner_j_guard_enabled));
    set_default_if_missing(
        hydro,
        "in_hydro_gauss_j_guard_enabled",
        py::cast(hydro_defaults.in_hydro_gauss_j_guard_enabled));
    set_default_if_missing(
        hydro,
        "in_hydro_rz_volume_guard_enabled",
        py::cast(hydro_defaults.in_hydro_rz_volume_guard_enabled));
    set_default_if_missing(
        hydro,
        "in_hydro_gauss_j_floor_rel",
        py::cast(hydro_defaults.in_hydro_gauss_j_floor_rel));
    set_default_if_missing(
        hydro,
        "in_hydro_rz_volume_floor_rel",
        py::cast(hydro_defaults.in_hydro_rz_volume_floor_rel));
    set_default_if_missing(
        hydro,
        "mesh_quality_dt_cfl_enabled",
        py::cast(hydro_defaults.mesh_quality_dt_cfl_enabled));
    set_default_if_missing(
        hydro,
        "mesh_quality_dt_safety_alpha",
        py::cast(hydro_defaults.mesh_quality_dt_safety_alpha));
    set_default_if_missing(
        hydro,
        "mesh_quality_dt_corner_j_enabled",
        py::cast(hydro_defaults.mesh_quality_dt_corner_j_enabled));
    set_default_if_missing(
        hydro,
        "mesh_quality_dt_gauss_j_enabled",
        py::cast(hydro_defaults.mesh_quality_dt_gauss_j_enabled));
    set_default_if_missing(
        hydro,
        "mesh_quality_dt_rz_volume_enabled",
        py::cast(hydro_defaults.mesh_quality_dt_rz_volume_enabled));
    set_default_if_missing(
        hydro,
        "mesh_quality_dt_axis_margin_additive",
        py::cast(hydro_defaults.mesh_quality_dt_axis_margin_additive));
    set_default_if_missing(
        hydro,
        "mesh_quality_dt_corner_j_floor_rel",
        py::cast(hydro_defaults.mesh_quality_dt_corner_j_floor_rel));
    set_default_if_missing(
        hydro,
        "mesh_quality_dt_gauss_j_floor_rel",
        py::cast(hydro_defaults.mesh_quality_dt_gauss_j_floor_rel));
    set_default_if_missing(
        hydro,
        "mesh_quality_dt_rz_volume_floor_rel",
        py::cast(hydro_defaults.mesh_quality_dt_rz_volume_floor_rel));
    set_default_if_missing(
        hydro,
        "ring7_quotient_enabled",
        py::cast(hydro_defaults.ring7_quotient_enabled));
    set_default_if_missing(
        hydro,
        "regime_aware_corner_j_guard_enabled",
        py::cast(hydro_defaults.regime_aware_corner_j_guard_enabled));
    set_default_if_missing(
        hydro,
        "axis_margin_guard_enabled",
        py::cast(hydro_defaults.axis_margin_guard_enabled));
    set_default_if_missing(
        hydro,
        "axis_margin_additive_in_action8_enabled",
        py::cast(hydro_defaults.axis_margin_additive_in_action8_enabled));
    set_default_if_missing(
        hydro,
        "axis_guard_band_cells",
        py::cast(hydro_defaults.axis_guard_band_cells));
    set_default_if_missing(
        hydro,
        "driver_full_step_retry_enabled",
        py::cast(hydro_defaults.driver_full_step_retry_enabled));
    set_default_if_missing(
        hydro,
        "driver_full_step_retry_max_attempts",
        py::cast(hydro_defaults.driver_full_step_retry_max_attempts));
    set_default_if_missing(
        hydro,
        "dispatcher_state_sensitive_bypass_enabled",
        py::cast(hydro_defaults.dispatcher_state_sensitive_bypass_enabled));
    set_default_if_missing(
        hydro,
        "dispatcher_state_sensitive_repair_cap_per_step",
        py::cast(hydro_defaults.dispatcher_state_sensitive_repair_cap_per_step));
    set_default_if_missing(
        hydro,
        "strategy_first_retry_enabled",
        py::cast(hydro_defaults.strategy_first_retry_enabled));
    set_default_if_missing(
        hydro,
        "strategy_first_max_same_dt_attempts",
        py::cast(hydro_defaults.strategy_first_max_same_dt_attempts));
    set_default_if_missing(
        hydro,
        "driver_retry_active_mesh_repair_enabled",
        py::cast(hydro_defaults.driver_retry_active_mesh_repair_enabled));
    set_default_if_missing(
        hydro,
        "driver_retry_corner_balance_threshold",
        py::cast(hydro_defaults.driver_retry_corner_balance_threshold));
    set_default_if_missing(
        hydro,
        "cascade_on_hydro_retry_enabled",
        py::cast(hydro_defaults.cascade_on_hydro_retry_enabled));
    set_default_if_missing(
        hydro,
        "driver_retry_use_suggested_dt_enabled",
        py::cast(hydro_defaults.driver_retry_use_suggested_dt_enabled));
    py::dict geometric_retry_stagnation;
    if (!dict_contains(hydro, "geometric_retry_stagnation")) {
      geometric_retry_stagnation = py::dict();
      hydro[py::str("geometric_retry_stagnation")] = geometric_retry_stagnation;
    } else if (try_get_child_dict(hydro,
                                  "geometric_retry_stagnation",
                                  &geometric_retry_stagnation)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(hydro[py::str("geometric_retry_stagnation")])) {
      geometric_retry_stagnation =
          hydro[py::str("geometric_retry_stagnation")].cast<py::dict>();
      const auto& gs_defaults = hydro_defaults.geometric_retry_stagnation;
      set_default_if_missing(geometric_retry_stagnation,
                             "enabled",
                             py::cast(gs_defaults.enabled));
      set_default_if_missing(geometric_retry_stagnation,
                             "same_cell_count_threshold",
                             py::cast(gs_defaults.same_cell_count_threshold));
      set_default_if_missing(geometric_retry_stagnation,
                             "sigma_rel_tol",
                             py::cast(gs_defaults.sigma_rel_tol));
      set_default_if_missing(geometric_retry_stagnation,
                             "dt_drop_factor",
                             py::cast(gs_defaults.dt_drop_factor));
      set_default_if_missing(geometric_retry_stagnation,
                             "force_diagnostic_dump",
                             py::cast(gs_defaults.force_diagnostic_dump));
    }
    set_default_if_missing(
        hydro,
        "mesh_geometry_soft_fail_enabled",
        py::cast(hydro_defaults.mesh_geometry_soft_fail_enabled));
    set_default_if_missing(
        hydro,
        "av_qcap_over_p",
        py::cast(hydro_defaults.av_qcap_over_p));
    set_default_if_missing(
        hydro,
        "av_model",
        py::str(av_model_to_string(hydro_defaults.av_model)));
    // Frozen configs predating this key ran BBSW; they resolve to BBSW
    // permanently, mirroring the swept-volume era migration.
    set_default_if_missing(
        hydro,
        "corner_mass_convention",
        py::str("bbsw_radial_v0"));
    // Frozen configs predating this key ran the first-order predictor-corrector;
    // they resolve to pc_v0 permanently, mirroring the corner_mass_convention
    // era pin.
    set_default_if_missing(
        hydro,
        "time_integration",
        py::str("pc_v0"));
    set_default_if_missing(
        hydro,
        "total_energy_identity_check",
        py::cast(hydro_defaults.total_energy_identity_check));
    set_default_if_missing(
        hydro,
        "rz_momentum_scheme",
        py::str(hydro_defaults.rz_momentum_scheme));
    set_default_if_missing(
        hydro,
        "axis_node_mass_convention",
        py::str(hydro_defaults.axis_node_mass_convention));
    set_default_if_missing(
        hydro,
        "av_qcap_center_band_only",
        py::cast(hydro_defaults.av_qcap_center_band_only));
    set_default_if_missing(
        hydro,
        "av_cfl_coefficient",
        py::cast(hydro_defaults.av_cfl_coefficient));
    set_default_if_missing(
        hydro,
        "csw_limiter_enabled",
        py::cast(hydro_defaults.csw_limiter_enabled));
    set_default_if_missing(
        hydro,
        "qei_evaluate_at_t_n",
        py::cast(hydro_defaults.qei_evaluate_at_t_n));
    set_default_if_missing(
        hydro,
        "qei_multiplier",
        py::cast(hydro_defaults.qei_multiplier));
    set_default_if_missing(
        hydro,
        "total_energy_remap_2d_rz",
        py::cast(hydro_defaults.total_energy_remap_2d_rz));
    set_default_if_missing(
        hydro,
        "work_split_audit_2d_rz",
        py::cast(hydro_defaults.work_split_audit_2d_rz));
    set_default_if_missing(
        hydro,
        "work_split_audit_cell_every_n_steps",
        py::cast(hydro_defaults.work_split_audit_cell_every_n_steps));
    set_default_if_missing(
        hydro,
        "hllc_z_flux_2d_rz",
        py::cast(hydro_defaults.hllc_z_flux_2d_rz));
    set_default_if_missing(
        hydro,
        "hllc_z_flux_audit_2d_rz",
        py::cast(hydro_defaults.hllc_z_flux_audit_2d_rz));
    set_default_if_missing(
        hydro,
        "bbs_axis_policy_enabled",
        py::cast(hydro_defaults.bbs_axis_policy_enabled));
    set_default_if_missing(
        hydro,
        "subzonal_mass_enabled",
        py::cast(hydro_defaults.subzonal_mass_enabled));
    set_default_if_missing(
        hydro,
        "subzonal_mass_lagrangian_invariant_enabled",
        py::cast(hydro_defaults.subzonal_mass_lagrangian_invariant_enabled));
    set_default_if_missing(
        hydro,
        "anti_hourglass_kappa",
        py::cast(hydro_defaults.anti_hourglass_kappa));
    set_default_if_missing(
        hydro,
        "subzonal_pressure_enabled",
        py::cast(hydro_defaults.subzonal_pressure_enabled));
    set_default_if_missing(
        hydro,
        "subzonal_dt_limiter_enabled",
        py::cast(hydro_defaults.subzonal_dt_limiter_enabled));
    set_default_if_missing(
        hydro,
        "subzonal_pressure_mode",
        py::cast(hydro_defaults.subzonal_pressure_mode));
    set_default_if_missing(
        hydro,
        "subzonal_band_mode",
        py::cast(hydro_defaults.subzonal_band_mode));
    set_default_if_missing(
        hydro,
        "subzonal_band_feather_layers",
        py::cast(hydro_defaults.subzonal_band_feather_layers));
    set_default_if_missing(
        hydro,
        "subzonal_merit_mode",
        py::cast(hydro_defaults.subzonal_merit_mode));
    set_default_if_missing(
        hydro,
        "subzonal_alpha1",
        py::cast(hydro_defaults.subzonal_alpha1));
    set_default_if_missing(
        hydro,
        "subzonal_alpha2",
        py::cast(hydro_defaults.subzonal_alpha2));
    set_default_if_missing(
        hydro,
        "subzonal_merit_power",
        py::cast(hydro_defaults.subzonal_merit_power));
    set_default_if_missing(
        hydro,
        "subzonal_merit_constant",
        py::cast(hydro_defaults.subzonal_merit_constant));
    set_default_if_missing(
        hydro,
        "hllc_z_flux_hlle_fallback",
        py::cast(hydro_defaults.hllc_z_flux_hlle_fallback));
    set_default_if_missing(
        hydro,
        "hllc_z_flux_strict_quasi_1d",
        py::cast(hydro_defaults.hllc_z_flux_strict_quasi_1d));
    set_default_if_missing(
        hydro,
        "work_split_audit_all_rows",
        py::cast(hydro_defaults.work_split_audit_all_rows));
  }

  py::dict debug;
  if (!dict_contains(numerics, "debug")) {
    debug = py::dict();
    numerics[py::str("debug")] = debug;
  } else if (try_get_child_dict(numerics, "debug", &debug)) {
    // Existing dict is updated below.
  }
  if (py::isinstance<py::dict>(numerics[py::str("debug")])) {
    debug = numerics[py::str("debug")].cast<py::dict>();
    const Config::NumericsConfig::DebugConfig debug_defaults;
    set_default_if_missing(
        debug,
        "trace_mesh_motion",
        py::cast(debug_defaults.trace_mesh_motion));
    set_default_if_missing(
        debug,
        "trace_mesh_node_selector",
        py::cast(debug_defaults.trace_mesh_node_selector));
    set_default_if_missing(
        debug,
        "trace_mesh_cell",
        py::cast(debug_defaults.trace_mesh_cell));
    set_default_if_missing(
        debug,
        "trace_max_steps",
        py::cast(debug_defaults.trace_max_steps));
  }

  py::dict diagnostics;
  if (!dict_contains(numerics, "diagnostics")) {
    diagnostics = py::dict();
    numerics[py::str("diagnostics")] = diagnostics;
  } else if (try_get_child_dict(numerics, "diagnostics", &diagnostics)) {
    // Existing dict is updated below.
  }
  if (py::isinstance<py::dict>(numerics[py::str("diagnostics")])) {
    diagnostics = numerics[py::str("diagnostics")].cast<py::dict>();
    const Config::NumericsConfig::DiagnosticsConfig diagnostics_defaults;
    set_default_if_missing(diagnostics,
                           "phase_resolved_energy",
                           py::cast(diagnostics_defaults.phase_resolved_energy));
    set_default_if_missing(
        diagnostics,
        "r_momentum_source_audit",
        py::cast(diagnostics_defaults.r_momentum_source_audit));
    set_default_if_missing(
        diagnostics,
        "dt_breakdown_history_enabled",
        py::cast(diagnostics_defaults.dt_breakdown_history_enabled));
    py::dict mesh_attribution;
    if (!dict_contains(diagnostics, "mesh_attribution")) {
      mesh_attribution = py::dict();
      diagnostics[py::str("mesh_attribution")] = mesh_attribution;
    } else if (try_get_child_dict(diagnostics, "mesh_attribution", &mesh_attribution)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(diagnostics[py::str("mesh_attribution")])) {
      mesh_attribution =
          diagnostics[py::str("mesh_attribution")].cast<py::dict>();
      const auto& attr_defaults = diagnostics_defaults.mesh_attribution;
      set_default_if_missing(mesh_attribution, "enabled", py::cast(attr_defaults.enabled));
      set_default_if_missing(mesh_attribution,
                             "record_node_displacements",
                             py::cast(attr_defaults.record_node_displacements));
      set_default_if_missing(mesh_attribution,
                             "dump_on_failure_only",
                             py::cast(attr_defaults.dump_on_failure_only));
      set_default_if_missing(mesh_attribution,
                             "enable_leave_one_out_replay",
                             py::cast(attr_defaults.enable_leave_one_out_replay));
    }
    py::dict icf;
    if (!dict_contains(diagnostics, "icf")) {
      icf = py::dict();
      diagnostics[py::str("icf")] = icf;
    } else if (try_get_child_dict(diagnostics, "icf", &icf)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(diagnostics[py::str("icf")])) {
      icf = diagnostics[py::str("icf")].cast<py::dict>();
      const auto& icf_defaults = diagnostics_defaults.icf;
      set_default_if_missing(icf, "enabled", py::cast(icf_defaults.enabled));
      set_default_if_missing(icf,
                             "rho_inner_threshold_g_per_cc",
                             py::cast(icf_defaults.rho_inner_threshold_g_per_cc));
      set_default_if_missing(icf,
                             "rho_outer_threshold_g_per_cc",
                             py::cast(icf_defaults.rho_outer_threshold_g_per_cc));
    }
    py::dict conservation;
    if (!dict_contains(diagnostics, "conservation")) {
      conservation = py::dict();
      diagnostics[py::str("conservation")] = conservation;
    } else if (try_get_child_dict(diagnostics, "conservation", &conservation)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(diagnostics[py::str("conservation")])) {
      conservation = diagnostics[py::str("conservation")].cast<py::dict>();
      set_default_if_missing(conservation,
                             "enabled",
                             py::cast(diagnostics_defaults.conservation.enabled));
    }
    py::dict ale_provenance_emission;
    if (!dict_contains(diagnostics, "ale_provenance_emission")) {
      ale_provenance_emission = py::dict();
      diagnostics[py::str("ale_provenance_emission")] =
          ale_provenance_emission;
    } else if (try_get_child_dict(diagnostics,
                                  "ale_provenance_emission",
                                  &ale_provenance_emission)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(
            diagnostics[py::str("ale_provenance_emission")])) {
      ale_provenance_emission =
          diagnostics[py::str("ale_provenance_emission")].cast<py::dict>();
      set_default_if_missing(
          ale_provenance_emission,
          "enabled",
          py::cast(diagnostics_defaults.ale_provenance_emission.enabled));
    }
    py::dict ale_velcoherence;
    if (!dict_contains(diagnostics, "ale_velcoherence")) {
      ale_velcoherence = py::dict();
      diagnostics[py::str("ale_velcoherence")] = ale_velcoherence;
    } else if (try_get_child_dict(diagnostics,
                                  "ale_velcoherence",
                                  &ale_velcoherence)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(
            diagnostics[py::str("ale_velcoherence")])) {
      ale_velcoherence =
          diagnostics[py::str("ale_velcoherence")].cast<py::dict>();
      const auto& vel_defaults = diagnostics_defaults.ale_velcoherence;
      set_default_if_missing(
          ale_velcoherence, "enabled", py::cast(vel_defaults.enabled));
      set_default_if_missing(ale_velcoherence,
                             "every_n_steps",
                             py::cast(vel_defaults.every_n_steps));
    }
    py::dict mesh_quality_min;
    if (!dict_contains(diagnostics, "mesh_quality_min")) {
      mesh_quality_min = py::dict();
      diagnostics[py::str("mesh_quality_min")] = mesh_quality_min;
    } else if (try_get_child_dict(diagnostics,
                                  "mesh_quality_min",
                                  &mesh_quality_min)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(
            diagnostics[py::str("mesh_quality_min")])) {
      mesh_quality_min =
          diagnostics[py::str("mesh_quality_min")].cast<py::dict>();
      set_default_if_missing(
          mesh_quality_min,
          "enabled",
          py::cast(diagnostics_defaults.mesh_quality_min.enabled));
    }
    py::dict production_audit;
    if (!dict_contains(diagnostics, "production_audit")) {
      production_audit = py::dict();
      diagnostics[py::str("production_audit")] = production_audit;
    } else if (try_get_child_dict(diagnostics,
                                  "production_audit",
                                  &production_audit)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(diagnostics[py::str("production_audit")])) {
      production_audit =
          diagnostics[py::str("production_audit")].cast<py::dict>();
      const auto& audit_defaults = diagnostics_defaults.production_audit;
      set_default_if_missing(production_audit,
                             "enabled",
                             py::cast(audit_defaults.enabled));
      set_default_if_missing(production_audit,
                             "tier",
                             py::cast(audit_defaults.tier));
      set_default_if_missing(production_audit,
                             "audit_json_path",
                             py::cast(audit_defaults.audit_json_path));
      py::dict budget;
      if (!dict_contains(production_audit, "escape_valve_budget")) {
        budget = py::dict();
        production_audit[py::str("escape_valve_budget")] = budget;
      } else if (try_get_child_dict(production_audit,
                                    "escape_valve_budget",
                                    &budget)) {
        // Existing dict is updated below.
      }
      if (py::isinstance<py::dict>(
              production_audit[py::str("escape_valve_budget")])) {
        budget =
            production_audit[py::str("escape_valve_budget")].cast<py::dict>();
        set_default_if_missing(
            budget,
            "mass_max",
            py::cast(audit_defaults.escape_valve_budget.mass_max));
        set_default_if_missing(
            budget,
            "energy_max",
            py::cast(audit_defaults.escape_valve_budget.energy_max));
      }
      if (!dict_contains(production_audit, "region_of_interest")) {
        production_audit[py::str("region_of_interest")] = py::list();
      }
      py::dict gcl;
      if (!dict_contains(production_audit, "gcl")) {
        gcl = py::dict();
        production_audit[py::str("gcl")] = gcl;
      } else if (try_get_child_dict(production_audit, "gcl", &gcl)) {
        // Existing dict is updated below.
      }
      if (py::isinstance<py::dict>(production_audit[py::str("gcl")])) {
        gcl = production_audit[py::str("gcl")].cast<py::dict>();
        set_default_if_missing(
            gcl, "enabled", py::cast(audit_defaults.gcl.enabled));
      }
      py::dict positivity;
      if (!dict_contains(production_audit, "positivity")) {
        positivity = py::dict();
        production_audit[py::str("positivity")] = positivity;
      } else if (try_get_child_dict(production_audit,
                                    "positivity",
                                    &positivity)) {
        // Existing dict is updated below.
      }
      if (py::isinstance<py::dict>(
              production_audit[py::str("positivity")])) {
        positivity = production_audit[py::str("positivity")].cast<py::dict>();
        set_default_if_missing(
            positivity,
            "enabled",
            py::cast(audit_defaults.positivity.enabled));
        set_default_if_missing(
            positivity,
            "fatal_on_neg",
            py::cast(audit_defaults.positivity.fatal_on_neg));
      }
    }
    py::dict mesh_degeneracy_forensics;
    if (!dict_contains(diagnostics, "mesh_degeneracy_forensics")) {
      mesh_degeneracy_forensics = py::dict();
      diagnostics[py::str("mesh_degeneracy_forensics")] =
          mesh_degeneracy_forensics;
    } else if (try_get_child_dict(diagnostics,
                                  "mesh_degeneracy_forensics",
                                  &mesh_degeneracy_forensics)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(
            diagnostics[py::str("mesh_degeneracy_forensics")])) {
      mesh_degeneracy_forensics =
          diagnostics[py::str("mesh_degeneracy_forensics")].cast<py::dict>();
      const auto& forensics_defaults =
          diagnostics_defaults.mesh_degeneracy_forensics;
      set_default_if_missing(mesh_degeneracy_forensics,
                             "enabled",
                             py::cast(forensics_defaults.enabled));
      set_default_if_missing(
          mesh_degeneracy_forensics,
          "corner_j_source_budget_enabled",
          py::cast(forensics_defaults.corner_j_source_budget_enabled));
      set_default_if_missing(
          mesh_degeneracy_forensics,
          "corner_j_source_budget_include_1_ring",
          py::cast(forensics_defaults.corner_j_source_budget_include_1_ring));
      set_default_if_missing(
          mesh_degeneracy_forensics,
          "velocity_history_enabled",
          py::cast(forensics_defaults.velocity_history_enabled));
      set_default_if_missing(
          mesh_degeneracy_forensics,
          "velocity_history_target_cell_c",
          py::cast(forensics_defaults.velocity_history_target_cell_c));
      set_default_if_missing(
          mesh_degeneracy_forensics,
          "velocity_history_sample_every_n_steps",
          py::cast(forensics_defaults.velocity_history_sample_every_n_steps));
      set_default_if_missing(
          mesh_degeneracy_forensics,
          "velocity_history_include_1_ring",
          py::cast(forensics_defaults.velocity_history_include_1_ring));
      set_default_if_missing(
          mesh_degeneracy_forensics,
          "velocity_history_max_records",
          py::cast(forensics_defaults.velocity_history_max_records));
      set_default_if_missing(mesh_degeneracy_forensics,
                             "same_cell_count",
                             py::cast(forensics_defaults.same_cell_count));
      set_default_if_missing(mesh_degeneracy_forensics,
                             "sigma_threshold",
                             py::cast(forensics_defaults.sigma_threshold));
      set_default_if_missing(mesh_degeneracy_forensics,
                             "max_dumps_per_run",
                             py::cast(forensics_defaults.max_dumps_per_run));
      set_default_if_missing(mesh_degeneracy_forensics,
                             "output_dir",
                             py::cast(forensics_defaults.output_dir));
    }
  }

  py::dict profile;
  if (!dict_contains(numerics, "profile")) {
    profile = py::dict();
    numerics[py::str("profile")] = profile;
  } else if (try_get_child_dict(numerics, "profile", &profile)) {
    // Existing dict is updated below.
  }
  if (py::isinstance<py::dict>(numerics[py::str("profile")])) {
    profile = numerics[py::str("profile")].cast<py::dict>();
    py::dict icf_standard_ale;
    if (!dict_contains(profile, "icf_standard_ale")) {
      icf_standard_ale = py::dict();
      profile[py::str("icf_standard_ale")] = icf_standard_ale;
    } else if (try_get_child_dict(
                   profile, "icf_standard_ale", &icf_standard_ale)) {
      // Existing dict is updated below.
    }
    if (py::isinstance<py::dict>(profile[py::str("icf_standard_ale")])) {
      icf_standard_ale =
          profile[py::str("icf_standard_ale")].cast<py::dict>();
      const Config::NumericsConfig::ProfileConfig::IcfStandardAleConfig defaults;
      set_default_if_missing(
          icf_standard_ale, "enabled", py::cast(defaults.enabled));
      set_default_if_missing(
          icf_standard_ale, "enforce", py::cast(defaults.enforce));
      set_default_if_missing(
          icf_standard_ale, "claim_level", py::cast(defaults.claim_level));
      py::dict allowed;
      if (!dict_contains(icf_standard_ale, "allowed_when_enabled")) {
        allowed = py::dict();
        icf_standard_ale[py::str("allowed_when_enabled")] = allowed;
      } else if (try_get_child_dict(
                     icf_standard_ale, "allowed_when_enabled", &allowed)) {
        // Existing dict is updated below.
      }
      if (py::isinstance<py::dict>(
              icf_standard_ale[py::str("allowed_when_enabled")])) {
        allowed =
            icf_standard_ale[py::str("allowed_when_enabled")].cast<py::dict>();
        const auto& allowed_defaults = defaults.allowed_when_enabled;
        set_default_if_missing(
            allowed,
            "ale_enabled_required_value",
            py::cast(allowed_defaults.ale_enabled_required_value));
        set_default_if_missing(
            allowed,
            "ale_axis_repair_mode_required_value",
            py::cast(allowed_defaults.ale_axis_repair_mode_required_value));
        set_default_if_missing(
            allowed,
            "ale_remap_scheme_allowed_values",
            py::cast(allowed_defaults.ale_remap_scheme_allowed_values));
        set_default_if_missing(
            allowed,
            "ale_donor_sign_fixed",
            py::cast(allowed_defaults.ale_donor_sign_fixed_allowed_values));
        set_default_if_missing(
            allowed,
            "hydro_driver_full_step_retry_enabled_required_value",
            py::cast(allowed_defaults
                         .hydro_driver_full_step_retry_enabled_required_value));
      }
      py::dict forbidden;
      if (!dict_contains(icf_standard_ale, "forbidden_when_enabled")) {
        forbidden = py::dict();
        icf_standard_ale[py::str("forbidden_when_enabled")] = forbidden;
      } else if (try_get_child_dict(
                     icf_standard_ale, "forbidden_when_enabled", &forbidden)) {
        // Existing dict is updated below.
      }
      if (py::isinstance<py::dict>(
              icf_standard_ale[py::str("forbidden_when_enabled")])) {
        forbidden =
            icf_standard_ale[py::str("forbidden_when_enabled")].cast<py::dict>();
        const auto& forbidden_defaults = defaults.forbidden_when_enabled;
        set_default_if_missing(
            forbidden,
            "hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value",
            py::cast(forbidden_defaults
                         .hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value));
        set_default_if_missing(
            forbidden,
            "ale_local_boundary_repair_enabled_forbidden_value",
            py::cast(forbidden_defaults
                         .ale_local_boundary_repair_enabled_forbidden_value));
        set_default_if_missing(
            forbidden,
            "ale_multi_node_boundary_repair_enabled_forbidden_value",
            py::cast(forbidden_defaults
                         .ale_multi_node_boundary_repair_enabled_forbidden_value));
        set_default_if_missing(
            forbidden,
            "ale_multi_node_interior_repair_enabled_forbidden_value",
            py::cast(forbidden_defaults
                         .ale_multi_node_interior_repair_enabled_forbidden_value));
        set_default_if_missing(
            forbidden,
            "ale_axis_variational_projection_enabled_forbidden_value",
            py::cast(forbidden_defaults
                         .ale_axis_variational_projection_enabled_forbidden_value));
        set_default_if_missing(
            forbidden,
            "ale_emergency_cell_deactivation_enabled_forbidden_value",
            py::cast(forbidden_defaults
                         .ale_emergency_cell_deactivation_enabled_forbidden_value));
        set_default_if_missing(
            forbidden,
            "hydro_driver_retry_active_mesh_repair_enabled_forbidden_value",
            py::cast(forbidden_defaults
                         .hydro_driver_retry_active_mesh_repair_enabled_forbidden_value));
      }
      py::dict escape;
      if (!dict_contains(icf_standard_ale, "escape_valves")) {
        escape = py::dict();
        icf_standard_ale[py::str("escape_valves")] = escape;
      } else if (try_get_child_dict(icf_standard_ale, "escape_valves", &escape)) {
        // Existing dict is updated below.
      }
      if (py::isinstance<py::dict>(
              icf_standard_ale[py::str("escape_valves")])) {
        escape = icf_standard_ale[py::str("escape_valves")].cast<py::dict>();
        const auto& escape_defaults = defaults.escape_valves;
        set_default_if_missing(
            escape,
            "allow_nonstandard_mesh_rescue",
            py::cast(escape_defaults.allow_nonstandard_mesh_rescue));
        set_default_if_missing(
            escape,
            "require_deck_reason",
            py::cast(escape_defaults.require_deck_reason));
        set_default_if_missing(
            escape,
            "mark_run_nonstandard",
            py::cast(escape_defaults.mark_run_nonstandard));
      }
    }
  }
}

py::dict parse_checkpoint_json_or_throw(const std::string& json_str) {
  py::object json = py::module_::import("json");
  py::object parsed = json.attr("loads")(json_str);
  if (!py::isinstance<py::dict>(parsed)) {
    throw ConfigError("Frozen checkpoint JSON must have an object root");
  }
  return parsed.cast<py::dict>();
}

void apply_checkpoint_migrations(py::dict& root) {
  int schema_version = read_schema_version_or_default(root, kCheckpointJsonSchemaV1);
  if (schema_version <= 0) {
    schema_version = kCheckpointJsonSchemaV1;
  }
  if (!dict_contains(root, "_schema_version")) {
    root[py::str("_schema_version")] = schema_version;
  }
  if (schema_version <= kCheckpointJsonSchemaV1) {
    apply_legacy_nlte_defaults(root);
    apply_legacy_mesh_defaults(root);
    apply_legacy_laser_defaults(root);
    apply_legacy_radiation_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV2;
    schema_version = kCheckpointJsonSchemaV2;
  }
  if (schema_version <= kCheckpointJsonSchemaV2) {
    apply_legacy_diagnostics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV3;
    schema_version = kCheckpointJsonSchemaV3;
  }
  if (schema_version <= kCheckpointJsonSchemaV3) {
    apply_legacy_radiation_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV4;
    schema_version = kCheckpointJsonSchemaV4;
  }
  if (schema_version <= kCheckpointJsonSchemaV4) {
    apply_legacy_radiation_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV5;
    schema_version = kCheckpointJsonSchemaV5;
  }
  if (schema_version <= kCheckpointJsonSchemaV5) {
    apply_legacy_radiation_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV6;
    schema_version = kCheckpointJsonSchemaV6;
  }
  if (schema_version <= kCheckpointJsonSchemaV6) {
    apply_legacy_radiation_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV7;
    schema_version = kCheckpointJsonSchemaV7;
  }
  if (schema_version <= kCheckpointJsonSchemaV7) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV8;
    schema_version = kCheckpointJsonSchemaV8;
  }
  if (schema_version <= kCheckpointJsonSchemaV8) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV9;
    schema_version = kCheckpointJsonSchemaV9;
  }
  if (schema_version <= kCheckpointJsonSchemaV9) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV10;
    schema_version = kCheckpointJsonSchemaV10;
  }
  if (schema_version <= kCheckpointJsonSchemaV10) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV11;
    schema_version = kCheckpointJsonSchemaV11;
  }
  if (schema_version <= kCheckpointJsonSchemaV11) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV12;
    schema_version = kCheckpointJsonSchemaV12;
  }
  if (schema_version <= kCheckpointJsonSchemaV12) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV13;
    schema_version = kCheckpointJsonSchemaV13;
  }
  if (schema_version <= kCheckpointJsonSchemaV13) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV14;
    schema_version = kCheckpointJsonSchemaV14;
  }
  if (schema_version <= kCheckpointJsonSchemaV14) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV15;
    schema_version = kCheckpointJsonSchemaV15;
  }
  if (schema_version <= kCheckpointJsonSchemaV15) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV16;
    schema_version = kCheckpointJsonSchemaV16;
  }
  if (schema_version <= kCheckpointJsonSchemaV16) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV17;
    schema_version = kCheckpointJsonSchemaV17;
  }
  if (schema_version <= kCheckpointJsonSchemaV17) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV18;
    schema_version = kCheckpointJsonSchemaV18;
  }
  if (schema_version <= kCheckpointJsonSchemaV18) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV19;
    schema_version = kCheckpointJsonSchemaV19;
  }
  if (schema_version <= kCheckpointJsonSchemaV19) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV20;
    schema_version = kCheckpointJsonSchemaV20;
  }
  if (schema_version <= kCheckpointJsonSchemaV20) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV21;
    schema_version = kCheckpointJsonSchemaV21;
  }
  if (schema_version <= kCheckpointJsonSchemaV21) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV22;
    schema_version = kCheckpointJsonSchemaV22;
  }
  if (schema_version <= kCheckpointJsonSchemaV22) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV23;
    schema_version = kCheckpointJsonSchemaV23;
  }
  if (schema_version <= kCheckpointJsonSchemaV23) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV24;
    schema_version = kCheckpointJsonSchemaV24;
  }
  if (schema_version <= kCheckpointJsonSchemaV24) {
    apply_legacy_numerics_defaults(root);
    root[py::str("_schema_version")] = kCheckpointJsonSchemaV25;
    schema_version = kCheckpointJsonSchemaV25;
  }
  apply_legacy_numerics_defaults(root);
  normalize_mesh_default_elision(root);
}

bool py_objects_equal(const py::object& lhs, const py::object& rhs) {
  const int result = PyObject_RichCompareBool(lhs.ptr(), rhs.ptr(), Py_EQ);
  if (result < 0) {
    throw py::error_already_set();
  }
  return result == 1;
}

}  // namespace

std::string Freeze::to_checkpoint_json(const NamelistConfig& config) {
  if (!Py_IsInitialized()) {
    throw ConfigError("Python interpreter is not initialized");
  }
  return dumps(serialize_root(config, nullptr, false, false, false), false);
}

bool Freeze::configs_equivalent(const std::string& json_a, const std::string& json_b) {
  if (!Py_IsInitialized()) {
    throw ConfigError("Python interpreter is not initialized");
  }
  try {
    py::dict normalized_a = parse_checkpoint_json_or_throw(json_a);
    py::dict normalized_b = parse_checkpoint_json_or_throw(json_b);
    apply_checkpoint_migrations(normalized_a);
    apply_checkpoint_migrations(normalized_b);
    return py_objects_equal(normalized_a, normalized_b);
  } catch (const py::error_already_set& e) {
    throw ConfigError(std::string("Failed to compare checkpoint frozen_config JSON: ") + e.what());
  }
}

std::string Freeze::to_json(const NamelistConfig& config, const FreezeExtras* extras) {
  if (!Py_IsInitialized()) {
    throw ConfigError("Python interpreter is not initialized");
  }
  return dumps(serialize_root(config, extras, true, true, true), false);
}

void Freeze::write(const NamelistConfig& config,
                   const std::filesystem::path& output_path,
                   const FreezeExtras* extras) {
  if (!Py_IsInitialized()) {
    throw ConfigError("Python interpreter is not initialized");
  }
  ensure_parent_directory(output_path);
  std::ofstream ofs(output_path, std::ios::binary | std::ios::trunc);
  if (!ofs) {
    throw ConfigError("Failed to open freeze output: " + output_path.string());
  }
  ofs << dumps(serialize_root(config, extras, true, true, true), false);
}

void Freeze::write_pretty(const NamelistConfig& config,
                          const std::filesystem::path& output_path,
                          const FreezeExtras* extras) {
  if (!Py_IsInitialized()) {
    throw ConfigError("Python interpreter is not initialized");
  }
  ensure_parent_directory(output_path);
  std::ofstream ofs(output_path, std::ios::binary | std::ios::trunc);
  if (!ofs) {
    throw ConfigError("Failed to open freeze output: " + output_path.string());
  }
  ofs << dumps(serialize_root(config, extras, true, true, true), true) << '\n';
}

}  // namespace tenryu::core::namelist

#else

namespace tenryu::core::namelist {

std::string Freeze::to_checkpoint_json(const NamelistConfig&) {
  throw ConfigError("Python support is disabled (TENRYU_ENABLE_PYTHON=OFF)");
}

bool Freeze::configs_equivalent(const std::string& json_a, const std::string& json_b) {
  return json_a == json_b;
}

std::string Freeze::to_json(const NamelistConfig&, const FreezeExtras*) {
  throw ConfigError("Python support is disabled (TENRYU_ENABLE_PYTHON=OFF)");
}

void Freeze::write(const NamelistConfig&, const std::filesystem::path&,
                   const FreezeExtras*) {
  throw ConfigError("Python support is disabled (TENRYU_ENABLE_PYTHON=OFF)");
}

void Freeze::write_pretty(const NamelistConfig&, const std::filesystem::path&,
                          const FreezeExtras*) {
  throw ConfigError("Python support is disabled (TENRYU_ENABLE_PYTHON=OFF)");
}

}  // namespace tenryu::core::namelist

#endif
