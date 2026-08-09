#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "hydro/bc_2d_rz_semantics.hpp"

namespace tenryu::materials {
struct EOSTableTriplet;
struct IonmixZbarTable;
}

namespace tenryu::core {

enum class RadiationMode {
  ImcDdmc,
  MultigroupDiffusion,
  SnTransport,
};

enum class TopologyScheme {
  SINGLE_BLOCK,
  MULTIBLOCK_CART_CORE_POLAR_SHELL,
  MULTIBLOCK_HALF_BUTTERFLY_5BLOCK,
  MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK,
  CONE_SHELL_SPINE,
  PENTAGON_BELT_SHELL,
};

enum class MultiblockTransitionScheme {
  HERMITE_BRIDGE,
  ROUNDED_HALF_BUTTERFLY,
  ROUNDED_CORE_SEAM,
};

// 1D radial geometry family selector, derived from Main.dimension
// ("1D_SPH" -> Spherical, "1D_CYL" -> Cylindrical; B1 spec
// docs/design/b1_1d_cyl_mode_spec.md). Cylindrical is per unit length
// in z: V = pi (r1^2 - r0^2), A(r) = 2 pi r.
enum class Geometry1D {
  Spherical,
  Cylindrical,
};

inline Geometry1D geometry_1d_from_dimension(const std::string& dimension) {
  return dimension == "1D_CYL" ? Geometry1D::Cylindrical
                               : Geometry1D::Spherical;
}

enum class AvModel {
  ScalarVnrLegacy,
  CswEdge,
  CswEdgePlusTensorLimited,
  CswEdgeCsw98,
};

enum class CornerMassConvention {
  BbswRadialV0 = 0,
  KinematicBasisRzV1 = 1,
};

enum class HydroTimeIntegration {
  PcV0 = 0,
  MidpointV1 = 1,
};

inline const char* hydro_time_integration_to_string(
    const HydroTimeIntegration integration) {
  switch (integration) {
    case HydroTimeIntegration::PcV0:
      return "pc_v0";
    case HydroTimeIntegration::MidpointV1:
      return "midpoint_v1";
  }
  return "pc_v0";
}

inline bool hydro_time_integration_from_string(
    const std::string& value,
    HydroTimeIntegration& integration) {
  if (value == "pc_v0") {
    integration = HydroTimeIntegration::PcV0;
    return true;
  }
  if (value == "midpoint_v1") {
    integration = HydroTimeIntegration::MidpointV1;
    return true;
  }
  return false;
}

inline const char* corner_mass_convention_to_string(
    const CornerMassConvention convention) {
  switch (convention) {
    case CornerMassConvention::BbswRadialV0:
      return "bbsw_radial_v0";
    case CornerMassConvention::KinematicBasisRzV1:
      return "kinematic_basis_rz_v1";
  }
  return "bbsw_radial_v0";
}

inline bool corner_mass_convention_from_string(
    const std::string& value,
    CornerMassConvention& convention) {
  if (value == "bbsw_radial_v0") {
    convention = CornerMassConvention::BbswRadialV0;
    return true;
  }
  if (value == "kinematic_basis_rz_v1") {
    convention = CornerMassConvention::KinematicBasisRzV1;
    return true;
  }
  return false;
}

inline const char* av_model_to_string(const AvModel model) {
  switch (model) {
    case AvModel::ScalarVnrLegacy:
      return "scalar_vnr_legacy";
    case AvModel::CswEdge:
      return "csw_edge";
    case AvModel::CswEdgePlusTensorLimited:
      return "csw_edge_plus_tensor_limited";
    case AvModel::CswEdgeCsw98:
      return "csw_edge_csw98";
  }
  return "scalar_vnr_legacy";
}

inline bool av_model_from_string(const std::string& value, AvModel& model) {
  if (value == "scalar_vnr_legacy") {
    model = AvModel::ScalarVnrLegacy;
    return true;
  }
  if (value == "csw_edge") {
    model = AvModel::CswEdge;
    return true;
  }
  if (value == "csw_edge_plus_tensor_limited") {
    model = AvModel::CswEdgePlusTensorLimited;
    return true;
  }
  if (value == "csw_edge_csw98") {
    model = AvModel::CswEdgeCsw98;
    return true;
  }
  return false;
}

enum class AvQcapScope {
  GLOBAL,
  TRI_FAN_RADIAL_INDEX,
  CENTROID_R_LE_R_MATCH,
};

enum class CenterCflScope {
  DISABLED,
  TRI_FAN_RADIAL_INDEX,
  CENTROID_R_LE_R_MATCH,
};

enum class CenterPerturbationDiagScope {
  DISABLED,
  TRI_FAN_FIRST_RING,
  CENTROID_R_INNERMOST_BINS,
};

struct Config {
  struct CallableInfo {
    std::string name;
    std::string repr;
    std::string source_hash = "unavailable";
    bool detected = false;
  };

  struct MainConfig {
    // Parser may replace this with namelist filename stem when Main.name is omitted.
    std::string name = "unnamed";
    std::string dimension = "1D_SPH";
    std::string temperature_model = "auto";
    // Resolved runtime mode: true=2T, false=1T.
    bool two_temperature = false;
    int dim = 1;
    double t_end = 0.0;
    std::uint64_t seed = 12345;
    int max_steps = 10'000'000;
    std::string verbosity = "normal";
    std::string restart_from;
    std::string units = "cgs_eV";
  };

  struct MeshConfig {
    struct GridSegment {
      double r_start = 0.0;
      double r_end = 0.0;
      int nr = 0;
    };

    struct AutoZoneRegion {
      double r_end = 0.0;
      int nz = 0;
      double rho_ref = 0.0;
      bool is_void = false;
      std::string material_group;
    };

    struct AutoZoneConfig {
      double mass_ratio_max = 1.3;
      int n_bridge_min = 2;
      int n_bridge_max = 10;
      double bridge_frac_max = 0.25;
      double rho_void_cut = 1.0e-6;
      double dr_min = 1.0e-8;
      double mass_ratio_hard_max = 2.0;
      int max_iter = 30;
      double bulk_mass_tol = 1.0e-3;
    };

    struct GradingConfig {
      double edge_ratio = 0.1;
      int sg_order = 4;
      double sg_sigma = 0.7;
      // k01 P0-1 (AI review 2026-07-26): graded width mapping.
      // "legacy_estimated_radius" — historic one-shot thin-shell estimate
      //   q_k = w_k / (r_est^2 + r_ref^2) with r_ref = L/sqrt(N) on
      //   origin segments (bit-preserving default).
      // "exact_measure_v2" — invert the cumulative geometric measure:
      //   r_k^d = r_in^d + W_k (r_out^d - r_in^d), W_k = cumsum(w)/sum(w),
      //   so constant-density cell masses match w_k to roundoff in every
      //   geometry (d=3 sph / 2 cyl / 1 planar).
      std::string mapping = "legacy_estimated_radius";
    };

    struct FloorsConfig {
      double rho_floor_gcc = 1e-10;
      double Te_floor_eV = 1e-3;
      double Ti_floor_eV = 1e-3;
    };

    int nr = -1;
    int nz = 1;
    double r_min = std::numeric_limits<double>::quiet_NaN();
    double r_max = std::numeric_limits<double>::quiet_NaN();
    double z_min = std::numeric_limits<double>::quiet_NaN();
    double z_max = std::numeric_limits<double>::quiet_NaN();
    std::string grid_type_r = "graded";
    std::string grid_type_z = "uniform";
    // W-G: 1D coordinate geometry ("spherical" | "cylindrical" | "planar").
    // Spherical is the historic default; non-spherical values are 1D_SPH-only
    // and validated against unsupported physics (see builder validation).
    std::string geometry_1d = "spherical";
    // Legacy storage retained for frozen-config compatibility.
    std::vector<GridSegment> grid_segments;
    std::string grid_segments_repr;
    std::vector<GridSegment> grid_segments_z;
    std::string grid_segments_z_repr;
    std::vector<GridSegment> grid_segments_theta;
    std::string grid_segments_theta_repr;
    std::vector<AutoZoneRegion> auto_regions;
    std::string auto_regions_axis = "r";  // "r" | "z" (z: 2D_RZ rect only)
    AutoZoneConfig auto_config;
    GradingConfig grading;
    std::vector<double> explicit_nodes;
    std::vector<double> explicit_nodes_z;
    std::vector<double> explicit_nodes_theta;
    double grid_ratio = std::numeric_limits<double>::quiet_NaN();
    std::string grid_bias;
    std::string motion = "lagrangian";
    std::string logical_mesh_2d = "rectangular_rz";
    std::string polar_center_treatment = "annular";
    int center_button_outer_node_ring = 2;
    bool polar_equal_mu_zoning = false;
    double spherical_polar_s_max = 1.0;
    // FIREX W3a-1: truncated polar span. 0.0 (default, bit-frozen) = the full
    // [0, pi] polar mesh. > 0 = the theta ladder starts at polar_theta_min and
    // the +z-side boundary column is a CUT FACE (a plain non-axis boundary that
    // the FIREX junction will later mate to the SHELL_PORT_COLLAR), while the
    // theta = pi side keeps its axis leg. Valid: 0 <= polar_theta_min < 2.6;
    // requires the spherical polar family; v1 center support: tri_fan only.
    double polar_theta_min = 0.0;
    double box_r_max = std::numeric_limits<double>::quiet_NaN();
    double box_z_min = std::numeric_limits<double>::quiet_NaN();
    double box_z_max = std::numeric_limits<double>::quiet_NaN();
    double box_center_z = 0.0;
    // Cone half-angle from the axis [rad].
    double cone_shell_alpha = std::numeric_limits<double>::quiet_NaN();
    // Physical wall thickness measured normal to the faces [cm].
    double cone_shell_wall_thickness =
        std::numeric_limits<double>::quiet_NaN();
    // Truncation radius [cm], interpreted by cone_shell_tip_radius_kind.
    double cone_shell_tip_radius = std::numeric_limits<double>::quiet_NaN();
    std::string cone_shell_tip_radius_kind = "inner_face";
    // Axial coordinate of the planar truncation [cm].
    double cone_shell_tip_z = std::numeric_limits<double>::quiet_NaN();
    // Mid-surface arclength from truncation to base [cm].
    double cone_shell_wall_length = std::numeric_limits<double>::quiet_NaN();
    // Base direction along the symmetry axis, sigma_z in {-1, +1}.
    int cone_shell_axis_sign = 1;
    int cone_shell_n_cells = 10;
    double cone_shell_n_growth = 1.25;
    double cone_shell_tip_size_factor = 3.0;
    double cone_shell_base_size_factor = 8.0;
    double cone_shell_tip_hold = std::numeric_limits<double>::quiet_NaN();
    double cone_shell_grading_length =
        std::numeric_limits<double>::quiet_NaN();
    double cone_shell_l_ratio_max = 1.12;
    double cone_shell_tip_rotation_length =
        std::numeric_limits<double>::quiet_NaN();
    std::string cone_shell_base_cut = "planar";
    double cone_shell_base_rotation_length =
        std::numeric_limits<double>::quiet_NaN();
    // Far-field target measure for CAVITY_CORE axis targets and EXTERIOR_CORE
    // box-face targets: "station_uniform" equidistributes targets in the
    // along-wall station index (far-field spacing ~= span/N_q, columns fan
    // out); "wall_phi" is the legacy Phi-linear transport of the along-wall
    // measure (preserves the v1 fine-band behavior for A/B).
    std::string cone_shell_farfield_target_measure = "station_uniform";
    double cone_shell_outer_vac_first_factor = 0.8;
    int cone_shell_outer_vac_layers = 10;
    double cone_shell_outer_vac_growth = 1.18;
    double cone_shell_inner_vac_first_factor = 1.0;
    int cone_shell_inner_vac_layers = 10;
    double cone_shell_inner_vac_growth = 1.15;
    double cone_shell_end_vac_first_factor = 1.0;
    int cone_shell_end_vac_layers = 10;
    double cone_shell_end_vac_growth = 1.15;
    // Builder-derived C4 topology data (not deck keys).
    int cone_shell_cavity_cells = 0;
    int cone_shell_exterior_cells = 0;
    int cone_shell_tip_fill_layers = 0;
    // Radians; the cone wall angle theta_c.
    double cone_theta_wall = std::numeric_limits<double>::quiet_NaN();
    // s_tip: wall becomes exactly theta_c at and beyond this radius.
    double cone_tip_radius = std::numeric_limits<double>::quiet_NaN();
    // s_a: blend starts here (0 through the prefix).
    double cone_activation_radius = std::numeric_limits<double>::quiet_NaN();
    // Fine band cells on the theta < theta_c side.
    int cone_fine_cells_minus = 4;
    // Fine band cells on the theta > theta_c side.
    int cone_fine_cells_plus = 4;
    double cone_angular_growth_max = 1.25;
    // v1: zero-thickness wall, one logical line.
    std::string cone_tip_style = "single_line";
    int polar_prefix_nr = -1;
    int morph_rings = 16;
    int collar_rings = 6;
    double morph_growth_max = 1.20;
    double spherical_polar_kappa = 0.5;
    TopologyScheme topology_scheme = TopologyScheme::SINGLE_BLOCK;
    bool topology_scheme_explicit = false;
    std::vector<int> pentagon_belt_layers;
    double multiblock_cart_core_r_c = std::numeric_limits<double>::quiet_NaN();
    double multiblock_cart_core_r_match = std::numeric_limits<double>::quiet_NaN();
    int multiblock_cart_core_n_c = -1;
    int multiblock_cart_core_bridge_layers = -1;
    std::string multiblock_cart_core_bridge_grading = "uniform";
    MultiblockTransitionScheme multiblock_transition_scheme =
        MultiblockTransitionScheme::HERMITE_BRIDGE;
    double multiblock_cap_p = 6.0;
    int multiblock_bridge_elliptic_sweeps = 0;
    double multiblock_bridge_elliptic_omega = 0.5;
    // BUG-25: every-step tangential projection of outer-shell corner Svec
    // pairs (I1-B S3-T3 G1 constant-state seam GCL closure). Statically
    // load-bearing for the seam-GCL constant-state gates; under strong drive
    // it deletes tangential restoring forces on the outer arc every step and
    // destabilizes the pole-adjacent outer cells
    // (docs/design/bug25_csr_pole_axis_node_dynamics_20260720.md). Follow-up:
    // replace with a boundary-acceleration projection, then default-flip.
    bool multiblock_outer_svec_tangent_balance = true;
    FloorsConfig floors;
  };

  struct MaterialsConfig {
    struct MatDef {
      std::string name;
      double A = 0.0;
      double Z = 0.0;

      // Supported models include "ideal_gas", "sesame", "ionmix", and "tmat".
      // SPEC default is "sesame"; implementation keeps "ideal_gas" for current test compatibility.
      std::string eos_model = "ideal_gas";
      std::string eos_file;
      int sesame_material_id = -1;
      double ideal_gas_gamma = 5.0 / 3.0;
      double cv_e_override = -1.0;  // [erg/(g*eV)] mass-specific electron heat capacity override (per unit mass per eV)
      double eos_T_ref_eV = -1.0;
      double eos_power_law_f_erg_g = 0.0;
      double eos_power_law_beta = 0.0;
      double eos_power_law_mu_rho = 0.0;
      double eos_power_law_gamma_p = 5.0 / 3.0;
      // Optional c_v softstep (beta_sec Gate-4 discriminator; 0 = absent):
      // e_e += step_D * step_w * softplus((T - step_Tc)/step_w).
      double eos_power_law_step_D_erg_g_eV = 0.0;
      double eos_power_law_step_Tc_eV = 0.0;
      double eos_power_law_step_w_eV = 0.0;
      std::string hydro_eos_backend = "legacy";
      double mg_T_ref_eV = 10.0;
      double mg_dT_rel = 0.1;
      // Loaded EOS tables: ion, electron, total (shared_ptr for cheap copy in Config).
      std::shared_ptr<const materials::EOSTableTriplet> eos_tables;
      // Restart safety hash for table EOS identity; 0 indicates ideal_gas/non-tabular EOS.
      std::uint64_t eos_signature = 0;

      // Supported models include "constant", "freq_dep_marshak", "table_nlte", and "tmat".
      // SPEC default is "ionmix"; implementation keeps "constant" for current test compatibility.
      std::string opacity_model = "constant";
      std::string opacity_file;
      bool tmat_skip_lte_repair = false;
      double kappa_a_constant = 0.0;
      double kappa_planck_override = -1.0;  // <0 = unset: Planck constant follows `kappa_a` (frozen behavior)
      double kappa_s_constant = 0.0;
      double opacity_power_law_kappa0_cm2_g = 0.0;
      double opacity_power_law_alpha_T = 0.0;
      double opacity_power_law_lambda_rho = 0.0;
      double opacity_power_law_T_ref_eV = 1.0;
      double opacity_power_law_rho_ref_g_cc = 1.0;
      bool is_void = false;
      std::string opacity_units = "cm2_per_g";

      // Legacy Non-LTE Fleck controls kept for namelist compatibility.
      // Jayenne separate-emissivity reformulation ignores these at runtime.
      std::string lambda_method = "finite_difference";
      double lambda_fd_delta_rel = 1.0e-4;
      double lambda_fd_abs_min = 1.0e-6;
      double nlte_f_min = 1.0e-4;
    };

    struct ZbarConfig {
      std::string model = "fixed";
      double fixed_value = -1.0;
      std::string table_file;
    };

    struct ZMomentTables {
      int ndens = 0;
      int ntemp = 0;
      std::vector<double> ni_grid;
      std::vector<double> T_grid_eV;
      std::vector<double> r2;   // Zeff/zbar
      std::vector<double> r4;   // <Z^4>/zbar^4
      int provider_material = -1;
    } zmoments;

    struct VoidConfig {
      double rho = 1e-10;
      double Te = 1e-3;
      double Ti = 1e-3;
    };

    std::vector<MatDef> materials;
    std::string opacity_mix_rule = "linear_mass";
    bool low_density_extrapolation = false;
    VoidConfig void_config;
    ZbarConfig zbar;
    std::vector<std::shared_ptr<const ::tenryu::materials::IonmixZbarTable>> zbar_tables;

    int first_nonvoid_material_index() const {
      for (int i = 0; i < static_cast<int>(materials.size()); ++i) {
        if (!materials[i].is_void) {
          return i;
        }
      }
      return -1;
    }
  };

  struct GeometryConfig {
    CallableInfo rho;
    CallableInfo Te;
    CallableInfo Ti;
    CallableInfo velocity;
    std::map<std::string, CallableInfo> volfrac;
    std::string radiation_field = "equilibrium";
    double radiation_field_Tr_eV = -1.0;  // required > 0 when radiation_field == "planck"
    bool enforce_sum_to_one = true;
  };

  struct RadiationConfig {
    struct PlanckFractionConfig {
      std::string method = "compute";
      int compute_N_T = 200;
      std::vector<double> compute_T_range_eV;
      std::vector<double> T_grid_eV;
      std::vector<std::vector<double>> b_g;
    };

    struct CensusCombConfig {
      bool enabled = false;
      int max_particles = 1000000;
      int min_per_bin = 1;
      double trigger_ratio = 1.0;
      double target_fraction = 0.8;
      double mode_weight_imc = 1.0;
      double mode_weight_ddmc = 0.5;
      bool adaptive_trigger = true;
      double adaptive_util_start = 0.70;
      double adaptive_util_end = 0.95;
      double trigger_ratio_floor = 0.85;
      double trigger_hysteresis = 0.05;
      bool ess_floor_enabled = false;
      double ess_min_tier0 = 16.0;
      double ess_min_tier1 = 8.0;
      int max_split_factor = 4;
    };

    struct RadLiteMeshConfig {
      bool enabled = false;
      double sigma_ratio_max = 2.0;
      bool nlte_auto = false;
    };

    struct ImcConfig {
      struct DifferenceConfig {
        bool enabled = false;
        double W_max = 1.0;
        double tau0 = 3.0;
        double chi0 = 1.0;
        bool face_transport = true;
      };

      struct NetElectronSourceSmoothingConfig {
        bool enabled = false;
        double alpha = 0.2;
        double tau_threshold = 4.0;
        int passes = 1;
        double grad_Te_scale = 0.3;
        double grad_rho_scale = 0.5;
        bool gradient_adaptive = false;
      };

      struct ConservativeSmootherConfig {
        bool enabled = false;
        int passes = 10;
        double alpha = 0.5;
      };

      bool enabled = false;
      double alpha = 1.0;
      double f_max = 1.0;
      bool corrected_fleck = false;
      int particles_per_cell_group = 50;
      // Parsed for input compatibility; v1.0 transport always uses continuous
      // absorption (implicit capture effectively hardcoded on).
      bool implicit_capture = true;
      double cutoff_fraction = 0.0;
      bool inelastic_scatter = true;
      double weight_cutoff = 1e-10;
      double roulette_survival = 0.1;
      double weight_split = 1e2;
      int max_split = 8;
      bool linearized_planck = false;
      bool source_tilting = false;
      bool source_localization = false;
      double sloc_ema_beta = 0.4;
      double sloc_sigma_floor = 0.1;
      double sloc_sigma_cap = 0.5;
      double sloc_tau_ref = 1.0;
      double spectral_bias_eta = 0.0;
      bool opacity_predictor = false;
      bool two_stage = false;
      DifferenceConfig difference;
      NetElectronSourceSmoothingConfig net_e_source_smoothing;
      ConservativeSmootherConfig conservative_smoother;
      // -1 = disabled (backward compatible), >0 = total particle cap
      int particle_budget = -1;
      CensusCombConfig census_comb;
      RadLiteMeshConfig rad_lite_mesh;
    };

    struct DdmcConfig {
      bool enabled = false;
      bool implicit_diffusion = false;
      double tau_ddmc = 4.0;
      double tau_rw = 0.0;
      double omega_ddmc = 0.9;
      double tau_ddmc_off = -1.0;
      double omega_ddmc_off = -1.0;
      int mode_hold = 0;
      double rate_max = 1.0e30;
      std::string leak_stencil = "9_kershaw";
      // Parsed for compatibility; v1.0 transport implements asymptotic_diffusion_limit only.
      std::string interface_method = "asymptotic_diffusion_limit";
      bool emissivity_preserving = true;
      std::string interface_exit_distribution = "cosine";
      bool rz_face_r_weight = true;
      std::string face_opacity_temperature = "radiative_mean";
      bool m_matrix_check = true;
    };

    struct DiffusionConfig {
      bool enabled = false;
      double tau_on = 5.0;
      double tau_off = 3.0;
      double reduced_flux_on = 0.15;
      double reduced_flux_off = 0.25;
      int mode_hold = 0;
      double rate_max = 1.0e30;
      int mode_update_interval = 10;
      int min_diffusion_island_cells = 5;
      int imc_guard_cells = 1;
      int sts_max_stages = 0;
      double sts_damping = 0.05;
      double sts_subcycle_eta = 0.8;
      int interface_particles_per_face_group = 32;
      int exit_particles_per_cell_group = 32;
      bool lte_entry_initialization = false;
      double lte_entry_energy_fraction_cap = 0.01;
    };

    struct MultigroupDiffusionConfig {
      struct BoundaryConfig {
        std::string inner_r = "reflect";
        std::string outer_r = "vacuum";
        std::string z = "vacuum";
        // z_bottom/z_top additionally accept "state_supply" in 2D_RZ grey FLD
        // when the matching hydro z-face is state_supply.
        std::string z_bottom = "vacuum";
        std::string z_top = "vacuum";
      };
      struct MarshakConfig {
        double flux_erg_per_cm2_s = 0.0;
        double flux_pulse_duration_s = -1.0;
      };
      struct AmgxConfig {
        std::string preset = "AGGREGATION_JACOBI";
      };

      std::string flux_limiter = "levermore_pomraning";
      int max_outer_iterations = 20;
      double outer_tol = 1.0e-5;
      // W-I (NUMERICS §6.7): matter-emission time linearization.
      // "fleck_cummings": f = 1/(1+z) blend (historic default).
      // "afi": Almost Fully Implicit (Larsen, Kumar & Morel, JCP 238
      // (2013)) — the Fleck blend is not CONSUMED (assembly + matter side
      // run with f == 1) and the outer iteration converges the fully
      // implicit emission. Requires a functioning outer iteration
      // (max_outer_iterations >= ~40 recommended at outer_tol = 1e-5).
      std::string fleck_mode = "fleck_cummings";
      // W-K (verdict 2026-07-04 D2): gamma_r=4/3 radiation compression on
      // the Lagrangian mesh. "none" keeps the frozen-density historic
      // behavior (bit-preserving); "gamma_r_43" scales every group by
      // (V0/V1)^{4/3} per hydro half and joins p_r=(1/3)Sum E_g to the
      // force-side pq (1D + deterministic FLD only in v1).
      // 2026-07-06: v1 default reverted same-day (+10% energy creation);
      // v3 force-work-conjugate payment + FLD rad_E_old snapshot fix landed
      // and the user RE-ADOPTED the default late-night on the fresh A/B
      // (deposited +1.65%, bang -2.28%, rho_peak +4.75%). "none" remains
      // the explicit opt-out for frozen-density historic behavior.
      std::string hydro_coupling = "gamma_r_43";
      // Heat-capacity source for the Fleck stiffness beta = 4*a_eV*Te^3/C_e.
      // "table": C_e from the SAME electron-EOS table the matter Newton
      // advances (Fleck & Cummings consistency; default since the 2026-07-11
      // external-AI verdict — docs/design/fleck_cv_default_flip_20260711.md).
      // "legacy": the pre-flip cv chain (override -> state cv -> ideal gas),
      // kept byte-identical as the explicit frozen compatibility mode.
      std::string fleck_cv_source = "table";
      // "tangent": beta = 4 a T^3 / C_v (frozen default). "guard":
      // beta = max(tangent, secant) => f = min(f_tan, f_sec) — one-sided
      // monotonicity limiter (verdict 2026-07-15 section 9.2). "secant": grey
      // chord beta = dB/dU_e over a 0-D-predicted step interval (table-EOS
      // cells only; 1D FLD only as of 2026-07-14).
      std::string fleck_beta = "tangent";
      // Fleck-factor time-shape form. "be" (frozen default): f = 1/(1+z),
      // the standard backward-Euler Fleck-Cummings factor. "exp_phi1":
      // f = (1 - exp(-z))/z = phi_1(-z) — exact retention for the
      // fixed-radiation scalar relaxation; 0 < f <= 1 and the stiff limit
      // keeps z*f -> 1 (the retired exp(-z) blend's z*f -> 0 failure mode
      // does not apply). 1D FLD only (2D fleck kernel is an independent
      // implementation). Design: docs/design/fleck_exp_source_20260716.md.
      std::string fleck_form = "be";
      // Matter-radiation source integrator for the 1D FLD step. "fleck"
      // (frozen default): the monolithic semi-implicit outer loop with the
      // Fleck factor. "exp_rosenbrock" (opt-in, 1D FLD only): Lie-split
      // step — grey: exact frozen-coefficient direct transfer
      // q = h*phi_1(-(1+beta)h)*(E - aT^4), U+=q, E-=q; multigroup
      // (G<=96, 2026-07-17): conservative diagonal-plus-rank-one
      // dE = phi1(K) r via the certified phi1 pole set — then a pure
      // diffusion solve with the exchange terms stripped; no outer Picard.
      // Second-order; 1-D Marshak feature gate green 2026-07-17 (still
      // opt-in — default flip is a user decision). Design:
      // docs/design/fleck_exp_source_20260716.md section 3 and
      // docs/design/exp_mg_phi1_20260717.md.
      std::string source_integrator = "fleck";
      std::string state_supply_boundary_policy = "local_D_current";
      bool diagnostic_radial_fourier_substage_enabled = false;
      double cg_inner_tol = 1.0e-10;
      // CG convergence normalization for the 2D FLD linear solve.
      // "r0" (historic default): ||r_k|| <= cg_inner_tol * ||r_initial||;
      //   with a good warm start r_initial is already small, so this keeps
      //   demanding 10 more digits from wherever the guess starts.
      // "rhs": ||r_k|| <= cg_inner_tol * max(||b||, tiny) — solve quality is
      //   fixed relative to the problem scale, independent of warm-start
      //   quality. Opt-in; "r0" path is bit-identical to historic behavior.
      std::string cg_tol_norm = "r0";
      // Opt-in Anderson acceleration of the FLD outer (emission linearization)
      // fixed-point iteration Te_{k+1} = G(Te_k). "none" (default) is
      // bit-identical historic plain iteration. "anderson" mixes the next
      // linearization temperature from the last anderson_m residuals
      // (Walker-Ni form, damping anderson_beta); the converged exit state is
      // always the raw Newton output, so the accepted fixed point obeys the
      // same outer_tol contract.
      std::string outer_accel = "none";
      int anderson_m = 2;
      double anderson_beta = 1.0;
      int cg_max_iter = 500;
      std::string cap_exit_policy = "warn";
      std::string linear_solver_1d = "cusparse_tridiag";
      std::string linear_solver_2d = "auto";
      bool linear_solver_2d_explicit = false;
      std::string linear_solver_2d_requested;
      std::string linear_solver_2d_resolved;
      double rgmg_smoother_omega = 0.67;
      AmgxConfig amgx_config;
      double opacity_floor = 1.0e-100;
      double opacity_cap = 1.0e20;
      BoundaryConfig boundary;
      MarshakConfig marshak;
      std::string z_boundary = "vacuum";
    };

    struct SnTransportConfig {
      struct BoundaryConfig {
        std::string inner_r = "reflect_parity";
        std::string outer_r = "vacuum";
        std::string z = "vacuum";
        std::string z_bottom = "vacuum";
        std::string z_top = "vacuum";
      };
      struct MarshakConfig {
        double flux_erg_per_cm2_s = 0.0;
        double flux_pulse_duration_s = -1.0;  // W-B2: 1D grey pulse gating
      };

      int n_angles = 16;
      std::string angular_quadrature = "level_symmetric_16";
      std::string spatial_scheme = "linear_characteristic";
      int max_outer_iterations = 20;
      int max_inner_iterations = 100;
      double outer_tol = 1.0e-4;
      double outer_tol_stagnation_factor = 0.5;
      double outer_tol_hydro_error_scale = 1.0e-5;
      double inner_tol = 1.0e-6;
      int inner_graph_unroll = 5;
      bool dsa_enabled = true;
      std::string diffusion_fallback_mode = "none";
      double tau_diffusion_on = 10.0;
      double tau_diffusion_off = 5.0;
      double opacity_floor = 1.0e-100;
      double opacity_cap = 1.0e20;
      bool timing_enabled = false;
      BoundaryConfig boundary;
      MarshakConfig marshak;
      std::string z_boundary = "vacuum";
    };

    // EXPERIMENTAL: HOLO (High-Order Low-Order) radiation acceleration.
    // Not validated for production use. Use Radiation.imc.difference instead.
    // Enabling HOLO with difference formulation is not supported.
    struct HoloConfig {
      bool enabled = false;  // default OFF; must be explicitly enabled
      std::string region = "shell";
      std::string material_group = "shell";
      double coupling_tau = 5.0;
      int guard_cells = 3;
      int blend_cells = 3;
      int min_lo_cells = 20;
      double q_min = 0.0;
      double q_max = 1.0;
      double tau_on = 5.0;
      double tau_off = 3.0;
      double reduced_flux_on = 0.15;
      double reduced_flux_off = 0.25;
      int update_interval = 10;
      int hold_on = 0;
      int min_dwell_steps = 20;
      int min_island_cells = 5;
      int core_margin_cells = 3;
      std::string solver = "implicit_1d";  // "implicit_1d" or "quasidiffusion_1d"
      std::string closure = "diffusion";
      double closure_relax = 0.2;
      int closure_smooth_passes = 1;
      double closure_smooth_alpha = 0.5;
      double consistency_alpha = 1.0;
      std::string boundary_flux = "physical";
      bool p_rr_tally = true;
      bool sn_closure = true;
      int sn_n_angles = 8;
      bool sn_material_coupling = false;
      int residual_particles_per_cell_group = 4;
    };

    struct BoundaryConfig {
      std::string type = "vacuum";
      std::string inner_r = "reflect";
      std::string outer_r = "vacuum";
      std::string bottom_z = "vacuum";
      std::string top_z = "vacuum";
      int marshak_particles = 1000;
      double marshak_Tr_eV = -1.0;
      CallableInfo marshak_Tr;
      std::map<std::string, CallableInfo> marshak_Tr_map;
    };

    bool enabled = true;
    RadiationMode mode = RadiationMode::MultigroupDiffusion;
    bool origin_parity_only = false;
    bool group_repack_hard_xray = false;
    bool diagnose_hard_xray_opacity = false;
    int groups = 1;
    std::vector<double> group_bounds_eV;
    std::vector<double> compute_T_range_eV;
    std::vector<double> opacity_T_range_eV;
    bool compute_T_range_auto_derived = false;
    PlanckFractionConfig planck_fraction;
    double volume_source_rate = 0.0;
    double volume_source_x_max = -1.0;
    ImcConfig imc;
    DdmcConfig ddmc;
    DiffusionConfig diffusion;
    MultigroupDiffusionConfig multigroup_diffusion;
    SnTransportConfig sn_transport;
    HoloConfig holo;
    BoundaryConfig boundary;
  };

  struct LaserConfig {
    struct AbsorptionConfig {
      std::string model = "inverse_bremsstrahlung";
      double eps_n = 1.0e-4;
      bool terminate = true;
      std::string terminate_mode = "escape";  // escape | deposit
      double coulomb_log_floor = 2.0;
      bool debug_dump_lasermesh = false;
    };

    struct LaserMeshConfig {
      struct GhostCoronaConfig {
        bool enabled = false;
        int n_out = 8;
        double ne_min_frac = 0.03;
        double ne_max_frac = 1.05;
        double Te_min_eV = 50.0;
        double zbar_min = 1.0;
        double zbar_max = 4.0;
        int handoff_cells = 4;
        double handoff_decay = 1.5;
        bool transition_enabled = false;
        double transition_resolved_nhat = 0.9;
        int transition_resolved_cells = 3;
        double transition_density_exponent = 1.0;
      };

      int nr = 128;
      int nz = 256;
      int nr_max = 4096;
      double r_max_factor = 1.5;
      double z_span_factor = 1.5;
      bool critical_clip = true;
      double critical_margin = std::numeric_limits<double>::quiet_NaN();
      std::string stretch_method = "density_gradient";
      double min_ratio = 0.2;
      double mesh_factor = 0.5;
      double rmax_n_hat_threshold = 0.001;
      GhostCoronaConfig ghost_corona;
    };

    struct RaytraceConfig {
      double cfl_ray = 0.8;
      double intensity_cutoff = 1.0e-6;
      double eps_crit = 1.0e-4;
      int max_steps = 100000;
      std::string integrator = "leapfrog";
      double test_kappa = -1.0;
      double ds_adapt_g_target = 0.05;
      double ds_adapt_tau_target = 0.05;
      // Max per-step ray rotation (rad) near turning points:
      // dtheta ~= |grad n_hat| ds / (2 (1 - n_hat)). Unlike the g/tau
      // targets (growth caps, m >= 1), this limiter may SHRINK the step
      // below cfl_ray * dr_node, making the turning-arc cost
      // mesh-independent (~pi/theta steps per turn). <= 0 disables.
      double ds_adapt_theta_target = 0.04;
      double ds_adapt_max_factor = 4.0;
      bool debug_one_ray = false;
    };

    struct DepositConfig {
      double conservation_tol = 1.0e-10;
      int deposit_smooth_passes = 0;
      double deposit_smooth_alpha = 0.25;
    };

    struct RaytraceSkipConfig {
      bool enabled = false;
      double threshold = 0.01;
      int max_consecutive = 10;
      std::string norm = "max_relative";
      double crit_guard = 0.01;
    };

    struct IBExt {
      struct ZeffTableConfig {
        int ndens = 0;
        int ntemp = 0;
        std::vector<double> ni_grid;    // cm^-3
        std::vector<double> T_grid_eV;
        std::vector<double> ratio;      // [nD*nT], d*nT+t
      } zeff_table;

      std::string zeff_model = "auto";            // auto | off | sequential_strip | table
      std::vector<double> species_z = {};         // nuclear charges (ascending)
      std::vector<double> species_x = {};         // number fractions (sum ~ 1)
      std::string coulomb_log_model = "debye";    // debye | laser_frequency
      std::string langdon_model = "off";          // off | legacy_vacuum_map
      double langdon_te_min_eV = 100.0;
    };

    struct RAConfig {
      bool enable = false;
      double chi_p = 0.5;
      double c_ra = 1.0;
    };

    // Multi-beam port table for the single-trace superposition modes
    // (design doc docs/design/multibeam_1d_superposition_20260727.md §5;
    // Follett PoP 32, 022709 (2025) sector/section ray tracing).
    struct LaserPortConfig {
      int port_id = -1;                      // required, unique, >= 0
      std::vector<double> direction;         // required unit-3-vector (normalized at parse)
      double roll_deg = 0.0;                 // rotation about the beam axis
      double power_weight = -1.0;            // required, > 0; sum over ports == 1
      double delta_lambda_nm = 0.0;          // per-port detuning
      std::string beam_class;                // identical-class label ("" = default class)
      std::string polarization = "unpolarized";  // v1: only "unpolarized"
    };
    struct PortConfigurationConfig {
      std::string normalization = "sum_weights_one";  // v1: only this value
      std::vector<LaserPortConfig> ports;   // empty = feature absent
    };

    struct CbetConfig {
      bool enable = false;
      double f_cbet = 1.0;
      double alpha_iaw = 0.2;
      double theta_cap = 0.3;
      double tol = 1.0e-3;
      int max_iters = 50;
      int n_impact_bins = 16;
      int n_phi = 8;
      double ne_frac_cutoff = 0.95;
      double k_a_floor = 1.0e-6;
      int max_segments_per_ray = 0;
      double test_chi = -1.0;
      // "legacy" = current pairwise-ray CBET unchanged; "port_section" =
      // single-trace port phase-space mode (S2+; config-only in S0).
      std::string geometry_mode = "legacy";
      int n_section_phi = 16;   // azimuthal sections (port_section only)
    };

    // Per-channel directional hot-electron source (multi-channel v2; design
    // docs/design/hote_directional_sources_20260710.md §2). Struct
    // initializers hold the "cone" defaults; mechanism-dependent defaults
    // (tpd/srs) are applied at parse time before per-key extraction.
    // eta_table wins over eta when detected (same precedence as the scalar
    // shorthand's eta_hot_table).
    struct HotEChannelConfig {
      std::string mechanism = "cone";     // "cone" | "tpd" | "srs"
      double capture_nc_fraction = 0.25;  // capture surface n_e/n_c; srs default 0.18
      double eta = 0.0;                   // conversion fraction of ray power at capture
      CallableInfo eta_table;             // optional deck callable eta(t)
      // eta_mode="model" per-channel knobs (sentinels -1/"" = mechanism default:
      // tpd C=1.0 inf=0.01 hard=0.03 relaxation "vu2012";
      // srs C=8.0 inf=0.08 hard=0.08 relaxation "fixed").
      double eval_nc_fraction = -1.0;      // default: capture_nc_fraction
      double threshold_multiplier = -1.0;  // C_k
      double eta_inf = -1.0;
      double eta_hard_cap = -1.0;
      double shape_coefficient = 1.0;      // a_k
      std::string relaxation_model;        // "vu2012" | "fixed"
      double relaxation_tau_s = 6.0e-12;
      double relaxation_tau_min_s = 3.0e-12;
      double relaxation_tau_max_s = 1.0e-11;
      double T_hot_eV = 5.0e4;            // tpd default 6.0e4, srs default 4.5e4
      int n_energy_groups = 30;
      double E_min_over_Th = 0.2;
      double E_max_over_Th = 8.0;
      double theta_div_deg = 60.0;        // cone/srs half-angle; srs default 20.0; invalid for tpd
      double tpd_theta_deg = 45.0;        // tpd ring center polar angle [deg]; tpd only
      double tpd_delta_deg = 10.0;        // tpd ring half-width [deg]; tpd only
      int n_mu = 6;
      int n_phi = 8;
    };

    // Physics-model eta(t) shared knobs (eta_mode="model"; design doc
    // docs/design/external-ai-responses/20260727-hote-eta-model-advice.md §18).
    struct HotEEtaModelConfig {
      double ln_filter_tau_s = 5.0e-12;  // EMA time constant for 1/L_n
      double eta_total_cap = 0.08;       // cap on the channel-sum efficiency
    };

    // Hot-electron preheat (1D, v1): prescribed quarter-critical source +
    // exponential multigroup spectrum + CSDA deposition. Default OFF.
    struct HotElectronConfig {
      bool enable = false;
      double source_nc_fraction = 0.25;  // source at n_e = fraction * n_crit
      double eta_hot = 0.0;              // constant conversion fraction of incident laser power
      CallableInfo eta_hot_table;        // optional deck callable eta(t); wins over eta_hot when detected
      // "legacy" = constant/table precedence (bit-identical); "model" = physics model.
      std::string eta_mode = "legacy";
      HotEEtaModelConfig eta_model;
      // Multi-beam overlap reductions for eta_mode="model" (design doc §4).
      std::string tpd_overlap_mode = "single_beam";   // | "common_wave_cluster"
      std::string srs_overlap_mode = "per_beam_class"; // v1: only value
      std::string illumination_metric = "fixed";       // | "equivalent_area"
      double common_wave_delta_theta_deg = -1.0;  // REQUIRED (>0) for cluster mode;
                                                  // no universal default by design
      double T_hot_eV = 5.0e4;           // hot-electron temperature [eV] (50 keV)
      int n_energy_groups = 30;
      double E_min_over_Th = 0.2;        // spectral window lower edge / T_hot
      double E_max_over_Th = 8.0;        // spectral window upper edge / T_hot
      std::string angular_model = "cone";  // "cone" (v2 chord transport) | "radial" (mu=1 verification mode)
      double theta_div_deg = 60.0;         // cone half-angle [deg]; 0 = single direction
      int n_mu = 6;                        // Gauss-Legendre nodes in mu over [cos(theta_div), 1]
      int n_phi = 8;                       // uniform azimuthal nodes over [0, 2pi)
      bool subtract_from_laser = true;   // false = deliberate double-count (sensitivity mode)
      std::string inner_bc = "deposit_residual";  // "deposit_residual" | "escape"
      double explicit_source_limit = 0.2;         // dt limiter safety factor f_E
      // Multi-channel form: Laser.hot_electron.sources = [dict(...), ...].
      // Mutually exclusive (parse-time error) with the scalar shorthand keys
      // above, which remain the single-cone-channel shorthand.
      static constexpr int kMaxSources = 4;  // kernel channel arrays are fixed-size
      bool sources_specified = false;
      std::vector<HotEChannelConfig> sources;
    };

    struct BeamDef {
      std::string name;
      std::vector<double> direction;
      double theta = std::numeric_limits<double>::quiet_NaN();
      double phi = std::numeric_limits<double>::quiet_NaN();
      double f_number = 8.0;
      std::vector<double> focus;
      double defocus_DR = 0.0;
      double delta_lambda_nm = 0.0;
      std::string profile_model;
      double profile_w0_um = -1.0;
      std::vector<double> profile_r_cm;  // "table" model: ascending radii [cm]
      std::vector<double> profile_I;     // "table" model: relative intensity >= 0
      int profile_m = -1;
      CallableInfo power;
    };

    // SPEC default is true; implementation keeps false as a safer default.
    bool enabled = false;
    double wavelength_nm = 351.0;
    std::string mode;
    // radial_absorption_1d parses rays/profile/f_number/focus/defocus, but they do not affect absorption.
    // Builder applies dimension-aware default when not explicitly set (1D_SPH=1000, 2D_RZ=128).
    int rays_per_beam = 1000;
    bool spherical_average = false;
    int ray_output_count = 0;
    bool ray_output_trajectory = false;
    int ray_output_max_steps = 10000;
    std::string profile_model = "gaussian";
    double profile_w0_um = -1.0;
    std::vector<double> profile_r_cm;  // "table" model: ascending radii [cm]
    std::vector<double> profile_I;     // "table" model: relative intensity >= 0
    int profile_m = 2;
    AbsorptionConfig absorption;
    LaserMeshConfig lasermesh;
    RaytraceConfig raytrace;
    // Backward-compatible scalar alias for legacy namelists.
    // Canonical settings live in raytrace_skip_config.
    // Skip is OFF by default and is an opt-in accelerator.
    // If only this scalar is specified and raytrace_skip > 0, parser maps it to
    // raytrace_skip_config.threshold and enables skip.
    double raytrace_skip = 0.0;
    RaytraceSkipConfig raytrace_skip_config;
    DepositConfig deposit;
    IBExt ib;
    RAConfig ra;
    PortConfigurationConfig port_configuration;
    CbetConfig cbet;
    HotElectronConfig hot_electron;
    std::vector<BeamDef> beams;

    [[nodiscard]] bool laser_phys_ext_active() const {
      return ib.zeff_model != "off" || ib.coulomb_log_model != "debye" ||
             ib.langdon_model != "off" || ra.enable ||
             absorption.terminate_mode == "deposit";
    }
  };

  // Nuclear burn kernel (1D_SPH): Bosch-Hale reactivities, per-cell
  // depletion network, Fraley or charged-product diffusion deposition.
  // Design: docs/design/burn_kernel_1d_v1_design_20260710.md. Default OFF;
  // enabled=false must be bitwise-identical to pre-burn binaries.
  struct BurnConfig {
    bool enabled = false;
    // Reaction channels; subset of {"DT","DD","D3He"}. "DD" enables both
    // D(d,p)T and D(d,n)3He branches.
    std::vector<std::string> fuels = {"DT", "DD"};
    std::string scheme = "fraley";  // 1D_SPH: "fraley"|"diffusion"|"mc"; 2D_RZ: "local"|"diffusion"
    int diffusion_groups = 30;              // scheme="diffusion": energy groups
    double diffusion_E_min_keV = 20.0;
    int mc_particles_per_cell = 16;         // scheme="mc": samples per cell/slot/step
    std::string partition = "li_petrasso";  // "li_petrasso" | "fraley" (DT-alpha-only)
    std::string screening = "none";  // "none"|"salpeter"|"chugunov_dewitt" (NUMERICS 13.x)
    // Material names (MatDef::name) whose inventory is fuel.
    std::vector<std::string> fuel_materials = {"DT"};
    double x_D = 0.5;   // fuel number fractions; must sum to 1 +- 1e-6
    double x_T = 0.5;
    double x_He3 = 0.0;
    double T_floor_keV = 0.2;            // reactivity floor (Bosch-Hale fit validity)
    double explicit_source_limit = 0.2;  // dt limiter fraction (hot_electron semantics)
    bool neutron_heating = false;        // v2-E two-line first-collision heating
    int neutron_heating_n_mu = 16;       // even Gauss-Legendre order over mu
    double eps_deplete = 0.1;            // max relative inventory change per substep
    int subcycle_max = 64;
    double vf_threshold = 1.0e-3;        // fuel-region detection threshold
  };

  struct NumericsConfig {
    /// True iff the simulation has a physical r=0 axis row:
    /// 2D_RZ geometry, r_min within axis_eps_cm, and an axis-like r_inner BC.
    /// Annular 2D_RZ decks prefer r_inner="reflect"; legacy r_inner="axis"
    /// remains reflective and must not make i==0 a physical axis internally.
    bool has_physical_rz_axis = false;
    /// Near-zero radial threshold for the physical-axis predicate [cm].
    double axis_eps_cm = 1.0e-12;

    struct PersistentLoopConfig {
      bool enabled = false;   // opt-in; OFF must be bit-identical to today
      int chunk_steps = 128;  // steps per persistent-kernel launch (K)
    };
    PersistentLoopConfig persistent_loop;

    struct DebugConfig {
      bool trace_mesh_motion = false;
      std::string trace_mesh_node_selector = "outer_equator";
      int trace_mesh_cell = 7;
      int trace_max_steps = 5;
    } debug;

    struct DiagnosticsConfig {
      struct MeshAttributionConfig {
        bool enabled = false;
        bool record_node_displacements = false;
        bool dump_on_failure_only = true;
        bool enable_leave_one_out_replay = false;
      };
      struct IcfDiagnosticsConfig {
        bool enabled = false;
        double rho_inner_threshold_g_per_cc = 0.0;
        double rho_outer_threshold_g_per_cc = 0.0;
      };
      struct HotspotGasDiagnosticsConfig {
        bool enabled = false;
        double R_g_cm = 0.0;
        double mass_drift_warn_rel = 1.0e-10;
      };
      struct ConservationDiagnosticsConfig {
        bool enabled = false;
      };
      struct AleProvenanceEmissionConfig {
        bool enabled = false;
      };
      struct ProductionAuditConfig {
        struct EscapeValveBudget {
          double mass_max = 0.0;
          double energy_max = 0.0;
        };
        struct RegionOfInterest {
          int i_min = 0;
          int i_max = 0;
          int j_min = 0;
          int j_max = 0;
        };
        struct GclConfig {
          bool enabled = false;
        };
        struct PositivityConfig {
          bool enabled = false;
          bool fatal_on_neg = false;
        };
        bool enabled = false;
        std::string tier = "none";
        std::string audit_json_path = "<output_dir>/audit_summary.json";
        EscapeValveBudget escape_valve_budget;
        std::vector<RegionOfInterest> region_of_interest;
        GclConfig gcl;
        PositivityConfig positivity;
      };
      struct MeshDegeneracyForensicsConfig {
        bool enabled = false;
        bool corner_j_source_budget_enabled = false;
        bool corner_j_source_budget_include_1_ring = false;
        bool velocity_history_enabled = false;
        int velocity_history_target_cell_c = -1;
        int velocity_history_sample_every_n_steps = 1;
        bool velocity_history_include_1_ring = true;
        int velocity_history_max_records = 5000;
        int same_cell_count = 3;
        double sigma_threshold = 0.5;
        int max_dumps_per_run = 100;
        std::string output_dir;
      };
      struct MeshQualityMinConfig {
        bool enabled = false;
      };
      // Shock-approach detector (shock-ahead reorientation S-A, read-only):
      // every N steps, bin the theta-averaged cell pressure radially, locate
      // the |dp/ds| ridge, and log the fitted inward speed and the predicted
      // arrival time at target_radius_cm (docs/design/
      // shock_ahead_button_reorientation_20260720.md).
      struct ShockApproachConfig {
        bool enabled = false;
        int every = 50;                 // >= 1
        double target_radius_cm = 0.0;  // > 0 required when enabled
        int bins = 192;                 // >= 16 radial bins
        double h_cell_cm = 0.0;         // crossing-time cell size; <= 0 => s_max/bins
      } shock_approach;
      struct AleVelCoherenceConfig {
        bool enabled = false;
        int every_n_steps = 1;
      };
      bool phase_resolved_energy = false;
      bool r_momentum_source_audit = false;
      bool dt_breakdown_history_enabled = true;
      MeshAttributionConfig mesh_attribution;
      IcfDiagnosticsConfig icf;
      HotspotGasDiagnosticsConfig hotspot_gas;
      ConservationDiagnosticsConfig conservation;
      AleProvenanceEmissionConfig ale_provenance_emission;
      MeshQualityMinConfig mesh_quality_min;
      AleVelCoherenceConfig ale_velcoherence;
      ProductionAuditConfig production_audit;
      MeshDegeneracyForensicsConfig mesh_degeneracy_forensics;
    };

    struct DtConfig {
      double initial_s = 1e-15;
      double cfl_hydro = 0.3;
      double cfl_cond = 0.25;
      double f_min_fleck = 0.01;
      double growth_factor = 1.2;
      double max_s = 1e-9;
      double min_s = 1e-20;
      // Stage 22 Wave 2: consecutive committed-step floor-stall detector.
      // 0 = disabled (bit-exact baseline preserved). 32 is the production
      // policy decks may opt into.
      int floor_stall_max_consecutive_steps = 0;
    };

    struct HydroConfig {
      bool enabled = true;
      bool compatible_energy = false;
      bool rho_e_linear_grid = false;
      bool eos_writeback = true;
      // BUG-24: EOS closure energy policy. "energy_authoritative" (default
      // since the 2026-07-18 user ruling; production re-baselined) keeps the
      // evolved energy on table-edge inverse clamps and routes all
      // table-ceiling evaluations through the high-T ideal tail;
      // "legacy" restores the pre-BUG-24 projection behavior (deletes the
      // super-ceiling excess; kept for reproduction of old baselines).
      std::string eos_closure_mode = "energy_authoritative";
      bool qei_evaluate_at_t_n = true;
      double qei_multiplier = 1.0;
      std::string exact_override = "none";
      bool total_energy_remap_2d_rz = false;
      bool work_split_audit_2d_rz = false;
      int work_split_audit_cell_every_n_steps = 0;
      bool work_split_audit_all_rows = false;
      bool hllc_z_flux_2d_rz = false;
      bool hllc_z_flux_audit_2d_rz = false;
      bool hllc_z_flux_hlle_fallback = true;
      bool hllc_z_flux_strict_quasi_1d = false;
      double axis_motion_floor_fraction = 0.0;
      double axis_margin_dt_floor_fraction = 0.0;
      bool volume_rate_cfl_enabled = false;
      double volume_rate_cfl_threshold = 0.5;
      bool tri_fan_center_cfl_enabled = false;
      double tri_fan_center_cfl_safety = 0.5;
      int tri_fan_center_cfl_band_radial_index = 3;
      // S-D predictive corner-Jacobian dt limiter (button bridge band): limits dt
      // so no corner Jacobian in the scoped band can shrink by more than
      // corner_j_predict_max_shrink in one step under the current node velocities
      // (exact quadratic-in-dt evaluation). Default OFF.
      bool corner_j_predict_cfl_enabled = false;
      double corner_j_predict_cfl_safety = 0.5;   // in (0, 1]
      // Anti-Zeno floor: the corner-J limiter never reduces dt below this
      // fraction of the acoustic CFL dt (a fast-closing corner that would demand
      // less is accepted at the floor — repair/AV own the transient).
      double corner_j_predict_floor_frac = 0.05;   // in (0, 1]
      double corner_j_predict_max_shrink = 0.25;  // in (0, 1)
      int corner_j_predict_shell_rings = 4;       // >= 0, shell rings past the seam
      bool tri_fan_center_perturbation_diag_enabled = false;
      AvQcapScope av_qcap_scope = AvQcapScope::GLOBAL;
      CenterCflScope center_cfl_scope = CenterCflScope::DISABLED;
      CenterPerturbationDiagScope center_perturbation_diag_scope =
          CenterPerturbationDiagScope::DISABLED;
      int center_perturbation_diag_radial_bins = 2;
      bool rz_geometric_cfl_enabled = false;
      double rz_geometric_cfl_etaV = 0.5;
      double rz_geometric_cfl_r_floor = 1.0e-10;
      bool rz_geometric_cfl_cumulative_protection_enabled = true;
      double rz_geometric_cfl_v_initial_floor = 0.1;
      bool rz_geometric_cfl_precise_u_half_enabled = false;
      bool trial_volume_cfl_enabled = false;
      double trial_volume_cfl_floor_fraction = 0.05;
      double trial_volume_cfl_shrink_fraction = 0.5;
      bool corner_jacobian_ale_trigger_enabled = false;
      double corner_jacobian_floor_eps = 1.0e-6;
      double corner_jacobian_ale_trigger_scale = 0.5;
      bool in_hydro_corner_j_guard_enabled = false;
      bool in_hydro_gauss_j_guard_enabled = false;
      bool in_hydro_rz_volume_guard_enabled = false;
      double in_hydro_gauss_j_floor_rel = 1.0e-8;
      double in_hydro_rz_volume_floor_rel = 1.0e-8;
      // PR 1: Mesh-quality dt CFL (HYDRA/DRACO-compliant Lagrangian admissibility envelope).
      bool mesh_quality_dt_cfl_enabled = false;
      double mesh_quality_dt_safety_alpha = 0.5;
      bool mesh_quality_dt_corner_j_enabled = true;
      bool mesh_quality_dt_gauss_j_enabled = true;
      bool mesh_quality_dt_rz_volume_enabled = true;
      bool mesh_quality_dt_axis_margin_additive = true;
      double mesh_quality_dt_corner_j_floor_rel = 1.0e-8;
      double mesh_quality_dt_gauss_j_floor_rel = 1.0e-8;
      double mesh_quality_dt_rz_volume_floor_rel = 1.0e-8;
      bool ring7_quotient_enabled = false;
      bool regime_aware_corner_j_guard_enabled = false;
      bool axis_margin_guard_enabled = false;
      bool axis_margin_additive_in_action8_enabled = false;
      int axis_guard_band_cells = 2;
      bool driver_full_step_retry_enabled = false;
      int driver_full_step_retry_max_attempts = 3;
      bool dispatcher_state_sensitive_bypass_enabled = false;
      int dispatcher_state_sensitive_repair_cap_per_step = 3;
      bool strategy_first_retry_enabled = false;
      int strategy_first_max_same_dt_attempts = 2;
      bool driver_retry_active_mesh_repair_enabled = false;
      double driver_retry_corner_balance_threshold = 0.01;
      bool cascade_on_hydro_retry_enabled = false;
      bool driver_retry_use_suggested_dt_enabled = false;
      struct GeometricRetryStagnation {
        bool enabled = false;
        int same_cell_count_threshold = 3;
        double sigma_rel_tol = 0.25;
        double dt_drop_factor = 1.0e-4;
        bool force_diagnostic_dump = true;
      } geometric_retry_stagnation;
      bool mesh_geometry_soft_fail_enabled = false;
      std::string boundary_1d = "free";
      struct Boundary2D {
        struct ZFaceConfig {
          std::string type = "free";
          double supply_rho_g_per_cc = 0.0;
          double supply_u_z_cm_per_s = 0.0;
          double supply_T_eV = 0.0;

          [[nodiscard]] bool is_state_supply() const {
            return type == "state_supply";
          }
        };

        std::string r_inner = "axis";
        std::string r_outer = "free";
        std::string z_bottom = "free";
        std::string z_top = "free";
        ZFaceConfig z_bottom_cfg{};
        ZFaceConfig z_top_cfg{};
        std::string mesh_tangential_target = "lagrangian";
        std::string state_supply_donor_mode = "interior_per_i";
        tenryu::hydro::BC2DRZConfig bc_config;

        Boundary2D() {
          sync_legacy_strings();
        }

        void sync_legacy_strings() {
          z_bottom = z_bottom_cfg.type;
          z_top = z_top_cfg.type;
          bc_config = make_bc_config();
        }

        [[nodiscard]] bool has_any_state_supply() const {
          return z_bottom_cfg.is_state_supply() || z_top_cfg.is_state_supply();
        }

        [[nodiscard]] bool mesh_tangential_uses_reference() const {
          return mesh_tangential_target == "reference";
        }

        [[nodiscard]] tenryu::hydro::BC2DRZConfig make_bc_config() const {
          auto bc = tenryu::hydro::BC2DRZConfig{
              make_radial_axis(r_inner),
              make_radial_axis(r_outer),
              make_z_axis(z_bottom_cfg),
              make_z_axis(z_top_cfg)};
          bc.state_supply_donor_mode = state_supply_donor_mode;
          return bc;
        }

       private:
        [[nodiscard]] tenryu::hydro::BCMeshTangentialKind clamped_tangential_kind() const {
          return mesh_tangential_uses_reference()
                     ? tenryu::hydro::BCMeshTangentialKind::ReferenceTarget
                     : tenryu::hydro::BCMeshTangentialKind::Lagrangian;
        }

        [[nodiscard]] tenryu::hydro::BC2DRZAxis make_radial_axis(
            const std::string& type) const {
          auto axis = tenryu::hydro::BC2DRZAxis{
              tenryu::hydro::BCMaterialNormalKind::Free,
              tenryu::hydro::BCMaterialTangentialKind::Free,
              tenryu::hydro::BCMeshNormalKind::Lagrangian,
              tenryu::hydro::BCMeshTangentialKind::Lagrangian};
          if (type == "axis" || type == "reflect") {
            axis.material_normal = tenryu::hydro::BCMaterialNormalKind::Slip;
            axis.mesh_normal = tenryu::hydro::BCMeshNormalKind::ClampedAtBoundary;
            axis.mesh_tangential = clamped_tangential_kind();
          } else if (type == "fixed") {
            axis.material_normal = tenryu::hydro::BCMaterialNormalKind::Slip;
            axis.material_tangential = tenryu::hydro::BCMaterialTangentialKind::NoSlip;
            axis.mesh_normal = tenryu::hydro::BCMeshNormalKind::ClampedAtBoundary;
            axis.mesh_tangential = tenryu::hydro::BCMeshTangentialKind::LagrangianFreeze;
          } else if (type == "pressure") {
            axis.material_normal = tenryu::hydro::BCMaterialNormalKind::Pressure;
            axis.mesh_normal = tenryu::hydro::BCMeshNormalKind::ClampedAtBoundary;
            axis.mesh_tangential = clamped_tangential_kind();
          }
          return axis;
        }

        [[nodiscard]] tenryu::hydro::BC2DRZAxis make_z_axis(
            const ZFaceConfig& face) const {
          auto axis = tenryu::hydro::BC2DRZAxis{
              tenryu::hydro::BCMaterialNormalKind::Free,
              tenryu::hydro::BCMaterialTangentialKind::Free,
              tenryu::hydro::BCMeshNormalKind::Lagrangian,
              tenryu::hydro::BCMeshTangentialKind::Lagrangian};
          if (face.type == "reflect") {
            axis.material_normal = tenryu::hydro::BCMaterialNormalKind::Slip;
            axis.mesh_normal = tenryu::hydro::BCMeshNormalKind::ClampedAtBoundary;
            axis.mesh_tangential = clamped_tangential_kind();
          } else if (face.type == "fixed") {
            axis.material_normal = tenryu::hydro::BCMaterialNormalKind::Slip;
            axis.material_tangential = tenryu::hydro::BCMaterialTangentialKind::NoSlip;
            axis.mesh_normal = tenryu::hydro::BCMeshNormalKind::ClampedAtBoundary;
            axis.mesh_tangential = tenryu::hydro::BCMeshTangentialKind::LagrangianFreeze;
          } else if (face.type == "state_supply") {
            axis.material_normal = tenryu::hydro::BCMaterialNormalKind::StateSupply;
            axis.mesh_normal = tenryu::hydro::BCMeshNormalKind::ClampedAtBoundary;
            axis.mesh_tangential = clamped_tangential_kind();
            axis.supply_rho = face.supply_rho_g_per_cc;
            axis.supply_u_z = face.supply_u_z_cm_per_s;
            axis.supply_T = face.supply_T_eV;
            axis.open_flow_remap_eligible = true;
          } else if (face.type == "pressure") {
            axis.material_normal = tenryu::hydro::BCMaterialNormalKind::Pressure;
            axis.mesh_normal = tenryu::hydro::BCMeshNormalKind::ClampedAtBoundary;
            axis.mesh_tangential = clamped_tangential_kind();
          }
          return axis;
        }
      } boundary_2d;
      // 1D_SPH raw decks that omit av_type are defaulted to "csw" by the
      // builder (2026-08-03, AV-modernization Stage 1A: limited CSW98 —
      // measured post-shock ripple 3.43% -> 1.97% rms with absorption and
      // bang-time parity). The struct default stays "vnr" because "csw"
      // here is 1D_SPH-only (2-D uses av_model) and frozen configs always
      // carry av_type explicitly.
      std::string av_type = "vnr";
      bool av_type_explicit = false;
      AvModel av_model = AvModel::ScalarVnrLegacy;
      CornerMassConvention corner_mass_convention =
          CornerMassConvention::KinematicBasisRzV1;
      HydroTimeIntegration time_integration = HydroTimeIntegration::MidpointV1;
      bool total_energy_identity_check = false;
      std::string rz_momentum_scheme = "volume_weighted";
      int rz_momentum_scheme_id = 0;
      // BUG-25: axis-node (r==0) nodal-mass convention for the multiblock CSR
      // path. "corner_subzonal" (legacy) sums exact r-weighted subzonal corner
      // masses — axis nodes carry half the structured path's equal-split mass
      // and over-accelerate 4/3 under drive. "equal_split" assembles axis-node
      // mass from the incident cells' equal-split shares (m/4 quads, m/3
      // triangles), matching the structured polar convention (2/3-lagging,
      // stable pole). Accepted values are {"corner_subzonal", "equal_split",
      // "equal_split_all"}. "equal_split" leaves non-axis nodes unaffected.
      // "equal_split_all" uses the equal-split shares for EVERY node (full structured-convention mass distribution; the diagnostic superset of "equal_split").
      // Corner-mass caches are unaffected.
      std::string axis_node_mass_convention = "corner_subzonal";
      double av_linear = 0.1;
      double av_quadratic = 1.5;
      double av_qcap_over_p = 0.0;
      bool av_qcap_center_band_only = false;
      double av_cfl_coefficient = 0.25;
      double csw_C1 = 0.5;
      double csw_C2 = 2.0;
      std::string csw_limiter = "van_leer";
      bool csw_limiter_enabled = true;
      double csw_shock_limiter_floor = 0.65;
      bool csw_zero_uniform_compression = true;
      bool csw_diagnostics = false;
      double av_limiter_J = 1.0;
      double av_heat_C = 0.0;
      bool post_shock_heat = false;
      double post_shock_heat_C = 0.1;
      double post_shock_heat_decay = 3.0;
      double post_shock_velocity_damping_C = 0.0;
      double bulk_viscosity_C = 0.0;
      double ion_art_heat_C = 0.0;
      // k01 P0-4 (AI review 2026-07-26): 1D node-crossing timestep guard.
      // Limits the step so one cell face cannot close more than this fraction
      // of the current cell width per step (raw geometric bound, not scaled
      // by cfl_hydro). 0 disables. Never binds while the acoustic+AV
      // denominator dominates (safety >= cfl_hydro / av_quadratic).
      double crossing_dt_safety = 0.5;
      // k01 P0-2/P0-3 (AI review 2026-07-26): 1D hydro time integrator.
      // "legacy_pc"  — historic predictor-corrector: predictor advances only
      //                u,r,rho to the half step (energy stays at t^n) and the
      //                legacy PdV uses (P^n + P^{n+1/2})/2 (quarter-step bias).
      // "midpoint_v2" — predictor also advances e_e/e_i to the half step with
      //                a dt/2 PdV stage update, and the corrector energy uses
      //                the stage P^{n+1/2}, Q^{n+1/2} directly (2nd order).
      std::string time_integrator = "legacy_pc";
      bool bbs_axis_policy_enabled = false;
      bool subzonal_mass_enabled = false;
      bool subzonal_mass_lagrangian_invariant_enabled = false;
      double anti_hourglass_kappa = 0.05;
      bool subzonal_pressure_enabled = false;
      bool subzonal_dt_limiter_enabled = true;
      std::string subzonal_pressure_mode = "uniform_cell";
      // S-D bridge-band subzonal activation: when "bridge_feather", the
      // subzonal-pressure corner forces (and their compatible-work terms) are
      // scaled per cell by a topology-derived weight: 1 on multiblock bridge
      // cells (block id 1), quintic-smoothstep falloff over
      // subzonal_band_feather_layers face-adjacency layers into the core and
      // shell, 0 beyond. "off" (default) is byte-identical to the global
      // behavior. Requires the multiblock button topology when set to
      // "bridge_feather" (validated at build).
      std::string subzonal_band_mode = "off";   // "off" | "bridge_feather"
      int subzonal_band_feather_layers = 2;     // >= 1
      std::string subzonal_merit_mode = "caramana_auto";
      double subzonal_alpha1 = 1.4142135623730951;
      double subzonal_alpha2 = 0.1;
      int subzonal_merit_power = 2;
      double subzonal_merit_constant = 1.0;
      struct Hourglass {
        bool enabled = false;
        double scale = 0.05;
        bool compatible_work_enabled = true;
        double activation_corner_j_ratio_threshold = 0.5;
        double activation_hourglass_amplitude_threshold = 0.01;
        std::string subzonal_pressure_model = "linearized";
        double max_force_per_node_fraction = 0.2;
      } hourglass;
      struct AdaptiveAVCoeff {
        double c1 = 0.1;
        double c2 = 1.5;
        double heat_C = 0.5;
        double Cpsv = 0.0;
        double cbulk = 0.0;
      };
      struct AdaptiveAVConfig {
        bool enabled = false;
        AdaptiveAVCoeff base{};
        AdaptiveAVCoeff primary{0.5, 1.5, 0.8, 0.75, 0.25};
        AdaptiveAVCoeff rebound{0.18, 1.5, 0.3, 0.0, 0.0};
        double taper_r_start = 0.25;
        double taper_r_end = 0.05;
        double hysteresis_w = 0.3;
        // k01 §8.1 (AI review 2026-07-26): physical hysteresis time scale [s].
        // > 0 replaces the per-step fixed blend hysteresis_w with
        // w(dt) = 1 - exp(-dt/tau) so the gate relaxation rate is
        // timestep-refinement invariant. 0 keeps the legacy fixed-w blend.
        double hysteresis_tau = 0.0;
        int support_ahead = 1;
        int support_behind = 10;
      } adaptive_av;
      // W-H (NUMERICS §3.1.x): Braginskii ion shear viscosity for the 1D
      // Lagrangian step. Default disabled => bit-identical trajectories.
      // model: "braginskii" (eta0 = 0.96 n_i kT_i tau_i, NRL tau_i/lnLambda_ii,
      // Mason-2014 mfp cap) | "constant" (uniform eta_const, verification).
      // species: "ion" (legacy) | "electron" | "both" (additive eta_i + eta_e).
      struct PlasmaViscosity {
        bool enabled = false;
        std::string model = "braginskii";
        // Electron-viscosity lane (2026-07-12): which Braginskii channel(s)
        // feed pi = -eta_eff W. "ion" = W-H legacy (bit-preserving default),
        // "electron" = electron channel only, "both" = additive
        // eta_eff = eta_i + eta_e (regime-automatic; NUMERICS 3.1.13).
        std::string species = "ion";
        double eta_const = 0.0;       // [poise], model=="constant" only
        double eta0_scale = 1.0;      // multiplier on Braginskii eta0
        double mfp_cap_cells = 20.0;  // lambda_ii <= C*dr cap (0=off)
        double lnlambda_fixed = 0.0;  // 0 => NRL lnLambda_ii; >0 fixed
        double dt_safety = 0.3;       // viscous CFL fraction
      } plasma_viscosity;
      bool av_eos_aware = false;
      double av_eos_gamma1_ref = 5.0 / 3.0;
      double av_eos_boost_max = 3.0;
      double odd_even_damping_C = 0.0;
      double ee_odd_even_C = 0.0;
      double hk_velocity_damper_C = 0.0;
      double hk_velocity_damper_tau_min = 8.0;
      double hk_velocity_damper_grad_Te_max = 0.2;
      double hk_velocity_damper_grad_rho_max = 0.3;
      int hk_velocity_damper_guard_cells = 25;
      std::string av_heat_to = "ion";
      CallableInfo pressure_drive_1d;
      std::map<std::string, CallableInfo> pressure_drive_2d;
    };

    struct ConductionConfig {
      bool enabled = true;
      std::string solver = "sts";  // "sts" | "implicit" | "hypre"
      // "net" (legacy): alpha from net power, pairs scaled by min(alpha_c, alpha_nb).
      // "donor": alpha from the outflow sum, each pair scaled by its donor's alpha —
      // guarantees Te_trial >= floor (BUG-18 fix, docs/design/bug18_...20260712.md).
      std::string sts_floor_limiter = "net";
      bool ion_conduction = false;
      double f_lim = 0.06;
      double mfp_limiter_C = 0.0;
      double sts_damping = 0.01;
      int sts_max_stages = 40;
      double sts_subcycle_eta = 0.9;
      // Liveness bound: max total STS stages (n_sub * stages_per_sub) one
      // conduction application may launch per train. A crushed cell's
      // dt_exp ~ dx^2 otherwise makes n_sub unbounded (multi-week grind,
      // 2026-07-30 gauss+SNB hang). 0 disables the bound (legacy).
      int sts_total_stages_max = 200000;
      std::string halo_strategy = "every";
      double hypre_rtol = 1e-8;
      int hypre_max_iter = 50;
      double test_kappa = -1.0;
      bool test_planar = false;
      // W-G2 kirchhoff (external verdict 2026-07-04): face conductivity closure.
      // "harmonic" (historic, bit-preserving default) or
      // "kirchhoff_same_material" (S_{5/2} secant on same-material smooth faces,
      // harmonic fallback on |ln(kappa0_R/kappa0_L)| > ln 10 and on voids;
      // production Spitzer path only — the constant test_kappa closure is
      // analytically identical under both policies).
      std::string face_kappa_policy = "kirchhoff_same_material";
      // SNB nonlocal electron heat transport (1D + 2D_RZ opt-in; designs
      // docs/design/snb_nonlocal_1d_20260710.md and
      // docs/design/2d_snb_port_spec.md). "none" preserves the legacy
      // local Spitzer-Harm + flux-limiter path bit-for-bit.
      std::string nonlocal_model = "none";  // "none" | "snb"
      int snb_n_groups = 24;
      double snb_E_max_over_Te = 20.0;
      std::string snb_mfp = "geometric_r2";  // "geometric_r2" | "original"
      std::string snb_efield = "none";       // "none" | "local"
      int snb_picard_max_iters = 8;
      double snb_picard_rtol = 0.01;
    };

    struct AleConfig {
      struct AlignDiagnosticsConfig {
        bool enabled = false;
        int every_n_steps = 0;
        double c_q_threshold = 0.2;
        double w_rho = 1.0;
        double w_p = 1.0;
        double floor_rel = 1.0e-12;
      };

      bool enabled = false;
      bool ale_identity_mode = false;
      bool ale_mover_diag = false;
      bool ale_preserve_lagrangian_velocity_carry = false;
      AlignDiagnosticsConfig align_diagnostics;
      int every_n_steps = 5;
      int force_rezone_every_n_steps = 0;
      int warmup_steps = 0;
      double relaxation = 0.2;
      double spacing_ratio_threshold = 1.5;
      double quality_threshold = 0.2;
      int max_iterations = 20;
      double convergence_tol = 1e-6;
      double max_displacement_fraction = 0.5;
      std::string remap_limiter = "van_leer";
      bool remap_ms_midpoint = false;
      bool remap_ms_post_check = false;
      int remap_ms_post_max_iter = 3;
      double remap_ms_rescale_floor = 0.01;
      bool ke_fixup = true;
      bool ke_conservation_closure = false;
      bool ke_conservation_closure_audit = false;
      bool ke_closure_redistribute_floor = false;
      bool debug_per_remap_log = false;
      int shock_sensor_guard_cells = 2;
      double density_jump_threshold = 0.1;
      double Te_jump_threshold = 0.2;
      double preventive_axis_guard_fraction = 0.1;
      // "fixed": legacy NODE_AXIS freezes both R and Z (default for bit-exact preservation).
      // "winslow": axis Z moves during rezone via mirrored-axis Winslow update (Phase 8a).
      // "lagrangian_tangential": axis Z moves during hydro Lagrangian step with monotone
      //                           z-order line search and global admissibility validation
      //                           (Phase 8b; required for origin-centered spherical Sedov).
      // "lagrangian": placeholder for unconstrained Lagrangian axis motion (NOT IMPLEMENTED).
      std::string axis_z_motion = "fixed";
      double winslow_axis_kappa = 0.7;
      // Shock-ahead button morph (S-C, docs/design/shock_ahead_button_reorientation_20260720.md):
      // time-scheduled conservative reorientation of the multiblock button core+bridge toward
      // the Shirley-Chiu equal-volume spherical target. Runs inside the multiblock CSR ALE step
      // as a prescribed-target transactional rezone+remap; default OFF (bit-frozen).
      struct ButtonMorphConfig {
        bool enabled = false;
        double t_start_s = 0.0;         // morph window start (absolute sim time, s)
        double t_end_s = 0.0;           // morph window end; > t_start_s when enabled
        double max_step_fraction = 0.05;  // per-transaction |dx| cap vs local min incident edge
        int every_n_steps = 1;          // morph transaction cadence (>= 1)
      } button_morph;
      // Phase 9 - Reference-barrier ALE (default-off, opt-in)
      bool reference_barrier_enabled = false;
      std::string reference_target = "none";
      double reference_blend_default = 1.0;
      double reference_volume_floor_rel = 1.0e-8;
      double reference_corner_j_floor_rel = 1.0e-8;
      double reference_gauss_j_floor_rel = 1.0e-8;
      int reference_linesearch_max_iters = 24;
      bool reference_force_engage_every_step = false;
      bool reference_trigger_axis_margin_enabled = true;
      double reference_trigger_axis_margin_threshold = 1.0e-2;
      bool reference_trigger_corner_j_ratio_enabled = true;
      double reference_trigger_corner_j_ratio_threshold = 0.5;
      bool dgcl_commit_gate = false;
      // Layer-T V10 debug injection: 0=off; K>=1 forces the axis-band attempt at
      // width K to roll back (see SPECIFICATION 6.4).
      int transaction_failure_inject_point = 0;
      double dgcl_commit_rtol = 1.0e-11;
      // B-prime: reference-barrier retry primitive (default-off).
      bool driver_retry_reference_barrier_enabled = false;
      int driver_retry_reference_barrier_K_axis = 4;
      double driver_retry_reference_barrier_eta_axis = 0.05;
      int driver_retry_reference_barrier_max_attempts = 6;
      int driver_retry_reference_barrier_same_sig_max = 3;
      int driver_retry_reference_barrier_cell_window = 2;
      double driver_retry_reference_barrier_dt_collapse_rel = 1.0e-3;
      double driver_retry_reference_barrier_lambda_collapse_threshold = 1.0e-3;
      int driver_retry_reference_barrier_lambda_collapse_count = 2;
      double driver_retry_reference_barrier_quality_progress_factor = 1.25;
      int driver_retry_reference_barrier_quality_progress_count = 2;
      double driver_retry_reference_barrier_rezone_freq_warn_fraction = 0.20;
      int driver_retry_reference_barrier_rezone_freq_window = 200;
      double driver_retry_reference_barrier_chi = 0.8;
      double driver_retry_reference_barrier_q_retry = 0.5;
      // Phase 9 - remap-damage gate (opt-in, default disabled)
      bool remap_damage_gate_enabled = false;
      double remap_damage_dmax = 0.05;
      double remap_damage_axis_eta = 0.02;
      bool remap_damage_axis_budget_enabled = false;
      double remap_damage_axis_budget_factor = 2.0;
      bool predictive_acceptance_enabled = false;
      double predictive_acceptance_axis_floor_fraction = 0.0;
      double predictive_acceptance_cell_vol_floor_fraction = 0.0;
      bool safe_backtrack_enabled = false;
      int safe_backtrack_min_exp = 20;
      int safe_backtrack_binary_iters = 8;
      bool corner_cell_aspect_protection_enabled = true;
      double corner_cell_aspect_eta = 0.5;
      std::string rezone_solver = "legacy_winslow";
      double m1_gamma_align = 0.0;
      double m1_lambda_tether = 0.0;
      double m1_theta_reg = 0.0;
      int m1_sweeps = 8;
      double m1_min_j_dec_rel = 0.0;
      double m1_barrier_beta = 1.0e-3;
      struct EulerWindowConfig {
        bool enabled = false;
        std::string shape = "rectangle";
        double r0 = 0.0;
        double r1 = 0.0;
        double z0 = 0.0;
        double z1 = 0.0;
        double cr = 0.0;
        double cz = 0.0;
        double rad_in = 0.0;
        double rad_out = 0.0;
        double transition_width = 0.0;
      } euler_window;
      bool rezone_local_admissibility_linesearch = false;
      double rezone_local_j_floor_rel = 1.0e-8;
      int rezone_local_linesearch_max_halves = 8;
      bool reject_zero_gauss_j = false;
      double zero_gauss_j_floor_rel = 1.0e-8;
      bool lambda_sweep_diagnostic_enabled = false;
      int lambda_sweep_target_cell_c = -1;
      int lambda_sweep_target_cell_i = -1;
      int lambda_sweep_target_cell_j = -1;
      int lambda_sweep_max_exp = 20;
      bool corner_jacobian_post_tangle_enabled = true;
      bool corner_post_tangle_strict_floor_enabled = false;
      bool local_boundary_repair_enabled = false;
      bool multi_node_boundary_repair_enabled = false;
      bool multi_node_interior_repair_enabled = false;
      bool axis_variational_projection_enabled = false;
      bool emergency_cell_deactivation_enabled = false;
      bool multiblock_cross_seam_rezone_enabled = false;
      bool multiblock_scaled_reference_enabled = false;
      bool multiblock_differential_reference_enabled = false;
      int multiblock_differential_reference_band_count = 64;
      double multiblock_differential_reference_smoothing_g0 = 0.03;
      double multiblock_differential_reference_nu = 0.10;
      double multiblock_differential_reference_eps_v = 0.03;
      double multiblock_differential_reference_s_cap_min_rel = 1.0e-3;
      double multiblock_differential_reference_xi_seam_tol = 1.0e-9;
      double multiblock_differential_reference_sigma_warn_floor = 0.5;
      bool multiblock_lagrangian_bulk_center_patch_reference_enabled = false;
      int multiblock_center_patch_ring_max = 4;
      double multiblock_center_patch_xi_center = 0.0;
      int multiblock_center_patch_halo_layers = 2;
      double multiblock_center_patch_vol_on = 0.05;
      double multiblock_center_patch_vol_off = 0.10;
      double multiblock_center_patch_cornerj_on = 0.03;
      double multiblock_center_patch_cornerj_off = 0.08;
      double multiblock_center_patch_gaussj_on = 0.03;
      double multiblock_center_patch_gaussj_off = 0.08;
      bool ale_reference_diagnostics_enabled = false;
      bool multiblock_path_admissibility_enabled = false;
      double path_admissibility_floor = 0.01;
      double dt_rejection_factor = 0.5;
      int max_dt_rejections = 8;
      bool axis_band_managed_remap_enabled = false;
      int axis_band_managed_remap_width = 3;
      int axis_band_managed_remap_max_width = 6;
      bool axis_band_managed_remap_every_hydro_half_step = true;
      double axis_band_managed_remap_margin_trigger = 1.0e-4;
      bool axis_band_managed_remap_equal_volume = true;
      bool axis_band_managed_remap_include_radiation_groups = true;
      // 5-block half-butterfly full-axis target-only rezone. Default OFF;
      // trigger fractions are dimensionless in (0, 1], eta floor is in (0, 1).
      bool axis_rezone_enabled = false;
      double axis_rezone_trigger_edge_fraction = 0.1;
      double axis_rezone_trigger_min_altitude_fraction = 0.1;
      double axis_rezone_eta_floor = 1.0e-2;
      bool core_freeze_enabled = false;
      std::string core_freeze_source = "gas_tracer";
      double core_freeze_tracer_cut = 0.5;
      int core_freeze_halo_layers = 1;
      bool core_freeze_apply_to_axis_rezone = true;
      bool core_freeze_skip_velocity_projection = true;

      // Phase 10 - axis repair mode
      std::string axis_repair_mode = "full_winslow";

      // Phase 11 - 2nd-order MS2 remap (opt-in, default legacy_split)
      std::string remap_scheme = "legacy_split";
      std::string remap_ms2_limiter = "van_leer";
      // Wave 2A (AI review R1): the corrected oriented convention
      // (dV > 0 = low-to-high transfer donor) is the production default.
      // The legacy reversed-donor convention survives only under
      // legacy_regression@2026-07-27 or an explicit namelist false; restarts
      // enforce their recorded resolved contract.
      bool swept_volume_sign_fixed = true;
      bool conservative_remap_enabled = false;
      std::string conservative_remap_target = "reference";
      bool conservative_remap_radiation_enabled = true;
      std::string conservative_remap_order = "first_order_donor";
      bool tri_fan_tracking_reference_enabled = false;
      std::string tri_fan_tracking_reference_mode = "seamless_converging";
      double tri_fan_tracking_reference_omega = 0.2;
      double tri_fan_tracking_reference_beta = 1.0;
      double tri_fan_tracking_reference_g0 = 0.03;
      double tri_fan_tracking_reference_nu = 0.15;
      // Scalar target for shocked/outer volume-ratio control; smooth interiors may relax.
      double tri_fan_tracking_reference_eps_v = 0.05;
      bool conservative_remap_lagrangian_bulk_enabled = false;
      int conservative_remap_lagrangian_bulk_center_node_ring_max = 4;
      bool central_pseudo_core_enabled = false;
      double central_pseudo_core_s_c = 0.0;
      // Dynamic complete-ring absorption of the central macro cell
      // (NUMERICS Exp1): ladder cap rings -> fan layers -> pure-gas shell
      // rows, with the mass-weighted material guard. The historical
      // TENRYU_I1B_RING_ABSORB* environment variables override these when
      // set (experimental-deck compatibility).
      bool central_pseudo_core_ring_absorption_enabled = false;
      // Volume-trigger fraction of the watched unit's arming-time minimum.
      double central_pseudo_core_ring_absorption_tau = 0.05;
      // Convergence-following global rezone (row-smoothing toward the
      // convergent reference; NUMERICS 13 refutation-lattice member kept in
      // the certified stack). env override: TENRYU_I1B_CONV_REZONE.
      bool conv_rezone_enabled = false;
      // Core1D master switch; env override: TENRYU_I1B_CORE_1D_SUBMODEL.
      bool central_pseudo_core_core1d_enabled = false;
      // Core1D build shell count [4,4096]; env override: TENRYU_I1B_CORE_1D_BUILD_SHELLS.
      int central_pseudo_core_core1d_build_shells = 48;
      // Core1D append split count [0,1024]; env override: TENRYU_I1B_CORE_1D_SPLIT_APPEND.
      int central_pseudo_core_core1d_split_append = 0;
      // Core1D linear VNR AV coefficient; env override: TENRYU_I1B_CORE_1D_AV_C1.
      double central_pseudo_core_core1d_av_c1 = 0.5;
      // Core1D quadratic VNR AV coefficient; env override: TENRYU_I1B_CORE_1D_AV_C2.
      double central_pseudo_core_core1d_av_c2 = 4.0;
      // Core1D CFL number; env override: TENRYU_I1B_CORE_1D_CFL.
      double central_pseudo_core_core1d_cfl = 0.25;
      // Core1D piston speed cap factor; env override: TENRYU_I1B_CORE_1D_PISTON_CAP.
      double central_pseudo_core_core1d_piston_cap = 10.0;
      // Core1D maximum substeps per 2D step; env override: TENRYU_I1B_CORE_1D_MAX_SUBSTEPS.
      int central_pseudo_core_core1d_max_substeps = 20000;
      // Core1D distribution-preserving append (per-cell u_r-sorted
      // sub-shells; NUMERICS 13.2 fidelity upgrade); env override:
      // TENRYU_I1B_CORE_1D_DIST_APPEND.
      bool central_pseudo_core_core1d_dist_append = false;
      // Spherical absorption gas-front mode; env override: TENRYU_I1B_SPHERICAL_ABSORB_GASFRONT.
      bool central_pseudo_core_spherical_absorb_gasfront = false;
      // Spherical absorption radius trigger; 0 disables, else (0,1); env override: TENRYU_I1B_SPHERICAL_ABSORB_ALPHA.
      double central_pseudo_core_spherical_absorb_alpha = 0.0;
      // Spherical absorption pressure-jump trigger; 0 disables, else >1; env override: TENRYU_I1B_SPHERICAL_ABSORB_PJUMP.
      double central_pseudo_core_spherical_absorb_pjump = 0.0;
      // Mixed-row emergency absorption; env override: TENRYU_I1B_MIXED_ABSORB.
      bool central_pseudo_core_mixed_absorb_enabled = false;
      // Absorption watch-row count [1,8]; env override: TENRYU_I1B_ABSORB_WATCH_ROWS.
      int central_pseudo_core_absorb_watch_rows = 1;
      // Terminal absorption master switch; env override: TENRYU_I1B_TERMINAL_ABSORB.
      bool central_pseudo_core_terminal_absorb_enabled = false;
      // Terminal rebound factor >1; env override: TENRYU_I1B_TERMINAL_REBOUND_FACTOR.
      double central_pseudo_core_terminal_rebound_factor = 1.02;
      // Terminal core1d tail chunk dt [s]; env override: TENRYU_I1B_TERMINAL_TAIL_DT.
      double central_pseudo_core_terminal_tail_dt_s = 1.0e-12;
      // Remap mass-closure rejection tolerance; 0 disables; env override: TENRYU_I1B_REMAP_CLOSURE_REJECT_TOL.
      double remap_mass_closure_reject_tol = 0.0;
      // Rezone closure cooldown steps; env override: TENRYU_I1B_REZONE_CLOSURE_COOLDOWN_STEPS.
      int rezone_closure_cooldown_steps = 50;
      // Option-B coherent transport bookkeeping; env override: TENRYU_I1B_OPTIONB_COHERENT.
      bool csr_optionb_coherent_enabled = false;
      // Option-B CSR velocity remap authority; env override: TENRYU_I1B_OPTIONB_VELREMAP.
      bool csr_optionb_velocity_remap_enabled = false;
      // Pole-axis BBSW closure; env override: TENRYU_I1B_POLE_AXIS_BBSW.
      bool pole_axis_bbsw_enabled = false;
      // Axis contact position guard; env override: TENRYU_I1B_AXIS_CONTACT_GUARD.
      bool axis_contact_guard_enabled = false;
      // Mass-floor sentinel absorption; env override: TENRYU_I1B_MASS_FLOOR_ABSORB.
      bool mass_floor_absorb_enabled = false;
      // Ring7 interior-patch remap; env override: TENRYU_I1B_INTERIOR_PATCH_REMAP.
      bool interior_patch_remap_enabled = false;
      // Optional tighter depth cap; 0 = unlimited up to the material /
      // topology guard.
      int central_pseudo_core_ring_absorption_max_rings = 0;
      // Mass-weighted per-row gas tracer floor (material guard).
      double central_pseudo_core_ring_absorption_gas_tracer_min = 0.99;
      // Loose per-cell hard bound (never absorb a majority-shell cell).
      double central_pseudo_core_ring_absorption_gas_tracer_cell_min = 0.5;
      // Pole-sector angular rezone (pole-shear verdict step 3): per pole,
      // retarget the first m_theta off-axis node columns of every active
      // polar-shell node row onto a reference angular ladder anchored at the
      // row's a=m_theta column, preserving spherical radius. The target rides
      // the same transactional guard and conservative remap as the
      // axis-chain rezone. Default-off; empirically NET-NEGATIVE at nr16
      // Case B (commit 5df5635e) — an available robustness lever, not a
      // recommendation. The historical TENRYU_I1B_POLE_REZONE* environment
      // variables override these when set (experimental-deck compatibility).
      bool pole_sector_rezone_enabled = false;
      // Off-axis node columns per pole to retarget (>= 2; runtime also caps
      // at ntheta/4).
      int pole_sector_rezone_m_theta = 4;
      // Blend fraction toward the reference ladder per fire, in (0, 1].
      double pole_sector_rezone_lambda = 0.5;
      // Reference ladder: "uniform" (initial uniform-theta zoning; identity
      // on a healthy mesh) or "equal_mu" (axisymmetric-volume-fraction
      // ladder; a large restructuring that churns the remap).
      std::string pole_sector_rezone_mode = "uniform";
      // Per-node deadband as a fraction of the anchor angle delta_M, in
      // [0, 1). 0 = no deadband — this matches the historical env-unset
      // behavior (5df5635e's "default 0.05" was the recommended magnitude
      // for its deadband run, not the env-unset value).
      double pole_sector_rezone_deadband_frac = 0.0;
    };

    struct PlicConfig {
      bool enabled = false;
      std::string normal_estimator = "youngs_seeded_LVIRA";
      std::string t0_volume_cut_method = "adaptive_subdivision_2x2";
      int t0_volume_cut_max_depth = 6;  // range [4, 16] (was [8, 16] in Wave A; 12 was too tight for hard step interfaces)
      double t0_volume_cut_volfrac_tol = 1.0e-10;
      double fast_path_threshold_min = 1.0e-10;
      double fast_path_threshold_max = 0.9999999999;
      int fast_path_halo_radius_cells = 1;
      int alpha_solver_max_iter = 50;
      double alpha_tolerance_rel = 1.0e-12;
      double thermodynamic_error_soft_threshold = 0.05;
      double thermodynamic_error_hard_threshold = 0.10;
      double class_d_dense_fraction_threshold = 0.01;
      std::string material_interface_per_cell_state = "off";
      bool production_comparable_gate_strict = true;
      double drift_sensor_max_relative = 0.1;
      double drift_sensor_max_swept_fraction = 0.05;
      double prev_normal_freshness_volfrac_threshold = 0.05;
      double plic_per_step_cost_target_fraction = 0.5;
      bool in_run_disabled = false;
      bool rho_material_aware_donor = false;
    };

    struct MaterialsSubConfig {
      bool per_material_conservation_enabled = false;
      double presence_threshold_volfrac = 1.0e-10;
      double presence_threshold_mass_density_g_per_cc = 1.0e-12;
      std::map<std::string, double> eos_table_validity_lower_bound_g_per_cc;
      bool lazy_cache_te_m_enabled = false;
      bool hdf5_emit_derived_per_material = false;
      bool deposit_redistribute_fallback_enabled = false;
      std::string deposit_redistribute_provenance_label =
          "TENRYU_EXTENDED_ALE_WAVE_F_DEPOSIT_REDISTRIBUTE_FALLBACK";
      double conservation_residual_warn_threshold_rel = 1.0e-12;
      double conservation_residual_hard_warning_threshold_rel = 1.0e-10;
    };

    struct Ale1dConfig {
      struct LaserSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.060;
        int sigma_min_cells = 4;
        int sigma_max_cells = 16;
        double peak_fraction = 0.35;
        double conf_low = 0.10;
        double conf_high = 0.40;
      };

      struct AblationSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.080;
        int sigma_min_cells = 3;
        int sigma_max_cells = 14;
        double peak_fraction = 0.40;
        double reference_density_gcc = 1.05;
        double rho_gate_frac = 0.07;
        double rho_gate_width = 0.02;
        double te_gate_low_eV = 0.5;
        double te_gate_high_eV = 2.0;
        double conf_low = 0.10;
        double conf_high = 0.35;
      };

      struct ShockSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.040;
        int sigma_min_cells = 2;
        int sigma_max_cells = 8;
        double peak_fraction = 0.35;
        double qvisc_conf_low = 0.03;
        double qvisc_conf_high = 0.10;
        double du_cs_conf_low = 0.03;
        double du_cs_conf_high = 0.15;
      };

      struct InterfaceSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.033;
        double target_cells_cap_fraction = 0.067;
        int max_features = 8;
        int min_separation_cells = 4;
        double jump_low = 0.05;
        double jump_high = 0.25;
        int sigma_min_cells = 2;
        int sigma_max_cells = 4;
        bool pin_interfaces = true;
      };

      struct CenterSensorConfig {
        bool enabled = true;
        double target_cells_fraction = 0.053;
        int sigma_min_cells = 6;
        int sigma_max_cells = 20;
        double search_x = 0.12;
      };

      struct RezoneConfig {
        double monitor_floor = 1.0;
        double monitor_wmax_ratio = 50.0;
        int monitor_smoothing_iterations = 2;
        bool monitor_smooth_across_protected_faces = false;
        double min_floor_fraction = 0.55;
        double gaussian_truncation_sigma = 3.0;

        bool spatial_monitor_enabled = true;
        double spatial_target_cells_fraction = 0.067;
        double spatial_power = 2.0;
        double laser_spatial_dr_min_cm = 2.5e-5;
        double laser_spatial_dr_max_cm = 2.0e-4;
        double ablation_spatial_dr_min_cm = 1.5e-5;
        double ablation_spatial_dr_max_cm = 1.2e-4;
        double shock_spatial_dr_min_cm = 1.0e-5;
        double shock_spatial_dr_max_cm = 8.0e-5;
      };

      struct MinWidthFloorConfig {
        bool enabled = false;
        double floor_cm = 0.0;          // trigger + guarantee: no cell below this after rezone
        double target_factor = 1.25;    // respace target = target_factor * floor_cm
        int relief_halfwidth_cells = 3;  // half-width of the minimum-cell relief neighborhood
        double max_growth_factor = 1.8;  // per-application cap: no cell grows more than this per rezone
      };

      struct RemapConfig {
        bool reject_multicell_sweeps = true;
        bool high_order_enabled = true;
        double limiter_theta = 1.5;
        int high_order_ramp_cells = 2;
        int radiation_high_order_ramp_cells = 2;
        bool fallback_to_first_order_on_bounds_fail = true;
        bool reject_strict_zero_flux_on_moving_protected_face = true;
      };

      bool enabled = false;  // EXPERIMENTAL: opt-in only

      // Trigger
      int every_n_steps = 100;
      int min_steps_between_ale = 50;
      bool enable_benefit_gate = true;
      double benefit_min_dt_gain = 1.5;
      double candidate_dt_penalty_max = 1.25;
      bool emergency_enabled = true;

      // Eligibility guards
      int min_cells = 256;
      double protected_fraction_max = 0.25;
      int min_movable_segment_warn = 24;
      int min_movable_segment_hard = 8;
      double max_node_displacement_fraction_mu = 0.35;
      double max_node_displacement_fraction_r = 0.35;
      bool ke_conservation_closure = false;

      // Conservation tolerances
      struct Tol {
        double soft = 0.0;
        double hard = 0.0;
      };
      Tol total_mass_tol{1e-12, 1e-9};
      Tol material_mass_tol{1e-11, 1e-8};
      Tol radiation_group_energy_tol{1e-8, 1e-5};
      Tol material_internal_energy_tol{1e-8, 1e-5};
      Tol total_material_energy_tol{1e-7, 1e-5};
      Tol global_total_energy_tol{1e-6, 1e-4};
      Tol kinetic_energy_drift_tol{1e-7, 1e-5};

      // Diagnostics
      bool diagnostics_enabled = true;
      int diagnostics_log_every_n_steps = 100;
      bool diagnostics_collect_step_result = true;
      bool diagnostics_fail_on_unexpected_apply = false;

      LaserSensorConfig laser_sensor;
      AblationSensorConfig ablation_sensor;
      ShockSensorConfig shock_sensor;
      InterfaceSensorConfig interface_sensor;
      CenterSensorConfig center_sensor;
      RezoneConfig rezone;
      MinWidthFloorConfig min_width_floor;
      RemapConfig remap;
    };

    struct ProfileConfig {
      struct IcfStandardAleConfig {
        bool enabled = false;
        bool enforce = true;
        std::string claim_level = "characterization";
        struct AllowedWhenEnabled {
          bool ale_enabled_required_value = true;
          std::string ale_axis_repair_mode_required_value = "full_winslow";
          std::vector<std::string> ale_remap_scheme_allowed_values = {
              "legacy_split", "ms2_moments"};
          // Empty/default permits both false and true.
          std::vector<bool> ale_donor_sign_fixed_allowed_values;
          bool hydro_driver_full_step_retry_enabled_required_value = true;
        } allowed_when_enabled;
        struct ForbiddenWhenEnabled {
          bool hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value = true;
          bool ale_local_boundary_repair_enabled_forbidden_value = true;
          bool ale_multi_node_boundary_repair_enabled_forbidden_value = true;
          bool ale_multi_node_interior_repair_enabled_forbidden_value = true;
          bool ale_axis_variational_projection_enabled_forbidden_value = true;
          bool ale_emergency_cell_deactivation_enabled_forbidden_value = true;
          bool hydro_driver_retry_active_mesh_repair_enabled_forbidden_value = true;
        } forbidden_when_enabled;
        struct EscapeValves {
          bool allow_nonstandard_mesh_rescue = false;
          bool require_deck_reason = true;
          bool mark_run_nonstandard = true;
        } escape_valves;
      };
      IcfStandardAleConfig icf_standard_ale;
      struct LegacyRegressionConfig {
        bool enabled = false;
        std::string revision = "2026-07-27";
      };
      LegacyRegressionConfig legacy_regression;
    };

    struct FloorsConfig {
      double rho = 1e-10;
      double Te = 1e-3;
      double Ti = 1e-3;
    };

    struct SafetyConfig {
      bool energy_fatal = false;
      bool nan_fatal = true;
      double energy_budget_tol = 1e-3;
      double opacity_floor = 1e-20;
      double opacity_cap = 1e20;
      // v1.0 warning path is wired in conduction; radiation/source floor-clamp
      // counters should be added to the same threshold monitoring in future.
      int clamp_warn_threshold = 100;
      int clamp_fatal_threshold = 10000;
      double overshoot_warn = 0.01;
      double overshoot_fatal = 0.10;
      bool overshoot_fatal_enabled = false;
    };

    DtConfig dt;
    double T_start_eV = 0.0;
    bool radiation_thermal_subcycle = false;
    HydroConfig hydro;
    ConductionConfig conduction;
    AleConfig ale;
    PlicConfig plic;
    MaterialsSubConfig materials;
    Ale1dConfig ale1d;
    ProfileConfig profile;
    FloorsConfig floors;
    bool positivity_clamp = true;
    SafetyConfig safety;
    DiagnosticsConfig diagnostics;
    int diagnostics_every = 1;
  };

  struct OutputConfig {
    std::string directory = "./output";
    std::string format = "hdf5";
    int plot_every = 100;
    int history_every = 1;
    int checkpoint_every = 1000;
    double plot_every_s = -1.0;
    double history_every_s = -1.0;
    double checkpoint_every_s = -1.0;
    bool plot_every_explicit = false;
    bool history_every_explicit = false;
    bool checkpoint_every_explicit = false;
    int checkpoint_keep_last = 2;
    std::string compression = "gzip";
    int compression_level = 4;
    bool save_namelist_copy = true;
    bool save_frozen_config = true;
    std::vector<std::string> plot_fields = {"rho", "Te", "Ti"};
  };

  struct DiagnosticsConfig {
    struct EnergyBudget {
      bool enabled = true;
      double warn_threshold = 1e-3;
    };
    struct ArealDensity {
      bool enabled = true;
      std::string r_range = "shell";
      std::vector<double> angles_deg = {0.0, 45.0, 90.0};
    };
    struct Sphericity {
      bool enabled = true;
      std::string surface = "isodensity";
      double rho_threshold = 10.0;
      std::vector<int> modes = {0, 2, 4};
    };
    struct LaserPattern {
      bool enabled = true;
      bool absorbed_power_profile = true;
      bool critical_surface = true;
      bool per_beam = false;
    };
    struct McStats {
      bool enabled = true;
      bool particle_counts = true;
      bool weight_stats = true;
      bool cell_particle_density = false;
      bool ddmc_fraction = true;
    };
    struct FleckDiag {
      bool enabled = false;
      int every = 10;
      std::vector<int> cells;
      double r_min_cm = -1.0;
      double r_max_cm = -1.0;
    };

    bool enabled = true;
    int every = 1;
    bool per_operator_radial_fourier_enabled = false;
    double radial_fourier_window_t_start_s = 1.35e-5;
    double radial_fourier_window_t_end_s = 1.70e-5;
    int radial_fourier_max_mode = -1;
    bool per_operator_radial_fourier_complex_enabled = false;
    std::vector<int> per_operator_radial_fourier_complex_m_targets = {14, 15, 16};
    std::vector<int> per_operator_radial_fourier_complex_j_targets = {
        507, 508, 509, 510, 511};
    std::vector<std::string> per_operator_radial_fourier_complex_fields = {
        "rho", "M", "V", "M_over_V", "P_r", "P_z", "u_r", "u_z",
        "E_e", "E_i", "E_rad", "T_e", "T_i", "x_r", "x_z",
        "A_r", "A_z", "Q_visc", "f_Fleck"};
    EnergyBudget energy_budget;
    ArealDensity areal_density;
    Sphericity sphericity;
    LaserPattern laser_pattern;
    McStats mc_stats;
    FleckDiag fleck_diag;
    bool overshoot_monitor = true;
  };

  struct ParallelConfig {
    struct Decomposition {
      std::string method = "slab";
      std::vector<int> dims;
      int min_cells_per_rank = 8;
    };
    struct Halo {
      std::string gpu_aware_mpi = "auto";
      int ghost_layers = 1;
    };
    struct Migration {
      std::string method = "batch";
      int max_substeps = 32;
      int emigrant_threshold = 1000;
      int initial_capacity = 10000;
      double growth_factor = 1.5;
    };

    Decomposition decomposition;
    Halo halo;
    Migration migration;
  };

  struct MetaConfig {
    int schema_version = 25;
    std::string namelist_source_path;
    std::string namelist_source_dir;
    std::string namelist_source_hash = "unavailable";
    // Canonical namelist freeze JSON for checkpoint compatibility checks.
    std::string frozen_config_json;
  };

  MainConfig main;
  MeshConfig mesh;
  MaterialsConfig materials;
  GeometryConfig geometry;
  RadiationConfig radiation;
  LaserConfig laser;
  NumericsConfig numerics;
  OutputConfig output;
  DiagnosticsConfig diagnostics;
  ParallelConfig parallel;
  BurnConfig burn;
  MetaConfig meta;
};

}  // namespace tenryu::core
