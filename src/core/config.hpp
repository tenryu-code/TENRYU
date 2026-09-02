#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "core/namelist/errors.hpp"
#include "hydro/bc_2d_rz_semantics.hpp"
#include "hydro/pressure_drive_perturbation.cuh"

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
  MULTIBLOCK_POLAR_TIER,
  MULTIBLOCK_POLAR_TIER_CART_CENTER,
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
  MimeticTensorV1,
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
    case AvModel::MimeticTensorV1:
      return "mimetic_tensor_v1";
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
  if (value == "mimetic_tensor_v1") {
    model = AvModel::MimeticTensorV1;
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
      // 2026-07-26 review: graded width mapping.
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
    struct ZoningIntentPinNL {
      double r = 0.0;
      bool ratio_jump_allowed = false;
    };
    struct ZoningIntentProfilePointNL {
      double r = 0.0;
      double w = 0.0;
    };
    struct ZoningIntentAnchorNL {
      double r = 0.0;
      double half_width = 0.0;
      double log_amplitude = 0.0;
    };
    struct ZoningIntentBandNL {
      double measure_frac_begin = 0.0;
      double measure_frac_end = 0.0;
      double cell_measure_min = 0.0;
      double cell_measure_max = 0.0;
    };
    struct ZoningIntentDensityRegionNL {
      double r_end = 0.0;   // outer radius of the constant-density region [cm]
      double rho = 0.0;     // [g/cc]
    };
    // Declarative high-freedom 1D zoning (measure equidistribution under hard
    // constraints); mutually exclusive with auto_regions and grid dicts.
    struct ZoningIntentNL {
      bool enabled = false;
      int n_cells = 0;
      std::string measure = "width";  // width|areal_mass|cylindrical_line_mass|spherical_cell_mass
      std::vector<ZoningIntentPinNL> pins;
      std::vector<ZoningIntentProfilePointNL> profile;
      std::vector<ZoningIntentAnchorNL> anchors;
      std::vector<ZoningIntentBandNL> bands;
      std::vector<ZoningIntentDensityRegionNL> density_regions;
      std::vector<double> extra_events;
      double dr_min = 0.0;
      double cell_measure_min = 0.0;
      double cell_measure_max = 0.0;
      double preferred_ratio = 1.3;
      double ratio_hard_max = 2.0;
      int min_cells_per_segment = 1;
    };
    ZoningIntentNL zoning_intent;
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
    // Outward node-ring counter with the fan first ring at zero; see the
    // transition_face_indices.push_back() schedule in make_polar_tier_layout().
    int polar_tier_cart_cut_ring = -1;
    std::string polar_tier_center_kind = "cart_box";
    std::string multiblock_cart_core_bridge_grading = "uniform";
    double multiblock_cart_core_bridge_spacing_floor = 0.0;
    double multiblock_cart_core_bridge_ratio_max = 1.05;
    double multiblock_theta_cap_widen_factor = 1.0;
    MultiblockTransitionScheme multiblock_transition_scheme =
        MultiblockTransitionScheme::HERMITE_BRIDGE;
    double multiblock_cap_p = 6.0;
    int multiblock_bridge_elliptic_sweeps = 0;
    double multiblock_bridge_elliptic_omega = 0.5;
    double polar_tier_chi_lo = 0.75;
    double polar_tier_chi_hi = 1.50;
    double polar_tier_belt_thickness_frac = 0.0;
    int polar_tier_belt_rows = 1;
    // The pole cap is active only when both m and alpha are positive.
    int polar_tier_pole_cap_m = 0;
    double polar_tier_pole_cap_alpha = 0.0;
    bool polar_tier_dendrite_enabled = false;
    bool polar_tier_native_pentagon = false;
    bool shell_polar_cap_dendrite = false;
    int shell_cap_rows_2x = 28;
    // T1 rows below S_theta on the 8-ladder.
    int polar_tier_dendrite_s_theta_rows_below = 5;
    int polar_tier_fan_sectors = 12;
    int polar_tier_min_tier_columns = 12;
    double polar_tier_fan_first_ring_radius_cm = 0.0;
    bool polar_tier_hydro_enabled = false;
    // Outer-shell tangent-balance fix: every-step tangential projection of corner Svec
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
      // 2026-07-11 review — docs/design/fleck_cv_default_flip_20260711.md).
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
      // Rosseland-opacity floor [1/cm] for the diffusion coefficient D = c*lambda/sigma.
      // A physically invisible floor (mfp = 10 km >> any lab-scale domain) that
      // regularizes the degenerate void limit: with sigma -> 0 and uniform E the
      // flux limiter stays at 1/3, D diverges, and the tridiagonal rows reach
      // ~1e98, which overflows both CR and QR solvers into NaN (2026-08-28 root
      // cause of the 550-cell zoning-deck cell-inversion crash). In gradient
      // regions lambda ~ 1/R makes D independent of sigma, so the floor only
      // acts on the degenerate direction.
      double opacity_floor = 1.0e-6;
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
      // Evaluation floor stays effectively zero: S_N transport is well-posed
      // at sigma = 0 and attenuation gates compare exp(-sigma L) directly.
      // The diffusion FORMS (DSA, AP-blend face flux) carry their own
      // 1e-6/cm denominator regularization (same rationale as the FLD
      // opacity_floor default, 2026-08-30).
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
      std::string langdon_model = "auto";         // auto | off | legacy_vacuum_map
      double langdon_te_min_eV = 100.0;
      bool langdon_auto_resolved = false;  // true when "auto" resolved to legacy_vacuum_map (not serialized)
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
      return (ib.zeff_model != "off" && ib.zeff_model != "auto") ||
             ib.coulomb_log_model != "debye" ||
             (ib.langdon_model != "off" && ib.langdon_model != "auto") || ra.enable ||
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

    struct ZReflectionConfig {
      // "off": no action. "audit": measure and log odd-parity defects at
      // stage boundaries, never modify state. Allowed modes are "off" and
      // "audit" only.
      // "enforce" was removed 2026-08-31 (user ruling: no answer-shaping state corrections).
      // Only legal with an exactly even problem (static eligibility below).
      std::string mode = "off";
    };
    ZReflectionConfig z_reflection;

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
      struct RefinementEstimator {
        bool enabled = false;
        int every = 25;              // steps between evaluations
        double filter_eps = 0.01;    // Lohner ripple filter
        double detect_cutoff = 0.8;  // front-band membership threshold
      };
      struct RefinementAutopilot {
        bool enabled = false;        // SHADOW mode: observe+log only
        std::string mode = "shadow";  // "shadow" (log only) | "arm_exit"
                                      // (request checkpoints in-window,
                                      // emit decision record, stop run)
        double ckpt_lead_h = 8.0;     // start checkpoint requests when
                                      // d_clear_h <= window_hi_h + this
        double e_on = 0.88;          // detection on-threshold (median column peak E)
        double e_off = 0.60;         // detection off-threshold
        double assoc_cut = 0.30;     // component membership cut
        double strong_cut = 0.60;    // innermost-component peak requirement
        int gap_bridge = 5;          // stitched-column gap tolerance (block rims)
        int persist = 3;             // consecutive observations for promotions
        double n_q_plan = 3.3;       // planning AV-layer width [cells]
        double chi_design = 0.018;   // design birth-quality target
        double s_rep_cm = 16.1e-4;   // replacement interface radius
        double handoff_cm = 4.5e-4;  // 2D->1D handoff radius
        double window_lo_h = 12.0;   // preferred firing window [parent cells]
        double window_hi_h = 24.0;
        int history = 9;             // front-observation ring buffer length
        double cov_min = 0.5;        // minimum tracked-column fraction
      };
      struct EvacuatedCellShadow {
        bool enabled = false;
        int every_n_steps = 250;              // >= 1
        double arm_mass_fraction = 1.0e-6;    // (0,1), > off_mass_fraction
        double off_mass_fraction = 1.0e-8;    // (0,1)
        double rho_vacuum_policy_g_per_cc = 1.0e-10;  // > 0
        double laser_wavelength_nm = 351.0;   // > 0 (n_crit estimate for the gate value)
        double laser_ne_over_ncrit_max = 1.0e-3;      // > 0 (logged threshold, no action)
      };
      struct AleProvenanceEmissionConfig {
        bool enabled = false;
      };
      struct ConductionEnergyRateExportConfig {
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
        // W1 (asym arc 2026-07-21): per-sector tracking. 0 = off (legacy
        // theta-averaged scalar path only). When > 0: that many equal-theta
        // sectors over [0, pi], each with its own ridge + quadratic tracker;
        // a solid-angle+confidence-weighted Legendre fit of ln s_f up to
        // modal_l_max; and the earliest confidence-bounded global morph
        // deadline (design doc asym_runtime_ale_controller_20260721 §2).
        int sectors = 0;                  // 0 (off) or >= 4, even
        int modal_l_max = 4;              // 0..4
        double sector_confidence_nu = 2.75;   // nu in t_arr - nu*sigma_t
        double sector_guard_crossings = 9.0;  // N_g in N_g * tau_c
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
      RefinementEstimator refinement_estimator;
      RefinementAutopilot refinement_autopilot;
      EvacuatedCellShadow evacuated_cell_shadow;
      AleProvenanceEmissionConfig ale_provenance_emission;
      ConductionEnergyRateExportConfig conduction_energy_rate_export;
      MeshQualityMinConfig mesh_quality_min;
      AleVelCoherenceConfig ale_velcoherence;
      ProductionAuditConfig production_audit;
      MeshDegeneracyForensicsConfig mesh_degeneracy_forensics;
    };

    struct DtConfig {
      double initial_s = 1e-15;
      double cfl_hydro = 0.3;
      // 2D hydro-CFL characteristic length for non-button cells.
      // "sqrt_area" preserves the legacy path; "min_altitude" is opt-in.
      std::string cfl_length_2d = "sqrt_area";
      bool edge_accel_displacement_cfl_enabled = false;
      double cfl_cond = 0.25;
      double f_min_fleck = 0.01;
      double growth_factor = 1.2;
      double max_s = 1e-9;
      double min_s = 1e-20;
      // Number of CONSECUTIVE steps with dt < min_s required before the
      // dt-floor abort fires. 1 = abort on the first sub-floor step (legacy
      // behavior). A genuine finite-time wall produces a monotone sub-floor
      // run and still dies within K steps; a one-step corner-geometry
      // transient survives.
      int min_consecutive_steps = 1;
      // Consecutive committed-step floor-stall detector.
      // 0 = disabled (bit-exact baseline preserved). 32 is the production
      // policy decks may opt into.
      int floor_stall_max_consecutive_steps = 0;
    };

    struct HydroConfig {
      bool enabled = true;
      bool compatible_energy = false;
      bool rho_e_linear_grid = false;
      bool eos_writeback = true;
      // EOS closure energy policy. "energy_authoritative" (default
      // since the 2026-07-18 user ruling; production re-baselined) keeps the
      // evolved energy on table-edge inverse clamps and routes all
      // table-ceiling evaluations through the high-T ideal tail;
      // "legacy" restores the pre-fix projection behavior (deletes the
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
          double drive_t_end_s = std::numeric_limits<double>::infinity();

          [[nodiscard]] bool is_state_supply() const {
            return type == "state_supply";
          }

          [[nodiscard]] bool supply_active(const double t) const {
            return is_state_supply() && t < drive_t_end_s;
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
      // builder (2026-08-03 AV modernization: limited CSW98 —
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
      // Axis-node (r==0) nodal-mass convention for the multiblock CSR
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
      double csw98_degenerate_side_floor_rel = 1.0e-2;
      double csw98_damper_impulse_beta = 0.0;
      std::string csw98_axisline_av_mode = "off";
      bool csw98_axisline_d1prime_cfl_enabled = true;
      bool csw98_limiter_shock_floor_enabled = false;
      bool csw98_axisline_work_planar_enabled = false;
      double tensor_av_C1 = 1.0;
      double tensor_av_C2 = 1.0;
      double av_qcap_over_p = 0.0;
      bool av_qcap_center_band_only = false;
      double av_cfl_coefficient = 0.25;
      double csw_C1 = 0.5;
      double csw_C2 = 2.0;
      std::string csw_limiter = "van_leer";
      bool csw_limiter_enabled = true;
      bool csw_axis_mirror_limiter = false;
      bool csw_rz_lift_enabled = false;
      double csw_rz_lift_guard_ratio = 4.0;
      bool csw_pole_floor_enabled = false;
      double csw_pole_floor_sigma0 = 1.0;
      double csw_pole_floor_theta0_rad = 0.033;
      double csw_pole_floor_thetaf_rad = 0.033;
      bool csw_pole_desens_enabled = false;
      double csw_pole_desens_alpha = 0.25;
      double csw_pole_desens_theta0_rad = 0.033;
      double csw_pole_desens_thetaf_rad = 0.033;
      bool csw_polar_slaving_enabled = false;
      int csw_polar_slaving_min_columns = 96;
      int csw_polar_slaving_full_columns = 4;
      int csw_polar_slaving_outer_columns = 6;
      double csw_polar_slaving_chi_on = 0.08;
      double csw_polar_slaving_chi_full = 0.20;
      double csw_polar_slaving_strength = 1.0;
      bool csw_polar_slaving_av_stiffness_cfl_enabled = false;
      double csw_polar_slaving_av_stiffness_sigma = 0.8;
      bool wake_heat_flux_enabled = false;
      double wake_heat_flux_CE = 0.10;
      double wake_heat_flux_theta_a_rad = 0.0327;
      double wake_heat_flux_theta_b_rad = 0.0982;
      bool wake_heat_flux_global_theta = false;
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
      // 2026-07-26 review: 1D node-crossing timestep guard.
      // Limits the step so one cell face cannot close more than this fraction
      // of the current cell width per step (raw geometric bound, not scaled
      // by cfl_hydro). 0 disables. Never binds while the acoustic+AV
      // denominator dominates (safety >= cfl_hydro / av_quadratic).
      double crossing_dt_safety = 0.5;
      // 2026-07-26 review: 1D hydro time integrator.
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
      bool pentagon_affine_null_enabled = true;
      double pentagon_affine_null_kappa = 0.03;
      bool subzonal_dt_limiter_enabled = true;
      bool aw_compatible_force_work = false;
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
      struct AxisProjectionConfig {
        bool enabled = false;
        bool shadow_only = true;
        double q_on = 0.20;
        double q_floor = 0.05;
        int patch_halfwidth = 2;
        int log_every_n_steps = 0;
      } axis_projection;
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
        // 2026-07-26 review: physical hysteresis time scale [s].
        // > 0 replaces the per-step fixed blend hysteresis_w with
        // w(dt) = 1 - exp(-dt/tau) so the gate relaxation rate is
        // timestep-refinement invariant. 0 keeps the legacy fixed-w blend.
        double hysteresis_tau = 0.0;
        int support_ahead = 1;
        int support_behind = 10;
      } adaptive_av;
      // Braginskii ion shear viscosity (NUMERICS §3.1.x) for the 1D
      // Lagrangian step. Default disabled => bit-identical trajectories.
      // model: "braginskii" (eta0 = 0.96 n_i kT_i tau_i, NRL tau_i/lnLambda_ii,
      // Mason-2014 mfp cap) | "constant" (uniform eta_const, verification).
      // species: "ion" (legacy) | "electron" | "both" (additive eta_i + eta_e).
      struct PlasmaViscosity {
        bool enabled = false;
        std::string model = "braginskii";
        // Electron-viscosity lane (2026-07-12): which Braginskii channel(s)
        // feed pi = -eta_eff W. "ion" = ion-only legacy (bit-preserving default),
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
      struct PressureDrivePerturbationConfig {
        bool enabled = false;
        // Resolved at namelist build time (explicit modes + drawn random modes).
        std::vector<int> mode_l;
        std::vector<double> mode_a;
        // Gaussian ring spots (theta0_rad, sigma_rad, amplitude).
        std::vector<double> spot_theta0;
        std::vector<double> spot_sigma;
        std::vector<double> spot_amp;
        // Random zonal spectrum inputs (kept for provenance/logging).
        bool random_enabled = false;
        long long random_seed = 0;
        int random_l_min = 2;
        int random_l_max = 8;
        double random_rms = 0.0;
        // Computed at build time over the endpoint-inclusive theta grid.
        double g_min = 1.0;
        double g_max = 1.0;
      } pressure_drive_perturbation;
      std::map<std::string, CallableInfo> pressure_drive_2d;
    };

    struct ConductionConfig {
      bool enabled = true;
      std::string solver = "sts";  // "sts" | "implicit" | "hypre"
      // "net" (legacy): alpha from net power, pairs scaled by min(alpha_c, alpha_nb).
      // "donor": alpha from the outflow sum, each pair scaled by its donor's alpha —
      // guarantees Te_trial >= floor (Te-floor guarantee fix, docs/design/bug18_...20260712.md).
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
      std::string mesh_mode = "fixed";
      std::string reale_core = "exact";
      double rezone_min_dt_s = 1.0e-14;
      bool tess_gpu_dual = false;
      bool tess_gpu_restrict = false;
      int dvclp_solver_rev = 0;
      int reale_lloyd_max = 4;
      double reale_short_edge_collapse_rel = 3.0e-2;
      bool reale_subdomain_rezone = false;
      double reale_subdomain_frac_max = 0.6;
      double reale_overlay_additivity_tol = 1.0e-4;
      bool reale_corner_mass_reset = true;
      bool reale_velocity_max_principle = false;
      double reale_dt_trigger_factor = 0.5;
      int reale_dt_trigger_cooldown = 0;
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
      // W2 (asym arc 2026-07-21): read-only runtime mesh-health monitor.
      // Evaluates, over the core+bridge (+shell_rows innermost shell rows)
      // region, the scale-independent corner quality Q_c = 2 J_c /
      // (|e1|^2 + |e2|^2) normalized by a healthy-reference snapshot, and the
      // predicted corner-J first-root horizon H_c = t_root / dt_acoustic, and
      // classifies OFF/WARNING/SOFT/HARD/RECOVERY with hysteresis
      // (design doc asym_runtime_ale_controller_20260721 §2). Diagnostic only.
      struct RuntimeControllerConfig {
        bool monitor_enabled = false;
        int monitor_every = 1;      // >= 1 evaluation cadence (steps)
        int shell_rows = 4;         // >= 0 innermost structured shell rows included
        int controller_shell_rows = 4; // >= 0 innermost shell rows in controller mandate
        int cap_columns = 2;        // >= 0 theta cells per axis using the radial cap target
        double q_soft = 0.42;       // (0,1]; entry: Qmin < q_soft
        double q_hard = 0.22;       // (0,q_soft)
        double q_recover = 0.60;    // (q_soft,1]
        double h_soft = 8.0;        // > 0 acoustic steps; entry: Hmin < h_soft
        double h_hard = 4.0;        // (0,h_soft)
        double h_recover = 12.0;    // > h_soft
        int soft_persistence = 2;   // >= 1 consecutive evaluations for a
                                    // Q-triggered SOFT entry (H-triggered and any
                                    // HARD entry are immediate)
        int recover_checks = 3;     // >= 1 consecutive healthy evaluations
        // W3a target construction (asym_runtime_ale_controller_20260721 §2).
        int winslow_sweeps = 4;             // 1..16 fixed virtual Jacobi sweeps
        double winslow_omega = 0.2;         // (0,1] relaxation passed to the smoother
        double beta_monitor_soft = 0.12;    // [0,1) monitor-target blend weight (soft)
        double beta_monitor_hard = 0.25;    // [0,1) monitor-target blend weight (hard)
        double beta_mass = 0.12;            // [0,0.5] rho*s^2 monitor term weight
        double beta_front = 0.12;           // [0,0.5] front-Gaussian monitor term weight
        double beta_theta = 0.15;           // [0,0.5] angular equal-angle restoration
        double g_max = 1.22;                // (1,1.6) adjacent radial spacing ratio cap
        double front_width_cells = 2.5;     // [1,8] w_f in local radial cells
        double cap_fraction = 0.05;         // (0,0.2] |dx| <= cap_fraction * h_p
        double cap_normal_fraction = 0.025; // (0,cap_fraction] shock-band normal cap
        // W3b controller (motion) — requires monitor_enabled; default off.
        bool controller_enabled = false;
        bool commit_rollback_enabled = true;
        // W3c: the controller is a post-crossing mechanism. Events are allowed
        // only once the selected sector-front statistic has passed the seam
        // inward by this many seam spacings (requires the sector detector); when
        // the detector is off, activation_time_s > 0 gates by time instead
        // (<=0: controller stays inactive without the detector).
        std::string activation_front_mode = "min";  // "min" or "mean"
        double activation_front_margin_hs = 2.0;   // >= 0
        double activation_time_s = -1.0;
        int cadence_soft = 2;        // >= 1 steps per rezone event in SOFT
        int cadence_hard = 1;        // >= 1, <= cadence_soft
        int cadence_recovery = 4;    // >= cadence_soft
        bool pre_step_enabled = true;    // run the event BEFORE the hydro step in HARD
        int failures_hard_force = 4;
        int failures_big_repair = 8;
        int escalation_max_failures = 12; // consecutive failed transactions
                                         // before the controller disengages
                                         // (dt guard keeps the run safe)
      } runtime_controller;
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
      bool mesh_epoch_enabled = false;  // accumulate accepted pre-hydro mesh repairs into the retry snapshot
      int mesh_epoch_max_per_step = 16;
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
        std::string role = "";
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
        double t_on_s = 0.0;
        double t_off_s = -1.0;
        int feather_min_layers = 3;
        int guard_layers = 1;
        std::string axis_core_transaction_mode = "static";
        std::string replay_table_path;
        double replay_tau_lead = 4.5e-12;
        double replay_tau_splice = 1.4e-12;
        double replay_beta = 1.0;
        bool axis_core_transition_passage_enabled = false;
        bool axis_core_ring_release_enabled = false;
      } euler_window;
      std::vector<EulerWindowConfig> euler_windows;
      struct BandAleConfig {
        bool enabled = false;
        double aspect_trigger = 0.25;
        double release_hysteresis = 1.5;
        double chi = 0.5;
        // Per-application node displacement cap for radial respace
        // targets, as a fraction of the local adjacent ring gap along
        // the node's spoke. Keeps every remap application sub-cell so
        // donor sweeps never cross a steep profile in one transaction;
        // the band re-engages on later steps until converged.
        double respace_move_cap_frac = 0.5;
        // Physical compression hold for estimator-band respace
        // installs: the band holds while any window cell has radial
        // velocity jump exceeding this Mach fraction of its sound
        // speed (the same compression measure as the pole-theta and
        // catchment holds). 0 disables the gate.
        double estimator_band_hold_mach = 0.30;
        std::string bands = "belts_axis";
        bool compose_with_rezone = false;
        std::string belt_target = "ring_mean";
        std::string center_target = "line";
        std::string axis_target = "z_laplacian";
        int axis_segment_halfwidth = 4;
        bool axis_shell_block_enabled = false;
        bool sigma_linesearch_enabled = true;
        bool transaction_energy_closure_enabled = false;
        double estimator_band_cut = 0.5;   // row-mean refine_error threshold
        double estimator_band_shock_hold = 0.90;  // hold the band while any
                                                  // window row-mean E >= this
        double estimator_band_front_hold_margin_rows = 16.0;  // §18.5: hold
        // the estimator band while the tracked front (median shell row) is
        // within this many rows of the window [first_ring, last_ring].
        std::string estimator_band_axis = "auto";  // §18.6: lattice axis the
        // band is layered along on SINGLE-BLOCK meshes. "auto": "i" (radial,
        // slow index) when Mesh.logical_mesh_2d starts with
        // "spherical_polar", else "j" (the rectangular laser slab ablates
        // along z = the fast index). Multiblock ignores it (always the
        // shell's radial i).
        int estimator_band_in_rows = 4;    // rows added inside flagged interval
        int estimator_band_out_rows = 4;   // rows added outside
        double estimator_band_eta_on = 1.5;   // spacing-nonuniformity engage
        double estimator_band_eta_off = 1.15; // release
        bool estimator_band_per_column = false;
        int estimator_band_pc_filter_halfwidth = 3;
        double estimator_band_pc_slope_limit = 0.35;
        double estimator_band_pc_slope_reject = 0.50;
        double estimator_band_pc_curvature_limit = 0.50;
        double estimator_band_pc_chi_max = 0.25;
        double estimator_band_pc_chi_step = 0.10;
        double estimator_band_pc_sigma_floor = 0.25;
        double estimator_band_pc_coverage_full = 0.80;
        double estimator_band_pc_coverage_min = 0.50;
        int estimator_band_pc_cooldown_events = 2;
        bool estimator_band_pc_phase_b = false;
        int estimator_band_pc_tube_dilate_rows = 5;
        int estimator_band_pc_tube_dilate_cols_extra = 2;
        double estimator_band_pc_ambiguous_hold_fraction = 0.20;
        bool closure_catchment_enabled = false;
        bool closure_catchment_forced_active = false;  // increment 1 only
        // Full-respace outer radius and outer identity radius [cm].
        double closure_catchment_s_catch_cm = 6.0e-4;
        double closure_catchment_s_protect_cm = 1.0e-3;
        double closure_catchment_spacing_floor_cm = 3.0e-6;
        double closure_catchment_ratio_max = 1.05;
        double closure_catchment_nu_max = 0.25;
        int closure_catchment_max_bites = 1;
        // Center of the continuous compression-Mach shock attenuation band;
        // <= 0 disables the factor.
        double closure_catchment_shock_hold = 0.30;
        // v2 common-mode lane (consult 2026-08-29). Relative comoving
        // engagement metrics with EMA collapse-rate forecast; quintic
        // smoothstep activations; physical-time cadence.
        double closure_catchment_eta_h_arm = 0.80;
        double closure_catchment_eta_h_full = 0.65;
        double closure_catchment_eta_m_arm = 0.75;
        double closure_catchment_eta_m_full = 0.60;
        double closure_catchment_reset_eta = 0.90;
        int closure_catchment_support_core_rows = 4;
        int closure_catchment_support_taper_rows = 2;
        double closure_catchment_accum_frac = 0.02;
        double closure_catchment_rearm_drop = 0.10;
        // Pole-theta maintenance (consult 2026-08-28): transactional
        // tangential re-equidistribution of near-pole node rows, triggered
        // by the target-normalized subzonal angular height H and the
        // target-relative Jacobian condition number kappa.
        bool pole_theta_enabled = false;
        // v2 routine polar lane: low-pass physical-reference band and
        // noise-normalized activation. The routine correction acts on
        // the high-pass angular-pitch defect d = ln(dtheta /
        // dtheta_phys); rows activate when the solid-angle L8 defect
        // exceeds noise_floor by 2x (full at 4x).
        // v2 routine polar lane master switch. Measured 2026-08-30: in
        // every exercised regime the routine lane is either unnecessary
        // (pre-stagnation stays in the LSB noise band without it) or harmful
        // (during stagnation it re-equalizes pitches against the physically
        // draining polar channels and pumps theta asymmetry through its pass
        // band); the hard lane, catchment, and pool carry all demonstrated
        // mandates. Kept for future regimes that demonstrate need.
        bool pole_theta_routine_enabled = false;
        int pole_theta_phys_lp = 0;
        int pole_theta_phys_lc = 4;
        double pole_theta_noise_floor = 1.0e-5;
        // Structural ceiling of the routine lane: rows whose high-pass
        // pitch defect exceeds this are treated as real structure (or
        // hard-lane pathology), not removable noise, and the routine
        // correction fades to exactly zero (smoothstep down between
        // 1x and 2x the ceiling).
        double pole_theta_noise_ceiling = 1.0e-3;
        double pole_theta_h_arm = 0.70;
        double pole_theta_h_fire = 0.55;
        double pole_theta_h_hard = 0.35;
        double pole_theta_h_release = 0.80;
        double pole_theta_kappa_arm = 2.5;
        double pole_theta_kappa_fire = 3.5;
        double pole_theta_kappa_hard = 6.0;
        double pole_theta_alpha = 0.30;
        double pole_theta_alpha_hard = 0.60;
        double pole_theta_deadband_frac = 0.05;
        double pole_theta_move_limit_frac = 0.35;
        double pole_theta_move_limit_hard_frac = 0.75;
        double pole_theta_cooldown_s = 1.0e-12;
        // Hard-lane base physical-time cooldown between committed
        // coupled fires; futility debt doubles it per unit of debt
        // (consult 7). 0 disables the cooldown/futility control.
        double pole_theta_cooldown_base_s = 1.0e-12;
        double pole_theta_predict_window_s = 2.0e-12;
        double pole_theta_predict_horizon_s = 5.0e-12;
        int pole_theta_halo_columns = 2;
        int pole_theta_halo_rows = 2;
        double pole_theta_post_h_floor = 0.65;
        // M2 curve-preserving target: reconstruct each node row as a
        // star-shaped curve s(theta) and move nodes along it. false =
        // the M1 radius-preserving rotation (spherical-legal).
        bool pole_theta_curve_preserving = false;
        // Comma-separated Legendre orders whose row-geometry amplitudes
        // the reconstruction preserves exactly (deck-seeded drive modes;
        // l = 0 and 1 are always protected). Empty = only {0, 1}.
        std::string pole_theta_protected_modes = "";
        // Legendre fit order of the row reconstruction; clamped at
        // runtime to floor(ntheta / 4).
        int pole_theta_fit_order = 12;
        // Hold a pole's arming/firing while a shock transits its scan half: a
        // cell's compressive radial velocity jump u_r,inner - u_r,outer above
        // this threshold times the local sound speed holds the pole.
        // <= 0 disables the hold.
        double pole_theta_shock_hold = 0.30;
        int shell_window_in_rows = 8;
        int shell_window_out_rows = 4;
        int shell_boundary_guard_rows = 2;
        double shell_min_spacing_frac = 0.5;
        std::string shell_front_metric = "grad_rho";
        std::string shell_target = "respace";
        bool axis_repair_enabled = false;
        double axis_repair_eta_on = 0.85;
        double axis_repair_eta_off = 0.95;
        double axis_repair_cap_rel = 0.05;
      } band_ale;
      struct EvacuatedCellConfig {
        struct EvacuatedCellClosureContactConfig {
          bool enabled = true;
          double gap_floor_fraction = 0.02;
          double gap_arm_fraction = 0.04;
          double live_mass_gate = 0.05;
          double live_volume_gate = 0.02;
          double refill_min_mass_fraction = 0.05;
          double refill_min_density_ratio = 0.2;
          double release_force_c = 1.0e-3;
          int release_persistence_stages = 2;
          double reengage_gap_margin = 0.01;
          double mortar_position_drift_beta = 0.02;
          bool surface_engage_enabled = false;
          bool lcp_apply_enabled = false;
          struct AxisEdgeCollapseConfig {
            bool enabled = false;  // default off = bit-inert
            double ulp_count = 4096.0;
            double h_ref_fraction = 1.0e-6;
            double release_hysteresis = 4.0;
            int persistence_window = 8;
            int persistence_min_closing = 7;
            int repair_recurrence_steps = 16;
            double repair_futility_fraction = 0.25;
          };
          AxisEdgeCollapseConfig axis_edge_collapse;
          struct FlankTangentialStripConfig {
            bool enabled = false;  // default off = bit-inert
            bool untangler_enabled = true;
            int band_layers = 2;           // active strip rows (i = 1..band_layers)
            int band_halfwidth_j = 6;      // j window half-width around the slot cell
            double arm_quality_ratio = 0.25;    // q_theta/reference below this arms
            double release_quality_ratio = 0.60;
            double min_progress_factor = 10.0;  // accept a trial when the band quality ratio improves by at least this factor
            int lead_steps = 32;                // predicted steps-to-floor arming
            int release_persistence_steps = 32;
            double release_shear_number = 0.02; // dt*max|du_t/dn| release bound
            double slip_handoff_ratio = 0.5;    // |delta_s|/h_s handoff threshold
            bool slip_patch_enabled = false;    // default off = bit-inert
          };
          FlankTangentialStripConfig flank_tangential_strip;
          bool seam_interface_owner_enabled = false;  // default off = bit-inert
        };
        bool enabled = false;  // action mode; default off = bit-inert
        int every_n_steps = 50;
        double arm_mass_fraction = 1.0e-6;
        double off_mass_fraction = 1.0e-8;
        double rho_vacuum_policy_g_per_cc = 1.0e-10;
        int off_hold_evaluations = 2;  // consecutive OFF-eligible evaluations
        double laser_ne_over_ncrit_max = 1.0e-3;
        double laser_wavelength_nm = 351.0;
        double coupling_fraction_max = 1.0e-8;  // coupling energy vs patch energy
        int max_cells_per_event = 4;  // reject the event if more would convert
        bool rematerialize_enabled = true;  // part of the policy's correctness
        int rematerialize_after_evaluations = 10;
        double rematerialize_volume_fraction = 0.05;  // trigger: vol < frac * V_ref
        double rematerialize_neighbor_change_max = 5.0e-2;  // per-donor relative clamp
        int rematerialize_dwell_evaluations = 5;  // no re-conversion for this many evals
        EvacuatedCellClosureContactConfig closure_contact;
      };
      EvacuatedCellConfig evacuated_cell;
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
      // 2026-07 review: the corrected oriented convention
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
      double central_pseudo_core_activation_time_s = 0.0;
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
      int t0_volume_cut_max_depth = 6;  // range [4, 16] (was [8, 16] initially; 12 was too tight for hard step interfaces)
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
        int retrigger_cooldown_steps = 0;  // after a floor-triggered attempt is not applied, skip the floor-trigger evaluation for this many steps (0 = evaluate every step)
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
    // Write one snapshot at run termination when the last cadence write
    // did not land on the final step (opt-in; gates that count snapshots
    // rely on the historical cadence-only behavior).
    bool write_final_snapshot = false;
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

inline hydro::PressureDrivePerturbationParams
make_pressure_drive_perturbation_params(
    const Config::NumericsConfig::HydroConfig::
        PressureDrivePerturbationConfig& config) {
  hydro::PressureDrivePerturbationParams params;
  params.n_modes = static_cast<int>(config.mode_l.size());
  for (int k = 0; k < params.n_modes; ++k) {
    params.mode_l[k] = config.mode_l[static_cast<std::size_t>(k)];
    params.mode_a[k] = config.mode_a[static_cast<std::size_t>(k)];
  }
  params.n_spots = static_cast<int>(config.spot_theta0.size());
  for (int m = 0; m < params.n_spots; ++m) {
    params.spot_theta0[m] =
        config.spot_theta0[static_cast<std::size_t>(m)];
    params.spot_sigma[m] =
        config.spot_sigma[static_cast<std::size_t>(m)];
    params.spot_amp[m] = config.spot_amp[static_cast<std::size_t>(m)];
  }
  return params;
}

enum class PolarTierJoinKind {
  ONE_TO_ONE,
  TWO_TO_ONE,
};

struct PolarTierJoinDescriptor {
  PolarTierJoinKind kind = PolarTierJoinKind::ONE_TO_ONE;
  // Half-open interval-index ranges on the outer and inner rings.
  int outer_interval_begin = 0;
  int outer_interval_end = 0;
  int inner_interval_begin = 0;
  int inner_interval_end = 0;
};

enum class PolarTierShellBandKind {
  SHELL_C2,
  SHELL_T21,
  SHELL_FINE,
};

struct PolarTierShellBandDescriptor {
  PolarTierShellBandKind kind = PolarTierShellBandKind::SHELL_FINE;
  // Half-open radial row span in the shell, with row zero at r_match.
  int row_begin = 0;
  int row_end = 0;
  // Coarse-side cap leaf width in master-column units.
  int cap_leaf_width = 1;
  int cells_per_row_half = 0;
  int cells_per_row_full = 0;
  long long n_cells = 0;
  // Nodes first owned by this band after shared interface rings are elided.
  long long n_nodes = 0;
};

enum class PolarTierEntityKind {
  SHELL,
  TIER,
  TRANSITION,
  FAN,
};

struct PolarTierScheduleEntry {
  PolarTierEntityKind kind = PolarTierEntityKind::FAN;
  // (kind, in_shell_chain, index) is the unique emitted-block key. Family
  // tiers/body transitions use their vector index. Shell-chain entries use
  // their shell_chain position. The plain SHELL and FAN use -1.
  int index = -1;
  bool in_shell_chain = false;
  // Outward canonical node-ring span; cell rows are [ring_begin, ring_end).
  int ring_begin = 0;
  int ring_end = 0;
  // Angular cell count; transitions record the coarse-side count.
  int columns = 0;
  // Realized inner/outer node-ring radii [cm]. These are exact for FAN,
  // TIER, body-TRANSITION, and the plain SHELL. Shell-chain entries use NaN;
  // truncation cuts are always below the shell and do not need those radii.
  double r_inner = 0.0;
  double r_outer = 0.0;
};

struct PolarTierLayout {
  double h_r = 0.0;
  double fan_radius = 0.0;
  int block_count = 0;
  long long n_cells = 0;
  long long n_nodes = 0;
  std::vector<int> tier_columns;
  std::vector<int> transition_face_indices;
  std::vector<double> transition_radii;
  std::vector<double> transition_chi_fine;
  std::vector<double> transition_chi_coarse;
  std::vector<double> transition_outer_radii;
  std::vector<double> transition_inner_radii;
  std::vector<std::vector<int>> transition_intermediate_columns;
  std::vector<std::vector<double>> transition_intermediate_radii;
  std::vector<int> tier_radial_rows;
  std::vector<double> tier_outer_radii;
  std::vector<double> tier_inner_radii;
  std::vector<double> tier_radial_spacings;
  // Dendrite-only descriptors. Ring order is T1a, S_theta-inner/T1b,
  // B1-inner, T2, T3, T4, T5; the legacy shell uses the T1a labels. Transition
  // order is S_theta, B1, S_theta2, B2, B3, B4. Full-ring labels are stored
  // because mesh connectivity indexes all C+1 nodes. Geometry construction
  // evaluates the north half through master label 96 and mirrors the south
  // half.
  std::vector<int> dendrite_native_tier_columns;
  std::vector<int> dendrite_actual_tier_columns;
  // Structured block rows after transition rows are assigned.
  std::vector<int> dendrite_tier_radial_rows;
  std::vector<int> dendrite_ring_interval_counts;
  std::vector<std::vector<int>> dendrite_master_theta_node_labels;
  std::vector<std::vector<PolarTierJoinDescriptor>>
      dendrite_transition_joins;
  long long shell_n_cells = 0;
  long long shell_n_nodes = 0;
  // Shell ring-label order is C2, FINE. The only shell transition is T21.
  std::vector<std::vector<int>> shell_master_theta_node_labels;
  std::vector<std::vector<PolarTierJoinDescriptor>> shell_transition_joins;
  std::vector<PolarTierShellBandDescriptor> shell_chain;
  std::vector<PolarTierScheduleEntry> schedule;
};

struct PolarTierEntityCounts {
  long long n_cells = 0;
  long long n_nodes = 0;
  long long n_blocks = 0;
};

inline PolarTierEntityCounts polar_tier_schedule_entry_counts(
    const Config::MeshConfig& mesh,
    const PolarTierLayout& layout,
    const PolarTierScheduleEntry& entry) {
  const long long rows =
      static_cast<long long>(entry.ring_end - entry.ring_begin);
  TENRYU_ASSERT(rows >= 0 && entry.columns > 0,
                "polar-tier schedule entry dimensions must be positive");
  if (entry.in_shell_chain) {
    TENRYU_ASSERT(mesh.shell_polar_cap_dendrite && entry.index >= 0 &&
                      static_cast<std::size_t>(entry.index) <
                          layout.shell_chain.size(),
                  "polar-tier shell-chain schedule index mismatch");
    const PolarTierShellBandDescriptor& band =
        layout.shell_chain[static_cast<std::size_t>(entry.index)];
    const PolarTierEntityKind expected_kind =
        band.kind == PolarTierShellBandKind::SHELL_T21
            ? PolarTierEntityKind::TRANSITION
            : PolarTierEntityKind::SHELL;
    TENRYU_ASSERT(entry.kind == expected_kind &&
                      rows == band.row_end - band.row_begin &&
                      entry.columns == band.cells_per_row_full,
                  "polar-tier shell-chain schedule entry mismatch");
    return PolarTierEntityCounts{band.n_cells, band.n_nodes, 1LL};
  }
  if (entry.kind == PolarTierEntityKind::SHELL) {
    TENRYU_ASSERT(!mesh.shell_polar_cap_dendrite && entry.index == -1 &&
                      rows == mesh.nr,
                  "polar-tier shell schedule entry mismatch");
    return PolarTierEntityCounts{
        rows * entry.columns,
        (rows + 1LL) * (static_cast<long long>(entry.columns) + 1LL),
        1LL};
  }
  if (entry.kind == PolarTierEntityKind::TIER) {
    TENRYU_ASSERT(entry.index >= 0 && rows > 0,
                  "polar-tier tier schedule entry mismatch");
    return PolarTierEntityCounts{
        rows * entry.columns,
        rows * (static_cast<long long>(entry.columns) + 1LL),
        1LL};
  }
  if (entry.kind == PolarTierEntityKind::FAN) {
    TENRYU_ASSERT(entry.index == -1 && entry.ring_begin == 0 &&
                      entry.ring_end == 0,
                  "polar-tier fan schedule entry mismatch");
    return PolarTierEntityCounts{entry.columns, 1LL, 1LL};
  }

  TENRYU_ASSERT(entry.kind == PolarTierEntityKind::TRANSITION &&
                    entry.index >= 0 && rows > 0,
                "polar-tier transition schedule entry mismatch");
  if (mesh.polar_tier_dendrite_enabled) {
    TENRYU_ASSERT(
        static_cast<std::size_t>(entry.index) <
                layout.dendrite_transition_joins.size() &&
            static_cast<std::size_t>(entry.index + 1) <
                layout.dendrite_master_theta_node_labels.size(),
        "polar-tier dendrite transition schedule index mismatch");
    long long cells = 0;
    long long centers = 0;
    for (const PolarTierJoinDescriptor& join :
         layout.dendrite_transition_joins[
             static_cast<std::size_t>(entry.index)]) {
      if (join.kind == PolarTierJoinKind::ONE_TO_ONE) {
        cells += 1LL;
      } else {
        cells += mesh.polar_tier_native_pentagon ? 1LL : 5LL;
        centers += mesh.polar_tier_native_pentagon ? 0LL : 1LL;
      }
    }
    const long long nodes =
        static_cast<long long>(
            layout.dendrite_master_theta_node_labels[
                static_cast<std::size_t>(entry.index + 1)]
                .size()) +
        centers;
    return PolarTierEntityCounts{cells, nodes, 1LL};
  }

  const long long coarse = entry.columns;
  if (rows == 1) {
    return PolarTierEntityCounts{5LL * coarse,
                                 2LL * coarse + 1LL, 1LL};
  }
  if (rows == 2) {
    return PolarTierEntityCounts{5LL * coarse,
                                 5LL * coarse / 2LL + 2LL, 1LL};
  }
  TENRYU_ASSERT(rows == 3,
                "polar-tier plain transition must span one to three rows");
  return PolarTierEntityCounts{6LL * coarse, 4LL * coarse + 3LL, 1LL};
}

inline PolarTierEntityCounts polar_tier_schedule_totals(
    const Config::MeshConfig& mesh,
    const PolarTierLayout& layout) {
  PolarTierEntityCounts totals;
  for (std::size_t entity = 0; entity < layout.schedule.size(); ++entity) {
    const PolarTierScheduleEntry& entry = layout.schedule[entity];
    for (std::size_t prior = 0; prior < entity; ++prior) {
      const PolarTierScheduleEntry& candidate = layout.schedule[prior];
      TENRYU_ASSERT(candidate.kind != entry.kind ||
                        candidate.in_shell_chain != entry.in_shell_chain ||
                        candidate.index != entry.index,
                    "polar-tier schedule block key must be unique");
    }
    const PolarTierEntityCounts counts =
        polar_tier_schedule_entry_counts(mesh, layout, entry);
    totals.n_cells += counts.n_cells;
    totals.n_nodes += counts.n_nodes;
    totals.n_blocks += counts.n_blocks;
  }
  return totals;
}

struct PolarTierTruncation {
  int cut_ring = -1;    // Echo of the input.
  int n_theta_cut = 0;  // Angular cell count of the tier containing the cut.
  double r_cut = 0.0;   // Realized radius of the cut node ring [cm].
  long long dropped_cells = 0;
  // Nodes strictly below the cut ring; cut-ring nodes are kept.
  long long dropped_nodes = 0;
  long long dropped_blocks = 0;
};

inline void validate_polar_tier_dendrite_config(
    const Config::MeshConfig& mesh) {
  if (mesh.shell_cap_rows_2x < 12 || mesh.shell_cap_rows_2x > 288 ||
      mesh.shell_cap_rows_2x == 287) {
    throw namelist::ConfigError(
        "Mesh.shell_cap_rows_2x must be in [12, 286] or equal 288");
  }
  if (mesh.shell_polar_cap_dendrite &&
      !mesh.polar_tier_dendrite_enabled) {
    throw namelist::ConfigError(
        "Mesh.shell_polar_cap_dendrite=true requires "
        "Mesh.polar_tier_dendrite_enabled=true");
  }
  if (mesh.shell_polar_cap_dendrite &&
      !mesh.polar_tier_native_pentagon) {
    throw namelist::ConfigError(
        "Mesh.shell_polar_cap_dendrite=true requires "
        "Mesh.polar_tier_native_pentagon=true");
  }
  if (mesh.polar_tier_native_pentagon &&
      !mesh.polar_tier_dendrite_enabled) {
    throw namelist::ConfigError(
        "Mesh.polar_tier_native_pentagon=true requires "
        "Mesh.polar_tier_dendrite_enabled=true");
  }
  if (!mesh.polar_tier_dendrite_enabled) {
    return;
  }
  if (mesh.shell_polar_cap_dendrite && mesh.nr != 288) {
    throw namelist::ConfigError(
        "Mesh.shell_polar_cap_dendrite=true requires Mesh.nr == 288");
  }
  if (mesh.topology_scheme != TopologyScheme::MULTIBLOCK_POLAR_TIER &&
      mesh.topology_scheme !=
          TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER) {
    throw namelist::ConfigError(
        "Mesh.polar_tier_dendrite_enabled=true requires "
        "Mesh.topology_scheme=\"multiblock_polar_tier\" or "
        "\"multiblock_polar_tier_cart_center\"");
  }
  if (mesh.nz != 192) {
    throw namelist::ConfigError(
        "Mesh.polar_tier_dendrite_enabled=true requires Mesh.nz "
        "(Ntheta) == 192");
  }
  if (mesh.polar_tier_min_tier_columns != 12) {
    throw namelist::ConfigError(
        "Mesh.polar_tier_dendrite_enabled=true requires "
        "Mesh.polar_tier_min_tier_columns == 12");
  }
  if (mesh.polar_tier_fan_sectors != 12) {
    throw namelist::ConfigError(
        "Mesh.polar_tier_dendrite_enabled=true requires "
        "Mesh.polar_tier_fan_sectors == 12");
  }
  if (mesh.polar_tier_belt_rows != 1) {
    throw namelist::ConfigError(
        "Mesh.polar_tier_dendrite_enabled=true requires "
        "Mesh.polar_tier_belt_rows == 1");
  }
  if (mesh.polar_tier_belt_thickness_frac != 0.0) {
    throw namelist::ConfigError(
        "Mesh.polar_tier_dendrite_enabled=true requires "
        "Mesh.polar_tier_belt_thickness_frac == 0");
  }
  if (mesh.polar_tier_pole_cap_m != 0) {
    throw namelist::ConfigError(
        "Mesh.polar_tier_dendrite_enabled=true requires the old pole-cap "
        "map off (Mesh.polar_tier_pole_cap_m == 0)");
  }
}

inline PolarTierLayout make_polar_tier_layout(
    const Config::MeshConfig& mesh) {
  validate_polar_tier_dendrite_config(mesh);
  PolarTierLayout layout;
  constexpr double pi =
      3.1415926535897932384626433832795028841971693993751058209749;
  if (mesh.nz <= 0 || mesh.nr <= 0 ||
      mesh.polar_tier_min_tier_columns <= 0 ||
      mesh.polar_tier_fan_sectors <= 0 ||
      (mesh.polar_tier_belt_rows != 1 &&
       mesh.polar_tier_belt_rows != 2 &&
       mesh.polar_tier_belt_rows != 3) ||
      (mesh.polar_tier_pole_cap_m != 0 &&
       (mesh.polar_tier_pole_cap_m < 4 ||
        mesh.polar_tier_pole_cap_m > 48)) ||
      !(std::isfinite(mesh.polar_tier_pole_cap_alpha) &&
        mesh.polar_tier_pole_cap_alpha >= 0.0 &&
        mesh.polar_tier_pole_cap_alpha <= 1.0) ||
      !(std::isfinite(mesh.polar_tier_chi_lo) &&
        mesh.polar_tier_chi_lo > 0.0) ||
      !(std::isfinite(mesh.polar_tier_fan_first_ring_radius_cm) &&
        mesh.polar_tier_fan_first_ring_radius_cm >= 0.0) ||
      !(std::isfinite(mesh.multiblock_cart_core_r_match) &&
        std::isfinite(mesh.spherical_polar_s_max) &&
        mesh.spherical_polar_s_max >
            mesh.multiblock_cart_core_r_match)) {
    return layout;
  }

  int columns = mesh.nz;
  layout.tier_columns.push_back(columns);
  while (columns > mesh.polar_tier_min_tier_columns && columns % 2 == 0) {
    columns /= 2;
    layout.tier_columns.push_back(columns);
  }
  if (layout.tier_columns.back() != mesh.polar_tier_min_tier_columns) {
    return PolarTierLayout{};
  }
  for (std::size_t tier = 1; tier < layout.tier_columns.size(); ++tier) {
    const int coarse_columns = layout.tier_columns[tier];
    if ((mesh.polar_tier_belt_rows == 2 && coarse_columns % 4 != 0) ||
        (mesh.polar_tier_belt_rows == 3 && coarse_columns % 6 != 0)) {
      return PolarTierLayout{};
    }
  }

  const double r_match = mesh.multiblock_cart_core_r_match;
  // The tier ladder continues the shell's INNERMOST radial spacing inward.
  // Graded/explicit shell schedules (e.g. an outer ambient staircase) must
  // not perturb it, so h_r comes from the first shell interval, falling back
  // to the uniform average only when no schedule is given. The uniform
  // branch is bit-identical to the historic expression.
  if (!mesh.explicit_nodes.empty()) {
    layout.h_r =
        mesh.explicit_nodes.size() >= 2U
            ? mesh.explicit_nodes[1] - mesh.explicit_nodes[0]
            : 0.0;
  } else if (!mesh.grid_segments.empty()) {
    const auto& first_segment = mesh.grid_segments.front();
    layout.h_r =
        first_segment.nr > 0
            ? (first_segment.r_end - first_segment.r_start) /
                  static_cast<double>(first_segment.nr)
            : 0.0;
  } else {
    layout.h_r =
        (mesh.spherical_polar_s_max - r_match) /
        static_cast<double>(mesh.nr);
  }
  if (!(std::isfinite(layout.h_r) && layout.h_r > 0.0)) {
    return PolarTierLayout{};
  }

  layout.fan_radius =
      mesh.polar_tier_fan_first_ring_radius_cm > 0.0
          ? mesh.polar_tier_fan_first_ring_radius_cm
          : layout.h_r /
                std::sin(pi /
                         static_cast<double>(
                             mesh.polar_tier_fan_sectors));
  if (!(std::isfinite(layout.fan_radius) && layout.fan_radius > 0.0)) {
    return PolarTierLayout{};
  }

  const int n_tiers = static_cast<int>(layout.tier_columns.size());
  layout.transition_face_indices.reserve(
      static_cast<std::size_t>(std::max(0, n_tiers - 1)));
  layout.transition_radii.reserve(
      static_cast<std::size_t>(std::max(0, n_tiers - 1)));
  layout.transition_chi_fine.reserve(
      static_cast<std::size_t>(std::max(0, n_tiers - 1)));
  layout.transition_chi_coarse.reserve(
      static_cast<std::size_t>(std::max(0, n_tiers - 1)));
  layout.transition_outer_radii.reserve(
      static_cast<std::size_t>(std::max(0, n_tiers - 1)));
  layout.transition_inner_radii.reserve(
      static_cast<std::size_t>(std::max(0, n_tiers - 1)));
  layout.transition_intermediate_columns.reserve(
      static_cast<std::size_t>(std::max(0, n_tiers - 1)));
  layout.transition_intermediate_radii.reserve(
      static_cast<std::size_t>(std::max(0, n_tiers - 1)));
  layout.tier_radial_rows.reserve(static_cast<std::size_t>(n_tiers));
  layout.tier_outer_radii.reserve(static_cast<std::size_t>(n_tiers));
  layout.tier_inner_radii.reserve(static_cast<std::size_t>(n_tiers));
  layout.tier_radial_spacings.reserve(static_cast<std::size_t>(n_tiers));

  double tier_outer_radius = r_match;
  for (int tier = 0; tier + 1 < n_tiers; ++tier) {
    const int n_fine =
        layout.tier_columns[static_cast<std::size_t>(tier)];
    const double target =
        mesh.polar_tier_chi_lo * layout.h_r *
        static_cast<double>(n_fine) / pi;
    const double face_real =
        std::round((r_match - target) / layout.h_r);
    if (!(std::isfinite(target) && std::isfinite(face_real) &&
          face_real >=
              static_cast<double>(std::numeric_limits<int>::min()) &&
          face_real <=
              static_cast<double>(std::numeric_limits<int>::max()))) {
      return PolarTierLayout{};
    }
    const int face_index = static_cast<int>(face_real);
    const double radius =
        r_match - static_cast<double>(face_index) * layout.h_r;
    const double chi_fine =
        radius * pi / (layout.h_r * static_cast<double>(n_fine));
    layout.transition_face_indices.push_back(face_index);
    layout.transition_radii.push_back(radius);
    layout.transition_chi_fine.push_back(chi_fine);
    layout.transition_chi_coarse.push_back(2.0 * chi_fine);
    const double span = tier_outer_radius - radius;
    const double rows_real = std::round(span / layout.h_r);
    if (!(std::isfinite(span) && span > 0.0 &&
          std::isfinite(rows_real) &&
          rows_real <=
              static_cast<double>(std::numeric_limits<int>::max()))) {
      return PolarTierLayout{};
    }
    const int rows =
        std::max(1, static_cast<int>(rows_real));
    layout.tier_radial_rows.push_back(rows);
    layout.tier_outer_radii.push_back(tier_outer_radius);
    layout.tier_inner_radii.push_back(radius);
    layout.tier_radial_spacings.push_back(
        span / static_cast<double>(rows));
    tier_outer_radius =
        r_match -
        (static_cast<double>(face_index) + 1.0) * layout.h_r;
  }
  const double innermost_span =
      tier_outer_radius - layout.fan_radius;
  const double innermost_rows_real =
      std::round(innermost_span / layout.h_r);
  if (!(std::isfinite(innermost_span) && innermost_span > 0.0 &&
        std::isfinite(innermost_rows_real) &&
        innermost_rows_real <=
            static_cast<double>(std::numeric_limits<int>::max()))) {
    return layout;
  }
  const int innermost_rows =
      std::max(1, static_cast<int>(innermost_rows_real));
  layout.tier_radial_rows.push_back(innermost_rows);
  layout.tier_outer_radii.push_back(tier_outer_radius);
  layout.tier_inner_radii.push_back(layout.fan_radius);
  layout.tier_radial_spacings.push_back(
      innermost_span / static_cast<double>(innermost_rows));

  if (mesh.polar_tier_belt_rows > 1) {
    if (mesh.polar_tier_belt_thickness_frac > 0.0) {
      return PolarTierLayout{};
    }
    for (int tier = 0; tier + 1 < n_tiers; ++tier) {
      const std::size_t outer_tier = static_cast<std::size_t>(tier);
      const std::size_t inner_tier = static_cast<std::size_t>(tier + 1);
      const double legacy_outer_radius =
          layout.tier_inner_radii[outer_tier];
      const double legacy_inner_radius =
          layout.tier_outer_radii[inner_tier];
      double outer_radius = legacy_outer_radius;
      double inner_radius = legacy_inner_radius;
      if (layout.tier_radial_rows[outer_tier] >= 2) {
        layout.tier_radial_rows[outer_tier] -= 1;
        layout.tier_inner_radii[outer_tier] +=
            layout.tier_radial_spacings[outer_tier];
        outer_radius = layout.tier_inner_radii[outer_tier];
      }
      if (layout.tier_radial_rows[inner_tier] >= 2) {
        layout.tier_radial_rows[inner_tier] -= 1;
        layout.tier_outer_radii[inner_tier] -=
            layout.tier_radial_spacings[inner_tier];
        inner_radius = layout.tier_outer_radii[inner_tier];
      }
      if (!(std::isfinite(outer_radius) && std::isfinite(inner_radius) &&
            outer_radius > inner_radius)) {
        return PolarTierLayout{};
      }
      layout.transition_outer_radii.push_back(outer_radius);
      layout.transition_inner_radii.push_back(inner_radius);

      const int coarse_columns =
          layout.tier_columns[static_cast<std::size_t>(tier + 1)];
      std::vector<int> intermediate_columns;
      std::vector<double> intermediate_radii;
      if (mesh.polar_tier_belt_rows == 2) {
        intermediate_columns.push_back(static_cast<int>(
            3LL * static_cast<long long>(coarse_columns) / 2LL));
        intermediate_radii.push_back(
            0.5 * (outer_radius + inner_radius));
      } else {
        const double span = outer_radius - inner_radius;
        intermediate_columns.push_back(static_cast<int>(
            5LL * static_cast<long long>(coarse_columns) / 3LL));
        intermediate_columns.push_back(static_cast<int>(
            4LL * static_cast<long long>(coarse_columns) / 3LL));
        intermediate_radii.push_back(outer_radius - span / 3.0);
        intermediate_radii.push_back(
            outer_radius - 2.0 * span / 3.0);
      }
      layout.transition_intermediate_columns.push_back(
          std::move(intermediate_columns));
      layout.transition_intermediate_radii.push_back(
          std::move(intermediate_radii));
    }
  }

  if (mesh.polar_tier_dendrite_enabled) {
    constexpr int master_columns = 192;
    constexpr int cap_half_width = 16;
    const std::vector<int> qualification_native_tier_columns{
        192, 96, 48, 24, 12};
    if (layout.tier_columns != qualification_native_tier_columns) {
      throw namelist::ConfigError(
          "Mesh.polar_tier_dendrite_enabled=true requires native tier "
          "columns [192, 96, 48, 24, 12]");
    }
    layout.dendrite_native_tier_columns = {192, 192, 96, 48, 24, 12};

    const auto make_master_labels =
        [&](const int native_columns,
            const int cap_intervals,
            const std::vector<int>& explicit_north_cap_labels)
        -> std::vector<int> {
      if (native_columns <= 0 || master_columns % native_columns != 0) {
        throw namelist::ConfigError(
            "Polar-tier dendrite ladder has a fractional native master "
            "stride");
      }
      const int native_stride = master_columns / native_columns;
      if (cap_intervals <= 0 ||
          cap_half_width % native_stride != 0) {
        throw namelist::ConfigError(
            "Polar-tier dendrite ladder has a fractional cap master label");
      }
      std::vector<int> north_cap_labels = explicit_north_cap_labels;
      if (north_cap_labels.empty()) {
        if (cap_half_width % cap_intervals != 0) {
          throw namelist::ConfigError(
              "Polar-tier dendrite ladder has a fractional cap master "
              "label");
        }
        const int cap_stride = cap_half_width / cap_intervals;
        for (int label = 0; label <= cap_half_width;
             label += cap_stride) {
          north_cap_labels.push_back(label);
        }
      }
      if (north_cap_labels.size() !=
              static_cast<std::size_t>(cap_intervals) + 1U ||
          north_cap_labels.front() != 0 ||
          north_cap_labels.back() != cap_half_width ||
          !std::is_sorted(north_cap_labels.begin(),
                          north_cap_labels.end()) ||
          std::adjacent_find(north_cap_labels.begin(),
                             north_cap_labels.end()) !=
              north_cap_labels.end()) {
        throw namelist::ConfigError(
            "Polar-tier dendrite explicit north cap ladder is not a "
            "strict 0..16 master partition");
      }
      std::vector<int> labels;
      labels.reserve(static_cast<std::size_t>(native_columns) + 1U);
      for (const int label : north_cap_labels) {
        labels.push_back(label);
      }
      for (int label = cap_half_width + native_stride;
           label <= master_columns - cap_half_width;
           label += native_stride) {
        labels.push_back(label);
      }
      for (std::size_t label = north_cap_labels.size() - 1U;
           label > 0U; --label) {
        labels.push_back(
            master_columns - north_cap_labels[label - 1U]);
      }
      if (labels.size() < 2U || labels.front() != 0 ||
          labels.back() != master_columns ||
          !std::is_sorted(labels.begin(), labels.end()) ||
          std::adjacent_find(labels.begin(), labels.end()) != labels.end()) {
        throw namelist::ConfigError(
            "Polar-tier dendrite ladder is not a strict 0..192 master "
            "partition");
      }
      for (std::size_t label = 0; label < labels.size(); ++label) {
        if (labels[label] +
                labels[labels.size() - 1U - label] !=
            master_columns) {
          throw namelist::ConfigError(
              "Polar-tier dendrite ladder is not exactly north/south "
              "mirrored");
        }
      }
      return labels;
    };

    if (mesh.shell_polar_cap_dendrite) {
      constexpr int required_cap_half_width = 16;
      if (cap_half_width != required_cap_half_width ||
          cap_half_width % 2 != 0) {
        throw namelist::ConfigError(
            "Shell-cap dendrite requires J_W == 16 and J_W mod 2 == 0");
      }
      layout.shell_master_theta_node_labels.reserve(2U);
      layout.shell_master_theta_node_labels.push_back(
          make_master_labels(master_columns, 8, {}));
      layout.shell_master_theta_node_labels.push_back(
          make_master_labels(master_columns, 16, {}));
      if (layout.shell_master_theta_node_labels[0].size() != 177U ||
          layout.shell_master_theta_node_labels[1].size() != 193U) {
        throw namelist::ConfigError(
            "Shell-cap dendrite must derive 176-column C2 and 192-column "
            "FINE ring labels");
      }
    }

    // S_theta consumes one T1 row between T1a and T1b, while S_theta2
    // consumes T2's first row. The regular-tier schedule is
    // [176, 176, 88, 44, 22, 12] with shell-cap dendrite enabled and
    // [192, 178, 88, 44, 22, 12] otherwise.
    const std::vector<int> actual_cap_intervals =
        mesh.shell_polar_cap_dendrite
            ? std::vector<int>{8, 8, 4, 2, 1, 1}
            : std::vector<int>{16, 9, 4, 2, 1, 1};
    layout.dendrite_actual_tier_columns.reserve(
        layout.dendrite_native_tier_columns.size());
    for (std::size_t tier = 0;
         tier < layout.dendrite_native_tier_columns.size(); ++tier) {
      const int native_columns =
          layout.dendrite_native_tier_columns[tier];
      if (master_columns % native_columns != 0) {
        throw namelist::ConfigError(
            "Polar-tier dendrite tier has a fractional native master "
            "stride");
      }
      const int native_stride = master_columns / native_columns;
      if (cap_half_width % native_stride != 0) {
        throw namelist::ConfigError(
            "Polar-tier dendrite cap boundary is not a native tier label");
      }
      const int native_cap_intervals =
          cap_half_width / native_stride;
      const int actual_columns =
          native_columns -
          2 * (native_cap_intervals - actual_cap_intervals[tier]);
      layout.dendrite_actual_tier_columns.push_back(actual_columns);
    }

    const std::vector<int> ring_native_columns{
        192, 192, 96, 96, 48, 24, 12};
    const std::vector<int> ring_cap_intervals =
        mesh.shell_polar_cap_dendrite
            ? std::vector<int>{8, 8, 6, 4, 2, 1, 1}
            : std::vector<int>{16, 9, 6, 4, 2, 1, 1};
    const std::vector<std::vector<int>> explicit_north_cap_labels =
        mesh.shell_polar_cap_dendrite
            ? std::vector<std::vector<int>>{
                  {},
                  {},
                  {0, 2, 4, 8, 12, 14, 16},
                  {0, 4, 8, 12, 16},
                  {0, 8, 16},
                  {},
                  {},
              }
            : std::vector<std::vector<int>>{
                  {},
                  {0, 1, 3, 5, 7, 9, 11, 13, 15, 16},
                  {0, 1, 5, 9, 13, 15, 16},
                  {0, 1, 9, 15, 16},
                  {0, 9, 16},
                  {},
                  {},
              };
    layout.dendrite_master_theta_node_labels.reserve(
        ring_native_columns.size());
    layout.dendrite_ring_interval_counts.reserve(
        ring_native_columns.size());
    for (std::size_t ring = 0; ring < ring_native_columns.size(); ++ring) {
      auto labels =
          mesh.shell_polar_cap_dendrite && ring < 2U
              ? layout.shell_master_theta_node_labels.front()
              : make_master_labels(ring_native_columns[ring],
                                   ring_cap_intervals[ring],
                                   explicit_north_cap_labels[ring]);
      layout.dendrite_ring_interval_counts.push_back(
          static_cast<int>(labels.size()) - 1);
      layout.dendrite_master_theta_node_labels.push_back(
          std::move(labels));
    }
    if (mesh.shell_polar_cap_dendrite &&
        (layout.dendrite_master_theta_node_labels[0] !=
             layout.shell_master_theta_node_labels[0] ||
         layout.dendrite_master_theta_node_labels[1] !=
             layout.shell_master_theta_node_labels[0])) {
      throw namelist::ConfigError(
          "Shell-cap C2 and both T1 ladders must match exactly");
    }

    const std::vector<std::size_t> tier_ring_indices{
        0, 1, 3, 4, 5, 6};
    for (std::size_t tier = 0; tier < tier_ring_indices.size(); ++tier) {
      const std::size_t ring = tier_ring_indices[tier];
      if (ring_native_columns[ring] !=
              layout.dendrite_native_tier_columns[tier] ||
          layout.dendrite_ring_interval_counts[ring] !=
              layout.dendrite_actual_tier_columns[tier]) {
        throw namelist::ConfigError(
            "Polar-tier dendrite native/actual tier ladder mismatch");
      }
    }

    const auto make_join_descriptors =
        [](const std::vector<int>& outer_labels,
           const std::vector<int>& inner_labels)
        -> std::vector<PolarTierJoinDescriptor> {
      if (outer_labels.size() < 2U || inner_labels.size() < 2U ||
          outer_labels.front() != inner_labels.front() ||
          outer_labels.back() != inner_labels.back()) {
        throw namelist::ConfigError(
            "Polar-tier dendrite join endpoints do not match");
      }
      std::vector<PolarTierJoinDescriptor> joins;
      joins.reserve(inner_labels.size() - 1U);
      int expected_outer_begin = 0;
      for (std::size_t inner = 0; inner + 1U < inner_labels.size();
           ++inner) {
        const auto outer_begin_it =
            std::lower_bound(outer_labels.begin(), outer_labels.end(),
                             inner_labels[inner]);
        const auto outer_end_it =
            std::lower_bound(outer_labels.begin(), outer_labels.end(),
                             inner_labels[inner + 1U]);
        if (outer_begin_it == outer_labels.end() ||
            outer_end_it == outer_labels.end() ||
            *outer_begin_it != inner_labels[inner] ||
            *outer_end_it != inner_labels[inner + 1U]) {
          throw namelist::ConfigError(
              "Polar-tier dendrite join has a gap or fractional endpoint");
        }
        const int outer_begin = static_cast<int>(
            std::distance(outer_labels.begin(), outer_begin_it));
        const int outer_end = static_cast<int>(
            std::distance(outer_labels.begin(), outer_end_it));
        const int ratio = outer_end - outer_begin;
        if (outer_begin != expected_outer_begin) {
          throw namelist::ConfigError(
              "Polar-tier dendrite join has an outer interval gap or "
              "overlap");
        }
        if (ratio != 1 && ratio != 2) {
          throw namelist::ConfigError(
              "Polar-tier dendrite join interval ratio must be exactly "
              "1 or 2");
        }
        joins.push_back(PolarTierJoinDescriptor{
            ratio == 1 ? PolarTierJoinKind::ONE_TO_ONE
                       : PolarTierJoinKind::TWO_TO_ONE,
            outer_begin,
            outer_end,
            static_cast<int>(inner),
            static_cast<int>(inner + 1U)});
        expected_outer_begin = outer_end;
      }
      if (expected_outer_begin !=
          static_cast<int>(outer_labels.size()) - 1) {
        throw namelist::ConfigError(
            "Polar-tier dendrite join does not consume every outer "
            "interval");
      }
      return joins;
    };

    if (mesh.shell_polar_cap_dendrite) {
      constexpr int shell_rows = 288;
      const bool full_shell_c2 = mesh.shell_cap_rows_2x == shell_rows;
      const int coarse_cohort_rows = mesh.shell_cap_rows_2x + 1;
      if (!full_shell_c2 && coarse_cohort_rows >= shell_rows) {
        throw namelist::ConfigError(
            "Shell-cap dendrite C2 and T21 rows must leave a FINE cohort");
      }

      if (!full_shell_c2) {
        layout.shell_transition_joins.reserve(1U);
        layout.shell_transition_joins.push_back(
            make_join_descriptors(
                layout.shell_master_theta_node_labels[1],
                layout.shell_master_theta_node_labels[0]));

        const auto join_kind_count =
            [](const std::vector<PolarTierJoinDescriptor>& joins,
               const PolarTierJoinKind kind) -> int {
          return static_cast<int>(std::count_if(
              joins.begin(), joins.end(),
              [&](const PolarTierJoinDescriptor& join) {
                return join.kind == kind;
              }));
        };
        if (layout.shell_transition_joins.size() != 1U ||
            join_kind_count(layout.shell_transition_joins[0],
                            PolarTierJoinKind::TWO_TO_ONE) != 16 ||
            join_kind_count(layout.shell_transition_joins[0],
                            PolarTierJoinKind::ONE_TO_ONE) != 160) {
          throw namelist::ConfigError(
              "Shell-cap dendrite transition joins do not match the "
              "bilateral cap/interior contract");
        }
      }

      const int c2_cells_per_row = static_cast<int>(
          layout.shell_master_theta_node_labels[0].size()) - 1;
      const int fine_cells_per_row = static_cast<int>(
          layout.shell_master_theta_node_labels[1].size()) - 1;
      const int c2_nodes_per_ring = c2_cells_per_row + 1;
      const int fine_nodes_per_ring = fine_cells_per_row + 1;

      const auto append_shell_band =
          [&](const PolarTierShellBandKind kind,
              const int row_begin,
              const int row_end,
              const int cap_leaf_width,
              const int cells_per_row_full,
              const long long owned_nodes) {
        if (row_begin < 0 || row_end <= row_begin ||
            cells_per_row_full <= 0 || cells_per_row_full % 2 != 0 ||
            owned_nodes <= 0) {
          throw namelist::ConfigError(
              "Shell-cap dendrite band descriptor is invalid");
        }
        const long long row_count =
            static_cast<long long>(row_end - row_begin);
        layout.shell_chain.push_back(PolarTierShellBandDescriptor{
            kind,
            row_begin,
            row_end,
            cap_leaf_width,
            cells_per_row_full / 2,
            cells_per_row_full,
            row_count * cells_per_row_full,
            owned_nodes});
      };

      int row = 0;
      append_shell_band(
          PolarTierShellBandKind::SHELL_C2,
          row,
          row + mesh.shell_cap_rows_2x,
          2,
          c2_cells_per_row,
          (static_cast<long long>(mesh.shell_cap_rows_2x) + 1LL) *
              c2_nodes_per_ring);
      row += mesh.shell_cap_rows_2x;
      if (!full_shell_c2) {
        const int t21_cells_per_row = static_cast<int>(
            layout.shell_transition_joins[0].size());
        append_shell_band(
            PolarTierShellBandKind::SHELL_T21,
            row,
            row + 1,
            2,
            t21_cells_per_row,
            fine_nodes_per_ring);
        row += 1;
        append_shell_band(
            PolarTierShellBandKind::SHELL_FINE,
            row,
            shell_rows,
            1,
            fine_cells_per_row,
            static_cast<long long>(shell_rows - row) *
                fine_nodes_per_ring);
      }

      int expected_row_begin = 0;
      long long descriptor_cells = 0;
      long long descriptor_nodes = 0;
      for (const PolarTierShellBandDescriptor& band : layout.shell_chain) {
        if (band.row_begin != expected_row_begin) {
          throw namelist::ConfigError(
              "Shell-cap dendrite chain has a radial gap or overlap");
        }
        expected_row_begin = band.row_end;
        descriptor_cells += band.n_cells;
        descriptor_nodes += band.n_nodes;
      }
      const std::size_t expected_chain_size = full_shell_c2 ? 1U : 3U;
      if (layout.shell_chain.size() != expected_chain_size ||
          expected_row_begin != shell_rows) {
        throw namelist::ConfigError(
            "Shell-cap dendrite chain must cover exactly 288 shell rows");
      }
      long long expected_cells =
          static_cast<long long>(mesh.shell_cap_rows_2x) *
          c2_cells_per_row;
      long long expected_nodes =
          (static_cast<long long>(mesh.shell_cap_rows_2x) + 1LL) *
          c2_nodes_per_ring;
      if (!full_shell_c2) {
        const int fine_rows = shell_rows - coarse_cohort_rows;
        const int t21_cells_per_row = static_cast<int>(
            layout.shell_transition_joins[0].size());
        expected_cells +=
            t21_cells_per_row +
            static_cast<long long>(fine_rows) * fine_cells_per_row;
        expected_nodes +=
            fine_nodes_per_ring +
            static_cast<long long>(fine_rows) * fine_nodes_per_ring;
      }
      layout.shell_n_cells = descriptor_cells;
      layout.shell_n_nodes = descriptor_nodes;
      if (layout.shell_n_cells != expected_cells ||
          layout.shell_n_nodes != expected_nodes) {
        throw namelist::ConfigError(
            "Shell-cap dendrite totals do not match descriptor sums");
      }
    }

    layout.dendrite_transition_joins.reserve(
        layout.dendrite_master_theta_node_labels.size() - 1U);
    for (std::size_t transition = 0;
         transition + 1U <
         layout.dendrite_master_theta_node_labels.size();
         ++transition) {
      layout.dendrite_transition_joins.push_back(
          make_join_descriptors(
              layout.dendrite_master_theta_node_labels[transition],
              layout.dendrite_master_theta_node_labels[transition + 1U]));
    }
    if (layout.dendrite_transition_joins.size() != 6U) {
      throw namelist::ConfigError(
          "Polar-tier dendrite must derive exactly six transitions");
    }

    constexpr int s_theta_rows = 1;
    const int t1b_rows =
        mesh.polar_tier_dendrite_s_theta_rows_below;
    const int t1a_rows = 61 - s_theta_rows - t1b_rows;
    if (layout.tier_radial_rows.size() != 5U ||
        layout.tier_radial_rows[0] !=
            t1a_rows + s_theta_rows + t1b_rows ||
        layout.tier_radial_rows[1] < 2) {
      throw namelist::ConfigError(
          "Polar-tier dendrite requires exactly 61 radial rows in T1 "
          "and one consumable radial row in T2");
    }
    layout.dendrite_tier_radial_rows = layout.tier_radial_rows;
    layout.dendrite_tier_radial_rows[0] = t1a_rows;
    layout.dendrite_tier_radial_rows.insert(
        layout.dendrite_tier_radial_rows.begin() + 1, t1b_rows);
    layout.dendrite_tier_radial_rows[2] -= 1;

    if (layout.dendrite_actual_tier_columns.back() !=
        mesh.polar_tier_fan_sectors) {
      throw namelist::ConfigError(
          "Polar-tier dendrite innermost ring/fan interval mismatch");
    }

    constexpr std::array<int, 6> tier_radial_indices =
        {{0, 0, 1, 2, 3, 4}};
    std::array<double, 6> dendrite_tier_outer{};
    std::array<double, 6> dendrite_tier_inner{};
    std::array<double, 6> dendrite_transition_outer{};
    std::array<double, 6> dendrite_transition_inner{};
    dendrite_tier_outer[0] = r_match;
    dendrite_tier_inner[0] =
        layout.tier_inner_radii[0] +
        static_cast<double>(layout.dendrite_tier_radial_rows[1] + 1) *
            layout.tier_radial_spacings[0];
    dendrite_transition_outer[0] = dendrite_tier_inner[0];
    dendrite_transition_inner[0] =
        layout.tier_inner_radii[0] +
        static_cast<double>(layout.dendrite_tier_radial_rows[1]) *
            layout.tier_radial_spacings[0];
    dendrite_tier_outer[1] = dendrite_transition_inner[0];
    dendrite_tier_inner[1] = layout.tier_inner_radii[0];
    dendrite_transition_outer[1] = dendrite_tier_inner[1];
    dendrite_transition_inner[1] = layout.tier_outer_radii[1];
    dendrite_transition_outer[2] = dendrite_transition_inner[1];
    dendrite_transition_inner[2] =
        layout.tier_outer_radii[1] - layout.tier_radial_spacings[1];
    dendrite_tier_outer[2] = dendrite_transition_inner[2];
    dendrite_tier_inner[2] = layout.tier_inner_radii[1];
    for (int tier = 3; tier < 6; ++tier) {
      const int radial_index =
          tier_radial_indices[static_cast<std::size_t>(tier)];
      const int transition = tier;
      dendrite_transition_outer[static_cast<std::size_t>(transition)] =
          dendrite_tier_inner[static_cast<std::size_t>(tier - 1)];
      dendrite_transition_inner[static_cast<std::size_t>(transition)] =
          layout.tier_outer_radii[static_cast<std::size_t>(radial_index)];
      dendrite_tier_outer[static_cast<std::size_t>(tier)] =
          dendrite_transition_inner[static_cast<std::size_t>(transition)];
      dendrite_tier_inner[static_cast<std::size_t>(tier)] =
          layout.tier_inner_radii[static_cast<std::size_t>(radial_index)];
    }

    int outward_ring = 0;
    layout.schedule.push_back(PolarTierScheduleEntry{
        PolarTierEntityKind::FAN,
        -1,
        false,
        0,
        0,
        layout.dendrite_actual_tier_columns.back(),
        0.0,
        layout.fan_radius});
    const auto append_dendrite_tier = [&](const int tier) {
      const int rows =
          layout.dendrite_tier_radial_rows[static_cast<std::size_t>(tier)];
      layout.schedule.push_back(PolarTierScheduleEntry{
          PolarTierEntityKind::TIER,
          tier,
          false,
          outward_ring,
          outward_ring + rows,
          layout.dendrite_actual_tier_columns[static_cast<std::size_t>(tier)],
          dendrite_tier_inner[static_cast<std::size_t>(tier)],
          dendrite_tier_outer[static_cast<std::size_t>(tier)]});
      outward_ring += rows;
    };
    const auto append_dendrite_transition = [&](const int transition) {
      layout.schedule.push_back(PolarTierScheduleEntry{
          PolarTierEntityKind::TRANSITION,
          transition,
          false,
          outward_ring,
          outward_ring + 1,
          static_cast<int>(
              layout.dendrite_transition_joins[
                  static_cast<std::size_t>(transition)]
                  .size()),
          dendrite_transition_inner[static_cast<std::size_t>(transition)],
          dendrite_transition_outer[static_cast<std::size_t>(transition)]});
      ++outward_ring;
    };
    append_dendrite_tier(5);
    append_dendrite_transition(5);
    append_dendrite_tier(4);
    append_dendrite_transition(4);
    append_dendrite_tier(3);
    append_dendrite_transition(3);
    append_dendrite_tier(2);
    append_dendrite_transition(2);
    append_dendrite_transition(1);
    append_dendrite_tier(1);
    append_dendrite_transition(0);
    append_dendrite_tier(0);
    if (mesh.shell_polar_cap_dendrite) {
      const int shell_inner_ring = outward_ring;
      const double nan = std::numeric_limits<double>::quiet_NaN();
      for (std::size_t band_index = 0;
           band_index < layout.shell_chain.size(); ++band_index) {
        const PolarTierShellBandDescriptor& band =
            layout.shell_chain[band_index];
        const PolarTierEntityKind kind =
            band.kind == PolarTierShellBandKind::SHELL_T21
                ? PolarTierEntityKind::TRANSITION
                : PolarTierEntityKind::SHELL;
        layout.schedule.push_back(PolarTierScheduleEntry{
            kind,
            static_cast<int>(band_index),
            true,
            shell_inner_ring + band.row_begin,
            shell_inner_ring + band.row_end,
            band.cells_per_row_full,
            nan,
            nan});
      }
      outward_ring += mesh.nr;
    } else {
      layout.schedule.push_back(PolarTierScheduleEntry{
          PolarTierEntityKind::SHELL,
          -1,
          false,
          outward_ring,
          outward_ring + mesh.nr,
          mesh.nz,
          r_match,
          mesh.spherical_polar_s_max});
      outward_ring += mesh.nr;
    }

    const int expected_block_count =
        (mesh.shell_polar_cap_dendrite
             ? static_cast<int>(layout.shell_chain.size())
             : 1) +
        static_cast<int>(layout.dendrite_actual_tier_columns.size()) +
        static_cast<int>(layout.dendrite_transition_joins.size()) + 1;
    if (mesh.shell_polar_cap_dendrite &&
        expected_block_count !=
            (mesh.shell_cap_rows_2x == mesh.nr ? 14 : 16)) {
      throw namelist::ConfigError(
          "Shell-cap polar-tier dendrite structural block count must be "
          "14 or 16");
    }
    TENRYU_ASSERT(layout.schedule.front().kind ==
                          PolarTierEntityKind::FAN &&
                      layout.schedule.back().kind ==
                          PolarTierEntityKind::SHELL,
                  "polar-tier dendrite schedule endpoints mismatch");
    int previous_ring_end = 0;
    for (std::size_t entity = 1; entity < layout.schedule.size(); ++entity) {
      const PolarTierScheduleEntry& entry = layout.schedule[entity];
      TENRYU_ASSERT(entry.ring_begin == previous_ring_end &&
                        entry.ring_end > entry.ring_begin &&
                        (entry.in_shell_chain
                             ? (std::isnan(entry.r_inner) &&
                                std::isnan(entry.r_outer))
                             : entry.r_outer > entry.r_inner),
                    "polar-tier dendrite schedule is not contiguous");
      previous_ring_end = entry.ring_end;
    }
    long long expected_n_cells =
        mesh.shell_polar_cap_dendrite
            ? layout.shell_n_cells
            : static_cast<long long>(mesh.nr) * master_columns;
    long long expected_n_nodes =
        mesh.shell_polar_cap_dendrite
            ? layout.shell_n_nodes
            : (static_cast<long long>(mesh.nr) + 1LL) *
                  (static_cast<long long>(master_columns) + 1LL);
    for (std::size_t tier = 0;
         tier < layout.dendrite_actual_tier_columns.size(); ++tier) {
      const long long columns = layout.dendrite_actual_tier_columns[tier];
      const long long rows = layout.dendrite_tier_radial_rows[tier];
      expected_n_cells += rows * columns;
      expected_n_nodes += rows * (columns + 1LL);
    }
    expected_n_cells += layout.dendrite_actual_tier_columns.back();
    expected_n_nodes += 1LL;
    for (const PolarTierScheduleEntry& entry : layout.schedule) {
      if (entry.kind != PolarTierEntityKind::TRANSITION ||
          entry.in_shell_chain) {
        continue;
      }
      const PolarTierEntityCounts transition_counts =
          polar_tier_schedule_entry_counts(mesh, layout, entry);
      expected_n_cells += transition_counts.n_cells;
      expected_n_nodes += transition_counts.n_nodes;
    }
    const PolarTierEntityCounts schedule_totals =
        polar_tier_schedule_totals(mesh, layout);
    layout.block_count = static_cast<int>(schedule_totals.n_blocks);
    layout.n_cells = schedule_totals.n_cells;
    layout.n_nodes = schedule_totals.n_nodes;
    TENRYU_ASSERT(layout.block_count == expected_block_count &&
                      layout.block_count ==
                          static_cast<int>(layout.schedule.size()) &&
                      layout.n_cells == expected_n_cells &&
                      layout.n_nodes == expected_n_nodes &&
                      layout.n_cells > 0 && layout.n_nodes > 0,
                  "polar-tier dendrite schedule totals mismatch");
    return layout;
  }

  const int expected_block_count = 2 * n_tiers + 1;
  long long expected_n_cells =
      static_cast<long long>(mesh.nr) * layout.tier_columns.front();
  long long expected_n_nodes =
      (static_cast<long long>(mesh.nr) + 1LL) *
      (static_cast<long long>(layout.tier_columns.front()) + 1LL);
  for (int tier = 0; tier < n_tiers; ++tier) {
    const long long n_columns =
        layout.tier_columns[static_cast<std::size_t>(tier)];
    const long long n_rows =
        layout.tier_radial_rows[static_cast<std::size_t>(tier)];
    expected_n_cells += n_rows * n_columns;
    expected_n_nodes += n_rows * (n_columns + 1LL);
    if (tier + 1 < n_tiers) {
      const long long n_coarse =
          layout.tier_columns[static_cast<std::size_t>(tier + 1)];
      if (mesh.polar_tier_belt_rows == 1) {
        expected_n_cells += 5LL * n_coarse;
        expected_n_nodes += 2LL * n_coarse + 1LL;
      } else if (mesh.polar_tier_belt_rows == 2) {
        expected_n_cells += 5LL * n_coarse;
        expected_n_nodes += 5LL * n_coarse / 2LL + 2LL;
      } else {
        expected_n_cells += 6LL * n_coarse;
        expected_n_nodes += 4LL * n_coarse + 3LL;
      }
    }
  }
  expected_n_cells += layout.tier_columns.back();
  expected_n_nodes += 1LL;

  // Zero deliberately uses the legacy expressions verbatim. This
  // experiment knob is discontinuous as the fraction approaches zero.
  if (mesh.polar_tier_belt_thickness_frac != 0.0) {
    for (int tier = 0; tier + 1 < n_tiers; ++tier) {
      const std::size_t outer_tier = static_cast<std::size_t>(tier);
      const std::size_t inner_tier = static_cast<std::size_t>(tier + 1);
      const double legacy_belt_outer_radius =
          layout.tier_inner_radii[outer_tier];
      const double legacy_belt_inner_radius =
          layout.tier_outer_radii[inner_tier];
      const double legacy_belt_thickness =
          legacy_belt_outer_radius - legacy_belt_inner_radius;
      const double dr_local = layout.tier_radial_spacings[inner_tier];
      const double belt_thickness =
          mesh.polar_tier_belt_thickness_frac * dr_local;
      const double adjacent_row_loss =
          0.5 * (belt_thickness - legacy_belt_thickness);
      if (adjacent_row_loss >= layout.tier_radial_spacings[outer_tier] ||
          adjacent_row_loss >= layout.tier_radial_spacings[inner_tier]) {
        throw namelist::ConfigError(
            "Numerics/Mesh polar_tier_belt_thickness_frac leaves a "
            "non-positive adjacent tier row");
      }
      const double legacy_belt_center =
          0.5 * (legacy_belt_outer_radius + legacy_belt_inner_radius);
      const double adjusted_belt_outer_radius =
          legacy_belt_center + 0.5 * belt_thickness;
      const double adjusted_belt_inner_radius =
          legacy_belt_center - 0.5 * belt_thickness;
      layout.transition_radii[outer_tier] = adjusted_belt_outer_radius;
      layout.tier_inner_radii[outer_tier] = adjusted_belt_outer_radius;
      layout.tier_outer_radii[inner_tier] = adjusted_belt_inner_radius;
    }
  }

  int outward_ring = 0;
  layout.schedule.push_back(PolarTierScheduleEntry{
      PolarTierEntityKind::FAN,
      -1,
      false,
      0,
      0,
      layout.tier_columns.back(),
      0.0,
      layout.fan_radius});
  for (int tier = n_tiers - 1; tier >= 0; --tier) {
    const std::size_t tier_index = static_cast<std::size_t>(tier);
    const int rows = layout.tier_radial_rows[tier_index];
    layout.schedule.push_back(PolarTierScheduleEntry{
        PolarTierEntityKind::TIER,
        tier,
        false,
        outward_ring,
        outward_ring + rows,
        layout.tier_columns[tier_index],
        layout.tier_inner_radii[tier_index],
        layout.tier_outer_radii[tier_index]});
    outward_ring += rows;
    if (tier > 0) {
      const int transition = tier - 1;
      const std::size_t transition_index =
          static_cast<std::size_t>(transition);
      const double transition_inner_radius =
          mesh.polar_tier_belt_rows > 1
              ? layout.transition_inner_radii[transition_index]
              : layout.tier_outer_radii[tier_index];
      const double transition_outer_radius =
          mesh.polar_tier_belt_rows > 1
              ? layout.transition_outer_radii[transition_index]
              : layout.tier_inner_radii[transition_index];
      layout.schedule.push_back(PolarTierScheduleEntry{
          PolarTierEntityKind::TRANSITION,
          transition,
          false,
          outward_ring,
          outward_ring + mesh.polar_tier_belt_rows,
          layout.tier_columns[tier_index],
          transition_inner_radius,
          transition_outer_radius});
      outward_ring += mesh.polar_tier_belt_rows;
    }
  }
  layout.schedule.push_back(PolarTierScheduleEntry{
      PolarTierEntityKind::SHELL,
      -1,
      false,
      outward_ring,
      outward_ring + mesh.nr,
      layout.tier_columns.front(),
      r_match,
      mesh.spherical_polar_s_max});

  TENRYU_ASSERT(layout.schedule.front().kind ==
                        PolarTierEntityKind::FAN &&
                    layout.schedule.back().kind ==
                        PolarTierEntityKind::SHELL,
                "polar-tier schedule endpoints mismatch");
  int previous_ring_end = 0;
  for (std::size_t entity = 1; entity < layout.schedule.size(); ++entity) {
    const PolarTierScheduleEntry& entry = layout.schedule[entity];
    TENRYU_ASSERT(entry.ring_begin == previous_ring_end &&
                      entry.ring_end > entry.ring_begin &&
                      entry.r_outer > entry.r_inner,
                  "polar-tier schedule is not contiguous");
    previous_ring_end = entry.ring_end;
  }
  const PolarTierEntityCounts schedule_totals =
      polar_tier_schedule_totals(mesh, layout);
  layout.block_count = static_cast<int>(schedule_totals.n_blocks);
  layout.n_cells = schedule_totals.n_cells;
  layout.n_nodes = schedule_totals.n_nodes;
  TENRYU_ASSERT(layout.block_count == expected_block_count &&
                    layout.block_count ==
                        static_cast<int>(layout.schedule.size()) &&
                    layout.n_cells == expected_n_cells &&
                    layout.n_nodes == expected_n_nodes,
                "polar-tier schedule totals mismatch");
  return layout;
}

inline PolarTierLayout make_polar_tier_layout_truncated(
    const Config::MeshConfig& mesh,
    const int cut_ring,
    PolarTierTruncation* out_info) {
  const PolarTierLayout full = make_polar_tier_layout(mesh);
  const std::size_t n_tiers = full.tier_columns.size();
  const std::size_t n_belts = n_tiers > 0U ? n_tiers - 1U : 0U;
  if (n_tiers == 0U || full.tier_radial_rows.size() != n_tiers ||
      full.tier_outer_radii.size() != n_tiers ||
      full.tier_inner_radii.size() != n_tiers ||
      full.tier_radial_spacings.size() != n_tiers ||
      full.transition_face_indices.size() != n_belts ||
      full.transition_radii.size() != n_belts ||
      full.transition_chi_fine.size() != n_belts ||
      full.transition_chi_coarse.size() != n_belts ||
      full.schedule.empty() ||
      full.schedule.front().kind != PolarTierEntityKind::FAN ||
      full.schedule.back().kind != PolarTierEntityKind::SHELL ||
      (mesh.polar_tier_belt_rows > 1 &&
       (full.transition_outer_radii.size() != n_belts ||
        full.transition_inner_radii.size() != n_belts ||
        full.transition_intermediate_columns.size() != n_belts ||
        full.transition_intermediate_radii.size() != n_belts))) {
    throw namelist::ConfigError(
        "hybrid truncation requires a valid full multiblock_polar_tier "
        "layout");
  }

  const auto shell_begin_it = std::find_if(
      full.schedule.begin(), full.schedule.end(),
      [](const PolarTierScheduleEntry& entry) {
        return entry.in_shell_chain ||
               entry.kind == PolarTierEntityKind::SHELL;
      });
  TENRYU_ASSERT(shell_begin_it != full.schedule.end(),
                "polar-tier schedule has no shell entry");
  const int shell_inner_ring = shell_begin_it->ring_begin;
  std::vector<int> admissible_rings;
  for (const PolarTierScheduleEntry& tier : full.schedule) {
    if (tier.kind != PolarTierEntityKind::TIER ||
        tier.columns % 4 != 0) {
      continue;
    }
    for (int ring = tier.ring_begin + 1; ring < tier.ring_end; ++ring) {
      if (ring <= 0 || ring >= shell_inner_ring) {
        continue;
      }
      bool away_from_belts = true;
      for (const PolarTierScheduleEntry& transition : full.schedule) {
        if (transition.kind != PolarTierEntityKind::TRANSITION ||
            transition.in_shell_chain) {
          continue;
        }
        const int distance =
            ring < transition.ring_begin
                ? transition.ring_begin - ring
                : (ring > transition.ring_end
                       ? ring - transition.ring_end
                       : 0);
        if (distance < 3) {
          away_from_belts = false;
          break;
        }
      }
      if (away_from_belts) {
        admissible_rings.push_back(ring);
      }
    }
  }
  std::sort(admissible_rings.begin(), admissible_rings.end());

  const auto admissible_it =
      std::lower_bound(admissible_rings.begin(), admissible_rings.end(),
                       cut_ring);
  if (admissible_it == admissible_rings.end() ||
      *admissible_it != cut_ring) {
    std::string nearest;
    if (admissible_it != admissible_rings.begin()) {
      nearest = std::to_string(*std::prev(admissible_it));
    }
    if (admissible_it != admissible_rings.end()) {
      if (!nearest.empty()) {
        nearest += ", ";
      }
      nearest += std::to_string(*admissible_it);
    }
    if (nearest.empty()) {
      nearest = "none";
    }
    throw namelist::ConfigError(
        "Mesh.polar_tier_cart_cut_ring=" + std::to_string(cut_ring) +
        " is not an admissible ordinary structured ring; nearest "
        "admissible ring(s): " + nearest);
  }

  const PolarTierScheduleEntry* cut_tier_entry = nullptr;
  for (const PolarTierScheduleEntry& entry : full.schedule) {
    if (entry.kind == PolarTierEntityKind::TIER &&
        cut_ring > entry.ring_begin && cut_ring < entry.ring_end) {
      cut_tier_entry = &entry;
      break;
    }
  }
  if (cut_tier_entry == nullptr) {
    throw namelist::ConfigError(
        "Mesh.polar_tier_cart_cut_ring does not lie in a structured tier");
  }

  PolarTierLayout truncated = full;
  const int kept_cut_tier_rows =
      cut_tier_entry->ring_end - cut_ring;
  const int dropped_cut_tier_rows =
      cut_ring - cut_tier_entry->ring_begin;
  double r_cut = 0.0;
  if (mesh.polar_tier_dendrite_enabled) {
    constexpr std::array<int, 6> tier_radial_indices =
        {{0, 0, 1, 2, 3, 4}};
    TENRYU_ASSERT(cut_tier_entry->index >= 0 &&
                      cut_tier_entry->index <
                          static_cast<int>(tier_radial_indices.size()),
                  "dendrite truncation cut tier index mismatch");
    const int dendrite_tier = cut_tier_entry->index;
    const int radial_index =
        tier_radial_indices[static_cast<std::size_t>(dendrite_tier)];
    int inner_row_offset = 0;
    if (dendrite_tier == 0) {
      inner_row_offset = full.dendrite_tier_radial_rows[1] + 1;
    }
    r_cut =
        full.tier_inner_radii[static_cast<std::size_t>(radial_index)] +
        static_cast<double>(dropped_cut_tier_rows + inner_row_offset) *
            full.tier_radial_spacings[static_cast<std::size_t>(radial_index)];
  } else {
    const std::size_t cut_tier =
        static_cast<std::size_t>(cut_tier_entry->index);
    const double ladder_inner_radius =
        mesh.polar_tier_belt_thickness_frac > 0.0 &&
                cut_tier + 1U < n_tiers
            ? mesh.multiblock_cart_core_r_match -
                  static_cast<double>(
                      full.transition_face_indices[cut_tier]) *
                      full.h_r
            : full.tier_inner_radii[cut_tier];
    r_cut =
        ladder_inner_radius +
        static_cast<double>(dropped_cut_tier_rows) *
            full.tier_radial_spacings[cut_tier];
  }

  truncated.schedule.clear();
  for (const PolarTierScheduleEntry& entry : full.schedule) {
    if (entry.kind == PolarTierEntityKind::FAN) {
      continue;
    }
    if (entry.in_shell_chain) {
      truncated.schedule.push_back(entry);
      continue;
    }
    if (entry.kind == PolarTierEntityKind::SHELL) {
      truncated.schedule.push_back(entry);
      continue;
    }
    if (entry.kind == PolarTierEntityKind::TRANSITION) {
      if (entry.ring_begin >= cut_ring) {
        truncated.schedule.push_back(entry);
      }
      continue;
    }
    if (entry.ring_end <= cut_ring) {
      continue;
    }
    PolarTierScheduleEntry kept_entry = entry;
    if (entry.ring_begin < cut_ring) {
      kept_entry.ring_begin = cut_ring;
      kept_entry.r_inner = r_cut;
    }
    truncated.schedule.push_back(kept_entry);
  }

  const auto resize_base_vectors = [&](const std::size_t kept_tiers,
                                       const std::size_t kept_belts) {
    truncated.tier_columns.resize(kept_tiers);
    truncated.tier_radial_rows.resize(kept_tiers);
    truncated.tier_outer_radii.resize(kept_tiers);
    truncated.tier_inner_radii.resize(kept_tiers);
    truncated.tier_radial_spacings.resize(kept_tiers);
    truncated.transition_face_indices.resize(kept_belts);
    truncated.transition_radii.resize(kept_belts);
    truncated.transition_chi_fine.resize(kept_belts);
    truncated.transition_chi_coarse.resize(kept_belts);
    truncated.transition_outer_radii.resize(
        std::min(kept_belts, truncated.transition_outer_radii.size()));
    truncated.transition_inner_radii.resize(
        std::min(kept_belts, truncated.transition_inner_radii.size()));
    truncated.transition_intermediate_columns.resize(
        std::min(kept_belts,
                 truncated.transition_intermediate_columns.size()));
    truncated.transition_intermediate_radii.resize(
        std::min(kept_belts,
                 truncated.transition_intermediate_radii.size()));
  };

  if (mesh.polar_tier_dendrite_enabled) {
    constexpr std::array<int, 6> tier_radial_indices =
        {{0, 0, 1, 2, 3, 4}};
    constexpr std::array<int, 6> tier_ring_indices =
        {{0, 1, 3, 4, 5, 6}};
    const int cut_dendrite_tier = cut_tier_entry->index;
    const int cut_radial_index =
        tier_radial_indices[static_cast<std::size_t>(cut_dendrite_tier)];
    const std::size_t kept_base_tiers =
        static_cast<std::size_t>(cut_radial_index + 1);
    const std::size_t kept_base_belts =
        static_cast<std::size_t>(cut_radial_index);
    resize_base_vectors(kept_base_tiers, kept_base_belts);
    int kept_base_cut_tier_rows = kept_cut_tier_rows;
    if (cut_dendrite_tier == 1) {
      kept_base_cut_tier_rows =
          full.dendrite_tier_radial_rows[0] + 1 + kept_cut_tier_rows;
    } else if (cut_dendrite_tier == 2) {
      kept_base_cut_tier_rows = 1 + kept_cut_tier_rows;
    }
    truncated.tier_radial_rows[static_cast<std::size_t>(cut_radial_index)] =
        kept_base_cut_tier_rows;
    truncated.tier_inner_radii[static_cast<std::size_t>(cut_radial_index)] =
        r_cut;

    const std::size_t kept_dendrite_tiers =
        static_cast<std::size_t>(cut_dendrite_tier + 1);
    const std::size_t kept_dendrite_transitions =
        static_cast<std::size_t>(
            tier_ring_indices[static_cast<std::size_t>(cut_dendrite_tier)]);
    truncated.dendrite_native_tier_columns.resize(kept_dendrite_tiers);
    truncated.dendrite_actual_tier_columns.resize(kept_dendrite_tiers);
    truncated.dendrite_tier_radial_rows.resize(kept_dendrite_tiers);
    truncated.dendrite_tier_radial_rows[
        static_cast<std::size_t>(cut_dendrite_tier)] = kept_cut_tier_rows;
    truncated.dendrite_transition_joins.resize(
        kept_dendrite_transitions);
    truncated.dendrite_master_theta_node_labels.resize(
        kept_dendrite_transitions + 1U);
    truncated.dendrite_ring_interval_counts.resize(
        kept_dendrite_transitions + 1U);
  } else {
    const std::size_t cut_tier =
        static_cast<std::size_t>(cut_tier_entry->index);
    const std::size_t kept_tiers = cut_tier + 1U;
    const std::size_t kept_belts = cut_tier;
    resize_base_vectors(kept_tiers, kept_belts);
    truncated.tier_radial_rows[cut_tier] = kept_cut_tier_rows;
    truncated.tier_inner_radii[cut_tier] = r_cut;
  }
  truncated.fan_radius = 0.0;

  TENRYU_ASSERT(!truncated.schedule.empty() &&
                    truncated.schedule.front().ring_begin == cut_ring &&
                    truncated.schedule.back().kind ==
                        PolarTierEntityKind::SHELL,
                "truncated polar-tier schedule endpoints mismatch");
  int previous_ring_end = cut_ring;
  for (const PolarTierScheduleEntry& entry : truncated.schedule) {
    TENRYU_ASSERT(entry.ring_begin == previous_ring_end &&
                      entry.ring_end > entry.ring_begin,
                  "truncated polar-tier schedule is not contiguous");
    previous_ring_end = entry.ring_end;
  }
  const PolarTierEntityCounts truncated_totals =
      polar_tier_schedule_totals(mesh, truncated);
  truncated.block_count = static_cast<int>(truncated_totals.n_blocks);
  truncated.n_cells = truncated_totals.n_cells;
  truncated.n_nodes = truncated_totals.n_nodes;
  TENRYU_ASSERT(truncated.block_count ==
                    static_cast<int>(truncated.schedule.size()),
                "truncated polar-tier schedule block count mismatch");

  if (out_info != nullptr) {
    *out_info = PolarTierTruncation{
        cut_ring,
        cut_tier_entry->columns,
        r_cut,
        full.n_cells - truncated.n_cells,
        full.n_nodes - truncated.n_nodes,
        static_cast<long long>(full.block_count - truncated.block_count)};
  }
  return truncated;
}

}  // namespace tenryu::core
