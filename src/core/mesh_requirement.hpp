#pragma once

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "core/namelist/frozen_table.hpp"

namespace tenryu::core {

// Optional campaign provenance. Absence preserves the original numerical path.
struct MeshEmpiricalRequirement {
  bool enabled = false;
  std::string reference_sha256;
  std::vector<std::string> case_ids;
  double surface_ceiling_g_cm2 = 0.0;
  double reference_apriori_g_cm2 = 0.0;
};

inline constexpr const char* kMeshConvergenceReferenceSha256 =
    "9877d1a8c61ead63c6b8a3822ccfc069d149bd849838da79eb5965edd62fd724";
inline constexpr double kMeshEmpiricalRelaxationCap = 9.220252473467259;

// Empty means valid. The independent assistant lint also verifies the prediction.
[[nodiscard]] std::string mesh_empirical_validation_error(
    const MeshEmpiricalRequirement& empirical);

struct MeshRequirementParams {
  bool enabled = true;
  std::string apply = "report";
  int zones_per_scale_length = 9;
  double intensity_exponent = 0.4;
  double intensity_reference_W_cm2 = 1.0e14;
  double scale_length_factor = 0.12;
  double ablation_mass_safety = 1.5;
  double formation_ablated_fraction = 0.1;
  double absorbed_fraction = 1.0;
  int shock_cells_per_separation = 8;
  double shock_event_min_separation_frac = 0.05;
  int min_cells_per_layer = 10;
  double zbar_override = 0.0;
  int n_bands = 6;
  MeshEmpiricalRequirement empirical;
};

struct MeshRequirementMaterial {
  std::string name;
  double A = 0.0;
  double Z = 0.0;
  bool is_void = false;
};

struct MeshRequirementInputs {
  int geometry_code = 0;
  double r_min = 0.0;
  double r_max = 0.0;
  double t_end = 0.0;
  double wavelength_nm = 351.0;
  bool laser_enabled = false;
  namelist::FrozenTable1D power_W;
  std::vector<MeshRequirementMaterial> materials;
  std::function<double(double)> rho0;
  std::function<int(double)> material_at;
  std::vector<double> breakpoints;
  double rho_void_cut = 1.0e-6;
  // Used only for recommendation coverage reporting, never by the model.
  std::string eos = "unknown", temperature_model = "unknown",
              conduction_solver = "unknown";
  bool radiation_enabled = false;
};

struct MeshRequirementProfilePoint {
  double depth_g_cm2, t_abl_s, L_c_cm, ceiling_g_cm2;
};
struct MeshRequirementTrackPoint {
  double t_s, L_c_cm, c_T_cm_s, I_W_cm2;
};
struct MeshRequirementShockEvent {
  double t_s, P_a_Mbar, I_W_cm2;
};
struct MeshRequirementShockPair {
  double t0_s, t1_s, u_s_cm_s, delta_mu_g_cm2;
};
struct MeshRequirementLayer {
  int material;
  double r_lo_cm, r_hi_cm;
};
struct MeshRequirementBand {
  std::string kind;
  double depth_lo_g_cm2, depth_hi_g_cm2, r_lo_cm, r_hi_cm;
  double areal_mass_max_g_cm2, mass_frac_lo, mass_frac_hi;
  double rho_max_gcc = 0.0;   // densest non-void initial density inside [r_lo, r_hi]
  double width_max_cm = 0.0;  // areal_mass_max_g_cm2 / rho_max_gcc (cell width a cell at the densest point may have and still respect the ceiling); +inf when no non-void sample
};
struct MeshRequirementMaterialInfo {
  std::string name;
  double A, Z, zbar, rho_c_gcc;
  bool is_void;
  std::string zbar_source;
};

struct MeshRequirementReport {
  bool applicable = false;
  std::string reason;
  MeshRequirementParams params;
  double wavelength_nm = 0.0;
  int geometry_code = 0;
  double r_min = 0.0, r_max = 0.0, R0_cm = 0.0, area_cm2 = 0.0,
         t_end_s = 0.0;
  int n_samples = 0;
  double peak_intensity_W_cm2 = 0.0, fluence_J_cm2 = 0.0;
  int ablator_material = -1;
  std::vector<MeshRequirementMaterialInfo> materials;
  double intensity_correction_factor = 1.0, mu_abl_total_g_cm2 = 0.0,
         ablated_mass_fraction = 0.0,
         depth_formation_g_cm2 = 0.0, t_formation_s = 0.0,
         L_c_formation_cm = 0.0, ceiling_formation_g_cm2 = 0.0;
  double dr_min_admissible_cm = std::numeric_limits<double>::infinity();
  std::vector<MeshRequirementProfilePoint> profile;
  std::vector<MeshRequirementTrackPoint> scale_length_track;
  bool shock_applicable = false;
  std::vector<MeshRequirementShockEvent> shock_events;
  std::vector<MeshRequirementShockPair> shock_pairs;
  double shock_ceiling_g_cm2 = 0.0, rho_payload_gcc = 0.0;
  std::vector<MeshRequirementLayer> layers;
  std::vector<MeshRequirementBand> bands_recommended;
  std::vector<std::string> notes;
  std::vector<double> table_t_s, table_mu_abl_g_cm2, table_L_c_cm;
  double apriori_ceiling_formation_g_cm2 = 0.0;
  struct ConditionLayer {
    std::string material;
    double rho_gcc, thickness_cm;
  };
  std::vector<ConditionLayer> condition_layers;
  std::vector<double> condition_intensity_W_cm2;
  std::string eos, temperature_model, conduction_solver;
  bool radiation_enabled = false;
};

struct MeshRequirementRuleCheck {
  bool applicable = false;
  int n_checked = 0, n_violations = 0;
  double max_ratio = 0.0;
  int worst_cell = -1;
  double worst_r_lo_cm = 0.0, worst_r_hi_cm = 0.0,
         worst_areal_mass_g_cm2 = 0.0, worst_ceiling_g_cm2 = 0.0;
};
struct MeshRequirementLayerViolation {
  int material;
  int n_cells;
  int required;
};
struct MeshRequirementCheck {
  bool ok = true;
  int n_cells = 0;
  MeshRequirementRuleCheck ablation, shock;
  bool layers_ok = true;
  std::vector<MeshRequirementLayerViolation> layer_violations;
};

[[nodiscard]] MeshRequirementReport build_mesh_requirement(
    const MeshRequirementInputs& in, const MeshRequirementParams& params);

[[nodiscard]] MeshRequirementCheck check_mesh_requirement(
    const MeshRequirementReport& report, const MeshRequirementInputs& in,
    const std::vector<double>& nodes, const std::vector<double>& rho_cells,
    const std::vector<int>& material_cells);

[[nodiscard]] double mesh_requirement_cell_areal_mass(
    int geometry_code, double R0_cm, double rho, double r_lo, double r_hi);

[[nodiscard]] std::string mesh_requirement_json(
    const MeshRequirementReport& report, const MeshRequirementCheck* check);

}  // namespace tenryu::core
