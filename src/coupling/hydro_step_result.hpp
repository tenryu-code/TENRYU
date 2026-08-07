#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "hydro/cap_energy_audit.hpp"
#include "hydro/mesh_regime.hpp"
#include "mesh/mesh_geometry_result.hpp"

namespace tenryu::coupling {

enum class HydroFailureStage : std::uint8_t {
  Default = 0,
  StepStart,
  PredictorCommit,
  PostPredictor,
  PostCorrector,
  RetryRestoreValidate,
};

enum class RetryActionHint : std::uint8_t {
  None = 0,
  ReduceDtOnly,
  RepairOnly,
  ForceAxisSpinePlusLocalAle,
  ForceBoundaryPatchRepair,
  ForceCdLocalRezone,
  ForceInteriorMultiNodeRepair,
  ForceFullWinslow,
};

// Result returned from a hydro half-step. Default construction preserves the
// baseline path when driver-level retry is not wired.
struct HydroStepResult {
  struct MeshForensicsNodeSample {
    double r_old = 0.0;
    double z_old = 0.0;
    double r_candidate = 0.0;
    double z_candidate = 0.0;
    double u_r_old = 0.0;
    double u_z_old = 0.0;
    double u_r_half = 0.0;
    double u_z_half = 0.0;
    double u_r_new = 0.0;
    double u_z_new = 0.0;
    double a_r_pressure = 0.0;
    double a_z_pressure = 0.0;
    double a_r_qvisc = 0.0;
    double a_z_qvisc = 0.0;
    double a_r_hourglass = 0.0;
    double a_z_hourglass = 0.0;
    double node_mass = 0.0;
    double corner_mass_into_cell = 0.0;
  };

  bool admissible = true;
  bool retry_required = false;
  const char* reason = "";
  int first_failing_cell = -1;
  int first_failing_corner = -1;
  // Legacy generic detector-populated minimum-quality scalar. Existing
  // producers keep their current meaning, e.g. trial-volume ratio or corner-J.
  double min_metric = 0.0;
  double suggested_dt = 0.0;

  HydroFailureStage stage = HydroFailureStage::Default;
  tenryu::hydro::MeshFailureRegime regime =
      tenryu::hydro::MeshFailureRegime::Unknown;
  RetryActionHint retry_action = RetryActionHint::None;
  tenryu::mesh::MeshGeometryFailureKind geometry_failure_kind =
      tenryu::mesh::MeshGeometryFailureKind::None;
  int first_failing_i = -1;
  int first_failing_j = -1;
  double failing_value = 0.0;
  double min_cell_vol = 0.0;
  double min_cell_area = 0.0;
  // Corner-J-specific signed minimum. Wave 1 leaves this at the default; later
  // in-hydro guards/classifiers may populate it.
  double min_corner_j = 0.0;
  bool gauss_j_failed = false;
  bool rz_volume_failed = false;
  double min_gauss_j_trial = std::numeric_limits<double>::infinity();
  double min_rz_volume_trial = std::numeric_limits<double>::infinity();
  double trial_scale = 0.0;
  int path_source_kind = 0;
  int path_metric_kind = 0;
  int path_block_id = -1;
  int path_local_i = -1;
  int path_local_j = -1;
  int path_old_geometry_inadmissible = 0;
  double min_path_margin = std::numeric_limits<double>::infinity();
  int min_path_margin_cell = -1;
  double q_balance = 0.0;
  int macro_q_begin = -1;
  int macro_q_end = -1;
  int macro_j_begin = -1;
  int macro_j_end = -1;
  int macro_span = 0;
  int macro_level = 0;
  double tau_warn = 0.0;
  double tau_zero = 0.0;
  double q_warn = 0.0;
  double forecast_q_min_now = std::numeric_limits<double>::infinity();
  double forecast_q_min_end = std::numeric_limits<double>::infinity();
  double forecast_q_min_path = std::numeric_limits<double>::infinity();
  int forecast_seed_count = 0;
  bool forecast_endpoint_valid = false;
  int forecast_first_cell = -1;
  int forecast_block_id = -1;
  int forecast_local_i = -1;
  int forecast_local_j = -1;
  std::vector<std::uint8_t> forecast_seed_mask;
  bool shell_subcycle_committed = false;
  double cap_energy_audit_W_ext_step = 0.0;
  double cap_energy_audit_W_pq = 0.0;
  double cap_energy_audit_W_uF = 0.0;
  double cap_energy_audit_W_apex_pinned = 0.0;
  tenryu::hydro::I1BSpuriousSensorSummary i1b_hydro_nonaffine_sensor{};
  // Stage 24 axis-classification fields. Populated when telemetry is enabled
  // or dispatcher_state_sensitive_bypass_enabled is active.
  bool state_sensitive = false;
  bool dt_sensitive = false;
  double dt_star_est = 0.0;
  double margin_rel = 0.0;
  int repair_generation = 0;
  int rung_reached = 0;
  bool axis_variational_projection_eligible = false;
  const char* axis_variational_projection_blocked_reason = "";
  int floor_stall_count = 0;
  bool mesh_forensics_sample_valid = false;
  std::array<MeshForensicsNodeSample, 4> mesh_forensics_nodes{};
};

}  // namespace tenryu::coupling
