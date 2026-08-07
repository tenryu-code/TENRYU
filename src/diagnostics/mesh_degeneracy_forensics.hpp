#pragma once

#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::diagnostics {

struct MeshDegeneracyForensicsRecord {
  struct NodeData {
    double r_old = 0.0;
    double z_old = 0.0;
    double r_candidate = 0.0;
    double z_candidate = 0.0;
    double dr = 0.0;
    double dz = 0.0;
    double dr_over_dt = 0.0;
    double dz_over_dt = 0.0;
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

  struct CornerJSourceBudget {
    std::array<double, 4> dJ_pressure{};
    std::array<double, 4> dJ_scalar_qvisc{};
    std::array<double, 4> dJ_anti_hourglass{};
    std::array<double, 4> dJ_predictor{};
    std::array<double, 4> dJ_corrector{};
    std::array<double, 4> dJ_ale_rezone{};
    std::array<double, 4> dJ_remap{};
    std::array<double, 4> J_post_predictor{};
    std::array<double, 4> J_post_corrector{};
    std::array<double, 4> J_post_ale_rezone{};
    std::array<double, 4> J_post_remap{};
    bool predictor_ran = false;
    bool corrector_ran = false;
    bool ale_rezone_ran = false;
    bool remap_ran = false;
    bool anti_hourglass_active = false;
    int dominant_source_corner = -1;
    int dominant_source_kind = -1;
    double dominant_source_dJ = 0.0;
  };

  int step = 0;
  int retry_index = 0;
  double dt = 0.0;
  double t = 0.0;
  std::string stage;
  std::string reason;
  int cell = -1;
  int i = -1;
  int j = -1;
  int failing_corner = -1;

  double J0 = 0.0;
  double J_floor = 0.0;
  double J1 = 0.0;
  double dJ_dsigma_at_0 = 0.0;
  double sigma_root_reported = 1.0;
  double sigma_root_recomputed = 1.0;
  double J_at_sigma_0 = 0.0;
  double J_at_sigma_010 = 0.0;
  double J_at_sigma_safe = 0.0;
  double J_at_sigma_050 = 0.0;
  double J_at_sigma_100 = 0.0;

  std::array<NodeData, 4> nodes{};

  double position_hourglass_amplitude_r = 0.0;
  double position_hourglass_amplitude_z = 0.0;
  double velocity_hourglass_amplitude_r = 0.0;
  double velocity_hourglass_amplitude_z = 0.0;
  double hourglass_to_affine_gradient_ratio = 0.0;
  double hourglass_to_sound_speed_ratio = 0.0;

  std::array<double, 4> volfrac_failing_cell{};
  std::array<std::string, 4> material_names{};
  bool is_mixed = false;
  double plic_normal_r = 0.0;
  double plic_normal_z = 0.0;
  double plic_centroid_r = 0.0;
  double plic_centroid_z = 0.0;
  double rho_failing_cell = 0.0;
  double Te_failing_cell = 0.0;
  double Ti_failing_cell = 0.0;
  double pressure_failing_cell = 0.0;
  double sound_speed_failing_cell = 0.0;
  double Q_visc_failing_cell = 0.0;
  double div_u_failing_cell = 0.0;
  double vorticity_failing_cell = 0.0;
  double laser_deposition_failing_cell = 0.0;

  double PdV_work = 0.0;
  double viscous_work = 0.0;
  double force_dot_velocity_work = 0.0;
  double kinetic_energy_change_4_nodes = 0.0;
  double local_total_energy_residual = 0.0;

  int same_cell_consecutive_count = 0;
  double sigma_band_min_recent = 1.0;
  double sigma_band_max_recent = 1.0;
  bool sigma_dt_invariant_confirmed = false;

  CornerJSourceBudget corner_j_source_budget{};
  bool corner_j_source_budget_enabled = false;
};

struct VelocityHistoryRecord {
  int step = 0;
  double t = 0.0;
  double dt = 0.0;
  int cell = -1;
  int i = -1;
  int j = -1;
  std::array<double, 4> r_old{};
  std::array<double, 4> z_old{};
  std::array<double, 4> u_r_old{};
  std::array<double, 4> u_z_old{};
  double velocity_hourglass_amplitude_r = 0.0;
  double velocity_hourglass_amplitude_z = 0.0;
  double hourglass_to_affine_gradient_ratio = 0.0;
  std::string phase;
};

struct MeshDegeneracyFailureEvent {
  int step = 0;
  int cell = -1;
  int corner = -1;
  int stage = 0;
  std::string reason;
  double sigma_safe = 1.0;
};

struct MeshDegeneracyTriggerDecision {
  bool eligible_reason = false;
  bool emit = false;
  int same_cell_consecutive_count = 0;
  double sigma_band_min_recent = 1.0;
  double sigma_band_max_recent = 1.0;
  bool sigma_dt_invariant_confirmed = false;
  int total_dumps_emitted = 0;
};

struct MeshDegeneracyForensicsPhasePositions {
  bool valid = false;
  bool ran = false;
  std::array<double, 4> r{};
  std::array<double, 4> z{};
};

struct MeshDegeneracyForensicsState {
  struct PhaseSnapshot {
    bool valid = false;
    bool ran = false;
    int step = -1;
    std::vector<double> r;
    std::vector<double> z;
  };

  int active_step = -1;
  int last_cell = -1;
  int last_corner = -1;
  int last_stage = -1;
  std::string last_reason;
  int same_cell_consecutive_count = 0;
  int total_dumps_emitted = 0;
  double sigma_band_min_recent = 1.0;
  double sigma_band_max_recent = 1.0;
  PhaseSnapshot post_lagrange_snapshot;
  PhaseSnapshot post_ale_rezone_snapshot;
  PhaseSnapshot post_remap_snapshot;
  int first_failure_cell_c = -1;
  int velocity_history_records_emitted = 0;

  MeshDegeneracyTriggerDecision observe_failure(
      const core::Config::NumericsConfig::DiagnosticsConfig::
          MeshDegeneracyForensicsConfig& cfg,
      const MeshDegeneracyFailureEvent& event);

  void reset_corner_j_source_budget_phase_snapshots(int step);
  void snapshot_corner_j_source_budget_phase(const core::State& state,
                                             const char* phase,
                                             bool ran);
  bool extract_corner_j_source_budget_phase_positions(
      const char* phase,
      int cell,
      int nz,
      MeshDegeneracyForensicsPhasePositions& out) const;
};

struct MeshDegeneracyForensicsContext {
  int retry_index = 0;
  double dt = 0.0;
  double t = 0.0;
  std::string stage;
  std::string reason;
  int cell = -1;
  int i = -1;
  int j = -1;
  int failing_corner = -1;
  double sigma_root_reported = 1.0;
  int same_cell_consecutive_count = 0;
  double sigma_band_min_recent = 1.0;
  double sigma_band_max_recent = 1.0;
  bool sigma_dt_invariant_confirmed = false;
  std::array<MeshDegeneracyForensicsRecord::NodeData, 4> nodes{};
  bool nodes_valid = false;
  bool anti_hourglass_active = false;
  MeshDegeneracyForensicsPhasePositions post_lagrange;
  MeshDegeneracyForensicsPhasePositions post_ale_rezone;
  MeshDegeneracyForensicsPhasePositions post_remap;
};

struct BilinearFitCoefficients {
  double a0 = 0.0;
  double a_xi = 0.0;
  double a_eta = 0.0;
  double a_xi_eta = 0.0;
};

double compute_corner_j_at_sigma(const std::array<double, 4>& r_old,
                                 const std::array<double, 4>& z_old,
                                 const std::array<double, 4>& r_candidate,
                                 const std::array<double, 4>& z_candidate,
                                 int corner,
                                 double sigma);

BilinearFitCoefficients fit_hourglass_amplitude(
    const std::array<double, 4>& values);

double recompute_corner_j_floor_root(const std::array<double, 4>& r_old,
                                     const std::array<double, 4>& z_old,
                                     const std::array<double, 4>& r_candidate,
                                     const std::array<double, 4>& z_candidate,
                                     int corner,
                                     double j_floor);

double compute_local_total_energy_residual(double kinetic_energy_change_4_nodes,
                                           double PdV_work,
                                           double viscous_work,
                                           double force_dot_velocity_work);

MeshDegeneracyForensicsRecord::CornerJSourceBudget
compute_corner_j_source_budget(const MeshDegeneracyForensicsContext& context);

MeshDegeneracyForensicsRecord build_mesh_degeneracy_forensics_record(
    const core::State& state,
    const core::Config& cfg,
    const MeshDegeneracyForensicsContext& context);

std::string serialize_mesh_degeneracy_forensics_record_jsonl(
    const MeshDegeneracyForensicsRecord& record);

std::filesystem::path mesh_degeneracy_forensics_output_path(
    const core::Config& cfg);

std::filesystem::path mesh_degeneracy_velocity_history_output_path(
    const core::Config& cfg,
    int i,
    int j);

void append_mesh_degeneracy_forensics_record_jsonl(
    const core::Config& cfg,
    const MeshDegeneracyForensicsRecord& record);

VelocityHistoryRecord build_velocity_history_record(const core::State& state,
                                                    int cell,
                                                    std::string phase,
                                                    int step,
                                                    double t,
                                                    double dt);

std::string serialize_velocity_history_record_jsonl(
    const VelocityHistoryRecord& record);

void append_velocity_history_record_jsonl(const core::Config& cfg,
                                          const VelocityHistoryRecord& record);

std::vector<int> velocity_history_cells_for_target(const core::Config& cfg,
                                                   int target_cell);

int emit_velocity_history(const core::State& state,
                          const core::Config& cfg,
                          int target_cell,
                          const std::string& phase,
                          int step,
                          double t,
                          double dt,
                          MeshDegeneracyForensicsState* forensics_state);

int emit_velocity_history(const core::State& state,
                          const core::Config& cfg,
                          int target_cell,
                          const std::string& phase);

}  // namespace tenryu::diagnostics

namespace tenryu::diagnostics::mesh_degeneracy_forensics {

using ::tenryu::diagnostics::VelocityHistoryRecord;
using ::tenryu::diagnostics::append_velocity_history_record_jsonl;
using ::tenryu::diagnostics::build_velocity_history_record;
using ::tenryu::diagnostics::emit_velocity_history;
using ::tenryu::diagnostics::mesh_degeneracy_velocity_history_output_path;
using ::tenryu::diagnostics::serialize_velocity_history_record_jsonl;
using ::tenryu::diagnostics::velocity_history_cells_for_target;

}  // namespace tenryu::diagnostics::mesh_degeneracy_forensics
