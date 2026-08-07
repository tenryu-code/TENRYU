#pragma once

#include <array>
#include <functional>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "diagnostics/diagnostics.hpp"
#include "hydro/ale_mode.hpp"
#include "hydro/cap_energy_audit.hpp"
#include "hydro/mesh_regime.hpp"
#include "parallel/partition.hpp"

namespace tenryu::parallel {
struct CommBuffers;
class Reduction;
}  // namespace tenryu::parallel

namespace tenryu::coupling {
struct ProfileObservability;
}

namespace tenryu::mesh {
struct MeshForecast;
}

namespace tenryu::hydro {

struct HydroEOSContext;

namespace ale {

inline bool ale_identity_mode_enabled(const core::Config& cfg) noexcept {
  return cfg.numerics.ale.ale_identity_mode;
}

struct AleDriverRetryContext {
  bool active = false;
  enum class Reason { none, corner_j, gauss_j, rz_volume } reason = Reason::none;
  bool post_ale_corner_balance_bad = false;
  int first_cell = -1;
  int first_corner = -1;
};

enum class AleStatus {
  Ok,
  PreRezoneInvalid,
  NoLambdaAdmissible,
  // The center-patch final admissibility check failed on an active block-0
  // cell and a central pseudo-core ring-absorption request was armed; the
  // driver must restore the full-step retry snapshot and retry the step.
  CenterPatchInadmissibleAbsorbRetry,
};

struct SafeBacktrackSearchResult {
  AleStatus status = AleStatus::Ok;
  double accepted_lambda = 0.0;
  int trials = 0;
};

struct AleStepResult {
  bool applied = false;
  bool rezone_triggered = false;
  bool rezone_converged = true;
  bool quality_rollback = false;
  double quality_min = 1.0;
  int rezone_iterations = 0;
  double rezone_residual = 0.0;
  double momentum_rel_err = 0.0;
  double mass_floor_delta = 0.0;
  std::vector<double> mass_floor_delta_per_material;
  double E_floor_injected = 0.0;
  double E_redistribution_unresolved = 0.0;
  double cap_energy_audit_D_K = 0.0;
  double eta_contact_step = 0.0;
  double eta_contact_cumulative = 0.0;
  double center_patch_sigma_accepted = -1.0;
  bool energy_audit_identity_skip = false;
  bool energy_audit_ale_fallback = false;
  tenryu::hydro::I1BSpuriousSensorSummary i1b_ale_ke_sensor{};
  int per_material_repair_count = 0;
  int accepted_remap_count = 0;
  int volfrac_degenerate = 0;
  int plic_interface_cells = 0;
  int plic_reconstruction_attempts = 0;
  int plic_reconstruction_successes = 0;
  int plic_axis_exempt_cells = 0;
  int plic_class_d_events = 0;
  int plic_repair_events = 0;
  bool plic_remap_used = false;
  bool plic_remap_fallback_engaged = false;
  bool plic_drift_triggered = false;
  bool plic_cf6_density_remap_used = false;
  double plic_max_volume_fraction_residual = 0.0;
  double plic_max_interface_centroid_drift_relative = 0.0;
  double plic_max_swept_fraction = 0.0;
  diagnostics::AleClosureAuditDiagnostics closure_audit{};
  // Worst CSR remap total-mass closure |(post-pre)/pre| observed during
  // this ALE phase (signed; see AleRemap2DRZResult::mass_closure_rel). The
  // driver rejects the step through the full-step retry snapshot when it
  // exceeds TENRYU_I1B_REMAP_CLOSURE_REJECT_TOL.
  double remap_mass_closure_rel = 0.0;
  // Stage 24 Wave 1: AleMode whose RezoneResult was accepted by the
  // request/escalation quality gate. ScheduledDefault means no request-mode
  // rezone was accepted.
  AleMode effective_mode_executed = AleMode::ScheduledDefault;
  AleStatus status = AleStatus::Ok;
};

// Rezone closure cooldown: armed by the driver when a committed rezone's
// remap violated total-mass closure and the step was rejected/restored. The
// Winslow center-patch and per-block rezone initiators (and non-forced axis
// rezone fires) stand down through the cooldown so the retried step does not
// immediately re-fire the violating mechanism.
void arm_rezone_closure_cooldown(int until_step);
bool rezone_closure_cooldown_active(int step);

struct ShellMomentumTopNode {
  int node = -1;
  int node_class = 0;
  double abs_dPr = 0.0;
  double dPr = 0.0;
  double dm = 0.0;
  double dvr = 0.0;
  double dvz = 0.0;
  bool mass_changed = false;
  bool velocity_changed = false;
};

struct ShellVelocityCheckNode {
  int node = -1;
  int flags = 0;
  double dv = 0.0;
  double m_before = 0.0;
  double m_after = 0.0;
  double vr_before = 0.0;
  double vr_after = 0.0;
  double vz_before = 0.0;
  double vz_after = 0.0;
};

struct ShellProtectedRezoneResult {
  bool attempted = false;
  bool succeeded = false;
  bool remap_applied = false;
  bool q_release_restored = false;
  bool rezone_path_valid = false;
  bool endpoint_valid_after = false;
  std::string status;
  double q_floor = 0.0;
  double q_warn = 0.0;
  double q_release = 0.0;
  double q_min_before = std::numeric_limits<double>::infinity();
  double q_min_after = -std::numeric_limits<double>::infinity();
  double q_min_path = -std::numeric_limits<double>::infinity();
  double tau_zero_after = 1.0;
  double sigma_accepted = 0.0;
  double max_active_delta = 0.0;
  double max_frozen_delta = 0.0;
  double max_boundary_dx = 0.0;
  double max_boundary_dvr = 0.0;
  double max_boundary_dvz = 0.0;
  double max_boundary_dm = 0.0;
  double dM = 0.0;
  double dPr = 0.0;
  double dPz = 0.0;
  double dE = 0.0;
  double dE_internal = 0.0;
  double dKE = 0.0;
  double dE_int_transport_residual = 0.0;
  double dKE_reconstruction_residual = 0.0;
  double dE_floor_deposit = 0.0;
  double dE_active_floor = 0.0;
  double dE_unexplained = 0.0;
  double dPr_boundary = 0.0;
  double dPz_boundary = 0.0;
  double dPr_active = 0.0;
  double dPz_active = 0.0;
  double p_scale = 0.0;
  bool transport_momentum_valid = false;
  double transport_pr = 0.0;
  double transport_pz = 0.0;
  double R_pi_r = 0.0;
  double R_pi_z = 0.0;
  double R_assm_r = 0.0;
  double R_assm_z = 0.0;
  double R_rec_r = 0.0;
  double R_rec_z = 0.0;
  double R_E = 0.0;
  int discarded_dual_faces = 0;
  double discarded_dual_mass = 0.0;
  double discarded_dual_pi_r = 0.0;
  double discarded_dual_pi_z = 0.0;
  int n_ext_cell_cmass_changed = 0;
  int n_ext_node_vel_changed = 0;
  int n_vel_changed_outside_velmask = 0;
  int first_ext_cell_cmass_changed = -1;
  int first_ext_node_vel_changed = -1;
  int max_ext_cell_cmass_changed_cell = -1;
  int max_ext_node_vel_changed_node = -1;
  double max_ext_cell_cmass_delta = 0.0;
  double max_ext_node_vel_delta = 0.0;
  std::array<ShellMomentumTopNode, 5> top_momentum_nodes{};
  std::array<ShellVelocityCheckNode, 4> velchk_nodes{};
  double active_floor_mass_delta = 0.0;
  int n_eint_floor_hits = 0;
  int n_active_floor_hits = 0;
  int seed_cells = 0;
  int shell_seed_cells = 0;
  int patch_cells = 0;
  int support_cells = 0;
  int affected_nodes = 0;
  int energy_closure_cells = 0;
  int energy_collar_cells = 0;
  int momentum_transport_cells = 0;
  int momentum_transport_collar_cells = 0;
  int momentum_transport_iterations = 0;
  int first_energy_collar_cell = -1;
  std::array<int, 8> energy_collar_sample{{-1, -1, -1, -1, -1, -1, -1, -1}};
  int closure_hard_frozen_nodes = 0;
  int closure_hsrc_axis_pole_center = 0;
  int closure_hsrc_core = 0;
  int closure_hsrc_deref = 0;
  int shrink_iterations = 0;
  int M_size_initial = 0;
  int M_size_final = 0;
  int n_M_removed_for_closure = 0;
  std::array<int, 4> closure_deref_node_id{{-1, -1, -1, -1}};
  std::array<int, 4> closure_deref_node_block{{-1, -1, -1, -1}};
  std::array<int, 4> closure_deref_node_local_i{{-1, -1, -1, -1}};
  std::array<int, 4> closure_deref_node_local_j{{-1, -1, -1, -1}};
  std::array<int, 4> closure_deref_node_subtype{{0, 0, 0, 0}};
  std::array<int, 8> closure_removed_M_node_id{{-1, -1, -1, -1, -1, -1, -1, -1}};
  std::array<int, 8> closure_removed_M_node_block{{-1, -1, -1, -1, -1, -1, -1, -1}};
  std::array<int, 8> closure_removed_M_node_local_i{{-1, -1, -1, -1, -1, -1, -1, -1}};
  std::array<int, 8> closure_removed_M_node_local_j{{-1, -1, -1, -1, -1, -1, -1, -1}};
  int affected_boundary_outer_nodes = 0;
  int closure_core_cells = 0;
  int active_nodes = 0;
  int boundary_nodes = 0;
  int frozen_nodes = 0;
  int winslow_iterations = 0;
  int barrier_sweeps = 0;
  int line_search_iters = 0;
  int first_bad_cell = -1;
};

namespace detail {

SafeBacktrackSearchResult select_safe_backtrack_lambda(
    int min_exp,
    int binary_iters,
    const std::function<bool(double)>& admissible_at_lambda);

int safe_backtrack_lambda_bin(double lambda, int max_exp);

mesh::MultiBlockTopology m1_forward_topology(const core::State& state);

}  // namespace detail

void reset_safe_backtrack_lambda_distribution();
void log_safe_backtrack_lambda_distribution(int max_exp);

void log_cell113_substage_trace(const core::State& state,
                                const core::Config& cfg,
                                const parallel::PartitionInfo& part,
                                const char* stage,
                                double dt_s,
                                double t_s,
                                const char* extra = nullptr);

void record_estep_trace_boundary(const core::State& state,
                                  const core::Config& cfg,
                                  const parallel::PartitionInfo& part,
                                  const parallel::Reduction* reduction,
                                  const char* stage,
                                  double dt_s = 0.0,
                                  double t_s = 0.0);

AleStepResult apply_ale(core::State& state, const core::Config& cfg,
                        const parallel::PartitionInfo& part = parallel::PartitionInfo{},
                        parallel::CommBuffers* bufs = nullptr,
                        const parallel::Reduction* reduction = nullptr,
                        const HydroEOSContext* eos_ctx = nullptr,
                        double dt_hydro_used = 0.0,
                        bool force_rezone = false,
                        const char* force_reason = nullptr,
                        tenryu::coupling::ProfileObservability* observability = nullptr,
                        const AleDriverRetryContext* retry_context = nullptr);

AleStepResult apply_ale_with_request(
    core::State& state, const core::Config& cfg,
    const AleRequest* request,
    const CellRegime* d_cell_regime,
    const parallel::PartitionInfo& part = parallel::PartitionInfo{},
    parallel::CommBuffers* bufs = nullptr,
    const parallel::Reduction* reduction = nullptr,
    const HydroEOSContext* eos_ctx = nullptr,
    double dt_hydro_used = 0.0,
    bool force_rezone = false,
    const char* force_reason = nullptr,
    tenryu::coupling::ProfileObservability* observability = nullptr,
    const AleDriverRetryContext* retry_context = nullptr);

AleStepResult apply_pole_axis_radial_order_repair(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part = parallel::PartitionInfo{},
    const parallel::Reduction* reduction = nullptr,
    const HydroEOSContext* eos_ctx = nullptr,
    double dt_hydro_used = 0.0);

ShellProtectedRezoneResult shell_protected_rezone(
    core::State& state,
    const core::Config& cfg,
    const std::vector<std::uint8_t>& seed_mask,
    const HydroEOSContext* eos_ctx = nullptr,
    double dt_hydro_used = 0.0,
    const parallel::Reduction* reduction = nullptr);

bool maybe_run_shell_rezone_replay_probe(
    core::State& state,
    const core::Config& cfg,
    const mesh::MeshForecast& forecast,
    const double* d_xr_source,
    const double* d_xz_source,
    const double* d_v_r,
    const double* d_v_z,
    double path_dt,
    const char* stage,
    const HydroEOSContext* eos_ctx = nullptr,
    const parallel::Reduction* reduction = nullptr);

}  // namespace ale
}  // namespace tenryu::hydro
