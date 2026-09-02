#pragma once

#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>

#include "core/config.hpp"
#include "core/state.hpp"
#include "coupling/driver_retry_snapshot.hpp"
#include "coupling/profile_observability.hpp"
#include "hydro/ale_mode.hpp"
#include "hydro/reference_barrier_ale.hpp"
#include "io/hdf5_reader.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::io {
class OutputManager;
}

namespace tenryu::hydro {
struct HydroEOSContext;
}

namespace tenryu::coupling {

enum class SplittingOrder {
  STRANG,
  SEQUENTIAL
};

struct SplitOperatorCallbacks {
  std::function<HydroStepResult(double, double, const char*)> hydro;
  std::function<void(double, double)> conduction;
  std::function<void(double, double)> laser;
  std::function<void(double, double)> burn;
  std::function<void(double, double)> radiation;
};

void execute_split_operators(SplittingOrder order,
                             double dt,
                             double t_n,
                             const SplitOperatorCallbacks& callbacks);

void initialize_eos_fields_if_needed(tenryu::core::State& state,
                                     tenryu::core::Config& cfg);

class Driver {
 public:
  explicit Driver(SplittingOrder order = SplittingOrder::STRANG) : order_(order) {}

  void run(tenryu::core::State& state, const tenryu::core::Config& cfg);
  void run(tenryu::core::State& state,
           const tenryu::core::Config& cfg,
           tenryu::io::OutputManager& out);
  void set_restart_photon_pool(tenryu::radiation::PhotonPool&& pool);
  void set_restart_checkpoint_prefix(std::string prefix) {
    restart_checkpoint_prefix_ = std::move(prefix);
  }
  void set_initial_plic_interface_cells_observed(int count) {
    initial_plic_interface_cells_observed_ = count;
  }
  void set_checkpoint_per_material_status(
      tenryu::io::PerMaterialCheckpointReadStatus status) {
    checkpoint_per_material_status_ = status;
  }

  void set_splitting_order(SplittingOrder order) { order_ = order; }
  [[nodiscard]] SplittingOrder splitting_order() const noexcept { return order_; }

 private:
  SplittingOrder order_ = SplittingOrder::STRANG;
  std::optional<tenryu::radiation::PhotonPool> restart_pool_;
  std::string restart_checkpoint_prefix_;
  int initial_plic_interface_cells_observed_ = 0;
  tenryu::io::PerMaterialCheckpointReadStatus checkpoint_per_material_status_ =
      tenryu::io::PerMaterialCheckpointReadStatus::MissingGroupEnabled;
  tenryu::coupling::DriverRetrySnapshot retry_snapshot_;
  tenryu::coupling::ProfileObservability profile_observability_;
};

double compute_dt(const tenryu::core::State& state,
                  const tenryu::core::Config& cfg,
                  double t_end);

struct DtLineage {
  int cycle = 0;
  double t_s = 0.0;
  double dt_chosen = 0.0;
  double dt_uncapped_by_contact = 0.0;
  // Set instead of asserting when dt collapses below Numerics.dt.min_s and
  // the caller passed defer_dt_floor_abort: the caller must either rescue
  // (e.g. central pseudo-core ring absorption) or re-raise the abort.
  bool dt_floor_abort_pending = false;
  double dt_hydro = 0.0;
  double dt_cond = 0.0;
  double dt_burn = 0.0;
  double dt_hot_e = 0.0;
  double dt_visc = 0.0;
  double dt_rad = 0.0;
  double dt_growth = 0.0;
  double dt_output = 0.0;
  double dt_remaining = 0.0;
  double dt_max_namelist = 0.0;
  std::string limiter;
  int hydro_argmin_cell = -1;
  double hydro_argmin_sqrt_area = 0.0;
  double hydro_argmin_cs = 0.0;
  int hydro_argmin_i = -1;
  int hydro_argmin_j = -1;
  double hydro_argmin_dt_at_cell = std::numeric_limits<double>::infinity();
  double hydro_argmin_rho = 0.0;
  double hydro_argmin_u_z = 0.0;
  std::string hydro_min_term = "other";
  std::string hydro_min_term_detail = "unknown";
  std::string hydro_min_other_term = "none";
  int hydro_min_cell = -1;
  int hydro_min_edge = -1;
  int hydro_min_node0 = -1;
  int hydro_min_node1 = -1;
  int hydro_min_other_node = -1;
  double hydro_min_L = 0.0;
  double hydro_min_du = 0.0;
  double hydro_min_accel = 0.0;
  double hydro_min_coefficient = 0.0;
  double hydro_min_raw_dt = std::numeric_limits<double>::infinity();
  double hydro_min_rho = 0.0;
  double hydro_min_mu = 0.0;
  double hydro_min_polar_lambda = 0.0;
  double hydro_min_polar_sigma = 0.0;
  double hydro_min_csw98_du_eff = 0.0;
  double hydro_min_tensor_rho = 0.0;
  double hydro_min_tensor_L = 0.0;
  double hydro_min_tensor_mu = 0.0;
  double hydro_min_crossing_dr = 0.0;
  double hydro_min_crossing_closing = 0.0;
  double hydro_min_crossing_safety = 0.0;
  double dt_hydro_acoustic = std::numeric_limits<double>::infinity();
  double dt_hydro_post_shock = std::numeric_limits<double>::infinity();
  double dt_hydro_edge_av = std::numeric_limits<double>::infinity();
  double dt_hydro_subzonal = std::numeric_limits<double>::infinity();
  int hydro_min_cell_raw = -1;
  double dt_hydro_edge_accel = std::numeric_limits<double>::infinity();
  double dt_hydro_axis_margin = std::numeric_limits<double>::infinity();
  double dt_hydro_volume_rate = std::numeric_limits<double>::infinity();
  double dt_hydro_tri_fan_center = std::numeric_limits<double>::infinity();
  double dt_hydro_corner_j_predict = std::numeric_limits<double>::infinity();
  int tri_fan_center_cfl_cell_id = -1;
  int tri_fan_center_cfl_i = -1;
  int tri_fan_center_cfl_j = -1;
  double tri_fan_center_cfl_h = 0.0;
  double tri_fan_center_cfl_h_2ap = 0.0;
  double tri_fan_center_cfl_min_altitude = 0.0;
  double tri_fan_center_cfl_c_eff = 0.0;
  double tri_fan_center_cfl_q_over_p = 0.0;
  double tri_fan_center_cfl_r4[4] = {};
  double tri_fan_center_cfl_z4[4] = {};
  int tri_fan_center_cfl_block_id = -1;
  int tri_fan_center_cfl_nverts = 0;
  double tri_fan_center_cfl_r8[8] = {};
  double tri_fan_center_cfl_z8[8] = {};
  double tri_fan_center_cfl_dt_global_over_dt_center =
      std::numeric_limits<double>::infinity();
  int volume_rate_argmin_cell = -1;
  double volume_rate_argmin_frac_rate = 0.0;
  int cond_argmin_cell = -1;
  double cond_argmin_deff = 0.0;
  double cond_argmin_dl = 0.0;
  int rad_argmin_cell = -1;
  std::string phase = "primary";
  bool ale_attempted = false;
  bool ale_applied = false;
  bool ale_rezone_triggered = false;
  bool ale_rezone_converged = false;
  bool ale_quality_rollback = false;
  double ale_quality_min = 0.0;
  int ale_rezone_iterations = 0;
  double ale_rezone_residual = 0.0;
  double post_ale_corner_trial_scale = 1.0;
  int post_ale_first_failing_cell = -1;
  int post_ale_first_failing_corner = -1;
  bool retry_active_repair_enabled = false;
  double retry_active_repair_q_balance = 1.0;
  int retry_active_repair_first_failing_cell = -1;
  bool retry_active_repair_invoked = false;
  double retry_active_repair_post_ale_q_balance = 1.0;
  bool retry_active_repair_post_ale_still_bad = false;
  int prev_step_kershaw_mmatrix_fix_count = 0;
  std::string retry_phase = "";
  int retry_attempts = 0;
  int retry_first_failing_cell = -1;
  int retry_first_failing_corner = -1;
  std::string retry_reason = "";
  double retry_dt_before = 0.0;
  double retry_dt_after = 0.0;
  double retry_min_metric = 0.0;
  // Axis-failure telemetry. Populated only when
  // TENRYU_STAGE24_TELEMETRY=1; otherwise left at defaults.
  int repair_generation = 0;
  int rung_reached = 0;
  bool axis_variational_projection_eligible = false;
  std::string axis_variational_projection_blocked_reason;
  std::string axis_variational_projection_engaged_via;
  int floor_stall_count = 0;
  bool retry_state_sensitive = false;
  bool retry_dt_sensitive = false;
  double retry_dt_star_est = 0.0;
  double retry_margin_rel = 0.0;
};

// P1 ladder re-anchor state (fix47/fix49): owned by the driver loop,
// applied inside compute_dt_lineage so the lineage jsonl records the
// post-re-anchor truth. Null pointer = no re-anchor (retry paths).
struct DtReanchorState {
  bool pending = false;
  double pre_spike_dt_chosen = -1.0;
};

DtLineage compute_dt_lineage(const tenryu::core::State& state,
                             const tenryu::core::Config& cfg,
                             double t_end,
                             const char* phase = "primary",
                             const tenryu::hydro::HydroEOSContext* eos_ctx = nullptr,
                             bool defer_dt_floor_abort = false,
                             DtReanchorState* reanchor = nullptr);

inline bool retry_active_mesh_repair_should_force(
    const bool active_mesh_repair_enabled,
    const int retry_attempts,
    const bool corner_balance_admissible) {
  return active_mesh_repair_enabled && retry_attempts > 0 &&
         !corner_balance_admissible;
}

struct StagePerStageTracking {
  int consecutive_count = 0;
  double sigma_band_min = 1.0;
  double sigma_band_max = 0.0;
};

struct GeometricRetryStagnationState {
  std::string last_reason;
  int last_cell = -1;
  int last_corner = -1;
  std::map<std::string, StagePerStageTracking> per_stage;
  double dt_at_first_failure = 0.0;
  int dump_count = 0;
};

struct RetryReferenceBarrierSignature {
  int failing_i = -1;
  int failing_j = -1;
  HydroFailureStage stage = HydroFailureStage::Default;
  std::uint64_t reason_hash = 0;
};

double compute_retry_dt(const HydroStepResult& result,
                        const tenryu::core::Config& cfg,
                        double failed_dt);

bool is_mesh_quality_failure(const HydroStepResult& result);

bool classify_failure_axis_band(const HydroStepResult& result,
                                const tenryu::core::State& state,
                                const tenryu::core::Config& cfg);

bool classify_failure_active_shell_edge_cross(
    const HydroStepResult& result,
    const tenryu::core::State& state);

RetryReferenceBarrierSignature make_retry_signature(
    const HydroStepResult& result);

bool same_retry_signature(const RetryReferenceBarrierSignature& a,
                          const RetryReferenceBarrierSignature& b,
                          int cell_window);

double compute_reference_barrier_retry_dt(const HydroStepResult& result,
                                          const tenryu::core::Config& cfg,
                                          double failed_dt);

bool observe_reference_barrier_lambda_result(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const tenryu::hydro::ReferenceBarrierAleResult& result);

bool observe_reference_barrier_dt_collapse(tenryu::core::State& state,
                                           const tenryu::core::Config& cfg,
                                           double retry_dt_before,
                                           double retry_dt_after);

void observe_geometric_retry_failure(GeometricRetryStagnationState& state,
                                     const HydroStepResult& failure,
                                     double current_dt);

bool detect_geometric_retry_stagnation(
    const GeometricRetryStagnationState& state,
    const tenryu::core::Config& cfg,
    const HydroStepResult& failure,
    double current_dt);

std::string format_geometric_retry_stagnation_log(
    const GeometricRetryStagnationState& state,
    const HydroStepResult& failure,
    int step,
    double t,
    double current_dt);

tenryu::hydro::ale::RepairPlan choose_repair_plan(
    const HydroStepResult& result,
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    double failed_dt,
    int retry_attempt);

}  // namespace tenryu::coupling
