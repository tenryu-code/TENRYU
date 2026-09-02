#include "coupling/driver.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <atomic>
#include <vector>

#include "core/constants.hpp"
#include "core/config_validate.hpp"
#include "core/device_pack.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/mesh_transaction.hpp"
#include "core/namelist/errors.hpp"
#include "core/state_per_material_init.hpp"
#include "coupling/dispatcher_decision.hpp"
#include "coupling/dt_controller_device.cuh"
#include "coupling/dt_floor_stall_predicate.hpp"
#include "coupling/driver_fld_energy.hpp"
#include "coupling/driver_reclose.hpp"
#include "coupling/driver_safety_audit.hpp"
#include "coupling/persistent_loop.hpp"
#include "coupling/reale_mode.hpp"
#include "coupling/source_terms.hpp"
#include "diagnostics/diagnostics.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/escape_valve_history.hpp"
#include "diagnostics/history_shape_diag_device.cuh"
#include "diagnostics/history_writer.hpp"
#include "diagnostics/mesh_deform_attribution.hpp"
#include "diagnostics/mesh_degeneracy_forensics.hpp"
#include "diagnostics/operator_energy_residuals.hpp"
#include "diagnostics/positivity_history.hpp"
#include "diagnostics/radial_fourier_audit.hpp"
#include "diagnostics/shock_approach.hpp"
#include "hydro/ale_1d_driver.cuh"
#include "hydro/ale_1d_types.cuh"
#include "hydro/ale_gcl.hpp"
#include "hydro/ale_axis_band_controller.cuh"
#include "hydro/ale_driver.cuh"
#include "hydro/ale_mode.hpp"
#include "hydro/ale_remap_2d_rz.hpp"
#include "hydro/ale_state_monitor.hpp"
#include "hydro/anti_hourglass.cuh"
#include "hydro/axis_edge_collapse.hpp"
#include "hydro/axis_projection.hpp"
#include "hydro/band_ale.hpp"
#include "hydro/boundary.hpp"
#include "hydro/boundary_2d.hpp"
#include "hydro/boundary_pressure_force.cuh"
#include "hydro/button_morph_indexing.hpp"
#include "hydro/cavity_boundary_graph.hpp"
#include "hydro/cap_energy_audit.hpp"
#include "hydro/conduction.cuh"
#include "hydro/braginskii_viscosity.cuh"
#include "coupling/rad_gamma_coupling.cuh"
#include "hydro/compatible_av_csw.cuh"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/cfl.hpp"
#include "hydro/compatible_force_work_2d.cuh"
#include "hydro/flank_tangential_strip.hpp"
#include "hydro/flank_tangential_strip_detail.hpp"
#include "hydro/hydro_1d.hpp"
#include "hydro/hydro_2d.hpp"
#include "hydro/wake_angular_heat_flux.hpp"
#include "hydro/eos_context.hpp"
#include "hydro/euler_window_blend.hpp"
#include "hydro/axis_cell_ledger.hpp"
#include "hydro/corner_collapse_ledger.hpp"
#include "hydro/core_clearance_controller.hpp"
#include "hydro/shadow_homothety_diag.hpp"
#include "hydro/entropy_stage_ledger.hpp"
#include "hydro/stripe_gate.hpp"
#include "hydro/mesh_regime.cuh"
#include "hydro/pole_angular_derefine.cuh"
#include "hydro/reference_barrier_ale.hpp"
#include "hydro/refinement_estimator.hpp"
#include "hydro/refinement_tracker.hpp"
#include "hydro/evacuated_cell_shadow.hpp"
#include "hydro/ring7_seam_quotient_remap.cuh"
#include "hydro/row_merge.cuh"
#include "hydro/runtime_ale_targets.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "hydro/state_supply_bc.hpp"
#include "io/output_manager.hpp"
#include "laser/laser.cuh"
#include "materials/eos_table.hpp"
#include "materials/ionmix_reader.hpp"
#include "materials/tmat_reader.hpp"
#include "materials/zbar_tf.hpp"
#include "materials/zmoment_state.cuh"
#include "mesh/mesh.hpp"
#include "mesh/z_reflection.hpp"
#include "parallel/partition.hpp"
#include "radiation/fleck.cuh"
#include "radiation/fld_2d_rz_gpu.cuh"
#include "radiation/group_structure.hpp"
#include "radiation/imc.hpp"
#include "radiation/planck_table.cuh"
#include "parallel/reduction.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"
#include "parallel/particle_migration.hpp"
#include "radiation/particle_reid.hpp"
#include "verification/escape_valve_audit.hpp"
#include "verification/positivity_tracker.hpp"
#include "burn/burn_constants.hpp"
#include "burn/corman_diffusion.cuh"
#include "burn/mc_transport.cuh"
#include "burn/burn_stage.hpp"
#include "burn/neutron_moments.hpp"
#include "burn/burn_stage_2d.hpp"
#include "burn/corman_diffusion_2d.cuh"
#include "burn/partition.hpp"
#include "burn/screening.hpp"

namespace tenryu::coupling {

template <typename T>
void ensure_device_buffer_capacity(core::DeviceBuffer<T>& buffer,
                                   const std::size_t n,
                                   const char* label) {
  if (buffer.size() >= n) {
    return;
  }
  buffer.reset(n);
  TENRYU_ASSERT(buffer.size() >= n,
                std::string(label) + " device buffer resize failed");
}

template <typename T>
void copy_host_to_device_prefix(core::DeviceBuffer<T>& buffer,
                                const std::vector<T>& host,
                                const char* label) {
  ensure_device_buffer_capacity(buffer, host.size(), label);
  if (host.empty()) {
    return;
  }
  const cudaError_t err = cudaMemcpy(buffer.data(), host.data(),
                                     host.size() * sizeof(T),
                                     cudaMemcpyHostToDevice);
  TENRYU_ASSERT(err == cudaSuccess,
                std::string(label) + " cudaMemcpy H2D failed");
}

template <typename T>
void zero_device_buffer_prefix(core::DeviceBuffer<T>& buffer,
                               const std::size_t n,
                               const char* label) {
  ensure_device_buffer_capacity(buffer, n, label);
  if (n == 0U) {
    return;
  }
  const cudaError_t err = cudaMemset(buffer.data(), 0, n * sizeof(T));
  TENRYU_ASSERT(err == cudaSuccess,
                std::string(label) + " cudaMemset failed");
}

bool any_nonzero_span(const std::vector<double>& values,
                      const std::size_t begin,
                      const std::size_t count) {
  const std::size_t end = begin + count;
  for (std::size_t i = begin; i < end; ++i) {
    if (values[i] != 0.0) {
      return true;
    }
  }
  return false;
}

double charged_species_Z(const int species) {
  return (species == burn::kHe3 || species == burn::kHe4) ? 2.0 : 1.0;
}

std::size_t burn_mc_pool_capacity_size(const int n_cells,
                                       const int particles_per_cell) {
  TENRYU_ASSERT(n_cells >= 0, "burn MC n_cells must be nonnegative");
  TENRYU_ASSERT(particles_per_cell >= 1,
                "burn MC particles_per_cell must be positive");
  const std::size_t capacity =
      static_cast<std::size_t>(n_cells) * 6U *
      static_cast<std::size_t>(particles_per_cell) * 64U;
  TENRYU_ASSERT(capacity <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "burn MC particle pool capacity exceeds INT_MAX");
  return capacity;
}

template <typename T>
void ensure_burn_mc_pool_field(core::DeviceBuffer<T>& buffer,
                               const std::size_t capacity,
                               const std::size_t live,
                               const char* label) {
  if (buffer.size() >= capacity) {
    return;
  }
  TENRYU_ASSERT(buffer.size() >= live,
                std::string(label) + " restored live count exceeds buffer size");
  std::vector<T> live_host;
  if (live > 0U) {
    buffer.copy_to_host(live_host);
    live_host.resize(live);
  }
  // DeviceBuffer::reset does not preserve data; resetting a live MC pool would
  // lose inflight particles. The formula sizes fresh pools once at first use;
  // after checkpoint restore, compact live entries are copied into the grown
  // pool before transport resumes.
  buffer.reset(capacity);
  if (!live_host.empty()) {
    copy_host_to_device_prefix(buffer, live_host, label);
  }
}
namespace {
constexpr int kPathSourceActiveFineChild = 1;
constexpr int kPathMetricEdgeCross = 1;
bool ring7_quotient_effective_enabled(const core::Config& cfg);
std::string format_sci(double value);
}  // namespace

bool is_multiblock_path_admissibility_failure(const HydroStepResult& result) {
  const std::string reason =
      result.reason != nullptr ? std::string(result.reason) : std::string();
  return reason == "multiblock_path_admissibility";
}

bool is_macro_repair_required_failure(const HydroStepResult& result) {
  const std::string reason =
      result.reason != nullptr ? std::string(result.reason) : std::string();
  return reason == "macro_repair_required";
}

bool is_pole_axis_radial_order_inversion_failure(
    const HydroStepResult& result) {
  const std::string reason =
      result.reason != nullptr ? std::string(result.reason) : std::string();
  return reason == "pole_axis_radial_order_inversion";
}

const char* path_guard_stage_name(const HydroFailureStage stage) {
  switch (stage) {
    case HydroFailureStage::PredictorCommit:
      return "predictor";
    case HydroFailureStage::PostCorrector:
      return "corrector";
    case HydroFailureStage::Default:
    case HydroFailureStage::StepStart:
    case HydroFailureStage::PostPredictor:
    case HydroFailureStage::RetryRestoreValidate:
      return "unknown";
  }
  return "unknown";
}

double compute_retry_dt(const HydroStepResult& result,
                        const tenryu::core::Config& cfg,
                        const double failed_dt) {
  const std::string reason =
      result.reason != nullptr ? std::string(result.reason) : std::string();
  if (is_macro_repair_required_failure(result) ||
      is_pole_axis_radial_order_inversion_failure(result)) {
    return failed_dt;
  }
  const bool is_geometric_failure =
      reason == "mesh_quality_corner_j" ||
      reason == "mesh_quality_gauss_j" ||
      reason == "mesh_quality_rz_volume" ||
      reason == "mesh_quality_planar_area" ||
      reason == "mesh_quality_axis_margin" ||
      reason == "in_hydro_corner_j" ||
      reason == "in_hydro_gauss_j" ||
      reason == "in_hydro_rz_volume" ||
      reason == "corner_j" ||
      reason == "trial_volume_cfl" ||
      reason == "multiblock_path_admissibility";
  if (is_geometric_failure && result.suggested_dt > 0.0 &&
      std::isfinite(result.suggested_dt) && result.suggested_dt < failed_dt) {
    if (result.suggested_dt < cfg.numerics.dt.min_s) {
      core::log_warning(
          "[driver-retry] required dt " + format_sci(result.suggested_dt) +
          " is below Numerics.dt.min_s " +
          format_sci(cfg.numerics.dt.min_s) + " for reason=" + reason +
          " — geometric defect not curable by dt descent (structural mesh "
          "action required); attempting once at the floor");
    }
    return std::max(cfg.numerics.dt.min_s, result.suggested_dt);
  }

  const double half_dt = 0.5 * failed_dt;
  if (cfg.numerics.hydro.driver_retry_use_suggested_dt_enabled &&
      result.suggested_dt > 0.0 && result.trial_scale > 0.0) {
    return std::min(half_dt, result.suggested_dt);
  }
  return half_dt;
}

int driver_full_step_retry_limit_for_failure(
    const HydroStepResult& result,
    const tenryu::core::Config& cfg) {
  if (is_multiblock_path_admissibility_failure(result)) {
    return cfg.numerics.ale.max_dt_rejections;
  }
  return cfg.numerics.hydro.driver_full_step_retry_max_attempts;
}

bool is_mesh_quality_failure(const HydroStepResult& result) {
  const std::string reason =
      result.reason != nullptr ? std::string(result.reason) : std::string();
  return reason == "mesh_quality_corner_j" ||
         reason == "mesh_quality_gauss_j" ||
         reason == "mesh_quality_volume" ||
         reason == "mesh_quality_rz_volume" ||
         reason == "mesh_quality_planar_area" ||
         reason == "mesh_quality_axis_margin" ||
         reason == "in_hydro_corner_j" ||
         reason == "in_hydro_gauss_j" ||
         reason == "in_hydro_rz_volume";
}

bool is_ring7_seam_rz_volume_retry_candidate(
    const HydroStepResult& result,
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  const std::string reason =
      result.reason != nullptr ? std::string(result.reason) : std::string();
  return ring7_quotient_effective_enabled(cfg) &&
         reason == "mesh_quality_rz_volume" &&
         tenryu::hydro::ring7_seam_retry_candidate_cell(
             state, cfg, result.first_failing_cell);
}

bool is_ring7_seam_path_admissibility_retry_candidate(
    const HydroStepResult& result,
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  return ring7_quotient_effective_enabled(cfg) &&
         is_multiblock_path_admissibility_failure(result) &&
         tenryu::hydro::ring7_seam_retry_candidate_cell(
             state, cfg, result.first_failing_cell);
}

bool classify_failure_axis_band(const HydroStepResult& result,
                                const tenryu::core::State& state,
                                const tenryu::core::Config& cfg) {
  int i = result.first_failing_i;
  int j = result.first_failing_j;
  if ((i < 0 || j < 0) && result.first_failing_cell >= 0 &&
      state.mesh.topo.nz > 0) {
    i = result.first_failing_cell / state.mesh.topo.nz;
    j = result.first_failing_cell % state.mesh.topo.nz;
  }
  if (i >= 0 && i <= cfg.numerics.ale.driver_retry_reference_barrier_K_axis) {
    return true;
  }
  if (i < 0 || j < 0 || state.mesh.cell_centroid_r.empty() ||
      state.mesh.cell_centroid_z.empty()) {
    return false;
  }
  const int cell = state.mesh.topo.cell_index(i, j);
  if (cell < 0 ||
      cell >= static_cast<int>(state.mesh.cell_centroid_r.size()) ||
      cell >= static_cast<int>(state.mesh.cell_centroid_z.size())) {
    return false;
  }
  const double r = state.mesh.cell_centroid_r[static_cast<std::size_t>(cell)];
  const double z = state.mesh.cell_centroid_z[static_cast<std::size_t>(cell)];
  const double radius = std::sqrt(r * r + z * z);
  if (radius > 0.0 &&
      r / radius < cfg.numerics.ale.driver_retry_reference_barrier_eta_axis) {
    return true;
  }
  if (i <= 8 && state.mesh.topo.nz > 1) {
    double dz = 0.0;
    if (j + 1 < state.mesh.topo.nz) {
      const int next = state.mesh.topo.cell_index(i, j + 1);
      dz = std::abs(state.mesh.cell_centroid_z[static_cast<std::size_t>(next)] - z);
    } else if (j > 0) {
      const int prev = state.mesh.topo.cell_index(i, j - 1);
      dz = std::abs(z - state.mesh.cell_centroid_z[static_cast<std::size_t>(prev)]);
    }
    if (dz > 0.0 && std::abs(z) < 2.0 * dz) {
      return true;
    }
  }
  return false;
}

bool classify_failure_active_shell_edge_cross(
    const HydroStepResult& result,
    const tenryu::core::State& state) {
  if (!is_multiblock_path_admissibility_failure(result) ||
      result.path_source_kind != kPathSourceActiveFineChild ||
      result.path_metric_kind != kPathMetricEdgeCross) {
    return false;
  }
  if (result.first_failing_cell >= 0 && !state.hydro_active.empty()) {
    const int c = result.first_failing_cell;
    if (c >= static_cast<int>(state.hydro_active.size()) ||
        state.hydro_active[static_cast<std::size_t>(c)] == 0) {
      return false;
    }
  }
  if (!state.mesh.topo.multiblock.has_value() || result.path_block_id < 0) {
    return false;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  if (result.path_block_id >= static_cast<int>(mb.blocks.size())) {
    return false;
  }
  return mb.blocks[static_cast<std::size_t>(result.path_block_id)].role ==
         tenryu::mesh::BlockRole::POLAR_SHELL;
}

RetryReferenceBarrierSignature make_retry_signature(
    const HydroStepResult& result) {
  const std::string reason =
      result.reason != nullptr ? std::string(result.reason) : std::string();
  RetryReferenceBarrierSignature sig;
  sig.failing_i = result.first_failing_i;
  sig.failing_j = result.first_failing_j;
  sig.stage = result.stage;
  sig.reason_hash = static_cast<std::uint64_t>(std::hash<std::string>{}(reason));
  return sig;
}

bool same_retry_signature(const RetryReferenceBarrierSignature& a,
                          const RetryReferenceBarrierSignature& b,
                          const int cell_window) {
  return std::abs(a.failing_i - b.failing_i) <= cell_window &&
         std::abs(a.failing_j - b.failing_j) <= cell_window &&
         a.stage == b.stage &&
         a.reason_hash == b.reason_hash;
}

double compute_reference_barrier_retry_dt(const HydroStepResult& result,
                                          const tenryu::core::Config& cfg,
                                          const double failed_dt) {
  double sigma_safe = result.trial_scale;
  if (!(sigma_safe >= 0.0 && std::isfinite(sigma_safe))) {
    sigma_safe = result.min_metric;
  }
  if (!(sigma_safe >= 0.0 && std::isfinite(sigma_safe))) {
    sigma_safe = 1.0;
  }
  sigma_safe = std::min(sigma_safe, 1.0);
  const double q_retry = cfg.numerics.ale.driver_retry_reference_barrier_q_retry;
  if (sigma_safe > 0.0 && std::isfinite(sigma_safe)) {
    return std::min(
        failed_dt * cfg.numerics.ale.driver_retry_reference_barrier_chi * sigma_safe,
        failed_dt * q_retry);
  }
  return failed_dt * q_retry;
}

bool observe_reference_barrier_lambda_result(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const tenryu::hydro::ReferenceBarrierAleResult& result) {
  if (!result.succeeded ||
      result.lambda_accepted <
          cfg.numerics.ale
              .driver_retry_reference_barrier_lambda_collapse_threshold) {
    ++state.driver_retry_reference_barrier_lambda_collapse_events;
    return state.driver_retry_reference_barrier_lambda_collapse_events >=
           static_cast<std::uint64_t>(
               cfg.numerics.ale
                   .driver_retry_reference_barrier_lambda_collapse_count);
  }
  ++state.driver_retry_reference_barrier_successes;
  state.driver_retry_reference_barrier_lambda_collapse_events = 0;
  return false;
}

bool observe_reference_barrier_dt_collapse(tenryu::core::State& state,
                                           const tenryu::core::Config& cfg,
                                           const double retry_dt_before,
                                           const double retry_dt_after) {
  if (retry_dt_after <
      cfg.numerics.ale.driver_retry_reference_barrier_dt_collapse_rel *
          retry_dt_before) {
    ++state.driver_retry_reference_barrier_dt_collapse_aborts;
    return true;
  }
  return false;
}

bool is_strategy_first_state_sensitive_hint(const RetryActionHint action,
                                            const int retry_attempt) {
  switch (action) {
    case RetryActionHint::ForceAxisSpinePlusLocalAle:
    case RetryActionHint::ForceBoundaryPatchRepair:
    case RetryActionHint::ForceCdLocalRezone:
    case RetryActionHint::ForceInteriorMultiNodeRepair:
    case RetryActionHint::RepairOnly:
      return true;
    case RetryActionHint::ForceFullWinslow:
      return retry_attempt == 0;
    case RetryActionHint::ReduceDtOnly:
    case RetryActionHint::None:
      return false;
  }
  return false;
}

tenryu::hydro::ale::RepairPlan choose_repair_plan(
    const HydroStepResult& result,
    const tenryu::core::State&,
    const tenryu::core::Config& cfg,
    const double failed_dt,
    const int retry_attempt) {
  tenryu::hydro::ale::RepairPlan plan;
  switch (result.retry_action) {
    case RetryActionHint::ForceAxisSpinePlusLocalAle:
      plan.ale_mode = tenryu::hydro::ale::AleMode::AxisSpinePlusLocal;
      break;
    case RetryActionHint::ForceBoundaryPatchRepair:
      plan.ale_mode = tenryu::hydro::ale::AleMode::BoundaryPatchProjection;
      break;
    case RetryActionHint::ForceCdLocalRezone:
      plan.ale_mode = tenryu::hydro::ale::AleMode::CdLocalWinslow;
      break;
    case RetryActionHint::ForceInteriorMultiNodeRepair:
      plan.ale_mode = cfg.numerics.ale.multi_node_interior_repair_enabled
                          ? tenryu::hydro::ale::AleMode::InteriorMultiNodeProjection
                          : tenryu::hydro::ale::AleMode::FullWinslow;
      break;
    case RetryActionHint::ForceFullWinslow:
    case RetryActionHint::RepairOnly:
      plan.ale_mode = tenryu::hydro::ale::AleMode::FullWinslow;
      break;
    case RetryActionHint::ReduceDtOnly:
    case RetryActionHint::None:
      plan.ale_mode = tenryu::hydro::ale::AleMode::ScheduledDefault;
      break;
  }
  plan.focus_cell = result.first_failing_cell;
  plan.retry_attempt = retry_attempt;
  const bool same_dt_strategy_retry =
      cfg.numerics.hydro.strategy_first_retry_enabled &&
      cfg.numerics.hydro.strategy_first_max_same_dt_attempts > 0 &&
      retry_attempt >= 0 &&
      retry_attempt <
          cfg.numerics.hydro.strategy_first_max_same_dt_attempts &&
      is_strategy_first_state_sensitive_hint(result.retry_action, retry_attempt);
  plan.dt_retry =
      same_dt_strategy_retry ? failed_dt : compute_retry_dt(result, cfg, failed_dt);
  plan.is_same_dt_strategy_retry = same_dt_strategy_retry;
  return plan;
}

namespace {

constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = tenryu::core::constants::proton_mass;
constexpr double kPi = 3.141592653589793238462643383279502884;

void persistent_loop_nlte_cuda_check(const cudaError_t err,
                                     const char* message) {
  TENRYU_ASSERT(err == cudaSuccess,
                std::string(message) + ": " + cudaGetErrorString(err));
}

struct PersistentLoopDeviceOpacityTable {
  double* d_log_temps = nullptr;
  double* d_log_numdens = nullptr;
  double* d_kappa_PA = nullptr;
  double* d_kappa_PE = nullptr;
  double* d_kappa_R = nullptr;
  int ntemp = 0;
  int ndens = 0;
  int ngroups = 0;
  double log_T_min = 0.0;
  double log_T_max = 0.0;
  double log_ni_min = 0.0;
  double log_ni_max = 0.0;
  double T_min = 0.0;
  double T_max = 0.0;

  PersistentLoopDeviceOpacityTable() = default;
  PersistentLoopDeviceOpacityTable(const PersistentLoopDeviceOpacityTable&) = delete;
  PersistentLoopDeviceOpacityTable& operator=(
      const PersistentLoopDeviceOpacityTable&) = delete;

  ~PersistentLoopDeviceOpacityTable() {
    release();
  }

  void upload(const materials::IonmixOpacityData& host) {
    release();
    TENRYU_ASSERT(host.ntemp > 0 && host.ndens > 0 && host.ngroups > 0,
                  "persistent_loop NLTE opacity table requires non-empty axes");
    const std::size_t n_table =
        static_cast<std::size_t>(host.ngroups) *
        static_cast<std::size_t>(host.ndens) *
        static_cast<std::size_t>(host.ntemp);
    TENRYU_ASSERT(host.kappa_PA.size() == n_table &&
                      host.kappa_PE.size() == n_table &&
                      host.kappa_R.size() == n_table,
                  "persistent_loop NLTE opacity table size mismatch");

    const auto upload_vec = [](double** dst,
                               const std::vector<double>& src,
                               const char* label) {
      persistent_loop_nlte_cuda_check(
          cudaMalloc(reinterpret_cast<void**>(dst),
                     sizeof(double) * src.size()),
          label);
      persistent_loop_nlte_cuda_check(
          cudaMemcpy(*dst, src.data(), sizeof(double) * src.size(),
                     cudaMemcpyHostToDevice),
          label);
    };
    upload_vec(&d_log_temps, host.log_temps,
               "persistent_loop NLTE upload log_temps failed");
    upload_vec(&d_log_numdens, host.log_numdens,
               "persistent_loop NLTE upload log_numdens failed");
    upload_vec(&d_kappa_PA, host.kappa_PA,
               "persistent_loop NLTE upload kappa_PA failed");
    upload_vec(&d_kappa_PE, host.kappa_PE,
               "persistent_loop NLTE upload kappa_PE failed");
    upload_vec(&d_kappa_R, host.kappa_R,
               "persistent_loop NLTE upload kappa_R failed");
    ntemp = host.ntemp;
    ndens = host.ndens;
    ngroups = host.ngroups;
    log_T_min = host.log_temps.front();
    log_T_max = host.log_temps.back();
    log_ni_min = host.log_numdens.front();
    log_ni_max = host.log_numdens.back();
    T_min = host.temps_eV.front();
    T_max = host.temps_eV.back();
  }

  void release() {
    if (d_kappa_R != nullptr) {
      persistent_loop_nlte_cuda_check(cudaFree(d_kappa_R),
                                      "persistent_loop NLTE free kappa_R failed");
      d_kappa_R = nullptr;
    }
    if (d_kappa_PE != nullptr) {
      persistent_loop_nlte_cuda_check(cudaFree(d_kappa_PE),
                                      "persistent_loop NLTE free kappa_PE failed");
      d_kappa_PE = nullptr;
    }
    if (d_kappa_PA != nullptr) {
      persistent_loop_nlte_cuda_check(cudaFree(d_kappa_PA),
                                      "persistent_loop NLTE free kappa_PA failed");
      d_kappa_PA = nullptr;
    }
    if (d_log_numdens != nullptr) {
      persistent_loop_nlte_cuda_check(
          cudaFree(d_log_numdens),
          "persistent_loop NLTE free log_numdens failed");
      d_log_numdens = nullptr;
    }
    if (d_log_temps != nullptr) {
      persistent_loop_nlte_cuda_check(cudaFree(d_log_temps),
                                      "persistent_loop NLTE free log_temps failed");
      d_log_temps = nullptr;
    }
    ntemp = 0;
    ndens = 0;
    ngroups = 0;
    log_T_min = 0.0;
    log_T_max = 0.0;
    log_ni_min = 0.0;
    log_ni_max = 0.0;
    T_min = 0.0;
    T_max = 0.0;
  }

  materials::IonmixOpacityDeviceView view() const {
    materials::IonmixOpacityDeviceView out{};
    out.log_temps = d_log_temps;
    out.log_numdens = d_log_numdens;
    out.kappa_PA = d_kappa_PA;
    out.kappa_PE = d_kappa_PE;
    out.kappa_R = d_kappa_R;
    out.ntemp = ntemp;
    out.ndens = ndens;
    out.ngroups = ngroups;
    out.log_T_min = log_T_min;
    out.log_T_max = log_T_max;
    out.log_ni_min = log_ni_min;
    out.log_ni_max = log_ni_max;
    out.T_min = T_min;
    out.T_max = T_max;
    return out;
  }
};

struct PersistentLoopNlteOpacityCache {
  std::unique_ptr<materials::IonmixOpacityData> host;
  PersistentLoopDeviceOpacityTable device;
  std::string key;
};

PersistentLoopNlteOpacityCache& persistent_loop_nlte_opacity_cache() {
  static PersistentLoopNlteOpacityCache cache;
  return cache;
}

materials::IonmixOpacityDeviceView persistent_loop_nlte_opacity_view(
    const core::Config& cfg) {
  if (!cfg.radiation.enabled || cfg.materials.materials.empty()) {
    return materials::IonmixOpacityDeviceView{};
  }
  const auto& mat = cfg.materials.materials.front();
  const bool use_nlte =
      mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";
  if (!use_nlte) {
    return materials::IonmixOpacityDeviceView{};
  }

  const int n_groups = std::max(cfg.radiation.groups, 1);
  std::ostringstream key;
  key << mat.opacity_model << '\n'
      << mat.opacity_file << '\n'
      << n_groups << '\n'
      << (cfg.radiation.group_repack_hard_xray ? 1 : 0) << '\n';
  for (const double bound : cfg.radiation.group_bounds_eV) {
    key << std::setprecision(17) << bound << ',';
  }

  auto& cache = persistent_loop_nlte_opacity_cache();
  const std::string key_str = key.str();
  if (cache.host != nullptr && cache.key == key_str &&
      cache.device.ngroups == n_groups) {
    return cache.device.view();
  }

  if (mat.opacity_model == "tmat") {
    const materials::TmatFile tmat = materials::load_tmat(mat.opacity_file);
    TENRYU_ASSERT(tmat.opacity.has_value(),
                  "persistent_loop tmat opacity.model requires /opacity payload");
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::tmat_to_ionmix_opacity(*tmat.opacity));
  } else {
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::load_ionmix_opacity(mat.opacity_file));
  }
  if (cfg.radiation.group_repack_hard_xray) {
    std::vector<double> target_bounds = cfg.radiation.group_bounds_eV;
    if (target_bounds.empty()) {
      const std::vector<double> range =
          radiation::resolve_compute_T_range_eV(cfg, false);
      target_bounds =
          radiation::repack_group_bounds_for_hard_xray(n_groups, range);
    }
    *cache.host =
        radiation::resample_opacity_groups_to_bounds(*cache.host, target_bounds);
  }
  TENRYU_ASSERT(cache.host->ngroups == n_groups,
                "persistent_loop NLTE opacity table group count must match "
                "Radiation.groups");
  cache.device.upload(*cache.host);
  cache.key = key_str;
  return cache.device.view();
}

// NOH-DET (VERIFICATION 3.2r): order-independent XOR-fold of a device double
// array's bit patterns, for paired-run first-divergence localization.
// Env-gated (TENRYU_STATE_HASH=1), read-only, default-inert.
static unsigned long long state_hash_xor_fold(const double* device_ptr,
                                              const std::size_t n) {
  if (device_ptr == nullptr || n == 0) {
    return 0ULL;
  }
  std::vector<double> host(n);
  cudaMemcpy(host.data(), device_ptr, n * sizeof(double),
             cudaMemcpyDeviceToHost);
  unsigned long long acc = 0ULL;
  for (std::size_t i = 0; i < n; ++i) {
    unsigned long long bits;
    std::memcpy(&bits, &host[i], sizeof(bits));
    // mix the index so permuted values do not cancel identically
    acc ^= bits + 0x9e3779b97f4a7c15ULL * static_cast<unsigned long long>(i + 1);
  }
  return acc;
}

std::optional<std::filesystem::path> derive_history_path_from_checkpoint_prefix(
    const std::string& checkpoint_prefix) {
  if (checkpoint_prefix.empty()) {
    return std::nullopt;
  }
  const std::filesystem::path checkpoint_path(checkpoint_prefix);
  std::string stem = checkpoint_path.stem().string();
  const std::size_t rank_pos = stem.rfind("_r");
  if (rank_pos != std::string::npos) {
    bool rank_suffix = rank_pos + 2 < stem.size();
    for (std::size_t i = rank_pos + 2; i < stem.size(); ++i) {
      rank_suffix = rank_suffix && std::isdigit(static_cast<unsigned char>(stem[i]));
    }
    if (rank_suffix) {
      stem = stem.substr(0, rank_pos);
    }
  }
  const std::size_t ckpt_pos = stem.rfind("_ckpt_");
  if (ckpt_pos == std::string::npos || ckpt_pos == 0) {
    return std::nullopt;
  }
  const std::string case_name = stem.substr(0, ckpt_pos);
  const std::filesystem::path parent = checkpoint_path.parent_path();
  const std::filesystem::path output_root =
      (parent.filename() == "checkpoints") ? parent.parent_path() : parent;
  return output_root / "results" / (case_name + "_history.h5");
}

inline bool use_exact_ideal_gas_hydro_backend(
    const core::Config::MaterialsConfig::MatDef& mat) {
  return mat.hydro_eos_backend == "exact_ideal_gas";
}

inline bool use_mie_gruneisen_hydro_backend(
    const core::Config::MaterialsConfig::MatDef& mat) {
  return mat.hydro_eos_backend == "mie_gruneisen";
}

void resolve_temperature_model(core::Config& cfg, const core::State& state) {
  if (cfg.main.temperature_model == "1T") {
    cfg.main.two_temperature = false;
    return;
  }
  if (cfg.main.temperature_model == "2T") {
    cfg.main.two_temperature = true;
    return;
  }
  if (cfg.main.temperature_model == "auto") {
    TENRYU_ASSERT(state.Te.size() == state.Ti.size(),
                  "Main.temperature_model=\"auto\" requires Te/Ti size match");
    const std::size_t n_cells = state.Te.size();
    std::vector<double> Te(n_cells, 0.0);
    std::vector<double> Ti(n_cells, 0.0);
    if (n_cells > 0) {
      state.Te.copy_to_host(Te.data());
      state.Ti.copy_to_host(Ti.data());
    }

    bool use_two_temp = false;
    for (std::size_t i = 0; i < n_cells; ++i) {
      if (std::abs(Te[i] - Ti[i]) > 1.0e-14) {
        use_two_temp = true;
        break;
      }
    }

    cfg.main.two_temperature = use_two_temp;
    cfg.main.temperature_model = use_two_temp ? "2T" : "1T";
    core::log_warning(
        "Main.temperature_model=\"auto\" resolved to \"" + cfg.main.temperature_model +
        "\" based on initial state. Consider setting explicitly.");
    return;
  }

  TENRYU_ASSERT(false,
                "Main.temperature_model must be \"1T\", \"2T\", or \"auto\"");
}

void update_hydro_active(core::State& state, const core::Config&) {
  if (state.hydro_active.empty()) {
    return;
  }

  bool changed = false;
  if (state.hydro_t_start_eV <= 0.0) {
    changed = std::any_of(state.hydro_active.begin(), state.hydro_active.end(),
                          [](const std::int8_t active) { return active != 1; });
    std::fill(state.hydro_active.begin(), state.hydro_active.end(), static_cast<std::int8_t>(1));
  } else {
    TENRYU_ASSERT(state.hydro_active.size() == state.Te.size(),
                  "hydro_active/Te size mismatch");

    std::vector<double> host_Te(state.Te.size(), 0.0);
    state.Te.copy_to_host(host_Te.data());

    for (std::size_t i = 0; i < state.hydro_active.size(); ++i) {
      if (state.hydro_active[i] != 0) {
        continue;
      }
      if (host_Te[i] >= state.hydro_t_start_eV) {
        state.hydro_active[i] = 1;
        changed = true;
      }
    }
  }

  if (!state.cell_is_void.empty()) {
    for (std::size_t i = 0; i < state.hydro_active.size(); ++i) {
      if (state.cell_is_void[i] != 0U) {
        if (state.hydro_active[i] != 0) {
          changed = true;
        }
        state.hydro_active[i] = 0;
      }
    }
  }

  if (state.mesh.button_center && state.mesh.button_center->enabled) {
    for (std::size_t i = 0; i < state.hydro_active.size(); ++i) {
      const int c = static_cast<int>(i);
      if (state.mesh.is_button_cell(c)) {
        if (state.hydro_active[i] != 1) {
          changed = true;
        }
        state.hydro_active[i] = 1;
      } else if (state.mesh.is_dormant_cell(c)) {
        if (state.hydro_active[i] != 0) {
          changed = true;
        }
        state.hydro_active[i] = 0;
      }
    }
  }
  if (!state.merge_tombstone.empty()) {
    TENRYU_ASSERT(state.merge_tombstone.size() == state.hydro_active.size(),
                  "merge_tombstone/hydro_active size mismatch");
    for (std::size_t i = 0; i < state.hydro_active.size(); ++i) {
      if (state.merge_tombstone[i] != 0U) {
        if (state.hydro_active[i] != 0) {
          changed = true;
        }
        state.hydro_active[i] = 0;
      }
    }
  }
  if (changed) {
    state.note_hydro_active_host_write();
  }
}

bool all_active_cells_collapsed_to_one_temperature(const core::State& state,
                                                   const core::Config& cfg) {
  TENRYU_ASSERT(state.Te.size() == state.Ti.size(),
                "2T collapse check requires Te/Ti size match");
  const std::size_t n_cells = state.Te.size();
  if (n_cells == 0) {
    return false;
  }
  TENRYU_ASSERT(state.hydro_active.empty() || state.hydro_active.size() == n_cells,
                "2T collapse check requires hydro_active/Te size match");
  return all_active_cells_collapsed_device(state, cfg.numerics.floors.Te);
}

void update_zbar_thomas_fermi(core::State& state, const core::Config& cfg) {
  if (cfg.materials.zbar.model != "thomas_fermi") {
    return;
  }

  const std::size_t n_cells = state.rho.size();
  const std::size_t n_materials = cfg.materials.materials.size();
  if (n_cells == 0 || n_materials == 0) {
    return;
  }

  TENRYU_ASSERT(state.Te.size() == n_cells,
                "update_zbar_thomas_fermi Te/rho size mismatch");
  TENRYU_ASSERT(state.zbar.size() == n_cells,
                "update_zbar_thomas_fermi zbar/rho size mismatch");
  TENRYU_ASSERT(state.cell_is_void.size() == n_cells,
                "update_zbar_thomas_fermi cell_is_void/rho size mismatch");
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "update_zbar_thomas_fermi requires at least one non-void material");
  const std::size_t first_nonvoid_idx = static_cast<std::size_t>(first_nonvoid);

  std::vector<double> rho(n_cells, 0.0);
  std::vector<double> Te(n_cells, 0.0);
  std::vector<double> zbar(n_cells, 0.0);
  state.rho.copy_to_host(rho.data());
  state.Te.copy_to_host(Te.data());

  std::vector<double> volfrac;
  if (n_materials > 1) {
    const std::size_t expected =
        n_cells * static_cast<std::size_t>(n_materials);
    TENRYU_ASSERT(state.volFrac.size() == expected,
                  "update_zbar_thomas_fermi volFrac size mismatch");
    volfrac.resize(expected, 0.0);
    state.volFrac.copy_to_host(volfrac.data());
  }

  for (std::size_t c = 0; c < n_cells; ++c) {
    if (n_materials == 1) {
      const auto& mat = cfg.materials.materials[first_nonvoid_idx];
      zbar[c] = materials::compute_zbar_tf(rho[c], Te[c], mat.Z, mat.A);
      continue;
    }

    double weighted = 0.0;
    double frac_sum = 0.0;
    const std::size_t base = c * n_materials;
    for (std::size_t m = 0; m < n_materials; ++m) {
      const auto& mat = cfg.materials.materials[m];
      if (mat.is_void) {
        continue;
      }
      const double f = std::max(volfrac[base + m], 0.0);
      frac_sum += f;
      const double zbar_m = materials::compute_zbar_tf(rho[c], Te[c], mat.Z, mat.A);
      weighted += f * zbar_m;
    }
    if (frac_sum > 1.0e-30) {
      weighted /= frac_sum;
    }
    zbar[c] = weighted;
  }
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (state.cell_is_void[c] != 0U) {
      zbar[c] = 0.0;
    }
  }

  state.zbar.copy_from_host(zbar.data());
}

void update_zbar_tabular(core::State& state, const core::Config& cfg) {
  if (cfg.materials.zbar.model != "tabular") {
    return;
  }

  const std::size_t n_cells = state.rho.size();
  const std::size_t n_materials = cfg.materials.materials.size();
  if (n_cells == 0 || n_materials == 0) {
    return;
  }

  TENRYU_ASSERT(cfg.materials.zbar_tables.size() == n_materials,
                "update_zbar_tabular: zbar_tables size mismatch");
  TENRYU_ASSERT(state.Te.size() == n_cells,
                "update_zbar_tabular Te/rho size mismatch");
  TENRYU_ASSERT(state.zbar.size() == n_cells,
                "update_zbar_tabular zbar/rho size mismatch");
  TENRYU_ASSERT(state.cell_is_void.size() == n_cells,
                "update_zbar_tabular cell_is_void/rho size mismatch");
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "update_zbar_tabular requires at least one non-void material");
  const std::size_t first_nonvoid_idx = static_cast<std::size_t>(first_nonvoid);

  std::vector<double> rho(n_cells, 0.0);
  std::vector<double> Te(n_cells, 0.0);
  std::vector<double> zbar(n_cells, 0.0);
  {
    std::vector<double> zbar_pack(2 * n_cells);
    const double* srcs[2] = {state.rho.data(), state.Te.data()};
    core::pack_pull_fields(srcs, 2, static_cast<int>(n_cells),
                           zbar_pack.data(),
                           "driver:update_zbar_tabular:pull");
    std::memcpy(rho.data(), zbar_pack.data(), n_cells * sizeof(double));
    std::memcpy(Te.data(), zbar_pack.data() + n_cells,
                n_cells * sizeof(double));
  }

  std::vector<double> volfrac;
  if (n_materials > 1) {
    const std::size_t expected = n_cells * n_materials;
    TENRYU_ASSERT(state.volFrac.size() == expected,
                  "update_zbar_tabular volFrac size mismatch");
    volfrac.resize(expected, 0.0);
    state.volFrac.copy_to_host(volfrac.data());
  }

  for (std::size_t c = 0; c < n_cells; ++c) {
    if (n_materials == 1) {
      zbar[c] = cfg.materials.zbar_tables[first_nonvoid_idx]->interpolate(rho[c], Te[c]);
      continue;
    }

    double weighted = 0.0;
    double frac_sum = 0.0;
    const std::size_t base = c * n_materials;
    for (std::size_t m = 0; m < n_materials; ++m) {
      if (cfg.materials.materials[m].is_void) {
        continue;
      }
      const double f = std::max(volfrac[base + m], 0.0);
      frac_sum += f;
      const double zbar_m = cfg.materials.zbar_tables[m]->interpolate(rho[c], Te[c]);
      weighted += f * zbar_m;
    }
    if (frac_sum > 1.0e-30) {
      weighted /= frac_sum;
    }
    zbar[c] = weighted;
  }
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (state.cell_is_void[c] != 0U) {
      zbar[c] = 0.0;
    }
  }

  state.zbar.copy_from_host(zbar.data());
}

void update_zbar_for_step(core::State& state, const core::Config& cfg) {
  if (cfg.materials.zbar.model == "fixed") {
    // fixed Zbar: no per-step update needed, initialized at setup.
    return;
  }
  update_zbar_thomas_fermi(state, cfg);
  update_zbar_tabular(state, cfg);
}

void update_next_output_time(double& t_next,
                             const double every_s,
                             const double t_now) {
  if (every_s <= 0.0 || t_next < 0.0) {
    return;
  }
  const double eps = 1.0e-14 * std::max(std::abs(t_now), every_s);
  while (t_next <= t_now + eps) {
    t_next += every_s;
  }
}

std::string format_sci(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(3) << value;
  return oss.str();
}

static std::optional<hydro::AleStateMachine> ale_monitor_machine;
static double ale_monitor_last_t = -std::numeric_limits<double>::infinity();
static int ale_monitor_evaluation_count = 0;
static int last_event_step;
static int consecutive_failures;
static bool controller_disengaged;
static bool controller_activated;
static double controller_activation_h_s;
static bool morph_window_suppression_logged;
static std::vector<double> latest_sector_front;

void reset_runtime_ale_monitor_state(const core::Config& cfg) {
  ale_monitor_machine.emplace(cfg.numerics.ale.runtime_controller);
  ale_monitor_evaluation_count = 0;
  last_event_step = 0;
  consecutive_failures = 0;
  controller_disengaged = false;
  controller_activated = false;
  controller_activation_h_s = std::numeric_limits<double>::quiet_NaN();
  morph_window_suppression_logged = false;
  latest_sector_front.clear();
}

bool runtime_ale_controller_suppressed_by_morph_window(
    const core::State& state,
    const core::Config& cfg) {
  const bool suppressed = cfg.numerics.ale.button_morph.enabled &&
                          state.t < cfg.numerics.ale.button_morph.t_end_s;
  if (suppressed && !morph_window_suppression_logged) {
    core::log_info("[runtime-ale] suppressed_morph_window=1");
    morph_window_suppression_logged = true;
  }
  return suppressed;
}

const double* runtime_ale_sector_front_or_null(const core::State& state) {
  const int ntheta = state.mesh.topo.nz;
  if (ntheta <= 0 ||
      latest_sector_front.size() != static_cast<std::size_t>(ntheta)) {
    return nullptr;
  }
  const bool any_valid = std::any_of(
      latest_sector_front.begin(), latest_sector_front.end(),
      [](const double value) { return value > 0.0; });
  return any_valid ? latest_sector_front.data() : nullptr;
}

bool runtime_ale_controller_activation_reached(const core::State& state,
                                               const core::Config& cfg) {
  if (controller_activated) {
    return true;
  }

  const int ntheta = state.mesh.topo.nz;
  const auto& runtime = cfg.numerics.ale.runtime_controller;
  double activation_front =
      runtime.activation_front_mode == "mean"
          ? 0.0
          : std::numeric_limits<double>::infinity();
  int valid_fronts = 0;
  if (ntheta > 0 &&
      latest_sector_front.size() == static_cast<std::size_t>(ntheta)) {
    for (const double value : latest_sector_front) {
      if (std::isfinite(value) && value > 0.0) {
        if (runtime.activation_front_mode == "mean") {
          activation_front += value;
        } else {
          activation_front = std::min(activation_front, value);
        }
        ++valid_fronts;
      }
    }
  }

  if (valid_fronts > 0) {
    if (runtime.activation_front_mode == "mean") {
      activation_front /= static_cast<double>(valid_fronts);
    }
    const auto idx = hydro::button_morph::make_button_indexing(cfg.mesh);
    if (!std::isfinite(controller_activation_h_s)) {
      TENRYU_ASSERT(state.x_r_initial.size() ==
                        static_cast<std::size_t>(idx.n_nodes_total) &&
                        state.x_z_initial.size() ==
                            static_cast<std::size_t>(idx.n_nodes_total),
                    "runtime ALE activation requires initial node coordinates");
      std::vector<double> initial_r;
      std::vector<double> initial_z;
      state.x_r_initial.copy_to_host(initial_r);
      state.x_z_initial.copy_to_host(initial_z);
      const int seam_node = hydro::button_morph::shell_node_id(idx, 0, 0);
      const int first_shell_node =
          hydro::button_morph::shell_node_id(idx, 1, 0);
      controller_activation_h_s =
          std::hypot(initial_r[static_cast<std::size_t>(first_shell_node)],
                     initial_z[static_cast<std::size_t>(first_shell_node)]) -
          std::hypot(initial_r[static_cast<std::size_t>(seam_node)],
                     initial_z[static_cast<std::size_t>(seam_node)]);
    }
    TENRYU_ASSERT(std::isfinite(controller_activation_h_s) &&
                      controller_activation_h_s > 0.0,
                  "runtime ALE activation shell first-row spacing must be positive and finite");
    const double bridge_inner_radius = idx.r_c;
    const double front_margin =
        runtime.activation_front_margin_hs * controller_activation_h_s;
    const double front_threshold = bridge_inner_radius + front_margin;
    if (!(activation_front < front_threshold)) {
      return false;
    }
    controller_activated = true;
    core::log_info("[runtime-ale] activated step=" +
                   std::to_string(state.step) + " t=" + format_sci(state.t) +
                   " gate=front " + runtime.activation_front_mode +
                   "_front=" + format_sci(activation_front) +
                   " threshold=" + format_sci(front_threshold) +
                   " bridge_inner_radius=" +
                   format_sci(bridge_inner_radius) +
                   " front_margin=" + format_sci(front_margin));
    return true;
  }

  if (!(runtime.activation_time_s > 0.0 &&
        state.t >= runtime.activation_time_s)) {
    return false;
  }
  controller_activated = true;
  core::log_info("[runtime-ale] activated step=" + std::to_string(state.step) +
                 " t=" + format_sci(state.t) +
                 " gate=time activation_time_s=" +
                 format_sci(runtime.activation_time_s));
  return true;
}

const char* ale_monitor_state_name(const hydro::AleMonitorState state) {
  switch (state) {
    case hydro::AleMonitorState::kOff:
      return "OFF";
    case hydro::AleMonitorState::kWarning:
      return "WARNING";
    case hydro::AleMonitorState::kSoft:
      return "SOFT";
    case hydro::AleMonitorState::kHard:
      return "HARD";
    case hydro::AleMonitorState::kRecovery:
      return "RECOVERY";
  }
  return "OFF";
}

const char* runtime_ale_mode_name(const hydro::AleMonitorState state) {
  switch (state) {
    case hydro::AleMonitorState::kSoft:
      return "soft";
    case hydro::AleMonitorState::kHard:
      return "hard";
    case hydro::AleMonitorState::kRecovery:
      return "recovery";
    case hydro::AleMonitorState::kOff:
    case hydro::AleMonitorState::kWarning:
      return "off";
  }
  return "off";
}

void log_runtime_ale_reject_geometry(const core::State& state,
                                     const int cell) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "runtime ALE rejection geometry requires CSR topology");
  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(cell >= 0 && cell < state.mesh.topo.n_cells,
                "runtime ALE rejection geometry cell out of range");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(state.mesh.topo.n_cells) + 1U,
                "runtime ALE rejection geometry requires CSR offsets");
  const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int end =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
  TENRYU_ASSERT(begin >= 0 && end - begin == 4 &&
                    static_cast<std::size_t>(end) <=
                        mb.cell_node_csr_indices.size(),
                "runtime ALE rejection geometry requires four CSR nodes");

  std::array<int, 4> nodes{};
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    nodes[k] =
        mb.cell_node_csr_indices[static_cast<std::size_t>(begin) + k];
  }
  std::vector<double> host_r;
  std::vector<double> host_z;
  state.x_r.copy_to_host(host_r);
  state.x_z.copy_to_host(host_z);
  std::array<double, 4> node_r{};
  std::array<double, 4> node_z{};
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    const int node = nodes[k];
    TENRYU_ASSERT(node >= 0 && node < state.mesh.topo.n_nodes,
                  "runtime ALE rejection geometry node out of range");
    node_r[k] = host_r[static_cast<std::size_t>(node)];
    node_z[k] = host_z[static_cast<std::size_t>(node)];
  }

  std::ostringstream os;
  os << "[runtime-ale-reject-geom] cell=" << cell;
  for (std::size_t k = 0; k < nodes.size(); ++k) {
    os << " n" << k << "=" << nodes[k] << ":(" << format_sci(node_r[k])
       << "," << format_sci(node_z[k]) << ")";
  }
  core::log_info(os.str());
}

void run_runtime_ale_controller_event_if_due(
    core::State& state,
    const core::Config& cfg,
    const hydro::AleMonitorState monitor_state) {
  const auto& runtime = cfg.numerics.ale.runtime_controller;
  if (!runtime.controller_enabled || controller_disengaged) {
    return;
  }
  if (runtime_ale_controller_suppressed_by_morph_window(state, cfg)) {
    return;
  }
  if (!runtime_ale_controller_activation_reached(state, cfg)) {
    return;
  }
  const int cadence = hydro::runtime_ale_cadence(
      monitor_state, runtime.cadence_soft, runtime.cadence_hard,
      runtime.cadence_recovery);
  if (cadence == 0 || state.step - last_event_step < cadence) {
    return;
  }

  hydro::RuntimeAleEscalation escalation;
  int escalation_level = 0;
  if (consecutive_failures >= runtime.failures_big_repair) {
    escalation.cap_scale = 2.0;
    escalation.sweep_scale = 2;
    escalation.force_hard = true;
    escalation_level = 2;
  } else if (consecutive_failures >= runtime.failures_hard_force) {
    escalation.force_hard = true;
    escalation_level = 1;
  }
  const hydro::RuntimeAleEventResult event = hydro::run_runtime_ale_event(
      state, cfg, monitor_state == hydro::AleMonitorState::kHard,
      escalation, runtime_ale_sector_front_or_null(state));
  last_event_step = state.step;
  if (event.succeeded) {
    consecutive_failures = 0;
  } else if (event.attempted) {
    ++consecutive_failures;
  }

  std::ostringstream os;
  os << "[runtime-ale] step=" << state.step
     << " t=" << format_sci(state.t)
     << " mode=" << runtime_ale_mode_name(monitor_state)
     << " esc=" << escalation_level
     << " attempted=" << (event.attempted ? 1 : 0)
     << " succeeded=" << (event.succeeded ? 1 : 0)
     << " rolled_back=" << (event.rolled_back ? 1 : 0)
     << " lambda=" << format_sci(event.lambda_accepted)
     << " iters=" << event.linesearch_iters
     << " max_disp=" << format_sci(event.max_displacement)
     << " capped=" << event.capped_nodes
     << " rclamp=" << event.half_plane_clamped_nodes
     << " capproj=" << event.cap_projected_nodes
     << " ordproj=" << event.ordinary_projected_nodes
     << " capinf=" << event.cap_infeasible_events
     << " patch_escalations=" << event.patch_escalations
     << " patch_nodes_max=" << event.patch_nodes_max
     << " patch_oversize_skips=" << event.patch_oversize_skips
     << " cm_cell=" << format_sci(event.corner_mass_cell_residual)
     << " failures=" << consecutive_failures;
  if (!event.succeeded) {
    os << " rej=" << event.last_reject_reason << ":"
       << event.last_reject_cell;
  }
  core::log_info(os.str());
  if (!event.succeeded && event.last_reject_cell >= 0) {
    log_runtime_ale_reject_geometry(state, event.last_reject_cell);
  }

  if (consecutive_failures >= runtime.escalation_max_failures) {
    controller_disengaged = true;
    core::log_warning(
        "[runtime-ale] DISENGAGED step=" + std::to_string(state.step) +
        " t=" + format_sci(state.t) +
        " failures=" + std::to_string(consecutive_failures));
  }
}

void emit_polar_tier_origin_force_diag(const core::State& state,
                                       const core::Config& cfg) {
  if (!hydro::polar_tier_fo_diag_enabled() || !cfg.diagnostics.enabled ||
      cfg.diagnostics.every < 1 ||
      state.step % cfg.diagnostics.every != 0) {
    return;
  }
  const auto& diag = state.polar_tier_origin_force_diag;
  if (!diag.valid) {
    return;
  }
  std::ostringstream os;
  os << std::scientific << std::setprecision(17)
     << "[polar-tier-FO] step=" << state.step
     << " t=" << state.t
     << " node=" << diag.node
     << " F_r=" << diag.raw_force_r
     << " F_z=" << diag.raw_force_z
     << " norm=" << std::hypot(diag.raw_force_r, diag.raw_force_z);
  core::log_info(os.str());
}

void run_ale_state_monitor(core::State& state,
                           const core::Config& cfg,
                           const double dt_acoustic_last,
                           ProfileObservability& observability) {
  const auto& runtime = cfg.numerics.ale.runtime_controller;
  if (!runtime.monitor_enabled || state.step % runtime.monitor_every != 0) {
    return;
  }

  if (!ale_monitor_machine.has_value() || state.t <= ale_monitor_last_t) {
    reset_runtime_ale_monitor_state(cfg);
  }
  ale_monitor_last_t = state.t;

  if (!state.ale_monitor_ref_valid) {
    hydro::capture_ale_monitor_reference(state, false);
  }
  if (cfg.numerics.ale.button_morph.enabled &&
      !state.ale_monitor_ref_post_morph &&
      state.t >= cfg.numerics.ale.button_morph.t_end_s) {
    hydro::capture_ale_monitor_reference(state, true);
  }

  const hydro::AleMonitorResult result =
      hydro::evaluate_ale_monitor(state, cfg, dt_acoustic_last);
  if (!result.valid) {
    return;
  }

  const hydro::AleMonitorState previous = ale_monitor_machine->state();
  const hydro::AleMonitorState current =
      ale_monitor_machine->update(result.q_min, result.h_min);
  ++ale_monitor_evaluation_count;

  if (current != previous &&
      (current == hydro::AleMonitorState::kRecovery ||
       current == hydro::AleMonitorState::kOff)) {
    consecutive_failures = 0;
    if (controller_disengaged) {
      core::log_info("[runtime-ale] RE-ENGAGED step=" +
                     std::to_string(state.step) +
                     " t=" + format_sci(state.t) +
                     " state=" + ale_monitor_state_name(current));
    }
    controller_disengaged = false;
  }

  observability.ale_monitor_observed = true;
  observability.ale_monitor_state = static_cast<int>(current);
  observability.ale_monitor_q_min = result.q_min;
  observability.ale_monitor_h_min = result.h_min;
  observability.ale_monitor_q_min_cell = result.q_min_cell;
  observability.ale_monitor_h_min_cell = result.h_min_cell;

  if (hydro::fold_diag_enabled()) {
    std::ostringstream os;
    os << "[fold-diag] step=" << state.step
       << " t=" << format_sci(state.t)
       << " rz_max=" << format_sci(result.r_z_max) << "@"
       << result.r_z_max_cell
       << " uth=" << format_sci(result.r_z_max_u_theta)
       << " suth=" << format_sci(result.r_z_max_s_u_theta)
       << " Hz_max=" << format_sci(result.h_metric_max) << "@"
       << result.h_metric_max_cell
       << " uth_h=" << format_sci(result.h_metric_max_u_theta);
    core::log_info(os.str());

    for (const hydro::AleMonitorTrackedCellResult& cell :
         result.tracked_cells) {
      std::ostringstream cell_os;
      cell_os << "[fold-cell] step=" << state.step
              << " t=" << format_sci(state.t)
              << " cell=" << cell.cell
              << " q=" << format_sci(cell.q)
              << " rz=" << format_sci(cell.r_z)
              << " Hz=" << format_sci(cell.h_metric)
              << " uth=" << format_sci(cell.u_theta)
              << " suth=" << format_sci(cell.s_u_theta)
              << " alt_min=" << format_sci(cell.alt_min);
      core::log_info(cell_os.str());
    }
  }

  if (current != previous || ale_monitor_evaluation_count % 50 == 0) {
    std::ostringstream os;
    os << "[ale-state] step=" << state.step
       << " t=" << format_sci(state.t)
       << " state=" << ale_monitor_state_name(current)
       << " q_min=" << format_sci(result.q_min)
       << " q_cell=" << result.q_min_cell
       << " h_min=" << format_sci(result.h_min)
       << " h_cell=" << result.h_min_cell
       << " eval_cells=" << result.evaluated_cells;
    core::log_info(os.str());
  }
  run_runtime_ale_controller_event_if_due(state, cfg, current);
}

void run_shock_approach_diag(const core::State& state, const core::Config& cfg) {
  const auto& sa = cfg.numerics.diagnostics.shock_approach;
  if (!sa.enabled || sa.every <= 0 || state.step % sa.every != 0) {
    return;
  }
  static tenryu::diagnostics::ShockApproachTracker tracker(16);
  static double last_t = -1.0;
  if (state.t <= last_t) {   // new run / time reset in-process
    tracker = tenryu::diagnostics::ShockApproachTracker(16);
  }
  last_t = state.t;

  const int n_cells = state.mesh.topo.n_cells;
  if (n_cells < 64) return;
  const std::vector<double>& cr = state.mesh.cell_centroid_r;
  const std::vector<double>& cz = state.mesh.cell_centroid_z;
  if (static_cast<int>(cr.size()) < n_cells ||
      static_cast<int>(cz.size()) < n_cells) {
    return;
  }
  std::vector<double> pe(n_cells), pi(n_cells);
  state.Pe.copy_to_host(pe.data());
  state.Pi.copy_to_host(pi.data());
  double s_max = 0.0;
  for (int c = 0; c < n_cells; ++c) {
    const double s = std::hypot(cr[c], cz[c]);
    if (!std::isfinite(s)) continue;
    s_max = std::max(s_max, s);
  }
  if (!(s_max > 0.0)) return;
  const int nb = sa.bins;
  std::vector<double> sum_p(nb, 0.0);
  std::vector<int> cnt(nb, 0);
  for (int c = 0; c < n_cells; ++c) {
    const double s = std::hypot(cr[c], cz[c]);
    if (!std::isfinite(s)) continue;
    int b = static_cast<int>(s / s_max * nb);
    if (b >= nb) b = nb - 1;
    sum_p[b] += pe[c] + pi[c];
    ++cnt[b];
  }
  std::vector<double> sb, pb;
  sb.reserve(nb); pb.reserve(nb);
  for (int b = 0; b < nb; ++b) {
    if (cnt[b] < 4) continue;
    const double p_mean = sum_p[b] / cnt[b];
    if (!std::isfinite(p_mean) || !(p_mean > 0.0)) continue;
    sb.push_back((b + 0.5) * s_max / nb);
    pb.push_back(p_mean);
  }
  if (static_cast<int>(sb.size()) < 8) return;
  const double ridge = tenryu::diagnostics::shock_ridge_radius(
      sb.data(), pb.data(), static_cast<int>(sb.size()));
  tracker.push(state.t, ridge);
  double v = 0.0, t_arr = 0.0, n_lead = 0.0;
  const bool have_v = tracker.speed(&v);
  const bool have_arr = (sa.target_radius_cm > 0.0) &&
                        tracker.predict_arrival(sa.target_radius_cm, &t_arr);
  const double h_cell = sa.h_cell_cm > 0.0 ? sa.h_cell_cm : s_max / nb;
  const bool have_lead = have_v && have_arr &&
      tenryu::diagnostics::crossing_time_lead(state.t, t_arr, v, h_cell, &n_lead);
  std::ostringstream os;
  os << "[shock_approach] step=" << state.step
     << " t=" << format_sci(state.t)
     << " s_ridge=" << format_sci(ridge)
     << " v_inward=" << (have_v ? format_sci(v) : "n/a")
     << " t_arrival=" << (have_arr ? format_sci(t_arr) : "n/a")
     << " n_lead=" << (have_lead ? format_sci(n_lead) : "n/a");
  core::log_info(os.str());

  if (sa.sectors > 0) {
    static tenryu::diagnostics::SectorShockState sectors_state(sa.sectors);
    if (sectors_state.n_sec != sa.sectors ||
        state.t <= sectors_state.last_call_t) {
      sectors_state =
          tenryu::diagnostics::SectorShockState(sa.sectors);
    }

    const int n_sec = sa.sectors;
    const double pi_value = std::acos(-1.0);
    const double dt_scan = sectors_state.last_call_t > 0.0
                               ? state.t - sectors_state.last_call_t
                               : 0.0;
    std::vector<double> sector_sum_p(
        static_cast<std::size_t>(n_sec * nb), 0.0);
    std::vector<int> sector_cnt(static_cast<std::size_t>(n_sec * nb), 0);
    for (int c = 0; c < n_cells; ++c) {
      const double s = std::hypot(cr[c], cz[c]);
      if (!std::isfinite(s)) continue;
      const double theta = std::atan2(cr[c], cz[c]);
      int k = static_cast<int>(theta / pi_value * n_sec);
      k = std::clamp(k, 0, n_sec - 1);
      int b = static_cast<int>(s / s_max * nb);
      if (b >= nb) b = nb - 1;
      const std::size_t index = static_cast<std::size_t>(k * nb + b);
      sector_sum_p[index] += pe[c] + pi[c];
      ++sector_cnt[index];
    }

    std::vector<tenryu::diagnostics::SectorFrontSample> front_samples(
        static_cast<std::size_t>(n_sec),
        tenryu::diagnostics::SectorFrontSample{0.0, false});
    std::vector<double> sector_sb;
    std::vector<double> sector_pb;
    sector_sb.reserve(static_cast<std::size_t>(nb));
    sector_pb.reserve(static_cast<std::size_t>(nb));
    for (int k = 0; k < n_sec; ++k) {
      sector_sb.clear();
      sector_pb.clear();
      for (int b = 0; b < nb; ++b) {
        const std::size_t index = static_cast<std::size_t>(k * nb + b);
        if (sector_cnt[index] < 2) continue;
        const double p_mean = sector_sum_p[index] / sector_cnt[index];
        if (!std::isfinite(p_mean) || !(p_mean > 0.0)) continue;
        sector_sb.push_back((b + 0.5) * s_max / nb);
        sector_pb.push_back(p_mean);
      }
      if (static_cast<int>(sector_sb.size()) < 8) continue;
      front_samples[static_cast<std::size_t>(k)].s_ridge =
          tenryu::diagnostics::shock_ridge_radius(
              sector_sb.data(),
              sector_pb.data(),
              static_cast<int>(sector_sb.size()));
      front_samples[static_cast<std::size_t>(k)].valid = true;
    }

    if (cfg.numerics.ale.runtime_controller.controller_enabled) {
      const int ntheta = state.mesh.topo.nz;
      latest_sector_front.assign(
          static_cast<std::size_t>(std::max(ntheta, 0)), -1.0);
      for (int j = 0; j < ntheta; ++j) {
        const int sector = std::clamp(
            static_cast<int>((static_cast<double>(j) + 0.5) * n_sec /
                             static_cast<double>(ntheta)),
            0, n_sec - 1);
        const auto& sample = front_samples[static_cast<std::size_t>(sector)];
        if (sample.valid) {
          latest_sector_front[static_cast<std::size_t>(j)] = sample.s_ridge;
        }
      }
    }

    std::vector<double> v_inward(static_cast<std::size_t>(n_sec), 0.0);
    std::vector<double> t_arrival(static_cast<std::size_t>(n_sec), 0.0);
    std::vector<int> have_v(static_cast<std::size_t>(n_sec), 0);
    std::vector<int> have_arrival(static_cast<std::size_t>(n_sec), 0);
    std::unique_ptr<bool[]> deadline_valid(new bool[n_sec]);
    int max_tracker_size = 0;
    for (int k = 0; k < n_sec; ++k) {
      deadline_valid[k] = false;
      const auto& sample = front_samples[static_cast<std::size_t>(k)];
      auto& sector_tracker =
          sectors_state.trackers[static_cast<std::size_t>(k)];
      if (sample.valid && state.t > sectors_state.last_call_t) {
        sector_tracker.push(state.t, sample.s_ridge);
      }
      have_v[static_cast<std::size_t>(k)] =
          sector_tracker.speed(&v_inward[static_cast<std::size_t>(k)]);
      have_arrival[static_cast<std::size_t>(k)] =
          (sa.target_radius_cm > 0.0) &&
          sector_tracker.predict_arrival(
              sa.target_radius_cm,
              &t_arrival[static_cast<std::size_t>(k)]);
      double residual_rms = 0.0;
      const bool have_residual = sector_tracker.fit_residual_rms(&residual_rms);
      if (have_v[static_cast<std::size_t>(k)] && have_residual) {
        sectors_state.last_sigma_t[static_cast<std::size_t>(k)] =
            residual_rms /
            std::max(std::abs(v_inward[static_cast<std::size_t>(k)]),
                     1.0e-30);
      }
      deadline_valid[k] = sample.valid &&
                          have_v[static_cast<std::size_t>(k)] &&
                          have_arrival[static_cast<std::size_t>(k)] &&
                          have_residual;
      if (deadline_valid[k]) {
        max_tracker_size = std::max(max_tracker_size, sector_tracker.size());
      }
    }

    std::vector<double> modal_mu(static_cast<std::size_t>(n_sec), 0.0);
    std::vector<double> modal_y(static_cast<std::size_t>(n_sec), 0.0);
    std::vector<double> modal_w(static_cast<std::size_t>(n_sec), 0.0);
    int valid_count = 0;
    double s_f_min = std::numeric_limits<double>::infinity();
    double s_f_max = -std::numeric_limits<double>::infinity();
    double s_f_sum = 0.0;
    for (int k = 0; k < n_sec; ++k) {
      const double theta_center =
          (static_cast<double>(k) + 0.5) * pi_value / n_sec;
      modal_mu[static_cast<std::size_t>(k)] = std::cos(theta_center);
      const auto& sample = front_samples[static_cast<std::size_t>(k)];
      if (!sample.valid) continue;
      ++valid_count;
      s_f_min = std::min(s_f_min, sample.s_ridge);
      s_f_max = std::max(s_f_max, sample.s_ridge);
      s_f_sum += sample.s_ridge;
      modal_y[static_cast<std::size_t>(k)] = std::log(sample.s_ridge);
      modal_w[static_cast<std::size_t>(k)] =
          std::cos(static_cast<double>(k) * pi_value / n_sec) -
          std::cos(static_cast<double>(k + 1) * pi_value / n_sec);
    }
    std::array<double, 5> modal_b{};
    tenryu::diagnostics::legendre_fit(
        modal_mu.data(),
        modal_y.data(),
        modal_w.data(),
        n_sec,
        sa.modal_l_max,
        modal_b.data());

    double new_deadline = 0.0;
    const bool have_new_deadline =
        tenryu::diagnostics::earliest_confidence_deadline(
            t_arrival.data(),
            sectors_state.last_sigma_t.data(),
            v_inward.data(),
            deadline_valid.get(),
            n_sec,
            sa.sector_confidence_nu,
            sa.sector_guard_crossings,
            h_cell,
            dt_scan,
            &new_deadline);
    bool deadline_held = false;
    const bool deadline_immature =
        have_new_deadline &&
        !(new_deadline > state.t && max_tracker_size >= 8);
    if (have_new_deadline) {
      tenryu::diagnostics::commit_sector_deadline(
          &sectors_state.committed_deadline_s,
          new_deadline,
          state.t,
          max_tracker_size,
          &deadline_held);
    }
    sectors_state.last_call_t = state.t;

    std::ostringstream sector_os;
    sector_os << "[shock_sectors] step=" << state.step
              << " t=" << format_sci(state.t)
              << " valid=" << valid_count << "/" << n_sec
              << " s_f_min="
              << (valid_count > 0 ? format_sci(s_f_min) : "n/a")
              << " s_f_max="
              << (valid_count > 0 ? format_sci(s_f_max) : "n/a")
              << " s_f_mean="
              << (valid_count > 0
                      ? format_sci(s_f_sum / static_cast<double>(valid_count))
                      : "n/a")
              << " b0=" << format_sci(modal_b[0])
              << " b1=" << format_sci(modal_b[1])
              << " b2=" << format_sci(modal_b[2])
              << " b3=" << format_sci(modal_b[3])
              << " b4=" << format_sci(modal_b[4])
              << " t_end="
              << (std::isfinite(sectors_state.committed_deadline_s)
                      ? format_sci(sectors_state.committed_deadline_s)
                      : "n/a")
              << " deadline_held=" << (deadline_held ? 1 : 0)
              << " deadline_immature=" << (deadline_immature ? 1 : 0)
              << " sectors=";
    bool first_sector = true;
    for (int k = 0; k < n_sec; ++k) {
      const auto& sample = front_samples[static_cast<std::size_t>(k)];
      if (!sample.valid) continue;
      if (!first_sector) sector_os << ",";
      first_sector = false;
      sector_os << k << ":" << format_sci(sample.s_ridge) << ":"
                << (have_v[static_cast<std::size_t>(k)]
                        ? format_sci(v_inward[static_cast<std::size_t>(k)])
                        : "n/a")
                << ":"
                << (have_arrival[static_cast<std::size_t>(k)]
                        ? format_sci(t_arrival[static_cast<std::size_t>(k)])
                        : "n/a");
    }
    core::log_info(sector_os.str());
  }
}

void insert_i1b_sensor_top(
    std::array<hydro::I1BSpuriousSensorCell,
               hydro::kI1BSpuriousSensorTopKMax>& top,
    const int top_k,
    const hydro::I1BSpuriousSensorCell& candidate) {
  if (top_k <= 0 || candidate.cell_id < 0 || !(candidate.score > 0.0) ||
      !std::isfinite(candidate.score)) {
    return;
  }
  for (int slot = 0; slot < top_k; ++slot) {
    if (candidate.score > top[static_cast<std::size_t>(slot)].score) {
      for (int dst = top_k - 1; dst > slot; --dst) {
        top[static_cast<std::size_t>(dst)] =
            top[static_cast<std::size_t>(dst - 1)];
      }
      top[static_cast<std::size_t>(slot)] = candidate;
      return;
    }
  }
}

void refresh_i1b_sensor_top_count(hydro::I1BSpuriousSensorSummary& summary,
                                  const int top_k) {
  summary.top_count = 0;
  for (int k = 0; k < top_k; ++k) {
    if (summary.top[static_cast<std::size_t>(k)].cell_id >= 0) {
      ++summary.top_count;
    }
  }
}

void merge_i1b_sensor_summary(hydro::I1BSpuriousSensorSummary& dst,
                              const hydro::I1BSpuriousSensorSummary& src,
                              const int rank,
                              const int top_k) {
  if (!src.valid) {
    return;
  }
  dst.valid = true;
  dst.sampled_cells += src.sampled_cells;
  dst.residual_ke_sum += src.residual_ke_sum;
  dst.residual_ke_max = std::max(dst.residual_ke_max, src.residual_ke_max);
  for (int k = 0; k < src.top_count && k < top_k; ++k) {
    hydro::I1BSpuriousSensorCell candidate =
        src.top[static_cast<std::size_t>(k)];
    candidate.rank = rank;
    insert_i1b_sensor_top(dst.top, top_k, candidate);
  }
  refresh_i1b_sensor_top_count(dst, top_k);
}

std::array<hydro::I1BSpuriousSensorCell,
           hydro::kI1BSpuriousSensorTopKMax>
gather_i1b_sensor_top(const hydro::I1BSpuriousSensorSummary& local,
                      const parallel::Reduction& reducer,
                      const int n_ranks,
                      const int top_k) {
  constexpr int kFields = 7;
  const int sendcount = hydro::kI1BSpuriousSensorTopKMax * kFields;
  std::vector<double> send(static_cast<std::size_t>(sendcount), 0.0);
  for (int k = 0; k < hydro::kI1BSpuriousSensorTopKMax; ++k) {
    const auto& cell = local.top[static_cast<std::size_t>(k)];
    const int base = k * kFields;
    send[static_cast<std::size_t>(base + 0)] = static_cast<double>(cell.rank);
    send[static_cast<std::size_t>(base + 1)] =
        static_cast<double>(cell.cell_id);
    send[static_cast<std::size_t>(base + 2)] =
        static_cast<double>(cell.active_nverts);
    send[static_cast<std::size_t>(base + 3)] = cell.residual_ke;
    send[static_cast<std::size_t>(base + 4)] = cell.affine_ke;
    send[static_cast<std::size_t>(base + 5)] = cell.eta2;
    send[static_cast<std::size_t>(base + 6)] = cell.score;
  }

  const int ranks = std::max(n_ranks, 1);
  std::vector<int> counts(static_cast<std::size_t>(ranks), sendcount);
  std::vector<int> displs(static_cast<std::size_t>(ranks), 0);
  for (int r = 1; r < ranks; ++r) {
    displs[static_cast<std::size_t>(r)] =
        displs[static_cast<std::size_t>(r - 1)] + sendcount;
  }
  std::vector<double> recv(static_cast<std::size_t>(sendcount) *
                               static_cast<std::size_t>(ranks),
                           0.0);
  reducer.allgatherv(send.data(), sendcount, recv.data(), counts.data(),
                     displs.data());

  std::array<hydro::I1BSpuriousSensorCell,
             hydro::kI1BSpuriousSensorTopKMax>
      global_top{};
  for (auto& cell : global_top) {
    cell.score = -1.0;
    cell.cell_id = -1;
    cell.rank = -1;
  }
  for (int r = 0; r < ranks; ++r) {
    for (int k = 0; k < hydro::kI1BSpuriousSensorTopKMax; ++k) {
      const int base = displs[static_cast<std::size_t>(r)] + k * kFields;
      hydro::I1BSpuriousSensorCell cell{};
      cell.rank = static_cast<int>(recv[static_cast<std::size_t>(base + 0)]);
      cell.cell_id =
          static_cast<int>(recv[static_cast<std::size_t>(base + 1)]);
      cell.active_nverts =
          static_cast<int>(recv[static_cast<std::size_t>(base + 2)]);
      cell.residual_ke = recv[static_cast<std::size_t>(base + 3)];
      cell.affine_ke = recv[static_cast<std::size_t>(base + 4)];
      cell.eta2 = recv[static_cast<std::size_t>(base + 5)];
      cell.score = recv[static_cast<std::size_t>(base + 6)];
      insert_i1b_sensor_top(global_top, top_k, cell);
    }
  }
  return global_top;
}

void append_i1b_sensor_top(
    std::ostringstream& oss,
    const std::array<hydro::I1BSpuriousSensorCell,
                     hydro::kI1BSpuriousSensorTopKMax>& top,
    const int top_k,
    const char* const residual_label,
    const char* const reference_label,
    const char* const fraction_label) {
  oss << '[';
  bool first = true;
  for (int k = 0; k < top_k; ++k) {
    const auto& cell = top[static_cast<std::size_t>(k)];
    if (cell.cell_id < 0 || !(cell.score > 0.0)) {
      continue;
    }
    if (!first) {
      oss << ',';
    }
    first = false;
    oss << "{rank=" << cell.rank << ",cell=" << cell.cell_id
        << ",nverts=" << cell.active_nverts << ',' << residual_label << '='
        << cell.residual_ke << ',' << reference_label << '=' << cell.affine_ke
        << ',' << fraction_label << '=' << cell.eta2 << '}';
  }
  oss << ']';
}

void emit_i1b_spurious_sensor_log(
    const core::State& state,
    const diagnostics::PhaseResolvedEnergyDiagnostics& phase_energy,
    const hydro::I1BSpuriousSensorSummary& hydro_local,
    const hydro::I1BSpuriousSensorSummary& ale_local,
    const parallel::PartitionInfo& part,
    const parallel::Reduction& reducer,
    const int top_k) {
  std::array<double, 4> sums = {
      hydro_local.residual_ke_sum,
      static_cast<double>(hydro_local.sampled_cells),
      ale_local.residual_ke_sum,
      static_cast<double>(ale_local.sampled_cells),
  };
  reducer.allreduce_sum(sums.data(), static_cast<int>(sums.size()));
  std::array<double, 2> maxima = {
      hydro_local.residual_ke_max,
      ale_local.residual_ke_max,
  };
  reducer.allreduce_max(maxima.data(), static_cast<int>(maxima.size()));
  const auto hydro_top =
      gather_i1b_sensor_top(hydro_local, reducer, part.n_ranks, top_k);
  const auto ale_top =
      gather_i1b_sensor_top(ale_local, reducer, part.n_ranks, top_k);

  if (part.rank != 0) {
    return;
  }
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6)
      << "[i1b-spurious-sensor] step=" << (state.step + 1)
      << " dE_hydro_kinetic=" << phase_energy.dE_hydro_kinetic
      << " dE_ale_kinetic=" << phase_energy.dE_ale_kinetic
      << " hydro_residual_ke_sum=" << sums[0]
      << " hydro_residual_ke_max=" << maxima[0]
      << " hydro_sampled=" << static_cast<long long>(sums[1])
      << " hydro_top=";
  append_i1b_sensor_top(
      oss, hydro_top, top_k, "residual_ke", "affine_ke", "eta2");
  oss << " ale_delta_ke_sum=" << sums[2]
      << " ale_delta_ke_max=" << maxima[1]
      << " ale_sampled=" << static_cast<long long>(sums[3])
      << " ale_top=";
  append_i1b_sensor_top(
      oss, ale_top, top_k, "delta_ke", "pre_projection_ke", "delta_frac");
  core::log_info(oss.str());
}

std::string mesh_forensics_stage_label(const HydroFailureStage stage) {
  switch (stage) {
    case HydroFailureStage::PredictorCommit:
    case HydroFailureStage::PostPredictor:
      return "predictor";
    case HydroFailureStage::PostCorrector:
      return "corrector";
    case HydroFailureStage::StepStart:
      return "step_start";
    case HydroFailureStage::RetryRestoreValidate:
      return "retry_restore";
    case HydroFailureStage::Default:
      break;
  }
  return "unknown";
}

std::string geometric_retry_stage_key(const HydroFailureStage stage) {
  return mesh_forensics_stage_label(stage);
}

const StagePerStageTracking* geometric_retry_stage_tracking(
    const GeometricRetryStagnationState& state,
    const HydroFailureStage stage) {
  const auto it = state.per_stage.find(geometric_retry_stage_key(stage));
  return it != state.per_stage.end() ? &it->second : nullptr;
}

int geometric_retry_total_count(const GeometricRetryStagnationState& state) {
  int total = 0;
  for (const auto& entry : state.per_stage) {
    total += entry.second.consecutive_count;
  }
  return total;
}

bool mesh_forensics_corner_reason(const std::string& reason) {
  return reason == "mesh_quality_corner_j" || reason == "in_hydro_corner_j";
}

hydro::ale::AleDriverRetryContext::Reason ale_retry_reason_from_hydro_reason(
    const char* reason) {
  if (reason == nullptr) {
    return hydro::ale::AleDriverRetryContext::Reason::none;
  }
  if (std::strcmp(reason, "corner_j") == 0) {
    return hydro::ale::AleDriverRetryContext::Reason::corner_j;
  }
  if (std::strcmp(reason, "gauss_j") == 0) {
    return hydro::ale::AleDriverRetryContext::Reason::gauss_j;
  }
  if (std::strcmp(reason, "rz_volume") == 0) {
    return hydro::ale::AleDriverRetryContext::Reason::rz_volume;
  }
  return hydro::ale::AleDriverRetryContext::Reason::none;
}

double mesh_forensics_sigma_safe(const HydroStepResult& result) {
  double sigma = result.trial_scale;
  if (!(sigma >= 0.0 && std::isfinite(sigma))) {
    sigma = result.min_metric;
  }
  if (!(sigma >= 0.0 && std::isfinite(sigma))) {
    return 1.0;
  }
  return std::min(sigma, 1.0);
}

tenryu::diagnostics::MeshDegeneracyFailureEvent mesh_forensics_event_from_result(
    const core::State& state,
    const HydroStepResult& result) {
  const std::string reason = result.reason != nullptr ? result.reason : "";
  tenryu::diagnostics::MeshDegeneracyFailureEvent event;
  event.step = state.step;
  event.cell = result.first_failing_cell;
  event.corner =
      mesh_forensics_corner_reason(reason) ? result.first_failing_corner : -1;
  event.stage = static_cast<int>(result.stage);
  event.reason = reason;
  event.sigma_safe = mesh_forensics_sigma_safe(result);
  return event;
}

tenryu::diagnostics::MeshDegeneracyForensicsContext
mesh_forensics_context_from_result(
    const core::State& state,
    const core::Config& cfg,
    const HydroStepResult& result,
    const int retry_attempts,
    const double dt,
    const tenryu::diagnostics::MeshDegeneracyTriggerDecision& decision,
    const tenryu::diagnostics::MeshDegeneracyForensicsState& forensics_state) {
  const std::string reason = result.reason != nullptr ? result.reason : "";
  tenryu::diagnostics::MeshDegeneracyForensicsContext context;
  context.retry_index = retry_attempts;
  context.dt = dt;
  context.t = state.t;
  context.stage = mesh_forensics_stage_label(result.stage);
  context.reason = reason;
  context.cell = result.first_failing_cell;
  context.i = result.first_failing_i;
  context.j = result.first_failing_j;
  if ((context.i < 0 || context.j < 0) && context.cell >= 0 &&
      state.mesh.topo.nz > 0) {
    context.i = context.cell / state.mesh.topo.nz;
    context.j = context.cell % state.mesh.topo.nz;
  }
  context.failing_corner =
      mesh_forensics_corner_reason(reason) ? result.first_failing_corner : -1;
  context.sigma_root_reported = mesh_forensics_sigma_safe(result);
  context.same_cell_consecutive_count = decision.same_cell_consecutive_count;
  context.sigma_band_min_recent = decision.sigma_band_min_recent;
  context.sigma_band_max_recent = decision.sigma_band_max_recent;
  context.sigma_dt_invariant_confirmed =
      decision.sigma_dt_invariant_confirmed;
  context.nodes_valid = result.mesh_forensics_sample_valid;
  context.anti_hourglass_active = cfg.numerics.hydro.hourglass.enabled;
  if (cfg.numerics.diagnostics.mesh_degeneracy_forensics
          .corner_j_source_budget_enabled) {
    (void)forensics_state.extract_corner_j_source_budget_phase_positions(
        "post_lagrange",
        context.cell,
        state.mesh.topo.nz,
        context.post_lagrange);
    (void)forensics_state.extract_corner_j_source_budget_phase_positions(
        "post_ale_rezone",
        context.cell,
        state.mesh.topo.nz,
        context.post_ale_rezone);
    (void)forensics_state.extract_corner_j_source_budget_phase_positions(
        "post_remap",
        context.cell,
        state.mesh.topo.nz,
        context.post_remap);
  }
  for (int k = 0; k < 4; ++k) {
    const auto idx = static_cast<std::size_t>(k);
    const auto& src = result.mesh_forensics_nodes[idx];
    auto& dst = context.nodes[idx];
    dst.r_old = src.r_old;
    dst.z_old = src.z_old;
    dst.r_candidate = src.r_candidate;
    dst.z_candidate = src.z_candidate;
    dst.u_r_old = src.u_r_old;
    dst.u_z_old = src.u_z_old;
    dst.u_r_half = src.u_r_half;
    dst.u_z_half = src.u_z_half;
    dst.u_r_new = src.u_r_new;
    dst.u_z_new = src.u_z_new;
    dst.a_r_pressure = src.a_r_pressure;
    dst.a_z_pressure = src.a_z_pressure;
    dst.a_r_qvisc = src.a_r_qvisc;
    dst.a_z_qvisc = src.a_z_qvisc;
    dst.a_r_hourglass = src.a_r_hourglass;
    dst.a_z_hourglass = src.a_z_hourglass;
    dst.node_mass = src.node_mass;
    dst.corner_mass_into_cell = src.corner_mass_into_cell;
  }
  return context;
}

bool maybe_emit_mesh_degeneracy_forensics(
    const core::State& state,
    const core::Config& cfg,
    const HydroStepResult& result,
    const int retry_attempts,
    const double dt,
    const int rank,
    tenryu::diagnostics::MeshDegeneracyForensicsState& forensics_state,
    const bool force_emit = false,
    const GeometricRetryStagnationState* stagnation_state = nullptr) {
  if (rank != 0 || !result.mesh_forensics_sample_valid) {
    return false;
  }
  const auto& forensics_cfg =
      cfg.numerics.diagnostics.mesh_degeneracy_forensics;
  const auto event = mesh_forensics_event_from_result(state, result);
  auto decision = forensics_state.observe_failure(forensics_cfg, event);
  if (force_emit) {
    decision.emit = true;
    if (stagnation_state != nullptr) {
      if (const auto* tracking =
              geometric_retry_stage_tracking(*stagnation_state, result.stage)) {
        decision.same_cell_consecutive_count = tracking->consecutive_count;
        decision.sigma_band_min_recent = tracking->sigma_band_min;
        decision.sigma_band_max_recent = tracking->sigma_band_max;
        const double sigma_mean =
            0.5 * (tracking->sigma_band_min + tracking->sigma_band_max);
        const double sigma_range =
            tracking->sigma_band_max - tracking->sigma_band_min;
        decision.sigma_dt_invariant_confirmed =
            sigma_mean > 0.0 &&
            sigma_range / sigma_mean <=
                cfg.numerics.hydro.geometric_retry_stagnation.sigma_rel_tol;
      }
    }
  }
  if (!decision.emit) {
    return false;
  }
  const auto context = mesh_forensics_context_from_result(
      state, cfg, result, retry_attempts, dt, decision, forensics_state);
  const auto record =
      tenryu::diagnostics::build_mesh_degeneracy_forensics_record(
          state, cfg, context);
  tenryu::diagnostics::append_mesh_degeneracy_forensics_record_jsonl(
      cfg, record);
  core::log_warning(
      "[mesh-degeneracy-forensics] step=" + std::to_string(state.step) +
      " retry=" + std::to_string(retry_attempts) +
      " cell=" + std::to_string(result.first_failing_cell) +
      " reason=" + event.reason +
      " sigma_safe=" + format_sci(event.sigma_safe) +
      (force_emit ? " -> forced dump emitted" : " -> dump emitted"));
  return true;
}

void assert_driver_retry_supported(const core::Config& cfg) {
  if (cfg.numerics.hydro.driver_full_step_retry_enabled &&
      cfg.radiation.mode == core::RadiationMode::ImcDdmc) {
    TENRYU_ASSERT(false,
                  "Numerics.hydro.driver_full_step_retry_enabled does not support "
                  "RadiationMode::ImcDdmc (v1 scope: deterministic FLD/SN only). "
                  "Set retry_enabled=false or change radiation.mode.");
  }
  if (hydro::i1b_path_guard_enabled() &&
      cfg.radiation.mode == core::RadiationMode::ImcDdmc) {
    TENRYU_ASSERT(false,
                  "TENRYU_I1B_PATH_GUARD does not support RadiationMode::ImcDdmc "
                  "because its full-step snapshot retry is deterministic FLD/SN only. "
                  "Disable the path guard or change radiation.mode.");
  }
}

std::string format_elapsed(std::chrono::steady_clock::duration elapsed) {
  using namespace std::chrono;
  const auto elapsed_ms = duration_cast<milliseconds>(elapsed).count();
  if (elapsed_ms < 60 * 1000) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1)
        << (static_cast<double>(elapsed_ms) / 1000.0) << "s";
    return oss.str();
  }
  const auto total_seconds = duration_cast<seconds>(elapsed).count();
  if (total_seconds < 3600) {
    const auto minutes = total_seconds / 60;
    const auto seconds_part = total_seconds % 60;
    return std::to_string(minutes) + "m " + std::to_string(seconds_part) + "s";
  }
  const auto hours = total_seconds / 3600;
  const auto minutes_part = (total_seconds % 3600) / 60;
  return std::to_string(hours) + "h " + std::to_string(minutes_part) + "m";
}

bool fleck_diag_has_radius_window(const core::Config::DiagnosticsConfig::FleckDiag& cfg) {
  return cfg.r_min_cm >= 0.0 && cfg.r_max_cm >= 0.0;
}

bool fleck_diag_requested(const core::Config& cfg) {
  return cfg.main.verbosity == "verbose" ||
         (cfg.diagnostics.enabled && cfg.diagnostics.fleck_diag.enabled);
}

void log_fleck_diagnostics_if_needed(const core::State& state,
                                     const core::Config& cfg,
                                     const radiation::IMC& imc) {
  const auto& diag = cfg.diagnostics.fleck_diag;
  if (!fleck_diag_requested(cfg)) {
    return;
  }

  const bool use_radius = fleck_diag_has_radius_window(diag);
  const bool has_selection = !diag.cells.empty() || use_radius;
  if (!has_selection) {
    if (diag.enabled) {
      static int warn_count = 0;
      ++warn_count;
      if (warn_count == 1 || warn_count % 100 == 0) {
        core::log_warning(
            "Diagnostics.fleck_diag is enabled but no cells or radius window were configured");
      }
    }
    return;
  }

  const int step_number = state.step + 1;
  if (step_number <= 0 || (step_number % std::max(diag.every, 1)) != 0) {
    return;
  }

  if (state.mesh.dim != 1) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning("Diagnostics.fleck_diag currently supports 1D_SPH only; skipping");
    }
    return;
  }

  const auto* coeffs = imc.last_cell_radiation_coeffs();
  if (coeffs == nullptr) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning(
          "Diagnostics.fleck_diag requested but no NLTE/TMAT cell-radiation coefficients are "
          "available for this step");
    }
    return;
  }

  const std::size_t n_cells = state.rho.size();
  TENRYU_ASSERT(coeffs->n_cells == static_cast<int>(n_cells),
                "fleck_diag coefficient/state cell count mismatch");
  TENRYU_ASSERT(coeffs->rho_eval.size() == n_cells,
                "fleck_diag requires rho_eval size == n_cells");
  TENRYU_ASSERT(coeffs->Te_eval.size() == n_cells,
                "fleck_diag requires Te_eval size == n_cells");
  TENRYU_ASSERT(coeffs->cv_e.size() == n_cells,
                "fleck_diag requires cv_e size == n_cells");
  TENRYU_ASSERT(coeffs->beta.size() == n_cells,
                "fleck_diag requires beta size == n_cells");
  TENRYU_ASSERT(coeffs->sigma_p_em.size() == n_cells,
                "fleck_diag requires sigma_p_em size == n_cells");
  TENRYU_ASSERT(coeffs->f.size() == n_cells,
                "fleck_diag requires f size == n_cells");
  TENRYU_ASSERT(coeffs->eta_tot.size() == n_cells,
                "fleck_diag requires eta_tot size == n_cells");
  TENRYU_ASSERT(state.rad_dep.size() % std::max<std::size_t>(n_cells, 1) == 0,
                "fleck_diag requires rad_dep size divisible by n_cells");
  TENRYU_ASSERT(state.rad_emit.empty() || state.rad_emit.size() == state.rad_dep.size(),
                "fleck_diag requires rad_emit size == rad_dep size when present");

  std::vector<char> selected(n_cells, 0);
  std::vector<int> cells;
  cells.reserve(diag.cells.size() + 8);

  int invalid_fixed = 0;
  for (const int cell : diag.cells) {
    if (cell < 0 || static_cast<std::size_t>(cell) >= n_cells) {
      ++invalid_fixed;
      continue;
    }
    if (!selected[static_cast<std::size_t>(cell)]) {
      selected[static_cast<std::size_t>(cell)] = 1;
      cells.push_back(cell);
    }
  }
  if (invalid_fixed > 0) {
    static int warn_count = 0;
    ++warn_count;
    if (warn_count == 1 || warn_count % 100 == 0) {
      core::log_warning("Diagnostics.fleck_diag ignored " + std::to_string(invalid_fixed) +
                        " out-of-range local cell indices");
    }
  }

  if (use_radius) {
    TENRYU_ASSERT(state.x_r.size() == n_cells + 1,
                  "fleck_diag requires x_r node count = n_cells + 1 in 1D");
    std::vector<double> node_r(state.x_r.size(), 0.0);
    state.x_r.copy_to_host(node_r.data());
    for (std::size_t c = 0; c < n_cells; ++c) {
      const double rc = 0.5 * (node_r[c] + node_r[c + 1]);
      if (rc < diag.r_min_cm || rc > diag.r_max_cm) {
        continue;
      }
      if (!selected[c]) {
        selected[c] = 1;
        cells.push_back(static_cast<int>(c));
      }
    }
  }

  if (cells.empty()) {
    return;
  }

  std::vector<double> rad_dep(state.rad_dep.size(), 0.0);
  std::vector<double> rad_emit(state.rad_dep.size(), 0.0);
  state.rad_dep.copy_to_host(rad_dep.data());
  if (!state.rad_emit.empty()) {
    state.rad_emit.copy_to_host(rad_emit.data());
  }
  const int n_groups =
      (n_cells > 0) ? static_cast<int>(state.rad_dep.size() / n_cells) : 0;

  for (const int cell : cells) {
    const std::size_t c = static_cast<std::size_t>(cell);
    double dep_sum = 0.0;
    double E_emit_cell = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx =
          c * static_cast<std::size_t>(n_groups) + static_cast<std::size_t>(g);
      dep_sum += rad_dep[idx];
      E_emit_cell += rad_emit[idx];
    }
    const double delta_E = dep_sum - E_emit_cell;

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(16);
    oss << "[fleck_diag] step=" << step_number
        << " cell=" << cell
        << " Te=" << coeffs->Te_eval[c]
        << " rho=" << coeffs->rho_eval[c]
        << " cv_e=" << coeffs->cv_e[c]
        << " sigma_p_em=" << coeffs->sigma_p_em[c]
        << " beta=" << coeffs->beta[c]
        << " f=" << coeffs->f[c]
        << " eta_tot=" << coeffs->eta_tot[c]
        << " E_emit=" << E_emit_cell
        << " dep_sum=" << dep_sum
        << " delta_E=" << delta_E;
    core::log_info(oss.str());
  }
}

void clear_laser_ray_output(core::State& state) {
  state.laser_ray_R0.clear();
  state.laser_ray_Z0.clear();
  state.laser_ray_vR0.clear();
  state.laser_ray_vZ0.clear();
  state.laser_ray_x0.clear();
  state.laser_ray_y0.clear();
  state.laser_ray_z0.clear();
  state.laser_ray_vx0.clear();
  state.laser_ray_vy0.clear();
  state.laser_ray_vz0.clear();
  state.laser_ray_power0.clear();
  state.laser_ray_beam_id.clear();
  state.laser_ray_is_3d = false;
}

void store_laser_ray_output(core::State& state,
                            const std::vector<laser::RayOutputData>& ray_output) {
  clear_laser_ray_output(state);
  bool has_3d = false;
  for (const auto& group : ray_output) {
    if (!group.rays_3d.empty()) {
      has_3d = true;
      break;
    }
  }
  state.laser_ray_is_3d = has_3d;
  for (const auto& group : ray_output) {
    if (has_3d) {
      for (const auto& ray : group.rays_3d) {
        state.laser_ray_x0.push_back(ray.x0);
        state.laser_ray_y0.push_back(ray.y0);
        state.laser_ray_z0.push_back(ray.z0);
        state.laser_ray_vx0.push_back(ray.vx0);
        state.laser_ray_vy0.push_back(ray.vy0);
        state.laser_ray_vz0.push_back(ray.vz0);
        state.laser_ray_power0.push_back(ray.power0);
        state.laser_ray_beam_id.push_back(group.beam_id);
      }
    } else {
      for (const auto& ray : group.rays_2d) {
        state.laser_ray_R0.push_back(ray.R0);
        state.laser_ray_Z0.push_back(ray.Z0);
        state.laser_ray_vR0.push_back(ray.vR0);
        state.laser_ray_vZ0.push_back(ray.vZ0);
        state.laser_ray_power0.push_back(ray.power0);
        state.laser_ray_beam_id.push_back(group.beam_id);
      }
    }
  }
}

void output_if_needed(core::State& state,
                      const core::Config& cfg,
                      io::OutputManager& out,
                      const radiation::IMC& imc,
                      const std::string& case_name,
                      const int rank) {
  if (out.should_plot(state.step, state.t, state, cfg)) {
    if (rank == 0) {
      out.write_snapshot(state, cfg, state.step, state.t, case_name, rank);
      out.write_run_info(state, cfg);
      core::log_info("[output] wrote snapshot (step=" + std::to_string(state.step) +
                     ", t=" + format_sci(state.t) + ")");
    }
    update_next_output_time(state.t_next_plot, cfg.output.plot_every_s, state.t);
  }

  if (out.should_history(state.step, state.t, state, cfg)) {
    update_next_output_time(state.t_next_history, cfg.output.history_every_s, state.t);
  }

  if (out.should_checkpoint(state.step, state.t, state, cfg)) {
    update_next_output_time(state.t_next_checkpoint, cfg.output.checkpoint_every_s,
                            state.t);
    if (rank == 0) {
      out.write_checkpoint(state,
                           cfg,
                           imc.photon_pool(),
                           state.step,
                           state.t,
                           case_name,
                           rank);
      core::log_info("[output] wrote checkpoint (step=" + std::to_string(state.step) +
                     ", t=" + format_sci(state.t) + ")");
    }
  }
}

void log_progress_if_needed(const core::State& state,
                            const core::Config& cfg,
                            const parallel::Reduction& reducer) {
  const bool verbose = (cfg.main.verbosity == "verbose");
  const bool normal = (cfg.main.verbosity == "normal");

  if (!(verbose || normal)) {
    return;
  }
  if (normal && (state.step % 100 != 0)) {
    return;
  }

  static const std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();
  const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
  double progress_percent = 0.0;
  if (cfg.main.t_end > 0.0) {
    progress_percent =
        std::clamp(100.0 * state.t / cfg.main.t_end, 0.0, 100.0);
  }

  std::ostringstream oss;
  oss << std::scientific << std::setprecision(3);
  oss << "[progress] step=" << state.step << "/" << cfg.main.max_steps
      << " t=" << state.t << "/" << cfg.main.t_end
      << " dt=" << state.dt
      << " (" << std::fixed << std::setprecision(1) << progress_percent << "%)"
      << " elapsed=" << format_elapsed(now - start_time);
  if (cfg.numerics.hydro.subzonal_pressure_enabled) {
    const double max_merit =
        reducer.allreduce_max(state.max_subzonal_merit_step);
    const double nonzero_force_cells = reducer.allreduce_sum(
        static_cast<double>(state.subzonal_nonzero_force_cells_step));
    const double max_abs_force =
        reducer.allreduce_max(state.max_abs_subzonal_force_step);
    const double max_abs_work =
        reducer.allreduce_max(state.max_abs_subzonal_work_step);
    oss << std::scientific << std::setprecision(3)
        << " subzonal_max_merit=" << max_merit
        << " subzonal_nonzero_force_cells="
        << static_cast<long long>(std::llround(nonzero_force_cells))
        << " subzonal_max_abs_force=" << max_abs_force
        << " subzonal_max_abs_work=" << max_abs_work;
  }
  core::log_info(oss.str());
}

bool eos_fields_are_uninitialized(const core::State& state) {
  const std::size_t n = state.rho.size();
  std::vector<double> ee(n, 0.0);
  std::vector<double> ei(n, 0.0);
  std::vector<double> Pe(n, 0.0);
  std::vector<double> Pi(n, 0.0);
  state.ee.copy_to_host(ee.data());
  state.ei.copy_to_host(ei.data());
  state.Pe.copy_to_host(Pe.data());
  state.Pi.copy_to_host(Pi.data());

  constexpr double kTol = 1.0e-30;
  for (std::size_t i = 0; i < n; ++i) {
    if (std::abs(ee[i]) > kTol || std::abs(ei[i]) > kTol || std::abs(Pe[i]) > kTol ||
        std::abs(Pi[i]) > kTol) {
      return false;
    }
  }
  return true;
}

void initialize_eos_fields_if_needed_impl(core::State& state, core::Config& cfg) {
  if (state.rho.empty() || cfg.materials.materials.empty()) {
    return;
  }
  if (cfg.main.temperature_model == "auto") {
    resolve_temperature_model(cfg, state);
  }
  if (!eos_fields_are_uninitialized(state)) {
    return;
  }

  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "initialize_eos_fields_if_needed requires at least one non-void material");
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(first_nonvoid)];
  if (mat.ideal_gas_gamma <= 1.0 || mat.A <= 0.0) {
    return;
  }

  const bool use_exact_hydro = use_exact_ideal_gas_hydro_backend(mat);

  const std::size_t n = state.rho.size();
  TENRYU_ASSERT(state.zbar.size() == n, "initialize_eos_fields_if_needed zbar/rho size mismatch");
  TENRYU_ASSERT(state.Te.size() == n, "initialize_eos_fields_if_needed Te/rho size mismatch");
  TENRYU_ASSERT(state.Ti.size() == n, "initialize_eos_fields_if_needed Ti/rho size mismatch");
  TENRYU_ASSERT(state.ee.size() == n, "initialize_eos_fields_if_needed ee/rho size mismatch");
  TENRYU_ASSERT(state.ei.size() == n, "initialize_eos_fields_if_needed ei/rho size mismatch");
  TENRYU_ASSERT(state.Pe.size() == n, "initialize_eos_fields_if_needed Pe/rho size mismatch");
  TENRYU_ASSERT(state.Pi.size() == n, "initialize_eos_fields_if_needed Pi/rho size mismatch");

  std::vector<double> rho(n, 0.0);
  std::vector<double> zbar(n, 0.0);
  std::vector<double> Te(n, 0.0);
  std::vector<double> Ti(n, 0.0);
  state.rho.copy_to_host(rho.data());
  state.zbar.copy_to_host(zbar.data());
  state.Te.copy_to_host(Te.data());
  state.Ti.copy_to_host(Ti.data());
  // Per-cell effective properties (multi-material fix): the historic scalar
  // first-nonvoid A/gamma poisoned every other material's initial energies.
  state.ensure_cell_material_props(cfg);
  std::vector<double> A_eff_h(n, 0.0);
  std::vector<double> gamma_eff_h(n, 0.0);
  state.A_eff.copy_to_host(A_eff_h.data());
  state.gamma_eff.copy_to_host(gamma_eff_h.data());

  // Preserve intentionally pressureless ICs used by verification/unit tests
  // (e.g. Noh): all thermal fields at floor with zero internal energies.
  bool all_at_floor = true;
  const double te_floor = cfg.numerics.floors.Te;
  const double ti_floor = cfg.numerics.floors.Ti;
  for (std::size_t i = 0; i < n; ++i) {
    if (Te[i] > te_floor * (1.0 + 1.0e-12) || Ti[i] > ti_floor * (1.0 + 1.0e-12)) {
      all_at_floor = false;
      break;
    }
  }
  if (all_at_floor) {
    return;
  }

  const bool use_two_temp = cfg.main.two_temperature;

  std::vector<double> ee(n, 0.0);
  std::vector<double> ei(n, 0.0);
  std::vector<double> Pe(n, 0.0);
  std::vector<double> Pi(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const double z = std::max(zbar[i], 0.0);
    const double rho_safe = std::max(rho[i], 1.0e-30);
    const double gamma = std::max(gamma_eff_h[i], 1.0 + 1.0e-12);
    const double A = std::max(A_eff_h[i], 1.0e-12);
    const double cv_i = kEvToErg / (A * kProtonMass * (gamma - 1.0));
    double cv_e = 0.0;
    if (mat.cv_e_override > 0.0) {
      cv_e = mat.cv_e_override / rho_safe;
    } else {
      // Known limitation: cv_e uses Zbar(T,rho) only; the T*dZbar/dT correction
      // from Thomas-Fermi ionization is not yet wired into this closure.
      cv_e = z * kEvToErg / (A * kProtonMass * (gamma - 1.0));
    }

    const double te = std::max(Te[i], cfg.numerics.floors.Te);
    const double ti = std::max(Ti[i], cfg.numerics.floors.Ti);
    Te[i] = te;

    if (use_two_temp) {
      Ti[i] = ti;
      if (mat.eos_tables && !use_exact_hydro) {
        ee[i] = mat.eos_tables->electron.energy(rho[i], te);
        ei[i] = mat.eos_tables->ion.energy(rho[i], ti);
        Pe[i] = mat.eos_tables->electron.pressure(rho[i], te);
        Pi[i] = mat.eos_tables->ion.pressure(rho[i], ti);
      } else {
        ee[i] = cv_e * te;
        ei[i] = cv_i * ti;
        Pe[i] = (gamma - 1.0) * rho[i] * ee[i];
        Pi[i] = (gamma - 1.0) * rho[i] * ei[i];
      }
    } else {
      Ti[i] = te;
      double e_total = 0.0;
      if (mat.eos_T_ref_eV > 0.0 && mat.cv_e_override > 0.0) {
        const double T_ref = mat.eos_T_ref_eV;
        const double T_ref3 = T_ref * T_ref * T_ref;
        const double alpha0 = mat.cv_e_override / (4.0 * T_ref3);
        const double T4 = te * te * te * te;
        e_total = alpha0 * T4 / rho_safe;
        ee[i] = e_total;
        Pe[i] = (gamma - 1.0) * rho[i] * e_total;
      } else if (mat.eos_tables && !use_exact_hydro &&
                 !mat.eos_tables->total.empty()) {
        // 1T ee-init fix (2026-07-20): the 1T branch initialized ee from the ideal-gas
        // cv even for table-EOS materials (the 2T branch above consults the
        // table). The legacy eos closure silently rewrote ee to the table
        // value on the first radiation step, while energy_authoritative
        // propagated the inconsistent init (Te collapsing to the table
        // inverse of the ideal ee) — the mechanism behind the registered
        // EA x exp_rosenbrock Marshak-feature front offset. Initialize from
        // the same 1T total table the runtime sync uses.
        e_total = mat.eos_tables->total.energy(rho[i], te);
        ee[i] = e_total;
        Pe[i] = mat.eos_tables->total.pressure(rho[i], te);
      } else {
        e_total = (cv_i + cv_e) * te;
        ee[i] = e_total;
        Pe[i] = (gamma - 1.0) * rho[i] * e_total;
      }
      ei[i] = 0.0;
      Pi[i] = 0.0;
    }
  }

  state.Te.copy_from_host(Te.data());
  state.Ti.copy_from_host(Ti.data());
  state.ee.copy_from_host(ee.data());
  state.ei.copy_from_host(ei.data());
  state.Pe.copy_from_host(Pe.data());
  state.Pi.copy_from_host(Pi.data());
}

static void sync_ee_from_Te_table(core::State& state, const core::Config& cfg) {
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  if (first_nonvoid < 0) {
    return;
  }
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(first_nonvoid)];
  const bool use_exact_hydro = use_exact_ideal_gas_hydro_backend(mat);
  if (!mat.eos_tables && !use_exact_hydro) {
    return;
  }

  const std::size_t n = state.rho.size();
  std::vector<double> rho(n, 0.0);
  std::vector<double> zbar(n, 0.0);
  std::vector<double> Te(n, 0.0);
  std::vector<double> Ti(n, 0.0);
  std::vector<double> ee(n, 0.0);
  std::vector<double> ei(n, 0.0);
  std::vector<double> Pe(n, 0.0);
  std::vector<double> Pi(n, 0.0);
  std::vector<double> cv_e(n, 0.0);
  std::vector<double> cv_i(n, 0.0);
  state.rho.copy_to_host(rho.data());
  state.zbar.copy_to_host(zbar.data());
  state.Te.copy_to_host(Te.data());
  state.Ti.copy_to_host(Ti.data());

  const double te_floor = cfg.numerics.floors.Te;
  const double ti_floor = cfg.numerics.floors.Ti;
  const bool two_temp = cfg.main.two_temperature;
  const bool energy_authoritative =
      (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
  // Host-side high-T ideal tail (mirror of the device helpers):
  // above the table's temperature ceiling, extend instead of saturating.
  struct HostTailThermo {
    double e;
    double P;
    double cv;
  };
  const auto eval_with_tail = [&](const auto& tab, const double rho_s,
                                  const double T_eV) -> HostTailThermo {
    HostTailThermo out{};
    if (energy_authoritative && !tab.T_grid_eV.empty()) {
      const double T_top = tab.T_grid_eV.back();
      if (std::isfinite(T_top) && T_top > 0.0 && T_eV > T_top) {
        const double e_top = tab.energy(rho_s, T_top);
        const double P_top = tab.pressure(rho_s, T_top);
        const double cv_top = std::max(tab.cv(rho_s, T_top), 0.0);
        if (std::isfinite(e_top) && std::isfinite(P_top) &&
            std::isfinite(cv_top) && cv_top > 0.0) {
          out.e = e_top + cv_top * (T_eV - T_top);
          out.P = P_top * (T_eV / T_top);
          out.cv = cv_top;
          return out;
        }
      }
    }
    out.e = tab.energy(rho_s, T_eV);
    out.P = tab.pressure(rho_s, T_eV);
    out.cv = std::max(tab.cv(rho_s, T_eV), 0.0);
    return out;
  };
  const double gamma = mat.ideal_gas_gamma;
  const double A = mat.A;
  const double cv_i_mass = kEvToErg / (A * kProtonMass * (gamma - 1.0));

  for (std::size_t i = 0; i < n; ++i) {
    if (state.cell_is_void.size() > i && state.cell_is_void[i] != 0U) {
      continue;
    }
    const double rho_safe = std::max(rho[i], 1.0e-30);
    const double te = std::max(Te[i], te_floor);
    const double z = std::max(zbar[i], 0.0);
    const double cv_e_mass =
        (mat.cv_e_override > 0.0)
            ? (mat.cv_e_override / rho_safe)
            : (z * kEvToErg / (A * kProtonMass * (gamma - 1.0)));
    const double cv_e_cell = std::max(cv_e_mass, 0.0);
    const double cv_i_cell = std::max(cv_i_mass, 0.0);
    cv_e[i] = two_temp ? cv_e_cell : (cv_e_cell + cv_i_cell);
    cv_i[i] = two_temp ? cv_i_cell : 0.0;

    if (use_exact_hydro) {
      if (two_temp) {
        const double ti = std::max(Ti[i], ti_floor);
        ee[i] = cv_e[i] * te;
        ei[i] = cv_i[i] * ti;
        Pe[i] = (gamma - 1.0) * rho[i] * ee[i];
        Pi[i] = (gamma - 1.0) * rho[i] * ei[i];
      } else {
        const double cv_total = cv_e[i] + cv_i[i];
        double e_total = 0.0;
        if (mat.eos_T_ref_eV > 0.0 && mat.cv_e_override > 0.0) {
          const double T_ref = mat.eos_T_ref_eV;
          const double T_ref3 = T_ref * T_ref * T_ref;
          const double alpha0 = mat.cv_e_override / (4.0 * T_ref3);
          const double T4 = te * te * te * te;
          e_total = alpha0 * T4 / rho_safe;
        } else {
          e_total = cv_total * te;
        }
        ee[i] = e_total;
        ei[i] = 0.0;
        Pe[i] = (gamma - 1.0) * rho[i] * e_total;
        Pi[i] = 0.0;
      }
      continue;
    }

    if (!two_temp && !mat.eos_tables->total.empty()) {
      const HostTailThermo th = eval_with_tail(mat.eos_tables->total, rho_safe, te);
      ee[i] = th.e;
      Pe[i] = th.P;
      cv_e[i] = th.cv;
      cv_i[i] = 0.0;
      continue;
    }
    {
      const HostTailThermo th_e =
          eval_with_tail(mat.eos_tables->electron, rho_safe, te);
      ee[i] = th_e.e;
      Pe[i] = th_e.P;
      cv_e[i] = th_e.cv;
    }
    cv_i[i] = 0.0;
    if (two_temp) {
      const double ti = std::max(Ti[i], ti_floor);
      const HostTailThermo th_i =
          eval_with_tail(mat.eos_tables->ion, rho_safe, ti);
      ei[i] = th_i.e;
      Pi[i] = th_i.P;
      cv_i[i] = th_i.cv;
    }
  }

  state.ee.copy_from_host(ee.data());
  state.Pe.copy_from_host(Pe.data());
  if (!state.cv_e.empty()) {
    state.cv_e.copy_from_host(cv_e.data());
  }
  if (!state.cv_i.empty()) {
    state.cv_i.copy_from_host(cv_i.data());
  }
  if (two_temp) {
    state.ei.copy_from_host(ei.data());
    state.Pi.copy_from_host(Pi.data());
  }
}

// Energy-authoritative conservative conduction handoff — book the
// solve's own cv-linear energy increment exactly, then re-derive Te/Pe/cv
// from the updated energy by tail-aware inversion of the table.
static void apply_conduction_energy_increment(
    core::State& state,
    const core::Config& cfg,
    const std::vector<double>& Te_old,
    const std::vector<double>& cv_old) {
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  if (first_nonvoid < 0) {
    sync_ee_from_Te_table(state, cfg);
    return;
  }
  const auto& mat =
      cfg.materials.materials[static_cast<std::size_t>(first_nonvoid)];
  if (!mat.eos_tables) {
    sync_ee_from_Te_table(state, cfg);
    return;
  }
  const auto& tab = cfg.main.two_temperature ? mat.eos_tables->electron
                                             : mat.eos_tables->total;
  if (tab.empty() || tab.T_grid_eV.empty()) {
    sync_ee_from_Te_table(state, cfg);
    return;
  }

  const std::size_t n = state.rho.size();
  if (Te_old.size() != n) {
    sync_ee_from_Te_table(state, cfg);
    return;
  }
  std::vector<double> rho(n, 0.0);
  std::vector<double> Te(n, 0.0);
  std::vector<double> ee(n, 0.0);
  std::vector<double> Pe(n, 0.0);
  std::vector<double> cv_e(n, 0.0);
  {
    std::vector<double> cond_pack(3 * n);
    const double* srcs[3] = {state.rho.data(), state.Te.data(),
                             state.ee.data()};
    core::pack_pull_fields(srcs, 3, static_cast<int>(n), cond_pack.data(),
                           "driver:apply_cond_incr:pull");
    std::memcpy(rho.data(), cond_pack.data(), n * sizeof(double));
    std::memcpy(Te.data(), cond_pack.data() + n, n * sizeof(double));
    std::memcpy(ee.data(), cond_pack.data() + 2 * n, n * sizeof(double));
  }

  const double te_floor = cfg.numerics.floors.Te;
  const double T_top = tab.T_grid_eV.back();

  for (std::size_t i = 0; i < n; ++i) {
    if (state.cell_is_void.size() > i && state.cell_is_void[i] != 0U) {
      continue;
    }
    const double rho_safe = std::max(rho[i], 1.0e-30);
    const double cv_used = (i < cv_old.size() && cv_old[i] > 0.0)
                               ? cv_old[i]
                               : std::max(tab.cv(rho_safe, std::max(Te_old[i],
                                                                    te_floor)),
                                          0.0);
    // Exact booking of the solve's energy motion in its own metric.
    ee[i] = std::max(ee[i] + cv_used * (Te[i] - Te_old[i]), 0.0);

    // Tail-aware inversion: T from the updated energy.
    double T_inv = tab.temperature_from_energy(rho_safe, ee[i]);
    bool in_tail = false;
    if (std::isfinite(T_top) && T_top > 0.0) {
      const double e_top = tab.energy(rho_safe, T_top);
      const double cv_top = std::max(tab.cv(rho_safe, T_top), 0.0);
      if (std::isfinite(e_top) && cv_top > 0.0 && ee[i] > e_top) {
        T_inv = T_top + (ee[i] - e_top) / cv_top;
        in_tail = true;
      }
    }
    if (!std::isfinite(T_inv) || T_inv < te_floor) {
      T_inv = te_floor;
    }
    Te[i] = T_inv;
    if (in_tail) {
      const double P_top = tab.pressure(rho_safe, T_top);
      const double cv_top = std::max(tab.cv(rho_safe, T_top), 0.0);
      Pe[i] = P_top * (T_inv / T_top);
      cv_e[i] = cv_top;
    } else {
      Pe[i] = tab.pressure(rho_safe, T_inv);
      cv_e[i] = std::max(tab.cv(rho_safe, T_inv), 0.0);
    }
  }

  const int n_push = state.cv_e.empty() ? 3 : 4;
  std::vector<double> cond_pack(static_cast<std::size_t>(n_push) * n);
  std::memcpy(cond_pack.data(), ee.data(), n * sizeof(double));
  std::memcpy(cond_pack.data() + n, Te.data(), n * sizeof(double));
  std::memcpy(cond_pack.data() + 2 * n, Pe.data(), n * sizeof(double));
  double* dsts[4] = {state.ee.data(), state.Te.data(), state.Pe.data(), nullptr};
  if (n_push == 4) {
    std::memcpy(cond_pack.data() + 3 * n, cv_e.data(), n * sizeof(double));
    dsts[3] = state.cv_e.data();
  }
  core::pack_push_fields(cond_pack.data(), n_push, static_cast<int>(n), dsts,
                         "driver:apply_cond_incr:push");
}

static void refresh_mie_gruneisen_thermo_from_energy(core::State& state,
                                                     const core::Config& cfg,
                                                     const hydro::HydroEOSContext& eos_ctx,
                                                     DriverRecloseContext& reclose_ctx) {
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  if (first_nonvoid < 0) {
    return;
  }
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(first_nonvoid)];
  if (!mat.eos_tables || !use_mie_gruneisen_hydro_backend(mat) || !cfg.main.two_temperature) {
    return;
  }

  if (state.mesh.dim == 1) {
    const materials::DeviceEOSTableView electron_view =
        eos_ctx.electron_view(first_nonvoid);
    const materials::DeviceEOSTableView ion_view =
        eos_ctx.ion_view(first_nonvoid);
    if (electron_view.n_rho > 0 && ion_view.n_rho > 0) {
      launch_tabular_eos_reclose(
          reclose_ctx,
          electron_view,
          ion_view,
          state.cell_is_void,
          state.rho.size(),
          state.rho.data(),
          state.ee.data(),
          state.ei.data(),
          state.Te.data(),
          state.Ti.data(),
          state.Pe.data(),
          state.Pi.data(),
          state.cv_e.empty() ? nullptr : state.cv_e.data(),
          state.cv_i.empty() ? nullptr : state.cv_i.data(),
          cfg.numerics.floors.Te,
          cfg.numerics.floors.Ti);
      return;
    }
  }

  const std::size_t n = state.rho.size();
  std::vector<double> rho(n, 0.0);
  std::vector<double> ee(n, 0.0);
  std::vector<double> ei(n, 0.0);
  std::vector<double> Te(n, 0.0);
  std::vector<double> Ti(n, 0.0);
  std::vector<double> Pe(n, 0.0);
  std::vector<double> Pi(n, 0.0);
  std::vector<double> cv_e(n, 0.0);
  std::vector<double> cv_i(n, 0.0);

  state.rho.copy_to_host(rho.data());
  state.ee.copy_to_host(ee.data());
  state.ei.copy_to_host(ei.data());
  state.Te.copy_to_host(Te.data());
  state.Ti.copy_to_host(Ti.data());
  state.Pe.copy_to_host(Pe.data());
  state.Pi.copy_to_host(Pi.data());

  const double te_floor = cfg.numerics.floors.Te;
  const double ti_floor = cfg.numerics.floors.Ti;
  for (std::size_t i = 0; i < n; ++i) {
    if (state.cell_is_void.size() > i && state.cell_is_void[i] != 0U) {
      continue;
    }
    const double rho_safe = std::max(rho[i], 1.0e-30);

    const double te_raw =
        mat.eos_tables->electron.temperature_from_energy(rho_safe, std::max(ee[i], 0.0));
    const double te =
        (std::isfinite(te_raw) && te_raw >= te_floor) ? te_raw : te_floor;
    Te[i] = te;
    ee[i] = mat.eos_tables->electron.energy(rho_safe, te);
    Pe[i] = mat.eos_tables->electron.pressure(rho_safe, te);
    cv_e[i] = std::max(mat.eos_tables->electron.cv(rho_safe, te), 0.0);

    const double ti_raw =
        mat.eos_tables->ion.temperature_from_energy(rho_safe, std::max(ei[i], 0.0));
    const double ti =
        (std::isfinite(ti_raw) && ti_raw >= ti_floor) ? ti_raw : ti_floor;
    Ti[i] = ti;
    ei[i] = mat.eos_tables->ion.energy(rho_safe, ti);
    Pi[i] = mat.eos_tables->ion.pressure(rho_safe, ti);
    cv_i[i] = std::max(mat.eos_tables->ion.cv(rho_safe, ti), 0.0);
  }

  state.ee.copy_from_host(ee.data());
  state.ei.copy_from_host(ei.data());
  state.Te.copy_from_host(Te.data());
  state.Ti.copy_from_host(Ti.data());
  state.Pe.copy_from_host(Pe.data());
  state.Pi.copy_from_host(Pi.data());
  if (!state.cv_e.empty()) {
    state.cv_e.copy_from_host(cv_e.data());
  }
  if (!state.cv_i.empty()) {
    state.cv_i.copy_from_host(cv_i.data());
  }
}

template <typename Tag>
double sum_field(const core::Field1D<Tag>& field) {
  if (field.empty()) {
    return 0.0;
  }
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  long double sum = 0.0L;
  for (const double value : host) {
    sum += static_cast<long double>(value);
  }
  return static_cast<double>(sum);
}

template <typename Tag>
std::vector<double> copy_field_to_host(const core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  if (!host.empty()) {
    field.copy_to_host(host.data());
  }
  return host;
}

struct I1OperatorProfileRow {
  int step = 0;
  int rank = 0;
  int j = 0;
  int remap_applied = 0;
  std::string hydro_half;
  std::string phase;
  double t_s = 0.0;
  double dt_op_s = 0.0;
  double z_cm = 0.0;
  double vol_sum_cm3 = 0.0;
  double rho_g_cm3 = 0.0;
  double Te_eV = 0.0;
  double u_z_cm_s = 0.0;
  double Pe_dyne_cm2 = 0.0;
  double Pi_dyne_cm2 = 0.0;
  double Qvisc_dyne_cm2 = 0.0;
  double E_rad_erg_cm3 = 0.0;
  double rho_u_z_g_cm2_s = 0.0;
};

struct I1OperatorProfileDump {
  bool enabled = false;
  int every_n_steps = 0;
  std::filesystem::path path;
  std::vector<I1OperatorProfileRow> rows;
};

bool env_flag_raw_enabled(const char* raw) {
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(),
                 [](const unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value != "0" && value != "false" && value != "off" &&
         value != "no";
}

bool env_flag_enabled(const char* name) {
  return env_flag_raw_enabled(std::getenv(name));
}

int env_nonnegative_int(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' || value < 0 ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(value);
}

double env_positive_double(const char* name, const double fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const double value = std::strtod(raw, &end);
  if (end == raw || *end != '\0' || !(value > 0.0) ||
      !std::isfinite(value)) {
    return fallback;
  }
  return value;
}

double i1b_fixed_dt() {
  return env_positive_double("TENRYU_I1B_FIXED_DT", 0.0);
}

std::string rank_suffix(const int rank) {
  std::ostringstream oss;
  oss << "rank" << std::setw(4) << std::setfill('0') << rank;
  return oss.str();
}

I1OperatorProfileDump make_i1_operator_profile_dump(
    const core::Config& cfg,
    const std::string& output_dir,
    const int rank) {
  I1OperatorProfileDump dump;
  dump.enabled = env_flag_enabled("TENRYU_I1_2D_RZ_OPERATOR_PROFILE_DUMP") &&
                 rank == 0 && cfg.main.dimension == "2D_RZ";
  if (!dump.enabled) {
    return dump;
  }
  dump.every_n_steps = env_nonnegative_int(
      "TENRYU_I1_2D_RZ_OPERATOR_PROFILE_DUMP_EVERY_N_STEPS", 0);
  dump.path = std::filesystem::path(output_dir) /
              ("operator_profiles_i1_2d_rz_" + rank_suffix(rank) + ".csv");
  return dump;
}

bool i1_operator_profile_dump_step_matches(
    const I1OperatorProfileDump& dump,
    const core::Config& cfg,
    const core::State& state,
    const int next_step,
    const double step_end_t) {
  if (!dump.enabled || next_step <= 0) {
    return false;
  }
  if (dump.every_n_steps > 0) {
    return (next_step % dump.every_n_steps) == 0;
  }
  const bool by_step =
      (cfg.output.plot_every > 0) && (next_step % cfg.output.plot_every == 0);
  bool by_time = false;
  if (cfg.output.plot_every_s > 0.0 && state.t_next_plot >= 0.0) {
    const double eps =
        1.0e-14 * std::max(std::abs(step_end_t), cfg.output.plot_every_s);
    by_time = step_end_t >= (state.t_next_plot - eps);
  }
  return by_step || by_time;
}

double finite_or_zero_host(const double value) {
  return std::isfinite(value) ? value : 0.0;
}

double field_value_or_zero(const std::vector<double>& values,
                           const std::size_t idx) {
  return idx < values.size() ? finite_or_zero_host(values[idx]) : 0.0;
}

double cell_center_node_average(const std::vector<double>& node_values,
                                const int i,
                                const int j,
                                const int nz) {
  const std::size_t n00 = static_cast<std::size_t>(i * (nz + 1) + j);
  const std::size_t n10 = static_cast<std::size_t>((i + 1) * (nz + 1) + j);
  const std::size_t n11 =
      static_cast<std::size_t>((i + 1) * (nz + 1) + j + 1);
  const std::size_t n01 = static_cast<std::size_t>(i * (nz + 1) + j + 1);
  return 0.25 * (field_value_or_zero(node_values, n00) +
                 field_value_or_zero(node_values, n10) +
                 field_value_or_zero(node_values, n11) +
                 field_value_or_zero(node_values, n01));
}

void maybe_capture_i1_operator_profile(
    I1OperatorProfileDump& dump,
    const core::State& state,
    const core::Config& cfg,
    const int step,
    const int rank,
    const char* hydro_half,
    const char* phase,
    const double t_s,
    const double dt_op_s,
    const bool remap_applied,
    const bool dump_this_step) {
  if (!dump.enabled || !dump_this_step || cfg.main.dimension != "2D_RZ" ||
      state.mesh.dim != 2) {
    return;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  if (nr <= 0 || nz <= 0 || n_cells <= 0 ||
      state.rho.size() != static_cast<std::size_t>(n_cells) ||
      state.vol.size() != static_cast<std::size_t>(n_cells)) {
    return;
  }

  const auto rho = copy_field_to_host(state.rho);
  const auto vol = copy_field_to_host(state.vol);
  const auto Te = copy_field_to_host(state.Te);
  const auto Pe = copy_field_to_host(state.Pe);
  const auto Pi = copy_field_to_host(state.Pi);
  const auto Qvisc = copy_field_to_host(state.Qvisc);
  const auto v_z = copy_field_to_host(state.v_z);
  const auto x_z = copy_field_to_host(state.x_z);
  const auto rad_E = copy_field_to_host(state.rad_E);

  int n_groups = std::max(cfg.radiation.groups, 1);
  const std::size_t expected_rad_size =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  if (rad_E.size() != expected_rad_size) {
    n_groups = (n_cells > 0 &&
                rad_E.size() % static_cast<std::size_t>(n_cells) == 0)
                   ? static_cast<int>(rad_E.size() /
                                      static_cast<std::size_t>(n_cells))
                   : 0;
  }

  const std::size_t row0 = dump.rows.size();
  dump.rows.resize(row0 + static_cast<std::size_t>(nz));
  for (int j = 0; j < nz; ++j) {
    long double wsum = 0.0L;
    long double z_sum = 0.0L;
    long double rho_sum = 0.0L;
    long double Te_sum = 0.0L;
    long double uz_sum = 0.0L;
    long double Pe_sum = 0.0L;
    long double Pi_sum = 0.0L;
    long double Qvisc_sum = 0.0L;
    long double Erad_sum = 0.0L;
    long double rho_uz_sum = 0.0L;
    for (int i = 0; i < nr; ++i) {
      const int c = i * nz + j;
      const std::size_t idx = static_cast<std::size_t>(c);
      const double V = field_value_or_zero(vol, idx);
      if (!(V > 0.0)) {
        continue;
      }
      const double rho_c = field_value_or_zero(rho, idx);
      const double uz_c = cell_center_node_average(v_z, i, j, nz);
      double Erad_c = 0.0;
      if (n_groups > 0) {
        const std::size_t base =
            idx * static_cast<std::size_t>(n_groups);
        for (int g = 0; g < n_groups; ++g) {
          Erad_c += field_value_or_zero(rad_E,
                                        base + static_cast<std::size_t>(g));
        }
      }
      const long double w = static_cast<long double>(V);
      wsum += w;
      z_sum += w * static_cast<long double>(
                       cell_center_node_average(x_z, i, j, nz));
      rho_sum += w * static_cast<long double>(rho_c);
      Te_sum += w * static_cast<long double>(field_value_or_zero(Te, idx));
      uz_sum += w * static_cast<long double>(uz_c);
      Pe_sum += w * static_cast<long double>(field_value_or_zero(Pe, idx));
      Pi_sum += w * static_cast<long double>(field_value_or_zero(Pi, idx));
      Qvisc_sum +=
          w * static_cast<long double>(field_value_or_zero(Qvisc, idx));
      Erad_sum += w * static_cast<long double>(Erad_c);
      rho_uz_sum += w * static_cast<long double>(rho_c) *
                    static_cast<long double>(uz_c);
    }

    I1OperatorProfileRow& row = dump.rows[row0 + static_cast<std::size_t>(j)];
    row.step = step;
    row.rank = rank;
    row.j = j;
    row.remap_applied = remap_applied ? 1 : 0;
    row.hydro_half = hydro_half != nullptr ? hydro_half : "unknown";
    row.phase = phase != nullptr ? phase : "unknown";
    row.t_s = t_s;
    row.dt_op_s = dt_op_s;
    row.vol_sum_cm3 = static_cast<double>(wsum);
    if (wsum > 0.0L) {
      row.z_cm = static_cast<double>(z_sum / wsum);
      row.rho_g_cm3 = static_cast<double>(rho_sum / wsum);
      row.Te_eV = static_cast<double>(Te_sum / wsum);
      row.u_z_cm_s = static_cast<double>(uz_sum / wsum);
      row.Pe_dyne_cm2 = static_cast<double>(Pe_sum / wsum);
      row.Pi_dyne_cm2 = static_cast<double>(Pi_sum / wsum);
      row.Qvisc_dyne_cm2 = static_cast<double>(Qvisc_sum / wsum);
      row.E_rad_erg_cm3 = static_cast<double>(Erad_sum / wsum);
      row.rho_u_z_g_cm2_s = static_cast<double>(rho_uz_sum / wsum);
    }
  }
}

void clear_i1_operator_profile_attempt(I1OperatorProfileDump& dump) {
  if (dump.enabled) {
    dump.rows.clear();
  }
}

void flush_i1_operator_profile_dump(I1OperatorProfileDump& dump) {
  if (!dump.enabled || dump.rows.empty()) {
    return;
  }
  std::filesystem::create_directories(dump.path.parent_path());
  std::error_code ec;
  const bool need_header =
      !std::filesystem::exists(dump.path, ec) ||
      std::filesystem::file_size(dump.path, ec) == 0U;
  std::ofstream os(dump.path, std::ios::binary | std::ios::app);
  TENRYU_ASSERT(os.good(), "Failed to open I1 operator profile CSV");
  if (need_header) {
    os << "step,rank,hydro_half,phase,t_s,dt_op_s,remap_applied,j,z_cm,"
          "vol_sum_cm3,rho_g_cm3,Te_eV,u_z_cm_s,Pe_dyne_cm2,Pi_dyne_cm2,"
          "Qvisc_dyne_cm2,E_rad_erg_cm3,rho_u_z_g_cm2_s\n";
  }
  os << std::scientific << std::setprecision(17);
  for (const I1OperatorProfileRow& row : dump.rows) {
    os << row.step << ',' << row.rank << ',' << row.hydro_half << ','
       << row.phase << ',' << row.t_s << ',' << row.dt_op_s << ','
       << row.remap_applied << ',' << row.j << ',' << row.z_cm << ','
       << row.vol_sum_cm3 << ',' << row.rho_g_cm3 << ',' << row.Te_eV << ','
       << row.u_z_cm_s << ',' << row.Pe_dyne_cm2 << ',' << row.Pi_dyne_cm2
       << ',' << row.Qvisc_dyne_cm2 << ',' << row.E_rad_erg_cm3 << ','
       << row.rho_u_z_g_cm2_s << '\n';
  }
  TENRYU_ASSERT(os.good(), "Failed while writing I1 operator profile CSV");
  os.flush();
  TENRYU_ASSERT(os.good(), "Failed to flush I1 operator profile CSV");
  dump.rows.clear();
}

double compute_holo_E_LO_total(const core::State& state, const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  if (!state.holo_core_mask_valid || state.holo_core_mask.size() != n_cells ||
      state.holo_E_LO.size() != n_cell_groups || state.vol.size() != n_cells) {
    return 0.0;
  }

  const auto E_LO = copy_field_to_host(state.holo_E_LO);
  const auto vol = copy_field_to_host(state.vol);
  long double total = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const double vol_c = vol[c];
    if (!std::isfinite(vol_c) || vol_c <= 0.0) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const double E = E_LO[c * n_groups + g];
      if (std::isfinite(E)) {
        total += static_cast<long double>(E) * static_cast<long double>(vol_c);
      }
    }
  }
  return static_cast<double>(total);
}

double compute_holo_particle_net_source_core(const core::State& state,
                                             const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  if (!state.holo_core_mask_valid || state.holo_core_mask.size() != n_cells ||
      state.rad_dep.size() != n_cell_groups || state.rad_emit.size() != n_cell_groups) {
    return 0.0;
  }

  const auto rad_dep = copy_field_to_host(state.rad_dep);
  const auto rad_emit = copy_field_to_host(state.rad_emit);
  long double total = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (state.holo_core_mask[c] == 0U) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const std::size_t key = c * n_groups + g;
      const double dep = rad_dep[key];
      const double emit = rad_emit[key];
      if (std::isfinite(dep) && std::isfinite(emit)) {
        total += static_cast<long double>(dep - emit);
      }
    }
  }
  return static_cast<double>(total);
}

struct HoloPrrSummary {
  double coverage = 0.0;
  double chi_min = 0.0;
  double chi_mean = 0.0;
  double chi_max = 0.0;
};

HoloPrrSummary compute_holo_prr_summary(const core::State& state,
                                        const core::Config& cfg) {
  HoloPrrSummary out{};
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  if (!state.holo_core_mask_valid || state.holo_core_mask.size() != n_cells ||
      state.holo_chi.size() != n_cell_groups ||
      state.holo_Prr_coverage.size() != n_cell_groups) {
    return out;
  }

  const auto chi = copy_field_to_host(state.holo_chi);
  const auto coverage = copy_field_to_host(state.holo_Prr_coverage);
  long double coverage_sum = 0.0L;
  long double chi_sum = 0.0L;
  double chi_min = std::numeric_limits<double>::infinity();
  double chi_max = -std::numeric_limits<double>::infinity();
  std::size_t count = 0;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (state.holo_core_mask[c] == 0U) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const std::size_t key = c * n_groups + g;
      const double cov = coverage[key];
      const double chi_value = chi[key];
      if (!std::isfinite(cov) || !std::isfinite(chi_value)) {
        continue;
      }
      coverage_sum += static_cast<long double>(std::clamp(cov, 0.0, 1.0));
      chi_sum += static_cast<long double>(chi_value);
      chi_min = std::min(chi_min, chi_value);
      chi_max = std::max(chi_max, chi_value);
      ++count;
    }
  }
  if (count > 0U) {
    const double inv = 1.0 / static_cast<double>(count);
    out.coverage = static_cast<double>(coverage_sum) * inv;
    out.chi_min = chi_min;
    out.chi_mean = static_cast<double>(chi_sum) * inv;
    out.chi_max = chi_max;
  }
  return out;
}

double max_finite_field_value(const core::CellField1D& field) {
  const auto host = copy_field_to_host(field);
  bool has_finite = false;
  double max_value = 0.0;
  for (const double value : host) {
    if (!std::isfinite(value)) {
      continue;
    }
    if (!has_finite) {
      max_value = value;
      has_finite = true;
    } else {
      max_value = std::max(max_value, value);
    }
  }
  return has_finite ? max_value : 0.0;
}

struct TemperatureAudit {
  double max_te = 0.0;
  bool te_has_non_finite = false;
  bool ti_has_non_finite = false;
  bool rho_has_non_finite = false;
  bool ee_has_non_finite = false;
  bool ei_has_non_finite = false;
};

struct OvershootMetrics {
  std::int64_t count = 0;
  double max_ratio = 0.0;
};

double marshak_boundary_temperature(const core::State& state,
                                    const core::Config& cfg,
                                    const double eval_time) {
  if (!cfg.radiation.enabled) {
    return 0.0;
  }
  const double default_tr = std::max(cfg.radiation.boundary.marshak_Tr_eV, 0.0);
  if (cfg.main.dimension == "1D_SPH") {
    // Diagnostics baseline only (overshoot metrics history): accept the FLD
    // nesting so mgd-marshak decks report overshoot against Tr instead of 0.
    // No solve-path consumer — the FLD influx reads marshak_Tr_eV/marshak_Tr_1d
    // directly in fld_1d_gpu.cu.
    const bool outer_is_marshak =
        cfg.radiation.boundary.outer_r == "marshak" ||
        (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion &&
         cfg.radiation.multigroup_diffusion.boundary.outer_r == "marshak");
    if (!outer_is_marshak) {
      return 0.0;
    }
    if (state.marshak_Tr_1d.has_value()) {
      return std::max(state.marshak_Tr_1d->eval(eval_time), 0.0);
    }
    return default_tr;
  }

  double t_boundary = 0.0;
  auto eval_face_temperature = [&](const std::string& key, const std::string& alias) {
    double t_face = default_tr;
    const auto it_primary = state.marshak_Tr_face_tables.find(key);
    if (it_primary != state.marshak_Tr_face_tables.end()) {
      t_face = it_primary->second.eval(eval_time);
    } else {
      const auto it_alias = state.marshak_Tr_face_tables.find(alias);
      if (it_alias != state.marshak_Tr_face_tables.end()) {
        t_face = it_alias->second.eval(eval_time);
      }
    }
    t_boundary = std::max(t_boundary, std::max(t_face, 0.0));
  };

  if (cfg.radiation.boundary.bottom_z == "marshak") {
    eval_face_temperature("bottom_z", "z_bottom");
  }
  if (cfg.radiation.boundary.top_z == "marshak") {
    eval_face_temperature("top_z", "z_top");
  }
  return t_boundary;
}

OvershootMetrics compute_overshoot_metrics(const core::State& state,
                                           const core::Config& cfg,
                                           const double te_max_before,
                                           const double eval_time) {
  OvershootMetrics out{};
  if (state.Te.empty()) {
    return out;
  }

  const double t_boundary = marshak_boundary_temperature(state, cfg, eval_time);
  const double t_max_n = std::max(te_max_before, t_boundary);
  const double denom = std::max(t_max_n, 1.0e-30);
  const OvershootDeviceMetrics device_metrics =
      compute_overshoot_metrics_device(state.Te.data(),
                                       static_cast<int>(state.Te.size()),
                                       t_max_n,
                                       denom);
  out.count = device_metrics.count;
  out.max_ratio = device_metrics.max_ratio;
  return out;
}

bool phase_safety_violation(const DeviceTemperatureAudit& device_audit,
                            const core::Config& cfg,
                            const double te_max_before) {
  const TemperatureAudit audit{
      device_audit.max_te,
      device_audit.te_has_non_finite,
      device_audit.ti_has_non_finite,
      device_audit.rho_has_non_finite,
      device_audit.ee_has_non_finite,
      device_audit.ei_has_non_finite,
  };
  if (cfg.numerics.safety.nan_fatal &&
      (audit.te_has_non_finite || audit.ti_has_non_finite ||
       audit.rho_has_non_finite || audit.ee_has_non_finite ||
       audit.ei_has_non_finite)) {
    return true;
  }
  if (cfg.numerics.safety.overshoot_fatal_enabled) {
    const double T_max_n = std::max(te_max_before, cfg.numerics.floors.Te);
    const double denom = std::max(T_max_n, 1.0e-30);
    const double overshoot_max =
        std::max((audit.max_te - T_max_n) / denom, 0.0);
    return overshoot_max > cfg.numerics.safety.overshoot_fatal;
  }
  return false;
}

void enforce_phase_safety_from_audit(
    const char* phase_name,
    const DeviceTemperatureAudit& device_audit,
    const core::Config& cfg,
    const double te_max_before,
    const int step_id,
    const double t_eval,
    const double dt_eval) {
  const TemperatureAudit audit{
      device_audit.max_te,
      device_audit.te_has_non_finite,
      device_audit.ti_has_non_finite,
      device_audit.rho_has_non_finite,
      device_audit.ee_has_non_finite,
      device_audit.ei_has_non_finite,
  };
  if (cfg.numerics.safety.nan_fatal &&
      (audit.te_has_non_finite || audit.ti_has_non_finite ||
       audit.rho_has_non_finite || audit.ee_has_non_finite ||
       audit.ei_has_non_finite)) {
    std::ostringstream oss;
    oss << "[SAFETY] step=" << step_id
        << " t=" << t_eval
        << " dt=" << dt_eval
        << ": Non-finite state field detected after phase=" << phase_name
        << " (nan_fatal=true)"
        << ", Te_non_finite=" << (audit.te_has_non_finite ? 1 : 0)
        << ", Ti_non_finite=" << (audit.ti_has_non_finite ? 1 : 0)
        << ", rho_non_finite=" << (audit.rho_has_non_finite ? 1 : 0)
        << ", ee_non_finite=" << (audit.ee_has_non_finite ? 1 : 0)
        << ", ei_non_finite=" << (audit.ei_has_non_finite ? 1 : 0);
    core::log_fatal(oss.str());
    TENRYU_ASSERT(false, oss.str());
  }

  if (cfg.numerics.safety.overshoot_fatal_enabled) {
    const double T_max_n = std::max(te_max_before, cfg.numerics.floors.Te);
    const double denom = std::max(T_max_n, 1.0e-30);
    const double overshoot_max =
        std::max((audit.max_te - T_max_n) / denom, 0.0);
    if (overshoot_max > cfg.numerics.safety.overshoot_fatal) {
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "[SAFETY] step=" << step_id
          << " t=" << t_eval
          << " dt=" << dt_eval
          << ": Temperature overshoot fatal threshold exceeded after phase=" << phase_name
          << ": overshoot_max=" << overshoot_max
          << " > safety.overshoot_fatal=" << cfg.numerics.safety.overshoot_fatal
          << ", T_max_n=" << T_max_n
          << ", T_max_np1=" << audit.max_te;
      core::log_fatal(oss.str());
      TENRYU_ASSERT(false, oss.str());
    }
  }
}

void enforce_phase_safety(const char* phase_name,
                          const core::State& state,
                          const core::Config& cfg,
                          const double te_max_before,
                          const int step_id,
                          const double t_eval,
                          const double dt_eval) {
  if (!cfg.numerics.safety.nan_fatal &&
      !cfg.numerics.safety.overshoot_fatal_enabled) {
    return;
  }
  enforce_phase_safety_from_audit(phase_name,
                                  compute_temperature_audit_device(state),
                                  cfg,
                                  te_max_before,
                                  step_id,
                                  t_eval,
                                  dt_eval);
}

std::int64_t saturating_i64(const std::uint64_t value) {
  constexpr std::uint64_t kI64Max =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (value > kI64Max) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return static_cast<std::int64_t>(value);
}

void accumulate_device_flags(core::DeviceErrorFlags& acc,
                             const core::DeviceErrorFlags& step) {
  acc.nan_particle = std::max(acc.nan_particle, step.nan_particle);
  acc.invalid_cell = std::max(acc.invalid_cell, step.invalid_cell);
  acc.invalid_boundary = std::max(acc.invalid_boundary, step.invalid_boundary);
  acc.pool_overflow = std::max(acc.pool_overflow, step.pool_overflow);
  acc.opacity_out_of_range = std::max(acc.opacity_out_of_range,
                                      step.opacity_out_of_range);
  if (step.infinite_loop > 0) {
    const auto acc_i64 = static_cast<std::int64_t>(acc.infinite_loop);
    const auto step_i64 = static_cast<std::int64_t>(step.infinite_loop);
    const auto sum_i64 = acc_i64 + step_i64;
    const auto cap_i64 = static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max());
    if (sum_i64 > cap_i64) {
      static bool warned = false;
      if (!warned) {
        core::log_warning("DeviceErrorFlags: infinite_loop counter saturated at INT32_MAX");
        warned = true;
      }
    }
    acc.infinite_loop = static_cast<std::int32_t>(std::min(sum_i64, cap_i64));
  }
  acc.ddmc_sigma_tot_zero = std::max(acc.ddmc_sigma_tot_zero,
                                     step.ddmc_sigma_tot_zero);
  acc.roulette_kill = std::max(acc.roulette_kill, step.roulette_kill);
}

// Remap total-mass closure rejection threshold (0 = gate off). Shared by
// the ALE-phase gate and the forced-repair routes.
double remap_closure_reject_tol(const core::Config& cfg) {
  static const char* raw =
      std::getenv("TENRYU_I1B_REMAP_CLOSURE_REJECT_TOL");
  if (raw != nullptr && raw[0] != '\0') {
    const double x = std::atof(raw);
    return (std::isfinite(x) && x > 0.0) ? x : 0.0;
  }
  return cfg.numerics.ale.remap_mass_closure_reject_tol;
}

double boundary_pressure_drive(const core::State& state, const double eval_time) {
  TENRYU_ASSERT(state.pressure_drive_1d.has_value(),
                "pressure boundary requires initialized pressure_drive_1d table");
  // The pooled-cap scalar remains the perturbed drive's exact l=0 angular mean by Legendre orthogonality.
  return state.pressure_drive_1d->eval(eval_time);
}

double compute_boundary_pdv_work_1d(const core::State& state,
                                    const core::Config& cfg,
                                    const double dt_op,
                                    const double t_op) {
  if (state.mesh.dim != 1 || dt_op <= 0.0 || state.x_r.empty() || state.v_r.empty()) {
    return 0.0;
  }

  const std::size_t n_nodes = state.x_r.size();
  if (n_nodes == 0 || state.v_r.size() != n_nodes) {
    return 0.0;
  }

  double outer_node_r = 0.0;
  double outer_node_v = 0.0;
  double out2[2] = {0.0, 0.0};
  core::pull_two_scalars(state.x_r.data() + (n_nodes - 1),
                         state.v_r.data() + (n_nodes - 1),
                         out2,
                         "driver:boundary_pdv:pull");
  outer_node_r = out2[0];
  outer_node_v = out2[1];

  const hydro::HydroBoundaryType bc_type = hydro::parse_boundary_type_1d(cfg);
  double p_outer = 0.0;
  if (bc_type == hydro::HydroBoundaryType::PRESSURE) {
    p_outer = boundary_pressure_drive(state, t_op + 0.5 * dt_op);
  }

  const double r_outer = std::max(outer_node_r, 0.0);
  const bool cylindrical_1d =
      state.mesh.geometry_code == 1 ||
      core::geometry_1d_from_dimension(cfg.main.dimension) ==
          core::Geometry1D::Cylindrical;
  const double area_outer =
      cylindrical_1d ? 2.0 * kPi * r_outer : 4.0 * kPi * r_outer * r_outer;
  const double v_outer = outer_node_v;
  return p_outer * area_outer * v_outer * dt_op;
}

diagnostics::EnergyTotals compute_energy_totals_for_state(const core::State& state,
                                                          const core::Config& cfg) {
  if (cfg.main.dimension == "2D_RZ") {
    return diagnostics::compute_energy_totals_2d(state);
  }
  return diagnostics::compute_energy_totals_1d(state);
}

struct ConservationTotals {
  double mass = 0.0;
  double r_momentum = 0.0;
  double z_momentum = 0.0;
};

ConservationTotals compute_conservation_totals_for_state(const core::State& state,
                                                         const core::Config& cfg) {
  ConservationTotals totals{};
  const auto rho = copy_field_to_host(state.rho);
  const auto vol = copy_field_to_host(state.vol);
  if (rho.empty() || rho.size() != vol.size()) {
    return totals;
  }

  const auto v_r = copy_field_to_host(state.v_r);
  const auto v_z = copy_field_to_host(state.v_z);
  long double mass = 0.0L;
  long double r_momentum = 0.0L;
  long double z_momentum = 0.0L;

  if (cfg.main.dimension == "2D_RZ" && state.mesh.dim == 2) {
    const int nr = state.mesh.topo.nr;
    const int nz = state.mesh.topo.nz;
    const int stride = nz + 1;
    const std::size_t expected_nodes =
        static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
    if (v_r.size() != expected_nodes || v_z.size() != expected_nodes ||
        rho.size() != static_cast<std::size_t>(nr * nz)) {
      return totals;
    }
    for (int i = 0; i < nr; ++i) {
      for (int j = 0; j < nz; ++j) {
        const std::size_t c = static_cast<std::size_t>(i * nz + j);
        const std::size_t n00 = static_cast<std::size_t>(i * stride + j);
        const std::size_t n10 = static_cast<std::size_t>((i + 1) * stride + j);
        const std::size_t n11 =
            static_cast<std::size_t>((i + 1) * stride + (j + 1));
        const std::size_t n01 = static_cast<std::size_t>(i * stride + (j + 1));
        const long double cell_mass =
            static_cast<long double>(rho[c]) * static_cast<long double>(vol[c]);
        const long double vr_cell = 0.25L * (static_cast<long double>(v_r[n00]) +
                                            static_cast<long double>(v_r[n10]) +
                                            static_cast<long double>(v_r[n11]) +
                                            static_cast<long double>(v_r[n01]));
        const long double vz_cell = 0.25L * (static_cast<long double>(v_z[n00]) +
                                            static_cast<long double>(v_z[n10]) +
                                            static_cast<long double>(v_z[n11]) +
                                            static_cast<long double>(v_z[n01]));
        mass += cell_mass;
        r_momentum += cell_mass * vr_cell;
        z_momentum += cell_mass * vz_cell;
      }
    }
  } else {
    if (v_r.size() != rho.size() + 1) {
      return totals;
    }
    for (std::size_t c = 0; c < rho.size(); ++c) {
      const long double cell_mass =
          static_cast<long double>(rho[c]) * static_cast<long double>(vol[c]);
      const long double vr_cell =
          0.5L * (static_cast<long double>(v_r[c]) +
                  static_cast<long double>(v_r[c + 1]));
      mass += cell_mass;
      r_momentum += cell_mass * vr_cell;
    }
  }

  totals.mass = static_cast<double>(mass);
  totals.r_momentum = static_cast<double>(r_momentum);
  totals.z_momentum = static_cast<double>(z_momentum);
  return totals;
}

double compute_r_momentum_hoop_source_for_state(const core::State& state) {
  const auto Pe = copy_field_to_host(state.Pe);
  const auto Pi = copy_field_to_host(state.Pi);
  const auto& area = state.mesh.cell_area;
  if (Pe.size() != Pi.size() || Pe.size() != area.size()) {
    return 0.0;
  }

  long double pressure_area_sum = 0.0L;
  for (std::size_t c = 0; c < Pe.size(); ++c) {
    pressure_area_sum +=
        (static_cast<long double>(Pe[c]) + static_cast<long double>(Pi[c])) *
        static_cast<long double>(area[c]);
  }
  return static_cast<double>(2.0L * static_cast<long double>(kPi) *
                             pressure_area_sum);
}

void reduce_energy_totals(diagnostics::EnergyTotals& totals,
                          const parallel::Reduction& reducer) {
  std::array<double, 3> values = {
      totals.E_int_e,
      totals.E_int_i,
      totals.E_kin,
  };
  reducer.allreduce_sum(values.data(), static_cast<int>(values.size()));
  totals.E_int_e = values[0];
  totals.E_int_i = values[1];
  totals.E_kin = values[2];
}

void reduce_conservation_totals(ConservationTotals& totals,
                                const parallel::Reduction& reducer) {
  std::array<double, 3> values = {
      totals.mass,
      totals.r_momentum,
      totals.z_momentum,
  };
  reducer.allreduce_sum(values.data(), static_cast<int>(values.size()));
  totals.mass = values[0];
  totals.r_momentum = values[1];
  totals.z_momentum = values[2];
}

std::string format_sci17(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(17) << value;
  return oss.str();
}

bool env_flag_value_with_presence(const char* name, bool* present = nullptr) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    if (present != nullptr) {
      *present = false;
    }
    return false;
  }
  if (present != nullptr) {
    *present = true;
  }
  return env_flag_raw_enabled(raw);
}

bool pole_axis_bbsw_effective_enabled(const core::Config& cfg) {
  static const char* const raw = std::getenv("TENRYU_I1B_POLE_AXIS_BBSW");
  static const bool env_present = raw != nullptr && raw[0] != '\0';
  static const bool env_enabled = env_present && env_flag_raw_enabled(raw);
  return env_present ? env_enabled : cfg.numerics.ale.pole_axis_bbsw_enabled;
}

bool mass_floor_absorb_effective_enabled(const core::Config& cfg) {
  static const char* const raw =
      std::getenv("TENRYU_I1B_MASS_FLOOR_ABSORB");
  static const bool env_present = raw != nullptr && raw[0] != '\0';
  static const bool env_enabled = env_present && raw[0] != '0';
  return env_present ? env_enabled : cfg.numerics.ale.mass_floor_absorb_enabled;
}

bool i1b_energy_audit_enabled() {
  return env_flag_enabled("TENRYU_I1B_ENERGY_AUDIT");
}

int i1b_energy_audit_every() {
  return std::max(1, env_nonnegative_int("TENRYU_I1B_ENERGY_AUDIT_EVERY", 1));
}

bool ring7_quotient_effective_enabled(const core::Config& cfg) {
  bool present = false;
  const bool env_enabled =
      env_flag_value_with_presence("TENRYU_I1B_RING7_QUOTIENT", &present);
  return present ? env_enabled : cfg.numerics.hydro.ring7_quotient_enabled;
}

const char* radiation_mode_name(const core::RadiationMode mode) {
  switch (mode) {
    case core::RadiationMode::ImcDdmc:
      return "imc_ddmc";
    case core::RadiationMode::MultigroupDiffusion:
      return "multigroup_diffusion";
    case core::RadiationMode::SnTransport:
      return "sn_transport";
  }
  return "unknown";
}

template <typename T>
std::vector<T> copy_device_array_to_host(const core::DeviceArray<T>& field) {
  std::vector<T> host;
  field.copy_to_host(host);
  return host;
}

struct EnergyAuditMassMomentum {
  double mass = 0.0;
  double p_r = 0.0;
  double p_z = 0.0;
};

struct EnergyAuditDriveWork {
  double method_a = 0.0;
  double method_b = 0.0;
};

struct EnergyAuditLedger {
  bool initialized = false;
  double E_total0 = 0.0;
  long double sum_R = 0.0L;
  long double sum_abs_W = 0.0L;
  long double sum_abs_Q = 0.0L;
  long double sum_abs_L = 0.0L;
};

int energy_audit_button_outer_node_ring(const core::State& state) {
  return (state.mesh.button_center && state.mesh.button_center->enabled)
             ? state.mesh.button_center->outer_node_ring
             : 0;
}

int energy_audit_active_nverts(const core::State& state, const int cell) {
  if (state.mesh.cell_nverts.size() ==
      static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    return mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, cell);
  }
  return state.mesh.corner_stride;
}

EnergyAuditMassMomentum compute_energy_audit_mass_momentum_local(
    const core::State& state,
    const core::Config& cfg) {
  EnergyAuditMassMomentum totals{};
  if (state.mesh.dim != 2 || state.mesh.topo.n_cells <= 0 ||
      state.mesh.topo.n_nodes <= 0) {
    return totals;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (state.mass.size() != static_cast<std::size_t>(n_cells) ||
      state.v_r.size() != static_cast<std::size_t>(n_nodes) ||
      state.v_z.size() != static_cast<std::size_t>(n_nodes) ||
      state.corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U) {
    return totals;
  }

  const auto mass = copy_field_to_host(state.mass);
  const auto corner_mass = copy_field_to_host(state.corner_mass);
  const auto v_r = copy_field_to_host(state.v_r);
  const auto v_z = copy_field_to_host(state.v_z);
  const auto x_r = copy_field_to_host(state.x_r);
  const auto x_z = copy_field_to_host(state.x_z);
  if (mass.size() != static_cast<std::size_t>(n_cells) ||
      corner_mass.size() != static_cast<std::size_t>(n_cells) * 4U ||
      v_r.size() != static_cast<std::size_t>(n_nodes) ||
      v_z.size() != static_cast<std::size_t>(n_nodes)) {
    return totals;
  }

  long double mass_total = 0.0L;
  for (const double m : mass) {
    mass_total += static_cast<long double>(m);
  }

  std::vector<double> node_mass(static_cast<std::size_t>(n_nodes), 0.0);
  const int button_outer_node_ring = energy_audit_button_outer_node_ring(state);
  const bool use_hydro_active =
      button_outer_node_ring >= 1 &&
      state.hydro_active.size() == static_cast<std::size_t>(n_cells);

  if (state.mesh.topo.multiblock.has_value()) {
    const auto reverse_offsets =
        copy_device_array_to_host(state.mesh.multiblock_reverse_csr_node_offsets);
    const auto reverse_cells =
        copy_device_array_to_host(state.mesh.multiblock_reverse_csr_node_cells);
    const auto reverse_corners =
        copy_device_array_to_host(state.mesh.multiblock_reverse_csr_node_corners);
    const auto cell_node_offsets =
        copy_device_array_to_host(state.mesh.multiblock_cell_node_csr_offsets);
    const auto cell_node_indices =
        copy_device_array_to_host(state.mesh.multiblock_cell_node_csr_indices);
    if (reverse_offsets.size() == static_cast<std::size_t>(n_nodes) + 1U &&
        reverse_cells.size() == reverse_corners.size()) {
      const bool control_volume_fallback =
          state.mesh.topo.multiblock->block_count == 5;
      const bool have_cell_csr =
          cell_node_offsets.size() == static_cast<std::size_t>(n_cells) + 1U &&
          !cell_node_indices.empty();
      for (int n = 0; n < n_nodes; ++n) {
        const int off = reverse_offsets[static_cast<std::size_t>(n)];
        const int end = reverse_offsets[static_cast<std::size_t>(n) + 1U];
        if (off < 0 || end < off ||
            static_cast<std::size_t>(end) > reverse_cells.size()) {
          continue;
        }
        double sum = 0.0;
        double fallback_sum = 0.0;
        double bbs_axis_sum = 0.0;
        const bool bbs_axis_node =
            cfg.numerics.hydro.bbs_axis_policy_enabled &&
            x_r.size() == static_cast<std::size_t>(n_nodes) &&
            std::fabs(x_r[static_cast<std::size_t>(n)]) <= 1.0e-300;
        for (int k = off; k < end; ++k) {
          const int c = reverse_cells[static_cast<std::size_t>(k)];
          const int corner = reverse_corners[static_cast<std::size_t>(k)];
          if (c < 0 || c >= n_cells || corner < 0 || corner >= 4) {
            continue;
          }
          if (use_hydro_active &&
              state.hydro_active[static_cast<std::size_t>(c)] == 0) {
            continue;
          }
          sum += corner_mass[static_cast<std::size_t>(c) * 4U +
                             static_cast<std::size_t>(corner)];
          const int active_nverts = energy_audit_active_nverts(state, c);
          const double m = mass[static_cast<std::size_t>(c)];
          if (m > 0.0 && std::isfinite(m)) {
            fallback_sum +=
                (active_nverts == 3) ? (m / 3.0) : (0.25 * m);
          }
          if (bbs_axis_node && have_cell_csr &&
              x_z.size() == static_cast<std::size_t>(n_nodes)) {
            const int cell_off =
                cell_node_offsets[static_cast<std::size_t>(c)];
            if (cell_off >= 0 &&
                static_cast<std::size_t>(cell_off + active_nverts) <=
                    cell_node_indices.size()) {
              double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
              if (active_nverts == 3) {
                const int n0 = cell_node_indices[static_cast<std::size_t>(cell_off + 0)];
                const int n1 = cell_node_indices[static_cast<std::size_t>(cell_off + 1)];
                const int n2 = cell_node_indices[static_cast<std::size_t>(cell_off + 2)];
                if (n0 >= 0 && n0 < n_nodes && n1 >= 0 && n1 < n_nodes &&
                    n2 >= 0 && n2 < n_nodes) {
                  hydro::rz::compute_triangle_corner_masses_exact(
                      std::fmax(m, 0.0),
                      x_r[static_cast<std::size_t>(n0)],
                      x_z[static_cast<std::size_t>(n0)],
                      x_r[static_cast<std::size_t>(n1)],
                      x_z[static_cast<std::size_t>(n1)],
                      x_r[static_cast<std::size_t>(n2)],
                      x_z[static_cast<std::size_t>(n2)],
                      m_corner);
                  bbs_axis_sum += std::fmax(m_corner[corner], 0.0);
                }
              } else {
                const int n00 = cell_node_indices[static_cast<std::size_t>(cell_off + 0)];
                const int n10 = cell_node_indices[static_cast<std::size_t>(cell_off + 1)];
                const int n11 = cell_node_indices[static_cast<std::size_t>(cell_off + 2)];
                const int n01 = cell_node_indices[static_cast<std::size_t>(cell_off + 3)];
                if (n00 >= 0 && n00 < n_nodes && n10 >= 0 && n10 < n_nodes &&
                    n11 >= 0 && n11 < n_nodes && n01 >= 0 && n01 < n_nodes &&
                    hydro::rz::compute_quad_corner_masses_rz_volume_no_planar_fallback(
                        std::fmax(m, 0.0),
                        x_r[static_cast<std::size_t>(n00)],
                        x_z[static_cast<std::size_t>(n00)],
                        x_r[static_cast<std::size_t>(n10)],
                        x_z[static_cast<std::size_t>(n10)],
                        x_r[static_cast<std::size_t>(n11)],
                        x_z[static_cast<std::size_t>(n11)],
                        x_r[static_cast<std::size_t>(n01)],
                        x_z[static_cast<std::size_t>(n01)],
                        m_corner)) {
                  bbs_axis_sum += std::fmax(m_corner[corner], 0.0);
                }
              }
            }
          }
        }
        const int degree = end - off;
        if (degree > 0 && control_volume_fallback && std::isfinite(sum) &&
            !(sum > 1.0e-300)) {
          if (bbs_axis_node) {
            sum = std::isfinite(bbs_axis_sum) ? bbs_axis_sum : sum;
          } else if (fallback_sum > 1.0e-300 &&
                     std::isfinite(fallback_sum) &&
                     sum >= -1.0e-8 * fallback_sum) {
            sum = fallback_sum;
          }
        }
        if (degree > 0 && !control_volume_fallback && std::isfinite(sum) &&
            !(sum > 1.0e-300)) {
          sum = 2.0e-300;
        }
        node_mass[static_cast<std::size_t>(n)] = sum;
      }
    }
  } else {
    const int nr = state.mesh.topo.nr;
    const int nz = state.mesh.topo.nz;
    const int stride = nz + 1;
    const std::size_t expected_nodes =
        static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
    if (nr > 0 && nz > 0 && expected_nodes == node_mass.size() &&
        x_r.size() == expected_nodes && x_z.size() == expected_nodes) {
      for (int c = 0; c < n_cells; ++c) {
        if (use_hydro_active &&
            state.hydro_active[static_cast<std::size_t>(c)] == 0) {
          continue;
        }
        if (button_outer_node_ring >= 1 && c == 0) {
          double centroid_r = 0.0;
          double centroid_z = 0.0;
          hydro::rz::button_polygon_area_centroid_from_nodes(
              x_r.data(), x_z.data(), button_outer_node_ring, nz,
              &centroid_r, &centroid_z);
          const double volume = hydro::rz::button_polygon_volume_from_nodes(
              x_r.data(), x_z.data(), button_outer_node_ring, nz);
          if (volume > 0.0 && std::isfinite(volume)) {
            const double rho_button = mass[0] / volume;
            for (int k = 0; k <= nz; ++k) {
              const int n = hydro::rz::button_seam_node_index(
                  button_outer_node_ring, k, nz);
              if (n >= 0 && n < n_nodes) {
                node_mass[static_cast<std::size_t>(n)] +=
                    hydro::rz::button_corner_mass_exact_subpolygon(
                        rho_button, x_r.data(), x_z.data(),
                        button_outer_node_ring, nz, k, centroid_r, centroid_z);
              }
            }
          }
          continue;
        }
        const int i = c / nz;
        const int j = c - i * nz;
        const int n00 = i * stride + j;
        const int n10 = (i + 1) * stride + j;
        const int n11 = (i + 1) * stride + (j + 1);
        const int n01 = i * stride + (j + 1);
        const std::size_t base = static_cast<std::size_t>(c) * 4U;
        node_mass[static_cast<std::size_t>(n00)] += corner_mass[base + 0U];
        node_mass[static_cast<std::size_t>(n10)] += corner_mass[base + 1U];
        node_mass[static_cast<std::size_t>(n11)] += corner_mass[base + 2U];
        if (energy_audit_active_nverts(state, c) != 3) {
          node_mass[static_cast<std::size_t>(n01)] += corner_mass[base + 3U];
        }
      }
    }
  }

  long double p_r = 0.0L;
  long double p_z = 0.0L;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t idx = static_cast<std::size_t>(n);
    p_r += static_cast<long double>(node_mass[idx]) *
           static_cast<long double>(v_r[idx]);
    p_z += static_cast<long double>(node_mass[idx]) *
           static_cast<long double>(v_z[idx]);
  }
  totals.mass = static_cast<double>(mass_total);
  totals.p_r = static_cast<double>(p_r);
  totals.p_z = static_cast<double>(p_z);
  return totals;
}

void energy_audit_accumulate_drive_faces(
    const std::vector<double>& r_before,
    const std::vector<double>& z_before,
    const std::vector<double>& r_after,
    const std::vector<double>& z_after,
    const int node_begin,
    const int nr,
    const int nz,
    const double p_ext,
    const bool rz_exact_endpoint,
    EnergyAuditDriveWork& work) {
  const int stride = nz + 1;
  if (node_begin < 0 || nr <= 0 || nz <= 0) {
    return;
  }
  const std::size_t shell_node_count =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  const std::size_t node_end =
      static_cast<std::size_t>(node_begin) + shell_node_count;
  if (node_end > r_before.size() || node_end > z_before.size() ||
      node_end > r_after.size() || node_end > z_after.size()) {
    return;
  }
  std::vector<double> r_mid(shell_node_count, 0.0);
  std::vector<double> z_mid(shell_node_count, 0.0);
  for (std::size_t i = 0; i < shell_node_count; ++i) {
    const std::size_t g = static_cast<std::size_t>(node_begin) + i;
    r_mid[i] = 0.5 * (r_before[g] + r_after[g]);
    z_mid[i] = 0.5 * (z_before[g] + z_after[g]);
  }

  for (int j = 0; j < nz; ++j) {
    const int n0_local = nr * stride + j;
    const int n1_local = nr * stride + (j + 1);
    const int n0 = node_begin + n0_local;
    const int n1 = node_begin + n1_local;
    if (rz_exact_endpoint) {
      const hydro::detail::RzBoundaryPressureEndpointAreaVectors areas =
          hydro::detail::r_outer_boundary_pressure_endpoint_area_vectors(
              r_mid.data(), z_mid.data(), nr, nz, j);
      work.method_a +=
          (-p_ext * areas.node0.r) *
              (r_after[static_cast<std::size_t>(n0)] -
               r_before[static_cast<std::size_t>(n0)]) +
          (-p_ext * areas.node0.z) *
              (z_after[static_cast<std::size_t>(n0)] -
               z_before[static_cast<std::size_t>(n0)]) +
          (-p_ext * areas.node1.r) *
              (r_after[static_cast<std::size_t>(n1)] -
               r_before[static_cast<std::size_t>(n1)]) +
          (-p_ext * areas.node1.z) *
              (z_after[static_cast<std::size_t>(n1)] -
               z_before[static_cast<std::size_t>(n1)]);
    } else {
      const hydro::detail::RzBoundaryPressureFaceAreaVector area =
          hydro::detail::r_outer_boundary_pressure_area_vector(
              r_mid.data(), z_mid.data(), nr, nz, j);
      work.method_a +=
          0.5 *
          ((-p_ext * area.r) *
               ((r_after[static_cast<std::size_t>(n0)] -
                 r_before[static_cast<std::size_t>(n0)]) +
                (r_after[static_cast<std::size_t>(n1)] -
                 r_before[static_cast<std::size_t>(n1)])) +
           (-p_ext * area.z) *
               ((z_after[static_cast<std::size_t>(n0)] -
                 z_before[static_cast<std::size_t>(n0)]) +
                (z_after[static_cast<std::size_t>(n1)] -
                 z_before[static_cast<std::size_t>(n1)])));
    }

    const double qr[4] = {
        r_before[static_cast<std::size_t>(n0)],
        r_before[static_cast<std::size_t>(n1)],
        r_after[static_cast<std::size_t>(n1)],
        r_after[static_cast<std::size_t>(n0)]};
    const double qz[4] = {
        z_before[static_cast<std::size_t>(n0)],
        z_before[static_cast<std::size_t>(n1)],
        z_after[static_cast<std::size_t>(n1)],
        z_after[static_cast<std::size_t>(n0)]};
    const double delta_v =
        hydro::rz::rz_polygon_volume_exact(qr, qz, 4);
    work.method_b += -p_ext * delta_v;
  }
}

EnergyAuditDriveWork compute_energy_audit_drive_work_local(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<double>& r_before,
    const std::vector<double>& z_before,
    const double t_before,
    const double dt) {
  EnergyAuditDriveWork work{};
  if (state.mesh.dim != 2 || !(dt > 0.0) ||
      r_before.size() != state.x_r.size() ||
      z_before.size() != state.x_z.size() ||
      !state.pressure_drive_1d.has_value()) {
    return work;
  }
  const auto r_outer_type =
      hydro::parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  if (r_outer_type != hydro::Boundary2DType::PRESSURE) {
    return work;
  }
  const auto r_after = copy_field_to_host(state.x_r);
  const auto z_after = copy_field_to_host(state.x_z);
  if (r_after.size() != r_before.size() || z_after.size() != z_before.size()) {
    return work;
  }
  const double p_ext = boundary_pressure_drive(state, t_before + 0.5 * dt);
  const bool rz_exact_endpoint =
      state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane;
  if (state.mesh.topo.multiblock.has_value()) {
    const mesh::BlockInfo& shell =
        mesh::mesh_topo_multiblock_outermost_polar_shell_block(
            *state.mesh.topo.multiblock);
    const int shell_node_begin =
        mesh::mesh_topo_multiblock_outermost_polar_shell_node_offset(
            state.mesh.topo);
    energy_audit_accumulate_drive_faces(r_before,
                                        z_before,
                                        r_after,
                                        z_after,
                                        shell_node_begin,
                                        shell.n_i_cells,
                                        shell.n_j_cells,
                                        p_ext,
                                        rz_exact_endpoint,
                                        work);
  } else {
    energy_audit_accumulate_drive_faces(r_before,
                                        z_before,
                                        r_after,
                                        z_after,
                                        0,
                                        state.mesh.topo.nr,
                                        state.mesh.topo.nz,
                                        p_ext,
                                        rz_exact_endpoint,
                                        work);
  }
  return work;
}

void reduce_energy_audit_mass_momentum(EnergyAuditMassMomentum& totals,
                                       const parallel::Reduction& reducer) {
  std::array<double, 3> values = {totals.mass, totals.p_r, totals.p_z};
  reducer.allreduce_sum(values.data(), static_cast<int>(values.size()));
  totals.mass = values[0];
  totals.p_r = values[1];
  totals.p_z = values[2];
}

void reduce_energy_audit_drive_work(EnergyAuditDriveWork& work,
                                    const parallel::Reduction& reducer) {
  std::array<double, 2> values = {work.method_a, work.method_b};
  reducer.allreduce_sum(values.data(), static_cast<int>(values.size()));
  work.method_a = values[0];
  work.method_b = values[1];
}

void emit_config_fingerprint_step0(const core::State& state,
                                   const core::Config& cfg,
                                   const std::string& case_name,
                                   const bool energy_audit_enabled,
                                   const int rank) {
  if (rank != 0 || state.step != 0) {
    return;
  }
  const bool ale_enabled = cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled;
  const bool pole_axis_bbsw_enabled = pole_axis_bbsw_effective_enabled(cfg);
  std::ostringstream oss;
  oss << "[config_fingerprint] step=0"
      << " bbsw_enabled=" << (pole_axis_bbsw_enabled ? 1 : 0)
      << " bbs_axis_policy="
      << (cfg.numerics.hydro.bbs_axis_policy_enabled ? 1 : 0)
      << " ale_enabled=" << (ale_enabled ? 1 : 0)
      << " conservative_remap="
      << (cfg.numerics.ale.conservative_remap_enabled ? 1 : 0)
      << " ring7_quotient="
      << (ring7_quotient_effective_enabled(cfg) ? 1 : 0)
      << " energy_audit=" << (energy_audit_enabled ? 1 : 0)
      << " case=" << case_name
      << " grid=" << state.mesh.topo.nr << "x" << state.mesh.topo.nz
      << " av_type=" << cfg.numerics.hydro.av_type
      << " av_model=" << core::av_model_to_string(cfg.numerics.hydro.av_model)
      << " av_linear=" << format_sci17(cfg.numerics.hydro.av_linear)
      << " av_quadratic=" << format_sci17(cfg.numerics.hydro.av_quadratic)
      << " subzonal_pressure="
      << (cfg.numerics.hydro.subzonal_pressure_enabled ? 1 : 0)
      << " subzonal_fixed_mass="
      << (hydro::corner_mass_lagrangian_invariant_enabled(cfg) ? 1 : 0)
      << " subzonal_mass_enabled="
      << (cfg.numerics.hydro.subzonal_mass_enabled ? 1 : 0)
      << " subzonal_mass_lagrangian_invariant_enabled="
      << (cfg.numerics.hydro.subzonal_mass_lagrangian_invariant_enabled ? 1 : 0)
      << " subzonal_merit_mode="
      << cfg.numerics.hydro.subzonal_merit_mode
      << " subzonal_alpha1="
      << format_sci17(cfg.numerics.hydro.subzonal_alpha1)
      << " subzonal_alpha2="
      << format_sci17(cfg.numerics.hydro.subzonal_alpha2)
      << " subzonal_merit_power="
      << cfg.numerics.hydro.subzonal_merit_power
      << " subzonal_merit_constant="
      << format_sci17(cfg.numerics.hydro.subzonal_merit_constant)
      << " pressure_r_outer="
      << (cfg.numerics.hydro.boundary_2d.r_outer == "pressure" ? 1 : 0)
      << " pressure_drive_1d="
      << (state.pressure_drive_1d.has_value() ? 1 : 0)
      << " radiation_enabled=" << (cfg.radiation.enabled ? 1 : 0)
      << " radiation_mode=" << radiation_mode_name(cfg.radiation.mode);
  const auto& pcfg =
      cfg.numerics.hydro.pressure_drive_perturbation;
  if (pcfg.enabled) {
    char drive_pert_fingerprint[192];
    std::snprintf(drive_pert_fingerprint,
                  sizeof(drive_pert_fingerprint),
                  " drive_pert=1 g_min=%.6e g_max=%.6e n_modes=%d n_spots=%d",
                  pcfg.g_min,
                  pcfg.g_max,
                  static_cast<int>(pcfg.mode_l.size()),
                  static_cast<int>(pcfg.spot_theta0.size()));
    oss << drive_pert_fingerprint;
  }
  core::log_info(oss.str());
  // AvModel::CswEdgePlusTensorLimited is rejected at config validation
  // (Stage G unimplemented), so these two checks cover every reachable
  // edge-AV model.
  if (cfg.numerics.hydro.av_model == core::AvModel::CswEdge ||
      cfg.numerics.hydro.av_model == core::AvModel::CswEdgeCsw98) {
    core::log_warning(
        "[av_diag] av_model is an edge-AV model: hydro/Qvisc frames and the "
        "diagnostics/av_max history group are populated only by the legacy scalar-vNR "
        "path and will read zero for this entire run. They are NOT evidence that the "
        "artificial viscosity is inactive. Use TENRYU_I1B_AV_ACTIVITY_LOG_EVERY=<n> for "
        "per-step edge-AV work and fire counts (judge activity by the work term: "
        "quiescent cells fire on roundoff-level velocity noise, so raw fire counts "
        "overstate activity), or TENRYU_DRIVE_LAYER_DIAG=<n> for the per-cell wav term.");
  }
}

void maybe_emit_i1b_energy_audit_step(
    const core::State& state,
    const core::Config& cfg,
    const parallel::Reduction& reducer,
    const diagnostics::EnergyTotals& energy_before_local,
    const diagnostics::EnergyTotals& energy_after_local,
    const double E_rad_before_local,
    const double E_rad_after_local,
    const double E_laser_in_local,
    const double E_laser_esc_local,
    const double E_cbet_iaw_local,
    const double E_rad_esc_local,
    const double E_marshak_in_local,
    const double E_volume_in_local,
    const std::vector<double>& r_before,
    const std::vector<double>& z_before,
    const double t_before,
    const double dt,
    const int every,
    const bool identity_skip_local,
    const bool ale_fallback_local,
    EnergyAuditLedger& ledger,
    const int rank) {
  const double identity_skip_value =
      reducer.allreduce_max(identity_skip_local ? 1.0 : 0.0);
  const double ale_fallback_value =
      reducer.allreduce_max(ale_fallback_local ? 1.0 : 0.0);
  const bool identity_skip = identity_skip_value > 0.5;
  const bool ale_fallback = ale_fallback_value > 0.5;
  const bool due =
      every <= 1 || (state.step >= 0 && (state.step % every) == 0);
  const bool should_log = due || identity_skip || ale_fallback;

  diagnostics::EnergyTotals energy_before = energy_before_local;
  diagnostics::EnergyTotals energy_after = energy_after_local;
  reduce_energy_totals(energy_before, reducer);
  reduce_energy_totals(energy_after, reducer);
  std::array<double, 5> external = {
      E_rad_before_local,
      E_rad_after_local,
      E_rad_esc_local,
      E_marshak_in_local,
      E_volume_in_local,
  };
  reducer.allreduce_sum(external.data(), static_cast<int>(external.size()));
  EnergyAuditMassMomentum mass_momentum =
      compute_energy_audit_mass_momentum_local(state, cfg);
  reduce_energy_audit_mass_momentum(mass_momentum, reducer);
  EnergyAuditDriveWork drive_work =
      compute_energy_audit_drive_work_local(state, cfg, r_before, z_before,
                                            t_before, dt);
  reduce_energy_audit_drive_work(drive_work, reducer);

  const double E_rad_before = external[0];
  const double E_rad_after = external[1];
  const double L_rad = E_laser_esc_local + E_cbet_iaw_local + external[2];
  const double Q_ext = E_laser_in_local + external[3] + external[4];
  const double E_total_before = energy_before.E_kin + energy_before.E_int_i +
                                energy_before.E_int_e + E_rad_before;
  const double E_total = energy_after.E_kin + energy_after.E_int_i +
                         energy_after.E_int_e + E_rad_after;
  if (!ledger.initialized) {
    ledger.initialized = true;
    ledger.E_total0 = E_total_before;
  }
  const double W_drive_budget = drive_work.method_a;
  const double R_num = (E_total - E_total_before) - W_drive_budget -
                       Q_ext + L_rad;
  ledger.sum_R += static_cast<long double>(R_num);
  ledger.sum_abs_W += static_cast<long double>(std::abs(W_drive_budget));
  ledger.sum_abs_Q += static_cast<long double>(std::abs(Q_ext));
  ledger.sum_abs_L += static_cast<long double>(std::abs(L_rad));
  const long double denom_ld = std::max(
      {static_cast<long double>(std::abs(ledger.E_total0)),
       static_cast<long double>(std::abs(E_total)),
       ledger.sum_abs_W,
       ledger.sum_abs_Q,
       ledger.sum_abs_L,
       1.0e-300L});
  const double eta_E =
      static_cast<double>(std::abs(ledger.sum_R) / denom_ld);

  if (should_log && rank == 0) {
    std::ostringstream oss;
    oss << "[energy_audit]"
        << " step=" << state.step
        << " t=" << format_sci17(state.t)
        << " dt=" << format_sci17(dt)
        << " E_total=" << format_sci17(E_total)
        << " K=" << format_sci17(energy_after.E_kin)
        << " Ui=" << format_sci17(energy_after.E_int_i)
        << " Ue=" << format_sci17(energy_after.E_int_e)
        << " Erad=" << format_sci17(E_rad_after)
        << " M_total=" << format_sci17(mass_momentum.mass)
        << " Pr=" << format_sci17(mass_momentum.p_r)
        << " Pz=" << format_sci17(mass_momentum.p_z)
        << " W_drive_A=" << format_sci17(drive_work.method_a)
        << " W_drive_B=" << format_sci17(drive_work.method_b)
        << " L_rad=" << format_sci17(L_rad)
        << " Q_ext=" << format_sci17(Q_ext)
        << " R_num=" << format_sci17(R_num)
        << " sum_R=" << format_sci17(static_cast<double>(ledger.sum_R))
        << " eta_E=" << format_sci17(eta_E)
        << " identity_skip_flag=" << (identity_skip ? 1 : 0)
        << " ale_fallback_flag=" << (ale_fallback ? 1 : 0);
    core::log_info(oss.str());
  }
}

double conservation_residual(const double current, const double reference) {
  const double delta = std::abs(current - reference);
  const double denom = std::abs(reference);
  return denom > 1.0e-30 ? delta / denom : delta;
}

double conservation_residual_with_scale(const double current,
                                        const double reference,
                                        const double scale_floor) {
  const double delta = std::abs(current - reference);
  const double denom = std::abs(reference);
  if (denom > 1.0e-30) {
    return delta / denom;
  }
  return delta / std::max(scale_floor, 1.0e-30);
}

double compute_initial_sound_speed_max(const core::State& state,
                                       const parallel::Reduction& reducer) {
  double cs_max = 0.0;
  if (!state.cs.empty()) {
    std::vector<double> cs(state.cs.size(), 0.0);
    state.cs.copy_to_host(cs.data());
    for (const double value : cs) {
      if (std::isfinite(value)) {
        cs_max = std::max(cs_max, std::abs(value));
      }
    }
  }
  return reducer.allreduce_max(cs_max);
}

void reduce_operator_snapshot(diagnostics::OperatorResidualSnapshot& snapshot,
                              const parallel::Reduction& reducer) {
  if (!snapshot.valid) {
    return;
  }
  std::array<double, 3> values = {
      snapshot.E_thermal_total,
      snapshot.E_kinetic_total,
      snapshot.E_kinetic_total_nodal,
  };
  reducer.allreduce_sum(values.data(), static_cast<int>(values.size()));
  snapshot.E_thermal_total = values[0];
  snapshot.E_kinetic_total = values[1];
  snapshot.E_kinetic_total_nodal = values[2];
}

EscapeValveEvent::CellState capture_escape_valve_cell_state(
    const core::State& state,
    const int cell_id) {
  EscapeValveEvent::CellState out;
  if (cell_id < 0) {
    return out;
  }
  const std::size_t c = static_cast<std::size_t>(cell_id);
  const std::size_t n_cells = state.rho.size();
  if (c >= n_cells) {
    return out;
  }

  const auto rho = copy_field_to_host(state.rho);
  const auto Pe = copy_field_to_host(state.Pe);
  const auto Pi = copy_field_to_host(state.Pi);
  const auto Te = copy_field_to_host(state.Te);
  const auto Ti = copy_field_to_host(state.Ti);
  if (rho.size() == n_cells) {
    out.rho = rho[c];
  }
  if (Pe.size() == n_cells) {
    out.p = Pe[c] + (Pi.size() == n_cells ? Pi[c] : 0.0);
  }
  if (Te.size() == n_cells) {
    out.Te = Te[c];
  }
  if (Ti.size() == n_cells) {
    out.Ti = Ti[c];
  }
  return out;
}

verification::EscapeValveAuditCellState make_audit_cell_state(
    const EscapeValveEvent::CellState& in) {
  verification::EscapeValveAuditCellState out;
  out.rho = in.rho;
  out.p = in.p;
  out.Te = in.Te;
  out.Ti = in.Ti;
  return out;
}

verification::EscapeValveAuditEvent make_escape_valve_audit_event(
    const EscapeValveEvent& in) {
  verification::EscapeValveAuditEvent out;
  out.flag_name = !in.flag_name.empty() ? in.flag_name : in.operator_inserted;
  if (const auto flag =
          verification::escape_valve_flag_from_name(out.flag_name)) {
    out.flag = *flag;
  }
  out.reason = in.reason;
  out.cell_id = in.cell_id;
  out.cell_i = in.cell_i;
  out.cell_j = in.cell_j;
  out.step = in.step;
  out.time = in.time;
  out.mass_delta = in.mass_delta;
  out.momentum_delta_r = in.momentum_delta_r;
  out.momentum_delta_z = in.momentum_delta_z;
  out.energy_delta = in.energy_delta;
  out.before_state = make_audit_cell_state(in.before_state);
  out.after_state = make_audit_cell_state(in.after_state);
  return out;
}

EscapeValveEvent make_escape_valve_event(
    const char* split_phase,
    const char* operator_inserted,
    const char* flag_name,
    const char* reason,
    int cell_id,
    const EscapeValveEvent::CellState& before_state,
    const EscapeValveEvent::CellState& after_state,
    double mass_delta,
    double momentum_delta_r,
    double momentum_delta_z,
    const diagnostics::OperatorResidualSnapshot& before,
    const diagnostics::OperatorResidualSnapshot& after,
    const int step,
    const double time,
    int mesh_nz) {
  EscapeValveEvent event;
  event.split_phase = split_phase != nullptr ? split_phase : "pre_hydro";
  event.operator_inserted =
      operator_inserted != nullptr ? operator_inserted : "unknown";
  event.order_degraded = true;
  if (before.valid) {
    event.E_thermal_before = before.E_thermal_total;
    event.E_kinetic_before = before.E_kinetic_total;
  }
  if (after.valid) {
    event.E_thermal_after = after.E_thermal_total;
    event.E_kinetic_after = after.E_kinetic_total;
  }
  event.step = step;
  event.time = time;
  event.cell_id = cell_id;
  if (cell_id >= 0 && mesh_nz > 0) {
    event.cell_i = cell_id / mesh_nz;
    event.cell_j = cell_id - event.cell_i * mesh_nz;
  }
  event.flag_name = flag_name != nullptr ? flag_name : "";
  event.reason = reason != nullptr ? reason : "";
  event.mass_delta = mass_delta;
  event.momentum_delta_r = momentum_delta_r;
  event.momentum_delta_z = momentum_delta_z;
  event.energy_delta =
      (event.E_thermal_after + event.E_kinetic_after) -
      (event.E_thermal_before + event.E_kinetic_before);
  event.before_state = before_state;
  event.after_state = after_state;
  return event;
}

const char* splitting_order_name(const SplittingOrder order) {
  switch (order) {
    case SplittingOrder::STRANG:
      return "strang";
    case SplittingOrder::SEQUENTIAL:
      return "sequential";
  }
  return "unknown";
}

std::string describe_split_sequence(const SplittingOrder order,
                                    const core::Config& cfg) {
  std::vector<std::string> operators;
  auto append = [&](const bool enabled, const char* label) {
    if (enabled) {
      operators.emplace_back(label);
    }
  };
  const bool embed_conduction_in_thermal_subcycle =
      cfg.numerics.radiation_thermal_subcycle && cfg.radiation.enabled &&
      !cfg.radiation.imc.two_stage;
  const bool standalone_conduction =
      cfg.numerics.conduction.enabled && !embed_conduction_in_thermal_subcycle;

  if (order == SplittingOrder::SEQUENTIAL) {
    append(cfg.numerics.hydro.enabled, "hydro(dt)");
    append(standalone_conduction, "conduction(dt)");
    append(cfg.laser.enabled, "laser(dt)");
    append(cfg.radiation.enabled, "radiation(dt)");
  } else {
    append(cfg.laser.enabled, "laser(dt)");
    append(cfg.numerics.hydro.enabled, "hydro(dt/2)");
    append(standalone_conduction, "conduction(dt)");
    append(cfg.radiation.enabled, "radiation(dt)");
    append(cfg.numerics.hydro.enabled, "hydro(dt/2)");
  }

  if (operators.empty()) {
    return "none";
  }

  std::ostringstream oss;
  for (std::size_t i = 0; i < operators.size(); ++i) {
    if (i != 0) {
      oss << " -> ";
    }
    oss << operators[i];
  }
  return oss.str();
}

}  // namespace

double compute_fld_rad_energy_total_host(const core::State& state,
                                         const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups = static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  if (state.rad_E.size() != n_cell_groups || state.vol.size() != n_cells) {
    return 0.0;
  }
  const auto rad_E = copy_field_to_host(state.rad_E);
  const auto vol = copy_field_to_host(state.vol);
  long double total = 0.0L;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const double V = vol[c];
    if (!std::isfinite(V) || V <= 0.0) {
      continue;
    }
    for (std::size_t g = 0; g < n_groups; ++g) {
      const double E = rad_E[c * n_groups + g];
      if (std::isfinite(E)) {
        total += static_cast<long double>(E) *
                 static_cast<long double>(V);
      }
    }
  }
  return static_cast<double>(total);
}

DeviceTemperatureAudit audit_temperatures_host(const core::State& state) {
  DeviceTemperatureAudit audit{};
  bool has_finite_te = false;
  const auto Te = copy_field_to_host(state.Te);
  for (const double value : Te) {
    if (!std::isfinite(value)) {
      audit.te_has_non_finite = true;
      continue;
    }
    if (!has_finite_te) {
      audit.max_te = value;
      has_finite_te = true;
    } else {
      audit.max_te = std::max(audit.max_te, value);
    }
  }

  const auto Ti = copy_field_to_host(state.Ti);
  for (const double value : Ti) {
    if (!std::isfinite(value)) {
      audit.ti_has_non_finite = true;
    }
  }

  const auto rho = copy_field_to_host(state.rho);
  for (const double value : rho) {
    if (!std::isfinite(value)) {
      audit.rho_has_non_finite = true;
    }
  }

  const auto ee = copy_field_to_host(state.ee);
  for (const double value : ee) {
    if (!std::isfinite(value)) {
      audit.ee_has_non_finite = true;
    }
  }

  const auto ei = copy_field_to_host(state.ei);
  for (const double value : ei) {
    if (!std::isfinite(value)) {
      audit.ei_has_non_finite = true;
    }
  }
  return audit;
}

void initialize_eos_fields_if_needed(core::State& state, core::Config& cfg) {
  initialize_eos_fields_if_needed_impl(state, cfg);
}

void observe_geometric_retry_failure(GeometricRetryStagnationState& state,
                                     const HydroStepResult& failure,
                                     const double current_dt) {
  const std::string reason =
      failure.reason != nullptr ? std::string(failure.reason) : std::string();
  const int cell = failure.first_failing_cell;
  const int corner = failure.first_failing_corner;
  const double sigma = mesh_forensics_sigma_safe(failure);
  const bool same_tuple = reason == state.last_reason &&
                          cell == state.last_cell &&
                          corner == state.last_corner;
  if (!same_tuple) {
    state = GeometricRetryStagnationState{};
    state.last_reason = reason;
    state.last_cell = cell;
    state.last_corner = corner;
    state.dt_at_first_failure =
        (std::isfinite(current_dt) && current_dt > 0.0) ? current_dt : 0.0;
  }

  auto& tracking = state.per_stage[geometric_retry_stage_key(failure.stage)];
  ++tracking.consecutive_count;
  tracking.sigma_band_min = std::min(tracking.sigma_band_min, sigma);
  tracking.sigma_band_max = std::max(tracking.sigma_band_max, sigma);
}

bool detect_geometric_retry_stagnation(
    const GeometricRetryStagnationState& state,
    const core::Config& cfg,
    const HydroStepResult&,
    const double current_dt) {
  const auto& gs = cfg.numerics.hydro.geometric_retry_stagnation;
  if (!gs.enabled) {
    return false;
  }

  if (state.dt_at_first_failure <= 0.0 ||
      current_dt / state.dt_at_first_failure > gs.dt_drop_factor) {
    return false;
  }

  for (const auto& entry : state.per_stage) {
    const auto& tracking = entry.second;
    if (tracking.consecutive_count < gs.same_cell_count_threshold) {
      continue;
    }
    const double sigma_mean =
        0.5 * (tracking.sigma_band_min + tracking.sigma_band_max);
    const double sigma_range =
        tracking.sigma_band_max - tracking.sigma_band_min;
    if (sigma_mean > 0.0 && sigma_range / sigma_mean <= gs.sigma_rel_tol) {
      return true;
    }
  }

  if (geometric_retry_total_count(state) >=
      2 * gs.same_cell_count_threshold) {
    return true;
  }

  return false;
}

std::string format_geometric_retry_stagnation_log(
    const GeometricRetryStagnationState& state,
    const HydroStepResult& failure,
    const int step,
    const double t,
    const double current_dt) {
  const auto* tracking =
      geometric_retry_stage_tracking(state, failure.stage);
  const StagePerStageTracking default_tracking;
  const StagePerStageTracking& current =
      tracking != nullptr ? *tracking : default_tracking;
  const double sigma_mean =
      0.5 * (current.sigma_band_min + current.sigma_band_max);
  const double sigma_range = current.sigma_band_max - current.sigma_band_min;
  const double sigma_rel =
      sigma_mean > 0.0 ? sigma_range / sigma_mean
                       : (sigma_range == 0.0 ? 0.0
                                             : std::numeric_limits<double>::infinity());
  const double dt_drop =
      state.dt_at_first_failure > 0.0
          ? current_dt / state.dt_at_first_failure
          : std::numeric_limits<double>::infinity();
  const std::string reason =
      failure.reason != nullptr ? std::string(failure.reason) : std::string();
  std::ostringstream oss;
  oss << "[geometric-retry-stagnation] step=" << step
      << " t=" << format_sci(t)
      << " cell=" << failure.first_failing_cell
      << " reason=" << reason
      << " same_count=" << current.consecutive_count
      << " total_count=" << geometric_retry_total_count(state)
      << " sigma_band=[" << format_sci(current.sigma_band_min) << ","
      << format_sci(current.sigma_band_max) << "]"
      << " sigma_rel=" << format_sci(sigma_rel)
      << " dt_first=" << format_sci(state.dt_at_first_failure)
      << " dt_current=" << format_sci(current_dt)
      << " dt_drop=" << format_sci(dt_drop);
  return oss.str();
}

void execute_split_operators(const SplittingOrder order,
                             const double dt,
                             const double t_n,
                             const SplitOperatorCallbacks& callbacks) {
  TENRYU_ASSERT(dt > 0.0, "execute_split_operators requires positive dt");
  TENRYU_ASSERT(static_cast<bool>(callbacks.hydro),
                "execute_split_operators requires hydro callback");
  TENRYU_ASSERT(static_cast<bool>(callbacks.conduction),
                "execute_split_operators requires conduction callback");
  TENRYU_ASSERT(static_cast<bool>(callbacks.laser),
                "execute_split_operators requires laser callback");
  TENRYU_ASSERT(static_cast<bool>(callbacks.burn),
                "execute_split_operators requires burn callback");
  TENRYU_ASSERT(static_cast<bool>(callbacks.radiation),
                "execute_split_operators requires radiation callback");

  if (order == SplittingOrder::SEQUENTIAL) {
    const HydroStepResult hydro_result = callbacks.hydro(dt, t_n, "sequential");
    if (hydro_result.retry_required) {
      return;
    }
    callbacks.conduction(dt, t_n);
    callbacks.laser(dt, t_n);
    callbacks.burn(dt, t_n);
    callbacks.radiation(dt, t_n);
    return;
  }

  const double dt_half = 0.5 * dt;
  const double t_mid = t_n + dt_half;
  callbacks.laser(dt, t_n);
  callbacks.burn(dt, t_n);
  HydroStepResult hydro_result = callbacks.hydro(dt_half, t_n, "strang_first");
  if (hydro_result.retry_required) {
    return;
  }
  // Operators after the leading laser source use phase-centered time.
  callbacks.conduction(dt, t_mid);
  callbacks.radiation(dt, t_mid);
  hydro_result = callbacks.hydro(dt_half, t_mid, "strang_second");
  if (hydro_result.retry_required) {
    return;
  }
}

namespace {

int parse_dt_env_int(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0' ||
      value < static_cast<long>(std::numeric_limits<int>::min()) ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(value);
}

inline bool stage24_telemetry_enabled() {
  static const bool enabled = []() {
    const char* raw = std::getenv("TENRYU_STAGE24_TELEMETRY");
    if (raw == nullptr || raw[0] == '\0') {
      return false;
    }
    return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
           std::strcmp(raw, "FALSE") != 0;
  }();
  return enabled;
}

std::string json_escape(const std::string& input) {
  std::string out;
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

void write_json_number(std::ostream& os, const char* key, const double value) {
  os << ",\"" << key << "\":";
  if (std::isfinite(value)) {
    os << value;
  } else {
    os << "null";
  }
}

std::filesystem::path dt_lineage_path() {
  if (const char* explicit_outdir = std::getenv("TENRYU_DT_LINEAGE_OUTDIR");
      explicit_outdir != nullptr && explicit_outdir[0] != '\0') {
    return std::filesystem::path(explicit_outdir) / "dt_lineage.jsonl";
  }
  if (const char* tenryu_outdir = std::getenv("TENRYU_OUTDIR");
      tenryu_outdir != nullptr && tenryu_outdir[0] != '\0') {
    return std::filesystem::path(tenryu_outdir) / "diag" / "dt_lineage.jsonl";
  }
  return std::filesystem::path("dt_lineage.jsonl");
}

bool dt_lineage_should_log(const core::State& state) {
  const int step0 = parse_dt_env_int("TENRYU_DT_LINEAGE_STEP0", 0);
  const int step1 = parse_dt_env_int("TENRYU_DT_LINEAGE_STEP1", 2000000000);
  const int every = parse_dt_env_int("TENRYU_DT_LINEAGE_EVERY", 0);
  const bool in_window = state.step >= step0 && state.step <= step1;
  const bool on_stride = every > 0 && state.step >= 0 && (state.step % every) == 0;
  return in_window || on_stride;
}

struct DtLineageContext {
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

thread_local DtLineageContext g_dt_lineage_context;

void reset_dt_lineage_ale_context() {
  const int prev_step_kershaw_mmatrix_fix_count =
      g_dt_lineage_context.prev_step_kershaw_mmatrix_fix_count;
  g_dt_lineage_context = DtLineageContext{};
  g_dt_lineage_context.prev_step_kershaw_mmatrix_fix_count =
      prev_step_kershaw_mmatrix_fix_count;
}

void set_dt_lineage_retry_context(const int retry_attempts,
                                  const HydroStepResult& result,
                                  const double dt_before,
                                  const double dt_after) {
  g_dt_lineage_context.retry_phase = "p1_retry";
  g_dt_lineage_context.retry_attempts = retry_attempts;
  g_dt_lineage_context.retry_first_failing_cell = result.first_failing_cell;
  g_dt_lineage_context.retry_first_failing_corner = result.first_failing_corner;
  g_dt_lineage_context.retry_reason = result.reason != nullptr ? result.reason : "";
  g_dt_lineage_context.retry_dt_before = dt_before;
  g_dt_lineage_context.retry_dt_after = dt_after;
  g_dt_lineage_context.retry_min_metric = result.min_metric;
}

void set_dt_lineage_stage24_context(
    const HydroStepResult& result,
    const int repair_generation,
    const int rung_reached,
    const int floor_stall_count,
    const bool axis_variational_projection_eligible,
    const char* axis_variational_projection_blocked_reason) {
  if (!stage24_telemetry_enabled()) {
    return;
  }
  g_dt_lineage_context.repair_generation = repair_generation;
  g_dt_lineage_context.rung_reached = rung_reached;
  g_dt_lineage_context.axis_variational_projection_eligible =
      axis_variational_projection_eligible;
  g_dt_lineage_context.axis_variational_projection_blocked_reason =
      axis_variational_projection_blocked_reason != nullptr
          ? axis_variational_projection_blocked_reason
          : "";
  g_dt_lineage_context.floor_stall_count = floor_stall_count;
  g_dt_lineage_context.retry_state_sensitive = result.state_sensitive;
  g_dt_lineage_context.retry_dt_sensitive = result.dt_sensitive;
  g_dt_lineage_context.retry_dt_star_est = result.dt_star_est;
  g_dt_lineage_context.retry_margin_rel = result.margin_rel;
}

void write_dt_lineage_jsonl(const core::State& state, const DtLineage& lineage) {
  if (!dt_lineage_should_log(state)) {
    return;
  }
  const std::filesystem::path path = dt_lineage_path();
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream os(path, std::ios::out | std::ios::app);
  TENRYU_ASSERT(os.good(), "dt lineage diagnostics failed to open JSONL output");
  os << std::scientific << std::setprecision(17);
  os << "{\"step\":" << state.step;
  write_json_number(os, "t", state.t);
  os << ",\"phase\":\"" << json_escape(lineage.phase) << "\"";
  write_json_number(os, "dt_chosen", lineage.dt_chosen);
  write_json_number(os, "dt_hydro", lineage.dt_hydro);
  write_json_number(os, "dt_cond", lineage.dt_cond);
  write_json_number(os, "dt_burn", lineage.dt_burn);
  write_json_number(os, "dt_hot_e", lineage.dt_hot_e);
  write_json_number(os, "dt_rad", lineage.dt_rad);
  write_json_number(os, "dt_growth", lineage.dt_growth);
  write_json_number(os, "dt_output", lineage.dt_output);
  write_json_number(os, "dt_remaining", lineage.dt_remaining);
  write_json_number(os, "dt_max", lineage.dt_max_namelist);
  os << ",\"limiter\":\"" << json_escape(lineage.limiter) << "\"";
  os << ",\"hydro_argmin_cell\":" << lineage.hydro_argmin_cell;
  os << ",\"hydro_argmin_i\":" << lineage.hydro_argmin_i;
  os << ",\"hydro_argmin_j\":" << lineage.hydro_argmin_j;
  write_json_number(os, "hydro_argmin_dt_at_cell", lineage.hydro_argmin_dt_at_cell);
  write_json_number(os, "hydro_argmin_sqrt_area", lineage.hydro_argmin_sqrt_area);
  write_json_number(os, "hydro_argmin_cs", lineage.hydro_argmin_cs);
  write_json_number(os, "hydro_argmin_rho", lineage.hydro_argmin_rho);
  write_json_number(os, "hydro_argmin_u_z", lineage.hydro_argmin_u_z);
  write_json_number(os, "dt_hydro_acoustic", lineage.dt_hydro_acoustic);
  write_json_number(os, "dt_hydro_post_shock", lineage.dt_hydro_post_shock);
  write_json_number(os, "dt_hydro_edge_av", lineage.dt_hydro_edge_av);
  write_json_number(os, "dt_hydro_subzonal", lineage.dt_hydro_subzonal);
  write_json_number(os,
                    "dt_hydro_edge_accel",
                    lineage.dt_hydro_edge_accel);
  write_json_number(os, "dt_hydro_axis_margin", lineage.dt_hydro_axis_margin);
  write_json_number(os, "dt_hydro_volume_rate", lineage.dt_hydro_volume_rate);
  write_json_number(os,
                    "dt_hydro_tri_fan_center",
                    lineage.dt_hydro_tri_fan_center);
  write_json_number(os,
                    "dt_hydro_corner_j_predict",
                    lineage.dt_hydro_corner_j_predict);
  os << ",\"tri_fan_center_cfl_cell_id\":"
     << lineage.tri_fan_center_cfl_cell_id;
  os << ",\"tri_fan_center_cfl_i\":" << lineage.tri_fan_center_cfl_i;
  os << ",\"tri_fan_center_cfl_j\":" << lineage.tri_fan_center_cfl_j;
  write_json_number(os,
                    "tri_fan_center_cfl_h",
                    lineage.tri_fan_center_cfl_h);
  write_json_number(os,
                    "tri_fan_center_cfl_c_eff",
                    lineage.tri_fan_center_cfl_c_eff);
  write_json_number(os,
                    "tri_fan_center_cfl_q_over_p",
                    lineage.tri_fan_center_cfl_q_over_p);
  write_json_number(os,
                    "tri_fan_center_cfl_dt_global_over_dt_center",
                    lineage.tri_fan_center_cfl_dt_global_over_dt_center);
  os << ",\"volume_rate_argmin_cell\":" << lineage.volume_rate_argmin_cell;
  write_json_number(os,
                    "volume_rate_argmin_frac_rate",
                    lineage.volume_rate_argmin_frac_rate);
  os << ",\"cond_argmin_cell\":" << lineage.cond_argmin_cell;
  write_json_number(os, "cond_argmin_deff", lineage.cond_argmin_deff);
  write_json_number(os, "cond_argmin_dl", lineage.cond_argmin_dl);
  os << ",\"rad_argmin_cell\":" << lineage.rad_argmin_cell;
  os << ",\"ale_attempted\":" << (lineage.ale_attempted ? "true" : "false");
  os << ",\"ale_applied\":" << (lineage.ale_applied ? "true" : "false");
  os << ",\"ale_rezone_triggered\":"
     << (lineage.ale_rezone_triggered ? "true" : "false");
  os << ",\"ale_rezone_converged\":"
     << (lineage.ale_rezone_converged ? "true" : "false");
  os << ",\"ale_quality_rollback\":"
     << (lineage.ale_quality_rollback ? "true" : "false");
  write_json_number(os, "ale_quality_min", lineage.ale_quality_min);
  os << ",\"ale_rezone_iterations\":" << lineage.ale_rezone_iterations;
  write_json_number(os, "ale_rezone_residual", lineage.ale_rezone_residual);
  write_json_number(os,
                    "post_ale_corner_trial_scale",
                    lineage.post_ale_corner_trial_scale);
  os << ",\"post_ale_first_failing_cell\":"
     << lineage.post_ale_first_failing_cell;
  os << ",\"post_ale_first_failing_corner\":"
     << lineage.post_ale_first_failing_corner;
  if (lineage.retry_active_repair_enabled) {
    write_json_number(os,
                      "retry_active_repair_q_balance",
                      lineage.retry_active_repair_q_balance);
    os << ",\"retry_active_repair_first_failing_cell\":"
       << lineage.retry_active_repair_first_failing_cell;
    os << ",\"retry_active_repair_invoked\":"
       << (lineage.retry_active_repair_invoked ? "true" : "false");
    write_json_number(os,
                      "retry_active_repair_post_ale_q_balance",
                      lineage.retry_active_repair_post_ale_q_balance);
    os << ",\"retry_active_repair_post_ale_still_bad\":"
       << (lineage.retry_active_repair_post_ale_still_bad ? "true" : "false");
  }
  os << ",\"prev_step_kershaw_mmatrix_fix_count\":"
     << lineage.prev_step_kershaw_mmatrix_fix_count;
  if (!lineage.retry_phase.empty()) {
    os << ",\"retry_phase\":\"" << json_escape(lineage.retry_phase) << "\"";
    os << ",\"retry_attempts\":" << lineage.retry_attempts;
    os << ",\"retry_first_failing_cell\":"
       << lineage.retry_first_failing_cell;
    os << ",\"retry_first_failing_corner\":"
       << lineage.retry_first_failing_corner;
    os << ",\"retry_reason\":\"" << json_escape(lineage.retry_reason) << "\"";
    write_json_number(os, "retry_dt_before", lineage.retry_dt_before);
    write_json_number(os, "retry_dt_after", lineage.retry_dt_after);
    write_json_number(os, "retry_min_metric", lineage.retry_min_metric);
  }
  if (stage24_telemetry_enabled()) {
    os << ",\"repair_generation\":" << lineage.repair_generation;
    os << ",\"rung_reached\":" << lineage.rung_reached;
    os << ",\"axis_variational_projection_eligible\":"
       << (lineage.axis_variational_projection_eligible ? "true" : "false");
    if (!lineage.axis_variational_projection_eligible) {
      os << ",\"axis_variational_projection_blocked_reason\":\""
         << json_escape(lineage.axis_variational_projection_blocked_reason)
         << "\"";
    }
    os << ",\"axis_variational_projection_engaged_via\":\""
       << (lineage.axis_variational_projection_engaged_via.empty()
               ? ""
               : json_escape(lineage.axis_variational_projection_engaged_via))
       << "\"";
    os << ",\"floor_stall_count\":" << lineage.floor_stall_count;
    os << ",\"retry_state_sensitive\":"
       << (lineage.retry_state_sensitive ? "true" : "false");
    os << ",\"retry_dt_sensitive\":"
       << (lineage.retry_dt_sensitive ? "true" : "false");
    write_json_number(os, "retry_dt_star_est", lineage.retry_dt_star_est);
    write_json_number(os, "retry_margin_rel", lineage.retry_margin_rel);
  }
  os << ",\"hydro_min_term\":\"" << json_escape(lineage.hydro_min_term)
     << "\"";
  os << ",\"hydro_min_term_detail\":\""
     << json_escape(lineage.hydro_min_term_detail) << "\"";
  os << ",\"hydro_min_other_term\":\""
     << json_escape(lineage.hydro_min_other_term) << "\"";
  os << ",\"hydro_min_cell\":" << lineage.hydro_min_cell;
  os << ",\"hydro_min_cell_raw\":" << lineage.hydro_min_cell_raw;
  os << ",\"hydro_min_edge\":" << lineage.hydro_min_edge;
  os << ",\"hydro_min_nodes\":[" << lineage.hydro_min_node0 << ","
     << lineage.hydro_min_node1 << "]";
  os << ",\"hydro_min_other_node\":" << lineage.hydro_min_other_node;
  write_json_number(os, "hydro_min_L", lineage.hydro_min_L);
  write_json_number(os, "hydro_min_du", lineage.hydro_min_du);
  write_json_number(os, "hydro_min_accel", lineage.hydro_min_accel);
  write_json_number(os,
                    "hydro_min_coefficient",
                    lineage.hydro_min_coefficient);
  write_json_number(os, "hydro_min_raw_dt", lineage.hydro_min_raw_dt);
  write_json_number(os, "hydro_min_rho", lineage.hydro_min_rho);
  write_json_number(os, "hydro_min_mu", lineage.hydro_min_mu);
  write_json_number(os,
                    "hydro_min_polar_lambda",
                    lineage.hydro_min_polar_lambda);
  write_json_number(os,
                    "hydro_min_polar_sigma",
                    lineage.hydro_min_polar_sigma);
  write_json_number(os,
                    "hydro_min_csw98_du_eff",
                    lineage.hydro_min_csw98_du_eff);
  write_json_number(os,
                    "hydro_min_tensor_rho",
                    lineage.hydro_min_tensor_rho);
  write_json_number(os,
                    "hydro_min_tensor_L",
                    lineage.hydro_min_tensor_L);
  write_json_number(os,
                    "hydro_min_tensor_mu",
                    lineage.hydro_min_tensor_mu);
  write_json_number(os,
                    "hydro_min_crossing_dr",
                    lineage.hydro_min_crossing_dr);
  write_json_number(os,
                    "hydro_min_crossing_closing",
                    lineage.hydro_min_crossing_closing);
  write_json_number(os,
                    "hydro_min_crossing_safety",
                    lineage.hydro_min_crossing_safety);
  os << "}\n";
}

const char* hydro_min_term_name(const hydro::HydroMinTermClass term) {
  switch (term) {
    case hydro::HydroMinTermClass::AcousticCell:
      return "acoustic_cell";
    case hydro::HydroMinTermClass::EdgeAv:
      return "edge_av";
    case hydro::HydroMinTermClass::EdgeCrossing:
      return "edge_crossing";
    case hydro::HydroMinTermClass::Other:
      return "other";
  }
  return "other";
}

const char* hydro_min_other_term_name(const hydro::HydroMinOtherTerm term) {
  switch (term) {
    case hydro::HydroMinOtherTerm::None:
      return "none";
    case hydro::HydroMinOtherTerm::PostShock:
      return "post_shock";
    case hydro::HydroMinOtherTerm::ArtificialHeat:
      return "artificial_heat";
    case hydro::HydroMinOtherTerm::SubzonalPressure:
      return "subzonal_pressure";
    case hydro::HydroMinOtherTerm::AxisMargin:
      return "axis_margin";
    case hydro::HydroMinOtherTerm::TriFanCenter:
      return "tri_fan_center";
    case hydro::HydroMinOtherTerm::CornerJPredict:
      return "corner_j_predict";
    case hydro::HydroMinOtherTerm::PoleAxisContact:
      return "pole_axis_contact";
    case hydro::HydroMinOtherTerm::RzGeometric:
      return "rz_geometric";
    case hydro::HydroMinOtherTerm::VolumeRate:
      return "volume_rate";
    case hydro::HydroMinOtherTerm::PolarSlavingStiffness:
      return "polar_slaving_stiffness";
    case hydro::HydroMinOtherTerm::CentralPseudoCoreAcoustic:
      return "central_pseudo_core_acoustic";
    case hydro::HydroMinOtherTerm::PoleAngularDerefineAcoustic:
      return "pole_angular_derefine_acoustic";
    case hydro::HydroMinOtherTerm::EdgeAccelDisplacement:
      return "edge_accel_displacement";
  }
  return "none";
}

const char* hydro_min_term_detail_name(const hydro::HydroMinTermDetail detail) {
  switch (detail) {
    case hydro::HydroMinTermDetail::Unknown:
      return "unknown";
    case hydro::HydroMinTermDetail::AcousticCell:
      return "acoustic_cell";
    case hydro::HydroMinTermDetail::CswEdgeAv:
      return "csw_edge_av";
    case hydro::HydroMinTermDetail::Csw98EdgeAv:
      return "csw98_edge_av_du_eff";
    case hydro::HydroMinTermDetail::TensorAv:
      return "mimetic_tensor_av";
    case hydro::HydroMinTermDetail::EdgeCrossing:
      return "node_crossing";
  }
  return "unknown";
}

void copy_hydro_min_contributor(
    const hydro::HydroMinContributor& contributor,
    DtLineage& lineage) {
  lineage.hydro_min_term = hydro_min_term_name(contributor.term_class);
  lineage.hydro_min_term_detail =
      hydro_min_term_detail_name(contributor.term_detail);
  lineage.hydro_min_other_term =
      hydro_min_other_term_name(contributor.other_term);
  lineage.hydro_min_cell = contributor.cell_id;
  lineage.hydro_min_cell_raw = contributor.cell_id_raw;
  lineage.hydro_min_edge = contributor.edge_id;
  lineage.hydro_min_node0 = contributor.node0;
  lineage.hydro_min_node1 = contributor.node1;
  lineage.hydro_min_other_node = contributor.other_node;
  lineage.hydro_min_L = contributor.length;
  lineage.hydro_min_du = contributor.du;
  lineage.hydro_min_accel = contributor.acceleration;
  lineage.hydro_min_coefficient = contributor.coefficient;
  lineage.hydro_min_raw_dt = contributor.raw_dt;
  lineage.hydro_min_rho = contributor.rho;
  lineage.hydro_min_mu = contributor.mu;
  lineage.hydro_min_polar_lambda = contributor.polar_lambda;
  lineage.hydro_min_polar_sigma = contributor.polar_sigma;
  if (contributor.term_detail == hydro::HydroMinTermDetail::Csw98EdgeAv) {
    lineage.hydro_min_csw98_du_eff = contributor.du;
  }
  if (contributor.term_detail == hydro::HydroMinTermDetail::TensorAv) {
    lineage.hydro_min_tensor_rho = contributor.rho;
    lineage.hydro_min_tensor_L = contributor.length;
    lineage.hydro_min_tensor_mu = contributor.mu;
  }
  if (contributor.term_detail == hydro::HydroMinTermDetail::EdgeCrossing) {
    lineage.hydro_min_crossing_dr = contributor.length;
    lineage.hydro_min_crossing_closing = contributor.du;
    lineage.hydro_min_crossing_safety = contributor.coefficient;
  }
}

bool dt_values_match(const double a, const double b) {
  if (!std::isfinite(a) || !std::isfinite(b)) {
    return false;
  }
  const double scale = std::max({std::abs(a), std::abs(b), 1.0e-300});
  return std::abs(a - b) <= 1.0e-12 * scale;
}

int dt_winner_code(const std::string& winner) {
  if (winner == "hydro") {
    return 1;
  }
  if (winner == "rad") {
    return 2;
  }
  if (winner == "growth") {
    return 3;
  }
  if (winner == "max") {
    return 4;
  }
  if (winner == "cond") {
    return 5;
  }
  if (winner == "post_shock") {
    return 6;
  }
  if (winner == "volume_rate") {
    return 7;
  }
  if (winner == "output") {
    return 8;
  }
  if (winner == "t_end") {
    return 9;
  }
  if (winner == "init") {
    return 10;
  }
  if (winner == "driver_retry") {
    return 11;
  }
  if (winner == "tri_fan_center") {
    return 12;
  }
  if (winner == "hot_electron") {
    return 13;
  }
  if (winner == "corner_j_predict") {
    return 14;
  }
  return 0;
}

std::string canonical_dt_winner(const DtLineage& lineage) {
  if (lineage.limiter == "conduction") {
    return "cond";
  }
  if (lineage.limiter == "radiation") {
    return "rad";
  }
  if (lineage.limiter == "hydro") {
    if (dt_values_match(lineage.dt_hydro, lineage.dt_hydro_post_shock)) {
      return "post_shock";
    }
    if (dt_values_match(lineage.dt_hydro, lineage.dt_hydro_tri_fan_center)) {
      return "tri_fan_center";
    }
    if (dt_values_match(lineage.dt_hydro,
                        lineage.dt_hydro_corner_j_predict)) {
      return "corner_j_predict";
    }
    if (dt_values_match(lineage.dt_hydro, lineage.dt_hydro_volume_rate)) {
      return "volume_rate";
    }
    return "hydro";
  }
  if (lineage.limiter == "unknown") {
    return "other";
  }
  if (lineage.limiter == "hot_electron") {
    return "hot_electron";
  }
  return lineage.limiter;
}

diagnostics::DtBreakdownHistoryRecord make_dt_breakdown_history_record(
    const core::State& state,
    const DtLineage& lineage) {
  (void)state;
  diagnostics::DtBreakdownHistoryRecord record;
  record.valid = true;
  record.cycle = lineage.cycle;
  record.t_s = lineage.t_s;
  record.dt_chosen = lineage.dt_chosen;
  record.dt_winner = canonical_dt_winner(lineage);
  record.dt_winner_code = dt_winner_code(record.dt_winner);
  record.dt_hydro = lineage.dt_hydro;
  record.dt_rad = lineage.dt_rad;
  record.dt_cond = lineage.dt_cond;
  record.dt_post_shock = lineage.dt_hydro_post_shock;
  record.dt_growth = lineage.dt_growth;
  record.dt_max = lineage.dt_max_namelist;
  record.dt_output = lineage.dt_output;
  record.dt_remaining = lineage.dt_remaining;
  record.dt_hydro_acoustic = lineage.dt_hydro_acoustic;
  record.dt_hydro_axis_margin = lineage.dt_hydro_axis_margin;
  record.dt_hydro_volume_rate = lineage.dt_hydro_volume_rate;
  record.dt_hydro_tri_fan_center = lineage.dt_hydro_tri_fan_center;
  record.tri_fan_center_cfl_cell_id = lineage.tri_fan_center_cfl_cell_id;
  record.tri_fan_center_cfl_i = lineage.tri_fan_center_cfl_i;
  record.tri_fan_center_cfl_j = lineage.tri_fan_center_cfl_j;
  record.tri_fan_center_cfl_h = lineage.tri_fan_center_cfl_h;
  record.tri_fan_center_cfl_c_eff = lineage.tri_fan_center_cfl_c_eff;
  record.tri_fan_center_cfl_q_over_p = lineage.tri_fan_center_cfl_q_over_p;
  record.tri_fan_center_cfl_dt_global_over_dt_center =
      lineage.tri_fan_center_cfl_dt_global_over_dt_center;
  record.hydro_winner_cell_id = lineage.hydro_argmin_cell;
  record.hydro_winner_i = lineage.hydro_argmin_i;
  record.hydro_winner_j = lineage.hydro_argmin_j;
  record.hydro_winner_dt_at_cell = lineage.hydro_argmin_dt_at_cell;
  record.hydro_winner_dl_at_cell = lineage.hydro_argmin_sqrt_area;
  record.hydro_winner_cs_at_cell = lineage.hydro_argmin_cs;
  record.hydro_winner_rho_at_cell = lineage.hydro_argmin_rho;
  record.hydro_winner_u_z_at_cell = lineage.hydro_argmin_u_z;
  return record;
}

}  // namespace

DtLineage compute_dt_lineage(const core::State& state,
                             const core::Config& cfg,
                             const double t_end,
                             const char* phase,
                             const hydro::HydroEOSContext* eos_ctx,
                             const bool defer_dt_floor_abort,
                             DtReanchorState* reanchor) {
  DtLineage lineage;
  lineage.cycle = state.step;
  lineage.t_s = state.t;
  lineage.phase = phase != nullptr ? phase : "primary";
  lineage.ale_attempted = g_dt_lineage_context.ale_attempted;
  lineage.ale_applied = g_dt_lineage_context.ale_applied;
  lineage.ale_rezone_triggered = g_dt_lineage_context.ale_rezone_triggered;
  lineage.ale_rezone_converged = g_dt_lineage_context.ale_rezone_converged;
  lineage.ale_quality_rollback = g_dt_lineage_context.ale_quality_rollback;
  lineage.ale_quality_min = g_dt_lineage_context.ale_quality_min;
  lineage.ale_rezone_iterations = g_dt_lineage_context.ale_rezone_iterations;
  lineage.ale_rezone_residual = g_dt_lineage_context.ale_rezone_residual;
  lineage.post_ale_corner_trial_scale =
      g_dt_lineage_context.post_ale_corner_trial_scale;
  lineage.post_ale_first_failing_cell =
      g_dt_lineage_context.post_ale_first_failing_cell;
  lineage.post_ale_first_failing_corner =
      g_dt_lineage_context.post_ale_first_failing_corner;
  lineage.retry_active_repair_enabled =
      g_dt_lineage_context.retry_active_repair_enabled;
  lineage.retry_active_repair_q_balance =
      g_dt_lineage_context.retry_active_repair_q_balance;
  lineage.retry_active_repair_first_failing_cell =
      g_dt_lineage_context.retry_active_repair_first_failing_cell;
  lineage.retry_active_repair_invoked =
      g_dt_lineage_context.retry_active_repair_invoked;
  lineage.retry_active_repair_post_ale_q_balance =
      g_dt_lineage_context.retry_active_repair_post_ale_q_balance;
  lineage.retry_active_repair_post_ale_still_bad =
      g_dt_lineage_context.retry_active_repair_post_ale_still_bad;
  lineage.prev_step_kershaw_mmatrix_fix_count =
      g_dt_lineage_context.prev_step_kershaw_mmatrix_fix_count;
  lineage.retry_phase = g_dt_lineage_context.retry_phase;
  lineage.retry_attempts = g_dt_lineage_context.retry_attempts;
  lineage.retry_first_failing_cell =
      g_dt_lineage_context.retry_first_failing_cell;
  lineage.retry_first_failing_corner =
      g_dt_lineage_context.retry_first_failing_corner;
  lineage.retry_reason = g_dt_lineage_context.retry_reason;
  lineage.retry_dt_before = g_dt_lineage_context.retry_dt_before;
  lineage.retry_dt_after = g_dt_lineage_context.retry_dt_after;
  lineage.retry_min_metric = g_dt_lineage_context.retry_min_metric;
  lineage.repair_generation = g_dt_lineage_context.repair_generation;
  lineage.rung_reached = g_dt_lineage_context.rung_reached;
  lineage.axis_variational_projection_eligible =
      g_dt_lineage_context.axis_variational_projection_eligible;
  lineage.axis_variational_projection_blocked_reason =
      g_dt_lineage_context.axis_variational_projection_blocked_reason;
  lineage.axis_variational_projection_engaged_via =
      g_dt_lineage_context.axis_variational_projection_engaged_via;
  lineage.floor_stall_count = g_dt_lineage_context.floor_stall_count;
  lineage.retry_state_sensitive = g_dt_lineage_context.retry_state_sensitive;
  lineage.retry_dt_sensitive = g_dt_lineage_context.retry_dt_sensitive;
  lineage.retry_dt_star_est = g_dt_lineage_context.retry_dt_star_est;
  lineage.retry_margin_rel = g_dt_lineage_context.retry_margin_rel;
  lineage.dt_remaining = t_end - state.t;
  lineage.dt_max_namelist = cfg.numerics.dt.max_s;
  double dt_new = cfg.numerics.dt.max_s;
  double dt_hydro = std::numeric_limits<double>::infinity();
  double dt_cond = std::numeric_limits<double>::infinity();
  double dt_brag = std::numeric_limits<double>::infinity();
  double dt_rad_val = std::numeric_limits<double>::infinity();
  double dt_growth = std::numeric_limits<double>::infinity();
  radiation::DtRadDiagnostics dt_rad_diag{};
  double dt_init = std::numeric_limits<double>::infinity();

  if (state.step == 0) {
    if (cfg.numerics.dt.initial_s > 0.0) {
      dt_new = cfg.numerics.dt.initial_s;
      dt_init = dt_new;
    } else {
      double dt_hydro0 = std::numeric_limits<double>::infinity();
      if ((cfg.main.dimension == "1D_SPH" || cfg.main.dimension == "1D_CYL" ||
           cfg.main.dimension == "2D_RZ") &&
          state.mesh.node_r != nullptr &&
          cfg.numerics.hydro.enabled) {
        dt_hydro0 = hydro::compute_dt_hydro(state, cfg);
      }
      if (std::isfinite(dt_hydro0) && dt_hydro0 > 0.0) {
        dt_new = 0.1 * dt_hydro0;
      } else {
        dt_new = cfg.numerics.dt.max_s;
      }
    }
  }

  if ((cfg.main.dimension == "1D_SPH" || cfg.main.dimension == "1D_CYL" ||
       cfg.main.dimension == "2D_RZ") &&
      state.mesh.node_r != nullptr) {
    if (cfg.numerics.hydro.enabled) {
      const hydro::HydroDtDiagnostics hydro_dt =
          hydro::compute_dt_hydro_diagnostics(state, cfg);
      dt_hydro = hydro_dt.dt;
      lineage.hydro_argmin_cell = hydro_dt.cfl_winner.cell_id;
      lineage.hydro_argmin_i = hydro_dt.cfl_winner.i;
      lineage.hydro_argmin_j = hydro_dt.cfl_winner.j;
      lineage.hydro_argmin_dt_at_cell = hydro_dt.cfl_winner.dt_at_cell;
      lineage.hydro_argmin_sqrt_area = hydro_dt.cfl_winner.dl_at_cell;
      lineage.hydro_argmin_cs = hydro_dt.cfl_winner.cs_at_cell;
      lineage.hydro_argmin_rho = hydro_dt.cfl_winner.rho_at_cell;
      lineage.hydro_argmin_u_z = hydro_dt.cfl_winner.u_z_at_cell;
      lineage.dt_hydro_acoustic = hydro_dt.acoustic_dt;
      lineage.dt_hydro_post_shock = hydro_dt.post_shock_dt;
      lineage.dt_hydro_edge_av = hydro_dt.edge_av_dt;
      lineage.dt_hydro_subzonal = hydro_dt.subzonal_pressure_dt;
      lineage.dt_hydro_edge_accel = hydro_dt.edge_accel_dt;
      lineage.dt_hydro_axis_margin = hydro_dt.axis_margin_dt;
      lineage.dt_hydro_volume_rate = hydro_dt.volume_rate_dt;
      lineage.dt_hydro_tri_fan_center = hydro_dt.tri_fan_center_cfl.dt;
      lineage.dt_hydro_corner_j_predict = hydro_dt.corner_j_predict_dt;
      lineage.tri_fan_center_cfl_cell_id =
          hydro_dt.tri_fan_center_cfl.cell_id;
      lineage.tri_fan_center_cfl_i = hydro_dt.tri_fan_center_cfl.i;
      lineage.tri_fan_center_cfl_j = hydro_dt.tri_fan_center_cfl.j;
      lineage.tri_fan_center_cfl_h = hydro_dt.tri_fan_center_cfl.h;
      lineage.tri_fan_center_cfl_h_2ap =
          hydro_dt.tri_fan_center_cfl.h_2ap;
      lineage.tri_fan_center_cfl_min_altitude =
          hydro_dt.tri_fan_center_cfl.min_altitude;
      lineage.tri_fan_center_cfl_c_eff = hydro_dt.tri_fan_center_cfl.c_eff;
      lineage.tri_fan_center_cfl_q_over_p =
          hydro_dt.tri_fan_center_cfl.q_over_p;
      lineage.tri_fan_center_cfl_nverts =
          hydro_dt.tri_fan_center_cfl.nverts;
      for (int k = 0; k < 8; ++k) {
        lineage.tri_fan_center_cfl_r8[k] =
            hydro_dt.tri_fan_center_cfl.r8[k];
        lineage.tri_fan_center_cfl_z8[k] =
            hydro_dt.tri_fan_center_cfl.z8[k];
      }
      for (int k = 0; k < 4; ++k) {
        lineage.tri_fan_center_cfl_r4[k] =
            lineage.tri_fan_center_cfl_r8[k];
        lineage.tri_fan_center_cfl_z4[k] =
            lineage.tri_fan_center_cfl_z8[k];
      }
      lineage.tri_fan_center_cfl_block_id =
          hydro_dt.tri_fan_center_cfl.block_id;
      dt_new = std::min(dt_new, dt_hydro);
    }
    if (cfg.numerics.conduction.enabled) {
      dt_cond = hydro::compute_dt_conduction(state, cfg, eos_ctx);
      dt_new = std::min(dt_new, dt_cond);
    }
    {
      // Braginskii viscosity explicit stability limit (config-gated,
      // with env diagnostic overrides; disabled => +inf/no effect).
      dt_brag = hydro::braginskii::compute_dt_braginskii(state, cfg);
      lineage.dt_visc = dt_brag;
      dt_new = std::min(dt_new, dt_brag);
    }
    if (cfg.radiation.enabled) {
      dt_rad_val = radiation::IMC::compute_dt_rad(state, cfg, &dt_rad_diag);
      dt_new = std::min(dt_new, dt_rad_val);
    }
  }
  const double dt_burn = (state.burn_dt_limit_s > 0.0)
                             ? state.burn_dt_limit_s
                             : std::numeric_limits<double>::infinity();
  dt_new = std::min(dt_new, dt_burn);
  const double dt_hot_e = (state.hot_e_dt_limit_s > 0.0)
                              ? state.hot_e_dt_limit_s
                              : std::numeric_limits<double>::infinity();
  dt_new = std::min(dt_new, dt_hot_e);

  if (state.step > 0 || state.dt > 0.0) {
    const double growth_base =
        (state.dt_growth_ref > 0.0) ? state.dt_growth_ref : state.dt;
    dt_growth = cfg.numerics.dt.growth_factor * growth_base;
    dt_new = std::min(dt_new, dt_growth);
  }

  double dt_uncapped = dt_new;
  if (std::isfinite(state.evacuated_cells.contact_dt_cap_s)) {
    dt_new = std::min(dt_new, state.evacuated_cells.contact_dt_cap_s);
  }

  dt_new = std::min(dt_new, cfg.numerics.dt.max_s);
  dt_uncapped = std::min(dt_uncapped, cfg.numerics.dt.max_s);

  double dt_output = std::numeric_limits<double>::infinity();
  auto output_gap = [&](const double t_next, const double every_s) {
    // FP output-boundary guard (same 1e-14 convention as
    // update_next_output_time): a raw gap inside the eps window can undercut
    // dt.min_s and abort; propose at least 2*eps so the step definitively
    // crosses the boundary and the target then advances; the floor also clears
    // dt.min_s — overshooting an output boundary by <=2*min_s is harmless
    // (the emit tolerance fires) whereas undercutting min_s aborts.
    const double eps = 1.0e-14 * std::max(std::abs(state.t), every_s);
    return std::max(t_next - state.t,
                    std::max(2.0 * eps, 2.0 * cfg.numerics.dt.min_s));
  };
  if (cfg.output.plot_every_s > 0.0 && state.t_next_plot > state.t) {
    dt_output =
        std::min(dt_output, output_gap(state.t_next_plot, cfg.output.plot_every_s));
  }
  if (cfg.output.history_every_s > 0.0 && state.t_next_history > state.t) {
    dt_output = std::min(
        dt_output, output_gap(state.t_next_history, cfg.output.history_every_s));
  }
  if (cfg.output.checkpoint_every_s > 0.0 &&
      state.t_next_checkpoint > state.t) {
    dt_output = std::min(
        dt_output,
        output_gap(state.t_next_checkpoint, cfg.output.checkpoint_every_s));
  }
  dt_new = std::min(dt_new, dt_output);
  dt_uncapped = std::min(dt_uncapped, dt_output);

  dt_new = std::min(dt_new, t_end - state.t);
  dt_uncapped = std::min(dt_uncapped, t_end - state.t);
  const bool retry_dt_forced =
      !g_dt_lineage_context.retry_phase.empty() &&
      g_dt_lineage_context.retry_dt_after > 0.0;
  if (retry_dt_forced) {
    dt_new = g_dt_lineage_context.retry_dt_after;
    dt_uncapped = dt_new;
  }
  const double fixed_dt = i1b_fixed_dt();
  if (fixed_dt > 0.0) {
    dt_new = std::min(fixed_dt, t_end - state.t);
    dt_uncapped = dt_new;
  }
  static std::uint64_t dt_forensic_calls_since_log = 100;
  if (dt_forensic_calls_since_log < 100) {
    ++dt_forensic_calls_since_log;
  }
  if (dt_new < 1.0e-18 && dt_forensic_calls_since_log >= 100) {
    core::log_warning(
        "[dt-forensic] step=" + std::to_string(state.step) +
        " dt_new=" + format_sci17(dt_new) +
        " dt_hydro=" + format_sci17(dt_hydro) +
        " dt_cond=" + format_sci17(dt_cond) +
        " dt_brag=" + format_sci17(dt_brag) +
        " dt_rad=" + format_sci17(dt_rad_val) +
        " dt_burn=" + format_sci17(dt_burn) +
        " dt_hot_e=" + format_sci17(dt_hot_e) +
        " dt_growth=" + format_sci17(dt_growth) +
        " max_s=" + format_sci17(cfg.numerics.dt.max_s) +
        " contact_cap=" +
        format_sci17(state.evacuated_cells.contact_dt_cap_s) +
        " dt_output=" + format_sci17(dt_output) +
        " dt_uncapped=" + format_sci17(dt_uncapped));
    dt_forensic_calls_since_log = 0;
  }
  lineage.dt_uncapped_by_contact = dt_uncapped;
  double dt_chosen = dt_new;
  lineage.dt_chosen = dt_chosen;
  lineage.dt_hydro = dt_hydro;
  lineage.dt_cond = dt_cond;
  lineage.dt_burn = dt_burn;
  lineage.dt_hot_e = dt_hot_e;
  lineage.dt_rad = dt_rad_val;
  lineage.dt_growth = dt_growth;
  lineage.dt_output = dt_output;
  lineage.rad_argmin_cell = dt_rad_diag.limiting_cell;
  if (retry_dt_forced) {
    lineage.limiter = "driver_retry";
  } else if (dt_chosen == dt_hydro) {
    lineage.limiter = "hydro";
  } else if (dt_chosen == dt_cond) {
    lineage.limiter = "conduction";
  } else if (dt_chosen == dt_burn) {
    lineage.limiter = "burn";
  } else if (dt_chosen == dt_hot_e) {
    lineage.limiter = "hot_electron";
  } else if (dt_chosen == lineage.dt_visc) {
    lineage.limiter = "braginskii";
  } else if (dt_chosen == dt_rad_val) {
    lineage.limiter = "radiation";
  } else if (dt_chosen == dt_growth) {
    lineage.limiter = "growth";
  } else if (dt_chosen == dt_output) {
    lineage.limiter = "output";
  } else if (dt_chosen == lineage.dt_remaining) {
    lineage.limiter = "t_end";
  } else if (dt_chosen == cfg.numerics.dt.max_s) {
    lineage.limiter = "max";
  } else if (dt_chosen == dt_init) {
    lineage.limiter = "init";
  } else {
    lineage.limiter = "unknown";
  }

  static const bool dt_ctrl_check_enabled = []() {
    const char* v = std::getenv("TENRYU_DT_CONTROLLER_CHECK");
    return v != nullptr && std::string(v) == "1";
  }();
  if (dt_ctrl_check_enabled && !retry_dt_forced && fixed_dt <= 0.0 &&
      state.step > 0 &&
      cfg.main.dimension == "1D_SPH" &&
      !cfg.numerics.hydro.volume_rate_cfl_enabled &&
      !hydro::central_pseudo_core::configured(cfg)) {
    tenryu::coupling::DtLadderIn in;
    in.dt_hydro = dt_hydro;
    in.dt_cond = dt_cond;
    in.dt_visc = lineage.dt_visc;
    in.dt_rad = dt_rad_val;
    in.dt_prev = state.dt;
    in.growth_factor = cfg.numerics.dt.growth_factor;
    in.dt_max = cfg.numerics.dt.max_s;
    in.dt_output = dt_output;
    in.dt_remaining = t_end - state.t;
    double out2[2] = {0.0, 0.0};
    run_dt_controller_check(in, out2);
    const int host_limiter_id = [&lineage]() {
      if (lineage.limiter == "hydro") {
        return kDtLimiterHydro;
      }
      if (lineage.limiter == "conduction") {
        return kDtLimiterConduction;
      }
      if (lineage.limiter == "hot_electron") {
        return kDtLimiterHotElectron;
      }
      if (lineage.limiter == "braginskii") {
        return kDtLimiterBraginskii;
      }
      if (lineage.limiter == "radiation") {
        return kDtLimiterRadiation;
      }
      if (lineage.limiter == "growth") {
        return kDtLimiterGrowth;
      }
      if (lineage.limiter == "output") {
        return kDtLimiterOutput;
      }
      if (lineage.limiter == "t_end") {
        return kDtLimiterTEnd;
      }
      if (lineage.limiter == "max") {
        return kDtLimiterMax;
      }
      return kDtLimiterUnknown;
    }();
    TENRYU_ASSERT(out2[0] == dt_chosen,
                  "dt controller device/host mismatch (dt)");
    TENRYU_ASSERT(static_cast<int>(out2[1]) == host_limiter_id,
                  "dt controller device/host mismatch (limiter)");
  }

  static const bool dt_lineage_enabled = []() {
    const char* v = std::getenv("TENRYU_DT_LINEAGE");
    return v != nullptr && std::string(v) == "1";
  }();
  if (dt_lineage_enabled) {
    if (cfg.numerics.hydro.enabled &&
        (cfg.main.dimension == "1D_SPH" || cfg.main.dimension == "1D_CYL" ||
         cfg.main.dimension == "2D_RZ") &&
        state.mesh.node_r != nullptr) {
      const hydro::HydroDtDiagnostics hydro_min_diag =
          hydro::compute_dt_hydro_diagnostics(state, cfg, true);
      copy_hydro_min_contributor(hydro_min_diag.min_contributor, lineage);
      lineage.dt_hydro_subzonal = hydro_min_diag.subzonal_pressure_dt;
      const hydro::HydroDtArgmin hydro_argmin =
          hydro::compute_dt_hydro_argmin(state, cfg);
      lineage.hydro_argmin_cell = hydro_argmin.argmin_cell;
      lineage.hydro_argmin_sqrt_area = hydro_argmin.sqrt_area_at_argmin;
      lineage.hydro_argmin_cs = hydro_argmin.cs_at_argmin;
      lineage.hydro_argmin_i = hydro_argmin.argmin_i;
      lineage.hydro_argmin_j = hydro_argmin.argmin_j;
      lineage.hydro_argmin_dt_at_cell = hydro_argmin.dt;
      lineage.hydro_argmin_rho = hydro_argmin.rho_at_argmin;
      lineage.hydro_argmin_u_z = hydro_argmin.u_z_at_argmin;
      lineage.dt_hydro_acoustic = hydro::compute_dt_hydro_acoustic(state, cfg);
      const hydro::AxisMarginDtArgmin axis_margin_argmin =
          hydro::compute_dt_hydro_axis_margin_argmin(state, cfg);
      lineage.dt_hydro_axis_margin = axis_margin_argmin.dt;
      lineage.dt_hydro_tri_fan_center = axis_margin_argmin.center_cfl.dt;
      lineage.tri_fan_center_cfl_cell_id =
          axis_margin_argmin.center_cfl.cell_id;
      lineage.tri_fan_center_cfl_i = axis_margin_argmin.center_cfl.i;
      lineage.tri_fan_center_cfl_j = axis_margin_argmin.center_cfl.j;
      lineage.tri_fan_center_cfl_h = axis_margin_argmin.center_cfl.h;
      lineage.tri_fan_center_cfl_h_2ap = axis_margin_argmin.center_cfl.h_2ap;
      lineage.tri_fan_center_cfl_min_altitude =
          axis_margin_argmin.center_cfl.min_altitude;
      lineage.tri_fan_center_cfl_c_eff = axis_margin_argmin.center_cfl.c_eff;
      lineage.tri_fan_center_cfl_q_over_p =
          axis_margin_argmin.center_cfl.q_over_p;
      lineage.tri_fan_center_cfl_nverts =
          axis_margin_argmin.center_cfl.nverts;
      for (int k = 0; k < 8; ++k) {
        lineage.tri_fan_center_cfl_r8[k] =
            axis_margin_argmin.center_cfl.r8[k];
        lineage.tri_fan_center_cfl_z8[k] =
            axis_margin_argmin.center_cfl.z8[k];
      }
      for (int k = 0; k < 4; ++k) {
        lineage.tri_fan_center_cfl_r4[k] =
            lineage.tri_fan_center_cfl_r8[k];
        lineage.tri_fan_center_cfl_z4[k] =
            lineage.tri_fan_center_cfl_z8[k];
      }
      lineage.tri_fan_center_cfl_block_id =
          axis_margin_argmin.center_cfl.block_id;
      if (cfg.numerics.hydro.volume_rate_cfl_enabled) {
        const hydro::VolumeRateDtArgmin volume_rate_argmin =
            hydro::compute_dt_hydro_volume_rate_argmin(state, cfg);
        lineage.dt_hydro_volume_rate = volume_rate_argmin.dt;
        lineage.volume_rate_argmin_cell = volume_rate_argmin.argmin_cell;
        lineage.volume_rate_argmin_frac_rate =
            volume_rate_argmin.frac_rate_at_argmin;
      }
    }
    if (cfg.numerics.conduction.enabled &&
        (cfg.main.dimension == "1D_SPH" || cfg.main.dimension == "1D_CYL" ||
         cfg.main.dimension == "2D_RZ") &&
        state.mesh.node_r != nullptr) {
      const hydro::ConductionDtArgmin cond_argmin =
          hydro::compute_dt_conduction_argmin(state, cfg, eos_ctx);
      lineage.cond_argmin_cell = cond_argmin.argmin_cell;
      lineage.cond_argmin_deff = cond_argmin.deff_at_argmin;
      lineage.cond_argmin_dl = cond_argmin.dl_at_argmin;
    }
  }

  if (dt_chosen < cfg.numerics.dt.min_s) {
    const double remaining = t_end - state.t;
    const double eps =
        1.0e-14 * std::max(std::abs(t_end), 1.0);
    if (remaining > eps) {
      core::log_warning("[dt_floor_abort] step=" + std::to_string(state.step) +
                        " t=" + format_sci(state.t) +
                        " remaining=" + format_sci(remaining) +
                        " dt_min=" + format_sci(cfg.numerics.dt.min_s) +
                        " limiter=" + lineage.limiter +
                        " dt_hydro=" + format_sci(dt_hydro) +
                        " dt_cond=" + format_sci(dt_cond) +
                        " dt_burn=" + format_sci(dt_burn) +
                        " dt_hot_e=" + format_sci(dt_hot_e) +
                        " dt_rad=" + format_sci(dt_rad_val) +
                        " dt_growth=" + format_sci(dt_growth) +
                        " dt_output=" + format_sci(dt_output) +
                        " dt_hydro_tri_fan_center=" +
                        format_sci(lineage.dt_hydro_tri_fan_center) +
                        " dt_hydro_corner_j_predict=" +
                        format_sci(lineage.dt_hydro_corner_j_predict) +
                        " dt_chosen=" + format_sci(dt_chosen) +
                        " hydro_min_term=" + lineage.hydro_min_other_term +
                        " hydro_min_cell=" +
                        std::to_string(lineage.hydro_min_cell) +
                        " hydro_min_cell_raw=" +
                        std::to_string(lineage.hydro_min_cell_raw) +
                        " hydro_min_raw_dt=" +
                        format_sci(lineage.hydro_min_raw_dt));
      core::log_warning("[dt_floor_abort_hydro_detail] cell=" +
                        std::to_string(lineage.hydro_argmin_cell) +
                        " i=" + std::to_string(lineage.hydro_argmin_i) +
                        " j=" + std::to_string(lineage.hydro_argmin_j) +
                        " sqrt_area=" +
                        format_sci(lineage.hydro_argmin_sqrt_area) +
                        " cs=" + format_sci(lineage.hydro_argmin_cs) +
                        " rho=" + format_sci(lineage.hydro_argmin_rho) +
                        " u_z=" + format_sci(lineage.hydro_argmin_u_z) +
                        " dt_hydro_acoustic=" +
                        format_sci(lineage.dt_hydro_acoustic) +
                        " dt_hydro_post_shock=" +
                        format_sci(lineage.dt_hydro_post_shock) +
                        " dt_hydro_edge_av=" +
                        format_sci(lineage.dt_hydro_edge_av) +
                        " dt_hydro_edge_accel=" +
                        format_sci(lineage.dt_hydro_edge_accel) +
                        " dt_hydro_subzonal=" +
                        format_sci(lineage.dt_hydro_subzonal) +
                        " dt_hydro_axis_margin=" +
                        format_sci(lineage.dt_hydro_axis_margin) +
                        " dt_hydro_volume_rate=" +
                        format_sci(lineage.dt_hydro_volume_rate) +
                        " dt_hydro_tri_fan_center=" +
                        format_sci(lineage.dt_hydro_tri_fan_center) +
                        " dt_hydro_corner_j_predict=" +
                        format_sci(lineage.dt_hydro_corner_j_predict));
      if (cfg.mesh.topology_scheme ==
              core::TopologyScheme::PENTAGON_BELT_SHELL &&
          lineage.tri_fan_center_cfl_cell_id < 0) {
        const std::string reason =
            cfg.numerics.hydro.center_cfl_scope ==
                    core::CenterCflScope::DISABLED
                ? "center_cfl_scope_disabled"
                : "no_eligible_center_cfl_cell";
        core::log_warning(
            "[dt_floor_abort_center_cfl_detail] SKIPPED "
            "topology=pentagon_belt_shell reason=" +
            reason);
      } else {
        std::string vertices;
        for (int k = 0; k < lineage.tri_fan_center_cfl_nverts; ++k) {
          vertices += " r" + std::to_string(k) + "=" +
                      format_sci(lineage.tri_fan_center_cfl_r8[k]) +
                      " z" + std::to_string(k) + "=" +
                      format_sci(lineage.tri_fan_center_cfl_z8[k]);
        }
        core::log_warning(
            "[dt_floor_abort_center_cfl_detail] stable_cell=" +
            std::to_string(lineage.tri_fan_center_cfl_cell_id) +
            " block_id=" +
            std::to_string(lineage.tri_fan_center_cfl_block_id) +
            " nverts=" +
            std::to_string(lineage.tri_fan_center_cfl_nverts) +
            " dt=" + format_sci(lineage.dt_hydro_tri_fan_center) +
            " h=" + format_sci(lineage.tri_fan_center_cfl_h) +
            " h_2ap=" + format_sci(lineage.tri_fan_center_cfl_h_2ap) +
            " min_altitude=" +
            format_sci(lineage.tri_fan_center_cfl_min_altitude) +
            " c_eff=" + format_sci(lineage.tri_fan_center_cfl_c_eff) +
            vertices);
      }
      if (dt_rad_diag.limiting_cell >= 0) {
        core::log_warning("[dt_floor_abort_rad_detail] cell=" +
                          std::to_string(dt_rad_diag.limiting_cell) +
                          " Te=" + format_sci(dt_rad_diag.limiting_Te) +
                          " rho=" + format_sci(dt_rad_diag.limiting_rho) +
                          " sigma_P=" + format_sci(dt_rad_diag.limiting_sigma_P) +
                          " beta=" + format_sci(dt_rad_diag.limiting_beta) +
                          " Cv_e=" + format_sci(dt_rad_diag.limiting_Cv_e) +
                          " dt_rad=" + format_sci(dt_rad_diag.dt_rad));
      }
    }
    if (defer_dt_floor_abort && remaining > eps) {
      // Caller opted in to handle the abort: it must either rescue the step
      // (e.g. central pseudo-core ring absorption) or re-raise this abort.
      lineage.dt_floor_abort_pending = true;
      lineage.dt_chosen = dt_chosen;
    } else {
      TENRYU_ASSERT(remaining <= eps,
                    "dt dropped below Numerics.dt.min_s");
      dt_chosen = remaining;
      lineage.dt_chosen = dt_chosen;
    }
  }
  if (std::isfinite(lineage.dt_hydro_tri_fan_center) &&
      lineage.dt_hydro_tri_fan_center > 0.0) {
    lineage.tri_fan_center_cfl_dt_global_over_dt_center =
        lineage.dt_chosen / lineage.dt_hydro_tri_fan_center;
  }

  if (cfg.main.verbosity == "verbose") {
    core::log_info("[dt_breakdown] step=" + std::to_string(state.step) +
                   " dt_hydro=" + format_sci(dt_hydro) +
                   " dt_cond=" + format_sci(dt_cond) +
                   " dt_burn=" + format_sci(dt_burn) +
                   " dt_hot_e=" + format_sci(dt_hot_e) +
                   " dt_rad=" + format_sci(dt_rad_val) +
                   " dt_growth=" + format_sci(dt_growth) +
                   " dt_max=" + format_sci(cfg.numerics.dt.max_s) +
                   " dt_chosen=" + format_sci(dt_chosen) +
                   " dt_winner=" + lineage.limiter +
                   " hydro_winner_cell=" +
                   std::to_string(lineage.hydro_argmin_cell) +
                   " hydro_winner_i=" +
                   std::to_string(lineage.hydro_argmin_i) +
                   " hydro_winner_j=" +
                   std::to_string(lineage.hydro_argmin_j) +
                   " hydro_winner_dt=" +
                   format_sci(lineage.hydro_argmin_dt_at_cell) +
                   " hydro_winner_dl=" +
                   format_sci(lineage.hydro_argmin_sqrt_area) +
                   " hydro_winner_cs=" +
                   format_sci(lineage.hydro_argmin_cs) +
                   " hydro_winner_rho=" +
                   format_sci(lineage.hydro_argmin_rho) +
                   " hydro_winner_u_z=" +
                   format_sci(lineage.hydro_argmin_u_z) +
                   " tri_fan_center_dt=" +
                   format_sci(lineage.dt_hydro_tri_fan_center) +
                   " corner_j_predict_dt=" +
                   format_sci(lineage.dt_hydro_corner_j_predict) +
                   " tri_fan_center_i=" +
                   std::to_string(lineage.tri_fan_center_cfl_i) +
                   " tri_fan_center_j=" +
                   std::to_string(lineage.tri_fan_center_cfl_j) +
                   " tri_fan_center_h=" +
                   format_sci(lineage.tri_fan_center_cfl_h) +
                   " tri_fan_center_c_eff=" +
                   format_sci(lineage.tri_fan_center_cfl_c_eff) +
                   " tri_fan_center_q_over_p=" +
                   format_sci(lineage.tri_fan_center_cfl_q_over_p));
    if (dt_rad_diag.limiting_cell >= 0) {
      core::log_info("[dt_rad_detail] cell=" + std::to_string(dt_rad_diag.limiting_cell) +
                     " Te=" + format_sci(dt_rad_diag.limiting_Te) +
                     " rho=" + format_sci(dt_rad_diag.limiting_rho) +
                     " sigma_P=" + format_sci(dt_rad_diag.limiting_sigma_P) +
                     " beta=" + format_sci(dt_rad_diag.limiting_beta) +
                     " Cv_e=" + format_sci(dt_rad_diag.limiting_Cv_e) +
                     " dt_rad=" + format_sci(dt_rad_diag.dt_rad));
    }
  }

  if (reanchor != nullptr) {
    if (reanchor->pending) {
      reanchor->pending = false;
      double candidate = lineage.dt_hydro;
      const double other_bounds[] = {
          lineage.dt_cond,     lineage.dt_burn,  lineage.dt_hot_e,
          lineage.dt_visc,     lineage.dt_rad,   lineage.dt_output,
          lineage.dt_remaining, lineage.dt_max_namelist};
      for (const double bound : other_bounds) {
        if (bound > 0.0 && std::isfinite(bound) && bound < candidate) {
          candidate = bound;
        }
      }
      if (reanchor->pre_spike_dt_chosen > 0.0 &&
          reanchor->pre_spike_dt_chosen < candidate) {
        candidate = reanchor->pre_spike_dt_chosen;
      }
      if (std::isfinite(candidate) && candidate > 0.0 &&
          candidate > lineage.dt_chosen) {
        lineage.dt_chosen = candidate;
        lineage.dt_uncapped_by_contact =
            std::max(lineage.dt_uncapped_by_contact, candidate);
        lineage.limiter = "rezone_reanchor";
      }
    }
    if (lineage.limiter != "hydro" && std::isfinite(lineage.dt_chosen) &&
        lineage.dt_chosen > 0.0) {
      reanchor->pre_spike_dt_chosen = lineage.dt_chosen;
    }
  }

  if (dt_lineage_enabled) {
    write_dt_lineage_jsonl(state, lineage);
  }
  return lineage;
}

double compute_dt(const core::State& state,
                  const core::Config& cfg,
                  const double t_end) {
  return compute_dt_lineage(state, cfg, t_end).dt_chosen;
}

void Driver::set_restart_photon_pool(radiation::PhotonPool&& pool) {
  restart_pool_ = std::move(pool);
}

void Driver::run(core::State& state, const core::Config& cfg_input) {
  core::Config cfg = cfg_input;
  resolve_temperature_model(cfg, state);
  assert_driver_retry_supported(cfg);
  io::OutputManager out;
  int io_rank = 0;
#if TENRYU_ENABLE_MPI
  {
    int mpi_up = 0;
    MPI_Initialized(&mpi_up);
    if (mpi_up != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &io_rank);
    }
  }
#endif
  out.init(cfg, io_rank);
  run(state, cfg, out);
}

void Driver::run(core::State& state,
                 const core::Config& cfg_input,
                 io::OutputManager& out) {
  core::Config cfg = cfg_input;
  resolve_temperature_model(cfg, state);
  assert_driver_retry_supported(cfg);
  const double fixed_dt = i1b_fixed_dt();
  if (fixed_dt > 0.0 && fixed_dt < cfg.numerics.dt.min_s) {
    throw core::namelist::ConfigError(
        "TENRYU_I1B_FIXED_DT must be >= Numerics.dt.min_s");
  }

  const bool dim_ok =
      (cfg.main.dimension == "1D_SPH" && cfg.main.dim == 1) ||
      (cfg.main.dimension == "1D_CYL" && cfg.main.dim == 1) ||
      (cfg.main.dimension == "2D_RZ" && cfg.main.dim == 2);
  TENRYU_ASSERT(dim_ok,
                "Config main.dimension/main.dim mismatch "
                "(expected 1D_SPH/1D_CYL<->1 or 2D_RZ<->2)");
  TENRYU_ASSERT(state.mesh.dim == cfg.main.dim,
                "State mesh dimension must match Config.main.dim");

  const bool f09_corner_mass_fallback_enabled =
      cfg.main.dimension == "2D_RZ" && cfg.numerics.hydro.enabled;
  if (f09_corner_mass_fallback_enabled) {
    hydro::rz::corner_mass_fallback_run_start();
  }
  hydro::build_axis_core_disk(state, cfg);
  // Restart intentionally re-baselines band-ALE A0 and runtime state from checkpoint geometry.
  hydro::band_ale::run_start(state, cfg);
  if (cfg.numerics.hydro.axis_projection.enabled) {
    hydro::axis_projection::run_start(cfg);
  }

  profile_observability_.reset_at_run_start(cfg);
  if (cfg.numerics.plic.enabled && initial_plic_interface_cells_observed_ > 0) {
    profile_observability_.interface_cells_observed +=
        initial_plic_interface_cells_observed_;
  }
  const bool production_audit_enabled =
      cfg.numerics.diagnostics.production_audit.enabled;
  ProfileObservability* profile_observability =
      (cfg.numerics.profile.icf_standard_ale.enabled || production_audit_enabled)
          ? &profile_observability_
          : nullptr;
  const bool icf_history_enabled = core::effective_diagnostics_icf_enabled(cfg);
  const bool conservation_history_enabled =
      core::effective_diagnostics_conservation_enabled(cfg);
  const bool ale_provenance_history_enabled =
      core::effective_diagnostics_ale_provenance_emission_enabled(cfg);
  const bool hotspot_gas_history_enabled =
      core::effective_diagnostics_hotspot_gas_enabled(cfg);
  const bool mesh_quality_min_enabled =
      cfg.numerics.diagnostics.mesh_quality_min.enabled;
  const bool ale_state_monitor_enabled =
      cfg.numerics.ale.runtime_controller.monitor_enabled;
  if (ale_state_monitor_enabled && ale_monitor_machine.has_value() &&
      state.t <= ale_monitor_last_t) {
    reset_runtime_ale_monitor_state(cfg);
  }
  if (cfg.numerics.profile.icf_standard_ale.enabled) {
    core::validate_icf_standard_ale_profile_config(cfg, profile_observability);
    log_profile_observability(*profile_observability, "run_start");
  }
  const auto finalize_profile_before_fatal = [&]() {
    hydro::corner_collapse_ledger_flush();
    hydro::ale::remap_dispatch_audit_flush();
    hydro::ale::gcl_audit_flush();
    hydro::core_clearance_controller_flush();
    hydro::shadow_homothety_flush();
    if (cfg.numerics.profile.icf_standard_ale.enabled) {
      log_profile_observability(*profile_observability, "before_fatal_abort");
      profile_observability->finalize_at_run_end(cfg, false);
    }
  };

  const std::string case_name =
      cfg.main.name.empty() ? std::string("unnamed") : cfg.main.name;
  diagnostics::HistoryWriter history_writer;
  history_writer.init(cfg, out.results_dir);
#if TENRYU_ENABLE_HDF5
  if (icf_history_enabled && !(profile_observability_.shell_initial_radius_cm() > 0.0)) {
    const std::string& restart_prefix =
        restart_checkpoint_prefix_.empty() ? cfg.main.restart_from
                                           : restart_checkpoint_prefix_;
    if (const auto history_path =
            derive_history_path_from_checkpoint_prefix(restart_prefix)) {
      if (const auto initial_radius =
              diagnostics::read_icf_initial_radius_from_history_file(
                  history_path->string())) {
        profile_observability_.set_shell_initial_radius_cm(*initial_radius);
      }
    }
  }
#endif

  hydro::Hydro1D hydro_1d;
  hydro::Hydro2D hydro_2d;
  hydro::MeshRegimeDeviceCache mesh_regime_cache;
  radiation::IMC imc;
  if (restart_pool_.has_value()) {
    imc.restore_photon_pool(std::move(*restart_pool_));
    restart_pool_.reset();
  }
  laser::LaserMesh laser_mesh;
  if (cfg.laser.enabled) {
    laser_mesh = laser::create_from_config(cfg);
  }
  std::vector<int> burn_fuel_mats;
  burn::PartitionTable burn_partition_table;
  core::DeviceBuffer<double> burn_r_node_dev;
  core::DeviceBuffer<double> burn_vol_dev;
  core::DeviceBuffer<double> burn_rho_dev;
  core::DeviceBuffer<double> burn_Te_dev;
  core::DeviceBuffer<double> burn_Ti_dev;
  core::DeviceBuffer<double> burn_ne_dev;
  core::DeviceBuffer<double> burn_S_birth_dev;
  // --- Nuclear burn init: fuel material indices, species inventories,
  //     LP partition table (init-time; no runtime Python) ---
  if (cfg.burn.scheme != "diffusion") {
    state.burn_diffusion_any = false;
  }
  if (cfg.burn.scheme != "mc") {
    state.burn_mc_any = false;
    state.burn_mc_live = 0;
  }
  if (cfg.burn.scheme != "diffusion" && cfg.burn.scheme != "mc") {
    state.E_burn_inflight = 0.0;
  }
  if (cfg.burn.enabled && state.mesh.dim == 2) {
    for (const auto& fname : cfg.burn.fuel_materials) {
      int found = -1;
      for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
        if (cfg.materials.materials[m].name == fname) {
          found = static_cast<int>(m);
          break;
        }
      }
      TENRYU_ASSERT(found >= 0, "Burn fuel material validation failed");
      burn_fuel_mats.push_back(found);
    }
    burn_partition_table = burn::build_partition_table(
        cfg.burn.x_D, cfg.burn.x_T, cfg.burn.x_He3);
    const int n_cells = static_cast<int>(state.rho.size());
    const std::size_t burn_inventory_size =
        static_cast<std::size_t>(n_cells) * burn::kNumSpecies;
    const bool burn_inventory_restored =
        state.burn_n_host.size() == burn_inventory_size;
    const bool burn_restart =
        !restart_checkpoint_prefix_.empty() || !cfg.main.restart_from.empty();
    if (burn_restart && state.burn_n_host.empty()) {
      throw std::runtime_error(
          "burn.enabled restart requires a checkpoint written by a burn-enabled run "
          "(hydro/burn_n_* datasets missing)");
    }
    if (!burn_inventory_restored) {
      const int n_mat = static_cast<int>(cfg.materials.materials.size());
      std::vector<double> vf_h(state.volFrac.size());
      state.volFrac.copy_to_host(vf_h.data());
      const double Abar_sp = cfg.burn.x_D * burn::species_A(burn::kD)
                           + cfg.burn.x_T * burn::species_A(burn::kT)
                           + cfg.burn.x_He3 * burn::species_A(burn::kHe3);
      // Host burn inventory stores specific inventories [1/g].
      state.burn_n_host.assign(burn_inventory_size, 0.0);
      for (int c = 0; c < n_cells; ++c) {
        double fuel_frac = 0.0;
        for (int m : burn_fuel_mats) fuel_frac += vf_h[c * n_mat + m];
        if (fuel_frac <= 0.0) continue;
        state.burn_n_host[c * burn::kNumSpecies + burn::kD] =
            cfg.burn.x_D * fuel_frac / (Abar_sp * core::constants::proton_mass);
        state.burn_n_host[c * burn::kNumSpecies + burn::kT] =
            cfg.burn.x_T * fuel_frac / (Abar_sp * core::constants::proton_mass);
        state.burn_n_host[c * burn::kNumSpecies + burn::kHe3] =
            cfg.burn.x_He3 * fuel_frac / (Abar_sp * core::constants::proton_mass);
      }
    }
    if (state.burn_rate_host.empty()) {
      state.burn_rate_host.assign(n_cells, 0.0);
    }
    if (state.burn_Q_e_host.empty()) {
      state.burn_Q_e_host.assign(n_cells, 0.0);
    }
    if (state.burn_Q_i_host.empty()) {
      state.burn_Q_i_host.assign(n_cells, 0.0);
    }
    if (state.burn_eps_cum_host.empty()) {
      state.burn_eps_cum_host.assign(n_cells, 0.0);
    }
  }
  // v1 MPI decomposition is r-slab only (docs/design/mpi_m18_20_20260717.md
  // §3): the flat cell index is r-major, so r-slab ownership is a contiguous
  // flat range and every kernel restriction is a uniform [begin, end) window.
  // 2D auto decomposition is forced to dims=[P, 1]; an explicit non-r-slab
  // request is rejected until v1.1.
  std::vector<int> decomposition_dims = cfg.parallel.decomposition.dims;
  {
    int probe_rank = 0;
    int probe_n_ranks = 1;
    parallel::Partition::query_world(&probe_rank, &probe_n_ranks);
    if (probe_n_ranks > 1 && cfg.main.dim == 2) {
      if (decomposition_dims.empty()) {
        decomposition_dims = {probe_n_ranks, 1};
        if (probe_rank == 0) {
          core::log_info("[parallel] v1 MPI forces r-slab decomposition: dims=[" +
                         std::to_string(probe_n_ranks) + ",1]");
        }
      } else {
        TENRYU_ASSERT(decomposition_dims.size() == 2 && decomposition_dims[1] == 1,
                      "v1 MPI supports r-slab decomposition only "
                      "(parallel.decomposition.dims=[P,1]); z-slab/cartesian "
                      "splits are deferred to v1.1");
      }
    }
  }
  const parallel::PartitionInfo part_info = parallel::Partition::build(
      state.mesh.topo.nr, std::max(state.mesh.topo.nz, 1),
      cfg.main.dim,
      cfg.parallel.decomposition.method,
      decomposition_dims,
      cfg.parallel.decomposition.min_cells_per_rank);
  reale_mode::validate_runtime_scope(cfg, part_info.n_ranks);
  if (part_info.n_ranks > 1) {
    const int nz_stride = std::max(part_info.global_nz, 1);
    state.owned_cell_begin = part_info.local_cell_range[0][0] * nz_stride;
    state.owned_cell_end = part_info.local_cell_range[0][1] * nz_stride;
    const int node_stride =
        cfg.main.dim == 2 ? (part_info.global_nz + 1) : 1;
    state.owned_node_begin = part_info.local_node_range[0][0] * node_stride;
    state.owned_node_end = part_info.local_node_range[0][1] * node_stride;
    // Host-side mesh validity checks must not FATAL on the stale non-owned
    // regions (design doc §3): restrict to owned + g ghost rows.
    const int n_cells_total = part_info.global_nr * nz_stride;
    const int ghost_span = part_info.ghost_layers * nz_stride;
    mesh::set_geometry_check_window(
        std::max(0, state.owned_cell_begin - ghost_span),
        std::min(n_cells_total, state.owned_cell_end + ghost_span));
  }
  parallel::Reduction reducer(part_info.n_ranks);
  mesh::ZReflectionMaps z_reflection_maps;
  bool z_reflection_audit_active = false;
  if (cfg.numerics.z_reflection.mode == "audit") {
    if (!state.mesh.topo.multiblock.has_value()) {
      z_reflection_maps.failure =
          "initial mesh has no MultiBlockTopology";
    } else {
      const auto initial_node_r = copy_field_to_host(state.x_r);
      const auto initial_node_z = copy_field_to_host(state.x_z);
      z_reflection_maps = mesh::build_z_reflection_maps(
          *state.mesh.topo.multiblock, initial_node_r, initial_node_z);
    }
    z_reflection_audit_active = z_reflection_maps.valid;
    if (!z_reflection_audit_active && part_info.rank == 0) {
      core::log_warning("[z-reflection] " + cfg.numerics.z_reflection.mode +
                        " unavailable: " +
                        z_reflection_maps.failure);
    }
  }
  hydro::ale::remap_dispatch_audit_run_start(state, &reducer, part_info.rank);
  hydro::ale::gcl_audit_run_start(&reducer, part_info.rank);
  hydro::core_clearance_controller_run_start(state, cfg, part_info.rank);
  hydro::shadow_homothety_run_start(state, cfg, part_info.rank);
  parallel::CommBuffers comm_buffers;
  comm_buffers.gpu_aware_mpi = parallel::detect_gpu_aware_mpi();
  bool persistent_loop_active = cfg.numerics.persistent_loop.enabled;
  static const bool persistent_loop_trace = [] {
    const char* v = std::getenv("TENRYU_PK_CHUNK_TRACE");
    return v && std::string(v) == "1";
  }();
  std::optional<radiation::PlanckTable> persistent_loop_planck;
  materials::IonmixOpacityDeviceView persistent_loop_nlte_opacity{};
  core::namelist::FrozenTable1DDeviceView persistent_laser_waveform{};
  if (persistent_loop_active) {
    if (!persistent_loop_supported_c1(state, cfg,
                                      cfg.laser.enabled ? &laser_mesh : nullptr)) {
      persistent_loop_active = false;
    } else {
      if (cfg.radiation.enabled) {
        persistent_loop_planck.emplace(
            radiation::build_planck_table_from_config(cfg));
        persistent_loop_nlte_opacity = persistent_loop_nlte_opacity_view(cfg);
      }
      if (cfg.laser.enabled) {
        if (persistent_loop_trace) {
          core::log_info("pk_host: setup waveform upload begin");
        }
        persistent_laser_waveform =
            core::namelist::upload_frozen_table_1d_device_view(
                state.laser_waveforms.front());
        if (persistent_loop_trace) {
          core::log_info("pk_host: setup waveform upload end");
        }
      }
      core::log_info("persistent_loop: C1/C2b/C3 megakernel active (chunk_steps=" +
                     std::to_string(cfg.numerics.persistent_loop.chunk_steps) +
                     ")");
    }
  }
  I1OperatorProfileDump i1_operator_profile_dump =
      make_i1_operator_profile_dump(cfg, out.output_dir, part_info.rank);
  if (i1_operator_profile_dump.enabled) {
    core::log_info("[op_profile] I1 2D_RZ operator profile dump enabled: " +
                   i1_operator_profile_dump.path.string());
  }
  verification::EscapeValveAudit escape_valve_audit;
  escape_valve_audit.initialize(cfg.numerics.diagnostics.production_audit,
                                out.output_dir,
                                state.mesh.topo.nz,
                                part_info.rank == 0);
  verification::PositivityTracker positivity_tracker;
  positivity_tracker.initialize(cfg.numerics.diagnostics.production_audit);
  std::size_t escape_valve_audit_processed_events = 0;
  const auto escape_valve_audit_history_path = [&]() {
    if (history_writer.enabled()) {
      return history_writer.path();
    }
    return (std::filesystem::path(out.results_dir) /
            (case_name + "_history.h5"))
        .string();
  };
  const auto write_escape_valve_audit_history = [&]() {
    if (escape_valve_audit.enabled() && part_info.rank == 0) {
      diagnostics::write_escape_valve_audit_history_file(
          escape_valve_audit_history_path(), escape_valve_audit.snapshot());
    }
  };
  const auto write_positivity_history =
      [&](const verification::PositivityScanResult& result) {
        if (positivity_tracker.enabled() && part_info.rank == 0) {
          diagnostics::write_positivity_history_file(
              escape_valve_audit_history_path(), result);
        }
      };
  const auto handle_escape_valve_audit_decision =
      [&](const verification::EscapeValveAuditDecision& decision) {
        if (!decision.fatal) {
          return;
        }
        out.set_termination_reason(decision.audit_status);
        write_escape_valve_audit_history();
        if (part_info.rank == 0) {
          out.write_run_info(state, cfg);
        }
        finalize_profile_before_fatal();
        const std::string message =
            decision.audit_status + ": " + decision.reason;
        TENRYU_ASSERT(false, message);
      };
  const auto handle_positivity_audit_decision =
      [&](const verification::PositivityAuditDecision& decision) {
        if (!decision.fatal) {
          return;
        }
        out.set_termination_reason(decision.audit_status);
        write_escape_valve_audit_history();
        if (part_info.rank == 0) {
          out.write_run_info(state, cfg);
        }
        finalize_profile_before_fatal();
        const std::string message =
            decision.audit_status + ": " + decision.reason;
        TENRYU_ASSERT(false, message);
      };
  const auto process_escape_valve_events_for_audit = [&]() {
    if (!escape_valve_audit.enabled()) {
      return;
    }
    auto& events = profile_observability_.escape_valve_events;
    if (escape_valve_audit_processed_events > events.size()) {
      escape_valve_audit_processed_events = 0;
    }
    for (std::size_t i = escape_valve_audit_processed_events;
         i < events.size();
         ++i) {
      const auto& event = events[i];
      const std::string flag_key =
          !event.flag_name.empty() ? event.flag_name : event.operator_inserted;
      if (!verification::escape_valve_flag_from_name(flag_key)) {
        continue;
      }
      handle_escape_valve_audit_decision(
          escape_valve_audit.record_event(make_escape_valve_audit_event(event)));
    }
    escape_valve_audit_processed_events = events.size();
    if (production_audit_enabled &&
        !cfg.numerics.profile.icf_standard_ale.enabled &&
        !ale_provenance_history_enabled) {
      events.clear();
      escape_valve_audit_processed_events = 0;
    }
  };
  if (cfg.numerics.ale.safe_backtrack_enabled) {
    hydro::ale::reset_safe_backtrack_lambda_distribution();
  }

  initialize_eos_fields_if_needed(state, cfg);
  if (state.mesh.dim == 1) {
    materials::zmoment_upload_tables(state, cfg);
  }
  hydro::HydroEOSContext eos_ctx;
  DriverRecloseContext reclose_ctx;
  eos_ctx.initialize(cfg);
  const std::size_t n = state.rho.size();
  if (eos_ctx.any_table ||
      ((!cfg.main.two_temperature ||
        cfg.numerics.hydro.wake_heat_flux_enabled) &&
       n > 0)) {
    if (state.cv_e.empty() && n > 0) {
      state.cv_e.reset(n);
    }
    if (state.cv_i.empty() && n > 0) {
      state.cv_i.reset(n);
    }
  }
  if (eos_ctx.any_table) {
    core::log_info("HydroEOSContext: table EOS active for " +
                   std::to_string(eos_ctx.n_materials) + " material(s)");
  }
  refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    switch (checkpoint_per_material_status_) {
      case io::PerMaterialCheckpointReadStatus::PresentEnabled:
        break;
      case io::PerMaterialCheckpointReadStatus::MissingGroupEnabled:
        core::populate_per_material_conserved_from_cell_mean(state, cfg);
        break;
      case io::PerMaterialCheckpointReadStatus::PresentEnabledIncomplete:
        throw std::runtime_error(
            "V22 checkpoint per-material data incomplete with "
            "numerics.materials.per_material_conservation_enabled=true");
      default:
        break;
    }
  }
  // --- Nuclear burn init: fuel material indices, species inventories,
  //     LP partition table (init-time; no runtime Python) ---
  if (cfg.burn.scheme != "diffusion") {
    state.burn_diffusion_any = false;
  }
  if (cfg.burn.scheme != "mc") {
    state.burn_mc_any = false;
    state.burn_mc_live = 0;
  }
  if (cfg.burn.scheme != "diffusion" && cfg.burn.scheme != "mc") {
    state.E_burn_inflight = 0.0;
  }
  if (cfg.burn.enabled && state.mesh.dim == 1) {
    for (const auto& fname : cfg.burn.fuel_materials) {
      int found = -1;
      for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
        if (cfg.materials.materials[m].name == fname) {
          found = static_cast<int>(m);
          break;
        }
      }
      TENRYU_ASSERT(found >= 0, "Burn fuel material validation failed");
      burn_fuel_mats.push_back(found);
    }
    burn_partition_table = burn::build_partition_table(
        cfg.burn.x_D, cfg.burn.x_T, cfg.burn.x_He3);
    const int n_cells = static_cast<int>(state.rho.size());
    const std::size_t burn_inventory_size =
        static_cast<std::size_t>(n_cells) * burn::kNumSpecies;
    const bool burn_inventory_restored =
        state.burn_n_host.size() == burn_inventory_size;
    const bool burn_restart =
        !restart_checkpoint_prefix_.empty() || !cfg.main.restart_from.empty();
    if (burn_restart && state.burn_n_host.empty()) {
      throw std::runtime_error(
          "burn.enabled restart requires a checkpoint written by a burn-enabled run "
          "(hydro/burn_n_* datasets missing)");
    }
    if (!burn_inventory_restored) {
      const int n_mat = static_cast<int>(cfg.materials.materials.size());
      std::vector<double> vf_h(state.volFrac.size());
      state.volFrac.copy_to_host(vf_h.data());
      const double Abar_sp = cfg.burn.x_D * burn::species_A(burn::kD)
                           + cfg.burn.x_T * burn::species_A(burn::kT)
                           + cfg.burn.x_He3 * burn::species_A(burn::kHe3);
      // Host burn inventory stores specific inventories [1/g].
      state.burn_n_host.assign(burn_inventory_size, 0.0);
      for (int c = 0; c < n_cells; ++c) {
        double fuel_frac = 0.0;
        for (int m : burn_fuel_mats) fuel_frac += vf_h[c * n_mat + m];
        if (fuel_frac <= 0.0) continue;
        state.burn_n_host[c * burn::kNumSpecies + burn::kD] =
            cfg.burn.x_D * fuel_frac / (Abar_sp * core::constants::proton_mass);
        state.burn_n_host[c * burn::kNumSpecies + burn::kT] =
            cfg.burn.x_T * fuel_frac / (Abar_sp * core::constants::proton_mass);
        state.burn_n_host[c * burn::kNumSpecies + burn::kHe3] =
            cfg.burn.x_He3 * fuel_frac / (Abar_sp * core::constants::proton_mass);
      }
    }
    if (state.burn_rate_host.empty()) {
      state.burn_rate_host.assign(n_cells, 0.0);
    }
    if (state.burn_Q_e_host.empty()) {
      state.burn_Q_e_host.assign(n_cells, 0.0);
    }
    if (state.burn_Q_i_host.empty()) {
      state.burn_Q_i_host.assign(n_cells, 0.0);
    }
    if (state.burn_eps_cum_host.empty()) {
      state.burn_eps_cum_host.assign(n_cells, 0.0);
    }
    if (cfg.burn.scheme == "diffusion" && !state.burn_Ng.empty()) {
      const std::size_t expected_burn_Ng_size =
          6U * static_cast<std::size_t>(cfg.burn.diffusion_groups) *
          static_cast<std::size_t>(n_cells);
      if (state.burn_Ng.size() != expected_burn_Ng_size) {
        throw std::runtime_error(
            "burn diffusion restart hydro/burn_Ng_slot* size does not match "
            "Burn.diffusion_groups*n_cells");
      }
    }
    if (cfg.burn.scheme == "mc") {
      if (burn_restart && !state.burn_mc_any) {
        throw std::runtime_error(
            "burn MC restart requires checkpoint hydro/burn_mc_* datasets");
      }
      const std::size_t mc_capacity =
          burn_mc_pool_capacity_size(n_cells, cfg.burn.mc_particles_per_cell);
      if (state.burn_mc_any) {
        if (state.burn_mc_live < 0 ||
            static_cast<std::size_t>(state.burn_mc_live) > mc_capacity) {
          throw std::runtime_error(
              "burn MC restart time_state/burn_mc_live is outside pool capacity");
        }
        const std::size_t live =
            static_cast<std::size_t>(state.burn_mc_live);
        if (state.burn_mc_r.size() != live ||
            state.burn_mc_mu.size() != live ||
            state.burn_mc_E.size() != live ||
            state.burn_mc_w.size() != live ||
            state.burn_mc_slot.size() != live ||
            state.burn_mc_alive.size() != live) {
          throw std::runtime_error(
              "burn MC restart hydro/burn_mc_* dataset sizes do not match "
              "time_state/burn_mc_live");
        }
      }
    }
  }
  state.radiation_device_flags = core::DeviceErrorFlags{};

  if (cfg.numerics.conduction.enabled &&
      cfg.numerics.conduction.solver == "hypre") {
    core::log_info("Conduction solver=hypre requested (initialization slot reserved)");
  }

  const bool physics_enabled = cfg.numerics.hydro.enabled ||
                               cfg.numerics.conduction.enabled ||
                               cfg.radiation.enabled ||
                               cfg.laser.enabled;
  const bool is_1d =
      (cfg.main.dimension == "1D_SPH" || cfg.main.dimension == "1D_CYL");
  const bool is_2d = (cfg.main.dimension == "2D_RZ");
  const bool r_momentum_source_audit_requested =
      cfg.numerics.diagnostics.r_momentum_source_audit && is_2d;
  bool r_momentum_source_audit_active = r_momentum_source_audit_requested;
  if (r_momentum_source_audit_requested) {
    std::vector<std::string> disabled_reasons;
    if (!cfg.numerics.hydro.enabled) {
      disabled_reasons.emplace_back("hydro disabled");
    }
    if (!hydro::compatible::compatible_force_work_enabled(cfg)) {
      disabled_reasons.emplace_back("legacy force path");
    }
    if (state.mesh.button_center && state.mesh.button_center->enabled) {
      disabled_reasons.emplace_back("button-center bypass");
    }
    if (cfg.numerics.hydro.hllc_z_flux_2d_rz) {
      disabled_reasons.emplace_back("HLLC early-return");
    }
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      disabled_reasons.emplace_back("multiblock mesh");
    }
    if (part_info.n_ranks > 1) {
      disabled_reasons.emplace_back("MPI multi-rank (node-sum ownership)");
    }
    if (!disabled_reasons.empty()) {
      r_momentum_source_audit_active = false;
      if (part_info.rank == 0) {
        std::ostringstream reason;
        for (std::size_t i = 0; i < disabled_reasons.size(); ++i) {
          if (i > 0U) {
            reason << ", ";
          }
          reason << disabled_reasons[i];
        }
        core::log_warning("[r-momentum-source-audit] disabled: " +
                          reason.str());
      }
    }
  }
  // v1 MPI limitations (redesign doc / mpi_m18d_laser_burn_spec.md §2):
  // CBET and hot-electron transport are global-mesh chord/ray algorithms
  // not yet rank-decomposed — fail loud instead of silently computing on
  // rank-inconsistent inputs. Candidates for consolidated-input redundant
  // execution in v1.1.
  if (part_info.n_ranks > 1 && cfg.laser.enabled) {
    TENRYU_ASSERT(!cfg.laser.cbet.enable,
                  "Laser CBET is not supported under MPI in v1 "
                  "(single-rank only; see mpi_m18d_laser_burn_spec.md)");
    TENRYU_ASSERT(!cfg.laser.hot_electron.enable,
                  "Laser hot-electron transport is not supported under "
                  "MPI in v1 (single-rank only; see "
                  "mpi_m18d_laser_burn_spec.md)");
  }
  if (part_info.n_ranks > 1) {
    // v1.1 deferral guards (redesign doc): these operators have not been
    // rank-decomposition-audited — fail loud instead of silently running
    // on windowed inputs they were not built for.
    TENRYU_ASSERT(cfg.numerics.conduction.nonlocal_model != "snb",
                  "SNB nonlocal conduction is not supported under MPI in "
                  "v1 (deferral list; redesign doc)");
    TENRYU_ASSERT(!cfg.numerics.hydro.plasma_viscosity.enabled,
                  "Braginskii plasma viscosity is not supported under MPI "
                  "in v1 (deferral list; redesign doc)");
    TENRYU_ASSERT(!cfg.numerics.materials.per_material_conservation_enabled,
                  "Per-material conservation (per-material EOS projection) "
                  "is not supported under MPI in v1: the 2D projector has "
                  "no rank decomposition (design doc §6m follow-up)");
    TENRYU_ASSERT(!state.mesh.topo.multiblock.has_value(),
                  "Multiblock meshes are not supported under MPI in v1: "
                  "the r-slab halo strips assume the structured "
                  "single-block index space (deferral list; equal-split "
                  "node-mass modes included)");
    TENRYU_ASSERT(cfg.numerics.hydro.time_integration !=
                      core::HydroTimeIntegration::MidpointV1,
                  "midpoint_v1 time integration is not supported under MPI "
                  "in v1 (merge seam: midpoint internals are not "
                  "rank-decomposed)");
  }
  bool warned_2t_collapse = false;
  bool warned_clamp_threshold = false;
  parallel::EmigrantBuffer emigrants;
  parallel::EmigrantBuffer immigrants;

  const auto h1d_dump7 = [&](const char* phase) {
    const char* dbg = std::getenv("TENRYU_H1D_DEBUG");
    if (dbg == nullptr || !is_1d) {
      return;
    }
    int dbg_rank = 0;
    if (const char* re = std::getenv("OMPI_COMM_WORLD_RANK")) {
      dbg_rank = std::atoi(re);
    }
    const int nc = static_cast<int>(state.rho.size());
    const auto dump_one = [&](const char* tag, const double* dptr) {
      if (dptr == nullptr || nc <= 0) {
        return;
      }
      std::vector<double> h(static_cast<std::size_t>(nc));
      if (cudaMemcpy(h.data(), dptr,
                     static_cast<std::size_t>(nc) * sizeof(double),
                     cudaMemcpyDeviceToHost) != cudaSuccess) {
        return;
      }
      static std::atomic<int> h1d_drv_seq{0};
      const std::string path = std::string(dbg) + "_" + phase + tag + "_q" +
                               std::to_string(h1d_drv_seq.fetch_add(1)) +
                               "_r" + std::to_string(dbg_rank) + ".bin";
      if (FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    };
    dump_one("ee", state.ee.data());
    dump_one("te", state.Te.data());
  };
  // Compute initial table-EOS sound speed for correct CFL at step 0.
  if (cfg.numerics.hydro.enabled && (is_1d || is_2d)) {
    if (is_1d) {
      hydro_1d.prepare_initial_sound_speed(state, cfg, &eos_ctx);
      h1d_dump7("d0");
    } else {
      hydro_2d.prepare_initial_sound_speed(state, cfg, &eos_ctx);
      // Corner masses (the compatible nodal-mass basis) must exist before
      // the initial snapshot/history record: otherwise step-0 outputs use a
      // different nodal mass than every subsequent step (no hydro/node_mass
      // at step 0 + a fake one-time offset in the history energy series).
      hydro_2d.prepare_initial_corner_mass(state, cfg);
      hydro_2d.apply_state_supply_boundary(state, cfg, &eos_ctx, false);
      hydro::reset_state_supply_all_tallies(state);
    }
  }
  if (hotspot_gas_history_enabled) {
    diagnostics::initialize_hotspot_gas_tracer(state, cfg);
  }

  if (physics_enabled) {
    core::log_info("[splitting] order=" + std::string(splitting_order_name(order_)) +
                   " sequence=" + describe_split_sequence(order_, cfg));
  }

  const bool energy_audit_enabled = i1b_energy_audit_enabled();
  const int energy_audit_every = i1b_energy_audit_every();
  emit_config_fingerprint_step0(state,
                                cfg,
                                case_name,
                                energy_audit_enabled,
                                part_info.rank);
  EnergyAuditLedger energy_audit_ledger;
  int f09_last_report_bbsw = 0;
  int f09_last_report_subpolygon = 0;
  const auto f09_fallback_line =
      [](const char* prefix,
         const hydro::rz::CornerMassFallbackRecorder& recorder) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(17)
            << prefix
            << " bbsw_fires=" << recorder.fire_count_bbsw
            << " subpoly_fires=" << recorder.fire_count_subpolygon
            << " first{cell=" << recorder.first.cell
            << " stage=" << recorder.first.stage
            << " orient=" << recorder.first.orient
            << " nan=0x" << std::hex << recorder.first.nan << std::dec
            << " vals=[" << recorder.first.vals[0]
            << "," << recorder.first.vals[1]
            << "," << recorder.first.vals[2]
            << "," << recorder.first.vals[3]
            << "," << recorder.first.vals[4] << "]}";
        return oss.str();
      };
  const auto maybe_report_f09_fallback = [&]() {
    if (!f09_corner_mass_fallback_enabled) {
      return;
    }
    const hydro::rz::CornerMassFallbackRecorder recorder =
        hydro::rz::corner_mass_fallback_copy_to_host();
    if (recorder.fire_count_bbsw == f09_last_report_bbsw &&
        recorder.fire_count_subpolygon == f09_last_report_subpolygon) {
      return;
    }
    f09_last_report_bbsw = recorder.fire_count_bbsw;
    f09_last_report_subpolygon = recorder.fire_count_subpolygon;
    if (part_info.rank == 0) {
      core::log_warning(
          f09_fallback_line("[f09-fallback]", recorder));
    }
  };

  int last_plot_step = -1;

  // Write initial snapshot (t=0) before main loop
  if (state.step == 0 && part_info.rank == 0) {
    out.write_snapshot(state, cfg, state.step, state.t, case_name, part_info.rank);
    out.write_run_info(state, cfg);
    core::log_info("[output] wrote initial snapshot (step=0, t=" + format_sci(state.t) + ")");
  }

  const auto next_output_time = [&]() {
    double t_next_output = std::numeric_limits<double>::infinity();
    if (cfg.output.plot_every_s > 0.0 && state.t_next_plot > state.t) {
      t_next_output = std::min(t_next_output, state.t_next_plot);
    }
    if (cfg.output.history_every_s > 0.0 && state.t_next_history > state.t) {
      t_next_output = std::min(t_next_output, state.t_next_history);
    }
    if (cfg.output.checkpoint_every_s > 0.0 &&
        state.t_next_checkpoint > state.t) {
      t_next_output = std::min(t_next_output, state.t_next_checkpoint);
    }
    return t_next_output;
  };

  const auto emit_due_outputs = [&](auto fill_history_snapshot) {
    const bool do_history = out.should_history(state.step, state.t, state, cfg);
    const bool do_dt_diagnostics =
        cfg.numerics.diagnostics.dt_breakdown_history_enabled;
    if ((do_history || do_dt_diagnostics) && history_writer.enabled()) {
      diagnostics::HistorySnapshot snapshot{};
      fill_history_snapshot(snapshot);
      if (part_info.rank == 0) {
        history_writer.append(state, snapshot);
      }
    }
    if (do_history) {
      maybe_report_f09_fallback();
    }

    flush_i1_operator_profile_dump(i1_operator_profile_dump);

    if (out.should_plot(state.step, state.t, state, cfg)) {
      if (part_info.rank == 0) {
        out.write_snapshot(state, cfg, state.step, state.t, case_name, part_info.rank);
        last_plot_step = state.step;
        out.write_run_info(state, cfg);
        core::log_info("[output] wrote snapshot (step=" +
                       std::to_string(state.step) +
                       ", t=" + format_sci(state.t) + ")");
      }
      update_next_output_time(state.t_next_plot, cfg.output.plot_every_s, state.t);
    }

    if (do_history) {
      update_next_output_time(state.t_next_history,
                              cfg.output.history_every_s,
                              state.t);
    }

    if (out.should_checkpoint(state.step, state.t, state, cfg)) {
      update_next_output_time(state.t_next_checkpoint,
                              cfg.output.checkpoint_every_s,
                              state.t);
      if (part_info.rank == 0) {
        out.write_checkpoint(state,
                             cfg,
                             imc.photon_pool(),
                             state.step,
                             state.t,
                             case_name,
                             part_info.rank);
        core::log_info("[output] wrote checkpoint (step=" +
                       std::to_string(state.step) +
                       ", t=" + format_sci(state.t) + ")");
      }
    }
  };

  if (persistent_loop_active && cfg.laser.enabled) {
    const bool persistent_lmesh_setup =
        state.mesh.dim == 1 && cfg.laser.mode == "raytrace_2d";
    if (persistent_lmesh_setup && persistent_loop_trace) {
      core::log_info("pk_host: setup lmesh prep/capacity begin");
    }
    prepare_persistent_laser_entry(state, cfg, laser_mesh);
    if (persistent_lmesh_setup && persistent_loop_trace) {
      core::log_info("pk_host: setup lmesh prep/capacity end");
    }
  }

  if (persistent_loop_active) {
    const bool trace = persistent_loop_trace;
    if (state.step == 0 && cfg.numerics.dt.initial_s > 0.0) {
      state.dt = cfg.numerics.dt.initial_s / cfg.numerics.dt.growth_factor;
      state.dt_growth_ref = state.dt;
    }
    while ((cfg.main.t_end - state.t) >
               1.0e-14 * std::max(std::abs(state.t),
                                   std::abs(cfg.main.t_end)) &&
           state.step < cfg.main.max_steps) {
      emit_due_outputs([&](diagnostics::HistorySnapshot& snapshot) {
        snapshot.energy = diagnostics::compute_energy_budget_1d(state);
      });
      update_next_output_time(state.t_next_plot, cfg.output.plot_every_s, state.t);
      update_next_output_time(state.t_next_history, cfg.output.history_every_s,
                              state.t);
      update_next_output_time(state.t_next_checkpoint,
                              cfg.output.checkpoint_every_s, state.t);

      const double t_next_output = next_output_time();

      const int remaining_steps = cfg.main.max_steps - state.step;
      const auto chunk = run_persistent_chunk(state,
                                              cfg,
                                              laser_mesh,
                                              persistent_loop_planck
                                                  ? persistent_loop_planck->device_view()
                                                  : radiation::PlanckTableDeviceView{},
                                              persistent_loop_nlte_opacity,
                                              persistent_laser_waveform,
                                              cfg.main.t_end,
                                              t_next_output,
                                              remaining_steps);
      TENRYU_ASSERT(chunk.error_code == 0, "persistent_loop: device error");
      log_progress_if_needed(state, cfg, reducer);
      TENRYU_ASSERT(chunk.steps_advanced > 0 || chunk.exit_reason != 0,
                    "persistent_loop: empty chunk without exit");
    }
    if (trace) {
      core::log_info("pk_tail: t=" + std::to_string(state.t) +
                     " t_next_plot=" + std::to_string(state.t_next_plot) +
                     " t_next_history=" +
                         std::to_string(state.t_next_history));
    }
    emit_due_outputs([&](diagnostics::HistorySnapshot& snapshot) {
      snapshot.energy = diagnostics::compute_energy_budget_1d(state);
    });
    return;
  }

  const bool verbose_phase_timing = (cfg.main.verbosity == "verbose");
  using Clock = std::chrono::steady_clock;
  auto ms = [](const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  ConservationTotals conservation_initial{};
  double conservation_momentum_scale_floor = 0.0;
  double conservation_residual_max_mass = 0.0;
  double conservation_residual_max_r_momentum = 0.0;
  double conservation_residual_max_z_momentum = 0.0;
  double vol_closure_residual_max = 0.0;
  double vol_closure_residual_max_mass = 0.0;
  double vol_closure_residual_max_internal_energy = 0.0;
  double vol_closure_residual_max_r_momentum = 0.0;
  double vol_closure_residual_max_z_momentum = 0.0;
  double r_momentum_source_audit_previous = 0.0;
  diagnostics::EnergyTotals compatible_gate_initial{};
  double compatible_gate_E0 = 0.0;
  if (production_audit_enabled) {
    conservation_initial = compute_conservation_totals_for_state(state, cfg);
    reduce_conservation_totals(conservation_initial, reducer);
    conservation_momentum_scale_floor =
        std::abs(conservation_initial.mass) * compute_initial_sound_speed_max(state, reducer);
  }
  if (r_momentum_source_audit_active) {
    r_momentum_source_audit_previous = reducer.allreduce_sum(
        hydro::compute_node_r_momentum_2d(state, cfg));
  }
  if (cfg.numerics.debug.trace_mesh_motion && is_2d &&
      cfg.numerics.hydro.enabled &&
      ((cfg.numerics.hydro.av_model == core::AvModel::CswEdge &&
        cfg.numerics.hydro.subzonal_pressure_enabled) ||
       cfg.numerics.hydro.av_model == core::AvModel::CswEdgeCsw98)) {
    compatible_gate_initial = compute_energy_totals_for_state(state, cfg);
    reduce_energy_totals(compatible_gate_initial, reducer);
    compatible_gate_E0 = compatible_gate_initial.E_int_e +
                         compatible_gate_initial.E_int_i +
                         compatible_gate_initial.E_kin;
  }

  int dt_floor_consecutive = 0;
  int dt_floor_stall_count = 0;
  tenryu::diagnostics::MeshDegeneracyForensicsState
      mesh_degeneracy_forensics_state;
  int ring7_seam_proactive_next_allowed_step = 0;
  DtReanchorState dt_reanchor;
  const auto mark_ring7_seam_rezone_request_step = [&](const int request_step) {
    const int cooldown_steps =
        env_nonnegative_int("TENRYU_I1B_RING7_REZONE_COOLDOWN_STEPS", 3);
    ring7_seam_proactive_next_allowed_step =
        std::max(ring7_seam_proactive_next_allowed_step,
                 request_step + cooldown_steps + 1);
  };

  h1d_dump7("d1");
  bool fixed_dt_logged = false;
  bool autopilot_fire_stop = false;
  while (state.t < cfg.main.t_end && state.step < cfg.main.max_steps) {
    static const std::pair<int, int> forced_ring_release = [] {
      const char* raw = std::getenv("TENRYU_I1B_RING_RELEASE_FORCE");
      if (raw == nullptr) {
        return std::pair<int, int>{0, 0};
      }
      const std::string value(raw);
      const std::size_t colon = value.find(':');
      const bool valid_shape =
          colon != std::string::npos && colon > 0 && colon + 1 < value.size() &&
          value.find(':', colon + 1) == std::string::npos;
      if (!valid_shape) {
        TENRYU_ASSERT(
            false,
            "TENRYU_I1B_RING_RELEASE_FORCE must be '<step>:<count>' with "
            "two positive integers");
        return std::pair<int, int>{0, 0};
      }
      const auto parse_positive_int = [](const std::string& token,
                                         int& parsed) {
        int result = 0;
        for (const char ch : token) {
          if (ch < '0' || ch > '9') {
            return false;
          }
          const int digit = ch - '0';
          if (result > (std::numeric_limits<int>::max() - digit) / 10) {
            return false;
          }
          result = result * 10 + digit;
        }
        if (result <= 0) {
          return false;
        }
        parsed = result;
        return true;
      };
      int step = 0;
      int count = 0;
      if (!parse_positive_int(value.substr(0, colon), step) ||
          !parse_positive_int(value.substr(colon + 1), count)) {
        TENRYU_ASSERT(
            false,
            "TENRYU_I1B_RING_RELEASE_FORCE must be '<step>:<count>' with "
            "two positive integers in the range [1, INT_MAX]");
        return std::pair<int, int>{0, 0};
      }
      return std::pair<int, int>{step, count};
    }();
    static bool forced_ring_release_fired = false;
    if (forced_ring_release.first > 0 && !forced_ring_release_fired &&
        state.step == forced_ring_release.first) {
      forced_ring_release_fired = true;
      int performed = 0;
      while (performed < forced_ring_release.second &&
             hydro::axis_core_disk_release_one_unit(state, cfg)) {
        ++performed;
      }
      core::log_info("[ring-release] FORCED step=" +
                     std::to_string(forced_ring_release.first) +
                     " requested=" +
                     std::to_string(forced_ring_release.second) +
                     " performed=" + std::to_string(performed));
    }
    const bool step_snapshot_wanted =
        cfg.numerics.hydro.driver_full_step_retry_enabled ||
        hydro::i1b_path_guard_enabled();
    const bool g31_transactional =
        hydro::core_clearance_controller_g31_active();
    if (step_snapshot_wanted && g31_transactional) {
      capture_driver_retry_snapshot(retry_snapshot_, state, cfg, nullptr);
    }
    hydro::core_clearance_controller_pre_lagrange_rezone(state, cfg);
    if (g31_transactional &&
        hydro::core_clearance_controller_g31_take_trial_reject()) {
      TENRYU_ASSERT(step_snapshot_wanted,
                    "G3.1 trial reject requires the step retry snapshot");
      restore_driver_retry_snapshot(state, cfg, retry_snapshot_, false, true,
                                    nullptr);
      core::log_warning("[g31] step=" + std::to_string(state.step) +
                        " trial remap rejected; pre-rezone state restored");
    }
    if (!fixed_dt_logged && fixed_dt > 0.0) {
      if (part_info.rank == 0) {
        core::log_info("[fixed-dt] forcing dt=" + format_sci(fixed_dt));
      }
      fixed_dt_logged = true;
    }
    if (state.mesh.logical == mesh::LogicalMesh2D::ConeShell) {
      throw core::namelist::ConfigError(
          "cone_shell stepping lands with a later stage; Stage C2 supports mesh build/init/output only");
    }
    if (state.mesh.logical == mesh::LogicalMesh2D::PolarInBox) {
      // The current scope stops before polar-only BC/axis/ALE machinery can
      // engage. The mesh build, initialization, and step-0 snapshot above are
      // supported; time integration is deferred to the BC refactor.
      throw core::namelist::ConfigError(
          "polar_in_box stepping is not yet supported (pending the BC refactor); the current scope supports mesh build/init/output only");
    }
    int retry_attempts = 0;
    int mesh_epoch_commits_this_step = 0;
    HydroStepResult last_hydro_result{};
    bool conduction_retry_requested = false;
    hydro::ConductionResult conduction_retry_result{};
    HydroStepResult retry_trigger_result{};
    double step_min_path_margin = std::numeric_limits<double>::infinity();
    int step_min_path_margin_cell = -1;
    GeometricRetryStagnationState geometric_retry_stagnation_state;
    double retry_dt_before = 0.0;
    double retry_dt_after = 0.0;
    double bcr_ladder_original_dt =
        std::numeric_limits<double>::quiet_NaN();
    double bcr_ladder_retry_dt =
        std::numeric_limits<double>::quiet_NaN();
    int bcr_ladder_path_guard_failures = 0;
    bool bcr_ladder_seed_retry_issued = false;
    std::optional<hydro::ale::RepairPlan> pending_repair_plan;
    int stage24_repair_generation = 0;
    int stage24_rung_reached = 0;
    bool stage24_av_eligible = false;
    const char* stage24_av_blocked_reason = "";
    std::string stage24_axis_variational_projection_engaged_via;
    bool stage24_axis_repair_tried_at_current_state = false;
    int stage24_repair_generation_this_step = 0;
    int measured_corner_j_last_failing_cell = -1;
    int measured_corner_j_last_failing_corner = -1;
    int measured_corner_j_same_failure_count = 0;
    bool stage24_pending_repair_state_sensitive = false;
    bool stage24_state_sensitive_repair_executed_this_step = false;
    bool stage24_repair_generation_cap_hit_logged = false;
    bool axis_band_applied_this_attempt = false;
    core::PendingTransaction pending_axis_band{};  // Layer-T §5: hard retry pending;
    // the scheduled controller path re-evaluates need per step and needs no soft pending.
    int same_dt_strategy_retries_attempted = 0;
    int same_dt_strategy_retries_succeeded = 0;
    bool same_dt_strategy_retry_pending_success = false;
    int bprime_attempts_this_step = 0;
    int bprime_same_signature_count = 0;
    RetryReferenceBarrierSignature bprime_last_signature{};
    bool bprime_have_last_signature = false;
    int bprime_quality_stagnation_count = 0;
    double bprime_last_quality = -std::numeric_limits<double>::infinity();
    bool ring7_seam_retry_requested_this_step = false;
    bool ring7_pole_cap_oracle_requested_this_step = false;
    bool shell_subcycle_committed_this_step = false;
    bool energy_audit_identity_skip_this_step = false;
    bool energy_audit_ale_fallback_this_step = false;
    static int maxmin_step = -1;
    static int maxmin_attempts_this_step = 0;

    if (step_snapshot_wanted && !g31_transactional) {
      capture_driver_retry_snapshot(retry_snapshot_, state, cfg, nullptr);
    }

    while (true) {
      hydro::entropy_ledger_step_begin(state, cfg);
      clear_i1_operator_profile_attempt(i1_operator_profile_dump);
      reset_dt_lineage_ale_context();
    g_dt_lineage_context.axis_variational_projection_engaged_via =
        stage24_axis_variational_projection_engaged_via;
    if (retry_attempts > 0) {
      set_dt_lineage_retry_context(retry_attempts,
                                   retry_trigger_result,
                                   retry_dt_before,
                                   retry_dt_after);
      set_dt_lineage_stage24_context(retry_trigger_result,
                                     stage24_repair_generation,
                                     stage24_rung_reached,
                                     dt_floor_stall_count,
                                     stage24_av_eligible,
                                     stage24_av_blocked_reason);
    }
    last_hydro_result = HydroStepResult{};
    conduction_retry_requested = false;
    conduction_retry_result = hydro::ConductionResult{};
    step_min_path_margin = std::numeric_limits<double>::infinity();
    step_min_path_margin_cell = -1;
    shell_subcycle_committed_this_step = false;
    const bool corner_j_source_budget_active =
        cfg.numerics.diagnostics.mesh_degeneracy_forensics.enabled &&
        cfg.numerics.diagnostics.mesh_degeneracy_forensics
            .corner_j_source_budget_enabled;
    if (corner_j_source_budget_active) {
      mesh_degeneracy_forensics_state
          .reset_corner_j_source_budget_phase_snapshots(state.step);
    }
    const auto t_step_start =
        verbose_phase_timing ? Clock::now() : Clock::time_point{};
    double step_hydro_ms = 0.0;
    double step_cond_ms = 0.0;
    double step_laser_ms = 0.0;
    double step_rad_ms = 0.0;
    double step_dt_ms = 0.0;
    double step_energy_ms = 0.0;
    double step_diag_ms = 0.0;
    double step_vol_closure_residual = 0.0;
    double r_momentum_source_impulse_step = 0.0;

    if (cfg.numerics.hydro.enabled) {
      update_hydro_active(state, cfg);
      hydro::reset_state_supply_step_tallies(state);
    }

    const bool active_repair_possible =
        is_2d && cfg.numerics.hydro.enabled && cfg.mesh.motion == "ale" &&
        cfg.numerics.ale.enabled &&
        cfg.numerics.hydro.driver_full_step_retry_enabled &&
        cfg.numerics.hydro.driver_retry_active_mesh_repair_enabled;
    hydro::CornerBalanceResult active_repair_balance{};
    if (active_repair_possible) {
      active_repair_balance = hydro::evaluate_corner_balance(
          state,
          cfg.numerics.hydro.driver_retry_corner_balance_threshold,
          &reducer,
          state.evacuated_cells.d_geometry_policy_exempt_cells.empty()
              ? nullptr
              : state.evacuated_cells.d_geometry_policy_exempt_cells.data());
      g_dt_lineage_context.retry_active_repair_enabled = true;
      g_dt_lineage_context.retry_active_repair_q_balance =
          active_repair_balance.min_q_balance;
      g_dt_lineage_context.retry_active_repair_first_failing_cell =
          active_repair_balance.first_failing_cell;
      if (!active_repair_balance.admissible) {
        core::log_warning(
            "[active-repair-stats] step=" + std::to_string(state.step) +
            " attempt=" + std::to_string(retry_attempts) +
            " q_balance=" + format_sci(active_repair_balance.min_q_balance) +
            " threshold=" +
            format_sci(cfg.numerics.hydro.driver_retry_corner_balance_threshold) +
            " first_cell=" +
            std::to_string(active_repair_balance.first_failing_cell) +
            " first_corner=" +
            std::to_string(active_repair_balance.first_failing_corner));
      }
    }

    if (physics_enabled) {
      TENRYU_ASSERT(state.mesh.node_r != nullptr,
                    "State fields not initialized: mesh.node_r is null");
      if (cfg.main.dimension == "2D_RZ") {
        TENRYU_ASSERT(state.mesh.node_z != nullptr,
                      "State fields not initialized: mesh.node_z is null for 2D_RZ");
      }
    }

    const auto t_dt_start = verbose_phase_timing ? Clock::now() : Clock::time_point{};
    DtLineage step_dt_lineage = compute_dt_lineage(
        state,
        cfg,
        cfg.main.t_end,
        "primary",
        &eos_ctx,
        /*defer_dt_floor_abort=*/true,
        &dt_reanchor);
    if (step_dt_lineage.dt_floor_abort_pending) {
      ++dt_floor_consecutive;
      // dt collapsed below Numerics.dt.min_s mid-run. If the collapse is
      // driven by an absorbable active central-core ring cell, arm a ring
      // absorption (executed at the start of this step's Lagrangian phase)
      // and run this one step at the dt floor; otherwise re-raise the abort.
      const int retry_target_cell =
          step_dt_lineage.hydro_min_cell_raw >= 0
              ? step_dt_lineage.hydro_min_cell_raw
              : step_dt_lineage.hydro_argmin_cell;
      if (hydro::central_pseudo_core::request_ring_absorption(
              state, cfg, retry_target_cell)) {
        core::log_warning(
            "[driver-retry] dt-floor collapse binding cell=" +
            std::to_string(retry_target_cell) + " (term=" +
            step_dt_lineage.hydro_min_other_term + ") acoustic_argmin=" +
            std::to_string(step_dt_lineage.hydro_argmin_cell) +
            "; ring absorption requested, clamping dt to dt.min_s for the "
            "absorption step");
        step_dt_lineage.dt_chosen =
            std::max(step_dt_lineage.dt_chosen, cfg.numerics.dt.min_s);
        step_dt_lineage.dt_uncapped_by_contact =
            std::max(step_dt_lineage.dt_uncapped_by_contact,
                     cfg.numerics.dt.min_s);
        step_dt_lineage.dt_floor_abort_pending = false;
      } else if (state.central_pseudo_core.built &&
                 state.central_pseudo_core.min_member_ring_count >
                     state.central_pseudo_core.member_ring_count) {
        // An absorption is ARMED but not yet executed (it runs at the start
        // of this step's Lagrangian phase via
        // maybe_absorb_first_active_ring). The repeat-pending guard makes
        // request_ring_absorption refuse here, but dying would discard the
        // pending cure (observed: emergency walk armed target=21, snapshot
        // restored, the next attempt's dt collapsed, the rescue request was
        // refused as "already pending", and the hard assert fired with the
        // cure one step away). Clamp to the floor and let the pending
        // absorption execute.
        core::log_warning(
            "[driver-retry] dt-floor collapse with a PENDING ring "
            "absorption (min_depth=" +
            std::to_string(
                state.central_pseudo_core.min_member_ring_count) +
            ", members=" +
            std::to_string(state.central_pseudo_core.member_ring_count) +
            "); clamping dt to dt.min_s for the absorption step");
        step_dt_lineage.dt_chosen =
            std::max(step_dt_lineage.dt_chosen, cfg.numerics.dt.min_s);
        step_dt_lineage.dt_uncapped_by_contact =
            std::max(step_dt_lineage.dt_uncapped_by_contact,
                     cfg.numerics.dt.min_s);
        step_dt_lineage.dt_floor_abort_pending = false;
      } else if (step_dt_lineage.limiter == "growth" &&
                 step_dt_lineage.dt_hydro >= cfg.numerics.dt.min_s) {
        // Post-rescue geometric recovery: the physical limiters are healthy
        // and only the growth cap drags dt below the floor. Clamp and let dt
        // grow back over the following steps.
        step_dt_lineage.dt_chosen =
            std::max(step_dt_lineage.dt_chosen, cfg.numerics.dt.min_s);
        step_dt_lineage.dt_uncapped_by_contact =
            std::max(step_dt_lineage.dt_uncapped_by_contact,
                     cfg.numerics.dt.min_s);
        step_dt_lineage.dt_floor_abort_pending = false;
      } else {
        if (dt_floor_consecutive >=
            cfg.numerics.dt.min_consecutive_steps) {
          history_writer.flush_pending();
          TENRYU_ASSERT(false, "dt dropped below Numerics.dt.min_s");
        }
        core::log_warning(
            "[dt_floor_transient] step=" + std::to_string(state.step) +
            " t=" + format_sci(state.t) +
            " dt=" + format_sci(step_dt_lineage.dt_chosen) +
            " consecutive=" + std::to_string(dt_floor_consecutive) + "/" +
            std::to_string(cfg.numerics.dt.min_consecutive_steps));
        step_dt_lineage.dt_chosen =
            std::max(step_dt_lineage.dt_chosen, cfg.numerics.dt.min_s);
        step_dt_lineage.dt_floor_abort_pending = false;
      }
    } else if (step_dt_lineage.dt_chosen >= cfg.numerics.dt.min_s) {
      dt_floor_consecutive = 0;
    }
    double dt = reducer.allreduce_min(step_dt_lineage.dt_chosen);
    double dt_ref =
        reducer.allreduce_min(step_dt_lineage.dt_uncapped_by_contact);
    if (verbose_phase_timing) {
      step_dt_ms += ms(t_dt_start, Clock::now());
    }
    TENRYU_ASSERT(dt > 0.0, "Driver produced non-positive dt");
    if (!std::isfinite(bcr_ladder_original_dt)) {
      bcr_ladder_original_dt = dt;
    }
    if (cfg.numerics.hydro.regime_aware_corner_j_guard_enabled) {
      mesh_regime_cache.invalidate();
    }

    const auto maybe_emit_velocity_history_phase =
        [&](const char* phase, const double phase_time) {
      const auto& velocity_history_cfg =
          cfg.numerics.diagnostics.mesh_degeneracy_forensics;
      if (!velocity_history_cfg.velocity_history_enabled || part_info.rank != 0) {
        return;
      }
      const int stride =
          std::max(velocity_history_cfg.velocity_history_sample_every_n_steps, 1);
      if ((state.step % stride) != 0) {
        return;
      }
      const int target_cell =
          velocity_history_cfg.velocity_history_target_cell_c >= 0
              ? velocity_history_cfg.velocity_history_target_cell_c
              : mesh_degeneracy_forensics_state.first_failure_cell_c;
      if (target_cell < 0) {
        return;
      }
      (void)tenryu::diagnostics::emit_velocity_history(
          state,
          cfg,
          target_cell,
          phase != nullptr ? std::string(phase) : std::string(),
          state.step,
          phase_time,
          dt,
          &mesh_degeneracy_forensics_state);
    };

    const auto run_axis_band_controller =
        [&](const bool evaluate_need,
            const int K_initial,
            const bool recompute_dt_after_success,
	            const char* recompute_phase) -> hydro::ale::AxisBandResult {
	      hydro::ale::AxisBandResult result;
	      if (!cfg.numerics.ale.axis_band_managed_remap_enabled ||
	          hydro::ale::ale_identity_mode_enabled(cfg)) {
	        return result;
	      }
      int K = K_initial;
      if (evaluate_need) {
        const auto decision =
            hydro::ale::evaluate_axis_band_need(state, cfg, &reducer);
        if (!decision.needed) {
          result.skip_reason = hydro::ale::AxisBandResult::SkipReason::NotNeeded;
          return result;
        }
        K = decision.recommended_K;
      }

      result = hydro::ale::apply_axis_band_managed_remap(
          state, cfg, part_info, &comm_buffers, &reducer, &eos_ctx, K);
      if (result.remap_succeeded) {
        axis_band_applied_this_attempt = true;
        stage24_state_sensitive_repair_executed_this_step = true;
        stage24_axis_variational_projection_engaged_via =
            "axis_band_managed_remap";
        g_dt_lineage_context.axis_variational_projection_engaged_via =
            stage24_axis_variational_projection_engaged_via;
        hydro::reset_volume_rate_cfl_history_after_ale(state);
        if (recompute_dt_after_success) {
          const double dt_before_axis_band = dt;
          const auto t_dt_axis_band_start =
              verbose_phase_timing ? Clock::now() : Clock::time_point{};
          step_dt_lineage =
              compute_dt_lineage(state,
                                 cfg,
                                 cfg.main.t_end,
                                 recompute_phase != nullptr
                                     ? recompute_phase
                                     : "post_axis_band_managed_remap",
                                 &eos_ctx);
          dt = reducer.allreduce_min(step_dt_lineage.dt_chosen);
          dt_ref =
              reducer.allreduce_min(step_dt_lineage.dt_uncapped_by_contact);
          if (verbose_phase_timing) {
            step_dt_ms += ms(t_dt_axis_band_start, Clock::now());
          }
          TENRYU_ASSERT(dt > 0.0,
                        "Driver produced non-positive dt after axis-band remap");
          if (std::isfinite(dt_before_axis_band) &&
              dt_before_axis_band > 0.0 &&
              dt < 0.5 * dt_before_axis_band) {
            core::log_warning(
                "[axis-band-controller] dt reduced by more than 2x after "
                "managed remap at step=" +
                std::to_string(state.step) + " dt_before=" +
                format_sci(dt_before_axis_band) + " dt_after=" +
                format_sci(dt));
          }
        }
      } else if (result.skip_reason ==
                 hydro::ale::AxisBandResult::SkipReason::AllKFailed) {
        core::log_warning("[axis-band-controller] all K failed at step=" +
                          std::to_string(state.step));
      }
      return result;
    };

	    if (cfg.numerics.ale.axis_band_managed_remap_enabled &&
	        !hydro::ale::ale_identity_mode_enabled(cfg) &&
	        !axis_band_applied_this_attempt) {
      (void)run_axis_band_controller(
          true,
          cfg.numerics.ale.axis_band_managed_remap_width,
          true,
          "post_axis_band_managed_remap");
    }

    // --- S2a deferred audit/scalar packet --------------------------------
    // 1D/2D single-rank: step-scoped audit/budget energy reductions launch
    // into one device arena and are read back with ONE packet D2H at the
    // step-end drain below. Values are bit-identical to the eager path:
    // same kernels, same launch geometry, same ascending long-double host
    // sums — only the blocking readbacks move. Multi-rank keeps the eager
    // path unchanged. TENRYU_DISABLE_SCALAR_PACK restores all eager reads;
    // any block-payload shortfall also falls back per capture.
    constexpr std::size_t kMaxDeferredPhaseSafetyRecords = 64;
    constexpr std::size_t kTemperatureAuditFlagCount = 6;
    enum class StepScalarSlot : std::size_t {
      PhaseSafetyBeforeMaxBegin = 0,
      PhaseSafetyAfterMaxBegin = kMaxDeferredPhaseSafetyRecords,
      RadiationOvershootBeforeMax = 2 * kMaxDeferredPhaseSafetyRecords,
      RadiationOvershootMaxRatio =
          2 * kMaxDeferredPhaseSafetyRecords + 1,
      Count = 2 * kMaxDeferredPhaseSafetyRecords + 2,
    };
    enum class StepViolationFlag : std::size_t {
      PhaseSafetyViolationBegin = 0,
      PhaseSafetyBeforeAuditBegin = kMaxDeferredPhaseSafetyRecords,
      PhaseSafetyAfterAuditBegin =
          kMaxDeferredPhaseSafetyRecords * (1 + kTemperatureAuditFlagCount),
      RadiationOvershootBeforeAuditBegin =
          kMaxDeferredPhaseSafetyRecords *
          (1 + 2 * kTemperatureAuditFlagCount),
      RadiationOvershootCount =
          kMaxDeferredPhaseSafetyRecords *
              (1 + 2 * kTemperatureAuditFlagCount) +
          kTemperatureAuditFlagCount,
      Count = kMaxDeferredPhaseSafetyRecords *
                  (1 + 2 * kTemperatureAuditFlagCount) +
              kTemperatureAuditFlagCount + 2,
    };
    struct WjDeferredArena {
      bool mode = false;
      unsigned char* d_packet = nullptr;
      unsigned char* h_packet = nullptr;
      double* d_scalars = nullptr;
      double* h_scalars = nullptr;
      int* d_flags = nullptr;
      int* h_flags = nullptr;
      double* d = nullptr;
      double* h = nullptr;
      int e_blocks = 0;
      int e_stride = 0;
      int r_blocks = 0;
      std::size_t capacity = 0;
      std::size_t used = 0;
      std::size_t header_bytes = 0;
    };
    WjDeferredArena wj_arena;
    const bool scalar_pack_disabled = [] {
      const char* value = std::getenv("TENRYU_DISABLE_SCALAR_PACK");
      return value != nullptr && std::atoi(value) != 0;
    }();
    if (!scalar_pack_disabled &&
        ((cfg.main.dimension == "2D_RZ") || state.mesh.dim == 1) &&
        part_info.n_ranks == 1 && !state.rho.empty()) {
      wj_arena.e_blocks = diagnostics::energy_reduce_blocks(
          static_cast<int>(state.rho.size()));
      const int e_quantities = state.mesh.dim == 1 ? 4 : 3;
      wj_arena.e_stride = e_quantities * wj_arena.e_blocks;
      wj_arena.r_blocks = fld_rad_energy_reduce_blocks(state, cfg);
      constexpr std::size_t kWjMaxEnergyCaptures = 320;
      constexpr std::size_t kWjMaxRadCaptures = 320;
      wj_arena.capacity =
          kWjMaxEnergyCaptures * static_cast<std::size_t>(wj_arena.e_stride) +
          kWjMaxRadCaptures *
              static_cast<std::size_t>(std::max(wj_arena.r_blocks, 0));
      if (wj_arena.e_blocks > 0 && wj_arena.capacity > 0) {
        const std::size_t scalar_bytes =
            static_cast<std::size_t>(StepScalarSlot::Count) * sizeof(double);
        const std::size_t flag_bytes =
            static_cast<std::size_t>(StepViolationFlag::Count) * sizeof(int);
        wj_arena.header_bytes = scalar_bytes + flag_bytes;
        const std::size_t packet_bytes =
            wj_arena.header_bytes + wj_arena.capacity * sizeof(double);
        wj_arena.d_packet = static_cast<unsigned char*>(
            core::device_scratch_acquire("driver:wj_audit_packet", packet_bytes));
        wj_arena.h_packet = static_cast<unsigned char*>(
            core::host_pinned_scratch_acquire("driver:wj_audit_packet_host",
                                              packet_bytes));
        wj_arena.d_scalars =
            reinterpret_cast<double*>(wj_arena.d_packet);
        wj_arena.h_scalars =
            reinterpret_cast<double*>(wj_arena.h_packet);
        wj_arena.d_flags =
            reinterpret_cast<int*>(wj_arena.d_packet + scalar_bytes);
        wj_arena.h_flags =
            reinterpret_cast<int*>(wj_arena.h_packet + scalar_bytes);
        wj_arena.d = reinterpret_cast<double*>(
            wj_arena.d_packet + wj_arena.header_bytes);
        wj_arena.h = reinterpret_cast<double*>(
            wj_arena.h_packet + wj_arena.header_bytes);
        const cudaError_t flags_reset_err = cudaMemsetAsync(
            wj_arena.d_flags, 0, flag_bytes);
        TENRYU_ASSERT(flags_reset_err == cudaSuccess,
                      "S2a scalar pack violation flag reset failed");
        wj_arena.mode = true;
      }
    }
    const auto wj_defer_energy_capture = [&]() -> long {
      if (!wj_arena.mode ||
          wj_arena.used + static_cast<std::size_t>(wj_arena.e_stride) >
              wj_arena.capacity) {
        return -1;
      }
      const std::size_t off = wj_arena.used;
      const int blocks =
          state.mesh.dim == 1
              ? diagnostics::compute_energy_totals_1d_to_slot(
                    state, wj_arena.d + off)
              : diagnostics::compute_energy_totals_2d_to_slot(
                    state, wj_arena.d + off);
      if (blocks != wj_arena.e_blocks) {
        return -1;
      }
      wj_arena.used += static_cast<std::size_t>(wj_arena.e_stride);
      return static_cast<long>(off);
    };
    const auto wj_defer_rad_capture = [&]() -> long {
      if (!wj_arena.mode || wj_arena.r_blocks <= 0 ||
          wj_arena.used + static_cast<std::size_t>(wj_arena.r_blocks) >
              wj_arena.capacity) {
        return -1;
      }
      const std::size_t off = wj_arena.used;
      const int blocks =
          compute_fld_rad_energy_total_device_to_slot(state, cfg,
                                                      wj_arena.d + off);
      if (blocks != wj_arena.r_blocks) {
        return -1;
      }
      wj_arena.used += static_cast<std::size_t>(wj_arena.r_blocks);
      return static_cast<long>(off);
    };
    struct DeferredPhaseSafetyRecord {
      std::string phase_name;
      bool before_staged = false;
      double before_host = 0.0;
      int step_id = 0;
      double t_eval = 0.0;
      double dt_eval = 0.0;
      bool completed = false;
    };
    std::vector<DeferredPhaseSafetyRecord> deferred_phase_safety_records;
    bool deferred_radiation_overshoot = false;
    const auto phase_scalar_index = [](const StepScalarSlot base,
                                       const std::size_t record) {
      return static_cast<std::size_t>(base) + record;
    };
    const auto phase_flag_index = [](const StepViolationFlag base,
                                     const std::size_t record) {
      return static_cast<std::size_t>(base) +
             record * kTemperatureAuditFlagCount;
    };
    const auto begin_phase_safety = [&](const bool stage_before,
                                        const double before_host) -> long {
      if (!wj_arena.mode ||
          (!cfg.numerics.safety.nan_fatal &&
           !cfg.numerics.safety.overshoot_fatal_enabled)) {
        return -1;
      }
      const std::size_t record = deferred_phase_safety_records.size();
      TENRYU_ASSERT(record < kMaxDeferredPhaseSafetyRecords,
                    "S2a phase safety scalar slot capacity exceeded");
      deferred_phase_safety_records.push_back(
          DeferredPhaseSafetyRecord{"", stage_before, before_host});
      if (stage_before) {
        const std::size_t scalar = phase_scalar_index(
            StepScalarSlot::PhaseSafetyBeforeMaxBegin, record);
        const std::size_t flags = phase_flag_index(
            StepViolationFlag::PhaseSafetyBeforeAuditBegin, record);
        compute_temperature_audit_device_to_slots(
            state, wj_arena.d_scalars + scalar, wj_arena.d_flags + flags);
      }
      return static_cast<long>(record);
    };
    const auto finish_phase_safety = [&](const long token,
                                         const char* phase_name,
                                         const double before_host,
                                         const int step_id,
                                         const double t_eval,
                                         const double dt_eval) {
      if (token < 0) {
        enforce_phase_safety(phase_name,
                             state,
                             cfg,
                             before_host,
                             step_id,
                             t_eval,
                             dt_eval);
        return;
      }
      const std::size_t record = static_cast<std::size_t>(token);
      TENRYU_ASSERT(record < deferred_phase_safety_records.size(),
                    "S2a phase safety token out of range");
      auto& deferred = deferred_phase_safety_records[record];
      deferred.phase_name = phase_name != nullptr ? phase_name : "";
      deferred.before_host = before_host;
      deferred.step_id = step_id;
      deferred.t_eval = t_eval;
      deferred.dt_eval = dt_eval;
      deferred.completed = true;

      const std::size_t after_scalar = phase_scalar_index(
          StepScalarSlot::PhaseSafetyAfterMaxBegin, record);
      const std::size_t before_flags = phase_flag_index(
          StepViolationFlag::PhaseSafetyBeforeAuditBegin, record);
      const std::size_t after_flags = phase_flag_index(
          StepViolationFlag::PhaseSafetyAfterAuditBegin, record);
      const std::size_t violation =
          static_cast<std::size_t>(
              StepViolationFlag::PhaseSafetyViolationBegin) +
          record;
      compute_temperature_audit_device_to_slots(
          state,
          wj_arena.d_scalars + after_scalar,
          wj_arena.d_flags + after_flags);
      compute_phase_safety_violation_device(
          deferred.before_staged
              ? wj_arena.d_scalars + phase_scalar_index(
                    StepScalarSlot::PhaseSafetyBeforeMaxBegin, record)
              : nullptr,
          deferred.before_staged ? wj_arena.d_flags + before_flags : nullptr,
          before_host,
          wj_arena.d_scalars + after_scalar,
          wj_arena.d_flags + after_flags,
          cfg.numerics.safety.nan_fatal,
          cfg.numerics.safety.overshoot_fatal_enabled,
          cfg.numerics.floors.Te,
          cfg.numerics.safety.overshoot_fatal,
          wj_arena.d_flags + violation);
    };
    struct OpEnergyCapture {
      diagnostics::OperatorResidualSnapshot snap{};
      long off = -1;
    };
    struct DeferredOpRecord {
      std::string name;
      OpEnergyCapture before;
      long after_off = -1;
      diagnostics::OperatorResidualSnapshot after_eager{};
      double delta_E_ext = 0.0;
      int step = 0;
      double time = 0.0;
    };
    std::vector<DeferredOpRecord> deferred_op_records;
    const auto t_energy_before_start =
        verbose_phase_timing ? Clock::now() : Clock::time_point{};
    diagnostics::EnergyTotals energy_before{};
    const long energy_before_off =
        wj_arena.mode ? wj_defer_energy_capture() : -1;
    if (energy_before_off < 0) {
      energy_before = compute_energy_totals_for_state(state, cfg);
    }
    if (verbose_phase_timing) {
      step_energy_ms += ms(t_energy_before_start, Clock::now());
    }
    const bool phase_resolved_energy_enabled =
        cfg.numerics.diagnostics.phase_resolved_energy;
    const bool cap_energy_audit_enabled = hydro::cap_energy_audit_enabled();
    const bool i1b_spurious_sensor_enabled =
        hydro::i1b_spurious_sensor_enabled();
    const int i1b_spurious_sensor_top_k =
        i1b_spurious_sensor_enabled ? hydro::i1b_spurious_sensor_top_k() : 0;
    const bool deterministic_radiation_mode =
        cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion ||
        cfg.radiation.mode == core::RadiationMode::SnTransport;
    // W-K gamma_r=4/3 radiation-compression coupling (verdict D2).
    // v1 scope: 1D + deterministic FLD only — enforced on cfg.radiation.mode:
    // hydro_coupling is a multigroup_diffusion-subtree key, so with the flipped
    // default it would otherwise leak to 1D SnTransport decks (unvalidated).
    const bool rad_gamma43_coupling_on =
        cfg.radiation.enabled &&
        cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion &&
        state.mesh.dim == 1 &&
        ((cfg.radiation.multigroup_diffusion.hydro_coupling == "gamma_r_43") ||
         rad_gamma::gamma_r_43_enabled_from_env());  // env stays diagnostic override
    const int rad_gamma43_n_groups = std::max(cfg.radiation.groups, 1);
    core::CellField1D rad_gamma43_vol_before;
    core::CellField1D rad_gamma43_p_r;
    core::CellField1D rad_gamma43_W_r;
    if (rad_gamma43_coupling_on) {
      rad_gamma43_vol_before.reset(state.rho.size());
      rad_gamma43_p_r.reset(state.rho.size());
      rad_gamma43_W_r.reset(state.rho.size());
    }
    // 1D deterministic radiation (FLD/SN) runs the FULL line redundantly on
    // every rank (design doc mpi_m18_20_20260717.md §4.3/§6h); the flags and
    // the line-Allgatherv helper live up here because the HYDRO callback
    // also needs them: the gamma_r work payment mutates rad_E through the
    // owned window only, so the paid line must be re-globalized from the
    // owners before any full-line consumer (the replicated solve, and the
    // next hydro half's radiation-pressure field) reads it.
    const bool replicated_radiation_1d =
        part_info.n_ranks > 1 && is_1d && deterministic_radiation_mode;
    const double replicated_tally_share =
        (replicated_radiation_1d && part_info.rank != 0) ? 0.0 : 1.0;
    // Allgatherv a per-cell line (per_cell entries per cell, cell-major)
    // from the owned cell strips; no-op unless MPI 1D is active. Callers:
    // the replicated radiation stage (self-gated), the gamma_r payment
    // (FLD-1D only), and the 1D laser input gather — every full-line
    // consumer of windowed matter.
    const auto allgatherv_1d_cell_line = [&](double* d_field,
                                             const int per_cell) {
      if (part_info.n_ranks <= 1 || !is_1d || d_field == nullptr ||
          per_cell <= 0) {
        return;
      }
      const int n_cells_1d = state.mesh.topo.n_cells;
      const int n_ranks = part_info.n_ranks;
      const int total = n_cells_1d * per_cell;
      std::vector<int> counts(n_ranks);
      std::vector<int> displs(n_ranks);
      const int base = n_cells_1d / n_ranks;
      const int rem = n_cells_1d % n_ranks;
      for (int p = 0; p < n_ranks; ++p) {
        counts[p] = (base + (p < rem ? 1 : 0)) * per_cell;
        displs[p] = (p * base + std::min(p, rem)) * per_cell;
      }
      std::vector<double> line(static_cast<std::size_t>(total));
      const cudaError_t d2h_err =
          cudaMemcpy(line.data(), d_field,
                     static_cast<std::size_t>(total) * sizeof(double),
                     cudaMemcpyDeviceToHost);
      TENRYU_ASSERT(d2h_err == cudaSuccess,
                    "1D cell-line allgatherv D2H failed");
      const int my_count = counts[part_info.rank];
      const int my_displ = displs[part_info.rank];
      std::vector<double> send(line.begin() + my_displ,
                               line.begin() + my_displ + my_count);
      reducer.allgatherv(send.data(), my_count, line.data(), counts.data(),
                         displs.data());
      const cudaError_t h2d_err =
          cudaMemcpy(d_field, line.data(),
                     static_cast<std::size_t>(total) * sizeof(double),
                     cudaMemcpyHostToDevice);
      TENRYU_ASSERT(h2d_err == cudaSuccess,
                    "1D cell-line allgatherv H2D failed");
    };
    // Node-line variant (n_cells+1 entries; disjoint node tiles aligned
    // to the cell tiles, the last rank also owning the outermost node —
    // same partition as the radiation-stage x_r gather).
    const auto allgatherv_1d_node_line = [&](double* d_field) {
      if (part_info.n_ranks <= 1 || !is_1d || d_field == nullptr) {
        return;
      }
      const int n_cells_1d = state.mesh.topo.n_cells;
      const int n_ranks = part_info.n_ranks;
      const int total = n_cells_1d + 1;
      std::vector<int> counts(n_ranks);
      std::vector<int> displs(n_ranks);
      const int base = n_cells_1d / n_ranks;
      const int rem = n_cells_1d % n_ranks;
      for (int p = 0; p < n_ranks; ++p) {
        counts[p] = (base + (p < rem ? 1 : 0)) + (p == n_ranks - 1 ? 1 : 0);
        displs[p] = p * base + std::min(p, rem);
      }
      std::vector<double> line(static_cast<std::size_t>(total));
      const cudaError_t d2h_err =
          cudaMemcpy(line.data(), d_field,
                     static_cast<std::size_t>(total) * sizeof(double),
                     cudaMemcpyDeviceToHost);
      TENRYU_ASSERT(d2h_err == cudaSuccess,
                    "1D node-line allgatherv D2H failed");
      const int my_count = counts[part_info.rank];
      const int my_displ = displs[part_info.rank];
      std::vector<double> send(line.begin() + my_displ,
                               line.begin() + my_displ + my_count);
      reducer.allgatherv(send.data(), my_count, line.data(), counts.data(),
                         displs.data());
      const cudaError_t h2d_err =
          cudaMemcpy(d_field, line.data(),
                     static_cast<std::size_t>(total) * sizeof(double),
                     cudaMemcpyHostToDevice);
      TENRYU_ASSERT(h2d_err == cudaSuccess,
                    "1D node-line allgatherv H2D failed");
    };
    // 2D r-slab analogs (OPEN-LASER-FAR): the 2D ray trace is replicated
    // per rank across the WHOLE mesh, so — exactly like the 1D branch
    // above — its matter/coordinate inputs must be owner-true everywhere,
    // not just in the owned+ghost window. r-slab flat cells (c = i*nz + j)
    // give each rank a contiguous span, so a plain Allgatherv over the
    // cell-tile-aligned windows re-globalizes the field. Node planes use
    // the disjoint cell-aligned tiling with the last rank owning the
    // outermost i-plane (same convention as the 1D node line).
    const auto allgatherv_2d_cell_field = [&](double* d_field,
                                              const int per_cell) {
      if (part_info.n_ranks <= 1 || !is_2d || d_field == nullptr ||
          per_cell <= 0) {
        return;
      }
      TENRYU_ASSERT(part_info.cart_dims[1] == 1,
                    "2D full-field allgatherv supports r-slab decompositions "
                    "only in v1 (laser trace replication)");
      const int nz = state.mesh.topo.nz;
      const int global_nr = part_info.global_nr;
      const int n_ranks = part_info.n_ranks;
      TENRYU_ASSERT(state.mesh.topo.n_cells == global_nr * nz,
                    "2D cell-field allgatherv requires the structured "
                    "c = i*nz + j layout (no multiblock)");
      const int total = global_nr * nz * per_cell;
      std::vector<int> counts(n_ranks);
      std::vector<int> displs(n_ranks);
      const int base = global_nr / n_ranks;
      const int rem = global_nr % n_ranks;
      for (int p = 0; p < n_ranks; ++p) {
        counts[p] = (base + (p < rem ? 1 : 0)) * nz * per_cell;
        displs[p] = (p * base + std::min(p, rem)) * nz * per_cell;
      }
      std::vector<double> line(static_cast<std::size_t>(total));
      const cudaError_t d2h_err =
          cudaMemcpy(line.data(), d_field,
                     static_cast<std::size_t>(total) * sizeof(double),
                     cudaMemcpyDeviceToHost);
      TENRYU_ASSERT(d2h_err == cudaSuccess,
                    "2D cell-field allgatherv D2H failed");
      const int my_count = counts[part_info.rank];
      const int my_displ = displs[part_info.rank];
      std::vector<double> send(line.begin() + my_displ,
                               line.begin() + my_displ + my_count);
      reducer.allgatherv(send.data(), my_count, line.data(), counts.data(),
                         displs.data());
      const cudaError_t h2d_err =
          cudaMemcpy(d_field, line.data(),
                     static_cast<std::size_t>(total) * sizeof(double),
                     cudaMemcpyHostToDevice);
      TENRYU_ASSERT(h2d_err == cudaSuccess,
                    "2D cell-field allgatherv H2D failed");
    };
    const auto allgatherv_2d_node_field = [&](double* d_field) {
      if (part_info.n_ranks <= 1 || !is_2d || d_field == nullptr) {
        return;
      }
      TENRYU_ASSERT(part_info.cart_dims[1] == 1,
                    "2D full-field allgatherv supports r-slab decompositions "
                    "only in v1 (laser trace replication)");
      const int nz = state.mesh.topo.nz;
      const int global_nr = part_info.global_nr;
      const int n_ranks = part_info.n_ranks;
      const int stride = nz + 1;
      const int total = (global_nr + 1) * stride;
      TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(total),
                    "2D node-field allgatherv requires the structured "
                    "(nr+1)x(nz+1) node layout (no multiblock)");
      std::vector<int> counts(n_ranks);
      std::vector<int> displs(n_ranks);
      const int base = global_nr / n_ranks;
      const int rem = global_nr % n_ranks;
      for (int p = 0; p < n_ranks; ++p) {
        counts[p] =
            ((base + (p < rem ? 1 : 0)) + (p == n_ranks - 1 ? 1 : 0)) * stride;
        displs[p] = (p * base + std::min(p, rem)) * stride;
      }
      std::vector<double> line(static_cast<std::size_t>(total));
      const cudaError_t d2h_err =
          cudaMemcpy(line.data(), d_field,
                     static_cast<std::size_t>(total) * sizeof(double),
                     cudaMemcpyDeviceToHost);
      TENRYU_ASSERT(d2h_err == cudaSuccess,
                    "2D node-field allgatherv D2H failed");
      const int my_count = counts[part_info.rank];
      const int my_displ = displs[part_info.rank];
      std::vector<double> send(line.begin() + my_displ,
                               line.begin() + my_displ + my_count);
      reducer.allgatherv(send.data(), my_count, line.data(), counts.data(),
                         displs.data());
      const cudaError_t h2d_err =
          cudaMemcpy(d_field, line.data(),
                     static_cast<std::size_t>(total) * sizeof(double),
                     cudaMemcpyHostToDevice);
      TENRYU_ASSERT(h2d_err == cudaSuccess,
                    "2D node-field allgatherv H2D failed");
    };
    long E_rad_before_off = -1;
    if (cfg.radiation.enabled && deterministic_radiation_mode &&
        wj_arena.mode) {
      E_rad_before_off = wj_defer_rad_capture();
    }
    double E_rad_before =
        (E_rad_before_off >= 0)
            ? 0.0
            : (cfg.radiation.enabled
                   ? (deterministic_radiation_mode
                          ? compute_fld_rad_energy_total_device(state, cfg)
                          : imc.census_energy())
                   : 0.0);
    const double rad_esc_before_total =
        cfg.radiation.enabled ? imc.escaped_energy_total() : 0.0;
    std::vector<double> energy_audit_r_before;
    std::vector<double> energy_audit_z_before;
    if (energy_audit_enabled) {
      energy_audit_r_before = copy_field_to_host(state.x_r);
      energy_audit_z_before = copy_field_to_host(state.x_z);
    }

    // W-J accounting fix: rad_E is an energy DENSITY the hydro operator never
    // touches, so the radiation-field energy change across a hydro half is
    // exactly sum(rad_E * dV) — a real dE_total contribution no operator
    // tally covers (the operator-split omission of radiation-field
    // compression work). Tallied read-only so epsilon_budget truly closes.
    double step_E_rad_mesh_advection = 0.0;
    struct DeferredRadMeshAdvection {
      long before_off = -1;
      long after_off = -1;
    };
    std::vector<DeferredRadMeshAdvection> deferred_rad_mesh_advection;
    double step_E_laser_in = 0.0;
    double step_E_burn_in = 0.0;
    double step_E_laser_esc = 0.0;
    double step_E_ra_dep = 0.0;
    double step_E_cbet_iaw = 0.0;
    double step_E_rad_esc = 0.0;
    double step_E_marshak_in = 0.0;
    double step_E_volume_in = 0.0;
    double step_E_numerical_loss = 0.0;
    double step_E_pdV_bdry = 0.0;
    double step_E_floor = 0.0;
    double step_E_redistribution_unresolved = 0.0;
    double cap_energy_audit_W_ext_step = 0.0;
    double cap_energy_audit_W_pq = 0.0;
    double cap_energy_audit_W_uF = 0.0;
    double cap_energy_audit_W_apex_pinned = 0.0;
    double cap_energy_audit_D_K = 0.0;
    bool cap_energy_audit_rezone_fired = false;
    hydro::I1BSpuriousSensorSummary i1b_hydro_sensor_step{};
    hydro::I1BSpuriousSensorSummary i1b_ale_sensor_step{};
    double step_holo_boundary_in = 0.0;
    double step_holo_boundary_out = 0.0;
    double step_holo_matter_delta = 0.0;
    double step_holo_source_balance_error = 0.0;
    const double prev_E_solver = state.E_solver;
    int step_clamp_count = 0;
    std::vector<laser::RayOutputData> step_ray_output;
    diagnostics::AleClosureAuditDiagnostics ale_closure_audit{};
    diagnostics::OperatorEnergyTracker operator_energy_tracker;
    operator_energy_tracker.enable(conservation_history_enabled);
    int conduction_operator_record_count = 0;
    const auto capture_operator_energy = [&]() {
      auto snapshot = diagnostics::capture_total_energy(state);
      reduce_operator_snapshot(snapshot, reducer);
      return snapshot;
    };
    const auto capture_operator_energy_cap = [&]() -> OpEnergyCapture {
      OpEnergyCapture cap;
      if (wj_arena.mode) {
        cap.off = wj_defer_energy_capture();
        if (cap.off >= 0) {
          return cap;
        }
      }
      cap.snap = capture_operator_energy();
      return cap;
    };
    const auto record_operator_energy =
        [&](const std::string& name,
            const OpEnergyCapture& before,
            const double delta_E_ext) {
      if (!operator_energy_tracker.enabled()) {
        return;
      }
      if (wj_arena.mode) {
        DeferredOpRecord rec;
        rec.name = name;
        rec.before = before;
        rec.after_off = wj_defer_energy_capture();
        if (rec.after_off < 0) {
          rec.after_eager = diagnostics::capture_total_energy(state);
          reduce_operator_snapshot(rec.after_eager, reducer);
        }
        rec.delta_E_ext = delta_E_ext;
        rec.step = state.step + 1;
        rec.time = state.t + dt;
        deferred_op_records.push_back(std::move(rec));
        return;
      }
      auto after = diagnostics::capture_total_energy(state);
      reduce_operator_snapshot(after, reducer);
      operator_energy_tracker.record(
          name, before.snap, after, delta_E_ext, state.step + 1, state.t + dt);
    };
    const auto log_ale_remap_floor_closure =
        [&](const char* phase, const hydro::ale::AleStepResult& ale_out) {
      double closure_values[2] = {
          ale_out.mass_floor_delta,
          ale_out.E_redistribution_unresolved,
      };
      reducer.allreduce_sum(closure_values, 2);
      const double mass_floor_delta = closure_values[0];
      const double E_redistribution_unresolved = closure_values[1];
      if (!(mass_floor_delta > 0.0)) {
        return;
      }
      diagnostics::EnergyTotals energy_totals =
          compute_energy_totals_for_state(state, cfg);
      reduce_energy_totals(energy_totals, reducer);
      const double E_total_internal = energy_totals.E_int_e + energy_totals.E_int_i;
      const double E_redistribution_unresolved_rel =
          E_redistribution_unresolved / std::max(E_total_internal, 1.0e-20);
      if (part_info.rank == 0) {
        core::log_warning(
            std::string("[ale-remap-floor-closure] step=") +
            std::to_string(state.step + 1) + " phase=" + phase +
            " mass_floor_delta=" + format_sci(mass_floor_delta) +
            " E_redistribution_unresolved=" +
            format_sci(E_redistribution_unresolved) +
            " E_redistribution_unresolved_rel=" +
            format_sci(E_redistribution_unresolved_rel));
      }
      // Endgame guard (verdict #5 Q4 / HANDOFF section 15): a remap mass-floor
      // injection means a cell is being crushed against a broken/degenerate
      // target (measured: 7 silent injections = +3% total mass at 2.5 ns,
      // then runaway absorption). Convert the silent rescue into the designed
      // one: arm a sentinel ring absorption (walks the macro boundary one
      // unit outward onto healthier mesh) and hard-stop if injections repeat
      // beyond a budget — an invalid run must die loudly, not accumulate.
      const bool floor_absorb_enabled = mass_floor_absorb_effective_enabled(cfg);
      if (floor_absorb_enabled) {
        static int floor_event_count = 0;
        ++floor_event_count;
        const int budget =
            env_nonnegative_int("TENRYU_I1B_MASS_FLOOR_MAX_EVENTS", 12);
        const bool armed =
            hydro::central_pseudo_core::request_macro_boundary_absorption(
                state, cfg);
        if (part_info.rank == 0) {
          core::log_warning(
              std::string("[ale-remap-floor-absorb] step=") +
              std::to_string(state.step + 1) +
              " event=" + std::to_string(floor_event_count) + "/" +
              std::to_string(budget) + " sentinel_absorption_armed=" +
              (armed ? "true" : "false"));
        }
        TENRYU_ASSERT(floor_event_count <= budget,
                      "ALE remap mass-floor injections exceeded the terminal-phase "
                      "budget (TENRYU_I1B_MASS_FLOOR_MAX_EVENTS) — the run is "
                      "no longer mass-conservative; aborting loudly instead "
                      "of accumulating");
      }
    };

    const auto refresh_void_mask_after_2d_ale = [&]() {
      const std::size_t n_cells = state.rho.size();
      const std::size_t n_mat = cfg.materials.materials.size();
      TENRYU_ASSERT(state.cell_is_void.size() == n_cells,
                    "ALE void mask update requires cell_is_void/rho size match");
      TENRYU_ASSERT(n_mat > 0, "ALE void mask update requires at least one material");
      const std::size_t expected = n_cells * n_mat;
      TENRYU_ASSERT(state.volFrac.size() == expected,
                    "ALE void mask update requires volFrac size == n_cells*n_mat");
      std::vector<double> host_vf(expected, 0.0);
      state.volFrac.copy_to_host(host_vf.data());
      constexpr double kFracTol = 1.0e-12;
      int void_mat = -1;
      for (std::size_t m = 0; m < n_mat; ++m) {
        if (cfg.materials.materials[m].is_void) {
          void_mat = static_cast<int>(m);
          break;
        }
      }
      bool volfrac_changed = false;
      bool hydro_active_changed = false;
      for (std::size_t c = 0; c < n_cells; ++c) {
        const std::size_t base = c * n_mat;
        if (state.cell_is_void[c] != 0U) {
          for (std::size_t m = 0; m < n_mat; ++m) {
            const double vf = (static_cast<int>(m) == void_mat) ? 1.0 : 0.0;
            if (host_vf[base + m] != vf) {
              host_vf[base + m] = vf;
              volfrac_changed = true;
            }
          }
          state.cell_is_void[c] = 1U;
          if (!state.hydro_active.empty()) {
            TENRYU_ASSERT(state.hydro_active.size() == n_cells,
                          "ALE void mask update requires hydro_active/rho size match");
            state.hydro_active[c] = 0;
            hydro_active_changed = true;
          }
          continue;
        }

        double nonvoid_sum = 0.0;
        for (std::size_t m = 0; m < n_mat; ++m) {
          if (!cfg.materials.materials[m].is_void) {
            nonvoid_sum += host_vf[base + m];
          }
        }
        state.cell_is_void[c] =
            static_cast<std::uint8_t>((nonvoid_sum <= kFracTol) ? 1U : 0U);
        if (state.cell_is_void[c] != 0U && !state.hydro_active.empty()) {
          TENRYU_ASSERT(state.hydro_active.size() == n_cells,
                        "ALE void mask update requires hydro_active/rho size match");
          state.hydro_active[c] = 0;
          hydro_active_changed = true;
        }
      }
      if (hydro_active_changed) {
        state.note_hydro_active_host_write();
      }
      if (volfrac_changed) {
        state.volFrac.copy_from_host(host_vf.data());
      }
    };

    const bool corner_j_ale_trigger_possible =
        is_2d && cfg.numerics.hydro.enabled && cfg.mesh.motion == "ale" &&
        cfg.numerics.ale.enabled &&
        cfg.numerics.hydro.corner_jacobian_ale_trigger_enabled;
    const bool pending_repair_possible =
        pending_repair_plan.has_value() && is_2d && cfg.numerics.hydro.enabled &&
        cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled &&
        pending_repair_plan->ale_mode != hydro::ale::AleMode::ScheduledDefault;
    const auto refresh_mesh_regime_cache_for_corner_guard =
        [&]() -> const hydro::CellRegime* {
          if (!cfg.numerics.hydro.regime_aware_corner_j_guard_enabled) {
            mesh_regime_cache.invalidate();
            return nullptr;
          }
          hydro::classify_mesh_regimes(state,
                                       dt,
                                       cfg.numerics.hydro.axis_guard_band_cells,
                                       mesh_regime_cache,
                                       nullptr,
                                       cfg.numerics.has_physical_rz_axis);
          return mesh_regime_cache.current();
        };
    const auto compute_corner_scale_for_ale_trigger =
        [&](const hydro::CellRegime* d_cell_regime) {
          const std::size_t n_cells =
              static_cast<std::size_t>(state.mesh.topo.nr) *
              static_cast<std::size_t>(state.mesh.topo.nz);
          const auto& inactive_mask =
              state.evacuated_cells.inactive_member_mask;
          const auto& contact_active_mask =
              state.evacuated_cells.contact_active_mask;
          TENRYU_ASSERT(inactive_mask.empty() || inactive_mask.size() == n_cells,
                        "corner-J ALE trigger requires evacuated inactive mask "
                        "size == n_cells");
          TENRYU_ASSERT(contact_active_mask.empty() ||
                            contact_active_mask.size() == n_cells,
                        "corner-J ALE trigger requires contact-active mask "
                        "size == n_cells");

          bool any_constraint_owned = false;
          for (std::size_t c = 0; c < n_cells; ++c) {
            if ((!inactive_mask.empty() && inactive_mask[c] != 0U) ||
                (!contact_active_mask.empty() &&
                 contact_active_mask[c] != 0U)) {
              any_constraint_owned = true;
              break;
            }
          }
          const auto evaluate = [&]() {
            return hydro::compute_corner_jacobian_trial_scale(
                state,
                dt,
                cfg.numerics.hydro.corner_jacobian_floor_eps,
                &reducer,
                d_cell_regime,
                cfg.numerics.hydro.regime_aware_corner_j_guard_enabled,
                cfg.numerics.hydro.axis_margin_guard_enabled,
                cfg.numerics.hydro.corner_jacobian_ale_trigger_scale,
                cfg.numerics.has_physical_rz_axis,
                !state.evacuated_cells.d_cell_axis_edge_collapsed.empty()
                    ? state.evacuated_cells.d_cell_axis_edge_collapsed.data()
                    : nullptr,
                !state.evacuated_cells.d_geometry_policy_exempt_cells.empty()
                    ? state.evacuated_cells.d_geometry_policy_exempt_cells.data()
                    : nullptr);
          };
          if (!any_constraint_owned) {
            return evaluate();
          }

          TENRYU_ASSERT(state.hydro_active.empty() ||
                            state.hydro_active.size() == n_cells,
                        "corner-J ALE trigger requires hydro_active size == n_cells");
          std::vector<std::int8_t> trigger_active = state.hydro_active.empty()
                                                        ? std::vector<std::int8_t>(
                                                              n_cells, 1)
                                                        : state.hydro_active;
          for (std::size_t c = 0; c < n_cells; ++c) {
            if ((!inactive_mask.empty() && inactive_mask[c] != 0U) ||
                (!contact_active_mask.empty() &&
                 contact_active_mask[c] != 0U)) {
              trigger_active[c] = 0;
            }
          }

          state.hydro_active.swap(trigger_active);
          try {
            auto result = evaluate();
            state.hydro_active.swap(trigger_active);
            return result;
          } catch (...) {
            state.hydro_active.swap(trigger_active);
            throw;
          }
        };
    if (corner_j_ale_trigger_possible || active_repair_possible ||
        pending_repair_possible) {
      hydro::CornerJacobianScaleResult corner_scale{};
      bool dt_predicate_fired = false;
      if (corner_j_ale_trigger_possible) {
        const hydro::CellRegime* d_cell_regime =
            refresh_mesh_regime_cache_for_corner_guard();
        corner_scale = compute_corner_scale_for_ale_trigger(d_cell_regime);
        dt_predicate_fired =
            cfg.numerics.hydro.regime_aware_corner_j_guard_enabled ||
                    cfg.numerics.hydro.axis_margin_guard_enabled
                ? corner_scale.trigger_recommended
                : corner_scale.scale <
                      cfg.numerics.hydro.corner_jacobian_ale_trigger_scale;
      }
      const bool static_predicate_fires_now =
          retry_active_mesh_repair_should_force(active_repair_possible,
                                                retry_attempts,
                                                active_repair_balance.admissible);
      const bool active_repair_force_this_attempt =
          static_predicate_fires_now && !pending_repair_possible;
	      if (!hydro::ale::ale_identity_mode_enabled(cfg) &&
	          (dt_predicate_fired || static_predicate_fires_now ||
	           pending_repair_possible)) {
        if (dt_predicate_fired) {
          core::log_warning(
              "[ale-stats] corner-J pre-hydro trigger: predicted_scale=" +
              format_sci(corner_scale.scale) + " threshold=" +
              format_sci(cfg.numerics.hydro.corner_jacobian_ale_trigger_scale) +
              " first_cell=" +
              std::to_string(corner_scale.first_trigger_cell >= 0
                                 ? corner_scale.first_trigger_cell
                                 : corner_scale.first_failing_cell) +
              " first_corner=" +
              std::to_string(corner_scale.first_trigger_corner >= 0
                                 ? corner_scale.first_trigger_corner
                                 : corner_scale.first_failing_corner) +
              " dt=" + format_sci(dt));
        }
        if (active_repair_force_this_attempt) {
          core::log_warning(
              "[active-repair] retry pre-hydro ALE: q_balance=" +
              format_sci(active_repair_balance.min_q_balance) + " threshold=" +
              format_sci(cfg.numerics.hydro.driver_retry_corner_balance_threshold) +
              " first_cell=" +
              std::to_string(active_repair_balance.first_failing_cell) +
              " first_corner=" +
              std::to_string(active_repair_balance.first_failing_corner) +
              " dt=" + format_sci(dt));
          if (profile_observability != nullptr) {
            profile_observability->note_class_c_fire(
                "hydro_driver_retry_active_mesh_repair");
          }
        }
        hydro::ale::AleRequest pending_request;
        const hydro::ale::AleRequest* pending_request_ptr = nullptr;
        const hydro::CellRegime* pending_regime_ptr = nullptr;
        if (pending_repair_possible) {
          pending_request.mode = pending_repair_plan->ale_mode;
          pending_request.force_rezone = true;
          pending_request.focus_cell = pending_repair_plan->focus_cell;
          pending_request.patch_radius_i = 2;
          pending_request.patch_radius_j = 2;
          if (pending_request.mode == hydro::ale::AleMode::AxisSpinePlusLocal) {
            pending_request.patch_radius_i =
                cfg.numerics.hydro.axis_guard_band_cells;
          }
          pending_request.source_regime = retry_trigger_result.regime;
          pending_request_ptr = &pending_request;
          if (cfg.numerics.hydro.regime_aware_corner_j_guard_enabled) {
            pending_regime_ptr = refresh_mesh_regime_cache_for_corner_guard();
          }
          core::log_warning(
              "[driver-retry] pending repair plan: attempt=" +
              std::to_string(pending_repair_plan->retry_attempt) +
              " mode=" + std::to_string(static_cast<int>(pending_request.mode)) +
              " focus_cell=" + std::to_string(pending_request.focus_cell));
        }
        g_dt_lineage_context.retry_active_repair_invoked =
            active_repair_force_this_attempt;
        const int active_repair_cell_id =
            active_repair_balance.first_failing_cell;
        EscapeValveEvent::CellState active_repair_pre_state{};
        if (active_repair_force_this_attempt &&
            profile_observability != nullptr) {
          active_repair_pre_state =
              capture_escape_valve_cell_state(state, active_repair_cell_id);
        }
        const bool capture_pre_hydro_ale_energy =
            operator_energy_tracker.enabled() ||
            (active_repair_force_this_attempt &&
             profile_observability != nullptr);
        // The escape-valve event consumes the snapshot VALUES mid-step, so
        // that sub-case captures eagerly even in deferred mode.
        const bool ale_pre_energy_values_needed_now =
            active_repair_force_this_attempt &&
            profile_observability != nullptr;
        OpEnergyCapture op_energy_before;
        if (capture_pre_hydro_ale_energy) {
          if (ale_pre_energy_values_needed_now) {
            op_energy_before.snap = capture_operator_energy();
          } else {
            op_energy_before = capture_operator_energy_cap();
          }
        }
        if (corner_j_source_budget_active) {
          mesh_degeneracy_forensics_state.snapshot_corner_j_source_budget_phase(
              state, "post_lagrange", false);
        }
        hydro::ale::AleDriverRetryContext retry_context;
        if (cfg.numerics.hydro.cascade_on_hydro_retry_enabled &&
            active_repair_force_this_attempt) {
          retry_context.reason =
              ale_retry_reason_from_hydro_reason(retry_trigger_result.reason);
          retry_context.active =
              retry_context.reason !=
              hydro::ale::AleDriverRetryContext::Reason::none;
          retry_context.post_ale_corner_balance_bad =
              !active_repair_balance.admissible;
          retry_context.first_cell = retry_trigger_result.first_failing_cell;
          retry_context.first_corner = retry_trigger_result.first_failing_corner;
        }
        const hydro::ale::AleDriverRetryContext* retry_context_ptr =
            retry_context.active ? &retry_context : nullptr;
        hydro::entropy_ledger_begin(
            state, cfg, hydro::EntropyLedgerStage::Ale);
        state.mesh.materialize_host_svec();
        const auto ale_out = hydro::ale::apply_ale_with_request(
            state,
            cfg,
            pending_request_ptr,
            pending_regime_ptr,
            part_info,
            &comm_buffers,
            &reducer,
            &eos_ctx,
            dt,
            true,
            pending_repair_possible
                ? "driver retry repair plan"
                : (active_repair_force_this_attempt
                       ? "retry: corner Jacobian balance below threshold"
                       : "corner-J predicted trial scale below trigger threshold"),
            profile_observability,
            retry_context_ptr);
        hydro::entropy_ledger_end(
            state, cfg, hydro::EntropyLedgerStage::Ale);
        energy_audit_identity_skip_this_step =
            energy_audit_identity_skip_this_step ||
            ale_out.energy_audit_identity_skip;
        energy_audit_ale_fallback_this_step =
            energy_audit_ale_fallback_this_step ||
            ale_out.energy_audit_ale_fallback;
        if ((stage24_telemetry_enabled() ||
             cfg.numerics.hydro.dispatcher_state_sensitive_bypass_enabled) &&
            ale_out.effective_mode_executed !=
                hydro::ale::AleMode::ScheduledDefault) {
          stage24_rung_reached =
              std::max(stage24_rung_reached,
                       static_cast<int>(ale_out.effective_mode_executed));
        }
        if (ale_out.effective_mode_executed ==
                hydro::ale::AleMode::AxisVariationalProjection &&
            pending_request_ptr != nullptr) {
          const char* engaged_via = nullptr;
          if (pending_request_ptr->mode ==
              hydro::ale::AleMode::AxisSpinePlusLocal) {
            engaged_via = "escalation";
          } else if (pending_request_ptr->mode ==
                     hydro::ale::AleMode::AxisVariationalProjection) {
            engaged_via = "primary";
          }
          if (engaged_via != nullptr &&
              stage24_axis_variational_projection_engaged_via.empty()) {
            stage24_axis_variational_projection_engaged_via = engaged_via;
            g_dt_lineage_context.axis_variational_projection_engaged_via =
                stage24_axis_variational_projection_engaged_via;
          }
        }
        if (stage24_pending_repair_state_sensitive && ale_out.applied) {
          stage24_state_sensitive_repair_executed_this_step = true;
        }
        stage24_pending_repair_state_sensitive = false;
        if (pending_repair_possible) {
          pending_repair_plan.reset();
        }
        g_dt_lineage_context.ale_attempted = true;
        g_dt_lineage_context.ale_applied = ale_out.applied;
        g_dt_lineage_context.ale_rezone_triggered = ale_out.rezone_triggered;
        g_dt_lineage_context.ale_rezone_converged = ale_out.rezone_converged;
        g_dt_lineage_context.ale_quality_rollback = ale_out.quality_rollback;
        g_dt_lineage_context.ale_quality_min = ale_out.quality_min;
        g_dt_lineage_context.ale_rezone_iterations = ale_out.rezone_iterations;
        g_dt_lineage_context.ale_rezone_residual = ale_out.rezone_residual;
        if (ale_out.rezone_triggered) {
          ale_closure_audit = ale_out.closure_audit;
        }
        if (corner_j_source_budget_active && ale_out.rezone_triggered) {
          mesh_degeneracy_forensics_state.snapshot_corner_j_source_budget_phase(
              state, "post_ale_rezone", true);
          mesh_degeneracy_forensics_state.snapshot_corner_j_source_budget_phase(
              state, "post_remap", ale_out.accepted_remap_count > 0);
        }
        if (ale_out.rezone_triggered) {
          maybe_emit_velocity_history_phase("post_ale_rezone", state.t);
          if (ale_out.accepted_remap_count > 0) {
            maybe_emit_velocity_history_phase("post_remap", state.t);
          }
        }
        state.ale_rezoned = ale_out.rezone_triggered;
        if (state.ale_rezoned) {
          state.ale_rezone_invocations += 1;
          refresh_void_mask_after_2d_ale();
        }
        step_E_floor += std::max(ale_out.E_floor_injected, 0.0);
        step_E_redistribution_unresolved += ale_out.E_redistribution_unresolved;
        if (active_repair_force_this_attempt &&
            profile_observability != nullptr) {
          const auto op_energy_after = capture_operator_energy();
          const auto active_repair_post_state =
              capture_escape_valve_cell_state(state, active_repair_cell_id);
          profile_observability->note_escape_valve_event(
              make_escape_valve_event("pre_hydro",
                                      "HydroDriverRetryActiveMeshRepair",
                                      "retry_active_mesh_repair_count",
                                      "retry: corner Jacobian balance below threshold",
                                      active_repair_cell_id,
                                      active_repair_pre_state,
                                      active_repair_post_state,
                                      0.0,
                                      0.0,
                                      0.0,
                                      op_energy_before.snap,
                                      op_energy_after,
                                      state.step,
                                      state.t,
                                      state.mesh.topo.nz));
        }
        process_escape_valve_events_for_audit();
        record_operator_energy("ale_pre_hydro",
                               op_energy_before,
                               std::max(ale_out.E_floor_injected, 0.0) +
                                   ale_out.E_redistribution_unresolved);
        log_ale_remap_floor_closure("pre_hydro", ale_out);
        hydro::CornerJacobianScaleResult post_ale_corner_scale{};
        if (corner_j_ale_trigger_possible) {
          const hydro::CellRegime* d_cell_regime =
              refresh_mesh_regime_cache_for_corner_guard();
          post_ale_corner_scale =
              compute_corner_scale_for_ale_trigger(d_cell_regime);
          g_dt_lineage_context.post_ale_corner_trial_scale =
              post_ale_corner_scale.scale;
          g_dt_lineage_context.post_ale_first_failing_cell =
              post_ale_corner_scale.first_failing_cell;
          g_dt_lineage_context.post_ale_first_failing_corner =
              post_ale_corner_scale.first_failing_corner;
          const bool post_ale_trigger_fired =
              cfg.numerics.hydro.regime_aware_corner_j_guard_enabled ||
                      cfg.numerics.hydro.axis_margin_guard_enabled
                  ? post_ale_corner_scale.trigger_recommended
                  : post_ale_corner_scale.scale <
                        cfg.numerics.hydro.corner_jacobian_ale_trigger_scale;
          if (post_ale_trigger_fired) {
            core::log_warning(
                "[ale-stats] corner-J pre-hydro trigger remains active after forced ALE: "
                "predicted_scale=" +
                format_sci(post_ale_corner_scale.scale) + " threshold=" +
                format_sci(cfg.numerics.hydro.corner_jacobian_ale_trigger_scale) +
                " first_cell=" +
                std::to_string(post_ale_corner_scale.first_trigger_cell >= 0
                                   ? post_ale_corner_scale.first_trigger_cell
                                   : post_ale_corner_scale.first_failing_cell) +
                " first_corner=" +
                std::to_string(post_ale_corner_scale.first_trigger_corner >= 0
                                   ? post_ale_corner_scale.first_trigger_corner
                                   : post_ale_corner_scale.first_failing_corner));
          }
        }
        hydro::CornerBalanceResult post_ale_balance{};
        if (active_repair_possible) {
          post_ale_balance = hydro::evaluate_corner_balance(
              state,
              cfg.numerics.hydro.driver_retry_corner_balance_threshold,
              &reducer,
              state.evacuated_cells.d_geometry_policy_exempt_cells.empty()
                  ? nullptr
                  : state.evacuated_cells.d_geometry_policy_exempt_cells.data());
          g_dt_lineage_context.retry_active_repair_post_ale_q_balance =
              post_ale_balance.min_q_balance;
          g_dt_lineage_context.retry_active_repair_post_ale_still_bad =
              !post_ale_balance.admissible;
          if (!post_ale_balance.admissible) {
            core::log_warning(
                "[active-repair] post-ALE corner balance still bad: q_balance=" +
                format_sci(post_ale_balance.min_q_balance) + " threshold=" +
                format_sci(cfg.numerics.hydro.driver_retry_corner_balance_threshold) +
                " first_cell=" +
                std::to_string(post_ale_balance.first_failing_cell) +
                " first_corner=" +
                std::to_string(post_ale_balance.first_failing_corner));
          }
        }
        if (cfg.numerics.ale.mesh_epoch_enabled && ale_out.applied &&
            mesh_epoch_commits_this_step <
                cfg.numerics.ale.mesh_epoch_max_per_step) {
          bool quality_improved = false;
          if (corner_j_ale_trigger_possible) {
            quality_improved =
                post_ale_corner_scale.scale > corner_scale.scale;
          } else if (active_repair_possible) {
            quality_improved = post_ale_balance.min_q_balance >
                               active_repair_balance.min_q_balance;
          }
          if (quality_improved) {
            capture_driver_retry_snapshot(retry_snapshot_, state, cfg, nullptr);
            ++mesh_epoch_commits_this_step;
            core::log_warning(
                "[mesh-epoch] committed k=" +
                std::to_string(mesh_epoch_commits_this_step) + "/" +
                std::to_string(cfg.numerics.ale.mesh_epoch_max_per_step) +
                " at step=" + std::to_string(state.step) +
                " (pre-hydro repair accumulated into the retry snapshot)");
          }
        }
        const auto t_dt_retry_start =
            verbose_phase_timing ? Clock::now() : Clock::time_point{};
        step_dt_lineage =
            compute_dt_lineage(state,
                               cfg,
                               cfg.main.t_end,
                               active_repair_force_this_attempt
                                   ? "post_active_mesh_repair_ale"
                                   : "post_corner_j_ale",
                               &eos_ctx);
        dt = reducer.allreduce_min(step_dt_lineage.dt_chosen);
        dt_ref =
            reducer.allreduce_min(step_dt_lineage.dt_uncapped_by_contact);
        if (verbose_phase_timing) {
          step_dt_ms += ms(t_dt_retry_start, Clock::now());
        }
        TENRYU_ASSERT(dt > 0.0, "Driver produced non-positive dt after pre-hydro ALE");
      }
    }
    const auto& runtime_ale_cfg = cfg.numerics.ale.runtime_controller;
    if (runtime_ale_cfg.pre_step_enabled &&
        runtime_ale_cfg.controller_enabled && !controller_disengaged &&
        ale_monitor_machine.has_value() &&
        ale_monitor_machine->state() == hydro::AleMonitorState::kHard &&
        !runtime_ale_controller_suppressed_by_morph_window(state, cfg) &&
        runtime_ale_controller_activation_reached(state, cfg)) {
      run_runtime_ale_controller_event_if_due(
          state, cfg, hydro::AleMonitorState::kHard);
    }
    const char* lm_invariant_dump_env =
        std::getenv("TENRYU_LM_INVARIANT_DUMP");
    const bool collect_step_ray_output =
        cfg.laser.ray_output_count > 0 || lm_invariant_dump_env != nullptr;
    const bool collect_trajectory =
        cfg.laser.ray_output_trajectory &&
        out.should_plot(state.step + 1, state.t + dt, state, cfg);
    const bool collect_density_diag =
        out.should_plot(state.step + 1, state.t + dt, state, cfg);
    clear_laser_ray_output(state);
    const bool embed_conduction_in_thermal_subcycle =
        cfg.numerics.radiation_thermal_subcycle && cfg.radiation.enabled &&
        !cfg.radiation.imc.two_stage;
    const bool phase_energy_trace_enabled = (cfg.main.verbosity == "verbose");
    // W-J (VERIFICATION 10.1 drive-phase eps audit): per-operator energy
    // ledger closure, read-only and default-inert. Enable with
    // TENRYU_ENERGY_OPERATOR_AUDIT=1; TENRYU_ENERGY_OPERATOR_AUDIT_TMAX
    // (seconds, default 1.2e-9) limits logging to t < TMAX (<=0: no limit).
    const bool wj_op_audit_enabled = [] {
      const char* s = std::getenv("TENRYU_ENERGY_OPERATOR_AUDIT");
      return s != nullptr && std::atoi(s) != 0;
    }();
    double wj_op_audit_tmax = 1.2e-9;
    if (const char* s = std::getenv("TENRYU_ENERGY_OPERATOR_AUDIT_TMAX")) {
      wj_op_audit_tmax = std::atof(s);
    }
    const bool wj_op_audit_active =
        wj_op_audit_enabled &&
        (!(wj_op_audit_tmax > 0.0) || state.t < wj_op_audit_tmax);
    struct WjOpAuditSnapshot {
      double E_matter = 0.0;
      double E_matter_nodal = 0.0;
      double E_rad = 0.0;
      double laser_in = 0.0;
      double burn_in = 0.0;
      double laser_esc = 0.0;
      double cbet_iaw = 0.0;
      double marshak_in = 0.0;
      double volume_in = 0.0;
      double rad_esc = 0.0;
      double numloss = 0.0;
      double pdv_bdry = 0.0;
      double floor = 0.0;
      double redistribution = 0.0;
      double rad_mesh_adv = 0.0;
      double solver = 0.0;
      // W-R1 deferred handles (-1 = the fields above already hold values)
      long e_off = -1;
      long r_off = -1;
      std::size_t rad_mesh_capture_count = 0;
    };
    struct WjDeferredLog {
      std::string op;
      WjOpAuditSnapshot b;
      WjOpAuditSnapshot a;
    };
    std::vector<WjDeferredLog> wj_deferred_logs;
    const auto wj_audit_capture = [&]() {
      WjOpAuditSnapshot s;
      bool wj_energy_deferred = false;
      if (wj_arena.mode) {
        s.e_off = wj_defer_energy_capture();
        wj_energy_deferred = s.e_off >= 0;
      }
      if (!wj_energy_deferred) {
        const auto m = capture_operator_energy();
        s.E_matter = m.E_thermal_total + m.E_kinetic_total;
        s.E_matter_nodal = m.E_thermal_total + m.E_kinetic_total_nodal;
      }
      if (cfg.radiation.enabled) {
        if (deterministic_radiation_mode) {
          if (wj_arena.mode) {
            s.r_off = wj_defer_rad_capture();
          }
          if (s.r_off < 0) {
            s.E_rad = compute_fld_rad_energy_total_device(state, cfg);
          }
        } else {
          s.E_rad = imc.census_energy();
        }
      }
      s.laser_in = step_E_laser_in;
      s.burn_in = step_E_burn_in;
      s.laser_esc = step_E_laser_esc;
      s.cbet_iaw = step_E_cbet_iaw;
      s.marshak_in = step_E_marshak_in;
      s.volume_in = step_E_volume_in;
      s.rad_esc = step_E_rad_esc;
      s.numloss = step_E_numerical_loss;
      s.pdv_bdry = step_E_pdV_bdry;
      s.floor = step_E_floor;
      s.redistribution = step_E_redistribution_unresolved;
      s.rad_mesh_adv = step_E_rad_mesh_advection;
      s.rad_mesh_capture_count = deferred_rad_mesh_advection.size();
      s.solver = state.E_solver;
      return s;
    };
    const auto wj_audit_emit = [&](const char* op,
                                   const WjOpAuditSnapshot& b,
                                   const WjOpAuditSnapshot& a) {
      const double d_matter = a.E_matter - b.E_matter;
      const double d_matter_nodal = a.E_matter_nodal - b.E_matter_nodal;
      const double d_rad = a.E_rad - b.E_rad;
      const double d_solver = a.solver - b.solver;
      const double d_floor = a.floor - b.floor;
      const double d_redis = a.redistribution - b.redistribution;
      const double d_src = (a.laser_in - b.laser_in) +
                           (a.burn_in - b.burn_in) +
                           (a.marshak_in - b.marshak_in) +
                           (a.volume_in - b.volume_in) + d_solver;
      const double d_sink = (a.laser_esc - b.laser_esc) +
                            (a.cbet_iaw - b.cbet_iaw) +
                            (a.rad_esc - b.rad_esc) +
                            (a.numloss - b.numloss) +
                            (a.pdv_bdry - b.pdv_bdry);
      const double d_rad_mesh_adv = a.rad_mesh_adv - b.rad_mesh_adv;
      const double resid = (d_matter + d_rad) - d_rad_mesh_adv -
                           (d_floor + d_redis) - (d_src - d_sink);
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(17)
          << "[wj_op_audit] step=" << (state.step + 1) << " t=" << state.t
          << " op=" << op << " dE_matter=" << d_matter
          << " dE_matter_nodal=" << d_matter_nodal << " dE_rad=" << d_rad
          << " d_laser_in=" << (a.laser_in - b.laser_in)
          << " d_burn_in=" << (a.burn_in - b.burn_in)
          << " d_laser_esc=" << (a.laser_esc - b.laser_esc)
          << " d_cbet_iaw=" << (a.cbet_iaw - b.cbet_iaw)
          << " d_marshak_in=" << (a.marshak_in - b.marshak_in)
          << " d_volume_in=" << (a.volume_in - b.volume_in)
          << " d_rad_esc=" << (a.rad_esc - b.rad_esc)
          << " d_numloss=" << (a.numloss - b.numloss)
          << " d_pdv=" << (a.pdv_bdry - b.pdv_bdry) << " d_floor=" << d_floor
          << " d_redis=" << d_redis << " d_solver=" << d_solver
          << " d_rad_mesh_adv=" << d_rad_mesh_adv
          << " resid=" << resid;
      core::log_info(oss.str());
    };
    const auto wj_audit_log = [&](const char* op, const WjOpAuditSnapshot& b) {
      if (!wj_op_audit_active) {
        return;
      }
      const WjOpAuditSnapshot a = wj_audit_capture();
      if (wj_arena.mode) {
        wj_deferred_logs.push_back(WjDeferredLog{std::string(op), b, a});
        return;
      }
      wj_audit_emit(op, b, a);
    };
    struct PhaseEnergySnapshot {
      diagnostics::EnergyTotals matter{};
      double rad = 0.0;
      long matter_off = -1;
      long rad_off = -1;
    };
    struct DeferredPhaseEnergyLog {
      std::string label;
      PhaseEnergySnapshot before;
      PhaseEnergySnapshot after;
    };
    std::vector<DeferredPhaseEnergyLog> deferred_phase_energy_logs;
    const auto capture_phase_energy = [&]() {
      PhaseEnergySnapshot snapshot{};
      if (phase_energy_trace_enabled) {
        if (wj_arena.mode) {
          snapshot.matter_off = wj_defer_energy_capture();
        }
        if (snapshot.matter_off < 0) {
          snapshot.matter = compute_energy_totals_for_state(state, cfg);
        }
        if (cfg.radiation.enabled) {
          if (deterministic_radiation_mode) {
            if (wj_arena.mode) {
              snapshot.rad_off = wj_defer_rad_capture();
            }
            if (snapshot.rad_off < 0) {
              snapshot.rad = compute_fld_rad_energy_total_device(state, cfg);
            }
          } else {
            snapshot.rad = imc.census_energy();
          }
        }
      }
      return snapshot;
    };
    const auto emit_phase_energy = [&](const char* label,
                                       const PhaseEnergySnapshot& before,
                                       const PhaseEnergySnapshot& after) {
      const double E_matter_before = before.matter.E_int_e + before.matter.E_int_i +
                                     before.matter.E_kin;
      const double E_matter_after = after.matter.E_int_e + after.matter.E_int_i +
                                    after.matter.E_kin;
      const double dE_matter = E_matter_after - E_matter_before;
      const double dE_rad = after.rad - before.rad;
      const double dE_total = dE_matter + dE_rad;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(17)
          << "[phase_energy_trace] step=" << (state.step + 1)
          << " label=" << label
          << " dE_matter=" << dE_matter
          << " dE_rad=" << dE_rad
          << " dE_total=" << dE_total;
      core::log_info(oss.str());
    };
    const auto log_phase_energy = [&](const char* label,
                                      const PhaseEnergySnapshot& before) {
      if (!phase_energy_trace_enabled) {
        return;
      }
      const PhaseEnergySnapshot after = capture_phase_energy();
      if (wj_arena.mode) {
        deferred_phase_energy_logs.push_back(
            DeferredPhaseEnergyLog{std::string(label), before, after});
        return;
      }
      emit_phase_energy(label, before, after);
    };
    const auto sum_field_for_phase_trace = [&](const auto& field) {
      double sum = 0.0;
      if (phase_energy_trace_enabled && !field.empty()) {
        const auto host = copy_field_to_host(field);
        for (const double value : host) {
          sum += value;
        }
      }
      return sum;
    };
    const auto record_mesh_attr_zero =
        [&](const diagnostics::mesh_attribution::MeshDeformSource source) {
          if (cfg.numerics.diagnostics.mesh_attribution.enabled &&
              cfg.numerics.diagnostics.mesh_attribution.record_node_displacements) {
            diagnostics::mesh_attribution::record_zero_source(source);
          }
        };
    const auto radial_fourier_audit_active_at = [&](const double t_s) {
      return (cfg.diagnostics.per_operator_radial_fourier_enabled ||
              cfg.diagnostics.per_operator_radial_fourier_complex_enabled) &&
             history_writer.enabled() && is_2d && part_info.rank == 0 &&
             t_s >= cfg.diagnostics.radial_fourier_window_t_start_s &&
             t_s < cfg.diagnostics.radial_fourier_window_t_end_s;
    };
    const auto emit_radial_fourier_audit_for_cycle =
        [&](const diagnostics::RadialFourierStageId stage,
            const diagnostics::RadialFourierStagePhase phase,
            const double t_s,
            const std::uint64_t cycle) {
          if (!radial_fourier_audit_active_at(t_s)) {
            return;
          }
          if (cfg.diagnostics.per_operator_radial_fourier_enabled) {
            const auto audit =
                diagnostics::compute_radial_fourier_audit(state, cfg);
            if (audit.valid) {
              const auto record = diagnostics::make_radial_fourier_audit_record(
                  audit,
                  cycle,
                  t_s,
                  stage,
                  phase);
              history_writer.append_radial_fourier_audit(record);
            }
          }
#if TENRYU_RFA_V2_KERNEL_LAUNCHES
          if (cfg.diagnostics.per_operator_radial_fourier_complex_enabled) {
            const auto audit =
                diagnostics::compute_radial_fourier_complex_audit(state, cfg);
            if (audit.valid) {
              const auto record =
                  diagnostics::make_radial_fourier_complex_audit_record(
                      audit,
                      cycle,
                      t_s,
                      dt,
                      stage,
                      phase);
              history_writer.append_radial_fourier_complex_audit(record);
            }
          }
#endif
        };
    const auto emit_radial_fourier_audit =
        [&](const diagnostics::RadialFourierStageId stage,
            const diagnostics::RadialFourierStagePhase phase,
            const double t_s) {
          emit_radial_fourier_audit_for_cycle(
              stage,
              phase,
              t_s,
              static_cast<std::uint64_t>(state.step + 1));
        };

    // Rank-uniform hydro step verdicts (design doc mpi_m18_20_20260717.md
    // §6b.5): every driver retry predicate branches on HydroStepResult, so
    // the decision-relevant scalars must be identical on all ranks or the
    // per-rank control flow diverges into a communication deadlock. Flags
    // compose by OR/AND; the descriptive fields are broadcast from the
    // winner rank (lowest global first_failing_cell, else lowest failing
    // rank), which reproduces single-rank-failure semantics exactly.
    const auto uniformize_hydro_step_result = [&](HydroStepResult& res) {
      if (part_info.n_ranks <= 1) {
        return;
      }
      constexpr double kNoCell = 1.0e18;
      double flags[3] = {res.admissible ? 1.0 : 0.0,
                         res.retry_required ? 1.0 : 0.0,
                         (res.retry_required || !res.admissible)
                             ? static_cast<double>(part_info.rank)
                             : kNoCell};
      double cell_key = res.first_failing_cell >= 0
                            ? static_cast<double>(res.first_failing_cell)
                            : kNoCell;
      reducer.allreduce_min(&flags[0], 1);
      reducer.allreduce_max(&flags[1], 1);
      reducer.allreduce_min(&flags[2], 1);
      double cell_key_global = cell_key;
      reducer.allreduce_min(&cell_key_global, 1);
      const bool any_failure = flags[0] < 0.5 || flags[1] > 0.5;
      res.admissible = flags[0] > 0.5;
      res.retry_required = flags[1] > 0.5;
      if (!any_failure && cell_key_global >= kNoCell) {
        return;
      }
      // Winner: the rank owning the globally-first failing cell; if no rank
      // reported a cell, the lowest rank that reported a failure flag.
      double winner_rank = kNoCell;
      if (cell_key_global < kNoCell) {
        winner_rank = (cell_key == cell_key_global)
                          ? static_cast<double>(part_info.rank)
                          : kNoCell;
        reducer.allreduce_min(&winner_rank, 1);
      } else {
        winner_rank = flags[2];
      }
      if (winner_rank >= kNoCell) {
        return;
      }
      const int root = static_cast<int>(winner_rank);
      double pack[40] = {
          static_cast<double>(res.first_failing_cell),
          static_cast<double>(res.first_failing_corner),
          static_cast<double>(res.first_failing_i),
          static_cast<double>(res.first_failing_j),
          res.min_metric,
          res.suggested_dt,
          static_cast<double>(static_cast<int>(res.stage)),
          static_cast<double>(static_cast<int>(res.regime)),
          static_cast<double>(static_cast<int>(res.retry_action)),
          static_cast<double>(static_cast<int>(res.geometry_failure_kind)),
          res.failing_value,
          res.min_cell_vol,
          res.min_cell_area,
          res.min_corner_j,
          res.gauss_j_failed ? 1.0 : 0.0,
          res.rz_volume_failed ? 1.0 : 0.0,
          res.min_gauss_j_trial,
          res.min_rz_volume_trial,
          res.trial_scale,
          static_cast<double>(res.path_source_kind),
          static_cast<double>(res.path_metric_kind),
          static_cast<double>(res.path_block_id),
          static_cast<double>(res.path_local_i),
          static_cast<double>(res.path_local_j),
          static_cast<double>(res.path_old_geometry_inadmissible),
          res.q_balance,
          static_cast<double>(res.macro_q_begin),
          static_cast<double>(res.macro_q_end),
          static_cast<double>(res.macro_j_begin),
          static_cast<double>(res.macro_j_end),
          static_cast<double>(res.macro_span),
          static_cast<double>(res.macro_level),
          res.state_sensitive ? 1.0 : 0.0,
          res.dt_sensitive ? 1.0 : 0.0,
          res.dt_star_est,
          res.margin_rel,
          static_cast<double>(res.repair_generation),
          static_cast<double>(res.rung_reached),
          res.forecast_endpoint_valid ? 1.0 : 0.0,
          static_cast<double>(res.forecast_first_cell),
      };
      reducer.broadcast(pack, 40, root);
      res.first_failing_cell = static_cast<int>(pack[0]);
      res.first_failing_corner = static_cast<int>(pack[1]);
      res.first_failing_i = static_cast<int>(pack[2]);
      res.first_failing_j = static_cast<int>(pack[3]);
      res.min_metric = pack[4];
      res.suggested_dt = pack[5];
      res.stage = static_cast<HydroFailureStage>(static_cast<int>(pack[6]));
      res.regime =
          static_cast<tenryu::hydro::MeshFailureRegime>(static_cast<int>(pack[7]));
      res.retry_action = static_cast<RetryActionHint>(static_cast<int>(pack[8]));
      res.geometry_failure_kind =
          static_cast<tenryu::mesh::MeshGeometryFailureKind>(
              static_cast<int>(pack[9]));
      res.failing_value = pack[10];
      res.min_cell_vol = pack[11];
      res.min_cell_area = pack[12];
      res.min_corner_j = pack[13];
      res.gauss_j_failed = pack[14] > 0.5;
      res.rz_volume_failed = pack[15] > 0.5;
      res.min_gauss_j_trial = pack[16];
      res.min_rz_volume_trial = pack[17];
      res.trial_scale = pack[18];
      res.path_source_kind = static_cast<int>(pack[19]);
      res.path_metric_kind = static_cast<int>(pack[20]);
      res.path_block_id = static_cast<int>(pack[21]);
      res.path_local_i = static_cast<int>(pack[22]);
      res.path_local_j = static_cast<int>(pack[23]);
      res.path_old_geometry_inadmissible = static_cast<int>(pack[24]);
      res.q_balance = pack[25];
      res.macro_q_begin = static_cast<int>(pack[26]);
      res.macro_q_end = static_cast<int>(pack[27]);
      res.macro_j_begin = static_cast<int>(pack[28]);
      res.macro_j_end = static_cast<int>(pack[29]);
      res.macro_span = static_cast<int>(pack[30]);
      res.macro_level = static_cast<int>(pack[31]);
      res.state_sensitive = pack[32] > 0.5;
      res.dt_sensitive = pack[33] > 0.5;
      res.dt_star_est = pack[34];
      res.margin_rel = pack[35];
      res.repair_generation = static_cast<int>(pack[36]);
      res.rung_reached = static_cast<int>(pack[37]);
      res.forecast_endpoint_valid = pack[38] > 0.5;
      res.forecast_first_cell = static_cast<int>(pack[39]);
      // Informational minima used by step-level tracking compose as minima.
      double mins[4] = {res.min_path_margin, res.forecast_q_min_now,
                        res.forecast_q_min_end, res.forecast_q_min_path};
      reducer.allreduce_min(mins, 4);
      res.min_path_margin = mins[0];
      res.forecast_q_min_now = mins[1];
      res.forecast_q_min_end = mins[2];
      res.forecast_q_min_path = mins[3];
    };

    SplitOperatorCallbacks callbacks;
    callbacks.hydro = [&](const double dt_op,
                           const double t_op,
                           const char* hydro_half) -> HydroStepResult {
      HydroStepResult res;
      bool axis_band_applied_this_half_step = false;
      if (!cfg.numerics.hydro.enabled) {
        last_hydro_result = res;
        return res;
      }
      const PhaseEnergySnapshot e_before_phase = capture_phase_energy();
      const auto t_phase_start =
          verbose_phase_timing ? Clock::now() : Clock::time_point{};
      update_zbar_for_step(state, cfg);
      const auto op_energy_before =
          operator_energy_tracker.enabled() ? capture_operator_energy_cap()
                                            : OpEnergyCapture{};
      const WjOpAuditSnapshot wj_b_hydro =
          wj_op_audit_active ? wj_audit_capture() : WjOpAuditSnapshot{};
      long rad_mesh_E_before_half_off = -1;
      if (cfg.radiation.enabled && deterministic_radiation_mode &&
          wj_arena.mode) {
        rad_mesh_E_before_half_off = wj_defer_rad_capture();
      }
      const double rad_mesh_E_before_half =
          (cfg.radiation.enabled && deterministic_radiation_mode &&
           rad_mesh_E_before_half_off < 0)
              ? compute_fld_rad_energy_total_device(state, cfg)
              : 0.0;
      double hydro_E_floor = 0.0;
      double hydro_r_momentum_source_impulse = 0.0;
      int hydro_clamp_count = 0;
      int hydro_rho_clamp_count = 0;
      const long phase_safety_token = begin_phase_safety(
          cfg.numerics.safety.overshoot_fatal_enabled, 0.0);
      const double te_max_before =
          (phase_safety_token < 0 &&
           cfg.numerics.safety.overshoot_fatal_enabled)
              ? compute_temperature_audit_device(state).max_te
              : 0.0;
      // Hydro module performs the NUMERICS §12.2.3 hydro halo exchanges
      // (cell pre-step + node post-step, including hydro_active/Qvisc support).
      emit_radial_fourier_audit(
          diagnostics::RadialFourierStageId::HydroLag,
          diagnostics::RadialFourierStagePhase::Before,
          t_op);
      if (is_1d) {
        if (rad_gamma43_coupling_on) {
          const cudaError_t wk_copy_err = cudaMemcpy(
              rad_gamma43_vol_before.data(), state.vol.data(),
              state.vol.size() * sizeof(double), cudaMemcpyDeviceToDevice);
          TENRYU_ASSERT(wk_copy_err == cudaSuccess,
                        "wk_gamma_r: vol_before device copy failed");
          rad_gamma::compute_radiation_pressure_field(
              rad_gamma43_p_r.data(), state.rad_E.data(),
              static_cast<int>(state.rho.size()), rad_gamma43_n_groups);
        }
        const auto& hk_sigma_R_max = imc.last_sigma_R_max();
        const std::vector<double>* hk_sigma_R_max_ptr =
            (hk_sigma_R_max.size() == state.rho.size()) ? &hk_sigma_R_max : nullptr;
        h1d_dump7("d2");
        hydro::entropy_ledger_begin(
            state, cfg, hydro::EntropyLedgerStage::Hydro);
        hydro::stripe_gate_begin(state, cfg);
        res = hydro_1d.lagrangian_step(
            state, dt_op, cfg, t_op, &hydro_E_floor, &hydro_clamp_count,
            &hydro_rho_clamp_count, part_info, &comm_buffers, nullptr, &eos_ctx,
            hk_sigma_R_max_ptr,
            rad_gamma43_coupling_on ? &rad_gamma43_p_r : nullptr,
            rad_gamma43_coupling_on ? &rad_gamma43_W_r : nullptr);
        hydro::stripe_gate_end(state, cfg, dt_op, t_op + dt_op);
        hydro::entropy_ledger_end(
            state, cfg, hydro::EntropyLedgerStage::Hydro);
        hydro::entropy_ledger_accumulate_work_buckets(state, cfg, dt_op);
      } else {
        hydro::entropy_ledger_begin(
            state, cfg, hydro::EntropyLedgerStage::Hydro);
        hydro::stripe_gate_begin(state, cfg);
        hydro::axis_cell_ledger_step_begin(state, cfg);
        res = hydro_2d.lagrangian_step(
            state, dt_op, cfg, t_op, &hydro_E_floor, &hydro_clamp_count,
            &hydro_rho_clamp_count, part_info, &comm_buffers, nullptr, hydro_half,
            &eos_ctx, &reducer, &mesh_regime_cache, profile_observability,
            r_momentum_source_audit_active
                ? &hydro_r_momentum_source_impulse
                : nullptr);
        state.mesh.materialize_host_svec();
        hydro::stripe_gate_end(state, cfg, dt_op, t_op + dt_op);
        hydro::entropy_ledger_end(
            state, cfg, hydro::EntropyLedgerStage::Hydro);
        hydro::entropy_ledger_accumulate_work_buckets(state, cfg, dt_op);
      }
      uniformize_hydro_step_result(res);
      last_hydro_result = res;
      if (res.min_path_margin < step_min_path_margin) {
        step_min_path_margin = res.min_path_margin;
        step_min_path_margin_cell = res.min_path_margin_cell;
      }
      if (res.retry_required) {
        if (verbose_phase_timing) {
          step_hydro_ms += ms(t_phase_start, Clock::now());
        }
        log_phase_energy("hydro_callback_retry", e_before_phase);
        return res;
      }
      if (is_2d) {
        hydro::corner_collapse_ledger_capture(
            state, state.step + 1, t_op + dt_op, dt_op,
            hydro::CornerCollapseLedgerTrial::Accepted);
      }
      if (cfg.numerics.hydro.wake_heat_flux_enabled && is_2d &&
          hydro::wake_angular_heat_flux::apply(state, cfg, dt_op)) {
        hydro_2d.reclose_after_energy_edit(state, cfg, &eos_ctx);
      }
      if (is_2d) {
        hydro::axis_cell_ledger_step_end(state, cfg, dt_op);
      }
      if (r_momentum_source_audit_active) {
        r_momentum_source_impulse_step += hydro_r_momentum_source_impulse;
      }
      if (cap_energy_audit_enabled) {
        cap_energy_audit_W_ext_step += res.cap_energy_audit_W_ext_step;
        cap_energy_audit_W_pq += res.cap_energy_audit_W_pq;
        cap_energy_audit_W_uF += res.cap_energy_audit_W_uF;
        cap_energy_audit_W_apex_pinned +=
            res.cap_energy_audit_W_apex_pinned;
      }
      if (i1b_spurious_sensor_enabled) {
        merge_i1b_sensor_summary(i1b_hydro_sensor_step,
                                 res.i1b_hydro_nonaffine_sensor,
                                 part_info.rank,
                                 i1b_spurious_sensor_top_k);
      }
      emit_radial_fourier_audit(
          diagnostics::RadialFourierStageId::HydroLag,
          diagnostics::RadialFourierStagePhase::After,
          t_op + dt_op);
      if (cfg.radiation.imc.difference.enabled) {
        imc.invalidate_difference_reference();
      }
      step_E_floor += std::max(hydro_E_floor, 0.0);
      step_clamp_count += std::max(hydro_clamp_count, 0);
      step_clamp_count += std::max(hydro_rho_clamp_count, 0);
      const double hydro_pdv_work =
          is_2d ? res.pressure_boundary_work_step
                : compute_boundary_pdv_work_1d(state, cfg, dt_op, t_op);
      // Legacy ledger convention: E_pdV_bdry is outward pdV flux, not work on the gas.
      step_E_pdV_bdry += is_2d ? -hydro_pdv_work : hydro_pdv_work;
      const std::string hydro_operator_name =
          hydro_half != nullptr ? std::string("hydro_") + hydro_half
                                : std::string("hydro");
      record_operator_energy(hydro_operator_name,
                             op_energy_before,
                             std::max(hydro_E_floor, 0.0) - hydro_pdv_work);
      wj_audit_log(hydro_operator_name.c_str(), wj_b_hydro);
      if (rad_gamma43_coupling_on && !res.retry_required) {
        // gamma_r=4/3 field update BEFORE the Delta(E_r V) tally below, so
        // E_rad_mesh_advection books exactly W_r = Delta(E_r V) in this mode
        // (and the frozen-density bookkeeping term when coupling is off) —
        // one tally, both modes close the eps ledger.
        // v3 (verdict 2026-07-06): the field pays the p_r force's ACTUAL
        // nodal work (same half-position areas, same ubar, same masking as
        // the momentum update) — force/work conjugacy; v1's exact adiabat
        // created (2/9) U delta^2 per half-step (positive both signs).
        // Owned-window payment: W_r is produced by the owned-window gamma_r
        // force-work kernel (hydro_1d), so the rad_E application must use
        // the same window (full-range would fold garbage W_r into non-owned
        // rad_E). Serial: [0, n_cells) — byte-identical to the old path.
        const auto gr_cw =
            state.owned_cell_window(static_cast<int>(state.rho.size()));
        const double rad_gamma_floor = rad_gamma::apply_gamma_r_43_work_update(
            state.rad_E.data(), rad_gamma43_W_r.data(),
            rad_gamma43_vol_before.data(), state.vol.data(), gr_cw.begin,
            gr_cw.end, rad_gamma43_n_groups);
        step_E_floor += std::max(rad_gamma_floor, 0.0);
        // Re-globalize the paid rad_E line from the owners: the replicated
        // 1D solve and the NEXT hydro half's radiation-pressure field both
        // read the full line (root cause of the §16.4 FLD-1D bitwise FAIL —
        // rank!=0 lines missed the hot-region payments; design doc §6h).
        allgatherv_1d_cell_line(state.rad_E.data(), rad_gamma43_n_groups);
      }
      if (cfg.radiation.enabled && deterministic_radiation_mode) {
        const long after_off = rad_mesh_E_before_half_off >= 0
                                   ? wj_defer_rad_capture()
                                   : -1;
        if (rad_mesh_E_before_half_off >= 0) {
          TENRYU_ASSERT(after_off >= 0,
                        "S2a radiation mesh audit slot capacity exceeded");
          deferred_rad_mesh_advection.push_back(
              DeferredRadMeshAdvection{rad_mesh_E_before_half_off, after_off});
        } else {
          step_E_rad_mesh_advection +=
              compute_fld_rad_energy_total_device(state, cfg) -
              rad_mesh_E_before_half;
        }
      }
	      if (is_2d && cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled &&
	          !hydro::ale::ale_identity_mode_enabled(cfg) &&
	          cfg.numerics.ale.conservative_remap_enabled &&
          // Band-ALE mode is transaction-only — the flow stays Lagrangian between band engagements, mirroring the multiblock arrangement (§18.6).
          !cfg.numerics.ale.band_ale.enabled &&
          !tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh) &&
          !cfg.numerics.hydro.hllc_z_flux_2d_rz) {
        const bool dump_operator_profile_this_step =
            i1_operator_profile_dump_step_matches(
                i1_operator_profile_dump, cfg, state, state.step + 1,
                state.t + dt);
        maybe_capture_i1_operator_profile(i1_operator_profile_dump,
                                          state,
                                          cfg,
                                          state.step + 1,
                                          part_info.rank,
                                          hydro_half,
                                          "pre_remap",
                                          t_op + dt_op,
                                          dt_op,
                                          false,
                                          dump_operator_profile_this_step);
        const auto op_energy_before_remap =
            operator_energy_tracker.enabled() ? capture_operator_energy_cap()
                                              : OpEnergyCapture{};
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::Remap,
            diagnostics::RadialFourierStagePhase::Before,
            t_op + dt_op);
        hydro::entropy_ledger_begin(
            state, cfg, hydro::EntropyLedgerStage::Ale);
        const auto remap_out =
            hydro::ale::ale_remap_2d_rz(state, cfg, &eos_ctx, dt_op);
        hydro::entropy_ledger_end(
            state, cfg, hydro::EntropyLedgerStage::Ale);
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::Remap,
            diagnostics::RadialFourierStagePhase::After,
            t_op + dt_op);
        maybe_capture_i1_operator_profile(i1_operator_profile_dump,
                                          state,
                                          cfg,
                                          state.step + 1,
                                          part_info.rank,
                                          hydro_half,
                                          "post_remap",
                                          t_op + dt_op,
                                          dt_op,
                                          remap_out.applied,
                                          dump_operator_profile_this_step);
        if (cap_energy_audit_enabled) {
          cap_energy_audit_D_K += remap_out.cap_energy_audit_D_K;
          cap_energy_audit_rezone_fired =
              cap_energy_audit_rezone_fired || remap_out.applied;
        }
        if (i1b_spurious_sensor_enabled) {
          merge_i1b_sensor_summary(i1b_ale_sensor_step,
                                   remap_out.i1b_ale_ke_sensor,
                                   part_info.rank,
                                   i1b_spurious_sensor_top_k);
        }
        if (remap_out.applied) {
          state.ale_rezoned = true;
          record_mesh_attr_zero(
              diagnostics::mesh_attribution::MeshDeformSource::Remap);
          const double remap_E_floor =
              cfg.numerics.hydro.total_energy_remap_2d_rz
                  ? std::max(remap_out.E_floor_injected, 0.0)
                  : 0.0;
          const double remap_E_redistribution_unresolved =
              cfg.numerics.hydro.total_energy_remap_2d_rz
                  ? remap_out.E_redistribution_unresolved
                  : 0.0;
          step_E_floor += remap_E_floor;
          step_E_redistribution_unresolved +=
              remap_E_redistribution_unresolved;
          record_operator_energy("ale_conservative_remap",
                                 op_energy_before_remap,
                                 remap_out.boundary_energy_flux_z_bottom +
                                     remap_out.boundary_energy_flux_z_top +
                                     remap_E_floor +
                                     remap_E_redistribution_unresolved);
        }
      }
      if (part_info.n_ranks > 1 && is_2d && state.mesh.topo.n_cells > 0 &&
          state.volFrac.size() >
              static_cast<std::size_t>(state.mesh.topo.n_cells)) {
        // Multi-material ghost freshness (M18e item 5): the 2D remap
        // mutates volFrac through the owned window, but the coefficient
        // -mixing consumers (conduction faces, radiation Newton
        // A_eff/gamma_eff, LM projection A_eff) read the volfrac-derived
        // props at ghost cells. Refresh the ghost strips at the mutation
        // point and rebuild the per-cell material props from the fresh
        // volFrac. Single-material states skip (size == n_cells).
        parallel::exchange_cell_strips_scaled(
            part_info, comm_buffers, state.volFrac.data(),
            static_cast<int>(state.volFrac.size() /
                             static_cast<std::size_t>(
                                 state.mesh.topo.n_cells)),
            19);
        state.invalidate_cell_material_props();
        state.ensure_cell_material_props(cfg);
      }
      refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
      finish_phase_safety(phase_safety_token,
                          "hydro",
                          te_max_before,
                          state.step + 1,
                          t_op + dt_op,
                          dt_op);
      const bool is_strang_hydro_half =
          hydro_half != nullptr &&
          (std::strcmp(hydro_half, "strang_first") == 0 ||
           std::strcmp(hydro_half, "strang_second") == 0);
	      if (is_strang_hydro_half &&
	          cfg.numerics.ale.axis_band_managed_remap_enabled &&
	          !hydro::ale::ale_identity_mode_enabled(cfg) &&
	          cfg.numerics.ale.axis_band_managed_remap_every_hydro_half_step &&
          !axis_band_applied_this_half_step) {
        hydro::entropy_ledger_begin(
            state, cfg, hydro::EntropyLedgerStage::Ale);
        const auto axis_result = run_axis_band_controller(
            true,
            cfg.numerics.ale.axis_band_managed_remap_width,
            false,
            nullptr);
        hydro::entropy_ledger_end(
            state, cfg, hydro::EntropyLedgerStage::Ale);
        axis_band_applied_this_half_step = axis_result.remap_succeeded;
      }
      if (verbose_phase_timing) {
        step_hydro_ms += ms(t_phase_start, Clock::now());
      }
      log_phase_energy("hydro_callback", e_before_phase);
      return res;
    };
    const auto run_conduction_phase = [&](const double dt_op, const double t_op) {
      if (!cfg.numerics.conduction.enabled) {
        return;
      }
      static const bool cond_ebal_enabled = [] {
        const char* s = std::getenv("TENRYU_HYDRO_EBAL");
        return s != nullptr && std::atoi(s) != 0;
      }();
      const double cond_ebal_U0 =
          cond_ebal_enabled
              ? diagnostics::compute_energy_budget_1d(state).E_total
              : 0.0;
      const bool cond_energy_authoritative =
          (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
      std::vector<double> cond_Te_old;
      std::vector<double> cond_cv_old;
      if (cond_energy_authoritative) {
        const std::size_t nc = state.Te.size();
        cond_Te_old.resize(nc);
        if (!state.cv_e.empty()) {
          cond_cv_old.resize(nc);
          std::vector<double> cond_pack(2 * nc);
          const double* srcs[2] = {state.Te.data(), state.cv_e.data()};
          core::pack_pull_fields(srcs, 2, static_cast<int>(nc),
                                 cond_pack.data(),
                                 "driver:run_conduction_phase:pull");
          std::memcpy(cond_Te_old.data(), cond_pack.data(),
                      nc * sizeof(double));
          std::memcpy(cond_cv_old.data(), cond_pack.data() + nc,
                      nc * sizeof(double));
        } else {
          state.Te.copy_to_host(cond_Te_old.data());
          cond_cv_old.assign(nc, 0.0);
        }
      }
      const PhaseEnergySnapshot e_before_phase = capture_phase_energy();
      const auto t_phase_start =
          verbose_phase_timing ? Clock::now() : Clock::time_point{};
      const long phase_safety_token = begin_phase_safety(
          cfg.numerics.safety.overshoot_fatal_enabled, 0.0);
      const double te_max_before =
          (phase_safety_token < 0 &&
           cfg.numerics.safety.overshoot_fatal_enabled)
              ? compute_temperature_audit_device(state).max_te
              : 0.0;
      update_zbar_for_step(state, cfg);
      const auto op_energy_before =
          operator_energy_tracker.enabled() ? capture_operator_energy_cap()
                                            : OpEnergyCapture{};
      // Conduction consumes Zbar from state.zbar as the single source of truth.
      // Conduction module performs pre/post and STS-stage Te halo exchanges.
      // Use CUDA default stream explicitly; halo exchange synchronizes this stream
      // before host-side MPI operations.
      const WjOpAuditSnapshot wj_b_cond =
          wj_op_audit_active ? wj_audit_capture() : WjOpAuditSnapshot{};
      cudaStream_t cond_stream = nullptr;
      std::vector<double> conduction_ee_before;
      if (cfg.numerics.diagnostics.conduction_energy_rate_export.enabled) {
        conduction_ee_before = copy_field_to_host(state.ee);
      }
      const hydro::ConductionResult conduction_result =
          hydro::conduction_step(
              state, dt_op, cfg, part_info, &comm_buffers, cond_stream, &eos_ctx);
      if (conduction_result.retry_required) {
        conduction_retry_requested = true;
        conduction_retry_result = conduction_result;
      }
      ++state.c1_solver_steps_total;
      state.c1_solver_residual_last = conduction_result.solver_residual;
      state.c1_solver_residual_max =
          std::max(state.c1_solver_residual_max, conduction_result.solver_residual);
      state.c1_solver_iter_last = conduction_result.solver_iterations;
      state.c1_solver_iter_max =
          std::max(state.c1_solver_iter_max, conduction_result.solver_iterations);
      state.c1_solver_cond_number_last = conduction_result.solver_cond_number_est;
      state.c1_solver_cond_number_max =
          std::max(state.c1_solver_cond_number_max,
                   conduction_result.solver_cond_number_est);
      if (cfg.numerics.conduction.nonlocal_model == "snb") {
        ++state.snb_steps_total;
        state.snb_picard_iters_last = conduction_result.snb_picard_iters;
        state.snb_picard_iters_max =
            std::max(state.snb_picard_iters_max, conduction_result.snb_picard_iters);
        if (!conduction_result.snb_converged) {
          ++state.snb_nonconverged_steps;
        }
        state.snb_picard_resid_last = conduction_result.snb_picard_resid;
        state.snb_cap_faces_99_last = conduction_result.snb_cap_faces_99;
        state.snb_cap_faces_50_last = conduction_result.snb_cap_faces_50;
        state.snb_cap_theta_min_last = conduction_result.snb_cap_theta_min;
        state.snb_cap_theta_min_run =
            std::min(state.snb_cap_theta_min_run, conduction_result.snb_cap_theta_min);
        state.snb_dq_over_qsh_max_last = conduction_result.snb_dq_over_qsh_max;
        state.snb_dq_over_qsh_max_run =
            std::max(state.snb_dq_over_qsh_max_run,
                     conduction_result.snb_dq_over_qsh_max);
        state.snb_solver_iters = conduction_result.snb_solver_iters;
        state.snb_solver_resid = conduction_result.snb_solver_resid;
      }
      record_mesh_attr_zero(
          diagnostics::mesh_attribution::MeshDeformSource::SpitzerConduction);
      g_dt_lineage_context.prev_step_kershaw_mmatrix_fix_count =
          conduction_result.mmatrix_fix_count;
      step_E_floor += std::max(conduction_result.E_floor_injected, 0.0);
      // SNB max-principle ceiling removals are accounted as numerical loss.
      step_E_numerical_loss += std::max(conduction_result.snb_E_ceiling_removed, 0.0);
      step_clamp_count += std::max(conduction_result.clamp_count, 0);
      finish_phase_safety(phase_safety_token,
                          "conduction",
                          te_max_before,
                          state.step + 1,
                          t_op + dt_op,
                          dt_op);
      // Post-conduction sync: re-derive ee/Pe from Te using the active hydro-EOS closure.
      // Table backends recover e(rho,T) from the EOS table, while exact_ideal_gas keeps the
      // same analytic ideal-gas projection to avoid reintroducing cold-curve offsets.
      const double cond_ebal_U1 =
          cond_ebal_enabled
              ? diagnostics::compute_energy_budget_1d(state).E_total
              : 0.0;
      if (cond_energy_authoritative) {
        apply_conduction_energy_increment(state, cfg, cond_Te_old,
                                          cond_cv_old);
      } else {
        sync_ee_from_Te_table(state, cfg);
      }
      if (cfg.numerics.diagnostics.conduction_energy_rate_export.enabled) {
        const std::vector<double> conduction_ee_after =
            copy_field_to_host(state.ee);
        const std::vector<double> conduction_rho_after =
            copy_field_to_host(state.rho);
        std::vector<double> conduction_e_rate(
            conduction_ee_before.size(), 0.0);
        for (std::size_t c = 0; c < conduction_e_rate.size(); ++c) {
          conduction_e_rate[c] =
              (conduction_ee_after[c] - conduction_ee_before[c]) *
              conduction_rho_after[c] / dt_op;
        }
        state.conduction_e_rate.copy_from_host(conduction_e_rate);
      }
      if (cond_ebal_enabled) {
        const double cond_ebal_U2 =
            diagnostics::compute_energy_budget_1d(state).E_total;
        std::ostringstream cond_os;
        cond_os << std::scientific << std::setprecision(9)
                << "[cond_ebal] t=" << t_op
                << " solve_dU=" << (cond_ebal_U1 - cond_ebal_U0)
                << " sync_dU=" << (cond_ebal_U2 - cond_ebal_U1);
        core::log_info(cond_os.str());
      }
      if (cfg.numerics.hydro.enabled && is_2d &&
          cfg.numerics.hydro.boundary_2d.has_any_state_supply()) {
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::BoundaryFill,
            diagnostics::RadialFourierStagePhase::Before,
            t_op + dt_op);
        hydro_2d.apply_state_supply_boundary(state, cfg, &eos_ctx, true);
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::BoundaryFill,
            diagnostics::RadialFourierStagePhase::After,
            t_op + dt_op);
      }
      std::string conduction_operator_name = "conduction";
      if (conduction_operator_record_count > 0) {
        conduction_operator_name +=
            "_" + std::to_string(conduction_operator_record_count);
      }
      ++conduction_operator_record_count;
      record_operator_energy(conduction_operator_name,
                             op_energy_before,
                             std::max(conduction_result.E_floor_injected, 0.0));
      wj_audit_log(conduction_operator_name.c_str(), wj_b_cond);
      if (verbose_phase_timing) {
        step_cond_ms += ms(t_phase_start, Clock::now());
      }
      log_phase_energy("conduction_callback", e_before_phase);
    };
    callbacks.conduction = [&](const double dt_op, const double t_op) {
      if (embed_conduction_in_thermal_subcycle) {
        return;
      }
      run_conduction_phase(dt_op, t_op);
    };
    callbacks.laser = [&](const double dt_op, const double t_op) {
      if (!cfg.laser.enabled) {
        return;
      }
      const PhaseEnergySnapshot e_before_phase = capture_phase_energy();
      const auto t_phase_start =
          verbose_phase_timing ? Clock::now() : Clock::time_point{};
      const auto op_energy_before =
          operator_energy_tracker.enabled() ? capture_operator_energy_cap()
                                            : OpEnergyCapture{};
      const WjOpAuditSnapshot wj_b_laser =
          wj_op_audit_active ? wj_audit_capture() : WjOpAuditSnapshot{};
      step_ray_output.clear();
      std::vector<laser::RayOutputData>* ray_output_ptr =
          collect_step_ray_output ? &step_ray_output : nullptr;
      if (part_info.n_ranks > 1 && is_1d) {
        // Full-line 1D laser inputs (Option C, spec
        // mpi_m18d_laser_burn_spec.md §2 d1): the LM projection builds
        // its hydro mirror from the WHOLE line and every rank traces it
        // redundantly (owned deposit masking + deposit Allgatherv are the
        // existing wiring), so the matter line must be owner-true first —
        // windowed hydro leaves far regions stale.
        allgatherv_1d_cell_line(state.rho.data(), 1);
        allgatherv_1d_cell_line(state.Te.data(), 1);
        allgatherv_1d_cell_line(state.zbar.data(), 1);
        if (!state.volFrac.empty() &&
            state.mesh.topo.n_cells > 0 &&
            state.volFrac.size() %
                    static_cast<std::size_t>(state.mesh.topo.n_cells) ==
                0) {
          allgatherv_1d_cell_line(
              state.volFrac.data(),
              static_cast<int>(state.volFrac.size() /
                               static_cast<std::size_t>(
                                   state.mesh.topo.n_cells)));
          state.invalidate_cell_material_props();
          state.ensure_cell_material_props(cfg);
        }
        allgatherv_1d_node_line(state.x_r.data());
      }
      if (part_info.n_ranks > 1 && is_2d &&
          std::getenv("TENRYU_LASER_ENTRY_EXCHANGE_OFF") == nullptr) {
        // Full-field 2D laser inputs (OPEN-LASER-FAR): the 2D ray trace is
        // replicated per rank across the WHOLE mesh (rays enter at r_max
        // and traverse every slab), so a ghost-strip exchange is NOT
        // enough — far-region rho/Te/zbar are stale under Option C and
        // each rank attenuated its rays through a different (stale) copy
        // of the other slabs (measured: laser-on P2-vs-P1 ee drift
        // 1.2e-4 rel at 10 steps vs 5.9e-16 P1 self-band). Re-globalize
        // every trace input from the owners, exactly like the 1D branch
        // above; this also supersedes the former LM stencil ghost
        // exchange (owner-true everywhere includes the strips).
        allgatherv_2d_cell_field(state.rho.data(), 1);
        allgatherv_2d_cell_field(state.Te.data(), 1);
        allgatherv_2d_cell_field(state.zbar.data(), 1);
        if (!state.volFrac.empty() &&
            state.mesh.topo.n_cells > 0 &&
            state.volFrac.size() %
                    static_cast<std::size_t>(state.mesh.topo.n_cells) ==
                0) {
          allgatherv_2d_cell_field(
              state.volFrac.data(),
              static_cast<int>(state.volFrac.size() /
                               static_cast<std::size_t>(
                                   state.mesh.topo.n_cells)));
          state.invalidate_cell_material_props();
          state.ensure_cell_material_props(cfg);
        }
        allgatherv_2d_node_field(state.x_r.data());
        allgatherv_2d_node_field(state.x_z.data());
      }
      // Laser reads Zbar internally from state.zbar while building its hydro mirror.
      laser::laser_step(state,
                        laser_mesh,
                        cfg.laser,
                        dt_op,
                        t_op,
                        part_info,
                        nullptr,
                        cfg.numerics.floors.rho,
                        cfg.numerics.floors.Te,
                        nullptr,
                        ray_output_ptr,
                        &reducer,
                        collect_trajectory,
                        cfg.main.verbosity == "verbose",
                        collect_density_diag,
                        out.output_dir);
      record_mesh_attr_zero(
          diagnostics::mesh_attribution::MeshDeformSource::RaytraceDeposition);
      if (collect_step_ray_output) {
        store_laser_ray_output(state, step_ray_output);
      }
      if (lm_invariant_dump_env != nullptr) {
        // SS16.6 deterministic-invariant dump (ratified 2026-07-18, design
        // doc §6p): ray count + per-ray launch records, bitwise-comparable
        // across ranks and against P=1 (rays are replicated, so any
        // divergence here is a real trace-input defect). Overwritten each
        // laser call; read-only observer of step_ray_output.
        int inv_rank = 0;
#if TENRYU_ENABLE_MPI
        {
          int mpi_initialized = 0;
          MPI_Initialized(&mpi_initialized);
          if (mpi_initialized != 0) {
            MPI_Comm_rank(MPI_COMM_WORLD, &inv_rank);
          }
        }
#endif
        const std::string inv_path = std::string(lm_invariant_dump_env) +
                                     "_ray_r" + std::to_string(inv_rank) +
                                     ".bin";
        if (FILE* fp = std::fopen(inv_path.c_str(), "wb")) {
          const std::int32_t n_beams =
              static_cast<std::int32_t>(step_ray_output.size());
          std::fwrite(&n_beams, sizeof(n_beams), 1, fp);
          for (const auto& beam : step_ray_output) {
            const std::int32_t bid = beam.beam_id;
            const std::int32_t n2 =
                static_cast<std::int32_t>(beam.rays_2d.size());
            const std::int32_t n3 =
                static_cast<std::int32_t>(beam.rays_3d.size());
            std::fwrite(&bid, sizeof(bid), 1, fp);
            std::fwrite(&n2, sizeof(n2), 1, fp);
            std::fwrite(&n3, sizeof(n3), 1, fp);
            if (n2 > 0) {
              std::fwrite(beam.rays_2d.data(), sizeof(beam.rays_2d[0]),
                          beam.rays_2d.size(), fp);
            }
            if (n3 > 0) {
              std::fwrite(beam.rays_3d.data(), sizeof(beam.rays_3d[0]),
                          beam.rays_3d.size(), fp);
            }
          }
          std::fclose(fp);
        }
      }
      const double commanded = std::max(laser_mesh.last_commanded_energy, 0.0);
      const double escaped = std::max(laser_mesh.last_unabsorbed_power, 0.0) * dt_op;
      const double ra_dep = std::max(laser_mesh.last_ra_power, 0.0) * dt_op;
      step_E_laser_in += commanded;
      step_E_laser_esc += escaped;
      step_E_ra_dep += ra_dep;
      step_E_laser_esc += std::max(state.hot_e_escaped_step, 0.0);
      step_E_cbet_iaw += state.E_cbet_iaw_step;
      const double sum_laser_dep_after_transfer =
          sum_field_for_phase_trace(state.laser_dep);
      double source_E_floor = 0.0;
      int source_clamp_count = 0;
      // Laser source-term closure reads state.zbar (single shared Zbar field).
      const PhaseEnergySnapshot e_before_inject = capture_phase_energy();
      const double inject_laser_source_terms_skipped_energy =
          inject_laser_source_terms(
              state, cfg, dt_op, &source_E_floor, &source_clamp_count);
      record_mesh_attr_zero(
          diagnostics::mesh_attribution::MeshDeformSource::LaserHeating);
      log_phase_energy("laser_inject_only", e_before_inject);
      step_E_numerical_loss += inject_laser_source_terms_skipped_energy;
      step_E_floor += std::max(source_E_floor, 0.0);
      step_clamp_count += std::max(source_clamp_count, 0);
      refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
      const double laser_delta_E_ext =
          std::max(commanded - escaped - state.E_cbet_iaw_step, 0.0) -
          std::max(inject_laser_source_terms_skipped_energy, 0.0) +
          std::max(source_E_floor, 0.0);
      record_operator_energy("laser", op_energy_before, laser_delta_E_ext);
      wj_audit_log("laser", wj_b_laser);
      if (wj_op_audit_active) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(17)
            << "[wj_laser_audit] step=" << (state.step + 1)
            << " commanded=" << commanded
            << " escaped_power_dt=" << escaped
            << " sum_laser_dep=" << sum_laser_dep_after_transfer
            << " skipped=" << inject_laser_source_terms_skipped_energy
            << " floor=" << source_E_floor
            << " cmd_minus_esc_minus_dep="
            << (commanded - escaped - sum_laser_dep_after_transfer);
        core::log_info(oss.str());
      }
      if (verbose_phase_timing) {
        step_laser_ms += ms(t_phase_start, Clock::now());
      }
      if (phase_energy_trace_enabled) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(17)
            << "[phase_energy_trace] step=" << (state.step + 1)
            << " label=laser_callback_extra"
            << " commanded=" << commanded
            << " unabsorbed=" << escaped
            << " sum_laser_dep_after_transfer=" << sum_laser_dep_after_transfer
            << " inject_laser_source_terms_skipped_energy="
            << inject_laser_source_terms_skipped_energy;
        core::log_info(oss.str());
      }
      log_phase_energy("laser_callback", e_before_phase);
    };
    callbacks.burn = [&](const double dt_op, const double t_op) {
      // merge union 2026-07-18: the 1D (v2/v2-E incl. neutron heating + MC)
      // and 2D (local/Corman diffusion) burn stages evolved on separate
      // lineages; dispatch on mesh.dim, bodies kept verbatim per family.
      if (state.mesh.dim == 1) {
      (void)t_op;
      state.burn_dt_limit_s = std::numeric_limits<double>::infinity();
      state.burn_released_step = 0.0;
      state.burn_dep_e_step = 0.0;
      state.burn_dep_i_step = 0.0;
      state.burn_esc_charged_step = 0.0;
      state.burn_esc_neutron_step = 0.0;
      state.burn_nh_dep_e_step = 0.0;
      state.burn_nh_dep_i_step = 0.0;
      state.burn_nh_degraded_step = 0.0;
      state.burn_nh_escaped_step = 0.0;
      state.burn_Ti_burn_dt_eV = 0.0;
      state.burn_Ti_burn_dd_eV = 0.0;
      state.burn_vr2_burn_dt = 0.0;
      state.burn_vr2_burn_dd = 0.0;
      state.burn_neutron_mean_shift_dt_keV = 0.0;
      state.burn_neutron_sigma_thermal_dt_keV = 0.0;
      state.burn_neutron_sigma_total_dt_keV = 0.0;
      state.burn_neutron_mean_shift_dd_keV = 0.0;
      state.burn_neutron_sigma_thermal_dd_keV = 0.0;
      state.burn_neutron_sigma_total_dd_keV = 0.0;
      if (!cfg.burn.enabled) {
        return;
      }
      const PhaseEnergySnapshot e_before_phase = capture_phase_energy();
      const auto op_energy_before =
          operator_energy_tracker.enabled() ? capture_operator_energy_cap()
                                            : OpEnergyCapture{};
      const WjOpAuditSnapshot wj_b_burn =
          wj_op_audit_active ? wj_audit_capture() : WjOpAuditSnapshot{};
      state.ensure_cell_material_props(cfg);
      const int n_cells = static_cast<int>(state.rho.size());
      const int n_mat = static_cast<int>(cfg.materials.materials.size());
      std::vector<double> rho_h(n_cells, 0.0);
      std::vector<double> vol_h(n_cells, 0.0);
      std::vector<double> Te_h(n_cells, 0.0);
      std::vector<double> Ti_h(n_cells, 0.0);
      std::vector<double> zbar_h(n_cells, 0.0);
      std::vector<double> A_eff_h(n_cells, 0.0);
      std::vector<double> ee_h(n_cells, 0.0);
      std::vector<double> ei_h(n_cells, 0.0);
      std::vector<double> vf_h(state.volFrac.size(), 0.0);
      std::vector<double> x_r_h(state.x_r.size(), 0.0);
      std::vector<double> v_r_node_h(state.v_r.size(), 0.0);
      std::vector<double> v_r_h(n_cells, 0.0);
      state.rho.copy_to_host(rho_h.data());
      state.vol.copy_to_host(vol_h.data());
      state.Te.copy_to_host(Te_h.data());
      state.Ti.copy_to_host(Ti_h.data());
      state.zbar.copy_to_host(zbar_h.data());
      state.A_eff.copy_to_host(A_eff_h.data());
      state.ee.copy_to_host(ee_h.data());
      state.ei.copy_to_host(ei_h.data());
      state.volFrac.copy_to_host(vf_h.data());
      state.x_r.copy_to_host(x_r_h.data());
      if (!v_r_node_h.empty()) {
        state.v_r.copy_to_host(v_r_node_h.data());
      }
      if (v_r_node_h.size() >= static_cast<std::size_t>(n_cells) + 1U) {
        for (int c = 0; c < n_cells; ++c) {
          const std::size_t idx = static_cast<std::size_t>(c);
          v_r_h[idx] = 0.5 * (v_r_node_h[idx] + v_r_node_h[idx + 1]);
        }
      }

      burn::BurnStageInputs bin;
      bin.n_cells = n_cells;
      bin.n_mat = n_mat;
      bin.r_node = x_r_h.data();
      bin.rho = rho_h.data();
      bin.vol = vol_h.data();
      bin.Te_eV = Te_h.data();
      bin.Ti_eV = Ti_h.data();
      bin.v_r = v_r_h.data();
      bin.zbar = zbar_h.data();
      bin.A_eff = A_eff_h.data();
      bin.ee = ee_h.data();
      bin.ei = ei_h.data();
      bin.volFrac = vf_h.data();
      bin.fuel_mat = burn_fuel_mats.data();
      bin.n_fuel_mat = static_cast<int>(burn_fuel_mats.size());
      const auto contains_fuel = [&](const char* fuel) {
        return std::find(cfg.burn.fuels.begin(),
                         cfg.burn.fuels.end(),
                         std::string(fuel)) != cfg.burn.fuels.end();
      };
      burn::BurnStageParams bp;
      bp.channels = {contains_fuel("DT"), contains_fuel("DD"), contains_fuel("D3He")};
      bp.use_fraley_partition = (cfg.burn.partition == "fraley");
      if (cfg.burn.screening == "salpeter") {
        bp.screening_mode = static_cast<int>(burn::ScreeningMode::kSalpeter);
      } else if (cfg.burn.screening == "chugunov_dewitt") {
        bp.screening_mode = static_cast<int>(burn::ScreeningMode::kChugunovDeWitt);
      } else {
        bp.screening_mode = static_cast<int>(burn::ScreeningMode::kNone);
      }
      bp.T_floor_keV = cfg.burn.T_floor_keV;
      bp.eps_deplete = cfg.burn.eps_deplete;
      bp.subcycle_max = cfg.burn.subcycle_max;
      bp.vf_threshold = cfg.burn.vf_threshold;
      bp.explicit_source_limit = cfg.burn.explicit_source_limit;
      bp.dt_s = dt_op;
      bp.neutron_heating = cfg.burn.neutron_heating;
      bp.neutron_heating_n_mu = cfg.burn.neutron_heating_n_mu;
      std::vector<double> dE_e;
      std::vector<double> dE_i;
      const auto store_burn_neutron_diagnostics =
          [&](const burn::BurnStageResult& r) {
            if (r.w_dt > 0.0) {
              state.burn_Ti_burn_dt_eV = r.wTi_dt / r.w_dt;
              state.burn_vr2_burn_dt = r.wvr2_dt / r.w_dt;
              const burn::BryskMoments moments = burn::brysk_moments(
                  0, state.burn_Ti_burn_dt_eV * 1.0e-3,
                  state.burn_vr2_burn_dt);
              state.burn_neutron_mean_shift_dt_keV = moments.mean_shift_keV;
              state.burn_neutron_sigma_thermal_dt_keV =
                  moments.sigma_thermal_keV;
              state.burn_neutron_sigma_total_dt_keV = moments.sigma_total_keV;
            }
            if (r.w_dd > 0.0) {
              state.burn_Ti_burn_dd_eV = r.wTi_dd / r.w_dd;
              state.burn_vr2_burn_dd = r.wvr2_dd / r.w_dd;
              const burn::BryskMoments moments = burn::brysk_moments(
                  1, state.burn_Ti_burn_dd_eV * 1.0e-3,
                  state.burn_vr2_burn_dd);
              state.burn_neutron_mean_shift_dd_keV = moments.mean_shift_keV;
              state.burn_neutron_sigma_thermal_dd_keV =
                  moments.sigma_thermal_keV;
              state.burn_neutron_sigma_total_dd_keV = moments.sigma_total_keV;
            }
          };
      if (cfg.burn.scheme == "diffusion") {
        bp.scheme = 1;
        std::vector<double> S_birth;
        const burn::BurnStageResult r =
            burn::compute_burn_step_1d(bin,
                                       bp,
                                       burn_partition_table,
                                       state.burn_n_host,
                                       dE_e,
                                       dE_i,
                                       state.burn_rate_host,
                                       state.burn_Q_e_host,
                                       state.burn_Q_i_host,
                                       &S_birth);
        store_burn_neutron_diagnostics(r);
        std::vector<double> nh_dE_e;
        std::vector<double> nh_dE_i;
        if (cfg.burn.neutron_heating) {
          nh_dE_e = dE_e;
          nh_dE_i = dE_i;
        }

        burn::CormanParams cp;
        cp.n_groups = cfg.burn.diffusion_groups;
        cp.E_min_keV = cfg.burn.diffusion_E_min_keV;
        cp.E_max_keV = 15500.0;
        cp.lnL_e = 0.0;
        cp.lnL_I = 0.0;

        const std::size_t n_cells_sz = static_cast<std::size_t>(n_cells);
        const std::size_t n_groups_sz =
            static_cast<std::size_t>(cp.n_groups);
        const std::size_t slot_cells = n_cells_sz;
        const std::size_t slot_ng_cells = n_groups_sz * n_cells_sz;
        const std::size_t burn_Ng_size = 6U * slot_ng_cells;
        if (state.burn_Ng.size() != burn_Ng_size) {
          state.burn_Ng.reset(burn_Ng_size);
        }
        if (state.burn_Ng_work.size() != burn_Ng_size) {
          state.burn_Ng_work.reset(burn_Ng_size);
        }
        if (state.burn_dep_e_dev.size() != n_cells_sz) {
          state.burn_dep_e_dev.reset(n_cells_sz);
        }
        if (state.burn_dep_i_dev.size() != n_cells_sz) {
          state.burn_dep_i_dev.reset(n_cells_sz);
        }
        zero_device_buffer_prefix(state.burn_dep_e_dev, n_cells_sz,
                                  "burn diffusion dep_e");
        zero_device_buffer_prefix(state.burn_dep_i_dev, n_cells_sz,
                                  "burn diffusion dep_i");
        ensure_device_buffer_capacity(
            state.burn_corman_scratch,
            burn::corman_diffusion_scratch_bytes(cp.n_groups, n_cells),
            "burn diffusion Corman scratch");

        std::vector<double> ne_h(n_cells_sz, 0.0);
        for (int c = 0; c < n_cells; ++c) {
          const double denom = A_eff_h[static_cast<std::size_t>(c)] *
                               core::constants::proton_mass;
          ne_h[static_cast<std::size_t>(c)] =
              (denom > 0.0)
                  ? zbar_h[static_cast<std::size_t>(c)] *
                        rho_h[static_cast<std::size_t>(c)] / denom
                  : 0.0;
        }
        copy_host_to_device_prefix(burn_r_node_dev, x_r_h,
                                   "burn diffusion r_node");
        copy_host_to_device_prefix(burn_vol_dev, vol_h,
                                   "burn diffusion vol");
        copy_host_to_device_prefix(burn_rho_dev, rho_h,
                                   "burn diffusion rho");
        copy_host_to_device_prefix(burn_Te_dev, Te_h,
                                   "burn diffusion Te");
        copy_host_to_device_prefix(burn_Ti_dev, Ti_h,
                                   "burn diffusion Ti");
        copy_host_to_device_prefix(burn_ne_dev, ne_h,
                                   "burn diffusion ne");
        copy_host_to_device_prefix(burn_S_birth_dev, S_birth,
                                   "burn diffusion S_birth");

        std::vector<double> burn_Ng_h;
        state.burn_Ng.copy_to_host(burn_Ng_h);
        constexpr int kChargedSpecies[6] = {
            burn::kHe4, burn::kT, burn::kP, burn::kHe3, burn::kHe4, burn::kP};
        constexpr double kChargedMeV[6] = {
            3.540, 1.010, 3.023, 0.820, 3.690, 14.663};
        double escaped_total = 0.0;
        double inflight_total = 0.0;
        double sourced_total = 0.0;
        for (int slot = 0; slot < 6; ++slot) {
          const std::size_t source_offset =
              static_cast<std::size_t>(slot) * slot_cells;
          const std::size_t Ng_offset =
              static_cast<std::size_t>(slot) * slot_ng_cells;
          const bool has_source =
              any_nonzero_span(S_birth, source_offset, slot_cells);
          const bool has_inflight =
              any_nonzero_span(burn_Ng_h, Ng_offset, slot_ng_cells);
          if (!has_source && !has_inflight) {
            continue;
          }
          const int species = kChargedSpecies[slot];
          double* const Ng_density = state.burn_Ng_work.data() + Ng_offset;
          burn::scale_rows_by_cell(
              Ng_density,
              state.burn_Ng.data() + Ng_offset,
              burn_rho_dev.data(),
              cp.n_groups,
              n_cells);
          const burn::CormanStepResult cr = burn::corman_diffusion_step(
              cp,
              n_cells,
              burn::species_A(species),
              charged_species_Z(species),
              kChargedMeV[slot] * 1000.0,
              burn_r_node_dev.data(),
              burn_vol_dev.data(),
              burn_rho_dev.data(),
              burn_Te_dev.data(),
              burn_Ti_dev.data(),
              burn_ne_dev.data(),
              burn_S_birth_dev.data() + source_offset,
              dt_op,
              Ng_density,
              state.burn_dep_e_dev.data(),
              state.burn_dep_i_dev.data(),
              state.burn_corman_scratch.data(),
              state.burn_corman_scratch.size(),
              nullptr);
          burn::divide_rows_by_cell(
              state.burn_Ng.data() + Ng_offset,
              Ng_density,
              burn_rho_dev.data(),
              cp.n_groups,
              n_cells);
          escaped_total += cr.escaped_erg;
          inflight_total += cr.inflight_erg;
          sourced_total += cr.sourced_erg;
        }
        (void)sourced_total;

        state.burn_dep_e_dev.copy_to_host(dE_e);
        state.burn_dep_i_dev.copy_to_host(dE_i);
        if (cfg.burn.neutron_heating && !nh_dE_e.empty()) {
          for (int c = 0; c < n_cells; ++c) {
            const std::size_t idx = static_cast<std::size_t>(c);
            dE_e[idx] += nh_dE_e[idx];
            dE_i[idx] += nh_dE_i[idx];
          }
        }
        double dep_e_total = 0.0;
        double dep_i_total = 0.0;
        for (int c = 0; c < n_cells; ++c) {
          const std::size_t idx = static_cast<std::size_t>(c);
          dep_e_total += dE_e[idx];
          dep_i_total += dE_i[idx];
          const double denom = rho_h[idx] * vol_h[idx];
          if (denom > 1.0e-30) {
            state.burn_eps_cum_host[idx] += (dE_e[idx] + dE_i[idx]) / denom;
          }
          if (vol_h[idx] > 0.0 && dt_op > 0.0) {
            state.burn_Q_e_host[idx] = dE_e[idx] / (vol_h[idx] * dt_op);
            state.burn_Q_i_host[idx] = dE_i[idx] / (vol_h[idx] * dt_op);
          } else {
            state.burn_Q_e_host[idx] = 0.0;
            state.burn_Q_i_host[idx] = 0.0;
          }
        }

        double dt_limit = std::numeric_limits<double>::infinity();
        for (int c = 0; c < n_cells; ++c) {
          const std::size_t idx = static_cast<std::size_t>(c);
          const double P_dep_c = (dE_e[idx] + dE_i[idx]) / dt_op;
          if (P_dep_c > 0.0) {
            const double e_cell =
                rho_h[idx] * vol_h[idx] * std::max(ee_h[idx] + ei_h[idx], 0.0);
            if (e_cell > 0.0) {
              const double cand =
                  cfg.burn.explicit_source_limit * e_cell / P_dep_c;
              dt_limit = std::min(dt_limit, cand);
            }
          }
        }

        double burn_E_floor = 0.0;
        int burn_clamps = 0;
        const double skipped = inject_burn_source_terms(
            state, cfg, dE_e, dE_i, &burn_E_floor, &burn_clamps);
        refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
        state.burn_enabled_any = true;
        state.burn_diffusion_any = true;
        state.burn_released_step = r.released_charged + r.released_neutron;
        state.burn_dep_e_step = dep_e_total;
        state.burn_dep_i_step = dep_i_total;
        state.burn_esc_charged_step = escaped_total;
        state.burn_esc_neutron_step = r.esc_neutron;
        state.burn_nh_dep_e_step = r.nh_dep_e;
        state.burn_nh_dep_i_step = r.nh_dep_i;
        state.burn_nh_degraded_step = r.nh_degraded;
        state.burn_nh_escaped_step = r.nh_escaped;
        state.burn_dt_limit_s = dt_limit;
        state.E_burn_released += state.burn_released_step;
        state.E_burn_dep_e += dep_e_total;
        state.E_burn_dep_i += dep_i_total;
        state.E_burn_esc_charged += escaped_total;
        state.E_burn_esc_neutron += r.esc_neutron;
        state.E_burn_inflight = inflight_total;
        state.N_burn_neutrons_dt += r.n_neutrons_dt;
        state.N_burn_neutrons_dd += r.n_neutrons_dd;
        step_E_numerical_loss += skipped;
        step_E_floor += std::max(burn_E_floor, 0.0);
        step_clamp_count += std::max(burn_clamps, 0);
        step_E_burn_in += dep_e_total + dep_i_total;
        const double burn_delta_E_ext =
            std::max(dep_e_total + dep_i_total - skipped, 0.0) +
            std::max(burn_E_floor, 0.0);
        record_operator_energy("burn", op_energy_before, burn_delta_E_ext);
      } else if (cfg.burn.scheme == "mc") {
        bp.scheme = 1;
        std::vector<double> S_birth;
        const burn::BurnStageResult r =
            burn::compute_burn_step_1d(bin,
                                       bp,
                                       burn_partition_table,
                                       state.burn_n_host,
                                       dE_e,
                                       dE_i,
                                       state.burn_rate_host,
                                       state.burn_Q_e_host,
                                       state.burn_Q_i_host,
                                       &S_birth);
        store_burn_neutron_diagnostics(r);
        std::vector<double> nh_dE_e;
        std::vector<double> nh_dE_i;
        if (cfg.burn.neutron_heating) {
          nh_dE_e = dE_e;
          nh_dE_i = dE_i;
        }

        const std::size_t n_cells_sz = static_cast<std::size_t>(n_cells);
        const std::size_t mc_capacity_size =
            burn_mc_pool_capacity_size(n_cells,
                                       cfg.burn.mc_particles_per_cell);
        const int mc_capacity = static_cast<int>(mc_capacity_size);
        TENRYU_ASSERT(state.burn_mc_live >= 0 &&
                          state.burn_mc_live <= mc_capacity,
                      "burn MC live count outside pool capacity");
        const std::size_t mc_live =
            static_cast<std::size_t>(state.burn_mc_live);
        ensure_burn_mc_pool_field(state.burn_mc_r, mc_capacity_size, mc_live,
                                  "burn MC r pool");
        ensure_burn_mc_pool_field(state.burn_mc_mu, mc_capacity_size, mc_live,
                                  "burn MC mu pool");
        ensure_burn_mc_pool_field(state.burn_mc_E, mc_capacity_size, mc_live,
                                  "burn MC E pool");
        ensure_burn_mc_pool_field(state.burn_mc_w, mc_capacity_size, mc_live,
                                  "burn MC w pool");
        ensure_burn_mc_pool_field(state.burn_mc_slot, mc_capacity_size, mc_live,
                                  "burn MC slot pool");
        ensure_burn_mc_pool_field(state.burn_mc_alive, mc_capacity_size, mc_live,
                                  "burn MC alive pool");
        if (state.burn_dep_e_dev.size() != n_cells_sz) {
          state.burn_dep_e_dev.reset(n_cells_sz);
        }
        if (state.burn_dep_i_dev.size() != n_cells_sz) {
          state.burn_dep_i_dev.reset(n_cells_sz);
        }
        zero_device_buffer_prefix(state.burn_dep_e_dev, n_cells_sz,
                                  "burn MC dep_e");
        zero_device_buffer_prefix(state.burn_dep_i_dev, n_cells_sz,
                                  "burn MC dep_i");

        std::vector<double> ne_h(n_cells_sz, 0.0);
        for (int c = 0; c < n_cells; ++c) {
          const double denom = A_eff_h[static_cast<std::size_t>(c)] *
                               core::constants::proton_mass;
          ne_h[static_cast<std::size_t>(c)] =
              (denom > 0.0)
                  ? zbar_h[static_cast<std::size_t>(c)] *
                        rho_h[static_cast<std::size_t>(c)] / denom
                  : 0.0;
        }
        copy_host_to_device_prefix(burn_r_node_dev, x_r_h,
                                   "burn MC r_node");
        copy_host_to_device_prefix(burn_vol_dev, vol_h,
                                   "burn MC vol");
        copy_host_to_device_prefix(burn_rho_dev, rho_h,
                                   "burn MC rho");
        copy_host_to_device_prefix(burn_Te_dev, Te_h,
                                   "burn MC Te");
        copy_host_to_device_prefix(burn_Ti_dev, Ti_h,
                                   "burn MC Ti");
        copy_host_to_device_prefix(burn_ne_dev, ne_h,
                                   "burn MC ne");
        copy_host_to_device_prefix(burn_S_birth_dev, S_birth,
                                   "burn MC S_birth");

        burn::McParams mp;
        mp.E_min_keV = cfg.burn.diffusion_E_min_keV;
        mp.particles_per_cell = cfg.burn.mc_particles_per_cell;
        mp.seed = static_cast<unsigned long long>(cfg.main.seed);
        const burn::McStepResult mr = burn::mc_transport_step(
            mp,
            n_cells,
            static_cast<long long>(state.step),
            burn_r_node_dev.data(),
            burn_rho_dev.data(),
            burn_Te_dev.data(),
            burn_Ti_dev.data(),
            burn_ne_dev.data(),
            burn_S_birth_dev.data(),
            burn_vol_dev.data(),
            dt_op,
            state.burn_mc_r.data(),
            state.burn_mc_mu.data(),
            state.burn_mc_E.data(),
            state.burn_mc_w.data(),
            state.burn_mc_slot.data(),
            state.burn_mc_alive.data(),
            mc_capacity,
            &state.burn_mc_live,
            state.burn_dep_e_dev.data(),
            state.burn_dep_i_dev.data(),
            nullptr);
        TENRYU_ASSERT(!mr.overflow, "burn MC particle pool capacity overflow");

        state.burn_dep_e_dev.copy_to_host(dE_e);
        state.burn_dep_i_dev.copy_to_host(dE_i);
        if (cfg.burn.neutron_heating && !nh_dE_e.empty()) {
          for (int c = 0; c < n_cells; ++c) {
            const std::size_t idx = static_cast<std::size_t>(c);
            dE_e[idx] += nh_dE_e[idx];
            dE_i[idx] += nh_dE_i[idx];
          }
        }
        double dep_e_total = 0.0;
        double dep_i_total = 0.0;
        for (int c = 0; c < n_cells; ++c) {
          const std::size_t idx = static_cast<std::size_t>(c);
          dep_e_total += dE_e[idx];
          dep_i_total += dE_i[idx];
          const double denom = rho_h[idx] * vol_h[idx];
          if (denom > 1.0e-30) {
            state.burn_eps_cum_host[idx] += (dE_e[idx] + dE_i[idx]) / denom;
          }
          if (vol_h[idx] > 0.0 && dt_op > 0.0) {
            state.burn_Q_e_host[idx] = dE_e[idx] / (vol_h[idx] * dt_op);
            state.burn_Q_i_host[idx] = dE_i[idx] / (vol_h[idx] * dt_op);
          } else {
            state.burn_Q_e_host[idx] = 0.0;
            state.burn_Q_i_host[idx] = 0.0;
          }
        }

        double dt_limit = std::numeric_limits<double>::infinity();
        for (int c = 0; c < n_cells; ++c) {
          const std::size_t idx = static_cast<std::size_t>(c);
          const double P_dep_c = (dE_e[idx] + dE_i[idx]) / dt_op;
          if (P_dep_c > 0.0) {
            const double e_cell =
                rho_h[idx] * vol_h[idx] * std::max(ee_h[idx] + ei_h[idx], 0.0);
            if (e_cell > 0.0) {
              const double cand =
                  cfg.burn.explicit_source_limit * e_cell / P_dep_c;
              dt_limit = std::min(dt_limit, cand);
            }
          }
        }

        double burn_E_floor = 0.0;
        int burn_clamps = 0;
        const double skipped = inject_burn_source_terms(
            state, cfg, dE_e, dE_i, &burn_E_floor, &burn_clamps);
        refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
        state.burn_enabled_any = true;
        state.burn_mc_any = true;
        state.burn_released_step = r.released_charged + r.released_neutron;
        state.burn_dep_e_step = dep_e_total;
        state.burn_dep_i_step = dep_i_total;
        state.burn_esc_charged_step = mr.escaped_erg;
        state.burn_esc_neutron_step = r.esc_neutron;
        state.burn_nh_dep_e_step = r.nh_dep_e;
        state.burn_nh_dep_i_step = r.nh_dep_i;
        state.burn_nh_degraded_step = r.nh_degraded;
        state.burn_nh_escaped_step = r.nh_escaped;
        state.burn_dt_limit_s = dt_limit;
        state.E_burn_released += state.burn_released_step;
        state.E_burn_dep_e += dep_e_total;
        state.E_burn_dep_i += dep_i_total;
        state.E_burn_esc_charged += mr.escaped_erg;
        state.E_burn_esc_neutron += r.esc_neutron;
        state.E_burn_inflight = mr.inflight_erg;
        state.N_burn_neutrons_dt += r.n_neutrons_dt;
        state.N_burn_neutrons_dd += r.n_neutrons_dd;
        step_E_numerical_loss += skipped;
        step_E_floor += std::max(burn_E_floor, 0.0);
        step_clamp_count += std::max(burn_clamps, 0);
        step_E_burn_in += dep_e_total + dep_i_total;
        const double burn_delta_E_ext =
            std::max(dep_e_total + dep_i_total - skipped, 0.0) +
            std::max(burn_E_floor, 0.0);
        record_operator_energy("burn", op_energy_before, burn_delta_E_ext);
      } else {
        const burn::BurnStageResult r =
            burn::compute_burn_step_1d(bin,
                                       bp,
                                       burn_partition_table,
                                       state.burn_n_host,
                                       dE_e,
                                       dE_i,
                                       state.burn_rate_host,
                                       state.burn_Q_e_host,
                                       state.burn_Q_i_host);
        store_burn_neutron_diagnostics(r);
        for (int c = 0; c < n_cells; ++c) {
          const double denom = rho_h[c] * vol_h[c];
          if (denom > 1.0e-30) {
            state.burn_eps_cum_host[c] += (dE_e[c] + dE_i[c]) / denom;
          }
        }
        double burn_E_floor = 0.0;
        int burn_clamps = 0;
        const double skipped = inject_burn_source_terms(
            state, cfg, dE_e, dE_i, &burn_E_floor, &burn_clamps);
        refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
        state.burn_enabled_any = true;
        state.burn_released_step = r.released_charged + r.released_neutron;
        state.burn_dep_e_step = r.dep_e;
        state.burn_dep_i_step = r.dep_i;
        state.burn_esc_charged_step = r.esc_charged;
        state.burn_esc_neutron_step = r.esc_neutron;
        state.burn_nh_dep_e_step = r.nh_dep_e;
        state.burn_nh_dep_i_step = r.nh_dep_i;
        state.burn_nh_degraded_step = r.nh_degraded;
        state.burn_nh_escaped_step = r.nh_escaped;
        state.burn_dt_limit_s = r.dt_limit_s;
        state.E_burn_released += state.burn_released_step;
        state.E_burn_dep_e += r.dep_e;
        state.E_burn_dep_i += r.dep_i;
        state.E_burn_esc_charged += r.esc_charged;
        state.E_burn_esc_neutron += r.esc_neutron;
        state.N_burn_neutrons_dt += r.n_neutrons_dt;
        state.N_burn_neutrons_dd += r.n_neutrons_dd;
        step_E_numerical_loss += skipped;
        step_E_floor += std::max(burn_E_floor, 0.0);
        step_clamp_count += std::max(burn_clamps, 0);
        step_E_burn_in += r.dep_e + r.dep_i;
        const double burn_delta_E_ext =
            std::max(r.dep_e + r.dep_i - skipped, 0.0) +
            std::max(burn_E_floor, 0.0);
        record_operator_energy("burn", op_energy_before, burn_delta_E_ext);
      }
      wj_audit_log("burn", wj_b_burn);
      log_phase_energy("burn_callback", e_before_phase);
      } else {
      (void)t_op;
      state.burn_dt_limit_s = std::numeric_limits<double>::infinity();
      state.burn_released_step = 0.0;
      state.burn_dep_e_step = 0.0;
      state.burn_dep_i_step = 0.0;
      state.burn_esc_charged_step = 0.0;
      state.burn_esc_neutron_step = 0.0;
      if (!cfg.burn.enabled) {
        return;
      }
      const PhaseEnergySnapshot e_before_phase = capture_phase_energy();
      const auto op_energy_before =
          operator_energy_tracker.enabled() ? capture_operator_energy_cap()
                                            : OpEnergyCapture{};
      const WjOpAuditSnapshot wj_b_burn =
          wj_op_audit_active ? wj_audit_capture() : WjOpAuditSnapshot{};
      if (part_info.n_ranks > 1 && is_1d) {
        // Full-line burn inputs (Option C 1D replication, spec
        // mpi_m18d_laser_burn_spec.md §2 d2): the host stage loops the
        // whole line on every rank; gather owner-true matter (and
        // volFrac before the material-prop rebuild below) so the
        // replicated stage — and its host inventories — stay
        // rank-identical by induction.
        allgatherv_1d_cell_line(state.rho.data(), 1);
        allgatherv_1d_cell_line(state.vol.data(), 1);
        allgatherv_1d_cell_line(state.Te.data(), 1);
        allgatherv_1d_cell_line(state.Ti.data(), 1);
        allgatherv_1d_cell_line(state.zbar.data(), 1);
        allgatherv_1d_cell_line(state.ee.data(), 1);
        allgatherv_1d_cell_line(state.ei.data(), 1);
        if (!state.volFrac.empty() && state.mesh.topo.n_cells > 0 &&
            state.volFrac.size() %
                    static_cast<std::size_t>(state.mesh.topo.n_cells) ==
                0) {
          allgatherv_1d_cell_line(
              state.volFrac.data(),
              static_cast<int>(
                  state.volFrac.size() /
                  static_cast<std::size_t>(state.mesh.topo.n_cells)));
          state.invalidate_cell_material_props();
        }
      }
      // 1D replication counts every burn tally on every rank — the step
      // ledger must take them ONCE (rank 0); 2D owned-window stages
      // produce disjoint partials (share 1) completed by the budget
      // Allreduce.
      const double burn_tally_share =
          (part_info.n_ranks > 1 && is_1d && part_info.rank != 0) ? 0.0
                                                                  : 1.0;
      state.ensure_cell_material_props(cfg);
      const int n_cells = static_cast<int>(state.rho.size());
      const int n_mat = static_cast<int>(cfg.materials.materials.size());
      std::vector<double> rho_h(n_cells, 0.0);
      std::vector<double> vol_h(n_cells, 0.0);
      std::vector<double> Te_h(n_cells, 0.0);
      std::vector<double> Ti_h(n_cells, 0.0);
      std::vector<double> zbar_h(n_cells, 0.0);
      std::vector<double> A_eff_h(n_cells, 0.0);
      std::vector<double> ee_h(n_cells, 0.0);
      std::vector<double> ei_h(n_cells, 0.0);
      std::vector<double> vf_h(state.volFrac.size(), 0.0);
      state.rho.copy_to_host(rho_h.data());
      state.vol.copy_to_host(vol_h.data());
      state.Te.copy_to_host(Te_h.data());
      state.Ti.copy_to_host(Ti_h.data());
      state.zbar.copy_to_host(zbar_h.data());
      state.A_eff.copy_to_host(A_eff_h.data());
      state.ee.copy_to_host(ee_h.data());
      state.ei.copy_to_host(ei_h.data());
      state.volFrac.copy_to_host(vf_h.data());

      burn::BurnStage2DInputs bin;
      bin.n_cells = n_cells;
      bin.n_mat = n_mat;
      bin.rho = rho_h.data();
      bin.vol = vol_h.data();
      bin.Te_eV = Te_h.data();
      bin.Ti_eV = Ti_h.data();
      bin.zbar = zbar_h.data();
      bin.A_eff = A_eff_h.data();
      bin.ee = ee_h.data();
      bin.ei = ei_h.data();
      bin.volFrac = vf_h.data();
      bin.fuel_mat = burn_fuel_mats.data();
      bin.n_fuel_mat = static_cast<int>(burn_fuel_mats.size());
      if (part_info.n_ranks > 1 && is_2d) {
        // Owned-window 2D stage (Option C): tallies become disjoint
        // partials completed by the budget Allreduce; 1D keeps the full
        // (replicated) window.
        const auto burn_cw = state.owned_cell_window(n_cells);
        bin.c_begin = burn_cw.begin;
        bin.c_end = burn_cw.end;
      }
      const auto contains_fuel = [&](const char* fuel) {
        return std::find(cfg.burn.fuels.begin(),
                         cfg.burn.fuels.end(),
                         std::string(fuel)) != cfg.burn.fuels.end();
      };
      burn::BurnStage2DParams bp;
      bp.channels = {contains_fuel("DT"), contains_fuel("DD"), contains_fuel("D3He")};
      bp.use_fraley_partition = (cfg.burn.partition == "fraley");
      if (cfg.burn.screening == "salpeter") {
        bp.screening_mode = static_cast<int>(burn::ScreeningMode::kSalpeter);
      } else if (cfg.burn.screening == "chugunov_dewitt") {
        bp.screening_mode = static_cast<int>(burn::ScreeningMode::kChugunovDeWitt);
      } else {
        bp.screening_mode = static_cast<int>(burn::ScreeningMode::kNone);
      }
      bp.T_floor_keV = cfg.burn.T_floor_keV;
      bp.eps_deplete = cfg.burn.eps_deplete;
      bp.subcycle_max = cfg.burn.subcycle_max;
      bp.vf_threshold = cfg.burn.vf_threshold;
      bp.explicit_source_limit = cfg.burn.explicit_source_limit;
      bp.dt_s = dt_op;
      std::vector<double> dE_e;
      std::vector<double> dE_i;
      if (cfg.burn.scheme == "mc") {
        TENRYU_ASSERT(false,
                      "unreachable: 2D burn mc is refused at validate");
      } else if (cfg.burn.scheme == "diffusion") {
        bp.scheme = 1;
        std::vector<double> S_birth;
        const burn::BurnStage2DResult r =
            burn::compute_burn_step_2d(bin,
                                       bp,
                                       burn_partition_table,
                                       state.burn_n_host,
                                       dE_e,
                                       dE_i,
                                       state.burn_rate_host,
                                       state.burn_Q_e_host,
                                       state.burn_Q_i_host,
                                       &S_birth);

        burn::CormanParams cp;
        cp.n_groups = cfg.burn.diffusion_groups;
        cp.E_min_keV = cfg.burn.diffusion_E_min_keV;
        cp.E_max_keV = 15500.0;
        cp.lnL_e = 0.0;
        cp.lnL_I = 0.0;

        const std::size_t n_cells_sz = static_cast<std::size_t>(n_cells);
        const std::size_t n_groups_sz =
            static_cast<std::size_t>(cp.n_groups);
        const std::size_t slot_cells = n_cells_sz;
        const std::size_t slot_ng_cells = n_groups_sz * n_cells_sz;
        const std::size_t burn_Ng_size = 6U * slot_ng_cells;
        if (state.burn_Ng.size() != burn_Ng_size) {
          state.burn_Ng.reset(burn_Ng_size);
        }
        if (state.burn_Ng_work.size() != burn_Ng_size) {
          state.burn_Ng_work.reset(burn_Ng_size);
        }
        if (state.burn_dep_e_dev.size() != n_cells_sz) {
          state.burn_dep_e_dev.reset(n_cells_sz);
        }
        if (state.burn_dep_i_dev.size() != n_cells_sz) {
          state.burn_dep_i_dev.reset(n_cells_sz);
        }
        const cudaError_t dep_e_zero = cudaMemset(
            state.burn_dep_e_dev.data(), 0, n_cells_sz * sizeof(double));
        TENRYU_ASSERT(dep_e_zero == cudaSuccess,
                      "burn 2D diffusion dep_e cudaMemset failed");
        const cudaError_t dep_i_zero = cudaMemset(
            state.burn_dep_i_dev.data(), 0, n_cells_sz * sizeof(double));
        TENRYU_ASSERT(dep_i_zero == cudaSuccess,
                      "burn 2D diffusion dep_i cudaMemset failed");

        std::vector<double> ne_h(n_cells_sz, 0.0);
        for (int c = 0; c < n_cells; ++c) {
          const double denom = A_eff_h[static_cast<std::size_t>(c)] *
                               core::constants::proton_mass;
          ne_h[static_cast<std::size_t>(c)] =
              (denom > 0.0)
                  ? zbar_h[static_cast<std::size_t>(c)] *
                        rho_h[static_cast<std::size_t>(c)] / denom
                  : 0.0;
        }
        double* const ne_dev = static_cast<double*>(core::device_scratch_acquire(
            "burn:cb2d:ne", n_cells_sz * sizeof(double)));
        double* const S_birth_dev =
            static_cast<double*>(core::device_scratch_acquire(
                "burn:cb2d:S_birth", S_birth.size() * sizeof(double)));
        const cudaError_t ne_copy =
            cudaMemcpy(ne_dev, ne_h.data(), n_cells_sz * sizeof(double),
                       cudaMemcpyHostToDevice);
        TENRYU_ASSERT(ne_copy == cudaSuccess,
                      "burn 2D diffusion ne cudaMemcpy H2D failed");
        const cudaError_t source_copy =
            cudaMemcpy(S_birth_dev, S_birth.data(),
                       S_birth.size() * sizeof(double),
                       cudaMemcpyHostToDevice);
        TENRYU_ASSERT(source_copy == cudaSuccess,
                      "burn 2D diffusion S_birth cudaMemcpy H2D failed");

        const auto boundary_class = [](const std::string& value) {
          return (value == "axis" || value == "reflect") ? 0 : 1;
        };
        burn::Corman2DBc corman_bc;
        corman_bc.r_inner =
            boundary_class(cfg.numerics.hydro.boundary_2d.r_inner);
        corman_bc.r_outer =
            boundary_class(cfg.numerics.hydro.boundary_2d.r_outer);
        corman_bc.z_bottom =
            boundary_class(cfg.numerics.hydro.boundary_2d.z_bottom);
        corman_bc.z_top =
            boundary_class(cfg.numerics.hydro.boundary_2d.z_top);

        std::vector<double> burn_Ng_h;
        state.burn_Ng.copy_to_host(burn_Ng_h);
        constexpr int kChargedSpecies[6] = {
            burn::kHe4, burn::kT, burn::kP, burn::kHe3, burn::kHe4, burn::kP};
        constexpr double kChargedMeV[6] = {
            3.540, 1.010, 3.023, 0.820, 3.690, 14.663};
        double escaped_total = 0.0;
        double inflight_total = 0.0;
        double sourced_total = 0.0;
        const int nr = state.mesh.topo.nr;
        const int nz = state.mesh.topo.nz;
        for (int slot = 0; slot < 6; ++slot) {
          const std::size_t source_offset =
              static_cast<std::size_t>(slot) * slot_cells;
          const std::size_t Ng_offset =
              static_cast<std::size_t>(slot) * slot_ng_cells;
          const bool has_source =
              any_nonzero_span(S_birth, source_offset, slot_cells);
          const bool has_inflight =
              any_nonzero_span(burn_Ng_h, Ng_offset, slot_ng_cells);
          if (!has_source && !has_inflight) {
            continue;
          }
          const int species = kChargedSpecies[slot];
          double* const Ng_density = state.burn_Ng_work.data() + Ng_offset;
          burn::corman2d_scale_rows_by_cell(
              Ng_density,
              state.burn_Ng.data() + Ng_offset,
              state.rho.data(),
              cp.n_groups,
              n_cells);
          const burn::CormanStepResult cr = burn::corman_diffusion_2d_step(
              cp,
              nr,
              nz,
              burn::species_A(species),
              charged_species_Z(species),
              kChargedMeV[slot] * 1000.0,
              state.x_r.data(),
              state.x_z.data(),
              state.vol.data(),
              state.rho.data(),
              state.Te.data(),
              state.Ti.data(),
              ne_dev,
              S_birth_dev + source_offset,
              dt_op,
              Ng_density,
              state.burn_dep_e_dev.data(),
              state.burn_dep_i_dev.data(),
              corman_bc,
              nullptr,
              (is_2d && part_info.n_ranks > 1) ? &part_info : nullptr,
              (is_2d && part_info.n_ranks > 1) ? &comm_buffers : nullptr,
              bin.c_begin,
              bin.c_end);
          burn::corman2d_divide_rows_by_cell(
              state.burn_Ng.data() + Ng_offset,
              Ng_density,
              state.rho.data(),
              cp.n_groups,
              n_cells);
          escaped_total += cr.escaped_erg;
          inflight_total += cr.inflight_erg;
          sourced_total += cr.sourced_erg;
        }
        (void)sourced_total;

        state.burn_dep_e_dev.copy_to_host(dE_e);
        state.burn_dep_i_dev.copy_to_host(dE_i);
        double dep_e_total = 0.0;
        double dep_i_total = 0.0;
        for (int c = 0; c < n_cells; ++c) {
          const std::size_t idx = static_cast<std::size_t>(c);
          dep_e_total += dE_e[idx];
          dep_i_total += dE_i[idx];
          const double denom = rho_h[idx] * vol_h[idx];
          if (denom > 1.0e-30) {
            state.burn_eps_cum_host[idx] += (dE_e[idx] + dE_i[idx]) / denom;
          }
          if (vol_h[idx] > 0.0 && dt_op > 0.0) {
            state.burn_Q_e_host[idx] = dE_e[idx] / (vol_h[idx] * dt_op);
            state.burn_Q_i_host[idx] = dE_i[idx] / (vol_h[idx] * dt_op);
          } else {
            state.burn_Q_e_host[idx] = 0.0;
            state.burn_Q_i_host[idx] = 0.0;
          }
        }

        double dt_limit = std::numeric_limits<double>::infinity();
        for (int c = 0; c < n_cells; ++c) {
          const std::size_t idx = static_cast<std::size_t>(c);
          const double P_dep_c = (dE_e[idx] + dE_i[idx]) / dt_op;
          if (P_dep_c > 0.0) {
            const double e_cell =
                rho_h[idx] * vol_h[idx] * std::max(ee_h[idx] + ei_h[idx], 0.0);
            if (e_cell > 0.0) {
              const double cand =
                  cfg.burn.explicit_source_limit * e_cell / P_dep_c;
              dt_limit = std::min(dt_limit, cand);
            }
          }
        }

        double burn_E_floor = 0.0;
        int burn_clamps = 0;
        const double skipped = inject_burn_source_terms(
            state, cfg, dE_e, dE_i, &burn_E_floor, &burn_clamps);
        refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
        state.burn_enabled_any = true;
        state.burn_diffusion_any = true;
        state.burn_released_step =
            burn_tally_share * (r.released_charged + r.released_neutron);
        state.burn_dep_e_step = burn_tally_share * dep_e_total;
        state.burn_dep_i_step = burn_tally_share * dep_i_total;
        state.burn_esc_charged_step = burn_tally_share * escaped_total;
        state.burn_esc_neutron_step = burn_tally_share * r.esc_neutron;
        state.burn_dt_limit_s = dt_limit;
        state.E_burn_released += state.burn_released_step;
        state.E_burn_dep_e += burn_tally_share * dep_e_total;
        state.E_burn_dep_i += burn_tally_share * dep_i_total;
        state.E_burn_esc_charged += burn_tally_share * escaped_total;
        state.E_burn_esc_neutron += burn_tally_share * r.esc_neutron;
        state.E_burn_inflight = burn_tally_share * inflight_total;
        state.N_burn_neutrons_dt += burn_tally_share * r.n_neutrons_dt;
        state.N_burn_neutrons_dd += burn_tally_share * r.n_neutrons_dd;
        step_E_numerical_loss += burn_tally_share * skipped;
        step_E_floor += burn_tally_share * std::max(burn_E_floor, 0.0);
        step_clamp_count += std::max(burn_clamps, 0);
        step_E_burn_in += burn_tally_share * (dep_e_total + dep_i_total);
        const double burn_delta_E_ext =
            std::max(dep_e_total + dep_i_total - skipped, 0.0) +
            std::max(burn_E_floor, 0.0);
        record_operator_energy("burn", op_energy_before, burn_delta_E_ext);
      } else {
        const burn::BurnStage2DResult r =
            burn::compute_burn_step_2d(bin,
                                       bp,
                                       burn_partition_table,
                                       state.burn_n_host,
                                       dE_e,
                                       dE_i,
                                       state.burn_rate_host,
                                       state.burn_Q_e_host,
                                       state.burn_Q_i_host);
        for (int c = 0; c < n_cells; ++c) {
          const double denom = rho_h[c] * vol_h[c];
          if (denom > 1.0e-30) {
            state.burn_eps_cum_host[c] += (dE_e[c] + dE_i[c]) / denom;
          }
        }
        double burn_E_floor = 0.0;
        int burn_clamps = 0;
        const double skipped = inject_burn_source_terms(
            state, cfg, dE_e, dE_i, &burn_E_floor, &burn_clamps);
        refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
        state.burn_enabled_any = true;
        state.burn_released_step =
            burn_tally_share * (r.released_charged + r.released_neutron);
        state.burn_dep_e_step = burn_tally_share * r.dep_e;
        state.burn_dep_i_step = burn_tally_share * r.dep_i;
        state.burn_esc_charged_step = burn_tally_share * r.esc_charged;
        state.burn_esc_neutron_step = burn_tally_share * r.esc_neutron;
        state.burn_dt_limit_s = r.dt_limit_s;
        state.E_burn_released += state.burn_released_step;
        state.E_burn_dep_e += burn_tally_share * r.dep_e;
        state.E_burn_dep_i += burn_tally_share * r.dep_i;
        state.E_burn_esc_charged += burn_tally_share * r.esc_charged;
        state.E_burn_esc_neutron += burn_tally_share * r.esc_neutron;
        state.N_burn_neutrons_dt += burn_tally_share * r.n_neutrons_dt;
        state.N_burn_neutrons_dd += burn_tally_share * r.n_neutrons_dd;
        step_E_numerical_loss += burn_tally_share * skipped;
        step_E_floor += burn_tally_share * std::max(burn_E_floor, 0.0);
        step_clamp_count += std::max(burn_clamps, 0);
        step_E_burn_in += burn_tally_share * (r.dep_e + r.dep_i);
        const double burn_delta_E_ext =
            std::max(r.dep_e + r.dep_i - skipped, 0.0) +
            std::max(burn_E_floor, 0.0);
        record_operator_energy("burn", op_energy_before, burn_delta_E_ext);
      }
      wj_audit_log("burn", wj_b_burn);
      log_phase_energy("burn_callback", e_before_phase);
      }
    };
    callbacks.radiation = [&](const double dt_op, const double t_op) {
      if (!cfg.radiation.enabled) {
        return;
      }
      // W-K diag hook (default-inert): skip the whole radiation operator so
      // gate decks can exercise the gamma_r=4/3 coupling on a field that
      // evolves under NOTHING else (verdict D2 "prescribed-volume adiabat";
      // the FLD stack has no deck-reachable inert regime — measured).
      if (const char* s = std::getenv("TENRYU_RAD_DIAG_SKIP_SOLVE")) {
        if (std::atoi(s) != 0) {
          return;
        }
      }
      hydro::entropy_ledger_begin(
          state, cfg, hydro::EntropyLedgerStage::Radiation);
      const PhaseEnergySnapshot e_before_phase = capture_phase_energy();
      const auto t_phase_start =
          verbose_phase_timing ? Clock::now() : Clock::time_point{};
      const WjOpAuditSnapshot wj_b_rad =
          wj_op_audit_active ? wj_audit_capture() : WjOpAuditSnapshot{};
      auto t_rad_subphase = t_phase_start;
      double rad_exchange_ms = 0.0;
      double rad_imc_ms = 0.0;
      double rad_migrate_ms = 0.0;
      double rad_dflags_ms = 0.0;
      double rad_inject_ms = 0.0;
      double rad_refresh_ms = 0.0;
      double rad_fleck_log_ms = 0.0;
      double rad_overshoot_ms = 0.0;
      double rad_safety_ms = 0.0;
      double rad_other_ms = 0.0;
      const auto mark_rad_subphase = [&](double& value) {
        if (verbose_phase_timing) {
          const auto t_now = Clock::now();
          value += ms(t_rad_subphase, t_now);
          t_rad_subphase = t_now;
        }
      };
      const long phase_safety_token =
          begin_phase_safety(wj_arena.mode, 0.0);
      const double te_max_before =
          wj_arena.mode ? 0.0
                        : compute_temperature_audit_device(state).max_te;
      const double* d_overshoot_before = nullptr;
      const int* d_overshoot_before_flags = nullptr;
      if (wj_arena.mode) {
        if (phase_safety_token >= 0) {
          const std::size_t record =
              static_cast<std::size_t>(phase_safety_token);
          d_overshoot_before =
              wj_arena.d_scalars + phase_scalar_index(
                  StepScalarSlot::PhaseSafetyBeforeMaxBegin, record);
          d_overshoot_before_flags =
              wj_arena.d_flags + phase_flag_index(
                  StepViolationFlag::PhaseSafetyBeforeAuditBegin, record);
        } else {
          const std::size_t scalar = static_cast<std::size_t>(
              StepScalarSlot::RadiationOvershootBeforeMax);
          const std::size_t flags = static_cast<std::size_t>(
              StepViolationFlag::RadiationOvershootBeforeAuditBegin);
          compute_temperature_audit_device_to_slots(
              state,
              wj_arena.d_scalars + scalar,
              wj_arena.d_flags + flags);
          d_overshoot_before = wj_arena.d_scalars + scalar;
          d_overshoot_before_flags = wj_arena.d_flags + flags;
        }
      }
      // 1D deterministic radiation (FLD/SN) runs the FULL line redundantly
      // on every rank (design doc mpi_m18_20_20260717.md §4.3): the matter
      // -side input lines are made globally valid by an Allgatherv, the
      // solver then produces bitwise rank-count-independent outputs, and
      // the (already-global) tallies enter the Allreduce'd step ledger from
      // rank 0 only. replicated_radiation_1d / replicated_tally_share /
      // allgatherv_1d_cell_line are defined in the enclosing run scope
      // (the hydro callback's gamma_r payment path shares them).
      const auto allgather_1d_radiation_inputs = [&]() {
        if (!replicated_radiation_1d) {
          return;
        }
        const int n_cells_1d = state.mesh.topo.n_cells;
        const int n_ranks = part_info.n_ranks;
        std::vector<int> cell_counts(n_ranks);
        std::vector<int> cell_displs(n_ranks);
        std::vector<int> node_counts(n_ranks);
        std::vector<int> node_displs(n_ranks);
        const int base = n_cells_1d / n_ranks;
        const int rem = n_cells_1d % n_ranks;
        for (int p = 0; p < n_ranks; ++p) {
          cell_counts[p] = base + (p < rem ? 1 : 0);
          cell_displs[p] = p * base + std::min(p, rem);
          // Disjoint node tiles aligned to cell tiles; the last rank also
          // owns the outermost node.
          node_counts[p] = cell_counts[p] + (p == n_ranks - 1 ? 1 : 0);
          node_displs[p] = cell_displs[p];
        }
        std::vector<double> line;
        const auto gather_line = [&](double* d_field, const int total,
                                     const std::vector<int>& counts,
                                     const std::vector<int>& displs) {
          if (d_field == nullptr || total <= 0) {
            return;
          }
          line.resize(static_cast<std::size_t>(total));
          const cudaError_t d2h_err =
              cudaMemcpy(line.data(), d_field,
                         static_cast<std::size_t>(total) * sizeof(double),
                         cudaMemcpyDeviceToHost);
          TENRYU_ASSERT(d2h_err == cudaSuccess,
                        "radiation line-gather D2H failed");
          const int my_count = counts[part_info.rank];
          const int my_displ = displs[part_info.rank];
          std::vector<double> send(line.begin() + my_displ,
                                   line.begin() + my_displ + my_count);
          reducer.allgatherv(send.data(), my_count, line.data(),
                             counts.data(), displs.data());
          const cudaError_t h2d_err =
              cudaMemcpy(d_field, line.data(),
                         static_cast<std::size_t>(total) * sizeof(double),
                         cudaMemcpyHostToDevice);
          TENRYU_ASSERT(h2d_err == cudaSuccess,
                        "radiation line-gather H2D failed");
        };
        const auto gather_cell_field = [&](auto& field) {
          if (!field.empty()) {
            gather_line(field.data(), static_cast<int>(field.size()),
                        cell_counts, cell_displs);
          }
        };
        gather_cell_field(state.rho);
        gather_cell_field(state.Te);
        gather_cell_field(state.Ti);
        gather_cell_field(state.vol);
        gather_cell_field(state.mass);
        gather_cell_field(state.ee);
        gather_cell_field(state.ei);
        gather_cell_field(state.zbar);
        gather_cell_field(state.cv_e);
        gather_cell_field(state.cv_i);
        // Multi-material replication inputs: gather volFrac (cell-major,
        // n_mat entries per cell) and rebuild the volfrac-derived per-cell
        // material props (A_eff/gamma_eff/dominant_material) from the now
        // -global volFrac, so the full-line SN/FLD Newton coefficient
        // paths (sn_material_newton A_eff/gamma_eff, fld_1d Fleck cv
        // fallback) see rank-count-independent values on every cell
        // (design doc mpi_m18_20_20260717.md §6h). Single-material decks:
        // the gather moves identical bytes and the recompute reproduces
        // identical values — bitwise no-op by construction.
        if (!state.volFrac.empty()) {
          TENRYU_ASSERT(state.volFrac.size() %
                                static_cast<std::size_t>(n_cells_1d) ==
                            0,
                        "1D radiation volFrac gather: size not a multiple "
                        "of n_cells");
          const int n_mat =
              static_cast<int>(state.volFrac.size() /
                               static_cast<std::size_t>(n_cells_1d));
          std::vector<int> vf_counts(n_ranks);
          std::vector<int> vf_displs(n_ranks);
          for (int p = 0; p < n_ranks; ++p) {
            vf_counts[p] = cell_counts[p] * n_mat;
            vf_displs[p] = cell_displs[p] * n_mat;
          }
          gather_line(state.volFrac.data(),
                      static_cast<int>(state.volFrac.size()), vf_counts,
                      vf_displs);
          state.invalidate_cell_material_props();
          state.ensure_cell_material_props(cfg);
        }
        // rad_E is radiation-persistent BUT windowed hydro mutates it (the
        // gamma_r work payment; ALE-1D remap would too), so the owner-true
        // line must be re-assembled before the replicated solve. Normally a
        // no-op re-gather (the post-payment gather already ran), kept as
        // the solve-entry invariant (design doc §6h).
        if (!state.rad_E.empty()) {
          allgatherv_1d_cell_line(state.rad_E.data(),
                                  std::max(cfg.radiation.groups, 1));
        }
        if (!state.x_r.empty()) {
          gather_line(state.x_r.data(), static_cast<int>(state.x_r.size()),
                      node_counts, node_displs);
        }
      };
      const auto exchange_radiation_halo = [&]() {
        if (part_info.n_ranks > 1) {
          cudaStream_t halo_stream = nullptr;
          double* rad_ptrs[] = {state.Te.data(), state.rho.data(), state.zbar.data()};
          parallel::exchange_cell_fields(part_info, comm_buffers, rad_ptrs, 3,
                                         state.mesh.topo.n_cells, halo_stream, 4);
        }
      };
      const auto migrate_radiation_particles = [&]() {
        if (part_info.n_ranks > 1) {
          cudaStream_t mig_stream = nullptr;
          parallel::detect_emigrants(imc.photon_pool(), part_info, emigrants, mig_stream);
          parallel::exchange_emigrants(
              part_info, comm_buffers, emigrants, immigrants, mig_stream);
          parallel::merge_immigrants(imc.photon_pool(), immigrants, false, mig_stream);
        }
      };
      const auto run_radiation_stage = [&](const double dt_stage,
                                           const double t_stage) {
        const bool deterministic_stage =
            cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion ||
            cfg.radiation.mode == core::RadiationMode::SnTransport;
        const double escaped_before = imc.escaped_energy_total();
        mark_rad_subphase(rad_other_ms);
        if (replicated_radiation_1d) {
          allgather_1d_radiation_inputs();
        } else {
          exchange_radiation_halo();
        }
        mark_rad_subphase(rad_exchange_ms);
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::FldSolve,
            diagnostics::RadialFourierStagePhase::Before,
            t_stage);
        imc.transport_step(state, cfg, dt_stage, part_info, &comm_buffers);
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::FldSolve,
            diagnostics::RadialFourierStagePhase::After,
            t_stage + dt_stage);
        auto fld_substage_audit_records =
            radiation::drain_fld_substage_audit_records();
        if (!fld_substage_audit_records.empty()) {
          const auto audit_cycle = static_cast<std::uint64_t>(state.step + 1);
          for (auto& record : fld_substage_audit_records) {
            record.cycle = audit_cycle;
            record.t_s = t_stage + dt_stage;
            record.dt_cycle = dt;
          }
          if (part_info.rank == 0 && history_writer.enabled()) {
            history_writer.append_fld_substage_audit_batch(
                fld_substage_audit_records);
          }
        }
        if (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion) {
          record_mesh_attr_zero(
              diagnostics::mesh_attribution::MeshDeformSource::FLDEnergyCoupling);
          record_mesh_attr_zero(
              diagnostics::mesh_attribution::MeshDeformSource::FLDRadiationPressure);
        } else if (cfg.radiation.mode == core::RadiationMode::SnTransport) {
          record_mesh_attr_zero(
              diagnostics::mesh_attribution::MeshDeformSource::SNTransport);
        }
        mark_rad_subphase(rad_imc_ms);
        const auto& holo_lo = imc.last_holo_lo_result();
        step_holo_boundary_in += holo_lo.boundary_E_in;
        step_holo_boundary_out += holo_lo.boundary_E_out;
        step_holo_matter_delta += holo_lo.matter_delta;
        step_holo_source_balance_error += holo_lo.conservation_error;
        mark_rad_subphase(rad_other_ms);
        if (!deterministic_stage) {
          migrate_radiation_particles();
        }
        mark_rad_subphase(rad_migrate_ms);
        accumulate_device_flags(state.radiation_device_flags,
                                imc.last_device_error_flags());
        mark_rad_subphase(rad_dflags_ms);
        const double escaped_after = imc.escaped_energy_total();
        const double deterministic_escaped =
            (cfg.radiation.mode == core::RadiationMode::SnTransport)
                ? state.sn_escaped_step
                : state.fld_escaped_step;
        const double deterministic_marshak =
            (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion)
                ? state.fld_marshak_in_step
                : ((cfg.radiation.mode == core::RadiationMode::SnTransport)
                       ? state.sn_marshak_in_step
                       : 0.0);
        step_E_rad_esc += deterministic_stage
                              ? replicated_tally_share *
                                    std::max(deterministic_escaped, 0.0)
                              : std::max(escaped_after - escaped_before, 0.0);
        step_E_marshak_in += deterministic_stage
                                  ? replicated_tally_share *
                                        std::max(deterministic_marshak, 0.0)
                                  : std::max(imc.last_marshak_in_step(), 0.0);
        const double deterministic_volume_in =
            (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion)
                ? state.fld_volume_source_in_step
                : ((cfg.radiation.mode == core::RadiationMode::SnTransport)
                       ? state.sn_volume_source_in_step
                       : 0.0);
        step_E_volume_in += deterministic_stage
                                ? replicated_tally_share *
                                      std::max(deterministic_volume_in, 0.0)
                                : std::max(imc.last_volume_source_step(), 0.0);
        step_E_numerical_loss += std::max(imc.last_numerical_loss_step(), 0.0);
        double source_E_floor = 0.0;
        int source_clamp_count = 0;
        mark_rad_subphase(rad_other_ms);
        // Radiation source-term closure reads state.zbar (single shared Zbar field).
        if (!deterministic_stage) {
          emit_radial_fourier_audit(
              diagnostics::RadialFourierStageId::NewtonSource,
              diagnostics::RadialFourierStagePhase::Before,
              t_stage + dt_stage);
          step_E_numerical_loss += inject_radiation_source_terms(
              state, cfg, dt_stage, &source_E_floor, &source_clamp_count,
              &imc.last_sigma_R_max(), &eos_ctx);
          emit_radial_fourier_audit(
              diagnostics::RadialFourierStageId::NewtonSource,
              diagnostics::RadialFourierStagePhase::After,
              t_stage + dt_stage);
        }
        mark_rad_subphase(rad_inject_ms);
        step_E_floor += std::max(source_E_floor, 0.0);
        step_clamp_count += std::max(source_clamp_count, 0);
        mark_rad_subphase(rad_other_ms);
      };
      const auto copy_device_field = [](auto& dst, const auto& src,
                                        const char* label) {
        dst.reset(src.size());
        if (src.empty()) {
          return;
        }
        const cudaError_t err = cudaMemcpy(
            dst.data(), src.data(), src.size() * sizeof(double),
            cudaMemcpyDeviceToDevice);
        TENRYU_ASSERT(err == cudaSuccess,
                      std::string(label) + " cudaMemcpy D2D failed");
      };
      const auto has_compressed_floor_hit = [&]() {
        TENRYU_ASSERT(state.Te.size() == state.rho.size(),
                      "thermal subcycle floor check requires Te/rho size match");
        const auto host_Te = copy_field_to_host(state.Te);
        const auto host_rho = copy_field_to_host(state.rho);
        const double te_floor_threshold = cfg.numerics.floors.Te + 0.5;
        for (std::size_t i = 0; i < host_Te.size(); ++i) {
          if (host_rho[i] > 5.0 && host_Te[i] <= te_floor_threshold) {
            return true;
          }
        }
        return false;
      };
      const auto predict_initial_thermal_substeps = [&]() {
        const std::size_t n_cells = state.rho.size();
        int n_sub = 1;
        if (state.step <= 0 || n_cells == 0) {
          return n_sub;
        }
        TENRYU_ASSERT(state.ee.size() == n_cells && state.Te.size() == n_cells,
                      "thermal subcycle prediction requires ee/Te/rho size match");
        const auto host_ee = copy_field_to_host(state.ee);
        const auto host_Te = copy_field_to_host(state.Te);
        const auto host_rho = copy_field_to_host(state.rho);
        TENRYU_ASSERT(host_ee.size() == n_cells,
                      "thermal subcycle prediction ee host size mismatch");

        double min_ratio = 1.0e30;
        const double te_floor = cfg.numerics.floors.Te;
        constexpr double dT_guard = 1.0;
        // Option C (§6o.4k): scan OWNED cells only — the full-range loop
        // read the stale-but-finite far regions, so the predicted substep
        // count (and with it dt_sub for the whole C stage) forked between
        // P=1 and P>1 (the gxii interface single-ulp seed). Rank-uniform
        // via Allreduce(MIN) of the ratio below.
        std::size_t scan_begin = 0;
        std::size_t scan_end = n_cells;
        if (part_info.n_ranks > 1) {
          const auto cw =
              state.owned_cell_window(static_cast<int>(n_cells));
          scan_begin = static_cast<std::size_t>(cw.begin);
          scan_end = static_cast<std::size_t>(cw.end);
        }
        for (std::size_t i = scan_begin; i < scan_end; ++i) {
          if (host_rho[i] < 5.0) {
            continue;
          }
          const double Te_margin = host_Te[i] - (te_floor + dT_guard);
          if (Te_margin < 5.0 && host_Te[i] > te_floor + dT_guard) {
            const double ratio = Te_margin / std::max(host_Te[i], 1.0);
            min_ratio = std::min(min_ratio, ratio);
          }
        }
        if (part_info.n_ranks > 1) {
          min_ratio = reducer.allreduce_min(min_ratio);
        }

        if (min_ratio < 1.0e20) {
          constexpr double C_th = 0.2;
          int predicted_n =
              static_cast<int>(std::ceil(1.0 / (C_th * min_ratio)));
          predicted_n = std::max(1, std::min(predicted_n, 16));
          while (n_sub < predicted_n) {
            n_sub *= 2;
          }
          n_sub = std::min(n_sub, 16);
        }
        return n_sub;
      };
      const auto run_single_stage_with_thermal_subcycling = [&]() {
        core::CellField1D backup_ee;
        core::CellField1D backup_ei;
        core::CellField1D backup_Te;
        core::CellField1D backup_Ti;
        core::CellField1D backup_Pe;
        core::CellField1D backup_Pi;
        copy_device_field(backup_ee, state.ee, "thermal-subcycle backup ee");
        copy_device_field(backup_ei, state.ei, "thermal-subcycle backup ei");
        copy_device_field(backup_Te, state.Te, "thermal-subcycle backup Te");
        copy_device_field(backup_Ti, state.Ti, "thermal-subcycle backup Ti");
        copy_device_field(backup_Pe, state.Pe, "thermal-subcycle backup Pe");
        copy_device_field(backup_Pi, state.Pi, "thermal-subcycle backup Pi");
        // 2026-07-26 review: the retry rollback must be
        // transactional for the deterministic FLD/S_N lanes. A failed
        // attempt has already advanced the radiation prognostic state
        // (rad_E; rad_E_old via the SN stage-end copy; sn_psi_prev = the
        // per-angle backward-Euler time source) and the step-ledger
        // accumulators (escaped/marshak/volume-source/numerical-loss and
        // the HOLO LO tallies). Restoring only the six thermodynamic
        // fields made the retried attempt start from a half-advanced
        // radiation field and double-counted every failed attempt's
        // boundary flows in the step budget. Empty fields (mode-dependent)
        // copy as no-ops. Bit-identical whenever no retry fires.
        core::GroupField1D backup_rad_E;
        core::GroupField1D backup_rad_E_old;
        core::GroupField1D backup_sn_psi_prev;
        copy_device_field(backup_rad_E, state.rad_E,
                          "thermal-subcycle backup rad_E");
        copy_device_field(backup_rad_E_old, state.rad_E_old,
                          "thermal-subcycle backup rad_E_old");
        copy_device_field(backup_sn_psi_prev, state.sn_psi_prev,
                          "thermal-subcycle backup sn_psi_prev");
        const double saved_step_E_rad_esc = step_E_rad_esc;
        const double saved_step_E_marshak_in = step_E_marshak_in;
        const double saved_step_E_volume_in = step_E_volume_in;
        const double saved_step_E_numerical_loss = step_E_numerical_loss;
        const double saved_step_holo_boundary_in = step_holo_boundary_in;
        const double saved_step_holo_boundary_out = step_holo_boundary_out;
        const double saved_step_holo_matter_delta = step_holo_matter_delta;
        const double saved_step_holo_source_balance_error =
            step_holo_source_balance_error;
        const auto restore_pre_radiation_state = [&]() {
          copy_device_field(state.ee, backup_ee, "thermal-subcycle restore ee");
          copy_device_field(state.ei, backup_ei, "thermal-subcycle restore ei");
          copy_device_field(state.Te, backup_Te, "thermal-subcycle restore Te");
          copy_device_field(state.Ti, backup_Ti, "thermal-subcycle restore Ti");
          copy_device_field(state.Pe, backup_Pe, "thermal-subcycle restore Pe");
          copy_device_field(state.Pi, backup_Pi, "thermal-subcycle restore Pi");
          copy_device_field(state.rad_E, backup_rad_E,
                            "thermal-subcycle restore rad_E");
          copy_device_field(state.rad_E_old, backup_rad_E_old,
                            "thermal-subcycle restore rad_E_old");
          copy_device_field(state.sn_psi_prev, backup_sn_psi_prev,
                            "thermal-subcycle restore sn_psi_prev");
          step_E_rad_esc = saved_step_E_rad_esc;
          step_E_marshak_in = saved_step_E_marshak_in;
          step_E_volume_in = saved_step_E_volume_in;
          step_E_numerical_loss = saved_step_E_numerical_loss;
          step_holo_boundary_in = saved_step_holo_boundary_in;
          step_holo_boundary_out = saved_step_holo_boundary_out;
          step_holo_matter_delta = saved_step_holo_matter_delta;
          step_holo_source_balance_error =
              saved_step_holo_source_balance_error;
        };

        int n_sub = predict_initial_thermal_substeps();
        if (std::getenv("TENRYU_H1D_DEBUG") != nullptr) {
          std::fprintf(stderr, "[nsub-debug] step=%lld n_sub=%d\n",
                       static_cast<long long>(state.step), n_sub);
        }
        constexpr int max_sub = 16;
        bool retry = true;
        bool first_attempt = true;
        while (retry && n_sub <= max_sub) {
          if (!first_attempt) {
            restore_pre_radiation_state();
          }
          if (n_sub > 1) {
            core::log_info("[thermal-subcycle] step=" +
                           std::to_string(state.step) + " n_sub=" +
                           std::to_string(n_sub));
          }

          first_attempt = false;
          retry = false;
          const double dt_sub = dt_op / static_cast<double>(n_sub);
          for (int m = 0; m < n_sub; ++m) {
            const double t_sub = t_op + static_cast<double>(m) * dt_sub;
            run_radiation_stage(dt_sub, t_sub);
            const PhaseEnergySnapshot e_before_qei = capture_phase_energy();
            emit_radial_fourier_audit(
                diagnostics::RadialFourierStageId::NewtonSource,
                diagnostics::RadialFourierStagePhase::Before,
                t_sub + dt_sub);
            apply_qei_coupling_substep(state, cfg, dt_sub, &eos_ctx);
            emit_radial_fourier_audit(
                diagnostics::RadialFourierStageId::NewtonSource,
                diagnostics::RadialFourierStagePhase::After,
                t_sub + dt_sub);
            log_phase_energy("qei_callback", e_before_qei);
            run_conduction_phase(dt_sub, t_sub);
            if (has_compressed_floor_hit() && n_sub < max_sub) {
              const int previous_n_sub = n_sub;
              n_sub *= 2;
              retry = true;
              core::log_info("[thermal-subcycle] floor hit at substep " +
                             std::to_string(m + 1) + "/" +
                             std::to_string(previous_n_sub) +
                             ", retrying with n_sub=" +
                             std::to_string(n_sub));
              break;
            }
          }
        }
      };

      update_zbar_for_step(state, cfg);
      mark_rad_subphase(rad_other_ms);
      if (cfg.radiation.imc.two_stage) {
        const double dt_half = 0.5 * dt_op;
        std::vector<double> rad_dep_total;
        std::vector<double> rad_emit_total;
        std::vector<double> holo_rad_dep_total;
        std::vector<double> holo_rad_emit_total;
        std::vector<double> delta_E_rad_total;
        const std::vector<double> delta_E_rad_prev_saved =
            copy_field_to_host(state.delta_E_rad_prev);
        auto accumulate_stage_tallies = [&]() {
          const auto stage_rad_dep = copy_field_to_host(state.rad_dep);
          const auto stage_rad_emit = copy_field_to_host(state.rad_emit);
          const auto stage_holo_rad_dep = copy_field_to_host(state.holo_rad_dep);
          const auto stage_holo_rad_emit = copy_field_to_host(state.holo_rad_emit);
          const auto stage_delta_E = copy_field_to_host(state.delta_E_rad_prev);
          if (rad_dep_total.empty()) {
            rad_dep_total.assign(stage_rad_dep.size(), 0.0);
          }
          if (rad_emit_total.empty()) {
            rad_emit_total.assign(stage_rad_emit.size(), 0.0);
          }
          if (delta_E_rad_total.empty()) {
            delta_E_rad_total.assign(stage_delta_E.size(), 0.0);
          }
          if (cfg.radiation.holo.enabled && holo_rad_dep_total.empty()) {
            holo_rad_dep_total.assign(stage_holo_rad_dep.size(), 0.0);
          }
          if (cfg.radiation.holo.enabled && holo_rad_emit_total.empty()) {
            holo_rad_emit_total.assign(stage_holo_rad_emit.size(), 0.0);
          }
          TENRYU_ASSERT(stage_rad_dep.size() == rad_dep_total.size(),
                        "two-stage radiation requires consistent rad_dep size across stages");
          TENRYU_ASSERT(stage_rad_emit.size() == rad_emit_total.size(),
                        "two-stage radiation requires consistent rad_emit size across stages");
          TENRYU_ASSERT(stage_delta_E.size() == delta_E_rad_total.size(),
                        "two-stage radiation requires consistent delta_E_rad_prev size across stages");
          if (cfg.radiation.holo.enabled) {
            TENRYU_ASSERT(stage_holo_rad_dep.size() == holo_rad_dep_total.size(),
                          "two-stage radiation requires consistent holo_rad_dep size across stages");
            TENRYU_ASSERT(stage_holo_rad_emit.size() == holo_rad_emit_total.size(),
                          "two-stage radiation requires consistent holo_rad_emit size across stages");
          }
          for (std::size_t i = 0; i < stage_rad_dep.size(); ++i) {
            rad_dep_total[i] += stage_rad_dep[i];
          }
          for (std::size_t i = 0; i < stage_rad_emit.size(); ++i) {
            rad_emit_total[i] += stage_rad_emit[i];
          }
          for (std::size_t i = 0; i < stage_delta_E.size(); ++i) {
            delta_E_rad_total[i] += stage_delta_E[i];
          }
          for (std::size_t i = 0; i < holo_rad_dep_total.size(); ++i) {
            holo_rad_dep_total[i] += stage_holo_rad_dep[i];
          }
          for (std::size_t i = 0; i < holo_rad_emit_total.size(); ++i) {
            holo_rad_emit_total[i] += stage_holo_rad_emit[i];
          }
        };

        run_radiation_stage(dt_half, t_op);
        accumulate_stage_tallies();
        mark_rad_subphase(rad_other_ms);

        // Mid-stage EOS re-closure: inject_radiation_source_terms() has already
        // projected ee -> Te/Pe; sync_ee_from_Te_table refreshes ee/Pe/Cv from Te.
        sync_ee_from_Te_table(state, cfg);
        refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
        if (delta_E_rad_prev_saved.empty()) {
          state.delta_E_rad_prev.reset(0);
        } else {
          if (state.delta_E_rad_prev.size() != delta_E_rad_prev_saved.size()) {
            state.delta_E_rad_prev.reset(delta_E_rad_prev_saved.size());
          }
          state.delta_E_rad_prev.copy_from_host(delta_E_rad_prev_saved.data());
        }
        update_zbar_for_step(state, cfg);
        mark_rad_subphase(rad_other_ms);

        run_radiation_stage(dt_half, t_op + dt_half);
        accumulate_stage_tallies();
        mark_rad_subphase(rad_other_ms);

        if (!rad_dep_total.empty()) {
          state.rad_dep.copy_from_host(rad_dep_total.data());
        }
        if (!rad_emit_total.empty()) {
          state.rad_emit.copy_from_host(rad_emit_total.data());
        }
        if (!holo_rad_dep_total.empty()) {
          state.holo_rad_dep.copy_from_host(holo_rad_dep_total.data());
        }
        if (!holo_rad_emit_total.empty()) {
          state.holo_rad_emit.copy_from_host(holo_rad_emit_total.data());
        }
        if (!delta_E_rad_total.empty()) {
          const std::size_t n_cells = state.rho.size();
          if (state.delta_E_rad_prev.size() != n_cells) {
            state.delta_E_rad_prev.reset(n_cells);
          }
          if (n_cells > 0) {
            state.delta_E_rad_prev.copy_from_host(delta_E_rad_total.data());
          }
        }
        mark_rad_subphase(rad_other_ms);
      } else {
        if (cfg.numerics.radiation_thermal_subcycle) {
          // TODO: snapshot/restore the IMC census pool so retries restart from
          // the same photon state instead of only restoring thermodynamics.
          run_single_stage_with_thermal_subcycling();
        } else {
          run_radiation_stage(dt_op, t_op);
        }
      }
      refresh_mie_gruneisen_thermo_from_energy(state, cfg, eos_ctx, reclose_ctx);
      mark_rad_subphase(rad_refresh_ms);
      if (cfg.numerics.hydro.enabled && is_2d &&
          cfg.numerics.hydro.boundary_2d.has_any_state_supply()) {
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::BoundaryFill,
            diagnostics::RadialFourierStagePhase::Before,
            t_op + dt_op);
        hydro_2d.apply_state_supply_boundary(state, cfg, &eos_ctx, true);
        emit_radial_fourier_audit(
            diagnostics::RadialFourierStageId::BoundaryFill,
            diagnostics::RadialFourierStagePhase::After,
            t_op + dt_op);
      }
      log_fleck_diagnostics_if_needed(state, cfg, imc);
      mark_rad_subphase(rad_fleck_log_ms);
      if (wj_arena.mode) {
        compute_overshoot_metrics_device_to_slots(
            state.Te.data(),
            static_cast<int>(state.Te.size()),
            d_overshoot_before,
            d_overshoot_before_flags,
            marshak_boundary_temperature(state, cfg, t_op + dt_op),
            wj_arena.d_flags + static_cast<std::size_t>(
                StepViolationFlag::RadiationOvershootCount),
            wj_arena.d_scalars + static_cast<std::size_t>(
                StepScalarSlot::RadiationOvershootMaxRatio));
        deferred_radiation_overshoot = true;
      } else {
        const OvershootMetrics overshoot =
            compute_overshoot_metrics(state, cfg, te_max_before, t_op + dt_op);
        imc.set_last_overshoot_metrics(overshoot.count, overshoot.max_ratio);
      }
      mark_rad_subphase(rad_overshoot_ms);
      mark_rad_subphase(rad_other_ms);
      finish_phase_safety(phase_safety_token,
                          "radiation",
                          te_max_before,
                          state.step + 1,
                          t_op + dt_op,
                          dt_op);
      mark_rad_subphase(rad_safety_ms);
      if (verbose_phase_timing) {
        const auto t_phase_end = Clock::now();
        rad_other_ms += ms(t_rad_subphase, t_phase_end);
        const double rad_total_ms = ms(t_phase_start, t_phase_end);
        step_rad_ms += rad_total_ms;
        std::ostringstream oss;
        oss << "[rad_subphase] step=" << state.step
            << std::fixed << std::setprecision(2)
            << " imc=" << rad_imc_ms
            << " exchange=" << rad_exchange_ms
            << " migrate=" << rad_migrate_ms
            << " dflags=" << rad_dflags_ms
            << " inject=" << rad_inject_ms
            << " refresh=" << rad_refresh_ms
            << " fleck_log=" << rad_fleck_log_ms
            << " overshoot=" << rad_overshoot_ms
            << " safety=" << rad_safety_ms
            << " other=" << rad_other_ms
            << " total=" << rad_total_ms << " ms";
        core::log_info(oss.str());
      }
      if (phase_energy_trace_enabled &&
          cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion) {
        const double rad_dep_sum = sum_field_for_phase_trace(state.rad_dep);
        const double rad_emit_sum = sum_field_for_phase_trace(state.rad_emit);
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(17)
            << "[phase_energy_trace] step=" << (state.step + 1)
            << " label=radiation_callback_extra"
            << " fld_escaped_step=" << state.fld_escaped_step
            << " fld_marshak_in_step=" << state.fld_marshak_in_step
            << " fld_state_supply_in_step="
            << state.fld_state_supply_in_step
            << " fld_state_supply_out_step="
            << state.fld_state_supply_out_step
            << " fld_state_supply_net_step="
            << state.fld_state_supply_net_step
            << " sum_rad_dep=" << rad_dep_sum
            << " sum_rad_emit=" << rad_emit_sum;
        core::log_info(oss.str());
      }
      wj_audit_log("radiation", wj_b_rad);
      log_phase_energy("radiation_callback", e_before_phase);
      hydro::entropy_ledger_end(
          state, cfg, hydro::EntropyLedgerStage::Radiation);
    };

    if (state.mesh.dim == 1) {
      if (state.zmom_active) {
        state.ensure_cell_material_props(cfg);
      }
      materials::zmoment_fill_fields(state, nullptr);
    }
    execute_split_operators(order_, dt, state.t, callbacks);

    // W-C (NUMERICS §6.8): the SN material Newton requests a global timestep
    // rejection when no positivity-preserving root exists above T_floor or
    // the adaptive bracket expansion fails. Reject and retry the full step at
    // half dt through the standard snapshot machinery; when retries are
    // disabled or exhausted, keep the historic warn-and-proceed behavior
    // (the kernel already clamped locally for this attempt).
    // Rank-uniform retry verdict (Option C): under the 2D distributed SN
    // solve the Newton flag is a per-rank OR over OWNED cells; a rank
    // -divergent global-dt retry would desynchronize every subsequent
    // collective. The Allreduce runs unconditionally on all ranks (the
    // flag read is cheap and the branch below must be uniform).
    int sn_retry_flag_uniform = state.sn_material_retry_flag;
    state.sn_material_retry_flag = 0;
    if (part_info.n_ranks > 1) {
      sn_retry_flag_uniform = static_cast<int>(reducer.allreduce_max(
          static_cast<double>(sn_retry_flag_uniform)));
    }
    if (sn_retry_flag_uniform != 0) {
      const int sn_retry_flag = sn_retry_flag_uniform;
      const bool sn_can_retry =
          cfg.numerics.hydro.driver_full_step_retry_enabled &&
          retry_attempts <
              cfg.numerics.hydro.driver_full_step_retry_max_attempts &&
          0.5 * dt > cfg.numerics.dt.min_s;
      if (sn_can_retry) {
        ++retry_attempts;
        // The while-loop top resets g_dt_lineage_context and re-populates it
        // from these persistent step-locals, so the dt/2 forcing must go
        // through them (a direct set_dt_lineage_retry_context call here would
        // be wiped by reset_dt_lineage_ale_context on the next attempt).
        retry_trigger_result = HydroStepResult{};
        retry_trigger_result.reason = "sn_material_newton_retry";
        retry_dt_before = dt;
        retry_dt_after = 0.5 * dt;
        core::log_warning(
            "[driver-retry] SN material Newton requested global timestep "
            "rejection (flag=" + std::to_string(sn_retry_flag) +
            "); restoring snapshot and retrying at dt/2, attempt=" +
            std::to_string(retry_attempts) + "/" +
            std::to_string(
                cfg.numerics.hydro.driver_full_step_retry_max_attempts));
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        continue;
      }
      core::log_warning(
          "SN material Newton timestep rejection unavailable "
          "(driver_full_step_retry disabled, budget exhausted, or dt floor); "
          "proceeding with the locally clamped state (flag=" +
          std::to_string(sn_retry_flag) + ")");
    }

    if (conduction_retry_requested) {
      const bool can_retry =
          cfg.numerics.hydro.driver_full_step_retry_enabled &&
          retry_attempts <
              cfg.numerics.hydro.driver_full_step_retry_max_attempts &&
          0.5 * dt > cfg.numerics.dt.min_s;
      if (can_retry) {
        ++retry_attempts;
        retry_trigger_result = HydroStepResult{};
        retry_trigger_result.reason = "sts_total_stages_overflow";
        retry_dt_before = dt;
        retry_dt_after = 0.5 * dt;
        core::log_warning(
            "[driver-retry] conduction STS stage overflow (total=" +
            std::to_string(conduction_retry_result.retry_total_stages) +
            ", n_sub=" + std::to_string(conduction_retry_result.retry_n_sub) +
            ", dt_exp=" + format_sci(conduction_retry_result.retry_dt_exp) +
            "); restoring snapshot and retrying at dt/2, attempt=" +
            std::to_string(retry_attempts) + "/" +
            std::to_string(
                cfg.numerics.hydro.driver_full_step_retry_max_attempts));
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        continue;
      }
      std::ostringstream oss;
      oss << "conduction STS stage overflow exhausted the driver retry "
             "budget (or retry disabled/dt floor)"
          << " at step=" << state.step << " t=" << format_sci(state.t)
          << " total_stages=" << conduction_retry_result.retry_total_stages
          << " n_sub=" << conduction_retry_result.retry_n_sub
          << " dt_exp=" << format_sci(conduction_retry_result.retry_dt_exp)
          << " dt=" << format_sci(dt)
          << " sts_total_stages_max="
          << cfg.numerics.conduction.sts_total_stages_max;
      finalize_profile_before_fatal();
      TENRYU_ASSERT(false, oss.str());
    }

    // W-B: 1D hydro soft failures (non-positive volume from compression-driven
    // node crossing) take the plain restore + dt/2 path. The 2D repair-plan
    // machinery below (ALE rungs, axis regimes, geometric stagnation) is
    // 2D_RZ-specific and must not see 1D results.
    if (is_1d && last_hydro_result.retry_required &&
        cfg.numerics.hydro.driver_full_step_retry_enabled) {
      const bool hydro_1d_can_retry =
          retry_attempts <
              cfg.numerics.hydro.driver_full_step_retry_max_attempts &&
          0.5 * dt > cfg.numerics.dt.min_s;
      if (hydro_1d_can_retry) {
        ++retry_attempts;
        retry_trigger_result = last_hydro_result;
        retry_dt_before = dt;
        retry_dt_after = 0.5 * dt;
        core::log_warning(
            "[driver-retry] 1D hydro soft failure (" +
            std::string(last_hydro_result.reason != nullptr
                            ? last_hydro_result.reason
                            : "") +
            ", cell=" + std::to_string(last_hydro_result.first_failing_cell) +
            "); restoring snapshot and retrying at dt/2, attempt=" +
            std::to_string(retry_attempts) + "/" +
            std::to_string(
                cfg.numerics.hydro.driver_full_step_retry_max_attempts));
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        continue;
      }
      std::ostringstream oss;
      oss << "1D hydro soft failure exhausted the driver retry budget"
          << " at step=" << state.step << " t=" << format_sci(state.t)
          << " cell=" << last_hydro_result.first_failing_cell
          << " value=" << format_sci(last_hydro_result.failing_value)
          << " reason="
          << (last_hydro_result.reason != nullptr ? last_hydro_result.reason
                                                  : "");
      finalize_profile_before_fatal();
      TENRYU_ASSERT(false, oss.str());
    }

    if (last_hydro_result.retry_required &&
        last_hydro_result.geometry_soft_failure.valid) {
      const auto& failure = last_hydro_result.geometry_soft_failure;
      hydro::core_clearance_controller_note_event(
          hydro::CoreClearanceControllerEventKind::GEOMETRY_RETRY);
      const int retry_limit =
          cfg.numerics.hydro.driver_full_step_retry_max_attempts;
      const double dt_new = 0.5 * failure.sigma_safe * dt;
      char path_guard_log[384];
      std::snprintf(
          path_guard_log,
          sizeof(path_guard_log),
          "[path-guard] step=%d stage=%s cell=%d k=%d predicate=%s "
          "min=%.6e sigma_safe=%.4f dt_new=%.6e",
          state.step,
          path_guard_stage_name(last_hydro_result.stage),
          failure.cell,
          failure.corner_or_slot,
          failure.predicate != nullptr ? failure.predicate : "",
          failure.min_value,
          failure.sigma_safe,
          dt_new);
      core::log_warning(path_guard_log);
      const bool witness_valid =
          failure.cell >= 0 && failure.predicate != nullptr &&
          failure.predicate[0] != '\0' &&
          std::isfinite(failure.min_value) &&
          std::isfinite(failure.sigma_safe) &&
          failure.sigma_safe >= 0.0 && failure.sigma_safe <= 1.0;
      const bool dt_valid =
          std::isfinite(dt_new) && dt_new > cfg.numerics.dt.min_s &&
          dt_new < dt;
      const bool bcr_ladder_active =
          hydro::core_clearance_controller_bcr_rezone_active();
      const bool bcr_ladder_terminal_failure =
          bcr_ladder_active && bcr_ladder_seed_retry_issued;
      if (bcr_ladder_terminal_failure) {
        core::log_warning(
            "[bcr-ladder] step=" + std::to_string(state.step) +
            " rung=5 cell=" + std::to_string(failure.cell) +
            " action=fail_closed");
      }
      static const bool maxmin_enabled = [] {
        const char* raw = std::getenv("TENRYU_I1B_MAXMIN_REZONE");
        return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
      }();
      if (maxmin_step != state.step) {
        maxmin_step = state.step;
        maxmin_attempts_this_step = 0;
      }
      const bool maxmin_witness_ok =
          failure.cell >= 0 && failure.predicate != nullptr &&
          failure.predicate[0] != '\0' &&
          std::isfinite(failure.min_value);
      if (maxmin_enabled && maxmin_witness_ok &&
          maxmin_attempts_this_step < 4 &&
          (cfg.numerics.hydro.driver_full_step_retry_enabled ||
           hydro::i1b_path_guard_enabled()) &&
          retry_attempts <
              cfg.numerics.hydro.driver_full_step_retry_max_attempts) {
        ++maxmin_attempts_this_step;
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        axis_band_applied_this_attempt = false;
        hydro::ale::reset_remap_mass_closure_step_max();
        const auto repair = hydro::ale::apply_maxmin_untangle_repair(
            state, cfg, part_info, &reducer, &eos_ctx, dt, failure.cell);
        if (repair.applied) {
          retry_trigger_result = last_hydro_result;
          retry_trigger_result.reason = "maxmin_rezone_retry";
          retry_dt_before = dt;
          retry_dt_after = dt;
          core::log_warning(
              "[driver-retry] step=" + std::to_string(state.step) +
              " attempt=" + std::to_string(retry_attempts) + "/" +
              std::to_string(
                  cfg.numerics.hydro.driver_full_step_retry_max_attempts) +
              " reason=maxmin_rezone attempt_in_step=" +
              std::to_string(maxmin_attempts_this_step) + " cell=" +
              std::to_string(failure.cell));
          continue;
        }
        // The repair either left state untouched or restored its transaction;
        // fall through to ring release and the existing dt ladder.
      }
      const bool ring_release_available =
          hydro::axis_core_ring_release_enabled(cfg) &&
          cfg.numerics.ale.euler_window.enabled &&
          cfg.numerics.ale.euler_window.role == "axis_survival_core";
      if (ring_release_available && witness_valid &&
          (cfg.numerics.hydro.driver_full_step_retry_enabled ||
           hydro::i1b_path_guard_enabled()) &&
          retry_attempts <
              cfg.numerics.hydro.driver_full_step_retry_max_attempts) {
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        axis_band_applied_this_attempt = false;
        if (hydro::axis_core_disk_release_one_unit(state, cfg)) {
          ++retry_attempts;
          retry_trigger_result = last_hydro_result;
          retry_trigger_result.reason = "ring_release_retry";
          retry_dt_before = dt;
          // guard-suggested dt: the release needs subsequent steps to yield; the current step is cured by the guard dt law
          retry_dt_after = dt_new;
          core::log_warning(
              "[driver-retry] step=" + std::to_string(state.step) +
              " attempt=" + std::to_string(retry_attempts) + "/" +
              std::to_string(
                  cfg.numerics.hydro.driver_full_step_retry_max_attempts) +
              " reason=ring_release first_cell=" +
              std::to_string(last_hydro_result.first_failing_cell) +
              " dt: " + format_sci(dt) + " -> " + format_sci(dt_new));
          dt = dt_new;
          dt_ref = dt;
          continue;
        }
        // Exhausted: fall through to the existing fail-closed / ladder / dt
        // logic on the restored state (subsequent restores of the same
        // snapshot are idempotent — do not restructure the existing code).
      }
      static int row_merge_step = -1;
      static int row_merge_attempts_this_step = 0;
      if (row_merge_step != state.step) { row_merge_step = state.step; row_merge_attempts_this_step = 0; }
      const bool merge_witness_ok =
          failure.cell >= 0 && failure.predicate != nullptr &&
          failure.predicate[0] != '\0' && std::isfinite(failure.min_value);
      if (hydro::row_merge_enabled() && merge_witness_ok &&
          row_merge_attempts_this_step < 1 &&
          (retry_attempts >= cfg.numerics.hydro.driver_full_step_retry_max_attempts - 1 || !dt_valid) &&
          (cfg.numerics.hydro.driver_full_step_retry_enabled ||
           hydro::i1b_path_guard_enabled())) {
        ++row_merge_attempts_this_step;
        restore_driver_retry_snapshot(state, cfg, retry_snapshot_, false, true, nullptr);
        axis_band_applied_this_attempt = false;
        auto merge = hydro::apply_tier_row_merge(state, cfg, failure.cell);
        if (!merge.committed && merge.reject_reason != nullptr &&
            std::strcmp(merge.reject_reason, "split_infeasible") == 0) {
          int repair_seed =
              (merge.first_bad_cell >= 0) ? merge.first_bad_cell : failure.cell;
          for (int round = 0; round < 2 && !merge.committed; ++round) {
            hydro::ale::reset_remap_mass_closure_step_max();
            const auto repair = hydro::ale::apply_maxmin_untangle_repair(
                state, cfg, part_info, &reducer, &eos_ctx, dt, repair_seed);
            if (!repair.applied) {
              core::log_warning(
                  "[row-merge] composed position repair failed seed=" +
                  std::to_string(repair_seed));
              break;
            }
            merge = hydro::apply_tier_row_merge(state, cfg, failure.cell);
            if (merge.committed) {
              core::log_warning(
                  "[row-merge] composed repair+merge committed round=" +
                  std::to_string(round + 1));
              break;
            }
            core::log_warning(
                std::string("[row-merge] composed merge failed: ") +
                (merge.reject_reason != nullptr ? merge.reject_reason
                                                : "(none)") +
                " cell=" + std::to_string(merge.first_bad_cell) +
                " round=" + std::to_string(round + 1));
            const int next_seed =
                (merge.first_bad_cell >= 0) ? merge.first_bad_cell : -1;
            if (next_seed < 0 || next_seed == repair_seed ||
                merge.reject_reason == nullptr ||
                std::strcmp(merge.reject_reason, "split_infeasible") != 0) {
              break;  // no new information — stop composing
            }
            repair_seed = next_seed;
          }
        }
        if (merge.committed) {
          capture_driver_retry_snapshot(retry_snapshot_, state, cfg, nullptr);  // the merged topology is the new step baseline
          retry_attempts = 0;  // topology changed: a fresh retry budget for the new discrete system
          maxmin_step = -1;  // topology changed: re-arm the max-min repair for the new discrete system
          retry_trigger_result = last_hydro_result;
          retry_trigger_result.reason = "row_merge_retry";
          retry_dt_before = dt;
          retry_dt_after = dt;
          core::log_warning("[driver-retry] step=" + std::to_string(state.step) +
              " reason=row_merge pairs=" + std::to_string(merge.merged_pairs) +
              " cell=" + std::to_string(failure.cell));
          continue;
        }
        core::log_warning(std::string("[driver-retry] row_merge rejected: ") +
            (merge.reject_reason != nullptr ? merge.reject_reason : "(none)") +
            " cell=" + std::to_string(merge.first_bad_cell));
        // fall through to fail-closed on the restored state
      }
      if (!witness_valid || bcr_ladder_terminal_failure ||
          (!bcr_ladder_active &&
           (retry_attempts >= retry_limit || !dt_valid))) {
        std::ostringstream oss;
        oss << "path-guard full-step retry failed closed"
            << " at step=" << state.step
            << " t=" << format_sci(state.t)
            << " stage=" << static_cast<int>(last_hydro_result.stage)
            << " cell=" << failure.cell
            << " k=" << failure.corner_or_slot
            << " predicate="
            << (failure.predicate != nullptr ? failure.predicate : "")
            << " min_value=" << format_sci(failure.min_value)
            << " sigma_safe=" << format_sci(failure.sigma_safe)
            << " dt_failed=" << format_sci(dt)
            << " dt_new=" << format_sci(dt_new)
            << " retry_attempts=" << retry_attempts
            << " retry_limit=" << retry_limit
            << " witness_valid=" << (witness_valid ? 1 : 0)
            << " dt_valid=" << (dt_valid ? 1 : 0);
        finalize_profile_before_fatal();
        TENRYU_ASSERT(false, oss.str());
      }

      restore_driver_retry_snapshot(
          state, cfg, retry_snapshot_, false, true, nullptr);
      axis_band_applied_this_attempt = false;
      retry_dt_before = dt;
      ++retry_attempts;
      retry_trigger_result = last_hydro_result;
      pending_repair_plan.reset();
      stage24_pending_repair_state_sensitive = false;

      if (bcr_ladder_active) {
        if (!std::isfinite(bcr_ladder_original_dt)) {
          bcr_ladder_original_dt = dt;
        }
        if (!std::isfinite(bcr_ladder_retry_dt)) {
          bcr_ladder_retry_dt = dt;
        }
        ++bcr_ladder_path_guard_failures;

        bool bcr_action_available = false;
        if (bcr_ladder_path_guard_failures == 1) {
          bcr_action_available =
              hydro::core_clearance_controller_promote_cell(failure.cell);
          retry_dt_after = dt;
          core::log_warning(
              "[bcr-ladder] step=" + std::to_string(state.step) +
              " rung=1 cell=" + std::to_string(failure.cell) +
              " action=promote_constraint");
          core::log_warning(
              "[bcr-ladder] step=" + std::to_string(state.step) +
              " rung=2 cell=" + std::to_string(failure.cell) +
              " action=same_dt_rezone_retry");
        } else {
          const double bcr_dt_new =
              0.5 * failure.sigma_safe * bcr_ladder_retry_dt;
          const bool bcr_dt_valid =
              std::isfinite(bcr_dt_new) &&
              bcr_dt_new > cfg.numerics.dt.min_s &&
              bcr_dt_new < bcr_ladder_retry_dt;
          const bool dt_collapsed =
              bcr_dt_new < 1.0e-4 * bcr_ladder_original_dt;
          if (dt_collapsed) {
            bcr_action_available =
                hydro::core_clearance_controller_request_feasible_seed(
                    failure.cell);
            bcr_ladder_seed_retry_issued = true;
            retry_dt_after = bcr_ladder_retry_dt;
            core::log_warning(
                "[bcr-ladder] step=" + std::to_string(state.step) +
                " rung=4 cell=" + std::to_string(failure.cell) +
                " action=direct_feasible_seed_rezone_retry");
          } else if (bcr_dt_valid) {
            bcr_action_available =
                hydro::core_clearance_controller_promote_cell(failure.cell);
            bcr_ladder_retry_dt = bcr_dt_new;
            retry_dt_after = bcr_ladder_retry_dt;
            core::log_warning(
                "[bcr-ladder] step=" + std::to_string(state.step) +
                " rung=3 cell=" + std::to_string(failure.cell) +
                " action=reduce_dt_rezone_retry");
          }
        }

        if (!bcr_action_available) {
          core::log_warning(
              "[bcr-ladder] step=" + std::to_string(state.step) +
              " rung=5 cell=" + std::to_string(failure.cell) +
              " action=fail_closed");
          std::ostringstream oss;
          oss << "path-guard BCR ladder failed closed"
              << " at step=" << state.step
              << " t=" << format_sci(state.t)
              << " stage=" << static_cast<int>(last_hydro_result.stage)
              << " cell=" << failure.cell
              << " k=" << failure.corner_or_slot
              << " predicate="
              << (failure.predicate != nullptr ? failure.predicate : "")
              << " min_value=" << format_sci(failure.min_value)
              << " sigma_safe=" << format_sci(failure.sigma_safe)
              << " dt_failed=" << format_sci(dt)
              << " dt_new=" << format_sci(dt_new)
              << " dt_original=" << format_sci(bcr_ladder_original_dt)
              << " retry_attempts=" << retry_attempts
              << " witness_valid=" << (witness_valid ? 1 : 0)
              << " dt_valid=" << (dt_valid ? 1 : 0);
          finalize_profile_before_fatal();
          TENRYU_ASSERT(false, oss.str());
        }
        hydro::core_clearance_controller_pre_lagrange_rezone(state, cfg);
        if (hydro::core_clearance_controller_g31_take_trial_reject()) {
          restore_driver_retry_snapshot(state, cfg, retry_snapshot_, false,
                                        true, nullptr);
          core::log_warning(
              "[g31] step=" + std::to_string(state.step) +
              " ladder trial remap rejected; snapshot restored");
        }
      } else {
        retry_dt_after = dt_new;
      }

      core::log_warning(
          "[driver-retry] step=" + std::to_string(state.step) +
          " attempt=" + std::to_string(retry_attempts) + "/" +
          std::to_string(retry_limit) + " reason=path_guard cell=" +
          std::to_string(failure.cell) + " k=" +
          std::to_string(failure.corner_or_slot) + " predicate=" +
          failure.predicate + " dt: " + format_sci(retry_dt_before) +
          " -> " + format_sci(retry_dt_after));
      continue;
    }

    const bool path_guard_geometry_safety_retry =
        hydro::i1b_path_guard_enabled() &&
        std::string(last_hydro_result.reason != nullptr
                        ? last_hydro_result.reason
                        : "") == "mesh_geometry";
    if (last_hydro_result.retry_required &&
        !cfg.numerics.hydro.driver_full_step_retry_enabled &&
        !path_guard_geometry_safety_retry) {
      std::ostringstream oss;
      oss << "Hydro soft failure requires driver_full_step_retry_enabled=true"
          << " at step=" << state.step
          << " t=" << format_sci(state.t)
          << " first_failing_cell="
          << last_hydro_result.first_failing_cell
          << " first_failing_corner="
          << last_hydro_result.first_failing_corner
          << " min_metric=" << format_sci(last_hydro_result.min_metric)
          << " reason="
          << (last_hydro_result.reason != nullptr ? last_hydro_result.reason : "")
          << " stage=" << static_cast<int>(last_hydro_result.stage)
          << " geometry_kind="
          << tenryu::mesh::mesh_geometry_failure_kind_name(
                 last_hydro_result.geometry_failure_kind)
          << " failing_i=" << last_hydro_result.first_failing_i
          << " failing_j=" << last_hydro_result.first_failing_j
          << " failing_value="
          << format_sci(last_hydro_result.failing_value);
      finalize_profile_before_fatal();
      TENRYU_ASSERT(false, oss.str());
    }

    if (last_hydro_result.retry_required &&
        (cfg.numerics.hydro.driver_full_step_retry_enabled ||
         path_guard_geometry_safety_retry)) {
      same_dt_strategy_retry_pending_success = false;
      observe_geometric_retry_failure(geometric_retry_stagnation_state,
                                      last_hydro_result,
                                      dt);
      if (detect_geometric_retry_stagnation(geometric_retry_stagnation_state,
                                            cfg,
                                            last_hydro_result,
                                            dt)) {
        if (part_info.rank == 0) {
          core::log_warning(format_geometric_retry_stagnation_log(
              geometric_retry_stagnation_state,
              last_hydro_result,
              state.step,
              state.t,
              dt));
        }
        if (cfg.numerics.hydro.geometric_retry_stagnation
                .force_diagnostic_dump) {
          const bool dump_emitted = maybe_emit_mesh_degeneracy_forensics(
              state,
              cfg,
              last_hydro_result,
              retry_attempts,
              dt,
              part_info.rank,
              mesh_degeneracy_forensics_state,
              true,
              &geometric_retry_stagnation_state);
          if (dump_emitted) {
            ++geometric_retry_stagnation_state.dump_count;
          }
        }
        const std::string original_reason =
            last_hydro_result.reason != nullptr ? last_hydro_result.reason : "";
        const auto* stagnation_tracking =
            geometric_retry_stage_tracking(geometric_retry_stagnation_state,
                                           last_hydro_result.stage);
        const StagePerStageTracking default_tracking;
        const StagePerStageTracking& current_tracking =
            stagnation_tracking != nullptr ? *stagnation_tracking
                                           : default_tracking;
        std::ostringstream oss;
        oss << "terminal_geometric_stagnation at step=" << state.step
            << " t=" << format_sci(state.t)
            << " final_dt_attempted=" << format_sci(dt)
            << " first_failing_cell="
            << last_hydro_result.first_failing_cell
            << " first_failing_corner="
            << last_hydro_result.first_failing_corner
            << " original_reason=" << original_reason
            << " same_count="
            << current_tracking.consecutive_count
            << " total_count="
            << geometric_retry_total_count(geometric_retry_stagnation_state)
            << " sigma_band=["
            << format_sci(current_tracking.sigma_band_min)
            << ","
            << format_sci(current_tracking.sigma_band_max)
            << "] dt_first="
            << format_sci(geometric_retry_stagnation_state.dt_at_first_failure);
        finalize_profile_before_fatal();
        TENRYU_ASSERT(false, oss.str());
      }
      maybe_emit_mesh_degeneracy_forensics(state,
                                           cfg,
                                           last_hydro_result,
                                           retry_attempts,
                                           dt,
                                           part_info.rank,
                                           mesh_degeneracy_forensics_state);
      const bool path_dt_rejection =
          is_multiblock_path_admissibility_failure(last_hydro_result);
      const bool bprime_axis_band_candidate =
          is_mesh_quality_failure(last_hydro_result) &&
          classify_failure_axis_band(last_hydro_result, state, cfg);
      const bool bprime_active_shell_edge_cross_candidate =
          classify_failure_active_shell_edge_cross(last_hydro_result, state);
      const bool bprime_retry_candidate =
          cfg.numerics.ale.driver_retry_reference_barrier_enabled &&
          is_2d && cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled &&
          (bprime_axis_band_candidate ||
           bprime_active_shell_edge_cross_candidate);
      const int retry_limit =
          driver_full_step_retry_limit_for_failure(last_hydro_result, cfg);
      const double ring7_pole_cap_sigma_trigger =
          env_positive_double("TENRYU_I1B_POLE_CAP_SIGMA_TRIGGER_MAX", 1.0e-2);
      const bool ring7_pole_cap_retry_candidate =
          !ring7_pole_cap_oracle_requested_this_step &&
          tenryu::hydro::ring7_driven_pole_cap_retry_candidate_cell(
              state, cfg, last_hydro_result.first_failing_cell) &&
          last_hydro_result.trial_scale > 0.0 &&
          last_hydro_result.trial_scale < ring7_pole_cap_sigma_trigger &&
          std::string(last_hydro_result.reason != nullptr
                          ? last_hydro_result.reason
                          : "") == "mesh_quality_rz_volume";
      const bool ring7_seam_retry_candidate =
          !ring7_seam_retry_requested_this_step &&
          (is_ring7_seam_rz_volume_retry_candidate(last_hydro_result,
                                                   state,
                                                   cfg) ||
           is_ring7_seam_path_admissibility_retry_candidate(last_hydro_result,
                                                            state,
                                                            cfg) ||
           (is_multiblock_path_admissibility_failure(last_hydro_result) &&
            tenryu::hydro::interior_patch_retry_candidate_cell(
                state, cfg, last_hydro_result.first_failing_cell)));
      // I1-B-R candidate (a): rebound-era TMOP repair route. A POLAR_SHELL
      // path-admissibility failure past TENRYU_I1B_TMOP_START_T gets an
      // immediate barrier-quality patch repair (the forced axis-repair
      // transaction, where the TMOP failure-path arm evaluates) before the
      // normal ladder proceeds — the rebound ring7 fold completes within
      // one step's retries, faster than any cadence trigger (dyncore29/30).
      // Bounded per step; falls through to the existing ladder on
      // exhaustion, so the worst case is the baseline behavior.
      {
        static const bool tmop_on = [] {
          const char* raw = std::getenv("TENRYU_I1B_TMOP_PATCH");
          return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
        }();
        static const double tmop_t0 = [] {
          const char* raw = std::getenv("TENRYU_I1B_TMOP_START_T");
          return raw != nullptr && raw[0] != '\0' ? std::atof(raw) : 0.0;
        }();
        static int tmop_step = -1;
        static int tmop_repairs = 0;
        if (tmop_step != state.step) {
          tmop_step = state.step;
          tmop_repairs = 0;
        }
        if (tmop_on && tmop_t0 > 0.0 && state.t >= tmop_t0 &&
            tmop_repairs < 2 &&
            is_multiblock_path_admissibility_failure(last_hydro_result) &&
            tenryu::hydro::interior_patch_retry_candidate_cell(
                state, cfg, last_hydro_result.first_failing_cell)) {
          ++tmop_repairs;
          restore_driver_retry_snapshot(
              state, cfg, retry_snapshot_, false, true, nullptr);
          axis_band_applied_this_attempt = false;
          hydro::ale::reset_remap_mass_closure_step_max();
          (void)hydro::ale::apply_pole_axis_radial_order_repair(
              state, cfg, part_info, &reducer, &eos_ctx, dt);
          if (remap_closure_reject_tol(cfg) > 0.0 &&
              std::abs(hydro::ale::remap_mass_closure_step_max()) >
                  remap_closure_reject_tol(cfg)) {
            // The repair's own remap fabricated mass: discard it and fall
            // through to the normal ladder on the restored state.
            core::log_warning(
                "[driver-retry] tmop_rebound_repair REMAP CLOSURE "
                "VIOLATION rel=" +
                std::to_string(hydro::ale::remap_mass_closure_step_max()) +
                "; repair discarded");
            restore_driver_retry_snapshot(
                state, cfg, retry_snapshot_, false, true, nullptr);
          } else {
            retry_dt_before = dt;
            retry_dt_after = dt;
            retry_attempts += 1;
            retry_trigger_result = last_hydro_result;
            pending_repair_plan.reset();
            stage24_pending_repair_state_sensitive = false;
            core::log_warning(
                "[driver-retry] step=" + std::to_string(state.step) +
                " attempt=" + std::to_string(retry_attempts) + "/" +
                std::to_string(retry_limit) +
                " reason=tmop_rebound_repair first_cell=" +
                std::to_string(last_hydro_result.first_failing_cell));
            continue;
          }
        }
      }
      if (is_pole_axis_radial_order_inversion_failure(last_hydro_result)) {
        if (retry_attempts >= retry_limit) {
          std::ostringstream oss;
          oss << "pole_axis_radial_order_inversion repair exhausted after "
              << retry_attempts
              << " attempts at step=" << state.step
              << " t=" << format_sci(state.t)
              << " first_failing_cell="
              << last_hydro_result.first_failing_cell
              << " min_metric="
              << format_sci(last_hydro_result.min_metric);
          finalize_profile_before_fatal();
          TENRYU_ASSERT(false, oss.str());
        }
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        axis_band_applied_this_attempt = false;
        hydro::ale::reset_remap_mass_closure_step_max();
        const auto repair_result =
            hydro::ale::apply_pole_axis_radial_order_repair(
                state, cfg, part_info, &reducer, &eos_ctx, dt);
        if (remap_closure_reject_tol(cfg) > 0.0 &&
            std::abs(hydro::ale::remap_mass_closure_step_max()) >
                remap_closure_reject_tol(cfg)) {
          // Poisoned repair: undo it and burn one retry attempt; the
          // entry-side budget check bounds the loop.
          core::log_warning(
              "[driver-retry] pole_axis_radial_order repair REMAP CLOSURE "
              "VIOLATION rel=" +
              std::to_string(hydro::ale::remap_mass_closure_step_max()) +
              "; repair discarded, retrying");
          restore_driver_retry_snapshot(
              state, cfg, retry_snapshot_, false, true, nullptr);
          retry_attempts += 1;
          continue;
        }
        if (!repair_result.applied) {
          std::ostringstream oss;
          oss << "pole_axis_radial_order_inversion repair failed"
              << " at step=" << state.step
              << " t=" << format_sci(state.t)
              << " first_failing_cell="
              << last_hydro_result.first_failing_cell
              << " q_range=" << last_hydro_result.macro_q_begin << ":"
              << last_hydro_result.macro_q_end
              << " j_range=" << last_hydro_result.macro_j_begin << ":"
              << last_hydro_result.macro_j_end
              << " min_metric="
              << format_sci(last_hydro_result.min_metric)
              << " rezone_triggered="
              << (repair_result.rezone_triggered ? "true" : "false")
              << " rezone_converged="
              << (repair_result.rezone_converged ? "true" : "false");
          finalize_profile_before_fatal();
          TENRYU_ASSERT(false, oss.str());
        }
        retry_dt_before = dt;
        retry_dt_after = dt;
        retry_attempts += 1;
        retry_trigger_result = last_hydro_result;
        pending_repair_plan.reset();
        stage24_pending_repair_state_sensitive = false;
        core::log_warning(
            "[driver-retry] step=" + std::to_string(state.step) +
            " attempt=" + std::to_string(retry_attempts) + "/" +
            std::to_string(retry_limit) +
            " reason=pole_axis_radial_order_inversion first_cell=" +
            std::to_string(last_hydro_result.first_failing_cell) +
            " q_range=" + std::to_string(last_hydro_result.macro_q_begin) +
            ":" + std::to_string(last_hydro_result.macro_q_end) +
            " j_range=" + std::to_string(last_hydro_result.macro_j_begin) +
            ":" + std::to_string(last_hydro_result.macro_j_end) +
            " dt: " + format_sci(retry_dt_before) + " -> " +
            format_sci(retry_dt_after) +
            " radial_order_repaired=true");
        continue;
      }
      if (is_macro_repair_required_failure(last_hydro_result)) {
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        axis_band_applied_this_attempt = false;
        const bool extended =
            hydro::pole_angular_derefine::extend_for_adjacent_active_child(
                state,
                cfg,
                last_hydro_result.first_failing_cell,
                true);
        if (extended) {
          retry_dt_before = dt;
          retry_dt_after = dt;
          retry_attempts += 1;
          retry_trigger_result = last_hydro_result;
          pending_repair_plan.reset();
          stage24_pending_repair_state_sensitive = false;
          core::log_warning(
              "[driver-retry] step=" + std::to_string(state.step) +
              " attempt=" + std::to_string(retry_attempts) + "/" +
              std::to_string(retry_limit) +
              " reason=pole_angular_derefine_extend first_cell=" +
              std::to_string(last_hydro_result.first_failing_cell) +
              " path_source=" +
              std::to_string(last_hydro_result.path_source_kind) +
              " metric_kind=" +
              std::to_string(last_hydro_result.path_metric_kind) +
              " metric_block=" +
              std::to_string(last_hydro_result.path_block_id) +
              " metric_local_i=" +
              std::to_string(last_hydro_result.path_local_i) +
              " metric_local_j=" +
              std::to_string(last_hydro_result.path_local_j) +
              " dt: " + format_sci(retry_dt_before) + " -> " +
              format_sci(retry_dt_after) + " macro_extended=true");
          continue;
        }
        const bool repaired =
            hydro::pole_angular_derefine::repair_macro_boundary(
                state,
                cfg,
                last_hydro_result.macro_q_begin,
                last_hydro_result.macro_j_begin,
                last_hydro_result.macro_j_end);
        if (!repaired) {
          std::ostringstream oss;
          oss << "macro_repair_required but no polar de-ref repair remained"
              << " at step=" << state.step
              << " t=" << format_sci(state.t)
              << " q_range=" << last_hydro_result.macro_q_begin << ":"
              << last_hydro_result.macro_q_end
              << " j_range=" << last_hydro_result.macro_j_begin << ":"
              << last_hydro_result.macro_j_end
              << " span=" << last_hydro_result.macro_span
              << " level=" << last_hydro_result.macro_level
              << " first_failing_cell="
              << last_hydro_result.first_failing_cell
              << " min_metric="
              << format_sci(last_hydro_result.min_metric);
          finalize_profile_before_fatal();
          TENRYU_ASSERT(false, oss.str());
        }
        retry_dt_before = dt;
        retry_dt_after = dt;
        retry_attempts += 1;
        retry_trigger_result = last_hydro_result;
        pending_repair_plan.reset();
        stage24_pending_repair_state_sensitive = false;
        core::log_warning(
            "[driver-retry] step=" + std::to_string(state.step) +
            " attempt=" + std::to_string(retry_attempts) + "/" +
            std::to_string(retry_limit) +
            " reason=macro_repair_required first_cell=" +
            std::to_string(last_hydro_result.first_failing_cell) +
            " q_range=" + std::to_string(last_hydro_result.macro_q_begin) +
            ":" + std::to_string(last_hydro_result.macro_q_end) +
            " j_range=" + std::to_string(last_hydro_result.macro_j_begin) +
            ":" + std::to_string(last_hydro_result.macro_j_end) +
            " span=" + std::to_string(last_hydro_result.macro_span) +
            " dt: " + format_sci(retry_dt_before) + " -> " +
            format_sci(retry_dt_after) + " macro_repaired=true");
        continue;
      }
      if (ring7_pole_cap_retry_candidate && retry_attempts < retry_limit) {
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        axis_band_applied_this_attempt = false;
        tenryu::hydro::request_ring7_pole_cap_oracle(
            state, last_hydro_result.first_failing_cell);
        ring7_pole_cap_oracle_requested_this_step = true;
        retry_dt_before = dt;
        retry_dt_after = dt;
        retry_attempts += 1;
        retry_trigger_result = last_hydro_result;
        pending_repair_plan.reset();
        stage24_pending_repair_state_sensitive = false;
        core::log_warning(
            "[driver-retry] step=" + std::to_string(state.step) +
            " attempt=" + std::to_string(retry_attempts) + "/" +
            std::to_string(retry_limit) +
            " reason=ring7_pole_cap_oracle_request first_cell=" +
            std::to_string(last_hydro_result.first_failing_cell) +
            " dt: " + format_sci(retry_dt_before) + " -> " +
            format_sci(retry_dt_after) +
            " ring7_pole_cap_oracle_requested=true");
        continue;
      }
      if (ring7_seam_retry_candidate && retry_attempts < retry_limit) {
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        axis_band_applied_this_attempt = false;
        tenryu::hydro::request_ring7_seam_rezone(
            state, last_hydro_result.first_failing_cell);
        ring7_seam_retry_requested_this_step = true;
        mark_ring7_seam_rezone_request_step(state.step);
        retry_dt_before = dt;
        retry_dt_after = dt;
        retry_attempts += 1;
        retry_trigger_result = last_hydro_result;
        pending_repair_plan.reset();
        stage24_pending_repair_state_sensitive = false;
        core::log_warning(
            "[driver-retry] step=" + std::to_string(state.step) +
            " attempt=" + std::to_string(retry_attempts) + "/" +
            std::to_string(retry_limit) +
            " reason=ring7_seam_rezone_request first_cell=" +
            std::to_string(last_hydro_result.first_failing_cell) +
            " dt: " + format_sci(retry_dt_before) + " -> " +
            format_sci(retry_dt_after) +
            " ring7_seam_rezone_requested=true");
        continue;
      }
      if (!bprime_retry_candidate && retry_attempts >= retry_limit) {
        // The macro-core boundary itself is path-inadmissible and dt
        // reduction cannot cure a violation that is static in dt (the
        // boundary cells have converged below the absolute margin floor).
        // Absorb the first active unit so the boundary moves outward, then
        // retry with a fresh budget on the restored pre-step state. Bounded:
        // each rescue requires a NEW absorption depth (the repeat guard
        // refuses re-requests), so rescues <= absorbable units.
        const std::string exhausted_reason =
            last_hydro_result.reason != nullptr ? last_hydro_result.reason
                                                : "";
        if (exhausted_reason == "macro_core_path_admissibility" &&
            hydro::central_pseudo_core::request_macro_boundary_absorption(
                state, cfg)) {
          core::log_warning(
              "[driver-retry] macro-core boundary inadmissible after " +
              std::to_string(retry_attempts) +
              " dt retries; ring absorption armed; restoring snapshot and "
              "retrying step with a fresh retry budget");
          restore_driver_retry_snapshot(
              state, cfg, retry_snapshot_, false, true, nullptr);
          axis_band_applied_this_attempt = false;
          retry_attempts = 0;
          continue;
        }
        if (path_dt_rejection &&
            last_hydro_result.path_source_kind == kPathSourceActiveFineChild &&
            last_hydro_result.path_metric_kind == kPathMetricEdgeCross) {
          restore_driver_retry_snapshot(
              state, cfg, retry_snapshot_, false, true, nullptr);
          axis_band_applied_this_attempt = false;
          const bool extended =
              hydro::pole_angular_derefine::extend_for_adjacent_active_child(
                  state,
                  cfg,
                  last_hydro_result.first_failing_cell,
                  true);
          if (extended) {
            core::log_warning(
                "[driver-retry] active fine child edge_cross rejected after " +
                std::to_string(retry_attempts) +
                " dt retries; polar angular de-ref span extended; retrying "
                "same dt with a fresh retry budget");
            retry_attempts = 0;
            retry_trigger_result = last_hydro_result;
            pending_repair_plan.reset();
            stage24_pending_repair_state_sensitive = false;
            continue;
          }
        }
        if (path_dt_rejection) {
          ++state.multiblock_path_dt_rejection_exhausted_count;
        }
        std::ostringstream oss;
        oss << "driver_full_step_retry exhausted after " << retry_attempts
            << " attempts at step=" << state.step
            << " t=" << format_sci(state.t)
            << " final_dt_attempted=" << format_sci(dt)
            << " first_failing_cell="
            << last_hydro_result.first_failing_cell
            << " first_failing_corner="
            << last_hydro_result.first_failing_corner
            << " min_metric=" << format_sci(last_hydro_result.min_metric)
            << " reason="
            << (last_hydro_result.reason != nullptr ? last_hydro_result.reason : "")
            << " retry_limit=" << retry_limit;
        int contact_front_center_j = last_hydro_result.first_failing_j;
        if (contact_front_center_j < 0 &&
            last_hydro_result.first_failing_cell >= 0 &&
            state.mesh.topo.nz > 0) {
          contact_front_center_j =
              last_hydro_result.first_failing_cell % state.mesh.topo.nz;
        }
        const int contact_front_j_begin =
            std::max(0, contact_front_center_j - 8);
        const int contact_front_j_end =
            std::min(state.mesh.topo.nz - 1, contact_front_center_j + 8);
        std::vector<double> contact_front_r;
        std::vector<double> contact_front_z;
        state.x_r.copy_to_host(contact_front_r);
        state.x_z.copy_to_host(contact_front_z);
        const int contact_front_node_stride = state.mesh.topo.nz + 1;
        for (int j = contact_front_j_begin; j <= contact_front_j_end; ++j) {
          double phi = std::numeric_limits<double>::quiet_NaN();
          bool engaged = false;
          const core::EvacContactSlot* closest_slot = nullptr;
          int closest_pair = -1;
          int closest_row = std::numeric_limits<int>::max();
          for (const core::EvacContactSlot& slot :
               state.contact_graph.records) {
            if (slot.state != core::EvacContactState::kActive ||
                slot.cell < 0 || slot.cell % state.mesh.topo.nz != j) {
              continue;
            }
            for (int pair = 0; pair < 2; ++pair) {
              const int pair_row =
                  std::min(slot.node_a[pair] / contact_front_node_stride,
                           slot.node_b[pair] / contact_front_node_stride);
              if (pair_row < closest_row) {
                closest_slot = &slot;
                closest_pair = pair;
                closest_row = pair_row;
              }
            }
          }
          if (closest_slot != nullptr) {
            phi = closest_slot->gap_pair[closest_pair];
            engaged = closest_slot->pair_engaged[closest_pair] != 0U;
          } else {
            const int row_zero_node = state.mesh.topo.node_index(0, j);
            int counterpart = -1;
            for (const core::EvacContactSlot& slot :
                 state.contact_graph.records) {
              for (int pair = 0; pair < 2; ++pair) {
                if (slot.node_a[pair] == row_zero_node) {
                  counterpart = slot.node_b[pair];
                } else if (slot.node_b[pair] == row_zero_node) {
                  counterpart = slot.node_a[pair];
                } else {
                  continue;
                }
                engaged = slot.pair_engaged[pair] != 0U;
                break;
              }
              if (counterpart >= 0) {
                break;
              }
            }
            if (counterpart >= 0 &&
                counterpart < static_cast<int>(contact_front_r.size()) &&
                counterpart < static_cast<int>(contact_front_z.size())) {
              phi = std::hypot(
                  contact_front_r[static_cast<std::size_t>(counterpart)] -
                      contact_front_r[static_cast<std::size_t>(row_zero_node)],
                  contact_front_z[static_cast<std::size_t>(counterpart)] -
                      contact_front_z[static_cast<std::size_t>(row_zero_node)]);
            } else {
              engaged = false;
            }
          }
          core::log_warning("[contact-front] j=" + std::to_string(j) +
                            " phi=" + format_sci(phi) + " engaged=" +
                            std::to_string(engaged ? 1 : 0));
        }
        finalize_profile_before_fatal();
        TENRYU_ASSERT(false, oss.str());
      }

      if (bprime_retry_candidate) {
        if (bprime_attempts_this_step >=
            cfg.numerics.ale.driver_retry_reference_barrier_max_attempts) {
          ++state.driver_retry_reference_barrier_max_attempts_aborts;
          finalize_profile_before_fatal();
          TENRYU_ASSERT(false,
                        "b-prime reference-barrier retry max attempts exhausted");
        }
        const RetryReferenceBarrierSignature sig =
            make_retry_signature(last_hydro_result);
        if (bprime_have_last_signature &&
            same_retry_signature(
                bprime_last_signature,
                sig,
                cfg.numerics.ale.driver_retry_reference_barrier_cell_window)) {
          ++bprime_same_signature_count;
        } else {
          bprime_last_signature = sig;
          bprime_have_last_signature = true;
          bprime_same_signature_count = 1;
        }
        if (bprime_same_signature_count >=
            cfg.numerics.ale.driver_retry_reference_barrier_same_sig_max) {
          ++state.driver_retry_reference_barrier_same_sig_aborts;
          finalize_profile_before_fatal();
          TENRYU_ASSERT(false,
                        "b-prime reference-barrier retry same-signature abort");
        }

        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        axis_band_applied_this_attempt = false;

        tenryu::core::NodeField1D target_r(
            static_cast<std::size_t>(state.mesh.topo.n_nodes));
        tenryu::core::NodeField1D target_z(
            static_cast<std::size_t>(state.mesh.topo.n_nodes));
        tenryu::core::Config retry_cfg = cfg;
        retry_cfg.numerics.ale.reference_barrier_enabled = true;
        if (retry_cfg.numerics.ale.reference_target == "none") {
          retry_cfg.numerics.ale.reference_target = "eulerian_initial";
        }
        tenryu::hydro::build_reference_target_mesh(
            state, retry_cfg, target_r.data(), target_z.data());
        const auto ref_result = tenryu::hydro::apply_reference_barrier_rezone(
            state, retry_cfg, target_r.data(), target_z.data());
        ++state.driver_retry_reference_barrier_attempts;
        ++bprime_attempts_this_step;

        const int rezone_window = std::max(
            1,
            std::min(
                cfg.numerics.ale.driver_retry_reference_barrier_rezone_freq_window,
                static_cast<int>(
                    state.driver_retry_reference_barrier_rezone_recent_steps
                        .size())));
        auto& recent = state.driver_retry_reference_barrier_rezone_recent_steps;
        recent[static_cast<std::size_t>(
            state.driver_retry_reference_barrier_recent_idx %
            static_cast<int>(recent.size()))] = 1;
        state.driver_retry_reference_barrier_recent_idx =
            (state.driver_retry_reference_barrier_recent_idx + 1) %
            static_cast<int>(recent.size());
        int recent_count = 0;
        for (int k = 0; k < rezone_window; ++k) {
          recent_count += recent[static_cast<std::size_t>(k)] != 0 ? 1 : 0;
        }
        if (static_cast<double>(recent_count) / static_cast<double>(rezone_window) >
            cfg.numerics.ale
                .driver_retry_reference_barrier_rezone_freq_warn_fraction) {
          core::log_warning("[b-prime] reference-barrier rezone frequency warning");
        }

        if (observe_reference_barrier_lambda_result(state, cfg, ref_result)) {
          finalize_profile_before_fatal();
          TENRYU_ASSERT(false,
                        "b-prime reference-barrier retry lambda collapse abort");
        }
        const double quality = std::min(
            ref_result.final_quality.min_rz_volume_rel,
            std::min(ref_result.final_quality.min_corner_j_rel,
                     ref_result.final_quality.min_gauss_j_rel));
        if (std::isfinite(quality) && std::isfinite(bprime_last_quality) &&
            quality <
                cfg.numerics.ale
                        .driver_retry_reference_barrier_quality_progress_factor *
                    bprime_last_quality) {
          ++bprime_quality_stagnation_count;
        } else {
          bprime_quality_stagnation_count = 0;
        }
        if (std::isfinite(quality)) {
          bprime_last_quality = quality;
        }
        if (bprime_quality_stagnation_count >=
            cfg.numerics.ale
                .driver_retry_reference_barrier_quality_progress_count) {
          ++state.driver_retry_reference_barrier_quality_stagnation_aborts;
          finalize_profile_before_fatal();
          TENRYU_ASSERT(false,
                        "b-prime reference-barrier retry quality stagnation abort");
        }

        const double dt_before_bprime = dt;
        const double dt_after_bprime =
            compute_reference_barrier_retry_dt(last_hydro_result, cfg, dt);
        if (observe_reference_barrier_dt_collapse(
                state, cfg, dt_before_bprime, dt_after_bprime)) {
          finalize_profile_before_fatal();
          TENRYU_ASSERT(false, "b-prime reference-barrier retry dt collapse abort");
        }
        retry_dt_before = dt_before_bprime;
        retry_dt_after = dt_after_bprime;
        dt = dt_after_bprime;
        dt_ref = dt;
        retry_attempts += 1;
        retry_trigger_result = last_hydro_result;
        pending_repair_plan.reset();
        stage24_pending_repair_state_sensitive = false;
        core::log_warning(
            "[b-prime] step=" + std::to_string(state.step) +
            " attempt=" + std::to_string(bprime_attempts_this_step) +
            " driver_retry_attempt=" + std::to_string(retry_attempts) +
            " classifier=" +
            (bprime_active_shell_edge_cross_candidate
                 ? "active_shell_edge_cross"
                 : "axis_band") +
            " reason=" +
            (last_hydro_result.reason != nullptr ? last_hydro_result.reason : "") +
            " first_cell=" +
            std::to_string(last_hydro_result.first_failing_cell) +
            " path_source=" +
            std::to_string(last_hydro_result.path_source_kind) +
            " metric_kind=" +
            std::to_string(last_hydro_result.path_metric_kind) +
            " metric_block=" +
            std::to_string(last_hydro_result.path_block_id) +
            " metric_local_i=" +
            std::to_string(last_hydro_result.path_local_i) +
            " metric_local_j=" +
            std::to_string(last_hydro_result.path_local_j) +
            " lambda=" + format_sci(ref_result.lambda_accepted) +
            " dt: " + format_sci(retry_dt_before) + " -> " +
            format_sci(retry_dt_after));
        continue;
      }

      hydro::ale::RepairPlan repair_plan;
      bool wave1_decision_applied = false;
      bool measured_state_sensitive_repair = false;
      DispatchDecisionKind wave1_decision_kind =
          DispatchDecisionKind::FallThrough;
      if (!wave1_decision_applied &&
          cfg.numerics.hydro.dispatcher_state_sensitive_bypass_enabled) {
        DispatchInputs in;
        in.last_hydro_result = last_hydro_result;
        in.feature_enabled = true;
        in.axis_repair_tried_at_current_state =
            stage24_axis_repair_tried_at_current_state;
        in.repair_generation_this_step =
            stage24_repair_generation_this_step;
        in.repair_generation_per_step_cap =
            cfg.numerics.hydro.dispatcher_state_sensitive_repair_cap_per_step;
        in.dt = dt;
        in.dt_min_s = cfg.numerics.dt.min_s;
        const DispatchDecision decision = classify_retry(in);
        wave1_decision_kind = decision.kind;
        switch (decision.kind) {
          case DispatchDecisionKind::RepairSameDt:
            if (cfg.numerics.ale.axis_band_managed_remap_enabled) {
              pending_axis_band.active = true;
              pending_axis_band.kind =
                  core::TransactionClientKind::kAxisBandRemap;
              pending_axis_band.severity =
                  core::TriggerSeverity::kHardAdmissibility;
              pending_axis_band.state_epoch =
                  static_cast<std::uint64_t>(state.step);
              pending_axis_band.reason_code =
                  static_cast<std::int32_t>(decision.kind);
              pending_axis_band.requested_k =
                  cfg.numerics.ale.axis_band_managed_remap_width;
              pending_repair_plan.reset();
              repair_plan.ale_mode = hydro::ale::AleMode::ScheduledDefault;
              repair_plan.dt_retry = decision.dt_retry;
              repair_plan.retry_attempt = retry_attempts + 1;
            } else {
              repair_plan = choose_repair_plan(
                  last_hydro_result, state, cfg, dt, retry_attempts);
              repair_plan.ale_mode = decision.forced_mode;
              repair_plan.dt_retry = decision.dt_retry;
            }
            wave1_decision_applied = true;
            stage24_axis_repair_tried_at_current_state = true;
            stage24_repair_generation_this_step += 1;
            stage24_pending_repair_state_sensitive = true;
            break;
          case DispatchDecisionKind::HalveToDtStar:
            repair_plan = choose_repair_plan(
                last_hydro_result, state, cfg, dt, retry_attempts);
            repair_plan.dt_retry = decision.dt_retry;
            repair_plan.is_same_dt_strategy_retry = false;
            wave1_decision_applied = true;
            stage24_pending_repair_state_sensitive = false;
            break;
          case DispatchDecisionKind::StateRepairCapHitFallThrough:
            repair_plan = choose_repair_plan(
                last_hydro_result, state, cfg, dt, retry_attempts);
            wave1_decision_applied = true;
            stage24_pending_repair_state_sensitive = false;
            if (cfg.main.verbosity == "debug" &&
                !stage24_repair_generation_cap_hit_logged) {
              core::log_warning(
                  "[driver-retry] repair_generation_cap_hit step=" +
                  std::to_string(state.step) +
                  " cap=" +
                  std::to_string(
                      cfg.numerics.hydro
                          .dispatcher_state_sensitive_repair_cap_per_step));
              stage24_repair_generation_cap_hit_logged = true;
            }
            break;
          case DispatchDecisionKind::FallThrough:
          default:
            break;
        }
      }
      if (!wave1_decision_applied) {
        const std::string measured_corner_j_reason =
            last_hydro_result.reason != nullptr ? last_hydro_result.reason : "";
        const bool measured_corner_j_failure =
            measured_corner_j_reason == "corner_j" ||
            measured_corner_j_reason == "mesh_quality_corner_j" ||
            measured_corner_j_reason == "in_hydro_corner_j";
        const bool measured_corner_j_same_failure =
            measured_corner_j_failure &&
            last_hydro_result.first_failing_cell ==
                measured_corner_j_last_failing_cell &&
            last_hydro_result.first_failing_corner ==
                measured_corner_j_last_failing_corner;
        const int measured_corner_j_prior_same_failure_count =
            measured_corner_j_same_failure
                ? measured_corner_j_same_failure_count
                : 0;
        if (measured_corner_j_same_failure) {
          measured_corner_j_same_failure_count += 1;
        } else if (measured_corner_j_failure) {
          measured_corner_j_last_failing_cell =
              last_hydro_result.first_failing_cell;
          measured_corner_j_last_failing_corner =
              last_hydro_result.first_failing_corner;
          measured_corner_j_same_failure_count = 1;
        } else {
          measured_corner_j_last_failing_cell = -1;
          measured_corner_j_last_failing_corner = -1;
          measured_corner_j_same_failure_count = 0;
        }
        const bool measured_corner_j_state_sensitive =
            measured_corner_j_failure &&
            last_hydro_result.first_failing_i >= 0 &&
            last_hydro_result.first_failing_i <=
                cfg.numerics.hydro.axis_guard_band_cells &&
            measured_corner_j_prior_same_failure_count >= 2 &&
            !stage24_axis_repair_tried_at_current_state &&
            stage24_repair_generation_this_step == 0 &&
            stage24_repair_generation_this_step <
                cfg.numerics.hydro
                    .dispatcher_state_sensitive_repair_cap_per_step;
        if (measured_corner_j_state_sensitive) {
          if (cfg.numerics.ale.axis_band_managed_remap_enabled) {
            pending_axis_band.active = true;
            pending_axis_band.kind =
                core::TransactionClientKind::kAxisBandRemap;
            pending_axis_band.severity =
                core::TriggerSeverity::kHardAdmissibility;
            pending_axis_band.state_epoch =
                static_cast<std::uint64_t>(state.step);
            pending_axis_band.reason_code = static_cast<std::int32_t>(
                DispatchDecisionKind::RepairSameDt);
            pending_axis_band.requested_k =
                cfg.numerics.ale.axis_band_managed_remap_width;
            pending_repair_plan.reset();
            repair_plan.ale_mode = hydro::ale::AleMode::ScheduledDefault;
            repair_plan.dt_retry = dt;
            repair_plan.retry_attempt = retry_attempts + 1;
          } else {
            repair_plan = choose_repair_plan(
                last_hydro_result, state, cfg, dt, retry_attempts);
            repair_plan.ale_mode = hydro::ale::AleMode::AxisSpinePlusLocal;
            repair_plan.dt_retry = dt;
          }
          wave1_decision_applied = true;
          measured_state_sensitive_repair = true;
          stage24_axis_repair_tried_at_current_state = true;
          stage24_repair_generation_this_step += 1;
          stage24_pending_repair_state_sensitive = true;
        } else {
          repair_plan = choose_repair_plan(
              last_hydro_result, state, cfg, dt, retry_attempts);
          stage24_pending_repair_state_sensitive = false;
        }
      }
      if (stage24_telemetry_enabled()) {
        const bool axis_regime =
            last_hydro_result.regime == hydro::MeshFailureRegime::AxisFace ||
            last_hydro_result.regime == hydro::MeshFailureRegime::AxisBand;
        const int max_attempts =
            cfg.numerics.hydro.driver_full_step_retry_max_attempts;
        const int floor_stall_max =
            cfg.numerics.dt.floor_stall_max_consecutive_steps;
        const bool floor_stall_blocks =
            floor_stall_max > 0 && dt_floor_stall_count >= floor_stall_max;
        const int chosen_rung_reached =
            repair_plan.ale_mode != hydro::ale::AleMode::ScheduledDefault
                ? std::max(stage24_rung_reached,
                           static_cast<int>(repair_plan.ale_mode))
                : stage24_rung_reached;
        const bool av_already_engaged =
            chosen_rung_reached >=
            static_cast<int>(hydro::ale::AleMode::AxisVariationalProjection);
        stage24_av_eligible =
            axis_regime &&
            cfg.numerics.ale.axis_variational_projection_enabled &&
            (retry_attempts < max_attempts) && !floor_stall_blocks &&
            !av_already_engaged;
        if (stage24_av_eligible) {
          stage24_av_blocked_reason = "";
        } else if (!cfg.numerics.ale.axis_variational_projection_enabled) {
          stage24_av_blocked_reason = "config_disabled";
        } else if (!axis_regime) {
          stage24_av_blocked_reason = "regime_not_axis";
        } else if (floor_stall_blocks) {
          stage24_av_blocked_reason = "floor_stall_aborted_first";
        } else if (retry_attempts >= max_attempts) {
          stage24_av_blocked_reason = "max_retries_exhausted";
        } else if (av_already_engaged) {
          stage24_av_blocked_reason = "av_already_engaged";
        } else {
          stage24_av_blocked_reason = "";
        }
      }
      restore_driver_retry_snapshot(
          state, cfg, retry_snapshot_, false, true, nullptr);
	      axis_band_applied_this_attempt = false;
	      if (pending_axis_band.active) {
	        pending_axis_band.active = false;
	        if (pending_axis_band.state_epoch !=
	            static_cast<std::uint64_t>(state.step)) {
	          core::log_warning(
	              std::string("[axis_band_pending] stale pending dropped: epoch=") +
	              std::to_string(pending_axis_band.state_epoch) +
	              " step=" + std::to_string(state.step));
	        } else if (!hydro::ale::ale_identity_mode_enabled(cfg)) {
	          const auto axis_result = run_axis_band_controller(
	              false,
	              pending_axis_band.requested_k,
	              true,
	              "post_axis_band_managed_remap");
	          if (axis_result.remap_succeeded) {
	            repair_plan.ale_mode = hydro::ale::AleMode::ScheduledDefault;
	            repair_plan.dt_retry = dt;
	            repair_plan.retry_attempt = retry_attempts + 1;
	            retry_dt_after = dt;
	            pending_repair_plan.reset();
	            stage24_pending_repair_state_sensitive = false;
	          } else if (axis_result.skip_reason ==
	                     hydro::ale::AxisBandResult::SkipReason::AllKFailed) {
	            repair_plan = choose_repair_plan(
	                last_hydro_result, state, cfg, dt, retry_attempts);
	            pending_repair_plan = repair_plan;
	            retry_dt_after = repair_plan.dt_retry;
	            stage24_pending_repair_state_sensitive = false;
	          }
	        }
	      }
	      if (is_2d && cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled &&
	          !hydro::ale::ale_identity_mode_enabled(cfg) &&
	          !state.cell_is_void.empty()) {
        refresh_void_mask_after_2d_ale();
      }
      repair_plan.retry_attempt = retry_attempts + 1;
      if (repair_plan.dt_retry != dt) {
        stage24_axis_repair_tried_at_current_state = false;
      }
      retry_dt_before = dt;
      retry_dt_after = repair_plan.dt_retry;
      if (repair_plan.is_same_dt_strategy_retry) {
        same_dt_strategy_retries_attempted += 1;
        same_dt_strategy_retry_pending_success = true;
      }
      retry_attempts += 1;
      retry_trigger_result = last_hydro_result;
      if (repair_plan.ale_mode != hydro::ale::AleMode::ScheduledDefault) {
        pending_repair_plan = repair_plan;
        if (stage24_telemetry_enabled() ||
            cfg.numerics.hydro.dispatcher_state_sensitive_bypass_enabled) {
          ++stage24_repair_generation;
          stage24_rung_reached = std::max(
              stage24_rung_reached, static_cast<int>(repair_plan.ale_mode));
        }
      } else {
        pending_repair_plan.reset();
      }
      const std::string retry_reason =
          last_hydro_result.reason != nullptr ? last_hydro_result.reason : "";
      if (retry_reason == "corner_j" &&
          cfg.numerics.ale.evacuated_cell.closure_contact
              .flank_tangential_strip.enabled) {
        const auto flank_strip =
            hydro::flank_strip_update_and_evaluate(state, cfg, dt);
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3)
            << "[flank-strip] step=" << state.step
            << " armed=" << static_cast<int>(flank_strip.armed_now)
            << " fire=" << static_cast<int>(flank_strip.would_fire)
            << " worst_ratio=" << flank_strip.worst_ratio
            << " cell=" << flank_strip.worst_cell
            << " corner=" << flank_strip.worst_corner
            << " n_pred=" << flank_strip.n_sliver_pred;
        core::log_warning(oss.str());
        const int failing_cell = last_hydro_result.first_failing_cell;
        const int failing_i = failing_cell >= 0
                                  ? failing_cell / state.mesh.topo.nz
                                  : -1;
        const int failing_j = failing_cell >= 0
                                  ? failing_cell % state.mesh.topo.nz
                                  : -1;
        const auto& strip_cfg =
            cfg.numerics.ale.evacuated_cell.closure_contact
                .flank_tangential_strip;
        const int slot_cell =
            hydro::detail::active_flank_strip_slot_cell(state);
        int j_first = -1;
        int j_last = -1;
        if (slot_cell >= 0) {
          hydro::detail::flank_strip_seam_window(
              state,
              slot_cell,
              strip_cfg.band_halfwidth_j,
              state.mesh.topo.nz,
              j_first,
              j_last,
              failing_j);
        }
        const bool failing_cell_in_band =
            failing_i >= 1 && failing_i <= strip_cfg.band_layers + 1 &&
            slot_cell >= 0 && failing_j >= j_first && failing_j <= j_last;
        if (flank_strip.would_fire && failing_cell_in_band) {
          const int displaced_mode = pending_repair_plan.has_value()
                                         ? static_cast<int>(
                                               pending_repair_plan->ale_mode)
                                         : -1;
          pending_repair_plan = hydro::ale::RepairPlan{};
          pending_repair_plan->ale_mode =
              hydro::ale::AleMode::FlankTangentialStrip;
          pending_repair_plan->focus_cell =
              last_hydro_result.first_failing_cell;
          pending_repair_plan->retry_attempt = retry_attempts;
          pending_repair_plan->dt_retry = retry_dt_after;
          pending_repair_plan->is_same_dt_strategy_retry = false;
          core::log_warning(
              "[flank-strip] plan queued cell=" +
              std::to_string(last_hydro_result.first_failing_cell) +
              " attempt=" + std::to_string(retry_attempts) +
              " displaced_mode=" + std::to_string(displaced_mode));
        }
      }
      if ((retry_reason == "corner_j" ||
           retry_reason == "geometry_inversion" ||
           retry_reason == "mesh_geometry") &&
          last_hydro_result.first_failing_i == 0) {
        const auto collapse =
            hydro::axis_edge_collapse_update_and_evaluate(
                state,
                cfg,
                last_hydro_result.first_failing_cell,
                retry_dt_after,
                cfg.numerics.dt.min_s);
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(3)
            << "[axis-edge-collapse] step=" << state.step
            << " cell=" << last_hydro_result.first_failing_cell
            << " h0=" << collapse.h0
            << " hdot=" << collapse.hdot
            << " h_retire=" << collapse.h_retire
            << " h_min_pred=" << collapse.h_min_pred
            << " closing=" << collapse.closing_count << "/"
            << collapse.window
            << " would_fire=" << static_cast<int>(collapse.would_fire)
            << " waiver=" << static_cast<int>(collapse.terminal_waiver);
        if (!collapse.eligible) {
          oss << " inel=" << collapse.ineligible_reason;
        }
        core::log_warning(oss.str());
        if (collapse.would_fire || collapse.terminal_waiver) {
          const auto transaction = hydro::axis_edge_collapse_execute(
              state, cfg, last_hydro_result.first_failing_cell);
          std::ostringstream commit_log;
          commit_log << std::scientific << std::setprecision(3)
                     << "[axis-edge-collapse-commit] step=" << state.step
                     << " cell=" << last_hydro_result.first_failing_cell
                     << " committed="
                     << static_cast<int>(transaction.committed)
                     << " reason=" << transaction.reject_reason
                     << " q_node=" << transaction.q_node
                     << " tri_vol=" << transaction.tri_volume
                     << " min_offaxis_j="
                     << transaction.min_offaxis_corner_j;
          core::log_warning(commit_log.str());
          if (transaction.committed) {
            capture_driver_retry_snapshot(
                retry_snapshot_, state, cfg, nullptr);
            retry_dt_before = dt;
            retry_dt_after = dt;
            pending_repair_plan.reset();
            stage24_pending_repair_state_sensitive = false;
            continue;
          }
        }
      }
      core::log_warning(
          "[driver-retry] step=" + std::to_string(state.step) +
          " attempt=" + std::to_string(retry_attempts) + "/" +
          std::to_string(driver_full_step_retry_limit_for_failure(
              last_hydro_result, cfg)) +
          " reason=" + retry_reason +
          " first_cell=" +
          std::to_string(last_hydro_result.first_failing_cell) +
          " first_corner=" +
          std::to_string(last_hydro_result.first_failing_corner) +
          " dt: " + format_sci(retry_dt_before) + " -> " +
          format_sci(retry_dt_after) +
          (repair_plan.is_same_dt_strategy_retry
               ? " same_dt_strategy_retry=true"
               : "") +
          (measured_state_sensitive_repair
               ? " kind=measured_state_sensitive_repair"
               : (cfg.numerics.hydro.dispatcher_state_sensitive_bypass_enabled
                      ? " kind=" + std::string(dispatch_decision_kind_name(
                                        wave1_decision_kind))
                      : "")));
      continue;
    }

    if (corner_j_source_budget_active) {
      mesh_degeneracy_forensics_state.snapshot_corner_j_source_budget_phase(
          state, "post_lagrange", true);
    }
    maybe_emit_velocity_history_phase("post_lagrange", state.t + dt);
    diagnostics::EnergyTotals energy_post_hydro{};
    long energy_post_hydro_off = -1;
    double E_rad_post_hydro = 0.0;
    long E_rad_post_hydro_off = -1;
    if (phase_resolved_energy_enabled || cap_energy_audit_enabled ||
        i1b_spurious_sensor_enabled) {
      const auto t_energy_post_hydro_start =
          verbose_phase_timing ? Clock::now() : Clock::time_point{};
      if (wj_arena.mode) {
        energy_post_hydro_off = wj_defer_energy_capture();
      }
      if (energy_post_hydro_off < 0) {
        energy_post_hydro = compute_energy_totals_for_state(state, cfg);
      }
      if (verbose_phase_timing) {
        step_energy_ms += ms(t_energy_post_hydro_start, Clock::now());
      }
    }
    if (cap_energy_audit_enabled && cfg.radiation.enabled) {
      if (deterministic_radiation_mode && wj_arena.mode) {
        E_rad_post_hydro_off = wj_defer_rad_capture();
      }
      if (deterministic_radiation_mode && E_rad_post_hydro_off < 0) {
        E_rad_post_hydro = compute_fld_rad_energy_total_device(state, cfg);
      } else if (!deterministic_radiation_mode) {
        E_rad_post_hydro = imc.census_energy();
      }
    }
    hydro::ale::log_cell113_substage_trace(state,
                                           cfg,
                                           part_info,
                                           "post_hydro_lagrangian",
                                           dt,
                                           state.t + dt);

    shell_subcycle_committed_this_step =
        last_hydro_result.shell_subcycle_committed;
    if (shell_subcycle_committed_this_step) {
      if (cap_energy_audit_enabled) {
        cap_energy_audit_rezone_fired = true;
      }
      record_mesh_attr_zero(
          diagnostics::mesh_attribution::MeshDeformSource::ALERezone);
      record_mesh_attr_zero(
          diagnostics::mesh_attribution::MeshDeformSource::Remap);
      step_dt_lineage.limiter = "i1b_shell_subcycle";
    }

    const int force_rezone_every_n_steps =
        cfg.numerics.ale.force_rezone_every_n_steps;
    const bool periodic_force_rezone =
        force_rezone_every_n_steps > 0 &&
        (state.step % force_rezone_every_n_steps) == 0;

    if (is_1d && cfg.numerics.hydro.enabled && cfg.numerics.ale1d.enabled) {
      // 1D V3 ALE is a conservative remap between hydro steps. Unlike the
      // pure Lagrangian 1D hydro update, state.mass may change after this call.
      hydro::entropy_ledger_begin(
          state, cfg, hydro::EntropyLedgerStage::Ale);
      const auto ale1d_out = hydro::ale1d::apply_ale_1d(state, cfg, &eos_ctx);
      hydro::entropy_ledger_end(
          state, cfg, hydro::EntropyLedgerStage::Ale);
      static int ale1d_reject_logged = 0;
      if ((ale1d_out.cadence_triggered || ale1d_out.quality_triggered ||
           ale1d_out.floor_triggered) &&
          !ale1d_out.applied && ale1d_reject_logged < 5) {
        std::ostringstream message;
        message << std::scientific << std::setprecision(3)
                << "[ale1d] step=" << state.step << " triggered(cad="
                << (ale1d_out.cadence_triggered ? 1 : 0)
                << " qual=" << (ale1d_out.quality_triggered ? 1 : 0)
                << " floor=" << (ale1d_out.floor_triggered ? 1 : 0)
                << ") NOT applied: reason="
                << hydro::ale1d::to_string(ale1d_out.skip_reason)
                << " remap_rejected=" << (ale1d_out.remap_rejected ? 1 : 0)
                << " mass_err=" << ale1d_out.mass_conservation_rel_err
                << " energy_err=" << ale1d_out.energy_conservation_rel_err
                << " radiation_conservation_rel_err="
                << ale1d_out.radiation_conservation_rel_err
                << " kinetic_energy_drift_rel="
                << ale1d_out.kinetic_energy_drift_rel;
        core::log_info(message.str());
        ++ale1d_reject_logged;
      }
      state.ale_rezoned = ale1d_out.applied;
      if (state.ale_rezoned) {
        state.ale_rezone_invocations += 1;
        // Remap moves volFrac between cells; the shared per-cell effective
        // material properties must be rebuilt before the next closure.
        state.invalidate_cell_material_props();
      }
	    } else if (is_2d && cfg.numerics.hydro.enabled &&
	               cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled &&
	               cfg.numerics.ale.conservative_remap_enabled &&
	               !hydro::ale::ale_identity_mode_enabled(cfg) &&
	               !cfg.numerics.ale.band_ale.enabled &&
	               !tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh) &&
	               !cfg.numerics.hydro.hllc_z_flux_2d_rz &&
	               !periodic_force_rezone) {
      // Single-block conservative reference remap is applied immediately after
      // each 2D Lagrangian hydro update. Multiblock and band-ALE flow instead
      // fall through to the transaction-managed apply_ale() path.
	    } else if (is_2d && cfg.numerics.hydro.enabled &&
	        cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled &&
	        !hydro::ale::ale_identity_mode_enabled(cfg) &&
	        !cfg.numerics.hydro.hllc_z_flux_2d_rz &&
	        !shell_subcycle_committed_this_step) {
      const PhaseEnergySnapshot e_before_phase = capture_phase_energy();
      const auto op_energy_before =
          operator_energy_tracker.enabled() ? capture_operator_energy_cap()
                                            : OpEnergyCapture{};
      hydro::ale::AleVolClosureReference ale_vol_closure_reference{};
      if (production_audit_enabled &&
          cfg.numerics.diagnostics.production_audit.gcl.enabled) {
        ale_vol_closure_reference =
            hydro::ale::capture_ale_vol_closure_reference(state);
      }
      // ALE module performs rezone node and remap cell halo exchanges.
      emit_radial_fourier_audit(
          diagnostics::RadialFourierStageId::Winslow,
          diagnostics::RadialFourierStagePhase::Before,
          state.t + dt);
      hydro::entropy_ledger_begin(
          state, cfg, hydro::EntropyLedgerStage::Ale);
      hydro::shadow_homothety_record_post_lagrange(
          state, state.step + 1, state.t + dt);
      state.mesh.materialize_host_svec();
      const auto ale_out =
          hydro::ale::apply_ale(state,
                                cfg,
                                part_info,
                                &comm_buffers,
                                &reducer,
                                &eos_ctx,
                                dt,
                                periodic_force_rezone,
                                periodic_force_rezone ? "periodic_force_rezone"
                                                      : nullptr,
                                profile_observability);
      hydro::entropy_ledger_end(
          state, cfg, hydro::EntropyLedgerStage::Ale);
      if (ale_out.status ==
          hydro::ale::AleStatus::CenterPatchInadmissibleAbsorbRetry) {
        // A central pseudo-core ring-absorption request was armed inside the
        // ALE driver (post-Lagrangian center-patch mesh inadmissible on an
        // active block-0 cell). The request lives in state.central_pseudo_core,
        // which the retry snapshot intentionally does not restore, so the
        // retried attempt executes the absorption at step start on the
        // restored pre-step state.
        if (state.central_pseudo_core.min_member_ring_count >
            state.central_pseudo_core.member_ring_count) {
          // Progress-based retry budget: an absorb-retry with a pending
          // absorption EXECUTES it at the start of the retried attempt
          // (members grow monotonically, bounded by the structural
          // backstop), so it is a cure in progress, not a futile retry.
          // Observed: the emergency walk needs one retry per row, and a
          // failing cell at depth 26 against members 19 exhausted the
          // 8-attempt budget two rows into the walk.
          retry_attempts = 0;
        }
        TENRYU_ASSERT(
            cfg.numerics.hydro.driver_full_step_retry_enabled &&
                retry_attempts <
                    cfg.numerics.hydro.driver_full_step_retry_max_attempts,
            "ALE center-patch ring-absorption retry unavailable "
            "(full-step retry disabled or budget exhausted)");
        ++retry_attempts;
        core::log_warning(
            "[driver-retry] ALE center-patch inadmissible; ring absorption "
            "armed; restoring snapshot and retrying step attempt=" +
            std::to_string(retry_attempts));
        restore_driver_retry_snapshot(
            state, cfg, retry_snapshot_, false, true, nullptr);
        axis_band_applied_this_attempt = false;
        continue;
      }
      const auto& axis_core_window = cfg.numerics.ale.euler_window;
      const bool axis_core_window_active =
          axis_core_window.enabled &&
          axis_core_window.role == "axis_survival_core" &&
          state.t >= axis_core_window.t_on_s &&
          (axis_core_window.t_off_s < 0.0 ||
           state.t < axis_core_window.t_off_s);
      if (axis_core_window_active &&
          ale_out.status ==
              hydro::ale::AleStatus::MandatoryWindowTransactionRejected) {
        // The ALE candidate coordinates are private to apply_ale() and are
        // never installed on rejection. Capture the live post-Lagrange state
        // before the driver retry restore; AleStepResult retains the candidate
        // corner coordinates needed by the terminal diagnostic.
        hydro::corner_collapse_ledger_capture(
            state, state.step + 1, state.t + dt, dt,
            hydro::CornerCollapseLedgerTrial::RejectedPostrestore);
        const bool ring_release_available =
            hydro::axis_core_ring_release_enabled(cfg) &&
            cfg.numerics.ale.euler_window.enabled &&
            cfg.numerics.ale.euler_window.role == "axis_survival_core";
        if (ring_release_available &&
            retry_attempts <
                cfg.numerics.hydro.driver_full_step_retry_max_attempts &&
            (cfg.numerics.hydro.driver_full_step_retry_enabled ||
             hydro::i1b_path_guard_enabled())) {
          restore_driver_retry_snapshot(
              state, cfg, retry_snapshot_, false, true, nullptr);
          axis_band_applied_this_attempt = false;
          if (hydro::axis_core_disk_release_one_unit(state, cfg)) {
            ++retry_attempts;
            retry_trigger_result = HydroStepResult{};
            retry_trigger_result.reason = "ring_release_retry";
            retry_trigger_result.first_failing_cell =
                ale_out.first_failing_cell;
            retry_dt_before = dt;
            retry_dt_after = dt;  // SAME dt — release addresses capacity,
                                  // not stiffness
            core::log_warning(
                "[driver-retry] step=" + std::to_string(state.step) +
                " attempt=" + std::to_string(retry_attempts) + "/" +
                std::to_string(
                    cfg.numerics.hydro.driver_full_step_retry_max_attempts) +
                " reason=ring_release first_cell=" +
                std::to_string(ale_out.first_failing_cell));
            continue;
          }
          // Release exhausted: fall through to the existing dt/2 logic on the
          // already-restored state. Its repeated snapshot restore is harmless.
        }
        static int row_merge_step = -1;
        static int row_merge_attempts_this_step = 0;
        if (row_merge_step != state.step) { row_merge_step = state.step; row_merge_attempts_this_step = 0; }
        if (hydro::row_merge_enabled() &&
            row_merge_attempts_this_step < 1 &&
            (retry_attempts >= cfg.numerics.hydro.driver_full_step_retry_max_attempts - 1 ||
             !(0.5 * dt > cfg.numerics.dt.min_s)) &&
            (cfg.numerics.hydro.driver_full_step_retry_enabled ||
             hydro::i1b_path_guard_enabled())) {
          ++row_merge_attempts_this_step;
          restore_driver_retry_snapshot(state, cfg, retry_snapshot_, false, true, nullptr);
          axis_band_applied_this_attempt = false;
          const auto merge = hydro::apply_tier_row_merge(
              state, cfg, ale_out.first_failing_cell);
          if (merge.committed) {
            capture_driver_retry_snapshot(retry_snapshot_, state, cfg, nullptr);  // the merged topology is the new step baseline
            retry_attempts = 0;  // topology changed: a fresh retry budget for the new discrete system
            maxmin_step = -1;  // topology changed: re-arm the max-min repair for the new discrete system
            retry_trigger_result = HydroStepResult{};
            retry_trigger_result.reason = "row_merge_retry";
            retry_trigger_result.first_failing_cell =
                ale_out.first_failing_cell;
            retry_dt_before = dt;
            retry_dt_after = dt;
            core::log_warning("[driver-retry] step=" + std::to_string(state.step) +
                " reason=row_merge pairs=" + std::to_string(merge.merged_pairs) +
                " cell=" + std::to_string(ale_out.first_failing_cell));
            continue;
          }
          core::log_warning(std::string("[driver-retry] row_merge rejected: ") +
              (merge.reject_reason != nullptr ? merge.reject_reason : "(none)") +
              " cell=" + std::to_string(merge.first_bad_cell));
          // fall through to the existing dt/2 logic on the restored state
        }
        const bool can_retry =
            cfg.numerics.hydro.driver_full_step_retry_enabled &&
            retry_attempts <
                cfg.numerics.hydro.driver_full_step_retry_max_attempts &&
            0.5 * dt > cfg.numerics.dt.min_s;
        if (can_retry) {
          ++retry_attempts;
          retry_trigger_result = HydroStepResult{};
          retry_trigger_result.reason =
              "mandatory_euler_window_transaction_rejected";
          retry_trigger_result.first_failing_cell =
              ale_out.first_failing_cell;
          retry_trigger_result.min_metric =
              std::min(ale_out.min_V, ale_out.min_J);
          retry_trigger_result.failing_value =
              ale_out.max_target_displacement;
          retry_dt_before = dt;
          retry_dt_after = 0.5 * dt;
          core::log_warning(
              "[driver-retry] mandatory axis-core Euler-window transaction "
              "rejected; restoring snapshot and retrying at dt/2, attempt=" +
              std::to_string(retry_attempts) + "/" +
              std::to_string(
                  cfg.numerics.hydro.driver_full_step_retry_max_attempts) +
              " first_failing_cell=" +
              std::to_string(ale_out.first_failing_cell) +
              " kind=" +
              std::string(mesh::mesh_geometry_failure_kind_name(
                  ale_out.geometry_failure_kind)) +
              " corner=" +
              std::to_string(ale_out.first_failing_corner) +
              " stable_cell=" +
              std::to_string(ale_out.first_failing_stable_cell) +
              " min_V=" + format_sci(ale_out.min_V) +
              " min_J=" + format_sci(ale_out.min_J) +
              " max_target_displacement=" +
              format_sci(ale_out.max_target_displacement));
          restore_driver_retry_snapshot(
              state, cfg, retry_snapshot_, false, true, nullptr);
          axis_band_applied_this_attempt = false;
          continue;
        }
        std::ostringstream oss;
        oss << "mandatory axis-core Euler-window transaction rejection "
               "exhausted the driver retry path"
            << " at step=" << state.step
            << " t=" << format_sci(state.t)
            << " first_failing_cell=" << ale_out.first_failing_cell
            << " kind=" << mesh::mesh_geometry_failure_kind_name(
                              ale_out.geometry_failure_kind)
            << " corner=" << ale_out.first_failing_corner
            << " stable_cell=" << ale_out.first_failing_stable_cell
            << " predicate_scope=full"
            << " failing_cell_area="
            << format_sci(ale_out.failing_cell_area)
            << " failing_state="
            << (ale_out.fails_without_target ? "post_lagrange"
                                             : "candidate")
            << " fails_without_target="
            << (ale_out.fails_without_target ? 1 : 0)
            << " min_V=" << format_sci(ale_out.min_V)
            << " min_J=" << format_sci(ale_out.min_J)
            << " max_target_displacement="
            << format_sci(ale_out.max_target_displacement);
        std::vector<double> rejection_postlag_r;
        std::vector<double> rejection_postlag_z;
        state.x_r.copy_to_host(rejection_postlag_r);
        state.x_z.copy_to_host(rejection_postlag_z);
        if (ale_out.rejection_corner_count == 0) {
          std::fprintf(stderr,
                       "[reject-corner] unavailable cell=%d recorded_count=0\n",
                       ale_out.first_failing_cell);
        }
        for (int corner = 0; corner < ale_out.rejection_corner_count;
             ++corner) {
          const std::size_t corner_index = static_cast<std::size_t>(corner);
          const int node = ale_out.rejection_corner_nodes[corner_index];
          if (node < 0 ||
              static_cast<std::size_t>(node) >= rejection_postlag_r.size() ||
              static_cast<std::size_t>(node) >= rejection_postlag_z.size()) {
            std::fprintf(
                stderr,
                "[reject-corner] node=%d postlag=(unavailable) "
                "candidate=(%.9e,%.9e)\n",
                node, ale_out.rejection_corner_candidate_r[corner_index],
                ale_out.rejection_corner_candidate_z[corner_index]);
            continue;
          }
          const std::size_t node_index = static_cast<std::size_t>(node);
          const double dr =
              ale_out.rejection_corner_candidate_r[corner_index] -
              rejection_postlag_r[node_index];
          const double dz =
              ale_out.rejection_corner_candidate_z[corner_index] -
              rejection_postlag_z[node_index];
          std::fprintf(
              stderr,
              "[reject-corner] node=%d postlag=(%.9e,%.9e) "
              "candidate=(%.9e,%.9e) dr=(%.3e,%.3e)\n",
              node, rejection_postlag_r[node_index],
              rejection_postlag_z[node_index],
              ale_out.rejection_corner_candidate_r[corner_index],
              ale_out.rejection_corner_candidate_z[corner_index], dr, dz);
        }
        std::fflush(stderr);
        finalize_profile_before_fatal();
        TENRYU_ASSERT(false, oss.str());
      }
      // Remap total-mass closure gate (env TENRYU_I1B_REMAP_CLOSURE_REJECT_TOL,
      // 0 = off): a committed rezone+remap that fabricates mass (measured:
      // +1.119e-1 total in ONE center-patch Winslow fire at gate drive) is
      // rejected through the full-step retry snapshot, and the rezone
      // initiators stand down for a cooldown so the retry does not re-fire
      // the violating mechanism.
      {
        static const char* closure_cooldown_raw =
            std::getenv("TENRYU_I1B_REZONE_CLOSURE_COOLDOWN_STEPS");
        const int closure_cooldown_steps = [&] {
          if (closure_cooldown_raw != nullptr &&
              closure_cooldown_raw[0] != '\0') {
            const int v = std::atoi(closure_cooldown_raw);
            return v > 0 ? v : 50;
          }
          return cfg.numerics.ale.rezone_closure_cooldown_steps;
        }();
        if (remap_closure_reject_tol(cfg) > 0.0 &&
            std::abs(ale_out.remap_mass_closure_rel) >
                remap_closure_reject_tol(cfg)) {
          TENRYU_ASSERT(
              cfg.numerics.hydro.driver_full_step_retry_enabled &&
                  retry_attempts <
                      cfg.numerics.hydro.driver_full_step_retry_max_attempts,
              "remap closure violation rejection unavailable (full-step "
              "retry disabled or budget exhausted)");
          ++retry_attempts;
          hydro::ale::arm_rezone_closure_cooldown(state.step +
                                                  closure_cooldown_steps);
          core::log_warning(
              "[driver-retry] REMAP MASS CLOSURE VIOLATION rel=" +
              std::to_string(ale_out.remap_mass_closure_rel) +
              " > tol=" + std::to_string(remap_closure_reject_tol(cfg)) +
              "; restoring snapshot and retrying step attempt=" +
              std::to_string(retry_attempts));
          restore_driver_retry_snapshot(
              state, cfg, retry_snapshot_, false, true, nullptr);
          axis_band_applied_this_attempt = false;
          continue;
        }
      }
      energy_audit_identity_skip_this_step =
          energy_audit_identity_skip_this_step ||
          ale_out.energy_audit_identity_skip;
      energy_audit_ale_fallback_this_step =
          energy_audit_ale_fallback_this_step ||
          ale_out.energy_audit_ale_fallback;
      if (cap_energy_audit_enabled) {
        cap_energy_audit_D_K += ale_out.cap_energy_audit_D_K;
        cap_energy_audit_rezone_fired =
            cap_energy_audit_rezone_fired ||
            ale_out.rezone_triggered || ale_out.accepted_remap_count > 0;
      }
      if (i1b_spurious_sensor_enabled) {
        merge_i1b_sensor_summary(i1b_ale_sensor_step,
                                 ale_out.i1b_ale_ke_sensor,
                                 part_info.rank,
                                 i1b_spurious_sensor_top_k);
      }
      emit_radial_fourier_audit(
          diagnostics::RadialFourierStageId::Winslow,
          diagnostics::RadialFourierStagePhase::After,
          state.t + dt);
      if (ale_out.rezone_triggered) {
        record_mesh_attr_zero(
            diagnostics::mesh_attribution::MeshDeformSource::ALERezone);
      }
      if (ale_vol_closure_reference.valid &&
          (ale_out.rezone_triggered || ale_out.accepted_remap_count > 0)) {
        const hydro::ale::AleVolClosureResidual vol_closure_residual =
            hydro::ale::compute_ale_vol_closure_residual(
                state, ale_vol_closure_reference, &reducer);
        if (vol_closure_residual.valid) {
          step_vol_closure_residual =
              vol_closure_residual.max_total_drift_rel();
          vol_closure_residual_max_mass =
              std::max(vol_closure_residual_max_mass,
                       vol_closure_residual.total_mass_drift_rel);
          vol_closure_residual_max_internal_energy =
              std::max(vol_closure_residual_max_internal_energy,
                       vol_closure_residual.total_internal_energy_drift_rel);
          vol_closure_residual_max_r_momentum =
              std::max(vol_closure_residual_max_r_momentum,
                       vol_closure_residual.total_r_momentum_drift_rel);
          vol_closure_residual_max_z_momentum =
              std::max(vol_closure_residual_max_z_momentum,
                       vol_closure_residual.total_z_momentum_drift_rel);
        }
      }
      if (ale_out.accepted_remap_count > 0) {
        record_mesh_attr_zero(diagnostics::mesh_attribution::MeshDeformSource::Remap);
      }
      if (ale_out.rezone_triggered) {
        ale_closure_audit = ale_out.closure_audit;
      }
      if (corner_j_source_budget_active && ale_out.rezone_triggered) {
        mesh_degeneracy_forensics_state.snapshot_corner_j_source_budget_phase(
            state, "post_ale_rezone", true);
        mesh_degeneracy_forensics_state.snapshot_corner_j_source_budget_phase(
            state, "post_remap", ale_out.accepted_remap_count > 0);
      }
      if (ale_out.rezone_triggered) {
        maybe_emit_velocity_history_phase("post_ale_rezone", state.t + dt);
        if (ale_out.accepted_remap_count > 0) {
          maybe_emit_velocity_history_phase("post_remap", state.t + dt);
        }
      }
      state.ale_rezoned = ale_out.rezone_triggered;
      if (state.ale_rezoned) {
        state.ale_rezone_invocations += 1;
        refresh_void_mask_after_2d_ale();
      }
      step_E_floor += std::max(ale_out.E_floor_injected, 0.0);
      step_E_redistribution_unresolved += ale_out.E_redistribution_unresolved;
      record_operator_energy("ale",
                             op_energy_before,
                             std::max(ale_out.E_floor_injected, 0.0) +
                                 ale_out.E_redistribution_unresolved);
      log_ale_remap_floor_closure("post_hydro", ale_out);
      process_escape_valve_events_for_audit();
      log_phase_energy("ale_2d_block", e_before_phase);
    } else if (shell_subcycle_committed_this_step) {
      state.ale_rezoned = true;
    } else {
      state.ale_rezoned = false;
    }

    if (is_2d) {
      state.mesh.materialize_host_svec();
    }

    if (cfg.main.two_temperature && !warned_2t_collapse &&
        all_active_cells_collapsed_to_one_temperature(state, cfg)) {
      core::log_warning("2T model collapse detected: all cells have Te ≈ Ti at step " +
                        std::to_string(state.step + 1) +
                        ". Check EOS or initial conditions.");
      warned_2t_collapse = true;
    }

    const auto t_energy_after_start =
        verbose_phase_timing ? Clock::now() : Clock::time_point{};
    diagnostics::EnergyTotals energy_after{};
    const long energy_after_off =
        wj_arena.mode ? wj_defer_energy_capture() : -1;
    if (energy_after_off < 0) {
      energy_after = compute_energy_totals_for_state(state, cfg);
    }
    if (verbose_phase_timing) {
      step_energy_ms += ms(t_energy_after_start, Clock::now());
    }
    long E_rad_after_off = -1;
    double E_rad_after = 0.0;
    if (cfg.radiation.enabled) {
      if (deterministic_radiation_mode && wj_arena.mode) {
        E_rad_after_off = wj_defer_rad_capture();
      }
      if (deterministic_radiation_mode && E_rad_after_off < 0) {
        E_rad_after = compute_fld_rad_energy_total_device(state, cfg);
      } else if (!deterministic_radiation_mode) {
        E_rad_after = imc.census_energy();
      }
    }
    // --- S2a drain: ONE async packet D2H + ONE stream sync materializes
    // every deferred scalar/flag and block-sum audit of this step.
    if (wj_arena.mode) {
      const std::size_t packet_bytes =
          wj_arena.header_bytes + wj_arena.used * sizeof(double);
      const cudaError_t wj_packet_err = cudaMemcpyAsync(
          wj_arena.h_packet,
          wj_arena.d_packet,
          packet_bytes,
          cudaMemcpyDeviceToHost,
          nullptr);
      TENRYU_ASSERT(wj_packet_err == cudaSuccess,
                    "S2a scalar pack async D2H failed");
      const cudaError_t wj_sync_err = cudaStreamSynchronize(nullptr);
      TENRYU_ASSERT(wj_sync_err == cudaSuccess,
                    "S2a scalar pack stream sync failed");
      const auto resolve_energy_totals = [&](const long off) {
        return state.mesh.dim == 1
                   ? diagnostics::materialize_energy_totals_1d_from_packet(
                         wj_arena.h + off, wj_arena.e_blocks)
                   : diagnostics::materialize_energy_totals_2d(
                         wj_arena.h + off, wj_arena.e_blocks);
      };
      const auto resolve_op_snapshot = [&](const long off) {
        const diagnostics::EnergyTotals t = resolve_energy_totals(off);
        diagnostics::OperatorResidualSnapshot s{};
        s.E_thermal_total = t.E_int_e + t.E_int_i;
        s.E_kinetic_total = t.E_kin;
        s.E_kinetic_total_nodal = t.E_kin_nodal;
        s.timestamp = state.t;
        s.valid = std::isfinite(s.E_thermal_total) &&
                  std::isfinite(s.E_kinetic_total);
        reduce_operator_snapshot(s, reducer);
        return s;
      };
      if (energy_before_off >= 0) {
        energy_before = resolve_energy_totals(energy_before_off);
      }
      if (energy_after_off >= 0) {
        energy_after = resolve_energy_totals(energy_after_off);
      }
      if (energy_post_hydro_off >= 0) {
        energy_post_hydro = resolve_energy_totals(energy_post_hydro_off);
      }
      if (E_rad_before_off >= 0) {
        E_rad_before = materialize_fld_rad_energy(
            wj_arena.h + E_rad_before_off, wj_arena.r_blocks);
      }
      if (E_rad_post_hydro_off >= 0) {
        E_rad_post_hydro = materialize_fld_rad_energy(
            wj_arena.h + E_rad_post_hydro_off, wj_arena.r_blocks);
      }
      if (E_rad_after_off >= 0) {
        E_rad_after = materialize_fld_rad_energy(
            wj_arena.h + E_rad_after_off, wj_arena.r_blocks);
      }
      if (deferred_radiation_overshoot) {
        imc.set_last_overshoot_metrics(
            wj_arena.h_flags[static_cast<std::size_t>(
                StepViolationFlag::RadiationOvershootCount)],
            wj_arena.h_scalars[static_cast<std::size_t>(
                StepScalarSlot::RadiationOvershootMaxRatio)]);
      }
      std::vector<double> rad_mesh_prefix(
          deferred_rad_mesh_advection.size() + 1U,
          step_E_rad_mesh_advection);
      for (std::size_t i = 0; i < deferred_rad_mesh_advection.size(); ++i) {
        const auto& capture = deferred_rad_mesh_advection[i];
        const double before = materialize_fld_rad_energy(
            wj_arena.h + capture.before_off, wj_arena.r_blocks);
        const double after = materialize_fld_rad_energy(
            wj_arena.h + capture.after_off, wj_arena.r_blocks);
        rad_mesh_prefix[i + 1U] = rad_mesh_prefix[i] + (after - before);
      }
      step_E_rad_mesh_advection = rad_mesh_prefix.back();
      for (const auto& rec : deferred_op_records) {
        const diagnostics::OperatorResidualSnapshot rec_before =
            rec.before.off >= 0 ? resolve_op_snapshot(rec.before.off)
                                : rec.before.snap;
        const diagnostics::OperatorResidualSnapshot rec_after =
            rec.after_off >= 0 ? resolve_op_snapshot(rec.after_off)
                               : rec.after_eager;
        operator_energy_tracker.record(rec.name, rec_before, rec_after,
                                       rec.delta_E_ext, rec.step, rec.time);
      }
      deferred_op_records.clear();
      const auto resolve_wj_snapshot = [&](WjOpAuditSnapshot& s) {
        if (s.e_off >= 0) {
          const diagnostics::EnergyTotals t = resolve_energy_totals(s.e_off);
          const double thermal = t.E_int_e + t.E_int_i;
          s.E_matter = thermal + t.E_kin;
          s.E_matter_nodal = thermal + t.E_kin_nodal;
          s.e_off = -1;
        }
        if (s.r_off >= 0) {
          s.E_rad = materialize_fld_rad_energy(wj_arena.h + s.r_off,
                                               wj_arena.r_blocks);
          s.r_off = -1;
        }
        TENRYU_ASSERT(s.rad_mesh_capture_count < rad_mesh_prefix.size(),
                      "S2a W-J radiation mesh prefix out of range");
        s.rad_mesh_adv = rad_mesh_prefix[s.rad_mesh_capture_count];
      };
      for (auto& wj_entry : wj_deferred_logs) {
        resolve_wj_snapshot(wj_entry.b);
        resolve_wj_snapshot(wj_entry.a);
        wj_audit_emit(wj_entry.op.c_str(), wj_entry.b, wj_entry.a);
      }
      wj_deferred_logs.clear();
      const auto resolve_phase_snapshot = [&](PhaseEnergySnapshot& snapshot) {
        if (snapshot.matter_off >= 0) {
          snapshot.matter = resolve_energy_totals(snapshot.matter_off);
          snapshot.matter_off = -1;
        }
        if (snapshot.rad_off >= 0) {
          snapshot.rad = materialize_fld_rad_energy(
              wj_arena.h + snapshot.rad_off, wj_arena.r_blocks);
          snapshot.rad_off = -1;
        }
      };
      for (auto& phase_log : deferred_phase_energy_logs) {
        resolve_phase_snapshot(phase_log.before);
        resolve_phase_snapshot(phase_log.after);
        emit_phase_energy(phase_log.label.c_str(),
                          phase_log.before,
                          phase_log.after);
      }
      deferred_phase_energy_logs.clear();

      for (std::size_t record = 0;
           record < deferred_phase_safety_records.size();
           ++record) {
        const auto& deferred = deferred_phase_safety_records[record];
        TENRYU_ASSERT(deferred.completed,
                      "S2a incomplete phase safety record at step drain");
        const std::size_t before_flags = phase_flag_index(
            StepViolationFlag::PhaseSafetyBeforeAuditBegin, record);
        const std::size_t after_flags = phase_flag_index(
            StepViolationFlag::PhaseSafetyAfterAuditBegin, record);
        const std::size_t after_scalar = phase_scalar_index(
            StepScalarSlot::PhaseSafetyAfterMaxBegin, record);
        DeviceTemperatureAudit staged_audit{};
        staged_audit.te_has_non_finite =
            wj_arena.h_flags[after_flags + 0U] != 0;
        staged_audit.ti_has_non_finite =
            wj_arena.h_flags[after_flags + 1U] != 0;
        staged_audit.rho_has_non_finite =
            wj_arena.h_flags[after_flags + 2U] != 0;
        staged_audit.ee_has_non_finite =
            wj_arena.h_flags[after_flags + 3U] != 0;
        staged_audit.ei_has_non_finite =
            wj_arena.h_flags[after_flags + 4U] != 0;
        staged_audit.max_te = wj_arena.h_flags[after_flags + 5U] != 0
                                  ? wj_arena.h_scalars[after_scalar]
                                  : 0.0;
        double before_max = deferred.before_host;
        if (deferred.before_staged) {
          const std::size_t before_scalar = phase_scalar_index(
              StepScalarSlot::PhaseSafetyBeforeMaxBegin, record);
          before_max = wj_arena.h_flags[before_flags + 5U] != 0
                           ? wj_arena.h_scalars[before_scalar]
                           : 0.0;
        }
        const std::size_t violation =
            static_cast<std::size_t>(
                StepViolationFlag::PhaseSafetyViolationBegin) +
            record;
        const bool device_flag = wj_arena.h_flags[violation] != 0;
        const bool host_violation =
            phase_safety_violation(staged_audit, cfg, before_max);
        if (device_flag || host_violation) {
          const DeviceTemperatureAudit forensic =
              audit_temperatures_host(state);
          (void)forensic;
          if (host_violation) {
            enforce_phase_safety_from_audit(deferred.phase_name.c_str(),
                                            staged_audit,
                                            cfg,
                                            before_max,
                                            deferred.step_id,
                                            deferred.t_eval,
                                            deferred.dt_eval);
          }
        }
      }
      deferred_phase_safety_records.clear();
    }
    diagnostics::PhaseResolvedEnergyDiagnostics phase_energy{};
    if (phase_resolved_energy_enabled || i1b_spurious_sensor_enabled) {
      diagnostics::EnergyTotals phase_pre_hydro = energy_before;
      diagnostics::EnergyTotals phase_post_hydro = energy_post_hydro;
      diagnostics::EnergyTotals phase_post_ale = energy_after;
      reduce_energy_totals(phase_pre_hydro, reducer);
      reduce_energy_totals(phase_post_hydro, reducer);
      reduce_energy_totals(phase_post_ale, reducer);
      phase_energy = diagnostics::compute_phase_resolved_energy_diagnostic(
          phase_pre_hydro, phase_post_hydro, phase_post_ale);
    }
    if (i1b_spurious_sensor_enabled) {
      emit_i1b_spurious_sensor_log(state,
                                   phase_energy,
                                   i1b_hydro_sensor_step,
                                   i1b_ale_sensor_step,
                                   part_info,
                                   reducer,
                                   i1b_spurious_sensor_top_k);
    }
    if (cfg.numerics.debug.trace_mesh_motion &&
        state.step < cfg.numerics.debug.trace_max_steps &&
        compatible_gate_E0 > 0.0 && std::isfinite(compatible_gate_E0)) {
      double internal_before = 0.0;
      double kinetic_before = 0.0;
      double internal_after = 0.0;
      double kinetic_after = 0.0;
      if (phase_energy.valid) {
        internal_before = phase_energy.internal_pre_hydro;
        kinetic_before = phase_energy.kinetic_pre_hydro;
        internal_after = phase_energy.internal_post_ale;
        kinetic_after = phase_energy.kinetic_post_ale;
      } else {
        diagnostics::EnergyTotals pre = energy_before;
        diagnostics::EnergyTotals post = energy_after;
        reduce_energy_totals(pre, reducer);
        reduce_energy_totals(post, reducer);
        internal_before = pre.E_int_e + pre.E_int_i;
        kinetic_before = pre.E_kin;
        internal_after = post.E_int_e + post.E_int_i;
        kinetic_after = post.E_kin;
      }
      const double step_residual =
          (internal_after - internal_before) + (kinetic_after - kinetic_before);
      const double cumulative =
          (internal_after + kinetic_after) -
          (compatible_gate_initial.E_int_e + compatible_gate_initial.E_int_i +
           compatible_gate_initial.E_kin);
      if (part_info.rank == 0) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(17)
            << "[compatible-gate] step=" << (state.step + 1)
            << " compatible_energy_step_residual_rel="
            << (std::abs(step_residual) / compatible_gate_E0)
            << " compatible_energy_cumulative_residual_rel="
            << (std::abs(cumulative) / compatible_gate_E0)
            << " dE_internal_plus_dK_erg=" << step_residual
            << " cumulative_dE_internal_plus_dK_erg=" << cumulative
            << " E0_erg=" << compatible_gate_E0;
        core::log_info(oss.str());
      }
    }
    // Multi-rank note: compute_energy_totals_for_state is expected to sum owned cells
    // only. Global SUM reduction is applied in compute_step_energy_budget via
    // budget_input.reduction below.
    if (cfg.radiation.enabled && !deterministic_radiation_mode) {
      const double rad_esc_after_total = imc.escaped_energy_total();
      step_E_rad_esc = std::max(rad_esc_after_total - rad_esc_before_total, 0.0);
    }

    diagnostics::EnergyBudgetStepInput budget_input{};
    budget_input.before = energy_before;
    budget_input.after = energy_after;
    static const bool state_hash_enabled = [] {
      const char* s = std::getenv("TENRYU_STATE_HASH");
      return s != nullptr && s[0] == '1';
    }();
    if (state_hash_enabled) {
      const unsigned long long h_xr =
          state_hash_xor_fold(state.x_r.data(), state.x_r.size());
      const unsigned long long h_vr =
          state_hash_xor_fold(state.v_r.data(), state.v_r.size());
      const unsigned long long h_rho =
          state_hash_xor_fold(state.rho.data(), state.rho.size());
      const unsigned long long h_ee =
          state_hash_xor_fold(state.ee.data(), state.ee.size());
      const unsigned long long h_ei =
          state_hash_xor_fold(state.ei.data(), state.ei.size());
      const unsigned long long h_q =
          state_hash_xor_fold(state.Qvisc.data(), state.Qvisc.size());
      unsigned long long dt_bits = 0ULL;
      static_assert(sizeof(dt_bits) == sizeof(double), "xor-fold dt bits");
      std::memcpy(&dt_bits, &dt, sizeof(dt_bits));
      std::ostringstream hash_os;
      hash_os << "[state_hash] step=" << (state.step + 1) << std::hex
              << " xr=" << h_xr << " vr=" << h_vr << " rho=" << h_rho
              << " ee=" << h_ee << " ei=" << h_ei << " q=" << h_q
              << " dt=" << dt_bits;
      core::log_info(hash_os.str());
    }
    // Under 1D replicated deterministic radiation the rad-field totals and
    // solver residual are already GLOBAL on every rank; only rank 0 feeds
    // them into the Allreduce(SUM) inside compute_step_energy_budget.
    const double replicated_rad_budget_share =
        (part_info.n_ranks > 1 && is_1d && cfg.radiation.enabled &&
         deterministic_radiation_mode && part_info.rank != 0)
            ? 0.0
            : 1.0;
    budget_input.E_rad_before = replicated_rad_budget_share * E_rad_before;
    budget_input.E_rad_after = replicated_rad_budget_share * E_rad_after;
    budget_input.E_rad_mesh_advection =
        replicated_rad_budget_share * step_E_rad_mesh_advection;
    budget_input.E_laser_in = step_E_laser_in;
    budget_input.E_burn_in = step_E_burn_in;
    budget_input.E_Marshak_in = step_E_marshak_in;
    budget_input.E_volume_in = step_E_volume_in;
    budget_input.E_burn_in = step_E_burn_in;
    budget_input.E_laser_esc = step_E_laser_esc;
    budget_input.E_cbet_iaw = step_E_cbet_iaw;
    budget_input.E_rad_esc = step_E_rad_esc;
    budget_input.E_numerical_loss = step_E_numerical_loss;
    budget_input.E_pdV_bdry = step_E_pdV_bdry;
    budget_input.E_floor = step_E_floor;
    // NOTE: E_safety currently equals E_floor (no separate safety corrections tracked yet).
    // When non-floor safety corrections are added (e.g., from temperature clamps in conduction),
    // E_safety should be split into E_floor + E_safety_other.
    budget_input.E_safety = step_E_floor;
    budget_input.E_redistribution_unresolved = step_E_redistribution_unresolved;
    budget_input.E_solver =
        replicated_rad_budget_share * (state.E_solver - prev_E_solver);
    budget_input.reduction = &reducer;
    const diagnostics::EnergyBudget step_budget =
        diagnostics::compute_step_energy_budget(budget_input);
    if (wj_op_audit_active) {
      const double wj_solver_src = std::max(step_budget.E_solver, 0.0);
      const double wj_solver_snk = std::max(-step_budget.E_solver, 0.0);
      const double wj_src = step_budget.E_laser_in + step_budget.E_Marshak_in +
                            step_budget.E_volume_in + step_budget.E_burn_in +
                            wj_solver_src;
      const double wj_snk = step_budget.E_laser_esc +
                            step_budget.E_cbet_iaw + step_budget.E_rad_esc +
                            step_budget.E_numerical_loss +
                            step_budget.E_pdV_bdry + wj_solver_snk;
      const double wj_safety_non_floor =
          std::max(step_budget.E_safety - step_budget.E_floor, 0.0);
      const double wj_artificial = step_budget.E_floor + wj_safety_non_floor +
                                   step_budget.E_redistribution_unresolved;
      const double wj_gap =
          (step_budget.dE_total - wj_artificial) - (wj_src - wj_snk);
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(17)
          << "[wj_step_audit] step=" << (state.step + 1) << " t=" << state.t
          << " eps=" << step_budget.epsilon_budget << " gap_signed=" << wj_gap
          << " dE_total=" << step_budget.dE_total
          << " laser_in=" << step_budget.E_laser_in
          << " laser_esc=" << step_budget.E_laser_esc
          << " cbet_iaw=" << step_budget.E_cbet_iaw
          << " E_denom=" << step_budget.E_denom;
      core::log_info(oss.str());
    }
    if (cap_energy_audit_enabled) {
      diagnostics::EnergyTotals audit_before = energy_before;
      diagnostics::EnergyTotals audit_after_lag = energy_post_hydro;
      diagnostics::EnergyTotals audit_after_remap = energy_after;
      reduce_energy_totals(audit_before, reducer);
      reduce_energy_totals(audit_after_lag, reducer);
      reduce_energy_totals(audit_after_remap, reducer);

      std::array<double, 8> audit_reduced = {
          E_rad_before,
          E_rad_post_hydro,
          E_rad_after,
          cap_energy_audit_W_ext_step,
          cap_energy_audit_D_K,
          cap_energy_audit_W_pq,
          cap_energy_audit_W_uF,
          cap_energy_audit_W_apex_pinned,
      };
      reducer.allreduce_sum(audit_reduced.data(),
                            static_cast<int>(audit_reduced.size()));
      const double S_other_rank_local =
          step_E_marshak_in + step_E_volume_in + step_E_burn_in - step_E_rad_esc -
          step_E_numerical_loss + (state.E_solver - prev_E_solver);
      const double S_other_step =
          step_E_laser_in - step_E_laser_esc - step_E_cbet_iaw +
          reducer.allreduce_sum(S_other_rank_local);
      const int rezone_fired =
          reducer.allreduce_max(cap_energy_audit_rezone_fired ? 1.0 : 0.0) >
                  0.5
              ? 1
              : 0;
      const auto matter_total =
          [](const diagnostics::EnergyTotals& totals) {
            return totals.E_int_e + totals.E_int_i + totals.E_kin;
          };
      const double E_before_total = matter_total(audit_before) + audit_reduced[0];
      const double E_after_lag_total =
          matter_total(audit_after_lag) + audit_reduced[1];
      const double E_after_remap_total =
          matter_total(audit_after_remap) + audit_reduced[2];
      const double W_ext_step = audit_reduced[3];
      const double D_K = audit_reduced[4];
      const double W_pq = audit_reduced[5];
      const double W_uF = audit_reduced[6];
      const double W_apex_pinned = audit_reduced[7];
      const double dKE = audit_after_lag.E_kin - audit_before.E_kin;
      const double dUint =
          (audit_after_lag.E_int_e + audit_after_lag.E_int_i) -
          (audit_before.E_int_e + audit_before.E_int_i);
      const double R_id = W_uF - W_pq;
      const double R_lag =
          E_after_lag_total - E_before_total - W_ext_step - S_other_step;
      const double R_remap = E_after_remap_total - E_after_lag_total;
      if (part_info.rank == 0) {
        std::ostringstream oss;
        oss << "[cap-energy-audit] step=" << (state.step + 1)
            << " E_before=" << format_sci(E_before_total)
            << " E_after_lag=" << format_sci(E_after_lag_total)
            << " E_after_remap=" << format_sci(E_after_remap_total)
            << " W_ext_step=" << format_sci(W_ext_step)
            << " S_other_step=" << format_sci(S_other_step)
            << " R_lag=" << format_sci(R_lag)
            << " R_remap=" << format_sci(R_remap)
            << " W_pq=" << format_sci(W_pq)
            << " W_uF=" << format_sci(W_uF)
            << " dKE=" << format_sci(dKE)
            << " dUint=" << format_sci(dUint)
            << " W_apex_pinned=" << format_sci(W_apex_pinned)
            << " R_id=" << format_sci(R_id)
            << " D_K=" << format_sci(D_K)
            << " rezone_fired=" << rezone_fired;
        core::log_info(oss.str());
      }
    }
    if (r_momentum_source_audit_active) {
      const double r_momentum_source_audit_current = reducer.allreduce_sum(
          hydro::compute_node_r_momentum_2d(state, cfg));
      const double dP_r =
          r_momentum_source_audit_current -
          r_momentum_source_audit_previous;
      const double I_r =
          reducer.allreduce_sum(r_momentum_source_impulse_step);
      const double S_hoop = reducer.allreduce_sum(
          compute_r_momentum_hoop_source_for_state(state));
      if (part_info.rank == 0) {
        std::ostringstream oss;
        oss << std::scientific << std::setprecision(17)
            << "[r-momentum-source-audit] dP_r=" << dP_r
            << " I_r=" << I_r
            << " residual=" << (dP_r - I_r)
            << " S_hoop=" << S_hoop;
        core::log_info(oss.str());
      }
      r_momentum_source_audit_previous = r_momentum_source_audit_current;
    }
    diagnostics::ConservationResiduals conservation_residuals{};
    if (production_audit_enabled) {
      ConservationTotals conservation_current =
          compute_conservation_totals_for_state(state, cfg);
      reduce_conservation_totals(conservation_current, reducer);
      conservation_residuals.mass =
          conservation_residual(conservation_current.mass, conservation_initial.mass);
      conservation_residuals.r_momentum = conservation_residual_with_scale(
          conservation_current.r_momentum,
          conservation_initial.r_momentum,
          conservation_momentum_scale_floor);
      conservation_residuals.z_momentum = conservation_residual_with_scale(
          conservation_current.z_momentum,
          conservation_initial.z_momentum,
          conservation_momentum_scale_floor);
      conservation_residuals.vol_closure = step_vol_closure_residual;
      conservation_residuals.valid = true;
      conservation_residual_max_mass =
          std::max(conservation_residual_max_mass, conservation_residuals.mass);
      conservation_residual_max_r_momentum = std::max(
          conservation_residual_max_r_momentum, conservation_residuals.r_momentum);
      conservation_residual_max_z_momentum = std::max(
          conservation_residual_max_z_momentum, conservation_residuals.z_momentum);
      vol_closure_residual_max =
          std::max(vol_closure_residual_max,
                   conservation_residuals.vol_closure);
    }

    if (phase_energy_trace_enabled) {
      const double dE_matter =
          (energy_after.E_int_e + energy_after.E_int_i + energy_after.E_kin) -
          (energy_before.E_int_e + energy_before.E_int_i + energy_before.E_kin);
      const double dE_rad = E_rad_after - E_rad_before;
      const double dE_total = dE_matter + dE_rad;
      const double expected_inflow =
          step_E_laser_in + step_E_burn_in - step_E_laser_esc -
          step_E_cbet_iaw - step_E_rad_esc + step_E_marshak_in;
      const double gap = expected_inflow - dE_total;
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(17)
          << "[phase_energy_summary] step=" << (state.step + 1)
          << " laser_in=" << step_E_laser_in
          << " burn_in=" << step_E_burn_in
          << " laser_esc=" << step_E_laser_esc
          << " cbet_iaw=" << step_E_cbet_iaw
          << " rad_esc=" << step_E_rad_esc
          << " marshak_in=" << step_E_marshak_in
          << " dE_matter=" << dE_matter
          << " dE_rad=" << dE_rad
          << " dE_total=" << dE_total
          << " state_supply_dM=" << state.state_supply_dM_step
          << " state_supply_dE=" << state.state_supply_dE_step
          << " state_supply_dPz=" << state.state_supply_dPz_step
          << " fld_state_supply_in="
          << state.fld_state_supply_in_step
          << " fld_state_supply_out="
          << state.fld_state_supply_out_step
          << " fld_state_supply_net="
          << state.fld_state_supply_net_step
          << " expected_inflow=" << expected_inflow
          << " gap=" << gap;
      core::log_info(oss.str());
    }

    if (cfg.numerics.safety.energy_fatal &&
        step_budget.epsilon_budget > cfg.numerics.safety.energy_budget_tol) {
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "Energy budget violation: epsilon_budget=" << step_budget.epsilon_budget
          << " exceeds safety.energy_budget_tol=" << cfg.numerics.safety.energy_budget_tol
          << " at step=" << (state.step + 1)
          << ", t=" << (state.t + dt)
          << ", dt=" << dt;
      TENRYU_ASSERT(false, oss.str());
    }

    if (!warned_clamp_threshold &&
        step_clamp_count > cfg.numerics.safety.clamp_warn_threshold) {
      std::ostringstream oss;
      oss << "[SAFETY] step=" << (state.step + 1)
          << " t=" << (state.t + dt)
          << " dt=" << dt
          << ": Floor clamp count exceeded safety.clamp_warn_threshold: clamp_count="
          << step_clamp_count
          << ", threshold=" << cfg.numerics.safety.clamp_warn_threshold;
      core::log_warning(oss.str());
      warned_clamp_threshold = true;
    }

    if (step_clamp_count > cfg.numerics.safety.clamp_fatal_threshold) {
      std::ostringstream oss;
      oss << "[SAFETY] step=" << (state.step + 1)
          << " t=" << (state.t + dt)
          << " dt=" << dt
          << ": Floor clamp count exceeded safety.clamp_fatal_threshold: clamp_count="
          << step_clamp_count
          << ", threshold=" << cfg.numerics.safety.clamp_fatal_threshold;
      TENRYU_ASSERT(false, oss.str());
    }

    if (step_clamp_count > 100) {
      static int high_clamp_warn_count = 0;
      ++high_clamp_warn_count;
      if (high_clamp_warn_count == 1 || high_clamp_warn_count % 100 == 0) {
        core::log_warning("High clamp count: " + std::to_string(step_clamp_count) +
                          " clamps this step (warning #" +
                          std::to_string(high_clamp_warn_count) + ")");
      }
    }

    maybe_emit_velocity_history_phase("step_end", state.t + dt);

    const int next_step = state.step + 1;
    if (ring7_quotient_effective_enabled(cfg) &&
        !state.ring7_seam_rezone_requested &&
        last_hydro_result.admissible &&
        !last_hydro_result.retry_required &&
        std::isfinite(step_min_path_margin)) {
      const double path_margin_trigger = env_positive_double(
          "TENRYU_I1B_RING7_PATH_MARGIN_TRIGGER", 5.0e-2);
      const int trigger_cell = step_min_path_margin_cell;
      if (step_min_path_margin < path_margin_trigger &&
          next_step >= ring7_seam_proactive_next_allowed_step &&
          tenryu::hydro::ring7_seam_retry_candidate_cell(
              state, cfg, trigger_cell)) {
        tenryu::hydro::request_ring7_seam_rezone(state, trigger_cell);
        mark_ring7_seam_rezone_request_step(next_step);
        core::log_warning(
            "[ring7_seam] proactive path-margin rezone armed for step=" +
            std::to_string(next_step) + " trigger_cell=" +
            std::to_string(trigger_cell) + " min_path_margin=" +
            format_sci(step_min_path_margin) + " trigger=" +
            format_sci(path_margin_trigger) + " cooldown_steps=" +
            std::to_string(env_nonnegative_int(
                "TENRYU_I1B_RING7_REZONE_COOLDOWN_STEPS", 3)));
      }
    }

    emit_radial_fourier_audit(
        diagnostics::RadialFourierStageId::DtController,
        diagnostics::RadialFourierStagePhase::Before,
        state.t + dt);
    state.t += dt;
    state.step += 1;
    state.dt = dt;
    state.dt_growth_ref = dt_ref;
    if (reale_mode::rezone_step(state, cfg, dt, step_dt_lineage.dt_hydro)) {
      hydro_2d.reclose_after_energy_edit(state, cfg, &eos_ctx);
      dt_reanchor.pending = true;
    }
    static const int state_hash_every = [] {
      const char* s = std::getenv("TENRYU_STATE_HASH_EVERY");
      return s != nullptr ? std::atoi(s) : 0;
    }();
    if (state_hash_every > 0 && state.step % state_hash_every == 0 &&
        part_info.rank == 0) {
      const auto fnv1a = [](const void* data,
                            const std::size_t size) -> std::uint64_t {
        constexpr std::uint64_t offset_basis = 14695981039346656037ULL;
        constexpr std::uint64_t prime = 1099511628211ULL;
        const auto* bytes = static_cast<const unsigned char*>(data);
        std::uint64_t hash = offset_basis;
        for (std::size_t i = 0; i < size; ++i) {
          hash ^= static_cast<std::uint64_t>(bytes[i]);
          hash *= prime;
        }
        return hash;
      };

      const auto x_r = copy_field_to_host(state.x_r);
      const auto x_z = copy_field_to_host(state.x_z);
      const auto v_r = copy_field_to_host(state.v_r);
      const auto v_z = copy_field_to_host(state.v_z);
      const auto mass = copy_field_to_host(state.mass);
      const auto vol = copy_field_to_host(state.vol);
      const auto rho = copy_field_to_host(state.rho);
      const auto ee = copy_field_to_host(state.ee);
      const auto ei = copy_field_to_host(state.ei);
      const auto corner_mass = copy_field_to_host(state.corner_mass);

      std::ostringstream line;
      line << std::scientific << std::setprecision(17)
           << "[state-hash] step=" << state.step << " t=" << state.t
           << std::hex << std::setfill('0')
           << " x_r=" << std::setw(16)
           << fnv1a(x_r.data(), x_r.size() * sizeof(double))
           << " x_z=" << std::setw(16)
           << fnv1a(x_z.data(), x_z.size() * sizeof(double))
           << " v_r=" << std::setw(16)
           << fnv1a(v_r.data(), v_r.size() * sizeof(double))
           << " v_z=" << std::setw(16)
           << fnv1a(v_z.data(), v_z.size() * sizeof(double))
           << " mass=" << std::setw(16)
           << fnv1a(mass.data(), mass.size() * sizeof(double))
           << " vol=" << std::setw(16)
           << fnv1a(vol.data(), vol.size() * sizeof(double))
           << " rho=" << std::setw(16)
           << fnv1a(rho.data(), rho.size() * sizeof(double))
           << " ee=" << std::setw(16)
           << fnv1a(ee.data(), ee.size() * sizeof(double))
           << " ei=" << std::setw(16)
           << fnv1a(ei.data(), ei.size() * sizeof(double))
           << " cm=" << std::setw(16)
           << fnv1a(corner_mass.data(), corner_mass.size() * sizeof(double))
           << std::dec << " dt=" << dt;
      core::log_info(line.str());
    }
    const bool z_reflection_audit_due =
        z_reflection_audit_active && part_info.rank == 0 &&
        state.step % 200 == 0;
    if (z_reflection_audit_due) {
      auto mass = copy_field_to_host(state.mass);
      auto ee = copy_field_to_host(state.ee);
      auto ei = copy_field_to_host(state.ei);
      auto velocity_r = copy_field_to_host(state.v_r);
      auto velocity_z = copy_field_to_host(state.v_z);
      auto node_r = copy_field_to_host(state.x_r);
      auto node_z = copy_field_to_host(state.x_z);
      const auto corner_mass = copy_field_to_host(state.corner_mass);

      const auto& topology = *state.mesh.topo.multiblock;
      const int n_cells = static_cast<int>(mass.size());
      const int n_nodes = static_cast<int>(node_r.size());
      const int corner_stride = state.corner_stride;
      std::vector<double> node_mass(node_r.size(), 0.0);
      bool node_mass_valid =
          node_z.size() == node_r.size() &&
          velocity_r.size() == node_r.size() &&
          velocity_z.size() == node_r.size() &&
          topology.cell_node_csr_offsets.size() == mass.size() + 1U &&
          corner_stride > 0 &&
          corner_mass.size() >=
              mass.size() * static_cast<std::size_t>(corner_stride);
      for (int c = 0; c < n_cells && node_mass_valid; ++c) {
        const int offset =
            topology.cell_node_csr_offsets[static_cast<std::size_t>(c)];
        const int end =
            topology.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
        const int nverts =
            state.mesh.cell_nverts.size() == mass.size()
                ? mesh::mesh_topo_cell_active_nverts(
                      state.mesh.cell_nverts, c)
                : end - offset;
        if (offset < 0 || end < offset || nverts < 0 ||
            offset + nverts > end || nverts > corner_stride ||
            static_cast<std::size_t>(end) >
                topology.cell_node_csr_indices.size()) {
          node_mass_valid = false;
          break;
        }
        for (int k = 0; k < nverts; ++k) {
          const int node = topology.cell_node_csr_indices[
              static_cast<std::size_t>(offset + k)];
          if (node < 0 || node >= n_nodes) {
            node_mass_valid = false;
            break;
          }
          node_mass[static_cast<std::size_t>(node)] +=
              corner_mass[static_cast<std::size_t>(c) *
                              static_cast<std::size_t>(corner_stride) +
                          static_cast<std::size_t>(k)];
        }
      }

      if (z_reflection_audit_due) {
        std::vector<double> energy;
        if (ee.size() == mass.size() &&
            (ei.empty() || ei.size() == mass.size())) {
          energy.resize(mass.size(), 0.0);
          for (std::size_t c = 0; c < mass.size(); ++c) {
            energy[c] = mass[c] * (ee[c] + (ei.empty() ? 0.0 : ei[c]));
          }
        }

        std::vector<double> momentum_r;
        std::vector<double> momentum_z;
        if (node_mass_valid) {
          momentum_r.resize(node_r.size(), 0.0);
          momentum_z.resize(node_r.size(), 0.0);
          for (std::size_t n = 0; n < node_r.size(); ++n) {
            momentum_r[n] = node_mass[n] * velocity_r[n];
            momentum_z[n] = node_mass[n] * velocity_z[n];
          }
        }

        const mesh::ZReflectionAudit audit = mesh::audit_z_reflection(
            z_reflection_maps, mass, momentum_r, momentum_z, energy,
            node_r, node_z);
        std::ostringstream line;
        line << std::scientific << std::setprecision(3)
             << "[z-reflection] audit step=" << state.step
             << " t=" << state.t
             << " odd_mass=" << audit.odd_mass
             << " odd_mom_z=" << audit.odd_mom_z
             << " odd_energy=" << audit.odd_energy
             << " odd_node_z=" << audit.odd_node_z;
        core::log_info(line.str());
      }
    }
    tenryu::hydro::evacuated_cell_contact_probe(state, cfg, "post_ale");
    tenryu::hydro::evacuated_cell_controller_step(state, cfg);
    tenryu::hydro::evacuated_cell_contact_probe(state, cfg, "pre_monitor");
    tenryu::hydro::evacuated_cell_contact_step(state, cfg);
    tenryu::hydro::evacuated_cell_contact_probe(state, cfg, "post_monitor");
    tenryu::hydro::rebuild_cavity_boundary_graph(state, cfg);
    hydro::entropy_ledger_log(state, cfg);
    emit_polar_tier_origin_force_diag(state, cfg);
    if (cfg.numerics.ale.multiblock_path_admissibility_enabled &&
        is_2d && tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      state.multiblock_path_dt_min_accepted =
          std::min(state.multiblock_path_dt_min_accepted, dt);
    }
    if (mesh_quality_min_enabled) {
      profile_observability_.note_committed_mesh_quality_min(state, &reducer);
    }
    if (ale_state_monitor_enabled) {
      run_ale_state_monitor(state,
                            cfg,
                            step_dt_lineage.dt_hydro_acoustic,
                            profile_observability_);
    }
    emit_radial_fourier_audit_for_cycle(
        diagnostics::RadialFourierStageId::DtController,
        diagnostics::RadialFourierStagePhase::After,
        state.t,
        static_cast<std::uint64_t>(state.step));
    if (same_dt_strategy_retry_pending_success) {
      same_dt_strategy_retries_succeeded += 1;
      same_dt_strategy_retry_pending_success = false;
    }
    if (cfg.numerics.dt.floor_stall_max_consecutive_steps > 0) {
      const double t_remaining = cfg.main.t_end - state.t;
      if (is_dt_floor_stall_step(
              dt, cfg.numerics.dt.min_s, t_remaining, step_dt_lineage.limiter)) {
        ++dt_floor_stall_count;
      } else {
        dt_floor_stall_count = 0;
      }
      if (cfg.numerics.hydro.dispatcher_state_sensitive_bypass_enabled &&
          stage24_state_sensitive_repair_executed_this_step) {
        dt_floor_stall_count = 0;
      }
      if (dt_floor_stall_count >
          cfg.numerics.dt.floor_stall_max_consecutive_steps) {
        std::ostringstream oss;
        oss << "dt floor-stall detector tripped: consecutive_committed_steps="
            << dt_floor_stall_count
            << " max_allowed="
            << cfg.numerics.dt.floor_stall_max_consecutive_steps
            << " dt.min_s=" << format_sci(cfg.numerics.dt.min_s)
            << " current_dt=" << format_sci(dt)
            << " t_end_minus_t=" << format_sci(t_remaining)
            << " limiter=" << step_dt_lineage.limiter
            << ". Set Numerics.dt.floor_stall_max_consecutive_steps=0 to disable.";
        TENRYU_ASSERT(false, oss.str());
      }
    } else {
      dt_floor_stall_count = 0;
    }
    // NOTE: E_safety and E_floor_injected are currently redundant by construction
    // because budget_input.E_safety reuses step_E_floor above.
    state.E_safety += std::max(step_budget.E_safety, 0.0);
    state.E_numerical_loss += std::max(step_E_numerical_loss, 0.0);
    state.E_laser_deposited +=
        std::max(step_E_laser_in - step_E_laser_esc - step_E_cbet_iaw, 0.0);
    state.E_laser_escaped += std::max(step_E_laser_esc, 0.0);
    state.E_laser_incident += std::max(step_E_laser_in, 0.0);
    state.E_ra_deposited += std::max(step_E_ra_dep, 0.0);
    state.E_cbet_iaw += step_E_cbet_iaw;
    state.E_hot_e_deposited += std::max(state.hot_e_deposited_step, 0.0);
    state.E_hot_e_escaped += std::max(state.hot_e_escaped_step, 0.0);
    state.E_rad_escaped += std::max(step_E_rad_esc, 0.0);
    state.E_floor_injected += std::max(step_E_floor, 0.0);
    state.E_pdV_bdry += step_E_pdV_bdry;
    state.E_Marshak_in += std::max(step_E_marshak_in, 0.0);
    state.E_solver += step_budget.E_solver;

    if (energy_audit_enabled) {
      maybe_emit_i1b_energy_audit_step(state,
                                       cfg,
                                       reducer,
                                       energy_before,
                                       energy_after,
                                       E_rad_before,
                                       E_rad_after,
                                       step_E_laser_in,
                                       step_E_laser_esc,
                                       step_E_cbet_iaw,
                                       step_E_rad_esc,
                                       step_E_marshak_in,
                                       step_E_volume_in,
                                       energy_audit_r_before,
                                       energy_audit_z_before,
                                       state.t - dt,
                                       dt,
                                       energy_audit_every,
                                       energy_audit_identity_skip_this_step,
                                       energy_audit_ale_fallback_this_step,
                                       energy_audit_ledger,
                                       part_info.rank);
    }

    process_escape_valve_events_for_audit();
    emit_radial_fourier_audit_for_cycle(
        diagnostics::RadialFourierStageId::Positivity,
        diagnostics::RadialFourierStagePhase::Before,
        state.t,
        static_cast<std::uint64_t>(state.step));
    const auto positivity_result =
        positivity_tracker.scan(state,
                                state.step,
                                state.t,
                                std::max(cfg.radiation.groups, 1),
                                &reducer);
    const auto positivity_decision =
        positivity_tracker.record_scan(positivity_result);
    write_positivity_history(positivity_result);
    handle_positivity_audit_decision(positivity_decision);
    emit_radial_fourier_audit_for_cycle(
        diagnostics::RadialFourierStageId::Positivity,
        diagnostics::RadialFourierStagePhase::After,
        state.t,
        static_cast<std::uint64_t>(state.step));

    if (is_2d) {
      state.mesh.materialize_host_svec();
    }
    const auto t_diag_start =
        verbose_phase_timing ? Clock::now() : Clock::time_point{};
    diagnostics::log_energy_budget_step(step_budget, cfg, state.step, state.t);
    diagnostics::log_ale_closure_audit_step(
        ale_closure_audit, step_budget, cfg, state.step, state.t);
    if (cfg.main.dim == 2) {
      tenryu::hydro::refinement_estimator_step(state, cfg);
      tenryu::hydro::refinement_tracker_step(state, cfg);
      tenryu::hydro::evacuated_cell_shadow_step(state, cfg);
    }

    emit_due_outputs([&](diagnostics::HistorySnapshot& snapshot) {
      if (cfg.numerics.diagnostics.dt_breakdown_history_enabled) {
        snapshot.dt_breakdown =
            make_dt_breakdown_history_record(state, step_dt_lineage);
      }
      if (hydro::braginskii::params_from_config(cfg).enabled) {
        const auto brag_diag =
            hydro::braginskii::compute_history_diagnostics(state, cfg);
        if (brag_diag.valid) {
          auto& pv = snapshot.plasma_viscosity;
          pv.valid = true;
          pv.cycle = static_cast<int>(state.step);
          pv.t_s = state.t;
          pv.eta_i_max = brag_diag.eta_i_max;
          pv.eta_e_max = brag_diag.eta_e_max;
          pv.eta_eff_max = brag_diag.eta_eff_max;
          pv.ratio_min = brag_diag.ratio_min;
          pv.ratio_geomean_masswt = brag_diag.ratio_geomean_masswt;
          pv.ratio_max = brag_diag.ratio_max;
          pv.n_cells_e_dom = brag_diag.n_cells_e_dom;
          pv.n_cells_i_dom = brag_diag.n_cells_i_dom;
          pv.n_cells_mixed = brag_diag.n_cells_mixed;
          pv.n_cells_active = brag_diag.n_cells_active;
          pv.heat_rate_i_tot = brag_diag.heat_rate_i_tot;
          pv.heat_rate_e_tot = brag_diag.heat_rate_e_tot;
        }
      }
      snapshot.energy = step_budget;
      snapshot.conservation_residuals = conservation_residuals;
      snapshot.clamp_count = step_clamp_count;
      snapshot.phase_energy = phase_energy;
      snapshot.ale_closure_audit = ale_closure_audit;
      if (cfg.radiation.enabled) {
        snapshot.mc.ddmc_mode_count = imc.last_ddmc_mode_count();
        snapshot.mc.imc_mode_count = imc.last_imc_mode_count();
        snapshot.mc.n_total = imc.last_n_total();
        snapshot.mc.n_imc_particles = imc.last_n_imc_particles();
        snapshot.mc.n_ddmc_particles = imc.last_n_ddmc_particles();
        snapshot.mc.n_census = imc.last_n_census();
        snapshot.mc.n_absorbed = imc.last_n_absorbed();
        snapshot.mc.n_escaped = imc.last_n_escaped();
        snapshot.mc.n_leaked = imc.last_ddmc_to_imc_conversions();
        snapshot.mc.ddmc_fraction = imc.last_ddmc_fraction();
        snapshot.mc.weight_min = imc.last_weight_min();
        snapshot.mc.weight_mean = imc.last_weight_mean();
        snapshot.mc.weight_max = imc.last_weight_max();
        snapshot.mc.overshoot_count = imc.last_overshoot_count();
        snapshot.mc.overshoot_max = imc.last_overshoot_max();
        snapshot.mc.mmatrix_violations = imc.last_mmatrix_violations();
        snapshot.mc.mmatrix_fallback_count = imc.last_mmatrix_fallback_count();
        snapshot.mc.omega_below_threshold = imc.last_omega_below_threshold();
        snapshot.mc.interface_transitions =
            saturating_i64(imc.last_interface_transitions());
        snapshot.mc.interface_reflections =
            saturating_i64(imc.last_interface_reflections());
        snapshot.mc.conversion_prob_violations =
            saturating_i64(imc.last_conversion_prob_violations());
        snapshot.mc.ddmc_to_imc_conversions = imc.last_ddmc_to_imc_conversions();
        snapshot.mc.rad_momentum_deposition = imc.last_rad_momentum_deposition();
        const auto& difference_ref = imc.last_reference_field_diagnostics();
        snapshot.mc.difference_reference_valid = difference_ref.valid ? 1 : 0;
        snapshot.mc.difference_eligible_cells = difference_ref.eligible_cells;
        snapshot.mc.difference_active_cells = difference_ref.active_cells;
        snapshot.mc.difference_strong_cells = difference_ref.strong_cells;
        snapshot.mc.difference_hybrid_suppressed_cells =
            difference_ref.hybrid_suppressed_cells;
        snapshot.mc.difference_W_min = difference_ref.W_min;
        snapshot.mc.difference_W_mean = difference_ref.W_mean;
        snapshot.mc.difference_W_max = difference_ref.W_max;
        snapshot.mc.difference_tau_min = difference_ref.tau_min;
        snapshot.mc.difference_tau_mean = difference_ref.tau_mean;
        snapshot.mc.difference_tau_max = difference_ref.tau_max;
        snapshot.mc.difference_chi_mean = difference_ref.chi_mean;
        snapshot.mc.difference_chi_max = difference_ref.chi_max;
        snapshot.mc.difference_reduced_flux_max = difference_ref.reduced_flux_max;
        snapshot.mc.difference_knudsen_max = difference_ref.knudsen_max;
        snapshot.mc.difference_front_grad_Te_max = difference_ref.front_grad_Te_max;
        snapshot.mc.difference_front_grad_rho_max = difference_ref.front_grad_rho_max;
        snapshot.mc.difference_E_ref_total = difference_ref.E_ref_total;
        const auto& holo = imc.last_holo_selector_diagnostics();
        snapshot.mc.holo_n_core_cells = holo.n_core_cells;
        snapshot.mc.holo_n_entered = holo.n_entered;
        snapshot.mc.holo_n_exited = holo.n_exited;
        snapshot.mc.holo_n_hard_exited = holo.n_hard_exited;
        snapshot.mc.holo_n_island_rejected = holo.n_island_rejected;
        snapshot.mc.holo_tau_R_min = holo.tau_R_min;
        snapshot.mc.holo_tau_R_max = holo.tau_R_max;
        snapshot.mc.holo_reduced_flux_max = holo.reduced_flux_max;
        snapshot.mc.holo_E_LO_total = compute_holo_E_LO_total(state, cfg);
        snapshot.mc.holo_E_LO_boundary_in = step_holo_boundary_in;
        snapshot.mc.holo_E_LO_boundary_out = step_holo_boundary_out;
        snapshot.mc.holo_matter_delta = step_holo_matter_delta;
        snapshot.mc.holo_source_balance_error = step_holo_source_balance_error;
        snapshot.mc.holo_particle_net_source_core =
            compute_holo_particle_net_source_core(state, cfg);
        snapshot.mc.holo_lo_particle_source_mismatch =
            step_holo_matter_delta - snapshot.mc.holo_particle_net_source_core;
        const HoloPrrSummary holo_prr = compute_holo_prr_summary(state, cfg);
        snapshot.mc.holo_Prr_coverage = holo_prr.coverage;
        snapshot.mc.holo_chi_min = holo_prr.chi_min;
        snapshot.mc.holo_chi_mean = holo_prr.chi_mean;
        snapshot.mc.holo_chi_max = holo_prr.chi_max;
      }
      if (cfg.diagnostics.areal_density.enabled ||
          cfg.diagnostics.sphericity.enabled) {
        diagnostics::ArealDensityDiagnostics areal_device{};
        diagnostics::SphericityDiagnostics sphericity_device{};
        const bool shape_device_ok =
            part_info.n_ranks == 1 &&
            diagnostics::compute_shape_history_device(
                state,
                cfg,
                cfg.diagnostics.areal_density.enabled ? &areal_device
                                                      : nullptr,
                cfg.diagnostics.sphericity.enabled ? &sphericity_device
                                                   : nullptr);
        if (cfg.diagnostics.areal_density.enabled) {
          snapshot.areal_density =
              shape_device_ok ? areal_device
                              : diagnostics::compute_areal_density(state, cfg);
        }
        if (cfg.diagnostics.sphericity.enabled) {
          snapshot.sphericity =
              shape_device_ok ? sphericity_device
                              : diagnostics::compute_sphericity(state, cfg);
        }
      }
      if (cfg.diagnostics.laser_pattern.enabled) {
        const laser::LaserMesh* mesh_ptr = cfg.laser.enabled ? &laser_mesh : nullptr;
        snapshot.laser_pattern =
            diagnostics::compute_laser_pattern(state, cfg, mesh_ptr);
      }
      snapshot.fld_solver.outer_iterations =
          static_cast<std::int64_t>(state.fld_outer_iterations);
      snapshot.fld_solver.outer_residual = state.fld_outer_residual;
      snapshot.fld_solver.outer_converged = state.fld_converged ? 1 : 0;
      if (icf_history_enabled) {
        snapshot.icf_shell = diagnostics::compute_icf_shell_diagnostics(
            state, cfg, profile_observability_.shell_initial_radius_cm());
        if (snapshot.icf_shell.valid &&
            !(profile_observability_.shell_initial_radius_cm() > 0.0)) {
          profile_observability_.set_shell_initial_radius_cm(
              snapshot.icf_shell.R_initial_cm);
        }
      }
      if (hotspot_gas_history_enabled) {
        snapshot.hotspot_gas =
            diagnostics::compute_hotspot_gas_diagnostics(state, cfg);
      }
      if (ale_provenance_history_enabled || mesh_quality_min_enabled ||
          ale_state_monitor_enabled) {
        snapshot.ale_provenance = &profile_observability_;
      }
      if (conservation_history_enabled) {
        snapshot.operator_residuals = operator_energy_tracker.entries();
      }
    });
    state.checkpoint_request = false;
    if (cfg.main.dim == 2 &&
        cfg.numerics.diagnostics.refinement_autopilot.mode == "arm_exit") {
      const auto* fire =
          tenryu::hydro::refinement_tracker_pending_fire();
      if (fire != nullptr) {
        if (out.last_checkpoint_step() == state.step) {
          if (part_info.rank == 0) {
            const auto& autopilot =
                cfg.numerics.diagnostics.refinement_autopilot;
            const std::filesystem::path decision_path =
                std::filesystem::path(out.output_dir) /
                "autopilot_decision.json";
            std::ostringstream decision_record;
            decision_record << std::scientific << std::setprecision(17);
            decision_record << "{\n";
            decision_record << "  \"checkpoint_path\": \""
                            << json_escape(out.last_checkpoint_path())
                            << "\",\n";
            decision_record << "  \"checkpoint_step\": " << fire->step
                            << ",\n";
            decision_record << "  \"t\": " << fire->t << ",\n";
            decision_record << "  \"s50_cm\": " << fire->s50_cm << ",\n";
            decision_record << "  \"s_lead_cm\": " << fire->s_lead_cm
                            << ",\n";
            decision_record << "  \"d_clear_h\": " << fire->d_clear_h
                            << ",\n";
            decision_record << "  \"a_old\": " << fire->a_old << ",\n";
            decision_record << "  \"dt_arrival_early_s\": "
                            << fire->dt_arrival_early_s << ",\n";
            decision_record << "  \"q99_e\": " << fire->q99_e << ",\n";
            decision_record << "  \"e_on\": " << autopilot.e_on << ",\n";
            decision_record << "  \"e_off\": " << autopilot.e_off << ",\n";
            decision_record << "  \"n_q_plan\": " << autopilot.n_q_plan
                            << ",\n";
            decision_record << "  \"chi_design\": "
                            << autopilot.chi_design << ",\n";
            decision_record << "  \"s_rep_cm\": " << autopilot.s_rep_cm
                            << ",\n";
            decision_record << "  \"handoff_cm\": "
                            << autopilot.handoff_cm << ",\n";
            decision_record << "  \"window_lo_h\": "
                            << autopilot.window_lo_h << ",\n";
            decision_record << "  \"window_hi_h\": "
                            << autopilot.window_hi_h << "\n";
            decision_record << "}\n";

            std::ofstream decision_file(
                decision_path, std::ios::binary | std::ios::trunc);
            TENRYU_ASSERT(
                decision_file.good(),
                "Failed to open autopilot_decision.json for writing");
            decision_file << decision_record.str();
            TENRYU_ASSERT(
                decision_file.good(),
                "Failed while writing autopilot_decision.json");
            decision_file.flush();
            TENRYU_ASSERT(
                decision_file.good(),
                "Failed to flush autopilot_decision.json");
            decision_file.close();
            TENRYU_ASSERT(
                !decision_file.fail(),
                "Failed to close autopilot_decision.json");
          }
          core::log_info(std::string("[refine_apilot] FIRE checkpoint=") +
                         out.last_checkpoint_path());
          tenryu::hydro::refinement_tracker_clear_fire();
          autopilot_fire_stop = true;
        } else {
          core::log_warning(
              "[refine_apilot] fire pending but no checkpoint was written "
              "this step; holding (fail-closed)");
          tenryu::hydro::refinement_tracker_clear_fire();
        }
      }
    }
    run_shock_approach_diag(state, cfg);
    log_progress_if_needed(state, cfg, reducer);
    if (cfg.main.verbosity == "verbose" &&
        (cfg.numerics.hydro.strategy_first_retry_enabled ||
         same_dt_strategy_retries_attempted > 0)) {
      core::log_info(
          "[driver-retry-strategy-first] step=" + std::to_string(state.step) +
          " same_dt_retries_attempted=" +
          std::to_string(same_dt_strategy_retries_attempted) +
          " same_dt_retries_succeeded=" +
          std::to_string(same_dt_strategy_retries_succeeded));
    }
    if (verbose_phase_timing) {
      step_diag_ms += ms(t_diag_start, Clock::now());
      const double step_total_ms = ms(t_step_start, Clock::now());
      const double known_phase_ms = step_hydro_ms + step_cond_ms + step_laser_ms +
                                    step_rad_ms + step_dt_ms + step_energy_ms +
                                    step_diag_ms;
      const double step_other_ms = std::max(0.0, step_total_ms - known_phase_ms);
      std::ostringstream oss;
      oss << "[phase_timing] step=" << state.step
          << " dt=" << format_sci(dt)
          << std::fixed << std::setprecision(2)
          << " hydro=" << step_hydro_ms
          << " cond=" << step_cond_ms
          << " laser=" << step_laser_ms
          << " rad=" << step_rad_ms
          << " dt_calc=" << step_dt_ms
          << " energy=" << step_energy_ms
          << " diag=" << step_diag_ms
          << " other=" << step_other_ms
          << " total=" << step_total_ms << " ms";
      core::log_info(oss.str());
    }
    hydro::suppress_tangential_velocity(state, state.t);
    hydro::accumulate_drive_work_ledger(
        state, cfg, state.dt, state.t - 0.5 * state.dt);
    hydro::core_clearance_controller_record_step(
        state, state.step, state.t);
    hydro::shadow_homothety_record_step(state, state.step, state.t);
    break;
    }
    if (autopilot_fire_stop) {
      break;
    }
  }

  hydro::stripe_gate_finalize();

  if (cfg.output.write_final_snapshot && part_info.rank == 0 &&
      last_plot_step != state.step) {
    out.write_snapshot(state, cfg, state.step, state.t, case_name,
                       part_info.rank);
    out.write_run_info(state, cfg);
    core::log_info("[output] wrote final snapshot (step=" +
                   std::to_string(state.step) +
                   ", t=" + format_sci(state.t) + ")");
  }

  if (autopilot_fire_stop) {
    out.set_termination_reason("autopilot_fire");
  } else if (state.t >= cfg.main.t_end) {
    out.set_termination_reason("t_end_reached");
  } else if (state.step >= cfg.main.max_steps) {
    out.set_termination_reason("max_steps_reached");
  } else {
    out.set_termination_reason("loop_exit_condition");
  }
  process_escape_valve_events_for_audit();
  const auto escape_valve_final_decision =
      escape_valve_audit.finalize(state.step, state.t);
  if (escape_valve_final_decision.fatal) {
    handle_escape_valve_audit_decision(escape_valve_final_decision);
  } else {
    write_escape_valve_audit_history();
  }
  const auto positivity_final_decision =
      positivity_tracker.finalize(state.step, state.t);
  handle_positivity_audit_decision(positivity_final_decision);
  if (production_audit_enabled) {
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(6)
        << "[production-audit-conservation] max_mass="
        << conservation_residual_max_mass << " max_R_momentum="
        << conservation_residual_max_r_momentum << " max_Z_momentum="
        << conservation_residual_max_z_momentum
        << " max_vol_closure=" << vol_closure_residual_max
        << " max_vol_closure_mass=" << vol_closure_residual_max_mass
        << " max_vol_closure_internal_energy="
        << vol_closure_residual_max_internal_energy
        << " max_vol_closure_R_momentum="
        << vol_closure_residual_max_r_momentum
        << " max_vol_closure_Z_momentum="
        << vol_closure_residual_max_z_momentum;
    core::log_info(oss.str());
  }
  if (f09_corner_mass_fallback_enabled) {
    const hydro::rz::CornerMassFallbackRecorder recorder =
        hydro::rz::corner_mass_fallback_copy_to_host();
    if (part_info.rank == 0 &&
        (recorder.fire_count_bbsw > 0 ||
         recorder.fire_count_subpolygon > 0)) {
      core::log_warning(
          f09_fallback_line("[f09-fallback] run total:", recorder));
    }
  }
  hydro::corner_collapse_ledger_flush();
  hydro::ale::remap_dispatch_audit_flush();
  hydro::ale::gcl_audit_flush();
  hydro::core_clearance_controller_flush();
  hydro::shadow_homothety_flush();
  if (cfg.numerics.hydro.axis_projection.enabled) {
    hydro::axis_projection::run_end(cfg, part_info.rank);
  }
  hydro::band_ale::run_end(cfg, part_info.rank);
  hydro::destroy_axis_core_disk();
  hydro::compatible::destroy_csw_polar_slaving_runtime();
  hydro::wake_angular_heat_flux::destroy();
  if (cfg.numerics.profile.icf_standard_ale.enabled ||
      ale_provenance_history_enabled ||
      mesh_quality_min_enabled ||
      ale_state_monitor_enabled) {
    profile_observability_.finalize_at_run_end(cfg, state.t >= cfg.main.t_end);
    if (cfg.numerics.profile.icf_standard_ale.enabled) {
      log_profile_observability(profile_observability_, "run_end");
    }
#if TENRYU_ENABLE_HDF5
    if ((ale_provenance_history_enabled || mesh_quality_min_enabled ||
         ale_state_monitor_enabled) &&
        history_writer.enabled() &&
        part_info.rank == 0) {
      history_writer.flush_pending();
      diagnostics::append_ale_provenance_final_to_history_file(
          history_writer.path(),
          profile_observability_,
          cfg,
          ale_provenance_name(profile_observability_.compute_label()),
          claim_level_name(profile_observability_.claim_level),
          state.t,
          state.step);
    }
#endif
  }
  // MPI gate tool: per-rank owned-slab dump for rank-count agreement checks
  // (design doc mpi_m18_20_20260717.md §5.4-5.6). Enabled by
  // TENRYU_MPI_DUMP_OWNED=<prefix>; compared by
  // tools/validation/compare_owned_dump.py.
  if (const char* dump_prefix = std::getenv("TENRYU_MPI_DUMP_OWNED")) {
    const int n_cells = static_cast<int>(state.rho.size());
    const int n_nodes = static_cast<int>(state.x_r.size());
    const auto cell_w = state.owned_cell_window(n_cells);
    const auto node_w = state.owned_node_window(n_nodes);
    const auto dump_field = [&](std::FILE* fp, const auto& f,
                                const core::State::LaunchWindow& w) {
      if (f.empty() || w.count() <= 0 ||
          static_cast<int>(f.size()) < w.end) {
        const int zero = 0;
        std::fwrite(&zero, sizeof(int), 1, fp);
        return;
      }
      std::vector<double> host(f.size());
      f.copy_to_host(host.data());
      const int count = w.count();
      std::fwrite(&count, sizeof(int), 1, fp);
      std::fwrite(host.data() + w.begin, sizeof(double), count, fp);
    };
    const std::string path = std::string(dump_prefix) + "_r" +
                             std::to_string(part_info.rank) + ".bin";
    std::FILE* fp = std::fopen(path.c_str(), "wb");
    if (fp != nullptr) {
      const int header[6] = {part_info.rank, part_info.n_ranks,
                             cell_w.begin, cell_w.end,
                             node_w.begin, node_w.end};
      std::fwrite(header, sizeof(int), 6, fp);
      dump_field(fp, state.rho, cell_w);
      dump_field(fp, state.ee, cell_w);
      dump_field(fp, state.ei, cell_w);
      dump_field(fp, state.Te, cell_w);
      dump_field(fp, state.Ti, cell_w);
      dump_field(fp, state.mass, cell_w);
      dump_field(fp, state.vol, cell_w);
      dump_field(fp, state.x_r, node_w);
      dump_field(fp, state.v_r, node_w);
      dump_field(fp, state.x_z, node_w);
      dump_field(fp, state.v_z, node_w);
      // rad_E is cell-major group-minor (rad_E[c*G+g]): owned cells map to
      // the contiguous flat range scaled by the group count.
      core::State::LaunchWindow rad_w = cell_w;
      if (!state.rad_E.empty() && n_cells > 0 &&
          state.rad_E.size() % static_cast<std::size_t>(n_cells) == 0) {
        const int n_groups =
            static_cast<int>(state.rad_E.size() / static_cast<std::size_t>(n_cells));
        rad_w.begin = cell_w.begin * n_groups;
        rad_w.end = cell_w.end * n_groups;
      }
      dump_field(fp, state.rad_E, rad_w);
      dump_field(fp, state.laser_dep, cell_w);
      dump_field(fp, state.Qvisc, cell_w);
      std::fclose(fp);
      core::log_info("[mpi-dump] wrote owned-slab dump: " + path);
    }
  }
  if (part_info.rank == 0) {
    if (cfg.numerics.ale.safe_backtrack_enabled) {
      hydro::ale::log_safe_backtrack_lambda_distribution(
          cfg.numerics.ale.safe_backtrack_min_exp);
    }
    out.write_run_info(state, cfg);
  }
}

}  // namespace tenryu::coupling
