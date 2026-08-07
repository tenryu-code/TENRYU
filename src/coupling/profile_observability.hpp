#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "mesh/mesh_geometry_result.hpp"

namespace tenryu::core {
struct State;
}

namespace tenryu::parallel {
class Reduction;
}

namespace tenryu::coupling {

enum class AleProvenance : std::uint8_t {
  LegacyDefaultUnclassified = 0,
  PublicBaseline,
  PublicBaselineFailedNoEscape,
  TenryuExtendedAle,
  TenryuExtendedAleWithCellDeactivation,
};

enum class ClaimLevel : std::uint8_t {
  Characterization = 0,
  PrePlicSmoke,
  ProductionComparable,
};

enum class ProductionComparableWaveFStatus : std::uint8_t {
  PASS = 0,
  PARTIAL_A_CR_PROGRESSION,
  PARTIAL_B_PRODUCTION_RESIDUAL,
  INCONCLUSIVE,
  FAIL,
  DISABLED,
};

struct ProductionComparableWaveFCriteria {
  bool plic_enabled = false;
  bool wave_f_enabled = false;
  bool reached_t_end = false;
  bool public_baseline_provenance = false;
  bool class_d_aggregate_ok = false;
  bool plic_success_rate_ok = false;
  bool per_material_conservation_residual_ok = false;
  double plic_success_rate = 0.0;
  double per_material_conservation_max_rel_residual = 0.0;
};

struct EscapeValveEvent {
  struct CellState {
    double rho = 0.0;
    double p = 0.0;
    double Te = 0.0;
    double Ti = 0.0;
  };

  std::string split_phase;
  std::string operator_inserted;
  bool order_degraded = true;
  double E_thermal_before = 0.0;
  double E_thermal_after = 0.0;
  double E_kinetic_before = 0.0;
  double E_kinetic_after = 0.0;
  // Stage 27 wires transfer terms as explicit zeros unless a rescue path
  // exposes source-term transfer data.
  double mass_transfer = 0.0;
  double momentum_transfer_R = 0.0;
  double momentum_transfer_Z = 0.0;
  double internal_energy_transfer = 0.0;
  int step = -1;
  double time = 0.0;
  int cell_id = -1;
  int cell_i = -1;
  int cell_j = -1;
  std::string flag_name;
  std::string reason;
  double mass_delta = 0.0;
  double momentum_delta_r = 0.0;
  double momentum_delta_z = 0.0;
  double energy_delta = 0.0;
  CellState before_state;
  CellState after_state;
};

struct ProfileObservability {
  bool profile_enabled = false;
  ClaimLevel claim_level = ClaimLevel::Characterization;

  int forbidden_config_violations = 0;
  int escape_valve_activations = 0;
  int class_c_runtime_fires = 0;
  int mesh_geometry_failures_observed = 0;
  int public_baseline_terminal_failures = 0;
  bool emergency_cell_deactivation_fired = false;

  // Stage 30 Wave A skeleton: class-(d) PLIC degradation tracking.
  // Wiring (note_plic_*) is Wave B. Wave A only declares storage.
  struct PlicDegradationEvent {
    std::uint8_t case_id = 0;
    std::uint8_t severity = 0;
    int cell_idx = -1;
    int i = -1;
    int j = -1;
    double eta_E = 0.0;
    double grad_F_magnitude = 0.0;
    std::string severity_metric_kind;
    double severity_metric_value = 0.0;
    std::string fallback_used;
    bool prev_normal_invalidated = false;
    int step = -1;
    double time = 0.0;
  };

  int class_d_runtime_fires_matrix[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
  int interface_cells_observed = 0;
  int interface_reconstruction_attempt_count = 0;
  int interface_reconstruction_success_count = 0;
  int axis_exempt_cells_count = 0;
  double plic_max_eta_E_observed = 0.0;
  double plic_max_volume_fraction_residual_observed = 0.0;
  double plic_min_grad_F_observed = 0.0;
  bool plic_remap_fallback_engaged = false;
  std::vector<PlicDegradationEvent> plic_events;
  std::size_t plic_events_emitted_count = 0;
  bool plic_gate_status_recorded = false;
  std::string plic_gate_status;

  int last_failing_cell = -1;
  int last_failing_i = -1;
  int last_failing_j = -1;
  double last_min_cell_vol = 0.0;
  double last_min_corner_j = 0.0;
  tenryu::mesh::MeshGeometryFailureKind last_failure_kind =
      tenryu::mesh::MeshGeometryFailureKind::None;

  bool mesh_quality_min_observed = false;
  double achieved_min_corner_j_rel = std::numeric_limits<double>::infinity();
  double achieved_min_gauss_j_rel = std::numeric_limits<double>::infinity();
  double achieved_min_rz_volume_rel = std::numeric_limits<double>::infinity();
  double achieved_min_edge_length_rel = std::numeric_limits<double>::infinity();
  double achieved_min_altitude_rel = std::numeric_limits<double>::infinity();
  double achieved_max_condition_number = 0.0;
  std::uint64_t negative_rz_volume_count_total = 0;

  bool reached_t_end = false;
  double R_shell_initial_cm = -1.0;
  std::vector<EscapeValveEvent> escape_valve_events;

  void reset_at_run_start(const tenryu::core::Config& cfg) {
    profile_enabled = cfg.numerics.profile.icf_standard_ale.enabled;
    claim_level = ClaimLevel::Characterization;
    if (profile_enabled) {
      const std::string& configured =
          cfg.numerics.profile.icf_standard_ale.claim_level;
      if (configured == "characterization") {
        claim_level = ClaimLevel::Characterization;
      } else if (configured == "pre_plic_smoke") {
        claim_level = ClaimLevel::PrePlicSmoke;
      } else if (configured == "production_comparable") {
        claim_level = ClaimLevel::ProductionComparable;
      } else {
        tenryu::core::log_warning(
            "[ale_provenance] unknown claim_level '" + configured +
            "'; falling back to characterization");
      }
    }
    forbidden_config_violations = 0;
    escape_valve_activations = 0;
    class_c_runtime_fires = 0;
    mesh_geometry_failures_observed = 0;
    public_baseline_terminal_failures = 0;
    emergency_cell_deactivation_fired = false;
    last_failing_cell = -1;
    last_failing_i = -1;
    last_failing_j = -1;
    last_min_cell_vol = 0.0;
    last_min_corner_j = 0.0;
    last_failure_kind = tenryu::mesh::MeshGeometryFailureKind::None;
    mesh_quality_min_observed = false;
    achieved_min_corner_j_rel = std::numeric_limits<double>::infinity();
    achieved_min_gauss_j_rel = std::numeric_limits<double>::infinity();
    achieved_min_rz_volume_rel = std::numeric_limits<double>::infinity();
    achieved_min_edge_length_rel = std::numeric_limits<double>::infinity();
    achieved_min_altitude_rel = std::numeric_limits<double>::infinity();
    achieved_max_condition_number = 0.0;
    negative_rz_volume_count_total = 0;
    reached_t_end = false;
    R_shell_initial_cm = -1.0;
    escape_valve_events.clear();
    for (auto& row : class_d_runtime_fires_matrix) {
      for (int& value : row) {
        value = 0;
      }
    }
    interface_cells_observed = 0;
    interface_reconstruction_attempt_count = 0;
    interface_reconstruction_success_count = 0;
    axis_exempt_cells_count = 0;
    plic_max_eta_E_observed = 0.0;
    plic_max_volume_fraction_residual_observed = 0.0;
    plic_min_grad_F_observed = 0.0;
    plic_remap_fallback_engaged = false;
    plic_events.clear();
    plic_events_emitted_count = 0;
    plic_gate_status_recorded = false;
    plic_gate_status.clear();
  }

  [[nodiscard]] double shell_initial_radius_cm() const {
    return R_shell_initial_cm;
  }

  void set_shell_initial_radius_cm(const double value) {
    R_shell_initial_cm = value;
  }

  void note_class_c_fire(const std::string& mechanism_name) {
    ++class_c_runtime_fires;
    tenryu::core::log_warning(
        "[ale_provenance] class-(c) fire: " + mechanism_name +
        ". Run will be classified TENRYU_EXTENDED_ALE.");
  }

  void note_emergency_cell_deactivation() {
    emergency_cell_deactivation_fired = true;
    ++class_c_runtime_fires;
    tenryu::core::log_warning(
        "[ale_provenance] emergency cell deactivation fired. Run will be "
        "classified TENRYU_EXTENDED_ALE_WITH_CELL_DEACTIVATION.");
  }

  void note_escape_valve_activation(const std::string& reason) {
    ++escape_valve_activations;
    tenryu::core::log_warning(
        "[ale_provenance] escape valve activation: " + reason);
  }

  void note_escape_valve_event(const EscapeValveEvent& evt) {
    escape_valve_events.push_back(evt);
    note_escape_valve_activation(evt.operator_inserted);
    tenryu::core::log_warning(
        "[ale_provenance] escape valve event: phase=" + evt.split_phase +
        " operator=" + evt.operator_inserted +
        " order_degraded=" + (evt.order_degraded ? "true" : "false"));
  }

  void note_plic_attempt_success(int cell_idx, double eta_E);
  void note_plic_attempt_failure(int cell_idx, const std::string& fallback_used);
  void note_axis_exempt_cell(int cell_idx);
  void note_plic_degradation(const PlicDegradationEvent& evt);
  void note_plic_remap_fallback(const std::string& reason, int step, double time);
  void note_committed_mesh_quality_min(
      const tenryu::core::State& state,
      const tenryu::parallel::Reduction* reduction);

  void note_mesh_geometry_failure(const tenryu::mesh::MeshGeometryResult& r) {
    ++mesh_geometry_failures_observed;
    last_failing_cell = r.failing_cell;
    last_failing_i = r.failing_i;
    last_failing_j = r.failing_j;
    last_min_cell_vol = r.min_cell_vol;
    last_min_corner_j = 0.0;
    last_failure_kind = r.kind;
    tenryu::core::log_warning(
        "[ale_provenance] mesh geometry failure observed: cell=" +
        std::to_string(r.failing_cell) + " kind=" +
        tenryu::mesh::mesh_geometry_failure_kind_name(r.kind));
  }

  void note_validator_violations(const int forbidden_config,
                                 const int escape_activations) {
    forbidden_config_violations = forbidden_config;
    escape_valve_activations += escape_activations;
  }

  void finalize_at_run_end(const tenryu::core::Config& cfg,
                           bool reached_t_end_in);

  AleProvenance compute_label() const;
  [[nodiscard]] std::string compute_final_class_d_aggregate() const;
  [[nodiscard]] std::string compute_plic_gate_status(
      const tenryu::core::Config& cfg) const;
  [[nodiscard]] ProductionComparableWaveFCriteria
  compute_production_comparable_wave_f_criteria(
      const tenryu::core::Config& cfg,
      double per_material_conservation_max_rel_residual) const;
  [[nodiscard]] ProductionComparableWaveFStatus
  compute_production_comparable_wave_f_status(
      const tenryu::core::Config& cfg,
      double per_material_conservation_max_rel_residual,
      double convergence_ratio,
      double epsilon_E,
      bool epsilon_E_sustained) const;
};

const char* ale_provenance_name(AleProvenance label);
const char* claim_level_name(ClaimLevel level);
const char* production_comparable_wave_f_status_name(
    ProductionComparableWaveFStatus status);

void log_profile_observability(const ProfileObservability& obs,
                               const char* phase_tag);

}  // namespace tenryu::coupling
