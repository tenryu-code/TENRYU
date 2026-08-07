#pragma once

#include <cstdint>

#include "coupling/hydro_step_result.hpp"
#include "hydro/ale_mode.hpp"

namespace tenryu::coupling {

enum class DispatchDecisionKind : std::uint8_t {
  FallThrough = 0,
  RepairSameDt,
  HalveToDtStar,
  StateRepairCapHitFallThrough,
};

inline const char* dispatch_decision_kind_name(const DispatchDecisionKind k) {
  switch (k) {
    case DispatchDecisionKind::FallThrough:
      return "fall_through";
    case DispatchDecisionKind::RepairSameDt:
      return "repair_same_dt";
    case DispatchDecisionKind::HalveToDtStar:
      return "halve_to_dt_star";
    case DispatchDecisionKind::StateRepairCapHitFallThrough:
      return "state_repair_cap_hit_fall_through";
  }
  return "unknown";
}

struct DispatchDecision {
  DispatchDecisionKind kind = DispatchDecisionKind::FallThrough;
  // Override knobs. Caller applies these AFTER computing the legacy repair plan.
  // - When kind == RepairSameDt: forced_mode and dt_retry are populated.
  // - When kind == HalveToDtStar: forced_mode is ignored; dt_retry is populated.
  // - When kind == FallThrough or StateRepairCapHitFallThrough: both are zero/Default.
  hydro::ale::AleMode forced_mode = hydro::ale::AleMode::ScheduledDefault;
  double dt_retry = 0.0;
  // Telemetry hints (always populated; caller decides what to log).
  bool axis_regime = false;
  bool state_sensitive_observed = false;
  bool dt_sensitive_observed = false;
  bool repair_cap_hit = false;
  bool dt_star_degenerate_reclassified = false;
};

struct DispatchInputs {
  HydroStepResult last_hydro_result{};
  bool feature_enabled = false;
  bool axis_repair_tried_at_current_state = false;
  int repair_generation_this_step = 0;
  int repair_generation_per_step_cap = 3;
  double dt = 0.0;
  // Round D-fix: Numerics.dt.min_s. Used to reject degenerate dt_star_est
  // targets that would drive retry dt below the admissible floor.
  double dt_min_s = 0.0;
};

DispatchDecision classify_retry(const DispatchInputs& in);

}  // namespace tenryu::coupling
