#include "coupling/dispatcher_decision.hpp"

#include <algorithm>
#include <cmath>

namespace tenryu::coupling {

DispatchDecision classify_retry(const DispatchInputs& in) {
  DispatchDecision out;

  out.kind = DispatchDecisionKind::FallThrough;
  out.forced_mode = hydro::ale::AleMode::ScheduledDefault;
  out.dt_retry = 0.0;

  if (!in.feature_enabled) {
    return out;
  }

  const HydroStepResult& r = in.last_hydro_result;

  out.axis_regime = (r.regime == hydro::MeshFailureRegime::AxisFace ||
                     r.regime == hydro::MeshFailureRegime::AxisBand);
  out.state_sensitive_observed = r.state_sensitive;
  out.dt_sensitive_observed = r.dt_sensitive;

  if (!out.axis_regime) {
    return out;
  }

  if (r.state_sensitive) {
    if (in.axis_repair_tried_at_current_state) {
      return out;
    }
    if (in.repair_generation_this_step >= in.repair_generation_per_step_cap) {
      out.kind = DispatchDecisionKind::StateRepairCapHitFallThrough;
      out.repair_cap_hit = true;
      return out;
    }
    out.kind = DispatchDecisionKind::RepairSameDt;
    out.forced_mode = hydro::ale::AleMode::AxisSpinePlusLocal;
    out.dt_retry = in.dt;
    return out;
  }

  if (r.dt_sensitive) {
    const double dt_star = r.dt_star_est;
    double dt_target = in.dt * 0.5;
    if (std::isfinite(dt_star) && dt_star > 0.0) {
      dt_target = std::min(dt_target, dt_star);
    }
    const double dt_floor = std::max(in.dt_min_s, in.dt * (1.0 / 16.0));
    if (dt_target <= dt_floor) {
      out.dt_star_degenerate_reclassified = true;
      if (in.axis_repair_tried_at_current_state) {
        return out;
      }
      if (in.repair_generation_this_step >= in.repair_generation_per_step_cap) {
        out.kind = DispatchDecisionKind::StateRepairCapHitFallThrough;
        out.repair_cap_hit = true;
        return out;
      }
      out.kind = DispatchDecisionKind::RepairSameDt;
      out.forced_mode = hydro::ale::AleMode::AxisSpinePlusLocal;
      out.dt_retry = in.dt;
      return out;
    }
    out.kind = DispatchDecisionKind::HalveToDtStar;
    out.forced_mode = hydro::ale::AleMode::ScheduledDefault;
    out.dt_retry = dt_target;
    return out;
  }

  return out;
}

}  // namespace tenryu::coupling
