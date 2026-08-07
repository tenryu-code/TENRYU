#pragma once

#include <cstdint>

#include "core/macros.hpp"

namespace tenryu::hydro::ale_motion {

/*
Design-doc note: quality/feature ALE motion trigger, default-off.

Purpose:
  Replace a driver-only fixed "every N steps" ALE cadence with an explicit
  predicate that rezones only when mesh quality has degraded or when a caller
  supplied maximum number of Lagrangian steps has elapsed.  This preserves
  Lagrangian motion when the mesh is still admissible and reduces unnecessary
  swept volume/remap drain.

Inputs and units:
  All metrics are dimensionless.  The caller must compute RZ volumes using the
  TENRYU convention dV = 2*pi*r*dr*dz in cgs coordinates before forming
  min_cell_volume_ratio.  No energy, EOS, or unit conversion enters this module.

Recommended starting thresholds for I1-B / five-block multiblock:
  min_cell_volume_ratio: on=0.05, off=0.10
  min_corner_j_rel:      on=0.03, off=0.08
  min_gauss_j_rel:       on=0.03, off=0.08
  max_aspect_ratio:      on=20,   off=12
  dt_ratio=dt/dt_prev:   on=0.35, off=0.70

Trigger logic:
  1. A quality latch turns on when any metric crosses its on threshold.
  2. The latch stays on until every enabled metric clears its off threshold.
  3. A step cap may independently request a gentle rezone.
  4. rezone_strength is in [0,1]: marginal degradation gives a gentle value,
     while invalid/near-critical quality gives an aggressive value.  The driver
     can use it to scale rezone relaxation/sigma and/or interpolate between a
     minimum and maximum iteration budget.

Example driver consumption:
  AleMotionTriggerState trigger_state;  // persisted by the driver/state layer
  AleMotionTriggerParams p;
  p.enabled = cfg.numerics.ale.motion_trigger_enabled;
  p.max_steps_since_last_rezone = cfg.numerics.ale.motion_trigger_max_steps;
  const AleMotionTriggerDecision d =
      evaluate_ale_motion_trigger(metrics, p, trigger_state, steps_since_rezone);
  trigger_state = d.next_state;
  if (d.should_rezone) {
    const double omega = cfg.numerics.ale.relaxation * d.rezone_strength;
    const int iters = scale_iteration_budget(
        cfg.numerics.ale.motion_trigger_min_iterations,
        cfg.numerics.ale.max_iterations,
        d.rezone_strength);
    ... run rezone/remap ...
  }

This header intentionally has no Config, State, logger, CUDA runtime, or
namelist dependency.  Integration should pass all policy values explicitly.
*/

enum AleMotionTriggerReason : std::uint32_t {
  kAleMotionTriggerReasonNone = 0U,
  kAleMotionTriggerReasonCornerJ = 1U << 0,
  kAleMotionTriggerReasonGaussJ = 1U << 1,
  kAleMotionTriggerReasonCellVolume = 1U << 2,
  kAleMotionTriggerReasonAspectRatio = 1U << 3,
  kAleMotionTriggerReasonDtCollapse = 1U << 4,
  kAleMotionTriggerReasonStepCap = 1U << 5,
  kAleMotionTriggerReasonHysteresisLatch = 1U << 6,
  kAleMotionTriggerReasonCriticalQuality = 1U << 7
};

struct AleMotionTriggerMetrics {
  double min_corner_j_rel = 1.0;
  double min_gauss_j_rel = 1.0;
  double max_aspect_ratio = 1.0;
  double min_cell_volume_ratio = 1.0;
  double dt_ratio = 1.0;
};

struct AleMotionTriggerParams {
  bool enabled = false;

  double min_corner_j_on = 3.0e-2;
  double min_corner_j_off = 8.0e-2;
  double min_corner_j_critical = 5.0e-3;

  double min_gauss_j_on = 3.0e-2;
  double min_gauss_j_off = 8.0e-2;
  double min_gauss_j_critical = 5.0e-3;

  double min_cell_volume_ratio_on = 5.0e-2;
  double min_cell_volume_ratio_off = 1.0e-1;
  double min_cell_volume_ratio_critical = 1.0e-2;

  double max_aspect_ratio_on = 20.0;
  double max_aspect_ratio_off = 12.0;
  double max_aspect_ratio_critical = 50.0;

  double dt_ratio_on = 0.35;
  double dt_ratio_off = 0.70;
  double dt_ratio_critical = 0.10;

  // <=0 disables elapsed-step forcing; the driver should pass its chosen cap.
  int max_steps_since_last_rezone = 0;

  // Suppresses repeated marginal quality rezones immediately after a rezone.
  // Critical/invalid quality bypasses this interval.
  int min_steps_between_quality_rezones = 1;

  double gentle_strength = 0.25;
  double cap_only_strength = 0.20;
  double max_strength = 1.0;
};

struct AleMotionTriggerState {
  bool quality_latched = false;
  std::uint32_t latched_reason_mask = kAleMotionTriggerReasonNone;
  int suppressed_quality_steps = 0;
  int last_steps_since_rezone = 0;
  double last_strength = 0.0;
};

struct AleMotionTriggerDecision {
  bool should_rezone = false;
  double rezone_strength = 0.0;
  std::uint32_t reason_mask = kAleMotionTriggerReasonNone;
  AleMotionTriggerState next_state{};
};

namespace detail {

TENRYU_HOST_DEVICE inline bool finite_value(const double x) {
  return x >= -1.0e300 && x <= 1.0e300;
}

TENRYU_HOST_DEVICE inline double max_value(const double a, const double b) {
  return a > b ? a : b;
}

TENRYU_HOST_DEVICE inline double clamp01(const double x) {
  if (!(x > 0.0)) {
    return 0.0;
  }
  if (x < 1.0) {
    return x;
  }
  return 1.0;
}

TENRYU_HOST_DEVICE inline double smoothstep01(const double x) {
  const double t = clamp01(x);
  return t * t * (3.0 - 2.0 * t);
}

TENRYU_HOST_DEVICE inline bool low_metric_enabled(const double on,
                                                  const double off) {
  return finite_value(on) && finite_value(off) && on > 0.0 && off > on;
}

TENRYU_HOST_DEVICE inline bool high_metric_enabled(const double on,
                                                   const double off) {
  return finite_value(on) && finite_value(off) && off > 0.0 && on > off;
}

TENRYU_HOST_DEVICE inline bool low_metric_on(const double value,
                                             const double on,
                                             const double off) {
  return low_metric_enabled(on, off) &&
         (!finite_value(value) || value < on);
}

TENRYU_HOST_DEVICE inline bool low_metric_off(const double value,
                                              const double on,
                                              const double off) {
  return !low_metric_enabled(on, off) ||
         (finite_value(value) && value >= off);
}

TENRYU_HOST_DEVICE inline bool low_metric_critical(const double value,
                                                   const double on,
                                                   const double critical) {
  if (!(finite_value(on) && on > 0.0)) {
    return false;
  }
  const double c =
      (finite_value(critical) && critical > 0.0 && critical < on)
          ? critical
          : 0.0;
  return !finite_value(value) || value <= c;
}

TENRYU_HOST_DEVICE inline double low_metric_severity(const double value,
                                                     const double on,
                                                     const double critical) {
  if (!(finite_value(on) && on > 0.0)) {
    return 0.0;
  }
  if (!finite_value(value) || value <= 0.0) {
    return 1.0;
  }
  if (!(value < on)) {
    return 0.0;
  }
  const double c =
      (finite_value(critical) && critical > 0.0 && critical < on)
          ? critical
          : 0.0;
  const double span = on - c;
  if (!(span > 0.0)) {
    return 1.0;
  }
  return smoothstep01((on - value) / span);
}

TENRYU_HOST_DEVICE inline bool high_metric_on(const double value,
                                              const double on,
                                              const double off) {
  return high_metric_enabled(on, off) &&
         (!finite_value(value) || value > on);
}

TENRYU_HOST_DEVICE inline bool high_metric_off(const double value,
                                               const double on,
                                               const double off) {
  return !high_metric_enabled(on, off) ||
         (finite_value(value) && value <= off);
}

TENRYU_HOST_DEVICE inline bool high_metric_critical(const double value,
                                                    const double on,
                                                    const double critical) {
  return finite_value(on) && on > 0.0 &&
         (!finite_value(value) ||
          (finite_value(critical) && critical > on && value >= critical));
}

TENRYU_HOST_DEVICE inline double high_metric_severity(const double value,
                                                      const double on,
                                                      const double critical) {
  if (!(finite_value(on) && on > 0.0)) {
    return 0.0;
  }
  if (!finite_value(value)) {
    return 1.0;
  }
  if (!(value > on)) {
    return 0.0;
  }
  const double c =
      (finite_value(critical) && critical > on) ? critical : (2.0 * on);
  const double span = c - on;
  if (!(span > 0.0)) {
    return 1.0;
  }
  return smoothstep01((value - on) / span);
}

TENRYU_HOST_DEVICE inline double sanitize_strength(const double value) {
  if (!(value > 0.0)) {
    return 0.0;
  }
  if (value < 1.0) {
    return value;
  }
  return 1.0;
}

TENRYU_HOST_DEVICE inline double quality_strength_from_severity(
    const AleMotionTriggerParams& params,
    const double severity) {
  const double gentle = sanitize_strength(params.gentle_strength);
  const double maximum = sanitize_strength(params.max_strength);
  const double base = gentle < maximum ? gentle : maximum;
  return base + (maximum - base) * clamp01(severity);
}

}  // namespace detail

TENRYU_HOST_DEVICE inline bool has_trigger_reason(
    const std::uint32_t mask,
    const AleMotionTriggerReason reason) {
  return (mask & static_cast<std::uint32_t>(reason)) != 0U;
}

TENRYU_HOST_DEVICE inline double ale_motion_trigger_quality_severity(
    const AleMotionTriggerMetrics& metrics,
    const AleMotionTriggerParams& params) {
  double severity = 0.0;
  severity = detail::max_value(
      severity,
      detail::low_metric_severity(metrics.min_corner_j_rel,
                                  params.min_corner_j_on,
                                  params.min_corner_j_critical));
  severity = detail::max_value(
      severity,
      detail::low_metric_severity(metrics.min_gauss_j_rel,
                                  params.min_gauss_j_on,
                                  params.min_gauss_j_critical));
  severity = detail::max_value(
      severity,
      detail::low_metric_severity(metrics.min_cell_volume_ratio,
                                  params.min_cell_volume_ratio_on,
                                  params.min_cell_volume_ratio_critical));
  severity = detail::max_value(
      severity,
      detail::high_metric_severity(metrics.max_aspect_ratio,
                                   params.max_aspect_ratio_on,
                                   params.max_aspect_ratio_critical));
  severity = detail::max_value(
      severity,
      detail::low_metric_severity(metrics.dt_ratio,
                                  params.dt_ratio_on,
                                  params.dt_ratio_critical));
  return detail::clamp01(severity);
}

TENRYU_HOST_DEVICE inline std::uint32_t ale_motion_trigger_on_reasons(
    const AleMotionTriggerMetrics& metrics,
    const AleMotionTriggerParams& params) {
  std::uint32_t mask = kAleMotionTriggerReasonNone;
  if (detail::low_metric_on(metrics.min_corner_j_rel,
                            params.min_corner_j_on,
                            params.min_corner_j_off)) {
    mask |= kAleMotionTriggerReasonCornerJ;
  }
  if (detail::low_metric_on(metrics.min_gauss_j_rel,
                            params.min_gauss_j_on,
                            params.min_gauss_j_off)) {
    mask |= kAleMotionTriggerReasonGaussJ;
  }
  if (detail::low_metric_on(metrics.min_cell_volume_ratio,
                            params.min_cell_volume_ratio_on,
                            params.min_cell_volume_ratio_off)) {
    mask |= kAleMotionTriggerReasonCellVolume;
  }
  if (detail::high_metric_on(metrics.max_aspect_ratio,
                             params.max_aspect_ratio_on,
                             params.max_aspect_ratio_off)) {
    mask |= kAleMotionTriggerReasonAspectRatio;
  }
  if (detail::low_metric_on(metrics.dt_ratio,
                            params.dt_ratio_on,
                            params.dt_ratio_off)) {
    mask |= kAleMotionTriggerReasonDtCollapse;
  }
  return mask;
}

TENRYU_HOST_DEVICE inline std::uint32_t ale_motion_trigger_critical_reasons(
    const AleMotionTriggerMetrics& metrics,
    const AleMotionTriggerParams& params) {
  std::uint32_t mask = kAleMotionTriggerReasonNone;
  if (detail::low_metric_critical(metrics.min_corner_j_rel,
                                  params.min_corner_j_on,
                                  params.min_corner_j_critical)) {
    mask |= kAleMotionTriggerReasonCornerJ;
  }
  if (detail::low_metric_critical(metrics.min_gauss_j_rel,
                                  params.min_gauss_j_on,
                                  params.min_gauss_j_critical)) {
    mask |= kAleMotionTriggerReasonGaussJ;
  }
  if (detail::low_metric_critical(metrics.min_cell_volume_ratio,
                                  params.min_cell_volume_ratio_on,
                                  params.min_cell_volume_ratio_critical)) {
    mask |= kAleMotionTriggerReasonCellVolume;
  }
  if (detail::high_metric_critical(metrics.max_aspect_ratio,
                                   params.max_aspect_ratio_on,
                                   params.max_aspect_ratio_critical)) {
    mask |= kAleMotionTriggerReasonAspectRatio;
  }
  if (detail::low_metric_critical(metrics.dt_ratio,
                                  params.dt_ratio_on,
                                  params.dt_ratio_critical)) {
    mask |= kAleMotionTriggerReasonDtCollapse;
  }
  return mask;
}

TENRYU_HOST_DEVICE inline bool ale_motion_trigger_all_quality_off(
    const AleMotionTriggerMetrics& metrics,
    const AleMotionTriggerParams& params) {
  return detail::low_metric_off(metrics.min_corner_j_rel,
                                params.min_corner_j_on,
                                params.min_corner_j_off) &&
         detail::low_metric_off(metrics.min_gauss_j_rel,
                                params.min_gauss_j_on,
                                params.min_gauss_j_off) &&
         detail::low_metric_off(metrics.min_cell_volume_ratio,
                                params.min_cell_volume_ratio_on,
                                params.min_cell_volume_ratio_off) &&
         detail::high_metric_off(metrics.max_aspect_ratio,
                                 params.max_aspect_ratio_on,
                                 params.max_aspect_ratio_off) &&
         detail::low_metric_off(metrics.dt_ratio,
                                params.dt_ratio_on,
                                params.dt_ratio_off);
}

TENRYU_HOST_DEVICE inline int scale_iteration_budget(const int min_iterations,
                                                     const int max_iterations,
                                                     const double strength) {
  if (max_iterations <= min_iterations) {
    return min_iterations > 0 ? min_iterations : 0;
  }
  const int lo = min_iterations > 0 ? min_iterations : 0;
  const int span = max_iterations - lo;
  const double s = detail::sanitize_strength(strength);
  return lo + static_cast<int>(static_cast<double>(span) * s + 0.5);
}

TENRYU_HOST_DEVICE inline AleMotionTriggerDecision evaluate_ale_motion_trigger(
    const AleMotionTriggerMetrics& metrics,
    const AleMotionTriggerParams& params,
    const AleMotionTriggerState& previous_state,
    const int steps_since_last_rezone) {
  AleMotionTriggerDecision decision;
  decision.next_state = previous_state;
  decision.next_state.last_steps_since_rezone = steps_since_last_rezone;
  decision.next_state.last_strength = 0.0;

  if (!params.enabled) {
    decision.next_state.quality_latched = false;
    decision.next_state.latched_reason_mask = kAleMotionTriggerReasonNone;
    decision.next_state.suppressed_quality_steps = 0;
    return decision;
  }

  const std::uint32_t on_reasons =
      ale_motion_trigger_on_reasons(metrics, params);
  const std::uint32_t critical_reasons =
      ale_motion_trigger_critical_reasons(metrics, params);
  bool quality_latched = previous_state.quality_latched;
  std::uint32_t latched_reasons = previous_state.latched_reason_mask;
  if (on_reasons != kAleMotionTriggerReasonNone) {
    quality_latched = true;
    latched_reasons = on_reasons;
  } else if (ale_motion_trigger_all_quality_off(metrics, params)) {
    quality_latched = false;
    latched_reasons = kAleMotionTriggerReasonNone;
  }

  const bool cap_due =
      params.max_steps_since_last_rezone > 0 &&
      steps_since_last_rezone >= params.max_steps_since_last_rezone;
  const bool quality_interval_suppressed =
      quality_latched &&
      critical_reasons == kAleMotionTriggerReasonNone &&
      params.min_steps_between_quality_rezones > 0 &&
      steps_since_last_rezone >= 0 &&
      steps_since_last_rezone < params.min_steps_between_quality_rezones;

  std::uint32_t reason_mask = kAleMotionTriggerReasonNone;
  if (quality_latched && !quality_interval_suppressed) {
    reason_mask |= latched_reasons != kAleMotionTriggerReasonNone
                       ? latched_reasons
                       : kAleMotionTriggerReasonHysteresisLatch;
    if (on_reasons == kAleMotionTriggerReasonNone) {
      reason_mask |= kAleMotionTriggerReasonHysteresisLatch;
    }
  }
  if (critical_reasons != kAleMotionTriggerReasonNone) {
    reason_mask |= critical_reasons | kAleMotionTriggerReasonCriticalQuality;
  }
  if (cap_due) {
    reason_mask |= kAleMotionTriggerReasonStepCap;
  }

  double strength = 0.0;
  const std::uint32_t non_cap_reasons =
      reason_mask & ~static_cast<std::uint32_t>(kAleMotionTriggerReasonStepCap);
  if (non_cap_reasons != kAleMotionTriggerReasonNone) {
    const double severity =
        ale_motion_trigger_quality_severity(metrics, params);
    strength = detail::quality_strength_from_severity(params, severity);
  }
  if (cap_due) {
    strength = detail::max_value(
        strength, detail::sanitize_strength(params.cap_only_strength));
  }

  decision.should_rezone = reason_mask != kAleMotionTriggerReasonNone;
  decision.rezone_strength =
      decision.should_rezone ? detail::sanitize_strength(strength) : 0.0;
  decision.reason_mask = reason_mask;
  decision.next_state.quality_latched = quality_latched;
  decision.next_state.latched_reason_mask = quality_latched
                                                ? latched_reasons
                                                : kAleMotionTriggerReasonNone;
  decision.next_state.suppressed_quality_steps =
      quality_interval_suppressed
          ? previous_state.suppressed_quality_steps + 1
          : 0;
  decision.next_state.last_strength = decision.rezone_strength;
  return decision;
}

}  // namespace tenryu::hydro::ale_motion
