#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "core/config.hpp"

#if defined(__CUDACC__)
#define TENRYU_ALE_MONITOR_HOST_DEVICE __host__ __device__
#else
#define TENRYU_ALE_MONITOR_HOST_DEVICE
#endif

namespace tenryu::core {
struct State;
}

namespace tenryu::hydro {

struct AleMonitorTrackedCellResult {
  int cell = -1;
  double q = -1.0;
  double r_z = -1.0;
  double h_metric = -1.0;
  double u_theta = 0.0;
  double s_u_theta = 0.0;
  double alt_min = 0.0;
};

struct AleMonitorResult {
  double q_min = 1.0;
  double h_min = std::numeric_limits<double>::infinity();
  int q_min_cell = -1;
  int h_min_cell = -1;
  double r_z_max = -1.0;
  int r_z_max_cell = -1;
  double r_z_max_u_theta = 0.0;
  double r_z_max_s_u_theta = 0.0;
  double h_metric_max = -1.0;
  int h_metric_max_cell = -1;
  double h_metric_max_u_theta = 0.0;
  double h_metric_max_s_u_theta = 0.0;
  std::vector<AleMonitorTrackedCellResult> tracked_cells;
  int evaluated_cells = 0;
  bool valid = false;
};

namespace detail {

TENRYU_ALE_MONITOR_HOST_DEVICE inline double ale_monitor_corner_quality(
    const double e1_r,
    const double e1_z,
    const double e2_r,
    const double e2_z,
    const double orientation) {
  const double denominator = e1_r * e1_r + e1_z * e1_z +
                             e2_r * e2_r + e2_z * e2_z;
  if (!(denominator > 0.0) || !std::isfinite(denominator)) {
    return 0.0;
  }
  const double jacobian = orientation * (e1_r * e2_z - e1_z * e2_r);
  return 2.0 * jacobian / denominator;
}

}  // namespace detail

// Read-only evaluation over the monitor region: all cells with
// id < mesh shell_cell_offset (core+bridge), plus the innermost
// `shell_rows` structured shell rows (row 0 at the seam). Uses the current
// node coordinates/velocities and the healthy-reference node snapshot for
// normalization. dt_acoustic_now is the current acoustic dt.
// Deterministic: fixed two-pass reduction.
AleMonitorResult evaluate_ale_monitor(const core::State& state,
                                      const core::Config& cfg,
                                      double dt_acoustic_now);

// Process-wide, environment-only gate for the fold-mode shadow diagnostics.
bool fold_diag_enabled();

// Capture/refresh the healthy-reference node snapshot and, when fold-mode
// diagnostics are enabled, the reference corner-subvolume fractions.
void capture_ale_monitor_reference(core::State& state, bool post_morph);

enum class AleMonitorState {
  kOff = 0,
  kWarning = 1,
  kSoft = 2,
  kHard = 3,
  kRecovery = 4,
};

class AleStateMachine {
 public:
  explicit AleStateMachine(
      const core::Config::NumericsConfig::AleConfig::RuntimeControllerConfig&
          cfg)
      : q_soft_(cfg.q_soft),
        q_hard_(cfg.q_hard),
        q_recover_(cfg.q_recover),
        h_soft_(cfg.h_soft),
        h_hard_(cfg.h_hard),
        h_recover_(cfg.h_recover),
        soft_persistence_(cfg.soft_persistence),
        recover_checks_(cfg.recover_checks) {}

  AleMonitorState update(const double q_min, const double h_min) {
    if (q_min < q_soft_) {
      ++q_streak_;
    } else {
      q_streak_ = 0;
    }

    const bool healthy = q_min > q_recover_ && h_min > h_recover_;
    if (healthy) {
      ++healthy_streak_;
    } else {
      healthy_streak_ = 0;
    }

    if (q_min < q_hard_ || h_min < h_hard_) {
      state_ = AleMonitorState::kHard;
      return state_;
    }

    if ((state_ == AleMonitorState::kOff ||
         state_ == AleMonitorState::kWarning ||
         state_ == AleMonitorState::kRecovery) &&
        (h_min < h_soft_ || q_streak_ >= soft_persistence_)) {
      state_ = AleMonitorState::kSoft;
      return state_;
    }

    if ((state_ == AleMonitorState::kSoft ||
         state_ == AleMonitorState::kHard) &&
        healthy_streak_ >= recover_checks_) {
      state_ = AleMonitorState::kRecovery;
      return state_;
    }

    if (state_ == AleMonitorState::kRecovery &&
        healthy_streak_ >= 2 * recover_checks_) {
      state_ = AleMonitorState::kOff;
      return state_;
    }

    if (state_ == AleMonitorState::kRecovery && healthy_streak_ == 0) {
      return state_;
    }

    const double q_warn = std::min(1.0, q_soft_ + 0.13);
    const double h_warn = h_soft_ + 4.0;
    if (state_ == AleMonitorState::kOff &&
        (q_min < q_warn || h_min < h_warn)) {
      state_ = AleMonitorState::kWarning;
      return state_;
    }

    if (state_ == AleMonitorState::kWarning && q_min >= q_warn &&
        h_min >= h_warn) {
      state_ = AleMonitorState::kOff;
    }
    return state_;
  }

  [[nodiscard]] AleMonitorState state() const { return state_; }

 private:
  double q_soft_ = 0.42;
  double q_hard_ = 0.22;
  double q_recover_ = 0.60;
  double h_soft_ = 8.0;
  double h_hard_ = 4.0;
  double h_recover_ = 12.0;
  int soft_persistence_ = 2;
  int recover_checks_ = 3;
  int q_streak_ = 0;
  int healthy_streak_ = 0;
  AleMonitorState state_ = AleMonitorState::kOff;
};

}  // namespace tenryu::hydro

#undef TENRYU_ALE_MONITOR_HOST_DEVICE
