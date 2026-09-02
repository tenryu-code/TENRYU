#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "hydro/ale_state_monitor.hpp"

namespace tenryu::core {
struct Config;
struct State;
template <typename T>
class DeviceArray;
}  // namespace tenryu::core

namespace tenryu::hydro {

struct RuntimeAleEscalation {
  double cap_scale = 1.0;
  int sweep_scale = 1;
  bool force_hard = false;
};

struct RuntimeAleTargetStats {
  double max_displacement = 0.0;   // after caps [cm]
  int capped_nodes = 0;            // nodes clamped by the total cap
  int normal_capped_nodes = 0;     // nodes clamped by the shock-normal cap
  int half_plane_clamped_nodes = 0;
  int cap_projected_nodes = 0;
  int ordinary_projected_nodes = 0;
  int cap_infeasible_events = 0;
  int patch_escalations = 0;
  int patch_nodes_max = 0;
  int patch_oversize_skips = 0;
  int mandate_nodes = 0;
  bool valid = false;
};

struct RuntimeAleEventResult {
  bool attempted = false;
  bool succeeded = false;
  bool rolled_back = false;
  double lambda_accepted = 0.0;
  int linesearch_iters = 0;
  int last_reject_reason = 0;  // 0=none, 1=corner_j, 2=admissibility table, 3=other
  int last_reject_cell = -1;
  double max_displacement = 0.0;
  int capped_nodes = 0;
  int half_plane_clamped_nodes = 0;
  int cap_projected_nodes = 0;
  int ordinary_projected_nodes = 0;
  int cap_infeasible_events = 0;
  int patch_escalations = 0;
  int patch_nodes_max = 0;
  int patch_oversize_skips = 0;
  int mandate_nodes = 0;
  double corner_mass_cell_residual = 0.0;
};

// Returns the required cadence for a monitor state (0 = no events).
inline int runtime_ale_cadence(const AleMonitorState state,
                               const int soft,
                               const int hard,
                               const int recovery) {
  switch (state) {
    case AleMonitorState::kSoft:
      return soft;
    case AleMonitorState::kHard:
      return hard;
    case AleMonitorState::kRecovery:
      return recovery;
    case AleMonitorState::kOff:
    case AleMonitorState::kWarning:
      return 0;
  }
  return 0;
}

// Return the maximum relative corner-mass closure residual over active cells.
inline void corner_mass_cell_closure_residual(
    const double* const corner_mass,
    const double* const cell_mass,
    const std::uint8_t* const active_cell,
    const int n_cells,
    double* const residual_out) {
  double max_residual = 0.0;
  constexpr double tiny = std::numeric_limits<double>::min();
  for (int cell = 0; cell < n_cells; ++cell) {
    if (active_cell[cell] == 0U) {
      continue;
    }
    const std::size_t base = static_cast<std::size_t>(cell) * 4U;
    const double corner_sum = corner_mass[base + 0U] + corner_mass[base + 1U] +
                              corner_mass[base + 2U] + corner_mass[base + 3U];
    const double residual =
        std::abs(corner_sum - cell_mass[cell]) / std::max(cell_mass[cell], tiny);
    max_residual = std::isfinite(residual)
                       ? std::max(max_residual, residual)
                       : std::numeric_limits<double>::infinity();
  }
  *residual_out = max_residual;
}

// Build the blended runtime-ALE target for the CURRENT state:
//   x_tar = x_L + omega_p * [(1-beta_M)(x_W - x_L) + beta_M (x_M - x_L)]
// with x_W = virtual multiblock Winslow smooth of x_L (mandate-masked), x_M
// = bridge-column monitor equidistribution + weak equal-angle target (falls
// back to x_W outside its domain), omega_p from local corner quality, and the
// displacement caps applied after the blend. Axis-cap nodes are excluded from
// x_W/x_M and use the W7b fixed-ray common radial ladder; their feasible
// projection follows the displacement caps. Ordinary mandate nodes use the
// two-dimensional local feasible-set projection at the same stage.
// front_s_per_column: nullable HOST array of ntheta doubles (the L<=4-filtered
// front radius per theta cell column, cm); nullptr disables the front monitor
// term. shock_normal_dir: nullable pair of HOST arrays (per-cell unit grad-p
// direction r/z) enabling the normal cap; nullptr applies the total cap only.
// Restores state coordinates/geometry exactly before returning (the virtual
// smoother runs on the live arrays with save/restore).
RuntimeAleTargetStats build_runtime_ale_target(
    core::State& state, const core::Config& cfg, bool hard_mode,
    const RuntimeAleEscalation& escalation,
    const double* front_s_per_column,
    const double* shock_normal_dir_r, const double* shock_normal_dir_z,
    core::DeviceArray<double>& target_x_r,
    core::DeviceArray<double>& target_x_z);

inline RuntimeAleTargetStats build_runtime_ale_target(
    core::State& state, const core::Config& cfg, const bool hard_mode,
    const double* const front_s_per_column,
    const double* const shock_normal_dir_r,
    const double* const shock_normal_dir_z,
    core::DeviceArray<double>& target_x_r,
    core::DeviceArray<double>& target_x_z) {
  return build_runtime_ale_target(
      state, cfg, hard_mode, RuntimeAleEscalation{}, front_s_per_column,
      shock_normal_dir_r, shock_normal_dir_z, target_x_r, target_x_z);
}

// Build the W3a target and execute one transactional rezone/remap through the
// reference-barrier engine. The runtime mandate activates core+bridge cells
// and freezes every node outside the W3a node mandate. front_s_per_column is
// the nullable W3a host array. A target below the null-event displacement
// threshold returns attempted=false without invoking the transaction.
RuntimeAleEventResult run_runtime_ale_event(
    core::State& state, const core::Config& cfg, bool hard_mode,
    const RuntimeAleEscalation& escalation,
    const double* front_s_per_column);

inline RuntimeAleEventResult run_runtime_ale_event(
    core::State& state, const core::Config& cfg, const bool hard_mode,
    const double* const front_s_per_column) {
  return run_runtime_ale_event(state, cfg, hard_mode, RuntimeAleEscalation{},
                               front_s_per_column);
}

namespace detail {

// Equidistribute n_rows interior node radii on [s_lo, s_hi] against the
// composite monitor built from base spacing h0(s) (piecewise from the input
// ladder), a mass term (rho_bins over the column, nullable), and a front
// Gaussian centered at s_front (s_front <= 0 disables). Pins the LAST
// interval to exactly h_seam (rows n_rows-1 and n_rows fixed per the seam
// contract) and enforces the adjacent-ratio limiter g_max by a single
// deterministic forward-backward sweep. Inputs/outputs are node radii arrays
// of length n_rows+1 (s_out[0]=s_lo, s_out[n_rows]=s_hi preserved).
void equidistribute_column(const double* s_in, int n_rows,
                           const double* rho_bins, int n_bins,
                           double s_front, double front_width,
                           double beta_mass, double beta_front,
                           double g_max, double h_seam, double* s_out);

// Clamp displacement (dr,dz) at a node with local length h_p: total cap
// cap_frac*h_p; when has_normal, also cap the component along (n_r,n_z) to
// cap_normal_frac*h_p (tangential keeps the total cap). Returns the
// clamped (dr,dz) through the pointers and true when any clamp fired.
bool clamp_displacement(double* dr, double* dz, double h_p,
                        double cap_frac, double cap_normal_frac,
                        bool has_normal, double n_r, double n_z);

}  // namespace detail
}  // namespace tenryu::hydro
