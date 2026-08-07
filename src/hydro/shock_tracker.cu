#include "hydro/shock_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tenryu::hydro {
namespace {

constexpr double kEps = 1.0e-30;
constexpr double kShockQFrac = 0.01;
constexpr double kPressureJumpThreshold = 0.02;
constexpr double kDensityJumpThreshold = 0.01;
constexpr double kTieFrac = 0.05;
constexpr int kBounceWarmupSteps = 50;
constexpr double kBounceVelocitySentinel = 1.0e5;

bool active_material_cell(const core::State& state, const int i) {
  if (!state.hydro_active.empty() && state.hydro_active[static_cast<std::size_t>(i)] == 0) {
    return false;
  }
  if (!state.cell_is_void.empty() && state.cell_is_void[static_cast<std::size_t>(i)] != 0) {
    return false;
  }
  return true;
}

double relative_jump(const std::vector<double>& a, const int i, const int j) {
  const double ai = a[static_cast<std::size_t>(i)];
  const double aj = a[static_cast<std::size_t>(j)];
  return std::abs(ai - aj) / std::max(std::max(std::abs(ai), std::abs(aj)), kEps);
}

double latch_initial_radius(core::State& state, const std::vector<double>& node_r,
                            const int n_cells) {
  if (state.adaptive_av_r0 > 0.0) {
    return state.adaptive_av_r0;
  }

  double r0 = 0.0;
  for (int i = 0; i < n_cells; ++i) {
    if (active_material_cell(state, i)) {
      r0 = std::max(r0, node_r[static_cast<std::size_t>(i + 1)]);
    }
  }
  if (!(r0 > 0.0) && !node_r.empty()) {
    r0 = node_r.back();
  }
  state.adaptive_av_r0 = r0;
  return r0;
}

}  // namespace

LeadingShockState update_leading_shock_tracker_1d(
    core::State& state,
    const core::Config& cfg,
    const core::CellField1D& q_probe,
    const core::CellField1D& div_u_probe,
    const double t,
    const double dt) {
  (void)cfg;
  (void)t;

  LeadingShockState out;
  const int n_cells = static_cast<int>(q_probe.size());
  if (n_cells <= 0 || div_u_probe.size() != q_probe.size() ||
      state.x_r.size() != static_cast<std::size_t>(n_cells + 1) ||
      state.v_r.size() != state.x_r.size() || state.rho.size() != q_probe.size() ||
      state.Pe.size() != q_probe.size() || state.Pi.size() != q_probe.size()) {
    state.adaptive_av_tracker_valid = false;
    out.bounce_seen = state.adaptive_av_bounce_seen;
    return out;
  }

  std::vector<double> q;
  std::vector<double> div_u;
  std::vector<double> rho;
  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> node_r;
  std::vector<double> node_u;
  q_probe.copy_to_host(q);
  div_u_probe.copy_to_host(div_u);
  state.rho.copy_to_host(rho);
  state.Pe.copy_to_host(Pe);
  state.Pi.copy_to_host(Pi);
  state.x_r.copy_to_host(node_r);
  state.v_r.copy_to_host(node_u);

  const double r0 = latch_initial_radius(state, node_r, n_cells);
  double q_max = 0.0;
  double rho_max = 0.0;
  for (int i = 0; i < n_cells; ++i) {
    if (!active_material_cell(state, i)) {
      continue;
    }
    q_max = std::max(q_max, q[static_cast<std::size_t>(i)]);
    rho_max = std::max(rho_max, rho[static_cast<std::size_t>(i)]);
  }
  if (!(q_max > 0.0)) {
    state.adaptive_av_tracker_valid = false;
    out.bounce_seen = state.adaptive_av_bounce_seen;
    return out;
  }

  std::vector<double> pressure(static_cast<std::size_t>(n_cells), 0.0);
  for (int i = 0; i < n_cells; ++i) {
    pressure[static_cast<std::size_t>(i)] =
        Pe[static_cast<std::size_t>(i)] + Pi[static_cast<std::size_t>(i)];
  }

  std::vector<unsigned char> shock(static_cast<std::size_t>(n_cells), 0u);
  const double q_threshold = kShockQFrac * q_max;
  for (int i = 0; i < n_cells; ++i) {
    if (!active_material_cell(state, i) || !(div_u[static_cast<std::size_t>(i)] < 0.0) ||
        !(q[static_cast<std::size_t>(i)] > q_threshold)) {
      continue;
    }
    double p_jump = 0.0;
    double rho_jump = 0.0;
    if (i > 0 && active_material_cell(state, i - 1)) {
      p_jump = std::max(p_jump, relative_jump(pressure, i, i - 1));
      rho_jump = std::max(rho_jump, relative_jump(rho, i, i - 1));
    }
    if (i + 1 < n_cells && active_material_cell(state, i + 1)) {
      p_jump = std::max(p_jump, relative_jump(pressure, i, i + 1));
      rho_jump = std::max(rho_jump, relative_jump(rho, i, i + 1));
    }
    if (p_jump > kPressureJumpThreshold || rho_jump > kDensityJumpThreshold) {
      shock[static_cast<std::size_t>(i)] = 1u;
    }
  }

  double shell_div_sum = 0.0;
  int shell_div_count = 0;
  const double shell_rho_threshold = 0.1 * rho_max;
  for (int i = 0; i < n_cells; ++i) {
    if (active_material_cell(state, i) &&
        rho[static_cast<std::size_t>(i)] > shell_rho_threshold) {
      shell_div_sum += div_u[static_cast<std::size_t>(i)];
      ++shell_div_count;
    }
  }
  out.shell_mean_div_u =
      (shell_div_count > 0) ? shell_div_sum / static_cast<double>(shell_div_count) : 0.0;

  for (int i = 0; i < n_cells;) {
    if (shock[static_cast<std::size_t>(i)] == 0u) {
      ++i;
      continue;
    }
    const int i0 = i;
    while (i < n_cells && shock[static_cast<std::size_t>(i)] != 0u) {
      ++i;
    }
    const int i1 = i - 1;

    double q_sum = 0.0;
    double r_sum = 0.0;
    double u_sum = 0.0;
    double dr_sum = 0.0;
    for (int c = i0; c <= i1; ++c) {
      const double qc = std::max(q[static_cast<std::size_t>(c)], 0.0);
      const double dr = std::max(node_r[static_cast<std::size_t>(c + 1)] -
                                     node_r[static_cast<std::size_t>(c)],
                                 0.0);
      const double rc = 0.5 * (node_r[static_cast<std::size_t>(c)] +
                               node_r[static_cast<std::size_t>(c + 1)]);
      const double uc = 0.5 * (node_u[static_cast<std::size_t>(c)] +
                               node_u[static_cast<std::size_t>(c + 1)]);
      q_sum += qc;
      r_sum += qc * rc;
      u_sum += qc * uc;
      dr_sum += qc * dr;
    }
    if (!(q_sum > 0.0)) {
      continue;
    }

    LeadingShockState candidate;
    candidate.valid = true;
    candidate.i0 = i0;
    candidate.i1 = i1;
    candidate.q_sum = q_sum;
    candidate.r_s = r_sum / q_sum;
    candidate.u_s = u_sum / q_sum;
    candidate.dr_s = std::max(dr_sum / q_sum, kEps);
    candidate.shell_mean_div_u = out.shell_mean_div_u;

    if (state.adaptive_av_tracker_valid && dt > 0.0) {
      candidate.u_s = (candidate.r_s - state.adaptive_av_last_rs) / dt;
    }

    bool take = !out.valid;
    if (out.valid) {
      if (candidate.q_sum > out.q_sum * (1.0 + kTieFrac)) {
        take = true;
      } else if (candidate.q_sum >= out.q_sum * (1.0 - kTieFrac)) {
        take = state.adaptive_av_bounce_seen ? (candidate.r_s > out.r_s)
                                             : (candidate.r_s < out.r_s);
      }
    }
    if (take) {
      out = candidate;
    }
  }

  if (out.valid) {
    bool bounce_seen = state.adaptive_av_bounce_seen;
    const bool had_valid = state.adaptive_av_tracker_valid;
    if (!bounce_seen && had_valid &&
        state.adaptive_av_tracker_steps >= kBounceWarmupSteps) {
      state.adaptive_av_rs_min = std::min(state.adaptive_av_rs_min, out.r_s);
      const bool rs_near_center = out.r_s < 0.02 * r0;
      const bool deep_compression = state.adaptive_av_rs_min < 0.3 * r0;
      const bool clear_rebound =
          out.r_s > 1.2 * state.adaptive_av_rs_min &&
          out.u_s > kBounceVelocitySentinel;
      if (rs_near_center || (deep_compression && clear_rebound)) {
        bounce_seen = true;
      }
    }
    state.adaptive_av_bounce_seen = bounce_seen;
    state.adaptive_av_last_rs = out.r_s;
    state.adaptive_av_last_us = out.u_s;
    state.adaptive_av_tracker_valid = true;
    ++state.adaptive_av_tracker_steps;
  } else {
    state.adaptive_av_tracker_valid = false;
  }

  out.bounce_seen = state.adaptive_av_bounce_seen;
  return out;
}

}  // namespace tenryu::hydro
