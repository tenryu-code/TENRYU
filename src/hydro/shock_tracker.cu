#include "hydro/shock_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

#include <cuda_runtime.h>

#include "core/device_ordered_sum.cuh"
#include "core/device_pack.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kEps = 1.0e-30;
constexpr double kShockQFrac = 0.01;
constexpr double kPressureJumpThreshold = 0.02;
constexpr double kDensityJumpThreshold = 0.01;
constexpr double kTieFrac = 0.05;
constexpr int kBounceWarmupSteps = 50;
constexpr double kBounceVelocitySentinel = 1.0e5;

// The scan of the tracker on the device (the host loops it replaces, value for value): the
// maxima over the active cells, the shock flags, the shell's mean divergence, the clusters of
// contiguous shock cells with their q-weighted sums in cell order, and the leading cluster chosen
// by the host rule (first strongest, ties by radius).
constexpr int kTrackBlock = 256;

struct TrackerScan {
  int valid = 0;
  int i0 = -1;
  int i1 = -1;
  double q_sum = 0.0;
  double r_s = 0.0;
  double u_s = 0.0;
  double dr_s = 0.0;
  double shell_mean_div_u = 0.0;
  double q_max = 0.0;
  double rho_max = 0.0;
  double r_active_max = 0.0;  // the largest outer node radius of an active cell
  double r_last = 0.0;        // the last node radius
};

// std::max(a, b) and std::min(a, b): b when it compares larger (smaller), else a.
__device__ inline double std_max(const double a, const double b) { return (a < b) ? b : a; }

__device__ inline bool active_cell(const std::int8_t* __restrict__ active,
                                   const std::uint8_t* __restrict__ is_void, const int i) {
  if (active != nullptr && active[i] == 0) {
    return false;
  }
  if (is_void != nullptr && is_void[i] != 0) {
    return false;
  }
  return true;
}

__device__ inline double relative_jump(const double ai, const double aj) {
  return fabs(ai - aj) / std_max(std_max(fabs(ai), fabs(aj)), kEps);
}

// One block: q_max, rho_max (from 0) and the outer radius of the active cells (from 0); the
// maxima do not depend on the order.
__global__ void tracker_maxima_kernel(const double* __restrict__ q,
                                      const double* __restrict__ rho,
                                      const double* __restrict__ node_r,
                                      const std::int8_t* __restrict__ active,
                                      const std::uint8_t* __restrict__ is_void, const int n,
                                      TrackerScan* __restrict__ scan) {
  __shared__ double sh[3][kTrackBlock];
  const int t = static_cast<int>(threadIdx.x);
  double q_max = 0.0;
  double rho_max = 0.0;
  double r_max = 0.0;
  for (int i = t; i < n; i += kTrackBlock) {
    if (active_cell(active, is_void, i)) {
      q_max = std_max(q_max, q[i]);
      rho_max = std_max(rho_max, rho[i]);
      r_max = std_max(r_max, node_r[i + 1]);
    }
  }
  sh[0][t] = q_max;
  sh[1][t] = rho_max;
  sh[2][t] = r_max;
  __syncthreads();
  for (int offset = kTrackBlock / 2; offset > 0; offset >>= 1) {
    if (t < offset) {
      for (int k = 0; k < 3; ++k) {
        sh[k][t] = std_max(sh[k][t], sh[k][t + offset]);
      }
    }
    __syncthreads();
  }
  if (t == 0) {
    TrackerScan out;
    out.q_max = sh[0][0];
    out.rho_max = sh[1][0];
    out.r_active_max = sh[2][0];
    out.r_last = node_r[n];
    *scan = out;
  }
}

// Per cell: the shock flag (active, compressing, q above 1 % of q_max, a pressure or density
// jump to an active neighbour) and the shell term (div u of an active cell denser than 10 % of
// rho_max, else 0) with its flag.
__global__ void tracker_flags_kernel(const double* __restrict__ q,
                                     const double* __restrict__ div_u,
                                     const double* __restrict__ rho,
                                     const double* __restrict__ Pe, const double* __restrict__ Pi,
                                     const std::int8_t* __restrict__ active,
                                     const std::uint8_t* __restrict__ is_void, const int n,
                                     const TrackerScan* __restrict__ scan,
                                     std::uint8_t* __restrict__ shock,
                                     double* __restrict__ shell_term,
                                     int* __restrict__ shell_flag) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  std::uint8_t is_shock = 0U;
  double term = 0.0;
  int in_shell = 0;
  if (active_cell(active, is_void, i)) {
    const double rho_threshold = 0.1 * scan->rho_max;
    if (rho[i] > rho_threshold) {
      term = div_u[i];
      in_shell = 1;
    }
    const double q_threshold = kShockQFrac * scan->q_max;
    if ((div_u[i] < 0.0) && (q[i] > q_threshold)) {
      double p_jump = 0.0;
      double rho_jump = 0.0;
      const double p_i = Pe[i] + Pi[i];
      if (i > 0 && active_cell(active, is_void, i - 1)) {
        p_jump = std_max(p_jump, relative_jump(p_i, Pe[i - 1] + Pi[i - 1]));
        rho_jump = std_max(rho_jump, relative_jump(rho[i], rho[i - 1]));
      }
      if (i + 1 < n && active_cell(active, is_void, i + 1)) {
        p_jump = std_max(p_jump, relative_jump(p_i, Pe[i + 1] + Pi[i + 1]));
        rho_jump = std_max(rho_jump, relative_jump(rho[i], rho[i + 1]));
      }
      if (p_jump > kPressureJumpThreshold || rho_jump > kDensityJumpThreshold) {
        is_shock = 1U;
      }
    }
  }
  shock[i] = is_shock;
  shell_term[i] = term;
  shell_flag[i] = in_shell;
}

// One block: the shell's sum in cell order (zero terms skipped, which leaves a sum from +0
// unchanged) and count; the clusters of contiguous shock cells in order, each one's sums added in
// cell order by one thread (every product rounded on its own); then thread 0 picks the leading
// cluster as the host loop did.
__global__ void tracker_clusters_kernel(const double* __restrict__ q,
                                        const double* __restrict__ node_r,
                                        const double* __restrict__ node_u, const int n,
                                        const std::uint8_t* __restrict__ shock,
                                        const double* __restrict__ shell_term,
                                        const int* __restrict__ shell_flag,
                                        int* __restrict__ run_begin,
                                        double* __restrict__ run_sums,
                                        int* __restrict__ run_end,
                                        const int tracker_valid, const double last_rs,
                                        const double dt, const int bounce_seen,
                                        TrackerScan* __restrict__ scan) {
  __shared__ double sh_values[kTrackBlock * 4];
  __shared__ int sh_scan[kTrackBlock];
  __shared__ int sh_runs;
  const int t = static_cast<int>(threadIdx.x);
  const double shell_sum = core::device_ordered::block_ordered_sum_nonzero<kTrackBlock, 4>(
      shell_term, n, 0.0, sh_values, sh_scan);
  int count = 0;
  for (int i = t; i < n; i += kTrackBlock) {
    count += shell_flag[i];
  }
  int count_total = 0;
  static_cast<void>(core::device_ordered::block_exclusive_prefix<kTrackBlock>(count, sh_scan,
                                                                             &count_total));
  // the cluster starts in order
  int n_runs = 0;
  for (int chunk = 0; chunk < n; chunk += kTrackBlock) {
    const int i = chunk + t;
    const int is_start =
        (i < n && shock[i] != 0U && (i == 0 || shock[i - 1] == 0U)) ? 1 : 0;
    int total = 0;
    const int at = core::device_ordered::block_exclusive_prefix<kTrackBlock>(is_start, sh_scan,
                                                                             &total);
    if (is_start != 0) {
      run_begin[n_runs + at] = i;
    }
    n_runs += total;
  }
  if (t == 0) {
    sh_runs = n_runs;
  }
  __syncthreads();
  const int runs = sh_runs;
  for (int r = t; r < runs; r += kTrackBlock) {
    const int i0 = run_begin[r];
    int i1 = i0;
    while (i1 + 1 < n && shock[i1 + 1] != 0U) {
      ++i1;
    }
    double q_sum = 0.0;
    double r_sum = 0.0;
    double u_sum = 0.0;
    double dr_sum = 0.0;
    for (int c = i0; c <= i1; ++c) {
      const double qc = std_max(q[c], 0.0);
      const double dr = std_max(node_r[c + 1] - node_r[c], 0.0);
      const double rc = 0.5 * (node_r[c] + node_r[c + 1]);
      const double uc = 0.5 * (node_u[c] + node_u[c + 1]);
      q_sum = __dadd_rn(q_sum, qc);
      r_sum = __dadd_rn(r_sum, __dmul_rn(qc, rc));
      u_sum = __dadd_rn(u_sum, __dmul_rn(qc, uc));
      dr_sum = __dadd_rn(dr_sum, __dmul_rn(qc, dr));
    }
    run_sums[4 * r + 0] = q_sum;
    run_sums[4 * r + 1] = r_sum;
    run_sums[4 * r + 2] = u_sum;
    run_sums[4 * r + 3] = dr_sum;
    run_end[r] = i1;
  }
  __syncthreads();
  if (t == 0) {
    TrackerScan out = *scan;  // the maxima and radii of the first kernel
    out.valid = 0;
    out.shell_mean_div_u =
        (count_total > 0) ? shell_sum / static_cast<double>(count_total) : 0.0;
    for (int r = 0; r < runs; ++r) {
      const double q_sum = run_sums[4 * r + 0];
      if (!(q_sum > 0.0)) {
        continue;
      }
      const double r_s = run_sums[4 * r + 1] / q_sum;
      double u_s = run_sums[4 * r + 2] / q_sum;
      const double dr_s = std_max(run_sums[4 * r + 3] / q_sum, kEps);
      if (tracker_valid != 0 && dt > 0.0) {
        u_s = (r_s - last_rs) / dt;
      }
      bool take = (out.valid == 0);
      if (out.valid != 0) {
        if (q_sum > out.q_sum * (1.0 + kTieFrac)) {
          take = true;
        } else if (q_sum >= out.q_sum * (1.0 - kTieFrac)) {
          take = (bounce_seen != 0) ? (r_s > out.r_s) : (r_s < out.r_s);
        }
      }
      if (take) {
        out.valid = 1;
        out.i0 = run_begin[r];
        out.i1 = run_end[r];
        out.q_sum = q_sum;
        out.r_s = r_s;
        out.u_s = u_s;
        out.dr_s = dr_s;
      }
    }
    *scan = out;
  }
}

void tracker_cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, std::string(message) + ": " + cudaGetErrorString(err));
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

  // The scan on the device; one result packet comes back.
  const std::size_t n = static_cast<std::size_t>(n_cells);
  const std::int8_t* d_active =
      (state.hydro_active.size() == n) ? state.hydro_active_device_ptr() : nullptr;
  const std::uint8_t* d_void =
      state.cell_is_void.empty() ? nullptr : core::device_cell_is_void(state.cell_is_void);
  auto* const d_scan = static_cast<TrackerScan*>(
      core::device_scratch_acquire("shock_tracker:scan", sizeof(TrackerScan)));
  auto* const d_shock = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("shock_tracker:shock", n));
  auto* const d_ints = static_cast<int*>(
      core::device_scratch_acquire("shock_tracker:ints", 3 * n * sizeof(int)));
  int* const d_shell_flag = d_ints;
  int* const d_run_begin = d_ints + n;
  int* const d_run_end = d_ints + 2 * n;
  auto* const d_doubles = static_cast<double*>(
      core::device_scratch_acquire("shock_tracker:doubles", 5 * n * sizeof(double)));
  double* const d_shell_term = d_doubles;
  double* const d_run_sums = d_doubles + n;
  tracker_maxima_kernel<<<1, kTrackBlock>>>(q_probe.data(), state.rho.data(), state.x_r.data(),
                                            d_active, d_void, n_cells, d_scan);
  tracker_cuda_check(cudaGetLastError(), "shock tracker maxima launch failed");
  tracker_flags_kernel<<<(n_cells + kTrackBlock - 1) / kTrackBlock, kTrackBlock>>>(
      q_probe.data(), div_u_probe.data(), state.rho.data(), state.Pe.data(), state.Pi.data(),
      d_active, d_void, n_cells, d_scan, d_shock, d_shell_term, d_shell_flag);
  tracker_cuda_check(cudaGetLastError(), "shock tracker flags launch failed");
  tracker_clusters_kernel<<<1, kTrackBlock>>>(
      q_probe.data(), state.x_r.data(), state.v_r.data(), n_cells, d_shock, d_shell_term,
      d_shell_flag, d_run_begin, d_run_sums, d_run_end,
      state.adaptive_av_tracker_valid ? 1 : 0, state.adaptive_av_last_rs, dt,
      state.adaptive_av_bounce_seen ? 1 : 0, d_scan);
  tracker_cuda_check(cudaGetLastError(), "shock tracker clusters launch failed");
  TrackerScan scan;
  tracker_cuda_check(cudaMemcpy(&scan, d_scan, sizeof(scan), cudaMemcpyDeviceToHost),
                     "shock tracker result copy failed");

  // The initial radius, latched on the first call: the outer radius of the active cells, else
  // the last node's.
  if (!(state.adaptive_av_r0 > 0.0)) {
    double r0 = scan.r_active_max;
    if (!(r0 > 0.0)) {
      r0 = scan.r_last;
    }
    state.adaptive_av_r0 = r0;
  }
  const double r0 = state.adaptive_av_r0;
  if (!(scan.q_max > 0.0)) {
    state.adaptive_av_tracker_valid = false;
    out.bounce_seen = state.adaptive_av_bounce_seen;
    return out;
  }
  out.shell_mean_div_u = scan.shell_mean_div_u;
  if (scan.valid != 0) {
    out.valid = true;
    out.i0 = scan.i0;
    out.i1 = scan.i1;
    out.q_sum = scan.q_sum;
    out.r_s = scan.r_s;
    out.u_s = scan.u_s;
    out.dr_s = scan.dr_s;
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
