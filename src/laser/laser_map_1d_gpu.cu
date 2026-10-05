#include "laser/laser_map_1d_gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "core/constants.hpp"
#include "core/device_pack.hpp"
#include "core/error.hpp"

// Compiled with -fmad=false (src/laser/CMakeLists.txt): the arithmetic repeats laser_mesh.cu
// (compute_dynamic_mesh_params_1d, build_graded_nodes_1d, geometric_widths_from_interface,
// estimate_critical_surface_1d, average_outer_cells, compute_ghost_sound_speed_cm_s and the scalar
// part of map_from_hydro_1d), deposit_transfer.cu (compute_cell_n_hat_approx,
// estimate_critical_surface_1d, find_allowed_supercritical_cell_1d_impl) and the resonance-
// absorption inputs of laser.cu laser_step operation by operation.

namespace tenryu::laser::laser_map_1d {
namespace {

constexpr double kProtonMass = 1.6726219e-24;  // laser_mesh.cu (= core::constants::proton_mass)
constexpr double kGradedCoreRatio = 1.08;
constexpr double kGradedCoronaRatio = 1.05;
constexpr int kFineSideTarget = 64;
constexpr int kGhostAnchorSpan = 3;
constexpr int kThreads = 256;
// Levels of the layout bisection resolved per round: the 2^8 - 1 midpoints of the next eight
// levels are counted in parallel, then the path is followed exactly as the sequential bisection.
constexpr int kBisectLevelsPerRound = 8;
constexpr int kBisectRounds = 5;  // 40 levels, as build_graded_nodes_1d
constexpr int kDoublingProbes = 64;

static_assert(kThreads >= (1 << kBisectLevelsPerRound), "one thread per bisection tree node");
static_assert(kThreads >= kDoublingProbes, "one thread per doubling probe");

inline void check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void ensure(T** ptr, std::size_t* capacity, const std::size_t needed, const char* message) {
  if (needed <= *capacity && *ptr != nullptr) {
    return;
  }
  if (*ptr != nullptr) {
    static_cast<void>(cudaFree(*ptr));
    *ptr = nullptr;
  }
  *capacity = 0;
  if (needed == 0) {
    return;
  }
  check(cudaMalloc(reinterpret_cast<void**>(ptr), needed * sizeof(T)), message);
  *capacity = needed;
}

// std::max, std::min and std::clamp of the host code (NaN and signed-zero behaviour included).
__device__ inline double host_max(const double a, const double b) { return (a < b) ? b : a; }
__device__ inline double host_min(const double a, const double b) { return (b < a) ? b : a; }
__device__ inline double host_clamp(const double v, const double lo, const double hi) {
  return (v < lo) ? lo : ((hi < v) ? hi : v);
}

// ---------------------------------------------------------------------------------------------
// per cell

__global__ void cells_kernel(const int n_cells, const double* __restrict__ rho,
                             const double* __restrict__ zbar, const double* __restrict__ Te,
                             const double* __restrict__ A_eff,
                             const std::uint8_t* __restrict__ mask, const double n_crit,
                             const double n_crit_safe, double* __restrict__ n_hat_layout,
                             double* __restrict__ n_hat_raw, double* __restrict__ n_hat_approx,
                             int* __restrict__ nonfinite_flags) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  // map_from_hydro_1d's checks of every cell, void ones included
  const int flags = (::isfinite(rho[c]) ? 0 : 1) | (::isfinite(zbar[c]) ? 0 : 2) |
                    (::isfinite(Te[c]) ? 0 : 4);
  if (flags != 0) {
    atomicOr(nonfinite_flags, flags);
  }
  if (mask != nullptr && mask[c] != 0U) {
    n_hat_layout[c] = 0.0;
    n_hat_raw[c] = 0.0;
    n_hat_approx[c] = 0.0;
    return;
  }
  const double rho_c = rho[c];
  const double zbar_c = zbar[c];
  const double A_c = A_eff[c];
  const double A_safe = host_max(A_c, 1.0e-30);
  // compute_dynamic_mesh_params_1d (and laser_step's resonance-absorption profile)
  const double n_e = host_max(0.0, rho_c) * host_max(0.0, zbar_c) / (A_safe * kProtonMass);
  n_hat_layout[c] = host_max(0.0, n_e / n_crit_safe);
  // map_from_hydro_1d
  n_hat_raw[c] = host_max(0.0, rho_c) * host_max(0.0, zbar_c) / (A_safe * kProtonMass * n_crit_safe);
  // deposit_transfer.cu compute_cell_n_hat_approx
  n_hat_approx[c] = (!(rho_c > 0.0) || !(zbar_c > 0.0) || !(A_c > 0.0) || !(n_crit > 0.0))
                        ? 0.0
                        : rho_c * zbar_c / (A_c * core::constants::proton_mass * n_crit);
}

// ---------------------------------------------------------------------------------------------
// block reductions (one block)

__device__ int block_max_int(int value, int* sh) {
  const int t = threadIdx.x;
  sh[t] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      sh[t] = ::max(sh[t], sh[t + stride]);
    }
    __syncthreads();
  }
  const int result = sh[0];
  __syncthreads();
  return result;
}

// The minimum of finite positive values (the identity is DBL_MAX, as the host's initial value):
// the order of the comparisons does not change it.
__device__ double block_min_double(double value, double* sh) {
  const int t = threadIdx.x;
  sh[t] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      sh[t] = host_min(sh[t], sh[t + stride]);
    }
    __syncthreads();
  }
  const double result = sh[0];
  __syncthreads();
  return result;
}

// ---------------------------------------------------------------------------------------------
// critical surface

struct CritEstimate {
  int fcrit;
  double r_interp;
};

__device__ double log_interpolated_radius(const double* n_hat, const double* r_edges, const int fcrit,
                                          const double fallback) {
  double r_interp = fallback;
  const int c_hi = fcrit - 1;
  const int c_lo = fcrit;
  const double n_hi = n_hat[c_hi];
  const double n_lo = n_hat[c_lo];
  const double r_hi = 0.5 * (r_edges[c_hi] + r_edges[c_hi + 1]);
  const double r_lo = 0.5 * (r_edges[c_lo] + r_edges[c_lo + 1]);
  if (n_hi > 1.0 && n_lo > 0.0 && n_lo < 1.0 && r_lo > r_hi) {
    const double denom = ::log(n_lo / n_hi);
    if (::isfinite(denom) && ::fabs(denom) > 1.0e-12) {
      const double theta = host_clamp(::log(1.0 / n_hi) / denom, 0.0, 1.0);
      r_interp = host_clamp(r_hi + theta * (r_lo - r_hi), r_hi, r_lo);
    }
  }
  return r_interp;
}

__device__ int first_crossing_face(const double* n_hat, const int n_cells, const int outermost_critical) {
  for (int c = outermost_critical; c < n_cells; ++c) {
    const bool this_supercritical = (n_hat[c] >= 1.0);
    const bool next_subcritical = (c + 1 >= n_cells) || (n_hat[c + 1] < 1.0);
    if (this_supercritical && next_subcritical) {
      return c + 1;
    }
  }
  return -1;
}

// laser_mesh.cu estimate_critical_surface_1d (fcrit may be the outer face n_cells; no
// interpolation there).
__device__ CritEstimate estimate_layout(const double* n_hat, const double* r_edges, const int n_cells,
                                        const int outermost_critical) {
  if (outermost_critical < 0) {
    return CritEstimate{-1, -1.0};
  }
  const int fcrit = first_crossing_face(n_hat, n_cells, outermost_critical);
  if (fcrit < 0 || fcrit >= n_cells + 1) {
    return CritEstimate{-1, -1.0};
  }
  double r_interp = r_edges[fcrit];
  if (fcrit > 0 && fcrit < n_cells) {
    r_interp = log_interpolated_radius(n_hat, r_edges, fcrit, r_interp);
  }
  return CritEstimate{fcrit, r_interp};
}

// deposit_transfer.cu estimate_critical_surface_1d (an interior face only).
__device__ CritEstimate estimate_deposit(const double* n_hat, const double* r_edges, const int n_cells,
                                         const int outermost_critical) {
  if (outermost_critical < 0) {
    return CritEstimate{-1, -1.0};
  }
  const int fcrit = first_crossing_face(n_hat, n_cells, outermost_critical);
  if (fcrit <= 0 || fcrit >= n_cells) {
    return CritEstimate{-1, -1.0};
  }
  return CritEstimate{fcrit, log_interpolated_radius(n_hat, r_edges, fcrit, r_edges[fcrit])};
}

// ---------------------------------------------------------------------------------------------
// graded node layout (build_graded_nodes_1d)

// geometric_widths_from_interface: the number of widths, and with out != nullptr the widths.
__device__ int geometric_widths(const double L, double d0, double g, double* out) {
  if (!(L > 0.0)) {
    return 0;
  }
  d0 = host_max(d0, 1.0e-12);
  g = host_max(g, 1.0001);
  double used = 0.0;
  double d = d0;
  double last = 0.0;
  int n = 0;
  while (used + d < L) {
    if (out != nullptr) {
      out[n] = d;
    }
    last = d;
    ++n;
    used += d;
    d *= g;
  }
  const double tail = L - used;
  if (tail > 0.0) {
    if (n > 0 && tail < 0.35 * last) {
      if (out != nullptr) {
        out[n - 1] += tail;
      }
    } else {
      if (out != nullptr) {
        out[n] = tail;
      }
      ++n;
    }
  }
  return n;
}

// append_uniform_widths: the count and the width.
__device__ int uniform_count(const double L, const double d_target, double* d_out) {
  if (!(L > 0.0)) {
    return 0;
  }
  const int n = ::max(1, static_cast<int>(::ceil(L / host_max(d_target, 1.0e-12))));
  *d_out = L / static_cast<double>(n);
  return n;
}

struct LayoutSpans {
  double dF;
  double R_left;
  double R_right;
};

__device__ LayoutSpans layout_spans(const double s, const double dF_base, const double R_crit_s,
                                    const double R_max_s) {
  const double dF = dF_base * host_max(s, 1.0);
  const double delta_target = kFineSideTarget * dF;
  const double delta_in = host_min(delta_target, R_crit_s);
  const double delta_out = host_min(delta_target, R_max_s - R_crit_s);
  return LayoutSpans{dF, R_crit_s - delta_in, R_crit_s + delta_out};
}

// The cell count of build_for_scale(s).
__device__ int count_for_scale(const double s, const double dF_base, const double R_crit_s,
                               const double R_max_s) {
  const LayoutSpans sp = layout_spans(s, dF_base, R_crit_s, R_max_s);
  double d_unused = 0.0;
  return geometric_widths(sp.R_left, sp.dF, kGradedCoreRatio, nullptr) +
         uniform_count(R_crit_s - sp.R_left, sp.dF, &d_unused) +
         uniform_count(sp.R_right - R_crit_s, sp.dF, &d_unused) +
         geometric_widths(R_max_s - sp.R_right, sp.dF, kGradedCoronaRatio, nullptr);
}

// The nodes of build_for_scale(s) (count cells, checked by the caller); widths is scratch for the
// core widths, which are laid out from the centre outward in reverse.
__device__ void build_for_scale(const double s, const double dF_base, const double R_crit_s,
                                const double R_max_s, double* widths, double* nodes) {
  const LayoutSpans sp = layout_spans(s, dF_base, R_crit_s, R_max_s);
  int k = 0;
  nodes[0] = 0.0;
  const int n_core = geometric_widths(sp.R_left, sp.dF, kGradedCoreRatio, widths);
  for (int i = n_core - 1; i >= 0; --i) {
    nodes[k + 1] = nodes[k] + host_max(widths[i], 1.0e-12);
    ++k;
  }
  double d = 0.0;
  const int n_left = uniform_count(R_crit_s - sp.R_left, sp.dF, &d);
  for (int i = 0; i < n_left; ++i) {
    nodes[k + 1] = nodes[k] + host_max(d, 1.0e-12);
    ++k;
  }
  const int n_right = uniform_count(sp.R_right - R_crit_s, sp.dF, &d);
  for (int i = 0; i < n_right; ++i) {
    nodes[k + 1] = nodes[k] + host_max(d, 1.0e-12);
    ++k;
  }
  // the corona widths in order, accumulated as they are generated (geometric_widths repeated
  // here so that the tail merge adds to the last width before it is accumulated)
  const double L = R_max_s - sp.R_right;
  if (L > 0.0) {
    const double d0 = host_max(sp.dF, 1.0e-12);
    const double g = host_max(kGradedCoronaRatio, 1.0001);
    double used = 0.0;
    double w = d0;
    int n = 0;
    double pending = 0.0;  // the last width, accumulated once the tail is known
    while (used + w < L) {
      if (n > 0) {
        nodes[k + 1] = nodes[k] + host_max(pending, 1.0e-12);
        ++k;
      }
      pending = w;
      ++n;
      used += w;
      w *= g;
    }
    const double tail = L - used;
    if (tail > 0.0 && n > 0 && tail < 0.35 * pending) {
      pending += tail;
      nodes[k + 1] = nodes[k] + host_max(pending, 1.0e-12);
      ++k;
    } else {
      if (n > 0) {
        nodes[k + 1] = nodes[k] + host_max(pending, 1.0e-12);
        ++k;
      }
      if (tail > 0.0) {
        nodes[k + 1] = nodes[k] + host_max(tail, 1.0e-12);
        ++k;
      }
    }
  }
  nodes[k] = R_max_s;
}

// build_graded_nodes_1d with one block; returns the cell count, nodes in `nodes`.
__device__ int graded_nodes(const double R_crit, const double dR_fine, const double R_max,
                            const int nr_max, double* widths, double* nodes) {
  __shared__ double s_lo;
  __shared__ double s_hi;
  __shared__ int s_result;
  __shared__ int s_counts[1 << kBisectLevelsPerRound];
  const int t = threadIdx.x;
  const double R_max_s = host_max(R_max, 1.0e-12);
  const double R_crit_s = host_clamp(R_crit, 0.0, R_max_s);
  const double dF_base = host_max(dR_fine, 1.0e-12);
  const int nr_max_s = ::max(nr_max, 4);

  if (t == 0) {
    const int count1 = count_for_scale(1.0, dF_base, R_crit_s, R_max_s);
    s_result = (count1 >= 4 ? count1 : 4) <= nr_max_s ? 1 : 0;  // enforce_min_cells(first) fits
    s_lo = 1.0;
    s_hi = 2.0;
  }
  __syncthreads();
  double scale = 1.0;
  if (s_result == 0) {
    // while (count(hi) > nr_max_s) hi *= 2: probe hi = 2^k, k = 1 .. kDoublingProbes
    if (t < kDoublingProbes) {
      const double hi_k = ::ldexp(1.0, t + 1);
      s_counts[t] = count_for_scale(hi_k, dF_base, R_crit_s, R_max_s) > nr_max_s ? 1 : 0;
    }
    __syncthreads();
    if (t == 0) {
      int k = 0;
      while (k < kDoublingProbes && s_counts[k] != 0) {
        ++k;
      }
      double hi = ::ldexp(1.0, k + 1);
      if (k == kDoublingProbes) {
        hi = ::ldexp(1.0, kDoublingProbes);
        while (count_for_scale(hi, dF_base, R_crit_s, R_max_s) > nr_max_s) {
          hi *= 2.0;
        }
      }
      s_hi = hi;
    }
    __syncthreads();
    // 40 bisection levels, eight per round
    for (int round = 0; round < kBisectRounds; ++round) {
      const double lo0 = s_lo;
      const double hi0 = s_hi;
      if (t >= 1 && t < (1 << kBisectLevelsPerRound)) {
        // tree node t (heap order): its interval follows the path bits below the leading one
        int depth = 0;
        while ((t >> (depth + 1)) != 0) {
          ++depth;
        }
        double lo = lo0;
        double hi = hi0;
        for (int level = depth - 1; level >= 0; --level) {
          const double mid = 0.5 * (lo + hi);
          if (((t >> level) & 1) != 0) {
            lo = mid;  // count(mid) > nr_max_s on the path
          } else {
            hi = mid;
          }
        }
        const double mid = 0.5 * (lo + hi);
        s_counts[t] = count_for_scale(mid, dF_base, R_crit_s, R_max_s) > nr_max_s ? 1 : 0;
      }
      __syncthreads();
      if (t == 0) {
        double lo = lo0;
        double hi = hi0;
        int node = 1;
        for (int level = 0; level < kBisectLevelsPerRound; ++level) {
          const double mid = 0.5 * (lo + hi);
          const int above = s_counts[node];
          if (above != 0) {
            lo = mid;
          } else {
            hi = mid;
          }
          node = 2 * node + above;
        }
        s_lo = lo;
        s_hi = hi;
      }
      __syncthreads();
    }
    scale = s_hi;
  }
  __syncthreads();
  if (t == 0) {
    const int count = count_for_scale(scale, dF_base, R_crit_s, R_max_s);
    if (count >= 4) {
      build_for_scale(scale, dF_base, R_crit_s, R_max_s, widths, nodes);
      s_result = count;
    } else {
      // enforce_min_cells: build_uniform_nodes(0, R_max_s, 4)
      const double dx = (R_max_s - 0.0) / 4.0;
      for (int i = 0; i <= 4; ++i) {
        nodes[i] = 0.0 + dx * static_cast<double>(i);
      }
      s_result = 4;
    }
  }
  __syncthreads();
  const int result = s_result;
  __syncthreads();
  return result;
}

// ---------------------------------------------------------------------------------------------
// the map

// average_outer_cells (an empty void mask on the host returned the fallback).
__device__ double average_outer(const double* values, const std::uint8_t* mask, const int n_cells,
                                const int outer, const double fallback) {
  if (outer < 0 || mask == nullptr) {
    return fallback;
  }
  double sum = 0.0;
  double w_sum = 0.0;
  for (int k = 0; k < kGhostAnchorSpan; ++k) {
    const int c = outer - k;
    if (c < 0 || c >= n_cells) {
      break;
    }
    if (mask[c] != 0U) {
      continue;
    }
    const double w = 1.0 / static_cast<double>(k + 1);
    sum += w * values[c];
    w_sum += w;
  }
  if (!(w_sum > 0.0)) {
    return fallback;
  }
  return sum / w_sum;
}

struct Layout {
  double R_max;  // the last node
  double R_crit;
  double dR_fine;
  int nr;
};

// compute_dynamic_mesh_params_1d's layout for an outer radius of at least min_outer (one block).
__device__ Layout layout_for(const double R_max0, const double min_dr_crit, const double R_crit_raw,
                             const double min_outer, const MapInputs& in, double* widths,
                             double* node_R) {
  const double R_max = host_max(host_max(R_max0, 4.0 * min_dr_crit), min_outer);
  const double R_crit = host_clamp(R_crit_raw, 0.0, R_max);
  const double dR_fine = host_max(in.mesh_factor * min_dr_crit, 1.0e-12);
  const int nr = graded_nodes(R_crit, dR_fine, R_max, in.nr_max, widths, node_R);
  const Layout layout{node_R[nr], R_crit, dR_fine, nr};
  __syncthreads();
  return layout;
}

__global__ void map_kernel(const int n_cells, const double* __restrict__ zbar,
                           const double* __restrict__ Te, const double* __restrict__ A_eff,
                           const double* __restrict__ x_r, const std::uint8_t* __restrict__ mask,
                           const double* __restrict__ n_hat_layout,
                           const double* __restrict__ n_hat_raw,
                           const double* __restrict__ n_hat_approx, const MapInputs in,
                           const int* __restrict__ nonfinite_flags,
                           double* __restrict__ widths, double* __restrict__ node_R,
                           double* __restrict__ node_Z, MapScalars* __restrict__ out) {
  __shared__ int sh_int[kThreads];
  __shared__ double sh_dbl[kThreads];
  __shared__ int s_fcrit_layout;
  __shared__ double s_R_crit_raw;
  __shared__ double s_r_window;
  __shared__ double s_R_max0;
  __shared__ double s_min_dr_crit;
  __shared__ double s_R_min_outer;
  __shared__ int s_relayout;
  const int t = threadIdx.x;

  // the last cells meeting each condition, and the smallest positive finite cell width
  int l_threshold = -1;
  int l_crit_layout = -1;
  int l_crit_raw = -1;
  int l_outer = -1;
  int l_super_real = -1;
  double l_min_dr = DBL_MAX;
  for (int c = t; c < n_cells; c += blockDim.x) {
    const bool is_void = mask != nullptr && mask[c] != 0U;
    if (n_hat_layout[c] >= in.rmax_n_hat_threshold) {
      l_threshold = c;
    }
    if (n_hat_layout[c] >= 1.0) {
      l_crit_layout = c;
    }
    if (n_hat_raw[c] >= 1.0) {
      l_crit_raw = c;
    }
    if (!is_void) {
      l_outer = c;
      if (n_hat_approx[c] >= 1.0) {
        l_super_real = c;
      }
    }
    const double dr = x_r[c + 1] - x_r[c];
    if ((dr > 0.0) && ::isfinite(dr)) {
      l_min_dr = host_min(l_min_dr, dr);
    }
  }
  const int outermost_threshold = block_max_int(l_threshold, sh_int);
  const int outermost_critical_layout = block_max_int(l_crit_layout, sh_int);
  const int outermost_critical_raw = block_max_int(l_crit_raw, sh_int);
  const int outer = block_max_int(l_outer, sh_int);
  const int outermost_supercritical_real = block_max_int(l_super_real, sh_int);
  const double min_dr_global = block_min_double(l_min_dr, sh_dbl);

  // compute_dynamic_mesh_params_1d: the critical surface, the outer radius, the window
  if (t == 0) {
    const CritEstimate crit =
        estimate_layout(n_hat_layout, x_r, n_cells, outermost_critical_layout);
    const double fallback_r_max = in.r_max_factor * host_max(in.target_radius, 1.0e-12);
    const double R_max0 = (outermost_threshold >= 0)
                              ? in.r_max_factor * x_r[outermost_threshold + 1]
                              : fallback_r_max;
    const double R_crit_raw =
        (crit.fcrit >= 0 && crit.fcrit < n_cells + 1) ? crit.r_interp : (0.5 * R_max0);
    s_fcrit_layout = crit.fcrit;
    s_R_crit_raw = R_crit_raw;
    s_r_window = 0.05 * host_max(R_crit_raw, 1.0e-12);
    s_R_max0 = R_max0;
  }
  __syncthreads();
  // the finest cell near the critical surface
  {
    const int fcrit = s_fcrit_layout;
    const double R_crit_raw = s_R_crit_raw;
    const double r_window = s_r_window;
    double l_min_crit = DBL_MAX;
    for (int c = t; c < n_cells; c += blockDim.x) {
      const double r_l = x_r[c];
      const double r_r = x_r[c + 1];
      const double dr = r_r - r_l;
      if (!(dr > 0.0) || !::isfinite(dr)) {
        continue;
      }
      const double r_c = 0.5 * (r_l + r_r);
      const bool near_fcrit = (fcrit >= 0) ? (::abs(c - fcrit) <= 10) : false;
      const bool near_rcrit = ::fabs(r_c - R_crit_raw) <= r_window;
      if (near_fcrit || near_rcrit) {
        l_min_crit = host_min(l_min_crit, dr);
      }
    }
    const double min_dr_crit_found = block_min_double(l_min_crit, sh_dbl);
    if (t == 0) {
      double min_dr_crit = min_dr_crit_found;
      if (!(min_dr_crit > 0.0) || !::isfinite(min_dr_crit)) {
        min_dr_crit = min_dr_global;
      }
      if (!(min_dr_crit > 0.0) || !::isfinite(min_dr_crit)) {
        min_dr_crit = host_max(s_R_max0 / 4.0, 1.0e-12);
      }
      s_min_dr_crit = min_dr_crit;
    }
    __syncthreads();
  }

  Layout layout = layout_for(s_R_max0, s_min_dr_crit, s_R_crit_raw, 0.0, in, widths, node_R);

  // the ghost corona of map_from_hydro_1d (the fine width of the first layout; the second has
  // the same)
  if (t == 0) {
    const int ghost_configured =
        (in.ghost_enabled != 0 && outer >= 0 && in.ghost_n_out > 0) ? 1 : 0;
    const double ghost_ne_min = host_max(in.ghost_ne_min_frac, 1.0e-12);
    const double ghost_ne_max = host_max(in.ghost_ne_max_frac, ghost_ne_min * 1.0001);
    const double outer_n_hat = (outer >= 0) ? n_hat_raw[outer] : 0.0;
    const double ghost_ne_inner =
        (outer_n_hat < 1.0) ? host_min(outer_n_hat, ghost_ne_max) : ghost_ne_max;
    double ghost_fade = 0.0;
    if (ghost_configured != 0) {
      int resolved_cells = 0;
      for (int c = outer; c >= 0; --c) {
        if (mask != nullptr && mask[c] != 0U) {
          continue;
        }
        if (n_hat_raw[c] < in.ghost_transition_resolved_nhat) {
          ++resolved_cells;
        } else {
          break;
        }
      }
      const double required_cells =
          static_cast<double>(::max(in.ghost_transition_resolved_cells, 1));
      ghost_fade =
          host_clamp(1.0 - static_cast<double>(resolved_cells) / required_cells, 0.0, 1.0);
    }
    const int use_ghost =
        (ghost_configured != 0 && ghost_fade > 0.0 && ghost_ne_inner > ghost_ne_min) ? 1 : 0;
    const double r_surface_outer = (outer >= 0) ? x_r[outer + 1] : 0.0;
    const double Te_anchor_raw = average_outer(Te, mask, n_cells, outer, in.ghost_Te_min_eV);
    const double Te_anchor = host_max(Te_anchor_raw, in.ghost_Te_min_eV);
    const double zbar_anchor =
        host_clamp(host_max(average_outer(zbar, mask, n_cells, outer, in.ghost_zbar_max),
                            in.ghost_zbar_min),
                   in.ghost_zbar_min, host_max(in.ghost_zbar_max, in.ghost_zbar_min));
    const double A_anchor =
        host_max(average_outer(A_eff, mask, n_cells, outer, in.material_A), 1.0e-30);
    // compute_ghost_sound_speed_cm_s
    const double Te_safe = host_max(Te_anchor, 0.0);
    const double z_safe = host_max(zbar_anchor, 1.0);
    const double A_safe = host_max(A_anchor, 1.0e-30);
    const double ghost_cs =
        ::sqrt(z_safe * core::constants::eV_to_erg * Te_safe / (A_safe * kProtonMass));
    const double ghost_width =
        (use_ghost != 0)
            ? host_max(ghost_fade * host_max(static_cast<double>(in.ghost_n_out) * layout.dR_fine,
                                             ghost_cs * in.t_since_turn_on),
                       1.0e-30)
            : 0.0;
    const double r_ghost_outer = r_surface_outer + ghost_width;
    s_relayout = (use_ghost != 0 && r_ghost_outer >= layout.R_max) ? 1 : 0;
    s_R_min_outer =
        r_ghost_outer + host_max(4.0 * layout.dR_fine, 0.05 * r_ghost_outer);
    const double ghost_log_span = (use_ghost != 0) ? ::log(ghost_ne_inner / ghost_ne_min) : 0.0;
    out->ghost_configured = ghost_configured;
    out->use_ghost_corona = use_ghost;
    out->outer_n_hat = outer_n_hat;
    out->ghost_fade = ghost_fade;
    out->ghost_ne_inner = ghost_ne_inner;
    out->ghost_ne_min = ghost_ne_min;
    out->ghost_width = ghost_width;
    out->ghost_scale_length =
        (ghost_log_span > 0.0) ? host_max(ghost_width / ghost_log_span, 1.0e-30) : ghost_width;
    out->ghost_cs = ghost_cs;
    out->r_surface_outer = r_surface_outer;
    out->r_ghost_outer = r_ghost_outer;
    out->Te_anchor = Te_anchor;
    out->zbar_anchor = zbar_anchor;
    out->n_hat_outer_minus_1 = (outer >= 1) ? n_hat_raw[outer - 1] : 0.0;
  }
  __syncthreads();
  if (s_relayout != 0) {
    layout = layout_for(s_R_max0, s_min_dr_crit, s_R_crit_raw, s_R_min_outer, in, widths, node_R);
  }

  if (t == 0) {
    out->R_max = layout.R_max;
    out->dR_fine = layout.dR_fine;
    out->R_crit = layout.R_crit;
    out->nr = layout.nr;
    out->nz = 2 * layout.nr;
    // map_from_hydro_1d's critical face, on the raw n_e / n_c
    out->fcrit_cell = estimate_layout(n_hat_raw, x_r, n_cells, outermost_critical_raw).fcrit;
    out->outer_surface_cell = outer;
    // find_allowed_supercritical_cell_1d_impl
    int allowed_cell = -1;
    int adjacent_cell = -1;
    int fallback_only = 0;
    double r_crit_allowed = -1.0;
    const int outermost_real = outer;
    if (outermost_real >= 0) {
      bool any_subcritical_real = false;
      for (int c = outermost_supercritical_real + 1; c <= outermost_real; ++c) {
        if (mask != nullptr && mask[c] != 0U) {
          continue;
        }
        if (n_hat_approx[c] < 1.0) {
          any_subcritical_real = true;
          break;
        }
      }
      const CritEstimate crit =
          estimate_deposit(n_hat_approx, x_r, n_cells, outermost_supercritical_real);
      const int f = crit.fcrit;
      if (f > 0 && f < n_cells && (mask == nullptr || mask[f] == 0U) &&
          (mask == nullptr || mask[f - 1] == 0U) && n_hat_approx[f - 1] >= 1.0 &&
          n_hat_approx[f] < 1.0) {
        allowed_cell = f - 1;
        adjacent_cell = f;
        r_crit_allowed = crit.r_interp;
      } else if (!any_subcritical_real) {
        allowed_cell = outermost_real;  // the outermost real cell
        r_crit_allowed = crit.r_interp;
        fallback_only = 1;
      }
    }
    out->allowed_cell = allowed_cell;
    out->critical_adjacent_subcritical_cell = adjacent_cell;
    out->fallback_only = fallback_only;
    out->r_crit_allowed = r_crit_allowed;
    // laser_step's resonance-absorption inputs: the outermost downward crossing of n_c
    double ra_r_crit = -1.0;
    double ra_ln = -1.0;
    int c_out = -1;
    for (int c = n_cells - 2; c >= 0; --c) {
      if (n_hat_layout[c] >= 1.0 && n_hat_layout[c + 1] < 1.0) {
        c_out = c;
        break;
      }
    }
    if (c_out >= 0) {
      const int inner = c_out;
      const int outer_c = c_out + 1;
      const double r_inner = 0.5 * (x_r[inner] + x_r[inner + 1]);
      const double r_outer = 0.5 * (x_r[outer_c] + x_r[outer_c + 1]);
      const double n_inner = n_hat_layout[inner];
      const double n_outer = n_hat_layout[outer_c];
      const double alpha = (1.0 - n_inner) / (n_outer - n_inner);
      ra_r_crit = r_inner + alpha * (r_outer - r_inner);
      const double dln = ::log(host_max(n_outer, 1.0e-12)) - ::log(host_max(n_inner, 1.0e-12));
      ra_ln = host_clamp(::fabs((r_outer - r_inner) / dln), 1.0e-5, 1.0);
    }
    out->ra_r_crit_cm = ra_r_crit;
    out->ra_ln_cm = ra_ln;
    // the Langdon collision-charge fallback: Zbar of the outermost real cell
    out->outer_zbar_cell = outer;
    out->outer_zbar = (outer >= 0) ? zbar[outer] : 0.0;
    out->nonfinite_flags = *nonfinite_flags;
  }
  // build_mirrored_Z_from_R
  const int nr = layout.nr;
  for (int k = t; k <= 2 * nr; k += blockDim.x) {
    node_Z[k] = (k <= nr) ? -node_R[nr - k] : node_R[k - nr];
  }
}

}  // namespace

struct Workspace::Impl {
  double* n_hat_layout = nullptr;
  double* n_hat_raw = nullptr;
  double* n_hat_approx = nullptr;
  std::size_t cap_cells1 = 0, cap_cells2 = 0, cap_cells3 = 0;
  double* widths = nullptr;
  double* node_R = nullptr;
  double* node_Z = nullptr;
  std::size_t cap_widths = 0, cap_node_R = 0, cap_node_Z = 0;
  MapScalars* scalars = nullptr;       // device
  MapScalars* scalars_host = nullptr;  // pinned
  int* flags = nullptr;                // device

  ~Impl() {
    void* ptrs[] = {n_hat_layout, n_hat_raw, n_hat_approx, widths, node_R, node_Z, scalars,
                    flags};
    for (void* p : ptrs) {
      if (p != nullptr) {
        static_cast<void>(cudaFree(p));
      }
    }
    if (scalars_host != nullptr) {
      static_cast<void>(cudaFreeHost(scalars_host));
    }
  }
};

Workspace::Workspace() : impl_(new Impl) {}
Workspace::~Workspace() { delete impl_; }

MapScalars map_scalars(Workspace& ws, const core::State& state, const double* A_eff,
                       const MapInputs& in, cudaStream_t stream) {
  auto& w = *ws.impl();
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(n_cells > 0 && static_cast<int>(state.x_r.size()) == n_cells + 1 &&
                    static_cast<int>(state.zbar.size()) == n_cells &&
                    static_cast<int>(state.Te.size()) == n_cells && A_eff != nullptr,
                "laser map 1D: state size mismatch");
  TENRYU_ASSERT(state.cell_is_void.empty() ||
                    static_cast<int>(state.cell_is_void.size()) == n_cells,
                "laser map 1D: void mask size mismatch");
  const std::size_t n = static_cast<std::size_t>(n_cells);
  ensure(&w.n_hat_layout, &w.cap_cells1, n, "laser map n_hat alloc");
  ensure(&w.n_hat_raw, &w.cap_cells2, n, "laser map n_hat alloc");
  ensure(&w.n_hat_approx, &w.cap_cells3, n, "laser map n_hat alloc");
  // the layout has at most max(nr_max, 4) cells
  const std::size_t nr_cap = static_cast<std::size_t>(std::max(in.nr_max, 4));
  ensure(&w.widths, &w.cap_widths, nr_cap + 1U, "laser map widths alloc");
  ensure(&w.node_R, &w.cap_node_R, nr_cap + 1U, "laser map node_R alloc");
  ensure(&w.node_Z, &w.cap_node_Z, 2U * nr_cap + 1U, "laser map node_Z alloc");
  if (w.scalars == nullptr) {
    check(cudaMalloc(reinterpret_cast<void**>(&w.scalars), sizeof(MapScalars)),
          "laser map scalars alloc");
    check(cudaMallocHost(reinterpret_cast<void**>(&w.scalars_host), sizeof(MapScalars)),
          "laser map scalars pinned alloc");
    check(cudaMalloc(reinterpret_cast<void**>(&w.flags), sizeof(int)), "laser map flags alloc");
  }
  check(cudaMemsetAsync(w.flags, 0, sizeof(int), stream), "laser map flags reset");
  const std::uint8_t* mask =
      state.cell_is_void.empty() ? nullptr : core::device_cell_is_void(state.cell_is_void);
  const double n_crit_safe = std::max(in.n_crit, 1.0e-30);
  cells_kernel<<<(n_cells + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      n_cells, state.rho.data(), state.zbar.data(), state.Te.data(), A_eff, mask, in.n_crit,
      n_crit_safe, w.n_hat_layout, w.n_hat_raw, w.n_hat_approx, w.flags);
  check(cudaGetLastError(), "laser map cells launch");
  map_kernel<<<1, kThreads, 0, stream>>>(n_cells, state.zbar.data(), state.Te.data(), A_eff,
                                         state.x_r.data(), mask, w.n_hat_layout, w.n_hat_raw,
                                         w.n_hat_approx, in, w.flags, w.widths, w.node_R,
                                         w.node_Z, w.scalars);
  check(cudaGetLastError(), "laser map launch");
  check(cudaMemcpyAsync(w.scalars_host, w.scalars, sizeof(MapScalars), cudaMemcpyDeviceToHost,
                        stream),
        "laser map scalars D2H");
  check(cudaStreamSynchronize(stream), "laser map scalars sync");
  const MapScalars result = *w.scalars_host;
  TENRYU_ASSERT(result.nr >= 4 && static_cast<std::size_t>(result.nr) <= nr_cap,
                "laser map 1D: node layout outside its bounds");
  return result;
}

const double* node_R(const Workspace& ws) { return ws.impl()->node_R; }
const double* node_Z(const Workspace& ws) { return ws.impl()->node_Z; }
const double* n_hat_layout(const Workspace& ws) { return ws.impl()->n_hat_layout; }
const double* n_hat_raw(const Workspace& ws) { return ws.impl()->n_hat_raw; }
const double* n_hat_approx(const Workspace& ws) { return ws.impl()->n_hat_approx; }

}  // namespace tenryu::laser::laser_map_1d
