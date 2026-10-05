#include "hydro/ale_1d_rezone_device.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "core/device_ordered_sum.cuh"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/glibc_libm_device.cuh"
#include "hydro/ale_1d_rezone.cuh"
#include "mesh/geometry_1d.cuh"

// The device build of the 1D ALE rezone candidate (ale_1d_rezone_device.cuh). Every function
// below names the host function whose operations it repeats; sums run in the host loop's order in
// one thread (or in index order after a compaction of the nonzero terms, which leaves a sum from +0
// unchanged), maxima and minima of values are order-free, std::max and std::min are written as
// their comparisons, products and sums are separately rounded (__dmul_rn, __dadd_rn), and exp and
// pow are glibc's (core::glibc_libm).

namespace tenryu::hydro::ale1d {
namespace {

constexpr int kBlock = 256;
constexpr int kSumBlock = 256;
constexpr int kSumPerThread = 4;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

int blocks_for(const long long n) {
  return static_cast<int>((n + kBlock - 1) / kBlock);
}

template <typename T>
T* scratch(const char* tag, const std::size_t count) {
  return static_cast<T*>(core::device_scratch_acquire(tag, std::max<std::size_t>(count, 1) * sizeof(T)));
}

int effective_cell_count(const core::State& state, const core::Config& cfg) {
  if (state.mesh.topo.n_cells > 0) {
    return state.mesh.topo.n_cells;
  }
  return cfg.mesh.nr;
}

struct DeviceFeature {
  int kind;
  int peak_cell_or_face;
  int pinned_face;
  double x_center;
  double sigma_x;
  double sigma_r;
  double confidence;
  double target_cells;
};

constexpr int kKindLaser = static_cast<int>(FeatureKind::LaserAbsorption);
constexpr int kKindAblation = static_cast<int>(FeatureKind::AblationFront);
constexpr int kKindShock = static_cast<int>(FeatureKind::Shock);
constexpr int kKindInterface = static_cast<int>(FeatureKind::MaterialInterface);

// The device state of one candidate build: the monitor's plan, the gate values, the outcome.
struct CandidateState {
  // monitor (build_monitor)
  double spatial_integral;
  double spatial_confidence;
  double spatial_amplitude;
  int spatial_on;
  // rezone
  int n_protected;
  double protected_fraction;
  int w_invalid;
  double w_min;
  double w_max;
  int n_segments;
  int min_movable_segment_size;
  int decided;  // the outcome is known before the equidistribution (a skip, or a uniform monitor)
  int uniform;
  int segment_failed;
  double max_mu;
  double max_r;
  double displacement_scale;
  int not_ordered;
  int success;
  int skip_reason;
  // floor candidate
  int no_relief;
  // finalize
  int n_protected_final;
  int geometry_invalid;
  double dt_current;
  double dt_candidate;
};

struct RezoneParams {
  int n;
  int n_features;
  int outer_fixed_mask;      // build_node_mask: boundary_1d == "fixed"
  int outer_fixed_enforce;   // enforce_boundary_candidate: "fixed" or "reflect"
  double r_min;
  double r_max;
  // build_monitor
  double w0;
  double w_max;
  int smoothing_iterations;
  int smooth_across_protected;
  double min_floor_fraction;
  double truncation_sigma;
  int spatial_enabled;  // spatial_monitor_enabled && spatial_target_cells_fraction > 0
  double spatial_fraction;
  double spatial_power;
  double laser_dr_min, laser_dr_max, ablation_dr_min, ablation_dr_max, shock_dr_min,
      shock_dr_max;
  // rezone gates
  double protected_fraction_max;
  int min_movable_segment_hard;
  int min_cells;
  double max_disp_mu;
  double max_disp_r;
  int geom;
};

__device__ __forceinline__ double std_max(const double a, const double b) {
  return (a < b) ? b : a;  // std::max(a, b)
}

__device__ __forceinline__ double std_min(const double a, const double b) {
  return (b < a) ? b : a;  // std::min(a, b)
}

// clamp_value (ale_1d_rezone.cu): std::min(std::max(x, lo), hi).
__device__ __forceinline__ double clamp_value(const double x, const double lo, const double hi) {
  return std_min(std_max(x, lo), hi);
}

__device__ __forceinline__ bool is_positive_finite(const double x) {
  return isfinite(x) && x > 0.0;
}

// build_mass_map with copy_mass_or_uniform: the masses when they are all finite and nonnegative
// with a positive sum, else 1 per cell; dx = m / total, node_x the prefix, cell_x the midpoints.
__global__ void mass_map_kernel(const double* __restrict__ mass, const int has_mass, const int n,
                                double* __restrict__ node_x, double* __restrict__ cell_x,
                                double* __restrict__ mdx) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  bool use_mass = false;
  double total = 0.0;
  if (has_mass != 0) {
    bool valid = true;
    for (int i = 0; i < n; ++i) {
      const double m = mass[i];
      if (!isfinite(m) || m < 0.0) {
        valid = false;
        break;
      }
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      sum = __dadd_rn(sum, mass[i]);
    }
    use_mass = valid && sum > 0.0;
    total = sum;
  }
  if (!use_mass) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
      sum = __dadd_rn(sum, 1.0);
    }
    total = sum;
  }
  double prefix = 0.0;
  for (int i = 0; i < n; ++i) {
    const double m = use_mass ? mass[i] : 1.0;
    const double d =
        (total > 0.0) ? __ddiv_rn(m, total) : __ddiv_rn(1.0, static_cast<double>(n));
    mdx[i] = d;
    node_x[i] = prefix;
    cell_x[i] = __dadd_rn(prefix, __dmul_rn(0.5, d));
    prefix = __dadd_rn(prefix, d);
  }
  node_x[n] = 1.0;
}

// copy_nodes_or_uniform without node radii: r_min + j (r_max - r_min) / n.
__global__ void uniform_nodes_kernel(const double r_min, const double r_max, const int n,
                                     double* __restrict__ r) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > n) {
    return;
  }
  const double dr = (n > 0) ? __ddiv_rn(__dsub_rn(r_max, r_min), static_cast<double>(n)) : 0.0;
  r[j] = __dadd_rn(r_min, __dmul_rn(static_cast<double>(j), dr));
}

// build_node_mask: node 0, node n when the outer boundary is fixed, and the pinned faces of the
// material interfaces (a face out of range replaced by the node nearest x_center, nearest_face).
__global__ void node_mask_kernel(const DeviceFeature* __restrict__ features, const RezoneParams p,
                                 const double* __restrict__ node_x,
                                 std::uint8_t* __restrict__ pinned, CandidateState* st) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int n = p.n;
  for (int j = 0; j <= n; ++j) {
    pinned[j] = 0U;
  }
  pinned[0] = 1U;
  if (p.outer_fixed_mask != 0) {
    pinned[n] = 1U;
  }
  for (int f = 0; f < p.n_features; ++f) {
    const DeviceFeature& feature = features[f];
    if (feature.kind != kKindInterface || feature.pinned_face == 0) {
      continue;
    }
    int face = feature.peak_cell_or_face;
    if (face < 0 || face > n) {
      int best = 0;
      for (int j = 1; j <= n; ++j) {
        if (fabs(__dsub_rn(node_x[j], feature.x_center)) <
            fabs(__dsub_rn(node_x[best], feature.x_center))) {
          best = j;
        }
      }
      face = best;
    }
    face = min(max(face, 0), n);
    pinned[face] = 1U;
  }
  int count = 0;
  for (int j = 0; j <= n; ++j) {
    count += (pinned[j] != 0U) ? 1 : 0;
  }
  st->n_protected = count;
  st->protected_fraction = __ddiv_rn(static_cast<double>(count), static_cast<double>(n + 1));
}

// build_monitor's Gaussian kernels in the mass coordinate: per feature f and cell i, the value
// exp(-q^2 / 2) (q = (cell_x - x_center) / sigma_x) within the truncation and its integrand
// g dx; zero for the features build_monitor skips.
__global__ void monitor_values_kernel(const DeviceFeature* __restrict__ features,
                                      const RezoneParams p, const double* __restrict__ cell_x,
                                      const double* __restrict__ mdx,
                                      double* __restrict__ values,
                                      double* __restrict__ integrand) {
  const long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const long long total = static_cast<long long>(p.n) * p.n_features;
  if (t >= total) {
    return;
  }
  const int f = static_cast<int>(t / p.n);
  const int i = static_cast<int>(t % p.n);
  const DeviceFeature& feature = features[f];
  double value = 0.0;
  double contribution = 0.0;
  if (feature.confidence > 0.0 && feature.target_cells > 0.0 &&
      is_positive_finite(feature.sigma_x)) {
    const double truncation = __dmul_rn(p.truncation_sigma, feature.sigma_x);
    const double dx = __dsub_rn(cell_x[i], feature.x_center);
    if (!(fabs(dx) > truncation)) {
      const double q = __ddiv_rn(dx, feature.sigma_x);
      value = core::glibc_libm::exp(__dmul_rn(__dmul_rn(-0.5, q), q));
      contribution = __dmul_rn(value, mdx[i]);
    }
  }
  values[t] = value;
  integrand[t] = contribution;
}

// The integral of each feature's kernel, its integrand summed in cell order (one block per
// feature).
__global__ void monitor_integrals_kernel(const double* __restrict__ integrand, const int n,
                                         double* __restrict__ integrals) {
  __shared__ double sh_values[kSumBlock * kSumPerThread];
  __shared__ int sh_scan[kSumBlock];
  const int f = blockIdx.x;
  const double sum = core::device_ordered::block_ordered_sum_nonzero<kSumBlock, kSumPerThread>(
      integrand + static_cast<long long>(f) * n, n, 0.0, sh_values, sh_scan);
  if (threadIdx.x == 0) {
    integrals[f] = sum;
  }
}

// build_monitor's spatial resolution pressure u_i (summed over the spatial features in their
// order) and its integrand u_i dx_i.
__global__ void spatial_monitor_kernel(const DeviceFeature* __restrict__ features,
                                       const RezoneParams p, const double* __restrict__ cell_x,
                                       const double* __restrict__ mdx,
                                       const double* __restrict__ r,
                                       double* __restrict__ u, double* __restrict__ u_integrand) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n) {
    return;
  }
  // cell_dr: std::max(r_{i+1} - r_i, 1e-14 std::max(|r_n|, kTiny))
  const double outer = std_max(fabs(r[p.n]), rezone_detail::kTiny);
  const double cell_dr = std_max(__dsub_rn(r[i + 1], r[i]), __dmul_rn(1.0e-14, outer));
  double value = 0.0;
  for (int f = 0; f < p.n_features; ++f) {
    const DeviceFeature& feature = features[f];
    const bool spatial_kind = feature.kind == kKindLaser || feature.kind == kKindAblation ||
                              feature.kind == kKindShock;
    if (!spatial_kind || !(feature.confidence > 0.0) || !is_positive_finite(feature.sigma_x)) {
      continue;
    }
    const double sigma = __dmul_rn(2.0, feature.sigma_x);
    // spatial_target_dr
    double lo = p.shock_dr_min;
    double hi = p.shock_dr_max;
    if (feature.kind == kKindLaser) {
      lo = p.laser_dr_min;
      hi = p.laser_dr_max;
    } else if (feature.kind == kKindAblation) {
      lo = p.ablation_dr_min;
      hi = p.ablation_dr_max;
    }
    const double target_dr = clamp_value(__ddiv_rn(feature.sigma_r, 3.0), lo, hi);
    const double dx = __ddiv_rn(__dsub_rn(cell_x[i], feature.x_center), sigma);
    const double g = core::glibc_libm::exp(__dmul_rn(__dmul_rn(-0.5, dx), dx));
    const double ratio = std_max(1.0, __ddiv_rn(cell_dr, target_dr));
    const double term = __dmul_rn(__dmul_rn(feature.confidence, g),
                                  __dsub_rn(core::glibc_libm::pow(ratio, p.spatial_power), 1.0));
    value = __dadd_rn(value, term);
  }
  u[i] = value;
  u_integrand[i] = __dmul_rn(value, mdx[i]);
}

__global__ void ordered_sum_kernel(const double* __restrict__ x, const int n,
                                   double* __restrict__ out) {
  __shared__ double sh_values[kSumBlock * kSumPerThread];
  __shared__ int sh_scan[kSumBlock];
  const double sum = core::device_ordered::block_ordered_sum_nonzero<kSumBlock, kSumPerThread>(
      x, n, 0.0, sh_values, sh_scan);
  if (threadIdx.x == 0) {
    *out = sum;
  }
}

// build_monitor's budget: the features with a kernel (integral at or above the floor) in their
// order, the spatial share, the cap of the active cells, and the amplitudes (0 for the features
// without a kernel).
__global__ void monitor_plan_kernel(const DeviceFeature* __restrict__ features,
                                    const RezoneParams p, const double* __restrict__ integrals,
                                    const double* __restrict__ spatial_integral,
                                    double* __restrict__ amplitudes, CandidateState* st) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const double n_d = static_cast<double>(p.n);
  double active_budget = 0.0;
  for (int f = 0; f < p.n_features; ++f) {
    const DeviceFeature& feature = features[f];
    amplitudes[f] = 0.0;
    if (!(feature.confidence > 0.0) || !(feature.target_cells > 0.0) ||
        !is_positive_finite(feature.sigma_x)) {
      continue;
    }
    if (integrals[f] < rezone_detail::kKernelIntegralFloor) {
      continue;
    }
    active_budget = __dadd_rn(active_budget, __dmul_rn(feature.confidence, feature.target_cells));
  }
  double s_integral = 0.0;
  double s_confidence = 0.0;
  if (p.spatial_enabled != 0) {
    for (int f = 0; f < p.n_features; ++f) {
      const DeviceFeature& feature = features[f];
      const bool spatial_kind = feature.kind == kKindLaser || feature.kind == kKindAblation ||
                                feature.kind == kKindShock;
      if (!spatial_kind || !(feature.confidence > 0.0) ||
          !is_positive_finite(feature.sigma_x)) {
        continue;
      }
      s_confidence = std_max(s_confidence, feature.confidence);
    }
    s_integral = *spatial_integral;
    if (s_integral >= rezone_detail::kKernelIntegralFloor) {
      active_budget = __dadd_rn(active_budget,
                                __dmul_rn(__dmul_rn(p.spatial_fraction, n_d), s_confidence));
    }
  }
  const double max_active_budget = __dmul_rn(__dsub_rn(1.0, p.min_floor_fraction), n_d);
  const double alpha_budget = (active_budget > max_active_budget && active_budget > 0.0)
                                  ? __ddiv_rn(max_active_budget, active_budget)
                                  : 1.0;
  double active_after_cap = 0.0;
  for (int f = 0; f < p.n_features; ++f) {
    const DeviceFeature& feature = features[f];
    if (!(feature.confidence > 0.0) || !(feature.target_cells > 0.0) ||
        !is_positive_finite(feature.sigma_x) ||
        integrals[f] < rezone_detail::kKernelIntegralFloor) {
      continue;
    }
    active_after_cap = __dadd_rn(
        active_after_cap,
        __dmul_rn(alpha_budget, __dmul_rn(feature.confidence, feature.target_cells)));
  }
  const double spatial_cells =
      (s_integral >= rezone_detail::kKernelIntegralFloor)
          ? __dmul_rn(__dmul_rn(__dmul_rn(alpha_budget, p.spatial_fraction), n_d), s_confidence)
          : 0.0;
  active_after_cap = __dadd_rn(active_after_cap, spatial_cells);
  const double n_floor = std_max(rezone_detail::kTiny, __dsub_rn(n_d, active_after_cap));
  for (int f = 0; f < p.n_features; ++f) {
    const DeviceFeature& feature = features[f];
    if (!(feature.confidence > 0.0) || !(feature.target_cells > 0.0) ||
        !is_positive_finite(feature.sigma_x) ||
        integrals[f] < rezone_detail::kKernelIntegralFloor) {
      continue;
    }
    const double n_star =
        __dmul_rn(alpha_budget, __dmul_rn(feature.confidence, feature.target_cells));
    amplitudes[f] = __ddiv_rn(n_star, __dmul_rn(n_floor, integrals[f]));
  }
  st->spatial_integral = s_integral;
  st->spatial_confidence = s_confidence;
  st->spatial_on = (spatial_cells > 0.0) ? 1 : 0;
  st->spatial_amplitude =
      (spatial_cells > 0.0) ? __ddiv_rn(spatial_cells, __dmul_rn(n_floor, s_integral)) : 0.0;
}

// W = w0 + the kernels' amplitudes times their values (in the features' order) + the spatial
// amplitude times u.
__global__ void monitor_assemble_kernel(const DeviceFeature* __restrict__ features,
                                        const RezoneParams p, const double* __restrict__ values,
                                        const double* __restrict__ amplitudes,
                                        const double* __restrict__ integrals,
                                        const double* __restrict__ u, const CandidateState* st,
                                        double* __restrict__ W) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n) {
    return;
  }
  double w = p.w0;
  for (int f = 0; f < p.n_features; ++f) {
    const DeviceFeature& feature = features[f];
    if (!(feature.confidence > 0.0) || !(feature.target_cells > 0.0) ||
        !is_positive_finite(feature.sigma_x) ||
        integrals[f] < rezone_detail::kKernelIntegralFloor) {
      continue;
    }
    w = __dadd_rn(w, __dmul_rn(amplitudes[f], values[static_cast<long long>(f) * p.n + i]));
  }
  if (st->spatial_on != 0) {
    w = __dadd_rn(w, __dmul_rn(st->spatial_amplitude, u[i]));
  }
  W[i] = w;
}

// One smoothing pass of build_monitor: (2 W_i + W_{i-1} + W_{i+1}) / (2 + 1 + 1), a neighbour
// across a pinned face left out unless the smoothing crosses protected faces; the last pass
// clamps W to [w0, w_max].
__global__ void monitor_smooth_kernel(const RezoneParams p, const std::uint8_t* __restrict__ pinned,
                                      const double* __restrict__ W, double* __restrict__ next,
                                      const int clamp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.n) {
    return;
  }
  double value = W[i];
  if (clamp >= 0) {  // a smoothing pass (clamp < 0: the clamp alone)
    double numerator = __dmul_rn(2.0, W[i]);
    double denominator = 2.0;
    if (i > 0 && (p.smooth_across_protected != 0 || pinned[i] == 0U)) {
      numerator = __dadd_rn(numerator, W[i - 1]);
      denominator = __dadd_rn(denominator, 1.0);
    }
    if (i + 1 < p.n && (p.smooth_across_protected != 0 || pinned[i + 1] == 0U)) {
      numerator = __dadd_rn(numerator, W[i + 1]);
      denominator = __dadd_rn(denominator, 1.0);
    }
    value = __ddiv_rn(numerator, denominator);
  }
  if (clamp != 0) {
    value = clamp_value(value, p.w0, p.w_max);
  }
  next[i] = value;
}

// The monitor's gate values for rezone: every W positive and finite, and its smallest and largest
// values (monitor_is_uniform).
__global__ void monitor_stats_kernel(const double* __restrict__ W, const int n,
                                     CandidateState* st) {
  __shared__ int sh_invalid[kBlock];
  __shared__ double sh_min[kBlock];
  __shared__ double sh_max[kBlock];
  const int t = static_cast<int>(threadIdx.x);
  int invalid = 0;
  double lo = CUDART_INF;
  double hi = -CUDART_INF;
  for (int i = t; i < n; i += kBlock) {
    const double w = W[i];
    if (!is_positive_finite(w)) {
      invalid = 1;
    }
    lo = (w < lo) ? w : lo;
    hi = (hi < w) ? w : hi;
  }
  sh_invalid[t] = invalid;
  sh_min[t] = lo;
  sh_max[t] = hi;
  __syncthreads();
  for (int offset = kBlock / 2; offset > 0; offset >>= 1) {
    if (t < offset) {
      sh_invalid[t] |= sh_invalid[t + offset];
      sh_min[t] = (sh_min[t + offset] < sh_min[t]) ? sh_min[t + offset] : sh_min[t];
      sh_max[t] = (sh_max[t] < sh_max[t + offset]) ? sh_max[t + offset] : sh_max[t];
    }
    __syncthreads();
  }
  if (t == 0) {
    st->w_invalid = sh_invalid[0];
    st->w_min = sh_min[0];
    st->w_max = sh_max[0];
  }
}

// rezone's gates before the equidistribution, in its order: the protected fraction, a monitor
// value that is not positive and finite, the smallest movable segment, the cell count, a uniform
// monitor. The segment break points (node 0, the pinned interior nodes, node n) go to breaks.
__global__ void rezone_gates_kernel(const RezoneParams p, const std::uint8_t* __restrict__ pinned,
                                    int* __restrict__ breaks, CandidateState* st) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int n = p.n;
  st->decided = 0;
  st->uniform = 0;
  st->success = 0;
  st->skip_reason = static_cast<int>(Ale1dSkipReason::None);
  st->min_movable_segment_size = 0;
  st->segment_failed = 0;
  st->max_mu = 0.0;
  st->max_r = 0.0;
  st->displacement_scale = 1.0;
  st->not_ordered = 0;
  if (st->protected_fraction > p.protected_fraction_max) {
    st->decided = 1;
    st->skip_reason = static_cast<int>(Ale1dSkipReason::ProtectedFractionTooHigh);
    return;
  }
  if (st->w_invalid != 0) {
    st->decided = 1;
    st->skip_reason = static_cast<int>(Ale1dSkipReason::CandidateInvalid);
    return;
  }
  int count = 0;
  breaks[count++] = 0;
  for (int j = 1; j < n; ++j) {
    if (pinned[j] != 0U) {
      breaks[count++] = j;
    }
  }
  breaks[count++] = n;
  st->n_segments = count - 1;
  int min_size = n;
  for (int b = 1; b < count; ++b) {
    min_size = min(min_size, breaks[b] - breaks[b - 1]);
  }
  st->min_movable_segment_size = min_size;
  if (min_size < p.min_movable_segment_hard) {
    st->decided = 1;
    st->skip_reason = static_cast<int>(Ale1dSkipReason::MovableSegmentTooSmall);
    return;
  }
  if (n < p.min_cells) {
    st->decided = 1;
    st->skip_reason = static_cast<int>(Ale1dSkipReason::NTooSmall);
    return;
  }
  // monitor_is_uniform: (hi - lo) <= 16 eps std::max(|hi|, 1)
  const double scale = std_max(fabs(st->w_max), 1.0);
  if (__dsub_rn(st->w_max, st->w_min) <=
      __dmul_rn(__dmul_rn(16.0, 2.220446049250313080847e-16), scale)) {
    st->decided = 1;
    st->uniform = 1;
  }
}

// rezone's equidistribution of W dx over each movable segment: one thread per segment, the
// segment's nodes placed where the running integral reaches equal fractions (x_new holds node_x).
__global__ void equidistribute_kernel(const RezoneParams p, const int* __restrict__ breaks,
                                      const double* __restrict__ W, const double* __restrict__ mdx,
                                      const double* __restrict__ node_x,
                                      double* __restrict__ x_new, CandidateState* st) {
  if (st->decided != 0) {
    return;
  }
  for (int b = 1 + static_cast<int>(threadIdx.x); b <= st->n_segments;
       b += static_cast<int>(blockDim.x)) {
    const int j_lo = breaks[b - 1];
    const int j_hi = breaks[b];
    const int segment_cells = j_hi - j_lo;
    if (segment_cells <= 0) {
      continue;
    }
    double segment_integral = 0.0;
    for (int i = j_lo; i < j_hi; ++i) {
      segment_integral = __dadd_rn(segment_integral, __dmul_rn(W[i], mdx[i]));
    }
    if (!(segment_integral > 0.0)) {
      atomicOr(&st->segment_failed, 1);
      continue;
    }
    int cell = j_lo;
    double cumulative = 0.0;
    for (int j = j_lo + 1; j < j_hi; ++j) {
      const double target = __ddiv_rn(
          __dmul_rn(static_cast<double>(j - j_lo), segment_integral),
          static_cast<double>(segment_cells));
      while (cell < j_hi - 1) {
        const double cell_integral = __dmul_rn(W[cell], mdx[cell]);
        if (__dadd_rn(cumulative, cell_integral) >= target) {
          break;
        }
        cumulative = __dadd_rn(cumulative, cell_integral);
        ++cell;
      }
      const double w = W[cell];
      const double local_dx =
          clamp_value(__ddiv_rn(__dsub_rn(target, cumulative), w), 0.0, mdx[cell]);
      x_new[j] = __dadd_rn(node_x[cell], local_dx);
    }
  }
}

// radius_from_mass_coordinate (ale_1d_rezone.cu).
__device__ double radius_from_mass_coordinate(const double* node_x, const double* r, const int n,
                                              const double x) {
  if (x <= node_x[0]) {
    return r[0];
  }
  if (x >= node_x[n]) {
    return r[n];
  }
  // std::upper_bound: the first node with node_x > x
  int lo = 0;
  int count = n + 1;
  while (count > 0) {
    const int step = count / 2;
    const int mid = lo + step;
    if (!(x < node_x[mid])) {
      lo = mid + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  int i = lo - 1;
  i = min(max(i, 0), n - 1);
  const double x0 = node_x[i];
  const double x1 = node_x[i + 1];
  if (!(x1 > x0)) {
    return r[i];
  }
  const double t = __ddiv_rn(__dsub_rn(x, x0), __dsub_rn(x1, x0));
  return __dadd_rn(__dmul_rn(__dsub_rn(1.0, t), r[i]), __dmul_rn(t, r[i + 1]));
}

// The candidate radii: the current radii for a uniform monitor (or a skip), else the radii of the
// equidistributed mass coordinates; with rescale, the displacement first scaled back
// (x_new = node_x + s (x_new - node_x)) when the displacement cap applies.
__global__ void candidate_radii_kernel(const RezoneParams p, const double* __restrict__ node_x,
                                       const double* __restrict__ r_old,
                                       double* __restrict__ x_new,
                                       double* __restrict__ r_candidate, const int rescale,
                                       const CandidateState* st) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > p.n) {
    return;
  }
  if (rescale != 0) {
    if (st->decided != 0 || st->segment_failed != 0 || !(st->displacement_scale < 1.0)) {
      return;
    }
    x_new[j] = __dadd_rn(node_x[j],
                         __dmul_rn(st->displacement_scale, __dsub_rn(x_new[j], node_x[j])));
  } else if (st->decided != 0) {
    r_candidate[j] = r_old[j];
    return;
  }
  r_candidate[j] = radius_from_mass_coordinate(node_x, r_old, p.n, x_new[j]);
}

// local_node_spacing (ale_1d_rezone.cu).
__device__ double local_node_spacing(const double* x, const int n, const int j) {
  double scale = CUDART_INF;
  if (j > 0) {
    scale = std_min(scale, fabs(__dsub_rn(x[j], x[j - 1])));
  }
  if (j < n) {
    scale = std_min(scale, fabs(__dsub_rn(x[j + 1], x[j])));
  }
  return (isfinite(scale) && scale > 0.0) ? scale : 1.0;
}

// compute_displacement_diagnostics: the largest node displacement over the local node spacing,
// in the mass coordinate and in radius; then (first pass) the displacement cap's scale.
__global__ void displacement_kernel(const RezoneParams p, const double* __restrict__ node_x,
                                    const double* __restrict__ r_old,
                                    const double* __restrict__ x_new,
                                    const double* __restrict__ r_candidate, const int second,
                                    CandidateState* st) {
  __shared__ double sh_mu[kBlock];
  __shared__ double sh_r[kBlock];
  if (st->decided != 0 || st->segment_failed != 0) {
    return;
  }
  if (second != 0 && !(st->displacement_scale < 1.0)) {
    return;
  }
  const int t = static_cast<int>(threadIdx.x);
  double max_mu = 0.0;
  double max_r = 0.0;
  for (int j = t; j <= p.n; j += kBlock) {
    const double mu = __ddiv_rn(fabs(__dsub_rn(x_new[j], node_x[j])),
                                local_node_spacing(node_x, p.n, j));
    const double dr = __ddiv_rn(fabs(__dsub_rn(r_candidate[j], r_old[j])),
                                local_node_spacing(r_old, p.n, j));
    max_mu = std_max(max_mu, mu);
    max_r = std_max(max_r, dr);
  }
  sh_mu[t] = max_mu;
  sh_r[t] = max_r;
  __syncthreads();
  for (int offset = kBlock / 2; offset > 0; offset >>= 1) {
    if (t < offset) {
      sh_mu[t] = std_max(sh_mu[t], sh_mu[t + offset]);
      sh_r[t] = std_max(sh_r[t], sh_r[t + offset]);
    }
    __syncthreads();
  }
  if (t == 0) {
    st->max_mu = sh_mu[0];
    st->max_r = sh_r[0];
    if (second == 0) {
      double scale = 1.0;
      if (sh_mu[0] > p.max_disp_mu) {
        scale = std_min(scale, __ddiv_rn(p.max_disp_mu, sh_mu[0]));
      }
      if (sh_r[0] > p.max_disp_r) {
        scale = std_min(scale, __ddiv_rn(p.max_disp_r, sh_r[0]));
      }
      st->displacement_scale = scale;
    }
  }
}

// candidate_is_strictly_ordered, and rezone's outcome.
__global__ void rezone_outcome_kernel(const int n, const double* __restrict__ r_candidate,
                                      CandidateState* st) {
  __shared__ int sh_bad[kBlock];
  const int t = static_cast<int>(threadIdx.x);
  int bad = 0;
  for (int j = 1 + t; j <= n; j += kBlock) {
    if (!(r_candidate[j] > r_candidate[j - 1])) {
      bad = 1;
    }
  }
  sh_bad[t] = bad;
  __syncthreads();
  for (int offset = kBlock / 2; offset > 0; offset >>= 1) {
    if (t < offset) {
      sh_bad[t] |= sh_bad[t + offset];
    }
    __syncthreads();
  }
  if (t != 0) {
    return;
  }
  st->not_ordered = sh_bad[0];
  if (st->decided != 0 && st->uniform == 0) {
    st->success = 0;  // a skip (its reason is set)
    return;
  }
  if (st->decided == 0 && st->segment_failed != 0) {
    st->success = 0;
    st->skip_reason = static_cast<int>(Ale1dSkipReason::CandidateInvalid);
    return;
  }
  if (sh_bad[0] != 0) {
    st->success = 0;
    st->skip_reason = static_cast<int>(Ale1dSkipReason::CandidateInvalid);
    return;
  }
  st->success = 1;
  st->skip_reason = static_cast<int>(Ale1dSkipReason::None);
}

// The driver's steps after a successful candidate: enforce_boundary_candidate (node 0 at r_min,
// node n at r_max for a fixed or reflecting outer boundary, both pinned; the pinned nodes
// counted), validate_candidate_geometry (finite ordered radii, positive widths and volumes), and
// acoustic_dt_bounds (the current mesh's min dr / c_s, and the candidate's with the largest sound
// speed of the current cells it overlaps; the two-pointer sweep in one thread).
__global__ void finalize_kernel(const RezoneParams p, double* __restrict__ r_candidate,
                                std::uint8_t* __restrict__ pinned,
                                const double* __restrict__ r_current,
                                const double* __restrict__ cs, const int has_cs,
                                CandidateState* st) {
  __shared__ int sh_count[kBlock];
  __shared__ int sh_bad[kBlock];
  __shared__ double sh_dt[kBlock];
  const int t = static_cast<int>(threadIdx.x);
  const int n = p.n;
  if (st->success == 0) {
    return;
  }
  if (t == 0) {
    r_candidate[0] = p.r_min;
    pinned[0] = 1U;
    if (p.outer_fixed_enforce != 0) {
      r_candidate[n] = p.r_max;
      pinned[n] = 1U;
    }
  }
  __syncthreads();
  int count = 0;
  int bad = 0;
  double dt_current = CUDART_INF;
  for (int j = t; j <= n; j += kBlock) {
    count += (pinned[j] != 0U) ? 1 : 0;
  }
  for (int i = t; i < n; i += kBlock) {
    const double r0 = r_candidate[i];
    const double r1 = r_candidate[i + 1];
    if (!isfinite(r0) || !isfinite(r1) || !(r1 > r0)) {
      bad = 1;
    } else {
      const double dx = __dsub_rn(r1, r0);
      const double vol = __dsub_rn(ale1d_volume_coordinate(r1, p.geom),
                                   ale1d_volume_coordinate(r0, p.geom));
      if (!(dx > 0.0) || !(vol > 0.0) || !isfinite(vol)) {
        bad = 1;
      }
    }
    if (has_cs != 0) {
      const double dr = __dsub_rn(r_current[i + 1], r_current[i]);
      if (cs[i] > 0.0 && dr > 0.0) {
        dt_current = std_min(dt_current, __ddiv_rn(dr, cs[i]));
      }
    }
  }
  sh_count[t] = count;
  sh_bad[t] = bad;
  sh_dt[t] = dt_current;
  __syncthreads();
  for (int offset = kBlock / 2; offset > 0; offset >>= 1) {
    if (t < offset) {
      sh_count[t] += sh_count[t + offset];
      sh_bad[t] |= sh_bad[t + offset];
      sh_dt[t] = std_min(sh_dt[t], sh_dt[t + offset]);
    }
    __syncthreads();
  }
  if (t != 0) {
    return;
  }
  st->n_protected_final = sh_count[0];
  st->geometry_invalid = sh_bad[0];
  double dt_candidate = CUDART_INF;
  if (has_cs != 0) {
    int k = 0;
    for (int i = 0; i < n; ++i) {
      const double a = r_candidate[i];
      const double b = r_candidate[i + 1];
      if (!(b > a)) {
        continue;
      }
      while (k + 1 < n && r_current[k + 1] <= a) {
        ++k;
      }
      double c_max = 0.0;
      for (int m = k; m < n && r_current[m] < b; ++m) {
        c_max = std_max(c_max, cs[m]);
      }
      if (c_max > 0.0) {
        dt_candidate = std_min(dt_candidate, __ddiv_rn(__dsub_rn(b, a), c_max));
      }
    }
  }
  st->dt_current = (has_cs != 0) ? sh_dt[0] : CUDART_INF;
  st->dt_candidate = dt_candidate;
}

// Stable merge sort of idx[0, k) by key[idx] ascending (key[b] < key[a] takes b first): the order
// std::stable_sort gives with operator<.
__device__ void stable_sort_by_key(int* idx, int* tmp, const double* key, const int k) {
  int* src = idx;
  int* dst = tmp;
  for (int width = 1; width < k; width *= 2) {
    for (int lo = 0; lo < k; lo += 2 * width) {
      const int mid = min(lo + width, k);
      const int hi = min(lo + 2 * width, k);
      int a = lo;
      int b = mid;
      int o = lo;
      while (a < mid && b < hi) {
        if (key[src[b]] < key[src[a]]) {
          dst[o++] = src[b++];
        } else {
          dst[o++] = src[a++];
        }
      }
      while (a < mid) {
        dst[o++] = src[a++];
      }
      while (b < hi) {
        dst[o++] = src[b++];
      }
    }
    int* swap = src;
    src = dst;
    dst = swap;
  }
  if (src != idx) {
    for (int i = 0; i < k; ++i) {
      idx[i] = src[i];
    }
  }
}

struct FloorParams {
  int n;
  int n_features;
  double floor_cm;
  double target_factor;
  int relief_halfwidth_cells;
  double max_growth_factor;
  double rho_eligible;  // 100 rho_floor
};

// The floor path's node mask: nodes 0 and n, and the pinned faces of the features in range.
__global__ void floor_mask_kernel(const DeviceFeature* __restrict__ features, const FloorParams p,
                                  std::uint8_t* __restrict__ pinned) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > p.n) {
    return;
  }
  std::uint8_t value = (j == 0 || j == p.n) ? 1U : 0U;
  for (int f = 0; f < p.n_features; ++f) {
    if (features[f].pinned_face != 0 && features[f].peak_cell_or_face == j) {
      value = 1U;
    }
  }
  pinned[j] = value;
}

// build_min_width_floor_candidate (ale_1d_rezone.cu) in one thread: the offenders (cells below
// the floor except the outer one) by width, stable; for each, the relief window bounded by
// pinned nodes and the outer cell, eligible cells only, its widths raised toward the target and
// rescaled to the window's span eight times, the nodes placed, kept within the old neighbours;
// the first window that keeps the mesh ordered and widens the offender by 0.1 % is the
// candidate. The ordering test counts the old mesh's disordered pairs outside the window
// (prefix counts) instead of scanning a copy.
__global__ void floor_candidate_kernel(const FloorParams p, const double* __restrict__ r,
                                       const double* __restrict__ rho,
                                       const std::uint8_t* __restrict__ pinned,
                                       double* __restrict__ dl, int* __restrict__ offenders,
                                       int* __restrict__ sort_tmp, int* __restrict__ bad_prefix,
                                       double* __restrict__ widths,
                                       double* __restrict__ original,
                                       double* __restrict__ targets,
                                       double* __restrict__ r_candidate, CandidateState* st) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int n = p.n;
  for (int j = 0; j <= n; ++j) {
    r_candidate[j] = r[j];
  }
  st->success = 1;
  st->skip_reason = static_cast<int>(Ale1dSkipReason::None);
  st->no_relief = 0;
  for (int i = 0; i < n; ++i) {
    dl[i] = __dsub_rn(r[i + 1], r[i]);
  }
  double min_dl_before = dl[0];
  for (int i = 1; i < n; ++i) {
    if (dl[i] < min_dl_before) {
      min_dl_before = dl[i];
    }
  }
  if (min_dl_before >= p.floor_cm) {
    return;  // no offender: the current mesh
  }
  int n_offenders = 0;
  for (int i = 0; i < n; ++i) {
    if (i == n - 1) {
      continue;
    }
    if (dl[i] < p.floor_cm) {
      offenders[n_offenders++] = i;
    }
  }
  stable_sort_by_key(offenders, sort_tmp, dl, n_offenders);
  // bad_prefix[j]: the disordered pairs (r_{i-1}, r_i) of the old mesh with i <= j
  bad_prefix[0] = 0;
  for (int j = 1; j <= n; ++j) {
    bad_prefix[j] = bad_prefix[j - 1] + ((r[j] > r[j - 1]) ? 0 : 1);
  }
  const double target_cap = __dmul_rn(p.target_factor, p.floor_cm);
  const double relief_threshold = 1.0 + 1.0e-3;
  for (int o = 0; o < n_offenders; ++o) {
    const int k = offenders[o];
    int w0 = max(0, k - p.relief_halfwidth_cells);
    int w1 = min(n - 1, k + p.relief_halfwidth_cells);
    for (int j = k; j > w0; --j) {
      if (j == n - 1 || pinned[j] != 0U) {
        w0 = j;
        break;
      }
    }
    for (int j = k + 1; j <= w1; ++j) {
      if (j == n - 1 || pinned[j] != 0U) {
        w1 = j - 1;
        break;
      }
    }
    bool window_is_eligible = true;
    for (int i = w0; i <= w1; ++i) {
      if (!(rho[i] > p.rho_eligible)) {
        window_is_eligible = false;
        break;
      }
    }
    if (!window_is_eligible) {
      continue;
    }
    const int count = w1 - w0 + 1;
    double span = 0.0;
    for (int i = 0; i < count; ++i) {
      widths[i] = dl[w0 + i];
      original[i] = dl[w0 + i];
      span = __dadd_rn(span, widths[i]);
    }
    const double target_eff = std_min(target_cap, __ddiv_rn(span, static_cast<double>(count)));
    for (int i = 0; i < count; ++i) {
      targets[i] = core::device_ordered::fmin_like_glibc(
          target_eff, __dmul_rn(original[i], p.max_growth_factor));
    }
    for (int iter = 0; iter < 8; ++iter) {
      for (int i = 0; i < count; ++i) {
        widths[i] = std_max(widths[i], targets[i]);
      }
      double width_sum = 0.0;
      for (int i = 0; i < count; ++i) {
        width_sum = __dadd_rn(width_sum, widths[i]);
      }
      const double scale = __ddiv_rn(span, width_sum);
      for (int i = 0; i < count; ++i) {
        widths[i] = __dmul_rn(widths[i], scale);
      }
    }
    // The window's nodes w0 + 1 .. w1 (w0 and w1 + 1 keep the old radii).
    double position = r[w0];
    for (int i = 0; i + 1 < count; ++i) {
      position = __dadd_rn(position, widths[i]);
      r_candidate[w0 + i + 1] = position;
    }
    for (int j = w0 + 1; j <= w1; ++j) {
      const double c = r_candidate[j];
      if (c > r[j] && c > r[j + 1]) {
        r_candidate[j] = __dsub_rn(r[j + 1], __dmul_rn(0.05, __dsub_rn(r[j + 1], r[j])));
      } else if (c < r[j] && c < r[j - 1]) {
        r_candidate[j] = __dadd_rn(r[j - 1], __dmul_rn(0.05, __dsub_rn(r[j], r[j - 1])));
      }
    }
    // candidate_is_strictly_ordered: the old pairs outside (w0, w1 + 1], the new ones inside.
    bool ordered = (bad_prefix[n] - (bad_prefix[w1 + 1] - bad_prefix[w0])) == 0;
    for (int j = w0 + 1; ordered && j <= w1 + 1; ++j) {
      if (!(r_candidate[j] > r_candidate[j - 1])) {
        ordered = false;
      }
    }
    const double relieved_width = __dsub_rn(r_candidate[k + 1], r_candidate[k]);
    if (ordered && relieved_width >= __dmul_rn(dl[k], relief_threshold)) {
      return;  // the candidate
    }
    for (int j = w0 + 1; j <= w1; ++j) {
      r_candidate[j] = r[j];  // back to the current mesh for the next window
    }
  }
  st->no_relief = 1;
}

// The remap's protected faces: the end nodes, the pinned nodes and the features' clamped peaks.
__global__ void protected_faces_kernel(const std::uint8_t* __restrict__ pinned,
                                       const DeviceFeature* __restrict__ features,
                                       const int n_features, const int n,
                                       std::uint8_t* __restrict__ faces) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > n) {
    return;
  }
  std::uint8_t value = (j == 0 || j == n || pinned[j] != 0U) ? 1U : 0U;
  for (int f = 0; f < n_features; ++f) {
    const int peak = features[f].peak_cell_or_face;
    if (peak >= 0 && min(peak, n) == j) {
      value = 1U;
    }
  }
  faces[j] = value;
}

std::vector<DeviceFeature> device_features(const std::vector<Ale1dFeature>& features) {
  std::vector<DeviceFeature> out;
  out.reserve(features.size());
  for (const Ale1dFeature& feature : features) {
    out.push_back({static_cast<int>(feature.kind), feature.peak_cell_or_face,
                   feature.pinned_face ? 1 : 0, feature.x_center, feature.sigma_x,
                   feature.sigma_r, feature.confidence, feature.target_cells});
  }
  return out;
}

const DeviceFeature* upload_features(const std::vector<Ale1dFeature>& features) {
  const std::vector<DeviceFeature> host = device_features(features);
  auto* d = scratch<DeviceFeature>("ale1d_device:features", host.size());
  if (!host.empty()) {
    cuda_check(cudaMemcpy(d, host.data(), host.size() * sizeof(DeviceFeature),
                          cudaMemcpyHostToDevice),
               "ALE1D device candidate: feature upload failed");
  }
  return d;
}

RezoneParams rezone_params(const core::Config& cfg, const int n, const int n_features,
                           const int geom) {
  const auto& rz = cfg.numerics.ale1d.rezone;
  const auto& ale = cfg.numerics.ale1d;
  RezoneParams p{};
  p.n = n;
  p.n_features = n_features;
  p.outer_fixed_mask = (cfg.numerics.hydro.boundary_1d == "fixed") ? 1 : 0;
  p.outer_fixed_enforce = (cfg.numerics.hydro.boundary_1d == "fixed" ||
                           cfg.numerics.hydro.boundary_1d == "reflect")
                              ? 1
                              : 0;
  p.r_min = cfg.mesh.r_min;
  p.r_max = cfg.mesh.r_max;
  p.w0 = rz.monitor_floor;
  p.w_max = rz.monitor_floor * rz.monitor_wmax_ratio;
  p.smoothing_iterations = rz.monitor_smoothing_iterations;
  p.smooth_across_protected = rz.monitor_smooth_across_protected_faces ? 1 : 0;
  p.min_floor_fraction = rz.min_floor_fraction;
  p.truncation_sigma = rz.gaussian_truncation_sigma;
  p.spatial_enabled =
      (rz.spatial_monitor_enabled && rz.spatial_target_cells_fraction > 0.0) ? 1 : 0;
  p.spatial_fraction = rz.spatial_target_cells_fraction;
  p.spatial_power = rz.spatial_power;
  p.laser_dr_min = rz.laser_spatial_dr_min_cm;
  p.laser_dr_max = rz.laser_spatial_dr_max_cm;
  p.ablation_dr_min = rz.ablation_spatial_dr_min_cm;
  p.ablation_dr_max = rz.ablation_spatial_dr_max_cm;
  p.shock_dr_min = rz.shock_spatial_dr_min_cm;
  p.shock_dr_max = rz.shock_spatial_dr_max_cm;
  p.protected_fraction_max = ale.protected_fraction_max;
  p.min_movable_segment_hard = ale.min_movable_segment_hard;
  p.min_cells = ale.min_cells;
  p.max_disp_mu = ale.max_node_displacement_fraction_mu;
  p.max_disp_r = ale.max_node_displacement_fraction_r;
  p.geom = geom;
  return p;
}

// The current node radii on the device (copy_nodes_or_uniform).
const double* current_nodes(const core::State& state, const RezoneParams& p) {
  if (state.x_r.size() == static_cast<std::size_t>(p.n + 1)) {
    return state.x_r.data();
  }
  auto* r = scratch<double>("ale1d_device:uniform_nodes", static_cast<std::size_t>(p.n + 1));
  uniform_nodes_kernel<<<blocks_for(p.n + 1), kBlock>>>(p.r_min, p.r_max, p.n, r);
  cuda_check(cudaGetLastError(), "ALE1D device candidate: uniform nodes launch failed");
  return r;
}

// The finalize step and the copy of the outcome to the host.
Ale1dDeviceCandidate finish(const core::State& state, const RezoneParams& p, double* r_candidate,
                            std::uint8_t* pinned, const DeviceFeature* features,
                            CandidateState* st) {
  const int n = p.n;
  const double* r_current = current_nodes(state, p);
  const bool has_cs = state.cs.size() == static_cast<std::size_t>(n);
  finalize_kernel<<<1, kBlock>>>(p, r_candidate, pinned, r_current,
                                 has_cs ? state.cs.data() : nullptr, has_cs ? 1 : 0, st);
  cuda_check(cudaGetLastError(), "ALE1D device candidate: finalize launch failed");
  CandidateState host{};
  cuda_check(cudaMemcpy(&host, st, sizeof(host), cudaMemcpyDeviceToHost),
             "ALE1D device candidate: outcome copy failed");
  Ale1dDeviceCandidate out;
  out.r_candidate = r_candidate;
  out.pinned = pinned;
  out.success = host.success != 0;
  out.skip_reason = static_cast<Ale1dSkipReason>(host.skip_reason);
  out.no_relief_available = host.no_relief != 0;
  out.n_protected_nodes = host.n_protected_final;
  out.geometry_valid = host.geometry_invalid == 0;
  out.dt_current = host.dt_current;
  out.dt_candidate = host.dt_candidate;
  out.max_node_displacement_mu = host.max_mu;
  out.max_node_displacement_r = host.max_r;
  out.protected_fraction = host.protected_fraction;
  out.min_movable_segment_size = host.min_movable_segment_size;
  out.features = features;
  out.n_features = p.n_features;
  return out;
}

}  // namespace

Ale1dDeviceCandidate rezone_candidate_device(const core::State& state, const core::Config& cfg,
                                             const std::vector<Ale1dFeature>& features) {
  const int n = effective_cell_count(state, cfg);
  TENRYU_ASSERT(n > 0, "ALE1D device rezone requires cells");
  const int n_features = static_cast<int>(features.size());
  const RezoneParams p = rezone_params(cfg, n, n_features, state.mesh.geometry_code);
  const std::size_t n_nodes = static_cast<std::size_t>(n + 1);
  const std::size_t n_cells = static_cast<std::size_t>(n);
  const std::size_t n_fc = n_cells * static_cast<std::size_t>(std::max(n_features, 1));

  auto* st = scratch<CandidateState>("ale1d_device:state", 1);
  cuda_check(cudaMemset(st, 0, sizeof(CandidateState)), "ALE1D device rezone: state reset failed");
  const DeviceFeature* d_features = upload_features(features);
  auto* node_x = scratch<double>("ale1d_device:node_x", n_nodes);
  auto* cell_x = scratch<double>("ale1d_device:cell_x", n_cells);
  auto* mdx = scratch<double>("ale1d_device:mass_dx", n_cells);
  auto* pinned = scratch<std::uint8_t>("ale1d_device:pinned", n_nodes);
  auto* values = scratch<double>("ale1d_device:monitor_values", n_fc);
  auto* integrand = scratch<double>("ale1d_device:monitor_integrand", n_fc);
  auto* integrals = scratch<double>("ale1d_device:monitor_integrals",
                                    static_cast<std::size_t>(std::max(n_features, 1)));
  auto* amplitudes = scratch<double>("ale1d_device:monitor_amplitudes",
                                     static_cast<std::size_t>(std::max(n_features, 1)));
  auto* u = scratch<double>("ale1d_device:spatial_u", n_cells);
  auto* u_integrand = scratch<double>("ale1d_device:spatial_integrand", n_cells);
  auto* s_integral = scratch<double>("ale1d_device:spatial_integral", 1);
  auto* W = scratch<double>("ale1d_device:monitor_w", n_cells);
  auto* W_next = scratch<double>("ale1d_device:monitor_w_next", n_cells);
  auto* breaks = scratch<int>("ale1d_device:breaks", n_nodes + 1);
  auto* x_new = scratch<double>("ale1d_device:x_new", n_nodes);
  auto* r_candidate = scratch<double>("ale1d_device:r_candidate", n_nodes);

  const bool has_mass = state.mass.size() == n_cells;
  mass_map_kernel<<<1, 1>>>(has_mass ? state.mass.data() : nullptr, has_mass ? 1 : 0, n, node_x,
                            cell_x, mdx);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: mass map launch failed");
  const double* r_old = current_nodes(state, p);
  node_mask_kernel<<<1, 1>>>(d_features, p, node_x, pinned, st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: node mask launch failed");

  // build_monitor
  if (n_features > 0) {
    const long long total = static_cast<long long>(n) * n_features;
    monitor_values_kernel<<<blocks_for(total), kBlock>>>(d_features, p, cell_x, mdx, values,
                                                          integrand);
    cuda_check(cudaGetLastError(), "ALE1D device rezone: monitor values launch failed");
    monitor_integrals_kernel<<<n_features, kSumBlock>>>(integrand, n, integrals);
    cuda_check(cudaGetLastError(), "ALE1D device rezone: monitor integrals launch failed");
  }
  cuda_check(cudaMemset(s_integral, 0, sizeof(double)),
             "ALE1D device rezone: spatial integral reset failed");
  if (p.spatial_enabled != 0) {
    spatial_monitor_kernel<<<blocks_for(n), kBlock>>>(d_features, p, cell_x, mdx, r_old, u,
                                                      u_integrand);
    cuda_check(cudaGetLastError(), "ALE1D device rezone: spatial monitor launch failed");
    ordered_sum_kernel<<<1, kSumBlock>>>(u_integrand, n, s_integral);
    cuda_check(cudaGetLastError(), "ALE1D device rezone: spatial integral launch failed");
  }
  monitor_plan_kernel<<<1, 1>>>(d_features, p, integrals, s_integral, amplitudes, st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: monitor plan launch failed");
  monitor_assemble_kernel<<<blocks_for(n), kBlock>>>(d_features, p, values, amplitudes, integrals,
                                                     u, st, W);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: monitor assembly launch failed");
  double* w_in = W;
  double* w_out = W_next;
  for (int iter = 0; iter < p.smoothing_iterations; ++iter) {
    const int last = (iter + 1 == p.smoothing_iterations) ? 1 : 0;
    monitor_smooth_kernel<<<blocks_for(n), kBlock>>>(p, pinned, w_in, w_out, last);
    cuda_check(cudaGetLastError(), "ALE1D device rezone: monitor smoothing launch failed");
    std::swap(w_in, w_out);
  }
  if (p.smoothing_iterations <= 0) {
    monitor_smooth_kernel<<<blocks_for(n), kBlock>>>(p, pinned, w_in, w_out, -1);
    cuda_check(cudaGetLastError(), "ALE1D device rezone: monitor clamp launch failed");
    std::swap(w_in, w_out);
  }
  const double* W_final = w_in;

  // rezone
  monitor_stats_kernel<<<1, kBlock>>>(W_final, n, st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: monitor stats launch failed");
  rezone_gates_kernel<<<1, 1>>>(p, pinned, breaks, st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: gates launch failed");
  cuda_check(cudaMemcpy(x_new, node_x, n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
             "ALE1D device rezone: x_new copy failed");
  equidistribute_kernel<<<1, kBlock>>>(p, breaks, W_final, mdx, node_x, x_new, st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: equidistribution launch failed");
  candidate_radii_kernel<<<blocks_for(n + 1), kBlock>>>(p, node_x, r_old, x_new, r_candidate, 0,
                                                        st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: candidate radii launch failed");
  displacement_kernel<<<1, kBlock>>>(p, node_x, r_old, x_new, r_candidate, 0, st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: displacement launch failed");
  candidate_radii_kernel<<<blocks_for(n + 1), kBlock>>>(p, node_x, r_old, x_new, r_candidate, 1,
                                                        st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: rescaled radii launch failed");
  displacement_kernel<<<1, kBlock>>>(p, node_x, r_old, x_new, r_candidate, 1, st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: rescaled displacement launch failed");
  rezone_outcome_kernel<<<1, kBlock>>>(n, r_candidate, st);
  cuda_check(cudaGetLastError(), "ALE1D device rezone: outcome launch failed");
  return finish(state, p, r_candidate, pinned, d_features, st);
}

Ale1dDeviceCandidate floor_candidate_device(const core::State& state, const core::Config& cfg,
                                            const std::vector<Ale1dFeature>& features) {
  const int n = effective_cell_count(state, cfg);
  TENRYU_ASSERT(n > 0, "ALE1D device floor candidate requires cells");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n + 1) &&
                    state.rho.size() == static_cast<std::size_t>(n),
                "ALE1D device floor candidate requires the node radii and the density");
  const int n_features = static_cast<int>(features.size());
  const RezoneParams p = rezone_params(cfg, n, n_features, state.mesh.geometry_code);
  const auto& floor = cfg.numerics.ale1d.min_width_floor;
  FloorParams fp{};
  fp.n = n;
  fp.n_features = n_features;
  fp.floor_cm = floor.floor_cm;
  fp.target_factor = floor.target_factor;
  fp.relief_halfwidth_cells = floor.relief_halfwidth_cells;
  fp.max_growth_factor = floor.max_growth_factor;
  fp.rho_eligible = 100.0 * cfg.numerics.floors.rho;
  const std::size_t n_nodes = static_cast<std::size_t>(n + 1);
  const std::size_t n_cells = static_cast<std::size_t>(n);

  auto* st = scratch<CandidateState>("ale1d_device:state", 1);
  cuda_check(cudaMemset(st, 0, sizeof(CandidateState)), "ALE1D device floor: state reset failed");
  const DeviceFeature* d_features = upload_features(features);
  auto* pinned = scratch<std::uint8_t>("ale1d_device:pinned", n_nodes);
  auto* dl = scratch<double>("ale1d_device:floor_dl", n_cells);
  auto* offenders = scratch<int>("ale1d_device:floor_offenders", n_cells);
  auto* sort_tmp = scratch<int>("ale1d_device:floor_sort", n_cells);
  auto* bad_prefix = scratch<int>("ale1d_device:floor_bad_prefix", n_nodes);
  auto* widths = scratch<double>("ale1d_device:floor_widths", n_cells);
  auto* original = scratch<double>("ale1d_device:floor_original", n_cells);
  auto* targets = scratch<double>("ale1d_device:floor_targets", n_cells);
  auto* r_candidate = scratch<double>("ale1d_device:r_candidate", n_nodes);

  floor_mask_kernel<<<blocks_for(n + 1), kBlock>>>(d_features, fp, pinned);
  cuda_check(cudaGetLastError(), "ALE1D device floor: mask launch failed");
  floor_candidate_kernel<<<1, 1>>>(fp, state.x_r.data(), state.rho.data(), pinned, dl, offenders,
                                   sort_tmp, bad_prefix, widths, original, targets, r_candidate,
                                   st);
  cuda_check(cudaGetLastError(), "ALE1D device floor: candidate launch failed");
  // A floor candidate that found no window is not a candidate (the driver's BenefitTooSmall): the
  // finalize step does not run for it.
  return finish(state, p, r_candidate, pinned, d_features, st);
}

const std::uint8_t* protected_faces_device(const Ale1dDeviceCandidate& candidate,
                                           const int n_cells) {
  auto* faces = scratch<std::uint8_t>("ale1d_device:protected_faces",
                                      static_cast<std::size_t>(n_cells + 1));
  protected_faces_kernel<<<blocks_for(n_cells + 1), kBlock>>>(
      candidate.pinned, static_cast<const DeviceFeature*>(candidate.features),
      candidate.n_features, n_cells, faces);
  cuda_check(cudaGetLastError(), "ALE1D device protected faces: launch failed");
  return faces;
}

}  // namespace tenryu::hydro::ale1d
