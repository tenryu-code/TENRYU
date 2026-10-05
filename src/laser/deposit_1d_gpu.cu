#include "laser/deposit_1d_gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "core/device_pack.hpp"
#include "core/error.hpp"

// Compiled with -fmad=false (src/laser/CMakeLists.txt): the arithmetic repeats deposit_transfer.cu
// (apply_deposit_redistribution_1d, distribute_to_stencil_1d, find_active_anchor_1d,
// count_resolved_subcritical_cells_1d, load_cell_mass_1d) operation by operation, and the
// double-double sums need exact two-sum steps.

namespace tenryu::laser::deposit_1d {
namespace {

constexpr int kThreads = 256;
constexpr double kGhostTransitionMinWeightFactor = 0.25;  // deposit_transfer.cu
constexpr double kGhostTransitionMaxWeightFactor = 4.0;
constexpr int kDepositSmoothBoundaryGuardCells = 1;

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

int blocks_for(const int n) { return std::max(1, (n + kThreads - 1) / kThreads); }

__device__ inline double host_max(const double a, const double b) { return (a < b) ? b : a; }
__device__ inline double host_min(const double a, const double b) { return (b < a) ? b : a; }
__device__ inline double host_clamp(const double v, const double lo, const double hi) {
  return (v < lo) ? lo : ((hi < v) ? hi : v);
}

// ---------------------------------------------------------------------------------------------
// double-double sums (the host's long double accumulations)

struct DD {
  double hi;
  double lo;
};

__device__ inline DD quick_two_sum(const double a, const double b) {
  const double s = a + b;
  return DD{s, b - (s - a)};
}

__device__ inline DD two_sum(const double a, const double b) {
  const double s = a + b;
  const double bb = s - a;
  return DD{s, (a - (s - bb)) + (b - bb)};
}

__device__ inline DD dd_add(const DD x, const double y) {
  DD s = two_sum(x.hi, y);
  s.lo += x.lo;
  return quick_two_sum(s.hi, s.lo);
}

__device__ inline DD dd_add(const DD x, const DD y) {
  DD s = two_sum(x.hi, y.hi);
  s.lo += x.lo + y.lo;
  return quick_two_sum(s.hi, s.lo);
}

__device__ inline DD dd_neg(const DD x) { return DD{-x.hi, -x.lo}; }

__device__ inline double dd_value(const DD x) { return x.hi + x.lo; }

// One block: the double-double sum of values[0 .. n) (tree order, deterministic).
__device__ DD block_dd_sum(const double* values, const int n, double* sh_hi, double* sh_lo) {
  const int t = threadIdx.x;
  DD acc{0.0, 0.0};
  for (int i = t; i < n; i += blockDim.x) {
    acc = dd_add(acc, values[i]);
  }
  sh_hi[t] = acc.hi;
  sh_lo[t] = acc.lo;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      const DD s = dd_add(DD{sh_hi[t], sh_lo[t]}, DD{sh_hi[t + stride], sh_lo[t + stride]});
      sh_hi[t] = s.hi;
      sh_lo[t] = s.lo;
    }
    __syncthreads();
  }
  const DD result{sh_hi[0], sh_lo[0]};
  __syncthreads();
  return result;
}

// ---------------------------------------------------------------------------------------------
// accumulation

__global__ void add_kernel(double* __restrict__ total, const double* __restrict__ src, const int n) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) {
    total[c] += src[c];
  }
}

__global__ void assign_divided_kernel(double* __restrict__ total, const double* __restrict__ src,
                                      const double divisor, const int n) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) {
    total[c] = src[c] / divisor;
  }
}

// ---------------------------------------------------------------------------------------------
// redistribution

struct DeviceResult {
  double blocked_power;
  double transition_blend;
  int resolved_cells;
  double sum_input_hi;  // the input sum as a double-double, completed by finish_kernel
  double sum_input_lo;
  double sum_input;
  double conservation_rel;
  int smoothing_ran;
  double smoothing_sum_before;
  double smoothing_rel;
  double energy_sum;
};

// The masks of the redistribution (build_blocked_cell_mask): blocked = void, or supercritical other
// than the allowed cell; receiver_mask = void or supercritical (the receivers of supercritical
// power are the subcritical real cells); and the cell masses (load_cell_mass_1d).
__global__ void masks_kernel(const int n, const std::uint8_t* __restrict__ mask,
                             const double* __restrict__ n_hat, const int allowed_cell,
                             const double* __restrict__ mass, const double* __restrict__ rho,
                             const double* __restrict__ vol, std::uint8_t* __restrict__ blocked,
                             std::uint8_t* __restrict__ receiver_mask,
                             double* __restrict__ cell_mass) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) {
    return;
  }
  const bool is_void = mask != nullptr && mask[c] != 0U;
  if (is_void) {
    blocked[c] = 1U;
    receiver_mask[c] = 1U;
  } else {
    blocked[c] = (n_hat[c] >= 1.0 && c != allowed_cell) ? 1U : 0U;
    receiver_mask[c] = (n_hat[c] >= 1.0) ? 1U : 0U;
  }
  double m = 0.0;
  if (mass != nullptr) {
    m = mass[c];
  } else if (vol != nullptr) {
    m = rho[c] * vol[c];
  }
  cell_mass[c] = (::isfinite(m) && m > 0.0) ? m : 0.0;
}

// For each cell the nearest unmasked cell above and below (-1 when none): one block, a chunk of
// cells per thread and the chunks' first and last unmasked cells combined in order.
__global__ void neighbours_kernel(const int n, const std::uint8_t* __restrict__ masked,
                                  int* __restrict__ next_open, int* __restrict__ prev_open) {
  __shared__ int sh_first[kThreads];
  __shared__ int sh_last[kThreads];
  const int t = threadIdx.x;
  const int chunk = (n + blockDim.x - 1) / blockDim.x;
  const int begin = ::min(n, t * chunk);
  const int end = ::min(n, begin + chunk);
  int first = INT_MAX;
  int last = -1;
  for (int c = begin; c < end; ++c) {
    if (masked[c] == 0U) {
      if (first == INT_MAX) {
        first = c;
      }
      last = c;
    }
  }
  sh_first[t] = first;
  sh_last[t] = last;
  __syncthreads();
  // the first open cell of the chunks above, the last of the chunks below
  int above = INT_MAX;
  for (int u = t + 1; u < static_cast<int>(blockDim.x); ++u) {
    if (sh_first[u] != INT_MAX) {
      above = sh_first[u];
      break;
    }
  }
  int below = -1;
  for (int u = t - 1; u >= 0; --u) {
    if (sh_last[u] >= 0) {
      below = sh_last[u];
      break;
    }
  }
  int running = above;
  for (int c = end - 1; c >= begin; --c) {
    next_open[c] = (running == INT_MAX) ? -1 : running;
    if (masked[c] == 0U) {
      running = c;
    }
  }
  running = below;
  for (int c = begin; c < end; ++c) {
    prev_open[c] = running;
    if (masked[c] == 0U) {
      running = c;
    }
  }
}

struct StencilParams {
  int handoff_cells;
  double handoff_decay;
  double transition_resolved_nhat;
  double transition_density_exponent;
};

// distribute_to_stencil_1d; with use_density the hand-off weights carry the density factor.
__device__ void distribute_to_stencil(double* dep, const std::uint8_t* blocked, const double* n_hat,
                                      const int n, const bool use_density, const int anchor,
                                      const int direction, const double power,
                                      const double transition_blend, const StencilParams& sp) {
  if (!(power > 0.0) || anchor < 0 || anchor >= n) {
    return;
  }
  const int span = ::max(sp.handoff_cells, 1);
  const double decay = host_max(sp.handoff_decay, 1.0e-12);
  const double resolved_nhat_safe = host_max(sp.transition_resolved_nhat, 1.0e-12);
  const double density_exponent = host_max(sp.transition_density_exponent, 0.0);
  double weight_sum = 0.0;
  // the stencil is visited twice instead of stored: the weights are recomputed identically
  for (int k = 0; k < span; ++k) {
    const int idx = anchor + direction * k;
    if (idx < 0 || idx >= n) {
      break;
    }
    if (blocked[idx] != 0U) {
      continue;
    }
    const double w_base = ::exp(-static_cast<double>(k) / decay);
    double w = w_base;
    if (transition_blend > 0.0) {
      double density_factor = 1.0;
      if (density_exponent > 0.0 && use_density) {
        density_factor = ::pow(host_clamp(n_hat[idx] / resolved_nhat_safe,
                                          kGhostTransitionMinWeightFactor,
                                          kGhostTransitionMaxWeightFactor),
                               density_exponent);
      }
      const double w_transition = w_base * density_factor;
      w = (1.0 - transition_blend) * w_base + transition_blend * w_transition;
    }
    weight_sum += w;
  }
  if (!(weight_sum > 0.0)) {
    dep[anchor] += power;
    return;
  }
  for (int k = 0; k < span; ++k) {
    const int idx = anchor + direction * k;
    if (idx < 0 || idx >= n) {
      break;
    }
    if (blocked[idx] != 0U) {
      continue;
    }
    const double w_base = ::exp(-static_cast<double>(k) / decay);
    double w = w_base;
    if (transition_blend > 0.0) {
      double density_factor = 1.0;
      if (density_exponent > 0.0 && use_density) {
        density_factor = ::pow(host_clamp(n_hat[idx] / resolved_nhat_safe,
                                          kGhostTransitionMinWeightFactor,
                                          kGhostTransitionMaxWeightFactor),
                               density_exponent);
      }
      const double w_transition = w_base * density_factor;
      w = (1.0 - transition_blend) * w_base + transition_blend * w_transition;
    }
    dep[idx] += power * (w / weight_sum);
  }
}

// find_active_anchor_1d on the precomputed nearest open cells of the mask.
__device__ int active_anchor(const int c, const int* next_open, const int* prev_open,
                             const bool prefer_outward, const bool allow_opposite_fallback,
                             int* direction) {
  *direction = 0;
  if (prefer_outward) {
    const int outward = next_open[c];
    if (outward >= 0) {
      *direction = 1;
      return outward;
    }
    if (!allow_opposite_fallback) {
      return -1;
    }
    const int inward = prev_open[c];
    if (inward >= 0) {
      *direction = -1;
    }
    return inward;
  }
  const int inward = prev_open[c];
  if (inward >= 0) {
    *direction = -1;
    return inward;
  }
  if (!allow_opposite_fallback) {
    return -1;
  }
  const int outward = next_open[c];
  if (outward >= 0) {
    *direction = 1;
  }
  return outward;
}

// The order-dependent part (one block; the transfers in one thread, in the host's order): the
// input sum, the transition blend, the hand-off of the critical-adjacent cell and the blocked
// cells' power. Only a blocked cell with power moves power, so in a run of cells none of which has
// that when the walk reaches the run no power moves: warp 0 tests 32 cells at a time and lane 0
// visits, in cell order, the cells of the runs that hold such a cell.
__global__ void transfer_kernel(const int n, double* __restrict__ dep,
                                const std::uint8_t* __restrict__ mask,
                                const double* __restrict__ n_hat,
                                const std::uint8_t* __restrict__ blocked,
                                const int* __restrict__ next_blocked_open,
                                const int* __restrict__ prev_blocked_open,
                                const int* __restrict__ next_receiver_open,
                                const int* __restrict__ prev_receiver_open,
                                const laser_map_1d::MapScalars m, const Inputs in,
                                DeviceResult* __restrict__ out) {
  __shared__ double sh_hi[kThreads];
  __shared__ double sh_lo[kThreads];
  __shared__ double sh_transition_blend;
  const DD sum_input = block_dd_sum(dep, n, sh_hi, sh_lo);
  if (threadIdx.x >= 32) {
    return;
  }
  const int lane = static_cast<int>(threadIdx.x);
  const StencilParams sp{in.handoff_cells, in.handoff_decay, in.transition_resolved_nhat,
                         in.transition_density_exponent};
  // Lane 0 computes the blend and the hand-off; the warp then walks the blocked cells.
  double transition_blend = 0.0;
  int resolved_cells = 0;
  if (lane == 0) {
    // the transition blend (count_resolved_subcritical_cells_1d from the outer surface)
    if (in.ghost_enabled != 0 && in.transition_enabled != 0 && mask != nullptr) {
      const int outer = m.outer_surface_cell;
      if (outer >= 0 && in.transition_resolved_nhat > 0.0 && in.n_crit > 0.0) {
        for (int c = outer; c >= 0; --c) {
          if (mask[c] != 0U) {
            continue;
          }
          if (n_hat[c] < in.transition_resolved_nhat) {
            ++resolved_cells;
          } else {
            break;
          }
        }
      }
      const double required_cells = static_cast<double>(::max(in.transition_resolved_cells, 1));
      transition_blend =
          host_clamp(1.0 - static_cast<double>(resolved_cells) / required_cells, 0.0, 1.0);
    }
    // the critical-adjacent subcritical cell hands its power outward
    if (m.fallback_only == 0 && m.critical_adjacent_subcritical_cell >= 0 &&
        m.critical_adjacent_subcritical_cell < n && transition_blend > 0.0) {
      const int anchor = m.critical_adjacent_subcritical_cell;
      const double power = dep[anchor];
      if (power > 0.0) {
        dep[anchor] = 0.0;
        distribute_to_stencil(dep, blocked, n_hat, n, false, anchor, -1, power, 0.0, sp);
      }
    }
    sh_transition_blend = transition_blend;
  }
  __syncwarp();
  transition_blend = sh_transition_blend;
  // blocked cells (void or supercritical) pass their power to the nearest receiver
  DD blocked_power{0.0, 0.0};
  const int* next_open = (m.fallback_only != 0) ? next_blocked_open : next_receiver_open;
  const int* prev_open = (m.fallback_only != 0) ? prev_blocked_open : prev_receiver_open;
  for (int chunk = 0; chunk < n; chunk += 32) {
    const int c_lane = chunk + lane;
    const bool has_power = c_lane < n && blocked[c_lane] != 0U && dep[c_lane] > 0.0;
    const unsigned chunk_power = __ballot_sync(0xffffffffU, has_power);
    if (lane == 0 && chunk_power != 0U) {
      for (int c = chunk; c < ::min(chunk + 32, n); ++c) {
        if (blocked[c] == 0U) {
          continue;
        }
        if (!(dep[c] > 0.0)) {
          continue;
        }
        const bool is_void_source = mask != nullptr && mask[c] != 0U;
        int direction = 0;
        const int target = active_anchor(c, next_open, prev_open, !is_void_source, is_void_source,
                                         &direction);
        if (target >= 0) {
          const double power = dep[c];
          if (is_void_source && in.ghost_enabled != 0) {
            distribute_to_stencil(dep, blocked, n_hat, n, transition_blend > 0.0, target,
                                  direction, power, transition_blend, sp);
          } else {
            dep[target] += power;
          }
        } else {
          blocked_power = dd_add(blocked_power, dep[c]);
        }
        dep[c] = 0.0;
      }
    }
    __syncwarp();
  }
  if (lane != 0) {
    return;
  }
  out->blocked_power = dd_value(blocked_power);
  out->transition_blend = transition_blend;
  out->resolved_cells = resolved_cells;
  out->sum_input_hi = sum_input.hi;
  out->sum_input_lo = sum_input.lo;
  out->sum_input = dd_value(sum_input);
  out->smoothing_ran = 0;
  out->smoothing_sum_before = 0.0;
  out->smoothing_rel = 0.0;
  out->conservation_rel = 0.0;
}

// One smoothing pass (cur -> next): per cell the face below then the face above, as the host's
// loop over the faces applied them.
__global__ void smooth_kernel(const int n, const double* __restrict__ cur, double* __restrict__ next,
                              const std::uint8_t* __restrict__ mask,
                              const std::uint8_t* __restrict__ blocked,
                              const double* __restrict__ cell_mass, const double smooth_alpha) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) {
    return;
  }
  const auto is_void = [&](const int c) { return mask != nullptr && mask[c] != 0U; };
  const auto blocked_or_void = [&](const int c) {
    if (c < 0 || c >= n) {
      return false;
    }
    return is_void(c) || blocked[c] != 0U;
  };
  const auto smoothable = [&](const int c) {
    if (c <= 0 || c >= n - 1) {
      return false;
    }
    if (is_void(c) || blocked[c] != 0U) {
      return false;
    }
    return cell_mass[c] > 0.0;
  };
  const auto guarded = [&](const int c) {
    for (int d = 1; d <= kDepositSmoothBoundaryGuardCells; ++d) {
      if (blocked_or_void(c - d) || blocked_or_void(c + d)) {
        return true;
      }
    }
    return false;
  };
  // the flux of face (left = c, right = c + 1), c in [1, n - 2]; active is false when skipped
  const auto face_flux = [&](const int c, bool* active) {
    *active = false;
    if (c < 1 || c > n - 2) {
      return 0.0;
    }
    const int left = c;
    const int right = c + 1;
    if (!smoothable(left) || !smoothable(right)) {
      return 0.0;
    }
    if (guarded(left) || guarded(right)) {
      return 0.0;
    }
    const double mass_left = cell_mass[left];
    const double mass_right = cell_mass[right];
    if (!(mass_left > 0.0) || !(mass_right > 0.0)) {
      return 0.0;
    }
    const double specific_left = cur[left] / mass_left;
    const double specific_right = cur[right] / mass_right;
    const double face_mass = host_min(mass_left, mass_right);
    if (!(face_mass > 0.0)) {
      return 0.0;
    }
    *active = true;
    return smooth_alpha * face_mass * (specific_right - specific_left);
  };
  double value = cur[k];
  bool active = false;
  const double flux_below = face_flux(k - 1, &active);
  if (active) {
    value -= flux_below;
  }
  const double flux_above = face_flux(k, &active);
  if (active) {
    value += flux_above;
  }
  next[k] = value;
}

// The hot-electron power, the energies of the owned cells, the conservation check.
__global__ void finish_kernel(const int n, double* __restrict__ dep,
                              const double* __restrict__ hot_e_extra, const double dt,
                              const int owned_begin, const int owned_end,
                              double* __restrict__ laser_dep, DeviceResult* __restrict__ out) {
  __shared__ double sh_hi[kThreads];
  __shared__ double sh_lo[kThreads];
  const int t = threadIdx.x;
  // hot-electron power (added where it is not zero) and its sum
  DD extra_acc{0.0, 0.0};
  for (int c = t; c < n; c += blockDim.x) {
    if (hot_e_extra != nullptr) {
      const double extra = hot_e_extra[c];
      if (extra != 0.0) {
        dep[c] += extra;
        extra_acc = dd_add(extra_acc, extra);
      }
    }
  }
  sh_hi[t] = extra_acc.hi;
  sh_lo[t] = extra_acc.lo;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      const DD s = dd_add(DD{sh_hi[t], sh_lo[t]}, DD{sh_hi[t + stride], sh_lo[t + stride]});
      sh_hi[t] = s.hi;
      sh_lo[t] = s.lo;
    }
    __syncthreads();
  }
  const DD hot_e_extra_sum{sh_hi[0], sh_lo[0]};
  __syncthreads();
  const DD sum_cells = block_dd_sum(dep, n, sh_hi, sh_lo);
  for (int c = t; c < n; c += blockDim.x) {
    laser_dep[c] = (c >= owned_begin && c < owned_end) ? dep[c] * dt : 0.0;
  }
  __syncthreads();
  // laser.cu sum_field_energy of the written field (a long double sum on the host)
  const DD energy_sum = block_dd_sum(laser_dep, n, sh_hi, sh_lo);
  if (t == 0) {
    out->energy_sum = dd_value(energy_sum);
    const DD sum_input{out->sum_input_hi, out->sum_input_lo};
    const DD blocked{out->blocked_power, 0.0};
    const DD expected = dd_add(sum_input, hot_e_extra_sum);
    const DD diff = dd_add(expected, dd_neg(dd_add(sum_cells, blocked)));
    const double denom = host_max(::fabs(dd_value(expected)), 1.0e-30);
    out->conservation_rel = ::fabs(dd_value(diff)) / denom;
  }
}

__global__ void smoothing_check_kernel(const int n, const double* __restrict__ before,
                                       const double* __restrict__ after,
                                       DeviceResult* __restrict__ out) {
  __shared__ double sh_hi[kThreads];
  __shared__ double sh_lo[kThreads];
  const DD sum_before = block_dd_sum(before, n, sh_hi, sh_lo);
  const DD sum_after = block_dd_sum(after, n, sh_hi, sh_lo);
  if (threadIdx.x == 0) {
    const double denom = host_max(::fabs(dd_value(sum_before)), 1.0e-30);
    out->smoothing_ran = 1;
    out->smoothing_sum_before = dd_value(sum_before);
    out->smoothing_rel = ::fabs(dd_value(dd_add(sum_before, dd_neg(sum_after)))) / denom;
  }
}

}  // namespace

struct Workspace::Impl {
  int n_cells = 0;
  double* total = nullptr;
  double* fold = nullptr;
  double* scratch = nullptr;  // the smoothing's second buffer and the copy before it
  double* before = nullptr;
  double* cell_mass = nullptr;
  std::uint8_t* blocked = nullptr;
  std::uint8_t* receiver_mask = nullptr;
  int* next_blocked_open = nullptr;
  int* prev_blocked_open = nullptr;
  int* next_receiver_open = nullptr;
  int* prev_receiver_open = nullptr;
  std::size_t cap_total = 0, cap_fold = 0, cap_scratch = 0, cap_before = 0, cap_mass = 0;
  std::size_t cap_blocked = 0, cap_receiver = 0, cap_n1 = 0, cap_n2 = 0, cap_n3 = 0, cap_n4 = 0;
  DeviceResult* result = nullptr;       // device
  DeviceResult* result_host = nullptr;  // pinned

  ~Impl() {
    void* ptrs[] = {total,     fold,          scratch,           before,
                    cell_mass, blocked,       receiver_mask,     next_blocked_open,
                    prev_blocked_open, next_receiver_open, prev_receiver_open, result};
    for (void* p : ptrs) {
      if (p != nullptr) {
        static_cast<void>(cudaFree(p));
      }
    }
    if (result_host != nullptr) {
      static_cast<void>(cudaFreeHost(result_host));
    }
  }
};

Workspace::Workspace() : impl_(new Impl) {}
Workspace::~Workspace() { delete impl_; }

void begin(Workspace& ws, const int n_cells, cudaStream_t stream) {
  auto& w = *ws.impl();
  TENRYU_ASSERT(n_cells > 0, "deposit 1D: no cells");
  ensure(&w.total, &w.cap_total, static_cast<std::size_t>(n_cells), "deposit total alloc");
  w.n_cells = n_cells;
  check(cudaMemsetAsync(w.total, 0, static_cast<std::size_t>(n_cells) * sizeof(double), stream),
        "deposit total reset");
}

void add(Workspace& ws, const double* src, cudaStream_t stream) {
  auto& w = *ws.impl();
  add_kernel<<<blocks_for(w.n_cells), kThreads, 0, stream>>>(w.total, src, w.n_cells);
  check(cudaGetLastError(), "deposit add launch");
}

void assign_divided(Workspace& ws, const double* src, const double divisor, cudaStream_t stream) {
  auto& w = *ws.impl();
  assign_divided_kernel<<<blocks_for(w.n_cells), kThreads, 0, stream>>>(w.total, src, divisor,
                                                                        w.n_cells);
  check(cudaGetLastError(), "deposit assign launch");
}

double* total(const Workspace& ws) { return ws.impl()->total; }

void keep_fold(Workspace& ws, const double* src, cudaStream_t stream) {
  auto& w = *ws.impl();
  ensure(&w.fold, &w.cap_fold, static_cast<std::size_t>(w.n_cells), "deposit fold alloc");
  check(cudaMemcpyAsync(w.fold, src, static_cast<std::size_t>(w.n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream),
        "deposit fold copy");
}

void add_fold(Workspace& ws, cudaStream_t stream) {
  auto& w = *ws.impl();
  TENRYU_ASSERT(w.fold != nullptr, "deposit fold: nothing kept");
  add(ws, w.fold, stream);
}

const double* fold(const Workspace& ws) { return ws.impl()->fold; }

Result redistribute(Workspace& ws, core::State& state, const laser_map_1d::Workspace& map,
                    const laser_map_1d::MapScalars& m, const double* hot_e_extra,
                    const Inputs& in, cudaStream_t stream) {
  auto& w = *ws.impl();
  const int n = w.n_cells;
  TENRYU_ASSERT(n > 0 && static_cast<int>(state.laser_dep.size()) == n &&
                    static_cast<int>(state.rho.size()) == n,
                "deposit 1D redistribution: size mismatch");
  TENRYU_ASSERT(in.dt > 0.0, "deposit 1D redistribution: dt must be positive");
  const std::size_t nn = static_cast<std::size_t>(n);
  ensure(&w.scratch, &w.cap_scratch, nn, "deposit scratch alloc");
  ensure(&w.before, &w.cap_before, nn, "deposit scratch alloc");
  ensure(&w.cell_mass, &w.cap_mass, nn, "deposit mass alloc");
  ensure(&w.blocked, &w.cap_blocked, nn, "deposit mask alloc");
  ensure(&w.receiver_mask, &w.cap_receiver, nn, "deposit mask alloc");
  ensure(&w.next_blocked_open, &w.cap_n1, nn, "deposit neighbour alloc");
  ensure(&w.prev_blocked_open, &w.cap_n2, nn, "deposit neighbour alloc");
  ensure(&w.next_receiver_open, &w.cap_n3, nn, "deposit neighbour alloc");
  ensure(&w.prev_receiver_open, &w.cap_n4, nn, "deposit neighbour alloc");
  if (w.result == nullptr) {
    check(cudaMalloc(reinterpret_cast<void**>(&w.result), sizeof(DeviceResult)),
          "deposit result alloc");
    check(cudaMallocHost(reinterpret_cast<void**>(&w.result_host), sizeof(DeviceResult)),
          "deposit result pinned alloc");
  }
  const std::uint8_t* mask =
      state.cell_is_void.empty() ? nullptr : core::device_cell_is_void(state.cell_is_void);
  const double* n_hat = laser_map_1d::n_hat_approx(map);
  // load_cell_mass_1d: the state's masses, else rho * vol
  const double* mass = (state.mass.size() == nn) ? state.mass.data() : nullptr;
  const double* vol = (mass == nullptr && state.vol.size() == nn) ? state.vol.data() : nullptr;
  masks_kernel<<<blocks_for(n), kThreads, 0, stream>>>(n, mask, n_hat, m.allowed_cell, mass,
                                                       state.rho.data(), vol, w.blocked,
                                                       w.receiver_mask, w.cell_mass);
  check(cudaGetLastError(), "deposit masks launch");
  neighbours_kernel<<<1, kThreads, 0, stream>>>(n, w.blocked, w.next_blocked_open,
                                                w.prev_blocked_open);
  check(cudaGetLastError(), "deposit neighbours launch");
  neighbours_kernel<<<1, kThreads, 0, stream>>>(n, w.receiver_mask, w.next_receiver_open,
                                                w.prev_receiver_open);
  check(cudaGetLastError(), "deposit neighbours launch");
  transfer_kernel<<<1, kThreads, 0, stream>>>(n, w.total, mask, n_hat, w.blocked,
                                              w.next_blocked_open, w.prev_blocked_open,
                                              w.next_receiver_open, w.prev_receiver_open, m, in,
                                              w.result);
  check(cudaGetLastError(), "deposit transfer launch");
  double* dep = w.total;
  if (in.smooth_passes > 0 && in.smooth_alpha > 0.0 && n > 2) {
    check(cudaMemcpyAsync(w.before, dep, nn * sizeof(double), cudaMemcpyDeviceToDevice, stream),
          "deposit smoothing copy");
    double* cur = dep;
    double* next = w.scratch;
    for (int pass = 0; pass < in.smooth_passes; ++pass) {
      smooth_kernel<<<blocks_for(n), kThreads, 0, stream>>>(n, cur, next, mask, w.blocked,
                                                            w.cell_mass, in.smooth_alpha);
      check(cudaGetLastError(), "deposit smoothing launch");
      double* const swap = cur;
      cur = next;
      next = swap;
    }
    if (cur != dep) {
      check(cudaMemcpyAsync(dep, cur, nn * sizeof(double), cudaMemcpyDeviceToDevice, stream),
            "deposit smoothing result copy");
    }
    smoothing_check_kernel<<<1, kThreads, 0, stream>>>(n, w.before, dep, w.result);
    check(cudaGetLastError(), "deposit smoothing check launch");
  }
  finish_kernel<<<1, kThreads, 0, stream>>>(n, dep, hot_e_extra, in.dt, in.owned_begin,
                                            in.owned_end, state.laser_dep.data(), w.result);
  check(cudaGetLastError(), "deposit finish launch");
  check(cudaMemcpyAsync(w.result_host, w.result, sizeof(DeviceResult), cudaMemcpyDeviceToHost,
                        stream),
        "deposit result D2H");
  check(cudaStreamSynchronize(stream), "deposit result sync");
  const DeviceResult& r = *w.result_host;
  Result result;
  result.blocked_power = std::max(0.0, r.blocked_power);
  result.transition_blend = r.transition_blend;
  result.resolved_cells = r.resolved_cells;
  result.sum_input = r.sum_input;
  result.conservation_rel = r.conservation_rel;
  result.smoothing_ran = r.smoothing_ran;
  result.smoothing_sum_before = r.smoothing_sum_before;
  result.smoothing_rel = r.smoothing_rel;
  result.energy_sum = r.energy_sum;
  return result;
}

}  // namespace tenryu::laser::deposit_1d
