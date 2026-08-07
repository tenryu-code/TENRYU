#include "coupling/driver_safety_audit.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"

namespace tenryu::coupling {
namespace {

constexpr int kBlockSize = 256;
constexpr int kFlagCount = 6;
constexpr unsigned int kTeNonFiniteBit = 1U << 0;
constexpr unsigned int kTiNonFiniteBit = 1U << 1;
constexpr unsigned int kRhoNonFiniteBit = 1U << 2;
constexpr unsigned int kEeNonFiniteBit = 1U << 3;
constexpr unsigned int kEiNonFiniteBit = 1U << 4;
constexpr unsigned int kHasFiniteTeBit = 1U << 5;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

struct DeviceAuditAccum {
  int flags[kFlagCount];
  double max_te;
};

struct AuditScratch {
  ~AuditScratch() {
    if (d_accum != nullptr) {
      (void)cudaFree(d_accum);
    }
  }

  AuditScratch() = default;
  AuditScratch(const AuditScratch&) = delete;
  AuditScratch& operator=(const AuditScratch&) = delete;

  void ensure_capacity(const std::size_t n_cells) {
    int current_device = 0;
    cuda_check(cudaGetDevice(&current_device),
               "temperature audit cudaGetDevice failed");
    if (d_accum != nullptr && current_device != device_id) {
      cuda_check(cudaFree(d_accum),
                 "temperature audit cudaFree stale accum failed");
      d_accum = nullptr;
      capacity_cells = 0;
    }
    if (d_accum == nullptr) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_accum),
                            sizeof(DeviceAuditAccum)),
                 "temperature audit cudaMalloc accum failed");
      device_id = current_device;
    }
    capacity_cells = std::max(capacity_cells, n_cells);
  }

  DeviceAuditAccum* d_accum = nullptr;
  std::size_t capacity_cells = 0;
  int device_id = -1;
};

AuditScratch& audit_scratch() {
  static AuditScratch scratch;
  return scratch;
}

__device__ double atomic_max_double_all(double* address, const double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value > __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__global__ void temperature_audit_kernel(const double* __restrict__ Te,
                                         const double* __restrict__ Ti,
                                         const double* __restrict__ rho,
                                         const double* __restrict__ ee,
                                         const double* __restrict__ ei,
                                         const std::size_t n_cells,
                                         int* __restrict__ flags_out,
                                         double* __restrict__ max_te_out) {
  __shared__ unsigned int s_flags[kBlockSize];
  __shared__ double s_max_te[kBlockSize];

  const int tid = threadIdx.x;
  unsigned int local_flags = 0U;
  bool has_finite_te = false;
  double local_max_te = -INFINITY;

  for (std::size_t c = static_cast<std::size_t>(blockIdx.x) * blockDim.x + tid;
       c < n_cells;
       c += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const double te = Te[c];
    if (!isfinite(te)) {
      local_flags |= kTeNonFiniteBit;
    } else {
      local_flags |= kHasFiniteTeBit;
      if (!has_finite_te || te > local_max_te) {
        local_max_te = te;
        has_finite_te = true;
      }
    }
    if (!isfinite(Ti[c])) {
      local_flags |= kTiNonFiniteBit;
    }
    if (!isfinite(rho[c])) {
      local_flags |= kRhoNonFiniteBit;
    }
    if (!isfinite(ee[c])) {
      local_flags |= kEeNonFiniteBit;
    }
    if (!isfinite(ei[c])) {
      local_flags |= kEiNonFiniteBit;
    }
  }

  s_flags[tid] = local_flags;
  s_max_te[tid] = has_finite_te ? local_max_te : -INFINITY;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_flags[tid] |= s_flags[tid + stride];
      if (s_max_te[tid + stride] > s_max_te[tid]) {
        s_max_te[tid] = s_max_te[tid + stride];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    const unsigned int flags = s_flags[0];
    if ((flags & kTeNonFiniteBit) != 0U) {
      atomicOr(flags_out + 0, 1);
    }
    if ((flags & kTiNonFiniteBit) != 0U) {
      atomicOr(flags_out + 1, 1);
    }
    if ((flags & kRhoNonFiniteBit) != 0U) {
      atomicOr(flags_out + 2, 1);
    }
    if ((flags & kEeNonFiniteBit) != 0U) {
      atomicOr(flags_out + 3, 1);
    }
    if ((flags & kEiNonFiniteBit) != 0U) {
      atomicOr(flags_out + 4, 1);
    }
    if ((flags & kHasFiniteTeBit) != 0U) {
      atomicOr(flags_out + 5, 1);
      atomic_max_double_all(max_te_out, s_max_te[0]);
    }
  }
}

__global__ void phase_safety_violation_kernel(
    const double* __restrict__ max_te_before_device,
    const int* __restrict__ before_flags,
    const double max_te_before_host,
    const double* __restrict__ max_te_after,
    const int* __restrict__ after_flags,
    const bool nan_fatal,
    const bool overshoot_fatal_enabled,
    const double te_floor,
    const double overshoot_fatal,
    int* __restrict__ violation) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  bool failed = false;
  if (nan_fatal) {
    failed = after_flags[0] != 0 || after_flags[1] != 0 ||
             after_flags[2] != 0 || after_flags[3] != 0 ||
             after_flags[4] != 0;
  }
  if (!failed && overshoot_fatal_enabled) {
    const double max_before =
        max_te_before_device != nullptr
            ? (before_flags[5] != 0 ? *max_te_before_device : 0.0)
            : max_te_before_host;
    const double max_after = after_flags[5] != 0 ? *max_te_after : 0.0;
    const double T_max_n = fmax(max_before, te_floor);
    const double denom = fmax(T_max_n, 1.0e-30);
    const double overshoot_max = fmax((max_after - T_max_n) / denom, 0.0);
    failed = overshoot_max > overshoot_fatal;
  }
  *violation = failed ? 1 : 0;
}

__global__ void initialize_temperature_audit_max_kernel(
    double* __restrict__ max_te_out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *max_te_out = -INFINITY;
  }
}

__global__ void overshoot_metrics_kernel(
    const double* __restrict__ te_values,
    const int n_cells,
    const double t_max_n,
    const double denom,
    OvershootDeviceMetrics* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  const double te = te_values[i];
  if (!isfinite(te)) {
    return;
  }
  const double ratio = (te - t_max_n) / denom;
  if (ratio > 0.0) {
    atomicAdd(&out->count, 1);
    atomicMax(
        reinterpret_cast<unsigned long long*>(&out->max_ratio),
        static_cast<unsigned long long>(__double_as_longlong(ratio)));
  }
}

__global__ void overshoot_metrics_from_audit_kernel(
    const double* __restrict__ te_values,
    const int n_cells,
    const double* __restrict__ max_te_before,
    const int* __restrict__ before_flags,
    const double boundary_temperature,
    int* __restrict__ count,
    double* __restrict__ max_ratio) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  const double te = te_values[i];
  if (!isfinite(te)) {
    return;
  }
  const double te_max_before_value =
      before_flags[5] != 0 ? *max_te_before : 0.0;
  const double t_max_n = fmax(te_max_before_value, boundary_temperature);
  const double denom = fmax(t_max_n, 1.0e-30);
  const double ratio = (te - t_max_n) / denom;
  if (ratio > 0.0) {
    atomicAdd(count, 1);
    atomicMax(
        reinterpret_cast<unsigned long long*>(max_ratio),
        static_cast<unsigned long long>(__double_as_longlong(ratio)));
  }
}

void launch_temperature_audit_to_slots(const core::State& state,
                                       double* d_max_te,
                                       int* d_flags) {
  const std::size_t n_cells = state.rho.size();
  TENRYU_ASSERT(state.Te.size() == n_cells,
                "temperature audit requires Te/rho size match");
  TENRYU_ASSERT(state.Ti.size() == n_cells,
                "temperature audit requires Ti/rho size match");
  TENRYU_ASSERT(state.ee.size() == n_cells,
                "temperature audit requires ee/rho size match");
  TENRYU_ASSERT(state.ei.size() == n_cells,
                "temperature audit requires ei/rho size match");
  TENRYU_ASSERT(d_max_te != nullptr && d_flags != nullptr,
                "temperature audit slot pointers must be non-null");

  cuda_check(cudaMemsetAsync(d_flags, 0, kFlagCount * sizeof(int)),
             "temperature audit slot flags reset failed");
  initialize_temperature_audit_max_kernel<<<1, 1>>>(d_max_te);
  cuda_check(cudaGetLastError(),
             "temperature audit slot max init launch failed");
  if (n_cells == 0U) {
    return;
  }

  const std::size_t raw_blocks =
      (n_cells + static_cast<std::size_t>(kBlockSize) - 1U) /
      static_cast<std::size_t>(kBlockSize);
  const int blocks = static_cast<int>(std::max<std::size_t>(
      1U, std::min<std::size_t>(raw_blocks, 4096U)));
  temperature_audit_kernel<<<blocks, kBlockSize>>>(state.Te.data(),
                                                   state.Ti.data(),
                                                   state.rho.data(),
                                                   state.ee.data(),
                                                   state.ei.data(),
                                                   n_cells,
                                                   d_flags,
                                                   d_max_te);
  cuda_check(cudaGetLastError(), "temperature audit kernel launch failed");
}

}  // namespace

DeviceTemperatureAudit compute_temperature_audit_device(const core::State& state) {
  const std::size_t n_cells = state.rho.size();
  TENRYU_ASSERT(state.Te.size() == n_cells,
                "temperature audit requires Te/rho size match");
  TENRYU_ASSERT(state.Ti.size() == n_cells,
                "temperature audit requires Ti/rho size match");
  TENRYU_ASSERT(state.ee.size() == n_cells,
                "temperature audit requires ee/rho size match");
  TENRYU_ASSERT(state.ei.size() == n_cells,
                "temperature audit requires ei/rho size match");

  DeviceTemperatureAudit out{};
  if (n_cells == 0U) {
    return out;
  }

  AuditScratch& scratch = audit_scratch();
  scratch.ensure_capacity(n_cells);

  DeviceAuditAccum init{};
  init.max_te = -std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpyAsync(scratch.d_accum,
                             &init,
                             sizeof(DeviceAuditAccum),
                             cudaMemcpyHostToDevice),
             "temperature audit init copy failed");

  const std::size_t raw_blocks =
      (n_cells + static_cast<std::size_t>(kBlockSize) - 1U) /
      static_cast<std::size_t>(kBlockSize);
  const int blocks = static_cast<int>(std::max<std::size_t>(
      1U, std::min<std::size_t>(raw_blocks, 4096U)));

  temperature_audit_kernel<<<blocks, kBlockSize>>>(state.Te.data(),
                                                   state.Ti.data(),
                                                   state.rho.data(),
                                                   state.ee.data(),
                                                   state.ei.data(),
                                                   n_cells,
                                                   reinterpret_cast<int*>(
                                                       scratch.d_accum),
                                                   reinterpret_cast<double*>(
                                                       reinterpret_cast<char*>(
                                                           scratch.d_accum) +
                                                       offsetof(DeviceAuditAccum,
                                                                max_te)));
  cuda_check(cudaGetLastError(), "temperature audit kernel launch failed");

  DeviceAuditAccum host{};
  cuda_check(cudaMemcpy(&host,
                        scratch.d_accum,
                        sizeof(DeviceAuditAccum),
                        cudaMemcpyDeviceToHost),
             "temperature audit result copy failed");

  out.te_has_non_finite = host.flags[0] != 0;
  out.ti_has_non_finite = host.flags[1] != 0;
  out.rho_has_non_finite = host.flags[2] != 0;
  out.ee_has_non_finite = host.flags[3] != 0;
  out.ei_has_non_finite = host.flags[4] != 0;
  out.max_te = host.flags[5] != 0 ? host.max_te : 0.0;
  return out;
}

void compute_temperature_audit_device_to_slots(const core::State& state,
                                               double* d_max_te,
                                               int* d_flags) {
  launch_temperature_audit_to_slots(state, d_max_te, d_flags);
}

void compute_phase_safety_violation_device(
    const double* d_max_te_before,
    const int* d_before_flags,
    const double max_te_before_host,
    const double* d_max_te_after,
    const int* d_after_flags,
    const bool nan_fatal,
    const bool overshoot_fatal_enabled,
    const double te_floor,
    const double overshoot_fatal,
    int* d_violation) {
  TENRYU_ASSERT(d_max_te_after != nullptr && d_after_flags != nullptr &&
                    d_violation != nullptr,
                "phase safety slot pointers must be non-null");
  TENRYU_ASSERT(d_max_te_before == nullptr || d_before_flags != nullptr,
                "phase safety staged before flags must accompany max slot");
  phase_safety_violation_kernel<<<1, 1>>>(d_max_te_before,
                                         d_before_flags,
                                         max_te_before_host,
                                         d_max_te_after,
                                         d_after_flags,
                                         nan_fatal,
                                         overshoot_fatal_enabled,
                                         te_floor,
                                         overshoot_fatal,
                                         d_violation);
  cuda_check(cudaGetLastError(),
             "phase safety violation kernel launch failed");
}

void compute_overshoot_metrics_device_to_slots(
    const double* d_te,
    const int n_cells,
    const double* d_max_te_before,
    const int* d_before_flags,
    const double boundary_temperature,
    int* d_count,
    double* d_max_ratio) {
  TENRYU_ASSERT(d_max_te_before != nullptr && d_before_flags != nullptr &&
                    d_count != nullptr && d_max_ratio != nullptr,
                "overshoot metric slot pointers must be non-null");
  cuda_check(cudaMemsetAsync(d_count, 0, sizeof(int)),
             "overshoot metric slot count reset failed");
  cuda_check(cudaMemsetAsync(d_max_ratio, 0, sizeof(double)),
             "overshoot metric slot max reset failed");
  if (n_cells <= 0) {
    return;
  }
  const int grid = (n_cells + kBlockSize - 1) / kBlockSize;
  overshoot_metrics_from_audit_kernel<<<grid, kBlockSize>>>(
      d_te,
      n_cells,
      d_max_te_before,
      d_before_flags,
      boundary_temperature,
      d_count,
      d_max_ratio);
  cuda_check(cudaGetLastError(),
             "overshoot metric slot kernel launch failed");
}

OvershootDeviceMetrics compute_overshoot_metrics_device(
    const double* d_te,
    const int n_cells,
    const double t_max_n,
    const double denom) {
  static_assert(sizeof(OvershootDeviceMetrics) == 16);
  OvershootDeviceMetrics out{};
  if (n_cells <= 0) {
    return out;
  }

  auto* d_out = static_cast<OvershootDeviceMetrics*>(
      core::device_scratch_acquire("driver_safety:overshoot_metrics",
                                   sizeof(OvershootDeviceMetrics)));
  cuda_check(cudaMemset(d_out, 0, sizeof(OvershootDeviceMetrics)),
             "overshoot metrics result reset failed");
  const int grid = (n_cells + kBlockSize - 1) / kBlockSize;
  overshoot_metrics_kernel<<<grid, kBlockSize>>>(
      d_te, n_cells, t_max_n, denom, d_out);
  cuda_check(cudaGetLastError(), "overshoot metrics kernel launch failed");
  cuda_check(cudaMemcpy(&out,
                        d_out,
                        sizeof(OvershootDeviceMetrics),
                        cudaMemcpyDeviceToHost),
             "overshoot metrics result readback failed");
  return out;
}

__global__ void collapse_2t_kernel(const double* __restrict__ te,
                                   const double* __restrict__ ti,
                                   const std::int8_t* __restrict__ active,
                                   const double te_floor,
                                   const int n,
                                   int* __restrict__ counts) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  if (active != nullptr && active[i] == 0) {
    return;
  }
  atomicAdd(counts, 1);
  const double te_v = te[i];
  const double ti_v = ti[i];
  if (!isfinite(te_v) || !isfinite(ti_v)) {
    return;
  }
  const double denom = fmax(fmax(te_v, te_floor), 1.0e-30);
  const double rel = fabs(te_v - ti_v) / denom;
  if (rel < 1.0e-10) {
    atomicAdd(counts + 1, 1);
  }
}

bool all_active_cells_collapsed_device(const core::State& state,
                                       const double te_floor) {
  const int n_cells = static_cast<int>(state.Te.size());
  if (n_cells == 0) {
    return false;
  }
  const std::int8_t* d_active = nullptr;
  if (!state.hydro_active.empty()) {
    auto* buf = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "safety_audit:collapse2t:active",
        static_cast<std::size_t>(n_cells) * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(buf,
                          state.hydro_active.data(),
                          static_cast<std::size_t>(n_cells) *
                              sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "2T collapse mask upload failed");
    d_active = buf;
  }
  auto* d_counts = static_cast<int*>(core::device_scratch_acquire(
      "safety_audit:collapse2t:counts", 2 * sizeof(int)));
  cuda_check(cudaMemset(d_counts, 0, 2 * sizeof(int)),
             "2T collapse counter reset failed");
  constexpr int kBlock = 256;
  const int grid = (n_cells + kBlock - 1) / kBlock;
  collapse_2t_kernel<<<grid, kBlock>>>(state.Te.data(),
                                       state.Ti.data(),
                                       d_active,
                                       te_floor,
                                       n_cells,
                                       d_counts);
  cuda_check(cudaGetLastError(), "2T collapse kernel launch failed");
  int counts[2] = {0, 0};
  cuda_check(cudaMemcpy(counts,
                        d_counts,
                        sizeof(counts),
                        cudaMemcpyDeviceToHost),
             "2T collapse counter readback failed");
  return counts[0] > 0 && counts[1] == counts[0];
}

}  // namespace tenryu::coupling
