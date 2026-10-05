#include "diagnostics/history_1d_gpu.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>
#include <cstring>

#include "core/error.hpp"
#include "core/device_ordered_sum.cuh"

// Compiled with -fmad=false (src/diagnostics/CMakeLists.txt): the products the host added in double
// are rounded before they are added, as on the host.

namespace tenryu::diagnostics::history_1d {
namespace {

constexpr int kThreads = 256;
constexpr int kSumPerThread = 8;

inline void check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

// ---------------------------------------------------------------------------------------------
// double-double sums (the host's long double sums)
//
// A sum or product that is not finite is kept as (value, 0): the error terms of an infinite value
// would be NaN, and the host's long double sum keeps an infinity (inf + x = inf, inf + -inf = NaN).

struct DD {
  double hi;
  double lo;
};

__device__ inline DD two_sum(const double a, const double b) {
  const double s = a + b;
  if (!isfinite(s)) {
    return DD{s, 0.0};
  }
  const double bb = s - a;
  const double err = (a - (s - bb)) + (b - bb);
  return DD{s, err};
}

__device__ inline DD quick_two_sum(const double a, const double b) {
  const double s = a + b;
  if (!isfinite(s)) {
    return DD{s, 0.0};
  }
  return DD{s, b - (s - a)};
}

__device__ inline DD two_prod(const double a, const double b) {
  const double p = a * b;
  if (!isfinite(p)) {
    return DD{p, 0.0};
  }
  return DD{p, __fma_rn(a, b, -p)};
}

__device__ inline DD dd_add(const DD x, const DD y) {
  DD s = two_sum(x.hi, y.hi);
  const DD t = two_sum(x.lo, y.lo);
  s.lo += t.hi;
  s = quick_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  return quick_two_sum(s.hi, s.lo);
}

__device__ inline DD dd_add(const DD x, const double y) {
  DD s = two_sum(x.hi, y);
  s.lo += x.lo;
  return quick_two_sum(s.hi, s.lo);
}

__device__ inline double dd_value(const DD x) { return x.hi + x.lo; }

// a / b rounded to double (one correction step: the error is far below the last place). An
// infinite or NaN quotient or divisor gives the plain quotient (inf / x, x / inf, inf / inf).
__device__ inline double dd_div(const DD a, const DD b) {
  const double q1 = a.hi / b.hi;
  if (!isfinite(q1) || !isfinite(b.hi)) {
    return q1;
  }
  const DD p = two_prod(q1, b.hi);
  const DD pb = dd_add(p, q1 * b.lo);
  const DD r = dd_add(a, DD{-pb.hi, -pb.lo});
  const double q2 = r.hi / b.hi;
  return q1 + q2;
}

// The block's sum of the threads' DD values in a fixed tree; the result in thread 0.
__device__ DD block_dd_sum(DD v, DD* sh) {
  const int t = threadIdx.x;
  sh[t] = v;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      sh[t] = dd_add(sh[t], sh[t + stride]);
    }
    __syncthreads();
  }
  const DD out = sh[0];
  __syncthreads();
  return out;
}

// ---------------------------------------------------------------------------------------------
// maxima with the host's rules

// std::max(m, x) = (m < x) ? x : m: a NaN x is passed over.
__device__ inline double max_keep(const double m, const double x) { return (m < x) ? x : m; }

__device__ double block_max_keep(double v, double* sh) {
  const int t = threadIdx.x;
  sh[t] = v;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      sh[t] = max_keep(sh[t], sh[t + stride]);
    }
    __syncthreads();
  }
  const double out = sh[0];
  __syncthreads();
  return out;
}

// *std::max_element(x, x + n): the first element, replaced by a later one only when larger; so a
// NaN first element stays, a later NaN is passed over, and among equal values (+0 and -0) the
// first wins. The (value, index) pairs are combined keeping the earlier index on equality.
__device__ double block_max_element(const double* x, const int n, double* sh_v, int* sh_i) {
  const int t = threadIdx.x;
  double best = 0.0;
  int best_i = -1;
  for (int c = t; c < n; c += kThreads) {
    const double v = x[c];
    if (v != v) {
      continue;
    }
    if (best_i < 0 || best < v) {
      best = v;
      best_i = c;
    }
  }
  sh_v[t] = best;
  sh_i[t] = best_i;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      const int bi = sh_i[t + stride];
      if (bi >= 0) {
        const int ai = sh_i[t];
        const double a = sh_v[t];
        const double b = sh_v[t + stride];
        if (ai < 0 || a < b || (!(b < a) && bi < ai)) {
          sh_v[t] = b;
          sh_i[t] = bi;
        }
      }
    }
    __syncthreads();
  }
  double out = sh_v[0];
  if (sh_i[0] < 0 || x[0] != x[0]) {
    out = x[0];  // every element NaN, or a NaN first element
  }
  __syncthreads();
  return out;
}

__device__ inline double clamp01(const double x) {
  if (!isfinite(x)) {
    return 0.0;
  }
  const double lo = (0.0 < x) ? x : 0.0;  // std::max(0.0, x)
  return (lo < 1.0) ? lo : 1.0;           // std::min(1.0, lo)
}

// ---------------------------------------------------------------------------------------------
// kernels (one block each)

__global__ void laser_dep_sum_kernel(const double* __restrict__ dep, const int n,
                                     double* __restrict__ out) {
  __shared__ DD sh[kThreads];
  DD s{0.0, 0.0};
  for (int c = threadIdx.x; c < n; c += kThreads) {
    s = dd_add(s, dep[c]);
  }
  s = block_dd_sum(s, sh);
  if (threadIdx.x == 0) {
    out[0] = dd_value(s);
  }
}

__global__ void absorption_r_kernel(const double* __restrict__ dep,
                                    const double* __restrict__ x_r, const int n,
                                    double* __restrict__ out) {
  __shared__ DD sh[kThreads];
  DD weighted{0.0, 0.0};
  DD total{0.0, 0.0};
  for (int c = threadIdx.x; c < n; c += kThreads) {
    const double q = dep[c];
    if (!(q > 0.0)) {
      continue;
    }
    const double r_center = 0.5 * (x_r[c] + x_r[c + 1]);
    weighted = dd_add(weighted, two_prod(q, r_center));
    total = dd_add(total, q);
  }
  weighted = block_dd_sum(weighted, sh);
  total = block_dd_sum(total, sh);
  if (threadIdx.x == 0) {
    out[0] = (total.hi > 0.0) ? dd_div(weighted, total) : 0.0;
  }
}

__global__ void plasma_kernel(const double* __restrict__ zbar, const double* __restrict__ mass,
                              const int n, double* __restrict__ out /* [3] */) {
  __shared__ DD sh[kThreads];
  __shared__ double sh_max[kThreads];
  DD weighted{0.0, 0.0};
  DD total{0.0, 0.0};
  double zmax = 0.0;
  for (int c = threadIdx.x; c < n; c += kThreads) {
    const double z = (zbar[c] < 0.0) ? 0.0 : zbar[c];  // std::max(zbar, 0.0)
    const double w = (mass != nullptr) ? ((mass[c] < 0.0) ? 0.0 : mass[c]) : 1.0;
    weighted = dd_add(weighted, two_prod(w, z));
    total = dd_add(total, w);
    zmax = max_keep(zmax, z);
  }
  weighted = block_dd_sum(weighted, sh);
  total = block_dd_sum(total, sh);
  zmax = block_max_keep(zmax, sh_max);
  if (threadIdx.x == 0) {
    const bool valid = !(total.hi <= 0.0);
    out[0] = valid ? 1.0 : 0.0;
    out[1] = valid ? dd_div(weighted, total) : 0.0;
    out[2] = zmax;
  }
}

__global__ void implosion_kernel(const double* __restrict__ rho, const double* __restrict__ mass,
                                 const double* __restrict__ Te,
                                 const double* __restrict__ centroid_r, const int n,
                                 double* __restrict__ out /* [5] */) {
  __shared__ DD sh[kThreads];
  __shared__ double sh_v[kThreads];
  __shared__ int sh_i[kThreads];
  const double rho_peak = block_max_element(rho, n, sh_v, sh_i);
  const double threshold = 0.1 * rho_peak;
  DD weighted{0.0, 0.0};
  DD total{0.0, 0.0};
  double best = 0.0;  // the least centroid r^2, the first cell having it
  int best_i = -1;
  if (centroid_r != nullptr) {
    for (int c = threadIdx.x; c < n; c += kThreads) {
      const double rc = centroid_r[c];
      const double dist2 = rc * rc;
      if (dist2 < CUDART_INF && (best_i < 0 || dist2 < best)) {  // NaN and inf never win
        best = dist2;
        best_i = c;
      }
      if (rho[c] < threshold) {
        continue;
      }
      const double r = (rc < 0.0) ? 0.0 : rc;  // std::max(centroid_r, 0.0)
      const double w = (mass != nullptr) ? ((mass[c] < 0.0) ? 0.0 : mass[c]) : 1.0;
      weighted = dd_add(weighted, two_prod(w, r));
      total = dd_add(total, w);
    }
  }
  weighted = block_dd_sum(weighted, sh);
  total = block_dd_sum(total, sh);
  sh_v[threadIdx.x] = best;
  sh_i[threadIdx.x] = best_i;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int bi = sh_i[threadIdx.x + stride];
      if (bi >= 0) {
        const int ai = sh_i[threadIdx.x];
        const double a = sh_v[threadIdx.x];
        const double b = sh_v[threadIdx.x + stride];
        if (ai < 0 || b < a || (!(a < b) && bi < ai)) {
          sh_v[threadIdx.x] = b;
          sh_i[threadIdx.x] = bi;
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const bool shell_valid = total.hi > 0.0;
    out[0] = rho_peak;
    out[1] = shell_valid ? 1.0 : 0.0;
    out[2] = shell_valid ? dd_div(weighted, total) : 0.0;
    const int center = (sh_i[0] >= 0) ? sh_i[0] : 0;
    out[3] = (centroid_r != nullptr && Te != nullptr) ? 1.0 : 0.0;
    out[4] = (centroid_r != nullptr && Te != nullptr) ? Te[center] : 0.0;
  }
}

__global__ void shape_kernel(const double* __restrict__ rho, const double* __restrict__ x_r,
                             const double* __restrict__ tracer, const int n, const int shell_only,
                             const double rho_threshold_cfg, double* __restrict__ terms,
                             double* __restrict__ out /* [4] */) {
  __shared__ double sh_v[kThreads];
  __shared__ int sh_i[kThreads];
  __shared__ double sh_values[kThreads * kSumPerThread];
  __shared__ int sh_scan[kThreads];
  const double rho_max = block_max_element(rho, n, sh_v, sh_i);
  const double shell_threshold = 0.1 * rho_max;
  const double sph_threshold =
      (rho_threshold_cfg < shell_threshold) ? shell_threshold : rho_threshold_cfg;
  double radius = 0.0;
  // the areal density's terms (integrate_rhoR_1d_host), then the tracer's
  for (int c = threadIdx.x; c < n; c += kThreads) {
    const double d = x_r[c + 1] - x_r[c];
    const double dr = (d < 0.0) ? 0.0 : d;  // std::max(d, 0.0)
    double term = 0.0;
    double term_hot = 0.0;
    if (!(dr <= 0.0)) {
      const double rho_c = (shell_only != 0 && rho[c] < shell_threshold) ? 0.0 : rho[c] * 1.0;
      term = rho_c * dr;
      if (tracer != nullptr) {
        term_hot = (rho[c] * clamp01(tracer[c])) * dr;
      }
    }
    terms[c] = term;
    terms[n + c] = term_hot;
    if (rho[c] >= sph_threshold) {
      radius = max_keep(radius, x_r[c + 1]);
    }
  }
  radius = block_max_keep(radius, sh_v);
  __syncthreads();
  const double rhoR = core::device_ordered::block_ordered_sum_nonzero<kThreads, kSumPerThread>(
      terms, n, 0.0, sh_values, sh_scan);
  const double rhoR_hot =
      core::device_ordered::block_ordered_sum_nonzero<kThreads, kSumPerThread>(
          terms + n, n, 0.0, sh_values, sh_scan);
  if (threadIdx.x == 0) {
    out[0] = rho_max;
    out[1] = rhoR;
    out[2] = rhoR_hot;
    out[3] = radius;
  }
}

// A small pinned readback buffer and a device output (one per process, deliberately leaked).
struct Readback {
  double* device = nullptr;
  double* host = nullptr;
  double* terms = nullptr;
  std::size_t cap_terms = 0;
};

Readback& readback() {
  static auto* r = [] {
    auto* rb = new Readback();
    check(cudaMalloc(reinterpret_cast<void**>(&rb->device), 8 * sizeof(double)),
          "history_1d readback alloc");
    check(cudaMallocHost(reinterpret_cast<void**>(&rb->host), 8 * sizeof(double)),
          "history_1d readback pinned alloc");
    return rb;
  }();
  return *r;
}

void read_back(const int count) {
  auto& rb = readback();
  check(cudaGetLastError(), "history_1d kernel launch");
  check(cudaMemcpy(rb.host, rb.device, static_cast<std::size_t>(count) * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "history_1d readback");
}

}  // namespace

double laser_dep_sum(const core::State& state) {
  const int n = static_cast<int>(state.laser_dep.size());
  if (n == 0) {
    return 0.0;
  }
  auto& rb = readback();
  laser_dep_sum_kernel<<<1, kThreads>>>(state.laser_dep.data(), n, rb.device);
  read_back(1);
  return rb.host[0];
}

double absorption_weighted_r(const core::State& state) {
  const int n = static_cast<int>(state.laser_dep.size());
  if (n == 0 || state.x_r.size() != static_cast<std::size_t>(n) + 1U) {
    return 0.0;
  }
  auto& rb = readback();
  absorption_r_kernel<<<1, kThreads>>>(state.laser_dep.data(), state.x_r.data(), n, rb.device);
  read_back(1);
  return rb.host[0];
}

Plasma plasma(const core::State& state) {
  Plasma out{};
  const int n = static_cast<int>(state.zbar.size());
  if (n == 0) {
    return out;
  }
  const bool has_mass = state.mass.size() == state.zbar.size();
  auto& rb = readback();
  plasma_kernel<<<1, kThreads>>>(state.zbar.data(), has_mass ? state.mass.data() : nullptr, n,
                                 rb.device);
  read_back(3);
  if (rb.host[0] == 0.0) {
    return out;  // the weights sum to no more than zero: nothing recorded
  }
  out.valid = true;
  out.zbar_mean = rb.host[1];
  out.zbar_max = rb.host[2];
  return out;
}

Implosion implosion(const core::State& state, const double* centroid_r) {
  Implosion out{};
  const int n = static_cast<int>(state.rho.size());
  if (n == 0) {
    return out;
  }
  const bool has_mass = state.mass.size() == state.rho.size();
  const bool has_Te = state.Te.size() == state.rho.size();
  auto& rb = readback();
  implosion_kernel<<<1, kThreads>>>(state.rho.data(), has_mass ? state.mass.data() : nullptr,
                                    has_Te ? state.Te.data() : nullptr, centroid_r, n, rb.device);
  read_back(5);
  out.rho_peak = rb.host[0];
  out.shell_valid = rb.host[1] != 0.0;
  out.shell_radius_mean = rb.host[2];
  out.has_center = rb.host[3] != 0.0;
  out.center_temperature = rb.host[4];
  return out;
}

Shape shape(const core::State& state, const bool shell_only, const double* gas_tracer,
            const double rho_threshold) {
  Shape out{};
  const int n = static_cast<int>(state.rho.size());
  if (n == 0 || state.x_r.size() != static_cast<std::size_t>(n) + 1U) {
    return out;
  }
  auto& rb = readback();
  const std::size_t need = 2U * static_cast<std::size_t>(n);
  if (need > rb.cap_terms) {
    if (rb.terms != nullptr) {
      static_cast<void>(cudaFree(rb.terms));
      rb.terms = nullptr;
    }
    check(cudaMalloc(reinterpret_cast<void**>(&rb.terms), need * sizeof(double)),
          "history_1d terms alloc");
    rb.cap_terms = need;
  }
  shape_kernel<<<1, kThreads>>>(state.rho.data(), state.x_r.data(), gas_tracer, n,
                                shell_only ? 1 : 0, rho_threshold, rb.terms, rb.device);
  read_back(4);
  out.rho_max = rb.host[0];
  out.rhoR = rb.host[1];
  out.rhoR_hotspot = rb.host[2];
  out.shell_radius = rb.host[3];
  return out;
}

}  // namespace tenryu::diagnostics::history_1d
