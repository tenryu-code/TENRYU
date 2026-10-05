#include "laser/ray_init_1d_gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/device_ordered_sum.cuh"

// Compiled with -fmad=false (src/laser/CMakeLists.txt): the arithmetic repeats ray_init.cu
// (ring_layout_1d, initialize_rays_1d) and beams.cu Beam::profile operation by operation.

namespace tenryu::laser::ray_init_1d {
namespace {

constexpr double kPi = 3.14159265358979323846;  // ray_init.cu
constexpr int kThreads = 256;

inline void check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ inline double host_max(const double a, const double b) { return (a < b) ? b : a; }

enum ProfileModel : int { kGaussian = 0, kSuperGaussian = 1, kFlatTop = 2, kTable = 3 };

struct ProfileParams {
  int model;
  double w0_cm;  // profile_w0_cm
  int m;         // profile_m
  const double* table_r;
  const double* table_I;
  int n_table;
};

// Beam::profile
__device__ double profile(const ProfileParams& p, const double R_cm) {
  const double w0 = host_max(p.w0_cm, 1.0e-30);
  const double x = R_cm / w0;
  if (p.model == kSuperGaussian) {
    const int m = ::max(p.m, 1);
    const double q = ::pow(::fabs(x), 2.0 * static_cast<double>(m));
    return ::exp(-2.0 * q);
  }
  if (p.model == kFlatTop) {
    return (R_cm <= w0) ? 1.0 : 0.0;
  }
  if (p.model == kTable) {
    if (p.n_table < 2) {
      return 0.0;
    }
    if (R_cm <= p.table_r[0]) {
      return p.table_I[0];
    }
    if (R_cm >= p.table_r[p.n_table - 1]) {
      return 0.0;
    }
    int lo_b = 0;
    int hi_b = p.n_table;
    while (lo_b < hi_b) {  // std::upper_bound
      const int mid = lo_b + (hi_b - lo_b) / 2;
      if (p.table_r[mid] <= R_cm) {
        lo_b = mid + 1;
      } else {
        hi_b = mid;
      }
    }
    const int hi = lo_b;
    const int lo = hi - 1;
    const double span = p.table_r[hi] - p.table_r[lo];
    const double a = (span > 0.0) ? (R_cm - p.table_r[lo]) / span : 0.0;
    return p.table_I[lo] + a * (p.table_I[hi] - p.table_I[lo]);
  }
  return ::exp(-2.0 * x * x);  // gaussian
}

constexpr int kSumPerThread = 8;

// One block: the weights, their sum in the rings' order (one thread, from the nonzero weights
// compacted in order), the ring powers.
__global__ void ring_powers_kernel(const ProfileParams p, const RingLayout layout,
                                   const int n_rings, const double beam_power,
                                   double* __restrict__ weights, double* __restrict__ ring_power) {
  __shared__ double s_sum_w;
  __shared__ int s_uniform;
  __shared__ double sh_values[kThreads * kSumPerThread];
  __shared__ int sh_scan[kThreads];
  const int t = threadIdx.x;
  for (int k = t; k < n_rings; k += blockDim.x) {
    const double Rk = layout.dR * (static_cast<double>(k) + 0.5);
    const double area = kPi * (2.0 * static_cast<double>(k) + 1.0) * layout.dR * layout.dR;
    weights[k] = profile(p, Rk * layout.reference_scale) * host_max(area, 0.0);
  }
  __syncthreads();
  const double sum_ordered = core::device_ordered::block_ordered_sum_nonzero<kThreads, kSumPerThread>(
      weights, n_rings, 0.0, sh_values, sh_scan);
  if (t == 0) {
    s_uniform = (sum_ordered > 0.0) ? 0 : 1;
    s_sum_w = (sum_ordered > 0.0) ? sum_ordered : static_cast<double>(n_rings);
  }
  __syncthreads();
  const double sum_w = s_sum_w;
  const bool uniform = s_uniform != 0;
  for (int k = t; k < n_rings; k += blockDim.x) {
    const double w = uniform ? 1.0 : weights[k];
    ring_power[k] = beam_power * (w / sum_w);
  }
}

__global__ void sphere_rays_kernel(const RingLayout layout, const int n_rings,
                                   const double* __restrict__ ring_power, double* __restrict__ R0,
                                   double* __restrict__ Z0, double* __restrict__ vR0,
                                   double* __restrict__ vZ0, double* __restrict__ vA0,
                                   double* __restrict__ power, double* __restrict__ power0) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_rings) {
    return;
  }
  const double Rk = layout.dR * (static_cast<double>(k) + 0.5);
  const double dz = layout.Z_init - layout.z_focus;
  const double L = ::sqrt(Rk * Rk + dz * dz);
  // along the line through the focus, toward -Z
  const double vR = (L > 0.0) ? ((dz >= 0.0) ? (-Rk / L) : (Rk / L)) : 0.0;
  const double vR2 = vR * vR;
  const double vZ = -::sqrt(host_max(0.0, 1.0 - vR2));
  R0[k] = Rk;
  Z0[k] = layout.Z_init;
  vR0[k] = vR;
  vZ0[k] = vZ;
  power[k] = ring_power[k];
  power0[k] = ring_power[k];
  vA0[k] = 0.0;
}

// The device copy of a beam's profile table (uploaded when it changes).
struct TableUpload {
  std::vector<double> r;
  std::vector<double> I;
  double* device = nullptr;  // [r | I]
  std::size_t capacity = 0;
};

TableUpload& table_upload() {
  static auto* upload = new TableUpload();  // deliberately leaked, as the other workspaces
  return *upload;
}

ProfileParams profile_params(const Beam& beam, cudaStream_t stream) {
  ProfileParams p{};
  p.w0_cm = beam.profile_w0_cm;
  p.m = beam.profile_m;
  if (beam.profile_model == "super_gaussian") {
    p.model = kSuperGaussian;
  } else if (beam.profile_model == "flat_top") {
    p.model = kFlatTop;
  } else if (beam.profile_model == "table") {
    p.model = kTable;
  } else {
    p.model = kGaussian;
  }
  if (p.model == kTable) {
    TENRYU_ASSERT(beam.profile_r_cm.size() == beam.profile_I.size(),
                  "1D ray init: profile table size mismatch");
    const std::size_t n = beam.profile_r_cm.size();
    p.n_table = static_cast<int>(n);
    if (n > 0) {
      TableUpload& u = table_upload();
      if (u.r != beam.profile_r_cm || u.I != beam.profile_I || u.device == nullptr) {
        if (u.capacity < 2U * n) {
          if (u.device != nullptr) {
            static_cast<void>(cudaFree(u.device));
            u.device = nullptr;
          }
          check(cudaMalloc(reinterpret_cast<void**>(&u.device), 2U * n * sizeof(double)),
                "1D ray init: profile table alloc");
          u.capacity = 2U * n;
        }
        u.r = beam.profile_r_cm;
        u.I = beam.profile_I;
        check(cudaMemcpyAsync(u.device, u.r.data(), n * sizeof(double), cudaMemcpyHostToDevice,
                              stream),
              "1D ray init: profile table H2D");
        check(cudaMemcpyAsync(u.device + n, u.I.data(), n * sizeof(double),
                              cudaMemcpyHostToDevice, stream),
              "1D ray init: profile table H2D");
      }
      p.table_r = u.device;
      p.table_I = u.device + n;
    }
  }
  return p;
}

}  // namespace

RingLayout ring_layout(const Beam& beam, const double Z_init, const int rays_per_beam) {
  RingLayout out;
  out.Z_init = Z_init;
  out.z_focus = beam.axial_focus_1d();
  const double R_beam =
      std::abs(out.Z_init - out.z_focus) / (2.0 * std::max(beam.f_number, 1.0e-12));
  out.dR = (rays_per_beam > 0) ? (R_beam / static_cast<double>(rays_per_beam)) : 0.0;
  const double launch_to_focus = std::abs(out.Z_init - out.z_focus);
  out.reference_scale =
      (launch_to_focus > 0.0) ? std::abs(out.z_focus) / launch_to_focus : 1.0;
  return out;
}

void ring_powers(const Beam& beam, const RingLayout& layout, const int rays_per_beam,
                 const double beam_power, double* ring_power, cudaStream_t stream) {
  TENRYU_ASSERT(rays_per_beam > 0 && ring_power != nullptr, "1D ray init: no rings");
  const ProfileParams p = profile_params(beam, stream);
  double* weights = static_cast<double*>(core::device_scratch_acquire(
      "ray_init:ring_weights", static_cast<std::size_t>(rays_per_beam) * sizeof(double)));
  ring_powers_kernel<<<1, kThreads, 0, stream>>>(p, layout, rays_per_beam, beam_power, weights,
                                                 ring_power);
  check(cudaGetLastError(), "1D ray init: ring powers launch");
}

void sphere_rays(const RingLayout& layout, const int rays_per_beam, const double* ring_power,
                 double* R0, double* Z0, double* vR0, double* vZ0, double* vA0, double* power,
                 double* power0, cudaStream_t stream) {
  sphere_rays_kernel<<<(rays_per_beam + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
      layout, rays_per_beam, ring_power, R0, Z0, vR0, vZ0, vA0, power, power0);
  check(cudaGetLastError(), "1D ray init: sphere rays launch");
}

}  // namespace tenryu::laser::ray_init_1d
