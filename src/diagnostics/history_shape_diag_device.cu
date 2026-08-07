#include "diagnostics/history_shape_diag_device.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"

namespace tenryu::diagnostics {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kMaxArealAngles = 16;
constexpr int kSphericitySamples = 64;
constexpr int kPacketDoubles = 128;
constexpr int kDirsDoubles = 52 + 2 * kSphericitySamples;
constexpr int kPacketAxesOk = 0;
constexpr int kPacketRhoMax = 1;
constexpr int kPacketArealBase = 2;
constexpr int kPacketHotspotBase = 18;
constexpr int kPacketRadiusBase = 34;
constexpr int kDirsArealBase = 4;
constexpr int kDirsSphericityBase = 52;
constexpr int AREAL_HOT_OFF = 513 * kMaxArealAngles;
constexpr int ISO_OFF = 2 * 513 * kMaxArealAngles;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

bool history_shape_device_disabled() {
  static const bool disabled = [] {
    const char* v = std::getenv("TENRYU_HISTORY_DIAG_NO_DEVICE");
    return v != nullptr && std::string(v) == "1";
  }();
  return disabled;
}

double legendre_p(const int ell, const double x) {
  if (ell == 0) {
    return 1.0;
  }
  if (ell == 1) {
    return x;
  }

  double p_nm2 = 1.0;
  double p_nm1 = x;
  for (int l = 2; l <= ell; ++l) {
    const double p_n = ((2.0 * l - 1.0) * x * p_nm1 - (l - 1.0) * p_nm2) /
                       static_cast<double>(l);
    p_nm2 = p_nm1;
    p_nm1 = p_n;
  }
  return p_nm1;
}

__device__ inline int device_upper_bound(const double* arr,
                                         const int n,
                                         const double x) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (!(x < arr[mid])) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ inline double clamp01_device(const double x) {
  if (!isfinite(x)) {
    return 0.0;
  }
  const double max0 = ((0.0 < x) ? x : 0.0);
  return ((max0 < 1.0) ? max0 : 1.0);
}

__device__ double sample_rho_2d_device(const double* node_r,
                                       const double* node_z,
                                       const double* rho,
                                       const int nr,
                                       const int nz,
                                       const int n_cells,
                                       const double r,
                                       const double z) {
  if (nr + 1 < 2 || nz + 1 < 2 || n_cells <= 0) {
    return 0.0;
  }
  if (r < node_r[0] || r >= node_r[nr]) {
    return 0.0;
  }
  if (z < node_z[0] || z >= node_z[nz]) {
    return 0.0;
  }

  const int r_it = device_upper_bound(node_r, nr + 1, r);
  const int z_it = device_upper_bound(node_z, nz + 1, z);
  if (r_it == 0 || z_it == 0) {
    return 0.0;
  }

  const int i = r_it - 1;
  const int j = z_it - 1;
  const int idx = i * nz + j;
  if (idx < 0 || idx >= n_cells) {
    return 0.0;
  }
  return rho[idx];
}

__device__ double ray_s_max_device(const double* node_r,
                                   const double* node_z,
                                   const int nr,
                                   const int nz,
                                   const double dir_r,
                                   const double dir_z) {
  if (nr + 1 < 2 || nz + 1 < 2) {
    return 0.0;
  }

  const double r_min = node_r[0];
  const double r_max = node_r[nr];
  const double z_min = node_z[0];
  const double z_max = node_z[nz];

  if (!(r_min <= 0.0 && 0.0 <= r_max && z_min <= 0.0 && 0.0 <= z_max)) {
    return 0.0;
  }

  double s_max = __longlong_as_double(0x7ff0000000000000LL);
  if (dir_r > 1.0e-14) {
    const double candidate = r_max / dir_r;
    s_max = ((candidate < s_max) ? candidate : s_max);
  }
  if (dir_z > 1.0e-14) {
    const double candidate = z_max / dir_z;
    s_max = ((candidate < s_max) ? candidate : s_max);
  }
  if (dir_z < -1.0e-14) {
    const double candidate = z_min / dir_z;
    s_max = ((candidate < s_max) ? candidate : s_max);
  }

  if (!isfinite(s_max) || s_max <= 0.0) {
    return 0.0;
  }
  return s_max;
}

__device__ double apply_shell_device(const double rho_value,
                                     const int shell_only,
                                     const double shell_threshold) {
  if (!shell_only) {
    return rho_value;
  }
  return (rho_value >= shell_threshold) ? rho_value : 0.0;
}

__global__ void shape_extract_axes_kernel(const double* x_r,
                                          const double* x_z,
                                          const int nr,
                                          const int nz,
                                          double* axes_r,
                                          double* axes_z,
                                          double* packet) {
  const int t = threadIdx.x;
  if (blockIdx.x != 0) {
    return;
  }

  const int stride = nz + 1;
  for (int i = t; i <= nr; i += 256) {
    axes_r[i] = x_r[i * stride];
  }
  for (int j = t; j <= nz; j += 256) {
    axes_z[j] = x_z[j];
  }
  __syncthreads();

  if (t != 0) {
    return;
  }

  int ok = 1;
  for (int i = 1; i <= nr; ++i) {
    if (axes_r[i] < axes_r[i - 1]) {
      ok = 0;
    }
  }
  for (int j = 1; j <= nz; ++j) {
    if (axes_z[j] < axes_z[j - 1]) {
      ok = 0;
    }
  }
  packet[kPacketAxesOk] = ok ? 1.0 : 0.0;
}

__global__ void shape_rho_max_kernel(const double* rho,
                                     const int n_cells,
                                     double* packet) {
  __shared__ double tile[256];
  __shared__ double m_shared;
  const int t = threadIdx.x;
  if (blockIdx.x != 0) {
    return;
  }

  if (t == 0) {
    m_shared = rho[0];
  }
  __syncthreads();
  for (int base = 0; base < n_cells; base += 256) {
    const int idx = base + t;
    if (idx < n_cells) {
      tile[t] = rho[idx];
    }
    __syncthreads();
    if (t == 0) {
      double m = m_shared;
      const int n_tile = ((n_cells - base) < 256) ? (n_cells - base) : 256;
      for (int i = 0; i < n_tile; ++i) {
        if (tile[i] > m) {
          m = tile[i];
        }
      }
      m_shared = m;
    }
    __syncthreads();
  }
  if (t == 0) {
    packet[kPacketRhoMax] = m_shared;
  }
}

__global__ void shape_hotspot_mask_kernel(const double* rho,
                                          const double* gas_tracer_Y,
                                          const int n_cells,
                                          double* hot) {
  const int stride = blockDim.x * gridDim.x;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n_cells; c += stride) {
    hot[c] = rho[c] * clamp01_device(gas_tracer_Y[c]);
  }
}

__global__ void shape_gather_areal_kernel(const double* axes_r,
                                          const double* axes_z,
                                          const double* rho,
                                          const double* hot,
                                          const int nr,
                                          const int nz,
                                          const int n_cells,
                                          const double* dirs,
                                          double* samples) {
  constexpr int kSamples = 512;
  const int n_angles = static_cast<int>(dirs[2]);
  const int has_hotspot = (dirs[3] != 0.0) ? 1 : 0;
  const int samples_per_ray = kSamples + 1;
  const int leg_stride = n_angles * samples_per_ray;
  const int total_lanes = leg_stride * (has_hotspot ? 2 : 1);
  const int stride = blockDim.x * gridDim.x;

  for (int lane = blockIdx.x * blockDim.x + threadIdx.x; lane < total_lanes;
       lane += stride) {
    const int leg = lane / leg_stride;
    const int leg_lane = lane - leg * leg_stride;
    const int a = leg_lane / samples_per_ray;
    const int k = leg_lane - a * samples_per_ray;
    const int dir_base = kDirsArealBase + 3 * a;
    const double dir_r = dirs[dir_base + 0];
    const double dir_z = dirs[dir_base + 1];
    const double zero_flag = dirs[dir_base + 2];

    if (zero_flag != 0.0) {
      continue;
    }

    const double s_max = ray_s_max_device(axes_r, axes_z, nr, nz, dir_r, dir_z);
    if (s_max <= 0.0) {
      continue;
    }

    const double ds = s_max / static_cast<double>(kSamples);
    const double* field = (leg == 0) ? rho : hot;
    double value = 0.0;
    if (k == 0) {
      value =
          sample_rho_2d_device(axes_r, axes_z, field, nr, nz, n_cells, 0.0, 0.0);
    } else {
      const double s = ds * static_cast<double>(k);
      const double r = dir_r * s;
      const double z = dir_z * s;
      value = sample_rho_2d_device(axes_r, axes_z, field, nr, nz, n_cells, r, z);
    }

    const int sample_idx =
        ((leg == 0) ? 0 : AREAL_HOT_OFF) + k * kMaxArealAngles + a;
    samples[sample_idx] = value;
  }
}

__global__ void shape_gather_iso_kernel(const double* axes_r,
                                        const double* axes_z,
                                        const double* rho,
                                        const int nr,
                                        const int nz,
                                        const int n_cells,
                                        const double* dirs,
                                        double* samples) {
  constexpr int kSamples = 1024;
  const int samples_per_ray = kSamples + 1;
  const int total_lanes = kSphericitySamples * samples_per_ray;
  const int stride = blockDim.x * gridDim.x;

  for (int lane = blockIdx.x * blockDim.x + threadIdx.x; lane < total_lanes;
       lane += stride) {
    const int s_idx = lane / samples_per_ray;
    const int k = lane - s_idx * samples_per_ray;
    const int dir_base = kDirsSphericityBase + 2 * s_idx;
    const double dir_r = dirs[dir_base + 0];
    const double dir_z = dirs[dir_base + 1];

    const double s_max = ray_s_max_device(axes_r, axes_z, nr, nz, dir_r, dir_z);
    if (s_max <= 0.0) {
      continue;
    }

    const double ds = s_max / static_cast<double>(kSamples);
    double value = 0.0;
    if (k == 0) {
      value =
          sample_rho_2d_device(axes_r, axes_z, rho, nr, nz, n_cells, 0.0, 0.0);
    } else {
      const double s = ds * static_cast<double>(k);
      const double r = dir_r * s;
      const double z = dir_z * s;
      value = sample_rho_2d_device(axes_r, axes_z, rho, nr, nz, n_cells, r, z);
    }

    samples[ISO_OFF + k * kSphericitySamples + s_idx] = value;
  }
}

__global__ void shape_accum_areal_kernel(const double* axes_r,
                                         const double* axes_z,
                                         const int nr,
                                         const int nz,
                                         const double* dirs,
                                         const double* samples,
                                         double* packet) {
  const int t = threadIdx.x;
  const int n_angles = static_cast<int>(dirs[2]);
  if (blockIdx.x != 0 || t >= n_angles) {
    return;
  }

  const int dir_base = kDirsArealBase + 3 * t;
  const double dir_r = dirs[dir_base + 0];
  const double dir_z = dirs[dir_base + 1];
  const double zero_flag = dirs[dir_base + 2];
  const int has_hotspot = (dirs[3] != 0.0) ? 1 : 0;

  if (zero_flag != 0.0) {
    packet[kPacketArealBase + t] = 0.0;
    if (has_hotspot) {
      packet[kPacketHotspotBase + t] = 0.0;
    }
    return;
  }

  const double s_max = ray_s_max_device(axes_r, axes_z, nr, nz, dir_r, dir_z);
  if (s_max <= 0.0) {
    packet[kPacketArealBase + t] = 0.0;
    if (has_hotspot) {
      packet[kPacketHotspotBase + t] = 0.0;
    }
    return;
  }

  constexpr int kSamples = 512;
  const double ds = s_max / static_cast<double>(kSamples);
  const int shell_only = (dirs[0] != 0.0) ? 1 : 0;
  const double shell_threshold = 0.1 * packet[kPacketRhoMax];

  double prev_s = 0.0;
  double prev = apply_shell_device(samples[t], shell_only, shell_threshold);
  double sum = 0.0;

  for (int k = 1; k <= kSamples; ++k) {
    const double s = ds * static_cast<double>(k);
    const double curr = apply_shell_device(
        samples[k * kMaxArealAngles + t], shell_only, shell_threshold);
    sum += 0.5 * (prev + curr) * (s - prev_s);
    prev_s = s;
    prev = curr;
  }

  packet[kPacketArealBase + t] = sum;
  if (has_hotspot) {
    prev_s = 0.0;
    prev = apply_shell_device(samples[AREAL_HOT_OFF + t], 0, shell_threshold);
    sum = 0.0;

    for (int k = 1; k <= kSamples; ++k) {
      const double s = ds * static_cast<double>(k);
      const double curr = apply_shell_device(
          samples[AREAL_HOT_OFF + k * kMaxArealAngles + t], 0, shell_threshold);
      sum += 0.5 * (prev + curr) * (s - prev_s);
      prev_s = s;
      prev = curr;
    }

    packet[kPacketHotspotBase + t] = sum;
  }
}

__global__ void shape_accum_iso_kernel(const double* axes_r,
                                       const double* axes_z,
                                       const int nr,
                                       const int nz,
                                       const double* dirs,
                                       const double* samples,
                                       double* packet) {
  const int s = threadIdx.x;
  const int ray = s;
  if (blockIdx.x != 0 || s >= kSphericitySamples) {
    return;
  }

  const double cfg_thr = dirs[1];
  const double shell_thr = 0.1 * packet[kPacketRhoMax];
  const double rho_threshold = ((cfg_thr < shell_thr) ? shell_thr : cfg_thr);
  const int dir_base = kDirsSphericityBase + 2 * s;
  const double dir_r = dirs[dir_base + 0];
  const double dir_z = dirs[dir_base + 1];
  const double s_max = ray_s_max_device(axes_r, axes_z, nr, nz, dir_r, dir_z);
  if (s_max <= 0.0) {
    packet[kPacketRadiusBase + s] = 0.0;
    return;
  }

  constexpr int kSamples = 1024;
  const double ds = s_max / static_cast<double>(kSamples);

  double prev_s = 0.0;
  double prev_rho = samples[ISO_OFF + s];
  double radius = (prev_rho >= rho_threshold) ? 0.0 : 0.0;

  for (int k = 1; k <= kSamples; ++k) {
    const double s = ds * static_cast<double>(k);
    const double curr_rho = samples[ISO_OFF + k * kSphericitySamples + ray];

    if (curr_rho >= rho_threshold) {
      radius = s;
    }
    if (prev_rho >= rho_threshold && curr_rho < rho_threshold) {
      const double denom = prev_rho - curr_rho;
      if (denom > 1.0e-30) {
        const double frac = (prev_rho - rho_threshold) / denom;
        const double cross = prev_s + frac * (s - prev_s);
        radius = ((radius < cross) ? cross : radius);
      } else {
        radius = ((radius < prev_s) ? prev_s : radius);
      }
    }

    prev_s = s;
    prev_rho = curr_rho;
  }

  packet[kPacketRadiusBase + s] = radius;
}

}  // namespace

bool compute_shape_history_device(const core::State& state,
                                  const core::Config& cfg,
                                  ArealDensityDiagnostics* areal,
                                  SphericityDiagnostics* sphericity) {
  if (history_shape_device_disabled()) {
    return false;
  }
  if (state.mesh.dim != 2) {
    return false;
  }
  if (state.rho.empty()) {
    return false;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0) {
    return false;
  }
  const int n_nodes = (nr + 1) * (nz + 1);
  if (static_cast<int>(state.x_r.size()) != n_nodes ||
      static_cast<int>(state.x_z.size()) != n_nodes) {
    return false;
  }

  std::vector<double> angles_deg = cfg.diagnostics.areal_density.angles_deg;
  if (angles_deg.empty()) {
    angles_deg = {0.0};
  }
  if (angles_deg.size() > static_cast<std::size_t>(kMaxArealAngles)) {
    return false;
  }
  const int n_angles = static_cast<int>(angles_deg.size());
  const int n_cells = static_cast<int>(state.rho.size());
  const bool has_hotspot =
      state.gas_tracer_initialized && state.gas_tracer_Y.size() == state.rho.size();
  std::vector<int> modes = cfg.diagnostics.sphericity.modes;
  if (modes.empty()) {
    modes = {0, 2, 4};
  }

  std::array<double, kDirsDoubles> h_dirs{};
  std::array<double, kSphericitySamples> mu{};
  h_dirs[0] = (cfg.diagnostics.areal_density.r_range == "shell") ? 1.0 : 0.0;
  h_dirs[1] = cfg.diagnostics.sphericity.rho_threshold;
  h_dirs[2] = static_cast<double>(n_angles);
  h_dirs[3] = has_hotspot ? 1.0 : 0.0;

  for (int a = 0; a < n_angles; ++a) {
    const double angle_deg = angles_deg[static_cast<std::size_t>(a)];
    const double theta = angle_deg * kPi / 180.0;
    const double dir_r = std::cos(theta);
    const double dir_z = std::sin(theta);
    const double zero_flag =
        (dir_r <= 0.0 && std::abs(dir_z) <= 1.0e-14) ? 1.0 : 0.0;
    const int dir_base = kDirsArealBase + 3 * a;
    h_dirs[static_cast<std::size_t>(dir_base + 0)] = dir_r;
    h_dirs[static_cast<std::size_t>(dir_base + 1)] = dir_z;
    h_dirs[static_cast<std::size_t>(dir_base + 2)] = zero_flag;
  }

  for (int s = 0; s < kSphericitySamples; ++s) {
    const double mu_s =
        static_cast<double>(s) / static_cast<double>(kSphericitySamples - 1);
    const double theta = std::acos(std::clamp(mu_s, 0.0, 1.0));
    const double angle_deg = theta * 180.0 / kPi;
    const double theta2 = angle_deg * kPi / 180.0;
    const double dir_r = std::cos(theta2);
    const double dir_z = std::sin(theta2);
    mu[static_cast<std::size_t>(s)] = mu_s;
    const int dir_base = kDirsSphericityBase + 2 * s;
    h_dirs[static_cast<std::size_t>(dir_base + 0)] = dir_r;
    h_dirs[static_cast<std::size_t>(dir_base + 1)] = dir_z;
  }

  double* axes_r = static_cast<double*>(core::device_scratch_acquire(
      "diag:shape_axes_r", static_cast<std::size_t>(nr + 1) * sizeof(double)));
  double* axes_z = static_cast<double*>(core::device_scratch_acquire(
      "diag:shape_axes_z", static_cast<std::size_t>(nz + 1) * sizeof(double)));
  double* hotspot = nullptr;
  if (has_hotspot) {
    hotspot = static_cast<double*>(core::device_scratch_acquire(
        "diag:shape_hotspot", static_cast<std::size_t>(n_cells) * sizeof(double)));
  }
  double* dirs = static_cast<double*>(core::device_scratch_acquire(
      "diag:shape_dirs", h_dirs.size() * sizeof(double)));
  double* packet = static_cast<double*>(core::device_scratch_acquire(
      "diag:shape_packet", kPacketDoubles * sizeof(double)));
  double* samples = static_cast<double*>(core::device_scratch_acquire(
      "diag:shape_samples",
      static_cast<std::size_t>(ISO_OFF + 1025 * kSphericitySamples) *
          sizeof(double)));
  double* packet_host = static_cast<double*>(core::host_pinned_scratch_acquire(
      "diag:shape_packet_host", kPacketDoubles * sizeof(double)));

  cuda_check(cudaMemcpy(dirs,
                        h_dirs.data(),
                        h_dirs.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "diagnostics shape: cudaMemcpy dirs failed");

  shape_extract_axes_kernel<<<1, 256>>>(
      state.x_r.data(), state.x_z.data(), nr, nz, axes_r, axes_z, packet);
  cuda_check(cudaGetLastError(), "diagnostics shape: extract axes launch failed");

  shape_rho_max_kernel<<<1, 256>>>(state.rho.data(), n_cells, packet);
  cuda_check(cudaGetLastError(), "diagnostics shape: rho max launch failed");

  if (has_hotspot) {
    const int blocks = (n_cells + 255) / 256;
    shape_hotspot_mask_kernel<<<blocks, 256>>>(
        state.rho.data(), state.gas_tracer_Y.data(), n_cells, hotspot);
    cuda_check(cudaGetLastError(), "diagnostics shape: hotspot mask launch failed");
  }

  const int areal_lanes = n_angles * (512 + 1) * (has_hotspot ? 2 : 1);
  const int areal_blocks = (areal_lanes + 255) / 256;
  shape_gather_areal_kernel<<<areal_blocks, 256>>>(
      axes_r, axes_z, state.rho.data(), hotspot, nr, nz, n_cells, dirs, samples);
  cuda_check(cudaGetLastError(), "diagnostics shape: areal gather launch failed");

  const int iso_lanes = kSphericitySamples * (1024 + 1);
  const int iso_blocks = (iso_lanes + 255) / 256;
  shape_gather_iso_kernel<<<iso_blocks, 256>>>(
      axes_r, axes_z, state.rho.data(), nr, nz, n_cells, dirs, samples);
  cuda_check(cudaGetLastError(), "diagnostics shape: iso gather launch failed");

  shape_accum_areal_kernel<<<1, n_angles>>>(
      axes_r, axes_z, nr, nz, dirs, samples, packet);
  cuda_check(cudaGetLastError(), "diagnostics shape: areal accumulate launch failed");

  shape_accum_iso_kernel<<<1, kSphericitySamples>>>(
      axes_r, axes_z, nr, nz, dirs, samples, packet);
  cuda_check(cudaGetLastError(), "diagnostics shape: iso accumulate launch failed");

  cuda_check(cudaMemcpy(packet_host,
                        packet,
                        kPacketDoubles * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "diagnostics shape: cudaMemcpy packet failed");

  if (packet_host[kPacketAxesOk] == 0.0) {
    return false;
  }

  if (areal != nullptr) {
    areal->angles_deg = angles_deg;
    areal->rhoR.assign(angles_deg.size(), 0.0);
    for (int a = 0; a < n_angles; ++a) {
      areal->rhoR[static_cast<std::size_t>(a)] =
          packet_host[kPacketArealBase + a];
    }
    if (has_hotspot) {
      areal->rhoR_hotspot_tracer.assign(angles_deg.size(), 0.0);
      for (int a = 0; a < n_angles; ++a) {
        areal->rhoR_hotspot_tracer[static_cast<std::size_t>(a)] =
            packet_host[kPacketHotspotBase + a];
      }
    } else {
      areal->rhoR_hotspot_tracer.clear();
    }
    areal->rhoR_fuel_tracer.clear();
  }

  if (sphericity != nullptr) {
    sphericity->modes = modes;
    sphericity->coefficients.assign(modes.size(), 0.0);
    for (std::size_t k = 0; k < sphericity->modes.size(); ++k) {
      const int ell = sphericity->modes[k];
      if (ell < 0) {
        sphericity->coefficients[k] = 0.0;
        continue;
      }

      double integral = 0.0;
      for (int s = 1; s < kSphericitySamples; ++s) {
        const std::size_t s0 = static_cast<std::size_t>(s - 1);
        const std::size_t s1 = static_cast<std::size_t>(s);
        const double dmu = mu[s1] - mu[s0];
        const double f0 =
            packet_host[kPacketRadiusBase + s - 1] * legendre_p(ell, mu[s0]);
        const double f1 =
            packet_host[kPacketRadiusBase + s] * legendre_p(ell, mu[s1]);
        integral += 0.5 * (f0 + f1) * dmu;
      }
      sphericity->coefficients[k] =
          (2.0 * static_cast<double>(ell) + 1.0) * integral;
    }
  }

  return true;
}

}  // namespace tenryu::diagnostics
