#include "laser/port_section_chi.hpp"

#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::laser::port_section {
namespace {

constexpr double kPi = 3.14159265358979323846;

inline void chi_device_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void ensure_device_capacity(T** ptr, std::size_t* capacity,
                            const std::size_t needed, const char* message) {
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
  chi_device_check(
      cudaMalloc(reinterpret_cast<void**>(ptr), needed * sizeof(T)), message);
  *capacity = needed;
}

template <typename T>
void copy_to_device_async(T* device, const T* host, const std::size_t count,
                          const cudaStream_t stream, const char* message) {
  if (count > 0) {
    chi_device_check(
        cudaMemcpyAsync(device, host, count * sizeof(T),
                        cudaMemcpyHostToDevice, stream),
        message);
  }
}

struct DeviceVec2 {
  double x;
  double y;
};

struct DeviceVec3 {
  double x;
  double y;
  double z;
};

__device__ double device_dot(const DeviceVec3& a, const DeviceVec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ DeviceVec3 device_to_lab(const double* frame,
                                    const DeviceVec3& local) {
  return {
      frame[0] * local.x + frame[3] * local.y + frame[6] * local.z,
      frame[1] * local.x + frame[4] * local.y + frame[7] * local.z,
      frame[2] * local.x + frame[5] * local.y + frame[8] * local.z};
}

__device__ DeviceVec3 device_lab_pos_dir(const double* frame,
                                         const double theta,
                                         const double phi) {
  const double sin_theta = ::sin(theta);
  const DeviceVec3 local{
      sin_theta * ::cos(phi),
      sin_theta * ::sin(phi),
      ::cos(theta)};
  return device_to_lab(frame, local);
}

__device__ DeviceVec2 device_meridional_direction(const double alpha,
                                                   const int sheet) {
  const double radial = sheet == 0 ? -::cos(alpha) : ::cos(alpha);
  return {radial, ::sin(alpha)};
}

__device__ DeviceVec3 device_lab_dir(const double* frame,
                                     const double theta,
                                     const double phi,
                                     const double alpha,
                                     const int sheet) {
  const double sin_theta = ::sin(theta);
  const double cos_theta = ::cos(theta);
  const double cos_phi = ::cos(phi);
  const double sin_phi = ::sin(phi);
  const DeviceVec3 r_hat{
      sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
  const DeviceVec3 theta_hat{
      cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta};
  const DeviceVec2 meridional =
      device_meridional_direction(alpha, sheet);
  const DeviceVec3 local{
      meridional.x * r_hat.x + meridional.y * theta_hat.x,
      meridional.x * r_hat.y + meridional.y * theta_hat.y,
      meridional.x * r_hat.z + meridional.y * theta_hat.z};
  return device_to_lab(frame, local);
}

__device__ bool device_lookup(const int* offsets,
                              const double* theta,
                              const double* alpha,
                              const double* power,
                              const double* area,
                              const std::uint8_t* in_limiter,
                              const int bin,
                              const double theta_p,
                              double* intensity_out,
                              double* alpha_out) {
  const int begin = offsets[bin];
  const int end = offsets[bin + 1];
  int first = begin;
  while (first < end && in_limiter[first] != 0) {
    ++first;
  }
  if (first == end) {
    return false;
  }

  int last = end - 1;
  while (last >= first && in_limiter[last] != 0) {
    --last;
  }
  if (theta_p > theta[last]) {
    return false;
  }

  if (theta_p <= theta[first]) {
    *intensity_out = power[first] / area[first];
    *alpha_out = alpha[first];
    return true;
  }

  int hi = first + 1;
  while (hi < end &&
         (in_limiter[hi] != 0 || theta[hi] < theta_p)) {
    ++hi;
  }
  if (hi == end) {
    return false;
  }
  if (theta[hi] == theta_p) {
    *intensity_out = power[hi] / area[hi];
    *alpha_out = alpha[hi];
    return true;
  }

  int lo = hi - 1;
  while (in_limiter[lo] != 0) {
    --lo;
  }
  const double weight =
      (theta_p - theta[lo]) / (theta[hi] - theta[lo]);
  const double lower_intensity = power[lo] / area[lo];
  const double upper_intensity = power[hi] / area[hi];
  *intensity_out =
      lower_intensity + weight * (upper_intensity - lower_intensity);
  *alpha_out = alpha[lo] + weight * (alpha[hi] - alpha[lo]);
  return true;
}

__global__ void build_chi_ps_kernel(
    const int* __restrict__ offsets,
    const double* __restrict__ theta,
    const double* __restrict__ alpha,
    const double* __restrict__ power,
    const double* __restrict__ area,
    const std::int32_t* __restrict__ ray_index,
    const std::uint8_t* __restrict__ in_limiter,
    const std::int32_t* __restrict__ ray_bin,
    const double* __restrict__ cell_chi_pref,
    const double* __restrict__ cell_c_a,
    const double* __restrict__ cell_u_r,
    const double* __restrict__ cell_k_bar,
    const std::uint8_t* __restrict__ cell_mask,
    const double* __restrict__ frames9,
    const double* __restrict__ omega_state,
    const std::int16_t* __restrict__ pair_p,
    const std::int16_t* __restrict__ pair_q,
    double* __restrict__ chi,
    unsigned long long* __restrict__ counters,
    const double f_cbet,
    const double alpha_iaw,
    const double k_a_floor,
    const int n_section_phi,
    const int G_ref,
    const int n_impact_bins,
    const int n_pairs,
    const int n_cells) {
  const std::int64_t t =
      blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  const std::int64_t total =
      static_cast<std::int64_t>(n_cells) * n_pairs;
  if (t >= total) {
    return;
  }
  const int cell = static_cast<int>(t / n_pairs);
  const int pair = static_cast<int>(t % n_pairs);
  if (cell >= n_cells) {
    return;
  }
  if (cell_mask[cell] == 0) {
    return;
  }

  const int p = pair_p[pair];
  const int q = pair_q[pair];
  const int port_p = p / G_ref;
  const int reference_p = p % G_ref;
  const int sheet_p = reference_p / n_impact_bins;
  const int bin_p = reference_p % n_impact_bins;
  const int port_q = q / G_ref;
  const int reference_q = q % G_ref;
  const int sheet_q = reference_q / n_impact_bins;

  const int bp = 2 * cell + sheet_p;
  const int seed_begin = offsets[bp];
  const int seed_end = offsets[bp + 1];
  double seed_power = 0.0;
  bool has_seed = false;
  for (int k = seed_begin; k < seed_end; ++k) {
    if (in_limiter[k] != 0) {
      continue;
    }
    if (ray_bin[ray_index[k]] != bin_p) {
      continue;
    }
    has_seed = true;
    seed_power += power[k];
  }

  if (has_seed) {
    atomicAdd(&counters[0], 1ULL);
  }

  double numerator = 0.0;
  double denominator = 0.0;
  bool has_pump = false;
  if (seed_power > 0.0) {
    const double* frame_p = frames9 + 9 * port_p;
    const double* frame_q = frames9 + 9 * port_q;
    for (int k = seed_begin; k < seed_end; ++k) {
      if (in_limiter[k] != 0) {
        continue;
      }
      if (ray_bin[ray_index[k]] != bin_p) {
        continue;
      }
      const double seed_weight = power[k] / seed_power;
      for (int m = 0; m < n_section_phi; ++m) {
        const double phi =
            (static_cast<double>(m) + 0.5) * 2.0 * kPi /
            static_cast<double>(n_section_phi);
        const DeviceVec3 position =
            device_lab_pos_dir(frame_p, theta[k], phi);
        const DeviceVec3 seed_direction =
            device_lab_dir(frame_p, theta[k], phi, alpha[k], sheet_p);

        const double v1 =
            position.x * frame_q[0] + position.y * frame_q[1] +
            position.z * frame_q[2];
        const double v2 =
            position.x * frame_q[3] + position.y * frame_q[4] +
            position.z * frame_q[5];
        const double vb =
            position.x * frame_q[6] + position.y * frame_q[7] +
            position.z * frame_q[8];
        const double theta_q =
            ::acos(::fmin(::fmax(vb, -1.0), 1.0));
        const double phi_q = ::atan2(v2, v1);
        double pump_intensity = 0.0;
        double pump_alpha = 0.0;
        const int bq = 2 * cell + sheet_q;
        if (!device_lookup(offsets, theta, alpha, power, area, in_limiter,
                           bq, theta_q, &pump_intensity, &pump_alpha)) {
          continue;
        }
        has_pump = true;
        denominator += seed_weight * pump_intensity;

        const DeviceVec3 pump_direction =
            device_lab_dir(frame_q, theta_q, phi_q, pump_alpha, sheet_q);
        const double cos_psi =
            device_dot(seed_direction, pump_direction);
        const double mu_p = device_dot(seed_direction, position);
        const double mu_q = device_dot(pump_direction, position);
        const double ka2 = ::fmax(2.0 * (1.0 - cos_psi), 0.0);
        const double ka = cell_k_bar[cell] * ::sqrt(ka2);
        if (ka < k_a_floor * cell_k_bar[cell] || ka <= 0.0) {
          continue;
        }

        const double g =
            (omega_state[q] - omega_state[p] -
             cell_k_bar[cell] * cell_u_r[cell] * (mu_q - mu_p)) /
            (ka * ::fmax(cell_c_a[cell], 1.0e-30));
        const double ga = g * alpha_iaw;
        const double one_g2 = 1.0 - g * g;
        const double Pg = ga / (ga * ga + one_g2 * one_g2);
        const double polarization =
            0.25 * f_cbet * (1.0 + cos_psi * cos_psi);
        numerator +=
            seed_weight * pump_intensity * polarization * Pg;
      }
    }
  }

  if (has_pump) {
    atomicAdd(&counters[1], 1ULL);
  }
  if (denominator > 0.0) {
    chi[t] = cell_chi_pref[cell] * numerator / denominator;
  }
}

__global__ void audit_chi_kernel(
    const double* __restrict__ chi,
    unsigned long long* __restrict__ abs_max_bits,
    unsigned long long* __restrict__ nonzero_count,
    const std::int64_t total) {
  double local_abs_max = 0.0;
  unsigned long long local_nonzero = 0;
  for (std::int64_t index =
           blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const double value = chi[index];
    local_abs_max = ::fmax(local_abs_max, ::fabs(value));
    local_nonzero += value != 0.0 ? 1ULL : 0ULL;
  }

  __shared__ double block_abs_max[256];
  __shared__ unsigned long long block_nonzero[256];
  block_abs_max[threadIdx.x] = local_abs_max;
  block_nonzero[threadIdx.x] = local_nonzero;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_abs_max[threadIdx.x] =
          ::fmax(block_abs_max[threadIdx.x],
                 block_abs_max[threadIdx.x + stride]);
      block_nonzero[threadIdx.x] += block_nonzero[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // Nonnegative doubles have the same ordering as their unsigned bit
    // representations, so this atomic maximum is deterministic.
    atomicMax(
        abs_max_bits,
        static_cast<unsigned long long>(
            __double_as_longlong(block_abs_max[0])));
    atomicAdd(nonzero_count, block_nonzero[0]);
  }
}

}  // namespace

struct ChiDeviceWorkspace::Impl {
  int* d_offsets = nullptr;
  double* d_theta = nullptr;
  double* d_alpha = nullptr;
  double* d_P = nullptr;
  double* d_area = nullptr;
  std::int32_t* d_ray_index = nullptr;
  std::uint8_t* d_in_limiter = nullptr;
  std::int32_t* d_ray_bin = nullptr;
  double* d_cell_chi_pref = nullptr;
  double* d_cell_c_a = nullptr;
  double* d_cell_u_r = nullptr;
  double* d_cell_k_bar = nullptr;
  std::uint8_t* d_cell_mask = nullptr;
  double* d_frames9 = nullptr;
  double* d_omega_state = nullptr;
  double* d_weight_state = nullptr;
  std::int16_t* d_pair_p = nullptr;
  std::int16_t* d_pair_q = nullptr;
  double* d_chi = nullptr;
  unsigned long long* d_readback = nullptr;

  std::size_t cap_offsets = 0;
  std::size_t cap_theta = 0;
  std::size_t cap_alpha = 0;
  std::size_t cap_P = 0;
  std::size_t cap_area = 0;
  std::size_t cap_ray_index = 0;
  std::size_t cap_in_limiter = 0;
  std::size_t cap_ray_bin = 0;
  std::size_t cap_cell_chi_pref = 0;
  std::size_t cap_cell_c_a = 0;
  std::size_t cap_cell_u_r = 0;
  std::size_t cap_cell_k_bar = 0;
  std::size_t cap_cell_mask = 0;
  std::size_t cap_frames9 = 0;
  std::size_t cap_omega_state = 0;
  std::size_t cap_weight_state = 0;
  std::size_t cap_pair_p = 0;
  std::size_t cap_pair_q = 0;
  std::size_t cap_chi = 0;
  std::size_t cap_readback = 0;

  sector_ps::FlatTable flat;
  std::vector<double> frames9;
  std::vector<double> omega_state;
  std::vector<double> weight_state;
  std::vector<std::int16_t> pair_p;
  std::vector<std::int16_t> pair_q;
  std::vector<double> port_delta_lambda_nm;
  std::vector<double> port_power_weight;
  std::array<unsigned long long, 4> host_readback{};

  bool initialized = false;
  int G_ps = 0;
  int n_pairs = 0;
  int n_ports = 0;
  int n_impact_bins = 0;
  double lambda0_nm = 0.0;

  ~Impl() {
    if (d_offsets != nullptr) {
      static_cast<void>(cudaFree(d_offsets));
    }
    if (d_theta != nullptr) {
      static_cast<void>(cudaFree(d_theta));
    }
    if (d_alpha != nullptr) {
      static_cast<void>(cudaFree(d_alpha));
    }
    if (d_P != nullptr) {
      static_cast<void>(cudaFree(d_P));
    }
    if (d_area != nullptr) {
      static_cast<void>(cudaFree(d_area));
    }
    if (d_ray_index != nullptr) {
      static_cast<void>(cudaFree(d_ray_index));
    }
    if (d_in_limiter != nullptr) {
      static_cast<void>(cudaFree(d_in_limiter));
    }
    if (d_ray_bin != nullptr) {
      static_cast<void>(cudaFree(d_ray_bin));
    }
    if (d_cell_chi_pref != nullptr) {
      static_cast<void>(cudaFree(d_cell_chi_pref));
    }
    if (d_cell_c_a != nullptr) {
      static_cast<void>(cudaFree(d_cell_c_a));
    }
    if (d_cell_u_r != nullptr) {
      static_cast<void>(cudaFree(d_cell_u_r));
    }
    if (d_cell_k_bar != nullptr) {
      static_cast<void>(cudaFree(d_cell_k_bar));
    }
    if (d_cell_mask != nullptr) {
      static_cast<void>(cudaFree(d_cell_mask));
    }
    if (d_frames9 != nullptr) {
      static_cast<void>(cudaFree(d_frames9));
    }
    if (d_omega_state != nullptr) {
      static_cast<void>(cudaFree(d_omega_state));
    }
    if (d_weight_state != nullptr) {
      static_cast<void>(cudaFree(d_weight_state));
    }
    if (d_pair_p != nullptr) {
      static_cast<void>(cudaFree(d_pair_p));
    }
    if (d_pair_q != nullptr) {
      static_cast<void>(cudaFree(d_pair_q));
    }
    if (d_chi != nullptr) {
      static_cast<void>(cudaFree(d_chi));
    }
    if (d_readback != nullptr) {
      static_cast<void>(cudaFree(d_readback));
    }
  }
};

ChiDeviceWorkspace::ChiDeviceWorkspace() : impl_(std::make_unique<Impl>()) {}

ChiDeviceWorkspace::~ChiDeviceWorkspace() = default;

ChiBuildDeviceView build_chi_ps_device_ws(
    const ChiBuildInput& input, const ChiDeviceCellFields& dev_fields,
    ChiDeviceWorkspace& ws, void* cuda_stream) {
  ChiDeviceWorkspace::Impl* const workspace = ws.impl();
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const std::int64_t G_ref =
      2 * static_cast<std::int64_t>(input.n_impact_bins);
  const std::int64_t G_ps_64 =
      static_cast<std::int64_t>(input.ports->ports.size()) * G_ref;
  const std::int64_t n_pairs_64 = G_ps_64 * (G_ps_64 - 1) / 2;
  const std::string pair_cap_message =
      "port_section v1 supports at most 65536 expanded pairs (got " +
      std::to_string(n_pairs_64) +
      "): reduce ports or bins (OMEGA/NIF-scale needs the S2+ "
      "class/tiling path)";
  TENRYU_ASSERT(n_pairs_64 <= 65536, pair_cap_message);

  const int G_ps = static_cast<int>(G_ps_64);
  const int n_pairs = static_cast<int>(n_pairs_64);
  const int n_ports = static_cast<int>(input.ports->ports.size());
  if (!workspace->initialized) {
    workspace->G_ps = G_ps;
    workspace->n_pairs = n_pairs;
    workspace->n_ports = n_ports;
    workspace->n_impact_bins = input.n_impact_bins;
    workspace->lambda0_nm = input.lambda0_nm;

    workspace->port_delta_lambda_nm.reserve(
        input.ports->ports.size());
    workspace->port_power_weight.reserve(input.ports->ports.size());
    for (const port_geom::Port& port : input.ports->ports) {
      workspace->port_delta_lambda_nm.push_back(port.delta_lambda_nm);
      workspace->port_power_weight.push_back(port.power_weight);
    }

    workspace->frames9.reserve(input.ports->frames.size() * 9);
    for (const port_geom::PortFrame& frame : input.ports->frames) {
      workspace->frames9.insert(workspace->frames9.end(),
                                frame.e1.begin(), frame.e1.end());
      workspace->frames9.insert(workspace->frames9.end(),
                                frame.e2.begin(), frame.e2.end());
      workspace->frames9.insert(workspace->frames9.end(),
                                frame.b.begin(), frame.b.end());
    }

    workspace->omega_state.resize(static_cast<std::size_t>(G_ps));
    workspace->weight_state.resize(static_cast<std::size_t>(G_ps));
    for (int state = 0; state < G_ps; ++state) {
      const int port_index = state / static_cast<int>(G_ref);
      const double lambda_cm =
          (input.lambda0_nm +
           input.ports->ports[static_cast<std::size_t>(port_index)]
               .delta_lambda_nm) *
          1.0e-7;
      workspace->omega_state[static_cast<std::size_t>(state)] =
          2.0 * kPi * core::constants::c_light / lambda_cm;
      workspace->weight_state[static_cast<std::size_t>(state)] =
          input.ports->ports[static_cast<std::size_t>(port_index)]
              .power_weight;
    }

    workspace->pair_p.reserve(static_cast<std::size_t>(n_pairs));
    workspace->pair_q.reserve(static_cast<std::size_t>(n_pairs));
    for (int p = 0; p < G_ps; ++p) {
      for (int q = p + 1; q < G_ps; ++q) {
        workspace->pair_p.push_back(static_cast<std::int16_t>(p));
        workspace->pair_q.push_back(static_cast<std::int16_t>(q));
      }
    }

    ensure_device_capacity(
        &workspace->d_frames9, &workspace->cap_frames9,
        workspace->frames9.size(), "chi ws frames allocation failed");
    ensure_device_capacity(
        &workspace->d_omega_state, &workspace->cap_omega_state,
        workspace->omega_state.size(), "chi ws omega allocation failed");
    ensure_device_capacity(
        &workspace->d_weight_state, &workspace->cap_weight_state,
        workspace->weight_state.size(), "chi ws weight allocation failed");
    ensure_device_capacity(
        &workspace->d_pair_p, &workspace->cap_pair_p,
        workspace->pair_p.size(), "chi ws pair_p allocation failed");
    ensure_device_capacity(
        &workspace->d_pair_q, &workspace->cap_pair_q,
        workspace->pair_q.size(), "chi ws pair_q allocation failed");
    copy_to_device_async(
        workspace->d_frames9, workspace->frames9.data(),
        workspace->frames9.size(), stream, "chi ws frames upload failed");
    copy_to_device_async(
        workspace->d_omega_state, workspace->omega_state.data(),
        workspace->omega_state.size(), stream, "chi ws omega upload failed");
    copy_to_device_async(
        workspace->d_weight_state, workspace->weight_state.data(),
        workspace->weight_state.size(), stream, "chi ws weight upload failed");
    copy_to_device_async(
        workspace->d_pair_p, workspace->pair_p.data(),
        workspace->pair_p.size(), stream, "chi ws pair_p upload failed");
    copy_to_device_async(
        workspace->d_pair_q, workspace->pair_q.data(),
        workspace->pair_q.size(), stream, "chi ws pair_q upload failed");
    workspace->initialized = true;
  } else {
    bool unchanged =
        workspace->G_ps == G_ps &&
        workspace->n_pairs == n_pairs &&
        workspace->n_ports == n_ports &&
        workspace->n_impact_bins == input.n_impact_bins &&
        workspace->lambda0_nm == input.lambda0_nm &&
        workspace->port_delta_lambda_nm.size() ==
            input.ports->ports.size() &&
        workspace->port_power_weight.size() == input.ports->ports.size();
    for (std::size_t port_index = 0;
         unchanged && port_index < input.ports->ports.size(); ++port_index) {
      unchanged =
          workspace->port_delta_lambda_nm[port_index] ==
              input.ports->ports[port_index].delta_lambda_nm &&
          workspace->port_power_weight[port_index] ==
              input.ports->ports[port_index].power_weight;
    }
    TENRYU_ASSERT(unchanged, "chi ws: port geometry changed mid-run");
  }

  sector_ps::flatten_table_into(*input.table, workspace->flat);
  TENRYU_ASSERT(workspace->flat.n_shells >= input.n_cells,
                "chi device: table/shell count mismatch");

  ensure_device_capacity(
      &workspace->d_offsets, &workspace->cap_offsets,
      workspace->flat.offsets.size(), "chi ws offsets allocation failed");
  ensure_device_capacity(
      &workspace->d_theta, &workspace->cap_theta,
      workspace->flat.theta.size(), "chi ws theta allocation failed");
  ensure_device_capacity(
      &workspace->d_alpha, &workspace->cap_alpha,
      workspace->flat.alpha.size(), "chi ws alpha allocation failed");
  ensure_device_capacity(
      &workspace->d_P, &workspace->cap_P, workspace->flat.P.size(),
      "chi ws power allocation failed");
  ensure_device_capacity(
      &workspace->d_area, &workspace->cap_area,
      workspace->flat.area.size(), "chi ws area allocation failed");
  ensure_device_capacity(
      &workspace->d_ray_index, &workspace->cap_ray_index,
      workspace->flat.ray_index.size(),
      "chi ws ray_index allocation failed");
  ensure_device_capacity(
      &workspace->d_in_limiter, &workspace->cap_in_limiter,
      workspace->flat.in_limiter.size(),
      "chi ws limiter allocation failed");
  ensure_device_capacity(
      &workspace->d_ray_bin, &workspace->cap_ray_bin,
      static_cast<std::size_t>(input.n_rays),
      "chi ws ray_bin allocation failed");

  copy_to_device_async(
      workspace->d_offsets, workspace->flat.offsets.data(),
      workspace->flat.offsets.size(), stream, "chi ws offsets upload failed");
  copy_to_device_async(
      workspace->d_theta, workspace->flat.theta.data(),
      workspace->flat.theta.size(), stream, "chi ws theta upload failed");
  copy_to_device_async(
      workspace->d_alpha, workspace->flat.alpha.data(),
      workspace->flat.alpha.size(), stream, "chi ws alpha upload failed");
  copy_to_device_async(
      workspace->d_P, workspace->flat.P.data(),
      workspace->flat.P.size(), stream, "chi ws power upload failed");
  copy_to_device_async(
      workspace->d_area, workspace->flat.area.data(),
      workspace->flat.area.size(), stream, "chi ws area upload failed");
  copy_to_device_async(
      workspace->d_ray_index, workspace->flat.ray_index.data(),
      workspace->flat.ray_index.size(), stream,
      "chi ws ray_index upload failed");
  copy_to_device_async(
      workspace->d_in_limiter, workspace->flat.in_limiter.data(),
      workspace->flat.in_limiter.size(), stream,
      "chi ws limiter upload failed");
  copy_to_device_async(
      workspace->d_ray_bin, input.ray_bin,
      static_cast<std::size_t>(input.n_rays), stream,
      "chi ws ray_bin upload failed");

  const double* d_cell_chi_pref = dev_fields.d_chi_pref;
  const double* d_cell_c_a = dev_fields.d_c_a;
  const double* d_cell_u_r = dev_fields.d_u_r;
  const double* d_cell_k_bar = dev_fields.d_k_bar;
  const std::uint8_t* d_cell_mask = dev_fields.d_mask;
  if (dev_fields.d_chi_pref != nullptr) {
    TENRYU_ASSERT(
        dev_fields.d_c_a != nullptr &&
            dev_fields.d_u_r != nullptr &&
            dev_fields.d_k_bar != nullptr &&
            dev_fields.d_mask != nullptr,
        "chi ws: device cell fields must all be set");
  } else {
    TENRYU_ASSERT(
        dev_fields.d_c_a == nullptr &&
            dev_fields.d_u_r == nullptr &&
            dev_fields.d_k_bar == nullptr &&
            dev_fields.d_mask == nullptr,
        "chi ws: device cell fields must all be null");
    TENRYU_ASSERT(
        input.cell_chi_pref != nullptr &&
            input.cell_c_a != nullptr &&
            input.cell_u_r != nullptr &&
            input.cell_k_bar != nullptr &&
            input.cell_mask != nullptr,
        "chi ws: host cell fields must all be set");

    const std::size_t n_cells = static_cast<std::size_t>(input.n_cells);
    ensure_device_capacity(
        &workspace->d_cell_chi_pref, &workspace->cap_cell_chi_pref,
        n_cells, "chi ws chi_pref allocation failed");
    ensure_device_capacity(
        &workspace->d_cell_c_a, &workspace->cap_cell_c_a,
        n_cells, "chi ws c_a allocation failed");
    ensure_device_capacity(
        &workspace->d_cell_u_r, &workspace->cap_cell_u_r,
        n_cells, "chi ws u_r allocation failed");
    ensure_device_capacity(
        &workspace->d_cell_k_bar, &workspace->cap_cell_k_bar,
        n_cells, "chi ws k_bar allocation failed");
    ensure_device_capacity(
        &workspace->d_cell_mask, &workspace->cap_cell_mask,
        n_cells, "chi ws cell_mask allocation failed");
    copy_to_device_async(
        workspace->d_cell_chi_pref, input.cell_chi_pref, n_cells, stream,
        "chi ws chi_pref upload failed");
    copy_to_device_async(
        workspace->d_cell_c_a, input.cell_c_a, n_cells, stream,
        "chi ws c_a upload failed");
    copy_to_device_async(
        workspace->d_cell_u_r, input.cell_u_r, n_cells, stream,
        "chi ws u_r upload failed");
    copy_to_device_async(
        workspace->d_cell_k_bar, input.cell_k_bar, n_cells, stream,
        "chi ws k_bar upload failed");
    copy_to_device_async(
        workspace->d_cell_mask, input.cell_mask, n_cells, stream,
        "chi ws cell_mask upload failed");
    d_cell_chi_pref = workspace->d_cell_chi_pref;
    d_cell_c_a = workspace->d_cell_c_a;
    d_cell_u_r = workspace->d_cell_u_r;
    d_cell_k_bar = workspace->d_cell_k_bar;
    d_cell_mask = workspace->d_cell_mask;
  }

  const std::int64_t total =
      static_cast<std::int64_t>(input.n_cells) * n_pairs;
  const std::size_t chi_size = static_cast<std::size_t>(total);
  ensure_device_capacity(
      &workspace->d_chi, &workspace->cap_chi, chi_size,
      "chi ws output allocation failed");
  ensure_device_capacity(
      &workspace->d_readback, &workspace->cap_readback, 4,
      "chi ws readback allocation failed");
  if (chi_size > 0) {
    chi_device_check(
        cudaMemsetAsync(workspace->d_chi, 0, chi_size * sizeof(double),
                        stream),
        "chi ws output memset failed");
  }
  chi_device_check(
      cudaMemsetAsync(workspace->d_readback, 0,
                      4 * sizeof(unsigned long long), stream),
      "chi ws readback memset failed");

  if (total > 0) {
    constexpr int threads_per_block = 256;
    const int blocks = static_cast<int>(
        (total + threads_per_block - 1) / threads_per_block);
    build_chi_ps_kernel<<<blocks, threads_per_block, 0, stream>>>(
        workspace->d_offsets, workspace->d_theta, workspace->d_alpha,
        workspace->d_P, workspace->d_area, workspace->d_ray_index,
        workspace->d_in_limiter, workspace->d_ray_bin, d_cell_chi_pref,
        d_cell_c_a, d_cell_u_r, d_cell_k_bar, d_cell_mask,
        workspace->d_frames9, workspace->d_omega_state,
        workspace->d_pair_p, workspace->d_pair_q, workspace->d_chi,
        workspace->d_readback, input.f_cbet, input.alpha_iaw,
        input.k_a_floor, input.n_section_phi, static_cast<int>(G_ref),
        input.n_impact_bins, n_pairs, input.n_cells);
    chi_device_check(cudaGetLastError(), "chi device kernel launch failed");

    audit_chi_kernel<<<blocks, threads_per_block, 0, stream>>>(
        workspace->d_chi, workspace->d_readback + 2,
        workspace->d_readback + 3, total);
    chi_device_check(cudaGetLastError(), "chi audit kernel launch failed");
  }
  chi_device_check(
      cudaMemcpyAsync(workspace->host_readback.data(), workspace->d_readback,
                      4 * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost, stream),
      "chi ws readback download failed");
  chi_device_check(
      cudaStreamSynchronize(stream), "chi ws stream sync failed");

  return ChiBuildDeviceView{
      G_ps,
      n_pairs,
      static_cast<long long>(workspace->host_readback[0]),
      static_cast<long long>(workspace->host_readback[1]),
      workspace->d_chi,
      ChiDeviceAudit{
          std::bit_cast<double>(workspace->host_readback[2]),
          static_cast<long long>(workspace->host_readback[3])},
      workspace->omega_state.data(),
      workspace->weight_state.data()};
}

ChiBuildResult build_chi_ps_device(const ChiBuildInput& input) {
  ChiDeviceWorkspace ws;
  const ChiBuildDeviceView view =
      build_chi_ps_device_ws(input, {}, ws, nullptr);

  ChiBuildResult result;
  result.G_ps = view.G_ps;
  result.n_pairs = view.n_pairs;
  result.pairs_with_seed = view.pairs_with_seed;
  result.pairs_with_pump = view.pairs_with_pump;
  result.chi.resize(static_cast<std::size_t>(input.n_cells) *
                    static_cast<std::size_t>(view.n_pairs));
  if (!result.chi.empty()) {
    chi_device_check(
        cudaMemcpy(result.chi.data(), view.d_chi,
                   result.chi.size() * sizeof(double),
                   cudaMemcpyDeviceToHost),
        "chi device output download failed");
  }
  if (view.G_ps > 0) {
    result.omega_state.assign(
        view.omega_state, view.omega_state + view.G_ps);
    result.weight_state.assign(
        view.weight_state, view.weight_state + view.G_ps);
  }
  return result;
}

}  // namespace tenryu::laser::port_section
