#include "radiation/diffusion_interface.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr int kBlock = 256;
constexpr double kFourPi = 12.56637061435917295385;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSigmaFloor = 1.0e-30;
constexpr double kEnergyFloor = 1.0e-30;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
std::vector<T> copy_device_array(const T* ptr,
                                 const std::size_t n,
                                 const char* message) {
  std::vector<T> out(n);
  if (n > 0U) {
    TENRYU_ASSERT(ptr != nullptr, message);
    cuda_check(cudaMemcpy(out.data(), ptr, sizeof(T) * n, cudaMemcpyDeviceToHost),
               message);
  }
  return out;
}

double incoming_face_current_energy_host(const std::vector<double>& face_current_in,
                                         const std::vector<std::uint8_t>& diff_cell,
                                         const int n_cells,
                                         const int n_groups) {
  long double sum = 0.0L;
  for (int c = 0; c < n_cells; ++c) {
    if (diff_cell[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    for (int g = 0; g < n_groups; ++g) {
      if (c == 0 || diff_cell[static_cast<std::size_t>(c - 1)] == 0U) {
        const std::size_t idx =
            static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups) +
            static_cast<std::size_t>(g);
        sum += static_cast<long double>(std::max(face_current_in[idx], 0.0));
      }
      if (c + 1 == n_cells ||
          diff_cell[static_cast<std::size_t>(c + 1)] == 0U) {
        const std::size_t idx =
            static_cast<std::size_t>(c + 1) * static_cast<std::size_t>(n_groups) +
            static_cast<std::size_t>(g);
        sum += static_cast<long double>(std::max(face_current_in[idx], 0.0));
      }
    }
  }
  return static_cast<double>(sum);
}

__global__ void fill_interface_phase_space_1d_kernel(
    const int start,
    const int n_new,
    const double* __restrict__ face_r,
    const double* __restrict__ normal_sign,
    double* __restrict__ time_remain,
    double* __restrict__ pos_r,
    double* __restrict__ pos_z,
    double* __restrict__ dir_r,
    double* __restrict__ dir_z,
    double* __restrict__ dir_phi,
    const std::uint64_t* __restrict__ global_id,
    std::uint32_t* __restrict__ rng_counter,
    const std::uint64_t user_seed,
    const std::uint64_t step_number,
    const double dt) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_new) {
    return;
  }

  const int idx = start + t;
  curandStatePhilox4_32_10_t rng;
  curand_init(global_id[idx] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(rng_counter[idx]),
              &rng);

  const double xi_mu = curand_uniform_double(&rng);
  const double xi_phi = curand_uniform_double(&rng);
  const double mu_abs = sqrt(fmax(xi_mu, 0.0));
  const double mu = ((normal_sign[t] >= 0.0) ? 1.0 : -1.0) * mu_abs;
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu_abs * mu_abs));

  pos_r[idx] = face_r[t];
  pos_z[idx] = 0.0;
  dir_r[idx] = mu;
  dir_z[idx] = sin_theta * cos(phi);
  dir_phi[idx] = sin_theta * sin(phi);
  time_remain[idx] = dt;
  rng_counter[idx] += 2U;
}

}  // namespace

DiffusionInterfaceResult spawn_imc_from_diffusion_faces(
    PhotonPool& pool,
    double* diff_E,
    const double* sigma_R,
    const double* vol,
    const double* node_r,
    const std::uint8_t* diff_cell,
    double* face_current_out,
    const int n_cells,
    const int n_groups,
    const double dt,
    const int particles_per_face_group,
    const std::uint64_t global_id_base,
    const std::uint64_t seed,
    const std::uint64_t step,
    const int max_pool_size) {
  TENRYU_ASSERT(n_cells >= 0, "spawn_imc_from_diffusion_faces requires n_cells >= 0");
  TENRYU_ASSERT(n_groups > 0, "spawn_imc_from_diffusion_faces requires n_groups > 0");
  TENRYU_ASSERT(particles_per_face_group >= 1,
                "spawn_imc_from_diffusion_faces requires particles_per_face_group >= 1");
  TENRYU_ASSERT(diff_E != nullptr, "spawn_imc_from_diffusion_faces requires diff_E");
  TENRYU_ASSERT(sigma_R != nullptr, "spawn_imc_from_diffusion_faces requires sigma_R");
  TENRYU_ASSERT(vol != nullptr, "spawn_imc_from_diffusion_faces requires vol");
  TENRYU_ASSERT(node_r != nullptr, "spawn_imc_from_diffusion_faces requires node_r");
  TENRYU_ASSERT(diff_cell != nullptr, "spawn_imc_from_diffusion_faces requires diff_cell");
  TENRYU_ASSERT(face_current_out != nullptr,
                "spawn_imc_from_diffusion_faces requires face_current_out");
  if (n_cells == 0 || !(dt > 0.0)) {
    return {};
  }

  const std::size_t n_cells_us = static_cast<std::size_t>(n_cells);
  const std::size_t n_groups_us = static_cast<std::size_t>(n_groups);
  const std::size_t n_total = n_cells_us * n_groups_us;
  const std::size_t n_faces = n_cells_us + 1U;
  const std::size_t n_face_values = n_faces * n_groups_us;

  std::vector<double> host_E =
      copy_device_array(diff_E, n_total, "diffusion interface copy diff_E failed");
  const std::vector<double> host_sigma =
      copy_device_array(sigma_R, n_total, "diffusion interface copy sigma_R failed");
  const std::vector<double> host_vol =
      copy_device_array(vol, n_cells_us, "diffusion interface copy vol failed");
  const std::vector<double> host_node_r =
      copy_device_array(node_r, n_faces, "diffusion interface copy node_r failed");
  const std::vector<std::uint8_t> host_diff_cell =
      copy_device_array(diff_cell, n_cells_us, "diffusion interface copy mask failed");

  std::vector<double> desired(n_face_values, 0.0);
  std::vector<std::int32_t> source_cell(n_face_values, -1);
  std::vector<double> desired_by_cell(n_total, 0.0);

  for (int face = 1; face < n_cells; ++face) {
    const int left = face - 1;
    const int right = face;
    const bool left_diff = host_diff_cell[static_cast<std::size_t>(left)] != 0U;
    const bool right_diff = host_diff_cell[static_cast<std::size_t>(right)] != 0U;
    if (left_diff == right_diff) {
      continue;
    }
    const int c_diff = left_diff ? left : right;
    const double r_face = std::max(host_node_r[static_cast<std::size_t>(face)], 0.0);
    const double area = kFourPi * r_face * r_face;
    const double dx = std::max(host_node_r[static_cast<std::size_t>(c_diff + 1)] -
                                   host_node_r[static_cast<std::size_t>(c_diff)],
                               0.0);
    const double V = host_vol[static_cast<std::size_t>(c_diff)];
    if (!(area > 0.0) || !(dx > 0.0) || !(V > 0.0)) {
      continue;
    }
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t cg =
          static_cast<std::size_t>(c_diff) * n_groups_us + static_cast<std::size_t>(g);
      const double E_density = std::max(host_E[cg], 0.0);
      if (!(E_density > 0.0)) {
        continue;
      }
      const double sigma = std::max(host_sigma[cg], kSigmaFloor);
      const double denom = 4.0 + 1.5 * sigma * dx;
      const double J_requested =
          tenryu::core::constants::c_light * E_density * area * dt / denom;
      const double J = J_requested;
      if (!(J > 0.0) || !std::isfinite(J)) {
        continue;
      }
      const std::size_t fg =
          static_cast<std::size_t>(face) * n_groups_us + static_cast<std::size_t>(g);
      desired[fg] = J;
      source_cell[fg] = c_diff;
      desired_by_cell[cg] += J;
    }
  }

  std::vector<double> face_out(n_face_values, 0.0);
  for (std::size_t fg = 0; fg < n_face_values; ++fg) {
    const int c_diff = source_cell[fg];
    if (c_diff < 0) {
      continue;
    }
    const int g = static_cast<int>(fg % n_groups_us);
    const std::size_t cg =
        static_cast<std::size_t>(c_diff) * n_groups_us + static_cast<std::size_t>(g);
    const double available =
        std::max(host_E[cg], 0.0) * std::max(host_vol[static_cast<std::size_t>(c_diff)], 0.0);
    const double available_positive = std::max(available - kEnergyFloor, 0.0);
    const double scale =
        (desired_by_cell[cg] > available_positive && desired_by_cell[cg] > 0.0)
            ? (available_positive / desired_by_cell[cg])
            : 1.0;
    face_out[fg] = desired[fg] * std::clamp(scale, 0.0, 1.0);
  }

  std::vector<std::int32_t> out_cell;
  std::vector<std::uint16_t> out_group;
  std::vector<double> out_energy;
  std::vector<double> out_face_r;
  std::vector<double> out_normal_sign;
  long double total_spawned = 0.0L;

  for (int face = 1; face < n_cells; ++face) {
    const int left = face - 1;
    const int right = face;
    const bool left_diff = host_diff_cell[static_cast<std::size_t>(left)] != 0U;
    const bool right_diff = host_diff_cell[static_cast<std::size_t>(right)] != 0U;
    if (left_diff == right_diff) {
      continue;
    }
    const int c_diff = left_diff ? left : right;
    const int c_imc = left_diff ? right : left;
    const double normal_sign = left_diff ? 1.0 : -1.0;
    const double r_face = host_node_r[static_cast<std::size_t>(face)];
    const double V = host_vol[static_cast<std::size_t>(c_diff)];
    if (!(V > 0.0)) {
      continue;
    }
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t fg =
          static_cast<std::size_t>(face) * n_groups_us + static_cast<std::size_t>(g);
      const double J = face_out[fg];
      if (!(J > 0.0)) {
        continue;
      }
      const std::size_t cg =
          static_cast<std::size_t>(c_diff) * n_groups_us + static_cast<std::size_t>(g);
      host_E[cg] = std::max(0.0, host_E[cg] - J / V);
      const double E_per = J / static_cast<double>(particles_per_face_group);
      for (int k = 0; k < particles_per_face_group; ++k) {
        out_cell.push_back(c_imc);
        out_group.push_back(static_cast<std::uint16_t>(g));
        out_face_r.push_back(r_face);
        out_normal_sign.push_back(normal_sign);
        if (k + 1 == particles_per_face_group) {
          out_energy.push_back(J - E_per * static_cast<double>(particles_per_face_group - 1));
        } else {
          out_energy.push_back(E_per);
        }
      }
      total_spawned += static_cast<long double>(J);
    }
  }

  cuda_check(cudaMemcpy(face_current_out,
                        face_out.data(),
                        sizeof(double) * face_out.size(),
                        cudaMemcpyHostToDevice),
             "diffusion interface copy face_current_out failed");
  cuda_check(cudaMemcpy(diff_E,
                        host_E.data(),
                        sizeof(double) * host_E.size(),
                        cudaMemcpyHostToDevice),
             "diffusion interface copy diff_E failed");

  const auto n_new_i64 = static_cast<std::int64_t>(out_energy.size());
  TENRYU_ASSERT(n_new_i64 <= static_cast<std::int64_t>(std::numeric_limits<int>::max()),
                "diffusion interface particle count exceeds int range");
  const int n_new = static_cast<int>(n_new_i64);
  if (n_new == 0) {
    return {};
  }
  const std::int64_t required_i64 =
      static_cast<std::int64_t>(pool.n_alive) + n_new_i64;
  TENRYU_ASSERT(required_i64 <= static_cast<std::int64_t>(max_pool_size),
                "diffusion interface particle creation exceeds max_pool_size");
  pool.reserve(static_cast<int>(required_i64), max_pool_size);
  const int start = pool.n_alive;

  std::vector<double> weight(static_cast<std::size_t>(n_new), 1.0);
  std::vector<double> time_remain(static_cast<std::size_t>(n_new), dt);
  std::vector<double> birth_energy = out_energy;
  std::vector<std::int8_t> sign(static_cast<std::size_t>(n_new), 1);
  std::vector<std::uint64_t> global_id(static_cast<std::size_t>(n_new), 0);
  std::vector<std::uint32_t> rng_counter(static_cast<std::size_t>(n_new), 0U);
  std::vector<std::uint8_t> mode(static_cast<std::size_t>(n_new), kModeIMC);
  std::vector<std::uint8_t> alive(static_cast<std::size_t>(n_new), kAlive);
  for (int i = 0; i < n_new; ++i) {
    global_id[static_cast<std::size_t>(i)] = global_id_base + static_cast<std::uint64_t>(i);
  }

  const std::size_t bytes_d = sizeof(double) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_i32 = sizeof(std::int32_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u16 = sizeof(std::uint16_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u64 = sizeof(std::uint64_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u32 = sizeof(std::uint32_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_i8 = sizeof(std::int8_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u8 = sizeof(std::uint8_t) * static_cast<std::size_t>(n_new);

  cuda_check(cudaMemcpy(pool.energy + start, out_energy.data(), bytes_d, cudaMemcpyHostToDevice),
             "diffusion interface copy energy failed");
  cuda_check(cudaMemcpy(pool.weight + start, weight.data(), bytes_d, cudaMemcpyHostToDevice),
             "diffusion interface copy weight failed");
  cuda_check(cudaMemcpy(pool.time_remain + start,
                        time_remain.data(),
                        bytes_d,
                        cudaMemcpyHostToDevice),
             "diffusion interface copy time_remain failed");
  cuda_check(cudaMemcpy(pool.birth_energy + start,
                        birth_energy.data(),
                        bytes_d,
                        cudaMemcpyHostToDevice),
             "diffusion interface copy birth_energy failed");
  cuda_check(cudaMemcpy(pool.sign + start, sign.data(), bytes_i8, cudaMemcpyHostToDevice),
             "diffusion interface copy sign failed");
  cuda_check(cudaMemcpy(pool.cell_id + start, out_cell.data(), bytes_i32, cudaMemcpyHostToDevice),
             "diffusion interface copy cell_id failed");
  cuda_check(cudaMemcpy(pool.group_id + start,
                        out_group.data(),
                        bytes_u16,
                        cudaMemcpyHostToDevice),
             "diffusion interface copy group_id failed");
  cuda_check(cudaMemcpy(pool.global_id + start,
                        global_id.data(),
                        bytes_u64,
                        cudaMemcpyHostToDevice),
             "diffusion interface copy global_id failed");
  cuda_check(cudaMemcpy(pool.rng_counter + start,
                        rng_counter.data(),
                        bytes_u32,
                        cudaMemcpyHostToDevice),
             "diffusion interface copy rng_counter failed");
  cuda_check(cudaMemcpy(pool.mode + start, mode.data(), bytes_u8, cudaMemcpyHostToDevice),
             "diffusion interface copy mode failed");
  cuda_check(cudaMemcpy(pool.alive + start, alive.data(), bytes_u8, cudaMemcpyHostToDevice),
             "diffusion interface copy alive failed");

  double* d_face_r = nullptr;
  double* d_normal_sign = nullptr;
  d_face_r = static_cast<double*>(
      core::device_scratch_acquire("diffusion_interface:face_r", bytes_d));
  d_normal_sign = static_cast<double*>(
      core::device_scratch_acquire("diffusion_interface:normal_sign", bytes_d));
  cuda_check(cudaMemcpy(d_face_r, out_face_r.data(), bytes_d, cudaMemcpyHostToDevice),
             "diffusion interface copy face_r failed");
  cuda_check(cudaMemcpy(d_normal_sign, out_normal_sign.data(), bytes_d, cudaMemcpyHostToDevice),
             "diffusion interface copy normal_sign failed");

  const int grid = (n_new + kBlock - 1) / kBlock;
  fill_interface_phase_space_1d_kernel<<<grid, kBlock>>>(start,
                                                         n_new,
                                                         d_face_r,
                                                         d_normal_sign,
                                                         pool.time_remain,
                                                         pool.pos_r,
                                                         pool.pos_z,
                                                         pool.dir_r,
                                                         pool.dir_z,
                                                         pool.dir_phi,
                                                         pool.global_id,
                                                         pool.rng_counter,
                                                         seed,
                                                         step,
                                                         dt);
  cuda_check(cudaGetLastError(), "diffusion interface phase-space launch failed");
  cuda_check(cudaDeviceSynchronize(), "diffusion interface phase-space execution failed");

  pool.n_alive += n_new;

  DiffusionInterfaceResult out;
  out.n_spawned = n_new;
  out.E_spawned = static_cast<double>(total_spawned);
  out.E_leaked_vacuum = 0.0;
  return out;
}

double deposit_diffusion_face_current_in(
    double* diff_E,
    const double* face_current_in,
    const double* vol,
    const std::uint8_t* diff_cell,
    const int n_cells,
    const int n_groups) {
  TENRYU_ASSERT(n_cells >= 0, "deposit_diffusion_face_current_in requires n_cells >= 0");
  TENRYU_ASSERT(n_groups > 0, "deposit_diffusion_face_current_in requires n_groups > 0");
  TENRYU_ASSERT(diff_E != nullptr, "deposit_diffusion_face_current_in requires diff_E");
  TENRYU_ASSERT(face_current_in != nullptr,
                "deposit_diffusion_face_current_in requires face_current_in");
  TENRYU_ASSERT(vol != nullptr, "deposit_diffusion_face_current_in requires vol");
  TENRYU_ASSERT(diff_cell != nullptr, "deposit_diffusion_face_current_in requires diff_cell");
  if (n_cells == 0) {
    return 0.0;
  }

  const std::size_t n_cells_us = static_cast<std::size_t>(n_cells);
  const std::size_t n_groups_us = static_cast<std::size_t>(n_groups);
  const std::size_t n_total = n_cells_us * n_groups_us;
  const std::size_t n_face_values = (n_cells_us + 1U) * n_groups_us;
  std::vector<double> host_E =
      copy_device_array(diff_E, n_total, "diffusion interface deposit copy diff_E failed");
  const std::vector<double> host_J =
      copy_device_array(face_current_in,
                        n_face_values,
                        "diffusion interface deposit copy face_current_in failed");
  const std::vector<double> host_vol =
      copy_device_array(vol, n_cells_us, "diffusion interface deposit copy vol failed");
  const std::vector<std::uint8_t> host_diff_cell =
      copy_device_array(diff_cell, n_cells_us, "diffusion interface deposit copy mask failed");

  const double E_in =
      incoming_face_current_energy_host(host_J, host_diff_cell, n_cells, n_groups);
  for (int c = 0; c < n_cells; ++c) {
    if (host_diff_cell[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const double V = host_vol[static_cast<std::size_t>(c)];
    if (!(V > 0.0)) {
      continue;
    }
    for (int g = 0; g < n_groups; ++g) {
      double J_in = 0.0;
      if (c == 0 || host_diff_cell[static_cast<std::size_t>(c - 1)] == 0U) {
        const std::size_t idx =
            static_cast<std::size_t>(c) * n_groups_us + static_cast<std::size_t>(g);
        J_in += std::max(host_J[idx], 0.0);
      }
      if (c + 1 == n_cells ||
          host_diff_cell[static_cast<std::size_t>(c + 1)] == 0U) {
        const std::size_t idx =
            static_cast<std::size_t>(c + 1) * n_groups_us + static_cast<std::size_t>(g);
        J_in += std::max(host_J[idx], 0.0);
      }
      if (J_in > 0.0) {
        host_E[static_cast<std::size_t>(c) * n_groups_us + static_cast<std::size_t>(g)] +=
            J_in / V;
      }
    }
  }

  cuda_check(cudaMemcpy(diff_E,
                        host_E.data(),
                        sizeof(double) * host_E.size(),
                        cudaMemcpyHostToDevice),
             "diffusion interface deposit copy diff_E failed");
  return E_in;
}

double diffusion_face_current_in_energy(const double* face_current_in,
                                        const std::uint8_t* diff_cell,
                                        const int n_cells,
                                        const int n_groups) {
  TENRYU_ASSERT(n_cells >= 0, "diffusion_face_current_in_energy requires n_cells >= 0");
  TENRYU_ASSERT(n_groups > 0, "diffusion_face_current_in_energy requires n_groups > 0");
  TENRYU_ASSERT(face_current_in != nullptr,
                "diffusion_face_current_in_energy requires face_current_in");
  TENRYU_ASSERT(diff_cell != nullptr,
                "diffusion_face_current_in_energy requires diff_cell");
  if (n_cells == 0) {
    return 0.0;
  }
  const std::size_t n_cells_us = static_cast<std::size_t>(n_cells);
  const std::size_t n_groups_us = static_cast<std::size_t>(n_groups);
  const std::vector<double> host_J =
      copy_device_array(face_current_in,
                        (n_cells_us + 1U) * n_groups_us,
                        "diffusion interface sum copy face_current_in failed");
  const std::vector<std::uint8_t> host_diff_cell =
      copy_device_array(diff_cell, n_cells_us, "diffusion interface sum copy mask failed");
  return incoming_face_current_energy_host(host_J, host_diff_cell, n_cells, n_groups);
}

}  // namespace tenryu::radiation
