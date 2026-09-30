#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/constants.hpp"
#include "core/device_error_flags.cuh"
#include "core/error.hpp"
#include "radiation/ddmc_coefficients.hpp"
#include "radiation/ddmc_transport_2d_gpu.cuh"
#include "radiation/particle_pool.cuh"

namespace {

constexpr int kFaceCount = 4;

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_ddmc_2d_gpu");
}

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

int node_index_2d_rz(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

int cell_index_2d_rz(const int i, const int j, const int nz) {
  return i * nz + j;
}

std::size_t cell_group_index(const int cell, const int group, const int n_groups) {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(group);
}

std::size_t face_group_index(const int cell,
                             const int face,
                             const int group,
                             const int n_groups) {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(kFaceCount) *
             static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(face) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(group);
}

std::size_t neighbor_index(const int cell, const int face) {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(kFaceCount) +
         static_cast<std::size_t>(face);
}

struct DDMC2DGpuCase {
  int nr = 0;
  int nz = 0;
  int n_cells = 0;
  int n_groups = 1;
  int n_particles = 0;
  int ddmc_start = 0;
  int n_ddmc = 0;
  std::uint8_t interface_exit_distribution = 0U;
  double dt = 0.0;
  std::uint64_t step_number = 0;
  std::uint64_t user_seed = 0;

  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<double> sigma_a_eff;
  std::vector<double> sigma_s_eff;
  std::vector<double> sigma_leak_face;
  std::vector<std::uint8_t> bc_face;
  std::vector<int> neighbor_face;
  std::vector<double> eta_cdf;
  std::vector<tenryu::radiation::TransportMode> ddmc_mode;

  std::vector<double> pos_r;
  std::vector<double> pos_z;
  std::vector<double> dir_r;
  std::vector<double> dir_z;
  std::vector<double> dir_phi;
  std::vector<double> energy;
  std::vector<double> time_remain;
  std::vector<std::int8_t> sign;
  std::vector<std::uint64_t> global_id;
  std::vector<std::uint32_t> rng_counter;
  std::vector<std::int32_t> cell_id;
  std::vector<std::uint16_t> group_id;
  std::vector<std::uint8_t> mode;
  std::vector<std::uint8_t> alive;
};

struct DDMC2DGpuResult {
  std::vector<double> pos_r;
  std::vector<double> pos_z;
  std::vector<double> dir_r;
  std::vector<double> dir_z;
  std::vector<double> dir_phi;
  std::vector<double> energy;
  std::vector<double> time_remain;
  std::vector<std::int32_t> cell_id;
  std::vector<std::uint16_t> group_id;
  std::vector<std::uint8_t> mode;
  std::vector<std::uint8_t> alive;

  std::vector<double> rad_dep;
  std::vector<double> rad_E_tally;
  std::vector<double> E_escape;
  double E_numerical_loss = 0.0;
  tenryu::core::DeviceErrorFlags flags{};

  unsigned long long ddmc_absorbed = 0ULL;
  unsigned long long ddmc_census = 0ULL;
  unsigned long long ddmc_leak_face0 = 0ULL;
  unsigned long long ddmc_leak_face1 = 0ULL;
  unsigned long long ddmc_leak_face2 = 0ULL;
  unsigned long long ddmc_leak_face3 = 0ULL;
  unsigned long long ddmc_leak_boundary = 0ULL;
  unsigned long long ddmc_converted_to_imc = 0ULL;
  unsigned long long ddmc_sigma_tot_zero = 0ULL;
  unsigned long long ddmc_max_events_reached = 0ULL;
};

DDMC2DGpuCase make_case_2d(const int nr,
                           const int nz,
                           const int n_groups,
                           const int n_particles) {
  DDMC2DGpuCase tc{};
  tc.nr = nr;
  tc.nz = nz;
  tc.n_cells = nr * nz;
  tc.n_groups = n_groups;
  tc.n_particles = n_particles;
  tc.ddmc_start = 0;
  tc.n_ddmc = n_particles;
  tc.dt = 1.0e-12;
  tc.step_number = 1;
  tc.user_seed = 12345;

  const std::size_t n_nodes =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  tc.node_r.assign(n_nodes, 0.0);
  tc.node_z.assign(n_nodes, 0.0);
  for (int i = 0; i <= nr; ++i) {
    for (int j = 0; j <= nz; ++j) {
      const int n = node_index_2d_rz(i, j, nz);
      tc.node_r[static_cast<std::size_t>(n)] = static_cast<double>(i);
      tc.node_z[static_cast<std::size_t>(n)] = static_cast<double>(j);
    }
  }

  const std::size_t n_cell_groups =
      static_cast<std::size_t>(tc.n_cells) * static_cast<std::size_t>(tc.n_groups);
  const std::size_t n_face_groups = n_cell_groups * static_cast<std::size_t>(kFaceCount);
  tc.sigma_a_eff.assign(n_cell_groups, 0.0);
  tc.sigma_s_eff.assign(n_cell_groups, 0.0);
  tc.sigma_leak_face.assign(n_face_groups, 0.0);
  tc.bc_face.assign(
      n_face_groups,
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal));
  tc.neighbor_face.assign(
      static_cast<std::size_t>(tc.n_cells) * static_cast<std::size_t>(kFaceCount), -1);
  tc.ddmc_mode.assign(n_cell_groups, tenryu::radiation::TransportMode::DDMC);

  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int cell = cell_index_2d_rz(i, j, nz);
      tc.neighbor_face[neighbor_index(cell, 0)] =
          (i > 0) ? cell_index_2d_rz(i - 1, j, nz) : -1;
      tc.neighbor_face[neighbor_index(cell, 1)] =
          (i + 1 < nr) ? cell_index_2d_rz(i + 1, j, nz) : -1;
      tc.neighbor_face[neighbor_index(cell, 2)] =
          (j > 0) ? cell_index_2d_rz(i, j - 1, nz) : -1;
      tc.neighbor_face[neighbor_index(cell, 3)] =
          (j + 1 < nz) ? cell_index_2d_rz(i, j + 1, nz) : -1;
    }
  }

  tc.pos_r.assign(static_cast<std::size_t>(n_particles), 0.5);
  tc.pos_z.assign(static_cast<std::size_t>(n_particles), 0.5);
  tc.dir_r.assign(static_cast<std::size_t>(n_particles), 0.0);
  tc.dir_z.assign(static_cast<std::size_t>(n_particles), 0.0);
  tc.dir_phi.assign(static_cast<std::size_t>(n_particles), 0.0);
  tc.energy.assign(static_cast<std::size_t>(n_particles), 1.0);
  tc.time_remain.assign(static_cast<std::size_t>(n_particles), tc.dt);
  tc.sign.assign(static_cast<std::size_t>(n_particles), 1);
  tc.global_id.resize(static_cast<std::size_t>(n_particles), 0ULL);
  tc.rng_counter.assign(static_cast<std::size_t>(n_particles), 0U);
  tc.cell_id.assign(static_cast<std::size_t>(n_particles), 0);
  tc.group_id.assign(static_cast<std::size_t>(n_particles), 0U);
  tc.mode.assign(static_cast<std::size_t>(n_particles),
                 tenryu::radiation::kModeDDMC);
  tc.alive.assign(static_cast<std::size_t>(n_particles),
                  tenryu::radiation::kAlive);

  for (int p = 0; p < n_particles; ++p) {
    tc.global_id[static_cast<std::size_t>(p)] = static_cast<std::uint64_t>(p + 100);
  }
  return tc;
}

DDMC2DGpuResult run_case_2d(const DDMC2DGpuCase& tc) {
  const std::size_t n_particles = static_cast<std::size_t>(tc.n_particles);
  const std::size_t n_cell_groups =
      static_cast<std::size_t>(tc.n_cells) * static_cast<std::size_t>(tc.n_groups);
  const std::size_t n_face_groups =
      n_cell_groups * static_cast<std::size_t>(kFaceCount);
  const std::size_t n_neighbors =
      static_cast<std::size_t>(tc.n_cells) * static_cast<std::size_t>(kFaceCount);
  const std::size_t n_groups = static_cast<std::size_t>(tc.n_groups);
  const std::size_t n_nodes =
      static_cast<std::size_t>(tc.nr + 1) * static_cast<std::size_t>(tc.nz + 1);

  tenryu::radiation::PhotonPool pool;
  pool.allocate(tc.n_particles);
  pool.n_alive = tc.n_particles;

  cuda_check(cudaMemcpy(pool.pos_r,
                        tc.pos_r.data(),
                        sizeof(double) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.pos_z,
                        tc.pos_z.data(),
                        sizeof(double) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_r,
                        tc.dir_r.data(),
                        sizeof(double) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_z,
                        tc.dir_z.data(),
                        sizeof(double) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_phi,
                        tc.dir_phi.data(),
                        sizeof(double) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.energy,
                        tc.energy.data(),
                        sizeof(double) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.time_remain,
                        tc.time_remain.data(),
                        sizeof(double) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.sign,
                        tc.sign.data(),
                        sizeof(std::int8_t) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.global_id,
                        tc.global_id.data(),
                        sizeof(std::uint64_t) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.rng_counter,
                        tc.rng_counter.data(),
                        sizeof(std::uint32_t) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.cell_id,
                        tc.cell_id.data(),
                        sizeof(std::int32_t) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.group_id,
                        tc.group_id.data(),
                        sizeof(std::uint16_t) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.mode,
                        tc.mode.data(),
                        sizeof(std::uint8_t) * n_particles,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.alive,
                        tc.alive.data(),
                        sizeof(std::uint8_t) * n_particles,
                        cudaMemcpyHostToDevice));

  double* d_sigma_a_eff = nullptr;
  double* d_sigma_s_eff = nullptr;
  double* d_sigma_leak_face = nullptr;
  std::uint8_t* d_bc_face = nullptr;
  int* d_neighbor_face = nullptr;
  double* d_eta_cdf = nullptr;
  tenryu::radiation::TransportMode* d_ddmc_mode = nullptr;
  double* d_node_r = nullptr;
  double* d_node_z = nullptr;
  double* d_rad_dep = nullptr;
  double* d_rad_E_tally = nullptr;
  double* d_E_escape = nullptr;
  double* d_E_numerical_loss = nullptr;
  unsigned long long* d_ddmc_absorbed = nullptr;
  unsigned long long* d_ddmc_census = nullptr;
  unsigned long long* d_ddmc_leak_face0 = nullptr;
  unsigned long long* d_ddmc_leak_face1 = nullptr;
  unsigned long long* d_ddmc_leak_face2 = nullptr;
  unsigned long long* d_ddmc_leak_face3 = nullptr;
  unsigned long long* d_ddmc_leak_boundary = nullptr;
  unsigned long long* d_ddmc_converted_to_imc = nullptr;
  unsigned long long* d_ddmc_sigma_tot_zero = nullptr;
  unsigned long long* d_ddmc_max_events_reached = nullptr;
  tenryu::core::DeviceErrorFlags* d_flags = nullptr;

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff),
                        sizeof(double) * n_cell_groups));
  if (!tc.sigma_s_eff.empty()) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_s_eff),
                          sizeof(double) * n_cell_groups));
  }
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_leak_face),
                        sizeof(double) * n_face_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bc_face),
                        sizeof(std::uint8_t) * n_face_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_neighbor_face),
                        sizeof(int) * n_neighbors));
  if (!tc.eta_cdf.empty()) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_eta_cdf),
                          sizeof(double) * n_cell_groups));
  }
  if (!tc.ddmc_mode.empty()) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_mode),
                          sizeof(tenryu::radiation::TransportMode) * n_cell_groups));
  }
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_node_r), sizeof(double) * n_nodes));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_node_z), sizeof(double) * n_nodes));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_dep), sizeof(double) * n_cell_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_E_tally),
                        sizeof(double) * n_cell_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_escape), sizeof(double) * n_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_numerical_loss), sizeof(double)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_absorbed),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_census),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_face0),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_face1),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_face2),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_face3),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_boundary),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_converted_to_imc),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_sigma_tot_zero),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_max_events_reached),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_flags), sizeof(*d_flags)));

  cuda_check(cudaMemcpy(d_sigma_a_eff,
                        tc.sigma_a_eff.data(),
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyHostToDevice));
  if (d_sigma_s_eff != nullptr) {
    cuda_check(cudaMemcpy(d_sigma_s_eff,
                          tc.sigma_s_eff.data(),
                          sizeof(double) * n_cell_groups,
                          cudaMemcpyHostToDevice));
  }
  cuda_check(cudaMemcpy(d_sigma_leak_face,
                        tc.sigma_leak_face.data(),
                        sizeof(double) * n_face_groups,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_bc_face,
                        tc.bc_face.data(),
                        sizeof(std::uint8_t) * n_face_groups,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_neighbor_face,
                        tc.neighbor_face.data(),
                        sizeof(int) * n_neighbors,
                        cudaMemcpyHostToDevice));
  if (d_eta_cdf != nullptr) {
    cuda_check(cudaMemcpy(d_eta_cdf,
                          tc.eta_cdf.data(),
                          sizeof(double) * n_cell_groups,
                          cudaMemcpyHostToDevice));
  }
  if (d_ddmc_mode != nullptr) {
    cuda_check(cudaMemcpy(d_ddmc_mode,
                          tc.ddmc_mode.data(),
                          sizeof(tenryu::radiation::TransportMode) * n_cell_groups,
                          cudaMemcpyHostToDevice));
  }
  cuda_check(cudaMemcpy(d_node_r,
                        tc.node_r.data(),
                        sizeof(double) * n_nodes,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_node_z,
                        tc.node_z.data(),
                        sizeof(double) * n_nodes,
                        cudaMemcpyHostToDevice));

  cuda_check(cudaMemset(d_rad_dep, 0, sizeof(double) * n_cell_groups));
  cuda_check(cudaMemset(d_rad_E_tally, 0, sizeof(double) * n_cell_groups));
  cuda_check(cudaMemset(d_E_escape, 0, sizeof(double) * n_groups));
  cuda_check(cudaMemset(d_E_numerical_loss, 0, sizeof(double)));
  cuda_check(cudaMemset(d_ddmc_absorbed, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_census, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_leak_face0, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_leak_face1, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_leak_face2, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_leak_face3, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_leak_boundary, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_converted_to_imc, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_sigma_tot_zero, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_max_events_reached, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_flags, 0, sizeof(*d_flags)));

  tenryu::radiation::DDMCTransport2DGPUInputs in{};
  in.pool = &pool;
  in.sigma_a_eff = d_sigma_a_eff;
  in.sigma_s_eff = d_sigma_s_eff;
  in.sigma_leak_face = d_sigma_leak_face;
  in.bc_face = d_bc_face;
  in.neighbor_face = d_neighbor_face;
  in.eta_cdf = d_eta_cdf;
  in.ddmc_mode = d_ddmc_mode;
  in.node_r = d_node_r;
  in.node_z = d_node_z;
  in.rad_dep = d_rad_dep;
  in.rad_E_tally = d_rad_E_tally;
  in.E_escape = d_E_escape;
  in.E_numerical_loss = d_E_numerical_loss;
  in.ddmc_absorbed = d_ddmc_absorbed;
  in.ddmc_census = d_ddmc_census;
  in.ddmc_leak_face0 = d_ddmc_leak_face0;
  in.ddmc_leak_face1 = d_ddmc_leak_face1;
  in.ddmc_leak_face2 = d_ddmc_leak_face2;
  in.ddmc_leak_face3 = d_ddmc_leak_face3;
  in.ddmc_leak_boundary = d_ddmc_leak_boundary;
  in.ddmc_converted_to_imc = d_ddmc_converted_to_imc;
  in.ddmc_sigma_tot_zero = d_ddmc_sigma_tot_zero;
  in.ddmc_max_events_reached = d_ddmc_max_events_reached;
  in.n_cells = tc.n_cells;
  in.n_groups = tc.n_groups;
  in.nr = tc.nr;
  in.nz = tc.nz;
  in.n_ddmc = tc.n_ddmc;
  in.ddmc_start = tc.ddmc_start;
  in.interface_exit_distribution = tc.interface_exit_distribution;
  in.dt = tc.dt;
  in.step_number = tc.step_number;
  in.user_seed = tc.user_seed;
  in.error_flags = d_flags;
  tenryu::radiation::ddmc_transport_2d_gpu_cuda(in);

  DDMC2DGpuResult out{};
  out.pos_r.resize(n_particles, 0.0);
  out.pos_z.resize(n_particles, 0.0);
  out.dir_r.resize(n_particles, 0.0);
  out.dir_z.resize(n_particles, 0.0);
  out.dir_phi.resize(n_particles, 0.0);
  out.energy.resize(n_particles, 0.0);
  out.time_remain.resize(n_particles, 0.0);
  out.cell_id.resize(n_particles, 0);
  out.group_id.resize(n_particles, 0U);
  out.mode.resize(n_particles, 0U);
  out.alive.resize(n_particles, 0U);
  out.rad_dep.resize(n_cell_groups, 0.0);
  out.rad_E_tally.resize(n_cell_groups, 0.0);
  out.E_escape.resize(n_groups, 0.0);

  cuda_check(cudaMemcpy(out.pos_r.data(),
                        pool.pos_r,
                        sizeof(double) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.pos_z.data(),
                        pool.pos_z,
                        sizeof(double) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.dir_r.data(),
                        pool.dir_r,
                        sizeof(double) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.dir_z.data(),
                        pool.dir_z,
                        sizeof(double) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.dir_phi.data(),
                        pool.dir_phi,
                        sizeof(double) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.energy.data(),
                        pool.energy,
                        sizeof(double) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.time_remain.data(),
                        pool.time_remain,
                        sizeof(double) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.cell_id.data(),
                        pool.cell_id,
                        sizeof(std::int32_t) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.group_id.data(),
                        pool.group_id,
                        sizeof(std::uint16_t) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.mode.data(),
                        pool.mode,
                        sizeof(std::uint8_t) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.alive.data(),
                        pool.alive,
                        sizeof(std::uint8_t) * n_particles,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.rad_dep.data(),
                        d_rad_dep,
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.rad_E_tally.data(),
                        d_rad_E_tally,
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out.E_escape.data(),
                        d_E_escape,
                        sizeof(double) * n_groups,
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.E_numerical_loss,
                        d_E_numerical_loss,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.flags,
                        d_flags,
                        sizeof(out.flags),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_absorbed,
                        d_ddmc_absorbed,
                        sizeof(out.ddmc_absorbed),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_census,
                        d_ddmc_census,
                        sizeof(out.ddmc_census),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_leak_face0,
                        d_ddmc_leak_face0,
                        sizeof(out.ddmc_leak_face0),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_leak_face1,
                        d_ddmc_leak_face1,
                        sizeof(out.ddmc_leak_face1),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_leak_face2,
                        d_ddmc_leak_face2,
                        sizeof(out.ddmc_leak_face2),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_leak_face3,
                        d_ddmc_leak_face3,
                        sizeof(out.ddmc_leak_face3),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_leak_boundary,
                        d_ddmc_leak_boundary,
                        sizeof(out.ddmc_leak_boundary),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_converted_to_imc,
                        d_ddmc_converted_to_imc,
                        sizeof(out.ddmc_converted_to_imc),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_sigma_tot_zero,
                        d_ddmc_sigma_tot_zero,
                        sizeof(out.ddmc_sigma_tot_zero),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_max_events_reached,
                        d_ddmc_max_events_reached,
                        sizeof(out.ddmc_max_events_reached),
                        cudaMemcpyDeviceToHost));

  cuda_check(cudaFree(d_flags));
  cuda_check(cudaFree(d_ddmc_max_events_reached));
  cuda_check(cudaFree(d_ddmc_sigma_tot_zero));
  cuda_check(cudaFree(d_ddmc_converted_to_imc));
  cuda_check(cudaFree(d_ddmc_leak_boundary));
  cuda_check(cudaFree(d_ddmc_leak_face3));
  cuda_check(cudaFree(d_ddmc_leak_face2));
  cuda_check(cudaFree(d_ddmc_leak_face1));
  cuda_check(cudaFree(d_ddmc_leak_face0));
  cuda_check(cudaFree(d_ddmc_census));
  cuda_check(cudaFree(d_ddmc_absorbed));
  cuda_check(cudaFree(d_E_numerical_loss));
  cuda_check(cudaFree(d_E_escape));
  cuda_check(cudaFree(d_rad_E_tally));
  cuda_check(cudaFree(d_rad_dep));
  cuda_check(cudaFree(d_node_z));
  cuda_check(cudaFree(d_node_r));
  if (d_ddmc_mode != nullptr) {
    cuda_check(cudaFree(d_ddmc_mode));
  }
  if (d_eta_cdf != nullptr) {
    cuda_check(cudaFree(d_eta_cdf));
  }
  cuda_check(cudaFree(d_neighbor_face));
  cuda_check(cudaFree(d_bc_face));
  cuda_check(cudaFree(d_sigma_leak_face));
  if (d_sigma_s_eff != nullptr) {
    cuda_check(cudaFree(d_sigma_s_eff));
  }
  cuda_check(cudaFree(d_sigma_a_eff));
  return out;
}

}  // namespace

TEST_CASE("DDMC 2D GPU absorbs particle in high absorption limit",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 2.5;
  tc.sigma_a_eff[0] = 1.0e10;

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.rad_dep[0] == Catch::Approx(2.5).epsilon(1.0e-12));
  REQUIRE(out.ddmc_absorbed == 1ULL);
  REQUIRE(out.ddmc_census == 0ULL);
  REQUIRE(out.flags.invalid_cell == 0);
  REQUIRE(out.flags.infinite_loop == 0);
}

TEST_CASE("DDMC 2D GPU deposits absorption in correct group bin",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 2, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 2.25;
  tc.group_id[0] = 1U;
  tc.cell_id[0] = 0;
  tc.sigma_a_eff[cell_group_index(0, 1, tc.n_groups)] = 1.0e10;
  const std::size_t leak_idx_g0 = face_group_index(0, 1, 0, tc.n_groups);
  tc.sigma_leak_face[leak_idx_g0] = 1.0e20;
  tc.bc_face[leak_idx_g0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.group_id[0] == 1U);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.rad_dep[cell_group_index(0, 0, tc.n_groups)] ==
          Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.rad_dep[cell_group_index(0, 1, tc.n_groups)] ==
          Catch::Approx(2.25).epsilon(1.0e-12));
  REQUIRE(out.ddmc_absorbed == 1ULL);
  REQUIRE(out.ddmc_leak_boundary == 0ULL);
  REQUIRE(out.flags.invalid_cell == 0);
}

TEST_CASE("DDMC 2D GPU censes particle when event time exceeds dt",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-12;
  tc.energy[0] = 3.0;
  tc.sigma_a_eff[0] = 1.0e-20;

  const DDMC2DGpuResult out = run_case_2d(tc);
  const double expected_tally = 3.0 * tenryu::core::constants::c_light * tc.dt;
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeDDMC);
  REQUIRE(out.time_remain[0] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out.energy[0] == Catch::Approx(3.0).epsilon(1.0e-12));
  REQUIRE(out.rad_dep[0] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out.rad_E_tally[0] == Catch::Approx(expected_tally).epsilon(1.0e-12));
  REQUIRE(out.ddmc_census == 1ULL);
  REQUIRE(out.ddmc_sigma_tot_zero == 0ULL);
  REQUIRE(out.flags.invalid_cell == 0);
}

TEST_CASE("DDMC 2D GPU leaks to vacuum on R_right face",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 1.75;
  const int cell = (tc.nr - 1) * tc.nz;
  tc.cell_id[0] = cell;
  tc.pos_r[0] = static_cast<double>(tc.nr) - 0.5;
  tc.pos_z[0] = 0.5;
  const std::size_t leak_idx = face_group_index(cell, 1, 0, tc.n_groups);
  tc.sigma_leak_face[leak_idx] = 1.0e6;
  tc.bc_face[leak_idx] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.E_escape[0] == Catch::Approx(1.75).epsilon(1.0e-12));
  REQUIRE(out.ddmc_leak_face1 >= 1ULL);
  REQUIRE(out.ddmc_leak_boundary == 1ULL);
  REQUIRE(out.flags.invalid_cell == 0);
}

TEST_CASE("DDMC 2D GPU signed opposite particles cancel tallies",
          "[radiation][ddmc][gpu][2d][signed]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  constexpr int n_particles = 2;
  const std::vector<double> energy(n_particles, 4.0);
  const std::vector<std::int8_t> sign = {1, -1};

  DDMC2DGpuCase absorb = make_case_2d(2, 2, 1, n_particles);
  absorb.dt = 1.0e-10;
  absorb.energy = energy;
  absorb.sign = sign;
  absorb.sigma_a_eff[0] = 1.0e10;
  DDMC2DGpuResult out_absorb = run_case_2d(absorb);
  REQUIRE(out_absorb.rad_dep[0] == Catch::Approx(0.0).margin(1.0e-12));

  DDMC2DGpuCase census = make_case_2d(2, 2, 1, n_particles);
  census.dt = 1.0e-12;
  census.energy = energy;
  census.sign = sign;
  census.sigma_a_eff[0] = 1.0e-20;
  DDMC2DGpuResult out_census = run_case_2d(census);
  REQUIRE(out_census.rad_E_tally[0] == Catch::Approx(0.0).margin(1.0e-12));

  DDMC2DGpuCase escape = make_case_2d(2, 2, 1, n_particles);
  escape.dt = 1.0e-10;
  escape.energy = energy;
  escape.sign = sign;
  const int cell = (escape.nr - 1) * escape.nz;
  escape.cell_id.assign(static_cast<std::size_t>(n_particles), cell);
  const std::size_t leak_idx = face_group_index(cell, 1, 0, escape.n_groups);
  escape.sigma_leak_face[leak_idx] = 1.0e10;
  escape.bc_face[leak_idx] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);
  DDMC2DGpuResult out_escape = run_case_2d(escape);
  REQUIRE(out_escape.E_escape[0] == Catch::Approx(0.0).margin(1.0e-12));
}

TEST_CASE("DDMC 2D GPU leaks to vacuum on Z_top face", "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 1.25;
  const int cell = tc.nz - 1;
  tc.cell_id[0] = cell;
  tc.pos_r[0] = 0.5;
  tc.pos_z[0] = static_cast<double>(tc.nz) - 0.5;
  const std::size_t leak_idx = face_group_index(cell, 3, 0, tc.n_groups);
  tc.sigma_leak_face[leak_idx] = 1.0e6;
  tc.bc_face[leak_idx] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.E_escape[0] == Catch::Approx(1.25).epsilon(1.0e-12));
  REQUIRE(out.ddmc_leak_face3 >= 1ULL);
  REQUIRE(out.ddmc_leak_boundary == 1ULL);
}

TEST_CASE("DDMC 2D GPU reflects on boundary face", "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-12;
  tc.energy[0] = 1.0;
  tc.dir_r[0] = 1.0;
  tc.dir_z[0] = 0.0;
  const std::size_t leak_idx = face_group_index(0, 0, 0, tc.n_groups);
  tc.sigma_leak_face[leak_idx] = 1.0e3;
  tc.bc_face[leak_idx] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Reflective);

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeDDMC);
  REQUIRE(out.energy[0] == Catch::Approx(1.0).epsilon(1.0e-12));
  REQUIRE(out.time_remain[0] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out.E_escape[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.rad_dep[0] == Catch::Approx(0.0).margin(1.0e-14));
  // Direction should remain finite after reflection(s)
  REQUIRE(std::isfinite(out.dir_r[0]));
  REQUIRE(std::isfinite(out.dir_z[0]));
  REQUIRE(std::isfinite(out.dir_phi[0]));
  // Multiple reflections may restore original direction; verify reflection occurred
  REQUIRE(out.ddmc_leak_face0 >= 2ULL);  // High sigma_leak ensures multiple reflections
  REQUIRE(out.ddmc_leak_boundary == 0ULL);
}

TEST_CASE("DDMC 2D GPU converts to IMC at interface", "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 5.0;
  const std::size_t leak_idx = face_group_index(0, 1, 0, tc.n_groups);
  tc.sigma_leak_face[leak_idx] = 1.0e6;
  tc.bc_face[leak_idx] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Interface);
  tc.neighbor_face[neighbor_index(0, 1)] = 2;

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeIMC);
  REQUIRE(out.cell_id[0] == 2);
  REQUIRE(out.energy[0] == Catch::Approx(5.0).epsilon(1.0e-12));
  REQUIRE(out.time_remain[0] > 0.0);
  REQUIRE(out.time_remain[0] < tc.dt);
  REQUIRE(out.ddmc_converted_to_imc == 1ULL);
  REQUIRE(out.ddmc_leak_face1 >= 1ULL);
  REQUIRE(out.E_escape[0] == Catch::Approx(0.0).margin(1.0e-14));
}

TEST_CASE("DDMC 2D GPU converts to IMC at interface with half_isotropic",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 5.0;
  const std::size_t leak_idx = face_group_index(0, 1, 0, tc.n_groups);
  tc.sigma_leak_face[leak_idx] = 1.0e6;
  tc.bc_face[leak_idx] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Interface);
  tc.neighbor_face[neighbor_index(0, 1)] = 2;
  tc.interface_exit_distribution = 1U;

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeIMC);
  REQUIRE(out.cell_id[0] == 2);
  REQUIRE(out.energy[0] == Catch::Approx(5.0).epsilon(1.0e-12));
  REQUIRE(out.time_remain[0] > 0.0);
  REQUIRE(out.time_remain[0] < tc.dt);
  REQUIRE(out.ddmc_converted_to_imc == 1ULL);
  REQUIRE(out.ddmc_leak_face1 >= 1ULL);
  REQUIRE(std::isfinite(out.dir_r[0]));
  REQUIRE(std::isfinite(out.dir_z[0]));
  REQUIRE(std::isfinite(out.dir_phi[0]));
  const double dir_norm = std::sqrt(out.dir_r[0] * out.dir_r[0] +
                                    out.dir_z[0] * out.dir_z[0] +
                                    out.dir_phi[0] * out.dir_phi[0]);
  REQUIRE(dir_norm == Catch::Approx(1.0).epsilon(1.0e-12));
}

TEST_CASE("DDMC 2D GPU negative leak triggers invalid_boundary",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 1.2;
  const std::size_t leak_idx = face_group_index(0, 0, 0, tc.n_groups);
  tc.sigma_leak_face[leak_idx] = -1.0;

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.rad_dep[0] == Catch::Approx(1.2).epsilon(1.0e-12));
  REQUIRE(out.ddmc_absorbed == 1ULL);
  REQUIRE(out.flags.invalid_boundary == 1);
}

TEST_CASE("DDMC 2D GPU NaN energy triggers nan_particle flag",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = std::nan("");

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.time_remain[0] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out.flags.nan_particle == 1);
}

TEST_CASE("DDMC 2D GPU max events deposits energy", "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(1, 1, 1, 1);
  tc.dt = 1.0e-12;
  tc.energy[0] = 1.0;
  tc.sigma_a_eff[0] = 0.0;
  for (int face = 0; face < kFaceCount; ++face) {
    const std::size_t leak_idx = face_group_index(0, face, 0, tc.n_groups);
    tc.sigma_leak_face[leak_idx] = 1.0e12;
    tc.bc_face[leak_idx] =
        static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Reflective);
  }

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.rad_dep[0] == Catch::Approx(1.0).epsilon(1.0e-12));
  REQUIRE(out.ddmc_max_events_reached >= 1ULL);
  REQUIRE(out.flags.infinite_loop >= 1);
}

TEST_CASE("DDMC 2D GPU same-cell NLTE scatter demotes to IMC",
          "[radiation][ddmc][gpu][2d][nlte]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(1, 1, 2, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 2.75;
  tc.group_id[0] = 0U;
  tc.sigma_a_eff[cell_group_index(0, 0, tc.n_groups)] = 0.0;
  tc.sigma_a_eff[cell_group_index(0, 1, tc.n_groups)] = 0.0;
  tc.sigma_s_eff[cell_group_index(0, 0, tc.n_groups)] = 1.0e6;
  tc.sigma_s_eff[cell_group_index(0, 1, tc.n_groups)] = 0.0;
  tc.eta_cdf = {0.0, 1.0};
  tc.ddmc_mode = {tenryu::radiation::TransportMode::DDMC,
                  tenryu::radiation::TransportMode::IMC};

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeIMC);
  REQUIRE(out.cell_id[0] == 0);
  REQUIRE(out.group_id[0] == 1U);
  REQUIRE(out.energy[0] == Catch::Approx(2.75).epsilon(1.0e-12));
  REQUIRE(out.time_remain[0] > 0.0);
  REQUIRE(out.time_remain[0] < tc.dt);
  REQUIRE(out.pos_r[0] >= 0.0);
  REQUIRE(out.pos_r[0] <= 1.0);
  REQUIRE(out.pos_z[0] >= 0.0);
  REQUIRE(out.pos_z[0] <= 1.0);
  const double dir_norm = std::sqrt(out.dir_r[0] * out.dir_r[0] +
                                    out.dir_z[0] * out.dir_z[0] +
                                    out.dir_phi[0] * out.dir_phi[0]);
  REQUIRE(dir_norm == Catch::Approx(1.0).epsilon(1.0e-12));
  REQUIRE(out.ddmc_converted_to_imc == 1ULL);
}

TEST_CASE("DDMC 2D GPU conserves energy over many particles", "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  constexpr int n_particles = 1024;
  DDMC2DGpuCase tc = make_case_2d(3, 3, 1, n_particles);
  tc.dt = 1.0e-12;
  tc.energy.assign(static_cast<std::size_t>(n_particles), 2.0);
  tc.time_remain.assign(static_cast<std::size_t>(n_particles), tc.dt);

  for (int p = 0; p < n_particles; ++p) {
    const int cell = p % tc.n_cells;
    const int i = cell / tc.nz;
    const int j = cell - i * tc.nz;
    tc.cell_id[static_cast<std::size_t>(p)] = cell;
    tc.pos_r[static_cast<std::size_t>(p)] = static_cast<double>(i) + 0.5;
    tc.pos_z[static_cast<std::size_t>(p)] = static_cast<double>(j) + 0.5;
  }

  for (int cell = 0; cell < tc.n_cells; ++cell) {
    const std::size_t cg = cell_group_index(cell, 0, tc.n_groups);
    tc.sigma_a_eff[cg] = 0.75 + 0.05 * static_cast<double>(cell % 4);
    for (int face = 0; face < kFaceCount; ++face) {
      const std::size_t fg = face_group_index(cell, face, 0, tc.n_groups);
      tc.sigma_leak_face[fg] = 0.5 + 0.1 * static_cast<double>(face);
      const int neighbor = tc.neighbor_face[neighbor_index(cell, face)];
      tc.bc_face[fg] = static_cast<std::uint8_t>(
          (neighbor >= 0) ? tenryu::radiation::DDMCBoundaryType::Internal
                          : tenryu::radiation::DDMCBoundaryType::Vacuum);
    }
  }

  const DDMC2DGpuResult out = run_case_2d(tc);
  const double initial_total = static_cast<double>(n_particles) * 2.0;
  const double dep_sum = std::accumulate(out.rad_dep.begin(), out.rad_dep.end(), 0.0);
  const double escape_sum =
      std::accumulate(out.E_escape.begin(), out.E_escape.end(), 0.0);
  double alive_sum = 0.0;
  for (int p = 0; p < n_particles; ++p) {
    if (out.alive[static_cast<std::size_t>(p)] == tenryu::radiation::kAlive) {
      alive_sum += out.energy[static_cast<std::size_t>(p)];
    }
  }

  const double final_total =
      dep_sum + escape_sum + alive_sum + std::max(out.E_numerical_loss, 0.0);
  REQUIRE(final_total == Catch::Approx(initial_total).epsilon(1.0e-9));
  REQUIRE(out.flags.invalid_cell == 0);
  REQUIRE(out.flags.infinite_loop == 0);
}

TEST_CASE("DDMC 2D GPU internal leak moves particle to neighbor",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 1.5;
  const std::size_t leak_idx = face_group_index(0, 1, 0, tc.n_groups);
  tc.sigma_leak_face[leak_idx] = 1.0e6;
  tc.bc_face[leak_idx] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal);
  tc.neighbor_face[neighbor_index(0, 1)] = 2;

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeDDMC);
  REQUIRE(out.cell_id[0] == 2);
  REQUIRE(out.energy[0] == Catch::Approx(1.5).epsilon(1.0e-12));
  REQUIRE(out.ddmc_leak_face1 >= 1ULL);
  REQUIRE(out.E_escape[0] == Catch::Approx(0.0).margin(1.0e-14));
}

TEST_CASE("DDMC 2D GPU sigma_tot floor triggers census",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 1, 1);
  tc.dt = 1.0e-12;
  tc.energy[0] = 2.0;
  tc.sigma_a_eff[0] = 0.0;

  const DDMC2DGpuResult out = run_case_2d(tc);
  const double expected_tally = 2.0 * tenryu::core::constants::c_light * tc.dt;
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeDDMC);
  REQUIRE(out.energy[0] == Catch::Approx(2.0).epsilon(1.0e-12));
  REQUIRE(out.time_remain[0] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out.rad_dep[0] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out.rad_E_tally[0] == Catch::Approx(expected_tally).epsilon(1.0e-12));
  REQUIRE(out.ddmc_sigma_tot_zero == 1ULL);
  REQUIRE(out.ddmc_census == 0ULL);
}

TEST_CASE("DDMC 2D GPU multi-hop internal leak traversal",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(3, 1, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 4.0;
  tc.cell_id[0] = 0;
  tc.pos_r[0] = 0.5;
  tc.pos_z[0] = 0.5;
  std::fill(tc.sigma_a_eff.begin(), tc.sigma_a_eff.end(), 0.0);

  for (int cell = 0; cell < 2; ++cell) {
    const std::size_t leak_idx = face_group_index(cell, 1, 0, tc.n_groups);
    tc.sigma_leak_face[leak_idx] = 1.0e6;
    tc.bc_face[leak_idx] =
        static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal);
  }

  const std::size_t leak_idx_vac = face_group_index(2, 1, 0, tc.n_groups);
  tc.sigma_leak_face[leak_idx_vac] = 1.0e6;
  tc.bc_face[leak_idx_vac] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.cell_id[0] == 2);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.E_escape[0] == Catch::Approx(4.0).epsilon(1.0e-10));
  REQUIRE(out.ddmc_leak_face1 >= 2ULL);
  REQUIRE(out.ddmc_leak_boundary == 1ULL);
}

TEST_CASE("DDMC 2D GPU multi-group vacuum leak stride isolation",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(2, 2, 2, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 3.0;
  tc.group_id[0] = 1U;
  tc.cell_id[0] = 0;

  const std::size_t leak_g1 = face_group_index(0, 1, 1, tc.n_groups);
  tc.sigma_leak_face[leak_g1] = 1.0e6;
  tc.bc_face[leak_g1] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.E_escape[1] == Catch::Approx(3.0).epsilon(1.0e-10));
  REQUIRE(out.E_escape[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.ddmc_leak_boundary == 1ULL);
  REQUIRE(out.ddmc_leak_face1 >= 1ULL);
}

TEST_CASE("DDMC 2D GPU single-column topology (1xN)",
          "[radiation][ddmc][gpu][2d]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC 2D GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMC2DGpuCase tc = make_case_2d(1, 3, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 2.0;
  tc.cell_id[0] = 0;
  tc.pos_r[0] = 0.5;
  tc.pos_z[0] = 0.5;

  for (int cell = 0; cell < 2; ++cell) {
    const std::size_t leak_idx = face_group_index(cell, 3, 0, tc.n_groups);
    tc.sigma_leak_face[leak_idx] = 1.0e6;
    tc.bc_face[leak_idx] =
        static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal);
  }

  const std::size_t leak_vac = face_group_index(2, 3, 0, tc.n_groups);
  tc.sigma_leak_face[leak_vac] = 1.0e6;
  tc.bc_face[leak_vac] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);

  const DDMC2DGpuResult out = run_case_2d(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.E_escape[0] == Catch::Approx(2.0).epsilon(1.0e-10));
  REQUIRE(out.ddmc_leak_face3 >= 2ULL);
  REQUIRE(out.ddmc_leak_boundary == 1ULL);
}
