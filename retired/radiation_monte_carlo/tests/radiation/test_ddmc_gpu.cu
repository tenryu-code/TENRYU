#include <algorithm>
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
#include "radiation/ddmc_transport_gpu.cuh"
#include "radiation/particle_pool.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_ddmc_gpu");
}

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

struct DDMCGpuCase {
  int n_cells = 0;
  int n_groups = 1;
  int n_particles = 0;
  int ddmc_start = 0;
  int n_ddmc = 0;
  double dt = 0.0;
  std::uint64_t step_number = 0;
  std::uint64_t user_seed = 0;

  std::vector<double> node_r;
  std::vector<double> sigma_a_eff;
  std::vector<double> sigma_s_eff;
  std::vector<double> sigma_leak_left;
  std::vector<double> sigma_leak_right;
  std::vector<std::uint8_t> bc_left;
  std::vector<std::uint8_t> bc_right;
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

struct DDMCGpuResult {
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
  unsigned long long ddmc_leak_left = 0ULL;
  unsigned long long ddmc_leak_right = 0ULL;
  unsigned long long ddmc_leak_boundary = 0ULL;
  unsigned long long ddmc_converted_to_imc = 0ULL;
  unsigned long long ddmc_sigma_tot_zero = 0ULL;
  unsigned long long ddmc_max_events_reached = 0ULL;
};

DDMCGpuCase make_case(const int n_cells,
                      const int n_groups,
                      const int n_particles) {
  DDMCGpuCase tc{};
  tc.n_cells = n_cells;
  tc.n_groups = n_groups;
  tc.n_particles = n_particles;
  tc.ddmc_start = 0;
  tc.n_ddmc = n_particles;
  tc.dt = 1.0e-12;
  tc.step_number = 3;
  tc.user_seed = 777;

  tc.node_r.resize(static_cast<std::size_t>(n_cells + 1), 0.0);
  for (int i = 0; i <= n_cells; ++i) {
    tc.node_r[static_cast<std::size_t>(i)] = static_cast<double>(i);
  }

  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  tc.sigma_a_eff.assign(n_cell_groups, 0.0);
  tc.sigma_s_eff.assign(n_cell_groups, 0.0);
  tc.sigma_leak_left.assign(n_cell_groups, 0.0);
  tc.sigma_leak_right.assign(n_cell_groups, 0.0);
  tc.bc_left.assign(
      n_cell_groups,
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal));
  tc.bc_right.assign(
      n_cell_groups,
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal));
  tc.ddmc_mode.assign(n_cell_groups, tenryu::radiation::TransportMode::DDMC);

  tc.pos_r.assign(static_cast<std::size_t>(n_particles), 0.5);
  tc.pos_z.assign(static_cast<std::size_t>(n_particles), 0.0);
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

DDMCGpuResult run_case(const DDMCGpuCase& tc) {
  const std::size_t n_particles = static_cast<std::size_t>(tc.n_particles);
  const std::size_t n_cell_groups =
      static_cast<std::size_t>(tc.n_cells) * static_cast<std::size_t>(tc.n_groups);
  const std::size_t n_groups = static_cast<std::size_t>(tc.n_groups);
  const std::size_t n_nodes = static_cast<std::size_t>(tc.n_cells + 1);

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
  double* d_sigma_leak_left = nullptr;
  double* d_sigma_leak_right = nullptr;
  std::uint8_t* d_bc_left = nullptr;
  std::uint8_t* d_bc_right = nullptr;
  double* d_eta_cdf = nullptr;
  tenryu::radiation::TransportMode* d_ddmc_mode = nullptr;
  double* d_node_r = nullptr;
  double* d_rad_dep = nullptr;
  double* d_rad_E_tally = nullptr;
  double* d_E_escape = nullptr;
  double* d_E_numerical_loss = nullptr;
  unsigned long long* d_ddmc_absorbed = nullptr;
  unsigned long long* d_ddmc_census = nullptr;
  unsigned long long* d_ddmc_leak_left = nullptr;
  unsigned long long* d_ddmc_leak_right = nullptr;
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
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_leak_left),
                        sizeof(double) * n_cell_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_leak_right),
                        sizeof(double) * n_cell_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bc_left),
                        sizeof(std::uint8_t) * n_cell_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bc_right),
                        sizeof(std::uint8_t) * n_cell_groups));
  if (!tc.eta_cdf.empty()) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_eta_cdf),
                          sizeof(double) * n_cell_groups));
  }
  if (!tc.ddmc_mode.empty()) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_mode),
                          sizeof(tenryu::radiation::TransportMode) * n_cell_groups));
  }
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_node_r), sizeof(double) * n_nodes));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_dep), sizeof(double) * n_cell_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_E_tally),
                        sizeof(double) * n_cell_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_escape), sizeof(double) * n_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_numerical_loss), sizeof(double)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_absorbed),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_census),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_left),
                        sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ddmc_leak_right),
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
  cuda_check(cudaMemcpy(d_sigma_leak_left,
                        tc.sigma_leak_left.data(),
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_sigma_leak_right,
                        tc.sigma_leak_right.data(),
                        sizeof(double) * n_cell_groups,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_bc_left,
                        tc.bc_left.data(),
                        sizeof(std::uint8_t) * n_cell_groups,
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_bc_right,
                        tc.bc_right.data(),
                        sizeof(std::uint8_t) * n_cell_groups,
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

  cuda_check(cudaMemset(d_rad_dep, 0, sizeof(double) * n_cell_groups));
  cuda_check(cudaMemset(d_rad_E_tally, 0, sizeof(double) * n_cell_groups));
  cuda_check(cudaMemset(d_E_escape, 0, sizeof(double) * n_groups));
  cuda_check(cudaMemset(d_E_numerical_loss, 0, sizeof(double)));
  cuda_check(cudaMemset(d_ddmc_absorbed, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_census, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_leak_left, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_leak_right, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_leak_boundary, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_converted_to_imc, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_sigma_tot_zero, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_ddmc_max_events_reached, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_flags, 0, sizeof(*d_flags)));

  tenryu::radiation::DDMCTransportGPUInputs in{};
  in.pool = &pool;
  in.sigma_a_eff = d_sigma_a_eff;
  in.sigma_s_eff = d_sigma_s_eff;
  in.sigma_leak_left = d_sigma_leak_left;
  in.sigma_leak_right = d_sigma_leak_right;
  in.bc_left = d_bc_left;
  in.bc_right = d_bc_right;
  in.eta_cdf = d_eta_cdf;
  in.ddmc_mode = d_ddmc_mode;
  in.node_r = d_node_r;
  in.rad_dep = d_rad_dep;
  in.rad_E_tally = d_rad_E_tally;
  in.E_escape = d_E_escape;
  in.E_numerical_loss = d_E_numerical_loss;
  in.ddmc_absorbed = d_ddmc_absorbed;
  in.ddmc_census = d_ddmc_census;
  in.ddmc_leak_left = d_ddmc_leak_left;
  in.ddmc_leak_right = d_ddmc_leak_right;
  in.ddmc_leak_boundary = d_ddmc_leak_boundary;
  in.ddmc_converted_to_imc = d_ddmc_converted_to_imc;
  in.ddmc_sigma_tot_zero = d_ddmc_sigma_tot_zero;
  in.ddmc_max_events_reached = d_ddmc_max_events_reached;
  in.n_cells = tc.n_cells;
  in.n_groups = tc.n_groups;
  in.n_ddmc = tc.n_ddmc;
  in.ddmc_start = tc.ddmc_start;
  in.dt = tc.dt;
  in.step_number = tc.step_number;
  in.user_seed = tc.user_seed;
  in.error_flags = d_flags;
  tenryu::radiation::ddmc_transport_gpu_cuda(in);

  DDMCGpuResult out{};
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
  cuda_check(cudaMemcpy(&out.ddmc_leak_left,
                        d_ddmc_leak_left,
                        sizeof(out.ddmc_leak_left),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out.ddmc_leak_right,
                        d_ddmc_leak_right,
                        sizeof(out.ddmc_leak_right),
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
  cuda_check(cudaFree(d_ddmc_leak_right));
  cuda_check(cudaFree(d_ddmc_leak_left));
  cuda_check(cudaFree(d_ddmc_leak_boundary));
  cuda_check(cudaFree(d_ddmc_census));
  cuda_check(cudaFree(d_ddmc_absorbed));
  cuda_check(cudaFree(d_E_numerical_loss));
  cuda_check(cudaFree(d_E_escape));
  cuda_check(cudaFree(d_rad_E_tally));
  cuda_check(cudaFree(d_rad_dep));
  cuda_check(cudaFree(d_node_r));
  if (d_ddmc_mode != nullptr) {
    cuda_check(cudaFree(d_ddmc_mode));
  }
  if (d_eta_cdf != nullptr) {
    cuda_check(cudaFree(d_eta_cdf));
  }
  cuda_check(cudaFree(d_bc_right));
  cuda_check(cudaFree(d_bc_left));
  cuda_check(cudaFree(d_sigma_leak_right));
  cuda_check(cudaFree(d_sigma_leak_left));
  if (d_sigma_s_eff != nullptr) {
    cuda_check(cudaFree(d_sigma_s_eff));
  }
  cuda_check(cudaFree(d_sigma_a_eff));
  return out;
}

}  // namespace

TEST_CASE("DDMC GPU absorbs particle in high absorption limit", "[radiation][ddmc][gpu]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMCGpuCase tc = make_case(1, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 2.5;
  tc.sigma_a_eff[0] = 1.0e6;
  tc.sigma_leak_left[0] = 0.0;
  tc.sigma_leak_right[0] = 0.0;

  const DDMCGpuResult out = run_case(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.rad_dep[0] == Catch::Approx(2.5).epsilon(1.0e-12));
  REQUIRE(out.ddmc_absorbed == 1ULL);
  REQUIRE(out.ddmc_census == 0ULL);
  REQUIRE(out.flags.invalid_cell == 0);
  REQUIRE(out.flags.infinite_loop == 0);
}

TEST_CASE("DDMC GPU censes particle when event time exceeds dt", "[radiation][ddmc][gpu]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMCGpuCase tc = make_case(1, 1, 1);
  tc.dt = 1.0e-12;
  tc.energy[0] = 3.0;
  tc.sigma_a_eff[0] = 1.0e-20;
  tc.sigma_leak_left[0] = 0.0;
  tc.sigma_leak_right[0] = 0.0;

  const DDMCGpuResult out = run_case(tc);
  const double expected_tally = 3.0 * tenryu::core::constants::c_light * tc.dt;
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeDDMC);
  REQUIRE(out.time_remain[0] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out.energy[0] == Catch::Approx(3.0).epsilon(1.0e-12));
  REQUIRE(out.rad_dep[0] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out.rad_E_tally[0] == Catch::Approx(expected_tally).epsilon(1.0e-12));
  REQUIRE(out.ddmc_census == 1ULL);
  REQUIRE(out.flags.invalid_cell == 0);
}

TEST_CASE("DDMC GPU leaks particle to vacuum boundary", "[radiation][ddmc][gpu]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMCGpuCase tc = make_case(1, 1, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 1.75;
  tc.sigma_a_eff[0] = 0.0;
  tc.sigma_leak_left[0] = 1.0e6;
  tc.sigma_leak_right[0] = 0.0;
  tc.bc_left[0] = static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);
  tc.bc_right[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Reflective);

  const DDMCGpuResult out = run_case(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.E_escape[0] == Catch::Approx(1.75).epsilon(1.0e-12));
  REQUIRE(out.rad_dep[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.ddmc_leak_boundary == 1ULL);
  REQUIRE(out.flags.invalid_cell == 0);
}

TEST_CASE("DDMC GPU signed all-positive tallies match legacy sums",
          "[radiation][ddmc][gpu][signed]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  constexpr int n_particles = 3;
  const std::vector<double> energy = {1.0, 2.0, 3.0};
  const std::vector<std::int8_t> sign(n_particles, 1);
  constexpr double expected_net = 6.0;

  DDMCGpuCase absorb = make_case(1, 1, n_particles);
  absorb.dt = 1.0e-10;
  absorb.energy = energy;
  absorb.sign = sign;
  absorb.sigma_a_eff[0] = 1.0e6;
  DDMCGpuResult out_absorb = run_case(absorb);
  REQUIRE(out_absorb.rad_dep[0] == Catch::Approx(expected_net).epsilon(1.0e-12));
  REQUIRE(out_absorb.ddmc_absorbed == static_cast<unsigned long long>(n_particles));

  DDMCGpuCase census = make_case(1, 1, n_particles);
  census.dt = 1.0e-12;
  census.energy = energy;
  census.sign = sign;
  DDMCGpuResult out_census = run_case(census);
  REQUIRE(out_census.rad_E_tally[0] ==
          Catch::Approx(expected_net * tenryu::core::constants::c_light * census.dt)
              .epsilon(1.0e-12));
  REQUIRE(out_census.ddmc_sigma_tot_zero ==
          static_cast<unsigned long long>(n_particles));

  DDMCGpuCase escape = make_case(1, 1, n_particles);
  escape.dt = 1.0e-10;
  escape.energy = energy;
  escape.sign = sign;
  escape.sigma_leak_left[0] = 1.0e6;
  escape.bc_left[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);
  escape.bc_right[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Reflective);
  DDMCGpuResult out_escape = run_case(escape);
  REQUIRE(out_escape.E_escape[0] == Catch::Approx(expected_net).epsilon(1.0e-12));
  REQUIRE(out_escape.ddmc_leak_boundary ==
          static_cast<unsigned long long>(n_particles));
}

TEST_CASE("DDMC GPU signed opposite particles cancel tallies",
          "[radiation][ddmc][gpu][signed]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  constexpr int n_particles = 2;
  const std::vector<double> energy(n_particles, 4.0);
  const std::vector<std::int8_t> sign = {1, -1};

  DDMCGpuCase absorb = make_case(1, 1, n_particles);
  absorb.dt = 1.0e-10;
  absorb.energy = energy;
  absorb.sign = sign;
  absorb.sigma_a_eff[0] = 1.0e6;
  DDMCGpuResult out_absorb = run_case(absorb);
  REQUIRE(out_absorb.rad_dep[0] == Catch::Approx(0.0).margin(1.0e-12));

  DDMCGpuCase census = make_case(1, 1, n_particles);
  census.dt = 1.0e-12;
  census.energy = energy;
  census.sign = sign;
  DDMCGpuResult out_census = run_case(census);
  REQUIRE(out_census.rad_E_tally[0] == Catch::Approx(0.0).margin(1.0e-12));

  DDMCGpuCase escape = make_case(1, 1, n_particles);
  escape.dt = 1.0e-10;
  escape.energy = energy;
  escape.sign = sign;
  escape.sigma_leak_left[0] = 1.0e6;
  escape.bc_left[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);
  escape.bc_right[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Reflective);
  DDMCGpuResult out_escape = run_case(escape);
  REQUIRE(out_escape.E_escape[0] == Catch::Approx(0.0).margin(1.0e-12));
}

TEST_CASE("DDMC GPU signed mixed particles tally net energy",
          "[radiation][ddmc][gpu][signed]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  constexpr int n_particles = 3;
  const std::vector<double> energy = {2.0, 3.0, 5.0};
  const std::vector<std::int8_t> sign = {1, -1, 1};
  constexpr double expected_net = 4.0;

  DDMCGpuCase absorb = make_case(1, 1, n_particles);
  absorb.dt = 1.0e-10;
  absorb.energy = energy;
  absorb.sign = sign;
  absorb.sigma_a_eff[0] = 1.0e6;
  DDMCGpuResult out_absorb = run_case(absorb);
  REQUIRE(out_absorb.rad_dep[0] == Catch::Approx(expected_net).epsilon(1.0e-12));

  DDMCGpuCase census = make_case(1, 1, n_particles);
  census.dt = 1.0e-12;
  census.energy = energy;
  census.sign = sign;
  DDMCGpuResult out_census = run_case(census);
  REQUIRE(out_census.rad_E_tally[0] ==
          Catch::Approx(expected_net * tenryu::core::constants::c_light * census.dt)
              .epsilon(1.0e-12));

  DDMCGpuCase escape = make_case(1, 1, n_particles);
  escape.dt = 1.0e-10;
  escape.energy = energy;
  escape.sign = sign;
  escape.sigma_leak_left[0] = 1.0e6;
  escape.bc_left[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);
  escape.bc_right[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Reflective);
  DDMCGpuResult out_escape = run_case(escape);
  REQUIRE(out_escape.E_escape[0] == Catch::Approx(expected_net).epsilon(1.0e-12));
}

TEST_CASE("DDMC GPU converts particle to IMC at interface", "[radiation][ddmc][gpu]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMCGpuCase tc = make_case(2, 1, 2);
  tc.dt = 1.0e-10;
  tc.ddmc_start = 1;
  tc.n_ddmc = 1;
  tc.node_r = {0.0, 1.0, 2.0};

  tc.mode[0] = tenryu::radiation::kModeIMC;
  tc.alive[0] = tenryu::radiation::kAlive;
  tc.energy[0] = 4.0;
  tc.cell_id[0] = 0;
  tc.time_remain[0] = tc.dt;

  tc.mode[1] = tenryu::radiation::kModeDDMC;
  tc.alive[1] = tenryu::radiation::kAlive;
  tc.energy[1] = 5.0;
  tc.cell_id[1] = 0;
  tc.group_id[1] = 0;
  tc.time_remain[1] = tc.dt;
  tc.pos_r[1] = 0.25;

  tc.sigma_a_eff[0] = 0.0;
  tc.sigma_a_eff[1] = 0.0;
  tc.sigma_leak_left[0] = 0.0;
  tc.sigma_leak_right[0] = 1.0e6;
  tc.bc_left[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal);
  tc.bc_right[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Interface);

  const DDMCGpuResult out = run_case(tc);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeIMC);
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.energy[0] == Catch::Approx(4.0).epsilon(1.0e-12));

  REQUIRE(out.alive[1] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[1] == tenryu::radiation::kModeIMC);
  REQUIRE(out.cell_id[1] == 1);
  REQUIRE(out.energy[1] == Catch::Approx(5.0).epsilon(1.0e-12));
  REQUIRE(out.time_remain[1] > 0.0);
  REQUIRE(out.time_remain[1] < tc.dt);
  REQUIRE(out.pos_r[1] == Catch::Approx(1.0).epsilon(1.0e-12));
  REQUIRE(out.dir_r[1] > 0.0);
  const double dir_norm2 = out.dir_r[1] * out.dir_r[1] +
                           out.dir_z[1] * out.dir_z[1] +
                           out.dir_phi[1] * out.dir_phi[1];
  REQUIRE(dir_norm2 == Catch::Approx(1.0).epsilon(1.0e-12));
  REQUIRE(out.ddmc_converted_to_imc == 1ULL);
  REQUIRE(out.E_escape[0] == Catch::Approx(0.0).margin(1.0e-14));
}

TEST_CASE("DDMC GPU same-cell NLTE scatter demotes to IMC",
          "[radiation][ddmc][gpu][nlte]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMCGpuCase tc = make_case(1, 2, 1);
  tc.dt = 1.0e-10;
  tc.energy[0] = 2.0;
  tc.group_id[0] = 0U;
  tc.sigma_a_eff[0] = 0.0;
  tc.sigma_a_eff[1] = 0.0;
  tc.sigma_s_eff[0] = 1.0e6;
  tc.sigma_s_eff[1] = 0.0;
  tc.eta_cdf = {0.0, 1.0};
  tc.ddmc_mode = {tenryu::radiation::TransportMode::DDMC,
                  tenryu::radiation::TransportMode::IMC};

  const DDMCGpuResult out = run_case(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out.mode[0] == tenryu::radiation::kModeIMC);
  REQUIRE(out.cell_id[0] == 0);
  REQUIRE(out.group_id[0] == 1U);
  REQUIRE(out.energy[0] == Catch::Approx(2.0).epsilon(1.0e-12));
  REQUIRE(out.time_remain[0] > 0.0);
  REQUIRE(out.time_remain[0] < tc.dt);
  REQUIRE(out.pos_r[0] >= tc.node_r[0]);
  REQUIRE(out.pos_r[0] <= tc.node_r[1]);
  const double dir_norm2 = out.dir_r[0] * out.dir_r[0] +
                           out.dir_z[0] * out.dir_z[0] +
                           out.dir_phi[0] * out.dir_phi[0];
  REQUIRE(dir_norm2 == Catch::Approx(1.0).epsilon(1.0e-12));
  REQUIRE(out.ddmc_converted_to_imc == 1ULL);
  REQUIRE(out.rad_dep[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.rad_dep[1] == Catch::Approx(0.0).margin(1.0e-14));
}

TEST_CASE("DDMC GPU conserves energy over many particles", "[radiation][ddmc][gpu]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  constexpr int n_particles = 4096;
  DDMCGpuCase tc = make_case(1, 1, n_particles);
  tc.dt = 1.0e-12;
  tc.energy.assign(static_cast<std::size_t>(n_particles), 2.0);
  tc.time_remain.assign(static_cast<std::size_t>(n_particles), tc.dt);
  tc.sigma_a_eff[0] = 1.0;
  tc.sigma_leak_left[0] = 1.0;
  tc.sigma_leak_right[0] = 1.0;
  tc.bc_left[0] = static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);
  tc.bc_right[0] = static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Vacuum);

  const DDMCGpuResult out = run_case(tc);
  const double initial_total = static_cast<double>(n_particles) * 2.0;
  const double dep_sum = out.rad_dep[0];
  const double escape_sum = out.E_escape[0];
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

TEST_CASE("DDMC GPU sigma_tot floor triggers census and counter",
          "[radiation][ddmc][gpu]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMCGpuCase tc = make_case(1, 1, 1);
  tc.dt = 1.0e-12;
  tc.energy[0] = 2.0;
  tc.sigma_a_eff[0] = 0.0;
  tc.sigma_leak_left[0] = 0.0;
  tc.sigma_leak_right[0] = 0.0;

  const DDMCGpuResult out = run_case(tc);
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

TEST_CASE("DDMC GPU invalid cell is killed with numerical loss",
          "[radiation][ddmc][gpu]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMCGpuCase tc = make_case(1, 1, 1);
  tc.dt = 1.0e-12;
  tc.energy[0] = 4.5;
  tc.cell_id[0] = tc.n_cells;
  tc.sigma_a_eff[0] = 1.0;

  const DDMCGpuResult out = run_case(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.E_numerical_loss == Catch::Approx(4.5).epsilon(1.0e-12));
  REQUIRE(out.flags.invalid_cell == 1);
}

TEST_CASE("DDMC GPU max events deposits energy and increments counter",
          "[radiation][ddmc][gpu]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMCGpuCase tc = make_case(2, 1, 1);
  tc.dt = 1.0e-12;
  tc.energy[0] = 3.25;
  tc.cell_id[0] = 0;
  tc.sigma_a_eff[0] = 0.0;
  tc.sigma_a_eff[1] = 0.0;
  tc.sigma_leak_left[0] = 0.0;
  tc.sigma_leak_right[0] = 1.0e8;
  tc.sigma_leak_left[1] = 1.0e8;
  tc.sigma_leak_right[1] = 0.0;
  tc.bc_left[0] = static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal);
  tc.bc_right[0] = static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal);
  tc.bc_left[1] = static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal);
  tc.bc_right[1] = static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Internal);

  const DDMCGpuResult out = run_case(tc);
  const double dep_sum = out.rad_dep[0] + out.rad_dep[1];
  REQUIRE(out.ddmc_max_events_reached == 1ULL);
  REQUIRE(out.flags.infinite_loop == 1);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(dep_sum == Catch::Approx(3.25).epsilon(1.0e-12));
}

TEST_CASE("DDMC GPU reflective leak fallback deposits and kills particle",
          "[radiation][ddmc][gpu]") {
  if (!has_cuda_device()) {
    INFO("Skipping DDMC GPU test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  DDMCGpuCase tc = make_case(1, 1, 1);
  tc.dt = 1.0e-12;
  tc.energy[0] = 1.0;
  tc.sigma_a_eff[0] = 0.0;
  tc.sigma_leak_left[0] = 1.0e4;
  tc.sigma_leak_right[0] = 0.0;
  tc.bc_left[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Reflective);
  tc.bc_right[0] =
      static_cast<std::uint8_t>(tenryu::radiation::DDMCBoundaryType::Reflective);

  const DDMCGpuResult out = run_case(tc);
  REQUIRE(out.alive[0] == tenryu::radiation::kDead);
  REQUIRE(out.energy[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.rad_dep[0] == Catch::Approx(1.0).epsilon(1.0e-12));
  REQUIRE(out.E_escape[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(out.ddmc_absorbed == 1ULL);
  REQUIRE(out.ddmc_census == 0ULL);
  REQUIRE(out.flags.invalid_boundary == 1);
}
