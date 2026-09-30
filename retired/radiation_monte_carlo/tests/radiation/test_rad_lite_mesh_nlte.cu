#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/device_error_flags.cuh"
#include "core/error.hpp"
#include "radiation/imc_transport_persistent.cuh"
#include "radiation/particle_pool.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_rad_lite_mesh_nlte");
}

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

}  // namespace

TEST_CASE("RadLite NLTE donor eta_cdf is selected from hydro owner",
          "[radiation][rad_lite][nlte]") {
  if (!has_cuda_device()) {
    INFO("Skipping RadLite NLTE test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  constexpr int n_particles = 2;
  constexpr int n_groups = 2;
  constexpr int n_rad_cells = 1;
  constexpr int n_hydro_cells = 2;
  constexpr double dt = 1.0e-13;

  tenryu::radiation::PhotonPool pool;
  pool.allocate(n_particles);
  pool.n_alive = n_particles;

  const std::vector<double> pos_r = {0.25, 1.75};
  const std::vector<double> pos_z = {0.0, 0.0};
  const std::vector<double> dir_r = {0.0, 0.0};
  const std::vector<double> dir_z = {0.0, 0.0};
  const std::vector<double> dir_phi = {0.0, 0.0};
  const std::vector<double> energy = {1.0, 1.0};
  const std::vector<double> birth_energy = {1.0, 1.0};
  const std::vector<double> weight = {1.0, 1.0};
  const std::vector<double> time_remain = {dt, dt};
  const std::vector<std::uint64_t> global_id = {11ULL, 12ULL};
  const std::vector<std::uint32_t> rng_counter = {0U, 0U};
  const std::vector<std::int32_t> cell_id = {0, 0};
  const std::vector<std::uint16_t> group_id = {0U, 0U};
  const std::vector<std::uint8_t> mode = {tenryu::radiation::kModeIMC,
                                          tenryu::radiation::kModeIMC};
  const std::vector<std::uint8_t> alive = {tenryu::radiation::kAlive,
                                           tenryu::radiation::kAlive};

  cuda_check(cudaMemcpy(pool.pos_r, pos_r.data(), sizeof(double) * pos_r.size(), cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.pos_z, pos_z.data(), sizeof(double) * pos_z.size(), cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_r, dir_r.data(), sizeof(double) * dir_r.size(), cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_z, dir_z.data(), sizeof(double) * dir_z.size(), cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_phi,
                        dir_phi.data(),
                        sizeof(double) * dir_phi.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.energy,
                        energy.data(),
                        sizeof(double) * energy.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.birth_energy,
                        birth_energy.data(),
                        sizeof(double) * birth_energy.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.weight,
                        weight.data(),
                        sizeof(double) * weight.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.time_remain,
                        time_remain.data(),
                        sizeof(double) * time_remain.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.global_id,
                        global_id.data(),
                        sizeof(std::uint64_t) * global_id.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.rng_counter,
                        rng_counter.data(),
                        sizeof(std::uint32_t) * rng_counter.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.cell_id,
                        cell_id.data(),
                        sizeof(std::int32_t) * cell_id.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.group_id,
                        group_id.data(),
                        sizeof(std::uint16_t) * group_id.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.mode,
                        mode.data(),
                        sizeof(std::uint8_t) * mode.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.alive,
                        alive.data(),
                        sizeof(std::uint8_t) * alive.size(),
                        cudaMemcpyHostToDevice));

  double* d_sigma_a_eff = nullptr;
  double* d_sigma_s_eff = nullptr;
  double* d_Te = nullptr;
  double* d_vol = nullptr;
  double* d_node_r = nullptr;
  double* d_eta_cdf_hydro = nullptr;
  double* d_hydro_node_r = nullptr;
  std::int32_t* d_rad_h_begin = nullptr;
  std::int32_t* d_rad_h_end = nullptr;
  double* d_rad_dep = nullptr;
  double* d_rad_E_tally = nullptr;
  double* d_E_escape = nullptr;
  double* d_E_numerical_loss = nullptr;
  unsigned long long* d_imc_absorbed = nullptr;
  unsigned long long* d_imc_escaped = nullptr;
  unsigned long long* d_cnt_boundary = nullptr;
  unsigned long long* d_cnt_scatter = nullptr;
  unsigned long long* d_cnt_census = nullptr;
  unsigned long long* d_cnt_absorb_kill = nullptr;
  unsigned long long* d_cnt_absorb_survive = nullptr;
  unsigned long long* d_cnt_roulette_kill = nullptr;
  tenryu::core::DeviceErrorFlags* d_flags = nullptr;

  const std::vector<double> sigma_a_eff = {0.0, 0.0};
  const std::vector<double> sigma_s_eff = {1.0e6, 0.0};
  const std::vector<double> Te = {1.0};
  const std::vector<double> vol = {2.0};
  const std::vector<double> node_r = {0.0, 2.0};
  const std::vector<double> eta_cdf_hydro = {
      1.0, 1.0,  // left hydro cell: always group 0
      0.0, 1.0   // right hydro cell: always group 1
  };
  const std::vector<double> hydro_node_r = {0.0, 1.0, 2.0};
  const std::vector<std::int32_t> rad_h_begin = {0};
  const std::vector<std::int32_t> rad_h_end = {2};

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff), sizeof(double) * n_rad_cells * n_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_s_eff), sizeof(double) * n_rad_cells * n_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_Te), sizeof(double) * n_rad_cells));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vol), sizeof(double) * n_rad_cells));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_node_r), sizeof(double) * (n_rad_cells + 1)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_eta_cdf_hydro),
                        sizeof(double) * n_hydro_cells * n_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_hydro_node_r),
                        sizeof(double) * (n_hydro_cells + 1)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_h_begin), sizeof(std::int32_t) * n_rad_cells));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_h_end), sizeof(std::int32_t) * n_rad_cells));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_dep), sizeof(double) * n_rad_cells * n_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_E_tally), sizeof(double) * n_rad_cells * n_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_escape), sizeof(double) * n_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_numerical_loss), sizeof(double)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_imc_absorbed), sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_imc_escaped), sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_boundary), sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_scatter), sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_census), sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_absorb_kill), sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_absorb_survive), sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cnt_roulette_kill), sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_flags), sizeof(*d_flags)));

  cuda_check(cudaMemcpy(d_sigma_a_eff,
                        sigma_a_eff.data(),
                        sizeof(double) * sigma_a_eff.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_sigma_s_eff,
                        sigma_s_eff.data(),
                        sizeof(double) * sigma_s_eff.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_Te, Te.data(), sizeof(double) * Te.size(), cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_vol, vol.data(), sizeof(double) * vol.size(), cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_node_r,
                        node_r.data(),
                        sizeof(double) * node_r.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_eta_cdf_hydro,
                        eta_cdf_hydro.data(),
                        sizeof(double) * eta_cdf_hydro.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_hydro_node_r,
                        hydro_node_r.data(),
                        sizeof(double) * hydro_node_r.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_rad_h_begin,
                        rad_h_begin.data(),
                        sizeof(std::int32_t) * rad_h_begin.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_rad_h_end,
                        rad_h_end.data(),
                        sizeof(std::int32_t) * rad_h_end.size(),
                        cudaMemcpyHostToDevice));

  cuda_check(cudaMemset(d_rad_dep, 0, sizeof(double) * n_rad_cells * n_groups));
  cuda_check(cudaMemset(d_rad_E_tally, 0, sizeof(double) * n_rad_cells * n_groups));
  cuda_check(cudaMemset(d_E_escape, 0, sizeof(double) * n_groups));
  cuda_check(cudaMemset(d_E_numerical_loss, 0, sizeof(double)));
  cuda_check(cudaMemset(d_imc_absorbed, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_imc_escaped, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_cnt_boundary, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_cnt_scatter, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_cnt_census, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_cnt_absorb_kill, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_cnt_absorb_survive, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_cnt_roulette_kill, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_flags, 0, sizeof(*d_flags)));

  tenryu::radiation::TransportInputs in{};
  in.pool = &pool;
  in.sigma_a_eff = d_sigma_a_eff;
  in.sigma_s_eff = d_sigma_s_eff;
  in.Te = d_Te;
  in.vol = d_vol;
  in.node_r = d_node_r;
  in.rad_dep = d_rad_dep;
  in.rad_E_tally = d_rad_E_tally;
  in.E_escape = d_E_escape;
  in.E_numerical_loss = d_E_numerical_loss;
  in.imc_absorbed = d_imc_absorbed;
  in.imc_escaped = d_imc_escaped;
  in.cnt_boundary = d_cnt_boundary;
  in.cnt_scatter = d_cnt_scatter;
  in.cnt_census = d_cnt_census;
  in.cnt_absorb_kill = d_cnt_absorb_kill;
  in.cnt_absorb_survive = d_cnt_absorb_survive;
  in.cnt_roulette_kill = d_cnt_roulette_kill;
  in.n_cells = n_rad_cells;
  in.n_groups = n_groups;
  in.n_imc = n_particles;
  in.dt = dt;
  in.eta_cdf_hydro = d_eta_cdf_hydro;
  in.hydro_node_r = d_hydro_node_r;
  in.rad_h_begin = d_rad_h_begin;
  in.rad_h_end = d_rad_h_end;
  in.inelastic_scatter = true;
  in.bc_inner = tenryu::radiation::kBoundaryReflect;
  in.bc_outer = tenryu::radiation::kBoundaryReflect;
  in.step_number = 9;
  in.user_seed = 424242ULL;
  in.error_flags = d_flags;

  tenryu::radiation::imc_transport_persistent_cuda(in);

  std::vector<std::uint16_t> out_group(n_particles, 0U);
  std::vector<std::uint8_t> out_alive(n_particles, 0U);
  std::vector<double> out_time(n_particles, 0.0);
  unsigned long long out_scatter = 0ULL;

  cuda_check(cudaMemcpy(out_group.data(),
                        pool.group_id,
                        sizeof(std::uint16_t) * out_group.size(),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out_alive.data(),
                        pool.alive,
                        sizeof(std::uint8_t) * out_alive.size(),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(out_time.data(),
                        pool.time_remain,
                        sizeof(double) * out_time.size(),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&out_scatter,
                        d_cnt_scatter,
                        sizeof(out_scatter),
                        cudaMemcpyDeviceToHost));

  REQUIRE(out_alive[0] == tenryu::radiation::kAlive);
  REQUIRE(out_alive[1] == tenryu::radiation::kAlive);
  REQUIRE(out_group[0] == 0U);
  REQUIRE(out_group[1] == 1U);
  REQUIRE(out_time[0] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out_time[1] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(out_scatter > 0ULL);

  cuda_check(cudaFree(d_flags));
  cuda_check(cudaFree(d_cnt_roulette_kill));
  cuda_check(cudaFree(d_cnt_absorb_survive));
  cuda_check(cudaFree(d_cnt_absorb_kill));
  cuda_check(cudaFree(d_cnt_census));
  cuda_check(cudaFree(d_cnt_scatter));
  cuda_check(cudaFree(d_cnt_boundary));
  cuda_check(cudaFree(d_imc_escaped));
  cuda_check(cudaFree(d_imc_absorbed));
  cuda_check(cudaFree(d_E_numerical_loss));
  cuda_check(cudaFree(d_E_escape));
  cuda_check(cudaFree(d_rad_E_tally));
  cuda_check(cudaFree(d_rad_dep));
  cuda_check(cudaFree(d_rad_h_end));
  cuda_check(cudaFree(d_rad_h_begin));
  cuda_check(cudaFree(d_hydro_node_r));
  cuda_check(cudaFree(d_eta_cdf_hydro));
  cuda_check(cudaFree(d_node_r));
  cuda_check(cudaFree(d_vol));
  cuda_check(cudaFree(d_Te));
  cuda_check(cudaFree(d_sigma_s_eff));
  cuda_check(cudaFree(d_sigma_a_eff));
}
