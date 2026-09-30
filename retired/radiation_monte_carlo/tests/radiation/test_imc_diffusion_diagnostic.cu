#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/device_error_flags.cuh"
#include "core/error.hpp"
#include "core/field.hpp"
#include "radiation/boundary.cuh"
#include "radiation/fleck.cuh"
#include "radiation/imc_transport_persistent.cuh"
#include "radiation/particle_pool.cuh"
#include "radiation/tally.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_imc_diffusion_diagnostic");
}

std::string format_rad_dep(const std::vector<double>& rad_dep) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6);
  for (std::size_t i = 0; i < rad_dep.size(); ++i) {
    oss << "cell[" << i << "] = " << rad_dep[i];
    if (i + 1 != rad_dep.size()) {
      oss << '\n';
    }
  }
  return oss.str();
}

}  // namespace

TEST_CASE("IMC transport deposits energy outside source cell", "[radiation][imc][diagnostic]") {
  int device_count = 0;
  const cudaError_t device_err = cudaGetDeviceCount(&device_count);
  if (device_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping IMC diffusion diagnostic because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  constexpr int n_cells = 10;
  constexpr int n_groups = 1;
  constexpr int n_particles = 100;
  constexpr int n_cell_groups = n_cells * n_groups;
  constexpr double r_min = 10000.0;
  constexpr double r_max = 10000.25;
  constexpr double dr = (r_max - r_min) / static_cast<double>(n_cells);
  constexpr double dt = 3.33e-12;
  constexpr double particle_energy = 1.0e6;
  constexpr double pi = 3.14159265358979323846;

  tenryu::core::Config cfg;
  tenryu::core::Config::MaterialsConfig::MatDef mat;
  mat.name = "diag_material";
  mat.A = 1.0;
  mat.ideal_gas_gamma = 5.0 / 3.0;
  mat.kappa_a_constant = 1.0;
  mat.cv_e_override = 548.8;
  cfg.materials.materials = {mat};
  cfg.radiation.imc.alpha = 1.0;
  cfg.radiation.imc.f_max = 1.0;
  cfg.main.seed = 12345;

  tenryu::core::CellField1D rho(n_cells);
  tenryu::core::CellField1D Te(n_cells);
  tenryu::core::CellField1D zbar(n_cells);
  tenryu::core::CellField1D vol(n_cells);
  tenryu::core::NodeField1D node_r(n_cells + 1);
  tenryu::core::GroupField1D rad_dep(n_cell_groups);

  std::vector<double> rho_host(n_cells, 1.0);
  std::vector<double> Te_host(n_cells, 1.0);
  std::vector<double> zbar_host(n_cells, 1.0);
  std::vector<double> node_r_host(n_cells + 1, 0.0);
  std::vector<double> vol_host(n_cells, 0.0);

  for (int i = 0; i <= n_cells; ++i) {
    node_r_host[i] = r_min + dr * static_cast<double>(i);
  }
  for (int c = 0; c < n_cells; ++c) {
    const double r_lo = node_r_host[c];
    const double r_hi = node_r_host[c + 1];
    vol_host[c] = (4.0 / 3.0) * pi * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
  }

  rho.copy_from_host(rho_host);
  Te.copy_from_host(Te_host);
  zbar.copy_from_host(zbar_host);
  vol.copy_from_host(vol_host);
  node_r.copy_from_host(node_r_host);

  double* d_sigma_a = nullptr;
  double* d_f_fleck = nullptr;
  double* d_sigma_a_eff = nullptr;
  double* d_sigma_s_eff = nullptr;
  double* d_rad_E_tally = nullptr;
  double* d_E_escape = nullptr;
  double* d_E_numerical_loss = nullptr;
  unsigned long long* d_imc_absorbed = nullptr;
  unsigned long long* d_imc_escaped = nullptr;
  tenryu::core::DeviceErrorFlags* d_error_flags = nullptr;

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a),
                        sizeof(double) * static_cast<std::size_t>(n_cell_groups)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_f_fleck), sizeof(double) * n_cells));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff),
                        sizeof(double) * static_cast<std::size_t>(n_cell_groups)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_s_eff),
                        sizeof(double) * static_cast<std::size_t>(n_cell_groups)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_E_tally),
                        sizeof(double) * static_cast<std::size_t>(n_cell_groups)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_escape), sizeof(double) * n_groups));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_error_flags), sizeof(*d_error_flags)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_numerical_loss), sizeof(double)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_imc_absorbed), sizeof(unsigned long long)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_imc_escaped), sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_E_numerical_loss, 0, sizeof(double)));
  cuda_check(cudaMemset(d_imc_absorbed, 0, sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_imc_escaped, 0, sizeof(unsigned long long)));

  const double sigma_a_value = mat.kappa_a_constant * rho_host[0];
  std::vector<double> sigma_a_host(n_cell_groups, sigma_a_value);
  std::vector<std::uint8_t> cell_is_void_host(n_cells, 0U);
  cuda_check(cudaMemcpy(d_sigma_a,
                        sigma_a_host.data(),
                        sizeof(double) * sigma_a_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemset(d_error_flags, 0, sizeof(*d_error_flags)));

  tenryu::radiation::FleckView fleck_view;
  fleck_view.rho = rho.data();
  fleck_view.Te = Te.data();
  fleck_view.zbar = zbar.data();
  fleck_view.cell_is_void = cell_is_void_host.data();
  fleck_view.sigma_a = d_sigma_a;
  fleck_view.f_fleck = d_f_fleck;
  fleck_view.sigma_a_eff = d_sigma_a_eff;
  fleck_view.sigma_s_eff = d_sigma_s_eff;
  fleck_view.n_cells = n_cells;
  fleck_view.n_groups = n_groups;
  fleck_view.dt = dt;
  tenryu::radiation::compute_fleck_and_sigma_eff_cuda(fleck_view, cfg);

  tenryu::radiation::zero_tallies_cuda(rad_dep.data(),
                                       d_rad_E_tally,
                                       d_E_escape,
                                       n_cells,
                                       n_groups);

  tenryu::radiation::PhotonPool pool;
  pool.allocate(n_particles);
  pool.n_alive = n_particles;

  const double r0 = r_min + 0.5 * dr;
  std::vector<double> pos_r_host(n_particles, r0);
  std::vector<double> pos_z_host(n_particles, 0.0);
  std::vector<double> dir_r_host(n_particles, 0.5);
  std::vector<double> dir_z_host(n_particles, 0.0);
  std::vector<double> dir_phi_host(n_particles, 0.0);
  std::vector<double> energy_host(n_particles, particle_energy);
  std::vector<double> weight_host(n_particles, 1.0);
  std::vector<double> time_remain_host(n_particles, dt);
  std::vector<double> birth_energy_host(n_particles, particle_energy);
  std::vector<std::uint64_t> global_id_host(n_particles, 0);
  std::vector<std::uint32_t> rng_counter_host(n_particles, 0);
  std::vector<std::int32_t> cell_id_host(n_particles, 0);
  std::vector<std::uint16_t> group_id_host(n_particles, 0);
  std::vector<std::uint8_t> mode_host(n_particles, tenryu::radiation::kModeIMC);
  std::vector<std::uint8_t> alive_host(n_particles, tenryu::radiation::kAlive);

  for (int p = 0; p < n_particles; ++p) {
    global_id_host[p] = static_cast<std::uint64_t>(p);
  }

  cuda_check(cudaMemcpy(pool.pos_r,
                        pos_r_host.data(),
                        sizeof(double) * pos_r_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.pos_z,
                        pos_z_host.data(),
                        sizeof(double) * pos_z_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_r,
                        dir_r_host.data(),
                        sizeof(double) * dir_r_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_z,
                        dir_z_host.data(),
                        sizeof(double) * dir_z_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_phi,
                        dir_phi_host.data(),
                        sizeof(double) * dir_phi_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.energy,
                        energy_host.data(),
                        sizeof(double) * energy_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.weight,
                        weight_host.data(),
                        sizeof(double) * weight_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.time_remain,
                        time_remain_host.data(),
                        sizeof(double) * time_remain_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.birth_energy,
                        birth_energy_host.data(),
                        sizeof(double) * birth_energy_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.global_id,
                        global_id_host.data(),
                        sizeof(std::uint64_t) * global_id_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.rng_counter,
                        rng_counter_host.data(),
                        sizeof(std::uint32_t) * rng_counter_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.cell_id,
                        cell_id_host.data(),
                        sizeof(std::int32_t) * cell_id_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.group_id,
                        group_id_host.data(),
                        sizeof(std::uint16_t) * group_id_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.mode,
                        mode_host.data(),
                        sizeof(std::uint8_t) * mode_host.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.alive,
                        alive_host.data(),
                        sizeof(std::uint8_t) * alive_host.size(),
                        cudaMemcpyHostToDevice));

  tenryu::radiation::TransportInputs t_in;
  t_in.pool = &pool;
  t_in.sigma_a_eff = d_sigma_a_eff;
  t_in.sigma_s_eff = d_sigma_s_eff;
  t_in.Te = Te.data();
  t_in.vol = vol.data();
  t_in.node_r = node_r.data();
  t_in.rad_dep = rad_dep.data();
  t_in.rad_E_tally = d_rad_E_tally;
  t_in.E_escape = d_E_escape;
  t_in.n_cells = n_cells;
  t_in.n_groups = n_groups;
  t_in.n_imc = n_particles;
  t_in.dt = dt;
  t_in.E_avg = particle_energy;
  t_in.w_cutoff = 1.0e-10;
  t_in.p_survival = 0.1;
  t_in.f_cutoff = 0.0;
  t_in.bc_inner = tenryu::radiation::kBoundaryReflect;
  t_in.bc_outer = tenryu::radiation::kBoundaryVacuum;
  t_in.step_number = 0;
  t_in.user_seed = cfg.main.seed;
  t_in.error_flags = d_error_flags;
  t_in.E_numerical_loss = d_E_numerical_loss;
  t_in.imc_absorbed = d_imc_absorbed;
  t_in.imc_escaped = d_imc_escaped;

  tenryu::radiation::imc_transport_persistent_cuda(t_in);

  std::vector<double> rad_dep_host(n_cell_groups, 0.0);
  rad_dep.copy_to_host(rad_dep_host.data());

  std::vector<std::int32_t> cell_after(n_particles, -1);
  cuda_check(cudaMemcpy(cell_after.data(),
                        pool.cell_id,
                        sizeof(std::int32_t) * cell_after.size(),
                        cudaMemcpyDeviceToHost));

  tenryu::core::DeviceErrorFlags host_flags{};
  cuda_check(cudaMemcpy(&host_flags,
                        d_error_flags,
                        sizeof(host_flags),
                        cudaMemcpyDeviceToHost));

  const double dep_cell0 = rad_dep_host[0];
  const double dep_cell1 = rad_dep_host[1];
  const double dep_beyond_cell0 =
      std::accumulate(rad_dep_host.begin() + 1, rad_dep_host.end(), 0.0);

  int nonzero_cells_beyond0 = 0;
  for (int c = 1; c < n_cells; ++c) {
    if (rad_dep_host[c] > 0.0) {
      nonzero_cells_beyond0 += 1;
    }
  }

  int moved_particles = 0;
  std::vector<int> cell_hist(n_cells, 0);
  for (const std::int32_t c : cell_after) {
    if (c > 0) {
      moved_particles += 1;
    }
    if (c >= 0 && c < n_cells) {
      cell_hist[c] += 1;
    }
  }

  const std::string rad_dep_text = format_rad_dep(rad_dep_host);
  std::ostringstream moved_oss;
  moved_oss << "moved_particles(cell_id>0): " << moved_particles << "/" << n_particles
            << "\ncell_id histogram:";
  for (int c = 0; c < n_cells; ++c) {
    moved_oss << " c" << c << "=" << cell_hist[c];
  }
  const std::string moved_text = moved_oss.str();

  INFO("IMC diffusion diagnostic rad_dep:\n" << rad_dep_text);
  INFO("IMC diffusion diagnostic particle motion:\n" << moved_text);
  std::cout << "[imc-diagnostic] rad_dep\n" << rad_dep_text << '\n';
  std::cout << "[imc-diagnostic] " << moved_text << '\n';

  const bool ok_dep_cell0 = dep_cell0 > 0.0;
  const bool ok_dep_cell1 = dep_cell1 > 0.0;
  const bool ok_any_beyond0 = dep_beyond_cell0 > 0.0;
  const bool ok_nonzero_beyond0 = nonzero_cells_beyond0 > 0;
  const bool ok_moved_particles = moved_particles > 0;
  const bool ok_invalid_cell_flag = (host_flags.invalid_cell == 0);
  const bool ok_infinite_loop_flag = (host_flags.infinite_loop == 0);

  cuda_check(cudaFree(d_imc_escaped));
  cuda_check(cudaFree(d_imc_absorbed));
  cuda_check(cudaFree(d_E_numerical_loss));
  cuda_check(cudaFree(d_error_flags));
  cuda_check(cudaFree(d_E_escape));
  cuda_check(cudaFree(d_rad_E_tally));
  cuda_check(cudaFree(d_sigma_s_eff));
  cuda_check(cudaFree(d_sigma_a_eff));
  cuda_check(cudaFree(d_f_fleck));
  cuda_check(cudaFree(d_sigma_a));

  CAPTURE(dep_cell0);
  CAPTURE(dep_cell1);
  CAPTURE(dep_beyond_cell0);
  CAPTURE(nonzero_cells_beyond0);
  CAPTURE(moved_particles);
  CAPTURE(host_flags.invalid_cell);
  CAPTURE(host_flags.infinite_loop);

  REQUIRE(ok_dep_cell0);
  REQUIRE(ok_dep_cell1);
  REQUIRE(ok_any_beyond0);
  REQUIRE(ok_nonzero_beyond0);
  REQUIRE(ok_moved_particles);
  CHECK(ok_invalid_cell_flag);
  CHECK(ok_infinite_loop_flag);
}
