#include <algorithm>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "materials/ionmix_reader.hpp"
#include "mesh/mesh.hpp"
#include "radiation/groups.cuh"
#include "radiation/nlte_coeffs.hpp"
#include "radiation/planck_table.cuh"
#include "radiation/source.cuh"

namespace {

void cuda_check(const cudaError_t err, const char* msg) {
  TENRYU_ASSERT(err == cudaSuccess, msg);
}

tenryu::core::Config make_cfg(const tenryu::materials::IonmixOpacityData& table) {
  tenryu::core::Config cfg;
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.main.seed = 12345;

  cfg.mesh.nr = 1;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 10.0;
  cfg.mesh.r_max = 11.0;
  cfg.mesh.grid_type_r = "uniform";

  tenryu::core::Config::MaterialsConfig::MatDef mat;
  mat.name = "nlte_src";
  mat.A = 6.5;
  mat.Z = 3.5;
  mat.opacity_model = "table_nlte";
  mat.opacity_file = "tests/data/ionmix_nlte_simple.cn4";
  mat.ideal_gas_gamma = 5.0 / 3.0;
  cfg.materials.materials = {mat};

  cfg.radiation.enabled = true;
  cfg.radiation.mode = tenryu::core::RadiationMode::ImcDdmc;
  cfg.radiation.groups = 4;
  cfg.radiation.group_bounds_eV = table.bounds_eV;
  cfg.radiation.compute_T_range_eV = {0.1, 100.0};
  cfg.radiation.boundary.inner_r = "reflect";
  cfg.radiation.boundary.outer_r = "reflect";
  cfg.radiation.imc.alpha = 1.0;
  cfg.radiation.imc.f_max = 1.0;
  cfg.radiation.imc.particles_per_cell_group = 2000;
  cfg.radiation.imc.cutoff_fraction = 0.0;
  cfg.radiation.imc.weight_cutoff = 1.0e-12;
  cfg.radiation.imc.roulette_survival = 1.0;

  cfg.numerics.floors.Te = 1.0e-3;
  cfg.numerics.floors.rho = 1.0e-10;
  return cfg;
}

}  // namespace

TEST_CASE("NLTE source energy matches f * eta_tot * V * dt", "[radiation][nlte][source]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping NLTE source test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  auto cfg = make_cfg(table);

  auto state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;
  std::vector<double> vol_h(state.vol.size(), 0.0);
  state.vol.copy_to_host(vol_h.data());

  const std::vector<double> rho_h = {1.0};
  const std::vector<double> Te_h = {10.0};
  const std::vector<double> zbar_h = {3.5};
  const std::vector<double> mass_h = {vol_h[0] * rho_h[0]};
  state.rho.copy_from_host(rho_h.data());
  state.Te.copy_from_host(Te_h.data());
  state.Ti.copy_from_host(Te_h.data());
  state.zbar.copy_from_host(zbar_h.data());
  state.mass.copy_from_host(mass_h.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups, 200, 0.1, 100.0);

  const double dt = 1.0e-11;
  const auto coeffs =
      tenryu::radiation::compute_nlte_coefficients(state, cfg, table, planck, 1, 4, dt);

  double* d_sigma_a_eff = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff),
                        sizeof(double) * coeffs.sigma_a_eff.size()),
             "test_imc_source_nlte cudaMalloc d_sigma_a_eff failed");
  cuda_check(cudaMemcpy(d_sigma_a_eff,
                        coeffs.sigma_a_eff.data(),
                        sizeof(double) * coeffs.sigma_a_eff.size(),
                        cudaMemcpyHostToDevice),
             "test_imc_source_nlte copy sigma_a_eff failed");

  tenryu::radiation::PhotonPool pool;
  const auto stats = tenryu::radiation::IMCSource::emit_thermal(state,
                                                                cfg,
                                                                planck,
                                                                d_sigma_a_eff,
                                                                pool,
                                                                200000,
                                                                dt,
                                                                0,
                                                                cfg.main.seed,
                                                                0,
                                                                &coeffs.eta,
                                                                &coeffs.f);

  std::vector<double> rad_emit_host(state.rad_emit.size(), 0.0);
  state.rad_emit.copy_to_host(rad_emit_host.data());
  const double emitted_from_field =
      std::accumulate(rad_emit_host.begin(), rad_emit_host.end(), 0.0);

  const double expected =
      coeffs.f[0] * coeffs.eta_tot[0] * std::max(vol_h[0], 0.0) * dt;

  REQUIRE(stats.n_thermal > 0);
  CHECK(emitted_from_field == Catch::Approx(expected).epsilon(0.01));
  CHECK((stats.E_thermal + stats.E_thermal_lost) ==
        Catch::Approx(expected).epsilon(0.02));

  cuda_check(cudaFree(d_sigma_a_eff), "test_imc_source_nlte cudaFree d_sigma_a_eff failed");
}

TEST_CASE("NLTE source cold cell produces near-zero emission", "[radiation][nlte][source]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping NLTE source test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  auto cfg = make_cfg(table);

  auto state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;

  const std::vector<double> rho_h = {1.0};
  const std::vector<double> Te_cold = {1.0e-3};
  const std::vector<double> zbar_h = {3.5};
  std::vector<double> vol_h(state.vol.size(), 0.0);
  state.vol.copy_to_host(vol_h.data());
  const std::vector<double> mass_h = {vol_h[0] * rho_h[0]};
  state.rho.copy_from_host(rho_h.data());
  state.Te.copy_from_host(Te_cold.data());
  state.Ti.copy_from_host(Te_cold.data());
  state.zbar.copy_from_host(zbar_h.data());
  state.mass.copy_from_host(mass_h.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups, 200, 0.1, 100.0);
  const double dt = 1.0e-11;
  const auto coeffs =
      tenryu::radiation::compute_nlte_coefficients(state, cfg, table, planck, 1, 4, dt);

  double* d_sigma_a_eff = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff),
                        sizeof(double) * coeffs.sigma_a_eff.size()),
             "test_imc_source_nlte cold-cell cudaMalloc d_sigma_a_eff failed");
  cuda_check(cudaMemcpy(d_sigma_a_eff,
                        coeffs.sigma_a_eff.data(),
                        sizeof(double) * coeffs.sigma_a_eff.size(),
                        cudaMemcpyHostToDevice),
             "test_imc_source_nlte cold-cell copy sigma_a_eff failed");

  tenryu::radiation::PhotonPool pool;
  const auto stats = tenryu::radiation::IMCSource::emit_thermal(state,
                                                                cfg,
                                                                planck,
                                                                d_sigma_a_eff,
                                                                pool,
                                                                200000,
                                                                dt,
                                                                0,
                                                                cfg.main.seed,
                                                                0,
                                                                &coeffs.eta,
                                                                &coeffs.f);

  CHECK(stats.E_thermal < 1.0);
  cuda_check(cudaFree(d_sigma_a_eff),
             "test_imc_source_nlte cold-cell cudaFree d_sigma_a_eff failed");
}

TEST_CASE("NLTE source: per-group emission matches eta_g fraction",
          "[radiation][nlte][source]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping NLTE source test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  auto cfg = make_cfg(table);

  auto state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;
  std::vector<double> vol_h(state.vol.size(), 0.0);
  state.vol.copy_to_host(vol_h.data());

  const std::vector<double> rho_h = {1.0};
  const std::vector<double> Te_h = {10.0};
  const std::vector<double> zbar_h = {3.5};
  const std::vector<double> mass_h = {vol_h[0] * rho_h[0]};
  state.rho.copy_from_host(rho_h.data());
  state.Te.copy_from_host(Te_h.data());
  state.Ti.copy_from_host(Te_h.data());
  state.zbar.copy_from_host(zbar_h.data());
  state.mass.copy_from_host(mass_h.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups, 200, 0.1, 100.0);

  const double dt = 1.0e-11;
  const auto coeffs =
      tenryu::radiation::compute_nlte_coefficients(state, cfg, table, planck, 1, 4, dt);

  double* d_sigma_a_eff = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff),
                        sizeof(double) * coeffs.sigma_a_eff.size()),
             "test_imc_source_nlte cudaMalloc d_sigma_a_eff failed");
  cuda_check(cudaMemcpy(d_sigma_a_eff,
                        coeffs.sigma_a_eff.data(),
                        sizeof(double) * coeffs.sigma_a_eff.size(),
                        cudaMemcpyHostToDevice),
             "test_imc_source_nlte copy sigma_a_eff failed");

  tenryu::radiation::PhotonPool pool;
  const auto stats = tenryu::radiation::IMCSource::emit_thermal(state,
                                                                cfg,
                                                                planck,
                                                                d_sigma_a_eff,
                                                                pool,
                                                                200000,
                                                                dt,
                                                                0,
                                                                cfg.main.seed,
                                                                0,
                                                                &coeffs.eta,
                                                                &coeffs.f);

  std::vector<double> rad_emit_host(state.rad_emit.size(), 0.0);
  state.rad_emit.copy_to_host(rad_emit_host.data());
  const double emitted_total =
      std::accumulate(rad_emit_host.begin(), rad_emit_host.end(), 0.0);

  REQUIRE(stats.n_thermal > 0);
  REQUIRE(coeffs.eta_tot[0] > 0.0);
  REQUIRE(emitted_total > 0.0);

  for (int g = 0; g < cfg.radiation.groups; ++g) {
    const std::size_t idx = static_cast<std::size_t>(g);
    const double emitted_fraction = rad_emit_host[idx] / emitted_total;
    const double expected_fraction = coeffs.eta[idx] / coeffs.eta_tot[0];
    CHECK(emitted_fraction == Catch::Approx(expected_fraction).epsilon(0.10));
  }

  cuda_check(cudaFree(d_sigma_a_eff), "test_imc_source_nlte cudaFree d_sigma_a_eff failed");
}

TEST_CASE("NLTE source: packet energy normalization", "[radiation][nlte][source]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping NLTE source test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  auto cfg = make_cfg(table);

  auto state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;
  std::vector<double> vol_h(state.vol.size(), 0.0);
  state.vol.copy_to_host(vol_h.data());

  const std::vector<double> rho_h = {1.0};
  const std::vector<double> Te_h = {10.0};
  const std::vector<double> zbar_h = {3.5};
  const std::vector<double> mass_h = {vol_h[0] * rho_h[0]};
  state.rho.copy_from_host(rho_h.data());
  state.Te.copy_from_host(Te_h.data());
  state.Ti.copy_from_host(Te_h.data());
  state.zbar.copy_from_host(zbar_h.data());
  state.mass.copy_from_host(mass_h.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups, 200, 0.1, 100.0);

  const double dt = 1.0e-11;
  const auto coeffs =
      tenryu::radiation::compute_nlte_coefficients(state, cfg, table, planck, 1, 4, dt);

  double* d_sigma_a_eff = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff),
                        sizeof(double) * coeffs.sigma_a_eff.size()),
             "test_imc_source_nlte cudaMalloc d_sigma_a_eff failed");
  cuda_check(cudaMemcpy(d_sigma_a_eff,
                        coeffs.sigma_a_eff.data(),
                        sizeof(double) * coeffs.sigma_a_eff.size(),
                        cudaMemcpyHostToDevice),
             "test_imc_source_nlte copy sigma_a_eff failed");

  tenryu::radiation::PhotonPool pool;
  const auto stats = tenryu::radiation::IMCSource::emit_thermal(state,
                                                                cfg,
                                                                planck,
                                                                d_sigma_a_eff,
                                                                pool,
                                                                200000,
                                                                dt,
                                                                0,
                                                                cfg.main.seed,
                                                                0,
                                                                &coeffs.eta,
                                                                &coeffs.f);

  REQUIRE(stats.n_thermal > 0);
  REQUIRE(pool.n_alive == stats.n_thermal);

  std::vector<double> packet_energy(static_cast<std::size_t>(pool.n_alive), 0.0);
  cuda_check(cudaMemcpy(packet_energy.data(),
                        pool.energy,
                        sizeof(double) * packet_energy.size(),
                        cudaMemcpyDeviceToHost),
             "test_imc_source_nlte copy packet energies failed");

  double expected_source_sum = 0.0;
  for (int g = 0; g < cfg.radiation.groups; ++g) {
    const std::size_t idx = static_cast<std::size_t>(g);
    expected_source_sum +=
        coeffs.f[0] * coeffs.eta[idx] * std::max(vol_h[0], 0.0) * dt;
  }

  const double packet_energy_sum =
      std::accumulate(packet_energy.begin(), packet_energy.end(), 0.0);
  CHECK((packet_energy_sum + stats.E_thermal_lost) ==
        Catch::Approx(expected_source_sum).epsilon(0.05));
  CHECK(packet_energy_sum == Catch::Approx(stats.E_thermal).epsilon(1.0e-12));

  for (const double e_packet : packet_energy) {
    CHECK(e_packet >= 0.0);
    CHECK(e_packet <= expected_source_sum);
  }

  cuda_check(cudaFree(d_sigma_a_eff), "test_imc_source_nlte cudaFree d_sigma_a_eff failed");
}

TEST_CASE("NLTE source: LTE switching", "[radiation][nlte][source][lte]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping NLTE source test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_lte_const.cn4");
  REQUIRE(table.is_lte);

  auto cfg = make_cfg(table);
  cfg.materials.materials.front().opacity_file = "tests/data/ionmix_lte_const.cn4";

  auto state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;
  std::vector<double> vol_h(state.vol.size(), 0.0);
  state.vol.copy_to_host(vol_h.data());

  const std::vector<double> rho_h = {1.0};
  const std::vector<double> Te_h = {10.0};
  const std::vector<double> zbar_h = {3.5};
  const std::vector<double> mass_h = {vol_h[0] * rho_h[0]};
  state.rho.copy_from_host(rho_h.data());
  state.Te.copy_from_host(Te_h.data());
  state.Ti.copy_from_host(Te_h.data());
  state.zbar.copy_from_host(zbar_h.data());
  state.mass.copy_from_host(mass_h.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups, 200, 0.1, 100.0);

  const double dt = 1.0e-11;
  const auto coeffs =
      tenryu::radiation::compute_nlte_coefficients(state, cfg, table, planck, 1, 4, dt);

  double* d_sigma_a_eff = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff),
                        sizeof(double) * coeffs.sigma_a_eff.size()),
             "test_imc_source_nlte cudaMalloc d_sigma_a_eff failed");
  cuda_check(cudaMemcpy(d_sigma_a_eff,
                        coeffs.sigma_a_eff.data(),
                        sizeof(double) * coeffs.sigma_a_eff.size(),
                        cudaMemcpyHostToDevice),
             "test_imc_source_nlte copy sigma_a_eff failed");

  tenryu::radiation::PhotonPool pool_nlte;
  const auto stats_nlte = tenryu::radiation::IMCSource::emit_thermal(state,
                                                                     cfg,
                                                                     planck,
                                                                     d_sigma_a_eff,
                                                                     pool_nlte,
                                                                     200000,
                                                                     dt,
                                                                     0,
                                                                     cfg.main.seed,
                                                                     0,
                                                                     &coeffs.eta,
                                                                     &coeffs.f);

  std::vector<double> rad_emit_nlte(state.rad_emit.size(), 0.0);
  state.rad_emit.copy_to_host(rad_emit_nlte.data());
  const double emitted_nlte =
      std::accumulate(rad_emit_nlte.begin(), rad_emit_nlte.end(), 0.0);

  tenryu::radiation::PhotonPool pool_lte;
  const auto stats_lte = tenryu::radiation::IMCSource::emit_thermal(state,
                                                                    cfg,
                                                                    planck,
                                                                    d_sigma_a_eff,
                                                                    pool_lte,
                                                                    200000,
                                                                    dt,
                                                                    1,
                                                                    cfg.main.seed,
                                                                    0,
                                                                    nullptr,
                                                                    nullptr);

  std::vector<double> rad_emit_lte(state.rad_emit.size(), 0.0);
  state.rad_emit.copy_to_host(rad_emit_lte.data());
  const double emitted_lte =
      std::accumulate(rad_emit_lte.begin(), rad_emit_lte.end(), 0.0);

  REQUIRE(stats_nlte.n_thermal > 0);
  REQUIRE(stats_lte.n_thermal > 0);
  CHECK(emitted_nlte == Catch::Approx(emitted_lte).epsilon(0.05));
  CHECK((stats_nlte.E_thermal + stats_nlte.E_thermal_lost) ==
        Catch::Approx(stats_lte.E_thermal + stats_lte.E_thermal_lost).epsilon(0.05));

  cuda_check(cudaFree(d_sigma_a_eff), "test_imc_source_nlte cudaFree d_sigma_a_eff failed");
}
