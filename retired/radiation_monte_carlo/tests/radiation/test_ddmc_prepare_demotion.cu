#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/error.hpp"
#include "radiation/ddmc.hpp"
#include "radiation/particle_pool.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_ddmc_prepare_demotion");
}

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return (err == cudaSuccess && device_count > 0);
}

}  // namespace

TEST_CASE("DDMC->IMC demotion resamples phase space at step boundary",
          "[radiation][ddmc][prepare]") {
  if (!has_cuda_device()) {
    SUCCEED("No CUDA device available");
    return;
  }

  const double nan = std::numeric_limits<double>::quiet_NaN();
  constexpr double dt = 2.5e-12;
  constexpr std::uint64_t step_number = 17;
  constexpr std::uint64_t global_id_init = 1234ULL;
  constexpr std::uint32_t rng_counter_init = 11U;

  auto make_cfg = []() {
    tenryu::core::Config cfg{};
    cfg.main.seed = 424242ULL;
    cfg.radiation.ddmc.enabled = true;
    cfg.radiation.ddmc.tau_ddmc = 4.0;
    cfg.radiation.ddmc.omega_ddmc = 0.9;
    cfg.radiation.ddmc.m_matrix_check = false;
    cfg.numerics.safety.opacity_floor = 1.0e-30;
    return cfg;
  };

  auto init_single_particle = [&](tenryu::radiation::PhotonPool& pool) {
    constexpr int n_particles = 1;
    pool.allocate(n_particles);
    pool.n_alive = n_particles;

    const double pos_r0 = nan;
    const double pos_z0 = nan;
    const double dir_r0 = nan;
    const double dir_z0 = nan;
    const double dir_phi0 = nan;
    const double energy0 = 3.25;
    const double time_remain0 = 0.0;
    const std::int32_t cell0 = 0;
    const std::uint16_t group0 = 0;
    const std::uint8_t mode0 = tenryu::radiation::kModeDDMC;
    const std::uint8_t alive0 = tenryu::radiation::kAlive;

    cuda_check(cudaMemcpy(pool.pos_r, &pos_r0, sizeof(pos_r0), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.pos_z, &pos_z0, sizeof(pos_z0), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.dir_r, &dir_r0, sizeof(dir_r0), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.dir_z, &dir_z0, sizeof(dir_z0), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.dir_phi, &dir_phi0, sizeof(dir_phi0), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.energy, &energy0, sizeof(energy0), cudaMemcpyHostToDevice));
    cuda_check(
        cudaMemcpy(pool.time_remain, &time_remain0, sizeof(time_remain0), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.global_id,
                          &global_id_init,
                          sizeof(global_id_init),
                          cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.rng_counter,
                          &rng_counter_init,
                          sizeof(rng_counter_init),
                          cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.cell_id, &cell0, sizeof(cell0), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.group_id, &group0, sizeof(group0), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.mode, &mode0, sizeof(mode0), cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(pool.alive, &alive0, sizeof(alive0), cudaMemcpyHostToDevice));
  };

  SECTION("1D demotion repopulates finite state and resets time_remain") {
    tenryu::core::Config cfg = make_cfg();

    constexpr int mesh_dim = 1;
    constexpr int n_cells = 1;
    constexpr int n_groups = 1;
    constexpr int nr = 0;
    constexpr int nz = 0;

    const std::vector<double> node_r = {1.0, 2.0};
    const std::vector<double> node_z;
    const std::vector<double> cell_vol = {1.0};
    const std::vector<double> rho = {1.0};
    const std::vector<double> Te = {1.0};
    const std::vector<double> sigma_R = {1.0e-3};
    const std::vector<double> sigma_a = {1.0e-3};
    const std::vector<double> sigma_a_eff = {1.0e-3};
    const std::vector<double> fleck_f = {0.1};

    tenryu::radiation::PhotonPool pool;
    init_single_particle(pool);

    const auto prep = tenryu::radiation::prepare_ddmc_step(
        pool,
        cfg,
        step_number,
        dt,
        mesh_dim,
        n_cells,
        n_groups,
        node_r,
        node_z,
        nr,
        nz,
        cell_vol,
        rho,
        Te,
        sigma_R,
        sigma_a,
        sigma_a_eff,
        fleck_f,
        nullptr,  // CellRadiationCoeffs (not needed for this test)
        tenryu::radiation::DDMCBoundaryType::Vacuum,
        tenryu::radiation::DDMCBoundaryType::Vacuum,
        tenryu::radiation::DDMCBoundaryType::Vacuum,
        tenryu::radiation::DDMCBoundaryType::Vacuum);
    REQUIRE(prep.active);

    double pos_r = nan;
    double pos_z = nan;
    double dir_r = nan;
    double dir_z = nan;
    double dir_phi = nan;
    double time_remain = -1.0;
    double energy = -1.0;
    std::uint32_t rng_counter = 0U;
    std::uint8_t mode = tenryu::radiation::kModeDDMC;
    std::uint16_t group_id = 99U;
    std::uint8_t alive = tenryu::radiation::kDead;
    cuda_check(cudaMemcpy(&pos_r, pool.pos_r, sizeof(pos_r), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&pos_z, pool.pos_z, sizeof(pos_z), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&dir_r, pool.dir_r, sizeof(dir_r), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&dir_z, pool.dir_z, sizeof(dir_z), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&dir_phi, pool.dir_phi, sizeof(dir_phi), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&time_remain, pool.time_remain, sizeof(time_remain), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&energy, pool.energy, sizeof(energy), cudaMemcpyDeviceToHost));
    cuda_check(
        cudaMemcpy(&rng_counter, pool.rng_counter, sizeof(rng_counter), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&mode, pool.mode, sizeof(mode), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&group_id, pool.group_id, sizeof(group_id), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&alive, pool.alive, sizeof(alive), cudaMemcpyDeviceToHost));

    REQUIRE(alive == tenryu::radiation::kAlive);
    REQUIRE(mode == tenryu::radiation::kModeIMC);
    REQUIRE(group_id == 0U);
    REQUIRE(energy == Catch::Approx(3.25).epsilon(1.0e-14));
    REQUIRE(time_remain == Catch::Approx(dt).epsilon(1.0e-14));
    REQUIRE(rng_counter > rng_counter_init);
    REQUIRE(std::isfinite(pos_r));
    REQUIRE(std::isfinite(pos_z));
    REQUIRE(std::isfinite(dir_r));
    REQUIRE(std::isfinite(dir_z));
    REQUIRE(std::isfinite(dir_phi));
    REQUIRE(pos_r >= node_r[0]);
    REQUIRE(pos_r <= node_r[1]);
    REQUIRE(pos_z == Catch::Approx(0.0).margin(1.0e-14));
  }

  SECTION("2D demotion repopulates finite state and resets time_remain") {
    tenryu::core::Config cfg = make_cfg();
    cfg.radiation.ddmc.leak_stencil = "4";

    constexpr int mesh_dim = 2;
    constexpr int n_cells = 1;
    constexpr int n_groups = 1;
    constexpr int nr = 1;
    constexpr int nz = 1;

    const std::vector<double> node_r = {1.0, 1.0, 2.0, 2.0};
    const std::vector<double> node_z = {0.0, 1.0, 0.0, 1.0};
    const std::vector<double> cell_vol = {1.0};
    const std::vector<double> rho = {1.0};
    const std::vector<double> Te = {1.0};
    const std::vector<double> sigma_R = {1.0e-3};
    const std::vector<double> sigma_a = {1.0e-3};
    const std::vector<double> sigma_a_eff = {1.0e-3};
    const std::vector<double> fleck_f = {0.1};

    tenryu::radiation::PhotonPool pool;
    init_single_particle(pool);

    const auto prep = tenryu::radiation::prepare_ddmc_step(
        pool,
        cfg,
        step_number,
        dt,
        mesh_dim,
        n_cells,
        n_groups,
        node_r,
        node_z,
        nr,
        nz,
        cell_vol,
        rho,
        Te,
        sigma_R,
        sigma_a,
        sigma_a_eff,
        fleck_f,
        nullptr,  // CellRadiationCoeffs
        tenryu::radiation::DDMCBoundaryType::Reflective,
        tenryu::radiation::DDMCBoundaryType::Vacuum,
        tenryu::radiation::DDMCBoundaryType::Vacuum,
        tenryu::radiation::DDMCBoundaryType::Vacuum);
    REQUIRE(prep.active);

    double pos_r = nan;
    double pos_z = nan;
    double dir_r = nan;
    double dir_z = nan;
    double dir_phi = nan;
    double time_remain = -1.0;
    double energy = -1.0;
    std::uint32_t rng_counter = 0U;
    std::uint8_t mode = tenryu::radiation::kModeDDMC;
    std::uint16_t group_id = 99U;
    std::uint8_t alive = tenryu::radiation::kDead;
    cuda_check(cudaMemcpy(&pos_r, pool.pos_r, sizeof(pos_r), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&pos_z, pool.pos_z, sizeof(pos_z), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&dir_r, pool.dir_r, sizeof(dir_r), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&dir_z, pool.dir_z, sizeof(dir_z), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&dir_phi, pool.dir_phi, sizeof(dir_phi), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&time_remain, pool.time_remain, sizeof(time_remain), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&energy, pool.energy, sizeof(energy), cudaMemcpyDeviceToHost));
    cuda_check(
        cudaMemcpy(&rng_counter, pool.rng_counter, sizeof(rng_counter), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&mode, pool.mode, sizeof(mode), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&group_id, pool.group_id, sizeof(group_id), cudaMemcpyDeviceToHost));
    cuda_check(cudaMemcpy(&alive, pool.alive, sizeof(alive), cudaMemcpyDeviceToHost));

    REQUIRE(alive == tenryu::radiation::kAlive);
    REQUIRE(mode == tenryu::radiation::kModeIMC);
    REQUIRE(group_id == 0U);
    REQUIRE(energy == Catch::Approx(3.25).epsilon(1.0e-14));
    REQUIRE(time_remain == Catch::Approx(dt).epsilon(1.0e-14));
    REQUIRE(rng_counter > rng_counter_init);
    REQUIRE(std::isfinite(pos_r));
    REQUIRE(std::isfinite(pos_z));
    REQUIRE(std::isfinite(dir_r));
    REQUIRE(std::isfinite(dir_z));
    REQUIRE(std::isfinite(dir_phi));
    REQUIRE(pos_r >= 1.0);
    REQUIRE(pos_r <= 2.0);
    REQUIRE(pos_z >= 0.0);
    REQUIRE(pos_z <= 1.0);
  }
}
