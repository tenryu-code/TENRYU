#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/error.hpp"
#include "radiation/composite_sort.cuh"
#include "radiation/particle_pool.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_composite_sort");
}

}  // namespace

TEST_CASE("Composite sort compacts and mode-partitions alive particles", "[radiation][sort]") {
  tenryu::radiation::PhotonPool pool;
  pool.allocate(6);
  pool.n_alive = 6;

  const std::vector<std::int32_t> cell = {3, 1, -1, 0, 2, 1};
  const std::vector<std::uint8_t> alive = {
      tenryu::radiation::kAlive,
      tenryu::radiation::kAlive,
      tenryu::radiation::kDead,
      tenryu::radiation::kAlive,
      tenryu::radiation::kAlive,
      tenryu::radiation::kAlive,
  };
  const std::vector<std::uint8_t> mode = {
      tenryu::radiation::kModeDDMC,
      tenryu::radiation::kModeIMC,
      tenryu::radiation::kModeIMC,
      tenryu::radiation::kModeIMC,
      tenryu::radiation::kModeDDMC,
      tenryu::radiation::kModeIMC,
  };
  const std::vector<std::uint16_t> group = {0, 0, 0, 0, 0, 0};
  const std::vector<double> energy = {3.0, 1.0, 9.0, 0.0, 2.0, 1.5};

  cuda_check(cudaMemcpy(pool.cell_id,
                        cell.data(),
                        sizeof(std::int32_t) * cell.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.alive,
                        alive.data(),
                        sizeof(std::uint8_t) * alive.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.mode,
                        mode.data(),
                        sizeof(std::uint8_t) * mode.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.group_id,
                        group.data(),
                        sizeof(std::uint16_t) * group.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.energy,
                        energy.data(),
                        sizeof(double) * energy.size(),
                        cudaMemcpyHostToDevice));

  const auto result = tenryu::radiation::composite_sort_and_partition(pool, 6, 10, 1);

  REQUIRE(result.n_alive == 5);
  REQUIRE(result.n_imc == 3);
  REQUIRE(result.n_ddmc == 2);

  std::vector<std::int32_t> sorted_cell(6, 0);
  std::vector<std::uint8_t> sorted_mode(6, 0);
  std::vector<std::uint8_t> sorted_alive(6, 0);
  std::vector<double> sorted_energy(6, 0.0);
  cuda_check(cudaMemcpy(sorted_cell.data(),
                        pool.cell_id,
                        sizeof(std::int32_t) * sorted_cell.size(),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(sorted_mode.data(),
                        pool.mode,
                        sizeof(std::uint8_t) * sorted_mode.size(),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(sorted_alive.data(),
                        pool.alive,
                        sizeof(std::uint8_t) * sorted_alive.size(),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(sorted_energy.data(),
                        pool.energy,
                        sizeof(double) * sorted_energy.size(),
                        cudaMemcpyDeviceToHost));

  // Alive particles are sorted by (mode, cell, group): IMC first, then DDMC.
  // IMC sorted by cell/group: {0,IMC,0.0}, {1,IMC,1.0}, {1,IMC,1.5}
  // DDMC sorted by cell/group: {2,DDMC,2.0}, {3,DDMC,3.0}
  const std::vector<std::int32_t> expected_cell = {0, 1, 1, 2, 3};
  const std::vector<std::uint8_t> expected_mode = {
      tenryu::radiation::kModeIMC,
      tenryu::radiation::kModeIMC,
      tenryu::radiation::kModeIMC,
      tenryu::radiation::kModeDDMC,
      tenryu::radiation::kModeDDMC,
  };
  const std::vector<double> expected_energy = {0.0, 1.0, 1.5, 2.0, 3.0};

  for (std::size_t i = 0; i < expected_cell.size(); ++i) {
    REQUIRE(sorted_alive[i] == tenryu::radiation::kAlive);
    REQUIRE(sorted_cell[i] == expected_cell[i]);
    REQUIRE(sorted_mode[i] == expected_mode[i]);
    REQUIRE(sorted_energy[i] == expected_energy[i]);
  }
  REQUIRE(sorted_alive[5] == tenryu::radiation::kDead);
}

TEST_CASE("Composite sort accumulates dropped energy on device", "[radiation][sort]") {
  tenryu::radiation::PhotonPool pool;
  pool.allocate(4);
  pool.n_alive = 4;

  const std::vector<std::int32_t> cell = {0, 1, -1, 2};
  const std::vector<std::uint8_t> alive = {
      tenryu::radiation::kAlive,
      tenryu::radiation::kDead,
      tenryu::radiation::kAlive,
      tenryu::radiation::kAlive,
  };
  const std::vector<std::uint8_t> mode = {
      tenryu::radiation::kModeIMC,
      tenryu::radiation::kModeIMC,
      tenryu::radiation::kModeDDMC,
      tenryu::radiation::kModeDDMC,
  };
  const std::vector<std::uint16_t> group = {0, 0, 0, 0};
  const std::vector<double> energy = {1.0, 2.0, 3.0, 4.0};

  cuda_check(cudaMemcpy(pool.cell_id,
                        cell.data(),
                        sizeof(std::int32_t) * cell.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.alive,
                        alive.data(),
                        sizeof(std::uint8_t) * alive.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.mode,
                        mode.data(),
                        sizeof(std::uint8_t) * mode.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.group_id,
                        group.data(),
                        sizeof(std::uint16_t) * group.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.energy,
                        energy.data(),
                        sizeof(double) * energy.size(),
                        cudaMemcpyHostToDevice));

  double* d_numerical_loss = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_numerical_loss), sizeof(double)));
  const double initial_loss = 1.5;
  cuda_check(cudaMemcpy(d_numerical_loss,
                        &initial_loss,
                        sizeof(double),
                        cudaMemcpyHostToDevice));

  const auto result =
      tenryu::radiation::composite_sort_and_partition(pool, 4, 2, 1, d_numerical_loss);

  REQUIRE(result.n_alive == 1);
  REQUIRE(result.n_imc == 1);
  REQUIRE(result.n_ddmc == 0);

  double numerical_loss = 0.0;
  cuda_check(cudaMemcpy(&numerical_loss,
                        d_numerical_loss,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  REQUIRE(numerical_loss == Catch::Approx(10.5));

  std::vector<std::uint8_t> sorted_alive(4, 0);
  cuda_check(cudaMemcpy(sorted_alive.data(),
                        pool.alive,
                        sizeof(std::uint8_t) * sorted_alive.size(),
                        cudaMemcpyDeviceToHost));
  REQUIRE(sorted_alive[0] == tenryu::radiation::kAlive);
  REQUIRE(sorted_alive[1] == tenryu::radiation::kDead);
  REQUIRE(sorted_alive[2] == tenryu::radiation::kDead);
  REQUIRE(sorted_alive[3] == tenryu::radiation::kDead);

  cuda_check(cudaFree(d_numerical_loss));
}
