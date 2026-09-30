#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#include "core/error.hpp"
#include "radiation/particle_pool.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_particle_pool");
}

}  // namespace

TEST_CASE("PhotonPool allocates all SoA fields", "[radiation][pool]") {
  tenryu::radiation::PhotonPool pool;
  pool.allocate(16);

  REQUIRE(pool.capacity == 16);
  REQUIRE(pool.pos_r != nullptr);
  REQUIRE(pool.pos_z != nullptr);
  REQUIRE(pool.dir_r != nullptr);
  REQUIRE(pool.dir_z != nullptr);
  REQUIRE(pool.dir_phi != nullptr);
  REQUIRE(pool.energy != nullptr);
  REQUIRE(pool.weight != nullptr);
  REQUIRE(pool.time_remain != nullptr);
  REQUIRE(pool.birth_energy != nullptr);
  REQUIRE(pool.sign != nullptr);
  REQUIRE(pool.global_id != nullptr);
  REQUIRE(pool.rng_counter != nullptr);
  REQUIRE(pool.cell_id != nullptr);
  REQUIRE(pool.group_id != nullptr);
  REQUIRE(pool.mode != nullptr);
  REQUIRE(pool.alive != nullptr);

  pool.n_alive = 8;
  std::vector<double> energy(8, 1.0);
  std::vector<std::int8_t> sign(8, -1);
  cuda_check(cudaMemcpy(pool.energy,
                        energy.data(),
                        sizeof(double) * energy.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.sign,
                        sign.data(),
                        sizeof(std::int8_t) * sign.size(),
                        cudaMemcpyHostToDevice));

  pool.reserve(40, 128);
  REQUIRE(pool.capacity >= 40);
  REQUIRE(pool.n_alive == 8);

  std::vector<double> out(8, 0.0);
  cuda_check(cudaMemcpy(out.data(),
                        pool.energy,
                        sizeof(double) * out.size(),
                        cudaMemcpyDeviceToHost));
  for (double e : out) {
    REQUIRE(e == 1.0);
  }
  std::vector<std::int8_t> sign_out(8, 0);
  cuda_check(cudaMemcpy(sign_out.data(),
                        pool.sign,
                        sizeof(std::int8_t) * sign_out.size(),
                        cudaMemcpyDeviceToHost));
  for (std::int8_t s : sign_out) {
    REQUIRE(s == -1);
  }
}
