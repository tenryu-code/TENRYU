#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#include "core/error.hpp"
#include "radiation/particle_pool.cuh"
#include "radiation/particle_reid.hpp"

namespace {

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

template <typename T>
void copy_to_device(T* dst, const std::vector<T>& src, const char* message) {
  const cudaError_t err =
      cudaMemcpy(dst, src.data(), sizeof(T) * src.size(), cudaMemcpyHostToDevice);
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
std::vector<T> copy_to_host(const T* src, const int n, const char* message) {
  std::vector<T> out(static_cast<std::size_t>(n));
  const cudaError_t err =
      cudaMemcpy(out.data(), src, sizeof(T) * out.size(), cudaMemcpyDeviceToHost);
  TENRYU_ASSERT(err == cudaSuccess, message);
  return out;
}

void upload_minimal_pool(tenryu::radiation::PhotonPool& pool,
                         const std::vector<double>& pos_r,
                         const std::vector<std::int32_t>& cell_id,
                         const std::vector<std::uint8_t>& mode,
                         const std::vector<std::uint8_t>& alive) {
  const int n = static_cast<int>(pos_r.size());
  REQUIRE(static_cast<int>(cell_id.size()) == n);
  REQUIRE(static_cast<int>(mode.size()) == n);
  REQUIRE(static_cast<int>(alive.size()) == n);
  copy_to_device(pool.pos_r, pos_r, "particle_reid test copy pos_r failed");
  copy_to_device(pool.cell_id, cell_id, "particle_reid test copy cell_id failed");
  copy_to_device(pool.mode, mode, "particle_reid test copy mode failed");
  copy_to_device(pool.alive, alive, "particle_reid test copy alive failed");
  pool.n_alive = n;
  pool.n_census = n;
}

}  // namespace

TEST_CASE("1D particle re-ID follows rezone mesh and skips DDMC/NaN",
          "[radiation][particle_reid]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  namespace rad = tenryu::radiation;
  constexpr int n_particles = 8;
  rad::PhotonPool pool;
  pool.allocate(n_particles);

  double* d_node_r = nullptr;
  const cudaError_t malloc_err =
      cudaMalloc(reinterpret_cast<void**>(&d_node_r), sizeof(double) * 4U);
  REQUIRE(malloc_err == cudaSuccess);

  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<std::uint8_t> mode = {
      rad::kModeIMC, rad::kModeIMC, rad::kModeIMC, rad::kModeDDMC,
      rad::kModeIMC, rad::kModeRW,  rad::kModeIMC, rad::kModeIMC};
  const std::vector<std::uint8_t> alive = {
      rad::kAlive, rad::kAlive, rad::kAlive, rad::kAlive,
      rad::kAlive, rad::kAlive, rad::kDead,  rad::kAlive};

  SECTION("contracted mesh") {
    const std::vector<double> node_r = {0.0, 0.8, 1.5, 3.0};
    copy_to_device(d_node_r, node_r, "particle_reid test copy contracted nodes failed");
    upload_minimal_pool(pool,
                        {0.2, 1.0, 2.7, 1.0, nan, 0.9, 1.2, -0.1},
                        {0, 0, 0, 0, 2, 2, 0, 2},
                        mode,
                        alive);

    const auto stats =
        rad::reidentify_finite_position_particles_1d_cuda(pool, d_node_r, 3);
    const auto cells =
        copy_to_host(pool.cell_id, n_particles, "particle_reid test copy cells failed");

    REQUIRE(cells == std::vector<std::int32_t>{0, 1, 2, 0, 2, 1, 0, 0});
    REQUIRE(stats.checked == 5);
    REQUIRE(stats.kept == 1);
    REQUIRE(stats.updated == 4);
    REQUIRE(stats.skipped_ddmc == 1);
    REQUIRE(stats.skipped_nan == 1);
    REQUIRE(stats.skipped_dead == 1);
    REQUIRE(stats.binary_search == 1);
    REQUIRE(stats.clamped == 1);
  }

  SECTION("expanded mesh") {
    const std::vector<double> node_r = {0.0, 1.5, 3.0, 4.5};
    copy_to_device(d_node_r, node_r, "particle_reid test copy expanded nodes failed");
    upload_minimal_pool(pool,
                        {1.2, 2.2, 4.2, 2.2, nan, 0.2, 3.2, 4.9},
                        {1, 0, 0, 0, 2, 0, 1, 0},
                        mode,
                        alive);

    const auto stats =
        rad::reidentify_finite_position_particles_1d_cuda(pool, d_node_r, 3);
    const auto cells =
        copy_to_host(pool.cell_id, n_particles, "particle_reid test copy cells failed");

    REQUIRE(cells == std::vector<std::int32_t>{0, 1, 2, 0, 2, 0, 1, 2});
    REQUIRE(stats.checked == 5);
    REQUIRE(stats.updated == 4);
    REQUIRE(stats.skipped_ddmc == 1);
    REQUIRE(stats.skipped_nan == 1);
    REQUIRE(stats.skipped_dead == 1);
    REQUIRE(stats.clamped == 1);
  }

  const cudaError_t free_err = cudaFree(d_node_r);
  REQUIRE(free_err == cudaSuccess);
}
