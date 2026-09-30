#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <vector>

#include "core/error.hpp"
#include "radiation/census_comb.cuh"
#include "radiation/particle_pool.cuh"

namespace {

using namespace tenryu::radiation;
using CensusCombConfig = tenryu::core::Config::RadiationConfig::CensusCombConfig;

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_census_comb");
}

template <typename T>
std::vector<T> read_field(const T* ptr, const int n) {
  std::vector<T> out(static_cast<std::size_t>(n));
  if (n > 0) {
    cuda_check(cudaMemcpy(out.data(),
                          ptr,
                          sizeof(T) * static_cast<std::size_t>(n),
                          cudaMemcpyDeviceToHost));
  }
  return out;
}

double* allocate_device_loss() {
  double* d_loss = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_loss), sizeof(double)));
  const double zero = 0.0;
  cuda_check(cudaMemcpy(d_loss, &zero, sizeof(double), cudaMemcpyHostToDevice));
  return d_loss;
}

void setup_pool(PhotonPool& pool,
                const int n,
                const std::vector<double>& energies,
                const std::vector<std::int32_t>& cells,
                const std::vector<std::uint16_t>& groups,
                const std::vector<std::uint8_t>& modes) {
  TENRYU_ASSERT(n > 0, "setup_pool requires n > 0");
  TENRYU_ASSERT(static_cast<int>(energies.size()) == n, "setup_pool energies size mismatch");
  TENRYU_ASSERT(static_cast<int>(cells.size()) == n, "setup_pool cells size mismatch");
  TENRYU_ASSERT(static_cast<int>(groups.size()) == n, "setup_pool groups size mismatch");
  TENRYU_ASSERT(static_cast<int>(modes.size()) == n, "setup_pool modes size mismatch");

  pool.allocate(n);
  pool.n_alive = n;
  pool.n_census = n;

  std::vector<std::uint8_t> alive(static_cast<std::size_t>(n), kAlive);
  std::vector<double> pos_r(static_cast<std::size_t>(n), 1.0);
  std::vector<double> pos_z(static_cast<std::size_t>(n), 0.5);
  std::vector<double> dir_r(static_cast<std::size_t>(n), 0.0);
  std::vector<double> dir_z(static_cast<std::size_t>(n), 1.0);
  std::vector<double> dir_phi(static_cast<std::size_t>(n), 0.0);
  std::vector<double> weight(static_cast<std::size_t>(n), 1.0);
  std::vector<double> time_remain(static_cast<std::size_t>(n), 0.0);
  std::vector<double> birth_energy(static_cast<std::size_t>(n), 0.0);
  std::vector<std::uint64_t> global_id(static_cast<std::size_t>(n));
  std::vector<std::uint32_t> rng_counter(static_cast<std::size_t>(n), 0U);
  for (int i = 0; i < n; ++i) {
    global_id[static_cast<std::size_t>(i)] = static_cast<std::uint64_t>(i);
    birth_energy[static_cast<std::size_t>(i)] = energies[static_cast<std::size_t>(i)];
  }

  for (int i = 0; i < n; ++i) {
    if (modes[static_cast<std::size_t>(i)] == kModeDDMC) {
      pos_r[static_cast<std::size_t>(i)] = std::numeric_limits<double>::quiet_NaN();
      pos_z[static_cast<std::size_t>(i)] = std::numeric_limits<double>::quiet_NaN();
      dir_r[static_cast<std::size_t>(i)] = std::numeric_limits<double>::quiet_NaN();
      dir_z[static_cast<std::size_t>(i)] = std::numeric_limits<double>::quiet_NaN();
      dir_phi[static_cast<std::size_t>(i)] = std::numeric_limits<double>::quiet_NaN();
    }
  }

  cuda_check(cudaMemcpy(pool.energy,
                        energies.data(),
                        sizeof(double) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.cell_id,
                        cells.data(),
                        sizeof(std::int32_t) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.group_id,
                        groups.data(),
                        sizeof(std::uint16_t) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.mode,
                        modes.data(),
                        sizeof(std::uint8_t) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.alive,
                        alive.data(),
                        sizeof(std::uint8_t) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.pos_r,
                        pos_r.data(),
                        sizeof(double) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.pos_z,
                        pos_z.data(),
                        sizeof(double) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_r,
                        dir_r.data(),
                        sizeof(double) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_z,
                        dir_z.data(),
                        sizeof(double) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.dir_phi,
                        dir_phi.data(),
                        sizeof(double) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.weight,
                        weight.data(),
                        sizeof(double) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.time_remain,
                        time_remain.data(),
                        sizeof(double) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.birth_energy,
                        birth_energy.data(),
                        sizeof(double) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.global_id,
                        global_id.data(),
                        sizeof(std::uint64_t) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.rng_counter,
                        rng_counter.data(),
                        sizeof(std::uint32_t) * static_cast<std::size_t>(n),
                        cudaMemcpyHostToDevice));
}

std::vector<double> read_energy(const PhotonPool& pool, const int n) {
  std::vector<double> out(static_cast<std::size_t>(n));
  if (n > 0) {
    cuda_check(cudaMemcpy(out.data(),
                          pool.energy,
                          sizeof(double) * static_cast<std::size_t>(n),
                          cudaMemcpyDeviceToHost));
  }
  return out;
}

}  // namespace

TEST_CASE("census_comb: empty pool returns zero", "[radiation][census_comb]") {
  tenryu::radiation::PhotonPool pool;
  pool.allocate(1);
  pool.n_alive = 0;
  pool.n_census = 0;

  CensusCombConfig cfg{};
  cfg.max_particles = 10;
  cfg.target_fraction = 0.5;
  cfg.min_per_bin = 1;
  cfg.trigger_ratio = 1.0;

  double* d_E_numerical_loss = allocate_device_loss();
  const auto result = tenryu::radiation::census_comb(pool,
                                                     0,
                                                     1,
                                                     1,
                                                     cfg,
                                                     0ULL,
                                                     0ULL,
                                                     1234U,
                                                     0,
                                                     d_E_numerical_loss);

  REQUIRE(result.n_alive_out == 0);
  REQUIRE(pool.n_alive == 0);
  REQUIRE(pool.n_census == 0);
  cuda_check(cudaFree(d_E_numerical_loss));
}

TEST_CASE("census_comb: energy conservation", "[radiation][census_comb]") {
  constexpr int n = 20;
  tenryu::radiation::PhotonPool pool;
  std::vector<double> energies(static_cast<std::size_t>(n), 0.0);
  std::vector<std::int32_t> cells(static_cast<std::size_t>(n), 0);
  std::vector<std::uint16_t> groups(static_cast<std::size_t>(n), 0);
  std::vector<std::uint8_t> modes(static_cast<std::size_t>(n), kModeIMC);
  for (int i = 0; i < n; ++i) {
    energies[static_cast<std::size_t>(i)] = static_cast<double>(i + 1);
    cells[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % 4);
  }
  setup_pool(pool, n, energies, cells, groups, modes);

  CensusCombConfig cfg{};
  cfg.max_particles = 10;
  cfg.target_fraction = 0.5;
  cfg.min_per_bin = 1;
  cfg.trigger_ratio = 1.0;

  double* d_E_numerical_loss = allocate_device_loss();
  const auto result = tenryu::radiation::census_comb(pool,
                                                     n,
                                                     4,
                                                     1,
                                                     cfg,
                                                     1000ULL,
                                                     0ULL,
                                                     7U,
                                                     5,
                                                     d_E_numerical_loss);

  REQUIRE(result.E_before ==
          Catch::Approx(result.E_after + result.E_killed_bins).margin(1.0e-10));
  REQUIRE(result.n_alive_out <= 5);
  cuda_check(cudaFree(d_E_numerical_loss));
}

TEST_CASE("census_comb: particle count matches target", "[radiation][census_comb]") {
  constexpr int n = 100;
  tenryu::radiation::PhotonPool pool;
  std::vector<double> energies(static_cast<std::size_t>(n), 1.0);
  std::vector<std::int32_t> cells(static_cast<std::size_t>(n), 0);
  std::vector<std::uint16_t> groups(static_cast<std::size_t>(n), 0);
  std::vector<std::uint8_t> modes(static_cast<std::size_t>(n), kModeIMC);
  setup_pool(pool, n, energies, cells, groups, modes);

  CensusCombConfig cfg{};
  cfg.max_particles = 100;
  cfg.target_fraction = 0.3;
  cfg.min_per_bin = 1;

  double* d_E_numerical_loss = allocate_device_loss();
  const auto result = tenryu::radiation::census_comb(pool,
                                                     n,
                                                     1,
                                                     1,
                                                     cfg,
                                                     2000ULL,
                                                     0ULL,
                                                     11U,
                                                     1,
                                                     d_E_numerical_loss);

  REQUIRE(result.n_alive_out == 30);
  REQUIRE(pool.n_alive == 30);
  cuda_check(cudaFree(d_E_numerical_loss));
}

TEST_CASE("census_comb: global_id uniqueness after combing",
          "[radiation][census_comb]") {
  constexpr int n = 50;
  tenryu::radiation::PhotonPool pool;
  std::vector<double> energies(static_cast<std::size_t>(n), 1.0);
  std::vector<std::int32_t> cells(static_cast<std::size_t>(n), 0);
  std::vector<std::uint16_t> groups(static_cast<std::size_t>(n), 0);
  std::vector<std::uint8_t> modes(static_cast<std::size_t>(n), kModeIMC);
  for (int i = 0; i < n; ++i) {
    cells[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % 5);
  }
  setup_pool(pool, n, energies, cells, groups, modes);

  CensusCombConfig cfg{};
  cfg.max_particles = 50;
  cfg.target_fraction = 0.5;
  cfg.min_per_bin = 1;

  double* d_E_numerical_loss = allocate_device_loss();
  const auto result = tenryu::radiation::census_comb(pool,
                                                     n,
                                                     5,
                                                     1,
                                                     cfg,
                                                     3000ULL,
                                                     0ULL,
                                                     19U,
                                                     2,
                                                     d_E_numerical_loss);

  const auto gids = read_field(pool.global_id, result.n_alive_out);
  const std::set<std::uint64_t> unique_ids(gids.begin(), gids.end());
  REQUIRE(unique_ids.size() == gids.size());
  cuda_check(cudaFree(d_E_numerical_loss));
}

TEST_CASE("census_comb: DDMC NaN preservation", "[radiation][census_comb]") {
  constexpr int n = 20;
  tenryu::radiation::PhotonPool pool;
  std::vector<double> energies(static_cast<std::size_t>(n), 1.0);
  std::vector<std::int32_t> cells(static_cast<std::size_t>(n), 0);
  std::vector<std::uint16_t> groups(static_cast<std::size_t>(n), 0);
  std::vector<std::uint8_t> modes(static_cast<std::size_t>(n), kModeIMC);
  for (int i = 10; i < n; ++i) {
    cells[static_cast<std::size_t>(i)] = 1;
    modes[static_cast<std::size_t>(i)] = kModeDDMC;
  }
  setup_pool(pool, n, energies, cells, groups, modes);

  CensusCombConfig cfg{};
  cfg.max_particles = 20;
  cfg.target_fraction = 0.5;
  cfg.min_per_bin = 1;

  double* d_E_numerical_loss = allocate_device_loss();
  const auto result = tenryu::radiation::census_comb(pool,
                                                     n,
                                                     2,
                                                     1,
                                                     cfg,
                                                     4000ULL,
                                                     0ULL,
                                                     23U,
                                                     3,
                                                     d_E_numerical_loss);

  const auto out_mode = read_field(pool.mode, result.n_alive_out);
  const auto out_pos_r = read_field(pool.pos_r, result.n_alive_out);
  const auto out_pos_z = read_field(pool.pos_z, result.n_alive_out);
  const auto out_dir_r = read_field(pool.dir_r, result.n_alive_out);
  const auto out_dir_z = read_field(pool.dir_z, result.n_alive_out);
  const auto out_dir_phi = read_field(pool.dir_phi, result.n_alive_out);

  int n_ddmc_out = 0;
  for (int i = 0; i < result.n_alive_out; ++i) {
    if (out_mode[static_cast<std::size_t>(i)] == kModeDDMC) {
      ++n_ddmc_out;
      REQUIRE(std::isnan(out_pos_r[static_cast<std::size_t>(i)]));
      REQUIRE(std::isnan(out_pos_z[static_cast<std::size_t>(i)]));
      REQUIRE(std::isnan(out_dir_r[static_cast<std::size_t>(i)]));
      REQUIRE(std::isnan(out_dir_z[static_cast<std::size_t>(i)]));
      REQUIRE(std::isnan(out_dir_phi[static_cast<std::size_t>(i)]));
    }
  }
  REQUIRE(n_ddmc_out > 0);
  cuda_check(cudaFree(d_E_numerical_loss));
}

TEST_CASE("census_comb: emergency path when bins exceed budget",
          "[radiation][census_comb]") {
  constexpr int n = 100;
  tenryu::radiation::PhotonPool pool;
  std::vector<double> energies(static_cast<std::size_t>(n), 1.0);
  std::vector<std::int32_t> cells(static_cast<std::size_t>(n), 0);
  std::vector<std::uint16_t> groups(static_cast<std::size_t>(n), 0);
  std::vector<std::uint8_t> modes(static_cast<std::size_t>(n), kModeIMC);
  for (int i = 0; i < n; ++i) {
    cells[static_cast<std::size_t>(i)] = i;
  }
  setup_pool(pool, n, energies, cells, groups, modes);

  CensusCombConfig cfg{};
  cfg.max_particles = 100;
  cfg.target_fraction = 0.8;
  cfg.min_per_bin = 2;

  double* d_E_numerical_loss = allocate_device_loss();
  const auto result = tenryu::radiation::census_comb(pool,
                                                     n,
                                                     100,
                                                     1,
                                                     cfg,
                                                     5000ULL,
                                                     0ULL,
                                                     29U,
                                                     4,
                                                     d_E_numerical_loss);

  REQUIRE(result.emergency);
  REQUIRE(result.n_alive_out <= 80);
  cuda_check(cudaFree(d_E_numerical_loss));
}

TEST_CASE("census_comb: zero-energy bin killed", "[radiation][census_comb]") {
  constexpr int n = 10;
  tenryu::radiation::PhotonPool pool;
  std::vector<double> energies = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<std::int32_t> cells = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  std::vector<std::uint16_t> groups(static_cast<std::size_t>(n), 0);
  std::vector<std::uint8_t> modes(static_cast<std::size_t>(n), kModeIMC);
  setup_pool(pool, n, energies, cells, groups, modes);

  CensusCombConfig cfg{};
  cfg.max_particles = 10;
  cfg.target_fraction = 0.5;
  cfg.min_per_bin = 1;

  double* d_E_numerical_loss = allocate_device_loss();
  const auto result = tenryu::radiation::census_comb(pool,
                                                     n,
                                                     2,
                                                     1,
                                                     cfg,
                                                     6000ULL,
                                                     0ULL,
                                                     31U,
                                                     5,
                                                     d_E_numerical_loss);

  const auto out_cells = read_field(pool.cell_id, result.n_alive_out);
  for (const auto cell : out_cells) {
    REQUIRE(cell == 1);
  }
  REQUIRE(result.zero_score == false);
  REQUIRE(result.E_killed_bins >= 0.0);
  cuda_check(cudaFree(d_E_numerical_loss));
}

TEST_CASE("census_comb: deterministic with same seed", "[radiation][census_comb]") {
  constexpr int n = 40;
  std::vector<double> energies(static_cast<std::size_t>(n), 0.0);
  std::vector<std::int32_t> cells(static_cast<std::size_t>(n), 0);
  std::vector<std::uint16_t> groups(static_cast<std::size_t>(n), 0);
  std::vector<std::uint8_t> modes(static_cast<std::size_t>(n), kModeIMC);
  for (int i = 0; i < n; ++i) {
    energies[static_cast<std::size_t>(i)] = static_cast<double>((i % 9) + 1);
    cells[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i % 4);
  }

  tenryu::radiation::PhotonPool pool_a;
  tenryu::radiation::PhotonPool pool_b;
  setup_pool(pool_a, n, energies, cells, groups, modes);
  setup_pool(pool_b, n, energies, cells, groups, modes);

  CensusCombConfig cfg{};
  cfg.max_particles = 40;
  cfg.target_fraction = 0.4;
  cfg.min_per_bin = 1;

  double* d_loss_a = allocate_device_loss();
  double* d_loss_b = allocate_device_loss();

  const auto result_a = tenryu::radiation::census_comb(pool_a,
                                                       n,
                                                       4,
                                                       1,
                                                       cfg,
                                                       7000ULL,
                                                       17ULL,
                                                       37U,
                                                       6,
                                                       d_loss_a);
  const auto result_b = tenryu::radiation::census_comb(pool_b,
                                                       n,
                                                       4,
                                                       1,
                                                       cfg,
                                                       7000ULL,
                                                       17ULL,
                                                       37U,
                                                       6,
                                                       d_loss_b);

  REQUIRE(result_a.n_alive_out == result_b.n_alive_out);
  REQUIRE(result_a.E_before == result_b.E_before);
  REQUIRE(result_a.E_after == result_b.E_after);
  REQUIRE(result_a.E_killed_bins == result_b.E_killed_bins);

  const auto energy_a = read_energy(pool_a, result_a.n_alive_out);
  const auto energy_b = read_energy(pool_b, result_b.n_alive_out);
  const auto gid_a = read_field(pool_a.global_id, result_a.n_alive_out);
  const auto gid_b = read_field(pool_b.global_id, result_b.n_alive_out);
  REQUIRE(energy_a == energy_b);
  REQUIRE(gid_a == gid_b);

  cuda_check(cudaFree(d_loss_a));
  cuda_check(cudaFree(d_loss_b));
}
