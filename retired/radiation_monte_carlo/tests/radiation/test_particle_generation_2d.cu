#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"
#include "radiation/source.cuh"

namespace {

void cuda_check(const cudaError_t err, const char* msg) {
  TENRYU_ASSERT(err == cudaSuccess, msg);
}

double ks_uniform_minus1_1(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  const double n = static_cast<double>(samples.size());
  double d_max = 0.0;
  for (std::size_t i = 0; i < samples.size(); ++i) {
    const double x = samples[i];
    const double f = std::clamp(0.5 * (x + 1.0), 0.0, 1.0);
    const double fn_lo = static_cast<double>(i) / n;
    const double fn_hi = static_cast<double>(i + 1) / n;
    d_max = std::max(d_max, std::abs(f - fn_lo));
    d_max = std::max(d_max, std::abs(fn_hi - f));
  }
  return d_max;
}

double chi_square_r_weight(const std::vector<double>& r_samples,
                           const double r_max,
                           const int n_bins) {
  std::vector<double> obs(static_cast<std::size_t>(n_bins), 0.0);
  for (const double r : r_samples) {
    int bin = static_cast<int>((r / r_max) * n_bins);
    if (bin < 0) {
      bin = 0;
    }
    if (bin >= n_bins) {
      bin = n_bins - 1;
    }
    obs[static_cast<std::size_t>(bin)] += 1.0;
  }

  const double n = static_cast<double>(r_samples.size());
  double chi2 = 0.0;
  for (int b = 0; b < n_bins; ++b) {
    const double r0 = r_max * static_cast<double>(b) / static_cast<double>(n_bins);
    const double r1 = r_max * static_cast<double>(b + 1) / static_cast<double>(n_bins);
    const double p = (r1 * r1 - r0 * r0) / (r_max * r_max);
    const double exp = std::max(n * p, 1.0);
    const double diff = obs[static_cast<std::size_t>(b)] - exp;
    chi2 += diff * diff / exp;
  }
  return chi2;
}

}  // namespace

TEST_CASE("2D particle generation follows R-weighted volume and isotropic direction",
          "[radiation][source][2d]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping particle generation 2D test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  tenryu::core::Config cfg;
  cfg.main.dim = 2;
  cfg.main.dimension = "2D_RZ";
  cfg.mesh.nr = 1;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = 1.0;
  cfg.mesh.z_min = 0.0;
  cfg.mesh.z_max = 1.0;
  cfg.mesh.grid_type_r = "uniform";
  cfg.mesh.grid_type_z = "uniform";
  cfg.radiation.enabled = true;
  cfg.radiation.mode = tenryu::core::RadiationMode::ImcDdmc;
  cfg.radiation.groups = 1;
  cfg.radiation.volume_source_rate = 1.0;
  cfg.radiation.volume_source_x_max = 1.0;
  cfg.radiation.imc.particles_per_cell_group = 50000;

  auto state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;

  tenryu::radiation::PhotonPool pool;
  const double dt = 1.0e-12;
  const auto stats = tenryu::radiation::IMCSource::emit_volume_source(
      state,
      cfg,
      pool,
      200000,
      dt,
      0,
      12345,
      0);

  REQUIRE(stats.n_thermal > 1000);
  REQUIRE(pool.n_alive == stats.n_thermal);

  const int n = pool.n_alive;
  std::vector<double> pos_r(static_cast<std::size_t>(n), 0.0);
  std::vector<double> pos_z(static_cast<std::size_t>(n), 0.0);
  std::vector<double> dir_r(static_cast<std::size_t>(n), 0.0);
  std::vector<double> dir_z(static_cast<std::size_t>(n), 0.0);
  std::vector<double> dir_phi(static_cast<std::size_t>(n), 0.0);

  cuda_check(cudaMemcpy(pos_r.data(), pool.pos_r, sizeof(double) * pos_r.size(),
                        cudaMemcpyDeviceToHost),
             "copy pos_r failed");
  cuda_check(cudaMemcpy(pos_z.data(), pool.pos_z, sizeof(double) * pos_z.size(),
                        cudaMemcpyDeviceToHost),
             "copy pos_z failed");
  cuda_check(cudaMemcpy(dir_r.data(), pool.dir_r, sizeof(double) * dir_r.size(),
                        cudaMemcpyDeviceToHost),
             "copy dir_r failed");
  cuda_check(cudaMemcpy(dir_z.data(), pool.dir_z, sizeof(double) * dir_z.size(),
                        cudaMemcpyDeviceToHost),
             "copy dir_z failed");
  cuda_check(cudaMemcpy(dir_phi.data(), pool.dir_phi, sizeof(double) * dir_phi.size(),
                        cudaMemcpyDeviceToHost),
             "copy dir_phi failed");

  double mean_r = 0.0;
  double mean_z = 0.0;
  double mean_dr = 0.0;
  double mean_dz = 0.0;
  double mean_dphi = 0.0;
  for (int i = 0; i < n; ++i) {
    REQUIRE(pos_r[static_cast<std::size_t>(i)] >= 0.0);
    REQUIRE(pos_r[static_cast<std::size_t>(i)] <= 1.0);
    REQUIRE(pos_z[static_cast<std::size_t>(i)] >= 0.0);
    REQUIRE(pos_z[static_cast<std::size_t>(i)] <= 1.0);

    const double norm = std::sqrt(dir_r[static_cast<std::size_t>(i)] *
                                      dir_r[static_cast<std::size_t>(i)] +
                                  dir_z[static_cast<std::size_t>(i)] *
                                      dir_z[static_cast<std::size_t>(i)] +
                                  dir_phi[static_cast<std::size_t>(i)] *
                                      dir_phi[static_cast<std::size_t>(i)]);
    REQUIRE(norm == Catch::Approx(1.0).epsilon(1.0e-10));

    mean_r += pos_r[static_cast<std::size_t>(i)];
    mean_z += pos_z[static_cast<std::size_t>(i)];
    mean_dr += dir_r[static_cast<std::size_t>(i)];
    mean_dz += dir_z[static_cast<std::size_t>(i)];
    mean_dphi += dir_phi[static_cast<std::size_t>(i)];
  }
  mean_r /= static_cast<double>(n);
  mean_z /= static_cast<double>(n);
  mean_dr /= static_cast<double>(n);
  mean_dz /= static_cast<double>(n);
  mean_dphi /= static_cast<double>(n);

  REQUIRE(mean_r == Catch::Approx(2.0 / 3.0).margin(0.02));
  REQUIRE(mean_z == Catch::Approx(0.5).margin(0.02));

  const double chi2 = chi_square_r_weight(pos_r, 1.0, 10);
  REQUIRE(chi2 < 35.0);

  const double ks = ks_uniform_minus1_1(dir_z);
  REQUIRE(ks < (2.0 / std::sqrt(static_cast<double>(n))));

  REQUIRE(std::abs(mean_dr) < 0.02);
  REQUIRE(std::abs(mean_dz) < 0.02);
  REQUIRE(std::abs(mean_dphi) < 0.02);
}
