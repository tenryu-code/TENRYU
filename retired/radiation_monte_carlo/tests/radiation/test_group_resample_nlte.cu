#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/config.hpp"
#include "core/state.hpp"
#include "materials/ionmix_reader.hpp"
#include "radiation/groups.cuh"
#include "radiation/nlte_coeffs.hpp"
#include "radiation/planck_table.cuh"

namespace {

tenryu::core::Config make_cfg(const tenryu::materials::IonmixOpacityData& table) {
  tenryu::core::Config cfg;
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;

  cfg.mesh.nr = 1;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 1.0;
  cfg.mesh.r_max = 2.0;
  cfg.mesh.grid_type_r = "uniform";

  tenryu::core::Config::MaterialsConfig::MatDef mat;
  mat.name = "nlte_resample";
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
  cfg.radiation.planck_fraction.compute_N_T = 200;
  cfg.radiation.imc.alpha = 1.0;
  cfg.radiation.imc.f_max = 1.0;
  cfg.numerics.floors.Te = 1.0e-3;
  return cfg;
}

int sample_from_cdf(const std::vector<double>& cdf,
                    const int n_groups,
                    const double xi) {
  for (int g = 0; g < n_groups; ++g) {
    if (xi <= cdf[static_cast<std::size_t>(g)]) {
      return g;
    }
  }
  return n_groups - 1;
}

}  // namespace

TEST_CASE("NLTE group resample follows eta CDF", "[radiation][nlte][group_resample]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping NLTE group-resample test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  const auto cfg = make_cfg(table);

  auto state = tenryu::core::State::allocate(cfg);
  const std::vector<double> rho_h = {1.0};
  const std::vector<double> Te_h = {10.0};
  const std::vector<double> zbar_h = {3.5};
  state.rho.copy_from_host(rho_h.data());
  state.Te.copy_from_host(Te_h.data());
  state.zbar.copy_from_host(zbar_h.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups,
               cfg.radiation.planck_fraction.compute_N_T,
               cfg.radiation.compute_T_range_eV[0],
               cfg.radiation.compute_T_range_eV[1]);

  const auto coeffs =
      tenryu::radiation::compute_nlte_coefficients(state, cfg, table, planck, 1, 4, 1.0e-11);

  const std::vector<double> cdf(coeffs.eta_cdf.begin(), coeffs.eta_cdf.begin() + 4);
  const double eta_tot = std::max(coeffs.eta_tot[0], 1.0e-30);

  constexpr int kSamples = 200000;
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  std::array<int, 4> counts = {0, 0, 0, 0};
  for (int i = 0; i < kSamples; ++i) {
    const int g = sample_from_cdf(cdf, 4, dist(rng));
    counts[static_cast<std::size_t>(g)] += 1;
  }

  double chi2 = 0.0;
  for (int g = 0; g < 4; ++g) {
    const double p = std::max(coeffs.eta[static_cast<std::size_t>(g)] / eta_tot, 1.0e-12);
    const double expected = p * static_cast<double>(kSamples);
    const double observed = static_cast<double>(counts[static_cast<std::size_t>(g)]);
    const double diff = observed - expected;
    chi2 += diff * diff / std::max(expected, 1.0);
  }

  // dof=3, conservative acceptance envelope.
  CHECK(chi2 < 30.0);
}

TEST_CASE("NLTE group resample: CDF monotonicity", "[radiation][nlte][group_resample]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping NLTE group-resample test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  const auto cfg = make_cfg(table);

  auto state = tenryu::core::State::allocate(cfg);
  const std::vector<double> rho_h = {1.0};
  const std::vector<double> Te_h = {10.0};
  const std::vector<double> zbar_h = {3.5};
  state.rho.copy_from_host(rho_h.data());
  state.Te.copy_from_host(Te_h.data());
  state.zbar.copy_from_host(zbar_h.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups,
               cfg.radiation.planck_fraction.compute_N_T,
               cfg.radiation.compute_T_range_eV[0],
               cfg.radiation.compute_T_range_eV[1]);

  const auto coeffs =
      tenryu::radiation::compute_nlte_coefficients(state, cfg, table, planck, 1, 4, 1.0e-11);

  CHECK(coeffs.eta_cdf[0] > 0.0);
  for (int g = 1; g < 4; ++g) {
    CHECK(coeffs.eta_cdf[static_cast<std::size_t>(g)] >=
          coeffs.eta_cdf[static_cast<std::size_t>(g - 1)]);
  }
}

TEST_CASE("NLTE group resample: CDF terminal normalization",
          "[radiation][nlte][group_resample]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping NLTE group-resample test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  const auto cfg = make_cfg(table);

  auto state = tenryu::core::State::allocate(cfg);
  const std::vector<double> rho_h = {1.0};
  const std::vector<double> Te_h = {10.0};
  const std::vector<double> zbar_h = {3.5};
  state.rho.copy_from_host(rho_h.data());
  state.Te.copy_from_host(Te_h.data());
  state.zbar.copy_from_host(zbar_h.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups,
               cfg.radiation.planck_fraction.compute_N_T,
               cfg.radiation.compute_T_range_eV[0],
               cfg.radiation.compute_T_range_eV[1]);

  const auto coeffs =
      tenryu::radiation::compute_nlte_coefficients(state, cfg, table, planck, 1, 4, 1.0e-11);

  CHECK(coeffs.eta_cdf[3] == Catch::Approx(1.0));
}

TEST_CASE("NLTE group resample: boundary group mapping",
          "[radiation][nlte][group_resample]") {
  int device_count = 0;
  const cudaError_t dev_err = cudaGetDeviceCount(&device_count);
  if (dev_err != cudaSuccess || device_count <= 0) {
    INFO("Skipping NLTE group-resample test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_nlte_simple.cn4");
  const auto cfg = make_cfg(table);

  auto state = tenryu::core::State::allocate(cfg);
  const std::vector<double> rho_h = {1.0};
  const std::vector<double> Te_h = {10.0};
  const std::vector<double> zbar_h = {3.5};
  state.rho.copy_from_host(rho_h.data());
  state.Te.copy_from_host(Te_h.data());
  state.zbar.copy_from_host(zbar_h.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups,
               cfg.radiation.planck_fraction.compute_N_T,
               cfg.radiation.compute_T_range_eV[0],
               cfg.radiation.compute_T_range_eV[1]);

  const auto coeffs =
      tenryu::radiation::compute_nlte_coefficients(state, cfg, table, planck, 1, 4, 1.0e-11);

  const std::vector<double> cdf(coeffs.eta_cdf.begin(), coeffs.eta_cdf.begin() + 4);
  const int g0 = sample_from_cdf(cdf, 4, 0.0);
  const int g_last = sample_from_cdf(cdf, 4, 1.0 - 1.0e-12);

  CHECK(g0 == 0);
  CHECK(g_last == 3);
}
