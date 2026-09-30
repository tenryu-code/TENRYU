#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/state.hpp"
#include "materials/ionmix_reader.hpp"
#include "radiation/groups.cuh"
#include "radiation/nlte_coeffs.hpp"
#include "radiation/planck_table.cuh"

namespace {

bool cuda_available() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

tenryu::core::Config make_cfg(const std::vector<double>& group_bounds_eV) {
  tenryu::core::Config cfg;
  cfg.main.dimension = "1D_SPH";
  cfg.main.dim = 1;
  cfg.mesh.nr = 1;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 1.0;
  cfg.mesh.r_max = 2.0;
  cfg.mesh.grid_type_r = "uniform";
  tenryu::core::Config::MaterialsConfig::MatDef mat;
  mat.name = "test";
  mat.A = 6.5;
  mat.Z = 3.5;
  mat.opacity_model = "constant";
  cfg.materials.materials = {mat};
  cfg.materials.zbar.model = "fixed";
  cfg.materials.zbar.fixed_value = 1.0;
  cfg.radiation.enabled = true;
  cfg.radiation.mode = tenryu::core::RadiationMode::ImcDdmc;
  cfg.radiation.groups = static_cast<int>(group_bounds_eV.size()) - 1;
  cfg.radiation.group_bounds_eV = group_bounds_eV;
  cfg.radiation.compute_T_range_eV = {0.1, 100.0};
  cfg.radiation.planck_fraction.compute_N_T = 128;
  return cfg;
}

int sample_from_cdf(const std::vector<double>& cdf, const double xi) {
  for (int g = 0; g < static_cast<int>(cdf.size()); ++g) {
    if (xi <= cdf[static_cast<std::size_t>(g)]) {
      return g;
    }
  }
  return static_cast<int>(cdf.size()) - 1;
}

}  // namespace

TEST_CASE("NLTE separate emissivity recovers LTE multigroup kernel",
          "[radiation][nlte][fleck]") {
  const std::vector<double> b = {0.1, 0.2, 0.3, 0.4};
  const std::vector<double> alpha = {1.0, 2.0, 3.0, 4.0};
  const double T = 10.0;
  const double source_scale =
      tenryu::core::constants::a_eV * tenryu::core::constants::c_light * T * T * T * T;

  std::vector<double> eta(alpha.size(), 0.0);
  for (std::size_t g = 0; g < alpha.size(); ++g) {
    eta[g] = alpha[g] * b[g] * source_scale;
  }

  const auto scalars = tenryu::radiation::compute_separate_emissivity_scalars(
      alpha, eta, b, T, 0.5, 1.0e-12, 1.0, 1.0e-6);

  CHECK(scalars.sigma_p_abs == Catch::Approx(3.0).margin(1.0e-12));
  CHECK(scalars.sigma_p_em == Catch::Approx(3.0).margin(1.0e-12));
  CHECK(scalars.gamma_diag == Catch::Approx(1.0).margin(1.0e-12));
  CHECK(scalars.s[0] == Catch::Approx(1.0 / 30.0).margin(1.0e-12));
  CHECK(scalars.s[1] == Catch::Approx(4.0 / 30.0).margin(1.0e-12));
  CHECK(scalars.s[2] == Catch::Approx(9.0 / 30.0).margin(1.0e-12));
  CHECK(scalars.s[3] == Catch::Approx(16.0 / 30.0).margin(1.0e-12));
}

TEST_CASE("NLTE separate emissivity uses s_g for redistribution",
          "[radiation][nlte][fleck]") {
  const std::vector<double> b = {0.1, 0.2, 0.3, 0.4};
  const std::vector<double> alpha = {4.0, 4.0, 1.0, 0.5};
  const std::vector<double> s = {0.02, 0.08, 0.30, 0.60};
  const double sigma_p_em = 1.2;
  const double T = 10.0;
  const double dt = 1.0e-12;
  const double beta = 0.5 / (tenryu::core::constants::c_light * dt);
  const double source_scale =
      tenryu::core::constants::a_eV * tenryu::core::constants::c_light * T * T * T * T;

  std::vector<double> eta(s.size(), 0.0);
  for (std::size_t g = 0; g < s.size(); ++g) {
    eta[g] = s[g] * sigma_p_em * source_scale;
  }

  const auto scalars = tenryu::radiation::compute_separate_emissivity_scalars(
      alpha, eta, b, T, beta, dt, 1.0, 1.0e-6);

  CHECK(scalars.f == Catch::Approx(0.625).margin(1.0e-12));

  std::vector<double> old_lte(alpha.size(), 0.0);
  for (std::size_t g = 0; g < alpha.size(); ++g) {
    old_lte[g] = alpha[g] * b[g] / scalars.sigma_p_abs;
  }
  double l1_old = 0.0;
  for (std::size_t g = 0; g < s.size(); ++g) {
    l1_old += std::abs(old_lte[g] - scalars.s[g]);
  }
  CHECK(l1_old > 0.5);

  constexpr int kSamples = 200000;
  std::mt19937_64 rng(12345);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  std::array<int, 4> counts = {0, 0, 0, 0};
  for (int i = 0; i < kSamples; ++i) {
    const int g = sample_from_cdf(scalars.cdf, dist(rng));
    counts[static_cast<std::size_t>(g)] += 1;
  }

  double chi2 = 0.0;
  for (int g = 0; g < 4; ++g) {
    const double expected = s[static_cast<std::size_t>(g)] * static_cast<double>(kSamples);
    const double observed = static_cast<double>(counts[static_cast<std::size_t>(g)]);
    const double diff = observed - expected;
    chi2 += diff * diff / std::max(expected, 1.0);
  }
  CHECK(chi2 < 30.0);
}

TEST_CASE("NLTE table correction applies linear J_g response",
          "[radiation][nlte][fleck]") {
  const std::vector<double> eta0 = {1.0, 2.0};
  const std::vector<double> alpha0 = {3.0, 4.0};
  const std::vector<double> J0 = {10.0, 20.0};
  const std::vector<double> deta_dJ = {0.1, 0.0,
                                       0.0, -0.05};
  const std::vector<double> dalpha_dJ = {0.2, 0.0,
                                         0.0, 0.1};
  const std::vector<double> J = {11.0, 18.0};

  std::vector<double> eta;
  std::vector<double> alpha;
  tenryu::radiation::apply_nlte_transport_linearized_correction(
      2, eta0, alpha0, J0, deta_dJ, dalpha_dJ, J, &eta, &alpha);

  REQUIRE(eta.size() == 2);
  REQUIRE(alpha.size() == 2);
  CHECK(eta[0] == Catch::Approx(1.1).margin(1.0e-12));
  CHECK(eta[1] == Catch::Approx(2.1).margin(1.0e-12));
  CHECK(alpha[0] == Catch::Approx(3.2).margin(1.0e-12));
  CHECK(alpha[1] == Catch::Approx(3.8).margin(1.0e-12));
}

TEST_CASE("NLTE spectrum reconstruction fills missing groups with Planck fallback",
          "[radiation][nlte][fleck]") {
  if (!cuda_available()) {
    INFO("Skipping NLTE spectrum reconstruction test because CUDA device is unavailable");
    SUCCEED();
    return;
  }

  const auto table =
      tenryu::materials::load_ionmix_opacity("tests/data/ionmix_lte_const.cn4");
  const auto cfg = make_cfg(table.bounds_eV);
  auto state = tenryu::core::State::allocate(cfg);

  const std::vector<double> rad_E = {1.0, 0.0, 2.0, 0.0};
  state.rad_E.copy_from_host(rad_E.data());

  tenryu::radiation::Groups groups(cfg.radiation.group_bounds_eV);
  tenryu::radiation::PlanckTable planck;
  planck.build(groups,
               cfg.radiation.planck_fraction.compute_N_T,
               cfg.radiation.compute_T_range_eV[0],
               cfg.radiation.compute_T_range_eV[1]);

  std::vector<double> J;
  std::vector<std::uint8_t> fallback;
  const auto diag = tenryu::radiation::reconstruct_nlte_group_spectrum(
      state, planck, 1, cfg.radiation.groups, &J, &fallback);

  REQUIRE(J.size() == 4);
  REQUIRE(fallback.size() == 4);
  CHECK(diag.fallback_cell_count == 1);
  CHECK(diag.fallback_group_count == 2);
  CHECK(fallback[1] == static_cast<std::uint8_t>(1));
  CHECK(fallback[3] == static_cast<std::uint8_t>(1));
  CHECK(std::all_of(J.begin(), J.end(), [](const double x) { return std::isfinite(x); }));
  CHECK(J[1] > 0.0);
  CHECK(J[3] > 0.0);
  CHECK(std::accumulate(J.begin(), J.end(), 0.0) == Catch::Approx(3.0).epsilon(1.0e-12));
}
