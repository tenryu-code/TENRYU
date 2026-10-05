#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_remap.cuh"
#include "mesh/mesh.hpp"

namespace {

constexpr double kFourPiOverThree =
    4.188790204786390984616857844372670512262892532500141094646;
constexpr double kPi = 3.141592653589793238462643383279502884;

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

tenryu::core::Config make_cfg(const int n,
                              const int n_groups = 1,
                              const int n_mat = 1) {
  tenryu::core::Config cfg;
  cfg.main.dim = 1;
  cfg.main.dimension = "1D_SPH";
  cfg.mesh.nr = n;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = 1.0;
  cfg.mesh.grid_type_r = "uniform";
  cfg.radiation.groups = n_groups;
  cfg.numerics.ale1d.enabled = true;

  cfg.materials.materials.clear();
  for (int m = 0; m < n_mat; ++m) {
    tenryu::core::Config::MaterialsConfig::MatDef mat;
    mat.name = "mat" + std::to_string(m);
    mat.A = 1.0;
    mat.Z = 1.0;
    cfg.materials.materials.push_back(mat);
  }
  return cfg;
}

template <typename Tag>
std::vector<double> to_host(const tenryu::core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

template <typename T>
std::vector<T> to_host(const tenryu::hydro::ale1d::DeviceArray<T>& field) {
  std::vector<T> host;
  field.copy_to_host(host);
  return host;
}

tenryu::hydro::ale1d::NodeConstraintMask default_mask(const int n) {
  tenryu::hydro::ale1d::NodeConstraintMask mask;
  mask.pinned.assign(static_cast<std::size_t>(n + 1), false);
  mask.pinned.front() = true;
  mask.pinned.back() = true;
  mask.n_protected_nodes = 2;
  return mask;
}

tenryu::core::State make_state(const tenryu::core::Config& cfg,
                               const std::vector<double>& rho,
                               const std::vector<double>& ee,
                               const std::vector<double>& ei,
                               const std::vector<double>& rad_e,
                               const std::vector<double>& volfrac) {
  tenryu::core::State state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;

  const int n = cfg.mesh.nr;
  const std::vector<double> vol = to_host(state.vol);
  std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    mass[static_cast<std::size_t>(i)] =
        rho[static_cast<std::size_t>(i)] * vol[static_cast<std::size_t>(i)];
  }

  state.rho.copy_from_host(rho);
  state.mass.copy_from_host(mass);
  state.ee.copy_from_host(ee);
  state.ei.copy_from_host(ei);
  state.rad_E.copy_from_host(rad_e);
  state.volFrac.copy_from_host(volfrac);
  return state;
}

std::vector<double> shifted_nodes(const std::vector<double>& old_r,
                                  const double fraction) {
  const int n = static_cast<int>(old_r.size()) - 1;
  std::vector<double> candidate = old_r;
  const double dr = old_r[1] - old_r[0];
  for (int j = 1; j < n; ++j) {
    candidate[static_cast<std::size_t>(j)] += fraction * dr;
  }
  return candidate;
}

std::vector<double> random_shifted_nodes(const std::vector<double>& old_r) {
  const int n = static_cast<int>(old_r.size()) - 1;
  std::vector<double> candidate = old_r;
  const double dr = old_r[1] - old_r[0];
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(-0.25, 0.25);
  for (int j = 1; j < n; ++j) {
    candidate[static_cast<std::size_t>(j)] += dist(rng) * dr;
  }
  return candidate;
}

long double sum_vector(const std::vector<double>& values) {
  long double sum = 0.0L;
  for (const double value : values) {
    sum += static_cast<long double>(value);
  }
  return sum;
}

long double material_total(const std::vector<double>& mass,
                           const std::vector<double>& volfrac,
                           const int n_mat,
                           const int mat) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < mass.size(); ++i) {
    sum += static_cast<long double>(mass[i]) *
           static_cast<long double>(volfrac[i * static_cast<std::size_t>(n_mat) +
                                            static_cast<std::size_t>(mat)]);
  }
  return sum;
}

long double energy_total(const std::vector<double>& mass,
                         const std::vector<double>& e) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < mass.size(); ++i) {
    sum += static_cast<long double>(mass[i]) * static_cast<long double>(e[i]);
  }
  return sum;
}

double volume_coordinate(const double r) {
  return kFourPiOverThree * r * r * r;
}

double smooth_average(const double ya, const double yb, const double y_total) {
  const double dy = yb - ya;
  const double k = 2.0 * kPi / y_total;
  return 1.0 + 0.1 * (std::cos(k * ya) - std::cos(k * yb)) / (k * dy);
}

template <typename T>
bool bitwise_equal(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  return lhs.size() == rhs.size() &&
         std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0;
}

}  // namespace

TEST_CASE("ALE1D first-order remap identity preserves fields",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 32;
  constexpr int n_groups = 2;
  constexpr int n_mat = 2;
  auto cfg = make_cfg(n, n_groups, n_mat);
  std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 0.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 0.0);
  std::vector<double> rad(static_cast<std::size_t>(n * n_groups), 0.0);
  std::vector<double> vf(static_cast<std::size_t>(n * n_mat), 0.0);
  for (int i = 0; i < n; ++i) {
    rho[static_cast<std::size_t>(i)] = 1.0 + 0.01 * i;
    ee[static_cast<std::size_t>(i)] = 2.0 + 0.03 * i;
    ei[static_cast<std::size_t>(i)] = 1.0 + 0.02 * i;
    rad[static_cast<std::size_t>(i * n_groups)] = 4.0 + 0.05 * i;
    rad[static_cast<std::size_t>(i * n_groups + 1)] = 2.0 + 0.04 * i;
    const double f0 = 0.25 + 0.5 * static_cast<double>(i) /
                                static_cast<double>(n - 1);
    vf[static_cast<std::size_t>(i * n_mat)] = f0;
    vf[static_cast<std::size_t>(i * n_mat + 1)] = 1.0 - f0;
  }
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, n_groups, n_mat);
  const auto result = tenryu::hydro::ale1d::remap_first_order(
      state, cfg, old_r, default_mask(n), scratch);

  REQUIRE(result.success);
  CHECK(result.mass_conservation_rel_err <= 1.0e-14);
  CHECK(result.material_mass_conservation_rel_err <= 1.0e-14);
  CHECK(result.ee_conservation_rel_err <= 1.0e-14);
  CHECK(result.ei_conservation_rel_err <= 1.0e-14);
  CHECK(result.radiation_conservation_rel_err <= 1.0e-14);

  const auto mass_new = to_host(scratch.mass_new);
  const auto ee_new = to_host(scratch.ee_new);
  const auto ei_new = to_host(scratch.ei_new);
  const auto rad_new = to_host(scratch.rad_E_new);
  const auto vf_new = to_host(scratch.volFrac_new);
  const auto mass_old = to_host(state.mass);
  for (int i = 0; i < n; ++i) {
    CHECK(mass_new[static_cast<std::size_t>(i)] ==
          Catch::Approx(mass_old[static_cast<std::size_t>(i)]).epsilon(1.0e-14));
    CHECK(ee_new[static_cast<std::size_t>(i)] ==
          Catch::Approx(ee[static_cast<std::size_t>(i)]).epsilon(1.0e-14));
    CHECK(ei_new[static_cast<std::size_t>(i)] ==
          Catch::Approx(ei[static_cast<std::size_t>(i)]).epsilon(1.0e-14));
  }
  for (std::size_t i = 0; i < rad.size(); ++i) {
    CHECK(rad_new[i] == Catch::Approx(rad[i]).epsilon(1.0e-14));
  }
  for (std::size_t i = 0; i < vf.size(); ++i) {
    CHECK(vf_new[i] == Catch::Approx(vf[i]).epsilon(1.0e-14));
  }
}

TEST_CASE("ALE1D first-order remap preserves uniform shifted fields",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 64;
  auto cfg = make_cfg(n, 1, 1);
  std::vector<double> rho(static_cast<std::size_t>(n), 2.5);
  std::vector<double> ee(static_cast<std::size_t>(n), 3.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.5);
  std::vector<double> rad(static_cast<std::size_t>(n), 7.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);
  const std::vector<double> candidate = shifted_nodes(old_r, 0.1);

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, 1, 1);
  const auto result = tenryu::hydro::ale1d::remap_first_order(
      state, cfg, candidate, default_mask(n), scratch);

  REQUIRE(result.success);
  const auto mass_new = to_host(scratch.mass_new);
  const auto vol_new = to_host(scratch.vol_new);
  const auto ee_new = to_host(scratch.ee_new);
  const auto rad_new = to_host(scratch.rad_E_new);
  for (int i = 0; i < n; ++i) {
    CHECK(mass_new[static_cast<std::size_t>(i)] /
              vol_new[static_cast<std::size_t>(i)] ==
          Catch::Approx(2.5).epsilon(1.0e-13));
    CHECK(ee_new[static_cast<std::size_t>(i)] ==
          Catch::Approx(3.0).epsilon(1.0e-13));
    CHECK(rad_new[static_cast<std::size_t>(i)] ==
          Catch::Approx(7.0).epsilon(1.0e-13));
  }
}

TEST_CASE("ALE1D first-order remap keeps radiation top-hat positive",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 80;
  auto cfg = make_cfg(n, 1, 1);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 0.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  for (int i = n / 3; i < 2 * n / 3; ++i) {
    rad[static_cast<std::size_t>(i)] = 10.0;
  }
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);
  const std::vector<double> candidate = shifted_nodes(old_r, 0.25);

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, 1, 1);
  const auto result = tenryu::hydro::ale1d::remap_first_order(
      state, cfg, candidate, default_mask(n), scratch);

  REQUIRE(result.success);
  const auto rad_new = to_host(scratch.rad_E_new);
  const auto [min_it, max_it] = std::minmax_element(rad_new.begin(), rad_new.end());
  CHECK(*min_it >= -1.0e-12);
  CHECK(*max_it <= 10.0 + 1.0e-12);
}

TEST_CASE("ALE1D first-order remap conserves mass for random displacements",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 96;
  auto cfg = make_cfg(n, 1, 1);
  std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  for (int i = 0; i < n; ++i) {
    rho[static_cast<std::size_t>(i)] = 1.0 + 0.2 * std::sin(0.13 * i);
  }
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);
  const std::vector<double> candidate = random_shifted_nodes(old_r);

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, 1, 1);
  const auto result = tenryu::hydro::ale1d::remap_first_order(
      state, cfg, candidate, default_mask(n), scratch);

  REQUIRE(result.success);
  const long double old_sum = sum_vector(to_host(state.mass));
  const long double new_sum = sum_vector(to_host(scratch.mass_new));
  CHECK(static_cast<double>(std::abs(new_sum - old_sum) / std::abs(old_sum)) <=
        1.0e-12);
}

TEST_CASE("ALE1D first-order remap conserves material masses across interface",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 64;
  constexpr int n_mat = 2;
  auto cfg = make_cfg(n, 1, n_mat);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vf(static_cast<std::size_t>(n * n_mat), 0.0);
  for (int i = 0; i < n; ++i) {
    const bool left = i < n / 2;
    vf[static_cast<std::size_t>(i * n_mat)] = left ? 1.0 : 0.0;
    vf[static_cast<std::size_t>(i * n_mat + 1)] = left ? 0.0 : 1.0;
  }
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);
  std::vector<double> candidate = old_r;
  const double dr = old_r[1] - old_r[0];
  candidate[static_cast<std::size_t>(n / 2)] += 0.3 * dr;

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, 1, n_mat);
  const auto result = tenryu::hydro::ale1d::remap_first_order(
      state, cfg, candidate, default_mask(n), scratch);

  REQUIRE(result.success);
  const auto old_mass = to_host(state.mass);
  const auto new_mass = to_host(scratch.mass_new);
  const auto new_vf = to_host(scratch.volFrac_new);
  for (int m = 0; m < n_mat; ++m) {
    const long double old_total = material_total(old_mass, vf, n_mat, m);
    const long double new_total = material_total(new_mass, new_vf, n_mat, m);
    CHECK(static_cast<double>(std::abs(new_total - old_total) /
                              std::abs(old_total)) <= 1.0e-12);
  }
}

TEST_CASE("ALE1D first-order remap conserves electron energy",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 96;
  auto cfg = make_cfg(n, 1, 1);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 0.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  for (int i = 0; i < n; ++i) {
    ee[static_cast<std::size_t>(i)] = 1.0 + 0.5 * std::sin(0.21 * i);
  }
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);
  const std::vector<double> candidate = random_shifted_nodes(old_r);

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, 1, 1);
  const auto result = tenryu::hydro::ale1d::remap_first_order(
      state, cfg, candidate, default_mask(n), scratch);

  REQUIRE(result.success);
  const long double old_total = energy_total(to_host(state.mass), ee);
  const long double new_total =
      energy_total(to_host(scratch.mass_new), to_host(scratch.ee_new));
  CHECK(static_cast<double>(std::abs(new_total - old_total) /
                            std::abs(old_total)) <= 1.0e-10);
}

TEST_CASE("ALE1D first-order remap rejects multi-cell face sweeps",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 32;
  auto cfg = make_cfg(n, 1, 1);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);
  std::vector<double> candidate = old_r;
  const double dr = old_r[1] - old_r[0];
  candidate[static_cast<std::size_t>(n / 2)] += 1.5 * dr;

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, 1, 1);
  const auto result = tenryu::hydro::ale1d::remap_first_order(
      state, cfg, candidate, default_mask(n), scratch);

  REQUIRE_FALSE(result.success);
  CHECK(result.skip_reason ==
        tenryu::hydro::ale1d::Ale1dSkipReason::CandidateInvalid);
  CHECK(result.n_invalid_sweeps > 0);
}

TEST_CASE("ALE1D first-order remap enforces zero flux on pinned face",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 64;
  auto cfg = make_cfg(n, 1, 1);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);
  std::vector<double> candidate = shifted_nodes(old_r, 0.2);
  auto mask = default_mask(n);
  mask.pinned[static_cast<std::size_t>(n / 2)] = true;
  candidate[static_cast<std::size_t>(n / 2)] =
      old_r[static_cast<std::size_t>(n / 2)];

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, 1, 1);
  const auto result =
      tenryu::hydro::ale1d::remap_first_order(state, cfg, candidate, mask, scratch);

  REQUIRE(result.success);
  const auto delta_y = to_host(scratch.delta_Y);
  const auto donor = to_host(scratch.donor);
  CHECK(delta_y[static_cast<std::size_t>(n / 2)] == Catch::Approx(0.0));
  CHECK(donor[static_cast<std::size_t>(n / 2)] == -1);
}

TEST_CASE("ALE1D high-order remap converges on smooth density",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  const auto run_case = [](const int n) {
    auto cfg = make_cfg(n, 1, 1);
    cfg.numerics.ale1d.remap.high_order_enabled = true;
    std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
    std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
    std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
    std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
    std::vector<double> vf(static_cast<std::size_t>(n), 1.0);

    auto state0 = tenryu::core::State::allocate(cfg);
    state0.mesh = tenryu::mesh::create_mesh(cfg, state0);
    const std::vector<double> old_r = to_host(state0.x_r);
    const double y_total = volume_coordinate(old_r.back());
    for (int i = 0; i < n; ++i) {
      const double ya = volume_coordinate(old_r[static_cast<std::size_t>(i)]);
      const double yb = volume_coordinate(old_r[static_cast<std::size_t>(i + 1)]);
      rho[static_cast<std::size_t>(i)] = smooth_average(ya, yb, y_total);
    }

    auto state = make_state(cfg, rho, ee, ei, rad, vf);
    const std::vector<double> candidate = shifted_nodes(old_r, 0.18);
    tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
    scratch.resize(n, 1, 1);
    const auto result = tenryu::hydro::ale1d::remap_v3(
        state, cfg, candidate, default_mask(n), {}, scratch);
    REQUIRE(result.success);
    REQUIRE(result.n_bound_fallback_cells == 0);

    const auto mass_new = to_host(scratch.mass_new);
    const auto vol_new = to_host(scratch.vol_new);
    double err = 0.0;
    double norm = 0.0;
    for (int i = 4; i < n - 4; ++i) {
      const double ya = volume_coordinate(candidate[static_cast<std::size_t>(i)]);
      const double yb =
          volume_coordinate(candidate[static_cast<std::size_t>(i + 1)]);
      const double exact = smooth_average(ya, yb, y_total);
      const double got =
          mass_new[static_cast<std::size_t>(i)] /
          vol_new[static_cast<std::size_t>(i)];
      err += std::abs(got - exact);
      norm += std::abs(exact);
    }
    return err / norm;
  };

  const double e256 = run_case(256);
  const double e512 = run_case(512);
  const double order = std::log(e256 / e512) / std::log(2.0);
  CHECK(order >= 1.6);
}

TEST_CASE("ALE1D high-order remap respects protected top-hat bounds",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 96;
  auto cfg = make_cfg(n, 1, 1);
  cfg.numerics.ale1d.remap.high_order_enabled = true;
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 0.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  for (int i = n / 3; i < 2 * n / 3; ++i) {
    rad[static_cast<std::size_t>(i)] = 10.0;
  }
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);
  const std::vector<double> candidate = shifted_nodes(old_r, 0.25);

  tenryu::hydro::ale1d::Ale1dRemapScratch low;
  low.resize(n, 1, 1);
  const auto low_result = tenryu::hydro::ale1d::remap_first_order(
      state, cfg, candidate, default_mask(n), low);
  REQUIRE(low_result.success);
  const auto low_rad = to_host(low.rad_E_new);
  const auto [lo_it, hi_it] = std::minmax_element(low_rad.begin(), low_rad.end());

  tenryu::hydro::ale1d::Ale1dRemapScratch high;
  high.resize(n, 1, 1);
  const auto high_result = tenryu::hydro::ale1d::remap_v3(
      state, cfg, candidate, default_mask(n), {n / 3}, high);
  REQUIRE(high_result.success);
  const auto high_rad = to_host(high.rad_E_new);
  const auto [hlo_it, hhi_it] =
      std::minmax_element(high_rad.begin(), high_rad.end());
  CHECK(*hlo_it >= *lo_it - 1.0e-12);
  CHECK(*hhi_it <= *hi_it + 1.0e-12);
}

TEST_CASE("ALE1D high-order remap builds cosine phi table",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 16;
  auto cfg = make_cfg(n, 1, 1);
  cfg.numerics.ale1d.remap.high_order_enabled = true;
  cfg.numerics.ale1d.remap.high_order_ramp_cells = 2;
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, 1, 1);
  const auto result = tenryu::hydro::ale1d::remap_v3(
      state, cfg, old_r, default_mask(n), {8}, scratch);
  REQUIRE(result.success);
  const auto phi = to_host(scratch.phi_face);
  CHECK(phi[8] == Catch::Approx(0.0));
  CHECK(phi[7] == Catch::Approx(0.5));
  CHECK(phi[6] == Catch::Approx(1.0));
}

TEST_CASE("ALE1D high-order remap records bound-preserving fallback",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 32;
  auto cfg = make_cfg(n, 1, 1);
  cfg.numerics.ale1d.remap.high_order_enabled = true;
  std::vector<double> rho = {
      4.639, 47.167, 17.445, 0.081, 0.031, 0.371, 14.969, 0.028,
      47.371, 0.043, 0.031, 0.114, 0.383, 85.918, 32.771, 7.165,
      0.092, 0.055, 0.4,   1.7,   0.13,  3.0,   0.2,   9.0,
      0.7,   0.04,  5.0,   0.09,  2.0,   0.3,   1.1,   0.8};
  std::vector<double> ee = {
      0.069, 55.067, 0.012, 0.012, 77.5,  0.077, 0.096, 89.955,
      9.571, 1.599,  0.37,  0.466, 0.055, 0.158, 2.164, 1.087,
      77.789, 0.204, 0.5,   2.0,   0.08,  7.0,   0.12,  20.0,
      0.4,    0.07,  12.0,  0.2,   4.0,   0.3,   1.5,   0.9};
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const std::vector<double> old_r = to_host(state.x_r);
  const std::vector<double> candidate = shifted_nodes(old_r, 0.25);

  tenryu::hydro::ale1d::Ale1dRemapScratch scratch;
  scratch.resize(n, 1, 1);
  const auto result = tenryu::hydro::ale1d::remap_v3(
      state, cfg, candidate, default_mask(n), {}, scratch);
  REQUIRE(result.success);
  CHECK(result.n_bound_fallback_cells > 0);
  const auto fallback = to_host(scratch.fallback_flags);
  CHECK(std::any_of(fallback.begin(), fallback.end(), [](const int flag) {
    return flag != 0;
  }));

  const auto ee_new = to_host(scratch.ee_new);
  for (int i = 0; i < n; ++i) {
    const int im = std::max(0, i - 1);
    const int ip = std::min(n - 1, i + 1);
    const double e_min =
        std::min({ee[static_cast<std::size_t>(im)],
                  ee[static_cast<std::size_t>(i)],
                  ee[static_cast<std::size_t>(ip)]});
    const double e_max =
        std::max({ee[static_cast<std::size_t>(im)],
                  ee[static_cast<std::size_t>(i)],
                  ee[static_cast<std::size_t>(ip)]});
    CHECK(ee_new[static_cast<std::size_t>(i)] >= e_min - 1.0e-10);
    CHECK(ee_new[static_cast<std::size_t>(i)] <= e_max + 1.0e-10);
  }
}

TEST_CASE("ALE1D mass-consistent specific fallback stays within bounds",
          "[hydro][ale1d][remap]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 16;
  constexpr int moving_face = 8;
  constexpr int receiving_cell = moving_face - 1;
  constexpr int donor_cell = moving_face;
  auto cfg = make_cfg(n, 1, 1);
  cfg.numerics.ale1d.remap.high_order_enabled = true;

  auto mesh_state = tenryu::core::State::allocate(cfg);
  mesh_state.mesh = tenryu::mesh::create_mesh(cfg, mesh_state);
  const std::vector<double> old_r = to_host(mesh_state.x_r);
  const double receiving_volume =
      volume_coordinate(old_r[static_cast<std::size_t>(moving_face)]) -
      volume_coordinate(old_r[static_cast<std::size_t>(receiving_cell)]);
  const double donor_volume =
      volume_coordinate(old_r[static_cast<std::size_t>(moving_face + 1)]) -
      volume_coordinate(old_r[static_cast<std::size_t>(moving_face)]);

  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);
  rho[static_cast<std::size_t>(donor_cell)] =
      5.0 * receiving_volume / donor_volume;
  rho[static_cast<std::size_t>(donor_cell + 1)] =
      5.0 * rho[static_cast<std::size_t>(donor_cell)];
  ee[static_cast<std::size_t>(receiving_cell)] = 1.0;
  ee[static_cast<std::size_t>(donor_cell)] = 3.0;
  ee[static_cast<std::size_t>(donor_cell + 1)] =
      ee[static_cast<std::size_t>(donor_cell)] *
      rho[static_cast<std::size_t>(donor_cell)] /
      rho[static_cast<std::size_t>(donor_cell + 1)];
  auto state = make_state(cfg, rho, ee, ei, rad, vf);
  const auto mass_old = to_host(state.mass);
  REQUIRE(mass_old[static_cast<std::size_t>(donor_cell)] /
              mass_old[static_cast<std::size_t>(receiving_cell)] ==
          Catch::Approx(5.0).epsilon(1.0e-14));

  std::vector<double> candidate = old_r;
  const double dr = old_r[1] - old_r[0];
  candidate[static_cast<std::size_t>(moving_face)] += 0.5 * dr;

  tenryu::hydro::ale1d::Ale1dRemapScratch first;
  first.resize(n, 1, 1);
  const auto first_result = tenryu::hydro::ale1d::remap_v3(
      state, cfg, candidate, default_mask(n), {}, first);
  REQUIRE(first_result.success);
  const auto first_fallback = to_host(first.fallback_flags);
  REQUIRE((first_fallback[static_cast<std::size_t>(receiving_cell)] &
           (1 << 1)) != 0);

  const auto mass_first = to_host(first.mass_new);
  const auto ee_first = to_host(first.ee_new);
  const double sweep_volume =
      volume_coordinate(candidate[static_cast<std::size_t>(moving_face)]) -
      volume_coordinate(old_r[static_cast<std::size_t>(moving_face)]);
  const double accepted_mass_flux =
      mass_first[static_cast<std::size_t>(receiving_cell)] -
      mass_old[static_cast<std::size_t>(receiving_cell)];
  const double old_density_basis_specific =
      (mass_old[static_cast<std::size_t>(receiving_cell)] *
           ee[static_cast<std::size_t>(receiving_cell)] +
       sweep_volume * rho[static_cast<std::size_t>(donor_cell)] *
           ee[static_cast<std::size_t>(donor_cell)]) /
      mass_first[static_cast<std::size_t>(receiving_cell)];
  REQUIRE(old_density_basis_specific >
          ee[static_cast<std::size_t>(donor_cell)] * (1.0 + 1.0e-12));

  const double mass_consistent_specific =
      (mass_old[static_cast<std::size_t>(receiving_cell)] *
           ee[static_cast<std::size_t>(receiving_cell)] +
       ee[static_cast<std::size_t>(donor_cell)] * accepted_mass_flux) /
      mass_first[static_cast<std::size_t>(receiving_cell)];
  CHECK(std::abs(ee_first[static_cast<std::size_t>(receiving_cell)] -
                 mass_consistent_specific) /
            std::abs(mass_consistent_specific) <=
        1.0e-12);

  for (int i = 0; i < n; ++i) {
    const int im = std::max(0, i - 1);
    const int ip = std::min(n - 1, i + 1);
    const double e_min =
        std::min({ee[static_cast<std::size_t>(im)],
                  ee[static_cast<std::size_t>(i)],
                  ee[static_cast<std::size_t>(ip)]});
    const double e_max =
        std::max({ee[static_cast<std::size_t>(im)],
                  ee[static_cast<std::size_t>(i)],
                  ee[static_cast<std::size_t>(ip)]});
    const double tol =
        1.0e-12 * std::max(std::abs(e_min), std::abs(e_max));
    CHECK(ee_first[static_cast<std::size_t>(i)] >= e_min - tol);
    CHECK(ee_first[static_cast<std::size_t>(i)] <= e_max + tol);
  }

  tenryu::hydro::ale1d::Ale1dRemapScratch second;
  second.resize(n, 1, 1);
  const auto second_result = tenryu::hydro::ale1d::remap_v3(
      state, cfg, candidate, default_mask(n), {}, second);
  REQUIRE(second_result.success);
  CHECK(first_result.n_bound_fallback_fields ==
        second_result.n_bound_fallback_fields);
  CHECK(first_result.n_bound_fallback_cells ==
        second_result.n_bound_fallback_cells);
  CHECK(bitwise_equal(mass_first, to_host(second.mass_new)));
  CHECK(bitwise_equal(ee_first, to_host(second.ee_new)));
  CHECK(bitwise_equal(to_host(first.ei_new), to_host(second.ei_new)));
  CHECK(bitwise_equal(to_host(first.rad_E_new), to_host(second.rad_E_new)));
  CHECK(bitwise_equal(to_host(first.volFrac_new),
                      to_host(second.volFrac_new)));
  CHECK(bitwise_equal(to_host(first.vol_new), to_host(second.vol_new)));
  CHECK(bitwise_equal(first_fallback, to_host(second.fallback_flags)));
}
