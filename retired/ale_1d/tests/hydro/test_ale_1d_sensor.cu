#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_sensor.cuh"
#include "mesh/mesh.hpp"

namespace {

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

double shell_centroid(const double r0, const double r1) {
  const double r0_3 = r0 * r0 * r0;
  const double r1_3 = r1 * r1 * r1;
  if (r1_3 - r0_3 > 0.0) {
    const double r0_4 = r0_3 * r0;
    const double r1_4 = r1_3 * r1;
    return 0.75 * (r1_4 - r0_4) / (r1_3 - r0_3);
  }
  return 0.5 * (r0 + r1);
}

tenryu::core::Config make_cfg(const int n_cells, const int n_mat = 1) {
  tenryu::core::Config cfg;
  cfg.main.dim = 1;
  cfg.main.dimension = "1D_SPH";
  cfg.mesh.nr = n_cells;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = 1.0;
  cfg.mesh.grid_type_r = "uniform";
  cfg.radiation.groups = 1;

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

std::vector<double> to_host(const tenryu::core::CellField1D& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

std::vector<double> nodes_to_host(const tenryu::core::NodeField1D& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

std::vector<double> cell_centroids(const tenryu::core::State& state) {
  const std::vector<double> x_r = nodes_to_host(state.x_r);
  std::vector<double> r(state.rho.size(), 0.0);
  for (std::size_t i = 0; i < r.size(); ++i) {
    r[i] = shell_centroid(x_r[i], x_r[i + 1]);
  }
  return r;
}

tenryu::core::State make_state(const tenryu::core::Config& cfg) {
  tenryu::core::State state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;

  const int n = cfg.mesh.nr;
  state.cs.reset(static_cast<std::size_t>(n));

  const std::vector<double> vol = to_host(state.vol);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
  std::vector<double> Te(static_cast<std::size_t>(n), 1.0e-3);
  std::vector<double> Ti(static_cast<std::size_t>(n), 1.0e-3);
  std::vector<double> Pe(static_cast<std::size_t>(n), 1.0);
  std::vector<double> Pi(static_cast<std::size_t>(n), 0.0);
  std::vector<double> Qvisc(static_cast<std::size_t>(n), 0.0);
  std::vector<double> cs(static_cast<std::size_t>(n), 1.0e5);
  std::vector<double> laser_dep(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    mass[static_cast<std::size_t>(i)] = rho[static_cast<std::size_t>(i)] *
                                        vol[static_cast<std::size_t>(i)];
  }

  state.rho.copy_from_host(rho);
  state.mass.copy_from_host(mass);
  state.Te.copy_from_host(Te);
  state.Ti.copy_from_host(Ti);
  state.Pe.copy_from_host(Pe);
  state.Pi.copy_from_host(Pi);
  state.Qvisc.copy_from_host(Qvisc);
  state.cs.copy_from_host(cs);
  state.laser_dep.copy_from_host(laser_dep);

  std::vector<double> v_r(static_cast<std::size_t>(n + 1), 0.0);
  state.v_r.copy_from_host(v_r);

  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  std::vector<double> volfrac(static_cast<std::size_t>(n * n_mat), 0.0);
  for (int i = 0; i < n; ++i) {
    volfrac[static_cast<std::size_t>(i * n_mat)] = 1.0;
  }
  state.volFrac.copy_from_host(volfrac);
  return state;
}

const tenryu::hydro::ale1d::Ale1dFeature* find_feature(
    const std::vector<tenryu::hydro::ale1d::Ale1dFeature>& features,
    const tenryu::hydro::ale1d::FeatureKind kind) {
  const auto it = std::find_if(features.begin(), features.end(),
                               [kind](const auto& feature) {
                                 return feature.kind == kind;
                               });
  return it == features.end() ? nullptr : &(*it);
}

int nearest_cell(const std::vector<double>& r, const double target) {
  return static_cast<int>(std::min_element(
             r.begin(), r.end(), [target](const double a, const double b) {
               return std::abs(a - target) < std::abs(b - target);
             }) -
         r.begin());
}

}  // namespace

TEST_CASE("ALE1D sensors emit only center for cold uniform fields",
          "[hydro][ale1d][sensor]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  tenryu::core::Config cfg = make_cfg(64);
  tenryu::core::State state = make_state(cfg);

  const auto features =
      tenryu::hydro::ale1d::compute_features(state, cfg, 1.0e-9);

  REQUIRE(features.size() == 1);
  REQUIRE(features.front().kind ==
          tenryu::hydro::ale1d::FeatureKind::CenterHotspot);
  CHECK(features.front().confidence == Catch::Approx(1.0));
}

TEST_CASE("ALE1D laser sensor finds synthetic Gaussian deposition peak",
          "[hydro][ale1d][sensor]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 128;
  constexpr double dt = 1.0e-9;
  constexpr double r_peak = 0.62;
  constexpr double sigma = 0.035;
  tenryu::core::Config cfg = make_cfg(n);
  tenryu::core::State state = make_state(cfg);
  const std::vector<double> r = cell_centroids(state);
  const std::vector<double> vol = to_host(state.vol);

  std::vector<double> laser_dep(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    const double x = (r[static_cast<std::size_t>(i)] - r_peak) / sigma;
    const double power = std::exp(-x * x);
    laser_dep[static_cast<std::size_t>(i)] =
        power * vol[static_cast<std::size_t>(i)] * dt;
  }
  state.laser_dep.copy_from_host(laser_dep);

  const auto features =
      tenryu::hydro::ale1d::compute_features(state, cfg, dt);
  const auto* laser = find_feature(
      features, tenryu::hydro::ale1d::FeatureKind::LaserAbsorption);
  REQUIRE(laser != nullptr);
  CHECK(std::abs(laser->peak_cell_or_face - nearest_cell(r, r_peak)) <= 2);
}

TEST_CASE("ALE1D ablation sensor finds synthetic tanh Te front",
          "[hydro][ale1d][sensor]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 128;
  constexpr double r_abl = 0.45;
  constexpr double width = 0.02;
  tenryu::core::Config cfg = make_cfg(n);
  tenryu::core::State state = make_state(cfg);
  const std::vector<double> r = cell_centroids(state);

  std::vector<double> Te(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    Te[static_cast<std::size_t>(i)] =
        50.0 + 200.0 * std::tanh((r[static_cast<std::size_t>(i)] - r_abl) / width);
  }
  state.Te.copy_from_host(Te);

  const auto features =
      tenryu::hydro::ale1d::compute_features(state, cfg, 1.0e-9);
  const auto* ablation = find_feature(
      features, tenryu::hydro::ale1d::FeatureKind::AblationFront);
  REQUIRE(ablation != nullptr);
  CHECK(std::abs(ablation->peak_cell_or_face - nearest_cell(r, r_abl)) <= 2);
}

TEST_CASE("ALE1D shock sensor finds synthetic Qvisc compression peak",
          "[hydro][ale1d][sensor]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 128;
  constexpr int i_shock = 73;
  tenryu::core::Config cfg = make_cfg(n);
  tenryu::core::State state = make_state(cfg);

  std::vector<double> Qvisc(static_cast<std::size_t>(n), 0.0);
  std::vector<double> Pe(static_cast<std::size_t>(n), 1.0);
  for (int i = 0; i < n; ++i) {
    const double x = (static_cast<double>(i - i_shock)) / 2.0;
    Qvisc[static_cast<std::size_t>(i)] =
        std::exp(-x * x) * Pe[static_cast<std::size_t>(i)];
  }
  state.Pe.copy_from_host(Pe);
  state.Qvisc.copy_from_host(Qvisc);

  std::vector<double> v_r(static_cast<std::size_t>(n + 1), 0.0);
  v_r[static_cast<std::size_t>(i_shock + 1)] = -2.0e5;
  state.v_r.copy_from_host(v_r);

  const auto features =
      tenryu::hydro::ale1d::compute_features(state, cfg, 1.0e-9);
  const auto* shock =
      find_feature(features, tenryu::hydro::ale1d::FeatureKind::Shock);
  REQUIRE(shock != nullptr);
  CHECK(std::abs(shock->peak_cell_or_face - i_shock) <= 1);
}

TEST_CASE("ALE1D material interface sensor emits full jump face",
          "[hydro][ale1d][sensor]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 96;
  constexpr int face = 37;
  tenryu::core::Config cfg = make_cfg(n, 2);
  tenryu::core::State state = make_state(cfg);

  std::vector<double> volfrac(static_cast<std::size_t>(n * 2), 0.0);
  for (int i = 0; i < n; ++i) {
    const bool left = i < face;
    volfrac[static_cast<std::size_t>(2 * i)] = left ? 1.0 : 0.0;
    volfrac[static_cast<std::size_t>(2 * i + 1)] = left ? 0.0 : 1.0;
  }
  state.volFrac.copy_from_host(volfrac);

  const auto features =
      tenryu::hydro::ale1d::compute_features(state, cfg, 1.0e-9);
  const auto* iface = find_feature(
      features, tenryu::hydro::ale1d::FeatureKind::MaterialInterface);
  REQUIRE(iface != nullptr);
  CHECK(iface->peak_cell_or_face == face);
  CHECK(iface->confidence == Catch::Approx(1.0));
  CHECK(iface->pinned_face);
}

TEST_CASE("ALE1D center sensor always emits on cold fields",
          "[hydro][ale1d][sensor]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  tenryu::core::Config cfg = make_cfg(32);
  tenryu::core::State state = make_state(cfg);

  const auto features =
      tenryu::hydro::ale1d::compute_features(state, cfg, 1.0e-9);
  const auto* center = find_feature(
      features, tenryu::hydro::ale1d::FeatureKind::CenterHotspot);
  REQUIRE(center != nullptr);
  CHECK(center->x_center == Catch::Approx(0.0));
  CHECK(center->r_center == Catch::Approx(0.0));
  CHECK(center->confidence == Catch::Approx(1.0));
}

TEST_CASE("ALE1D sensor confidence uses smooth transitions",
          "[hydro][ale1d][sensor]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 96;
  constexpr int i_shock = 40;
  tenryu::core::Config cfg = make_cfg(n);
  cfg.numerics.ale1d.laser_sensor.enabled = false;
  cfg.numerics.ale1d.ablation_sensor.enabled = false;
  cfg.numerics.ale1d.interface_sensor.enabled = false;
  cfg.numerics.ale1d.center_sensor.enabled = false;
  tenryu::core::State state = make_state(cfg);

  std::vector<double> Qvisc(static_cast<std::size_t>(n), 0.0);
  Qvisc[static_cast<std::size_t>(i_shock)] = 0.06;
  state.Qvisc.copy_from_host(Qvisc);

  std::vector<double> v_r(static_cast<std::size_t>(n + 1), 0.0);
  v_r[static_cast<std::size_t>(i_shock + 1)] = -0.09e5;
  state.v_r.copy_from_host(v_r);

  const auto features =
      tenryu::hydro::ale1d::compute_features(state, cfg, 1.0e-9);
  const auto* shock =
      find_feature(features, tenryu::hydro::ale1d::FeatureKind::Shock);
  REQUIRE(shock != nullptr);
  CHECK(shock->confidence > 0.0);
  CHECK(shock->confidence < 1.0);
}
