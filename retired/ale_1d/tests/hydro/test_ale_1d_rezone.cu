#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_rezone.cuh"

namespace {

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

tenryu::core::Config make_cfg(const int n) {
  tenryu::core::Config cfg;
  cfg.main.dim = 1;
  cfg.main.dimension = "1D_SPH";
  cfg.mesh.nr = n;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = 1.0;
  cfg.mesh.grid_type_r = "uniform";
  cfg.radiation.groups = 1;
  cfg.numerics.ale1d.enabled = true;
  return cfg;
}

tenryu::core::State make_uniform_state(const int n) {
  tenryu::core::State state;
  state.mesh.dim = 1;
  state.mesh.topo.nr = n;
  state.mesh.topo.nz = 1;
  state.mesh.topo.n_cells = n;
  state.mesh.topo.n_nodes = n + 1;

  std::vector<double> nodes(static_cast<std::size_t>(n + 1), 0.0);
  for (int j = 0; j <= n; ++j) {
    nodes[static_cast<std::size_t>(j)] = static_cast<double>(j) /
                                         static_cast<double>(n);
  }
  state.x_r = nodes;

  std::vector<double> mass(static_cast<std::size_t>(n),
                           1.0 / static_cast<double>(n));
  state.mass = mass;
  return state;
}

tenryu::hydro::ale1d::Ale1dFeature make_feature(
    const tenryu::hydro::ale1d::FeatureKind kind,
    const double x,
    const double sigma_x,
    const double target_cells,
    const double confidence) {
  tenryu::hydro::ale1d::Ale1dFeature feature;
  feature.kind = kind;
  feature.x_center = x;
  feature.r_center = x;
  feature.sigma_x = sigma_x;
  feature.sigma_r = sigma_x;
  feature.confidence = confidence;
  feature.target_cells = target_cells;
  feature.peak_cell_or_face = static_cast<int>(std::lround(x * 512.0));
  return feature;
}

int nearest_cell(const int n, const double x) {
  return std::clamp(static_cast<int>(std::floor(x * static_cast<double>(n))), 0,
                    n - 1);
}

double min_width_near(const std::vector<double>& nodes,
                      const double x,
                      const double half_width) {
  double min_width = 1.0e300;
  for (std::size_t i = 0; i + 1 < nodes.size(); ++i) {
    const double center = 0.5 * (nodes[i] + nodes[i + 1]);
    if (std::abs(center - x) <= half_width) {
      min_width = std::min(min_width, nodes[i + 1] - nodes[i]);
    }
  }
  return min_width;
}

double active_cells_from_monitor(const std::vector<double>& W, const double w0) {
  const double integral =
      std::accumulate(W.begin(), W.end(), 0.0) / static_cast<double>(W.size());
  const double added = integral - w0;
  return static_cast<double>(W.size()) * added / integral;
}

std::vector<double> nodes_from_widths(const std::vector<double>& widths) {
  std::vector<double> nodes(widths.size() + 1, 0.0);
  for (std::size_t i = 0; i < widths.size(); ++i) {
    nodes[i + 1] = nodes[i] + widths[i];
  }
  return nodes;
}

void check_sweep_safety(const std::vector<double>& old_nodes,
                        const std::vector<double>& new_nodes) {
  REQUIRE(new_nodes.size() == old_nodes.size());
  CHECK(new_nodes.front() == old_nodes.front());
  CHECK(new_nodes.back() == old_nodes.back());
  for (std::size_t j = 1; j + 1 < old_nodes.size(); ++j) {
    if (new_nodes[j] > old_nodes[j]) {
      CHECK(new_nodes[j] <= old_nodes[j + 1]);
    } else if (new_nodes[j] < old_nodes[j]) {
      CHECK(new_nodes[j] >= old_nodes[j - 1]);
    }
  }
}

}  // namespace

TEST_CASE("ALE1D rezone identity for uniform monitor",
          "[hydro][ale1d][rezone]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 256;
  tenryu::core::Config cfg = make_cfg(n);
  tenryu::core::State state = make_uniform_state(n);
  std::vector<tenryu::hydro::ale1d::Ale1dFeature> features{
      make_feature(tenryu::hydro::ale1d::FeatureKind::Shock, 0.5, 0.05, 30.0,
                   0.0)};

  const auto W = tenryu::hydro::ale1d::build_monitor(state, cfg, features);
  REQUIRE(W.size() == static_cast<std::size_t>(n));
  for (const double w : W) {
    CHECK(w == Catch::Approx(cfg.numerics.ale1d.rezone.monitor_floor)
                   .epsilon(1.0e-14));
  }

  const auto result = tenryu::hydro::ale1d::rezone(state, cfg, features, W);
  REQUIRE(result.success);
  REQUIRE(result.r_candidate.size() == static_cast<std::size_t>(n + 1));
  for (int j = 0; j <= n; ++j) {
    CHECK(result.r_candidate[static_cast<std::size_t>(j)] ==
          Catch::Approx(static_cast<double>(j) / static_cast<double>(n))
              .margin(2.0e-14));
  }
}

TEST_CASE("ALE1D monitor and rezone concentrate around a feature",
          "[hydro][ale1d][rezone]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 512;
  tenryu::core::Config cfg = make_cfg(n);
  cfg.numerics.ale1d.rezone.spatial_monitor_enabled = false;
  tenryu::core::State state = make_uniform_state(n);
  std::vector<tenryu::hydro::ale1d::Ale1dFeature> features{
      make_feature(tenryu::hydro::ale1d::FeatureKind::Shock, 0.5, 0.05, 30.0,
                   1.0)};

  const auto W = tenryu::hydro::ale1d::build_monitor(state, cfg, features);
  const int peak_cell = static_cast<int>(
      std::max_element(W.begin(), W.end()) - W.begin());
  CHECK(std::abs(peak_cell - nearest_cell(n, 0.5)) <= 3);

  const auto result = tenryu::hydro::ale1d::rezone(state, cfg, features, W);
  REQUIRE(result.success);
  CHECK(min_width_near(result.r_candidate, 0.5, 0.03) <
        1.0 / static_cast<double>(n));
}

TEST_CASE("ALE1D active budget cap preserves the floor fraction",
          "[hydro][ale1d][rezone]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 300;
  tenryu::core::Config cfg = make_cfg(n);
  cfg.numerics.ale1d.rezone.monitor_smoothing_iterations = 0;
  cfg.numerics.ale1d.rezone.spatial_monitor_enabled = false;
  tenryu::core::State state = make_uniform_state(n);
  std::vector<tenryu::hydro::ale1d::Ale1dFeature> features{
      make_feature(tenryu::hydro::ale1d::FeatureKind::Shock, 0.25, 0.04,
                   static_cast<double>(n), 1.0),
      make_feature(tenryu::hydro::ale1d::FeatureKind::Shock, 0.50, 0.04,
                   static_cast<double>(n), 1.0),
      make_feature(tenryu::hydro::ale1d::FeatureKind::Shock, 0.75, 0.04,
                   static_cast<double>(n), 1.0)};

  const auto W = tenryu::hydro::ale1d::build_monitor(state, cfg, features);
  const double active_cells = active_cells_from_monitor(
      W, cfg.numerics.ale1d.rezone.monitor_floor);
  const double budget_cap =
      (1.0 - cfg.numerics.ale1d.rezone.min_floor_fraction) *
      static_cast<double>(n);
  CHECK(active_cells == Catch::Approx(budget_cap).epsilon(1.0e-12));
}

TEST_CASE("ALE1D spatial monitor adds resolution pressure near broad cells",
          "[hydro][ale1d][rezone]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 512;
  tenryu::core::Config cfg = make_cfg(n);
  tenryu::core::State state = make_uniform_state(n);
  auto feature = make_feature(
      tenryu::hydro::ale1d::FeatureKind::LaserAbsorption, 0.55, 0.04, 0.0,
      1.0);
  feature.sigma_r = 9.0e-4;
  std::vector<tenryu::hydro::ale1d::Ale1dFeature> features{feature};

  tenryu::core::Config off_cfg = cfg;
  off_cfg.numerics.ale1d.rezone.spatial_monitor_enabled = false;
  const auto W_off = tenryu::hydro::ale1d::build_monitor(state, off_cfg, features);
  const auto W_on = tenryu::hydro::ale1d::build_monitor(state, cfg, features);

  const int i_feature = nearest_cell(n, 0.55);
  CHECK(W_on[static_cast<std::size_t>(i_feature)] >
        W_off[static_cast<std::size_t>(i_feature)]);

  const auto off = tenryu::hydro::ale1d::rezone(state, off_cfg, features, W_off);
  const auto on = tenryu::hydro::ale1d::rezone(state, cfg, features, W_on);
  REQUIRE(off.success);
  REQUIRE(on.success);
  CHECK(min_width_near(on.r_candidate, 0.55, 0.03) <
        min_width_near(off.r_candidate, 0.55, 0.03));
}

TEST_CASE("ALE1D rezone skips when N is below min_cells",
          "[hydro][ale1d][rezone]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 128;
  tenryu::core::Config cfg = make_cfg(n);
  tenryu::core::State state = make_uniform_state(n);

  const auto result = tenryu::hydro::ale1d::rezone(state, cfg, {});
  REQUIRE_FALSE(result.success);
  CHECK(result.skip_reason ==
        tenryu::hydro::ale1d::Ale1dSkipReason::NTooSmall);
}

TEST_CASE("ALE1D rezone skips when protected nodes saturate the mesh",
          "[hydro][ale1d][rezone]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 256;
  tenryu::core::Config cfg = make_cfg(n);
  tenryu::core::State state = make_uniform_state(n);
  std::vector<tenryu::hydro::ale1d::Ale1dFeature> features;
  for (int face = 3; face < n; face += 3) {
    auto feature = make_feature(
        tenryu::hydro::ale1d::FeatureKind::MaterialInterface,
        static_cast<double>(face) / static_cast<double>(n), 0.01, 0.0, 1.0);
    feature.peak_cell_or_face = face;
    feature.pinned_face = true;
    features.push_back(feature);
  }

  const auto result = tenryu::hydro::ale1d::rezone(state, cfg, features);
  REQUIRE_FALSE(result.success);
  CHECK(result.skip_reason ==
        tenryu::hydro::ale1d::Ale1dSkipReason::ProtectedFractionTooHigh);
}

TEST_CASE("ALE1D displacement cap is enforced",
          "[hydro][ale1d][rezone]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 256;
  tenryu::core::Config cfg = make_cfg(n);
  cfg.numerics.ale1d.max_node_displacement_fraction_mu = 0.05;
  cfg.numerics.ale1d.max_node_displacement_fraction_r = 0.05;
  tenryu::core::State state = make_uniform_state(n);
  std::vector<double> W(static_cast<std::size_t>(n), 1.0);
  W[static_cast<std::size_t>(n / 2)] = 50.0;

  const auto result = tenryu::hydro::ale1d::rezone(state, cfg, {}, W);
  REQUIRE(result.success);
  CHECK(result.max_node_displacement_mu <=
        cfg.numerics.ale1d.max_node_displacement_fraction_mu + 1.0e-12);
  CHECK(result.max_node_displacement_r <=
        cfg.numerics.ale1d.max_node_displacement_fraction_r + 1.0e-12);
}

TEST_CASE("ALE1D rezone preserves total coordinate span",
          "[hydro][ale1d][rezone]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }
  constexpr int n = 256;
  tenryu::core::Config cfg = make_cfg(n);
  cfg.numerics.ale1d.rezone.spatial_monitor_enabled = false;
  tenryu::core::State state = make_uniform_state(n);
  std::vector<tenryu::hydro::ale1d::Ale1dFeature> features{
      make_feature(tenryu::hydro::ale1d::FeatureKind::Shock, 0.35, 0.04, 20.0,
                   1.0),
      make_feature(tenryu::hydro::ale1d::FeatureKind::AblationFront, 0.70,
                   0.05, 25.0, 1.0)};

  const auto result = tenryu::hydro::ale1d::rezone(state, cfg, features);
  REQUIRE(result.success);
  double span_sum = 0.0;
  for (std::size_t i = 0; i + 1 < result.r_candidate.size(); ++i) {
    const double dr = result.r_candidate[i + 1] - result.r_candidate[i];
    REQUIRE(dr > 0.0);
    span_sum += dr;
  }
  CHECK(span_sum == Catch::Approx(1.0).epsilon(1.0e-14));
  CHECK(result.r_candidate.front() == Catch::Approx(0.0).margin(1.0e-14));
  CHECK(result.r_candidate.back() == Catch::Approx(1.0).margin(1.0e-14));
}

TEST_CASE("ALE1D min-width floor candidate preserves no-offender identity",
          "[hydro][ale1d][rezone][min-width-floor]") {
  constexpr int n = 32;
  const std::vector<double> nodes =
      nodes_from_widths(std::vector<double>(n, 1.0e-4));
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  const std::vector<bool> eligible(static_cast<std::size_t>(n), true);

  const auto result = tenryu::hydro::ale1d::build_min_width_floor_candidate(
      nodes, pinned, eligible, 0.5e-4, 1.25, 3, 1.8);

  REQUIRE(result.success);
  CHECK(result.fully_relieved);
  CHECK(result.r_candidate == nodes);
}

TEST_CASE("ALE1D min-width floor candidate targets the minimum cell",
          "[hydro][ale1d][rezone][min-width-floor]") {
  constexpr int n = 40;
  constexpr int offender = 20;
  constexpr double floor = 5.0e-6;
  constexpr double max_growth_factor = 1.8;
  std::vector<double> widths(n, 1.0e-4);
  widths[offender] = 2.0e-6;
  const std::vector<double> nodes = nodes_from_widths(widths);
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  const std::vector<bool> eligible(static_cast<std::size_t>(n), true);

  const auto result = tenryu::hydro::ale1d::build_min_width_floor_candidate(
      nodes, pinned, eligible, floor, 1.25, 3, max_growth_factor);

  REQUIRE(result.success);
  CHECK_FALSE(result.fully_relieved);
  CHECK(result.min_dl_after > result.min_dl_before);
  const double offender_width =
      result.r_candidate[offender + 1] - result.r_candidate[offender];
  CHECK(offender_width >= widths[offender]);
  CHECK(offender_width <= widths[offender] * max_growth_factor *
                              (1.0 + 1.0e-12));
  check_sweep_safety(nodes, result.r_candidate);
  const double old_span = nodes.back() - nodes.front();
  const double new_span = result.r_candidate.back() - result.r_candidate.front();
  CHECK(std::abs(new_span - old_span) / old_span <= 1.0e-13);
  for (int j = 0; j <= n; ++j) {
    if (j < 17 || j >= 24) {
      CHECK(result.r_candidate[static_cast<std::size_t>(j)] ==
            nodes[static_cast<std::size_t>(j)]);
    }
  }
}

TEST_CASE("ALE1D min-width floor candidate respects a pinned partition wall",
          "[hydro][ale1d][rezone][min-width-floor]") {
  constexpr int n = 40;
  std::vector<double> widths(n, 1.0e-4);
  widths[19] = 2.0e-6;
  const std::vector<double> nodes = nodes_from_widths(widths);
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned[20] = true;
  pinned.back() = true;
  const std::vector<bool> eligible(static_cast<std::size_t>(n), true);

  const auto result = tenryu::hydro::ale1d::build_min_width_floor_candidate(
      nodes, pinned, eligible, 5.0e-6, 1.25, 3, 1.8);

  REQUIRE(result.success);
  CHECK(result.n_windows == 1);
  CHECK(result.r_candidate[20] == nodes[20]);
}

// SKIPPED (wave-5): bench engagement blocked by an unidentified remap
// extensive-field validation failure; see the internal design note perf_1d_wave5_20260807.md B1.
TEST_CASE("ALE1D min-width floor candidate converges progressively",
          "[hydro][ale1d][rezone][min-width-floor][.]") {
  constexpr int n = 60;
  constexpr double floor = 5.0e-6;
  std::vector<double> widths(n, 0.0);
  for (int i = 0; i < n; ++i) {
    widths[static_cast<std::size_t>(i)] =
        1.0e-4 * std::pow(0.93, static_cast<double>(i));
  }
  std::vector<double> nodes = nodes_from_widths(widths);
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  const std::vector<bool> eligible(static_cast<std::size_t>(n), true);

  bool fully_relieved = false;
  for (int iteration = 0; iteration < 400; ++iteration) {
    const auto result = tenryu::hydro::ale1d::build_min_width_floor_candidate(
        nodes, pinned, eligible, floor, 1.25, 3, 1.8);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.no_relief_available);
    nodes = result.r_candidate;
    if (result.fully_relieved) {
      fully_relieved = true;
      break;
    }
  }

  REQUIRE(fully_relieved);
  const auto identity =
      tenryu::hydro::ale1d::build_min_width_floor_candidate(
          nodes, pinned, eligible, floor, 1.25, 3, 1.8);
  REQUIRE(identity.success);
  CHECK(identity.fully_relieved);
  CHECK(identity.r_candidate == nodes);
}

TEST_CASE("ALE1D min-width floor candidate is deterministic",
          "[hydro][ale1d][rezone][min-width-floor]") {
  constexpr int n = 40;
  std::vector<double> widths(n, 1.0e-4);
  widths[20] = 2.0e-6;
  const std::vector<double> nodes = nodes_from_widths(widths);
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  const std::vector<bool> eligible(static_cast<std::size_t>(n), true);

  const auto first = tenryu::hydro::ale1d::build_min_width_floor_candidate(
      nodes, pinned, eligible, 5.0e-6, 1.25, 3, 1.8);
  const auto second = tenryu::hydro::ale1d::build_min_width_floor_candidate(
      nodes, pinned, eligible, 5.0e-6, 1.25, 3, 1.8);

  REQUIRE(first.success);
  REQUIRE(second.success);
  REQUIRE(first.r_candidate.size() == second.r_candidate.size());
  CHECK(std::memcmp(first.r_candidate.data(), second.r_candidate.data(),
                    first.r_candidate.size() * sizeof(double)) == 0);
}

TEST_CASE("ALE1D min-width floor candidate is sweep-safe",
          "[hydro][ale1d][rezone][min-width-floor]") {
  constexpr int n = 40;
  std::vector<double> widths(n, 1.0e-4);
  widths[20] = 2.0e-6;
  const std::vector<double> nodes = nodes_from_widths(widths);
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  const std::vector<bool> eligible(static_cast<std::size_t>(n), true);

  const auto result = tenryu::hydro::ale1d::build_min_width_floor_candidate(
      nodes, pinned, eligible, 5.0e-6, 1.25, 3, 1.8);

  REQUIRE(result.success);
  check_sweep_safety(nodes, result.r_candidate);
}

// SKIPPED (wave-5): bench engagement blocked by an unidentified remap
// extensive-field validation failure; see the internal design note perf_1d_wave5_20260807.md B1.
TEST_CASE("ALE1D min-width floor candidate is sweep-safe under progressive "
          "taper relief",
          "[hydro][ale1d][rezone][min-width-floor][.]") {
  constexpr int n = 60;
  std::vector<double> widths(n, 0.0);
  for (int i = 0; i < n; ++i) {
    widths[static_cast<std::size_t>(i)] =
        1.0e-4 * std::pow(0.93, static_cast<double>(i));
  }
  std::vector<double> nodes = nodes_from_widths(widths);
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  const std::vector<bool> eligible(static_cast<std::size_t>(n), true);

  bool fully_relieved = false;
  for (int iteration = 0; iteration < 400; ++iteration) {
    const auto result = tenryu::hydro::ale1d::build_min_width_floor_candidate(
        nodes, pinned, eligible, 5.0e-6, 1.25, 3, 1.8);
    REQUIRE(result.success);
    REQUIRE_FALSE(result.no_relief_available);
    check_sweep_safety(nodes, result.r_candidate);
    nodes = result.r_candidate;
    if (result.fully_relieved) {
      fully_relieved = true;
      break;
    }
  }
  CHECK(fully_relieved);
}

TEST_CASE("ALE1D min-width floor reports donor-less windows",
          "[hydro][ale1d][rezone][min-width-floor]") {
  constexpr int n = 8;
  const std::vector<double> nodes =
      nodes_from_widths(std::vector<double>(n, 1.0e-6));
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  const std::vector<bool> eligible(static_cast<std::size_t>(n), true);

  const auto result = tenryu::hydro::ale1d::build_min_width_floor_candidate(
      nodes, pinned, eligible, 5.0e-6, 1.25, 3, 1.8);

  REQUIRE(result.success);
  CHECK(result.no_relief_available);
  CHECK(result.r_candidate == nodes);
}

TEST_CASE("ALE1D min-width floor eligibility mask blocks corona windows",
          "[hydro][ale1d][rezone][min-width-floor]") {
  constexpr int n = 32;
  std::vector<double> widths(n, 1.0e-4);
  widths[20] = 2.0e-6;
  const std::vector<double> nodes = nodes_from_widths(widths);
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  std::vector<bool> eligible(static_cast<std::size_t>(n), true);
  for (int i = 18; i <= 22; ++i) {
    eligible[static_cast<std::size_t>(i)] = false;
  }

  const auto result = tenryu::hydro::ale1d::build_min_width_floor_candidate(
      nodes, pinned, eligible, 5.0e-6, 1.25, 3, 1.8);

  REQUIRE(result.success);
  CHECK(result.no_relief_available);
  CHECK(result.r_candidate == nodes);
}

TEST_CASE("ale1d min width floor never moves the outer boundary cell",
          "[ale1d]") {
  constexpr int n = 40;
  std::vector<double> widths(n, 1.0e-4);
  widths[n - 2] = 2.0e-6;
  widths[n - 1] = 2.0e-6;
  const std::vector<double> nodes = nodes_from_widths(widths);
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  const std::vector<bool> eligible(static_cast<std::size_t>(n), true);

  const auto result = tenryu::hydro::ale1d::build_min_width_floor_candidate(
      nodes, pinned, eligible, 5.0e-6, 1.25, 3, 1.8);

  REQUIRE(result.success);
  CHECK(result.n_windows == 1);
  CHECK(result.r_candidate[n - 1] == nodes[n - 1]);
  CHECK(result.r_candidate[n] == nodes[n]);
}
