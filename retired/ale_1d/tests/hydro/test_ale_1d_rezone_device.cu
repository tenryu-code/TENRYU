// ale1d::rezone_candidate_device and ale1d::floor_candidate_device (the 1D ALE rezone candidate
// built on the device) against the host build the driver used: rezone (with build_monitor) or
// build_min_width_floor_candidate, then the driver's boundary nodes, geometry check and acoustic
// time-step bounds, written below as the driver writes them. Bit for bit on random meshes,
// masses, densities, sound speeds and features, in the three geometries and the boundary types.
// Runs where the host's glibc is the build core::glibc_libm reproduces (glibc >= 2.28, x86-64
// with FMA and AVX2) and a device is present; skips otherwise.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/glibc_libm_host.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_driver.cuh"
#include "hydro/ale_1d_rezone.cuh"
#include "hydro/ale_1d_rezone_device.cuh"

namespace {

namespace ale1d = tenryu::hydro::ale1d;

bool comparable() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0 &&
         tenryu::core::glibc_libm::host_has_reproduced_build();
}

std::uint64_t bits_of(const double v) {
  std::uint64_t u = 0;
  std::memcpy(&u, &v, sizeof(u));
  return u;
}

struct Case {
  tenryu::core::Config cfg;
  tenryu::core::State state;
  std::vector<double> r;
  std::vector<double> mass;
  std::vector<double> rho;
  std::vector<double> cs;
  std::vector<ale1d::Ale1dFeature> features;
  int n = 0;
  int geom = 0;
};

// A random mesh: n cells of random widths (a few much smaller), masses, densities with some cells
// below 100 rho_floor, sound speeds with some zeros, and up to seven features of every kind.
Case make_case(std::mt19937_64& rng, const int n, const int geom, const std::string& boundary,
               const bool spatial, const int smoothing, const double max_disp,
               const bool bad_mass, const int n_features) {
  std::uniform_real_distribution<double> unit(0.0, 1.0);
  Case c;
  c.n = n;
  c.geom = geom;
  auto& cfg = c.cfg;
  cfg.main.dim = 1;
  cfg.main.dimension = "1D_SPH";
  cfg.mesh.nr = n;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = (geom == 2) ? 0.1 : 0.0;
  cfg.numerics.ale1d.enabled = true;
  cfg.numerics.hydro.boundary_1d = boundary;
  cfg.numerics.ale1d.rezone.spatial_monitor_enabled = spatial;
  cfg.numerics.ale1d.rezone.monitor_smoothing_iterations = smoothing;
  cfg.numerics.ale1d.max_node_displacement_fraction_mu = max_disp;
  cfg.numerics.ale1d.max_node_displacement_fraction_r = max_disp;
  cfg.numerics.floors.rho = 1.0e-6;
  c.r.assign(static_cast<std::size_t>(n + 1), cfg.mesh.r_min);
  for (int i = 0; i < n; ++i) {
    double width = 1.0e-3 * (0.5 + unit(rng));
    if (unit(rng) < 0.02) {
      width *= 1.0e-2;  // a crushed cell
    }
    c.r[static_cast<std::size_t>(i + 1)] = c.r[static_cast<std::size_t>(i)] + width;
  }
  cfg.mesh.r_max = c.r.back();
  c.mass.resize(static_cast<std::size_t>(n));
  c.rho.resize(static_cast<std::size_t>(n));
  c.cs.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    c.mass[static_cast<std::size_t>(i)] = std::pow(10.0, -3.0 + 3.0 * unit(rng));
    c.rho[static_cast<std::size_t>(i)] = (unit(rng) < 0.05) ? 1.0e-5 : std::pow(10.0, -2.0 + 3.0 * unit(rng));
    c.cs[static_cast<std::size_t>(i)] = (unit(rng) < 0.05) ? 0.0 : 1.0e5 * (0.5 + unit(rng));
  }
  if (bad_mass) {
    c.mass[static_cast<std::size_t>(n / 3)] = -1.0;  // the uniform mass map
  }
  const ale1d::FeatureKind kinds[] = {
      ale1d::FeatureKind::LaserAbsorption, ale1d::FeatureKind::AblationFront,
      ale1d::FeatureKind::Shock, ale1d::FeatureKind::MaterialInterface,
      ale1d::FeatureKind::CenterHotspot};
  for (int f = 0; f < n_features; ++f) {
    ale1d::Ale1dFeature feature;
    feature.kind = kinds[static_cast<int>(unit(rng) * 5.0) % 5];
    feature.x_center = unit(rng);
    feature.r_center = cfg.mesh.r_min + feature.x_center * (cfg.mesh.r_max - cfg.mesh.r_min);
    feature.sigma_x = 0.005 + 0.1 * unit(rng);
    feature.sigma_r = 1.0e-5 + 3.0e-4 * unit(rng);
    feature.confidence = (unit(rng) < 0.1) ? 0.0 : unit(rng);
    feature.target_cells = (unit(rng) < 0.1) ? 0.0 : 5.0 + 40.0 * unit(rng);
    const double place = unit(rng);
    feature.peak_cell_or_face = (place < 0.1) ? -1 : (place < 0.15 ? n + 7 : static_cast<int>(unit(rng) * n));
    feature.pinned_face = unit(rng) < 0.5;
    c.features.push_back(feature);
  }
  auto& state = c.state;
  state.mesh.dim = 1;
  state.mesh.topo.nr = n;
  state.mesh.topo.nz = 1;
  state.mesh.topo.n_cells = n;
  state.mesh.topo.n_nodes = n + 1;
  state.mesh.geometry_code = geom;
  state.x_r = c.r;
  state.mass = c.mass;
  state.rho = c.rho;
  state.cs = c.cs;
  return c;
}

struct Outcome {
  bool success = false;
  ale1d::Ale1dSkipReason skip_reason = ale1d::Ale1dSkipReason::None;
  bool no_relief = false;
  std::vector<double> r;
  std::vector<std::uint8_t> pinned;
  int n_protected = 0;
  bool geometry_valid = false;
  double dt_current = 0.0;
  double dt_candidate = 0.0;
};

// The driver's host steps after a successful candidate (ale_1d_driver.cu).
void finish_host(Outcome& out, const Case& c) {
  const bool fixed = c.cfg.numerics.hydro.boundary_1d == "fixed" ||
                     c.cfg.numerics.hydro.boundary_1d == "reflect";
  out.r.front() = c.cfg.mesh.r_min;
  out.pinned.front() = 1U;
  if (fixed) {
    out.r.back() = c.cfg.mesh.r_max;
    out.pinned.back() = 1U;
  }
  out.n_protected = static_cast<int>(std::count(out.pinned.begin(), out.pinned.end(), 1U));
  out.geometry_valid = true;
  for (int i = 0; i < c.n; ++i) {
    const double r0 = out.r[static_cast<std::size_t>(i)];
    const double r1 = out.r[static_cast<std::size_t>(i + 1)];
    if (!std::isfinite(r0) || !std::isfinite(r1) || !(r1 > r0)) {
      out.geometry_valid = false;
      break;
    }
    const double dx = r1 - r0;
    const double vol =
        ale1d::ale1d_volume_coordinate(r1, c.geom) - ale1d::ale1d_volume_coordinate(r0, c.geom);
    if (!(dx > 0.0) || !(vol > 0.0) || !std::isfinite(vol)) {
      out.geometry_valid = false;
      break;
    }
  }
  const ale1d::Ale1dAcousticDtBounds bounds = ale1d::acoustic_dt_bounds(c.r, c.cs, out.r);
  out.dt_current = bounds.current;
  out.dt_candidate = bounds.candidate;
}

Outcome host_rezone(const Case& c) {
  const ale1d::Ale1dRezoneResult result = ale1d::rezone(c.state, c.cfg, c.features);
  Outcome out;
  out.success = result.success;
  out.skip_reason = result.skip_reason;
  if (!out.success) {
    return out;
  }
  out.r = result.r_candidate;
  out.pinned.resize(result.node_mask.pinned.size());
  for (std::size_t j = 0; j < out.pinned.size(); ++j) {
    out.pinned[j] = result.node_mask.pinned[j] ? 1U : 0U;
  }
  finish_host(out, c);
  return out;
}

Outcome host_floor(const Case& c) {
  const int n = c.n;
  std::vector<bool> pinned(static_cast<std::size_t>(n + 1), false);
  pinned.front() = true;
  pinned.back() = true;
  for (const ale1d::Ale1dFeature& feature : c.features) {
    const int face = feature.peak_cell_or_face;
    if (feature.pinned_face && face >= 0 && face <= n) {
      pinned[static_cast<std::size_t>(face)] = true;
    }
  }
  std::vector<bool> eligible(static_cast<std::size_t>(n), false);
  for (int i = 0; i < n; ++i) {
    eligible[static_cast<std::size_t>(i)] =
        c.rho[static_cast<std::size_t>(i)] > 100.0 * c.cfg.numerics.floors.rho;
  }
  const auto& floor = c.cfg.numerics.ale1d.min_width_floor;
  const ale1d::MinWidthFloorCandidateResult result = ale1d::build_min_width_floor_candidate(
      c.r, pinned, eligible, floor.floor_cm, floor.target_factor, floor.relief_halfwidth_cells,
      floor.max_growth_factor);
  Outcome out;
  out.success = result.success;
  out.no_relief = result.no_relief_available;
  if (!out.success || out.no_relief) {
    return out;
  }
  out.r = result.r_candidate;
  out.pinned.resize(pinned.size());
  for (std::size_t j = 0; j < pinned.size(); ++j) {
    out.pinned[j] = pinned[j] ? 1U : 0U;
  }
  finish_host(out, c);
  return out;
}

Outcome device_outcome(const ale1d::Ale1dDeviceCandidate& candidate, const int n) {
  Outcome out;
  out.success = candidate.success;
  out.skip_reason = candidate.skip_reason;
  out.no_relief = candidate.no_relief_available;
  out.n_protected = candidate.n_protected_nodes;
  out.geometry_valid = candidate.geometry_valid;
  out.dt_current = candidate.dt_current;
  out.dt_candidate = candidate.dt_candidate;
  out.r.resize(static_cast<std::size_t>(n + 1));
  out.pinned.resize(static_cast<std::size_t>(n + 1));
  REQUIRE(cudaMemcpy(out.r.data(), candidate.r_candidate, out.r.size() * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  REQUIRE(cudaMemcpy(out.pinned.data(), candidate.pinned, out.pinned.size(),
                     cudaMemcpyDeviceToHost) == cudaSuccess);
  return out;
}

// Equal outcomes: the same decision, and for a candidate the same radii, mask and gate values.
bool same(const Outcome& host, const Outcome& dev, const bool floor_path) {
  if (floor_path) {
    if (host.no_relief != dev.no_relief) {
      UNSCOPED_INFO("no_relief host " << host.no_relief << " device " << dev.no_relief);
      return false;
    }
    if (host.no_relief) {
      return true;
    }
  } else {
    if (host.success != dev.success || host.skip_reason != dev.skip_reason) {
      UNSCOPED_INFO("success host " << host.success << " device " << dev.success << " reason "
                                    << ale1d::to_string(host.skip_reason) << " / "
                                    << ale1d::to_string(dev.skip_reason));
      return false;
    }
    if (!host.success) {
      return true;
    }
  }
  for (std::size_t j = 0; j < host.r.size(); ++j) {
    if (bits_of(host.r[j]) != bits_of(dev.r[j])) {
      UNSCOPED_INFO("node " << j << std::hexfloat << " host " << host.r[j] << " device "
                            << dev.r[j] << std::defaultfloat);
      return false;
    }
  }
  if (host.pinned != dev.pinned || host.n_protected != dev.n_protected ||
      host.geometry_valid != dev.geometry_valid) {
    UNSCOPED_INFO("mask or geometry differs");
    return false;
  }
  if (host.geometry_valid &&
      (bits_of(host.dt_current) != bits_of(dev.dt_current) ||
       bits_of(host.dt_candidate) != bits_of(dev.dt_candidate))) {
    UNSCOPED_INFO("dt bounds host " << host.dt_current << " " << host.dt_candidate << " device "
                                    << dev.dt_current << " " << dev.dt_candidate);
    return false;
  }
  return true;
}

}  // namespace

TEST_CASE("ALE1D rezone candidate on the device equals the host build bit for bit",
          "[hydro][ale1d][rezone][device]") {
  if (!comparable()) {
    SKIP("needs a device and glibc >= 2.28 on an x86-64 processor with FMA and AVX2");
  }
  std::mt19937_64 rng(20261002ULL);
  int compared = 0;
  int candidates = 0;
  for (const int geom : {0, 1, 2}) {
    for (const std::string boundary : {"free", "fixed", "reflect"}) {
      for (int variant = 0; variant < 6; ++variant) {
        const int n = (variant % 2 == 0) ? 300 : 517;
        const bool spatial = variant % 3 != 2;
        const int smoothing = variant % 3;
        const double max_disp = (variant < 3) ? 0.35 : 0.05;
        const bool bad_mass = variant == 5;
        const int n_features = (variant == 1) ? 0 : 1 + variant;
        Case c = make_case(rng, n, geom, boundary, spatial, smoothing, max_disp, bad_mass,
                           n_features);
        INFO("geom " << geom << " boundary " << boundary << " variant " << variant);
        const Outcome host = host_rezone(c);
        const Outcome dev =
            device_outcome(ale1d::rezone_candidate_device(c.state, c.cfg, c.features), n);
        CHECK(same(host, dev, false));
        ++compared;
        candidates += host.success ? 1 : 0;
      }
    }
  }
  INFO("cases " << compared << " with a candidate " << candidates);
  CHECK(candidates > compared / 2);
}

TEST_CASE("ALE1D min-width floor candidate on the device equals the host build bit for bit",
          "[hydro][ale1d][rezone][device]") {
  if (!comparable()) {
    SKIP("needs a device and glibc >= 2.28 on an x86-64 processor with FMA and AVX2");
  }
  std::mt19937_64 rng(1002ULL);
  int relieved = 0;
  int compared = 0;
  for (const int geom : {0, 2}) {
    for (const std::string boundary : {"free", "reflect"}) {
      for (int variant = 0; variant < 8; ++variant) {
        const int n = 300 + 37 * variant;
        Case c = make_case(rng, n, geom, boundary, true, 2, 0.35, false, variant % 4);
        auto& floor = c.cfg.numerics.ale1d.min_width_floor;
        floor.enabled = true;
        floor.floor_cm = (variant == 7) ? 1.0e-9 : 2.0e-4;  // variant 7: no offender
        floor.relief_halfwidth_cells = 1 + variant % 4;
        floor.max_growth_factor = (variant % 2 == 0) ? 1.8 : 1.2;
        INFO("geom " << geom << " boundary " << boundary << " variant " << variant);
        const Outcome host = host_floor(c);
        const Outcome dev =
            device_outcome(ale1d::floor_candidate_device(c.state, c.cfg, c.features), n);
        CHECK(same(host, dev, true));
        ++compared;
        relieved += host.no_relief ? 0 : 1;
      }
    }
  }
  INFO("cases " << compared << " relieved " << relieved);
  CHECK(relieved > 0);
}
