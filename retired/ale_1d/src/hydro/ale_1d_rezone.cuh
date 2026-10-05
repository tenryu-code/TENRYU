#pragma once

#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_types.cuh"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro::ale1d {

// Constants of the rezone shared by the host build and the device build
// (ale_1d_rezone_device.cu): the smallest length scale and the smallest kernel integral the
// monitor keeps.
namespace rezone_detail {
inline constexpr double kTiny = 1.0e-30;
inline constexpr double kKernelIntegralFloor = 1.0e-14;
}  // namespace rezone_detail

// The volume coordinate V(r) of the candidate geometry check (4 pi r^3 / 3 in spherical geometry,
// the shell volume from 0 otherwise).
__host__ __device__ inline double ale1d_volume_coordinate(const double r, const int geom) {
  constexpr double kFourPiOverThree =
      4.188790204786390984616857844372670512262892532500141094646;
  return (geom == 0) ? (kFourPiOverThree * r * r * r)
                     : tenryu::mesh::geometry_1d_shell_volume_cubes(geom, 0.0, r);
}

using Ale1dRezoneConfig =
    core::Config::NumericsConfig::Ale1dConfig::RezoneConfig;

struct NodeConstraintMask {
  std::vector<bool> pinned;
  int n_protected_nodes = 0;
};

struct Ale1dRezoneResult {
  bool success = false;
  Ale1dSkipReason skip_reason = Ale1dSkipReason::None;
  std::vector<double> r_candidate;
  NodeConstraintMask node_mask;
  double max_node_displacement_mu = 0.0;
  double max_node_displacement_r = 0.0;
  double protected_fraction = 0.0;
  int min_movable_segment_size = 0;
};

struct MinWidthFloorCandidateResult {
  bool success = false;
  bool fully_relieved = false;
  bool no_relief_available = false;
  std::vector<double> r_candidate;
  double min_dl_before = 0.0;
  double min_dl_after = 0.0;
  int n_windows = 0;
};

MinWidthFloorCandidateResult build_min_width_floor_candidate(
    const std::vector<double>& r_nodes,
    const std::vector<bool>& pinned,
    const std::vector<bool>& eligible,
    double floor_cm,
    double target_factor,
    int relief_halfwidth_cells,
    double max_growth_factor);

std::vector<double> build_monitor(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<Ale1dFeature>& features);

Ale1dRezoneResult rezone(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<Ale1dFeature>& features,
    const std::vector<double>& W_smooth_clipped);

Ale1dRezoneResult rezone(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<Ale1dFeature>& features);

}  // namespace tenryu::hydro::ale1d
