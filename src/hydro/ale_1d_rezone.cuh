#pragma once

#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_types.cuh"

namespace tenryu::hydro::ale1d {

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
