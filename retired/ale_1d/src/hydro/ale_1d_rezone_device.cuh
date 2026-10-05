#pragma once

#include <cstdint>
#include <limits>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_types.cuh"

namespace tenryu::hydro::ale1d {

// A rezone candidate built on the device (NUMERICS §3.4.1). The candidate node radii and the node
// mask stay on the device for the remap and the commit; the host reads the values its gates need
// in one copy. The device computes what the host path computes: rezone(state, cfg, features)
// with build_monitor, or build_min_width_floor_candidate for the min-width floor trigger; then
// the driver's boundary nodes (enforce_boundary_candidate), its geometry check
// (validate_candidate_geometry) and the acoustic time-step bounds (acoustic_dt_bounds). Each
// operation is the host's, in the host's order and rounding, exp and pow those of the host
// (core::glibc_libm): the same candidate bit for bit (on x86-64 hosts where glibc selects its FMA
// build of exp and pow).
struct Ale1dDeviceCandidate {
  const double* r_candidate = nullptr;   // n + 1 node radii (device)
  const std::uint8_t* pinned = nullptr;  // n + 1, 1 where the node is pinned (device)
  bool success = false;
  Ale1dSkipReason skip_reason = Ale1dSkipReason::None;
  bool no_relief_available = false;  // the floor candidate found no window to relieve
  int n_protected_nodes = 0;         // after the boundary nodes are pinned
  bool geometry_valid = false;
  double dt_current = std::numeric_limits<double>::infinity();
  double dt_candidate = std::numeric_limits<double>::infinity();
  // rezone's displacement diagnostics, protected fraction and smallest movable segment
  double max_node_displacement_mu = 0.0;
  double max_node_displacement_r = 0.0;
  double protected_fraction = 0.0;
  int min_movable_segment_size = 0;
  // the features as uploaded for the build (device), for protected_faces_device
  const void* features = nullptr;
  int n_features = 0;
};

// The candidate of the cadence and mesh-quality triggers: rezone(state, cfg, features).
Ale1dDeviceCandidate rezone_candidate_device(const core::State& state,
                                             const core::Config& cfg,
                                             const std::vector<Ale1dFeature>& features);

// The candidate of the min-width floor trigger: build_min_width_floor_candidate on the current
// node radii, with the nodes 0 and n and the pinned faces of the features pinned, and the cells
// with rho > 100 rho_floor eligible.
Ale1dDeviceCandidate floor_candidate_device(const core::State& state,
                                            const core::Config& cfg,
                                            const std::vector<Ale1dFeature>& features);

// The remap's protected faces on the device: the end nodes, the pinned nodes, and the faces of
// the candidate's features (protected_faces_from_features: each nonnegative peak, clamped to
// [0, n]). n + 1 entries, valid until the next call.
const std::uint8_t* protected_faces_device(const Ale1dDeviceCandidate& candidate, int n_cells);

}  // namespace tenryu::hydro::ale1d
