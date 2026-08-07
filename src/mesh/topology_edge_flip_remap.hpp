#pragma once

#include <cstdint>
#include <vector>

#include "mesh/mesh.hpp"
#include "mesh/topology_edge_flip.hpp"

namespace tenryu::mesh {

struct EdgeFlipRemapResult {
  bool applied;
  double exchanged_area;
  double mass_moved;
};

// Conservatively remaps cell-extensive fields using exact planar overlap areas.
// RZ moment refinement is intentionally deferred to a later slice.
EdgeFlipRemapResult topology_edge_flip_with_remap(
    MultiBlockTopology& topology,
    std::vector<std::uint8_t>& cell_nverts,
    const double* node_r,
    const double* node_z,
    int n_nodes,
    EdgeFlipEvent& event,
    double* mass,
    double* mom_r,
    double* mom_z,
    double* e_int);

}  // namespace tenryu::mesh
