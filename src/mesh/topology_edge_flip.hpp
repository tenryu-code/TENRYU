#pragma once

#include <cstdint>
#include <vector>

#include "mesh/mesh.hpp"

namespace tenryu::mesh {

struct EdgeFlipEvent {
  int cell_a;
  int cell_b;
  int shared_na;
  int shared_nb;
  int opposite_nc;
  int opposite_nd;
};

bool topology_edge_flip_apply(
    MultiBlockTopology& topology,
    std::vector<std::uint8_t>& cell_nverts,
    const double* node_r,
    const double* node_z,
    int n_nodes,
    EdgeFlipEvent& event);

EdgeFlipEvent topology_edge_flip_reverse(const EdgeFlipEvent& event);

}  // namespace tenryu::mesh
