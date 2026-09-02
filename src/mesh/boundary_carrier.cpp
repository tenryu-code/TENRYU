#include "mesh/boundary_carrier.hpp"

#include <cstddef>

namespace tenryu::mesh {

BoundaryCarrier carrier_from_boundary_loop(const std::vector<int>& nodes,
                                           const std::vector<double>& node_r,
                                           const std::vector<double>& node_z) {
  (void)node_z;
  BoundaryCarrier carrier;
  carrier.masters.reserve(nodes.size());
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    const int node = nodes[index];
    carrier.masters.push_back(
        {static_cast<std::uint64_t>(index), node,
         node_r[static_cast<std::size_t>(node)] == 0.0
             ? CarrierBcClass::kAxis
             : CarrierBcClass::kOuter});
  }
  carrier.valid = nodes.size() >= 3U;
  return carrier;
}

}  // namespace tenryu::mesh
