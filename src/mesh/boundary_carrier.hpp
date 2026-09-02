#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::mesh {

enum class CarrierBcClass : std::uint8_t {
  kOuter = 0,
  kAxis = 1,
};

struct CarrierVertex {
  std::uint64_t stable_id = 0;  // dense loop position at capture, frozen
  int mesh_node = -1;           // current generation's node index; refreshed at install
  CarrierBcClass bc_class = CarrierBcClass::kOuter;
};

// Persistent Lagrangian boundary carrier (C1 contract §1). Masters are ordered as a
// CCW closed loop; edge k connects masters[k] to masters[(k+1) % size]. Cardinality is
// frozen at capture. Master positions live in the hydro node arrays via mesh_node.
struct BoundaryCarrier {
  std::vector<CarrierVertex> masters;
  bool valid = false;
};

// Builds a carrier from a closed boundary node loop (post-compaction order). bc_class
// is kAxis for vertices with r == 0.0, else kOuter.
BoundaryCarrier carrier_from_boundary_loop(const std::vector<int>& nodes,
                                           const std::vector<double>& node_r,
                                           const std::vector<double>& node_z);

}  // namespace tenryu::mesh
