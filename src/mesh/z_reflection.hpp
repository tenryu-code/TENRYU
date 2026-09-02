#pragma once

#include <string>
#include <vector>

namespace tenryu::mesh {

struct MultiBlockTopology;

struct ZReflectionMaps {
  bool valid = false;
  std::vector<int> node_mirror;  // node id -> mirrored node id
  std::vector<int> cell_mirror;  // cell id -> mirrored cell id
  std::string failure;           // reason when !valid
};

ZReflectionMaps build_z_reflection_maps(
    const MultiBlockTopology& topology,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z);

struct ZReflectionAudit {
  double odd_mass = 0.0;    // ||m - S m|| / (2 ||m||)
  double odd_mom_r = 0.0;   // P_r channel (even rule)
  double odd_mom_z = 0.0;   // P_z channel (odd rule)
  double odd_energy = 0.0;
  double odd_node_r = 0.0;  // node coordinate channels
  double odd_node_z = 0.0;
};

ZReflectionAudit audit_z_reflection(
    const ZReflectionMaps& maps,
    const std::vector<double>& mass,
    const std::vector<double>& mom_r,
    const std::vector<double>& mom_z,
    const std::vector<double>& energy,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z);

}  // namespace tenryu::mesh
