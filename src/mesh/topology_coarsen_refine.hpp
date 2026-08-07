#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace tenryu::mesh {

struct MultiBlockTopology;

struct CellLineageRecord {
  int cell_a;
  int cell_b;
  int id_stable_a;
  int id_stable_b;
  std::array<int, 8> cycle_a;
  std::array<int, 8> cycle_b;
  std::uint8_t nverts_a;
  std::uint8_t nverts_b;
  double mass_b;
  double mom_r_b;
  double mom_z_b;
  double e_int_b;
  double vol_a;
  double vol_b;
  double mass_a;
  double mom_r_a;
  double mom_z_a;
  double e_int_a;
};

bool topology_coarsen_pair(
    MultiBlockTopology& topology,
    std::vector<std::uint8_t>& cell_nverts,
    const double* node_r,
    const double* node_z,
    int n_nodes,
    int cell_a,
    int cell_b,
    double* mass,
    double* mom_r,
    double* mom_z,
    double* e_int,
    double* vol,
    CellLineageRecord& lineage_out);

bool topology_refine_from_lineage(
    MultiBlockTopology& topology,
    std::vector<std::uint8_t>& cell_nverts,
    double* mass,
    double* mom_r,
    double* mom_z,
    double* e_int,
    double* vol,
    const CellLineageRecord& lineage);

}  // namespace tenryu::mesh
