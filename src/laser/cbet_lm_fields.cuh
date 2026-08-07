#pragma once

#include <cstdint>
#include <vector>

#include "laser/laser_mesh.cuh"

namespace tenryu::laser {

// Host-side laser-mesh node fields captured for the CBET 2D cell pack (all sized
// n_nodes_r*n_nodes_z except the axes). Filled by map_from_hydro_2d when requested.
struct CbetLmFields {
  std::vector<double> node_R;      // [n_nodes_r]
  std::vector<double> node_Z;      // [n_nodes_z]
  std::vector<double> n_hat_raw;   // unclipped ne/nc at nodes
  std::vector<double> Te;          // eV
  std::vector<double> Zbar;
  std::vector<double> Ti;          // eV
  std::vector<double> u_R;         // cm/s (hydro cell-mean radial velocity, interp to nodes)
  std::vector<double> u_Z;         // cm/s
  std::vector<double> A_eff;
  std::vector<std::uint8_t> covered;  // 1 = node inside real hydro coverage and non-void source
};

void map_from_hydro_2d_cbet(LaserMesh& mesh,
                            const core::State& state,
                            const core::Config::LaserConfig& laser_cfg,
                            CbetLmFields& cbet_out,
                            cudaStream_t stream = nullptr);

}  // namespace tenryu::laser
