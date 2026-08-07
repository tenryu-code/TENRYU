#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "laser/hot_electron_2d.cuh"

namespace tenryu::laser::hot_electron {

struct DeviceMeshView2D {
  int n_cells = 0;
  const int* cell_nodes = nullptr;
  const int* cell_neighbor = nullptr;
  const double* node_R = nullptr;
  const double* node_Z = nullptr;

  TENRYU_HOTE_HOST_DEVICE MeshView2D as_mesh_view() const {
    MeshView2D out;
    out.n_cells = n_cells;
    out.cell_nodes = cell_nodes;
    out.cell_neighbor = cell_neighbor;
    out.node_R = node_R;
    out.node_Z = node_Z;
    return out;
  }
};

struct DeviceMeshView2DScratch {
  int* cell_nodes = nullptr;
  int* cell_neighbor = nullptr;
  double* node_R = nullptr;
  double* node_Z = nullptr;
};

DeviceMeshView2D upload_mesh_view_2d(
    const MeshView2DStorage& storage,
    const std::vector<double>& node_R_host,
    const std::vector<double>& node_Z_host,
    int n_cells,
    cudaStream_t stream,
    DeviceMeshView2DScratch& scratch);

DepositResult2D deposit_hot_electrons_cone_2d_device(
    const HotEChannelSpec& spec,
    const std::vector<HotESource2D>& sources,
    const double* d_rho,
    const double* d_zbar,
    const double* d_A_eff,
    const double* d_Te_eV,
    const std::vector<std::uint8_t>& cell_is_void_host,
    const DeviceMeshView2D& dview,
    double mesh_scale,
    int max_segments,
    cudaStream_t stream,
    std::vector<double>& dep_power_cell);

}  // namespace tenryu::laser::hot_electron
