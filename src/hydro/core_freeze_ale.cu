#include "hydro/core_freeze_ale.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "hydro/ale_remap.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale::core_freeze {
namespace {

void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

std::string format_scientific(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6) << value;
  return oss.str();
}

bool has_active_cell_nverts(const tenryu::mesh::Mesh& mesh) {
  return mesh.cell_nverts.size() == static_cast<std::size_t>(mesh.topo.n_cells);
}

int active_nverts_host(const tenryu::mesh::Mesh& mesh, const int cell) {
  return has_active_cell_nverts(mesh)
             ? tenryu::mesh::mesh_topo_cell_active_nverts(mesh.cell_nverts, cell)
             : tenryu::mesh::kMeshTopoCellStorageSlots;
}

void mark_structured_cell_nodes(const tenryu::mesh::Mesh& mesh,
                                const int cell,
                                std::vector<std::uint8_t>& node_mask) {
  const int nz = mesh.topo.nz;
  const int i = cell / nz;
  const int j = cell - i * nz;
  const int stride = nz + 1;
  const int nodes[4] = {
      i * stride + j,
      (i + 1) * stride + j,
      (i + 1) * stride + (j + 1),
      i * stride + (j + 1),
  };
  const int nverts = active_nverts_host(mesh, cell);
  for (int k = 0; k < nverts; ++k) {
    const int n = nodes[k];
    if (n >= 0 && n < mesh.topo.n_nodes) {
      node_mask[static_cast<std::size_t>(n)] = 1U;
    }
  }
}

void mark_multiblock_cell_nodes(const tenryu::mesh::Mesh& mesh,
                                const int cell,
                                std::vector<std::uint8_t>& node_mask) {
  const auto& mb = *mesh.topo.multiblock;
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int nverts = active_nverts_host(mesh, cell);
  for (int k = 0; k < nverts; ++k) {
    const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    TENRYU_ASSERT(n >= 0 && n < mesh.topo.n_nodes,
                  "core freeze cell-node CSR node id out of range");
    node_mask[static_cast<std::size_t>(n)] = 1U;
  }
}

void dilate_structured_cells(const tenryu::mesh::Mesh& mesh,
                             const std::vector<std::uint8_t>& in,
                             std::vector<std::uint8_t>& out) {
  const int nr = mesh.topo.nr;
  const int nz = mesh.topo.nz;
  for (int c = 0; c < mesh.topo.n_cells; ++c) {
    if (in[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int i = c / nz;
    const int j = c - i * nz;
    if (i > 0) {
      out[static_cast<std::size_t>((i - 1) * nz + j)] = 1U;
    }
    if (i + 1 < nr) {
      out[static_cast<std::size_t>((i + 1) * nz + j)] = 1U;
    }
    if (j > 0) {
      out[static_cast<std::size_t>(i * nz + (j - 1))] = 1U;
    }
    if (j + 1 < nz) {
      out[static_cast<std::size_t>(i * nz + (j + 1))] = 1U;
    }
  }
}

void dilate_multiblock_cells(const tenryu::mesh::Mesh& mesh,
                             const std::vector<std::uint8_t>& in,
                             std::vector<std::uint8_t>& out) {
  const auto& mb = *mesh.topo.multiblock;
  TENRYU_ASSERT(mb.face_adj_csr_offsets.size() ==
                    static_cast<std::size_t>(mesh.topo.n_cells + 1),
                "core freeze requires face-adjacency CSR offsets");
  TENRYU_ASSERT(mb.face_adj_csr_indices.size() ==
                    static_cast<std::size_t>(mesh.topo.n_cells) *
                        tenryu::mesh::kMeshTopoCellStorageSlots,
                "core freeze requires face-adjacency CSR indices");
  for (int c = 0; c < mesh.topo.n_cells; ++c) {
    if (in[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts_host(mesh, c);
    for (int local = 0; local < tenryu::mesh::kMeshTopoCellStorageSlots; ++local) {
      if (!tenryu::mesh::mesh_topo_local_face_is_active(nverts, local)) {
        continue;
      }
      const int nb = mb.face_adj_csr_indices[static_cast<std::size_t>(off + local)];
      if (nb >= 0 && nb < mesh.topo.n_cells) {
        out[static_cast<std::size_t>(nb)] = 1U;
      }
    }
  }
}

struct HostMask {
  std::vector<std::uint8_t> frozen_cells;
  std::vector<std::uint8_t> frozen_nodes;
  int frozen_cell_count = 0;
  int frozen_node_count = 0;
};

HostMask build_host_mask(core::State& state, const core::Config& cfg) {
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0,
                "core freeze requires positive mesh sizes");
  TENRYU_ASSERT(cfg.numerics.ale.core_freeze_source == "gas_tracer",
                "Numerics.ale.core_freeze_source=\"gas_tracer\" is the only "
                "implemented core-freeze source");
  TENRYU_ASSERT(state.gas_tracer_Y.size() == static_cast<std::size_t>(n_cells),
                "core freeze source gas_tracer requires state.gas_tracer_Y "
                "with one entry per cell");

  std::vector<double> tracer;
  state.gas_tracer_Y.copy_to_host(tracer);
  HostMask mask;
  mask.frozen_cells.assign(static_cast<std::size_t>(n_cells), 0U);
  mask.frozen_nodes.assign(static_cast<std::size_t>(n_nodes), 0U);
  const double cut = cfg.numerics.ale.core_freeze_tracer_cut;
  for (int c = 0; c < n_cells; ++c) {
    const double y = tracer[static_cast<std::size_t>(c)];
    if (std::isfinite(y) && y >= cut) {
      mask.frozen_cells[static_cast<std::size_t>(c)] = 1U;
    }
  }

  const bool multiblock = state.mesh.topo.multiblock.has_value();
  const int halo_layers = cfg.numerics.ale.core_freeze_halo_layers;
  for (int layer = 0; layer < halo_layers; ++layer) {
    std::vector<std::uint8_t> next = mask.frozen_cells;
    if (multiblock) {
      dilate_multiblock_cells(state.mesh, mask.frozen_cells, next);
    } else {
      dilate_structured_cells(state.mesh, mask.frozen_cells, next);
    }
    mask.frozen_cells.swap(next);
  }

  for (int c = 0; c < n_cells; ++c) {
    if (mask.frozen_cells[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    ++mask.frozen_cell_count;
    if (multiblock) {
      mark_multiblock_cell_nodes(state.mesh, c, mask.frozen_nodes);
    } else {
      mark_structured_cell_nodes(state.mesh, c, mask.frozen_nodes);
    }
  }
  for (const std::uint8_t frozen : mask.frozen_nodes) {
    if (frozen != 0U) {
      ++mask.frozen_node_count;
    }
  }
  return mask;
}

__device__ void atomic_max_double(double* addr, const double value) {
  auto* addr_ull = reinterpret_cast<unsigned long long int*>(addr);
  unsigned long long int old = *addr_ull;
  unsigned long long int assumed = 0ULL;
  while (value > __longlong_as_double(static_cast<long long>(old))) {
    assumed = old;
    old = atomicCAS(addr_ull, assumed, __double_as_longlong(value));
    if (assumed == old) {
      break;
    }
  }
}

__global__ void restore_frozen_nodes_kernel(double* __restrict__ target_r,
                                            double* __restrict__ target_z,
                                            const double* __restrict__ current_r,
                                            const double* __restrict__ current_z,
                                            const std::uint8_t* __restrict__ frozen_nodes,
                                            const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes || frozen_nodes[n] == 0U) {
    return;
  }
  target_r[n] = current_r[n];
  target_z[n] = current_z[n];
}

__global__ void frozen_node_displacement_kernel(
    const double* __restrict__ target_r,
    const double* __restrict__ target_z,
    const double* __restrict__ current_r,
    const double* __restrict__ current_z,
    const std::uint8_t* __restrict__ frozen_nodes,
    double* __restrict__ max_displacement,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes || frozen_nodes[n] == 0U) {
    return;
  }
  const double dr = target_r[n] - current_r[n];
  const double dz = target_z[n] - current_z[n];
  const double disp = sqrt(dr * dr + dz * dz);
  if (isfinite(disp)) {
    atomic_max_double(max_displacement, disp);
  }
}

__global__ void frozen_cell_swept_volume_csr_kernel(
    const double* __restrict__ current_r,
    const double* __restrict__ current_z,
    const double* __restrict__ target_r,
    const double* __restrict__ target_z,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ frozen_cells,
    double* __restrict__ swept_abs_sum,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || frozen_cells[c] == 0U) {
    return;
  }
  const int off = cell_node_offsets[c];
  const int nverts = tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double swept_abs = 0.0;
  for (int local = 0; local < tenryu::mesh::kMeshTopoCellStorageSlots; ++local) {
    int corner0 = 0;
    int corner1 = 0;
    if (!tenryu::mesh::mesh_topo_active_local_face_corners(
            nverts, local, &corner0, &corner1)) {
      continue;
    }
    const int na = cell_node_indices[off + corner0];
    const int nb = cell_node_indices[off + corner1];
    const double dV = tenryu::hydro::ale::detail::face_swept_volume_outward(
        current_r[na],
        current_z[na],
        current_r[nb],
        current_z[nb],
        target_r[na],
        target_z[na],
        target_r[nb],
        target_z[nb]);
    if (isfinite(dV)) {
      swept_abs += fabs(dV);
    }
  }
  if (swept_abs > 0.0 && isfinite(swept_abs)) {
    atomicAdd(swept_abs_sum, swept_abs);
  }
}

__global__ void frozen_cell_swept_volume_structured_kernel(
    const double* __restrict__ current_r,
    const double* __restrict__ current_z,
    const double* __restrict__ target_r,
    const double* __restrict__ target_z,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ frozen_cells,
    double* __restrict__ swept_abs_sum,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells || frozen_cells[c] == 0U) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int nodes[4] = {
      i * stride + j,
      (i + 1) * stride + j,
      (i + 1) * stride + (j + 1),
      i * stride + (j + 1),
  };
  const int nverts = tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double swept_abs = 0.0;
  for (int local = 0; local < tenryu::mesh::kMeshTopoCellStorageSlots; ++local) {
    int corner0 = 0;
    int corner1 = 0;
    if (!tenryu::mesh::mesh_topo_active_local_face_corners(
            nverts, local, &corner0, &corner1)) {
      continue;
    }
    const int na = nodes[corner0];
    const int nb = nodes[corner1];
    const double dV = tenryu::hydro::ale::detail::face_swept_volume_outward(
        current_r[na],
        current_z[na],
        current_r[nb],
        current_z[nb],
        target_r[na],
        target_z[na],
        target_r[nb],
        target_z[nb]);
    if (isfinite(dV)) {
      swept_abs += fabs(dV);
    }
  }
  if (swept_abs > 0.0 && isfinite(swept_abs)) {
    atomicAdd(swept_abs_sum, swept_abs);
  }
}

void log_diagnostics(const core::State& state,
                     const char* path_tag,
                     const CoreFreezeDiagnostics& diag) {
  std::ostringstream oss;
  oss << "[core_freeze] step=" << state.step
      << " path=" << (path_tag != nullptr ? path_tag : "unknown")
      << " frozen_cells=" << diag.frozen_cell_count
      << " frozen_nodes=" << diag.frozen_node_count
      << " velocity_projection_skipped_nodes="
      << diag.velocity_projection_skipped_node_count
      << " max_frozen_node_displacement="
      << format_scientific(diag.max_frozen_node_displacement)
      << " frozen_cell_swept_volume_abs_sum="
      << format_scientific(diag.frozen_cell_swept_volume_abs_sum);
  core::log_info(oss.str());
}

}  // namespace

CoreFreezeDiagnostics restore_target_if_enabled(
    core::State& state,
    const core::Config& cfg,
    double* d_target_r,
    double* d_target_z,
    const double* d_current_r,
    const double* d_current_z,
    const bool for_axis_rezone,
    const char* path_tag,
    core::DeviceArray<std::uint8_t>* d_frozen_nodes_out) {
  CoreFreezeDiagnostics diag;
  if (d_frozen_nodes_out != nullptr) {
    d_frozen_nodes_out->reset(0);
  }
  if (!cfg.numerics.ale.core_freeze_enabled) {
    return diag;
  }
  diag.enabled = true;
  if (for_axis_rezone && !cfg.numerics.ale.core_freeze_apply_to_axis_rezone) {
    log_diagnostics(state, path_tag, diag);
    return diag;
  }

  TENRYU_ASSERT(d_target_r != nullptr && d_target_z != nullptr &&
                    d_current_r != nullptr && d_current_z != nullptr,
                "core freeze restore requires non-null coordinate arrays");
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  HostMask host_mask = build_host_mask(state, cfg);
  diag.frozen_cell_count = host_mask.frozen_cell_count;
  diag.frozen_node_count = host_mask.frozen_node_count;

  core::DeviceArray<std::uint8_t> d_frozen_cells(
      static_cast<std::size_t>(n_cells));
  core::DeviceArray<std::uint8_t> d_frozen_nodes_local;
  core::DeviceArray<std::uint8_t>* d_frozen_nodes =
      d_frozen_nodes_out != nullptr ? d_frozen_nodes_out : &d_frozen_nodes_local;
  d_frozen_nodes->reset(static_cast<std::size_t>(n_nodes));
  d_frozen_cells.copy_from_host(host_mask.frozen_cells);
  d_frozen_nodes->copy_from_host(host_mask.frozen_nodes);
  if (d_frozen_nodes_out != nullptr &&
      cfg.numerics.ale.core_freeze_skip_velocity_projection) {
    diag.velocity_projection_skipped_node_count = host_mask.frozen_node_count;
  }

  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr = nullptr;
  if (has_active_cell_nverts(state.mesh)) {
    d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
    d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts_ptr = d_cell_nverts.data();
  }

  if (diag.frozen_node_count > 0) {
    const int blocks_nodes = (n_nodes + 255) / 256;
    restore_frozen_nodes_kernel<<<blocks_nodes, 256>>>(
        d_target_r,
        d_target_z,
        d_current_r,
        d_current_z,
        d_frozen_nodes->data(),
        n_nodes);
    cuda_check(cudaGetLastError(), "core freeze restore kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "core freeze restore kernel synchronize failed");
  }

  double* d_max_disp = nullptr;
  double* d_swept_abs_sum = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_max_disp), sizeof(double)),
             "core freeze max displacement allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_swept_abs_sum), sizeof(double)),
             "core freeze swept-volume sum allocation failed");
  cuda_check(cudaMemset(d_max_disp, 0, sizeof(double)),
             "core freeze max displacement memset failed");
  cuda_check(cudaMemset(d_swept_abs_sum, 0, sizeof(double)),
             "core freeze swept-volume sum memset failed");

  const int blocks_nodes = (n_nodes + 255) / 256;
  frozen_node_displacement_kernel<<<blocks_nodes, 256>>>(
      d_target_r,
      d_target_z,
      d_current_r,
      d_current_z,
      d_frozen_nodes->data(),
      d_max_disp,
      n_nodes);
  cuda_check(cudaGetLastError(),
             "core freeze node displacement diagnostic kernel launch failed");

  const int blocks_cells = (n_cells + 255) / 256;
  if (state.mesh.topo.multiblock.has_value()) {
    frozen_cell_swept_volume_csr_kernel<<<blocks_cells, 256>>>(
        d_current_r,
        d_current_z,
        d_target_r,
        d_target_z,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts_ptr,
        d_frozen_cells.data(),
        d_swept_abs_sum,
        n_cells);
    cuda_check(cudaGetLastError(),
               "core freeze CSR swept-volume diagnostic kernel launch failed");
  } else {
    frozen_cell_swept_volume_structured_kernel<<<blocks_cells, 256>>>(
        d_current_r,
        d_current_z,
        d_target_r,
        d_target_z,
        d_cell_nverts_ptr,
        d_frozen_cells.data(),
        d_swept_abs_sum,
        state.mesh.topo.nr,
        state.mesh.topo.nz);
    cuda_check(cudaGetLastError(),
               "core freeze structured swept-volume diagnostic kernel launch failed");
  }
  cuda_check(cudaDeviceSynchronize(),
             "core freeze diagnostic kernels synchronize failed");
  cuda_check(cudaMemcpy(&diag.max_frozen_node_displacement,
                        d_max_disp,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "core freeze max displacement copy failed");
  cuda_check(cudaMemcpy(&diag.frozen_cell_swept_volume_abs_sum,
                        d_swept_abs_sum,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "core freeze swept-volume sum copy failed");
  cuda_check(cudaFree(d_swept_abs_sum),
             "core freeze swept-volume sum free failed");
  cuda_check(cudaFree(d_max_disp), "core freeze max displacement free failed");

  log_diagnostics(state, path_tag, diag);
  return diag;
}

}  // namespace tenryu::hydro::ale::core_freeze
