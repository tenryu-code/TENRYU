#include "coupling/reale_trigger.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::coupling::reale_trigger {
namespace {

double* g_ref_r = nullptr;
double* g_ref_z = nullptr;
int g_ref_nodes = -1;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

int button_outer_node_ring_or_zero(const mesh::Mesh& mesh) {
  return (mesh.button_center && mesh.button_center->enabled)
             ? mesh.button_center->outer_node_ring
             : 0;
}

__device__ double atomic_max_double(double* address, const double val) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (val > __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__global__ void max_displacement_ratio_kernel(
    double* __restrict__ max_ratio2,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ ref_r,
    const double* __restrict__ ref_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const int nr,
    const int nz,
    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (button_outer_node_ring > 0 && c == 0) {
    return;
  }

  int structured_nodes[4];
  const int* nodes = nullptr;
  int nv = 0;
  if (cell_node_csr_offsets != nullptr &&
      cell_node_csr_indices != nullptr) {
    const int off = cell_node_csr_offsets[c];
    nv = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    nodes = cell_node_csr_indices + off;
  } else {
    const int i = c / nz;
    const int j = c - i * nz;
    const int stride = nz + 1;
    const int n00 = i * stride + j;
    const int n10 = (i + 1) * stride + j;
    const int n11 = (i + 1) * stride + (j + 1);
    const int n01 = i * stride + (j + 1);
    structured_nodes[0] = n00;
    structured_nodes[1] = n10;
    structured_nodes[2] = n11;
    structured_nodes[3] = n01;
    nodes = structured_nodes;
    nv = 4;
  }

  double max_disp2 = 0.0;
  double a2 = 0.0;
  for (int k = 0; k < nv; ++k) {
    const int l = (k + 1 == nv) ? 0 : k + 1;
    const int n_k = nodes[k];
    const int n_l = nodes[l];
    const double dr = x_r[n_k] - ref_r[n_k];
    const double dz = x_z[n_k] - ref_z[n_k];
    max_disp2 = fmax(max_disp2, dr * dr + dz * dz);
    a2 += x_r[n_k] * x_z[n_l] - x_r[n_l] * x_z[n_k];
  }
  const double area = 0.5 * fabs(a2);
  const double ratio2 = (!(area > 0.0) || !isfinite(area))
                            ? 4.0
                            : max_disp2 / (0.25 * area);

  // The atomic max is order-independent because max is associative and
  // commutative and every contribution is a non-NaN finite value, so this
  // census is run-to-run deterministic.
  atomic_max_double(max_ratio2, ratio2);
}

}  // namespace

bool reference_valid(const core::State& state) {
  return g_ref_nodes == state.mesh.topo.n_nodes && g_ref_nodes > 0;
}

void snapshot_reference(const core::State& state) {
  const int n_nodes = state.mesh.topo.n_nodes;
  if (g_ref_nodes != n_nodes) {
    if (g_ref_r != nullptr) {
      cuda_check(cudaFree(g_ref_r),
                 "ReALE trigger: cudaFree reference r failed");
    }
    if (g_ref_z != nullptr) {
      cuda_check(cudaFree(g_ref_z),
                 "ReALE trigger: cudaFree reference z failed");
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&g_ref_r),
                          static_cast<std::size_t>(n_nodes) * sizeof(double)),
               "ReALE trigger: cudaMalloc reference r failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&g_ref_z),
                          static_cast<std::size_t>(n_nodes) * sizeof(double)),
               "ReALE trigger: cudaMalloc reference z failed");
  }
  cuda_check(cudaMemcpy(g_ref_r, state.x_r.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "ReALE trigger: copy reference r failed");
  cuda_check(cudaMemcpy(g_ref_z, state.x_z.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "ReALE trigger: copy reference z failed");
  g_ref_nodes = n_nodes;
}

double max_displacement_ratio(const core::State& state) {
  if (!reference_valid(state)) {
    return -1.0;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  const bool is_multiblock = state.mesh.topo.multiblock.has_value();

  double* d_max_ratio2 = static_cast<double*>(
      core::device_scratch_acquire("reale_trigger:max_ratio2", sizeof(double)));
  const double zero = 0.0;
  cuda_check(cudaMemcpy(d_max_ratio2, &zero, sizeof(double),
                        cudaMemcpyHostToDevice),
             "ReALE trigger: init max ratio failed");

  const std::uint8_t* d_cell_nverts = nullptr;
  core::DeviceArray<std::uint8_t> cell_nverts_storage;
  if (is_multiblock && !state.mesh.cell_nverts.empty()) {
    cell_nverts_storage.reset(static_cast<std::size_t>(n_cells));
    cell_nverts_storage.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts = cell_nverts_storage.data();
  }

  const int button_outer_node_ring =
      button_outer_node_ring_or_zero(state.mesh);
  const int blocks = (n_cells + 255) / 256;
  max_displacement_ratio_kernel<<<blocks, 256>>>(
      d_max_ratio2, state.x_r.data(), state.x_z.data(), g_ref_r, g_ref_z,
      is_multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data()
                    : nullptr,
      is_multiblock ? state.mesh.multiblock_cell_node_csr_indices.data()
                    : nullptr,
      d_cell_nverts, n_cells, nr, nz, button_outer_node_ring);
  cuda_check(cudaGetLastError(), "ReALE trigger kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "ReALE trigger kernel execution failed");

  double max_ratio2 = 0.0;
  cuda_check(cudaMemcpy(&max_ratio2, d_max_ratio2, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "ReALE trigger: copy max ratio failed");
  return std::sqrt(max_ratio2);
}

}  // namespace tenryu::coupling::reale_trigger
