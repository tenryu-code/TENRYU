#include "hydro/compatible_force_work_2d.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "core/error.hpp"
#include "hydro/hydro_multiblock_topology.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::compatible {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ inline double atomic_add_double(double* address, const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

void zero_array(double* ptr, const std::size_t n, const char* message) {
  if (ptr == nullptr || n == 0U) {
    return;
  }
  cuda_check(cudaMemset(ptr, 0, n * sizeof(double)), message);
}

bool has_tri_cell_nverts(const std::vector<std::uint8_t>& cell_nverts,
                         const int n_cells) {
  if (cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  return std::any_of(cell_nverts.begin(), cell_nverts.end(),
                     [](const std::uint8_t nverts) {
                       return nverts != 4U;
                     });
}

const std::uint8_t* upload_cell_nverts_if_nonquad(
    core::DeviceArray<std::uint8_t>& d_cell_nverts,
    const core::State& state,
    const int n_cells) {
  if (!has_tri_cell_nverts(state.mesh.cell_nverts, n_cells)) {
    return nullptr;
  }
  d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
  d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
  return d_cell_nverts.data();
}

CompatibleForceBuffers force_buffers(core::State& state) {
  CompatibleForceBuffers out{};
  out.corner_p_r = state.corner_force_p_r.data();
  out.corner_p_z = state.corner_force_p_z.data();
  out.corner_sub_r = state.corner_force_sub_r.data();
  out.corner_sub_z = state.corner_force_sub_z.data();
  out.corner_q_r = state.corner_force_q_r.data();
  out.corner_q_z = state.corner_force_q_z.data();
  out.edge_av_r = state.edge_force_av_r.data();
  out.edge_av_z = state.edge_force_av_z.data();
  out.n_cells = static_cast<int>(state.rho.size());
  out.n_edges = static_cast<int>(state.edge_force_av_r.size());
  return out;
}

CompatibleWorkBuffers work_buffers(core::State& state) {
  CompatibleWorkBuffers out{};
  out.work_p = state.work_p_per_cell.data();
  out.work_sub = state.work_sub_per_cell.data();
  out.work_av = state.work_av_per_cell.data();
  out.n_cells = static_cast<int>(state.rho.size());
  return out;
}

__device__ inline int structured_node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

__device__ inline void local_face_corners(const int local,
                                          int* corner0,
                                          int* corner1);

__device__ inline bool local_face_corners(const int active_nverts,
                                          const int local,
                                          int* corner0,
                                          int* corner1) {
  if (active_nverts >= 5) {
    return mesh::mesh_topo_active_local_face_corners(active_nverts, local,
                                                     corner0, corner1);
  }
  if (active_nverts != 3) {
    local_face_corners(local, corner0, corner1);
    return true;
  }
  return mesh::mesh_topo_active_local_face_corners(active_nverts, local,
                                                   corner0, corner1);
}

__device__ inline void local_face_corners(const int local,
                                          int* corner0,
                                          int* corner1) {
  if (local == 0) {
    *corner0 = 0;
    *corner1 = 3;
  } else if (local == 1) {
    *corner0 = 1;
    *corner1 = 2;
  } else if (local == 2) {
    *corner0 = 0;
    *corner1 = 1;
  } else {
    *corner0 = 3;
    *corner1 = 2;
  }
}

__global__ void assemble_pressure_corner_forces_kernel(
    CompatibleForceBuffers force,
    const double* __restrict__ cell_pressure,
    const double* __restrict__ svec_r,
    const double* __restrict__ svec_z,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_nverts,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= force.n_cells) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  const double p = active ? cell_pressure[c] : 0.0;
  const int base = c * corner_stride;
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts >= 5) {
    for (int k = 0; k < active_nverts; ++k) {
      force.corner_p_r[base + k] = p * svec_r[base + k];
      force.corner_p_z[base + k] = p * svec_z[base + k];
    }
  } else {
    for (int k = 0; k < 4; ++k) {
      force.corner_p_r[base + k] = p * svec_r[base + k];
      force.corner_p_z[base + k] = p * svec_z[base + k];
    }
  }
  const int padding_begin = (active_nverts >= 5) ? active_nverts : 4;
  for (int k = padding_begin; k < corner_stride; ++k) {
    force.corner_p_r[base + k] = 0.0;
    force.corner_p_z[base + k] = 0.0;
  }
}

__global__ void assemble_scalar_av_corner_forces_kernel(
    CompatibleForceBuffers force,
    const double* __restrict__ cell_q,
    const double* __restrict__ svec_r,
    const double* __restrict__ svec_z,
    const std::int8_t* __restrict__ hydro_active,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= force.n_cells) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  const double q = active ? cell_q[c] : 0.0;
  const int base = c * corner_stride;
  for (int k = 0; k < 4; ++k) {
    force.corner_q_r[base + k] = q * svec_r[base + k];
    force.corner_q_z[base + k] = q * svec_z[base + k];
  }
}

__global__ void sum_corner_forces_structured_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const CompatibleForceBuffers force,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      structured_node_index(i, j, nz),
      structured_node_index(i + 1, j, nz),
      structured_node_index(i + 1, j + 1, nz),
      structured_node_index(i, j + 1, nz),
  };
  const int base = c * corner_stride;
  for (int k = 0; k < 4; ++k) {
    atomic_add_double(force_r + nodes[k],
                      force.corner_p_r[base + k] + force.corner_sub_r[base + k]);
    atomic_add_double(force_z + nodes[k],
                      force.corner_p_z[base + k] + force.corner_sub_z[base + k]);
  }
}

__global__ void sum_scalar_av_corner_forces_structured_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const CompatibleForceBuffers force,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nr * nz) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      structured_node_index(i, j, nz),
      structured_node_index(i + 1, j, nz),
      structured_node_index(i + 1, j + 1, nz),
      structured_node_index(i, j + 1, nz),
  };
  const int base = c * corner_stride;
  for (int k = 0; k < 4; ++k) {
    atomic_add_double(force_r + nodes[k], force.corner_q_r[base + k]);
    atomic_add_double(force_z + nodes[k], force.corner_q_z[base + k]);
  }
}

__global__ void sum_corner_forces_multiblock_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const CompatibleForceBuffers force,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const int n_nodes_total,
    const int corner_stride) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes_total) {
    return;
  }

  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  double fr = 0.0;
  double fz = 0.0;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (hydro_active != nullptr && hydro_active[c] == 0) {
      continue;
    }
    const int corner = reverse_csr_node_corners[p];
    const int idx = c * corner_stride + corner;
    fr += force.corner_p_r[idx] + force.corner_sub_r[idx];
    fz += force.corner_p_z[idx] + force.corner_sub_z[idx];
  }
  force_r[n] = fr;
  force_z[n] = fz;
}

__global__ void sum_edge_forces_structured_kernel(double* __restrict__ force_r,
                                                  double* __restrict__ force_z,
                                                  const CompatibleForceBuffers force,
                                                  const int nr,
                                                  const int nz) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges || e >= force.n_edges) {
    return;
  }

  int n0 = -1;
  int n1 = -1;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i + 1, j, nz);
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i, j + 1, nz);
  }

  const double fr = force.edge_av_r[e];
  const double fz = force.edge_av_z[e];
  atomic_add_double(force_r + n0, -fr);
  atomic_add_double(force_z + n0, -fz);
  atomic_add_double(force_r + n1, +fr);
  atomic_add_double(force_z + n1, +fz);
}

__global__ void sum_edge_forces_multiblock_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const CompatibleForceBuffers force,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int edge_offset,
    const int n_faces) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int e = edge_offset + f;
  if (e >= force.n_edges) {
    return;
  }

  const int c = face_cell[f];
  const int local = face_local[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int off = cell_node_csr_offsets[c];
  const int n0 = cell_node_csr_indices[off + corner0];
  const int n1 = cell_node_csr_indices[off + corner1];

  const double fr = force.edge_av_r[e];
  const double fz = force.edge_av_z[e];
  atomic_add_double(force_r + n0, -fr);
  atomic_add_double(force_z + n0, -fz);
  atomic_add_double(force_r + n1, +fr);
  atomic_add_double(force_z + n1, +fz);
}

__global__ void compatible_work_structured_kernel(
    CompatibleWorkBuffers work,
    const CompatibleForceBuffers force,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells || c >= work.n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    work.work_p[c] = 0.0;
    work.work_sub[c] = 0.0;
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      structured_node_index(i, j, nz),
      structured_node_index(i + 1, j, nz),
      structured_node_index(i + 1, j + 1, nz),
      structured_node_index(i, j + 1, nz),
  };
  const int base = c * corner_stride;
  double wp = 0.0;
  double ws = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int n = nodes[k];
    wp -= force.corner_p_r[base + k] * v_r[n] +
          force.corner_p_z[base + k] * v_z[n];
    ws -= force.corner_sub_r[base + k] * v_r[n] +
          force.corner_sub_z[base + k] * v_z[n];
  }
  work.work_p[c] = wp;
  work.work_sub[c] = ws;
}

__global__ void compatible_work_multiblock_kernel(
    CompatibleWorkBuffers work,
    const CompatibleForceBuffers force,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells_total,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells_total || c >= work.n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    work.work_p[c] = 0.0;
    work.work_sub[c] = 0.0;
    return;
  }

  const int off = cell_node_csr_offsets[c];
  const int base = c * corner_stride;
  double wp = 0.0;
  double ws = 0.0;
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts == 3) {
    for (int k = 0; k < 3; ++k) {
      const int n = cell_node_csr_indices[off + k];
      wp -= force.corner_p_r[base + k] * v_r[n] +
            force.corner_p_z[base + k] * v_z[n];
      ws -= force.corner_sub_r[base + k] * v_r[n] +
            force.corner_sub_z[base + k] * v_z[n];
    }
  } else if (active_nverts >= 5) {
    for (int k = 0; k < active_nverts; ++k) {
      const int n = cell_node_csr_indices[off + k];
      wp -= force.corner_p_r[base + k] * v_r[n] +
            force.corner_p_z[base + k] * v_z[n];
      ws -= force.corner_sub_r[base + k] * v_r[n] +
            force.corner_sub_z[base + k] * v_z[n];
    }
  } else {
    for (int k = 0; k < 4; ++k) {
      const int n = cell_node_csr_indices[off + k];
      wp -= force.corner_p_r[base + k] * v_r[n] +
            force.corner_p_z[base + k] * v_z[n];
      ws -= force.corner_sub_r[base + k] * v_r[n] +
            force.corner_sub_z[base + k] * v_z[n];
    }
  }
  work.work_p[c] = wp;
  work.work_sub[c] = ws;
}

__global__ void scalar_av_work_structured_kernel(
    CompatibleWorkBuffers work,
    const CompatibleForceBuffers force,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= nr * nz || c >= work.n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      structured_node_index(i, j, nz),
      structured_node_index(i + 1, j, nz),
      structured_node_index(i + 1, j + 1, nz),
      structured_node_index(i, j + 1, nz),
  };
  const int base = c * corner_stride;
  double work_q = 0.0;
  for (int k = 0; k < 4; ++k) {
    work_q -= force.corner_q_r[base + k] * v_r[nodes[k]] +
              force.corner_q_z[base + k] * v_z[nodes[k]];
  }
  work.work_av[c] += work_q;
}

__device__ inline void add_signed_av_work(double* work_av,
                                          const int c,
                                          const double value) {
  if (c < 0) {
    return;
  }
  // SIGNED on purpose: the corrector evaluates this work with the exact
  // time-centered velocities, where small negative per-edge deposits are
  // the true KE pairing of the (velocity-dependent) AV force. Clamping
  // them non-negative biased the total-energy closure by a
  // truncation-ordered residual (G4 5-block: 5.2e-10 at Nc=16 vs 1.7e-16
  // with AV off).
  // Non-finite contributions and signed zeros deposit an explicit +0.0 so
  // the accumulator's +/-0 normalization matches the legacy clamp bitwise
  // at zero-force (dormant-boundary) edges; genuine negative work stays
  // SIGNED for the exact closure.
  const double finite_value = isfinite(value) ? value : 0.0;
  atomic_add_double(work_av + c, finite_value == 0.0 ? 0.0 : finite_value);
}

__device__ inline bool cell_is_active(const std::int8_t* hydro_active,
                                      const int c) {
  return c >= 0 && (hydro_active == nullptr || hydro_active[c] != 0);
}

__global__ void av_work_structured_kernel(CompatibleWorkBuffers work,
                                          const CompatibleForceBuffers force,
                                          const double* __restrict__ v_r,
                                          const double* __restrict__ v_z,
                                          const std::int8_t* __restrict__ hydro_active,
                                          const int nr,
                                          const int nz) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges || e >= force.n_edges) {
    return;
  }

  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  int cell_b = -1;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i + 1, j, nz);
    if (j > 0) {
      cell_a = i * nz + (j - 1);
    }
    if (j < nz) {
      cell_b = i * nz + j;
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i, j + 1, nz);
    if (i > 0) {
      cell_a = (i - 1) * nz + j;
    }
    if (i < nr) {
      cell_b = i * nz + j;
    }
  }

  const double dv_r = v_r[n1] - v_r[n0];
  const double dv_z = v_z[n1] - v_z[n0];
  const double w = -(force.edge_av_r[e] * dv_r + force.edge_av_z[e] * dv_z);
  const bool active_a = cell_is_active(hydro_active, cell_a);
  const bool active_b = cell_is_active(hydro_active, cell_b);
  if (active_a && active_b) {
    add_signed_av_work(work.work_av, cell_a, 0.5 * w);
    add_signed_av_work(work.work_av, cell_b, 0.5 * w);
  } else if (active_a) {
    add_signed_av_work(work.work_av, cell_a, w);
  } else if (active_b) {
    add_signed_av_work(work.work_av, cell_b, w);
  } else {
    return;
  }
}

__global__ void av_work_multiblock_kernel(
    CompatibleWorkBuffers work,
    const CompatibleForceBuffers force,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const int edge_offset,
    const int n_faces,
    const bool has_cell_b) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int e = edge_offset + f;
  if (e >= force.n_edges) {
    return;
  }

  const int c = face_cell_a[f];
  const int local = face_local[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int off = cell_node_csr_offsets[c];
  const int n0 = cell_node_csr_indices[off + corner0];
  const int n1 = cell_node_csr_indices[off + corner1];

  const double dv_r = v_r[n1] - v_r[n0];
  const double dv_z = v_z[n1] - v_z[n0];
  const double w = -(force.edge_av_r[e] * dv_r + force.edge_av_z[e] * dv_z);
  const bool active_a = cell_is_active(hydro_active, c);
  const int cell_b = has_cell_b ? face_cell_b[f] : -1;
  const bool active_b = cell_is_active(hydro_active, cell_b);
  if (has_cell_b && active_a && active_b) {
    add_signed_av_work(work.work_av, c, 0.5 * w);
    add_signed_av_work(work.work_av, cell_b, 0.5 * w);
  } else if (active_a) {
    add_signed_av_work(work.work_av, c, w);
  } else if (active_b) {
    add_signed_av_work(work.work_av, cell_b, w);
  }
}

template <typename T>
const T* raw_or_null(const thrust::device_vector<T>& values) {
  return values.empty() ? nullptr : thrust::raw_pointer_cast(values.data());
}

}  // namespace

bool compatible_force_work_enabled(const core::Config& cfg) {
  // csw_edge_csw98 runs the compatible path with or without subzonal
  // pressure (I1-B column A/B isolation); the certified csw_edge pairing
  // is unchanged.
  return cfg.main.dim == 2 &&
         ((cfg.numerics.hydro.av_model == core::AvModel::CswEdge &&
           cfg.numerics.hydro.subzonal_pressure_enabled) ||
          cfg.numerics.hydro.av_model == core::AvModel::CswEdgeCsw98);
}

bool compatible_force_work_buffers_requested(const core::Config& cfg) {
  return cfg.main.dim == 2 &&
         (cfg.numerics.hydro.subzonal_mass_enabled ||
          cfg.numerics.hydro.hourglass.enabled ||
          cfg.numerics.hydro.av_model != core::AvModel::ScalarVnrLegacy ||
          cfg.numerics.hydro.time_integration ==
              core::HydroTimeIntegration::MidpointV1);
}

std::size_t compatible_edge_force_count(const core::State& state,
                                        const core::Config& cfg) {
  if (cfg.main.dim != 2) {
    return 0U;
  }
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    const std::size_t exact =
        mb.unique_internal_faces.size() + mb.boundary_faces.size();
    if (exact > 0U) {
      return exact;
    }
  } else if (state.mesh.topo.nr > 0 && state.mesh.topo.nz > 0) {
    return static_cast<std::size_t>(state.mesh.topo.n_edges());
  }

  const std::size_t n_cells = state.rho.size();
  const std::size_t n_nodes = state.x_r.size();
  return (n_cells + n_nodes > 0U) ? (n_cells + n_nodes - 1U) : 0U;
}

void ensure_compatible_force_work_buffers(core::State& state,
                                          const core::Config& cfg,
                                          const bool force_requested) {
  if (!force_requested && !compatible_force_work_buffers_requested(cfg)) {
    return;
  }

  const std::size_t n_cells = state.rho.size();
  const std::size_t n_corners =
      n_cells * static_cast<std::size_t>(state.corner_stride);
  const std::size_t n_edges = compatible_edge_force_count(state, cfg);
  if (state.corner_force_p_r.size() != n_corners) {
    state.corner_force_p_r.reset(n_corners);
  }
  if (state.corner_force_p_z.size() != n_corners) {
    state.corner_force_p_z.reset(n_corners);
  }
  if (state.corner_force_sub_r.size() != n_corners) {
    state.corner_force_sub_r.reset(n_corners);
  }
  if (state.corner_force_sub_z.size() != n_corners) {
    state.corner_force_sub_z.reset(n_corners);
  }
  if (force_requested &&
      state.corner_force_q_r.size() != n_corners) {
    state.corner_force_q_r.reset(n_corners);
  }
  if (force_requested &&
      state.corner_force_q_z.size() != n_corners) {
    state.corner_force_q_z.reset(n_corners);
  }
  if (state.edge_force_av_r.size() != n_edges) {
    state.edge_force_av_r.reset(n_edges);
  }
  if (state.edge_force_av_z.size() != n_edges) {
    state.edge_force_av_z.reset(n_edges);
  }
  if (state.work_p_per_cell.size() != n_cells) {
    state.work_p_per_cell.reset(n_cells);
  }
  if (state.work_sub_per_cell.size() != n_cells) {
    state.work_sub_per_cell.reset(n_cells);
  }
  if (state.work_av_per_cell.size() != n_cells) {
    state.work_av_per_cell.reset(n_cells);
  }
}

void zero_compatible_force_work_buffers(core::State& state) {
  zero_array(state.corner_force_p_r.data(), state.corner_force_p_r.size(),
             "compatible force zero corner_force_p_r failed");
  zero_array(state.corner_force_p_z.data(), state.corner_force_p_z.size(),
             "compatible force zero corner_force_p_z failed");
  zero_array(state.corner_force_sub_r.data(), state.corner_force_sub_r.size(),
             "compatible force zero corner_force_sub_r failed");
  zero_array(state.corner_force_sub_z.data(), state.corner_force_sub_z.size(),
             "compatible force zero corner_force_sub_z failed");
  zero_array(state.corner_force_q_r.data(), state.corner_force_q_r.size(),
             "compatible force zero corner_force_q_r failed");
  zero_array(state.corner_force_q_z.data(), state.corner_force_q_z.size(),
             "compatible force zero corner_force_q_z failed");
  zero_array(state.edge_force_av_r.data(), state.edge_force_av_r.size(),
             "compatible force zero edge_force_av_r failed");
  zero_array(state.edge_force_av_z.data(), state.edge_force_av_z.size(),
             "compatible force zero edge_force_av_z failed");
  zero_array(state.work_p_per_cell.data(), state.work_p_per_cell.size(),
             "compatible work zero work_p_per_cell failed");
  zero_array(state.work_sub_per_cell.data(), state.work_sub_per_cell.size(),
             "compatible work zero work_sub_per_cell failed");
  zero_array(state.work_av_per_cell.data(), state.work_av_per_cell.size(),
             "compatible work zero work_av_per_cell failed");

  state.count_edge_compressive_edges_step = 0;
  state.max_corner_density_spread_step = 0.0;
}

void launch_assemble_pressure_corner_forces_2d(
    core::State& state,
    const core::CellField1D& cell_pressure,
    const core::CellField1D& svec_r,
    const core::CellField1D& svec_z,
    const std::int8_t* hydro_active) {
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(cell_pressure.size() == static_cast<std::size_t>(n_cells),
                "compatible pressure force requires pressure per cell");
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  TENRYU_ASSERT(svec_r.size() == n_corners &&
                    svec_z.size() == n_corners,
                "compatible pressure force requires configured Svec entries per cell");
  TENRYU_ASSERT(state.corner_force_p_r.size() == n_corners &&
                    state.corner_force_p_z.size() == n_corners,
                "compatible pressure force buffers are not allocated");
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
  const int blocks = (n_cells + 255) / 256;
  assemble_pressure_corner_forces_kernel<<<blocks, 256>>>(
      force_buffers(state),
      cell_pressure.data(),
      svec_r.data(),
      svec_z.data(),
      hydro_active,
      d_cell_nverts_ptr,
      state.corner_stride);
}

void launch_assemble_scalar_av_corner_forces_2d(
    core::State& state,
    const core::CellField1D& cell_q,
    const core::CellField1D& svec_r,
    const core::CellField1D& svec_z,
    const std::int8_t* hydro_active) {
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(cell_q.size() == static_cast<std::size_t>(n_cells),
                "midpoint scalar AV force requires Q per cell");
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  TENRYU_ASSERT(state.corner_force_q_r.size() == n_corners &&
                    state.corner_force_q_z.size() == n_corners,
                "midpoint scalar AV corner-force buffers are not allocated");
  const int blocks = (n_cells + 255) / 256;
  assemble_scalar_av_corner_forces_kernel<<<blocks, 256>>>(
      force_buffers(state), cell_q.data(), svec_r.data(), svec_z.data(),
      hydro_active, state.corner_stride);
}

void launch_sum_compatible_forces_to_nodes_2d(double* force_r,
                                              double* force_z,
                                              const std::int8_t* hydro_active,
                                              const core::State& state,
                                              const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  TENRYU_ASSERT(force_r != nullptr && force_z != nullptr,
                "compatible nodal force accumulation requires force buffers");
  cuda_check(cudaMemset(force_r, 0, static_cast<std::size_t>(n_nodes) * sizeof(double)),
             "compatible force zero nodal force_r failed");
  cuda_check(cudaMemset(force_z, 0, static_cast<std::size_t>(n_nodes) * sizeof(double)),
             "compatible force zero nodal force_z failed");

  auto& mutable_state = const_cast<core::State&>(state);
  const CompatibleForceBuffers force = force_buffers(mutable_state);
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_offsets.size() ==
                      static_cast<std::size_t>(n_nodes) + 1U,
                  "compatible multiblock force requires reverse CSR offsets");
    core::DeviceArray<std::uint8_t> d_cell_nverts;
    const std::uint8_t* d_cell_nverts_ptr =
        upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
    const int blocks_nodes = (n_nodes + 255) / 256;
    sum_corner_forces_multiblock_kernel<<<blocks_nodes, 256>>>(
        force_r,
        force_z,
        force,
        hydro_active,
        state.mesh.multiblock_reverse_csr_node_offsets.data(),
        state.mesh.multiblock_reverse_csr_node_cells.data(),
        state.mesh.multiblock_reverse_csr_node_corners.data(),
        n_nodes,
        state.corner_stride);
    const auto& mb = *state.mesh.topo.multiblock;
    const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
    const int n_boundary = static_cast<int>(mb.boundary_faces.size());
    if (n_internal > 0) {
      const int blocks = (n_internal + 255) / 256;
      sum_edge_forces_multiblock_kernel<<<blocks, 256>>>(
          force_r,
          force_z,
          force,
          raw_or_null(mb.d_unique_face_cell_a),
          raw_or_null(mb.d_unique_face_local_a),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts_ptr,
          0,
          n_internal);
    }
    if (n_boundary > 0) {
      const int blocks = (n_boundary + 255) / 256;
      sum_edge_forces_multiblock_kernel<<<blocks, 256>>>(
          force_r,
          force_z,
          force,
          raw_or_null(mb.d_boundary_face_cell),
          raw_or_null(mb.d_boundary_face_local),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts_ptr,
          n_internal,
          n_boundary);
    }
  } else {
    const int blocks_cells = (n_cells + 255) / 256;
    sum_corner_forces_structured_kernel<<<blocks_cells, 256>>>(
        force_r, force_z, force, hydro_active, state.mesh.topo.nr,
        state.mesh.topo.nz, state.corner_stride);
    const int n_edges = static_cast<int>(compatible_edge_force_count(state, cfg));
    const int blocks_edges = (n_edges + 255) / 256;
    sum_edge_forces_structured_kernel<<<blocks_edges, 256>>>(
        force_r, force_z, force, state.mesh.topo.nr, state.mesh.topo.nz);
  }
}

void launch_sum_scalar_av_corner_forces_to_nodes_2d(
    double* force_r,
    double* force_z,
    const std::int8_t* hydro_active,
    const core::State& state) {
  TENRYU_ASSERT(!state.mesh.topo.multiblock.has_value(),
                "midpoint scalar AV corner-force scatter is structured-only");
  const int n_cells = static_cast<int>(state.rho.size());
  const int blocks = (n_cells + 255) / 256;
  auto& mutable_state = const_cast<core::State&>(state);
  sum_scalar_av_corner_forces_structured_kernel<<<blocks, 256>>>(
      force_r, force_z, force_buffers(mutable_state), hydro_active,
      state.mesh.topo.nr, state.mesh.topo.nz, state.corner_stride);
}

void launch_compute_compatible_work_2d(core::State& state,
                                       const core::Config& cfg,
                                       const double* v_r,
                                       const double* v_z,
                                       const std::int8_t* hydro_active) {
  const int n_cells = static_cast<int>(state.rho.size());
  const CompatibleForceBuffers force = force_buffers(state);
  const CompatibleWorkBuffers work = work_buffers(state);
  TENRYU_ASSERT(work.work_p != nullptr && work.work_sub != nullptr &&
                    work.work_av != nullptr,
                "compatible work buffers are not allocated");
  zero_array(work.work_p, state.work_p_per_cell.size(),
             "compatible work zero pressure work failed");
  zero_array(work.work_sub, state.work_sub_per_cell.size(),
             "compatible work zero subzonal work failed");
  zero_array(work.work_av, state.work_av_per_cell.size(),
             "compatible work zero AV work failed");
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "compatible multiblock work requires cell-node CSR offsets");
    core::DeviceArray<std::uint8_t> d_cell_nverts;
    const std::uint8_t* d_cell_nverts_ptr =
        upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
    const int blocks_cells = (n_cells + 255) / 256;
    compatible_work_multiblock_kernel<<<blocks_cells, 256>>>(
        work,
        force,
        v_r,
        v_z,
        hydro_active,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts_ptr,
        n_cells,
        state.corner_stride);

    const auto& mb = *state.mesh.topo.multiblock;
    const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
    const int n_boundary = static_cast<int>(mb.boundary_faces.size());
    if (n_internal > 0) {
      const int blocks = (n_internal + 255) / 256;
      av_work_multiblock_kernel<<<blocks, 256>>>(
          work,
          force,
          v_r,
          v_z,
          raw_or_null(mb.d_unique_face_cell_a),
          raw_or_null(mb.d_unique_face_cell_b),
          raw_or_null(mb.d_unique_face_local_a),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts_ptr,
          hydro_active,
          0,
          n_internal,
          true);
    }
    if (n_boundary > 0) {
      const int blocks = (n_boundary + 255) / 256;
      av_work_multiblock_kernel<<<blocks, 256>>>(
          work,
          force,
          v_r,
          v_z,
          raw_or_null(mb.d_boundary_face_cell),
          nullptr,
          raw_or_null(mb.d_boundary_face_local),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts_ptr,
          hydro_active,
          n_internal,
          n_boundary,
          false);
    }
  } else {
    const int blocks_cells = (n_cells + 255) / 256;
    compatible_work_structured_kernel<<<blocks_cells, 256>>>(
        work, force, v_r, v_z, hydro_active, state.mesh.topo.nr,
        state.mesh.topo.nz, state.corner_stride);
    const int n_edges = static_cast<int>(compatible_edge_force_count(state, cfg));
    const int blocks_edges = (n_edges + 255) / 256;
    av_work_structured_kernel<<<blocks_edges, 256>>>(
        work, force, v_r, v_z, hydro_active, state.mesh.topo.nr,
        state.mesh.topo.nz);
  }
}

void launch_compute_scalar_av_compatible_work_2d(
    core::State& state,
    const double* v_r,
    const double* v_z,
    const std::int8_t* hydro_active) {
  TENRYU_ASSERT(!state.mesh.topo.multiblock.has_value(),
                "midpoint scalar AV compatible work is structured-only");
  const int n_cells = static_cast<int>(state.rho.size());
  const int blocks = (n_cells + 255) / 256;
  scalar_av_work_structured_kernel<<<blocks, 256>>>(
      work_buffers(state), force_buffers(state), v_r, v_z, hydro_active,
      state.mesh.topo.nr, state.mesh.topo.nz, state.corner_stride);
}

}  // namespace tenryu::hydro::compatible
