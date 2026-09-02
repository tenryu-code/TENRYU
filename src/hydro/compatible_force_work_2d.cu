#include "hydro/compatible_force_work_2d.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "hydro/compatible_av_tensor.cuh"
#include "hydro/hydro_multiblock_topology.cuh"
#include "hydro/pentagon_aws_condensation.cuh"
#include "hydro/pentagon_geometry.cuh"
#include "hydro/rz_area_weighted.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::compatible {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

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

__host__ __device__ inline void check_pentagon_aws_s_vector_closure(
    const double Sr[8],
    const double Sz[8]) {
  double sum_r = 0.0;
  double sum_z = 0.0;
  double sum_abs = 0.0;
  for (int k = 0; k < 5; ++k) {
    sum_r = sum_r + Sr[k];
    sum_z = sum_z + Sz[k];
    sum_abs = (sum_abs + fabs(Sr[k])) + fabs(Sz[k]);
  }
  const double rel =
      (fabs(sum_r) + fabs(sum_z)) / fmax(sum_abs, DBL_MIN);
  if (!(rel <= 128.0 * DBL_EPSILON)) {
#ifdef __CUDA_ARCH__
    __trap();  // pentagon AWS S-vector closure failed
#else
    ::tenryu::core::tenryu_abort(
        "rel <= 128.0 * DBL_EPSILON",
        "pentagon AWS S-vector closure failed",
        __FILE__, __LINE__);
#endif
  }
}

__host__ __device__ inline void check_aw_planar_s_vector_closure(
    const double* Sr,
    const double* Sz,
    const int nverts) {
  double sum_r = 0.0;
  double sum_z = 0.0;
  double sum_abs = 0.0;
  for (int k = 0; k < nverts; ++k) {
    sum_r = sum_r + Sr[k];
    sum_z = sum_z + Sz[k];
    sum_abs = (sum_abs + fabs(Sr[k])) + fabs(Sz[k]);
  }
  const double rel =
      (fabs(sum_r) + fabs(sum_z)) / fmax(sum_abs, DBL_MIN);
  if (!(rel <= 128.0 * DBL_EPSILON)) {
#ifdef __CUDA_ARCH__
    __trap();  // AW planar S-vector closure failed
#else
    ::tenryu::core::tenryu_abort(
        "rel <= 128.0 * DBL_EPSILON",
        "AW planar S-vector closure failed",
        __FILE__, __LINE__);
#endif
  }
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

CompatibleForceBuffers work_force_buffers(core::State& state,
                                          const core::Config& cfg) {
  CompatibleForceBuffers out = force_buffers(state);
  if (cfg.numerics.hydro.aw_compatible_force_work) {
    TENRYU_ASSERT(state.corner_force_p_rz_r.data() != nullptr &&
                      state.corner_force_p_rz_z.data() != nullptr,
                  "AW-compatible RZ pressure work buffers are not allocated");
    out.corner_p_r = state.corner_force_p_rz_r.data();
    out.corner_p_z = state.corner_force_p_rz_z.data();
  }
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

__global__ void assemble_pressure_corner_forces_ring_page_kernel(
    double* __restrict__ out_corner_p_r,
    double* __restrict__ out_corner_p_z,
    const double* __restrict__ cell_pressure,
    const double* __restrict__ svec_r,
    const double* __restrict__ svec_z,
    const std::int8_t* __restrict__ hydro_active,
    const mesh::CellRingPage* __restrict__ pages,
    const int n_pages,
    const int corner_stride) {
  const int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pidx >= n_pages) {
    return;
  }

  const mesh::CellRingPage page = pages[pidx];
  const int c = page.owner;
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  const double p = active ? cell_pressure[c] : 0.0;
  for (int s = 1; s <= static_cast<int>(page.owned_count); ++s) {
    const int ring_position =
        static_cast<int>(page.page_ordinal) * 14 + (s - 1);
    const int idx = c * corner_stride + ring_position;
    out_corner_p_r[idx] = p * svec_r[idx];
    out_corner_p_z[idx] = p * svec_z[idx];
  }
}

__global__ void assemble_pressure_corner_forces_ring_page_triangle_fixup_kernel(
    double* __restrict__ out_corner_p_r,
    double* __restrict__ out_corner_p_z,
    const double* __restrict__ cell_pressure,
    const double* __restrict__ svec_r,
    const double* __restrict__ svec_z,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells ||
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c) != 3) {
    return;
  }

  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  const double p = active ? cell_pressure[c] : 0.0;
  const int base = c * corner_stride;
  out_corner_p_r[base + 3] = p * svec_r[base + 3];
  out_corner_p_z[base + 3] = p * svec_z[base + 3];
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

__global__ void assemble_aw_pressure_corner_forces_structured_kernel(
    CompatibleForceBuffers force,
    double* __restrict__ corner_p_rz_r,
    double* __restrict__ corner_p_rz_z,
    const double* __restrict__ cell_pressure,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= force.n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  const double r[4] = {x_r[n00], x_r[n10], x_r[n11], x_r[n01]};
  const double z[4] = {x_z[n00], x_z[n10], x_z[n11], x_z[n01]};
  double planar_Sr[4];
  double planar_Sz[4];
  rz::aw_planar_corner_vectors(r, z, planar_Sr, planar_Sz);

  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  const double p = active ? cell_pressure[c] : 0.0;
  const int base = c * 4;
  // 2*pi*r_k*p*S^A_k equals the sum of the two incident
  // pi*p*r_k*N_edge endpoint forces.
  for (int k = 0; k < 4; ++k) {
    const double planar_force_r = p * planar_Sr[k];
    const double planar_force_z = p * planar_Sz[k];
    force.corner_p_r[base + k] = planar_force_r;
    force.corner_p_z[base + k] = planar_force_z;
    const double rz_weight = 2.0 * kPi * r[k];
    corner_p_rz_r[base + k] = rz_weight * planar_force_r;
    corner_p_rz_z[base + k] = rz_weight * planar_force_z;
  }
}

__global__ void assemble_aw_pressure_corner_forces_multiblock_kernel(
    CompatibleForceBuffers force,
    double* __restrict__ corner_p_rz_r,
    double* __restrict__ corner_p_rz_z,
    const double* __restrict__ cell_pressure,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= force.n_cells) {
    return;
  }
  const int off = cell_node_csr_offsets[c];
  double planar_Sr[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
  double planar_Sz[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts == 3) {
    const int n0 = cell_node_csr_indices[off + 0];
    const int n1 = cell_node_csr_indices[off + 1];
    const int n2 = cell_node_csr_indices[off + 2];
    const double r[3] = {x_r[n0], x_r[n1], x_r[n2]};
    const double z[3] = {x_z[n0], x_z[n1], x_z[n2]};
    rz::aw_planar_triangle_corner_vectors(r, z, planar_Sr, planar_Sz);
  } else if (active_nverts == 5) {
    const int n0 = cell_node_csr_indices[off + 0];
    const int n1 = cell_node_csr_indices[off + 1];
    const int n2 = cell_node_csr_indices[off + 2];
    const int n3 = cell_node_csr_indices[off + 3];
    const int n4 = cell_node_csr_indices[off + 4];
    const PentagonPoint x[5] = {
        {x_r[n0], x_z[n0]},
        {x_r[n1], x_z[n1]},
        {x_r[n2], x_z[n2]},
        {x_r[n3], x_z[n3]},
        {x_r[n4], x_z[n4]},
    };
    double Sr[5];
    double Sz[5];
    pentagon_aws_condensed_planar_s(x, Sr, Sz);
    for (int k = 0; k < 5; ++k) {
      planar_Sr[k] = Sr[k];
      planar_Sz[k] = Sz[k];
    }
    check_pentagon_aws_s_vector_closure(planar_Sr, planar_Sz);
  } else if (active_nverts == 4) {
    const int n0 = cell_node_csr_indices[off + 0];
    const int n1 = cell_node_csr_indices[off + 1];
    const int n2 = cell_node_csr_indices[off + 2];
    const int n3 = cell_node_csr_indices[off + 3];
    const double r[4] = {x_r[n0], x_r[n1], x_r[n2], x_r[n3]};
    const double z[4] = {x_z[n0], x_z[n1], x_z[n2], x_z[n3]};
    rz::aw_planar_corner_vectors(r, z, planar_Sr, planar_Sz);
  } else {
    // class-B whole-polygon consumer: multi-page cells unsupported until C3-final
    if (active_nverts > 14) {
#ifdef __CUDA_ARCH__
      __trap();
#else
      ::tenryu::core::tenryu_abort(
          "active_nverts <= 14",
          "class-B whole-polygon consumer: multi-page cells unsupported until "
          "C3-final",
          __FILE__, __LINE__);
#endif
    }
    double r[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    double z[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    for (int k = 0; k < active_nverts; ++k) {
      const int n = cell_node_csr_indices[off + k];
      r[k] = x_r[n];
      z[k] = x_z[n];
    }
    rz::aw_planar_polygon_corner_vectors(
        r, z, active_nverts, planar_Sr, planar_Sz);
    check_aw_planar_s_vector_closure(planar_Sr, planar_Sz, active_nverts);
  }

  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  const double p = active ? cell_pressure[c] : 0.0;
  const int base = c * corner_stride;
  for (int k = 0; k < corner_stride; ++k) {
    if (k < active_nverts) {
      const double planar_force_r = p * planar_Sr[k];
      const double planar_force_z = p * planar_Sz[k];
      force.corner_p_r[base + k] = planar_force_r;
      force.corner_p_z[base + k] = planar_force_z;
      const int n = cell_node_csr_indices[off + k];
      const double rz_weight = 2.0 * kPi * x_r[n];
      corner_p_rz_r[base + k] = rz_weight * planar_force_r;
      corner_p_rz_z[base + k] = rz_weight * planar_force_z;
    } else {
      force.corner_p_r[base + k] = 0.0;
      force.corner_p_z[base + k] = 0.0;
      corner_p_rz_r[base + k] = 0.0;
      corner_p_rz_z[base + k] = 0.0;
    }
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

__device__ inline bool cell_is_active(const std::int8_t* hydro_active,
                                      const int c) {
  return c >= 0 && (hydro_active == nullptr || hydro_active[c] != 0);
}

__device__ inline bool edge_rz_lift_alpha(const bool lift_enabled,
                                          const bool aw_rz_work,
                                          const double guard_ratio,
                                          const double r0,
                                          const double r1,
                                          double* alpha0,
                                          double* alpha1,
                                          double* rbar) {
  *alpha0 = 1.0;
  *alpha1 = 1.0;
  *rbar = 0.0;
  if (!lift_enabled || !aw_rz_work) {
    return false;
  }
  if (!(r0 > 0.0) || !(r1 > 0.0)) {
    return false;
  }
  const double edge_rbar = 0.5 * (r0 + r1);
  if (!(edge_rbar <= guard_ratio * fmin(r0, r1))) {
    return false;
  }
  *alpha0 = edge_rbar / r0;
  *alpha1 = edge_rbar / r1;
  *rbar = edge_rbar;
  return true;
}

__global__ void sum_edge_forces_structured_kernel(double* __restrict__ force_r,
                                                  double* __restrict__ force_z,
                                                  const CompatibleForceBuffers force,
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

  const bool active_a = cell_is_active(hydro_active, cell_a);
  const bool active_b = cell_is_active(hydro_active, cell_b);
  if (!active_a && !active_b) {
    return;
  }

  const double fr = force.edge_av_r[e];
  const double fz = force.edge_av_z[e];
  atomic_add_double(force_r + n0, -fr);
  atomic_add_double(force_z + n0, -fz);
  atomic_add_double(force_r + n1, +fr);
  atomic_add_double(force_z + n1, +fz);
}

__device__ inline bool multiblock_edge_force_contribution(
    const CompatibleForceBuffers force,
    const double* __restrict__ x_r,
    const bool csw_rz_lift_enabled,
    const bool aw_rz_work,
    const double csw_rz_lift_guard_ratio,
    const int* __restrict__ unique_face_cell_a,
    const int* __restrict__ unique_face_cell_b,
    const int* __restrict__ unique_face_local,
    const int* __restrict__ boundary_face_cell,
    const int* __restrict__ boundary_face_local,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const int n_internal,
    const int n_boundary,
    const int e,
    const std::int8_t end,
    double* contribution_r,
    double* contribution_z) {
  if (e >= force.n_edges) {
    return false;
  }

  const bool internal = e < n_internal;
  const int f = internal ? e : e - n_internal;
  if (f < 0 || (internal ? f >= n_internal : f >= n_boundary)) {
    return false;
  }
  const int c =
      internal ? unique_face_cell_a[f] : boundary_face_cell[f];
  const int local =
      internal ? unique_face_local[f] : boundary_face_local[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return false;
  }
  const int off = cell_node_csr_offsets[c];
  const int n0 = cell_node_csr_indices[off + corner0];
  const int n1 = cell_node_csr_indices[off + corner1];

  const bool active_a = cell_is_active(hydro_active, c);
  const int cell_b = internal ? unique_face_cell_b[f] : -1;
  const bool active_b = cell_is_active(hydro_active, cell_b);
  if (!active_a && !active_b) {
    return false;
  }

  const double fr = force.edge_av_r[e];
  const double fz = force.edge_av_z[e];
  double alpha0 = 1.0;
  double alpha1 = 1.0;
  double rbar = 0.0;
  if (edge_rz_lift_alpha(csw_rz_lift_enabled, aw_rz_work,
                         csw_rz_lift_guard_ratio, x_r[n0], x_r[n1],
                         &alpha0, &alpha1, &rbar)) {
    *contribution_r = end == 0 ? -alpha0 * fr : +alpha1 * fr;
    *contribution_z = end == 0 ? -alpha0 * fz : +alpha1 * fz;
  } else {
    *contribution_r = end == 0 ? -fr : +fr;
    *contribution_z = end == 0 ? -fz : +fz;
  }
  return true;
}

__global__ void sum_edge_forces_multiblock_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const CompatibleForceBuffers force,
    const double* __restrict__ x_r,
    const bool csw_rz_lift_enabled,
    const bool aw_rz_work,
    const double csw_rz_lift_guard_ratio,
    const int* __restrict__ unique_face_cell_a,
    const int* __restrict__ unique_face_cell_b,
    const int* __restrict__ unique_face_local,
    const int* __restrict__ boundary_face_cell,
    const int* __restrict__ boundary_face_local,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ node_edge_csr_offsets,
    const int* __restrict__ node_edge_csr_edges,
    const std::int8_t* __restrict__ node_edge_csr_end,
    const int n_internal,
    const int n_boundary,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }

  double fr = 0.0;
  double fz = 0.0;
  const int begin = node_edge_csr_offsets[n];
  const int end = node_edge_csr_offsets[n + 1];
  for (int k = begin; k < end; ++k) {
    double edge_fr = 0.0;
    double edge_fz = 0.0;
    if (multiblock_edge_force_contribution(
            force, x_r, csw_rz_lift_enabled, aw_rz_work,
            csw_rz_lift_guard_ratio, unique_face_cell_a,
            unique_face_cell_b, unique_face_local, boundary_face_cell,
            boundary_face_local, cell_node_csr_offsets,
            cell_node_csr_indices, cell_nverts, hydro_active, n_internal,
            n_boundary, node_edge_csr_edges[k], node_edge_csr_end[k],
            &edge_fr, &edge_fz)) {
      fr += edge_fr;
      fz += edge_fz;
    }
  }
  force_r[n] = force_r[n] + fr;
  force_z[n] = force_z[n] + fz;
}

__global__ void compatible_work_structured_kernel(
    CompatibleWorkBuffers work,
    const CompatibleForceBuffers force,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const bool aw_rz_work,
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
    // Subzonal work in the RZ measure, m_z*edot = -sum_p u_p . (2 pi r_p f_p^sub,planar); planar in csw mode.
    const double rzw = aw_rz_work ? 2.0 * kPi * x_r[n] : 1.0;
    ws -= rzw * (force.corner_sub_r[base + k] * v_r[n] +
                 force.corner_sub_z[base + k] * v_z[n]);
  }
  work.work_p[c] = wp;
  work.work_sub[c] = ws;
}

__global__ void compatible_work_multiblock_kernel(
    CompatibleWorkBuffers work,
    const CompatibleForceBuffers force,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const bool aw_rz_work,
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
  // Subzonal work in the RZ measure, m_z*edot = -sum_p u_p . (2 pi r_p f_p^sub,planar); planar in csw mode.
  if (active_nverts == 3) {
    for (int k = 0; k < 3; ++k) {
      const int n = cell_node_csr_indices[off + k];
      wp -= force.corner_p_r[base + k] * v_r[n] +
            force.corner_p_z[base + k] * v_z[n];
      const double rzw = aw_rz_work ? 2.0 * kPi * x_r[n] : 1.0;
      ws -= rzw * (force.corner_sub_r[base + k] * v_r[n] +
                   force.corner_sub_z[base + k] * v_z[n]);
    }
  } else if (active_nverts >= 5) {
    for (int k = 0; k < active_nverts; ++k) {
      const int n = cell_node_csr_indices[off + k];
      wp -= force.corner_p_r[base + k] * v_r[n] +
            force.corner_p_z[base + k] * v_z[n];
      const double rzw = aw_rz_work ? 2.0 * kPi * x_r[n] : 1.0;
      ws -= rzw * (force.corner_sub_r[base + k] * v_r[n] +
                   force.corner_sub_z[base + k] * v_z[n]);
    }
  } else {
    for (int k = 0; k < 4; ++k) {
      const int n = cell_node_csr_indices[off + k];
      wp -= force.corner_p_r[base + k] * v_r[n] +
            force.corner_p_z[base + k] * v_z[n];
      const double rzw = aw_rz_work ? 2.0 * kPi * x_r[n] : 1.0;
      ws -= rzw * (force.corner_sub_r[base + k] * v_r[n] +
                   force.corner_sub_z[base + k] * v_z[n]);
    }
  }
  work.work_p[c] = wp;
  work.work_sub[c] = ws;
}

struct CompatibleWorkRingPagePartial {
  double work_p;
  double work_sub;
};

__global__ void compatible_work_ring_page_partials_kernel(
    CompatibleWorkRingPagePartial* __restrict__ page_partials,
    const CompatibleForceBuffers force,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const bool aw_rz_work,
    const mesh::CellRingPage* __restrict__ pages,
    const int n_pages,
    const int corner_stride) {
  const int pidx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pidx >= n_pages) {
    return;
  }

  const mesh::CellRingPage page = pages[pidx];
  const int c = page.owner;
  double wp = 0.0;
  double ws = 0.0;
  // The CSR triangle, quad, and general branches use these same expressions
  // over exactly active_nverts corners, so one pagewise loop covers all shapes.
  for (int s = 1; s <= static_cast<int>(page.owned_count); ++s) {
    const int ring_position =
        static_cast<int>(page.page_ordinal) * 14 + (s - 1);
    const int idx = c * corner_stride + ring_position;
    const int n = page.slots[s];
    wp -= force.corner_p_r[idx] * v_r[n] + force.corner_p_z[idx] * v_z[n];
    const double rzw = aw_rz_work ? 2.0 * kPi * x_r[n] : 1.0;
    ws -= rzw * (force.corner_sub_r[idx] * v_r[n] +
                 force.corner_sub_z[idx] * v_z[n]);
  }
  page_partials[pidx] = {wp, ws};
}

__global__ void compatible_work_ring_page_finalize_kernel(
    CompatibleWorkBuffers work,
    const CompatibleWorkRingPagePartial* __restrict__ page_partials,
    const int* __restrict__ cell_page_begin,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells_total) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells_total || c >= work.n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    work.work_p[c] = 0.0;
    work.work_sub[c] = 0.0;
    return;
  }

  double wp = 0.0;
  double ws = 0.0;
  for (int p = cell_page_begin[c]; p < cell_page_begin[c + 1]; ++p) {
    const CompatibleWorkRingPagePartial partial = page_partials[p];
    wp += partial.work_p;
    ws += partial.work_sub;
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

__global__ void av_work_structured_kernel(CompatibleWorkBuffers work,
                                          const CompatibleForceBuffers force,
                                          const double* __restrict__ v_r,
                                          const double* __restrict__ v_z,
                                          const double* __restrict__ x_r,
                                          const bool aw_rz_work,
                                          const bool axisline_work_planar,
                                          const double axis_eps_cm,
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

  // Edge-AV RZ work pairing: -sum_p u_p . F_rz,p = -F_e . Delta(2 pi r v); an axis-aligned edge (r0=r1=0) deposits exactly zero RZ work while its planar momentum force still acts.
  double w0r = aw_rz_work ? 2.0 * kPi * x_r[n0] : 1.0;
  double w1r = aw_rz_work ? 2.0 * kPi * x_r[n1] : 1.0;
  // R-A 2026-08-20: axis-line edges book the planar pair power — the 2*pi*r pairing nulls the AV dissipation on-axis while the planar momentum force acts (Ledger A363).
  if (axisline_work_planar && aw_rz_work &&
      x_r[n0] <= axis_eps_cm && x_r[n1] <= axis_eps_cm) {
    w0r = 1.0;
    w1r = 1.0;
  }
  const double dv_r = w1r * v_r[n1] - w0r * v_r[n0];
  const double dv_z = w1r * v_z[n1] - w0r * v_z[n0];
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
    double* __restrict__ work_av_edge,
    const CompatibleForceBuffers force,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const bool aw_rz_work,
    const bool axisline_work_planar,
    const double axis_eps_cm,
    const bool csw_rz_lift_enabled,
    const double csw_rz_lift_guard_ratio,
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

  // Edge-AV RZ work pairing: -sum_p u_p . F_rz,p = -F_e . Delta(2 pi r v); an axis-aligned edge (r0=r1=0) deposits exactly zero RZ work while its planar momentum force still acts.
  double alpha0 = 1.0;
  double alpha1 = 1.0;
  double rbar = 0.0;
  double w0r = 0.0;
  double w1r = 0.0;
  if (edge_rz_lift_alpha(csw_rz_lift_enabled, aw_rz_work,
                         csw_rz_lift_guard_ratio, x_r[n0], x_r[n1],
                         &alpha0, &alpha1, &rbar)) {
    w0r = 2.0 * kPi * rbar;
    w1r = w0r;
  } else {
    w0r = aw_rz_work ? 2.0 * kPi * x_r[n0] : 1.0;
    w1r = aw_rz_work ? 2.0 * kPi * x_r[n1] : 1.0;
  }
  // R-A 2026-08-20: axis-line edges book the planar pair power — the 2*pi*r pairing nulls the AV dissipation on-axis while the planar momentum force acts (Ledger A363).
  if (axisline_work_planar && aw_rz_work &&
      x_r[n0] <= axis_eps_cm && x_r[n1] <= axis_eps_cm) {
    w0r = 1.0;
    w1r = 1.0;
  }
  const double dv_r = w1r * v_r[n1] - w0r * v_r[n0];
  const double dv_z = w1r * v_z[n1] - w0r * v_z[n0];
  const double w = -(force.edge_av_r[e] * dv_r + force.edge_av_z[e] * dv_z);
  const bool active_a = cell_is_active(hydro_active, c);
  const int cell_b = has_cell_b ? face_cell_b[f] : -1;
  const bool active_b = cell_is_active(hydro_active, cell_b);
  if (has_cell_b && active_a && active_b) {
    work_av_edge[2 * e] = 0.5 * w;
    work_av_edge[2 * e + 1] = 0.5 * w;
  } else if (active_a) {
    work_av_edge[2 * e] = w;
  } else if (active_b) {
    work_av_edge[2 * e + 1] = w;
  }
}

__global__ void gather_multiblock_signed_av_work_kernel(
    CompatibleWorkBuffers work,
    const double* __restrict__ work_av_edge,
    const int* __restrict__ cell_edge_offsets,
    const int* __restrict__ cell_edge_edges,
    const std::int8_t* __restrict__ cell_edge_side,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || c >= work.n_cells) {
    return;
  }
  double work_av = 0.0;
  for (int k = cell_edge_offsets[c]; k < cell_edge_offsets[c + 1]; ++k) {
    const int e = cell_edge_edges[k];
    const int side = static_cast<int>(cell_edge_side[k]);
    const double value = work_av_edge[2 * e + side];
    const double finite_value = isfinite(value) ? value : 0.0;
    work_av += finite_value == 0.0 ? 0.0 : finite_value;
  }
  const double finite_work_av = isfinite(work_av) ? work_av : 0.0;
  work.work_av[c] += finite_work_av == 0.0 ? 0.0 : finite_work_av;
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
         (cfg.numerics.hydro.aw_compatible_force_work ||
          (cfg.numerics.hydro.av_model == core::AvModel::CswEdge &&
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
  if (cfg.numerics.hydro.aw_compatible_force_work) {
    if (state.corner_force_p_rz_r.size() != n_corners) {
      state.corner_force_p_rz_r.reset(n_corners);
    }
    if (state.corner_force_p_rz_z.size() != n_corners) {
      state.corner_force_p_rz_z.reset(n_corners);
    }
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
  zero_array(state.corner_force_p_rz_r.data(), state.corner_force_p_rz_r.size(),
             "compatible force zero corner_force_p_rz_r failed");
  zero_array(state.corner_force_p_rz_z.data(), state.corner_force_p_rz_z.size(),
             "compatible force zero corner_force_p_rz_z failed");
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
    const std::int8_t* hydro_active,
    const core::Config& cfg) {
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
  core::DeviceArray<std::uint8_t> d_cell_nverts{
      "compatible_force_work:pressure_cell_nverts"};
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
  const int blocks = (n_cells + 255) / 256;
  if (cfg.numerics.hydro.aw_compatible_force_work) {
    TENRYU_ASSERT(state.corner_force_p_rz_r.size() == n_corners &&
                      state.corner_force_p_rz_z.size() == n_corners,
                  "AW-compatible RZ pressure work buffers are not allocated");
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                        static_cast<std::size_t>(n_cells) + 1U &&
                        state.mesh.multiblock_cell_node_csr_indices.size() ==
                            static_cast<std::size_t>(n_cells) *
                                static_cast<std::size_t>(state.corner_stride),
                    "AW-compatible pressure force requires cell-node CSR");
      core::DeviceArray<std::uint8_t> d_cell_nverts{
          "compatible_pressure_cell_nverts"};
      const std::uint8_t* const cell_nverts =
          upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
      assemble_aw_pressure_corner_forces_multiblock_kernel<<<blocks, 256>>>(
          force_buffers(state),
          state.corner_force_p_rz_r.data(),
          state.corner_force_p_rz_z.data(),
          cell_pressure.data(),
          state.x_r.data(),
          state.x_z.data(),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          cell_nverts,
          hydro_active,
          state.corner_stride);
    } else {
      assemble_aw_pressure_corner_forces_structured_kernel<<<blocks, 256>>>(
          force_buffers(state),
          state.corner_force_p_rz_r.data(),
          state.corner_force_p_rz_z.data(),
          cell_pressure.data(),
          state.x_r.data(),
          state.x_z.data(),
          hydro_active,
          state.mesh.topo.nz);
    }
    return;
  }

  const mesh::RingPageSet* const ring_pages = state.mesh.ring_pages;
  const char* const ring_page_forces_env =
      std::getenv("TENRYU_RING_PAGE_FORCES");
  const bool ring_page_forces_requested =
      ring_pages != nullptr && ring_pages->valid &&
      ring_page_forces_env != nullptr &&
      (std::strcmp(ring_page_forces_env, "1") == 0 ||
       std::strcmp(ring_page_forces_env, "shadow") == 0);
  const bool ring_page_shadow =
      ring_page_forces_requested &&
      std::strcmp(ring_page_forces_env, "shadow") == 0;
  bool ring_page_has_multiple_pages = false;
  if (ring_page_forces_requested) {
    ring_page_has_multiple_pages = std::any_of(
        ring_pages->pages.begin(), ring_pages->pages.end(),
        [](const mesh::CellRingPage& page) {
          return page.page_ordinal > 0U;
        });
  }
  // C3-w3 retains legacy-stride force storage, so multi-page cells refuse the
  // pagewise path for this step and fall back entirely to the CSR kernel.
  const bool use_ring_page_forces =
      ring_page_forces_requested && !ring_page_has_multiple_pages;

  mesh::CellRingPage* d_ring_pages = nullptr;
  double* d_ring_page_corner_p_r = nullptr;
  double* d_ring_page_corner_p_z = nullptr;
  int n_ring_pages = 0;
  if (use_ring_page_forces) {
    n_ring_pages = static_cast<int>(ring_pages->pages.size());
    const std::size_t ring_page_bytes =
        ring_pages->pages.size() * sizeof(mesh::CellRingPage);
    const std::size_t corner_bytes = n_corners * sizeof(double);
    d_ring_pages = static_cast<mesh::CellRingPage*>(
        core::device_scratch_acquire(
            "compatible_pressure:d_ring_pages", ring_page_bytes));
    d_ring_page_corner_p_r = static_cast<double*>(
        core::device_scratch_acquire(
            "compatible_pressure:d_ring_page_corner_p_r", corner_bytes));
    d_ring_page_corner_p_z = static_cast<double*>(
        core::device_scratch_acquire(
            "compatible_pressure:d_ring_page_corner_p_z", corner_bytes));
    cuda_check(cudaMemcpy(d_ring_pages, ring_pages->pages.data(),
                          ring_page_bytes, cudaMemcpyHostToDevice),
               "compatible pressure memcpy ring pages failed");
    // Full-buffer zeroing reproduces the CSR padding writes; the triangle
    // slot-3 product is restored below to preserve its signed-zero bytes.
    cuda_check(cudaMemset(d_ring_page_corner_p_r, 0, corner_bytes),
               "compatible pressure memset ring page corner_p_r failed");
    cuda_check(cudaMemset(d_ring_page_corner_p_z, 0, corner_bytes),
               "compatible pressure memset ring page corner_p_z failed");
  }

  assemble_pressure_corner_forces_kernel<<<blocks, 256>>>(
      force_buffers(state),
      cell_pressure.data(),
      svec_r.data(),
      svec_z.data(),
      hydro_active,
      d_cell_nverts_ptr,
      state.corner_stride);

  if (use_ring_page_forces) {
    const int page_blocks = (n_ring_pages + 255) / 256;
    assemble_pressure_corner_forces_ring_page_kernel<<<page_blocks, 256>>>(
        d_ring_page_corner_p_r,
        d_ring_page_corner_p_z,
        cell_pressure.data(),
        svec_r.data(),
        svec_z.data(),
        hydro_active,
        d_ring_pages,
        n_ring_pages,
        state.corner_stride);
    cuda_check(cudaGetLastError(),
               "compatible pressure ring page kernel launch failed");
    assemble_pressure_corner_forces_ring_page_triangle_fixup_kernel
        <<<blocks, 256>>>(
            d_ring_page_corner_p_r,
            d_ring_page_corner_p_z,
            cell_pressure.data(),
            svec_r.data(),
            svec_z.data(),
            hydro_active,
            d_cell_nverts_ptr,
            n_cells,
            state.corner_stride);
    cuda_check(cudaGetLastError(),
               "compatible pressure ring page triangle fixup launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "compatible pressure ring page kernel execution failed");

    if (ring_page_shadow) {
      std::vector<double> csr_corner_p_r(n_corners, 0.0);
      std::vector<double> csr_corner_p_z(n_corners, 0.0);
      std::vector<double> ring_page_corner_p_r(n_corners, 0.0);
      std::vector<double> ring_page_corner_p_z(n_corners, 0.0);
      const std::size_t corner_bytes = n_corners * sizeof(double);
      cuda_check(cudaMemcpy(csr_corner_p_r.data(),
                            state.corner_force_p_r.data(), corner_bytes,
                            cudaMemcpyDeviceToHost),
                 "compatible pressure memcpy CSR corner_p_r failed");
      cuda_check(cudaMemcpy(csr_corner_p_z.data(),
                            state.corner_force_p_z.data(), corner_bytes,
                            cudaMemcpyDeviceToHost),
                 "compatible pressure memcpy CSR corner_p_z failed");
      cuda_check(cudaMemcpy(ring_page_corner_p_r.data(),
                            d_ring_page_corner_p_r, corner_bytes,
                            cudaMemcpyDeviceToHost),
                 "compatible pressure memcpy ring page corner_p_r failed");
      cuda_check(cudaMemcpy(ring_page_corner_p_z.data(),
                            d_ring_page_corner_p_z, corner_bytes,
                            cudaMemcpyDeviceToHost),
                 "compatible pressure memcpy ring page corner_p_z failed");

      const auto field_matches =
          [n_cells](const char* const field,
                    const std::vector<double>& csr,
                    const std::vector<double>& pagewise,
                    const std::size_t width) {
            for (std::size_t c = 0; c < static_cast<std::size_t>(n_cells);
                 ++c) {
              const std::size_t offset = c * width;
              if (std::memcmp(csr.data() + offset,
                              pagewise.data() + offset,
                              width * sizeof(double)) != 0) {
                std::fprintf(
                    stderr,
                    "[ring_page_forces_shadow] MISMATCH field=%s cell=%zu\n",
                    field, c);
                return false;
              }
            }
            return true;
          };
      bool shadow_matches = true;
      if (!field_matches("corner_p_r", csr_corner_p_r,
                         ring_page_corner_p_r,
                         static_cast<std::size_t>(state.corner_stride))) {
        shadow_matches = false;
      }
      if (!field_matches("corner_p_z", csr_corner_p_z,
                         ring_page_corner_p_z,
                         static_cast<std::size_t>(state.corner_stride))) {
        shadow_matches = false;
      }
      if (!shadow_matches) {
        TENRYU_ASSERT(false, "ring page forces shadow divergence");
      }
      std::fprintf(stderr,
                   "[ring_page_forces_shadow] ok cells=%d pages=%zu\n",
                   n_cells, ring_pages->pages.size());
    } else {
      const std::size_t corner_bytes = n_corners * sizeof(double);
      cuda_check(cudaMemcpy(state.corner_force_p_r.data(),
                            d_ring_page_corner_p_r, corner_bytes,
                            cudaMemcpyDeviceToDevice),
                 "compatible pressure copy ring page corner_p_r failed");
      cuda_check(cudaMemcpy(state.corner_force_p_z.data(),
                            d_ring_page_corner_p_z, corner_bytes,
                            cudaMemcpyDeviceToDevice),
                 "compatible pressure copy ring page corner_p_z failed");
    }
  }
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
    core::DeviceArray<std::uint8_t> d_cell_nverts{
        "compatible_force_work:sum_cell_nverts"};
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
    const int n_edges = n_internal + n_boundary;
    if (n_edges > 0) {
      TENRYU_ASSERT(
          state.mesh.multiblock_node_edge_csr_offsets.size() ==
                  static_cast<std::size_t>(n_nodes) + 1U &&
              state.mesh.multiblock_node_edge_csr_edges.size() ==
                  2U * static_cast<std::size_t>(n_edges) &&
              state.mesh.multiblock_node_edge_csr_end.size() ==
                  2U * static_cast<std::size_t>(n_edges),
          "compatible multiblock edge force requires node-edge CSR");
      sum_edge_forces_multiblock_kernel<<<blocks_nodes, 256>>>(
          force_r, force_z, force, state.x_r.data(),
          cfg.numerics.hydro.csw_rz_lift_enabled,
          cfg.numerics.hydro.aw_compatible_force_work,
          cfg.numerics.hydro.csw_rz_lift_guard_ratio,
          raw_or_null(mb.d_unique_face_cell_a),
          raw_or_null(mb.d_unique_face_cell_b),
          raw_or_null(mb.d_unique_face_local_a),
          raw_or_null(mb.d_boundary_face_cell),
          raw_or_null(mb.d_boundary_face_local),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts_ptr, hydro_active,
          state.mesh.multiblock_node_edge_csr_offsets.data(),
          state.mesh.multiblock_node_edge_csr_edges.data(),
          state.mesh.multiblock_node_edge_csr_end.data(), n_internal,
          n_boundary, n_nodes);
    }
  } else {
    const int blocks_cells = (n_cells + 255) / 256;
    sum_corner_forces_structured_kernel<<<blocks_cells, 256>>>(
        force_r, force_z, force, hydro_active, state.mesh.topo.nr,
        state.mesh.topo.nz, state.corner_stride);
    const int n_edges = static_cast<int>(compatible_edge_force_count(state, cfg));
    const int blocks_edges = (n_edges + 255) / 256;
    sum_edge_forces_structured_kernel<<<blocks_edges, 256>>>(
        force_r, force_z, force, hydro_active, state.mesh.topo.nr,
        state.mesh.topo.nz);
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
  const CompatibleForceBuffers force = work_force_buffers(state, cfg);
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
  const bool tensor_av =
      cfg.numerics.hydro.av_model == core::AvModel::MimeticTensorV1;
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "compatible multiblock work requires cell-node CSR offsets");
    core::DeviceArray<std::uint8_t> d_cell_nverts{
        "compatible_force_work:work_cell_nverts"};
    const std::uint8_t* d_cell_nverts_ptr =
        upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
    const int blocks_cells = (n_cells + 255) / 256;

    const mesh::RingPageSet* const ring_pages = state.mesh.ring_pages;
    const char* const ring_page_forces_env =
        std::getenv("TENRYU_RING_PAGE_FORCES");
    const bool ring_page_forces_requested =
        ring_pages != nullptr && ring_pages->valid &&
        ring_page_forces_env != nullptr &&
        (std::strcmp(ring_page_forces_env, "1") == 0 ||
         std::strcmp(ring_page_forces_env, "shadow") == 0);
    const bool ring_page_shadow =
        ring_page_forces_requested &&
        std::strcmp(ring_page_forces_env, "shadow") == 0;
    bool ring_page_has_multiple_pages = false;
    if (ring_page_forces_requested) {
      ring_page_has_multiple_pages = std::any_of(
          ring_pages->pages.begin(), ring_pages->pages.end(),
          [](const mesh::CellRingPage& page) {
            return page.page_ordinal > 0U;
          });
    }
    // C3-w3 retains legacy-stride work inputs, so multi-page cells refuse the
    // pagewise path for this step and fall back entirely to the CSR kernel.
    const bool use_ring_page_forces =
        ring_page_forces_requested && !ring_page_has_multiple_pages;

    mesh::CellRingPage* d_ring_pages = nullptr;
    int* d_cell_page_begin = nullptr;
    CompatibleWorkRingPagePartial* d_ring_page_partials = nullptr;
    double* d_ring_page_work_p = nullptr;
    double* d_ring_page_work_sub = nullptr;
    int n_ring_pages = 0;
    if (use_ring_page_forces) {
      n_ring_pages = static_cast<int>(ring_pages->pages.size());
      std::vector<int> cell_page_begin(n_cells + 1U, n_ring_pages);
      int next_cell = 0;
      for (int p = 0; p < n_ring_pages; ++p) {
        const int owner =
            ring_pages->pages[static_cast<std::size_t>(p)].owner;
        while (next_cell <= owner) {
          cell_page_begin[static_cast<std::size_t>(next_cell)] = p;
          ++next_cell;
        }
      }
      while (next_cell <= n_cells) {
        cell_page_begin[static_cast<std::size_t>(next_cell)] = n_ring_pages;
        ++next_cell;
      }
      const std::size_t ring_page_bytes =
          ring_pages->pages.size() * sizeof(mesh::CellRingPage);
      const std::size_t cell_page_begin_bytes =
          (n_cells + 1U) * sizeof(int);
      const std::size_t cell_scalar_bytes =
          static_cast<std::size_t>(n_cells) * sizeof(double);
      d_ring_pages = static_cast<mesh::CellRingPage*>(
          core::device_scratch_acquire(
              "compatible_work:d_ring_pages", ring_page_bytes));
      d_cell_page_begin = static_cast<int*>(core::device_scratch_acquire(
          "compatible_work:d_cell_page_begin", cell_page_begin_bytes));
      d_ring_page_partials = static_cast<CompatibleWorkRingPagePartial*>(
          core::device_scratch_acquire(
              "compatible_work:d_ring_page_partials",
              ring_pages->pages.size() *
                  sizeof(CompatibleWorkRingPagePartial)));
      d_ring_page_work_p = static_cast<double*>(core::device_scratch_acquire(
          "compatible_work:d_ring_page_work_p", cell_scalar_bytes));
      d_ring_page_work_sub = static_cast<double*>(
          core::device_scratch_acquire(
              "compatible_work:d_ring_page_work_sub", cell_scalar_bytes));
      cuda_check(cudaMemcpy(d_ring_pages, ring_pages->pages.data(),
                            ring_page_bytes, cudaMemcpyHostToDevice),
                 "compatible work memcpy ring pages failed");
      cuda_check(cudaMemcpy(d_cell_page_begin, cell_page_begin.data(),
                            cell_page_begin_bytes, cudaMemcpyHostToDevice),
                 "compatible work memcpy cell page begin failed");
    }

    compatible_work_multiblock_kernel<<<blocks_cells, 256>>>(
        work,
        force,
        v_r,
        v_z,
        state.x_r.data(),
        cfg.numerics.hydro.aw_compatible_force_work,
        hydro_active,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts_ptr,
        n_cells,
        state.corner_stride);

    if (use_ring_page_forces) {
      const int page_blocks = (n_ring_pages + 255) / 256;
      compatible_work_ring_page_partials_kernel<<<page_blocks, 256>>>(
          d_ring_page_partials,
          force,
          v_r,
          v_z,
          state.x_r.data(),
          cfg.numerics.hydro.aw_compatible_force_work,
          d_ring_pages,
          n_ring_pages,
          state.corner_stride);
      cuda_check(cudaGetLastError(),
                 "compatible work ring page P1 launch failed");
      CompatibleWorkBuffers ring_page_work = work;
      ring_page_work.work_p = d_ring_page_work_p;
      ring_page_work.work_sub = d_ring_page_work_sub;
      compatible_work_ring_page_finalize_kernel<<<blocks_cells, 256>>>(
          ring_page_work,
          d_ring_page_partials,
          d_cell_page_begin,
          hydro_active,
          n_cells);
      cuda_check(cudaGetLastError(),
                 "compatible work ring page P2 launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "compatible work ring page kernel execution failed");

      if (ring_page_shadow) {
        std::vector<double> csr_work_p(static_cast<std::size_t>(n_cells), 0.0);
        std::vector<double> csr_work_sub(static_cast<std::size_t>(n_cells),
                                         0.0);
        std::vector<double> ring_page_work_p(
            static_cast<std::size_t>(n_cells), 0.0);
        std::vector<double> ring_page_work_sub(
            static_cast<std::size_t>(n_cells), 0.0);
        const std::size_t cell_scalar_bytes =
            static_cast<std::size_t>(n_cells) * sizeof(double);
        cuda_check(cudaMemcpy(csr_work_p.data(), work.work_p,
                              cell_scalar_bytes, cudaMemcpyDeviceToHost),
                   "compatible work memcpy CSR work_p failed");
        cuda_check(cudaMemcpy(csr_work_sub.data(), work.work_sub,
                              cell_scalar_bytes, cudaMemcpyDeviceToHost),
                   "compatible work memcpy CSR work_sub failed");
        cuda_check(cudaMemcpy(ring_page_work_p.data(), d_ring_page_work_p,
                              cell_scalar_bytes, cudaMemcpyDeviceToHost),
                   "compatible work memcpy ring page work_p failed");
        cuda_check(cudaMemcpy(ring_page_work_sub.data(), d_ring_page_work_sub,
                              cell_scalar_bytes, cudaMemcpyDeviceToHost),
                   "compatible work memcpy ring page work_sub failed");

        const auto field_matches =
            [n_cells](const char* const field,
                      const std::vector<double>& csr,
                      const std::vector<double>& pagewise) {
              for (std::size_t c = 0;
                   c < static_cast<std::size_t>(n_cells); ++c) {
                if (std::memcmp(csr.data() + c, pagewise.data() + c,
                                sizeof(double)) != 0) {
                  std::fprintf(
                      stderr,
                      "[ring_page_forces_shadow] MISMATCH field=%s cell=%zu\n",
                      field, c);
                  return false;
                }
              }
              return true;
            };
        bool shadow_matches = true;
        if (!field_matches("work_p", csr_work_p, ring_page_work_p)) {
          shadow_matches = false;
        }
        if (!field_matches("work_sub", csr_work_sub, ring_page_work_sub)) {
          shadow_matches = false;
        }
        if (!shadow_matches) {
          TENRYU_ASSERT(false, "ring page forces shadow divergence");
        }
        std::fprintf(stderr,
                     "[ring_page_forces_shadow] ok cells=%d pages=%zu\n",
                     n_cells, ring_pages->pages.size());
      } else {
        const std::size_t cell_scalar_bytes =
            static_cast<std::size_t>(n_cells) * sizeof(double);
        cuda_check(cudaMemcpy(work.work_p, d_ring_page_work_p,
                              cell_scalar_bytes, cudaMemcpyDeviceToDevice),
                   "compatible work copy ring page work_p failed");
        cuda_check(cudaMemcpy(work.work_sub, d_ring_page_work_sub,
                              cell_scalar_bytes, cudaMemcpyDeviceToDevice),
                   "compatible work copy ring page work_sub failed");
      }
    }

    const auto& mb = *state.mesh.topo.multiblock;
    const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
    const int n_boundary = static_cast<int>(mb.boundary_faces.size());
    const int n_edges = n_internal + n_boundary;
    double* d_work_av_edge = nullptr;
    if (!tensor_av && n_edges > 0) {
      TENRYU_ASSERT(force.n_edges == n_edges,
                    "compatible multiblock work edge buffer size mismatch");
      const std::size_t cell_edge_incidence_count =
          2U * static_cast<std::size_t>(n_internal) +
          static_cast<std::size_t>(n_boundary);
      TENRYU_ASSERT(state.mesh.multiblock_cell_edge_csr_offsets.size() ==
                        static_cast<std::size_t>(n_cells) + 1U,
                    "compatible multiblock work requires cell-edge CSR offsets");
      TENRYU_ASSERT(state.mesh.multiblock_cell_edge_csr_edges.size() ==
                            cell_edge_incidence_count &&
                        state.mesh.multiblock_cell_edge_csr_side.size() ==
                            cell_edge_incidence_count,
                    "compatible multiblock work requires cell-edge CSR entries");
      const std::size_t work_av_edge_bytes =
          2U * static_cast<std::size_t>(n_edges) * sizeof(double);
      d_work_av_edge = static_cast<double*>(core::device_scratch_acquire(
          "compatible_av_csw:work_av_edge", work_av_edge_bytes));
      cuda_check(cudaMemset(d_work_av_edge, 0, work_av_edge_bytes),
                 "compatible multiblock work zero work_av_edge failed");
    }
    if (!tensor_av && n_internal > 0) {
      const int blocks = (n_internal + 255) / 256;
      av_work_multiblock_kernel<<<blocks, 256>>>(
          d_work_av_edge,
          force,
          v_r,
          v_z,
          state.x_r.data(),
          cfg.numerics.hydro.aw_compatible_force_work,
          cfg.numerics.hydro.csw98_axisline_work_planar_enabled,
          cfg.numerics.axis_eps_cm,
          cfg.numerics.hydro.csw_rz_lift_enabled,
          cfg.numerics.hydro.csw_rz_lift_guard_ratio,
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
    if (!tensor_av && n_boundary > 0) {
      const int blocks = (n_boundary + 255) / 256;
      av_work_multiblock_kernel<<<blocks, 256>>>(
          d_work_av_edge,
          force,
          v_r,
          v_z,
          state.x_r.data(),
          cfg.numerics.hydro.aw_compatible_force_work,
          cfg.numerics.hydro.csw98_axisline_work_planar_enabled,
          cfg.numerics.axis_eps_cm,
          cfg.numerics.hydro.csw_rz_lift_enabled,
          cfg.numerics.hydro.csw_rz_lift_guard_ratio,
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
    if (!tensor_av && n_edges > 0) {
      gather_multiblock_signed_av_work_kernel<<<blocks_cells, 256>>>(
          work, d_work_av_edge,
          state.mesh.multiblock_cell_edge_csr_offsets.data(),
          state.mesh.multiblock_cell_edge_csr_edges.data(),
          state.mesh.multiblock_cell_edge_csr_side.data(), n_cells);
    }
  } else {
    const int blocks_cells = (n_cells + 255) / 256;
    compatible_work_structured_kernel<<<blocks_cells, 256>>>(
        work, force, v_r, v_z, state.x_r.data(),
        cfg.numerics.hydro.aw_compatible_force_work, hydro_active,
        state.mesh.topo.nr, state.mesh.topo.nz, state.corner_stride);
    if (!tensor_av) {
      const int n_edges =
          static_cast<int>(compatible_edge_force_count(state, cfg));
      const int blocks_edges = (n_edges + 255) / 256;
      av_work_structured_kernel<<<blocks_edges, 256>>>(
          work, force, v_r, v_z, state.x_r.data(),
          cfg.numerics.hydro.aw_compatible_force_work,
          cfg.numerics.hydro.csw98_axisline_work_planar_enabled,
          cfg.numerics.axis_eps_cm, hydro_active,
          state.mesh.topo.nr, state.mesh.topo.nz);
    }
  }
  if (tensor_av) {
    launch_compute_tensor_av_work_2d(state, cfg, v_r, v_z, hydro_active);
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
