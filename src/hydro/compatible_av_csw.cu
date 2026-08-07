#include "hydro/compatible_av_csw.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

#include "core/error.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/pole_angular_derefine.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::compatible {
namespace {

constexpr double kPi = 3.1415926535897932384626433832795028841971693993751;
constexpr double kTiny = 1.0e-300;
constexpr double kRatioTiny = 1.0e-30;

struct CswKernelParams {
  double c1 = 1.0;
  double c2 = 1.0;
  double gamma = 5.0 / 3.0;
  int limiter_enabled = 1;
};

struct CswEdgeAvCflWinner {
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  int block_id = -1;
  double r0 = 0.0;
  double z0 = 0.0;
  double r1 = 0.0;
  double z1 = 0.0;
  double dx = 0.0;
  double du = 0.0;
};

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
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline double atomic_min_double(double* address, const double val) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (val < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ inline bool finite_device(const double x) {
  return isfinite(x);
}

__device__ inline bool edge_dt_matches(const double edge_dt,
                                       const double target_dt) {
  constexpr double kRelTol = 1.0e-6;
  return isfinite(edge_dt) && isfinite(target_dt) &&
         fabs(edge_dt - target_dt) <=
             kRelTol * fmax(fabs(target_dt), kTiny);
}

__device__ inline bool cell_active(const std::int8_t* hydro_active,
                                   const int c) {
  return c >= 0 && (hydro_active == nullptr || hydro_active[c] != 0);
}

__device__ inline int structured_node_index(const int i,
                                            const int j,
                                            const int nz) {
  return i * (nz + 1) + j;
}

__device__ inline void local_face_corners(const int local,
                                          int* corner0,
                                          int* corner1);

__device__ inline void outward_face_corners(const int local,
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

__device__ inline bool outward_face_corners(const int active_nverts,
                                            const int local,
                                            int* corner0,
                                            int* corner1) {
  if (active_nverts >= 5) {
    return mesh::mesh_topo_active_local_face_corners(active_nverts, local,
                                                     corner0, corner1);
  }
  if (active_nverts != 3) {
    outward_face_corners(local, corner0, corner1);
    return true;
  }
  return mesh::mesh_topo_active_local_face_corners(active_nverts, local,
                                                   corner0, corner1);
}

__device__ inline void outward_face_corners(const int local,
                                            int* corner0,
                                            int* corner1) {
  if (local == 0) {
    *corner0 = 3;
    *corner1 = 0;
  } else if (local == 1) {
    *corner0 = 1;
    *corner1 = 2;
  } else if (local == 2) {
    *corner0 = 0;
    *corner1 = 1;
  } else {
    *corner0 = 2;
    *corner1 = 3;
  }
}

__device__ inline int opposite_face(const int local) {
  return (local == 0) ? 1 : (local == 1) ? 0 : (local == 2) ? 3 : 2;
}

__device__ inline int opposite_face(const int active_nverts, const int local) {
  return (active_nverts == 3) ? -1 : opposite_face(local);
}

__device__ inline void structured_cell_nodes(const int c,
                                             const int nz,
                                             int nodes[4]) {
  const int i = c / nz;
  const int j = c - i * nz;
  nodes[0] = structured_node_index(i, j, nz);
  nodes[1] = structured_node_index(i + 1, j, nz);
  nodes[2] = structured_node_index(i + 1, j + 1, nz);
  nodes[3] = structured_node_index(i, j + 1, nz);
}

__device__ inline void csr_cell_nodes(const int c,
                                      const int* __restrict__ offsets,
                                      const int* __restrict__ indices,
                                      int nodes[4]) {
  const int off = offsets[c];
  nodes[0] = indices[off + 0];
  nodes[1] = indices[off + 1];
  nodes[2] = indices[off + 2];
  nodes[3] = indices[off + 3];
}

__device__ inline void csr_cell_active_nodes(
    const int c,
    const int active_nverts,
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int nodes[mesh::kMeshTopoCellStorageSlotsMax]) {
  if (active_nverts == 3) {
    const int off = offsets[c];
    nodes[0] = indices[off + 0];
    nodes[1] = indices[off + 1];
    nodes[2] = indices[off + 2];
    return;
  }
  if (active_nverts == 4) {
    csr_cell_nodes(c, offsets, indices, nodes);
    return;
  }
  const int off = offsets[c];
  for (int k = 0; k < active_nverts; ++k) {
    nodes[k] = indices[off + k];
  }
}

__device__ inline int structured_face_neighbor(const int c,
                                               const int local,
                                               const int nr,
                                               const int nz) {
  const int i = c / nz;
  const int j = c - i * nz;
  if (local == 0) {
    return (i > 0) ? ((i - 1) * nz + j) : -1;
  }
  if (local == 1) {
    return (i + 1 < nr) ? ((i + 1) * nz + j) : -1;
  }
  if (local == 2) {
    return (j > 0) ? (i * nz + (j - 1)) : -1;
  }
  return (j + 1 < nz) ? (i * nz + (j + 1)) : -1;
}

__device__ inline int reciprocal_face_lookup(
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int cell,
    const int neighbor) {
  if (cell < 0 || neighbor < 0) {
    return -1;
  }
  const int off = face_adj_offsets[neighbor];
  const int end = face_adj_offsets[neighbor + 1];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, neighbor);
  for (int p = off; p < end; ++p) {
    if (!mesh::mesh_topo_local_face_is_active(active_nverts, p - off)) {
      continue;
    }
    if (face_adj_indices[p] == cell) {
      return p - off;
    }
  }
  return -1;
}

__device__ inline double harmonic_positive(const double a, const double b) {
  const bool good_a = a > 0.0 && finite_device(a);
  const bool good_b = b > 0.0 && finite_device(b);
  if (good_a && good_b && (a + b) > 0.0) {
    return 2.0 * a * b / (a + b);
  }
  if (good_a) {
    return a;
  }
  if (good_b) {
    return b;
  }
  return 0.0;
}

__device__ inline void edge_svec_from_cell(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nodes[4],
    const int local,
    double* s_r,
    double* s_z) {
  int c0 = 0;
  int c1 = 0;
  outward_face_corners(local, &c0, &c1);
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const double r0 = x_r[n0];
  const double z0 = x_z[n0];
  const double r1 = x_r[n1];
  const double z1 = x_z[n1];
  const double coeff = kPi * (r0 + r1);
  *s_r = coeff * (z1 - z0);
  *s_z = -coeff * (r1 - r0);
}

__device__ inline bool edge_svec_from_cell(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nodes[4],
    const int active_nverts,
    const int local,
    double* s_r,
    double* s_z) {
  int c0 = 0;
  int c1 = 0;
  if (!outward_face_corners(active_nverts, local, &c0, &c1)) {
    return false;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const double r0 = x_r[n0];
  const double z0 = x_z[n0];
  const double r1 = x_r[n1];
  const double z1 = x_z[n1];
  const double coeff = kPi * (r0 + r1);
  *s_r = coeff * (z1 - z0);
  *s_z = -coeff * (r1 - r0);
  return true;
}

__device__ inline double projected_edge_ratio(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int nodes[4],
    const int local,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double current_grad) {
  if (!(current_grad > kTiny) || !finite_device(current_grad)) {
    return 1.0;
  }
  int c0 = 0;
  int c1 = 0;
  local_face_corners(local, &c0, &c1);
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double proj_dx = dx_r * xhat_r + dx_z * xhat_z;
  const double denom = proj_dx * current_grad;
  const double scale = fmax(fabs(proj_dx * current_grad), 1.0);
  if (!(fabs(denom) > kRatioTiny * scale) || !finite_device(denom)) {
    return 1.0;
  }
  const double ratio = (du_r * uhat_r + du_z * uhat_z) / denom;
  return finite_device(ratio) ? ratio : 1.0;
}

__device__ inline double projected_edge_ratio(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int nodes[4],
    const int active_nverts,
    const int local,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double current_grad) {
  if (!(current_grad > kTiny) || !finite_device(current_grad)) {
    return 1.0;
  }
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, local, &c0, &c1)) {
    return 1.0;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double proj_dx = dx_r * xhat_r + dx_z * xhat_z;
  const double denom = proj_dx * current_grad;
  const double scale = fmax(fabs(proj_dx * current_grad), 1.0);
  if (!(fabs(denom) > kRatioTiny * scale) || !finite_device(denom)) {
    return 1.0;
  }
  const double ratio = (du_r * uhat_r + du_z * uhat_z) / denom;
  return finite_device(ratio) ? ratio : 1.0;
}

__device__ inline double limiter_from_ratios(const double r0,
                                             const double r1) {
  const double limited = fmin(fmin(1.0, 2.0 * r0),
                             fmin(2.0 * r1, 0.5 * (r0 + r1)));
  return fmax(0.0, limited);
}

__device__ inline double structured_limiter(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double current_grad) {
  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  const double r_inside =
      projected_edge_ratio(x_r, x_z, v_r, v_z, nodes, opposite_face(local),
                           xhat_r, xhat_z, uhat_r, uhat_z, current_grad);

  double r_outside = 1.0;
  const int n = structured_face_neighbor(c, local, nr, nz);
  if (cell_active(hydro_active, n)) {
    int neighbor_nodes[4] = {0, 0, 0, 0};
    structured_cell_nodes(n, nz, neighbor_nodes);
    r_outside = projected_edge_ratio(
        x_r, x_z, v_r, v_z, neighbor_nodes, opposite_face(local), xhat_r,
        xhat_z, uhat_r, uhat_z, current_grad);
  }
  return limiter_from_ratios(r_inside, r_outside);
}

__device__ inline double multiblock_limiter(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double current_grad) {
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  const int inside_local = opposite_face(active_nverts, local);
  const double r_inside =
      projected_edge_ratio(x_r, x_z, v_r, v_z, nodes, active_nverts,
                           inside_local,
                           xhat_r, xhat_z, uhat_r, uhat_z, current_grad);

  double r_outside = 1.0;
  const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
  const int reciprocal =
      reciprocal_face_lookup(face_adj_offsets, face_adj_indices, cell_nverts, c,
                             neighbor);
  if (cell_active(hydro_active, neighbor) && reciprocal >= 0) {
    const int neighbor_active_nverts =
        mesh::mesh_topo_cell_active_nverts(cell_nverts, neighbor);
    int neighbor_nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
    csr_cell_active_nodes(neighbor, neighbor_active_nverts, cell_node_offsets,
                          cell_node_indices, neighbor_nodes);
    r_outside = projected_edge_ratio(
        x_r, x_z, v_r, v_z, neighbor_nodes, neighbor_active_nverts,
        reciprocal, xhat_r, xhat_z, uhat_r, uhat_z, current_grad);
  }
  return limiter_from_ratios(r_inside, r_outside);
}

__device__ inline void apply_csw_contribution(
    double* __restrict__ work_av,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int n0,
    const int n1,
    const int c,
    const int neighbor,
    const int nodes[4],
    const int active_nverts,
    const int local,
    const double psi,
    const CswKernelParams params,
    double* edge_force_r,
    double* edge_force_z) {
  double s_r = 0.0;
  double s_z = 0.0;
  if (active_nverts == 3) {
    if (!edge_svec_from_cell(x_r, x_z, nodes, active_nverts, local, &s_r,
                             &s_z)) {
      return;
    }
  } else {
    edge_svec_from_cell(x_r, x_z, nodes, local, &s_r, &s_z);
  }

  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double du_mag = hypot(du_r, du_z);
  const double compression = du_r * s_r + du_z * s_z;
  if (compression <= 0.0 && finite_device(compression)) {
    atomicAdd(compressive_count, 1);
  }
  if (!(compression <= 0.0) || !(du_mag > kTiny) ||
      !finite_device(du_mag)) {
    return;
  }

  const double rho_e =
      harmonic_positive(rho[c], (neighbor >= 0) ? rho[neighbor] : rho[c]);
  const double cs_e =
      harmonic_positive(cs[c], (neighbor >= 0) ? cs[neighbor] : cs[c]);
  if (!(rho_e > 0.0) || !(cs_e >= 0.0)) {
    return;
  }

  const double limiter_scale = fmax(0.0, 1.0 - fmin(1.0, fmax(0.0, psi)));
  if (!(limiter_scale > 0.0)) {
    return;
  }
  const double c2g = params.c2 * (params.gamma + 1.0) * 0.25;
  const double wave =
      c2g * du_mag +
      sqrt(c2g * c2g * du_mag * du_mag + params.c1 * params.c1 * cs_e * cs_e);
  const double force_scale = rho_e * wave * limiter_scale * compression / du_mag;
  if (!finite_device(force_scale)) {
    return;
  }
  const double fr = force_scale * du_r;
  const double fz = force_scale * du_z;
  const double work = -(fr * du_r + fz * du_z);
  if (!(work >= 0.0) || !finite_device(work)) {
    atomicAdd(negative_work_count, 1);
  } else {
    atomic_add_double(work_av + c, work);
  }
  *edge_force_r += fr;
  *edge_force_z += fz;
}

__device__ inline void structured_cell_face_contribution(
    double* __restrict__ work_av,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const CswKernelParams params,
    double* edge_force_r,
    double* edge_force_z) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  int corner0 = 0;
  int corner1 = 0;
  local_face_corners(local, &corner0, &corner1);
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    double s_r = 0.0;
    double s_z = 0.0;
    edge_svec_from_cell(x_r, x_z, nodes, local, &s_r, &s_z);
    if (du_r * s_r + du_z * s_z <= 0.0) {
      atomicAdd(compressive_count, 1);
    }
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  const double current_grad = du_mag / dx_mag;
  const double psi = params.limiter_enabled != 0
                         ? structured_limiter(x_r, x_z, v_r, v_z, hydro_active,
                                              c, local, nr, nz, xhat_r, xhat_z,
                                              uhat_r, uhat_z, current_grad)
                         : 0.0;
  const int neighbor = structured_face_neighbor(c, local, nr, nz);
  apply_csw_contribution(work_av, compressive_count, negative_work_count, x_r,
                         x_z, v_r, v_z, rho, cs, n0, n1, c,
                         cell_active(hydro_active, neighbor) ? neighbor : -1,
                         nodes, mesh::kMeshTopoCellStorageSlots, local, psi,
                         params, edge_force_r, edge_force_z);
}

__device__ inline void multiblock_cell_face_contribution(
    double* __restrict__ work_av,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local,
    const CswKernelParams params,
    double* edge_force_r,
    double* edge_force_z) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    double s_r = 0.0;
    double s_z = 0.0;
    if (!edge_svec_from_cell(x_r, x_z, nodes, active_nverts, local, &s_r,
                             &s_z)) {
      return;
    }
    if (du_r * s_r + du_z * s_z <= 0.0) {
      atomicAdd(compressive_count, 1);
    }
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  const double current_grad = du_mag / dx_mag;
  const double psi =
      params.limiter_enabled != 0
          ? multiblock_limiter(x_r, x_z, v_r, v_z, hydro_active,
                               cell_node_offsets, cell_node_indices,
                               face_adj_offsets, face_adj_indices, cell_nverts,
                               c, local, xhat_r, xhat_z, uhat_r, uhat_z,
                               current_grad)
          : 0.0;
  const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
  apply_csw_contribution(work_av, compressive_count, negative_work_count, x_r,
                         x_z, v_r, v_z, rho, cs, n0, n1, c,
                         cell_active(hydro_active, neighbor) ? neighbor : -1,
                         nodes, active_nverts, local, psi, params,
                         edge_force_r, edge_force_z);
}

// ==== csw_edge_csw98: CSW98-faithful median-mesh edge AV (I1-B Stage-G W1).
// Force per CSW98 Eq. 20, f = q_Kur (1-psi) (dv_hat.S) dv_hat, with the
// median-mesh S of compatible_av_csw.cuh (RZ pi*(r_c+r_m) weighting) and
// the Eq. 18 logical-line continuation-edge limiter (van Leer psi, Eq. 12).
// q_Kur = rho_e * wave * |dv| with the same Kuropatenko wave speed and
// harmonic rho_e/cs_e as the old mode, so the force assembly below is the
// old expression with the projection vector replaced. Deposit convention
// inherited: buffer force F is applied -F at n0 / +F at n1 by
// sum_edge_forces_*; compression = dv.S < 0 makes force_scale negative, so
// the pair force opposes dv and the assembly work -F.dv is positive.

__device__ inline double csw98_continuation_ratio(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int na,
    const int nb,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double du_mag,
    const double dx_mag,
    int* miss) {
  const double dxn_r = x_r[nb] - x_r[na];
  const double dxn_z = x_z[nb] - x_z[na];
  const double dxn_mag = hypot(dxn_r, dxn_z);
  if (!(dxn_mag > kTiny) || !finite_device(dxn_mag)) {
    if (miss != nullptr) {
      atomicAdd(miss, 1);
    }
    return 1.0;
  }
  const double den_geom = dxn_r * xhat_r + dxn_z * xhat_z;
  if (!(fabs(den_geom) > kRatioTiny * dxn_mag)) {
    if (miss != nullptr) {
      atomicAdd(miss, 1);
    }
    return 1.0;
  }
  const double dun_r = v_r[nb] - v_r[na];
  const double dun_z = v_z[nb] - v_z[na];
  const double ratio =
      ((dun_r * uhat_r + dun_z * uhat_z) * dx_mag) / (den_geom * du_mag);
  return finite_device(ratio) ? ratio : 1.0;
}

__device__ inline double csw98_structured_limiter(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double du_mag,
    const double dx_mag) {
  const int i = c / nz;
  const int j = c - i * nz;
  int c_prev = -1;
  int c_next = -1;
  if (local == 2 || local == 3) {
    c_prev = (i > 0) ? (c - nz) : -1;
    c_next = (i + 1 < nr) ? (c + nz) : -1;
  } else {
    c_prev = (j > 0) ? (c - 1) : -1;
    c_next = (j + 1 < nz) ? (c + 1) : -1;
  }
  double r_prev = 1.0;
  double r_next = 1.0;
  int corner0 = 0;
  int corner1 = 0;
  local_face_corners(local, &corner0, &corner1);
  if (cell_active(hydro_active, c_prev)) {
    int nodes_nb[4] = {0, 0, 0, 0};
    structured_cell_nodes(c_prev, nz, nodes_nb);
    r_prev = csw98_continuation_ratio(
        x_r, x_z, v_r, v_z, nodes_nb[corner0], nodes_nb[corner1], xhat_r,
        xhat_z, uhat_r, uhat_z, du_mag, dx_mag, nullptr);
  }
  if (cell_active(hydro_active, c_next)) {
    int nodes_nb[4] = {0, 0, 0, 0};
    structured_cell_nodes(c_next, nz, nodes_nb);
    r_next = csw98_continuation_ratio(
        x_r, x_z, v_r, v_z, nodes_nb[corner0], nodes_nb[corner1], xhat_r,
        xhat_z, uhat_r, uhat_z, du_mag, dx_mag, nullptr);
  }
  return limiter_from_ratios(r_prev, r_next);
}

__device__ inline double csw98_multiblock_limiter(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local,
    const int nodes[4],
    const int active_nverts,
    const int n0,
    const int n1,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double du_mag,
    const double dx_mag,
    int* miss) {
  double ratios[2] = {1.0, 1.0};
  for (int e = 0; e < 2; ++e) {
    const int n = (e == 0) ? n0 : n1;
    int l_perp = -1;
    const int face_scan_count =
        (active_nverts <= 4) ? 4 : active_nverts;
    for (int lf = 0; lf < face_scan_count; ++lf) {
      if (lf == local) {
        continue;
      }
      int a0 = 0;
      int a1 = 0;
      if (!local_face_corners(active_nverts, lf, &a0, &a1)) {
        continue;
      }
      if (nodes[a0] == n || nodes[a1] == n) {
        l_perp = lf;
        break;
      }
    }
    if (l_perp < 0) {
      if (miss != nullptr) {
        atomicAdd(miss, 1);
      }
      continue;
    }
    const int nb = face_adj_indices[face_adj_offsets[c] + l_perp];
    if (!cell_active(hydro_active, nb)) {
      if (miss != nullptr) {
        atomicAdd(miss, 1);
      }
      continue;
    }
    const int nb_nverts =
        mesh::mesh_topo_cell_active_nverts(cell_nverts, nb);
    int nb_nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
    csr_cell_active_nodes(nb, nb_nverts, cell_node_offsets, cell_node_indices,
                          nb_nodes);
    int cont_a = -1;
    int cont_b = -1;
    const int nb_face_scan_count = (nb_nverts <= 4) ? 4 : nb_nverts;
    for (int lf = 0; lf < nb_face_scan_count; ++lf) {
      int b0 = 0;
      int b1 = 0;
      if (!local_face_corners(nb_nverts, lf, &b0, &b1)) {
        continue;
      }
      const int m0 = nb_nodes[b0];
      const int m1 = nb_nodes[b1];
      if (m0 != n && m1 != n) {
        continue;
      }
      if (face_adj_indices[face_adj_offsets[nb] + lf] == c) {
        continue;
      }
      cont_a = m0;
      cont_b = m1;
      break;
    }
    if (cont_a < 0) {
      if (miss != nullptr) {
        atomicAdd(miss, 1);
      }
      continue;
    }
    ratios[e] = csw98_continuation_ratio(
        x_r, x_z, v_r, v_z, cont_a, cont_b, xhat_r, xhat_z, uhat_r, uhat_z,
        du_mag, dx_mag, miss);
  }
  return limiter_from_ratios(ratios[0], ratios[1]);
}

__device__ inline void csw98_side_force(
    double* __restrict__ work_av,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int n0,
    const int n1,
    const int corner0,
    const int corner1,
    const int c,
    const int neighbor,
    const int nodes[4],
    const int active_nverts,
    const double psi,
    const CswKernelParams params,
    double* edge_force_r,
    double* edge_force_z) {
  const int nv = active_nverts;
  double cell_r[kCsw98MaxSideVecs];
  double cell_z[kCsw98MaxSideVecs];
  for (int k = 0; k < nv; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z, nv, corner0, corner1,
                              &s_r, &s_z)) {
    return;
  }
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double du_mag = hypot(du_r, du_z);
  const double compression = du_r * s_r + du_z * s_z;
  if (!(compression < 0.0) || !(du_mag > kTiny) ||
      !finite_device(compression) || !finite_device(du_mag)) {
    return;
  }
  atomicAdd(fire_count, 1);
  const double rho_e =
      harmonic_positive(rho[c], (neighbor >= 0) ? rho[neighbor] : rho[c]);
  const double cs_e =
      harmonic_positive(cs[c], (neighbor >= 0) ? cs[neighbor] : cs[c]);
  if (!(rho_e > 0.0) || !(cs_e >= 0.0)) {
    return;
  }
  const double limiter_scale = fmax(0.0, 1.0 - fmin(1.0, fmax(0.0, psi)));
  if (!(limiter_scale > 0.0)) {
    return;
  }
  const double c2g = params.c2 * (params.gamma + 1.0) * 0.25;
  const double wave =
      c2g * du_mag +
      sqrt(c2g * c2g * du_mag * du_mag + params.c1 * params.c1 * cs_e * cs_e);
  const double force_scale = rho_e * wave * limiter_scale * compression / du_mag;
  if (!finite_device(force_scale)) {
    return;
  }
  const double fr = force_scale * du_r;
  const double fz = force_scale * du_z;
  const double work = -(fr * du_r + fz * du_z);
  if (!(work >= 0.0) || !finite_device(work)) {
    atomicAdd(negative_work_count, 1);
  } else {
    atomic_add_double(work_av + c, work);
  }
  *edge_force_r += fr;
  *edge_force_z += fz;
}

__device__ inline void csw98_structured_cell_face_contribution(
    double* __restrict__ work_av,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const CswKernelParams params,
    double* edge_force_r,
    double* edge_force_z) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  int corner0 = 0;
  int corner1 = 0;
  local_face_corners(local, &corner0, &corner1);
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  const double psi =
      params.limiter_enabled != 0
          ? csw98_structured_limiter(x_r, x_z, v_r, v_z, hydro_active, c,
                                     local, nr, nz, xhat_r, xhat_z, uhat_r,
                                     uhat_z, du_mag, dx_mag)
          : 0.0;
  const int neighbor = structured_face_neighbor(c, local, nr, nz);
  csw98_side_force(work_av, fire_count, negative_work_count, x_r, x_z, v_r,
                   v_z, rho, cs, n0, n1, corner0, corner1, c,
                   cell_active(hydro_active, neighbor) ? neighbor : -1, nodes,
                   mesh::kMeshTopoCellStorageSlots, psi, params, edge_force_r,
                   edge_force_z);
}

__device__ inline void csw98_multiblock_cell_face_contribution(
    double* __restrict__ work_av,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local,
    const CswKernelParams params,
    double* edge_force_r,
    double* edge_force_z,
    int* limiter_miss) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  if (params.limiter_enabled != 0 && limiter_miss != nullptr) {
    // limiter evaluation counter (diag buffer slot [1]); reached only
    // when dx_mag/du_mag > tiny, i.e. the limiter actually runs.
    atomicAdd(limiter_miss + 1, 1);
  }
  const double psi =
      params.limiter_enabled != 0
          ? csw98_multiblock_limiter(
                x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
                cell_node_indices, face_adj_offsets, face_adj_indices,
                cell_nverts, c, local, nodes, active_nverts, n0, n1, xhat_r,
                xhat_z, uhat_r, uhat_z, du_mag, dx_mag, limiter_miss)
          : 0.0;
  const int neighbor = face_adj_indices[face_adj_offsets[c] + local];
  csw98_side_force(work_av, fire_count, negative_work_count, x_r, x_z, v_r,
                   v_z, rho, cs, n0, n1, corner0, corner1, c,
                   cell_active(hydro_active, neighbor) ? neighbor : -1, nodes,
                   active_nverts, psi, params, edge_force_r, edge_force_z);
}

__device__ inline void csw98_structured_side_cfl(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nr,
    const int nz,
    const CswKernelParams params,
    const double coefficient) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  int corner0 = 0;
  int corner1 = 0;
  local_face_corners(local, &corner0, &corner1);
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  double cell_r[4];
  double cell_z[4];
  for (int k = 0; k < mesh::kMeshTopoCellStorageSlots; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z, mesh::kMeshTopoCellStorageSlots,
                              corner0, corner1, &s_r, &s_z)) {
    return;
  }
  const double compression = du_r * s_r + du_z * s_z;
  if (!(compression < 0.0) || !finite_device(compression)) {
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  const double psi =
      params.limiter_enabled != 0
          ? csw98_structured_limiter(x_r, x_z, v_r, v_z, hydro_active, c,
                                     local, nr, nz, xhat_r, xhat_z, uhat_r,
                                     uhat_z, du_mag, dx_mag)
          : 0.0;
  const double s_mag = hypot(s_r, s_z);
  if (!(s_mag > 0.0)) {
    return;
  }
  const double proj = fabs(compression) / (du_mag * s_mag);
  const double psi_clamped = fmin(1.0, fmax(0.0, psi));
  const double du_eff = du_mag * fmax(0.0, 1.0 - psi_clamped) * proj;
  if (du_eff > kTiny && isfinite(du_eff) && dx_mag > 0.0) {
    atomic_min_double(min_dt, coefficient * dx_mag / du_eff);
  }
}

__device__ inline void csw98_multiblock_side_cfl(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local,
    const CswKernelParams params,
    const double coefficient,
    int* __restrict__ winner_index,
    const int candidate_index,
    const double target_dt) {
  if (!cell_active(hydro_active, c)) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double dx_mag = hypot(dx_r, dx_z);
  const double du_mag = hypot(du_r, du_z);
  if (!(dx_mag > kTiny) || !(du_mag > kTiny)) {
    return;
  }

  const int nv = active_nverts;
  double cell_r[kCsw98MaxSideVecs];
  double cell_z[kCsw98MaxSideVecs];
  for (int k = 0; k < nv; ++k) {
    cell_r[k] = x_r[nodes[k]];
    cell_z[k] = x_z[nodes[k]];
  }
  double s_r = 0.0;
  double s_z = 0.0;
  if (!csw98_c2_svec_for_side(cell_r, cell_z, nv, corner0, corner1,
                              &s_r, &s_z)) {
    return;
  }
  const double compression = du_r * s_r + du_z * s_z;
  if (!(compression < 0.0) || !finite_device(compression)) {
    return;
  }

  const double xhat_r = dx_r / dx_mag;
  const double xhat_z = dx_z / dx_mag;
  const double uhat_r = du_r / du_mag;
  const double uhat_z = du_z / du_mag;
  const double psi =
      params.limiter_enabled != 0
          ? csw98_multiblock_limiter(
                x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
                cell_node_indices, face_adj_offsets, face_adj_indices,
                cell_nverts, c, local, nodes, active_nverts, n0, n1, xhat_r,
                xhat_z, uhat_r, uhat_z, du_mag, dx_mag, nullptr)
          : 0.0;
  const double s_mag = hypot(s_r, s_z);
  if (!(s_mag > 0.0)) {
    return;
  }
  const double proj = fabs(compression) / (du_mag * s_mag);
  const double psi_clamped = fmin(1.0, fmax(0.0, psi));
  const double du_eff = du_mag * fmax(0.0, 1.0 - psi_clamped) * proj;
  if (du_eff > kTiny && isfinite(du_eff) && dx_mag > 0.0) {
    const double dt = coefficient * dx_mag / du_eff;
    atomic_min_double(min_dt, dt);
    if (winner_index != nullptr && edge_dt_matches(dt, target_dt)) {
      atomicMin(winner_index, candidate_index);
    }
  }
}

__global__ void csw98_structured_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const CswKernelParams params) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }

  double fr = 0.0;
  double fz = 0.0;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    if (j > 0) {
      csw98_structured_cell_face_contribution(
          work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z, rho,
          cs, hydro_active, i * nz + (j - 1), 3, nr, nz, params, &fr, &fz);
    }
    if (j < nz) {
      csw98_structured_cell_face_contribution(
          work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z, rho,
          cs, hydro_active, i * nz + j, 2, nr, nz, params, &fr, &fz);
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    if (i > 0) {
      csw98_structured_cell_face_contribution(
          work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z, rho,
          cs, hydro_active, (i - 1) * nz + j, 1, nr, nz, params, &fr, &fz);
    }
    if (i < nr) {
      csw98_structured_cell_face_contribution(
          work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z, rho,
          cs, hydro_active, i * nz + j, 0, nr, nz, params, &fr, &fz);
    }
  }
  edge_force_r[e] = fr;
  edge_force_z[e] = fz;
}

__global__ void csw98_multiblock_internal_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const CswKernelParams params,
    int* __restrict__ limiter_miss_count) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double fr = 0.0;
  double fz = 0.0;
  csw98_multiblock_cell_face_contribution(
      work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z, rho, cs,
      hydro_active, cell_node_offsets, cell_node_indices, face_adj_offsets,
      face_adj_indices, cell_nverts, face_cell_a[f], face_local_a[f], params,
      &fr, &fz, limiter_miss_count);
  csw98_multiblock_cell_face_contribution(
      work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z, rho, cs,
      hydro_active, cell_node_offsets, cell_node_indices, face_adj_offsets,
      face_adj_indices, cell_nverts, face_cell_b[f], face_local_b[f], params,
      &fr, &fz, limiter_miss_count);
  edge_force_r[f] = fr;
  edge_force_z[f] = fz;
}

__global__ void csw98_multiblock_boundary_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av,
    int* __restrict__ fire_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int edge_offset,
    const int n_faces,
    const CswKernelParams params,
    int* __restrict__ limiter_miss_count) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double fr = 0.0;
  double fz = 0.0;
  csw98_multiblock_cell_face_contribution(
      work_av, fire_count, negative_work_count, x_r, x_z, v_r, v_z, rho, cs,
      hydro_active, cell_node_offsets, cell_node_indices, face_adj_offsets,
      face_adj_indices, cell_nverts, face_cell[f], face_local[f], params, &fr,
      &fz, limiter_miss_count);
  edge_force_r[edge_offset + f] = fr;
  edge_force_z[edge_offset + f] = fz;
}

__global__ void csw_structured_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const CswKernelParams params) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }

  double fr = 0.0;
  double fz = 0.0;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    if (j > 0) {
      structured_cell_face_contribution(
          work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
          rho, cs, hydro_active, i * nz + (j - 1), 3, nr, nz, params, &fr,
          &fz);
    }
    if (j < nz) {
      structured_cell_face_contribution(
          work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
          rho, cs, hydro_active, i * nz + j, 2, nr, nz, params, &fr, &fz);
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    if (i > 0) {
      structured_cell_face_contribution(
          work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
          rho, cs, hydro_active, (i - 1) * nz + j, 1, nr, nz, params, &fr,
          &fz);
    }
    if (i < nr) {
      structured_cell_face_contribution(
          work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z,
          rho, cs, hydro_active, i * nz + j, 0, nr, nz, params, &fr, &fz);
    }
  }
  edge_force_r[e] = fr;
  edge_force_z[e] = fz;
}

__global__ void csw_multiblock_internal_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const CswKernelParams params) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double fr = 0.0;
  double fz = 0.0;
  multiblock_cell_face_contribution(
      work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z, rho,
      cs, hydro_active, cell_node_offsets, cell_node_indices, face_adj_offsets,
      face_adj_indices, cell_nverts, face_cell_a[f], face_local_a[f], params,
      &fr, &fz);
  multiblock_cell_face_contribution(
      work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z, rho,
      cs, hydro_active, cell_node_offsets, cell_node_indices, face_adj_offsets,
      face_adj_indices, cell_nverts, face_cell_b[f], face_local_b[f], params,
      &fr, &fz);
  edge_force_r[f] = fr;
  edge_force_z[f] = fz;
}

__global__ void csw_multiblock_boundary_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av,
    int* __restrict__ compressive_count,
    int* __restrict__ negative_work_count,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int edge_offset,
    const int n_faces,
    const CswKernelParams params) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double fr = 0.0;
  double fz = 0.0;
  multiblock_cell_face_contribution(
      work_av, compressive_count, negative_work_count, x_r, x_z, v_r, v_z, rho,
      cs, hydro_active, cell_node_offsets, cell_node_indices, face_adj_offsets,
      face_adj_indices, cell_nverts, face_cell[f], face_local[f], params, &fr,
      &fz);
  edge_force_r[edge_offset + f] = fr;
  edge_force_z[edge_offset + f] = fz;
}

__global__ void clip_negative_work_kernel(double* __restrict__ work_av,
                                          int* __restrict__ negative_count,
                                          const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double w = work_av[c];
  if (!(w >= 0.0) || !isfinite(w)) {
    work_av[c] = 0.0;
    atomicAdd(negative_count, 1);
  }
}

// I1-B pole tangential damper (pole-shear verdict step 7, experimental,
// env-gated default-off). The pole-column angular-shear mode is a radially
// alternating tangential slip of adjacent polar-shell node rows in the same
// near-pole angular column; it is AV-blind (div(u) ~ 0 along the slip).
// This damper applies, on every radial pole-column node pair (q,k)-(q+1,k)
// with column a in [1, M) of either pole, the pairwise tangential drag
//   F = -C_theta * rho_e * cs_e * |S_edge| * (du . t_hat) t_hat
// (du = v[n1]-v[n0], t_hat = in-plane unit normal of the edge, |S_edge| the
// same RZ edge area convention as the CSW AV svec). The force is accumulated
// into the edge AV force buffers in the canonical cell_a orientation, so the
// momentum kick (-F at n0, +F at n1 via sum_edge_forces) and the corrector's
// time-centered SIGNED work recompute inherit the exact compatible
// total-energy closure unchanged. Momentum is conserved pairwise by
// construction. Explicit-stability margin is ~1/C_theta acoustic steps
// (no extra dt limiter at the small default C_theta).
bool pole_damper_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

double pole_damper_ctheta() {
  static const double c_theta = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER_CTHETA");
    const double v = raw != nullptr ? std::atof(raw) : 0.0;
    return (std::isfinite(v) && v > 0.0 && v <= 10.0) ? v : 0.05;
  }();
  return c_theta;
}

int pole_damper_m() {
  static const int m = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER_M");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 1 ? v : 4;
  }();
  return m;
}

// Optional node-row band gate [q_lo, q_hi] for the damped pairs (both pair
// rows must lie inside). Default unrestricted. Motivation (v1 empirical):
// all-rows damping at C_theta=0.05 left the dense-band sawtooth essentially
// unchanged while its dissipated heat, deposited half/half into the
// near-pole LOW-MASS GAS sliver cells, produced a ~50 keV hot cell whose
// axis-margin dt collapse killed the run EARLIER (2.51 ns) than the no-op
// baseline (2.864 ns). A dense-shell-only band deposits the slip heat into
// rho ~ 0.25 cells where it is thermally negligible.
int pole_damper_q_lo() {
  static const int q_lo = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER_Q_LO");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : 0;
  }();
  return q_lo;
}

int pole_damper_q_hi() {
  static const int q_hi = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_DAMPER_Q_HI");
    const int v = raw != nullptr ? std::atoi(raw) : 0;
    return v > 0 ? v : (1 << 28);
  }();
  return q_hi;
}

__global__ void pole_damper_internal_force_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    double* __restrict__ work_av,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const int shell_node_begin,
    const int ntheta,
    const int n_node_rows,
    const int q_begin_active,
    const int m_columns,
    const int q_lo,
    const int q_hi,
    const double c_theta,
    int* __restrict__ damped_pair_count) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int cell_a = face_cell_a[f];
  const int cell_b = face_cell_b[f];
  // Canonical endpoint orientation: cell_a's corner pair, identical to
  // sum_edge_forces_multiblock_kernel / av_work_multiblock_kernel. The pair
  // formula is orientation-covariant, so consistency with the downstream
  // kernels is the only requirement.
  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, cell_a);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(cell_a, active_nverts, cell_node_offsets,
                        cell_node_indices, nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, face_local_a[f], &corner0, &corner1)) {
    return;
  }
  const int n0 = nodes[corner0];
  const int n1 = nodes[corner1];
  // Shell membership + same angular column + radially adjacent node rows.
  const int npc = ntheta + 1;
  const int s0 = n0 - shell_node_begin;
  const int s1 = n1 - shell_node_begin;
  if (s0 < 0 || s1 < 0) {
    return;
  }
  const int q0 = s0 / npc;
  const int k0 = s0 - q0 * npc;
  const int q1 = s1 / npc;
  const int k1 = s1 - q1 * npc;
  if (q0 >= n_node_rows || q1 >= n_node_rows || k0 != k1) {
    return;
  }
  const int dq = q1 - q0;
  if (dq != 1 && dq != -1) {
    return;
  }
  // Rows interior to the central macro cell are virtual; never touch them.
  if (q0 < q_begin_active || q1 < q_begin_active) {
    return;
  }
  // Optional row-band gate: both pair rows inside [q_lo, q_hi].
  const int q_pair_min = q0 < q1 ? q0 : q1;
  const int q_pair_max = q0 < q1 ? q1 : q0;
  if (q_pair_min < q_lo || q_pair_max > q_hi) {
    return;
  }
  // Near-pole columns a in [1, m_columns): north a = k, south a = ntheta - k.
  // a = 0 is the axis column itself (PAVA/axis-constrained), excluded.
  const int a_north = k0;
  const int a_south = ntheta - k0;
  const int a = a_north < a_south ? a_north : a_south;
  if (a < 1 || a >= m_columns) {
    return;
  }
  if (!cell_active(hydro_active, cell_a) || !cell_active(hydro_active, cell_b)) {
    return;
  }
  const double dx_r = x_r[n1] - x_r[n0];
  const double dx_z = x_z[n1] - x_z[n0];
  const double h = hypot(dx_r, dx_z);
  if (!(h > kTiny)) {
    return;
  }
  const double that_r = -dx_z / h;
  const double that_z = dx_r / h;
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  const double du_t = du_r * that_r + du_z * that_z;
  if (!finite_device(du_t) || du_t == 0.0) {
    return;
  }
  const double rho_e = harmonic_positive(rho[cell_a], rho[cell_b]);
  const double cs_e = harmonic_positive(cs[cell_a], cs[cell_b]);
  // Same RZ edge-area convention as edge_svec_from_cell: |S| = pi*(r0+r1)*h.
  const double area = kPi * (x_r[n0] + x_r[n1]) * h;
  const double coef = c_theta * rho_e * cs_e * area;
  if (!(coef > 0.0) || !finite_device(coef)) {
    return;
  }
  const double f_t = -coef * du_t;
  edge_force_r[f] += f_t * that_r;
  edge_force_z[f] += f_t * that_z;
  // Assembly-stage work estimate, same convention as apply_csw_contribution
  // (w = -F.du >= 0 here by construction); the corrector recomputes the
  // exact signed time-centered work from the summed edge forces.
  const double w = coef * du_t * du_t;
  if (finite_device(w) && w > 0.0) {
    atomic_add_double(work_av + cell_a, 0.5 * w);
    atomic_add_double(work_av + cell_b, 0.5 * w);
  }
  if (damped_pair_count != nullptr) {
    atomicAdd(damped_pair_count, 1);
  }
}

__device__ inline bool structured_edge_is_compressive(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int local,
    const int nz) {
  if (!cell_active(hydro_active, c)) {
    return false;
  }
  int nodes[4] = {0, 0, 0, 0};
  structured_cell_nodes(c, nz, nodes);
  int c0 = 0;
  int c1 = 0;
  local_face_corners(local, &c0, &c1);
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  double s_r = 0.0;
  double s_z = 0.0;
  edge_svec_from_cell(x_r, x_z, nodes, local, &s_r, &s_z);
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  return hypot(du_r, du_z) > kTiny && (du_r * s_r + du_z * s_z) <= 0.0;
}

__device__ inline bool multiblock_edge_is_compressive(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c,
    const int local) {
  if (!cell_active(hydro_active, c)) {
    return false;
  }
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, local, &c0, &c1)) {
    return false;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  double s_r = 0.0;
  double s_z = 0.0;
  if (active_nverts == 3) {
    if (!edge_svec_from_cell(x_r, x_z, nodes, active_nverts, local, &s_r,
                             &s_z)) {
      return false;
    }
  } else {
    edge_svec_from_cell(x_r, x_z, nodes, local, &s_r, &s_z);
  }
  const double du_r = v_r[n1] - v_r[n0];
  const double du_z = v_z[n1] - v_z[n0];
  return hypot(du_r, du_z) > kTiny && (du_r * s_r + du_z * s_z) <= 0.0;
}

__global__ void csw_structured_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double coefficient) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }

  int n0 = -1;
  int n1 = -1;
  bool compressive = false;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i + 1, j, nz);
    if (j > 0) {
      compressive = compressive ||
                    structured_edge_is_compressive(
                        x_r, x_z, v_r, v_z, hydro_active, i * nz + (j - 1), 3,
                        nz);
    }
    if (j < nz) {
      compressive = compressive ||
                    structured_edge_is_compressive(x_r, x_z, v_r, v_z,
                                                   hydro_active, i * nz + j, 2,
                                                   nz);
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    n0 = structured_node_index(i, j, nz);
    n1 = structured_node_index(i, j + 1, nz);
    if (i > 0) {
      compressive = compressive ||
                    structured_edge_is_compressive(
                        x_r, x_z, v_r, v_z, hydro_active, (i - 1) * nz + j, 1,
                        nz);
    }
    if (i < nr) {
      compressive = compressive ||
                    structured_edge_is_compressive(x_r, x_z, v_r, v_z,
                                                   hydro_active, i * nz + j, 0,
                                                   nz);
    }
  }
  if (!compressive) {
    return;
  }
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du)) {
    atomic_min_double(min_dt, coefficient * dx / du);
  }
}

__global__ void csw98_structured_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const CswKernelParams params,
    const double coefficient) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }

  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    if (j > 0) {
      csw98_structured_side_cfl(min_dt, x_r, x_z, v_r, v_z, hydro_active,
                                i * nz + (j - 1), 3, nr, nz, params,
                                coefficient);
    }
    if (j < nz) {
      csw98_structured_side_cfl(min_dt, x_r, x_z, v_r, v_z, hydro_active,
                                i * nz + j, 2, nr, nz, params, coefficient);
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    if (i > 0) {
      csw98_structured_side_cfl(min_dt, x_r, x_z, v_r, v_z, hydro_active,
                                (i - 1) * nz + j, 1, nr, nz, params,
                                coefficient);
    }
    if (i < nr) {
      csw98_structured_side_cfl(min_dt, x_r, x_z, v_r, v_z, hydro_active,
                                i * nz + j, 0, nr, nz, params, coefficient);
    }
  }
}

__device__ inline bool structured_cfl_edge_info(
    int* __restrict__ n0,
    int* __restrict__ n1,
    int* __restrict__ cell_a,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int e) {
  const int n_radial = nr * (nz + 1);
  bool compressive = false;
  *cell_a = -1;
  if (e < n_radial) {
    const int i = e / (nz + 1);
    const int j = e - i * (nz + 1);
    *n0 = structured_node_index(i, j, nz);
    *n1 = structured_node_index(i + 1, j, nz);
    if (j > 0 &&
        structured_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                       i * nz + (j - 1), 3, nz)) {
      compressive = true;
      *cell_a = i * nz + (j - 1);
    }
    if (j < nz &&
        structured_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                       i * nz + j, 2, nz)) {
      compressive = true;
      if (*cell_a < 0) {
        *cell_a = i * nz + j;
      }
    }
  } else {
    const int a = e - n_radial;
    const int i = a / nz;
    const int j = a - i * nz;
    *n0 = structured_node_index(i, j, nz);
    *n1 = structured_node_index(i, j + 1, nz);
    if (i > 0 &&
        structured_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                       (i - 1) * nz + j, 1, nz)) {
      compressive = true;
      *cell_a = (i - 1) * nz + j;
    }
    if (i < nr &&
        structured_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                       i * nz + j, 0, nz)) {
      compressive = true;
      if (*cell_a < 0) {
        *cell_a = i * nz + j;
      }
    }
  }
  return compressive;
}

__device__ inline void fill_csw_edge_av_cfl_winner(
    CswEdgeAvCflWinner* __restrict__ winner,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ cell_block_id,
    const int n0,
    const int n1,
    const int cell_a) {
  winner->n0 = n0;
  winner->n1 = n1;
  winner->cell_a = cell_a;
  winner->block_id =
      (cell_block_id != nullptr && cell_a >= 0) ? cell_block_id[cell_a] : -1;
  winner->r0 = x_r[n0];
  winner->z0 = x_z[n0];
  winner->r1 = x_r[n1];
  winner->z1 = x_z[n1];
  winner->dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  winner->du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
}

__global__ void csw_structured_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double coefficient,
    const double target_dt) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_edges = nr * (nz + 1) + (nr + 1) * nz;
  if (e >= n_edges) {
    return;
  }
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  if (!structured_cfl_edge_info(&n0, &n1, &cell_a, x_r, x_z, v_r, v_z,
                                hydro_active, nr, nz, e)) {
    return;
  }
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du) &&
      edge_dt_matches(coefficient * dx / du, target_dt)) {
    atomicMin(winner_index, e);
  }
}

__global__ void csw_structured_cfl_winner_values_kernel(
    CswEdgeAvCflWinner* __restrict__ winner,
    const int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz) {
  const int e = *winner_index;
  const int n_edges = nr * (nz + 1) + (nr + 1) * nz;
  if (e < 0 || e >= n_edges) {
    return;
  }
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  if (structured_cfl_edge_info(&n0, &n1, &cell_a, x_r, x_z, v_r, v_z,
                               hydro_active, nr, nz, e)) {
    fill_csw_edge_av_cfl_winner(winner, x_r, x_z, v_r, v_z, nullptr, n0, n1,
                                cell_a);
  }
}

__global__ void csw_multiblock_internal_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const double coefficient) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int ca = face_cell_a[f];
  const int la = face_local_a[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, ca);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(ca, active_nverts, cell_node_offsets,
                        cell_node_indices, nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, la, &c0, &c1)) {
    return;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const bool compressive =
      multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                     cell_node_offsets, cell_node_indices,
                                     cell_nverts, ca, la) ||
      multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                     cell_node_offsets, cell_node_indices,
                                     cell_nverts, face_cell_b[f],
                                     face_local_b[f]);
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (compressive && dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du)) {
    atomic_min_double(min_dt, coefficient * dx / du);
  }
}

__global__ void csw98_multiblock_internal_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const CswKernelParams params,
    const double coefficient) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      face_cell_a[f], face_local_a[f], params, coefficient, nullptr, -1,
      0.0);
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      face_cell_b[f], face_local_b[f], params, coefficient, nullptr, -1,
      0.0);
}

__device__ inline bool multiblock_internal_cfl_edge_info(
    int* __restrict__ n0,
    int* __restrict__ n1,
    int* __restrict__ cell_a,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int f) {
  const int ca = face_cell_a[f];
  const int la = face_local_a[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, ca);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(ca, active_nverts, cell_node_offsets,
                        cell_node_indices, nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, la, &c0, &c1)) {
    return false;
  }
  *n0 = nodes[c0];
  *n1 = nodes[c1];
  *cell_a = ca;
  return multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                        cell_node_offsets, cell_node_indices,
                                        cell_nverts, ca, la) ||
         multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                        cell_node_offsets, cell_node_indices,
                                        cell_nverts, face_cell_b[f],
                                        face_local_b[f]);
}

__global__ void csw_multiblock_internal_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const double coefficient,
    const double target_dt) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  if (!multiblock_internal_cfl_edge_info(
          &n0, &n1, &cell_a, x_r, x_z, v_r, v_z, hydro_active, face_cell_a,
          face_cell_b, face_local_a, face_local_b, cell_node_offsets,
          cell_node_indices, cell_nverts, f)) {
    return;
  }
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du) &&
      edge_dt_matches(coefficient * dx / du, target_dt)) {
    atomicMin(winner_index, f);
  }
}

__global__ void csw_multiblock_boundary_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const double coefficient) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const int c = face_cell[f];
  const int local = face_local[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, local, &c0, &c1)) {
    return;
  }
  const int n0 = nodes[c0];
  const int n1 = nodes[c1];
  const bool compressive =
      multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                     cell_node_offsets, cell_node_indices,
                                     cell_nverts, c, local);
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (compressive && dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du)) {
    atomic_min_double(min_dt, coefficient * dx / du);
  }
}

__global__ void csw98_multiblock_boundary_cfl_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const CswKernelParams params,
    const double coefficient) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      face_cell[f], face_local[f], params, coefficient, nullptr, -1,
      0.0);
}

__global__ void csw98_multiblock_internal_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_faces,
    const CswKernelParams params,
    const double coefficient,
    const double target_dt) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      face_cell_a[f], face_local_a[f], params, coefficient, winner_index,
      2 * f, target_dt);
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      face_cell_b[f], face_local_b[f], params, coefficient, winner_index,
      2 * f + 1, target_dt);
}

__global__ void csw98_multiblock_boundary_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const int* __restrict__ face_adj_offsets,
    const int* __restrict__ face_adj_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_internal_faces,
    const int n_boundary_faces,
    const CswKernelParams params,
    const double coefficient,
    const double target_dt) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_boundary_faces) {
    return;
  }
  csw98_multiblock_side_cfl(
      min_dt, x_r, x_z, v_r, v_z, hydro_active, cell_node_offsets,
      cell_node_indices, face_adj_offsets, face_adj_indices, cell_nverts,
      face_cell[f], face_local[f], params, coefficient, winner_index,
      2 * n_internal_faces + f, target_dt);
}

__global__ void csw98_multiblock_cfl_winner_values_kernel(
    CswEdgeAvCflWinner* __restrict__ winner,
    const int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const int* __restrict__ face_local_a,
    const int* __restrict__ face_local_b,
    const int* __restrict__ boundary_face_cell,
    const int* __restrict__ boundary_face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_block_id,
    const int n_internal_faces,
    const int n_boundary_faces) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const int candidate = *winner_index;
  const int n_internal_candidates = 2 * n_internal_faces;
  int cell = -1;
  int local = -1;
  if (candidate >= 0 && candidate < n_internal_candidates) {
    const int f = candidate / 2;
    if ((candidate & 1) == 0) {
      cell = face_cell_a[f];
      local = face_local_a[f];
    } else {
      cell = face_cell_b[f];
      local = face_local_b[f];
    }
  } else {
    const int f = candidate - n_internal_candidates;
    if (f < 0 || f >= n_boundary_faces) {
      return;
    }
    cell = boundary_face_cell[f];
    local = boundary_face_local[f];
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(cell, active_nverts, cell_node_offsets,
                        cell_node_indices, nodes);
  int corner0 = 0;
  int corner1 = 0;
  if (!local_face_corners(active_nverts, local, &corner0, &corner1)) {
    return;
  }
  fill_csw_edge_av_cfl_winner(
      winner, x_r, x_z, v_r, v_z, cell_block_id, nodes[corner0],
      nodes[corner1], cell);
}

__device__ inline bool multiblock_boundary_cfl_edge_info(
    int* __restrict__ n0,
    int* __restrict__ n1,
    int* __restrict__ cell_a,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int f) {
  const int c = face_cell[f];
  const int local = face_local[f];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[mesh::kMeshTopoCellStorageSlotsMax] = {0};
  csr_cell_active_nodes(c, active_nverts, cell_node_offsets, cell_node_indices,
                        nodes);
  int c0 = 0;
  int c1 = 0;
  if (!local_face_corners(active_nverts, local, &c0, &c1)) {
    return false;
  }
  *n0 = nodes[c0];
  *n1 = nodes[c1];
  *cell_a = c;
  return multiblock_edge_is_compressive(x_r, x_z, v_r, v_z, hydro_active,
                                        cell_node_offsets, cell_node_indices,
                                        cell_nverts, c, local);
}

__global__ void csw_multiblock_boundary_cfl_winner_index_kernel(
    int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ face_cell,
    const int* __restrict__ face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_internal,
    const int n_faces,
    const double coefficient,
    const double target_dt) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  if (!multiblock_boundary_cfl_edge_info(
          &n0, &n1, &cell_a, x_r, x_z, v_r, v_z, hydro_active, face_cell,
          face_local, cell_node_offsets, cell_node_indices, cell_nverts, f)) {
    return;
  }
  const double dx = hypot(x_r[n1] - x_r[n0], x_z[n1] - x_z[n0]);
  const double du = hypot(v_r[n1] - v_r[n0], v_z[n1] - v_z[n0]);
  if (dx > 0.0 && du > kTiny && isfinite(dx) && isfinite(du) &&
      edge_dt_matches(coefficient * dx / du, target_dt)) {
    atomicMin(winner_index, n_internal + f);
  }
}

__global__ void csw_multiblock_cfl_winner_values_kernel(
    CswEdgeAvCflWinner* __restrict__ winner,
    const int* __restrict__ winner_index,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ unique_face_cell_a,
    const int* __restrict__ unique_face_cell_b,
    const int* __restrict__ unique_face_local_a,
    const int* __restrict__ unique_face_local_b,
    const int* __restrict__ boundary_face_cell,
    const int* __restrict__ boundary_face_local,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_block_id,
    const int n_internal,
    const int n_boundary) {
  const int idx = *winner_index;
  int n0 = -1;
  int n1 = -1;
  int cell_a = -1;
  bool compressive = false;
  if (idx >= 0 && idx < n_internal) {
    compressive = multiblock_internal_cfl_edge_info(
        &n0, &n1, &cell_a, x_r, x_z, v_r, v_z, hydro_active, unique_face_cell_a,
        unique_face_cell_b, unique_face_local_a, unique_face_local_b,
        cell_node_offsets, cell_node_indices, cell_nverts, idx);
  } else if (idx >= n_internal && idx < n_internal + n_boundary) {
    compressive = multiblock_boundary_cfl_edge_info(
        &n0, &n1, &cell_a, x_r, x_z, v_r, v_z, hydro_active, boundary_face_cell,
        boundary_face_local, cell_node_offsets, cell_node_indices, cell_nverts,
        idx - n_internal);
  }
  if (compressive) {
    fill_csw_edge_av_cfl_winner(winner, x_r, x_z, v_r, v_z, cell_block_id, n0,
                                n1, cell_a);
  }
}

template <typename T>
const T* raw_or_null(const thrust::device_vector<T>& values) {
  return values.empty() ? nullptr : thrust::raw_pointer_cast(values.data());
}

std::int8_t* upload_hydro_active_or_null(const core::State& state,
                                         const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<std::int8_t> active;
  if (!state.hydro_active.empty()) {
    TENRYU_ASSERT(state.hydro_active.size() == static_cast<std::size_t>(n_cells),
                  "CSW edge AV hydro_active size mismatch");
    active = state.hydro_active;
  }
  core::State& mutable_state = const_cast<core::State&>(state);
  if (central_pseudo_core::configured(cfg)) {
    central_pseudo_core::ensure_built(mutable_state, cfg);
  }
  pole_angular_derefine::ensure_built(mutable_state, cfg);
  const auto apply_inactive = [&](const std::vector<std::uint8_t>& inactive) {
    if (inactive.size() != static_cast<std::size_t>(n_cells)) {
      return;
    }
    if (active.empty()) {
      active.assign(static_cast<std::size_t>(n_cells), 1);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (inactive[static_cast<std::size_t>(c)] != 0U) {
        active[static_cast<std::size_t>(c)] = 0;
      }
    }
  };
  apply_inactive(mutable_state.central_pseudo_core.inactive_member_mask);
  apply_inactive(mutable_state.pole_angular_derefine.inactive_member_mask);
  if (active.empty()) {
    return nullptr;
  }
  std::int8_t* d_active = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_active),
                        active.size() * sizeof(std::int8_t)),
             "CSW edge AV: cudaMalloc hydro_active failed");
  cuda_check(cudaMemcpy(d_active, active.data(),
                        active.size() * sizeof(std::int8_t),
                        cudaMemcpyHostToDevice),
             "CSW edge AV: cudaMemcpy hydro_active failed");
  return d_active;
}

bool has_nonquad_cell_nverts(const std::vector<std::uint8_t>& cell_nverts,
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
    const core::State& state) {
  const int n_cells = static_cast<int>(state.rho.size());
  if (!has_nonquad_cell_nverts(state.mesh.cell_nverts, n_cells)) {
    return nullptr;
  }
  d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
  d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
  return d_cell_nverts.data();
}

std::string format_csw_scientific(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << value;
  return oss.str();
}

}  // namespace

void launch_compute_csw_edge_av_2d(core::State& state,
                                   const core::Config& cfg,
                                   const core::CellField1D& cell_cs,
                                   const double* v_r,
                                   const double* v_z,
                                   const std::int8_t* hydro_active) {
  if (cfg.main.dim != 2 ||
      (cfg.numerics.hydro.av_model != core::AvModel::CswEdge &&
       cfg.numerics.hydro.av_model != core::AvModel::CswEdgeCsw98)) {
    return;
  }
  const bool use_csw98 =
      cfg.numerics.hydro.av_model == core::AvModel::CswEdgeCsw98;
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  TENRYU_ASSERT(cell_cs.size() == state.rho.size(),
                "CSW edge AV requires cell sound speed per cell");
  TENRYU_ASSERT(state.edge_force_av_r.size() == state.edge_force_av_z.size(),
                "CSW edge AV force buffer size mismatch");
  TENRYU_ASSERT(state.work_av_per_cell.size() == state.rho.size(),
                "CSW edge AV work buffer is not allocated");
  TENRYU_ASSERT(state.mesh.topo.n_cells == n_cells,
                "CSW edge AV topo/state cell-size mismatch");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == n_nodes,
                "CSW edge AV topo/state node-size mismatch");

  CswKernelParams params;
  params.c1 = cfg.numerics.hydro.av_linear;
  params.c2 = cfg.numerics.hydro.av_quadratic;
  params.gamma = cfg.materials.materials.empty()
                     ? (5.0 / 3.0)
                     : cfg.materials.materials.front().ideal_gas_gamma;
  params.limiter_enabled = cfg.numerics.hydro.csw_limiter_enabled ? 1 : 0;

  if (!state.edge_force_av_r.empty()) {
    cuda_check(cudaMemset(state.edge_force_av_r.data(), 0,
                          state.edge_force_av_r.size() * sizeof(double)),
               "CSW edge AV: zero edge_force_av_r failed");
    cuda_check(cudaMemset(state.edge_force_av_z.data(), 0,
                          state.edge_force_av_z.size() * sizeof(double)),
               "CSW edge AV: zero edge_force_av_z failed");
  }
  if (!state.work_av_per_cell.empty()) {
    cuda_check(cudaMemset(state.work_av_per_cell.data(), 0,
                          state.work_av_per_cell.size() * sizeof(double)),
               "CSW edge AV: zero work_av_per_cell failed");
  }

  int* d_compressive_count = nullptr;
  int* d_negative_work_count = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_compressive_count), sizeof(int)),
             "CSW edge AV: cudaMalloc compressive count failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_negative_work_count), sizeof(int)),
             "CSW edge AV: cudaMalloc negative work count failed");
  cuda_check(cudaMemset(d_compressive_count, 0, sizeof(int)),
             "CSW edge AV: zero compressive count failed");
  cuda_check(cudaMemset(d_negative_work_count, 0, sizeof(int)),
             "CSW edge AV: zero negative work count failed");

  int* d_limiter_miss = nullptr;
  static bool csw98_limiter_diag_logged = false;
  const bool csw98_limiter_diag =
      use_csw98 && !csw98_limiter_diag_logged &&
      std::getenv("TENRYU_CSW98_LIMITER_DIAG") != nullptr;
  if (csw98_limiter_diag) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_limiter_miss),
                          2 * sizeof(int)),
               "csw98 AV: cudaMalloc limiter diag counters failed");
    cuda_check(cudaMemset(d_limiter_miss, 0, 2 * sizeof(int)),
               "csw98 AV: zero limiter diag counters failed");
  }

  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "CSW edge AV multiblock topology metadata missing");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "CSW edge AV requires multiblock cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "CSW edge AV requires multiblock cell-node CSR indices");
    const auto& mb = *state.mesh.topo.multiblock;
    TENRYU_ASSERT(mb.face_adj_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "CSW edge AV requires face adjacency CSR offsets");
    TENRYU_ASSERT(mb.face_adj_csr_indices.size() >=
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "CSW edge AV requires face adjacency CSR indices");
    const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
    const int n_boundary = static_cast<int>(mb.boundary_faces.size());
    TENRYU_ASSERT(state.edge_force_av_r.size() ==
                      static_cast<std::size_t>(n_internal + n_boundary),
                  "CSW edge AV multiblock edge buffer size mismatch");
    thrust::device_vector<int> d_face_adj_offsets(mb.face_adj_csr_offsets.begin(),
                                                  mb.face_adj_csr_offsets.end());
    thrust::device_vector<int> d_face_adj_indices(mb.face_adj_csr_indices.begin(),
                                                  mb.face_adj_csr_indices.end());
    core::DeviceArray<std::uint8_t> d_cell_nverts;
    const std::uint8_t* d_cell_nverts_ptr =
        upload_cell_nverts_if_nonquad(d_cell_nverts, state);
    if (n_internal > 0) {
      const int blocks = (n_internal + 255) / 256;
      if (use_csw98) {
        csw98_multiblock_internal_force_kernel<<<blocks, 256>>>(
            state.edge_force_av_r.data(),
            state.edge_force_av_z.data(),
            state.work_av_per_cell.data(),
            d_compressive_count,
            d_negative_work_count,
            state.x_r.data(),
            state.x_z.data(),
            v_r,
            v_z,
            state.rho.data(),
            cell_cs.data(),
            hydro_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(d_face_adj_offsets),
            raw_or_null(d_face_adj_indices),
            d_cell_nverts_ptr,
            n_internal,
            params,
            d_limiter_miss);
      } else {
        csw_multiblock_internal_force_kernel<<<blocks, 256>>>(
            state.edge_force_av_r.data(),
            state.edge_force_av_z.data(),
            state.work_av_per_cell.data(),
            d_compressive_count,
            d_negative_work_count,
            state.x_r.data(),
            state.x_z.data(),
            v_r,
            v_z,
            state.rho.data(),
            cell_cs.data(),
            hydro_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(d_face_adj_offsets),
            raw_or_null(d_face_adj_indices),
            d_cell_nverts_ptr,
            n_internal,
            params);
      }
    }
    if (n_boundary > 0) {
      const int blocks = (n_boundary + 255) / 256;
      if (use_csw98) {
        csw98_multiblock_boundary_force_kernel<<<blocks, 256>>>(
            state.edge_force_av_r.data(),
            state.edge_force_av_z.data(),
            state.work_av_per_cell.data(),
            d_compressive_count,
            d_negative_work_count,
            state.x_r.data(),
            state.x_z.data(),
            v_r,
            v_z,
            state.rho.data(),
            cell_cs.data(),
            hydro_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(d_face_adj_offsets),
            raw_or_null(d_face_adj_indices),
            d_cell_nverts_ptr,
            n_internal,
            n_boundary,
            params,
            d_limiter_miss);
      } else {
        csw_multiblock_boundary_force_kernel<<<blocks, 256>>>(
            state.edge_force_av_r.data(),
            state.edge_force_av_z.data(),
            state.work_av_per_cell.data(),
            d_compressive_count,
            d_negative_work_count,
            state.x_r.data(),
            state.x_z.data(),
            v_r,
            v_z,
            state.rho.data(),
            cell_cs.data(),
            hydro_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(d_face_adj_offsets),
            raw_or_null(d_face_adj_indices),
            d_cell_nverts_ptr,
            n_internal,
            n_boundary,
            params);
      }
    }
    // I1-B pole tangential damper: extra pairwise tangential edge drag on
    // near-pole shell columns, accumulated into the same edge AV buffers so
    // momentum kick and corrector time-centered work closure are inherited.
    // Env-gated default-off (TENRYU_I1B_POLE_DAMPER).
    if (pole_damper_enabled() && n_internal > 0) {
      const mesh::BlockInfo* shell_block = nullptr;
      int north_fan_rows = 0;
      for (const auto& block : mb.blocks) {
        if (block.role == mesh::BlockRole::POLAR_SHELL) {
          shell_block = &block;
        } else if (block.role == mesh::BlockRole::NORTH_FAN) {
          north_fan_rows = block.n_i_cells;
        }
      }
      if (shell_block != nullptr && shell_block->n_j_cells >= 4 &&
          shell_block->n_i_cells >= 1) {
        int q_begin_active = 0;
        if (state.central_pseudo_core.built && mb.has_trifan_cap) {
          q_begin_active =
              std::max(0, state.central_pseudo_core.member_ring_count -
                              mb.n_cap - north_fan_rows);
        }
        static bool pole_damper_logged = false;
        int* d_pair_count = nullptr;
        if (!pole_damper_logged) {
          cuda_check(
              cudaMalloc(reinterpret_cast<void**>(&d_pair_count), sizeof(int)),
              "pole damper: cudaMalloc pair count failed");
          cuda_check(cudaMemset(d_pair_count, 0, sizeof(int)),
                     "pole damper: zero pair count failed");
        }
        const int blocks_damper = (n_internal + 255) / 256;
        pole_damper_internal_force_kernel<<<blocks_damper, 256>>>(
            state.edge_force_av_r.data(),
            state.edge_force_av_z.data(),
            state.work_av_per_cell.data(),
            state.x_r.data(),
            state.x_z.data(),
            v_r,
            v_z,
            state.rho.data(),
            cell_cs.data(),
            hydro_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_internal,
            shell_block->owned_node_begin,
            shell_block->n_j_cells,
            shell_block->n_i_cells + 1,
            q_begin_active,
            pole_damper_m(),
            pole_damper_q_lo(),
            pole_damper_q_hi(),
            pole_damper_ctheta(),
            d_pair_count);
        if (!pole_damper_logged) {
          int pair_count = 0;
          cuda_check(cudaMemcpy(&pair_count, d_pair_count, sizeof(int),
                                cudaMemcpyDeviceToHost),
                     "pole damper: copy pair count failed");
          cuda_check(cudaFree(d_pair_count),
                     "pole damper: cudaFree pair count failed");
          core::log_info(
              "[pole_damper] first fire: C_theta=" +
              format_csw_scientific(pole_damper_ctheta()) +
              " m_columns=" + std::to_string(pole_damper_m()) +
              " q_band=[" + std::to_string(pole_damper_q_lo()) + "," +
              std::to_string(pole_damper_q_hi()) + "]" +
              " q_begin_active=" + std::to_string(q_begin_active) +
              " damped_pairs=" + std::to_string(pair_count));
          pole_damper_logged = true;
        }
      }
    }
  } else {
    const int n_edges = state.mesh.topo.n_edges();
    TENRYU_ASSERT(state.edge_force_av_r.size() == static_cast<std::size_t>(n_edges),
                  "CSW edge AV structured edge buffer size mismatch");
    const int blocks = (n_edges + 255) / 256;
    if (use_csw98) {
      csw98_structured_force_kernel<<<blocks, 256>>>(
          state.edge_force_av_r.data(),
          state.edge_force_av_z.data(),
          state.work_av_per_cell.data(),
          d_compressive_count,
          d_negative_work_count,
          state.x_r.data(),
          state.x_z.data(),
          v_r,
          v_z,
          state.rho.data(),
          cell_cs.data(),
          hydro_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          params);
    } else {
      csw_structured_force_kernel<<<blocks, 256>>>(
          state.edge_force_av_r.data(),
          state.edge_force_av_z.data(),
          state.work_av_per_cell.data(),
          d_compressive_count,
          d_negative_work_count,
          state.x_r.data(),
          state.x_z.data(),
          v_r,
          v_z,
          state.rho.data(),
          cell_cs.data(),
          hydro_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          params);
    }
  }

  if (csw98_limiter_diag) {
    int counters[2] = {0, 0};
    cuda_check(cudaMemcpy(counters, d_limiter_miss, 2 * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "csw98 AV: copy limiter diag counters failed");
    cuda_check(cudaFree(d_limiter_miss),
               "csw98 AV: cudaFree limiter diag counters failed");
    if (counters[1] > 0) {
      // report the first launch where the limiter actually ran
      // (evals > 0); stay armed through vacuous t~0 launches.
      core::log_info(
          "[csw98-limiter-diag] first active launch: evals=" +
          std::to_string(counters[1]) +
          " unresolved continuation members: " +
          std::to_string(counters[0]));
      csw98_limiter_diag_logged = true;
    }
  }

  const int blocks_cells = (n_cells + 255) / 256;
  clip_negative_work_kernel<<<blocks_cells, 256>>>(
      state.work_av_per_cell.data(), d_negative_work_count, n_cells);
  cuda_check(cudaGetLastError(), "CSW edge AV kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "CSW edge AV kernel execution failed");

  int compressive_count = 0;
  int negative_count = 0;
  cuda_check(cudaMemcpy(&compressive_count, d_compressive_count, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "CSW edge AV: copy compressive count failed");
  cuda_check(cudaMemcpy(&negative_count, d_negative_work_count, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "CSW edge AV: copy negative work count failed");
  cuda_check(cudaFree(d_negative_work_count),
             "CSW edge AV: cudaFree negative work count failed");
  cuda_check(cudaFree(d_compressive_count),
             "CSW edge AV: cudaFree compressive count failed");

  state.count_edge_compressive_edges_step = compressive_count;
  if (negative_count > 0) {
    core::log_warning("CSW edge AV clipped " + std::to_string(negative_count) +
                      " negative/non-finite AV work contribution(s)");
  }
  // [av_activity] read-only export (env TENRYU_I1B_AV_ACTIVITY_LOG_EVERY=N,
  // default off): per-call sums of the per-call-zeroed work_av buffer and
  // the fire/negative counters, aggregated per hydro STEP (the launcher
  // runs k>=2 times per step: predictor + corrector accelerations, plus
  // any retry/subcycle re-evaluations -- calls= reports the completed
  // step's call count). A completed step is detected on the first call
  // with a different state.step; its totals print with one-step latency
  // when completed_step % N == 0.
  static const int activity_every = [] {
    const char* raw = std::getenv("TENRYU_I1B_AV_ACTIVITY_LOG_EVERY");
    const int v = raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : 0;
    return v > 0 ? v : 0;
  }();
  if (activity_every > 0) {
    static long long act_prev_step = -1;
    static double act_prev_t = 0.0;
    static int act_calls = 0;
    static long long act_fire = 0;
    static long long act_neg = 0;
    static long double act_work_step = 0.0L;
    static long double act_work_cum = 0.0L;
    const double call_work = thrust::reduce(
        thrust::device,
        thrust::device_pointer_cast(state.work_av_per_cell.data()),
        thrust::device_pointer_cast(state.work_av_per_cell.data() + n_cells),
        0.0,
        thrust::plus<double>());
    if (act_prev_step >= 0 && state.step != act_prev_step) {
      if (act_prev_step % activity_every == 0) {
        std::fprintf(stderr,
                     "[av_activity] step=%lld t=%.6e calls=%d "
                     "fire_count=%lld work_av_step=%.6e work_av_cum=%.10e "
                     "negative_work_count=%lld\n",
                     act_prev_step,
                     act_prev_t,
                     act_calls,
                     act_fire,
                     static_cast<double>(act_work_step),
                     static_cast<double>(act_work_cum),
                     act_neg);
      }
      act_calls = 0;
      act_fire = 0;
      act_neg = 0;
      act_work_step = 0.0L;
    }
    act_prev_step = state.step;
    act_prev_t = state.t;
    ++act_calls;
    act_fire += compressive_count;
    act_neg += negative_count;
    act_work_step += call_work;
    act_work_cum += call_work;
  }
}

double compute_csw_edge_av_cfl_dt(const core::State& state,
                                  const core::Config& cfg) {
  if (cfg.main.dim != 2 ||
      (cfg.numerics.hydro.av_model != core::AvModel::CswEdge &&
       cfg.numerics.hydro.av_model != core::AvModel::CswEdgeCsw98)) {
    return std::numeric_limits<double>::infinity();
  }
  const bool use_csw98 =
      cfg.numerics.hydro.av_model == core::AvModel::CswEdgeCsw98;
  const double coefficient = cfg.numerics.hydro.av_cfl_coefficient;
  if (!(coefficient > 0.0) || state.rho.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size() &&
                    state.x_z.size() == state.v_z.size() &&
                    state.x_r.size() == state.x_z.size(),
                "CSW edge AV CFL requires matching node arrays");

  double* d_min_dt = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_min_dt), sizeof(double)),
             "CSW edge AV CFL: cudaMalloc min_dt failed");
  const double inf = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(d_min_dt, &inf, sizeof(double), cudaMemcpyHostToDevice),
             "CSW edge AV CFL: init min_dt failed");
  std::int8_t* d_active = upload_hydro_active_or_null(state, cfg);
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state);
  CswKernelParams params;
  params.c1 = cfg.numerics.hydro.av_linear;
  params.c2 = cfg.numerics.hydro.av_quadratic;
  params.gamma = cfg.materials.materials.empty()
                     ? (5.0 / 3.0)
                     : cfg.materials.materials.front().ideal_gas_gamma;
  params.limiter_enabled = cfg.numerics.hydro.csw_limiter_enabled ? 1 : 0;

  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "CSW edge AV CFL multiblock topology metadata missing");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      state.rho.size() + 1U,
                  "CSW edge AV CFL requires multiblock cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      state.rho.size() *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "CSW edge AV CFL requires multiblock cell-node CSR indices");
    const auto& mb = *state.mesh.topo.multiblock;
    const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
    const int n_boundary = static_cast<int>(mb.boundary_faces.size());
    thrust::device_vector<int> d_face_adj_offsets(mb.face_adj_csr_offsets.begin(),
                                                  mb.face_adj_csr_offsets.end());
    thrust::device_vector<int> d_face_adj_indices(mb.face_adj_csr_indices.begin(),
                                                  mb.face_adj_csr_indices.end());
    if (n_internal > 0) {
      const int blocks = (n_internal + 255) / 256;
      if (use_csw98) {
        csw98_multiblock_internal_cfl_kernel<<<blocks, 256>>>(
            d_min_dt,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(d_face_adj_offsets),
            raw_or_null(d_face_adj_indices),
            d_cell_nverts_ptr,
            n_internal,
            params,
            coefficient);
      } else {
        csw_multiblock_internal_cfl_kernel<<<blocks, 256>>>(
            d_min_dt,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_internal,
            coefficient);
      }
    }
    if (n_boundary > 0) {
      const int blocks = (n_boundary + 255) / 256;
      if (use_csw98) {
        csw98_multiblock_boundary_cfl_kernel<<<blocks, 256>>>(
            d_min_dt,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            raw_or_null(d_face_adj_offsets),
            raw_or_null(d_face_adj_indices),
            d_cell_nverts_ptr,
            n_boundary,
            params,
            coefficient);
      } else {
        csw_multiblock_boundary_cfl_kernel<<<blocks, 256>>>(
            d_min_dt,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_boundary,
            coefficient);
      }
    }
  } else {
    const int n_edges = state.mesh.topo.n_edges();
    const int blocks = (n_edges + 255) / 256;
    if (use_csw98) {
      csw98_structured_cfl_kernel<<<blocks, 256>>>(
          d_min_dt,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          params,
          coefficient);
    } else {
      csw_structured_cfl_kernel<<<blocks, 256>>>(
          d_min_dt,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          coefficient);
    }
  }
  cuda_check(cudaGetLastError(), "CSW edge AV CFL kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "CSW edge AV CFL kernel execution failed");

  double min_dt = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(&min_dt, d_min_dt, sizeof(double), cudaMemcpyDeviceToHost),
             "CSW edge AV CFL: copy min_dt failed");
  if (use_csw98 && mesh::mesh_topo_is_multiblock(cfg.mesh) &&
      std::isfinite(min_dt) && min_dt < 1.0e-15 &&
      cfg.numerics.debug.trace_mesh_motion) {
    int* d_winner_index = nullptr;
    CswEdgeAvCflWinner* d_winner = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_winner_index), sizeof(int)),
               "CSW98 edge AV CFL: cudaMalloc winner_index failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_winner),
                          sizeof(CswEdgeAvCflWinner)),
               "CSW98 edge AV CFL: cudaMalloc winner failed");
    const int no_winner = std::numeric_limits<int>::max();
    const CswEdgeAvCflWinner empty_winner;
    cuda_check(cudaMemcpy(d_winner_index, &no_winner, sizeof(int),
                          cudaMemcpyHostToDevice),
               "CSW98 edge AV CFL: init winner_index failed");
    cuda_check(cudaMemcpy(d_winner, &empty_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyHostToDevice),
               "CSW98 edge AV CFL: init winner failed");

    const auto& mb = *state.mesh.topo.multiblock;
    const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
    const int n_boundary = static_cast<int>(mb.boundary_faces.size());
    thrust::device_vector<int> d_face_adj_offsets(
        mb.face_adj_csr_offsets.begin(), mb.face_adj_csr_offsets.end());
    thrust::device_vector<int> d_face_adj_indices(
        mb.face_adj_csr_indices.begin(), mb.face_adj_csr_indices.end());
    if (n_internal > 0) {
      const int blocks = (n_internal + 255) / 256;
      csw98_multiblock_internal_cfl_winner_index_kernel<<<blocks, 256>>>(
          d_winner_index,
          d_min_dt,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          raw_or_null(mb.d_unique_face_cell_a),
          raw_or_null(mb.d_unique_face_cell_b),
          raw_or_null(mb.d_unique_face_local_a),
          raw_or_null(mb.d_unique_face_local_b),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          raw_or_null(d_face_adj_offsets),
          raw_or_null(d_face_adj_indices),
          d_cell_nverts_ptr,
          n_internal,
          params,
          coefficient,
          min_dt);
    }
    if (n_boundary > 0) {
      const int blocks = (n_boundary + 255) / 256;
      csw98_multiblock_boundary_cfl_winner_index_kernel<<<blocks, 256>>>(
          d_winner_index,
          d_min_dt,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          raw_or_null(mb.d_boundary_face_cell),
          raw_or_null(mb.d_boundary_face_local),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          raw_or_null(d_face_adj_offsets),
          raw_or_null(d_face_adj_indices),
          d_cell_nverts_ptr,
          n_internal,
          n_boundary,
          params,
          coefficient,
          min_dt);
    }
    cuda_check(cudaGetLastError(),
               "CSW98 edge AV CFL winner index kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "CSW98 edge AV CFL winner index kernel execution failed");

    int* d_cell_block_id = nullptr;
    if (mb.cell_block_id.size() == state.rho.size()) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_block_id),
                            mb.cell_block_id.size() * sizeof(int)),
                 "CSW98 edge AV CFL: cudaMalloc cell_block_id failed");
      cuda_check(cudaMemcpy(d_cell_block_id, mb.cell_block_id.data(),
                            mb.cell_block_id.size() * sizeof(int),
                            cudaMemcpyHostToDevice),
                 "CSW98 edge AV CFL: copy cell_block_id failed");
    }
    csw98_multiblock_cfl_winner_values_kernel<<<1, 1>>>(
        d_winner,
        d_winner_index,
        state.x_r.data(),
        state.x_z.data(),
        state.v_r.data(),
        state.v_z.data(),
        raw_or_null(mb.d_unique_face_cell_a),
        raw_or_null(mb.d_unique_face_cell_b),
        raw_or_null(mb.d_unique_face_local_a),
        raw_or_null(mb.d_unique_face_local_b),
        raw_or_null(mb.d_boundary_face_cell),
        raw_or_null(mb.d_boundary_face_local),
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts_ptr,
        d_cell_block_id,
        n_internal,
        n_boundary);
    cuda_check(cudaGetLastError(),
               "CSW98 edge AV CFL winner values kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "CSW98 edge AV CFL winner values kernel execution failed");

    CswEdgeAvCflWinner winner;
    cuda_check(cudaMemcpy(&winner, d_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyDeviceToHost),
               "CSW98 edge AV CFL: copy winner failed");
    if (winner.cell_a >= 0) {
      pole_angular_derefine::assert_active_cell(
          state, winner.cell_a, "CSW98 edge AV CFL winner");
      int stable_cell = winner.cell_a;
      if (mb.cell_id_stable.size() == state.rho.size()) {
        stable_cell =
            mb.cell_id_stable[static_cast<std::size_t>(winner.cell_a)];
      }
      core::log_warning("[csw98-edge-av-cfl-winner] min_dt=" +
                        format_csw_scientific(min_dt) +
                        " n0=" + std::to_string(winner.n0) +
                        " n1=" + std::to_string(winner.n1) +
                        " r0=" + format_csw_scientific(winner.r0) +
                        " z0=" + format_csw_scientific(winner.z0) +
                        " r1=" + format_csw_scientific(winner.r1) +
                        " z1=" + format_csw_scientific(winner.z1) +
                        " dx=" + format_csw_scientific(winner.dx) +
                        " du=" + format_csw_scientific(winner.du) +
                        " stable_cell=" + std::to_string(stable_cell) +
                        " block_id=" + std::to_string(winner.block_id));
    } else {
      core::log_warning(
          "[csw98-edge-av-cfl-winner] UNAVAILABLE min_dt=" +
          format_csw_scientific(min_dt));
    }
    if (d_cell_block_id != nullptr) {
      cuda_check(cudaFree(d_cell_block_id),
                 "CSW98 edge AV CFL: cudaFree cell_block_id failed");
    }
    cuda_check(cudaFree(d_winner),
               "CSW98 edge AV CFL: cudaFree winner failed");
    cuda_check(cudaFree(d_winner_index),
               "CSW98 edge AV CFL: cudaFree winner_index failed");
  }
  if (!use_csw98 && std::isfinite(min_dt) && min_dt < 1.0e-15 &&
      cfg.numerics.debug.trace_mesh_motion) {
    int* d_winner_index = nullptr;
    CswEdgeAvCflWinner* d_winner = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_winner_index), sizeof(int)),
               "CSW edge AV CFL: cudaMalloc winner_index failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_winner),
                          sizeof(CswEdgeAvCflWinner)),
               "CSW edge AV CFL: cudaMalloc winner failed");
    const int no_winner = std::numeric_limits<int>::max();
    const CswEdgeAvCflWinner empty_winner;
    cuda_check(cudaMemcpy(d_winner_index, &no_winner, sizeof(int),
                          cudaMemcpyHostToDevice),
               "CSW edge AV CFL: init winner_index failed");
    cuda_check(cudaMemcpy(d_winner, &empty_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyHostToDevice),
               "CSW edge AV CFL: init winner failed");
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      const auto& mb = *state.mesh.topo.multiblock;
      const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
      const int n_boundary = static_cast<int>(mb.boundary_faces.size());
      if (n_internal > 0) {
        const int blocks = (n_internal + 255) / 256;
        csw_multiblock_internal_cfl_winner_index_kernel<<<blocks, 256>>>(
            d_winner_index,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_unique_face_cell_a),
            raw_or_null(mb.d_unique_face_cell_b),
            raw_or_null(mb.d_unique_face_local_a),
            raw_or_null(mb.d_unique_face_local_b),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_internal,
            coefficient,
            min_dt);
      }
      if (n_boundary > 0) {
        const int blocks = (n_boundary + 255) / 256;
        csw_multiblock_boundary_cfl_winner_index_kernel<<<blocks, 256>>>(
            d_winner_index,
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            d_active,
            raw_or_null(mb.d_boundary_face_cell),
            raw_or_null(mb.d_boundary_face_local),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_nverts_ptr,
            n_internal,
            n_boundary,
            coefficient,
            min_dt);
      }
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL winner index kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL winner index kernel execution failed");
      int* d_cell_block_id = nullptr;
      if (mb.cell_block_id.size() == state.rho.size()) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_block_id),
                              mb.cell_block_id.size() * sizeof(int)),
                   "CSW edge AV CFL: cudaMalloc cell_block_id failed");
        cuda_check(cudaMemcpy(d_cell_block_id, mb.cell_block_id.data(),
                              mb.cell_block_id.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "CSW edge AV CFL: copy cell_block_id failed");
      }
      csw_multiblock_cfl_winner_values_kernel<<<1, 1>>>(
          d_winner,
          d_winner_index,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          raw_or_null(mb.d_unique_face_cell_a),
          raw_or_null(mb.d_unique_face_cell_b),
          raw_or_null(mb.d_unique_face_local_a),
          raw_or_null(mb.d_unique_face_local_b),
          raw_or_null(mb.d_boundary_face_cell),
          raw_or_null(mb.d_boundary_face_local),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts_ptr,
          d_cell_block_id,
          n_internal,
          n_boundary);
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL winner values kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL winner values kernel execution failed");
      if (d_cell_block_id != nullptr) {
        cuda_check(cudaFree(d_cell_block_id),
                   "CSW edge AV CFL: cudaFree cell_block_id failed");
      }
    } else {
      const int n_edges = state.mesh.topo.n_edges();
      const int blocks = (n_edges + 255) / 256;
      csw_structured_cfl_winner_index_kernel<<<blocks, 256>>>(
          d_winner_index,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          coefficient,
          min_dt);
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL winner index kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL winner index kernel execution failed");
      csw_structured_cfl_winner_values_kernel<<<1, 1>>>(
          d_winner,
          d_winner_index,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          d_active,
          state.mesh.topo.nr,
          state.mesh.topo.nz);
      cuda_check(cudaGetLastError(),
                 "CSW edge AV CFL winner values kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "CSW edge AV CFL winner values kernel execution failed");
    }
    CswEdgeAvCflWinner winner;
    cuda_check(cudaMemcpy(&winner, d_winner, sizeof(CswEdgeAvCflWinner),
                          cudaMemcpyDeviceToHost),
               "CSW edge AV CFL: copy winner failed");
    pole_angular_derefine::assert_active_cell(
        state, winner.cell_a, "CSW edge AV CFL winner");
    core::log_warning("[csw-edge-av-cfl-winner] min_dt=" +
                      format_csw_scientific(min_dt) +
                      " n0=" + std::to_string(winner.n0) +
                      " n1=" + std::to_string(winner.n1) +
                      " r0=" + format_csw_scientific(winner.r0) +
                      " z0=" + format_csw_scientific(winner.z0) +
                      " r1=" + format_csw_scientific(winner.r1) +
                      " z1=" + format_csw_scientific(winner.z1) +
                      " dx=" + format_csw_scientific(winner.dx) +
                      " du=" + format_csw_scientific(winner.du) +
                      " cell_a=" + std::to_string(winner.cell_a) +
                      " block_id=" + std::to_string(winner.block_id));
    cuda_check(cudaFree(d_winner), "CSW edge AV CFL: cudaFree winner failed");
    cuda_check(cudaFree(d_winner_index),
               "CSW edge AV CFL: cudaFree winner_index failed");
  }
  if (d_active != nullptr) {
    cuda_check(cudaFree(d_active), "CSW edge AV CFL: cudaFree hydro_active failed");
  }
  cuda_check(cudaFree(d_min_dt), "CSW edge AV CFL: cudaFree min_dt failed");
  return min_dt;
}

}  // namespace tenryu::hydro::compatible
