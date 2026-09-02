#include "hydro/compatible_av_tensor.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include "core/error.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/compatible_force_work_2d.cuh"
#include "hydro/hydro_multiblock_topology.cuh"
#include "hydro/pole_angular_derefine.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::compatible {
namespace {

constexpr double kPi = 3.1415926535897932384626433832795028841971693993751;
constexpr double kTiny = 1.0e-300;
constexpr double kRatioTiny = 1.0e-30;
constexpr int kTensorMaxVerts = mesh::kMeshTopoCellStorageSlotsMax;

struct TensorKernelParams {
  double c1 = 1.0;
  double c2 = 1.0;
  double gamma = 5.0 / 3.0;
  int limiter_enabled = 1;
};

struct TensorTopologyDevice {
  const int* cell_node_offsets = nullptr;
  const int* cell_node_indices = nullptr;
  const int* face_adj_offsets = nullptr;
  const int* face_adj_indices = nullptr;
  const std::uint8_t* cell_nverts = nullptr;
  int nr = 0;
  int nz = 0;
  int corner_stride = 4;
  int multiblock = 0;
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
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
    int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral]) {
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

__device__ inline void tensor_cell_nodes(const TensorTopologyDevice topo,
                                         const int c,
                                         int nodes[kTensorMaxVerts],
                                         int* nverts) {
  if (topo.multiblock != 0) {
    *nverts = mesh::mesh_topo_cell_active_nverts(topo.cell_nverts, c);
    csr_cell_active_nodes(c, *nverts, topo.cell_node_offsets,
                          topo.cell_node_indices, nodes);
    return;
  }
  *nverts = 4;
  structured_cell_nodes(c, topo.nz, nodes);
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

__device__ inline int tensor_face_neighbor(const TensorTopologyDevice topo,
                                           const int c,
                                           const int local) {
  if (topo.multiblock != 0) {
    return topo.face_adj_indices[topo.face_adj_offsets[c] + local];
  }
  return structured_face_neighbor(c, local, topo.nr, topo.nz);
}

// verbatim copy of csw98_* (compatible_av_csw.cuh) — do NOT edit the
// original TU (bit-frozen csw98 gates)
__host__ __device__ inline bool tensor_median_svec_cart(const double r_a,
                                                       const double z_a,
                                                       const double r_b,
                                                       const double z_b,
                                                       const double c_r,
                                                       const double c_z,
                                                       double* s_r,
                                                       double* s_z) {
  const double m_r = 0.5 * (r_a + r_b);
  const double m_z = 0.5 * (z_a + z_b);
  const double d_r = c_r - m_r;
  const double d_z = c_z - m_z;
  const double t_r = r_b - r_a;
  const double t_z = z_b - z_a;
  // R(-90deg)(d) = (d_z, -d_r); orientation sign = S0.t = t_r*d_z - t_z*d_r.
  const double orient = t_r * d_z - t_z * d_r;
  const double scale = (fabs(t_r) + fabs(t_z)) * (fabs(d_r) + fabs(d_z));
  if (!(scale > 0.0) || !(fabs(orient) > 1.0e-30 * scale)) {
    *s_r = 0.0;
    *s_z = 0.0;
    return false;
  }
  if (orient > 0.0) {
    *s_r = d_z;
    *s_z = -d_r;
  } else {
    *s_r = -d_z;
    *s_z = d_r;
  }
  return true;
}

// verbatim copy of csw98_* (compatible_av_csw.cuh) — do NOT edit the
// original TU (bit-frozen csw98 gates)
__host__ __device__ inline double tensor_winding_orientation(const double* r,
                                                             const double* z,
                                                             const int nverts) {
  double area2 = 0.0;
  double scale = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    area2 += r[k] * z[kp] - r[kp] * z[k];
    scale += r[k] * r[k] + z[k] * z[k];
  }
  const double threshold = 64.0 * 2.220446049250313e-16 * scale;
  if (!(fabs(area2) > threshold)) {
  #ifdef __CUDA_ARCH__
    __trap();  // degenerate/zero-area cell reached AV orientation: fail loud (kernel abort -> CUDA_CHECK)
  #else
    ::tenryu::core::tenryu_abort(
        "fabs(area2) > threshold",
        "csw98_winding_orientation: zero/near-zero signed area (degenerate cell) "
        "must not silently orient +1 (AI-review Amendment A)",
        __FILE__, __LINE__);
  #endif
  }
  return (area2 >= 0.0) ? 1.0 : -1.0;
}

// verbatim copy of csw98_* (compatible_av_csw.cuh) — do NOT edit the
// original TU (bit-frozen csw98 gates)
__host__ __device__ inline void tensor_planar_corner_gradients(
    const double* r,
    const double* z,
    const int nverts,
    double* a_r,
    double* a_z) {
  const double orientation = tensor_winding_orientation(r, z, nverts);
  for (int k = 0; k < nverts; ++k) {
    const int km = (k + nverts - 1) % nverts;
    const int kp = (k + 1) % nverts;
    a_r[k] = 0.5 * orientation * (z[kp] - z[km]);
    a_z[k] = 0.5 * orientation * (r[km] - r[kp]);
  }
}

// verbatim copy of csw98_* (compatible_av_csw.cu) — do NOT edit the
// original TU (bit-frozen csw98 gates)
__device__ inline double tensor_limiter_from_ratios(const double r0,
                                                    const double r1) {
  const double limited = fmin(fmin(1.0, 2.0 * r0),
                             fmin(2.0 * r1, 0.5 * (r0 + r1)));
  return fmax(0.0, limited);
}

// verbatim copy of csw98_* (compatible_av_csw.cu) — do NOT edit the
// original TU (bit-frozen csw98 gates)
__device__ inline double tensor_continuation_ratio_values(
    const double xa_r,
    const double xa_z,
    const double va_r,
    const double va_z,
    const double xb_r,
    const double xb_z,
    const double vb_r,
    const double vb_z,
    const double xhat_r,
    const double xhat_z,
    const double uhat_r,
    const double uhat_z,
    const double du_mag,
    const double dx_mag,
    int* miss) {
  const double dxn_r = xb_r - xa_r;
  const double dxn_z = xb_z - xa_z;
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
  const double dun_r = vb_r - va_r;
  const double dun_z = vb_z - va_z;
  const double ratio =
      ((dun_r * uhat_r + dun_z * uhat_z) * dx_mag) / (den_geom * du_mag);
  return finite_device(ratio) ? ratio : 1.0;
}

// verbatim copy of csw98_* (compatible_av_csw.cu) — do NOT edit the
// original TU (bit-frozen csw98 gates)
__device__ inline double tensor_continuation_ratio(
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

// verbatim copy of csw98_* (compatible_av_csw.cu) — do NOT edit the
// original TU (bit-frozen csw98 gates)
__device__ inline double tensor_structured_limiter(
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
    const double dx_mag,
    const bool aw_planar,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
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
  if (aw_planar && (local == 0 || local == 1) &&
      ((aw_axis_slave_theta0_active && j == 0) ||
       (aw_axis_slave_theta_pi_active && j == nz - 1))) {
    int nodes[4] = {0, 0, 0, 0};
    structured_cell_nodes(c, nz, nodes);
    // Complete the missing boundary-side continuation with H=diag(-1, 1).
    if (aw_axis_slave_theta0_active && j == 0) {
      const int axis = nodes[corner0];
      const int off_axis = nodes[corner1];
      r_prev = tensor_continuation_ratio_values(
          -x_r[off_axis], x_z[off_axis], -v_r[off_axis], v_z[off_axis],
          x_r[axis], x_z[axis], v_r[axis], v_z[axis], xhat_r, xhat_z,
          uhat_r, uhat_z, du_mag, dx_mag, nullptr);
    }
    if (aw_axis_slave_theta_pi_active && j == nz - 1) {
      const int off_axis = nodes[corner0];
      const int axis = nodes[corner1];
      r_next = tensor_continuation_ratio_values(
          x_r[axis], x_z[axis], v_r[axis], v_z[axis], -x_r[off_axis],
          x_z[off_axis], -v_r[off_axis], v_z[off_axis], xhat_r, xhat_z,
          uhat_r, uhat_z, du_mag, dx_mag, nullptr);
    }
  }
  if (cell_active(hydro_active, c_prev)) {
    int nodes_nb[4] = {0, 0, 0, 0};
    structured_cell_nodes(c_prev, nz, nodes_nb);
    r_prev = tensor_continuation_ratio(
        x_r, x_z, v_r, v_z, nodes_nb[corner0], nodes_nb[corner1], xhat_r,
        xhat_z, uhat_r, uhat_z, du_mag, dx_mag, nullptr);
  }
  if (cell_active(hydro_active, c_next)) {
    int nodes_nb[4] = {0, 0, 0, 0};
    structured_cell_nodes(c_next, nz, nodes_nb);
    r_next = tensor_continuation_ratio(
        x_r, x_z, v_r, v_z, nodes_nb[corner0], nodes_nb[corner1], xhat_r,
        xhat_z, uhat_r, uhat_z, du_mag, dx_mag, nullptr);
  }
  return tensor_limiter_from_ratios(r_prev, r_next);
}

// verbatim copy of csw98_* (compatible_av_csw.cu) — do NOT edit the
// original TU (bit-frozen csw98 gates)
__device__ inline double tensor_multiblock_limiter(
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
    int nb_nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {0};
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
    ratios[e] = tensor_continuation_ratio(
        x_r, x_z, v_r, v_z, cont_a, cont_b, xhat_r, xhat_z, uhat_r, uhat_z,
        du_mag, dx_mag, miss);
  }
  return tensor_limiter_from_ratios(ratios[0], ratios[1]);
}

__device__ inline double tensor_planar_area(const double r[kTensorMaxVerts],
                                            const double z[kTensorMaxVerts],
                                            const int nverts) {
  double area2 = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : k + 1;
    area2 += r[k] * z[kp] - r[kp] * z[k];
  }
  return 0.5 * fabs(area2);
}

__global__ void tensor_sensor_l0_kernel(
    double* __restrict__ sigma,
    double* __restrict__ du_cell,
    double* __restrict__ length0,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ cell_cs,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const TensorTopologyDevice topo,
    const TensorKernelParams params,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (!cell_active(hydro_active, c)) {
    sigma[c] = 0.0;
    du_cell[c] = 0.0;
    length0[c] = 0.0;
    return;
  }

  int nodes[kTensorMaxVerts] = {0};
  int nverts = 0;
  tensor_cell_nodes(topo, c, nodes, &nverts);
  double r[kTensorMaxVerts] = {0.0};
  double z[kTensorMaxVerts] = {0.0};
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  double ubar_r = 0.0;
  double ubar_z = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int n = nodes[k];
    r[k] = x_r[n];
    z[k] = x_z[n];
    centroid_r += r[k];
    centroid_z += z[k];
    ubar_r += v_r[n];
    ubar_z += v_z[n];
  }
  const double inv_nverts = 1.0 / static_cast<double>(nverts);
  centroid_r *= inv_nverts;
  centroid_z *= inv_nverts;
  ubar_r *= inv_nverts;
  ubar_z *= inv_nverts;
  const double area = tensor_planar_area(r, z, nverts);
  const double ubar_mag = hypot(ubar_r, ubar_z);
  double length = sqrt(area);
  if (ubar_mag >= 1.0e-12 * fmax(cell_cs[c], 1.0)) {
    const double dhat_r = ubar_r / ubar_mag;
    const double dhat_z = ubar_z / ubar_mag;
    double xmin = r[0] * dhat_r + z[0] * dhat_z;
    double xmax = xmin;
    for (int k = 1; k < nverts; ++k) {
      const double projection = r[k] * dhat_r + z[k] * dhat_z;
      xmin = fmin(xmin, projection);
      xmax = fmax(xmax, projection);
    }
    length = xmax - xmin;
  }

  double sigma_c = 0.0;
  double du_c = 0.0;
  const int face_scan_count = (nverts <= 4) ? 4 : nverts;
  for (int local = 0; local < face_scan_count; ++local) {
    int corner0 = 0;
    int corner1 = 0;
    if (!local_face_corners(nverts, local, &corner0, &corner1)) {
      continue;
    }
    const int n0 = nodes[corner0];
    const int n1 = nodes[corner1];
    double s_r = 0.0;
    double s_z = 0.0;
    if (!tensor_median_svec_cart(x_r[n0], x_z[n0], x_r[n1], x_z[n1],
                                 centroid_r, centroid_z, &s_r, &s_z)) {
      continue;
    }
    const double delta_u_r = v_r[n1] - v_r[n0];
    const double delta_u_z = v_z[n1] - v_z[n0];
    const double q_ce = fmax(
        0.0, -(delta_u_r * s_r + delta_u_z * s_z) /
                 (hypot(s_r, s_z) + kTiny));
    du_c = fmax(du_c, q_ce);
    if (!(q_ce > 0.0)) {
      continue;
    }
    double psi = 0.0;
    if (params.limiter_enabled != 0) {
      const double dx_r = x_r[n1] - x_r[n0];
      const double dx_z = x_z[n1] - x_z[n0];
      const double dx_mag = hypot(dx_r, dx_z);
      const double du_mag = hypot(delta_u_r, delta_u_z);
      if (dx_mag > kTiny && du_mag > kTiny && finite_device(dx_mag) &&
          finite_device(du_mag)) {
        const double xhat_r = dx_r / dx_mag;
        const double xhat_z = dx_z / dx_mag;
        const double uhat_r = delta_u_r / du_mag;
        const double uhat_z = delta_u_z / du_mag;
        if (topo.multiblock != 0) {
          psi = tensor_multiblock_limiter(
              x_r, x_z, v_r, v_z, hydro_active, topo.cell_node_offsets,
              topo.cell_node_indices, topo.face_adj_offsets,
              topo.face_adj_indices, topo.cell_nverts, c, local, nodes, nverts,
              n0, n1, xhat_r, xhat_z, uhat_r, uhat_z, du_mag, dx_mag,
              nullptr);
        } else {
          psi = tensor_structured_limiter(
              x_r, x_z, v_r, v_z, hydro_active, c, local, topo.nr, topo.nz,
              xhat_r, xhat_z, uhat_r, uhat_z, du_mag, dx_mag, true,
              aw_axis_slave_theta0_active, aw_axis_slave_theta_pi_active);
        }
      }
    }
    psi = fmin(1.0, fmax(0.0, psi));
    sigma_c = fmax(sigma_c, 1.0 - psi);
  }
  sigma[c] = sigma_c;
  du_cell[c] = du_c;
  length0[c] = length;
}

__global__ void tensor_jacobi_length_kernel(
    double* __restrict__ length_out,
    const double* __restrict__ length_in,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const TensorTopologyDevice topo) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (!cell_active(hydro_active, c)) {
    length_out[c] = 0.0;
    return;
  }
  const int nverts = topo.multiblock != 0
                         ? mesh::mesh_topo_cell_active_nverts(topo.cell_nverts, c)
                         : 4;
  const int face_scan_count = (nverts <= 4) ? 4 : nverts;
  double neighbor_sum = 0.0;
  int neighbor_count = 0;
  for (int local = 0; local < face_scan_count; ++local) {
    const int neighbor = tensor_face_neighbor(topo, c, local);
    if (!cell_active(hydro_active, neighbor)) {
      continue;
    }
    neighbor_sum += length_in[neighbor];
    ++neighbor_count;
  }
  length_out[c] = neighbor_count > 0
                      ? 0.5 * length_in[c] +
                            0.5 * neighbor_sum /
                                static_cast<double>(neighbor_count)
                      : length_in[c];
}

// mu is smoothed with the same two-pass Jacobi as L: a cell-local near-binary
// sigma checkerboards mu behind shocks (ALE-seeded even-odd ringing, ledger
// A75); smoothing removes the wavelength-2 mode the grad(mu) force amplifies.
__global__ void tensor_mu_raw_kernel(
    double* __restrict__ mu_raw,
    const double* __restrict__ sigma,
    const double* __restrict__ du_cell,
    const double* __restrict__ length_cell,
    const double* __restrict__ rho,
    const double* __restrict__ cell_cs,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const TensorKernelParams params) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (!cell_active(hydro_active, c)) {
    mu_raw[c] = 0.0;
    return;
  }
  const double a = params.c2 * (params.gamma + 1.0) * 0.25 * du_cell[c];
  const double linear = params.c1 * cell_cs[c];
  mu_raw[c] = sigma[c] * rho[c] * length_cell[c] *
              (a + sqrt(a * a + linear * linear));
}

__device__ inline bool tensor_ldlt_inverse(const double m[3][3],
                                           double inverse[3][3]) {
  const double trace = m[0][0] + m[1][1] + m[2][2];
  const double threshold = 1.0e-14 * trace;
  const double d0 = m[0][0];
  if (!(d0 > threshold)) {
    return false;
  }
  const double l10 = m[1][0] / d0;
  const double l20 = m[2][0] / d0;
  const double d1 = m[1][1] - l10 * l10 * d0;
  if (!(d1 > threshold)) {
    return false;
  }
  const double l21 = (m[2][1] - l20 * l10 * d0) / d1;
  const double d2 =
      m[2][2] - l20 * l20 * d0 - l21 * l21 * d1;
  if (!(d2 > threshold)) {
    return false;
  }
  for (int col = 0; col < 3; ++col) {
    const double b0 = col == 0 ? 1.0 : 0.0;
    const double b1 = col == 1 ? 1.0 : 0.0;
    const double b2 = col == 2 ? 1.0 : 0.0;
    const double y0 = b0;
    const double y1 = b1 - l10 * y0;
    const double y2 = b2 - l20 * y0 - l21 * y1;
    const double z0 = y0 / d0;
    const double z1 = y1 / d1;
    const double z2 = y2 / d2;
    const double x2 = z2;
    const double x1 = z1 - l21 * x2;
    const double x0 = z0 - l10 * x1 - l20 * x2;
    inverse[0][col] = x0;
    inverse[1][col] = x1;
    inverse[2][col] = x2;
  }
  return true;
}

__global__ void tensor_force_mu_kernel(
    double* __restrict__ corner_force_r,
    double* __restrict__ corner_force_z,
    const double* __restrict__ mu_cell,
    int* __restrict__ tensor_pc_degenerate,
    const double* __restrict__ length_cell,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const TensorTopologyDevice topo) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int base = c * topo.corner_stride;
  if (!cell_active(hydro_active, c)) {
    for (int k = 0; k < topo.corner_stride; ++k) {
      corner_force_r[base + k] = 0.0;
      corner_force_z[base + k] = 0.0;
    }
    return;
  }

  const int active_nverts = topo.multiblock != 0
                                ? mesh::mesh_topo_cell_active_nverts(
                                      topo.cell_nverts, c)
                                : 4;
  if (active_nverts > 8) {
#ifdef __CUDA_ARCH__
    __trap();
#else
    ::tenryu::core::tenryu_abort(
        "active_nverts <= 8",
        "mimetic tensor AV: cell valence exceeds kTensorMaxVerts",
        __FILE__, __LINE__);
#endif
  }
  int nodes[kTensorMaxVerts] = {0};
  int nverts = 0;
  tensor_cell_nodes(topo, c, nodes, &nverts);
  double r[kTensorMaxVerts] = {0.0};
  double z[kTensorMaxVerts] = {0.0};
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  for (int k = 0; k < nverts; ++k) {
    r[k] = x_r[nodes[k]];
    z[k] = x_z[nodes[k]];
    centroid_r += r[k];
    centroid_z += z[k];
  }
  const double inv_nverts = 1.0 / static_cast<double>(nverts);
  centroid_r *= inv_nverts;
  centroid_z *= inv_nverts;
  const double area = tensor_planar_area(r, z, nverts);
  double length = length_cell[c];
  if (!(length > 0.0)) {
    length = sqrt(area);
  }
  if (!(length > 0.0)) {
    length = kTiny;
  }

  double b[kTensorMaxVerts][3] = {{0.0}};
  double m[3][3] = {{0.0}};
  for (int k = 0; k < nverts; ++k) {
    b[k][0] = 1.0;
    b[k][1] = (r[k] - centroid_r) / length;
    b[k][2] = (z[k] - centroid_z) / length;
    for (int row = 0; row < 3; ++row) {
      for (int col = 0; col < 3; ++col) {
        m[row][col] += b[k][row] * b[k][col];
      }
    }
  }

  double inverse[3][3] = {{0.0}};
  const bool pc_good = tensor_ldlt_inverse(m, inverse);
  if (!pc_good) {
    atomicAdd(tensor_pc_degenerate, 1);
  }
  double projection[kTensorMaxVerts][kTensorMaxVerts] = {{0.0}};
  if (pc_good) {
    for (int i = 0; i < nverts; ++i) {
      for (int j = i; j < nverts; ++j) {
        double solved[3] = {0.0, 0.0, 0.0};
        for (int row = 0; row < 3; ++row) {
          solved[row] = inverse[row][0] * b[j][0] +
                        inverse[row][1] * b[j][1] +
                        inverse[row][2] * b[j][2];
        }
        const double value =
            (i == j ? 1.0 : 0.0) -
            (b[i][0] * solved[0] + b[i][1] * solved[1] +
             b[i][2] * solved[2]);
        projection[i][j] = value;
        projection[j][i] = value;
      }
    }
  }

  double gradient_r[kTensorMaxVerts] = {0.0};
  double gradient_z[kTensorMaxVerts] = {0.0};
  tensor_planar_corner_gradients(r, z, nverts, gradient_r, gradient_z);
  const double mu = mu_cell[c];
  for (int i = 0; i < nverts; ++i) {
    double force_r = 0.0;
    double force_z = 0.0;
    for (int j = 0; j < nverts; ++j) {
      const double stiffness =
          mu * ((gradient_r[i] * gradient_r[j] +
                 gradient_z[i] * gradient_z[j]) /
                    area +
                projection[i][j]);
      force_r -= stiffness * v_r[nodes[j]];
      force_z -= stiffness * v_z[nodes[j]];
    }
    corner_force_r[base + i] = force_r;
    corner_force_z[base + i] = force_z;
  }
  for (int k = nverts; k < topo.corner_stride; ++k) {
    corner_force_r[base + k] = 0.0;
    corner_force_z[base + k] = 0.0;
  }
}

__global__ void tensor_sum_corner_forces_structured_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ corner_force_r,
    const double* __restrict__ corner_force_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int corner_stride) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }
  const int node_stride = nz + 1;
  const int i = n / node_stride;
  const int j = n - i * node_stride;
  double fr = 0.0;
  double fz = 0.0;
  for (int di = -1; di <= 0; ++di) {
    for (int dj = -1; dj <= 0; ++dj) {
      const int ic = i + di;
      const int jc = j + dj;
      if (ic < 0 || ic >= nr || jc < 0 || jc >= nz) {
        continue;
      }
      const int c = ic * nz + jc;
      if (!cell_active(hydro_active, c)) {
        continue;
      }
      int corner = 3;
      if (di == 0 && dj == 0) {
        corner = 0;
      } else if (di == -1 && dj == 0) {
        corner = 1;
      } else if (di == -1 && dj == -1) {
        corner = 2;
      }
      const int idx = c * corner_stride + corner;
      fr += corner_force_r[idx];
      fz += corner_force_z[idx];
    }
  }
  force_r[n] += fr;
  force_z[n] += fz;
}

__global__ void tensor_sum_corner_forces_multiblock_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ corner_force_r,
    const double* __restrict__ corner_force_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const int n_nodes,
    const int corner_stride) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  double fr = 0.0;
  double fz = 0.0;
  for (int p = off; p < end; ++p) {
    const int c = reverse_csr_node_cells[p];
    if (!cell_active(hydro_active, c)) {
      continue;
    }
    const int corner = reverse_csr_node_corners[p];
    const int idx = c * corner_stride + corner;
    fr += corner_force_r[idx];
    fz += corner_force_z[idx];
  }
  force_r[n] += fr;
  force_z[n] += fz;
}

__global__ void tensor_work_kernel(
    double* __restrict__ work_av,
    const double* __restrict__ corner_force_r,
    const double* __restrict__ corner_force_z,
    const double* __restrict__ x_r,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const TensorTopologyDevice topo) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (!cell_active(hydro_active, c)) {
    work_av[c] = 0.0;
    return;
  }
  const int active_nverts = topo.multiblock != 0
                                ? mesh::mesh_topo_cell_active_nverts(
                                      topo.cell_nverts, c)
                                : 4;
  if (active_nverts > 8) {
#ifdef __CUDA_ARCH__
    __trap();
#else
    ::tenryu::core::tenryu_abort(
        "active_nverts <= 8",
        "mimetic tensor AV: cell valence exceeds kTensorMaxVerts",
        __FILE__, __LINE__);
#endif
  }
  int nodes[kTensorMaxVerts] = {0};
  int nverts = 0;
  tensor_cell_nodes(topo, c, nodes, &nverts);
  const int base = c * topo.corner_stride;
  double work = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int n = nodes[k];
    const double rz_weight = 2.0 * kPi * x_r[n];
    work -= corner_force_r[base + k] * rz_weight * v_r[n] +
            corner_force_z[base + k] * rz_weight * v_z[n];
  }
  const double finite_work = isfinite(work) ? work : 0.0;
  work_av[c] = finite_work == 0.0 ? 0.0 : finite_work;
}

__global__ void tensor_cfl_kernel(
    double* __restrict__ min_dt,
    int* __restrict__ winner_cell,
    const double target_dt,
    const double* __restrict__ mu_cell,
    const double* __restrict__ length_cell,
    const double* __restrict__ rho,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || !cell_active(hydro_active, c)) {
    return;
  }
  const double length = length_cell[c];
  const double mu = mu_cell[c];
  if (mu == 0.0) {
    return;
  }
  const double dt =
      0.25 * rho[c] * length * length / (mu + kTiny);
  atomic_min_double(min_dt, dt);
  if (winner_cell != nullptr && dt == target_dt) {
    atomicMin(winner_cell, c);
  }
}

template <typename T>
const T* raw_or_null(const thrust::device_vector<T>& values) {
  return values.empty() ? nullptr : thrust::raw_pointer_cast(values.data());
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

std::int8_t* upload_hydro_active_or_null(const core::State& state,
                                         const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<std::int8_t> active;
  if (!state.hydro_active.empty()) {
    TENRYU_ASSERT(state.hydro_active.size() == static_cast<std::size_t>(n_cells),
                  "tensor AV hydro_active size mismatch");
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
             "tensor AV: cudaMalloc hydro_active failed");
  cuda_check(cudaMemcpy(d_active, active.data(),
                        active.size() * sizeof(std::int8_t),
                        cudaMemcpyHostToDevice),
             "tensor AV: cudaMemcpy hydro_active failed");
  return d_active;
}

TensorKernelParams tensor_params(const core::Config& cfg) {
  TensorKernelParams params;
  params.c1 = cfg.numerics.hydro.tensor_av_C1;
  params.c2 = cfg.numerics.hydro.tensor_av_C2;
  params.gamma = cfg.materials.materials.empty()
                     ? (5.0 / 3.0)
                     : cfg.materials.materials.front().ideal_gas_gamma;
  params.limiter_enabled = cfg.numerics.hydro.csw_limiter_enabled ? 1 : 0;
  return params;
}

TensorTopologyDevice structured_topology(const core::State& state) {
  TensorTopologyDevice topo;
  topo.nr = state.mesh.topo.nr;
  topo.nz = state.mesh.topo.nz;
  topo.corner_stride = state.corner_stride;
  return topo;
}

TensorTopologyDevice multiblock_topology(
    const core::State& state,
    const std::uint8_t* cell_nverts,
    const thrust::device_vector<int>& face_adj_offsets,
    const thrust::device_vector<int>& face_adj_indices) {
  TensorTopologyDevice topo;
  topo.cell_node_offsets = state.mesh.multiblock_cell_node_csr_offsets.data();
  topo.cell_node_indices = state.mesh.multiblock_cell_node_csr_indices.data();
  topo.face_adj_offsets = raw_or_null(face_adj_offsets);
  topo.face_adj_indices = raw_or_null(face_adj_indices);
  topo.cell_nverts = cell_nverts;
  topo.corner_stride = state.corner_stride;
  topo.multiblock = 1;
  return topo;
}

void assert_multiblock_tensor_topology(const core::State& state) {
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "tensor AV multiblock topology metadata missing");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "tensor AV requires multiblock cell-node CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) *
                        static_cast<std::size_t>(state.corner_stride),
                "tensor AV requires multiblock cell-node CSR indices");
  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(mb.face_adj_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "tensor AV requires face adjacency CSR offsets");
  TENRYU_ASSERT(mb.face_adj_csr_indices.size() >=
                    static_cast<std::size_t>(n_cells) *
                        static_cast<std::size_t>(state.corner_stride),
                "tensor AV requires face adjacency CSR indices");
}

}  // namespace

void launch_compute_tensor_av_2d(
    core::State& state,
    const core::Config& cfg,
    const core::CellField1D& cell_cs,
    const double* v_r,
    const double* v_z,
    const std::int8_t* hydro_active,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  if (cfg.main.dim != 2 ||
      cfg.numerics.hydro.av_model != core::AvModel::MimeticTensorV1) {
    return;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(cell_cs.size() == state.rho.size(),
                "tensor AV requires cell sound speed per cell");
  TENRYU_ASSERT(v_r != nullptr && v_z != nullptr,
                "tensor AV requires nodal velocity arrays");
  ensure_compatible_force_work_buffers(state, cfg, true);
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  TENRYU_ASSERT(state.corner_force_q_r.size() == n_corners &&
                    state.corner_force_q_z.size() == n_corners,
                "tensor AV corner-force buffers are not allocated");
  if (n_cells <= 0) {
    state.tensor_pc_degenerate = 0;
    return;
  }
  cuda_check(cudaMemset(state.corner_force_q_r.data(), 0,
                        n_corners * sizeof(double)),
             "tensor AV: zero corner_force_q_r failed");
  cuda_check(cudaMemset(state.corner_force_q_z.data(), 0,
                        n_corners * sizeof(double)),
             "tensor AV: zero corner_force_q_z failed");
  if (!state.edge_force_av_r.empty()) {
    cuda_check(cudaMemset(state.edge_force_av_r.data(), 0,
                          state.edge_force_av_r.size() * sizeof(double)),
               "tensor AV: zero edge_force_av_r failed");
    cuda_check(cudaMemset(state.edge_force_av_z.data(), 0,
                          state.edge_force_av_z.size() * sizeof(double)),
               "tensor AV: zero edge_force_av_z failed");
  }
  core::CellField1D sigma{"tensor_av_sigma"};
  core::CellField1D du_cell{"tensor_av_du"};
  core::CellField1D length_a{"tensor_av_length_a"};
  core::CellField1D length_b{"tensor_av_length_b"};
  core::CellField1D mu_a{"tensor_av_mu"};
  core::CellField1D mu_b{"tensor_av_mu_b"};
  sigma.reset(state.rho.size());
  du_cell.reset(state.rho.size());
  length_a.reset(state.rho.size());
  length_b.reset(state.rho.size());
  mu_a.reset(state.rho.size());
  mu_b.reset(state.rho.size());
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* cell_nverts =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state);
  thrust::device_vector<int> d_face_adj_offsets;
  thrust::device_vector<int> d_face_adj_indices;
  TensorTopologyDevice topo = structured_topology(state);
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    assert_multiblock_tensor_topology(state);
    const auto& mb = *state.mesh.topo.multiblock;
    d_face_adj_offsets.assign(mb.face_adj_csr_offsets.begin(),
                              mb.face_adj_csr_offsets.end());
    d_face_adj_indices.assign(mb.face_adj_csr_indices.begin(),
                              mb.face_adj_csr_indices.end());
    topo = multiblock_topology(state, cell_nverts, d_face_adj_offsets,
                               d_face_adj_indices);
  }
  const TensorKernelParams params = tensor_params(cfg);
  const int blocks = (n_cells + 255) / 256;
  tensor_sensor_l0_kernel<<<blocks, 256>>>(
      sigma.data(), du_cell.data(), length_a.data(), state.x_r.data(),
      state.x_z.data(), v_r, v_z, cell_cs.data(), hydro_active, n_cells, topo,
      params, aw_axis_slave_theta0_active, aw_axis_slave_theta_pi_active);
  tensor_jacobi_length_kernel<<<blocks, 256>>>(
      length_b.data(), length_a.data(), hydro_active, n_cells, topo);
  tensor_jacobi_length_kernel<<<blocks, 256>>>(
      length_a.data(), length_b.data(), hydro_active, n_cells, topo);
  tensor_mu_raw_kernel<<<blocks, 256>>>(
      mu_a.data(), sigma.data(), du_cell.data(), length_a.data(),
      state.rho.data(), cell_cs.data(), hydro_active, n_cells, params);
  tensor_jacobi_length_kernel<<<blocks, 256>>>(
      mu_b.data(), mu_a.data(), hydro_active, n_cells, topo);
  tensor_jacobi_length_kernel<<<blocks, 256>>>(
      mu_a.data(), mu_b.data(), hydro_active, n_cells, topo);
  int* d_pc_degenerate = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_pc_degenerate), sizeof(int)),
             "tensor AV: cudaMalloc tensor_pc_degenerate failed");
  cuda_check(cudaMemset(d_pc_degenerate, 0, sizeof(int)),
             "tensor AV: zero tensor_pc_degenerate failed");
  tensor_force_mu_kernel<<<blocks, 256>>>(
      state.corner_force_q_r.data(), state.corner_force_q_z.data(),
      mu_a.data(), d_pc_degenerate, length_a.data(), state.x_r.data(),
      state.x_z.data(), v_r, v_z, hydro_active, n_cells, topo);
  cuda_check(cudaGetLastError(), "tensor AV kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "tensor AV kernel execution failed");
  cuda_check(cudaMemcpy(&state.tensor_pc_degenerate, d_pc_degenerate,
                        sizeof(int), cudaMemcpyDeviceToHost),
             "tensor AV: copy tensor_pc_degenerate failed");
  cuda_check(cudaFree(d_pc_degenerate),
             "tensor AV: cudaFree tensor_pc_degenerate failed");
  launch_compute_tensor_av_work_2d(state, cfg, v_r, v_z, hydro_active);
}

void launch_sum_tensor_av_corner_forces_to_nodes_2d(
    double* force_r,
    double* force_z,
    const std::int8_t* hydro_active,
    const core::State& state) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  TENRYU_ASSERT(force_r != nullptr && force_z != nullptr,
                "tensor AV nodal gather requires force buffers");
  TENRYU_ASSERT(state.corner_force_q_r.size() == n_corners &&
                    state.corner_force_q_z.size() == n_corners,
                "tensor AV nodal gather requires corner-force buffers");
  if (n_nodes <= 0) {
    return;
  }
  const int blocks = (n_nodes + 255) / 256;
  if (state.mesh.topo.multiblock.has_value()) {
    TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_offsets.size() ==
                      static_cast<std::size_t>(n_nodes) + 1U,
                  "tensor AV multiblock gather requires reverse CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_cells.size() ==
                      state.mesh.multiblock_reverse_csr_node_corners.size(),
                  "tensor AV multiblock gather reverse CSR payload mismatch");
    tensor_sum_corner_forces_multiblock_kernel<<<blocks, 256>>>(
        force_r, force_z, state.corner_force_q_r.data(),
        state.corner_force_q_z.data(), hydro_active,
        state.mesh.multiblock_reverse_csr_node_offsets.data(),
        state.mesh.multiblock_reverse_csr_node_cells.data(),
        state.mesh.multiblock_reverse_csr_node_corners.data(), n_nodes,
        state.corner_stride);
  } else {
    tensor_sum_corner_forces_structured_kernel<<<blocks, 256>>>(
        force_r, force_z, state.corner_force_q_r.data(),
        state.corner_force_q_z.data(), hydro_active, state.mesh.topo.nr,
        state.mesh.topo.nz, state.corner_stride);
  }
}

void launch_compute_tensor_av_work_2d(
    core::State& state,
    const core::Config& cfg,
    const double* v_r,
    const double* v_z,
    const std::int8_t* hydro_active) {
  if (cfg.numerics.hydro.av_model != core::AvModel::MimeticTensorV1) {
    return;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(state.work_av_per_cell.size() == state.rho.size(),
                "tensor AV work buffer is not allocated");
  if (!state.work_av_per_cell.empty()) {
    cuda_check(cudaMemset(state.work_av_per_cell.data(), 0,
                          state.work_av_per_cell.size() * sizeof(double)),
               "tensor AV: zero work_av_per_cell failed");
  }
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  if (state.corner_force_q_r.size() != n_corners ||
      state.corner_force_q_z.size() != n_corners || n_cells <= 0) {
    return;
  }
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* cell_nverts =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state);
  TensorTopologyDevice topo = structured_topology(state);
  if (state.mesh.topo.multiblock.has_value()) {
    topo.cell_node_offsets = state.mesh.multiblock_cell_node_csr_offsets.data();
    topo.cell_node_indices = state.mesh.multiblock_cell_node_csr_indices.data();
    topo.cell_nverts = cell_nverts;
    topo.multiblock = 1;
  }
  const int blocks = (n_cells + 255) / 256;
  tensor_work_kernel<<<blocks, 256>>>(
      state.work_av_per_cell.data(), state.corner_force_q_r.data(),
      state.corner_force_q_z.data(), state.x_r.data(), v_r, v_z, hydro_active,
      n_cells, topo);
}

double compute_tensor_av_cfl_dt(
    const core::State& state,
    const core::Config& cfg,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active,
    TensorAvCflArgmin* argmin) {
  if (argmin != nullptr) {
    *argmin = TensorAvCflArgmin{};
  }
  if (cfg.main.dim != 2 ||
      cfg.numerics.hydro.av_model != core::AvModel::MimeticTensorV1 ||
      state.rho.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(state.cs.size() == state.rho.size(),
                "tensor AV CFL requires cell sound speed per cell");
  std::int8_t* d_active = upload_hydro_active_or_null(state, cfg);
  core::CellField1D sigma{"tensor_av_cfl_sigma"};
  core::CellField1D du_cell{"tensor_av_cfl_du"};
  core::CellField1D length_a{"tensor_av_cfl_length_a"};
  core::CellField1D length_b{"tensor_av_cfl_length_b"};
  core::CellField1D mu_a{"tensor_av_cfl_mu"};
  core::CellField1D mu_b{"tensor_av_cfl_mu_b"};
  sigma.reset(state.rho.size());
  du_cell.reset(state.rho.size());
  length_a.reset(state.rho.size());
  length_b.reset(state.rho.size());
  mu_a.reset(state.rho.size());
  mu_b.reset(state.rho.size());
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* cell_nverts =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state);
  thrust::device_vector<int> d_face_adj_offsets;
  thrust::device_vector<int> d_face_adj_indices;
  TensorTopologyDevice topo = structured_topology(state);
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    assert_multiblock_tensor_topology(state);
    const auto& mb = *state.mesh.topo.multiblock;
    d_face_adj_offsets.assign(mb.face_adj_csr_offsets.begin(),
                              mb.face_adj_csr_offsets.end());
    d_face_adj_indices.assign(mb.face_adj_csr_indices.begin(),
                              mb.face_adj_csr_indices.end());
    topo = multiblock_topology(state, cell_nverts, d_face_adj_offsets,
                               d_face_adj_indices);
  }
  const TensorKernelParams params = tensor_params(cfg);
  const int blocks = (n_cells + 255) / 256;
  tensor_sensor_l0_kernel<<<blocks, 256>>>(
      sigma.data(), du_cell.data(), length_a.data(), state.x_r.data(),
      state.x_z.data(), state.v_r.data(), state.v_z.data(), state.cs.data(),
      d_active, n_cells, topo, params, aw_axis_slave_theta0_active,
      aw_axis_slave_theta_pi_active);
  tensor_jacobi_length_kernel<<<blocks, 256>>>(
      length_b.data(), length_a.data(), d_active, n_cells, topo);
  tensor_jacobi_length_kernel<<<blocks, 256>>>(
      length_a.data(), length_b.data(), d_active, n_cells, topo);
  tensor_mu_raw_kernel<<<blocks, 256>>>(
      mu_a.data(), sigma.data(), du_cell.data(), length_a.data(),
      state.rho.data(), state.cs.data(), d_active, n_cells, params);
  tensor_jacobi_length_kernel<<<blocks, 256>>>(
      mu_b.data(), mu_a.data(), d_active, n_cells, topo);
  tensor_jacobi_length_kernel<<<blocks, 256>>>(
      mu_a.data(), mu_b.data(), d_active, n_cells, topo);
  double* d_min_dt = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_min_dt), sizeof(double)),
             "tensor AV CFL: cudaMalloc min_dt failed");
  const double inf = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(d_min_dt, &inf, sizeof(double), cudaMemcpyHostToDevice),
             "tensor AV CFL: init min_dt failed");
  tensor_cfl_kernel<<<blocks, 256>>>(
      d_min_dt, nullptr, 0.0, mu_a.data(), length_a.data(), state.rho.data(),
      d_active, n_cells);
  cuda_check(cudaGetLastError(), "tensor AV CFL kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "tensor AV CFL kernel execution failed");
  double min_dt = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(&min_dt, d_min_dt, sizeof(double), cudaMemcpyDeviceToHost),
             "tensor AV CFL: copy min_dt failed");
  if (argmin != nullptr && std::isfinite(min_dt)) {
    int* d_winner_cell = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_winner_cell), sizeof(int)),
               "tensor AV CFL: cudaMalloc winner_cell failed");
    cuda_check(cudaMemcpy(d_winner_cell, &n_cells, sizeof(int),
                          cudaMemcpyHostToDevice),
               "tensor AV CFL: init winner_cell failed");
    tensor_cfl_kernel<<<blocks, 256>>>(
        d_min_dt, d_winner_cell, min_dt, mu_a.data(), length_a.data(),
        state.rho.data(), d_active, n_cells);
    cuda_check(cudaGetLastError(),
               "tensor AV CFL winner kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "tensor AV CFL winner kernel execution failed");
    int winner_cell = n_cells;
    cuda_check(cudaMemcpy(&winner_cell, d_winner_cell, sizeof(int),
                          cudaMemcpyDeviceToHost),
               "tensor AV CFL: copy winner_cell failed");
    if (winner_cell >= 0 && winner_cell < n_cells) {
      argmin->dt = min_dt;
      argmin->cell_id = winner_cell;
      cuda_check(cudaMemcpy(&argmin->rho, state.rho.data() + winner_cell,
                            sizeof(double), cudaMemcpyDeviceToHost),
                 "tensor AV CFL: copy winner rho failed");
      cuda_check(cudaMemcpy(&argmin->length, length_a.data() + winner_cell,
                            sizeof(double), cudaMemcpyDeviceToHost),
                 "tensor AV CFL: copy winner length failed");
      cuda_check(cudaMemcpy(&argmin->mu, mu_a.data() + winner_cell,
                            sizeof(double), cudaMemcpyDeviceToHost),
                 "tensor AV CFL: copy winner mu failed");
    }
    cuda_check(cudaFree(d_winner_cell),
               "tensor AV CFL: cudaFree winner_cell failed");
  }
  if (d_active != nullptr) {
    cuda_check(cudaFree(d_active),
               "tensor AV CFL: cudaFree hydro_active failed");
  }
  cuda_check(cudaFree(d_min_dt), "tensor AV CFL: cudaFree min_dt failed");
  return min_dt;
}

}  // namespace tenryu::hydro::compatible
