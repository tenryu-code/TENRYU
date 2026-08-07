#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "hydro/rz_corner_mass.cuh"

namespace tenryu::hydro::ale::staggered_subcell_remap {

/*
Design-doc block: staggered subcell intersection-distribution remap.

This header is a cfg-free, default-off primitive module for the
Loubere-Shashkov / Kenamond-Kuzmin-Shashkov staggered remap used by the
2D RZ indirect-ALE path.  It does not call into the ALE driver and does
not own mesh state.  All topology, coordinates, fields, overlap CSR, and
temporary buffers are explicit function parameters.

Geometry and units:
  - Coordinates are cgs centimeters in the RZ meridian.
  - RZ volumes are exact polygon volumes V = 2*pi*int r dA, evaluated by
    rz_polygon_volume_exact on each staggered corner subcell.
  - Mass is grams, momentum is g*cm/s, and energy is erg.  No internal
    unit conversion is performed here; eV floors remain a caller concern.

Subcell gather:
  For a cell c with active vertices k, define source subcells s=(c,k) by
  the ordered polygon
    [x_k, 0.5*(x_k+x_{k+1}), x_c, 0.5*(x_{k-1}+x_k)].
  With V_s from rz_polygon_volume_exact and V_c=sum_s V_s, this module
  gathers
    m_s = m_c V_s/V_c,
    p_s = m_s u_k,
    I_s = m_s e_c,
    K_s = 0.5 m_s |u_k|^2.
  Therefore sum_{s in c} m_s = m_c and sum_{s in c} I_s = m_c e_c.

Intersection/distribution:
  Given a source-subcell CSR adjacency to candidate target subcells
  s_tilde, the intersection kernel computes
    W_{s_tilde,s} = |Omega_{s_tilde} intersect Omega_s|_RZ
  by convex clipping in the RZ meridian and exact RZ polygon volume.  The
  distribution weights are normalized per source subcell,
    A_{s_tilde,s} = W_{s_tilde,s}/sum_q W_{q,s},
  with the final positive entry adjusted so
    sum_{s_tilde} A_{s_tilde,s} = 1.

Remap conservation:
  Target subcell conserved packets are accumulated as
    Q_{s_tilde} = sum_s A_{s_tilde,s} Q_s,
  independently for m, p_r, p_z, I, and K.  For every source subcell whose
  weight status is OK, sum_{s_tilde} A_{s_tilde,s} Q_s = Q_s; hence global
  sums of the remapped packet fields are conserved up to atomic roundoff.

Scatter and compatible total-energy recovery:
  Node velocity is recovered from remapped subcell momenta:
    u_n = (sum_{s_tilde -> n} p_{s_tilde}) /
          (sum_{s_tilde -> n} m_{s_tilde}).
  For each cell c, define
    E_c^sub = sum_{s_tilde in c} (I_{s_tilde}+K_{s_tilde}),
    K_c^node = sum_{s_tilde in c} 0.5 m_{s_tilde}|u_{n(s_tilde)}|^2.
  The compatible total-energy fix returns
    I_c = E_c^sub - K_c^node.
  The optional nonnegative finalizer conservatively redistributes any
  negative-cell deficit over positive internal-energy cells.  If the raw
  global internal energy is negative, nonnegative conservation is
  impossible; the finalizer reports the unresolved deficit in fix_sums[3].
*/

constexpr int kSubcellsPerCell = 4;
constexpr int kMaxPolygonVerts = 16;
constexpr double kTiny = 1.0e-300;
constexpr double kDoubleEps =
    2.220446049250313080847263336181640625e-16;

enum class WeightStatus : std::uint8_t {
  OK = 0,
  INACTIVE_SOURCE = 1,
  EMPTY_OVERLAP = 2,
};

enum CompatibleEnergyFixSum : int {
  kFixPositive = 0,
  kFixDeficit = 1,
  kFixRawTotal = 2,
  kFixUnresolvedDeficit = 3,
  kFixCount = 4,
};

namespace detail {

__host__ __device__ inline double max2(const double a, const double b) {
  return a > b ? a : b;
}

__host__ __device__ inline double min2(const double a, const double b) {
  return a < b ? a : b;
}

__host__ __device__ inline double nonnegative_finite(const double x) {
  return tenryu::hydro::rz::finite_double(x) && x > 0.0 ? x : 0.0;
}

__host__ __device__ inline int active_nverts(
    const std::uint8_t* __restrict__ cell_nverts,
    const int cell) {
  if (cell_nverts == nullptr) {
    return 4;
  }
  const int nverts = static_cast<int>(cell_nverts[cell]);
  return nverts == 3 ? 3 : 4;
}

__host__ __device__ inline int subcell_index(const int cell,
                                             const int corner) {
  return cell * kSubcellsPerCell + corner;
}

__host__ __device__ inline double cross2(const double ar,
                                         const double az,
                                         const double br,
                                         const double bz) {
  return ar * bz - az * br;
}

__host__ __device__ inline double polygon_area2(const double* r,
                                                const double* z,
                                                const int nverts) {
  return tenryu::hydro::rz::rz_polygon_area2_exact(r, z, nverts);
}

__host__ __device__ inline double polygon_length_scale(const double* r,
                                                       const double* z,
                                                       const int nverts) {
  double scale = 1.0e-300;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : k + 1;
    const double er = r[kp] - r[k];
    const double ez = z[kp] - z[k];
    scale = max2(scale, max2(fabs(r[k]), fabs(z[k])));
    scale = max2(scale, sqrt(er * er + ez * ez));
  }
  return scale;
}

__host__ __device__ inline bool load_cell_nodes(
    const int cell,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    double* r,
    double* z,
    int* node_ids) {
  const int nverts = active_nverts(cell_nverts, cell);
  const int off = cell_node_csr_offsets[cell];
  for (int k = 0; k < nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    if (node_ids != nullptr) {
      node_ids[k] = n;
    }
    r[k] = x_r[n];
    z[k] = x_z[n];
    if (!tenryu::hydro::rz::finite_double(r[k]) ||
        !tenryu::hydro::rz::finite_double(z[k])) {
      return false;
    }
  }
  return true;
}

__host__ __device__ inline int build_corner_subcell_polygon(
    const int cell,
    const int corner,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    double* r_sub,
    double* z_sub,
    int* corner_node) {
  const int nverts = active_nverts(cell_nverts, cell);
  if (corner < 0 || corner >= nverts) {
    if (corner_node != nullptr) {
      *corner_node = -1;
    }
    return 0;
  }

  double r[4] = {0.0, 0.0, 0.0, 0.0};
  double z[4] = {0.0, 0.0, 0.0, 0.0};
  int node_ids[4] = {-1, -1, -1, -1};
  if (!load_cell_nodes(cell,
                       x_r,
                       x_z,
                       cell_node_csr_offsets,
                       cell_node_csr_indices,
                       cell_nverts,
                       r,
                       z,
                       node_ids)) {
    if (corner_node != nullptr) {
      *corner_node = -1;
    }
    return 0;
  }

  double rc = 0.0;
  double zc = 0.0;
  for (int k = 0; k < nverts; ++k) {
    rc += r[k];
    zc += z[k];
  }
  rc /= static_cast<double>(nverts);
  zc /= static_cast<double>(nverts);

  const int kp = (corner + 1 == nverts) ? 0 : corner + 1;
  const int km = (corner + nverts - 1) % nverts;
  r_sub[0] = r[corner];
  z_sub[0] = z[corner];
  r_sub[1] = 0.5 * (r[corner] + r[kp]);
  z_sub[1] = 0.5 * (z[corner] + z[kp]);
  r_sub[2] = rc;
  z_sub[2] = zc;
  r_sub[3] = 0.5 * (r[km] + r[corner]);
  z_sub[3] = 0.5 * (z[km] + z[corner]);

  if (corner_node != nullptr) {
    *corner_node = node_ids[corner];
  }
  return 4;
}

__host__ __device__ inline double subcell_volume_exact(
    const int subcell,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  const int cell = subcell / kSubcellsPerCell;
  const int corner = subcell - cell * kSubcellsPerCell;
  double r[4] = {0.0, 0.0, 0.0, 0.0};
  double z[4] = {0.0, 0.0, 0.0, 0.0};
  const int nverts = build_corner_subcell_polygon(cell,
                                                  corner,
                                                  x_r,
                                                  x_z,
                                                  cell_node_csr_offsets,
                                                  cell_node_csr_indices,
                                                  cell_nverts,
                                                  r,
                                                  z,
                                                  nullptr);
  if (nverts < 3) {
    return 0.0;
  }
  const double V =
      fabs(tenryu::hydro::rz::rz_polygon_volume_exact(r, z, nverts));
  return tenryu::hydro::rz::finite_double(V) ? V : 0.0;
}

__host__ __device__ inline bool inside_clip_edge(const double qr,
                                                 const double qz,
                                                 const double ar,
                                                 const double az,
                                                 const double br,
                                                 const double bz,
                                                 const double clip_area2,
                                                 const double tol) {
  const double c = cross2(br - ar, bz - az, qr - ar, qz - az);
  return clip_area2 >= 0.0 ? c >= -tol : c <= tol;
}

__host__ __device__ inline void line_intersection(const double p0r,
                                                  const double p0z,
                                                  const double p1r,
                                                  const double p1z,
                                                  const double ar,
                                                  const double az,
                                                  const double br,
                                                  const double bz,
                                                  double* rr,
                                                  double* zz) {
  const double dr = p1r - p0r;
  const double dz = p1z - p0z;
  const double er = br - ar;
  const double ez = bz - az;
  const double denom = cross2(dr, dz, er, ez);
  const double scale = max2(sqrt(dr * dr + dz * dz),
                            sqrt(er * er + ez * ez));
  const double tol = 256.0 * kDoubleEps * max2(scale * scale, 1.0e-300);
  if (!(fabs(denom) > tol) || !tenryu::hydro::rz::finite_double(denom)) {
    *rr = p1r;
    *zz = p1z;
    return;
  }
  double t = cross2(ar - p0r, az - p0z, er, ez) / denom;
  if (!tenryu::hydro::rz::finite_double(t)) {
    t = 0.0;
  }
  t = max2(0.0, min2(1.0, t));
  *rr = p0r + t * dr;
  *zz = p0z + t * dz;
}

__host__ __device__ inline int clip_polygon_against_edge(
    const double* in_r,
    const double* in_z,
    const int in_n,
    const double ar,
    const double az,
    const double br,
    const double bz,
    const double clip_area2,
    const double tol,
    double* out_r,
    double* out_z) {
  if (in_n <= 0) {
    return 0;
  }

  int out_n = 0;
  double sr = in_r[in_n - 1];
  double sz = in_z[in_n - 1];
  bool s_inside = inside_clip_edge(sr, sz, ar, az, br, bz, clip_area2, tol);

  for (int i = 0; i < in_n; ++i) {
    const double er = in_r[i];
    const double ez = in_z[i];
    const bool e_inside =
        inside_clip_edge(er, ez, ar, az, br, bz, clip_area2, tol);

    if (e_inside != s_inside && out_n < kMaxPolygonVerts) {
      line_intersection(sr, sz, er, ez, ar, az, br, bz, &out_r[out_n],
                        &out_z[out_n]);
      ++out_n;
    }
    if (e_inside && out_n < kMaxPolygonVerts) {
      out_r[out_n] = er;
      out_z[out_n] = ez;
      ++out_n;
    }

    sr = er;
    sz = ez;
    s_inside = e_inside;
  }
  return out_n;
}

__host__ __device__ inline double subcell_intersection_volume_exact(
    const int source_subcell,
    const int target_subcell,
    const double* __restrict__ x_r_source,
    const double* __restrict__ x_z_source,
    const double* __restrict__ x_r_target,
    const double* __restrict__ x_z_target,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  const int source_cell = source_subcell / kSubcellsPerCell;
  const int source_corner = source_subcell - source_cell * kSubcellsPerCell;
  const int target_cell = target_subcell / kSubcellsPerCell;
  const int target_corner = target_subcell - target_cell * kSubcellsPerCell;

  double source_r[4] = {0.0, 0.0, 0.0, 0.0};
  double source_z[4] = {0.0, 0.0, 0.0, 0.0};
  double target_r[4] = {0.0, 0.0, 0.0, 0.0};
  double target_z[4] = {0.0, 0.0, 0.0, 0.0};
  const int source_n =
      build_corner_subcell_polygon(source_cell,
                                   source_corner,
                                   x_r_source,
                                   x_z_source,
                                   cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_nverts,
                                   source_r,
                                   source_z,
                                   nullptr);
  const int target_n =
      build_corner_subcell_polygon(target_cell,
                                   target_corner,
                                   x_r_target,
                                   x_z_target,
                                   cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_nverts,
                                   target_r,
                                   target_z,
                                   nullptr);
  if (source_n < 3 || target_n < 3) {
    return 0.0;
  }

  const double clip_area2 = polygon_area2(target_r, target_z, target_n);
  const double scale =
      max2(polygon_length_scale(source_r, source_z, source_n),
           polygon_length_scale(target_r, target_z, target_n));
  const double tol = 1024.0 * kDoubleEps * max2(scale * scale, 1.0e-300);
  if (!(fabs(clip_area2) > tol) ||
      !tenryu::hydro::rz::finite_double(clip_area2)) {
    return 0.0;
  }

  double work0_r[kMaxPolygonVerts];
  double work0_z[kMaxPolygonVerts];
  double work1_r[kMaxPolygonVerts];
  double work1_z[kMaxPolygonVerts];
  int n_work = source_n;
  for (int k = 0; k < source_n; ++k) {
    work0_r[k] = source_r[k];
    work0_z[k] = source_z[k];
  }

  for (int edge = 0; edge < target_n; ++edge) {
    const int ep = (edge + 1 == target_n) ? 0 : edge + 1;
    n_work = clip_polygon_against_edge(work0_r,
                                       work0_z,
                                       n_work,
                                       target_r[edge],
                                       target_z[edge],
                                       target_r[ep],
                                       target_z[ep],
                                       clip_area2,
                                       tol,
                                       work1_r,
                                       work1_z);
    if (n_work < 3) {
      return 0.0;
    }
    for (int k = 0; k < n_work; ++k) {
      work0_r[k] = work1_r[k];
      work0_z[k] = work1_z[k];
    }
  }

  const double V = fabs(
      tenryu::hydro::rz::rz_polygon_volume_exact(work0_r, work0_z, n_work));
  return tenryu::hydro::rz::finite_double(V) ? V : 0.0;
}

__host__ __device__ inline int corner_node_for_subcell(
    const int subcell,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts) {
  const int cell = subcell / kSubcellsPerCell;
  const int corner = subcell - cell * kSubcellsPerCell;
  double r[4] = {0.0, 0.0, 0.0, 0.0};
  double z[4] = {0.0, 0.0, 0.0, 0.0};
  int node = -1;
  build_corner_subcell_polygon(cell,
                               corner,
                               x_r,
                               x_z,
                               cell_node_csr_offsets,
                               cell_node_csr_indices,
                               cell_nverts,
                               r,
                               z,
                               &node);
  return node;
}

}  // namespace detail

#ifdef __CUDACC__

__global__ void compute_subcell_volumes_kernel(
    double* __restrict__ subcell_volume,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  for (int k = 0; k < kSubcellsPerCell; ++k) {
    const int s = detail::subcell_index(c, k);
    subcell_volume[s] =
        detail::subcell_volume_exact(s,
                                     x_r,
                                     x_z,
                                     cell_node_csr_offsets,
                                     cell_node_csr_indices,
                                     cell_nverts);
  }
}

__global__ void gather_cell_nodal_conserved_to_subcells_kernel(
    double* __restrict__ subcell_volume,
    double* __restrict__ subcell_mass,
    double* __restrict__ subcell_momentum_r,
    double* __restrict__ subcell_momentum_z,
    double* __restrict__ subcell_internal_energy,
    double* __restrict__ subcell_kinetic_energy,
    const double* __restrict__ cell_mass,
    const double* __restrict__ cell_density,
    const double* __restrict__ cell_specific_internal_energy,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  double V_sub[kSubcellsPerCell] = {0.0, 0.0, 0.0, 0.0};
  double V_cell = 0.0;
  for (int k = 0; k < kSubcellsPerCell; ++k) {
    const int s = detail::subcell_index(c, k);
    V_sub[k] = detail::subcell_volume_exact(s,
                                            x_r,
                                            x_z,
                                            cell_node_csr_offsets,
                                            cell_node_csr_indices,
                                            cell_nverts);
    V_cell += V_sub[k];
    if (subcell_volume != nullptr) {
      subcell_volume[s] = V_sub[k];
    }
  }

  double m_cell = 0.0;
  if (cell_mass != nullptr && tenryu::hydro::rz::finite_double(cell_mass[c]) &&
      cell_mass[c] > 0.0) {
    m_cell = cell_mass[c];
  } else if (cell_density != nullptr && V_cell > 0.0 &&
             tenryu::hydro::rz::finite_double(cell_density[c])) {
    m_cell = detail::nonnegative_finite(cell_density[c]) * V_cell;
  }

  const double e_int =
      cell_specific_internal_energy != nullptr
          ? detail::nonnegative_finite(cell_specific_internal_energy[c])
          : 0.0;
  const int off = cell_node_csr_offsets[c];
  const int active_nverts = detail::active_nverts(cell_nverts, c);
  const double inv_V_cell = V_cell > 0.0 ? 1.0 / V_cell : 0.0;

  for (int k = 0; k < kSubcellsPerCell; ++k) {
    const int s = detail::subcell_index(c, k);
    double m = 0.0;
    double pr = 0.0;
    double pz = 0.0;
    double I = 0.0;
    double K = 0.0;
    if (k < active_nverts && inv_V_cell > 0.0) {
      const int n = cell_node_csr_indices[off + k];
      const double vr =
          tenryu::hydro::rz::finite_double(v_r_node[n]) ? v_r_node[n] : 0.0;
      const double vz =
          tenryu::hydro::rz::finite_double(v_z_node[n]) ? v_z_node[n] : 0.0;
      m = m_cell * V_sub[k] * inv_V_cell;
      pr = m * vr;
      pz = m * vz;
      I = m * e_int;
      K = 0.5 * m * (vr * vr + vz * vz);
    }
    subcell_mass[s] = m;
    subcell_momentum_r[s] = pr;
    subcell_momentum_z[s] = pz;
    subcell_internal_energy[s] = I;
    subcell_kinetic_energy[s] = K;
  }
}

__global__ void build_intersection_distribution_weights_kernel(
    double* __restrict__ distribution_weight,
    double* __restrict__ intersection_volume,
    double* __restrict__ source_coverage_ratio,
    std::uint8_t* __restrict__ source_status,
    const int* __restrict__ source_pair_offsets,
    const int* __restrict__ pair_target_subcell,
    const double* __restrict__ x_r_source,
    const double* __restrict__ x_z_source,
    const double* __restrict__ x_r_target,
    const double* __restrict__ x_z_target,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_source_subcells) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_source_subcells) {
    return;
  }

  const int off = source_pair_offsets[s];
  const int end = source_pair_offsets[s + 1];
  const double V_source =
      detail::subcell_volume_exact(s,
                                   x_r_source,
                                   x_z_source,
                                   cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_nverts);
  if (!(V_source > 0.0)) {
    for (int e = off; e < end; ++e) {
      distribution_weight[e] = 0.0;
      if (intersection_volume != nullptr) {
        intersection_volume[e] = 0.0;
      }
    }
    if (source_coverage_ratio != nullptr) {
      source_coverage_ratio[s] = 0.0;
    }
    if (source_status != nullptr) {
      source_status[s] =
          static_cast<std::uint8_t>(WeightStatus::INACTIVE_SOURCE);
    }
    return;
  }

  double V_sum = 0.0;
  int last_positive = -1;
  for (int e = off; e < end; ++e) {
    const int st = pair_target_subcell[e];
    const double V =
        detail::subcell_intersection_volume_exact(s,
                                                  st,
                                                  x_r_source,
                                                  x_z_source,
                                                  x_r_target,
                                                  x_z_target,
                                                  cell_node_csr_offsets,
                                                  cell_node_csr_indices,
                                                  cell_nverts);
    const double Vp = detail::nonnegative_finite(V);
    if (intersection_volume != nullptr) {
      intersection_volume[e] = Vp;
    }
    if (Vp > 0.0) {
      V_sum += Vp;
      last_positive = e;
    }
  }

  if (source_coverage_ratio != nullptr) {
    source_coverage_ratio[s] = V_sum / V_source;
  }
  if (!(V_sum > 0.0) || last_positive < 0) {
    for (int e = off; e < end; ++e) {
      distribution_weight[e] = 0.0;
    }
    if (source_status != nullptr) {
      source_status[s] =
          static_cast<std::uint8_t>(WeightStatus::EMPTY_OVERLAP);
    }
    return;
  }

  double sum_A = 0.0;
  for (int e = off; e < end; ++e) {
    double A = 0.0;
    const double V =
        intersection_volume != nullptr
            ? intersection_volume[e]
            : detail::subcell_intersection_volume_exact(s,
                                                        pair_target_subcell[e],
                                                        x_r_source,
                                                        x_z_source,
                                                        x_r_target,
                                                        x_z_target,
                                                        cell_node_csr_offsets,
                                                        cell_node_csr_indices,
                                                        cell_nverts);
    if (e == last_positive) {
      A = detail::max2(0.0, 1.0 - sum_A);
    } else if (V > 0.0 && tenryu::hydro::rz::finite_double(V)) {
      A = V / V_sum;
      sum_A += A;
    }
    distribution_weight[e] = tenryu::hydro::rz::finite_double(A) ? A : 0.0;
  }
  if (source_status != nullptr) {
    source_status[s] = static_cast<std::uint8_t>(WeightStatus::OK);
  }
}

__global__ void build_distribution_weights_from_overlap_volumes_kernel(
    double* __restrict__ distribution_weight,
    double* __restrict__ source_coverage_ratio,
    std::uint8_t* __restrict__ source_status,
    const int* __restrict__ source_pair_offsets,
    const double* __restrict__ pair_overlap_volume,
    const double* __restrict__ source_subcell_volume,
    const int n_source_subcells) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_source_subcells) {
    return;
  }
  const int off = source_pair_offsets[s];
  const int end = source_pair_offsets[s + 1];
  const double V_source =
      source_subcell_volume != nullptr ? source_subcell_volume[s] : 0.0;
  double V_sum = 0.0;
  int last_positive = -1;
  for (int e = off; e < end; ++e) {
    const double V = detail::nonnegative_finite(pair_overlap_volume[e]);
    if (V > 0.0) {
      V_sum += V;
      last_positive = e;
    }
  }

  if (source_coverage_ratio != nullptr) {
    source_coverage_ratio[s] = V_source > 0.0 ? V_sum / V_source : 0.0;
  }
  if (!(V_sum > 0.0) || last_positive < 0) {
    for (int e = off; e < end; ++e) {
      distribution_weight[e] = 0.0;
    }
    if (source_status != nullptr) {
      source_status[s] =
          static_cast<std::uint8_t>(V_source > 0.0
                                        ? WeightStatus::EMPTY_OVERLAP
                                        : WeightStatus::INACTIVE_SOURCE);
    }
    return;
  }

  double sum_A = 0.0;
  for (int e = off; e < end; ++e) {
    double A = 0.0;
    if (e == last_positive) {
      A = detail::max2(0.0, 1.0 - sum_A);
    } else {
      const double V = detail::nonnegative_finite(pair_overlap_volume[e]);
      if (V > 0.0) {
        A = V / V_sum;
        sum_A += A;
      }
    }
    distribution_weight[e] = tenryu::hydro::rz::finite_double(A) ? A : 0.0;
  }
  if (source_status != nullptr) {
    source_status[s] = static_cast<std::uint8_t>(WeightStatus::OK);
  }
}

__global__ void zero_subcell_conserved_kernel(
    double* __restrict__ subcell_mass,
    double* __restrict__ subcell_momentum_r,
    double* __restrict__ subcell_momentum_z,
    double* __restrict__ subcell_internal_energy,
    double* __restrict__ subcell_kinetic_energy,
    const int n_subcells) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_subcells) {
    return;
  }
  subcell_mass[s] = 0.0;
  subcell_momentum_r[s] = 0.0;
  subcell_momentum_z[s] = 0.0;
  subcell_internal_energy[s] = 0.0;
  subcell_kinetic_energy[s] = 0.0;
}

__global__ void remap_subcell_conserved_kernel(
    double* __restrict__ target_mass,
    double* __restrict__ target_momentum_r,
    double* __restrict__ target_momentum_z,
    double* __restrict__ target_internal_energy,
    double* __restrict__ target_kinetic_energy,
    const double* __restrict__ source_mass,
    const double* __restrict__ source_momentum_r,
    const double* __restrict__ source_momentum_z,
    const double* __restrict__ source_internal_energy,
    const double* __restrict__ source_kinetic_energy,
    const int* __restrict__ source_pair_offsets,
    const int* __restrict__ pair_target_subcell,
    const double* __restrict__ distribution_weight,
    const int n_source_subcells,
    const int n_target_subcells) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_source_subcells) {
    return;
  }
  const double m = source_mass[s];
  const double pr = source_momentum_r[s];
  const double pz = source_momentum_z[s];
  const double I = source_internal_energy[s];
  const double K = source_kinetic_energy[s];
  if (!tenryu::hydro::rz::finite_double(m) ||
      !tenryu::hydro::rz::finite_double(pr) ||
      !tenryu::hydro::rz::finite_double(pz) ||
      !tenryu::hydro::rz::finite_double(I) ||
      !tenryu::hydro::rz::finite_double(K)) {
    return;
  }

  const int off = source_pair_offsets[s];
  const int end = source_pair_offsets[s + 1];
  for (int e = off; e < end; ++e) {
    const int st = pair_target_subcell[e];
    if (st < 0 || st >= n_target_subcells) {
      continue;
    }
    const double A = distribution_weight[e];
    if (!(A > 0.0) || !tenryu::hydro::rz::finite_double(A)) {
      continue;
    }
    atomicAdd(&target_mass[st], A * m);
    atomicAdd(&target_momentum_r[st], A * pr);
    atomicAdd(&target_momentum_z[st], A * pz);
    atomicAdd(&target_internal_energy[st], A * I);
    atomicAdd(&target_kinetic_energy[st], A * K);
  }
}

__global__ void zero_node_momentum_sums_kernel(
    double* __restrict__ node_mass,
    double* __restrict__ node_momentum_r,
    double* __restrict__ node_momentum_z,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  node_mass[n] = 0.0;
  node_momentum_r[n] = 0.0;
  node_momentum_z[n] = 0.0;
}

__global__ void accumulate_subcells_to_nodes_kernel(
    double* __restrict__ node_mass,
    double* __restrict__ node_momentum_r,
    double* __restrict__ node_momentum_z,
    const double* __restrict__ subcell_mass,
    const double* __restrict__ subcell_momentum_r,
    const double* __restrict__ subcell_momentum_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_subcells) {
  const int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_subcells) {
    return;
  }
  const int node = detail::corner_node_for_subcell(s,
                                                   x_r,
                                                   x_z,
                                                   cell_node_csr_offsets,
                                                   cell_node_csr_indices,
                                                   cell_nverts);
  if (node < 0) {
    return;
  }
  const double m = detail::nonnegative_finite(subcell_mass[s]);
  if (!(m > 0.0)) {
    return;
  }
  const double pr = tenryu::hydro::rz::finite_double(subcell_momentum_r[s])
                        ? subcell_momentum_r[s]
                        : 0.0;
  const double pz = tenryu::hydro::rz::finite_double(subcell_momentum_z[s])
                        ? subcell_momentum_z[s]
                        : 0.0;
  atomicAdd(&node_mass[node], m);
  atomicAdd(&node_momentum_r[node], pr);
  atomicAdd(&node_momentum_z[node], pz);
}

__global__ void finish_node_velocity_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ node_mass,
    const double* __restrict__ node_momentum_r,
    const double* __restrict__ node_momentum_z,
    const std::uint8_t* __restrict__ node_flags,
    const int n_nodes,
    const bool enforce_axis_radial_zero,
    const bool pin_center_velocity) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  const double m = node_mass[n];
  double vr = 0.0;
  double vz = 0.0;
  if (m > 0.0 && tenryu::hydro::rz::finite_double(m)) {
    vr = node_momentum_r[n] / m;
    vz = node_momentum_z[n] / m;
  }
  if (!tenryu::hydro::rz::finite_double(vr)) {
    vr = 0.0;
  }
  if (!tenryu::hydro::rz::finite_double(vz)) {
    vz = 0.0;
  }

  if (node_flags != nullptr) {
    const std::uint8_t flags = node_flags[n];
    if (pin_center_velocity &&
        (flags & tenryu::mesh::NODE_CENTER) != 0U) {
      vr = 0.0;
      vz = 0.0;
    } else if (enforce_axis_radial_zero &&
               (flags & (tenryu::mesh::NODE_AXIS |
                         tenryu::mesh::NODE_POLE_AXIS)) != 0U) {
      vr = 0.0;
    }
  }
  v_r_node[n] = vr;
  v_z_node[n] = vz;
}

__global__ void zero_compatible_energy_fix_sums_kernel(
    double* __restrict__ compatible_energy_fix_sums) {
  const int i = threadIdx.x;
  if (blockIdx.x == 0 && i < kFixCount) {
    compatible_energy_fix_sums[i] = 0.0;
  }
}

__global__ void scatter_subcells_to_cells_compatible_raw_kernel(
    double* __restrict__ cell_density,
    double* __restrict__ cell_mass,
    double* __restrict__ cell_internal_energy,
    double* __restrict__ cell_specific_internal_energy,
    double* __restrict__ cell_total_subcell_energy,
    const double* __restrict__ subcell_volume,
    const double* __restrict__ subcell_mass,
    const double* __restrict__ subcell_internal_energy,
    const double* __restrict__ subcell_kinetic_energy,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    double* __restrict__ compatible_energy_fix_sums) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const int active_nverts = detail::active_nverts(cell_nverts, c);
  const int off = cell_node_csr_offsets[c];
  double V_cell = 0.0;
  double m_cell = 0.0;
  double E_sub = 0.0;
  double K_nodal = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const int s = detail::subcell_index(c, k);
    double V = subcell_volume != nullptr ? subcell_volume[s] : 0.0;
    if (!(V > 0.0) || !tenryu::hydro::rz::finite_double(V)) {
      V = detail::subcell_volume_exact(s,
                                       x_r,
                                       x_z,
                                       cell_node_csr_offsets,
                                       cell_node_csr_indices,
                                       cell_nverts);
    }
    const double m = detail::nonnegative_finite(subcell_mass[s]);
    const double I = detail::nonnegative_finite(subcell_internal_energy[s]);
    const double K = detail::nonnegative_finite(subcell_kinetic_energy[s]);
    const int node = cell_node_csr_indices[off + k];
    const double vr =
        tenryu::hydro::rz::finite_double(v_r_node[node]) ? v_r_node[node]
                                                         : 0.0;
    const double vz =
        tenryu::hydro::rz::finite_double(v_z_node[node]) ? v_z_node[node]
                                                         : 0.0;
    V_cell += V;
    m_cell += m;
    E_sub += I + K;
    K_nodal += 0.5 * m * (vr * vr + vz * vz);
  }

  const double I_raw = E_sub - K_nodal;
  if (cell_density != nullptr) {
    cell_density[c] = V_cell > 0.0 ? m_cell / V_cell : 0.0;
  }
  if (cell_mass != nullptr) {
    cell_mass[c] = m_cell;
  }
  if (cell_total_subcell_energy != nullptr) {
    cell_total_subcell_energy[c] = E_sub;
  }
  cell_internal_energy[c] =
      tenryu::hydro::rz::finite_double(I_raw) ? I_raw : 0.0;
  if (cell_specific_internal_energy != nullptr) {
    cell_specific_internal_energy[c] =
        m_cell > 0.0 ? cell_internal_energy[c] / m_cell : 0.0;
  }

  if (compatible_energy_fix_sums != nullptr) {
    const double raw = tenryu::hydro::rz::finite_double(I_raw) ? I_raw : 0.0;
    if (raw > 0.0) {
      atomicAdd(&compatible_energy_fix_sums[kFixPositive], raw);
    } else if (raw < 0.0) {
      atomicAdd(&compatible_energy_fix_sums[kFixDeficit], -raw);
    }
    atomicAdd(&compatible_energy_fix_sums[kFixRawTotal], raw);
  }
}

__global__ void finalize_nonnegative_internal_energy_kernel(
    double* __restrict__ cell_internal_energy,
    double* __restrict__ cell_specific_internal_energy,
    const double* __restrict__ cell_mass,
    const int n_cells,
    double* __restrict__ compatible_energy_fix_sums) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double positive = compatible_energy_fix_sums[kFixPositive];
  const double deficit = compatible_energy_fix_sums[kFixDeficit];
  const double raw_total = compatible_energy_fix_sums[kFixRawTotal];
  double I = cell_internal_energy[c];
  if (!tenryu::hydro::rz::finite_double(I)) {
    I = 0.0;
  }

  if (deficit > 0.0) {
    if (raw_total >= 0.0 && positive > 0.0) {
      I = I > 0.0 ? I * detail::max2(0.0, raw_total / positive) : 0.0;
    } else {
      I = 0.0;
      if (c == 0) {
        compatible_energy_fix_sums[kFixUnresolvedDeficit] = -raw_total;
      }
    }
  }
  if (I < 0.0 || !tenryu::hydro::rz::finite_double(I)) {
    I = 0.0;
  }
  cell_internal_energy[c] = I;
  if (cell_specific_internal_energy != nullptr) {
    const double m = cell_mass != nullptr ? cell_mass[c] : 0.0;
    cell_specific_internal_energy[c] =
        m > 0.0 && tenryu::hydro::rz::finite_double(m) ? I / m : 0.0;
  }
}

inline int blocks_for(const int n, const int threads_per_block) {
  return n > 0 ? (n + threads_per_block - 1) / threads_per_block : 1;
}

inline cudaError_t launch_compute_subcell_volumes(
    double* subcell_volume,
    const double* x_r,
    const double* x_z,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const int n_cells,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  compute_subcell_volumes_kernel<<<blocks_for(n_cells, threads_per_block),
                                   threads_per_block,
                                   0,
                                   stream>>>(subcell_volume,
                                             x_r,
                                             x_z,
                                             cell_node_csr_offsets,
                                             cell_node_csr_indices,
                                             cell_nverts,
                                             n_cells);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_gather_cell_nodal_conserved_to_subcells(
    double* subcell_volume,
    double* subcell_mass,
    double* subcell_momentum_r,
    double* subcell_momentum_z,
    double* subcell_internal_energy,
    double* subcell_kinetic_energy,
    const double* cell_mass,
    const double* cell_density,
    const double* cell_specific_internal_energy,
    const double* x_r,
    const double* x_z,
    const double* v_r_node,
    const double* v_z_node,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const int n_cells,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  gather_cell_nodal_conserved_to_subcells_kernel<<<
      blocks_for(n_cells, threads_per_block),
      threads_per_block,
      0,
      stream>>>(subcell_volume,
                subcell_mass,
                subcell_momentum_r,
                subcell_momentum_z,
                subcell_internal_energy,
                subcell_kinetic_energy,
                cell_mass,
                cell_density,
                cell_specific_internal_energy,
                x_r,
                x_z,
                v_r_node,
                v_z_node,
                cell_node_csr_offsets,
                cell_node_csr_indices,
                cell_nverts,
                n_cells);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_build_intersection_distribution_weights(
    double* distribution_weight,
    double* intersection_volume,
    double* source_coverage_ratio,
    std::uint8_t* source_status,
    const int* source_pair_offsets,
    const int* pair_target_subcell,
    const double* x_r_source,
    const double* x_z_source,
    const double* x_r_target,
    const double* x_z_target,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const int n_source_subcells,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  build_intersection_distribution_weights_kernel<<<
      blocks_for(n_source_subcells, threads_per_block),
      threads_per_block,
      0,
      stream>>>(distribution_weight,
                intersection_volume,
                source_coverage_ratio,
                source_status,
                source_pair_offsets,
                pair_target_subcell,
                x_r_source,
                x_z_source,
                x_r_target,
                x_z_target,
                cell_node_csr_offsets,
                cell_node_csr_indices,
                cell_nverts,
                n_source_subcells);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_build_distribution_weights_from_overlap_volumes(
    double* distribution_weight,
    double* source_coverage_ratio,
    std::uint8_t* source_status,
    const int* source_pair_offsets,
    const double* pair_overlap_volume,
    const double* source_subcell_volume,
    const int n_source_subcells,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  build_distribution_weights_from_overlap_volumes_kernel<<<
      blocks_for(n_source_subcells, threads_per_block),
      threads_per_block,
      0,
      stream>>>(distribution_weight,
                source_coverage_ratio,
                source_status,
                source_pair_offsets,
                pair_overlap_volume,
                source_subcell_volume,
                n_source_subcells);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_zero_subcell_conserved(
    double* subcell_mass,
    double* subcell_momentum_r,
    double* subcell_momentum_z,
    double* subcell_internal_energy,
    double* subcell_kinetic_energy,
    const int n_subcells,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  zero_subcell_conserved_kernel<<<blocks_for(n_subcells, threads_per_block),
                                  threads_per_block,
                                  0,
                                  stream>>>(subcell_mass,
                                            subcell_momentum_r,
                                            subcell_momentum_z,
                                            subcell_internal_energy,
                                            subcell_kinetic_energy,
                                            n_subcells);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_remap_subcell_conserved(
    double* target_mass,
    double* target_momentum_r,
    double* target_momentum_z,
    double* target_internal_energy,
    double* target_kinetic_energy,
    const double* source_mass,
    const double* source_momentum_r,
    const double* source_momentum_z,
    const double* source_internal_energy,
    const double* source_kinetic_energy,
    const int* source_pair_offsets,
    const int* pair_target_subcell,
    const double* distribution_weight,
    const int n_source_subcells,
    const int n_target_subcells,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  remap_subcell_conserved_kernel<<<
      blocks_for(n_source_subcells, threads_per_block),
      threads_per_block,
      0,
      stream>>>(target_mass,
                target_momentum_r,
                target_momentum_z,
                target_internal_energy,
                target_kinetic_energy,
                source_mass,
                source_momentum_r,
                source_momentum_z,
                source_internal_energy,
                source_kinetic_energy,
                source_pair_offsets,
                pair_target_subcell,
                distribution_weight,
                n_source_subcells,
                n_target_subcells);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_zero_node_momentum_sums(
    double* node_mass,
    double* node_momentum_r,
    double* node_momentum_z,
    const int n_nodes,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  zero_node_momentum_sums_kernel<<<blocks_for(n_nodes, threads_per_block),
                                   threads_per_block,
                                   0,
                                   stream>>>(node_mass,
                                             node_momentum_r,
                                             node_momentum_z,
                                             n_nodes);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_accumulate_subcells_to_nodes(
    double* node_mass,
    double* node_momentum_r,
    double* node_momentum_z,
    const double* subcell_mass,
    const double* subcell_momentum_r,
    const double* subcell_momentum_z,
    const double* x_r,
    const double* x_z,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const int n_subcells,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  accumulate_subcells_to_nodes_kernel<<<blocks_for(n_subcells,
                                                   threads_per_block),
                                        threads_per_block,
                                        0,
                                        stream>>>(node_mass,
                                                  node_momentum_r,
                                                  node_momentum_z,
                                                  subcell_mass,
                                                  subcell_momentum_r,
                                                  subcell_momentum_z,
                                                  x_r,
                                                  x_z,
                                                  cell_node_csr_offsets,
                                                  cell_node_csr_indices,
                                                  cell_nverts,
                                                  n_subcells);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_finish_node_velocity(
    double* v_r_node,
    double* v_z_node,
    const double* node_mass,
    const double* node_momentum_r,
    const double* node_momentum_z,
    const std::uint8_t* node_flags,
    const int n_nodes,
    const bool enforce_axis_radial_zero,
    const bool pin_center_velocity,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  finish_node_velocity_kernel<<<blocks_for(n_nodes, threads_per_block),
                                threads_per_block,
                                0,
                                stream>>>(v_r_node,
                                          v_z_node,
                                          node_mass,
                                          node_momentum_r,
                                          node_momentum_z,
                                          node_flags,
                                          n_nodes,
                                          enforce_axis_radial_zero,
                                          pin_center_velocity);
  return cudaPeekAtLastError();
}

inline cudaError_t launch_scatter_subcells_to_cells_compatible(
    double* cell_density,
    double* cell_mass,
    double* cell_internal_energy,
    double* cell_specific_internal_energy,
    double* cell_total_subcell_energy,
    const double* subcell_volume,
    const double* subcell_mass,
    const double* subcell_internal_energy,
    const double* subcell_kinetic_energy,
    const double* v_r_node,
    const double* v_z_node,
    const double* x_r,
    const double* x_z,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const int n_cells,
    double* compatible_energy_fix_sums,
    const bool apply_nonnegative_conservative_fix,
    cudaStream_t stream = 0,
    const int threads_per_block = 256) {
  if (apply_nonnegative_conservative_fix &&
      compatible_energy_fix_sums != nullptr) {
    zero_compatible_energy_fix_sums_kernel<<<1, kFixCount, 0, stream>>>(
        compatible_energy_fix_sums);
  }
  scatter_subcells_to_cells_compatible_raw_kernel<<<
      blocks_for(n_cells, threads_per_block),
      threads_per_block,
      0,
      stream>>>(cell_density,
                cell_mass,
                cell_internal_energy,
                cell_specific_internal_energy,
                cell_total_subcell_energy,
                subcell_volume,
                subcell_mass,
                subcell_internal_energy,
                subcell_kinetic_energy,
                v_r_node,
                v_z_node,
                x_r,
                x_z,
                cell_node_csr_offsets,
                cell_node_csr_indices,
                cell_nverts,
                n_cells,
                apply_nonnegative_conservative_fix
                    ? compatible_energy_fix_sums
                    : nullptr);
  if (apply_nonnegative_conservative_fix &&
      compatible_energy_fix_sums != nullptr) {
    finalize_nonnegative_internal_energy_kernel<<<
        blocks_for(n_cells, threads_per_block),
        threads_per_block,
        0,
        stream>>>(cell_internal_energy,
                  cell_specific_internal_energy,
                  cell_mass,
                  n_cells,
                  compatible_energy_fix_sums);
  }
  return cudaPeekAtLastError();
}

#endif  // __CUDACC__

}  // namespace tenryu::hydro::ale::staggered_subcell_remap
