#pragma once

#include <cuda_runtime.h>

#include "mesh/poly_clip.cuh"

namespace tenryu::mesh::overlay {

inline constexpr int kMaxDonors = 16;
inline constexpr int kSubcellsPerQuad = 4;
inline constexpr int kMaxSubcellDonors = 40;  // 3x3 cells x 4 subcells = 36, padded

enum class OverlayCellStatus : int {
  kOk = 0,
  kClipFailure = 1,
  kPartitionFailure = 2,
};

struct OverlayCellResult {
  OverlayCellStatus status;
  int n_donors_hit;
  double mass_per_radian;
  double mass_planar;
  double area_sum;
  double mr_sum;
  tenryu::mesh::moments::OverlayPartitionCheck partition;
};

/**
 * Gathers piecewise-constant donor data into one convex CCW receiver. Donors
 * are packed with donor d beginning at d*donor_stride. Both RZ per-radian and
 * planar mass measures are accumulated; the caller chooses the density sense
 * represented by rho.
 */
__host__ __device__ inline OverlayCellResult overlay_gather_cell(
    const double* recv_r, const double* recv_z, const int recv_n,
    const double* donor_r, const double* donor_z, const int* donor_n,
    const int donor_stride, const double* rho, const int n_donors,
    const double rtol, const double abs_floor) {
  OverlayCellResult result{};
  result.status = OverlayCellStatus::kOk;

  for (int d = 0; d < n_donors; ++d) {
    const clip::OverlayPieceMoments piece =
        clip::overlay_piece(
            donor_r + d * donor_stride, donor_z + d * donor_stride,
            donor_n[d], recv_r, recv_z, recv_n);
    if (piece.status == clip::ClipStatus::kEmpty) {
      continue;
    }
    if (piece.status != clip::ClipStatus::kOk) {
      result.status = OverlayCellStatus::kClipFailure;
      result.mass_per_radian = 0.0;
      result.mass_planar = 0.0;
      result.area_sum = 0.0;
      result.mr_sum = 0.0;
      return result;
    }

    result.area_sum += piece.m.area;
    result.mr_sum += piece.m.mr;
    result.mass_per_radian =
        fma(rho[d], piece.m.mr, result.mass_per_radian);
    result.mass_planar =
        fma(rho[d], piece.m.area, result.mass_planar);
    ++result.n_donors_hit;
  }

  const moments::PolyRZMoments recv_moments =
      moments::poly_rz_moments_fan(recv_r, recv_z, recv_n);
  result.partition = moments::overlay_partition_check(
      result.area_sum, result.mr_sum, recv_moments.area, recv_moments.mr,
      rtol, abs_floor);
  result.status = result.partition.ok ? OverlayCellStatus::kOk
                                      : OverlayCellStatus::kPartitionFailure;
  return result;
}

/**
 * Packs the 3x3 old-cell neighborhood of structured cell (ci,cj). The output
 * vertex stride is four and each donor uses the structured CCW quad order
 * (i,j),(i+1,j),(i+1,j+1),(i,j+1).
 */
__host__ __device__ inline int pack_structured_neighborhood(
    const int nr, const int nz, const int ci, const int cj,
    const double* old_node_r, const double* old_node_z,
    const double* rho_all, double* donor_r, double* donor_z, int* donor_n,
    double* rho_out) {
  const int i_begin = ci > 0 ? ci - 1 : 0;
  const int i_end = ci + 1 < nr ? ci + 1 : nr - 1;
  const int j_begin = cj > 0 ? cj - 1 : 0;
  const int j_end = cj + 1 < nz ? cj + 1 : nz - 1;
  int n_donors = 0;

  for (int i = i_begin; i <= i_end; ++i) {
    for (int j = j_begin; j <= j_end; ++j) {
      const int node00 = i * (nz + 1) + j;
      const int node10 = (i + 1) * (nz + 1) + j;
      const int node11 = (i + 1) * (nz + 1) + j + 1;
      const int node01 = i * (nz + 1) + j + 1;
      const int offset = n_donors * 4;

      donor_r[offset] = old_node_r[node00];
      donor_z[offset] = old_node_z[node00];
      donor_r[offset + 1] = old_node_r[node10];
      donor_z[offset + 1] = old_node_z[node10];
      donor_r[offset + 2] = old_node_r[node11];
      donor_z[offset + 2] = old_node_z[node11];
      donor_r[offset + 3] = old_node_r[node01];
      donor_z[offset + 3] = old_node_z[node01];
      donor_n[n_donors] = 4;
      rho_out[n_donors] = rho_all[i * nz + j];
      ++n_donors;
    }
  }

  return n_donors;
}

// Corner subcell k (k in 0..3) of structured cell (ci,cj): the CCW quad
// (v_k, m_{k,k+1}, c, m_{k-1,k}) where v_k are the cell's CCW vertices
// ((i,j),(i+1,j),(i+1,j+1),(i,j+1)), m_{a,b} = 0.5*(v_a+v_b) edge midpoints and
// c = 0.25*(v_0+v_1+v_2+v_3) the vertex mean (all exact power-of-two averages).
// Median subcells of a convex quad are convex; the receiver-side clipper precheck
// fail-louds on violations.
__host__ __device__ inline void structured_subcell_polygon(
    const int nr, const int nz, const int ci, const int cj, const int k,
    const double* node_r, const double* node_z,
    double out_r[4], double out_z[4]) {
  (void)nr;
  const int node[4] = {
      ci * (nz + 1) + cj,
      (ci + 1) * (nz + 1) + cj,
      (ci + 1) * (nz + 1) + cj + 1,
      ci * (nz + 1) + cj + 1};
  const int next = k + 1 < 4 ? k + 1 : 0;
  const int previous = k > 0 ? k - 1 : 3;

  out_r[0] = node_r[node[k]];
  out_z[0] = node_z[node[k]];
  out_r[1] = 0.5 * (node_r[node[k]] + node_r[node[next]]);
  out_z[1] = 0.5 * (node_z[node[k]] + node_z[node[next]]);
  out_r[2] =
      0.25 * (((node_r[node[0]] + node_r[node[1]]) +
               node_r[node[2]]) +
              node_r[node[3]]);
  out_z[2] =
      0.25 * (((node_z[node[0]] + node_z[node[1]]) +
               node_z[node[2]]) +
              node_z[node[3]]);
  out_r[3] = 0.5 * (node_r[node[previous]] + node_r[node[k]]);
  out_z[3] = 0.5 * (node_z[node[previous]] + node_z[node[k]]);
}

// Packs the 36 subcell donors (3x3 old-cell neighborhood x 4 corner subcells) of
// structured cell (ci,cj): polygons via structured_subcell_polygon on the OLD mesh,
// donor densities rho_out[d] = subcell_mass[global subcell index] / (that subcell's own
// per-radian mr from poly_rz_moments_fan). Global subcell index = cell*4 + k with
// cell = i*nz + j. Donor order: cells in the clamped i-major/j-inner order (same as
// pack_structured_neighborhood), subcells k = 0..3 within each cell. Returns the donor
// count (<= 36). Caller arrays have capacity kMaxSubcellDonors (stride 4).
__host__ __device__ inline int pack_structured_subcell_neighborhood(
    const int nr, const int nz, const int ci, const int cj,
    const double* old_node_r, const double* old_node_z,
    const double* subcell_mass,
    double* donor_r, double* donor_z, int* donor_n, double* rho_out) {
  const int i_begin = ci > 0 ? ci - 1 : 0;
  const int i_end = ci + 1 < nr ? ci + 1 : nr - 1;
  const int j_begin = cj > 0 ? cj - 1 : 0;
  const int j_end = cj + 1 < nz ? cj + 1 : nz - 1;
  int n_donors = 0;

  for (int i = i_begin; i <= i_end; ++i) {
    for (int j = j_begin; j <= j_end; ++j) {
      const int cell = i * nz + j;
      for (int k = 0; k < kSubcellsPerQuad; ++k) {
        const int offset = n_donors * 4;
        structured_subcell_polygon(
            nr, nz, i, j, k, old_node_r, old_node_z,
            donor_r + offset, donor_z + offset);
        donor_n[n_donors] = 4;
        const moments::PolyRZMoments donor_moments =
            moments::poly_rz_moments_fan(
                donor_r + offset, donor_z + offset, 4);
        rho_out[n_donors] =
            subcell_mass[cell * kSubcellsPerQuad + k] / donor_moments.mr;
        ++n_donors;
      }
    }
  }

  return n_donors;
}

// Gathers the receiver subcell (ci,cj,k) of the NEW mesh against the 36 old subcell
// donors: builds the receiver subcell polygon, packs donors, and calls
// overlay_gather_cell (the existing engine accepts any n_donors; capacity is
// caller-side). mass_per_radian of the result IS the transferred m(cn).
__host__ inline OverlayCellResult structured_subcell_gather(
    const int nr, const int nz, const int ci, const int cj, const int k,
    const double* old_node_r, const double* old_node_z,
    const double* new_node_r, const double* new_node_z,
    const double* subcell_mass, const double rtol,
    const double abs_floor) {
  double recv_r[4]{};
  double recv_z[4]{};
  structured_subcell_polygon(
      nr, nz, ci, cj, k, new_node_r, new_node_z, recv_r, recv_z);

  double donor_r[kMaxSubcellDonors * 4]{};
  double donor_z[kMaxSubcellDonors * 4]{};
  int donor_n[kMaxSubcellDonors]{};
  double donor_rho[kMaxSubcellDonors]{};
  const int n_donors = pack_structured_subcell_neighborhood(
      nr, nz, ci, cj, old_node_r, old_node_z, subcell_mass,
      donor_r, donor_z, donor_n, donor_rho);
  return overlay_gather_cell(
      recv_r, recv_z, 4, donor_r, donor_z, donor_n, 4, donor_rho,
      n_donors, rtol, abs_floor);
}

struct StructuredOverlayShadowResult {
  bool all_ok;
  int n_cells;
  int first_bad_cell;
  double total_mass_per_radian_old;
  double total_mass_per_radian_new;
  double total_mass_planar_old;
  double total_mass_planar_new;
  double max_partition_area_residual;
  double max_partition_mr_residual;
};

/**
 * Runs a host-side structured-quad overlay gather without wiring the result
 * into stepping. Global old/new conservation holds when both meshes tile the
 * same domain and every receiver is covered by its 3x3 old-cell neighborhood;
 * coverage violations are reported as partition failures.
 */
__host__ inline StructuredOverlayShadowResult structured_overlay_shadow(
    const int nr, const int nz, const double* old_node_r,
    const double* old_node_z, const double* new_node_r,
    const double* new_node_z, const double* rho, const double rtol,
    const double abs_floor) {
  StructuredOverlayShadowResult result{};
  result.all_ok = true;
  result.n_cells = nr * nz;
  result.first_bad_cell = -1;

  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int cell = i * nz + j;
      const int node00 = i * (nz + 1) + j;
      const int node10 = (i + 1) * (nz + 1) + j;
      const int node11 = (i + 1) * (nz + 1) + j + 1;
      const int node01 = i * (nz + 1) + j + 1;
      const double old_r[4] = {
          old_node_r[node00], old_node_r[node10],
          old_node_r[node11], old_node_r[node01]};
      const double old_z[4] = {
          old_node_z[node00], old_node_z[node10],
          old_node_z[node11], old_node_z[node01]};
      const moments::PolyRZMoments old_moments =
          moments::poly_rz_moments_fan(old_r, old_z, 4);
      result.total_mass_per_radian_old =
          fma(rho[cell], old_moments.mr,
              result.total_mass_per_radian_old);
      result.total_mass_planar_old =
          fma(rho[cell], old_moments.area,
              result.total_mass_planar_old);
    }
  }

  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int cell = i * nz + j;
      const int node00 = i * (nz + 1) + j;
      const int node10 = (i + 1) * (nz + 1) + j;
      const int node11 = (i + 1) * (nz + 1) + j + 1;
      const int node01 = i * (nz + 1) + j + 1;
      const double recv_r[4] = {
          new_node_r[node00], new_node_r[node10],
          new_node_r[node11], new_node_r[node01]};
      const double recv_z[4] = {
          new_node_z[node00], new_node_z[node10],
          new_node_z[node11], new_node_z[node01]};

      if (clip::clip_precheck(recv_r, recv_z, 4) !=
          clip::ClipStatus::kOk) {
        result.all_ok = false;
        if (result.first_bad_cell < 0) {
          result.first_bad_cell = cell;
        }
        continue;
      }

      double donor_r[kMaxDonors * 4]{};
      double donor_z[kMaxDonors * 4]{};
      int donor_n[kMaxDonors]{};
      double donor_rho[kMaxDonors]{};
      const int n_donors = pack_structured_neighborhood(
          nr, nz, i, j, old_node_r, old_node_z, rho, donor_r, donor_z,
          donor_n, donor_rho);
      const OverlayCellResult cell_result = overlay_gather_cell(
          recv_r, recv_z, 4, donor_r, donor_z, donor_n, 4, donor_rho,
          n_donors, rtol, abs_floor);

      result.total_mass_per_radian_new += cell_result.mass_per_radian;
      result.total_mass_planar_new += cell_result.mass_planar;
      result.max_partition_area_residual =
          fmax(result.max_partition_area_residual,
               cell_result.partition.area_residual);
      result.max_partition_mr_residual =
          fmax(result.max_partition_mr_residual,
               cell_result.partition.mr_residual);
      if (cell_result.status != OverlayCellStatus::kOk) {
        result.all_ok = false;
        if (result.first_bad_cell < 0) {
          result.first_bad_cell = cell;
        }
      }
    }
  }

  return result;
}

/**
 * Linear-reconstruction gather: donor d carries the P1 intensive
 * reconstruction q_d(r,z) =
 * c0[d] + gr[d]*(r - r0[d]) + gz[d]*(z - z0[d]); each nonempty overlay piece
 * contributes integral(q_d * r)dA (per-radian) and integral(q_d)dA (planar)
 * evaluated EXACTLY via the F4 linear moment forms on the clipped piece
 * polygon. Geometry partition gate identical to overlay_gather_cell.
 */
__host__ __device__ inline OverlayCellResult overlay_gather_cell_linear(
    const double* recv_r, const double* recv_z, const int recv_n,
    const double* donor_r, const double* donor_z, const int* donor_n,
    const int donor_stride,
    const double* c0, const double* gr, const double* gz,
    const double* r0, const double* z0,
    const int n_donors, const double rtol, const double abs_floor) {
  OverlayCellResult result{};
  result.status = OverlayCellStatus::kOk;

  for (int d = 0; d < n_donors; ++d) {
    const clip::ClipResult piece =
        clip::clip_convex(
            donor_r + d * donor_stride, donor_z + d * donor_stride,
            donor_n[d], recv_r, recv_z, recv_n);
    if (piece.status == clip::ClipStatus::kEmpty) {
      continue;
    }
    if (piece.status != clip::ClipStatus::kOk || piece.n < 3) {
      result.status = OverlayCellStatus::kClipFailure;
      result.mass_per_radian = 0.0;
      result.mass_planar = 0.0;
      result.area_sum = 0.0;
      result.mr_sum = 0.0;
      return result;
    }

    const moments::PolyRZMoments piece_moments =
        moments::poly_rz_moments_fan(piece.r, piece.z, piece.n);
    result.area_sum += piece_moments.area;
    result.mr_sum += piece_moments.mr;
    result.mass_per_radian += moments::poly_linear_r_moment(
        piece.r, piece.z, piece.n, c0[d], gr[d], gz[d], r0[d], z0[d]);
    result.mass_planar += moments::poly_linear_area_moment(
        piece.r, piece.z, piece.n, c0[d], gr[d], gz[d], r0[d], z0[d]);
    ++result.n_donors_hit;
  }

  const moments::PolyRZMoments recv_moments =
      moments::poly_rz_moments_fan(recv_r, recv_z, recv_n);
  result.partition = moments::overlay_partition_check(
      result.area_sum, result.mr_sum, recv_moments.area, recv_moments.mr,
      rtol, abs_floor);
  result.status = result.partition.ok ? OverlayCellStatus::kOk
                                      : OverlayCellStatus::kPartitionFailure;
  return result;
}

struct DonorAnchors {
  double r0_v, z0_v;  // Per-radian-neutral anchor: (mrr/mr, mrz/mr).
  double r0_a, z0_a;  // Planar-neutral anchor: (mr/area,mz/area).
};

/**
 * Returns measure-consistent anchors from the donor polygon's own fan moments.
 * With c0 equal to the donor mean in the corresponding measure, the donor
 * integral of the reconstruction is exactly c0*mr or c0*area. Valid mesh-cell
 * donors with positive mr and area are a caller-owned precondition.
 */
__host__ __device__ inline DonorAnchors donor_anchors(
    const double* r, const double* z, const int n) {
  const moments::PolyRZMoments m =
      moments::poly_rz_moments_fan(r, z, n);
  DonorAnchors anchors{};
  anchors.r0_v = m.mrr / m.mr;
  anchors.z0_v = m.mrz / m.mr;
  anchors.r0_a = m.mr / m.area;
  anchors.z0_a = m.mz / m.area;
  return anchors;
}

__host__ __device__ inline double minmod(
    const double a, const double b) {
  if (a > 0.0 && b > 0.0) {
    return a < b ? a : b;
  }
  if (a < 0.0 && b < 0.0) {
    return a > b ? a : b;
  }
  return 0.0;
}

/**
 * Computes central-neighbor minmod gradients of per-cell values q at structured
 * cell (ci,cj). Cell coordinates are planar centroids (mr/area,mz/area) of
 * old-mesh quad fan moments. A missing boundary-side difference is zero, so
 * that component is locally first order at the boundary.
 */
__host__ __device__ inline void structured_minmod_gradient(
    const int nr, const int nz, const int ci, const int cj,
    const double* old_node_r, const double* old_node_z,
    const double* q, double& gr, double& gz) {
  const int cell_i[5] = {ci, ci - 1, ci + 1, ci, ci};
  const int cell_j[5] = {cj, cj, cj, cj - 1, cj + 1};
  double centroid_r[5]{};
  double centroid_z[5]{};

  for (int side = 0; side < 5; ++side) {
    const int i = cell_i[side];
    const int j = cell_j[side];
    if (i < 0 || i >= nr || j < 0 || j >= nz) {
      continue;
    }

    const int node00 = i * (nz + 1) + j;
    const int node10 = (i + 1) * (nz + 1) + j;
    const int node11 = (i + 1) * (nz + 1) + j + 1;
    const int node01 = i * (nz + 1) + j + 1;
    const double cell_r[4] = {
        old_node_r[node00], old_node_r[node10],
        old_node_r[node11], old_node_r[node01]};
    const double cell_z[4] = {
        old_node_z[node00], old_node_z[node10],
        old_node_z[node11], old_node_z[node01]};
    const moments::PolyRZMoments m =
        moments::poly_rz_moments_fan(cell_r, cell_z, 4);
    centroid_r[side] = m.mr / m.area;
    centroid_z[side] = m.mz / m.area;
  }

  const int cell = ci * nz + cj;
  const double forward_r =
      ci + 1 < nr
          ? (q[(ci + 1) * nz + cj] - q[cell]) /
                (centroid_r[2] - centroid_r[0])
          : 0.0;
  const double backward_r =
      ci > 0
          ? (q[cell] - q[(ci - 1) * nz + cj]) /
                (centroid_r[0] - centroid_r[1])
          : 0.0;
  const double forward_z =
      cj + 1 < nz
          ? (q[ci * nz + cj + 1] - q[cell]) /
                (centroid_z[4] - centroid_z[0])
          : 0.0;
  const double backward_z =
      cj > 0
          ? (q[cell] - q[ci * nz + cj - 1]) /
                (centroid_z[0] - centroid_z[3])
          : 0.0;
  gr = minmod(forward_r, backward_r);
  gz = minmod(forward_z, backward_z);
}

/**
 * Returns the Barth-Jespersen scale in [0,1] that bounds the donor-vertex
 * reconstruction by [q_min,q_max]. The per-radian anchor and mean are used;
 * the shared scale consequently gives only conservative approximate bounds for
 * the planar measure.
 */
__host__ __device__ inline double barth_jespersen_alpha(
    const double* donor_r, const double* donor_z, const int n,
    const double c0, const double gr, const double gz,
    const double r0, const double z0,
    const double q_min, const double q_max) {
  double alpha = 1.0;
  for (int v = 0; v < n; ++v) {
    const double dq =
        fma(gz, donor_z[v] - z0,
            fma(gr, donor_r[v] - r0, 0.0));
    double alpha_v = 1.0;
    if (dq > 0.0) {
      alpha_v = fmin(1.0, (q_max - c0) / dq);
    } else if (dq < 0.0) {
      alpha_v = fmin(1.0, (q_min - c0) / dq);
    }
    alpha = fmin(alpha, alpha_v);
  }
  return alpha;
}

/**
 * Assembles a limited P0B linear gather for one structured receiver. Donor
 * values q are per-radian cell means; gradients use planar-centroid minmod and
 * are scaled by donor-local Barth-Jespersen bounds over 3x3 cell means.
 * Per-radian donor anchors make that transfer anchor-exact. Because the P0B-3
 * gather applies the same reconstruction to both measures, the planar sum and
 * planar bounds are approximate rather than anchor-exact.
 */
__host__ inline OverlayCellResult structured_limited_gather(
    const int nr, const int nz, const int ci, const int cj,
    const double* old_node_r, const double* old_node_z,
    const double* new_node_r, const double* new_node_z,
    const double* q, const double rtol, const double abs_floor) {
  const int node00 = ci * (nz + 1) + cj;
  const int node10 = (ci + 1) * (nz + 1) + cj;
  const int node11 = (ci + 1) * (nz + 1) + cj + 1;
  const int node01 = ci * (nz + 1) + cj + 1;
  const double recv_r[4] = {
      new_node_r[node00], new_node_r[node10],
      new_node_r[node11], new_node_r[node01]};
  const double recv_z[4] = {
      new_node_z[node00], new_node_z[node10],
      new_node_z[node11], new_node_z[node01]};

  double donor_r[kMaxDonors * 4]{};
  double donor_z[kMaxDonors * 4]{};
  int donor_n[kMaxDonors]{};
  double c0_v[kMaxDonors]{};
  const int n_donors = pack_structured_neighborhood(
      nr, nz, ci, cj, old_node_r, old_node_z, q,
      donor_r, donor_z, donor_n, c0_v);
  double gr[kMaxDonors]{};
  double gz[kMaxDonors]{};
  double r0[kMaxDonors]{};
  double z0[kMaxDonors]{};

  const int i_begin = ci > 0 ? ci - 1 : 0;
  const int i_end = ci + 1 < nr ? ci + 1 : nr - 1;
  const int j_begin = cj > 0 ? cj - 1 : 0;
  const int j_end = cj + 1 < nz ? cj + 1 : nz - 1;
  int d = 0;
  for (int i = i_begin; i <= i_end; ++i) {
    for (int j = j_begin; j <= j_end; ++j) {
      const DonorAnchors anchors =
          donor_anchors(donor_r + 4 * d, donor_z + 4 * d, donor_n[d]);
      r0[d] = anchors.r0_v;
      z0[d] = anchors.z0_v;

      double donor_gr = 0.0;
      double donor_gz = 0.0;
      structured_minmod_gradient(
          nr, nz, i, j, old_node_r, old_node_z, q, donor_gr, donor_gz);

      double q_min = q[i * nz + j];
      double q_max = q_min;
      const int neighbor_i_begin = i > 0 ? i - 1 : 0;
      const int neighbor_i_end = i + 1 < nr ? i + 1 : nr - 1;
      const int neighbor_j_begin = j > 0 ? j - 1 : 0;
      const int neighbor_j_end = j + 1 < nz ? j + 1 : nz - 1;
      for (int ni = neighbor_i_begin; ni <= neighbor_i_end; ++ni) {
        for (int nj = neighbor_j_begin; nj <= neighbor_j_end; ++nj) {
          q_min = fmin(q_min, q[ni * nz + nj]);
          q_max = fmax(q_max, q[ni * nz + nj]);
        }
      }

      const double alpha = barth_jespersen_alpha(
          donor_r + 4 * d, donor_z + 4 * d, donor_n[d], c0_v[d],
          donor_gr, donor_gz, r0[d], z0[d], q_min, q_max);
      gr[d] = fma(alpha, donor_gr, 0.0);
      gz[d] = fma(alpha, donor_gz, 0.0);
      ++d;
    }
  }

  return overlay_gather_cell_linear(
      recv_r, recv_z, 4, donor_r, donor_z, donor_n, 4,
      c0_v, gr, gz, r0, z0, n_donors, rtol, abs_floor);
}

}  // namespace tenryu::mesh::overlay
