#pragma once

#include <cuda_runtime.h>

#include "mesh/rz_moments.cuh"

namespace tenryu::mesh::clip {

inline constexpr int kMaxClipVerts = 16;

enum class ClipStatus : int {
  kOk = 0,
  kEmpty = 1,
  kClipperNotConvex = 2,
  kClipperNotCCW = 3,
  kOverflow = 4,
  kDegenerateClipper = 5,
};

struct ClipResult {
  ClipStatus status;
  int n;
  double r[kMaxClipVerts];
  double z[kMaxClipVerts];
};

/**
 * Returns the signed half-plane evaluation for q against directed edge p0->p1.
 * For a CCW clipper, nonnegative values are in the closed interior half-plane.
 */
__host__ __device__ inline double edge_side(
    const double p0r, const double p0z, const double p1r, const double p1z,
    const double qr, const double qz) {
  return fma(p1r - p0r, qz - p0z,
             -((p1z - p0z) * (qr - p0r)));
}

/**
 * Intersects segment a->b with a line from endpoint side evaluations of
 * opposite half-plane classifications. The subtraction, division, and fma
 * coordinates are correctly rounded per operation and deterministic between
 * host and device, but are not exact-real coordinates. Partition gates are
 * therefore relative-tolerance based.
 */
__host__ __device__ inline void segment_line_intersection(
    const double ar, const double az, const double br, const double bz,
    const double da, const double db, double& r, double& z) {
  const double t = da / (da - db);
  r = fma(t, br - ar, ar);
  z = fma(t, bz - az, az);
}

/**
 * Validates that the clipper is strictly convex and CCW. nc is required to be
 * at least three by the caller.
 */
__host__ __device__ inline ClipStatus clip_precheck(
    const double* cr, const double* cz, const int nc) {
  bool has_negative = false;
  bool has_degenerate = false;
  for (int k = 0; k < nc; ++k) {
    const int k1 = (k + 1 == nc) ? 0 : k + 1;
    const int k2 = (k1 + 1 == nc) ? 0 : k1 + 1;
    const moments::Orient2D orientation =
        moments::orient2d_deterministic(
            cr[k], cz[k], cr[k1], cz[k1], cr[k2], cz[k2]);
    if (orientation == moments::Orient2D::kNegative) {
      has_negative = true;
    } else if (orientation == moments::Orient2D::kDegenerate) {
      has_degenerate = true;
    }
  }

  if (has_degenerate) {
    return ClipStatus::kDegenerateClipper;
  }
  if (!has_negative) {
    return ClipStatus::kOk;
  }

  const moments::PolyRZMoments clipper_moments =
      moments::poly_rz_moments_fan(cr, cz, nc);
  return clipper_moments.area < 0.0 ? ClipStatus::kClipperNotCCW
                                    : ClipStatus::kClipperNotConvex;
}

/**
 * Clips a simple CCW subject polygon against a convex CCW clipper. Subject
 * orientation is a caller-owned precondition and is not checked. The closed
 * half-plane convention keeps points whose edge-side evaluation is exactly
 * zero.
 *
 * Subject and clipper counts must each be in [3,8]. An invalid input count is
 * reported with kOverflow and n=0, using the same fail-loud status as a fixed
 * buffer capacity violation.
 */
__host__ __device__ inline ClipResult clip_convex(
    const double* sr, const double* sz, const int ns,
    const double* cr, const double* cz, const int nc) {
  ClipResult result{};
  result.status = ClipStatus::kOverflow;
  if (ns < 3 || ns > 8 || nc < 3 || nc > 8) {
    return result;
  }

  const ClipStatus precheck = clip_precheck(cr, cz, nc);
  if (precheck != ClipStatus::kOk) {
    result.status = precheck;
    return result;
  }

  double buffer_a_r[kMaxClipVerts]{};
  double buffer_a_z[kMaxClipVerts]{};
  double buffer_b_r[kMaxClipVerts]{};
  double buffer_b_z[kMaxClipVerts]{};
  for (int k = 0; k < ns; ++k) {
    buffer_a_r[k] = sr[k];
    buffer_a_z[k] = sz[k];
  }

  double* input_r = buffer_a_r;
  double* input_z = buffer_a_z;
  double* output_r = buffer_b_r;
  double* output_z = buffer_b_z;
  int input_n = ns;

  for (int edge = 0; edge < nc; ++edge) {
    const int edge_next = (edge + 1 == nc) ? 0 : edge + 1;
    int output_n = 0;

    for (int k = 0; k < input_n; ++k) {
      const int next = (k + 1 == input_n) ? 0 : k + 1;
      const double ar = input_r[k];
      const double az = input_z[k];
      const double br = input_r[next];
      const double bz = input_z[next];
      const double da =
          edge_side(cr[edge], cz[edge], cr[edge_next], cz[edge_next],
                    ar, az);
      const double db =
          edge_side(cr[edge], cz[edge], cr[edge_next], cz[edge_next],
                    br, bz);
      const bool a_inside = da >= 0.0;
      const bool b_inside = db >= 0.0;

      if (a_inside != b_inside) {
        if (output_n == kMaxClipVerts) {
          result.status = ClipStatus::kOverflow;
          return result;
        }
        segment_line_intersection(
            ar, az, br, bz, da, db, output_r[output_n], output_z[output_n]);
        ++output_n;
      }
      if (b_inside) {
        if (output_n == kMaxClipVerts) {
          result.status = ClipStatus::kOverflow;
          return result;
        }
        output_r[output_n] = br;
        output_z[output_n] = bz;
        ++output_n;
      }
    }

    if (output_n < 3) {
      result.status = ClipStatus::kEmpty;
      return result;
    }

    input_n = output_n;
    double* swap_r = input_r;
    input_r = output_r;
    output_r = swap_r;
    double* swap_z = input_z;
    input_z = output_z;
    output_z = swap_z;
  }

  result.status = ClipStatus::kOk;
  result.n = input_n;
  for (int k = 0; k < input_n; ++k) {
    result.r[k] = input_r[k];
    result.z[k] = input_z[k];
  }
  return result;
}

struct OverlayPieceMoments {
  ClipStatus status;
  int n;
  moments::PolyRZMoments m;
};

/**
 * Clips an overlay piece and evaluates its F4 polygon moments. Empty and
 * non-kOk results carry zero-initialized moments.
 */
__host__ __device__ inline OverlayPieceMoments overlay_piece(
    const double* sr, const double* sz, const int ns,
    const double* cr, const double* cz, const int nc) {
  const ClipResult clipped = clip_convex(sr, sz, ns, cr, cz, nc);
  OverlayPieceMoments piece{};
  piece.status = clipped.status;
  piece.n = clipped.n;
  if (clipped.status == ClipStatus::kOk) {
    piece.m = moments::poly_rz_moments_fan(
        clipped.r, clipped.z, clipped.n);
  }
  return piece;
}

}  // namespace tenryu::mesh::clip
