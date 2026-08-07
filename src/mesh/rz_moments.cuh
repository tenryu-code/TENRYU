#pragma once

#include <cuda_runtime.h>

namespace tenryu::mesh::moments {

/**
 * This library is the single geometry authority defined by design F4. The
 * existing tenryu::hydro::ale::detail::rz_signed_quad_volume remains the
 * production call site until the integration step. Unit tests assert agreement
 * to relative 1e-13 between 2*pi*poly_rz_moments_fan(quad).mr and that call
 * site.
 *
 * All arithmetic is contraction-immune BY CONSTRUCTION, in two layers: (1)
 * inside every function, each contractible multiply-add is an explicit fma with
 * fixed parenthesization; (2) no public function returns a value produced by a
 * bare non-exact multiply — final products are written fma(a,b,0.0)
 * (identically round(a*b); exact-zero products canonicalize to +0.0) so that
 * inlining call sites (e.g. the fan/star accumulators' +=) leave nothing for
 * -ffmad to contract. Exact power-of-two scales (0.5*x) are
 * contraction-neutral and remain bare. Results are bitwise identical between
 * host and device and independent of -ffp-contract / -march. Do not 'simplify'
 * any of this: a bare a*b that escapes into a caller's add reintroduces the
 * divergence (measured: poly mrz 1-ulp, 2026-07-27).
 */

constexpr double kPi = 3.141592653589793238462643383279502884;

/**
 * Returns the signed triangle area
 * A = 0.5*fma(r_b-r_a, z_c-z_a, -((r_c-r_a)*(z_b-z_a))).
 */
__host__ __device__ inline double tri_signed_area(
    const double ra, const double za, const double rb, const double zb,
    const double rc, const double zc) {
  return 0.5 * fma(rb - ra, zc - za, -((rc - ra) * (zb - za)));
}

struct TriRZMoments {
  double area;
  double mr;
  double mz;
  double mrr;
  double mrz;
};

/**
 * Returns signed triangle moments A, integral(r)dA, integral(z)dA,
 * integral(r^2)dA, and integral(r*z)dA in their closed P1 forms, with
 * rr = fma(sr,sr,fma(ra,ra,fma(rb,rb,rc*rc))) and
 * rz = fma(sr,sz,fma(ra,za,fma(rb,zb,rc*zc))),
 * mrr = fma(A/12,rr,0.0), and mrz = fma(A/12,rz,0.0).
 */
__host__ __device__ inline TriRZMoments tri_rz_moments(
    const double ra, const double za, const double rb, const double zb,
    const double rc, const double zc) {
  const double area = tri_signed_area(ra, za, rb, zb, rc, zc);
  const double sr = (ra + rb) + rc;
  const double sz = (za + zb) + zc;

  TriRZMoments moments{};
  moments.area = area;
  moments.mr = (area * sr) / 3.0;
  moments.mz = (area * sz) / 3.0;
  const double q = area / 12.0;
  const double rr = fma(sr, sr, fma(ra, ra, fma(rb, rb, rc * rc)));
  const double rz = fma(sr, sz, fma(ra, za, fma(rb, zb, rc * zc)));
  moments.mrr = fma(q, rr, 0.0);
  moments.mrz = fma(q, rz, 0.0);
  return moments;
}

/**
 * Writes integral(lambda_i*r)dA =
 * fma(A/12,fma(2,r_i,r_j)+r_k,0.0) with (i,j,k) cyclic =
 * (a,b,c),(b,c,a),(c,a,b) for the three triangle vertices.
 */
__host__ __device__ inline void tri_lambda_r_moments(
    const double ra, const double za, const double rb, const double zb,
    const double rc, const double zc, double lam_r[3]) {
  const double area = tri_signed_area(ra, za, rb, zb, rc, zc);
  const double q = area / 12.0;
  lam_r[0] = fma(q, fma(2.0, ra, rb) + rc, 0.0);
  lam_r[1] = fma(q, fma(2.0, rb, rc) + ra, 0.0);
  lam_r[2] = fma(q, fma(2.0, rc, ra) + rb, 0.0);
}

/**
 * Writes the three barycentric moments integral(lambda_i)dA = A/3.
 */
__host__ __device__ inline void tri_lambda_moments(
    const double ra, const double za, const double rb, const double zb,
    const double rc, const double zc, double lam[3]) {
  const double value = tri_signed_area(ra, za, rb, zb, rc, zc) / 3.0;
  lam[0] = value;
  lam[1] = value;
  lam[2] = value;
}

/**
 * Returns the physical revolved triangle volume
 * V = fma(2*pi,integral(r)dA,0.0).
 */
__host__ __device__ inline double tri_rz_volume(
    const double ra, const double za, const double rb, const double zb,
    const double rc, const double zc) {
  return fma(2.0 * kPi, tri_rz_moments(ra, za, rb, zb, rc, zc).mr, 0.0);
}

/**
 * Returns integral((c0 + gr*(r-r0) + gz*(z-z0))*r)dA =
 * fma(gz, fma(-z0,mr,mrz),
 *     fma(gr, fma(-r0,mr,mrr), c0*mr)).
 */
__host__ __device__ inline double tri_linear_r_moment(
    const TriRZMoments& m, const double c0, const double gr, const double gz,
    const double r0, const double z0) {
  const double t_r = fma(-r0, m.mr, m.mrr);
  const double t_z = fma(-z0, m.mr, m.mrz);
  return fma(gz, t_z, fma(gr, t_r, c0 * m.mr));
}

/**
 * Returns integral(c0 + gr*(r-r0) + gz*(z-z0))dA =
 * fma(gz, fma(-z0,A,mz),
 *     fma(gr, fma(-r0,A,mr), c0*A)).
 */
__host__ __device__ inline double tri_linear_area_moment(
    const TriRZMoments& m, const double c0, const double gr, const double gz,
    const double r0, const double z0) {
  const double t_r = fma(-r0, m.area, m.mr);
  const double t_z = fma(-z0, m.area, m.mz);
  return fma(gz, t_z, fma(gr, t_r, c0 * m.area));
}

struct PolyRZMoments {
  double area;
  double mr;
  double mz;
  double mrr;
  double mrz;
};

/**
 * Returns signed polygon moments by the fan sum
 * sum_{k=1}^{n-2} moments(v0, vk, v(k+1)).
 */
__host__ __device__ inline PolyRZMoments poly_rz_moments_fan(
    const double* r, const double* z, const int n) {
  PolyRZMoments moments{};
  for (int k = 1; k < n - 1; ++k) {
    const TriRZMoments tri =
        tri_rz_moments(r[0], z[0], r[k], z[k], r[k + 1], z[k + 1]);
    moments.area += tri.area;
    moments.mr += tri.mr;
    moments.mz += tri.mz;
    moments.mrr += tri.mrr;
    moments.mrz += tri.mrz;
  }
  return moments;
}

/**
 * Returns signed polygon moments by the star sum
 * sum_{k=0}^{n-1} moments(star, vk, v((k+1) mod n)).
 */
__host__ __device__ inline PolyRZMoments poly_rz_moments_star(
    const double* r, const double* z, const int n, const double rs,
    const double zs) {
  PolyRZMoments moments{};
  for (int k = 0; k < n; ++k) {
    const int next = (k + 1 == n) ? 0 : k + 1;
    const TriRZMoments tri =
        tri_rz_moments(rs, zs, r[k], z[k], r[next], z[next]);
    moments.area += tri.area;
    moments.mr += tri.mr;
    moments.mz += tri.mz;
    moments.mrr += tri.mrr;
    moments.mrz += tri.mrz;
  }
  return moments;
}

/**
 * Writes the arithmetic vertex mean
 * (rs,zs) = (sum_k r_k/n, sum_k z_k/n).
 */
__host__ __device__ inline void poly_vertex_mean(
    const double* r, const double* z, const int n, double& rs, double& zs) {
  rs = 0.0;
  zs = 0.0;
  for (int k = 0; k < n; ++k) {
    rs += r[k];
    zs += z[k];
  }
  rs /= static_cast<double>(n);
  zs /= static_cast<double>(n);
}

/**
 * Returns the physical revolved polygon volume
 * V = fma(2*pi,poly_rz_moments_fan(r,z,n).mr,0.0).
 */
__host__ __device__ inline double poly_rz_volume(
    const double* r, const double* z, const int n) {
  return fma(2.0 * kPi, poly_rz_moments_fan(r, z, n).mr, 0.0);
}

/**
 * Returns the straight-segment length
 * L = sqrt(fma(r1-r0, r1-r0, (z1-z0)*(z1-z0))).
 */
__host__ __device__ inline double segment_length(
    const double r0, const double z0, const double r1, const double z1) {
  const double dr = r1 - r0;
  const double dz = z1 - z0;
  return sqrt(fma(dr, dr, dz * dz));
}

/**
 * Returns the exact per-radian face area
 * integral(r)ds = fma(0.5*(r0+r1),L,0.0).
 */
__host__ __device__ inline double segment_per_radian_face_area(
    const double r0, const double z0, const double r1, const double z1) {
  return fma(0.5 * (r0 + r1), segment_length(r0, z0, r1, z1), 0.0);
}

/**
 * Returns the physical revolved face area
 * integral(2*pi*r)ds = fma(2*pi,fma(0.5*(r0+r1),L,0.0),0.0).
 */
__host__ __device__ inline double segment_rz_face_area(
    const double r0, const double z0, const double r1, const double z1) {
  return fma(2.0 * kPi, segment_per_radian_face_area(r0, z0, r1, z1), 0.0);
}

enum class Orient2D : int {
  kNegative = -1,
  kDegenerate = 0,
  kPositive = 1,
};

/**
 * Returns the sign of A = tri_signed_area(a,b,c), with exactly zero mapped to
 * kDegenerate. An adaptive exact-arithmetic upgrade is a P0B overlay work item;
 * the P0A contract is determinism, not exactness under near-ties.
 */
__host__ __device__ inline Orient2D orient2d_deterministic(
    const double ra, const double za, const double rb, const double zb,
    const double rc, const double zc) {
  const double area = tri_signed_area(ra, za, rb, zb, rc, zc);
  if (area > 0.0) {
    return Orient2D::kPositive;
  }
  if (area < 0.0) {
    return Orient2D::kNegative;
  }
  return Orient2D::kDegenerate;
}

struct OverlayPartitionCheck {
  bool ok;
  double area_residual;
  double mr_residual;
};

/**
 * Returns residuals |sum-ref|/max(|ref|,abs_floor) for signed area and mr;
 * ok is true exactly when both residuals are at most rtol.
 */
__host__ __device__ inline OverlayPartitionCheck overlay_partition_check(
    const double sum_piece_area, const double sum_piece_mr,
    const double ref_area, const double ref_mr, const double rtol,
    const double abs_floor) {
  OverlayPartitionCheck check{};
  check.area_residual =
      fabs(sum_piece_area - ref_area) / fmax(fabs(ref_area), abs_floor);
  check.mr_residual =
      fabs(sum_piece_mr - ref_mr) / fmax(fabs(ref_mr), abs_floor);
  check.ok =
      check.area_residual <= rtol && check.mr_residual <= rtol;
  return check;
}

/**
 * Returns integral over the polygon (fan from vertex 0) of
 * (c0 + gr*(r-r0) + gz*(z-z0)) * r dA, as the sum of per-triangle
 * tri_linear_r_moment values (signed, orientation follows the vertex order).
 */
__host__ __device__ inline double poly_linear_r_moment(
    const double* r, const double* z, const int n,
    const double c0, const double gr, const double gz,
    const double r0, const double z0) {
  double acc = 0.0;
  for (int k = 1; k < n - 1; ++k) {
    const TriRZMoments tri =
        tri_rz_moments(r[0], z[0], r[k], z[k], r[k + 1], z[k + 1]);
    acc += tri_linear_r_moment(tri, c0, gr, gz, r0, z0);
  }
  return acc;
}

/**
 * Returns integral over the polygon (fan from vertex 0) of
 * (c0 + gr*(r-r0) + gz*(z-z0)) dA, as the sum of per-triangle
 * tri_linear_area_moment values (signed, orientation follows the vertex
 * order).
 */
__host__ __device__ inline double poly_linear_area_moment(
    const double* r, const double* z, const int n,
    const double c0, const double gr, const double gz,
    const double r0, const double z0) {
  double acc = 0.0;
  for (int k = 1; k < n - 1; ++k) {
    const TriRZMoments tri =
        tri_rz_moments(r[0], z[0], r[k], z[k], r[k + 1], z[k + 1]);
    acc += tri_linear_area_moment(tri, c0, gr, gz, r0, z0);
  }
  return acc;
}

inline constexpr int kMaxStarP1Verts = 16;

/**
 * Condensed star-P1 vertex weights of a simple star-visible polygon (CCW, n in
 * [3, kMaxStarP1Verts]): w_j = integral over the polygon of Phi_j * r dA
 * (per-radian measure), where
 * Phi_j|T_i = (1/n)*lambda_star + [j==i]*lambda_i
 *              + [j==i+1]*lambda_{i+1}
 * on T_i = (x*, v_i, v_{i+1}) and x* is the arithmetic vertex mean.
 *
 * Every T_i must be positively oriented: x* must see every edge of the CCW
 * polygon. Degenerate or reflex-blocked stars produce signed cancellation; the
 * nonnegativity gate catches such misuse.
 *
 * Returns the polygon's star-fan mr for the caller's partition check.
 */
__host__ __device__ inline double star_p1_vertex_r_moments(
    const double* r, const double* z, const int n, double* w) {
  double rs = 0.0;
  double zs = 0.0;
  poly_vertex_mean(r, z, n, rs, zs);

  for (int j = 0; j < n; ++j) {
    w[j] = 0.0;
  }

  const double inv_n = 1.0 / static_cast<double>(n);
  double fan_mr = 0.0;
  for (int i = 0; i < n; ++i) {
    const int next = (i + 1 == n) ? 0 : i + 1;
    double lam_r[3]{};
    tri_lambda_r_moments(
        rs, zs, r[i], z[i], r[next], z[next], lam_r);
    for (int j = 0; j < n; ++j) {
      w[j] = fma(inv_n, lam_r[0], w[j]);
    }
    w[i] += lam_r[1];
    w[next] += lam_r[2];
    fan_mr +=
        tri_rz_moments(rs, zs, r[i], z[i], r[next], z[next]).mr;
  }
  return fan_mr;
}

/**
 * Planar condensed star-P1 vertex weights w_j = integral(Phi_j)dA.
 * Per T_i, all three barycentric moments are A_i/3, so lam[0]/n is
 * contributed to every w_j, lam[1] to w_i, and lam[2] to w_{i+1}.
 *
 * The same star-visibility precondition as star_p1_vertex_r_moments applies.
 * Degenerate or reflex-blocked stars produce signed cancellation; the
 * nonnegativity gate catches such misuse.
 *
 * Returns the polygon's star-fan area.
 */
__host__ __device__ inline double star_p1_vertex_area_moments(
    const double* r, const double* z, const int n, double* w) {
  double rs = 0.0;
  double zs = 0.0;
  poly_vertex_mean(r, z, n, rs, zs);

  for (int j = 0; j < n; ++j) {
    w[j] = 0.0;
  }

  const double inv_n = 1.0 / static_cast<double>(n);
  double fan_area = 0.0;
  for (int i = 0; i < n; ++i) {
    const int next = (i + 1 == n) ? 0 : i + 1;
    double lam[3]{};
    tri_lambda_moments(
        rs, zs, r[i], z[i], r[next], z[next], lam);
    for (int j = 0; j < n; ++j) {
      w[j] = fma(inv_n, lam[0], w[j]);
    }
    w[i] += lam[1];
    w[next] += lam[2];
    fan_area += tri_signed_area(
        rs, zs, r[i], z[i], r[next], z[next]);
  }
  return fan_area;
}

}  // namespace tenryu::mesh::moments
