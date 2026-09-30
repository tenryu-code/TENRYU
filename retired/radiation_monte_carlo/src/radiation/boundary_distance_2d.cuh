#pragma once

#include <cmath>

#include "face_geometry_2d.cuh"

// Boundary distance now uses general edge-ray intersection against actual
// quadrilateral face endpoints for 2D_RZ cells.

namespace tenryu::radiation {

#if defined(__CUDACC__)
#define TENRYU_HOST_DEVICE __host__ __device__
#else
#define TENRYU_HOST_DEVICE
#endif

struct BoundaryHit2D {
  double s = INFINITY;
  int face = -1;      // 0=R_left, 1=R_right, 2=Z_bottom, 3=Z_top
  int neighbor = -1;  // valid only when is_boundary=false
  bool is_boundary = true;
};

TENRYU_HOST_DEVICE inline int cell_index_2d_rz(const int i,
                                                const int j,
                                                const int nz) {
  return i * nz + j;
}

TENRYU_HOST_DEVICE inline int node_index_2d_rz(const int i,
                                                const int j,
                                                const int nz) {
  return i * (nz + 1) + j;
}

TENRYU_HOST_DEVICE inline FaceGeom2D face_geom_2d_compat(
    const int face_id,
    const double r00,
    const double z00,
    const double r10,
    const double z10,
    const double r11,
    const double z11,
    const double r01,
    const double z01) {
#if defined(__CUDA_ARCH__)
  FaceGeom2D geom{};

  switch (face_id) {
    case 0:  // R_left: vertex3 -> vertex0
      geom.r1 = r01;
      geom.z1 = z01;
      geom.r2 = r00;
      geom.z2 = z00;
      break;
    case 1:  // R_right: vertex1 -> vertex2
      geom.r1 = r10;
      geom.z1 = z10;
      geom.r2 = r11;
      geom.z2 = z11;
      break;
    case 2:  // Z_bottom: vertex0 -> vertex1
      geom.r1 = r00;
      geom.z1 = z00;
      geom.r2 = r10;
      geom.z2 = z10;
      break;
    case 3:  // Z_top: vertex2 -> vertex3
      geom.r1 = r11;
      geom.z1 = z11;
      geom.r2 = r01;
      geom.z2 = z01;
      break;
    default:
      geom.r1 = r01;
      geom.z1 = z01;
      geom.r2 = r00;
      geom.z2 = z00;
      break;
  }

  const double dr = geom.r2 - geom.r1;
  const double dz = geom.z2 - geom.z1;
  const double L = sqrt(dr * dr + dz * dz);
  geom.length = L;
  geom.r_mid = 0.5 * (geom.r1 + geom.r2);

  if (L > 0.0) {
    const double inv_L = 1.0 / L;
    geom.nr = dz * inv_L;
    geom.nz = -dr * inv_L;
    geom.tr = dr * inv_L;
    geom.tz = dz * inv_L;
  } else {
    geom.nr = 0.0;
    geom.nz = 0.0;
    geom.tr = 0.0;
    geom.tz = 0.0;
  }

  return geom;
#else
  return tenryu::radiation::compute_face_geom(
      face_id, r00, z00, r10, z10, r11, z11, r01, z01);
#endif
}

TENRYU_HOST_DEVICE inline void push_off_face_2d_compat(double& r,
                                                        double& z,
                                                        const double nr,
                                                        const double nz,
                                                        const double eps) {
#if defined(__CUDA_ARCH__)
  r += nr * eps;
  z += nz * eps;
#else
  tenryu::radiation::push_off_face_2d(r, z, nr, nz, eps);
#endif
}

TENRYU_HOST_DEVICE inline void consider_candidate(BoundaryHit2D* hit,
                                                   const double s,
                                                   const int face,
                                                   const int neighbor,
                                                   const bool is_boundary,
                                                   const double eps) {
  if (!(s > eps) || !isfinite(s)) {
    return;
  }
  if (s + eps < hit->s || (fabs(s - hit->s) <= eps && face < hit->face)) {
    hit->s = s;
    hit->face = face;
    hit->neighbor = neighbor;
    hit->is_boundary = is_boundary;
  }
}

TENRYU_HOST_DEVICE inline int solve_quadratic_real_roots(const double a,
                                                          const double b,
                                                          const double c,
                                                          const double eps_det,
                                                          double* root0,
                                                          double* root1) {
  if (fabs(a) <= eps_det) {
    if (fabs(b) <= eps_det) {
      return 0;
    }
    const double x = -c / b;
    *root0 = x;
    *root1 = x;
    return 1;
  }

  double disc = b * b - 4.0 * a * c;
  if (disc < -eps_det) {
    return 0;
  }
  disc = fmax(disc, 0.0);
  const double sqrt_disc = sqrt(disc);

  if (sqrt_disc <= eps_det) {
    const double x = -0.5 * b / a;
    *root0 = x;
    *root1 = x;
    return 1;
  }

  const double q = -0.5 * (b + copysign(sqrt_disc, b));
  if (fabs(q) <= eps_det) {
    const double x = -0.5 * b / a;
    *root0 = x;
    *root1 = x;
    return 1;
  }

  const double x0 = q / a;
  const double x1 = c / q;
  *root0 = fmin(x0, x1);
  *root1 = fmax(x0, x1);
  return 2;
}

TENRYU_HOST_DEVICE inline double intersect_ray_edge_2d_rz(
    const double R,
    const double Z,
    const double Omega_r,
    const double Omega_z,
    const double omega_r_sq,
    const double p1r,
    const double p1z,
    const double p2r,
    const double p2z,
    const double eps_geom,
    const double eps_det) {
  const double dr = p2r - p1r;
  const double dz = p2z - p1z;
  const double dz_sq = dz * dz;

  const double z_delta0 = Z - p1z;
  const double k0 = p1r * dz + dr * z_delta0;
  const double k1 = dr * Omega_z;

  const double a = dz_sq * omega_r_sq - k1 * k1;
  const double b = 2.0 * (dz_sq * R * Omega_r - k0 * k1);
  const double c = dz_sq * R * R - k0 * k0;

  // Use a relative discriminant tolerance: when a,b,c are O(1), the
  // discriminant b^2-4ac can suffer catastrophic cancellation producing
  // a tiny negative residual (e.g. -1e-16).  A fixed eps_det=1e-30 is
  // far too tight; scale by max(|b^2|, |4ac|, 1) to keep relative
  // precision ~ 4*eps_machine.
  const double disc_scale = fmax(fmax(b * b, fabs(4.0 * a * c)), 1.0);
  const double disc_tol = fmax(eps_det, 4.0e-15 * disc_scale);

  double roots[2] = {INFINITY, INFINITY};
  const int root_count = solve_quadratic_real_roots(
      a, b, c, disc_tol, &roots[0], &roots[1]);

  double s_best = INFINITY;
  for (int k = 0; k < root_count; ++k) {
    const double s_try = roots[k];
    if (!(s_try > eps_geom) || !isfinite(s_try)) {
      continue;
    }

    double t = INFINITY;
    if (fabs(dz) >= fabs(dr) && fabs(dz) > eps_det) {
      t = (Z + Omega_z * s_try - p1z) / dz;
    } else if (fabs(dr) > eps_det) {
      const double r_sq_hit =
          R * R + 2.0 * R * Omega_r * s_try + omega_r_sq * s_try * s_try;
      if (r_sq_hit < -eps_det) {
        continue;
      }
      const double r_hit = sqrt(fmax(r_sq_hit, 0.0));
      t = (r_hit - p1r) / dr;
    } else {
      continue;
    }

    if (!isfinite(t) || t < -eps_geom || t > 1.0 + eps_geom) {
      continue;
    }

    s_best = fmin(s_best, s_try);
  }

  return s_best;
}

TENRYU_HOST_DEVICE inline BoundaryHit2D boundary_distance_2d_rz(
    const double R,
    const double Z,
    const double Omega_r,
    const double Omega_z,
    const double Omega_phi,
    const int cell,
    const int nr,
    const int nz,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z) {
  constexpr double eps_geom = 1.0e-12;
  constexpr double eps_det = 1.0e-30;
  BoundaryHit2D hit{};

  if (cell < 0 || nr <= 0 || nz <= 0) {
    return hit;
  }

  const int i = cell / nz;
  const int j = cell - i * nz;
  if (i < 0 || i >= nr || j < 0 || j >= nz) {
    return hit;
  }

  const int n00 = node_index_2d_rz(i, j, nz);
  const int n10 = node_index_2d_rz(i + 1, j, nz);
  const int n11 = node_index_2d_rz(i + 1, j + 1, nz);
  const int n01 = node_index_2d_rz(i, j + 1, nz);

  const double r00 = node_r[n00];
  const double r10 = node_r[n10];
  const double r11 = node_r[n11];
  const double r01 = node_r[n01];
  const double z00 = node_z[n00];
  const double z10 = node_z[n10];
  const double z11 = node_z[n11];
  const double z01 = node_z[n01];
  const double omega_r_sq = Omega_r * Omega_r + Omega_phi * Omega_phi;

  for (int face = 0; face < 4; ++face) {
    const FaceGeom2D geom =
        face_geom_2d_compat(face, r00, z00, r10, z10, r11, z11, r01, z01);

    const double s = intersect_ray_edge_2d_rz(R,
                                              Z,
                                              Omega_r,
                                              Omega_z,
                                              omega_r_sq,
                                              geom.r1,
                                              geom.z1,
                                              geom.r2,
                                              geom.z2,
                                              eps_geom,
                                              eps_det);

    if (!isfinite(s)) {
      continue;
    }

    bool is_boundary = true;
    int neighbor = -1;
    if (face == 0) {
      is_boundary = (i == 0);
      neighbor = is_boundary ? -1 : cell_index_2d_rz(i - 1, j, nz);
    } else if (face == 1) {
      is_boundary = (i == nr - 1);
      neighbor = is_boundary ? -1 : cell_index_2d_rz(i + 1, j, nz);
    } else if (face == 2) {
      is_boundary = (j == 0);
      neighbor = is_boundary ? -1 : cell_index_2d_rz(i, j - 1, nz);
    } else {
      is_boundary = (j == nz - 1);
      neighbor = is_boundary ? -1 : cell_index_2d_rz(i, j + 1, nz);
    }

    consider_candidate(&hit, s, face, neighbor, is_boundary, eps_geom);
  }

  return hit;
}

#undef TENRYU_HOST_DEVICE

}  // namespace tenryu::radiation
