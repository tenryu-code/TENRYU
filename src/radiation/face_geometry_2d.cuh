#pragma once

#ifndef TENRYU_RADIATION_FACE_GEOMETRY_2D_CUH_
#define TENRYU_RADIATION_FACE_GEOMETRY_2D_CUH_

#include <cmath>
#include <cstdint>

// Shared 2D_RZ face geometry helpers for radiation transport.
// See NUMERICS §6.3.2 and CUDA_KERNELS §6.4.2.

namespace tenryu::radiation {

#if defined(__CUDACC__)
#define TENRYU_FACE_GEOM_HD __host__ __device__
#else
#define TENRYU_FACE_GEOM_HD
#endif

struct FaceGeom2D {
  double r1, z1;   // endpoint 1
  double r2, z2;   // endpoint 2
  double nr, nz;   // outward unit normal (dz, -dr) / L
  double tr, tz;   // unit tangent along edge
  double length;   // edge length L
  double r_mid;    // midpoint R coordinate = (r1+r2)/2
};

TENRYU_FACE_GEOM_HD inline FaceGeom2D compute_face_geom(
    int face_id,
    double r00,
    double z00,  // vertex 0 (R_lo, Z_lo)
    double r10,
    double z10,  // vertex 1 (R_hi, Z_lo)
    double r11,
    double z11,  // vertex 2 (R_hi, Z_hi)
    double r01,
    double z01   // vertex 3 (R_lo, Z_hi)
) {
  FaceGeom2D geom{};

  // Physical face mapping (CUDA_KERNELS §6.4.2).
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
    default:  // Fallback to face 0 convention if caller passes invalid face.
      geom.r1 = r01;
      geom.z1 = z01;
      geom.r2 = r00;
      geom.z2 = z00;
      break;
  }

  const double dr = geom.r2 - geom.r1;
  const double dz = geom.z2 - geom.z1;
  const double L = std::sqrt(dr * dr + dz * dz);

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
}

TENRYU_FACE_GEOM_HD inline void reflect_direction_2d(double& dir_r,
                                                     double& dir_z,
                                                     const double nr,
                                                     const double nz) {
  const double dot = dir_r * nr + dir_z * nz;
  dir_r -= 2.0 * dot * nr;
  dir_z -= 2.0 * dot * nz;
}

TENRYU_FACE_GEOM_HD inline double face_mu_2d(const double dir_r,
                                             const double dir_z,
                                             const double nr,
                                             const double nz) {
  return dir_r * nr + dir_z * nz;
}

TENRYU_FACE_GEOM_HD inline void push_off_face_2d(double& r,
                                                 double& z,
                                                 const double nr,
                                                 const double nz,
                                                 const double eps) {
  r += nr * eps;
  z += nz * eps;
}

}  // namespace tenryu::radiation

#endif  // TENRYU_RADIATION_FACE_GEOMETRY_2D_CUH_
