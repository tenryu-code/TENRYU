#pragma once

#include <cmath>

#ifndef TENRYU_HOST_DEVICE
#define TENRYU_HOST_DEVICE __host__ __device__
#endif

namespace tenryu::hydro::kershaw {

enum BoundaryType : int {
  BC_INTERIOR = 0,
  BC_REFLECT = 1,
  BC_VACUUM = 2,
};

TENRYU_HOST_DEVICE inline bool outside_cell(const int i,
                                            const int j,
                                            const int nr,
                                            const int nz) {
  return (i < 0) || (i >= nr) || (j < 0) || (j >= nz);
}

TENRYU_HOST_DEVICE inline void fold_boundary_coefficient(double& a_center,
                                                         double& a_neighbor,
                                                         const int ni,
                                                         const int nj,
                                                         const int nr,
                                                         const int nz,
                                                         const int bc_type,
                                                         const double d_ext,
                                                         const double dx) {
  if (!outside_cell(ni, nj, nr, nz)) {
    return;
  }

  if (bc_type == BC_VACUUM) {
    const double den = d_ext + 0.5 * dx;
    if (fabs(den) > 1.0e-30) {
      const double robin = -(d_ext - 0.5 * dx) / den;
      a_center += a_neighbor * robin;
    } else {
      a_center += a_neighbor;
    }
  } else {
    // Reflect/Neumann: ghost equals interior value.
    a_center += a_neighbor;
  }

  a_neighbor = 0.0;
}

TENRYU_HOST_DEVICE inline int normal_bc_for_neighbor(const int i,
                                                      const int j,
                                                      const int nr,
                                                      const int nz,
                                                      const int bc_r_lo,
                                                      const int bc_r_hi,
                                                      const int bc_z_lo,
                                                      const int bc_z_hi) {
  if (i < 0) {
    return bc_r_lo;
  }
  if (i >= nr) {
    return bc_r_hi;
  }
  if (j < 0) {
    return bc_z_lo;
  }
  if (j >= nz) {
    return bc_z_hi;
  }
  return BC_INTERIOR;
}

}  // namespace tenryu::hydro::kershaw

