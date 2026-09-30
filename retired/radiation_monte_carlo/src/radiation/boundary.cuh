#pragma once

#include <cmath>

namespace tenryu::radiation {

enum BoundaryCode : int {
  kBoundaryVacuum = 0,
  kBoundaryReflect = 1,
  kBoundaryMarshak = 2,
};

__device__ inline void apply_reflect_1d(double* dir_r,
                                         double* dir_z,
                                         double* dir_phi) {
  *dir_r = -*dir_r;
  // Keep tangential components unchanged for specular reflection in spherical geometry.
  (void)dir_z;
  (void)dir_phi;
}

__device__ inline bool is_escape_boundary(const int bc) {
  return bc == kBoundaryVacuum || bc == kBoundaryMarshak;
}

}  // namespace tenryu::radiation
