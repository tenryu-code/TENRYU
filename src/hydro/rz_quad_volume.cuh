#pragma once

#include <cuda_runtime.h>

namespace tenryu::hydro::ale {
namespace detail {

constexpr double kPi = 3.141592653589793238462643383279502884;

// Signed volume of a 4-point polygon revolved around the z-axis.
// Vertices are in (r, z) coordinates; counterclockwise order gives positive volume.
// V = (pi/3) * sum_k (r_k + r_{k+1}) * (r_k z_{k+1} - r_{k+1} z_k).
__host__ __device__ inline double rz_signed_quad_volume(const double r0,
                                                        const double z0,
                                                        const double r1,
                                                        const double z1,
                                                        const double r2,
                                                        const double z2,
                                                        const double r3,
                                                        const double z3) {
  const double r_c = 0.25 * (r0 + r1 + r2 + r3);
  const double z_c = 0.25 * (z0 + z1 + z2 + z3);

  const double R0 = r0 - r_c;
  const double Z0 = z0 - z_c;
  const double R1 = r1 - r_c;
  const double Z1 = z1 - z_c;
  const double R2 = r2 - r_c;
  const double Z2 = z2 - z_c;
  const double R3 = r3 - r_c;
  const double Z3 = z3 - z_c;

  const double cross01 = R0 * Z1 - R1 * Z0;
  const double cross12 = R1 * Z2 - R2 * Z1;
  const double cross23 = R2 * Z3 - R3 * Z2;
  const double cross30 = R3 * Z0 - R0 * Z3;

  const double w01 = R0 + R1 + 3.0 * r_c;
  const double w12 = R1 + R2 + 3.0 * r_c;
  const double w23 = R2 + R3 + 3.0 * r_c;
  const double w30 = R3 + R0 + 3.0 * r_c;

  return (kPi / 3.0) *
         (cross01 * w01 + cross12 * w12 + cross23 * w23 + cross30 * w30);
}

}  // namespace detail
}  // namespace tenryu::hydro::ale
