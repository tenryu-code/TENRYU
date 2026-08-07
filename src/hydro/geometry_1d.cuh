#pragma once

#include "core/config.hpp"

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace tenryu::hydro {

// B1 1D geometry family (docs/design/b1_1d_cyl_mode_spec.md).
// Spherical branches reproduce the historical per-site arithmetic
// verbatim (bitwise regime); Cylindrical is per unit length in z.
inline constexpr double kGeom1dFourPi = 12.566370614359172953850573533118;
inline constexpr double kGeom1dTwoPi =
    6.2831853071795864769252867665590057683943387987502;
inline constexpr double kGeom1dFourPiOverThree =
    4.188790204786390984616857844372670512262892532500141094646;
inline constexpr double kGeom1dPi =
    3.1415926535897932384626433832795028841971693993751;

template <core::Geometry1D G>
__host__ __device__ inline double geom1d_area(const double r) {
  if constexpr (G == core::Geometry1D::Spherical) {
    return kGeom1dFourPi * r * r;
  } else {
    return kGeom1dTwoPi * r;
  }
}

template <core::Geometry1D G>
__host__ __device__ inline double geom1d_area_avg(const double ra,
                                                  const double rb) {
  if constexpr (G == core::Geometry1D::Spherical) {
    return 0.5 * kGeom1dFourPi * (ra * ra + rb * rb);
  } else {
    return 0.5 * kGeom1dTwoPi * (ra + rb);
  }
}

template <core::Geometry1D G>
__host__ __device__ inline double geom1d_cell_volume(const double r0,
                                                     const double r1) {
  if constexpr (G == core::Geometry1D::Spherical) {
    const double dr = r1 - r0;
    return kGeom1dFourPiOverThree * dr * (r1 * r1 + r1 * r0 + r0 * r0);
  } else {
    const double dr = r1 - r0;
    return kGeom1dPi * dr * (r1 + r0);
  }
}

}  // namespace tenryu::hydro
