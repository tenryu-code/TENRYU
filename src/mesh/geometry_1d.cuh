#pragma once

// W-G: 1D coordinate-geometry helpers (spherical / cylindrical / planar).
//
// BITWISE INVARIANCE CONTRACT (NUMERICS §3, W-G): the spherical branch must
// evaluate the exact floating-point expressions the code used before this
// abstraction existed (same literals, same operation order), so that every
// pre-W-G golden remains bit-identical with geometry_1d unset. Do NOT
// "simplify" the spherical arithmetic here.
//
// Cylindrical is per unit axial length; planar is per unit area.
// The geometry code is threaded as a plain int kernel argument (uniform per
// launch, warp-divergence-free) rather than a template parameter to avoid
// tripling kernel instantiations.

#if defined(__CUDACC__)
#define TENRYU_GEOM1D_HD __host__ __device__
#else
#define TENRYU_GEOM1D_HD
#endif

namespace tenryu::mesh {

enum class Geometry1D : int {
  kSpherical = 0,
  kCylindrical = 1,
  kPlanar = 2,
};

namespace geometry_1d_detail {
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kFourPi = 12.566370614359172953850573533118;
constexpr double kFourPiOverThree = 4.1887902047863909846168578443727;
constexpr double kTwoPi = 6.283185307179586476925286766559;
}  // namespace geometry_1d_detail

TENRYU_GEOM1D_HD inline double geometry_1d_face_area(const int geom,
                                                        const double r) {
  using namespace geometry_1d_detail;
  if (geom == static_cast<int>(Geometry1D::kCylindrical)) {
    return kTwoPi * r;
  }
  if (geom == static_cast<int>(Geometry1D::kPlanar)) {
    return 1.0;
  }
  return kFourPi * r * r;
}

// Shell volume between radii r0 <= r1. The spherical form reproduces the
// historic (kFourPi / 3.0) * (r1 - r0) * (r1*r1 + r1*r0 + r0*r0) expression
// used by the hydro/void-follower paths bit-for-bit.
TENRYU_GEOM1D_HD inline double geometry_1d_shell_volume(const int geom,
                                                           const double r0,
                                                           const double r1) {
  using namespace geometry_1d_detail;
  if (geom == static_cast<int>(Geometry1D::kCylindrical)) {
    return kPi * (r1 - r0) * (r1 + r0);
  }
  if (geom == static_cast<int>(Geometry1D::kPlanar)) {
    return r1 - r0;
  }
  return (kFourPi / 3.0) * (r1 - r0) * (r1 * r1 + r1 * r0 + r0 * r0);
}

// Cube-difference form used by the mesh recompute / verify helpers:
// (4pi/3) * (r1^3 - r0^3). Kept as a distinct helper because its rounding
// differs from the factored form above and both spellings exist in the
// pre-W-G code; each call site must keep its historic spelling.
TENRYU_GEOM1D_HD inline double geometry_1d_shell_volume_cubes(
    const int geom,
    const double r0,
    const double r1) {
  using namespace geometry_1d_detail;
  if (geom == static_cast<int>(Geometry1D::kCylindrical)) {
    return kPi * (r1 * r1 - r0 * r0);
  }
  if (geom == static_cast<int>(Geometry1D::kPlanar)) {
    return r1 - r0;
  }
  return kFourPiOverThree * (r1 * r1 * r1 - r0 * r0 * r0);
}

}  // namespace tenryu::mesh
