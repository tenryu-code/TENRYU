#pragma once
#ifndef TENRYU_MESH_SPHERICAL_WEDGE_GEOMETRY_CUH_
#define TENRYU_MESH_SPHERICAL_WEDGE_GEOMETRY_CUH_

#include <cmath>

#if defined(__CUDACC__)
#define TENRYU_SPHERICAL_WEDGE_HD __host__ __device__
#else
#define TENRYU_SPHERICAL_WEDGE_HD
#endif

namespace tenryu::mesh {

namespace spherical_wedge_detail {
constexpr double kPi = 3.141592653589793238462643383279502884;
}

TENRYU_SPHERICAL_WEDGE_HD inline double spherical_wedge_volume(
    const double s_low,
    const double s_high,
    const double theta_low,
    const double theta_high) noexcept {
  const double ds3 = s_high * s_high * s_high - s_low * s_low * s_low;
  const double dcos = std::cos(theta_low) - std::cos(theta_high);
  return (2.0 * spherical_wedge_detail::kPi / 3.0) * ds3 * dcos;
}

TENRYU_SPHERICAL_WEDGE_HD inline double spherical_radial_face_area(
    const double s,
    const double theta_low,
    const double theta_high) noexcept {
  const double dcos = std::cos(theta_low) - std::cos(theta_high);
  return 2.0 * spherical_wedge_detail::kPi * s * s * dcos;
}

TENRYU_SPHERICAL_WEDGE_HD inline double spherical_angular_face_area(
    const double theta,
    const double s_low,
    const double s_high) noexcept {
  const double ds2 = s_high * s_high - s_low * s_low;
  return spherical_wedge_detail::kPi * std::sin(theta) * ds2;
}

}  // namespace tenryu::mesh

#endif  // TENRYU_MESH_SPHERICAL_WEDGE_GEOMETRY_CUH_
