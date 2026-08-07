#pragma once

#include <array>
#include <cmath>

#include "core/config.hpp"

#ifndef TENRYU_HOST_DEVICE
#define TENRYU_HOST_DEVICE __host__ __device__
#endif

namespace tenryu::laser {

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

TENRYU_HOST_DEVICE inline Vec3 make_vec3(const double x,
                                         const double y,
                                         const double z) {
  return Vec3{x, y, z};
}

TENRYU_HOST_DEVICE inline Vec3 add(const Vec3& a, const Vec3& b) {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

TENRYU_HOST_DEVICE inline Vec3 sub(const Vec3& a, const Vec3& b) {
  return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

TENRYU_HOST_DEVICE inline Vec3 mul(const Vec3& a, const double s) {
  return Vec3{a.x * s, a.y * s, a.z * s};
}

TENRYU_HOST_DEVICE inline double dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

TENRYU_HOST_DEVICE inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3{a.y * b.z - a.z * b.y,
              a.z * b.x - a.x * b.z,
              a.x * b.y - a.y * b.x};
}

TENRYU_HOST_DEVICE inline double norm(const Vec3& a) {
  return ::sqrt(dot(a, a));
}

TENRYU_HOST_DEVICE inline Vec3 normalize(const Vec3& a) {
  const double n = norm(a);
  if (!(n > 0.0)) {
    return Vec3{0.0, 0.0, 1.0};
  }
  const double inv = 1.0 / n;
  return Vec3{a.x * inv, a.y * inv, a.z * inv};
}

struct BeamBasis {
  Vec3 u_hat{1.0, 0.0, 0.0};
  Vec3 w_hat{0.0, 1.0, 0.0};
  Vec3 d_hat{0.0, 0.0, 1.0};
};

TENRYU_HOST_DEVICE inline BeamBasis build_beam_basis(const Vec3& d_hat_in) {
  const Vec3 d_hat = normalize(d_hat_in);
  const Vec3 e_z{0.0, 0.0, 1.0};
  const Vec3 e_x{1.0, 0.0, 0.0};
  const Vec3 e_ref = (::fabs(dot(d_hat, e_z)) < 0.9) ? e_z : e_x;
  const Vec3 u_hat = normalize(cross(e_ref, d_hat));
  const Vec3 w_hat = normalize(cross(d_hat, u_hat));
  return BeamBasis{u_hat, w_hat, d_hat};
}

TENRYU_HOST_DEVICE inline void to_laser_rz(const double x,
                                           const double y,
                                           const double z,
                                           double& R,
                                           double& Z) {
  R = ::sqrt(x * x + y * y);
  Z = z;
}

TENRYU_HOST_DEVICE inline Vec3 grad_2d_to_3d(const double dndR,
                                             const double dndZ,
                                             const double x,
                                             const double y,
                                             const double z,
                                             const double dx_lm_min) {
  (void)z;
  const double R = ::sqrt(x * x + y * y);
  const double R_floor = 1.0e-12 * dx_lm_min;
  if (R < R_floor || !(R > 0.0)) {
    return Vec3{0.0, 0.0, dndZ};
  }
  const double invR = 1.0 / R;
  return Vec3{dndR * x * invR, dndR * y * invR, dndZ};
}

struct CoordinateTransform {
  Vec3 origin{0.0, 0.0, 0.0};
  Vec3 axis_x{1.0, 0.0, 0.0};
  Vec3 axis_y{0.0, 1.0, 0.0};
  Vec3 axis_z{0.0, 0.0, 1.0};

  TENRYU_HOST_DEVICE inline void to_beam_rz(const Vec3& r_3d,
                                            double& R,
                                            double& Z) const {
    const Vec3 rel = sub(r_3d, origin);
    const double x = dot(rel, axis_x);
    const double y = dot(rel, axis_y);
    Z = dot(rel, axis_z);
    R = ::sqrt(x * x + y * y);
  }

  TENRYU_HOST_DEVICE inline double from_beam_rz(const double R,
                                                const double Z) const {
    return ::sqrt(R * R + Z * Z);
  }
};

std::array<double, 3> resolve_beam_direction(
    const tenryu::core::Config::LaserConfig::BeamDef& beam);

std::array<double, 3> resolve_focus_1d(
    const tenryu::core::Config::LaserConfig::BeamDef& beam,
    double target_radius_cm,
    const std::array<double, 3>& target_center = {0.0, 0.0, 0.0});

CoordinateTransform make_coordinate_transform_1d(
    const tenryu::core::Config::LaserConfig::BeamDef& beam,
    double target_radius_cm,
    const std::array<double, 3>& target_center = {0.0, 0.0, 0.0});

}  // namespace tenryu::laser
