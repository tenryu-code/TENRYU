#pragma once

#include <algorithm>
#include <cmath>
#include <numbers>

#include "core/error.hpp"

namespace tenryu::mesh::assembly {

// Host-only analytic geometry for the LAYERED_TEE collar in the RZ half-plane
// (r >= 0). The spherical shell is centered at the origin with inner radius
// R_in and outer radius R_out, where 0 < R_in < R_out. The cone-wall outer
// face is P(q) = A + q*d for apex A=(r_a,z_a), q >= 0, and unit direction
// d=(d_r,d_z), where r_a >= 0 and d_r > 0. Polar angle is
// theta=atan2(r,z) in (0,pi), measured from +z.

// Smallest q >= 0 for which |A + q*d| = R.
inline double cone_face_circle_intersection_q(
    const double r_a, const double z_a, const double d_r,
    const double d_z, const double R) {
  TENRYU_ASSERT(std::isfinite(r_a) && std::isfinite(z_a) &&
                    std::isfinite(d_r) && std::isfinite(d_z) &&
                    std::isfinite(R),
                "cone face circle intersection inputs must be finite");
  TENRYU_ASSERT(r_a >= 0.0,
                "cone face apex radial coordinate must be non-negative");
  TENRYU_ASSERT(d_r > 0.0,
                "cone face direction must point toward increasing radius");
  TENRYU_ASSERT(R > 0.0, "circle radius must be positive");
  TENRYU_ASSERT(std::abs(std::hypot(d_r, d_z) - 1.0) <= 1.0e-12,
                "cone face direction must be unit length to 1e-12");

  const double projection = r_a * d_r + z_a * d_z;
  const double radius_residual = r_a * r_a + z_a * z_a - R * R;
  const double discriminant =
      projection * projection - radius_residual;
  TENRYU_ASSERT(discriminant > 0.0,
                "cone face line must intersect the circle with positive "
                "discriminant");

  const double root = std::sqrt(discriminant);
  const double q_near = -projection - root;
  if (q_near >= 0.0) {
    return q_near;
  }
  const double q_far = -projection + root;
  TENRYU_ASSERT(q_far >= 0.0,
                "cone face circle intersection must have q >= 0");
  return q_far;
}

struct LayeredTeeFrame {
  // Triple points: outer-face x shell-inner and outer-face x shell-outer.
  double q_tri_inner = 0.0;
  double q_tri_outer = 0.0;
  double theta_tri_inner = 0.0;
  double theta_tri_outer = 0.0;

  // Constant polar cut for every shell radial row, outside the intersection
  // band by the prescribed margin.
  double theta_cut = 0.0;

  // Analytic collar descriptors. The four curves are parameterized on [0,1]:
  // S: theta=theta_cut, R_in -> R_out;
  // N: cone outer face, q_tri_inner -> q_tri_outer;
  // W: R=R_in, theta_cut -> theta_tri_inner;
  // E: R=R_out, theta_cut -> theta_tri_outer.
  double R_in = 0.0, R_out = 0.0;
  double r_a = 0.0, z_a = 0.0, d_r = 0.0, d_z = 0.0;
};

inline LayeredTeeFrame make_layered_tee_frame(
    const double R_in, const double R_out, const double r_a,
    const double z_a, const double d_r, const double d_z,
    const double cut_margin_theta) {
  TENRYU_ASSERT(std::isfinite(R_in) && std::isfinite(R_out),
                "layered tee shell radii must be finite");
  TENRYU_ASSERT(R_in > 0.0 && R_out > R_in,
                "layered tee shell radii must satisfy 0 < R_in < R_out");
  TENRYU_ASSERT(std::isfinite(cut_margin_theta) &&
                    cut_margin_theta > 0.0,
                "layered tee cut angular margin must be positive and finite");

  LayeredTeeFrame frame;
  frame.R_in = R_in;
  frame.R_out = R_out;
  frame.r_a = r_a;
  frame.z_a = z_a;
  frame.d_r = d_r;
  frame.d_z = d_z;
  frame.q_tri_inner =
      cone_face_circle_intersection_q(r_a, z_a, d_r, d_z, R_in);
  frame.q_tri_outer =
      cone_face_circle_intersection_q(r_a, z_a, d_r, d_z, R_out);
  TENRYU_ASSERT(frame.q_tri_outer > frame.q_tri_inner,
                "layered tee outer triple point must follow the inner triple "
                "point along the cone face");

  const double inner_r = r_a + frame.q_tri_inner * d_r;
  const double inner_z = z_a + frame.q_tri_inner * d_z;
  const double outer_r = r_a + frame.q_tri_outer * d_r;
  const double outer_z = z_a + frame.q_tri_outer * d_z;
  TENRYU_ASSERT(inner_r > 0.0 && outer_r > 0.0,
                "layered tee triple points must lie in the open r > 0 "
                "half-plane");

  frame.theta_tri_inner = std::atan2(inner_r, inner_z);
  frame.theta_tri_outer = std::atan2(outer_r, outer_z);
  frame.theta_cut =
      std::max(frame.theta_tri_inner, frame.theta_tri_outer) +
      cut_margin_theta;
  TENRYU_ASSERT(frame.theta_cut < std::numbers::pi_v<double>,
                "layered tee cut angle must stay below pi");
  return frame;
}

namespace detail {

inline void layered_tee_boundary_arguments(
    const double u, double* const r, double* const z) {
  TENRYU_ASSERT(std::isfinite(u) && u >= 0.0 && u <= 1.0,
                "layered tee boundary parameter must be in [0,1]");
  TENRYU_ASSERT(r != nullptr && z != nullptr,
                "layered tee boundary outputs must not be null");
}

inline void layered_tee_polar_point(const double radius,
                                    const double theta,
                                    double* const r,
                                    double* const z) {
  *r = radius * std::sin(theta);
  *z = radius * std::cos(theta);
}

inline void layered_tee_cone_point(const LayeredTeeFrame& frame,
                                   const double q, double* const r,
                                   double* const z) {
  *r = frame.r_a + q * frame.d_r;
  *z = frame.z_a + q * frame.d_z;
}

inline void layered_tee_inner_cut_corner(const LayeredTeeFrame& frame,
                                         double* const r,
                                         double* const z) {
  layered_tee_polar_point(frame.R_in, frame.theta_cut, r, z);
}

inline void layered_tee_outer_cut_corner(const LayeredTeeFrame& frame,
                                         double* const r,
                                         double* const z) {
  layered_tee_polar_point(frame.R_out, frame.theta_cut, r, z);
}

inline void layered_tee_inner_triple_corner(const LayeredTeeFrame& frame,
                                            double* const r,
                                            double* const z) {
  layered_tee_cone_point(frame, frame.q_tri_inner, r, z);
}

inline void layered_tee_outer_triple_corner(const LayeredTeeFrame& frame,
                                            double* const r,
                                            double* const z) {
  layered_tee_cone_point(frame, frame.q_tri_outer, r, z);
}

}  // namespace detail

inline void layered_tee_south(const LayeredTeeFrame& frame, const double u,
                              double* const r, double* const z) {
  detail::layered_tee_boundary_arguments(u, r, z);
  if (u == 0.0) {
    detail::layered_tee_inner_cut_corner(frame, r, z);
    return;
  }
  if (u == 1.0) {
    detail::layered_tee_outer_cut_corner(frame, r, z);
    return;
  }
  const double radius = frame.R_in + u * (frame.R_out - frame.R_in);
  detail::layered_tee_polar_point(radius, frame.theta_cut, r, z);
}

inline void layered_tee_north(const LayeredTeeFrame& frame, const double u,
                              double* const r, double* const z) {
  detail::layered_tee_boundary_arguments(u, r, z);
  if (u == 0.0) {
    detail::layered_tee_inner_triple_corner(frame, r, z);
    return;
  }
  if (u == 1.0) {
    detail::layered_tee_outer_triple_corner(frame, r, z);
    return;
  }
  const double q =
      frame.q_tri_inner + u * (frame.q_tri_outer - frame.q_tri_inner);
  detail::layered_tee_cone_point(frame, q, r, z);
}

inline void layered_tee_west(const LayeredTeeFrame& frame, const double u,
                             double* const r, double* const z) {
  detail::layered_tee_boundary_arguments(u, r, z);
  if (u == 0.0) {
    detail::layered_tee_inner_cut_corner(frame, r, z);
    return;
  }
  if (u == 1.0) {
    detail::layered_tee_inner_triple_corner(frame, r, z);
    return;
  }
  const double theta = frame.theta_cut +
                       u * (frame.theta_tri_inner - frame.theta_cut);
  detail::layered_tee_polar_point(frame.R_in, theta, r, z);
}

inline void layered_tee_east(const LayeredTeeFrame& frame, const double u,
                             double* const r, double* const z) {
  detail::layered_tee_boundary_arguments(u, r, z);
  if (u == 0.0) {
    detail::layered_tee_outer_cut_corner(frame, r, z);
    return;
  }
  if (u == 1.0) {
    detail::layered_tee_outer_triple_corner(frame, r, z);
    return;
  }
  const double theta = frame.theta_cut +
                       u * (frame.theta_tri_outer - frame.theta_cut);
  detail::layered_tee_polar_point(frame.R_out, theta, r, z);
}

struct TriplePointFrame {
  double shell_tangent_r = 0.0, shell_tangent_z = 0.0;
  double cone_dir_r = 0.0, cone_dir_z = 0.0;
  double crossing_angle = 0.0;
};

inline TriplePointFrame layered_tee_triple_frame(
    const LayeredTeeFrame& frame, const bool outer) {
  const double radius = outer ? frame.R_out : frame.R_in;
  const double q = outer ? frame.q_tri_outer : frame.q_tri_inner;
  double triple_r = 0.0;
  double triple_z = 0.0;
  detail::layered_tee_cone_point(frame, q, &triple_r, &triple_z);

  TriplePointFrame triple;
  triple.shell_tangent_r = triple_z / radius;
  triple.shell_tangent_z = -triple_r / radius;
  triple.cone_dir_r = frame.d_r;
  triple.cone_dir_z = frame.d_z;
  const double tangent_dot_cone =
      triple.shell_tangent_r * triple.cone_dir_r +
      triple.shell_tangent_z * triple.cone_dir_z;
  triple.crossing_angle =
      std::acos(std::clamp(std::abs(tangent_dot_cone), 0.0, 1.0));
  TENRYU_ASSERT(triple.crossing_angle > 1.0e-3 &&
                    triple.crossing_angle < std::numbers::pi_v<double>,
                "layered tee triple-point crossing angle must be in "
                "(1e-3,pi)");
  return triple;
}

}  // namespace tenryu::mesh::assembly
