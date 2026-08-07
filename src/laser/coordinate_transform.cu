#include "laser/coordinate_transform.cuh"

#include <algorithm>

namespace tenryu::laser {

std::array<double, 3> resolve_beam_direction(
    const tenryu::core::Config::LaserConfig::BeamDef& beam) {
  Vec3 dir{0.0, 0.0, 0.0};
  if (beam.direction.size() == 3) {
    dir = Vec3{beam.direction[0], beam.direction[1], beam.direction[2]};
  } else {
    const double st = std::sin(beam.theta);
    dir = Vec3{st * std::cos(beam.phi), st * std::sin(beam.phi), std::cos(beam.theta)};
  }
  const Vec3 d_hat = normalize(dir);
  return {d_hat.x, d_hat.y, d_hat.z};
}

std::array<double, 3> resolve_focus_1d(
    const tenryu::core::Config::LaserConfig::BeamDef& beam,
    const double target_radius_cm,
    const std::array<double, 3>& target_center) {
  if (beam.focus.size() == 3) {
    return {beam.focus[0], beam.focus[1], beam.focus[2]};
  }

  const auto dir = resolve_beam_direction(beam);
  return {
      target_center[0] + beam.defocus_DR * target_radius_cm * dir[0],
      target_center[1] + beam.defocus_DR * target_radius_cm * dir[1],
      target_center[2] + beam.defocus_DR * target_radius_cm * dir[2],
  };
}

CoordinateTransform make_coordinate_transform_1d(
    const tenryu::core::Config::LaserConfig::BeamDef& beam,
    const double target_radius_cm,
    const std::array<double, 3>& target_center) {
  const auto dir_arr = resolve_beam_direction(beam);
  const auto focus = resolve_focus_1d(beam, target_radius_cm, target_center);

  (void)focus;  // focus is resolved here for rule consistency; frame origin stays at target center.

  CoordinateTransform tf;
  tf.origin = Vec3{target_center[0], target_center[1], target_center[2]};
  const BeamBasis basis = build_beam_basis(Vec3{dir_arr[0], dir_arr[1], dir_arr[2]});
  tf.axis_x = basis.u_hat;
  tf.axis_y = basis.w_hat;
  tf.axis_z = basis.d_hat;

  return tf;
}

}  // namespace tenryu::laser
