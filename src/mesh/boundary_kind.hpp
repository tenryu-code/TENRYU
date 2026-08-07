#pragma once

#include <cstdint>

namespace tenryu::mesh {

enum class BoundaryKind : std::uint8_t {
  Interior = 0,
  RectRInnerAxis,
  RectROuter,
  RectZBottom,
  RectZTop,
  RZAxisTheta0,
  RZAxisThetaPi,
  SphericalInnerCore,
  SphericalOuterFree,
  SEAM_CORE_BRIDGE,
  SEAM_BRIDGE_SHELL,
  AXIS_SHELL,
  PolarCutFace,
};

static_assert(static_cast<int>(BoundaryKind::Interior) == 0);
static_assert(static_cast<int>(BoundaryKind::RectRInnerAxis) == 1);
static_assert(static_cast<int>(BoundaryKind::RectROuter) == 2);
static_assert(static_cast<int>(BoundaryKind::RectZBottom) == 3);
static_assert(static_cast<int>(BoundaryKind::RectZTop) == 4);
static_assert(static_cast<int>(BoundaryKind::RZAxisTheta0) == 5);
static_assert(static_cast<int>(BoundaryKind::RZAxisThetaPi) == 6);
static_assert(static_cast<int>(BoundaryKind::SphericalInnerCore) == 7);
static_assert(static_cast<int>(BoundaryKind::SphericalOuterFree) == 8);
static_assert(static_cast<int>(BoundaryKind::PolarCutFace) == 12);

inline const char* boundary_kind_name(const BoundaryKind k) {
  switch (k) {
    case BoundaryKind::Interior:
      return "interior";
    case BoundaryKind::RectRInnerAxis:
      return "rect_r_inner_axis";
    case BoundaryKind::RectROuter:
      return "rect_r_outer";
    case BoundaryKind::RectZBottom:
      return "rect_z_bottom";
    case BoundaryKind::RectZTop:
      return "rect_z_top";
    case BoundaryKind::RZAxisTheta0:
      return "rz_axis_theta0";
    case BoundaryKind::RZAxisThetaPi:
      return "rz_axis_theta_pi";
    case BoundaryKind::SphericalInnerCore:
      return "spherical_inner_core";
    case BoundaryKind::SphericalOuterFree:
      return "spherical_outer_free";
    case BoundaryKind::SEAM_CORE_BRIDGE:
      return "seam_core_bridge";
    case BoundaryKind::SEAM_BRIDGE_SHELL:
      return "seam_bridge_shell";
    case BoundaryKind::AXIS_SHELL:
      return "axis_shell";
    case BoundaryKind::PolarCutFace:
      return "polar_cut_face";
  }
  return "unknown";
}

}  // namespace tenryu::mesh
