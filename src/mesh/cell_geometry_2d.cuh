#pragma once
#ifndef TENRYU_MESH_CELL_GEOMETRY_2D_CUH_
#define TENRYU_MESH_CELL_GEOMETRY_2D_CUH_

#include <cmath>

#if defined(__CUDACC__)
#define TENRYU_CELL_GEOM_HD __host__ __device__
#else
#define TENRYU_CELL_GEOM_HD
#endif

namespace tenryu::mesh {

struct CellWidths2D {
  double h_R, h_Z;
  double A_R_left, A_R_right, A_Z_bottom, A_Z_top;
};

namespace detail {

constexpr double kPi = 3.14159265358979323846;
constexpr double kWidthFloor = 1.0e-30;

TENRYU_CELL_GEOM_HD inline bool finite_double(const double x) {
  return x == x && x != INFINITY && x != -INFINITY;
}

TENRYU_CELL_GEOM_HD inline double min2(const double a, const double b) {
  return (a < b) ? a : b;
}

TENRYU_CELL_GEOM_HD inline double max2(const double a, const double b) {
  return (a > b) ? a : b;
}

TENRYU_CELL_GEOM_HD inline double bbox_width_or_floor(const double lo,
                                                       const double hi) {
  const double width = hi - lo;
  return (width > 0.0 && finite_double(width)) ? width : kWidthFloor;
}

TENRYU_CELL_GEOM_HD inline double width_from_volume_area(
    const double volume,
    const double area_avg,
    const double bbox_width) {
  if (area_avg > 0.0 && finite_double(area_avg)) {
    return volume / area_avg;
  }
  return bbox_width_or_floor(0.0, bbox_width);
}

}  // namespace detail

TENRYU_CELL_GEOM_HD inline CellWidths2D compute_cell_widths_2d(
    const double* node_r, const double* node_z, const double* vol,
    int nr, int nz, int c) {
  (void)nr;

  const int i = c / nz;
  const int j = c - i * nz;

  const int n00 = i * (nz + 1) + j;
  const int n10 = (i + 1) * (nz + 1) + j;
  const int n11 = (i + 1) * (nz + 1) + (j + 1);
  const int n01 = i * (nz + 1) + (j + 1);

  const double r00 = node_r[n00];
  const double z00 = node_z[n00];
  const double r10 = node_r[n10];
  const double z10 = node_z[n10];
  const double r11 = node_r[n11];
  const double z11 = node_z[n11];
  const double r01 = node_r[n01];
  const double z01 = node_z[n01];

  const double L_R_left =
      std::sqrt((r01 - r00) * (r01 - r00) + (z01 - z00) * (z01 - z00));
  const double L_R_right =
      std::sqrt((r11 - r10) * (r11 - r10) + (z11 - z10) * (z11 - z10));

  CellWidths2D out{};
  out.A_R_left = 2.0 * detail::kPi * (0.5 * (r01 + r00)) * L_R_left;
  out.A_R_right = 2.0 * detail::kPi * (0.5 * (r10 + r11)) * L_R_right;
  out.A_Z_bottom = detail::kPi * std::fabs(r10 * r10 - r00 * r00);
  out.A_Z_top = detail::kPi * std::fabs(r01 * r01 - r11 * r11);

  const double r_min =
      detail::min2(detail::min2(r00, r10), detail::min2(r11, r01));
  const double r_max =
      detail::max2(detail::max2(r00, r10), detail::max2(r11, r01));
  const double z_min =
      detail::min2(detail::min2(z00, z10), detail::min2(z11, z01));
  const double z_max =
      detail::max2(detail::max2(z00, z10), detail::max2(z11, z01));

  const double A_R_avg = 0.5 * (out.A_R_left + out.A_R_right);
  const double A_Z_avg = 0.5 * (out.A_Z_bottom + out.A_Z_top);

  out.h_R = detail::width_from_volume_area(vol[c], A_R_avg, r_max - r_min);
  out.h_Z = detail::width_from_volume_area(vol[c], A_Z_avg, z_max - z_min);

  return out;
}

}  // namespace tenryu::mesh

#endif  // TENRYU_MESH_CELL_GEOMETRY_2D_CUH_
