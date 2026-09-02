#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define TENRYU_PENTAGON_GEOMETRY_HD __host__ __device__
#else
#define TENRYU_PENTAGON_GEOMETRY_HD
#endif

namespace tenryu::hydro {

struct PentagonPoint {
  double r;
  double z;
};

inline constexpr int kPolygonGeometryMaxVerts = 8;

inline constexpr double kPentagonGeometryPi =
    3.1415926535897932384626433832795028841971693993751;

namespace pentagon_geometry_detail {

TENRYU_PENTAGON_GEOMETRY_HD inline bool finite_double(const double value) {
  return value == value && value != INFINITY && value != -INFINITY;
}

TENRYU_PENTAGON_GEOMETRY_HD inline double edge_cross(
    const PentagonPoint a,
    const PentagonPoint b) {
  const double first_product = a.r * b.z;
  const double second_product = b.r * a.z;
  return first_product - second_product;
}

}  // namespace pentagon_geometry_detail

TENRYU_PENTAGON_GEOMETRY_HD inline PentagonPoint pentagon_center(
    const PentagonPoint x[5]) {
  return {
      0.2 * ((x[0].r + x[4].r) + (x[1].r + x[3].r) + x[2].r),
      0.2 * ((x[0].z + x[4].z) + (x[1].z + x[3].z) + x[2].z),
  };
}

TENRYU_PENTAGON_GEOMETRY_HD inline void pentagon_edge_midpoints(
    const PentagonPoint x[5],
    PentagonPoint midpoints[5]) {
  for (int k = 0; k < 5; ++k) {
    const int kp1 = (k == 4) ? 0 : k + 1;
    midpoints[k].r = 0.5 * (x[k].r + x[kp1].r);
    midpoints[k].z = 0.5 * (x[k].z + x[kp1].z);
  }
}

TENRYU_PENTAGON_GEOMETRY_HD inline double polygon_planar_area(
    const PentagonPoint* x,
    const int n) {
  if (n == 4) {
    const double term0 =
        pentagon_geometry_detail::edge_cross(x[0], x[1]);
    const double term1 =
        pentagon_geometry_detail::edge_cross(x[1], x[2]);
    const double term2 =
        pentagon_geometry_detail::edge_cross(x[2], x[3]);
    const double term3 =
        pentagon_geometry_detail::edge_cross(x[3], x[0]);
    // This fixed pairing is invariant, bit for bit, when a quadrilateral is
    // reflected in z and its traversal is reversed. Pentagon corner
    // subpolygons therefore preserve their mirror-paired values exactly.
    return 0.5 * ((term0 + term3) + (term1 + term2));
  }

  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    const int ip1 = (i + 1 == n) ? 0 : i + 1;
    const double term =
        pentagon_geometry_detail::edge_cross(x[i], x[ip1]);
    sum = sum + term;
  }
  return 0.5 * sum;
}

TENRYU_PENTAGON_GEOMETRY_HD inline double polygon_rz_volume(
    const PentagonPoint* x,
    const int n) {
  if (n == 4) {
    const double term0 = (x[0].r + x[1].r) *
                         pentagon_geometry_detail::edge_cross(x[0], x[1]);
    const double term1 = (x[1].r + x[2].r) *
                         pentagon_geometry_detail::edge_cross(x[1], x[2]);
    const double term2 = (x[2].r + x[3].r) *
                         pentagon_geometry_detail::edge_cross(x[2], x[3]);
    const double term3 = (x[3].r + x[0].r) *
                         pentagon_geometry_detail::edge_cross(x[3], x[0]);
    // Match polygon_planar_area's fixed mirror-invariant quadrilateral order.
    return (kPentagonGeometryPi / 3.0) *
           ((term0 + term3) + (term1 + term2));
  }

  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    const int ip1 = (i + 1 == n) ? 0 : i + 1;
    const double radius_sum = x[i].r + x[ip1].r;
    const double cross =
        pentagon_geometry_detail::edge_cross(x[i], x[ip1]);
    const double term = radius_sum * cross;
    sum = sum + term;
  }
  return (kPentagonGeometryPi / 3.0) * sum;
}

TENRYU_PENTAGON_GEOMETRY_HD inline void pentagon_corner_rz_volumes(
    const PentagonPoint x[5],
    double volumes[5]) {
  const PentagonPoint center = pentagon_center(x);
  PentagonPoint midpoints[5];
  pentagon_edge_midpoints(x, midpoints);

  for (int k = 0; k < 5; ++k) {
    const int km1 = (k == 0) ? 4 : k - 1;
    const PentagonPoint corner[4] = {
        x[k],
        midpoints[k],
        center,
        midpoints[km1],
    };
    volumes[k] = polygon_rz_volume(corner, 4);
  }
}

TENRYU_PENTAGON_GEOMETRY_HD inline void pentagon_corner_planar_areas(
    const PentagonPoint x[5],
    double areas[5]) {
  const PentagonPoint center = pentagon_center(x);
  PentagonPoint midpoints[5];
  pentagon_edge_midpoints(x, midpoints);

  for (int k = 0; k < 5; ++k) {
    const int km1 = (k == 0) ? 4 : k - 1;
    const PentagonPoint corner[4] = {
        x[k],
        midpoints[k],
        center,
        midpoints[km1],
    };
    areas[k] = polygon_planar_area(corner, 4);
  }
}

// n-generic (n<=8) subzonal corner geometry for reconnected polygons;
// additive only — the fixed-n pentagon/quad paths above are frozen for
// bitwise neutrality.

TENRYU_PENTAGON_GEOMETRY_HD inline PentagonPoint polygon_center(
    const PentagonPoint* x,
    const int n) {
  PentagonPoint sum = {0.0, 0.0};
  for (int i = 0; i < n; ++i) {
    sum.r = sum.r + x[i].r;
    sum.z = sum.z + x[i].z;
  }
  const double inverse_count = 1.0 / n;
  return {
      sum.r * inverse_count,
      sum.z * inverse_count,
  };
}

TENRYU_PENTAGON_GEOMETRY_HD inline void polygon_edge_midpoints_n(
    const PentagonPoint* x,
    const int n,
    PentagonPoint* midpoints) {
  for (int k = 0; k < n; ++k) {
    const int kp1 = (k + 1 == n) ? 0 : k + 1;
    midpoints[k].r = 0.5 * (x[k].r + x[kp1].r);
    midpoints[k].z = 0.5 * (x[k].z + x[kp1].z);
  }
}

TENRYU_PENTAGON_GEOMETRY_HD inline void polygon_corner_planar_areas_n(
    const PentagonPoint* x,
    const int n,
    double* areas) {
  const PentagonPoint center = polygon_center(x, n);
  PentagonPoint midpoints[kPolygonGeometryMaxVerts];
  polygon_edge_midpoints_n(x, n, midpoints);

  for (int k = 0; k < n; ++k) {
    const int km1 = (k == 0) ? n - 1 : k - 1;
    const PentagonPoint corner[4] = {
        x[k],
        midpoints[k],
        center,
        midpoints[km1],
    };
    areas[k] = polygon_planar_area(corner, 4);
  }
}

TENRYU_PENTAGON_GEOMETRY_HD inline void polygon_corner_rz_volumes_n(
    const PentagonPoint* x,
    const int n,
    double* volumes) {
  const PentagonPoint center = polygon_center(x, n);
  PentagonPoint midpoints[kPolygonGeometryMaxVerts];
  polygon_edge_midpoints_n(x, n, midpoints);

  for (int k = 0; k < n; ++k) {
    const int km1 = (k == 0) ? n - 1 : k - 1;
    const PentagonPoint corner[4] = {
        x[k],
        midpoints[k],
        center,
        midpoints[km1],
    };
    volumes[k] = polygon_rz_volume(corner, 4);
  }
}

TENRYU_PENTAGON_GEOMETRY_HD inline void polygon_fan_areas_n(
    const PentagonPoint* x,
    const int n,
    double* areas) {
  const PentagonPoint center = polygon_center(x, n);
  for (int k = 0; k < n; ++k) {
    const int kp1 = (k + 1 == n) ? 0 : k + 1;
    const PentagonPoint triangle[3] = {
        center,
        x[k],
        x[kp1],
    };
    areas[k] = polygon_planar_area(triangle, 3);
  }
}

TENRYU_PENTAGON_GEOMETRY_HD inline double polygon_fan_min_altitude_n(
    const PentagonPoint* x,
    const int n) {
  const PentagonPoint center = polygon_center(x, n);
  double areas[kPolygonGeometryMaxVerts];
  polygon_fan_areas_n(x, n, areas);

  double minimum_altitude = INFINITY;
  for (int k = 0; k < n; ++k) {
    const double absolute_area = fabs(areas[k]);
    if (absolute_area == 0.0 ||
        !pentagon_geometry_detail::finite_double(absolute_area)) {
      return 0.0;
    }

    const int kp1 = (k + 1 == n) ? 0 : k + 1;
    const double edge_dr = x[kp1].r - x[k].r;
    const double edge_dz = x[kp1].z - x[k].z;
    const double next_dr = center.r - x[kp1].r;
    const double next_dz = center.z - x[kp1].z;
    const double previous_dr = x[k].r - center.r;
    const double previous_dz = x[k].z - center.z;
    const double edge_dr_squared = edge_dr * edge_dr;
    const double edge_dz_squared = edge_dz * edge_dz;
    const double next_dr_squared = next_dr * next_dr;
    const double next_dz_squared = next_dz * next_dz;
    const double previous_dr_squared = previous_dr * previous_dr;
    const double previous_dz_squared = previous_dz * previous_dz;
    const double edge_length =
        sqrt(edge_dr_squared + edge_dz_squared);
    const double next_length =
        sqrt(next_dr_squared + next_dz_squared);
    const double previous_length =
        sqrt(previous_dr_squared + previous_dz_squared);
    const double maximum_length =
        fmax(edge_length, fmax(next_length, previous_length));
    if (!(maximum_length > 0.0) ||
        !pentagon_geometry_detail::finite_double(maximum_length)) {
      return 0.0;
    }

    const double altitude = 2.0 * absolute_area / maximum_length;
    if (!pentagon_geometry_detail::finite_double(altitude)) {
      return 0.0;
    }
    minimum_altitude = fmin(minimum_altitude, altitude);
  }
  return minimum_altitude;
}

TENRYU_PENTAGON_GEOMETRY_HD inline double polygon_hydraulic_length_n(
    const PentagonPoint* x,
    const int n) {
  double perimeter = 0.0;
  for (int k = 0; k < n; ++k) {
    const int kp1 = (k + 1 == n) ? 0 : k + 1;
    const double dr = x[kp1].r - x[k].r;
    const double dz = x[kp1].z - x[k].z;
    const double dr_squared = dr * dr;
    const double dz_squared = dz * dz;
    const double edge_length = sqrt(dr_squared + dz_squared);
    perimeter = perimeter + edge_length;
  }
  if (perimeter == 0.0) {
    return 0.0;
  }
  return 2.0 * fabs(polygon_planar_area(x, n)) / perimeter;
}

TENRYU_PENTAGON_GEOMETRY_HD inline double polygon_acoustic_length_n(
    const PentagonPoint* x,
    const int n) {
  return fmin(polygon_hydraulic_length_n(x, n),
              polygon_fan_min_altitude_n(x, n));
}

TENRYU_PENTAGON_GEOMETRY_HD inline bool polygon_kernel_ok_n(
    const PentagonPoint* x,
    const int n,
    const double orient_sign,
    const double area_floor) {
  double areas[kPolygonGeometryMaxVerts];
  polygon_fan_areas_n(x, n, areas);
  for (int k = 0; k < n; ++k) {
    if (!(orient_sign * areas[k] > area_floor)) {
      return false;
    }
  }
  return true;
}

TENRYU_PENTAGON_GEOMETRY_HD inline void pentagon_fan_areas(
    const PentagonPoint x[5],
    double areas[5]) {
  const PentagonPoint center = pentagon_center(x);
  for (int k = 0; k < 5; ++k) {
    const int kp1 = (k == 4) ? 0 : k + 1;
    const PentagonPoint triangle[3] = {
        center,
        x[k],
        x[kp1],
    };
    areas[k] = polygon_planar_area(triangle, 3);
  }
}

TENRYU_PENTAGON_GEOMETRY_HD inline double pentagon_fan_min_altitude(
    const PentagonPoint x[5]) {
  const PentagonPoint center = pentagon_center(x);
  double areas[5];
  pentagon_fan_areas(x, areas);

  double minimum_altitude = INFINITY;
  for (int k = 0; k < 5; ++k) {
    const double absolute_area = fabs(areas[k]);
    if (absolute_area == 0.0 ||
        !pentagon_geometry_detail::finite_double(absolute_area)) {
      return 0.0;
    }

    const int kp1 = (k == 4) ? 0 : k + 1;
    const double edge_dr = x[kp1].r - x[k].r;
    const double edge_dz = x[kp1].z - x[k].z;
    const double next_dr = center.r - x[kp1].r;
    const double next_dz = center.z - x[kp1].z;
    const double previous_dr = x[k].r - center.r;
    const double previous_dz = x[k].z - center.z;
    const double edge_dr_squared = edge_dr * edge_dr;
    const double edge_dz_squared = edge_dz * edge_dz;
    const double next_dr_squared = next_dr * next_dr;
    const double next_dz_squared = next_dz * next_dz;
    const double previous_dr_squared = previous_dr * previous_dr;
    const double previous_dz_squared = previous_dz * previous_dz;
    const double edge_length =
        sqrt(edge_dr_squared + edge_dz_squared);
    const double next_length =
        sqrt(next_dr_squared + next_dz_squared);
    const double previous_length =
        sqrt(previous_dr_squared + previous_dz_squared);
    const double maximum_length =
        fmax(edge_length, fmax(next_length, previous_length));
    if (!(maximum_length > 0.0) ||
        !pentagon_geometry_detail::finite_double(maximum_length)) {
      return 0.0;
    }

    const double altitude = 2.0 * absolute_area / maximum_length;
    if (!pentagon_geometry_detail::finite_double(altitude)) {
      return 0.0;
    }
    minimum_altitude = fmin(minimum_altitude, altitude);
  }
  return minimum_altitude;
}

TENRYU_PENTAGON_GEOMETRY_HD inline double pentagon_hydraulic_length(
    const PentagonPoint x[5]) {
  double perimeter = 0.0;
  for (int k = 0; k < 5; ++k) {
    const int kp1 = (k == 4) ? 0 : k + 1;
    const double dr = x[kp1].r - x[k].r;
    const double dz = x[kp1].z - x[k].z;
    const double dr_squared = dr * dr;
    const double dz_squared = dz * dz;
    const double edge_length = sqrt(dr_squared + dz_squared);
    perimeter = perimeter + edge_length;
  }
  if (perimeter == 0.0) {
    return 0.0;
  }
  return 2.0 * fabs(polygon_planar_area(x, 5)) / perimeter;
}

TENRYU_PENTAGON_GEOMETRY_HD inline double pentagon_acoustic_length(
    const PentagonPoint x[5]) {
  return fmin(pentagon_hydraulic_length(x),
              pentagon_fan_min_altitude(x));
}

TENRYU_PENTAGON_GEOMETRY_HD inline bool pentagon_kernel_ok(
    const PentagonPoint x[5],
    const double orient_sign,
    const double area_floor) {
  double areas[5];
  pentagon_fan_areas(x, areas);
  for (int k = 0; k < 5; ++k) {
    if (!(orient_sign * areas[k] > area_floor)) {
      return false;
    }
  }
  return true;
}

}  // namespace tenryu::hydro

#undef TENRYU_PENTAGON_GEOMETRY_HD
