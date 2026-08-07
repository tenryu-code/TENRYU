#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "mesh/interface_tfi.hpp"
#include "mesh/layered_tee_geometry.hpp"

namespace tenryu::mesh::assembly {

struct LayeredTeeCollarGrid {
  int n_s = 0;  // Cells along south/north (radial-ladder direction).
  int n_n = 0;  // Cells along west/east (cut-to-port direction).
  // Node coordinates, (n_s+1) x (n_n+1), index = i + j*(n_s+1),
  // i along south (u_s), j along west (u_w); j=0 row ON the south curve,
  // j=n_n row ON the north curve, i=0 column ON the west curve, i=n_s column
  // ON the east curve.
  std::vector<double> r;
  std::vector<double> z;
};

namespace detail {

inline void layered_tee_require_positive_jacobian(
    const TfiPatch& patch) {
  const double minimum = min_corner_jacobian(patch);
  int minimum_i = 0;
  int minimum_j = 0;
  double located_minimum = std::numeric_limits<double>::infinity();
  const std::size_t nj_size = static_cast<std::size_t>(patch.nj);
  const auto index = [nj_size](const int i, const int j) {
    return static_cast<std::size_t>(i) * (nj_size + 1U) +
           static_cast<std::size_t>(j);
  };
  for (int i = 0; i < patch.ni; ++i) {
    for (int j = 0; j < patch.nj; ++j) {
      const std::size_t corners[4] = {
          index(i, j), index(i + 1, j), index(i + 1, j + 1),
          index(i, j + 1)};
      for (int k = 0; k < 4; ++k) {
        const std::size_t current = corners[k];
        const std::size_t next = corners[(k + 1) % 4];
        const std::size_t previous = corners[(k + 3) % 4];
        const double next_r = patch.r[next] - patch.r[current];
        const double next_z = patch.z[next] - patch.z[current];
        const double previous_r = patch.r[previous] - patch.r[current];
        const double previous_z = patch.z[previous] - patch.z[current];
        const double jacobian =
            next_r * previous_z - next_z * previous_r;
        if (jacobian < located_minimum) {
          located_minimum = jacobian;
          minimum_i = i;
          minimum_j = j;
        }
      }
    }
  }
  TENRYU_ASSERT(
      minimum > 0.0,
      "layered tee collar cell (" + std::to_string(minimum_i) + "," +
          std::to_string(minimum_j) +
          ") must have positive corner Jacobians");
}

inline std::vector<double> layered_tee_geometric_depths(
    const double h, const int n, const double growth) {
  std::vector<double> depths(static_cast<std::size_t>(n) + 1U, 0.0);
  double interval = h;
  for (int k = 1; k <= n; ++k) {
    const std::size_t index = static_cast<std::size_t>(k);
    depths[index] = depths[index - 1U] + interval;
    interval *= growth;
  }
  return depths;
}

}  // namespace detail

inline LayeredTeeCollarGrid build_layered_tee_collar(
    const LayeredTeeFrame& frame, const int n_s, const int n_n,
    const int smooth_iters) {
  TENRYU_ASSERT(n_s >= 2, "layered tee collar n_s must be at least two");
  TENRYU_ASSERT(n_n >= 2, "layered tee collar n_n must be at least two");

  BoundaryChain south;
  BoundaryChain north;
  BoundaryChain west;
  BoundaryChain east;
  const std::size_t n_s_nodes = static_cast<std::size_t>(n_s) + 1U;
  const std::size_t n_n_nodes = static_cast<std::size_t>(n_n) + 1U;
  south.r.resize(n_s_nodes);
  south.z.resize(n_s_nodes);
  north.r.resize(n_s_nodes);
  north.z.resize(n_s_nodes);
  west.r.resize(n_n_nodes);
  west.z.resize(n_n_nodes);
  east.r.resize(n_n_nodes);
  east.z.resize(n_n_nodes);

  for (int i = 0; i <= n_s; ++i) {
    const std::size_t index = static_cast<std::size_t>(i);
    const double u = static_cast<double>(i) / static_cast<double>(n_s);
    layered_tee_south(frame, u, &south.r[index], &south.z[index]);
    layered_tee_north(frame, u, &north.r[index], &north.z[index]);
  }
  for (int j = 0; j <= n_n; ++j) {
    const std::size_t index = static_cast<std::size_t>(j);
    const double u = static_cast<double>(j) / static_cast<double>(n_n);
    layered_tee_west(frame, u, &west.r[index], &west.z[index]);
    layered_tee_east(frame, u, &east.r[index], &east.z[index]);
  }

  TfiPatch patch = tfi_patch(south, north, west, east);
  detail::layered_tee_require_positive_jacobian(patch);

  if (smooth_iters > 0) {
    std::vector<std::uint8_t> pinned(n_s_nodes * n_n_nodes, 0U);
    for (int i = 0; i <= n_s; ++i) {
      for (int j = 0; j <= n_n; ++j) {
        if (i == 0 || i == n_s || j == 0 || j == n_n) {
          const std::size_t index =
              static_cast<std::size_t>(i) * n_n_nodes +
              static_cast<std::size_t>(j);
          pinned[index] = 1U;
        }
      }
    }
    smooth_patch_winslow(patch, pinned, smooth_iters);
    detail::layered_tee_require_positive_jacobian(patch);
  }

  LayeredTeeCollarGrid grid;
  grid.n_s = n_s;
  grid.n_n = n_n;
  grid.r.resize(n_s_nodes * n_n_nodes);
  grid.z.resize(n_s_nodes * n_n_nodes);
  for (int i = 0; i <= n_s; ++i) {
    for (int j = 0; j <= n_n; ++j) {
      const std::size_t patch_index =
          static_cast<std::size_t>(i) * n_n_nodes +
          static_cast<std::size_t>(j);
      const std::size_t grid_index =
          static_cast<std::size_t>(i) +
          static_cast<std::size_t>(j) * n_s_nodes;
      grid.r[grid_index] = patch.r[patch_index];
      grid.z[grid_index] = patch.z[patch_index];
    }
  }
  return grid;
}

inline OffsetCornerSpec layered_tee_triple_corner_spec(
    const LayeredTeeFrame& frame, const bool outer,
    const double collar_centroid_r, const double collar_centroid_z,
    const double h_a, const double h_b, const int n_a, const int n_b,
    const double growth) {
  TENRYU_ASSERT(std::isfinite(collar_centroid_r) &&
                    std::isfinite(collar_centroid_z),
                "layered tee collar centroid must be finite");
  TENRYU_ASSERT(std::isfinite(h_a) && h_a > 0.0,
                "layered tee corner h_a must be positive and finite");
  TENRYU_ASSERT(std::isfinite(h_b) && h_b > 0.0,
                "layered tee corner h_b must be positive and finite");
  TENRYU_ASSERT(n_a >= 1,
                "layered tee corner n_a must be at least one");
  TENRYU_ASSERT(n_b >= 1,
                "layered tee corner n_b must be at least one");
  TENRYU_ASSERT(std::isfinite(growth) && growth >= 1.0,
                "layered tee corner growth must be finite and at least one");

  OffsetCornerSpec spec;
  if (outer) {
    detail::layered_tee_outer_triple_corner(
        frame, &spec.corner_r, &spec.corner_z);
  } else {
    detail::layered_tee_inner_triple_corner(
        frame, &spec.corner_r, &spec.corner_z);
  }

  spec.normal_a_r = -frame.d_z;
  spec.normal_a_z = frame.d_r;
  double cone_side_dot =
      spec.normal_a_r * (collar_centroid_r - spec.corner_r) +
      spec.normal_a_z * (collar_centroid_z - spec.corner_z);
  TENRYU_ASSERT(cone_side_dot != 0.0,
                "layered tee cone normal must not be tangent to the collar "
                "centroid direction");
  if (cone_side_dot > 0.0) {
    spec.normal_a_r = -spec.normal_a_r;
    spec.normal_a_z = -spec.normal_a_z;
    cone_side_dot = -cone_side_dot;
  }
  TENRYU_ASSERT(cone_side_dot < 0.0,
                "layered tee cone normal must point away from the collar");

  const double corner_radius = std::hypot(spec.corner_r, spec.corner_z);
  TENRYU_ASSERT(corner_radius > 0.0,
                "layered tee triple corner radius must be positive");
  const double shell_sign = outer ? 1.0 : -1.0;
  spec.normal_b_r = shell_sign * spec.corner_r / corner_radius;
  spec.normal_b_z = shell_sign * spec.corner_z / corner_radius;
  spec.depths_a =
      detail::layered_tee_geometric_depths(h_a, n_a, growth);
  spec.depths_b =
      detail::layered_tee_geometric_depths(h_b, n_b, growth);

  double det = spec.normal_a_r * spec.normal_b_z -
               spec.normal_a_z * spec.normal_b_r;
  if (det <= 0.0) {
    std::swap(spec.normal_a_r, spec.normal_b_r);
    std::swap(spec.normal_a_z, spec.normal_b_z);
    spec.depths_a.swap(spec.depths_b);
    det = spec.normal_a_r * spec.normal_b_z -
          spec.normal_a_z * spec.normal_b_r;
  }
  TENRYU_ASSERT(det > 0.0,
                "layered tee corner family determinant must be positive");
  TENRYU_ASSERT(det >= 0.199368,
                "layered tee corner conditioning gate requires det >= "
                "0.199368");
  return spec;
}

}  // namespace tenryu::mesh::assembly
