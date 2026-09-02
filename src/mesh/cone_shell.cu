#include "mesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/cone_shell_ladder.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/hydro_multiblock_topology.cuh"

namespace tenryu::mesh {
namespace {

struct ConeShellPoint {
  double r = 0.0;
  double z = 0.0;
};

// Append-only block registry for later cone-shell stages.
enum class ConeShellBlock : int {
  WALL = 0,
  OUTER_NF = 1,
  INNER_NF = 2,
  END_NF = 3,
  TIP_CORNER_O = 4,
  TIP_CORNER_I = 5,
  CAVITY_CORE = 6,
  EXTERIOR_CORE = 7,
  TIP_FILL_WEST = 8,
  TIP_FILL_MID = 9,
  TIP_FILL_EAST = 10,
};

struct ConeShellCornerTemplate {
  int ring_count = 0;
  int point = -1;
  std::vector<int> a;
  std::vector<int> b;
  std::vector<int> c;
};

enum class ConeShellWallFace {
  Inner,
  Outer,
};

int cone_shell_face_j(const tenryu::core::Config::MeshConfig& mesh_cfg,
                      const ConeShellWallFace face) {
  if (mesh_cfg.cone_shell_axis_sign > 0) {
    return face == ConeShellWallFace::Inner ? mesh_cfg.nz : 0;
  }
  return face == ConeShellWallFace::Inner ? 0 : mesh_cfg.nz;
}

std::vector<double> build_strip_distance_ladder(const double h_n0,
                                                const double first_factor,
                                                const int layers,
                                                const double growth) {
  TENRYU_ASSERT(layers >= 1 && h_n0 > 0.0 && first_factor > 0.0 &&
                    growth > 0.0,
                "cone_shell strip distance ladder inputs must be positive "
                "with layers >= 1");
  std::vector<double> distance(static_cast<std::size_t>(layers + 1), 0.0);
  double width = first_factor * h_n0;
  for (int layer = 1; layer <= layers; ++layer) {
    distance[static_cast<std::size_t>(layer)] =
        distance[static_cast<std::size_t>(layer - 1)] + width;
    TENRYU_ASSERT(distance[static_cast<std::size_t>(layer)] >
                      distance[static_cast<std::size_t>(layer - 1)],
                  "cone_shell strip distance ladder must be strictly increasing");
    width *= growth;
  }
  return distance;
}

double strip_last_width(const double h_n0,
                        const double first_factor,
                        const int layers,
                        const double growth) {
  double width = first_factor * h_n0;
  for (int layer = 1; layer < layers; ++layer) {
    width *= growth;
  }
  return width;
}

int strip_node_id(const int interior_offset,
                  const int shared_wall_node,
                  const int layer,
                  const int station,
                  const int station_count) {
  if (layer == 0) {
    return shared_wall_node;
  }
  return interior_offset + (layer - 1) * station_count + station;
}

double cone_shell_end_rotation(const double q,
                               const double rotation_length) {
  if (q >= rotation_length) {
    return 0.0;
  }
  return 1.0 - tenryu::core::cone_shell_q5(q / rotation_length);
}

double cone_shell_base_rotation(const double q,
                                const double wall_length,
                                const double rotation_length) {
  if (q <= wall_length - rotation_length) {
    return 0.0;
  }
  return tenryu::core::cone_shell_q5(
      (q - (wall_length - rotation_length)) / rotation_length);
}

double geometric_sum(const double ratio, const int count) {
  double sum = 0.0;
  double term = 1.0;
  for (int k = 0; k < count; ++k) {
    sum += term;
    term *= ratio;
  }
  return sum;
}

std::vector<double> build_mixed_ladder(
    const double clearance,
    const double first_width,
    const int count,
    const char* name) {
  TENRYU_ASSERT(clearance > 0.0 && first_width > 0.0 && count >= 6 &&
                    count <= 32,
                std::string("cone_shell ") + name +
                    " mixed ladder inputs must be positive with K in [6, 32]");
  std::vector<double> distance(static_cast<std::size_t>(count + 1), 0.0);
  if (static_cast<double>(count) * first_width >= clearance) {
    const double width = clearance / static_cast<double>(count);
    TENRYU_ASSERT(
        width >= 0.5 * first_width,
        std::string("cone_shell ") + name +
            " mixed ladder violates the short-station chi hard band");
    for (int layer = 1; layer < count; ++layer) {
      distance[static_cast<std::size_t>(layer)] =
          clearance * (static_cast<double>(layer) /
                       static_cast<double>(count));
    }
  } else {
    const double upper = first_width * geometric_sum(1.70, count);
    TENRYU_ASSERT(
        upper >= clearance,
        std::string("cone_shell ") + name +
            " mixed ladder requires growth above 1.7; clearance=" +
            std::to_string(clearance) + ", upper=" +
            std::to_string(upper));
    double lo = 1.0;
    double hi = 1.70;
    for (int iteration = 0; iteration < 80; ++iteration) {
      const double mid = 0.5 * (lo + hi);
      if (first_width * geometric_sum(mid, count) < clearance) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const double ratio = 0.5 * (lo + hi);
    double width = first_width;
    for (int layer = 1; layer < count; ++layer) {
      distance[static_cast<std::size_t>(layer)] =
          distance[static_cast<std::size_t>(layer - 1)] + width;
      width *= ratio;
    }
  }
  distance.back() = clearance;
  for (int layer = 0; layer < count; ++layer) {
    TENRYU_ASSERT(distance[static_cast<std::size_t>(layer + 1)] >
                      distance[static_cast<std::size_t>(layer)],
                  std::string("cone_shell ") + name +
                      " mixed ladder must be strictly increasing");
  }
  return distance;
}

struct ConeShellHermiteCurve {
  ConeShellPoint p0;
  ConeShellPoint m0;
  ConeShellPoint p1;
  ConeShellPoint m1;
};

ConeShellPoint cone_shell_hermite_eval(const ConeShellHermiteCurve& curve,
                                       const double u) {
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
  const double h10 = u3 - 2.0 * u2 + u;
  const double h01 = -2.0 * u3 + 3.0 * u2;
  const double h11 = u3 - u2;
  return ConeShellPoint{
      h00 * curve.p0.r + h10 * curve.m0.r + h01 * curve.p1.r +
          h11 * curve.m1.r,
      h00 * curve.p0.z + h10 * curve.m0.z + h01 * curve.p1.z +
          h11 * curve.m1.z,
  };
}

double cone_shell_hermite_speed(const ConeShellHermiteCurve& curve,
                                const double u) {
  const double u2 = u * u;
  const double d00 = 6.0 * u2 - 6.0 * u;
  const double d10 = 3.0 * u2 - 4.0 * u + 1.0;
  const double d01 = -6.0 * u2 + 6.0 * u;
  const double d11 = 3.0 * u2 - 2.0 * u;
  const double dr = d00 * curve.p0.r + d10 * curve.m0.r +
                    d01 * curve.p1.r + d11 * curve.m1.r;
  const double dz = d00 * curve.p0.z + d10 * curve.m0.z +
                    d01 * curve.p1.z + d11 * curve.m1.z;
  return std::hypot(dr, dz);
}

// Fixed 8-point Gauss-Legendre abscissae/weights on [-1, 1].
constexpr std::array<double, 4> kGl8Abscissae{{
    0.18343464249564980493,
    0.52553240991632898582,
    0.79666647741362673959,
    0.96028985649753623168,
}};
constexpr std::array<double, 4> kGl8Weights{{
    0.36268378337836198297,
    0.31370664587788728734,
    0.22238103445337447054,
    0.10122853629037625915,
}};

double cone_shell_hermite_arc_gl8(const ConeShellHermiteCurve& curve,
                                  const double a,
                                  const double b) {
  const double half = 0.5 * (b - a);
  const double center = 0.5 * (a + b);
  double sum = 0.0;
  for (std::size_t k = 0; k < kGl8Abscissae.size(); ++k) {
    const double offset = half * kGl8Abscissae[k];
    sum += kGl8Weights[k] *
           (cone_shell_hermite_speed(curve, center - offset) +
            cone_shell_hermite_speed(curve, center + offset));
  }
  return half * sum;
}

constexpr int kHermiteArcSubintervals = 64;

std::array<double, kHermiteArcSubintervals + 1>
cone_shell_hermite_arc_prefix(const ConeShellHermiteCurve& curve) {
  std::array<double, kHermiteArcSubintervals + 1> prefix{};
  prefix[0] = 0.0;
  for (int i = 0; i < kHermiteArcSubintervals; ++i) {
    const double a = static_cast<double>(i) /
                     static_cast<double>(kHermiteArcSubintervals);
    const double b = static_cast<double>(i + 1) /
                     static_cast<double>(kHermiteArcSubintervals);
    prefix[static_cast<std::size_t>(i + 1)] =
        prefix[static_cast<std::size_t>(i)] +
        cone_shell_hermite_arc_gl8(curve, a, b);
  }
  return prefix;
}

double cone_shell_hermite_arc_at(
    const ConeShellHermiteCurve& curve,
    const std::array<double, kHermiteArcSubintervals + 1>& prefix,
    const double u) {
  if (u <= 0.0) {
    return 0.0;
  }
  if (u >= 1.0) {
    return prefix[kHermiteArcSubintervals];
  }
  const double scaled =
      u * static_cast<double>(kHermiteArcSubintervals);
  int index = static_cast<int>(scaled);
  if (index >= kHermiteArcSubintervals) {
    index = kHermiteArcSubintervals - 1;
  }
  const double a = static_cast<double>(index) /
                   static_cast<double>(kHermiteArcSubintervals);
  return prefix[static_cast<std::size_t>(index)] +
         cone_shell_hermite_arc_gl8(curve, a, u);
}

double cone_shell_hermite_invert_arclength(
    const ConeShellHermiteCurve& curve,
    const std::array<double, kHermiteArcSubintervals + 1>& prefix,
    const double target,
    const char* name) {
  TENRYU_ASSERT(target > 0.0 &&
                    target < prefix[kHermiteArcSubintervals],
                std::string("cone_shell ") + name +
                    " arclength inversion target must lie strictly inside "
                    "the curve length");
  double lo = 0.0;
  double hi = 1.0;
  for (int iteration = 0; iteration < 64; ++iteration) {
    const double mid = 0.5 * (lo + hi);
    if (cone_shell_hermite_arc_at(curve, prefix, mid) < target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

double determinant(const ConeShellPoint& a, const ConeShellPoint& b) {
  return a.r * b.z - b.r * a.z;
}

double dot(const ConeShellPoint& a, const ConeShellPoint& b) {
  return a.r * b.r + a.z * b.z;
}

ConeShellPoint subtract(const ConeShellPoint& a, const ConeShellPoint& b) {
  return {a.r - b.r, a.z - b.z};
}

double exact_rz_volume(const std::array<ConeShellPoint, 4>& points) {
  double sum = 0.0;
  for (int corner = 0; corner < 4; ++corner) {
    const ConeShellPoint& a = points[static_cast<std::size_t>(corner)];
    const ConeShellPoint& b =
        points[static_cast<std::size_t>((corner + 1) % 4)];
    sum += (a.r + b.r) * (a.r * b.z - b.r * a.z);
  }
  constexpr double kPiOverThree =
      1.0471975511965977461542144610931676280657231331250352736615;
  return kPiOverThree * sum;
}

std::array<ConeShellPoint, 4> quad_points(
    const std::array<int, 4>& nodes,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  std::array<ConeShellPoint, 4> points{};
  for (int corner = 0; corner < 4; ++corner) {
    const int node = nodes[static_cast<std::size_t>(corner)];
    points[static_cast<std::size_t>(corner)] = {
        node_r[static_cast<std::size_t>(node)],
        node_z[static_cast<std::size_t>(node)],
    };
  }
  return points;
}

double signed_quad_area_twice(
    const std::array<ConeShellPoint, 4>& points) {
  double area_twice = 0.0;
  for (int corner = 0; corner < 4; ++corner) {
    area_twice += determinant(
        points[static_cast<std::size_t>(corner)],
        points[static_cast<std::size_t>((corner + 1) % 4)]);
  }
  return area_twice;
}

std::array<int, 4> orient_quad_positive(
    std::array<int, 4> nodes,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  double area_twice =
      signed_quad_area_twice(quad_points(nodes, node_r, node_z));
  TENRYU_ASSERT(area_twice != 0.0,
                "cone_shell corner quad signed area must be nonzero");
  if (area_twice < 0.0) {
    std::swap(nodes[1], nodes[3]);
    area_twice = signed_quad_area_twice(quad_points(nodes, node_r, node_z));
  }
  TENRYU_ASSERT(area_twice > 0.0,
                "cone_shell corner quad signed area must be positive");
  return nodes;
}

void set_csr_offsets(std::vector<int>& offsets,
                     const int rows,
                     const int stride) {
  offsets.resize(static_cast<std::size_t>(rows + 1));
  for (int row = 0; row <= rows; ++row) {
    offsets[static_cast<std::size_t>(row)] = stride * row;
  }
}

void set_cell_nodes(MultiBlockTopology& mb,
                    const int cell,
                    const std::array<int, 4>& nodes,
                    const int stride) {
  const int offset =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  for (int corner = 0; corner < stride; ++corner) {
    mb.cell_node_csr_indices[static_cast<std::size_t>(offset + corner)] =
        (corner < 4) ? nodes[static_cast<std::size_t>(corner)] : 0;
  }
}

std::pair<int, int> local_face_nodes(const std::array<int, 4>& nodes,
                                     const int local_face) {
  switch (local_face) {
    case 0:
      return {nodes[3], nodes[0]};
    case 1:
      return {nodes[1], nodes[2]};
    case 2:
      return {nodes[0], nodes[1]};
    case 3:
      return {nodes[2], nodes[3]};
  }
  TENRYU_ASSERT(false, "cone_shell local face index must be in [0, 3]");
  return {-1, -1};
}

struct EdgeIncidence {
  std::array<int, 2> cell{{-1, -1}};
  std::array<int, 2> local_face{{-1, -1}};
  int count = 0;
};

using EdgeIncidenceMap = std::map<std::pair<int, int>, EdgeIncidence>;

EdgeIncidenceMap build_edge_incidence(const MultiBlockTopology& mb,
                                      const int n_cells) {
  EdgeIncidenceMap incidence;
  for (int cell = 0; cell < n_cells; ++cell) {
    const int offset =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const std::array<int, 4> nodes{{
        mb.cell_node_csr_indices[static_cast<std::size_t>(offset + 0)],
        mb.cell_node_csr_indices[static_cast<std::size_t>(offset + 1)],
        mb.cell_node_csr_indices[static_cast<std::size_t>(offset + 2)],
        mb.cell_node_csr_indices[static_cast<std::size_t>(offset + 3)],
    }};
    for (int local_face = 0; local_face < 4; ++local_face) {
      auto edge = local_face_nodes(nodes, local_face);
      if (edge.second < edge.first) {
        std::swap(edge.first, edge.second);
      }
      EdgeIncidence& entry = incidence[edge];
      TENRYU_ASSERT(entry.count < 2,
                    "cone_shell global edge incidence exceeds two cells");
      entry.cell[static_cast<std::size_t>(entry.count)] = cell;
      entry.local_face[static_cast<std::size_t>(entry.count)] = local_face;
      ++entry.count;
    }
  }
  return incidence;
}

void build_face_adjacency_and_boundary_tags(
    MeshTopology& topo,
    MultiBlockTopology& mb,
    const int stride,
    const ConeShellTipPort& tip_port,
    const tenryu::core::Config::MeshConfig& mesh_cfg,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<std::pair<int, int>>& expected_open_base_edges) {
  const int n_cells = topo.n_cells;
  set_csr_offsets(mb.face_adj_csr_offsets, n_cells, stride);
  mb.face_adj_csr_indices.assign(
      static_cast<std::size_t>(stride * n_cells), -1);
  mb.face_bc_tags.assign(
      static_cast<std::size_t>(stride * n_cells),
      static_cast<int>(BoundaryKind::Interior));

  const EdgeIncidenceMap incidence = build_edge_incidence(mb, n_cells);

  topo.node_flags.assign(static_cast<std::size_t>(topo.n_nodes), NODE_NONE);
  const double eps = std::numeric_limits<double>::epsilon();
  const double z_scale = std::max(
      mesh_cfg.cone_shell_wall_length,
      std::max(std::abs(mesh_cfg.box_z_min), std::abs(mesh_cfg.box_z_max)));
  const double box_z_tolerance = 128.0 * eps * z_scale;
  const bool c4b_census = [] {
    const char* env = std::getenv("TENRYU_CONE_C4B_CENSUS");
    return env != nullptr && env[0] == '1';
  }();
  const bool c4b_open_base = mesh_cfg.cone_shell_base_cut == "wall_normal";
  int c4b_bad_edges = 0;
  for (const auto& item : incidence) {
    const EdgeIncidence& entry = item.second;
    TENRYU_ASSERT(entry.count == 1 || entry.count == 2,
                  "cone_shell global edge incidence must be one or two");
    if (entry.count == 2) {
      const int cell_a = entry.cell[0];
      const int cell_b = entry.cell[1];
      const int face_a = entry.local_face[0];
      const int face_b = entry.local_face[1];
      mb.face_adj_csr_indices[
          static_cast<std::size_t>(stride * cell_a + face_a)] = cell_b;
      mb.face_adj_csr_indices[
          static_cast<std::size_t>(stride * cell_b + face_b)] = cell_a;
      continue;
    }

    const int cell = entry.cell[0];
    const int local_face = entry.local_face[0];
    const int node0 = item.first.first;
    const int node1 = item.first.second;
    BoundaryKind kind = BoundaryKind::SphericalOuterFree;
    if (node_r[static_cast<std::size_t>(node0)] == 0.0 &&
        node_r[static_cast<std::size_t>(node1)] == 0.0) {
      kind = BoundaryKind::RectRInnerAxis;
    } else if (node_r[static_cast<std::size_t>(node0)] ==
                   mesh_cfg.box_r_max &&
               node_r[static_cast<std::size_t>(node1)] ==
                   mesh_cfg.box_r_max) {
      kind = BoundaryKind::RectROuter;
    } else if (std::abs(node_z[static_cast<std::size_t>(node0)] -
                        mesh_cfg.box_z_min) <= box_z_tolerance &&
               std::abs(node_z[static_cast<std::size_t>(node1)] -
                        mesh_cfg.box_z_min) <= box_z_tolerance) {
      kind = BoundaryKind::RectZBottom;
    } else if (std::abs(node_z[static_cast<std::size_t>(node0)] -
                        mesh_cfg.box_z_max) <= box_z_tolerance &&
               std::abs(node_z[static_cast<std::size_t>(node1)] -
                        mesh_cfg.box_z_max) <= box_z_tolerance) {
      kind = BoundaryKind::RectZTop;
    }
    const bool c4b_ok = kind == BoundaryKind::RectRInnerAxis ||
                        kind == BoundaryKind::RectROuter ||
                        kind == BoundaryKind::RectZBottom ||
                        kind == BoundaryKind::RectZTop;
    if (!c4b_ok) {
      ++c4b_bad_edges;
      const std::string detail =
          "cone_shell C4b watertightness gate found a non-axis, non-box "
          "boundary edge: n0=(" +
          std::to_string(node_r[static_cast<std::size_t>(node0)]) + "," +
          std::to_string(node_z[static_cast<std::size_t>(node0)]) + ") n1=(" +
          std::to_string(node_r[static_cast<std::size_t>(node1)]) + "," +
          std::to_string(node_z[static_cast<std::size_t>(node1)]) + ")";
      if (c4b_census) {
        tenryu::core::log_warning(detail);
      } else if (!c4b_open_base) {
        TENRYU_ASSERT(false, detail);
      } else {
        const std::pair<int, int> key{std::min(node0, node1),
                                      std::max(node0, node1)};
        TENRYU_ASSERT(
            std::binary_search(expected_open_base_edges.begin(),
                               expected_open_base_edges.end(), key),
            detail +
                " (edge is not on the expected wall-normal open-base "
                "chain)");
      }
    }
    mb.face_bc_tags[static_cast<std::size_t>(stride * cell + local_face)] =
        static_cast<int>(kind);
    // C4b is init-only: these tags support audits and HDF5 and do not wire
    // boundary-condition or hydro semantics.
    for (const int node : {node0, node1}) {
      topo.node_flags[static_cast<std::size_t>(node)] |= NODE_BOUNDARY;
      if (kind == BoundaryKind::RectRInnerAxis) {
        topo.node_flags[static_cast<std::size_t>(node)] |= NODE_AXIS;
      } else {
        topo.node_flags[static_cast<std::size_t>(node)] |=
            NODE_OUTER_PHYSICAL_BOUNDARY;
      }
    }
  }

  if (c4b_open_base) {
    if (!c4b_census) {
      TENRYU_ASSERT(
          c4b_bad_edges ==
              static_cast<int>(expected_open_base_edges.size()),
          "cone_shell wall_normal open-base gate: tagged " +
              std::to_string(c4b_bad_edges) +
              " free boundary edges but the expected base chain has " +
              std::to_string(expected_open_base_edges.size()));
    }
    if (c4b_bad_edges > 0) {
      tenryu::core::log_info(
          "cone_shell wall_normal open-base standalone: " +
          std::to_string(c4b_bad_edges) +
          " boundary edges tagged free (exact expected base chain; base "
          "closure lands with the conforming-assembly stages)");
    }
  } else {
    TENRYU_ASSERT(c4b_bad_edges == 0,
                  "cone_shell C4b watertightness gate: " +
                      std::to_string(c4b_bad_edges) +
                      " unclassified boundary edges");
  }

  TENRYU_ASSERT(!tip_port.node_ids.empty() &&
                    tip_port.node_ids.size() == tip_port.segment.size(),
                "cone_shell TIP_PORT metadata must be nonempty and parallel");
  for (std::size_t i = 0; i < tip_port.node_ids.size(); ++i) {
    const int node = tip_port.node_ids[i];
    TENRYU_ASSERT(node >= 0 && node < topo.n_nodes,
                  "cone_shell TIP_PORT node id is out of range");
    TENRYU_ASSERT(tip_port.segment[i] <= 2U,
                  "cone_shell TIP_PORT segment label is out of range");
    TENRYU_ASSERT(
        (topo.node_flags[static_cast<std::size_t>(node)] & NODE_BOUNDARY) == 0,
        "cone_shell C4b TIP_PORT nodes must be interior");
    if (i > 0U) {
      TENRYU_ASSERT(tip_port.segment[i] >= tip_port.segment[i - 1U],
                    "cone_shell TIP_PORT segment order must be fixed");
      int node0 = tip_port.node_ids[i - 1U];
      int node1 = node;
      if (node0 == node1) {
        TENRYU_ASSERT(
            tip_port.segment[i] == tip_port.segment[i - 1U] + 1U,
            "cone_shell TIP_PORT duplicate must be a shared segment endpoint");
      } else {
        if (node1 < node0) {
          std::swap(node0, node1);
        }
        const auto edge = incidence.find({node0, node1});
        TENRYU_ASSERT(
            edge != incidence.end() && edge->second.count == 2,
            "cone_shell C4b TIP_PORT consecutive nodes must form a shared "
            "interior edge");
      }
    }
    for (std::size_t previous = 0; previous < i; ++previous) {
      if (tip_port.node_ids[previous] == node) {
        TENRYU_ASSERT(
            previous + 1U == i &&
                tip_port.segment[previous] + 1U == tip_port.segment[i],
            "cone_shell TIP_PORT node duplicates are limited to shared "
            "segment endpoints");
      }
    }
  }
  TENRYU_ASSERT(tip_port.segment.front() == 0U &&
                    tip_port.segment.back() == 2U,
                "cone_shell TIP_PORT must run from inner to outer segment");
}

void upload_unique_face_payload(MultiBlockTopology& mb) {
  std::vector<int> unique_cell_a;
  std::vector<int> unique_cell_b;
  std::vector<int> unique_local_a;
  std::vector<int> unique_local_b;
  std::vector<std::int8_t> unique_bc_tag;
  for (const UniqueOrientedFace& face : mb.unique_internal_faces) {
    unique_cell_a.push_back(face.cell_a);
    unique_cell_b.push_back(face.cell_b);
    unique_local_a.push_back(face.local_a);
    unique_local_b.push_back(face.local_b);
    unique_bc_tag.push_back(face.bc_tag);
  }
  std::vector<int> boundary_cell;
  std::vector<int> boundary_local;
  std::vector<std::int8_t> boundary_bc_tag;
  for (const UniqueOrientedFace& face : mb.boundary_faces) {
    boundary_cell.push_back(face.cell_a);
    boundary_local.push_back(face.local_a);
    boundary_bc_tag.push_back(face.bc_tag);
  }
  mb.d_unique_face_cell_a.assign(unique_cell_a.begin(), unique_cell_a.end());
  mb.d_unique_face_cell_b.assign(unique_cell_b.begin(), unique_cell_b.end());
  mb.d_unique_face_local_a.assign(unique_local_a.begin(), unique_local_a.end());
  mb.d_unique_face_local_b.assign(unique_local_b.begin(), unique_local_b.end());
  mb.d_unique_face_bc_tag.assign(unique_bc_tag.begin(), unique_bc_tag.end());
  mb.d_boundary_face_cell.assign(boundary_cell.begin(), boundary_cell.end());
  mb.d_boundary_face_local.assign(boundary_local.begin(), boundary_local.end());
  mb.d_boundary_face_bc_tag.assign(boundary_bc_tag.begin(),
                                   boundary_bc_tag.end());
  mb.d_face_adj_csr_offsets.assign(mb.face_adj_csr_offsets.begin(),
                                   mb.face_adj_csr_offsets.end());
  mb.d_face_adj_csr_indices.assign(mb.face_adj_csr_indices.begin(),
                                   mb.face_adj_csr_indices.end());
}

}  // namespace

Mesh create_cone_shell_mesh(const tenryu::core::Config& cfg,
                            tenryu::core::State& state) {
  const auto& mesh_cfg = cfg.mesh;
  TENRYU_ASSERT(cfg.main.dim == 2 && cfg.main.dimension == "2D_RZ",
                "cone_shell requires Main.dimension='2D_RZ'");
  TENRYU_ASSERT(mesh_cfg.topology_scheme ==
                    tenryu::core::TopologyScheme::CONE_SHELL_SPINE,
                "cone_shell requires cone_shell_spine topology");

  const tenryu::core::ConeShellAlongWallSpec ladder_spec{
      mesh_cfg.cone_shell_wall_length,
      mesh_cfg.cone_shell_wall_thickness,
      mesh_cfg.cone_shell_n_cells,
      mesh_cfg.cone_shell_n_growth,
      mesh_cfg.cone_shell_tip_size_factor,
      mesh_cfg.cone_shell_base_size_factor,
      mesh_cfg.cone_shell_tip_hold,
      mesh_cfg.cone_shell_grading_length,
      mesh_cfg.cone_shell_l_ratio_max,
      mesh_cfg.cone_shell_tip_rotation_length,
      mesh_cfg.cone_shell_base_rotation_length,
      mesh_cfg.cone_shell_base_cut == "planar",
  };
  const tenryu::core::ConeShellAlongWallLadder along =
      tenryu::core::build_cone_shell_along_wall_ladder(ladder_spec);
  TENRYU_ASSERT(along.counts_valid,
                "cone_shell along-wall ladder count is invalid");
  TENRYU_ASSERT(along.adjacent_ratio_valid,
                "cone_shell adjacent along-wall ratio exceeds limit after 8 "
                "increments; max=" +
                    std::to_string(along.max_adjacent_ratio));
  TENRYU_ASSERT(along.q.size() ==
                    static_cast<std::size_t>(mesh_cfg.nr + 1),
                "cone_shell builder/generator N_l mismatch");
  TENRYU_ASSERT(along.q.front() == 0.0 &&
                    along.q.back() == mesh_cfg.cone_shell_wall_length,
                "cone_shell along-wall endpoints must be pinned exactly");
  for (std::size_t i = 0; i + 1U < along.q.size(); ++i) {
    TENRYU_ASSERT(along.q[i + 1U] > along.q[i],
                  "cone_shell along-wall ladder must be strictly increasing");
  }

  std::ostringstream ratio_message;
  ratio_message << "[cone-shell] N_l=" << mesh_cfg.nr
                << " max_adjacent_l_ratio="
                << std::setprecision(std::numeric_limits<double>::max_digits10)
                << along.max_adjacent_ratio;
  tenryu::core::log_info(ratio_message.str());

  const std::vector<double> normal =
      tenryu::core::build_cone_shell_normal_ladder(
          mesh_cfg.cone_shell_wall_thickness,
          mesh_cfg.cone_shell_n_cells,
          mesh_cfg.cone_shell_n_growth);
  const double half_thickness = 0.5 * mesh_cfg.cone_shell_wall_thickness;
  TENRYU_ASSERT(normal.size() ==
                    static_cast<std::size_t>(mesh_cfg.cone_shell_n_cells + 1),
                "cone_shell through-wall ladder count mismatch");
  TENRYU_ASSERT(normal.front() == -half_thickness &&
                    normal.back() == half_thickness &&
                    normal[static_cast<std::size_t>(
                        mesh_cfg.cone_shell_n_cells / 2)] == 0.0,
                "cone_shell through-wall endpoints and midpoint must be pinned "
                "exactly");
  for (std::size_t j = 0; j + 1U < normal.size(); ++j) {
    TENRYU_ASSERT(normal[j + 1U] > normal[j],
                  "cone_shell through-wall ladder must be strictly increasing");
  }

  std::vector<double> j_normal(normal.size(), 0.0);
  if (mesh_cfg.cone_shell_axis_sign > 0) {
    for (std::size_t j = 0; j < normal.size(); ++j) {
      j_normal[j] = normal[normal.size() - 1U - j];
    }
  } else {
    j_normal = normal;
  }

  const double alpha = mesh_cfg.cone_shell_alpha;
  constexpr double kHalfPi =
      1.5707963267948966192313216916397514420985846996876;
  TENRYU_ASSERT(std::isfinite(alpha) && alpha > 0.0 && alpha < kHalfPi,
                "cone_shell builder requires finite alpha in (0, pi/2)");
  const double sin_alpha = std::sin(alpha);
  const double cos_alpha = std::cos(alpha);
  const double tan_alpha = std::tan(alpha);
  const double sigma = static_cast<double>(mesh_cfg.cone_shell_axis_sign);
  const ConeShellPoint tau{sin_alpha, sigma * cos_alpha};
  const ConeShellPoint nu{cos_alpha, -sigma * sin_alpha};
  const double mid_tip_radius =
      mesh_cfg.cone_shell_tip_radius_kind == "inner_face"
          ? mesh_cfg.cone_shell_tip_radius + half_thickness / cos_alpha
          : mesh_cfg.cone_shell_tip_radius;
  const ConeShellPoint tip_center{mid_tip_radius,
                                  mesh_cfg.cone_shell_tip_z};
  double rotation_derivative_bound =
      1.875 / mesh_cfg.cone_shell_tip_rotation_length;
  if (mesh_cfg.cone_shell_base_cut == "planar") {
    rotation_derivative_bound = std::max(
        rotation_derivative_bound,
        1.875 / mesh_cfg.cone_shell_base_rotation_length);
  }
  const double analytic_margin =
      1.0 - half_thickness * tan_alpha * rotation_derivative_bound;
  TENRYU_ASSERT(analytic_margin >= 0.25,
                "cone_shell analytic wall-Jacobian margin must be >= 0.25; "
                "got " +
                    std::to_string(analytic_margin));

  Mesh mesh;
  mesh.corner_stride =
      tenryu::core::corner_stride_for_config(cfg);
  TENRYU_ASSERT(mesh.corner_stride == state.corner_stride,
                "Mesh/state corner stride mismatch");
  mesh.dim = 2;
  mesh.logical = LogicalMesh2D::ConeShell;
  mesh.polar_center_treatment = PolarCenterTreatment::Annular;
  mesh.topo.nr = mesh_cfg.nr;
  mesh.topo.nz = mesh_cfg.nz;
  mesh.topo.n_cells = mesh_topo_n_cells_total(mesh_cfg);
  mesh.topo.n_nodes = mesh_topo_n_nodes_total(mesh_cfg);
  mesh.cell_nverts.assign(static_cast<std::size_t>(mesh.topo.n_cells), 4U);
  TENRYU_ASSERT(state.x_r.size() ==
                    static_cast<std::size_t>(mesh.topo.n_nodes) &&
                    state.x_z.size() ==
                        static_cast<std::size_t>(mesh.topo.n_nodes),
                "cone_shell State node storage size mismatch");

  std::vector<double> host_x_r(static_cast<std::size_t>(mesh.topo.n_nodes),
                               0.0);
  std::vector<double> host_x_z(static_cast<std::size_t>(mesh.topo.n_nodes),
                               0.0);
  for (int i = 0; i <= mesh.topo.nr; ++i) {
    const double q = along.q[static_cast<std::size_t>(i)];
    const double end_rotation = cone_shell_end_rotation(
        q, mesh_cfg.cone_shell_tip_rotation_length);
    double rotation = end_rotation;
    if (mesh_cfg.cone_shell_base_cut == "planar") {
      rotation += cone_shell_base_rotation(
          q, mesh_cfg.cone_shell_wall_length,
          mesh_cfg.cone_shell_base_rotation_length);
    }
    for (int j = 0; j <= mesh.topo.nz; ++j) {
      const double n = j_normal[static_cast<std::size_t>(j)];
      const double s = q + n * tan_alpha * rotation;
      const double r = tip_center.r + s * tau.r + n * nu.r;
      const double z = tip_center.z + s * tau.z + n * nu.z;
      const int node = mesh.topo.node_index(i, j);
      host_x_r[static_cast<std::size_t>(node)] = r;
      host_x_z[static_cast<std::size_t>(node)] = z;
      TENRYU_ASSERT(std::isfinite(r) && std::isfinite(z) && r > 0.0,
                    "cone_shell emitted nodes must be finite with r > 0");
    }
  }

  const int wall_node_count = (mesh.topo.nr + 1) * (mesh.topo.nz + 1);
  const int along_station_count = mesh.topo.nr + 1;
  const int end_station_count = mesh.topo.nz + 1;
  const int outer_node_offset = wall_node_count;
  const int inner_node_offset =
      outer_node_offset +
      mesh_cfg.cone_shell_outer_vac_layers * along_station_count;
  const int end_node_offset =
      inner_node_offset +
      mesh_cfg.cone_shell_inner_vac_layers * along_station_count;
  const int outer_corner_layers = mesh_topo_cone_shell_corner_layers(
      mesh_cfg.cone_shell_outer_vac_layers,
      mesh_cfg.cone_shell_end_vac_layers);
  const int inner_corner_layers = mesh_topo_cone_shell_corner_layers(
      mesh_cfg.cone_shell_inner_vac_layers,
      mesh_cfg.cone_shell_end_vac_layers);
  const int outer_corner_node_offset =
      end_node_offset +
      mesh_cfg.cone_shell_end_vac_layers * end_station_count;
  const int inner_corner_node_offset =
      outer_corner_node_offset + outer_corner_layers;
  const int cavity_node_offset =
      inner_corner_node_offset + inner_corner_layers;
  const int exterior_node_offset =
      cavity_node_offset +
      mesh_cfg.cone_shell_cavity_cells * along_station_count;
  const int tip_fill_node_offset =
      exterior_node_offset +
      mesh_cfg.cone_shell_exterior_cells * along_station_count;
  const int tip_fill_layers = mesh_cfg.cone_shell_tip_fill_layers;
  const int tip_fill_west_segments =
      mesh_topo_cone_shell_tip_fill_west_segments(mesh_cfg);
  const int tip_fill_mid_segments =
      mesh_topo_cone_shell_tip_fill_mid_segments(mesh_cfg);
  const int tip_fill_east_segments =
      mesh_topo_cone_shell_tip_fill_east_segments(mesh_cfg);
  const int tip_fill_west_owned_node_count =
      (tip_fill_west_segments + 1) * tip_fill_layers - 1;
  const int tip_fill_mid_owned_node_count =
      (tip_fill_mid_segments - 1) * (tip_fill_layers - 1);
  const int tip_fill_east_owned_node_count =
      (tip_fill_east_segments + 1) * tip_fill_layers - 1;
  const int tip_fill_west_node_offset = tip_fill_node_offset;
  const int tip_fill_mid_node_offset =
      tip_fill_west_node_offset + tip_fill_west_owned_node_count;
  const int tip_fill_east_node_offset =
      tip_fill_mid_node_offset + tip_fill_mid_owned_node_count;
  TENRYU_ASSERT(
      tip_fill_east_node_offset + tip_fill_east_owned_node_count ==
          mesh.topo.n_nodes,
      "cone_shell C4b node offsets must cover the node pool "
      "exactly");

  const double h_n0 = along.normal_face_width;
  const std::vector<double> outer_distance = build_strip_distance_ladder(
      h_n0, mesh_cfg.cone_shell_outer_vac_first_factor,
      mesh_cfg.cone_shell_outer_vac_layers,
      mesh_cfg.cone_shell_outer_vac_growth);
  const std::vector<double> inner_distance = build_strip_distance_ladder(
      h_n0, mesh_cfg.cone_shell_inner_vac_first_factor,
      mesh_cfg.cone_shell_inner_vac_layers,
      mesh_cfg.cone_shell_inner_vac_growth);
  const std::vector<double> end_distance = build_strip_distance_ladder(
      h_n0, mesh_cfg.cone_shell_end_vac_first_factor,
      mesh_cfg.cone_shell_end_vac_layers,
      mesh_cfg.cone_shell_end_vac_growth);
  const double eps = std::numeric_limits<double>::epsilon();
  const auto assert_chi_identity = [eps, h_n0](
                                       const std::vector<double>& distance,
                                       const double factor) {
    const double chi = (distance[1] - distance[0]) / h_n0;
    TENRYU_ASSERT(std::abs(chi - factor) <= 4.0 * eps * std::abs(factor),
                  "cone_shell strip chi identity gate failed");
  };
  assert_chi_identity(outer_distance,
                      mesh_cfg.cone_shell_outer_vac_first_factor);
  assert_chi_identity(inner_distance,
                      mesh_cfg.cone_shell_inner_vac_first_factor);
  assert_chi_identity(end_distance,
                      mesh_cfg.cone_shell_end_vac_first_factor);

  std::vector<ConeShellPoint> through_direction(
      static_cast<std::size_t>(along_station_count));
  for (int station = 0; station < along_station_count; ++station) {
    const double q = along.q[static_cast<std::size_t>(station)];
    const double tip_rotation = cone_shell_end_rotation(
        q, mesh_cfg.cone_shell_tip_rotation_length);
    double rotation = tip_rotation;
    if (mesh_cfg.cone_shell_base_cut == "planar") {
      rotation += cone_shell_base_rotation(
          q, mesh_cfg.cone_shell_wall_length,
          mesh_cfg.cone_shell_base_rotation_length);
    }
    if (rotation == 0.0) {
      through_direction[static_cast<std::size_t>(station)] = nu;
    } else {
      const ConeShellPoint raw{
          nu.r + tan_alpha * rotation * tau.r,
          nu.z + tan_alpha * rotation * tau.z,
      };
      const double norm = std::hypot(raw.r, raw.z);
      TENRYU_ASSERT(std::isfinite(norm) && norm > 0.0,
                    "cone_shell through-wall direction norm must be positive");
      through_direction[static_cast<std::size_t>(station)] =
          {raw.r / norm, raw.z / norm};
    }
  }

  const int outer_face_j =
      cone_shell_face_j(mesh_cfg, ConeShellWallFace::Outer);
  const int inner_face_j =
      cone_shell_face_j(mesh_cfg, ConeShellWallFace::Inner);
  for (int station = 0; station < along_station_count; ++station) {
    const int outer_wall_node = mesh.topo.node_index(station, outer_face_j);
    const int inner_wall_node = mesh.topo.node_index(station, inner_face_j);
    TENRYU_ASSERT(strip_node_id(outer_node_offset, outer_wall_node, 0,
                                station, along_station_count) ==
                      outer_wall_node,
                  "cone_shell OUTER_NF layer-0 node must be wall-owned");
    TENRYU_ASSERT(strip_node_id(inner_node_offset, inner_wall_node, 0,
                                station, along_station_count) ==
                      inner_wall_node,
                  "cone_shell INNER_NF layer-0 node must be wall-owned");
  }
  for (int station = 0; station < end_station_count; ++station) {
    const int tip_wall_node = mesh.topo.node_index(0, station);
    TENRYU_ASSERT(strip_node_id(end_node_offset, tip_wall_node, 0, station,
                                end_station_count) == tip_wall_node,
                  "cone_shell END_NF layer-0 node must be wall-owned");
  }

  for (int layer = 1; layer <= mesh_cfg.cone_shell_outer_vac_layers;
       ++layer) {
    const double distance = outer_distance[static_cast<std::size_t>(layer)];
    for (int station = 0; station < along_station_count; ++station) {
      const int wall_node = mesh.topo.node_index(station, outer_face_j);
      const int node = strip_node_id(outer_node_offset, wall_node, layer,
                                     station, along_station_count);
      const ConeShellPoint direction =
          through_direction[static_cast<std::size_t>(station)];
      if (direction.r == nu.r && direction.z == nu.z) {
        host_x_r[static_cast<std::size_t>(node)] =
            host_x_r[static_cast<std::size_t>(wall_node)] + distance * nu.r;
        host_x_z[static_cast<std::size_t>(node)] =
            host_x_z[static_cast<std::size_t>(wall_node)] + distance * nu.z;
      } else {
        host_x_r[static_cast<std::size_t>(node)] =
            host_x_r[static_cast<std::size_t>(wall_node)] +
            distance * direction.r;
        host_x_z[static_cast<std::size_t>(node)] =
            host_x_z[static_cast<std::size_t>(wall_node)] +
            distance * direction.z;
      }
      TENRYU_ASSERT(
          std::isfinite(host_x_r[static_cast<std::size_t>(node)]) &&
              std::isfinite(host_x_z[static_cast<std::size_t>(node)]) &&
              host_x_r[static_cast<std::size_t>(node)] > 0.0,
          "cone_shell OUTER_NF nodes must be finite with r > 0");
    }
  }
  for (int layer = 1; layer <= mesh_cfg.cone_shell_inner_vac_layers;
       ++layer) {
    const double distance = inner_distance[static_cast<std::size_t>(layer)];
    for (int station = 0; station < along_station_count; ++station) {
      const int wall_node = mesh.topo.node_index(station, inner_face_j);
      const int node = strip_node_id(inner_node_offset, wall_node, layer,
                                     station, along_station_count);
      const ConeShellPoint direction =
          through_direction[static_cast<std::size_t>(station)];
      if (direction.r == nu.r && direction.z == nu.z) {
        host_x_r[static_cast<std::size_t>(node)] =
            host_x_r[static_cast<std::size_t>(wall_node)] - distance * nu.r;
        host_x_z[static_cast<std::size_t>(node)] =
            host_x_z[static_cast<std::size_t>(wall_node)] - distance * nu.z;
      } else {
        host_x_r[static_cast<std::size_t>(node)] =
            host_x_r[static_cast<std::size_t>(wall_node)] -
            distance * direction.r;
        host_x_z[static_cast<std::size_t>(node)] =
            host_x_z[static_cast<std::size_t>(wall_node)] -
            distance * direction.z;
      }
      TENRYU_ASSERT(
          std::isfinite(host_x_r[static_cast<std::size_t>(node)]) &&
              std::isfinite(host_x_z[static_cast<std::size_t>(node)]) &&
              host_x_r[static_cast<std::size_t>(node)] > 0.0,
          "cone_shell INNER_NF nodes must be finite with r > 0");
    }
  }
  for (int layer = 1; layer <= mesh_cfg.cone_shell_end_vac_layers; ++layer) {
    const double distance = end_distance[static_cast<std::size_t>(layer)];
    for (int station = 0; station < end_station_count; ++station) {
      const int wall_node = mesh.topo.node_index(0, station);
      const int node = strip_node_id(end_node_offset, wall_node, layer,
                                     station, end_station_count);
      host_x_r[static_cast<std::size_t>(node)] =
          host_x_r[static_cast<std::size_t>(wall_node)];
      host_x_z[static_cast<std::size_t>(node)] =
          host_x_z[static_cast<std::size_t>(wall_node)] - sigma * distance;
      TENRYU_ASSERT(
          std::isfinite(host_x_r[static_cast<std::size_t>(node)]) &&
              std::isfinite(host_x_z[static_cast<std::size_t>(node)]) &&
              host_x_r[static_cast<std::size_t>(node)] > 0.0,
          "cone_shell END_NF nodes must be finite with r > 0");
    }
  }

  const ConeShellPoint end_normal{0.0, -sigma};
  const ConeShellPoint tip_through_direction = through_direction.front();
  const ConeShellPoint inner_normal{-tip_through_direction.r,
                                    -tip_through_direction.z};
  const int outer_point = mesh.topo.node_index(0, outer_face_j);
  const int inner_point = mesh.topo.node_index(0, inner_face_j);

  const auto emit_corner_nodes =
      [&host_x_r, &host_x_z](
          const char* name, const int ring_count, const int point,
          const int corner_node_offset, const ConeShellPoint& normal_a,
          const ConeShellPoint& normal_b,
          const std::vector<double>& distance_a,
          const std::vector<double>& distance_b,
          std::vector<int> a, std::vector<int> b) {
        ConeShellCornerTemplate corner;
        corner.ring_count = ring_count;
        corner.point = point;
        corner.a = std::move(a);
        corner.b = std::move(b);
        corner.c.assign(static_cast<std::size_t>(ring_count + 1), point);

        const double matrix_det = determinant(normal_a, normal_b);
        constexpr double kRadiansToDegrees =
            57.295779513082320876798154814105170332405472466564;
        const double included_angle_degrees =
            std::asin(std::clamp(std::abs(matrix_det), 0.0, 1.0)) *
            kRadiansToDegrees;
        TENRYU_ASSERT(
            std::abs(matrix_det) >= 0.2,
            std::string("cone_shell ") + name +
                " normal matrix conditioning failed: included_angle_degrees=" +
                std::to_string(included_angle_degrees));

        const ConeShellPoint point_coordinate{
            host_x_r[static_cast<std::size_t>(point)],
            host_x_z[static_cast<std::size_t>(point)],
        };
        double previous_radius = 0.0;
        for (int layer = 1; layer <= ring_count; ++layer) {
          const double d_a = distance_a[static_cast<std::size_t>(layer)];
          const double d_b = distance_b[static_cast<std::size_t>(layer)];
          const ConeShellPoint delta{
              (d_a * normal_b.z - normal_a.z * d_b) / matrix_det,
              (normal_a.r * d_b - d_a * normal_b.r) / matrix_det,
          };
          const int node = corner_node_offset + layer - 1;
          corner.c[static_cast<std::size_t>(layer)] = node;
          host_x_r[static_cast<std::size_t>(node)] =
              point_coordinate.r + delta.r;
          host_x_z[static_cast<std::size_t>(node)] =
              point_coordinate.z + delta.z;
          TENRYU_ASSERT(
              std::isfinite(host_x_r[static_cast<std::size_t>(node)]) &&
                  std::isfinite(host_x_z[static_cast<std::size_t>(node)]) &&
                  host_x_r[static_cast<std::size_t>(node)] > 0.0,
              std::string("cone_shell ") + name +
                  " nodes must be finite with r > 0");
          TENRYU_ASSERT(
              dot(normal_a, delta) > 0.0 && dot(normal_b, delta) > 0.0,
              std::string("cone_shell ") + name +
                  " C_k must lie strictly inside the vacuum sector");
          const double radius = std::hypot(delta.r, delta.z);
          TENRYU_ASSERT(
              layer == 1 || radius > previous_radius,
              std::string("cone_shell ") + name +
                  " C_k offsets must be strictly nested");
          previous_radius = radius;
        }
        return corner;
      };

  std::vector<int> outer_a(
      static_cast<std::size_t>(outer_corner_layers + 1), outer_point);
  std::vector<int> outer_b(
      static_cast<std::size_t>(outer_corner_layers + 1), outer_point);
  for (int layer = 1; layer <= outer_corner_layers; ++layer) {
    outer_a[static_cast<std::size_t>(layer)] = strip_node_id(
        outer_node_offset, outer_point, layer, 0, along_station_count);
    outer_b[static_cast<std::size_t>(layer)] = strip_node_id(
        end_node_offset, outer_point, layer, outer_face_j, end_station_count);
  }
  const ConeShellCornerTemplate outer_corner = emit_corner_nodes(
      "TIP_CORNER_O", outer_corner_layers, outer_point,
      outer_corner_node_offset, tip_through_direction, end_normal,
      outer_distance, end_distance,
      std::move(outer_a), std::move(outer_b));

  std::vector<int> inner_a(
      static_cast<std::size_t>(inner_corner_layers + 1), inner_point);
  std::vector<int> inner_b(
      static_cast<std::size_t>(inner_corner_layers + 1), inner_point);
  for (int layer = 1; layer <= inner_corner_layers; ++layer) {
    inner_a[static_cast<std::size_t>(layer)] = strip_node_id(
        end_node_offset, inner_point, layer, inner_face_j, end_station_count);
    inner_b[static_cast<std::size_t>(layer)] = strip_node_id(
        inner_node_offset, inner_point, layer, 0, along_station_count);
  }
  const ConeShellCornerTemplate inner_corner = emit_corner_nodes(
      "TIP_CORNER_I", inner_corner_layers, inner_point,
      inner_corner_node_offset, end_normal, inner_normal, end_distance,
      inner_distance, std::move(inner_a), std::move(inner_b));

  const double inner_h_start = strip_last_width(
      h_n0, mesh_cfg.cone_shell_inner_vac_first_factor,
      mesh_cfg.cone_shell_inner_vac_layers,
      mesh_cfg.cone_shell_inner_vac_growth);
  const double outer_h_start = strip_last_width(
      h_n0, mesh_cfg.cone_shell_outer_vac_first_factor,
      mesh_cfg.cone_shell_outer_vac_layers,
      mesh_cfg.cone_shell_outer_vac_growth);
  const auto cavity_node_id = [cavity_node_offset, along_station_count](
                                  const int layer, const int station) {
    TENRYU_ASSERT(layer >= 1,
                  "cone_shell CAVITY_CORE owned layer must be positive");
    return cavity_node_offset + (layer - 1) * along_station_count + station;
  };
  const auto exterior_node_id = [exterior_node_offset, along_station_count](
                                    const int layer, const int station) {
    TENRYU_ASSERT(layer >= 1,
                  "cone_shell EXTERIOR_CORE owned layer must be positive");
    return exterior_node_offset + (layer - 1) * along_station_count + station;
  };
  const bool farfield_station_uniform =
      mesh_cfg.cone_shell_farfield_target_measure == "station_uniform";
  const auto farfield_measure = [&](const int station, const double q) {
    if (farfield_station_uniform) {
      return static_cast<double>(station) /
             static_cast<double>(mesh.topo.nr);
    }
    return tenryu::core::cone_shell_along_wall_monitor_measure(
        q, ladder_spec);
  };
  const double z_base =
      mesh_cfg.cone_shell_tip_z +
      sigma * mesh_cfg.cone_shell_wall_length * cos_alpha;
  const int first_outer_wall_node = mesh.topo.node_index(0, outer_face_j);
  const int first_outer_near_node = strip_node_id(
      outer_node_offset, first_outer_wall_node,
      mesh_cfg.cone_shell_outer_vac_layers, 0, along_station_count);
  const double exterior_start_z =
      host_x_z[static_cast<std::size_t>(first_outer_near_node)];

  for (int station = 0; station < along_station_count; ++station) {
    const int wall_node = mesh.topo.node_index(station, inner_face_j);
    const int near_node = strip_node_id(
        inner_node_offset, wall_node,
        mesh_cfg.cone_shell_inner_vac_layers, station,
        along_station_count);
    const ConeShellPoint near{
        host_x_r[static_cast<std::size_t>(near_node)],
        host_x_z[static_cast<std::size_t>(near_node)],
    };
    const ConeShellPoint direction =
        through_direction[static_cast<std::size_t>(station)];
    const double phi = farfield_measure(
        station, along.q[static_cast<std::size_t>(station)]);
    const ConeShellPoint axis{
        +0.0,
        mesh_cfg.cone_shell_tip_z +
            phi * (z_base - mesh_cfg.cone_shell_tip_z),
    };
    // The width ladder spans the Hermite arclength (composite 8-point
    // Gauss-Legendre on 64 fixed subintervals); interior node parameters
    // solve S(u_k) = s_k by a fixed 64-iteration bisection.
    const double clearance = std::hypot(near.r, near.z - axis.z);
    TENRYU_ASSERT(std::isfinite(clearance) && clearance > 0.0,
                  "cone_shell CAVITY_CORE chord must be positive");
    const double lambda_near = std::min(clearance, 8.0 * inner_h_start);
    const ConeShellPoint tangent_near{
        -direction.r * lambda_near,
        -direction.z * lambda_near,
    };
    const ConeShellPoint tangent_axis{-clearance, 0.0};
    const ConeShellHermiteCurve cavity_curve{near, tangent_near, axis,
                                             tangent_axis};
    const std::array<double, kHermiteArcSubintervals + 1> arc_prefix =
        cone_shell_hermite_arc_prefix(cavity_curve);
    const double arc_length = arc_prefix[kHermiteArcSubintervals];
    TENRYU_ASSERT(std::isfinite(arc_length) &&
                      arc_length >= clearance * (1.0 - 1.0e-12),
                  "cone_shell CAVITY_CORE Hermite arclength must be at "
                  "least the chord");
    const std::vector<double> distance = build_mixed_ladder(
        arc_length, inner_h_start, mesh_cfg.cone_shell_cavity_cells,
        "CAVITY_CORE");
    double previous_u = 0.0;
    for (int layer = 1; layer <= mesh_cfg.cone_shell_cavity_cells; ++layer) {
      const int node = cavity_node_id(layer, station);
      ConeShellPoint point{+0.0, axis.z};
      if (layer != mesh_cfg.cone_shell_cavity_cells) {
        const double u = cone_shell_hermite_invert_arclength(
            cavity_curve, arc_prefix,
            distance[static_cast<std::size_t>(layer)], "CAVITY_CORE");
        TENRYU_ASSERT(u > previous_u,
                      "cone_shell CAVITY_CORE arclength parameters must be "
                      "strictly increasing");
        previous_u = u;
        point = cone_shell_hermite_eval(cavity_curve, u);
      }
      host_x_r[static_cast<std::size_t>(node)] = point.r;
      host_x_z[static_cast<std::size_t>(node)] = point.z;
      if (layer == 1) {
        const double first_step =
            std::hypot(point.r - near.r, point.z - near.z);
        TENRYU_ASSERT(
            first_step >= 0.5 * distance[1] &&
                first_step <= 1.5 * distance[1],
            "cone_shell CAVITY_CORE first Hermite segment must stay in "
            "the [0.5, 1.5] band of the first ladder width");
      }
      TENRYU_ASSERT(
          std::isfinite(host_x_r[static_cast<std::size_t>(node)]) &&
              std::isfinite(host_x_z[static_cast<std::size_t>(node)]) &&
              host_x_r[static_cast<std::size_t>(node)] >= 0.0,
          "cone_shell CAVITY_CORE nodes must be finite with r >= 0");
    }
  }

  double previous_exterior_target_z = 0.0;
  for (int station = 0; station < along_station_count; ++station) {
    const double q = along.q[static_cast<std::size_t>(station)];
    const int wall_node = mesh.topo.node_index(station, outer_face_j);
    const int near_node = strip_node_id(
        outer_node_offset, wall_node,
        mesh_cfg.cone_shell_outer_vac_layers, station,
        along_station_count);
    const ConeShellPoint near{
        host_x_r[static_cast<std::size_t>(near_node)],
        host_x_z[static_cast<std::size_t>(near_node)],
    };
    const ConeShellPoint direction =
        through_direction[static_cast<std::size_t>(station)];
    const double phi = farfield_measure(station, q);
    double target_z =
        exterior_start_z + phi * (z_base - exterior_start_z);
    if (station == 0) {
      target_z = near.z;
    } else if (station == mesh.topo.nr) {
      target_z = z_base;
    }
    const ConeShellPoint target{mesh_cfg.box_r_max, target_z};
    if (station > 0) {
      TENRYU_ASSERT(
          sigma * (target.z - previous_exterior_target_z) > 0.0,
          "cone_shell EXTERIOR_CORE RMax targets must be strictly monotone");
    }
    previous_exterior_target_z = target.z;
    const double chord =
        std::hypot(target.r - near.r, target.z - near.z);
    TENRYU_ASSERT(std::isfinite(chord) && chord > 0.0,
                  "cone_shell EXTERIOR_CORE chord must be positive");
    const double lambda_near = std::min(chord, 8.0 * outer_h_start);
    const ConeShellPoint tangent_near{
        direction.r * lambda_near,
        direction.z * lambda_near,
    };
    const ConeShellPoint tangent_target{+chord, 0.0};
    const ConeShellHermiteCurve exterior_curve{near, tangent_near, target,
                                               tangent_target};
    const std::array<double, kHermiteArcSubintervals + 1> arc_prefix =
        cone_shell_hermite_arc_prefix(exterior_curve);
    const double arc_length = arc_prefix[kHermiteArcSubintervals];
    TENRYU_ASSERT(std::isfinite(arc_length) &&
                      arc_length >= chord * (1.0 - 1.0e-12),
                  "cone_shell EXTERIOR_CORE Hermite arclength must be at "
                  "least the chord");
    const std::vector<double> distance = build_mixed_ladder(
        arc_length, outer_h_start, mesh_cfg.cone_shell_exterior_cells,
        "EXTERIOR_CORE");
    double previous_u = 0.0;
    for (int layer = 1; layer <= mesh_cfg.cone_shell_exterior_cells; ++layer) {
      const int node = exterior_node_id(layer, station);
      ConeShellPoint point = target;
      if (layer != mesh_cfg.cone_shell_exterior_cells) {
        const double u = cone_shell_hermite_invert_arclength(
            exterior_curve, arc_prefix,
            distance[static_cast<std::size_t>(layer)], "EXTERIOR_CORE");
        TENRYU_ASSERT(u > previous_u,
                      "cone_shell EXTERIOR_CORE arclength parameters must "
                      "be strictly increasing");
        previous_u = u;
        point = cone_shell_hermite_eval(exterior_curve, u);
      }
      host_x_r[static_cast<std::size_t>(node)] = point.r;
      host_x_z[static_cast<std::size_t>(node)] = point.z;
      if (layer == 1) {
        const double first_step =
            std::hypot(point.r - near.r, point.z - near.z);
        TENRYU_ASSERT(
            first_step >= 0.5 * distance[1] &&
                first_step <= 1.5 * distance[1],
            "cone_shell EXTERIOR_CORE first Hermite segment must stay in "
            "the [0.5, 1.5] band of the first ladder width");
      }
      TENRYU_ASSERT(
          std::isfinite(host_x_r[static_cast<std::size_t>(node)]) &&
              std::isfinite(host_x_z[static_cast<std::size_t>(node)]) &&
              host_x_r[static_cast<std::size_t>(node)] > 0.0 &&
              host_x_r[static_cast<std::size_t>(node)] <=
                  mesh_cfg.box_r_max,
          "cone_shell EXTERIOR_CORE nodes must be finite with r in "
          "(0, box_r_max]");
    }
  }

  const double reference_length = mesh_cfg.cone_shell_wall_length;
  const double geometry_tolerance = 128.0 * eps * reference_length;
  const double tip_tolerance =
      128.0 * eps *
      std::max(mesh_cfg.cone_shell_wall_length,
               std::abs(mesh_cfg.cone_shell_tip_z));
  for (int j = 0; j <= mesh.topo.nz; ++j) {
    const int node = mesh.topo.node_index(0, j);
    TENRYU_ASSERT(
        std::abs(host_x_z[static_cast<std::size_t>(node)] -
                 mesh_cfg.cone_shell_tip_z) <= tip_tolerance,
        "cone_shell tip edge is not planar within the residual band");
  }
  const double base_tolerance =
      128.0 * eps *
      std::max(mesh_cfg.cone_shell_wall_length, std::abs(z_base));
  if (mesh_cfg.cone_shell_base_cut == "planar") {
    for (int j = 0; j <= mesh.topo.nz; ++j) {
      const int node = mesh.topo.node_index(mesh.topo.nr, j);
      TENRYU_ASSERT(
          std::abs(host_x_z[static_cast<std::size_t>(node)] - z_base) <=
              base_tolerance,
          "cone_shell base edge is not planar within the residual band");
    }
  }
  for (const auto& strip :
       {std::pair{outer_node_offset,
                  mesh_cfg.cone_shell_outer_vac_layers},
        std::pair{inner_node_offset,
                  mesh_cfg.cone_shell_inner_vac_layers}}) {
    for (int layer = 0; layer <= strip.second; ++layer) {
      const int tip_wall = mesh.topo.node_index(
          0, strip.first == outer_node_offset ? outer_face_j : inner_face_j);
      const int tip_node = strip_node_id(
          strip.first, tip_wall, layer, 0, along_station_count);
      TENRYU_ASSERT(
          std::abs(host_x_z[static_cast<std::size_t>(tip_node)] -
                   mesh_cfg.cone_shell_tip_z) <= tip_tolerance,
          "cone_shell tip-station strip node is outside the planar band");
      if (mesh_cfg.cone_shell_base_cut == "planar") {
        const int base_wall = mesh.topo.node_index(
            mesh.topo.nr,
            strip.first == outer_node_offset ? outer_face_j : inner_face_j);
        const int base_node = strip_node_id(
            strip.first, base_wall, layer, mesh.topo.nr,
            along_station_count);
        TENRYU_ASSERT(
            std::abs(host_x_z[static_cast<std::size_t>(base_node)] - z_base) <=
                base_tolerance,
            "cone_shell base-station strip node is outside the planar band");
      }
    }
  }
  for (int layer = 1; layer <= mesh_cfg.cone_shell_cavity_cells; ++layer) {
    const int tip_node = cavity_node_id(layer, 0);
    TENRYU_ASSERT(
        std::abs(host_x_z[static_cast<std::size_t>(tip_node)] -
                 mesh_cfg.cone_shell_tip_z) <= tip_tolerance,
        "cone_shell CAVITY_CORE tip side is outside the planar band");
    if (mesh_cfg.cone_shell_base_cut == "planar") {
      const int base_node = cavity_node_id(layer, mesh.topo.nr);
      TENRYU_ASSERT(
          std::abs(host_x_z[static_cast<std::size_t>(base_node)] - z_base) <=
              base_tolerance,
          "cone_shell CAVITY_CORE base side is outside the planar band");
    }
  }
  for (int layer = 1; layer <= mesh_cfg.cone_shell_exterior_cells; ++layer) {
    const int tip_node = exterior_node_id(layer, 0);
    TENRYU_ASSERT(
        std::abs(host_x_z[static_cast<std::size_t>(tip_node)] -
                 mesh_cfg.cone_shell_tip_z) <= tip_tolerance,
        "cone_shell EXTERIOR_CORE tip side is outside the planar band");
    if (mesh_cfg.cone_shell_base_cut == "planar") {
      const int base_node = exterior_node_id(layer, mesh.topo.nr);
      TENRYU_ASSERT(
          std::abs(host_x_z[static_cast<std::size_t>(base_node)] - z_base) <=
              base_tolerance,
          "cone_shell EXTERIOR_CORE base side is outside the planar band");
    }
  }
  for (int i = 0; i <= mesh.topo.nr; ++i) {
    for (const int j : {0, mesh.topo.nz}) {
      const int node = mesh.topo.node_index(i, j);
      const double dr = host_x_r[static_cast<std::size_t>(node)] - tip_center.r;
      const double dz = host_x_z[static_cast<std::size_t>(node)] - tip_center.z;
      const double n_geom = dr * nu.r + dz * nu.z;
      TENRYU_ASSERT(
          std::abs(n_geom - j_normal[static_cast<std::size_t>(j)]) <=
              geometry_tolerance,
          "cone_shell face normal-coordinate residual exceeds tolerance");
    }
  }

  constexpr double kTenDegrees =
      0.17453292519943295769236907684886127134428718885417;
  for (int i = 0; i <= mesh.topo.nr; ++i) {
    const double q = along.q[static_cast<std::size_t>(i)];
    const double end_rotation = cone_shell_end_rotation(
        q, mesh_cfg.cone_shell_tip_rotation_length);
    double rotation = end_rotation;
    if (mesh_cfg.cone_shell_base_cut == "planar") {
      rotation += cone_shell_base_rotation(
          q, mesh_cfg.cone_shell_wall_length,
          mesh_cfg.cone_shell_base_rotation_length);
    }
    const ConeShellPoint expected{
        nu.r + tan_alpha * rotation * tau.r,
        nu.z + tan_alpha * rotation * tau.z,
    };
    const double expected_norm = std::hypot(expected.r, expected.z);
    for (int j = 0; j < mesh.topo.nz; ++j) {
      const int node0 = mesh.topo.node_index(i, j);
      const int node1 = mesh.topo.node_index(i, j + 1);
      const ConeShellPoint edge{
          host_x_r[static_cast<std::size_t>(node1)] -
              host_x_r[static_cast<std::size_t>(node0)],
          host_x_z[static_cast<std::size_t>(node1)] -
              host_x_z[static_cast<std::size_t>(node0)],
      };
      const double edge_norm = std::hypot(edge.r, edge.z);
      const double direction_cosine = std::clamp(
          std::abs(edge.r * expected.r + edge.z * expected.z) /
              (edge_norm * expected_norm),
          0.0, 1.0);
      const double angle = std::acos(direction_cosine);
      TENRYU_ASSERT(
          angle <= kTenDegrees,
          rotation == 0.0
              ? "cone_shell through-wall angle leakage exceeds 10 degrees "
                "outside the rotation zone"
              : "cone_shell through-wall direction differs from the analytic "
                "rotation-zone direction by more than 10 degrees");
    }
  }

  for (int i = 0; i < mesh.topo.nr; ++i) {
    for (int j = 0; j < mesh.topo.nz; ++j) {
      const std::array<int, 4> nodes{{
          mesh.topo.node_index(i, j),
          mesh.topo.node_index(i + 1, j),
          mesh.topo.node_index(i + 1, j + 1),
          mesh.topo.node_index(i, j + 1),
      }};
      std::array<ConeShellPoint, 4> points{};
      for (int corner = 0; corner < 4; ++corner) {
        const int node = nodes[static_cast<std::size_t>(corner)];
        points[static_cast<std::size_t>(corner)] = {
            host_x_r[static_cast<std::size_t>(node)],
            host_x_z[static_cast<std::size_t>(node)],
        };
      }
      const ConeShellPoint d_i_bottom = subtract(points[1], points[0]);
      const ConeShellPoint d_i_top = subtract(points[2], points[3]);
      const ConeShellPoint d_j_left = subtract(points[3], points[0]);
      const ConeShellPoint d_j_right = subtract(points[2], points[1]);
      const std::array<double, 4> corner_jacobians{{
          determinant(d_i_bottom, d_j_left),
          determinant(d_i_bottom, d_j_right),
          determinant(d_i_top, d_j_right),
          determinant(d_i_top, d_j_left),
      }};
      for (const double jacobian : corner_jacobians) {
        TENRYU_ASSERT(jacobian > 0.0,
                      "cone_shell cell corner Jacobian must be positive");
      }
      TENRYU_ASSERT(exact_rz_volume(points) > 0.0,
                    "cone_shell exact RZ revolved cell volume must be positive");
    }
  }

  mesh.edge_tags.clear();
  mesh.topo.multiblock = mesh_topo_make_empty_multiblock_topology(mesh_cfg);
  auto& mb = *mesh.topo.multiblock;
  const int topology_stride = mesh.corner_stride;
  TENRYU_ASSERT(
      mb.block_count == kConeShellSpineBlockCount,
      "cone_shell C4b block count must match the named constant");
  mb.cell_block_id.assign(static_cast<std::size_t>(mesh.topo.n_cells), -1);
  mb.cell_id_stable.resize(static_cast<std::size_t>(mesh.topo.n_cells));
  mb.cell_orientation_sign.assign(
      static_cast<std::size_t>(mesh.topo.n_cells), 1);
  set_csr_offsets(mb.cell_node_csr_offsets, mesh.topo.n_cells,
                  topology_stride);
  mb.cell_node_csr_indices.assign(
      static_cast<std::size_t>(topology_stride * mesh.topo.n_cells), -1);

  int cell = 0;
  for (int i = 0; i < mesh.topo.nr; ++i) {
    for (int j = 0; j < mesh.topo.nz; ++j) {
      set_cell_nodes(mb, cell,
                     {{mesh.topo.node_index(i, j),
                       mesh.topo.node_index(i + 1, j),
                       mesh.topo.node_index(i + 1, j + 1),
                       mesh.topo.node_index(i, j + 1)}},
                     topology_stride);
      mb.cell_block_id[static_cast<std::size_t>(cell)] =
          static_cast<int>(ConeShellBlock::WALL);
      mb.cell_id_stable[static_cast<std::size_t>(cell)] = cell;
      ++cell;
    }
  }
  const int wall_cell_count = cell;

  const auto append_face_strip_cells =
      [&mb, &cell, &mesh, along_station_count, topology_stride](
          const int face_j, const int interior_offset, const int layers,
          const ConeShellBlock block) {
        for (int layer = 0; layer < layers; ++layer) {
          for (int station = 0; station < mesh.topo.nr; ++station) {
            const int wall_node0 = mesh.topo.node_index(station, face_j);
            const int wall_node1 = mesh.topo.node_index(station + 1, face_j);
            const int lower0 = strip_node_id(
                interior_offset, wall_node0, layer, station,
                along_station_count);
            const int lower1 = strip_node_id(
                interior_offset, wall_node1, layer, station + 1,
                along_station_count);
            const int upper0 = strip_node_id(
                interior_offset, wall_node0, layer + 1, station,
                along_station_count);
            const int upper1 = strip_node_id(
                interior_offset, wall_node1, layer + 1, station + 1,
                along_station_count);
            const std::array<int, 4> nodes =
                face_j == 0
                    ? std::array<int, 4>{{lower1, lower0, upper0, upper1}}
                    : std::array<int, 4>{{lower0, lower1, upper1, upper0}};
            set_cell_nodes(mb, cell, nodes, topology_stride);
            mb.cell_block_id[static_cast<std::size_t>(cell)] =
                static_cast<int>(block);
            mb.cell_id_stable[static_cast<std::size_t>(cell)] = cell;
            ++cell;
          }
        }
      };
  const int outer_cell_begin = cell;
  append_face_strip_cells(
      outer_face_j, outer_node_offset,
      mesh_cfg.cone_shell_outer_vac_layers, ConeShellBlock::OUTER_NF);
  const int inner_cell_begin = cell;
  append_face_strip_cells(
      inner_face_j, inner_node_offset,
      mesh_cfg.cone_shell_inner_vac_layers, ConeShellBlock::INNER_NF);

  const int end_cell_begin = cell;
  for (int layer = 0; layer < mesh_cfg.cone_shell_end_vac_layers; ++layer) {
    for (int station = 0; station < mesh.topo.nz; ++station) {
      const int wall_node0 = mesh.topo.node_index(0, station);
      const int wall_node1 = mesh.topo.node_index(0, station + 1);
      const int lower0 = strip_node_id(end_node_offset, wall_node0, layer,
                                       station, end_station_count);
      const int lower1 = strip_node_id(end_node_offset, wall_node1, layer,
                                       station + 1, end_station_count);
      const int upper0 = strip_node_id(end_node_offset, wall_node0, layer + 1,
                                       station, end_station_count);
      const int upper1 = strip_node_id(end_node_offset, wall_node1, layer + 1,
                                       station + 1, end_station_count);
      set_cell_nodes(mb, cell, {{lower0, lower1, upper1, upper0}},
                     topology_stride);
      mb.cell_block_id[static_cast<std::size_t>(cell)] =
          static_cast<int>(ConeShellBlock::END_NF);
      mb.cell_id_stable[static_cast<std::size_t>(cell)] = cell;
      ++cell;
    }
  }

  const auto append_corner_cells =
      [&mb, &cell, &host_x_r, &host_x_z, topology_stride](
          const ConeShellCornerTemplate& corner,
          const ConeShellBlock block) {
        const auto append_quad =
            [&mb, &cell, &host_x_r, &host_x_z, block, topology_stride](
                const std::array<int, 4>& unordered_nodes) {
              const std::array<int, 4> nodes = orient_quad_positive(
                  unordered_nodes, host_x_r, host_x_z);
              set_cell_nodes(mb, cell, nodes, topology_stride);
              mb.cell_block_id[static_cast<std::size_t>(cell)] =
                  static_cast<int>(block);
              mb.cell_id_stable[static_cast<std::size_t>(cell)] = cell;
              ++cell;
            };
        append_quad({{corner.point, corner.a[1], corner.c[1], corner.b[1]}});
        for (int layer = 1; layer < corner.ring_count; ++layer) {
          append_quad({{corner.a[static_cast<std::size_t>(layer)],
                        corner.a[static_cast<std::size_t>(layer + 1)],
                        corner.c[static_cast<std::size_t>(layer + 1)],
                        corner.c[static_cast<std::size_t>(layer)]}});
          append_quad({{corner.c[static_cast<std::size_t>(layer)],
                        corner.c[static_cast<std::size_t>(layer + 1)],
                        corner.b[static_cast<std::size_t>(layer + 1)],
                        corner.b[static_cast<std::size_t>(layer)]}});
        }
      };
  const int outer_corner_cell_begin = cell;
  append_corner_cells(outer_corner, ConeShellBlock::TIP_CORNER_O);
  const int inner_corner_cell_begin = cell;
  append_corner_cells(inner_corner, ConeShellBlock::TIP_CORNER_I);

  const auto cavity_layer_node =
      [&mesh, inner_face_j, inner_node_offset, along_station_count,
       &cavity_node_id, &mesh_cfg](const int layer, const int station) {
        if (layer == 0) {
          const int wall_node = mesh.topo.node_index(station, inner_face_j);
          return strip_node_id(
              inner_node_offset, wall_node,
              mesh_cfg.cone_shell_inner_vac_layers, station,
              along_station_count);
        }
        return cavity_node_id(layer, station);
      };
  const int cavity_cell_begin = cell;
  for (int layer = 0; layer < mesh_cfg.cone_shell_cavity_cells; ++layer) {
    for (int station = 0; station < mesh.topo.nr; ++station) {
      const std::array<int, 4> nodes = orient_quad_positive(
          {{cavity_layer_node(layer, station),
            cavity_layer_node(layer, station + 1),
            cavity_layer_node(layer + 1, station + 1),
            cavity_layer_node(layer + 1, station)}},
          host_x_r, host_x_z);
      set_cell_nodes(mb, cell, nodes, topology_stride);
      mb.cell_block_id[static_cast<std::size_t>(cell)] =
          static_cast<int>(ConeShellBlock::CAVITY_CORE);
      mb.cell_id_stable[static_cast<std::size_t>(cell)] = cell;
      if (layer + 1 == mesh_cfg.cone_shell_cavity_cells) {
        const int axis0 = cavity_layer_node(layer + 1, station);
        const int axis1 = cavity_layer_node(layer + 1, station + 1);
        const int off0 = cavity_layer_node(layer, station);
        const int off1 = cavity_layer_node(layer, station + 1);
        TENRYU_ASSERT(
            host_x_r[static_cast<std::size_t>(axis0)] == +0.0 &&
                host_x_r[static_cast<std::size_t>(axis1)] == +0.0 &&
                !std::signbit(host_x_r[static_cast<std::size_t>(axis0)]) &&
                !std::signbit(host_x_r[static_cast<std::size_t>(axis1)]) &&
                host_x_r[static_cast<std::size_t>(off0)] > 0.0 &&
                host_x_r[static_cast<std::size_t>(off1)] > 0.0,
            "cone_shell CAVITY_CORE axis cell must have two consecutive "
            "axis nodes and two strictly off-axis nodes");
      }
      ++cell;
    }
  }

  const auto exterior_layer_node =
      [&mesh, outer_face_j, outer_node_offset, along_station_count,
       &exterior_node_id, &mesh_cfg](const int layer, const int station) {
        if (layer == 0) {
          const int wall_node = mesh.topo.node_index(station, outer_face_j);
          return strip_node_id(
              outer_node_offset, wall_node,
              mesh_cfg.cone_shell_outer_vac_layers, station,
              along_station_count);
        }
        return exterior_node_id(layer, station);
      };
  const int exterior_cell_begin = cell;
  for (int layer = 0; layer < mesh_cfg.cone_shell_exterior_cells; ++layer) {
    for (int station = 0; station < mesh.topo.nr; ++station) {
      const std::array<int, 4> nodes = orient_quad_positive(
          {{exterior_layer_node(layer, station),
            exterior_layer_node(layer, station + 1),
            exterior_layer_node(layer + 1, station + 1),
            exterior_layer_node(layer + 1, station)}},
          host_x_r, host_x_z);
      set_cell_nodes(mb, cell, nodes, topology_stride);
      mb.cell_block_id[static_cast<std::size_t>(cell)] =
          static_cast<int>(ConeShellBlock::EXTERIOR_CORE);
      mb.cell_id_stable[static_cast<std::size_t>(cell)] = cell;
      ++cell;
    }
  }
  const int c4a_cell_end = cell;
  TENRYU_ASSERT(
      c4a_cell_end +
              mesh_topo_cone_shell_tip_fill_cell_count(mesh_cfg) ==
          mesh.topo.n_cells,
      "cone_shell C4b cell counts must cover the cell pool exactly");

  constexpr std::uint8_t kTipPortInnerCorner = 0U;
  constexpr std::uint8_t kTipPortEnd = 1U;
  constexpr std::uint8_t kTipPortOuterCorner = 2U;
  ConeShellTipPort& tip_port = mb.cone_shell_tip_port;
  const auto append_tip_port_node = [&tip_port](
                                        const int node,
                                        const std::uint8_t segment) {
    tip_port.node_ids.push_back(node);
    tip_port.segment.push_back(segment);
  };

  // Fixed C-shaped order: inner A_K-C_K-B_K, END_NF far edge from inner to
  // outer, then outer B_K-C_K-A_K. The full-depth corner templates consume
  // both strip side edges completely.
  for (int layer = mesh_cfg.cone_shell_inner_vac_layers;
       layer >= inner_corner.ring_count; --layer) {
    append_tip_port_node(
        strip_node_id(inner_node_offset, inner_point, layer, 0,
                      along_station_count),
        kTipPortInnerCorner);
  }
  append_tip_port_node(
      inner_corner.c[static_cast<std::size_t>(inner_corner.ring_count)],
      kTipPortInnerCorner);
  for (int layer = inner_corner.ring_count;
       layer <= mesh_cfg.cone_shell_end_vac_layers; ++layer) {
    append_tip_port_node(
        strip_node_id(end_node_offset, inner_point, layer, inner_face_j,
                      end_station_count),
        kTipPortInnerCorner);
  }

  const int end_station_step = outer_face_j > inner_face_j ? 1 : -1;
  for (int station = inner_face_j;; station += end_station_step) {
    append_tip_port_node(
        strip_node_id(end_node_offset, mesh.topo.node_index(0, station),
                      mesh_cfg.cone_shell_end_vac_layers, station,
                      end_station_count),
        kTipPortEnd);
    if (station == outer_face_j) {
      break;
    }
  }

  for (int layer = mesh_cfg.cone_shell_end_vac_layers;
       layer >= outer_corner.ring_count; --layer) {
    append_tip_port_node(
        strip_node_id(end_node_offset, outer_point, layer, outer_face_j,
                      end_station_count),
        kTipPortOuterCorner);
  }
  append_tip_port_node(
      outer_corner.c[static_cast<std::size_t>(outer_corner.ring_count)],
      kTipPortOuterCorner);
  for (int layer = outer_corner.ring_count;
       layer <= mesh_cfg.cone_shell_outer_vac_layers; ++layer) {
    append_tip_port_node(
        strip_node_id(outer_node_offset, outer_point, layer, 0,
                      along_station_count),
        kTipPortOuterCorner);
  }

  const std::size_t expected_inner_port_nodes = static_cast<std::size_t>(
      mesh_cfg.cone_shell_inner_vac_layers +
      mesh_cfg.cone_shell_end_vac_layers - 2 * inner_corner.ring_count + 3);
  const std::size_t expected_end_port_nodes =
      static_cast<std::size_t>(end_station_count);
  const std::size_t expected_outer_port_nodes = static_cast<std::size_t>(
      mesh_cfg.cone_shell_outer_vac_layers +
      mesh_cfg.cone_shell_end_vac_layers - 2 * outer_corner.ring_count + 3);
  TENRYU_ASSERT(
      tip_port.node_ids.size() == expected_inner_port_nodes +
                                      expected_end_port_nodes +
                                      expected_outer_port_nodes,
      "cone_shell TIP_PORT must contain every far-boundary segment node");

  const EdgeIncidenceMap c4a_incidence =
      build_edge_incidence(mb, c4a_cell_end);
  std::vector<int> tip_fill_west_north;
  tip_fill_west_north.reserve(
      static_cast<std::size_t>(tip_fill_west_segments + 1));
  for (int layer = mesh_cfg.cone_shell_cavity_cells; layer >= 0; --layer) {
    tip_fill_west_north.push_back(cavity_layer_node(layer, 0));
  }

  const int inner_a_node = inner_corner.b[static_cast<std::size_t>(
      inner_corner.ring_count)];
  const int inner_c_node = inner_corner.c[static_cast<std::size_t>(
      inner_corner.ring_count)];
  const int inner_b_node = inner_corner.a[static_cast<std::size_t>(
      inner_corner.ring_count)];
  const int outer_b_node = outer_corner.b[static_cast<std::size_t>(
      outer_corner.ring_count)];
  const int outer_c_node = outer_corner.c[static_cast<std::size_t>(
      outer_corner.ring_count)];
  const int outer_a_node = outer_corner.a[static_cast<std::size_t>(
      outer_corner.ring_count)];

  TENRYU_ASSERT(
      tip_fill_west_north.back() == inner_a_node,
      "cone_shell TIP_FILL_WEST north must end at corner A_K^I");

  std::vector<int> tip_fill_mid_north;
  tip_fill_mid_north.reserve(
      static_cast<std::size_t>(tip_fill_mid_segments + 1));
  const auto append_mid_north = [&tip_fill_mid_north](const int node) {
    if (tip_fill_mid_north.empty() || tip_fill_mid_north.back() != node) {
      tip_fill_mid_north.push_back(node);
    }
  };
  append_mid_north(inner_c_node);
  append_mid_north(inner_b_node);
  for (int station = inner_face_j;; station += end_station_step) {
    append_mid_north(strip_node_id(
        end_node_offset, mesh.topo.node_index(0, station),
        mesh_cfg.cone_shell_end_vac_layers, station, end_station_count));
    if (station == outer_face_j) {
      break;
    }
  }
  append_mid_north(outer_b_node);
  append_mid_north(outer_c_node);

  std::vector<int> tip_fill_east_north;
  tip_fill_east_north.reserve(
      static_cast<std::size_t>(tip_fill_east_segments + 1));
  tip_fill_east_north.push_back(outer_a_node);
  for (int layer = 1; layer <= mesh_cfg.cone_shell_exterior_cells; ++layer) {
    tip_fill_east_north.push_back(exterior_layer_node(layer, 0));
  }

  TENRYU_ASSERT(
      tip_fill_east_north.front() == exterior_layer_node(0, 0),
      "cone_shell TIP_FILL_EAST north must start at corner A_K^O");
  TENRYU_ASSERT(
      static_cast<int>(tip_fill_west_north.size()) ==
              tip_fill_west_segments + 1 &&
          static_cast<int>(tip_fill_mid_north.size()) ==
              tip_fill_mid_segments + 1 &&
          static_cast<int>(tip_fill_east_north.size()) ==
              tip_fill_east_segments + 1,
      "cone_shell TIP_FILL block north node counts must match topology");

  const double end_far_z =
      mesh_cfg.cone_shell_tip_z - sigma * end_distance.back();
  const auto assert_north =
      [&c4a_incidence, &host_x_r, &host_x_z, tip_tolerance](
          const std::vector<int>& nodes, const bool horizontal,
          const double expected_z, const char* name) {
        for (std::size_t index = 0; index < nodes.size(); ++index) {
          const int node = nodes[index];
          if (horizontal) {
            TENRYU_ASSERT(
                std::abs(host_x_z[static_cast<std::size_t>(node)] -
                         expected_z) <= tip_tolerance,
                std::string("cone_shell ") + name +
                    " north is outside the horizontal residual band");
          }
          if (index == 0U) {
            continue;
          }
          const int previous = nodes[index - 1U];
          TENRYU_ASSERT(
              host_x_r[static_cast<std::size_t>(node)] >
                  host_x_r[static_cast<std::size_t>(previous)],
              std::string("cone_shell ") + name +
                  " north radii must be strictly increasing");
          int node0 = previous;
          int node1 = node;
          if (node1 < node0) {
            std::swap(node0, node1);
          }
          const auto edge = c4a_incidence.find({node0, node1});
          TENRYU_ASSERT(
              edge != c4a_incidence.end() && edge->second.count == 1,
              std::string("cone_shell ") + name +
                  " north must reuse connected C4a boundary edges");
        }
      };
  assert_north(tip_fill_west_north, true, mesh_cfg.cone_shell_tip_z,
               "TIP_FILL_WEST");
  assert_north(tip_fill_mid_north, true, end_far_z, "TIP_FILL_MID");
  assert_north(tip_fill_east_north, false, 0.0, "TIP_FILL_EAST");

  const int north_axis_node = tip_fill_west_north.front();
  const int north_rmax_node = tip_fill_east_north.back();
  const double d1_r = host_x_r[static_cast<std::size_t>(inner_c_node)];
  const double d2_r = host_x_r[static_cast<std::size_t>(outer_c_node)];
  TENRYU_ASSERT(
      host_x_r[static_cast<std::size_t>(north_axis_node)] == +0.0 &&
          !std::signbit(
              host_x_r[static_cast<std::size_t>(north_axis_node)]) &&
          host_x_r[static_cast<std::size_t>(north_rmax_node)] ==
              mesh_cfg.box_r_max &&
          host_x_r[static_cast<std::size_t>(inner_a_node)] == d1_r &&
          host_x_r[static_cast<std::size_t>(outer_a_node)] == d2_r &&
          +0.0 < d1_r && d1_r < d2_r && d2_r < mesh_cfg.box_r_max,
      "cone_shell TIP_FILL block endpoint radii must satisfy "
      "0 < r(D1) < r(D2) < box_r_max");

  const auto build_bottom_r = [](const int segment_count,
                                 const double r_begin,
                                 const double r_end) {
    TENRYU_ASSERT(segment_count > 0,
                  "cone_shell TIP_FILL block must contain at least one cell");
    std::vector<double> bottom_r(
        static_cast<std::size_t>(segment_count + 1), 0.0);
    for (int index = 0; index <= segment_count; ++index) {
      double r =
          r_begin + (r_end - r_begin) *
                        (static_cast<double>(index) /
                         static_cast<double>(segment_count));
      if (index == 0) {
        r = r_begin;
      } else if (index == segment_count) {
        r = r_end;
      }
      bottom_r[static_cast<std::size_t>(index)] = r;
    }
    return bottom_r;
  };
  const std::vector<double> tip_fill_west_bottom_r = build_bottom_r(
      tip_fill_west_segments, +0.0, d1_r);
  const std::vector<double> tip_fill_mid_bottom_r = build_bottom_r(
      tip_fill_mid_segments, d1_r, d2_r);
  const std::vector<double> tip_fill_east_bottom_r = build_bottom_r(
      tip_fill_east_segments, d2_r, mesh_cfg.box_r_max);

  const double tip_box_z = mesh_cfg.cone_shell_axis_sign > 0
                               ? mesh_cfg.box_z_min
                               : mesh_cfg.box_z_max;
  const double h_lt = mesh_cfg.cone_shell_tip_size_factor * h_n0;
  const double tip_fill_depth =
      std::abs(tip_box_z - mesh_cfg.cone_shell_tip_z);
  const int expected_tip_fill_layers = static_cast<int>(std::clamp(
      std::ceil(tip_fill_depth / (2.0 * h_lt)), 6.0, 48.0));
  TENRYU_ASSERT(
      expected_tip_fill_layers == tip_fill_layers,
      "cone_shell TIP_FILL builder/generator uniform layer count mismatch");

  const auto tip_fill_west_layer_node =
      [tip_fill_west_node_offset, tip_fill_west_segments, tip_fill_layers,
       inner_c_node,
       &tip_fill_west_north](const int layer, const int index) {
        TENRYU_ASSERT(layer >= 0 && layer <= tip_fill_layers && index >= 0 &&
                          index <= tip_fill_west_segments,
                      "cone_shell TIP_FILL_WEST node index out of range");
        if (layer == 0) {
          return tip_fill_west_north[static_cast<std::size_t>(index)];
        }
        if (index < tip_fill_west_segments) {
          return tip_fill_west_node_offset +
                 (layer - 1) * tip_fill_west_segments + index;
        }
        if (layer == 1) {
          return inner_c_node;
        }
        return tip_fill_west_node_offset +
               tip_fill_west_segments * tip_fill_layers + layer - 2;
      };
  const auto tip_fill_mid_layer_node =
      [tip_fill_mid_node_offset, tip_fill_mid_segments, tip_fill_layers,
       &tip_fill_mid_north, &tip_fill_west_north,
       &tip_fill_west_layer_node](
          const int layer, const int index, const auto& east_layer_node) {
        TENRYU_ASSERT(layer >= 0 && layer < tip_fill_layers && index >= 0 &&
                          index <= tip_fill_mid_segments,
                      "cone_shell TIP_FILL_MID node index out of range");
        if (layer == 0) {
          return tip_fill_mid_north[static_cast<std::size_t>(index)];
        }
        if (index == 0) {
          return tip_fill_west_layer_node(
              layer + 1,
              static_cast<int>(tip_fill_west_north.size()) - 1);
        }
        if (index == tip_fill_mid_segments) {
          return east_layer_node(layer + 1, 0);
        }
        return tip_fill_mid_node_offset +
               (layer - 1) * (tip_fill_mid_segments - 1) + index - 1;
      };
  const auto tip_fill_east_layer_node =
      [tip_fill_east_node_offset, tip_fill_east_segments, tip_fill_layers,
       outer_c_node,
       &tip_fill_east_north](const int layer, const int index) {
        TENRYU_ASSERT(layer >= 0 && layer <= tip_fill_layers && index >= 0 &&
                          index <= tip_fill_east_segments,
                      "cone_shell TIP_FILL_EAST node index out of range");
        if (layer == 0) {
          return tip_fill_east_north[static_cast<std::size_t>(index)];
        }
        if (index > 0) {
          return tip_fill_east_node_offset +
                 (layer - 1) * tip_fill_east_segments + index - 1;
        }
        if (layer == 1) {
          return outer_c_node;
        }
        return tip_fill_east_node_offset +
               tip_fill_east_segments * tip_fill_layers + layer - 2;
      };

  const auto set_fill_node =
      [&host_x_r, &host_x_z, tip_box_z](
          const int node, const double north_r, const double north_z,
          const double bottom_r, const double t, const bool bottom) {
        if (bottom) {
          host_x_r[static_cast<std::size_t>(node)] = bottom_r;
          host_x_z[static_cast<std::size_t>(node)] = tip_box_z;
        } else {
          host_x_r[static_cast<std::size_t>(node)] =
              north_r + (bottom_r - north_r) * t;
          host_x_z[static_cast<std::size_t>(node)] =
              north_z + (tip_box_z - north_z) * t;
        }
        TENRYU_ASSERT(
            std::isfinite(host_x_r[static_cast<std::size_t>(node)]) &&
                std::isfinite(host_x_z[static_cast<std::size_t>(node)]) &&
                host_x_r[static_cast<std::size_t>(node)] >= 0.0,
            "cone_shell TIP_FILL nodes must be finite with r >= 0");
      };

  for (int index = 0; index < tip_fill_west_segments; ++index) {
    const int north_node =
        tip_fill_west_north[static_cast<std::size_t>(index)];
    const double north_r = host_x_r[static_cast<std::size_t>(north_node)];
    const double north_z = host_x_z[static_cast<std::size_t>(north_node)];
    for (int layer = 1; layer <= tip_fill_layers; ++layer) {
      const int node = tip_fill_west_layer_node(layer, index);
      const double t = static_cast<double>(layer) /
                       static_cast<double>(tip_fill_layers);
      set_fill_node(
          node, north_r, north_z,
          tip_fill_west_bottom_r[static_cast<std::size_t>(index)], t,
          layer == tip_fill_layers);
      if (index == 0) {
        host_x_r[static_cast<std::size_t>(node)] = +0.0;
      }
    }
  }
  for (int layer = 2; layer <= tip_fill_layers; ++layer) {
    const int node = tip_fill_west_layer_node(
        layer, tip_fill_west_segments);
    const double t = static_cast<double>(layer - 1) /
                     static_cast<double>(tip_fill_layers - 1);
    set_fill_node(
        node, d1_r, host_x_z[static_cast<std::size_t>(inner_c_node)], d1_r,
        t, layer == tip_fill_layers);
    host_x_r[static_cast<std::size_t>(node)] = d1_r;
  }

  for (int index = 1; index <= tip_fill_east_segments; ++index) {
    const int north_node =
        tip_fill_east_north[static_cast<std::size_t>(index)];
    const double north_r = host_x_r[static_cast<std::size_t>(north_node)];
    const double north_z = host_x_z[static_cast<std::size_t>(north_node)];
    for (int layer = 1; layer <= tip_fill_layers; ++layer) {
      const int node = tip_fill_east_layer_node(layer, index);
      const double t = static_cast<double>(layer) /
                       static_cast<double>(tip_fill_layers);
      set_fill_node(
          node, north_r, north_z,
          tip_fill_east_bottom_r[static_cast<std::size_t>(index)], t,
          layer == tip_fill_layers);
      if (index == tip_fill_east_segments) {
        host_x_r[static_cast<std::size_t>(node)] = mesh_cfg.box_r_max;
      }
    }
  }
  for (int layer = 2; layer <= tip_fill_layers; ++layer) {
    const int node = tip_fill_east_layer_node(layer, 0);
    const double t = static_cast<double>(layer - 1) /
                     static_cast<double>(tip_fill_layers - 1);
    set_fill_node(
        node, d2_r, host_x_z[static_cast<std::size_t>(outer_c_node)], d2_r,
        t, layer == tip_fill_layers);
    host_x_r[static_cast<std::size_t>(node)] = d2_r;
  }

  for (int index = 1; index < tip_fill_mid_segments; ++index) {
    const int north_node =
        tip_fill_mid_north[static_cast<std::size_t>(index)];
    const double north_r = host_x_r[static_cast<std::size_t>(north_node)];
    const double north_z = host_x_z[static_cast<std::size_t>(north_node)];
    for (int layer = 1; layer < tip_fill_layers; ++layer) {
      const int node = tip_fill_mid_layer_node(
          layer, index, tip_fill_east_layer_node);
      const double t = static_cast<double>(layer) /
                       static_cast<double>(tip_fill_layers - 1);
      set_fill_node(
          node, north_r, north_z,
          tip_fill_mid_bottom_r[static_cast<std::size_t>(index)], t,
          layer + 1 == tip_fill_layers);
    }
  }

  for (int layer = 0; layer < tip_fill_layers; ++layer) {
    TENRYU_ASSERT(
        tip_fill_mid_layer_node(layer, 0, tip_fill_east_layer_node) ==
                tip_fill_west_layer_node(layer + 1,
                                         tip_fill_west_segments) &&
            tip_fill_mid_layer_node(layer, tip_fill_mid_segments,
                                    tip_fill_east_layer_node) ==
                tip_fill_east_layer_node(layer + 1, 0),
        "cone_shell TIP_FILL_MID side columns must reuse WEST/EAST node ids");
  }

  const auto append_tip_fill_block =
      [&mb, &cell, &host_x_r, &host_x_z, topology_stride](
          const int layer_count, const int segment_count,
          const ConeShellBlock block, const auto& layer_node) {
        for (int layer = 0; layer < layer_count; ++layer) {
          for (int segment = 0; segment < segment_count; ++segment) {
            const std::array<int, 4> nodes = orient_quad_positive(
                {{layer_node(layer, segment),
                  layer_node(layer, segment + 1),
                  layer_node(layer + 1, segment + 1),
                  layer_node(layer + 1, segment)}},
                host_x_r, host_x_z);
            set_cell_nodes(mb, cell, nodes, topology_stride);
            mb.cell_block_id[static_cast<std::size_t>(cell)] =
                static_cast<int>(block);
            mb.cell_id_stable[static_cast<std::size_t>(cell)] = cell;
            ++cell;
          }
        }
      };
  const int tip_fill_west_cell_begin = cell;
  append_tip_fill_block(tip_fill_layers, tip_fill_west_segments,
                        ConeShellBlock::TIP_FILL_WEST,
                        tip_fill_west_layer_node);
  const int tip_fill_mid_cell_begin = cell;
  const auto mid_layer_node =
      [&tip_fill_mid_layer_node, &tip_fill_east_layer_node](
          const int layer, const int index) {
        return tip_fill_mid_layer_node(layer, index,
                                       tip_fill_east_layer_node);
      };
  append_tip_fill_block(tip_fill_layers - 1, tip_fill_mid_segments,
                        ConeShellBlock::TIP_FILL_MID, mid_layer_node);
  const int tip_fill_east_cell_begin = cell;
  append_tip_fill_block(tip_fill_layers, tip_fill_east_segments,
                        ConeShellBlock::TIP_FILL_EAST,
                        tip_fill_east_layer_node);
  TENRYU_ASSERT(
      tip_fill_west_cell_begin == c4a_cell_end &&
          tip_fill_west_cell_begin < tip_fill_mid_cell_begin &&
          tip_fill_mid_cell_begin < tip_fill_east_cell_begin &&
          cell == mesh.topo.n_cells,
      "cone_shell TIP_FILL cell blocks must cover the final cell pool");

  const auto assert_face_strip_layer0_identity =
      [&mb, &mesh](const int cell_begin, const int face_j,
                   const char* message) {
        for (int station = 0; station < mesh.topo.nr; ++station) {
          const int offset = mb.cell_node_csr_offsets[static_cast<std::size_t>(
              cell_begin + station)];
          const int wall_node0 = mesh.topo.node_index(station, face_j);
          const int wall_node1 = mesh.topo.node_index(station + 1, face_j);
          TENRYU_ASSERT(
              mb.cell_node_csr_indices[static_cast<std::size_t>(
                  offset + (face_j == 0 ? 1 : 0))] == wall_node0 &&
                  mb.cell_node_csr_indices[static_cast<std::size_t>(
                      offset + (face_j == 0 ? 0 : 1))] == wall_node1,
              message);
        }
      };
  assert_face_strip_layer0_identity(
      outer_cell_begin, outer_face_j,
      "cone_shell OUTER_NF connectivity must reuse wall layer-0 nodes");
  assert_face_strip_layer0_identity(
      inner_cell_begin, inner_face_j,
      "cone_shell INNER_NF connectivity must reuse wall layer-0 nodes");
  for (int station = 0; station < mesh.topo.nz; ++station) {
    const int offset = mb.cell_node_csr_offsets[static_cast<std::size_t>(
        end_cell_begin + station)];
    TENRYU_ASSERT(
        mb.cell_node_csr_indices[static_cast<std::size_t>(offset)] ==
                mesh.topo.node_index(0, station) &&
            mb.cell_node_csr_indices[static_cast<std::size_t>(offset + 1)] ==
                mesh.topo.node_index(0, station + 1),
        "cone_shell END_NF connectivity must reuse wall layer-0 nodes");
  }

  for (int near_field_cell = wall_cell_count;
       near_field_cell < mesh.topo.n_cells; ++near_field_cell) {
    const int offset =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(near_field_cell)];
    std::array<ConeShellPoint, 4> points{};
    for (int corner = 0; corner < 4; ++corner) {
      const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(
          offset + corner)];
      points[static_cast<std::size_t>(corner)] = {
          host_x_r[static_cast<std::size_t>(node)],
          host_x_z[static_cast<std::size_t>(node)],
      };
    }
    const ConeShellPoint d_i_bottom = subtract(points[1], points[0]);
    const ConeShellPoint d_i_top = subtract(points[2], points[3]);
    const ConeShellPoint d_j_left = subtract(points[3], points[0]);
    const ConeShellPoint d_j_right = subtract(points[2], points[1]);
    const std::array<double, 4> corner_jacobians{{
        determinant(d_i_bottom, d_j_left),
        determinant(d_i_bottom, d_j_right),
        determinant(d_i_top, d_j_right),
        determinant(d_i_top, d_j_left),
    }};
    for (const double jacobian : corner_jacobians) {
      TENRYU_ASSERT(jacobian > 0.0,
                    "cone_shell near-field cell corner Jacobian must be "
                    "positive");
    }
    TENRYU_ASSERT(
        exact_rz_volume(points) > 0.0,
        "cone_shell near-field exact RZ revolved cell volume must be positive");
  }

  TENRYU_ASSERT(outer_corner_cell_begin < inner_corner_cell_begin &&
                    inner_corner_cell_begin < cavity_cell_begin &&
                    cavity_cell_begin < exterior_cell_begin &&
                    exterior_cell_begin < mesh.topo.n_cells,
                "cone_shell corner block cell ranges must be ordered");
  std::vector<std::pair<int, int>> expected_open_base_edges;
  if (mesh_cfg.cone_shell_base_cut == "wall_normal") {
    const auto push_chain_edge = [&expected_open_base_edges](const int a,
                                                             const int b) {
      expected_open_base_edges.emplace_back(std::min(a, b),
                                            std::max(a, b));
    };
    const int base_station = mesh.topo.nr;
    const int outer_wall_base =
        mesh.topo.node_index(base_station, outer_face_j);
    const int inner_wall_base =
        mesh.topo.node_index(base_station, inner_face_j);
    for (int layer = 0; layer < mesh_cfg.cone_shell_exterior_cells;
         ++layer) {
      const int a =
          layer == 0
              ? strip_node_id(outer_node_offset, outer_wall_base,
                              mesh_cfg.cone_shell_outer_vac_layers,
                              base_station, along_station_count)
              : exterior_node_id(layer, base_station);
      const int b = exterior_node_id(layer + 1, base_station);
      push_chain_edge(a, b);
    }
    for (int layer = 0; layer < mesh_cfg.cone_shell_outer_vac_layers;
         ++layer) {
      push_chain_edge(
          strip_node_id(outer_node_offset, outer_wall_base, layer,
                        base_station, along_station_count),
          strip_node_id(outer_node_offset, outer_wall_base, layer + 1,
                        base_station, along_station_count));
    }
    for (int j = 0; j < mesh.topo.nz; ++j) {
      push_chain_edge(mesh.topo.node_index(base_station, j),
                      mesh.topo.node_index(base_station, j + 1));
    }
    for (int layer = 0; layer < mesh_cfg.cone_shell_inner_vac_layers;
         ++layer) {
      push_chain_edge(
          strip_node_id(inner_node_offset, inner_wall_base, layer,
                        base_station, along_station_count),
          strip_node_id(inner_node_offset, inner_wall_base, layer + 1,
                        base_station, along_station_count));
    }
    for (int layer = 0; layer < mesh_cfg.cone_shell_cavity_cells;
         ++layer) {
      const int a =
          layer == 0
              ? strip_node_id(inner_node_offset, inner_wall_base,
                              mesh_cfg.cone_shell_inner_vac_layers,
                              base_station, along_station_count)
              : cavity_node_id(layer, base_station);
      const int b = cavity_node_id(layer + 1, base_station);
      push_chain_edge(a, b);
    }
    std::sort(expected_open_base_edges.begin(),
              expected_open_base_edges.end());
  }
  build_face_adjacency_and_boundary_tags(
      mesh.topo, mb, topology_stride, tip_port, mesh_cfg, host_x_r, host_x_z,
      expected_open_base_edges);
  tenryu::hydro::build_unique_oriented_face_list(
      mb, mb.cell_id_stable, mb.unique_internal_faces, mb.boundary_faces,
      nullptr, topology_stride);
  const bool final_open_base = mesh_cfg.cone_shell_base_cut == "wall_normal";
  for (const UniqueOrientedFace& face : mb.boundary_faces) {
    const BoundaryKind kind = static_cast<BoundaryKind>(face.bc_tag);
    const bool box_kind = kind == BoundaryKind::RectRInnerAxis ||
                          kind == BoundaryKind::RectROuter ||
                          kind == BoundaryKind::RectZBottom ||
                          kind == BoundaryKind::RectZTop;
    const bool open_base_kind =
        final_open_base && kind == BoundaryKind::SphericalOuterFree;
    TENRYU_ASSERT(box_kind || open_base_kind,
                  "cone_shell C4b final watertightness gate found a non-axis, "
                  "non-box boundary face");
  }
  upload_unique_face_payload(mb);
  mesh.multiblock_cell_node_csr_offsets.reset(
      mb.cell_node_csr_offsets.size());
  mesh.multiblock_cell_node_csr_offsets.copy_from_host(
      mb.cell_node_csr_offsets);
  mesh.multiblock_cell_node_csr_indices.reset(
      mb.cell_node_csr_indices.size());
  mesh.multiblock_cell_node_csr_indices.copy_from_host(
      mb.cell_node_csr_indices);
  mesh.multiblock_cell_orientation_sign_device.reset(
      mb.cell_orientation_sign.size());
  if (!mb.cell_orientation_sign.empty()) {
    mesh.multiblock_cell_orientation_sign_device.copy_from_host(
        mb.cell_orientation_sign);
  }
  mesh.multiblock_cell_nverts_device.reset(mesh.cell_nverts.size());
  if (!mesh.cell_nverts.empty()) {
    mesh.multiblock_cell_nverts_device.copy_from_host(mesh.cell_nverts);
  }
  ++mesh.topology_serial;
  const tenryu::hydro::ReverseCellNodeCSR reverse_csr =
      tenryu::hydro::build_reverse_cell_node_csr(
          mb, mesh.topo.n_nodes, nullptr, topology_stride);
  mesh.multiblock_reverse_csr_node_offsets.reset(
      reverse_csr.node_offsets.size());
  mesh.multiblock_reverse_csr_node_offsets.copy_from_host(
      reverse_csr.node_offsets);
  mesh.multiblock_reverse_csr_node_cells.reset(reverse_csr.node_cells.size());
  mesh.multiblock_reverse_csr_node_cells.copy_from_host(reverse_csr.node_cells);
  mesh.multiblock_reverse_csr_node_corners.reset(
      reverse_csr.node_corners.size());
  mesh.multiblock_reverse_csr_node_corners.copy_from_host(
      reverse_csr.node_corners);
  rebuild_multiblock_node_edge_csr(mesh);
  rebuild_multiblock_csw_line_topology(mesh);

  state.x_r.copy_from_host(host_x_r.data());
  state.x_z.copy_from_host(host_x_z.data());
  if (state.x_r_initial.size() == state.x_r.size()) {
    state.x_r_initial.copy_from_host(host_x_r.data());
  }
  if (state.x_z_initial.size() == state.x_z.size()) {
    state.x_z_initial.copy_from_host(host_x_z.data());
  }
  if (state.x_r_reference.size() == state.x_r.size()) {
    state.x_r_reference.copy_from_host(host_x_r.data());
  }
  if (state.x_z_reference.size() == state.x_z.size()) {
    state.x_z_reference.copy_from_host(host_x_z.data());
  }

  mesh.node_r = state.x_r.data();
  mesh.node_z = state.x_z.data();
  mesh.recompute_geometry();
  if (state.cell_is_void.size() ==
      static_cast<std::size_t>(mesh.topo.n_cells)) {
    std::fill(state.cell_is_void.begin(), state.cell_is_void.end(),
              static_cast<std::uint8_t>(0));
  }
  return mesh;
}

}  // namespace tenryu::mesh
