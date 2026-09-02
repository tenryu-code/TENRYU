#include "hydro/row_merge.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/hydro_multiblock_topology.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/mesh.hpp"
#include "mesh/poly_clip.cuh"
#include "mesh/rz_moments.cuh"

namespace tenryu::hydro {
namespace {

constexpr int kStorageSlots = mesh::kMeshTopoCellStorageSlotsMax;
constexpr int kStagingSlots = 2 * kStorageSlots;
constexpr double kTwoPi =
    6.283185307179586476925286766559005768394338798750211641949;

struct Polygon {
  int n = 0;
  std::array<int, kStagingSlots> nodes{};
  std::array<double, kStagingSlots> r{};
  std::array<double, kStagingSlots> z{};
};

struct PhysicalMoments {
  double volume = 0.0;
  double first_r = 0.0;
  double first_z = 0.0;
};

struct Subcell {
  std::array<double, 4> r{};
  std::array<double, 4> z{};
};

struct Triangle {
  std::array<double, 3> r{};
  std::array<double, 3> z{};
};

struct CornerPacket {
  double mass = 0.0;
  double momentum_r = 0.0;
  double momentum_z = 0.0;
  double electron_internal = 0.0;
  double ion_internal = 0.0;
  double kinetic_cons = 0.0;
};

struct CornerTransfer {
  std::size_t source_corner = 0;
  std::size_t destination_corner = 0;
  int source_node = -1;
  int destination_node = -1;
  double mass = 0.0;
};

struct PairPlan {
  int row_cell = -1;
  int partner = -1;
  int survivor = -1;
  int absorbed = -1;
  int third_cell = -1;
  int split_part_count = 0;
  bool keep_union = false;
  bool skip_merge = false;
  Polygon union_polygon;
  Polygon half_a;
  Polygon half_b;
  Polygon half_c;
  int chord_u = -1;
  int chord_v = -1;
  int chord2_u = -1;
  int chord2_v = -1;
};

RowMergeResult reject(RowMergeResult result,
                      const char* reason,
                      const int cell,
                      const long double defect =
                          std::numeric_limits<long double>::quiet_NaN(),
                      const long double scale =
                          std::numeric_limits<long double>::quiet_NaN()) {
  result.committed = false;
  result.first_bad_cell = cell;
  result.reject_reason = reason;
  std::ostringstream log;
  log << "[row-merge] REJECT reason=" << reason << " cell=" << cell;
  if (std::isfinite(defect) && std::isfinite(scale)) {
    log << std::setprecision(17) << " defect=" << defect
        << " scale=" << scale;
  }
  core::log_warning(log.str());
  return result;
}

int active_nverts(const std::vector<std::uint8_t>& cell_nverts,
                  const int cell) {
  return mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
}

Polygon cell_polygon(const int cell,
                     const std::vector<int>& csr_offsets,
                     const std::vector<int>& csr_indices,
                     const std::vector<std::uint8_t>& cell_nverts,
                     const std::vector<double>& x_r,
                     const std::vector<double>& x_z) {
  Polygon polygon;
  polygon.n = active_nverts(cell_nverts, cell);
  const int off = csr_offsets[static_cast<std::size_t>(cell)];
  for (int k = 0; k < polygon.n; ++k) {
    const int node = csr_indices[static_cast<std::size_t>(off + k)];
    polygon.nodes[static_cast<std::size_t>(k)] = node;
    polygon.r[static_cast<std::size_t>(k)] =
        x_r[static_cast<std::size_t>(node)];
    polygon.z[static_cast<std::size_t>(k)] =
        x_z[static_cast<std::size_t>(node)];
  }
  return polygon;
}

void polygon_centroid(const Polygon& polygon,
                      double& centroid_r,
                      double& centroid_z) {
  rz::rz_polygon_area_centroid_exact(
      polygon.r.data(), polygon.z.data(), polygon.n,
      &centroid_r, &centroid_z);
}

double polygon_centroid_radius(const Polygon& polygon) {
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  polygon_centroid(polygon, centroid_r, centroid_z);
  return std::hypot(centroid_r, centroid_z);
}

double polygon_centroid_z(const Polygon& polygon) {
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  polygon_centroid(polygon, centroid_r, centroid_z);
  return centroid_z;
}

double polygon_radial_extent(const Polygon& polygon) {
  double radius_min = std::numeric_limits<double>::infinity();
  double radius_max = -std::numeric_limits<double>::infinity();
  for (int k = 0; k < polygon.n; ++k) {
    const double radius =
        std::hypot(polygon.r[static_cast<std::size_t>(k)],
                   polygon.z[static_cast<std::size_t>(k)]);
    radius_min = std::min(radius_min, radius);
    radius_max = std::max(radius_max, radius);
  }
  return radius_max - radius_min;
}

PhysicalMoments physical_moments(const Polygon& polygon,
                                 const int orientation_sign) {
  const auto moments = mesh::moments::poly_rz_moments_fan(
      polygon.r.data(), polygon.z.data(), polygon.n);
  const double orientation = static_cast<double>(orientation_sign);
  PhysicalMoments result;
  result.volume = orientation * rz::rz_polygon_volume_exact(
                                    polygon.r.data(), polygon.z.data(),
                                    polygon.n);
  result.first_r = orientation * kTwoPi * moments.mrr;
  result.first_z = orientation * kTwoPi * moments.mrz;
  return result;
}

bool relative_identity(const double lhs, const double rhs) {
  const double scale = std::max(
      {std::abs(lhs), std::abs(rhs), std::numeric_limits<double>::min()});
  return std::abs(lhs - rhs) <=
         64.0 * std::numeric_limits<double>::epsilon() * scale;
}

enum class UnionStatus {
  kOk,
  kPairingFailed,
  kSharedEdgeAmbiguous,
  kOrientation,
};

bool point_on_segment(double ar,
                      double az,
                      double br,
                      double bz,
                      double pr,
                      double pz);

UnionStatus build_union_polygon(const Polygon& a,
                                const Polygon& b,
                                const int cell_a,
                                const int cell_b,
                                const ReverseCellNodeCSR& reverse_csr,
                                const std::vector<std::int8_t>& hydro_active,
                                const int orientation_sign,
                                const std::vector<double>& x_r,
                                const std::vector<double>& x_z,
                                Polygon& union_polygon,
                                int& shared_lo_out,
                                int& shared_hi_out) {
  std::array<std::uint8_t, kStagingSlots> shared_a{};
  std::array<std::uint8_t, kStagingSlots> shared_b{};
  int shared_count = 0;
  for (int ka = 0; ka < a.n; ++ka) {
    const int a0 = a.nodes[static_cast<std::size_t>(ka)];
    const int a1 = a.nodes[static_cast<std::size_t>((ka + 1) % a.n)];
    for (int kb = 0; kb < b.n; ++kb) {
      const int b0 = b.nodes[static_cast<std::size_t>(kb)];
      const int b1 = b.nodes[static_cast<std::size_t>((kb + 1) % b.n)];
      if (std::min(a0, a1) == std::min(b0, b1) &&
          std::max(a0, a1) == std::max(b0, b1)) {
        if (shared_a[static_cast<std::size_t>(ka)] != 0U ||
            shared_b[static_cast<std::size_t>(kb)] != 0U) {
          return UnionStatus::kSharedEdgeAmbiguous;
        }
        shared_a[static_cast<std::size_t>(ka)] = 1U;
        shared_b[static_cast<std::size_t>(kb)] = 1U;
        ++shared_count;
      }
    }
  }
  if (shared_count == 0 || shared_count == a.n || shared_count == b.n) {
    return UnionStatus::kPairingFailed;
  }

  const auto chain_start = [&](const auto& shared, const int n) {
    int start = -1;
    int starts = 0;
    for (int k = 0; k < n; ++k) {
      if (shared[static_cast<std::size_t>(k)] != 0U &&
          shared[static_cast<std::size_t>((k + n - 1) % n)] == 0U) {
        start = k;
        ++starts;
      }
    }
    return starts == 1 ? start : -1;
  };
  const int chain_start_a = chain_start(shared_a, a.n);
  const int chain_start_b = chain_start(shared_b, b.n);
  if (chain_start_a < 0 || chain_start_b < 0) {
    return UnionStatus::kPairingFailed;
  }
  for (int step = 0; step < shared_count; ++step) {
    if (shared_a[static_cast<std::size_t>(
            (chain_start_a + step) % a.n)] == 0U ||
        shared_b[static_cast<std::size_t>(
            (chain_start_b + step) % b.n)] == 0U) {
      return UnionStatus::kPairingFailed;
    }
  }
  bool chain_same_direction = true;
  bool chain_opposite_direction = true;
  for (int step = 0; step <= shared_count; ++step) {
    const int node_a = a.nodes[static_cast<std::size_t>(
        (chain_start_a + step) % a.n)];
    const int node_b_same = b.nodes[static_cast<std::size_t>(
        (chain_start_b + step) % b.n)];
    const int node_b_opposite = b.nodes[static_cast<std::size_t>(
        (chain_start_b + shared_count - step) % b.n)];
    chain_same_direction =
        chain_same_direction && node_a == node_b_same;
    chain_opposite_direction =
        chain_opposite_direction && node_a == node_b_opposite;
  }
  if (!chain_same_direction && !chain_opposite_direction) {
    return UnionStatus::kPairingFailed;
  }

  const int shared_begin = a.nodes[static_cast<std::size_t>(chain_start_a)];
  const int shared_end = a.nodes[static_cast<std::size_t>(
      (chain_start_a + shared_count) % a.n)];
  const int shared_lo = std::min(shared_begin, shared_end);
  const int shared_hi = std::max(shared_begin, shared_end);
  shared_lo_out = shared_lo;
  shared_hi_out = shared_hi;

  TENRYU_ASSERT(a.n + b.n - shared_count - 1 <= kStagingSlots,
                "row merge union exceeds staging capacity");

  std::vector<int> retained_chain_nodes;
  for (int step = 1; step < shared_count; ++step) {
    const int node = a.nodes[static_cast<std::size_t>(
        (chain_start_a + step) % a.n)];
    const int off = reverse_csr.node_offsets[static_cast<std::size_t>(node)];
    const int next =
        reverse_csr.node_offsets[static_cast<std::size_t>(node) + 1U];
    for (int p = off; p < next; ++p) {
      const int incident_cell =
          reverse_csr.node_cells[static_cast<std::size_t>(p)];
      if (incident_cell != cell_a && incident_cell != cell_b &&
          hydro_active[static_cast<std::size_t>(incident_cell)] != 0) {
        retained_chain_nodes.push_back(node);
        break;
      }
    }
  }

  std::array<int, kStagingSlots> graph_nodes{};
  std::array<std::array<int, 2>, kStagingSlots> adjacency{};
  std::array<int, kStagingSlots> degree{};
  graph_nodes.fill(-1);
  for (auto& adjacent : adjacency) {
    adjacent.fill(-1);
  }
  int graph_size = 0;

  const auto graph_index = [&](const int node) {
    for (int i = 0; i < graph_size; ++i) {
      if (graph_nodes[static_cast<std::size_t>(i)] == node) {
        return i;
      }
    }
    return -1;
  };
  const auto add_graph_node = [&](const int node) {
    const int existing = graph_index(node);
    if (existing >= 0) {
      return existing;
    }
    if (graph_size == kStagingSlots) {
      return -1;
    }
    const int inserted = graph_size++;
    graph_nodes[static_cast<std::size_t>(inserted)] = node;
    return inserted;
  };
  const auto add_boundary_edge = [&](const int node0,
                                     const int node1) {
    const int i0 = add_graph_node(node0);
    const int i1 = add_graph_node(node1);
    if (i0 < 0 || i1 < 0 || i0 == i1 ||
        degree[static_cast<std::size_t>(i0)] == 2 ||
        degree[static_cast<std::size_t>(i1)] == 2) {
      return false;
    }
    adjacency[static_cast<std::size_t>(i0)]
             [static_cast<std::size_t>(degree[static_cast<std::size_t>(i0)]++)] =
        i1;
    adjacency[static_cast<std::size_t>(i1)]
             [static_cast<std::size_t>(degree[static_cast<std::size_t>(i1)]++)] =
        i0;
    return true;
  };

  const std::array<const Polygon*, 2> polygons{{&a, &b}};
  const std::array<const std::array<std::uint8_t, kStagingSlots>*, 2>
      shared_edges{{&shared_a, &shared_b}};
  for (int polygon_index = 0; polygon_index < 2; ++polygon_index) {
    const Polygon& polygon = *polygons[static_cast<std::size_t>(polygon_index)];
    const auto& shared =
        *shared_edges[static_cast<std::size_t>(polygon_index)];
    for (int k = 0; k < polygon.n; ++k) {
      if (shared[static_cast<std::size_t>(k)] != 0U) {
        continue;
      }
      if (!add_boundary_edge(
              polygon.nodes[static_cast<std::size_t>(k)],
              polygon.nodes[static_cast<std::size_t>((k + 1) % polygon.n)])) {
        return UnionStatus::kSharedEdgeAmbiguous;
      }
    }
  }

  if (graph_size != a.n + b.n - 2 * shared_count) {
    return UnionStatus::kSharedEdgeAmbiguous;
  }
  int start = 0;
  for (int i = 0; i < graph_size; ++i) {
    if (degree[static_cast<std::size_t>(i)] != 2) {
      return UnionStatus::kSharedEdgeAmbiguous;
    }
    if (graph_nodes[static_cast<std::size_t>(i)] <
        graph_nodes[static_cast<std::size_t>(start)]) {
      start = i;
    }
  }
  if (graph_nodes[static_cast<std::size_t>(
          adjacency[static_cast<std::size_t>(start)][0])] >
      graph_nodes[static_cast<std::size_t>(
          adjacency[static_cast<std::size_t>(start)][1])]) {
    std::swap(adjacency[static_cast<std::size_t>(start)][0],
              adjacency[static_cast<std::size_t>(start)][1]);
  }

  const auto retain_flat_chain_nodes = [&](Polygon& candidate) {
    const int base_n = candidate.n;
    std::array<std::vector<int>, kStagingSlots> edge_insertions;
    for (const int node : retained_chain_nodes) {
      int containing_edge = -1;
      for (int k = 0; k < base_n; ++k) {
        const int next = (k + 1) % base_n;
        const double ar = candidate.r[static_cast<std::size_t>(k)];
        const double az = candidate.z[static_cast<std::size_t>(k)];
        const double br = candidate.r[static_cast<std::size_t>(next)];
        const double bz = candidate.z[static_cast<std::size_t>(next)];
        const double pr = x_r[static_cast<std::size_t>(node)];
        const double pz = x_z[static_cast<std::size_t>(node)];
        const double edge_r = br - ar;
        const double edge_z = bz - az;
        const double projection = (pr - ar) * edge_r + (pz - az) * edge_z;
        const double edge_length2 = edge_r * edge_r + edge_z * edge_z;
        if (point_on_segment(ar, az, br, bz, pr, pz) &&
            projection > 0.0 && projection < edge_length2) {
          if (containing_edge >= 0) {
            return false;
          }
          containing_edge = k;
        }
      }
      if (containing_edge < 0) {
        return false;
      }
      edge_insertions[static_cast<std::size_t>(containing_edge)]
          .push_back(node);
    }

    Polygon expanded;
    for (int k = 0; k < base_n; ++k) {
      if (expanded.n == kStagingSlots) {
        return false;
      }
      expanded.nodes[static_cast<std::size_t>(expanded.n)] =
          candidate.nodes[static_cast<std::size_t>(k)];
      expanded.r[static_cast<std::size_t>(expanded.n)] =
          candidate.r[static_cast<std::size_t>(k)];
      expanded.z[static_cast<std::size_t>(expanded.n)] =
          candidate.z[static_cast<std::size_t>(k)];
      ++expanded.n;

      auto& insertions = edge_insertions[static_cast<std::size_t>(k)];
      const double ar = candidate.r[static_cast<std::size_t>(k)];
      const double az = candidate.z[static_cast<std::size_t>(k)];
      const int next = (k + 1) % base_n;
      const double edge_r =
          candidate.r[static_cast<std::size_t>(next)] - ar;
      const double edge_z =
          candidate.z[static_cast<std::size_t>(next)] - az;
      std::sort(insertions.begin(), insertions.end(),
                [&](const int lhs, const int rhs) {
                  const double lhs_projection =
                      (x_r[static_cast<std::size_t>(lhs)] - ar) * edge_r +
                      (x_z[static_cast<std::size_t>(lhs)] - az) * edge_z;
                  const double rhs_projection =
                      (x_r[static_cast<std::size_t>(rhs)] - ar) * edge_r +
                      (x_z[static_cast<std::size_t>(rhs)] - az) * edge_z;
                  if (lhs_projection != rhs_projection) {
                    return lhs_projection < rhs_projection;
                  }
                  return lhs < rhs;
                });
      for (const int node : insertions) {
        if (expanded.n == kStagingSlots) {
          return false;
        }
        expanded.nodes[static_cast<std::size_t>(expanded.n)] = node;
        expanded.r[static_cast<std::size_t>(expanded.n)] =
            x_r[static_cast<std::size_t>(node)];
        expanded.z[static_cast<std::size_t>(expanded.n)] =
            x_z[static_cast<std::size_t>(node)];
        ++expanded.n;
      }
    }
    candidate = expanded;
    return true;
  };

  const auto walk = [&](const int first, Polygon& candidate) {
    std::array<std::uint8_t, kStagingSlots> seen{};
    candidate = Polygon{};
    candidate.nodes[0] = graph_nodes[static_cast<std::size_t>(start)];
    candidate.n = 1;
    seen[static_cast<std::size_t>(start)] = 1U;
    int previous = start;
    int current = first;
    while (current != start) {
      if (current < 0 || current >= graph_size ||
          seen[static_cast<std::size_t>(current)] != 0U ||
          candidate.n == graph_size) {
        return false;
      }
      candidate.nodes[static_cast<std::size_t>(candidate.n++)] =
          graph_nodes[static_cast<std::size_t>(current)];
      seen[static_cast<std::size_t>(current)] = 1U;
      const int next =
          adjacency[static_cast<std::size_t>(current)][0] == previous
              ? adjacency[static_cast<std::size_t>(current)][1]
              : adjacency[static_cast<std::size_t>(current)][0];
      previous = current;
      current = next;
    }
    if (candidate.n != graph_size) {
      return false;
    }
    for (int k = 0; k < candidate.n; ++k) {
      const int node = candidate.nodes[static_cast<std::size_t>(k)];
      candidate.r[static_cast<std::size_t>(k)] =
          x_r[static_cast<std::size_t>(node)];
      candidate.z[static_cast<std::size_t>(k)] =
          x_z[static_cast<std::size_t>(node)];
    }
    return retain_flat_chain_nodes(candidate);
  };

  for (int direction = 0; direction < 2; ++direction) {
    Polygon candidate;
    if (!walk(adjacency[static_cast<std::size_t>(start)]
                       [static_cast<std::size_t>(direction)],
              candidate)) {
      return UnionStatus::kSharedEdgeAmbiguous;
    }
    const double area2 = rz::rz_polygon_area2_exact(
        candidate.r.data(), candidate.z.data(), candidate.n);
    if (static_cast<double>(orientation_sign) * area2 > 0.0) {
      union_polygon = candidate;
      return UnionStatus::kOk;
    }
  }
  return UnionStatus::kOrientation;
}

bool point_on_segment(const double ar,
                      const double az,
                      const double br,
                      const double bz,
                      const double pr,
                      const double pz) {
  const auto orientation = mesh::moments::orient2d_deterministic(
      ar, az, br, bz, pr, pz);
  return orientation == mesh::moments::Orient2D::kDegenerate &&
         pr >= std::min(ar, br) && pr <= std::max(ar, br) &&
         pz >= std::min(az, bz) && pz <= std::max(az, bz);
}

bool segments_intersect(const double ar,
                        const double az,
                        const double br,
                        const double bz,
                        const double cr,
                        const double cz,
                        const double dr,
                        const double dz) {
  const auto abc = mesh::moments::orient2d_deterministic(
      ar, az, br, bz, cr, cz);
  const auto abd = mesh::moments::orient2d_deterministic(
      ar, az, br, bz, dr, dz);
  const auto cda = mesh::moments::orient2d_deterministic(
      cr, cz, dr, dz, ar, az);
  const auto cdb = mesh::moments::orient2d_deterministic(
      cr, cz, dr, dz, br, bz);
  if (abc == mesh::moments::Orient2D::kDegenerate &&
      point_on_segment(ar, az, br, bz, cr, cz)) {
    return true;
  }
  if (abd == mesh::moments::Orient2D::kDegenerate &&
      point_on_segment(ar, az, br, bz, dr, dz)) {
    return true;
  }
  if (cda == mesh::moments::Orient2D::kDegenerate &&
      point_on_segment(cr, cz, dr, dz, ar, az)) {
    return true;
  }
  if (cdb == mesh::moments::Orient2D::kDegenerate &&
      point_on_segment(cr, cz, dr, dz, br, bz)) {
    return true;
  }
  return static_cast<int>(abc) * static_cast<int>(abd) < 0 &&
         static_cast<int>(cda) * static_cast<int>(cdb) < 0;
}

bool polygon_is_simple(const Polygon& polygon) {
  for (int edge_a = 0; edge_a < polygon.n; ++edge_a) {
    const int edge_a_next = (edge_a + 1) % polygon.n;
    if (polygon.r[static_cast<std::size_t>(edge_a)] ==
            polygon.r[static_cast<std::size_t>(edge_a_next)] &&
        polygon.z[static_cast<std::size_t>(edge_a)] ==
            polygon.z[static_cast<std::size_t>(edge_a_next)]) {
      return false;
    }
    for (int edge_b = edge_a + 1; edge_b < polygon.n; ++edge_b) {
      const int edge_b_next = (edge_b + 1) % polygon.n;
      if (edge_a == edge_b_next || edge_a_next == edge_b) {
        continue;
      }
      if (segments_intersect(
              polygon.r[static_cast<std::size_t>(edge_a)],
              polygon.z[static_cast<std::size_t>(edge_a)],
              polygon.r[static_cast<std::size_t>(edge_a_next)],
              polygon.z[static_cast<std::size_t>(edge_a_next)],
              polygon.r[static_cast<std::size_t>(edge_b)],
              polygon.z[static_cast<std::size_t>(edge_b)],
              polygon.r[static_cast<std::size_t>(edge_b_next)],
              polygon.z[static_cast<std::size_t>(edge_b_next)])) {
        return false;
      }
    }
  }
  return true;
}

bool polygon_has_positive_corner_jacobians(
    const Polygon& polygon,
    const int orientation_sign) {
  for (int k = 0; k < polygon.n; ++k) {
    const int next = (k + 1) % polygon.n;
    const int previous = (k + polygon.n - 1) % polygon.n;
    const double jacobian = static_cast<double>(orientation_sign) *
        ((polygon.r[static_cast<std::size_t>(next)] -
          polygon.r[static_cast<std::size_t>(k)]) *
             (polygon.z[static_cast<std::size_t>(previous)] -
              polygon.z[static_cast<std::size_t>(k)]) -
         (polygon.z[static_cast<std::size_t>(next)] -
          polygon.z[static_cast<std::size_t>(k)]) *
             (polygon.r[static_cast<std::size_t>(previous)] -
              polygon.r[static_cast<std::size_t>(k)]));
    if (!(jacobian > 0.0)) {
      return false;
    }
  }
  return true;
}

Polygon boundary_half(const Polygon& union_polygon,
                      const int begin,
                      const int end) {
  Polygon half;
  int current = begin;
  while (true) {
    const std::size_t source = static_cast<std::size_t>(current);
    const std::size_t target = static_cast<std::size_t>(half.n++);
    half.nodes[target] = union_polygon.nodes[source];
    half.r[target] = union_polygon.r[source];
    half.z[target] = union_polygon.z[source];
    if (current == end) {
      break;
    }
    current = (current + 1) % union_polygon.n;
  }
  return half;
}

void rotate_to_smallest_node(Polygon& polygon) {
  int start = 0;
  for (int k = 1; k < polygon.n; ++k) {
    if (polygon.nodes[static_cast<std::size_t>(k)] <
        polygon.nodes[static_cast<std::size_t>(start)]) {
      start = k;
    }
  }
  Polygon rotated;
  rotated.n = polygon.n;
  for (int k = 0; k < polygon.n; ++k) {
    const int source = (start + k) % polygon.n;
    rotated.nodes[static_cast<std::size_t>(k)] =
        polygon.nodes[static_cast<std::size_t>(source)];
    rotated.r[static_cast<std::size_t>(k)] =
        polygon.r[static_cast<std::size_t>(source)];
    rotated.z[static_cast<std::size_t>(k)] =
        polygon.z[static_cast<std::size_t>(source)];
  }
  polygon = rotated;
}

bool polygon_contains_node(const Polygon& polygon, const int node) {
  for (int k = 0; k < polygon.n; ++k) {
    if (polygon.nodes[static_cast<std::size_t>(k)] == node) {
      return true;
    }
  }
  return false;
}

using Edge = std::pair<int, int>;

struct BoundaryEdgeInstall {
  int cell = -1;
  int local_face = -1;
  std::int8_t bc_tag = 0;
};

Edge undirected_edge(const int node0, const int node1) {
  return {std::min(node0, node1), std::max(node0, node1)};
}

int local_face_for_polygon_edge(const int nverts,
                                const int local_edge) {
  for (int local_face = 0; local_face < kStorageSlots; ++local_face) {
    if (mesh::mesh_topo_local_face_canonical_edge0(nverts, local_face) ==
        local_edge) {
      return local_face;
    }
  }
  return -1;
}

void append_polygon_edges(const Polygon& polygon,
                          std::vector<Edge>& edges) {
  for (int k = 0; k < polygon.n; ++k) {
    const int node0 = polygon.nodes[static_cast<std::size_t>(k)];
    const int node1 = polygon.nodes[
        static_cast<std::size_t>((k + 1) % polygon.n)];
    edges.push_back(undirected_edge(node0, node1));
  }
}

int directed_edge_sense(const Polygon& polygon,
                        const int node_u,
                        const int node_v) {
  int sense = 0;
  for (int k = 0; k < polygon.n; ++k) {
    const int node0 = polygon.nodes[static_cast<std::size_t>(k)];
    const int node1 = polygon.nodes[
        static_cast<std::size_t>((k + 1) % polygon.n)];
    if (node0 == node_u && node1 == node_v) {
      if (sense != 0) {
        return 0;
      }
      sense = 1;
    } else if (node0 == node_v && node1 == node_u) {
      if (sense != 0) {
        return 0;
      }
      sense = -1;
    }
  }
  return sense;
}

bool split_edge_multiset_matches(const Polygon& union_polygon,
                                 const Polygon& half_a,
                                 const Polygon& half_b,
                                 const int chord_u,
                                 const int chord_v) {
  if (chord_u == chord_v) {
    return false;
  }

  const int half_a_chord_sense =
      directed_edge_sense(half_a, chord_u, chord_v);
  const int half_b_chord_sense =
      directed_edge_sense(half_b, chord_u, chord_v);
  if (half_a_chord_sense == 0 ||
      half_b_chord_sense != -half_a_chord_sense) {
    return false;
  }

  std::vector<Edge> expected;
  expected.reserve(static_cast<std::size_t>(union_polygon.n) + 2U);
  append_polygon_edges(union_polygon, expected);
  expected.push_back(undirected_edge(chord_u, chord_v));
  expected.push_back(undirected_edge(chord_u, chord_v));

  std::vector<Edge> emitted;
  emitted.reserve(static_cast<std::size_t>(half_a.n + half_b.n));
  append_polygon_edges(half_a, emitted);
  append_polygon_edges(half_b, emitted);

  std::sort(expected.begin(), expected.end());
  std::sort(emitted.begin(), emitted.end());
  return emitted == expected;
}

bool three_way_split_edge_multiset_matches(
    const Polygon& union_polygon,
    const Polygon& half_a,
    const Polygon& half_b,
    const Polygon& half_c,
    const int chord_u,
    const int chord_v,
    const int chord2_u,
    const int chord2_v) {
  if (chord_u == chord_v || chord2_u == chord2_v ||
      chord_u == chord2_u || chord_u == chord2_v ||
      chord_v == chord2_u || chord_v == chord2_v ||
      undirected_edge(chord_u, chord_v) ==
          undirected_edge(chord2_u, chord2_v)) {
    return false;
  }

  const std::array<const Polygon*, 3> parts{{&half_a, &half_b, &half_c}};
  const std::array<Edge, 2> chords{{
      undirected_edge(chord_u, chord_v),
      undirected_edge(chord2_u, chord2_v),
  }};
  for (const Edge& chord : chords) {
    int occurrence_count = 0;
    int sense_sum = 0;
    for (const Polygon* part : parts) {
      const int sense = directed_edge_sense(*part, chord.first, chord.second);
      if (sense != 0) {
        ++occurrence_count;
        sense_sum += sense;
      }
    }
    if (occurrence_count != 2 || sense_sum != 0) {
      return false;
    }
  }

  std::vector<Edge> expected;
  expected.reserve(static_cast<std::size_t>(union_polygon.n) + 4U);
  append_polygon_edges(union_polygon, expected);
  for (const Edge& chord : chords) {
    expected.push_back(chord);
    expected.push_back(chord);
  }

  std::vector<Edge> emitted;
  emitted.reserve(
      static_cast<std::size_t>(half_a.n + half_b.n + half_c.n));
  append_polygon_edges(half_a, emitted);
  append_polygon_edges(half_b, emitted);
  append_polygon_edges(half_c, emitted);

  std::sort(expected.begin(), expected.end());
  std::sort(emitted.begin(), emitted.end());
  return emitted == expected;
}

int polygon_node_index(const Polygon& polygon, const int node) {
  for (int k = 0; k < polygon.n; ++k) {
    if (polygon.nodes[static_cast<std::size_t>(k)] == node) {
      return k;
    }
  }
  return -1;
}

bool build_three_way_parts(const Polygon& union_polygon,
                           const int chord1_begin,
                           const int chord1_end,
                           const int chord2_begin,
                           const int chord2_end,
                           std::array<Polygon, 3>& parts) {
  const Polygon first =
      boundary_half(union_polygon, chord1_begin, chord1_end);
  const Polygon second =
      boundary_half(union_polygon, chord1_end, chord1_begin);
  const int chord2_node0 =
      union_polygon.nodes[static_cast<std::size_t>(chord2_begin)];
  const int chord2_node1 =
      union_polygon.nodes[static_cast<std::size_t>(chord2_end)];
  const bool split_first =
      polygon_contains_node(first, chord2_node0) &&
      polygon_contains_node(first, chord2_node1);
  const bool split_second =
      polygon_contains_node(second, chord2_node0) &&
      polygon_contains_node(second, chord2_node1);
  if (split_first == split_second) {
    return false;
  }

  const Polygon& target = split_first ? first : second;
  const int local_begin = polygon_node_index(target, chord2_node0);
  const int local_end = polygon_node_index(target, chord2_node1);
  if (local_begin < 0 || local_end < 0 || local_begin == local_end) {
    return false;
  }
  const Polygon target_first = boundary_half(target, local_begin, local_end);
  const Polygon target_second = boundary_half(target, local_end, local_begin);
  if (split_first) {
    parts = {{target_first, target_second, second}};
  } else {
    parts = {{first, target_first, target_second}};
  }
  return true;
}

int choose_split(const Polygon& union_polygon,
                 const int shared_lo,
                 const int shared_hi,
                 const int survivor_smallest_node,
                 const int orientation_sign,
                 Polygon& half_a,
                 Polygon& half_b,
                 Polygon& half_c,
                 int& chord_u,
                 int& chord_v,
                 int& chord2_u,
                 int& chord2_v) {
  struct Candidate {
    int begin = -1;
    int end = -1;
    int imbalance = 0;
    int node_lo = -1;
    int node_hi = -1;
  };

  int shared_lo_index = -1;
  int shared_hi_index = -1;
  for (int k = 0; k < union_polygon.n; ++k) {
    const int node = union_polygon.nodes[static_cast<std::size_t>(k)];
    if (node == shared_lo) {
      shared_lo_index = k;
    }
    if (node == shared_hi) {
      shared_hi_index = k;
    }
  }
  TENRYU_ASSERT(shared_lo_index >= 0 && shared_hi_index >= 0,
                "row merge shared-boundary endpoint missing from union");
  half_c = Polygon{};
  chord2_u = -1;
  chord2_v = -1;

  std::vector<Candidate> candidates;
  int excl_distance = 0;
  int excl_shared_endpoint = 0;
  int excl_valence = 0;
  int excl_nonsimple = 0;
  for (int begin = 0; begin < union_polygon.n; ++begin) {
    for (int end = begin + 1; end < union_polygon.n; ++end) {
      const int boundary_distance = end - begin;
      if (boundary_distance == 1 ||
          boundary_distance == union_polygon.n - 1) {
        ++excl_distance;
        continue;
      }
      // Axis-row unions carry three collinear r=0 nodes. Every chord avoiding
      // the shared endpoints leaves the middle axis node as an exactly-zero
      // Jacobian corner, so single-shared-endpoint chords are the only feasible
      // splits and are legitimate new topologies. The geometric gates
      // (simple/area/volume/cornerJ) remain the safety authority.
      // the null chord would recreate the original crushed pair
      if ((begin == shared_lo_index && end == shared_hi_index) ||
          (begin == shared_hi_index && end == shared_lo_index)) {
        ++excl_shared_endpoint;
        continue;
      }
      const int nv_first = boundary_distance + 1;
      const int nv_second = union_polygon.n - boundary_distance + 1;
      if (nv_first < 3 || nv_first > 8 ||
          nv_second < 3 || nv_second > 8) {
        ++excl_valence;
        continue;
      }

      const Polygon first = boundary_half(union_polygon, begin, end);
      const Polygon second = boundary_half(union_polygon, end, begin);
      if (!polygon_is_simple(first) || !polygon_is_simple(second)) {
        ++excl_nonsimple;
        continue;
      }
      Candidate candidate;
      candidate.begin = begin;
      candidate.end = end;
      candidate.imbalance = std::abs(nv_first - nv_second);
      candidate.node_lo = std::min(
          union_polygon.nodes[static_cast<std::size_t>(begin)],
          union_polygon.nodes[static_cast<std::size_t>(end)]);
      candidate.node_hi = std::max(
          union_polygon.nodes[static_cast<std::size_t>(begin)],
          union_polygon.nodes[static_cast<std::size_t>(end)]);
      candidates.push_back(candidate);
    }
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
              if (lhs.imbalance != rhs.imbalance) {
                return lhs.imbalance < rhs.imbalance;
              }
              if (lhs.node_lo != rhs.node_lo) {
                return lhs.node_lo < rhs.node_lo;
              }
              return lhs.node_hi < rhs.node_hi;
            });

  int kill_area = 0;
  int kill_volume = 0;
  int kill_cornerj = 0;
  for (const Candidate& candidate : candidates) {
    Polygon first = boundary_half(
        union_polygon, candidate.begin, candidate.end);
    Polygon second = boundary_half(
        union_polygon, candidate.end, candidate.begin);
    const double first_area2 = rz::rz_polygon_area2_exact(
        first.r.data(), first.z.data(), first.n);
    const double second_area2 = rz::rz_polygon_area2_exact(
        second.r.data(), second.z.data(), second.n);
    const PhysicalMoments first_moments =
        physical_moments(first, orientation_sign);
    const PhysicalMoments second_moments =
        physical_moments(second, orientation_sign);
    if (static_cast<double>(orientation_sign) * first_area2 <= 0.0 ||
        static_cast<double>(orientation_sign) * second_area2 <= 0.0) {
      ++kill_area;
      continue;
    }
    if (!(first_moments.volume > 0.0) ||
        !(second_moments.volume > 0.0)) {
      ++kill_volume;
      continue;
    }
    if (!polygon_has_positive_corner_jacobians(first, orientation_sign) ||
        !polygon_has_positive_corner_jacobians(second, orientation_sign)) {
      ++kill_cornerj;
      continue;
    }

    rotate_to_smallest_node(first);
    rotate_to_smallest_node(second);
    if (polygon_contains_node(first, survivor_smallest_node)) {
      half_a = first;
      half_b = second;
    } else {
      TENRYU_ASSERT(polygon_contains_node(second, survivor_smallest_node),
                    "row merge survivor node missing from split halves");
      half_a = second;
      half_b = first;
    }
    chord_u = candidate.node_lo;
    chord_v = candidate.node_hi;
    return 2;
  }

  struct Chord {
    int begin = -1;
    int end = -1;
    int node_lo = -1;
    int node_hi = -1;
  };
  struct ThreeWayCandidate {
    std::array<Chord, 2> chords{};
    std::array<int, 4> chord_nodes{};
    int imbalance = 0;
  };

  std::vector<ThreeWayCandidate> three_way_candidates;
  int three_way_pairs_enumerated = 0;
  int three_way_excl_shared_endpoint = 0;
  int three_way_excl_crossing = 0;
  int three_way_excl_partition = 0;
  int three_way_excl_valence = 0;
  int three_way_excl_nonsimple = 0;
  int three_way_kill_area = 0;
  int three_way_kill_volume = 0;
  int three_way_kill_cornerj = 0;
  if (union_polygon.n >= 9 &&
      union_polygon.n <= 2 * kStorageSlots - 2) {
    std::vector<Chord> chords;
    for (int begin = 0; begin < union_polygon.n; ++begin) {
      for (int end = begin + 1; end < union_polygon.n; ++end) {
        const int boundary_distance = end - begin;
        if (boundary_distance == 1 ||
            boundary_distance == union_polygon.n - 1) {
          continue;
        }
        const int node0 =
            union_polygon.nodes[static_cast<std::size_t>(begin)];
        const int node1 =
            union_polygon.nodes[static_cast<std::size_t>(end)];
        if (undirected_edge(node0, node1) ==
            undirected_edge(shared_lo, shared_hi)) {
          ++three_way_excl_shared_endpoint;
          continue;
        }
        chords.push_back(Chord{
            begin, end, std::min(node0, node1), std::max(node0, node1)});
      }
    }

    for (std::size_t first_index = 0; first_index < chords.size();
         ++first_index) {
      for (std::size_t second_index = first_index + 1;
           second_index < chords.size(); ++second_index) {
        ++three_way_pairs_enumerated;
        Chord first = chords[first_index];
        Chord second = chords[second_index];
        if (first.begin == second.begin || first.begin == second.end ||
            first.end == second.begin || first.end == second.end) {
          ++three_way_excl_shared_endpoint;
          continue;
        }
        const bool second_begin_inside =
            second.begin > first.begin && second.begin < first.end;
        const bool second_end_inside =
            second.end > first.begin && second.end < first.end;
        if (second_begin_inside != second_end_inside) {
          ++three_way_excl_crossing;
          continue;
        }
        if (std::array<int, 2>{{second.node_lo, second.node_hi}} <
            std::array<int, 2>{{first.node_lo, first.node_hi}}) {
          std::swap(first, second);
        }
        std::array<Polygon, 3> parts{};
        if (!build_three_way_parts(
                union_polygon, first.begin, first.end,
                second.begin, second.end, parts)) {
          ++three_way_excl_partition;
          continue;
        }
        int min_part_nv = parts[0].n;
        int max_part_nv = parts[0].n;
        bool valid_valence = true;
        bool simple = true;
        for (const Polygon& part : parts) {
          min_part_nv = std::min(min_part_nv, part.n);
          max_part_nv = std::max(max_part_nv, part.n);
          valid_valence =
              valid_valence && part.n >= 3 && part.n <= kStorageSlots;
          simple = simple && polygon_is_simple(part);
        }
        if (!valid_valence) {
          ++three_way_excl_valence;
          continue;
        }
        if (!simple) {
          ++three_way_excl_nonsimple;
          continue;
        }
        ThreeWayCandidate candidate;
        candidate.chords = {{first, second}};
        candidate.chord_nodes = {{
            first.node_lo, first.node_hi,
            second.node_lo, second.node_hi,
        }};
        candidate.imbalance = max_part_nv - min_part_nv;
        three_way_candidates.push_back(candidate);
      }
    }
    std::sort(
        three_way_candidates.begin(), three_way_candidates.end(),
        [](const ThreeWayCandidate& lhs, const ThreeWayCandidate& rhs) {
          if (lhs.imbalance != rhs.imbalance) {
            return lhs.imbalance < rhs.imbalance;
          }
          return lhs.chord_nodes < rhs.chord_nodes;
        });

    for (const ThreeWayCandidate& candidate : three_way_candidates) {
      std::array<Polygon, 3> parts{};
      const Chord& first = candidate.chords[0];
      const Chord& second = candidate.chords[1];
      TENRYU_ASSERT(
          build_three_way_parts(
              union_polygon, first.begin, first.end,
              second.begin, second.end, parts),
          "row merge three-way split candidate lost its partition");

      bool positive_area = true;
      bool positive_volume = true;
      bool positive_corner_jacobians = true;
      for (const Polygon& part : parts) {
        const double area2 = rz::rz_polygon_area2_exact(
            part.r.data(), part.z.data(), part.n);
        const PhysicalMoments moments =
            physical_moments(part, orientation_sign);
        positive_area =
            positive_area &&
            static_cast<double>(orientation_sign) * area2 > 0.0;
        positive_volume = positive_volume && moments.volume > 0.0;
        positive_corner_jacobians =
            positive_corner_jacobians &&
            polygon_has_positive_corner_jacobians(part, orientation_sign);
      }
      if (!positive_area) {
        ++three_way_kill_area;
        continue;
      }
      if (!positive_volume) {
        ++three_way_kill_volume;
        continue;
      }
      if (!positive_corner_jacobians) {
        ++three_way_kill_cornerj;
        continue;
      }

      for (Polygon& part : parts) {
        rotate_to_smallest_node(part);
      }
      int survivor_part = -1;
      for (int part = 0; part < 3; ++part) {
        if (polygon_contains_node(
                parts[static_cast<std::size_t>(part)],
                survivor_smallest_node)) {
          survivor_part = part;
          break;
        }
      }
      TENRYU_ASSERT(survivor_part >= 0,
                    "row merge survivor node missing from split parts");
      half_a = parts[static_cast<std::size_t>(survivor_part)];
      int output_part = 0;
      for (int part = 0; part < 3; ++part) {
        if (part == survivor_part) {
          continue;
        }
        if (output_part == 0) {
          half_b = parts[static_cast<std::size_t>(part)];
        } else {
          half_c = parts[static_cast<std::size_t>(part)];
        }
        ++output_part;
      }
      chord_u = first.node_lo;
      chord_v = first.node_hi;
      chord2_u = second.node_lo;
      chord2_v = second.node_hi;
      return 3;
    }
  }
  std::ostringstream diag;
  diag << "[row-merge] split diag n=" << union_polygon.n
       << " candidates=" << candidates.size()
       << " excl(dist=" << excl_distance
       << ",null=" << excl_shared_endpoint
       << ",val=" << excl_valence
       << ",nonsimple=" << excl_nonsimple
       << ") kill(area=" << kill_area
       << ",vol=" << kill_volume
       << ",cornerJ=" << kill_cornerj << ")"
       << " three_way(pairs=" << three_way_pairs_enumerated
       << ",candidates=" << three_way_candidates.size()
       << ",excl(endpoint_or_null=" << three_way_excl_shared_endpoint
       << ",cross=" << three_way_excl_crossing
       << ",partition=" << three_way_excl_partition
       << ",val=" << three_way_excl_valence
       << ",nonsimple=" << three_way_excl_nonsimple
       << ") kill(area=" << three_way_kill_area
       << ",vol=" << three_way_kill_volume
       << ",cornerJ=" << three_way_kill_cornerj << ")) nodes=";
  diag << std::scientific << std::setprecision(6);
  for (int k = 0; k < union_polygon.n; ++k) {
    if (k != 0) {
      diag << ' ';
    }
    diag << union_polygon.nodes[static_cast<std::size_t>(k)] << ":("
         << union_polygon.r[static_cast<std::size_t>(k)] << ','
         << union_polygon.z[static_cast<std::size_t>(k)] << ')';
  }
  core::log_warning(diag.str());
  return 0;
}

Subcell make_subcell(const Polygon& polygon, const int vertex) {
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  polygon_centroid(polygon, centroid_r, centroid_z);
  const int next = (vertex + 1) % polygon.n;
  const int previous = (vertex + polygon.n - 1) % polygon.n;
  Subcell subcell;
  subcell.r = {{
      polygon.r[static_cast<std::size_t>(vertex)],
      0.5 * (polygon.r[static_cast<std::size_t>(vertex)] +
             polygon.r[static_cast<std::size_t>(next)]),
      centroid_r,
      0.5 * (polygon.r[static_cast<std::size_t>(previous)] +
             polygon.r[static_cast<std::size_t>(vertex)]),
  }};
  subcell.z = {{
      polygon.z[static_cast<std::size_t>(vertex)],
      0.5 * (polygon.z[static_cast<std::size_t>(vertex)] +
             polygon.z[static_cast<std::size_t>(next)]),
      centroid_z,
      0.5 * (polygon.z[static_cast<std::size_t>(previous)] +
             polygon.z[static_cast<std::size_t>(vertex)]),
  }};
  if (rz::rz_polygon_area2_exact(subcell.r.data(), subcell.z.data(), 4) < 0.0) {
    std::swap(subcell.r[1], subcell.r[3]);
    std::swap(subcell.z[1], subcell.z[3]);
  }
  return subcell;
}

std::array<Triangle, 2> split_subcell(const Subcell& subcell) {
  std::array<Triangle, 2> triangles{};
  triangles[0].r = {{subcell.r[0], subcell.r[1], subcell.r[2]}};
  triangles[0].z = {{subcell.z[0], subcell.z[1], subcell.z[2]}};
  triangles[1].r = {{subcell.r[0], subcell.r[2], subcell.r[3]}};
  triangles[1].z = {{subcell.z[0], subcell.z[2], subcell.z[3]}};
  for (Triangle& triangle : triangles) {
    if (rz::rz_polygon_area2_exact(
            triangle.r.data(), triangle.z.data(), 3) < 0.0) {
      std::swap(triangle.r[1], triangle.r[2]);
      std::swap(triangle.z[1], triangle.z[2]);
    }
  }
  return triangles;
}

double triangle_intersection_volume(const Triangle& subject,
                                    const Triangle& clipper) {
  const auto clipped = mesh::clip::clip_convex(
      subject.r.data(), subject.z.data(), 3,
      clipper.r.data(), clipper.z.data(), 3);
  if (clipped.n < 3 || clipped.status != mesh::clip::ClipStatus::kOk) {
    return 0.0;
  }
  return std::abs(rz::rz_polygon_volume_exact(
      clipped.r, clipped.z, clipped.n));
}

double intersection_volume(const Subcell& subject,
                           const Subcell& clipper) {
  const auto subject_triangles = split_subcell(subject);
  const auto clipper_triangles = split_subcell(clipper);
  double volume = 0.0;
  for (const Triangle& subject_triangle : subject_triangles) {
    for (const Triangle& clipper_triangle : clipper_triangles) {
      volume += triangle_intersection_volume(
          subject_triangle, clipper_triangle);
    }
  }
  return volume;
}

}  // namespace

bool row_merge_enabled() {
  const char* value = std::getenv("TENRYU_I1B_ROW_MERGE");
  return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

RowMergeResult apply_tier_row_merge(core::State& state,
                                    const core::Config& cfg,
                                    const int failing_cell) {
  RowMergeResult result;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (!row_merge_enabled() || state.mesh.dim != 2 ||
      !mesh::mesh_topo_polar_tier_family(cfg.mesh) ||
      !state.mesh.topo.multiblock.has_value() || failing_cell < 0 ||
      failing_cell >= n_cells) {
    return result;
  }

  const auto& mb = *state.mesh.topo.multiblock;
  if (n_cells <= 0 || n_nodes <= 0 ||
      !mesh::mesh_topo_polar_tier_family(cfg.mesh) ||
      mb.blocks.empty() ||
      mb.cell_block_id.size() != static_cast<std::size_t>(n_cells) ||
      mb.cell_orientation_sign.size() != static_cast<std::size_t>(n_cells) ||
      mb.cell_node_csr_offsets.size() != static_cast<std::size_t>(n_cells + 1) ||
      mb.face_adj_csr_offsets.size() != static_cast<std::size_t>(n_cells + 1) ||
      state.mesh.cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return result;
  }

  TENRYU_ASSERT(state.corner_stride == kStorageSlots &&
                    state.mesh.corner_stride == kStorageSlots,
                "row merge in-place polygon growth requires corner_stride == 8");

  const auto block_role = [&](const int cell) {
    const int block = mb.cell_block_id[static_cast<std::size_t>(cell)];
    TENRYU_ASSERT(block >= 0 &&
                      block < static_cast<int>(mb.blocks.size()),
                  "row merge cell block id out of range");
    return mb.blocks[static_cast<std::size_t>(block)].role;
  };

  int reference_block = -1;
  int subject_cell = failing_cell;
  if (block_role(failing_cell) == mesh::BlockRole::TRANSITION_BELT) {
    reference_block = mb.cell_block_id[static_cast<std::size_t>(failing_cell)];
  } else {
    const int face_off =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(failing_cell)];
    const int face_next =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(failing_cell + 1)];
    const int nv = active_nverts(state.mesh.cell_nverts, failing_cell);
    for (int local = 0; local < face_next - face_off; ++local) {
      int corner0 = -1;
      int corner1 = -1;
      if (!mesh::mesh_topo_active_local_face_corners(
              nv, local, &corner0, &corner1)) {
        continue;
      }
      const int neighbor = mb.face_adj_csr_indices[
          static_cast<std::size_t>(face_off + local)];
      if (neighbor >= 0 && neighbor < n_cells &&
          block_role(neighbor) == mesh::BlockRole::TRANSITION_BELT) {
        reference_block =
            mb.cell_block_id[static_cast<std::size_t>(neighbor)];
        subject_cell = neighbor;
        break;
      }
    }
  }
  if (reference_block < 0) {
    return reject(result, "not_tier_row", failing_cell);
  }

  result.engaged = true;
  if (cfg.materials.materials.size() > 1U) {
    static std::once_flag warning_once;
    std::call_once(warning_once, []() {
      core::log_warning(
          "[row-merge] multimaterial/MOF projection is unsupported in v1");
    });
    return reject(result, "multimaterial_unsupported", failing_cell);
  }

  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "row merge requires one declared material");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells * kStorageSlots) &&
                    mb.face_adj_csr_indices.size() ==
                        static_cast<std::size_t>(n_cells * kStorageSlots),
                "row merge requires uniform eight-slot topology CSR arrays");
  for (int cell = 0; cell < n_cells; ++cell) {
    TENRYU_ASSERT(
        mb.cell_node_csr_offsets[static_cast<std::size_t>(cell + 1)] -
                mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)] ==
            kStorageSlots,
        "row merge requires eight cell-node CSR slots per cell");
    TENRYU_ASSERT(
        mb.face_adj_csr_offsets[static_cast<std::size_t>(cell + 1)] -
                mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)] ==
            kStorageSlots,
        "row merge requires eight face-adjacency CSR slots per cell");
  }

  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_r_initial.size() ==
                        static_cast<std::size_t>(n_nodes) &&
                    state.x_z_initial.size() ==
                        static_cast<std::size_t>(n_nodes),
                "row merge coordinate fields must match mesh node count");
  std::vector<double> x_r;
  std::vector<double> x_z;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);

  const auto& csr_offsets = mb.cell_node_csr_offsets;
  const auto& csr_indices = mb.cell_node_csr_indices;
  const auto& orientation_sign = mb.cell_orientation_sign;
  const auto& cell_nverts = state.mesh.cell_nverts;
  std::map<Edge, BoundaryEdgeInstall> boundary_edge_install;
  for (const mesh::UniqueOrientedFace& face : mb.boundary_faces) {
    TENRYU_ASSERT(face.cell_a >= 0 && face.cell_a < n_cells &&
                      face.cell_b == -1 && face.local_b == -1,
                  "row merge found malformed boundary-face metadata");
    const int nverts = active_nverts(cell_nverts, face.cell_a);
    int corner0 = -1;
    int corner1 = -1;
    TENRYU_ASSERT(mesh::mesh_topo_active_local_face_corners(
                      nverts, face.local_a, &corner0, &corner1),
                  "row merge found invalid boundary-face local index");
    const int off = csr_offsets[static_cast<std::size_t>(face.cell_a)];
    const Edge edge = undirected_edge(
        csr_indices[static_cast<std::size_t>(off + corner0)],
        csr_indices[static_cast<std::size_t>(off + corner1)]);
    const auto inserted = boundary_edge_install.emplace(
        edge, BoundaryEdgeInstall{-1, -1, face.bc_tag});
    TENRYU_ASSERT(inserted.second,
                  "row merge found duplicate boundary edge");
  }
  const Polygon failing_polygon = cell_polygon(
      failing_cell, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
  const double failing_radius = polygon_centroid_radius(failing_polygon);
  const double half_pitch = 0.5 * polygon_radial_extent(failing_polygon);

  struct BlockStats {
    double min_radius = std::numeric_limits<double>::infinity();
    double max_radius = -std::numeric_limits<double>::infinity();
    double mean_radius = 0.0;
    double mean_z = 0.0;
  };
  const auto block_stats = [&](const int block) {
    const auto& info = mb.blocks[static_cast<std::size_t>(block)];
    BlockStats stats;
    for (int offset = 0; offset < info.cell_count; ++offset) {
      const int cell = info.cell_begin + offset;
      const Polygon polygon = cell_polygon(
          cell, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
      const double radius = polygon_centroid_radius(polygon);
      stats.min_radius = std::min(stats.min_radius, radius);
      stats.max_radius = std::max(stats.max_radius, radius);
      stats.mean_radius += radius;
      stats.mean_z += polygon_centroid_z(polygon);
    }
    TENRYU_ASSERT(info.cell_count > 0,
                  "row merge transition block must contain cells");
    stats.mean_radius /= static_cast<double>(info.cell_count);
    stats.mean_z /= static_cast<double>(info.cell_count);
    return stats;
  };

  std::vector<int> orbit_blocks;
  for (int block = 0; block < static_cast<int>(mb.blocks.size()); ++block) {
    if (mb.blocks[static_cast<std::size_t>(block)].role !=
        mesh::BlockRole::TRANSITION_BELT) {
      continue;
    }
    const BlockStats stats = block_stats(block);
    if (failing_radius >= stats.min_radius - half_pitch &&
        failing_radius <= stats.max_radius + half_pitch) {
      orbit_blocks.push_back(block);
    }
  }
  if (orbit_blocks.empty()) {
    orbit_blocks.push_back(reference_block);
    const BlockStats reference_stats = block_stats(reference_block);
    for (int block = 0; block < static_cast<int>(mb.blocks.size()); ++block) {
      if (block == reference_block ||
          mb.blocks[static_cast<std::size_t>(block)].role !=
              mesh::BlockRole::TRANSITION_BELT) {
        continue;
      }
      const BlockStats candidate_stats = block_stats(block);
      const double tolerance =
          0.01 * std::max(std::abs(reference_stats.mean_radius),
                          std::numeric_limits<double>::min());
      if (std::abs(candidate_stats.mean_radius - reference_stats.mean_radius) <=
              tolerance &&
          candidate_stats.mean_z * reference_stats.mean_z < 0.0) {
        orbit_blocks.push_back(block);
        break;
      }
    }
  }
  std::sort(orbit_blocks.begin(), orbit_blocks.end());

  std::vector<int> row_cells;
  for (const int block : orbit_blocks) {
    const auto& info = mb.blocks[static_cast<std::size_t>(block)];
    if (info.n_i_cells == 1) {
      for (int offset = 0; offset < info.cell_count; ++offset) {
        row_cells.push_back(info.cell_begin + offset);
      }
      continue;
    }

    TENRYU_ASSERT(info.n_i_cells > 1 && info.n_j_cells > 0 &&
                      info.n_i_cells * info.n_j_cells == info.cell_count,
                  "row merge multi-row transition block metadata mismatch");
    int selected_row = -1;
    if (failing_cell >= info.cell_begin &&
        failing_cell < info.cell_begin + info.cell_count) {
      selected_row = (failing_cell - info.cell_begin) / info.n_j_cells;
    } else {
      double best_distance = std::numeric_limits<double>::infinity();
      for (int row = 0; row < info.n_i_cells; ++row) {
        double mean_radius = 0.0;
        for (int column = 0; column < info.n_j_cells; ++column) {
          const int cell =
              info.cell_begin + row * info.n_j_cells + column;
          mean_radius += polygon_centroid_radius(cell_polygon(
              cell, csr_offsets, csr_indices, cell_nverts, x_r, x_z));
        }
        mean_radius /= static_cast<double>(info.n_j_cells);
        const double distance = std::abs(mean_radius - failing_radius);
        if (distance < best_distance) {
          best_distance = distance;
          selected_row = row;
        }
      }
    }
    TENRYU_ASSERT(selected_row >= 0 && selected_row < info.n_i_cells,
                  "row merge failed to select transition block row");
    for (int column = 0; column < info.n_j_cells; ++column) {
      row_cells.push_back(
          info.cell_begin + selected_row * info.n_j_cells + column);
    }
  }
  std::sort(row_cells.begin(), row_cells.end());
  row_cells.erase(std::unique(row_cells.begin(), row_cells.end()),
                  row_cells.end());
  if (row_cells.empty()) {
    return reject(result, "pairing_failed", failing_cell);
  }

  std::vector<std::uint8_t> in_orbit(static_cast<std::size_t>(n_cells), 0U);
  for (const int cell : row_cells) {
    in_orbit[static_cast<std::size_t>(cell)] = 1U;
  }
  std::vector<std::int8_t> hydro_active = state.hydro_active;
  const bool hydro_active_state_empty = hydro_active.empty();
  std::size_t hydro_active_zero_count = 0;
  if (hydro_active_state_empty) {
    hydro_active.assign(static_cast<std::size_t>(n_cells),
                        static_cast<std::int8_t>(1));
  } else {
    hydro_active_zero_count = static_cast<std::size_t>(std::count(
        hydro_active.begin(), hydro_active.end(), static_cast<std::int8_t>(0)));
  }
  std::ostringstream hydro_active_log;
  hydro_active_log << "[row-merge] entry hydro_active: state_empty="
                   << (hydro_active_state_empty ? 1 : 0)
                   << " zeros=" << hydro_active_zero_count;
  core::log_warning(hydro_active_log.str());
  TENRYU_ASSERT(hydro_active.size() == static_cast<std::size_t>(n_cells),
                "row merge hydro_active size mismatch");

  std::vector<int> partner_use_count(static_cast<std::size_t>(n_cells), 0);
  std::vector<PairPlan> plans;
  plans.reserve(row_cells.size());
  for (const int row_cell : row_cells) {
    if (hydro_active[static_cast<std::size_t>(row_cell)] == 0) {
      if (row_cell == subject_cell) {
        return reject(result, "pairing_failed", row_cell);
      }
      continue;
    }
    const Polygon row_polygon = cell_polygon(
        row_cell, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
    const int face_off =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(row_cell)];
    const int face_next =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(row_cell + 1)];
    std::vector<std::pair<double, int>> partner_candidates;
    partner_candidates.reserve(static_cast<std::size_t>(face_next - face_off));
    for (int local = 0; local < face_next - face_off; ++local) {
      int corner0 = -1;
      int corner1 = -1;
      if (!mesh::mesh_topo_active_local_face_corners(
              row_polygon.n, local, &corner0, &corner1)) {
        continue;
      }
      const int neighbor = mb.face_adj_csr_indices[
          static_cast<std::size_t>(face_off + local)];
      if (neighbor < 0 || neighbor >= n_cells) {
        continue;
      }
      const double midpoint_r =
          0.5 * (row_polygon.r[static_cast<std::size_t>(corner0)] +
                 row_polygon.r[static_cast<std::size_t>(corner1)]);
      const double midpoint_z =
          0.5 * (row_polygon.z[static_cast<std::size_t>(corner0)] +
                 row_polygon.z[static_cast<std::size_t>(corner1)]);
      const double midpoint_radius = std::hypot(midpoint_r, midpoint_z);
      partner_candidates.emplace_back(midpoint_radius, neighbor);
    }
    std::sort(partner_candidates.begin(), partner_candidates.end());
    int partner = -1;
    for (const auto& candidate : partner_candidates) {
      const int neighbor = candidate.second;
      if (in_orbit[static_cast<std::size_t>(neighbor)] != 0U ||
          hydro_active[static_cast<std::size_t>(neighbor)] == 0 ||
          partner_use_count[static_cast<std::size_t>(neighbor)] != 0) {
        continue;
      }
      partner = neighbor;
      break;
    }
    PairPlan plan;
    plan.row_cell = row_cell;
    plan.partner = partner;
    if (partner < 0) {
      if (row_cell == subject_cell ||
          (partner >= 0 && partner == subject_cell)) {
        return reject(result, "pairing_failed", row_cell);
      }
      plan.skip_merge = true;
      plans.push_back(plan);
      ++result.skipped_pairs;
      core::log_warning(
          "[row-merge] pair row_cell=" + std::to_string(row_cell) +
          " skip_merge (pairing)");
      continue;
    }
    ++partner_use_count[static_cast<std::size_t>(partner)];
    plan.survivor = std::min(row_cell, partner);
    plan.absorbed = std::max(row_cell, partner);
    plans.push_back(plan);
  }

  const auto require_cell_field = [&](const core::CellField1D& field,
                                      const char* message) {
    TENRYU_ASSERT(field.size() == static_cast<std::size_t>(n_cells), message);
  };
  require_cell_field(state.rho, "row merge rho size mismatch");
  require_cell_field(state.mass, "row merge mass size mismatch");
  require_cell_field(state.vol, "row merge volume size mismatch");
  require_cell_field(state.ee, "row merge ee size mismatch");
  require_cell_field(state.ei, "row merge ei size mismatch");
  TENRYU_ASSERT(state.corner_mass.size() ==
                    static_cast<std::size_t>(n_cells * kStorageSlots),
                "row merge corner_mass size mismatch");
  TENRYU_ASSERT(state.v_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.v_z.size() == static_cast<std::size_t>(n_nodes),
                "row merge velocity size mismatch");

  std::vector<double> rho;
  std::vector<double> mass;
  std::vector<double> vol;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> corner_mass;
  std::vector<double> v_r;
  std::vector<double> v_z;
  state.rho.copy_to_host(rho);
  state.mass.copy_to_host(mass);
  state.vol.copy_to_host(vol);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.corner_mass.copy_to_host(corner_mass);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);

  std::vector<double> rad_E;
  const int radiation_groups = cfg.radiation.groups;
  if (cfg.radiation.enabled) {
    TENRYU_ASSERT(radiation_groups > 0 &&
                      state.rad_E.size() ==
                          static_cast<std::size_t>(n_cells) *
                              static_cast<std::size_t>(radiation_groups),
                  "row merge radiation group field size mismatch");
    state.rad_E.copy_to_host(rad_E);
  }

  std::vector<int> csr_indices_new = csr_indices;
  std::vector<std::uint8_t> cell_nverts_new = cell_nverts;
  std::vector<int> orientation_sign_new = orientation_sign;
  std::vector<double> rho_new = rho;
  std::vector<double> mass_new = mass;
  std::vector<double> vol_new = vol;
  std::vector<double> ee_new = ee;
  std::vector<double> ei_new = ei;
  std::vector<double> corner_mass_new = corner_mass;
  std::vector<double> rad_E_new = rad_E;
  std::vector<double> v_r_new = v_r;
  std::vector<double> v_z_new = v_z;
  std::vector<double> corner_momentum_r_new(corner_mass.size(), 0.0);
  std::vector<double> corner_momentum_z_new(corner_mass.size(), 0.0);
  std::vector<double> corner_electron_internal_new(corner_mass.size(), 0.0);
  std::vector<double> corner_ion_internal_new(corner_mass.size(), 0.0);
  std::vector<double> corner_kinetic_cons_new(corner_mass.size(), 0.0);
  const std::vector<std::int8_t> hydro_active_old = hydro_active;
  const ReverseCellNodeCSR reverse_csr = build_reverse_cell_node_csr(
      mb, n_nodes, &cell_nverts, kStorageSlots);
  int kept_union_pairs = 0;

  // Pass 1: classify every pair before assigning any third-part slots.
  for (PairPlan& plan : plans) {
    if (plan.skip_merge) {
      continue;
    }
    const bool pair_contains_subject =
        plan.row_cell == failing_cell || plan.partner == failing_cell ||
        plan.row_cell == subject_cell || plan.partner == subject_cell;
    const char* skip_reason = nullptr;
    {
      const Polygon old_a = cell_polygon(
          plan.survivor, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
      const Polygon old_b = cell_polygon(
          plan.absorbed, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
      int shared_lo = -1;
      int shared_hi = -1;
      const UnionStatus union_status = build_union_polygon(
          old_a, old_b, plan.survivor, plan.absorbed,
          reverse_csr, hydro_active_old,
          orientation_sign[static_cast<std::size_t>(plan.survivor)],
          x_r, x_z, plan.union_polygon, shared_lo, shared_hi);
      if (union_status == UnionStatus::kPairingFailed) {
        if (pair_contains_subject) {
          return reject(result, "pairing_failed", plan.row_cell);
        }
        skip_reason = "pairing";
        goto classification_finished;
      }
      if (union_status == UnionStatus::kSharedEdgeAmbiguous) {
        return reject(result, "shared_edge_ambiguous", plan.row_cell);
      }
      if (union_status == UnionStatus::kOrientation) {
        return reject(result, "union_orientation", plan.row_cell);
      }
      if (plan.union_polygon.n > 2 * kStorageSlots - 2) {
        if (pair_contains_subject) {
          return reject(result, "valence_overflow", plan.row_cell);
        }
        skip_reason = "valence";
        goto classification_finished;
      }

      const PhysicalMoments moments_a = physical_moments(
          old_a, orientation_sign[static_cast<std::size_t>(plan.survivor)]);
      const PhysicalMoments moments_b = physical_moments(
          old_b, orientation_sign[static_cast<std::size_t>(plan.absorbed)]);
      const PhysicalMoments moments_union = physical_moments(
          plan.union_polygon,
          orientation_sign[static_cast<std::size_t>(plan.survivor)]);
      const double old_volume_sum = moments_a.volume + moments_b.volume;
      const double volume_tolerance =
          64.0 * std::numeric_limits<double>::epsilon() *
          moments_union.volume;
      if (!(moments_union.volume > 0.0) ||
          std::abs(moments_union.volume - old_volume_sum) >
              volume_tolerance ||
          !relative_identity(moments_union.first_r,
                             moments_a.first_r + moments_b.first_r) ||
          !relative_identity(moments_union.first_z,
                             moments_a.first_z + moments_b.first_z)) {
        return reject(result, "volume_identity", plan.row_cell);
      }

      const std::array<int, 2> old_cells{{plan.survivor, plan.absorbed}};
      const double merged_mass =
          mass[static_cast<std::size_t>(old_cells[0])] +
          mass[static_cast<std::size_t>(old_cells[1])];
      if (!(merged_mass > 0.0) || !std::isfinite(merged_mass)) {
        if (pair_contains_subject) {
          return reject(result, "energy_floor", plan.row_cell);
        }
        skip_reason = "energy_floor";
        goto classification_finished;
      }
      for (const int old_cell : old_cells) {
        const std::size_t cell = static_cast<std::size_t>(old_cell);
        if (!(mass[cell] > 0.0) || !std::isfinite(mass[cell]) ||
            !std::isfinite(ee[cell]) || !std::isfinite(ei[cell]) ||
            ee[cell] < 0.0 || ei[cell] < 0.0) {
          if (pair_contains_subject) {
            return reject(result, "energy_floor", plan.row_cell);
          }
          skip_reason = "energy_floor";
          goto classification_finished;
        }
        const Polygon old_polygon = cell_polygon(
            old_cell, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
        double old_corner_mass_sum = 0.0;
        for (int q = 0; q < old_polygon.n; ++q) {
          const std::size_t corner =
              cell * static_cast<std::size_t>(kStorageSlots) +
              static_cast<std::size_t>(q);
          const double old_corner_mass = corner_mass[corner];
          const int node = old_polygon.nodes[static_cast<std::size_t>(q)];
          const double speed2 =
              v_r[static_cast<std::size_t>(node)] *
                  v_r[static_cast<std::size_t>(node)] +
              v_z[static_cast<std::size_t>(node)] *
                  v_z[static_cast<std::size_t>(node)];
          if (!std::isfinite(old_corner_mass) || old_corner_mass < 0.0) {
            return reject(result, "corner_defect", plan.row_cell);
          }
          if (!std::isfinite(speed2)) {
            if (pair_contains_subject) {
              return reject(result, "momentum_defect", plan.row_cell);
            }
            skip_reason = "momentum_defect";
            goto classification_finished;
          }
          old_corner_mass_sum += old_corner_mass;
        }
        if (!(old_corner_mass_sum > 0.0) ||
            !std::isfinite(old_corner_mass_sum) ||
            !relative_identity(old_corner_mass_sum, mass[cell])) {
          return reject(result, "corner_defect", plan.row_cell);
        }
      }

      int survivor_smallest_node = old_a.nodes[0];
      for (int k = 1; k < old_a.n; ++k) {
        survivor_smallest_node = std::min(
            survivor_smallest_node,
            old_a.nodes[static_cast<std::size_t>(k)]);
      }
      plan.split_part_count = choose_split(
          plan.union_polygon, shared_lo, shared_hi, survivor_smallest_node,
          orientation_sign[static_cast<std::size_t>(plan.survivor)],
          plan.half_a, plan.half_b, plan.half_c,
          plan.chord_u, plan.chord_v, plan.chord2_u, plan.chord2_v);
      if (plan.split_part_count == 0) {
        if (plan.union_polygon.n > kStorageSlots) {
          if (pair_contains_subject) {
            core::log_warning(
                "[row-merge] split subject row_cell=" +
                std::to_string(plan.row_cell) +
                " partner=" + std::to_string(plan.partner));
            return reject(result, "split_infeasible", plan.row_cell);
          }
          skip_reason = "split_infeasible";
          goto classification_finished;
        }
        plan.keep_union = true;
        ++kept_union_pairs;
        core::log_warning(
            "[row-merge] pair row_cell=" + std::to_string(plan.row_cell) +
            " partner=" + std::to_string(plan.partner) +
            " keep_union (split infeasible)");
      }
      if (plan.split_part_count == 2 &&
          !split_edge_multiset_matches(
              plan.union_polygon, plan.half_a, plan.half_b,
              plan.chord_u, plan.chord_v)) {
        return reject(result, "split_edge_multiset", plan.row_cell);
      }
      if (plan.split_part_count == 3 &&
          !three_way_split_edge_multiset_matches(
              plan.union_polygon, plan.half_a, plan.half_b, plan.half_c,
              plan.chord_u, plan.chord_v,
              plan.chord2_u, plan.chord2_v)) {
        return reject(result, "split_edge_multiset", plan.row_cell);
      }
    }
classification_finished:
    if (skip_reason != nullptr) {
      plan.skip_merge = true;
      ++result.skipped_pairs;
      core::log_warning(
          "[row-merge] pair row_cell=" + std::to_string(plan.row_cell) +
          " partner=" + std::to_string(plan.partner) +
          " skip_merge (" + skip_reason + ")");
    }
  }

  std::vector<std::uint8_t> free_cell_candidate(
      static_cast<std::size_t>(n_cells), 0U);
  for (int cell = 0; cell < n_cells; ++cell) {
    if (hydro_active_old[static_cast<std::size_t>(cell)] == 0) {
      free_cell_candidate[static_cast<std::size_t>(cell)] = 1U;
    }
  }
  for (const PairPlan& plan : plans) {
    if (!plan.skip_merge && plan.keep_union) {
      free_cell_candidate[static_cast<std::size_t>(plan.absorbed)] = 1U;
    }
  }
  std::vector<int> free_cells;
  for (int cell = 0; cell < n_cells; ++cell) {
    if (free_cell_candidate[static_cast<std::size_t>(cell)] != 0U) {
      free_cells.push_back(cell);
    }
  }

  // Pass 2: pair order is ascending row_cell; free_cells is ascending cell id.
  std::size_t next_free_cell = 0;
  std::vector<std::uint8_t> reused_third_cell(
      static_cast<std::size_t>(n_cells), 0U);
  std::vector<std::uint8_t> staged_output_cell(
      static_cast<std::size_t>(n_cells), 0U);
  for (PairPlan& plan : plans) {
    if (plan.skip_merge || plan.split_part_count != 3) {
      continue;
    }
    if (next_free_cell == free_cells.size()) {
      const bool pair_contains_subject =
          plan.row_cell == failing_cell || plan.partner == failing_cell ||
          plan.row_cell == subject_cell || plan.partner == subject_cell;
      if (pair_contains_subject) {
        return reject(result, "no_free_cell", plan.row_cell);
      }
      plan.skip_merge = true;
      ++result.skipped_pairs;
      core::log_warning(
          "[row-merge] pair row_cell=" + std::to_string(plan.row_cell) +
          " partner=" + std::to_string(plan.partner) +
          " skip_merge (no_free_cell)");
      continue;
    }
    plan.third_cell = free_cells[next_free_cell++];
    reused_third_cell[static_cast<std::size_t>(plan.third_cell)] = 1U;
    staged_output_cell[static_cast<std::size_t>(plan.third_cell)] = 1U;
  }
  std::ostringstream third_slots;
  third_slots << "[row-merge] third slots: ";
  bool first_third_slot = true;
  for (const PairPlan& plan : plans) {
    if (plan.third_cell < 0) {
      continue;
    }
    if (!first_third_slot) {
      third_slots << ",";
    }
    third_slots << plan.third_cell;
    first_third_slot = false;
  }
  core::log_warning(third_slots.str());

  // Stage every keep_union tombstone before any destination reactivation.
  // A reused slot is therefore overwritten only by its final split3 values.
  // Inactive CSR/nverts stay intact; split3 overwrites every CSR slot and nverts.
  std::vector<std::uint8_t> tombstone_cell(
      static_cast<std::size_t>(n_cells), 0U);
  for (const PairPlan& plan : plans) {
    if (plan.skip_merge || !plan.keep_union) {
      continue;
    }
    const std::size_t absorbed = static_cast<std::size_t>(plan.absorbed);
    tombstone_cell[absorbed] = 1U;
    mass_new[absorbed] = 0.0;
    rho_new[absorbed] = 0.0;
    ee_new[absorbed] = 0.0;
    ei_new[absorbed] = 0.0;
    hydro_active[absorbed] = 0;
    for (int k = 0; k < kStorageSlots; ++k) {
      const std::size_t corner =
          absorbed * static_cast<std::size_t>(kStorageSlots) +
          static_cast<std::size_t>(k);
      corner_mass_new[corner] = 0.0;
      corner_momentum_r_new[corner] = 0.0;
      corner_momentum_z_new[corner] = 0.0;
      corner_electron_internal_new[corner] = 0.0;
      corner_ion_internal_new[corner] = 0.0;
      corner_kinetic_cons_new[corner] = 0.0;
    }
    if (cfg.radiation.enabled) {
      for (int group = 0; group < radiation_groups; ++group) {
        rad_E_new[
            absorbed * static_cast<std::size_t>(radiation_groups) +
            static_cast<std::size_t>(group)] = 0.0;
      }
    }
  }

  double dV_rel_max = 0.0;
  double corner_defect_max = 0.0;
  std::vector<int> affected_nodes;
  std::vector<CornerTransfer> transfers;

  for (PairPlan& plan : plans) {
    if (plan.skip_merge) {
      continue;
    }
    double pair_dV_rel_max = 0.0;
    double pair_corner_defect_max = 0.0;
    const Polygon old_a = cell_polygon(
        plan.survivor, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
    const Polygon old_b = cell_polygon(
        plan.absorbed, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
    const PhysicalMoments moments_a = physical_moments(
        old_a, orientation_sign[static_cast<std::size_t>(plan.survivor)]);
    const PhysicalMoments moments_b = physical_moments(
        old_b, orientation_sign[static_cast<std::size_t>(plan.absorbed)]);
    const PhysicalMoments moments_union = physical_moments(
        plan.union_polygon,
        orientation_sign[static_cast<std::size_t>(plan.survivor)]);
    const double old_volume_sum = moments_a.volume + moments_b.volume;
    const double volume_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        moments_union.volume;
    pair_dV_rel_max = std::max(
        pair_dV_rel_max,
        std::abs(moments_union.volume - old_volume_sum) /
            moments_union.volume);

    const std::array<int, 2> old_cells{{plan.survivor, plan.absorbed}};
    const double merged_mass =
        mass[static_cast<std::size_t>(old_cells[0])] +
        mass[static_cast<std::size_t>(old_cells[1])];
    const bool split_selected = plan.split_part_count != 0;
    std::array<Polygon, 3> destination_polygons{};
    std::array<PhysicalMoments, 3> destination_moments{};
    std::array<int, 3> destination_cells{{
        plan.survivor, plan.absorbed, plan.third_cell}};
    int destination_count = 1;
    destination_polygons[0] = plan.union_polygon;
    destination_moments[0] = moments_union;
    if (split_selected) {
      destination_count = plan.split_part_count;
      destination_polygons[0] = plan.half_a;
      destination_polygons[1] = plan.half_b;
      destination_polygons[2] = plan.half_c;
      destination_moments[0] = physical_moments(
          plan.half_a,
          orientation_sign[static_cast<std::size_t>(plan.survivor)]);
      destination_moments[1] = physical_moments(
          plan.half_b,
          orientation_sign[static_cast<std::size_t>(plan.survivor)]);
      if (destination_count == 3) {
        destination_moments[2] = physical_moments(
            plan.half_c,
            orientation_sign[static_cast<std::size_t>(plan.survivor)]);
      }
      double half_volume_sum = 0.0;
      for (int destination = 0; destination < destination_count;
           ++destination) {
        half_volume_sum +=
            destination_moments[static_cast<std::size_t>(destination)].volume;
      }
      if (std::abs(half_volume_sum - moments_union.volume) >
          volume_tolerance) {
        return reject(result, "volume_identity", plan.row_cell);
      }
      pair_dV_rel_max = std::max(
          pair_dV_rel_max,
          std::abs(half_volume_sum - moments_union.volume) /
              moments_union.volume);
    }

    std::array<std::array<Subcell, kStorageSlots>, 3>
        destination_subcells{};
    for (int destination = 0; destination < destination_count;
         ++destination) {
      const Polygon& polygon =
          destination_polygons[static_cast<std::size_t>(destination)];
      for (int p = 0; p < polygon.n; ++p) {
        destination_subcells[static_cast<std::size_t>(destination)]
                            [static_cast<std::size_t>(p)] =
            make_subcell(polygon, p);
      }
    }

    // Consult-24 §§3.2-3.3: one non-negative overlap mass operator carries
    // mass, momentum, both internal energies, and conservative kinetic energy.
    std::array<std::array<CornerPacket, kStorageSlots>, 3>
        destination_packets{};
    std::vector<CornerTransfer> pair_transfers;
    double pair_mass = 0.0;
    for (const int old_cell : old_cells) {
      const std::size_t source_cell = static_cast<std::size_t>(old_cell);
      const Polygon old_polygon = cell_polygon(
          old_cell, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
      for (int q = 0; q < old_polygon.n; ++q) {
        const std::size_t source_corner =
            source_cell * static_cast<std::size_t>(kStorageSlots) +
            static_cast<std::size_t>(q);
        const double source_mass = corner_mass[source_corner];
        const int source_node =
            old_polygon.nodes[static_cast<std::size_t>(q)];
        const double source_vr = v_r[static_cast<std::size_t>(source_node)];
        const double source_vz = v_z[static_cast<std::size_t>(source_node)];
        const double source_speed2 =
            source_vr * source_vr + source_vz * source_vz;
        const Subcell source_subcell = make_subcell(old_polygon, q);
        std::array<std::array<double, kStorageSlots>, 3> overlaps{};
        double overlap_sum = 0.0;
        for (int destination = 0; destination < destination_count;
             ++destination) {
          const Polygon& polygon =
              destination_polygons[static_cast<std::size_t>(destination)];
          for (int p = 0; p < polygon.n; ++p) {
            const double overlap = intersection_volume(
                source_subcell,
                destination_subcells[static_cast<std::size_t>(destination)]
                                    [static_cast<std::size_t>(p)]);
            if (!std::isfinite(overlap) || overlap < 0.0) {
              return reject(result, "signed_transfer", plan.row_cell);
            }
            overlaps[static_cast<std::size_t>(destination)]
                    [static_cast<std::size_t>(p)] = overlap;
            overlap_sum += overlap;
          }
        }
        if (!(overlap_sum > 0.0) || !std::isfinite(overlap_sum)) {
          return reject(result, "corner_defect", plan.row_cell);
        }

        double donor_mass_sum = 0.0;
        for (int destination = 0; destination < destination_count;
             ++destination) {
          const Polygon& polygon =
              destination_polygons[static_cast<std::size_t>(destination)];
          const int destination_cell =
              destination_cells[static_cast<std::size_t>(destination)];
          for (int p = 0; p < polygon.n; ++p) {
            const double overlap =
                overlaps[static_cast<std::size_t>(destination)]
                        [static_cast<std::size_t>(p)];
            if (overlap == 0.0) {
              continue;
            }
            const double fraction = overlap / overlap_sum;
            const double transfer_mass = source_mass * fraction;
            if (!std::isfinite(fraction) || fraction < 0.0 ||
                !std::isfinite(transfer_mass) || transfer_mass < 0.0) {
              return reject(result, "signed_transfer", plan.row_cell);
            }
            CornerPacket& packet =
                destination_packets[static_cast<std::size_t>(destination)]
                                   [static_cast<std::size_t>(p)];
            packet.mass += transfer_mass;
            packet.momentum_r += transfer_mass * source_vr;
            packet.momentum_z += transfer_mass * source_vz;
            packet.electron_internal += transfer_mass * ee[source_cell];
            packet.ion_internal += transfer_mass * ei[source_cell];
            packet.kinetic_cons += 0.5 * transfer_mass * source_speed2;
            donor_mass_sum += transfer_mass;
            const std::size_t destination_corner =
                static_cast<std::size_t>(destination_cell) *
                    static_cast<std::size_t>(kStorageSlots) +
                static_cast<std::size_t>(p);
            pair_transfers.push_back(CornerTransfer{
                source_corner,
                destination_corner,
                source_node,
                polygon.nodes[static_cast<std::size_t>(p)],
                transfer_mass,
            });
          }
        }
        const double donor_scale = std::max(
            std::abs(source_mass), std::numeric_limits<double>::min());
        const double donor_defect =
            std::abs(donor_mass_sum - source_mass) / donor_scale;
        pair_corner_defect_max =
            std::max(pair_corner_defect_max, donor_defect);
        if (!relative_identity(donor_mass_sum, source_mass)) {
          return reject(result, "corner_defect", plan.row_cell);
        }
      }
    }

    std::array<double, 3> destination_mass{};
    std::array<double, 3> destination_electron_internal{};
    std::array<double, 3> destination_ion_internal{};
    for (int destination = 0; destination < destination_count;
         ++destination) {
      const Polygon& polygon =
          destination_polygons[static_cast<std::size_t>(destination)];
      for (int p = 0; p < polygon.n; ++p) {
        const CornerPacket& packet =
            destination_packets[static_cast<std::size_t>(destination)]
                               [static_cast<std::size_t>(p)];
        if (!std::isfinite(packet.mass) || packet.mass < 0.0 ||
            !std::isfinite(packet.momentum_r) ||
            !std::isfinite(packet.momentum_z) ||
            !std::isfinite(packet.electron_internal) ||
            packet.electron_internal < 0.0 ||
            !std::isfinite(packet.ion_internal) ||
            packet.ion_internal < 0.0 ||
            !std::isfinite(packet.kinetic_cons) ||
            packet.kinetic_cons < 0.0) {
          return reject(result, "positivity_broken", plan.row_cell);
        }
        destination_mass[static_cast<std::size_t>(destination)] +=
            packet.mass;
        destination_electron_internal[static_cast<std::size_t>(destination)] +=
            packet.electron_internal;
        destination_ion_internal[static_cast<std::size_t>(destination)] +=
            packet.ion_internal;
      }
      if (!(destination_mass[static_cast<std::size_t>(destination)] > 0.0) ||
          !std::isfinite(
              destination_mass[static_cast<std::size_t>(destination)])) {
        return reject(result, "positivity_broken", plan.row_cell);
      }
      pair_mass += destination_mass[static_cast<std::size_t>(destination)];
    }
    if (!relative_identity(pair_mass, merged_mass)) {
      return reject(result, "corner_defect", plan.row_cell);
    }

    const std::size_t survivor = static_cast<std::size_t>(plan.survivor);
    for (int destination = 0; destination < destination_count;
         ++destination) {
      const std::size_t cell = static_cast<std::size_t>(
          destination_cells[static_cast<std::size_t>(destination)]);
      const Polygon& polygon =
          destination_polygons[static_cast<std::size_t>(destination)];
      mass_new[cell] = destination_mass[static_cast<std::size_t>(destination)];
      rho_new[cell] = mass_new[cell] /
                      destination_moments[static_cast<std::size_t>(destination)]
                          .volume;
      vol_new[cell] =
          destination_moments[static_cast<std::size_t>(destination)].volume;
      ee_new[cell] =
          destination_electron_internal[static_cast<std::size_t>(destination)] /
          mass_new[cell];
      ei_new[cell] =
          destination_ion_internal[static_cast<std::size_t>(destination)] /
          mass_new[cell];
      hydro_active[cell] = 1;
      orientation_sign_new[cell] =
          orientation_sign[static_cast<std::size_t>(plan.survivor)];
      const int off = csr_offsets[cell];
      for (int k = 0; k < kStorageSlots; ++k) {
        const std::size_t corner =
            cell * static_cast<std::size_t>(kStorageSlots) +
            static_cast<std::size_t>(k);
        csr_indices_new[static_cast<std::size_t>(off + k)] =
            k < polygon.n ? polygon.nodes[static_cast<std::size_t>(k)] : 0;
        const CornerPacket packet =
            k < polygon.n
                ? destination_packets[static_cast<std::size_t>(destination)]
                                     [static_cast<std::size_t>(k)]
                : CornerPacket{};
        corner_mass_new[corner] = packet.mass;
        corner_momentum_r_new[corner] = packet.momentum_r;
        corner_momentum_z_new[corner] = packet.momentum_z;
        corner_electron_internal_new[corner] = packet.electron_internal;
        corner_ion_internal_new[corner] = packet.ion_internal;
        corner_kinetic_cons_new[corner] = packet.kinetic_cons;
      }
      cell_nverts_new[cell] = static_cast<std::uint8_t>(polygon.n);
    }

    if (cfg.radiation.enabled) {
      for (int group = 0; group < radiation_groups; ++group) {
        double radiation_energy = 0.0;
        for (const int old_cell : old_cells) {
          const std::size_t index =
              static_cast<std::size_t>(old_cell) *
                  static_cast<std::size_t>(radiation_groups) +
              static_cast<std::size_t>(group);
          radiation_energy +=
              rad_E[index] * vol[static_cast<std::size_t>(old_cell)];
        }
        if (split_selected) {
          double destination_volume = 0.0;
          for (int destination = 0; destination < destination_count;
               ++destination) {
            destination_volume +=
                destination_moments[static_cast<std::size_t>(destination)]
                    .volume;
          }
          const double radiation_density =
              radiation_energy / destination_volume;
          for (int destination = 0; destination < destination_count;
               ++destination) {
            const std::size_t destination_cell = static_cast<std::size_t>(
                destination_cells[static_cast<std::size_t>(destination)]);
            rad_E_new[
                destination_cell *
                    static_cast<std::size_t>(radiation_groups) +
                static_cast<std::size_t>(group)] = radiation_density;
          }
        } else {
          rad_E_new[survivor * static_cast<std::size_t>(radiation_groups) +
                    static_cast<std::size_t>(group)] =
              radiation_energy / moments_union.volume;
        }
      }
    }

    for (const int old_cell : old_cells) {
      const Polygon old_polygon = cell_polygon(
          old_cell, csr_offsets, csr_indices, cell_nverts, x_r, x_z);
      for (int k = 0; k < old_polygon.n; ++k) {
        affected_nodes.push_back(
            old_polygon.nodes[static_cast<std::size_t>(k)]);
      }
    }
    for (int destination = 0; destination < destination_count;
         ++destination) {
      const Polygon& polygon =
          destination_polygons[static_cast<std::size_t>(destination)];
      for (int k = 0; k < polygon.n; ++k) {
        affected_nodes.push_back(polygon.nodes[static_cast<std::size_t>(k)]);
      }
    }
    transfers.insert(
        transfers.end(), pair_transfers.begin(), pair_transfers.end());
    dV_rel_max = std::max(dV_rel_max, pair_dV_rel_max);
    corner_defect_max =
        std::max(corner_defect_max, pair_corner_defect_max);
  }

  std::sort(affected_nodes.begin(), affected_nodes.end());
  affected_nodes.erase(
      std::unique(affected_nodes.begin(), affected_nodes.end()),
      affected_nodes.end());

  // Consult-24 §3.1: the closure contains the union of the complete old and
  // new dual stars of every affected node. Unchanged corners in those cells
  // are identity transfers.
  std::vector<std::uint8_t> affected_node_mask(
      static_cast<std::size_t>(n_nodes), 0U);
  for (const int node : affected_nodes) {
    affected_node_mask[static_cast<std::size_t>(node)] = 1U;
  }
  for (const PairPlan& plan : plans) {
    if (plan.skip_merge) {
      continue;
    }
    staged_output_cell[static_cast<std::size_t>(plan.survivor)] = 1U;
    if (plan.split_part_count >= 2) {
      staged_output_cell[static_cast<std::size_t>(plan.absorbed)] = 1U;
    }
  }
  std::vector<std::uint8_t> closure_cell(
      static_cast<std::size_t>(n_cells), 0U);
  std::vector<std::uint8_t> star_cell(
      static_cast<std::size_t>(n_cells), 0U);
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    const int off = csr_offsets[cell];
    if (hydro_active_old[cell] != 0) {
      const int old_nverts = active_nverts(cell_nverts, cell_index);
      for (int k = 0; k < old_nverts; ++k) {
        const int node = csr_indices[static_cast<std::size_t>(off + k)];
        if (affected_node_mask[static_cast<std::size_t>(node)] != 0U) {
          closure_cell[cell] = 1U;
          break;
        }
      }
    }
    if (hydro_active[cell] != 0) {
      const int new_nverts = active_nverts(cell_nverts_new, cell_index);
      for (int k = 0; k < new_nverts; ++k) {
        const int node =
            csr_indices_new[static_cast<std::size_t>(off + k)];
        if (affected_node_mask[static_cast<std::size_t>(node)] != 0U) {
          closure_cell[cell] = 1U;
          star_cell[cell] = 1U;
          break;
        }
      }
    }
  }
  std::vector<std::uint8_t> positivity_audit_cell(
      static_cast<std::size_t>(n_cells), 0U);
  std::vector<std::uint8_t> pure_star_cell(
      static_cast<std::size_t>(n_cells), 0U);
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (hydro_active[cell] != 0 &&
        (staged_output_cell[cell] != 0U || star_cell[cell] != 0U)) {
      positivity_audit_cell[cell] = 1U;
    }
    if (star_cell[cell] != 0U && staged_output_cell[cell] == 0U &&
        reused_third_cell[cell] == 0U && tombstone_cell[cell] == 0U) {
      pure_star_cell[cell] = 1U;
      mass_new[cell] = mass[cell];
      rho_new[cell] = rho[cell];
      vol_new[cell] = vol[cell];
      ee_new[cell] = ee[cell];
      ei_new[cell] = ei[cell];
    }
  }
  const auto reject_staged_positivity = [&](const int cell_index,
                                             const char* site) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    std::ostringstream detail;
    detail << std::setprecision(17)
           << "[row-merge] positivity_broken detail cell=" << cell_index
           << " site=" << site
           << " staged_ee=" << ee_new[cell]
           << " staged_ei=" << ei_new[cell]
           << " staged_mass=" << mass_new[cell]
           << " live_mass=" << mass[cell]
           << " hydro_active=" << static_cast<int>(hydro_active[cell])
           << " live_active=" << static_cast<int>(hydro_active_old[cell])
           << " reused_slot="
           << static_cast<int>(reused_third_cell[cell])
           << " tombstone=" << static_cast<int>(tombstone_cell[cell])
           << " star=" << static_cast<int>(star_cell[cell])
           << " third_slot_plans=";
    bool first_third_slot_plan = true;
    for (const PairPlan& plan : plans) {
      if (plan.third_cell != cell_index) {
        continue;
      }
      if (!first_third_slot_plan) {
        detail << ",";
      }
      detail << plan.row_cell << "(skip_merge="
             << static_cast<int>(plan.skip_merge) << ")";
      first_third_slot_plan = false;
    }
    core::log_warning(detail.str());
    return reject(result, "positivity_broken", cell_index);
  };
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (positivity_audit_cell[cell] == 0U ||
        staged_output_cell[cell] != 0U) {
      continue;
    }
    if (hydro_active_old[cell] == 0 || !(mass[cell] > 0.0) ||
        !std::isfinite(mass[cell]) || !std::isfinite(ee[cell]) ||
        !std::isfinite(ei[cell]) || ee[cell] < 0.0 || ei[cell] < 0.0) {
      return reject_staged_positivity(cell_index, "audit_nonoutput_state");
    }
    const int nverts = active_nverts(cell_nverts_new, cell_index);
    const int off = csr_offsets[cell];
    for (int k = 0; k < nverts; ++k) {
      const std::size_t corner =
          cell * static_cast<std::size_t>(kStorageSlots) +
          static_cast<std::size_t>(k);
      const int node =
          csr_indices_new[static_cast<std::size_t>(off + k)];
      const double identity_mass = corner_mass[corner];
      const double speed2 =
          v_r[static_cast<std::size_t>(node)] *
              v_r[static_cast<std::size_t>(node)] +
          v_z[static_cast<std::size_t>(node)] *
              v_z[static_cast<std::size_t>(node)];
      if (!std::isfinite(identity_mass) || identity_mass < 0.0 ||
          !std::isfinite(speed2)) {
        return reject_staged_positivity(cell_index, "audit_nonoutput_corner");
      }
      corner_momentum_r_new[corner] =
          identity_mass * v_r[static_cast<std::size_t>(node)];
      corner_momentum_z_new[corner] =
          identity_mass * v_z[static_cast<std::size_t>(node)];
      corner_electron_internal_new[corner] = identity_mass * ee[cell];
      corner_ion_internal_new[corner] = identity_mass * ei[cell];
      corner_kinetic_cons_new[corner] = 0.5 * identity_mass * speed2;
      transfers.push_back(
          CornerTransfer{corner, corner, node, node, identity_mass});
    }
  }

  // Re-evaluate every transported field from the one transfer record set.
  std::vector<double> operator_mass(corner_mass.size(), 0.0);
  std::vector<double> operator_momentum_r(corner_mass.size(), 0.0);
  std::vector<double> operator_momentum_z(corner_mass.size(), 0.0);
  std::vector<double> operator_electron_internal(corner_mass.size(), 0.0);
  std::vector<double> operator_ion_internal(corner_mass.size(), 0.0);
  std::vector<double> operator_kinetic_cons(corner_mass.size(), 0.0);
  std::vector<std::map<int, double>> nodal_source_mass(
      static_cast<std::size_t>(n_nodes));
  for (const CornerTransfer& transfer : transfers) {
    if (!std::isfinite(transfer.mass) || transfer.mass < 0.0) {
      return reject(result, "signed_transfer", failing_cell);
    }
    const std::size_t source_cell =
        transfer.source_corner / static_cast<std::size_t>(kStorageSlots);
    const double source_vr =
        v_r[static_cast<std::size_t>(transfer.source_node)];
    const double source_vz =
        v_z[static_cast<std::size_t>(transfer.source_node)];
    const double source_speed2 =
        source_vr * source_vr + source_vz * source_vz;
    operator_mass[transfer.destination_corner] += transfer.mass;
    operator_momentum_r[transfer.destination_corner] +=
        transfer.mass * source_vr;
    operator_momentum_z[transfer.destination_corner] +=
        transfer.mass * source_vz;
    operator_electron_internal[transfer.destination_corner] +=
        transfer.mass * ee[source_cell];
    operator_ion_internal[transfer.destination_corner] +=
        transfer.mass * ei[source_cell];
    operator_kinetic_cons[transfer.destination_corner] +=
        0.5 * transfer.mass * source_speed2;
    const std::size_t destination_cell =
        transfer.destination_corner /
        static_cast<std::size_t>(kStorageSlots);
    if (staged_output_cell[destination_cell] != 0U) {
      nodal_source_mass[static_cast<std::size_t>(transfer.destination_node)]
                       [transfer.source_node] += transfer.mass;
    }
  }

  std::vector<double> nodal_mass_new(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> nodal_momentum_r_new(
      static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> nodal_momentum_z_new(
      static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> nodal_kinetic_cons(
      static_cast<std::size_t>(n_nodes), 0.0);
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (positivity_audit_cell[cell] == 0U) {
      continue;
    }
    const int nverts = active_nverts(cell_nverts_new, cell_index);
    const int off = csr_offsets[cell];
    for (int k = 0; k < nverts; ++k) {
      const std::size_t corner =
          cell * static_cast<std::size_t>(kStorageSlots) +
          static_cast<std::size_t>(k);
      const int node =
          csr_indices_new[static_cast<std::size_t>(off + k)];
      if (!relative_identity(operator_mass[corner], corner_mass_new[corner])) {
        return reject(result, "corner_defect", cell_index);
      }
      if (!relative_identity(operator_momentum_r[corner],
                             corner_momentum_r_new[corner]) ||
          !relative_identity(operator_momentum_z[corner],
                             corner_momentum_z_new[corner])) {
        return reject(result, "momentum_defect", cell_index);
      }
      if (!relative_identity(operator_electron_internal[corner],
                             corner_electron_internal_new[corner]) ||
          !relative_identity(operator_ion_internal[corner],
                             corner_ion_internal_new[corner]) ||
          !relative_identity(operator_kinetic_cons[corner],
                             corner_kinetic_cons_new[corner])) {
        return reject(result, "energy_defect", cell_index);
      }
    }
  }

  // Consult-24 §3.4: gather every active corner in the complete dual star of
  // each affected node. Transaction corners carry transported fields; every
  // other corner carries its identity mass at the old nodal velocity.
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (hydro_active[cell] == 0) {
      continue;
    }
    const int nverts = active_nverts(cell_nverts_new, cell_index);
    const int off = csr_offsets[cell];
    for (int k = 0; k < nverts; ++k) {
      const std::size_t corner =
          cell * static_cast<std::size_t>(kStorageSlots) +
          static_cast<std::size_t>(k);
      const int node =
          csr_indices_new[static_cast<std::size_t>(off + k)];
      if (affected_node_mask[static_cast<std::size_t>(node)] == 0U) {
        continue;
      }
      const bool transported = staged_output_cell[cell] != 0U;
      const double corner_mass_value =
          transported ? corner_mass_new[corner] : corner_mass[corner];
      const double corner_momentum_r =
          transported
              ? corner_momentum_r_new[corner]
              : corner_mass_value * v_r[static_cast<std::size_t>(node)];
      const double corner_momentum_z =
          transported
              ? corner_momentum_z_new[corner]
              : corner_mass_value * v_z[static_cast<std::size_t>(node)];
      const double old_speed2 =
          v_r[static_cast<std::size_t>(node)] *
              v_r[static_cast<std::size_t>(node)] +
          v_z[static_cast<std::size_t>(node)] *
              v_z[static_cast<std::size_t>(node)];
      const double corner_kinetic_cons =
          transported ? corner_kinetic_cons_new[corner]
                      : 0.5 * corner_mass_value * old_speed2;
      nodal_mass_new[static_cast<std::size_t>(node)] +=
          corner_mass_value;
      nodal_momentum_r_new[static_cast<std::size_t>(node)] +=
          corner_momentum_r;
      nodal_momentum_z_new[static_cast<std::size_t>(node)] +=
          corner_momentum_z;
      nodal_kinetic_cons[static_cast<std::size_t>(node)] +=
          corner_kinetic_cons;
      if (!transported) {
        nodal_source_mass[static_cast<std::size_t>(node)][node] +=
            corner_mass_value;
      }
    }
  }

  // Reconstruct p/m on the full star and evaluate Q with the non-negative
  // pairwise weighted-variance identity over the same complete support.
  std::vector<double> nodal_q(static_cast<std::size_t>(n_nodes), 0.0);
  double delta_u_max = 0.0;
  double axis_pr_absorbed = 0.0;
  long double axis_pr_projection_total = 0.0L;
  for (const int node : affected_nodes) {
    const double new_node_mass =
        nodal_mass_new[static_cast<std::size_t>(node)];
    double target_pr =
        nodal_momentum_r_new[static_cast<std::size_t>(node)];
    const double target_pz =
        nodal_momentum_z_new[static_cast<std::size_t>(node)];
    const double kinetic_cons =
        nodal_kinetic_cons[static_cast<std::size_t>(node)];
    if (!(new_node_mass > 0.0) || !std::isfinite(new_node_mass) ||
        !std::isfinite(target_pr) || !std::isfinite(target_pz)) {
      return reject(result, "momentum_defect", failing_cell);
    }
    if (!std::isfinite(kinetic_cons) || kinetic_cons < 0.0) {
      return reject(result, "variance_negative", failing_cell);
    }
    double source_mass_sum = 0.0;
    const auto& source_map =
        nodal_source_mass[static_cast<std::size_t>(node)];
    for (const auto& source : source_map) {
      source_mass_sum += source.second;
    }
    const double eps64 =
        64.0 * std::numeric_limits<double>::epsilon();
    if (!relative_identity(source_mass_sum, new_node_mass)) {
      return reject(result, "corner_defect", failing_cell);
    }
    double axis_pr = 0.0;
    if (x_r[static_cast<std::size_t>(node)] == 0.0) {
      axis_pr = target_pr;
      target_pr = 0.0;
      axis_pr_absorbed += std::abs(axis_pr);
      axis_pr_projection_total += static_cast<long double>(axis_pr);
    }
    v_r_new[static_cast<std::size_t>(node)] =
        x_r[static_cast<std::size_t>(node)] == 0.0
            ? 0.0
            : target_pr / new_node_mass;
    v_z_new[static_cast<std::size_t>(node)] = target_pz / new_node_mass;
    const double recovered_pr =
        new_node_mass * v_r_new[static_cast<std::size_t>(node)];
    const double recovered_pz =
        new_node_mass * v_z_new[static_cast<std::size_t>(node)];
    if (!relative_identity(recovered_pr, target_pr) ||
        !relative_identity(recovered_pz, target_pz)) {
      return reject(result, "momentum_defect", failing_cell);
    }
    delta_u_max = std::max(
        delta_u_max,
        std::max(
            std::abs(v_r_new[static_cast<std::size_t>(node)] -
                     v_r[static_cast<std::size_t>(node)]),
            std::abs(v_z_new[static_cast<std::size_t>(node)] -
                     v_z[static_cast<std::size_t>(node)])));
    const double new_speed2 =
        v_r_new[static_cast<std::size_t>(node)] *
            v_r_new[static_cast<std::size_t>(node)] +
        v_z_new[static_cast<std::size_t>(node)] *
            v_z_new[static_cast<std::size_t>(node)];
    const double kinetic_act = 0.5 * new_node_mass * new_speed2;
    const double q_difference = kinetic_cons - kinetic_act;
    const double q_scale = std::max(
        {std::abs(kinetic_cons), std::abs(kinetic_act),
         std::numeric_limits<double>::min()});
    if (!std::isfinite(q_difference) ||
        q_difference < -eps64 * q_scale) {
      return reject(result, "variance_negative", failing_cell);
    }
    long double q_variance_accum = 0.0L;
    for (auto first = source_map.begin(); first != source_map.end(); ++first) {
      for (auto second = std::next(first); second != source_map.end();
           ++second) {
        const double delta_vr =
            v_r[static_cast<std::size_t>(first->first)] -
            v_r[static_cast<std::size_t>(second->first)];
        const double delta_vz =
            v_z[static_cast<std::size_t>(first->first)] -
            v_z[static_cast<std::size_t>(second->first)];
        const long double variance_term =
            0.5L * static_cast<long double>(first->second) *
            static_cast<long double>(second->second) /
            static_cast<long double>(new_node_mass) *
            static_cast<long double>(delta_vr * delta_vr +
                                     delta_vz * delta_vz);
        q_variance_accum += variance_term;
      }
    }
    const double q_variance = static_cast<double>(q_variance_accum);
    if (!std::isfinite(q_variance) || q_variance < 0.0) {
      return reject(result, "variance_negative", failing_cell);
    }
    const double q_deposit =
        q_variance + 0.5 * axis_pr * axis_pr / new_node_mass;
    if (std::abs(q_deposit - q_difference) > eps64 * q_scale) {
      return reject(result, "energy_defect", failing_cell);
    }
    nodal_q[static_cast<std::size_t>(node)] = q_deposit;
  }

  // Consult-24 §3.5: deposit Q to ion internal energy with corner-mass weights.
  std::vector<double> corner_q_deposit(corner_mass.size(), 0.0);
  std::vector<double> nodal_q_scattered(
      static_cast<std::size_t>(n_nodes), 0.0);
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (star_cell[cell] == 0U) {
      continue;
    }
    const int nverts = active_nverts(cell_nverts_new, cell_index);
    const int off = csr_offsets[cell];
    for (int k = 0; k < nverts; ++k) {
      const int node =
          csr_indices_new[static_cast<std::size_t>(off + k)];
      if (affected_node_mask[static_cast<std::size_t>(node)] == 0U) {
        continue;
      }
      const std::size_t corner =
          cell * static_cast<std::size_t>(kStorageSlots) +
          static_cast<std::size_t>(k);
      const double q_share =
          corner_mass_new[corner] /
          nodal_mass_new[static_cast<std::size_t>(node)] *
          nodal_q[static_cast<std::size_t>(node)];
      if (!std::isfinite(q_share) || q_share < 0.0) {
        return reject_staged_positivity(cell_index, "qdeposit");
      }
      corner_q_deposit[corner] = q_share;
      nodal_q_scattered[static_cast<std::size_t>(node)] += q_share;
    }
  }
  long double q_total_accum = 0.0L;
  for (const int node : affected_nodes) {
    if (!relative_identity(
            nodal_q_scattered[static_cast<std::size_t>(node)],
            nodal_q[static_cast<std::size_t>(node)])) {
      return reject(result, "energy_defect", failing_cell);
    }
    q_total_accum += nodal_q[static_cast<std::size_t>(node)];
  }

  long double old_mass_total = 0.0L;
  long double old_momentum_r_total = 0.0L;
  long double old_momentum_z_total = 0.0L;
  long double old_electron_internal_total = 0.0L;
  long double old_ion_internal_total = 0.0L;
  long double kinetic_old_total = 0.0L;
  long double momentum_r_scale = std::numeric_limits<double>::min();
  long double momentum_z_scale = std::numeric_limits<double>::min();
  // Before-state audit: every old corner of every cell in the affected-star
  // union carries its original corner mass at v_old.
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (closure_cell[cell] == 0U || hydro_active_old[cell] == 0) {
      continue;
    }
    const int nverts = active_nverts(cell_nverts, cell_index);
    const int off = csr_offsets[cell];
    for (int k = 0; k < nverts; ++k) {
      const std::size_t corner =
          cell * static_cast<std::size_t>(kStorageSlots) +
          static_cast<std::size_t>(k);
      const int node = csr_indices[static_cast<std::size_t>(off + k)];
      const double source_mass = corner_mass[corner];
      const double source_pr =
          source_mass * v_r[static_cast<std::size_t>(node)];
      const double source_pz =
          source_mass * v_z[static_cast<std::size_t>(node)];
      const double source_speed2 =
          v_r[static_cast<std::size_t>(node)] *
              v_r[static_cast<std::size_t>(node)] +
          v_z[static_cast<std::size_t>(node)] *
              v_z[static_cast<std::size_t>(node)];
      old_mass_total += source_mass;
      old_momentum_r_total += source_pr;
      old_momentum_z_total += source_pz;
      old_electron_internal_total += source_mass * ee[cell];
      old_ion_internal_total += source_mass * ei[cell];
      kinetic_old_total += 0.5L * source_mass * source_speed2;
      momentum_r_scale += std::abs(source_pr);
      momentum_z_scale += std::abs(source_pz);
    }
  }

  long double new_mass_total = 0.0L;
  long double new_momentum_r_total = 0.0L;
  long double new_momentum_z_total = 0.0L;
  long double new_electron_internal_total = 0.0L;
  long double new_ion_internal_transported_total = 0.0L;
  long double new_ion_internal_total = 0.0L;
  long double new_kinetic_cons_total = 0.0L;
  long double kinetic_new_total = 0.0L;
  // After-state audit: exactly the final transaction outputs and Q-star cells.
  // Use v_new at affected nodes and v_old at every other corner.
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (positivity_audit_cell[cell] == 0U) {
      continue;
    }
    double cell_mass = 0.0;
    double cell_electron_internal = 0.0;
    double cell_ion_internal_transported = 0.0;
    double cell_q = 0.0;
    const int nverts = active_nverts(cell_nverts_new, cell_index);
    const int off = csr_offsets[cell];
    for (int k = 0; k < nverts; ++k) {
      const std::size_t corner =
          cell * static_cast<std::size_t>(kStorageSlots) +
          static_cast<std::size_t>(k);
      const int node =
          csr_indices_new[static_cast<std::size_t>(off + k)];
      const double corner_mass_value = corner_mass_new[corner];
      const bool node_affected =
          affected_node_mask[static_cast<std::size_t>(node)] != 0U;
      const double output_vr =
          node_affected ? v_r_new[static_cast<std::size_t>(node)]
                        : v_r[static_cast<std::size_t>(node)];
      const double output_vz =
          node_affected ? v_z_new[static_cast<std::size_t>(node)]
                        : v_z[static_cast<std::size_t>(node)];
      const double output_speed2 =
          output_vr * output_vr + output_vz * output_vz;
      if (!std::isfinite(corner_mass_value) || corner_mass_value < 0.0 ||
          !std::isfinite(corner_electron_internal_new[corner]) ||
          corner_electron_internal_new[corner] < 0.0 ||
          !std::isfinite(corner_ion_internal_new[corner]) ||
          corner_ion_internal_new[corner] < 0.0 ||
          !std::isfinite(output_speed2)) {
        return reject_staged_positivity(cell_index, "audit_output_corner");
      }
      const double output_pr = corner_mass_value * output_vr;
      const double output_pz = corner_mass_value * output_vz;
      cell_mass += corner_mass_value;
      cell_electron_internal += corner_electron_internal_new[corner];
      cell_ion_internal_transported += corner_ion_internal_new[corner];
      cell_q += corner_q_deposit[corner];
      new_mass_total += corner_mass_value;
      new_momentum_r_total += output_pr;
      new_momentum_z_total += output_pz;
      new_electron_internal_total += corner_electron_internal_new[corner];
      new_ion_internal_transported_total +=
          corner_ion_internal_new[corner];
      new_ion_internal_total +=
          corner_ion_internal_new[corner] + corner_q_deposit[corner];
      new_kinetic_cons_total += corner_kinetic_cons_new[corner];
      kinetic_new_total += 0.5L * corner_mass_value * output_speed2;
      momentum_r_scale += std::abs(output_pr);
      momentum_z_scale += std::abs(output_pz);
    }
    const double cell_ion_internal =
        cell_ion_internal_transported + cell_q;
    const double staged_mass = mass_new[cell];
    if (cell_mass > 0.0 && std::isfinite(cell_mass)) {
      if (pure_star_cell[cell] != 0U) {
        ei_new[cell] = cell_ion_internal / mass_new[cell];
      } else {
        mass_new[cell] = cell_mass;
        ee_new[cell] = cell_electron_internal / mass_new[cell];
        ei_new[cell] = cell_ion_internal / mass_new[cell];
        rho_new[cell] = mass_new[cell] / vol_new[cell];
      }
    }
    if (!(cell_mass > 0.0) || !std::isfinite(cell_mass) ||
        !relative_identity(cell_mass, staged_mass) ||
        !std::isfinite(cell_electron_internal) ||
        cell_electron_internal < 0.0 || !std::isfinite(cell_ion_internal) ||
        cell_ion_internal < 0.0 || !(vol_new[cell] > 0.0) ||
        !std::isfinite(vol_new[cell])) {
      return reject_staged_positivity(cell_index, "star_commit_totals");
    }
    if (!std::isfinite(ee_new[cell]) || ee_new[cell] < 0.0 ||
        !std::isfinite(ei_new[cell]) || ei_new[cell] < 0.0 ||
        !std::isfinite(rho_new[cell]) || !(rho_new[cell] > 0.0)) {
      return reject_staged_positivity(cell_index, "star_commit_state");
    }
    if (!relative_identity(mass_new[cell] * ee_new[cell],
                           cell_electron_internal) ||
        !relative_identity(mass_new[cell] * ei_new[cell],
                           cell_ion_internal)) {
      return reject(result, "energy_defect", cell_index);
    }
  }

  // Consult-24 §3.6: certify the closed component, not each primal cell.
  const long double mass_residual_accum = new_mass_total - old_mass_total;
  const long double momentum_r_residual_accum =
      new_momentum_r_total + axis_pr_projection_total -
      old_momentum_r_total;
  const long double momentum_z_residual_accum =
      new_momentum_z_total - old_momentum_z_total;
  const long double old_total_energy =
      old_electron_internal_total + old_ion_internal_total +
      kinetic_old_total;
  const long double new_total_energy =
      new_electron_internal_total + new_ion_internal_total +
      kinetic_new_total;
  const long double energy_residual_accum =
      new_total_energy - old_total_energy;
  const long double mass_scale = std::max(
      {std::abs(old_mass_total), std::abs(new_mass_total),
       static_cast<long double>(std::numeric_limits<double>::min())});
  const long double energy_scale = std::max(
      {std::abs(old_total_energy), std::abs(new_total_energy),
       static_cast<long double>(std::numeric_limits<double>::min())});
  const long double eps64 =
      64.0L * std::numeric_limits<double>::epsilon();
  if (std::abs(mass_residual_accum) > eps64 * mass_scale) {
    return reject(result, "corner_defect", failing_cell);
  }
  const long double momentum_r_defect =
      std::abs(momentum_r_residual_accum);
  const long double momentum_z_defect =
      std::abs(momentum_z_residual_accum);
  if (momentum_r_defect > eps64 * momentum_r_scale ||
      momentum_z_defect > eps64 * momentum_z_scale) {
    const bool report_r =
        momentum_r_defect / momentum_r_scale >=
        momentum_z_defect / momentum_z_scale;
    return reject(result, "momentum_defect", failing_cell,
                  report_r ? momentum_r_defect : momentum_z_defect,
                  report_r ? momentum_r_scale : momentum_z_scale);
  }
  if (std::abs(energy_residual_accum) > eps64 * energy_scale) {
    return reject(result, "energy_defect", failing_cell);
  }
  const long double electron_scale = std::max(
      {std::abs(old_electron_internal_total),
       std::abs(new_electron_internal_total),
       static_cast<long double>(std::numeric_limits<double>::min())});
  const long double ion_scale = std::max(
      {std::abs(old_ion_internal_total),
       std::abs(new_ion_internal_transported_total),
       static_cast<long double>(std::numeric_limits<double>::min())});
  const long double kinetic_cons_scale = std::max(
      {std::abs(kinetic_old_total), std::abs(new_kinetic_cons_total),
       static_cast<long double>(std::numeric_limits<double>::min())});
  if (std::abs(new_electron_internal_total -
               old_electron_internal_total) > eps64 * electron_scale ||
      std::abs(new_ion_internal_transported_total -
               old_ion_internal_total) > eps64 * ion_scale ||
      std::abs(new_kinetic_cons_total - kinetic_old_total) >
          eps64 * kinetic_cons_scale) {
    return reject(result, "energy_defect", failing_cell);
  }
  const long double ion_deposit =
      new_ion_internal_total - old_ion_internal_total;
  const long double q_scale = std::max(
      {std::abs(ion_deposit), std::abs(q_total_accum),
       static_cast<long double>(std::numeric_limits<double>::min())});
  if (std::abs(ion_deposit - q_total_accum) > eps64 * q_scale) {
    return reject(result, "energy_defect", failing_cell);
  }
  const long double delta_kinetic_total_accum =
      kinetic_new_total - kinetic_old_total;
  if (std::abs(delta_kinetic_total_accum + q_total_accum) >
      eps64 * energy_scale) {
    return reject(result, "energy_defect", failing_cell);
  }
  const double delta_kinetic_total =
      static_cast<double>(delta_kinetic_total_accum);
  const double q_total = static_cast<double>(q_total_accum);
  const double mass_residual = static_cast<double>(mass_residual_accum);
  const double momentum_r_residual =
      static_cast<double>(momentum_r_residual_accum);
  const double momentum_z_residual =
      static_cast<double>(momentum_z_residual_accum);
  const double total_energy_residual =
      static_cast<double>(energy_residual_accum);

  for (const PairPlan& plan : plans) {
    if (plan.keep_union || plan.skip_merge) {
      continue;
    }
    const int survivor_orientation =
        orientation_sign[static_cast<std::size_t>(plan.survivor)];
    const Polygon emitted_a = cell_polygon(
        plan.survivor, csr_offsets, csr_indices_new, cell_nverts_new,
        x_r, x_z);
    const Polygon emitted_b = cell_polygon(
        plan.absorbed, csr_offsets, csr_indices_new, cell_nverts_new,
        x_r, x_z);
    if (orientation_sign_new[static_cast<std::size_t>(plan.survivor)] !=
            survivor_orientation ||
        orientation_sign_new[static_cast<std::size_t>(plan.absorbed)] !=
            survivor_orientation) {
      return reject(result, "split_edge_multiset", plan.row_cell);
    }
    if (plan.split_part_count == 2 &&
        !split_edge_multiset_matches(
            plan.union_polygon, emitted_a, emitted_b,
            plan.chord_u, plan.chord_v)) {
      return reject(result, "split_edge_multiset", plan.row_cell);
    }
    if (plan.split_part_count == 3) {
      const Polygon emitted_c = cell_polygon(
          plan.third_cell, csr_offsets, csr_indices_new, cell_nverts_new,
          x_r, x_z);
      if (orientation_sign_new[static_cast<std::size_t>(plan.third_cell)] !=
              survivor_orientation ||
          !three_way_split_edge_multiset_matches(
              plan.union_polygon, emitted_a, emitted_b, emitted_c,
              plan.chord_u, plan.chord_v,
              plan.chord2_u, plan.chord2_v)) {
        return reject(result, "split_edge_multiset", plan.row_cell);
      }
    }
  }

  for (int cell = 0; cell < n_cells; ++cell) {
    if (hydro_active[static_cast<std::size_t>(cell)] == 0) {
      continue;
    }
    const int nverts = active_nverts(cell_nverts_new, cell);
    const int off = csr_offsets[static_cast<std::size_t>(cell)];
    for (int local_edge = 0; local_edge < nverts; ++local_edge) {
      const int node0 = csr_indices_new[
          static_cast<std::size_t>(off + local_edge)];
      const int node1 = csr_indices_new[static_cast<std::size_t>(
          off + (local_edge + 1) % nverts)];
      const auto found =
          boundary_edge_install.find(undirected_edge(node0, node1));
      if (found == boundary_edge_install.end()) {
        continue;
      }
      if (found->second.cell >= 0) {
        return reject(result, "split_edge_multiset", cell);
      }
      const int local_face =
          local_face_for_polygon_edge(nverts, local_edge);
      TENRYU_ASSERT(local_face >= 0,
                    "row merge could not encode a boundary polygon edge");
      found->second.cell = cell;
      found->second.local_face = local_face;
    }
  }
  std::vector<mesh::UniqueOrientedFace> boundary_faces_new;
  boundary_faces_new.reserve(boundary_edge_install.size());
  for (const auto& [edge, install] : boundary_edge_install) {
    (void)edge;
    if (install.cell < 0 || install.local_face < 0) {
      return reject(result, "split_edge_multiset", failing_cell);
    }
    boundary_faces_new.push_back(mesh::UniqueOrientedFace{
        install.cell, -1, install.local_face, -1, install.bc_tag});
  }

  const auto copy_staged_range = [](double* destination,
                                    const double* source,
                                    const std::size_t count,
                                    const char* message) {
    const cudaError_t error = cudaMemcpy(
        destination, source, count * sizeof(double), cudaMemcpyHostToDevice);
    TENRYU_ASSERT(error == cudaSuccess, message);
  };
  auto& mb_install = *state.mesh.topo.multiblock;
  if (state.hydro_active.empty()) {
    state.hydro_active = hydro_active_old;
  }
  if (state.merge_tombstone.empty()) {
    state.merge_tombstone.assign(static_cast<std::size_t>(n_cells), 0U);
  }
  TENRYU_ASSERT(state.merge_tombstone.size() == static_cast<std::size_t>(n_cells),
                "row merge merge_tombstone size mismatch");
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (reused_third_cell[cell] != 0U) {
      state.merge_tombstone[cell] = 0U;
    } else if (tombstone_cell[cell] != 0U && staged_output_cell[cell] == 0U) {
      state.merge_tombstone[cell] = 1U;
    }
  }
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (pure_star_cell[cell] != 0U) {
      copy_staged_range(
          state.ei.data() + cell, ei_new.data() + cell, 1U,
          "row merge pure-star ei commit cudaMemcpy failed");
      continue;
    }
    if (staged_output_cell[cell] == 0U && tombstone_cell[cell] == 0U) {
      continue;
    }
    copy_staged_range(
        state.rho.data() + cell, rho_new.data() + cell, 1U,
        "row merge rho commit cudaMemcpy failed");
    copy_staged_range(
        state.mass.data() + cell, mass_new.data() + cell, 1U,
        "row merge mass commit cudaMemcpy failed");
    copy_staged_range(
        state.vol.data() + cell, vol_new.data() + cell, 1U,
        "row merge volume commit cudaMemcpy failed");
    copy_staged_range(
        state.ee.data() + cell, ee_new.data() + cell, 1U,
        "row merge ee commit cudaMemcpy failed");
    copy_staged_range(
        state.ei.data() + cell, ei_new.data() + cell, 1U,
        "row merge ei commit cudaMemcpy failed");
    const std::size_t corner_begin =
        cell * static_cast<std::size_t>(kStorageSlots);
    copy_staged_range(
        state.corner_mass.data() + corner_begin,
        corner_mass_new.data() + corner_begin,
        static_cast<std::size_t>(kStorageSlots),
        "row merge corner-mass commit cudaMemcpy failed");
    if (cfg.radiation.enabled) {
      const std::size_t radiation_begin =
          cell * static_cast<std::size_t>(radiation_groups);
      copy_staged_range(
          state.rad_E.data() + radiation_begin,
          rad_E_new.data() + radiation_begin,
          static_cast<std::size_t>(radiation_groups),
          "row merge radiation commit cudaMemcpy failed");
    }
    state.hydro_active[cell] = hydro_active[cell];
    if (staged_output_cell[cell] != 0U) {
      const int off = csr_offsets[cell];
      std::copy_n(
          csr_indices_new.begin() + off, kStorageSlots,
          mb_install.cell_node_csr_indices.begin() + off);
      mb_install.cell_orientation_sign[cell] = orientation_sign_new[cell];
      state.mesh.cell_nverts[cell] = cell_nverts_new[cell];
    }
  }
  state.v_r.copy_from_host(v_r_new);
  state.v_z.copy_from_host(v_z_new);

  mb_install.boundary_faces = std::move(boundary_faces_new);
  for (std::size_t cell = 0; cell < state.merge_tombstone.size(); ++cell) {
    if (state.merge_tombstone[cell] != 0U) {
      state.hydro_active[cell] = 0;
    }
  }
  state.note_hydro_active_host_write();

  mesh::rebuild_runtime_topology_caches(state, cfg);

  result.committed = true;
  result.merged_pairs =
      static_cast<int>(plans.size()) - result.skipped_pairs;
  result.first_bad_cell = -1;
  result.delta_kinetic_total = delta_kinetic_total;
  result.q_total = q_total;
  result.mass_residual = mass_residual;
  result.momentum_r_residual = momentum_r_residual;
  result.momentum_z_residual = momentum_z_residual;
  result.total_energy_residual = total_energy_residual;
  result.reject_reason = nullptr;
  const double kinetic_shift_relative = static_cast<double>(
      delta_kinetic_total_accum /
      std::max(
          std::abs(kinetic_old_total),
          static_cast<long double>(std::numeric_limits<double>::min())));
  std::ostringstream log;
  log << std::setprecision(17)
      << "[row-merge] step=" << state.step
      << " orbit_cells=" << row_cells.size()
      << " pairs=" << plans.size()
      << " kept_union_pairs=" << kept_union_pairs
      << " keep_union_absorbed_ids=";
  bool first_keep_union_absorbed = true;
  for (const PairPlan& plan : plans) {
    if (plan.skip_merge || !plan.keep_union) {
      continue;
    }
    if (!first_keep_union_absorbed) {
      log << ",";
    }
    log << plan.absorbed;
    first_keep_union_absorbed = false;
  }
  log << " skipped_pairs=" << result.skipped_pairs
      << " committed dV_rel_max=" << dV_rel_max
      << " corner_defect_max=" << corner_defect_max
      << " du_max=" << delta_u_max
      << " dK=" << delta_kinetic_total
      << " Q_total=" << q_total
      << " axis_pr=" << axis_pr_absorbed
      << " mass_residual=" << mass_residual
      << " momentum_residual=(" << momentum_r_residual << ","
      << momentum_z_residual << ")"
      << " energy_residual=" << total_energy_residual
      << " K_shift_rel=" << kinetic_shift_relative;
  for (const PairPlan& plan : plans) {
    if (plan.skip_merge) {
      continue;
    }
    if (plan.keep_union) {
      log << " keep_union=" << plan.union_polygon.n;
    } else if (plan.split_part_count == 3) {
      log << " split=(" << plan.half_a.n << "," << plan.half_b.n << ","
          << plan.half_c.n << ")"
          << " chords=(" << plan.chord_u << "," << plan.chord_v << ";"
          << plan.chord2_u << "," << plan.chord2_v << ")"
          << " third_cell=" << plan.third_cell;
    } else {
      log << " split=(" << plan.half_a.n << "," << plan.half_b.n << ")"
          << " chord=(" << plan.chord_u << "," << plan.chord_v << ")";
    }
  }
  core::log_warning(log.str());

  std::vector<double> mass_postcommit;
  state.mass.copy_to_host(mass_postcommit);
  const std::vector<std::int8_t> hydro_active_postcommit =
      state.hydro_active;
  TENRYU_ASSERT(
      mass_postcommit.size() == static_cast<std::size_t>(n_cells) &&
          hydro_active_postcommit.size() == static_cast<std::size_t>(n_cells),
      "row merge postcommit invariant sweep size mismatch");
  for (int cell_index = 0; cell_index < n_cells; ++cell_index) {
    const std::size_t cell = static_cast<std::size_t>(cell_index);
    if (hydro_active_postcommit[cell] != 1 ||
        mass_postcommit[cell] > 0.0) {
      continue;
    }

    const char* role = "untouched";
    if (reused_third_cell[cell] != 0U) {
      role = "third-part";
    } else {
      for (const PairPlan& plan : plans) {
        if (!plan.skip_merge && plan.survivor == cell_index) {
          role = "survivor";
          break;
        }
        if (!plan.skip_merge && plan.absorbed == cell_index) {
          role = plan.keep_union ? "keep_union-absorbed"
                                 : "absorbed-dest";
          break;
        }
        if (plan.skip_merge &&
            (plan.row_cell == cell_index || plan.partner == cell_index)) {
          role = "skip";
        }
      }
      if (std::strcmp(role, "untouched") == 0 &&
          pure_star_cell[cell] != 0U) {
        role = "pure-star";
      }
    }

    std::ostringstream invariant;
    invariant << std::setprecision(17)
              << "[row-merge] POSTCOMMIT INVARIANT BROKEN cell="
              << cell_index << " mass=" << mass_postcommit[cell]
              << " role=" << role
              << " masks{staged_output="
              << static_cast<int>(staged_output_cell[cell])
              << " tombstone_cell="
              << static_cast<int>(tombstone_cell[cell])
              << " reused_third_cell="
              << static_cast<int>(reused_third_cell[cell])
              << " pure_star_cell="
              << static_cast<int>(pure_star_cell[cell])
              << " star_cell=" << static_cast<int>(star_cell[cell])
              << "}";
    bool found_plan = false;
    for (const PairPlan& plan : plans) {
      if (plan.row_cell != cell_index && plan.partner != cell_index &&
          plan.survivor != cell_index && plan.absorbed != cell_index &&
          plan.third_cell != cell_index) {
        continue;
      }
      invariant << (found_plan ? ", plan{" : " plan{")
                << "row_cell=" << plan.row_cell
                << " partner=" << plan.partner
                << " keep_union=" << static_cast<int>(plan.keep_union)
                << " skip=" << static_cast<int>(plan.skip_merge) << "}";
      found_plan = true;
    }
    const std::string message = invariant.str();
    core::log_warning(message);
    TENRYU_ASSERT(false, message);
  }
  return result;
}

}  // namespace tenryu::hydro
