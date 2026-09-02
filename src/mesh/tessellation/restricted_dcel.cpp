#include "mesh/tessellation/voronoi_dual.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "mesh/tessellation/exact_types.hpp"

namespace tenryu::mesh::tess {
namespace {

int orient_edge_site(const HomogeneousPoint& first,
                     const HomogeneousPoint& second,
                     const Site& site) {
  const ApproxBound direction_r_approx = approx_sub(
      approx_mul(approx_of_expansion(second.x),
                 approx_of_expansion(first.w)),
      approx_mul(approx_of_expansion(first.x),
                 approx_of_expansion(second.w)));
  const ApproxBound direction_z_approx = approx_sub(
      approx_mul(approx_of_expansion(second.y),
                 approx_of_expansion(first.w)),
      approx_mul(approx_of_expansion(first.y),
                 approx_of_expansion(second.w)));
  const ApproxBound site_r_approx = approx_sub(
      approx_mul(approx_of_expansion(first.w),
                 ApproxBound{site.r, 0.0}),
      approx_of_expansion(first.x));
  const ApproxBound site_z_approx = approx_sub(
      approx_mul(approx_of_expansion(first.w),
                 ApproxBound{site.z, 0.0}),
      approx_of_expansion(first.y));
  const ApproxBound determinant_approx = approx_sub(
      approx_mul(direction_r_approx, site_z_approx),
      approx_mul(direction_z_approx, site_r_approx));
  const ApproxSign sign_approx = approx_sign(determinant_approx);
  if (sign_approx.certain) {
    return sign_approx.sign;
  }

  const Expansion direction_r = expansion_difference(
      expansion_product(second.x, first.w),
      expansion_product(first.x, second.w));
  const Expansion direction_z = expansion_difference(
      expansion_product(second.y, first.w),
      expansion_product(first.y, second.w));
  const Expansion site_r = expansion_difference(
      expansion_scale(first.w, site.r), first.x);
  const Expansion site_z = expansion_difference(
      expansion_scale(first.w, site.z), first.y);
  return expansion_sign(
      expansion_cross(direction_r, direction_z, site_r, site_z));
}

RestrictedHalfEdgeId append_half_edge(RestrictedDcel& dcel,
                                      const RestrictedHalfEdge& edge) {
  TENRYU_ASSERT(
      dcel.half_edges.size() <= static_cast<std::size_t>(
                                     std::numeric_limits<RestrictedHalfEdgeId>::max()),
      "restricted half-edge count exceeds RestrictedHalfEdgeId range");
  const RestrictedHalfEdgeId id =
      static_cast<RestrictedHalfEdgeId>(dcel.half_edges.size());
  dcel.half_edges.push_back(edge);
  return id;
}

void append_twin_pair(RestrictedDcel& dcel,
                      const RestrictedNodeId first_origin,
                      const RestrictedNodeId second_origin,
                      const SiteId first_face,
                      const SiteId second_face,
                      const DomainSegmentId boundary_segment) {
  TENRYU_ASSERT(first_origin != second_origin,
                "restricted edge requires distinct endpoints");
  const RestrictedHalfEdgeId first = append_half_edge(
      dcel, RestrictedHalfEdge{first_origin, kInvalidId, kInvalidId,
                               first_face, boundary_segment});
  const RestrictedHalfEdgeId second = append_half_edge(
      dcel, RestrictedHalfEdge{second_origin, first, kInvalidId,
                               second_face, boundary_segment});
  dcel.half_edges[first].twin = second;
}

// Exact-match index for the unique outgoing half-edge of (face, origin).
// Buckets per face with origin-sorted entries reproduce the former
// whole-array sort + lower_bound lookups result-identically: every query
// is a unique exact match (duplicates fail the same assert below).
struct FaceOutgoingIndex {
  std::vector<RestrictedHalfEdgeId> offsets;  // size = face_count + 1
  std::vector<std::pair<RestrictedNodeId, RestrictedHalfEdgeId>> entries;
};

FaceOutgoingIndex build_face_outgoing_index(const RestrictedDcel& dcel,
                                            const bool check_twins) {
  const SiteId face_count = dcel.exterior_face + 1;
  FaceOutgoingIndex index;
  index.offsets.assign(static_cast<std::size_t>(face_count) + 1, 0);
  const RestrictedHalfEdgeId edge_count =
      static_cast<RestrictedHalfEdgeId>(dcel.half_edges.size());
  for (RestrictedHalfEdgeId edge = 0; edge < edge_count; ++edge) {
    const RestrictedHalfEdge& half_edge = dcel.half_edges[edge];
    if (check_twins) {
      TENRYU_ASSERT(half_edge.twin >= 0 && half_edge.twin < edge_count &&
                        dcel.half_edges[half_edge.twin].twin == edge,
                    "twin_missing");
    }
    TENRYU_ASSERT(half_edge.left_face >= 0 &&
                      half_edge.left_face < face_count,
                  "restricted half-edge face is out of range");
    ++index.offsets[static_cast<std::size_t>(half_edge.left_face) + 1];
  }
  for (std::size_t face = 1; face < index.offsets.size(); ++face) {
    index.offsets[face] += index.offsets[face - 1];
  }
  index.entries.resize(dcel.half_edges.size());
  std::vector<RestrictedHalfEdgeId> cursor(index.offsets.begin(),
                                           index.offsets.end() - 1);
  for (RestrictedHalfEdgeId edge = 0; edge < edge_count; ++edge) {
    const RestrictedHalfEdge& half_edge = dcel.half_edges[edge];
    index.entries[static_cast<std::size_t>(
        cursor[static_cast<std::size_t>(half_edge.left_face)]++)] =
        std::make_pair(half_edge.origin, edge);
  }
  for (SiteId face = 0; face < face_count; ++face) {
    const auto first =
        index.entries.begin() + index.offsets[static_cast<std::size_t>(face)];
    const auto last = index.entries.begin() +
                      index.offsets[static_cast<std::size_t>(face) + 1];
    std::sort(first, last);
    for (auto entry = first + (first == last ? 0 : 1); entry < last;
         ++entry) {
      TENRYU_ASSERT(entry->first != (entry - 1)->first,
                    "restricted face has multiple outgoing edges at one node");
    }
  }
  return index;
}

RestrictedHalfEdgeId find_outgoing(const FaceOutgoingIndex& index,
                                   const SiteId face,
                                   const RestrictedNodeId origin,
                                   const char* missing_message) {
  const auto first =
      index.entries.begin() + index.offsets[static_cast<std::size_t>(face)];
  const auto last =
      index.entries.begin() + index.offsets[static_cast<std::size_t>(face) + 1];
  const auto entry = std::lower_bound(
      first, last, origin,
      [](const std::pair<RestrictedNodeId, RestrictedHalfEdgeId>& lhs,
         const RestrictedNodeId lookup) { return lhs.first < lookup; });
  TENRYU_ASSERT(entry != last && entry->first == origin, missing_message);
  return entry->second;
}

void stitch_face_cycles(RestrictedDcel& dcel,
                        double* sort_ms = nullptr,
                        double* link_ms = nullptr) {
  std::chrono::steady_clock::time_point sort_start;
  if (sort_ms != nullptr) {
    sort_start = std::chrono::steady_clock::now();
  }
  const FaceOutgoingIndex index =
      build_face_outgoing_index(dcel, /*check_twins=*/true);
  if (sort_ms != nullptr) {
    *sort_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - sort_start)
                   .count();
  }
  std::chrono::steady_clock::time_point link_start;
  if (link_ms != nullptr) {
    link_start = std::chrono::steady_clock::now();
  }
  for (RestrictedHalfEdgeId edge = 0;
       edge < static_cast<RestrictedHalfEdgeId>(dcel.half_edges.size());
       ++edge) {
    const RestrictedNodeId destination =
        dcel.half_edges[dcel.half_edges[edge].twin].origin;
    dcel.half_edges[edge].next = find_outgoing(
        index, dcel.half_edges[edge].left_face, destination, "twin_missing");
  }
  if (link_ms != nullptr) {
    *link_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - link_start)
                   .count();
  }
}

std::vector<RestrictedNodeId> trace_face(
    const RestrictedDcel& dcel,
    const SiteId face,
    const RestrictedHalfEdgeId first) {
  std::vector<RestrictedNodeId> cycle;
  RestrictedHalfEdgeId edge = first;
  do {
    TENRYU_ASSERT(edge >= 0 &&
                      edge < static_cast<RestrictedHalfEdgeId>(
                                 dcel.half_edges.size()) &&
                      dcel.half_edges[edge].left_face == face,
                  "restricted face cycle is invalid");
    TENRYU_ASSERT(cycle.size() <= dcel.half_edges.size(),
                  "restricted face cycle does not close");
    cycle.push_back(dcel.half_edges[edge].origin);
    edge = dcel.half_edges[edge].next;
  } while (edge != first);
  return cycle;
}

void attach_nodes(RestrictedDcel& dcel,
                  std::vector<RestrictedNode>&& records,
                  const std::vector<RestrictedNodeId>& source) {
  dcel.nodes.clear();
  dcel.nodes.reserve(source.size());
  for (const RestrictedNodeId origin : source) {
    TENRYU_ASSERT(origin >= 0 &&
                      origin < static_cast<RestrictedNodeId>(
                                   records.size()),
                  "restricted node source is out of range");
    dcel.nodes.push_back(std::move(records[origin]));
  }
}

void canonicalize_nodes_borrowed(
    const DelaunayTriangulation& dt,
    RestrictedDcel& dcel,
    const std::vector<RestrictedNode>& records,
    std::vector<RestrictedNodeId>& source_out,
    double* nodes_ms = nullptr,
    double* remap_ms = nullptr,
    double* first_ms = nullptr) {
  std::vector<RestrictedNodeId> remap(records.size(), kInvalidId);
  source_out.clear();
  source_out.reserve(records.size());

  std::chrono::steady_clock::time_point nodes_start;
  if (nodes_ms != nullptr) {
    nodes_start = std::chrono::steady_clock::now();
  }
  for (RestrictedCell& cell : dcel.cells) {
    TENRYU_ASSERT(!cell.cycle.empty(), "cell_nonsimple");
    const auto first = std::min_element(
        cell.cycle.begin(), cell.cycle.end(),
        [&](const RestrictedNodeId lhs, const RestrictedNodeId rhs) {
          return records[lhs].key < records[rhs].key;
        });
    std::rotate(cell.cycle.begin(), first, cell.cycle.end());
    for (const RestrictedNodeId old_node : cell.cycle) {
      if (remap[old_node] != kInvalidId) {
        continue;
      }
      TENRYU_ASSERT(
          source_out.size() <= static_cast<std::size_t>(
                                   std::numeric_limits<RestrictedNodeId>::max()),
          "restricted node count exceeds RestrictedNodeId range");
      remap[old_node] =
          static_cast<RestrictedNodeId>(source_out.size());
      source_out.push_back(old_node);
    }
  }
  if (nodes_ms != nullptr) {
    *nodes_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - nodes_start)
                    .count();
  }

  std::chrono::steady_clock::time_point remap_start;
  if (remap_ms != nullptr) {
    remap_start = std::chrono::steady_clock::now();
  }
  for (RestrictedHalfEdge& edge : dcel.half_edges) {
    TENRYU_ASSERT(remap[edge.origin] != kInvalidId,
                  "restricted half-edge references an unused node");
    edge.origin = remap[edge.origin];
  }
  for (RestrictedCell& cell : dcel.cells) {
    for (RestrictedNodeId& node : cell.cycle) {
      node = remap[node];
    }
    const auto minimum =
        std::min_element(cell.cycle.begin(), cell.cycle.end());
    std::rotate(cell.cycle.begin(), minimum, cell.cycle.end());
  }
  if (remap_ms != nullptr) {
    *remap_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - remap_start)
                    .count();
  }

  std::chrono::steady_clock::time_point first_start;
  if (first_ms != nullptr) {
    first_start = std::chrono::steady_clock::now();
  }
  const FaceOutgoingIndex index =
      build_face_outgoing_index(dcel, /*check_twins=*/false);
  for (RestrictedCell& cell : dcel.cells) {
    cell.first_edge = find_outgoing(
        index, cell.site, cell.cycle.front(),
        "restricted cell has no canonical first edge");
  }
  if (first_ms != nullptr) {
    *first_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - first_start)
                    .count();
  }

  TENRYU_ASSERT(dcel.cells.size() == dt.sites.size(),
                "restricted DCEL cell count mismatch");
}

void canonicalize_nodes(const DelaunayTriangulation& dt,
                        RestrictedDcel& dcel,
                        double* nodes_ms = nullptr,
                        double* remap_ms = nullptr,
                        double* first_ms = nullptr) {
  std::vector<RestrictedNode> records = std::move(dcel.nodes);
  std::vector<RestrictedNodeId> source;
  canonicalize_nodes_borrowed(dt, dcel, records, source, nodes_ms,
                              remap_ms, first_ms);
  attach_nodes(dcel, std::move(records), source);
}

AxisAuditCounters audit_axis_bits(const RestrictedGeometry& geometry) {
  static const bool census_enabled =
      std::getenv("TENRYU_TESS_AXIS") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_AXIS")) == std::string("1");
  AxisAuditCounters counters;
  if (census_enabled) {
    counters.nodes_total = geometry.nodes.size();
  }
  for (const RestrictedNode& node : geometry.nodes) {
    TENRYU_ASSERT(node.constructed[0] >= 0.0, "axis_bits_violation");
    bool node_has_axis_support = false;
    for (const DomainSegmentId support : node.boundary_segments) {
      TENRYU_ASSERT(support >= 0 &&
                        support < static_cast<DomainSegmentId>(
                                      geometry.domain_segments.size()),
                    "boundary_support_missing");
      if (geometry.domain_segments[support].axis) {
        node_has_axis_support = true;
        TENRYU_ASSERT(node.constructed[0] == 0.0 &&
                          !std::signbit(node.constructed[0]),
                      "axis_bits_violation");
      }
    }
    if (census_enabled && node_has_axis_support) {
      ++counters.axis_support_nodes;
    }
  }
  if (census_enabled) {
    for (const DomainSegmentRecord& segment : geometry.domain_segments) {
      if (segment.axis) {
        ++counters.axis_segments;
      }
    }
    for (const BoundaryInterval& interval : geometry.boundary_intervals) {
      if (interval.segment >= 0 &&
          interval.segment < static_cast<DomainSegmentId>(
                                 geometry.domain_segments.size()) &&
          geometry.domain_segments[static_cast<std::size_t>(interval.segment)]
              .axis) {
        ++counters.axis_intervals;
      }
    }
  }
  return counters;
}

struct CarrierCoverageInterval {
  ExactRational lo;
  ExactRational hi;
};

bool carrier_endpoint_lambda(const RestrictedGeometry& geometry,
                             const DomainSegmentRecord& segment,
                             const RestrictedNodeId node_id,
                             ExactRational& lambda) {
  if (node_id < 0 ||
      node_id >= static_cast<RestrictedNodeId>(geometry.nodes.size())) {
    return false;
  }
  const RestrictedNode& node = geometry.nodes[node_id];
  if (node.boundary_param.valid &&
      node.boundary_param.edge == segment.id) {
    lambda = node.boundary_param.lambda;
    return true;
  }
  if (std::find(node.domain_vertices.begin(), node.domain_vertices.end(),
                segment.start_vertex) != node.domain_vertices.end()) {
    lambda = make_exact_rational(Expansion{0.0}, Expansion{1.0});
    return true;
  }
  if (std::find(node.domain_vertices.begin(), node.domain_vertices.end(),
                segment.end_vertex) != node.domain_vertices.end()) {
    lambda = make_exact_rational(Expansion{1.0}, Expansion{1.0});
    return true;
  }
  return false;
}

void check_carrier_coverage(RestrictedGeometry& geometry) {
  static const bool timing_enabled =
      std::getenv("TENRYU_TESS_TIMING") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_TIMING")) == std::string("1");
  std::chrono::steady_clock::time_point carrier_start;
  if (timing_enabled) {
    carrier_start = std::chrono::steady_clock::now();
  }
  std::vector<std::vector<CarrierCoverageInterval>> intervals_by_edge(
      geometry.domain_segments.size());
  for (const BoundaryInterval& interval : geometry.boundary_intervals) {
    if (interval.segment < 0 ||
        interval.segment >= static_cast<DomainSegmentId>(
                                geometry.domain_segments.size())) {
      geometry.carrier_coverage_ok = false;
      continue;
    }
    const DomainSegmentRecord& segment =
        geometry.domain_segments[static_cast<std::size_t>(interval.segment)];
    ExactRational lo;
    ExactRational hi;
    if (!carrier_endpoint_lambda(geometry, segment, interval.endpoint0, lo) ||
        !carrier_endpoint_lambda(geometry, segment, interval.endpoint1, hi)) {
      geometry.carrier_coverage_ok = false;
      continue;
    }
    intervals_by_edge[static_cast<std::size_t>(interval.segment)].push_back(
        CarrierCoverageInterval{std::move(lo), std::move(hi)});
  }

  const ExactRational zero =
      make_exact_rational(Expansion{0.0}, Expansion{1.0});
  const ExactRational one =
      make_exact_rational(Expansion{1.0}, Expansion{1.0});
  for (std::vector<CarrierCoverageInterval>& intervals : intervals_by_edge) {
    if (intervals.empty()) {
      geometry.carrier_coverage_ok = false;
      continue;
    }
    std::stable_sort(
        intervals.begin(), intervals.end(),
        [](const CarrierCoverageInterval& lhs,
           const CarrierCoverageInterval& rhs) {
          const int lo_order = exact_rational_compare(lhs.lo, rhs.lo);
          if (lo_order != 0) {
            return lo_order < 0;
          }
          return exact_rational_compare(lhs.hi, rhs.hi) < 0;
        });
    for (std::size_t index = 0; index < intervals.size(); ++index) {
      if (exact_rational_compare(intervals[index].lo,
                                 intervals[index].hi) >= 0) {
        geometry.carrier_coverage_ok = false;
      }
      if (index == 0 &&
          exact_rational_compare(intervals[index].lo, zero) != 0) {
        geometry.carrier_coverage_ok = false;
      }
      if (index > 0 &&
          exact_rational_compare(intervals[index - 1].hi,
                                 intervals[index].lo) != 0) {
        geometry.carrier_coverage_ok = false;
      }
    }
    if (exact_rational_compare(intervals.back().hi, one) != 0) {
      geometry.carrier_coverage_ok = false;
    }
  }
  if (timing_enabled) {
    const double carrier_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - carrier_start)
            .count();
    std::fprintf(stderr,
                 "[tess_timing_carrier] ms=%.1f n_intervals=%zu\n",
                 carrier_ms, geometry.boundary_intervals.size());
  }
}

struct BorrowedAssembly {
  RestrictedDcel dcel;
  std::vector<RestrictedNodeId> source;
};

BorrowedAssembly assemble_restricted_dcel_borrowed(
    const DelaunayTriangulation& dt,
    const RestrictedGeometry& geometry,
    AxisAuditCounters* axis_counters) {
  static const bool timing_enabled =
      std::getenv("TENRYU_TESS_TIMING") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_TIMING")) == std::string("1");
  double audit_ms = 0.0;
  double interior_ms = 0.0;
  double bnd_ms = 0.0;
  double stitch_sort_ms = 0.0;
  double stitch_link_ms = 0.0;
  double faces_ms = 0.0;
  double canon_nodes_ms = 0.0;
  double canon_remap_ms = 0.0;
  double canon_first_ms = 0.0;
  TENRYU_ASSERT(geometry.boundary_intervals.size() >=
                    geometry.domain_segments.size(),
                "boundary_interval_unowned");
  std::chrono::steady_clock::time_point audit_start;
  if (timing_enabled) {
    audit_start = std::chrono::steady_clock::now();
  }
  const AxisAuditCounters axis_audit = audit_axis_bits(geometry);
  if (axis_counters != nullptr) {
    *axis_counters = axis_audit;
  }
  if (timing_enabled) {
    audit_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - audit_start)
                   .count();
  }

  const std::size_t n_int = geometry.interior_edges.size();
  RestrictedDcel dcel;
  std::vector<RestrictedNodeId> source;
  TENRYU_ASSERT(dt.sites.size() <= static_cast<std::size_t>(
                                      std::numeric_limits<SiteId>::max()),
                "restricted face count exceeds SiteId range");
  dcel.exterior_face = static_cast<SiteId>(dt.sites.size());
  dcel.half_edges.reserve(2 * (geometry.interior_edges.size() +
                               geometry.boundary_intervals.size()));

  const bool origin_dump =
      std::getenv("TENRYU_TESS_ORIGIN_DUMP") != nullptr;
  if (origin_dump) {
    for (std::size_t node = 0; node < geometry.nodes.size(); ++node) {
      const RestrictedNode& record = geometry.nodes[node];
      if (std::fabs(record.constructed[0]) < 1e-10 &&
          std::fabs(record.constructed[1]) < 1e-10) {
        std::fprintf(stderr, "[origin_node] id=%zu c=%.17e,%.17e\n",
                     node, record.constructed[0], record.constructed[1]);
      }
    }
  }

  std::chrono::steady_clock::time_point interior_start;
  if (timing_enabled) {
    interior_start = std::chrono::steady_clock::now();
  }
  for (const RestrictedDualEdge& edge : geometry.interior_edges) {
    if (origin_dump) {
      const RestrictedNode& n0 = geometry.nodes[edge.endpoint0];
      const RestrictedNode& n1 = geometry.nodes[edge.endpoint1];
      const bool near0 = std::fabs(n0.constructed[0]) < 1e-10 &&
                         std::fabs(n0.constructed[1]) < 1e-10;
      const bool near1 = std::fabs(n1.constructed[0]) < 1e-10 &&
                         std::fabs(n1.constructed[1]) < 1e-10;
      if (near0 || near1) {
        std::fprintf(stderr,
                     "[origin_edge] sites=%d,%d e0=%d e1=%d "
                     "e0_c=%.17e,%.17e e1_c=%.17e,%.17e\n",
                     static_cast<int>(edge.sites[0]),
                     static_cast<int>(edge.sites[1]),
                     static_cast<int>(edge.endpoint0),
                     static_cast<int>(edge.endpoint1), n0.constructed[0],
                     n0.constructed[1], n1.constructed[0],
                     n1.constructed[1]);
      }
    }
    TENRYU_ASSERT(edge.endpoint0 >= 0 && edge.endpoint1 >= 0 &&
                      edge.endpoint0 < static_cast<RestrictedNodeId>(
                                           geometry.nodes.size()) &&
                      edge.endpoint1 < static_cast<RestrictedNodeId>(
                                           geometry.nodes.size()),
                  "restricted edge endpoint is out of range");
    const SiteId first_site = edge.sites[0];
    const SiteId second_site = edge.sites[1];
    const int first_orientation = orient_edge_site(
        geometry.nodes[edge.endpoint0].exact_point,
        geometry.nodes[edge.endpoint1].exact_point, dt.sites[first_site]);
    const int second_orientation = orient_edge_site(
        geometry.nodes[edge.endpoint0].exact_point,
        geometry.nodes[edge.endpoint1].exact_point, dt.sites[second_site]);
    const bool separating_bisector =
        first_orientation != 0 &&
        second_orientation == -first_orientation;
    if (!separating_bisector) {
      const RestrictedNode& n0 = geometry.nodes[edge.endpoint0];
      const RestrictedNode& n1 = geometry.nodes[edge.endpoint1];
      const double edge_dr = n1.constructed[0] - n0.constructed[0];
      const double edge_dz = n1.constructed[1] - n0.constructed[1];
      const double l2 = edge_dr * edge_dr + edge_dz * edge_dz;
      const double site_dr =
          dt.sites[second_site].r - dt.sites[first_site].r;
      const double site_dz =
          dt.sites[second_site].z - dt.sites[first_site].z;
      // WeldScales are built after this DCEL assembly. Site spacing is the
      // local resolution scale for the fixed half-spacing tolerance.
      const double site_dist2 =
          site_dr * site_dr + site_dz * site_dz;
      // Non-separating dual edges shorter than half the site spacing are
      // the doomed degenerate-quadruple class (a true separating edge has
      // length O(site spacing)); tolerate them only when the reale
      // short-edge collapse band is active, which welds their endpoints
      // right after assembly.
      const double eta = reale_short_edge_collapse_rel();
      const bool tolerated = l2 < 0.25 * site_dist2;
      if (eta > 0.0 && tolerated) {
        std::fprintf(stderr,
                     "[dcel_sep_tolerated] sites=%d,%d l=%.3e\n",
                     static_cast<int>(first_site),
                     static_cast<int>(second_site), std::sqrt(l2));
        if (first_site < second_site) {
          append_twin_pair(dcel, edge.endpoint0, edge.endpoint1,
                           first_site, second_site, kInvalidId);
        } else {
          append_twin_pair(dcel, edge.endpoint1, edge.endpoint0,
                           first_site, second_site, kInvalidId);
        }
        continue;
      }
      std::fprintf(stderr,
                   "[dcel_sep_fail] sites=%d,%d orient=%d,%d e0=%d e1=%d "
                   "e0_bp=%d e0_edge=%d e0_lam=%.17e e1_bp=%d e1_edge=%d "
                   "e1_lam=%.17e e0_c=%.17e,%.17e e1_c=%.17e,%.17e\n",
                   static_cast<int>(first_site),
                   static_cast<int>(second_site), first_orientation,
                   second_orientation, static_cast<int>(edge.endpoint0),
                   static_cast<int>(edge.endpoint1),
                   n0.boundary_param.valid ? 1 : 0,
                   n0.boundary_param.valid
                       ? static_cast<int>(n0.boundary_param.edge)
                       : -1,
                   n0.lambda_cache, n1.boundary_param.valid ? 1 : 0,
                   n1.boundary_param.valid
                       ? static_cast<int>(n1.boundary_param.edge)
                       : -1,
                   n1.lambda_cache, n0.constructed[0], n0.constructed[1],
                   n1.constructed[0], n1.constructed[1]);
    }
    TENRYU_ASSERT(first_orientation != 0 &&
                      second_orientation == -first_orientation,
                  "restricted dual edge is not a separating bisector");
    if (first_orientation > 0) {
      append_twin_pair(dcel, edge.endpoint0, edge.endpoint1,
                       first_site, second_site, kInvalidId);
    } else {
      append_twin_pair(dcel, edge.endpoint1, edge.endpoint0,
                       first_site, second_site, kInvalidId);
    }
  }
  if (timing_enabled) {
    interior_ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - interior_start)
                      .count();
  }

  std::chrono::steady_clock::time_point bnd_start;
  if (timing_enabled) {
    bnd_start = std::chrono::steady_clock::now();
  }
  for (const BoundaryInterval& interval : geometry.boundary_intervals) {
    TENRYU_ASSERT(interval.segment >= 0 &&
                      interval.segment < static_cast<DomainSegmentId>(
                                             geometry.domain_segments.size()) &&
                      interval.owner >= 0 &&
                      interval.owner < static_cast<SiteId>(dt.sites.size()),
                  "boundary_interval_unowned");
    append_twin_pair(dcel, interval.endpoint0, interval.endpoint1,
                     interval.owner, dcel.exterior_face,
                     interval.segment);
  }
  if (timing_enabled) {
    bnd_ms = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - bnd_start)
                 .count();
  }

  stitch_face_cycles(dcel,
                     timing_enabled ? &stitch_sort_ms : nullptr,
                     timing_enabled ? &stitch_link_ms : nullptr);
  dcel.cells.reserve(dt.sites.size());
  std::vector<std::size_t> face_edge_count(dt.sites.size(), 0);
  std::vector<RestrictedHalfEdgeId> first_edge(dt.sites.size(), kInvalidId);
  for (RestrictedHalfEdgeId edge = 0;
       edge < static_cast<RestrictedHalfEdgeId>(dcel.half_edges.size());
       ++edge) {
    const SiteId face = dcel.half_edges[edge].left_face;
    if (face == dcel.exterior_face) {
      continue;
    }
    TENRYU_ASSERT(face >= 0 &&
                      face < static_cast<SiteId>(dt.sites.size()),
                  "restricted half-edge face is out of range");
    ++face_edge_count[face];
    if (first_edge[face] == kInvalidId) {
      first_edge[face] = edge;
    }
  }
  std::chrono::steady_clock::time_point faces_start;
  if (timing_enabled) {
    faces_start = std::chrono::steady_clock::now();
  }
  for (SiteId site = 0; site < static_cast<SiteId>(dt.sites.size()); ++site) {
    TENRYU_ASSERT(first_edge[site] != kInvalidId,
                  "generator_outside_cell");
    std::vector<RestrictedNodeId> cycle =
        trace_face(dcel, site, first_edge[site]);
    TENRYU_ASSERT(cycle.size() == face_edge_count[site] &&
                      cycle.size() >= 3,
                  "cell_nonsimple");
    std::vector<RestrictedNodeId> distinct = cycle;
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()),
                   distinct.end());
    TENRYU_ASSERT(distinct.size() >= 3, "cell_nonsimple");
    dcel.cells.push_back(
        RestrictedCell{site, first_edge[site], std::move(cycle)});
  }
  const std::vector<RestrictedNodeId> exterior_cycle =
      trace_face(dcel, dcel.exterior_face,
                 static_cast<RestrictedHalfEdgeId>(
                     2 * geometry.interior_edges.size() + 1));
  if (timing_enabled) {
    faces_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - faces_start)
                   .count();
  }
  TENRYU_ASSERT(exterior_cycle.size() ==
                    geometry.boundary_intervals.size(),
                "boundary_interval_unowned");

  canonicalize_nodes_borrowed(dt, dcel, geometry.nodes, source,
                              timing_enabled ? &canon_nodes_ms : nullptr,
                              timing_enabled ? &canon_remap_ms : nullptr,
                              timing_enabled ? &canon_first_ms : nullptr);
  if (timing_enabled) {
    std::fprintf(stderr,
                 "[tess_timing_dcel] audit_ms=%.1f interior_ms=%.1f "
                 "bnd_ms=%.1f stitch_sort_ms=%.1f stitch_link_ms=%.1f "
                 "faces_ms=%.1f canon_nodes_ms=%.1f canon_remap_ms=%.1f "
                 "canon_first_ms=%.1f n_int=%zu n_he=%zu n_nodes=%zu\n",
                 audit_ms, interior_ms, bnd_ms, stitch_sort_ms,
                 stitch_link_ms, faces_ms, canon_nodes_ms, canon_remap_ms,
                 canon_first_ms, n_int, dcel.half_edges.size(),
                 source.size());
  }
  return BorrowedAssembly{std::move(dcel), std::move(source)};
}

}  // namespace

RestrictedDcel assemble_restricted_dcel(
    const DelaunayTriangulation& dt,
    RestrictedGeometry geometry,
    AxisAuditCounters* axis_counters) {
  BorrowedAssembly assembly =
      assemble_restricted_dcel_borrowed(dt, geometry, axis_counters);
  attach_nodes(assembly.dcel, std::move(geometry.nodes),
               assembly.source);
  return std::move(assembly.dcel);
}

namespace {

void normalize_sort_and_rank_sites(std::vector<Site>& sites) {
  for (Site& site : sites) {
    if (site.r == 0.0) {
      site.r = 0.0;
    }
    if (site.z == 0.0) {
      site.z = 0.0;
    }
  }
  std::sort(sites.begin(), sites.end(),
            [](const Site& lhs, const Site& rhs) {
              return lhs.stable_id < rhs.stable_id;
            });
  for (std::size_t index = 0; index < sites.size(); ++index) {
    TENRYU_ASSERT(index <= std::numeric_limits<std::uint32_t>::max(),
                  "site rank exceeds uint32 range");
    sites[index].rank = static_cast<std::uint32_t>(index);
  }
}

RestrictedVoronoiResult finish_restricted_voronoi(
    DelaunayTriangulation dt,
    std::vector<DomainPoint> domain,
    const bool used_static_fallback,
    const bool used_cecr,
    const bool used_warm_entry,
    std::string failure_code,
    const std::chrono::steady_clock::time_point pipeline_start,
    const std::chrono::steady_clock::time_point delaunay_start,
    const std::chrono::steady_clock::time_point delaunay_end) {
  static const bool timing_enabled =
      std::getenv("TENRYU_TESS_TIMING") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_TIMING")) == std::string("1");
  static const bool weld_diag_enabled =
      std::getenv("TENRYU_WELD_DIAG") != nullptr &&
      std::string(std::getenv("TENRYU_WELD_DIAG")) == std::string("1");
  static const bool dvclp_diag_enabled =
      std::getenv("TENRYU_DVCLP_DIAG") != nullptr &&
      std::string(std::getenv("TENRYU_DVCLP_DIAG")) == std::string("1");
  static const bool axis_diag_enabled =
      std::getenv("TENRYU_TESS_AXIS") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_AXIS")) == std::string("1");
  const auto dual_start = std::chrono::steady_clock::now();
  VoronoiDual dual = build_voronoi_dual(dt);
  const auto dual_end = std::chrono::steady_clock::now();
  const auto restrict_start = std::chrono::steady_clock::now();
  RestrictedGeometry geometry =
      restrict_voronoi_dual(dt, dual, std::move(domain));
  const auto restrict_end = std::chrono::steady_clock::now();
  const auto arrangement_start = std::chrono::steady_clock::now();
  build_boundary_arrangement(dt, geometry);
  check_carrier_coverage(geometry);
  const auto arrangement_end = std::chrono::steady_clock::now();
  const auto dcel_start = std::chrono::steady_clock::now();
  bool carrier_representability_ok =
      geometry.carrier_representability_ok;
  bool carrier_coverage_ok = geometry.carrier_coverage_ok;
  AxisAuditCounters axis_counters;
  BorrowedAssembly raw =
      assemble_restricted_dcel_borrowed(dt, geometry, &axis_counters);
  const auto raw_dcel_end = std::chrono::steady_clock::now();
  const auto weld_start = std::chrono::steady_clock::now();
  WeldScales scales = compute_weld_scales_borrowed(
      raw.dcel, geometry.nodes, raw.source, dt.sites.size());
  WeldOutcome weld = weld_degenerate_edges(dt, geometry, scales);
  const auto weld_end = std::chrono::steady_clock::now();
  if (weld_diag_enabled) {
    std::fprintf(stderr,
                 "[weld] contracted=%d clusters=%d max_cluster=%d "
                 "max_disp=%.3e applied=%d passes=%d extended_forbidden=%zu\n",
                 weld.contracted_edges, weld.clusters,
                 weld.max_cluster_size, weld.max_displacement,
                 weld.applied ? 1 : 0, weld.passes,
                 weld.extended_band_forbidden);
  }
  if (!weld.ok) {
    return RestrictedVoronoiResult{
        std::move(dt), RestrictedDcel{}, false, used_static_fallback,
        used_cecr, weld.failure_code, std::move(weld),
        carrier_representability_ok, carrier_coverage_ok};
  }
  double final_dcel_ms = 0.0;
  double carrier2_ms = 0.0;
  RestrictedDcel dcel;
  if (weld.applied) {
    const auto carrier2_start = std::chrono::steady_clock::now();
    check_carrier_coverage(geometry);
    const auto carrier2_end = std::chrono::steady_clock::now();
    carrier2_ms =
        std::chrono::duration<double, std::milli>(carrier2_end -
                                                  carrier2_start)
            .count();
    carrier_representability_ok = geometry.carrier_representability_ok;
    carrier_coverage_ok = geometry.carrier_coverage_ok;
    const auto final_dcel_start = std::chrono::steady_clock::now();
    dcel =
        assemble_restricted_dcel(dt, std::move(geometry), &axis_counters);
    const auto final_dcel_end = std::chrono::steady_clock::now();
    final_dcel_ms =
        std::chrono::duration<double, std::milli>(final_dcel_end -
                                                  final_dcel_start)
            .count();
  } else {
    const auto attach_start = std::chrono::steady_clock::now();
    attach_nodes(raw.dcel, std::move(geometry.nodes), raw.source);
    dcel = std::move(raw.dcel);
    const auto attach_end = std::chrono::steady_clock::now();
    final_dcel_ms =
        std::chrono::duration<double, std::milli>(attach_end -
                                                  attach_start)
            .count();
  }
  std::string audit_code;
  const auto audit_start = std::chrono::steady_clock::now();
  const bool audit_ok = audit_weld_subfloor(dcel, scales, &audit_code);
  const auto audit_end = std::chrono::steady_clock::now();
  if (!audit_ok) {
    return RestrictedVoronoiResult{
        std::move(dt), RestrictedDcel{}, false, used_static_fallback,
        used_cecr, std::move(audit_code), std::move(weld),
        carrier_representability_ok, carrier_coverage_ok};
  }
  const auto pipeline_end = std::chrono::steady_clock::now();
  if (weld_diag_enabled || dvclp_diag_enabled) {
    const double total_ms =
        std::chrono::duration<double, std::milli>(pipeline_end -
                                                  pipeline_start)
            .count();
    const double delaunay_ms =
        std::chrono::duration<double, std::milli>(delaunay_end -
                                                  delaunay_start)
            .count();
    const double dcel_ms =
        std::chrono::duration<double, std::milli>(dual_end - dual_start)
            .count() +
        std::chrono::duration<double, std::milli>(restrict_end -
                                                  restrict_start)
            .count() +
        std::chrono::duration<double, std::milli>(arrangement_end -
                                                  arrangement_start)
            .count() +
        std::chrono::duration<double, std::milli>(raw_dcel_end - dcel_start)
            .count() +
        final_dcel_ms;
    const double weld_ms =
        std::chrono::duration<double, std::milli>(weld_end - weld_start)
            .count();
    const double audit_ms =
        std::chrono::duration<double, std::milli>(audit_end - audit_start)
            .count();
    const double other_ms =
        total_ms - delaunay_ms - dcel_ms - weld_ms - audit_ms;
    std::fprintf(stderr,
                 "[tess_ms] total=%.0f dt=%.0f dcel=%.0f weld=%.0f "
                 "audit=%.0f other=%.0f warm=%d\n",
                 total_ms, delaunay_ms, dcel_ms, weld_ms, audit_ms,
                 other_ms, used_warm_entry ? 1 : 0);
  }
  if (axis_diag_enabled) {
    std::fprintf(stderr,
                 "[tess_axis] nodes=%zu axis_nodes=%zu axis_segments=%zu "
                 "axis_intervals=%zu warm=%d\n",
                 axis_counters.nodes_total, axis_counters.axis_support_nodes,
                 axis_counters.axis_segments, axis_counters.axis_intervals,
                 used_warm_entry ? 1 : 0);
  }
  if (timing_enabled) {
    std::fprintf(
        stderr,
        "[tess_timing] delaunay_ms=%.1f dual_ms=%.1f restrict_ms=%.1f "
        "arrangement_ms=%.1f weld_ms=%.1f dcel_ms=%.1f carrier2_ms=%.1f\n",
        std::chrono::duration<double, std::milli>(delaunay_end -
                                                  delaunay_start)
            .count(),
        std::chrono::duration<double, std::milli>(dual_end - dual_start)
            .count(),
        std::chrono::duration<double, std::milli>(restrict_end -
                                                  restrict_start)
            .count(),
        std::chrono::duration<double, std::milli>(arrangement_end -
                                                  arrangement_start)
            .count(),
        std::chrono::duration<double, std::milli>(weld_end - weld_start)
            .count(),
        std::chrono::duration<double, std::milli>(raw_dcel_end - dcel_start)
                .count() +
            final_dcel_ms,
        carrier2_ms);
  }
  return RestrictedVoronoiResult{std::move(dt), std::move(dcel), true,
                                 used_static_fallback, used_cecr,
                                 std::move(failure_code),
                                 std::move(weld),
                                 carrier_representability_ok,
                                 carrier_coverage_ok};
}

}  // namespace

RestrictedVoronoiResult build_restricted_voronoi(
    std::vector<Site> sites,
    std::vector<DomainPoint> domain) {
  const auto pipeline_start = std::chrono::steady_clock::now();
  normalize_sort_and_rank_sites(sites);

  const auto delaunay_start = std::chrono::steady_clock::now();
  DelaunayResult delaunay = build_delaunay(std::move(sites));
  const auto delaunay_end = std::chrono::steady_clock::now();
  if (!delaunay.ok) {
    return RestrictedVoronoiResult{
        std::move(delaunay.dt), RestrictedDcel{}, false, false, false,
        std::move(delaunay.failure_code)};
  }
  return finish_restricted_voronoi(
      std::move(delaunay.dt), std::move(domain), false, false, false, {},
      pipeline_start, delaunay_start, delaunay_end);
}

RestrictedVoronoiResult build_restricted_voronoi_warm(
    const DelaunayTriangulation& previous,
    std::vector<Site> sites,
    std::vector<DomainPoint> domain) {
  const auto pipeline_start = std::chrono::steady_clock::now();
  normalize_sort_and_rank_sites(sites);

  const auto delaunay_start = std::chrono::steady_clock::now();
  WarmUpdateResult warm = warm_update_delaunay(previous, std::move(sites));
  const auto delaunay_end = std::chrono::steady_clock::now();
  if (!warm.ok) {
    return RestrictedVoronoiResult{
        std::move(warm.dt), RestrictedDcel{}, false,
        warm.used_static_fallback, warm.used_cecr,
        std::move(warm.failure_code)};
  }
  return finish_restricted_voronoi(
      std::move(warm.dt), std::move(domain), warm.used_static_fallback,
      warm.used_cecr, true, std::move(warm.failure_code), pipeline_start,
      delaunay_start, delaunay_end);
}

}  // namespace tenryu::mesh::tess
