#include "mesh/tessellation/voronoi_dual.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "core/error.hpp"

namespace tenryu::mesh::tess {
namespace {

// Non-ReALE callers retain the exact-only weld path until the ReALE driver
// installs its namelist value before target construction.
double g_reale_short_edge_collapse_rel = 0.0;

template <typename T>
void sort_unique(std::vector<T>& values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

Expansion absolute_expansion(Expansion value) {
  if (expansion_sign(value) < 0) {
    value = expansion_negate(value);
  }
  return value;
}

double expansion_value(const Expansion& value) {
  return std::accumulate(value.begin(), value.end(), 0.0);
}

double ulp_gap(const double value) {
  const double upward =
      std::nextafter(value, std::numeric_limits<double>::infinity()) - value;
  const double downward =
      value - std::nextafter(value, -std::numeric_limits<double>::infinity());
  return std::max(upward, downward);
}

Expansion edge_delta(const std::array<double, 2>& first,
                     const std::array<double, 2>& second) {
  const Expansion radial = expansion_scale(
      expansion_sum(Expansion{ulp_gap(first[0])},
                    Expansion{ulp_gap(second[0])}),
      0.5);
  const Expansion axial = expansion_scale(
      expansion_sum(Expansion{ulp_gap(first[1])},
                    Expansion{ulp_gap(second[1])}),
      0.5);
  return expansion_sum(radial, axial);
}

Expansion squared_distance(const std::array<double, 2>& first,
                           const std::array<double, 2>& second) {
  const Expansion dr = exact_difference(second[0], first[0]);
  const Expansion dz = exact_difference(second[1], first[1]);
  return expansion_sum(expansion_product(dr, dr),
                       expansion_product(dz, dz));
}

void require_scale(const WeldScales& scales, const SiteId site) {
  TENRYU_ASSERT(site >= 0 &&
                    site < static_cast<SiteId>(scales.area.size()) &&
                    site < static_cast<SiteId>(scales.perimeter.size()),
                "weld scale site is out of range");
  TENRYU_ASSERT(expansion_sign(scales.area[site]) > 0 &&
                    expansion_sign(scales.perimeter[site]) > 0,
                "weld scale must be positive");
}

int compare_h(const WeldScales& scales,
              const SiteId first,
              const SiteId second) {
  require_scale(scales, first);
  require_scale(scales, second);
  return expansion_sign(expansion_difference(
      expansion_product(scales.area[first], scales.perimeter[second]),
      expansion_product(scales.area[second], scales.perimeter[first])));
}

// Per-site caches over the immutable WeldScales: the ApproxBound
// enclosures feed a certain-sign screen for compare_h (a certain sign
// is the exact sign; ties/uncertain fall back to the exact
// expansions), and the expansion_value pair reproduces
// measure_pair_chi's h_double bitwise (same expansion_value calls,
// evaluated once per site).
struct HScaleCache {
  std::vector<ApproxBound> area;
  std::vector<ApproxBound> perimeter;
  std::vector<double> area_value;
  std::vector<double> perimeter_value;
  std::vector<std::uint8_t> ready;

  explicit HScaleCache(const std::size_t n_sites)
      : area(n_sites), perimeter(n_sites), area_value(n_sites, 0.0),
        perimeter_value(n_sites, 0.0), ready(n_sites, 0) {}

  void materialize(const WeldScales& scales, const SiteId site) {
    if (ready[static_cast<std::size_t>(site)] != 0) {
      return;
    }
    area[static_cast<std::size_t>(site)] =
        approx_of_expansion(scales.area[site]);
    perimeter[static_cast<std::size_t>(site)] =
        approx_of_expansion(scales.perimeter[site]);
    area_value[static_cast<std::size_t>(site)] =
        expansion_value(scales.area[site]);
    perimeter_value[static_cast<std::size_t>(site)] =
        expansion_value(scales.perimeter[site]);
    ready[static_cast<std::size_t>(site)] = 1;
  }
};

int compare_h_screened(const WeldScales& scales,
                       HScaleCache& cache,
                       const SiteId first,
                       const SiteId second) {
  require_scale(scales, first);
  require_scale(scales, second);
  cache.materialize(scales, first);
  cache.materialize(scales, second);
  const ApproxSign screened = approx_sign(approx_sub(
      approx_mul(cache.area[static_cast<std::size_t>(first)],
                 cache.perimeter[static_cast<std::size_t>(second)]),
      approx_mul(cache.area[static_cast<std::size_t>(second)],
                 cache.perimeter[static_cast<std::size_t>(first)])));
  if (screened.certain) {
    return screened.sign;
  }
  return compare_h(scales, first, second);
}

SiteId min_h_site(const WeldScales& scales,
                  HScaleCache& cache,
                  const std::vector<SiteId>& sites) {
  TENRYU_ASSERT(!sites.empty(), "weld edge has no incident cell");
  SiteId minimum = sites.front();
  require_scale(scales, minimum);
  for (std::size_t index = 1; index < sites.size(); ++index) {
    const SiteId candidate = sites[index];
    const int order = compare_h_screened(scales, cache, candidate, minimum);
    if (order < 0 || (order == 0 && candidate < minimum)) {
      minimum = candidate;
    }
  }
  return minimum;
}

struct EdgeMeasure {
  Expansion length_squared;
  Expansion delta;
  SiteId h_site = kInvalidId;
  double length_squared_double = 0.0;
  double delta_double = 0.0;
  double h_double = 0.0;
};

// Candidate admission, the complete-link certificate, and the final audit use
// this single canonical chi measurement. A certificate-refused pair is safely
// unweldable: the edge-AV CFL and crossing guards are du-referenced, DV-CLP
// clips du to edge budgets, and the staged-dt gate catches dt-dangerous commits.
// The robust floor is chi = l^2 / (delta h) <= 4: the +/-2-ulp application
// band on l and the h-attribution spread between the pair's two node-h values
// each carry a factor-2-class ambiguity, so measured chi <= 2^2 cannot be
// robustly distinguished from the floor. This is derived, not tunable.
EdgeMeasure measure_pair_chi(const RestrictedNode& first,
                             const RestrictedNode& second,
                             const SiteId first_h_site,
                             const SiteId second_h_site,
                             const WeldScales& scales,
                             HScaleCache& cache) {
  EdgeMeasure measure;
  measure.h_site = first_h_site;
  const int order = compare_h_screened(scales, cache, second_h_site,
                                       first_h_site);
  if (order < 0 || (order == 0 && second_h_site < first_h_site)) {
    measure.h_site = second_h_site;
  }
  const double dr = second.constructed[0] - first.constructed[0];
  const double dz = second.constructed[1] - first.constructed[1];
  measure.length_squared_double = dr * dr + dz * dz;
  measure.delta_double =
      0.5 * (ulp_gap(first.constructed[0]) +
             ulp_gap(second.constructed[0])) +
      0.5 * (ulp_gap(first.constructed[1]) +
             ulp_gap(second.constructed[1]));
  cache.materialize(scales, measure.h_site);
  measure.h_double = 2.0 * cache.area_value[measure.h_site] /
                     cache.perimeter_value[measure.h_site];
  if (measure.length_squared_double >
      8.0 * (measure.delta_double * measure.h_double)) {
    return measure;
  }
  measure.length_squared =
      squared_distance(first.constructed, second.constructed);
  measure.delta = edge_delta(first.constructed, second.constructed);
  return measure;
}

bool is_subfloor(const EdgeMeasure& measure,
                 const WeldScales& scales) {
  if (measure.length_squared_double >
      8.0 * (measure.delta_double * measure.h_double)) {
    return false;
  }
  const Expansion retain = expansion_product(
      measure.length_squared, scales.perimeter[measure.h_site]);
  const Expansion collapse = expansion_scale(
      expansion_product(measure.delta, scales.area[measure.h_site]), 8.0);
  return expansion_sign(expansion_difference(retain, collapse)) <= 0;
}

// The strict band chi <= 1 (the pre-robust-floor criterion l^2 P <= 2 delta A).
// The fail-loud invariant -- a sub-noise edge between unweldable nodes means
// the geometry is broken -- holds only here. In the extended band (1,4] the
// edge is not proven noise, so fail-loud classes are retained instead.
// Precondition: only call on a measure that passed is_subfloor, so the exact
// expansion fields are populated whenever the 2x double screen admits.
bool is_subfloor_strict(const EdgeMeasure& measure,
                        const WeldScales& scales) {
  if (measure.length_squared_double >
      2.0 * (measure.delta_double * measure.h_double)) {
    return false;
  }
  const Expansion retain = expansion_product(
      measure.length_squared, scales.perimeter[measure.h_site]);
  const Expansion collapse = expansion_scale(
      expansion_product(measure.delta, scales.area[measure.h_site]), 2.0);
  return expansion_sign(expansion_difference(retain, collapse)) <= 0;
}

Expansion rational_difference(const Expansion& first_coordinate,
                              const Expansion& first_weight,
                              const Expansion& second_coordinate,
                              const Expansion& second_weight) {
  return expansion_difference(
      expansion_product(first_coordinate, second_weight),
      expansion_product(second_coordinate, first_weight));
}

int projection_compare(const HomogeneousPoint& first,
                       const HomogeneousPoint& second,
                       const HomogeneousPoint& segment_start,
                       const HomogeneousPoint& segment_end) {
  const Expansion difference_r = rational_difference(
      first.x, first.w, second.x, second.w);
  const Expansion difference_z = rational_difference(
      first.y, first.w, second.y, second.w);
  const Expansion direction_r = rational_difference(
      segment_end.x, segment_end.w, segment_start.x, segment_start.w);
  const Expansion direction_z = rational_difference(
      segment_end.y, segment_end.w, segment_start.y, segment_start.w);
  return expansion_sign(expansion_sum(
      expansion_product(difference_r, direction_r),
      expansion_product(difference_z, direction_z)));
}

class WeldComponents {
 public:
  WeldComponents(const RestrictedGeometry& geometry,
                 const std::vector<SiteId>& node_h_sites)
      : geometry_(geometry),
        node_h_sites_(node_h_sites),
        parent_(geometry.nodes.size()),
        members_(geometry.nodes.size()),
        representative_(geometry.nodes.size(), kInvalidId) {
    std::iota(parent_.begin(), parent_.end(), RestrictedNodeId{0});
    for (RestrictedNodeId node = 0;
         node < static_cast<RestrictedNodeId>(geometry.nodes.size()); ++node) {
      members_[node].push_back(node);
      representative_[node] = node;
    }
  }

  RestrictedNodeId find(const RestrictedNodeId node) {
    TENRYU_ASSERT(node >= 0 &&
                      node < static_cast<RestrictedNodeId>(parent_.size()),
                  "weld node is out of range");
    RestrictedNodeId root = node;
    while (parent_[root] != root) {
      root = parent_[root];
    }
    RestrictedNodeId current = node;
    while (parent_[current] != current) {
      const RestrictedNodeId next = parent_[current];
      parent_[current] = root;
      current = next;
    }
    return root;
  }

  const std::vector<RestrictedNodeId>& members(const RestrictedNodeId root) {
    return members_[find(root)];
  }

  RestrictedNodeId representative(const RestrictedNodeId root) {
    return representative_[find(root)];
  }

  bool has_master(const RestrictedNodeId root) {
    for (const RestrictedNodeId member : members(root)) {
      if (!geometry_.nodes[member].domain_vertices.empty()) {
        return true;
      }
    }
    return false;
  }

  std::vector<DomainSegmentId> boundary_segments(
      const RestrictedNodeId root) {
    std::vector<DomainSegmentId> segments;
    for (const RestrictedNodeId member : members(root)) {
      segments.insert(segments.end(),
                      geometry_.nodes[member].boundary_segments.begin(),
                      geometry_.nodes[member].boundary_segments.end());
    }
    sort_unique(segments);
    return segments;
  }

  bool complete_link(const RestrictedNodeId first_root,
                     const RestrictedNodeId second_root,
                     const WeldScales& scales,
                     HScaleCache& cache) {
    for (const RestrictedNodeId first : members(first_root)) {
      for (const RestrictedNodeId second : members(second_root)) {
        TENRYU_ASSERT(node_h_sites_[first] != kInvalidId &&
                          node_h_sites_[second] != kInvalidId,
                      "weld node has no incident cell");
        const EdgeMeasure measure = measure_pair_chi(
            geometry_.nodes[first], geometry_.nodes[second],
            node_h_sites_[first], node_h_sites_[second], scales, cache);
        if (!is_subfloor(measure, scales)) {
          return false;
        }
      }
    }
    return true;
  }

  RestrictedNodeId unite(const RestrictedNodeId first_root,
                         const RestrictedNodeId second_root) {
    RestrictedNodeId first = find(first_root);
    RestrictedNodeId second = find(second_root);
    TENRYU_ASSERT(first != second, "weld requires distinct components");
    if (second < first) {
      std::swap(first, second);
    }
    parent_[second] = first;
    members_[first].insert(members_[first].end(),
                           members_[second].begin(), members_[second].end());
    std::sort(members_[first].begin(), members_[first].end());
    members_[second].clear();
    representative_[first] = select_representative(members_[first]);
    representative_[second] = kInvalidId;
    return first;
  }

  std::vector<RestrictedNodeId> roots() {
    std::vector<RestrictedNodeId> result;
    for (RestrictedNodeId node = 0;
         node < static_cast<RestrictedNodeId>(parent_.size()); ++node) {
      if (find(node) == node) {
        result.push_back(node);
      }
    }
    return result;
  }

 private:
  RestrictedNodeId select_representative(
      const std::vector<RestrictedNodeId>& component) const {
    RestrictedNodeId master = kInvalidId;
    for (const RestrictedNodeId member : component) {
      if (!geometry_.nodes[member].domain_vertices.empty()) {
        TENRYU_ASSERT(master == kInvalidId,
                      "weld component contains multiple masters");
        master = member;
      }
    }
    if (master != kInvalidId) {
      return master;
    }

    std::vector<DomainSegmentId> common_segments =
        geometry_.nodes[component.front()].boundary_segments;
    sort_unique(common_segments);
    for (std::size_t index = 1; index < component.size(); ++index) {
      std::vector<DomainSegmentId> member_segments =
          geometry_.nodes[component[index]].boundary_segments;
      sort_unique(member_segments);
      std::vector<DomainSegmentId> intersection;
      std::set_intersection(common_segments.begin(), common_segments.end(),
                            member_segments.begin(), member_segments.end(),
                            std::back_inserter(intersection));
      common_segments = std::move(intersection);
    }

    RestrictedNodeId selected = component.front();
    if (!common_segments.empty()) {
      const DomainSegmentId segment_id = common_segments.front();
      TENRYU_ASSERT(
          segment_id >= 0 &&
              segment_id < static_cast<DomainSegmentId>(
                               geometry_.domain_segments.size()),
          "weld boundary segment is out of range");
      const DomainSegmentRecord& segment =
          geometry_.domain_segments[segment_id];
      const HomogeneousPoint& start =
          geometry_.nodes[segment.start_node].exact_point;
      const HomogeneousPoint& end =
          geometry_.nodes[segment.end_node].exact_point;
      for (std::size_t index = 1; index < component.size(); ++index) {
        const RestrictedNodeId candidate = component[index];
        const int order = projection_compare(
            geometry_.nodes[candidate].exact_point,
            geometry_.nodes[selected].exact_point, start, end);
        if (order < 0 ||
            (order == 0 && geometry_.nodes[candidate].key <
                               geometry_.nodes[selected].key)) {
          selected = candidate;
        }
      }
      return selected;
    }

    for (std::size_t index = 1; index < component.size(); ++index) {
      const RestrictedNodeId candidate = component[index];
      if (geometry_.nodes[candidate].key < geometry_.nodes[selected].key) {
        selected = candidate;
      }
    }
    return selected;
  }

  const RestrictedGeometry& geometry_;
  const std::vector<SiteId>& node_h_sites_;
  std::vector<RestrictedNodeId> parent_;
  std::vector<std::vector<RestrictedNodeId>> members_;
  std::vector<RestrictedNodeId> representative_;
};

bool shares_segment(const std::vector<DomainSegmentId>& first,
                    const std::vector<DomainSegmentId>& second) {
  std::size_t first_index = 0;
  std::size_t second_index = 0;
  while (first_index < first.size() && second_index < second.size()) {
    if (first[first_index] == second[second_index]) {
      return true;
    }
    if (first[first_index] < second[second_index]) {
      ++first_index;
    } else {
      ++second_index;
    }
  }
  return false;
}

enum class CandidateClass {
  kAllowed,
  kMasterMaster,
  kForbidden,
};

CandidateClass classify_candidate_classes(
    const bool first_master,
    const bool second_master,
    const std::vector<DomainSegmentId>& first_segments,
    const std::vector<DomainSegmentId>& second_segments) {
  if (first_master && second_master) {
    return CandidateClass::kMasterMaster;
  }
  if (first_segments.empty() && second_segments.empty()) {
    return CandidateClass::kAllowed;
  }
  if (!first_segments.empty() && !second_segments.empty() &&
      shares_segment(first_segments, second_segments)) {
    return CandidateClass::kAllowed;
  }
  return CandidateClass::kForbidden;
}

CandidateClass classify_candidate(WeldComponents& components,
                                  const RestrictedNodeId first_root,
                                  const RestrictedNodeId second_root) {
  const bool first_master = components.has_master(first_root);
  const bool second_master = components.has_master(second_root);
  if (first_master && second_master) {
    return CandidateClass::kMasterMaster;
  }
  const std::vector<DomainSegmentId> first_segments =
      components.boundary_segments(first_root);
  const std::vector<DomainSegmentId> second_segments =
      components.boundary_segments(second_root);
  return classify_candidate_classes(first_master, second_master,
                                    first_segments, second_segments);
}

const char* residual_verdict(const CandidateClass classification) {
  switch (classification) {
    case CandidateClass::kMasterMaster:
      return "master_master";
    case CandidateClass::kForbidden:
      return "forbidden_class_pair";
    case CandidateClass::kAllowed:
      return "blocked_by_prior_weld_chain";
  }
  return "unknown";
}

struct WeldAuditPair {
  std::array<RestrictedNodeId, 2> nodes{};
  std::vector<SiteId> sites;
  CandidateClass classification = CandidateClass::kAllowed;
  double separation = 0.0;
  double chi = 0.0;
  double delta_e = 0.0;
  double h_e = 0.0;
  bool strict = false;
};

struct WeldCandidate {
  std::size_t source_index = 0;
  RestrictedNodeId first_root = kInvalidId;
  RestrictedNodeId second_root = kInvalidId;
  RestrictedNodeId first_representative = kInvalidId;
  RestrictedNodeId second_representative = kInvalidId;
  double chi_dbl = 0.0;
  CanonicalNodeKey low_key;
  CanonicalNodeKey high_key;
};

// D9: Candidate ordering deliberately uses the cached binary64 chi proxy.
// It is a pure function of snapped doubles and canonical keys, so traversal,
// thread, and hash order cannot affect it. Exact arithmetic remains confined to
// the weld condition, complete-link certificate, and final audit. Results may
// differ from the former exact-chi order only in interacting multi-candidate
// clusters; class rules, complete-link certification, and the audit are
// unchanged. The source index only distinguishes otherwise identical source
// records in the cascade container.
bool candidate_less(const WeldCandidate& first,
                    const WeldCandidate& second) {
  if (first.chi_dbl != second.chi_dbl) {
    return first.chi_dbl < second.chi_dbl;
  }
  if (!(first.low_key == second.low_key)) {
    return first.low_key < second.low_key;
  }
  if (!(first.high_key == second.high_key)) {
    return first.high_key < second.high_key;
  }
  return first.source_index < second.source_index;
}

struct CandidateLess {
  bool operator()(const WeldCandidate& first,
                  const WeldCandidate& second) const {
    return candidate_less(first, second);
  }
};

struct WeldSourceEdge {
  RestrictedNodeId first_node = kInvalidId;
  RestrictedNodeId second_node = kInvalidId;
  std::vector<SiteId> sites;
};

std::vector<SiteId> edge_sites(const RestrictedDualEdge& edge) {
  std::vector<SiteId> sites{edge.sites[0], edge.sites[1]};
  sort_unique(sites);
  return sites;
}

std::vector<SiteId> interval_sites(const BoundaryInterval& interval) {
  return std::vector<SiteId>{interval.owner};
}

void add_node_sites(std::vector<std::vector<SiteId>>& node_sites,
                    const RestrictedNodeId node,
                    const std::vector<SiteId>& sites) {
  TENRYU_ASSERT(node >= 0 &&
                    node < static_cast<RestrictedNodeId>(node_sites.size()),
                "weld edge endpoint is out of range");
  node_sites[node].insert(node_sites[node].end(), sites.begin(), sites.end());
}

std::vector<SiteId> select_node_h_sites(
    std::vector<std::vector<SiteId>> node_sites,
    const WeldScales& scales,
    HScaleCache& cache) {
  std::vector<SiteId> result(node_sites.size(), kInvalidId);
  for (std::size_t node = 0; node < node_sites.size(); ++node) {
    sort_unique(node_sites[node]);
    if (!node_sites[node].empty()) {
      result[node] = min_h_site(scales, cache, node_sites[node]);
    }
  }
  return result;
}

std::vector<SiteId> build_node_h_sites(const RestrictedGeometry& geometry,
                                       const WeldScales& scales,
                                       HScaleCache& cache) {
  std::vector<std::vector<SiteId>> node_sites(geometry.nodes.size());
  for (const RestrictedDualEdge& edge : geometry.interior_edges) {
    const std::vector<SiteId> sites = edge_sites(edge);
    add_node_sites(node_sites, edge.endpoint0, sites);
    add_node_sites(node_sites, edge.endpoint1, sites);
  }
  for (const BoundaryInterval& interval : geometry.boundary_intervals) {
    const std::vector<SiteId> sites = interval_sites(interval);
    add_node_sites(node_sites, interval.endpoint0, sites);
    add_node_sites(node_sites, interval.endpoint1, sites);
  }
  return select_node_h_sites(std::move(node_sites), scales, cache);
}

void set_candidate_keys(WeldCandidate& candidate,
                        const RestrictedGeometry& geometry) {
  const CanonicalNodeKey& first =
      geometry.nodes[candidate.first_representative].key;
  const CanonicalNodeKey& second =
      geometry.nodes[candidate.second_representative].key;
  if (second < first) {
    candidate.low_key = second;
    candidate.high_key = first;
  } else {
    candidate.low_key = first;
    candidate.high_key = second;
  }
}

bool make_candidate(WeldCandidate& candidate,
                    WeldOutcome& outcome,
                    WeldComponents& components,
                    const RestrictedGeometry& geometry,
                    const WeldScales& scales,
                    HScaleCache& cache,
                    const std::vector<SiteId>& node_h_sites,
                    const std::vector<bool>& blocked,
                    const std::size_t source_index,
                    const WeldSourceEdge& source) {
  if (blocked[source_index]) {
    return false;
  }
  const RestrictedNodeId first_root = components.find(source.first_node);
  const RestrictedNodeId second_root = components.find(source.second_node);
  if (first_root == second_root) {
    return false;
  }
  const RestrictedNodeId first_representative =
      components.representative(first_root);
  const RestrictedNodeId second_representative =
      components.representative(second_root);
  TENRYU_ASSERT(node_h_sites[first_representative] != kInvalidId &&
                    node_h_sites[second_representative] != kInvalidId,
                "weld node has no incident cell");
  const EdgeMeasure measure = measure_pair_chi(
      geometry.nodes[first_representative],
      geometry.nodes[second_representative],
      node_h_sites[first_representative], node_h_sites[second_representative],
      scales, cache);
  if (!is_subfloor(measure, scales)) {
    return false;
  }

  const CandidateClass classification =
      classify_candidate(components, first_root, second_root);
  if (classification == CandidateClass::kMasterMaster ||
      classification == CandidateClass::kForbidden) {
    // Fail loud only in the strict band; an extended-band pair is retained
    // silently (no candidate, no error) and counted for diagnostics.
    if (is_subfloor_strict(measure, scales)) {
      outcome.ok = false;
      outcome.failure_code = classification == CandidateClass::kMasterMaster
                                 ? "weld_master_master"
                                 : "weld_forbidden_class_pair";
    } else {
      ++outcome.extended_band_forbidden;
    }
    return false;
  }

  candidate.source_index = source_index;
  candidate.first_root = first_root;
  candidate.second_root = second_root;
  candidate.first_representative = first_representative;
  candidate.second_representative = second_representative;
  candidate.chi_dbl = measure.length_squared_double /
                      (measure.delta_double * measure.h_double);
  set_candidate_keys(candidate, geometry);
  return true;
}

void compact_welded_nodes(RestrictedGeometry& geometry,
                          WeldComponents& components,
                          const std::vector<SiteId>& node_h_sites,
                          const WeldScales& scales,
                          HScaleCache& cache,
                          const std::vector<std::uint8_t>& geometric_components,
                          WeldOutcome& outcome) {
  std::vector<RestrictedNodeId> roots = components.roots();
  std::sort(roots.begin(), roots.end(),
            [&](const RestrictedNodeId first,
                const RestrictedNodeId second) {
              return geometry.nodes[components.representative(first)].key <
                     geometry.nodes[components.representative(second)].key;
            });

  std::vector<RestrictedNodeId> remap(geometry.nodes.size(), kInvalidId);
  std::vector<RestrictedNode> compact;
  compact.reserve(roots.size());
  for (const RestrictedNodeId root : roots) {
    TENRYU_ASSERT(compact.size() <= static_cast<std::size_t>(
                                        std::numeric_limits<RestrictedNodeId>::max()),
                  "restricted node count exceeds RestrictedNodeId range");
    const RestrictedNodeId representative = components.representative(root);
    RestrictedNode merged = geometry.nodes[representative];
    const std::vector<RestrictedNodeId>& members = components.members(root);
    if (members.size() > 1) {
      ++outcome.clusters;
      outcome.max_cluster_size = std::max(
          outcome.max_cluster_size, static_cast<int>(members.size()));
    }
    for (const RestrictedNodeId member : members) {
      remap[member] = static_cast<RestrictedNodeId>(compact.size());
      if (members.size() > 1) {
        TENRYU_ASSERT(
            (!geometric_components.empty() &&
             geometric_components[static_cast<std::size_t>(root)] != 0) ||
                (node_h_sites[member] != kInvalidId &&
                 node_h_sites[representative] != kInvalidId &&
                 is_subfloor(
                     measure_pair_chi(
                         geometry.nodes[member],
                         geometry.nodes[representative], node_h_sites[member],
                         node_h_sites[representative], scales, cache),
                     scales)),
            "weld complete-link certificate is missing");
        const double dr = geometry.nodes[member].constructed[0] -
                          geometry.nodes[representative].constructed[0];
        const double dz = geometry.nodes[member].constructed[1] -
                          geometry.nodes[representative].constructed[1];
        outcome.max_displacement =
            std::max(outcome.max_displacement, std::hypot(dr, dz));
      }
      merged.aliases.push_back(geometry.nodes[member].key);
      merged.aliases.insert(merged.aliases.end(),
                            geometry.nodes[member].aliases.begin(),
                            geometry.nodes[member].aliases.end());
      merged.exact_witnesses.insert(
          merged.exact_witnesses.end(),
          geometry.nodes[member].exact_witnesses.begin(),
          geometry.nodes[member].exact_witnesses.end());
      merged.boundary_segments.insert(
          merged.boundary_segments.end(),
          geometry.nodes[member].boundary_segments.begin(),
          geometry.nodes[member].boundary_segments.end());
      merged.domain_vertices.insert(
          merged.domain_vertices.end(),
          geometry.nodes[member].domain_vertices.begin(),
          geometry.nodes[member].domain_vertices.end());
    }
    sort_unique(merged.aliases);
    std::sort(merged.exact_witnesses.begin(),
              merged.exact_witnesses.end(),
              [](const RestrictedExactWitness& first,
                 const RestrictedExactWitness& second) {
                return first.key < second.key;
              });
    merged.exact_witnesses.erase(
        std::unique(merged.exact_witnesses.begin(),
                    merged.exact_witnesses.end(),
                    [](const RestrictedExactWitness& first,
                       const RestrictedExactWitness& second) {
                      return first.key == second.key;
                    }),
        merged.exact_witnesses.end());
    sort_unique(merged.boundary_segments);
    sort_unique(merged.domain_vertices);
    compact.push_back(std::move(merged));
  }

  for (RestrictedDualEdge& edge : geometry.interior_edges) {
    TENRYU_ASSERT(edge.endpoint0 >= 0 && edge.endpoint1 >= 0 &&
                      edge.endpoint0 < static_cast<RestrictedNodeId>(
                                           remap.size()) &&
                      edge.endpoint1 < static_cast<RestrictedNodeId>(
                                           remap.size()),
                  "weld edge endpoint is out of range");
    edge.endpoint0 = remap[edge.endpoint0];
    edge.endpoint1 = remap[edge.endpoint1];
  }
  geometry.interior_edges.erase(
      std::remove_if(geometry.interior_edges.begin(),
                     geometry.interior_edges.end(),
                     [](const RestrictedDualEdge& edge) {
                       return edge.endpoint0 == edge.endpoint1;
                     }),
      geometry.interior_edges.end());
  for (DomainSegmentRecord& segment : geometry.domain_segments) {
    TENRYU_ASSERT(segment.start_node >= 0 && segment.end_node >= 0 &&
                      segment.start_node < static_cast<RestrictedNodeId>(
                                               remap.size()) &&
                      segment.end_node < static_cast<RestrictedNodeId>(
                                             remap.size()),
                  "weld domain endpoint is out of range");
    segment.start_node = remap[segment.start_node];
    segment.end_node = remap[segment.end_node];
  }
  geometry.nodes = std::move(compact);
  geometry.identity_contractions.clear();
}

template <typename ConstructedAt>
WeldScales compute_weld_scales_impl(const RestrictedDcel& raw_dcel,
                                    const std::size_t n_sites,
                                    const std::size_t node_count,
                                    ConstructedAt constructed_at) {
  WeldScales scales;
  scales.area.resize(n_sites);
  scales.perimeter.resize(n_sites);
  for (const RestrictedCell& cell : raw_dcel.cells) {
    TENRYU_ASSERT(cell.site >= 0 &&
                      cell.site < static_cast<SiteId>(n_sites),
                  "weld scale site is out of range");
    TENRYU_ASSERT(cell.cycle.size() >= 3,
                  "weld scale requires a cell cycle");
    Expansion twice_area;
    Expansion perimeter;
    for (std::size_t index = 0; index < cell.cycle.size(); ++index) {
      const RestrictedNodeId first_id = cell.cycle[index];
      const RestrictedNodeId second_id =
          cell.cycle[(index + 1) % cell.cycle.size()];
      TENRYU_ASSERT(first_id >= 0 && second_id >= 0 &&
                        first_id < static_cast<RestrictedNodeId>(
                                       node_count) &&
                        second_id < static_cast<RestrictedNodeId>(
                                        node_count),
                    "weld cell node is out of range");
      const std::array<double, 2>& first = constructed_at(first_id);
      const std::array<double, 2>& second = constructed_at(second_id);
      twice_area = expansion_sum(
          twice_area,
          expansion_difference(
              expansion_product(Expansion{first[0]}, Expansion{second[1]}),
              expansion_product(Expansion{second[0]}, Expansion{first[1]})));
      perimeter = expansion_sum(
          perimeter,
          expansion_sum(absolute_expansion(exact_difference(second[0],
                                                             first[0])),
                        absolute_expansion(exact_difference(second[1],
                                                             first[1]))));
    }
    scales.area[cell.site] =
        expansion_scale(absolute_expansion(std::move(twice_area)), 0.5);
    scales.perimeter[cell.site] = std::move(perimeter);
    require_scale(scales, cell.site);
  }
  return scales;
}

}  // namespace

void set_reale_short_edge_collapse_rel(const double collapse_rel) {
  g_reale_short_edge_collapse_rel = collapse_rel;
}

double reale_short_edge_collapse_rel() {
  return g_reale_short_edge_collapse_rel;
}

WeldScales compute_weld_scales(const RestrictedDcel& raw_dcel,
                               const std::size_t n_sites) {
  return compute_weld_scales_impl(
      raw_dcel, n_sites, raw_dcel.nodes.size(),
      [&raw_dcel](const RestrictedNodeId id)
          -> const std::array<double, 2>& {
        return raw_dcel.nodes[id].constructed;
      });
}

WeldScales compute_weld_scales_borrowed(
    const RestrictedDcel& raw_dcel,
    const std::vector<RestrictedNode>& records,
    const std::vector<RestrictedNodeId>& source,
    const std::size_t n_sites) {
  return compute_weld_scales_impl(
      raw_dcel, n_sites, source.size(),
      [&records, &source](const RestrictedNodeId id)
          -> const std::array<double, 2>& {
        const RestrictedNodeId origin =
            source[static_cast<std::size_t>(id)];
        TENRYU_ASSERT(origin >= 0 &&
                          origin < static_cast<RestrictedNodeId>(
                                       records.size()),
                      "restricted node source is out of range");
        return records[static_cast<std::size_t>(origin)].constructed;
      });
}

namespace {

WeldOutcome weld_degenerate_edges_pass(const DelaunayTriangulation& dt,
                                       RestrictedGeometry& geometry,
                                       const WeldScales& scales,
                                       HScaleCache& cache) {
  WeldOutcome outcome;
  const std::vector<SiteId> node_h_sites =
      build_node_h_sites(geometry, scales, cache);
  WeldComponents components(geometry, node_h_sites);
  const std::size_t interior_count = geometry.interior_edges.size();
  const std::size_t source_count =
      interior_count + geometry.boundary_intervals.size();
  std::vector<WeldSourceEdge> sources;
  sources.reserve(source_count);
  std::vector<std::vector<std::size_t>> incident_sources(
      geometry.nodes.size());
  const auto append_source = [&](const RestrictedNodeId first_node,
                                 const RestrictedNodeId second_node,
                                 std::vector<SiteId> sites) {
    TENRYU_ASSERT(first_node >= 0 && second_node >= 0 &&
                      first_node < static_cast<RestrictedNodeId>(
                                       incident_sources.size()) &&
                      second_node < static_cast<RestrictedNodeId>(
                                        incident_sources.size()),
                  "weld edge endpoint is out of range");
    const std::size_t source_index = sources.size();
    sources.push_back(
        WeldSourceEdge{first_node, second_node, std::move(sites)});
    incident_sources[first_node].push_back(source_index);
    incident_sources[second_node].push_back(source_index);
  };
  for (const RestrictedDualEdge& edge : geometry.interior_edges) {
    append_source(edge.endpoint0, edge.endpoint1, edge_sites(edge));
  }
  for (const BoundaryInterval& interval : geometry.boundary_intervals) {
    append_source(interval.endpoint0, interval.endpoint1,
                  interval_sites(interval));
  }
  TENRYU_ASSERT(sources.size() == source_count,
                "weld source edge count mismatch");

  const double geometric_eta = g_reale_short_edge_collapse_rel;
  std::vector<bool> blocked(source_count, false);
  std::vector<WeldCandidate> candidates;
  candidates.reserve(source_count);
  for (std::size_t source_index = 0; source_index < sources.size();
       ++source_index) {
    WeldCandidate candidate;
    if (make_candidate(candidate, outcome, components, geometry, scales, cache,
                       node_h_sites, blocked, source_index,
                       sources[source_index])) {
      candidates.push_back(std::move(candidate));
    }
    if (!outcome.ok) {
      outcome.passes = 1;
      return outcome;
    }
  }
  if (candidates.empty() && !(geometric_eta > 0.0)) {
    return outcome;
  }
  outcome.passes = candidates.empty() ? 0 : 1;
  std::sort(candidates.begin(), candidates.end(), CandidateLess{});

  // D10: Enumerate and sort the source edges once. Merges only revisit source
  // edges incident to the merged components, placing fresh candidates in this
  // deterministic cascade set; stale and duplicate work is harmless because
  // every selected source is revalidated against current representatives.
  std::set<WeldCandidate, CandidateLess> cascade;
  std::size_t cursor = 0;
  while (cursor < candidates.size() || !cascade.empty()) {
    WeldCandidate selected;
    if (cascade.empty() ||
        (cursor < candidates.size() &&
         candidate_less(candidates[cursor], *cascade.begin()))) {
      selected = candidates[cursor];
      ++cursor;
    } else {
      const auto next = cascade.begin();
      selected = *next;
      cascade.erase(next);
    }

    WeldCandidate current;
    if (!make_candidate(current, outcome, components, geometry, scales, cache,
                        node_h_sites, blocked, selected.source_index,
                        sources[selected.source_index])) {
      if (!outcome.ok) {
        return outcome;
      }
      continue;
    }
    if (!components.complete_link(current.first_root,
                                  current.second_root, scales, cache)) {
      blocked[current.source_index] = true;
      continue;
    }

    std::vector<std::size_t> affected_sources;
    for (const RestrictedNodeId member :
         components.members(current.first_root)) {
      affected_sources.insert(affected_sources.end(),
                              incident_sources[member].begin(),
                              incident_sources[member].end());
    }
    for (const RestrictedNodeId member :
         components.members(current.second_root)) {
      affected_sources.insert(affected_sources.end(),
                              incident_sources[member].begin(),
                              incident_sources[member].end());
    }
    sort_unique(affected_sources);

    components.unite(current.first_root, current.second_root);
    ++outcome.contracted_edges;
    for (const std::size_t source_index : affected_sources) {
      WeldCandidate candidate;
      if (make_candidate(candidate, outcome, components, geometry, scales,
                         cache, node_h_sites, blocked, source_index,
                         sources[source_index])) {
        cascade.insert(std::move(candidate));
      }
      if (!outcome.ok) {
        return outcome;
      }
    }
  }

  std::vector<std::uint8_t> geometric_components;
  if (geometric_eta > 0.0) {
    const double eta_squared = geometric_eta * geometric_eta;
    for (const WeldSourceEdge& source : sources) {
      const RestrictedNodeId first_root = components.find(source.first_node);
      const RestrictedNodeId second_root = components.find(source.second_node);
      if (first_root == second_root) {
        continue;
      }
      TENRYU_ASSERT(node_h_sites[source.first_node] != kInvalidId &&
                        node_h_sites[source.second_node] != kInvalidId,
                    "weld node has no incident cell");
      cache.materialize(scales, node_h_sites[source.first_node]);
      cache.materialize(scales, node_h_sites[source.second_node]);
      const double dr = geometry.nodes[source.second_node].constructed[0] -
                        geometry.nodes[source.first_node].constructed[0];
      const double dz = geometry.nodes[source.second_node].constructed[1] -
                        geometry.nodes[source.first_node].constructed[1];
      const double length_squared = dr * dr + dz * dz;
      const double area = std::min(
          cache.area_value[static_cast<std::size_t>(
              node_h_sites[source.first_node])],
          cache.area_value[static_cast<std::size_t>(
              node_h_sites[source.second_node])]);
      if (!(length_squared < eta_squared * area) ||
          classify_candidate(components, first_root, second_root) !=
              CandidateClass::kAllowed) {
        continue;
      }
      if (geometric_components.empty()) {
        geometric_components.resize(geometry.nodes.size(), 0);
      }
      const RestrictedNodeId root =
          components.unite(first_root, second_root);
      geometric_components[static_cast<std::size_t>(root)] = 1;
      ++outcome.contracted_edges;
      outcome.passes = 1;
    }
  }

  if (outcome.contracted_edges == 0) {
    return outcome;
  }
  compact_welded_nodes(geometry, components, node_h_sites, scales, cache,
                       geometric_components, outcome);
  rebuild_boundary_intervals(dt, geometry);
  outcome.applied = true;
  return outcome;
}

}  // namespace

WeldOutcome weld_degenerate_edges(const DelaunayTriangulation& dt,
                                  RestrictedGeometry& geometry,
                                  const WeldScales& scales) {
  constexpr int kMaxWeldPasses = 8;
  WeldOutcome outcome;
  HScaleCache cache(scales.area.size());
  // Every applied pass strictly reduces the node count, so termination is
  // inherent; this structural cap is only a backstop.
  for (int pass = 0; pass < kMaxWeldPasses; ++pass) {
    WeldOutcome pass_outcome =
        weld_degenerate_edges_pass(dt, geometry, scales, cache);
    outcome.passes += pass_outcome.passes;
    outcome.contracted_edges += pass_outcome.contracted_edges;
    outcome.clusters += pass_outcome.clusters;
    outcome.max_cluster_size =
        std::max(outcome.max_cluster_size, pass_outcome.max_cluster_size);
    outcome.max_displacement =
        std::max(outcome.max_displacement, pass_outcome.max_displacement);
    outcome.applied = outcome.applied || pass_outcome.applied;
    // Overwrite, not accumulate: retained pairs persist across passes, so the
    // final pass census is the final-state count.
    outcome.extended_band_forbidden = pass_outcome.extended_band_forbidden;
    if (!pass_outcome.ok) {
      outcome.ok = false;
      outcome.failure_code = std::move(pass_outcome.failure_code);
      return outcome;
    }
    if (!pass_outcome.applied) {
      return outcome;
    }
  }
  return outcome;
}

bool audit_weld_subfloor(const RestrictedDcel& final_dcel,
                         const WeldScales& scales,
                         std::string* failure_code) {
  if (failure_code != nullptr) {
    failure_code->clear();
  }
  HScaleCache cache(scales.area.size());
  std::vector<std::vector<SiteId>> node_sites(final_dcel.nodes.size());
  std::vector<std::pair<std::array<RestrictedNodeId, 2>, SiteId>> edge_sites;
  for (const RestrictedCell& cell : final_dcel.cells) {
    TENRYU_ASSERT(cell.cycle.size() >= 3,
                  "weld audit requires a cell cycle");
    for (std::size_t index = 0; index < cell.cycle.size(); ++index) {
      RestrictedNodeId first = cell.cycle[index];
      RestrictedNodeId second =
          cell.cycle[(index + 1) % cell.cycle.size()];
      TENRYU_ASSERT(first >= 0 && second >= 0 &&
                        first < static_cast<RestrictedNodeId>(
                                    final_dcel.nodes.size()) &&
                        second < static_cast<RestrictedNodeId>(
                                     final_dcel.nodes.size()),
                    "weld audit node is out of range");
      if (first == second) {
        continue;
      }
      node_sites[first].push_back(cell.site);
      node_sites[second].push_back(cell.site);
      if (second < first) {
        std::swap(first, second);
      }
      edge_sites.push_back({{{first, second}}, cell.site});
    }
  }
  const std::vector<SiteId> node_h_sites =
      select_node_h_sites(std::move(node_sites), scales, cache);

  std::sort(edge_sites.begin(), edge_sites.end());
  std::vector<WeldAuditPair> residuals;
  std::size_t group_begin = 0;
  while (group_begin < edge_sites.size()) {
    std::size_t group_end = group_begin + 1;
    while (group_end < edge_sites.size() &&
           edge_sites[group_end].first == edge_sites[group_begin].first) {
      ++group_end;
    }
    const std::array<RestrictedNodeId, 2>& edge =
        edge_sites[group_begin].first;
    std::vector<SiteId> sites;
    sites.reserve(group_end - group_begin);
    for (std::size_t index = group_begin; index < group_end; ++index) {
      sites.push_back(edge_sites[index].second);
    }
    sort_unique(sites);
    TENRYU_ASSERT(node_h_sites[edge[0]] != kInvalidId &&
                      node_h_sites[edge[1]] != kInvalidId,
                  "weld audit node has no incident cell");
    const EdgeMeasure measure = measure_pair_chi(
        final_dcel.nodes[edge[0]], final_dcel.nodes[edge[1]],
        node_h_sites[edge[0]], node_h_sites[edge[1]], scales, cache);
    if (is_subfloor(measure, scales)) {
      const RestrictedNode& first = final_dcel.nodes[edge[0]];
      const RestrictedNode& second = final_dcel.nodes[edge[1]];
      WeldAuditPair residual;
      residual.nodes = edge;
      residual.sites = sites;
      residual.classification = classify_candidate_classes(
          !first.domain_vertices.empty(), !second.domain_vertices.empty(),
          first.boundary_segments, second.boundary_segments);
      residual.separation = std::hypot(
          second.constructed[0] - first.constructed[0],
          second.constructed[1] - first.constructed[1]);
      residual.chi = measure.length_squared_double /
                     (measure.delta_double * measure.h_double);
      residual.delta_e = measure.delta_double;
      residual.h_e = measure.h_double;
      residual.strict = is_subfloor_strict(measure, scales);
      residuals.push_back(std::move(residual));
    }
    group_begin = group_end;
  }

  if (residuals.empty()) {
    return true;
  }

  std::size_t master_master = 0;
  std::size_t forbidden_class_pair = 0;
  std::size_t blocked_by_prior_weld_chain = 0;
  std::size_t strict_count = 0;
  std::vector<double> chis;
  chis.reserve(residuals.size());
  for (const WeldAuditPair& residual : residuals) {
    chis.push_back(residual.chi);
    if (residual.strict) {
      ++strict_count;
    }
    switch (residual.classification) {
      case CandidateClass::kMasterMaster:
        ++master_master;
        break;
      case CandidateClass::kForbidden:
        ++forbidden_class_pair;
        break;
      case CandidateClass::kAllowed:
        ++blocked_by_prior_weld_chain;
        break;
    }
  }
  std::sort(chis.begin(), chis.end());
  const std::size_t middle = chis.size() / 2;
  const double chi_median =
      chis.size() % 2 == 0 ? 0.5 * (chis[middle - 1] + chis[middle])
                           : chis[middle];
  std::fprintf(stderr,
               "[weld_audit] residual=%zu master_master=%zu "
               "forbidden_class_pair=%zu blocked_by_prior_weld_chain=%zu "
               "strict=%zu chi_min=%.17g chi_median=%.17g chi_max=%.17g\n",
               residuals.size(), master_master, forbidden_class_pair,
               blocked_by_prior_weld_chain, strict_count, chis.front(),
               chi_median, chis.back());

  const std::size_t pair_limit = std::min<std::size_t>(8, residuals.size());
  for (std::size_t index = 0; index < pair_limit; ++index) {
    const WeldAuditPair& residual = residuals[index];
    const RestrictedNode& first = final_dcel.nodes[residual.nodes[0]];
    const RestrictedNode& second = final_dcel.nodes[residual.nodes[1]];
    std::fprintf(stderr,
                 "[weld_audit_pair] nodes=%d,%d sites=",
                 static_cast<int>(residual.nodes[0]),
                 static_cast<int>(residual.nodes[1]));
    for (std::size_t site_index = 0; site_index < residual.sites.size();
         ++site_index) {
      std::fprintf(stderr, "%s%d", site_index == 0 ? "" : ",",
                   static_cast<int>(residual.sites[site_index]));
    }
    std::fprintf(stderr,
                 " first=(r=%.17g,z=%.17g) second=(r=%.17g,z=%.17g) "
                 "separation=%.17g chi=%.17g delta_e=%.17g h_e=%.17g "
                 "verdict=%s band=%s\n",
                 first.constructed[0], first.constructed[1],
                 second.constructed[0], second.constructed[1],
                 residual.separation, residual.chi, residual.delta_e,
                 residual.h_e, residual_verdict(residual.classification),
                 residual.strict ? "strict" : "extended");
  }
  if (strict_count > 0) {
    if (failure_code != nullptr) {
      *failure_code = "weld_residual_subfloor";
    }
    return false;
  }
  return true;
}

}  // namespace tenryu::mesh::tess
