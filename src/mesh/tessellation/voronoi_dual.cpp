#include "mesh/tessellation/voronoi_dual.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"

namespace tenryu::mesh::tess {
namespace {

struct StableEdgeKey {
  std::uint64_t lo;
  std::uint64_t hi;

  bool operator<(const StableEdgeKey& other) const {
    return lo < other.lo || (lo == other.lo && hi < other.hi);
  }
};

struct EdgeSeed {
  StableEdgeKey key;
  HalfEdgeId half_edge;
};

std::uint64_t coordinate_bits(const double value) {
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

bool bitwise_identical(const std::array<double, 2>& first,
                       const std::array<double, 2>& second) {
  return coordinate_bits(first[0]) == coordinate_bits(second[0]) &&
         coordinate_bits(first[1]) == coordinate_bits(second[1]);
}

std::array<SiteId, 2> sorted_primal_sites(
    const DelaunayTriangulation& dt,
    const HalfEdgeId edge) {
  std::array<SiteId, 2> sites{{dt.he_origin[edge],
                               dt.he_origin[dt.he_next[edge]]}};
  if (dt.sites[sites[1]].stable_id < dt.sites[sites[0]].stable_id) {
    std::swap(sites[0], sites[1]);
  }
  return sites;
}

std::vector<EdgeSeed> canonical_edge_seeds(
    const DelaunayTriangulation& dt) {
  std::vector<EdgeSeed> seeds;
  for (HalfEdgeId edge = 0;
       edge < static_cast<HalfEdgeId>(dt.he_origin.size()); ++edge) {
    const HalfEdgeId twin = dt.he_twin[edge];
    if (twin != kInvalidId && edge > twin) {
      continue;
    }
    const std::array<SiteId, 2> sites = sorted_primal_sites(dt, edge);
    seeds.push_back(EdgeSeed{
        StableEdgeKey{dt.sites[sites[0]].stable_id,
                      dt.sites[sites[1]].stable_id},
        edge});
  }
  std::sort(seeds.begin(), seeds.end(),
            [](const EdgeSeed& first, const EdgeSeed& second) {
              return first.key < second.key;
            });
  for (std::size_t index = 1; index < seeds.size(); ++index) {
    TENRYU_ASSERT(seeds[index - 1].key < seeds[index].key,
                  "dual_feature_duplicated");
  }
  return seeds;
}

class ClassUnionFind {
 public:
  explicit ClassUnionFind(const std::size_t count) : parent_(count) {
    std::iota(parent_.begin(), parent_.end(), VoronoiClassId{0});
  }

  VoronoiClassId find(const VoronoiClassId class_id) {
    TENRYU_ASSERT(class_id >= 0 &&
                      class_id < static_cast<VoronoiClassId>(parent_.size()),
                  "Voronoi class id is out of range");
    VoronoiClassId root = class_id;
    while (parent_[root] != root) {
      root = parent_[root];
    }
    VoronoiClassId current = class_id;
    while (parent_[current] != current) {
      const VoronoiClassId next = parent_[current];
      parent_[current] = root;
      current = next;
    }
    return root;
  }

  void unite(const VoronoiClassId first, const VoronoiClassId second) {
    VoronoiClassId first_root = find(first);
    VoronoiClassId second_root = find(second);
    if (first_root == second_root) {
      return;
    }
    if (second_root < first_root) {
      std::swap(first_root, second_root);
    }
    parent_[second_root] = first_root;
  }

 private:
  std::vector<VoronoiClassId> parent_;
};

}  // namespace

VoronoiDual build_voronoi_dual(const DelaunayTriangulation& dt) {
  static const bool timing_enabled =
      std::getenv("TENRYU_TESS_TIMING") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_TIMING")) == std::string("1");
  double vvc_ms = 0.0;
  double seeds_ms = 0.0;
  double mach_ms = 0.0;
  double nodemap_ms = 0.0;
  double feat_ms = 0.0;
  std::chrono::steady_clock::time_point vvc_start;
  if (timing_enabled) {
    vvc_start = std::chrono::steady_clock::now();
  }
  VoronoiVertexClasses vertex_classes =
      build_voronoi_vertex_classes(dt);
  if (timing_enabled) {
    vvc_ms = std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - vvc_start)
                 .count();
  }
  std::chrono::steady_clock::time_point seeds_start;
  if (timing_enabled) {
    seeds_start = std::chrono::steady_clock::now();
  }
  const std::vector<EdgeSeed> seeds = canonical_edge_seeds(dt);
  if (timing_enabled) {
    seeds_ms = std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - seeds_start)
                   .count();
  }

  ClassUnionFind machine_nodes(vertex_classes.classes.size());
  std::chrono::steady_clock::time_point mach_start;
  if (timing_enabled) {
    mach_start = std::chrono::steady_clock::now();
  }
  for (const EdgeSeed& seed : seeds) {
    const HalfEdgeId edge = seed.half_edge;
    const HalfEdgeId twin = dt.he_twin[edge];
    if (twin == kInvalidId) {
      continue;
    }
    const VoronoiClassId first =
        vertex_classes.triangle_class[dt.he_face[edge]];
    const VoronoiClassId second =
        vertex_classes.triangle_class[dt.he_face[twin]];
    if (first == second) {
      continue;
    }

    // Machine-identity contraction is intentionally edge-local. A global
    // coordinate map must not merge disconnected constructed points.
    if (bitwise_identical(vertex_classes.classes[first].constructed,
                          vertex_classes.classes[second].constructed)) {
      machine_nodes.unite(first, second);
    }
  }
  if (timing_enabled) {
    mach_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - mach_start)
                  .count();
  }

  std::chrono::steady_clock::time_point nodemap_start;
  if (timing_enabled) {
    nodemap_start = std::chrono::steady_clock::now();
  }
  VoronoiDual dual;
  dual.classes = std::move(vertex_classes.classes);
  dual.triangle_class = std::move(vertex_classes.triangle_class);
  dual.class_node.assign(dual.classes.size(), kInvalidId);

  std::map<VoronoiClassId, std::vector<VoronoiClassId>> node_members;
  for (VoronoiClassId class_id = 0;
       class_id < static_cast<VoronoiClassId>(dual.classes.size());
       ++class_id) {
    node_members[machine_nodes.find(class_id)].push_back(class_id);
  }
  dual.nodes.reserve(node_members.size());
  for (auto& entry : node_members) {
    TENRYU_ASSERT(
        dual.nodes.size() <=
            static_cast<std::size_t>(
                std::numeric_limits<VoronoiNodeId>::max()),
        "Voronoi node count exceeds VoronoiNodeId range");
    const VoronoiNodeId node_id =
        static_cast<VoronoiNodeId>(dual.nodes.size());
    for (const VoronoiClassId class_id : entry.second) {
      dual.class_node[class_id] = node_id;
      TENRYU_ASSERT(
          bitwise_identical(dual.classes[entry.first].constructed,
                            dual.classes[class_id].constructed),
          "machine-identity Voronoi node has inconsistent coordinates");
    }
    dual.nodes.push_back(VoronoiDualNode{
        entry.first, std::move(entry.second),
        dual.classes[entry.first].constructed});
  }
  if (timing_enabled) {
    nodemap_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - nodemap_start)
                     .count();
  }

  dual.registry.reserve(seeds.size());
  dual.features.reserve(seeds.size());
  std::chrono::steady_clock::time_point feat_start;
  if (timing_enabled) {
    feat_start = std::chrono::steady_clock::now();
  }
  for (const EdgeSeed& seed : seeds) {
    HalfEdgeId edge = seed.half_edge;
    HalfEdgeId twin = dt.he_twin[edge];
    if (twin != kInvalidId &&
        dual.triangle_class[dt.he_face[twin]] <
            dual.triangle_class[dt.he_face[edge]]) {
      std::swap(edge, twin);
    }
    const std::array<SiteId, 2> primal_sites =
        sorted_primal_sites(dt, edge);
    const VoronoiClassId first_class =
        dual.triangle_class[dt.he_face[edge]];
    const VoronoiClassId second_class =
        twin == kInvalidId ? kInvalidId
                           : dual.triangle_class[dt.he_face[twin]];

    DualFeatureRegistryEntry registry_entry;
    registry_entry.delaunay_half_edge = edge;
    registry_entry.primal_sites = primal_sites;
    registry_entry.first_class = first_class;
    registry_entry.second_class = second_class;

    if (second_class == first_class) {
      registry_entry.disposition =
          DualEdgeDisposition::kCocircularContracted;
      dual.registry.push_back(std::move(registry_entry));
      continue;
    }

    const VoronoiNodeId first_node = dual.class_node[first_class];
    const VoronoiNodeId second_node =
        second_class == kInvalidId ? kInvalidId
                                   : dual.class_node[second_class];
    if (second_class != kInvalidId && first_node == second_node) {
      registry_entry.disposition =
          DualEdgeDisposition::kMachineIdentityContracted;
      dual.registry.push_back(std::move(registry_entry));
      continue;
    }

    TENRYU_ASSERT(
        dual.features.size() <=
            static_cast<std::size_t>(
                std::numeric_limits<DualFeatureId>::max()),
        "Voronoi feature count exceeds DualFeatureId range");
    registry_entry.feature =
        static_cast<DualFeatureId>(dual.features.size());

    VoronoiDualFeature feature;
    feature.delaunay_half_edge = edge;
    feature.primal_sites = primal_sites;
    feature.first_class = first_class;
    feature.second_class = second_class;
    feature.first_node = first_node;
    feature.second_node = second_node;
    if (second_class == kInvalidId) {
      registry_entry.disposition = DualEdgeDisposition::kRayFeature;
      feature.kind = DualFeatureKind::kUnboundedRay;
      const SiteId origin = dt.he_origin[edge];
      const SiteId destination = dt.he_origin[dt.he_next[edge]];
      feature.ray_direction = ExactDirectionWitness{
          exact_difference(dt.sites[destination].z, dt.sites[origin].z),
          exact_difference(dt.sites[origin].r,
                           dt.sites[destination].r)};
    } else {
      registry_entry.disposition = DualEdgeDisposition::kBoundedFeature;
      feature.kind = DualFeatureKind::kBoundedSegment;
    }
    dual.features.push_back(std::move(feature));
    dual.registry.push_back(std::move(registry_entry));
  }
  if (timing_enabled) {
    feat_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - feat_start)
                  .count();
  }

  if (timing_enabled) {
    std::fprintf(stderr,
                 "[tess_timing_dual] vvc_ms=%.1f seeds_ms=%.1f "
                 "mach_ms=%.1f nodemap_ms=%.1f feat_ms=%.1f n_seeds=%zu "
                 "n_nodes=%zu n_feat=%zu\n",
                 vvc_ms, seeds_ms, mach_ms, nodemap_ms, feat_ms,
                 seeds.size(), dual.nodes.size(), dual.features.size());
  }
  return dual;
}

}  // namespace tenryu::mesh::tess
