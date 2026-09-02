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
#include "mesh/tessellation/gpu_circumcenter.hpp"
#include "mesh/tessellation/sos_policy.hpp"

namespace tenryu::mesh::tess {
namespace {

bool g_tess_gpu_dual_enabled = false;
bool g_tess_gpu_forced_fallback = false;
std::size_t g_tess_gpu_last_fallback_count = 0;

struct TriangleDescriptor {
  VoronoiTripleKey key;
  std::array<SiteId, 3> sites{};
};

TriangleDescriptor triangle_descriptor(const DelaunayTriangulation& dt,
                                       const TriangleId triangle) {
  TENRYU_ASSERT(triangle >= 0 &&
                    triangle < static_cast<TriangleId>(dt.tri_edge.size()),
                "Voronoi triangle id is out of range");
  const HalfEdgeId edge0 = dt.tri_edge[triangle];
  const HalfEdgeId edge1 = dt.he_next[edge0];
  const HalfEdgeId edge2 = dt.he_next[edge1];
  std::array<SiteId, 3> sites{{dt.he_origin[edge0], dt.he_origin[edge1],
                               dt.he_origin[edge2]}};
  std::sort(sites.begin(), sites.end(),
            [&dt](const SiteId lhs, const SiteId rhs) {
              return dt.sites[lhs].stable_id < dt.sites[rhs].stable_id;
            });
  return TriangleDescriptor{
      VoronoiTripleKey{{dt.sites[sites[0]].stable_id,
                        dt.sites[sites[1]].stable_id,
                        dt.sites[sites[2]].stable_id}},
      sites};
}

bool same_exact_point(const HomogeneousPoint& a,
                      const HomogeneousPoint& b) {
  const Expansion x_difference = expansion_difference(
      expansion_product(a.x, b.w), expansion_product(b.x, a.w));
  const Expansion y_difference = expansion_difference(
      expansion_product(a.y, b.w), expansion_product(b.y, a.w));
  return expansion_sign(x_difference) == 0 &&
         expansion_sign(y_difference) == 0;
}

double expansion_value(const Expansion& expansion) {
  return std::accumulate(expansion.begin(), expansion.end(), 0.0);
}

std::array<double, 2> construct_cartesian(
    const HomogeneousPoint& point) {
  const double denominator = expansion_value(point.w);
  TENRYU_ASSERT(denominator > 0.0,
                "Voronoi circumcenter denominator must be positive");
  return {{expansion_value(point.x) / denominator,
           expansion_value(point.y) / denominator}};
}

HomogeneousPoint witness_from_gpu_view(
    const GpuCircumcenterBatchView& view, const TriangleId triangle) {
  const std::size_t t = static_cast<std::size_t>(triangle);
  const double* x = view.x_terms + t * kGpuCcXYCap;
  const double* y = view.y_terms + t * kGpuCcXYCap;
  const double* w = view.w_terms + t * kGpuCcWCap;
  return HomogeneousPoint{
      Expansion(x, x + view.x_count[t]),
      Expansion(y, y + view.y_count[t]),
      Expansion(w, w + view.w_count[t])};
}

class TriangleUnionFind {
 public:
  explicit TriangleUnionFind(
      const std::vector<TriangleDescriptor>& descriptors)
      : parent_(descriptors.size()), descriptors_(descriptors) {
    std::iota(parent_.begin(), parent_.end(), TriangleId{0});
  }

  TriangleId find(const TriangleId triangle) {
    TENRYU_ASSERT(triangle >= 0 &&
                      triangle < static_cast<TriangleId>(parent_.size()),
                  "Voronoi union-find triangle is out of range");
    TriangleId root = triangle;
    while (parent_[root] != root) {
      root = parent_[root];
    }
    TriangleId current = triangle;
    while (parent_[current] != current) {
      const TriangleId next = parent_[current];
      parent_[current] = root;
      current = next;
    }
    return root;
  }

  void unite(const TriangleId first, const TriangleId second) {
    TriangleId first_root = find(first);
    TriangleId second_root = find(second);
    if (first_root == second_root) {
      return;
    }
    if (descriptors_[second_root].key < descriptors_[first_root].key) {
      std::swap(first_root, second_root);
    }
    parent_[second_root] = first_root;
  }

 private:
  std::vector<TriangleId> parent_;
  const std::vector<TriangleDescriptor>& descriptors_;
};

}  // namespace

void set_tess_gpu_dual_enabled(const bool enabled) {
  g_tess_gpu_dual_enabled = enabled;
}

bool tess_gpu_dual_enabled() { return g_tess_gpu_dual_enabled; }

void set_tess_gpu_forced_fallback_for_tests(const bool enabled) {
  g_tess_gpu_forced_fallback = enabled;
}

bool tess_gpu_forced_fallback_for_tests() {
  return g_tess_gpu_forced_fallback;
}

std::size_t tess_gpu_last_fallback_count() {
  return g_tess_gpu_last_fallback_count;
}

VoronoiVertexClasses build_voronoi_vertex_classes(
    const DelaunayTriangulation& dt) {
  static const bool timing_enabled =
      std::getenv("TENRYU_TESS_TIMING") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_TIMING")) == std::string("1");
  double desc_ms = 0.0;
  double pred_ms = 0.0;
  double members_ms = 0.0;
  double classes_ms = 0.0;
  double cc_ms = 0.0;
  std::size_t n_multi = 0;
  std::size_t n_cc = 0;
  TENRYU_ASSERT(dt.he_origin.size() == dt.he_twin.size() &&
                    dt.he_origin.size() == dt.he_next.size() &&
                    dt.he_origin.size() == dt.he_face.size() &&
                    dt.he_origin.size() == 3 * dt.tri_edge.size(),
                "Voronoi dual requires a certified Delaunay triangulation");
  TENRYU_ASSERT(
      dt.tri_edge.size() <=
          static_cast<std::size_t>(std::numeric_limits<VoronoiClassId>::max()),
      "Voronoi class count exceeds VoronoiClassId range");

  const bool gpu_enabled = g_tess_gpu_dual_enabled;
  const bool gpu_forced_fallback = g_tess_gpu_forced_fallback;
  const bool shadow_enabled =
      std::getenv("TENRYU_TESS_GPU_CC_SHADOW") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_GPU_CC_SHADOW")) ==
          std::string("1");

  std::vector<TriangleDescriptor> descriptors;
  descriptors.reserve(dt.tri_edge.size());
  std::chrono::steady_clock::time_point desc_start;
  if (timing_enabled) {
    desc_start = std::chrono::steady_clock::now();
  }
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    descriptors.push_back(triangle_descriptor(dt, triangle));
  }
  if (timing_enabled) {
    desc_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - desc_start)
                  .count();
  }

  GpuCircumcenterBatchView gpu_view;
  std::vector<std::int32_t> gpu_triplets;
  std::vector<double> gpu_site_rz;
  if (gpu_enabled && !gpu_forced_fallback && !descriptors.empty()) {
    gpu_triplets.reserve(3 * descriptors.size());
    for (const TriangleDescriptor& descriptor : descriptors) {
      for (const SiteId site : descriptor.sites) {
        gpu_triplets.push_back(static_cast<std::int32_t>(site));
      }
    }
    gpu_site_rz.reserve(2 * dt.sites.size());
    for (const Site& site : dt.sites) {
      gpu_site_rz.push_back(site.r);
      gpu_site_rz.push_back(site.z);
    }
    gpu_view = gpu_circumcenter_batch(descriptors.size(),
                                      gpu_triplets.data(),
                                      gpu_site_rz.data(), dt.sites.size());
  }

  TriangleUnionFind classes(descriptors);
  std::chrono::steady_clock::time_point pred_start;
  if (timing_enabled) {
    pred_start = std::chrono::steady_clock::now();
  }
  for (HalfEdgeId edge = 0;
       edge < static_cast<HalfEdgeId>(dt.he_origin.size()); ++edge) {
    const HalfEdgeId twin = dt.he_twin[edge];
    if (twin == kInvalidId || edge > twin) {
      continue;
    }
    TENRYU_ASSERT(twin >= 0 &&
                      twin < static_cast<HalfEdgeId>(dt.he_origin.size()) &&
                      dt.he_twin[twin] == edge,
                  "Voronoi dual requires reciprocal Delaunay twins");

    const HalfEdgeId edge_next = dt.he_next[edge];
    const HalfEdgeId edge_opposite = dt.he_next[edge_next];
    const HalfEdgeId twin_opposite = dt.he_next[dt.he_next[twin]];
    const SiteId a = dt.he_origin[edge];
    const SiteId b = dt.he_origin[edge_next];
    const SiteId c = dt.he_origin[edge_opposite];
    const SiteId d = dt.he_origin[twin_opposite];
    // Filtered predicates (SS8-w2fix36 semantics): the semi-static filter
    // answers certain-sign cases in doubles and falls back to the exact
    // path otherwise — same guaranteed sign, so this is exact-equivalent.
    TENRYU_ASSERT(orient2d_sign_tess(dt.sites[a], dt.sites[b],
                                     dt.sites[c]) > 0,
                  "Voronoi dual requires exact CCW Delaunay triangles");
    if (incircle_sign_tess(dt.sites[a], dt.sites[b],
                           dt.sites[c], dt.sites[d]) == 0) {
      classes.unite(dt.he_face[edge], dt.he_face[twin]);
    }
  }
  if (timing_enabled) {
    pred_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - pred_start)
                  .count();
  }

  std::map<VoronoiTripleKey, std::vector<TriangleId>> members;
  std::chrono::steady_clock::time_point members_start;
  if (timing_enabled) {
    members_start = std::chrono::steady_clock::now();
  }
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    const TriangleId representative = classes.find(triangle);
    members[descriptors[representative].key].push_back(triangle);
  }
  if (timing_enabled) {
    members_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - members_start)
                     .count();
  }

  VoronoiVertexClasses result;
  result.triangle_class.assign(dt.tri_edge.size(), kInvalidId);
  result.classes.reserve(members.size());
  std::chrono::steady_clock::time_point classes_start;
  if (timing_enabled) {
    classes_start = std::chrono::steady_clock::now();
  }
  std::size_t fallback_count = 0;
  double materialize_ns = 0.0;
  for (auto& entry : members) {
    TENRYU_ASSERT(
        result.classes.size() <=
            static_cast<std::size_t>(
                std::numeric_limits<VoronoiClassId>::max()),
        "Voronoi class count exceeds VoronoiClassId range");
    std::vector<TriangleId>& triangles = entry.second;
    std::sort(triangles.begin(), triangles.end(),
              [&descriptors](const TriangleId lhs, const TriangleId rhs) {
                return descriptors[lhs].key < descriptors[rhs].key;
              });
    const TriangleId representative = triangles.front();
    TENRYU_ASSERT(descriptors[representative].key == entry.first,
                  "Voronoi class representative is not canonical");
    const std::array<SiteId, 3>& representative_sites =
        descriptors[representative].sites;
    const auto materialize_start =
        timing_enabled && gpu_enabled
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
    const bool gpu_rep_hit =
        gpu_enabled && !gpu_forced_fallback &&
        gpu_view.fallback != nullptr &&
        gpu_view.fallback[static_cast<std::size_t>(representative)] == 0;
    if (gpu_enabled && !gpu_rep_hit) {
      ++fallback_count;
    }
    std::chrono::steady_clock::time_point cc_start;
    if (timing_enabled && !gpu_rep_hit) {
      cc_start = std::chrono::steady_clock::now();
    }
    const HomogeneousPoint witness =
        gpu_rep_hit ? witness_from_gpu_view(gpu_view, representative)
                    : circumcenter_exact(
                          dt.sites[representative_sites[0]],
                          dt.sites[representative_sites[1]],
                          dt.sites[representative_sites[2]]);
    if (timing_enabled && !gpu_rep_hit) {
      cc_ms += std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - cc_start)
                   .count();
      ++n_cc;
    }
    if (timing_enabled && gpu_enabled) {
      materialize_ns += std::chrono::duration<double, std::nano>(
                            std::chrono::steady_clock::now() -
                            materialize_start)
                            .count();
    }

    // Singleton classes (the overwhelming majority) would compare the
    // representative's circumcenter against itself recomputed — a pure
    // tautology. Verify members only for genuine multi-triangle
    // (cocircular) classes.
    if (triangles.size() > 1) {
      if (timing_enabled) {
        ++n_multi;
      }
      for (const TriangleId triangle : triangles) {
        const std::array<SiteId, 3>& sites = descriptors[triangle].sites;
        const bool gpu_member_hit =
            gpu_enabled && !gpu_forced_fallback &&
            gpu_view.fallback != nullptr &&
            gpu_view.fallback[static_cast<std::size_t>(triangle)] == 0;
        std::chrono::steady_clock::time_point member_cc_start;
        if (timing_enabled && !gpu_member_hit) {
          member_cc_start = std::chrono::steady_clock::now();
        }
        const HomogeneousPoint member_witness =
            gpu_member_hit
                ? witness_from_gpu_view(gpu_view, triangle)
                : circumcenter_exact(dt.sites[sites[0]], dt.sites[sites[1]],
                                     dt.sites[sites[2]]);
        if (timing_enabled && !gpu_member_hit) {
          cc_ms += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - member_cc_start)
                       .count();
          ++n_cc;
        }
        TENRYU_ASSERT(same_exact_point(witness, member_witness),
                      "cocircular_class_inconsistent");
      }
    }

    const VoronoiClassId class_id =
        static_cast<VoronoiClassId>(result.classes.size());
    for (const TriangleId triangle : triangles) {
      TENRYU_ASSERT(result.triangle_class[triangle] == kInvalidId,
                    "Delaunay triangle belongs to multiple Voronoi classes");
      result.triangle_class[triangle] = class_id;
    }
    const std::array<double, 2> constructed =
        gpu_rep_hit
            ? std::array<double, 2>{
                  gpu_view.constructed[2 * static_cast<std::size_t>(
                                               representative)],
                  gpu_view.constructed[2 * static_cast<std::size_t>(
                                               representative) +
                                       1]}
            : construct_cartesian(witness);
    if (shadow_enabled && gpu_rep_hit) {
      const HomogeneousPoint host_witness = circumcenter_exact(
          dt.sites[representative_sites[0]],
          dt.sites[representative_sites[1]],
          dt.sites[representative_sites[2]]);
      const std::array<double, 2> host_constructed =
          construct_cartesian(host_witness);
      TENRYU_ASSERT(
          witness.x.size() == host_witness.x.size() &&
              witness.y.size() == host_witness.y.size() &&
              witness.w.size() == host_witness.w.size() &&
              std::memcmp(witness.x.data(), host_witness.x.data(),
                          witness.x.size() * sizeof(double)) == 0 &&
              std::memcmp(witness.y.data(), host_witness.y.data(),
                          witness.y.size() * sizeof(double)) == 0 &&
              std::memcmp(witness.w.data(), host_witness.w.data(),
                          witness.w.size() * sizeof(double)) == 0 &&
              std::memcmp(constructed.data(), host_constructed.data(),
                          2 * sizeof(double)) == 0,
          "cc_gpu_mismatch");
    }
    result.classes.push_back(VoronoiVertexClass{
        entry.first, representative_sites, std::move(triangles), witness,
        constructed});
  }

  if (gpu_enabled) {
    g_tess_gpu_last_fallback_count = fallback_count;
    if (timing_enabled) {
      std::fprintf(stderr,
                   "[tess_timing_vvc_gpu] gpu_ms=%.1f d2h_ms=%.1f "
                   "materialize_ms=%.1f fallback_n=%zu n_classes=%zu\n",
                   gpu_view.gpu_ms, gpu_view.d2h_ms,
                   materialize_ns / 1.0e6, fallback_count,
                   result.classes.size());
    }
  }
  if (timing_enabled) {
    classes_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - classes_start)
                     .count();
  }

  TENRYU_ASSERT(std::find(result.triangle_class.begin(),
                          result.triangle_class.end(), kInvalidId) ==
                    result.triangle_class.end(),
                "Delaunay triangle has no Voronoi class");
  if (timing_enabled) {
    std::fprintf(stderr,
                 "[tess_timing_vvc] desc_ms=%.1f pred_ms=%.1f "
                 "members_ms=%.1f classes_ms=%.1f cc_ms=%.1f n_tri=%zu "
                 "n_classes=%zu n_multi=%zu n_cc=%zu\n",
                 desc_ms, pred_ms, members_ms, classes_ms, cc_ms,
                 dt.tri_edge.size(), result.classes.size(), n_multi, n_cc);
  }
  return result;
}

}  // namespace tenryu::mesh::tess
