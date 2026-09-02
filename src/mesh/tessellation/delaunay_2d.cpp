#include "mesh/tessellation/delaunay_2d.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "mesh/tessellation/sos_policy.hpp"

namespace tenryu::mesh::tess {

struct CecrTimingData {
  double hull_ms = 0.0;
  double seed_ms = 0.0;
  double components_ms = 0.0;
  double d0_ms = 0.0;
  double d1_ms = 0.0;
  double d2_ms = 0.0;
  double d3_ms = 0.0;
  double d4_ms = 0.0;
  int growth_iterations = 0;
  double retriangulate_ms = 0.0;
  double splice_ms = 0.0;
  double replace_ms = 0.0;
  double legalize_ms = 0.0;
  double finalize_ms = 0.0;
  int patch_faces = 0;
  int flips = 0;
  double audit_ms = 0.0;
};

class CecrStageTimer {
 public:
  explicit CecrStageTimer(double* elapsed_ms) : elapsed_ms_(elapsed_ms) {
    if (elapsed_ms_ != nullptr) {
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~CecrStageTimer() { stop(); }

  void stop() {
    if (elapsed_ms_ != nullptr) {
      *elapsed_ms_ +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start_)
              .count();
      elapsed_ms_ = nullptr;
    }
  }

  CecrStageTimer(const CecrStageTimer&) = delete;
  CecrStageTimer& operator=(const CecrStageTimer&) = delete;

 private:
  double* elapsed_ms_;
  std::chrono::steady_clock::time_point start_{};
};

CecrCavitySet cecr_certify_and_grow(
    const DelaunayTriangulation& previous,
    const DelaunayTriangulation& moved,
    const CecrHullChange& hull_change,
    std::vector<std::uint8_t> in_patch,
    CecrTimingData* timing);

DelaunayResult build_delaunay(std::vector<Site> sites,
                              CecrTimingData* timing);

namespace {

class PredicateCounterReporter {
 public:
  PredicateCounterReporter() {
    static const bool diagnostic_enabled =
        (std::getenv("TENRYU_WELD_DIAG") != nullptr &&
         std::string(std::getenv("TENRYU_WELD_DIAG")) == std::string("1")) ||
        (std::getenv("TENRYU_DVCLP_DIAG") != nullptr &&
         std::string(std::getenv("TENRYU_DVCLP_DIAG")) == std::string("1"));
    if (diagnostic_enabled) {
      previous_ = detail::g_predicate_counters;
      previous_active_ = detail::g_predicate_counters_active;
      detail::g_predicate_counters = &data_;
      detail::g_predicate_counters_active = true;
      enabled_ = true;
    }
  }

  ~PredicateCounterReporter() {
    if (!enabled_) {
      return;
    }
    detail::g_predicate_counters = previous_;
    detail::g_predicate_counters_active = previous_active_;
    std::fprintf(stderr,
                 "[dt_pred] incircle_fast=%zu incircle_exact=%zu "
                 "orient_fast=%zu orient_exact=%zu flips=%zu "
                 "locate_steps=%zu\n",
                 data_.incircle_fast, data_.incircle_exact,
                 data_.orient_fast, data_.orient_exact, data_.flips,
                 data_.locate_steps);
  }

  PredicateCounterReporter(const PredicateCounterReporter&) = delete;
  PredicateCounterReporter& operator=(const PredicateCounterReporter&) =
      delete;

 private:
  detail::PredicateCounters data_{};
  detail::PredicateCounters* previous_ = nullptr;
  bool previous_active_ = false;
  bool enabled_ = false;
};

class CecrTimingReporter {
 public:
  CecrTimingReporter() {
    const char* const timing_env = std::getenv("TENRYU_CECR_TIMING");
    if (timing_env != nullptr && timing_env[0] != '\0') {
      timing_ = &data_;
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~CecrTimingReporter() {
    if (timing_ == nullptr) {
      return;
    }
    const double total_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start_)
            .count();
    std::fprintf(
        stderr,
        "[cecr_timing] hull=%.0f seed=%.0f comp=%.0f d0=%.0f d1=%.0f "
        "d2=%.0f d3=%.0f d4=%.0f grow_iters=%d retri=%.0f splice=%.0f "
        "replace=%.0f legalize=%.0f finalize=%.0f patch_faces=%d flips=%d "
        "audit=%.0f total=%.0f\n",
        data_.hull_ms, data_.seed_ms, data_.components_ms, data_.d0_ms,
        data_.d1_ms, data_.d2_ms, data_.d3_ms, data_.d4_ms,
        data_.growth_iterations, data_.retriangulate_ms, data_.splice_ms,
        data_.replace_ms, data_.legalize_ms, data_.finalize_ms,
        data_.patch_faces, data_.flips, data_.audit_ms, total_ms);
  }

  CecrTimingData* timing() { return timing_; }

 private:
  CecrTimingData data_{};
  std::chrono::steady_clock::time_point start_{};
  CecrTimingData* timing_ = nullptr;
};

struct CoordinateKey {
  std::uint64_t r;
  std::uint64_t z;

  bool operator==(const CoordinateKey& other) const {
    return r == other.r && z == other.z;
  }
};

struct CoordinateKeyHash {
  std::size_t operator()(const CoordinateKey& key) const {
    const std::size_t r = static_cast<std::size_t>(key.r ^ (key.r >> 32));
    const std::size_t z = static_cast<std::size_t>(key.z ^ (key.z >> 32));
    return r ^ (z + 0x9e3779b9U + (r << 6) + (r >> 2));
  }
};

std::uint64_t coordinate_bits(const double value) {
  if (value == 0.0) {
    return 0;
  }
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

CoordinateKey coordinate_key(const Site& site) {
  return CoordinateKey{coordinate_bits(site.r), coordinate_bits(site.z)};
}

struct EdgeKey {
  SiteId lo;
  SiteId hi;

  bool operator<(const EdgeKey& other) const {
    return lo < other.lo || (lo == other.lo && hi < other.hi);
  }
};

EdgeKey edge_key(const SiteId a, const SiteId b) {
  return EdgeKey{std::min(a, b), std::max(a, b)};
}

struct LegalizationEdgeKey {
  std::uint64_t lo;
  std::uint64_t hi;

  bool operator<(const LegalizationEdgeKey& other) const {
    return lo < other.lo || (lo == other.lo && hi < other.hi);
  }
};

struct InternalTriangle {
  std::array<SiteId, 3> vertex{};
  std::array<TriangleId, 3> neighbor{{kInvalidId, kInvalidId, kInvalidId}};
  bool alive = true;
};

struct EdgeRef {
  TriangleId tri;
  int edge;
};

struct Attachment {
  TriangleId outside_tri;
  int outside_edge;
};

enum class BuilderMode { STATIC, WARM };

class DelaunayBuilder {
 public:
  explicit DelaunayBuilder(const std::vector<Site>& sites,
                           const BuilderMode mode = BuilderMode::STATIC)
      : sites_(sites), mode_(mode) {}

  DelaunayBuilder(const std::vector<Site>& sites,
                  const DelaunayTriangulation& imported,
                  const BuilderMode mode = BuilderMode::STATIC,
                  const bool nonconvex_hull = false,
                  CecrTimingData* timing = nullptr)
      : sites_(sites),
        mode_(mode),
        nonconvex_hull_(nonconvex_hull),
        timing_(timing) {
    import_triangulation(imported);
  }

  bool ok() const { return failure_code_.empty(); }

  const std::string& failure_code() const { return failure_code_; }

  void initialize(const SiteId a, const SiteId b, const SiteId c) {
    std::array<SiteId, 3> vertices{{a, b, c}};
    if (orientation(vertices) < 0) {
      std::swap(vertices[0], vertices[1]);
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(orientation(vertices) > 0, "initial_triangle_orientation")) {
        return;
      }
    } else {
      TENRYU_ASSERT(orientation(vertices) > 0,
                    "initial Delaunay triangle must be exactly CCW");
    }
    last_triangle_ = add_triangle(vertices);
  }

  void insert(const SiteId site_id) {
    if (!ok()) {
      return;
    }
    const Location location = locate(site_id);
    if (!ok()) {
      return;
    }
    if (location.tri == kInvalidId) {
      if (nonconvex_hull_) {
        check(false, "patch_site_unlocated");
        return;
      }
      insert_outside_hull(site_id);
      return;
    }
    if (location.on_edge != kInvalidId) {
      insert_on_edge(site_id, location.on_edge);
      return;
    }
    insert_in_cavity(site_id, location.tri);
  }

  void legalize_all() {
    if (!ok()) {
      return;
    }
    std::set<LegalizationEdgeKey> pending;
    std::map<LegalizationEdgeKey, EdgeRef> edge_index;
    for (TriangleId triangle = 0;
         triangle < static_cast<TriangleId>(triangles_.size()); ++triangle) {
      if (!triangles_[triangle].alive) {
        continue;
      }
      for (int edge = 0; edge < 3; ++edge) {
        const TriangleId neighbor = triangles_[triangle].neighbor[edge];
        if (neighbor == kInvalidId) {
          continue;
        }
        const LegalizationEdgeKey key = legalization_edge_key(
            triangles_[triangle].vertex[edge],
            triangles_[triangle].vertex[(edge + 1) % 3]);
        edge_index.emplace(key, EdgeRef{triangle, edge});
        if (triangle > neighbor) {
          continue;
        }
        const bool illegal = edge_is_illegal(triangle, edge);
        if (!ok()) {
          return;
        }
        if (illegal) {
          pending.insert(key);
        }
      }
    }

    while (!pending.empty()) {
      const LegalizationEdgeKey key = *pending.begin();
      pending.erase(pending.begin());
      const auto indexed = edge_index.find(key);
      if (indexed == edge_index.end()) {
        continue;
      }
      const EdgeRef candidate = indexed->second;
      const std::vector<TriangleId> created =
          flip_edge_if_illegal(candidate.tri, candidate.edge);
      if (!ok()) {
        return;
      }
      if (created.empty()) {
        continue;
      }

      edge_index.erase(key);
      std::set<LegalizationEdgeKey> affected_keys;
      for (const TriangleId triangle : created) {
        const InternalTriangle& current = triangles_[triangle];
        for (int edge = 0; edge < 3; ++edge) {
          affected_keys.insert(legalization_edge_key(
              current.vertex[edge], current.vertex[(edge + 1) % 3]));
        }
      }
      for (const LegalizationEdgeKey& affected : affected_keys) {
        edge_index.erase(affected);
      }
      for (const TriangleId triangle : created) {
        const InternalTriangle& current = triangles_[triangle];
        for (int edge = 0; edge < 3; ++edge) {
          const TriangleId neighbor = current.neighbor[edge];
          if (neighbor == kInvalidId) {
            continue;
          }
          const LegalizationEdgeKey affected = legalization_edge_key(
              current.vertex[edge], current.vertex[(edge + 1) % 3]);
          const int reverse_edge = find_directed_edge(
              neighbor, current.vertex[(edge + 1) % 3],
              current.vertex[edge]);
          if (!ok()) {
            return;
          }
          const EdgeRef first{triangle, edge};
          const EdgeRef second{neighbor, reverse_edge};
          const EdgeRef representative =
              std::tie(first.tri, first.edge) <
                      std::tie(second.tri, second.edge)
                  ? first
                  : second;
          const auto inserted = edge_index.emplace(affected, representative);
          if (!inserted.second &&
              std::tie(representative.tri, representative.edge) <
                  std::tie(inserted.first->second.tri,
                           inserted.first->second.edge)) {
            inserted.first->second = representative;
          }
        }
      }

      for (const TriangleId triangle : created) {
        const InternalTriangle& current = triangles_[triangle];
        for (int edge = 0; edge < 3; ++edge) {
          const TriangleId neighbor = current.neighbor[edge];
          if (neighbor == kInvalidId ||
              std::find(created.begin(), created.end(), neighbor) !=
                  created.end()) {
            continue;
          }
          pending.insert(legalization_edge_key(
              current.vertex[edge], current.vertex[(edge + 1) % 3]));
        }
      }
      last_triangle_ = created.front();
    }
  }

  void splice_cecr_patch(const CecrCavity& cavity,
                         const CecrPatch& patch) {
    if (!ok()) {
      return;
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(cavity.ok && patch.ok, "cecr_inputs")) {
        return;
      }
    } else {
      TENRYU_ASSERT(cavity.ok && patch.ok,
                    "CECR splice requires certified inputs");
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(cavity.in_patch.size() <= triangles_.size(),
                 "cecr_patch_size")) {
        return;
      }
    } else {
      TENRYU_ASSERT(cavity.in_patch.size() <= triangles_.size(),
                    "CECR splice patch exceeds imported triangles");
    }
    std::vector<TriangleId> removed;
    for (TriangleId triangle = 0;
         triangle < static_cast<TriangleId>(cavity.in_patch.size());
         ++triangle) {
      if (cavity.in_patch[triangle] != 0) {
        removed.push_back(triangle);
      }
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(!removed.empty(), "cecr_empty_patch")) {
        return;
      }
    } else {
      TENRYU_ASSERT(!removed.empty(), "CECR splice region must not be empty");
    }
    // Interior cavities have no old hull arcs, so discarded is empty and the
    // Stage-1 replace_region call path is unchanged.
    std::vector<EdgeKey> discarded;
    discarded.reserve(cavity.old_arc_edges.size());
    for (const auto& edge : cavity.old_arc_edges) {
      discarded.push_back(edge_key(edge.first, edge.second));
    }
    replace_region(removed, patch.triangles, {}, discarded);
  }

  DelaunayTriangulation finalize() {
    DelaunayTriangulation dt;
    dt.sites = sites_;
    if (!ok()) {
      return dt;
    }

    std::vector<TriangleId> compact_id(triangles_.size(), kInvalidId);
    for (TriangleId old_id = 0;
         old_id < static_cast<TriangleId>(triangles_.size()); ++old_id) {
      if (triangles_[old_id].alive) {
        compact_id[old_id] = static_cast<TriangleId>(dt.tri_edge.size());
        const bool half_edge_count_in_range =
            dt.tri_edge.size() <= static_cast<std::size_t>(
                                      (std::numeric_limits<HalfEdgeId>::max() -
                                       2) /
                                      3);
        if (mode_ == BuilderMode::WARM) {
          if (!check(half_edge_count_in_range,
                     "finalize_half_edge_range")) {
            return dt;
          }
        } else {
          TENRYU_ASSERT(
              dt.tri_edge.size() <= static_cast<std::size_t>(
                                        (std::numeric_limits<HalfEdgeId>::max() -
                                         2) /
                                        3),
              "Delaunay half-edge count exceeds HalfEdgeId range");
        }
        dt.tri_edge.push_back(static_cast<HalfEdgeId>(
            3 * static_cast<HalfEdgeId>(dt.tri_edge.size())));
      }
    }

    const std::size_t half_edge_count = 3 * dt.tri_edge.size();
    if (mode_ == BuilderMode::WARM) {
      if (!check(half_edge_count <= static_cast<std::size_t>(
                                         std::numeric_limits<HalfEdgeId>::max()),
                 "finalize_half_edge_range")) {
        return dt;
      }
    } else {
      TENRYU_ASSERT(
          half_edge_count <=
              static_cast<std::size_t>(std::numeric_limits<HalfEdgeId>::max()),
          "Delaunay half-edge count exceeds HalfEdgeId range");
    }
    dt.he_origin.resize(half_edge_count);
    dt.he_twin.assign(half_edge_count, kInvalidId);
    dt.he_next.resize(half_edge_count);
    dt.he_face.resize(half_edge_count);

    for (TriangleId old_id = 0;
         old_id < static_cast<TriangleId>(triangles_.size()); ++old_id) {
      if (!triangles_[old_id].alive) {
        continue;
      }
      const TriangleId face = compact_id[old_id];
      const HalfEdgeId base = static_cast<HalfEdgeId>(3 * face);
      const InternalTriangle& triangle = triangles_[old_id];
      for (int edge = 0; edge < 3; ++edge) {
        const HalfEdgeId half_edge = base + edge;
        dt.he_origin[half_edge] = triangle.vertex[edge];
        dt.he_next[half_edge] = base + (edge + 1) % 3;
        dt.he_face[half_edge] = face;

        const TriangleId neighbor = triangle.neighbor[edge];
        if (neighbor == kInvalidId) {
          continue;
        }
        if (mode_ == BuilderMode::WARM) {
          if (!check(neighbor >= 0 &&
                         neighbor <
                             static_cast<TriangleId>(triangles_.size()) &&
                         triangles_[neighbor].alive,
                     "finalize_neighbor")) {
            return dt;
          }
        } else {
          TENRYU_ASSERT(
              neighbor >= 0 &&
                  neighbor < static_cast<TriangleId>(triangles_.size()) &&
                  triangles_[neighbor].alive,
              "live triangle has an invalid neighbor");
        }
        const SiteId a = triangle.vertex[edge];
        const SiteId b = triangle.vertex[(edge + 1) % 3];
        const int reverse_edge = find_directed_edge(neighbor, b, a);
        if (!ok()) {
          return dt;
        }
        if (mode_ == BuilderMode::WARM) {
          if (!check(reverse_edge >= 0, "finalize_reverse_edge")) {
            return dt;
          }
        } else {
          TENRYU_ASSERT(reverse_edge >= 0,
                        "neighbor is missing the reverse Delaunay edge");
        }
        dt.he_twin[half_edge] =
            static_cast<HalfEdgeId>(3 * compact_id[neighbor] + reverse_edge);
      }
    }
    return dt;
  }

 private:
  bool check(const bool condition, const char* code) const {
    if (condition) {
      return true;
    }
    if (mode_ == BuilderMode::STATIC) {
      TENRYU_ASSERT(condition, code);
    }
    if (failure_code_.empty()) {
      failure_code_ = code;
    }
    return false;
  }

  int orientation(const std::array<SiteId, 3>& vertices) const {
    return orient2d_sign_tess(sites_[vertices[0]], sites_[vertices[1]],
                              sites_[vertices[2]]);
  }

  TriangleId add_triangle(const std::array<SiteId, 3>& vertices) {
    if (mode_ == BuilderMode::WARM) {
      if (!check(orientation(vertices) > 0, "triangle_orientation")) {
        return kInvalidId;
      }
    } else {
      TENRYU_ASSERT(orientation(vertices) > 0,
                    "Delaunay triangle must be exactly CCW");
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(
              triangles_.size() <= static_cast<std::size_t>(
                                       (std::numeric_limits<HalfEdgeId>::max() -
                                        2) /
                                       3),
              "triangle_allocation_range")) {
        return kInvalidId;
      }
    } else {
      TENRYU_ASSERT(
          triangles_.size() <= static_cast<std::size_t>(
                                   (std::numeric_limits<HalfEdgeId>::max() - 2) /
                                   3),
          "Delaunay triangle allocation exceeds half-edge id range");
    }
    const TriangleId id = static_cast<TriangleId>(triangles_.size());
    triangles_.push_back(InternalTriangle{vertices});
    return id;
  }

  HalfEdgeId internal_half_edge_id(const TriangleId triangle,
                                   const int edge) const {
    const std::int64_t value = 3LL * triangle + edge;
    if (mode_ == BuilderMode::WARM) {
      if (!check(value <= std::numeric_limits<HalfEdgeId>::max(),
                 "internal_half_edge_range")) {
        return kInvalidId;
      }
    } else {
      TENRYU_ASSERT(value <= std::numeric_limits<HalfEdgeId>::max(),
                    "internal half-edge id exceeds HalfEdgeId range");
    }
    return static_cast<HalfEdgeId>(value);
  }

  void import_triangulation(const DelaunayTriangulation& imported) {
    const std::size_t half_edge_count = imported.he_origin.size();
    if (mode_ == BuilderMode::WARM) {
      if (!check(imported.sites.size() == sites_.size(),
                 "import_site_count")) {
        return;
      }
    } else {
      TENRYU_ASSERT(imported.sites.size() == sites_.size(),
                    "imported Delaunay site count mismatch");
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(imported.he_twin.size() == half_edge_count &&
                     imported.he_next.size() == half_edge_count &&
                     imported.he_face.size() == half_edge_count &&
                     half_edge_count == 3 * imported.tri_edge.size(),
                 "import_arrays")) {
        return;
      }
    } else {
      TENRYU_ASSERT(imported.he_twin.size() == half_edge_count &&
                        imported.he_next.size() == half_edge_count &&
                        imported.he_face.size() == half_edge_count &&
                        half_edge_count == 3 * imported.tri_edge.size(),
                    "imported Delaunay arrays are inconsistent");
    }

    triangles_.reserve(imported.tri_edge.size());
    std::vector<unsigned char> visited(half_edge_count, 0);
    for (TriangleId triangle = 0;
         triangle < static_cast<TriangleId>(imported.tri_edge.size());
         ++triangle) {
      const HalfEdgeId edge0 = imported.tri_edge[triangle];
      if (mode_ == BuilderMode::WARM) {
        if (!check(edge0 >= 0 &&
                       edge0 < static_cast<HalfEdgeId>(half_edge_count),
                   "import_triangle_edge")) {
          return;
        }
      } else {
        TENRYU_ASSERT(edge0 >= 0 &&
                          edge0 < static_cast<HalfEdgeId>(half_edge_count),
                      "imported Delaunay triangle edge is invalid");
      }
      const HalfEdgeId edge1 = imported.he_next[edge0];
      if (mode_ == BuilderMode::WARM) {
        if (!check(edge1 >= 0 &&
                       edge1 < static_cast<HalfEdgeId>(half_edge_count),
                   "import_triangle_cycle")) {
          return;
        }
      } else {
        TENRYU_ASSERT(edge1 >= 0 &&
                          edge1 < static_cast<HalfEdgeId>(half_edge_count),
                      "imported Delaunay triangle cycle is invalid");
      }
      const HalfEdgeId edge2 = imported.he_next[edge1];
      if (mode_ == BuilderMode::WARM) {
        if (!check(edge2 >= 0 &&
                       edge2 < static_cast<HalfEdgeId>(half_edge_count) &&
                       imported.he_next[edge2] == edge0 && edge0 != edge1 &&
                       edge1 != edge2 && edge2 != edge0,
                   "import_triangle_cycle")) {
          return;
        }
      } else {
        TENRYU_ASSERT(edge2 >= 0 &&
                          edge2 < static_cast<HalfEdgeId>(half_edge_count) &&
                          imported.he_next[edge2] == edge0 && edge0 != edge1 &&
                          edge1 != edge2 && edge2 != edge0,
                      "imported Delaunay triangle cycle is invalid");
      }
      const std::array<HalfEdgeId, 3> edges{{edge0, edge1, edge2}};
      std::array<SiteId, 3> vertices{};
      for (int edge = 0; edge < 3; ++edge) {
        const HalfEdgeId half_edge = edges[edge];
        if (mode_ == BuilderMode::WARM) {
          if (!check(!visited[half_edge] &&
                         imported.he_face[half_edge] == triangle,
                     "import_triangle_ownership")) {
            return;
          }
        } else {
          TENRYU_ASSERT(!visited[half_edge] &&
                            imported.he_face[half_edge] == triangle,
                        "imported Delaunay triangle ownership is invalid");
        }
        visited[half_edge] = 1;
        vertices[edge] = imported.he_origin[half_edge];
        if (mode_ == BuilderMode::WARM) {
          if (!check(vertices[edge] >= 0 &&
                         vertices[edge] <
                             static_cast<SiteId>(sites_.size()),
                     "import_vertex")) {
            return;
          }
        } else {
          TENRYU_ASSERT(vertices[edge] >= 0 &&
                            vertices[edge] < static_cast<SiteId>(sites_.size()),
                        "imported Delaunay vertex is invalid");
        }
      }
      triangles_.push_back(InternalTriangle{vertices});
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(std::find(visited.begin(), visited.end(), 0) == visited.end(),
                 "import_unrepresented_half_edge")) {
        return;
      }
    } else {
      TENRYU_ASSERT(
          std::find(visited.begin(), visited.end(), 0) == visited.end(),
          "imported Delaunay half-edge is unrepresented");
    }

    for (TriangleId triangle = 0;
         triangle < static_cast<TriangleId>(triangles_.size()); ++triangle) {
      const HalfEdgeId edge0 = imported.tri_edge[triangle];
      std::array<HalfEdgeId, 3> edges{{
          edge0, imported.he_next[edge0],
          imported.he_next[imported.he_next[edge0]]}};
      for (int edge = 0; edge < 3; ++edge) {
        const HalfEdgeId twin = imported.he_twin[edges[edge]];
        if (twin == kInvalidId) {
          continue;
        }
        if (mode_ == BuilderMode::WARM) {
          if (!check(twin >= 0 &&
                         twin < static_cast<HalfEdgeId>(half_edge_count) &&
                         imported.he_twin[twin] == edges[edge],
                     "import_twin")) {
            return;
          }
        } else {
          TENRYU_ASSERT(twin >= 0 &&
                            twin < static_cast<HalfEdgeId>(half_edge_count) &&
                            imported.he_twin[twin] == edges[edge],
                        "imported Delaunay twin is invalid");
        }
        const TriangleId neighbor = imported.he_face[twin];
        if (mode_ == BuilderMode::WARM) {
          if (!check(
                  neighbor >= 0 &&
                      neighbor < static_cast<TriangleId>(triangles_.size()) &&
                      imported.he_origin[edges[edge]] ==
                          imported.he_origin[imported.he_next[twin]] &&
                      imported.he_origin[twin] ==
                          imported.he_origin[imported.he_next[edges[edge]]],
                  "import_adjacency")) {
            return;
          }
        } else {
          TENRYU_ASSERT(
              neighbor >= 0 &&
                  neighbor < static_cast<TriangleId>(triangles_.size()) &&
                  imported.he_origin[edges[edge]] ==
                      imported.he_origin[imported.he_next[twin]] &&
                  imported.he_origin[twin] ==
                      imported.he_origin[imported.he_next[edges[edge]]],
              "imported Delaunay adjacency is inconsistent");
        }
        triangles_[triangle].neighbor[edge] = neighbor;
      }
    }
    last_triangle_ = triangles_.empty() ? kInvalidId : TriangleId{0};
  }

  int find_directed_edge(const TriangleId triangle,
                         const SiteId origin,
                         const SiteId destination) const {
    if (mode_ == BuilderMode::WARM) {
      if (!check(triangle >= 0 &&
                     triangle < static_cast<TriangleId>(triangles_.size()),
                 "triangle_id")) {
        return -1;
      }
    } else {
      TENRYU_ASSERT(triangle >= 0 &&
                        triangle < static_cast<TriangleId>(triangles_.size()),
                    "Delaunay triangle id is out of range");
    }
    const InternalTriangle& candidate = triangles_[triangle];
    for (int edge = 0; edge < 3; ++edge) {
      if (candidate.vertex[edge] == origin &&
          candidate.vertex[(edge + 1) % 3] == destination) {
        return edge;
      }
    }
    return -1;
  }

  int edge_opposite_site(const TriangleId triangle,
                         const SiteId site_id) const {
    const InternalTriangle& candidate = triangles_[triangle];
    for (int edge = 0; edge < 3; ++edge) {
      if (candidate.vertex[(edge + 2) % 3] == site_id) {
        return edge;
      }
    }
    return -1;
  }

  Location locate(const SiteId query) const {
    if (mode_ == BuilderMode::WARM) {
      if (!check(last_triangle_ >= 0 &&
                     last_triangle_ <
                         static_cast<TriangleId>(triangles_.size()) &&
                     triangles_[last_triangle_].alive,
                 "remembered_location")) {
        return Location{};
      }
    } else {
      TENRYU_ASSERT(last_triangle_ != kInvalidId &&
                        triangles_[last_triangle_].alive,
                    "remembered Delaunay location is invalid");
    }
    TriangleId triangle = last_triangle_;

    for (std::size_t step = 0; step <= triangles_.size(); ++step) {
      detail::count_locate_step();
      if (mode_ == BuilderMode::WARM) {
        if (!check(triangle >= 0 &&
                       triangle <
                           static_cast<TriangleId>(triangles_.size()) &&
                       triangles_[triangle].alive,
                   "walk_triangle")) {
          return Location{};
        }
      }
      const InternalTriangle& current = triangles_[triangle];
      if (mode_ == BuilderMode::STATIC) {
        TENRYU_ASSERT(current.alive,
                      "Delaunay walk entered a dead triangle");
      }

      std::array<int, 3> side{};
      bool inside = true;
      for (int edge = 0; edge < 3; ++edge) {
        side[edge] = orient2d_sign_tess(
            sites_[current.vertex[edge]],
            sites_[current.vertex[(edge + 1) % 3]], sites_[query]);
        inside = inside && side[edge] >= 0;
      }
      if (inside) {
        HalfEdgeId on_edge = kInvalidId;
        for (int edge = 0; edge < 3; ++edge) {
          if (side[edge] == 0 &&
              on_segment_exact(sites_[current.vertex[edge]],
                               sites_[current.vertex[(edge + 1) % 3]],
                               sites_[query])) {
            const HalfEdgeId candidate =
                internal_half_edge_id(triangle, edge);
            if (!ok()) {
              return Location{};
            }
            if (on_edge == kInvalidId || candidate < on_edge) {
              on_edge = candidate;
            }
          }
        }
        return Location{triangle, on_edge};
      }

      TriangleId next_triangle = kInvalidId;
      HalfEdgeId next_half_edge = kInvalidId;
      for (int edge = 0; edge < 3; ++edge) {
        if (side[edge] >= 0) {
          continue;
        }
        const TriangleId neighbor = current.neighbor[edge];
        if (neighbor == kInvalidId) {
          if (nonconvex_hull_) {
            // The imported patch boundary is not convex: exiting through a
            // hull edge does not prove the query lies outside the patch.
            return locate_by_exhaustive_scan(query);
          }
          return Location{};
        }
        const SiteId a = current.vertex[edge];
        const SiteId b = current.vertex[(edge + 1) % 3];
        const int reverse_edge = find_directed_edge(neighbor, b, a);
        if (!ok()) {
          return Location{};
        }
        if (mode_ == BuilderMode::WARM) {
          if (!check(reverse_edge >= 0, "walk_neighbor_reciprocity")) {
            return Location{};
          }
        } else {
          TENRYU_ASSERT(reverse_edge >= 0,
                        "Delaunay walk neighbor is not reciprocal");
        }
        const HalfEdgeId candidate =
            internal_half_edge_id(neighbor, reverse_edge);
        if (!ok()) {
          return Location{};
        }
        if (next_half_edge == kInvalidId || candidate < next_half_edge) {
          next_triangle = neighbor;
          next_half_edge = candidate;
        }
      }
      if (mode_ == BuilderMode::WARM) {
        if (!check(next_triangle != kInvalidId, "walk_next_triangle")) {
          return Location{};
        }
      } else {
        TENRYU_ASSERT(next_triangle != kInvalidId,
                      "Delaunay walk has no exact next triangle");
      }
      triangle = next_triangle;
    }

    if (mode_ == BuilderMode::WARM) {
      check(false, "walk_termination");
    } else {
      TENRYU_ASSERT(false, "remembering Delaunay walk failed to terminate");
    }
    return Location{};
  }

  // Deterministic total containment scan for nonconvex imported patches:
  // ascending live-triangle order, exact side tests; returns the smallest
  // containing triangle (with the same exact on-edge classification as the
  // walk), or an empty Location when no live triangle contains the query.
  Location locate_by_exhaustive_scan(const SiteId query) const {
    const auto locate_in_triangle = [&](const TriangleId triangle) {
      const InternalTriangle& current = triangles_[triangle];
      std::array<int, 3> side{};
      bool inside = true;
      for (int edge = 0; edge < 3; ++edge) {
        side[edge] = orient2d_sign_tess(
            sites_[current.vertex[edge]],
            sites_[current.vertex[(edge + 1) % 3]], sites_[query]);
        inside = inside && side[edge] >= 0;
      }
      if (!inside) {
        return Location{};
      }
      HalfEdgeId on_edge = kInvalidId;
      for (int edge = 0; edge < 3; ++edge) {
        if (side[edge] == 0 &&
            on_segment_exact(sites_[current.vertex[edge]],
                             sites_[current.vertex[(edge + 1) % 3]],
                             sites_[query])) {
          const HalfEdgeId candidate =
              internal_half_edge_id(triangle, edge);
          if (!ok()) {
            return Location{};
          }
          if (on_edge == kInvalidId || candidate < on_edge) {
            on_edge = candidate;
          }
        }
      }
      return Location{triangle, on_edge};
    };

    std::vector<TriangleId> live_triangles;
    double minimum_r = std::numeric_limits<double>::max();
    double maximum_r = std::numeric_limits<double>::lowest();
    double minimum_z = std::numeric_limits<double>::max();
    double maximum_z = std::numeric_limits<double>::lowest();
    for (TriangleId triangle = 0;
         triangle < static_cast<TriangleId>(triangles_.size()); ++triangle) {
      const InternalTriangle& current = triangles_[triangle];
      if (!current.alive) {
        continue;
      }
      live_triangles.push_back(triangle);
      for (const SiteId vertex : current.vertex) {
        minimum_r = std::min(minimum_r, sites_[vertex].r);
        maximum_r = std::max(maximum_r, sites_[vertex].r);
        minimum_z = std::min(minimum_z, sites_[vertex].z);
        maximum_z = std::max(maximum_z, sites_[vertex].z);
      }
    }

    const auto full_scan = [&]() {
      for (const TriangleId triangle : live_triangles) {
        detail::count_locate_step();
        const Location location = locate_in_triangle(triangle);
        if (!ok() || location.tri != kInvalidId) {
          return location;
        }
      }
      return Location{};
    };

    if (live_triangles.empty()) {
      return Location{};
    }
    const double count_scale =
        std::sqrt(static_cast<double>(live_triangles.size()));
    const double extent_r = maximum_r - minimum_r;
    const double extent_z = maximum_z - minimum_z;
    const double cell_size_r = extent_r / count_scale;
    const double cell_size_z = extent_z / count_scale;
    if (!(cell_size_r > 0.0) || !(cell_size_z > 0.0) ||
        !std::isfinite(cell_size_r) || !std::isfinite(cell_size_z)) {
      return full_scan();
    }
    const std::size_t cells_per_axis = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(count_scale)));
    const auto cell_index = [](const double value, const double minimum,
                               const double cell_size,
                               const std::size_t cell_count) {
      const double scaled = std::floor((value - minimum) / cell_size);
      if (!(scaled > 0.0)) {
        return std::size_t{0};
      }
      if (scaled >= static_cast<double>(cell_count)) {
        return cell_count - 1;
      }
      return static_cast<std::size_t>(scaled);
    };

    std::vector<std::vector<TriangleId>> grid(cells_per_axis *
                                               cells_per_axis);
    for (const TriangleId triangle : live_triangles) {
      const InternalTriangle& current = triangles_[triangle];
      double triangle_minimum_r = std::numeric_limits<double>::max();
      double triangle_maximum_r = std::numeric_limits<double>::lowest();
      double triangle_minimum_z = std::numeric_limits<double>::max();
      double triangle_maximum_z = std::numeric_limits<double>::lowest();
      for (const SiteId vertex : current.vertex) {
        triangle_minimum_r =
            std::min(triangle_minimum_r, sites_[vertex].r);
        triangle_maximum_r =
            std::max(triangle_maximum_r, sites_[vertex].r);
        triangle_minimum_z =
            std::min(triangle_minimum_z, sites_[vertex].z);
        triangle_maximum_z =
            std::max(triangle_maximum_z, sites_[vertex].z);
      }
      const std::size_t first_r = cell_index(
          triangle_minimum_r, minimum_r, cell_size_r, cells_per_axis);
      const std::size_t last_r = cell_index(
          triangle_maximum_r, minimum_r, cell_size_r, cells_per_axis);
      const std::size_t first_z = cell_index(
          triangle_minimum_z, minimum_z, cell_size_z, cells_per_axis);
      const std::size_t last_z = cell_index(
          triangle_maximum_z, minimum_z, cell_size_z, cells_per_axis);
      for (std::size_t z = first_z; z <= last_z; ++z) {
        for (std::size_t r = first_r; r <= last_r; ++r) {
          grid[z * cells_per_axis + r].push_back(triangle);
        }
      }
    }

    const std::size_t query_r =
        cell_index(sites_[query].r, minimum_r, cell_size_r, cells_per_axis);
    const std::size_t query_z =
        cell_index(sites_[query].z, minimum_z, cell_size_z, cells_per_axis);
    const std::vector<TriangleId>& candidates =
        grid[query_z * cells_per_axis + query_r];
    for (const TriangleId triangle : candidates) {
      detail::count_locate_step();
      const Location location = locate_in_triangle(triangle);
      if (!ok() || location.tri != kInvalidId) {
        return location;
      }
    }
    return full_scan();
  }

  std::vector<TriangleId> replace_region(
      const std::vector<TriangleId>& removed,
      const std::vector<std::array<SiteId, 3>>& replacement,
      std::map<EdgeKey, Attachment> extra_attachments = {},
      const std::vector<EdgeKey>& discarded_boundary_edges = {}) {
    if (!ok()) {
      return {};
    }
    const std::size_t old_triangle_count = triangles_.size();
    std::vector<unsigned char> is_removed(old_triangle_count, 0);
    for (const TriangleId triangle : removed) {
      if (mode_ == BuilderMode::WARM) {
        if (!check(triangle >= 0 &&
                       triangle < static_cast<TriangleId>(old_triangle_count) &&
                       triangles_[triangle].alive,
                   "region_triangle")) {
          return {};
        }
      } else {
        TENRYU_ASSERT(
            triangle >= 0 &&
                triangle < static_cast<TriangleId>(old_triangle_count) &&
                triangles_[triangle].alive,
            "Delaunay region contains an invalid triangle");
      }
      if (mode_ == BuilderMode::WARM) {
        if (!check(!is_removed[triangle], "region_duplicate")) {
          return {};
        }
      } else {
        TENRYU_ASSERT(!is_removed[triangle],
                      "Delaunay region contains a duplicate triangle");
      }
      is_removed[triangle] = 1;
    }

    std::map<EdgeKey, Attachment> attachments =
        std::move(extra_attachments);
    for (const TriangleId triangle : removed) {
      const InternalTriangle& old = triangles_[triangle];
      for (int edge = 0; edge < 3; ++edge) {
        const TriangleId neighbor = old.neighbor[edge];
        if (neighbor != kInvalidId && is_removed[neighbor]) {
          continue;
        }
        const SiteId a = old.vertex[edge];
        const SiteId b = old.vertex[(edge + 1) % 3];
        int outside_edge = -1;
        if (neighbor != kInvalidId) {
          outside_edge = find_directed_edge(neighbor, b, a);
          if (!ok()) {
            return {};
          }
          if (mode_ == BuilderMode::WARM) {
            if (!check(outside_edge >= 0 &&
                           triangles_[neighbor].neighbor[outside_edge] ==
                               triangle,
                       "region_boundary_reciprocity")) {
              return {};
            }
          } else {
            TENRYU_ASSERT(outside_edge >= 0 &&
                              triangles_[neighbor].neighbor[outside_edge] ==
                                  triangle,
                          "Delaunay region boundary is not reciprocal");
          }
        }
        const auto inserted = attachments.emplace(
            edge_key(a, b), Attachment{neighbor, outside_edge});
        if (mode_ == BuilderMode::WARM) {
          if (!check(inserted.second, "region_boundary_nonmanifold")) {
            return {};
          }
        } else {
          TENRYU_ASSERT(inserted.second,
                        "Delaunay region boundary is nonmanifold");
        }
      }
    }

    for (const EdgeKey& discarded : discarded_boundary_edges) {
      if (mode_ == BuilderMode::WARM) {
        if (!check(attachments.erase(discarded) == 1,
                   "discarded_boundary_edge")) {
          return {};
        }
      } else {
        TENRYU_ASSERT(attachments.erase(discarded) == 1,
                      "discarded Delaunay boundary edge is absent");
      }
    }
    for (const TriangleId triangle : removed) {
      triangles_[triangle].alive = false;
    }

    std::vector<TriangleId> created;
    created.reserve(replacement.size());
    for (const auto& vertices : replacement) {
      const TriangleId triangle = add_triangle(vertices);
      if (!ok()) {
        return {};
      }
      created.push_back(triangle);
    }

    std::map<EdgeKey, EdgeRef> unmatched;
    for (const TriangleId triangle : created) {
      InternalTriangle& current = triangles_[triangle];
      for (int edge = 0; edge < 3; ++edge) {
        const SiteId a = current.vertex[edge];
        const SiteId b = current.vertex[(edge + 1) % 3];
        const EdgeKey key = edge_key(a, b);

        const auto attachment = attachments.find(key);
        if (attachment != attachments.end()) {
          const TriangleId outside = attachment->second.outside_tri;
          const int outside_edge = attachment->second.outside_edge;
          if (outside != kInvalidId) {
            if (mode_ == BuilderMode::WARM) {
              if (!check(outside >= 0 &&
                             outside <
                                 static_cast<TriangleId>(triangles_.size()) &&
                             triangles_[outside].alive && outside_edge >= 0 &&
                             outside_edge < 3,
                         "attachment")) {
                return {};
              }
            } else {
              TENRYU_ASSERT(triangles_[outside].alive && outside_edge >= 0,
                            "Delaunay attachment is invalid");
            }
            const InternalTriangle& outside_triangle = triangles_[outside];
            if (mode_ == BuilderMode::WARM) {
              if (!check(
                      outside_triangle.vertex[outside_edge] == b &&
                          outside_triangle
                                  .vertex[(outside_edge + 1) % 3] == a,
                      "attachment_orientation")) {
                return {};
              }
            } else {
              TENRYU_ASSERT(
                  outside_triangle.vertex[outside_edge] == b &&
                      outside_triangle.vertex[(outside_edge + 1) % 3] == a,
                  "Delaunay attachment orientation is inconsistent");
            }
            current.neighbor[edge] = outside;
            triangles_[outside].neighbor[outside_edge] = triangle;
          }
          attachments.erase(attachment);
          continue;
        }

        const auto pending = unmatched.find(key);
        if (pending == unmatched.end()) {
          unmatched.emplace(key, EdgeRef{triangle, edge});
          continue;
        }
        const EdgeRef other = pending->second;
        if (mode_ == BuilderMode::WARM) {
          if (!check(
                  triangles_[other.tri].vertex[other.edge] == b &&
                      triangles_[other.tri]
                              .vertex[(other.edge + 1) % 3] == a,
                  "new_triangle_orientation")) {
            return {};
          }
        } else {
          TENRYU_ASSERT(
              triangles_[other.tri].vertex[other.edge] == b &&
                  triangles_[other.tri].vertex[(other.edge + 1) % 3] == a,
              "new Delaunay triangles have inconsistent orientation");
        }
        current.neighbor[edge] = other.tri;
        triangles_[other.tri].neighbor[other.edge] = triangle;
        unmatched.erase(pending);
      }
    }

    if (mode_ == BuilderMode::WARM) {
      if (!check(attachments.empty(), "missing_boundary_edge")) {
        return {};
      }
    } else {
      TENRYU_ASSERT(attachments.empty(),
                    "replacement omitted a Delaunay region boundary edge");
    }
    return created;
  }

  void insert_in_cavity(const SiteId site_id,
                        const TriangleId containing_triangle) {
    std::vector<unsigned char> in_cavity(triangles_.size(), 0);
    std::vector<TriangleId> pending{containing_triangle};
    std::vector<TriangleId> cavity;
    while (!pending.empty()) {
      const TriangleId triangle = pending.back();
      pending.pop_back();
      if (mode_ == BuilderMode::WARM) {
        if (!check(triangle >= 0 &&
                       triangle <
                           static_cast<TriangleId>(triangles_.size()) &&
                       triangles_[triangle].alive,
                   "cavity_triangle")) {
          return;
        }
      }
      if (in_cavity[triangle]) {
        continue;
      }
      const InternalTriangle& candidate = triangles_[triangle];
      if (mode_ == BuilderMode::STATIC) {
        TENRYU_ASSERT(candidate.alive,
                      "Delaunay cavity reached a dead triangle");
      }
      const int in_circle =
          incircle_sos(sites_[candidate.vertex[0]],
                       sites_[candidate.vertex[1]],
                       sites_[candidate.vertex[2]], sites_[site_id]);
      if (in_circle < 0) {
        continue;
      }
      if (mode_ == BuilderMode::WARM) {
        if (!check(in_circle > 0, "cavity_incircle_sign")) {
          return;
        }
      } else {
        TENRYU_ASSERT(in_circle > 0,
                      "incircle_sos returned an unresolved sign");
      }
      in_cavity[triangle] = 1;
      cavity.push_back(triangle);
      for (const TriangleId neighbor : candidate.neighbor) {
        if (neighbor != kInvalidId && !in_cavity[neighbor]) {
          pending.push_back(neighbor);
        }
      }
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(!cavity.empty(), "empty_cavity")) {
        return;
      }
    } else {
      TENRYU_ASSERT(!cavity.empty(),
                    "containing triangle is absent from Delaunay cavity");
    }

    std::vector<std::array<SiteId, 3>> replacement;
    for (const TriangleId triangle : cavity) {
      const InternalTriangle& removed = triangles_[triangle];
      for (int edge = 0; edge < 3; ++edge) {
        const TriangleId neighbor = removed.neighbor[edge];
        if (neighbor != kInvalidId && in_cavity[neighbor]) {
          continue;
        }
        const SiteId a = removed.vertex[edge];
        const SiteId b = removed.vertex[(edge + 1) % 3];
        if (mode_ == BuilderMode::WARM) {
          if (!check(orient2d_sign_tess(sites_[a], sites_[b],
                                        sites_[site_id]) > 0,
                     "cavity_boundary_visibility")) {
            return;
          }
        } else {
          TENRYU_ASSERT(
              orient2d_sign_tess(sites_[a], sites_[b], sites_[site_id]) > 0,
              "Delaunay cavity boundary is not visible from inserted site");
        }
        replacement.push_back({a, b, site_id});
      }
    }

    const std::vector<TriangleId> created =
        replace_region(cavity, replacement);
    if (!ok()) {
      return;
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(!created.empty(), "empty_cavity_replacement")) {
        return;
      }
    } else {
      TENRYU_ASSERT(!created.empty(), "Delaunay cavity produced no triangles");
    }
    last_triangle_ = created.front();
  }

  void insert_on_edge(const SiteId site_id,
                      const HalfEdgeId half_edge) {
    const TriangleId triangle = half_edge / 3;
    const int edge = half_edge % 3;
    if (mode_ == BuilderMode::WARM) {
      if (!check(triangle >= 0 &&
                     triangle < static_cast<TriangleId>(triangles_.size()) &&
                     triangles_[triangle].alive,
                 "point_on_edge_location")) {
        return;
      }
    } else {
      TENRYU_ASSERT(triangle >= 0 &&
                        triangle < static_cast<TriangleId>(triangles_.size()) &&
                        triangles_[triangle].alive,
                    "point-on-edge location is invalid");
    }
    const InternalTriangle& left = triangles_[triangle];
    const SiteId a = left.vertex[edge];
    const SiteId b = left.vertex[(edge + 1) % 3];
    const SiteId c = left.vertex[(edge + 2) % 3];
    if (mode_ == BuilderMode::WARM) {
      if (!check(on_segment_exact(sites_[a], sites_[b], sites_[site_id]),
                 "point_on_edge_containment")) {
        return;
      }
    } else {
      TENRYU_ASSERT(on_segment_exact(sites_[a], sites_[b], sites_[site_id]),
                    "point-on-edge insertion requires exact containment");
    }

    std::vector<TriangleId> removed{triangle};
    std::vector<std::array<SiteId, 3>> replacement{
        {a, site_id, c}, {site_id, b, c}};
    std::vector<EdgeKey> discarded_boundary_edges;

    const TriangleId twin_triangle = left.neighbor[edge];
    if (twin_triangle == kInvalidId) {
      discarded_boundary_edges.push_back(edge_key(a, b));
    } else {
      const int twin_edge = find_directed_edge(twin_triangle, b, a);
      if (!ok()) {
        return;
      }
      if (mode_ == BuilderMode::WARM) {
        if (!check(twin_edge >= 0, "point_on_edge_twin")) {
          return;
        }
      } else {
        TENRYU_ASSERT(twin_edge >= 0,
                      "point-on-edge twin is not reciprocal");
      }
      const SiteId d = triangles_[twin_triangle].vertex[(twin_edge + 2) % 3];
      removed.push_back(twin_triangle);
      replacement.push_back({b, site_id, d});
      replacement.push_back({site_id, a, d});
    }

    const std::vector<TriangleId> created = replace_region(
        removed, replacement, {}, discarded_boundary_edges);
    if (!ok()) {
      return;
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(!created.empty(), "point_on_edge_replacement")) {
        return;
      }
    }
    last_triangle_ = created.front();
    legalize_inserted_site(site_id, created);
  }

  void insert_outside_hull(const SiteId site_id) {
    std::map<EdgeKey, Attachment> attachments;
    std::vector<std::array<SiteId, 3>> replacement;
    for (TriangleId triangle = 0;
         triangle < static_cast<TriangleId>(triangles_.size()); ++triangle) {
      if (!triangles_[triangle].alive) {
        continue;
      }
      const InternalTriangle& candidate = triangles_[triangle];
      for (int edge = 0; edge < 3; ++edge) {
        if (candidate.neighbor[edge] != kInvalidId) {
          continue;
        }
        const SiteId a = candidate.vertex[edge];
        const SiteId b = candidate.vertex[(edge + 1) % 3];
        // Hull half-edges are triangle-oriented. Reversing them gives the
        // exterior orientation for W2's exact orient2d > 0 visibility test.
        if (orient2d_sign_tess(sites_[b], sites_[a], sites_[site_id]) <= 0) {
          continue;
        }
        const auto inserted = attachments.emplace(
            edge_key(a, b), Attachment{triangle, edge});
        if (mode_ == BuilderMode::WARM) {
          if (!check(inserted.second, "visible_hull_edge_duplicate")) {
            return;
          }
        } else {
          TENRYU_ASSERT(inserted.second,
                        "visible Delaunay hull edge is duplicated");
        }
        replacement.push_back({b, a, site_id});
      }
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(!replacement.empty(), "outside_hull_visibility")) {
        return;
      }
    } else {
      TENRYU_ASSERT(!replacement.empty(),
                    "outside Delaunay site has no exactly visible hull edge");
    }

    const std::vector<TriangleId> created =
        replace_region({}, replacement, std::move(attachments));
    if (!ok()) {
      return;
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(!created.empty(), "outside_hull_replacement")) {
        return;
      }
    }
    last_triangle_ = created.front();
    legalize_inserted_site(site_id, created);
  }

  LegalizationEdgeKey legalization_edge_key(const SiteId a,
                                             const SiteId b) const {
    const std::uint64_t a_id = sites_[a].stable_id;
    const std::uint64_t b_id = sites_[b].stable_id;
    return LegalizationEdgeKey{std::min(a_id, b_id), std::max(a_id, b_id)};
  }

  bool edge_is_illegal(const TriangleId triangle, const int edge) const {
    if (mode_ == BuilderMode::WARM) {
      if (!check(triangle >= 0 &&
                     triangle < static_cast<TriangleId>(triangles_.size()) &&
                     triangles_[triangle].alive && edge >= 0 && edge < 3,
                 "legalization_edge")) {
        return false;
      }
    } else {
      TENRYU_ASSERT(triangle >= 0 &&
                        triangle < static_cast<TriangleId>(triangles_.size()) &&
                        triangles_[triangle].alive,
                    "Delaunay legalization edge is invalid");
    }
    const InternalTriangle& left = triangles_[triangle];
    const TriangleId right_id = left.neighbor[edge];
    if (mode_ == BuilderMode::WARM) {
      if (!check(right_id >= 0 &&
                     right_id < static_cast<TriangleId>(triangles_.size()) &&
                     triangles_[right_id].alive,
                 "legalization_interior_edge")) {
        return false;
      }
    } else {
      TENRYU_ASSERT(
          right_id >= 0 &&
              right_id < static_cast<TriangleId>(triangles_.size()) &&
              triangles_[right_id].alive,
          "Delaunay legalization requires an interior edge");
    }
    const SiteId a = left.vertex[edge];
    const SiteId b = left.vertex[(edge + 1) % 3];
    const SiteId c = left.vertex[(edge + 2) % 3];
    const int right_edge = find_directed_edge(right_id, b, a);
    if (!ok()) {
      return false;
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(right_edge >= 0, "legalization_twin")) {
        return false;
      }
    } else {
      TENRYU_ASSERT(right_edge >= 0,
                    "Delaunay legalization twin is not reciprocal");
    }
    const SiteId d = triangles_[right_id].vertex[(right_edge + 2) % 3];
    return incircle_sos(sites_[a], sites_[b], sites_[c], sites_[d]) > 0;
  }

  std::vector<TriangleId> flip_edge_if_illegal(const TriangleId triangle,
                                                const int edge) {
    if (!ok()) {
      return {};
    }
    const InternalTriangle& left = triangles_[triangle];
    const TriangleId right_id = left.neighbor[edge];
    if (right_id == kInvalidId) {
      return {};
    }
    const SiteId a = left.vertex[edge];
    const SiteId b = left.vertex[(edge + 1) % 3];
    const SiteId c = left.vertex[(edge + 2) % 3];
    const int right_edge = find_directed_edge(right_id, b, a);
    if (!ok()) {
      return {};
    }
    if (mode_ == BuilderMode::WARM) {
      if (!check(right_edge >= 0, "legalization_twin")) {
        return {};
      }
    } else {
      TENRYU_ASSERT(right_edge >= 0,
                    "Delaunay legalization twin is not reciprocal");
    }
    const SiteId d = triangles_[right_id].vertex[(right_edge + 2) % 3];

    if (incircle_sos(sites_[a], sites_[b], sites_[c], sites_[d]) < 0) {
      return {};
    }

    const std::array<SiteId, 3> first{{c, d, b}};
    const std::array<SiteId, 3> second{{d, c, a}};
    if (orientation(first) <= 0 || orientation(second) <= 0) {
      return {};
    }

    std::vector<TriangleId> created =
        replace_region({triangle, right_id}, {first, second});
    if (!created.empty()) {
      detail::count_flip();
      if (timing_ != nullptr) {
        ++timing_->flips;
      }
    }
    return created;
  }

  void legalize_inserted_site(const SiteId site_id,
                              const std::vector<TriangleId>& initial) {
    std::vector<EdgeRef> pending;
    pending.reserve(initial.size());
    for (const TriangleId triangle : initial) {
      const int edge = edge_opposite_site(triangle, site_id);
      if (mode_ == BuilderMode::WARM) {
        if (!check(edge >= 0, "inserted_site_replacement")) {
          return;
        }
      } else {
        TENRYU_ASSERT(edge >= 0,
                      "inserted site is absent from replacement triangle");
      }
      pending.push_back(EdgeRef{triangle, edge});
    }

    while (!pending.empty()) {
      const EdgeRef candidate = pending.back();
      pending.pop_back();
      if (candidate.tri < 0 ||
          candidate.tri >= static_cast<TriangleId>(triangles_.size()) ||
          !triangles_[candidate.tri].alive) {
        continue;
      }
      const int edge = edge_opposite_site(candidate.tri, site_id);
      if (edge < 0 || edge != candidate.edge) {
        continue;
      }

      const std::vector<TriangleId> created =
          flip_edge_if_illegal(candidate.tri, edge);
      if (!ok()) {
        return;
      }
      if (created.empty()) {
        continue;
      }
      for (const TriangleId triangle : created) {
        const int opposite = edge_opposite_site(triangle, site_id);
        if (mode_ == BuilderMode::WARM) {
          if (!check(opposite >= 0, "flipped_inserted_site")) {
            return;
          }
        } else {
          TENRYU_ASSERT(opposite >= 0,
                        "flipped triangle lost the inserted site");
        }
        pending.push_back(EdgeRef{triangle, opposite});
      }
      last_triangle_ = created.front();
    }
  }

  const std::vector<Site>& sites_;
  BuilderMode mode_ = BuilderMode::STATIC;
  bool nonconvex_hull_ = false;
  mutable std::string failure_code_;
  std::vector<InternalTriangle> triangles_;
  TriangleId last_triangle_ = kInvalidId;
  CecrTimingData* timing_ = nullptr;
};

std::array<SiteId, 3> triangle_vertices(
    const DelaunayTriangulation& dt,
    const TriangleId triangle) {
  TENRYU_ASSERT(triangle >= 0 &&
                    triangle < static_cast<TriangleId>(dt.tri_edge.size()),
                "Delaunay triangle id is out of range");
  const HalfEdgeId edge0 = dt.tri_edge[triangle];
  TENRYU_ASSERT(edge0 >= 0 &&
                    edge0 < static_cast<HalfEdgeId>(dt.he_origin.size()),
                "Delaunay triangle edge is invalid");
  const HalfEdgeId edge1 = dt.he_next[edge0];
  TENRYU_ASSERT(edge1 >= 0 &&
                    edge1 < static_cast<HalfEdgeId>(dt.he_origin.size()),
                "Delaunay triangle cycle is invalid");
  const HalfEdgeId edge2 = dt.he_next[edge1];
  TENRYU_ASSERT(edge2 >= 0 &&
                    edge2 < static_cast<HalfEdgeId>(dt.he_origin.size()) &&
                    dt.he_next[edge2] == edge0,
                "Delaunay triangle cycle is invalid");
  const std::array<SiteId, 3> vertices{{dt.he_origin[edge0],
                                        dt.he_origin[edge1],
                                        dt.he_origin[edge2]}};
  for (const SiteId site : vertices) {
    TENRYU_ASSERT(site >= 0 && site < static_cast<SiteId>(dt.sites.size()),
                  "Delaunay triangle vertex is invalid");
  }
  return vertices;
}

Expansion signed_twice_area(const Site& a, const Site& b, const Site& c) {
  return expansion_cross(exact_difference(b.r, a.r),
                         exact_difference(b.z, a.z),
                         exact_difference(c.r, a.r),
                         exact_difference(c.z, a.z));
}

}  // namespace

std::vector<SiteId> exact_convex_hull(const std::vector<Site>& sites) {
  std::vector<SiteId> sorted(sites.size());
  std::iota(sorted.begin(), sorted.end(), SiteId{0});
  std::sort(sorted.begin(), sorted.end(),
            [&sites](const SiteId lhs, const SiteId rhs) {
              if (sites[lhs].r != sites[rhs].r) {
                return sites[lhs].r < sites[rhs].r;
              }
              if (sites[lhs].z != sites[rhs].z) {
                return sites[lhs].z < sites[rhs].z;
              }
              return sites[lhs].stable_id < sites[rhs].stable_id;
            });

  std::vector<SiteId> lower;
  lower.reserve(sorted.size());
  for (const SiteId site : sorted) {
    while (lower.size() >= 2 &&
           orient2d_sign_tess(sites[lower[lower.size() - 2]],
                              sites[lower.back()], sites[site]) < 0) {
      lower.pop_back();
    }
    lower.push_back(site);
  }

  std::vector<SiteId> upper;
  upper.reserve(sorted.size());
  for (auto site = sorted.rbegin(); site != sorted.rend(); ++site) {
    while (upper.size() >= 2 &&
           orient2d_sign_tess(sites[upper[upper.size() - 2]],
                              sites[upper.back()], sites[*site]) < 0) {
      upper.pop_back();
    }
    upper.push_back(*site);
  }

  TENRYU_ASSERT(!lower.empty() && !upper.empty(),
                "warm Delaunay hull requires sites");
  lower.pop_back();
  upper.pop_back();
  lower.insert(lower.end(), upper.begin(), upper.end());
  return lower;
}

namespace {

std::vector<SiteId> triangulation_hull_cycle(
    const DelaunayTriangulation& dt) {
  TENRYU_ASSERT(dt.he_twin.size() == dt.he_origin.size() &&
                    dt.he_next.size() == dt.he_origin.size(),
                "Delaunay hull arrays are inconsistent");
  std::map<SiteId, SiteId> next_site;
  std::set<SiteId> incoming;
  for (HalfEdgeId edge = 0;
       edge < static_cast<HalfEdgeId>(dt.he_origin.size()); ++edge) {
    if (dt.he_twin[edge] != kInvalidId) {
      continue;
    }
    const HalfEdgeId next = dt.he_next[edge];
    TENRYU_ASSERT(next >= 0 &&
                      next < static_cast<HalfEdgeId>(dt.he_origin.size()),
                  "Delaunay hull edge has invalid next edge");
    const SiteId origin = dt.he_origin[edge];
    const SiteId destination = dt.he_origin[next];
    TENRYU_ASSERT(next_site.emplace(origin, destination).second,
                  "Delaunay hull has duplicate outgoing edges");
    TENRYU_ASSERT(incoming.insert(destination).second,
                  "Delaunay hull has duplicate incoming edges");
  }
  TENRYU_ASSERT(!next_site.empty() && incoming.size() == next_site.size(),
                "Delaunay hull cycle is empty or inconsistent");

  std::vector<SiteId> cycle;
  cycle.reserve(next_site.size());
  const SiteId first = next_site.begin()->first;
  SiteId current = first;
  do {
    TENRYU_ASSERT(cycle.size() < next_site.size(),
                  "Delaunay hull cycle does not close");
    cycle.push_back(current);
    const auto next = next_site.find(current);
    TENRYU_ASSERT(next != next_site.end(),
                  "Delaunay hull cycle has a missing edge");
    current = next->second;
  } while (current != first);
  TENRYU_ASSERT(cycle.size() == next_site.size(),
                "Delaunay hull contains multiple cycles");
  return cycle;
}

bool cyclic_sequences_equal(const std::vector<SiteId>& lhs,
                            const std::vector<SiteId>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  const auto rhs_start = std::find(rhs.begin(), rhs.end(), lhs.front());
  if (rhs_start == rhs.end()) {
    return false;
  }
  const std::size_t offset =
      static_cast<std::size_t>(rhs_start - rhs.begin());
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    if (lhs[index] != rhs[(offset + index) % rhs.size()]) {
      return false;
    }
  }
  return true;
}

bool exact_area_identity(const DelaunayTriangulation& dt,
                         const std::vector<SiteId>& hull) {
  Expansion triangle_area;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    const std::array<SiteId, 3> vertices =
        triangle_vertices(dt, triangle);
    triangle_area = expansion_sum(
        triangle_area,
        signed_twice_area(dt.sites[vertices[0]], dt.sites[vertices[1]],
                          dt.sites[vertices[2]]));
  }

  Expansion hull_area;
  for (std::size_t index = 1; index + 1 < hull.size(); ++index) {
    hull_area = expansion_sum(
        hull_area,
        signed_twice_area(dt.sites[hull[0]], dt.sites[hull[index]],
                          dt.sites[hull[index + 1]]));
  }
  return expansion_sign(expansion_difference(triangle_area, hull_area)) == 0;
}

struct EarSeedResult {
  std::vector<std::array<SiteId, 3>> triangles;
  bool ok = false;
};

bool point_in_closed_triangle(const std::vector<Site>& sites,
                              const SiteId point,
                              const std::array<SiteId, 3>& triangle) {
  return orient2d_sign_tess(sites[triangle[0]], sites[triangle[1]],
                            sites[point]) >= 0 &&
         orient2d_sign_tess(sites[triangle[1]], sites[triangle[2]],
                            sites[point]) >= 0 &&
         orient2d_sign_tess(sites[triangle[2]], sites[triangle[0]],
                            sites[point]) >= 0;
}

bool point_strictly_inside_cycle(const std::vector<Site>& sites,
                                 const SiteId point,
                                 const std::vector<SiteId>& cycle) {
  int winding_number = 0;
  for (std::size_t index = 0; index < cycle.size(); ++index) {
    const Site& first = sites[cycle[index]];
    const Site& second = sites[cycle[(index + 1) % cycle.size()]];
    if (on_segment_exact(first, second, sites[point])) {
      return false;
    }
    const int side = orient2d_sign_tess(first, second, sites[point]);
    if (first.z <= sites[point].z && second.z > sites[point].z && side > 0) {
      ++winding_number;
    } else if (second.z <= sites[point].z &&
               first.z > sites[point].z && side < 0) {
      --winding_number;
    }
  }
  return winding_number != 0;
}

EarSeedResult deterministic_ear_seed(
    const std::vector<Site>& sites,
    const std::vector<SiteId>& boundary_cycle) {
  EarSeedResult result;
  if (boundary_cycle.size() < 3) {
    return result;
  }
  std::vector<SiteId> active = boundary_cycle;
  while (active.size() > 3) {
    bool found = false;
    std::tuple<std::uint64_t, std::uint64_t, std::uint64_t> best_key;
    std::size_t best_index = 0;
    for (std::size_t index = 0; index < active.size(); ++index) {
      const SiteId previous =
          active[(index + active.size() - 1) % active.size()];
      const SiteId current = active[index];
      const SiteId next = active[(index + 1) % active.size()];
      const std::array<SiteId, 3> ear{{previous, current, next}};
      if (orient2d_sign_tess(sites[previous], sites[current], sites[next]) <=
          0) {
        continue;
      }
      bool contains_vertex = false;
      for (const SiteId candidate : active) {
        if (candidate == previous || candidate == current ||
            candidate == next) {
          continue;
        }
        if (point_in_closed_triangle(sites, candidate, ear)) {
          contains_vertex = true;
          break;
        }
      }
      if (contains_vertex) {
        continue;
      }
      const std::uint64_t previous_id = sites[previous].stable_id;
      const std::uint64_t next_id = sites[next].stable_id;
      const auto key = std::make_tuple(
          sites[current].stable_id, std::min(previous_id, next_id),
          std::max(previous_id, next_id));
      if (!found || key < best_key) {
        found = true;
        best_key = key;
        best_index = index;
      }
    }
    if (!found) {
      return result;
    }
    const SiteId previous =
        active[(best_index + active.size() - 1) % active.size()];
    const SiteId current = active[best_index];
    const SiteId next = active[(best_index + 1) % active.size()];
    result.triangles.push_back({{previous, current, next}});
    active.erase(active.begin() + static_cast<std::ptrdiff_t>(best_index));
  }
  if (orient2d_sign_tess(sites[active[0]], sites[active[1]],
                         sites[active[2]]) <= 0) {
    return result;
  }
  result.triangles.push_back({{active[0], active[1], active[2]}});
  result.ok = true;
  return result;
}

bool make_imported_patch_seed(
    const DelaunayTriangulation& moved,
    const std::vector<std::array<SiteId, 3>>& triangles,
    DelaunayTriangulation& seed) {
  seed.sites = moved.sites;
  seed.tri_edge.resize(triangles.size());
  seed.he_origin.resize(3 * triangles.size());
  seed.he_twin.assign(3 * triangles.size(), kInvalidId);
  seed.he_next.resize(3 * triangles.size());
  seed.he_face.resize(3 * triangles.size());
  std::map<EdgeKey, HalfEdgeId> unmatched;
  std::set<EdgeKey> matched;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(triangles.size()); ++triangle) {
    const HalfEdgeId base = static_cast<HalfEdgeId>(3 * triangle);
    seed.tri_edge[triangle] = base;
    for (int edge = 0; edge < 3; ++edge) {
      const HalfEdgeId half_edge = base + edge;
      seed.he_origin[half_edge] = triangles[triangle][edge];
      seed.he_next[half_edge] = base + (edge + 1) % 3;
      seed.he_face[half_edge] = triangle;
      const SiteId first = triangles[triangle][edge];
      const SiteId second = triangles[triangle][(edge + 1) % 3];
      const EdgeKey key = edge_key(first, second);
      if (matched.count(key) != 0) {
        return false;
      }
      const auto match = unmatched.find(key);
      if (match == unmatched.end()) {
        unmatched.emplace(key, half_edge);
        continue;
      }
      const HalfEdgeId twin = match->second;
      if (seed.he_twin[twin] != kInvalidId ||
          seed.he_origin[twin] != second ||
          seed.he_origin[seed.he_next[twin]] != first) {
        return false;
      }
      seed.he_twin[half_edge] = twin;
      seed.he_twin[twin] = half_edge;
      unmatched.erase(match);
      matched.insert(key);
    }
  }
  return true;
}

std::vector<SiteId> cecr_morton_order(const std::vector<Site>& sites) {
  double r_min = sites.front().r;
  double r_max = sites.front().r;
  double z_min = sites.front().z;
  double z_max = sites.front().z;
  for (std::size_t index = 1; index < sites.size(); ++index) {
    r_min = std::min(r_min, sites[index].r);
    r_max = std::max(r_max, sites[index].r);
    z_min = std::min(z_min, sites[index].z);
    z_max = std::max(z_max, sites[index].z);
  }
  std::vector<std::uint64_t> morton_keys(sites.size(), 0);
  for (std::size_t index = 0; index < sites.size(); ++index) {
    const double normalized_r =
        r_max == r_min ? 0.0 : (sites[index].r - r_min) / (r_max - r_min);
    const double normalized_z =
        z_max == z_min ? 0.0 : (sites[index].z - z_min) / (z_max - z_min);
    const std::uint32_t quantized_r = static_cast<std::uint32_t>(
        std::floor(normalized_r * 4294967295.0));
    const std::uint32_t quantized_z = static_cast<std::uint32_t>(
        std::floor(normalized_z * 4294967295.0));
    for (std::uint32_t bit = 0; bit < 32; ++bit) {
      morton_keys[index] |=
          static_cast<std::uint64_t>((quantized_r >> bit) & 1U) <<
          (2U * bit);
      morton_keys[index] |= static_cast<std::uint64_t>(
                                (quantized_z >> bit) & 1U)
                            << (2U * bit + 1U);
    }
  }
  std::vector<SiteId> order(sites.size());
  std::iota(order.begin(), order.end(), SiteId{0});
  std::sort(order.begin(), order.end(),
            [&morton_keys, &sites](const SiteId first, const SiteId second) {
              if (morton_keys[first] != morton_keys[second]) {
                return morton_keys[first] < morton_keys[second];
              }
              return sites[first].stable_id < sites[second].stable_id;
            });
  return order;
}

std::array<std::uint64_t, 3> cecr_patch_face_key(
    const std::vector<Site>& sites,
    const std::array<SiteId, 3>& triangle) {
  std::array<std::uint64_t, 3> key{{sites[triangle[0]].stable_id,
                                    sites[triangle[1]].stable_id,
                                    sites[triangle[2]].stable_id}};
  std::sort(key.begin(), key.end());
  return key;
}

WarmUpdateResult warm_static_fallback(std::vector<Site> sites,
                                      std::string failure_code,
                                      CecrTimingData* timing) {
  DelaunayResult fallback = build_delaunay(std::move(sites), timing);
  return WarmUpdateResult{std::move(fallback.dt), fallback.ok, true, false,
                          std::move(failure_code)};
}

DelaunayResult input_failure(std::vector<Site> sites,
                             const std::string& failure_code) {
  DelaunayTriangulation dt;
  dt.sites = std::move(sites);
  return DelaunayResult{std::move(dt), false, failure_code};
}

}  // namespace

DelaunayResult build_delaunay(std::vector<Site> sites) {
  PredicateCounterReporter predicate_counter_reporter;
  return build_delaunay(std::move(sites), nullptr);
}

DelaunayResult build_delaunay(std::vector<Site> sites,
                              CecrTimingData* timing) {
  TENRYU_ASSERT(
      sites.size() <=
          static_cast<std::size_t>(std::numeric_limits<SiteId>::max()),
      "Delaunay site count exceeds SiteId range");

  std::unordered_set<CoordinateKey, CoordinateKeyHash> coordinates;
  coordinates.reserve(sites.size());
  for (std::size_t index = 0; index < sites.size(); ++index) {
    const Site& site = sites[index];
    TENRYU_ASSERT(index <= std::numeric_limits<std::uint32_t>::max() &&
                      site.rank == static_cast<std::uint32_t>(index),
                  "Delaunay site ranks must be dense in ascending stable-id order");
    if (index > 0) {
      TENRYU_ASSERT(sites[index - 1].stable_id < site.stable_id,
                    "Delaunay sites must be sorted by unique stable id");
    }
    if (!std::isfinite(site.r) || !std::isfinite(site.z)) {
      return input_failure(std::move(sites), "nan_coordinate");
    }
    if (site.r < 0.0) {
      return input_failure(std::move(sites), "negative_r");
    }
    if (site.r == 0.0 && std::signbit(site.r)) {
      return input_failure(std::move(sites), "minus_zero_unnormalized");
    }
    if (!coordinates.insert(coordinate_key(site)).second) {
      return input_failure(std::move(sites), "duplicate_generator");
    }
  }

  if (sites.size() < 3) {
    return input_failure(std::move(sites), "all_sites_collinear");
  }

  double r_min = sites[0].r;
  double r_max = sites[0].r;
  double z_min = sites[0].z;
  double z_max = sites[0].z;
  for (std::size_t index = 1; index < sites.size(); ++index) {
    r_min = std::min(r_min, sites[index].r);
    r_max = std::max(r_max, sites[index].r);
    z_min = std::min(z_min, sites[index].z);
    z_max = std::max(z_max, sites[index].z);
  }

  std::vector<std::uint64_t> morton_keys(sites.size(), 0);
  for (std::size_t index = 0; index < sites.size(); ++index) {
    const double normalized_r =
        r_max == r_min ? 0.0 : (sites[index].r - r_min) / (r_max - r_min);
    const double normalized_z =
        z_max == z_min ? 0.0 : (sites[index].z - z_min) / (z_max - z_min);
    const std::uint32_t quantized_r = static_cast<std::uint32_t>(
        std::floor(normalized_r * 4294967295.0));
    const std::uint32_t quantized_z = static_cast<std::uint32_t>(
        std::floor(normalized_z * 4294967295.0));
    for (std::uint32_t bit = 0; bit < 32; ++bit) {
      morton_keys[index] |=
          static_cast<std::uint64_t>((quantized_r >> bit) & 1U) << (2U * bit);
      morton_keys[index] |= static_cast<std::uint64_t>(
                                (quantized_z >> bit) & 1U)
                            << (2U * bit + 1U);
    }
  }

  std::vector<SiteId> insertion_sequence(sites.size());
  std::iota(insertion_sequence.begin(), insertion_sequence.end(), SiteId{0});
  std::sort(insertion_sequence.begin(), insertion_sequence.end(),
            [&morton_keys, &sites](const SiteId lhs, const SiteId rhs) {
              if (morton_keys[lhs] != morton_keys[rhs]) {
                return morton_keys[lhs] < morton_keys[rhs];
              }
              return sites[lhs].stable_id < sites[rhs].stable_id;
            });
  const char* const insertion_override =
      std::getenv("TENRYU_TESS_INSERTION");
  if (insertion_override != nullptr &&
      std::string(insertion_override) == std::string("ascending")) {
    std::iota(insertion_sequence.begin(), insertion_sequence.end(), SiteId{0});
  }

  std::size_t third_position = sites.size();
  for (std::size_t candidate_position = 2;
       candidate_position < insertion_sequence.size(); ++candidate_position) {
    if (orient2d_sign_tess(sites[insertion_sequence[0]],
                           sites[insertion_sequence[1]],
                           sites[insertion_sequence[candidate_position]]) != 0) {
      third_position = candidate_position;
      break;
    }
  }
  if (third_position == sites.size()) {
    return input_failure(std::move(sites), "all_sites_collinear");
  }

  DelaunayBuilder builder(sites);
  builder.initialize(insertion_sequence[0], insertion_sequence[1],
                     insertion_sequence[third_position]);
  for (std::size_t position = 2; position < third_position; ++position) {
    builder.insert(insertion_sequence[position]);
  }
  for (std::size_t position = third_position + 1;
       position < insertion_sequence.size(); ++position) {
    builder.insert(insertion_sequence[position]);
  }

  DelaunayTriangulation dt = builder.finalize();
  DelaunayAuditResult audit;
  {
    CecrStageTimer audit_timer(
        timing != nullptr ? &timing->audit_ms : nullptr);
    audit = audit_delaunay(dt);
  }
  if (!audit.ok) {
    return DelaunayResult{std::move(dt), false, audit.failure_code};
  }
  return DelaunayResult{std::move(dt), true, {}};
}

WarmUpdateResult warm_update_delaunay(
    const DelaunayTriangulation& previous,
    std::vector<Site> new_sites) {
  PredicateCounterReporter predicate_counter_reporter;
  CecrTimingReporter timing_reporter;
  CecrTimingData* const timing = timing_reporter.timing();
  TENRYU_ASSERT(previous.sites.size() == new_sites.size(),
                "warm Delaunay stable-id sequences must have equal length");
  for (std::size_t index = 0; index < new_sites.size(); ++index) {
    TENRYU_ASSERT(index <= std::numeric_limits<std::uint32_t>::max() &&
                      previous.sites[index].rank ==
                          static_cast<std::uint32_t>(index) &&
                      new_sites[index].rank ==
                          static_cast<std::uint32_t>(index),
                  "warm Delaunay site ranks must be dense");
    TENRYU_ASSERT(previous.sites[index].stable_id ==
                      new_sites[index].stable_id,
                  "warm Delaunay stable-id sequences must match exactly");
    if (index > 0) {
      TENRYU_ASSERT(previous.sites[index - 1].stable_id <
                        previous.sites[index].stable_id,
                    "warm Delaunay stable ids must be ascending");
    }
  }

  DelaunayTriangulation working = previous;
  for (std::size_t index = 0; index < new_sites.size(); ++index) {
    working.sites[index].r = new_sites[index].r;
    working.sites[index].z = new_sites[index].z;
  }

  CecrHullChange hull_change;
  {
    CecrStageTimer hull_timer(
        timing != nullptr ? &timing->hull_ms : nullptr);
    hull_change = cecr_hull_change(previous, working);
  }

  bool has_orientation_defect = false;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(working.tri_edge.size());
       ++triangle) {
    const std::array<SiteId, 3> vertices =
        triangle_vertices(working, triangle);
    if (orient2d_sign_tess(working.sites[vertices[0]],
                           working.sites[vertices[1]],
                           working.sites[vertices[2]]) <= 0) {
      has_orientation_defect = true;
      break;
    }
  }

  if (has_orientation_defect || !hull_change.cycles_equal) {
    CecrSeedResult seed;
    {
      CecrStageTimer seed_timer(
          timing != nullptr ? &timing->seed_ms : nullptr);
      seed = cecr_seed_defects(previous, working, hull_change);
    }
    if (has_orientation_defect && !seed.any_defect) {
      return warm_static_fallback(std::move(new_sites),
                                  "warm_precheck_orientation", timing);
    }
    const CecrComponents components =
        cecr_form_components(working, seed.in_patch);
    const CecrClosureResult closure =
        cecr_close_disks(working, components, seed.in_patch);
    if (!closure.ok) {
      return warm_static_fallback(std::move(new_sites),
                                  closure.reason.c_str(), timing);
    }
    const CecrCavitySet cavities =
        cecr_certify_and_grow(previous, working, hull_change,
                              closure.in_patch, timing);
    if (!cavities.ok) {
      return warm_static_fallback(std::move(new_sites),
                                  cavities.reason.c_str(), timing);
    }

    std::vector<CecrPatch> patches;
    patches.reserve(cavities.cavities.size());
    for (const CecrCavity& cavity : cavities.cavities) {
      CecrPatch patch;
      {
        CecrStageTimer retriangulate_timer(
            timing != nullptr ? &timing->retriangulate_ms : nullptr);
        patch = cecr_retriangulate(working, cavity);
      }
      if (!patch.ok) {
        return warm_static_fallback(std::move(new_sites),
                                    patch.reason.c_str(), timing);
      }
      if (timing != nullptr) {
        timing->patch_faces += static_cast<int>(patch.triangles.size());
      }
      patches.push_back(std::move(patch));
    }

    DelaunayBuilder builder(working.sites, working, BuilderMode::WARM, false,
                            timing);
    if (!builder.ok()) {
      return warm_static_fallback(
          std::move(new_sites),
          std::string("warm_internal_") + builder.failure_code(), timing);
    }
    DelaunayTriangulation legalized;
    {
      CecrStageTimer splice_timer(
          timing != nullptr ? &timing->splice_ms : nullptr);
      {
        CecrStageTimer replace_timer(
            timing != nullptr ? &timing->replace_ms : nullptr);
        for (std::size_t index = 0; index < cavities.cavities.size(); ++index) {
          builder.splice_cecr_patch(cavities.cavities[index], patches[index]);
          if (!builder.ok()) {
            replace_timer.stop();
            splice_timer.stop();
            return warm_static_fallback(
                std::move(new_sites),
                std::string("warm_internal_") + builder.failure_code(),
                timing);
          }
        }
      }
      {
        CecrStageTimer legalize_timer(
            timing != nullptr ? &timing->legalize_ms : nullptr);
        builder.legalize_all();
        if (!builder.ok()) {
          legalize_timer.stop();
          splice_timer.stop();
          return warm_static_fallback(
              std::move(new_sites),
              std::string("warm_internal_") + builder.failure_code(),
              timing);
        }
      }
      {
        CecrStageTimer finalize_timer(
            timing != nullptr ? &timing->finalize_ms : nullptr);
        legalized = builder.finalize();
        if (!builder.ok()) {
          finalize_timer.stop();
          splice_timer.stop();
          return warm_static_fallback(
              std::move(new_sites),
              std::string("warm_internal_") + builder.failure_code(),
              timing);
        }
      }
    }

    const std::vector<SiteId>& exact_hull = hull_change.h_new;
    const std::vector<SiteId> current_hull =
        triangulation_hull_cycle(legalized);
    if (!cyclic_sequences_equal(exact_hull, current_hull)) {
      return warm_static_fallback(std::move(new_sites),
                                  "warm_precheck_hull", timing);
    }
    if (!exact_area_identity(legalized, exact_hull)) {
      return warm_static_fallback(std::move(new_sites),
                                  "warm_precheck_area", timing);
    }
    DelaunayAuditResult audit;
    {
      CecrStageTimer audit_timer(
          timing != nullptr ? &timing->audit_ms : nullptr);
      audit = audit_delaunay(legalized);
    }
    if (!audit.ok) {
      return warm_static_fallback(std::move(new_sites),
                                  "warm_audit_failed", timing);
    }
    return WarmUpdateResult{std::move(legalized), true, false, true, {}};
  }

  const std::vector<SiteId>& exact_hull = hull_change.h_new;
  const std::vector<SiteId> current_hull =
      triangulation_hull_cycle(working);
  if (!cyclic_sequences_equal(exact_hull, current_hull)) {
    return warm_static_fallback(std::move(new_sites),
                                "warm_precheck_hull", timing);
  }
  if (!exact_area_identity(working, exact_hull)) {
    return warm_static_fallback(std::move(new_sites),
                                "warm_precheck_area", timing);
  }

  DelaunayBuilder builder(working.sites, working, BuilderMode::WARM, false,
                          timing);
  if (!builder.ok()) {
    return warm_static_fallback(
        std::move(new_sites),
        std::string("warm_internal_") + builder.failure_code(), timing);
  }
  DelaunayTriangulation legalized;
  {
    CecrStageTimer splice_timer(
        timing != nullptr ? &timing->splice_ms : nullptr);
    {
      CecrStageTimer legalize_timer(
          timing != nullptr ? &timing->legalize_ms : nullptr);
      builder.legalize_all();
      if (!builder.ok()) {
        legalize_timer.stop();
        splice_timer.stop();
        return warm_static_fallback(
            std::move(new_sites),
            std::string("warm_internal_") + builder.failure_code(), timing);
      }
    }
    {
      CecrStageTimer finalize_timer(
          timing != nullptr ? &timing->finalize_ms : nullptr);
      legalized = builder.finalize();
      if (!builder.ok()) {
        finalize_timer.stop();
        splice_timer.stop();
        return warm_static_fallback(
            std::move(new_sites),
            std::string("warm_internal_") + builder.failure_code(), timing);
      }
    }
  }
  DelaunayAuditResult audit;
  {
    CecrStageTimer audit_timer(
        timing != nullptr ? &timing->audit_ms : nullptr);
    audit = audit_delaunay(legalized);
  }
  if (!audit.ok) {
    return warm_static_fallback(std::move(new_sites),
                                "warm_audit_failed", timing);
  }
  return WarmUpdateResult{std::move(legalized), true, false, false, {}};
}

CecrPatch cecr_retriangulate(const DelaunayTriangulation& moved,
                             const CecrCavity& cavity) {
  const auto failure = [](const char* reason) {
    CecrPatch result;
    result.reason = reason;
    return result;
  };
  if (!cavity.ok || cavity.boundary_cycle.size() < 3) {
    return failure("cecr_e1_ear_seed");
  }
  std::set<SiteId> boundary_sites;
  for (const SiteId site : cavity.boundary_cycle) {
    if (site < 0 || site >= static_cast<SiteId>(moved.sites.size()) ||
        !boundary_sites.insert(site).second) {
      return failure("cecr_e1_ear_seed");
    }
  }

  EarSeedResult ear_seed =
      deterministic_ear_seed(moved.sites, cavity.boundary_cycle);
  if (!ear_seed.ok) {
    return failure("cecr_e1_ear_seed");
  }

  std::set<SiteId> interior_sites;
  for (const SiteId site : cavity.interior_sites) {
    if (site < 0 || site >= static_cast<SiteId>(moved.sites.size()) ||
        boundary_sites.count(site) != 0 ||
        !interior_sites.insert(site).second ||
        !point_strictly_inside_cycle(moved.sites, site,
                                     cavity.boundary_cycle)) {
      return failure("cecr_e2_insert");
    }
  }
  DelaunayTriangulation seed;
  if (!make_imported_patch_seed(moved, ear_seed.triangles, seed) ||
      !cyclic_sequences_equal(triangulation_hull_cycle(seed),
                              cavity.boundary_cycle)) {
    return failure("cecr_e2_insert");
  }

  DelaunayBuilder builder(moved.sites, seed, BuilderMode::WARM, true);
  if (!builder.ok()) {
    return failure("cecr_e2_insert");
  }
  builder.legalize_all();
  if (!builder.ok()) {
    return failure("cecr_e2_insert");
  }
  const std::vector<SiteId> morton_order = cecr_morton_order(moved.sites);
  for (const SiteId site : morton_order) {
    if (interior_sites.count(site) != 0) {
      builder.insert(site);
      if (!builder.ok()) {
        return failure("cecr_e2_insert");
      }
    }
  }
  const DelaunayTriangulation local = builder.finalize();
  if (!builder.ok()) {
    return failure("cecr_e3_extract");
  }

  const std::vector<SiteId> local_hull =
      triangulation_hull_cycle(local);
  if (!cyclic_sequences_equal(local_hull, cavity.boundary_cycle)) {
    return failure("cecr_e3_extract");
  }
  // Interior cavities have no span edges, reducing exactly to the Stage-1
  // extraction checks above and below.
  std::set<EdgeKey> patch_hull_edges;
  for (HalfEdgeId half_edge = 0;
       half_edge < static_cast<HalfEdgeId>(local.he_twin.size());
       ++half_edge) {
    if (local.he_twin[half_edge] == kInvalidId) {
      patch_hull_edges.insert(edge_key(
          local.he_origin[half_edge],
          local.he_origin[local.he_next[half_edge]]));
    }
  }
  for (const auto& span_edge : cavity.span_edges) {
    if (patch_hull_edges.count(
            edge_key(span_edge.first, span_edge.second)) == 0) {
      return failure("cecr_e3_extract");
    }
  }
  std::set<SiteId> allowed = boundary_sites;
  allowed.insert(interior_sites.begin(), interior_sites.end());
  std::set<SiteId> represented;
  std::vector<std::array<SiteId, 3>> triangles;
  triangles.reserve(local.tri_edge.size());
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(local.tri_edge.size());
       ++triangle) {
    const std::array<SiteId, 3> vertices =
        triangle_vertices(local, triangle);
    if (orient2d_sign_tess(local.sites[vertices[0]],
                           local.sites[vertices[1]],
                           local.sites[vertices[2]]) <= 0) {
      return failure("cecr_e3_extract");
    }
    for (const SiteId site : vertices) {
      if (allowed.count(site) == 0) {
        return failure("cecr_e3_extract");
      }
      represented.insert(site);
    }
    triangles.push_back(vertices);
  }
  if (represented != allowed) {
    return failure("cecr_e3_extract");
  }
  std::sort(triangles.begin(), triangles.end(),
            [&moved](const std::array<SiteId, 3>& first,
                     const std::array<SiteId, 3>& second) {
              return cecr_patch_face_key(moved.sites, first) <
                     cecr_patch_face_key(moved.sites, second);
            });

  Expansion triangle_area;
  for (const std::array<SiteId, 3>& triangle : triangles) {
    triangle_area = expansion_sum(
        triangle_area,
        signed_twice_area(moved.sites[triangle[0]],
                          moved.sites[triangle[1]],
                          moved.sites[triangle[2]]));
  }
  Expansion boundary_area;
  for (std::size_t index = 1;
       index + 1 < cavity.boundary_cycle.size(); ++index) {
    boundary_area = expansion_sum(
        boundary_area,
        signed_twice_area(moved.sites[cavity.boundary_cycle[0]],
                          moved.sites[cavity.boundary_cycle[index]],
                          moved.sites[cavity.boundary_cycle[index + 1]]));
  }
  if (expansion_sign(
          expansion_difference(triangle_area, boundary_area)) != 0) {
    return failure("cecr_e3_extract");
  }
  return CecrPatch{std::move(triangles), true, {}};
}

}  // namespace tenryu::mesh::tess
