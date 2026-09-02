#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mesh/tessellation/exact_types.hpp"

namespace tenryu::mesh::tess {

using SiteId = std::int32_t;
using HalfEdgeId = std::int32_t;
using TriangleId = std::int32_t;

inline constexpr std::int32_t kInvalidId = -1;

struct Location {
  TriangleId tri = kInvalidId;
  HalfEdgeId on_edge = kInvalidId;
};

struct DelaunayTriangulation {
  std::vector<Site> sites;
  std::vector<SiteId> he_origin;
  std::vector<HalfEdgeId> he_twin;
  std::vector<HalfEdgeId> he_next;
  std::vector<TriangleId> he_face;
  std::vector<HalfEdgeId> tri_edge;
};

struct DelaunayResult {
  DelaunayTriangulation dt;
  bool ok = false;
  std::string failure_code;
};

struct WarmUpdateResult {
  DelaunayTriangulation dt;
  bool ok = false;
  bool used_static_fallback = false;
  bool used_cecr = false;
  std::string failure_code;
};

struct DelaunayAuditResult {
  bool ok = false;
  std::string failure_code;
};

struct CecrSeedResult {
  std::vector<TriangleId> defect_triangles;
  std::vector<std::uint8_t> in_patch;
  bool any_defect = false;
};

struct CecrHullChange {
  std::vector<SiteId> h_old;
  std::vector<SiteId> h_new;
  std::vector<SiteId> entrants;
  std::vector<SiteId> leavers;
  bool cycles_equal = false;
};

struct CecrComponents {
  std::vector<int> component_of_triangle;
  int component_count = 0;
};

struct CecrClosureResult {
  std::vector<std::uint8_t> in_patch;
  bool ok = false;
  std::string reason;
};

struct CecrCavity {
  std::vector<SiteId> boundary_cycle;
  // Empty for interior cavities, preserving the Stage-1 splice path.
  std::vector<std::pair<SiteId, SiteId>> old_arc_edges;
  std::vector<std::pair<SiteId, SiteId>> span_edges;
  std::vector<SiteId> interior_sites;
  std::vector<std::uint8_t> in_patch;
  bool ok = false;
  std::string reason;
};

struct CecrCavitySet {
  std::vector<CecrCavity> cavities;
  bool ok = false;
  std::string reason;
};

struct CecrPatch {
  std::vector<std::array<SiteId, 3>> triangles;
  bool ok = false;
  std::string reason;
};

DelaunayResult build_delaunay(std::vector<Site> sites);

WarmUpdateResult warm_update_delaunay(const DelaunayTriangulation& previous,
                                      std::vector<Site> new_sites);

DelaunayAuditResult audit_delaunay(const DelaunayTriangulation& dt);

std::vector<SiteId> exact_convex_hull(const std::vector<Site>& sites);

CecrHullChange cecr_hull_change(const DelaunayTriangulation& previous,
                                const DelaunayTriangulation& moved);

CecrSeedResult cecr_seed_defects(const DelaunayTriangulation& previous,
                                 const DelaunayTriangulation& moved);

CecrSeedResult cecr_seed_defects(const DelaunayTriangulation& previous,
                                 const DelaunayTriangulation& moved,
                                 const CecrHullChange& hull_change);

CecrComponents cecr_form_components(
    const DelaunayTriangulation& moved,
    const std::vector<std::uint8_t>& in_patch);

CecrClosureResult cecr_close_disks(
    const DelaunayTriangulation& moved,
    CecrComponents components,
    std::vector<std::uint8_t> in_patch);

CecrCavitySet cecr_certify_and_grow(
    const DelaunayTriangulation& previous,
    const DelaunayTriangulation& moved,
    std::vector<std::uint8_t> in_patch);

CecrCavitySet cecr_certify_and_grow(
    const DelaunayTriangulation& previous,
    const DelaunayTriangulation& moved,
    const CecrHullChange& hull_change,
    std::vector<std::uint8_t> in_patch);

CecrPatch cecr_retriangulate(const DelaunayTriangulation& moved,
                             const CecrCavity& cavity);

}  // namespace tenryu::mesh::tess
