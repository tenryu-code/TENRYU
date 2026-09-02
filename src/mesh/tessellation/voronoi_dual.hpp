#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "mesh/tessellation/delaunay_2d.hpp"
#include "mesh/voronoi_rz.cuh"

namespace tenryu::mesh::tess {

using VoronoiClassId = std::int32_t;
using VoronoiNodeId = std::int32_t;
using DualFeatureId = std::int32_t;
using RestrictedNodeId = std::int32_t;
using RestrictedHalfEdgeId = std::int32_t;
using DomainSegmentId = std::int32_t;
using DomainVertexId = std::int32_t;

struct BoundaryParam {
  DomainSegmentId edge = kInvalidId;
  ExactRational lambda;
  bool valid = false;
};

struct VoronoiTripleKey {
  std::array<std::uint64_t, 3> stable_ids{};

  bool operator<(const VoronoiTripleKey& other) const {
    return stable_ids < other.stable_ids;
  }

  bool operator==(const VoronoiTripleKey& other) const {
    return stable_ids == other.stable_ids;
  }
};

struct VoronoiVertexClass {
  VoronoiTripleKey key;
  std::array<SiteId, 3> representative_sites{};
  std::vector<TriangleId> triangles;
  HomogeneousPoint circumcenter;
  std::array<double, 2> constructed{};
};

struct VoronoiVertexClasses {
  std::vector<VoronoiVertexClass> classes;
  std::vector<VoronoiClassId> triangle_class;
};

struct ExactDirectionWitness {
  Expansion r;
  Expansion z;
};

enum class DualFeatureKind : std::uint8_t {
  kBoundedSegment,
  kUnboundedRay,
};

enum class DualEdgeDisposition : std::uint8_t {
  kCocircularContracted,
  kMachineIdentityContracted,
  kBoundedFeature,
  kRayFeature,
};

struct VoronoiDualNode {
  VoronoiClassId representative_class = kInvalidId;
  std::vector<VoronoiClassId> classes;
  std::array<double, 2> constructed{};
};

struct VoronoiDualFeature {
  DualFeatureKind kind = DualFeatureKind::kBoundedSegment;
  HalfEdgeId delaunay_half_edge = kInvalidId;
  std::array<SiteId, 2> primal_sites{};
  VoronoiClassId first_class = kInvalidId;
  VoronoiClassId second_class = kInvalidId;
  VoronoiNodeId first_node = kInvalidId;
  VoronoiNodeId second_node = kInvalidId;
  ExactDirectionWitness ray_direction;
};

struct DualFeatureRegistryEntry {
  HalfEdgeId delaunay_half_edge = kInvalidId;
  std::array<SiteId, 2> primal_sites{};
  VoronoiClassId first_class = kInvalidId;
  VoronoiClassId second_class = kInvalidId;
  DualEdgeDisposition disposition =
      DualEdgeDisposition::kCocircularContracted;
  DualFeatureId feature = kInvalidId;
};

struct VoronoiDual {
  std::vector<VoronoiVertexClass> classes;
  std::vector<VoronoiClassId> triangle_class;
  std::vector<VoronoiDualNode> nodes;
  std::vector<VoronoiNodeId> class_node;
  std::vector<DualFeatureRegistryEntry> registry;
  std::vector<VoronoiDualFeature> features;
};

struct DomainPoint {
  double r = 0.0;
  double z = 0.0;
};

enum class CanonicalNodeKeyKind : std::uint8_t {
  kDomainVertex,
  kVoronoiTriple,
  kBoundaryCrossing,
};

struct CanonicalNodeKey {
  CanonicalNodeKeyKind kind = CanonicalNodeKeyKind::kBoundaryCrossing;
  DomainVertexId domain_vertex = kInvalidId;
  VoronoiTripleKey voronoi_triple;
  DomainSegmentId domain_segment = kInvalidId;
  std::array<std::uint64_t, 2> crossing_sites{};

  bool operator<(const CanonicalNodeKey& other) const;
  bool operator==(const CanonicalNodeKey& other) const;
};

struct RestrictedExactWitness {
  CanonicalNodeKey key;
  HomogeneousPoint point;
};

struct RestrictedNode {
  CanonicalNodeKey key;
  std::vector<CanonicalNodeKey> aliases;
  HomogeneousPoint exact_point;
  std::vector<RestrictedExactWitness> exact_witnesses;
  std::array<double, 2> constructed{};
  BoundaryParam boundary_param;
  double lambda_cache = 0.0;
  std::vector<DomainSegmentId> boundary_segments;
  std::vector<DomainVertexId> domain_vertices;
};

struct RestrictedDualEdge {
  std::array<SiteId, 2> sites{};
  RestrictedNodeId endpoint0 = kInvalidId;
  RestrictedNodeId endpoint1 = kInvalidId;
};

struct DomainSegmentRecord {
  DomainSegmentId id = kInvalidId;
  RestrictedNodeId start_node = kInvalidId;
  RestrictedNodeId end_node = kInvalidId;
  DomainVertexId start_vertex = kInvalidId;
  DomainVertexId end_vertex = kInvalidId;
  bool axis = false;
};

struct BoundaryInterval {
  DomainSegmentId segment = kInvalidId;
  RestrictedNodeId endpoint0 = kInvalidId;
  RestrictedNodeId endpoint1 = kInvalidId;
  SiteId owner = kInvalidId;
};

struct RestrictedGeometry {
  std::vector<RestrictedNode> nodes;
  std::vector<RestrictedDualEdge> interior_edges;
  std::vector<DomainSegmentRecord> domain_segments;
  std::vector<BoundaryInterval> boundary_intervals;
  std::vector<std::array<RestrictedNodeId, 2>> identity_contractions;
  bool carrier_representability_ok = true;
  bool carrier_coverage_ok = true;
};

struct RestrictedHalfEdge {
  RestrictedNodeId origin = kInvalidId;
  RestrictedHalfEdgeId twin = kInvalidId;
  RestrictedHalfEdgeId next = kInvalidId;
  SiteId left_face = kInvalidId;
  DomainSegmentId boundary_segment = kInvalidId;
};

struct RestrictedCell {
  SiteId site = kInvalidId;
  RestrictedHalfEdgeId first_edge = kInvalidId;
  std::vector<RestrictedNodeId> cycle;
};

struct RestrictedDcel {
  std::vector<RestrictedNode> nodes;
  std::vector<RestrictedHalfEdge> half_edges;
  std::vector<RestrictedCell> cells;
  SiteId exterior_face = kInvalidId;
};

struct WeldOutcome {
  bool ok = true;              // false rejects the rezone
  bool applied = false;        // at least one contraction committed
  std::string failure_code;    // weld_master_master or weld_forbidden_class_pair
  int passes = 0;              // candidate-bearing weld passes executed
  int contracted_edges = 0;    // committed pairwise unions
  int clusters = 0;            // compacted multi-node components
  int max_cluster_size = 0;
  double max_displacement = 0.0;  // snapped member-to-representative distance
  // Fail-loud-class (forbidden / master-master) pairs retained because their
  // chi lies in the extended band (1,4]: not proven noise, so protection wins
  // over the loud invariant. Final-pass census of source-edge instances.
  std::size_t extended_band_forbidden = 0;
};

struct WeldScales {
  // Exact expansions frozen from raw DCEL cell cycles.
  std::vector<Expansion> area;
  std::vector<Expansion> perimeter;  // L1 perimeter
};

// Relative geometric short-edge collapse band for the restricted-Voronoi
// weld stage. Values <= 0 disable the geometric band.
void set_reale_short_edge_collapse_rel(double collapse_rel);
double reale_short_edge_collapse_rel();

struct RestrictedVoronoiResult {
  DelaunayTriangulation dt;
  RestrictedDcel dcel;
  bool ok = false;
  bool used_static_fallback = false;
  bool used_cecr = false;
  std::string failure_code;
  WeldOutcome weld;
  bool carrier_representability_ok = true;
  bool carrier_coverage_ok = true;
};

// T2-GPU gate (default OFF). Single-host-thread use, like the
// tessellation itself. When ON, build_voronoi_vertex_classes computes
// per-triangle circumcenter witnesses on the GPU (byte-identical term
// sequences; per-triangle fallback recomputes on host).
void set_tess_gpu_dual_enabled(bool enabled);
bool tess_gpu_dual_enabled();
// Test hook: force every class onto the host fallback path.
void set_tess_gpu_forced_fallback_for_tests(bool enabled);
// Classes recomputed on host during the last gate-ON
// build_voronoi_vertex_classes call (untouched when the gate is OFF).
std::size_t tess_gpu_last_fallback_count();
// T2-b gate (default OFF): GPU certified classify inside
// restrict_voronoi_dual. Effective only when the dual gate is also ON
// (the classify kernel consumes the dual batch's device buffers);
// namelist validation enforces the pairing.
void set_tess_gpu_restrict_enabled(bool enabled);
bool tess_gpu_restrict_enabled();
// Introspection used by the restrict stage and tests.
bool tess_gpu_forced_fallback_for_tests();
// Host-shortlisted nodes during the last gate-ON classify (untouched
// when the restrict gate is OFF).
std::size_t tess_gpu_last_classify_shortlist();

VoronoiVertexClasses build_voronoi_vertex_classes(
    const DelaunayTriangulation& dt);

VoronoiDual build_voronoi_dual(const DelaunayTriangulation& dt);

RestrictedGeometry restrict_voronoi_dual(
    const DelaunayTriangulation& dt,
    const VoronoiDual& dual,
    std::vector<DomainPoint> domain);

void build_boundary_arrangement(const DelaunayTriangulation& dt,
                                RestrictedGeometry& geometry);

void rebuild_boundary_intervals(const DelaunayTriangulation& dt,
                                RestrictedGeometry& geometry);

struct AxisAuditCounters {
  std::size_t nodes_total = 0;
  std::size_t axis_support_nodes = 0;
  std::size_t axis_segments = 0;
  std::size_t axis_intervals = 0;
};

RestrictedDcel assemble_restricted_dcel(
    const DelaunayTriangulation& dt,
    RestrictedGeometry geometry,
    AxisAuditCounters* axis_counters = nullptr);

WeldScales compute_weld_scales(const RestrictedDcel& raw_dcel,
                               std::size_t n_sites);

// Borrowed-skeleton variant: the raw DCEL carries empty nodes; node
// records are read as records[source[id]]. Enumeration order and
// arithmetic are identical to compute_weld_scales.
WeldScales compute_weld_scales_borrowed(
    const RestrictedDcel& raw_dcel,
    const std::vector<RestrictedNode>& records,
    const std::vector<RestrictedNodeId>& source,
    std::size_t n_sites);

WeldOutcome weld_degenerate_edges(const DelaunayTriangulation& dt,
                                  RestrictedGeometry& geometry,
                                  const WeldScales& scales);

bool audit_weld_subfloor(const RestrictedDcel& final_dcel,
                         const WeldScales& scales,
                         std::string* failure_code);

RestrictedVoronoiResult build_restricted_voronoi(
    std::vector<Site> sites,
    std::vector<DomainPoint> domain);

RestrictedVoronoiResult build_restricted_voronoi_warm(
    const DelaunayTriangulation& previous,
    std::vector<Site> sites,
    std::vector<DomainPoint> domain);

std::vector<std::uint8_t> canonical_serialize(
    const DelaunayTriangulation& dt,
    const RestrictedDcel& dcel);

mesh::voronoi::Tessellation to_legacy_tessellation(
    const DelaunayTriangulation& dt,
    const RestrictedDcel& dcel);

struct TessellationComparison {
  bool topology_equal = false;
  std::string first_divergence;
  double max_coord_delta = 0.0;
  double max_volume_delta = 0.0;
  int max_valence = 0;
  std::size_t nv_over_8 = 0;
  std::size_t nodes_old = 0;
  std::size_t nodes_new = 0;
};

TessellationComparison compare_tessellations(
    const mesh::voronoi::Tessellation& legacy,
    const mesh::voronoi::Tessellation& shadow);

}  // namespace tenryu::mesh::tess
