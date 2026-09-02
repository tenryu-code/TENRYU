#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tenryu::mesh::voronoi {

struct Generator {
  double r = 0.0;
  double z = 0.0;
  std::uint64_t id = 0;  // immutable global id -- determinism authority
};

struct DomainBoundary {
  // Closed convex or simply-connected polygonal domain boundary in (r,z),
  // counter-clockwise, r >= 0 everywhere; segments lying exactly on r == 0
  // are the axis. v1 supports a single boundary loop.
  std::vector<double> r;
  std::vector<double> z;
};

struct VoronoiCell {
  std::uint64_t generator_id = 0;
  std::vector<int> node_indices;  // polygon node ids, CCW
  std::vector<double> r;  // polygon vertices, CCW
  std::vector<double> z;
  std::vector<std::uint64_t> neighbor_ids;  // per-edge generator or boundary
  double volume_rz = 0.0;  // pi/3-exact RZ volume from poly_geom moments
};
inline constexpr std::uint64_t kBoundaryNeighbor = ~std::uint64_t{0};

enum class TessNodeClass : std::uint8_t {
  kInterior = 0,
  kCarrierMaster = 1,
  kCarrierSlave = 2,
};

struct TessNodeBoundary {
  TessNodeClass cls = TessNodeClass::kInterior;
  int domain_vertex = -1;  // valid for kCarrierMaster
  int carrier_edge = -1;  // valid for kCarrierSlave
  double lambda_cache = 0.0;  // valid for kCarrierSlave
};

struct Tessellation {
  std::vector<double> node_r;  // canonical shared-node coordinates
  std::vector<double> node_z;
  // Parallel to node_r/node_z for the exact core; empty for the legacy builder.
  std::vector<TessNodeBoundary> node_boundary;
  std::vector<VoronoiCell> cells;  // sorted by generator_id ascending
  std::size_t collapsed_edges = 0;  // accepted valence-reduction contractions
  int weld_contracted = 0;
  int weld_clusters = 0;
  double weld_max_displacement = 0.0;
  bool carrier_representability_ok = true;
  bool carrier_coverage_ok = true;
  bool valid = false;
  std::string reject_reason;  // empty when valid
};

// Deterministic constrained Voronoi tessellation accelerated by a uniform
// spatial hash. Over-valent cells are reduced to at most eight vertices by
// globally contracting admissible shortest edges to their exact midpoints.
Tessellation build_constrained_voronoi(
    const std::vector<Generator>& generators,
    const DomainBoundary& domain);

namespace voronoi_detail {

// Test-only hook for byte-comparison against the all-pairs reference path.
void set_slow_path_for_tests(bool enabled);

}  // namespace voronoi_detail

}  // namespace tenryu::mesh::voronoi
