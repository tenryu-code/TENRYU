#pragma once

#include <cstdint>
#include <vector>

#include "mesh/reale_contracts.hpp"

namespace tenryu::mesh::reale {

struct ConnectivityView {
  const int* cell_node_offsets;
  const int* cell_node_indices;
  const std::uint8_t* cell_nverts;
  const int* cell_orientation_sign;
  int n_cells;
  int n_nodes;
  const std::uint8_t* hydro_active;    // Nullable.
  const std::uint8_t* cell_protected;  // Nullable; 1 excludes reconnection.
};

struct GeometryView {
  const double* node_r;
  const double* node_z;
};

struct ProtectedTopologyView {
  const std::uint8_t* cell_protected;  // Nullable; 1 excludes reconnection.
};

struct G1Witness {
  int cell;
  int corner;
};

struct Cavity {
  std::vector<EntityId> cells;
  std::vector<EntityId> boundary_nodes;
  int rings;
  ConnectivityView connectivity;
};

// Owns every span backing store used by its TopologyTransaction.
struct Candidate {
  ReconnectOp op{ReconnectOp::FaceFlip};
  std::vector<EntityId> donors;
  std::vector<EntityId> donor_nodes;

  std::vector<EntityId> fixed_boundary_loop_nodes;
  std::vector<CavityBoundaryLoop> fixed_boundary;

  std::vector<EntityId> target_cell_ids;
  std::vector<EntityId> target_node_ids;
  std::vector<std::int32_t> cell_node_offsets;
  std::vector<EntityId> cell_node_ids;

  std::vector<EntityId> created_cells;
  std::vector<EntityId> deleted_cells;
  std::vector<EntityId> created_nodes;
  std::vector<EntityId> deleted_nodes;

  Candidate() = default;
  Candidate(const Candidate& other);
  Candidate& operator=(const Candidate& other);
  Candidate(Candidate&& other) noexcept;
  Candidate& operator=(Candidate&& other) noexcept;

  void refresh_fixed_boundary_view();
};

struct CandidateSet {
  std::vector<Candidate> ordered;
};

struct MeshView {
  EntityId event_id;
};

void expand_cavity(Cavity& cavity, const ConnectivityView& connectivity);

}  // namespace tenryu::mesh::reale
