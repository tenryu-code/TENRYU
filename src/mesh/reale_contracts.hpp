#pragma once

#include <cstdint>
#include <span>

namespace tenryu::core {

struct State;

}  // namespace tenryu::core

namespace tenryu::mesh::reale {

/// Stable identifier for a ReALE mesh entity.
using EntityId = std::uint64_t;

/// Signed index type for ReALE storage extents and offsets.
using StorageIndex = std::int32_t;

/// Pi/3-exact RZ-authority volume and first moments in cm^3, cm^4, and cm^4.
struct RZMoments {
  double volume;
  double moment_r;
  double moment_z;
};

/// Stable identifier for a cell-local corner.
struct CornerId {
  EntityId cell;
  std::uint16_t local_corner;
};

/// Cavity boundary node identifiers in canonical cyclic order.
struct CavityBoundaryLoop {
  std::span<const EntityId> node_ids;
};

/// Canonical operation selected for a ReALE topology transaction.
enum class ReconnectOp : std::uint8_t {
  FaceFlip,
  CellUnion,
  GeneratorDelete,
  PatchRetessellate,
  GeneratorInsert
};

/// Complete immutable input and target-connectivity record for one topology event.
struct TopologyTransaction {
  EntityId event_id;
  ReconnectOp operation;

  std::span<const EntityId> donor_cells;
  std::span<const EntityId> donor_nodes;
  std::span<const CavityBoundaryLoop> fixed_boundary;

  std::span<const EntityId> target_cell_ids;
  std::span<const EntityId> target_node_ids;
  std::span<const std::int32_t> cell_node_offsets;
  std::span<const EntityId> cell_node_ids;

  std::span<const EntityId> created_cells;
  std::span<const EntityId> deleted_cells;
  std::span<const EntityId> created_nodes;
  std::span<const EntityId> deleted_nodes;

  std::uint64_t topology_hash;
};

/// Exact RZ overlap moments linking one target corner to one donor corner.
struct OverlayEntry {
  CornerId target;
  CornerId donor;
  RZMoments moments;
};

/// Conservation, affine-exactness, DeBar, and full-step audit for one committed event.
struct ReconnectCommitReport {
  EntityId event_id;
  std::uint64_t topology_hash_before;
  std::uint64_t topology_hash_after;
  double mass_residual_g;
  double momentum_r_residual_g_cm_s;
  double momentum_z_residual_g_cm_s;
  double energy_residual_erg;
  double affine_max_error;
  bool debar_pass;
  bool g1_full_step_pass;
};

struct Cavity;
struct CandidateSet;
struct Candidate;
struct ConnectivityView;
struct ProtectedTopologyView;
struct GeometryView;
struct G1Witness;
struct MeshView;

/// W2 FROZEN v1 contract: build the minimal cavity around a G1 witness.
Cavity build_minimal_cavity(const G1Witness& witness,
                            const ConnectivityView& connectivity);

/// W2 FROZEN v1 contract: enumerate canonical candidates for a cavity.
CandidateSet enumerate_candidates(const Cavity& cavity,
                                  const ProtectedTopologyView& protected_topology,
                                  const GeometryView& geometry);

/// W2 FROZEN v1 contract: build a target transaction from a selected candidate.
TopologyTransaction build_target_transaction(const Candidate& candidate,
                                             const MeshView& mesh);

/// W3 FROZEN v1 contract: fill caller storage and TENRYU_ASSERT on capacity overflow.
void compute_overlay(const TopologyTransaction& transaction,
                     const GeometryView& geometry,
                     OverlayEntry* out_entries,
                     StorageIndex out_capacity,
                     StorageIndex* out_count);

/// W4 FROZEN v1 contract: project staggered state and fill commit residual fields.
void project_staggered_state(const TopologyTransaction& transaction,
                             const OverlayEntry* overlay_entries,
                             StorageIndex overlay_count,
                             core::State& state,
                             ReconnectCommitReport& report);

}  // namespace tenryu::mesh::reale
