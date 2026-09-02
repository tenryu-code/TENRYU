#include "mesh/reconnect.cuh"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace tenryu::mesh::reale {

std::uint64_t polygon_topo_hash(const EntityId* node_ids, int n);

namespace detail {  // W1 externs

using ::tenryu::mesh::reale::polygon_topo_hash;

struct EdgeUse {
  int cell;
  int local_edge;
};

using Edge = std::pair<int, int>;

bool cell_is_active(const ConnectivityView& connectivity, const int cell) {
  return cell >= 0 && cell < connectivity.n_cells &&
         (connectivity.hydro_active == nullptr ||
          connectivity.hydro_active[cell] != 0U);
}

bool cell_is_protected(const ConnectivityView& connectivity, const int cell) {
  return connectivity.cell_protected != nullptr &&
         connectivity.cell_protected[cell] != 0U;
}

bool cell_is_eligible(const ConnectivityView& connectivity, const int cell) {
  return cell_is_active(connectivity, cell) &&
         !cell_is_protected(connectivity, cell);
}

int cell_nverts(const ConnectivityView& connectivity, const int cell) {
  return static_cast<int>(connectivity.cell_nverts[cell]);
}

std::vector<EntityId> cell_cycle(const ConnectivityView& connectivity,
                                 const int cell) {
  const int begin = connectivity.cell_node_offsets[cell];
  const int nverts = cell_nverts(connectivity, cell);
  std::vector<EntityId> cycle;
  cycle.reserve(static_cast<std::size_t>(nverts));
  for (int corner = 0; corner < nverts; ++corner) {
    cycle.push_back(static_cast<EntityId>(
        connectivity.cell_node_indices[begin + corner]));
  }
  if (connectivity.cell_orientation_sign[cell] < 0) {
    std::reverse(cycle.begin(), cycle.end());
  }
  return cycle;
}

void rotate_to_minimum(std::vector<EntityId>& cycle) {
  if (cycle.empty()) {
    return;
  }
  const auto minimum = std::min_element(cycle.begin(), cycle.end());
  std::rotate(cycle.begin(), minimum, cycle.end());
}

std::vector<EntityId> sorted_unique_nodes(
    const std::vector<std::vector<EntityId>>& cycles) {
  std::vector<EntityId> nodes;
  for (const std::vector<EntityId>& cycle : cycles) {
    nodes.insert(nodes.end(), cycle.begin(), cycle.end());
  }
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  return nodes;
}

std::vector<EntityId> sorted_unique_donor_nodes(
    const ConnectivityView& connectivity,
    const std::vector<EntityId>& donors) {
  std::vector<std::vector<EntityId>> cycles;
  cycles.reserve(donors.size());
  for (const EntityId donor : donors) {
    cycles.push_back(cell_cycle(connectivity, static_cast<int>(donor)));
  }
  return sorted_unique_nodes(cycles);
}

Edge canonical_edge(const EntityId first, const EntityId second) {
  const int a = static_cast<int>(first);
  const int b = static_cast<int>(second);
  return a < b ? Edge{a, b} : Edge{b, a};
}

std::map<Edge, std::vector<EdgeUse>> internal_edge_uses(
    const Cavity& cavity) {
  std::map<Edge, std::vector<EdgeUse>> uses;
  for (const EntityId cell_id : cavity.cells) {
    const int cell = static_cast<int>(cell_id);
    if (!cell_is_eligible(cavity.connectivity, cell)) {
      continue;
    }
    const std::vector<EntityId> cycle = cell_cycle(cavity.connectivity, cell);
    const int nverts = static_cast<int>(cycle.size());
    for (int edge = 0; edge < nverts; ++edge) {
      const Edge key = canonical_edge(
          cycle[edge], cycle[(edge + 1) % nverts]);
      uses[key].push_back(EdgeUse{cell, edge});
    }
  }
  for (auto iterator = uses.begin(); iterator != uses.end();) {
    if (iterator->second.size() != 2U) {
      iterator = uses.erase(iterator);
    } else {
      ++iterator;
    }
  }
  return uses;
}

std::vector<EntityId> union_boundary_loop(
    const ConnectivityView& connectivity,
    const EdgeUse& first_use,
    const EdgeUse& second_use) {
  std::vector<EntityId> first = cell_cycle(connectivity, first_use.cell);
  std::vector<EntityId> second = cell_cycle(connectivity, second_use.cell);

  int first_edge = first_use.local_edge;
  int second_edge = second_use.local_edge;
  const EntityId first_start = first[first_edge];
  const EntityId first_end = first[(first_edge + 1) % first.size()];
  const EntityId second_start = second[second_edge];
  const EntityId second_end = second[(second_edge + 1) % second.size()];

  if (second_start == first_start && second_end == first_end) {
    std::reverse(second.begin(), second.end());
    second_edge = static_cast<int>(second.size()) - second_edge - 2;
    if (second_edge < 0) {
      second_edge += static_cast<int>(second.size());
    }
  }

  std::vector<EntityId> boundary;
  boundary.reserve(first.size() + second.size() - 2U);
  for (std::size_t offset = 1; offset <= first.size(); ++offset) {
    boundary.push_back(first[(static_cast<std::size_t>(first_edge) + offset) %
                             first.size()]);
  }
  for (std::size_t offset = 2; offset < second.size(); ++offset) {
    boundary.push_back(
        second[(static_cast<std::size_t>(second_edge) + offset) %
               second.size()]);
  }
  rotate_to_minimum(boundary);
  return boundary;
}

std::vector<EntityId> boundary_path(const std::vector<EntityId>& boundary,
                                    const int first,
                                    const int last) {
  std::vector<EntityId> path;
  int position = first;
  while (true) {
    path.push_back(boundary[static_cast<std::size_t>(position)]);
    if (position == last) {
      break;
    }
    position = (position + 1) % static_cast<int>(boundary.size());
  }
  rotate_to_minimum(path);
  return path;
}

void set_target_connectivity(
    Candidate& candidate,
    std::vector<std::pair<EntityId, std::vector<EntityId>>> target_cells) {
  std::sort(target_cells.begin(), target_cells.end(),
            [](const auto& lhs, const auto& rhs) {
              return std::tie(lhs.first, lhs.second) <
                     std::tie(rhs.first, rhs.second);
            });

  candidate.target_cell_ids.clear();
  candidate.cell_node_offsets.clear();
  candidate.cell_node_ids.clear();
  candidate.cell_node_offsets.push_back(0);
  std::vector<std::vector<EntityId>> target_cycles;
  target_cycles.reserve(target_cells.size());
  for (auto& target_cell : target_cells) {
    candidate.target_cell_ids.push_back(target_cell.first);
    target_cycles.push_back(std::move(target_cell.second));
    candidate.cell_node_ids.insert(candidate.cell_node_ids.end(),
                                   target_cycles.back().begin(),
                                   target_cycles.back().end());
    candidate.cell_node_offsets.push_back(
        static_cast<std::int32_t>(candidate.cell_node_ids.size()));
  }
  candidate.target_node_ids = sorted_unique_nodes(target_cycles);
}

Candidate base_candidate(const ReconnectOp op,
                         const ConnectivityView& connectivity,
                         const EdgeUse& first_use,
                         const EdgeUse& second_use,
                         const std::vector<EntityId>& boundary) {
  Candidate candidate;
  candidate.op = op;
  candidate.donors = {
      static_cast<EntityId>(std::min(first_use.cell, second_use.cell)),
      static_cast<EntityId>(std::max(first_use.cell, second_use.cell))};
  candidate.donor_nodes =
      sorted_unique_donor_nodes(connectivity, candidate.donors);
  candidate.fixed_boundary_loop_nodes = boundary;
  candidate.refresh_fixed_boundary_view();
  return candidate;
}

void append_face_flip_candidates(CandidateSet& candidates,
                                 const ConnectivityView& connectivity,
                                 const EdgeUse& first_use,
                                 const EdgeUse& second_use,
                                 const std::vector<EntityId>& boundary) {
  if (cell_nverts(connectivity, first_use.cell) != 4 ||
      cell_nverts(connectivity, second_use.cell) != 4 ||
      boundary.size() != 6U) {
    return;
  }

  const std::vector<EntityId> first_cycle =
      cell_cycle(connectivity, first_use.cell);
  const Edge original_diagonal = canonical_edge(
      first_cycle[first_use.local_edge],
      first_cycle[(first_use.local_edge + 1) % first_cycle.size()]);
  for (int diagonal_start = 0; diagonal_start < 3; ++diagonal_start) {
    const int diagonal_end = diagonal_start + 3;
    if (canonical_edge(boundary[diagonal_start], boundary[diagonal_end]) ==
        original_diagonal) {
      continue;
    }
    std::vector<EntityId> first_target =
        boundary_path(boundary, diagonal_start, diagonal_end);
    std::vector<EntityId> second_target =
        boundary_path(boundary, diagonal_end, diagonal_start);
    if (second_target < first_target) {
      std::swap(first_target, second_target);
    }

    Candidate candidate = base_candidate(ReconnectOp::FaceFlip, connectivity,
                                         first_use, second_use, boundary);
    set_target_connectivity(
        candidate,
        {{candidate.donors[0], std::move(first_target)},
         {candidate.donors[1], std::move(second_target)}});
    candidates.ordered.push_back(std::move(candidate));
  }
}

void append_cell_union_candidate(CandidateSet& candidates,
                                 const ConnectivityView& connectivity,
                                 const EdgeUse& first_use,
                                 const EdgeUse& second_use,
                                 const std::vector<EntityId>& boundary) {
  if (boundary.size() > 8U) {
    return;
  }
  Candidate candidate = base_candidate(ReconnectOp::CellUnion, connectivity,
                                       first_use, second_use, boundary);
  set_target_connectivity(
      candidate, {{candidate.donors[0], boundary}});
  candidate.deleted_cells.push_back(candidate.donors[1]);
  candidates.ordered.push_back(std::move(candidate));
}

int operation_priority(const ReconnectOp operation) {
  switch (operation) {
    case ReconnectOp::FaceFlip:
      return 0;
    case ReconnectOp::CellUnion:
      return 1;
    case ReconnectOp::GeneratorDelete:
      return 2;
    case ReconnectOp::PatchRetessellate:
      return 3;
    case ReconnectOp::GeneratorInsert:
      return 4;
  }
  return 5;
}

bool candidate_less(const Candidate& lhs, const Candidate& rhs) {
  const int lhs_priority = operation_priority(lhs.op);
  const int rhs_priority = operation_priority(rhs.op);
  if (lhs_priority != rhs_priority) {
    return lhs_priority < rhs_priority;
  }
  return std::tie(lhs.donors, lhs.cell_node_offsets, lhs.cell_node_ids,
                  lhs.target_cell_ids) <
         std::tie(rhs.donors, rhs.cell_node_offsets, rhs.cell_node_ids,
                  rhs.target_cell_ids);
}

void recompute_cavity_boundary(Cavity& cavity) {
  std::set<int> cavity_cells;
  for (const EntityId cell : cavity.cells) {
    cavity_cells.insert(static_cast<int>(cell));
  }

  std::set<int> cavity_nodes;
  for (const EntityId cell_id : cavity.cells) {
    const int cell = static_cast<int>(cell_id);
    const int begin = cavity.connectivity.cell_node_offsets[cell];
    const int nverts = cell_nverts(cavity.connectivity, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      cavity_nodes.insert(
          cavity.connectivity.cell_node_indices[begin + corner]);
    }
  }

  std::set<EntityId> boundary_nodes;
  for (int cell = 0; cell < cavity.connectivity.n_cells; ++cell) {
    if (!cell_is_active(cavity.connectivity, cell) ||
        cavity_cells.count(cell) != 0U) {
      continue;
    }
    const int begin = cavity.connectivity.cell_node_offsets[cell];
    const int nverts = cell_nverts(cavity.connectivity, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = cavity.connectivity.cell_node_indices[begin + corner];
      if (cavity_nodes.count(node) != 0U) {
        boundary_nodes.insert(static_cast<EntityId>(node));
      }
    }
  }
  cavity.boundary_nodes.assign(boundary_nodes.begin(), boundary_nodes.end());
}

std::uint64_t combine_target_topology_hash(const Candidate& candidate) {
  constexpr std::uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  std::uint64_t hash = kFnvOffsetBasis;
  for (std::size_t cell = 0; cell < candidate.target_cell_ids.size(); ++cell) {
    const std::int32_t begin = candidate.cell_node_offsets[cell];
    const std::int32_t end = candidate.cell_node_offsets[cell + 1U];
    const std::uint64_t polygon_hash = detail::polygon_topo_hash(
        candidate.cell_node_ids.data() + begin, end - begin);
    for (unsigned int byte = 0; byte < 8U; ++byte) {
      hash ^= (polygon_hash >> (8U * byte)) & 0xffU;
      hash *= kFnvPrime;
    }
  }
  return hash;
}

}  // namespace detail

Candidate::Candidate(const Candidate& other)
    : op(other.op),
      donors(other.donors),
      donor_nodes(other.donor_nodes),
      fixed_boundary_loop_nodes(other.fixed_boundary_loop_nodes),
      target_cell_ids(other.target_cell_ids),
      target_node_ids(other.target_node_ids),
      cell_node_offsets(other.cell_node_offsets),
      cell_node_ids(other.cell_node_ids),
      created_cells(other.created_cells),
      deleted_cells(other.deleted_cells),
      created_nodes(other.created_nodes),
      deleted_nodes(other.deleted_nodes) {
  refresh_fixed_boundary_view();
}

Candidate& Candidate::operator=(const Candidate& other) {
  if (this == &other) {
    return *this;
  }
  op = other.op;
  donors = other.donors;
  donor_nodes = other.donor_nodes;
  fixed_boundary_loop_nodes = other.fixed_boundary_loop_nodes;
  target_cell_ids = other.target_cell_ids;
  target_node_ids = other.target_node_ids;
  cell_node_offsets = other.cell_node_offsets;
  cell_node_ids = other.cell_node_ids;
  created_cells = other.created_cells;
  deleted_cells = other.deleted_cells;
  created_nodes = other.created_nodes;
  deleted_nodes = other.deleted_nodes;
  refresh_fixed_boundary_view();
  return *this;
}

Candidate::Candidate(Candidate&& other) noexcept
    : op(other.op),
      donors(std::move(other.donors)),
      donor_nodes(std::move(other.donor_nodes)),
      fixed_boundary_loop_nodes(std::move(other.fixed_boundary_loop_nodes)),
      target_cell_ids(std::move(other.target_cell_ids)),
      target_node_ids(std::move(other.target_node_ids)),
      cell_node_offsets(std::move(other.cell_node_offsets)),
      cell_node_ids(std::move(other.cell_node_ids)),
      created_cells(std::move(other.created_cells)),
      deleted_cells(std::move(other.deleted_cells)),
      created_nodes(std::move(other.created_nodes)),
      deleted_nodes(std::move(other.deleted_nodes)) {
  refresh_fixed_boundary_view();
  other.fixed_boundary.clear();
}

Candidate& Candidate::operator=(Candidate&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  op = other.op;
  donors = std::move(other.donors);
  donor_nodes = std::move(other.donor_nodes);
  fixed_boundary_loop_nodes = std::move(other.fixed_boundary_loop_nodes);
  target_cell_ids = std::move(other.target_cell_ids);
  target_node_ids = std::move(other.target_node_ids);
  cell_node_offsets = std::move(other.cell_node_offsets);
  cell_node_ids = std::move(other.cell_node_ids);
  created_cells = std::move(other.created_cells);
  deleted_cells = std::move(other.deleted_cells);
  created_nodes = std::move(other.created_nodes);
  deleted_nodes = std::move(other.deleted_nodes);
  refresh_fixed_boundary_view();
  other.fixed_boundary.clear();
  return *this;
}

void Candidate::refresh_fixed_boundary_view() {
  fixed_boundary.clear();
  if (!fixed_boundary_loop_nodes.empty()) {
    fixed_boundary.push_back(CavityBoundaryLoop{
        std::span<const EntityId>(fixed_boundary_loop_nodes)});
  }
}

Cavity build_minimal_cavity(const G1Witness& witness,
                            const ConnectivityView& connectivity) {
  Cavity cavity{{}, {}, 0, connectivity};
  if (!detail::cell_is_eligible(connectivity, witness.cell)) {
    return cavity;
  }

  std::set<int> witness_nodes;
  const int witness_begin = connectivity.cell_node_offsets[witness.cell];
  const int witness_nverts = detail::cell_nverts(connectivity, witness.cell);
  for (int corner = 0; corner < witness_nverts; ++corner) {
    witness_nodes.insert(
        connectivity.cell_node_indices[witness_begin + corner]);
  }

  for (int cell = 0; cell < connectivity.n_cells; ++cell) {
    if (!detail::cell_is_eligible(connectivity, cell)) {
      continue;
    }
    const int begin = connectivity.cell_node_offsets[cell];
    const int nverts = detail::cell_nverts(connectivity, cell);
    bool shares_witness_node = false;
    for (int corner = 0; corner < nverts; ++corner) {
      if (witness_nodes.count(connectivity.cell_node_indices[begin + corner]) !=
          0U) {
        shares_witness_node = true;
        break;
      }
    }
    if (shares_witness_node) {
      cavity.cells.push_back(static_cast<EntityId>(cell));
    }
  }
  cavity.rings = 1;
  detail::recompute_cavity_boundary(cavity);
  return cavity;
}

void expand_cavity(Cavity& cavity, const ConnectivityView& connectivity) {
  cavity.connectivity = connectivity;
  if (cavity.cells.empty()) {
    return;
  }

  std::set<int> closure_nodes;
  for (const EntityId cell_id : cavity.cells) {
    const int cell = static_cast<int>(cell_id);
    if (!detail::cell_is_eligible(connectivity, cell)) {
      continue;
    }
    const int begin = connectivity.cell_node_offsets[cell];
    const int nverts = detail::cell_nverts(connectivity, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      closure_nodes.insert(connectivity.cell_node_indices[begin + corner]);
    }
  }

  std::set<int> expanded_cells;
  for (const EntityId cell : cavity.cells) {
    const int cell_index = static_cast<int>(cell);
    if (detail::cell_is_eligible(connectivity, cell_index)) {
      expanded_cells.insert(cell_index);
    }
  }
  for (int cell = 0; cell < connectivity.n_cells; ++cell) {
    if (!detail::cell_is_eligible(connectivity, cell)) {
      continue;
    }
    const int begin = connectivity.cell_node_offsets[cell];
    const int nverts = detail::cell_nverts(connectivity, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      if (closure_nodes.count(connectivity.cell_node_indices[begin + corner]) !=
          0U) {
        expanded_cells.insert(cell);
        break;
      }
    }
  }

  cavity.cells.clear();
  for (const int cell : expanded_cells) {
    cavity.cells.push_back(static_cast<EntityId>(cell));
  }
  ++cavity.rings;
  detail::recompute_cavity_boundary(cavity);
}

CandidateSet enumerate_candidates(
    const Cavity& cavity,
    const ProtectedTopologyView& protected_topology,
    const GeometryView& geometry) {
  (void)geometry;
  CandidateSet candidates;
  const std::map<detail::Edge, std::vector<detail::EdgeUse>> internal_edges =
      detail::internal_edge_uses(cavity);
  for (const auto& [edge, uses] : internal_edges) {
    (void)edge;
    const detail::EdgeUse first_use = uses[0];
    const detail::EdgeUse second_use = uses[1];
    if ((protected_topology.cell_protected != nullptr &&
         protected_topology.cell_protected[first_use.cell] != 0U) ||
        (protected_topology.cell_protected != nullptr &&
         protected_topology.cell_protected[second_use.cell] != 0U)) {
      continue;
    }
    const std::vector<EntityId> boundary = detail::union_boundary_loop(
        cavity.connectivity, first_use, second_use);
    detail::append_face_flip_candidates(candidates, cavity.connectivity,
                                        first_use, second_use, boundary);
    detail::append_cell_union_candidate(candidates, cavity.connectivity,
                                        first_use, second_use, boundary);
  }
  std::sort(candidates.ordered.begin(), candidates.ordered.end(),
            detail::candidate_less);
  return candidates;
}

TopologyTransaction build_target_transaction(const Candidate& candidate,
                                             const MeshView& mesh) {
  TopologyTransaction transaction{};
  transaction.event_id = mesh.event_id;
  transaction.operation = candidate.op;
  transaction.donor_cells = std::span<const EntityId>(candidate.donors);
  transaction.donor_nodes = std::span<const EntityId>(candidate.donor_nodes);
  transaction.fixed_boundary =
      std::span<const CavityBoundaryLoop>(candidate.fixed_boundary);
  transaction.target_cell_ids =
      std::span<const EntityId>(candidate.target_cell_ids);
  transaction.target_node_ids =
      std::span<const EntityId>(candidate.target_node_ids);
  transaction.cell_node_offsets =
      std::span<const std::int32_t>(candidate.cell_node_offsets);
  transaction.cell_node_ids =
      std::span<const EntityId>(candidate.cell_node_ids);
  transaction.created_cells =
      std::span<const EntityId>(candidate.created_cells);
  transaction.deleted_cells =
      std::span<const EntityId>(candidate.deleted_cells);
  transaction.created_nodes =
      std::span<const EntityId>(candidate.created_nodes);
  transaction.deleted_nodes =
      std::span<const EntityId>(candidate.deleted_nodes);
  transaction.topology_hash = detail::combine_target_topology_hash(candidate);
  return transaction;
}

}  // namespace tenryu::mesh::reale
