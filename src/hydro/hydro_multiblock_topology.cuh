#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "core/macros.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {

/// Reverse cell-node CSR: for each node n, the list of (cell, corner_index)
/// pairs incident on n, in deterministic block-major + cell-id order.
///
/// Layout:
///   node_offsets[n] = first incident-corner index for node n in node_cells/node_corners
///   node_offsets[n+1] - node_offsets[n] = degree of node n (count of incident corners)
///   node_cells[k]   = cell id of the k-th incident corner
///   node_corners[k] = corner index in {0,...,active_nverts-1} within that cell
///
/// Construction guarantees:
///   - Deterministic: same CSR builds across runs
///   - Sorted: incident corners listed in (cell_id_stable ASC, corner_index ASC) order
///   - Total length: sum of degrees = sum over cells of active nverts
///     (= 4 * n_cells when cell_nverts is absent; forward storage capacity is
///     stride * n_cells)
struct ReverseCellNodeCSR {
  std::vector<int> node_offsets;
  std::vector<int> node_cells;
  std::vector<int> node_corners;
};

/// Build reverse cell-node CSR from forward MultiBlockTopology::cell_node_csr_*.
/// Both inputs and outputs are host vectors.
inline ReverseCellNodeCSR build_reverse_cell_node_csr(
    const mesh::MultiBlockTopology& mb,
    const int n_nodes_total,
    const std::vector<std::uint8_t>* cell_nverts = nullptr,
    const int stride = mesh::kMeshTopoCellStorageSlots) {
  TENRYU_ASSERT(n_nodes_total >= 0,
                "reverse cell-node CSR requires non-negative node count");
  TENRYU_ASSERT(!mb.cell_node_csr_offsets.empty(),
                "reverse cell-node CSR requires forward CSR offsets");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.front() == 0,
                "forward cell-node CSR offsets must start at zero");
  const int n_cells =
      static_cast<int>(mb.cell_node_csr_offsets.size()) - 1;
  TENRYU_ASSERT(mb.cell_id_stable.size() ==
                    static_cast<std::size_t>(n_cells),
                "reverse cell-node CSR requires cell_id_stable per cell");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.back() >= 0,
                "forward cell-node CSR total entry count must be non-negative");
  TENRYU_ASSERT(static_cast<std::size_t>(mb.cell_node_csr_offsets.back()) ==
                    mb.cell_node_csr_indices.size(),
                "forward cell-node CSR offset/index size mismatch");
  TENRYU_ASSERT(cell_nverts == nullptr || cell_nverts->empty() ||
                    cell_nverts->size() == static_cast<std::size_t>(n_cells),
                "reverse cell-node CSR requires empty or per-cell cell_nverts");

  std::vector<int> degree(static_cast<std::size_t>(n_nodes_total), 0);
  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int next =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
    TENRYU_ASSERT(next >= off, "forward cell-node CSR offsets must be monotone");
    TENRYU_ASSERT(next - off == stride,
                  "multiblock cell-node CSR width must match mesh stride");
    const int active_nverts = (cell_nverts == nullptr)
                                  ? mesh::kMeshTopoCellStorageSlots
                                  : mesh::mesh_topo_cell_active_nverts(
                                        *cell_nverts, c);
    for (int p = off; p < off + active_nverts; ++p) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(p)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes_total,
                    "forward cell-node CSR node id out of range");
      ++degree[static_cast<std::size_t>(n)];
    }
  }

  ReverseCellNodeCSR out;
  out.node_offsets.assign(static_cast<std::size_t>(n_nodes_total) + 1U, 0);
  for (int n = 0; n < n_nodes_total; ++n) {
    out.node_offsets[static_cast<std::size_t>(n) + 1U] =
        out.node_offsets[static_cast<std::size_t>(n)] +
        degree[static_cast<std::size_t>(n)];
  }

  const int total = out.node_offsets.back();
  out.node_cells.assign(static_cast<std::size_t>(total), -1);
  out.node_corners.assign(static_cast<std::size_t>(total), -1);

  std::vector<int> cursor(static_cast<std::size_t>(n_nodes_total), 0);
  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int next =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
    TENRYU_ASSERT(next - off == stride,
                  "multiblock cell-node CSR width must match mesh stride");
    const int active_nverts = (cell_nverts == nullptr)
                                  ? mesh::kMeshTopoCellStorageSlots
                                  : mesh::mesh_topo_cell_active_nverts(
                                        *cell_nverts, c);
    for (int p = off; p < off + active_nverts; ++p) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(p)];
      const int corner = p - off;
      TENRYU_ASSERT(corner >= 0 && corner < active_nverts,
                    "forward cell-node CSR corner index out of range");
      const int dst = out.node_offsets[static_cast<std::size_t>(n)] +
                      cursor[static_cast<std::size_t>(n)]++;
      out.node_cells[static_cast<std::size_t>(dst)] = c;
      out.node_corners[static_cast<std::size_t>(dst)] = corner;
    }
  }

  for (int n = 0; n < n_nodes_total; ++n) {
    const int off = out.node_offsets[static_cast<std::size_t>(n)];
    const int next = out.node_offsets[static_cast<std::size_t>(n) + 1U];
    std::vector<std::pair<int, int>> incident;
    incident.reserve(static_cast<std::size_t>(next - off));
    for (int p = off; p < next; ++p) {
      incident.emplace_back(out.node_cells[static_cast<std::size_t>(p)],
                            out.node_corners[static_cast<std::size_t>(p)]);
    }
    std::sort(incident.begin(),
              incident.end(),
              [&mb](const std::pair<int, int>& a,
                    const std::pair<int, int>& b) {
                const int stable_a =
                    mb.cell_id_stable[static_cast<std::size_t>(a.first)];
                const int stable_b =
                    mb.cell_id_stable[static_cast<std::size_t>(b.first)];
                if (stable_a != stable_b) {
                  return stable_a < stable_b;
                }
                return a.second < b.second;
              });
    for (int p = off; p < next; ++p) {
      const auto& entry = incident[static_cast<std::size_t>(p - off)];
      out.node_cells[static_cast<std::size_t>(p)] = entry.first;
      out.node_corners[static_cast<std::size_t>(p)] = entry.second;
    }
  }

  return out;
}

inline int find_reciprocal_face(const mesh::MultiBlockTopology& mb,
                                const int cell_b,
                                const int cell_a,
                                const std::vector<std::uint8_t>* cell_nverts =
                                    nullptr) {
  TENRYU_ASSERT(cell_b >= 0, "reciprocal face lookup requires valid cell_b");
  TENRYU_ASSERT(static_cast<std::size_t>(cell_b) + 1U <
                    mb.face_adj_csr_offsets.size(),
                "reciprocal face lookup cell_b out of CSR range");
  const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(cell_b)];
  const int next =
      mb.face_adj_csr_offsets[static_cast<std::size_t>(cell_b) + 1U];
  TENRYU_ASSERT(next >= off, "face adjacency CSR offsets must be monotone");
  const int active_nverts = (cell_nverts == nullptr)
                                ? mesh::kMeshTopoCellStorageSlots
                                : mesh::mesh_topo_cell_active_nverts(
                                      *cell_nverts, cell_b);
  for (int p = off; p < next; ++p) {
    const int local = p - off;
    if (!mesh::mesh_topo_local_face_is_active(active_nverts, local)) {
      continue;
    }
    if (mb.face_adj_csr_indices[static_cast<std::size_t>(p)] == cell_a) {
      return local;
    }
  }
  TENRYU_ASSERT(false, "unique face builder found non-bilateral adjacency");
  return -1;
}

inline void build_unique_oriented_face_list(
    const mesh::MultiBlockTopology& mb,
    const std::vector<int>& cell_id_stable,
    std::vector<mesh::UniqueOrientedFace>& unique_internal,
    std::vector<mesh::UniqueOrientedFace>& boundary,
    const std::vector<std::uint8_t>* cell_nverts = nullptr,
    const int stride = mesh::kMeshTopoCellStorageSlots) {
  TENRYU_ASSERT(!mb.face_adj_csr_offsets.empty(),
                "unique face builder requires face adjacency CSR offsets");
  TENRYU_ASSERT(mb.face_adj_csr_offsets.front() == 0,
                "face adjacency CSR offsets must start at zero");
  const int n_cells =
      static_cast<int>(mb.face_adj_csr_offsets.size()) - 1;
  TENRYU_ASSERT(n_cells >= 0,
                "unique face builder requires non-negative cell count");
  TENRYU_ASSERT(cell_id_stable.size() == static_cast<std::size_t>(n_cells),
                "unique face builder requires stable id per cell");
  TENRYU_ASSERT(mb.face_adj_csr_offsets.back() >= 0,
                "face adjacency CSR entry count must be non-negative");
  const std::size_t n_face_entries =
      static_cast<std::size_t>(mb.face_adj_csr_offsets.back());
  TENRYU_ASSERT(mb.face_adj_csr_indices.size() == n_face_entries,
                "face adjacency CSR index size mismatch");
  TENRYU_ASSERT(mb.face_bc_tags.size() == n_face_entries,
                "face boundary tag size mismatch");
  TENRYU_ASSERT(cell_nverts == nullptr || cell_nverts->empty() ||
                    cell_nverts->size() == static_cast<std::size_t>(n_cells),
                "unique face builder requires empty or per-cell cell_nverts");

  unique_internal.clear();
  boundary.clear();
  unique_internal.reserve(n_face_entries / 2U);
  boundary.reserve(n_face_entries / 4U);

  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
    const int next =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(c) + 1U];
    TENRYU_ASSERT(next >= off, "face adjacency CSR offsets must be monotone");
    TENRYU_ASSERT(next - off == stride,
                  "multiblock face CSR width must match mesh stride");
    const int active_nverts = (cell_nverts == nullptr)
                                  ? mesh::kMeshTopoCellStorageSlots
                                  : mesh::mesh_topo_cell_active_nverts(
                                        *cell_nverts, c);
    for (int p = off; p < next; ++p) {
      const int local = p - off;
      if (!mesh::mesh_topo_local_face_is_active(active_nverts, local)) {
        continue;
      }
      const int neighbor =
          mb.face_adj_csr_indices[static_cast<std::size_t>(p)];
      const int bc_tag = mb.face_bc_tags[static_cast<std::size_t>(p)];
      TENRYU_ASSERT(bc_tag >= 0 && bc_tag <= 127,
                    "unique face builder bc_tag exceeds int8 range");
      if (neighbor == -1) {
        boundary.push_back(mesh::UniqueOrientedFace{
            c, -1, local, -1, static_cast<std::int8_t>(bc_tag)});
        continue;
      }

      TENRYU_ASSERT(neighbor >= 0 && neighbor < n_cells,
                    "unique face builder neighbor cell out of range");
      const int stable_c = cell_id_stable[static_cast<std::size_t>(c)];
      const int stable_n = cell_id_stable[static_cast<std::size_t>(neighbor)];
      TENRYU_ASSERT(stable_c >= 0 && stable_n >= 0,
                    "unique face builder requires valid stable cell ids");
      TENRYU_ASSERT(stable_c != stable_n,
                    "unique face builder requires unique stable cell ids");
      if (stable_c < stable_n) {
        const int local_b = find_reciprocal_face(mb, neighbor, c, cell_nverts);
        unique_internal.push_back(mesh::UniqueOrientedFace{
            c, neighbor, local, local_b, static_cast<std::int8_t>(bc_tag)});
      }
    }
  }

  std::sort(unique_internal.begin(),
            unique_internal.end(),
            [&cell_id_stable](const mesh::UniqueOrientedFace& a,
                              const mesh::UniqueOrientedFace& b) {
              const int stable_a =
                  cell_id_stable[static_cast<std::size_t>(a.cell_a)];
              const int stable_b =
                  cell_id_stable[static_cast<std::size_t>(b.cell_a)];
              if (stable_a != stable_b) {
                return stable_a < stable_b;
              }
              if (a.local_a != b.local_a) {
                return a.local_a < b.local_a;
              }
              return a.cell_b < b.cell_b;
            });
}

TENRYU_HOST_DEVICE inline int reverse_csr_node_degree(
    const int* node_offsets,
    const int n) {
  return node_offsets[n + 1] - node_offsets[n];
}

TENRYU_HOST_DEVICE inline int reverse_csr_node_cell(
    const int* node_offsets,
    const int* node_cells,
    const int n,
    const int k) {
  return node_cells[node_offsets[n] + k];
}

TENRYU_HOST_DEVICE inline int reverse_csr_node_corner(
    const int* node_offsets,
    const int* node_corners,
    const int n,
    const int k) {
  return node_corners[node_offsets[n] + k];
}

}  // namespace tenryu::hydro
