#include "hydro/mortar_chain_extract.hpp"

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

namespace tenryu::hydro {
namespace {

using Edge = std::pair<int, int>;

Edge canonical_edge(const int a, const int b) {
  return std::minmax(a, b);
}

}  // namespace

BoundaryChain mortar_extract_boundary_chain(
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const int n_cells,
    const int corner_stride,
    const int n_nodes,
    const std::uint8_t* node_flags,
    const std::uint8_t node_flag_mask,
    const double* node_r,
    const double* node_z) {
  BoundaryChain chain;
  if (cell_node_csr_offsets == nullptr ||
      cell_node_csr_indices == nullptr ||
      n_cells <= 0 ||
      corner_stride <= 0 ||
      n_nodes <= 0 ||
      node_flags == nullptr ||
      node_r == nullptr ||
      node_z == nullptr) {
    return chain;
  }

  std::map<Edge, int> face_counts;
  for (int cell = 0; cell < n_cells; ++cell) {
    const int begin = cell_node_csr_offsets[cell];
    const int end = cell_node_csr_offsets[cell + 1];
    if (begin < 0 || end < begin || end - begin != corner_stride) {
      return {};
    }

    std::vector<int> nodes;
    if (cell_nverts != nullptr) {
      const int nverts = static_cast<int>(cell_nverts[cell]);
      if (nverts < 2 || nverts > corner_stride) {
        return {};
      }
      nodes.reserve(static_cast<std::size_t>(nverts));
      for (int local = 0; local < nverts; ++local) {
        const int node = cell_node_csr_indices[begin + local];
        if (node < 0 || node >= n_nodes) {
          return {};
        }
        nodes.push_back(node);
      }
    } else {
      if (corner_stride != 4) {
        return {};
      }
      nodes.reserve(4U);
      for (int local = 0; local < 4; ++local) {
        const int node = cell_node_csr_indices[begin + local];
        if (node >= n_nodes) {
          return {};
        }
        if (node >= 0) {
          nodes.push_back(node);
        }
      }
      if (nodes.size() < 2U) {
        return {};
      }
    }

    for (std::size_t local = 0; local < nodes.size(); ++local) {
      const int a = nodes[local];
      const int b = nodes[(local + 1U) % nodes.size()];
      if (a == b) {
        return {};
      }
      ++face_counts[canonical_edge(a, b)];
    }
  }

  std::vector<Edge> class_edges;
  for (const auto& [edge, count] : face_counts) {
    if (count != 1) {
      continue;
    }
    const auto matches_class = [&](const int node) {
      return (node_flags[node] & node_flag_mask) == node_flag_mask;
    };
    if (matches_class(edge.first) && matches_class(edge.second)) {
      class_edges.push_back(edge);
    }
  }
  if (class_edges.empty()) {
    return {};
  }

  std::vector<std::vector<int>> neighbors(
      static_cast<std::size_t>(n_nodes));
  for (const Edge& edge : class_edges) {
    neighbors[static_cast<std::size_t>(edge.first)].push_back(edge.second);
    neighbors[static_cast<std::size_t>(edge.second)].push_back(edge.first);
  }

  std::vector<int> endpoints;
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t degree = neighbors[static_cast<std::size_t>(node)].size();
    if (degree == 1U) {
      endpoints.push_back(node);
    } else if (degree > 2U) {
      return {};
    }
  }
  if (endpoints.size() != 2U) {
    return {};
  }

  const int start = std::min(endpoints[0], endpoints[1]);
  int previous = -1;
  int current = start;
  while (true) {
    chain.node_ids.push_back(current);
    chain.r.push_back(node_r[current]);
    chain.z.push_back(node_z[current]);

    int next = -1;
    for (const int neighbor : neighbors[static_cast<std::size_t>(current)]) {
      if (neighbor != previous) {
        if (next >= 0) {
          return {};
        }
        next = neighbor;
      }
    }
    if (next < 0) {
      break;
    }
    previous = current;
    current = next;
  }

  if (chain.node_ids.size() != class_edges.size() + 1U ||
      current != std::max(endpoints[0], endpoints[1])) {
    return {};
  }
  return chain;
}

MortarSlideLine mortar_build_from_chains(const BoundaryChain& master,
                                         const BoundaryChain& slave) {
  return mortar_build(
      master.r.data(),
      master.z.data(),
      static_cast<int>(master.r.size()),
      slave.r.data(),
      slave.z.data(),
      static_cast<int>(slave.r.size()));
}

}  // namespace tenryu::hydro
