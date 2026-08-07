#include "mesh/topology_edge_flip.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tenryu::mesh {
namespace {

struct CellTriangle {
  int cell;
  int begin;
  int end;
  std::array<int, 3> nodes;
};

bool load_triangle(const MultiBlockTopology& topology,
                   const std::vector<std::uint8_t>& cell_nverts,
                   const int cell,
                   const int n_nodes,
                   CellTriangle& triangle) {
  if (cell < 0 ||
      static_cast<std::size_t>(cell) >= cell_nverts.size() ||
      static_cast<std::size_t>(cell + 1) >=
          topology.cell_node_csr_offsets.size() ||
      cell_nverts[static_cast<std::size_t>(cell)] != 3U) {
    return false;
  }

  const int begin =
      topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int end =
      topology.cell_node_csr_offsets[static_cast<std::size_t>(cell + 1)];
  const int stride = end - begin;
  if (begin < 0 || (stride != 4 && stride != 8) ||
      static_cast<std::size_t>(end) >
          topology.cell_node_csr_indices.size()) {
    return false;
  }

  triangle.cell = cell;
  triangle.begin = begin;
  triangle.end = end;
  for (int local = 0; local < 3; ++local) {
    const int node = topology.cell_node_csr_indices[
        static_cast<std::size_t>(begin + local)];
    if (node < 0 || node >= n_nodes) {
      return false;
    }
    triangle.nodes[static_cast<std::size_t>(local)] = node;
  }
  if (triangle.nodes[0] == triangle.nodes[1] ||
      triangle.nodes[1] == triangle.nodes[2] ||
      triangle.nodes[2] == triangle.nodes[0]) {
    return false;
  }
  for (int local = 3; local < stride; ++local) {
    if (topology.cell_node_csr_indices[
            static_cast<std::size_t>(begin + local)] != -1) {
      return false;
    }
  }
  return true;
}

int find_node(const std::array<int, 3>& nodes, const int node) {
  for (int local = 0; local < 3; ++local) {
    if (nodes[static_cast<std::size_t>(local)] == node) {
      return local;
    }
  }
  return -1;
}

bool find_opposite(const std::array<int, 3>& nodes,
                   const int shared_na,
                   const int shared_nb,
                   int& opposite) {
  if (shared_na == shared_nb ||
      find_node(nodes, shared_na) < 0 ||
      find_node(nodes, shared_nb) < 0) {
    return false;
  }
  for (const int node : nodes) {
    if (node != shared_na && node != shared_nb) {
      opposite = node;
      return true;
    }
  }
  return false;
}

double cross(const double ar,
             const double az,
             const double br,
             const double bz,
             const double cr,
             const double cz) {
  return (br - ar) * (cz - bz) - (bz - az) * (cr - br);
}

bool quad_is_strictly_convex(const std::array<int, 4>& quad,
                             const double* node_r,
                             const double* node_z) {
  double first_cross = 0.0;
  bool first_positive = false;
  for (int corner = 0; corner < 4; ++corner) {
    const int a = quad[static_cast<std::size_t>(corner)];
    const int b = quad[static_cast<std::size_t>((corner + 1) % 4)];
    const int c = quad[static_cast<std::size_t>((corner + 2) % 4)];
    const double value =
        cross(node_r[a], node_z[a], node_r[b], node_z[b],
              node_r[c], node_z[c]);
    if (!std::isfinite(value) || value == 0.0) {
      return false;
    }
    if (corner == 0) {
      first_cross = value;
      first_positive = first_cross > 0.0;
    } else if ((value > 0.0) != first_positive) {
      return false;
    }
  }
  return true;
}

double triangle_twice_signed_area(const std::array<int, 3>& nodes,
                                  const double* node_r,
                                  const double* node_z) {
  double twice_area = 0.0;
  for (int local = 0; local < 3; ++local) {
    const int next = (local + 1) % 3;
    const int node = nodes[static_cast<std::size_t>(local)];
    const int next_node = nodes[static_cast<std::size_t>(next)];
    twice_area +=
        node_r[node] * node_z[next_node] -
        node_r[next_node] * node_z[node];
  }
  return twice_area;
}

void write_triangle(MultiBlockTopology& topology,
                    const CellTriangle& triangle,
                    const std::array<int, 3>& nodes) {
  for (int local = 0; local < 3; ++local) {
    topology.cell_node_csr_indices[
        static_cast<std::size_t>(triangle.begin + local)] =
        nodes[static_cast<std::size_t>(local)];
  }
}

}  // namespace

bool topology_edge_flip_apply(
    MultiBlockTopology& topology,
    std::vector<std::uint8_t>& cell_nverts,
    const double* node_r,
    const double* node_z,
    const int n_nodes,
    EdgeFlipEvent& event) {
  if (event.cell_a == event.cell_b || node_r == nullptr ||
      node_z == nullptr || n_nodes <= 0 ||
      event.shared_na < 0 || event.shared_na >= n_nodes ||
      event.shared_nb < 0 || event.shared_nb >= n_nodes) {
    return false;
  }

  CellTriangle cell_a{};
  CellTriangle cell_b{};
  if (!load_triangle(topology, cell_nverts, event.cell_a, n_nodes, cell_a) ||
      !load_triangle(topology, cell_nverts, event.cell_b, n_nodes, cell_b) ||
      static_cast<std::size_t>(event.cell_a) >=
          topology.cell_id_stable.size() ||
      static_cast<std::size_t>(event.cell_b) >=
          topology.cell_id_stable.size()) {
    return false;
  }

  int opposite_nc = -1;
  int opposite_nd = -1;
  if (!find_opposite(cell_a.nodes, event.shared_na, event.shared_nb,
                     opposite_nc) ||
      !find_opposite(cell_b.nodes, event.shared_na, event.shared_nb,
                     opposite_nd) ||
      opposite_nc == opposite_nd) {
    return false;
  }

  int shared_count = 0;
  for (const int node_a : cell_a.nodes) {
    if (find_node(cell_b.nodes, node_a) >= 0) {
      ++shared_count;
    }
  }
  if (shared_count != 2) {
    return false;
  }

  const std::array<int, 4> quad{
      event.shared_na, opposite_nc, event.shared_nb, opposite_nd};
  if (!quad_is_strictly_convex(quad, node_r, node_z)) {
    return false;
  }

  std::array<int, 3> new_a = cell_a.nodes;
  std::array<int, 3> new_b = cell_b.nodes;
  new_a[static_cast<std::size_t>(
      find_node(new_a, event.shared_nb))] = opposite_nd;
  new_b[static_cast<std::size_t>(
      find_node(new_b, event.shared_na))] = opposite_nc;

  const double area2_a =
      triangle_twice_signed_area(new_a, node_r, node_z);
  const double area2_b =
      triangle_twice_signed_area(new_b, node_r, node_z);
  if (!std::isfinite(area2_a) || !std::isfinite(area2_b) ||
      area2_a <= 0.0 || area2_b <= 0.0) {
    return false;
  }

  const int stable_a =
      topology.cell_id_stable[static_cast<std::size_t>(event.cell_a)];
  const int stable_b =
      topology.cell_id_stable[static_cast<std::size_t>(event.cell_b)];
  const bool a_first =
      stable_a < stable_b ||
      (stable_a == stable_b && event.cell_a < event.cell_b);
  if (a_first) {
    write_triangle(topology, cell_a, new_a);
    write_triangle(topology, cell_b, new_b);
  } else {
    write_triangle(topology, cell_b, new_b);
    write_triangle(topology, cell_a, new_a);
  }

  event.opposite_nc = opposite_nc;
  event.opposite_nd = opposite_nd;
  return true;
}

EdgeFlipEvent topology_edge_flip_reverse(const EdgeFlipEvent& event) {
  return {
      event.cell_a,
      event.cell_b,
      event.opposite_nc,
      event.opposite_nd,
      event.shared_na,
      event.shared_nb,
  };
}

}  // namespace tenryu::mesh
