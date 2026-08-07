#include "mesh/topology_coarsen_refine.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "mesh/mesh.hpp"

namespace tenryu::mesh {

namespace {

constexpr int kCellStride = 8;
constexpr double kPi = 3.141592653589793238462643383279502884;

bool cell_row_offset(const MultiBlockTopology& topology,
                     const int cell,
                     const std::size_t n_cells,
                     std::size_t& offset_out) {
  if (cell < 0 || static_cast<std::size_t>(cell) >= n_cells ||
      topology.cell_node_csr_offsets.size() < n_cells + 1U) {
    return false;
  }

  const int begin =
      topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int end =
      topology.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
  if (begin < 0 || end - begin != kCellStride ||
      static_cast<std::size_t>(end) >
          topology.cell_node_csr_indices.size()) {
    return false;
  }

  offset_out = static_cast<std::size_t>(begin);
  return true;
}

bool read_cycle(const MultiBlockTopology& topology,
                const std::vector<std::uint8_t>& cell_nverts,
                const int cell,
                const int n_nodes,
                std::array<int, kCellStride>& cycle_out,
                std::size_t& offset_out) {
  const std::size_t n_cells = cell_nverts.size();
  if (!cell_row_offset(topology, cell, n_cells, offset_out)) {
    return false;
  }

  const int nverts =
      static_cast<int>(cell_nverts[static_cast<std::size_t>(cell)]);
  if (nverts < 3 || nverts > kCellStride) {
    return false;
  }

  cycle_out.fill(-1);
  for (int k = 0; k < nverts; ++k) {
    const int node =
        topology.cell_node_csr_indices[offset_out +
                                       static_cast<std::size_t>(k)];
    if (node < 0 || node >= n_nodes) {
      return false;
    }
    for (int previous = 0; previous < k; ++previous) {
      if (cycle_out[static_cast<std::size_t>(previous)] == node) {
        return false;
      }
    }
    cycle_out[static_cast<std::size_t>(k)] = node;
  }
  return true;
}

bool merge_cycles(const std::array<int, kCellStride>& cycle_a,
                  const int nverts_a,
                  const std::array<int, kCellStride>& cycle_b,
                  const int nverts_b,
                  std::vector<int>& union_cycle) {
  std::array<int, 2> shared{{-1, -1}};
  int shared_count = 0;
  for (int ka = 0; ka < nverts_a; ++ka) {
    const int node_a = cycle_a[static_cast<std::size_t>(ka)];
    for (int kb = 0; kb < nverts_b; ++kb) {
      if (node_a == cycle_b[static_cast<std::size_t>(kb)]) {
        if (shared_count >= static_cast<int>(shared.size())) {
          return false;
        }
        shared[static_cast<std::size_t>(shared_count++)] = node_a;
        break;
      }
    }
  }
  if (shared_count != 2 || nverts_a + nverts_b - 2 > kCellStride) {
    return false;
  }

  int edge_a = -1;
  for (int k = 0; k < nverts_a; ++k) {
    const int next = (k + 1) % nverts_a;
    const int node = cycle_a[static_cast<std::size_t>(k)];
    const int node_next = cycle_a[static_cast<std::size_t>(next)];
    if ((node == shared[0] && node_next == shared[1]) ||
        (node == shared[1] && node_next == shared[0])) {
      edge_a = k;
      break;
    }
  }
  if (edge_a < 0) {
    return false;
  }

  const int edge_start_a = cycle_a[static_cast<std::size_t>(edge_a)];
  const int edge_end_a =
      cycle_a[static_cast<std::size_t>((edge_a + 1) % nverts_a)];
  int edge_b = -1;
  for (int k = 0; k < nverts_b; ++k) {
    const int next = (k + 1) % nverts_b;
    if (cycle_b[static_cast<std::size_t>(k)] == edge_end_a &&
        cycle_b[static_cast<std::size_t>(next)] == edge_start_a) {
      edge_b = k;
      break;
    }
  }
  if (edge_b < 0) {
    return false;
  }

  union_cycle.clear();
  union_cycle.reserve(static_cast<std::size_t>(nverts_a + nverts_b - 2));
  union_cycle.push_back(edge_start_a);

  int k = (edge_b + 2) % nverts_b;
  while (cycle_b[static_cast<std::size_t>(k)] != edge_end_a) {
    union_cycle.push_back(cycle_b[static_cast<std::size_t>(k)]);
    k = (k + 1) % nverts_b;
  }
  union_cycle.push_back(edge_end_a);

  k = (edge_a + 2) % nverts_a;
  while (cycle_a[static_cast<std::size_t>(k)] != edge_start_a) {
    union_cycle.push_back(cycle_a[static_cast<std::size_t>(k)]);
    k = (k + 1) % nverts_a;
  }

  if (union_cycle.size() !=
      static_cast<std::size_t>(nverts_a + nverts_b - 2)) {
    return false;
  }

  const auto first_a =
      std::find(union_cycle.begin(), union_cycle.end(), cycle_a[0]);
  if (first_a == union_cycle.end()) {
    return false;
  }
  std::rotate(union_cycle.begin(), first_a, union_cycle.end());
  return true;
}

double revolved_polygon_volume(const std::vector<int>& cycle,
                               const double* node_r,
                               const double* node_z) {
  double sum = 0.0;
  for (std::size_t k = 0; k < cycle.size(); ++k) {
    const std::size_t next = (k + 1U) % cycle.size();
    const int node_a = cycle[k];
    const int node_b = cycle[next];
    const double r_a = node_r[node_a];
    const double z_a = node_z[node_a];
    const double r_b = node_r[node_b];
    const double z_b = node_z[node_b];
    sum += (r_a * z_b - r_b * z_a) * (r_a + r_b);
  }
  return (kPi / 3.0) * sum;
}

bool axis_merge_is_admissible(
    const std::array<int, kCellStride>& cycle_a,
    const int nverts_a,
    const std::array<int, kCellStride>& cycle_b,
    const int nverts_b,
    const std::vector<int>& union_cycle,
    const double* node_r) {
  bool has_axis_node = false;
  for (int k = 0; k < nverts_a; ++k) {
    has_axis_node =
        has_axis_node || node_r[cycle_a[static_cast<std::size_t>(k)]] == 0.0;
  }
  for (int k = 0; k < nverts_b; ++k) {
    has_axis_node =
        has_axis_node || node_r[cycle_b[static_cast<std::size_t>(k)]] == 0.0;
  }
  if (!has_axis_node) {
    return true;
  }

  for (const int node : union_cycle) {
    if (node_r[node] < 0.0) {
      return false;
    }
  }
  for (int k = 0; k < nverts_a; ++k) {
    const int node = cycle_a[static_cast<std::size_t>(k)];
    if (node_r[node] == 0.0 &&
        std::find(union_cycle.begin(), union_cycle.end(), node) ==
            union_cycle.end()) {
      return false;
    }
  }
  for (int k = 0; k < nverts_b; ++k) {
    const int node = cycle_b[static_cast<std::size_t>(k)];
    if (node_r[node] == 0.0 &&
        std::find(union_cycle.begin(), union_cycle.end(), node) ==
            union_cycle.end()) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool topology_coarsen_pair(
    MultiBlockTopology& topology,
    std::vector<std::uint8_t>& cell_nverts,
    const double* node_r,
    const double* node_z,
    const int n_nodes,
    const int cell_a,
    const int cell_b,
    double* mass,
    double* mom_r,
    double* mom_z,
    double* e_int,
    double* vol,
    CellLineageRecord& lineage_out) {
  const std::size_t n_cells = cell_nverts.size();
  if (node_r == nullptr || node_z == nullptr || mass == nullptr ||
      mom_r == nullptr || mom_z == nullptr || e_int == nullptr ||
      vol == nullptr || n_nodes <= 0 || cell_a == cell_b ||
      cell_a < 0 || cell_b < 0 ||
      static_cast<std::size_t>(cell_a) >= n_cells ||
      static_cast<std::size_t>(cell_b) >= n_cells ||
      topology.cell_id_stable.size() < n_cells) {
    return false;
  }

  std::array<int, kCellStride> cycle_a;
  std::array<int, kCellStride> cycle_b;
  std::size_t offset_a = 0;
  std::size_t offset_b = 0;
  if (!read_cycle(topology, cell_nverts, cell_a, n_nodes, cycle_a, offset_a) ||
      !read_cycle(topology, cell_nverts, cell_b, n_nodes, cycle_b, offset_b)) {
    return false;
  }

  const int nverts_a =
      static_cast<int>(cell_nverts[static_cast<std::size_t>(cell_a)]);
  const int nverts_b =
      static_cast<int>(cell_nverts[static_cast<std::size_t>(cell_b)]);
  std::vector<int> union_cycle;
  if (!merge_cycles(cycle_a, nverts_a, cycle_b, nverts_b, union_cycle) ||
      !axis_merge_is_admissible(cycle_a, nverts_a, cycle_b, nverts_b,
                                union_cycle, node_r)) {
    return false;
  }

  const double union_volume =
      revolved_polygon_volume(union_cycle, node_r, node_z);

  CellLineageRecord lineage{};
  lineage.cell_a = cell_a;
  lineage.cell_b = cell_b;
  lineage.id_stable_a =
      topology.cell_id_stable[static_cast<std::size_t>(cell_a)];
  lineage.id_stable_b =
      topology.cell_id_stable[static_cast<std::size_t>(cell_b)];
  lineage.cycle_a = cycle_a;
  lineage.cycle_b = cycle_b;
  lineage.nverts_a = static_cast<std::uint8_t>(nverts_a);
  lineage.nverts_b = static_cast<std::uint8_t>(nverts_b);
  lineage.mass_b = mass[cell_b];
  lineage.mom_r_b = mom_r[cell_b];
  lineage.mom_z_b = mom_z[cell_b];
  lineage.e_int_b = e_int[cell_b];
  lineage.vol_a = vol[cell_a];
  lineage.vol_b = vol[cell_b];
  lineage.mass_a = mass[cell_a];
  lineage.mom_r_a = mom_r[cell_a];
  lineage.mom_z_a = mom_z[cell_a];
  lineage.e_int_a = e_int[cell_a];

  for (int k = 0; k < kCellStride; ++k) {
    topology.cell_node_csr_indices[offset_a + static_cast<std::size_t>(k)] =
        k < static_cast<int>(union_cycle.size())
            ? union_cycle[static_cast<std::size_t>(k)]
            : -1;
    topology.cell_node_csr_indices[offset_b + static_cast<std::size_t>(k)] = -1;
  }
  cell_nverts[static_cast<std::size_t>(cell_a)] =
      static_cast<std::uint8_t>(union_cycle.size());
  cell_nverts[static_cast<std::size_t>(cell_b)] = 0;

  mass[cell_a] += mass[cell_b];
  mom_r[cell_a] += mom_r[cell_b];
  mom_z[cell_a] += mom_z[cell_b];
  e_int[cell_a] += e_int[cell_b];
  vol[cell_a] = union_volume;

  mass[cell_b] = 0.0;
  mom_r[cell_b] = 0.0;
  mom_z[cell_b] = 0.0;
  e_int[cell_b] = 0.0;
  vol[cell_b] = 0.0;
  lineage_out = lineage;
  return true;
}

bool topology_refine_from_lineage(
    MultiBlockTopology& topology,
    std::vector<std::uint8_t>& cell_nverts,
    double* mass,
    double* mom_r,
    double* mom_z,
    double* e_int,
    double* vol,
    const CellLineageRecord& lineage) {
  const std::size_t n_cells = cell_nverts.size();
  if (mass == nullptr || mom_r == nullptr || mom_z == nullptr ||
      e_int == nullptr || vol == nullptr || lineage.cell_a == lineage.cell_b ||
      lineage.nverts_a < 3 || lineage.nverts_a > kCellStride ||
      lineage.nverts_b < 3 || lineage.nverts_b > kCellStride) {
    return false;
  }

  std::size_t offset_a = 0;
  std::size_t offset_b = 0;
  if (!cell_row_offset(topology, lineage.cell_a, n_cells, offset_a) ||
      !cell_row_offset(topology, lineage.cell_b, n_cells, offset_b)) {
    return false;
  }

  for (int k = 0; k < kCellStride; ++k) {
    topology.cell_node_csr_indices[offset_a + static_cast<std::size_t>(k)] =
        lineage.cycle_a[static_cast<std::size_t>(k)];
    topology.cell_node_csr_indices[offset_b + static_cast<std::size_t>(k)] =
        lineage.cycle_b[static_cast<std::size_t>(k)];
  }
  cell_nverts[static_cast<std::size_t>(lineage.cell_a)] = lineage.nverts_a;
  cell_nverts[static_cast<std::size_t>(lineage.cell_b)] = lineage.nverts_b;

  mass[lineage.cell_a] = lineage.mass_a;
  mass[lineage.cell_b] = lineage.mass_b;
  mom_r[lineage.cell_a] = lineage.mom_r_a;
  mom_r[lineage.cell_b] = lineage.mom_r_b;
  mom_z[lineage.cell_a] = lineage.mom_z_a;
  mom_z[lineage.cell_b] = lineage.mom_z_b;
  e_int[lineage.cell_a] = lineage.e_int_a;
  e_int[lineage.cell_b] = lineage.e_int_b;
  vol[lineage.cell_a] = lineage.vol_a;
  vol[lineage.cell_b] = lineage.vol_b;
  return true;
}

}  // namespace tenryu::mesh
