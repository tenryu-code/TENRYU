#include "hydro/axis_projection.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::axis_projection {
namespace {

struct MonitoredCell {
  int cell = -1;
  int nverts = 0;
  bool active_eligible = true;
  bool is_transition = false;
  std::array<int, mesh::kMeshTopoCellStorageSlots> nodes{};
  // Retained for diagnostics; projection quality uses the current cell mean.
  std::array<double, mesh::kMeshTopoCellStorageSlots> initial_corner_area{};
};

struct PatchCell {
  int cell = -1;
  int nverts = 0;
  std::array<int, mesh::kMeshTopoCellStorageSlots> nodes{};
  // Retained for diagnostics; projection floors use the current cell mean.
  std::array<double, mesh::kMeshTopoCellStorageSlots> initial_corner_area{};
  double frozen_floor = 0.0;
};

struct IncidentCorner {
  int cell = -1;
  int corner = -1;
};

struct AreaConstraint {
  int cell = -1;
  int corner = -1;
  double predicted_area = 0.0;
  double floor = 0.0;
  std::array<int, 3> nodes{};
  std::array<int, 3> patch_nodes{};
  std::array<double, 3> gradient_r{};
  std::array<double, 3> gradient_z{};
};

struct RuntimeState {
  bool initialized = false;
  bool initial_geometry_captured = false;
  bool incident_table_initialized = false;
  bool engaged = false;
  std::uint64_t steps_below_threshold = 0;
  std::uint64_t applied_projections = 0;
  std::uint64_t rejected_projections = 0;
  std::uint64_t rejected_energy_projections = 0;
  std::uint64_t skipped_unowned_patches = 0;
  double E_axis_projection = 0.0;
  double global_min_q_pred = std::numeric_limits<double>::infinity();
  std::vector<MonitoredCell> monitored_cells;
  std::vector<std::vector<IncidentCorner>> node_incident_corners;
  std::vector<double> x_r_initial_host;
  std::vector<double> x_z_initial_host;
  std::vector<double> x_r_old_host;
  std::vector<double> x_z_old_host;
  std::vector<double> pos_v_r_host;
  std::vector<double> pos_v_z_host;
  std::vector<double> corner_mass_host;
  std::vector<double> cell_mass_host;
  std::vector<double> ion_energy_old_host;
};

RuntimeState* g_runtime = nullptr;

bool applicable(const core::State& state, const core::Config& cfg) {
  return cfg.main.dimension == "2D_RZ" &&
         mesh::mesh_topo_is_multiblock(cfg.mesh) &&
         state.mesh.topo.multiblock.has_value() &&
         mesh::mesh_topo_polar_tier_family(cfg.mesh);
}

void initialize_monitored_cells(const core::State& state,
                                const core::Config& cfg,
                                RuntimeState& runtime) {
  const auto& topology = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(
      state.mesh.topo.node_flags.size() ==
          static_cast<std::size_t>(state.mesh.topo.n_nodes),
      "axis projection shadow monitor requires node flags");
  std::vector<int> cell_ids;
  std::vector<int> active_cell_ids;
  std::vector<int> transition_cell_ids;
  for (const auto& block : topology.blocks) {
    if (block.role != mesh::BlockRole::POLAR_TIER) {
      continue;
    }
    for (int i = 0; i < block.n_i_cells; ++i) {
      const int row_begin = block.cell_begin + i * block.n_j_cells;
      cell_ids.push_back(row_begin);
      cell_ids.push_back(row_begin + block.n_j_cells - 1);
    }
  }
  active_cell_ids = cell_ids;
  for (const auto& block : topology.blocks) {
    if (block.role != mesh::BlockRole::TRANSITION_BELT ||
        block.n_i_cells != 1) {
      continue;
    }
    for (int cell = block.cell_begin;
         cell < block.cell_begin + block.cell_count; ++cell) {
      const mesh::MeshTopoCellCornerNodesN corners =
          mesh::mesh_topo_cell_corner_nodes_n(
              state.mesh.topo, state.mesh.cell_nverts, cell, cfg.mesh);
      bool touches_axis = false;
      for (int k = 0; k < corners.count; ++k) {
        const int node = corners.values[static_cast<std::size_t>(k)];
        if ((state.mesh.topo.node_flags[static_cast<std::size_t>(node)] &
             mesh::NODE_POLE_AXIS) != 0U) {
          touches_axis = true;
          break;
        }
      }
      if (touches_axis) {
        cell_ids.push_back(cell);
        active_cell_ids.push_back(cell);
        transition_cell_ids.push_back(cell);
      }
    }
  }
  std::sort(active_cell_ids.begin(), active_cell_ids.end());
  active_cell_ids.erase(
      std::unique(active_cell_ids.begin(), active_cell_ids.end()),
      active_cell_ids.end());
  std::sort(transition_cell_ids.begin(), transition_cell_ids.end());
  transition_cell_ids.erase(
      std::unique(transition_cell_ids.begin(), transition_cell_ids.end()),
      transition_cell_ids.end());
  std::sort(cell_ids.begin(), cell_ids.end());
  cell_ids.erase(std::unique(cell_ids.begin(), cell_ids.end()),
                 cell_ids.end());

  runtime.monitored_cells.reserve(cell_ids.size());
  for (const int cell : cell_ids) {
    const mesh::MeshTopoCellCornerNodesN corners =
        mesh::mesh_topo_cell_corner_nodes_n(
            state.mesh.topo, state.mesh.cell_nverts, cell, cfg.mesh);
    TENRYU_ASSERT(
        corners.count == 3 || corners.count == 4,
        "axis projection shadow monitor requires three- or four-corner cells");
    MonitoredCell monitored;
    monitored.cell = cell;
    monitored.nverts = corners.count;
    monitored.active_eligible =
        std::binary_search(active_cell_ids.begin(),
                           active_cell_ids.end(),
                           cell);
    monitored.is_transition =
        std::binary_search(transition_cell_ids.begin(),
                           transition_cell_ids.end(),
                           cell);
    for (int k = 0; k < corners.count; ++k) {
      monitored.nodes[static_cast<std::size_t>(k)] =
          corners.values[static_cast<std::size_t>(k)];
    }
    runtime.monitored_cells.push_back(monitored);
  }
  runtime.initialized = true;
}

void copy_device_field(const double* source,
                       const std::size_t size,
                       std::vector<double>& destination,
                       const char* message) {
  TENRYU_ASSERT(source != nullptr || size == 0,
                "axis projection shadow monitor received a null field");
  destination.resize(size);
  if (size == 0) {
    return;
  }
  const cudaError_t error =
      cudaMemcpy(destination.data(),
                 source,
                 size * sizeof(double),
                 cudaMemcpyDeviceToHost);
  TENRYU_ASSERT(error == cudaSuccess, message);
}

void copy_host_field(const std::vector<double>& source,
                     double* destination,
                     const char* message) {
  TENRYU_ASSERT(destination != nullptr || source.empty(),
                "axis projection received a null destination field");
  if (source.empty()) {
    return;
  }
  const cudaError_t error =
      cudaMemcpy(destination,
                 source.data(),
                 source.size() * sizeof(double),
                 cudaMemcpyHostToDevice);
  TENRYU_ASSERT(error == cudaSuccess, message);
}

double corner_area(const std::vector<double>& x_r,
                   const std::vector<double>& x_z,
                   const MonitoredCell& monitored,
                   const int corner) {
  const int previous = (corner + monitored.nverts - 1) % monitored.nverts;
  const int next = (corner + 1) % monitored.nverts;
  const int node = monitored.nodes[static_cast<std::size_t>(corner)];
  const int node_previous =
      monitored.nodes[static_cast<std::size_t>(previous)];
  const int node_next = monitored.nodes[static_cast<std::size_t>(next)];
  const double next_r =
      x_r[static_cast<std::size_t>(node_next)] -
      x_r[static_cast<std::size_t>(node)];
  const double next_z =
      x_z[static_cast<std::size_t>(node_next)] -
      x_z[static_cast<std::size_t>(node)];
  const double previous_r =
      x_r[static_cast<std::size_t>(node_previous)] -
      x_r[static_cast<std::size_t>(node)];
  const double previous_z =
      x_z[static_cast<std::size_t>(node_previous)] -
      x_z[static_cast<std::size_t>(node)];
  return next_r * previous_z - previous_r * next_z;
}

double corner_area(const std::vector<double>& x_r,
                   const std::vector<double>& x_z,
                   const PatchCell& patch_cell,
                   const int corner) {
  const int previous = (corner + patch_cell.nverts - 1) % patch_cell.nverts;
  const int next = (corner + 1) % patch_cell.nverts;
  const int node = patch_cell.nodes[static_cast<std::size_t>(corner)];
  const int node_previous =
      patch_cell.nodes[static_cast<std::size_t>(previous)];
  const int node_next = patch_cell.nodes[static_cast<std::size_t>(next)];
  const double next_r =
      x_r[static_cast<std::size_t>(node_next)] -
      x_r[static_cast<std::size_t>(node)];
  const double next_z =
      x_z[static_cast<std::size_t>(node_next)] -
      x_z[static_cast<std::size_t>(node)];
  const double previous_r =
      x_r[static_cast<std::size_t>(node_previous)] -
      x_r[static_cast<std::size_t>(node)];
  const double previous_z =
      x_z[static_cast<std::size_t>(node_previous)] -
      x_z[static_cast<std::size_t>(node)];
  return next_r * previous_z - previous_r * next_z;
}

double predicted_corner_area(const RuntimeState& runtime,
                             const MonitoredCell& monitored,
                             const int corner,
                             const double dt) {
  const int previous = (corner + monitored.nverts - 1) % monitored.nverts;
  const int next = (corner + 1) % monitored.nverts;
  const int node = monitored.nodes[static_cast<std::size_t>(corner)];
  const int node_previous =
      monitored.nodes[static_cast<std::size_t>(previous)];
  const int node_next = monitored.nodes[static_cast<std::size_t>(next)];
  const auto predicted_r = [&](const int n) {
    return runtime.x_r_old_host[static_cast<std::size_t>(n)] +
           dt * runtime.pos_v_r_host[static_cast<std::size_t>(n)];
  };
  const auto predicted_z = [&](const int n) {
    return runtime.x_z_old_host[static_cast<std::size_t>(n)] +
           dt * runtime.pos_v_z_host[static_cast<std::size_t>(n)];
  };
  const double node_r = predicted_r(node);
  const double node_z = predicted_z(node);
  const double next_r = predicted_r(node_next) - node_r;
  const double next_z = predicted_z(node_next) - node_z;
  const double previous_r = predicted_r(node_previous) - node_r;
  const double previous_z = predicted_z(node_previous) - node_z;
  return next_r * previous_z - previous_r * next_z;
}

void capture_initial_geometry(RuntimeState& runtime) {
  for (auto& monitored : runtime.monitored_cells) {
    for (int k = 0; k < monitored.nverts; ++k) {
      const double area =
          corner_area(runtime.x_r_initial_host,
                      runtime.x_z_initial_host,
                      monitored,
                      k);
      TENRYU_ASSERT(
          std::isfinite(area) && area > 0.0,
          "axis projection shadow monitor requires positive initial corner area");
      monitored.initial_corner_area[static_cast<std::size_t>(k)] = area;
    }
  }
  runtime.initial_geometry_captured = true;
}

bool build_patch_cells(const core::State& state,
                       const core::Config& cfg,
                       const RuntimeState& runtime,
                       const int argmin_cell,
                       std::vector<PatchCell>& patch_cells) {
  const auto& topology = *state.mesh.topo.multiblock;
  const mesh::BlockInfo* anchor_block = nullptr;
  int anchor_row = -1;
  bool lower_side = false;
  for (const auto& block : topology.blocks) {
    if (block.role != mesh::BlockRole::POLAR_TIER ||
        block.n_i_cells <= 0 || block.n_j_cells <= 0) {
      continue;
    }
    const int block_cells = block.n_i_cells * block.n_j_cells;
    if (argmin_cell < block.cell_begin ||
        argmin_cell >= block.cell_begin + block_cells) {
      continue;
    }
    const int local = argmin_cell - block.cell_begin;
    anchor_row = local / block.n_j_cells;
    const int column = local % block.n_j_cells;
    if (column != 0 && column != block.n_j_cells - 1) {
      return false;
    }
    lower_side = column == 0;
    anchor_block = &block;
    break;
  }
  if (anchor_block == nullptr) {
    return false;
  }

  const int row_begin =
      std::max(0, anchor_row - cfg.numerics.hydro.axis_projection.patch_halfwidth);
  const int row_end =
      std::min(anchor_block->n_i_cells - 1,
               anchor_row +
                   cfg.numerics.hydro.axis_projection.patch_halfwidth);
  std::vector<int> columns;
  if (anchor_block->n_j_cells == 1) {
    columns.push_back(0);
  } else if (lower_side) {
    columns.push_back(0);
    columns.push_back(1);
  } else {
    columns.push_back(anchor_block->n_j_cells - 2);
    columns.push_back(anchor_block->n_j_cells - 1);
  }

  std::vector<int> cell_ids;
  for (int row = row_begin; row <= row_end; ++row) {
    for (const int column : columns) {
      cell_ids.push_back(anchor_block->cell_begin +
                         row * anchor_block->n_j_cells + column);
    }
  }
  std::sort(cell_ids.begin(), cell_ids.end());
  cell_ids.erase(std::unique(cell_ids.begin(), cell_ids.end()),
                 cell_ids.end());

  patch_cells.clear();
  patch_cells.reserve(cell_ids.size());
  for (const int cell : cell_ids) {
    const mesh::MeshTopoCellCornerNodesN corners =
        mesh::mesh_topo_cell_corner_nodes_n(
            state.mesh.topo, state.mesh.cell_nverts, cell, cfg.mesh);
    TENRYU_ASSERT(
        corners.count == mesh::kMeshTopoCellStorageSlots,
        "active axis projection requires four-corner POLAR_TIER patch cells");
    PatchCell patch_cell;
    patch_cell.cell = cell;
    patch_cell.nverts = corners.count;
    for (int k = 0; k < corners.count; ++k) {
      patch_cell.nodes[static_cast<std::size_t>(k)] =
          corners.values[static_cast<std::size_t>(k)];
    }
    for (int k = 0; k < patch_cell.nverts; ++k) {
      const double area =
          corner_area(runtime.x_r_initial_host,
                      runtime.x_z_initial_host,
                      patch_cell,
                      k);
      TENRYU_ASSERT(
          std::isfinite(area) && area > 0.0,
          "active axis projection requires positive initial patch corner area");
      patch_cell.initial_corner_area[static_cast<std::size_t>(k)] = area;
    }
    patch_cells.push_back(patch_cell);
  }
  return true;
}

void initialize_incident_table(const core::State& state,
                               const core::Config& cfg,
                               RuntimeState& runtime) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  TENRYU_ASSERT(state.mesh.topo.n_cells == n_cells &&
                    state.mesh.topo.n_nodes == n_nodes,
                "active axis projection topology/state size mismatch");
  runtime.node_incident_corners.assign(
      static_cast<std::size_t>(n_nodes), {});
  for (int cell = 0; cell < n_cells; ++cell) {
    const mesh::MeshTopoCellCornerNodesN corners =
        mesh::mesh_topo_cell_corner_nodes_n(
            state.mesh.topo, state.mesh.cell_nverts, cell, cfg.mesh);
    for (int k = 0; k < corners.count; ++k) {
      const int node = corners.values[static_cast<std::size_t>(k)];
      TENRYU_ASSERT(node >= 0 && node < n_nodes,
                    "active axis projection incident node out of range");
      runtime.node_incident_corners[static_cast<std::size_t>(node)].push_back(
          IncidentCorner{cell, k});
    }
  }
  runtime.incident_table_initialized = true;
}

bool is_single_row_transition_cell(const core::State& state,
                                   const int cell) {
  const auto& topology = *state.mesh.topo.multiblock;
  for (const auto& block : topology.blocks) {
    if (block.role == mesh::BlockRole::TRANSITION_BELT &&
        block.n_i_cells == 1 &&
        cell >= block.cell_begin &&
        cell < block.cell_begin + block.cell_count) {
      return true;
    }
  }
  return false;
}

bool build_node_neighborhood_patch_cells(
    const core::State& state,
    const core::Config& cfg,
    const RuntimeState& runtime,
    const int argmin_cell,
    std::vector<PatchCell>& patch_cells) {
  if (!is_single_row_transition_cell(state, argmin_cell)) {
    return false;
  }
  TENRYU_ASSERT(
      runtime.incident_table_initialized,
      "active axis projection node-neighborhood patch requires incident table");

  const mesh::MeshTopoCellCornerNodesN argmin_corners =
      mesh::mesh_topo_cell_corner_nodes_n(
          state.mesh.topo, state.mesh.cell_nverts, argmin_cell, cfg.mesh);
  TENRYU_ASSERT(
      argmin_corners.count == 3 || argmin_corners.count == 4,
      "active axis projection transition argmin requires three or four corners");
  std::vector<int> cell_ids;
  for (int k = 0; k < argmin_corners.count; ++k) {
    const int node = argmin_corners.values[static_cast<std::size_t>(k)];
    TENRYU_ASSERT(
        node >= 0 &&
            node < static_cast<int>(runtime.node_incident_corners.size()),
        "active axis projection transition argmin node out of range");
    for (const IncidentCorner& incident :
         runtime.node_incident_corners[static_cast<std::size_t>(node)]) {
      cell_ids.push_back(incident.cell);
    }
  }
  std::sort(cell_ids.begin(), cell_ids.end());
  cell_ids.erase(std::unique(cell_ids.begin(), cell_ids.end()),
                 cell_ids.end());

  patch_cells.clear();
  patch_cells.reserve(cell_ids.size());
  for (const int cell : cell_ids) {
    const mesh::MeshTopoCellCornerNodesN corners =
        mesh::mesh_topo_cell_corner_nodes_n(
            state.mesh.topo, state.mesh.cell_nverts, cell, cfg.mesh);
    TENRYU_ASSERT(
        corners.count == 3 || corners.count == 4,
        "active axis projection node-neighborhood patch requires three- or "
        "four-corner cells");
    PatchCell patch_cell;
    patch_cell.cell = cell;
    patch_cell.nverts = corners.count;
    for (int k = 0; k < corners.count; ++k) {
      patch_cell.nodes[static_cast<std::size_t>(k)] =
          corners.values[static_cast<std::size_t>(k)];
    }
    for (int k = 0; k < patch_cell.nverts; ++k) {
      const double area =
          corner_area(runtime.x_r_initial_host,
                      runtime.x_z_initial_host,
                      patch_cell,
                      k);
      TENRYU_ASSERT(
          std::isfinite(area) && area > 0.0,
          "active axis projection requires positive initial patch corner area");
      patch_cell.initial_corner_area[static_cast<std::size_t>(k)] = area;
    }
    patch_cells.push_back(patch_cell);
  }
  return true;
}

double corrected_predicted_corner_area(
    const RuntimeState& runtime,
    const PatchCell& patch_cell,
    const int corner,
    const double dt,
    const std::vector<int>& node_to_patch,
    const std::vector<double>& du_r,
    const std::vector<double>& du_z) {
  const int previous = (corner + patch_cell.nverts - 1) % patch_cell.nverts;
  const int next = (corner + 1) % patch_cell.nverts;
  const int node = patch_cell.nodes[static_cast<std::size_t>(corner)];
  const int node_previous =
      patch_cell.nodes[static_cast<std::size_t>(previous)];
  const int node_next = patch_cell.nodes[static_cast<std::size_t>(next)];
  const auto corrected_r = [&](const int n) {
    const int patch_node = node_to_patch[static_cast<std::size_t>(n)];
    TENRYU_ASSERT(patch_node >= 0,
                  "active axis projection patch-node lookup failed");
    return runtime.x_r_old_host[static_cast<std::size_t>(n)] +
           dt * (runtime.pos_v_r_host[static_cast<std::size_t>(n)] +
                 du_r[static_cast<std::size_t>(patch_node)]);
  };
  const auto corrected_z = [&](const int n) {
    const int patch_node = node_to_patch[static_cast<std::size_t>(n)];
    TENRYU_ASSERT(patch_node >= 0,
                  "active axis projection patch-node lookup failed");
    return runtime.x_z_old_host[static_cast<std::size_t>(n)] +
           dt * (runtime.pos_v_z_host[static_cast<std::size_t>(n)] +
                 du_z[static_cast<std::size_t>(patch_node)]);
  };
  const double node_r = corrected_r(node);
  const double node_z = corrected_z(node);
  const double next_r = corrected_r(node_next) - node_r;
  const double next_z = corrected_z(node_next) - node_z;
  const double previous_r = corrected_r(node_previous) - node_r;
  const double previous_z = corrected_z(node_previous) - node_z;
  return next_r * previous_z - previous_r * next_z;
}

std::vector<AreaConstraint> build_constraints(
    const RuntimeState& runtime,
    const core::Config& cfg,
    std::vector<PatchCell>& patch_cells,
    const std::vector<int>& node_to_patch,
    const std::vector<double>& zero_du_r,
    const std::vector<double>& zero_du_z,
    const double dt) {
  std::vector<AreaConstraint> constraints;
  for (auto& patch_cell : patch_cells) {
    std::array<double, mesh::kMeshTopoCellStorageSlots> predicted_areas{};
    double predicted_area_sum = 0.0;
    for (int k = 0; k < patch_cell.nverts; ++k) {
      predicted_areas[static_cast<std::size_t>(k)] =
          corrected_predicted_corner_area(runtime,
                                          patch_cell,
                                          k,
                                          dt,
                                          node_to_patch,
                                          zero_du_r,
                                          zero_du_z);
      predicted_area_sum += predicted_areas[static_cast<std::size_t>(k)];
    }
    const double mean_pred =
        predicted_area_sum / static_cast<double>(patch_cell.nverts);
    // Freeze the current cell mean for all corner floors in this assembly.
    patch_cell.frozen_floor =
        cfg.numerics.hydro.axis_projection.q_floor * mean_pred;
    for (int k = 0; k < patch_cell.nverts; ++k) {
      const double floor = patch_cell.frozen_floor;
      const double predicted =
          predicted_areas[static_cast<std::size_t>(k)];
      if (std::isfinite(predicted) && predicted >= floor) {
        continue;
      }

      const int previous = (k + patch_cell.nverts - 1) % patch_cell.nverts;
      const int next = (k + 1) % patch_cell.nverts;
      const int node = patch_cell.nodes[static_cast<std::size_t>(k)];
      const int node_next =
          patch_cell.nodes[static_cast<std::size_t>(next)];
      const int node_previous =
          patch_cell.nodes[static_cast<std::size_t>(previous)];
      const auto predicted_r = [&](const int n) {
        return runtime.x_r_old_host[static_cast<std::size_t>(n)] +
               dt * runtime.pos_v_r_host[static_cast<std::size_t>(n)];
      };
      const auto predicted_z = [&](const int n) {
        return runtime.x_z_old_host[static_cast<std::size_t>(n)] +
               dt * runtime.pos_v_z_host[static_cast<std::size_t>(n)];
      };
      const double r0 = predicted_r(node);
      const double z0 = predicted_z(node);
      const double rn = predicted_r(node_next);
      const double zn = predicted_z(node_next);
      const double rp = predicted_r(node_previous);
      const double zp = predicted_z(node_previous);

      AreaConstraint constraint;
      constraint.cell = patch_cell.cell;
      constraint.corner = k;
      constraint.predicted_area = predicted;
      constraint.floor = floor;
      constraint.nodes = {{node, node_next, node_previous}};
      for (int p = 0; p < 3; ++p) {
        constraint.patch_nodes[static_cast<std::size_t>(p)] =
            node_to_patch[static_cast<std::size_t>(
                constraint.nodes[static_cast<std::size_t>(p)])];
        TENRYU_ASSERT(
            constraint.patch_nodes[static_cast<std::size_t>(p)] >= 0,
            "active axis projection constraint node is outside patch");
      }
      // For A=(rn-r0)(zp-z0)-(rp-r0)(zn-z0):
      // dA/d(r0,z0)=(zn-zp,rp-rn),
      // dA/d(rn,zn)=(zp-z0,r0-rp),
      // dA/d(rp,zp)=(z0-zn,rn-r0).
      // The three r gradients and the three z gradients each sum to zero.
      constraint.gradient_r = {{
          zn - zp,
          zp - z0,
          z0 - zn,
      }};
      constraint.gradient_z = {{
          rp - rn,
          r0 - rp,
          rn - r0,
      }};
      constraints.push_back(constraint);
    }
  }
  std::sort(constraints.begin(),
            constraints.end(),
            [](const AreaConstraint& lhs, const AreaConstraint& rhs) {
              if (lhs.cell != rhs.cell) {
                return lhs.cell < rhs.cell;
              }
              return lhs.corner < rhs.corner;
            });
  return constraints;
}

struct ProjectionSolveResult {
  bool converged = false;
  int truncated_constraints = 0;
};

ProjectionSolveResult project_constraints(
    const core::State& state,
    const std::vector<AreaConstraint>& constraints,
    const std::vector<double>& nodal_mass,
    const double dt,
    std::vector<double>& du_r,
    std::vector<double>& du_z) {
  TENRYU_ASSERT(state.mesh.topo.node_flags.size() == state.x_r.size(),
                "active axis projection requires node flags");
  constexpr int kMaxCandidates = 12;
  constexpr int kMaxIterations = 8;
  constexpr double kPivotFloor = 1.0e-300;
  constexpr double kRelativeSlack = 1.0e-12;

  ProjectionSolveResult result;
  std::vector<int> candidate_indices(constraints.size());
  std::vector<double> all_residuals(constraints.size(), 0.0);
  for (std::size_t i = 0; i < constraints.size(); ++i) {
    candidate_indices[i] = static_cast<int>(i);
    all_residuals[i] =
        (constraints[i].floor - constraints[i].predicted_area) / dt;
    if (!std::isfinite(all_residuals[i]) || all_residuals[i] <= 0.0) {
      return result;
    }
  }
  std::sort(candidate_indices.begin(),
            candidate_indices.end(),
            [&](const int lhs, const int rhs) {
              if (all_residuals[static_cast<std::size_t>(lhs)] !=
                  all_residuals[static_cast<std::size_t>(rhs)]) {
                return all_residuals[static_cast<std::size_t>(lhs)] >
                       all_residuals[static_cast<std::size_t>(rhs)];
              }
              return lhs < rhs;
            });
  if (candidate_indices.size() >
      static_cast<std::size_t>(kMaxCandidates)) {
    result.truncated_constraints =
        static_cast<int>(candidate_indices.size()) - kMaxCandidates;
    candidate_indices.resize(static_cast<std::size_t>(kMaxCandidates));
  }
  std::sort(candidate_indices.begin(), candidate_indices.end());

  const int candidate_count = static_cast<int>(candidate_indices.size());
  std::array<double, kMaxCandidates> b{};
  std::array<std::array<int, 3>, kMaxCandidates> patch_node{};
  std::array<std::array<double, 3>, kMaxCandidates> gradient_r{};
  std::array<std::array<double, 3>, kMaxCandidates> gradient_z{};
  for (int i = 0; i < candidate_count; ++i) {
    const int constraint_index =
        candidate_indices[static_cast<std::size_t>(i)];
    const auto& constraint =
        constraints[static_cast<std::size_t>(constraint_index)];
    b[static_cast<std::size_t>(i)] =
        all_residuals[static_cast<std::size_t>(constraint_index)];
    for (int p = 0; p < 3; ++p) {
      const int node = constraint.nodes[static_cast<std::size_t>(p)];
      patch_node[static_cast<std::size_t>(i)]
                [static_cast<std::size_t>(p)] =
          constraint.patch_nodes[static_cast<std::size_t>(p)];
      const bool on_axis =
          (state.mesh.topo.node_flags[static_cast<std::size_t>(node)] &
           mesh::NODE_POLE_AXIS) != 0U;
      gradient_r[static_cast<std::size_t>(i)]
                [static_cast<std::size_t>(p)] =
          on_axis ? 0.0
                  : constraint.gradient_r[static_cast<std::size_t>(p)];
      gradient_z[static_cast<std::size_t>(i)]
                [static_cast<std::size_t>(p)] =
          constraint.gradient_z[static_cast<std::size_t>(p)];
    }
  }

  std::array<std::array<double, kMaxCandidates>, kMaxCandidates> S{};
  for (int i = 0; i < candidate_count; ++i) {
    for (int j = i; j < candidate_count; ++j) {
      double value = 0.0;
      for (int p = 0; p < 3; ++p) {
        for (int q = 0; q < 3; ++q) {
          if (patch_node[static_cast<std::size_t>(i)]
                        [static_cast<std::size_t>(p)] !=
              patch_node[static_cast<std::size_t>(j)]
                        [static_cast<std::size_t>(q)]) {
            continue;
          }
          const int node =
              patch_node[static_cast<std::size_t>(i)]
                        [static_cast<std::size_t>(p)];
          value +=
              (gradient_r[static_cast<std::size_t>(i)]
                         [static_cast<std::size_t>(p)] *
                   gradient_r[static_cast<std::size_t>(j)]
                             [static_cast<std::size_t>(q)] +
               gradient_z[static_cast<std::size_t>(i)]
                         [static_cast<std::size_t>(p)] *
                   gradient_z[static_cast<std::size_t>(j)]
                             [static_cast<std::size_t>(q)]) /
              nodal_mass[static_cast<std::size_t>(node)];
        }
      }
      S[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
          value;
      S[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] =
          value;
    }
  }

  std::array<bool, kMaxCandidates> active{};
  for (int i = 0; i < candidate_count; ++i) {
    active[static_cast<std::size_t>(i)] = true;
  }
  for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
    std::array<int, kMaxCandidates> active_indices{};
    int active_count = 0;
    for (int i = 0; i < candidate_count; ++i) {
      if (active[static_cast<std::size_t>(i)]) {
        active_indices[static_cast<std::size_t>(active_count++)] = i;
      }
    }

    std::array<std::array<double, kMaxCandidates>, kMaxCandidates>
        matrix{};
    std::array<double, kMaxCandidates> rhs{};
    for (int row = 0; row < active_count; ++row) {
      const int i = active_indices[static_cast<std::size_t>(row)];
      rhs[static_cast<std::size_t>(row)] =
          b[static_cast<std::size_t>(i)];
      for (int column = 0; column < active_count; ++column) {
        const int j =
            active_indices[static_cast<std::size_t>(column)];
        matrix[static_cast<std::size_t>(row)]
              [static_cast<std::size_t>(column)] =
            S[static_cast<std::size_t>(i)]
             [static_cast<std::size_t>(j)];
      }
    }

    bool singular = false;
    for (int column = 0; column < active_count; ++column) {
      int pivot_row = column;
      double pivot_magnitude =
          std::abs(matrix[static_cast<std::size_t>(column)]
                         [static_cast<std::size_t>(column)]);
      for (int row = column + 1; row < active_count; ++row) {
        const double magnitude =
            std::abs(matrix[static_cast<std::size_t>(row)]
                           [static_cast<std::size_t>(column)]);
        if (magnitude > pivot_magnitude) {
          pivot_magnitude = magnitude;
          pivot_row = row;
        }
      }
      if (!std::isfinite(pivot_magnitude) ||
          pivot_magnitude < kPivotFloor) {
        singular = true;
        break;
      }
      if (pivot_row != column) {
        std::swap(matrix[static_cast<std::size_t>(pivot_row)],
                  matrix[static_cast<std::size_t>(column)]);
        std::swap(rhs[static_cast<std::size_t>(pivot_row)],
                  rhs[static_cast<std::size_t>(column)]);
      }
      const double pivot =
          matrix[static_cast<std::size_t>(column)]
                [static_cast<std::size_t>(column)];
      for (int row = column + 1; row < active_count; ++row) {
        const double factor =
            matrix[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(column)] /
            pivot;
        matrix[static_cast<std::size_t>(row)]
              [static_cast<std::size_t>(column)] = 0.0;
        for (int k = column + 1; k < active_count; ++k) {
          matrix[static_cast<std::size_t>(row)]
                [static_cast<std::size_t>(k)] -=
              factor *
              matrix[static_cast<std::size_t>(column)]
                    [static_cast<std::size_t>(k)];
        }
        rhs[static_cast<std::size_t>(row)] -=
            factor * rhs[static_cast<std::size_t>(column)];
      }
    }
    if (singular) {
      if (active_count == 0) {
        return result;
      }
      active[static_cast<std::size_t>(
          active_indices[static_cast<std::size_t>(active_count - 1)])] =
          false;
      continue;
    }

    std::array<double, kMaxCandidates> lambda{};
    bool finite_solution = true;
    for (int row = active_count - 1; row >= 0; --row) {
      double value = rhs[static_cast<std::size_t>(row)];
      for (int column = row + 1; column < active_count; ++column) {
        const int j =
            active_indices[static_cast<std::size_t>(column)];
        value -=
            matrix[static_cast<std::size_t>(row)]
                  [static_cast<std::size_t>(column)] *
            lambda[static_cast<std::size_t>(j)];
      }
      const int i = active_indices[static_cast<std::size_t>(row)];
      lambda[static_cast<std::size_t>(i)] =
          value /
          matrix[static_cast<std::size_t>(row)]
                [static_cast<std::size_t>(row)];
      if (!std::isfinite(lambda[static_cast<std::size_t>(i)])) {
        finite_solution = false;
        break;
      }
    }
    if (!finite_solution) {
      return result;
    }

    int most_negative = -1;
    for (int i = 0; i < candidate_count; ++i) {
      if (!active[static_cast<std::size_t>(i)] ||
          lambda[static_cast<std::size_t>(i)] >= 0.0) {
        continue;
      }
      if (most_negative < 0 ||
          lambda[static_cast<std::size_t>(i)] <
              lambda[static_cast<std::size_t>(most_negative)] ||
          (lambda[static_cast<std::size_t>(i)] ==
               lambda[static_cast<std::size_t>(most_negative)] &&
           i > most_negative)) {
        most_negative = i;
      }
    }
    if (most_negative >= 0) {
      active[static_cast<std::size_t>(most_negative)] = false;
      continue;
    }

    std::vector<double> candidate_du_r(nodal_mass.size(), 0.0);
    std::vector<double> candidate_du_z(nodal_mass.size(), 0.0);
    for (int i = 0; i < candidate_count; ++i) {
      if (!active[static_cast<std::size_t>(i)]) {
        continue;
      }
      for (int p = 0; p < 3; ++p) {
        const int node =
            patch_node[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(p)];
        const double inverse_mass =
            1.0 / nodal_mass[static_cast<std::size_t>(node)];
        candidate_du_r[static_cast<std::size_t>(node)] +=
            lambda[static_cast<std::size_t>(i)] *
            gradient_r[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(p)] *
            inverse_mass;
        candidate_du_z[static_cast<std::size_t>(node)] +=
            lambda[static_cast<std::size_t>(i)] *
            gradient_z[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(p)] *
            inverse_mass;
      }
    }

    int most_violated = -1;
    double largest_relative_violation = 0.0;
    for (int i = 0; i < candidate_count; ++i) {
      if (active[static_cast<std::size_t>(i)]) {
        continue;
      }
      double achieved = 0.0;
      for (int p = 0; p < 3; ++p) {
        const int node =
            patch_node[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(p)];
        achieved +=
            gradient_r[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(p)] *
                candidate_du_r[static_cast<std::size_t>(node)] +
            gradient_z[static_cast<std::size_t>(i)]
                      [static_cast<std::size_t>(p)] *
                candidate_du_z[static_cast<std::size_t>(node)];
      }
      const double relative_violation =
          (b[static_cast<std::size_t>(i)] - achieved) /
          b[static_cast<std::size_t>(i)];
      if (relative_violation > kRelativeSlack &&
          (most_violated < 0 ||
           relative_violation > largest_relative_violation ||
           (relative_violation == largest_relative_violation &&
            i > most_violated))) {
        most_violated = i;
        largest_relative_violation = relative_violation;
      }
    }
    if (most_violated >= 0) {
      active[static_cast<std::size_t>(most_violated)] = true;
      continue;
    }

    du_r.swap(candidate_du_r);
    du_z.swap(candidate_du_z);
    result.converged = true;
    return result;
  }
  return result;
}

bool exact_patch_check(const RuntimeState& runtime,
                       const std::vector<PatchCell>& patch_cells,
                       const std::vector<int>& node_to_patch,
                       const std::vector<double>& du_r,
                       const std::vector<double>& du_z,
                       const double dt) {
  for (const auto& patch_cell : patch_cells) {
    for (int k = 0; k < patch_cell.nverts; ++k) {
      const double area =
          corrected_predicted_corner_area(runtime,
                                          patch_cell,
                                          k,
                                          dt,
                                          node_to_patch,
                                          du_r,
                                          du_z);
      if (!std::isfinite(area) || area < patch_cell.frozen_floor) {
        return false;
      }
    }
  }
  return true;
}

void log_projection_result(const int step,
                           const double time,
                           const bool applied,
                           const double dE,
                           const int constraints,
                           const int patch_cells,
                           const char* suffix = "") {
  std::ostringstream log;
  log << "[axis-proj] step=" << step
      << " t=" << std::scientific << std::setprecision(6) << time
      << (applied ? " applied dE=" : " rejected dE=") << dE
      << " constraints=" << constraints
      << " patch_cells=" << patch_cells
      << suffix;
  if (applied) {
    core::log_info(log.str());
  } else {
    core::log_warning(log.str());
  }
}

void apply_active_projection(const core::State& state,
                             const core::Config& cfg,
                             RuntimeState& runtime,
                             double* pos_v_r,
                             double* pos_v_z,
                             double* ion_energy_old,
                             const double dt,
                             const double predicted_time,
                             const int step,
                             const int argmin_cell,
                             const int owned_begin,
                             const int owned_end) {
  if (argmin_cell < owned_begin || argmin_cell >= owned_end) {
    return;
  }
  TENRYU_ASSERT(std::isfinite(dt) && dt > 0.0,
                "active axis projection requires positive finite dt");

  if (!runtime.incident_table_initialized) {
    initialize_incident_table(state, cfg, runtime);
  }
  std::vector<PatchCell> patch_cells;
  if (is_single_row_transition_cell(state, argmin_cell)) {
    TENRYU_ASSERT(
        build_node_neighborhood_patch_cells(
            state, cfg, runtime, argmin_cell, patch_cells),
        "active axis projection failed to build transition node-neighborhood "
        "patch");
  } else {
    TENRYU_ASSERT(
        build_patch_cells(state, cfg, runtime, argmin_cell, patch_cells),
        "active axis projection argmin is not a monitored POLAR_TIER side cell");
  }
  for (const auto& patch_cell : patch_cells) {
    if (patch_cell.cell < owned_begin || patch_cell.cell >= owned_end) {
      ++runtime.skipped_unowned_patches;
      return;
    }
  }

  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  TENRYU_ASSERT(
      state.corner_stride > 0 &&
          state.corner_mass.size() ==
              static_cast<std::size_t>(n_cells) *
                  static_cast<std::size_t>(state.corner_stride),
      "active axis projection requires a complete corner-mass field");
  copy_device_field(
      state.corner_mass.data(),
      state.corner_mass.size(),
      runtime.corner_mass_host,
      "active axis projection corner_mass D2H copy failed");

  std::vector<int> patch_nodes;
  for (const auto& patch_cell : patch_cells) {
    for (int k = 0; k < patch_cell.nverts; ++k) {
      patch_nodes.push_back(
          patch_cell.nodes[static_cast<std::size_t>(k)]);
    }
  }
  std::sort(patch_nodes.begin(), patch_nodes.end());
  patch_nodes.erase(std::unique(patch_nodes.begin(), patch_nodes.end()),
                    patch_nodes.end());
  std::vector<int> node_to_patch(static_cast<std::size_t>(n_nodes), -1);
  for (std::size_t p = 0; p < patch_nodes.size(); ++p) {
    const int node = patch_nodes[p];
    TENRYU_ASSERT(node >= 0 && node < n_nodes,
                  "active axis projection patch node out of range");
    node_to_patch[static_cast<std::size_t>(node)] =
        static_cast<int>(p);
  }

  std::vector<double> nodal_mass(patch_nodes.size(), 0.0);
  for (std::size_t p = 0; p < patch_nodes.size(); ++p) {
    const int node = patch_nodes[p];
    double mass = 0.0;
    for (const IncidentCorner& incident :
         runtime.node_incident_corners[static_cast<std::size_t>(node)]) {
      const std::size_t index =
          static_cast<std::size_t>(incident.cell) *
              static_cast<std::size_t>(state.corner_stride) +
          static_cast<std::size_t>(incident.corner);
      mass += runtime.corner_mass_host[index];
    }
    TENRYU_ASSERT(std::isfinite(mass) && mass > 0.0,
                  "active axis projection requires positive nodal mass");
    nodal_mass[p] = mass;
  }

  std::vector<double> du_r(patch_nodes.size(), 0.0);
  std::vector<double> du_z(patch_nodes.size(), 0.0);
  const std::vector<AreaConstraint> constraints =
      build_constraints(runtime,
                        cfg,
                        patch_cells,
                        node_to_patch,
                        du_r,
                        du_z,
                        dt);
  if (constraints.empty()) {
    return;
  }
  const ProjectionSolveResult solve =
      project_constraints(state, constraints, nodal_mass, dt, du_r, du_z);
  if (solve.truncated_constraints > 0) {
    std::ostringstream log;
    log << "[axis-proj] step=" << step
        << " t=" << std::scientific << std::setprecision(6)
        << predicted_time
        << " candidate_constraints=12"
        << " truncated_constraints=" << solve.truncated_constraints;
    core::log_warning(log.str());
  }
  if (!solve.converged) {
    ++runtime.rejected_projections;
    log_projection_result(step,
                          predicted_time,
                          false,
                          0.0,
                          static_cast<int>(constraints.size()),
                          static_cast<int>(patch_cells.size()),
                          " reason=qp_no_converge");
    return;
  }

  if (!exact_patch_check(runtime,
                         patch_cells,
                         node_to_patch,
                         du_r,
                         du_z,
                         dt)) {
    ++runtime.rejected_projections;
    log_projection_result(step,
                          predicted_time,
                          false,
                          0.0,
                          static_cast<int>(constraints.size()),
                          static_cast<int>(patch_cells.size()));
    return;
  }

  double dE = 0.0;
  for (std::size_t p = 0; p < patch_nodes.size(); ++p) {
    const int node = patch_nodes[p];
    const double old_r =
        runtime.pos_v_r_host[static_cast<std::size_t>(node)];
    const double old_z =
        runtime.pos_v_z_host[static_cast<std::size_t>(node)];
    const double new_r = old_r + du_r[p];
    const double new_z = old_z + du_z[p];
    dE += 0.5 * nodal_mass[p] *
          (old_r * old_r + old_z * old_z -
           new_r * new_r - new_z * new_z);
  }
  TENRYU_ASSERT(std::isfinite(dE),
                "active axis projection produced non-finite energy transfer");
  if (dE < 0.0) {
    ++runtime.rejected_energy_projections;
    log_projection_result(step,
                          predicted_time,
                          false,
                          dE,
                          static_cast<int>(constraints.size()),
                          static_cast<int>(patch_cells.size()),
                          " reason=dE<0");
    return;
  }

  copy_device_field(
      state.mass.data(),
      state.mass.size(),
      runtime.cell_mass_host,
      "active axis projection cell mass D2H copy failed");
  copy_device_field(
      ion_energy_old,
      state.mass.size(),
      runtime.ion_energy_old_host,
      "active axis projection ion energy D2H copy failed");
  double patch_mass = 0.0;
  for (const auto& patch_cell : patch_cells) {
    const double mass =
        runtime.cell_mass_host[static_cast<std::size_t>(patch_cell.cell)];
    TENRYU_ASSERT(std::isfinite(mass) && mass > 0.0,
                  "active axis projection requires positive patch cell mass");
    patch_mass += mass;
  }
  TENRYU_ASSERT(std::isfinite(patch_mass) && patch_mass > 0.0,
                "active axis projection requires positive patch mass");
  for (const auto& patch_cell : patch_cells) {
    const std::size_t cell = static_cast<std::size_t>(patch_cell.cell);
    const double cell_mass = runtime.cell_mass_host[cell];
    const double weight = cell_mass / patch_mass;
    runtime.ion_energy_old_host[cell] += dE * weight / cell_mass;
    TENRYU_ASSERT(
        std::isfinite(runtime.ion_energy_old_host[cell]),
        "active axis projection produced non-finite ion internal energy");
  }

  for (std::size_t p = 0; p < patch_nodes.size(); ++p) {
    const std::size_t node = static_cast<std::size_t>(patch_nodes[p]);
    runtime.pos_v_r_host[node] += du_r[p];
    runtime.pos_v_z_host[node] += du_z[p];
  }
  copy_host_field(
      runtime.ion_energy_old_host,
      ion_energy_old,
      "active axis projection ion energy H2D copy failed");
  copy_host_field(
      runtime.pos_v_r_host,
      pos_v_r,
      "active axis projection pos_v_r H2D copy failed");
  copy_host_field(
      runtime.pos_v_z_host,
      pos_v_z,
      "active axis projection pos_v_z H2D copy failed");

  ++runtime.applied_projections;
  runtime.E_axis_projection += dE;
  log_projection_result(step,
                        predicted_time,
                        true,
                        dE,
                        static_cast<int>(constraints.size()),
                        static_cast<int>(patch_cells.size()));
}

void log_line(const int step,
              const double time,
              const double q_pred,
              const int cell,
              const bool is_transition,
              const char* suffix) {
  std::ostringstream log;
  log << "[axis-proj-shadow] step=" << step
      << " t=" << std::scientific << std::setprecision(6) << time
      << " q_pred=" << q_pred
      << " cell=" << cell;
  if (is_transition) {
    log << " class=transition";
  }
  if (suffix != nullptr) {
    log << suffix;
  }
  core::log_info(log.str());
}

}  // namespace

void run_start(const core::Config& cfg) {
  if (!cfg.numerics.hydro.axis_projection.enabled) {
    return;
  }
  delete g_runtime;
  g_runtime = new RuntimeState;
}

void observe_precommit(const core::State& state,
                       const core::Config& cfg,
                       const double* x_r_old,
                       const double* x_z_old,
                       double* pos_v_r,
                       double* pos_v_z,
                       double* ion_energy_old,
                       const double dt,
                       const double predicted_time,
                       const parallel::Reduction* reduction,
                       const int rank) {
  if (!cfg.numerics.hydro.axis_projection.enabled ||
      !applicable(state, cfg)) {
    return;
  }
  if (g_runtime == nullptr) {
    g_runtime = new RuntimeState;
  }
  RuntimeState& runtime = *g_runtime;
  if (!runtime.initialized) {
    initialize_monitored_cells(state, cfg, runtime);
  }
  if (runtime.monitored_cells.empty()) {
    return;
  }

  const std::size_t n_nodes = state.x_r.size();
  copy_device_field(
      x_r_old,
      n_nodes,
      runtime.x_r_old_host,
      "axis projection shadow x_r_old D2H copy failed");
  copy_device_field(
      x_z_old,
      n_nodes,
      runtime.x_z_old_host,
      "axis projection shadow x_z_old D2H copy failed");
  copy_device_field(
      pos_v_r,
      n_nodes,
      runtime.pos_v_r_host,
      "axis projection shadow pos_v_r D2H copy failed");
  copy_device_field(
      pos_v_z,
      n_nodes,
      runtime.pos_v_z_host,
      "axis projection shadow pos_v_z D2H copy failed");
  if (!runtime.initial_geometry_captured) {
    TENRYU_ASSERT(
        state.x_r_initial.size() == n_nodes &&
            state.x_z_initial.size() == n_nodes,
        "axis projection shadow monitor requires initial node coordinates");
    state.x_r_initial.copy_to_host(runtime.x_r_initial_host);
    state.x_z_initial.copy_to_host(runtime.x_z_initial_host);
    capture_initial_geometry(runtime);
  }

  const int owned_begin = state.owned_cells_begin_or(0);
  const int owned_end =
      state.owned_cells_end_or(static_cast<int>(state.rho.size()));
  double local_q_pred = std::numeric_limits<double>::infinity();
  int local_argmin_cell = std::numeric_limits<int>::max();
  double local_active_q_pred = std::numeric_limits<double>::infinity();
  int local_active_argmin_cell = std::numeric_limits<int>::max();
  for (const auto& monitored : runtime.monitored_cells) {
    if (monitored.cell < owned_begin || monitored.cell >= owned_end) {
      continue;
    }
    std::array<double, mesh::kMeshTopoCellStorageSlots> predicted_areas{};
    double predicted_area_sum = 0.0;
    for (int k = 0; k < monitored.nverts; ++k) {
      predicted_areas[static_cast<std::size_t>(k)] =
          predicted_corner_area(runtime, monitored, k, dt);
      predicted_area_sum += predicted_areas[static_cast<std::size_t>(k)];
    }
    const double mean_pred =
        predicted_area_sum / static_cast<double>(monitored.nverts);
    double q_cell = 0.0;
    if (std::isfinite(mean_pred) && mean_pred > 0.0) {
      q_cell = std::numeric_limits<double>::infinity();
      for (int k = 0; k < monitored.nverts; ++k) {
        const double ratio =
            predicted_areas[static_cast<std::size_t>(k)] / mean_pred;
        q_cell =
            std::min(q_cell,
                     std::isfinite(ratio) ? ratio : 0.0);
      }
    }
    if (q_cell < local_q_pred ||
        (q_cell == local_q_pred &&
         monitored.cell < local_argmin_cell)) {
      local_q_pred = q_cell;
      local_argmin_cell = monitored.cell;
    }
    if (monitored.active_eligible &&
        (q_cell < local_active_q_pred ||
         (q_cell == local_active_q_pred &&
          monitored.cell < local_active_argmin_cell))) {
      local_active_q_pred = q_cell;
      local_active_argmin_cell = monitored.cell;
    }
  }

  const double q_pred =
      reduction != nullptr
          ? reduction->allreduce_min(local_q_pred)
          : local_q_pred;
  const std::uint64_t no_cell =
      std::numeric_limits<std::uint64_t>::max();
  const bool owns_global_min =
      local_argmin_cell != std::numeric_limits<int>::max() &&
      local_q_pred == q_pred;
  std::uint64_t argmin_candidate =
      owns_global_min ? static_cast<std::uint64_t>(local_argmin_cell) : no_cell;
  if (reduction != nullptr) {
    argmin_candidate = reduction->allreduce_min(argmin_candidate);
  }
  if (argmin_candidate == no_cell) {
    return;
  }
  const int argmin_cell = static_cast<int>(argmin_candidate);
  const double active_q_pred =
      reduction != nullptr
          ? reduction->allreduce_min(local_active_q_pred)
          : local_active_q_pred;
  const bool owns_active_global_min =
      local_active_argmin_cell != std::numeric_limits<int>::max() &&
      local_active_q_pred == active_q_pred;
  std::uint64_t active_argmin_candidate =
      owns_active_global_min
          ? static_cast<std::uint64_t>(local_active_argmin_cell)
          : no_cell;
  if (reduction != nullptr) {
    active_argmin_candidate =
        reduction->allreduce_min(active_argmin_candidate);
  }
  const int active_argmin_cell =
      active_argmin_candidate != no_cell
          ? static_cast<int>(active_argmin_candidate)
          : -1;
  const auto argmin_monitored =
      std::lower_bound(runtime.monitored_cells.begin(),
                       runtime.monitored_cells.end(),
                       argmin_cell,
                       [](const MonitoredCell& monitored, const int cell) {
                         return monitored.cell < cell;
                       });
  TENRYU_ASSERT(
      argmin_monitored != runtime.monitored_cells.end() &&
          argmin_monitored->cell == argmin_cell,
      "axis projection shadow argmin cell is not monitored");

  runtime.global_min_q_pred =
      std::min(runtime.global_min_q_pred, q_pred);
  const bool below =
      q_pred < cfg.numerics.hydro.axis_projection.q_on;
  if (below) {
    ++runtime.steps_below_threshold;
  }
  const int step = state.step + 1;
  if (rank == 0 && below && !runtime.engaged) {
    log_line(step,
             predicted_time,
             q_pred,
             argmin_cell,
             argmin_monitored->is_transition,
             nullptr);
  }
  if (rank == 0 && !below && runtime.engaged) {
    log_line(step,
             predicted_time,
             q_pred,
             argmin_cell,
             argmin_monitored->is_transition,
             " engaged=0");
  }
  runtime.engaged = below;

  const int log_every =
      cfg.numerics.hydro.axis_projection.log_every_n_steps;
  if (rank == 0 && log_every > 0 && step % log_every == 0) {
    log_line(step,
             predicted_time,
             q_pred,
             argmin_cell,
             argmin_monitored->is_transition,
             " periodic=1");
  }
  const bool active_below =
      active_q_pred < cfg.numerics.hydro.axis_projection.q_on;
  if (!cfg.numerics.hydro.axis_projection.shadow_only &&
      active_below &&
      active_argmin_cell >= 0) {
    apply_active_projection(state,
                            cfg,
                            runtime,
                            pos_v_r,
                            pos_v_z,
                            ion_energy_old,
                            dt,
                            predicted_time,
                            step,
                            active_argmin_cell,
                            owned_begin,
                            owned_end);
  }
}

void run_end(const core::Config& cfg, const int rank) {
  if (!cfg.numerics.hydro.axis_projection.enabled ||
      g_runtime == nullptr) {
    return;
  }
  if (rank == 0 && g_runtime->initial_geometry_captured) {
    std::ostringstream log;
    if (cfg.numerics.hydro.axis_projection.shadow_only) {
      log << "[axis-proj-shadow-summary] steps_below_threshold="
          << g_runtime->steps_below_threshold
          << " global_min_q_pred=" << std::scientific
          << std::setprecision(6) << g_runtime->global_min_q_pred;
    } else {
      log << "[axis-proj-summary] steps_below_threshold="
          << g_runtime->steps_below_threshold
          << " global_min_q_pred=" << std::scientific
          << std::setprecision(6) << g_runtime->global_min_q_pred
          << " applied_projections=" << g_runtime->applied_projections
          << " rejected_projections=" << g_runtime->rejected_projections
          << " rejected_energy_projections="
          << g_runtime->rejected_energy_projections
          << " skipped_unowned_patches="
          << g_runtime->skipped_unowned_patches
          << " E_axis_projection=" << g_runtime->E_axis_projection;
    }
    core::log_info(log.str());
  }
  delete g_runtime;
  g_runtime = nullptr;
}

}  // namespace tenryu::hydro::axis_projection
