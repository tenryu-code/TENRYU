#include "hydro/euler_window_blend.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/namelist/errors.hpp"
#include "core/state.hpp"
#include "hydro/ale_remap.cuh"
#include "mesh/mesh.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct AxisCoreStructuralUnit {
  int block_id = -1;
  int radial_row = -1;
  mesh::BlockRole role = mesh::BlockRole::CENTRAL_CORE;
  double outer_radius = 0.0;
  std::vector<int> cells;
  std::vector<int> nodes;
  std::vector<int> outer_nodes;
};

enum class AxisCoreTransitionPassageState : std::uint8_t {
  Convergence = 0,
  Armed = 1,
  Passage = 2,
};

struct AxisCoreTransitionPatch {
  int belt_block_id = -1;
  double inner_radius = 0.0;
  double median_inner_width = 0.0;
  std::vector<int> cells;
  std::vector<int> nodes;
  std::vector<double> feather;
  std::vector<double> optimal_r;
  std::vector<double> optimal_z;
  std::vector<int> sensor_cells;
  std::vector<int> swept_cells;
  std::vector<int> swept_local_faces;
  std::vector<int> swept_neighbors;
  core::DeviceArray<int> d_cells;
  core::DeviceArray<int> d_nodes;
  core::DeviceArray<int> d_node_cell_offsets;
  core::DeviceArray<int> d_node_cells;
  core::DeviceArray<double> d_feather;
  core::DeviceArray<std::uint8_t> d_inactive_cell_mask;
  core::DeviceArray<int> d_sensor_cells;
  core::DeviceArray<int> d_swept_cells;
  core::DeviceArray<int> d_swept_local_faces;
  core::DeviceArray<int> d_swept_neighbors;
  core::DeviceArray<double> d_cell_center_r;
  core::DeviceArray<double> d_cell_center_z;
  core::DeviceArray<double> d_sensor_strength;
  core::DeviceArray<double> d_sensor_radius;
  core::DeviceArray<double> d_swept_fraction;
  AxisCoreTransitionPassageState state =
      AxisCoreTransitionPassageState::Convergence;
  double last_shock_radius = 0.0;
  std::array<std::uint64_t, 10> alpha_hist{};
  std::uint64_t passage_steps = 0U;
  std::uint64_t escalate_count = 0U;
  int low_alpha_consecutive = 0;
  bool escalated_latched = false;
};

struct AxisCoreTransitionPatchCandidate {
  int belt_block_id = -1;
  double inner_radius = 0.0;
  double median_inner_width = 0.0;
  std::vector<int> cells;
  std::vector<int> sensor_cells;
};

struct AxisCoreDiskRuntime {
  int n_cells = 0;
  int n_nodes = 0;
  double s_p = 0.0;
  double s_f = 0.0;
  std::vector<double> initial_radius;
  std::vector<double> chi;
  std::vector<double> captured_r;
  std::vector<double> captured_z;
  std::vector<std::uint8_t> plateau_node;
  std::vector<std::uint8_t> feather_node;
  std::vector<std::uint8_t> guard_node;
  std::vector<std::uint8_t> mask;
  std::vector<int> plateau_nodes;
  std::vector<int> plateau_cells;
  std::vector<int> feather_cells;
  std::vector<int> guard_cells;
  std::vector<AxisCoreJunctionLedgerCell> junction_ledger_cells;
  core::DeviceArray<std::uint8_t> seam_donor_cell_mask;
  std::size_t seam_donor_cell_count = 0;
  bool always_moving_mode = false;
  long long follow_count = 0;
  long long repair_count = 0;
  bool uniform_mass_warning_emitted = false;
  bool transition_passage_enabled = false;
  std::vector<AxisCoreTransitionPatch> transition_patches;
  std::vector<int> transition_central_cells;
  core::DeviceArray<int> d_transition_central_cells;
  core::DeviceArray<double> d_transition_central_mass_ur;
  core::DeviceArray<double> d_transition_central_mass_cs;
  core::DeviceArray<double> d_transition_central_mass;
  core::DeviceArray<int> d_transition_face_adj_offsets;
  core::DeviceArray<int> d_transition_face_adj_indices;
  core::DeviceArray<int> d_transition_cell_orientation_sign;
  core::DeviceArray<std::uint8_t> d_transition_cell_nverts;
  core::DeviceArray<double> d_transition_jacobi_r_a;
  core::DeviceArray<double> d_transition_jacobi_z_a;
  core::DeviceArray<double> d_transition_jacobi_r_b;
  core::DeviceArray<double> d_transition_jacobi_z_b;
  core::DeviceArray<double> d_transition_candidate_r;
  core::DeviceArray<double> d_transition_candidate_z;
};

AxisCoreDiskRuntime* g_axis_core_disk_runtime = nullptr;
int g_axis_core_released_units = 0;

constexpr double kTransitionPassagePressureFloor = 1.0e-30;

__global__ void axis_core_transition_central_outflow_kernel(
    const int* __restrict__ cells,
    const int n_selected,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ velocity_r,
    const double* __restrict__ velocity_z,
    const double* __restrict__ sound_speed,
    const double* __restrict__ cell_mass,
    const std::int8_t* __restrict__ hydro_active,
    double* __restrict__ mass_ur,
    double* __restrict__ mass_cs,
    double* __restrict__ selected_mass) {
  const int selected = blockIdx.x * blockDim.x + threadIdx.x;
  if (selected >= n_selected) {
    return;
  }
  const int cell = cells[selected];
  if (hydro_active != nullptr && hydro_active[cell] == 0) {
    mass_ur[selected] = 0.0;
    mass_cs[selected] = 0.0;
    selected_mass[selected] = 0.0;
    return;
  }
  const int nverts = static_cast<int>(cell_nverts[cell]);
  const int offset = cell_node_csr_offsets[cell];
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  double mean_vr = 0.0;
  double mean_vz = 0.0;
  for (int local = 0; local < nverts; ++local) {
    const int node = cell_node_csr_indices[offset + local];
    centroid_r += node_r[node];
    centroid_z += node_z[node];
    mean_vr += velocity_r[node];
    mean_vz += velocity_z[node];
  }
  const double inv_nverts = 1.0 / static_cast<double>(nverts);
  centroid_r *= inv_nverts;
  centroid_z *= inv_nverts;
  mean_vr *= inv_nverts;
  mean_vz *= inv_nverts;
  const double radius = hypot(centroid_r, centroid_z);
  const double radial_velocity =
      radius > 0.0
          ? (mean_vr * centroid_r + mean_vz * centroid_z) / radius
          : 0.0;
  const double mass = cell_mass[cell];
  selected_mass[selected] = mass;
  mass_ur[selected] = mass * radial_velocity;
  mass_cs[selected] = mass * sound_speed[cell];
}

__global__ void axis_core_transition_sensor_kernel(
    const int* __restrict__ cells,
    const int n_selected,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ velocity_r,
    const double* __restrict__ velocity_z,
    const double* __restrict__ pressure_e,
    const double* __restrict__ pressure_i,
    const double* __restrict__ sound_speed,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    double* __restrict__ strength,
    double* __restrict__ radius_out) {
  const int selected = blockIdx.x * blockDim.x + threadIdx.x;
  if (selected >= n_selected) {
    return;
  }
  const int cell = cells[selected];
  if (hydro_active != nullptr && hydro_active[cell] == 0) {
    strength[selected] = -kInf;
    radius_out[selected] = 0.0;
    return;
  }
  const int nverts = static_cast<int>(cell_nverts[cell]);
  const int offset = cell_node_csr_offsets[cell];
  double area2 = 0.0;
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  for (int local = 0; local < nverts; ++local) {
    const int next = local + 1 == nverts ? 0 : local + 1;
    const int node0 = cell_node_csr_indices[offset + local];
    const int node1 = cell_node_csr_indices[offset + next];
    area2 += node_r[node0] * node_z[node1] -
             node_r[node1] * node_z[node0];
    centroid_r += node_r[node0];
    centroid_z += node_z[node0];
  }
  const double area = 0.5 * fabs(area2);
  const double cs = sound_speed[cell];
  const double pressure_cell = pressure_e[cell] + pressure_i[cell];
  const double pressure_denominator =
      pressure_cell + kTransitionPassagePressureFloor;
  if (!(area > 0.0) || !(cs > 0.0) ||
      !(pressure_denominator > 0.0) || !isfinite(area) || !isfinite(cs) ||
      !isfinite(pressure_cell)) {
    strength[selected] = -kInf;
    radius_out[selected] = 0.0;
    return;
  }

  const double orientation = area2 >= 0.0 ? 1.0 : -1.0;
  double divergence_numerator = 0.0;
  double pressure_gradient_r_numerator = 0.0;
  double pressure_gradient_z_numerator = 0.0;
  const int face_offset = face_adj_csr_offsets[cell];
  const int face_slot_count = face_adj_csr_offsets[cell + 1] - face_offset;
  for (int local_face = 0;
       local_face < face_slot_count;
       ++local_face) {
    int corner0 = 0;
    int corner1 = 0;
    if (!mesh::mesh_topo_active_local_face_corners(
            nverts, local_face, &corner0, &corner1)) {
      continue;
    }
    const int node0 = cell_node_csr_indices[offset + corner0];
    const int node1 = cell_node_csr_indices[offset + corner1];
    const double dr = node_r[node1] - node_r[node0];
    const double dz = node_z[node1] - node_z[node0];
    const double normal_r_length = orientation * dz;
    const double normal_z_length = -orientation * dr;
    const double face_vr = 0.5 * (velocity_r[node0] + velocity_r[node1]);
    const double face_vz = 0.5 * (velocity_z[node0] + velocity_z[node1]);
    divergence_numerator +=
        face_vr * normal_r_length + face_vz * normal_z_length;

    const int neighbor = face_adj_csr_indices[face_offset + local_face];
    double face_pressure = pressure_cell;
    if (neighbor >= 0 && neighbor < n_cells &&
        (hydro_active == nullptr || hydro_active[neighbor] != 0)) {
      face_pressure =
          0.5 * (pressure_cell + pressure_e[neighbor] + pressure_i[neighbor]);
    }
    pressure_gradient_r_numerator += face_pressure * normal_r_length;
    pressure_gradient_z_numerator += face_pressure * normal_z_length;
  }

  const double h = sqrt(area);
  const double divergence = divergence_numerator / area;
  const double grad_pressure =
      hypot(pressure_gradient_r_numerator,
            pressure_gradient_z_numerator) /
      area;
  strength[selected] =
      fmax(0.0, -h * divergence / cs) +
      0.5 * h * grad_pressure / pressure_denominator;
  const double inv_nverts = 1.0 / static_cast<double>(nverts);
  centroid_r *= inv_nverts;
  centroid_z *= inv_nverts;
  radius_out[selected] = hypot(centroid_r, centroid_z);
}

__global__ void axis_core_transition_cell_centers_kernel(
    const int* __restrict__ cells,
    const int n_patch_cells,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    double* __restrict__ center_r,
    double* __restrict__ center_z) {
  const int local_cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_cell >= n_patch_cells) {
    return;
  }
  const int cell = cells[local_cell];
  const int nverts = static_cast<int>(cell_nverts[cell]);
  const int offset = cell_node_csr_offsets[cell];
  double sum_r = 0.0;
  double sum_z = 0.0;
  for (int local = 0; local < nverts; ++local) {
    const int node = cell_node_csr_indices[offset + local];
    sum_r += node_r[node];
    sum_z += node_z[node];
  }
  const double inv_nverts = 1.0 / static_cast<double>(nverts);
  center_r[local_cell] = sum_r * inv_nverts;
  center_z[local_cell] = sum_z * inv_nverts;
}

__global__ void axis_core_transition_jacobi_nodes_kernel(
    const int* __restrict__ nodes,
    const int n_patch_nodes,
    const int* __restrict__ node_cell_offsets,
    const int* __restrict__ node_cells,
    const double* __restrict__ feather,
    const double* __restrict__ center_r,
    const double* __restrict__ center_z,
    const double* __restrict__ current_r,
    const double* __restrict__ current_z,
    double* __restrict__ next_r,
    double* __restrict__ next_z) {
  const int local_node = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_node >= n_patch_nodes) {
    return;
  }
  const int begin = node_cell_offsets[local_node];
  const int end = node_cell_offsets[local_node + 1];
  const int node = nodes[local_node];
  if (feather[local_node] == 0.0) {
    next_r[node] = current_r[node];
    next_z[node] = current_z[node];
    return;
  }
  double average_r = 0.0;
  double average_z = 0.0;
  for (int entry = begin; entry < end; ++entry) {
    const int local_cell = node_cells[entry];
    average_r += center_r[local_cell];
    average_z += center_z[local_cell];
  }
  const double inv_count = 1.0 / static_cast<double>(end - begin);
  average_r *= inv_count;
  average_z *= inv_count;
  next_r[node] = 0.5 * average_r + 0.5 * current_r[node];
  next_z[node] = 0.5 * average_z + 0.5 * current_z[node];
}

__global__ void axis_core_transition_swept_fraction_kernel(
    const int* __restrict__ cells,
    const int* __restrict__ local_faces,
    const int* __restrict__ neighbors,
    const int n_faces,
    const double* __restrict__ lagrangian_r,
    const double* __restrict__ lagrangian_z,
    const double* __restrict__ candidate_r,
    const double* __restrict__ candidate_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ cell_volume,
    double* __restrict__ swept_fraction) {
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  if (face >= n_faces) {
    return;
  }
  const int cell = cells[face];
  const int neighbor = neighbors[face];
  const double denominator = fmin(cell_volume[cell], cell_volume[neighbor]);
  if (!(denominator > 0.0) || !isfinite(denominator)) {
    swept_fraction[face] = kInf;
    return;
  }
  const double swept_volume =
      ale::detail::csr_face_swept_volume_outward(
          lagrangian_r,
          lagrangian_z,
          candidate_r,
          candidate_z,
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_orientation_sign,
          cell,
          local_faces[face],
          cell_nverts);
  swept_fraction[face] = fabs(swept_volume) / denominator;
}

bool junction_audit_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_JUNCTION_AUDIT");
    return raw != nullptr && raw[0] != '\0' && std::atoi(raw) != 0;
  }();
  return enabled;
}

int junction_ledger_cadence() {
  static const int cadence = [] {
    const char* raw = std::getenv("TENRYU_I1B_JUNCTION_LEDGER");
    return raw != nullptr && raw[0] != '\0'
               ? std::max(0, std::atoi(raw))
               : 0;
  }();
  return cadence;
}

bool seam_donor_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_SEAM_DONOR");
    return raw != nullptr && raw[0] != '\0' && std::atoi(raw) != 0;
  }();
  return enabled;
}

void sort_unique(std::vector<int>& values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

struct AxisCoreRadialNeighbor {
  int block_id = -1;
  mesh::BlockSide side = mesh::BlockSide::I_MINUS;
};

struct AxisCoreJunctionCohort {
  int belt_block_id = -1;
  std::array<AxisCoreRadialNeighbor, 2> radial_neighbors;
  std::vector<int> cells;
};

bool find_axis_core_radial_neighbor(
    const mesh::MultiBlockTopology& topology,
    const int belt_block_id,
    const mesh::BlockSide belt_side,
    AxisCoreRadialNeighbor& neighbor) {
  int matches = 0;
  for (const auto& seam : topology.seams) {
    if (seam.block_a == belt_block_id && seam.side_a == belt_side) {
      neighbor = {seam.block_b, seam.side_b};
      ++matches;
    }
    if (seam.block_b == belt_block_id && seam.side_b == belt_side) {
      neighbor = {seam.block_a, seam.side_a};
      ++matches;
    }
  }
  return matches == 1;
}

std::vector<int> axis_core_seam_adjacent_row(
    const mesh::MultiBlockTopology& topology,
    const AxisCoreRadialNeighbor& neighbor) {
  if (neighbor.block_id < 0 ||
      neighbor.block_id >= static_cast<int>(topology.blocks.size())) {
    return {};
  }
  const auto& block =
      topology.blocks[static_cast<std::size_t>(neighbor.block_id)];
  if (block.n_i_cells <= 0 || block.n_j_cells <= 0 ||
      block.cell_count != block.n_i_cells * block.n_j_cells ||
      (neighbor.side != mesh::BlockSide::I_MINUS &&
       neighbor.side != mesh::BlockSide::I_PLUS)) {
    return {};
  }
  const int row = neighbor.side == mesh::BlockSide::I_MINUS
                      ? 0
                      : block.n_i_cells - 1;
  const int begin = block.cell_begin + row * block.n_j_cells;
  std::vector<int> cells;
  cells.reserve(static_cast<std::size_t>(block.n_j_cells));
  for (int j = 0; j < block.n_j_cells; ++j) {
    cells.push_back(begin + j);
  }
  return cells;
}

std::vector<AxisCoreJunctionCohort> make_axis_core_junction_cohorts(
    const core::State& state,
    const std::vector<int>& plateau_cells) {
  const auto& topology = *state.mesh.topo.multiblock;
  std::vector<std::uint8_t> in_plateau(
      static_cast<std::size_t>(state.mesh.topo.n_cells), 0U);
  for (const int cell : plateau_cells) {
    if (cell >= 0 && cell < state.mesh.topo.n_cells) {
      in_plateau[static_cast<std::size_t>(cell)] = 1U;
    }
  }

  std::vector<AxisCoreJunctionCohort> cohorts;
  for (int block_id = 0;
       block_id < static_cast<int>(topology.blocks.size());
       ++block_id) {
    const auto& belt = topology.blocks[static_cast<std::size_t>(block_id)];
    if (belt.role != mesh::BlockRole::TRANSITION_BELT ||
        belt.cell_count <= 0 ||
        in_plateau[static_cast<std::size_t>(belt.cell_begin)] == 0U) {
      continue;
    }
    AxisCoreJunctionCohort cohort;
    cohort.belt_block_id = block_id;
    const bool have_outer = find_axis_core_radial_neighbor(
        topology, block_id, mesh::BlockSide::I_MINUS,
        cohort.radial_neighbors[0]);
    const bool have_inner = find_axis_core_radial_neighbor(
        topology, block_id, mesh::BlockSide::I_PLUS,
        cohort.radial_neighbors[1]);
    if (!have_outer || !have_inner) {
      continue;
    }
    for (int cell = belt.cell_begin;
         cell < belt.cell_begin + belt.cell_count;
         ++cell) {
      cohort.cells.push_back(cell);
    }
    for (const auto& neighbor : cohort.radial_neighbors) {
      const std::vector<int> row =
          axis_core_seam_adjacent_row(topology, neighbor);
      cohort.cells.insert(cohort.cells.end(), row.begin(), row.end());
    }
    sort_unique(cohort.cells);
    cohorts.push_back(std::move(cohort));
  }
  return cohorts;
}

double axis_core_transition_cell_centroid_radius(
    const mesh::MultiBlockTopology& topology,
    const std::vector<std::uint8_t>& cell_nverts,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const int cell) {
  const int nverts = static_cast<int>(
      cell_nverts[static_cast<std::size_t>(cell)]);
  const int offset = topology.cell_node_csr_offsets[
      static_cast<std::size_t>(cell)];
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  for (int local = 0; local < nverts; ++local) {
    const int node = topology.cell_node_csr_indices[
        static_cast<std::size_t>(offset + local)];
    centroid_r += node_r[static_cast<std::size_t>(node)];
    centroid_z += node_z[static_cast<std::size_t>(node)];
  }
  const double inv_nverts = 1.0 / static_cast<double>(nverts);
  return std::hypot(centroid_r * inv_nverts,
                    centroid_z * inv_nverts);
}

double axis_core_transition_cell_radial_width(
    const mesh::MultiBlockTopology& topology,
    const std::vector<std::uint8_t>& cell_nverts,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const int cell) {
  const int nverts = static_cast<int>(
      cell_nverts[static_cast<std::size_t>(cell)]);
  const int offset = topology.cell_node_csr_offsets[
      static_cast<std::size_t>(cell)];
  double radius_min = std::numeric_limits<double>::infinity();
  double radius_max = 0.0;
  for (int local = 0; local < nverts; ++local) {
    const int node = topology.cell_node_csr_indices[
        static_cast<std::size_t>(offset + local)];
    const double radius = std::hypot(
        node_r[static_cast<std::size_t>(node)],
        node_z[static_cast<std::size_t>(node)]);
    radius_min = std::min(radius_min, radius);
    radius_max = std::max(radius_max, radius);
  }
  return radius_max - radius_min;
}

std::vector<int> axis_core_transition_inward_sensor_rows(
    const mesh::MultiBlockTopology& topology,
    const AxisCoreRadialNeighbor& inner_neighbor) {
  TENRYU_ASSERT(
      inner_neighbor.block_id >= 0 &&
          inner_neighbor.block_id < static_cast<int>(topology.blocks.size()),
      "transition-passage inner neighbor block is out of range");
  const auto& block = topology.blocks[
      static_cast<std::size_t>(inner_neighbor.block_id)];
  TENRYU_ASSERT(
      block.n_i_cells > 0 && block.n_j_cells > 0 &&
          block.cell_count == block.n_i_cells * block.n_j_cells &&
          (inner_neighbor.side == mesh::BlockSide::I_MINUS ||
           inner_neighbor.side == mesh::BlockSide::I_PLUS),
      "transition-passage inner sensor block is not radial structured");
  const int row_count = std::min(8, block.n_i_cells);
  std::vector<int> cells;
  cells.reserve(static_cast<std::size_t>(row_count * block.n_j_cells));
  for (int depth = 0; depth < row_count; ++depth) {
    const int row = inner_neighbor.side == mesh::BlockSide::I_MINUS
                        ? depth
                        : block.n_i_cells - 1 - depth;
    const int begin = block.cell_begin + row * block.n_j_cells;
    for (int column = 0; column < block.n_j_cells; ++column) {
      cells.push_back(begin + column);
    }
  }
  sort_unique(cells);
  return cells;
}

void build_axis_core_transition_passage(
    const core::State& state,
    const mesh::MultiBlockTopology& topology,
    AxisCoreDiskRuntime& runtime) {
  const int n_cells = runtime.n_cells;
  const int n_nodes = runtime.n_nodes;
  const std::vector<std::uint8_t> cell_nverts =
      state.mesh.cell_nverts.empty()
          ? std::vector<std::uint8_t>(
                static_cast<std::size_t>(n_cells), 4U)
          : state.mesh.cell_nverts;
  TENRYU_ASSERT(
      cell_nverts.size() == static_cast<std::size_t>(n_cells) &&
          topology.cell_node_csr_offsets.size() ==
              static_cast<std::size_t>(n_cells) + 1U &&
          topology.face_adj_csr_offsets.size() ==
              static_cast<std::size_t>(n_cells) + 1U &&
          topology.face_adj_csr_indices.size() ==
              topology.cell_node_csr_indices.size() &&
          topology.cell_orientation_sign.size() ==
              static_cast<std::size_t>(n_cells) &&
          state.mesh.multiblock_cell_node_csr_offsets.size() ==
              topology.cell_node_csr_offsets.size() &&
          state.mesh.multiblock_cell_node_csr_indices.size() ==
              topology.cell_node_csr_indices.size(),
      "transition-passage requires complete host/device CSR metadata");
  const std::vector<AxisCoreJunctionCohort> cohorts =
      make_axis_core_junction_cohorts(state, runtime.plateau_cells);
  std::vector<std::uint8_t> in_plateau(
      static_cast<std::size_t>(n_cells), 0U);
  for (const int cell : runtime.plateau_cells) {
    in_plateau[static_cast<std::size_t>(cell)] = 1U;
  }
  std::size_t expected_patch_count = 0U;
  for (const auto& block : topology.blocks) {
    if (block.role == mesh::BlockRole::TRANSITION_BELT &&
        block.cell_count > 0 &&
        in_plateau[static_cast<std::size_t>(block.cell_begin)] != 0U) {
      ++expected_patch_count;
    }
  }
  if (cohorts.empty() || cohorts.size() != expected_patch_count) {
    throw core::namelist::ConfigError(
        "axis_core_transition_passage_enabled=true requires a complete "
        "independent patch for every TRANSITION_BELT inside the axis-core "
        "plateau");
  }

  std::vector<std::vector<int>> incident_cells(
      static_cast<std::size_t>(n_nodes));
  for (int cell = 0; cell < n_cells; ++cell) {
    const int nverts = static_cast<int>(
        cell_nverts[static_cast<std::size_t>(cell)]);
    const int offset = topology.cell_node_csr_offsets[
        static_cast<std::size_t>(cell)];
    for (int local = 0; local < nverts; ++local) {
      const int node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(offset + local)];
      incident_cells[static_cast<std::size_t>(node)].push_back(cell);
    }
  }
  for (auto& cells : incident_cells) {
    sort_unique(cells);
  }

  std::vector<AxisCoreTransitionPatchCandidate> candidates;
  candidates.reserve(cohorts.size());
  for (const auto& cohort : cohorts) {
    AxisCoreTransitionPatchCandidate candidate;
    candidate.belt_block_id = cohort.belt_block_id;
    candidate.cells = cohort.cells;
    sort_unique(candidate.cells);
    candidate.inner_radius = std::numeric_limits<double>::infinity();
    for (const int cell : candidate.cells) {
      TENRYU_ASSERT(
          cell >= 0 && cell < n_cells,
          "transition-passage patch cell is out of range");
      candidate.inner_radius = std::min(
          candidate.inner_radius,
          axis_core_transition_cell_centroid_radius(
              topology,
              cell_nverts,
              runtime.captured_r,
              runtime.captured_z,
              cell));
    }

    const std::vector<int> inner_row = axis_core_seam_adjacent_row(
        topology, cohort.radial_neighbors[1]);
    TENRYU_ASSERT(!inner_row.empty(),
                  "transition-passage inner patch row is empty");
    std::vector<double> inner_widths;
    inner_widths.reserve(inner_row.size());
    for (const int cell : inner_row) {
      inner_widths.push_back(axis_core_transition_cell_radial_width(
          topology,
          cell_nverts,
          runtime.captured_r,
          runtime.captured_z,
          cell));
    }
    std::sort(inner_widths.begin(), inner_widths.end());
    const std::size_t middle = inner_widths.size() / 2U;
    candidate.median_inner_width =
        (inner_widths.size() & 1U) != 0U
            ? inner_widths[middle]
            : 0.5 * (inner_widths[middle - 1U] + inner_widths[middle]);
    TENRYU_ASSERT(
        std::isfinite(candidate.inner_radius) &&
            candidate.inner_radius > 0.0 &&
            std::isfinite(candidate.median_inner_width) &&
            candidate.median_inner_width > 0.0,
        "transition-passage captured patch radial scale is invalid");

    candidate.sensor_cells = candidate.cells;
    const std::vector<int> inward_sensor =
        axis_core_transition_inward_sensor_rows(
            topology, cohort.radial_neighbors[1]);
    candidate.sensor_cells.insert(candidate.sensor_cells.end(),
                                  inward_sensor.begin(),
                                  inward_sensor.end());
    sort_unique(candidate.sensor_cells);
    candidates.push_back(std::move(candidate));
  }

  std::vector<int> cell_owner(static_cast<std::size_t>(n_cells), -1);
  std::vector<int> patch_parent(candidates.size(), -1);
  for (int patch_index = 0;
       patch_index < static_cast<int>(candidates.size());
       ++patch_index) {
    patch_parent[static_cast<std::size_t>(patch_index)] = patch_index;
  }
  const auto find_patch_root = [&](int patch_index) {
    while (patch_parent[static_cast<std::size_t>(patch_index)] !=
           patch_index) {
      patch_parent[static_cast<std::size_t>(patch_index)] =
          patch_parent[static_cast<std::size_t>(
              patch_parent[static_cast<std::size_t>(patch_index)])];
      patch_index = patch_parent[static_cast<std::size_t>(patch_index)];
    }
    return patch_index;
  };
  for (int patch_index = 0;
       patch_index < static_cast<int>(candidates.size());
       ++patch_index) {
    for (const int cell :
         candidates[static_cast<std::size_t>(patch_index)].cells) {
      const int owner = cell_owner[static_cast<std::size_t>(cell)];
      if (owner < 0) {
        cell_owner[static_cast<std::size_t>(cell)] = patch_index;
        continue;
      }
      const int owner_root = find_patch_root(owner);
      const int patch_root = find_patch_root(patch_index);
      const int merged_root = std::min(owner_root, patch_root);
      patch_parent[static_cast<std::size_t>(owner_root)] = merged_root;
      patch_parent[static_cast<std::size_t>(patch_root)] = merged_root;
    }
  }

  std::vector<AxisCoreTransitionPatchCandidate> merged_candidates;
  std::vector<int> root_to_merged(candidates.size(), -1);
  merged_candidates.reserve(candidates.size());
  for (int patch_index = 0;
       patch_index < static_cast<int>(candidates.size());
       ++patch_index) {
    const int root = find_patch_root(patch_index);
    int& merged_index = root_to_merged[static_cast<std::size_t>(root)];
    const auto& candidate = candidates[static_cast<std::size_t>(patch_index)];
    if (merged_index < 0) {
      merged_index = static_cast<int>(merged_candidates.size());
      merged_candidates.push_back(candidate);
      continue;
    }
    auto& merged = merged_candidates[static_cast<std::size_t>(merged_index)];
    merged.inner_radius = std::min(merged.inner_radius,
                                   candidate.inner_radius);
    merged.median_inner_width = std::min(
        merged.median_inner_width, candidate.median_inner_width);
    merged.cells.insert(merged.cells.end(),
                        candidate.cells.begin(), candidate.cells.end());
    merged.sensor_cells.insert(merged.sensor_cells.end(),
                               candidate.sensor_cells.begin(),
                               candidate.sensor_cells.end());
  }
  for (auto& candidate : merged_candidates) {
    sort_unique(candidate.cells);
    sort_unique(candidate.sensor_cells);
  }
  const std::size_t merged_overlap_count =
      candidates.size() - merged_candidates.size();

  runtime.transition_patches.reserve(merged_candidates.size());
  for (const auto& candidate : merged_candidates) {
    AxisCoreTransitionPatch patch;
    patch.belt_block_id = candidate.belt_block_id;
    patch.inner_radius = candidate.inner_radius;
    patch.median_inner_width = candidate.median_inner_width;
    patch.cells = candidate.cells;
    patch.sensor_cells = candidate.sensor_cells;
    std::vector<std::uint8_t> in_patch(
        static_cast<std::size_t>(n_cells), 0U);
    for (const int cell : patch.cells) {
      TENRYU_ASSERT(
          cell >= 0 && cell < n_cells,
          "transition-passage patch cell is out of range");
      in_patch[static_cast<std::size_t>(cell)] = 1U;
      const int nverts = static_cast<int>(
          cell_nverts[static_cast<std::size_t>(cell)]);
      const int offset = topology.cell_node_csr_offsets[
          static_cast<std::size_t>(cell)];
      for (int local = 0; local < nverts; ++local) {
        patch.nodes.push_back(topology.cell_node_csr_indices[
            static_cast<std::size_t>(offset + local)]);
      }
    }
    sort_unique(patch.nodes);

    std::vector<int> distance(static_cast<std::size_t>(n_nodes), -1);
    for (const int node : patch.nodes) {
      bool boundary = false;
      for (const int incident :
           incident_cells[static_cast<std::size_t>(node)]) {
        if (in_patch[static_cast<std::size_t>(incident)] == 0U) {
          boundary = true;
          break;
        }
      }
      if (boundary) {
        distance[static_cast<std::size_t>(node)] = 0;
      }
    }
    for (int ring = 0; ring < 2; ++ring) {
      std::vector<int> next_distance = distance;
      for (const int cell : patch.cells) {
        const int nverts = static_cast<int>(
            cell_nverts[static_cast<std::size_t>(cell)]);
        const int offset = topology.cell_node_csr_offsets[
            static_cast<std::size_t>(cell)];
        for (int local = 0; local < nverts; ++local) {
          const int next = local + 1 == nverts ? 0 : local + 1;
          const int node0 = topology.cell_node_csr_indices[
              static_cast<std::size_t>(offset + local)];
          const int node1 = topology.cell_node_csr_indices[
              static_cast<std::size_t>(offset + next)];
          if (distance[static_cast<std::size_t>(node0)] == ring &&
              next_distance[static_cast<std::size_t>(node1)] < 0) {
            next_distance[static_cast<std::size_t>(node1)] = ring + 1;
          }
          if (distance[static_cast<std::size_t>(node1)] == ring &&
              next_distance[static_cast<std::size_t>(node0)] < 0) {
            next_distance[static_cast<std::size_t>(node0)] = ring + 1;
          }
        }
      }
      distance.swap(next_distance);
    }
    patch.feather.reserve(patch.nodes.size());
    for (const int node : patch.nodes) {
      const int ring = distance[static_cast<std::size_t>(node)];
      patch.feather.push_back(ring == 0 ? 0.0 : (ring == 1 ? 0.5 : 1.0));
    }

    std::map<int, int> patch_cell_local;
    for (int local_cell = 0;
         local_cell < static_cast<int>(patch.cells.size());
         ++local_cell) {
      patch_cell_local.emplace(
          patch.cells[static_cast<std::size_t>(local_cell)], local_cell);
    }
    std::vector<int> node_cell_offsets(patch.nodes.size() + 1U, 0);
    std::vector<int> node_cells;
    for (std::size_t local_node = 0;
         local_node < patch.nodes.size();
         ++local_node) {
      const int node = patch.nodes[local_node];
      for (const int incident :
           incident_cells[static_cast<std::size_t>(node)]) {
        const auto found = patch_cell_local.find(incident);
        if (found != patch_cell_local.end()) {
          node_cells.push_back(found->second);
        }
      }
      TENRYU_ASSERT(
          node_cells.size() >
              static_cast<std::size_t>(node_cell_offsets[local_node]),
          "transition-passage patch node has no incident patch cell");
      node_cell_offsets[local_node + 1U] =
          static_cast<int>(node_cells.size());
    }

    for (const int cell : patch.cells) {
      const int nverts = static_cast<int>(
          cell_nverts[static_cast<std::size_t>(cell)]);
      const int face_offset = topology.face_adj_csr_offsets[
          static_cast<std::size_t>(cell)];
      const int face_slot_count = topology.face_adj_csr_offsets[
          static_cast<std::size_t>(cell) + 1U] - face_offset;
      for (int local = 0; local < face_slot_count; ++local) {
        if (!mesh::mesh_topo_local_face_is_active(nverts, local)) {
          continue;
        }
        const int neighbor = topology.face_adj_csr_indices[
            static_cast<std::size_t>(face_offset + local)];
        if (neighbor < 0 || neighbor >= n_cells) {
          continue;
        }
        if (in_patch[static_cast<std::size_t>(neighbor)] != 0U &&
            cell > neighbor) {
          continue;
        }
        patch.swept_cells.push_back(cell);
        patch.swept_local_faces.push_back(local);
        patch.swept_neighbors.push_back(neighbor);
      }
    }
    TENRYU_ASSERT(
        !patch.swept_cells.empty() &&
            patch.swept_cells.size() == patch.swept_local_faces.size() &&
            patch.swept_cells.size() == patch.swept_neighbors.size(),
        "transition-passage patch swept-face list is invalid");

    patch.optimal_r.resize(patch.nodes.size(), 0.0);
    patch.optimal_z.resize(patch.nodes.size(), 0.0);
    patch.d_cells.reset(patch.cells.size());
    patch.d_cells.copy_from_host(patch.cells);
    patch.d_nodes.reset(patch.nodes.size());
    patch.d_nodes.copy_from_host(patch.nodes);
    patch.d_node_cell_offsets.reset(node_cell_offsets.size());
    patch.d_node_cell_offsets.copy_from_host(node_cell_offsets);
    patch.d_node_cells.reset(node_cells.size());
    patch.d_node_cells.copy_from_host(node_cells);
    patch.d_feather.reset(patch.feather.size());
    patch.d_feather.copy_from_host(patch.feather);
    std::vector<std::uint8_t> inactive_cell_mask(
        static_cast<std::size_t>(n_cells), 1U);
    for (const int cell : patch.cells) {
      inactive_cell_mask[static_cast<std::size_t>(cell)] = 0U;
    }
    patch.d_inactive_cell_mask.reset(inactive_cell_mask.size());
    patch.d_inactive_cell_mask.copy_from_host(inactive_cell_mask);
    patch.d_sensor_cells.reset(patch.sensor_cells.size());
    patch.d_sensor_cells.copy_from_host(patch.sensor_cells);
    patch.d_swept_cells.reset(patch.swept_cells.size());
    patch.d_swept_cells.copy_from_host(patch.swept_cells);
    patch.d_swept_local_faces.reset(patch.swept_local_faces.size());
    patch.d_swept_local_faces.copy_from_host(patch.swept_local_faces);
    patch.d_swept_neighbors.reset(patch.swept_neighbors.size());
    patch.d_swept_neighbors.copy_from_host(patch.swept_neighbors);
    patch.d_cell_center_r.reset(patch.cells.size());
    patch.d_cell_center_z.reset(patch.cells.size());
    patch.d_sensor_strength.reset(patch.sensor_cells.size());
    patch.d_sensor_radius.reset(patch.sensor_cells.size());
    patch.d_swept_fraction.reset(patch.swept_cells.size());
    runtime.transition_patches.push_back(std::move(patch));
  }

  int central_block_id = -1;
  double central_block_radius = std::numeric_limits<double>::infinity();
  for (int block_id = 0;
       block_id < static_cast<int>(topology.blocks.size());
       ++block_id) {
    const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
    if (block.role != mesh::BlockRole::POLAR_TIER || block.cell_count <= 0) {
      continue;
    }
    double radius_sum = 0.0;
    for (int cell = block.cell_begin;
         cell < block.cell_begin + block.cell_count;
         ++cell) {
      radius_sum += axis_core_transition_cell_centroid_radius(
          topology,
          cell_nverts,
          runtime.captured_r,
          runtime.captured_z,
          cell);
    }
    const double mean_radius =
        radius_sum / static_cast<double>(block.cell_count);
    if (mean_radius < central_block_radius) {
      central_block_radius = mean_radius;
      central_block_id = block_id;
    }
  }
  if (central_block_id < 0) {
    throw core::namelist::ConfigError(
        "axis_core_transition_passage_enabled=true requires a POLAR_TIER "
        "block for the global central-outflow ARM");
  }
  const auto& central_block =
      topology.blocks[static_cast<std::size_t>(central_block_id)];
  for (int cell = central_block.cell_begin;
       cell < central_block.cell_begin + central_block.cell_count;
       ++cell) {
    runtime.transition_central_cells.push_back(cell);
  }

  runtime.d_transition_central_cells.reset(
      runtime.transition_central_cells.size());
  runtime.d_transition_central_cells.copy_from_host(
      runtime.transition_central_cells);
  runtime.d_transition_central_mass_ur.reset(
      runtime.transition_central_cells.size());
  runtime.d_transition_central_mass_cs.reset(
      runtime.transition_central_cells.size());
  runtime.d_transition_central_mass.reset(
      runtime.transition_central_cells.size());
  runtime.d_transition_face_adj_offsets.reset(
      topology.face_adj_csr_offsets.size());
  runtime.d_transition_face_adj_offsets.copy_from_host(
      topology.face_adj_csr_offsets);
  runtime.d_transition_face_adj_indices.reset(
      topology.face_adj_csr_indices.size());
  runtime.d_transition_face_adj_indices.copy_from_host(
      topology.face_adj_csr_indices);
  runtime.d_transition_cell_orientation_sign.reset(
      topology.cell_orientation_sign.size());
  runtime.d_transition_cell_orientation_sign.copy_from_host(
      topology.cell_orientation_sign);
  runtime.d_transition_cell_nverts.reset(cell_nverts.size());
  runtime.d_transition_cell_nverts.copy_from_host(cell_nverts);
  runtime.d_transition_jacobi_r_a.reset(static_cast<std::size_t>(n_nodes));
  runtime.d_transition_jacobi_z_a.reset(static_cast<std::size_t>(n_nodes));
  runtime.d_transition_jacobi_r_b.reset(static_cast<std::size_t>(n_nodes));
  runtime.d_transition_jacobi_z_b.reset(static_cast<std::size_t>(n_nodes));
  runtime.d_transition_candidate_r.reset(static_cast<std::size_t>(n_nodes));
  runtime.d_transition_candidate_z.reset(static_cast<std::size_t>(n_nodes));

  std::ostringstream message;
  message << "[transition-passage] patches="
          << runtime.transition_patches.size()
          << " (merged " << merged_overlap_count << " overlapping)"
          << " central_block=" << central_block_id;
  core::log_info(message.str());
}

std::vector<AxisCoreJunctionLedgerCell>
make_axis_core_junction_ledger_cells(
    const mesh::MultiBlockTopology& topology,
    const std::vector<AxisCoreJunctionCohort>& cohorts) {
  std::vector<AxisCoreJunctionLedgerCell> entries;
  const auto append_equator_cells =
      [&](const int belt_block_id,
          const mesh::BlockInfo& block,
          const int row) {
        if (block.n_i_cells <= 0 || block.n_j_cells < 2 ||
            row < 0 || row >= block.n_i_cells) {
          return;
        }
        const int j_mid = block.n_j_cells / 2;
        const int row_begin =
            block.cell_begin + row * block.n_j_cells;
        entries.push_back({belt_block_id, row_begin + j_mid - 1});
        entries.push_back({belt_block_id, row_begin + j_mid});
      };

  for (const auto& cohort : cohorts) {
    const auto& belt = topology.blocks[
        static_cast<std::size_t>(cohort.belt_block_id)];
    append_equator_cells(cohort.belt_block_id, belt, 0);
    for (const auto& neighbor : cohort.radial_neighbors) {
      if (neighbor.block_id < 0 ||
          neighbor.block_id >= static_cast<int>(topology.blocks.size())) {
        continue;
      }
      const auto& block = topology.blocks[
          static_cast<std::size_t>(neighbor.block_id)];
      const int row = neighbor.side == mesh::BlockSide::I_MINUS
                          ? 0
                          : block.n_i_cells - 1;
      append_equator_cells(cohort.belt_block_id, block, row);
    }
  }
  return entries;
}

struct AxisCoreAuditFaceSide {
  int cell = -1;
  int local = -1;
  int node_a = -1;
  int node_b = -1;
};

int axis_core_audit_polygon_nverts(
    const std::vector<int>& node_indices,
    const int offset,
    const int stored_nverts) {
  // CENTER_FAN triangles use four storage slots with the apex repeated in
  // the last slot. Audit their three geometric vertices and three real faces.
  if (stored_nverts == 4 &&
      node_indices[static_cast<std::size_t>(offset)] ==
          node_indices[static_cast<std::size_t>(offset + 3)]) {
    return 3;
  }
  return stored_nverts;
}

void run_axis_core_junction_audit(
    const core::State& state,
    const std::vector<double>& captured_r,
    const std::vector<double>& captured_z,
    const std::vector<AxisCoreJunctionCohort>& cohorts) {
  const auto& topology = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  const auto& node_offsets = topology.cell_node_csr_offsets;
  const auto& node_indices = topology.cell_node_csr_indices;
  const auto& face_offsets = topology.face_adj_csr_offsets;
  const auto& face_indices = topology.face_adj_csr_indices;
  const auto& cell_nverts = state.mesh.cell_nverts;

  std::vector<int> cohort_cells;
  for (const auto& cohort : cohorts) {
    cohort_cells.insert(
        cohort_cells.end(), cohort.cells.begin(), cohort.cells.end());
  }
  sort_unique(cohort_cells);
  std::vector<std::uint8_t> in_cohort(
      static_cast<std::size_t>(n_cells), 0U);
  for (const int cell : cohort_cells) {
    if (cell >= 0 && cell < n_cells) {
      in_cohort[static_cast<std::size_t>(cell)] = 1U;
    }
  }

  std::map<std::pair<int, int>, std::vector<AxisCoreAuditFaceSide>> owners;
  for (int cell = 0; cell < n_cells; ++cell) {
    if (static_cast<std::size_t>(cell) >= cell_nverts.size() ||
        static_cast<std::size_t>(cell + 1) >= node_offsets.size()) {
      continue;
    }
    const int stored_nverts = static_cast<int>(
        cell_nverts[static_cast<std::size_t>(cell)]);
    const int offset = node_offsets[static_cast<std::size_t>(cell)];
    const int node_end = node_offsets[static_cast<std::size_t>(cell + 1)];
    if (stored_nverts < 3 || stored_nverts > state.corner_stride ||
        offset < 0 || node_end < offset ||
        static_cast<std::size_t>(node_end) > node_indices.size() ||
        stored_nverts > node_end - offset) {
      continue;
    }
    const int polygon_nverts = axis_core_audit_polygon_nverts(
        node_indices, offset, stored_nverts);
    for (int local = 0; local < state.corner_stride; ++local) {
      int corner_a = -1;
      int corner_b = -1;
      if (!mesh::mesh_topo_active_local_face_corners(
              polygon_nverts, local, &corner_a, &corner_b)) {
        continue;
      }
      const int node_a = node_indices[
          static_cast<std::size_t>(offset + corner_a)];
      const int node_b = node_indices[
          static_cast<std::size_t>(offset + corner_b)];
      if (node_a == node_b) {
        continue;
      }
      owners[std::minmax(node_a, node_b)].push_back(
          {cell, local, node_a, node_b});
    }
  }

  std::size_t failures = 0U;
  std::size_t asymmetric_pairs = 0U;
  const auto log_failure = [&](const std::string& detail) {
    ++failures;
    core::log_warning("[junction-audit] failure " + detail);
  };
  std::set<std::pair<int, int>> audited_internal_faces;
  std::set<std::pair<int, int>> audited_mirror_pairs;

  for (const int cell : cohort_cells) {
    if (cell < 0 || cell >= n_cells ||
        static_cast<std::size_t>(cell) >= cell_nverts.size() ||
        static_cast<std::size_t>(cell + 1) >= node_offsets.size()) {
      log_failure("kind=cell-range cell=" + std::to_string(cell));
      continue;
    }
    const std::size_t cell_index = static_cast<std::size_t>(cell);
    const int stored_nverts = static_cast<int>(cell_nverts[cell_index]);
    const int offset = node_offsets[cell_index];
    const int node_end = node_offsets[cell_index + 1U];
    if (stored_nverts < 3 || stored_nverts > state.corner_stride ||
        offset < 0 || node_end < offset ||
        static_cast<std::size_t>(node_end) > node_indices.size() ||
        stored_nverts > node_end - offset) {
      log_failure("kind=cell-cycle cell=" + std::to_string(cell) +
                  " nverts=" + std::to_string(stored_nverts));
      continue;
    }
    const int polygon_nverts = axis_core_audit_polygon_nverts(
        node_indices, offset, stored_nverts);

    std::set<int> unique_nodes;
    std::set<std::pair<int, int>> unique_faces;
    std::set<int> unique_neighbors;
    int boundary_faces = 0;
    double twice_area = 0.0;
    double closure_r = 0.0;
    double closure_z = 0.0;
    double perimeter = 0.0;
    const bool have_face_row =
        face_offsets.size() == static_cast<std::size_t>(n_cells) + 1U;
    const int face_begin = have_face_row ? face_offsets[cell_index] : 0;
    const int face_end = have_face_row ? face_offsets[cell_index + 1U] : 0;
    if (!have_face_row || face_begin < 0 || face_end < face_begin ||
        static_cast<std::size_t>(face_end) > face_indices.size() ||
        face_end - face_begin != state.corner_stride) {
      log_failure("kind=face-csr-row cell=" + std::to_string(cell));
    }

    for (int local = 0; local < polygon_nverts; ++local) {
      const int node_a = node_indices[
          static_cast<std::size_t>(offset + local)];
      const int node_b = node_indices[
          static_cast<std::size_t>(
              offset + (local + 1) % polygon_nverts)];
      if (!unique_nodes.insert(node_a).second) {
        log_failure("kind=duplicate-node cell=" + std::to_string(cell) +
                    " node=" + std::to_string(node_a));
      }
      if (node_a < 0 || node_a >= n_nodes ||
          node_b < 0 || node_b >= n_nodes ||
          static_cast<std::size_t>(node_a) >= captured_r.size() ||
          static_cast<std::size_t>(node_a) >= captured_z.size() ||
          static_cast<std::size_t>(node_b) >= captured_r.size() ||
          static_cast<std::size_t>(node_b) >= captured_z.size()) {
        log_failure("kind=node-range cell=" + std::to_string(cell) +
                    " local=" + std::to_string(local) +
                    " node_a=" + std::to_string(node_a) +
                    " node_b=" + std::to_string(node_b));
        continue;
      }
      const double r_a = captured_r[static_cast<std::size_t>(node_a)];
      const double z_a = captured_z[static_cast<std::size_t>(node_a)];
      const double r_b = captured_r[static_cast<std::size_t>(node_b)];
      const double z_b = captured_z[static_cast<std::size_t>(node_b)];
      twice_area += r_a * z_b - r_b * z_a;
      const double dr = r_b - r_a;
      const double dz = z_b - z_a;
      const double edge_length = std::hypot(dr, dz);
      const int orientation =
          cell_index < topology.cell_orientation_sign.size()
              ? topology.cell_orientation_sign[cell_index]
              : 0;
      closure_r += static_cast<double>(orientation) * dz;
      closure_z -= static_cast<double>(orientation) * dr;
      perimeter += edge_length;
    }

    for (int local = 0; local < state.corner_stride; ++local) {
      int corner_a = -1;
      int corner_b = -1;
      if (!mesh::mesh_topo_active_local_face_corners(
              polygon_nverts, local, &corner_a, &corner_b)) {
        continue;
      }
      const int node_a = node_indices[
          static_cast<std::size_t>(offset + corner_a)];
      const int node_b = node_indices[
          static_cast<std::size_t>(offset + corner_b)];
      if (node_a == node_b) {
        log_failure("kind=degenerate-face cell=" + std::to_string(cell) +
                    " local=" + std::to_string(local) +
                    " node=" + std::to_string(node_a));
        continue;
      }
      const auto face_key = std::minmax(node_a, node_b);
      unique_faces.insert(face_key);

      int neighbor = -2;
      if (have_face_row && face_begin + local < face_end &&
          static_cast<std::size_t>(face_begin + local) <
              face_indices.size()) {
        neighbor = face_indices[
            static_cast<std::size_t>(face_begin + local)];
        if (neighbor >= 0) {
          unique_neighbors.insert(neighbor);
        } else {
          ++boundary_faces;
        }
      }
      const auto owner_it = owners.find(face_key);
      const std::size_t owner_count =
          owner_it == owners.end() ? 0U : owner_it->second.size();
      const bool internal = neighbor >= 0 || owner_count > 1U;
      if (internal && audited_internal_faces.insert(face_key).second) {
        if (owner_count != 2U) {
          log_failure("kind=face-owner-count cell=" +
                      std::to_string(cell) +
                      " local=" + std::to_string(local) +
                      " node_a=" + std::to_string(node_a) +
                      " node_b=" + std::to_string(node_b) +
                      " owners=" + std::to_string(owner_count));
        } else {
          const auto& side_a = owner_it->second[0];
          const auto& side_b = owner_it->second[1];
          if (side_a.node_a != side_b.node_b ||
              side_a.node_b != side_b.node_a) {
            log_failure("kind=face-orientation cell_a=" +
                        std::to_string(side_a.cell) +
                        " local_a=" + std::to_string(side_a.local) +
                        " cell_b=" + std::to_string(side_b.cell) +
                        " local_b=" + std::to_string(side_b.local));
          }
          for (const auto* side : {&side_a, &side_b}) {
            const int other = side->cell == side_a.cell
                                  ? side_b.cell
                                  : side_a.cell;
            int csr_neighbor = -2;
            if (side->cell >= 0 && side->cell < n_cells &&
                face_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U) {
              const int side_begin = face_offsets[
                  static_cast<std::size_t>(side->cell)];
              const int side_end = face_offsets[
                  static_cast<std::size_t>(side->cell) + 1U];
              if (side_begin >= 0 && side_end >= side_begin &&
                  static_cast<std::size_t>(side_end) <=
                      face_indices.size() &&
                  side->local >= 0 &&
                  side->local < side_end - side_begin) {
                csr_neighbor = face_indices[static_cast<std::size_t>(
                    side_begin + side->local)];
              }
            }
            if (csr_neighbor != other) {
              log_failure("kind=face-adjacency cell=" +
                          std::to_string(side->cell) +
                          " local=" + std::to_string(side->local) +
                          " neighbor=" + std::to_string(csr_neighbor) +
                          " owner=" + std::to_string(other));
            }
          }
        }
      }
    }

    const int orientation =
        cell_index < topology.cell_orientation_sign.size()
            ? topology.cell_orientation_sign[cell_index]
            : 0;
    if (orientation * twice_area <= 0.0) {
      log_failure("kind=planar-area cell=" + std::to_string(cell) +
                  " orientation=" + std::to_string(orientation));
    }
    const double closure_rel =
        perimeter > 0.0
            ? std::hypot(closure_r, closure_z) / perimeter
            : std::numeric_limits<double>::infinity();
    if (!(closure_rel <= 1.0e-12)) {
      std::ostringstream detail;
      detail << std::scientific << std::setprecision(17)
             << "kind=closure cell=" << cell
             << " relative=" << closure_rel;
      log_failure(detail.str());
    }
    if (stored_nverts == 5 &&
        (unique_faces.size() != 5U ||
         unique_neighbors.size() +
                 static_cast<std::size_t>(boundary_faces) !=
             5U)) {
      log_failure("kind=pentagon-topology cell=" +
                  std::to_string(cell) +
                  " faces=" + std::to_string(unique_faces.size()) +
                  " neighbors=" +
                  std::to_string(unique_neighbors.size()) +
                  " boundary=" + std::to_string(boundary_faces));
    }

    if (cell_index >= topology.cell_block_id.size()) {
      log_failure("kind=mirror-block cell=" + std::to_string(cell));
      continue;
    }
    const int block_id = topology.cell_block_id[cell_index];
    if (block_id < 0 ||
        block_id >= static_cast<int>(topology.blocks.size())) {
      log_failure("kind=mirror-block cell=" + std::to_string(cell) +
                  " block=" + std::to_string(block_id));
      continue;
    }
    const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
    const int local_cell = cell - block.cell_begin;
    if (block.n_j_cells <= 0 || local_cell < 0 ||
        local_cell >= block.cell_count) {
      log_failure("kind=mirror-index cell=" + std::to_string(cell) +
                  " block=" + std::to_string(block_id));
      continue;
    }
    const int row = local_cell / block.n_j_cells;
    const int column = local_cell % block.n_j_cells;
    const int mirror = block.cell_begin + row * block.n_j_cells +
                       (block.n_j_cells - 1 - column);
    const auto pair = std::minmax(cell, mirror);
    if (!audited_mirror_pairs.insert(pair).second) {
      continue;
    }
    if (mirror < 0 || mirror >= n_cells ||
        in_cohort[static_cast<std::size_t>(mirror)] == 0U ||
        static_cast<std::size_t>(mirror) >= cell_nverts.size() ||
        cell_nverts[static_cast<std::size_t>(mirror)] !=
            cell_nverts[cell_index]) {
      ++asymmetric_pairs;
      log_failure("kind=mirror-asymmetry cell=" + std::to_string(cell) +
                  " mirror=" + std::to_string(mirror) +
                  " block=" + std::to_string(block_id) +
                  " column=" + std::to_string(column));
    }
  }

  core::log_info("[junction-audit] asymmetric_pairs=" +
                 std::to_string(asymmetric_pairs));
  core::log_info("[junction-audit] cohorts=" +
                 std::to_string(cohorts.size()) +
                 " cells=" + std::to_string(cohort_cells.size()) +
                 " failures=" + std::to_string(failures));
}

double ring_mean_radius(const std::vector<int>& nodes,
                        const std::vector<double>& node_r,
                        const std::vector<double>& node_z) {
  TENRYU_ASSERT(!nodes.empty(),
                "axis-core structural unit outer ring is empty");
  double sum = 0.0;
  double compensation = 0.0;
  for (const int node : nodes) {
    const double radius = std::hypot(
        node_r[static_cast<std::size_t>(node)],
        node_z[static_cast<std::size_t>(node)]);
    const double corrected = radius - compensation;
    const double next = sum + corrected;
    compensation = (next - sum) - corrected;
    sum = next;
  }
  return sum / static_cast<double>(nodes.size());
}

void append_unit_nodes(const mesh::MultiBlockTopology& topology,
                       const std::vector<std::uint8_t>& cell_nverts,
                       const int corner_stride,
                       AxisCoreStructuralUnit& unit) {
  for (const int cell : unit.cells) {
    const int nverts = static_cast<int>(
        cell_nverts[static_cast<std::size_t>(cell)]);
    TENRYU_ASSERT(nverts >= 3 && nverts <= corner_stride,
                  "axis-core structural unit cell cycle is invalid");
    const int offset = topology.cell_node_csr_offsets[
        static_cast<std::size_t>(cell)];
    for (int local = 0; local < nverts; ++local) {
      unit.nodes.push_back(
          topology.cell_node_csr_indices[
              static_cast<std::size_t>(offset + local)]);
    }
  }
  sort_unique(unit.nodes);
}

std::vector<AxisCoreStructuralUnit> make_axis_core_structural_units(
    const core::State& state,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  static int non_quad_warning_step = std::numeric_limits<int>::min();
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "axis-core requires multiblock topology");
  const auto& topology = *state.mesh.topo.multiblock;
  const std::vector<std::uint8_t> cell_nverts =
      state.mesh.cell_nverts.empty()
          ? std::vector<std::uint8_t>(
                static_cast<std::size_t>(state.mesh.topo.n_cells), 4U)
          : state.mesh.cell_nverts;
  std::vector<AxisCoreStructuralUnit> units;
  for (int block_id = 0;
       block_id < static_cast<int>(topology.blocks.size());
       ++block_id) {
    const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
    if (block.role == mesh::BlockRole::CENTER_FAN) {
      AxisCoreStructuralUnit unit;
      unit.block_id = block_id;
      unit.radial_row = 0;
      unit.role = block.role;
      for (int cell = block.cell_begin;
           cell < block.cell_begin + block.cell_count;
           ++cell) {
        unit.cells.push_back(cell);
      }
      append_unit_nodes(topology, cell_nverts, state.corner_stride, unit);
      for (const int node : unit.nodes) {
        if (node_r[static_cast<std::size_t>(node)] != 0.0 ||
            node_z[static_cast<std::size_t>(node)] != 0.0) {
          unit.outer_nodes.push_back(node);
        }
      }
      sort_unique(unit.outer_nodes);
      unit.outer_radius =
          ring_mean_radius(unit.outer_nodes, node_r, node_z);
      units.push_back(std::move(unit));
      continue;
    }
    if (block.role == mesh::BlockRole::POLAR_TIER ||
        block.role == mesh::BlockRole::POLAR_SHELL) {
      TENRYU_ASSERT(
          block.n_i_cells > 0 && block.n_j_cells > 0 &&
              block.cell_count == block.n_i_cells * block.n_j_cells,
          "axis-core structured radial block dimensions are invalid");
      for (int row = 0; row < block.n_i_cells; ++row) {
        AxisCoreStructuralUnit unit;
        unit.block_id = block_id;
        unit.radial_row = row;
        unit.role = block.role;
        const int row_begin = block.cell_begin + row * block.n_j_cells;
        for (int column = 0; column < block.n_j_cells; ++column) {
          const int cell = row_begin + column;
          const std::uint8_t nverts =
              cell_nverts[static_cast<std::size_t>(cell)];
          if (nverts != 4U) {
            if (non_quad_warning_step != state.step) {
              core::log_warning(
                  "[euler-window] non-quad cell skipped in structured scan "
                  "cell=" + std::to_string(cell) +
                  " nv=" + std::to_string(nverts));
              non_quad_warning_step = state.step;
            }
            continue;
          }
          unit.cells.push_back(cell);
          const int offset = topology.cell_node_csr_offsets[
              static_cast<std::size_t>(cell)];
          unit.outer_nodes.push_back(
              topology.cell_node_csr_indices[
                  static_cast<std::size_t>(offset + 2)]);
          unit.outer_nodes.push_back(
              topology.cell_node_csr_indices[
                  static_cast<std::size_t>(offset + 3)]);
        }
        if (unit.cells.empty()) {
          continue;
        }
        sort_unique(unit.outer_nodes);
        append_unit_nodes(topology, cell_nverts, state.corner_stride, unit);
        unit.outer_radius =
            ring_mean_radius(unit.outer_nodes, node_r, node_z);
        units.push_back(std::move(unit));
      }
      continue;
    }
    if (block.role == mesh::BlockRole::TRANSITION_BELT) {
      AxisCoreStructuralUnit unit;
      unit.block_id = block_id;
      unit.radial_row = 0;
      unit.role = block.role;
      for (int cell = block.cell_begin;
           cell < block.cell_begin + block.cell_count;
           ++cell) {
        unit.cells.push_back(cell);
      }
      append_unit_nodes(topology, cell_nverts, state.corner_stride, unit);
      TENRYU_ASSERT(
          static_cast<std::size_t>(block_id) <
                  topology.dendrite_block_rings.size() &&
              !topology.dendrite_block_rings[
                   static_cast<std::size_t>(block_id)]
                   .outer_ring.empty(),
          "axis-core transition row ring export is incomplete");
      unit.outer_nodes = topology.dendrite_block_rings[
          static_cast<std::size_t>(block_id)].outer_ring;
      sort_unique(unit.outer_nodes);
      unit.outer_radius =
          ring_mean_radius(unit.outer_nodes, node_r, node_z);
      units.push_back(std::move(unit));
    }
  }
  std::sort(
      units.begin(), units.end(),
      [](const AxisCoreStructuralUnit& lhs,
         const AxisCoreStructuralUnit& rhs) {
        if (lhs.outer_radius != rhs.outer_radius) {
          return lhs.outer_radius < rhs.outer_radius;
        }
        if (lhs.block_id != rhs.block_id) {
          return lhs.block_id < rhs.block_id;
        }
        return lhs.radial_row > rhs.radial_row;
      });
  TENRYU_ASSERT(!units.empty() &&
                    units.front().role == mesh::BlockRole::CENTER_FAN,
                "axis-core structural units must begin with the center fan");
  for (std::size_t unit = 1; unit < units.size(); ++unit) {
    TENRYU_ASSERT(
        units[unit].outer_radius > units[unit - 1U].outer_radius,
        "axis-core structural unit radii must increase strictly");
  }
  return units;
}

void compute_axis_core_chi(
    const mesh::MultiBlockTopology& topology,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const double s_p,
    const double s_f,
    std::vector<double>& initial_radius,
    std::vector<double>& chi) {
  TENRYU_ASSERT(s_f > s_p,
                "axis-core feather radii must have positive width");
  const int n_nodes = static_cast<int>(node_r.size());
  initial_radius.assign(
      static_cast<std::size_t>(n_nodes),
      std::numeric_limits<double>::quiet_NaN());
  chi.assign(
      static_cast<std::size_t>(n_nodes),
      std::numeric_limits<double>::quiet_NaN());
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    if (node_z[node_index] < 0.0) {
      continue;
    }
    const int south = topology.south_node_of[node_index];
    TENRYU_ASSERT(
        south >= 0 && south < n_nodes,
        "axis-core canonical node has no south mirror");
    const double radius = std::hypot(node_r[node_index], node_z[node_index]);
    double weight = 0.0;
    if (radius <= s_p) {
      weight = 1.0;
    } else if (radius < s_f) {
      double xi = (radius - s_p) / (s_f - s_p);
      xi = fmin(fmax(xi, 0.0), 1.0);
      // Factored form is non-negative in FP for xi in [0,1] (raw polynomial cancels to -eps at xi~1).
      const double one_minus_xi = 1.0 - xi;
      weight = one_minus_xi * one_minus_xi * (1.0 + 2.0 * xi);
    }
    initial_radius[node_index] = radius;
    chi[node_index] = weight;
    initial_radius[static_cast<std::size_t>(south)] = radius;
    chi[static_cast<std::size_t>(south)] = weight;
  }
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    if (topology.south_node_of[node_index] < 0) {
      ::tenryu::core::tenryu_abort(
          "topology.south_node_of[node_index] >= 0",
          "axis-core node has no mirror entry: node=" + std::to_string(node) +
              " r=" + std::to_string(node_r[node_index]) +
              " z=" + std::to_string(node_z[node_index]),
          __FILE__, __LINE__);
    }
    if (!(std::isfinite(initial_radius[node_index]) &&
          std::isfinite(chi[node_index]) && chi[node_index] >= 0.0 &&
          chi[node_index] <= 1.0)) {
      ::tenryu::core::tenryu_abort(
          "axis-core chi must be finite and in [0,1]",
          "axis-core chi must be finite and in [0,1]: node=" +
              std::to_string(node) + " r=" +
              std::to_string(node_r[node_index]) + " z=" +
              std::to_string(node_z[node_index]) + " initial_radius=" +
              std::to_string(initial_radius[node_index]) + " chi=" +
              std::to_string(chi[node_index]) + " south_node_of=" +
              std::to_string(topology.south_node_of[node_index]),
          __FILE__, __LINE__);
    }
    const int mirror = topology.south_node_of[node_index];
    TENRYU_ASSERT(
        std::bit_cast<std::uint64_t>(chi[node_index]) ==
            std::bit_cast<std::uint64_t>(
                chi[static_cast<std::size_t>(mirror)]),
        "axis-core mirrored chi values must be bitwise identical");
  }
}

double feather_edge_jump(
    const core::State& state,
    const std::vector<AxisCoreStructuralUnit>& units,
    const std::size_t feather_begin,
    const std::size_t feather_end,
    const std::vector<double>& chi) {
  const auto& topology = *state.mesh.topo.multiblock;
  const std::vector<std::uint8_t> cell_nverts =
      state.mesh.cell_nverts.empty()
          ? std::vector<std::uint8_t>(
                static_cast<std::size_t>(state.mesh.topo.n_cells), 4U)
          : state.mesh.cell_nverts;
  double max_jump = 0.0;
  for (std::size_t unit = feather_begin; unit <= feather_end; ++unit) {
    for (const int cell : units[unit].cells) {
      const int nverts =
          static_cast<int>(cell_nverts[static_cast<std::size_t>(cell)]);
      const int offset = topology.cell_node_csr_offsets[
          static_cast<std::size_t>(cell)];
      for (int local = 0; local < nverts; ++local) {
        const int next = (local + 1) % nverts;
        const int node_a = topology.cell_node_csr_indices[
            static_cast<std::size_t>(offset + local)];
        const int node_b = topology.cell_node_csr_indices[
            static_cast<std::size_t>(offset + next)];
        max_jump = std::max(
            max_jump,
            std::abs(chi[static_cast<std::size_t>(node_a)] -
                     chi[static_cast<std::size_t>(node_b)]));
      }
    }
  }
  return max_jump;
}

std::uint64_t axis_core_chi_hash(const std::vector<double>& chi) {
  std::uint64_t hash = 0;
  for (const double value : chi) {
    hash = std::rotl(hash, 7) ^ std::bit_cast<std::uint64_t>(value);
  }
  return hash;
}

std::uint64_t axis_core_mask_hash(
    const std::vector<std::uint8_t>& mask) {
  std::uint64_t hash = 0;
  for (std::size_t node = 0; node < mask.size(); ++node) {
    if (mask[node] != 0U) {
      hash = std::rotl(hash, 7) ^
             static_cast<std::uint64_t>(node + 1U);
    }
  }
  return hash;
}

void validate_spec(const EulerWindowSpec& spec) {
  if (spec.transition_width <= 0.0) {
    throw std::invalid_argument(
        "Eulerian window transition width must be positive");
  }

  switch (spec.shape) {
    case EulerWindowSpec::Shape::Rectangle:
      if (spec.r1 <= spec.r0 || spec.z1 <= spec.z0) {
        throw std::invalid_argument(
            "Eulerian rectangle window bounds must have positive extent");
      }
      break;
    case EulerWindowSpec::Shape::Annulus:
      if (spec.rad_out <= spec.rad_in) {
        throw std::invalid_argument(
            "Eulerian annulus window radii must have positive extent");
      }
      break;
    default:
      throw std::invalid_argument("Eulerian window shape is invalid");
  }
}

double signed_distance_outside(const EulerWindowSpec& spec,
                               const double r,
                               const double z) {
  if (spec.shape == EulerWindowSpec::Shape::Rectangle) {
    return std::max(
        std::max(spec.r0 - r, r - spec.r1),
        std::max(spec.z0 - z, z - spec.z1));
  }

  const double radius = std::hypot(r - spec.cr, z - spec.cz);
  return std::max(spec.rad_in - radius, radius - spec.rad_out);
}

double transition_weight(const double signed_distance,
                         const double transition_width) {
  if (signed_distance <= 0.0) {
    return 1.0;
  }
  if (signed_distance >= transition_width) {
    return 0.0;
  }

  const double t = signed_distance / transition_width;
  const double smoothstep = t * t * (3.0 - 2.0 * t);
  return 1.0 - smoothstep;
}

}  // namespace

bool axis_core_ring_release_enabled(const core::Config& cfg) {
  static const bool env_enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_RING_RELEASE");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return env_enabled ||
         cfg.numerics.ale.euler_window.axis_core_ring_release_enabled;
}

int axis_core_released_units() {
  return g_axis_core_released_units;
}

void axis_core_set_released_units(const int n) {
  TENRYU_ASSERT(n >= 0,
                "AXIS_CORE_RELEASED_UNITS must be non-negative");
  g_axis_core_released_units = n;
}

void euler_window_cell_weights(const EulerWindowSpec& spec,
                               const double* cell_cr,
                               const double* cell_cz,
                               const int n_cells,
                               double* w_cell) {
  validate_spec(spec);
  if (cell_cr == nullptr || cell_cz == nullptr || w_cell == nullptr) {
    throw std::invalid_argument(
        "Eulerian window cell-weight arrays must be non-null");
  }

  for (int cell = 0; cell < n_cells; ++cell) {
    const std::size_t index = static_cast<std::size_t>(cell);
    w_cell[index] = transition_weight(
        signed_distance_outside(spec, cell_cr[index], cell_cz[index]),
        spec.transition_width);
  }
}

void euler_window_node_weights(const int* cell_node_csr_offsets,
                               const int* cell_node_csr_indices,
                               const std::uint8_t* cell_nverts,
                               const int n_cells,
                               const int corner_stride,
                               const int n_nodes,
                               const double* w_cell,
                               double* w_node) {
  if (cell_node_csr_offsets == nullptr ||
      cell_node_csr_indices == nullptr || w_cell == nullptr ||
      w_node == nullptr) {
    throw std::invalid_argument(
        "Eulerian window node-weight arrays must be non-null");
  }

  std::fill(w_node, w_node + n_nodes, 0.0);
  std::vector<int> incident_count(static_cast<std::size_t>(n_nodes), 0);
  for (int cell = 0; cell < n_cells; ++cell) {
    const int nverts =
        cell_nverts == nullptr ? 4 : static_cast<int>(cell_nverts[cell]);
    if (nverts > corner_stride) {
      throw std::invalid_argument(
          "Eulerian window cell vertex count exceeds corner stride");
    }
    const int offset = cell_node_csr_offsets[cell];
    for (int local = 0; local < nverts; ++local) {
      const int node = cell_node_csr_indices[offset + local];
      w_node[node] += w_cell[cell];
      ++incident_count[static_cast<std::size_t>(node)];
    }
  }

  for (int node = 0; node < n_nodes; ++node) {
    const int count = incident_count[static_cast<std::size_t>(node)];
    if (count > 0) {
      w_node[node] /= static_cast<double>(count);
    }
  }
}

void euler_window_node_weights(
    const std::vector<EulerWindowSpec>& specs,
    const double* cell_cr,
    const double* cell_cz,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const int n_cells,
    const int corner_stride,
    const int n_nodes,
    double* w_node) {
  if (cell_cr == nullptr || cell_cz == nullptr || w_node == nullptr) {
    throw std::invalid_argument(
        "Eulerian window multi-window arrays must be non-null");
  }

  std::vector<double> w_cell(static_cast<std::size_t>(n_cells), 0.0);
  if (specs.empty()) {
    std::fill(w_node, w_node + n_nodes, 0.0);
    return;
  }
  euler_window_cell_weights(
      specs.front(), cell_cr, cell_cz, n_cells, w_cell.data());
  euler_window_node_weights(
      cell_node_csr_offsets,
      cell_node_csr_indices,
      cell_nverts,
      n_cells,
      corner_stride,
      n_nodes,
      w_cell.data(),
      w_node);

  std::vector<double> w_window_node(
      static_cast<std::size_t>(n_nodes), 0.0);
  for (std::size_t window = 1; window < specs.size(); ++window) {
    euler_window_cell_weights(
        specs[window], cell_cr, cell_cz, n_cells, w_cell.data());
    euler_window_node_weights(
        cell_node_csr_offsets,
        cell_node_csr_indices,
        cell_nverts,
        n_cells,
        corner_stride,
        n_nodes,
        w_cell.data(),
        w_window_node.data());
    for (int node = 0; node < n_nodes; ++node) {
      const auto index = static_cast<std::size_t>(node);
      // Plain max of C1 window fields is only C0 at crossings. Users should
      // merge overlapping windows when smoothness at crossings is required.
      w_node[index] = std::max(w_node[index], w_window_node[index]);
    }
  }
}

void euler_window_blend_targets(const double* w_node,
                                const int n_nodes,
                                const double* x_lagr_r,
                                const double* x_lagr_z,
                                const double* x_euler_r,
                                const double* x_euler_z,
                                double* x_target_r,
                                double* x_target_z) {
  if (w_node == nullptr || x_lagr_r == nullptr || x_lagr_z == nullptr ||
      x_euler_r == nullptr || x_euler_z == nullptr ||
      x_target_r == nullptr || x_target_z == nullptr) {
    throw std::invalid_argument(
        "Eulerian window blend arrays must be non-null");
  }

  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    const double weight = w_node[index];
    if (weight == 1.0) {
      x_target_r[index] = x_euler_r[index];
      x_target_z[index] = x_euler_z[index];
    } else if (weight == 0.0) {
      x_target_r[index] = x_lagr_r[index];
      x_target_z[index] = x_lagr_z[index];
    } else {
      x_target_r[index] =
          weight * x_euler_r[index] +
          (1.0 - weight) * x_lagr_r[index];
      x_target_z[index] =
          weight * x_euler_z[index] +
          (1.0 - weight) * x_lagr_z[index];
    }
  }
}

void build_axis_core_disk(core::State& state, const core::Config& cfg) {
  delete g_axis_core_disk_runtime;
  g_axis_core_disk_runtime = nullptr;

  const auto& config = cfg.numerics.ale.euler_window;
  if (!config.enabled || config.role != "axis_survival_core") {
    return;
  }
  if (!state.mesh.topo.multiblock.has_value() ||
      state.mesh.topo.multiblock->south_node_of.empty()) {
    throw core::namelist::ConfigError(
        "axis_survival_core requires the mirrored polar-tier topology");
  }
  const auto& topology = *state.mesh.topo.multiblock;
  const core::PolarTierLayout layout =
      core::make_polar_tier_layout(cfg.mesh);
  if (topology.block_count != layout.block_count ||
      topology.dendrite_block_rings.empty()) {
    throw core::namelist::ConfigError(
        "axis_survival_core requires the dendrite/native polar-tier topology");
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(
      n_cells > 0 && n_nodes > 0 &&
          topology.south_node_of.size() ==
              static_cast<std::size_t>(n_nodes),
      "axis-core runtime requires a complete mirrored topology");
  TENRYU_ASSERT(
      topology.cell_node_csr_offsets.size() ==
              static_cast<std::size_t>(n_cells) + 1U &&
          state.mesh.cell_nverts.size() ==
              static_cast<std::size_t>(n_cells),
      "axis-core runtime requires complete cell topology");

  auto runtime = std::make_unique<AxisCoreDiskRuntime>();
  runtime->n_cells = n_cells;
  runtime->n_nodes = n_nodes;
  runtime->always_moving_mode =
      config.axis_core_transaction_mode == "always_moving";
  runtime->transition_passage_enabled =
      config.axis_core_transition_passage_enabled;
  state.x_r_initial.copy_to_host(runtime->captured_r);
  state.x_z_initial.copy_to_host(runtime->captured_z);
  TENRYU_ASSERT(
      runtime->captured_r.size() == static_cast<std::size_t>(n_nodes) &&
          runtime->captured_z.size() == static_cast<std::size_t>(n_nodes),
      "axis-core captured coordinate count mismatch");

  const std::vector<AxisCoreStructuralUnit> units =
      make_axis_core_structural_units(
          state, runtime->captured_r, runtime->captured_z);
  std::size_t plateau_end = units.size();
  for (std::size_t unit = 0; unit < units.size(); ++unit) {
    if (units[unit].role == mesh::BlockRole::POLAR_SHELL) {
      break;
    }
    if (units[unit].outer_radius >= config.rad_out) {
      plateau_end = unit;
      break;
    }
  }
  if (g_axis_core_released_units > 0) {
    TENRYU_ASSERT(
        plateau_end != units.size(),
        "axis-core ring release requires a resolved plateau unit");
    const std::size_t released =
        static_cast<std::size_t>(g_axis_core_released_units);
    if (plateau_end < released) {
      throw core::namelist::ConfigError(
          "axis_survival_core: ring release exhausted the plateau unit ladder");
    }
    plateau_end -= released;
    core::log_info("[ring-release] rebuild released_units=" +
                   std::to_string(g_axis_core_released_units) +
                   " plateau_end_unit=" + std::to_string(plateau_end));
  }
  if (plateau_end == units.size() || plateau_end + 1U >= units.size()) {
    throw core::namelist::ConfigError(
        "axis_survival_core: requested plateau radius leaves insufficient "
        "structural layers for the feather");
  }
  runtime->s_p = units[plateau_end].outer_radius;
  const double plateau_radius_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(runtime->s_p, 1.0e-300);
  for (const int node : units[plateau_end].outer_nodes) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    const double captured_radius = std::hypot(
        runtime->captured_r[node_index],
        runtime->captured_z[node_index]);
    TENRYU_ASSERT(
        std::isfinite(captured_radius) &&
            std::abs(captured_radius - runtime->s_p) <=
                plateau_radius_tolerance,
        "axis-core captured plateau radius must match snapped s_p");
  }

  std::size_t feather_end = units.size();
  std::vector<double> candidate_radius;
  std::vector<double> candidate_chi;
  for (std::size_t unit = plateau_end + 1U;
       unit < units.size() &&
       units[unit].role != mesh::BlockRole::POLAR_SHELL;
       ++unit) {
    const double candidate_s_f = units[unit].outer_radius;
    compute_axis_core_chi(
        topology,
        runtime->captured_r,
        runtime->captured_z,
        runtime->s_p,
        candidate_s_f,
        candidate_radius,
        candidate_chi);
    const int layer_count =
        static_cast<int>(unit - plateau_end);
    const bool enough_layers =
        layer_count >= config.feather_min_layers;
    const bool enough_width =
        candidate_s_f - runtime->s_p >= config.transition_width;
    const bool jump_ok =
        feather_edge_jump(
            state,
            units,
            plateau_end + 1U,
            unit,
            candidate_chi) <= 0.5;
    if (enough_layers && enough_width && jump_ok) {
      feather_end = unit;
      runtime->s_f = candidate_s_f;
      runtime->initial_radius = std::move(candidate_radius);
      runtime->chi = std::move(candidate_chi);
      break;
    }
  }
  if (feather_end == units.size()) {
    throw core::namelist::ConfigError(
        "axis_survival_core: requested plateau radius leaves insufficient "
        "structural layers for the feather");
  }

  std::vector<std::uint8_t> cell_membership(
      static_cast<std::size_t>(n_cells), 0U);
  const auto add_unit_cells =
      [&](const AxisCoreStructuralUnit& unit,
          const std::uint8_t membership,
          std::vector<int>& destination) {
        for (const int cell : unit.cells) {
          TENRYU_ASSERT(
              cell >= 0 && cell < n_cells &&
                  cell_membership[static_cast<std::size_t>(cell)] == 0U,
              "axis-core structural cell belongs to multiple sets");
          cell_membership[static_cast<std::size_t>(cell)] = membership;
          destination.push_back(cell);
        }
      };
  for (std::size_t unit = 0; unit <= plateau_end; ++unit) {
    add_unit_cells(units[unit], 1U, runtime->plateau_cells);
  }
  for (std::size_t unit = plateau_end + 1U;
       unit <= feather_end;
       ++unit) {
    add_unit_cells(units[unit], 2U, runtime->feather_cells);
  }
  const int requested_guard_layers = std::max(0, config.guard_layers);
  const std::size_t guard_end = std::min(
      units.size(),
      feather_end + 1U +
          static_cast<std::size_t>(requested_guard_layers));
  for (std::size_t unit = feather_end + 1U;
       unit < guard_end;
       ++unit) {
    add_unit_cells(units[unit], 3U, runtime->guard_cells);
  }

  if (runtime->transition_passage_enabled) {
    build_axis_core_transition_passage(state, topology, *runtime);
  }

  runtime->plateau_node.assign(static_cast<std::size_t>(n_nodes), 0U);
  runtime->feather_node.assign(static_cast<std::size_t>(n_nodes), 0U);
  runtime->guard_node.assign(static_cast<std::size_t>(n_nodes), 0U);
  runtime->mask.assign(static_cast<std::size_t>(n_nodes), 0U);
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    const double radius = runtime->initial_radius[node_index];
    if (radius <= runtime->s_p) {
      runtime->plateau_node[node_index] = 1U;
    } else if (radius < runtime->s_f) {
      runtime->feather_node[node_index] = 1U;
    }
    if (runtime->always_moving_mode &&
        runtime->chi[node_index] == 1.0) {
      runtime->plateau_nodes.push_back(node);
    }
    runtime->mask[node_index] = runtime->chi[node_index] > 0.0 ? 1U : 0U;
  }
  for (std::size_t unit = feather_end + 1U;
       unit < guard_end;
       ++unit) {
    for (const int node : units[unit].nodes) {
      runtime->guard_node[static_cast<std::size_t>(node)] = 1U;
    }
  }

  TENRYU_ASSERT(
      runtime->initial_radius.size() == static_cast<std::size_t>(n_nodes) &&
          runtime->chi.size() == static_cast<std::size_t>(n_nodes) &&
          runtime->mask.size() == static_cast<std::size_t>(n_nodes),
      "axis-core node storage count mismatch");
  for (std::size_t unit = 0; unit <= feather_end; ++unit) {
    const std::uint8_t expected = unit <= plateau_end ? 1U : 2U;
    for (const int cell : units[unit].cells) {
      TENRYU_ASSERT(
          cell_membership[static_cast<std::size_t>(cell)] == expected,
          "axis-core plateau+feather structural sets contain a hole");
    }
  }
  for (int block_id = 0;
       block_id < static_cast<int>(topology.blocks.size());
       ++block_id) {
    const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
    if (block.role == mesh::BlockRole::CENTER_FAN) {
      for (int cell = block.cell_begin;
           cell < block.cell_begin + block.cell_count;
           ++cell) {
        TENRYU_ASSERT(
            cell_membership[static_cast<std::size_t>(cell)] == 1U,
            "axis-core plateau must contain the complete center fan");
      }
    }
    if (block.role == mesh::BlockRole::TRANSITION_BELT) {
      const std::uint8_t first = cell_membership[
          static_cast<std::size_t>(block.cell_begin)];
      for (int cell = block.cell_begin;
           cell < block.cell_begin + block.cell_count;
           ++cell) {
        TENRYU_ASSERT(
            cell_membership[static_cast<std::size_t>(cell)] == first,
            "axis-core transition row must never be partially selected");
      }
    }
  }
  TENRYU_ASSERT(
      state.mesh.topo.node_flags.size() == static_cast<std::size_t>(n_nodes),
      "axis-core validation requires node flags");
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    TENRYU_ASSERT(
        runtime->chi[node_index] >= 0.0 &&
            runtime->chi[node_index] <= 1.0,
        "axis-core chi is outside [0,1]");
    if ((state.mesh.topo.node_flags[node_index] & mesh::NODE_AXIS) != 0U &&
        runtime->initial_radius[node_index] < runtime->s_f) {
      TENRYU_ASSERT(
          runtime->mask[node_index] != 0U,
          "axis-core plateau+feather mask must include both pole axes");
    }
  }

  const std::uint64_t chi_hash = axis_core_chi_hash(runtime->chi);
  const std::uint64_t mask_hash = axis_core_mask_hash(runtime->mask);
  std::ostringstream log;
  log << std::scientific << std::setprecision(6)
      << "[axis-core] plateau_cells=" << runtime->plateau_cells.size()
      << " feather_cells=" << runtime->feather_cells.size()
      << " guard_cells=" << runtime->guard_cells.size()
      << " s_p=" << runtime->s_p
      << " s_f=" << runtime->s_f
      << " chi_hash=" << std::hex << std::setw(16) << std::setfill('0')
      << static_cast<unsigned long long>(chi_hash)
      << " mask_hash=" << std::setw(16)
      << static_cast<unsigned long long>(mask_hash);
  core::log_info(log.str());

  if (junction_audit_enabled() || junction_ledger_cadence() > 0 ||
      seam_donor_enabled()) {
    const std::vector<AxisCoreJunctionCohort> junction_cohorts =
        make_axis_core_junction_cohorts(state, runtime->plateau_cells);
    if (seam_donor_enabled()) {
      std::vector<std::uint8_t> seam_donor_cell_mask(
          static_cast<std::size_t>(n_cells), 0U);
      for (const auto& cohort : junction_cohorts) {
        for (const int cell : cohort.cells) {
          TENRYU_ASSERT(
              cell >= 0 && cell < n_cells,
              "axis-core seam-donor cohort cell is out of range");
          std::uint8_t& masked = seam_donor_cell_mask[
              static_cast<std::size_t>(cell)];
          if (masked == 0U) {
            masked = 1U;
            ++runtime->seam_donor_cell_count;
          }
        }
      }
      runtime->seam_donor_cell_mask.reset(
          static_cast<std::size_t>(n_cells));
      runtime->seam_donor_cell_mask.copy_from_host(
          seam_donor_cell_mask);
    }
    if (junction_ledger_cadence() > 0) {
      runtime->junction_ledger_cells =
          make_axis_core_junction_ledger_cells(topology, junction_cohorts);
    }
    if (junction_audit_enabled()) {
      static bool audit_has_run = false;
      if (!audit_has_run) {
        run_axis_core_junction_audit(
            state,
            runtime->captured_r,
            runtime->captured_z,
            junction_cohorts);
        audit_has_run = true;
      }
    }
  }
  g_axis_core_disk_runtime = runtime.release();
}

bool axis_core_disk_release_one_unit(core::State& state,
                                     const core::Config& cfg) {
  const int previous = g_axis_core_released_units;
  ++g_axis_core_released_units;
  try {
    build_axis_core_disk(state, cfg);
  } catch (const core::namelist::ConfigError& error) {
    g_axis_core_released_units = previous;
    build_axis_core_disk(state, cfg);
    core::log_warning(std::string("[ring-release] EXHAUSTED: ") +
                      error.what());
    return false;
  }
  const AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  std::ostringstream log;
  log << std::scientific << std::setprecision(6)
      << "[ring-release] n=" << g_axis_core_released_units
      << " trigger=driver"
      << " s_p=" << runtime.s_p
      << " s_f=" << runtime.s_f
      << " plateau_cells=" << runtime.plateau_cells.size()
      << " feather_cells=" << runtime.feather_cells.size()
      << " guard_cells=" << runtime.guard_cells.size();
  core::log_info(log.str());
  return true;
}

void destroy_axis_core_disk() {
  delete g_axis_core_disk_runtime;
  g_axis_core_disk_runtime = nullptr;
}

AxisCoreDiskObservation axis_core_disk_observe(const core::State& state) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "axis-core observation requested without initialized runtime");
  const AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.n_cells == state.mesh.topo.n_cells &&
          runtime.n_nodes == state.mesh.topo.n_nodes,
      "axis-core runtime topology changed during the run");
  return {
      runtime.s_p,
      runtime.s_f,
      runtime.initial_radius,
      runtime.chi,
      runtime.captured_r,
      runtime.captured_z,
      runtime.plateau_cells,
      runtime.feather_cells,
      runtime.guard_cells,
  };
}

std::vector<AxisCoreJunctionLedgerCell>
axis_core_disk_junction_ledger_cells(const core::State& state) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "axis-core junction ledger requested without initialized runtime");
  const AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.n_cells == state.mesh.topo.n_cells &&
          runtime.n_nodes == state.mesh.topo.n_nodes,
      "axis-core runtime topology changed during the run");
  return runtime.junction_ledger_cells;
}

const std::uint8_t*
axis_core_disk_seam_donor_cell_mask(const core::State& state) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "axis-core seam-donor mask requested without initialized runtime");
  const AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.n_cells == state.mesh.topo.n_cells &&
          runtime.n_nodes == state.mesh.topo.n_nodes,
      "axis-core runtime topology changed during the run");
  if (runtime.seam_donor_cell_mask.empty()) {
    return nullptr;
  }
  TENRYU_ASSERT(
      runtime.seam_donor_cell_mask.size() ==
          static_cast<std::size_t>(runtime.n_cells),
      "axis-core seam-donor mask size mismatch");
  static bool logged = false;
  if (!logged) {
    core::log_info(
        "[seam-donor] cohort_cells=" +
        std::to_string(runtime.seam_donor_cell_count) + " active");
    logged = true;
  }
  return runtime.seam_donor_cell_mask.data();
}

void axis_core_disk_build_target(const core::State& state,
                                 std::vector<double>& target_r,
                                 std::vector<double>& target_z) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "axis-core target requested without initialized runtime");
  const AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.n_nodes == state.mesh.topo.n_nodes,
      "axis-core runtime topology changed during the run");
  std::vector<double> lagrangian_r;
  std::vector<double> lagrangian_z;
  state.x_r.copy_to_host(lagrangian_r);
  state.x_z.copy_to_host(lagrangian_z);
  target_r.resize(static_cast<std::size_t>(runtime.n_nodes));
  target_z.resize(static_cast<std::size_t>(runtime.n_nodes));
  for (int node = 0; node < runtime.n_nodes; ++node) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    const double weight = runtime.chi[node_index];
    if (weight == 1.0) {
      target_r[node_index] = runtime.captured_r[node_index];
      target_z[node_index] = runtime.captured_z[node_index];
    } else if (weight == 0.0) {
      target_r[node_index] = lagrangian_r[node_index];
      target_z[node_index] = lagrangian_z[node_index];
    } else {
      target_r[node_index] =
          lagrangian_r[node_index] +
          weight * (runtime.captured_r[node_index] -
                    lagrangian_r[node_index]);
      target_z[node_index] =
          lagrangian_z[node_index] +
          weight * (runtime.captured_z[node_index] -
                    lagrangian_z[node_index]);
    }
  }
}

double axis_core_disk_build_moving_target(
    const core::State& state,
    std::vector<double>& lagrangian_r,
    std::vector<double>& lagrangian_z,
    std::vector<double>& target_r,
    std::vector<double>& target_z) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "axis-core moving target requested without initialized runtime");
  AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.always_moving_mode,
      "axis-core moving target requested outside moving-target mode");
  TENRYU_ASSERT(
      runtime.n_nodes == state.mesh.topo.n_nodes,
      "axis-core runtime topology changed during the run");
  state.x_r.copy_to_host(lagrangian_r);
  state.x_z.copy_to_host(lagrangian_z);

  std::vector<double> node_planar_mass;
  const bool planar_mass_available =
      state.node_planar_mass.size() ==
      static_cast<std::size_t>(runtime.n_nodes);
  if (planar_mass_available) {
    state.node_planar_mass.copy_to_host(node_planar_mass);
  } else if (!runtime.uniform_mass_warning_emitted) {
    core::log_warning(
        "[axis-core-repair] node_planar_mass unavailable; using uniform "
        "alpha-fit weights");
    runtime.uniform_mass_warning_emitted = true;
  }

  double numerator = 0.0;
  double denominator = 0.0;
  for (const int node : runtime.plateau_nodes) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    const double weight =
        planar_mass_available ? node_planar_mass[node_index] : 1.0;
    numerator +=
        weight *
        (runtime.captured_r[node_index] * lagrangian_r[node_index] +
         runtime.captured_z[node_index] * lagrangian_z[node_index]);
    denominator +=
        weight *
        (runtime.captured_r[node_index] * runtime.captured_r[node_index] +
         runtime.captured_z[node_index] * runtime.captured_z[node_index]);
  }
  TENRYU_ASSERT(
      std::isfinite(numerator) && std::isfinite(denominator) &&
          denominator > 0.0,
      "axis-core moving-target alpha fit is invalid");
  const double alpha = numerator / denominator;
  TENRYU_ASSERT(std::isfinite(alpha),
                "axis-core moving-target alpha is non-finite");

  target_r.resize(static_cast<std::size_t>(runtime.n_nodes));
  target_z.resize(static_cast<std::size_t>(runtime.n_nodes));
  for (int node = 0; node < runtime.n_nodes; ++node) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    const double weight = runtime.chi[node_index];
    const double moving_r = alpha * runtime.captured_r[node_index];
    const double moving_z = alpha * runtime.captured_z[node_index];
    if (weight == 1.0) {
      target_r[node_index] = moving_r;
      target_z[node_index] = moving_z;
    } else if (weight == 0.0) {
      target_r[node_index] = lagrangian_r[node_index];
      target_z[node_index] = lagrangian_z[node_index];
    } else {
      target_r[node_index] =
          lagrangian_r[node_index] +
          weight * (moving_r - lagrangian_r[node_index]);
      target_z[node_index] =
          lagrangian_z[node_index] +
          weight * (moving_z - lagrangian_z[node_index]);
    }
  }
  return alpha;
}

AxisCoreTransitionPassageStep axis_core_transition_passage_update(
    core::State& state,
    const double time,
    const bool emit_logs,
    const parallel::Reduction* reduction) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "transition-passage update requested without initialized runtime");
  AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  AxisCoreTransitionPassageStep step;
  step.enabled = runtime.transition_passage_enabled;
  if (!runtime.transition_passage_enabled) {
    return step;
  }
  TENRYU_ASSERT(
      runtime.n_cells == state.mesh.topo.n_cells &&
          runtime.n_nodes == state.mesh.topo.n_nodes &&
          !runtime.transition_patches.empty() &&
          !runtime.transition_central_cells.empty(),
      "transition-passage runtime topology changed during the run");
  TENRYU_ASSERT(
      state.mass.size() == static_cast<std::size_t>(runtime.n_cells) &&
          state.Pe.size() == static_cast<std::size_t>(runtime.n_cells) &&
          state.Pi.size() == static_cast<std::size_t>(runtime.n_cells) &&
          state.cs.size() == static_cast<std::size_t>(runtime.n_cells) &&
          state.v_r.size() == static_cast<std::size_t>(runtime.n_nodes) &&
          state.v_z.size() == static_cast<std::size_t>(runtime.n_nodes),
      "transition-passage requires complete hydro fields");
  const std::int8_t* hydro_active =
      state.hydro_active.empty() ? nullptr : state.hydro_active_device_ptr();
  constexpr int threads = 256;

  bool has_convergence_patch = false;
  for (const auto& patch : runtime.transition_patches) {
    has_convergence_patch =
        has_convergence_patch ||
        patch.state == AxisCoreTransitionPassageState::Convergence;
  }
  if (has_convergence_patch) {
    const int n_central =
        static_cast<int>(runtime.transition_central_cells.size());
    const int blocks = (n_central + threads - 1) / threads;
    axis_core_transition_central_outflow_kernel<<<blocks, threads>>>(
        runtime.d_transition_central_cells.data(),
        n_central,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        runtime.d_transition_cell_nverts.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.v_r.data(),
        state.v_z.data(),
        state.cs.data(),
        state.mass.data(),
        hydro_active,
        runtime.d_transition_central_mass_ur.data(),
        runtime.d_transition_central_mass_cs.data(),
        runtime.d_transition_central_mass.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> mass_ur;
    std::vector<double> mass_cs;
    std::vector<double> selected_mass;
    runtime.d_transition_central_mass_ur.copy_to_host(mass_ur);
    runtime.d_transition_central_mass_cs.copy_to_host(mass_cs);
    runtime.d_transition_central_mass.copy_to_host(selected_mass);
    double numerator = 0.0;
    double numerator_compensation = 0.0;
    double sound_speed_numerator = 0.0;
    double sound_speed_numerator_compensation = 0.0;
    double denominator = 0.0;
    double denominator_compensation = 0.0;
    for (std::size_t selected = 0; selected < mass_ur.size(); ++selected) {
      const double numerator_corrected =
          mass_ur[selected] - numerator_compensation;
      const double numerator_next = numerator + numerator_corrected;
      numerator_compensation =
          (numerator_next - numerator) - numerator_corrected;
      numerator = numerator_next;
      const double sound_speed_numerator_corrected =
          mass_cs[selected] - sound_speed_numerator_compensation;
      const double sound_speed_numerator_next =
          sound_speed_numerator + sound_speed_numerator_corrected;
      sound_speed_numerator_compensation =
          (sound_speed_numerator_next - sound_speed_numerator) -
          sound_speed_numerator_corrected;
      sound_speed_numerator = sound_speed_numerator_next;
      const double denominator_corrected =
          selected_mass[selected] - denominator_compensation;
      const double denominator_next = denominator + denominator_corrected;
      denominator_compensation =
          (denominator_next - denominator) - denominator_corrected;
      denominator = denominator_next;
    }
    if (reduction != nullptr) {
      numerator = reduction->allreduce_sum(numerator);
      sound_speed_numerator =
          reduction->allreduce_sum(sound_speed_numerator);
      denominator = reduction->allreduce_sum(denominator);
    }
    TENRYU_ASSERT(
        std::isfinite(numerator) && std::isfinite(sound_speed_numerator) &&
            std::isfinite(denominator) && denominator > 0.0,
        "transition-passage central mass-weighted outflow is invalid");
    const double mean_u_r = numerator / denominator;
    const double mean_c_s = sound_speed_numerator / denominator;
    if (mean_u_r > 0.01 * mean_c_s) {
      for (auto& patch : runtime.transition_patches) {
        if (patch.state != AxisCoreTransitionPassageState::Convergence) {
          continue;
        }
        patch.state = AxisCoreTransitionPassageState::Armed;
        if (emit_logs) {
          std::ostringstream message;
          message << std::scientific << std::setprecision(17)
                  << "[transition-passage] ARMED t=" << time
                  << " patch=" << patch.belt_block_id;
          core::log_info(message.str());
        }
      }
    }
  }

  for (int patch_index = 0;
       patch_index < static_cast<int>(runtime.transition_patches.size());
       ++patch_index) {
    AxisCoreTransitionPatch& patch =
        runtime.transition_patches[static_cast<std::size_t>(patch_index)];
    const int n_sensor = static_cast<int>(patch.sensor_cells.size());
    const int blocks = (n_sensor + threads - 1) / threads;
    axis_core_transition_sensor_kernel<<<blocks, threads>>>(
        patch.d_sensor_cells.data(),
        n_sensor,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        runtime.d_transition_face_adj_offsets.data(),
        runtime.d_transition_face_adj_indices.data(),
        runtime.d_transition_cell_nverts.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.v_r.data(),
        state.v_z.data(),
        state.Pe.data(),
        state.Pi.data(),
        state.cs.data(),
        hydro_active,
        runtime.n_cells,
        patch.d_sensor_strength.data(),
        patch.d_sensor_radius.data());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> strength;
    std::vector<double> radius;
    patch.d_sensor_strength.copy_to_host(strength);
    patch.d_sensor_radius.copy_to_host(radius);
    double best_strength = -std::numeric_limits<double>::infinity();
    int best_cell = std::numeric_limits<int>::max();
    double best_radius = 0.0;
    for (std::size_t selected = 0; selected < strength.size(); ++selected) {
      const int cell = patch.sensor_cells[selected];
      if (!std::isfinite(strength[selected])) {
        continue;
      }
      if (strength[selected] > best_strength ||
          (strength[selected] == best_strength && cell < best_cell)) {
        best_strength = strength[selected];
        best_cell = cell;
        best_radius = radius[selected];
      }
    }
    TENRYU_ASSERT(
        best_cell != std::numeric_limits<int>::max() &&
            std::isfinite(best_strength) && std::isfinite(best_radius),
        "transition-passage sensor domain has no active finite cell");
    patch.last_shock_radius = best_radius;
    if (patch.state == AxisCoreTransitionPassageState::Armed &&
        best_radius <= patch.inner_radius &&
        patch.inner_radius - best_radius <=
            6.0 * patch.median_inner_width &&
        best_strength >= 0.5) {
      patch.state = AxisCoreTransitionPassageState::Passage;
      if (emit_logs) {
        std::ostringstream message;
        message << std::scientific << std::setprecision(17)
                << "[transition-passage] PASSAGE t=" << time
                << " r_s=" << best_radius
                << " patch=" << patch.belt_block_id;
        core::log_info(message.str());
      }
    }
    if (patch.state == AxisCoreTransitionPassageState::Passage) {
      step.passage_patches.push_back(
          {patch_index, patch.belt_block_id, best_radius});
    }
  }
  return step;
}

void axis_core_transition_passage_prepare_optimal_targets(
    const core::State& state) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "transition-passage Jacobi target requested without initialized runtime");
  AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(runtime.transition_passage_enabled,
                "transition-passage Jacobi target requested while disabled");
  constexpr int threads = 256;
  const std::size_t node_bytes =
      static_cast<std::size_t>(runtime.n_nodes) * sizeof(double);
  for (auto& patch : runtime.transition_patches) {
    if (patch.state != AxisCoreTransitionPassageState::Passage) {
      continue;
    }
    CUDA_CHECK(cudaMemcpy(runtime.d_transition_jacobi_r_a.data(),
                          state.x_r.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_transition_jacobi_z_a.data(),
                          state.x_z.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_transition_jacobi_r_b.data(),
                          state.x_r.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(runtime.d_transition_jacobi_z_b.data(),
                          state.x_z.data(),
                          node_bytes,
                          cudaMemcpyDeviceToDevice));
    double* current_r = runtime.d_transition_jacobi_r_a.data();
    double* current_z = runtime.d_transition_jacobi_z_a.data();
    double* next_r = runtime.d_transition_jacobi_r_b.data();
    double* next_z = runtime.d_transition_jacobi_z_b.data();
    const int n_patch_cells = static_cast<int>(patch.cells.size());
    const int n_patch_nodes = static_cast<int>(patch.nodes.size());
    const int cell_blocks = (n_patch_cells + threads - 1) / threads;
    const int node_blocks = (n_patch_nodes + threads - 1) / threads;
    for (int iteration = 0; iteration < 6; ++iteration) {
      axis_core_transition_cell_centers_kernel<<<cell_blocks, threads>>>(
          patch.d_cells.data(),
          n_patch_cells,
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          runtime.d_transition_cell_nverts.data(),
          current_r,
          current_z,
          patch.d_cell_center_r.data(),
          patch.d_cell_center_z.data());
      axis_core_transition_jacobi_nodes_kernel<<<node_blocks, threads>>>(
          patch.d_nodes.data(),
          n_patch_nodes,
          patch.d_node_cell_offsets.data(),
          patch.d_node_cells.data(),
          patch.d_feather.data(),
          patch.d_cell_center_r.data(),
          patch.d_cell_center_z.data(),
          current_r,
          current_z,
          next_r,
          next_z);
      std::swap(current_r, next_r);
      std::swap(current_z, next_z);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<double> optimal_r(static_cast<std::size_t>(runtime.n_nodes));
    std::vector<double> optimal_z(static_cast<std::size_t>(runtime.n_nodes));
    CUDA_CHECK(cudaMemcpy(optimal_r.data(),
                          current_r,
                          node_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(optimal_z.data(),
                          current_z,
                          node_bytes,
                          cudaMemcpyDeviceToHost));
    for (std::size_t local_node = 0;
         local_node < patch.nodes.size();
         ++local_node) {
      const int node = patch.nodes[local_node];
      patch.optimal_r[local_node] = optimal_r[static_cast<std::size_t>(node)];
      patch.optimal_z[local_node] = optimal_z[static_cast<std::size_t>(node)];
    }
  }
}

void axis_core_transition_passage_compose_patch_target(
    const int patch_index,
    const double alpha,
    const std::vector<double>& lagrangian_r,
    const std::vector<double>& lagrangian_z,
    const std::vector<double>& normal_target_r,
    const std::vector<double>& normal_target_z,
    std::vector<double>& target_r,
    std::vector<double>& target_z) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "transition-passage target composition requested without runtime");
  const AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.transition_passage_enabled && patch_index >= 0 &&
          patch_index < static_cast<int>(runtime.transition_patches.size()) &&
          alpha >= 0.0 && alpha <= 1.0,
      "transition-passage target composition arguments are invalid");
  const std::size_t n_nodes = static_cast<std::size_t>(runtime.n_nodes);
  TENRYU_ASSERT(
      lagrangian_r.size() == n_nodes && lagrangian_z.size() == n_nodes &&
          normal_target_r.size() == n_nodes &&
          normal_target_z.size() == n_nodes && target_r.size() == n_nodes &&
          target_z.size() == n_nodes,
      "transition-passage target composition node count mismatch");
  const AxisCoreTransitionPatch& patch =
      runtime.transition_patches[static_cast<std::size_t>(patch_index)];
  TENRYU_ASSERT(
      patch.state == AxisCoreTransitionPassageState::Passage &&
          patch.nodes.size() == patch.feather.size() &&
          patch.nodes.size() == patch.optimal_r.size() &&
          patch.nodes.size() == patch.optimal_z.size(),
      "transition-passage target composition patch storage mismatch");
  for (std::size_t local_node = 0;
       local_node < patch.nodes.size();
       ++local_node) {
    const int node = patch.nodes[local_node];
    const std::size_t node_index = static_cast<std::size_t>(node);
    const double feather = patch.feather[local_node];
    if (feather == 0.0) {
      target_r[node_index] = normal_target_r[node_index];
      target_z[node_index] = normal_target_z[node_index];
      continue;
    }
    target_r[node_index] =
        lagrangian_r[node_index] +
        alpha * feather * (patch.optimal_r[local_node] -
                           lagrangian_r[node_index]);
    target_z[node_index] =
        lagrangian_z[node_index] +
        alpha * feather * (patch.optimal_z[local_node] -
                           lagrangian_z[node_index]);
  }
}

const std::uint8_t* axis_core_transition_passage_inactive_cell_mask(
    const core::State& state,
    const int patch_index) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "transition-passage cell mask requested without initialized runtime");
  const AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.transition_passage_enabled &&
          runtime.n_cells == state.mesh.topo.n_cells && patch_index >= 0 &&
          patch_index < static_cast<int>(runtime.transition_patches.size()),
      "transition-passage cell mask patch is invalid");
  const AxisCoreTransitionPatch& patch =
      runtime.transition_patches[static_cast<std::size_t>(patch_index)];
  TENRYU_ASSERT(
      patch.d_inactive_cell_mask.size() ==
          static_cast<std::size_t>(runtime.n_cells),
      "transition-passage cell mask size mismatch");
  return patch.d_inactive_cell_mask.data();
}

double axis_core_transition_passage_max_swept_fraction(
    const core::State& state,
    const int patch_index,
    const std::vector<double>& candidate_r,
    const std::vector<double>& candidate_z) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "transition-passage swept gate requested without initialized runtime");
  AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.transition_passage_enabled && patch_index >= 0 &&
          patch_index < static_cast<int>(runtime.transition_patches.size()) &&
          candidate_r.size() == static_cast<std::size_t>(runtime.n_nodes) &&
          candidate_z.size() == static_cast<std::size_t>(runtime.n_nodes) &&
          state.vol.size() == static_cast<std::size_t>(runtime.n_cells),
      "transition-passage swept gate arguments are invalid");
  AxisCoreTransitionPatch& patch =
      runtime.transition_patches[static_cast<std::size_t>(patch_index)];
  runtime.d_transition_candidate_r.copy_from_host(candidate_r);
  runtime.d_transition_candidate_z.copy_from_host(candidate_z);
  constexpr int threads = 256;
  const int n_faces = static_cast<int>(patch.swept_cells.size());
  const int blocks = (n_faces + threads - 1) / threads;
  axis_core_transition_swept_fraction_kernel<<<blocks, threads>>>(
      patch.d_swept_cells.data(),
      patch.d_swept_local_faces.data(),
      patch.d_swept_neighbors.data(),
      n_faces,
      state.x_r.data(),
      state.x_z.data(),
      runtime.d_transition_candidate_r.data(),
      runtime.d_transition_candidate_z.data(),
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      runtime.d_transition_cell_orientation_sign.data(),
      runtime.d_transition_cell_nverts.data(),
      state.vol.data(),
      patch.d_swept_fraction.data());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<double> swept_fraction;
  patch.d_swept_fraction.copy_to_host(swept_fraction);
  double maximum = 0.0;
  for (const double fraction : swept_fraction) {
    if (!std::isfinite(fraction)) {
      return std::numeric_limits<double>::infinity();
    }
    maximum = std::max(maximum, fraction);
  }
  return maximum;
}

void axis_core_transition_passage_record_alpha(
    const int patch_index,
    const double alpha,
    const long long step,
    const double time,
    const bool emit_logs) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "transition-passage alpha record requested without initialized runtime");
  AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.transition_passage_enabled && patch_index >= 0 &&
          patch_index < static_cast<int>(runtime.transition_patches.size()),
      "transition-passage alpha record patch is invalid");
  AxisCoreTransitionPatch& patch =
      runtime.transition_patches[static_cast<std::size_t>(patch_index)];
  int histogram_bin = 9;
  for (int exponent = 0; exponent <= 8; ++exponent) {
    if (alpha == std::ldexp(1.0, -exponent)) {
      histogram_bin = exponent;
      break;
    }
  }
  TENRYU_ASSERT(
      (histogram_bin < 9 && alpha > 0.0) ||
          (histogram_bin == 9 && alpha == 0.0),
      "transition-passage recorded alpha is outside the fixed ladder");
  ++patch.alpha_hist[static_cast<std::size_t>(histogram_bin)];
  ++patch.passage_steps;

  const bool escalated_condition =
      alpha < (1.0 / 16.0) &&
      patch.inner_radius - patch.last_shock_radius <=
          4.0 * patch.median_inner_width;
  if (escalated_condition) {
    ++patch.low_alpha_consecutive;
    if (patch.low_alpha_consecutive >= 2 && !patch.escalated_latched) {
      patch.escalated_latched = true;
      ++patch.escalate_count;
      if (emit_logs) {
        std::ostringstream message;
        message << std::scientific << std::setprecision(17)
                << "[transition-passage] ESCALATE morph-needed"
                << " patch=" << patch.belt_block_id
                << " t=" << time
                << " count=" << patch.escalate_count;
        core::log_warning(message.str());
      }
    }
  } else {
    patch.low_alpha_consecutive = 0;
    patch.escalated_latched = false;
  }

  if (emit_logs && patch.passage_steps % 200U == 0U) {
    std::ostringstream message;
    message << "[transition-passage] steps=" << patch.passage_steps
            << " alpha_hist=[";
    for (std::size_t bin = 0; bin < patch.alpha_hist.size(); ++bin) {
      if (bin != 0U) {
        message << ',';
      }
      message << patch.alpha_hist[bin];
    }
    message << "] escalate=" << patch.escalate_count
            << " patch=" << patch.belt_block_id
            << " step=" << step;
    core::log_info(message.str());
  }
}

AxisCoreFollowRepairDecision axis_core_disk_select_always_moving() {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "axis-core always-moving selection requested without initialized "
      "runtime");
  AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  TENRYU_ASSERT(
      runtime.always_moving_mode,
      "axis-core always-moving selection requested outside always_moving "
      "mode");
  ++runtime.repair_count;
  return {
      true,
      runtime.follow_count,
      runtime.repair_count,
  };
}

void axis_core_disk_assert_plateau(const core::State& state) {
  TENRYU_ASSERT(
      g_axis_core_disk_runtime != nullptr,
      "axis-core plateau check requested without initialized runtime");
  const AxisCoreDiskRuntime& runtime = *g_axis_core_disk_runtime;
  std::vector<double> current_r;
  std::vector<double> current_z;
  state.x_r.copy_to_host(current_r);
  state.x_z.copy_to_host(current_z);
  TENRYU_ASSERT(
      current_r.size() == static_cast<std::size_t>(runtime.n_nodes) &&
          current_z.size() == static_cast<std::size_t>(runtime.n_nodes),
      "axis-core plateau check node count mismatch");
  for (int node = 0; node < runtime.n_nodes; ++node) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    if (runtime.plateau_node[node_index] == 0U) {
      continue;
    }
    TENRYU_ASSERT(
        std::bit_cast<std::uint64_t>(current_r[node_index]) ==
                std::bit_cast<std::uint64_t>(
                    runtime.captured_r[node_index]) &&
            std::bit_cast<std::uint64_t>(current_z[node_index]) ==
                std::bit_cast<std::uint64_t>(
                    runtime.captured_z[node_index]),
        "axis-core plateau node differs from its captured coordinate: node=" +
            std::to_string(node));
  }
}

}  // namespace tenryu::hydro
