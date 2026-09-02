#include "hydro/flank_tangential_strip.hpp"
#include "hydro/flank_tangential_strip_detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/corner_jacobian_quality.cuh"
#include "mesh/candidate_mesh_admissibility.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kFlankStripHardRatio = 0.05;

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2U;
  if ((values.size() & 1U) != 0U) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1U] + values[mid]);
}

int active_slot_cell(const core::State& state) {
  for (const core::EvacContactSlot& slot :
       state.contact_graph.records) {
    if (slot.state == core::EvacContactState::kActive) {
      return slot.cell;
    }
  }
  return -1;
}

void gather_cell_nodes(const core::State& state,
                       const std::vector<double>& x_r,
                       const std::vector<double>& x_z,
                       const int i,
                       const int j,
                       double* quad_r,
                       double* quad_z) {
  const int nodes[4] = {
      state.mesh.topo.node_index(i, j),
      state.mesh.topo.node_index(i + 1, j),
      state.mesh.topo.node_index(i + 1, j + 1),
      state.mesh.topo.node_index(i, j + 1),
  };
  for (int corner = 0; corner < 4; ++corner) {
    const std::size_t node = static_cast<std::size_t>(nodes[corner]);
    quad_r[corner] = x_r[node];
    quad_z[corner] = x_z[node];
  }
}

bool band_reference_is_zeroed(const core::State& state,
                              const int slot_cell,
                              const int band_layers,
                              const int band_halfwidth_j) {
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const auto& reference = state.evacuated_cells.flank_strip_q_theta_ref;
  if (reference.empty()) {
    return true;
  }
  if (reference.size() !=
      4U * static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    return true;
  }
  const int i_last = std::min(nr - 1, band_layers + 1);
  int j_first = -1;
  int j_last = -1;
  detail::flank_strip_seam_window(
      state, slot_cell, band_halfwidth_j, nz, j_first, j_last);
  for (int i = 1; i <= i_last; ++i) {
    for (int j = j_first; j <= j_last; ++j) {
      const int cell = state.mesh.topo.cell_index(i, j);
      const std::size_t base = 4U * static_cast<std::size_t>(cell);
      for (int corner = 0; corner < 4; ++corner) {
        if (reference[base + static_cast<std::size_t>(corner)] != 0.0) {
          return false;
        }
      }
    }
  }
  return true;
}

bool cell_is_excluded(const core::State& state, const int cell) {
  const std::size_t index = static_cast<std::size_t>(cell);
  const auto& contact_active = state.evacuated_cells.contact_active_mask;
  if (!contact_active.empty() && contact_active[index] != 0U) {
    return true;
  }
  const auto& collapsed = state.evacuated_cells.cell_axis_edge_collapsed;
  if (!collapsed.empty() && collapsed[index] != 0U) {
    return true;
  }
  return !state.hydro_active.empty() && state.hydro_active[index] == 0;
}

bool cell_is_geometry_policy_exempt(const core::State& state,
                                    const int cell) {
  const auto& exempt =
      state.evacuated_cells.geometry_policy_exempt_cells;
  const std::size_t index = static_cast<std::size_t>(cell);
  return index < exempt.size() && exempt[index] != 0U;
}

double band_minimum_quality_ratio(const core::State& state,
                                  const std::vector<double>& x_r,
                                  const std::vector<double>& x_z,
                                  const int slot_cell,
                                  const int band_layers,
                                  const int band_halfwidth_j,
                                  int& argmin_cell,
                                  const int focus_j) {
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const auto& reference = state.evacuated_cells.flank_strip_q_theta_ref;
  double worst_ratio = std::numeric_limits<double>::infinity();
  argmin_cell = -1;
  const int i_last = std::min(nr - 1, band_layers + 1);
  int j_first = -1;
  int j_last = -1;
  detail::flank_strip_seam_window(
      state, slot_cell, band_halfwidth_j, nz, j_first, j_last, focus_j);
  for (int i = 1; i <= i_last; ++i) {
    for (int j = j_first; j <= j_last; ++j) {
      const int cell = state.mesh.topo.cell_index(i, j);
      if (cell_is_excluded(state, cell)) {
        continue;
      }
      const std::size_t base = 4U * static_cast<std::size_t>(cell);
      double quad_r[4];
      double quad_z[4];
      gather_cell_nodes(state, x_r, x_z, i, j, quad_r, quad_z);
      for (int corner = 0; corner < 4; ++corner) {
        const double q_ref =
            reference[base + static_cast<std::size_t>(corner)];
        if (q_ref <= 0.0) {
          continue;
        }
        const double ratio =
            flank_strip_corner_angle_quality(quad_r, quad_z, corner) / q_ref;
        if (ratio < worst_ratio) {
          worst_ratio = ratio;
          argmin_cell = cell;
        }
      }
    }
  }
  return worst_ratio;
}

bool band_candidate_is_admissible(const core::State& state,
                                  const std::vector<double>& candidate_r,
                                  const std::vector<double>& candidate_z,
                                  const int slot_cell,
                                  const int band_layers,
                                  const int band_halfwidth_j,
                                  const double j_floor_rel,
                                  const int focus_j) {
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int i_last = std::min(nr - 1, band_layers + 1);
  int j_first = -1;
  int j_last = -1;
  detail::flank_strip_seam_window(
      state, slot_cell, band_halfwidth_j, nz, j_first, j_last, focus_j);
  for (int i = 1; i <= i_last; ++i) {
    for (int j = j_first; j <= j_last; ++j) {
      double candidate_quad_r[4];
      double candidate_quad_z[4];
      gather_cell_nodes(state,
                        candidate_r,
                        candidate_z,
                        i,
                        j,
                        candidate_quad_r,
                        candidate_quad_z);
      const int node_i[4] = {i, i + 1, i + 1, i};
      if (!trial_cell_admissible(
              candidate_quad_r, candidate_quad_z, node_i, j_floor_rel)) {
        return false;
      }
    }
  }
  return true;
}

struct UntanglerConstraint {
  double a = 0.0;
  double b = 0.0;
  double d = 0.0;
  double scale = 0.0;
};

struct UntanglerPlane {
  std::array<double, 3> coefficients{};
  double rhs = 0.0;
};

double determinant_3x3(
    const std::array<std::array<double, 3>, 3>& matrix) {
  return matrix[0][0] *
             (matrix[1][1] * matrix[2][2] -
              matrix[1][2] * matrix[2][1]) -
         matrix[0][1] *
             (matrix[1][0] * matrix[2][2] -
              matrix[1][2] * matrix[2][0]) +
         matrix[0][2] *
             (matrix[1][0] * matrix[2][1] -
              matrix[1][1] * matrix[2][0]);
}

bool solve_3x3(const std::array<UntanglerPlane, 3>& planes,
               std::array<double, 3>& solution) {
  std::array<std::array<double, 3>, 3> matrix{};
  double row_norm_product = 1.0;
  for (int row = 0; row < 3; ++row) {
    matrix[static_cast<std::size_t>(row)] =
        planes[static_cast<std::size_t>(row)].coefficients;
    const auto& coefficients =
        planes[static_cast<std::size_t>(row)].coefficients;
    row_norm_product *= std::sqrt(
        coefficients[0] * coefficients[0] +
        coefficients[1] * coefficients[1] +
        coefficients[2] * coefficients[2]);
  }
  const double determinant = determinant_3x3(matrix);
  const double singular_threshold =
      128.0 * std::numeric_limits<double>::epsilon() * row_norm_product;
  if (!std::isfinite(determinant) ||
      !std::isfinite(singular_threshold) ||
      std::abs(determinant) <= singular_threshold) {
    return false;
  }
  for (int column = 0; column < 3; ++column) {
    auto replaced = matrix;
    for (int row = 0; row < 3; ++row) {
      replaced[static_cast<std::size_t>(row)]
              [static_cast<std::size_t>(column)] =
          planes[static_cast<std::size_t>(row)].rhs;
    }
    solution[static_cast<std::size_t>(column)] =
        determinant_3x3(replaced) / determinant;
    if (!std::isfinite(solution[static_cast<std::size_t>(column)])) {
      return false;
    }
  }
  return true;
}

double untangler_objective(
    const std::vector<UntanglerConstraint>& constraints,
    const double r,
    const double z) {
  double objective = std::numeric_limits<double>::infinity();
  for (const UntanglerConstraint& constraint : constraints) {
    objective = std::min(
        objective,
        (constraint.a * r + constraint.b * z + constraint.d) /
            constraint.scale);
  }
  return objective;
}

bool untangler_candidate_is_better(const double t,
                                    const double r,
                                    const double z,
                                    const double best_t,
                                    const double best_r,
                                    const double best_z) {
  return t > best_t ||
         (t == best_t &&
          (r > best_r || (r == best_r && z > best_z)));
}

bool maximize_untangler_barrier(
    const std::vector<UntanglerConstraint>& constraints,
    const double r0,
    const double z0,
    const double trust_radius,
    double& optimized_r,
    double& optimized_z) {
  if (constraints.empty() || !(trust_radius > 0.0) ||
      !std::isfinite(trust_radius)) {
    return false;
  }

  const double r_min = r0 - trust_radius;
  const double r_max = r0 + trust_radius;
  const double z_min = z0 - trust_radius;
  const double z_max = z0 + trust_radius;
  if (!std::isfinite(r_min) || !std::isfinite(r_max) ||
      !std::isfinite(z_min) || !std::isfinite(z_max)) {
    return false;
  }
  std::vector<UntanglerPlane> planes;
  planes.reserve(constraints.size() + 4U);
  for (const UntanglerConstraint& constraint : constraints) {
    planes.push_back(
        {{{constraint.a, constraint.b, -constraint.scale}},
         -constraint.d});
  }
  planes.push_back({{{1.0, 0.0, 0.0}}, r_min});
  planes.push_back({{{-1.0, 0.0, 0.0}}, -r_max});
  planes.push_back({{{0.0, 1.0, 0.0}}, z_min});
  planes.push_back({{{0.0, -1.0, 0.0}}, -z_max});

  const double current_t = untangler_objective(constraints, r0, z0);
  if (!std::isfinite(current_t)) {
    return false;
  }
  double best_t = current_t;
  double best_r = r0;
  double best_z = z0;
  const double r_tolerance =
      128.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, std::abs(r_min), std::abs(r_max)});
  const double z_tolerance =
      128.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, std::abs(z_min), std::abs(z_max)});
  for (std::size_t first = 0; first < planes.size(); ++first) {
    for (std::size_t second = first + 1U;
         second < planes.size();
         ++second) {
      for (std::size_t third = second + 1U;
           third < planes.size();
           ++third) {
        const std::array<UntanglerPlane, 3> active = {
            planes[first], planes[second], planes[third]};
        std::array<double, 3> vertex{};
        if (!solve_3x3(active, vertex)) {
          continue;
        }
        bool feasible = true;
        for (const UntanglerPlane& plane : planes) {
          const double lhs =
              plane.coefficients[0] * vertex[0] +
              plane.coefficients[1] * vertex[1] +
              plane.coefficients[2] * vertex[2];
          const double evaluation_scale =
              std::abs(plane.coefficients[0] * vertex[0]) +
              std::abs(plane.coefficients[1] * vertex[1]) +
              std::abs(plane.coefficients[2] * vertex[2]) +
              std::abs(plane.rhs);
          const double feasibility_tolerance =
              256.0 * std::numeric_limits<double>::epsilon() *
              std::max(evaluation_scale,
                       std::numeric_limits<double>::min());
          if (!std::isfinite(lhs) || !std::isfinite(evaluation_scale) ||
              lhs < plane.rhs - feasibility_tolerance) {
            feasible = false;
            break;
          }
        }
        if (!feasible || vertex[0] < r_min - r_tolerance ||
            vertex[0] > r_max + r_tolerance ||
            vertex[1] < z_min - z_tolerance ||
            vertex[1] > z_max + z_tolerance) {
          continue;
        }
        const double r = std::clamp(vertex[0], r_min, r_max);
        const double z = std::clamp(vertex[1], z_min, z_max);
        const double t = untangler_objective(constraints, r, z);
        if (!std::isfinite(t) ||
            !untangler_candidate_is_better(
                t, r, z, best_t, best_r, best_z)) {
          continue;
        }
        best_t = t;
        best_r = r;
        best_z = z;
      }
    }
  }

  if (!(best_t > current_t)) {
    return false;
  }
  optimized_r = best_r;
  optimized_z = best_z;
  return true;
}

double local_mean_edge_length(const core::State& state,
                              const std::vector<double>& x_r,
                              const std::vector<double>& x_z,
                              const int node_i,
                              const int node_j) {
  constexpr int kNeighborDi[4] = {-1, 0, 0, 1};
  constexpr int kNeighborDj[4] = {0, -1, 1, 0};
  const int node = state.mesh.topo.node_index(node_i, node_j);
  double total_length = 0.0;
  int edge_count = 0;
  for (int neighbor = 0; neighbor < 4; ++neighbor) {
    const int neighbor_i = node_i + kNeighborDi[neighbor];
    const int neighbor_j = node_j + kNeighborDj[neighbor];
    if (neighbor_i < 0 || neighbor_i > state.mesh.topo.nr ||
        neighbor_j < 0 || neighbor_j > state.mesh.topo.nz) {
      continue;
    }
    const int neighbor_node =
        state.mesh.topo.node_index(neighbor_i, neighbor_j);
    const double length = std::hypot(
        x_r[static_cast<std::size_t>(neighbor_node)] -
            x_r[static_cast<std::size_t>(node)],
        x_z[static_cast<std::size_t>(neighbor_node)] -
            x_z[static_cast<std::size_t>(node)]);
    if (!(length > 0.0) || !std::isfinite(length)) {
      continue;
    }
    total_length += length;
    ++edge_count;
  }
  return edge_count > 0 ? total_length / static_cast<double>(edge_count)
                        : 0.0;
}

void run_local_jacobian_untangler_impl(
    const core::State& state,
    const int target_cell,
    const int i_first,
    const int i_last,
    const int j_first,
    const int j_last,
    const std::vector<std::uint8_t>& fixed_node,
    std::vector<double>& candidate_r,
    std::vector<double>& candidate_z) {
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (target_cell < 0 || target_cell >= state.mesh.topo.n_cells) {
    return;
  }
  const int target_j = target_cell % nz;
  const int neighborhood_cell_j_first = std::max(0, target_j - 2);
  const int neighborhood_cell_j_last = std::min(nz - 1, target_j + 2);
  const int movable_j_first =
      std::max(j_first + 1, neighborhood_cell_j_first);
  const int movable_j_last =
      std::min(j_last - 1, neighborhood_cell_j_last + 1);
  if (movable_j_first > movable_j_last) {
    return;
  }

  std::vector<std::uint8_t> free_node(candidate_r.size(), 0U);
  for (int i = i_first; i <= i_last; ++i) {
    for (int j = movable_j_first; j <= movable_j_last; ++j) {
      const int node = state.mesh.topo.node_index(i, j);
      if (fixed_node[static_cast<std::size_t>(node)] == 0U) {
        free_node[static_cast<std::size_t>(node)] = 1U;
      }
    }
  }

  for (int sweep = 0; sweep < 3; ++sweep) {
    for (std::size_t node_index = 0;
         node_index < free_node.size();
         ++node_index) {
      if (free_node[node_index] == 0U) {
        continue;
      }
      const int node = static_cast<int>(node_index);
      const int node_i = node / (nz + 1);
      const int node_j = node % (nz + 1);
      const double length_scale = local_mean_edge_length(
          state, candidate_r, candidate_z, node_i, node_j);
      const double probe_distance = 1.0e-3 * length_scale;
      if (!(probe_distance > 0.0) || !std::isfinite(probe_distance)) {
        continue;
      }

      const double r0 = candidate_r[node_index];
      const double z0 = candidate_z[node_index];
      std::vector<UntanglerConstraint> constraints;
      constraints.reserve(12U);
      const int cell_i_first = std::max(0, node_i - 1);
      const int cell_i_last = std::min(nr - 1, node_i);
      const int cell_j_first = std::max(0, node_j - 1);
      const int cell_j_last = std::min(nz - 1, node_j);
      for (int cell_i = cell_i_first; cell_i <= cell_i_last; ++cell_i) {
        for (int cell_j = cell_j_first;
             cell_j <= cell_j_last;
             ++cell_j) {
          const int cell = state.mesh.topo.cell_index(cell_i, cell_j);
          if (cell_is_geometry_policy_exempt(state, cell)) {
            continue;
          }
          const int cell_nodes[4] = {
              state.mesh.topo.node_index(cell_i, cell_j),
              state.mesh.topo.node_index(cell_i + 1, cell_j),
              state.mesh.topo.node_index(cell_i + 1, cell_j + 1),
              state.mesh.topo.node_index(cell_i, cell_j + 1),
          };
          int node_corner = -1;
          for (int corner = 0; corner < 4; ++corner) {
            if (cell_nodes[corner] == node) {
              node_corner = corner;
              break;
            }
          }
          if (node_corner < 0) {
            continue;
          }
          double quad_r[4];
          double quad_z[4];
          gather_cell_nodes(state,
                            candidate_r,
                            candidate_z,
                            cell_i,
                            cell_j,
                            quad_r,
                            quad_z);
          for (int corner = 0; corner < 4; ++corner) {
            if (corner == ((node_corner + 2) & 3)) {
              continue;
            }
            const double j0 = corner_jacobian_from_quad(
                quad_r, quad_z, corner);
            quad_r[node_corner] = r0 + probe_distance;
            const double probe_r_j = corner_jacobian_from_quad(
                quad_r, quad_z, corner);
            quad_r[node_corner] = r0;
            quad_z[node_corner] = z0 + probe_distance;
            const double probe_z_j = corner_jacobian_from_quad(
                quad_r, quad_z, corner);
            quad_z[node_corner] = z0;
            const double a = (probe_r_j - j0) / probe_distance;
            const double b = (probe_z_j - j0) / probe_distance;
            const double gradient = std::hypot(a, b);
            const double scale = gradient * length_scale;
            const double d = j0 - a * r0 - b * z0;
            if (!(scale > 0.0) || !std::isfinite(a) ||
                !std::isfinite(b) || !std::isfinite(d) ||
                !std::isfinite(scale)) {
              continue;
            }
            constraints.push_back({a, b, d, scale});
          }
        }
      }

      double optimized_r = r0;
      double optimized_z = z0;
      if (maximize_untangler_barrier(constraints,
                                     r0,
                                     z0,
                                     0.25 * length_scale,
                                     optimized_r,
                                     optimized_z)) {
        candidate_r[node_index] = optimized_r;
        candidate_z[node_index] = optimized_z;
      }
    }
  }
}

}  // namespace

double flank_strip_corner_angle_quality(const double* r,
                                        const double* z,
                                        const int corner) {
  const int kp = (corner + 1) & 3;
  const int km = (corner + 3) & 3;
  const double edge_plus_r = r[kp] - r[corner];
  const double edge_plus_z = z[kp] - z[corner];
  const double edge_minus_r = r[km] - r[corner];
  const double edge_minus_z = z[km] - z[corner];
  const double edge_plus_length = std::hypot(edge_plus_r, edge_plus_z);
  const double edge_minus_length = std::hypot(edge_minus_r, edge_minus_z);
  if (edge_plus_length == 0.0 || edge_minus_length == 0.0) {
    return 0.0;
  }
  return corner_jacobian_from_quad(r, z, corner) /
         (edge_plus_length * edge_minus_length);
}

namespace detail {

int active_flank_strip_slot_cell(const core::State& state) {
  return active_slot_cell(state);
}

void flank_strip_seam_window(const core::State& state,
                             const int slot_cell,
                             const int band_halfwidth_j,
                             const int nz,
                             int& j_first,
                             int& j_last,
                             const int focus_j) {
  const int slot_j = slot_cell % nz;
  int extent_min = slot_j;
  int extent_max = slot_j;
  bool has_collapsed_cell = false;
  const auto& collapsed = state.evacuated_cells.cell_axis_edge_collapsed;
  for (int j = 0; j < nz; ++j) {
    const int cell = state.mesh.topo.cell_index(0, j);
    const std::size_t index = static_cast<std::size_t>(cell);
    if (index >= collapsed.size() || collapsed[index] == 0U) {
      continue;
    }
    has_collapsed_cell = true;
    extent_min = std::min(extent_min, j);
    extent_max = std::max(extent_max, j);
  }
  if (focus_j >= 0) {
    // A failing band cell adjacent to the window is itself evidence that the
    // shear zone reaches there; repair must not reject that target merely for
    // lying one column outside the static margin.
    extent_min = std::min(extent_min, focus_j);
    extent_max = std::max(extent_max, focus_j);
  }
  j_first = std::max(0, extent_min - band_halfwidth_j);
  const int high_side_node_extension = has_collapsed_cell ? 1 : 0;
  j_last = std::min(
      nz - 1, extent_max + high_side_node_extension + band_halfwidth_j);
}

void flank_strip_extend_reference_for_window(
    core::State& state,
    const core::Config& cfg,
    const int slot_cell,
    const int band_layers,
    const int band_halfwidth_j,
    const int focus_j) {
  const auto& strip_cfg = cfg.numerics.ale.evacuated_cell.closure_contact
                              .flank_tangential_strip;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  if (!strip_cfg.enabled || state.mesh.dim != 2 || slot_cell < 0 ||
      slot_cell >= n_cells || nr <= 1 || nz <= 0 ||
      state.x_r.size() != n_nodes || state.x_z.size() != n_nodes) {
    return;
  }

  auto& reference = state.evacuated_cells.flank_strip_q_theta_ref;
  auto& h_reference = state.evacuated_cells.flank_strip_h_ref;
  const std::size_t reference_size =
      4U * static_cast<std::size_t>(n_cells);
  if ((!reference.empty() && reference.size() != reference_size) ||
      (!h_reference.empty() && h_reference.size() != n_nodes)) {
    return;
  }
  if (reference.empty()) {
    reference.assign(reference_size, 0.0);
  }
  if (h_reference.empty()) {
    h_reference.assign(n_nodes, 0.0);
  }

  std::vector<double> x_r;
  std::vector<double> x_z;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  const int i_last = std::min(nr - 1, band_layers + 1);
  int j_first = -1;
  int j_last = -1;
  flank_strip_seam_window(state,
                          slot_cell,
                          band_halfwidth_j,
                          nz,
                          j_first,
                          j_last,
                          focus_j);
  for (int i = 1; i <= i_last; ++i) {
    for (int j = j_first; j <= j_last; ++j) {
      const int cell = state.mesh.topo.cell_index(i, j);
      const std::size_t base = 4U * static_cast<std::size_t>(cell);
      double quad_r[4];
      double quad_z[4];
      gather_cell_nodes(state, x_r, x_z, i, j, quad_r, quad_z);
      for (int corner = 0; corner < 4; ++corner) {
        double& q_ref = reference[base + static_cast<std::size_t>(corner)];
        if (!(q_ref > 0.0) || !std::isfinite(q_ref)) {
          q_ref = flank_strip_corner_angle_quality(quad_r, quad_z, corner);
        }
      }
      if (j < j_last) {
        const int node = state.mesh.topo.node_index(i, j);
        double& h_ref = h_reference[static_cast<std::size_t>(node)];
        if (!(h_ref > 0.0) || !std::isfinite(h_ref)) {
          const int node_next = state.mesh.topo.node_index(i, j + 1);
          h_ref = std::hypot(
              x_r[static_cast<std::size_t>(node_next)] -
                  x_r[static_cast<std::size_t>(node)],
              x_z[static_cast<std::size_t>(node_next)] -
                  x_z[static_cast<std::size_t>(node)]);
        }
      }
    }
  }
  for (int j = j_first; j < j_last; ++j) {
    const int node = state.mesh.topo.node_index(0, j);
    double& h_ref = h_reference[static_cast<std::size_t>(node)];
    if (!(h_ref > 0.0) || !std::isfinite(h_ref)) {
      const int node_next = state.mesh.topo.node_index(0, j + 1);
      h_ref = std::hypot(
          x_r[static_cast<std::size_t>(node_next)] -
              x_r[static_cast<std::size_t>(node)],
          x_z[static_cast<std::size_t>(node_next)] -
              x_z[static_cast<std::size_t>(node)]);
    }
  }
}

bool flank_strip_cell_is_geometry_policy_exempt(const core::State& state,
                                                const int cell) {
  return cell_is_geometry_policy_exempt(state, cell);
}

bool flank_strip_node_is_contact_owned(const core::State& state,
                                       const int node) {
  if (node < 0 || node >= state.mesh.topo.n_nodes) {
    return false;
  }
  for (const core::EvacContactSlot& slot :
       state.contact_graph.records) {
    const bool active_slot =
        slot.state == core::EvacContactState::kActive;
    for (int pair = 0; pair < 2; ++pair) {
      if ((active_slot || slot.pair_engaged[pair] != 0U) &&
          (slot.node_a[pair] == node || slot.node_b[pair] == node)) {
        return true;
      }
    }
  }
  const auto& node_axis_alias = state.evacuated_cells.node_axis_alias;
  if (static_cast<std::size_t>(node) < node_axis_alias.size() &&
      node_axis_alias[static_cast<std::size_t>(node)] >= 0) {
    return true;
  }
  const int nz = state.mesh.topo.nz;
  if (nz <= 0) {
    return false;
  }
  const int node_i = node / (nz + 1);
  if (std::find(node_axis_alias.begin(), node_axis_alias.end(), node) !=
      node_axis_alias.end()) {
    return true;
  }
  const int node_j = node % (nz + 1);
  const auto& exempt =
      state.evacuated_cells.geometry_policy_exempt_cells;
  for (int cell_i = std::max(0, node_i - 1);
       cell_i <= std::min(state.mesh.topo.nr - 1, node_i);
       ++cell_i) {
    for (int cell_j = std::max(0, node_j - 1);
         cell_j <= std::min(nz - 1, node_j);
         ++cell_j) {
      const int cell = state.mesh.topo.cell_index(cell_i, cell_j);
      if (static_cast<std::size_t>(cell) < exempt.size() &&
          exempt[static_cast<std::size_t>(cell)] != 0U) {
        return true;
      }
    }
  }
  return false;
}

bool build_flank_strip_row_targets(
    const core::State& state,
    FlankStripRowPolyline& row) {
  const auto& h_reference = state.evacuated_cells.flank_strip_h_ref;
  const std::size_t row_size = row.nodes.size();
  row.targets_valid = false;
  if (row_size < 2U || row.s.size() != row_size ||
      row.fixed.size() != row_size ||
      h_reference.size() !=
          static_cast<std::size_t>(state.mesh.topo.n_nodes)) {
    return false;
  }
  row.target_s = row.s;

  std::size_t anchor_first = 0U;
  while (anchor_first < row_size - 1U) {
    std::size_t anchor_last = anchor_first + 1U;
    while (anchor_last < row_size - 1U &&
           row.fixed[anchor_last] == 0U) {
      ++anchor_last;
    }
    if (anchor_last == anchor_first + 1U) {
      anchor_first = anchor_last;
      continue;
    }
    double reference_length = 0.0;
    for (std::size_t local_j = anchor_first;
         local_j < anchor_last;
         ++local_j) {
      const int node = row.nodes[local_j];
      if (node < 0 || static_cast<std::size_t>(node) >= h_reference.size()) {
        return false;
      }
      const double h = h_reference[static_cast<std::size_t>(node)];
      if (!(h > 0.0) || !std::isfinite(h)) {
        return false;
      }
      reference_length += h;
    }
    if (!(reference_length > 0.0) || !std::isfinite(reference_length)) {
      return false;
    }
    double reference_prefix = 0.0;
    const double current_length =
        row.s[anchor_last] - row.s[anchor_first];
    for (std::size_t local_j = anchor_first + 1U;
         local_j < anchor_last;
         ++local_j) {
      reference_prefix += h_reference[static_cast<std::size_t>(
          row.nodes[local_j - 1U])];
      row.target_s[local_j] =
          row.s[anchor_first] +
          current_length * reference_prefix / reference_length;
    }
    anchor_first = anchor_last;
  }
  row.targets_valid = true;
  return true;
}

template <typename CopySeam>
double flank_strip_slip_mismatch_signed_ratio_impl(
    const core::State& state,
    const int j_first,
    const int j_last,
    CopySeam&& copy_seam) {
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  const auto& h_reference = state.evacuated_cells.flank_strip_h_ref;
  if (state.mesh.dim != 2 || active_flank_strip_slot_cell(state) < 0 ||
      nr <= 1 || nz <= 0 || h_reference.size() != n_nodes) {
    return 0.0;
  }
  if (j_first < 0 || j_first >= j_last || j_last >= nz) {
    return 0.0;
  }

  const int window_size = j_last - j_first + 1;
  const int first_node = state.mesh.topo.node_index(1, j_first);
  std::vector<double> seam_r(static_cast<std::size_t>(window_size));
  std::vector<double> seam_z(static_cast<std::size_t>(window_size));
  copy_seam(first_node, seam_r, seam_z);

  FlankStripRowPolyline seam_row;
  seam_row.i = 1;
  seam_row.nodes.reserve(static_cast<std::size_t>(window_size));
  seam_row.r.reserve(static_cast<std::size_t>(window_size));
  seam_row.z.reserve(static_cast<std::size_t>(window_size));
  seam_row.s.assign(static_cast<std::size_t>(window_size), 0.0);
  seam_row.target_s.assign(static_cast<std::size_t>(window_size), 0.0);
  seam_row.fixed.assign(static_cast<std::size_t>(window_size), 0U);
  for (int local_j = 0; local_j < window_size; ++local_j) {
    const int node = state.mesh.topo.node_index(1, j_first + local_j);
    seam_row.nodes.push_back(node);
    seam_row.r.push_back(seam_r[static_cast<std::size_t>(local_j)]);
    seam_row.z.push_back(seam_z[static_cast<std::size_t>(local_j)]);
    seam_row.fixed[static_cast<std::size_t>(local_j)] =
        flank_strip_node_is_contact_owned(state, node) ? 1U : 0U;
    if (local_j > 0) {
      seam_row.s[static_cast<std::size_t>(local_j)] =
          seam_row.s[static_cast<std::size_t>(local_j - 1)] +
          std::hypot(
              seam_row.r[static_cast<std::size_t>(local_j)] -
                  seam_row.r[static_cast<std::size_t>(local_j - 1)],
              seam_row.z[static_cast<std::size_t>(local_j)] -
                  seam_row.z[static_cast<std::size_t>(local_j - 1)]);
    }
  }
  seam_row.fixed.front() = 1U;
  seam_row.fixed.back() = 1U;
  if (!build_flank_strip_row_targets(state, seam_row)) {
    return 0.0;
  }

  double mismatch_max_ratio = 0.0;
  double mismatch_signed_ratio = 0.0;
  for (int local_j = 1; local_j < window_size - 1; ++local_j) {
    if (seam_row.fixed[static_cast<std::size_t>(local_j)] != 0U) {
      continue;
    }
    const double h_s = h_reference[static_cast<std::size_t>(
        seam_row.nodes[static_cast<std::size_t>(local_j)])];
    const double delta_s =
        seam_row.s[static_cast<std::size_t>(local_j)] -
        seam_row.target_s[static_cast<std::size_t>(local_j)];
    const double mismatch_ratio = std::abs(delta_s) / h_s;
    if (mismatch_ratio > mismatch_max_ratio) {
      mismatch_max_ratio = mismatch_ratio;
      mismatch_signed_ratio = delta_s / h_s;
    }
  }
  return mismatch_signed_ratio;
}

double flank_strip_slip_mismatch_signed_ratio(const core::State& state,
                                              const int j_first,
                                              const int j_last) {
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  if (state.x_r.size() != n_nodes || state.x_z.size() != n_nodes) {
    return 0.0;
  }
  return flank_strip_slip_mismatch_signed_ratio_impl(
      state,
      j_first,
      j_last,
      [&](const int first_node,
          std::vector<double>& seam_r,
          std::vector<double>& seam_z) {
        cudaError_t err = cudaMemcpy(seam_r.data(),
                                     state.x_r.data() + first_node,
                                     seam_r.size() * sizeof(double),
                                     cudaMemcpyDeviceToHost);
        TENRYU_ASSERT(err == cudaSuccess,
                      "flank strip slip mismatch r copy failed");
        err = cudaMemcpy(seam_z.data(),
                         state.x_z.data() + first_node,
                         seam_z.size() * sizeof(double),
                         cudaMemcpyDeviceToHost);
        TENRYU_ASSERT(err == cudaSuccess,
                      "flank strip slip mismatch z copy failed");
      });
}

double flank_strip_slip_mismatch_signed_ratio(
    const core::State& state,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    const int j_first,
    const int j_last) {
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  if (x_r.size() != n_nodes || x_z.size() != n_nodes) {
    return 0.0;
  }
  return flank_strip_slip_mismatch_signed_ratio_impl(
      state,
      j_first,
      j_last,
      [&](const int first_node,
          std::vector<double>& seam_r,
          std::vector<double>& seam_z) {
        std::copy_n(x_r.begin() + first_node, seam_r.size(), seam_r.begin());
        std::copy_n(x_z.begin() + first_node, seam_z.size(), seam_z.begin());
      });
}

void run_local_jacobian_untangler(
    const core::State& state,
    const int target_cell,
    const int i_first,
    const int i_last,
    const int j_first,
    const int j_last,
    const std::vector<std::uint8_t>& fixed_node,
    std::vector<double>& candidate_r,
    std::vector<double>& candidate_z) {
  run_local_jacobian_untangler_impl(state,
                                    target_cell,
                                    i_first,
                                    i_last,
                                    j_first,
                                    j_last,
                                    fixed_node,
                                    candidate_r,
                                    candidate_z);
}

double flank_strip_row_blend(const int i, const int band_layers) {
  const double eta = static_cast<double>(i - 1) /
                     static_cast<double>(band_layers + 1);
  return 1.0 - 3.0 * eta * eta + 2.0 * eta * eta * eta;
}

FlankStripWindowPolylines build_flank_strip_window_polylines(
    const core::State& state,
    const core::Config& cfg,
    const int slot_cell,
    const std::vector<double>& original_r,
    const std::vector<double>& original_z,
    const int focus_j) {
  FlankStripWindowPolylines window;
  window.slot_cell = slot_cell;
  const auto& strip_cfg = cfg.numerics.ale.evacuated_cell.closure_contact
                              .flank_tangential_strip;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  const auto& h_reference = state.evacuated_cells.flank_strip_h_ref;
  if (slot_cell < 0 || slot_cell >= state.mesh.topo.n_cells || nr <= 1 ||
      nz <= 0 || original_r.size() != n_nodes ||
      original_z.size() != n_nodes || h_reference.size() != n_nodes) {
    return window;
  }

  window.slot_j = slot_cell % nz;
  window.i_last = std::min(nr - 1, strip_cfg.band_layers + 1);
  flank_strip_seam_window(state,
                          slot_cell,
                          strip_cfg.band_halfwidth_j,
                          nz,
                          window.j_first,
                          window.j_last,
                          focus_j);
  window.window_size = window.j_last - window.j_first + 1;
  window.fixed_node.assign(n_nodes, 0U);

  for (int i = 0; i <= window.i_last; ++i) {
    for (int j = window.j_first; j <= window.j_last; ++j) {
      const int node = state.mesh.topo.node_index(i, j);
      window.fixed_node[static_cast<std::size_t>(node)] =
          flank_strip_node_is_contact_owned(state, node) ? 1U : 0U;
    }
  }
  bool targets_valid = window.window_size >= 2;
  window.rows.reserve(
      static_cast<std::size_t>(std::max(window.i_last, 0)));
  for (int i = 1; i <= window.i_last && targets_valid; ++i) {
    FlankStripRowPolyline row;
    row.i = i;
    row.nodes.reserve(static_cast<std::size_t>(window.window_size));
    row.r.reserve(static_cast<std::size_t>(window.window_size));
    row.z.reserve(static_cast<std::size_t>(window.window_size));
    row.s.assign(static_cast<std::size_t>(window.window_size), 0.0);
    row.target_s.assign(static_cast<std::size_t>(window.window_size), 0.0);
    row.delta_s.assign(static_cast<std::size_t>(window.window_size), 0.0);
    row.fixed.assign(static_cast<std::size_t>(window.window_size), 0U);
    for (int local_j = 0; local_j < window.window_size; ++local_j) {
      const int node =
          state.mesh.topo.node_index(i, window.j_first + local_j);
      row.nodes.push_back(node);
      row.r.push_back(original_r[static_cast<std::size_t>(node)]);
      row.z.push_back(original_z[static_cast<std::size_t>(node)]);
      row.fixed[static_cast<std::size_t>(local_j)] =
          window.fixed_node[static_cast<std::size_t>(node)];
      if (local_j > 0) {
        row.s[static_cast<std::size_t>(local_j)] =
            row.s[static_cast<std::size_t>(local_j - 1)] +
            std::hypot(
                row.r[static_cast<std::size_t>(local_j)] -
                    row.r[static_cast<std::size_t>(local_j - 1)],
                row.z[static_cast<std::size_t>(local_j)] -
                    row.z[static_cast<std::size_t>(local_j - 1)]);
      }
    }
    row.fixed.front() = 1U;
    row.fixed.back() = 1U;
    targets_valid = build_flank_strip_row_targets(state, row);

    const double weight = flank_strip_row_blend(i, strip_cfg.band_layers);
    for (int local_j = 1; local_j < window.window_size - 1; ++local_j) {
      if (row.fixed[static_cast<std::size_t>(local_j)] != 0U) {
        continue;
      }
      const double cap = 0.25 * std::min(
          row.s[static_cast<std::size_t>(local_j)] -
              row.s[static_cast<std::size_t>(local_j - 1)],
          row.s[static_cast<std::size_t>(local_j + 1)] -
              row.s[static_cast<std::size_t>(local_j)]);
      const double displacement = weight *
          (row.target_s[static_cast<std::size_t>(local_j)] -
           row.s[static_cast<std::size_t>(local_j)]);
      row.delta_s[static_cast<std::size_t>(local_j)] =
          std::clamp(displacement, -cap, cap);
    }
    window.rows.push_back(std::move(row));
  }
  window.valid = targets_valid;
  return window;
}

bool interpolate_flank_strip_polyline(
    const FlankStripRowPolyline& row,
    const double candidate_s,
    double& candidate_r,
    double& candidate_z) {
  if (row.s.size() < 2U || row.r.size() != row.s.size() ||
      row.z.size() != row.s.size() || !std::isfinite(candidate_s)) {
    return false;
  }
  const double clamped_s =
      std::clamp(candidate_s, row.s.front(), row.s.back());
  const auto upper = std::upper_bound(
      row.s.begin(), row.s.end(), clamped_s);
  const std::size_t segment =
      upper == row.s.end()
          ? row.s.size() - 2U
          : static_cast<std::size_t>(upper - row.s.begin() - 1);
  const double segment_length = row.s[segment + 1U] - row.s[segment];
  if (!(segment_length > 0.0)) {
    return false;
  }
  const double fraction = std::clamp(
      (clamped_s - row.s[segment]) / segment_length, 0.0, 1.0);
  candidate_r = row.r[segment] +
                fraction * (row.r[segment + 1U] - row.r[segment]);
  candidate_z = row.z[segment] +
                fraction * (row.z[segment + 1U] - row.z[segment]);
  return true;
}

double flank_strip_band_minimum_quality_ratio(
    const core::State& state,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    const int slot_cell,
    const int band_layers,
    const int band_halfwidth_j,
    int& argmin_cell,
    const int focus_j) {
  return band_minimum_quality_ratio(state,
                                    x_r,
                                    x_z,
                                    slot_cell,
                                    band_layers,
                                    band_halfwidth_j,
                                    argmin_cell,
                                    focus_j);
}

}  // namespace detail

void flank_strip_capture_reference(core::State& state,
                                   const core::Config& cfg) {
  const auto& strip_cfg = cfg.numerics.ale.evacuated_cell.closure_contact
                              .flank_tangential_strip;
  const int slot_cell = detail::active_flank_strip_slot_cell(state);
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  if (!strip_cfg.enabled || state.mesh.dim != 2 || slot_cell < 0 ||
      nr <= 1 || nz <= 0) {
    return;
  }

  auto& reference = state.evacuated_cells.flank_strip_q_theta_ref;
  if (reference.empty()) {
    reference.assign(4U * static_cast<std::size_t>(n_cells), 0.0);
  }
  if (reference.size() != 4U * static_cast<std::size_t>(n_cells)) {
    return;
  }
  auto& h_reference = state.evacuated_cells.flank_strip_h_ref;
  h_reference.assign(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0.0);

  std::vector<double> x_r;
  std::vector<double> x_z;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  const int i_last = std::min(nr - 1, strip_cfg.band_layers + 1);
  int j_first = -1;
  int j_last = -1;
  detail::flank_strip_seam_window(
      state, slot_cell, strip_cfg.band_halfwidth_j, nz, j_first, j_last);
  for (int i = 1; i <= i_last; ++i) {
    for (int j = j_first; j <= j_last; ++j) {
      const int cell = state.mesh.topo.cell_index(i, j);
      const std::size_t base = 4U * static_cast<std::size_t>(cell);
      double quad_r[4];
      double quad_z[4];
      gather_cell_nodes(state, x_r, x_z, i, j, quad_r, quad_z);
      for (int corner = 0; corner < 4; ++corner) {
        reference[base + static_cast<std::size_t>(corner)] =
            flank_strip_corner_angle_quality(quad_r, quad_z, corner);
      }
      if (j < j_last) {
        const int node = state.mesh.topo.node_index(i, j);
        const int node_next = state.mesh.topo.node_index(i, j + 1);
        h_reference[static_cast<std::size_t>(node)] = std::hypot(
            x_r[static_cast<std::size_t>(node_next)] -
                x_r[static_cast<std::size_t>(node)],
            x_z[static_cast<std::size_t>(node_next)] -
                x_z[static_cast<std::size_t>(node)]);
      }
    }
  }
  for (int j = j_first; j < j_last; ++j) {
    const int node = state.mesh.topo.node_index(0, j);
    const int node_next = state.mesh.topo.node_index(0, j + 1);
    h_reference[static_cast<std::size_t>(node)] = std::hypot(
        x_r[static_cast<std::size_t>(node_next)] -
            x_r[static_cast<std::size_t>(node)],
        x_z[static_cast<std::size_t>(node_next)] -
            x_z[static_cast<std::size_t>(node)]);
  }
}

FlankStripEvaluation flank_strip_update_and_evaluate(
    core::State& state,
    const core::Config& cfg,
    const double dt_step) {
  FlankStripEvaluation result;
  const auto& strip_cfg = cfg.numerics.ale.evacuated_cell.closure_contact
                              .flank_tangential_strip;
  const int slot_cell = detail::active_flank_strip_slot_cell(state);
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (!strip_cfg.enabled || state.mesh.dim != 2 || slot_cell < 0 ||
      nr <= 1 || nz <= 0) {
    return result;
  }

  auto& monitor = state.evacuated_cells.flank_strip_monitor;
  if (monitor.slot_cell != slot_cell) {
    monitor = core::EvacuatedCellsState::FlankStripMonitor{};
    monitor.slot_cell = slot_cell;
  }
  if (band_reference_is_zeroed(state,
                               slot_cell,
                               strip_cfg.band_layers,
                               strip_cfg.band_halfwidth_j)) {
    flank_strip_capture_reference(state, cfg);
  }
  const auto& reference = state.evacuated_cells.flank_strip_q_theta_ref;
  if (reference.size() !=
      4U * static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    result.armed_now = monitor.armed;
    return result;
  }

  std::vector<double> x_r;
  std::vector<double> x_z;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  double worst_ratio = std::numeric_limits<double>::infinity();
  int worst_cell = -1;
  int worst_corner = -1;
  const int i_last = std::min(nr - 1, strip_cfg.band_layers + 1);
  int j_first = -1;
  int j_last = -1;
  detail::flank_strip_seam_window(
      state, slot_cell, strip_cfg.band_halfwidth_j, nz, j_first, j_last);
  for (int i = 1; i <= i_last; ++i) {
    for (int j = j_first; j <= j_last; ++j) {
      const int cell = state.mesh.topo.cell_index(i, j);
      if (cell_is_excluded(state, cell)) {
        continue;
      }
      const std::size_t base = 4U * static_cast<std::size_t>(cell);
      double quad_r[4];
      double quad_z[4];
      gather_cell_nodes(state, x_r, x_z, i, j, quad_r, quad_z);
      for (int corner = 0; corner < 4; ++corner) {
        const double q_ref =
            reference[base + static_cast<std::size_t>(corner)];
        if (q_ref <= 0.0) {
          continue;
        }
        const double ratio =
            flank_strip_corner_angle_quality(quad_r, quad_z, corner) / q_ref;
        if (ratio < worst_ratio) {
          worst_ratio = ratio;
          worst_cell = cell;
          worst_corner = corner;
        }
      }
    }
  }
  if (worst_cell < 0) {
    result.armed_now = monitor.armed;
    return result;
  }

  monitor.worst_cell = worst_cell;
  monitor.worst_corner = worst_corner;
  monitor.worst_quality_ratio = worst_ratio;
  result.worst_ratio = worst_ratio;
  result.worst_cell = worst_cell;
  result.worst_corner = worst_corner;

  bool sampled_now = false;
  if (state.step != monitor.last_sample_step) {
    const int head = monitor.ring_head;
    monitor.lnq[head] = std::log(std::max(
        worst_ratio, std::numeric_limits<double>::min()));
    monitor.dt_sample[head] = dt_step;
    monitor.ring_head =
        (head + 1) % core::EvacuatedCellsState::FlankStripMonitor::kRingCapacity;
    monitor.n_samples = std::min(
        monitor.n_samples + 1,
        core::EvacuatedCellsState::FlankStripMonitor::kRingCapacity);
    monitor.last_sample_step = state.step;
    sampled_now = true;
  }

  const int capacity =
      core::EvacuatedCellsState::FlankStripMonitor::kRingCapacity;
  const int oldest =
      monitor.n_samples == capacity ? monitor.ring_head : 0;
  const auto ring_index = [&](const int sample) {
    return (oldest + sample) % capacity;
  };
  std::vector<double> slope_samples;
  if (monitor.n_samples > 1) {
    slope_samples.reserve(
        static_cast<std::size_t>(monitor.n_samples - 1));
  }
  for (int sample = 1; sample < monitor.n_samples; ++sample) {
    const int index = ring_index(sample);
    const int older = ring_index(sample - 1);
    if (monitor.dt_sample[index] > 0.0) {
      slope_samples.push_back(
          (monitor.lnq[index] - monitor.lnq[older]) /
          monitor.dt_sample[index]);
    }
  }
  const double slope = median(std::move(slope_samples));
  if (slope < 0.0 && dt_step > 0.0) {
    result.n_sliver_pred =
        std::log(std::max(worst_ratio,
                          std::numeric_limits<double>::min()) /
                 kFlankStripHardRatio) /
        (-slope) / dt_step;
  }

  result.would_fire =
      worst_ratio < strip_cfg.arm_quality_ratio ||
      result.n_sliver_pred < static_cast<double>(strip_cfg.lead_steps);
  if (result.would_fire) {
    monitor.armed = true;
    monitor.release_streak = 0;
  } else if (monitor.armed && sampled_now) {
    if (worst_ratio > strip_cfg.release_quality_ratio) {
      ++monitor.release_streak;
      if (monitor.release_streak >= strip_cfg.release_persistence_steps) {
        monitor.armed = false;
        monitor.release_streak = 0;
      }
    } else {
      monitor.release_streak = 0;
    }
  }
  result.armed_now = monitor.armed;
  return result;
}

FlankStripRezoneResult run_flank_tangential_strip_rezone(
    core::State& state,
    const core::Config& cfg,
    const int focus_cell) {
  FlankStripRezoneResult result;
  const auto& strip_cfg = cfg.numerics.ale.evacuated_cell.closure_contact
                              .flank_tangential_strip;
  const int slot_cell = detail::active_flank_strip_slot_cell(state);
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  if (!strip_cfg.enabled || state.mesh.dim != 2 || slot_cell < 0 ||
      nr <= 1 || nz <= 0) {
    return result;
  }
  result.triggered = true;
  const int focus_i =
      focus_cell >= 0 && focus_cell < state.mesh.topo.n_cells
          ? focus_cell / nz
          : -1;
  const bool focus_is_in_band =
      focus_i >= 1 && focus_i <= std::min(nr - 1, strip_cfg.band_layers);
  result.target_cell =
      focus_is_in_band
          ? focus_cell
          : state.evacuated_cells.flank_strip_monitor.worst_cell;
  const int focus_j = result.target_cell % nz;
  detail::flank_strip_extend_reference_for_window(
      state,
      cfg,
      slot_cell,
      strip_cfg.band_layers,
      strip_cfg.band_halfwidth_j,
      focus_j);
  const auto& q_reference =
      state.evacuated_cells.flank_strip_q_theta_ref;
  const auto& h_reference = state.evacuated_cells.flank_strip_h_ref;
  if (q_reference.size() !=
          4U * static_cast<std::size_t>(state.mesh.topo.n_cells) ||
      h_reference.size() != n_nodes) {
    return result;
  }

  std::vector<double> original_r;
  std::vector<double> original_z;
  state.x_r.copy_to_host(original_r);
  state.x_z.copy_to_host(original_z);
  result.q_before = detail::flank_strip_band_minimum_quality_ratio(
      state,
      original_r,
      original_z,
      slot_cell,
      strip_cfg.band_layers,
      strip_cfg.band_halfwidth_j,
      result.band_argmin_cell_before,
      focus_j);

  const detail::FlankStripWindowPolylines window =
      detail::build_flank_strip_window_polylines(
          state, cfg, slot_cell, original_r, original_z, focus_j);
  const int i_last = window.i_last;
  const int j_first = window.j_first;
  const int j_last = window.j_last;
  const int window_size = window.window_size;
  const std::vector<std::uint8_t>& fixed_node = window.fixed_node;
  const std::vector<detail::FlankStripRowPolyline>& rows = window.rows;
  const bool targets_valid = window.valid;

  result.slip_mismatch_signed_ratio =
      detail::flank_strip_slip_mismatch_signed_ratio(
          state, original_r, original_z, j_first, j_last);
  result.slip_mismatch_max_ratio =
      std::abs(result.slip_mismatch_signed_ratio);

  const auto capture_target_corner_j =
      [&](const std::vector<double>& x_r,
          const std::vector<double>& x_z,
          double* target_corner_j) {
        if (result.target_cell < 0 ||
            result.target_cell >= state.mesh.topo.n_cells) {
          return;
        }
        const int target_i = result.target_cell / nz;
        const int target_j = result.target_cell % nz;
        double quad_r[4];
        double quad_z[4];
        gather_cell_nodes(
            state, x_r, x_z, target_i, target_j, quad_r, quad_z);
        for (int corner = 0; corner < 4; ++corner) {
          target_corner_j[corner] =
              tenryu::hydro::corner_jacobian_from_quad(
                  quad_r, quad_z, corner);
        }
      };
  capture_target_corner_j(
      original_r, original_z, result.target_corner_j_before);
  const double target_min_corner_j_before =
      std::min({result.target_corner_j_before[0],
                result.target_corner_j_before[1],
                result.target_corner_j_before[2],
                result.target_corner_j_before[3]});

  const auto candidate_is_accepted =
      [&](const std::vector<double>& x_r,
          const std::vector<double>& x_z) {
        result.q_after = detail::flank_strip_band_minimum_quality_ratio(
            state,
            x_r,
            x_z,
            slot_cell,
            strip_cfg.band_layers,
            strip_cfg.band_halfwidth_j,
            result.band_argmin_cell_after,
            focus_j);
        const bool quality_accepted =
            std::isfinite(result.q_after) &&
            (result.q_after >=
                 result.q_before * strip_cfg.min_progress_factor ||
             result.q_after >=
                 result.q_before +
                     0.25 *
                         (strip_cfg.release_quality_ratio - result.q_before) ||
             result.q_after > strip_cfg.release_quality_ratio);
        double target_corner_j_after[4] = {};
        capture_target_corner_j(x_r, x_z, target_corner_j_after);
        const double target_min_corner_j_after =
            std::min({target_corner_j_after[0],
                      target_corner_j_after[1],
                      target_corner_j_after[2],
                      target_corner_j_after[3]});
        const bool focus_accepted =
            !focus_is_in_band ||
            target_min_corner_j_after > target_min_corner_j_before;
        return quality_accepted && focus_accepted &&
               band_candidate_is_admissible(
                   state,
                   x_r,
                   x_z,
                   slot_cell,
                   strip_cfg.band_layers,
                   strip_cfg.band_halfwidth_j,
                   mesh::CandidateMeshAdmissibilityFloors{}.gauss_j_rel,
                   focus_j);
      };
  const auto accept_candidate =
      [&](const std::vector<double>& x_r,
          const std::vector<double>& x_z,
          const int used_untangler) {
        if (!candidate_is_accepted(x_r, x_z)) {
          return false;
        }
        result.converged = true;
        result.used_untangler = used_untangler;
        for (std::size_t node = 0; node < n_nodes; ++node) {
          const double displacement =
              std::hypot(x_r[node] - original_r[node],
                         x_z[node] - original_z[node]);
          if (!(displacement > 0.0)) {
            continue;
          }
          result.max_displacement =
              std::max(result.max_displacement, displacement);
          const int i = static_cast<int>(node) / (nz + 1);
          const int j = static_cast<int>(node) % (nz + 1);
          if (i < 1 || i > strip_cfg.band_layers + 1 ||
              j < j_first || j > j_last) {
            ++result.moved_nodes_outside_band;
          }
        }
        state.x_r.copy_from_host(x_r);
        state.x_z.copy_from_host(x_z);
        capture_target_corner_j(x_r, x_z, result.target_corner_j_after);
        return true;
      };

  std::vector<double> candidate_r = original_r;
  std::vector<double> candidate_z = original_z;
  bool equidistribution_trials_failed = false;
  if (targets_valid && std::isfinite(result.q_before)) {
    double alpha = 1.0;
    for (int trial = 0; trial < 4; ++trial) {
      candidate_r = original_r;
      candidate_z = original_z;
      for (const detail::FlankStripRowPolyline& row : rows) {
        for (int local_j = 1; local_j < window_size - 1; ++local_j) {
          const double delta_s =
              row.delta_s[static_cast<std::size_t>(local_j)];
          if (delta_s == 0.0) {
            continue;
          }
          const double candidate_s =
              row.s[static_cast<std::size_t>(local_j)] + alpha * delta_s;
          double interpolated_r = 0.0;
          double interpolated_z = 0.0;
          if (!detail::interpolate_flank_strip_polyline(
                  row, candidate_s, interpolated_r, interpolated_z)) {
            continue;
          }
          const int node = row.nodes[static_cast<std::size_t>(local_j)];
          candidate_r[static_cast<std::size_t>(node)] = interpolated_r;
          candidate_z[static_cast<std::size_t>(node)] = interpolated_z;
        }
      }

      result.alpha_halvings = trial;
      if (accept_candidate(candidate_r, candidate_z, 0)) {
        return result;
      }
      if (trial < 3) {
        alpha *= 0.5;
      }
    }
    equidistribution_trials_failed = true;
  }

  if (equidistribution_trials_failed && strip_cfg.untangler_enabled &&
      result.target_cell >= 0 &&
      result.target_cell < state.mesh.topo.n_cells) {
    candidate_r = original_r;
    candidate_z = original_z;
    std::vector<std::uint8_t> untangler_fixed_node = fixed_node;
    for (int cell = 0; cell < state.mesh.topo.n_cells; ++cell) {
      if (!cell_is_geometry_policy_exempt(state, cell)) {
        continue;
      }
      const int cell_i = cell / nz;
      const int cell_j = cell % nz;
      const int exempt_nodes[4] = {
          state.mesh.topo.node_index(cell_i, cell_j),
          state.mesh.topo.node_index(cell_i + 1, cell_j),
          state.mesh.topo.node_index(cell_i + 1, cell_j + 1),
          state.mesh.topo.node_index(cell_i, cell_j + 1),
      };
      for (const int node : exempt_nodes) {
        untangler_fixed_node[static_cast<std::size_t>(node)] = 1U;
      }
    }
    detail::run_local_jacobian_untangler(state,
                                         result.target_cell,
                                         1,
                                         i_last,
                                         j_first,
                                         j_last,
                                         untangler_fixed_node,
                                         candidate_r,
                                         candidate_z);
    if (accept_candidate(candidate_r, candidate_z, 1)) {
      return result;
    }
  }

  state.x_r.copy_from_host(original_r);
  state.x_z.copy_from_host(original_z);
  capture_target_corner_j(
      original_r, original_z, result.target_corner_j_after);
  return result;
}

}  // namespace tenryu::hydro
