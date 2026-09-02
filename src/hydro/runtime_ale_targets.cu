#include "hydro/runtime_ale_targets.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <sstream>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/ale_rezone.cuh"
#include "hydro/ale_state_monitor.hpp"
#include "hydro/button_morph_indexing.hpp"
#include "hydro/cfl.hpp"
#include "hydro/reference_barrier_ale.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {
namespace {

using button_morph::ButtonIndexing;

struct ClampOutcome {
  bool total = false;
  bool normal = false;
};

ClampOutcome clamp_displacement_impl(double* const dr,
                                     double* const dz,
                                     const double h_p,
                                     const double cap_frac,
                                     const double cap_normal_frac,
                                     const bool has_normal,
                                     const double n_r,
                                     const double n_z) {
  ClampOutcome outcome;
  if (dr == nullptr || dz == nullptr || !(std::isfinite(h_p) && h_p > 0.0) ||
      !(std::isfinite(cap_frac) && cap_frac > 0.0) ||
      !std::isfinite(*dr) || !std::isfinite(*dz)) {
    return outcome;
  }

  const double total_cap = cap_frac * h_p;
  const double magnitude = std::hypot(*dr, *dz);
  if (magnitude > total_cap) {
    const double scale = total_cap / magnitude;
    *dr *= scale;
    *dz *= scale;
    outcome.total = true;
  }

  const double normal_norm = std::hypot(n_r, n_z);
  if (has_normal && std::isfinite(cap_normal_frac) &&
      cap_normal_frac > 0.0 && std::isfinite(normal_norm) &&
      normal_norm > 0.0) {
    const double unit_r = n_r / normal_norm;
    const double unit_z = n_z / normal_norm;
    const double component = *dr * unit_r + *dz * unit_z;
    const double normal_cap = cap_normal_frac * h_p;
    if (std::abs(component) > normal_cap) {
      const double limited = std::copysign(normal_cap, component);
      *dr += (limited - component) * unit_r;
      *dz += (limited - component) * unit_z;
      outcome.normal = true;
    }
  }
  return outcome;
}

int core_boundary_node_id(const ButtonIndexing& idx, const int k) {
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta,
                "runtime ALE core boundary index out of range");
  if (k <= idx.n_c) {
    return button_morph::core_node_id(idx, k, 2 * idx.n_c);
  }
  if (k <= 3 * idx.n_c) {
    return button_morph::core_node_id(idx, idx.n_c, 3 * idx.n_c - k);
  }
  return button_morph::core_node_id(idx, 4 * idx.n_c - k, 0);
}

int bridge_node_id(const ButtonIndexing& idx, const int layer, const int k) {
  TENRYU_ASSERT(layer >= 0 && layer <= idx.n_b,
                "runtime ALE bridge node layer out of range");
  if (layer == 0) {
    return core_boundary_node_id(idx, k);
  }
  if (layer == idx.n_b) {
    return button_morph::shell_node_id(idx, 0, k);
  }
  return button_morph::bridge_interior_node_id(idx, layer, k);
}

int bridge_cell_id(const ButtonIndexing& idx, const int layer, const int k) {
  TENRYU_ASSERT(layer >= 0 && layer < idx.n_b,
                "runtime ALE bridge cell layer out of range");
  TENRYU_ASSERT(k >= 0 && k < idx.ntheta,
                "runtime ALE bridge cell column out of range");
  return idx.bridge_cell_offset + layer * idx.ntheta + k;
}

int cap_chain_node_id(const ButtonIndexing& idx,
                      const int row,
                      const int k) {
  TENRYU_ASSERT(row >= 0, "runtime ALE cap row out of range");
  if (row <= idx.n_b) {
    return bridge_node_id(idx, row, k);
  }
  return button_morph::shell_node_id(idx, row - idx.n_b, k);
}

int cap_chain_cell_id(const ButtonIndexing& idx,
                      const int row,
                      const int k) {
  TENRYU_ASSERT(row >= 0, "runtime ALE cap cell row out of range");
  if (row < idx.n_b) {
    return bridge_cell_id(idx, row, k);
  }
  return idx.shell_cell_offset + (row - idx.n_b) * idx.ntheta + k;
}

int cap_ray_index(const ButtonIndexing& idx,
                  const int axis,
                  const int ray) {
  TENRYU_ASSERT(axis == 0 || axis == 1,
                "runtime ALE cap axis out of range");
  TENRYU_ASSERT(ray >= 0 && ray <= idx.ntheta,
                "runtime ALE cap ray out of range");
  return axis == 0 ? ray : idx.ntheta - ray;
}

void cap_ray_direction(const ButtonIndexing& idx,
                       const int k,
                       double* const ray_r,
                       double* const ray_z) {
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta,
                "runtime ALE cap direction index out of range");
  if (k == 0) {
    *ray_r = 0.0;
    *ray_z = 1.0;
    return;
  }
  if (k == idx.ntheta) {
    *ray_r = 0.0;
    *ray_z = -1.0;
    return;
  }
  const double theta = std::numbers::pi_v<double> * static_cast<double>(k) /
                       static_cast<double>(idx.ntheta);
  *ray_r = std::sin(theta);
  *ray_z = std::cos(theta);
}

double ray_front_radius(const double* const front_s_per_column,
                        const int ntheta,
                        const int k) {
  if (front_s_per_column == nullptr) {
    return -1.0;
  }
  const auto valid = [](const double value) {
    return std::isfinite(value) && value > 0.0;
  };
  if (k == 0) {
    return valid(front_s_per_column[0]) ? front_s_per_column[0] : -1.0;
  }
  if (k == ntheta) {
    return valid(front_s_per_column[ntheta - 1])
               ? front_s_per_column[ntheta - 1]
               : -1.0;
  }
  const double left = front_s_per_column[k - 1];
  const double right = front_s_per_column[k];
  const bool left_valid = valid(left);
  const bool right_valid = valid(right);
  if (left_valid && right_valid) {
    return 0.5 * (left + right);
  }
  if (left_valid) {
    return left;
  }
  if (right_valid) {
    return right;
  }
  return -1.0;
}

double smoothstep_activation(const double quality,
                             const double q_hard,
                             const double q_recover) {
  const double u = std::clamp((quality - q_hard) / (q_recover - q_hard),
                              0.0, 1.0);
  return 1.0 - u * u * (3.0 - 2.0 * u);
}

void append_unique_neighbor(std::vector<int>& neighbors, const int node) {
  if (std::find(neighbors.begin(), neighbors.end(), node) == neighbors.end()) {
    neighbors.push_back(node);
  }
}

void restrict_lambda_for_ratio(const double base_a,
                               const double base_b,
                               const double target_a,
                               const double target_b,
                               const double g_max,
                               double* const lambda) {
  const double base_value = base_a - g_max * base_b;
  const double slope = (target_a - base_a) -
                       g_max * (target_b - base_b);
  if (base_value + *lambda * slope <= 0.0 || !(slope > 0.0)) {
    return;
  }
  *lambda = std::clamp(-base_value / slope, 0.0, *lambda);
}

std::vector<double> ratio_feasible_baseline(const int n_rows,
                                            const double free_span,
                                            const double h_seam,
                                            const double g_max) {
  std::vector<double> lower(static_cast<std::size_t>(n_rows), h_seam);
  std::vector<double> upper(static_cast<std::size_t>(n_rows), h_seam);
  for (int i = n_rows - 2; i >= 0; --i) {
    lower[static_cast<std::size_t>(i)] =
        lower[static_cast<std::size_t>(i + 1)] / g_max;
    upper[static_cast<std::size_t>(i)] =
        upper[static_cast<std::size_t>(i + 1)] * g_max;
  }

  const int free_rows = n_rows - 1;
  const auto sum_for_level = [&](const double level) {
    double sum = 0.0;
    for (int i = 0; i < free_rows; ++i) {
      sum += std::clamp(level, lower[static_cast<std::size_t>(i)],
                        upper[static_cast<std::size_t>(i)]);
    }
    return sum;
  };

  double level_lo = 0.0;
  double level_hi = std::max(free_span, h_seam);
  for (int iter = 0; iter < 96; ++iter) {
    const double level_mid = 0.5 * (level_lo + level_hi);
    if (sum_for_level(level_mid) < free_span) {
      level_lo = level_mid;
    } else {
      level_hi = level_mid;
    }
  }

  std::vector<double> spacing(static_cast<std::size_t>(n_rows), h_seam);
  const double level = 0.5 * (level_lo + level_hi);
  for (int i = 0; i < free_rows; ++i) {
    spacing[static_cast<std::size_t>(i)] =
        std::clamp(level, lower[static_cast<std::size_t>(i)],
                   upper[static_cast<std::size_t>(i)]);
  }
  return spacing;
}

struct AffineExpression {
  std::vector<double> coefficients;
  double constant = 0.0;
};

struct AffineConstraint {
  std::vector<double> coefficients;
  double constant = 0.0;
  double lower_bound = 0.0;
};

struct AffineVector2 {
  double coefficient = 0.0;
  double constant_r = 0.0;
  double constant_z = 0.0;
};

struct AffineConstraint2 {
  double coefficient_r = 0.0;
  double coefficient_z = 0.0;
  double constant = 0.0;
  double lower_bound = 0.0;
  double tolerance = 0.0;
  int cell = -1;
};

AffineVector2 affine_point_difference(
    const std::vector<int>& cell_nodes,
    const int head,
    const int tail,
    const int moving_node,
    const std::vector<double>& current_r,
    const std::vector<double>& current_z) {
  AffineVector2 expression;
  const int head_node = cell_nodes[static_cast<std::size_t>(head)];
  const int tail_node = cell_nodes[static_cast<std::size_t>(tail)];
  if (head_node == moving_node) {
    expression.coefficient += 1.0;
  } else {
    expression.constant_r += current_r[static_cast<std::size_t>(head_node)];
    expression.constant_z += current_z[static_cast<std::size_t>(head_node)];
  }
  if (tail_node == moving_node) {
    expression.coefficient -= 1.0;
  } else {
    expression.constant_r -= current_r[static_cast<std::size_t>(tail_node)];
    expression.constant_z -= current_z[static_cast<std::size_t>(tail_node)];
  }
  return expression;
}

AffineConstraint2 affine_cross_constraint(
    const AffineVector2& left,
    const AffineVector2& right,
    const double scale,
    const double lower_bound) {
  // cross(alpha*x + u, beta*x + v) is affine because cross(x, x) is zero.
  AffineConstraint2 constraint;
  constraint.coefficient_r =
      scale * (left.coefficient * right.constant_z -
               right.coefficient * left.constant_z);
  constraint.coefficient_z =
      scale * (-left.coefficient * right.constant_r +
               right.coefficient * left.constant_r);
  constraint.constant =
      scale * (left.constant_r * right.constant_z -
               left.constant_z * right.constant_r);
  constraint.lower_bound = lower_bound;
  return constraint;
}

void append_ordinary_cell_constraints(
    const std::vector<int>& cell_nodes,
    const int cell,
    const int moving_node,
    const std::vector<double>& reference_r,
    const std::vector<double>& reference_z,
    const std::vector<double>& fixed_r,
    const std::vector<double>& fixed_z,
    std::vector<AffineConstraint2>* const constraints) {
  const int nverts = static_cast<int>(cell_nodes.size());
  double old_area2 = 0.0;
  double length_scale = 0.0;
  for (int corner = 0; corner < nverts; ++corner) {
    const int next = (corner + 1 == nverts) ? 0 : corner + 1;
    const int node = cell_nodes[static_cast<std::size_t>(corner)];
    const int node_next = cell_nodes[static_cast<std::size_t>(next)];
    old_area2 += reference_r[static_cast<std::size_t>(node)] *
                     reference_z[static_cast<std::size_t>(node_next)] -
                 reference_r[static_cast<std::size_t>(node_next)] *
                     reference_z[static_cast<std::size_t>(node)];
    for (int other = corner + 1; other < nverts; ++other) {
      const int other_node = cell_nodes[static_cast<std::size_t>(other)];
      length_scale = std::max(
          length_scale,
          std::hypot(reference_r[static_cast<std::size_t>(other_node)] -
                         reference_r[static_cast<std::size_t>(node)],
                     reference_z[static_cast<std::size_t>(other_node)] -
                         reference_z[static_cast<std::size_t>(node)]));
    }
  }
  const double orientation = old_area2 < 0.0 ? -1.0 : 1.0;
  const double margin = 64.0 * std::numeric_limits<double>::epsilon() *
                        length_scale * length_scale;

  for (int corner = 0; corner < nverts; ++corner) {
    const int next = (corner + 1 == nverts) ? 0 : corner + 1;
    const int prev = (corner == 0) ? nverts - 1 : corner - 1;
    const AffineVector2 next_edge = affine_point_difference(
        cell_nodes, next, corner, moving_node, fixed_r, fixed_z);
    const AffineVector2 prev_edge = affine_point_difference(
        cell_nodes, prev, corner, moving_node, fixed_r, fixed_z);
    AffineConstraint2 constraint = affine_cross_constraint(
        next_edge, prev_edge, orientation, margin);
    constraint.tolerance = margin;
    constraint.cell = cell;
    constraints->push_back(constraint);
  }

  double centroid_constant_r = 0.0;
  double centroid_constant_z = 0.0;
  for (const int node : cell_nodes) {
    if (node != moving_node) {
      centroid_constant_r += fixed_r[static_cast<std::size_t>(node)];
      centroid_constant_z += fixed_z[static_cast<std::size_t>(node)];
    }
  }
  centroid_constant_r /= static_cast<double>(nverts);
  centroid_constant_z /= static_cast<double>(nverts);
  for (int corner = 0; corner < nverts; ++corner) {
    const int next = (corner + 1 == nverts) ? 0 : corner + 1;
    const int corner_node = cell_nodes[static_cast<std::size_t>(corner)];
    const AffineVector2 edge = affine_point_difference(
        cell_nodes, next, corner, moving_node, fixed_r, fixed_z);
    AffineVector2 center_edge;
    center_edge.coefficient = 1.0 / static_cast<double>(nverts);
    center_edge.constant_r = centroid_constant_r;
    center_edge.constant_z = centroid_constant_z;
    if (corner_node == moving_node) {
      center_edge.coefficient -= 1.0;
    } else {
      center_edge.constant_r -=
          fixed_r[static_cast<std::size_t>(corner_node)];
      center_edge.constant_z -=
          fixed_z[static_cast<std::size_t>(corner_node)];
    }
    AffineConstraint2 constraint = affine_cross_constraint(
        edge, center_edge, 0.5 * orientation, margin);
    constraint.tolerance = margin;
    constraint.cell = cell;
    constraints->push_back(constraint);
  }
}

struct OrdinaryProjectionResult {
  bool projected = false;
  int failing_cell = -1;
};

OrdinaryProjectionResult project_ordinary_node(
    const int node,
    const std::vector<std::vector<int>>& node_cells,
    const std::vector<int>& csr_offsets,
    const std::vector<int>& csr_indices,
    const std::vector<double>& current_r,
    const std::vector<double>& current_z,
    double* const target_r,
    double* const target_z) {
  std::vector<AffineConstraint2> constraints;
  for (const int cell : node_cells[static_cast<std::size_t>(node)]) {
    const int begin = csr_offsets[static_cast<std::size_t>(cell)];
    const int end = csr_offsets[static_cast<std::size_t>(cell + 1)];
    const std::vector<int> cell_nodes(
        csr_indices.begin() + begin, csr_indices.begin() + end);
    append_ordinary_cell_constraints(
        cell_nodes, cell, node, current_r, current_z, current_r, current_z,
        &constraints);
  }

  OrdinaryProjectionResult result;
  constexpr int projection_passes = 64;
  for (int pass = 0; pass < projection_passes; ++pass) {
    for (const AffineConstraint2& constraint : constraints) {
      const double value = constraint.coefficient_r * *target_r +
                           constraint.coefficient_z * *target_z +
                           constraint.constant;
      const double norm2 =
          constraint.coefficient_r * constraint.coefficient_r +
          constraint.coefficient_z * constraint.coefficient_z;
      if (value >= constraint.lower_bound ||
          !(std::isfinite(value) && std::isfinite(norm2) && norm2 > 0.0)) {
        continue;
      }
      const double scale = (constraint.lower_bound - value) / norm2;
      *target_r += scale * constraint.coefficient_r;
      *target_z += scale * constraint.coefficient_z;
      result.projected = true;
    }
  }
  for (const AffineConstraint2& constraint : constraints) {
    const double value = constraint.coefficient_r * *target_r +
                         constraint.coefficient_z * *target_z +
                         constraint.constant;
    if (!std::isfinite(value) ||
        value + constraint.tolerance < constraint.lower_bound) {
      result.failing_cell = constraint.cell;
      break;
    }
  }
  return result;
}

bool cap_chain_location(const ButtonIndexing& idx,
                        const int node,
                        int* const row,
                        int* const k) {
  if (node >= idx.bridge_interior_node_offset &&
      node < idx.shell_node_offset) {
    const int local = node - idx.bridge_interior_node_offset;
    *row = local / (idx.ntheta + 1) + 1;
    *k = local % (idx.ntheta + 1);
    return true;
  }
  if (node >= idx.shell_node_offset && node < idx.n_nodes_total) {
    const int local = node - idx.shell_node_offset;
    *row = idx.n_b + local / (idx.ntheta + 1);
    *k = local % (idx.ntheta + 1);
    return true;
  }
  for (int boundary_k = 0; boundary_k <= idx.ntheta; ++boundary_k) {
    if (core_boundary_node_id(idx, boundary_k) == node) {
      *row = 0;
      *k = boundary_k;
      return true;
    }
  }
  return false;
}

void append_patch_node(const int node,
                       std::vector<std::uint8_t>* const membership,
                       std::vector<int>* const nodes) {
  if (node < 0 || static_cast<std::size_t>(node) >= membership->size() ||
      (*membership)[static_cast<std::size_t>(node)] != 0U) {
    return;
  }
  (*membership)[static_cast<std::size_t>(node)] = 1U;
  nodes->push_back(node);
}

std::vector<int> collect_coupled_patch_nodes(
    const ButtonIndexing& idx,
    const int failing_node,
    const int failing_cell,
    const std::vector<std::vector<int>>& node_cells,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<int>& csr_offsets,
    const std::vector<int>& csr_indices) {
  std::vector<std::uint8_t> membership(
      static_cast<std::size_t>(idx.n_nodes_total), 0U);
  std::vector<int> nodes;
  append_patch_node(failing_node, &membership, &nodes);
  if (failing_cell >= 0 &&
      static_cast<std::size_t>(failing_cell + 1) < csr_offsets.size()) {
    const int begin = csr_offsets[static_cast<std::size_t>(failing_cell)];
    const int end = csr_offsets[static_cast<std::size_t>(failing_cell + 1)];
    for (int offset = begin; offset < end; ++offset) {
      append_patch_node(csr_indices[static_cast<std::size_t>(offset)],
                        &membership, &nodes);
    }
  }

  const std::vector<int> seed_nodes = nodes;
  const int last_cap_row = idx.n_b + idx.nr_shell;
  for (const int node : seed_nodes) {
    int row = 0;
    int k = 0;
    if (cap_chain_location(idx, node, &row, &k)) {
      for (int radial_row = std::max(0, row - 2);
           radial_row <= std::min(last_cap_row, row + 2); ++radial_row) {
        append_patch_node(cap_chain_node_id(idx, radial_row, k), &membership,
                          &nodes);
      }
      continue;
    }

    std::vector<int> frontier{node};
    for (int layer = 0; layer < 2; ++layer) {
      std::vector<int> next;
      for (const int frontier_node : frontier) {
        for (const int neighbor :
             neighbors[static_cast<std::size_t>(frontier_node)]) {
          if (membership[static_cast<std::size_t>(neighbor)] == 0U) {
            append_patch_node(neighbor, &membership, &nodes);
            next.push_back(neighbor);
          }
        }
      }
      frontier.swap(next);
    }
  }

  std::vector<int> boundary_nodes;
  for (const int node : nodes) {
    int row = 0;
    int k = 0;
    if (cap_chain_location(idx, node, &row, &k) && row == 0) {
      boundary_nodes.push_back(node);
    }
  }
  std::vector<int> core_frontier;
  for (const int boundary_node : boundary_nodes) {
    for (const int cell : node_cells[static_cast<std::size_t>(boundary_node)]) {
      const int begin = csr_offsets[static_cast<std::size_t>(cell)];
      const int end = csr_offsets[static_cast<std::size_t>(cell + 1)];
      for (int offset = begin; offset < end; ++offset) {
        const int node = csr_indices[static_cast<std::size_t>(offset)];
        const bool new_node =
            membership[static_cast<std::size_t>(node)] == 0U;
        if (new_node) {
          append_patch_node(node, &membership, &nodes);
        }
        if (new_node && node < idx.n_nodes_core) {
          core_frontier.push_back(node);
        }
      }
    }
  }
  for (const int core_node : core_frontier) {
    for (const int neighbor : neighbors[static_cast<std::size_t>(core_node)]) {
      if (neighbor < idx.n_nodes_core) {
        append_patch_node(neighbor, &membership, &nodes);
      }
    }
  }

  std::sort(nodes.begin(), nodes.end());
  return nodes;
}

struct CoupledPatchOutcome {
  int node_count = 0;
  bool oversize = false;
};

CoupledPatchOutcome project_coupled_patch(
    const ButtonIndexing& idx,
    const int failing_node,
    const int failing_cell,
    const std::vector<std::uint8_t>& mandate,
    const std::vector<std::uint8_t>& cap_node,
    const std::vector<double>& cap_ray_r,
    const std::vector<double>& cap_ray_z,
    const std::vector<std::vector<int>>& node_cells,
    const std::vector<std::vector<int>>& neighbors,
    const std::vector<int>& csr_offsets,
    const std::vector<int>& csr_indices,
    const std::vector<double>& current_r,
    const std::vector<double>& current_z,
    std::vector<double>* const target_r,
    std::vector<double>* const target_z) {
  CoupledPatchOutcome outcome;
  const std::vector<int> patch_nodes = collect_coupled_patch_nodes(
      idx, failing_node, failing_cell, node_cells, neighbors, csr_offsets,
      csr_indices);
  outcome.node_count = static_cast<int>(patch_nodes.size());
  constexpr int patch_node_limit = 64;
  if (outcome.node_count > patch_node_limit) {
    outcome.oversize = true;
    return outcome;
  }
  for (const int node : patch_nodes) {
    const std::size_t n = static_cast<std::size_t>(node);
    if (mandate[n] == 0U) {
      continue;
    }
    if (cap_node[n] != 0U) {
      const double scalar = (*target_r)[n] * cap_ray_r[n] +
                            (*target_z)[n] * cap_ray_z[n];
      (*target_r)[n] = scalar * cap_ray_r[n];
      (*target_z)[n] = scalar * cap_ray_z[n];
    } else if (current_r[n] == 0.0) {
      (*target_r)[n] = 0.0;
    }
  }

  constexpr int projection_passes = 128;
  for (int pass = 0; pass < projection_passes; ++pass) {
    for (const int node : patch_nodes) {
      const std::size_t n = static_cast<std::size_t>(node);
      if (mandate[n] == 0U) {
        continue;
      }
      std::vector<AffineConstraint2> constraints;
      for (const int cell : node_cells[n]) {
        const int begin = csr_offsets[static_cast<std::size_t>(cell)];
        const int end = csr_offsets[static_cast<std::size_t>(cell + 1)];
        const std::vector<int> cell_nodes(
            csr_indices.begin() + begin, csr_indices.begin() + end);
        append_ordinary_cell_constraints(
            cell_nodes, cell, node, current_r, current_z, *target_r, *target_z,
            &constraints);
      }

      const bool constrained_to_line =
          cap_node[n] != 0U || current_r[n] == 0.0;
      const double direction_r =
          cap_node[n] != 0U ? cap_ray_r[n] : 0.0;
      const double direction_z =
          cap_node[n] != 0U ? cap_ray_z[n] : 1.0;
      for (const AffineConstraint2& constraint : constraints) {
        const double value =
            constraint.coefficient_r * (*target_r)[n] +
            constraint.coefficient_z * (*target_z)[n] + constraint.constant;
        if (value >= constraint.lower_bound || !std::isfinite(value)) {
          continue;
        }
        if (constrained_to_line) {
          const double coefficient =
              constraint.coefficient_r * direction_r +
              constraint.coefficient_z * direction_z;
          const double norm2 = coefficient * coefficient;
          if (!(std::isfinite(norm2) && norm2 > 0.0)) {
            continue;
          }
          const double scale = (constraint.lower_bound - value) / norm2;
          (*target_r)[n] += scale * coefficient * direction_r;
          (*target_z)[n] += scale * coefficient * direction_z;
        } else {
          const double norm2 =
              constraint.coefficient_r * constraint.coefficient_r +
              constraint.coefficient_z * constraint.coefficient_z;
          if (!(std::isfinite(norm2) && norm2 > 0.0)) {
            continue;
          }
          const double scale = (constraint.lower_bound - value) / norm2;
          (*target_r)[n] += scale * constraint.coefficient_r;
          (*target_z)[n] += scale * constraint.coefficient_z;
        }
      }
      if (current_r[n] == 0.0) {
        (*target_r)[n] = 0.0;
      }
    }
  }
  return outcome;
}

AffineExpression cap_gap_expression(const int interval,
                                    const int n_rows,
                                    const double inner,
                                    const double outer) {
  const int n_variables = n_rows - 1;
  AffineExpression expression;
  expression.coefficients.assign(static_cast<std::size_t>(n_variables), 0.0);
  if (n_rows == 1) {
    expression.constant = outer - inner;
  } else if (interval == 0) {
    expression.constant = -inner;
    expression.coefficients[0] = 1.0;
  } else if (interval == n_rows - 1) {
    expression.constant = outer;
    expression.coefficients[static_cast<std::size_t>(n_variables - 1)] = -1.0;
  } else {
    expression.coefficients[static_cast<std::size_t>(interval)] = 1.0;
    expression.coefficients[static_cast<std::size_t>(interval - 1)] = -1.0;
  }
  return expression;
}

double evaluate_affine(const AffineConstraint& constraint,
                       const std::vector<double>& variables) {
  double value = constraint.constant;
  for (std::size_t i = 0; i < variables.size(); ++i) {
    value += constraint.coefficients[i] * variables[i];
  }
  return value;
}

bool project_common_cap_ladder(
    const std::vector<double>& desired,
    const std::vector<double>& inner_endpoints,
    const std::vector<double>& outer_endpoints,
    const std::vector<double>& minimum_gaps,
    const double g_max,
    std::vector<double>* const common_ladder) {
  if (common_ladder == nullptr || desired.size() < 2U ||
      inner_endpoints.empty() ||
      inner_endpoints.size() != outer_endpoints.size()) {
    return false;
  }
  const int n_rows = static_cast<int>(desired.size()) - 1;
  const int n_variables = n_rows - 1;
  const int n_rays = static_cast<int>(inner_endpoints.size());
  if (minimum_gaps.size() !=
      static_cast<std::size_t>(n_rays * n_rows)) {
    return false;
  }

  std::vector<AffineConstraint> constraints;
  constraints.reserve(static_cast<std::size_t>(n_rays * (3 * n_rows - 2)));
  for (int ray = 0; ray < n_rays; ++ray) {
    const double inner = inner_endpoints[static_cast<std::size_t>(ray)];
    const double outer = outer_endpoints[static_cast<std::size_t>(ray)];
    double required_span = 0.0;
    for (int row = 0; row < n_rows; ++row) {
      required_span += minimum_gaps[static_cast<std::size_t>(
          ray * n_rows + row)];
    }
    const double span = outer - inner;
    const double span_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max({std::abs(inner), std::abs(outer), std::abs(span)});
    if (!(std::isfinite(inner) && std::isfinite(outer) && span > 0.0) ||
        required_span > span + span_tolerance) {
      return false;
    }

    std::vector<AffineExpression> gaps;
    gaps.reserve(static_cast<std::size_t>(n_rows));
    for (int row = 0; row < n_rows; ++row) {
      gaps.push_back(cap_gap_expression(row, n_rows, inner, outer));
      constraints.push_back(
          {gaps.back().coefficients, gaps.back().constant,
           minimum_gaps[static_cast<std::size_t>(ray * n_rows + row)]});
    }
    for (int row = 1; row < n_rows; ++row) {
      AffineConstraint forward;
      AffineConstraint backward;
      forward.coefficients.assign(static_cast<std::size_t>(n_variables), 0.0);
      backward.coefficients.assign(static_cast<std::size_t>(n_variables), 0.0);
      for (int variable = 0; variable < n_variables; ++variable) {
        forward.coefficients[static_cast<std::size_t>(variable)] =
            g_max * gaps[static_cast<std::size_t>(row - 1)]
                        .coefficients[static_cast<std::size_t>(variable)] -
            gaps[static_cast<std::size_t>(row)]
                .coefficients[static_cast<std::size_t>(variable)];
        backward.coefficients[static_cast<std::size_t>(variable)] =
            g_max * gaps[static_cast<std::size_t>(row)]
                        .coefficients[static_cast<std::size_t>(variable)] -
            gaps[static_cast<std::size_t>(row - 1)]
                .coefficients[static_cast<std::size_t>(variable)];
      }
      forward.constant =
          g_max * gaps[static_cast<std::size_t>(row - 1)].constant -
          gaps[static_cast<std::size_t>(row)].constant;
      backward.constant =
          g_max * gaps[static_cast<std::size_t>(row)].constant -
          gaps[static_cast<std::size_t>(row - 1)].constant;
      constraints.push_back(std::move(forward));
      constraints.push_back(std::move(backward));
    }
  }

  std::vector<double> variables;
  variables.reserve(static_cast<std::size_t>(n_variables));
  for (int row = 1; row < n_rows; ++row) {
    variables.push_back(desired[static_cast<std::size_t>(row)]);
  }
  constexpr int projection_passes = 256;
  const auto project_constraint = [&](const AffineConstraint& constraint) {
    const double value = evaluate_affine(constraint, variables);
    if (value >= constraint.lower_bound) {
      return;
    }
    double norm2 = 0.0;
    for (const double coefficient : constraint.coefficients) {
      norm2 += coefficient * coefficient;
    }
    if (!(std::isfinite(value) && std::isfinite(norm2) && norm2 > 0.0)) {
      return;
    }
    const double scale = (constraint.lower_bound - value) / norm2;
    for (std::size_t i = 0; i < variables.size(); ++i) {
      variables[i] += scale * constraint.coefficients[i];
    }
  };
  for (int pass = 0; pass < projection_passes; ++pass) {
    for (const AffineConstraint& constraint : constraints) {
      project_constraint(constraint);
    }
    for (auto it = constraints.rbegin(); it != constraints.rend(); ++it) {
      project_constraint(*it);
    }
  }

  const double scale = std::max(
      std::abs(outer_endpoints.front() - inner_endpoints.front()),
      std::numeric_limits<double>::min());
  const double tolerance =
      256.0 * std::numeric_limits<double>::epsilon() * scale;
  for (const AffineConstraint& constraint : constraints) {
    const double value = evaluate_affine(constraint, variables);
    if (!std::isfinite(value) || value + tolerance < constraint.lower_bound) {
      return false;
    }
  }

  common_ladder->assign(desired.size(), 0.0);
  (*common_ladder)[0] = desired[0];
  for (int row = 1; row < n_rows; ++row) {
    (*common_ladder)[static_cast<std::size_t>(row)] =
        variables[static_cast<std::size_t>(row - 1)];
  }
  (*common_ladder)[static_cast<std::size_t>(n_rows)] = desired.back();
  return true;
}

double oriented_corner_j(const std::vector<double>& r,
                         const std::vector<double>& z,
                         const int corner,
                         const double orientation) {
  const int nverts = static_cast<int>(r.size());
  const int next = (corner + 1 == nverts) ? 0 : corner + 1;
  const int prev = (corner == 0) ? nverts - 1 : corner - 1;
  return orientation *
         ((r[static_cast<std::size_t>(next)] -
           r[static_cast<std::size_t>(corner)]) *
              (z[static_cast<std::size_t>(prev)] -
               z[static_cast<std::size_t>(corner)]) -
          (z[static_cast<std::size_t>(next)] -
           z[static_cast<std::size_t>(corner)]) *
              (r[static_cast<std::size_t>(prev)] -
               r[static_cast<std::size_t>(corner)]));
}

std::vector<double> cap_cell_constraint_values(
    const std::vector<int>& cell_nodes,
    const int moving_node,
    const double scalar,
    const double ray_r,
    const double ray_z,
    const std::vector<double>& current_r,
    const std::vector<double>& current_z,
    const std::vector<double>& target_r,
    const std::vector<double>& target_z) {
  const int nverts = static_cast<int>(cell_nodes.size());
  std::vector<double> r(static_cast<std::size_t>(nverts), 0.0);
  std::vector<double> z(static_cast<std::size_t>(nverts), 0.0);
  double old_area2 = 0.0;
  for (int corner = 0; corner < nverts; ++corner) {
    const int next = (corner + 1 == nverts) ? 0 : corner + 1;
    const int node = cell_nodes[static_cast<std::size_t>(corner)];
    const int node_next = cell_nodes[static_cast<std::size_t>(next)];
    r[static_cast<std::size_t>(corner)] =
        node == moving_node ? scalar * ray_r
                            : target_r[static_cast<std::size_t>(node)];
    z[static_cast<std::size_t>(corner)] =
        node == moving_node ? scalar * ray_z
                            : target_z[static_cast<std::size_t>(node)];
    old_area2 += current_r[static_cast<std::size_t>(node)] *
                     current_z[static_cast<std::size_t>(node_next)] -
                 current_r[static_cast<std::size_t>(node_next)] *
                     current_z[static_cast<std::size_t>(node)];
  }
  const double orientation = old_area2 < 0.0 ? -1.0 : 1.0;
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  for (int corner = 0; corner < nverts; ++corner) {
    centroid_r += r[static_cast<std::size_t>(corner)];
    centroid_z += z[static_cast<std::size_t>(corner)];
  }
  centroid_r /= static_cast<double>(nverts);
  centroid_z /= static_cast<double>(nverts);

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(2 * nverts));
  for (int corner = 0; corner < nverts; ++corner) {
    values.push_back(oriented_corner_j(r, z, corner, orientation));
  }
  for (int corner = 0; corner < nverts; ++corner) {
    const int next = (corner + 1 == nverts) ? 0 : corner + 1;
    values.push_back(
        0.5 * orientation *
        ((r[static_cast<std::size_t>(next)] -
          r[static_cast<std::size_t>(corner)]) *
             (centroid_z - z[static_cast<std::size_t>(corner)]) -
         (z[static_cast<std::size_t>(next)] -
          z[static_cast<std::size_t>(corner)]) *
             (centroid_r - r[static_cast<std::size_t>(corner)])));
  }
  return values;
}

bool cap_node_feasible_interval(
    const int node,
    const double ray_r,
    const double ray_z,
    const std::vector<std::vector<int>>& node_cells,
    const std::vector<int>& csr_offsets,
    const std::vector<int>& csr_indices,
    const std::vector<double>& current_r,
    const std::vector<double>& current_z,
    const std::vector<double>& target_r,
    const std::vector<double>& target_z,
    double* const lower,
    double* const upper) {
  *lower = 0.0;
  *upper = std::numeric_limits<double>::infinity();
  for (const int cell : node_cells[static_cast<std::size_t>(node)]) {
    const int begin = csr_offsets[static_cast<std::size_t>(cell)];
    const int end = csr_offsets[static_cast<std::size_t>(cell + 1)];
    std::vector<int> cell_nodes(
        csr_indices.begin() + begin, csr_indices.begin() + end);
    const std::vector<double> at_zero = cap_cell_constraint_values(
        cell_nodes, node, 0.0, ray_r, ray_z, current_r, current_z, target_r,
        target_z);
    const std::vector<double> at_one = cap_cell_constraint_values(
        cell_nodes, node, 1.0, ray_r, ray_z, current_r, current_z, target_r,
        target_z);
    for (std::size_t constraint = 0; constraint < at_zero.size();
         ++constraint) {
      const double intercept = at_zero[constraint];
      const double slope = at_one[constraint] - intercept;
      if (!(std::isfinite(intercept) && std::isfinite(slope))) {
        return false;
      }
      const double slope_tolerance =
          64.0 * std::numeric_limits<double>::epsilon() *
          std::max({std::abs(intercept), std::abs(at_one[constraint]),
                    std::numeric_limits<double>::min()});
      if (std::abs(slope) <= slope_tolerance) {
        if (intercept < 0.0) {
          return false;
        }
        continue;
      }
      const double bound = -intercept / slope;
      if (!std::isfinite(bound)) {
        return false;
      }
      if (slope > 0.0) {
        *lower = std::max(*lower, bound);
      } else {
        *upper = std::min(*upper, bound);
      }
    }
  }
  return std::isfinite(*lower) && *lower <= *upper;
}

}  // namespace

namespace detail {

void equidistribute_column(const double* const s_in,
                           const int n_rows,
                           const double* const rho_bins,
                           const int n_bins,
                           const double s_front,
                           const double front_width,
                           const double beta_mass,
                           const double beta_front,
                           const double g_max,
                           const double h_seam,
                           double* const s_out) {
  if (s_in == nullptr || s_out == nullptr || n_rows < 1) {
    return;
  }
  std::copy(s_in, s_in + n_rows + 1, s_out);
  if (n_rows == 1) {
    return;
  }

  bool valid_ladder = std::isfinite(s_in[0]);
  for (int i = 0; i < n_rows; ++i) {
    valid_ladder = valid_ladder && std::isfinite(s_in[i + 1]) &&
                   s_in[i + 1] > s_in[i];
  }
  const double s_lo = s_in[0];
  const double s_hi = s_in[n_rows];
  const double pinned_inner = s_hi - h_seam;
  valid_ladder = valid_ladder && std::isfinite(h_seam) && h_seam > 0.0 &&
                 pinned_inner > s_lo && std::isfinite(g_max) && g_max > 1.0;
  if (!valid_ladder) {
    return;
  }

  const int free_rows = n_rows - 1;
  std::vector<double> widths(static_cast<std::size_t>(n_rows), 0.0);
  std::vector<double> midpoints(static_cast<std::size_t>(n_rows), 0.0);
  double I0 = 0.0;
  double Im = 0.0;
  double If = 0.0;
  const bool use_mass = rho_bins != nullptr && n_bins > 0 && beta_mass > 0.0;
  const bool use_front = std::isfinite(s_front) && s_front > 0.0 &&
                         std::isfinite(front_width) && front_width > 0.0 &&
                         beta_front > 0.0;
  for (int i = 0; i < n_rows; ++i) {
    const double width = s_in[i + 1] - s_in[i];
    const double midpoint = 0.5 * (s_in[i] + s_in[i + 1]);
    widths[static_cast<std::size_t>(i)] = width;
    midpoints[static_cast<std::size_t>(i)] = midpoint;
    I0 += width / width;
    if (use_mass) {
      const int bin = std::min(i, n_bins - 1);
      const double rho = rho_bins[bin];
      if (std::isfinite(rho) && rho > 0.0) {
        Im += rho * midpoint * midpoint * width;
      }
    }
    if (use_front) {
      const double u = (midpoint - s_front) / front_width;
      If += std::exp(-u * u) * width;
    }
  }

  const double mass_weight = use_mass && Im > 0.0 ? beta_mass : 0.0;
  const double front_weight = use_front && If > 0.0 ? beta_front : 0.0;
  const double base_weight =
      std::max(0.0, 1.0 - mass_weight - front_weight);
  std::vector<double> monitor_integral(static_cast<std::size_t>(free_rows),
                                       0.0);
  double monitor_total = 0.0;
  for (int i = 0; i < free_rows; ++i) {
    const double width = widths[static_cast<std::size_t>(i)];
    const double midpoint = midpoints[static_cast<std::size_t>(i)];
    double monitor = base_weight * (1.0 / width) / I0;
    if (mass_weight > 0.0) {
      const int bin = std::min(i, n_bins - 1);
      const double rho = rho_bins[bin];
      if (std::isfinite(rho) && rho > 0.0) {
        monitor += mass_weight * rho * midpoint * midpoint / Im;
      }
    }
    if (front_weight > 0.0) {
      const double u = (midpoint - s_front) / front_width;
      monitor += front_weight * std::exp(-u * u) / If;
    }
    monitor_integral[static_cast<std::size_t>(i)] = monitor * width;
    monitor_total += monitor * width;
  }
  if (!(std::isfinite(monitor_total) && monitor_total > 0.0)) {
    return;
  }

  std::vector<double> desired_nodes(static_cast<std::size_t>(n_rows) + 1U,
                                    0.0);
  desired_nodes[0] = s_lo;
  desired_nodes[static_cast<std::size_t>(free_rows)] = pinned_inner;
  desired_nodes[static_cast<std::size_t>(n_rows)] = s_hi;
  int interval = 0;
  double cumulative_before = 0.0;
  for (int node = 1; node < free_rows; ++node) {
    const double target = monitor_total * static_cast<double>(node) /
                          static_cast<double>(free_rows);
    while (interval + 1 < free_rows &&
           cumulative_before +
                   monitor_integral[static_cast<std::size_t>(interval)] <
               target) {
      cumulative_before +=
          monitor_integral[static_cast<std::size_t>(interval)];
      ++interval;
    }
    const double interval_integral =
        monitor_integral[static_cast<std::size_t>(interval)];
    const double fraction = std::clamp(
        (target - cumulative_before) / interval_integral, 0.0, 1.0);
    desired_nodes[static_cast<std::size_t>(node)] =
        s_in[interval] + fraction * widths[static_cast<std::size_t>(interval)];
  }

  std::vector<double> desired_spacing(static_cast<std::size_t>(n_rows), 0.0);
  for (int i = 0; i < n_rows; ++i) {
    desired_spacing[static_cast<std::size_t>(i)] =
        desired_nodes[static_cast<std::size_t>(i + 1)] -
        desired_nodes[static_cast<std::size_t>(i)];
  }
  std::vector<double> baseline = ratio_feasible_baseline(
      n_rows, pinned_inner - s_lo, h_seam, g_max);

  double lambda = 1.0;
  for (int i = 1; i < n_rows; ++i) {
    restrict_lambda_for_ratio(
        baseline[static_cast<std::size_t>(i)],
        baseline[static_cast<std::size_t>(i - 1)],
        desired_spacing[static_cast<std::size_t>(i)],
        desired_spacing[static_cast<std::size_t>(i - 1)], g_max, &lambda);
  }
  for (int i = n_rows - 2; i >= 0; --i) {
    restrict_lambda_for_ratio(
        baseline[static_cast<std::size_t>(i)],
        baseline[static_cast<std::size_t>(i + 1)],
        desired_spacing[static_cast<std::size_t>(i)],
        desired_spacing[static_cast<std::size_t>(i + 1)], g_max, &lambda);
  }

  s_out[0] = s_lo;
  for (int i = 0; i < free_rows; ++i) {
    const double spacing = baseline[static_cast<std::size_t>(i)] +
                           lambda *
                               (desired_spacing[static_cast<std::size_t>(i)] -
                                baseline[static_cast<std::size_t>(i)]);
    s_out[i + 1] = s_out[i] + spacing;
  }
  s_out[free_rows] = pinned_inner;
  s_out[n_rows] = s_hi;
}

bool clamp_displacement(double* const dr,
                        double* const dz,
                        const double h_p,
                        const double cap_frac,
                        const double cap_normal_frac,
                        const bool has_normal,
                        const double n_r,
                        const double n_z) {
  const ClampOutcome outcome = clamp_displacement_impl(
      dr, dz, h_p, cap_frac, cap_normal_frac, has_normal, n_r, n_z);
  return outcome.total || outcome.normal;
}

}  // namespace detail

RuntimeAleTargetStats build_runtime_ale_target(
    core::State& state,
    const core::Config& cfg,
    const bool hard_mode,
    const RuntimeAleEscalation& escalation,
    const double* const front_s_per_column,
    const double* const shock_normal_dir_r,
    const double* const shock_normal_dir_z,
    core::DeviceArray<double>& target_x_r,
    core::DeviceArray<double>& target_x_z) {
  RuntimeAleTargetStats stats;
  const ButtonIndexing idx = button_morph::make_button_indexing(cfg.mesh);
  const auto& runtime_cfg = cfg.numerics.ale.runtime_controller;
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "runtime ALE targets require multiblock topology metadata");
  TENRYU_ASSERT(idx.n_nodes_total == state.mesh.topo.n_nodes &&
                    idx.n_cells_total == state.mesh.topo.n_cells,
                "runtime ALE target topology size mismatch");
  const int n_nodes = idx.n_nodes_total;
  const int n_cells = idx.n_cells_total;
  const std::size_t node_bytes =
      static_cast<std::size_t>(n_nodes) * sizeof(double);
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z.size() == static_cast<std::size_t>(n_nodes),
                "runtime ALE target coordinate size mismatch");
  TENRYU_ASSERT(state.rho.size() == static_cast<std::size_t>(n_cells),
                "runtime ALE target density size mismatch");

  std::vector<double> x_l_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> x_l_z(static_cast<std::size_t>(n_nodes), 0.0);
  state.x_r.copy_to_host(x_l_r.data());
  state.x_z.copy_to_host(x_l_z.data());

  std::vector<std::uint8_t> mandate(static_cast<std::size_t>(n_nodes), 0U);
  for (int node = 0; node < idx.shell_node_offset; ++node) {
    mandate[static_cast<std::size_t>(node)] = 1U;
  }
  for (int ring = 0;
       ring < cfg.numerics.ale.runtime_controller.controller_shell_rows;
       ++ring) {
    for (int k = 0; k <= idx.ntheta; ++k) {
      const int node = button_morph::shell_node_id(idx, ring, k);
      mandate[static_cast<std::size_t>(node)] = 1U;
    }
  }
  const int center_node = button_morph::core_node_id(idx, 0, idx.n_c);
  mandate[static_cast<std::size_t>(center_node)] = 0U;
  for (const std::uint8_t active : mandate) {
    stats.mandate_nodes += active != 0U ? 1 : 0;
  }

  const int cap_columns =
      std::min(runtime_cfg.cap_columns, idx.ntheta / 2);
  const int cap_chain_rows =
      idx.n_b + std::max(runtime_cfg.controller_shell_rows - 1, 0);
  std::vector<std::uint8_t> cap_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<double> cap_ray_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> cap_ray_z(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<std::uint8_t> winslow_mandate = mandate;
  if (cap_columns > 0) {
    for (int axis = 0; axis < 2; ++axis) {
      for (int ray = 0; ray <= cap_columns; ++ray) {
        const int k = cap_ray_index(idx, axis, ray);
        double ray_r = 0.0;
        double ray_z = 0.0;
        cap_ray_direction(idx, k, &ray_r, &ray_z);
        for (int row = 0; row <= cap_chain_rows; ++row) {
          const int node = cap_chain_node_id(idx, row, k);
          cap_node[static_cast<std::size_t>(node)] = 1U;
          cap_ray_r[static_cast<std::size_t>(node)] = ray_r;
          cap_ray_z[static_cast<std::size_t>(node)] = ray_z;
          winslow_mandate[static_cast<std::size_t>(node)] = 0U;
        }
      }
    }
  }

  auto* const d_saved_r = static_cast<double*>(core::device_scratch_acquire(
      "runtime_ale_targets:saved_r", node_bytes));
  auto* const d_saved_z = static_cast<double*>(core::device_scratch_acquire(
      "runtime_ale_targets:saved_z", node_bytes));
  auto* const d_mandate = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("runtime_ale_targets:mandate",
                                   winslow_mandate.size() *
                                       sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMemcpy(d_saved_r, state.x_r.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_saved_z, state.x_z.data(), node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_mandate, winslow_mandate.data(),
                        winslow_mandate.size(),
                        cudaMemcpyHostToDevice));

  const int winslow_sweeps =
      std::min(runtime_cfg.winslow_sweeps * escalation.sweep_scale, 32);
  (void)ale::multiblock_winslow_smooth_with_seams_stats(
      state, cfg, winslow_sweeps, runtime_cfg.winslow_omega, d_mandate);
  std::vector<double> x_w_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> x_w_z(static_cast<std::size_t>(n_nodes), 0.0);
  state.x_r.copy_to_host(x_w_r.data());
  state.x_z.copy_to_host(x_w_z.data());

  CUDA_CHECK(cudaMemcpy(state.x_r.data(), d_saved_r, node_bytes,
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(), d_saved_z, node_bytes,
                        cudaMemcpyDeviceToDevice));
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;

  std::vector<double> rho(static_cast<std::size_t>(n_cells), 0.0);
  state.rho.copy_to_host(rho.data());
  std::vector<double> x_m_r = x_w_r;
  std::vector<double> x_m_z = x_w_z;
  std::vector<double> s_in(static_cast<std::size_t>(idx.n_b) + 1U, 0.0);
  std::vector<double> s_out(static_cast<std::size_t>(idx.n_b) + 1U, 0.0);
  std::vector<double> rho_bins(static_cast<std::size_t>(idx.n_b), 0.0);
  constexpr double pi = std::numbers::pi_v<double>;
  for (int k = 0; k <= idx.ntheta; ++k) {
    for (int layer = 0; layer <= idx.n_b; ++layer) {
      const int node = bridge_node_id(idx, layer, k);
      s_in[static_cast<std::size_t>(layer)] =
          std::hypot(x_l_r[static_cast<std::size_t>(node)],
                     x_l_z[static_cast<std::size_t>(node)]);
    }
    for (int layer = 0; layer < idx.n_b; ++layer) {
      if (k == 0) {
        rho_bins[static_cast<std::size_t>(layer)] =
            rho[static_cast<std::size_t>(bridge_cell_id(idx, layer, 0))];
      } else if (k == idx.ntheta) {
        rho_bins[static_cast<std::size_t>(layer)] = rho[static_cast<std::size_t>(
            bridge_cell_id(idx, layer, idx.ntheta - 1))];
      } else {
        const double left = rho[static_cast<std::size_t>(
            bridge_cell_id(idx, layer, k - 1))];
        const double right = rho[static_cast<std::size_t>(
            bridge_cell_id(idx, layer, k))];
        rho_bins[static_cast<std::size_t>(layer)] = 0.5 * (left + right);
      }
    }

    const double s_front =
        ray_front_radius(front_s_per_column, idx.ntheta, k);
    double local_width = s_in[1] - s_in[0];
    if (s_front > 0.0) {
      int local_row = idx.n_b - 1;
      for (int layer = 0; layer < idx.n_b; ++layer) {
        if (s_front <= s_in[static_cast<std::size_t>(layer + 1)]) {
          local_row = layer;
          break;
        }
      }
      local_width = s_in[static_cast<std::size_t>(local_row + 1)] -
                    s_in[static_cast<std::size_t>(local_row)];
    }
    const double h_seam = s_in[static_cast<std::size_t>(idx.n_b)] -
                          s_in[static_cast<std::size_t>(idx.n_b - 1)];
    detail::equidistribute_column(
        s_in.data(), idx.n_b, rho_bins.data(), idx.n_b, s_front,
        runtime_cfg.front_width_cells * local_width, runtime_cfg.beta_mass,
        runtime_cfg.beta_front, runtime_cfg.g_max, h_seam, s_out.data());

    const double theta0 = pi * static_cast<double>(k) /
                          static_cast<double>(idx.ntheta);
    for (int layer = 0; layer <= idx.n_b; ++layer) {
      const int node = bridge_node_id(idx, layer, k);
      const std::size_t n = static_cast<std::size_t>(node);
      const double theta_l = std::atan2(x_l_r[n], x_l_z[n]);
      const double theta_m =
          (1.0 - runtime_cfg.beta_theta) * theta_l +
          runtime_cfg.beta_theta * theta0;
      x_m_r[n] = s_out[static_cast<std::size_t>(layer)] * std::sin(theta_m);
      x_m_z[n] = s_out[static_cast<std::size_t>(layer)] * std::cos(theta_m);
    }
  }

  const bool effective_hard_mode = hard_mode || escalation.force_hard;
  std::vector<double> cap_target_r = x_l_r;
  std::vector<double> cap_target_z = x_l_z;
  bool cap_target_feasible = true;
  if (cap_columns > 0) {
    TENRYU_ASSERT(state.v_r.size() == static_cast<std::size_t>(n_nodes) &&
                      state.v_z.size() == static_cast<std::size_t>(n_nodes),
                  "runtime ALE cap velocity size mismatch");
    std::vector<double> velocity_r(static_cast<std::size_t>(n_nodes), 0.0);
    std::vector<double> velocity_z(static_cast<std::size_t>(n_nodes), 0.0);
    state.v_r.copy_to_host(velocity_r.data());
    state.v_z.copy_to_host(velocity_z.data());
    const double dt_acoustic = compute_dt_hydro_acoustic(state, cfg);
    const double lookahead =
        (effective_hard_mode ? 2.0 : 4.0) * dt_acoustic;
    const int n_cap_rays = cap_columns + 1;

    for (int axis = 0; axis < 2; ++axis) {
      std::vector<double> average_ladder(
          static_cast<std::size_t>(cap_chain_rows) + 1U, 0.0);
      std::vector<double> average_density(
          static_cast<std::size_t>(cap_chain_rows), 0.0);
      std::vector<double> inner_endpoints(
          static_cast<std::size_t>(n_cap_rays), 0.0);
      std::vector<double> outer_endpoints(
          static_cast<std::size_t>(n_cap_rays), 0.0);
      std::vector<double> minimum_gaps(
          static_cast<std::size_t>(n_cap_rays * cap_chain_rows), 0.0);

      for (int ray = 0; ray < n_cap_rays; ++ray) {
        const int k = cap_ray_index(idx, axis, ray);
        double ray_r = 0.0;
        double ray_z = 0.0;
        cap_ray_direction(idx, k, &ray_r, &ray_z);
        for (int row = 0; row <= cap_chain_rows; ++row) {
          const int node = cap_chain_node_id(idx, row, k);
          const double radius =
              std::hypot(x_l_r[static_cast<std::size_t>(node)],
                         x_l_z[static_cast<std::size_t>(node)]);
          average_ladder[static_cast<std::size_t>(row)] +=
              radius / static_cast<double>(n_cap_rays);
          if (row == 0) {
            inner_endpoints[static_cast<std::size_t>(ray)] = radius;
          }
          if (row == cap_chain_rows) {
            outer_endpoints[static_cast<std::size_t>(ray)] = radius;
          }
        }
        for (int row = 0; row < cap_chain_rows; ++row) {
          const int inner_node = cap_chain_node_id(idx, row, k);
          const int outer_node = cap_chain_node_id(idx, row + 1, k);
          const double inner_radius =
              std::hypot(x_l_r[static_cast<std::size_t>(inner_node)],
                         x_l_z[static_cast<std::size_t>(inner_node)]);
          const double outer_radius =
              std::hypot(x_l_r[static_cast<std::size_t>(outer_node)],
                         x_l_z[static_cast<std::size_t>(outer_node)]);
          const double gap = outer_radius - inner_radius;
          const double inner_velocity =
              velocity_r[static_cast<std::size_t>(inner_node)] * ray_r +
              velocity_z[static_cast<std::size_t>(inner_node)] * ray_z;
          const double outer_velocity =
              velocity_r[static_cast<std::size_t>(outer_node)] * ray_r +
              velocity_z[static_cast<std::size_t>(outer_node)] * ray_z;
          const double gap_rate = outer_velocity - inner_velocity;
          const double minimum_gap = 0.10 * gap - lookahead * gap_rate;
          minimum_gaps[static_cast<std::size_t>(
              ray * cap_chain_rows + row)] = std::max(0.0, minimum_gap);
          if (!(std::isfinite(gap) && gap > 0.0 &&
                std::isfinite(minimum_gap))) {
            cap_target_feasible = false;
          }
        }
      }

      for (int row = 0; row < cap_chain_rows; ++row) {
        for (int offset = 0; offset < cap_columns; ++offset) {
          const int column =
              axis == 0 ? offset : idx.ntheta - 1 - offset;
          const int cell = cap_chain_cell_id(idx, row, column);
          average_density[static_cast<std::size_t>(row)] +=
              rho[static_cast<std::size_t>(cell)] /
              static_cast<double>(cap_columns);
        }
      }

      double cap_front = 0.0;
      int valid_front_columns = 0;
      if (front_s_per_column != nullptr) {
        for (int offset = 0; offset < cap_columns; ++offset) {
          const int column =
              axis == 0 ? offset : idx.ntheta - 1 - offset;
          const double front = front_s_per_column[column];
          if (std::isfinite(front) && front > 0.0) {
            cap_front += front;
            ++valid_front_columns;
          }
        }
      }
      cap_front = valid_front_columns > 0
                      ? cap_front / static_cast<double>(valid_front_columns)
                      : -1.0;
      double local_width = average_ladder[1] - average_ladder[0];
      if (cap_front > 0.0) {
        int local_row = cap_chain_rows - 1;
        for (int row = 0; row < cap_chain_rows; ++row) {
          if (cap_front <= average_ladder[static_cast<std::size_t>(row + 1)]) {
            local_row = row;
            break;
          }
        }
        local_width = average_ladder[static_cast<std::size_t>(local_row + 1)] -
                      average_ladder[static_cast<std::size_t>(local_row)];
      }
      const double outer_spacing =
          average_ladder[static_cast<std::size_t>(cap_chain_rows)] -
          average_ladder[static_cast<std::size_t>(cap_chain_rows - 1)];
      std::vector<double> desired_ladder(
          static_cast<std::size_t>(cap_chain_rows) + 1U, 0.0);
      detail::equidistribute_column(
          average_ladder.data(), cap_chain_rows, average_density.data(),
          cap_chain_rows, cap_front,
          runtime_cfg.front_width_cells * local_width, runtime_cfg.beta_mass,
          runtime_cfg.beta_front, runtime_cfg.g_max, outer_spacing,
          desired_ladder.data());

      std::vector<double> common_ladder;
      const bool axis_feasible =
          cap_target_feasible &&
          project_common_cap_ladder(
              desired_ladder, inner_endpoints, outer_endpoints, minimum_gaps,
              runtime_cfg.g_max, &common_ladder);
      cap_target_feasible = cap_target_feasible && axis_feasible;
      if (!axis_feasible) {
        continue;
      }
      for (int ray = 0; ray < n_cap_rays; ++ray) {
        const int k = cap_ray_index(idx, axis, ray);
        double ray_r = 0.0;
        double ray_z = 0.0;
        cap_ray_direction(idx, k, &ray_r, &ray_z);
        for (int row = 0; row <= cap_chain_rows; ++row) {
          const int node = cap_chain_node_id(idx, row, k);
          double radius = common_ladder[static_cast<std::size_t>(row)];
          if (row == 0) {
            radius = inner_endpoints[static_cast<std::size_t>(ray)];
          } else if (row == cap_chain_rows) {
            radius = outer_endpoints[static_cast<std::size_t>(ray)];
          }
          cap_target_r[static_cast<std::size_t>(node)] = radius * ray_r;
          cap_target_z[static_cast<std::size_t>(node)] = radius * ray_z;
        }
      }
    }
    if (!cap_target_feasible) {
      cap_target_r = x_l_r;
      cap_target_z = x_l_z;
      stats.cap_infeasible_events = 1;
    }
  }

  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "runtime ALE target CSR offset size mismatch");
  std::vector<double> q_node(static_cast<std::size_t>(n_nodes),
                             std::numeric_limits<double>::infinity());
  std::vector<double> h_node(static_cast<std::size_t>(n_nodes),
                             std::numeric_limits<double>::infinity());
  std::vector<std::vector<int>> neighbors(static_cast<std::size_t>(n_nodes));
  std::vector<std::vector<int>> node_cells(static_cast<std::size_t>(n_nodes));
  std::vector<std::uint8_t> has_normal(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<double> normal_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> normal_z(static_cast<std::size_t>(n_nodes), 0.0);
  const bool normal_arrays_present =
      shock_normal_dir_r != nullptr && shock_normal_dir_z != nullptr;
  for (int cell = 0; cell < n_cells; ++cell) {
    const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int end = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell + 1)];
    const int nv = end - begin;
    TENRYU_ASSERT(nv >= 3 &&
                      static_cast<std::size_t>(end) <=
                          mb.cell_node_csr_indices.size(),
                  "runtime ALE target CSR cell range invalid");
    double shoelace = 0.0;
    for (int corner = 0; corner < nv; ++corner) {
      const int next = (corner + 1 == nv) ? 0 : corner + 1;
      const int node = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + corner)];
      const int node_next = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + next)];
      shoelace += x_l_r[static_cast<std::size_t>(node)] *
                       x_l_z[static_cast<std::size_t>(node_next)] -
                   x_l_r[static_cast<std::size_t>(node_next)] *
                       x_l_z[static_cast<std::size_t>(node)];
    }
    const double orientation = shoelace < 0.0 ? -1.0 : 1.0;
    double q_cell = std::numeric_limits<double>::infinity();
    for (int corner = 0; corner < nv; ++corner) {
      const int next = (corner + 1 == nv) ? 0 : corner + 1;
      const int prev = (corner == 0) ? nv - 1 : corner - 1;
      const int node = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + corner)];
      const int node_next = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + next)];
      const int node_prev = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + prev)];
      const double e1_r = x_l_r[static_cast<std::size_t>(node_next)] -
                          x_l_r[static_cast<std::size_t>(node)];
      const double e1_z = x_l_z[static_cast<std::size_t>(node_next)] -
                          x_l_z[static_cast<std::size_t>(node)];
      const double e2_r = x_l_r[static_cast<std::size_t>(node_prev)] -
                          x_l_r[static_cast<std::size_t>(node)];
      const double e2_z = x_l_z[static_cast<std::size_t>(node_prev)] -
                          x_l_z[static_cast<std::size_t>(node)];
      q_cell = std::min(q_cell, detail::ale_monitor_corner_quality(
                                     e1_r, e1_z, e2_r, e2_z, orientation));
      append_unique_neighbor(neighbors[static_cast<std::size_t>(node)],
                             node_next);
      append_unique_neighbor(neighbors[static_cast<std::size_t>(node_next)],
                             node);
    }

    const double h_cell = detail::csr_polygon_characteristic_length_from_nodes(
        x_l_r.data(), x_l_z.data(),
        mb.cell_node_csr_indices.data() + begin, nv);
    double cell_normal_r = 0.0;
    double cell_normal_z = 0.0;
    bool cell_has_normal = false;
    if (normal_arrays_present) {
      const double nr = shock_normal_dir_r[cell];
      const double nz = shock_normal_dir_z[cell];
      const double norm = std::hypot(nr, nz);
      if (std::isfinite(nr) && std::isfinite(nz) && std::isfinite(norm) &&
          norm > 0.0) {
        cell_normal_r = nr / norm;
        cell_normal_z = nz / norm;
        cell_has_normal = true;
      }
    }
    for (int corner = 0; corner < nv; ++corner) {
      const int node = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + corner)];
      const std::size_t n = static_cast<std::size_t>(node);
      node_cells[n].push_back(cell);
      q_node[n] = std::min(q_node[n], q_cell);
      if (std::isfinite(h_cell) && h_cell > 0.0) {
        h_node[n] = std::min(h_node[n], h_cell);
      }
      if (cell_has_normal && has_normal[n] == 0U) {
        has_normal[n] = 1U;
        normal_r[n] = cell_normal_r;
        normal_z[n] = cell_normal_z;
      }
    }
  }

  std::vector<double> omega(static_cast<std::size_t>(n_nodes), 0.0);
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t n = static_cast<std::size_t>(node);
    if (mandate[n] != 0U && std::isfinite(q_node[n])) {
      omega[n] = smoothstep_activation(q_node[n], runtime_cfg.q_hard,
                                       runtime_cfg.q_recover);
    }
  }
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<double> next = omega;
    for (int node = 0; node < n_nodes; ++node) {
      const std::size_t n = static_cast<std::size_t>(node);
      if (mandate[n] == 0U) {
        next[n] = 0.0;
        continue;
      }
      double sum = 0.0;
      int count = 0;
      for (const int neighbor : neighbors[n]) {
        const std::size_t nb = static_cast<std::size_t>(neighbor);
        if (mandate[nb] != 0U) {
          sum += omega[nb];
          ++count;
        }
      }
      if (count > 0) {
        next[n] = sum / static_cast<double>(count);
      }
    }
    omega.swap(next);
  }

  const double beta_monitor = effective_hard_mode
                                  ? runtime_cfg.beta_monitor_hard
                                  : runtime_cfg.beta_monitor_soft;
  const double cap_fraction = runtime_cfg.cap_fraction * escalation.cap_scale;
  const double cap_normal_fraction =
      runtime_cfg.cap_normal_fraction * escalation.cap_scale;
  std::vector<double> target_r = x_l_r;
  std::vector<double> target_z = x_l_z;
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t n = static_cast<std::size_t>(node);
    if (mandate[n] == 0U) {
      continue;
    }
    double dr = 0.0;
    double dz = 0.0;
    if (cap_node[n] != 0U) {
      dr = omega[n] * (cap_target_r[n] - x_l_r[n]);
      dz = omega[n] * (cap_target_z[n] - x_l_z[n]);
    } else {
      dr = omega[n] *
           ((1.0 - beta_monitor) * (x_w_r[n] - x_l_r[n]) +
            beta_monitor * (x_m_r[n] - x_l_r[n]));
      dz = omega[n] *
           ((1.0 - beta_monitor) * (x_w_z[n] - x_l_z[n]) +
            beta_monitor * (x_m_z[n] - x_l_z[n]));
    }
    if (x_l_r[n] == 0.0) {
      dr = 0.0;
    }
    const ClampOutcome outcome = clamp_displacement_impl(
        &dr, &dz, h_node[n], cap_fraction, cap_normal_fraction,
        has_normal[n] != 0U, normal_r[n], normal_z[n]);
    stats.capped_nodes += outcome.total ? 1 : 0;
    stats.normal_capped_nodes += outcome.normal ? 1 : 0;
    target_r[n] = x_l_r[n] + dr;
    target_z[n] = x_l_z[n] + dz;
    if (x_l_r[n] == 0.0) {
      target_r[n] = 0.0;
    }
  }

  std::vector<std::pair<int, int>> projection_failures;
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t n = static_cast<std::size_t>(node);
    if (mandate[n] == 0U || cap_node[n] != 0U || x_l_r[n] == 0.0) {
      continue;
    }
    const OrdinaryProjectionResult projection = project_ordinary_node(
        node, node_cells, mb.cell_node_csr_offsets,
        mb.cell_node_csr_indices, x_l_r, x_l_z, &target_r[n], &target_z[n]);
    stats.ordinary_projected_nodes += projection.projected ? 1 : 0;
    if (projection.failing_cell >= 0) {
      projection_failures.emplace_back(node, projection.failing_cell);
    }
  }
  for (const auto& [failing_node, failing_cell] : projection_failures) {
    ++stats.patch_escalations;
    const CoupledPatchOutcome patch = project_coupled_patch(
        idx, failing_node, failing_cell, mandate, cap_node, cap_ray_r,
        cap_ray_z, node_cells, neighbors, mb.cell_node_csr_offsets,
        mb.cell_node_csr_indices, x_l_r, x_l_z, &target_r, &target_z);
    stats.patch_nodes_max = std::max(stats.patch_nodes_max, patch.node_count);
    stats.patch_oversize_skips += patch.oversize ? 1 : 0;
  }

  if (cap_columns > 0 && cap_target_feasible) {
    std::vector<std::uint8_t> projected(
        static_cast<std::size_t>(n_nodes), 0U);
    constexpr int geometry_projection_passes = 2;
    for (int pass = 0; pass < geometry_projection_passes; ++pass) {
      for (int axis = 0; axis < 2; ++axis) {
        for (int ray = 0; ray <= cap_columns; ++ray) {
          const int k = cap_ray_index(idx, axis, ray);
          double ray_r = 0.0;
          double ray_z = 0.0;
          cap_ray_direction(idx, k, &ray_r, &ray_z);
          for (int row = 1; row < cap_chain_rows; ++row) {
            const int node = cap_chain_node_id(idx, row, k);
            const std::size_t n = static_cast<std::size_t>(node);
            if (mandate[n] == 0U) {
              continue;
            }
            double lower = 0.0;
            double upper = std::numeric_limits<double>::infinity();
            if (!cap_node_feasible_interval(
                    node, ray_r, ray_z, node_cells,
                    mb.cell_node_csr_offsets, mb.cell_node_csr_indices,
                    x_l_r, x_l_z, target_r, target_z, &lower, &upper)) {
              continue;
            }
            const double raw_r = target_r[n];
            const double raw_z = target_z[n];
            const double raw_scalar = raw_r * ray_r + raw_z * ray_z;
            const double projected_scalar =
                std::clamp(raw_scalar, lower, upper);
            const double new_r = projected_scalar * ray_r;
            const double new_z = projected_scalar * ray_z;
            const double projection_tolerance =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max(std::abs(projected_scalar),
                         std::numeric_limits<double>::min());
            if (std::hypot(new_r - raw_r, new_z - raw_z) >
                projection_tolerance) {
              projected[n] = 1U;
            }
            target_r[n] = new_r;
            target_z[n] = new_z;
          }
        }
      }
    }
    for (const std::uint8_t was_projected : projected) {
      stats.cap_projected_nodes += was_projected != 0U ? 1 : 0;
    }
  }
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t n = static_cast<std::size_t>(node);
    double dr = target_r[n] - x_l_r[n];
    const double unclamped_target_r = x_l_r[n] + dr;
    target_r[n] = std::max(0.0, unclamped_target_r);
    stats.half_plane_clamped_nodes += unclamped_target_r < 0.0 ? 1 : 0;
    dr = target_r[n] - x_l_r[n];
    const double dz = target_z[n] - x_l_z[n];
    stats.max_displacement =
        std::max(stats.max_displacement, std::hypot(dr, dz));
  }

  target_x_r.reset(static_cast<std::size_t>(n_nodes));
  target_x_z.reset(static_cast<std::size_t>(n_nodes));
  target_x_r.copy_from_host(target_r);
  target_x_z.copy_from_host(target_z);
  stats.valid = true;
  return stats;
}

RuntimeAleEventResult run_runtime_ale_event(
    core::State& state,
    const core::Config& cfg,
    const bool hard_mode,
    const RuntimeAleEscalation& escalation,
    const double* const front_s_per_column) {
  RuntimeAleEventResult result;
  core::DeviceArray<double> target_x_r("runtime_ale_event:target_x_r");
  core::DeviceArray<double> target_x_z("runtime_ale_event:target_x_z");
  const RuntimeAleTargetStats target_stats = build_runtime_ale_target(
      state, cfg, hard_mode, escalation, front_s_per_column, nullptr, nullptr,
      target_x_r, target_x_z);
  result.max_displacement = target_stats.max_displacement;
  result.capped_nodes = target_stats.capped_nodes;
  result.half_plane_clamped_nodes = target_stats.half_plane_clamped_nodes;
  result.cap_projected_nodes = target_stats.cap_projected_nodes;
  result.ordinary_projected_nodes = target_stats.ordinary_projected_nodes;
  result.cap_infeasible_events = target_stats.cap_infeasible_events;
  result.patch_escalations = target_stats.patch_escalations;
  result.patch_nodes_max = target_stats.patch_nodes_max;
  result.patch_oversize_skips = target_stats.patch_oversize_skips;
  result.mandate_nodes = target_stats.mandate_nodes;
  if (target_stats.max_displacement < 1.0e-14) {
    return result;
  }

  const ButtonIndexing idx = button_morph::make_button_indexing(cfg.mesh);
  std::vector<std::uint8_t> active_cell_mask(
      static_cast<std::size_t>(idx.n_cells_total), 0U);
  for (int cell = 0; cell < idx.shell_cell_offset; ++cell) {
    active_cell_mask[static_cast<std::size_t>(cell)] = 1U;
  }
  for (int row = 0;
       row < cfg.numerics.ale.runtime_controller.controller_shell_rows;
       ++row) {
    for (int k = 0; k < idx.ntheta; ++k) {
      const int cell = idx.shell_cell_offset + row * idx.ntheta + k;
      active_cell_mask[static_cast<std::size_t>(cell)] = 1U;
    }
  }
  std::vector<std::uint8_t> frozen_velocity_node_mask(
      static_cast<std::size_t>(idx.n_nodes_total), 1U);
  for (int node = 0; node < idx.shell_node_offset; ++node) {
    frozen_velocity_node_mask[static_cast<std::size_t>(node)] = 0U;
  }
  for (int ring = 0;
       ring < cfg.numerics.ale.runtime_controller.controller_shell_rows;
       ++ring) {
    for (int k = 0; k <= idx.ntheta; ++k) {
      const int node = button_morph::shell_node_id(idx, ring, k);
      frozen_velocity_node_mask[static_cast<std::size_t>(node)] = 0U;
    }
  }
  const int center_node = button_morph::core_node_id(idx, 0, idx.n_c);
  frozen_velocity_node_mask[static_cast<std::size_t>(center_node)] = 1U;

  auto* const d_active_cell_mask = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("runtime_ale_event:active_cell_mask",
                                   active_cell_mask.size()));
  auto* const d_frozen_velocity_node_mask = static_cast<std::uint8_t*>(
      core::device_scratch_acquire(
          "runtime_ale_event:frozen_velocity_node_mask",
          frozen_velocity_node_mask.size()));
  CUDA_CHECK(cudaMemcpy(d_active_cell_mask, active_cell_mask.data(),
                        active_cell_mask.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_frozen_velocity_node_mask,
                        frozen_velocity_node_mask.data(),
                        frozen_velocity_node_mask.size(),
                        cudaMemcpyHostToDevice));

  const ReferenceBarrierScope scope{d_active_cell_mask,
                                    d_frozen_velocity_node_mask,
                                    true};
  const ReferenceBarrierAleResult barrier = apply_reference_barrier_rezone(
      state, cfg, target_x_r.data(), target_x_z.data(), &scope);
  result.attempted = barrier.engaged;
  result.succeeded = barrier.succeeded;
  result.rolled_back = barrier.rolled_back;
  result.lambda_accepted = barrier.lambda_accepted;
  result.linesearch_iters = barrier.linesearch_iters;
  result.last_reject_reason = barrier.last_reject_reason;
  result.last_reject_cell = barrier.last_reject_cell;
  if (barrier.succeeded) {
    // W4d: diagnostic-only post-remap audit; the geometry-only pre-commit
    // clamp-free gate owns transaction acceptance.
    TENRYU_ASSERT(state.corner_mass.size() == active_cell_mask.size() * 4U,
                  "[runtime-ale-corner-mass] corner_mass size mismatch");
    TENRYU_ASSERT(state.mass.size() == active_cell_mask.size(),
                  "[runtime-ale-corner-mass] cell_mass size mismatch");
    std::vector<double> corner_mass;
    std::vector<double> cell_mass;
    state.corner_mass.copy_to_host(corner_mass);
    state.mass.copy_to_host(cell_mass);
    corner_mass_cell_closure_residual(
        corner_mass.data(), cell_mass.data(), active_cell_mask.data(),
        idx.n_cells_total, &result.corner_mass_cell_residual);

    std::ostringstream message;
    message.setf(std::ios::scientific);
    message.precision(17);
    message << "[runtime-ale-corner-mass] cm_cell="
            << result.corner_mass_cell_residual;
    if (result.corner_mass_cell_residual > 1.0e-11) {
      core::log_warning(message.str());
    }
  }
  return result;
}

}  // namespace tenryu::hydro
