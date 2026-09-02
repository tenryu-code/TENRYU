#include "hydro/reale_remap.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"

namespace tenryu::hydro::reale::detail {

inline constexpr std::size_t kDvclpNewtonDofCap = 256;
inline constexpr int kDvclpDykstraSweepCap = 256;
inline constexpr int kDvclpQuotientSweepCap = 64;

struct DvclpTestCaps {
  std::size_t dof_cap = kDvclpNewtonDofCap;
  int sweep_cap = kDvclpDykstraSweepCap;
  int quotient_sweep_cap = kDvclpQuotientSweepCap;
  int translation_sweep_cap = kDvclpDykstraSweepCap;
};

struct DvclpEdgeIncidence {
  int cell = -1;
  std::size_t corner0 = 0;
  std::size_t corner1 = 0;
};

struct DvclpEdgeBudget {
  double length = 0.0;
  double raw_jump = 0.0;
  double budget = 0.0;
  int node0 = -1;
  int node1 = -1;
  int walk_segments = 0;
  bool walk_failed = false;
  std::array<DvclpEdgeIncidence, 2> incident_corners{};
  int incident_count = 0;
};

struct DvclpProjectionResult {
  bool ok = true;
  std::string reason;
  int events = 0;
  int sweeps = 0;
  int alternations = 0;
  int consensus_ms = 0;
  int sweep_ms = 0;
  int solve_ms = 0;
  int certify_ms = 0;
  double q_total = 0.0;
  long double axis_projection_r = 0.0L;
  double max_R_before = 0.0;
  double max_R_after = 0.0;
  std::size_t unconstrained = 0;
  std::size_t representation_limited = 0;
  std::size_t consensus_edges = 0;
  std::size_t consensus_components = 0;
  std::size_t consensus_nodes = 0;
  std::size_t homothety_components = 0;
  std::size_t homothety_expansions = 0;
  std::size_t newton_capped = 0;
  double alpha_min = 1.0;
  std::size_t quotient_certified_edges = 0;
  std::size_t resolution_certified_edges = 0;
  std::size_t lattice_certified_edges = 0;
  int nonstar_fallback_cells = 0;
  int degenerate_skipped_triangles = 0;
  std::size_t kkt_certified_components = 0;
  std::size_t kkt_fail_eq = 0;
  std::size_t kkt_fail_stationarity = 0;
  std::size_t kkt_fail_cone = 0;
  double kkt_lam_max = 0.0;
  std::size_t qr_dropped_rows = 0;
};

struct DvclpKktCertificate {
  bool certified = false;
  // 0 = none, 1 = equality/band re-check (78), 2 = stationarity (74),
  // 3 = normal cone (75)/(79).
  int fail_clause = 0;
  double lam_max = 0.0;
  std::size_t active_edges = 0;
};

bool build_dvclp_edge_budgets_core(
    const RemapMesh& old_mesh,
    const RemapMesh& target_mesh,
    const RemapFields& old_fields,
    const double* target_node_vr,
    const double* target_node_vz,
    const std::vector<std::pair<int, int>>& ring_edges,
    const std::vector<double>& edge_lengths,
    const std::vector<std::vector<DvclpEdgeIncidence>>& edge_incidences,
    const std::vector<std::vector<int>>& candidate_cells,
    std::vector<DvclpEdgeBudget>& budgets,
    std::string& reason,
    int& nonstar_fallback_cells,
    int& degenerate_skipped_triangles);

DvclpProjectionResult project_dvclp_velocities_core(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const double* nodal_mass,
    const double* corner_mass,
    double* corner_momentum_r,
    double* corner_momentum_z,
    double* node_vr,
    double* node_vz,
    double* cell_q_projection,
    const DvclpTestCaps& test_caps = {},
    const int solver_rev = 0);

DvclpKktCertificate certify_dvclp_kkt(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const std::vector<std::size_t>& component_edges,
    const std::vector<int>& members,
    const std::vector<bool>& node_z_in_component,
    const std::vector<bool>& node_r_in_component,
    const double* nodal_mass,
    const std::vector<double>& candidate_vr,
    const std::vector<double>& candidate_vz,
    const std::vector<double>& snapshot_vr,
    const std::vector<double>& snapshot_vz);

namespace {

bool dvclp_component_diagnostic_enabled() {
  const char* dvclp_env = std::getenv("TENRYU_DVCLP_DIAG");
  return dvclp_env != nullptr && std::atoi(dvclp_env) == 1;
}

struct DonorTriangle {
  std::array<double, 3> r{};
  std::array<double, 3> z{};
  double gradient_rr = 0.0;
  double gradient_rz = 0.0;
  double gradient_zr = 0.0;
  double gradient_zz = 0.0;
  double orientation = 1.0;
  int cell = -1;
  int apex = -1;
  int triangle = -1;
};

double node_r(const RemapMesh& mesh, const int begin, const int vertex) {
  return mesh.node_r[mesh.csr_indices[begin + vertex]];
}

double node_z(const RemapMesh& mesh, const int begin, const int vertex) {
  return mesh.node_z[mesh.csr_indices[begin + vertex]];
}

bool build_donor_triangles(
    const RemapMesh& old_mesh,
    const RemapFields& old_fields,
    std::vector<DonorTriangle>& triangles,
    std::vector<std::vector<std::size_t>>& triangles_by_cell,
    std::string& reason) {
  triangles.clear();
  triangles_by_cell.assign(static_cast<std::size_t>(old_mesh.n_cells), {});
  int nonstar_fallback_cells = 0;
  int degenerate_skipped_triangles = 0;
  int first_fallback_cell = -1;
  double first_fallback_margin = 0.0;
  int first_degenerate_cell = -1;
  for (int cell = 0; cell < old_mesh.n_cells; ++cell) {
    const int begin = old_mesh.csr_offsets[cell];
    const int nverts = static_cast<int>(old_mesh.nverts[cell]);
    double ring_twice_area = 0.0;
    for (int vertex = 0; vertex < nverts; ++vertex) {
      const int next = vertex + 1 == nverts ? 0 : vertex + 1;
      ring_twice_area +=
          node_r(old_mesh, begin, vertex) * node_z(old_mesh, begin, next) -
          node_r(old_mesh, begin, next) * node_z(old_mesh, begin, vertex);
    }
    const double ring_orientation = ring_twice_area > 0.0 ? 1.0 : -1.0;

    int selected_apex = -1;
    for (int apex = 0; apex < nverts; ++apex) {
      bool valid = true;
      const double apex_r = node_r(old_mesh, begin, apex);
      const double apex_z = node_z(old_mesh, begin, apex);
      for (int triangle = 0; triangle < nverts - 2; ++triangle) {
        const int vertex_b = (apex + triangle + 1) % nverts;
        const int vertex_c = (apex + triangle + 2) % nverts;
        const double twice_area =
            (node_r(old_mesh, begin, vertex_b) - apex_r) *
                (node_z(old_mesh, begin, vertex_c) - apex_z) -
            (node_r(old_mesh, begin, vertex_c) - apex_r) *
                (node_z(old_mesh, begin, vertex_b) - apex_z);
        if (!(ring_orientation * twice_area > 0.0)) {
          valid = false;
          break;
        }
      }
      if (valid) {
        selected_apex = apex;
        break;
      }
    }
    if (selected_apex < 0) {
      int best_apex = -1;
      double best_margin = -std::numeric_limits<double>::infinity();
      for (int apex = 0; apex < nverts; ++apex) {
        const double apex_r = node_r(old_mesh, begin, apex);
        const double apex_z = node_z(old_mesh, begin, apex);
        double margin = std::numeric_limits<double>::infinity();
        for (int triangle = 0; triangle < nverts - 2; ++triangle) {
          const int vertex_b = (apex + triangle + 1) % nverts;
          const int vertex_c = (apex + triangle + 2) % nverts;
          const double twice_area =
              (node_r(old_mesh, begin, vertex_b) - apex_r) *
                  (node_z(old_mesh, begin, vertex_c) - apex_z) -
              (node_r(old_mesh, begin, vertex_c) - apex_r) *
                  (node_z(old_mesh, begin, vertex_b) - apex_z);
          margin = std::min(margin, ring_orientation * twice_area);
        }
        if (margin > best_margin) {
          best_apex = apex;
          best_margin = margin;
        }
      }
      selected_apex = best_apex;
      ++nonstar_fallback_cells;
      if (first_fallback_cell < 0) {
        first_fallback_cell = cell;
        first_fallback_margin = best_margin;
      }
      if (nonstar_fallback_cells == 1) {
        std::fprintf(stderr,
                     "[dvclp_nonstar] cell=%d nverts=%d "
                     "ring_twice_area=%.17e\n",
                     cell, nverts, ring_twice_area);
        double min_edge = std::numeric_limits<double>::infinity();
        int min_edge_at = -1;
        double max_edge = 0.0;
        int folds = 0;
        for (int vertex = 0; vertex < nverts; ++vertex) {
          const int previous = vertex == 0 ? nverts - 1 : vertex - 1;
          const int next = vertex + 1 == nverts ? 0 : vertex + 1;
          const double vertex_r = node_r(old_mesh, begin, vertex);
          const double vertex_z = node_z(old_mesh, begin, vertex);
          const double delta_r_next =
              node_r(old_mesh, begin, next) - vertex_r;
          const double delta_z_next =
              node_z(old_mesh, begin, next) - vertex_z;
          const double edge_len_next =
              std::hypot(delta_r_next, delta_z_next);
          if (edge_len_next < min_edge) {
            min_edge = edge_len_next;
            min_edge_at = vertex;
          }
          if (edge_len_next > max_edge) {
            max_edge = edge_len_next;
          }
          const double fold_cross =
              (vertex_r - node_r(old_mesh, begin, previous)) *
                  delta_z_next -
              (vertex_z - node_z(old_mesh, begin, previous)) *
                  delta_r_next;
          if (ring_orientation * fold_cross < 0.0) {
            ++folds;
          }
          if (vertex < 40) {
            std::fprintf(stderr,
                         "[dvclp_nonstar_v] k=%d node=%d r=%.17g z=%.17g "
                         "edge_len_next=%.17e\n",
                         vertex, old_mesh.csr_indices[begin + vertex],
                         vertex_r, vertex_z, edge_len_next);
          }
        }
        if (nverts > 40) {
          std::fprintf(stderr, "... nverts=%d truncated\n", nverts);
        }
        std::fprintf(stderr,
                     "[dvclp_nonstar_sum] min_edge=%.17e min_edge_at=%d "
                     "max_edge=%.17e folds=%d\n",
                     min_edge, min_edge_at, max_edge, folds);
        const int diagnostic_apices[3] = {0, 1, nverts / 2};
        for (int candidate = 0; candidate < 3; ++candidate) {
          const int apex = diagnostic_apices[candidate];
          bool duplicate = false;
          for (int prior = 0; prior < candidate; ++prior) {
            if (diagnostic_apices[prior] == apex) {
              duplicate = true;
              break;
            }
          }
          if (duplicate || apex < 0 || apex >= nverts) {
            continue;
          }
          const double apex_r = node_r(old_mesh, begin, apex);
          const double apex_z = node_z(old_mesh, begin, apex);
          for (int triangle = 0; triangle < nverts - 2; ++triangle) {
            const int vertex_b = (apex + triangle + 1) % nverts;
            const int vertex_c = (apex + triangle + 2) % nverts;
            const double twice_area =
                (node_r(old_mesh, begin, vertex_b) - apex_r) *
                    (node_z(old_mesh, begin, vertex_c) - apex_z) -
                (node_r(old_mesh, begin, vertex_c) - apex_r) *
                    (node_z(old_mesh, begin, vertex_b) - apex_z);
            if (!(ring_orientation * twice_area > 0.0)) {
              std::fprintf(stderr,
                           "[dvclp_nonstar_apex] apex=%d fail_tri=%d "
                           "twice_area=%.17e\n",
                           apex, triangle, twice_area);
              break;
            }
          }
        }
      }
    }

    for (int triangle_index = 0; triangle_index < nverts - 2;
         ++triangle_index) {
      const int ring_vertices[3] = {
          selected_apex,
          (selected_apex + triangle_index + 1) % nverts,
          (selected_apex + triangle_index + 2) % nverts};
      DonorTriangle triangle;
      triangle.cell = cell;
      triangle.apex = selected_apex;
      triangle.triangle = triangle_index;
      for (int vertex = 0; vertex < 3; ++vertex) {
        triangle.r[vertex] =
            node_r(old_mesh, begin, ring_vertices[vertex]);
        triangle.z[vertex] =
            node_z(old_mesh, begin, ring_vertices[vertex]);
      }

      const double delta_r_b = triangle.r[1] - triangle.r[0];
      const double delta_z_b = triangle.z[1] - triangle.z[0];
      const double delta_r_c = triangle.r[2] - triangle.r[0];
      const double delta_z_c = triangle.z[2] - triangle.z[0];
      const double determinant =
          delta_r_b * delta_z_c - delta_r_c * delta_z_b;
      if (determinant == 0.0) {
        ++degenerate_skipped_triangles;
        if (first_degenerate_cell < 0) {
          first_degenerate_cell = cell;
        }
        continue;
      }
      triangle.orientation = determinant > 0.0 ? 1.0 : -1.0;

      const int node_a = old_mesh.csr_indices[begin + ring_vertices[0]];
      const int node_b = old_mesh.csr_indices[begin + ring_vertices[1]];
      const int node_c = old_mesh.csr_indices[begin + ring_vertices[2]];
      const double delta_vr_b =
          old_fields.node_vr[node_b] - old_fields.node_vr[node_a];
      const double delta_vr_c =
          old_fields.node_vr[node_c] - old_fields.node_vr[node_a];
      const double delta_vz_b =
          old_fields.node_vz[node_b] - old_fields.node_vz[node_a];
      const double delta_vz_c =
          old_fields.node_vz[node_c] - old_fields.node_vz[node_a];
      triangle.gradient_rr =
          (delta_vr_b * delta_z_c - delta_vr_c * delta_z_b) /
          determinant;
      triangle.gradient_rz =
          (-delta_vr_b * delta_r_c + delta_vr_c * delta_r_b) /
          determinant;
      triangle.gradient_zr =
          (delta_vz_b * delta_z_c - delta_vz_c * delta_z_b) /
          determinant;
      triangle.gradient_zz =
          (-delta_vz_b * delta_r_c + delta_vz_c * delta_r_b) /
          determinant;
      triangles_by_cell[static_cast<std::size_t>(cell)].push_back(
          triangles.size());
      triangles.push_back(triangle);
    }
  }
  if (nonstar_fallback_cells > 0) {
    std::fprintf(stderr,
                 "[dvclp_nonstar_fallback] cells=%d first_cell=%d "
                 "first_margin=%.17e\n",
                 nonstar_fallback_cells, first_fallback_cell,
                 first_fallback_margin);
  }
  if (degenerate_skipped_triangles > 0) {
    std::fprintf(stderr,
                 "[dvclp_degenerate_skip] triangles=%d first_cell=%d\n",
                 degenerate_skipped_triangles, first_degenerate_cell);
  }
  (void)reason;
  return true;
}

bool contains_point(const DonorTriangle& triangle,
                    const double r,
                    const double z) {
  for (int edge = 0; edge < 3; ++edge) {
    const int next = edge + 1 == 3 ? 0 : edge + 1;
    const double side =
        triangle.orientation *
        ((triangle.r[next] - triangle.r[edge]) * (z - triangle.z[edge]) -
         (triangle.z[next] - triangle.z[edge]) * (r - triangle.r[edge]));
    if (!(side >= 0.0)) {
      return false;
    }
  }
  return true;
}

int locate_triangle(
    const double r,
    const double z,
    const std::vector<int>& candidate_cells,
    const std::vector<DonorTriangle>& triangles,
    const std::vector<std::vector<std::size_t>>& triangles_by_cell) {
  int selected = -1;
  for (const int cell : candidate_cells) {
    for (const std::size_t triangle_index :
         triangles_by_cell[static_cast<std::size_t>(cell)]) {
      const DonorTriangle& triangle = triangles[triangle_index];
      if (!contains_point(triangle, r, z)) {
        continue;
      }
      if (selected < 0) {
        selected = static_cast<int>(triangle_index);
        continue;
      }
      const DonorTriangle& previous =
          triangles[static_cast<std::size_t>(selected)];
      if (triangle.cell < previous.cell ||
          (triangle.cell == previous.cell &&
           (triangle.apex < previous.apex ||
            (triangle.apex == previous.apex &&
             triangle.triangle < previous.triangle)))) {
        selected = static_cast<int>(triangle_index);
      }
    }
  }
  return selected;
}

void append_intersection_parameters(
    const double start_r,
    const double start_z,
    const double delta_r,
    const double delta_z,
    double edge_a_r,
    double edge_a_z,
    double edge_b_r,
    double edge_b_z,
    std::vector<double>& parameters) {
  if (edge_b_r < edge_a_r ||
      (edge_b_r == edge_a_r && edge_b_z < edge_a_z)) {
    std::swap(edge_a_r, edge_b_r);
    std::swap(edge_a_z, edge_b_z);
  }
  const double edge_delta_r = edge_b_r - edge_a_r;
  const double edge_delta_z = edge_b_z - edge_a_z;
  const double denominator =
      delta_r * edge_delta_z - delta_z * edge_delta_r;
  const double offset_r = edge_a_r - start_r;
  const double offset_z = edge_a_z - start_z;
  if (denominator == 0.0) {
    const double collinearity =
        offset_r * delta_z - offset_z * delta_r;
    if (collinearity != 0.0) {
      return;
    }
    double edge_start = 0.0;
    double edge_end = 0.0;
    if (delta_r != 0.0) {
      edge_start = (edge_a_r - start_r) / delta_r;
      edge_end = (edge_b_r - start_r) / delta_r;
    } else if (delta_z != 0.0) {
      edge_start = (edge_a_z - start_z) / delta_z;
      edge_end = (edge_b_z - start_z) / delta_z;
    } else {
      return;
    }
    const double overlap_start =
        std::max(0.0, std::min(edge_start, edge_end));
    const double overlap_end =
        std::min(1.0, std::max(edge_start, edge_end));
    if (overlap_start > overlap_end) {
      return;
    }
    parameters.push_back(overlap_start);
    parameters.push_back(overlap_end);
    return;
  }
  const double raw_parameter =
      (offset_r * edge_delta_z - offset_z * edge_delta_r) / denominator;
  const double edge_parameter =
      (offset_r * delta_z - offset_z * delta_r) / denominator;
  if (raw_parameter < 0.0 || raw_parameter > 1.0 ||
      edge_parameter < 0.0 || edge_parameter > 1.0) {
    return;
  }
  parameters.push_back(std::clamp(raw_parameter, 0.0, 1.0));
}

void walk_segment(
    const double start_r,
    const double start_z,
    const double delta_r,
    const double delta_z,
    const std::vector<int>& candidate_cells,
    const std::vector<DonorTriangle>& triangles,
    const std::vector<std::vector<std::size_t>>& triangles_by_cell,
    double& budget,
    int& walk_segments,
    bool& walk_failed) {
  budget = 0.0;
  walk_segments = 0;
  walk_failed = false;
  std::vector<double> crossings{0.0, 1.0};
  for (const int cell : candidate_cells) {
    for (const std::size_t triangle_index :
         triangles_by_cell[static_cast<std::size_t>(cell)]) {
      const DonorTriangle& triangle = triangles[triangle_index];
      for (int edge = 0; edge < 3; ++edge) {
        const int next = edge + 1 == 3 ? 0 : edge + 1;
        append_intersection_parameters(
            start_r, start_z, delta_r, delta_z, triangle.r[edge],
            triangle.z[edge], triangle.r[next], triangle.z[next],
            crossings);
      }
    }
  }
  std::sort(crossings.begin(), crossings.end());
  crossings.erase(std::unique(crossings.begin(), crossings.end()),
                  crossings.end());

  for (std::size_t interval = 0; interval + 1 < crossings.size();
       ++interval) {
    const double current = crossings[interval];
    const double next = crossings[interval + 1];
    if (next == current) {
      continue;
    }
    if (!(next > current) || walk_segments >= 4096) {
      walk_failed = true;
      return;
    }
    const double midpoint = current + 0.5 * (next - current);
    if (!(midpoint > current) || !(midpoint < next)) {
      continue;
    }
    const double midpoint_r = start_r + midpoint * delta_r;
    const double midpoint_z = start_z + midpoint * delta_z;
    int current_triangle = locate_triangle(
        midpoint_r, midpoint_z, candidate_cells, triangles,
        triangles_by_cell);
    if (current_triangle < 0 ||
        !contains_point(
            triangles[static_cast<std::size_t>(current_triangle)],
            midpoint_r, midpoint_z)) {
      const double current_r = start_r + current * delta_r;
      const double current_z = start_z + current * delta_z;
      current_triangle = locate_triangle(
          current_r, current_z, candidate_cells, triangles,
          triangles_by_cell);
      if (current_triangle < 0 ||
          !contains_point(
              triangles[static_cast<std::size_t>(current_triangle)],
              current_r, current_z)) {
        walk_failed = true;
        return;
      }
    }
    const DonorTriangle& triangle =
        triangles[static_cast<std::size_t>(current_triangle)];
    const double derivative_r =
        triangle.gradient_rr * delta_r + triangle.gradient_rz * delta_z;
    const double derivative_z =
        triangle.gradient_zr * delta_r + triangle.gradient_zz * delta_z;
    budget += (next - current) *
              std::hypot(derivative_r, derivative_z);
    ++walk_segments;
  }
}

}  // namespace

bool build_dvclp_edge_budgets_core(
    const RemapMesh& old_mesh,
    const RemapMesh& target_mesh,
    const RemapFields& old_fields,
    const double* target_node_vr,
    const double* target_node_vz,
    const std::vector<std::pair<int, int>>& ring_edges,
    const std::vector<double>& edge_lengths,
    const std::vector<std::vector<DvclpEdgeIncidence>>& edge_incidences,
    const std::vector<std::vector<int>>& candidate_cells,
    std::vector<DvclpEdgeBudget>& budgets,
    std::string& reason,
    int& nonstar_fallback_cells,
    int& degenerate_skipped_triangles) {
  std::vector<DonorTriangle> triangles;
  std::vector<std::vector<std::size_t>> triangles_by_cell;
  if (!build_donor_triangles(old_mesh, old_fields, triangles,
                             triangles_by_cell, reason)) {
    return false;
  }
  nonstar_fallback_cells = 0;
  degenerate_skipped_triangles = 0;
  for (int cell = 0; cell < old_mesh.n_cells; ++cell) {
    const int begin = old_mesh.csr_offsets[cell];
    const int nverts = static_cast<int>(old_mesh.nverts[cell]);
    const int emitted = static_cast<int>(
        triangles_by_cell[static_cast<std::size_t>(cell)].size());
    degenerate_skipped_triangles += nverts - 2 - emitted;
    bool used_fallback = emitted != nverts - 2;
    if (!used_fallback) {
      double ring_twice_area = 0.0;
      for (int vertex = 0; vertex < nverts; ++vertex) {
        const int next = vertex + 1 == nverts ? 0 : vertex + 1;
        ring_twice_area +=
            node_r(old_mesh, begin, vertex) *
                node_z(old_mesh, begin, next) -
            node_r(old_mesh, begin, next) *
                node_z(old_mesh, begin, vertex);
      }
      const double ring_orientation =
          ring_twice_area > 0.0 ? 1.0 : -1.0;
      for (const std::size_t triangle_index :
           triangles_by_cell[static_cast<std::size_t>(cell)]) {
        if (triangles[triangle_index].orientation != ring_orientation) {
          used_fallback = true;
          break;
        }
      }
    }
    if (used_fallback) {
      ++nonstar_fallback_cells;
    }
  }
  budgets.clear();
  budgets.reserve(ring_edges.size());
  for (std::size_t edge_index = 0; edge_index < ring_edges.size();
       ++edge_index) {
    DvclpEdgeBudget edge;
    edge.length = edge_lengths[edge_index];
    edge.node0 = ring_edges[edge_index].first;
    edge.node1 = ring_edges[edge_index].second;
    if (edge_incidences[edge_index].size() >
        edge.incident_corners.size()) {
      char buffer[128];
      std::snprintf(buffer, sizeof(buffer),
                    "dvclp_nonmanifold_edge: n0=%d n1=%d",
                    edge.node0, edge.node1);
      reason = buffer;
      return false;
    }
    edge.incident_count =
        static_cast<int>(edge_incidences[edge_index].size());
    std::copy(edge_incidences[edge_index].begin(),
              edge_incidences[edge_index].end(),
              edge.incident_corners.begin());
    edge.raw_jump = std::hypot(
        target_node_vr[edge.node1] - target_node_vr[edge.node0],
        target_node_vz[edge.node1] - target_node_vz[edge.node0]);
    const double start_r = target_mesh.node_r[edge.node0];
    const double start_z = target_mesh.node_z[edge.node0];
    walk_segment(
        start_r, start_z, target_mesh.node_r[edge.node1] - start_r,
        target_mesh.node_z[edge.node1] - start_z,
        candidate_cells[edge_index], triangles, triangles_by_cell,
        edge.budget, edge.walk_segments, edge.walk_failed);
    budgets.push_back(edge);
  }
  return true;
}

namespace {

double projection_jump(const RemapMesh& target_mesh,
                       const DvclpEdgeBudget& edge,
                       const double* node_vr,
                       const double* node_vz,
                       double& delta_vr,
                       double& delta_vz) {
  const bool axis0 = target_mesh.node_r[edge.node0] == 0.0;
  const bool axis1 = target_mesh.node_r[edge.node1] == 0.0;
  delta_vr = axis0 || axis1
                 ? 0.0
                 : node_vr[edge.node1] - node_vr[edge.node0];
  delta_vz = node_vz[edge.node1] - node_vz[edge.node0];
  return std::hypot(delta_vr, delta_vz);
}

double budget_upper_bound(const DvclpEdgeBudget& edge) {
  // The walk accumulates walk_segments fused-multiply-add-free products and
  // sums, each contributing <= eps relative; the jump norm adds a hypot
  // (<= 2 eps).  The factor 8 is the conservative headroom used by the
  // project's certified predicates.
  return edge.budget *
         (1.0 + 8.0 * (double)(edge.walk_segments + 2) *
                    std::numeric_limits<double>::epsilon());
}

double projection_ratio(const double jump, const double budget) {
  if (budget == 0.0) {
    return jump == 0.0
               ? 0.0
               : std::numeric_limits<double>::infinity();
  }
  return jump / budget;
}

bool bitwise_equal(const double left, const double right) {
  return std::memcmp(&left, &right, sizeof(double)) == 0;
}

double corner_reduced_mass_sum(const DvclpEdgeBudget& edge,
                               const double* corner_mass) {
  double weight_sum = 0.0;
  for (int incidence_index = 0;
       incidence_index < edge.incident_count; ++incidence_index) {
    const DvclpEdgeIncidence& incidence =
        edge.incident_corners[static_cast<std::size_t>(incidence_index)];
    const double corner_mass0 = corner_mass[incidence.corner0];
    const double corner_mass1 = corner_mass[incidence.corner1];
    const double denominator = corner_mass0 + corner_mass1;
    if (denominator > 0.0) {
      weight_sum += corner_mass0 * corner_mass1 / denominator;
    }
  }
  return weight_sum;
}

struct DvclpNodeCorner {
  int cell = -1;
  std::size_t corner = 0;
};

struct DvclpConsensusGraph {
  std::vector<int> parent;
  std::vector<std::vector<int>> members;
  std::vector<int> parent_vr;
  std::vector<std::vector<int>> members_vr;
  std::vector<std::vector<DvclpNodeCorner>> node_corners;
  std::vector<int> component_roots;
};

int consensus_root(std::vector<int>& parent, const int node) {
  int root = node;
  while (parent[static_cast<std::size_t>(root)] != root) {
    root = parent[static_cast<std::size_t>(root)];
  }
  int current = node;
  while (parent[static_cast<std::size_t>(current)] != current) {
    const int next = parent[static_cast<std::size_t>(current)];
    parent[static_cast<std::size_t>(current)] = root;
    current = next;
  }
  return root;
}

DvclpConsensusGraph build_consensus_graph(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const std::vector<char>& consensus_constrained) {
  std::size_t node_count = 0;
  for (int cell = 0; cell < target_mesh.n_cells; ++cell) {
    const int begin = target_mesh.csr_offsets[cell];
    const int nverts = static_cast<int>(target_mesh.nverts[cell]);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = target_mesh.csr_indices[begin + corner];
      node_count = std::max(node_count, static_cast<std::size_t>(node + 1));
    }
  }

  DvclpConsensusGraph graph;
  graph.parent.resize(node_count);
  graph.members.resize(node_count);
  graph.parent_vr.resize(node_count);
  graph.members_vr.resize(node_count);
  graph.node_corners.resize(node_count);
  std::vector<bool> in_consensus(node_count, false);
  for (std::size_t node = 0; node < node_count; ++node) {
    graph.parent[node] = static_cast<int>(node);
    graph.parent_vr[node] = static_cast<int>(node);
  }
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (!consensus_constrained[edge_index]) {
      continue;
    }
    in_consensus[static_cast<std::size_t>(edge.node0)] = true;
    in_consensus[static_cast<std::size_t>(edge.node1)] = true;
    const int root0 = consensus_root(graph.parent, edge.node0);
    const int root1 = consensus_root(graph.parent, edge.node1);
    if (root0 != root1) {
      const int smaller_root = std::min(root0, root1);
      const int larger_root = std::max(root0, root1);
      graph.parent[static_cast<std::size_t>(larger_root)] = smaller_root;
    }
    if (target_mesh.node_r[edge.node0] != 0.0 &&
        target_mesh.node_r[edge.node1] != 0.0) {
      const int root_vr0 = consensus_root(graph.parent_vr, edge.node0);
      const int root_vr1 = consensus_root(graph.parent_vr, edge.node1);
      if (root_vr0 != root_vr1) {
        const int smaller_root = std::min(root_vr0, root_vr1);
        const int larger_root = std::max(root_vr0, root_vr1);
        graph.parent_vr[static_cast<std::size_t>(larger_root)] =
            smaller_root;
      }
    }
  }
  for (std::size_t node = 0; node < node_count; ++node) {
    if (!in_consensus[node]) {
      continue;
    }
    const int root = consensus_root(graph.parent, static_cast<int>(node));
    graph.members[static_cast<std::size_t>(root)].push_back(
        static_cast<int>(node));
    const int root_vr =
        consensus_root(graph.parent_vr, static_cast<int>(node));
    graph.members_vr[static_cast<std::size_t>(root_vr)].push_back(
        static_cast<int>(node));
  }
  for (std::size_t root = 0; root < node_count; ++root) {
    if (graph.members[root].size() >= 2U) {
      graph.component_roots.push_back(static_cast<int>(root));
    }
  }
  for (int cell = 0; cell < target_mesh.n_cells; ++cell) {
    const int begin = target_mesh.csr_offsets[cell];
    const int nverts = static_cast<int>(target_mesh.nverts[cell]);
    for (int local_corner = 0; local_corner < nverts; ++local_corner) {
      const int node = target_mesh.csr_indices[begin + local_corner];
      const std::size_t corner =
          static_cast<std::size_t>(cell) * target_mesh.corner_stride +
          static_cast<std::size_t>(local_corner);
      graph.node_corners[static_cast<std::size_t>(node)].push_back(
          {cell, corner});
    }
  }
  return graph;
}

bool consensus_edge_satisfied(const RemapMesh& target_mesh,
                              const DvclpEdgeBudget& edge,
                              const double* node_vr,
                              const double* node_vz) {
  const bool axis = target_mesh.node_r[edge.node0] == 0.0 ||
                    target_mesh.node_r[edge.node1] == 0.0;
  return bitwise_equal(node_vz[edge.node0], node_vz[edge.node1]) &&
         (axis || bitwise_equal(node_vr[edge.node0], node_vr[edge.node1]));
}

bool mark_affected_consensus_components(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const std::vector<char>& consensus_constrained,
    const DvclpConsensusGraph& graph,
    const double* node_vr,
    const double* node_vz,
    std::vector<bool>& affected) {
  std::fill(affected.begin(), affected.end(), false);
  bool any_violated = false;
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (!consensus_constrained[edge_index] ||
        consensus_edge_satisfied(target_mesh, edge, node_vr, node_vz)) {
      continue;
    }
    const int root = graph.parent[static_cast<std::size_t>(edge.node0)];
    affected[static_cast<std::size_t>(root)] = true;
    any_violated = true;
  }
  return any_violated;
}

bool apply_consensus_components(
    const RemapMesh& target_mesh,
    const DvclpConsensusGraph& graph,
    const std::vector<bool>& affected,
    const double* nodal_mass,
    const double* corner_mass,
    double* corner_momentum_r,
    double* corner_momentum_z,
    double* node_vr,
    double* node_vz,
    double* cell_q_projection,
    DvclpProjectionResult& result) {
  std::vector<long double> cell_component_mass(
      static_cast<std::size_t>(target_mesh.n_cells), 0.0L);
  std::vector<bool> cell_touched(
      static_cast<std::size_t>(target_mesh.n_cells), false);
  std::vector<int> touched_cells;
  std::vector<double> target_vr(graph.parent.size(), 0.0);
  std::vector<double> target_vz(graph.parent.size(), 0.0);
  for (const int root : graph.component_roots) {
    if (!affected[static_cast<std::size_t>(root)]) {
      continue;
    }
    const std::vector<int>& members =
        graph.members[static_cast<std::size_t>(root)];
    long double mass_sum = 0.0L;
    long double momentum_z = 0.0L;
    long double abs_momentum_z = 0.0L;
    bool vz_unchanged = true;
    bool contains_axis = false;
    for (const int node : members) {
      const long double mass =
          static_cast<long double>(nodal_mass[node]);
      const long double velocity_z =
          static_cast<long double>(node_vz[node]);
      mass_sum += mass;
      momentum_z += mass * velocity_z;
      abs_momentum_z += std::fabs(mass * velocity_z);
      target_vr[static_cast<std::size_t>(node)] = node_vr[node];
      target_vz[static_cast<std::size_t>(node)] = node_vz[node];
      if (!bitwise_equal(node_vz[members[0]], node_vz[node])) {
        vz_unchanged = false;
      }
      if (target_mesh.node_r[node] == 0.0) {
        TENRYU_ASSERT(node_vr[node] == 0.0,
                      "DVCLP axis-node radial velocity must be zero");
        contains_axis = true;
      }
    }
    if (!vz_unchanged) {
      const double vz_target =
          static_cast<double>(momentum_z / mass_sum);
      long double delta_sum_z = 0.0L;
      for (const int node : members) {
        target_vz[static_cast<std::size_t>(node)] = vz_target;
        delta_sum_z +=
            static_cast<long double>(nodal_mass[node]) *
            (static_cast<long double>(vz_target) -
             static_cast<long double>(node_vz[node]));
      }
      const long double imbalance_bound_z =
          static_cast<long double>(members.size()) *
          static_cast<long double>(std::numeric_limits<double>::epsilon()) *
          abs_momentum_z;
      if (std::fabs(delta_sum_z) > imbalance_bound_z) {
        char buffer[256];
        std::snprintf(
            buffer, sizeof(buffer),
            "dvclp_consensus_imbalance: root=%d nodes=%zu "
            "dr=%.21Lg br=%.21Lg dz=%.21Lg bz=%.21Lg",
            root, members.size(), 0.0L, 0.0L, delta_sum_z,
            imbalance_bound_z);
        result.ok = false;
        result.reason = buffer;
        return false;
      }
    }

    std::vector<int> vr_roots;
    for (const int node : members) {
      const int vr_root =
          graph.parent_vr[static_cast<std::size_t>(node)];
      if (graph.members_vr[static_cast<std::size_t>(vr_root)].size() >=
          2U) {
        vr_roots.push_back(vr_root);
      }
    }
    std::sort(vr_roots.begin(), vr_roots.end());
    vr_roots.erase(std::unique(vr_roots.begin(), vr_roots.end()),
                   vr_roots.end());

    bool all_vr_groups_unchanged = true;
    long double delta_sum_r = 0.0L;
    for (const int vr_root : vr_roots) {
      const std::vector<int>& group =
          graph.members_vr[static_cast<std::size_t>(vr_root)];
      bool vr_unchanged = true;
      long double group_mass_sum = 0.0L;
      long double group_momentum_r = 0.0L;
      long double group_abs_momentum_r = 0.0L;
      for (const int node : group) {
        const long double mass =
            static_cast<long double>(nodal_mass[node]);
        const long double velocity_r =
            static_cast<long double>(node_vr[node]);
        group_mass_sum += mass;
        group_momentum_r += mass * velocity_r;
        group_abs_momentum_r += std::fabs(mass * velocity_r);
        if (!bitwise_equal(node_vr[group[0]], node_vr[node])) {
          vr_unchanged = false;
        }
      }
      if (vr_unchanged) {
        continue;
      }
      all_vr_groups_unchanged = false;
      const double vr_target =
          static_cast<double>(group_momentum_r / group_mass_sum);
      long double group_delta_sum_r = 0.0L;
      for (const int node : group) {
        target_vr[static_cast<std::size_t>(node)] = vr_target;
        group_delta_sum_r +=
            static_cast<long double>(nodal_mass[node]) *
            (static_cast<long double>(vr_target) -
             static_cast<long double>(node_vr[node]));
      }
      const long double imbalance_bound_r =
          static_cast<long double>(group.size()) *
          static_cast<long double>(std::numeric_limits<double>::epsilon()) *
          group_abs_momentum_r;
      if (std::fabs(group_delta_sum_r) > imbalance_bound_r) {
        char buffer[256];
        std::snprintf(
            buffer, sizeof(buffer),
            "dvclp_consensus_imbalance: root=%d nodes=%zu "
            "dr=%.21Lg br=%.21Lg dz=%.21Lg bz=%.21Lg",
            vr_root, group.size(), group_delta_sum_r, imbalance_bound_r,
            0.0L, 0.0L);
        result.ok = false;
        result.reason = buffer;
        return false;
      }
      delta_sum_r += group_delta_sum_r;
    }
    // Non-axis telescope rounding remains unledgered within kEps64*scale.
    if (contains_axis) {
      result.axis_projection_r -= delta_sum_r;
    }
    if (vz_unchanged && all_vr_groups_unchanged) {
      continue;
    }

    long double kinetic_before = 0.0L;
    long double kinetic_after = 0.0L;
    for (const int node : members) {
      const long double mass =
          static_cast<long double>(nodal_mass[node]);
      const long double velocity_r =
          static_cast<long double>(node_vr[node]);
      const long double velocity_z =
          static_cast<long double>(node_vz[node]);
      const long double target_velocity_r =
          static_cast<long double>(target_vr[static_cast<std::size_t>(node)]);
      const long double target_velocity_z =
          static_cast<long double>(target_vz[static_cast<std::size_t>(node)]);
      kinetic_before +=
          0.5L * mass *
          (velocity_r * velocity_r + velocity_z * velocity_z);
      kinetic_after +=
          0.5L * mass *
          (target_velocity_r * target_velocity_r +
           target_velocity_z * target_velocity_z);
    }
    const long double component_q = kinetic_before - kinetic_after;
    const long double variance_bound =
        8.0L * static_cast<long double>(members.size() + 2U) *
        static_cast<long double>(std::numeric_limits<double>::epsilon()) *
        (std::fabs(kinetic_before) + std::fabs(kinetic_after));
    if (component_q < -variance_bound) {
      char buffer[224];
      std::snprintf(
          buffer, sizeof(buffer),
          "dvclp_consensus_variance: root=%d nodes=%zu Q=%.21Lg "
          "bound=%.21Lg",
          root, members.size(), component_q, variance_bound);
      result.ok = false;
      result.reason = buffer;
      return false;
    }

    touched_cells.clear();
    long double component_corner_mass = 0.0L;
    for (const int node : members) {
      const long double mass =
          static_cast<long double>(nodal_mass[node]);
      const long double target_velocity_r =
          static_cast<long double>(target_vr[static_cast<std::size_t>(node)]);
      const long double target_velocity_z =
          static_cast<long double>(target_vz[static_cast<std::size_t>(node)]);
      const long double delta_momentum_r =
          mass * (target_velocity_r -
                  static_cast<long double>(node_vr[node]));
      const long double delta_momentum_z =
          mass * (target_velocity_z -
                  static_cast<long double>(node_vz[node]));
      for (const DvclpNodeCorner& incidence :
           graph.node_corners[static_cast<std::size_t>(node)]) {
        const long double incidence_mass =
            static_cast<long double>(corner_mass[incidence.corner]);
        const double fraction =
            corner_mass[incidence.corner] / nodal_mass[node];
        corner_momentum_r[incidence.corner] +=
            fraction * static_cast<double>(delta_momentum_r);
        corner_momentum_z[incidence.corner] +=
            fraction * static_cast<double>(delta_momentum_z);
        if (!cell_touched[static_cast<std::size_t>(incidence.cell)]) {
          cell_touched[static_cast<std::size_t>(incidence.cell)] = true;
          touched_cells.push_back(incidence.cell);
        }
        cell_component_mass[static_cast<std::size_t>(incidence.cell)] +=
            incidence_mass;
        component_corner_mass += incidence_mass;
      }
      node_vr[node] = target_vr[static_cast<std::size_t>(node)];
      node_vz[node] = target_vz[static_cast<std::size_t>(node)];
    }

    // This is the true kinetic-energy change of the assignment; depositing it signed keeps total energy conserved.
    const double component_q_double = static_cast<double>(component_q);
    std::sort(touched_cells.begin(), touched_cells.end());
    for (const int cell : touched_cells) {
      const long double cell_mass =
          cell_component_mass[static_cast<std::size_t>(cell)];
      if (cell_mass != 0.0L) {
        const double fraction =
            static_cast<double>(cell_mass / component_corner_mass);
        cell_q_projection[cell] += fraction * component_q_double;
      }
      cell_component_mass[static_cast<std::size_t>(cell)] = 0.0L;
      cell_touched[static_cast<std::size_t>(cell)] = false;
    }
    result.q_total += component_q_double;
  }
  return true;
}

struct DvclpConeRow {
  std::size_t edge_index = 0;
  double beta = 0.0;
  double scale = 1.0;
  int radial0 = -1;
  int radial1 = -1;
  int axial0 = -1;
  int axial1 = -1;
};

bool solve_dense_symmetric_ldlt(
    std::vector<double> matrix,
    std::vector<double> rhs,
    std::vector<double>& solution,
    std::size_t& zero_pivot) {
  const std::size_t size = rhs.size();
  std::vector<double> lower(size * size, 0.0);
  std::vector<double> diagonal(size, 0.0);
  std::vector<std::size_t> permutation(size, 0);
  for (std::size_t index = 0; index < size; ++index) {
    permutation[index] = index;
  }

  for (std::size_t column = 0; column < size; ++column) {
    std::size_t pivot = column;
    double pivot_magnitude =
        std::fabs(matrix[column * size + column]);
    for (std::size_t candidate = column + 1; candidate < size;
         ++candidate) {
      const double candidate_magnitude =
          std::fabs(matrix[candidate * size + candidate]);
      if (candidate_magnitude > pivot_magnitude ||
          (candidate_magnitude == pivot_magnitude &&
           permutation[candidate] < permutation[pivot])) {
        pivot = candidate;
        pivot_magnitude = candidate_magnitude;
      }
    }
    if (matrix[pivot * size + pivot] == 0.0) {
      zero_pivot = permutation[pivot];
      return false;
    }
    if (pivot != column) {
      for (std::size_t index = 0; index < size; ++index) {
        std::swap(matrix[column * size + index],
                  matrix[pivot * size + index]);
      }
      for (std::size_t index = 0; index < size; ++index) {
        std::swap(matrix[index * size + column],
                  matrix[index * size + pivot]);
      }
      for (std::size_t previous = 0; previous < column; ++previous) {
        std::swap(lower[column * size + previous],
                  lower[pivot * size + previous]);
      }
      std::swap(rhs[column], rhs[pivot]);
      std::swap(permutation[column], permutation[pivot]);
    }

    lower[column * size + column] = 1.0;
    diagonal[column] = matrix[column * size + column];
    for (std::size_t row = column + 1; row < size; ++row) {
      lower[row * size + column] =
          matrix[row * size + column] / diagonal[column];
    }
    for (std::size_t row = column + 1; row < size; ++row) {
      for (std::size_t entry = row; entry < size; ++entry) {
        const double updated =
            matrix[entry * size + row] -
            lower[entry * size + column] * diagonal[column] *
                lower[row * size + column];
        matrix[entry * size + row] = updated;
        matrix[row * size + entry] = updated;
      }
    }
  }

  std::vector<double> forward(size, 0.0);
  std::vector<double> scaled(size, 0.0);
  std::vector<double> permuted_solution(size, 0.0);
  for (std::size_t row = 0; row < size; ++row) {
    forward[row] = rhs[row];
    for (std::size_t column = 0; column < row; ++column) {
      forward[row] -= lower[row * size + column] * forward[column];
    }
    scaled[row] = forward[row] / diagonal[row];
  }
  for (std::size_t reverse = size; reverse-- > 0;) {
    permuted_solution[reverse] = scaled[reverse];
    for (std::size_t row = reverse + 1; row < size; ++row) {
      permuted_solution[reverse] -=
          lower[row * size + reverse] * permuted_solution[row];
    }
  }
  solution.assign(size, 0.0);
  for (std::size_t index = 0; index < size; ++index) {
    solution[permutation[index]] = permuted_solution[index];
  }
  return true;
}

}  // namespace

// D3 ladder stage (consult-33 §4.4 (73)-(79), Euclidean edge-ball form):
// active-set KKT certificate for a Dykstra-feasible component.  Constructs a
// dual witness on the active subgraph (leaf elimination over a stable spanning
// forest, r and z flows independent; cycle-closing and inactive edges carry
// lambda = 0) and verifies stationarity and the normal-cone condition under
// derived compensated-summation envelopes.  The velocities are never modified:
// Dykstra's paired mass-metric updates conserve component momentum to
// rounding, so the equality-KKT primal correction (the translation null-mode)
// is sub-roundoff by construction; the certificate upgrades the label or
// leaves the feasible result untouched.
DvclpKktCertificate certify_dvclp_kkt(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const std::vector<std::size_t>& component_edges,
    const std::vector<int>& members,
    const std::vector<bool>& node_z_in_component,
    const std::vector<bool>& node_r_in_component,
    const double* nodal_mass,
    const std::vector<double>& candidate_vr,
    const std::vector<double>& candidate_vz,
    const std::vector<double>& snapshot_vr,
    const std::vector<double>& snapshot_vz) {
  const double eps = std::numeric_limits<double>::epsilon();
  const std::size_t n = members.size();
  const std::size_t edge_count = component_edges.size();
  const auto node_slot = [&](const int node) {
    return static_cast<std::size_t>(
        std::lower_bound(members.begin(), members.end(), node) -
        members.begin());
  };
  const auto is_axis_edge = [&](const DvclpEdgeBudget& edge) {
    return target_mesh.node_r[edge.node0] == 0.0 ||
           target_mesh.node_r[edge.node1] == 0.0;
  };

  std::vector<char> active(edge_count, false);
  std::vector<double> dvr(edge_count, 0.0);
  std::vector<double> dvz(edge_count, 0.0);
  std::vector<double> lambda_r(edge_count, 0.0);
  std::vector<double> lambda_z(edge_count, 0.0);
  std::size_t active_edges = 0;
  for (std::size_t k = 0; k < edge_count; ++k) {
    const DvclpEdgeBudget& edge = budgets[component_edges[k]];
    const double jump = projection_jump(
        target_mesh, edge, candidate_vr.data(), candidate_vz.data(),
        dvr[k], dvz[k]);
    if (edge.budget == 0.0) {
      if (dvr[k] != 0.0 || dvz[k] != 0.0) {
        return {false, 1, 0.0, active_edges};
      }
      active[k] = true;
    } else {
      active[k] =
          jump >= edge.budget *
                      (1.0 - 8.0 * (double)(edge.walk_segments + 2) * eps);
    }
    if (active[k]) {
      ++active_edges;
    }
  }

  const auto construct_flow = [&](const bool radial) {
    std::vector<int> parent(n, 0);
    for (std::size_t slot = 0; slot < n; ++slot) {
      parent[slot] = static_cast<int>(slot);
    }
    const auto find_root = [&](int slot) {
      while (parent[static_cast<std::size_t>(slot)] != slot) {
        parent[static_cast<std::size_t>(slot)] =
            parent[static_cast<std::size_t>(
                parent[static_cast<std::size_t>(slot)])];
        slot = parent[static_cast<std::size_t>(slot)];
      }
      return slot;
    };

    std::vector<char> tree_edge(edge_count, false);
    for (std::size_t k = 0; k < edge_count; ++k) {
      const DvclpEdgeBudget& edge = budgets[component_edges[k]];
      if (!active[k] || (radial && is_axis_edge(edge))) {
        continue;
      }
      const std::size_t slot0 = node_slot(edge.node0);
      const std::size_t slot1 = node_slot(edge.node1);
      const int root0 = find_root(static_cast<int>(slot0));
      const int root1 = find_root(static_cast<int>(slot1));
      if (root0 == root1) {
        continue;
      }
      const int smaller_root = std::min(root0, root1);
      const int larger_root = std::max(root0, root1);
      parent[static_cast<std::size_t>(larger_root)] = smaller_root;
      tree_edge[k] = true;
    }

    const std::vector<double>& candidate =
        radial ? candidate_vr : candidate_vz;
    const std::vector<double>& snapshot =
        radial ? snapshot_vr : snapshot_vz;
    std::vector<double>& lambda = radial ? lambda_r : lambda_z;
    std::vector<double> g(n, 0.0);
    for (std::size_t slot = 0; slot < n; ++slot) {
      const int node = members[slot];
      const bool participating =
          !radial ||
          node_r_in_component[static_cast<std::size_t>(node)];
      if (participating) {
        g[slot] = nodal_mass[node] *
                  (candidate[static_cast<std::size_t>(node)] -
                   snapshot[slot]);
      }
    }

    std::vector<std::vector<std::size_t>> adjacency(n);
    std::vector<std::size_t> degree(n, 0);
    for (std::size_t k = 0; k < edge_count; ++k) {
      if (!tree_edge[k]) {
        continue;
      }
      const DvclpEdgeBudget& edge = budgets[component_edges[k]];
      const std::size_t slot0 = node_slot(edge.node0);
      const std::size_t slot1 = node_slot(edge.node1);
      adjacency[slot0].push_back(k);
      adjacency[slot1].push_back(k);
    }
    for (std::size_t slot = 0; slot < n; ++slot) {
      degree[slot] = adjacency[slot].size();
    }

    std::vector<char> processed(edge_count, false);
    std::vector<std::size_t> queue;
    for (std::size_t slot = 0; slot < n; ++slot) {
      const int node = members[slot];
      const bool participating =
          !radial ||
          node_r_in_component[static_cast<std::size_t>(node)];
      if (participating && degree[slot] == 1) {
        queue.push_back(slot);
      }
    }
    std::size_t queue_head = 0;
    while (queue_head < queue.size()) {
      const std::size_t i = queue[queue_head++];
      if (degree[i] != 1) {
        continue;
      }
      std::size_t k = edge_count;
      for (const std::size_t incident : adjacency[i]) {
        if (!processed[incident]) {
          k = incident;
          break;
        }
      }
      const DvclpEdgeBudget& edge = budgets[component_edges[k]];
      const double sigma_i = members[i] == edge.node1 ? 1.0 : -1.0;
      lambda[k] = -sigma_i * g[i];
      const int other_node =
          members[i] == edge.node0 ? edge.node1 : edge.node0;
      const std::size_t j = node_slot(other_node);
      const double sigma_j = -sigma_i;
      g[j] += sigma_j * lambda[k];
      processed[k] = true;
      --degree[i];
      --degree[j];
      if (degree[j] == 1) {
        queue.push_back(j);
      }
    }
  };

  construct_flow(false);
  construct_flow(true);

  double lam_max = 0.0;
  for (std::size_t k = 0; k < edge_count; ++k) {
    if (active[k]) {
      lam_max =
          std::max(lam_max, std::hypot(lambda_r[k], lambda_z[k]));
    }
  }

  std::vector<std::vector<std::size_t>> adjacency_z(n);
  std::vector<std::vector<std::size_t>> adjacency_r(n);
  for (std::size_t k = 0; k < edge_count; ++k) {
    if (!active[k]) {
      continue;
    }
    const DvclpEdgeBudget& edge = budgets[component_edges[k]];
    const std::size_t slot0 = node_slot(edge.node0);
    const std::size_t slot1 = node_slot(edge.node1);
    adjacency_z[slot0].push_back(k);
    if (slot1 != slot0) {
      adjacency_z[slot1].push_back(k);
    }
    if (!is_axis_edge(edge)) {
      adjacency_r[slot0].push_back(k);
      if (slot1 != slot0) {
        adjacency_r[slot1].push_back(k);
      }
    }
  }

  const auto stationarity_holds = [&](const std::size_t slot,
                                       const bool radial) {
    const int node = members[slot];
    const std::vector<double>& candidate =
        radial ? candidate_vr : candidate_vz;
    const std::vector<double>& snapshot =
        radial ? snapshot_vr : snapshot_vz;
    const std::vector<double>& lambda = radial ? lambda_r : lambda_z;
    const std::vector<std::size_t>& adjacency =
        radial ? adjacency_r[slot] : adjacency_z[slot];
    double sum = 0.0;
    double comp = 0.0;
    double a = 0.0;
    const auto add_term = [&](const double t) {
      const double next = sum + t;
      if (std::fabs(sum) >= std::fabs(t)) {
        comp += (sum - next) + t;
      } else {
        comp += (t - next) + sum;
      }
      sum = next;
      a += std::fabs(t);
    };
    const double t0 =
        nodal_mass[node] *
        (candidate[static_cast<std::size_t>(node)] - snapshot[slot]);
    add_term(t0);
    for (const std::size_t k : adjacency) {
      const DvclpEdgeBudget& edge = budgets[component_edges[k]];
      const double sigma_node = members[slot] == edge.node1 ? 1.0 : -1.0;
      const double t = sigma_node * lambda[k];
      add_term(t);
    }
    const double s = sum + comp;
    const std::size_t deg = adjacency.size();
    const double envelope = 8.0 * (double)(deg + 2) * eps * a;
    return std::fabs(s) <= envelope;
  };

  for (std::size_t slot = 0; slot < n; ++slot) {
    const int node = members[slot];
    if (node_z_in_component[static_cast<std::size_t>(node)] &&
        !stationarity_holds(slot, false)) {
      return {false, 2, lam_max, active_edges};
    }
    if (node_r_in_component[static_cast<std::size_t>(node)] &&
        !stationarity_holds(slot, true)) {
      return {false, 2, lam_max, active_edges};
    }
  }

  for (std::size_t k = 0; k < edge_count; ++k) {
    const DvclpEdgeBudget& edge = budgets[component_edges[k]];
    if (!active[k] || edge.budget == 0.0) {
      continue;
    }
    const double lr = lambda_r[k];
    const double lz = lambda_z[k];
    if (lr == 0.0 && lz == 0.0) {
      continue;
    }
    const double dot = lr * dvr[k] + lz * dvz[k];
    const double adot =
        std::fabs(lr * dvr[k]) + std::fabs(lz * dvz[k]);
    const double cross = lr * dvz[k] - lz * dvr[k];
    const double across =
        std::fabs(lr * dvz[k]) + std::fabs(lz * dvr[k]);
    if (!(dot >= -16.0 * eps * adot &&
          std::fabs(cross) <= 16.0 * eps * across)) {
      return {false, 3, lam_max, active_edges};
    }
  }

  return {true, 0, lam_max, active_edges};
}

namespace {

bool apply_homothetic_completion(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const std::vector<char>& consensus_constrained,
    const DvclpConsensusGraph& graph,
    const std::vector<int>& unit_z,
    const std::vector<int>& unit_r,
    const std::vector<double>& unit_mass_z,
    const std::vector<double>& unit_mass_r,
    const double* nodal_mass,
    const double* corner_mass,
    double* corner_momentum_r,
    double* corner_momentum_z,
    double* node_vr,
    double* node_vz,
    double* cell_q_projection,
    std::vector<char>& processed_inner_nodes,
    std::vector<char>& screened_omitted,
    DvclpProjectionResult& result,
    const DvclpTestCaps& test_caps,
    const int solver_rev) {
  const std::chrono::steady_clock::time_point tp_entry =
      std::chrono::steady_clock::now();
  static const bool solve_split_enabled = [] {
    const char* const value = std::getenv("TENRYU_TESS_TIMING");
    return value != nullptr && std::string(value) == std::string("1");
  }();
  const std::size_t node_count = unit_z.size();
  const double eps = std::numeric_limits<double>::epsilon();
  processed_inner_nodes.assign(node_count, false);
  screened_omitted.assign(budgets.size(), false);
  std::vector<int> component_parent(node_count, 0);
  for (std::size_t unit = 0; unit < node_count; ++unit) {
    component_parent[unit] = static_cast<int>(unit);
  }
  const auto unite_units = [&](const int left, const int right) {
    const int left_root = consensus_root(component_parent, left);
    const int right_root = consensus_root(component_parent, right);
    if (left_root == right_root) {
      return;
    }
    const int smaller_root = std::min(left_root, right_root);
    const int larger_root = std::max(left_root, right_root);
    component_parent[static_cast<std::size_t>(larger_root)] = smaller_root;
  };

  // Full connectivity defines the outer conserved families and their feasible
  // consensus reference.  Screening below creates the inner components; the
  // homothety_expansions field is retained at zero for schema stability.
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (edge.walk_failed || consensus_constrained[edge_index]) {
      continue;
    }
    const int unit_z0 = unit_z[static_cast<std::size_t>(edge.node0)];
    const int unit_z1 = unit_z[static_cast<std::size_t>(edge.node1)];
    const int unit_r0 = unit_r[static_cast<std::size_t>(edge.node0)];
    const int unit_r1 = unit_r[static_cast<std::size_t>(edge.node1)];
    unite_units(unit_z0, unit_z1);
    unite_units(unit_r0, unit_r1);
    unite_units(unit_z0, unit_r0);
  }
  const std::chrono::steady_clock::time_point tp_union =
      std::chrono::steady_clock::now();

  std::vector<char> violating(budgets.size(), false);
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (edge.walk_failed || consensus_constrained[edge_index]) {
      continue;
    }
    double delta_vr = 0.0;
    double delta_vz = 0.0;
    const double jump = projection_jump(
        target_mesh, edge, node_vr, node_vz, delta_vr, delta_vz);
    violating[edge_index] = jump > budget_upper_bound(edge);
  }
  const std::chrono::steady_clock::time_point tp_violation =
      std::chrono::steady_clock::now();

  std::vector<bool> affected_components(node_count, false);
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (!violating[edge_index]) {
      continue;
    }
    const int root = consensus_root(
        component_parent,
        unit_z[static_cast<std::size_t>(edge.node0)]);
    affected_components[static_cast<std::size_t>(root)] = true;
  }
  const std::chrono::steady_clock::time_point tp_affected =
      std::chrono::steady_clock::now();

  std::vector<bool> node_in_component(node_count, false);
  std::vector<bool> node_z_in_component(node_count, false);
  std::vector<bool> node_r_in_component(node_count, false);
  std::vector<int> outer_members;
  std::vector<int> members;
  std::vector<std::size_t> component_kkt_edges;
  std::vector<int> z_units;
  std::vector<int> r_units;
  std::vector<double> rho_z(node_count, 0.0);
  std::vector<double> rho_r(node_count, 0.0);
  std::vector<char> retained(budgets.size(), false);
  std::vector<int> dof_z(node_count, -1);
  std::vector<int> dof_r(node_count, -1);
  std::vector<double> candidate_vr(node_vr, node_vr + node_count);
  std::vector<double> candidate_vz(node_vz, node_vz + node_count);
  std::vector<double> target_vr(node_count, 0.0);
  std::vector<double> target_vz(node_count, 0.0);
  std::vector<long double> cell_component_mass(
      static_cast<std::size_t>(target_mesh.n_cells), 0.0L);
  std::vector<bool> cell_touched(
      static_cast<std::size_t>(target_mesh.n_cells), false);
  std::vector<int> touched_cells;
  std::size_t split_components = 0;
  std::size_t split_members_total = 0;
  for (std::size_t outer_root_index = 0; outer_root_index < node_count;
       ++outer_root_index) {
    if (consensus_root(component_parent,
                       static_cast<int>(outer_root_index)) !=
            static_cast<int>(outer_root_index) ||
        !affected_components[outer_root_index]) {
      continue;
    }
    ++split_components;
    const int outer_component_root = static_cast<int>(outer_root_index);
    bool outer_contains_axis = false;
    outer_members.clear();
    std::fill(node_in_component.begin(), node_in_component.end(), false);
    for (std::size_t node = 0; node < node_count; ++node) {
      const int z_root = consensus_root(component_parent, unit_z[node]);
      const int r_root = consensus_root(component_parent, unit_r[node]);
      if (z_root == outer_component_root ||
          r_root == outer_component_root) {
        node_in_component[node] = true;
        outer_members.push_back(static_cast<int>(node));
        outer_contains_axis =
            outer_contains_axis || target_mesh.node_r[node] == 0.0;
      }
    }
    split_members_total += outer_members.size();

    long double outer_mass_sum = 0.0L;
    long double outer_momentum_r = 0.0L;
    long double outer_momentum_z = 0.0L;
    for (const int node : outer_members) {
      const long double mass = static_cast<long double>(nodal_mass[node]);
      outer_mass_sum += mass;
      outer_momentum_r +=
          mass * static_cast<long double>(node_vr[node]);
      outer_momentum_z +=
          mass * static_cast<long double>(node_vz[node]);
    }
    const double outer_mean_r =
        outer_contains_axis
            ? 0.0
            : static_cast<double>(outer_momentum_r / outer_mass_sum);
    const double outer_mean_z =
        static_cast<double>(outer_momentum_z / outer_mass_sum);
    long double centered_energy_sum = 0.0L;
    for (const int node : outer_members) {
      const long double mass = static_cast<long double>(nodal_mass[node]);
      const long double delta_r =
          static_cast<long double>(node_vr[node]) - outer_mean_r;
      const long double delta_z =
          static_cast<long double>(node_vz[node]) - outer_mean_z;
      centered_energy_sum +=
          0.5L * mass * (delta_r * delta_r + delta_z * delta_z);
    }
    const double centered_energy = static_cast<double>(centered_energy_sum);

    z_units.clear();
    r_units.clear();
    for (const int node : outer_members) {
      const std::size_t node_index = static_cast<std::size_t>(node);
      if (consensus_root(component_parent, unit_z[node_index]) ==
          outer_component_root) {
        z_units.push_back(unit_z[node_index]);
      }
      if (consensus_root(component_parent, unit_r[node_index]) ==
          outer_component_root) {
        r_units.push_back(unit_r[node_index]);
      }
    }
    std::sort(z_units.begin(), z_units.end());
    z_units.erase(std::unique(z_units.begin(), z_units.end()), z_units.end());
    std::sort(r_units.begin(), r_units.end());
    r_units.erase(std::unique(r_units.begin(), r_units.end()), r_units.end());
    for (const int unit : z_units) {
      rho_z[static_cast<std::size_t>(unit)] =
          std::sqrt(2.0 * centered_energy /
                    unit_mass_z[static_cast<std::size_t>(unit)]);
    }
    for (const int unit : r_units) {
      rho_r[static_cast<std::size_t>(unit)] =
          std::sqrt(2.0 * centered_energy /
                    unit_mass_r[static_cast<std::size_t>(unit)]);
    }

    for (std::size_t edge_index = 0; edge_index < budgets.size();
         ++edge_index) {
      const DvclpEdgeBudget& edge = budgets[edge_index];
      if (edge.walk_failed || consensus_constrained[edge_index] ||
          consensus_root(
              component_parent,
              unit_z[static_cast<std::size_t>(edge.node0)]) !=
              outer_component_root) {
        continue;
      }
      if (violating[edge_index]) {
        retained[edge_index] = true;
        continue;
      }
      double delta_vr = 0.0;
      double delta_vz = 0.0;
      const double jump = projection_jump(
          target_mesh, edge, node_vr, node_vz, delta_vr, delta_vz);
      const double slack = budget_upper_bound(edge) - jump;
      const int z0 = unit_z[static_cast<std::size_t>(edge.node0)];
      const int z1 = unit_z[static_cast<std::size_t>(edge.node1)];
      double movement_bound =
          rho_z[static_cast<std::size_t>(z0)] +
          rho_z[static_cast<std::size_t>(z1)];
      const bool axis = target_mesh.node_r[edge.node0] == 0.0 ||
                        target_mesh.node_r[edge.node1] == 0.0;
      if (!axis) {
        const int r0 = unit_r[static_cast<std::size_t>(edge.node0)];
        const int r1 = unit_r[static_cast<std::size_t>(edge.node1)];
        // The sum over both directions bounds the possible jump change.
        movement_bound += rho_r[static_cast<std::size_t>(r0)] +
                          rho_r[static_cast<std::size_t>(r1)];
      }
      retained[edge_index] =
          !(slack * (1.0 - 4.0 * eps) >
            movement_bound * (1.0 + 4.0 * eps));
      screened_omitted[edge_index] = !retained[edge_index];
    }
    if (dvclp_component_diagnostic_enabled()) {
      std::size_t outer_retained = 0;
      std::size_t outer_omitted = 0;
      std::size_t outer_violating = 0;
      for (std::size_t edge_index = 0; edge_index < budgets.size();
           ++edge_index) {
        const DvclpEdgeBudget& edge = budgets[edge_index];
        if (edge.walk_failed || consensus_constrained[edge_index] ||
            consensus_root(
                component_parent,
                unit_z[static_cast<std::size_t>(edge.node0)]) !=
                outer_component_root) {
          continue;
        }
        if (violating[edge_index]) {
          ++outer_violating;
        }
        if (retained[edge_index]) {
          ++outer_retained;
        } else if (screened_omitted[edge_index]) {
          ++outer_omitted;
        }
      }
      std::fprintf(stderr,
                   "[dvclp_outer] root=%d members=%zu U=%.9e "
                   "retained=%zu omitted=%zu violating=%zu\n",
                   outer_component_root, outer_members.size(),
                   centered_energy, outer_retained, outer_omitted,
                   outer_violating);
    }
  }
  const std::chrono::steady_clock::time_point tp_components =
      std::chrono::steady_clock::now();

  std::vector<int> inner_parent(node_count, 0);
  for (std::size_t unit = 0; unit < node_count; ++unit) {
    inner_parent[unit] = static_cast<int>(unit);
  }
  const auto unite_inner_units = [&](const int left, const int right) {
    const int left_root = consensus_root(inner_parent, left);
    const int right_root = consensus_root(inner_parent, right);
    if (left_root == right_root) {
      return;
    }
    const int smaller_root = std::min(left_root, right_root);
    const int larger_root = std::max(left_root, right_root);
    inner_parent[static_cast<std::size_t>(larger_root)] = smaller_root;
  };
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    if (!retained[edge_index]) {
      continue;
    }
    const DvclpEdgeBudget& edge = budgets[edge_index];
    const int z0 = unit_z[static_cast<std::size_t>(edge.node0)];
    const int z1 = unit_z[static_cast<std::size_t>(edge.node1)];
    const int r0 = unit_r[static_cast<std::size_t>(edge.node0)];
    const int r1 = unit_r[static_cast<std::size_t>(edge.node1)];
    unite_inner_units(z0, z1);
    unite_inner_units(r0, r1);
    unite_inner_units(z0, r0);
  }

  std::vector<bool> inner_components(node_count, false);
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    if (!retained[edge_index]) {
      continue;
    }
    const DvclpEdgeBudget& edge = budgets[edge_index];
    const int root = consensus_root(
        inner_parent, unit_z[static_cast<std::size_t>(edge.node0)]);
    inner_components[static_cast<std::size_t>(root)] = true;
  }
  // Units incident to no retained edge are frozen because their constraints
  // stay inactive and the objective pins them at the stored value.
  for (std::size_t root_index = 0; root_index < node_count; ++root_index) {
    if (consensus_root(inner_parent, static_cast<int>(root_index)) !=
            static_cast<int>(root_index) ||
        !inner_components[root_index]) {
      continue;
    }
    const int component_root = static_cast<int>(root_index);
    long double momentum_r = 0.0L;
    long double momentum_z = 0.0L;
    long double abs_momentum_r = 0.0L;
    long double abs_momentum_z = 0.0L;
    bool contains_axis = false;
    std::fill(node_in_component.begin(), node_in_component.end(), false);
    std::fill(node_z_in_component.begin(), node_z_in_component.end(),
              false);
    std::fill(node_r_in_component.begin(), node_r_in_component.end(),
              false);
    for (std::size_t node = 0; node < node_count; ++node) {
      const int z_root = consensus_root(inner_parent, unit_z[node]);
      const int r_root = consensus_root(inner_parent, unit_r[node]);
      node_z_in_component[node] = z_root == component_root;
      node_r_in_component[node] =
          target_mesh.node_r[node] != 0.0 && r_root == component_root;
      if (node_z_in_component[node] || node_r_in_component[node]) {
        node_in_component[node] = true;
      }
    }
    // Diagnostic-only observation state for the [dvclp_comp] line; never
    // read by the solver.
    std::size_t diag_in_edges = 0;
    std::size_t diag_rows = 0;
    std::size_t diag_dup_rows = 0;
    int diag_dedup = 0;
    int diag_qr_drops = 0;
    std::size_t diag_active0 = 0;
    int diag_newton_iters = 0;
    int diag_newton_exit = -2;
    int diag_ls_bitwise = 0;
    int diag_ldlt_drops = 0;
    int diag_ls_accepts = 0;
    double diag_ratio0 = 0.0;
    double diag_ratio_end = 0.0;
    std::size_t diag_nzero = 0;
    const bool diag_on = dvclp_component_diagnostic_enabled();
    members.clear();
    z_units.clear();
    r_units.clear();
    for (std::size_t node = 0; node < node_count; ++node) {
      if (!node_in_component[node]) {
        continue;
      }
      processed_inner_nodes[node] = true;
      members.push_back(static_cast<int>(node));
      if (node_z_in_component[node]) {
        z_units.push_back(unit_z[node]);
      }
      if (target_mesh.node_r[node] == 0.0) {
        TENRYU_ASSERT(node_vr[node] == 0.0,
                      "DVCLP axis-node radial velocity must be zero");
        contains_axis = true;
      } else if (node_r_in_component[node]) {
        r_units.push_back(unit_r[node]);
      }
    }
    std::sort(z_units.begin(), z_units.end());
    z_units.erase(std::unique(z_units.begin(), z_units.end()), z_units.end());
    std::sort(r_units.begin(), r_units.end());
    r_units.erase(std::unique(r_units.begin(), r_units.end()), r_units.end());

    const auto edge_inside_component = [&](const DvclpEdgeBudget& edge) {
      const std::size_t node0 = static_cast<std::size_t>(edge.node0);
      const std::size_t node1 = static_cast<std::size_t>(edge.node1);
      if (!node_z_in_component[node0] ||
          !node_z_in_component[node1]) {
        return false;
      }
      const bool axis = target_mesh.node_r[edge.node0] == 0.0 ||
                        target_mesh.node_r[edge.node1] == 0.0;
      return axis ||
             (node_r_in_component[node0] &&
              node_r_in_component[node1]);
    };

    long double node_mass_sum_r = 0.0L;
    long double node_mass_sum_z = 0.0L;
    momentum_r = 0.0L;
    momentum_z = 0.0L;
    abs_momentum_r = 0.0L;
    abs_momentum_z = 0.0L;
    for (const int node : members) {
      const long double mass = static_cast<long double>(nodal_mass[node]);
      const long double velocity_r = static_cast<long double>(node_vr[node]);
      const long double velocity_z = static_cast<long double>(node_vz[node]);
      if (node_r_in_component[static_cast<std::size_t>(node)]) {
        node_mass_sum_r += mass;
        momentum_r += mass * velocity_r;
        abs_momentum_r += std::fabs(mass * velocity_r);
      }
      if (node_z_in_component[static_cast<std::size_t>(node)]) {
        node_mass_sum_z += mass;
        momentum_z += mass * velocity_z;
        abs_momentum_z += std::fabs(mass * velocity_z);
      }
    }

    if (diag_on) {
      for (std::size_t edge_index = 0; edge_index < budgets.size();
           ++edge_index) {
        const DvclpEdgeBudget& edge = budgets[edge_index];
        if (edge.walk_failed || !edge_inside_component(edge)) {
          continue;
        }
        ++diag_in_edges;
        double delta_vr = 0.0;
        double delta_vz = 0.0;
        const double jump = projection_jump(
            target_mesh, edge, node_vr, node_vz, delta_vr, delta_vz);
        const double bound = budget_upper_bound(edge);
        if (bound > 0.0) {
          diag_ratio0 = std::max(diag_ratio0, jump / bound);
        }
      }
    }
    const std::size_t dof_count = z_units.size() + r_units.size();
    // Oversized early-time stencil-noise kick bands use intended alpha clipping; this certificate-carried structural cap leaves exact projection as Phase-3 debt.
    const bool component_newton_capped =
        dof_count > test_caps.dof_cap;
    int solver_class = 0;
    int dykstra_sweeps = 0;
    int quotient_clusters = 0;
    std::size_t quotient_cut_edges = 0;
    int quotient_sweeps = 0;
    bool dykstra_feasible = false;
    std::vector<double> snapshot_vr;
    std::vector<double> snapshot_vz;
    if (component_newton_capped) {
      ++result.newton_capped;
      solver_class = 2;
      snapshot_vr.reserve(members.size());
      snapshot_vz.reserve(members.size());
      for (const int node : members) {
        snapshot_vr.push_back(candidate_vr[static_cast<std::size_t>(node)]);
        snapshot_vz.push_back(candidate_vz[static_cast<std::size_t>(node)]);
      }

      std::vector<double> correction_r(budgets.size(), 0.0);
      std::vector<double> correction_z(budgets.size(), 0.0);
      for (int sweep = 0; sweep < test_caps.sweep_cap; ++sweep) {
        for (std::size_t edge_index = 0; edge_index < budgets.size();
             ++edge_index) {
          const DvclpEdgeBudget& edge = budgets[edge_index];
          if (edge.walk_failed || !edge_inside_component(edge)) {
            continue;
          }
          const int node0 = edge.node0;
          const int node1 = edge.node1;
          const double mass0 = nodal_mass[node0];
          const double mass1 = nodal_mass[node1];
          const double mass_sum = mass0 + mass1;
          const double weight0 = mass0 / mass_sum;
          const double weight1 = mass1 / mass_sum;

          // projection_jump orients d as node1-node0.  Re-add the
          // endpoint-supported Dykstra correction p = y - P(y).
          candidate_vr[static_cast<std::size_t>(node0)] -=
              weight1 * correction_r[edge_index];
          candidate_vz[static_cast<std::size_t>(node0)] -=
              weight1 * correction_z[edge_index];
          candidate_vr[static_cast<std::size_t>(node1)] +=
              weight0 * correction_r[edge_index];
          candidate_vz[static_cast<std::size_t>(node1)] +=
              weight0 * correction_z[edge_index];

          double delta_vr = 0.0;
          double delta_vz = 0.0;
          const double jump = projection_jump(
              target_mesh, edge, candidate_vr.data(), candidate_vz.data(),
              delta_vr, delta_vz);
          const double bound = budget_upper_bound(edge);
          if (!(jump > bound)) {
            correction_r[edge_index] = 0.0;
            correction_z[edge_index] = 0.0;
            continue;
          }

          const double gamma = 1.0 - bound / jump;
          correction_r[edge_index] = gamma * delta_vr;
          correction_z[edge_index] = gamma * delta_vz;
          candidate_vr[static_cast<std::size_t>(node0)] +=
              weight1 * correction_r[edge_index];
          candidate_vz[static_cast<std::size_t>(node0)] +=
              weight1 * correction_z[edge_index];
          candidate_vr[static_cast<std::size_t>(node1)] -=
              weight0 * correction_r[edge_index];
          candidate_vz[static_cast<std::size_t>(node1)] -=
              weight0 * correction_z[edge_index];
        }

        dykstra_sweeps = sweep + 1;
        dykstra_feasible = true;
        for (std::size_t edge_index = 0; edge_index < budgets.size();
             ++edge_index) {
          const DvclpEdgeBudget& edge = budgets[edge_index];
          if (edge.walk_failed || !edge_inside_component(edge)) {
            continue;
          }
          double delta_vr = 0.0;
          double delta_vz = 0.0;
          const double jump = projection_jump(
              target_mesh, edge, candidate_vr.data(), candidate_vz.data(),
              delta_vr, delta_vz);
          if (jump > budget_upper_bound(edge)) {
            dykstra_feasible = false;
            break;
          }
        }
        if (dykstra_feasible) {
          solver_class = 1;
          break;
        }
      }
      if (!dykstra_feasible) {
        for (std::size_t member_index = 0;
             member_index < members.size(); ++member_index) {
          const std::size_t node =
              static_cast<std::size_t>(members[member_index]);
          candidate_vr[node] = snapshot_vr[member_index];
          candidate_vz[node] = snapshot_vz[member_index];
        }

        std::vector<int> translation_parent(node_count, 0);
        for (std::size_t node = 0; node < node_count; ++node) {
          translation_parent[node] = static_cast<int>(node);
        }
        const auto unite_translation_nodes =
            [&](const int left, const int right) {
              const int left_root =
                  consensus_root(translation_parent, left);
              const int right_root =
                  consensus_root(translation_parent, right);
              if (left_root == right_root) {
                return;
              }
              const int smaller_root = std::min(left_root, right_root);
              const int larger_root = std::max(left_root, right_root);
              translation_parent[static_cast<std::size_t>(larger_root)] =
                  smaller_root;
            };
        for (std::size_t edge_index = 0; edge_index < budgets.size();
             ++edge_index) {
          const DvclpEdgeBudget& edge = budgets[edge_index];
          if (edge.walk_failed || !edge_inside_component(edge) ||
              violating[edge_index]) {
            continue;
          }
          unite_translation_nodes(edge.node0, edge.node1);
        }

        std::vector<int> root_to_cluster(node_count, -1);
        std::vector<int> node_cluster(node_count, -1);
        std::vector<std::vector<int>> cluster_members;
        for (const int node : members) {
          const int root = consensus_root(translation_parent, node);
          int& cluster = root_to_cluster[static_cast<std::size_t>(root)];
          if (cluster < 0) {
            cluster = static_cast<int>(cluster_members.size());
            cluster_members.emplace_back();
          }
          node_cluster[static_cast<std::size_t>(node)] = cluster;
          cluster_members[static_cast<std::size_t>(cluster)].push_back(node);
        }
        quotient_clusters = static_cast<int>(cluster_members.size());

        std::vector<double> cluster_mass(cluster_members.size(), 0.0);
        std::vector<double> translation_r(cluster_members.size(), 0.0);
        std::vector<double> translation_z(cluster_members.size(), 0.0);
        std::vector<char> radial_frozen(cluster_members.size(), false);
        for (std::size_t cluster = 0; cluster < cluster_members.size();
             ++cluster) {
          for (const int node : cluster_members[cluster]) {
            cluster_mass[cluster] += nodal_mass[node];
            radial_frozen[cluster] =
                radial_frozen[cluster] ||
                target_mesh.node_r[node] == 0.0;
          }
        }

        struct DvclpTranslationCut {
          int cluster_k = -1;
          int cluster_l = -1;
          double delta_vr = 0.0;
          double delta_vz = 0.0;
          double bound = 0.0;
        };
        std::vector<DvclpTranslationCut> cut_edges;
        for (std::size_t edge_index = 0; edge_index < budgets.size();
             ++edge_index) {
          const DvclpEdgeBudget& edge = budgets[edge_index];
          if (edge.walk_failed || !edge_inside_component(edge) ||
              !violating[edge_index]) {
            continue;
          }
          double delta_vr = 0.0;
          double delta_vz = 0.0;
          projection_jump(target_mesh, edge, candidate_vr.data(),
                          candidate_vz.data(), delta_vr, delta_vz);
          // k is the node1 cluster so that projection_jump's node1-node0
          // orientation is exactly (tau_k-tau_l)+d0.
          cut_edges.push_back(
              {node_cluster[static_cast<std::size_t>(edge.node1)],
               node_cluster[static_cast<std::size_t>(edge.node0)],
               delta_vr, delta_vz, budget_upper_bound(edge)});
        }
        quotient_cut_edges = cut_edges.size();

        std::vector<double> correction_r(cut_edges.size(), 0.0);
        std::vector<double> correction_z(cut_edges.size(), 0.0);
        bool quotient_feasible = false;
        for (int sweep = 0; sweep < test_caps.translation_sweep_cap;
             ++sweep) {
          for (std::size_t cut_index = 0; cut_index < cut_edges.size();
               ++cut_index) {
            const DvclpTranslationCut& cut = cut_edges[cut_index];
            if (cut.cluster_k == cut.cluster_l) {
              continue;
            }
            const std::size_t cluster_k =
                static_cast<std::size_t>(cut.cluster_k);
            const std::size_t cluster_l =
                static_cast<std::size_t>(cut.cluster_l);
            const double mass_k = cluster_mass[cluster_k];
            const double mass_l = cluster_mass[cluster_l];
            const double mass_sum = mass_k + mass_l;
            const double weight_k = mass_k / mass_sum;
            const double weight_l = mass_l / mass_sum;
            const bool frozen_k = radial_frozen[cluster_k];
            const bool frozen_l = radial_frozen[cluster_l];

            if (!frozen_k && !frozen_l) {
              translation_r[cluster_k] +=
                  weight_l * correction_r[cut_index];
              translation_r[cluster_l] -=
                  weight_k * correction_r[cut_index];
            } else if (!frozen_k) {
              translation_r[cluster_k] += correction_r[cut_index];
            } else if (!frozen_l) {
              translation_r[cluster_l] -= correction_r[cut_index];
            }
            translation_z[cluster_k] +=
                weight_l * correction_z[cut_index];
            translation_z[cluster_l] -=
                weight_k * correction_z[cut_index];

            const double shifted_r =
                (translation_r[cluster_k] - translation_r[cluster_l]) +
                cut.delta_vr;
            const double shifted_z =
                (translation_z[cluster_k] - translation_z[cluster_l]) +
                cut.delta_vz;
            const double shifted_norm = std::hypot(shifted_r, shifted_z);
            if (!(shifted_norm > cut.bound)) {
              correction_r[cut_index] = 0.0;
              correction_z[cut_index] = 0.0;
              continue;
            }

            const double gamma = 1.0 - cut.bound / shifted_norm;
            correction_r[cut_index] =
                frozen_k && frozen_l ? 0.0 : gamma * shifted_r;
            correction_z[cut_index] = gamma * shifted_z;
            if (!frozen_k && !frozen_l) {
              translation_r[cluster_k] -=
                  weight_l * correction_r[cut_index];
              translation_r[cluster_l] +=
                  weight_k * correction_r[cut_index];
            } else if (!frozen_k) {
              translation_r[cluster_k] -= correction_r[cut_index];
            } else if (!frozen_l) {
              translation_r[cluster_l] += correction_r[cut_index];
            }
            translation_z[cluster_k] -=
                weight_l * correction_z[cut_index];
            translation_z[cluster_l] +=
                weight_k * correction_z[cut_index];
          }

          quotient_sweeps = sweep + 1;
          quotient_feasible = true;
          for (const DvclpTranslationCut& cut : cut_edges) {
            const std::size_t cluster_k =
                static_cast<std::size_t>(cut.cluster_k);
            const std::size_t cluster_l =
                static_cast<std::size_t>(cut.cluster_l);
            const double shifted_r =
                (translation_r[cluster_k] - translation_r[cluster_l]) +
                cut.delta_vr;
            const double shifted_z =
                (translation_z[cluster_k] - translation_z[cluster_l]) +
                cut.delta_vz;
            if (std::hypot(shifted_r, shifted_z) > cut.bound) {
              quotient_feasible = false;
              break;
            }
          }
          if (quotient_feasible) {
            break;
          }
        }

        if (quotient_feasible) {
          for (std::size_t cluster = 0; cluster < cluster_members.size();
               ++cluster) {
            for (const int node : cluster_members[cluster]) {
              if (!radial_frozen[cluster]) {
                candidate_vr[static_cast<std::size_t>(node)] +=
                    translation_r[cluster];
              }
              candidate_vz[static_cast<std::size_t>(node)] +=
                  translation_z[cluster];
            }
          }
          solver_class = 3;
          dykstra_feasible = true;
        } else {
          for (std::size_t member_index = 0;
               member_index < members.size(); ++member_index) {
            const std::size_t node =
                static_cast<std::size_t>(members[member_index]);
            candidate_vr[node] = snapshot_vr[member_index];
            candidate_vz[node] = snapshot_vz[member_index];
          }
        }
      }
      if (diag_on) {
        std::vector<std::vector<std::size_t>> adjacency(node_count);
        std::vector<std::size_t> self_loops;
        for (std::size_t edge_index = 0; edge_index < budgets.size();
             ++edge_index) {
          const DvclpEdgeBudget& edge = budgets[edge_index];
          if (edge.walk_failed || !edge_inside_component(edge)) {
            continue;
          }
          if (edge.node0 == edge.node1) {
            self_loops.push_back(edge_index);
            continue;
          }
          adjacency[static_cast<std::size_t>(edge.node0)].push_back(
              edge_index);
          adjacency[static_cast<std::size_t>(edge.node1)].push_back(
              edge_index);
        }

        int block_count = 0;
        int bridge_count = 0;
        int trivial_block_count = 0;
        std::size_t largest_block_edges = 0;
        std::size_t largest_block_nodes = 0;
        std::size_t second_largest_block_edges = 0;
        const auto record_block = [&](const std::size_t block_edges,
                                      const std::size_t block_nodes) {
          ++block_count;
          if (block_edges <= 2U) {
            ++trivial_block_count;
          }
          if (block_edges > largest_block_edges) {
            second_largest_block_edges = largest_block_edges;
            largest_block_edges = block_edges;
            largest_block_nodes = block_nodes;
          } else if (block_edges > second_largest_block_edges) {
            second_largest_block_edges = block_edges;
          }
        };
        for (const std::size_t edge_index : self_loops) {
          (void)edge_index;
          record_block(1U, 1U);
        }

        struct DvclpBlockcutFrame {
          int node = -1;
          std::size_t next_adjacency = 0;
        };
        const std::size_t no_edge = budgets.size();
        std::vector<int> discovery(node_count, -1);
        std::vector<int> low(node_count, -1);
        std::vector<int> parent(node_count, -1);
        std::vector<int> child_count(node_count, 0);
        std::vector<std::size_t> parent_edge(node_count, no_edge);
        std::vector<char> articulation(node_count, false);
        std::vector<int> block_node_stamp(node_count, -1);
        std::vector<std::size_t> edge_stack;
        std::vector<DvclpBlockcutFrame> dfs_stack;
        edge_stack.reserve(diag_in_edges);
        dfs_stack.reserve(members.size());
        int discovery_time = 0;
        int block_stamp = 0;

        for (const int start : members) {
          const std::size_t start_index = static_cast<std::size_t>(start);
          if (discovery[start_index] >= 0) {
            continue;
          }
          discovery[start_index] = discovery_time;
          low[start_index] = discovery_time;
          ++discovery_time;
          dfs_stack.push_back({start, 0U});

          while (!dfs_stack.empty()) {
            DvclpBlockcutFrame& frame = dfs_stack.back();
            const int node = frame.node;
            const std::size_t node_index = static_cast<std::size_t>(node);
            if (frame.next_adjacency < adjacency[node_index].size()) {
              const std::size_t edge_index =
                  adjacency[node_index][frame.next_adjacency++];
              if (edge_index == parent_edge[node_index]) {
                continue;
              }
              const DvclpEdgeBudget& edge = budgets[edge_index];
              const int neighbor =
                  edge.node0 == node ? edge.node1 : edge.node0;
              const std::size_t neighbor_index =
                  static_cast<std::size_t>(neighbor);
              if (discovery[neighbor_index] < 0) {
                parent[neighbor_index] = node;
                parent_edge[neighbor_index] = edge_index;
                ++child_count[node_index];
                edge_stack.push_back(edge_index);
                discovery[neighbor_index] = discovery_time;
                low[neighbor_index] = discovery_time;
                ++discovery_time;
                dfs_stack.push_back({neighbor, 0U});
              } else if (discovery[neighbor_index] < discovery[node_index]) {
                edge_stack.push_back(edge_index);
                low[node_index] =
                    std::min(low[node_index], discovery[neighbor_index]);
              }
              continue;
            }

            dfs_stack.pop_back();
            const int parent_node = parent[node_index];
            if (parent_node < 0) {
              articulation[node_index] = child_count[node_index] > 1;
              continue;
            }
            const std::size_t parent_index =
                static_cast<std::size_t>(parent_node);
            low[parent_index] = std::min(low[parent_index], low[node_index]);
            if (low[node_index] > discovery[parent_index]) {
              ++bridge_count;
            }
            if (low[node_index] < discovery[parent_index]) {
              continue;
            }
            if (parent[parent_index] >= 0) {
              articulation[parent_index] = true;
            }

            ++block_stamp;
            std::size_t block_edges = 0;
            std::size_t block_nodes = 0;
            while (!edge_stack.empty()) {
              const std::size_t block_edge_index = edge_stack.back();
              edge_stack.pop_back();
              ++block_edges;
              const DvclpEdgeBudget& block_edge = budgets[block_edge_index];
              const std::size_t node0 =
                  static_cast<std::size_t>(block_edge.node0);
              const std::size_t node1 =
                  static_cast<std::size_t>(block_edge.node1);
              if (block_node_stamp[node0] != block_stamp) {
                block_node_stamp[node0] = block_stamp;
                ++block_nodes;
              }
              if (block_node_stamp[node1] != block_stamp) {
                block_node_stamp[node1] = block_stamp;
                ++block_nodes;
              }
              if (block_edge_index == parent_edge[node_index]) {
                break;
              }
            }
            record_block(block_edges, block_nodes);
          }
        }

        int articulation_count = 0;
        for (const int node : members) {
          if (articulation[static_cast<std::size_t>(node)]) {
            ++articulation_count;
          }
        }
        std::fprintf(
            stderr,
            "[dvclp_blockcut] root=%zu nodes=%zu in_edges=%zu blocks=%d "
            "artic=%d bridges=%d big_edges=%zu big_nodes=%zu "
            "second_edges=%zu trivial=%d\n",
            root_index, members.size(), diag_in_edges, block_count,
            articulation_count, bridge_count, largest_block_edges,
            largest_block_nodes, second_largest_block_edges,
            trivial_block_count);
      }
    } else {
      std::vector<double> dof_mass;
      std::vector<double> u;
      std::vector<double> x;
      dof_mass.reserve(dof_count);
      u.reserve(dof_count);
      x.reserve(dof_count);

      long double z_mass_sum = 0.0L;
      long double z_momentum = 0.0L;
      for (const int unit : z_units) {
        const double mass = unit_mass_z[static_cast<std::size_t>(unit)];
        const double value = node_vz[unit];
        z_mass_sum += static_cast<long double>(mass);
        z_momentum += static_cast<long double>(mass) * value;
      }
      const double center_z = static_cast<double>(z_momentum / z_mass_sum);
      for (const int unit : z_units) {
        dof_z[static_cast<std::size_t>(unit)] =
            static_cast<int>(dof_mass.size());
        dof_mass.push_back(unit_mass_z[static_cast<std::size_t>(unit)]);
        const double centered = node_vz[unit] - center_z;
        u.push_back(centered);
        x.push_back(centered);
      }

      long double r_mass_sum = 0.0L;
      long double r_momentum = 0.0L;
      for (const int unit : r_units) {
        const double mass = unit_mass_r[static_cast<std::size_t>(unit)];
        const double value = node_vr[unit];
        r_mass_sum += static_cast<long double>(mass);
        r_momentum += static_cast<long double>(mass) * value;
      }
      const double center_r =
          contains_axis || r_units.empty()
              ? 0.0
              : static_cast<double>(r_momentum / r_mass_sum);
      for (const int unit : r_units) {
        dof_r[static_cast<std::size_t>(unit)] =
            static_cast<int>(dof_mass.size());
        dof_mass.push_back(unit_mass_r[static_cast<std::size_t>(unit)]);
        const double centered = node_vr[unit] - center_r;
        u.push_back(centered);
        x.push_back(centered);
      }

      std::vector<DvclpConeRow> rows;
      std::vector<std::size_t> active;
      for (std::size_t edge_index = 0; edge_index < budgets.size();
           ++edge_index) {
        const DvclpEdgeBudget& edge = budgets[edge_index];
        if (edge.walk_failed || consensus_constrained[edge_index] ||
            !retained[edge_index] || !edge_inside_component(edge)) {
          continue;
        }
        DvclpConeRow row;
        row.edge_index = edge_index;
        row.beta = budget_upper_bound(edge);
        // Consult eq. 15: power-of-two row scaling is exact in binary64.
        if (row.beta > 0.0) {
          row.scale = std::exp2(-std::ilogb(row.beta));
        }
        const bool axis = target_mesh.node_r[edge.node0] == 0.0 ||
                          target_mesh.node_r[edge.node1] == 0.0;
        const int unit_r0 = unit_r[static_cast<std::size_t>(edge.node0)];
        const int unit_r1 = unit_r[static_cast<std::size_t>(edge.node1)];
        if (!axis && unit_r0 != unit_r1) {
          row.radial0 = dof_r[static_cast<std::size_t>(unit_r0)];
          row.radial1 = dof_r[static_cast<std::size_t>(unit_r1)];
        }
        const int unit_z0 = unit_z[static_cast<std::size_t>(edge.node0)];
        const int unit_z1 = unit_z[static_cast<std::size_t>(edge.node1)];
        if (unit_z0 != unit_z1) {
          row.axial0 = dof_z[static_cast<std::size_t>(unit_z0)];
          row.axial1 = dof_z[static_cast<std::size_t>(unit_z1)];
        }
        if (row.radial0 < 0 && row.axial0 < 0) {
          continue;
        }
        rows.push_back(row);
        if (violating[edge_index]) {
          active.push_back(rows.size() - 1U);
        }
      }
      if (diag_on) {
        diag_rows = rows.size();
        diag_active0 = active.size();
        std::map<std::array<int, 4>, int> support_counts;
        for (const DvclpConeRow& row : rows) {
          ++support_counts[{row.radial0, row.radial1, row.axial0,
                            row.axial1}];
        }
        for (const auto& [support, count] : support_counts) {
          if (count > 1) {
            diag_dup_rows += static_cast<std::size_t>(count);
          }
        }
      }

      std::vector<DvclpConeRow> deduplicated_rows;
      std::vector<char> deduplicated_active;
      std::map<std::array<int, 4>, std::size_t> row_by_support;
      deduplicated_rows.reserve(rows.size());
      deduplicated_active.reserve(rows.size());
      for (const DvclpConeRow& row : rows) {
        std::array<int, 4> support{
            row.radial0, row.radial1, row.axial0, row.axial1};
        if (support[1] < support[0]) {
          std::swap(support[0], support[1]);
        }
        if (support[3] < support[2]) {
          std::swap(support[2], support[3]);
        }
        const auto [position, inserted] =
            row_by_support.emplace(support, deduplicated_rows.size());
        if (inserted) {
          deduplicated_rows.push_back(row);
          deduplicated_active.push_back(violating[row.edge_index]);
          continue;
        }
        ++diag_dedup;
        DvclpConeRow& kept = deduplicated_rows[position->second];
        if (row.beta < kept.beta) {
          kept.beta = row.beta;
          kept.scale = row.scale;
        }
        deduplicated_active[position->second] =
            deduplicated_active[position->second] ||
            violating[row.edge_index];
      }
      rows = std::move(deduplicated_rows);
      active.clear();
      for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
        if (deduplicated_active[row_index]) {
          active.push_back(row_index);
        }
      }

      const auto row_value = [&](const DvclpConeRow& row,
                                 const std::vector<double>& values,
                                 double& radial,
                                 double& axial) {
        radial = row.radial0 < 0
                     ? 0.0
                     : values[static_cast<std::size_t>(row.radial1)] -
                           values[static_cast<std::size_t>(row.radial0)];
        axial = row.axial0 < 0
                    ? 0.0
                    : values[static_cast<std::size_t>(row.axial1)] -
                          values[static_cast<std::size_t>(row.axial0)];
        return std::hypot(radial, axial);
      };
      const auto add_gradient = [](const DvclpConeRow& row,
                                   const double radial_scale,
                                   const double axial_scale,
                                   std::vector<double>& values) {
        if (row.radial0 >= 0) {
          values[static_cast<std::size_t>(row.radial0)] -= radial_scale;
          values[static_cast<std::size_t>(row.radial1)] += radial_scale;
        }
        if (row.axial0 >= 0) {
          values[static_cast<std::size_t>(row.axial0)] -= axial_scale;
          values[static_cast<std::size_t>(row.axial1)] += axial_scale;
        }
      };
      const auto merit = [&](const std::vector<double>& values,
                             const std::vector<std::size_t>& active_rows,
                             const std::vector<double>& multipliers) {
        std::vector<double> stationarity(dof_count, 0.0);
        for (std::size_t dof = 0; dof < dof_count; ++dof) {
          stationarity[dof] =
              dof_mass[dof] * (values[dof] - u[dof]);
        }
        double merit_value = 0.0;
        for (std::size_t position = 0; position < active_rows.size();
             ++position) {
          const DvclpConeRow& row = rows[active_rows[position]];
          double radial = 0.0;
          double axial = 0.0;
          const double norm = row_value(row, values, radial, axial);
          if (norm > 0.0) {
            add_gradient(row,
                         multipliers[position] * row.scale * radial / norm,
                         multipliers[position] * row.scale * axial / norm,
                         stationarity);
          }
          const double feasibility = row.scale * (norm - row.beta);
          merit_value += feasibility * feasibility;
        }
        for (const double residual : stationarity) {
          merit_value += residual * residual;
        }
        return merit_value;
      };

      // Four removals is a structural anti-cycling cap, not a tunable
      // tolerance.
      constexpr int removal_lock_count = 4;
      std::vector<int> removal_counts(rows.size(), 0);
      std::vector<double> multipliers(active.size(), 0.0);
      for (int iteration = 0; iteration < 64; ++iteration) {
        diag_newton_iters = iteration + 1;
        bool refactor = true;
        bool newton_stopped = false;
        std::vector<double> step;
        while (refactor) {
          refactor = false;
          if (solver_rev >= 1 && active.size() >= 2U) {
            const std::size_t qr_m = active.size();
            std::vector<double> qr_columns(qr_m * dof_count, 0.0);
            bool qr_columns_valid = true;
            double qr_norm0_max = 0.0;
            for (std::size_t position = 0; position < qr_m; ++position) {
              const DvclpConeRow& row = rows[active[position]];
              double radial = 0.0;
              double axial = 0.0;
              const double norm = row_value(row, x, radial, axial);
              if (!(norm > 0.0)) {
                qr_columns_valid = false;
                break;
              }
              std::vector<double> gradient(dof_count, 0.0);
              add_gradient(row, row.scale * (radial / norm),
                           row.scale * (axial / norm), gradient);
              double norm0_sq = 0.0;
              for (std::size_t dof = 0; dof < dof_count; ++dof) {
                qr_columns[position * dof_count + dof] = gradient[dof];
                norm0_sq += gradient[dof] * gradient[dof];
              }
              qr_norm0_max = std::max(qr_norm0_max, std::sqrt(norm0_sq));
            }
            if (qr_columns_valid && qr_norm0_max > 0.0) {
              std::vector<std::size_t> qr_perm(qr_m, 0);
              for (std::size_t position = 0; position < qr_m; ++position) {
                qr_perm[position] = position;
              }
              const double qr_tol =
                  64.0 * eps *
                  static_cast<double>(qr_m + dof_count) * qr_norm0_max;
              const std::size_t qr_max_rank = std::min(qr_m, dof_count);
              std::size_t qr_rank = 0;
              for (std::size_t step = 0; step < qr_max_rank; ++step) {
                std::size_t best = step;
                double best_norm_sq = -1.0;
                for (std::size_t candidate = step; candidate < qr_m;
                     ++candidate) {
                  double norm_sq = 0.0;
                  for (std::size_t dof = step; dof < dof_count; ++dof) {
                    const double value =
                        qr_columns[candidate * dof_count + dof];
                    norm_sq += value * value;
                  }
                  if (norm_sq > best_norm_sq ||
                      (norm_sq == best_norm_sq &&
                       qr_perm[candidate] < qr_perm[best])) {
                    best = candidate;
                    best_norm_sq = norm_sq;
                  }
                }
                if (best != step) {
                  for (std::size_t dof = 0; dof < dof_count; ++dof) {
                    std::swap(qr_columns[step * dof_count + dof],
                              qr_columns[best * dof_count + dof]);
                  }
                  std::swap(qr_perm[step], qr_perm[best]);
                }
                const double pivot_norm =
                    std::sqrt(std::max(best_norm_sq, 0.0));
                if (!(pivot_norm > qr_tol)) {
                  break;
                }
                qr_rank = step + 1;
                std::vector<double> reflector(dof_count - step, 0.0);
                for (std::size_t dof = step; dof < dof_count; ++dof) {
                  reflector[dof - step] =
                      qr_columns[step * dof_count + dof];
                }
                reflector[0] +=
                    reflector[0] >= 0.0 ? pivot_norm : -pivot_norm;
                double reflector_sq = 0.0;
                for (const double value : reflector) {
                  reflector_sq += value * value;
                }
                if (reflector_sq == 0.0) {
                  continue;
                }
                for (std::size_t column = step + 1; column < qr_m;
                     ++column) {
                  double dot = 0.0;
                  for (std::size_t dof = step; dof < dof_count; ++dof) {
                    dot += reflector[dof - step] *
                           qr_columns[column * dof_count + dof];
                  }
                  const double coefficient = 2.0 * dot / reflector_sq;
                  for (std::size_t dof = step; dof < dof_count; ++dof) {
                    qr_columns[column * dof_count + dof] -=
                        coefficient * reflector[dof - step];
                  }
                }
              }
              if (qr_rank < qr_m) {
                std::vector<std::size_t> dependent(
                    qr_perm.begin() +
                        static_cast<std::ptrdiff_t>(qr_rank),
                    qr_perm.end());
                std::sort(dependent.begin(), dependent.end());
                for (std::size_t index = dependent.size(); index-- > 0;) {
                  const std::size_t position = dependent[index];
                  active.erase(active.begin() +
                               static_cast<std::ptrdiff_t>(position));
                  multipliers.erase(
                      multipliers.begin() +
                      static_cast<std::ptrdiff_t>(position));
                  ++diag_qr_drops;
                }
              }
            }
          }
          const std::size_t system_size = dof_count + active.size();
          std::vector<double> matrix(system_size * system_size, 0.0);
          std::vector<double> rhs(system_size, 0.0);
          std::vector<double> stationarity(dof_count, 0.0);
          for (std::size_t dof = 0; dof < dof_count; ++dof) {
            matrix[dof * system_size + dof] = dof_mass[dof];
            stationarity[dof] = dof_mass[dof] * (x[dof] - u[dof]);
          }

          for (std::size_t position = 0; position < active.size();
               ++position) {
            const DvclpConeRow& row = rows[active[position]];
            double radial = 0.0;
            double axial = 0.0;
            const double norm = row_value(row, x, radial, axial);
            if (!(norm > 0.0)) {
              newton_stopped = true;
              break;
            }
            const double normal_r = radial / norm;
            const double normal_z = axial / norm;
            std::vector<double> gradient(dof_count, 0.0);
            add_gradient(row, row.scale * normal_r,
                         row.scale * normal_z, gradient);
            for (std::size_t dof = 0; dof < dof_count; ++dof) {
              stationarity[dof] += multipliers[position] * gradient[dof];
              matrix[dof * system_size + dof_count + position] =
                  gradient[dof];
              matrix[(dof_count + position) * system_size + dof] =
                  gradient[dof];
            }
            rhs[dof_count + position] =
                -row.scale * (norm - row.beta);

            if (row.radial0 >= 0 && row.axial0 >= 0) {
              std::vector<double> tangent(dof_count, 0.0);
              add_gradient(row, -normal_z, normal_r, tangent);
              const double scale =
                  multipliers[position] * row.scale / norm;
              for (std::size_t row_dof = 0; row_dof < dof_count;
                   ++row_dof) {
                for (std::size_t column_dof = 0;
                     column_dof < dof_count; ++column_dof) {
                  matrix[row_dof * system_size + column_dof] +=
                      scale * tangent[row_dof] * tangent[column_dof];
                }
              }
            }
          }
          if (newton_stopped) {
            break;
          }
          for (std::size_t dof = 0; dof < dof_count; ++dof) {
            rhs[dof] = -stationarity[dof];
          }

          std::size_t zero_pivot = system_size;
          if (!solve_dense_symmetric_ldlt(
                  std::move(matrix), std::move(rhs), step, zero_pivot)) {
            ++diag_ldlt_drops;
            if (active.empty()) {
              newton_stopped = true;
              break;
            }
            std::size_t drop = active.size() - 1U;
            if (zero_pivot >= dof_count &&
                zero_pivot - dof_count < active.size()) {
              drop = zero_pivot - dof_count;
            }
            active.erase(active.begin() + static_cast<std::ptrdiff_t>(drop));
            multipliers.erase(
                multipliers.begin() + static_cast<std::ptrdiff_t>(drop));
            refactor = true;
          }
        }
        if (newton_stopped) {
          diag_newton_exit = 0;
          break;
        }

        const double merit_before = merit(x, active, multipliers);
        bool accepted = false;
        bool bitwise_no_op = true;
        std::vector<double> trial_x(dof_count, 0.0);
        std::vector<double> trial_multipliers(active.size(), 0.0);
        for (int line_search = 0; line_search <= 16; ++line_search) {
          const double fraction = std::ldexp(1.0, -line_search);
          bitwise_no_op = true;
          for (std::size_t dof = 0; dof < dof_count; ++dof) {
            trial_x[dof] = x[dof] + fraction * step[dof];
            bitwise_no_op =
                bitwise_no_op && bitwise_equal(trial_x[dof], x[dof]);
          }
          for (std::size_t position = 0; position < active.size();
               ++position) {
            trial_multipliers[position] =
                multipliers[position] +
                fraction * step[dof_count + position];
          }
          // This plain-double merit only steers Newton; correctness comes from
          // the downstream homothetic certificate and final violation count.
          if (merit(trial_x, active, trial_multipliers) < merit_before) {
            x = trial_x;
            multipliers = trial_multipliers;
            accepted = true;
            ++diag_ls_accepts;
            break;
          }
        }
        if (!accepted) {
          diag_newton_exit = 1;
          diag_ls_bitwise = bitwise_no_op ? 1 : 0;
          break;
        }

        bool removed = false;
        std::vector<std::size_t> retained_active;
        std::vector<double> retained_multipliers;
        retained_active.reserve(active.size());
        retained_multipliers.reserve(multipliers.size());
        for (std::size_t position = 0; position < active.size(); ++position) {
          const std::size_t row_index = active[position];
          if (multipliers[position] < 0.0 &&
              removal_counts[row_index] < removal_lock_count) {
            ++removal_counts[row_index];
            removed = true;
            continue;
          }
          retained_active.push_back(row_index);
          retained_multipliers.push_back(multipliers[position]);
        }
        active = std::move(retained_active);
        multipliers = std::move(retained_multipliers);

        std::vector<bool> is_active(rows.size(), false);
        for (const std::size_t active_row : active) {
          is_active[active_row] = true;
        }
        bool added = false;
        for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
          if (is_active[row_index]) {
            continue;
          }
          double radial = 0.0;
          double axial = 0.0;
          const double jump = row_value(rows[row_index], x, radial, axial);
          if (!(jump > rows[row_index].beta * (1.0 + 4.0 * eps))) {
            continue;
          }
          const auto position =
              std::lower_bound(active.begin(), active.end(), row_index);
          const std::size_t offset =
              static_cast<std::size_t>(position - active.begin());
          active.insert(position, row_index);
          multipliers.insert(
              multipliers.begin() + static_cast<std::ptrdiff_t>(offset), 0.0);
          is_active[row_index] = true;
          added = true;
        }
        if (!removed && !added && bitwise_no_op) {
          diag_newton_exit = 2;
          break;
        }
      }
      if (diag_newton_exit == -2) {
        diag_newton_exit = 3;
      }

      for (const int unit : z_units) {
        candidate_vz[static_cast<std::size_t>(unit)] =
            x[static_cast<std::size_t>(dof_z[static_cast<std::size_t>(unit)])] +
            center_z;
      }
      for (const int unit : r_units) {
        candidate_vr[static_cast<std::size_t>(unit)] =
            x[static_cast<std::size_t>(dof_r[static_cast<std::size_t>(unit)])] +
            center_r;
      }

      // Constant shifts are null modes (axis rows mask vr; differences are invariant),
      // preserving exact feasibility and restoring momentum so the telescope sees only rounding.
      long double z_momentum_before = 0.0L;
      long double z_momentum_candidate = 0.0L;
      for (const int unit : z_units) {
        const std::size_t dof =
            static_cast<std::size_t>(dof_z[static_cast<std::size_t>(unit)]);
        const long double mass =
            static_cast<long double>(unit_mass_z[static_cast<std::size_t>(unit)]);
        z_momentum_before +=
            mass * static_cast<long double>(u[dof] + center_z);
        z_momentum_candidate +=
            mass * static_cast<long double>(
                       candidate_vz[static_cast<std::size_t>(unit)]);
      }
      const double z_shift = static_cast<double>(
          (z_momentum_before - z_momentum_candidate) / z_mass_sum);
      for (const int unit : z_units) {
        candidate_vz[static_cast<std::size_t>(unit)] += z_shift;
      }

      if (!r_units.empty()) {
        long double r_momentum_before = 0.0L;
        long double r_momentum_candidate = 0.0L;
        for (const int unit : r_units) {
          const std::size_t dof =
              static_cast<std::size_t>(dof_r[static_cast<std::size_t>(unit)]);
          const long double mass = static_cast<long double>(
              unit_mass_r[static_cast<std::size_t>(unit)]);
          r_momentum_before +=
              mass * static_cast<long double>(u[dof] + center_r);
          r_momentum_candidate +=
              mass * static_cast<long double>(
                         candidate_vr[static_cast<std::size_t>(unit)]);
        }
        const double r_shift = static_cast<double>(
            (r_momentum_before - r_momentum_candidate) / r_mass_sum);
        for (const int unit : r_units) {
          candidate_vr[static_cast<std::size_t>(unit)] += r_shift;
        }
      }
      // Newton restoration leaves only rounding-scale radial drift; axis vr is not a dof, so nothing is ledgered.
    }

    for (const int node : members) {
      if (node_z_in_component[static_cast<std::size_t>(node)]) {
        candidate_vz[static_cast<std::size_t>(node)] =
            candidate_vz[static_cast<std::size_t>(unit_z[node])];
      }
      if (target_mesh.node_r[node] == 0.0) {
        candidate_vr[static_cast<std::size_t>(node)] = 0.0;
      } else if (node_r_in_component[static_cast<std::size_t>(node)]) {
        candidate_vr[static_cast<std::size_t>(node)] =
            candidate_vr[static_cast<std::size_t>(unit_r[node])];
      }
    }

    if (solver_class == 1 || solver_class == 3) {
      bool post_collapse_feasible = true;
      for (std::size_t edge_index = 0; edge_index < budgets.size();
           ++edge_index) {
        const DvclpEdgeBudget& edge = budgets[edge_index];
        if (edge.walk_failed || !edge_inside_component(edge)) {
          continue;
        }
        double delta_vr = 0.0;
        double delta_vz = 0.0;
        const double jump = projection_jump(
            target_mesh, edge, candidate_vr.data(), candidate_vz.data(),
            delta_vr, delta_vz);
        if (jump > budget_upper_bound(edge)) {
          post_collapse_feasible = false;
          break;
        }
      }
      if (!post_collapse_feasible) {
        for (std::size_t member_index = 0;
             member_index < members.size(); ++member_index) {
          const std::size_t node =
              static_cast<std::size_t>(members[member_index]);
          candidate_vr[node] = snapshot_vr[member_index];
          candidate_vz[node] = snapshot_vz[member_index];
        }
        for (const int node : members) {
          if (node_z_in_component[static_cast<std::size_t>(node)]) {
            candidate_vz[static_cast<std::size_t>(node)] =
                candidate_vz[static_cast<std::size_t>(unit_z[node])];
          }
          if (target_mesh.node_r[node] == 0.0) {
            candidate_vr[static_cast<std::size_t>(node)] = 0.0;
          } else if (node_r_in_component[static_cast<std::size_t>(node)]) {
            candidate_vr[static_cast<std::size_t>(node)] =
                candidate_vr[static_cast<std::size_t>(unit_r[node])];
          }
        }
        dykstra_feasible = false;
        solver_class = 2;
      }
    }

    int kkt_certified_flag = 0;
    double kkt_component_lam_max = 0.0;
    if (solver_class == 1) {
      TENRYU_ASSERT(snapshot_vr.size() == members.size(),
                    "DVCLP KKT certificate requires the pre-Dykstra snapshot");
      component_kkt_edges.clear();
      for (std::size_t edge_index = 0; edge_index < budgets.size();
           ++edge_index) {
        const DvclpEdgeBudget& edge = budgets[edge_index];
        if (edge.walk_failed || !edge_inside_component(edge)) {
          continue;
        }
        component_kkt_edges.push_back(edge_index);
      }
      const DvclpKktCertificate certificate = certify_dvclp_kkt(
          target_mesh, budgets, component_kkt_edges, members,
          node_z_in_component, node_r_in_component, nodal_mass,
          candidate_vr, candidate_vz, snapshot_vr, snapshot_vz);
      kkt_component_lam_max = certificate.lam_max;
      if (certificate.certified) {
        solver_class = 4;
        kkt_certified_flag = 1;
        ++result.kkt_certified_components;
        result.kkt_lam_max =
            std::max(result.kkt_lam_max, certificate.lam_max);
      } else if (certificate.fail_clause == 1) {
        ++result.kkt_fail_eq;
      } else if (certificate.fail_clause == 2) {
        ++result.kkt_fail_stationarity;
      } else {
        ++result.kkt_fail_cone;
      }
    }

    long double candidate_momentum_r = 0.0L;
    long double candidate_momentum_z = 0.0L;
    for (const int node : members) {
      const long double mass = static_cast<long double>(nodal_mass[node]);
      if (node_r_in_component[static_cast<std::size_t>(node)]) {
        candidate_momentum_r +=
            mass * candidate_vr[static_cast<std::size_t>(node)];
      }
      if (node_z_in_component[static_cast<std::size_t>(node)]) {
        candidate_momentum_z +=
            mass * candidate_vz[static_cast<std::size_t>(node)];
      }
    }
    const double mean_vr =
        contains_axis || r_units.empty()
            ? 0.0
            : static_cast<double>(candidate_momentum_r / node_mass_sum_r);
    const double mean_vz =
        static_cast<double>(candidate_momentum_z / node_mass_sum_z);

    if (diag_on) {
      for (std::size_t edge_index = 0; edge_index < budgets.size();
           ++edge_index) {
        const DvclpEdgeBudget& edge = budgets[edge_index];
        if (edge.walk_failed || !edge_inside_component(edge)) {
          continue;
        }
        double delta_vr = 0.0;
        double delta_vz = 0.0;
        const double jump = projection_jump(
            target_mesh, edge, candidate_vr.data(), candidate_vz.data(),
            delta_vr, delta_vz);
        const double bound = budget_upper_bound(edge);
        if (bound > 0.0) {
          diag_ratio_end = std::max(diag_ratio_end, jump / bound);
        }
      }
    }
    bool candidate_feasible = true;
    for (std::size_t edge_index = 0; edge_index < budgets.size();
         ++edge_index) {
      const DvclpEdgeBudget& edge = budgets[edge_index];
      // Omitted edges wholly inside this inner component are re-evaluated;
      // cross-component and frozen-unit edges retain the screening guarantee.
      if (edge.walk_failed || !edge_inside_component(edge)) {
        continue;
      }
      double delta_vr = 0.0;
      double delta_vz = 0.0;
      const double jump = projection_jump(
          target_mesh, edge, candidate_vr.data(), candidate_vz.data(),
          delta_vr, delta_vz);
      if (jump > budget_upper_bound(edge)) {
        candidate_feasible = false;
        break;
      }
    }

    double alpha_star = 1.0;
    if (!candidate_feasible) {
      double alpha_feas = 1.0;
      for (std::size_t edge_index = 0; edge_index < budgets.size();
           ++edge_index) {
        const DvclpEdgeBudget& edge = budgets[edge_index];
        if (edge.walk_failed || !edge_inside_component(edge)) {
          continue;
        }
        double delta_vr = 0.0;
        double delta_vz = 0.0;
        const double jump = projection_jump(
            target_mesh, edge, candidate_vr.data(), candidate_vz.data(),
            delta_vr, delta_vz);
        if (!(jump > budget_upper_bound(edge))) {
          continue;
        }
        const double difference_upper = jump * (1.0 + 4.0 * eps);
        const double infinity = std::numeric_limits<double>::infinity();
        const auto component_ulp = [&](const double value) {
          const double magnitude = std::fabs(value);
          return std::nextafter(magnitude, infinity) - magnitude;
        };
        double allow = 0.0;
        if (!bitwise_equal(node_vz[edge.node0], node_vz[edge.node1])) {
          allow += 2.0 * (component_ulp(node_vz[edge.node0]) +
                          component_ulp(node_vz[edge.node1]));
        }
        const bool axis_edge = target_mesh.node_r[edge.node0] == 0.0 ||
                               target_mesh.node_r[edge.node1] == 0.0;
        if (!axis_edge &&
            !bitwise_equal(node_vr[edge.node0], node_vr[edge.node1])) {
          allow += 2.0 * (component_ulp(node_vr[edge.node0]) +
                          component_ulp(node_vr[edge.node1]));
        }
        // Per unequal direction, both endpoint writes round once
        // (<= ulp(|target|) each); current-value ulps bound targets up to one
        // binade, whose shift is covered by 2.  Equal directions round
        // identically and cancel exactly.
        const double numerator =
            budget_upper_bound(edge) * (1.0 - 4.0 * eps) -
            allow;
        if (numerator <= 0.0) {
          alpha_feas = 0.0;
          ++diag_nzero;
        } else {
          alpha_feas = std::min(
              alpha_feas, numerator / difference_upper);
        }
      }
      alpha_feas = std::min(alpha_feas, 1.0);
      alpha_star =
          alpha_feas > 0.0 ? std::nextafter(alpha_feas, 0.0) : 0.0;
    }

    if (diag_on) {
      std::fprintf(
          stderr,
          "[dvclp_comp] root=%zu nodes=%zu zu=%zu ru=%zu dofs=%zu "
          "capped=%d in_edges=%zu ratio0=%.9e rows=%zu dup_rows=%zu dedup=%d "
          "active0=%zu iters=%d exit=%d lsbw=%d drops=%d accepts=%d "
          "ratio_end=%.9e feasible=%d alpha=%.9e nzero=%zu "
          "solver=%d dysweeps=%d qk=%d qcut=%zu qsweeps=%d kkt=%d lam_max=%.9e qrd=%d\n",
          root_index, members.size(), z_units.size(), r_units.size(),
          dof_count,
          component_newton_capped ? 1 : 0, diag_in_edges, diag_ratio0,
          diag_rows, diag_dup_rows, diag_dedup, diag_active0, diag_newton_iters,
          diag_newton_exit, diag_ls_bitwise, diag_ldlt_drops,
          diag_ls_accepts, diag_ratio_end,
          candidate_feasible ? 1 : 0, alpha_star, diag_nzero,
          solver_class, dykstra_sweeps, quotient_clusters,
          quotient_cut_edges, quotient_sweeps, kkt_certified_flag,
          kkt_component_lam_max, diag_qr_drops);
    }
    result.qr_dropped_rows += static_cast<std::size_t>(diag_qr_drops);
    result.alpha_min = std::min(result.alpha_min, alpha_star);
    if (!dykstra_feasible) {
      ++result.homothety_components;
    }

    for (const int node : members) {
      if (node_r_in_component[static_cast<std::size_t>(node)]) {
        if (dykstra_feasible) {
          target_vr[static_cast<std::size_t>(node)] =
              candidate_vr[static_cast<std::size_t>(node)];
        } else {
          const double deviation_r =
              candidate_vr[static_cast<std::size_t>(node)] - mean_vr;
          target_vr[static_cast<std::size_t>(node)] =
              mean_vr + alpha_star * deviation_r;
        }
      }
      if (node_z_in_component[static_cast<std::size_t>(node)]) {
        if (dykstra_feasible) {
          target_vz[static_cast<std::size_t>(node)] =
              candidate_vz[static_cast<std::size_t>(node)];
        } else {
          const double deviation_z =
              candidate_vz[static_cast<std::size_t>(node)] - mean_vz;
          target_vz[static_cast<std::size_t>(node)] =
              mean_vz + alpha_star * deviation_z;
        }
      }
      if (target_mesh.node_r[node] == 0.0) {
        TENRYU_ASSERT(candidate_vr[static_cast<std::size_t>(node)] == 0.0,
                      "DVCLP axis-node radial velocity must be zero");
      }
    }

    long double delta_sum_r = 0.0L;
    long double delta_sum_z = 0.0L;
    long double kinetic_before = 0.0L;
    long double kinetic_after = 0.0L;
    for (const int node : members) {
      const std::size_t node_index = static_cast<std::size_t>(node);
      const long double mass = static_cast<long double>(nodal_mass[node]);
      const long double velocity_r = static_cast<long double>(node_vr[node]);
      const long double velocity_z = static_cast<long double>(node_vz[node]);
      const long double target_velocity_r =
          node_r_in_component[node_index]
              ? static_cast<long double>(target_vr[node_index])
              : velocity_r;
      const long double target_velocity_z =
          node_z_in_component[node_index]
              ? static_cast<long double>(target_vz[node_index])
              : velocity_z;
      if (node_r_in_component[node_index]) {
        delta_sum_r += mass * (target_velocity_r - velocity_r);
        kinetic_before += 0.5L * mass * velocity_r * velocity_r;
        kinetic_after +=
            0.5L * mass * target_velocity_r * target_velocity_r;
      }
      if (node_z_in_component[node_index]) {
        delta_sum_z += mass * (target_velocity_z - velocity_z);
        kinetic_before += 0.5L * mass * velocity_z * velocity_z;
        kinetic_after +=
            0.5L * mass * target_velocity_z * target_velocity_z;
      }
    }
    const long double imbalance_bound_r =
        static_cast<long double>(members.size()) *
        static_cast<long double>(eps) * abs_momentum_r;
    const long double imbalance_bound_z =
        static_cast<long double>(members.size()) *
        static_cast<long double>(eps) * abs_momentum_z;
    // Only the fallback contraction changes radial momentum in an axis
    // component; its pinned-zero mean gives the expected (alpha-1)*P_r term.
    const long double telescope_r =
        contains_axis
            ? delta_sum_r -
                  (static_cast<long double>(alpha_star) - 1.0L) * momentum_r
            : delta_sum_r;
    if (std::fabs(telescope_r) > imbalance_bound_r ||
        std::fabs(delta_sum_z) > imbalance_bound_z) {
      char buffer[256];
      std::snprintf(
          buffer, sizeof(buffer),
          "dvclp_homothety_imbalance: root=%zu nodes=%zu "
          "dr=%.21Lg br=%.21Lg dz=%.21Lg bz=%.21Lg",
          root_index, members.size(), telescope_r, imbalance_bound_r,
          delta_sum_z, imbalance_bound_z);
      result.ok = false;
      result.reason = buffer;
      return false;
    }

    // The metric projection cannot increase centered kinetic energy, and the
    // certified contraction can only decrease it further.
    const long double component_q = kinetic_before - kinetic_after;
    const long double variance_bound =
        8.0L * static_cast<long double>(members.size() + 2U) *
        static_cast<long double>(eps) *
        (std::fabs(kinetic_before) + std::fabs(kinetic_after));
    if (component_q < -variance_bound) {
      char buffer[224];
      std::snprintf(
          buffer, sizeof(buffer),
          "dvclp_homothety_variance: root=%zu nodes=%zu Q=%.21Lg "
          "bound=%.21Lg",
          root_index, members.size(), component_q, variance_bound);
      result.ok = false;
      result.reason = buffer;
      return false;
    }
    if (contains_axis) {
      result.axis_projection_r -= delta_sum_r;
    }

    touched_cells.clear();
    long double component_corner_mass = 0.0L;
    for (const int node : members) {
      const std::size_t node_index = static_cast<std::size_t>(node);
      const long double mass = static_cast<long double>(nodal_mass[node]);
      const long double target_velocity_r =
          node_r_in_component[node_index]
              ? static_cast<long double>(target_vr[node_index])
              : static_cast<long double>(node_vr[node]);
      const long double target_velocity_z =
          node_z_in_component[node_index]
              ? static_cast<long double>(target_vz[node_index])
              : static_cast<long double>(node_vz[node]);
      const long double delta_momentum_r =
          mass * (target_velocity_r -
                  static_cast<long double>(node_vr[node]));
      const long double delta_momentum_z =
          mass * (target_velocity_z -
                  static_cast<long double>(node_vz[node]));
      for (const DvclpNodeCorner& incidence :
           graph.node_corners[static_cast<std::size_t>(node)]) {
        const long double incidence_mass =
            static_cast<long double>(corner_mass[incidence.corner]);
        const double fraction =
            corner_mass[incidence.corner] / nodal_mass[node];
        corner_momentum_r[incidence.corner] +=
            fraction * static_cast<double>(delta_momentum_r);
        corner_momentum_z[incidence.corner] +=
            fraction * static_cast<double>(delta_momentum_z);
        if (!cell_touched[static_cast<std::size_t>(incidence.cell)]) {
          cell_touched[static_cast<std::size_t>(incidence.cell)] = true;
          touched_cells.push_back(incidence.cell);
        }
        cell_component_mass[static_cast<std::size_t>(incidence.cell)] +=
            incidence_mass;
        component_corner_mass += incidence_mass;
      }
      if (node_r_in_component[node_index]) {
        node_vr[node] = target_vr[node_index];
      }
      if (node_z_in_component[node_index]) {
        node_vz[node] = target_vz[node_index];
      }
    }

    const double component_q_double = static_cast<double>(component_q);
    std::sort(touched_cells.begin(), touched_cells.end());
    for (const int cell : touched_cells) {
      const long double cell_mass =
          cell_component_mass[static_cast<std::size_t>(cell)];
      if (cell_mass != 0.0L) {
        const double fraction =
            static_cast<double>(cell_mass / component_corner_mass);
        cell_q_projection[cell] += fraction * component_q_double;
      }
      cell_component_mass[static_cast<std::size_t>(cell)] = 0.0L;
      cell_touched[static_cast<std::size_t>(cell)] = false;
    }
    result.q_total += component_q_double;

    for (const int unit : z_units) {
      dof_z[static_cast<std::size_t>(unit)] = -1;
    }
    for (const int unit : r_units) {
      dof_r[static_cast<std::size_t>(unit)] = -1;
    }
  }
  const std::chrono::steady_clock::time_point tp_return =
      std::chrono::steady_clock::now();
  if (solve_split_enabled) {
    std::fprintf(
        stderr,
        "[dvclp_solve_split] union_ms=%.1f viol_ms=%.1f affected_ms=%.1f "
        "components_ms=%.1f tail_ms=%.1f components=%zu members=%zu\n",
        std::chrono::duration<double, std::milli>(tp_union - tp_entry)
            .count(),
        std::chrono::duration<double, std::milli>(tp_violation - tp_union)
            .count(),
        std::chrono::duration<double, std::milli>(tp_affected - tp_violation)
            .count(),
        std::chrono::duration<double, std::milli>(tp_components - tp_affected)
            .count(),
        std::chrono::duration<double, std::milli>(tp_return - tp_components)
            .count(),
        split_components, split_members_total);
  }
  return true;
}

std::size_t count_projection_violations(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const std::vector<char>& consensus_constrained,
    const std::vector<bool>& representation_converged,
    const double* node_vr,
    const double* node_vz,
    double& max_ratio) {
  std::size_t violations = 0;
  max_ratio = 0.0;
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (edge.walk_failed || representation_converged[edge_index]) {
      continue;
    }
    double delta_vr = 0.0;
    double delta_vz = 0.0;
    const double jump = projection_jump(
        target_mesh, edge, node_vr, node_vz, delta_vr, delta_vz);
    const bool zero_budget_satisfied =
        !consensus_constrained[edge_index] ||
        consensus_edge_satisfied(target_mesh, edge, node_vr, node_vz);
    max_ratio = std::max(
        max_ratio,
        zero_budget_satisfied
            ? projection_ratio(jump, edge.budget)
            : std::numeric_limits<double>::infinity());
    if (!zero_budget_satisfied || jump > budget_upper_bound(edge)) {
      ++violations;
    }
  }
  return violations;
}

struct DvclpClassifyCounters {
  std::size_t quotient_certified_edges = 0;
  std::size_t resolution_certified_edges = 0;
  std::size_t lattice_certified_edges = 0;
};

// SS8-w2fix23: three-layer consensus classification (design doc
// docs/design/dvclp_eq52_floor_20260813.md).  Layer 0: exact zero budgets.
// Layer 1 (eq. 52'): every feasible value of the component provably rounds
// to the stored component mean.  Layer 2 (C2): the mean collapse differs
// from every feasible completion by less than the energy ledger's own
// rounding envelope for the component.  Certification is per direction on
// a Kruskal merge tree with adoption by any passing ancestor.
DvclpClassifyCounters classify_dvclp_consensus_edges(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const double* nodal_mass,
    const double* node_vr,
    const double* node_vz,
    const std::size_t node_count,
    std::vector<char>& consensus_constrained) {
  const double eps = std::numeric_limits<double>::epsilon();
  DvclpClassifyCounters counters;

  std::vector<std::size_t> order;
  order.reserve(budgets.size());
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    if (!budgets[edge_index].walk_failed) {
      order.push_back(edge_index);
    }
  }
  std::sort(order.begin(), order.end(),
            [&](const std::size_t left, const std::size_t right) {
              const double left_upper = budget_upper_bound(budgets[left]);
              const double right_upper = budget_upper_bound(budgets[right]);
              if (left_upper != right_upper) {
                return left_upper < right_upper;
              }
              return left < right;
            });

  struct Node {
    int parent = -1;
    bool pass = false;
    bool quotient = false;
  };
  struct Direction {
    std::vector<int> parent;
    std::vector<double> mass;
    std::vector<double> ref;
    std::vector<double> off;
    std::vector<double> energy;
    std::vector<double> tree;
    std::vector<std::size_t> count;
    std::vector<double> vmin;
    std::vector<double> vmax;
    std::vector<Node> nodes;
    std::vector<int> latest_node;
    std::vector<int> edge_node;
    std::vector<char> adopted;
    std::vector<char> adopted_quotient;

    Direction(const double* velocity,
              const double* nodal_mass,
              const std::size_t node_count,
              const std::size_t edge_count)
        : parent(node_count, 0),
          mass(node_count, 0.0),
          ref(node_count, 0.0),
          off(node_count, 0.0),
          energy(node_count, 0.0),
          tree(node_count, 0.0),
          count(node_count, 1U),
          vmin(node_count, 0.0),
          vmax(node_count, 0.0),
          latest_node(node_count, -1),
          edge_node(edge_count, -1) {
      for (std::size_t node = 0; node < node_count; ++node) {
        parent[node] = static_cast<int>(node);
        mass[node] = nodal_mass[node];
        ref[node] = velocity[node];
        vmin[node] = velocity[node];
        vmax[node] = velocity[node];
      }
    }

    void process_edge(const std::size_t edge_index,
                      const DvclpEdgeBudget& edge,
                      const double beta_upper,
                      const double eps) {
      const int root0 = consensus_root(parent, edge.node0);
      const int root1 = consensus_root(parent, edge.node1);
      if (root0 == root1) {
        edge_node[edge_index] =
            latest_node[static_cast<std::size_t>(root0)];
        return;
      }
      const int small_root = std::min(root0, root1);
      const int large_root = std::max(root0, root1);
      const std::size_t small_index =
          static_cast<std::size_t>(small_root);
      const std::size_t large_index =
          static_cast<std::size_t>(large_root);
      const double mass_small = mass[small_index];
      const double mass_large = mass[large_index];
      const double mass_sum = mass_small + mass_large;
      const std::size_t count_sum = count[small_index] + count[large_index];
      const double d = (ref[large_index] - ref[small_index]) +
                       (off[large_index] - off[small_index]);
      double energy_merged;
      double off_merged;
      if (mass_sum > 0.0) {
        const double reduced = mass_small * mass_large / mass_sum;
        energy_merged = (energy[small_index] + energy[large_index]) +
                        0.5 * reduced * d * d;
        energy_merged = energy_merged * (1.0 - 16.0 * eps);
        if (!(energy_merged > 0.0)) {
          energy_merged = 0.0;
        }
        off_merged = off[small_index] + mass_large / mass_sum * d;
      } else {
        energy_merged = 0.0;
        off_merged = 0.0;
      }
      double tree_merged = std::nextafter(
          tree[small_index] + tree[large_index],
          std::numeric_limits<double>::infinity());
      tree_merged = std::nextafter(
          tree_merged + beta_upper,
          std::numeric_limits<double>::infinity());
      const double vmin_merged =
          std::min(vmin[small_index], vmin[large_index]);
      const double vmax_merged =
          std::max(vmax[small_index], vmax[large_index]);

      const bool uniform_pass = vmin_merged == vmax_merged;
      const double q = ref[small_index] + off_merged;
      bool quotient_pass = false;
      if (std::isfinite(q) && mass_sum > 0.0) {
        const double half_low =
            (q - std::nextafter(
                     q, -std::numeric_limits<double>::infinity())) *
            0.5;
        const double half_high =
            (std::nextafter(q, std::numeric_limits<double>::infinity()) -
             q) *
            0.5;
        double low_need = std::nextafter(
            q - vmin_merged, std::numeric_limits<double>::infinity());
        low_need = std::nextafter(
            low_need + tree_merged,
            std::numeric_limits<double>::infinity());
        double high_need = std::nextafter(
            vmax_merged - q, std::numeric_limits<double>::infinity());
        high_need = std::nextafter(
            high_need + tree_merged,
            std::numeric_limits<double>::infinity());
        quotient_pass = low_need < half_low && high_need < half_high;
      }
      bool resolution_pass = false;
      if (mass_sum > 0.0) {
        double k_box = 0.5 * mass_sum * tree_merged * tree_merged;
        k_box = k_box *
                (1.0 + static_cast<double>(count_sum + 8U) * eps);
        double resolution =
            8.0 * static_cast<double>(count_sum + 2U) * eps *
            energy_merged;
        resolution = resolution * (1.0 - 4.0 * eps);
        resolution_pass = k_box <= resolution;
      }
      const bool pass = uniform_pass || quotient_pass || resolution_pass;

      const int node_id = static_cast<int>(nodes.size());
      nodes.push_back({-1, pass, quotient_pass});
      if (latest_node[static_cast<std::size_t>(root0)] >= 0) {
        nodes[static_cast<std::size_t>(
                  latest_node[static_cast<std::size_t>(root0)])]
            .parent = node_id;
      }
      if (latest_node[static_cast<std::size_t>(root1)] >= 0) {
        nodes[static_cast<std::size_t>(
                  latest_node[static_cast<std::size_t>(root1)])]
            .parent = node_id;
      }
      parent[large_index] = small_root;
      mass[small_index] = mass_sum;
      off[small_index] = off_merged;
      energy[small_index] = energy_merged;
      tree[small_index] = tree_merged;
      count[small_index] = count_sum;
      vmin[small_index] = vmin_merged;
      vmax[small_index] = vmax_merged;
      latest_node[small_index] = node_id;
      edge_node[edge_index] = node_id;
    }

    void settle_adoption() {
      adopted.assign(nodes.size(), 0);
      adopted_quotient.assign(nodes.size(), 0);
      for (std::size_t node = nodes.size(); node-- > 0;) {
        const Node& tree_node = nodes[node];
        if (tree_node.parent >= 0 &&
            adopted[static_cast<std::size_t>(tree_node.parent)]) {
          adopted[node] = 1;
          adopted_quotient[node] =
              adopted_quotient[static_cast<std::size_t>(tree_node.parent)];
        } else if (tree_node.pass) {
          adopted[node] = 1;
          adopted_quotient[node] = static_cast<char>(tree_node.quotient);
        }
      }
    }
  };

  Direction r_direction(node_vr, nodal_mass, node_count, budgets.size());
  Direction z_direction(node_vz, nodal_mass, node_count, budgets.size());
  for (const std::size_t edge_index : order) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    const double beta_upper = budget_upper_bound(edge);
    const bool axis_edge = target_mesh.node_r[edge.node0] == 0.0 ||
                           target_mesh.node_r[edge.node1] == 0.0;
    z_direction.process_edge(edge_index, edge, beta_upper, eps);
    if (!axis_edge) {
      r_direction.process_edge(edge_index, edge, beta_upper, eps);
    }
  }
  r_direction.settle_adoption();
  z_direction.settle_adoption();

  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (edge.walk_failed) {
      consensus_constrained[edge_index] = false;
      continue;
    }
    if (edge.budget == 0.0) {
      consensus_constrained[edge_index] = true;
      continue;
    }
    const bool mass_ok = nodal_mass[edge.node0] > 0.0 &&
                         nodal_mass[edge.node1] > 0.0;
    const bool axis_edge = target_mesh.node_r[edge.node0] == 0.0 ||
                           target_mesh.node_r[edge.node1] == 0.0;
    const int z_node = z_direction.edge_node[edge_index];
    bool certified =
        mass_ok && z_node >= 0 &&
        z_direction.adopted[static_cast<std::size_t>(z_node)];
    bool quotient_label =
        certified &&
        z_direction.adopted_quotient[static_cast<std::size_t>(z_node)];
    if (certified && !axis_edge) {
      const int r_node = r_direction.edge_node[edge_index];
      certified =
          r_node >= 0 &&
          r_direction.adopted[static_cast<std::size_t>(r_node)];
      quotient_label =
          certified && quotient_label &&
          r_direction.adopted_quotient[static_cast<std::size_t>(r_node)];
    }
    consensus_constrained[edge_index] = certified;
    if (certified) {
      if (quotient_label) {
        ++counters.quotient_certified_edges;
      } else {
        ++counters.resolution_certified_edges;
      }
    }
  }
  // B' LATTICE_ZERO (consult-33 §5.6, user-approved 2026-08-15): a budget
  // below the smallest nonzero representable velocity difference at the
  // endpoints admits no representable du != 0 — equality is the only
  // representable feasible point.  min over both endpoints and both
  // components is the conservative (under-admitting) lattice floor: a
  // cross-binade pair can realize the finer operand's spacing.  Guard
  // (§5.6): admission stands on this lattice proof alone — never admit an
  // edge merely because the homothety fallback would flatten it.
  const auto ulp_gap = [](const double value) {
    const double magnitude = std::fabs(value);
    return std::nextafter(magnitude,
                          std::numeric_limits<double>::infinity()) -
           magnitude;
  };
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (edge.walk_failed || consensus_constrained[edge_index]) {
      continue;
    }
    const double floor_r =
        std::min(ulp_gap(node_vr[edge.node0]), ulp_gap(node_vr[edge.node1]));
    const double floor_z =
        std::min(ulp_gap(node_vz[edge.node0]), ulp_gap(node_vz[edge.node1]));
    const double lattice_floor = std::min(floor_r, floor_z);
    if (budget_upper_bound(edge) < lattice_floor) {
      consensus_constrained[edge_index] = true;
      ++counters.lattice_certified_edges;
    }
  }
  return counters;
}

}  // namespace

DvclpProjectionResult project_dvclp_velocities_core(
    const RemapMesh& target_mesh,
    const std::vector<DvclpEdgeBudget>& budgets,
    const double* nodal_mass,
    const double* corner_mass,
    double* corner_momentum_r,
    double* corner_momentum_z,
    double* node_vr,
    double* node_vz,
    double* cell_q_projection,
    const DvclpTestCaps& test_caps,
    const int solver_rev) {
  DvclpProjectionResult result;
  using DvclpClock = std::chrono::steady_clock;
  const auto elapsed_ms = [](const DvclpClock::time_point start) {
    return static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                DvclpClock::now() - start)
                                .count());
  };
  // Steady-clock measurements are wall-only telemetry and do not affect deterministic computed values.
  const DvclpClock::time_point consensus_start = DvclpClock::now();
  std::size_t node_count = 0;
  for (int cell = 0; cell < target_mesh.n_cells; ++cell) {
    const int begin = target_mesh.csr_offsets[cell];
    const int nverts = static_cast<int>(target_mesh.nverts[cell]);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = target_mesh.csr_indices[begin + corner];
      node_count = std::max(node_count, static_cast<std::size_t>(node + 1));
    }
  }
  std::vector<char> consensus_constrained(budgets.size(), false);
  // SS8-w2fix23: local, Galilean-consistent, component-proof classification
  // (exact zero contraction + eq. 52' representable quotient + ledger-resolution
  // collapse); see docs/design/dvclp_eq52_floor_20260813.md.
  const DvclpClassifyCounters classify_counters =
      classify_dvclp_consensus_edges(
          target_mesh, budgets, nodal_mass, node_vr, node_vz, node_count,
          consensus_constrained);
  result.quotient_certified_edges =
      classify_counters.quotient_certified_edges;
  result.resolution_certified_edges =
      classify_counters.resolution_certified_edges;
  result.lattice_certified_edges =
      classify_counters.lattice_certified_edges;
  result.consensus_edges = static_cast<std::size_t>(std::count(
      consensus_constrained.begin(), consensus_constrained.end(), true));
  std::vector<bool> representation_converged(budgets.size(), false);
  for (const DvclpEdgeBudget& edge : budgets) {
    if (target_mesh.node_r[edge.node0] == 0.0) {
      TENRYU_ASSERT(node_vr[edge.node0] == 0.0,
                    "DVCLP axis-node radial velocity must be zero");
    }
    if (target_mesh.node_r[edge.node1] == 0.0) {
      TENRYU_ASSERT(node_vr[edge.node1] == 0.0,
                    "DVCLP axis-node radial velocity must be zero");
    }
  }
  for (const DvclpEdgeBudget& edge : budgets) {
    if (edge.walk_failed) {
      ++result.unconstrained;
    }
  }
  count_projection_violations(
      target_mesh, budgets, consensus_constrained,
      representation_converged, node_vr, node_vz, result.max_R_before);
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (edge.walk_failed || !(nodal_mass[edge.node0] > 0.0) ||
        !(nodal_mass[edge.node1] > 0.0) ||
        corner_reduced_mass_sum(edge, corner_mass) != 0.0) {
      continue;
    }
    char buffer[192];
    std::snprintf(
        buffer, sizeof(buffer),
        "dvclp_corner_mass_inconsistent: edge=%zu n0=%d n1=%d",
        edge_index, edge.node0, edge.node1);
    result.ok = false;
    result.reason = buffer;
    result.consensus_ms = elapsed_ms(consensus_start);
    return result;
  }

  const DvclpConsensusGraph consensus_graph =
      build_consensus_graph(target_mesh, budgets, consensus_constrained);
  result.consensus_components =
      consensus_graph.component_roots.size();
  for (const int root : consensus_graph.component_roots) {
    result.consensus_nodes +=
        consensus_graph.members[static_cast<std::size_t>(root)].size();
  }
  std::vector<bool> affected_consensus_components(
      consensus_graph.parent.size(), false);
  for (const int root : consensus_graph.component_roots) {
    affected_consensus_components[static_cast<std::size_t>(root)] = true;
  }

  std::vector<int> unit_z(node_count, 0);
  std::vector<int> unit_r(node_count, 0);
  std::vector<double> unit_mass_z(node_count, 0.0);
  std::vector<double> unit_mass_r(node_count, 0.0);
  for (std::size_t node = 0; node < node_count; ++node) {
    const int root_z =
        consensus_graph.parent[static_cast<std::size_t>(node)];
    const int root_r =
        consensus_graph.parent_vr[static_cast<std::size_t>(node)];
    unit_z[node] =
        consensus_graph.members[static_cast<std::size_t>(root_z)].size() >=
                2U
            ? root_z
            : static_cast<int>(node);
    unit_r[node] =
        target_mesh.node_r[node] != 0.0 &&
                consensus_graph
                        .members_vr[static_cast<std::size_t>(root_r)]
                        .size() >= 2U
            ? root_r
            : static_cast<int>(node);
    unit_mass_z[node] = nodal_mass[node];
    unit_mass_r[node] = nodal_mass[node];
  }
  for (std::size_t root = 0; root < node_count; ++root) {
    const std::vector<int>& members_z = consensus_graph.members[root];
    if (members_z.size() >= 2U) {
      long double mass_sum = 0.0L;
      for (const int node : members_z) {
        mass_sum += static_cast<long double>(nodal_mass[node]);
      }
      unit_mass_z[root] = static_cast<double>(mass_sum);
    }
    const std::vector<int>& members_r = consensus_graph.members_vr[root];
    if (members_r.size() >= 2U) {
      long double mass_sum = 0.0L;
      for (const int node : members_r) {
        mass_sum += static_cast<long double>(nodal_mass[node]);
      }
      unit_mass_r[root] = static_cast<double>(mass_sum);
    }
  }

  std::vector<double> offset_vr(node_count, 0.0);
  std::vector<double> offset_vz(node_count, 0.0);
  std::vector<bool> unit_touched_r(node_count, false);
  std::vector<bool> unit_touched_z(node_count, false);
  std::vector<int> touched_units_r;
  std::vector<int> touched_units_z;
  const auto effective_projection_jump =
      [&](const DvclpEdgeBudget& edge, double& delta_vr,
          double& delta_vz) {
        const bool axis0 = target_mesh.node_r[edge.node0] == 0.0;
        const bool axis1 = target_mesh.node_r[edge.node1] == 0.0;
        const double effective_vr0 =
            node_vr[edge.node0] +
            offset_vr[static_cast<std::size_t>(
                unit_r[static_cast<std::size_t>(edge.node0)])];
        const double effective_vr1 =
            node_vr[edge.node1] +
            offset_vr[static_cast<std::size_t>(
                unit_r[static_cast<std::size_t>(edge.node1)])];
        const double effective_vz0 =
            node_vz[edge.node0] +
            offset_vz[static_cast<std::size_t>(
                unit_z[static_cast<std::size_t>(edge.node0)])];
        const double effective_vz1 =
            node_vz[edge.node1] +
            offset_vz[static_cast<std::size_t>(
                unit_z[static_cast<std::size_t>(edge.node1)])];
        delta_vr =
            axis0 || axis1 ? 0.0 : effective_vr1 - effective_vr0;
        delta_vz = effective_vz1 - effective_vz0;
        return std::hypot(delta_vr, delta_vz);
      };

  result.alternations = 1;
  if (!consensus_graph.component_roots.empty()) {
    if (!apply_consensus_components(
            target_mesh, consensus_graph,
            affected_consensus_components, nodal_mass, corner_mass,
            corner_momentum_r, corner_momentum_z, node_vr, node_vz,
            cell_q_projection, result)) {
      result.consensus_ms = elapsed_ms(consensus_start);
      return result;
    }
  }
  result.consensus_ms = elapsed_ms(consensus_start);
  // Sweep-invariant per-edge data, gathered once so the 64-sweep scan does
  // not stream the 96-byte DvclpEdgeBudget records nor the member-list
  // headers every pass. Values are computed by the same expressions the
  // scan used inline; the scan decisions are therefore unchanged.
  std::vector<char> scan_skip_static(budgets.size());
  std::vector<int> scan_node0(budgets.size());
  std::vector<int> scan_node1(budgets.size());
  std::vector<char> scan_direct_eligible(budgets.size());
  std::vector<char> scan_axis_pair(budgets.size());
  std::vector<double> scan_bound(budgets.size());
  std::vector<double> scan_bound2(budgets.size());
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    scan_skip_static[edge_index] =
        edge.walk_failed || consensus_constrained[edge_index];
    scan_node0[edge_index] = edge.node0;
    scan_node1[edge_index] = edge.node1;
    const int unit_z0 = unit_z[static_cast<std::size_t>(edge.node0)];
    const int unit_z1 = unit_z[static_cast<std::size_t>(edge.node1)];
    const int unit_r0 = unit_r[static_cast<std::size_t>(edge.node0)];
    const int unit_r1 = unit_r[static_cast<std::size_t>(edge.node1)];
    scan_direct_eligible[edge_index] =
        unit_z0 == edge.node0 && unit_r0 == edge.node0 &&
        unit_z1 == edge.node1 && unit_r1 == edge.node1 &&
        consensus_graph.members[static_cast<std::size_t>(unit_z0)]
                .size() < 2U &&
        consensus_graph.members_vr[static_cast<std::size_t>(unit_r0)]
                .size() < 2U &&
        consensus_graph.members[static_cast<std::size_t>(unit_z1)]
                .size() < 2U &&
        consensus_graph.members_vr[static_cast<std::size_t>(unit_r1)]
                .size() < 2U;
    scan_axis_pair[edge_index] =
        target_mesh.node_r[edge.node0] == 0.0 ||
        target_mesh.node_r[edge.node1] == 0.0;
    scan_bound[edge_index] = budget_upper_bound(edge);
    scan_bound2[edge_index] =
        scan_bound[edge_index] * scan_bound[edge_index];
  }
  bool sweep_offsets_all_zero = true;
  const DvclpClock::time_point sweep_start = DvclpClock::now();
  std::fill(representation_converged.begin(),
            representation_converged.end(), false);
  std::fill(offset_vr.begin(), offset_vr.end(), 0.0);
  std::fill(offset_vz.begin(), offset_vz.end(), 0.0);
  std::fill(unit_touched_r.begin(), unit_touched_r.end(), false);
  std::fill(unit_touched_z.begin(), unit_touched_z.end(), false);
  touched_units_r.clear();
  touched_units_z.clear();

  // Deterministic Gauss-Seidel variant of the consult's colored cyclic
  // projection: every update is visible to later edges in stable Stage-1
  // enumeration order.  Equality-constrained edges are handled only by
  // exact component consensus; the convex-feasibility argument applies to
  // the remaining constraints.
  for (int sweep = 0; sweep < test_caps.quotient_sweep_cap; ++sweep) {
      int sweep_events = 0;
      for (std::size_t edge_index = 0; edge_index < budgets.size();
           ++edge_index) {
        if (scan_skip_static[edge_index] ||
            representation_converged[edge_index]) {
          continue;
        }
        const DvclpEdgeBudget& edge = budgets[edge_index];
        const int node0 = scan_node0[edge_index];
        const int node1 = scan_node1[edge_index];
        const int unit_z0 = unit_z[static_cast<std::size_t>(node0)];
        const int unit_z1 = unit_z[static_cast<std::size_t>(node1)];
        const int unit_r0 = unit_r[static_cast<std::size_t>(node0)];
        const int unit_r1 = unit_r[static_cast<std::size_t>(node1)];
        const bool direct_update = scan_direct_eligible[edge_index] != 0;
        // sweep_offsets_all_zero == true certifies every offset slot still
        // holds 0.0, so the four explicit reads below are redundant then;
        // the predicate value is unchanged either way.
        const bool direct_offsets_zero =
            direct_update &&
            (sweep_offsets_all_zero ||
             (offset_vr[static_cast<std::size_t>(unit_r0)] == 0.0 &&
              offset_vr[static_cast<std::size_t>(unit_r1)] == 0.0 &&
              offset_vz[static_cast<std::size_t>(unit_z0)] == 0.0 &&
              offset_vz[static_cast<std::size_t>(unit_z1)] == 0.0));
        // Deltas: identical expressions to projection_jump /
        // effective_projection_jump, evaluated inline so the certified
        // screen below can defer the hypot.
        const bool axis_pair = scan_axis_pair[edge_index] != 0;
        double delta_vr0 = 0.0;
        double delta_vz0 = 0.0;
        if (direct_offsets_zero) {
          delta_vr0 =
              axis_pair ? 0.0 : node_vr[node1] - node_vr[node0];
          delta_vz0 = node_vz[node1] - node_vz[node0];
        } else {
          const double effective_vr0 =
              node_vr[node0] +
              offset_vr[static_cast<std::size_t>(unit_r0)];
          const double effective_vr1 =
              node_vr[node1] +
              offset_vr[static_cast<std::size_t>(unit_r1)];
          const double effective_vz0 =
              node_vz[node0] +
              offset_vz[static_cast<std::size_t>(unit_z0)];
          const double effective_vz1 =
              node_vz[node1] +
              offset_vz[static_cast<std::size_t>(unit_z1)];
          delta_vr0 =
              axis_pair ? 0.0 : effective_vr1 - effective_vr0;
          delta_vz0 = effective_vz1 - effective_vz0;
        }
        // Certified screen for the decision `hypot(delta) > bound`:
        // squared magnitudes with a 32*eps relative band. The band
        // dominates the <=3*eps relative error of s and the <=2*eps
        // relative error of std::hypot, so a fast-path verdict always
        // matches the exact comparison; inside the band (and for any
        // non-normal operand) the original hypot comparison decides.
        const double screen_s =
            delta_vr0 * delta_vr0 + delta_vz0 * delta_vz0;
        const double screen_b2 = scan_bound2[edge_index];
        constexpr double kScreenBand =
            32.0 * std::numeric_limits<double>::epsilon();
        double jump0;
        if (std::isfinite(screen_s) &&
            screen_s >= std::numeric_limits<double>::min() &&
            std::isfinite(screen_b2) &&
            screen_b2 >= std::numeric_limits<double>::min() &&
            screen_s < screen_b2 * (1.0 - kScreenBand)) {
          continue;
        } else if (std::isfinite(screen_s) &&
                   screen_s >= std::numeric_limits<double>::min() &&
                   std::isfinite(screen_b2) &&
                   screen_b2 >= std::numeric_limits<double>::min() &&
                   screen_s > screen_b2 * (1.0 + kScreenBand)) {
          jump0 = std::hypot(delta_vr0, delta_vz0);
        } else {
          jump0 = std::hypot(delta_vr0, delta_vz0);
          if (!(jump0 > scan_bound[edge_index])) {
            continue;
          }
        }

        double projection_q = 0.0;
        if (direct_update) {
          const double mass0 = nodal_mass[node0];
          const double mass1 = nodal_mass[node1];
          const double mass_sum = mass0 + mass1;
          const double node0_vr0 = node_vr[node0];
          const double node0_vz0 = node_vz[node0];
          const double node1_vr0 = node_vr[node1];
          const double node1_vz0 = node_vz[node1];
          const double effective_node0_vr0 =
              direct_offsets_zero
                  ? node0_vr0
                  : node0_vr0 +
                        offset_vr[static_cast<std::size_t>(unit_r0)];
          const double effective_node0_vz0 =
              direct_offsets_zero
                  ? node0_vz0
                  : node0_vz0 +
                        offset_vz[static_cast<std::size_t>(unit_z0)];
          const double effective_node1_vr0 =
              direct_offsets_zero
                  ? node1_vr0
                  : node1_vr0 +
                        offset_vr[static_cast<std::size_t>(unit_r1)];
          const double effective_node1_vz0 =
              direct_offsets_zero
                  ? node1_vz0
                  : node1_vz0 +
                        offset_vz[static_cast<std::size_t>(unit_z1)];
          double q_final = edge.budget / jump0;
          double projected_delta_vr = 0.0;
          double projected_delta_vz = 0.0;
          // Consult section 9.2: scale inward by representable units, not a
          // dimensionless epsilon.
          for (int inward_step = 0; inward_step <= 32; ++inward_step) {
            const double one_minus_q = 1.0 - q_final;
            const double node0_vr_updated =
                node0_vr0 + mass1 / mass_sum * one_minus_q * delta_vr0;
            const double node0_vz_updated =
                node0_vz0 + mass1 / mass_sum * one_minus_q * delta_vz0;
            const double node1_vr_updated =
                node1_vr0 - mass0 / mass_sum * one_minus_q * delta_vr0;
            const double node1_vz_updated =
                node1_vz0 - mass0 / mass_sum * one_minus_q * delta_vz0;
            const double effective_node0_vr_updated =
                direct_offsets_zero
                    ? node0_vr_updated
                    : node0_vr_updated +
                          offset_vr[static_cast<std::size_t>(unit_r0)];
            const double effective_node0_vz_updated =
                direct_offsets_zero
                    ? node0_vz_updated
                    : node0_vz_updated +
                          offset_vz[static_cast<std::size_t>(unit_z0)];
            const double effective_node1_vr_updated =
                direct_offsets_zero
                    ? node1_vr_updated
                    : node1_vr_updated +
                          offset_vr[static_cast<std::size_t>(unit_r1)];
            const double effective_node1_vz_updated =
                direct_offsets_zero
                    ? node1_vz_updated
                    : node1_vz_updated +
                          offset_vz[static_cast<std::size_t>(unit_z1)];
            // This discrete, no-tunables criterion is consistent with
            // consult section 9.2: a violation smaller than one representable
            // unit of the constrained quantities is certified at the
            // representation level.
            if (bitwise_equal(effective_node0_vr_updated,
                              effective_node0_vr0) &&
                bitwise_equal(effective_node0_vz_updated,
                              effective_node0_vz0) &&
                bitwise_equal(effective_node1_vr_updated,
                              effective_node1_vr0) &&
                bitwise_equal(effective_node1_vz_updated,
                              effective_node1_vz0)) {
              node_vr[node0] = node0_vr0;
              node_vz[node0] = node0_vz0;
              node_vr[node1] = node1_vr0;
              node_vz[node1] = node1_vz0;
              representation_converged[edge_index] = true;
              break;
            }
            node_vr[node0] = node0_vr_updated;
            node_vz[node0] = node0_vz_updated;
            node_vr[node1] = node1_vr_updated;
            node_vz[node1] = node1_vz_updated;
            const double projected_jump =
                direct_offsets_zero
                    ? projection_jump(target_mesh, edge, node_vr, node_vz,
                                      projected_delta_vr,
                                      projected_delta_vz)
                    : effective_projection_jump(
                          edge, projected_delta_vr, projected_delta_vz);
            if (!(projected_jump > budget_upper_bound(edge)) ||
                inward_step == 32) {
              break;
            }
            q_final = std::nextafter(q_final, 0.0);
          }
          if (representation_converged[edge_index]) {
            continue;
          }

          const double one_minus_q = 1.0 - q_final;
          const double reduced_mass = mass0 * mass1 / mass_sum;
          const double impulse_r =
              reduced_mass * one_minus_q * delta_vr0;
          const double impulse_z =
              reduced_mass * one_minus_q * delta_vz0;
          const double projected_target = q_final * jump0;
          projection_q =
              0.5 * reduced_mass *
              (jump0 * jump0 - projected_target * projected_target);

          const double corner_weight_sum =
              corner_reduced_mass_sum(edge, corner_mass);
          for (int incidence_index = 0;
               incidence_index < edge.incident_count; ++incidence_index) {
            const DvclpEdgeIncidence& incidence =
                edge.incident_corners[
                    static_cast<std::size_t>(incidence_index)];
            const double corner_mass0 = corner_mass[incidence.corner0];
            const double corner_mass1 = corner_mass[incidence.corner1];
            const double denominator = corner_mass0 + corner_mass1;
            const double weight =
                denominator > 0.0
                    ? corner_mass0 * corner_mass1 / denominator
                    : 0.0;
            const double theta = weight / corner_weight_sum;
            corner_momentum_r[incidence.corner0] += theta * impulse_r;
            corner_momentum_z[incidence.corner0] += theta * impulse_z;
            corner_momentum_r[incidence.corner1] -= theta * impulse_r;
            corner_momentum_z[incidence.corner1] -= theta * impulse_z;
            cell_q_projection[incidence.cell] += theta * projection_q;
          }
        } else {
          const bool axis_edge =
              target_mesh.node_r[node0] == 0.0 ||
              target_mesh.node_r[node1] == 0.0;
          const double offset_vr0 =
              offset_vr[static_cast<std::size_t>(unit_r0)];
          const double offset_vr1 =
              offset_vr[static_cast<std::size_t>(unit_r1)];
          const double offset_vz0 =
              offset_vz[static_cast<std::size_t>(unit_z0)];
          const double offset_vz1 =
              offset_vz[static_cast<std::size_t>(unit_z1)];
          const double effective_node0_vr0 = node_vr[node0] + offset_vr0;
          const double effective_node0_vz0 = node_vz[node0] + offset_vz0;
          const double effective_node1_vr0 = node_vr[node1] + offset_vr1;
          const double effective_node1_vz0 = node_vz[node1] + offset_vz1;
          const double mass_r0 =
              unit_mass_r[static_cast<std::size_t>(unit_r0)];
          const double mass_r1 =
              unit_mass_r[static_cast<std::size_t>(unit_r1)];
          const double mass_z0 =
              unit_mass_z[static_cast<std::size_t>(unit_z0)];
          const double mass_z1 =
              unit_mass_z[static_cast<std::size_t>(unit_z1)];
          const double mass_sum_r = mass_r0 + mass_r1;
          const double mass_sum_z = mass_z0 + mass_z1;
          double q_final = edge.budget / jump0;
          double candidate_delta_vr = 0.0;
          double candidate_delta_vz = 0.0;
          for (int inward_step = 0; inward_step <= 32; ++inward_step) {
            const double one_minus_q = 1.0 - q_final;
            double offset_vr0_updated = offset_vr0;
            double offset_vr1_updated = offset_vr1;
            double offset_vz0_updated = offset_vz0;
            double offset_vz1_updated = offset_vz1;
            if (!axis_edge && unit_r0 != unit_r1) {
              offset_vr0_updated +=
                  mass_r1 / mass_sum_r * one_minus_q * delta_vr0;
              offset_vr1_updated -=
                  mass_r0 / mass_sum_r * one_minus_q * delta_vr0;
            }
            if (unit_z0 != unit_z1) {
              offset_vz0_updated +=
                  mass_z1 / mass_sum_z * one_minus_q * delta_vz0;
              offset_vz1_updated -=
                  mass_z0 / mass_sum_z * one_minus_q * delta_vz0;
            }
            const double effective_node0_vr_updated =
                node_vr[node0] + offset_vr0_updated;
            const double effective_node0_vz_updated =
                node_vz[node0] + offset_vz0_updated;
            const double effective_node1_vr_updated =
                node_vr[node1] + offset_vr1_updated;
            const double effective_node1_vz_updated =
                node_vz[node1] + offset_vz1_updated;
            if (bitwise_equal(effective_node0_vr_updated,
                              effective_node0_vr0) &&
                bitwise_equal(effective_node0_vz_updated,
                              effective_node0_vz0) &&
                bitwise_equal(effective_node1_vr_updated,
                              effective_node1_vr0) &&
                bitwise_equal(effective_node1_vz_updated,
                              effective_node1_vz0)) {
              offset_vr[static_cast<std::size_t>(unit_r0)] = offset_vr0;
              sweep_offsets_all_zero = false;
              offset_vr[static_cast<std::size_t>(unit_r1)] = offset_vr1;
              sweep_offsets_all_zero = false;
              offset_vz[static_cast<std::size_t>(unit_z0)] = offset_vz0;
              sweep_offsets_all_zero = false;
              offset_vz[static_cast<std::size_t>(unit_z1)] = offset_vz1;
              sweep_offsets_all_zero = false;
              representation_converged[edge_index] = true;
              break;
            }
            if (!axis_edge && unit_r0 != unit_r1) {
              offset_vr[static_cast<std::size_t>(unit_r0)] =
                  offset_vr0_updated;
              sweep_offsets_all_zero = false;
              offset_vr[static_cast<std::size_t>(unit_r1)] =
                  offset_vr1_updated;
              sweep_offsets_all_zero = false;
              if (!unit_touched_r[static_cast<std::size_t>(unit_r0)]) {
                unit_touched_r[static_cast<std::size_t>(unit_r0)] = true;
                touched_units_r.push_back(unit_r0);
              }
              if (!unit_touched_r[static_cast<std::size_t>(unit_r1)]) {
                unit_touched_r[static_cast<std::size_t>(unit_r1)] = true;
                touched_units_r.push_back(unit_r1);
              }
            }
            if (unit_z0 != unit_z1) {
              offset_vz[static_cast<std::size_t>(unit_z0)] =
                  offset_vz0_updated;
              sweep_offsets_all_zero = false;
              offset_vz[static_cast<std::size_t>(unit_z1)] =
                  offset_vz1_updated;
              sweep_offsets_all_zero = false;
              if (!unit_touched_z[static_cast<std::size_t>(unit_z0)]) {
                unit_touched_z[static_cast<std::size_t>(unit_z0)] = true;
                touched_units_z.push_back(unit_z0);
              }
              if (!unit_touched_z[static_cast<std::size_t>(unit_z1)]) {
                unit_touched_z[static_cast<std::size_t>(unit_z1)] = true;
                touched_units_z.push_back(unit_z1);
              }
            }
            const double projected_jump = effective_projection_jump(
                edge, candidate_delta_vr, candidate_delta_vz);
            if (!(projected_jump > budget_upper_bound(edge)) ||
                inward_step == 32) {
              break;
            }
            q_final = std::nextafter(q_final, 0.0);
          }
          if (representation_converged[edge_index]) {
            continue;
          }

          const double projected_delta_vr = q_final * delta_vr0;
          const double projected_delta_vz = q_final * delta_vz0;
          if (!axis_edge && unit_r0 != unit_r1) {
            const double reduced_mass_r =
                mass_r0 * mass_r1 / mass_sum_r;
            projection_q +=
                0.5 * reduced_mass_r *
                (delta_vr0 * delta_vr0 -
                 projected_delta_vr * projected_delta_vr);
          }
          if (unit_z0 != unit_z1) {
            const double reduced_mass_z =
                mass_z0 * mass_z1 / mass_sum_z;
            projection_q +=
                0.5 * reduced_mass_z *
                (delta_vz0 * delta_vz0 -
                 projected_delta_vz * projected_delta_vz);
          }

          const double corner_weight_sum =
              corner_reduced_mass_sum(edge, corner_mass);
          for (int incidence_index = 0;
               incidence_index < edge.incident_count; ++incidence_index) {
            const DvclpEdgeIncidence& incidence =
                edge.incident_corners[
                    static_cast<std::size_t>(incidence_index)];
            const double corner_mass0 = corner_mass[incidence.corner0];
            const double corner_mass1 = corner_mass[incidence.corner1];
            const double denominator = corner_mass0 + corner_mass1;
            const double weight =
                denominator > 0.0
                    ? corner_mass0 * corner_mass1 / denominator
                    : 0.0;
            const double theta = weight / corner_weight_sum;
            cell_q_projection[incidence.cell] += theta * projection_q;
          }
        }
        result.q_total += projection_q;
        ++result.events;
        ++sweep_events;
      }
      ++result.sweeps;
      if (sweep_events == 0) {
        break;
      }
  }

  std::sort(touched_units_z.begin(), touched_units_z.end());
  for (const int unit : touched_units_z) {
      const double delta = offset_vz[static_cast<std::size_t>(unit)];
      if (delta != 0.0) {
        const std::vector<int>& members =
            consensus_graph.members[static_cast<std::size_t>(unit)];
        if (members.size() >= 2U) {
          for (const int node : members) {
            const double delta_momentum = nodal_mass[node] * delta;
            for (const DvclpNodeCorner& incidence :
                 consensus_graph
                     .node_corners[static_cast<std::size_t>(node)]) {
              const double fraction =
                  corner_mass[incidence.corner] / nodal_mass[node];
              corner_momentum_z[incidence.corner] +=
                  fraction * delta_momentum;
            }
            node_vz[node] += delta;
          }
        } else {
          const int node = unit;
          const double delta_momentum = nodal_mass[node] * delta;
          for (const DvclpNodeCorner& incidence :
               consensus_graph.node_corners[static_cast<std::size_t>(node)]) {
            const double fraction =
                corner_mass[incidence.corner] / nodal_mass[node];
            corner_momentum_z[incidence.corner] +=
                fraction * delta_momentum;
          }
          node_vz[node] += delta;
        }
      }
      offset_vz[static_cast<std::size_t>(unit)] = 0.0;
      unit_touched_z[static_cast<std::size_t>(unit)] = false;
  }
  touched_units_z.clear();

  std::sort(touched_units_r.begin(), touched_units_r.end());
  for (const int unit : touched_units_r) {
      const double delta = offset_vr[static_cast<std::size_t>(unit)];
      if (delta != 0.0) {
        const std::vector<int>& members =
            consensus_graph.members_vr[static_cast<std::size_t>(unit)];
        if (members.size() >= 2U) {
          for (const int node : members) {
            const double delta_momentum = nodal_mass[node] * delta;
            for (const DvclpNodeCorner& incidence :
                 consensus_graph
                     .node_corners[static_cast<std::size_t>(node)]) {
              const double fraction =
                  corner_mass[incidence.corner] / nodal_mass[node];
              corner_momentum_r[incidence.corner] +=
                  fraction * delta_momentum;
            }
            node_vr[node] += delta;
          }
        } else {
          const int node = unit;
          const double delta_momentum = nodal_mass[node] * delta;
          for (const DvclpNodeCorner& incidence :
               consensus_graph.node_corners[static_cast<std::size_t>(node)]) {
            const double fraction =
                corner_mass[incidence.corner] / nodal_mass[node];
            corner_momentum_r[incidence.corner] +=
                fraction * delta_momentum;
          }
          node_vr[node] += delta;
        }
      }
      offset_vr[static_cast<std::size_t>(unit)] = 0.0;
      unit_touched_r[static_cast<std::size_t>(unit)] = false;
  }
  touched_units_r.clear();

  result.sweep_ms = elapsed_ms(sweep_start);
  const DvclpClock::time_point solve_start = DvclpClock::now();
  if (!consensus_graph.component_roots.empty()) {
    mark_affected_consensus_components(
        target_mesh, budgets, consensus_constrained, consensus_graph,
        node_vr, node_vz, affected_consensus_components);
  }
  std::vector<char> processed_inner_nodes;
  std::vector<char> screened_omitted;
  if (!apply_homothetic_completion(
          target_mesh, budgets, consensus_constrained, consensus_graph,
          unit_z, unit_r, unit_mass_z, unit_mass_r, nodal_mass, corner_mass,
          corner_momentum_r, corner_momentum_z, node_vr, node_vz,
          cell_q_projection, processed_inner_nodes, screened_omitted,
          result, test_caps, solver_rev)) {
    result.solve_ms = elapsed_ms(solve_start);
    return result;
  }

  result.solve_ms = elapsed_ms(solve_start);
  const DvclpClock::time_point certify_start = DvclpClock::now();
  double remaining_max_ratio = 0.0;
  const std::size_t remaining = count_projection_violations(
      target_mesh, budgets, consensus_constrained,
      representation_converged, node_vr, node_vz, remaining_max_ratio);
  if (remaining != 0U) {
    char buffer[192];
    std::snprintf(
        buffer, sizeof(buffer),
        "dvclp_projection_unconverged: remaining=%zu max_R=%.17g",
        remaining, remaining_max_ratio);
    result.ok = false;
    result.reason = buffer;
    result.certify_ms = elapsed_ms(certify_start);
    std::size_t printed = 0;
    for (std::size_t edge_index = 0;
         edge_index < budgets.size() && printed < 4U; ++edge_index) {
      const DvclpEdgeBudget& edge = budgets[edge_index];
      if (edge.walk_failed || representation_converged[edge_index]) {
        continue;
      }
      double delta_vr = 0.0;
      double delta_vz = 0.0;
      const double jump = projection_jump(
          target_mesh, edge, node_vr, node_vz, delta_vr, delta_vz);
      const bool zero_budget_satisfied =
          !consensus_constrained[edge_index] ||
          consensus_edge_satisfied(
              target_mesh, edge, node_vr, node_vz);
      const double upper_bound = budget_upper_bound(edge);
      if (zero_budget_satisfied && !(jump > upper_bound)) {
        continue;
      }
      const int node0 = edge.node0;
      const int node1 = edge.node1;
      std::fprintf(
          stderr,
          "[dvclp_reject_edge] edge_index=%zu node0=%d node1=%d "
          "node_r0=%.17g node_r1=%.17g axis0=%d axis1=%d "
          "budget=%.17g upper_bound=%.17g jump=%.17g ratio=%.17g "
          "consensus_constrained=%d unit_z0=%d unit_z1=%d "
          "unit_r0=%d unit_r1=%d walk_segments=%d "
          "inner0=%d inner1=%d screened_omitted=%d\n",
          edge_index, node0, node1, target_mesh.node_r[node0],
          target_mesh.node_r[node1],
          static_cast<int>(target_mesh.node_r[node0] == 0.0),
          static_cast<int>(target_mesh.node_r[node1] == 0.0),
          edge.budget, upper_bound, jump, jump / upper_bound,
          static_cast<int>(consensus_constrained[edge_index]),
          unit_z[static_cast<std::size_t>(node0)],
          unit_z[static_cast<std::size_t>(node1)],
          unit_r[static_cast<std::size_t>(node0)],
          unit_r[static_cast<std::size_t>(node1)], edge.walk_segments,
          static_cast<int>(
              processed_inner_nodes[static_cast<std::size_t>(node0)]),
          static_cast<int>(
              processed_inner_nodes[static_cast<std::size_t>(node1)]),
          static_cast<int>(screened_omitted[edge_index]));
      ++printed;
    }
    return result;
  }

  // Certification uses the same plain-double arithmetic as the projection.
  // With the section 9.2 inward scaling above, any genuine >1 residual takes
  // the reject path without an empirical epsilon.
  result.max_R_after = 0.0;
  for (std::size_t edge_index = 0; edge_index < budgets.size();
       ++edge_index) {
    const DvclpEdgeBudget& edge = budgets[edge_index];
    if (edge.walk_failed) {
      continue;
    }
    double delta_vr = 0.0;
    double delta_vz = 0.0;
    const double jump = projection_jump(
        target_mesh, edge, node_vr, node_vz, delta_vr, delta_vz);
    result.max_R_after = std::max(
        result.max_R_after, projection_ratio(jump, edge.budget));
    const bool jump_within_budget = jump <= budget_upper_bound(edge);
    const bool plain_certified =
        consensus_constrained[edge_index]
            ? consensus_edge_satisfied(
                  target_mesh, edge, node_vr, node_vz)
            : jump_within_budget;
    bool representation_certified = false;
    if (!jump_within_budget) {
      const int node0 = edge.node0;
      const int node1 = edge.node1;
      const int unit_z0 = unit_z[static_cast<std::size_t>(node0)];
      const int unit_z1 = unit_z[static_cast<std::size_t>(node1)];
      const int unit_r0 = unit_r[static_cast<std::size_t>(node0)];
      const int unit_r1 = unit_r[static_cast<std::size_t>(node1)];
      const double q = edge.budget / jump;
      const double one_minus_q = 1.0 - q;
      const bool direct_update =
          unit_z0 == node0 && unit_r0 == node0 && unit_z1 == node1 &&
          unit_r1 == node1 &&
          consensus_graph.members[static_cast<std::size_t>(unit_z0)].size() <
              2U &&
          consensus_graph.members_vr[static_cast<std::size_t>(unit_r0)]
                  .size() < 2U &&
          consensus_graph.members[static_cast<std::size_t>(unit_z1)].size() <
              2U &&
          consensus_graph.members_vr[static_cast<std::size_t>(unit_r1)]
                  .size() < 2U;
      if (direct_update) {
        const double mass0 = nodal_mass[node0];
        const double mass1 = nodal_mass[node1];
        const double mass_sum = mass0 + mass1;
        const double node0_vr_updated =
            node_vr[node0] +
            mass1 / mass_sum * one_minus_q * delta_vr;
        const double node0_vz_updated =
            node_vz[node0] +
            mass1 / mass_sum * one_minus_q * delta_vz;
        const double node1_vr_updated =
            node_vr[node1] -
            mass0 / mass_sum * one_minus_q * delta_vr;
        const double node1_vz_updated =
            node_vz[node1] -
            mass0 / mass_sum * one_minus_q * delta_vz;
        representation_certified =
            bitwise_equal(node0_vr_updated, node_vr[node0]) &&
            bitwise_equal(node0_vz_updated, node_vz[node0]) &&
            bitwise_equal(node1_vr_updated, node_vr[node1]) &&
            bitwise_equal(node1_vz_updated, node_vz[node1]);
      } else {
        const bool axis_edge =
            target_mesh.node_r[node0] == 0.0 ||
            target_mesh.node_r[node1] == 0.0;
        double offset_r0 = 0.0;
        double offset_r1 = 0.0;
        double offset_z0 = 0.0;
        double offset_z1 = 0.0;
        if (!axis_edge && unit_r0 != unit_r1) {
          const double mass_r0 =
              unit_mass_r[static_cast<std::size_t>(unit_r0)];
          const double mass_r1 =
              unit_mass_r[static_cast<std::size_t>(unit_r1)];
          const double mass_sum_r = mass_r0 + mass_r1;
          offset_r0 =
              mass_r1 / mass_sum_r * one_minus_q * delta_vr;
          offset_r1 =
              -mass_r0 / mass_sum_r * one_minus_q * delta_vr;
        }
        if (unit_z0 != unit_z1) {
          const double mass_z0 =
              unit_mass_z[static_cast<std::size_t>(unit_z0)];
          const double mass_z1 =
              unit_mass_z[static_cast<std::size_t>(unit_z1)];
          const double mass_sum_z = mass_z0 + mass_z1;
          offset_z0 =
              mass_z1 / mass_sum_z * one_minus_q * delta_vz;
          offset_z1 =
              -mass_z0 / mass_sum_z * one_minus_q * delta_vz;
        }
        const double node0_vr_updated = node_vr[node0] + offset_r0;
        const double node0_vz_updated = node_vz[node0] + offset_z0;
        const double node1_vr_updated = node_vr[node1] + offset_r1;
        const double node1_vz_updated = node_vz[node1] + offset_z1;
        representation_certified =
            bitwise_equal(node0_vr_updated, node_vr[node0]) &&
            bitwise_equal(node0_vz_updated, node_vz[node0]) &&
            bitwise_equal(node1_vr_updated, node_vr[node1]) &&
            bitwise_equal(node1_vz_updated, node_vz[node1]);
      }
      if (representation_certified) {
        ++result.representation_limited;
      }
    }
    const bool certified = plain_certified || representation_certified;
    if (!certified) {
      char buffer[224];
      std::snprintf(
          buffer, sizeof(buffer),
          "dvclp_certification_failed: edge=%zu n0=%d n1=%d "
          "D=%.17g B=%.17g R=%.17g",
          edge_index, edge.node0, edge.node1, jump, edge.budget,
          projection_ratio(jump, edge.budget));
      result.ok = false;
      result.reason = buffer;
      result.certify_ms = elapsed_ms(certify_start);
      return result;
    }
  }
  result.certify_ms = elapsed_ms(certify_start);
  return result;
}

}  // namespace tenryu::hydro::reale::detail
