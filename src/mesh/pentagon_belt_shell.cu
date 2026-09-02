#include "mesh/mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "core/axis_tolerance.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/hydro_multiblock_topology.cuh"
#include "mesh/spherical_wedge_geometry.cuh"

namespace tenryu::mesh {
namespace {

using GridSegment = tenryu::core::Config::MeshConfig::GridSegment;
using GradingConfig = tenryu::core::Config::MeshConfig::GradingConfig;

double graded_grid_gap_tolerance(const double r_max) {
  return 1.0e-14 * std::abs(r_max);
}

void graded_grid_validate_segments(const std::vector<GridSegment>& segments) {
  TENRYU_ASSERT(!segments.empty(), "graded grid requires at least one segment");
  const double r_max = segments.back().r_end;
  const double gap_tol = graded_grid_gap_tolerance(r_max);
  for (std::size_t s = 0; s < segments.size(); ++s) {
    const auto& seg = segments[s];
    TENRYU_ASSERT(seg.nr > 0, "graded grid requires each segment.nr > 0");
    TENRYU_ASSERT(seg.r_end > seg.r_start,
                  "graded grid requires r_end > r_start in every segment");
    if (s > 0) {
      const double gap = std::abs(seg.r_start - segments[s - 1].r_end);
      TENRYU_ASSERT(gap <= gap_tol,
                    "graded grid segment boundary gap between segment " +
                        std::to_string(s - 1) + " and " + std::to_string(s));
    }
  }
}

void graded_grid_validate_config(const GradingConfig& grading) {
  TENRYU_ASSERT(grading.edge_ratio > 0.0 && grading.edge_ratio < 1.0,
                "graded grid requires grading.edge_ratio in (0, 1)");
  TENRYU_ASSERT(grading.sg_order >= 2 && (grading.sg_order % 2) == 0,
                "graded grid requires grading.sg_order to be an even integer >= 2");
  TENRYU_ASSERT(grading.sg_sigma > 0.0 && grading.sg_sigma < 1.0,
                "graded grid requires grading.sg_sigma in (0, 1)");
}

void graded_grid_apply_boundary_targets(std::vector<double>& widths,
                                        const double length,
                                        const bool has_left_target,
                                        const double left_target,
                                        const bool has_right_target,
                                        const double right_target,
                                        const std::size_t segment_index) {
  TENRYU_ASSERT(!widths.empty(),
                "graded grid segment width array must be non-empty");
  const double tol = 1.0e-12 * std::max(1.0, std::abs(length));
  const int n = static_cast<int>(widths.size());

  if (!has_left_target && !has_right_target) {
    return;
  }

  if (n == 1) {
    const double target = has_left_target ? left_target : right_target;
    if (has_left_target && has_right_target) {
      TENRYU_ASSERT(
          std::abs(left_target - right_target) <= tol,
          "graded grid cannot satisfy both boundary width targets with nr=1 "
          "for segment " +
              std::to_string(segment_index));
    }
    TENRYU_ASSERT(
        std::abs(length - target) <= tol,
        "graded grid cannot satisfy boundary width target with nr=1 "
        "for segment " +
            std::to_string(segment_index));
    widths[0] = length;
    return;
  }

  if (has_left_target && has_right_target) {
    if (n == 2) {
      TENRYU_ASSERT(
          std::abs((left_target + right_target) - length) <= tol,
          "graded grid cannot satisfy both boundary width targets with nr=2 "
          "for segment " +
              std::to_string(segment_index));
      widths[0] = left_target;
      widths[1] = length - left_target;
      TENRYU_ASSERT(
          std::abs(widths[1] - right_target) <= tol,
          "graded grid nr=2 boundary correction drift exceeded tolerance");
      return;
    }

    const double interior_old = length - widths.front() - widths.back();
    const double interior_new = length - left_target - right_target;
    TENRYU_ASSERT(
        interior_old > 0.0,
        "graded grid interior width sum must be > 0 before correction");
    TENRYU_ASSERT(
        interior_new > 0.0,
        "graded grid interior width sum must remain > 0 after correction");
    const double scale = interior_new / interior_old;
    for (int k = 1; k < n - 1; ++k) {
      widths[static_cast<std::size_t>(k)] *= scale;
    }
    widths.front() = left_target;
    widths.back() = right_target;
    return;
  }

  if (has_left_target) {
    const double tail_old = length - widths.front();
    const double tail_new = length - left_target;
    TENRYU_ASSERT(
        tail_old > 0.0,
        "graded grid non-boundary width sum must be > 0 before left correction");
    TENRYU_ASSERT(
        tail_new > 0.0,
        "graded grid non-boundary width sum must remain > 0 after left correction");
    const double scale = tail_new / tail_old;
    for (int k = 1; k < n; ++k) {
      widths[static_cast<std::size_t>(k)] *= scale;
    }
    widths.front() = left_target;
    return;
  }

  const double head_old = length - widths.back();
  const double head_new = length - right_target;
  TENRYU_ASSERT(
      head_old > 0.0,
      "graded grid non-boundary width sum must be > 0 before right correction");
  TENRYU_ASSERT(
      head_new > 0.0,
      "graded grid non-boundary width sum must remain > 0 after right correction");
  const double scale = head_new / head_old;
  for (int k = 0; k < n - 1; ++k) {
    widths[static_cast<std::size_t>(k)] *= scale;
  }
  widths.back() = right_target;
}

double graded_super_gaussian_weight(const GradingConfig& grading,
                                    const int k,
                                    const int nr) {
  const double xi =
      (static_cast<double>(k) + 0.5) / static_cast<double>(nr);
  const double u = std::abs(2.0 * xi - 1.0);
  const double exponent =
      std::pow(u / grading.sg_sigma, static_cast<double>(grading.sg_order));
  return grading.edge_ratio +
         (1.0 - grading.edge_ratio) * std::exp(-exponent);
}

std::vector<double> build_graded_nodes(
    const std::vector<GridSegment>& segments,
    const GradingConfig& grading,
    const bool pin_segment_boundaries = false,
    const int measure_dim = 0) {
  graded_grid_validate_segments(segments);
  graded_grid_validate_config(grading);
  const bool use_exact_measure =
      grading.mapping == "exact_measure_v2" && measure_dim >= 1 &&
      measure_dim <= 3;

  std::vector<std::vector<double>> segment_widths(segments.size());
  std::vector<double> first_targets(segments.size(), 0.0);
  std::vector<double> last_targets(segments.size(), 0.0);
  std::vector<bool> has_first_target(segments.size(), false);
  std::vector<bool> has_last_target(segments.size(), false);

  for (std::size_t s = 0; s < segments.size(); ++s) {
    const auto& seg = segments[s];
    const double length = seg.r_end - seg.r_start;
    auto& widths = segment_widths[s];
    widths.resize(static_cast<std::size_t>(seg.nr), 0.0);

    if (use_exact_measure) {
      const auto measure = [&](const double r) {
        if (measure_dim == 3) {
          return r * r * r;
        }
        if (measure_dim == 2) {
          return r * r;
        }
        return r;
      };
      const auto inverse_measure = [&](const double y) {
        if (measure_dim == 3) {
          return std::cbrt(y);
        }
        if (measure_dim == 2) {
          return std::sqrt(std::max(y, 0.0));
        }
        return y;
      };
      double w_sum = 0.0;
      std::vector<double> w_cum(static_cast<std::size_t>(seg.nr), 0.0);
      for (int k = 0; k < seg.nr; ++k) {
        const double weight =
            graded_super_gaussian_weight(grading, k, seg.nr);
        TENRYU_ASSERT(
            std::isfinite(weight) && weight > 0.0,
            "graded grid exact_measure_v2 requires positive weights");
        w_sum += weight;
        w_cum[static_cast<std::size_t>(k)] = w_sum;
      }
      TENRYU_ASSERT(std::isfinite(w_sum) && w_sum > 0.0,
                    "graded grid produced invalid segment weight sum");
      const double m_in = measure(seg.r_start);
      const double m_out = measure(seg.r_end);
      double r_prev = seg.r_start;
      for (int k = 0; k < seg.nr; ++k) {
        const double W_k =
            w_cum[static_cast<std::size_t>(k)] / w_sum;
        const double r_next =
            (k + 1 == seg.nr)
                ? seg.r_end
                : inverse_measure(m_in + W_k * (m_out - m_in));
        widths[static_cast<std::size_t>(k)] = r_next - r_prev;
        r_prev = r_next;
      }
      continue;
    }

    if (1.0 - grading.edge_ratio <= 1.0e-12) {
      const double dr = length / static_cast<double>(seg.nr);
      std::fill(widths.begin(), widths.end(), dr);
      continue;
    }

    double q_sum = 0.0;
    for (int k = 0; k < seg.nr; ++k) {
      const double xi =
          (static_cast<double>(k) + 0.5) / static_cast<double>(seg.nr);
      const double weight =
          graded_super_gaussian_weight(grading, k, seg.nr);
      const double r_est = seg.r_start + length * xi;
      const double r_ref =
          (seg.r_start < 1.0e-12)
              ? (length / std::sqrt(static_cast<double>(seg.nr)))
              : 0.0;
      const double q = weight / (r_est * r_est + r_ref * r_ref);
      widths[static_cast<std::size_t>(k)] = q;
      q_sum += q;
    }

    TENRYU_ASSERT(std::isfinite(q_sum) && q_sum > 0.0,
                  "graded grid produced invalid segment weight sum");
    const double scale = length / q_sum;
    for (double& width : widths) {
      width *= scale;
    }
  }

  for (std::size_t s = 0; s + 1 < segments.size(); ++s) {
    const double dr_left = segment_widths[s].back();
    const double dr_right = segment_widths[s + 1].front();
    const double dr_match = std::sqrt(dr_left * dr_right);
    last_targets[s] = dr_match;
    has_last_target[s] = true;
    first_targets[s + 1] = dr_match;
    has_first_target[s + 1] = true;
  }

  for (std::size_t s = 0; s < segments.size(); ++s) {
    const double length = segments[s].r_end - segments[s].r_start;
    graded_grid_apply_boundary_targets(
        segment_widths[s], length, has_first_target[s], first_targets[s],
        has_last_target[s], last_targets[s], s);

    double width_sum = 0.0;
    for (const double width : segment_widths[s]) {
      TENRYU_ASSERT(std::isfinite(width) && width > 0.0,
                    "graded grid produced non-positive cell width");
      width_sum += width;
    }
    const double tol = 1.0e-12 * std::max(1.0, std::abs(length));
    TENRYU_ASSERT(
        std::abs(width_sum - length) <= tol,
        "graded grid segment width sum mismatch after boundary correction");
  }

  std::size_t total_cells = 0;
  for (const auto& seg : segments) {
    total_cells += static_cast<std::size_t>(seg.nr);
  }

  std::vector<double> nodes(total_cells + 1, 0.0);
  nodes[0] = segments.front().r_start;
  std::size_t node_index = 0;
  for (std::size_t s = 0; s < segments.size(); ++s) {
    for (const double width : segment_widths[s]) {
      nodes[node_index + 1] = nodes[node_index] + width;
      ++node_index;
    }
    if (pin_segment_boundaries) {
      nodes[node_index] = segments[s].r_end;
    }
  }
  nodes.back() = segments.back().r_end;
  for (std::size_t i = 0; i + 1 < nodes.size(); ++i) {
    TENRYU_ASSERT(nodes[i + 1] > nodes[i],
                  "graded grid nodes must be strictly increasing");
  }
  return nodes;
}

int boundary_tag(const BoundaryKind kind) {
  return static_cast<int>(kind);
}

void set_csr_offsets(std::vector<int>& offsets,
                     const int rows,
                     const int stride) {
  offsets.resize(static_cast<std::size_t>(rows + 1));
  for (int row = 0; row <= rows; ++row) {
    offsets[static_cast<std::size_t>(row)] = stride * row;
  }
}

void set_cell_nodes(MultiBlockTopology& mb,
                    const int cell,
                    const std::array<int, 4>& nodes,
                    const int stride) {
  const int offset =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  for (int corner = 0; corner < stride; ++corner) {
    mb.cell_node_csr_indices[static_cast<std::size_t>(offset + corner)] =
        (corner < 4) ? nodes[static_cast<std::size_t>(corner)] : 0;
  }
}

void set_cell_nodes(MultiBlockTopology& mb,
                    const int cell,
                    const std::array<int, 5>& nodes,
                    const int stride) {
  const int offset =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  for (int corner = 0; corner < stride; ++corner) {
    mb.cell_node_csr_indices[static_cast<std::size_t>(offset + corner)] =
        (corner < 5) ? nodes[static_cast<std::size_t>(corner)] : 0;
  }
}

template <std::size_t N>
void set_cell_faces(MultiBlockTopology& mb,
                    const int cell,
                    const std::array<int, N>& adjacent,
                    const std::array<int, N>& tags,
                    const int stride) {
  const int offset =
      mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
  for (int face = 0; face < stride; ++face) {
    mb.face_adj_csr_indices[static_cast<std::size_t>(offset + face)] =
        (face < static_cast<int>(N))
            ? adjacent[static_cast<std::size_t>(face)]
            : -1;
    mb.face_bc_tags[static_cast<std::size_t>(offset + face)] =
        (face < static_cast<int>(N))
            ? tags[static_cast<std::size_t>(face)]
            : boundary_tag(BoundaryKind::Interior);
  }
}

void assert_bilateral_face_adjacency(
    const MultiBlockTopology& mb,
    const int stride,
    const std::vector<std::uint8_t>& cell_nverts) {
  const int n_cells =
      static_cast<int>(mb.face_adj_csr_offsets.size()) - 1;
  TENRYU_ASSERT(cell_nverts.size() ==
                    static_cast<std::size_t>(n_cells),
                "pentagon-belt face validation requires cell_nverts");
  for (int cell = 0; cell < n_cells; ++cell) {
    const int offset =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
    const int next =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(cell + 1)];
    TENRYU_ASSERT(next - offset == stride,
                  "pentagon-belt face CSR width must match mesh stride");
    const int active_nverts =
        mesh_topo_cell_active_nverts(cell_nverts, cell);
    for (int local = 0; local < stride; ++local) {
      const int neighbor =
          mb.face_adj_csr_indices[static_cast<std::size_t>(offset + local)];
      if (!mesh_topo_local_face_is_active(active_nverts, local)) {
        TENRYU_ASSERT(
            neighbor < 0,
            "inactive pentagon-belt face adjacency slot must be empty");
        continue;
      }
      if (neighbor < 0) {
        continue;
      }
      TENRYU_ASSERT(neighbor < n_cells,
                    "pentagon-belt neighbor cell out of range");
      const int neighbor_offset =
          mb.face_adj_csr_offsets[static_cast<std::size_t>(neighbor)];
      const int neighbor_active_nverts =
          mesh_topo_cell_active_nverts(cell_nverts, neighbor);
      bool found_inverse = false;
      for (int neighbor_local = 0;
           neighbor_local < neighbor_active_nverts;
           ++neighbor_local) {
        if (mb.face_adj_csr_indices[static_cast<std::size_t>(
                neighbor_offset + neighbor_local)] == cell) {
          found_inverse = true;
          break;
        }
      }
      TENRYU_ASSERT(
          found_inverse,
          "pentagon-belt face adjacency is not bilateral");
    }
  }
}

void upload_unique_face_payload(MultiBlockTopology& mb) {
  std::vector<int> unique_cell_a;
  std::vector<int> unique_cell_b;
  std::vector<int> unique_local_a;
  std::vector<int> unique_local_b;
  std::vector<std::int8_t> unique_bc_tag;
  unique_cell_a.reserve(mb.unique_internal_faces.size());
  unique_cell_b.reserve(mb.unique_internal_faces.size());
  unique_local_a.reserve(mb.unique_internal_faces.size());
  unique_local_b.reserve(mb.unique_internal_faces.size());
  unique_bc_tag.reserve(mb.unique_internal_faces.size());
  for (const UniqueOrientedFace& face : mb.unique_internal_faces) {
    unique_cell_a.push_back(face.cell_a);
    unique_cell_b.push_back(face.cell_b);
    unique_local_a.push_back(face.local_a);
    unique_local_b.push_back(face.local_b);
    unique_bc_tag.push_back(face.bc_tag);
  }

  std::vector<int> boundary_cell;
  std::vector<int> boundary_local;
  std::vector<std::int8_t> boundary_bc_tag;
  boundary_cell.reserve(mb.boundary_faces.size());
  boundary_local.reserve(mb.boundary_faces.size());
  boundary_bc_tag.reserve(mb.boundary_faces.size());
  for (const UniqueOrientedFace& face : mb.boundary_faces) {
    boundary_cell.push_back(face.cell_a);
    boundary_local.push_back(face.local_a);
    boundary_bc_tag.push_back(face.bc_tag);
  }

  mb.d_unique_face_cell_a.assign(unique_cell_a.begin(), unique_cell_a.end());
  mb.d_unique_face_cell_b.assign(unique_cell_b.begin(), unique_cell_b.end());
  mb.d_unique_face_local_a.assign(unique_local_a.begin(),
                                  unique_local_a.end());
  mb.d_unique_face_local_b.assign(unique_local_b.begin(),
                                  unique_local_b.end());
  mb.d_unique_face_bc_tag.assign(unique_bc_tag.begin(), unique_bc_tag.end());
  mb.d_boundary_face_cell.assign(boundary_cell.begin(), boundary_cell.end());
  mb.d_boundary_face_local.assign(boundary_local.begin(),
                                  boundary_local.end());
  mb.d_boundary_face_bc_tag.assign(boundary_bc_tag.begin(),
                                   boundary_bc_tag.end());
  mb.d_face_adj_csr_offsets.assign(mb.face_adj_csr_offsets.begin(),
                                   mb.face_adj_csr_offsets.end());
  mb.d_face_adj_csr_indices.assign(mb.face_adj_csr_indices.begin(),
                                   mb.face_adj_csr_indices.end());
}

bool is_belt_layer(const tenryu::core::Config::MeshConfig& cfg,
                   const int layer) {
  return std::binary_search(cfg.pentagon_belt_layers.begin(),
                            cfg.pentagon_belt_layers.end(), layer);
}

}  // namespace

Mesh create_pentagon_belt_shell_mesh(const tenryu::core::Config& cfg,
                                     tenryu::core::State& state) {
  const auto& mesh_cfg = cfg.mesh;
  TENRYU_ASSERT(cfg.main.dim == 2 && cfg.main.dimension == "2D_RZ",
                "pentagon-belt requires Main.dimension='2D_RZ'");
  TENRYU_ASSERT(mesh_cfg.topology_scheme ==
                    tenryu::core::TopologyScheme::PENTAGON_BELT_SHELL,
                "pentagon-belt builder requires pentagon_belt_shell topology");
  TENRYU_ASSERT(mesh_cfg.logical_mesh_2d ==
                    "spherical_polar_halfplane" &&
                    mesh_cfg.polar_center_treatment == "annular",
                "pentagon-belt builder requires annular spherical polar mesh");

  Mesh mesh;
  mesh.corner_stride =
      tenryu::core::corner_stride_for_config(cfg);
  TENRYU_ASSERT(mesh.corner_stride == 8,
                "pentagon-belt corner stride must be 8");
  TENRYU_ASSERT(mesh.corner_stride == state.corner_stride,
                "Mesh/state corner stride mismatch");
  mesh.dim = 2;
  mesh.geometry_code =
      (mesh_cfg.geometry_1d == "cylindrical")
          ? 1
          : ((mesh_cfg.geometry_1d == "planar") ? 2 : 0);
  mesh.logical = LogicalMesh2D::SphericalPolarHalfplane;
  mesh.polar_center_treatment = PolarCenterTreatment::Annular;
  mesh.topo.nr = mesh_cfg.nr;
  mesh.topo.nz = mesh_cfg.nz;
  mesh.topo.n_cells = mesh_topo_n_cells_total(mesh_cfg);
  mesh.topo.n_nodes = mesh_topo_n_nodes_total(mesh_cfg);
  mesh.edge_tags.clear();
  mesh.cell_nverts.assign(static_cast<std::size_t>(mesh.topo.n_cells), 4U);

  TENRYU_ASSERT(state.x_r.size() ==
                    static_cast<std::size_t>(mesh.topo.n_nodes) &&
                    state.x_z.size() ==
                        static_cast<std::size_t>(mesh.topo.n_nodes),
                "pentagon-belt State node storage size mismatch");

  const int nr = mesh.topo.nr;
  const int nz = mesh.topo.nz;
  std::vector<int> ring_zones(static_cast<std::size_t>(nr + 1), 0);
  std::vector<int> ring_node_begin(static_cast<std::size_t>(nr + 1), 0);
  int node_count = 0;
  for (int ring = 0; ring <= nr; ++ring) {
    ring_zones[static_cast<std::size_t>(ring)] =
        mesh_topo_pentagon_belt_ring_zones(mesh_cfg, ring);
    ring_node_begin[static_cast<std::size_t>(ring)] = node_count;
    node_count += ring_zones[static_cast<std::size_t>(ring)] + 1;
  }
  TENRYU_ASSERT(node_count == mesh.topo.n_nodes,
                "pentagon-belt ring node counts do not match total");

  std::vector<int> layer_cell_begin(static_cast<std::size_t>(nr + 1), 0);
  int cell_count = 0;
  for (int layer = 0; layer < nr; ++layer) {
    layer_cell_begin[static_cast<std::size_t>(layer)] = cell_count;
    cell_count += ring_zones[static_cast<std::size_t>(layer)];
  }
  layer_cell_begin[static_cast<std::size_t>(nr)] = cell_count;
  TENRYU_ASSERT(cell_count == mesh.topo.n_cells,
                "pentagon-belt layer cell counts do not match total");

  const double s_max = mesh_cfg.spherical_polar_s_max;
  TENRYU_ASSERT(s_max > 0.0,
                "Mesh.spherical_polar_s_max must be > 0");
  TENRYU_ASSERT(mesh_cfg.spherical_polar_kappa > 0.0,
                "Mesh.spherical_polar_kappa must be > 0");
  std::vector<double> s_nodes(static_cast<std::size_t>(nr + 1), 0.0);
  // Intentional inline copy of the single-block polar radial ladder logic.
  const bool polar_explicit_nodes = !mesh_cfg.explicit_nodes.empty();
  const bool polar_grid_segments = !mesh_cfg.grid_segments.empty();
  if (polar_explicit_nodes || polar_grid_segments) {
    if (polar_explicit_nodes) {
      TENRYU_ASSERT(
          static_cast<int>(mesh_cfg.explicit_nodes.size()) == nr + 1,
          "Mesh.explicit_nodes count mismatch for polar prefix: got " +
              std::to_string(mesh_cfg.explicit_nodes.size()) +
              " nodes, expected " + std::to_string(nr + 1));
      s_nodes = mesh_cfg.explicit_nodes;
    } else {
      s_nodes =
          build_graded_nodes(mesh_cfg.grid_segments, mesh_cfg.grading, true);
      TENRYU_ASSERT(static_cast<int>(s_nodes.size()) == nr + 1,
                    "Graded polar node count mismatch: got " +
                        std::to_string(s_nodes.size()) +
                        " nodes, expected " + std::to_string(nr + 1));
    }
    for (std::size_t k = 1; k < s_nodes.size(); ++k) {
      TENRYU_ASSERT(s_nodes[k] > s_nodes[k - 1],
                    "polar radial nodes must be strictly increasing");
    }
    TENRYU_ASSERT(
        s_nodes.front() > 0.0,
        "graded/explicit polar nodes require s(0) > 0 for annular center treatment");
    TENRYU_ASSERT(
        std::abs(s_nodes.back() - s_max) <=
            1.0e-12 * std::max(1.0, std::abs(s_max)),
        "graded/explicit polar nodes must end at "
        "Mesh.spherical_polar_s_max");
    double max_ratio = 1.0;
    for (std::size_t k = 2; k < s_nodes.size(); ++k) {
      const double d0 = s_nodes[k - 1] - s_nodes[k - 2];
      const double d1 = s_nodes[k] - s_nodes[k - 1];
      const double ratio = std::max(d1 / d0, d0 / d1);
      max_ratio = std::max(max_ratio, ratio);
    }
    tenryu::core::log_info(
        "[mesh-graded] spherical_polar radial nodes: n=" +
        std::to_string(nr + 1) +
        " s0=" + std::to_string(s_nodes.front()) +
        " s_max=" + std::to_string(s_nodes.back()) +
        " max_adjacent_dr_ratio=" + std::to_string(max_ratio));
  } else {
    const double kappa = mesh_cfg.spherical_polar_kappa;
    const double delta_s =
        s_max / (static_cast<double>(nr) + kappa);
    for (int i = 0; i <= nr; ++i) {
      s_nodes[static_cast<std::size_t>(i)] =
          (static_cast<double>(i) + kappa) * delta_s;
    }
  }

  std::vector<double> theta_fine(static_cast<std::size_t>(nz + 1), 0.0);
  if (mesh_cfg.polar_equal_mu_zoning) {
    for (int j = 0; j <= nz; ++j) {
      double mu =
          1.0 - 2.0 * static_cast<double>(j) / static_cast<double>(nz);
      mu = std::clamp(mu, -1.0, 1.0);
      theta_fine[static_cast<std::size_t>(j)] = std::acos(mu);
    }
  } else {
    for (int j = 0; j <= nz; ++j) {
      theta_fine[static_cast<std::size_t>(j)] =
          static_cast<double>(j) * spherical_wedge_detail::kPi /
          static_cast<double>(nz);
    }
  }

  constexpr double center_z = 0.0;
  std::vector<double> host_x_r(static_cast<std::size_t>(mesh.topo.n_nodes),
                               0.0);
  std::vector<double> host_x_z(static_cast<std::size_t>(mesh.topo.n_nodes),
                               0.0);
  for (int ring = 0; ring <= nr; ++ring) {
    const double s = s_nodes[static_cast<std::size_t>(ring)];
    const int ntheta = ring_zones[static_cast<std::size_t>(ring)];
    TENRYU_ASSERT(nz % ntheta == 0,
                  "pentagon-belt ring count must divide finest theta count");
    const int subsample = nz / ntheta;
    const int northern_end = ntheta / 2;
    for (int j = 0; j <= northern_end; ++j) {
      const double theta =
          theta_fine[static_cast<std::size_t>(j * subsample)];
      const int node =
          ring_node_begin[static_cast<std::size_t>(ring)] + j;
      host_x_r[static_cast<std::size_t>(node)] =
          (j == 0) ? 0.0 : s * std::sin(theta);
      host_x_z[static_cast<std::size_t>(node)] =
          (2 * j == ntheta) ? center_z + 0.0
                            : center_z + s * std::cos(theta);
    }
    for (int j = northern_end + 1; j <= ntheta; ++j) {
      const int northern_j = ntheta - j;
      const int node =
          ring_node_begin[static_cast<std::size_t>(ring)] + j;
      const int northern_node =
          ring_node_begin[static_cast<std::size_t>(ring)] + northern_j;
      host_x_r[static_cast<std::size_t>(node)] =
          host_x_r[static_cast<std::size_t>(northern_node)];
      host_x_z[static_cast<std::size_t>(node)] =
          center_z -
          (host_x_z[static_cast<std::size_t>(northern_node)] - center_z);
    }
  }

  mesh.topo.node_flags.assign(static_cast<std::size_t>(mesh.topo.n_nodes),
                              NODE_NONE);
  const double axis_tol = tenryu::core::axis_tolerance(s_max);
  for (int ring = 0; ring <= nr; ++ring) {
    const int ntheta = ring_zones[static_cast<std::size_t>(ring)];
    for (int j = 0; j <= ntheta; ++j) {
      const int node =
          ring_node_begin[static_cast<std::size_t>(ring)] + j;
      std::uint8_t flags = NODE_NONE;
      if (std::fabs(host_x_r[static_cast<std::size_t>(node)]) <= axis_tol) {
        flags |= NODE_AXIS | NODE_POLE_AXIS | NODE_BOUNDARY;
      }
      if (ring == 0) {
        flags |= NODE_BOUNDARY | NODE_INNER_PHYSICAL_BOUNDARY;
      }
      if (ring == nr) {
        flags |= NODE_BOUNDARY | NODE_OUTER_PHYSICAL_BOUNDARY;
      }
      mesh.topo.node_flags[static_cast<std::size_t>(node)] = flags;
    }
  }

  mesh.topo.multiblock =
      mesh_topo_make_empty_multiblock_topology(mesh_cfg);
  auto& mb = *mesh.topo.multiblock;
  const int stride = mesh.corner_stride;
  TENRYU_ASSERT(
      mb.block_count ==
          2 * static_cast<int>(mesh_cfg.pentagon_belt_layers.size()) + 1,
      "pentagon-belt block count mismatch");
  mb.blocks.reserve(static_cast<std::size_t>(mb.block_count));
  mb.cell_block_id.assign(static_cast<std::size_t>(mesh.topo.n_cells), -1);
  mb.cell_id_stable.resize(static_cast<std::size_t>(mesh.topo.n_cells));
  mb.cell_orientation_sign.assign(
      static_cast<std::size_t>(mesh.topo.n_cells), -1);
  set_csr_offsets(mb.cell_node_csr_offsets, mesh.topo.n_cells, stride);
  set_csr_offsets(mb.face_adj_csr_offsets, mesh.topo.n_cells, stride);
  const std::size_t slot_count =
      static_cast<std::size_t>(mesh.topo.n_cells) *
      static_cast<std::size_t>(stride);
  mb.cell_node_csr_indices.assign(slot_count, -1);
  mb.face_adj_csr_indices.assign(slot_count, -1);
  mb.face_bc_tags.assign(slot_count,
                         boundary_tag(BoundaryKind::Interior));

  std::vector<int> layer_block_id(static_cast<std::size_t>(nr), -1);
  const auto append_block =
      [&mb, &layer_block_id, &layer_cell_begin](
          const int layer_begin,
          const int layer_end,
          const int n_i_cells,
          const int n_j_cells,
          const int owned_node_begin,
          const int owned_node_count) {
        TENRYU_ASSERT(layer_begin >= 0 && layer_end > layer_begin,
                      "pentagon-belt block must contain a radial layer");
        TENRYU_ASSERT(n_i_cells > 0 && n_j_cells > 0,
                      "pentagon-belt block dimensions must be positive");
        const int block_id = static_cast<int>(mb.blocks.size());
        const int cell_begin =
            layer_cell_begin[static_cast<std::size_t>(layer_begin)];
        const int cell_count = n_i_cells * n_j_cells;
        mb.blocks.push_back(
            {BlockRole::PENTAGON_BELT,
             n_i_cells,
             n_j_cells,
             cell_begin,
             cell_count,
             owned_node_begin,
             owned_node_count});
        for (int layer = layer_begin; layer < layer_end; ++layer) {
          layer_block_id[static_cast<std::size_t>(layer)] = block_id;
        }
      };

  int band_layer_begin = 0;
  bool first_band = true;
  for (const int belt_layer : mesh_cfg.pentagon_belt_layers) {
    const int band_n_i = belt_layer - band_layer_begin;
    const int band_n_j =
        ring_zones[static_cast<std::size_t>(band_layer_begin)];
    const int band_owned_ring_begin =
        first_band ? 0 : band_layer_begin + 1;
    const int band_owned_begin =
        ring_node_begin[static_cast<std::size_t>(band_owned_ring_begin)];
    const int band_owned_end =
        ring_node_begin[static_cast<std::size_t>(belt_layer)] +
        ring_zones[static_cast<std::size_t>(belt_layer)] + 1;
    append_block(band_layer_begin, belt_layer, band_n_i, band_n_j,
                 band_owned_begin, band_owned_end - band_owned_begin);

    const int row_owned_begin =
        ring_node_begin[static_cast<std::size_t>(belt_layer + 1)];
    const int row_owned_count =
        ring_zones[static_cast<std::size_t>(belt_layer + 1)] + 1;
    append_block(belt_layer, belt_layer + 1, 1,
                 ring_zones[static_cast<std::size_t>(belt_layer)],
                 row_owned_begin, row_owned_count);
    band_layer_begin = belt_layer + 1;
    first_band = false;
  }

  const int final_band_n_i = nr - band_layer_begin;
  const int final_band_n_j =
      ring_zones[static_cast<std::size_t>(band_layer_begin)];
  const int final_owned_ring_begin = band_layer_begin + 1;
  const int final_owned_begin =
      ring_node_begin[static_cast<std::size_t>(final_owned_ring_begin)];
  const int final_owned_end =
      ring_node_begin[static_cast<std::size_t>(nr)] +
      ring_zones[static_cast<std::size_t>(nr)] + 1;
  append_block(band_layer_begin, nr, final_band_n_i, final_band_n_j,
               final_owned_begin, final_owned_end - final_owned_begin);

  TENRYU_ASSERT(static_cast<int>(mb.blocks.size()) == mb.block_count,
                "pentagon-belt block table size mismatch");
  int block_cell_sum = 0;
  int block_owned_node_sum = 0;
  for (const BlockInfo& block : mb.blocks) {
    block_cell_sum += block.cell_count;
    block_owned_node_sum += block.owned_node_count;
  }
  TENRYU_ASSERT(block_cell_sum == mesh.topo.n_cells,
                "pentagon-belt block cells do not cover the mesh");
  TENRYU_ASSERT(block_owned_node_sum == mesh.topo.n_nodes,
                "pentagon-belt block node ownership does not cover the mesh");

  for (int cell = 0; cell < mesh.topo.n_cells; ++cell) {
    mb.cell_id_stable[static_cast<std::size_t>(cell)] = cell;
  }
  for (int layer = 0; layer < nr; ++layer) {
    const int layer_begin =
        layer_cell_begin[static_cast<std::size_t>(layer)];
    const int layer_end =
        layer_cell_begin[static_cast<std::size_t>(layer + 1)];
    const int block_id =
        layer_block_id[static_cast<std::size_t>(layer)];
    TENRYU_ASSERT(block_id >= 0 && block_id < mb.block_count,
                  "pentagon-belt layer block id was not initialized");
    for (int cell = layer_begin; cell < layer_end; ++cell) {
      mb.cell_block_id[static_cast<std::size_t>(cell)] = block_id;
    }
  }

  const auto node_id =
      [&ring_node_begin](const int ring, const int theta_index) {
        return ring_node_begin[static_cast<std::size_t>(ring)] + theta_index;
      };
  const auto cell_id =
      [&layer_cell_begin](const int layer, const int theta_index) {
        return layer_cell_begin[static_cast<std::size_t>(layer)] +
               theta_index;
      };

  for (int layer = 0; layer < nr; ++layer) {
    const int ntheta = ring_zones[static_cast<std::size_t>(layer)];
    if (is_belt_layer(mesh_cfg, layer)) {
      TENRYU_ASSERT(
          ring_zones[static_cast<std::size_t>(layer + 1)] == 2 * ntheta,
          "pentagon-belt row must refine theta count by exactly 2");
      for (int m = 0; m < ntheta; ++m) {
        const int cell = cell_id(layer, m);
        mesh.cell_nverts[static_cast<std::size_t>(cell)] = 5U;
        set_cell_nodes(
            mb, cell,
            std::array<int, 5>{{
                node_id(layer, m),
                node_id(layer + 1, 2 * m),
                node_id(layer + 1, 2 * m + 1),
                node_id(layer + 1, 2 * m + 2),
                node_id(layer, m + 1),
            }},
            stride);

        std::array<int, 5> adjacent{{
            -1,
            cell_id(layer + 1, 2 * m),
            cell_id(layer + 1, 2 * m + 1),
            -1,
            cell_id(layer - 1, m),
        }};
        std::array<int, 5> tags{{
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
        }};
        if (m > 0) {
          adjacent[0] = cell_id(layer, m - 1);
        } else {
          tags[0] = boundary_tag(BoundaryKind::AXIS_SHELL);
        }
        if (m + 1 < ntheta) {
          adjacent[3] = cell_id(layer, m + 1);
        } else {
          tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
        }
        set_cell_faces(mb, cell, adjacent, tags, stride);
      }
      continue;
    }

    TENRYU_ASSERT(
        ring_zones[static_cast<std::size_t>(layer + 1)] == ntheta,
        "pentagon-belt quad band rings must have equal theta counts");
    for (int m = 0; m < ntheta; ++m) {
      const int cell = cell_id(layer, m);
      set_cell_nodes(
          mb, cell,
          std::array<int, 4>{{
              node_id(layer, m),
              node_id(layer + 1, m),
              node_id(layer + 1, m + 1),
              node_id(layer, m + 1),
          }},
          stride);

      std::array<int, 4> adjacent{{-1, -1, -1, -1}};
      std::array<int, 4> tags{{
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
      }};
      if (layer == 0) {
        tags[0] = boundary_tag(BoundaryKind::SphericalInnerCore);
      } else if (is_belt_layer(mesh_cfg, layer - 1)) {
        adjacent[0] = cell_id(layer - 1, m / 2);
      } else {
        adjacent[0] = cell_id(layer - 1, m);
      }
      if (layer + 1 == nr) {
        tags[1] = boundary_tag(BoundaryKind::SphericalOuterFree);
      } else if (is_belt_layer(mesh_cfg, layer + 1)) {
        adjacent[1] = cell_id(layer + 1, m);
      } else {
        adjacent[1] = cell_id(layer + 1, m);
      }
      if (m > 0) {
        adjacent[2] = cell_id(layer, m - 1);
      } else {
        tags[2] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      if (m + 1 < ntheta) {
        adjacent[3] = cell_id(layer, m + 1);
      } else {
        tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      set_cell_faces(mb, cell, adjacent, tags, stride);
    }
  }

  assert_bilateral_face_adjacency(mb, stride, mesh.cell_nverts);
  tenryu::hydro::build_unique_oriented_face_list(
      mb, mb.cell_id_stable, mb.unique_internal_faces, mb.boundary_faces,
      &mesh.cell_nverts, stride);
  upload_unique_face_payload(mb);

  mesh.multiblock_cell_node_csr_offsets.reset(
      mb.cell_node_csr_offsets.size());
  mesh.multiblock_cell_node_csr_offsets.copy_from_host(
      mb.cell_node_csr_offsets);
  mesh.multiblock_cell_node_csr_indices.reset(
      mb.cell_node_csr_indices.size());
  mesh.multiblock_cell_node_csr_indices.copy_from_host(
      mb.cell_node_csr_indices);
  mesh.multiblock_cell_orientation_sign_device.reset(
      mb.cell_orientation_sign.size());
  if (!mb.cell_orientation_sign.empty()) {
    mesh.multiblock_cell_orientation_sign_device.copy_from_host(
        mb.cell_orientation_sign);
  }
  mesh.multiblock_cell_nverts_device.reset(mesh.cell_nverts.size());
  if (!mesh.cell_nverts.empty()) {
    mesh.multiblock_cell_nverts_device.copy_from_host(mesh.cell_nverts);
  }
  ++mesh.topology_serial;
  const tenryu::hydro::ReverseCellNodeCSR reverse_csr =
      tenryu::hydro::build_reverse_cell_node_csr(
          mb, mesh.topo.n_nodes, &mesh.cell_nverts, stride);
  mesh.multiblock_reverse_csr_node_offsets.reset(
      reverse_csr.node_offsets.size());
  mesh.multiblock_reverse_csr_node_offsets.copy_from_host(
      reverse_csr.node_offsets);
  mesh.multiblock_reverse_csr_node_cells.reset(
      reverse_csr.node_cells.size());
  mesh.multiblock_reverse_csr_node_cells.copy_from_host(
      reverse_csr.node_cells);
  mesh.multiblock_reverse_csr_node_corners.reset(
      reverse_csr.node_corners.size());
  mesh.multiblock_reverse_csr_node_corners.copy_from_host(
      reverse_csr.node_corners);
  rebuild_multiblock_node_edge_csr(mesh);
  rebuild_multiblock_csw_line_topology(mesh);

  state.x_r.copy_from_host(host_x_r.data());
  state.x_z.copy_from_host(host_x_z.data());
  if (state.x_r_initial.size() == state.x_r.size()) {
    state.x_r_initial.copy_from_host(host_x_r.data());
  }
  if (state.x_z_initial.size() == state.x_z.size()) {
    state.x_z_initial.copy_from_host(host_x_z.data());
  }
  if (state.x_r_reference.size() == state.x_r.size()) {
    state.x_r_reference.copy_from_host(host_x_r.data());
  }
  if (state.x_z_reference.size() == state.x_z.size()) {
    state.x_z_reference.copy_from_host(host_x_z.data());
  }
  state.cell_nverts_host.resize(mesh.cell_nverts.size());
  for (std::size_t cell = 0; cell < mesh.cell_nverts.size(); ++cell) {
    state.cell_nverts_host[cell] =
        static_cast<std::int8_t>(mesh.cell_nverts[cell]);
  }

  mesh.node_r = state.x_r.data();
  mesh.node_z = state.x_z.data();
  mesh.recompute_geometry();

  double total_vol = 0.0;
  double total_vol_compensation = 0.0;
  for (const double vol : mesh.cell_vol) {
    const double y = vol - total_vol_compensation;
    const double t = total_vol + y;
    total_vol_compensation = (t - total_vol) - y;
    total_vol = t;
  }
  constexpr double four_pi_over_three =
      4.188790204786390984616857844372670512262892532500141094646;
  const double s_inner = s_nodes.front();
  const double s_outer = s_nodes.back();
  const double expected_total_vol =
      four_pi_over_three *
      (s_outer * s_outer * s_outer - s_inner * s_inner * s_inner);
  const double rel_err =
      std::abs(total_vol - expected_total_vol) / expected_total_vol;
  const double min_angular_count =
      static_cast<double>(
          *std::min_element(ring_zones.begin(), ring_zones.end()));
  const double total_vol_rel_tol =
      std::max(1.0e-12,
               10.0 / (min_angular_count * min_angular_count));
  // The polygonal half-plane boundary has O(1/N^2) approximation error vs
  // the analytic curved region; this tolerance tracks that refinement behavior.
  TENRYU_ASSERT(
      std::isfinite(total_vol) && std::isfinite(rel_err) &&
          rel_err <= total_vol_rel_tol,
      "pentagon-belt initial total volume mismatch: rel_err=" +
          std::to_string(rel_err));
  state.cell_vol_initial = mesh.cell_vol;
  return mesh;
}

}  // namespace tenryu::mesh
