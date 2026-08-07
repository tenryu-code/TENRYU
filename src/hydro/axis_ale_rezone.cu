#include "hydro/axis_ale_rezone.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "hydro/hydro_multiblock_topology.cuh"
#include "mesh/candidate_mesh_admissibility.hpp"

namespace tenryu::hydro::axis_ale {
namespace {

constexpr double kCellTargetShearAuditThreshold =
    kAxisAleCoreShearTriggerThreshold;

void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

struct PavaBlock {
  int begin = 0;
  int end = 0;
  double weight = 0.0;
  double mean = 0.0;
};

double edge_length(const std::array<double, 4>& r,
                   const std::array<double, 4>& z,
                   const int k) {
  const int kp = (k + 1) & 3;
  return std::hypot(r[static_cast<std::size_t>(kp)] -
                        r[static_cast<std::size_t>(k)],
                    z[static_cast<std::size_t>(kp)] -
                        z[static_cast<std::size_t>(k)]);
}

double min_edge_length_from_quad(const std::array<double, 4>& r,
                                 const std::array<double, 4>& z) {
  double min_edge = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 4; ++k) {
    const double length = edge_length(r, z, k);
    if (!std::isfinite(length)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    min_edge = std::min(min_edge, length);
  }
  return min_edge;
}

double edge_length_from_polygon(const std::array<double, 4>& r,
                                const std::array<double, 4>& z,
                                const int k,
                                const int active_nverts) {
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    return edge_length(r, z, k);
  }
  const int kp = (k + 1) % active_nverts;
  return std::hypot(r[static_cast<std::size_t>(kp)] -
                        r[static_cast<std::size_t>(k)],
                    z[static_cast<std::size_t>(kp)] -
                        z[static_cast<std::size_t>(k)]);
}

double min_edge_length_from_polygon(const std::array<double, 4>& r,
                                    const std::array<double, 4>& z,
                                    const int active_nverts) {
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    return min_edge_length_from_quad(r, z);
  }
  double min_edge = std::numeric_limits<double>::infinity();
  for (int k = 0; k < active_nverts; ++k) {
    const double length = edge_length_from_polygon(r, z, k, active_nverts);
    if (!std::isfinite(length)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    min_edge = std::min(min_edge, length);
  }
  return min_edge;
}

double max_edge_length_from_quad(const std::array<double, 4>& r,
                                 const std::array<double, 4>& z) {
  double max_edge = 0.0;
  for (int k = 0; k < 4; ++k) {
    const double length = edge_length(r, z, k);
    if (!std::isfinite(length)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    max_edge = std::max(max_edge, length);
  }
  return max_edge;
}

double min_altitude_from_quad(const std::array<double, 4>& r,
                              const std::array<double, 4>& z) {
  double min_altitude = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 4; ++k) {
    const int e0 = (k + 1) & 3;
    const int e1 = (k + 2) & 3;
    const double er = r[static_cast<std::size_t>(e1)] -
                      r[static_cast<std::size_t>(e0)];
    const double ez = z[static_cast<std::size_t>(e1)] -
                      z[static_cast<std::size_t>(e0)];
    const double len = std::hypot(er, ez);
    if (!(len > 0.0) || !std::isfinite(len)) {
      return 0.0;
    }
    const double pr = r[static_cast<std::size_t>(k)] -
                      r[static_cast<std::size_t>(e0)];
    const double pz = z[static_cast<std::size_t>(k)] -
                      z[static_cast<std::size_t>(e0)];
    const double altitude = std::abs(pr * ez - pz * er) / len;
    if (!std::isfinite(altitude)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    min_altitude = std::min(min_altitude, altitude);
  }
  return min_altitude;
}

double max_edge_length_from_polygon(const std::array<double, 4>& r,
                                    const std::array<double, 4>& z,
                                    const int active_nverts) {
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    return max_edge_length_from_quad(r, z);
  }
  double max_edge = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const double length = edge_length_from_polygon(r, z, k, active_nverts);
    if (!std::isfinite(length)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    max_edge = std::max(max_edge, length);
  }
  return max_edge;
}

double min_altitude_from_polygon(const std::array<double, 4>& r,
                                 const std::array<double, 4>& z,
                                 const int active_nverts) {
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    return min_altitude_from_quad(r, z);
  }
  double min_altitude = std::numeric_limits<double>::infinity();
  for (int k = 0; k < active_nverts; ++k) {
    const int e0 = (k + 1) % active_nverts;
    const int e1 = (k + 2) % active_nverts;
    const double er = r[static_cast<std::size_t>(e1)] -
                      r[static_cast<std::size_t>(e0)];
    const double ez = z[static_cast<std::size_t>(e1)] -
                      z[static_cast<std::size_t>(e0)];
    const double len = std::hypot(er, ez);
    if (!(len > 0.0) || !std::isfinite(len)) {
      return 0.0;
    }
    const double pr = r[static_cast<std::size_t>(k)] -
                      r[static_cast<std::size_t>(e0)];
    const double pz = z[static_cast<std::size_t>(k)] -
                      z[static_cast<std::size_t>(e0)];
    const double altitude = std::abs(pr * ez - pz * er) / len;
    if (!std::isfinite(altitude)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    min_altitude = std::min(min_altitude, altitude);
  }
  return min_altitude;
}

double cross2(const double ar, const double az, const double br, const double bz) {
  return ar * bz - az * br;
}

double corner_j_from_quad(const std::array<double, 4>& r,
                          const std::array<double, 4>& z,
                          const int corner) {
  const int kp = (corner + 1) & 3;
  const int km = (corner + 3) & 3;
  return cross2(r[static_cast<std::size_t>(kp)] -
                    r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(kp)] -
                    z[static_cast<std::size_t>(corner)],
                r[static_cast<std::size_t>(km)] -
                    r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(km)] -
                    z[static_cast<std::size_t>(corner)]);
}

double min_corner_j_from_quad(const std::array<double, 4>& r,
                              const std::array<double, 4>& z,
                              const int orientation_sign) {
  double min_j = std::numeric_limits<double>::infinity();
  const double sign = orientation_sign < 0 ? -1.0 : 1.0;
  for (int k = 0; k < 4; ++k) {
    const double j = sign * corner_j_from_quad(r, z, k);
    if (!std::isfinite(j)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    min_j = std::min(min_j, j);
  }
  return min_j;
}

double corner_j_from_polygon(const std::array<double, 4>& r,
                             const std::array<double, 4>& z,
                             const int corner,
                             const int active_nverts) {
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    return corner_j_from_quad(r, z, corner);
  }
  const int kp = (corner + 1) % active_nverts;
  const int km = (corner + active_nverts - 1) % active_nverts;
  return cross2(r[static_cast<std::size_t>(kp)] -
                    r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(kp)] -
                    z[static_cast<std::size_t>(corner)],
                r[static_cast<std::size_t>(km)] -
                    r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(km)] -
                    z[static_cast<std::size_t>(corner)]);
}

double min_corner_j_from_polygon(const std::array<double, 4>& r,
                                 const std::array<double, 4>& z,
                                 const int active_nverts,
                                 const int orientation_sign) {
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    return min_corner_j_from_quad(r, z, orientation_sign);
  }
  double min_j = std::numeric_limits<double>::infinity();
  const double sign = orientation_sign < 0 ? -1.0 : 1.0;
  for (int k = 0; k < active_nverts; ++k) {
    const double j = sign * corner_j_from_polygon(r, z, k, active_nverts);
    if (!std::isfinite(j)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    min_j = std::min(min_j, j);
  }
  return min_j;
}

double aspect_ratio_from_quad(const std::array<double, 4>& r,
                              const std::array<double, 4>& z) {
  const double min_altitude = min_altitude_from_quad(r, z);
  const double max_edge = max_edge_length_from_quad(r, z);
  if (!(min_altitude > 0.0) || !std::isfinite(max_edge)) {
    return std::numeric_limits<double>::infinity();
  }
  return max_edge / min_altitude;
}

double aspect_ratio_from_polygon(const std::array<double, 4>& r,
                                 const std::array<double, 4>& z,
                                 const int active_nverts) {
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    return aspect_ratio_from_quad(r, z);
  }
  const double min_altitude = min_altitude_from_polygon(r, z, active_nverts);
  const double max_edge = max_edge_length_from_polygon(r, z, active_nverts);
  if (!(min_altitude > 0.0) || !std::isfinite(max_edge)) {
    return std::numeric_limits<double>::infinity();
  }
  return max_edge / min_altitude;
}

std::vector<double> copy_node_field(const double* field,
                                    const int n_nodes,
                                    const char* message) {
  TENRYU_ASSERT(field != nullptr, "axis ALE diagnostics require node coordinates");
  std::vector<double> host(static_cast<std::size_t>(n_nodes), 0.0);
  cuda_check(cudaMemcpy(host.data(),
                        field,
                        host.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             message);
  return host;
}

bool mesh_has_trifan_cap(const mesh::Mesh& mesh) {
  return mesh.topo.multiblock.has_value() && mesh.topo.multiblock->has_trifan_cap;
}

bool node_is_pinned_cap_origin(const mesh::Mesh& mesh, const int node) {
  if (!mesh_has_trifan_cap(mesh) || node < 0 || node >= mesh.topo.n_nodes ||
      mesh.topo.node_flags.size() != static_cast<std::size_t>(mesh.topo.n_nodes)) {
    return false;
  }
  const std::uint8_t flags = mesh.topo.node_flags[static_cast<std::size_t>(node)];
  return (flags & mesh::NODE_CENTER) != 0U;
}

double polar_shell_axis_order_floor(const AxisAleRezoneInput& input,
                                    const int ia,
                                    const int ib) {
  const double za = input.z_tilde[static_cast<std::size_t>(ia)];
  const double zb = input.z_tilde[static_cast<std::size_t>(ib)];
  const double scale =
      std::max({1.0, std::abs(za), std::abs(zb), std::abs(zb - za)});
  double floor =
      std::max(1.0e-12 * scale,
               100.0 * std::numeric_limits<double>::epsilon() * scale);
  const int edge = std::min(ia, ib);
  if (std::abs(ia - ib) == 1 && edge >= 0 &&
      static_cast<std::size_t>(edge) < input.initial_edge_length.size() &&
      static_cast<std::size_t>(edge) < input.adjacent_cell_area.size()) {
    const double l0 = input.initial_edge_length[static_cast<std::size_t>(edge)];
    const double area = input.adjacent_cell_area[static_cast<std::size_t>(edge)];
    if (std::isfinite(l0) && l0 > 0.0 &&
        std::isfinite(area) && area > 0.0) {
      floor = std::max(floor, input.eta_floor * std::min(l0, std::sqrt(area)));
    }
  }
  return floor;
}

void enforce_polar_shell_axis_radial_order(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input,
    std::vector<double>& z_target) {
  if (!mesh_has_trifan_cap(mesh) || input.node_ids.size() != z_target.size()) {
    return;
  }
  const auto& mb = *mesh.topo.multiblock;
  const mesh::BlockInfo& shell =
      mesh::mesh_topo_multiblock_polar_shell_block(mb);
  const int n_nodes = mesh.topo.n_nodes;
  if (n_nodes <= 0 || shell.n_i_cells <= 0 || shell.n_j_cells <= 0) {
    return;
  }
  std::vector<int> index_by_node(static_cast<std::size_t>(n_nodes), -1);
  for (int i = 0; i < static_cast<int>(input.node_ids.size()); ++i) {
    const int n = input.node_ids[static_cast<std::size_t>(i)];
    if (n >= 0 && n < n_nodes) {
      index_by_node[static_cast<std::size_t>(n)] = i;
    }
  }

  const int stride = shell.n_j_cells + 1;
  const auto shell_node = [&](const int q, const int j) {
    return shell.owned_node_begin + q * stride + j;
  };
  const auto project_column = [&](const int j, const bool north_pole) {
    for (int q = shell.n_i_cells - 1; q >= 0; --q) {
      const int inner_node = shell_node(q, j);
      const int outer_node = shell_node(q + 1, j);
      if (inner_node < 0 || inner_node >= n_nodes ||
          outer_node < 0 || outer_node >= n_nodes) {
        continue;
      }
      const int ii = index_by_node[static_cast<std::size_t>(inner_node)];
      const int io = index_by_node[static_cast<std::size_t>(outer_node)];
      if (ii < 0 || io < 0) {
        continue;
      }
      const double floor = polar_shell_axis_order_floor(input, ii, io);
      if (north_pole) {
        const double upper = z_target[static_cast<std::size_t>(io)] - floor;
        if (!(z_target[static_cast<std::size_t>(ii)] < upper)) {
          z_target[static_cast<std::size_t>(ii)] = upper;
        }
      } else {
        const double lower = z_target[static_cast<std::size_t>(io)] + floor;
        if (!(z_target[static_cast<std::size_t>(ii)] > lower)) {
          z_target[static_cast<std::size_t>(ii)] = lower;
        }
      }
    }
  };

  project_column(0, true);
  project_column(shell.n_j_cells, false);
}

int cell_active_nverts(const mesh::Mesh& mesh, const int cell) {
  TENRYU_ASSERT(cell >= 0 && cell < mesh.topo.n_cells,
                "axis ALE active nverts cell out of range");
  if (mesh.cell_nverts.size() == static_cast<std::size_t>(mesh.topo.n_cells)) {
    return mesh::mesh_topo_cell_active_nverts(mesh.cell_nverts, cell);
  }
  return mesh::kMeshTopoCellStorageSlots;
}

const std::vector<std::uint8_t>* active_cell_nverts_view(const mesh::Mesh& mesh) {
  if (mesh.cell_nverts.size() != static_cast<std::size_t>(mesh.topo.n_cells)) {
    return nullptr;
  }
  return std::any_of(mesh.cell_nverts.begin(),
                     mesh.cell_nverts.end(),
                     [](const std::uint8_t nverts) { return nverts == 3U; })
             ? &mesh.cell_nverts
             : nullptr;
}

bool is_central_core_cell(const mesh::MultiBlockTopology& mb, const int cell) {
  if (cell < 0 || static_cast<std::size_t>(cell) >= mb.cell_block_id.size()) {
    return false;
  }
  const int block_id = mb.cell_block_id[static_cast<std::size_t>(cell)];
  if (block_id < 0 || static_cast<std::size_t>(block_id) >= mb.blocks.size()) {
    return false;
  }
  return mb.blocks[static_cast<std::size_t>(block_id)].role ==
         mesh::BlockRole::CENTRAL_CORE;
}

bool is_fan_cell(const mesh::MultiBlockTopology& mb, const int cell) {
  if (cell < 0 || static_cast<std::size_t>(cell) >= mb.cell_block_id.size()) {
    return false;
  }
  const int block_id = mb.cell_block_id[static_cast<std::size_t>(cell)];
  if (block_id < 0 || static_cast<std::size_t>(block_id) >= mb.blocks.size()) {
    return false;
  }
  const mesh::BlockRole role = mb.blocks[static_cast<std::size_t>(block_id)].role;
  return role == mesh::BlockRole::NORTH_FAN ||
         role == mesh::BlockRole::EAST_FAN ||
         role == mesh::BlockRole::SOUTH_FAN;
}

bool is_polar_shell_cell(const mesh::MultiBlockTopology& mb, const int cell) {
  if (cell < 0 || static_cast<std::size_t>(cell) >= mb.cell_block_id.size()) {
    return false;
  }
  const int block_id = mb.cell_block_id[static_cast<std::size_t>(cell)];
  if (block_id < 0 || static_cast<std::size_t>(block_id) >= mb.blocks.size()) {
    return false;
  }
  return mb.blocks[static_cast<std::size_t>(block_id)].role ==
         mesh::BlockRole::POLAR_SHELL;
}

int shell_block_id(const mesh::MultiBlockTopology& mb) {
  for (int b = 0; b < static_cast<int>(mb.blocks.size()); ++b) {
    if (mb.blocks[static_cast<std::size_t>(b)].role ==
        mesh::BlockRole::POLAR_SHELL) {
      return b;
    }
  }
  return -1;
}

void append_unique_node(std::vector<int>& nodes, const int self, const int node) {
  if (node == self) {
    return;
  }
  if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) {
    nodes.push_back(node);
  }
}

double rz_quad_volume_exact(const std::array<double, 4>& r,
                            const std::array<double, 4>& z) {
  double sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    sum += (r[static_cast<std::size_t>(k)] +
            r[static_cast<std::size_t>(kp)]) *
           (r[static_cast<std::size_t>(k)] *
                z[static_cast<std::size_t>(kp)] -
            r[static_cast<std::size_t>(kp)] *
                z[static_cast<std::size_t>(k)]);
  }
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  return pi_over_three * sum;
}

double rz_cell_volume_exact(const std::array<double, 4>& r,
                            const std::array<double, 4>& z,
                            const int active_nverts) {
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    return rz_quad_volume_exact(r, z);
  }
  return tenryu::mesh::rz_triangle_volume_exact(r[0], z[0],
                                                r[1], z[1],
                                                r[2], z[2]);
}

double hourglass_shear_from_quad(const std::array<double, 4>& r,
                                 const std::array<double, 4>& z) {
  double edge_sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    edge_sum += edge_length(r, z, k);
  }
  const double edge_scale = 0.25 * edge_sum;
  if (!(edge_scale > 0.0) || !std::isfinite(edge_scale)) {
    return std::numeric_limits<double>::infinity();
  }
  const double hr = r[0] - r[1] + r[2] - r[3];
  const double hz = z[0] - z[1] + z[2] - z[3];
  return std::hypot(hr, hz) / edge_scale;
}

AxisAleCellTargetQuality quality_from_quad(const std::array<double, 4>& r,
                                           const std::array<double, 4>& z,
                                           const int orientation_sign) {
  AxisAleCellTargetQuality quality;
  quality.shear = hourglass_shear_from_quad(r, z);
  quality.aspect_ratio = aspect_ratio_from_quad(r, z);
  quality.min_corner_j =
      min_corner_j_from_quad(r, z, orientation_sign);
  // The aspect ratio is a reported diagnostic only and is INF for a
  // degenerate quad (coincident vertices). Sampling must not require it:
  // a degenerate candidate has corner_j = 0 and must be SAMPLED so the
  // audit can fail it (an unsampled degenerate candidate would evade the
  // audit fallback and be installed).
  quality.sampled = std::isfinite(quality.shear) &&
                    std::isfinite(quality.min_corner_j);
  return quality;
}

AxisAleCellTargetQuality quality_from_active_cell(
    const std::array<double, 4>& r,
    const std::array<double, 4>& z,
    const int active_nverts,
    const int orientation_sign) {
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    return quality_from_quad(r, z, orientation_sign);
  }
  AxisAleCellTargetQuality quality;
  quality.shear = 0.0;
  quality.aspect_ratio = aspect_ratio_from_polygon(r, z, active_nverts);
  quality.min_corner_j =
      min_corner_j_from_polygon(r, z, active_nverts, orientation_sign);
  // See quality_from_quad: aspect is diagnostic-only and INF when
  // degenerate; sampling requires only the decision metrics.
  quality.sampled = std::isfinite(quality.shear) &&
                    std::isfinite(quality.min_corner_j);
  return quality;
}

double sigma2_from_columns(const double a0,
                           const double a1,
                           const double b0,
                           const double b1) {
  const double aa = a0 * a0 + a1 * a1;
  const double bb = b0 * b0 + b1 * b1;
  const double ab = a0 * b0 + a1 * b1;
  const double trace = aa + bb;
  const double det = aa * bb - ab * ab;
  if (!(std::isfinite(trace) && std::isfinite(det)) || trace < 0.0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double disc = std::max(0.0, trace * trace - 4.0 * det);
  const double lambda_min = 0.5 * (trace - std::sqrt(disc));
  return std::sqrt(std::max(0.0, lambda_min));
}

double diagonal_corner_j2_from_quad(const std::array<double, 4>& r,
                                    const std::array<double, 4>& z) {
  return cross2(r[3] - r[2], z[3] - z[2],
                r[1] - r[2], z[1] - z[2]);
}

double score_min4(const double a,
                  const double b,
                  const double c,
                  const double d) {
  double out = std::numeric_limits<double>::infinity();
  const std::array<double, 4> values{{a, b, c, d}};
  for (const double v : values) {
    if (!std::isfinite(v)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    out = std::min(out, v);
  }
  return out;
}

bool full_patch_barrier_accepts(
    const mesh::Mesh& mesh,
    const std::vector<double>& candidate_r,
    const std::vector<double>& candidate_z,
    const std::vector<double>& reference_r,
    const std::vector<double>& reference_z,
    const double scale,
    const double floor_rel,
    int& limited_cell) {
  limited_cell = -1;
  if (!mesh.topo.multiblock.has_value()) {
    return true;
  }
  const auto& mb = *mesh.topo.multiblock;
  bool sampled = false;
  for (int c = 0; c < mesh.topo.n_cells; ++c) {
    if (!is_central_core_cell(mb, c)) {
      continue;
    }
    const int active_nverts = cell_active_nverts(mesh, c);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    bool touches_axis = false;
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      touches_axis =
          touches_axis ||
          ((mesh.topo.node_flags[static_cast<std::size_t>(n)] &
            mesh::NODE_AXIS) != 0U);
    }
    if (!touches_axis) {
      continue;
    }
    const AxisAleDiagonalCornerMetrics m =
        compute_axis_ale_diagonal_corner_metrics(mesh,
                                                 candidate_r,
                                                 candidate_z,
                                                 reference_r,
                                                 reference_z,
                                                 scale,
                                                 c);
    if (!m.sampled) {
      continue;
    }
    sampled = true;
    const int n0 = m.nodes[0];
    const int n1 = m.nodes[1];
    const int n3 = m.nodes[3];
    const double b0 =
        0.5 *
        (std::hypot(reference_r[static_cast<std::size_t>(n1)] -
                       reference_r[static_cast<std::size_t>(n0)],
                   reference_z[static_cast<std::size_t>(n1)] -
                       reference_z[static_cast<std::size_t>(n0)]) +
         std::hypot(reference_r[static_cast<std::size_t>(n3)] -
                       reference_r[static_cast<std::size_t>(n0)],
                   reference_z[static_cast<std::size_t>(n3)] -
                       reference_z[static_cast<std::size_t>(n0)]));
    const double b_floor = floor_rel * scale * b0;
    if (!(std::isfinite(m.j2) && m.j2 > 0.0 &&
          std::isfinite(m.psi_eff) && std::isfinite(m.psi_target) &&
          m.psi_eff >= floor_rel * m.psi_target &&
          std::isfinite(m.b_eff) && m.b_eff >= b_floor &&
          std::isfinite(m.s2) && m.s2 >= floor_rel &&
          std::isfinite(m.q2) && m.q2 >= floor_rel &&
          std::isfinite(m.h2) && m.h2 >= floor_rel)) {
      limited_cell = c;
      return false;
    }
  }
  return sampled;
}

}  // namespace

std::vector<double> project_weighted_lower_bound_pava(
    const std::vector<double>& z_tilde,
    const std::vector<double>& lumped_mass,
    const std::vector<double>& initial_edge_length,
    const std::vector<double>& adjacent_cell_area,
    const double eta_floor,
    const double dt) {
  const int n = static_cast<int>(z_tilde.size());
  TENRYU_ASSERT(lumped_mass.size() == z_tilde.size(),
                "axis ALE PAVA requires one mass per node");
  if (n == 0) {
    return {};
  }
  TENRYU_ASSERT(std::isfinite(dt) && dt > 0.0,
                "axis ALE PAVA requires positive finite dt");
  TENRYU_ASSERT(std::isfinite(eta_floor) && eta_floor >= 0.0,
                "axis ALE PAVA requires finite non-negative eta_floor");
  TENRYU_ASSERT(initial_edge_length.size() == static_cast<std::size_t>(n - 1),
                "axis ALE PAVA requires one initial length per edge");
  TENRYU_ASSERT(adjacent_cell_area.size() == static_cast<std::size_t>(n - 1),
                "axis ALE PAVA requires one adjacent area per edge");

  std::vector<double> prefix(static_cast<std::size_t>(n), 0.0);
  for (int e = 0; e + 1 < n; ++e) {
    const double l0 = initial_edge_length[static_cast<std::size_t>(e)];
    const double area = adjacent_cell_area[static_cast<std::size_t>(e)];
    TENRYU_ASSERT(std::isfinite(l0) && l0 > 0.0,
                  "axis ALE PAVA initial edge length must be positive");
    TENRYU_ASSERT(std::isfinite(area) && area > 0.0,
                  "axis ALE PAVA adjacent cell area must be positive");
    const double l_min = eta_floor * std::min(l0, std::sqrt(area));
    TENRYU_ASSERT(std::isfinite(l_min) && l_min >= 0.0,
                  "axis ALE PAVA spacing floor must be finite non-negative");
    prefix[static_cast<std::size_t>(e + 1)] =
        prefix[static_cast<std::size_t>(e)] + l_min;
  }

  std::vector<PavaBlock> blocks;
  blocks.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const double z = z_tilde[static_cast<std::size_t>(i)];
    const double m = lumped_mass[static_cast<std::size_t>(i)];
    TENRYU_ASSERT(std::isfinite(z),
                  "axis ALE PAVA predicted z must be finite");
    TENRYU_ASSERT(std::isfinite(m) && m > 0.0,
                  "axis ALE PAVA node mass must be positive");
    const double y = z - prefix[static_cast<std::size_t>(i)];
    blocks.push_back(PavaBlock{i, i + 1, m, y});
    while (blocks.size() >= 2U) {
      PavaBlock& prev = blocks[blocks.size() - 2U];
      PavaBlock& curr = blocks[blocks.size() - 1U];
      if (prev.mean <= curr.mean) {
        break;
      }
      const double merged_weight = prev.weight + curr.weight;
      TENRYU_ASSERT(std::isfinite(merged_weight) && merged_weight > 0.0,
                    "axis ALE PAVA merged weight must be positive");
      const double merged_mean =
          (prev.weight * prev.mean + curr.weight * curr.mean) /
          merged_weight;
      prev.end = curr.end;
      prev.weight = merged_weight;
      prev.mean = merged_mean;
      blocks.pop_back();
    }
  }

  std::vector<double> y_hat(static_cast<std::size_t>(n), 0.0);
  for (const PavaBlock& block : blocks) {
    for (int i = block.begin; i < block.end; ++i) {
      y_hat[static_cast<std::size_t>(i)] = block.mean;
    }
  }

  std::vector<double> z_target(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    z_target[static_cast<std::size_t>(i)] =
        y_hat[static_cast<std::size_t>(i)] +
        prefix[static_cast<std::size_t>(i)];
  }
  return z_target;
}

FirstOffAxisRingDiagnostics compute_first_off_axis_ring_diagnostics(
    const mesh::Mesh& mesh,
    const std::vector<int>& axis_node_ids) {
  FirstOffAxisRingDiagnostics diag;
  if (axis_node_ids.empty() || !mesh.topo.multiblock.has_value()) {
    return diag;
  }
  const int n_nodes = mesh.topo.n_nodes;
  const int n_cells = mesh.topo.n_cells;
  TENRYU_ASSERT(n_nodes >= 0 && n_cells >= 0,
                "axis ALE diagnostics require non-negative mesh sizes");
  const auto& mb = *mesh.topo.multiblock;
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "axis ALE diagnostics require cell-node CSR offsets");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) * 4U,
                "axis ALE diagnostics require cell-node CSR indices");

  std::vector<std::uint8_t> is_axis_chain(static_cast<std::size_t>(n_nodes), 0U);
  for (const int n : axis_node_ids) {
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "axis ALE diagnostics node id out of range");
    is_axis_chain[static_cast<std::size_t>(n)] = 1U;
  }

  const std::vector<double> r =
      copy_node_field(mesh.node_r, n_nodes, "axis ALE diagnostics memcpy node_r failed");
  const std::vector<double> z =
      copy_node_field(mesh.node_z, n_nodes, "axis ALE diagnostics memcpy node_z failed");

  for (int c = 0; c < n_cells; ++c) {
    const int active_nverts = cell_active_nverts(mesh, c);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    bool touches_axis = false;
    bool has_off_axis_node = false;
    std::array<int, 4> nodes{};
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "axis ALE diagnostics cell node out of range");
      nodes[static_cast<std::size_t>(k)] = n;
      if (is_axis_chain[static_cast<std::size_t>(n)] != 0U) {
        touches_axis = true;
      } else {
        has_off_axis_node = true;
      }
    }
    if (!touches_axis || !has_off_axis_node) {
      continue;
    }

    std::array<double, 4> cr{};
    std::array<double, 4> cz{};
    for (int k = 0; k < active_nverts; ++k) {
      const int n = nodes[static_cast<std::size_t>(k)];
      cr[static_cast<std::size_t>(k)] = r[static_cast<std::size_t>(n)];
      cz[static_cast<std::size_t>(k)] = z[static_cast<std::size_t>(n)];
    }
    const double min_edge =
        min_edge_length_from_polygon(cr, cz, active_nverts);
    const double min_altitude =
        min_altitude_from_polygon(cr, cz, active_nverts);
    if (std::isfinite(min_edge) && min_edge < diag.min_edge_length) {
      diag.min_edge_length = min_edge;
      diag.min_edge_cell = c;
    }
    if (std::isfinite(min_altitude) && min_altitude < diag.min_altitude) {
      diag.min_altitude = min_altitude;
      diag.min_altitude_cell = c;
    }
    diag.sampled = true;
  }
  return diag;
}

AxisAleCoreQualityTrigger compute_axis_ale_core_quality_trigger(
    const mesh::Mesh& mesh,
    const double shear_threshold,
    const AxisAleCoreQualityBaseline* baseline) {
  AxisAleCoreQualityTrigger trigger;
  trigger.shear_threshold = shear_threshold;
  TENRYU_ASSERT(std::isfinite(shear_threshold) && shear_threshold >= 0.0,
                "axis ALE core quality trigger requires non-negative finite shear threshold");
  if (!mesh.topo.multiblock.has_value()) {
    return trigger;
  }
  const auto& mb = *mesh.topo.multiblock;
  if (mb.block_count != 5) {
    return trigger;
  }
  const int n_nodes = mesh.topo.n_nodes;
  const int n_cells = mesh.topo.n_cells;
  TENRYU_ASSERT(n_nodes >= 0 && n_cells >= 0,
                "axis ALE core quality trigger requires non-negative mesh sizes");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "axis ALE core quality trigger requires cell-node CSR offsets");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) * 4U,
                "axis ALE core quality trigger requires cell-node CSR indices");
  TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                    static_cast<std::size_t>(n_cells),
                "axis ALE core quality trigger requires cell orientation signs");

  std::vector<std::uint8_t> core_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> fan_incident_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> seam_node(static_cast<std::size_t>(n_nodes), 0U);
  for (int c = 0; c < n_cells; ++c) {
    const bool central_core = is_central_core_cell(mb, c);
    const bool fan = is_fan_cell(mb, c);
    if (!central_core && !fan) {
      continue;
    }
    const int active_nverts = cell_active_nverts(mesh, c);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "axis ALE core quality trigger cell node out of range");
      if (central_core) {
        core_node[static_cast<std::size_t>(n)] = 1U;
      } else {
        fan_incident_node[static_cast<std::size_t>(n)] = 1U;
      }
    }
  }
  for (int n = 0; n < n_nodes; ++n) {
    seam_node[static_cast<std::size_t>(n)] =
        (core_node[static_cast<std::size_t>(n)] != 0U &&
         fan_incident_node[static_cast<std::size_t>(n)] != 0U)
            ? 1U
            : 0U;
  }

  const std::vector<double> r =
      copy_node_field(mesh.node_r, n_nodes, "axis ALE core quality trigger memcpy node_r failed");
  const std::vector<double> z =
      copy_node_field(mesh.node_z, n_nodes, "axis ALE core quality trigger memcpy node_z failed");

  for (int c = 0; c < n_cells; ++c) {
    const bool central_core = is_central_core_cell(mb, c);
    const bool fan = is_fan_cell(mb, c);
    const int active_nverts = cell_active_nverts(mesh, c);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    bool sample_cell = central_core;
    if (!sample_cell && fan) {
      for (int k = 0; k < active_nverts; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        sample_cell =
            sample_cell || seam_node[static_cast<std::size_t>(n)] != 0U;
      }
    }
    if (!sample_cell) {
      continue;
    }
    std::array<double, 4> cr{};
    std::array<double, 4> cz{};
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "axis ALE core quality trigger cell node out of range");
      cr[static_cast<std::size_t>(k)] = r[static_cast<std::size_t>(n)];
      cz[static_cast<std::size_t>(k)] = z[static_cast<std::size_t>(n)];
    }
    const AxisAleCellTargetQuality quality =
        quality_from_active_cell(
            cr,
            cz,
            active_nverts,
            mb.cell_orientation_sign[static_cast<std::size_t>(c)]);
    if (!quality.sampled) {
      continue;
    }
    trigger.sampled = true;
    ++trigger.sampled_cells;
    // The shear metric is the EXCESS over the cell's own pristine baseline
    // when one is supplied: rounded/D-shape seam cells are intrinsically
    // keystone-shaped, so absolute shear does not mean damage. The
    // corner-J leg stays absolute (a fold is damage regardless of shape).
    double shear_metric = quality.shear;
    if (baseline != nullptr && baseline->valid &&
        static_cast<std::size_t>(c) < baseline->shear_by_cell.size() &&
        std::isfinite(
            baseline->shear_by_cell[static_cast<std::size_t>(c)])) {
      shear_metric =
          quality.shear -
          baseline->shear_by_cell[static_cast<std::size_t>(c)];
    }
    if (shear_metric > trigger.max_shear) {
      trigger.max_shear = shear_metric;
      trigger.max_shear_cell = c;
    }
    if (quality.min_corner_j < trigger.min_corner_j) {
      trigger.min_corner_j = quality.min_corner_j;
      trigger.min_corner_j_cell = c;
    }
    if (shear_metric > shear_threshold || quality.min_corner_j <= 0.0) {
      trigger.active = true;
    }
  }
  return trigger;
}

AxisAleCoreQualityBaseline collect_axis_ale_core_quality_baseline(
    const mesh::Mesh& mesh) {
  AxisAleCoreQualityBaseline out;
  if (!mesh.topo.multiblock.has_value()) {
    return out;
  }
  const auto& mb = *mesh.topo.multiblock;
  if (mb.block_count != 5) {
    return out;
  }
  const int n_cells = mesh.topo.n_cells;
  out.shear_by_cell.assign(static_cast<std::size_t>(n_cells),
                           std::numeric_limits<double>::quiet_NaN());
  const int n_nodes = mesh.topo.n_nodes;
  const std::vector<double> r = copy_node_field(
      mesh.node_r, n_nodes,
      "axis ALE core quality baseline memcpy node_r failed");
  const std::vector<double> z = copy_node_field(
      mesh.node_z, n_nodes,
      "axis ALE core quality baseline memcpy node_z failed");
  for (int c = 0; c < n_cells; ++c) {
    if (!is_central_core_cell(mb, c) && !is_fan_cell(mb, c)) {
      continue;
    }
    const int active_nverts = cell_active_nverts(mesh, c);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    std::array<double, 4> cr{};
    std::array<double, 4> cz{};
    for (int k = 0; k < active_nverts; ++k) {
      const int n =
          mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      cr[static_cast<std::size_t>(k)] = r[static_cast<std::size_t>(n)];
      cz[static_cast<std::size_t>(k)] = z[static_cast<std::size_t>(n)];
    }
    const AxisAleCellTargetQuality quality = quality_from_active_cell(
        cr, cz, active_nverts,
        mb.cell_orientation_sign[static_cast<std::size_t>(c)]);
    if (quality.sampled) {
      out.shear_by_cell[static_cast<std::size_t>(c)] = quality.shear;
      out.valid = true;
    }
  }
  return out;
}

AxisAlePatchTarget compute_axis_ale_patch_winslow_target(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input,
    const AxisAleRezoneResult& axis_target) {
  AxisAlePatchTarget patch;
  TENRYU_ASSERT(std::isfinite(input.patch_volume_contraction_floor_frac) &&
                    input.patch_volume_contraction_floor_frac > 0.0 &&
                    input.patch_volume_contraction_floor_frac <= 1.0,
                "axis ALE patch target contraction floor must be in (0, 1]");
  TENRYU_ASSERT(std::isfinite(input.patch_volume_aggregate_rel_tol) &&
                    input.patch_volume_aggregate_rel_tol >= 0.0 &&
                    input.patch_volume_aggregate_rel_tol < 1.0,
                "axis ALE patch target aggregate volume tolerance must be in [0, 1)");
  patch.volume_guard.contraction_floor_frac =
      input.patch_volume_contraction_floor_frac;
  patch.volume_guard.aggregate_rel_tol =
      input.patch_volume_aggregate_rel_tol;
  if (!axis_target.active || axis_target.node_ids.empty() ||
      !mesh.topo.multiblock.has_value()) {
    return patch;
  }
  const auto& mb = *mesh.topo.multiblock;
  if (mb.block_count != 5) {
    return patch;
  }
  const int n_nodes = mesh.topo.n_nodes;
  const int n_cells = mesh.topo.n_cells;
  TENRYU_ASSERT(n_nodes >= 0 && n_cells >= 0,
                "axis ALE patch target requires non-negative mesh sizes");
  TENRYU_ASSERT(input.node_ids.size() == axis_target.node_ids.size() &&
                    input.z_tilde.size() == axis_target.z_target.size() &&
                    axis_target.node_ids.size() == axis_target.z_target.size(),
                "axis ALE patch target requires paired axis arrays");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "axis ALE patch target requires cell-node CSR offsets");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) * 4U,
                "axis ALE patch target requires cell-node CSR indices");
  TENRYU_ASSERT(mb.cell_block_id.size() == static_cast<std::size_t>(n_cells),
                "axis ALE patch target requires cell block ids");
  TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                    static_cast<std::size_t>(n_cells),
                "axis ALE patch target requires cell orientation signs");
  TENRYU_ASSERT(input.patch_winslow_iterations >= 0,
                "axis ALE patch Winslow iterations must be non-negative");
  TENRYU_ASSERT(std::isfinite(input.patch_winslow_relaxation) &&
                    input.patch_winslow_relaxation >= 0.0 &&
                    input.patch_winslow_relaxation <= 1.0,
                "axis ALE patch Winslow relaxation must be in [0, 1]");

  std::vector<std::uint8_t> is_axis(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<double> axis_z_target(static_cast<std::size_t>(n_nodes),
                                    std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = 0; i < axis_target.node_ids.size(); ++i) {
    const int n = axis_target.node_ids[i];
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "axis ALE patch target axis node id out of range");
    is_axis[static_cast<std::size_t>(n)] = 1U;
    axis_z_target[static_cast<std::size_t>(n)] =
        node_is_pinned_cap_origin(mesh, n) ? 0.0 : axis_target.z_target[i];
  }

  std::vector<std::uint8_t> core_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> fan_incident_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> seam_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> fan_buffer_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> patch_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> fixed_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> guard_cell(static_cast<std::size_t>(n_cells), 0U);

  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int active_nverts = cell_active_nverts(mesh, c);
    if (is_central_core_cell(mb, c)) {
      guard_cell[static_cast<std::size_t>(c)] = 1U;
      for (int k = 0; k < active_nverts; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        TENRYU_ASSERT(n >= 0 && n < n_nodes,
                      "axis ALE patch target core cell node out of range");
        core_node[static_cast<std::size_t>(n)] = 1U;
      }
    } else if (is_fan_cell(mb, c)) {
      for (int k = 0; k < active_nverts; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        TENRYU_ASSERT(n >= 0 && n < n_nodes,
                      "axis ALE patch target fan cell node out of range");
        fan_incident_node[static_cast<std::size_t>(n)] = 1U;
      }
    }
  }
  for (int n = 0; n < n_nodes; ++n) {
    seam_node[static_cast<std::size_t>(n)] =
        (core_node[static_cast<std::size_t>(n)] != 0U &&
         fan_incident_node[static_cast<std::size_t>(n)] != 0U)
            ? 1U
            : 0U;
  }
  for (int c = 0; c < n_cells; ++c) {
    if (!is_fan_cell(mb, c)) {
      continue;
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int active_nverts = cell_active_nverts(mesh, c);
    bool touches_core_seam = false;
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      touches_core_seam =
          touches_core_seam || seam_node[static_cast<std::size_t>(n)] != 0U;
    }
    if (!touches_core_seam) {
      continue;
    }
    guard_cell[static_cast<std::size_t>(c)] = 1U;
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      fan_buffer_node[static_cast<std::size_t>(n)] = 1U;
    }
  }

  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t nu = static_cast<std::size_t>(n);
    patch_node[nu] =
        (core_node[nu] != 0U || fan_buffer_node[nu] != 0U) ? 1U : 0U;
    if (patch_node[nu] == 0U) {
      continue;
    }
    ++patch.patch_nodes;
    if (core_node[nu] != 0U) {
      ++patch.core_nodes;
    }
    if (seam_node[nu] != 0U) {
      ++patch.seam_nodes;
    }
    if (fan_buffer_node[nu] != 0U && core_node[nu] == 0U) {
      ++patch.fan_transition_nodes;
      ++patch.fixed_fan_boundary_nodes;
      fixed_node[nu] = 1U;
    }
    if (is_axis[nu] != 0U) {
      ++patch.fixed_axis_nodes;
      fixed_node[nu] = 1U;
    }
  }
  if (patch.patch_nodes == 0) {
    return patch;
  }

  const std::vector<double> r =
      copy_node_field(mesh.node_r, n_nodes, "axis ALE patch target memcpy node_r failed");
  const std::vector<double> z =
      copy_node_field(mesh.node_z, n_nodes, "axis ALE patch target memcpy node_z failed");
  std::vector<double> cur_r = r;
  std::vector<double> cur_z = z;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t nu = static_cast<std::size_t>(n);
    if (patch_node[nu] != 0U && is_axis[nu] != 0U) {
      cur_r[nu] = 0.0;
      cur_z[nu] = axis_z_target[nu];
    }
  }
  std::vector<double> next_r = cur_r;
  std::vector<double> next_z = cur_z;

  const ReverseCellNodeCSR reverse_csr =
      build_reverse_cell_node_csr(
          mb, n_nodes, active_cell_nverts_view(mesh), mesh.corner_stride);
  const double omega = input.patch_winslow_relaxation;
  for (int iter = 0; iter < input.patch_winslow_iterations; ++iter) {
    next_r = cur_r;
    next_z = cur_z;
    if (omega <= 0.0) {
      break;
    }
    for (int n = 0; n < n_nodes; ++n) {
      const std::size_t nu = static_cast<std::size_t>(n);
      if (patch_node[nu] == 0U || fixed_node[nu] != 0U) {
        continue;
      }
      const int incident_begin = reverse_csr.node_offsets[nu];
      const int incident_end = reverse_csr.node_offsets[nu + 1U];
      if (incident_end <= incident_begin) {
        continue;
      }
      bool seam_touching = seam_node[nu] != 0U;
      int incident_block = -1;
      for (int p = incident_begin; p < incident_end; ++p) {
        const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
        const int block = mb.cell_block_id[static_cast<std::size_t>(c)];
        if (incident_block < 0) {
          incident_block = block;
        } else if (block != incident_block) {
          seam_touching = true;
        }
      }

      std::vector<int> neighbors;
      neighbors.reserve(16U);
      for (int p = incident_begin; p < incident_end; ++p) {
        const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
        const int cell_begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
        const int cell_end =
            mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
        const int active_nverts = cell_active_nverts(mesh, c);
        if (seam_touching) {
          const int corner = reverse_csr.node_corners[static_cast<std::size_t>(p)];
          TENRYU_ASSERT(corner >= 0 && corner < active_nverts &&
                            cell_end - cell_begin == 4,
                        "axis ALE patch target seam corner out of range");
          const int local_a = (corner + 1) % active_nverts;
          const int local_b = (corner + active_nverts - 1) % active_nverts;
          append_unique_node(
              neighbors,
              n,
              mb.cell_node_csr_indices[static_cast<std::size_t>(cell_begin + local_a)]);
          append_unique_node(
              neighbors,
              n,
              mb.cell_node_csr_indices[static_cast<std::size_t>(cell_begin + local_b)]);
          continue;
        }
        for (int q = cell_begin; q < cell_begin + active_nverts; ++q) {
          append_unique_node(
              neighbors,
              n,
              mb.cell_node_csr_indices[static_cast<std::size_t>(q)]);
        }
      }
      if (neighbors.empty()) {
        continue;
      }
      double sum_r = 0.0;
      double sum_z = 0.0;
      for (const int nb : neighbors) {
        TENRYU_ASSERT(nb >= 0 && nb < n_nodes,
                      "axis ALE patch target neighbor out of range");
        sum_r += cur_r[static_cast<std::size_t>(nb)];
        sum_z += cur_z[static_cast<std::size_t>(nb)];
      }
      const double inv = 1.0 / static_cast<double>(neighbors.size());
      next_r[nu] = (1.0 - omega) * cur_r[nu] + omega * sum_r * inv;
      next_z[nu] = (1.0 - omega) * cur_z[nu] + omega * sum_z * inv;
    }
    cur_r.swap(next_r);
    cur_z.swap(next_z);
  }

  std::vector<double> full_target_r = r;
  std::vector<double> full_target_z = z;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t nu = static_cast<std::size_t>(n);
    if (is_axis[nu] != 0U) {
      full_target_r[nu] = 0.0;
      full_target_z[nu] = axis_z_target[nu];
    }
  }
  patch.active = true;
  patch.node_ids.reserve(static_cast<std::size_t>(patch.core_nodes));
  patch.r_target.reserve(static_cast<std::size_t>(patch.core_nodes));
  patch.z_target.reserve(static_cast<std::size_t>(patch.core_nodes));
  patch.beta.reserve(static_cast<std::size_t>(patch.core_nodes));
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t nu = static_cast<std::size_t>(n);
    if (patch_node[nu] == 0U || is_axis[nu] != 0U) {
      continue;
    }
    const double beta = core_node[nu] != 0U ? 1.0 : 0.0;
    full_target_r[nu] = r[nu] + beta * (cur_r[nu] - r[nu]);
    full_target_z[nu] = z[nu] + beta * (cur_z[nu] - z[nu]);
    if (beta > 0.0) {
      patch.node_ids.push_back(n);
      patch.r_target.push_back(full_target_r[nu]);
      patch.z_target.push_back(full_target_z[nu]);
      patch.beta.push_back(beta);
    }
  }

  AxisAleTargetVolumeGuard guard;
  guard.contraction_floor_frac = input.patch_volume_contraction_floor_frac;
  guard.aggregate_rel_tol = input.patch_volume_aggregate_rel_tol;
  for (int c = 0; c < n_cells; ++c) {
    if (guard_cell[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int active_nverts = cell_active_nverts(mesh, c);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    std::array<double, 4> current_r{};
    std::array<double, 4> current_z{};
    std::array<double, 4> target_r{};
    std::array<double, 4> target_z{};
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      const std::size_t nu = static_cast<std::size_t>(n);
      current_r[static_cast<std::size_t>(k)] = r[nu];
      current_z[static_cast<std::size_t>(k)] = z[nu];
      target_r[static_cast<std::size_t>(k)] = full_target_r[nu];
      target_z[static_cast<std::size_t>(k)] = full_target_z[nu];
    }
    const double sign =
        mb.cell_orientation_sign[static_cast<std::size_t>(c)] < 0 ? -1.0 : 1.0;
    const double current_volume =
        sign * rz_cell_volume_exact(current_r, current_z, active_nverts);
    const double target_volume =
        sign * rz_cell_volume_exact(target_r, target_z, active_nverts);
    const double ratio =
        (std::isfinite(current_volume) && current_volume > 0.0 &&
         std::isfinite(target_volume))
            ? target_volume / current_volume
            : -std::numeric_limits<double>::infinity();
    if (ratio < guard.min_ratio) {
      guard.min_ratio = ratio;
      guard.min_current_volume = current_volume;
      guard.min_target_volume = target_volume;
      guard.min_ratio_cell = c;
    }
    if (std::isfinite(current_volume)) {
      guard.aggregate_current_volume += current_volume;
    }
    if (std::isfinite(target_volume)) {
      guard.aggregate_target_volume += target_volume;
    }
    const double allowed = guard.contraction_floor_frac * current_volume;
    if (!(std::isfinite(current_volume) && current_volume > 0.0 &&
          std::isfinite(target_volume) && target_volume >= allowed)) {
      guard.passed = false;
    }
    guard.sampled = true;
  }
  if (guard.sampled) {
    if (std::isfinite(guard.aggregate_current_volume) &&
        guard.aggregate_current_volume > 0.0 &&
        std::isfinite(guard.aggregate_target_volume)) {
      guard.aggregate_ratio =
          guard.aggregate_target_volume / guard.aggregate_current_volume;
      const double aggregate_allowed =
          (1.0 - guard.aggregate_rel_tol) *
          guard.aggregate_current_volume;
      if (guard.aggregate_target_volume < aggregate_allowed) {
        guard.passed = false;
      }
    } else {
      guard.passed = false;
      guard.aggregate_ratio = -std::numeric_limits<double>::infinity();
    }
  }
  patch.volume_guard = guard;
  return patch;
}

AxisAlePatchTarget compute_axis_ale_scaled_full_patch_target(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input,
    const AxisAleRezoneResult& axis_target,
    const std::vector<double>& scaled_target_r,
    const std::vector<double>& scaled_target_z,
    const AxisAleScaledFullPatchOptions& options) {
  AxisAlePatchTarget patch;
  if (!axis_target.active || axis_target.node_ids.empty() ||
      !mesh.topo.multiblock.has_value()) {
    return patch;
  }
  const auto& mb = *mesh.topo.multiblock;
  if (mb.block_count != 5) {
    return patch;
  }
  const int n_nodes = mesh.topo.n_nodes;
  const int n_cells = mesh.topo.n_cells;
  TENRYU_ASSERT(n_nodes >= 0 && n_cells >= 0,
                "axis ALE scaled full patch target requires non-negative mesh sizes");
  TENRYU_ASSERT(scaled_target_r.size() == static_cast<std::size_t>(n_nodes) &&
                    scaled_target_z.size() == static_cast<std::size_t>(n_nodes),
                "axis ALE scaled full patch target requires one scaled coordinate per node");
  TENRYU_ASSERT(axis_target.node_ids.size() == axis_target.z_target.size(),
                "axis ALE scaled full patch target requires paired axis arrays");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "axis ALE scaled full patch target requires cell-node CSR offsets");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) * 4U,
                "axis ALE scaled full patch target requires cell-node CSR indices");
  TENRYU_ASSERT(options.polar_shell_rings >= 0,
                "axis ALE scaled full patch shell ring count must be non-negative");
  TENRYU_ASSERT(options.winslow_iterations >= 0,
                "axis ALE scaled full patch Winslow iterations must be non-negative");
  TENRYU_ASSERT(std::isfinite(options.winslow_relaxation) &&
                    options.winslow_relaxation >= 0.0 &&
                    options.winslow_relaxation <= 1.0,
                "axis ALE scaled full patch Winslow relaxation must be in [0, 1]");
  TENRYU_ASSERT(std::isfinite(options.diagonal_barrier_floor_rel) &&
                    options.diagonal_barrier_floor_rel > 0.0,
                "axis ALE scaled full patch barrier floor must be positive");
  TENRYU_ASSERT(options.diagonal_barrier_max_halves >= 0,
                "axis ALE scaled full patch barrier iteration count must be non-negative");

  std::vector<std::uint8_t> is_axis(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<double> axis_z_target(static_cast<std::size_t>(n_nodes),
                                    std::numeric_limits<double>::quiet_NaN());
  for (std::size_t i = 0; i < axis_target.node_ids.size(); ++i) {
    const int n = axis_target.node_ids[i];
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "axis ALE scaled full patch axis node id out of range");
    is_axis[static_cast<std::size_t>(n)] = 1U;
    axis_z_target[static_cast<std::size_t>(n)] =
        node_is_pinned_cap_origin(mesh, n) ? 0.0 : axis_target.z_target[i];
  }

  std::vector<std::uint8_t> core_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> fan_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> shell_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> seam_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> patch_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<std::uint8_t> fixed_node(static_cast<std::size_t>(n_nodes), 0U);
  std::vector<double> beta_node(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<std::uint8_t> guard_cell(static_cast<std::size_t>(n_cells), 0U);

  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int active_nverts = cell_active_nverts(mesh, c);
    const bool core = is_central_core_cell(mb, c);
    const bool fan = is_fan_cell(mb, c);
    if (!(core || fan)) {
      continue;
    }
    guard_cell[static_cast<std::size_t>(c)] = 1U;
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "axis ALE scaled full patch cell node out of range");
      if (core) {
        core_node[static_cast<std::size_t>(n)] = 1U;
      }
      if (fan) {
        fan_node[static_cast<std::size_t>(n)] = 1U;
      }
    }
  }
  const int shell_block = shell_block_id(mb);
  int shell_node_begin = -1;
  int shell_stride = 0;
  if (shell_block >= 0) {
    const auto& block = mb.blocks[static_cast<std::size_t>(shell_block)];
    shell_node_begin = block.owned_node_begin;
    shell_stride = block.n_j_cells + 1;
  }
  const int shell_rings = options.polar_shell_rings;
  for (int c = 0; c < n_cells; ++c) {
    if (!is_polar_shell_cell(mb, c)) {
      continue;
    }
    const int block_id = mb.cell_block_id[static_cast<std::size_t>(c)];
    const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
    const int q = (c - block.cell_begin) / block.n_j_cells;
    if (q >= shell_rings) {
      continue;
    }
    guard_cell[static_cast<std::size_t>(c)] = 1U;
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int active_nverts = cell_active_nverts(mesh, c);
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "axis ALE scaled full patch shell node out of range");
      shell_node[static_cast<std::size_t>(n)] = 1U;
    }
  }

  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t nu = static_cast<std::size_t>(n);
    seam_node[nu] = (core_node[nu] != 0U && fan_node[nu] != 0U) ? 1U : 0U;
    patch_node[nu] =
        (core_node[nu] != 0U || fan_node[nu] != 0U || shell_node[nu] != 0U)
            ? 1U
            : 0U;
    if (patch_node[nu] == 0U) {
      continue;
    }
    double beta = 1.0;
    if (shell_node[nu] != 0U && core_node[nu] == 0U && fan_node[nu] == 0U &&
        shell_node_begin >= 0 && shell_stride > 0 && n >= shell_node_begin) {
      const int q = (n - shell_node_begin) / shell_stride;
      beta = shell_rings > 0
                 ? std::max(0.0,
                            static_cast<double>(shell_rings - q) /
                                static_cast<double>(shell_rings))
                 : 0.0;
    }
    beta_node[nu] = beta;
    ++patch.patch_nodes;
    if (core_node[nu] != 0U) {
      ++patch.core_nodes;
    }
    if (seam_node[nu] != 0U) {
      ++patch.seam_nodes;
    }
    if (fan_node[nu] != 0U && core_node[nu] == 0U) {
      ++patch.fan_transition_nodes;
    }
    if (is_axis[nu] != 0U) {
      ++patch.fixed_axis_nodes;
      fixed_node[nu] = 1U;
    }
    if (core_node[nu] != 0U && is_axis[nu] == 0U) {
      fixed_node[nu] = 1U;
    }
    if (beta <= 0.0) {
      ++patch.fixed_fan_boundary_nodes;
      fixed_node[nu] = 1U;
    }
  }
  if (patch.patch_nodes == 0) {
    return patch;
  }

  const std::vector<double> r =
      copy_node_field(mesh.node_r, n_nodes, "axis ALE scaled full patch memcpy node_r failed");
  const std::vector<double> z =
      copy_node_field(mesh.node_z, n_nodes, "axis ALE scaled full patch memcpy node_z failed");
  std::vector<double> cur_r = r;
  std::vector<double> cur_z = z;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t nu = static_cast<std::size_t>(n);
    if (patch_node[nu] == 0U) {
      continue;
    }
    if (is_axis[nu] != 0U) {
      cur_r[nu] = 0.0;
      cur_z[nu] = axis_z_target[nu];
      continue;
    }
    const double beta = beta_node[nu];
    cur_r[nu] = r[nu] + beta * (scaled_target_r[nu] - r[nu]);
    cur_z[nu] = z[nu] + beta * (scaled_target_z[nu] - z[nu]);
  }
  std::vector<double> next_r = cur_r;
  std::vector<double> next_z = cur_z;

  const ReverseCellNodeCSR reverse_csr =
      build_reverse_cell_node_csr(
          mb, n_nodes, active_cell_nverts_view(mesh), mesh.corner_stride);
  const double omega = options.winslow_relaxation;
  for (int iter = 0; iter < options.winslow_iterations; ++iter) {
    next_r = cur_r;
    next_z = cur_z;
    if (omega <= 0.0) {
      break;
    }
    for (int n = 0; n < n_nodes; ++n) {
      const std::size_t nu = static_cast<std::size_t>(n);
      if (patch_node[nu] == 0U || fixed_node[nu] != 0U) {
        continue;
      }
      const int incident_begin = reverse_csr.node_offsets[nu];
      const int incident_end = reverse_csr.node_offsets[nu + 1U];
      if (incident_end <= incident_begin) {
        continue;
      }
      bool seam_touching = seam_node[nu] != 0U;
      int incident_block = -1;
      for (int p = incident_begin; p < incident_end; ++p) {
        const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
        const int block = mb.cell_block_id[static_cast<std::size_t>(c)];
        if (incident_block < 0) {
          incident_block = block;
        } else if (block != incident_block) {
          seam_touching = true;
        }
      }

      std::vector<int> neighbors;
      neighbors.reserve(16U);
      for (int p = incident_begin; p < incident_end; ++p) {
        const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
        const int cell_begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
        const int cell_end =
            mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
        const int active_nverts = cell_active_nverts(mesh, c);
        if (seam_touching) {
          const int corner = reverse_csr.node_corners[static_cast<std::size_t>(p)];
          TENRYU_ASSERT(corner >= 0 && corner < active_nverts &&
                            cell_end - cell_begin == 4,
                        "axis ALE scaled full patch seam corner out of range");
          append_unique_node(
              neighbors,
              n,
              mb.cell_node_csr_indices[static_cast<std::size_t>(
                  cell_begin + ((corner + 1) % active_nverts))]);
          append_unique_node(
              neighbors,
              n,
              mb.cell_node_csr_indices[static_cast<std::size_t>(
                  cell_begin + ((corner + active_nverts - 1) %
                                active_nverts))]);
          continue;
        }
        for (int q = cell_begin; q < cell_begin + active_nverts; ++q) {
          append_unique_node(
              neighbors,
              n,
              mb.cell_node_csr_indices[static_cast<std::size_t>(q)]);
        }
      }
      if (neighbors.empty()) {
        continue;
      }
      double sum_r = 0.0;
      double sum_z = 0.0;
      for (const int nb : neighbors) {
        TENRYU_ASSERT(nb >= 0 && nb < n_nodes,
                      "axis ALE scaled full patch neighbor out of range");
        sum_r += cur_r[static_cast<std::size_t>(nb)];
        sum_z += cur_z[static_cast<std::size_t>(nb)];
      }
      const double inv = 1.0 / static_cast<double>(neighbors.size());
      next_r[nu] = (1.0 - omega) * cur_r[nu] + omega * sum_r * inv;
      next_z[nu] = (1.0 - omega) * cur_z[nu] + omega * sum_z * inv;
    }
    cur_r.swap(next_r);
    cur_z.swap(next_z);
  }

  std::vector<double> full_target_r = r;
  std::vector<double> full_target_z = z;
  for (std::size_t i = 0; i < axis_target.node_ids.size(); ++i) {
    const int n = axis_target.node_ids[i];
    full_target_r[static_cast<std::size_t>(n)] = 0.0;
    full_target_z[static_cast<std::size_t>(n)] =
        axis_z_target[static_cast<std::size_t>(n)];
  }
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t nu = static_cast<std::size_t>(n);
    if (patch_node[nu] == 0U || is_axis[nu] != 0U || beta_node[nu] <= 0.0) {
      continue;
    }
    full_target_r[nu] = cur_r[nu];
    full_target_z[nu] = cur_z[nu];
    patch.node_ids.push_back(n);
    patch.r_target.push_back(cur_r[nu]);
    patch.z_target.push_back(cur_z[nu]);
    patch.beta.push_back(beta_node[nu]);
  }

  if (options.diagonal_barrier_enabled) {
    patch.diagonal_barrier_sampled = true;
    double lambda = 1.0;
    int limited_cell = -1;
    std::vector<double> candidate_r = r;
    std::vector<double> candidate_z = z;
    for (int iter = 0; iter <= options.diagonal_barrier_max_halves; ++iter) {
      candidate_r = r;
      candidate_z = z;
      for (std::size_t i = 0; i < axis_target.node_ids.size(); ++i) {
        const int n = axis_target.node_ids[i];
        const std::size_t nu = static_cast<std::size_t>(n);
        candidate_r[nu] = r[nu] + lambda * (0.0 - r[nu]);
        candidate_z[nu] =
            z[nu] + lambda * (axis_z_target[nu] - z[nu]);
      }
      for (std::size_t i = 0; i < patch.node_ids.size(); ++i) {
        const int n = patch.node_ids[i];
        const std::size_t nu = static_cast<std::size_t>(n);
        candidate_r[nu] = r[nu] + lambda * (patch.r_target[i] - r[nu]);
        candidate_z[nu] = z[nu] + lambda * (patch.z_target[i] - z[nu]);
      }
      if (full_patch_barrier_accepts(mesh,
                                     candidate_r,
                                     candidate_z,
                                     scaled_target_r,
                                     scaled_target_z,
                                     1.0,
                                     options.diagonal_barrier_floor_rel,
                                     limited_cell)) {
        patch.diagonal_barrier_passed = true;
        patch.diagonal_barrier_lambda = lambda;
        patch.diagonal_barrier_iters = iter;
        patch.diagonal_barrier_limited_cell = -1;
        break;
      }
      patch.diagonal_barrier_passed = false;
      patch.diagonal_barrier_limited_cell = limited_cell;
      lambda *= 0.5;
    }
    if (patch.diagonal_barrier_lambda < 1.0) {
      for (std::size_t i = 0; i < patch.node_ids.size(); ++i) {
        const int n = patch.node_ids[i];
        const std::size_t nu = static_cast<std::size_t>(n);
        patch.r_target[i] =
            r[nu] + patch.diagonal_barrier_lambda * (patch.r_target[i] - r[nu]);
        patch.z_target[i] =
            z[nu] + patch.diagonal_barrier_lambda * (patch.z_target[i] - z[nu]);
      }
    }
  }

  AxisAleTargetVolumeGuard guard;
  guard.contraction_floor_frac = input.patch_volume_contraction_floor_frac;
  guard.aggregate_rel_tol = input.patch_volume_aggregate_rel_tol;
  for (int c = 0; c < n_cells; ++c) {
    if (guard_cell[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int active_nverts = cell_active_nverts(mesh, c);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    std::array<double, 4> current_r{};
    std::array<double, 4> current_z{};
    std::array<double, 4> target_r{};
    std::array<double, 4> target_z{};
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      const std::size_t nu = static_cast<std::size_t>(n);
      current_r[static_cast<std::size_t>(k)] = r[nu];
      current_z[static_cast<std::size_t>(k)] = z[nu];
      target_r[static_cast<std::size_t>(k)] = full_target_r[nu];
      target_z[static_cast<std::size_t>(k)] = full_target_z[nu];
    }
    const double sign =
        mb.cell_orientation_sign[static_cast<std::size_t>(c)] < 0 ? -1.0 : 1.0;
    const double current_volume =
        sign * rz_cell_volume_exact(current_r, current_z, active_nverts);
    const double target_volume =
        sign * rz_cell_volume_exact(target_r, target_z, active_nverts);
    const double ratio =
        (std::isfinite(current_volume) && current_volume > 0.0 &&
         std::isfinite(target_volume))
            ? target_volume / current_volume
            : -std::numeric_limits<double>::infinity();
    if (ratio < guard.min_ratio) {
      guard.min_ratio = ratio;
      guard.min_current_volume = current_volume;
      guard.min_target_volume = target_volume;
      guard.min_ratio_cell = c;
    }
    if (std::isfinite(current_volume)) {
      guard.aggregate_current_volume += current_volume;
    }
    if (std::isfinite(target_volume)) {
      guard.aggregate_target_volume += target_volume;
    }
    guard.sampled = true;
  }
  if (guard.sampled) {
    if (std::isfinite(guard.aggregate_current_volume) &&
        guard.aggregate_current_volume > 0.0 &&
        std::isfinite(guard.aggregate_target_volume)) {
      guard.aggregate_ratio =
          guard.aggregate_target_volume / guard.aggregate_current_volume;
    } else {
      guard.passed = false;
      guard.aggregate_ratio = -std::numeric_limits<double>::infinity();
    }
  }
  patch.volume_guard = guard;
  patch.active = true;
  return patch;
}

AxisAleDiagonalCornerMetrics compute_axis_ale_diagonal_corner_metrics(
    const mesh::Mesh& mesh,
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<double>& r_reference,
    const std::vector<double>& z_reference,
    const double scale,
    const int cell) {
  AxisAleDiagonalCornerMetrics metrics;
  metrics.cell = cell;
  if (!mesh.topo.multiblock.has_value() || cell < 0 ||
      cell >= mesh.topo.n_cells || !(std::isfinite(scale) && scale > 0.0)) {
    return metrics;
  }
  const int n_nodes = mesh.topo.n_nodes;
  if (r.size() != static_cast<std::size_t>(n_nodes) ||
      z.size() != static_cast<std::size_t>(n_nodes) ||
      r_reference.size() != static_cast<std::size_t>(n_nodes) ||
      z_reference.size() != static_cast<std::size_t>(n_nodes)) {
    return metrics;
  }
  const auto& mb = *mesh.topo.multiblock;
  if (mb.cell_node_csr_offsets.size() != static_cast<std::size_t>(mesh.topo.n_cells + 1) ||
      mb.cell_node_csr_indices.size() != static_cast<std::size_t>(mesh.topo.n_cells) * 4U) {
    return metrics;
  }
  const int active_nverts = cell_active_nverts(mesh, cell);
  if (active_nverts == 3) {
    return metrics;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  std::array<double, 4> cr{};
  std::array<double, 4> cz{};
  std::array<double, 4> wr{};
  std::array<double, 4> wz{};
  for (int k = 0; k < 4; ++k) {
    const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    if (n < 0 || n >= n_nodes) {
      return metrics;
    }
    metrics.nodes[static_cast<std::size_t>(k)] = n;
    cr[static_cast<std::size_t>(k)] = r[static_cast<std::size_t>(n)];
    cz[static_cast<std::size_t>(k)] = z[static_cast<std::size_t>(n)];
    wr[static_cast<std::size_t>(k)] =
        scale * r_reference[static_cast<std::size_t>(n)];
    wz[static_cast<std::size_t>(k)] =
        scale * z_reference[static_cast<std::size_t>(n)];
  }

  const double j2 = diagonal_corner_j2_from_quad(cr, cz);
  const double j2_target = diagonal_corner_j2_from_quad(wr, wz);
  const double b_eff =
      0.5 * (std::hypot(cr[1] - cr[0], cz[1] - cz[0]) +
             std::hypot(cr[3] - cr[0], cz[3] - cz[0]));
  const double b_target =
      0.5 * (std::hypot(wr[1] - wr[0], wz[1] - wz[0]) +
             std::hypot(wr[3] - wr[0], wz[3] - wz[0]));
  const double psi_eff = b_eff > 0.0 ? j2 / b_eff
                                     : std::numeric_limits<double>::quiet_NaN();
  const double psi_target =
      b_target > 0.0 ? j2_target / b_target
                     : std::numeric_limits<double>::quiet_NaN();
  const double e1r = cr[3] - cr[2];
  const double e1z = cz[3] - cz[2];
  const double e2r = cr[1] - cr[2];
  const double e2z = cz[1] - cz[2];
  const double w1r = wr[3] - wr[2];
  const double w1z = wz[3] - wz[2];
  const double w2r = wr[1] - wr[2];
  const double w2z = wz[1] - wz[2];
  metrics.j2 = j2;
  metrics.b_eff = b_eff;
  metrics.psi_eff = psi_eff;
  metrics.psi_target = psi_target;
  metrics.p_psi = psi_target != 0.0 ? psi_eff / psi_target
                                    : std::numeric_limits<double>::quiet_NaN();
  metrics.sigma2 = sigma2_from_columns(e1r, e1z, e2r, e2z);
  metrics.sigma2_target = sigma2_from_columns(w1r, w1z, w2r, w2z);
  metrics.s2 =
      metrics.sigma2_target > 0.0 ? metrics.sigma2 / metrics.sigma2_target
                                  : std::numeric_limits<double>::quiet_NaN();
  const double denom = e1r * e1r + e1z * e1z + e2r * e2r + e2z * e2z;
  metrics.q2 = denom > 0.0 ? 2.0 * j2 / denom
                           : std::numeric_limits<double>::quiet_NaN();
  metrics.min_altitude = min_altitude_from_quad(cr, cz);
  const std::array<double, 4> ref_r{{
      r_reference[static_cast<std::size_t>(metrics.nodes[0])],
      r_reference[static_cast<std::size_t>(metrics.nodes[1])],
      r_reference[static_cast<std::size_t>(metrics.nodes[2])],
      r_reference[static_cast<std::size_t>(metrics.nodes[3])],
  }};
  const std::array<double, 4> ref_z{{
      z_reference[static_cast<std::size_t>(metrics.nodes[0])],
      z_reference[static_cast<std::size_t>(metrics.nodes[1])],
      z_reference[static_cast<std::size_t>(metrics.nodes[2])],
      z_reference[static_cast<std::size_t>(metrics.nodes[3])],
  }};
  metrics.min_altitude_reference = min_altitude_from_quad(ref_r, ref_z);
  metrics.h2 = metrics.min_altitude_reference > 0.0
                   ? metrics.min_altitude /
                         (scale * metrics.min_altitude_reference)
                   : std::numeric_limits<double>::quiet_NaN();
  metrics.score_min =
      score_min4(metrics.p_psi, metrics.s2, metrics.q2, metrics.h2);
  metrics.sampled = std::isfinite(metrics.j2) &&
                    std::isfinite(metrics.p_psi) &&
                    std::isfinite(metrics.s2) &&
                    std::isfinite(metrics.q2) &&
                    std::isfinite(metrics.h2);
  return metrics;
}

AxisAleDiagonalCornerMetrics compute_axis_ale_worst_diagonal_corner_metrics(
    const mesh::Mesh& mesh,
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<double>& r_reference,
    const std::vector<double>& z_reference,
    const double scale) {
  AxisAleDiagonalCornerMetrics worst;
  if (!mesh.topo.multiblock.has_value()) {
    return worst;
  }
  const auto& mb = *mesh.topo.multiblock;
  for (int c = 0; c < mesh.topo.n_cells; ++c) {
    if (!is_central_core_cell(mb, c)) {
      continue;
    }
    const AxisAleDiagonalCornerMetrics m =
        compute_axis_ale_diagonal_corner_metrics(
            mesh, r, z, r_reference, z_reference, scale, c);
    if (!m.sampled) {
      continue;
    }
    if (!worst.sampled ||
        m.score_min < worst.score_min ||
        (m.score_min == worst.score_min && c < worst.cell)) {
      worst = m;
    }
  }
  return worst;
}

AxisAleCellTargetAudit audit_axis_ale_patch_target_cell(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input,
    const AxisAleRezoneResult& axis_target,
    const AxisAlePatchTarget& patch_target,
    const int cell) {
  (void)input;
  AxisAleCellTargetAudit audit;
  audit.cell = cell;
  if (!axis_target.active || !mesh.topo.multiblock.has_value()) {
    return audit;
  }
  const auto& mb = *mesh.topo.multiblock;
  const int n_nodes = mesh.topo.n_nodes;
  const int n_cells = mesh.topo.n_cells;
  if (cell < 0 || cell >= n_cells) {
    return audit;
  }
  TENRYU_ASSERT(axis_target.node_ids.size() == axis_target.z_target.size(),
                "axis ALE patch cell audit requires paired axis arrays");
  TENRYU_ASSERT(patch_target.node_ids.size() == patch_target.r_target.size() &&
                    patch_target.node_ids.size() ==
                        patch_target.z_target.size(),
                "axis ALE patch cell audit requires paired patch arrays");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "axis ALE patch cell audit requires cell-node CSR offsets");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) * 4U,
                "axis ALE patch cell audit requires cell-node CSR indices");

  const std::vector<double> r =
      copy_node_field(mesh.node_r, n_nodes, "axis ALE patch cell audit memcpy node_r failed");
  const std::vector<double> z =
      copy_node_field(mesh.node_z, n_nodes, "axis ALE patch cell audit memcpy node_z failed");
  std::vector<double> axis_r = r;
  std::vector<double> axis_z = z;
  std::vector<double> patch_r = r;
  std::vector<double> patch_z = z;
  for (std::size_t i = 0; i < axis_target.node_ids.size(); ++i) {
    const int n = axis_target.node_ids[i];
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "axis ALE patch cell audit axis node id out of range");
    const double z_target =
        node_is_pinned_cap_origin(mesh, n) ? 0.0 : axis_target.z_target[i];
    axis_r[static_cast<std::size_t>(n)] = 0.0;
    axis_z[static_cast<std::size_t>(n)] = z_target;
    patch_r[static_cast<std::size_t>(n)] = 0.0;
    patch_z[static_cast<std::size_t>(n)] = z_target;
  }
  for (std::size_t i = 0; i < patch_target.node_ids.size(); ++i) {
    const int n = patch_target.node_ids[i];
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "axis ALE patch cell audit patch node id out of range");
    patch_r[static_cast<std::size_t>(n)] = patch_target.r_target[i];
    patch_z[static_cast<std::size_t>(n)] = patch_target.z_target[i];
  }

  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int active_nverts = cell_active_nverts(mesh, cell);
  std::array<double, 4> axis_cell_r{};
  std::array<double, 4> axis_cell_z{};
  std::array<double, 4> patch_cell_r{};
  std::array<double, 4> patch_cell_z{};
  for (int k = 0; k < active_nverts; ++k) {
    const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "axis ALE patch cell audit cell node out of range");
    audit.nodes[static_cast<std::size_t>(k)] = n;
    axis_cell_r[static_cast<std::size_t>(k)] = axis_r[static_cast<std::size_t>(n)];
    axis_cell_z[static_cast<std::size_t>(k)] = axis_z[static_cast<std::size_t>(n)];
    patch_cell_r[static_cast<std::size_t>(k)] = patch_r[static_cast<std::size_t>(n)];
    patch_cell_z[static_cast<std::size_t>(k)] = patch_z[static_cast<std::size_t>(n)];
  }
  int orientation_sign = 1;
  if (mb.cell_orientation_sign.size() == static_cast<std::size_t>(n_cells)) {
    orientation_sign = mb.cell_orientation_sign[static_cast<std::size_t>(cell)];
  }
  audit.axis_only =
      quality_from_active_cell(
          axis_cell_r, axis_cell_z, active_nverts, orientation_sign);
  audit.patch =
      quality_from_active_cell(
          patch_cell_r, patch_cell_z, active_nverts, orientation_sign);
  audit.sampled = audit.axis_only.sampled && audit.patch.sampled;
  if (audit.sampled) {
    audit.shear_threshold = kCellTargetShearAuditThreshold;
    audit.meaningful_shear =
        audit.axis_only.shear > audit.shear_threshold;
    audit.shear_tolerance =
        1.0e-12 * std::max(1.0, std::abs(audit.axis_only.shear));
    audit.shear_reduced =
        !audit.meaningful_shear ||
        audit.patch.shear + audit.shear_tolerance < audit.axis_only.shear;
    audit.corner_j_tolerance =
        1.0e-12 * std::max(1.0, std::abs(audit.axis_only.min_corner_j));
    // Corner validity: the candidate must not FOLD a corner (min corner J
    // may not drop below min(axis_only, 0)). Rounded/D-shape seam cells
    // carry a collinear corner with corner_j == 0 by construction, so an
    // absolute > 0 test fails a no-op and a strictly-relative test fails a
    // legitimate repair that restores the designed flat corner after a
    // perturbation convexified it. A candidate with a zero-length edge
    // (coincident vertices, aspect = inf) is always rejected -- that is
    // the collapse signature the corner test cannot see when the corner
    // is already flat.
    audit.corner_j_ok =
        std::isfinite(audit.patch.aspect_ratio) &&
        audit.patch.min_corner_j + audit.corner_j_tolerance >=
            std::min(audit.axis_only.min_corner_j, 0.0);
    audit.passed = audit.shear_reduced && audit.corner_j_ok;
  }
  return audit;
}

void project_polar_shell_axis_radial_order_target(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input,
    AxisAleRezoneResult& target) {
  if (!target.active || target.node_ids != input.node_ids ||
      target.z_target.size() != input.node_ids.size()) {
    return;
  }
  enforce_polar_shell_axis_radial_order(mesh, input, target.z_target);
}

AxisAleRezoneResult compute_axis_ale_rezone_target(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input) {
  AxisAleRezoneResult result;
  if (input.node_ids.empty()) {
    return result;
  }
  TENRYU_ASSERT(input.node_ids.size() == input.z_tilde.size() &&
                    input.node_ids.size() == input.lumped_mass.size(),
                "axis ALE rezone target requires paired node/z/mass arrays");
  TENRYU_ASSERT(input.initial_edge_length.size() + 1U == input.node_ids.size() &&
                    input.adjacent_cell_area.size() + 1U == input.node_ids.size(),
                "axis ALE rezone target requires one edge metric per chain edge");

  std::vector<std::size_t> active_index;
  std::vector<std::size_t> fixed_origin_index;
  active_index.reserve(input.node_ids.size());
  for (std::size_t i = 0; i < input.node_ids.size(); ++i) {
    const int node = input.node_ids[i];
    const double z = input.z_tilde[i];
    const double m = input.lumped_mass[i];
    // Upper-bound validation only applies against a real mesh; a
    // default-constructed mesh (n_nodes == 0) marks the detached synthetic
    // projection mode used by the unit tests of the pure PAVA target.
    TENRYU_ASSERT(node >= 0 &&
                      (mesh.topo.n_nodes <= 0 || node < mesh.topo.n_nodes),
                  "axis ALE rezone target node id out of range");
    TENRYU_ASSERT(std::isfinite(z),
                  "axis ALE rezone target predicted z must be finite");
    TENRYU_ASSERT(std::isfinite(m) && m >= 0.0,
                  "axis ALE rezone target node mass must be finite non-negative");
    if (node_is_pinned_cap_origin(mesh, node)) {
      fixed_origin_index.push_back(i);
      continue;
    }
    if (m > 0.0) {
      active_index.push_back(i);
    }
  }
  if (active_index.empty()) {
    if (!fixed_origin_index.empty()) {
      result.active = true;
      result.node_ids = input.node_ids;
      result.z_target = input.z_tilde;
      for (const std::size_t i : fixed_origin_index) {
        result.z_target[i] = 0.0;
      }
      enforce_polar_shell_axis_radial_order(mesh, input, result.z_target);
      result.first_off_axis_ring =
          compute_first_off_axis_ring_diagnostics(mesh, input.node_ids);
    }
    return result;
  }

  std::vector<double> active_z_tilde;
  std::vector<double> active_lumped_mass;
  active_z_tilde.reserve(active_index.size());
  active_lumped_mass.reserve(active_index.size());
  for (const std::size_t i : active_index) {
    active_z_tilde.push_back(input.z_tilde[i]);
    active_lumped_mass.push_back(input.lumped_mass[i]);
  }

  std::vector<double> active_initial_edge_length;
  std::vector<double> active_adjacent_cell_area;
  if (active_index.size() >= 2U) {
    active_initial_edge_length.reserve(active_index.size() - 1U);
    active_adjacent_cell_area.reserve(active_index.size() - 1U);
  }
  for (std::size_t e = 0; e + 1U < active_index.size(); ++e) {
    const std::size_t begin = active_index[e];
    const std::size_t end = active_index[e + 1U];
    TENRYU_ASSERT(end > begin,
                  "axis ALE active chain indices must be strictly increasing");
    double length = 0.0;
    for (std::size_t p = begin; p < end; ++p) {
      const double l0 = input.initial_edge_length[p];
      TENRYU_ASSERT(std::isfinite(l0) && l0 > 0.0,
                    "axis ALE active chain initial edge length must be positive");
      length += l0;
    }
    active_initial_edge_length.push_back(length);

    double area = (end == begin + 1U)
                      ? input.adjacent_cell_area[begin]
                      : length * length;
    if (!(std::isfinite(area) && area > 0.0)) {
      area = length * length;
    }
    active_adjacent_cell_area.push_back(area);
  }

  result.active = true;
  result.node_ids = input.node_ids;
  result.z_target = input.z_tilde;
  for (const std::size_t i : fixed_origin_index) {
    result.z_target[i] = 0.0;
  }
  const std::vector<double> active_z_target =
      project_weighted_lower_bound_pava(active_z_tilde,
                                        active_lumped_mass,
                                        active_initial_edge_length,
                                        active_adjacent_cell_area,
                                        input.eta_floor,
                                        input.dt);
  TENRYU_ASSERT(active_z_target.size() == active_index.size(),
                "axis ALE active PAVA output size mismatch");
  for (std::size_t i = 0; i < active_index.size(); ++i) {
    result.z_target[active_index[i]] = active_z_target[i];
  }
  for (const std::size_t i : fixed_origin_index) {
    result.z_target[i] = 0.0;
  }
  enforce_polar_shell_axis_radial_order(mesh, input, result.z_target);
  result.first_off_axis_ring =
      compute_first_off_axis_ring_diagnostics(mesh, input.node_ids);
  return result;
}

}  // namespace tenryu::hydro::axis_ale
