#include "mesh/mesh.hpp"
#include "mesh/geometry_1d.cuh"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/config.hpp"
#include "core/device_pack.hpp"
#include "core/error.hpp"
#include "core/namelist/errors.hpp"
#include "core/axis_tolerance.hpp"
#include "core/state.hpp"
#include "hydro/hydro_multiblock_topology.cuh"
#include "mesh/spherical_wedge_geometry.cuh"

namespace tenryu::mesh {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline bool finite_double(const double value) {
  return std::isfinite(value);
}

constexpr double kPolarButtonOrientationSign = -1.0;

__host__ __device__ double rz_polygon_volume_exact(const double* r,
                                                   const double* z,
                                                   int nverts);
__host__ __device__ double rz_polygon_area2(const double* r,
                                            const double* z,
                                            int nverts);
__host__ __device__ void rz_polygon_area_centroid(const double* r,
                                                  const double* z,
                                                  int nverts,
                                                  double* centroid_r,
                                                  double* centroid_z);

void assert_geometry_failure(const MeshGeometryResult& result) {
  const std::string message = format_legacy_failure_message(result);
  core::log_warning(message);
  TENRYU_ASSERT(false, message);
}

// Option-C decomposition (design doc mpi_m18_20_20260717.md §3): geometry in
// non-owned non-ghost regions is stale by construction; validity checks must
// only inspect the owned(+ghost) window or they FATAL on another rank's
// in-flight motion. Set once by the driver when n_ranks > 1; -1 = full mesh.
int g_geometry_check_begin = 0;
int g_geometry_check_end = -1;
inline bool cell_in_geometry_check_window(const int c) {
  return g_geometry_check_end < 0 ||
         (c >= g_geometry_check_begin && c < g_geometry_check_end);
}

MeshGeometryResult make_geometry_failure(
    const MeshGeometryFailureKind kind,
    const int failing_cell,
    const int failing_i,
    const int failing_j,
    const double failing_value,
    const double min_cell_vol,
    const double min_cell_area,
    const char* reason) {
  MeshGeometryResult result{};
  result.admissible = false;
  result.recoverable = true;
  result.kind = kind;
  result.failing_cell = failing_cell;
  result.failing_i = failing_i;
  result.failing_j = failing_j;
  result.failing_value = failing_value;
  result.min_cell_vol = min_cell_vol;
  result.min_cell_area = min_cell_area;
  result.reason = reason;
  return result;
}

bool has_nonfinite_1d_node_coordinate(const Mesh& mesh, const int c) {
  std::array<double, 2> r{};
  cuda_check(cudaMemcpy(r.data(), mesh.node_r + c, r.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Mesh geometry check memcpy 1D node_r failed");
  return !finite_double(r[0]) || !finite_double(r[1]);
}

bool has_nonfinite_2d_node_coordinate(const Mesh& mesh, const int c) {
  const int i = c / mesh.topo.nz;
  const int j = c - i * mesh.topo.nz;
  const int stride = mesh.topo.nz + 1;
  const int nodes[4] = {
      i * stride + j,
      (i + 1) * stride + j,
      (i + 1) * stride + (j + 1),
      i * stride + (j + 1),
  };
  for (const int n : nodes) {
    double r = 0.0;
    double z = 0.0;
    cuda_check(cudaMemcpy(&r, mesh.node_r + n, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh geometry check memcpy 2D node_r failed");
    cuda_check(cudaMemcpy(&z, mesh.node_z + n, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh geometry check memcpy 2D node_z failed");
    if (!finite_double(r) || !finite_double(z)) {
      return true;
    }
  }
  return false;
}

void load_button_polygon_nodes(const Mesh& mesh,
                               std::vector<double>& r,
                               std::vector<double>& z) {
  TENRYU_ASSERT(mesh.button_center && mesh.button_center->enabled,
                "button polygon node load requires button topology");
  TENRYU_ASSERT(mesh.node_r != nullptr && mesh.node_z != nullptr,
                "button polygon node load requires 2D node coordinates");
  const int outer_ring = mesh.button_center->outer_node_ring;
  TENRYU_ASSERT(outer_ring >= 1 && outer_ring <= mesh.topo.nr,
                "button outer node ring out of range");
  const int nverts = mesh.topo.nz + 1;
  TENRYU_ASSERT(nverts >= 2, "button polygon requires at least two nodes");
  r.assign(static_cast<std::size_t>(nverts), 0.0);
  z.assign(static_cast<std::size_t>(nverts), 0.0);
  for (int j = 0; j <= mesh.topo.nz; ++j) {
    const int n = mesh.topo.node_index(outer_ring, j);
    cuda_check(cudaMemcpy(&r[static_cast<std::size_t>(j)],
                          mesh.node_r + n,
                          sizeof(double),
                          cudaMemcpyDeviceToHost),
               "button polygon memcpy node_r failed");
    cuda_check(cudaMemcpy(&z[static_cast<std::size_t>(j)],
                          mesh.node_z + n,
                          sizeof(double),
                          cudaMemcpyDeviceToHost),
               "button polygon memcpy node_z failed");
  }
}

bool has_nonfinite_button_node_coordinate(const Mesh& mesh) {
  std::vector<double> r;
  std::vector<double> z;
  load_button_polygon_nodes(mesh, r, z);
  for (std::size_t k = 0; k < r.size(); ++k) {
    if (!finite_double(r[k]) || !finite_double(z[k])) {
      return true;
    }
  }
  return false;
}

void load_all_node_coordinates(const Mesh& mesh,
                               std::vector<double>& r,
                               std::vector<double>& z) {
  TENRYU_ASSERT(mesh.node_r != nullptr && mesh.node_z != nullptr,
                "2D node coordinate load requires node coordinates");
  const std::size_t n_nodes = static_cast<std::size_t>(mesh.topo.n_nodes);
  r.assign(n_nodes, 0.0);
  z.assign(n_nodes, 0.0);
  if (n_nodes == 0U) {
    return;
  }
  cuda_check(cudaMemcpy(r.data(), mesh.node_r, n_nodes * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "node coordinate memcpy node_r failed");
  cuda_check(cudaMemcpy(z.data(), mesh.node_z, n_nodes * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "node coordinate memcpy node_z failed");
}

bool button_polygon_corner_areas_positive(const Mesh& mesh,
                                          const double area_floor,
                                          int* bad_corner,
                                          double* min_area) {
  std::vector<double> r;
  std::vector<double> z;
  load_button_polygon_nodes(mesh, r, z);
  const int nverts = static_cast<int>(r.size());
  for (int k = 0; k < nverts; ++k) {
    if (!finite_double(r[static_cast<std::size_t>(k)]) ||
        !finite_double(z[static_cast<std::size_t>(k)])) {
      if (bad_corner != nullptr) {
        *bad_corner = k;
      }
      if (min_area != nullptr) {
        *min_area = -std::numeric_limits<double>::infinity();
      }
      return false;
    }
  }

  double centroid_r = 0.0;
  double centroid_z = 0.0;
  rz_polygon_area_centroid(r.data(), z.data(), nverts,
                           &centroid_r, &centroid_z);
  if (!finite_double(centroid_r) || !finite_double(centroid_z)) {
    if (bad_corner != nullptr) {
      *bad_corner = 0;
    }
    if (min_area != nullptr) {
      *min_area = -std::numeric_limits<double>::infinity();
    }
    return false;
  }

  double min_value = std::numeric_limits<double>::infinity();
  bool ok = true;
  int first_bad = -1;
  for (int k = 0; k < nverts; ++k) {
    const int kp1 = (k + 1) % nverts;
    const double rk = r[static_cast<std::size_t>(k)] - centroid_r;
    const double zk = z[static_cast<std::size_t>(k)] - centroid_z;
    const double rkp1 = r[static_cast<std::size_t>(kp1)] - centroid_r;
    const double zkp1 = z[static_cast<std::size_t>(kp1)] - centroid_z;
    const double area =
        0.5 * kPolarButtonOrientationSign * (rk * zkp1 - rkp1 * zk);
    if (finite_double(area) && area < min_value) {
      min_value = area;
    }
    if (!(area > area_floor)) {
      ok = false;
      if (first_bad < 0) {
        first_bad = k;
      }
    }
  }
  if (bad_corner != nullptr) {
    *bad_corner = first_bad;
  }
  if (min_area != nullptr) {
    *min_area = std::isfinite(min_value) ? min_value : 0.0;
  }
  return ok;
}

bool has_nonfinite_multiblock_node_coordinate(const Mesh& mesh, const int c) {
  TENRYU_ASSERT(mesh.topo.multiblock.has_value(),
                "multiblock node check requires multiblock topology");
  const auto& mb = *mesh.topo.multiblock;
  TENRYU_ASSERT(static_cast<std::size_t>(c) + 1U <
                    mb.cell_node_csr_offsets.size(),
                "multiblock cell-node CSR offset missing");
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
  const int next = mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
  TENRYU_ASSERT(next - off == mesh.corner_stride,
                "multiblock cell-node CSR width must match mesh stride");
  const int active_nverts =
      mesh.cell_nverts.size() == mesh.cell_vol.size()
          ? mesh_topo_cell_active_nverts(mesh.cell_nverts, c)
          : kMeshTopoCellStorageSlots;
  for (int k = 0; k < active_nverts; ++k) {
    TENRYU_ASSERT(static_cast<std::size_t>(off + k) <
                      mb.cell_node_csr_indices.size(),
                  "multiblock cell-node CSR index missing");
    const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    TENRYU_ASSERT(n >= 0 && n < mesh.topo.n_nodes,
                  "multiblock cell node index out of range");
    double r = 0.0;
    double z = 0.0;
    cuda_check(cudaMemcpy(&r, mesh.node_r + n, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh geometry check memcpy multiblock node_r failed");
    cuda_check(cudaMemcpy(&z, mesh.node_z + n, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh geometry check memcpy multiblock node_z failed");
    if (!finite_double(r) || !finite_double(z)) {
      return true;
    }
  }
  return false;
}

MeshGeometryResult validate_1d_geometry(
    const Mesh& mesh,
    const MeshGeometryCheckOptions& opts) {
  MeshGeometryResult result{};
  double min_cell_vol = 0.0;
  bool min_initialized = false;
  const std::size_t n_cells = mesh.cell_vol.size();
  for (std::size_t c_u = 0; c_u < n_cells; ++c_u) {
    if (!cell_in_geometry_check_window(static_cast<int>(c_u))) {
      continue;
    }
    const double vol = mesh.cell_vol[c_u];
    if (finite_double(vol) && (!min_initialized || vol < min_cell_vol)) {
      min_cell_vol = vol;
      min_initialized = true;
    }
    if (!(vol > opts.volume_floor)) {
      const int c = static_cast<int>(c_u);
      const MeshGeometryFailureKind kind =
          has_nonfinite_1d_node_coordinate(mesh, c)
              ? MeshGeometryFailureKind::NonFiniteNodeCoordinate
              : (!finite_double(vol)
                     ? MeshGeometryFailureKind::NonFiniteCellVolume
                     : MeshGeometryFailureKind::NonPositiveCellVolume);
      result = make_geometry_failure(kind, c, c, -1, vol, min_cell_vol, 0.0,
                                     "cell_volume");
      if (opts.policy == MeshGeometryFailurePolicy::HardAssert) {
        assert_geometry_failure(result);
      }
      return result;
    }
  }
  result.min_cell_vol = min_initialized ? min_cell_vol : 0.0;
  return result;
}

MeshGeometryResult validate_2d_geometry(
    const Mesh& mesh,
    const MeshGeometryCheckOptions& opts) {
  MeshGeometryResult result{};
  double min_cell_vol = 0.0;
  double min_cell_area = 0.0;
  bool min_vol_initialized = false;
  bool min_area_initialized = false;
  const std::size_t n_cells = mesh.cell_vol.size();
  for (std::size_t c_u = 0; c_u < n_cells; ++c_u) {
    const int c = static_cast<int>(c_u);
    const int i = c / mesh.topo.nz;
    const int j = c - i * mesh.topo.nz;
    if (mesh.is_dormant_cell(c) || mesh.is_geometry_exempt_cell(c)) {
      continue;
    }
    if (opts.inactive_cell_mask != nullptr &&
        opts.inactive_cell_mask[c_u] != 0U) {
      continue;
    }
    if (!cell_in_geometry_check_window(c)) {
      continue;
    }
    const double vol = mesh.cell_vol[c_u];
    if (finite_double(vol) && (!min_vol_initialized || vol < min_cell_vol)) {
      min_cell_vol = vol;
      min_vol_initialized = true;
    }
    if (!(vol > opts.volume_floor)) {
      const MeshGeometryFailureKind kind =
          mesh.is_button_cell(c)
              ? (has_nonfinite_button_node_coordinate(mesh)
                     ? MeshGeometryFailureKind::NonFiniteNodeCoordinate
                     : (!finite_double(vol)
                            ? MeshGeometryFailureKind::NonFiniteCellVolume
                            : MeshGeometryFailureKind::NonPositiveCellVolume))
              : has_nonfinite_2d_node_coordinate(mesh, c)
              ? MeshGeometryFailureKind::NonFiniteNodeCoordinate
              : (!finite_double(vol)
                     ? MeshGeometryFailureKind::NonFiniteCellVolume
                     : MeshGeometryFailureKind::NonPositiveCellVolume);
      result = make_geometry_failure(kind, c, i, j, vol, min_cell_vol,
                                     min_area_initialized ? min_cell_area : 0.0,
                                     "cell_volume");
      if (opts.policy == MeshGeometryFailurePolicy::HardAssert) {
        assert_geometry_failure(result);
      }
      return result;
    }
    const double area = mesh.cell_area[c_u];
    if (finite_double(area) && (!min_area_initialized || area < min_cell_area)) {
      min_cell_area = area;
      min_area_initialized = true;
    }
    if (!(area > opts.area_floor)) {
      const MeshGeometryFailureKind kind =
          mesh.is_button_cell(c)
              ? (has_nonfinite_button_node_coordinate(mesh)
                     ? MeshGeometryFailureKind::NonFiniteNodeCoordinate
                     : (!finite_double(area)
                            ? MeshGeometryFailureKind::NonFiniteCellArea
                            : MeshGeometryFailureKind::NonPositiveCellArea))
              : has_nonfinite_2d_node_coordinate(mesh, c)
              ? MeshGeometryFailureKind::NonFiniteNodeCoordinate
              : (!finite_double(area)
                     ? MeshGeometryFailureKind::NonFiniteCellArea
                     : MeshGeometryFailureKind::NonPositiveCellArea);
      result = make_geometry_failure(kind, c, i, j, area,
                                     min_vol_initialized ? min_cell_vol : 0.0,
                                     min_cell_area, "cell_area");
      if (opts.policy == MeshGeometryFailurePolicy::HardAssert) {
        assert_geometry_failure(result);
      }
      return result;
    }
    if (mesh.is_button_cell(c)) {
      int bad_corner = -1;
      double min_button_area = 0.0;
      if (!button_polygon_corner_areas_positive(
              mesh, opts.area_floor, &bad_corner, &min_button_area)) {
        const MeshGeometryFailureKind kind =
            has_nonfinite_button_node_coordinate(mesh)
                ? MeshGeometryFailureKind::NonFiniteNodeCoordinate
                : MeshGeometryFailureKind::NonPositiveCellArea;
        result = make_geometry_failure(
            kind, c, i, j, min_button_area,
            min_vol_initialized ? min_cell_vol : 0.0,
            min_area_initialized ? std::min(min_cell_area, min_button_area)
                                 : min_button_area,
            "button_corner_area");
        result.failing_j = bad_corner;
        if (opts.policy == MeshGeometryFailurePolicy::HardAssert) {
          assert_geometry_failure(result);
        }
        return result;
      }
    }
  }
  result.min_cell_vol = min_vol_initialized ? min_cell_vol : 0.0;
  result.min_cell_area = min_area_initialized ? min_cell_area : 0.0;
  return result;
}

MeshGeometryResult validate_multiblock_geometry(
    const Mesh& mesh,
    const MeshGeometryCheckOptions& opts) {
  MeshGeometryResult result{};
  double min_cell_vol = 0.0;
  double min_cell_area = 0.0;
  bool min_vol_initialized = false;
  bool min_area_initialized = false;
  const std::size_t n_cells = mesh.cell_vol.size();
  for (std::size_t c_u = 0; c_u < n_cells; ++c_u) {
    const int c = static_cast<int>(c_u);
    const int i = (mesh.topo.nz > 0) ? (c / mesh.topo.nz) : c;
    const int j = (mesh.topo.nz > 0) ? (c - i * mesh.topo.nz) : -1;
    if (mesh.is_geometry_exempt_cell(c)) {
      continue;
    }
    if (opts.inactive_cell_mask != nullptr &&
        opts.inactive_cell_mask[c_u] != 0U) {
      continue;
    }
    if (!cell_in_geometry_check_window(c)) {
      continue;
    }
    const double vol = mesh.cell_vol[c_u];
    if (finite_double(vol) && (!min_vol_initialized || vol < min_cell_vol)) {
      min_cell_vol = vol;
      min_vol_initialized = true;
    }
    if (!(vol > opts.volume_floor)) {
      const MeshGeometryFailureKind kind =
          has_nonfinite_multiblock_node_coordinate(mesh, c)
              ? MeshGeometryFailureKind::NonFiniteNodeCoordinate
              : (!finite_double(vol)
                     ? MeshGeometryFailureKind::NonFiniteCellVolume
                     : MeshGeometryFailureKind::NonPositiveCellVolume);
      result = make_geometry_failure(kind, c, i, j, vol, min_cell_vol,
                                     min_area_initialized ? min_cell_area : 0.0,
                                     "cell_volume");
      if (opts.policy == MeshGeometryFailurePolicy::HardAssert) {
        assert_geometry_failure(result);
      }
      return result;
    }
    const double area = mesh.cell_area[c_u];
    if (finite_double(area) && (!min_area_initialized || area < min_cell_area)) {
      min_cell_area = area;
      min_area_initialized = true;
    }
    if (!(area > opts.area_floor)) {
      const MeshGeometryFailureKind kind =
          has_nonfinite_multiblock_node_coordinate(mesh, c)
              ? MeshGeometryFailureKind::NonFiniteNodeCoordinate
              : (!finite_double(area)
                     ? MeshGeometryFailureKind::NonFiniteCellArea
                     : MeshGeometryFailureKind::NonPositiveCellArea);
      result = make_geometry_failure(kind, c, i, j, area,
                                     min_vol_initialized ? min_cell_vol : 0.0,
                                     min_cell_area, "cell_area");
      if (opts.policy == MeshGeometryFailurePolicy::HardAssert) {
        assert_geometry_failure(result);
      }
      return result;
    }
  }
  result.min_cell_vol = min_vol_initialized ? min_cell_vol : 0.0;
  result.min_cell_area = min_area_initialized ? min_cell_area : 0.0;
  return result;
}

std::vector<double> build_uniform_nodes(const double x_min,
                                        const double x_max,
                                        const int n_cells) {
  TENRYU_ASSERT(n_cells > 0, "Number of cells must be > 0");
  std::vector<double> nodes(static_cast<std::size_t>(n_cells + 1), 0.0);
  const double dx = (x_max - x_min) / static_cast<double>(n_cells);
  for (int i = 0; i <= n_cells; ++i) {
    nodes[static_cast<std::size_t>(i)] = x_min + dx * static_cast<double>(i);
  }
  return nodes;
}

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
  TENRYU_ASSERT(!widths.empty(), "graded grid segment width array must be non-empty");
  const double tol = 1.0e-12 * std::max(1.0, std::abs(length));
  const int n = static_cast<int>(widths.size());

  if (!has_left_target && !has_right_target) {
    return;
  }

  if (n == 1) {
    const double target = has_left_target ? left_target : right_target;
    if (has_left_target && has_right_target) {
      TENRYU_ASSERT(std::abs(left_target - right_target) <= tol,
                    "graded grid cannot satisfy both boundary width targets with nr=1 "
                    "for segment " + std::to_string(segment_index));
    }
    TENRYU_ASSERT(std::abs(length - target) <= tol,
                  "graded grid cannot satisfy boundary width target with nr=1 "
                  "for segment " + std::to_string(segment_index));
    widths[0] = length;
    return;
  }

  if (has_left_target && has_right_target) {
    if (n == 2) {
      TENRYU_ASSERT(std::abs((left_target + right_target) - length) <= tol,
                    "graded grid cannot satisfy both boundary width targets with nr=2 "
                    "for segment " + std::to_string(segment_index));
      widths[0] = left_target;
      widths[1] = length - left_target;
      TENRYU_ASSERT(std::abs(widths[1] - right_target) <= tol,
                    "graded grid nr=2 boundary correction drift exceeded tolerance");
      return;
    }

    const double interior_old = length - widths.front() - widths.back();
    const double interior_new = length - left_target - right_target;
    TENRYU_ASSERT(interior_old > 0.0,
                  "graded grid interior width sum must be > 0 before correction");
    TENRYU_ASSERT(interior_new > 0.0,
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
    TENRYU_ASSERT(tail_old > 0.0,
                  "graded grid non-boundary width sum must be > 0 before left correction");
    TENRYU_ASSERT(tail_new > 0.0,
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
  TENRYU_ASSERT(head_old > 0.0,
                "graded grid non-boundary width sum must be > 0 before right correction");
  TENRYU_ASSERT(head_new > 0.0,
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
  const double xi = (static_cast<double>(k) + 0.5) / static_cast<double>(nr);
  const double u = std::abs(2.0 * xi - 1.0);
  const double exponent =
      std::pow(u / grading.sg_sigma, static_cast<double>(grading.sg_order));
  return grading.edge_ratio + (1.0 - grading.edge_ratio) * std::exp(-exponent);
}

// measure_dim: 0 => always use the legacy estimated-radius mapping;
// 1/2/3 (planar/cylindrical/spherical geometric measure) enables the opt-in
// grading.mapping == "exact_measure_v2" inversion (2026-07-26 review):
// node positions come from the cumulative weight fraction
// W_k = cumsum(w)/sum(w) mapped through the exact shell measure
// r_k^d = r_in^d + W_k (r_out^d - r_in^d), so constant-density cell masses
// are proportional to w_k to roundoff — including origin segments, where the
// legacy one-shot 1/r_est^2 estimate (with its L/sqrt(N) regularizer) only
// approximates the target distribution.
std::vector<double> build_graded_nodes(const std::vector<GridSegment>& segments,
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
      // exact_measure_v2: cell mass proportional to w_k (constant density),
      // valid for every edge_ratio including the ~1 equal-mass limit.
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
        const double weight = graded_super_gaussian_weight(grading, k, seg.nr);
        TENRYU_ASSERT(std::isfinite(weight) && weight > 0.0,
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
        const double W_k = w_cum[static_cast<std::size_t>(k)] / w_sum;
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
      const double xi = (static_cast<double>(k) + 0.5) / static_cast<double>(seg.nr);
      const double weight = graded_super_gaussian_weight(grading, k, seg.nr);
      const double r_est = seg.r_start + length * xi;
      const double r_ref = (seg.r_start < 1.0e-12)
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
    graded_grid_apply_boundary_targets(segment_widths[s],
                                       length,
                                       has_first_target[s],
                                       first_targets[s],
                                       has_last_target[s],
                                       last_targets[s],
                                       s);

    double width_sum = 0.0;
    for (const double width : segment_widths[s]) {
      TENRYU_ASSERT(std::isfinite(width) && width > 0.0,
                    "graded grid produced non-positive cell width");
      width_sum += width;
    }
    const double tol = 1.0e-12 * std::max(1.0, std::abs(length));
    TENRYU_ASSERT(std::abs(width_sum - length) <= tol,
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
    TENRYU_ASSERT(nodes[i + 1] > nodes[i], "graded grid nodes must be strictly increasing");
  }
  return nodes;
}

void set_1d_node_flags(MeshTopology& topo) {
  topo.node_flags.assign(static_cast<std::size_t>(topo.nr + 1), NODE_NONE);
  for (int j = 0; j <= topo.nr; ++j) {
    std::uint8_t flags = NODE_NONE;
    if (j == 0) {
      flags |= NODE_CENTER;
    }
    if (j == topo.nr) {
      flags |= NODE_BOUNDARY;
    }
    topo.node_flags[static_cast<std::size_t>(j)] = flags;
  }
}

void set_2d_node_flags(MeshTopology& topo,
                       const std::vector<double>& node_r,
                       const bool mark_pole_axis = false,
                       const bool mark_theta0_pole_axis = true) {
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == topo.n_nodes,
                "set_2d_node_flags requires node_r size == topo.n_nodes");
  topo.node_flags.assign(static_cast<std::size_t>(topo.n_nodes), NODE_NONE);
  double r_max = 0.0;
  for (const double r : node_r) {
    r_max = std::max(r_max, std::abs(r));
  }
  const double axis_tol = core::axis_tolerance(r_max);
  for (int i = 0; i <= topo.nr; ++i) {
    for (int j = 0; j <= topo.nz; ++j) {
      std::uint8_t flags = NODE_NONE;
      const int n = topo.node_index(i, j);
      if (std::abs(node_r[static_cast<std::size_t>(n)]) <= axis_tol) {
        flags |= NODE_AXIS;
      }
      if (i == topo.nr || j == 0 || j == topo.nz) {
        flags |= NODE_BOUNDARY;
      }
      if (mark_theta0_pole_axis) {
        if (mark_pole_axis && i > 0 && (j == 0 || j == topo.nz)) {
          flags |= NODE_POLE_AXIS;
        }
      } else if (mark_pole_axis && i > 0 && j == topo.nz) {
        flags |= NODE_POLE_AXIS;
      }
      topo.node_flags[static_cast<std::size_t>(n)] = flags;
    }
  }
}

void set_rectangular_edge_tags(Mesh& mesh) {
  mesh.edge_tags.assign(static_cast<std::size_t>(mesh.topo.n_edges()),
                        BoundaryKind::Interior);
  for (int i = 0; i < mesh.topo.nr; ++i) {
    mesh.edge_tags[static_cast<std::size_t>(mesh.topo.radial_edge_index(i, 0))] =
        BoundaryKind::RectZBottom;
    mesh.edge_tags[static_cast<std::size_t>(mesh.topo.radial_edge_index(i, mesh.topo.nz))] =
        BoundaryKind::RectZTop;
  }
  for (int j = 0; j < mesh.topo.nz; ++j) {
    mesh.edge_tags[static_cast<std::size_t>(mesh.topo.angular_edge_index(0, j))] =
        BoundaryKind::RectRInnerAxis;
    mesh.edge_tags[static_cast<std::size_t>(mesh.topo.angular_edge_index(mesh.topo.nr, j))] =
        BoundaryKind::RectROuter;
  }
}

void set_spherical_polar_edge_tags(Mesh& mesh,
                                   const bool has_polar_cut_face = false) {
  mesh.edge_tags.assign(static_cast<std::size_t>(mesh.topo.n_edges()),
                        BoundaryKind::Interior);
  for (int i = 0; i < mesh.topo.nr; ++i) {
    if (has_polar_cut_face) {
      mesh.edge_tags[static_cast<std::size_t>(
          mesh.topo.radial_edge_index(i, 0))] = BoundaryKind::PolarCutFace;
    } else {
      mesh.edge_tags[static_cast<std::size_t>(mesh.topo.radial_edge_index(i, 0))] =
          BoundaryKind::RZAxisTheta0;
    }
    mesh.edge_tags[static_cast<std::size_t>(mesh.topo.radial_edge_index(i, mesh.topo.nz))] =
        BoundaryKind::RZAxisThetaPi;
  }
  for (int j = 0; j < mesh.topo.nz; ++j) {
    mesh.edge_tags[static_cast<std::size_t>(mesh.topo.angular_edge_index(0, j))] =
        BoundaryKind::SphericalInnerCore;
    mesh.edge_tags[static_cast<std::size_t>(mesh.topo.angular_edge_index(mesh.topo.nr, j))] =
        BoundaryKind::SphericalOuterFree;
  }
}

struct RZPoint {
  double r = 0.0;
  double z = 0.0;
};

double geometric_sum(const double q, const int count) {
  if (count <= 0) {
    return 0.0;
  }
  if (std::abs(q - 1.0) <= 1.0e-12) {
    return static_cast<double>(count);
  }
  double sum = 0.0;
  double term = 1.0;
  for (int k = 0; k < count; ++k) {
    sum += term;
    term *= q;
  }
  return sum;
}

double solve_geometric_ratio(const double clearance,
                             const double first_width,
                             const int count) {
  TENRYU_ASSERT(first_width > 0.0,
                "polar_in_box first morph width must be positive");
  TENRYU_ASSERT(count > 0, "polar_in_box morph ring count must be positive");
  if (!(clearance >= first_width)) {
    throw core::namelist::ConfigError(
        "polar_in_box inset leaves less than one prefix-exit cell width; "
        "increase morph_rings or the box margin");
  }

  const auto residual = [&](const double q) {
    return first_width * geometric_sum(q, count) - clearance;
  };
  if (std::abs(residual(1.0)) <=
      1.0e-14 * std::max(clearance, first_width)) {
    return 1.0;
  }

  double lo = 0.0;
  double hi = 1.0;
  if (residual(1.0) < 0.0) {
    lo = 1.0;
    hi = 2.0;
    while (residual(hi) < 0.0) {
      hi *= 2.0;
      TENRYU_ASSERT(std::isfinite(hi) && hi < 1.0e6,
                    "polar_in_box geometric-ratio bracket overflow");
    }
  }
  for (int iter = 0; iter < 96; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (residual(mid) < 0.0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 0.5 * (lo + hi);
}

double delayed_quintic_blend(const double xi) {
  constexpr double delay_fraction = 0.20;
  const double u = std::clamp(
      (xi - delay_fraction) / (1.0 - delay_fraction), 0.0, 1.0);
  return u * u * u * (10.0 + u * (-15.0 + 6.0 * u));
}

int nearest_theta_index(const std::vector<double>& theta,
                        const double target) {
  TENRYU_ASSERT(!theta.empty(), "polar_in_box theta ladder must be non-empty");
  int nearest = 0;
  double best = std::abs(theta.front() - target);
  for (int j = 1; j < static_cast<int>(theta.size()); ++j) {
    const double distance =
        std::abs(theta[static_cast<std::size_t>(j)] - target);
    if (distance < best) {
      nearest = j;
      best = distance;
    }
  }
  return nearest;
}

RZPoint polar_in_box_inset_point(const tenryu::core::Config::MeshConfig& cfg,
                                 const std::vector<double>& theta,
                                 const int j_tr,
                                 const int j_br,
                                 const int j,
                                 const double inset) {
  const int ntheta = static_cast<int>(theta.size()) - 1;
  const bool final_ring = inset == 0.0;
  const double r_side = final_ring ? cfg.box_r_max : cfg.box_r_max - inset;
  const double z_top = final_ring ? cfg.box_z_max : cfg.box_z_max - inset;
  const double z_bottom =
      final_ring ? cfg.box_z_min : cfg.box_z_min + inset;
  if (j == j_tr) {
    return {r_side, z_top};
  }
  if (j == j_br) {
    return {r_side, z_bottom};
  }
  if (j < j_tr) {
    const double u =
        (theta[static_cast<std::size_t>(j)] - theta.front()) /
        (theta[static_cast<std::size_t>(j_tr)] - theta.front());
    return {u * r_side, z_top};
  }
  if (j < j_br) {
    const double u =
        (theta[static_cast<std::size_t>(j)] -
         theta[static_cast<std::size_t>(j_tr)]) /
        (theta[static_cast<std::size_t>(j_br)] -
         theta[static_cast<std::size_t>(j_tr)]);
    return {r_side, z_top - u * (z_top - z_bottom)};
  }
  const double u =
      (theta[static_cast<std::size_t>(j)] -
       theta[static_cast<std::size_t>(j_br)]) /
      (theta[static_cast<std::size_t>(ntheta)] -
       theta[static_cast<std::size_t>(j_br)]);
  return {(1.0 - u) * r_side, z_bottom};
}

double polar_angle(const RZPoint& point, const double center_z) {
  return std::atan2(point.r, point.z - center_z);
}

double polar_in_box_q_rect(const double half_width_r,
                           const double z_min,
                           const double z_max,
                           const double center_z,
                           const double phi) {
  const double sin_phi = std::sin(phi);
  const double cos_phi = std::cos(phi);
  double q = std::numeric_limits<double>::infinity();
  if (sin_phi > 0.0) {
    const double candidate = half_width_r / sin_phi;
    if (std::isfinite(candidate) && candidate > 0.0) {
      q = std::min(q, candidate);
    }
  }
  if (cos_phi > 0.0) {
    const double candidate = (z_max - center_z) / cos_phi;
    if (std::isfinite(candidate) && candidate > 0.0) {
      q = std::min(q, candidate);
    }
  }
  if (cos_phi < 0.0) {
    const double candidate = (z_min - center_z) / cos_phi;
    if (std::isfinite(candidate) && candidate > 0.0) {
      q = std::min(q, candidate);
    }
  }
  TENRYU_ASSERT(std::isfinite(q) && q > 0.0,
                "polar_in_box rectangle ray must have a finite positive intersection");
  return q;
}

double median(std::vector<double> values) {
  TENRYU_ASSERT(!values.empty(), "polar_in_box median requires samples");
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2U;
  if ((values.size() & 1U) != 0U) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1U] + values[mid]);
}

struct PolarInBoxCollarLayout {
  double inset = 0.0;
  double first_width = 0.0;
  double ratio = 1.0;
  std::vector<double> morph_ratio;
  std::vector<RZPoint> morph_target;
};

PolarInBoxCollarLayout build_polar_in_box_collar_layout(
    const tenryu::core::Config::MeshConfig& cfg,
    const std::vector<double>& theta,
    const int j_tr,
    const int j_br) {
  const int ntheta = static_cast<int>(theta.size()) - 1;
  const int nm = cfg.morph_rings;
  const int nc = cfg.collar_rings;
  const double center_z = cfg.box_center_z;
  const double s_b = cfg.explicit_nodes.back();
  const double delta_b =
      cfg.explicit_nodes.back() - cfg.explicit_nodes[cfg.explicit_nodes.size() - 2U];
  const double q_c = std::min(cfg.morph_growth_max, 1.10);
  const double collar_factor = geometric_sum(q_c, nc);

  const auto evaluate = [&](const double inset,
                            std::vector<double>* q_out,
                            std::vector<RZPoint>* target_out) {
    std::vector<double> projected;
    projected.reserve(static_cast<std::size_t>(ntheta + 1));
    std::vector<double> ratios(static_cast<std::size_t>(ntheta + 1), 1.0);
    std::vector<RZPoint> targets(static_cast<std::size_t>(ntheta + 1));
    const double xi_prev = static_cast<double>(nm - 1) / static_cast<double>(nm);
    const double blend_prev = delayed_quintic_blend(xi_prev);
    for (int j = 0; j <= ntheta; ++j) {
      const RZPoint target = polar_in_box_inset_point(
          cfg, theta, j_tr, j_br, j, inset);
      const double target_phi = polar_angle(target, center_z);
      const double clearance =
          std::hypot(target.r, target.z - center_z) - s_b;
      const double q = solve_geometric_ratio(clearance, delta_b, nm);
      const double phi_prev =
          theta[static_cast<std::size_t>(j)] +
          blend_prev *
              (target_phi - theta[static_cast<std::size_t>(j)]);
      const double radius_prev =
          s_b + delta_b * geometric_sum(q, nm - 1);
      const RZPoint previous{
          (j == 0 || j == ntheta) ? 0.0 : radius_prev * std::sin(phi_prev),
          center_z + radius_prev * std::cos(phi_prev)};
      double thickness = 0.0;
      if (j < j_tr) {
        thickness = target.z - previous.z;
      } else if (j <= j_br) {
        thickness = target.r - previous.r;
      } else {
        thickness = previous.z - target.z;
      }
      if (!(std::isfinite(thickness) && thickness > 0.0)) {
        throw core::namelist::ConfigError(
            "polar_in_box morph exit has non-positive side-normal thickness");
      }
      projected.push_back(thickness);
      ratios[static_cast<std::size_t>(j)] = q;
      targets[static_cast<std::size_t>(j)] = target;
    }
    if (q_out != nullptr) {
      *q_out = std::move(ratios);
    }
    if (target_out != nullptr) {
      *target_out = std::move(targets);
    }
    return median(std::move(projected));
  };

  const double max_inset =
      std::min({cfg.box_r_max,
                cfg.box_z_max - center_z,
                center_z - cfg.box_z_min}) -
      s_b - delta_b;
  if (!(max_inset > 0.0)) {
    throw core::namelist::ConfigError(
        "polar_in_box box margin cannot fit the morph/collar join");
  }

  double lo = 0.0;
  double hi = 0.0;
  bool bracketed = false;
  for (int sample = 1; sample <= 256; ++sample) {
    const double candidate =
        max_inset * static_cast<double>(sample) / 257.0;
    const double f_candidate =
        candidate - collar_factor * evaluate(candidate, nullptr, nullptr);
    if (f_candidate >= 0.0) {
      hi = candidate;
      bracketed = true;
      break;
    }
    lo = candidate;
  }
  if (!bracketed) {
    throw core::namelist::ConfigError(
        "polar_in_box collar inset is incompatible with the box margin; "
        "increase morph_rings or the box margin");
  }
  for (int iter = 0; iter < 96; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double f_mid =
        mid - collar_factor * evaluate(mid, nullptr, nullptr);
    if (f_mid < 0.0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  PolarInBoxCollarLayout layout;
  layout.inset = 0.5 * (lo + hi);
  layout.first_width =
      evaluate(layout.inset, &layout.morph_ratio, &layout.morph_target);
  layout.ratio = q_c;
  for (const double q : layout.morph_ratio) {
    if (!(q <= cfg.morph_growth_max * (1.0 + 1.0e-14))) {
      throw core::namelist::ConfigError(
          "polar_in_box morph growth exceeds Mesh.morph_growth_max; "
          "increase morph_rings or the box margin");
    }
  }
  return layout;
}

struct PolarInBoxExteriorContext {
  enum class BlendScheduleOverride : std::uint8_t {
    Baseline,
    Cone,
  };

  const tenryu::core::Config::MeshConfig& cfg;
  const std::vector<double>& theta;
  int j_tr = -1;
  int j_br = -1;
  PolarInBoxCollarLayout collar;
  double delta_b = 0.0;
  double target_angle_override = std::numeric_limits<double>::quiet_NaN();
  BlendScheduleOverride blend_schedule_override =
      BlendScheduleOverride::Baseline;
  const std::vector<double>* cone_blend = nullptr;
};

PolarInBoxExteriorContext build_polar_in_box_exterior_context(
    const tenryu::core::Config::MeshConfig& cfg,
    const std::vector<double>& theta,
    const int j_tr,
    const int j_br) {
  const PolarInBoxCollarLayout collar =
      build_polar_in_box_collar_layout(cfg, theta, j_tr, j_br);
  const double delta_b =
      cfg.explicit_nodes.back() -
      cfg.explicit_nodes[cfg.explicit_nodes.size() - 2U];
  return {cfg, theta, j_tr, j_br, collar, delta_b};
}

struct PolarInBoxExteriorRay {
  // Ring indices p = 0..(morph_rings+collar_rings-1), where p=0 is
  // the first ring beyond the polar prefix and the last entry is on the box.
  std::vector<RZPoint> nodes;
};

PolarInBoxExteriorRay polar_in_box_exterior_ray(
    const int j,
    const RZPoint lock_node,
    const double lock_radius,
    const PolarInBoxExteriorContext& context) {
  const auto& cfg = context.cfg;
  const auto& theta_nodes = context.theta;
  const int ntheta = static_cast<int>(theta_nodes.size()) - 1;
  const double center_z = cfg.box_center_z;
  const double s_max = lock_radius;
  const auto& collar = context.collar;
  const double delta_b = context.delta_b;
  TENRYU_ASSERT(j >= 0 && j <= ntheta,
                "polar_in_box exterior ray index out of range");
  TENRYU_ASSERT(std::isfinite(lock_node.r) && std::isfinite(lock_node.z),
                "polar_in_box lock node must be finite");

  PolarInBoxExteriorRay ray;
  ray.nodes.reserve(static_cast<std::size_t>(cfg.morph_rings +
                                             cfg.collar_rings));
  if (std::isfinite(context.target_angle_override)) {
    TENRYU_ASSERT(
        context.blend_schedule_override ==
                PolarInBoxExteriorContext::BlendScheduleOverride::Cone &&
            context.cone_blend != nullptr &&
            context.cone_blend->size() ==
                static_cast<std::size_t>(cfg.morph_rings + cfg.collar_rings),
        "polar_in_box cone ray requires a complete cone blend schedule");
    const double target_phi = context.target_angle_override;
    TENRYU_ASSERT(target_phi > 0.0 &&
                      target_phi < spherical_wedge_detail::kPi,
                  "polar_in_box overridden ray angle must be in (0, pi)");
    const double first_target_radius = polar_in_box_q_rect(
        cfg.box_r_max - collar.inset,
        cfg.box_z_min + collar.inset,
        cfg.box_z_max - collar.inset,
        center_z,
        target_phi);
    const double morph_ratio = solve_geometric_ratio(
        first_target_radius - s_max, delta_b, cfg.morph_rings);
    if (!(morph_ratio <= cfg.morph_growth_max * (1.0 + 1.0e-14))) {
      throw core::namelist::ConfigError(
          "polar_in_box cone morph growth exceeds Mesh.morph_growth_max; "
          "increase morph_rings or the box margin");
    }
    for (int k = 1; k <= cfg.morph_rings; ++k) {
      const double beta =
          (*context.cone_blend)[static_cast<std::size_t>(k - 1)];
      const double phi =
          (1.0 - beta) * theta_nodes[static_cast<std::size_t>(j)] +
          beta * target_phi;
      const double radius =
          (k == cfg.morph_rings)
              ? first_target_radius
              : s_max + delta_b * geometric_sum(morph_ratio, k);
      ray.nodes.push_back(
          {radius * std::sin(phi), center_z + radius * std::cos(phi)});
    }

    const double unscaled_collar =
        collar.first_width * geometric_sum(collar.ratio, cfg.collar_rings);
    const double collar_scale = collar.inset / unscaled_collar;
    for (int k = 1; k <= cfg.collar_rings; ++k) {
      const int rings_to_box = cfg.collar_rings - k;
      const double inset =
          (rings_to_box == 0)
              ? 0.0
              : collar_scale * collar.first_width *
                    std::pow(collar.ratio, k) *
                    geometric_sum(collar.ratio, rings_to_box);
      const double beta = (*context.cone_blend)[static_cast<std::size_t>(
          cfg.morph_rings + k - 1)];
      const double phi =
          (1.0 - beta) * theta_nodes[static_cast<std::size_t>(j)] +
          beta * target_phi;
      const double radius = polar_in_box_q_rect(
          cfg.box_r_max - inset,
          cfg.box_z_min + inset,
          cfg.box_z_max - inset,
          center_z,
          phi);
      ray.nodes.push_back(
          {radius * std::sin(phi), center_z + radius * std::cos(phi)});
    }
    return ray;
  }

  for (int k = 1; k <= cfg.morph_rings; ++k) {
    const double xi = static_cast<double>(k) /
                      static_cast<double>(cfg.morph_rings);
    const double angular_blend = delayed_quintic_blend(xi);
    const RZPoint target =
        collar.morph_target[static_cast<std::size_t>(j)];
    const double target_phi = polar_angle(target, center_z);
    const double phi = theta_nodes[static_cast<std::size_t>(j)] +
                       angular_blend *
                           (target_phi -
                            theta_nodes[static_cast<std::size_t>(j)]);
    if (k == cfg.morph_rings) {
      ray.nodes.push_back(target);
    } else {
      const double radius =
          s_max +
          delta_b * geometric_sum(
                        collar.morph_ratio[static_cast<std::size_t>(j)], k);
      RZPoint point;
      point.r = (j == 0 || j == ntheta) ? 0.0 : radius * std::sin(phi);
      point.z = center_z + radius * std::cos(phi);
      ray.nodes.push_back(point);
    }
  }

  const double unscaled_collar =
      collar.first_width * geometric_sum(collar.ratio, cfg.collar_rings);
  const double collar_scale = collar.inset / unscaled_collar;
  for (int k = 1; k <= cfg.collar_rings; ++k) {
    const int rings_to_box = cfg.collar_rings - k;
    const double inset =
        (rings_to_box == 0)
            ? 0.0
            : collar_scale * collar.first_width *
                  std::pow(collar.ratio, k) *
                  geometric_sum(collar.ratio, rings_to_box);
    const RZPoint point = polar_in_box_inset_point(
        cfg, theta_nodes, context.j_tr, context.j_br, j, inset);
    ray.nodes.push_back({(j == 0 || j == ntheta) ? 0.0 : point.r, point.z});
  }
  return ray;
}

double cone_quintic_blend(const double radius,
                          const double activation_radius,
                          const double tip_radius) {
  const double t = std::clamp(
      (radius - activation_radius) / (tip_radius - activation_radius),
      0.0,
      1.0);
  const double t2 = t * t;
  const double t3 = t2 * t;
  return 6.0 * t3 * t2 - 15.0 * t2 * t2 + 10.0 * t3;
}

void apply_polar_in_box_cone_postpass(
    const std::vector<RZPoint>& lock_nodes,
    std::vector<PolarInBoxExteriorRay>* exterior_rays,
    const PolarInBoxExteriorContext& baseline_context) {
  const auto& cfg = baseline_context.cfg;
  if (!std::isfinite(cfg.cone_theta_wall)) {
    return;
  }
  const char* const cone_debug_env = std::getenv("TENRYU_CONE_DEBUG");
  const bool cone_debug = cone_debug_env != nullptr &&
                          cone_debug_env[0] == '1' &&
                          cone_debug_env[1] == '\0';
  TENRYU_ASSERT(exterior_rays != nullptr,
                "polar_in_box cone post-pass requires exterior rays");
  const int ntheta = static_cast<int>(baseline_context.theta.size()) - 1;
  const int exterior_count = cfg.morph_rings + cfg.collar_rings;
  TENRYU_ASSERT(lock_nodes.size() == static_cast<std::size_t>(ntheta + 1) &&
                    exterior_rays->size() ==
                        static_cast<std::size_t>(ntheta + 1),
                "polar_in_box cone post-pass ray count mismatch");
  std::vector<double> baseline_box_phi(static_cast<std::size_t>(ntheta + 1));
  for (int j = 0; j <= ntheta; ++j) {
    const RZPoint point =
        (*exterior_rays)[static_cast<std::size_t>(j)].nodes.back();
    baseline_box_phi[static_cast<std::size_t>(j)] =
        polar_angle(point, cfg.box_center_z);
  }
  int j_w = 1;
  double wall_distance =
      std::abs(baseline_box_phi[1] - cfg.cone_theta_wall);
  for (int j = 2; j < ntheta; ++j) {
    const double distance = std::abs(
        baseline_box_phi[static_cast<std::size_t>(j)] - cfg.cone_theta_wall);
    if (distance < wall_distance) {
      j_w = j;
      wall_distance = distance;
    }
  }
  const std::vector<double>* debug_target_phi = nullptr;
  const std::vector<bool>* debug_fully_overridden_ray = nullptr;
  const std::vector<bool>* debug_blended_ray = nullptr;
  const std::vector<double>* debug_transition_weight = nullptr;
  double debug_dphi_w_pre_clamp =
      std::numeric_limits<double>::quiet_NaN();
  double debug_dphi_w_post_clamp =
      std::numeric_limits<double>::quiet_NaN();
  double debug_h_wall = std::numeric_limits<double>::quiet_NaN();
  double debug_r_wall = std::numeric_limits<double>::quiet_NaN();
  int debug_n_t_minus = -1;
  int debug_n_t_plus = -1;
  int debug_tip_ring_index = -1;
  const auto format_cone_debug_double = [](const double value) {
    std::ostringstream stream;
    stream << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return stream.str();
  };
  const auto log_cone_anchor_table = [&](const char* reason) {
    if (!cone_debug) {
      return;
    }
    core::log_info(std::string("[cone-debug] anchor_table reason=") + reason);
    core::log_info(
        "[cone-debug] j_w=" + std::to_string(j_w) +
        " dphi_w_pre_clamp=" +
        format_cone_debug_double(debug_dphi_w_pre_clamp) +
        " dphi_w_post_clamp=" +
        format_cone_debug_double(debug_dphi_w_post_clamp));
    core::log_info(
        "[cone-debug] h_wall=" + format_cone_debug_double(debug_h_wall) +
        " R_wall=" + format_cone_debug_double(debug_r_wall) +
        " n_t_minus=" +
        (debug_n_t_minus >= 0 ? std::to_string(debug_n_t_minus) : "-") +
        " n_t_plus=" +
        (debug_n_t_plus >= 0 ? std::to_string(debug_n_t_plus) : "-") +
        " tip_ring_index=" +
        (debug_tip_ring_index >= 0
             ? std::to_string(debug_tip_ring_index)
             : "-"));
    core::log_info(
        "[cone-debug] j_tr=" + std::to_string(baseline_context.j_tr) +
        " j_tr_box_phi=" +
        format_cone_debug_double(
            baseline_box_phi[static_cast<std::size_t>(baseline_context.j_tr)]) +
        " j_br=" + std::to_string(baseline_context.j_br) +
        " j_br_box_phi=" +
        format_cone_debug_double(
            baseline_box_phi[static_cast<std::size_t>(baseline_context.j_br)]));
    core::log_info(
        "[cone-debug] columns: j prefix_theta baseline_box_phi target_phi "
        "transition_weight moved");
    const int first_j = std::max(0, j_w - 12);
    const int last_j = std::min(ntheta, j_w + 12);
    for (int j = first_j; j <= last_j; ++j) {
      const std::size_t index = static_cast<std::size_t>(j);
      const bool fully_overridden =
          debug_fully_overridden_ray != nullptr &&
          (*debug_fully_overridden_ray)[index];
      const bool blended =
          debug_blended_ray != nullptr && (*debug_blended_ray)[index];
      const bool target_unchanged =
          debug_target_phi == nullptr ||
          (*debug_target_phi)[index] == baseline_box_phi[index];
      const std::string target =
          target_unchanged
              ? "base"
              : format_cone_debug_double((*debug_target_phi)[index]);
      const std::string weight =
          blended && debug_transition_weight != nullptr
              ? format_cone_debug_double((*debug_transition_weight)[index])
              : "-";
      core::log_info(
          "[cone-debug] j=" + std::to_string(j) +
          " prefix_theta=" +
          format_cone_debug_double(baseline_context.theta[index]) +
          " baseline_box_phi=" +
          format_cone_debug_double(baseline_box_phi[index]) +
          " target_phi=" + target + " transition_weight=" + weight +
          " moved=" + (fully_overridden || blended ? "1" : "0"));
    }
  };
  if (j_w < 2 || j_w > ntheta - 2 ||
      std::abs(j_w - baseline_context.j_tr) < 2 ||
      std::abs(j_w - baseline_context.j_br) < 2) {
    log_cone_anchor_table("wall-neighborhood-capacity");
    throw core::namelist::ConfigError(
        "polar_in_box cone angular capacity/anchor violation in wall neighborhood");
  }

  std::vector<double> wall_baseline_radius(
      static_cast<std::size_t>(exterior_count + 1), 0.0);
  wall_baseline_radius.front() =
      std::hypot(lock_nodes[static_cast<std::size_t>(j_w)].r,
                 lock_nodes[static_cast<std::size_t>(j_w)].z -
                     cfg.box_center_z);
  int tip_p = -1;
  for (int p = 0; p < exterior_count; ++p) {
    const RZPoint& point =
        (*exterior_rays)[static_cast<std::size_t>(j_w)]
            .nodes[static_cast<std::size_t>(p)];
    const double radius =
        std::hypot(point.r, point.z - cfg.box_center_z);
    wall_baseline_radius[static_cast<std::size_t>(p + 1)] = radius;
    if (tip_p < 0 && radius >= cfg.cone_tip_radius) {
      tip_p = p;
    }
  }
  debug_tip_ring_index =
      tip_p >= 0 ? cfg.polar_prefix_nr + tip_p + 1 : -1;
  if (tip_p < 0) {
    log_cone_anchor_table("tip-radius-capacity");
    throw core::namelist::ConfigError(
        "polar_in_box cone tip radius exceeds the wall-ray box radius");
  }
  const double r_wall =
      wall_baseline_radius[static_cast<std::size_t>(tip_p + 1)];
  const double h_wall =
      r_wall - wall_baseline_radius[static_cast<std::size_t>(tip_p)];
  debug_h_wall = h_wall;
  debug_r_wall = r_wall;
  if (cone_debug &&
      !(std::isfinite(h_wall) && h_wall > 0.0 &&
        std::isfinite(r_wall) && r_wall > 0.0)) {
    log_cone_anchor_table("wall-tip-spacing-assert");
  }
  TENRYU_ASSERT(std::isfinite(h_wall) && h_wall > 0.0 &&
                    std::isfinite(r_wall) && r_wall > 0.0,
                "polar_in_box cone wall tip spacing must be finite and positive");
  const double dphi_w_pre_clamp = h_wall / r_wall;
  const double dphi_w = std::min(
      dphi_w_pre_clamp,
      0.5 * std::min(
                baseline_box_phi[static_cast<std::size_t>(j_w + 1)] -
                    baseline_box_phi[static_cast<std::size_t>(j_w)],
                baseline_box_phi[static_cast<std::size_t>(j_w)] -
                    baseline_box_phi[static_cast<std::size_t>(j_w - 1)]));
  debug_dphi_w_pre_clamp = dphi_w_pre_clamp;
  debug_dphi_w_post_clamp = dphi_w;
  if (cone_debug && !(std::isfinite(dphi_w) && dphi_w > 0.0)) {
    log_cone_anchor_table("fine-angular-spacing-assert");
  }
  TENRYU_ASSERT(std::isfinite(dphi_w) && dphi_w > 0.0,
                "polar_in_box cone fine angular spacing must be finite and positive");

  std::vector<double> target_phi = baseline_box_phi;
  std::vector<bool> fully_overridden_ray(
      static_cast<std::size_t>(ntheta + 1), false);
  std::vector<bool> blended_ray(static_cast<std::size_t>(ntheta + 1), false);
  std::vector<double> transition_weight(
      static_cast<std::size_t>(ntheta + 1), 0.0);
  debug_target_phi = &target_phi;
  debug_fully_overridden_ray = &fully_overridden_ray;
  debug_blended_ray = &blended_ray;
  debug_transition_weight = &transition_weight;
  target_phi[static_cast<std::size_t>(j_w)] = cfg.cone_theta_wall;
  fully_overridden_ray[static_cast<std::size_t>(j_w)] = true;

  const auto set_fine_band = [&](const int direction,
                                 const int fine_cells,
                                 const char* neighborhood) {
    const double fine_end_phi =
        cfg.cone_theta_wall +
        static_cast<double>(direction * fine_cells) * dphi_w;
    bool contains_corner = false;
    for (const int corner_j :
         {baseline_context.j_tr, baseline_context.j_br}) {
      const int corner_index_from_wall = direction * (corner_j - j_w);
      const double corner_phi =
          baseline_box_phi[static_cast<std::size_t>(corner_j)];
      const double from_wall = static_cast<double>(direction) *
                               (corner_phi - cfg.cone_theta_wall);
      const double to_edge = static_cast<double>(direction) *
                             (fine_end_phi - corner_phi);
      if ((corner_index_from_wall >= 0 &&
           corner_index_from_wall <= fine_cells) ||
          (from_wall > 0.0 && to_edge > 0.0)) {
        contains_corner = true;
        break;
      }
    }
    if (!(fine_end_phi > 0.0 &&
          fine_end_phi < spherical_wedge_detail::kPi) ||
        contains_corner) {
      log_cone_anchor_table(
          direction < 0 ? "minus-fine-band-capacity"
                        : "plus-fine-band-capacity");
      throw core::namelist::ConfigError(
          std::string("polar_in_box cone angular capacity/anchor violation in ") +
          neighborhood + " fine-band neighborhood");
    }
    for (int m = 1; m <= fine_cells; ++m) {
      const int j = j_w + direction * m;
      if (j <= 0 || j >= ntheta) {
        log_cone_anchor_table(
            direction < 0 ? "minus-fine-band-index"
                          : "plus-fine-band-index");
        throw core::namelist::ConfigError(
            std::string("polar_in_box cone angular capacity/anchor violation in ") +
            neighborhood + " fine-band neighborhood");
      }
      target_phi[static_cast<std::size_t>(j)] =
          cfg.cone_theta_wall +
          static_cast<double>(direction * m) * dphi_w;
      fully_overridden_ray[static_cast<std::size_t>(j)] = true;
    }
  };
  set_fine_band(-1, cfg.cone_fine_cells_minus, "minus");
  set_fine_band(1, cfg.cone_fine_cells_plus, "plus");
  const int minus_outer_j = j_w - cfg.cone_fine_cells_minus;
  if (!(target_phi[static_cast<std::size_t>(minus_outer_j)] >
        baseline_box_phi[static_cast<std::size_t>(minus_outer_j - 1)])) {
    log_cone_anchor_table("minus-fine-band-order");
    throw core::namelist::ConfigError(
        "polar_in_box cone angular capacity/anchor violation in minus fine-band neighborhood");
  }
  const int plus_outer_j = j_w + cfg.cone_fine_cells_plus;
  if (!(target_phi[static_cast<std::size_t>(plus_outer_j)] <
        baseline_box_phi[static_cast<std::size_t>(plus_outer_j + 1)])) {
    log_cone_anchor_table("plus-fine-band-order");
    throw core::namelist::ConfigError(
        "polar_in_box cone angular capacity/anchor violation in plus fine-band neighborhood");
  }

  const auto set_transition = [&](const int direction,
                                  const int fine_cells,
                                  const char* neighborhood) {
    constexpr int min_transition_rays = 4;
    const int fine_j = j_w + direction * fine_cells;
    const int first_transition_j = fine_j + direction;
    const double dphi_base = std::abs(
        baseline_box_phi[static_cast<std::size_t>(first_transition_j)] -
        baseline_box_phi[static_cast<std::size_t>(fine_j)]);
    const double grading_rays = std::ceil(
        std::abs(std::log(dphi_base / dphi_w)) /
        std::log(cfg.cone_angular_growth_max));
    const int available_transition_rays =
        direction < 0 ? fine_j - 2 : ntheta - 2 - fine_j;
    if (cone_debug && std::isfinite(grading_rays) && grading_rays >= 0.0 &&
        grading_rays <=
            static_cast<double>(std::numeric_limits<int>::max())) {
      const int diagnostic_transition_rays =
          std::max(min_transition_rays, static_cast<int>(grading_rays));
      if (direction < 0) {
        debug_n_t_minus = diagnostic_transition_rays;
      } else {
        debug_n_t_plus = diagnostic_transition_rays;
      }
    }
    if (!(std::isfinite(grading_rays) && dphi_base > 0.0) ||
        grading_rays > static_cast<double>(available_transition_rays)) {
      log_cone_anchor_table(
          direction < 0 ? "minus-transition-capacity"
                        : "plus-transition-capacity");
      throw core::namelist::ConfigError(
          std::string("polar_in_box cone angular capacity/anchor violation in ") +
          neighborhood + " transition neighborhood");
    }
    const int transition_rays = std::max(
        min_transition_rays, static_cast<int>(grading_rays));
    if (direction < 0) {
      debug_n_t_minus = transition_rays;
    } else {
      debug_n_t_plus = transition_rays;
    }
    if (transition_rays > available_transition_rays) {
      log_cone_anchor_table(
          direction < 0 ? "minus-transition-minimum-capacity"
                        : "plus-transition-minimum-capacity");
      throw core::namelist::ConfigError(
          std::string("polar_in_box cone angular capacity/anchor violation in ") +
          neighborhood + " transition neighborhood");
    }
    for (int r = 1; r <= transition_rays; ++r) {
      const int j = fine_j + direction * r;
      target_phi[static_cast<std::size_t>(j)] =
          baseline_box_phi[static_cast<std::size_t>(j)];
      const double t =
          1.0 - static_cast<double>(r) /
                    static_cast<double>(transition_rays + 1);
      const double t2 = t * t;
      const double t3 = t2 * t;
      transition_weight[static_cast<std::size_t>(j)] =
          6.0 * t3 * t2 - 15.0 * t2 * t2 + 10.0 * t3;
      blended_ray[static_cast<std::size_t>(j)] = true;
    }
  };
  set_transition(-1, cfg.cone_fine_cells_minus, "minus");
  set_transition(1, cfg.cone_fine_cells_plus, "plus");
  log_cone_anchor_table("configured-anchor-layout");

  for (int j = 1; j <= ntheta; ++j) {
    if (!(target_phi[static_cast<std::size_t>(j)] >
          target_phi[static_cast<std::size_t>(j - 1)])) {
      log_cone_anchor_table("box-angle-order");
      throw core::namelist::ConfigError(
          "polar_in_box cone angular capacity violation in box-angle neighborhood j=" +
          std::to_string(j - 1) + "/" + std::to_string(j));
    }
  }

  std::vector<double> cone_blend(static_cast<std::size_t>(exterior_count));
  for (int p = 0; p < exterior_count; ++p) {
    cone_blend[static_cast<std::size_t>(p)] = cone_quintic_blend(
        wall_baseline_radius[static_cast<std::size_t>(p + 1)],
        cfg.cone_activation_radius,
        cfg.cone_tip_radius);
  }
  for (int j = 0; j <= ntheta; ++j) {
    const bool fully_overridden =
        fully_overridden_ray[static_cast<std::size_t>(j)];
    const bool blended = blended_ray[static_cast<std::size_t>(j)];
    if (!fully_overridden && !blended) {
      continue;
    }
    PolarInBoxExteriorContext override_context = baseline_context;
    override_context.target_angle_override =
        target_phi[static_cast<std::size_t>(j)];
    override_context.blend_schedule_override =
        PolarInBoxExteriorContext::BlendScheduleOverride::Cone;
    override_context.cone_blend = &cone_blend;
    const double lock_radius =
        std::hypot(lock_nodes[static_cast<std::size_t>(j)].r,
                   lock_nodes[static_cast<std::size_t>(j)].z -
                       cfg.box_center_z);
    PolarInBoxExteriorRay radial_ray = polar_in_box_exterior_ray(
        j,
        lock_nodes[static_cast<std::size_t>(j)],
        lock_radius,
        override_context);
    if (fully_overridden) {
      (*exterior_rays)[static_cast<std::size_t>(j)] =
          std::move(radial_ray);
      continue;
    }
    const double weight = transition_weight[static_cast<std::size_t>(j)];
    for (int p = 0; p < exterior_count; ++p) {
      RZPoint& baseline_point =
          (*exterior_rays)[static_cast<std::size_t>(j)]
              .nodes[static_cast<std::size_t>(p)];
      const RZPoint& radial_point =
          radial_ray.nodes[static_cast<std::size_t>(p)];
      baseline_point.r =
          (1.0 - weight) * baseline_point.r + weight * radial_point.r;
      baseline_point.z =
          (1.0 - weight) * baseline_point.z + weight * radial_point.z;
    }
  }

  for (int p = 0; p < exterior_count; ++p) {
    double previous_phi = -1.0;
    for (int j = 0; j <= ntheta; ++j) {
      const RZPoint& point =
          (*exterior_rays)[static_cast<std::size_t>(j)]
              .nodes[static_cast<std::size_t>(p)];
      const double phi = polar_angle(point, cfg.box_center_z);
      if (cone_debug && !(j == 0 || phi > previous_phi)) {
        log_cone_anchor_table("exterior-ring-angle-order-assert");
      }
      TENRYU_ASSERT(j == 0 || phi > previous_phi,
                    "polar_in_box cone exterior-ring angles must be strictly ordered");
      previous_phi = phi;
    }
  }
  for (int j = 0; j <= ntheta; ++j) {
    if (!fully_overridden_ray[static_cast<std::size_t>(j)]) {
      continue;
    }
    double previous_radius =
        std::hypot(lock_nodes[static_cast<std::size_t>(j)].r,
                   lock_nodes[static_cast<std::size_t>(j)].z -
                       cfg.box_center_z);
    for (int p = 0; p < exterior_count; ++p) {
      const RZPoint& point =
          (*exterior_rays)[static_cast<std::size_t>(j)]
              .nodes[static_cast<std::size_t>(p)];
      const double radius =
          std::hypot(point.r, point.z - cfg.box_center_z);
      if (cone_debug && !(radius > previous_radius)) {
        log_cone_anchor_table("exterior-ray-radius-nesting-assert");
      }
      TENRYU_ASSERT(radius > previous_radius,
                    "polar_in_box cone exterior-ray radii must be strictly nested");
      previous_radius = radius;
    }
  }
  for (int p = 0; p < exterior_count; ++p) {
    for (int j = 0; j < ntheta; ++j) {
      const RZPoint& n00 =
          (p == 0)
              ? lock_nodes[static_cast<std::size_t>(j)]
              : (*exterior_rays)[static_cast<std::size_t>(j)]
                    .nodes[static_cast<std::size_t>(p - 1)];
      const RZPoint& n10 =
          (*exterior_rays)[static_cast<std::size_t>(j)]
              .nodes[static_cast<std::size_t>(p)];
      const RZPoint& n11 =
          (*exterior_rays)[static_cast<std::size_t>(j + 1)]
              .nodes[static_cast<std::size_t>(p)];
      const RZPoint& n01 =
          (p == 0)
              ? lock_nodes[static_cast<std::size_t>(j + 1)]
              : (*exterior_rays)[static_cast<std::size_t>(j + 1)]
                    .nodes[static_cast<std::size_t>(p - 1)];
      const double r[4] = {n00.r, n10.r, n11.r, n01.r};
      const double z[4] = {n00.z, n10.z, n11.z, n01.z};
      if (cone_debug && !(-rz_polygon_area2(r, z, 4) > 0.0)) {
        const int q = cfg.polar_prefix_nr + p;
        const int k = j;
        log_cone_anchor_table("exterior-cell-positive-assert");
        core::log_info("[cone-debug] failing_exterior_cell q=" +
                       std::to_string(q) + " k=" + std::to_string(k));
        for (int node = 0; node < 4; ++node) {
          core::log_info(
              "[cone-debug] failing_exterior_cell_node index=" +
              std::to_string(node) +
              " r=" + format_cone_debug_double(r[node]) +
              " z=" + format_cone_debug_double(z[node]));
        }
      }
      TENRYU_ASSERT(-rz_polygon_area2(r, z, 4) > 0.0,
                    "polar_in_box cone exterior cell signed area must be positive");
    }
  }

  const double theta_ulp = std::max(
      std::nextafter(cfg.cone_theta_wall,
                     std::numeric_limits<double>::infinity()) -
          cfg.cone_theta_wall,
      cfg.cone_theta_wall -
          std::nextafter(cfg.cone_theta_wall,
                         -std::numeric_limits<double>::infinity()));
  for (int p = tip_p; p < exterior_count; ++p) {
    const RZPoint& point =
        (*exterior_rays)[static_cast<std::size_t>(j_w)]
            .nodes[static_cast<std::size_t>(p)];
    const double phi = polar_angle(point, cfg.box_center_z);
    if (cone_debug &&
        !(std::abs(phi - cfg.cone_theta_wall) <= 2.0 * theta_ulp)) {
      log_cone_anchor_table("wall-angle-assert");
    }
    TENRYU_ASSERT(std::abs(phi - cfg.cone_theta_wall) <= 2.0 * theta_ulp,
                  "polar_in_box cone wall angle must equal theta_wall within two ulp");
    if (p + 1 < exterior_count) {
      const RZPoint& next =
          (*exterior_rays)[static_cast<std::size_t>(j_w)]
              .nodes[static_cast<std::size_t>(p + 1)];
      const double z0 = point.z - cfg.box_center_z;
      const double z1 = next.z - cfg.box_center_z;
      const double cross_value = point.r * z1 - z0 * next.r;
      const double bound =
          8.0 * DBL_EPSILON * std::hypot(point.r, z0) *
          std::hypot(next.r, z1);
      if (cone_debug && !(std::abs(cross_value) <= bound)) {
        log_cone_anchor_table("wall-straightness-assert");
      }
      TENRYU_ASSERT(std::abs(cross_value) <= bound,
                    "polar_in_box cone wall straightness exceeds eight ulp");
    }
  }

  constexpr double tan_twenty_degrees = 0.36397023426620234;
  const double wall_shift =
      std::abs(cfg.cone_theta_wall -
               baseline_context.theta[static_cast<std::size_t>(j_w)]);
  const double l_tip =
      cfg.cone_tip_radius - cfg.cone_activation_radius;
  const double design_skew_bound =
      1.875 * cfg.cone_tip_radius * wall_shift / l_tip;
  if (cone_debug && !(design_skew_bound <= tan_twenty_degrees)) {
    log_cone_anchor_table("design-skew-bound-assert");
  }
  TENRYU_ASSERT(
      design_skew_bound <= tan_twenty_degrees,
      "polar_in_box cone L_tip design bound requires "
      "1.875*s_tip*|dphi_shift|/L_tip <= tan(20 deg)");
  for (int p = 0; p < exterior_count; ++p) {
    const double s0 = wall_baseline_radius[static_cast<std::size_t>(p)];
    const double s1 = wall_baseline_radius[static_cast<std::size_t>(p + 1)];
    const double lo = std::max(s0, cfg.cone_activation_radius);
    const double hi = std::min(s1, cfg.cone_tip_radius);
    if (!(hi > lo)) {
      continue;
    }
    const double beta_lo = cone_quintic_blend(
        lo, cfg.cone_activation_radius, cfg.cone_tip_radius);
    const double beta_hi = cone_quintic_blend(
        hi, cfg.cone_activation_radius, cfg.cone_tip_radius);
    const double skew = hi * wall_shift * (beta_hi - beta_lo) / (hi - lo);
    if (cone_debug && !(skew <= tan_twenty_degrees)) {
      log_cone_anchor_table("tip-skew-assert");
    }
    TENRYU_ASSERT(
        skew <= tan_twenty_degrees,
        "polar_in_box cone L_tip skew gate requires s*|dphi/ds| <= tan(20 deg)");
  }
}

__host__ __device__ double rz_polygon_volume_exact(const double* r,
                                                   const double* z,
                                                   const int nverts) {
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp1 = (k + 1) % nverts;
    sum += (r[k] + r[kp1]) * (r[k] * z[kp1] - r[kp1] * z[k]);
  }
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  return pi_over_three * sum;
}

__host__ __device__ double rz_polygon_area2(const double* r,
                                            const double* z,
                                            const int nverts) {
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp1 = (k + 1) % nverts;
    sum += r[k] * z[kp1] - r[kp1] * z[k];
  }
  return sum;
}

__host__ __device__ void rz_polygon_area_centroid(const double* r,
                                                  const double* z,
                                                  const int nverts,
                                                  double* centroid_r,
                                                  double* centroid_z) {
  double cross_sum = 0.0;
  double cr_sum = 0.0;
  double cz_sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp1 = (k + 1) % nverts;
    const double cross = r[k] * z[kp1] - r[kp1] * z[k];
    cross_sum += cross;
    cr_sum += (r[k] + r[kp1]) * cross;
    cz_sum += (z[k] + z[kp1]) * cross;
  }
  if (fabs(cross_sum) > 0.0) {
    const double inv = 1.0 / (3.0 * cross_sum);
    *centroid_r = cr_sum * inv;
    *centroid_z = cz_sum * inv;
    return;
  }
  double r_avg = 0.0;
  double z_avg = 0.0;
  for (int k = 0; k < nverts; ++k) {
    r_avg += r[k];
    z_avg += z[k];
  }
  *centroid_r = r_avg / static_cast<double>(nverts);
  *centroid_z = z_avg / static_cast<double>(nverts);
}

__host__ __device__ void rz_polygon_svec(const double* r,
                                         const double* z,
                                         const int nverts,
                                         const int k,
                                         const double centroid_r,
                                         const double centroid_z,
                                         const double orientation_sign,
                                         double* sr_out,
                                         double* sz_out) {
  const int km1 = (k + nverts - 1) % nverts;
  const int kp1 = (k + 1) % nverts;

  const double trkm1 = r[km1] - centroid_r;
  const double trk = r[k] - centroid_r;
  const double trkp1 = r[kp1] - centroid_r;
  const double tzkm1 = z[km1] - centroid_z;
  const double tzk = z[k] - centroid_z;
  const double tzkp1 = z[kp1] - centroid_z;

  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  const double sr =
      pi_over_three *
      (2.0 * trk * (tzkp1 - tzkm1) + trkp1 * (tzkp1 - tzk) +
       trkm1 * (tzk - tzkm1) + 3.0 * centroid_r * (tzkp1 - tzkm1));
  const double sz = pi_over_three *
                    (trkm1 * (trkm1 + trk) - trkp1 * (trk + trkp1) +
                     3.0 * centroid_r * (trkm1 - trkp1));
  *sr_out = orientation_sign * sr;
  *sz_out = orientation_sign * sz;
}

struct MultiblockIndexing {
  tenryu::core::TopologyScheme topology_scheme =
      tenryu::core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL;
  int n_c = 0;
  int n_b = 0;
  int ntheta = 0;
  int nr_shell = 0;
  int n_cells_core = 0;
  int n_cells_bridge = 0;
  int n_cells_shell = 0;
  int n_cells_total = 0;
  int n_cells_north_fan = 0;
  int n_cells_east_fan = 0;
  int n_cells_south_fan = 0;
  int n_nodes_core = 0;
  int n_nodes_bridge_interior = 0;
  int n_nodes_shell = 0;
  int n_nodes_total = 0;
  bool has_trifan_cap = false;
  int n_cap = 0;
  int n_cells_cap = 0;
  int n_nodes_cap = 0;
  int bridge_cell_offset = 0;
  int north_fan_cell_offset = 0;
  int east_fan_cell_offset = 0;
  int south_fan_cell_offset = 0;
  int shell_cell_offset = 0;
  int bridge_interior_node_offset = 0;
  int north_fan_node_offset = 0;
  int east_fan_node_offset = 0;
  int south_fan_node_offset = 0;
  int shell_node_offset = 0;
  int n_nodes_north_fan_interior = 0;
  int n_nodes_east_fan_interior = 0;
  int n_nodes_south_fan_interior = 0;
  double r_c = 0.0;
  double r_match = 0.0;
  double s_max = 0.0;
  std::vector<double> shell_s_nodes;
};

constexpr int kMultiblockCoreBlock = 0;
constexpr int kMultiblockBridgeBlock = 1;
constexpr int kMultiblockShellBlock = 2;
constexpr int kHalfButterflyNorthFanBlock = 1;
constexpr int kHalfButterflyEastFanBlock = 2;
constexpr int kHalfButterflySouthFanBlock = 3;
constexpr int kHalfButterflyShellBlock = 4;

int boundary_tag(const BoundaryKind kind) {
  return static_cast<int>(kind);
}

MultiblockIndexing make_multiblock_indexing(
    const tenryu::core::Config::MeshConfig& cfg) {
  MultiblockIndexing idx;
  const bool polar_in_box = cfg.logical_mesh_2d == "polar_in_box";
  idx.topology_scheme = cfg.topology_scheme;
  idx.n_c = cfg.multiblock_cart_core_n_c;
  idx.n_b = cfg.multiblock_cart_core_bridge_layers;
  idx.ntheta = 4 * idx.n_c;
  idx.nr_shell = cfg.nr;
  idx.r_c = cfg.multiblock_cart_core_r_c;
  idx.r_match = cfg.multiblock_cart_core_r_match;
  idx.s_max = cfg.spherical_polar_s_max;
  idx.has_trifan_cap = mesh_topo_has_trifan_cap(cfg);
  if (idx.has_trifan_cap) {
    idx.n_cap = mesh_topo_n_cap(cfg);
    idx.n_cells_cap = mesh_topo_n_cells_cap(cfg);
    idx.n_nodes_cap = mesh_topo_n_nodes_cap(cfg);
  }

  TENRYU_ASSERT(idx.n_c > 0, "multiblock N_c must be positive");
  TENRYU_ASSERT(idx.n_b > 0, "multiblock bridge layer count must be positive");
  TENRYU_ASSERT(idx.nr_shell > 0, "multiblock shell radial count must be positive");
  TENRYU_ASSERT(cfg.nz == idx.ntheta,
                "multiblock Mesh.nz must equal 4*Mesh.multiblock_cart_core_n_c");
  TENRYU_ASSERT(std::isfinite(idx.r_c) && idx.r_c > 0.0,
                "multiblock r_c must be finite and positive");
  TENRYU_ASSERT(std::isfinite(idx.r_match) &&
                    std::sqrt(2.0) * idx.r_c < idx.r_match,
                "multiblock r_match must exceed sqrt(2)*r_c");
  TENRYU_ASSERT(std::isfinite(idx.s_max) && idx.r_match < idx.s_max,
                "multiblock s_max must exceed r_match");

  idx.n_cells_core = mesh_topo_n_cells_core(cfg);
  idx.n_cells_bridge = mesh_topo_n_cells_bridge(cfg);
  idx.n_cells_shell = mesh_topo_n_cells_shell(cfg);
  idx.n_cells_total = mesh_topo_n_cells_total(cfg);
  idx.n_cells_north_fan = idx.n_b * idx.n_c;
  idx.n_cells_east_fan = idx.n_b * (2 * idx.n_c);
  idx.n_cells_south_fan = idx.n_b * idx.n_c;
  idx.n_nodes_core = mesh_topo_n_nodes_core(cfg);
  idx.n_nodes_bridge_interior = mesh_topo_n_nodes_bridge_interior(cfg);
  idx.n_nodes_shell = mesh_topo_n_nodes_shell(cfg);
  idx.n_nodes_total = mesh_topo_n_nodes_total(cfg);
  idx.n_nodes_north_fan_interior =
      (idx.n_b - 1) * (idx.n_c + 1);
  idx.n_nodes_east_fan_interior = (idx.n_b - 1) * (2 * idx.n_c);
  idx.n_nodes_south_fan_interior = (idx.n_b - 1) * idx.n_c;
  if (idx.has_trifan_cap) {
    TENRYU_ASSERT(idx.n_cap == idx.n_c,
                  "trifan cap sizing requires N_cap == N_c");
    TENRYU_ASSERT(idx.n_cells_cap == idx.n_cells_core,
                  "trifan cap cells must be the multiblock core count");
    TENRYU_ASSERT(idx.n_nodes_cap == idx.n_nodes_core,
                  "trifan cap nodes must be the multiblock core node count");
  }
  TENRYU_ASSERT(idx.n_cells_total <= std::numeric_limits<int>::max() / 4,
                "multiblock CSR face/corner slot count exceeds int range");
  idx.bridge_cell_offset = idx.n_cells_core;
  idx.north_fan_cell_offset = idx.n_cells_core;
  idx.east_fan_cell_offset =
      idx.north_fan_cell_offset + idx.n_cells_north_fan;
  idx.south_fan_cell_offset =
      idx.east_fan_cell_offset + idx.n_cells_east_fan;
  idx.shell_cell_offset = idx.n_cells_core + idx.n_cells_bridge;
  idx.bridge_interior_node_offset = idx.n_nodes_core;
  idx.north_fan_node_offset = idx.n_nodes_core;
  idx.east_fan_node_offset =
      idx.north_fan_node_offset + idx.n_nodes_north_fan_interior;
  idx.south_fan_node_offset =
      idx.east_fan_node_offset + idx.n_nodes_east_fan_interior;
  idx.shell_node_offset = idx.n_nodes_core + idx.n_nodes_bridge_interior;
  TENRYU_ASSERT(idx.n_cells_north_fan + idx.n_cells_east_fan +
                        idx.n_cells_south_fan ==
                    idx.n_cells_bridge,
                "half-butterfly fan cell counts must match bridge total");
  TENRYU_ASSERT(idx.n_nodes_north_fan_interior +
                        idx.n_nodes_east_fan_interior +
                        idx.n_nodes_south_fan_interior ==
                    idx.n_nodes_bridge_interior,
                "half-butterfly fan owned node counts must match transition total");
  TENRYU_ASSERT(idx.south_fan_cell_offset + idx.n_cells_south_fan ==
                    idx.shell_cell_offset,
                "half-butterfly shell cell offset mismatch");
  TENRYU_ASSERT(idx.south_fan_node_offset +
                        idx.n_nodes_south_fan_interior ==
                    idx.shell_node_offset,
                "half-butterfly shell node offset mismatch");
  if (!cfg.explicit_nodes.empty()) {
    const int expected_node_count =
        polar_in_box
            ? idx.nr_shell + 1 - cfg.morph_rings - cfg.collar_rings
            : idx.nr_shell + 1;
    TENRYU_ASSERT(
        static_cast<int>(cfg.explicit_nodes.size()) == expected_node_count,
        "multiblock shell explicit node count mismatch: got " +
            std::to_string(cfg.explicit_nodes.size()) +
            " nodes, expected " + std::to_string(expected_node_count));
    TENRYU_ASSERT(cfg.explicit_nodes.front() == idx.r_match,
                  "multiblock shell explicit nodes must start exactly at "
                  "r_match");
    if (!polar_in_box) {
      TENRYU_ASSERT(cfg.explicit_nodes.back() == idx.s_max,
                    "multiblock shell explicit nodes must end exactly at "
                    "spherical_polar_s_max");
    }
    for (std::size_t k = 1; k < cfg.explicit_nodes.size(); ++k) {
      TENRYU_ASSERT(cfg.explicit_nodes[k] > cfg.explicit_nodes[k - 1],
                    "multiblock shell explicit nodes must be strictly "
                    "increasing");
    }
    idx.shell_s_nodes = cfg.explicit_nodes;
    if (polar_in_box) {
      const int prefix_node_count =
          static_cast<int>(cfg.explicit_nodes.size());
      const double lock_radius = cfg.explicit_nodes.back();
      TENRYU_ASSERT(static_cast<int>(idx.shell_s_nodes.size()) ==
                        prefix_node_count,
                    "polar_in_box multiblock shell prefix count mismatch");
      TENRYU_ASSERT(idx.shell_s_nodes.back() == lock_radius,
                    "polar_in_box multiblock shell prefix must end exactly at "
                    "the lock radius");
    }
  } else if (!cfg.grid_segments.empty()) {
    TENRYU_ASSERT(!polar_in_box,
                  "polar_in_box multiblock shell requires explicit prefix nodes");
    std::vector<double> s_nodes =
        build_graded_nodes(cfg.grid_segments, cfg.grading, true);
    TENRYU_ASSERT(static_cast<int>(s_nodes.size()) == idx.nr_shell + 1,
                  "multiblock shell graded node count mismatch: got " +
                      std::to_string(s_nodes.size()) + " nodes, expected " +
                      std::to_string(idx.nr_shell + 1));
    const double r_match_tol =
        1.0e-12 * std::max(1.0, std::abs(idx.r_match));
    const double s_max_tol = 1.0e-12 * std::max(1.0, std::abs(idx.s_max));
    TENRYU_ASSERT(std::abs(s_nodes.front() - idx.r_match) <= r_match_tol,
                  "multiblock shell graded nodes must start at r_match");
    TENRYU_ASSERT(std::abs(s_nodes.back() - idx.s_max) <= s_max_tol,
                  "multiblock shell graded nodes must end at "
                  "spherical_polar_s_max");
    for (std::size_t k = 1; k < s_nodes.size(); ++k) {
      TENRYU_ASSERT(s_nodes[k] > s_nodes[k - 1],
                    "multiblock shell nodes must be strictly increasing");
    }
    TENRYU_ASSERT(s_nodes[1] > idx.r_match,
                  "multiblock shell nodes: second node must exceed r_match");
    TENRYU_ASSERT(s_nodes[s_nodes.size() - 2] < idx.s_max,
                  "multiblock shell nodes: second-to-last node must lie "
                  "below s_max");
    s_nodes.front() = idx.r_match;
    s_nodes.back() = idx.s_max;
    double max_ratio = 1.0;
    for (std::size_t k = 2; k < s_nodes.size(); ++k) {
      const double d0 = s_nodes[k - 1] - s_nodes[k - 2];
      const double d1 = s_nodes[k] - s_nodes[k - 1];
      const double ratio = std::max(d1 / d0, d0 / d1);
      max_ratio = std::max(max_ratio, ratio);
    }
    tenryu::core::log_info(
        "[mesh-graded] multiblock shell radial nodes: n=" +
        std::to_string(idx.nr_shell + 1) +
        " s0=" + std::to_string(s_nodes.front()) +
        " s_max=" + std::to_string(s_nodes.back()) +
        " max_adjacent_dr_ratio=" + std::to_string(max_ratio));
    idx.shell_s_nodes = std::move(s_nodes);
  } else {
    TENRYU_ASSERT(!polar_in_box,
                  "polar_in_box multiblock shell requires explicit prefix nodes");
    idx.shell_s_nodes.assign(static_cast<std::size_t>(idx.nr_shell) + 1, 0.0);
    for (int q = 0; q <= idx.nr_shell; ++q) {
      idx.shell_s_nodes[static_cast<std::size_t>(q)] =
          idx.r_match + (idx.s_max - idx.r_match) *
                            static_cast<double>(q) /
                            static_cast<double>(idx.nr_shell);
    }
  }
  return idx;
}

int core_node_id(const MultiblockIndexing& idx, const int i, const int j) {
  TENRYU_ASSERT(i >= 0 && i <= idx.n_c, "core node i out of range");
  TENRYU_ASSERT(j >= 0 && j <= 2 * idx.n_c, "core node j out of range");
  return i * (2 * idx.n_c + 1) + j;
}

int core_cell_id(const MultiblockIndexing& idx, const int i, const int j) {
  TENRYU_ASSERT(i >= 0 && i < idx.n_c, "core cell i out of range");
  TENRYU_ASSERT(j >= 0 && j < 2 * idx.n_c, "core cell j out of range");
  return i * (2 * idx.n_c) + j;
}

int bridge_cell_id(const MultiblockIndexing& idx, const int l, const int k) {
  TENRYU_ASSERT(l >= 0 && l < idx.n_b, "bridge cell layer out of range");
  TENRYU_ASSERT(k >= 0 && k < idx.ntheta, "bridge cell angular index out of range");
  return idx.bridge_cell_offset + l * idx.ntheta + k;
}

int shell_cell_id(const MultiblockIndexing& idx, const int q, const int k) {
  TENRYU_ASSERT(q >= 0 && q < idx.nr_shell, "shell cell radial index out of range");
  TENRYU_ASSERT(k >= 0 && k < idx.ntheta, "shell cell angular index out of range");
  return idx.shell_cell_offset + q * idx.ntheta + k;
}

int bridge_interior_node_id(const MultiblockIndexing& idx,
                            const int l,
                            const int k) {
  TENRYU_ASSERT(l > 0 && l < idx.n_b, "bridge interior layer out of range");
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta, "bridge node angular index out of range");
  return idx.bridge_interior_node_offset + (l - 1) * (idx.ntheta + 1) + k;
}

int shell_node_id(const MultiblockIndexing& idx, const int q, const int k) {
  TENRYU_ASSERT(q >= 0 && q <= idx.nr_shell, "shell node radial index out of range");
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta, "shell node angular index out of range");
  return idx.shell_node_offset + q * (idx.ntheta + 1) + k;
}

bool multiblock_indexing_is_half_butterfly(
    const MultiblockIndexing& idx) {
  return idx.topology_scheme ==
         tenryu::core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK;
}

bool multiblock_indexing_has_trifan_cap(const MultiblockIndexing& idx) {
  return idx.has_trifan_cap;
}

int cap_apex_node_id(const MultiblockIndexing& idx) {
  TENRYU_ASSERT(idx.has_trifan_cap, "cap apex id requires trifan cap indexing");
  TENRYU_ASSERT(idx.n_nodes_cap > 0, "cap node count must be positive");
  return 0;
}

int cap_ring_node_id(const MultiblockIndexing& idx,
                     const int l,
                     const int k) {
  TENRYU_ASSERT(idx.has_trifan_cap, "cap ring id requires trifan cap indexing");
  TENRYU_ASSERT(idx.n_cap > 0, "cap radial count must be positive");
  TENRYU_ASSERT(l >= 1 && l <= idx.n_cap, "cap ring layer out of range");
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta,
                "cap ring angular index out of range");
  const int node = 1 + (l - 1) * (idx.ntheta + 1) + k;
  TENRYU_ASSERT(node >= 0 && node < idx.n_nodes_cap,
                "cap ring node id out of range");
  return node;
}

int cap_cell_id(const MultiblockIndexing& idx, const int l, const int k) {
  TENRYU_ASSERT(idx.has_trifan_cap, "cap cell id requires trifan cap indexing");
  TENRYU_ASSERT(l >= 0 && l < idx.n_cap, "cap cell layer out of range");
  TENRYU_ASSERT(k >= 0 && k < idx.ntheta,
                "cap cell angular index out of range");
  return l * idx.ntheta + k;
}

int fan_angular_cell_count(const MultiblockIndexing& idx,
                           const BlockRole fan) {
  switch (fan) {
    case BlockRole::NORTH_FAN:
    case BlockRole::SOUTH_FAN:
      return idx.n_c;
    case BlockRole::EAST_FAN:
      return 2 * idx.n_c;
    default:
      TENRYU_ASSERT(false, "invalid half-butterfly fan role");
      return 0;
  }
}

int fan_global_angle_node_index(const MultiblockIndexing& idx,
                                const BlockRole fan,
                                const int k) {
  switch (fan) {
    case BlockRole::NORTH_FAN:
      TENRYU_ASSERT(k >= 0 && k <= idx.n_c,
                    "north fan angular node index out of range");
      return k;
    case BlockRole::EAST_FAN:
      TENRYU_ASSERT(k >= 0 && k <= 2 * idx.n_c,
                    "east fan angular node index out of range");
      return idx.n_c + k;
    case BlockRole::SOUTH_FAN:
      TENRYU_ASSERT(k >= 0 && k <= idx.n_c,
                    "south fan angular node index out of range");
      return 3 * idx.n_c + k;
    default:
      TENRYU_ASSERT(false, "invalid half-butterfly fan role");
      return 0;
  }
}

int fan_cell_id(const MultiblockIndexing& idx,
                const BlockRole fan,
                const int l,
                const int k) {
  TENRYU_ASSERT(l >= 0 && l < idx.n_b, "fan cell layer out of range");
  const int n_j = fan_angular_cell_count(idx, fan);
  TENRYU_ASSERT(k >= 0 && k < n_j, "fan cell angular index out of range");
  switch (fan) {
    case BlockRole::NORTH_FAN:
      return idx.north_fan_cell_offset + l * n_j + k;
    case BlockRole::EAST_FAN:
      return idx.east_fan_cell_offset + l * n_j + k;
    case BlockRole::SOUTH_FAN:
      return idx.south_fan_cell_offset + l * n_j + k;
    default:
      TENRYU_ASSERT(false, "invalid half-butterfly fan role");
      return -1;
  }
}

int fan_inner_boundary_node_id(const MultiblockIndexing& idx,
                               const BlockRole fan,
                               const int k) {
  switch (fan) {
    case BlockRole::NORTH_FAN:
      TENRYU_ASSERT(k >= 0 && k <= idx.n_c,
                    "north fan inner node index out of range");
      if (idx.has_trifan_cap) {
        return cap_ring_node_id(idx, idx.n_cap, k);
      }
      return core_node_id(idx, k, 2 * idx.n_c);
    case BlockRole::EAST_FAN:
      TENRYU_ASSERT(k >= 0 && k <= 2 * idx.n_c,
                    "east fan inner node index out of range");
      if (idx.has_trifan_cap) {
        return cap_ring_node_id(idx, idx.n_cap, idx.n_c + k);
      }
      return core_node_id(idx, idx.n_c, 2 * idx.n_c - k);
    case BlockRole::SOUTH_FAN:
      TENRYU_ASSERT(k >= 0 && k <= idx.n_c,
                    "south fan inner node index out of range");
      if (idx.has_trifan_cap) {
        return cap_ring_node_id(idx, idx.n_cap, 3 * idx.n_c + k);
      }
      return core_node_id(idx, idx.n_c - k, 0);
    default:
      TENRYU_ASSERT(false, "invalid half-butterfly fan role");
      return -1;
  }
}

int fan_node_id(const MultiblockIndexing& idx,
                const BlockRole fan,
                const int l,
                const int k) {
  TENRYU_ASSERT(l >= 0 && l <= idx.n_b, "fan node layer out of range");
  const int g = fan_global_angle_node_index(idx, fan, k);
  if (l == 0) {
    return fan_inner_boundary_node_id(idx, fan, k);
  }
  if (l == idx.n_b) {
    return shell_node_id(idx, 0, g);
  }
  switch (fan) {
    case BlockRole::NORTH_FAN:
      return idx.north_fan_node_offset + (l - 1) * (idx.n_c + 1) + k;
    case BlockRole::EAST_FAN:
      if (k == 0) {
        return idx.north_fan_node_offset + (l - 1) * (idx.n_c + 1) +
               idx.n_c;
      }
      return idx.east_fan_node_offset + (l - 1) * (2 * idx.n_c) + (k - 1);
    case BlockRole::SOUTH_FAN:
      if (k == 0) {
        return idx.east_fan_node_offset + (l - 1) * (2 * idx.n_c) +
               (2 * idx.n_c - 1);
      }
      return idx.south_fan_node_offset + (l - 1) * idx.n_c + (k - 1);
    default:
      TENRYU_ASSERT(false, "invalid half-butterfly fan role");
      return -1;
  }
}

RZPoint polar_halfplane_point(const MultiblockIndexing& idx,
                              const double s,
                              const int k) {
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta, "polar angular index out of range");
  if (k == 0) {
    return {0.0, s};
  }
  if (k == idx.ntheta) {
    return {0.0, -s};
  }
  const double theta =
      static_cast<double>(k) * spherical_wedge_detail::kPi /
      static_cast<double>(idx.ntheta);
  double r = s * std::sin(theta);
  double z = s * std::cos(theta);
  if (2 * k == idx.ntheta) {
    z = 0.0;
  }
  return {r, z};
}

RZPoint core_boundary_point(const MultiblockIndexing& idx, const int k) {
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta, "core seam angular index out of range");
  const double h = idx.r_c / static_cast<double>(idx.n_c);
  if (k <= idx.n_c) {
    return {static_cast<double>(k) * h, idx.r_c};
  }
  if (k <= 3 * idx.n_c) {
    return {idx.r_c, idx.r_c - static_cast<double>(k - idx.n_c) * h};
  }
  return {idx.r_c - static_cast<double>(k - 3 * idx.n_c) * h, -idx.r_c};
}

RZPoint cap_ring_point(const MultiblockIndexing& idx,
                       const int l,
                       const int k) {
  TENRYU_ASSERT(idx.has_trifan_cap, "cap point requires trifan cap indexing");
  TENRYU_ASSERT(l >= 1 && l <= idx.n_cap, "cap point layer out of range");
  const double rho = static_cast<double>(l) / static_cast<double>(idx.n_cap);
  const RZPoint outer = core_boundary_point(idx, k);
  return {rho * outer.r, rho * outer.z};
}

int core_boundary_node_id(const MultiblockIndexing& idx, const int k) {
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta, "core seam angular index out of range");
  if (k <= idx.n_c) {
    return core_node_id(idx, k, 2 * idx.n_c);
  }
  if (k <= 3 * idx.n_c) {
    return core_node_id(idx, idx.n_c, 3 * idx.n_c - k);
  }
  return core_node_id(idx, 4 * idx.n_c - k, 0);
}

RZPoint rz_lerp(const RZPoint a, const RZPoint b, const double t) {
  return {(1.0 - t) * a.r + t * b.r, (1.0 - t) * a.z + t * b.z};
}

RZPoint rz_add(const RZPoint a, const RZPoint b) {
  return {a.r + b.r, a.z + b.z};
}

RZPoint rz_sub(const RZPoint a, const RZPoint b) {
  return {a.r - b.r, a.z - b.z};
}

RZPoint bridge_point_at_blend(const MultiblockIndexing& idx,
                              const double w,
                              const int k) {
  const RZPoint inner = core_boundary_point(idx, k);
  const RZPoint outer = polar_halfplane_point(idx, idx.r_match, k);
  return {(1.0 - w) * inner.r + w * outer.r,
          (1.0 - w) * inner.z + w * outer.z};
}

RZPoint bridge_point(const MultiblockIndexing& idx, const int l, const int k) {
  TENRYU_ASSERT(l >= 0 && l <= idx.n_b, "bridge node layer out of range");
  const double eta = static_cast<double>(l) / static_cast<double>(idx.n_b);
  const double w = 3.0 * eta * eta - 2.0 * eta * eta * eta;
  return bridge_point_at_blend(idx, w, k);
}

RZPoint half_butterfly_fan_inner_point(const MultiblockIndexing& idx,
                                       const BlockRole fan,
                                       const int k) {
  const int n_j = fan_angular_cell_count(idx, fan);
  TENRYU_ASSERT(k >= 0 && k <= n_j, "fan inner node index out of range");
  const double eta = static_cast<double>(k) / static_cast<double>(n_j);
  switch (fan) {
    case BlockRole::NORTH_FAN:
      return {idx.r_c * eta, idx.r_c};
    case BlockRole::EAST_FAN:
      return {idx.r_c, idx.r_c * (1.0 - 2.0 * eta)};
    case BlockRole::SOUTH_FAN:
      return {idx.r_c * (1.0 - eta), -idx.r_c};
    default:
      TENRYU_ASSERT(false, "invalid half-butterfly fan role");
      return {};
  }
}

RZPoint half_butterfly_fan_outer_point(const MultiblockIndexing& idx,
                                       const BlockRole fan,
                                       const int k) {
  return polar_halfplane_point(
      idx, idx.r_match, fan_global_angle_node_index(idx, fan, k));
}

RZPoint half_butterfly_fan_side_point(const MultiblockIndexing& idx,
                                      const BlockRole fan,
                                      const int k_side,
                                      const int l) {
  TENRYU_ASSERT(l >= 0 && l <= idx.n_b, "fan side layer out of range");
  const int n_j = fan_angular_cell_count(idx, fan);
  TENRYU_ASSERT(k_side == 0 || k_side == n_j,
                "fan side must be an angular edge");
  const double rho = static_cast<double>(l) / static_cast<double>(idx.n_b);
  return rz_lerp(half_butterfly_fan_inner_point(idx, fan, k_side),
                 half_butterfly_fan_outer_point(idx, fan, k_side), rho);
}

RZPoint half_butterfly_fan_tfi_point(const MultiblockIndexing& idx,
                                     const BlockRole fan,
                                     const int l,
                                     const int k) {
  TENRYU_ASSERT(l >= 0 && l <= idx.n_b, "fan node layer out of range");
  const int n_j = fan_angular_cell_count(idx, fan);
  TENRYU_ASSERT(k >= 0 && k <= n_j, "fan node angular index out of range");
  if (l == 0) {
    return half_butterfly_fan_inner_point(idx, fan, k);
  }
  if (l == idx.n_b) {
    return half_butterfly_fan_outer_point(idx, fan, k);
  }
  if (k == 0 || k == n_j) {
    return half_butterfly_fan_side_point(idx, fan, k, l);
  }

  const double rho = static_cast<double>(l) / static_cast<double>(idx.n_b);
  const double eta = static_cast<double>(k) / static_cast<double>(n_j);
  const RZPoint inner = half_butterfly_fan_inner_point(idx, fan, k);
  const RZPoint outer = half_butterfly_fan_outer_point(idx, fan, k);
  const RZPoint side_0 = half_butterfly_fan_side_point(idx, fan, 0, l);
  const RZPoint side_1 = half_butterfly_fan_side_point(idx, fan, n_j, l);
  const RZPoint inner_0 = half_butterfly_fan_inner_point(idx, fan, 0);
  const RZPoint inner_1 = half_butterfly_fan_inner_point(idx, fan, n_j);
  const RZPoint outer_0 = half_butterfly_fan_outer_point(idx, fan, 0);
  const RZPoint outer_1 = half_butterfly_fan_outer_point(idx, fan, n_j);
  const RZPoint cap_blend = rz_lerp(inner, outer, rho);
  const RZPoint side_blend = rz_lerp(side_0, side_1, eta);
  const RZPoint bilinear_inner = rz_lerp(inner_0, inner_1, eta);
  const RZPoint bilinear_outer = rz_lerp(outer_0, outer_1, eta);
  const RZPoint bilinear = rz_lerp(bilinear_inner, bilinear_outer, rho);
  return rz_sub(rz_add(cap_blend, side_blend), bilinear);
}

int fan_grid_node_index(const int n_j, const int l, const int k) {
  TENRYU_ASSERT(n_j > 0, "fan grid angular count must be positive");
  TENRYU_ASSERT(l >= 0, "fan grid layer out of range");
  TENRYU_ASSERT(k >= 0 && k <= n_j, "fan grid k out of range");
  return l * (n_j + 1) + k;
}

void smooth_half_butterfly_fan_grid(const MultiblockIndexing& idx,
                                    const BlockRole fan,
                                    const int sweeps,
                                    const double omega,
                                    std::vector<RZPoint>& grid) {
  if (sweeps <= 0) {
    return;
  }
  TENRYU_ASSERT(std::isfinite(omega) && omega > 0.0 && omega < 2.0,
                "fan elliptic omega must be finite and in (0, 2)");
  const int n_j = fan_angular_cell_count(idx, fan);
  TENRYU_ASSERT(grid.size() ==
                    static_cast<std::size_t>((idx.n_b + 1) * (n_j + 1)),
                "fan grid size mismatch");
  std::vector<RZPoint> next = grid;
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    next = grid;
    for (int l = 1; l < idx.n_b; ++l) {
      for (int k = 1; k < n_j; ++k) {
        const int north_idx = fan_grid_node_index(n_j, l - 1, k);
        const int south_idx = fan_grid_node_index(n_j, l + 1, k);
        const int west_idx = fan_grid_node_index(n_j, l, k - 1);
        const int east_idx = fan_grid_node_index(n_j, l, k + 1);
        const RZPoint north = grid[static_cast<std::size_t>(north_idx)];
        const RZPoint south = grid[static_cast<std::size_t>(south_idx)];
        const RZPoint west = grid[static_cast<std::size_t>(west_idx)];
        const RZPoint east = grid[static_cast<std::size_t>(east_idx)];
        const RZPoint average{0.25 * (north.r + south.r + west.r + east.r),
                              0.25 * (north.z + south.z + west.z + east.z)};
        const int node = fan_grid_node_index(n_j, l, k);
        next[static_cast<std::size_t>(node)] =
            rz_lerp(grid[static_cast<std::size_t>(node)], average, omega);
      }
    }
    grid.swap(next);
  }
}

std::vector<RZPoint> build_half_butterfly_fan_grid(
    const MultiblockIndexing& idx,
    const tenryu::core::Config::MeshConfig& cfg,
    const BlockRole fan) {
  const int n_j = fan_angular_cell_count(idx, fan);
  std::vector<RZPoint> grid(
      static_cast<std::size_t>((idx.n_b + 1) * (n_j + 1)));
  for (int l = 0; l <= idx.n_b; ++l) {
    for (int k = 0; k <= n_j; ++k) {
      grid[static_cast<std::size_t>(fan_grid_node_index(n_j, l, k))] =
          half_butterfly_fan_tfi_point(idx, fan, l, k);
    }
  }
  smooth_half_butterfly_fan_grid(idx,
                                 fan,
                                 cfg.multiblock_bridge_elliptic_sweeps,
                                 cfg.multiblock_bridge_elliptic_omega,
                                 grid);
  return grid;
}

int bridge_grid_node_index(const MultiblockIndexing& idx,
                           const int l,
                           const int k) {
  TENRYU_ASSERT(l >= 0 && l <= idx.n_b, "bridge grid layer out of range");
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta, "bridge grid k out of range");
  return l * (idx.ntheta + 1) + k;
}

double rounded_cap_extent(const MultiblockIndexing& idx, const double cap_p) {
  TENRYU_ASSERT(std::isfinite(cap_p) && cap_p > 2.0,
                "rounded cap exponent must be finite and > 2");
  const double a = idx.r_c * std::pow(2.0, 1.0 / cap_p);
  TENRYU_ASSERT(std::isfinite(a) && a < idx.r_match,
                "rounded cap extent must stay inside r_match");
  return a;
}

RZPoint rounded_cap_point(const MultiblockIndexing& idx,
                          const double cap_p,
                          const int k) {
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta,
                "rounded cap angular index out of range");
  const double a = rounded_cap_extent(idx, cap_p);
  const double theta =
      static_cast<double>(k) * spherical_wedge_detail::kPi /
      static_cast<double>(idx.ntheta);
  const double sin_theta = std::abs(std::sin(theta));
  const double cos_theta = std::abs(std::cos(theta));
  const double denom =
      std::pow(std::pow(sin_theta, cap_p) + std::pow(cos_theta, cap_p),
               1.0 / cap_p);
  TENRYU_ASSERT(std::isfinite(denom) && denom > 0.0,
                "rounded cap denominator must be positive");
  return polar_halfplane_point(idx, a / denom, k);
}

std::pair<int, int> bridge_half_butterfly_region(const MultiblockIndexing& idx,
                                                 const int k) {
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta,
                "half-butterfly k out of range");
  if (k <= idx.n_c) {
    return {0, idx.n_c};
  }
  if (k <= 3 * idx.n_c) {
    return {idx.n_c, 3 * idx.n_c};
  }
  return {3 * idx.n_c, idx.ntheta};
}

RZPoint rounded_half_butterfly_tfi_point(const MultiblockIndexing& idx,
                                         const double cap_p,
                                         const int l,
                                         const int k) {
  TENRYU_ASSERT(l > 0 && l < idx.n_b,
                "rounded bridge TFI layer must be interior");
  const auto [k0, k1] = bridge_half_butterfly_region(idx, k);
  TENRYU_ASSERT(k1 > k0, "half-butterfly region must have positive width");
  const double u =
      static_cast<double>(k - k0) / static_cast<double>(k1 - k0);
  const double eta = static_cast<double>(l) / static_cast<double>(idx.n_b);

  const RZPoint bottom = rounded_cap_point(idx, cap_p, k);
  const RZPoint top = polar_halfplane_point(idx, idx.r_match, k);
  const RZPoint left_bottom = rounded_cap_point(idx, cap_p, k0);
  const RZPoint right_bottom = rounded_cap_point(idx, cap_p, k1);
  const RZPoint left_top = polar_halfplane_point(idx, idx.r_match, k0);
  const RZPoint right_top = polar_halfplane_point(idx, idx.r_match, k1);
  const RZPoint left = rz_lerp(left_bottom, left_top, eta);
  const RZPoint right = rz_lerp(right_bottom, right_top, eta);
  const RZPoint bottom_top_blend = rz_lerp(bottom, top, eta);
  const RZPoint side_blend = rz_lerp(left, right, u);
  const RZPoint bilinear_bottom = rz_lerp(left_bottom, right_bottom, u);
  const RZPoint bilinear_top = rz_lerp(left_top, right_top, u);
  const RZPoint bilinear = rz_lerp(bilinear_bottom, bilinear_top, eta);
  return rz_sub(rz_add(side_blend, bottom_top_blend), bilinear);
}

void smooth_rounded_bridge_grid(const MultiblockIndexing& idx,
                                const int sweeps,
                                const double omega,
                                std::vector<RZPoint>& grid) {
  if (sweeps <= 0) {
    return;
  }
  TENRYU_ASSERT(std::isfinite(omega) && omega > 0.0 && omega < 2.0,
                "bridge elliptic omega must be finite and in (0, 2)");
  std::vector<RZPoint> next = grid;
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    next = grid;
    for (int l = 1; l < idx.n_b; ++l) {
      for (int k = 1; k < idx.ntheta; ++k) {
        const int north_idx = bridge_grid_node_index(idx, l - 1, k);
        const int south_idx = bridge_grid_node_index(idx, l + 1, k);
        const int west_idx = bridge_grid_node_index(idx, l, k - 1);
        const int east_idx = bridge_grid_node_index(idx, l, k + 1);
        const RZPoint north = grid[static_cast<std::size_t>(north_idx)];
        const RZPoint south = grid[static_cast<std::size_t>(south_idx)];
        const RZPoint west = grid[static_cast<std::size_t>(west_idx)];
        const RZPoint east = grid[static_cast<std::size_t>(east_idx)];
        const RZPoint average{0.25 * (north.r + south.r + west.r + east.r),
                              0.25 * (north.z + south.z + west.z + east.z)};
        const int node = bridge_grid_node_index(idx, l, k);
        next[static_cast<std::size_t>(node)] =
            rz_lerp(grid[static_cast<std::size_t>(node)], average, omega);
      }
    }
    grid.swap(next);
  }
}

std::vector<RZPoint> build_rounded_bridge_grid(
    const MultiblockIndexing& idx,
    const tenryu::core::Config::MeshConfig& cfg) {
  const bool rounded_core_seam =
      cfg.multiblock_transition_scheme ==
      tenryu::core::MultiblockTransitionScheme::ROUNDED_CORE_SEAM;
  std::vector<RZPoint> grid(
      static_cast<std::size_t>((idx.n_b + 1) * (idx.ntheta + 1)));
  for (int l = 0; l <= idx.n_b; ++l) {
    for (int k = 0; k <= idx.ntheta; ++k) {
      RZPoint p{};
      if (l == 0) {
        p = rounded_core_seam ? rounded_cap_point(idx, cfg.multiblock_cap_p, k)
                              : core_boundary_point(idx, k);
      } else if (l == idx.n_b) {
        p = polar_halfplane_point(idx, idx.r_match, k);
      } else {
        p = rounded_half_butterfly_tfi_point(
            idx, cfg.multiblock_cap_p, l, k);
      }
      grid[static_cast<std::size_t>(bridge_grid_node_index(idx, l, k))] = p;
    }
  }
  smooth_rounded_bridge_grid(idx,
                             cfg.multiblock_bridge_elliptic_sweeps,
                             cfg.multiblock_bridge_elliptic_omega,
                             grid);
  return grid;
}

void smooth_rounded_core_grid(const MultiblockIndexing& idx,
                              const int sweeps,
                              const double omega,
                              std::vector<RZPoint>& grid) {
  if (sweeps <= 0) {
    return;
  }
  TENRYU_ASSERT(std::isfinite(omega) && omega > 0.0 && omega < 2.0,
                "core elliptic omega must be finite and in (0, 2)");
  std::vector<RZPoint> next = grid;
  for (int sweep = 0; sweep < sweeps; ++sweep) {
    next = grid;
    for (int i = 1; i < idx.n_c; ++i) {
      for (int j = 1; j < 2 * idx.n_c; ++j) {
        const RZPoint west =
            grid[static_cast<std::size_t>(core_node_id(idx, i - 1, j))];
        const RZPoint east =
            grid[static_cast<std::size_t>(core_node_id(idx, i + 1, j))];
        const RZPoint south =
            grid[static_cast<std::size_t>(core_node_id(idx, i, j - 1))];
        const RZPoint north =
            grid[static_cast<std::size_t>(core_node_id(idx, i, j + 1))];
        const RZPoint average{0.25 * (west.r + east.r + south.r + north.r),
                              0.25 * (west.z + east.z + south.z + north.z)};
        const int node = core_node_id(idx, i, j);
        next[static_cast<std::size_t>(node)] =
            rz_lerp(grid[static_cast<std::size_t>(node)], average, omega);
      }
    }
    grid.swap(next);
  }
}

std::vector<RZPoint> build_rounded_core_grid(
    const MultiblockIndexing& idx,
    const tenryu::core::Config::MeshConfig& cfg) {
  const int j_top = 2 * idx.n_c;
  const double cap_a = rounded_cap_extent(idx, cfg.multiblock_cap_p);
  std::vector<RZPoint> grid(static_cast<std::size_t>(idx.n_nodes_core));

  for (int j = 0; j <= j_top; ++j) {
    grid[static_cast<std::size_t>(core_node_id(idx, 0, j))] =
        RZPoint{0.0,
                (static_cast<double>(j - idx.n_c) /
                 static_cast<double>(idx.n_c)) *
                    cap_a};
  }
  for (int i = 0; i <= idx.n_c; ++i) {
    grid[static_cast<std::size_t>(core_node_id(idx, i, j_top))] =
        rounded_cap_point(idx, cfg.multiblock_cap_p, i);
    grid[static_cast<std::size_t>(core_node_id(idx, i, 0))] =
        rounded_cap_point(idx, cfg.multiblock_cap_p, 4 * idx.n_c - i);
  }
  for (int j = 0; j <= j_top; ++j) {
    grid[static_cast<std::size_t>(core_node_id(idx, idx.n_c, j))] =
        rounded_cap_point(idx, cfg.multiblock_cap_p, 3 * idx.n_c - j);
  }

  const RZPoint bottom_left =
      grid[static_cast<std::size_t>(core_node_id(idx, 0, 0))];
  const RZPoint bottom_right =
      grid[static_cast<std::size_t>(core_node_id(idx, idx.n_c, 0))];
  const RZPoint top_left =
      grid[static_cast<std::size_t>(core_node_id(idx, 0, j_top))];
  const RZPoint top_right =
      grid[static_cast<std::size_t>(core_node_id(idx, idx.n_c, j_top))];
  for (int i = 1; i < idx.n_c; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(idx.n_c);
    for (int j = 1; j < j_top; ++j) {
      const double v = static_cast<double>(j) / static_cast<double>(j_top);
      const RZPoint left =
          grid[static_cast<std::size_t>(core_node_id(idx, 0, j))];
      const RZPoint right =
          grid[static_cast<std::size_t>(core_node_id(idx, idx.n_c, j))];
      const RZPoint bottom =
          grid[static_cast<std::size_t>(core_node_id(idx, i, 0))];
      const RZPoint top =
          grid[static_cast<std::size_t>(core_node_id(idx, i, j_top))];
      const RZPoint side_blend = rz_lerp(left, right, u);
      const RZPoint cap_blend = rz_lerp(bottom, top, v);
      const RZPoint bilinear_bottom =
          rz_lerp(bottom_left, bottom_right, u);
      const RZPoint bilinear_top = rz_lerp(top_left, top_right, u);
      const RZPoint bilinear = rz_lerp(bilinear_bottom, bilinear_top, v);
      grid[static_cast<std::size_t>(core_node_id(idx, i, j))] =
          rz_sub(rz_add(side_blend, cap_blend), bilinear);
    }
  }

  smooth_rounded_core_grid(idx,
                           cfg.multiblock_bridge_elliptic_sweeps,
                           cfg.multiblock_bridge_elliptic_omega,
                           grid);
  return grid;
}

int bridge_node_id(const MultiblockIndexing& idx, const int l, const int k) {
  TENRYU_ASSERT(l >= 0 && l <= idx.n_b, "bridge node layer out of range");
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta, "bridge node angular index out of range");
  if (l == 0) {
    return core_boundary_node_id(idx, k);
  }
  if (l == idx.n_b) {
    return shell_node_id(idx, 0, k);
  }
  return bridge_interior_node_id(idx, l, k);
}

RZPoint shell_point(const MultiblockIndexing& idx, const int q, const int k) {
  TENRYU_ASSERT(q >= 0 &&
                    static_cast<std::size_t>(q) < idx.shell_s_nodes.size(),
                "shell node radial index out of range");
  const double s = idx.shell_s_nodes[static_cast<std::size_t>(q)];
  return polar_halfplane_point(idx, s, k);
}

double seam_coord_tol(const double coord) {
  return std::max(1.0e-14 * std::max(1.0, std::fabs(coord)), 1.0e-30);
}

void assert_seam_node_matches(const std::vector<double>& node_r,
                              const std::vector<double>& node_z,
                              const int node,
                              const RZPoint expected,
                              const std::string& location) {
  TENRYU_ASSERT(node >= 0 &&
                    static_cast<std::size_t>(node) < node_r.size() &&
                    node_r.size() == node_z.size(),
                "seam node index out of range at " + location);
  const double r = node_r[static_cast<std::size_t>(node)];
  const double z = node_z[static_cast<std::size_t>(node)];
  TENRYU_ASSERT(std::fabs(r - expected.r) < seam_coord_tol(expected.r) &&
                    std::fabs(z - expected.z) < seam_coord_tol(expected.z),
                "seam node coordinate mismatch at " + location);
}

int append_node(std::vector<double>& node_r,
                std::vector<double>& node_z,
                const RZPoint p) {
  TENRYU_ASSERT(std::isfinite(p.r) && std::isfinite(p.z),
                "multiblock node coordinate must be finite");
  const int id = static_cast<int>(node_r.size());
  node_r.push_back(p.r);
  node_z.push_back(p.z);
  return id;
}

void set_multiblock_node_flags(MeshTopology& topo,
                               const MultiblockIndexing& idx,
                               const std::vector<double>& node_r) {
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == topo.n_nodes,
                "multiblock node flag setup requires all nodes");
  topo.node_flags.assign(static_cast<std::size_t>(topo.n_nodes), NODE_NONE);
  const double axis_tol = core::axis_tolerance(idx.s_max);
  for (int n = 0; n < topo.n_nodes; ++n) {
    std::uint8_t flags = NODE_NONE;
    if (std::fabs(node_r[static_cast<std::size_t>(n)]) <= axis_tol) {
      flags |= NODE_AXIS;
      flags |= NODE_POLE_AXIS;
      flags |= NODE_BOUNDARY;
    }
    topo.node_flags[static_cast<std::size_t>(n)] = flags;
  }
  if (idx.has_trifan_cap) {
    topo.node_flags[static_cast<std::size_t>(cap_apex_node_id(idx))] =
        NODE_CENTER | NODE_AXIS | NODE_BOUNDARY;
  }
  for (int k = 0; k <= idx.ntheta; ++k) {
    const int n = shell_node_id(idx, idx.nr_shell, k);
    topo.node_flags[static_cast<std::size_t>(n)] |=
        NODE_BOUNDARY | NODE_OUTER_PHYSICAL_BOUNDARY;
  }
}

void set_csr_offsets(std::vector<int>& offsets,
                     const int n_rows,
                     const int stride) {
  offsets.resize(static_cast<std::size_t>(n_rows + 1));
  for (int c = 0; c <= n_rows; ++c) {
    offsets[static_cast<std::size_t>(c)] = stride * c;
  }
}

void set_cell_nodes(MultiBlockTopology& mb,
                    const int c,
                    const std::array<int, 4>& nodes,
                    const int stride) {
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
  for (int f = 0; f < stride; ++f) {
    mb.cell_node_csr_indices[static_cast<std::size_t>(off + f)] =
        (f < 4) ? nodes[static_cast<std::size_t>(f)] : 0;
  }
}

void set_cell_faces(MultiBlockTopology& mb,
                    const int c,
                    const std::array<int, 4>& adj,
                    const std::array<int, 4>& tags,
                    const int stride) {
  const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
  for (int f = 0; f < stride; ++f) {
    mb.face_adj_csr_indices[static_cast<std::size_t>(off + f)] =
        (f < 4) ? adj[static_cast<std::size_t>(f)] : -1;
    mb.face_bc_tags[static_cast<std::size_t>(off + f)] =
        (f < 4) ? tags[static_cast<std::size_t>(f)]
                : boundary_tag(BoundaryKind::Interior);
  }
}

void validate_multiblock_metadata(const MultiBlockTopology& mb,
                                  const MultiblockIndexing& idx) {
  TENRYU_ASSERT(static_cast<int>(mb.blocks.size()) == mb.block_count,
                "multiblock block table size mismatch");
  int cell_count_sum = 0;
  int owned_node_count_sum = 0;
  for (const BlockInfo& block : mb.blocks) {
    TENRYU_ASSERT(block.n_i_cells > 0,
                  "multiblock block i cell count must be positive");
    TENRYU_ASSERT(block.n_j_cells > 0,
                  "multiblock block j cell count must be positive");
    TENRYU_ASSERT(static_cast<long long>(block.n_i_cells) *
                          static_cast<long long>(block.n_j_cells) ==
                      static_cast<long long>(block.cell_count),
                  "multiblock block logical dimensions must match cell count");
    TENRYU_ASSERT(block.cell_begin >= 0,
                  "multiblock block cell begin must be non-negative");
    TENRYU_ASSERT(block.cell_count >= 0,
                  "multiblock block cell count must be non-negative");
    TENRYU_ASSERT(block.cell_begin + block.cell_count <= idx.n_cells_total,
                  "multiblock block cell range exceeds total cell count");
    TENRYU_ASSERT(block.owned_node_begin >= 0,
                  "multiblock block owned node begin must be non-negative");
    TENRYU_ASSERT(block.owned_node_count >= 0,
                  "multiblock block owned node count must be non-negative");
    TENRYU_ASSERT(block.owned_node_begin + block.owned_node_count <=
                      idx.n_nodes_total,
                  "multiblock block owned node range exceeds total node count");
    cell_count_sum += block.cell_count;
    owned_node_count_sum += block.owned_node_count;
  }
  TENRYU_ASSERT(cell_count_sum == idx.n_cells_total,
                "multiblock block cell counts must sum to total cells");
  TENRYU_ASSERT(owned_node_count_sum == idx.n_nodes_total,
                "multiblock block owned node counts must sum to total nodes");
  int seam_index_end = 0;
  for (const SeamInfo& seam : mb.seams) {
    TENRYU_ASSERT(seam.block_a >= 0 && seam.block_a < mb.block_count,
                  "multiblock seam block_a out of range");
    TENRYU_ASSERT(seam.block_b >= 0 && seam.block_b < mb.block_count,
                  "multiblock seam block_b out of range");
    TENRYU_ASSERT(seam.orientation == 1 || seam.orientation == -1,
                  "multiblock seam orientation must be +/-1");
    TENRYU_ASSERT(seam.index_begin == seam_index_end,
                  "multiblock seam index range must be contiguous");
    TENRYU_ASSERT(seam.index_count > 0,
                  "multiblock seam index count must be positive");
    seam_index_end += seam.index_count;
  }
}

void initialize_three_block_metadata(MultiBlockTopology& mb,
                                     const MultiblockIndexing& idx) {
  mb.blocks = {{
      {BlockRole::CENTRAL_CORE,
       idx.n_c,
       2 * idx.n_c,
       0,
       idx.n_cells_core,
       0,
       idx.n_nodes_core},
      {BlockRole::BRIDGE,
       idx.n_b,
       idx.ntheta,
       idx.bridge_cell_offset,
       idx.n_cells_bridge,
       idx.bridge_interior_node_offset,
       idx.n_nodes_bridge_interior},
      {BlockRole::POLAR_SHELL,
       idx.nr_shell,
       idx.ntheta,
       idx.shell_cell_offset,
       idx.n_cells_shell,
       idx.shell_node_offset,
       idx.n_nodes_shell},
  }};
  mb.seams = {{
      {kMultiblockCoreBlock,
       BlockSide::COMPOSITE_OUTER,
       kMultiblockBridgeBlock,
       BlockSide::I_MINUS,
       1,
       0,
       idx.ntheta},
      {kMultiblockBridgeBlock,
       BlockSide::I_PLUS,
       kMultiblockShellBlock,
       BlockSide::I_MINUS,
       1,
       idx.ntheta,
       idx.ntheta},
  }};

  validate_multiblock_metadata(mb, idx);
}

void initialize_half_butterfly_metadata(MultiBlockTopology& mb,
                                        const MultiblockIndexing& idx) {
  mb.blocks = {{
      {BlockRole::CENTRAL_CORE,
       idx.n_c,
       2 * idx.n_c,
       0,
       idx.n_cells_core,
       0,
       idx.n_nodes_core},
      {BlockRole::NORTH_FAN,
       idx.n_b,
       idx.n_c,
       idx.north_fan_cell_offset,
       idx.n_cells_north_fan,
       idx.north_fan_node_offset,
       idx.n_nodes_north_fan_interior},
      {BlockRole::EAST_FAN,
       idx.n_b,
       2 * idx.n_c,
       idx.east_fan_cell_offset,
       idx.n_cells_east_fan,
       idx.east_fan_node_offset,
       idx.n_nodes_east_fan_interior},
      {BlockRole::SOUTH_FAN,
       idx.n_b,
       idx.n_c,
       idx.south_fan_cell_offset,
       idx.n_cells_south_fan,
       idx.south_fan_node_offset,
       idx.n_nodes_south_fan_interior},
      {BlockRole::POLAR_SHELL,
       idx.nr_shell,
       idx.ntheta,
       idx.shell_cell_offset,
       idx.n_cells_shell,
       idx.shell_node_offset,
       idx.n_nodes_shell},
  }};

  int seam_begin = 0;
  const auto add_seam = [&](const int block_a,
                            const BlockSide side_a,
                            const int block_b,
                            const BlockSide side_b,
                            const int orientation,
                            const int count) {
    mb.seams.push_back(
        {block_a, side_a, block_b, side_b, orientation, seam_begin, count});
    seam_begin += count;
  };
  add_seam(kMultiblockCoreBlock, BlockSide::J_PLUS,
           kHalfButterflyNorthFanBlock, BlockSide::I_MINUS, 1, idx.n_c);
  add_seam(kMultiblockCoreBlock, BlockSide::I_PLUS,
           kHalfButterflyEastFanBlock, BlockSide::I_MINUS, -1, 2 * idx.n_c);
  add_seam(kMultiblockCoreBlock, BlockSide::J_MINUS,
           kHalfButterflySouthFanBlock, BlockSide::I_MINUS, -1, idx.n_c);
  add_seam(kHalfButterflyNorthFanBlock, BlockSide::J_PLUS,
           kHalfButterflyEastFanBlock, BlockSide::J_MINUS, 1, idx.n_b);
  add_seam(kHalfButterflyEastFanBlock, BlockSide::J_PLUS,
           kHalfButterflySouthFanBlock, BlockSide::J_MINUS, 1, idx.n_b);
  add_seam(kHalfButterflyNorthFanBlock, BlockSide::I_PLUS,
           kHalfButterflyShellBlock, BlockSide::I_MINUS, 1, idx.n_c);
  add_seam(kHalfButterflyEastFanBlock, BlockSide::I_PLUS,
           kHalfButterflyShellBlock, BlockSide::I_MINUS, 1, 2 * idx.n_c);
  add_seam(kHalfButterflySouthFanBlock, BlockSide::I_PLUS,
           kHalfButterflyShellBlock, BlockSide::I_MINUS, 1, idx.n_c);

  validate_multiblock_metadata(mb, idx);
}

void initialize_trifan_cap_metadata(MultiBlockTopology& mb,
                                    const MultiblockIndexing& idx) {
  TENRYU_ASSERT(idx.has_trifan_cap,
                "trifan cap metadata requires trifan cap indexing");
  mb.blocks = {{
      {BlockRole::CENTRAL_CORE,
       idx.n_cap,
       idx.ntheta,
       0,
       idx.n_cells_cap,
       0,
       idx.n_nodes_cap},
      {BlockRole::NORTH_FAN,
       idx.n_b,
       idx.n_c,
       idx.north_fan_cell_offset,
       idx.n_cells_north_fan,
       idx.north_fan_node_offset,
       idx.n_nodes_north_fan_interior},
      {BlockRole::EAST_FAN,
       idx.n_b,
       2 * idx.n_c,
       idx.east_fan_cell_offset,
       idx.n_cells_east_fan,
       idx.east_fan_node_offset,
       idx.n_nodes_east_fan_interior},
      {BlockRole::SOUTH_FAN,
       idx.n_b,
       idx.n_c,
       idx.south_fan_cell_offset,
       idx.n_cells_south_fan,
       idx.south_fan_node_offset,
       idx.n_nodes_south_fan_interior},
      {BlockRole::POLAR_SHELL,
       idx.nr_shell,
       idx.ntheta,
       idx.shell_cell_offset,
       idx.n_cells_shell,
       idx.shell_node_offset,
       idx.n_nodes_shell},
  }};

  int seam_begin = 0;
  const auto add_seam = [&](const int block_a,
                            const BlockSide side_a,
                            const int block_b,
                            const BlockSide side_b,
                            const int orientation,
                            const int count) {
    mb.seams.push_back(
        {block_a, side_a, block_b, side_b, orientation, seam_begin, count});
    seam_begin += count;
  };
  add_seam(kMultiblockCoreBlock, BlockSide::J_PLUS,
           kHalfButterflyNorthFanBlock, BlockSide::I_MINUS, 1, idx.n_c);
  add_seam(kMultiblockCoreBlock, BlockSide::I_PLUS,
           kHalfButterflyEastFanBlock, BlockSide::I_MINUS, -1, 2 * idx.n_c);
  add_seam(kMultiblockCoreBlock, BlockSide::J_MINUS,
           kHalfButterflySouthFanBlock, BlockSide::I_MINUS, -1, idx.n_c);
  add_seam(kHalfButterflyNorthFanBlock, BlockSide::J_PLUS,
           kHalfButterflyEastFanBlock, BlockSide::J_MINUS, 1, idx.n_b);
  add_seam(kHalfButterflyEastFanBlock, BlockSide::J_PLUS,
           kHalfButterflySouthFanBlock, BlockSide::J_MINUS, 1, idx.n_b);
  add_seam(kHalfButterflyNorthFanBlock, BlockSide::I_PLUS,
           kHalfButterflyShellBlock, BlockSide::I_MINUS, 1, idx.n_c);
  add_seam(kHalfButterflyEastFanBlock, BlockSide::I_PLUS,
           kHalfButterflyShellBlock, BlockSide::I_MINUS, 1, 2 * idx.n_c);
  add_seam(kHalfButterflySouthFanBlock, BlockSide::I_PLUS,
           kHalfButterflyShellBlock, BlockSide::I_MINUS, 1, idx.n_c);

  validate_multiblock_metadata(mb, idx);
}

int core_neighbor_for_bridge_inner_face(const MultiblockIndexing& idx,
                                        const int k) {
  TENRYU_ASSERT(k >= 0 && k < idx.ntheta,
                "bridge inner face angular index out of range");
  if (k < idx.n_c) {
    return core_cell_id(idx, k, 2 * idx.n_c - 1);
  }
  if (k < 3 * idx.n_c) {
    return core_cell_id(idx, idx.n_c - 1, 3 * idx.n_c - k - 1);
  }
  return core_cell_id(idx, 4 * idx.n_c - k - 1, 0);
}

void assert_bilateral_face_adjacency(
    const MultiBlockTopology& mb,
    const int stride,
    const std::vector<std::uint8_t>* cell_nverts = nullptr) {
  const int n_cells = static_cast<int>(mb.face_adj_csr_offsets.size()) - 1;
  TENRYU_ASSERT(cell_nverts == nullptr || cell_nverts->empty() ||
                    cell_nverts->size() == static_cast<std::size_t>(n_cells),
                "multiblock face validation requires empty or per-cell cell_nverts");
  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
    const int next = mb.face_adj_csr_offsets[static_cast<std::size_t>(c + 1)];
    TENRYU_ASSERT(next - off == stride,
                  "multiblock face CSR width must match mesh stride");
    const int active_nverts = (cell_nverts == nullptr || cell_nverts->empty())
                                  ? kMeshTopoCellStorageSlots
                                  : mesh_topo_cell_active_nverts(
                                        *cell_nverts, c);
    for (int p = off; p < next; ++p) {
      const int local = p - off;
      const int n = mb.face_adj_csr_indices[static_cast<std::size_t>(p)];
      if (!mesh_topo_local_face_is_active(active_nverts, local)) {
        TENRYU_ASSERT(n < 0,
                      "inactive multiblock face adjacency slot must be empty");
        continue;
      }
      if (n < 0) {
        continue;
      }
      TENRYU_ASSERT(n < n_cells, "multiblock neighbor cell out of range");
      const int noff = mb.face_adj_csr_offsets[static_cast<std::size_t>(n)];
      const int nnext = mb.face_adj_csr_offsets[static_cast<std::size_t>(n + 1)];
      bool found_inverse = false;
      const int neighbor_active_nverts =
          (cell_nverts == nullptr || cell_nverts->empty())
              ? kMeshTopoCellStorageSlots
              : mesh_topo_cell_active_nverts(*cell_nverts, n);
      for (int q = noff; q < nnext; ++q) {
        const int neighbor_local = q - noff;
        if (!mesh_topo_local_face_is_active(neighbor_active_nverts,
                                            neighbor_local)) {
          continue;
        }
        if (mb.face_adj_csr_indices[static_cast<std::size_t>(q)] == c) {
          found_inverse = true;
          break;
        }
      }
      TENRYU_ASSERT(found_inverse, "multiblock face adjacency is not bilateral");
    }
  }
}

void assert_half_butterfly_shared_node_maps(const MultiblockIndexing& idx) {
  TENRYU_ASSERT(multiblock_indexing_is_half_butterfly(idx),
                "half-butterfly shared-node checks require half-butterfly indexing");
  for (int k = 0; k <= idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::NORTH_FAN, 0, k) ==
                      core_node_id(idx, k, 2 * idx.n_c),
                  "north fan inner seam must resolve to central top nodes");
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::SOUTH_FAN, 0, k) ==
                      core_node_id(idx, idx.n_c - k, 0),
                  "south fan inner seam must resolve to central bottom nodes");
  }
  for (int k = 0; k <= 2 * idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::EAST_FAN, 0, k) ==
                      core_node_id(idx, idx.n_c, 2 * idx.n_c - k),
                  "east fan inner seam must resolve to central right nodes");
  }
  for (int l = 0; l <= idx.n_b; ++l) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::NORTH_FAN, l, idx.n_c) ==
                      fan_node_id(idx, BlockRole::EAST_FAN, l, 0),
                  "north-east fan diagonal seam must share node ids");
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::EAST_FAN, l, 2 * idx.n_c) ==
                      fan_node_id(idx, BlockRole::SOUTH_FAN, l, 0),
                  "east-south fan diagonal seam must share node ids");
  }
  for (int k = 0; k <= idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::NORTH_FAN, idx.n_b, k) ==
                      shell_node_id(idx, 0, k),
                  "north fan outer seam must resolve to shell inner ring");
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::SOUTH_FAN, idx.n_b, k) ==
                      shell_node_id(idx, 0, 3 * idx.n_c + k),
                  "south fan outer seam must resolve to shell inner ring");
  }
  for (int k = 0; k <= 2 * idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::EAST_FAN, idx.n_b, k) ==
                      shell_node_id(idx, 0, idx.n_c + k),
                  "east fan outer seam must resolve to shell inner ring");
  }
}

void assert_trifan_cap_shared_node_maps(const MultiblockIndexing& idx);

int cell_node_incidence_count(const MultiBlockTopology& mb,
                              const int node_id) {
  int count = 0;
  for (int c = 0; c + 1 < static_cast<int>(mb.cell_node_csr_offsets.size());
       ++c) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int next =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(c + 1)];
    for (int p = off; p < next; ++p) {
      if (mb.cell_node_csr_indices[static_cast<std::size_t>(p)] == node_id) {
        ++count;
      }
    }
  }
  return count;
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
  mb.d_unique_face_local_a.assign(unique_local_a.begin(), unique_local_a.end());
  mb.d_unique_face_local_b.assign(unique_local_b.begin(), unique_local_b.end());
  mb.d_unique_face_bc_tag.assign(unique_bc_tag.begin(), unique_bc_tag.end());
  mb.d_boundary_face_cell.assign(boundary_cell.begin(), boundary_cell.end());
  mb.d_boundary_face_local.assign(boundary_local.begin(), boundary_local.end());
  mb.d_boundary_face_bc_tag.assign(boundary_bc_tag.begin(),
                                   boundary_bc_tag.end());
}

MultiBlockTopology build_half_butterfly_topology(
    const MultiblockIndexing& idx,
    const int stride) {
  MultiBlockTopology mb;
  mb.block_count = 5;
  mb.n_cells_core = idx.n_cells_core;
  mb.n_cells_bridge = idx.n_cells_bridge;
  mb.n_cells_shell = idx.n_cells_shell;
  mb.n_nodes_core = idx.n_nodes_core;
  mb.n_nodes_bridge_interior = idx.n_nodes_bridge_interior;
  mb.n_nodes_shell = idx.n_nodes_shell;
  initialize_half_butterfly_metadata(mb, idx);
  assert_half_butterfly_shared_node_maps(idx);

  mb.cell_block_id.assign(static_cast<std::size_t>(idx.n_cells_total), -1);
  mb.cell_id_stable.assign(static_cast<std::size_t>(idx.n_cells_total), -1);
  mb.cell_orientation_sign.assign(static_cast<std::size_t>(idx.n_cells_total),
                                  0);
  set_csr_offsets(mb.cell_node_csr_offsets, idx.n_cells_total, stride);
  set_csr_offsets(mb.face_adj_csr_offsets, idx.n_cells_total, stride);
  const std::size_t n_face_slots =
      static_cast<std::size_t>(idx.n_cells_total) *
      static_cast<std::size_t>(stride);
  mb.cell_node_csr_indices.assign(n_face_slots, -1);
  mb.face_adj_csr_indices.assign(n_face_slots, -1);
  mb.face_bc_tags.assign(n_face_slots,
                         boundary_tag(BoundaryKind::Interior));

  for (int i = 0; i < idx.n_c; ++i) {
    for (int j = 0; j < 2 * idx.n_c; ++j) {
      const int c = core_cell_id(idx, i, j);
      mb.cell_block_id[static_cast<std::size_t>(c)] = kMultiblockCoreBlock;
      mb.cell_id_stable[static_cast<std::size_t>(c)] = c;
      mb.cell_orientation_sign[static_cast<std::size_t>(c)] = 1;
      set_cell_nodes(mb, c,
                     {{
                         core_node_id(idx, i, j),
                         core_node_id(idx, i + 1, j),
                         core_node_id(idx, i + 1, j + 1),
                         core_node_id(idx, i, j + 1),
                     }},
                     stride);
      std::array<int, 4> adj = {{-1, -1, -1, -1}};
      std::array<int, 4> tags = {{
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
      }};
      if (i > 0) {
        adj[0] = core_cell_id(idx, i - 1, j);
      } else {
        tags[0] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      if (i + 1 < idx.n_c) {
        adj[1] = core_cell_id(idx, i + 1, j);
      } else {
        adj[1] = fan_cell_id(idx, BlockRole::EAST_FAN, 0,
                             2 * idx.n_c - j - 1);
        tags[1] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
      }
      if (j > 0) {
        adj[2] = core_cell_id(idx, i, j - 1);
      } else {
        adj[2] = fan_cell_id(idx, BlockRole::SOUTH_FAN, 0,
                             idx.n_c - i - 1);
        tags[2] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
      }
      if (j + 1 < 2 * idx.n_c) {
        adj[3] = core_cell_id(idx, i, j + 1);
      } else {
        adj[3] = fan_cell_id(idx, BlockRole::NORTH_FAN, 0, i);
        tags[3] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
      }
      set_cell_faces(mb, c, adj, tags, stride);
    }
  }

  const auto set_fan_cells = [&](const BlockRole fan, const int block_id) {
    const int n_j = fan_angular_cell_count(idx, fan);
    for (int l = 0; l < idx.n_b; ++l) {
      for (int k = 0; k < n_j; ++k) {
        const int c = fan_cell_id(idx, fan, l, k);
        mb.cell_block_id[static_cast<std::size_t>(c)] = block_id;
        mb.cell_id_stable[static_cast<std::size_t>(c)] = c;
        mb.cell_orientation_sign[static_cast<std::size_t>(c)] = -1;
        set_cell_nodes(mb, c,
                       {{
                           fan_node_id(idx, fan, l, k),
                           fan_node_id(idx, fan, l + 1, k),
                           fan_node_id(idx, fan, l + 1, k + 1),
                           fan_node_id(idx, fan, l, k + 1),
                       }},
                       stride);
        std::array<int, 4> adj = {{-1, -1, -1, -1}};
        std::array<int, 4> tags = {{
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
        }};
        if (l > 0) {
          adj[0] = fan_cell_id(idx, fan, l - 1, k);
        } else {
          switch (fan) {
            case BlockRole::NORTH_FAN:
              adj[0] = core_cell_id(idx, k, 2 * idx.n_c - 1);
              break;
            case BlockRole::EAST_FAN:
              adj[0] = core_cell_id(idx, idx.n_c - 1,
                                    2 * idx.n_c - k - 1);
              break;
            case BlockRole::SOUTH_FAN:
              adj[0] = core_cell_id(idx, idx.n_c - k - 1, 0);
              break;
            default:
              TENRYU_ASSERT(false, "invalid half-butterfly fan role");
          }
          tags[0] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
        }
        if (l + 1 < idx.n_b) {
          adj[1] = fan_cell_id(idx, fan, l + 1, k);
        } else {
          adj[1] =
              shell_cell_id(idx, 0, fan_global_angle_node_index(idx, fan, k));
          tags[1] = boundary_tag(BoundaryKind::SEAM_BRIDGE_SHELL);
        }
        if (k > 0) {
          adj[2] = fan_cell_id(idx, fan, l, k - 1);
        } else {
          switch (fan) {
            case BlockRole::NORTH_FAN:
              tags[2] = boundary_tag(BoundaryKind::AXIS_SHELL);
              break;
            case BlockRole::EAST_FAN:
              adj[2] = fan_cell_id(idx, BlockRole::NORTH_FAN, l,
                                   idx.n_c - 1);
              tags[2] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
              break;
            case BlockRole::SOUTH_FAN:
              adj[2] = fan_cell_id(idx, BlockRole::EAST_FAN, l,
                                   2 * idx.n_c - 1);
              tags[2] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
              break;
            default:
              TENRYU_ASSERT(false, "invalid half-butterfly fan role");
          }
        }
        if (k + 1 < n_j) {
          adj[3] = fan_cell_id(idx, fan, l, k + 1);
        } else {
          switch (fan) {
            case BlockRole::NORTH_FAN:
              adj[3] = fan_cell_id(idx, BlockRole::EAST_FAN, l, 0);
              tags[3] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
              break;
            case BlockRole::EAST_FAN:
              adj[3] = fan_cell_id(idx, BlockRole::SOUTH_FAN, l, 0);
              tags[3] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
              break;
            case BlockRole::SOUTH_FAN:
              tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
              break;
            default:
              TENRYU_ASSERT(false, "invalid half-butterfly fan role");
          }
        }
        set_cell_faces(mb, c, adj, tags, stride);
      }
    }
  };
  set_fan_cells(BlockRole::NORTH_FAN, kHalfButterflyNorthFanBlock);
  set_fan_cells(BlockRole::EAST_FAN, kHalfButterflyEastFanBlock);
  set_fan_cells(BlockRole::SOUTH_FAN, kHalfButterflySouthFanBlock);

  for (int q = 0; q < idx.nr_shell; ++q) {
    for (int k = 0; k < idx.ntheta; ++k) {
      const int c = shell_cell_id(idx, q, k);
      mb.cell_block_id[static_cast<std::size_t>(c)] =
          kHalfButterflyShellBlock;
      mb.cell_id_stable[static_cast<std::size_t>(c)] = c;
      mb.cell_orientation_sign[static_cast<std::size_t>(c)] = -1;
      set_cell_nodes(mb, c,
                     {{
                         shell_node_id(idx, q, k),
                         shell_node_id(idx, q + 1, k),
                         shell_node_id(idx, q + 1, k + 1),
                         shell_node_id(idx, q, k + 1),
                     }},
                     stride);
      std::array<int, 4> adj = {{-1, -1, -1, -1}};
      std::array<int, 4> tags = {{
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
      }};
      if (q > 0) {
        adj[0] = shell_cell_id(idx, q - 1, k);
      } else {
        if (k < idx.n_c) {
          adj[0] = fan_cell_id(idx, BlockRole::NORTH_FAN, idx.n_b - 1, k);
        } else if (k < 3 * idx.n_c) {
          adj[0] =
              fan_cell_id(idx, BlockRole::EAST_FAN, idx.n_b - 1, k - idx.n_c);
        } else {
          adj[0] = fan_cell_id(idx, BlockRole::SOUTH_FAN, idx.n_b - 1,
                               k - 3 * idx.n_c);
        }
        tags[0] = boundary_tag(BoundaryKind::SEAM_BRIDGE_SHELL);
      }
      if (q + 1 < idx.nr_shell) {
        adj[1] = shell_cell_id(idx, q + 1, k);
      } else {
        tags[1] = boundary_tag(BoundaryKind::SphericalOuterFree);
      }
      if (k > 0) {
        adj[2] = shell_cell_id(idx, q, k - 1);
      } else {
        tags[2] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      if (k + 1 < idx.ntheta) {
        adj[3] = shell_cell_id(idx, q, k + 1);
      } else {
        tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      set_cell_faces(mb, c, adj, tags, stride);
    }
  }

  TENRYU_ASSERT(cell_node_incidence_count(
                    mb, core_node_id(idx, idx.n_c, 2 * idx.n_c)) == 3,
                "half-butterfly north-east central vertex must have valence 3");
  TENRYU_ASSERT(cell_node_incidence_count(
                    mb, core_node_id(idx, idx.n_c, 0)) == 3,
                "half-butterfly east-south central vertex must have valence 3");

  for (int c = 0; c < idx.n_cells_total; ++c) {
    TENRYU_ASSERT(mb.cell_block_id[static_cast<std::size_t>(c)] >= 0,
                  "half-butterfly cell block id was not initialized");
    TENRYU_ASSERT(mb.cell_id_stable[static_cast<std::size_t>(c)] == c,
                  "half-butterfly stable cell id must be block-major");
    TENRYU_ASSERT(mb.cell_orientation_sign[static_cast<std::size_t>(c)] != 0,
                  "half-butterfly cell orientation sign was not initialized");
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    for (int q = 0; q < 4; ++q) {
      const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(off + q)];
      TENRYU_ASSERT(node >= 0 && node < idx.n_nodes_total,
                    "half-butterfly cell node id out of range");
    }
  }
  assert_bilateral_face_adjacency(mb, stride);
  return mb;
}

MultiBlockTopology build_trifan_cap_topology(
    const MultiblockIndexing& idx,
    const int stride,
    const std::vector<std::uint8_t>* cell_nverts = nullptr) {
  TENRYU_ASSERT(idx.has_trifan_cap,
                "trifan cap topology requires cap indexing");
  MultiBlockTopology mb;
  mb.block_count = 5;
  mb.n_cells_core = idx.n_cells_core;
  mb.n_cells_bridge = idx.n_cells_bridge;
  mb.n_cells_shell = idx.n_cells_shell;
  mb.n_nodes_core = idx.n_nodes_core;
  mb.n_nodes_bridge_interior = idx.n_nodes_bridge_interior;
  mb.n_nodes_shell = idx.n_nodes_shell;
  mb.has_trifan_cap = true;
  mb.n_cap = idx.n_cap;
  mb.n_cells_cap = idx.n_cells_cap;
  mb.n_nodes_cap = idx.n_nodes_cap;
  initialize_trifan_cap_metadata(mb, idx);
  assert_trifan_cap_shared_node_maps(idx);
  // Cross-check the public mesh_topo_trifan_fan_node_id mirror against the
  // canonical file-local indexing for every (fan, layer, angular) tuple so
  // the two can never silently diverge.
  for (const BlockRole fan :
       {BlockRole::NORTH_FAN, BlockRole::EAST_FAN, BlockRole::SOUTH_FAN}) {
    const int fan_n_j = (fan == BlockRole::EAST_FAN) ? 2 * idx.n_c : idx.n_c;
    for (int l = 0; l <= idx.n_b; ++l) {
      for (int k = 0; k <= fan_n_j; ++k) {
        TENRYU_ASSERT(mesh_topo_trifan_fan_node_id(mb, fan, l, k) ==
                          fan_node_id(idx, fan, l, k),
                      "public trifan fan node accessor diverges from the "
                      "canonical mesh indexing");
      }
    }
  }

  mb.cell_block_id.assign(static_cast<std::size_t>(idx.n_cells_total), -1);
  mb.cell_id_stable.assign(static_cast<std::size_t>(idx.n_cells_total), -1);
  mb.cell_orientation_sign.assign(static_cast<std::size_t>(idx.n_cells_total),
                                  0);
  set_csr_offsets(mb.cell_node_csr_offsets, idx.n_cells_total, stride);
  set_csr_offsets(mb.face_adj_csr_offsets, idx.n_cells_total, stride);
  const std::size_t n_face_slots =
      static_cast<std::size_t>(idx.n_cells_total) *
      static_cast<std::size_t>(stride);
  mb.cell_node_csr_indices.assign(n_face_slots, -1);
  mb.face_adj_csr_indices.assign(n_face_slots, -1);
  mb.face_bc_tags.assign(n_face_slots,
                         boundary_tag(BoundaryKind::Interior));

  const auto fan_neighbor_for_cap_outer = [&](const int k) {
    TENRYU_ASSERT(k >= 0 && k < idx.ntheta,
                  "cap outer face angular index out of range");
    if (k < idx.n_c) {
      return fan_cell_id(idx, BlockRole::NORTH_FAN, 0, k);
    }
    if (k < 3 * idx.n_c) {
      return fan_cell_id(idx, BlockRole::EAST_FAN, 0, k - idx.n_c);
    }
    return fan_cell_id(idx, BlockRole::SOUTH_FAN, 0, k - 3 * idx.n_c);
  };

  for (int l = 0; l < idx.n_cap; ++l) {
    for (int k = 0; k < idx.ntheta; ++k) {
      const int c = cap_cell_id(idx, l, k);
      mb.cell_block_id[static_cast<std::size_t>(c)] = kMultiblockCoreBlock;
      mb.cell_id_stable[static_cast<std::size_t>(c)] = c;
      mb.cell_orientation_sign[static_cast<std::size_t>(c)] = -1;
      if (l == 0) {
        set_cell_nodes(mb, c,
                       {{
                           cap_apex_node_id(idx),
                           cap_ring_node_id(idx, 1, k),
                           cap_ring_node_id(idx, 1, k + 1),
                           0,
                       }},
                       stride);
      } else {
        set_cell_nodes(mb, c,
                       {{
                           cap_ring_node_id(idx, l, k),
                           cap_ring_node_id(idx, l + 1, k),
                           cap_ring_node_id(idx, l + 1, k + 1),
                           cap_ring_node_id(idx, l, k + 1),
                       }},
                       stride);
      }

      std::array<int, 4> adj = {{-1, -1, -1, -1}};
      std::array<int, 4> tags = {{
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
      }};
      if (l == 0) {
        adj[1] = cap_cell_id(idx, 1, k);
        if (k > 0) {
          adj[2] = cap_cell_id(idx, 0, k - 1);
        } else {
          tags[2] = boundary_tag(BoundaryKind::AXIS_SHELL);
        }
        if (k + 1 < idx.ntheta) {
          adj[3] = cap_cell_id(idx, 0, k + 1);
        } else {
          tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
        }
      } else {
        adj[0] = cap_cell_id(idx, l - 1, k);
        if (l + 1 < idx.n_cap) {
          adj[1] = cap_cell_id(idx, l + 1, k);
        } else {
          adj[1] = fan_neighbor_for_cap_outer(k);
          tags[1] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
        }
        if (k > 0) {
          adj[2] = cap_cell_id(idx, l, k - 1);
        } else {
          tags[2] = boundary_tag(BoundaryKind::AXIS_SHELL);
        }
        if (k + 1 < idx.ntheta) {
          adj[3] = cap_cell_id(idx, l, k + 1);
        } else {
          tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
        }
      }
      set_cell_faces(mb, c, adj, tags, stride);
    }
  }

  const auto set_fan_cells = [&](const BlockRole fan, const int block_id) {
    const int n_j = fan_angular_cell_count(idx, fan);
    for (int l = 0; l < idx.n_b; ++l) {
      for (int k = 0; k < n_j; ++k) {
        const int c = fan_cell_id(idx, fan, l, k);
        mb.cell_block_id[static_cast<std::size_t>(c)] = block_id;
        mb.cell_id_stable[static_cast<std::size_t>(c)] = c;
        mb.cell_orientation_sign[static_cast<std::size_t>(c)] = -1;
        set_cell_nodes(mb, c,
                       {{
                           fan_node_id(idx, fan, l, k),
                           fan_node_id(idx, fan, l + 1, k),
                           fan_node_id(idx, fan, l + 1, k + 1),
                           fan_node_id(idx, fan, l, k + 1),
                       }},
                       stride);
        std::array<int, 4> adj = {{-1, -1, -1, -1}};
        std::array<int, 4> tags = {{
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
            boundary_tag(BoundaryKind::Interior),
        }};
        if (l > 0) {
          adj[0] = fan_cell_id(idx, fan, l - 1, k);
        } else {
          adj[0] =
              cap_cell_id(idx, idx.n_cap - 1,
                          fan_global_angle_node_index(idx, fan, k));
          tags[0] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
        }
        if (l + 1 < idx.n_b) {
          adj[1] = fan_cell_id(idx, fan, l + 1, k);
        } else {
          adj[1] =
              shell_cell_id(idx, 0, fan_global_angle_node_index(idx, fan, k));
          tags[1] = boundary_tag(BoundaryKind::SEAM_BRIDGE_SHELL);
        }
        if (k > 0) {
          adj[2] = fan_cell_id(idx, fan, l, k - 1);
        } else {
          switch (fan) {
            case BlockRole::NORTH_FAN:
              tags[2] = boundary_tag(BoundaryKind::AXIS_SHELL);
              break;
            case BlockRole::EAST_FAN:
              adj[2] = fan_cell_id(idx, BlockRole::NORTH_FAN, l,
                                   idx.n_c - 1);
              tags[2] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
              break;
            case BlockRole::SOUTH_FAN:
              adj[2] = fan_cell_id(idx, BlockRole::EAST_FAN, l,
                                   2 * idx.n_c - 1);
              tags[2] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
              break;
            default:
              TENRYU_ASSERT(false, "invalid half-butterfly fan role");
          }
        }
        if (k + 1 < n_j) {
          adj[3] = fan_cell_id(idx, fan, l, k + 1);
        } else {
          switch (fan) {
            case BlockRole::NORTH_FAN:
              adj[3] = fan_cell_id(idx, BlockRole::EAST_FAN, l, 0);
              tags[3] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
              break;
            case BlockRole::EAST_FAN:
              adj[3] = fan_cell_id(idx, BlockRole::SOUTH_FAN, l, 0);
              tags[3] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
              break;
            case BlockRole::SOUTH_FAN:
              tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
              break;
            default:
              TENRYU_ASSERT(false, "invalid half-butterfly fan role");
          }
        }
        set_cell_faces(mb, c, adj, tags, stride);
      }
    }
  };
  set_fan_cells(BlockRole::NORTH_FAN, kHalfButterflyNorthFanBlock);
  set_fan_cells(BlockRole::EAST_FAN, kHalfButterflyEastFanBlock);
  set_fan_cells(BlockRole::SOUTH_FAN, kHalfButterflySouthFanBlock);

  for (int q = 0; q < idx.nr_shell; ++q) {
    for (int k = 0; k < idx.ntheta; ++k) {
      const int c = shell_cell_id(idx, q, k);
      mb.cell_block_id[static_cast<std::size_t>(c)] =
          kHalfButterflyShellBlock;
      mb.cell_id_stable[static_cast<std::size_t>(c)] = c;
      mb.cell_orientation_sign[static_cast<std::size_t>(c)] = -1;
      set_cell_nodes(mb, c,
                     {{
                         shell_node_id(idx, q, k),
                         shell_node_id(idx, q + 1, k),
                         shell_node_id(idx, q + 1, k + 1),
                         shell_node_id(idx, q, k + 1),
                     }},
                     stride);
      std::array<int, 4> adj = {{-1, -1, -1, -1}};
      std::array<int, 4> tags = {{
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
      }};
      if (q > 0) {
        adj[0] = shell_cell_id(idx, q - 1, k);
      } else {
        if (k < idx.n_c) {
          adj[0] = fan_cell_id(idx, BlockRole::NORTH_FAN, idx.n_b - 1, k);
        } else if (k < 3 * idx.n_c) {
          adj[0] =
              fan_cell_id(idx, BlockRole::EAST_FAN, idx.n_b - 1, k - idx.n_c);
        } else {
          adj[0] = fan_cell_id(idx, BlockRole::SOUTH_FAN, idx.n_b - 1,
                               k - 3 * idx.n_c);
        }
        tags[0] = boundary_tag(BoundaryKind::SEAM_BRIDGE_SHELL);
      }
      if (q + 1 < idx.nr_shell) {
        adj[1] = shell_cell_id(idx, q + 1, k);
      } else {
        tags[1] = boundary_tag(BoundaryKind::SphericalOuterFree);
      }
      if (k > 0) {
        adj[2] = shell_cell_id(idx, q, k - 1);
      } else {
        tags[2] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      if (k + 1 < idx.ntheta) {
        adj[3] = shell_cell_id(idx, q, k + 1);
      } else {
        tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      set_cell_faces(mb, c, adj, tags, stride);
    }
  }

  for (int c = 0; c < idx.n_cells_total; ++c) {
    TENRYU_ASSERT(mb.cell_block_id[static_cast<std::size_t>(c)] >= 0,
                  "trifan cap cell block id was not initialized");
    TENRYU_ASSERT(mb.cell_id_stable[static_cast<std::size_t>(c)] == c,
                  "trifan cap stable cell id must be block-major");
    TENRYU_ASSERT(mb.cell_orientation_sign[static_cast<std::size_t>(c)] != 0,
                  "trifan cap cell orientation sign was not initialized");
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int active_nverts =
        (cell_nverts == nullptr || cell_nverts->empty())
            ? kMeshTopoCellStorageSlots
            : mesh_topo_cell_active_nverts(*cell_nverts, c);
    for (int q = 0; q < active_nverts; ++q) {
      const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(off + q)];
      TENRYU_ASSERT(node >= 0 && node < idx.n_nodes_total,
                    "trifan cap cell node id out of range");
    }
  }
  assert_bilateral_face_adjacency(mb, stride, cell_nverts);
  return mb;
}

MultiBlockTopology build_multiblock_topology(
    const MultiblockIndexing& idx,
    const int stride,
    const std::vector<std::uint8_t>* cell_nverts = nullptr) {
  if (multiblock_indexing_is_half_butterfly(idx)) {
    return build_half_butterfly_topology(idx, stride);
  }
  if (multiblock_indexing_has_trifan_cap(idx)) {
    return build_trifan_cap_topology(idx, stride, cell_nverts);
  }

  MultiBlockTopology mb;
  mb.block_count = 3;
  mb.n_cells_core = idx.n_cells_core;
  mb.n_cells_bridge = idx.n_cells_bridge;
  mb.n_cells_shell = idx.n_cells_shell;
  mb.n_nodes_core = idx.n_nodes_core;
  mb.n_nodes_bridge_interior = idx.n_nodes_bridge_interior;
  mb.n_nodes_shell = idx.n_nodes_shell;
  initialize_three_block_metadata(mb, idx);
  mb.cell_block_id.assign(static_cast<std::size_t>(idx.n_cells_total), -1);
  mb.cell_id_stable.assign(static_cast<std::size_t>(idx.n_cells_total), -1);
  mb.cell_orientation_sign.assign(static_cast<std::size_t>(idx.n_cells_total),
                                  0);
  set_csr_offsets(mb.cell_node_csr_offsets, idx.n_cells_total, stride);
  set_csr_offsets(mb.face_adj_csr_offsets, idx.n_cells_total, stride);
  const std::size_t n_face_slots =
      static_cast<std::size_t>(idx.n_cells_total) *
      static_cast<std::size_t>(stride);
  mb.cell_node_csr_indices.assign(n_face_slots, -1);
  mb.face_adj_csr_indices.assign(n_face_slots, -1);
  mb.face_bc_tags.assign(n_face_slots,
                         boundary_tag(BoundaryKind::Interior));

  for (int i = 0; i < idx.n_c; ++i) {
    for (int j = 0; j < 2 * idx.n_c; ++j) {
      const int c = core_cell_id(idx, i, j);
      mb.cell_block_id[static_cast<std::size_t>(c)] = kMultiblockCoreBlock;
      mb.cell_id_stable[static_cast<std::size_t>(c)] = c;
      mb.cell_orientation_sign[static_cast<std::size_t>(c)] = 1;
      set_cell_nodes(mb, c,
                     {{
                         core_node_id(idx, i, j),
                         core_node_id(idx, i + 1, j),
                         core_node_id(idx, i + 1, j + 1),
                         core_node_id(idx, i, j + 1),
                     }},
                     stride);

      std::array<int, 4> adj = {{-1, -1, -1, -1}};
      std::array<int, 4> tags = {{
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
      }};
      if (i > 0) {
        adj[0] = core_cell_id(idx, i - 1, j);
      } else {
        tags[0] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      if (i + 1 < idx.n_c) {
        adj[1] = core_cell_id(idx, i + 1, j);
      } else {
        adj[1] = bridge_cell_id(idx, 0, 3 * idx.n_c - j - 1);
        tags[1] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
      }
      if (j > 0) {
        adj[2] = core_cell_id(idx, i, j - 1);
      } else {
        adj[2] = bridge_cell_id(idx, 0, 4 * idx.n_c - i - 1);
        tags[2] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
      }
      if (j + 1 < 2 * idx.n_c) {
        adj[3] = core_cell_id(idx, i, j + 1);
      } else {
        adj[3] = bridge_cell_id(idx, 0, i);
        tags[3] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
      }
      set_cell_faces(mb, c, adj, tags, stride);
    }
  }

  for (int l = 0; l < idx.n_b; ++l) {
    for (int k = 0; k < idx.ntheta; ++k) {
      const int c = bridge_cell_id(idx, l, k);
      mb.cell_block_id[static_cast<std::size_t>(c)] = kMultiblockBridgeBlock;
      mb.cell_id_stable[static_cast<std::size_t>(c)] = c;
      mb.cell_orientation_sign[static_cast<std::size_t>(c)] = -1;
      set_cell_nodes(mb, c,
                     {{
                         bridge_node_id(idx, l, k),
                         bridge_node_id(idx, l + 1, k),
                         bridge_node_id(idx, l + 1, k + 1),
                         bridge_node_id(idx, l, k + 1),
                     }},
                     stride);

      std::array<int, 4> adj = {{-1, -1, -1, -1}};
      std::array<int, 4> tags = {{
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
      }};
      if (l > 0) {
        adj[0] = bridge_cell_id(idx, l - 1, k);
      } else {
        adj[0] = core_neighbor_for_bridge_inner_face(idx, k);
        tags[0] = boundary_tag(BoundaryKind::SEAM_CORE_BRIDGE);
      }
      if (l + 1 < idx.n_b) {
        adj[1] = bridge_cell_id(idx, l + 1, k);
      } else {
        adj[1] = shell_cell_id(idx, 0, k);
        tags[1] = boundary_tag(BoundaryKind::SEAM_BRIDGE_SHELL);
      }
      if (k > 0) {
        adj[2] = bridge_cell_id(idx, l, k - 1);
      } else {
        tags[2] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      if (k + 1 < idx.ntheta) {
        adj[3] = bridge_cell_id(idx, l, k + 1);
      } else {
        tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      set_cell_faces(mb, c, adj, tags, stride);
    }
  }

  for (int q = 0; q < idx.nr_shell; ++q) {
    for (int k = 0; k < idx.ntheta; ++k) {
      const int c = shell_cell_id(idx, q, k);
      mb.cell_block_id[static_cast<std::size_t>(c)] = kMultiblockShellBlock;
      mb.cell_id_stable[static_cast<std::size_t>(c)] = c;
      mb.cell_orientation_sign[static_cast<std::size_t>(c)] = -1;
      set_cell_nodes(mb, c,
                     {{
                         shell_node_id(idx, q, k),
                         shell_node_id(idx, q + 1, k),
                         shell_node_id(idx, q + 1, k + 1),
                         shell_node_id(idx, q, k + 1),
                     }},
                     stride);

      std::array<int, 4> adj = {{-1, -1, -1, -1}};
      std::array<int, 4> tags = {{
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
          boundary_tag(BoundaryKind::Interior),
      }};
      if (q > 0) {
        adj[0] = shell_cell_id(idx, q - 1, k);
      } else {
        adj[0] = bridge_cell_id(idx, idx.n_b - 1, k);
        tags[0] = boundary_tag(BoundaryKind::SEAM_BRIDGE_SHELL);
      }
      if (q + 1 < idx.nr_shell) {
        adj[1] = shell_cell_id(idx, q + 1, k);
      } else {
        tags[1] = boundary_tag(BoundaryKind::SphericalOuterFree);
      }
      if (k > 0) {
        adj[2] = shell_cell_id(idx, q, k - 1);
      } else {
        tags[2] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      if (k + 1 < idx.ntheta) {
        adj[3] = shell_cell_id(idx, q, k + 1);
      } else {
        tags[3] = boundary_tag(BoundaryKind::AXIS_SHELL);
      }
      set_cell_faces(mb, c, adj, tags, stride);
    }
  }

  TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                    static_cast<std::size_t>(idx.n_cells_total),
                "multiblock cell orientation sign size mismatch");
  for (int c = 0; c < idx.n_cells_total; ++c) {
    TENRYU_ASSERT(mb.cell_block_id[static_cast<std::size_t>(c)] >= 0,
                  "multiblock cell block id was not initialized");
    TENRYU_ASSERT(mb.cell_id_stable[static_cast<std::size_t>(c)] == c,
                  "multiblock stable cell id must be block-major");
    TENRYU_ASSERT(mb.cell_orientation_sign[static_cast<std::size_t>(c)] != 0,
                  "multiblock cell orientation sign was not initialized");
  }
  assert_bilateral_face_adjacency(mb, stride);
  return mb;
}

void build_half_butterfly_nodes(const MultiblockIndexing& idx,
                                const tenryu::core::Config::MeshConfig& cfg,
                                std::vector<double>& node_r,
                                std::vector<double>& node_z) {
  TENRYU_ASSERT(cfg.multiblock_transition_scheme ==
                    tenryu::core::MultiblockTransitionScheme::HERMITE_BRIDGE,
                "half-butterfly geometry requires hermite_bridge transition");
  node_r.clear();
  node_z.clear();
  node_r.reserve(static_cast<std::size_t>(idx.n_nodes_total));
  node_z.reserve(static_cast<std::size_t>(idx.n_nodes_total));

  const double h = idx.r_c / static_cast<double>(idx.n_c);
  for (int i = 0; i <= idx.n_c; ++i) {
    for (int j = 0; j <= 2 * idx.n_c; ++j) {
      const RZPoint p{static_cast<double>(i) * h,
                      static_cast<double>(j - idx.n_c) * h};
      const int n = append_node(node_r, node_z, p);
      TENRYU_ASSERT(n == core_node_id(idx, i, j),
                    "half-butterfly core node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.n_nodes_core,
                "half-butterfly core node count mismatch");

  const std::vector<RZPoint> north_grid =
      build_half_butterfly_fan_grid(idx, cfg, BlockRole::NORTH_FAN);
  const std::vector<RZPoint> east_grid =
      build_half_butterfly_fan_grid(idx, cfg, BlockRole::EAST_FAN);
  const std::vector<RZPoint> south_grid =
      build_half_butterfly_fan_grid(idx, cfg, BlockRole::SOUTH_FAN);
  const int north_nj = fan_angular_cell_count(idx, BlockRole::NORTH_FAN);
  const int east_nj = fan_angular_cell_count(idx, BlockRole::EAST_FAN);
  const int south_nj = fan_angular_cell_count(idx, BlockRole::SOUTH_FAN);

  for (int l = 1; l < idx.n_b; ++l) {
    for (int k = 0; k <= idx.n_c; ++k) {
      const int n = append_node(
          node_r, node_z,
          north_grid[static_cast<std::size_t>(
              fan_grid_node_index(north_nj, l, k))]);
      TENRYU_ASSERT(n == fan_node_id(idx, BlockRole::NORTH_FAN, l, k),
                    "north fan node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.east_fan_node_offset,
                "north fan owned node count mismatch");

  for (int l = 1; l < idx.n_b; ++l) {
    for (int k = 1; k <= 2 * idx.n_c; ++k) {
      const int n = append_node(
          node_r, node_z,
          east_grid[static_cast<std::size_t>(
              fan_grid_node_index(east_nj, l, k))]);
      TENRYU_ASSERT(n == fan_node_id(idx, BlockRole::EAST_FAN, l, k),
                    "east fan node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.south_fan_node_offset,
                "east fan owned node count mismatch");

  for (int l = 1; l < idx.n_b; ++l) {
    for (int k = 1; k <= idx.n_c; ++k) {
      const int n = append_node(
          node_r, node_z,
          south_grid[static_cast<std::size_t>(
              fan_grid_node_index(south_nj, l, k))]);
      TENRYU_ASSERT(n == fan_node_id(idx, BlockRole::SOUTH_FAN, l, k),
                    "south fan node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.shell_node_offset,
                "south fan owned node count mismatch");

  for (int q = 0; q <= idx.nr_shell; ++q) {
    for (int k = 0; k <= idx.ntheta; ++k) {
      const int n = append_node(node_r, node_z, shell_point(idx, q, k));
      TENRYU_ASSERT(n == shell_node_id(idx, q, k),
                    "half-butterfly shell node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.n_nodes_total,
                "half-butterfly node count mismatch");

  for (int k = 0; k <= idx.n_c; ++k) {
    assert_seam_node_matches(node_r, node_z,
                             fan_node_id(idx, BlockRole::NORTH_FAN, 0, k),
                             north_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(north_nj, 0, k))],
                             "half north-central seam k=" +
                                 std::to_string(k));
    assert_seam_node_matches(node_r, node_z,
                             fan_node_id(idx, BlockRole::SOUTH_FAN, 0, k),
                             south_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(south_nj, 0, k))],
                             "half south-central seam k=" +
                                 std::to_string(k));
  }
  for (int k = 0; k <= 2 * idx.n_c; ++k) {
    assert_seam_node_matches(node_r, node_z,
                             fan_node_id(idx, BlockRole::EAST_FAN, 0, k),
                             east_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(east_nj, 0, k))],
                             "half east-central seam k=" +
                                 std::to_string(k));
  }
  for (int k = 0; k <= idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::NORTH_FAN, idx.n_b, k) ==
                      shell_node_id(idx, 0, k),
                  "north fan outer seam must share shell nodes");
    assert_seam_node_matches(node_r, node_z,
                             fan_node_id(idx, BlockRole::NORTH_FAN, idx.n_b, k),
                             north_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(north_nj, idx.n_b, k))],
                             "half north-shell seam k=" +
                                 std::to_string(k));
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::SOUTH_FAN, idx.n_b, k) ==
                      shell_node_id(idx, 0, 3 * idx.n_c + k),
                  "south fan outer seam must share shell nodes");
    assert_seam_node_matches(node_r, node_z,
                             fan_node_id(idx, BlockRole::SOUTH_FAN, idx.n_b, k),
                             south_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(south_nj, idx.n_b, k))],
                             "half south-shell seam k=" +
                                 std::to_string(k));
  }
  for (int k = 0; k <= 2 * idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::EAST_FAN, idx.n_b, k) ==
                      shell_node_id(idx, 0, idx.n_c + k),
                  "east fan outer seam must share shell nodes");
    assert_seam_node_matches(node_r, node_z,
                             fan_node_id(idx, BlockRole::EAST_FAN, idx.n_b, k),
                             east_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(east_nj, idx.n_b, k))],
                             "half east-shell seam k=" + std::to_string(k));
  }
  for (int l = 0; l <= idx.n_b; ++l) {
    const int ne_node = fan_node_id(idx, BlockRole::NORTH_FAN, l, idx.n_c);
    TENRYU_ASSERT(ne_node == fan_node_id(idx, BlockRole::EAST_FAN, l, 0),
                  "north-east fan diagonal must share node ids");
    assert_seam_node_matches(node_r, node_z, ne_node,
                             north_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(north_nj, l, idx.n_c))],
                             "half north-east diagonal north l=" +
                                 std::to_string(l));
    assert_seam_node_matches(node_r, node_z, ne_node,
                             east_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(east_nj, l, 0))],
                             "half north-east diagonal east l=" +
                                 std::to_string(l));

    const int es_node = fan_node_id(idx, BlockRole::EAST_FAN, l, 2 * idx.n_c);
    TENRYU_ASSERT(es_node == fan_node_id(idx, BlockRole::SOUTH_FAN, l, 0),
                  "east-south fan diagonal must share node ids");
    assert_seam_node_matches(node_r, node_z, es_node,
                             east_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(east_nj, l, 2 * idx.n_c))],
                             "half east-south diagonal east l=" +
                                 std::to_string(l));
    assert_seam_node_matches(node_r, node_z, es_node,
                             south_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(south_nj, l, 0))],
                             "half east-south diagonal south l=" +
                                 std::to_string(l));
  }
  for (int j = 0; j <= 2 * idx.n_c; ++j) {
    const int node = core_node_id(idx, 0, j);
    TENRYU_ASSERT(node_r[static_cast<std::size_t>(node)] == 0.0,
                  "half-butterfly central axis must have R=0 exactly");
  }
  for (int l = 0; l <= idx.n_b; ++l) {
    const int north_axis = fan_node_id(idx, BlockRole::NORTH_FAN, l, 0);
    const int south_axis = fan_node_id(idx, BlockRole::SOUTH_FAN, l, idx.n_c);
    TENRYU_ASSERT(node_r[static_cast<std::size_t>(north_axis)] == 0.0,
                  "half-butterfly north fan axis must have R=0 exactly");
    TENRYU_ASSERT(node_r[static_cast<std::size_t>(south_axis)] == 0.0,
                  "half-butterfly south fan axis must have R=0 exactly");
  }
  for (int k = 0; k <= idx.ntheta; ++k) {
    assert_seam_node_matches(node_r, node_z, shell_node_id(idx, 0, k),
                             polar_halfplane_point(idx, idx.r_match, k),
                             "half fan-shell seam k=" + std::to_string(k));
  }
}

void build_trifan_cap_nodes(const MultiblockIndexing& idx,
                            const tenryu::core::Config::MeshConfig& cfg,
                            std::vector<double>& node_r,
                            std::vector<double>& node_z) {
  TENRYU_ASSERT(idx.has_trifan_cap,
                "trifan cap geometry requires cap indexing");
  TENRYU_ASSERT(cfg.multiblock_transition_scheme ==
                    tenryu::core::MultiblockTransitionScheme::HERMITE_BRIDGE,
                "trifan cap geometry requires hermite_bridge transition");
  node_r.clear();
  node_z.clear();
  node_r.reserve(static_cast<std::size_t>(idx.n_nodes_total));
  node_z.reserve(static_cast<std::size_t>(idx.n_nodes_total));

  const int apex = append_node(node_r, node_z, RZPoint{0.0, 0.0});
  TENRYU_ASSERT(apex == cap_apex_node_id(idx),
                "trifan cap apex node id ordering mismatch");
  for (int l = 1; l <= idx.n_cap; ++l) {
    for (int k = 0; k <= idx.ntheta; ++k) {
      const int n = append_node(node_r, node_z, cap_ring_point(idx, l, k));
      TENRYU_ASSERT(n == cap_ring_node_id(idx, l, k),
                    "trifan cap ring node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.n_nodes_core,
                "trifan cap node count mismatch");

  const std::vector<RZPoint> north_grid =
      build_half_butterfly_fan_grid(idx, cfg, BlockRole::NORTH_FAN);
  const std::vector<RZPoint> east_grid =
      build_half_butterfly_fan_grid(idx, cfg, BlockRole::EAST_FAN);
  const std::vector<RZPoint> south_grid =
      build_half_butterfly_fan_grid(idx, cfg, BlockRole::SOUTH_FAN);
  const int north_nj = fan_angular_cell_count(idx, BlockRole::NORTH_FAN);
  const int east_nj = fan_angular_cell_count(idx, BlockRole::EAST_FAN);
  const int south_nj = fan_angular_cell_count(idx, BlockRole::SOUTH_FAN);

  for (int l = 1; l < idx.n_b; ++l) {
    for (int k = 0; k <= idx.n_c; ++k) {
      const int n = append_node(
          node_r, node_z,
          north_grid[static_cast<std::size_t>(
              fan_grid_node_index(north_nj, l, k))]);
      TENRYU_ASSERT(n == fan_node_id(idx, BlockRole::NORTH_FAN, l, k),
                    "north fan node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.east_fan_node_offset,
                "north fan owned node count mismatch");

  for (int l = 1; l < idx.n_b; ++l) {
    for (int k = 1; k <= 2 * idx.n_c; ++k) {
      const int n = append_node(
          node_r, node_z,
          east_grid[static_cast<std::size_t>(
              fan_grid_node_index(east_nj, l, k))]);
      TENRYU_ASSERT(n == fan_node_id(idx, BlockRole::EAST_FAN, l, k),
                    "east fan node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.south_fan_node_offset,
                "east fan owned node count mismatch");

  for (int l = 1; l < idx.n_b; ++l) {
    for (int k = 1; k <= idx.n_c; ++k) {
      const int n = append_node(
          node_r, node_z,
          south_grid[static_cast<std::size_t>(
              fan_grid_node_index(south_nj, l, k))]);
      TENRYU_ASSERT(n == fan_node_id(idx, BlockRole::SOUTH_FAN, l, k),
                    "south fan node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.shell_node_offset,
                "south fan owned node count mismatch");

  for (int q = 0; q <= idx.nr_shell; ++q) {
    for (int k = 0; k <= idx.ntheta; ++k) {
      const int n = append_node(node_r, node_z, shell_point(idx, q, k));
      TENRYU_ASSERT(n == shell_node_id(idx, q, k),
                    "trifan cap shell node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.n_nodes_total,
                "trifan cap total node count mismatch");

  for (int k = 0; k <= idx.n_c; ++k) {
    assert_seam_node_matches(node_r, node_z,
                             fan_node_id(idx, BlockRole::NORTH_FAN, 0, k),
                             north_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(north_nj, 0, k))],
                             "trifan cap north fan inner seam k=" +
                                 std::to_string(k));
    assert_seam_node_matches(node_r, node_z,
                             fan_node_id(idx, BlockRole::SOUTH_FAN, 0, k),
                             south_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(south_nj, 0, k))],
                             "trifan cap south fan inner seam k=" +
                                 std::to_string(k));
  }
  for (int k = 0; k <= 2 * idx.n_c; ++k) {
    assert_seam_node_matches(node_r, node_z,
                             fan_node_id(idx, BlockRole::EAST_FAN, 0, k),
                             east_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(east_nj, 0, k))],
                             "trifan cap east fan inner seam k=" +
                                 std::to_string(k));
  }
  for (int k = 0; k <= idx.ntheta; ++k) {
    assert_seam_node_matches(node_r, node_z,
                             cap_ring_node_id(idx, idx.n_cap, k),
                             core_boundary_point(idx, k),
                             "trifan cap outer ring k=" + std::to_string(k));
    assert_seam_node_matches(node_r, node_z, shell_node_id(idx, 0, k),
                             polar_halfplane_point(idx, idx.r_match, k),
                             "trifan cap fan-shell seam k=" +
                                 std::to_string(k));
  }
  for (int l = 0; l <= idx.n_b; ++l) {
    const int ne_node = fan_node_id(idx, BlockRole::NORTH_FAN, l, idx.n_c);
    TENRYU_ASSERT(ne_node == fan_node_id(idx, BlockRole::EAST_FAN, l, 0),
                  "north-east fan diagonal must share node ids");
    assert_seam_node_matches(node_r, node_z, ne_node,
                             north_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(north_nj, l, idx.n_c))],
                             "trifan cap north-east diagonal north l=" +
                                 std::to_string(l));
    assert_seam_node_matches(node_r, node_z, ne_node,
                             east_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(east_nj, l, 0))],
                             "trifan cap north-east diagonal east l=" +
                                 std::to_string(l));

    const int es_node = fan_node_id(idx, BlockRole::EAST_FAN, l, 2 * idx.n_c);
    TENRYU_ASSERT(es_node == fan_node_id(idx, BlockRole::SOUTH_FAN, l, 0),
                  "east-south fan diagonal must share node ids");
    assert_seam_node_matches(node_r, node_z, es_node,
                             east_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(east_nj, l, 2 * idx.n_c))],
                             "trifan cap east-south diagonal east l=" +
                                 std::to_string(l));
    assert_seam_node_matches(node_r, node_z, es_node,
                             south_grid[static_cast<std::size_t>(
                                 fan_grid_node_index(south_nj, l, 0))],
                             "trifan cap east-south diagonal south l=" +
                                 std::to_string(l));
  }
  TENRYU_ASSERT(node_r[static_cast<std::size_t>(cap_apex_node_id(idx))] == 0.0,
                "trifan cap apex must have R=0 exactly");
  TENRYU_ASSERT(node_z[static_cast<std::size_t>(cap_apex_node_id(idx))] == 0.0,
                "trifan cap apex must have Z=0 exactly");
  for (int l = 1; l <= idx.n_cap; ++l) {
    TENRYU_ASSERT(node_r[static_cast<std::size_t>(
                      cap_ring_node_id(idx, l, 0))] == 0.0,
                  "trifan cap north axis ring endpoint must have R=0 exactly");
    TENRYU_ASSERT(node_r[static_cast<std::size_t>(
                      cap_ring_node_id(idx, l, idx.ntheta))] == 0.0,
                  "trifan cap south axis ring endpoint must have R=0 exactly");
  }
}

void build_multiblock_nodes(const MultiblockIndexing& idx,
                            const tenryu::core::Config::MeshConfig& cfg,
                            std::vector<double>& node_r,
                            std::vector<double>& node_z,
                            int& polar_in_box_j_tr,
                            int& polar_in_box_j_br) {
  if (multiblock_indexing_is_half_butterfly(idx)) {
    build_half_butterfly_nodes(idx, cfg, node_r, node_z);
    return;
  }
  if (multiblock_indexing_has_trifan_cap(idx)) {
    build_trifan_cap_nodes(idx, cfg, node_r, node_z);
    return;
  }

  node_r.clear();
  node_z.clear();
  node_r.reserve(static_cast<std::size_t>(idx.n_nodes_total));
  node_z.reserve(static_cast<std::size_t>(idx.n_nodes_total));

  const auto transition_scheme = cfg.multiblock_transition_scheme;
  const bool rounded_core_seam =
      transition_scheme ==
      tenryu::core::MultiblockTransitionScheme::ROUNDED_CORE_SEAM;
  const std::vector<RZPoint> rounded_core_grid =
      rounded_core_seam ? build_rounded_core_grid(idx, cfg) : std::vector<RZPoint>{};
  if (rounded_core_seam) {
    for (int i = 0; i <= idx.n_c; ++i) {
      for (int j = 0; j <= 2 * idx.n_c; ++j) {
        const int expected = core_node_id(idx, i, j);
        const int n = append_node(
            node_r, node_z,
            rounded_core_grid[static_cast<std::size_t>(expected)]);
        TENRYU_ASSERT(n == expected, "core node id ordering mismatch");
      }
    }
  } else {
    const double h = idx.r_c / static_cast<double>(idx.n_c);
    for (int i = 0; i <= idx.n_c; ++i) {
      for (int j = 0; j <= 2 * idx.n_c; ++j) {
        const RZPoint p{static_cast<double>(i) * h,
                        static_cast<double>(j - idx.n_c) * h};
        const int n = append_node(node_r, node_z, p);
        TENRYU_ASSERT(n == core_node_id(idx, i, j),
                      "core node id ordering mismatch");
      }
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.n_nodes_core,
                "core node count mismatch");

  const bool rounded_bridge =
      transition_scheme ==
          tenryu::core::MultiblockTransitionScheme::ROUNDED_HALF_BUTTERFLY ||
      rounded_core_seam;
  const std::vector<RZPoint> rounded_grid =
      rounded_bridge ? build_rounded_bridge_grid(idx, cfg) : std::vector<RZPoint>{};
  std::vector<double> bridge_w_levels;
  if (cfg.multiblock_cart_core_bridge_grading == "quintic_log") {
    const double h_core = idx.r_c / static_cast<double>(idx.n_c);
    const double h_shell = idx.shell_s_nodes[1] - idx.shell_s_nodes[0];
    TENRYU_ASSERT(h_core > 0.0,
                  "quintic-log bridge core spacing must be positive");
    TENRYU_ASSERT(h_shell > 0.0,
                  "quintic-log bridge shell spacing must be positive");
    std::vector<double> sigma(static_cast<std::size_t>(idx.n_b), 0.0);
    double sigma_sum = 0.0;
    for (int l = 0; l < idx.n_b; ++l) {
      const double t =
          (static_cast<double>(l) + 0.5) / static_cast<double>(idx.n_b);
      const double t2 = t * t;
      const double t3 = t2 * t;
      const double t4 = t3 * t;
      const double t5 = t4 * t;
      const double S = 6.0 * t5 - 15.0 * t4 + 10.0 * t3;
      const double value =
          std::exp(std::log(h_core) +
                   (std::log(h_shell) - std::log(h_core)) * S);
      TENRYU_ASSERT(std::isfinite(value) && value > 0.0,
                    "quintic-log bridge sigma must be finite and positive");
      sigma[static_cast<std::size_t>(l)] = value;
      sigma_sum += value;
    }
    bridge_w_levels.assign(static_cast<std::size_t>(idx.n_b), 0.0);
    for (int l = 1; l < idx.n_b; ++l) {
      bridge_w_levels[static_cast<std::size_t>(l)] =
          bridge_w_levels[static_cast<std::size_t>(l - 1)] +
          sigma[static_cast<std::size_t>(l - 1)] / sigma_sum;
      const double w = bridge_w_levels[static_cast<std::size_t>(l)];
      TENRYU_ASSERT(
          w > bridge_w_levels[static_cast<std::size_t>(l - 1)] && w < 1.0,
          "quintic-log bridge blend levels must increase strictly in (0, 1)");
    }
  }
  for (int l = 1; l < idx.n_b; ++l) {
    for (int k = 0; k <= idx.ntheta; ++k) {
      const RZPoint p =
          rounded_bridge
              ? rounded_grid[static_cast<std::size_t>(
                    bridge_grid_node_index(idx, l, k))]
              : (cfg.multiblock_cart_core_bridge_grading == "quintic_log"
                     ? bridge_point_at_blend(
                           idx,
                           bridge_w_levels[static_cast<std::size_t>(l)], k)
                     : bridge_point(idx, l, k));
      const int n = append_node(node_r, node_z, p);
      TENRYU_ASSERT(n == bridge_interior_node_id(idx, l, k),
                    "bridge interior node id ordering mismatch");
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.shell_node_offset,
                "bridge interior node count mismatch");

  const bool polar_in_box = cfg.logical_mesh_2d == "polar_in_box";
  const int prefix_node_count =
      polar_in_box ? static_cast<int>(idx.shell_s_nodes.size())
                   : idx.nr_shell + 1;
  for (int q = 0; q < prefix_node_count; ++q) {
    for (int k = 0; k <= idx.ntheta; ++k) {
      const int n = append_node(node_r, node_z, shell_point(idx, q, k));
      TENRYU_ASSERT(n == shell_node_id(idx, q, k),
                    "shell node id ordering mismatch");
    }
  }
  if (polar_in_box) {
    std::vector<double> theta_nodes(
        static_cast<std::size_t>(idx.ntheta + 1), 0.0);
    for (int k = 0; k <= idx.ntheta; ++k) {
      theta_nodes[static_cast<std::size_t>(k)] =
          static_cast<double>(k) * spherical_wedge_detail::kPi /
          static_cast<double>(idx.ntheta);
    }
    const double theta_tr =
        std::atan2(cfg.box_r_max, cfg.box_z_max - cfg.box_center_z);
    const double theta_br =
        spherical_wedge_detail::kPi -
        std::atan2(cfg.box_r_max, cfg.box_center_z - cfg.box_z_min);
    polar_in_box_j_tr = nearest_theta_index(theta_nodes, theta_tr);
    polar_in_box_j_br = nearest_theta_index(theta_nodes, theta_br);
    if (!(polar_in_box_j_tr > 0 &&
          polar_in_box_j_tr < polar_in_box_j_br &&
          polar_in_box_j_br < idx.ntheta)) {
      throw core::namelist::ConfigError(
          "polar_in_box corner bands require distinct interior theta indices; increase Mesh.nz");
    }

    const PolarInBoxExteriorContext exterior_context =
        build_polar_in_box_exterior_context(
            cfg, theta_nodes, polar_in_box_j_tr, polar_in_box_j_br);
    const int lock_q = prefix_node_count - 1;
    const double lock_radius = idx.shell_s_nodes.back();
    std::vector<RZPoint> lock_nodes;
    lock_nodes.reserve(static_cast<std::size_t>(idx.ntheta + 1));
    std::vector<PolarInBoxExteriorRay> exterior_rays;
    exterior_rays.reserve(static_cast<std::size_t>(idx.ntheta + 1));
    for (int k = 0; k <= idx.ntheta; ++k) {
      const int n = shell_node_id(idx, lock_q, k);
      const RZPoint lock_node{node_r[static_cast<std::size_t>(n)],
                              node_z[static_cast<std::size_t>(n)]};
      lock_nodes.push_back(lock_node);
      exterior_rays.push_back(polar_in_box_exterior_ray(
          k, lock_node, lock_radius, exterior_context));
    }
    apply_polar_in_box_cone_postpass(
        lock_nodes, &exterior_rays, exterior_context);
    const int exterior_node_count = cfg.morph_rings + cfg.collar_rings;
    TENRYU_ASSERT(prefix_node_count + exterior_node_count ==
                      idx.nr_shell + 1,
                  "polar_in_box multiblock shell ring count mismatch");
    for (int p = 0; p < cfg.morph_rings; ++p) {
      const int ring = p + 1;
      const double xi = static_cast<double>(ring) /
                        static_cast<double>(cfg.morph_rings);
      const double angular_blend = delayed_quintic_blend(xi);
      double previous_phi = -1.0;
      for (int k = 0; k <= idx.ntheta; ++k) {
        const RZPoint target =
            exterior_context.collar.morph_target[static_cast<std::size_t>(k)];
        const double target_phi = polar_angle(target, cfg.box_center_z);
        const double phi = theta_nodes[static_cast<std::size_t>(k)] +
                           angular_blend *
                               (target_phi -
                                theta_nodes[static_cast<std::size_t>(k)]);
        if (!(k == 0 || phi > previous_phi)) {
          throw core::namelist::ConfigError(
              "polar_in_box morph angular labels are not strictly monotone");
        }
        previous_phi = phi;
        const RZPoint point =
            exterior_rays[static_cast<std::size_t>(k)]
                .nodes[static_cast<std::size_t>(p)];
        const int q = prefix_node_count + p;
        const int n = append_node(node_r, node_z, point);
        TENRYU_ASSERT(n == shell_node_id(idx, q, k),
                      "shell node id ordering mismatch");
      }
    }
    for (int p = cfg.morph_rings; p < exterior_node_count; ++p) {
      for (int k = 0; k <= idx.ntheta; ++k) {
        const RZPoint point =
            exterior_rays[static_cast<std::size_t>(k)]
                .nodes[static_cast<std::size_t>(p)];
        const int q = prefix_node_count + p;
        const int n = append_node(node_r, node_z, point);
        TENRYU_ASSERT(n == shell_node_id(idx, q, k),
                      "shell node id ordering mismatch");
      }
    }
  }
  TENRYU_ASSERT(static_cast<int>(node_r.size()) == idx.n_nodes_total,
                "multiblock node count mismatch");

  for (int k = 0; k <= idx.ntheta; ++k) {
    const RZPoint core_seam_expected =
        rounded_core_seam ? rounded_cap_point(idx, cfg.multiblock_cap_p, k)
                          : core_boundary_point(idx, k);
    assert_seam_node_matches(node_r, node_z, core_boundary_node_id(idx, k),
                             core_seam_expected,
                             "core-bridge inner k=" + std::to_string(k));
    assert_seam_node_matches(node_r, node_z, shell_node_id(idx, 0, k),
                             polar_halfplane_point(idx, idx.r_match, k),
                             "bridge-shell outer k=" + std::to_string(k));
  }
}

Mesh create_multiblock_cart_core_polar_shell_mesh(
    const tenryu::core::Config& cfg,
    tenryu::core::State& state) {
  Mesh mesh;
  mesh.corner_stride =
      tenryu::core::corner_stride_for_scheme(cfg.mesh.topology_scheme);
  TENRYU_ASSERT(mesh.corner_stride == state.corner_stride,
                "Mesh/state corner stride mismatch");
  mesh.dim = cfg.main.dim;
  const bool legacy_cylindrical_1d =
      cfg.main.dim == 1 && cfg.main.dimension == "1D_CYL";
  mesh.geometry_code =
      (legacy_cylindrical_1d || cfg.mesh.geometry_1d == "cylindrical")
          ? 1
          : ((cfg.mesh.geometry_1d == "planar") ? 2 : 0);
  mesh.logical = parse_logical_mesh_2d(cfg.mesh.logical_mesh_2d);
  mesh.polar_center_treatment =
      parse_polar_center_treatment(cfg.mesh.polar_center_treatment);
  mesh.multiblock_outer_svec_tangent_balance =
      cfg.mesh.multiblock_outer_svec_tangent_balance;

  TENRYU_ASSERT(mesh.dim == 2,
                "multiblock topology requires Main.dim == 2");
  TENRYU_ASSERT(cfg.main.dimension == "2D_RZ",
                "multiblock topology requires Main.dimension='2D_RZ'");
  TENRYU_ASSERT(mesh_topo_is_multiblock(cfg.mesh),
                "multiblock builder called for non-multiblock topology");

  const MultiblockIndexing idx = make_multiblock_indexing(cfg.mesh);
  mesh.topo.nr = cfg.mesh.nr;
  mesh.topo.nz = cfg.mesh.nz;
  mesh.topo.n_cells = idx.n_cells_total;
  mesh.topo.n_nodes = idx.n_nodes_total;
  mesh.cell_nverts.assign(static_cast<std::size_t>(mesh.topo.n_cells), 4U);
  if (multiblock_indexing_has_trifan_cap(idx)) {
    for (int k = 0; k < idx.ntheta; ++k) {
      mesh.cell_nverts[static_cast<std::size_t>(cap_cell_id(idx, 0, k))] = 3U;
    }
  }
  mesh.edge_tags.clear();

  std::vector<double> host_x_r;
  std::vector<double> host_x_z;
  int polar_in_box_j_tr = -1;
  int polar_in_box_j_br = -1;
  build_multiblock_nodes(idx, cfg.mesh, host_x_r, host_x_z,
                         polar_in_box_j_tr, polar_in_box_j_br);
  mesh.topo.polar_in_box_j_tr = polar_in_box_j_tr;
  mesh.topo.polar_in_box_j_br = polar_in_box_j_br;
  set_multiblock_node_flags(mesh.topo, idx, host_x_r);
  const std::vector<std::uint8_t>* active_cell_nverts =
      multiblock_indexing_has_trifan_cap(idx) ? &mesh.cell_nverts : nullptr;
  mesh.topo.multiblock =
      build_multiblock_topology(idx, mesh.corner_stride, active_cell_nverts);
  auto& mb = *mesh.topo.multiblock;
  tenryu::hydro::build_unique_oriented_face_list(
      mb, mb.cell_id_stable, mb.unique_internal_faces, mb.boundary_faces,
      active_cell_nverts, mesh.corner_stride);
  upload_unique_face_payload(mb);
  mesh.multiblock_cell_node_csr_offsets.reset(mb.cell_node_csr_offsets.size());
  mesh.multiblock_cell_node_csr_offsets.copy_from_host(
      mb.cell_node_csr_offsets);
  mesh.multiblock_cell_node_csr_indices.reset(mb.cell_node_csr_indices.size());
  mesh.multiblock_cell_node_csr_indices.copy_from_host(
      mb.cell_node_csr_indices);
  const tenryu::hydro::ReverseCellNodeCSR reverse_csr =
      tenryu::hydro::build_reverse_cell_node_csr(
          mb, mesh.topo.n_nodes, active_cell_nverts, mesh.corner_stride);
  mesh.multiblock_reverse_csr_node_offsets.reset(
      reverse_csr.node_offsets.size());
  mesh.multiblock_reverse_csr_node_offsets.copy_from_host(
      reverse_csr.node_offsets);
  mesh.multiblock_reverse_csr_node_cells.reset(reverse_csr.node_cells.size());
  mesh.multiblock_reverse_csr_node_cells.copy_from_host(reverse_csr.node_cells);
  mesh.multiblock_reverse_csr_node_corners.reset(
      reverse_csr.node_corners.size());
  mesh.multiblock_reverse_csr_node_corners.copy_from_host(
      reverse_csr.node_corners);

  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(mesh.topo.n_nodes),
                "State.x_r size does not match multiblock mesh node count");
  TENRYU_ASSERT(state.x_z.size() == static_cast<std::size_t>(mesh.topo.n_nodes),
                "State.x_z size does not match multiblock mesh node count");
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
  constexpr double pi =
      3.1415926535897932384626433832795028841971693993751058209749;
  const double expected_total_vol =
      (mesh.logical == LogicalMesh2D::PolarInBox)
          ? (pi * cfg.mesh.box_r_max * cfg.mesh.box_r_max *
             (cfg.mesh.box_z_max - cfg.mesh.box_z_min))
          : ((mesh.geometry_code == 0)
                 ? (four_pi_over_three * idx.s_max * idx.s_max * idx.s_max)
                 : geometry_1d_shell_volume_cubes(mesh.geometry_code, 0.0,
                                                  idx.s_max));
  const double rel_err =
      std::abs(total_vol - expected_total_vol) / expected_total_vol;
  const double total_vol_rel_tol =
      std::max(1.0e-12, 10.0 / (static_cast<double>(idx.n_c) *
                                static_cast<double>(idx.n_c)));
  // The polygonal half-plane boundary has O(1/N_c^2) approximation error vs
  // the analytic curved region; this tolerance tracks that refinement behavior.
  TENRYU_ASSERT(std::isfinite(total_vol) &&
                    std::isfinite(rel_err) &&
                    rel_err <= total_vol_rel_tol,
                "multiblock initial total volume mismatch: rel_err=" +
                    std::to_string(rel_err));
  state.cell_vol_initial = mesh.cell_vol;
  return mesh;
}

__device__ inline void recompute_geometry_1d_kernel_body(
    const int i,
    double* __restrict__ cell_vol,
    double* __restrict__ cell_centroid_r,
    double* __restrict__ cell_centroid_z,
    const double* __restrict__ node_r,
    const int nr,
    const int geom) {
  constexpr double four_pi_over_three =
      4.188790204786390984616857844372670512262892532500141094646;
  const double r0 = node_r[i];
  const double r1 = node_r[i + 1];
  const double dr = r1 - r0;
  // W-G pattern v2: spherical keeps the historic expression verbatim
  // (bitwise contract); non-spherical routes through the shared helper.
  const double vol =
      (geom == 0)
          ? (four_pi_over_three * dr * (r1 * r1 + r1 * r0 + r0 * r0))
          : geometry_1d_shell_volume(geom, r0, r1);

  cell_vol[i] = vol;
  cell_centroid_r[i] = 0.5 * (r0 + r1);
  cell_centroid_z[i] = 0.0;
}

__global__ void recompute_geometry_1d_kernel(double* __restrict__ cell_vol,
                                             double* __restrict__ cell_centroid_r,
                                             double* __restrict__ cell_centroid_z,
                                             const double* __restrict__ node_r,
                                             const int nr,
                                             const int geom) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr) {
    return;
  }

  recompute_geometry_1d_kernel_body(i, cell_vol, cell_centroid_r,
                                    cell_centroid_z, node_r, nr, geom);
}

template <int ORIENT>
__global__ void recompute_geometry_2d_kernel(double* __restrict__ cell_vol,
                                             double* __restrict__ cell_area,
                                             double* __restrict__ cell_centroid_r,
                                             double* __restrict__ cell_centroid_z,
                                             double* __restrict__ cell_Svec_r,
                                             double* __restrict__ cell_Svec_z,
                                             const double* __restrict__ node_r,
                                             const double* __restrict__ node_z,
                                             const int nr,
                                             const int nz,
                                             const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  double r[4] = {node_r[n00], node_r[n10], node_r[n11], node_r[n01]};
  double z[4] = {node_z[n00], node_z[n10], node_z[n11], node_z[n01]};

  const double centroid_r = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double centroid_z = 0.25 * (z[0] + z[1] + z[2] + z[3]);

  double vol_sum = 0.0;
  double area_cross_sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp1 = (k + 1) & 3;
    const double rk = r[k] - centroid_r;
    const double rkp1 = r[kp1] - centroid_r;
    const double zk = z[k] - centroid_z;
    const double zkp1 = z[kp1] - centroid_z;
    const double cross_local = rk * zkp1 - rkp1 * zk;
    const double weight = rk + rkp1 + 3.0 * centroid_r;
    vol_sum += cross_local * weight;
    area_cross_sum += cross_local;
  }

  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  const double vol = pi_over_three * vol_sum;
  const double area = 0.5 * fabs(area_cross_sum);

  if constexpr (ORIENT < 0) {
    cell_vol[c] = -vol;
  } else {
    cell_vol[c] = vol;
  }
  cell_area[c] = area;
  cell_centroid_r[c] = centroid_r;
  cell_centroid_z[c] = centroid_z;

  for (int k = 0; k < 4; ++k) {
    const int km1 = (k + 3) & 3;
    const int kp1 = (k + 1) & 3;

    const double trkm1 = r[km1] - centroid_r;
    const double trk = r[k] - centroid_r;
    const double trkp1 = r[kp1] - centroid_r;
    const double tzkm1 = z[km1] - centroid_z;
    const double tzk = z[k] - centroid_z;
    const double tzkp1 = z[kp1] - centroid_z;

    const double sr =
        pi_over_three *
        (2.0 * trk * (tzkp1 - tzkm1) + trkp1 * (tzkp1 - tzk) +
         trkm1 * (tzk - tzkm1) + 3.0 * centroid_r * (tzkp1 - tzkm1));
    const double sz = pi_over_three *
                      (trkm1 * (trkm1 + trk) - trkp1 * (trk + trkp1) +
                       3.0 * centroid_r * (trkm1 - trkp1));

    const int idx = c * corner_stride + k;
    if constexpr (ORIENT < 0) {
      cell_Svec_r[idx] = -sr;
      cell_Svec_z[idx] = -sz;
    } else {
      cell_Svec_r[idx] = sr;
      cell_Svec_z[idx] = sz;
    }
  }
}

__global__ void recompute_geometry_2d_kernel_polar(
    double* __restrict__ cell_vol,
    double* __restrict__ cell_area,
    double* __restrict__ cell_centroid_r,
    double* __restrict__ cell_centroid_z,
    double* __restrict__ cell_Svec_r,
    double* __restrict__ cell_Svec_z,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const int nr,
    const int nz,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  double r[4] = {node_r[n00], node_r[n10], node_r[n11], node_r[n01]};
  double z[4] = {node_z[n00], node_z[n10], node_z[n11], node_z[n01]};

  const double s_low = 0.5 * (std::hypot(r[0], z[0]) + std::hypot(r[3], z[3]));
  const double s_high = 0.5 * (std::hypot(r[1], z[1]) + std::hypot(r[2], z[2]));
  const double theta_low = 0.5 * (std::atan2(r[0], z[0]) + std::atan2(r[1], z[1]));
  const double theta_high = 0.5 * (std::atan2(r[2], z[2]) + std::atan2(r[3], z[3]));

  const double centroid_r = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double centroid_z = 0.25 * (z[0] + z[1] + z[2] + z[3]);

  double area_cross_sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp1 = (k + 1) & 3;
    const double rk = r[k] - centroid_r;
    const double rkp1 = r[kp1] - centroid_r;
    const double zk = z[k] - centroid_z;
    const double zkp1 = z[kp1] - centroid_z;
    area_cross_sum += rk * zkp1 - rkp1 * zk;
  }

  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  cell_vol[c] = spherical_wedge_volume(s_low, s_high, theta_low, theta_high);
  cell_area[c] = 0.5 * fabs(area_cross_sum);
  cell_centroid_r[c] = centroid_r;
  cell_centroid_z[c] = centroid_z;

  for (int k = 0; k < 4; ++k) {
    const int km1 = (k + 3) & 3;
    const int kp1 = (k + 1) & 3;

    const double trkm1 = r[km1] - centroid_r;
    const double trk = r[k] - centroid_r;
    const double trkp1 = r[kp1] - centroid_r;
    const double tzkm1 = z[km1] - centroid_z;
    const double tzk = z[k] - centroid_z;
    const double tzkp1 = z[kp1] - centroid_z;

    const double sr =
        pi_over_three *
        (2.0 * trk * (tzkp1 - tzkm1) + trkp1 * (tzkp1 - tzk) +
         trkm1 * (tzk - tzkm1) + 3.0 * centroid_r * (tzkp1 - tzkm1));
    const double sz = pi_over_three *
                      (trkm1 * (trkm1 + trk) - trkp1 * (trk + trkp1) +
                       3.0 * centroid_r * (trkm1 - trkp1));

    const int idx = c * corner_stride + k;
    cell_Svec_r[idx] = sr;
    cell_Svec_z[idx] = sz;
  }
}

__global__ void recompute_geometry_2d_kernel_polar_tri_fan(
    double* __restrict__ cell_vol,
    double* __restrict__ cell_area,
    double* __restrict__ cell_centroid_r,
    double* __restrict__ cell_centroid_z,
    double* __restrict__ cell_Svec_r,
    double* __restrict__ cell_Svec_z,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const std::uint8_t* __restrict__ cell_nverts,
    const int nr,
    const int nz,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  double r[kMeshTopoCellStorageSlotsMax] = {
      node_r[n00], node_r[n10], node_r[n11], node_r[n01]};
  double z[kMeshTopoCellStorageSlotsMax] = {
      node_z[n00], node_z[n10], node_z[n11], node_z[n01]};
  const int nverts = mesh_topo_cell_active_nverts(cell_nverts, c);

  double centroid_r = 0.0;
  double centroid_z = 0.0;
  rz_polygon_area_centroid(r, z, nverts, &centroid_r, &centroid_z);

  constexpr double polar_orientation_sign = -1.0;
  const double signed_volume = rz_polygon_volume_exact(r, z, nverts);
  const double signed_area2 = rz_polygon_area2(r, z, nverts);

  cell_vol[c] = polar_orientation_sign * signed_volume;
  cell_area[c] = 0.5 * polar_orientation_sign * signed_area2;
  cell_centroid_r[c] = centroid_r;
  cell_centroid_z[c] = centroid_z;

  for (int k = 0; k < nverts; ++k) {
    double sr = 0.0;
    double sz = 0.0;
    rz_polygon_svec(r, z, nverts, k, centroid_r, centroid_z,
                    polar_orientation_sign, &sr, &sz);
    const int idx = c * corner_stride + k;
    cell_Svec_r[idx] = sr;
    cell_Svec_z[idx] = sz;
  }
  for (int k = nverts; k < 4; ++k) {
    const int idx = c * corner_stride + k;
    cell_Svec_r[idx] = 0.0;
    cell_Svec_z[idx] = 0.0;
  }
}

__global__ void recompute_geometry_2d_kernel_multiblock_csr(
    double* __restrict__ cell_vol,
    double* __restrict__ cell_area,
    double* __restrict__ cell_centroid_r,
    double* __restrict__ cell_centroid_z,
    double* __restrict__ cell_Svec_r,
    double* __restrict__ cell_Svec_z,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const int off = cell_node_csr_offsets[c];
  const int nverts = mesh_topo_cell_active_nverts(cell_nverts, c);
  double r[kMeshTopoCellStorageSlotsMax] = {0.0};
  double z[kMeshTopoCellStorageSlotsMax] = {0.0};
  for (int k = 0; k < nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    r[k] = node_r[n];
    z[k] = node_z[n];
  }

  double centroid_r = 0.0;
  double centroid_z = 0.0;
  rz_polygon_area_centroid(r, z, nverts, &centroid_r, &centroid_z);

  const double orientation_sign =
      static_cast<double>(cell_orientation_sign[c]);
  const double signed_volume = rz_polygon_volume_exact(r, z, nverts);
  const double signed_area2 = rz_polygon_area2(r, z, nverts);

  cell_vol[c] = orientation_sign * signed_volume;
  cell_area[c] = 0.5 * orientation_sign * signed_area2;
  cell_centroid_r[c] = centroid_r;
  cell_centroid_z[c] = centroid_z;

  for (int k = 0; k < nverts; ++k) {
    double sr = 0.0;
    double sz = 0.0;
    rz_polygon_svec(r, z, nverts, k, centroid_r, centroid_z, orientation_sign,
                    &sr, &sz);
    const int idx = c * corner_stride + k;
    cell_Svec_r[idx] = sr;
    cell_Svec_z[idx] = sz;
  }
  for (int k = nverts; k < 4; ++k) {
    const int idx = c * corner_stride + k;
    cell_Svec_r[idx] = 0.0;
    cell_Svec_z[idx] = 0.0;
  }
}

template <typename RingCellId>
void balance_ring_svec_tangent(
    const RingCellId& cell_id,
    const int ring_left_corner,
    const int ring_right_corner,
    const int zone_count,
    const double* ring_node_r,
    const double* ring_node_z,
    std::vector<double>& svec_r,
    std::vector<double>& svec_z,
    const int corner_stride) {
  if (zone_count <= 1) {
    return;
  }

  for (int k = 1; k < zone_count; ++k) {
    const double R = ring_node_r[k];
    const double Z = ring_node_z[k];
    const double s = std::hypot(R, Z);
    if (!(s > 0.0)) {
      continue;
    }

    const double tangent_r = Z / s;
    const double tangent_z = -R / s;
    const int left_cell = cell_id(k - 1);
    const int right_cell = cell_id(k);
    const int left_idx = left_cell * corner_stride + ring_right_corner;
    const int right_idx = right_cell * corner_stride + ring_left_corner;
    const double tangent_sum =
        (svec_r[static_cast<std::size_t>(left_idx)] +
         svec_r[static_cast<std::size_t>(right_idx)]) *
            tangent_r +
        (svec_z[static_cast<std::size_t>(left_idx)] +
         svec_z[static_cast<std::size_t>(right_idx)]) *
            tangent_z;
    const double correction_r = 0.5 * tangent_sum * tangent_r;
    const double correction_z = 0.5 * tangent_sum * tangent_z;
    svec_r[static_cast<std::size_t>(left_idx)] -= correction_r;
    svec_z[static_cast<std::size_t>(left_idx)] -= correction_z;
    svec_r[static_cast<std::size_t>(right_idx)] -= correction_r;
    svec_z[static_cast<std::size_t>(right_idx)] -= correction_z;
  }
}

void balance_multiblock_outer_shell_svec_tangent(
    const MeshTopology& topo,
    const MultiBlockTopology& mb,
    const std::vector<double>& host_node_r,
    const std::vector<double>& host_node_z,
    std::vector<double>& svec_r,
    std::vector<double>& svec_z,
    const int corner_stride) {
  if (topo.nr <= 0 || topo.nz <= 1) {
    return;
  }
  TENRYU_ASSERT(host_node_r.size() == static_cast<std::size_t>(topo.n_nodes) &&
                    host_node_z.size() == static_cast<std::size_t>(topo.n_nodes),
                "multiblock Svec balance requires host node coordinates");
  TENRYU_ASSERT(
      svec_r.size() == static_cast<std::size_t>(topo.n_cells) *
                           static_cast<std::size_t>(corner_stride) &&
          svec_z.size() == static_cast<std::size_t>(topo.n_cells) *
                               static_cast<std::size_t>(corner_stride),
      "multiblock Svec balance requires configured Svec slots per cell");
  TENRYU_ASSERT(mb.n_cells_shell == topo.nr * topo.nz,
                "multiblock Svec balance shell cell count mismatch");
  TENRYU_ASSERT(mb.n_nodes_shell == (topo.nr + 1) * (topo.nz + 1),
                "multiblock Svec balance shell node count mismatch");

  const int shell_cell_offset = mb.n_cells_core + mb.n_cells_bridge;
  const int shell_node_offset = mb.n_nodes_core + mb.n_nodes_bridge_interior;
  const int outer_cell_q = topo.nr - 1;
  const int outer_node_q = topo.nr;
  const int angular_stride = topo.nz + 1;
  const int outer_cell_offset =
      shell_cell_offset + outer_cell_q * topo.nz;
  const int outer_node_offset =
      shell_node_offset + outer_node_q * angular_stride;
  const auto outer_cell_id =
      [outer_cell_offset](const int k) {
        return outer_cell_offset + k;
      };
  balance_ring_svec_tangent(
      outer_cell_id, 1, 2, topo.nz,
      host_node_r.data() + outer_node_offset,
      host_node_z.data() + outer_node_offset,
      svec_r, svec_z, corner_stride);
}

void balance_pentagon_belt_shell_svec_tangent(
    const MeshTopology& topo,
    const MultiBlockTopology& mb,
    const std::vector<double>& host_node_r,
    const std::vector<double>& host_node_z,
    std::vector<double>& svec_r,
    std::vector<double>& svec_z,
    const int corner_stride) {
  TENRYU_ASSERT(!mb.blocks.empty() &&
                    mb.blocks.front().role == BlockRole::PENTAGON_BELT,
                "pentagon-belt Svec balance requires belt blocks");
  TENRYU_ASSERT(host_node_r.size() == static_cast<std::size_t>(topo.n_nodes) &&
                    host_node_z.size() == static_cast<std::size_t>(topo.n_nodes),
                "pentagon-belt Svec balance requires host node coordinates");
  TENRYU_ASSERT(
      svec_r.size() == static_cast<std::size_t>(topo.n_cells) *
                           static_cast<std::size_t>(corner_stride) &&
          svec_z.size() == static_cast<std::size_t>(topo.n_cells) *
                               static_cast<std::size_t>(corner_stride),
      "pentagon-belt Svec balance requires configured Svec slots per cell");

  const int outer_zone_count = topo.nz;
  const int outer_cell_offset = topo.n_cells - outer_zone_count;
  const int outer_node_offset = topo.n_nodes - (outer_zone_count + 1);
  const auto outer_cell_id =
      [outer_cell_offset](const int k) {
        return outer_cell_offset + k;
      };
  balance_ring_svec_tangent(
      outer_cell_id, 1, 2, outer_zone_count,
      host_node_r.data() + outer_node_offset,
      host_node_z.data() + outer_node_offset,
      svec_r, svec_z, corner_stride);

  const int inner_zone_count = mb.blocks.front().n_j_cells;
  const auto inner_cell_id =
      [](const int k) {
        return k;
      };
  balance_ring_svec_tangent(
      inner_cell_id, 0, 3, inner_zone_count,
      host_node_r.data(), host_node_z.data(),
      svec_r, svec_z, corner_stride);
}

void apply_button_center_geometry(Mesh& mesh) {
  if (!mesh.button_center || !mesh.button_center->enabled) {
    return;
  }
  TENRYU_ASSERT(mesh.dim == 2,
                "button center geometry requires a 2D mesh");
  TENRYU_ASSERT(mesh.logical == LogicalMesh2D::SphericalPolarHalfplane,
                "button center geometry requires spherical_polar_halfplane");
  TENRYU_ASSERT(!mesh.topo.multiblock.has_value(),
                "button center geometry is single-block only");
  TENRYU_ASSERT(mesh.cell_vol.size() ==
                    static_cast<std::size_t>(mesh.topo.n_cells),
                "button center geometry requires per-cell volumes");
  TENRYU_ASSERT(mesh.cell_area.size() ==
                    static_cast<std::size_t>(mesh.topo.n_cells),
                "button center geometry requires per-cell areas");
  TENRYU_ASSERT(mesh.cell_centroid_r.size() ==
                    static_cast<std::size_t>(mesh.topo.n_cells) &&
                    mesh.cell_centroid_z.size() ==
                        static_cast<std::size_t>(mesh.topo.n_cells),
                "button center geometry requires per-cell centroids");
  TENRYU_ASSERT(
      mesh.cell_Svec_r.size() ==
              static_cast<std::size_t>(mesh.topo.n_cells) *
                  static_cast<std::size_t>(mesh.corner_stride) &&
          mesh.cell_Svec_z.size() ==
              static_cast<std::size_t>(mesh.topo.n_cells) *
                  static_cast<std::size_t>(mesh.corner_stride),
      "button center geometry requires configured Svec slots per cell");

  std::vector<double> r;
  std::vector<double> z;
  load_button_polygon_nodes(mesh, r, z);
  const int nverts = static_cast<int>(r.size());
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  rz_polygon_area_centroid(r.data(), z.data(), nverts,
                           &centroid_r, &centroid_z);
  const int c_button = mesh.topo.cell_index(0, 0);
  mesh.cell_vol[static_cast<std::size_t>(c_button)] =
      kPolarButtonOrientationSign *
      rz_polygon_volume_exact(r.data(), z.data(), nverts);
  mesh.cell_area[static_cast<std::size_t>(c_button)] =
      0.5 * kPolarButtonOrientationSign *
      rz_polygon_area2(r.data(), z.data(), nverts);
  mesh.cell_centroid_r[static_cast<std::size_t>(c_button)] = centroid_r;
  mesh.cell_centroid_z[static_cast<std::size_t>(c_button)] = centroid_z;
  for (int k = 0; k < 4; ++k) {
    const std::size_t idx =
        static_cast<std::size_t>(c_button) *
            static_cast<std::size_t>(mesh.corner_stride) +
        static_cast<std::size_t>(k);
    mesh.cell_Svec_r[idx] = 0.0;
    mesh.cell_Svec_z[idx] = 0.0;
  }

  std::vector<double> node_r;
  std::vector<double> node_z;
  load_all_node_coordinates(mesh, node_r, node_z);
  for (int c = 0; c < mesh.topo.n_cells; ++c) {
    if (mesh.is_button_cell(c) || mesh.is_dormant_cell(c)) {
      continue;
    }
    const int i = c / mesh.topo.nz;
    const int j = c - i * mesh.topo.nz;
    const int nodes[4] = {
        mesh.topo.node_index(i, j),
        mesh.topo.node_index(i + 1, j),
        mesh.topo.node_index(i + 1, j + 1),
        mesh.topo.node_index(i, j + 1),
    };
    double cell_r[4] = {0.0, 0.0, 0.0, 0.0};
    double cell_z[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
      const std::size_t n_u = static_cast<std::size_t>(nodes[k]);
      cell_r[k] = node_r[n_u];
      cell_z[k] = node_z[n_u];
    }

    double cell_centroid_r = 0.0;
    double cell_centroid_z = 0.0;
    rz_polygon_area_centroid(cell_r, cell_z, 4, &cell_centroid_r,
                             &cell_centroid_z);

    const std::size_t c_u = static_cast<std::size_t>(c);
    mesh.cell_vol[c_u] =
        kPolarButtonOrientationSign * rz_polygon_volume_exact(cell_r, cell_z, 4);
    mesh.cell_area[c_u] =
        0.5 * kPolarButtonOrientationSign * rz_polygon_area2(cell_r, cell_z, 4);
    mesh.cell_centroid_r[c_u] = cell_centroid_r;
    mesh.cell_centroid_z[c_u] = cell_centroid_z;

    const std::size_t base =
        c_u * static_cast<std::size_t>(mesh.corner_stride);
    for (int k = 0; k < 4; ++k) {
      double sr = 0.0;
      double sz = 0.0;
      rz_polygon_svec(cell_r, cell_z, 4, k, cell_centroid_r, cell_centroid_z,
                      kPolarButtonOrientationSign, &sr, &sz);
      const std::size_t idx = base + static_cast<std::size_t>(k);
      mesh.cell_Svec_r[idx] = sr;
      mesh.cell_Svec_z[idx] = sz;
    }
  }

  for (int c = 0; c < mesh.topo.n_cells; ++c) {
    if (!mesh.is_dormant_cell(c)) {
      continue;
    }
    const std::size_t c_u = static_cast<std::size_t>(c);
    mesh.cell_vol[c_u] = 0.0;
    mesh.cell_area[c_u] = 0.0;
    mesh.cell_centroid_r[c_u] = 0.0;
    mesh.cell_centroid_z[c_u] = 0.0;
    for (int k = 0; k < 4; ++k) {
      const std::size_t idx =
          c_u * static_cast<std::size_t>(mesh.corner_stride) +
          static_cast<std::size_t>(k);
      mesh.cell_Svec_r[idx] = 0.0;
      mesh.cell_Svec_z[idx] = 0.0;
    }
  }
}

void assert_trifan_cap_shared_node_maps(const MultiblockIndexing& idx) {
  TENRYU_ASSERT(idx.has_trifan_cap,
                "trifan cap shared-node checks require cap indexing");
  for (int k = 0; k <= idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::NORTH_FAN, 0, k) ==
                      cap_ring_node_id(idx, idx.n_cap, k),
                  "north fan inner seam must resolve to cap outer ring");
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::SOUTH_FAN, 0, k) ==
                      cap_ring_node_id(idx, idx.n_cap, 3 * idx.n_c + k),
                  "south fan inner seam must resolve to cap outer ring");
  }
  for (int k = 0; k <= 2 * idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::EAST_FAN, 0, k) ==
                      cap_ring_node_id(idx, idx.n_cap, idx.n_c + k),
                  "east fan inner seam must resolve to cap outer ring");
  }
  for (int l = 0; l <= idx.n_b; ++l) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::NORTH_FAN, l, idx.n_c) ==
                      fan_node_id(idx, BlockRole::EAST_FAN, l, 0),
                  "north-east fan diagonal seam must share node ids");
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::EAST_FAN, l, 2 * idx.n_c) ==
                      fan_node_id(idx, BlockRole::SOUTH_FAN, l, 0),
                  "east-south fan diagonal seam must share node ids");
  }
  for (int k = 0; k <= idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::NORTH_FAN, idx.n_b, k) ==
                      shell_node_id(idx, 0, k),
                  "north fan outer seam must resolve to shell inner ring");
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::SOUTH_FAN, idx.n_b, k) ==
                      shell_node_id(idx, 0, 3 * idx.n_c + k),
                  "south fan outer seam must resolve to shell inner ring");
  }
  for (int k = 0; k <= 2 * idx.n_c; ++k) {
    TENRYU_ASSERT(fan_node_id(idx, BlockRole::EAST_FAN, idx.n_b, k) ==
                      shell_node_id(idx, 0, idx.n_c + k),
                  "east fan outer seam must resolve to shell inner ring");
  }
}

}  // namespace

FullAxisNodeChain build_full_axis_node_chain(
    const Mesh& mesh,
    const std::vector<double>& corner_mass) {
  FullAxisNodeChain chain;
  if (!mesh.topo.multiblock.has_value()) {
    return chain;
  }
  const MultiBlockTopology& mb = *mesh.topo.multiblock;
  if (mb.block_count != 5) {
    return chain;
  }

  const int n_nodes = mesh.topo.n_nodes;
  const int n_cells = mesh.topo.n_cells;
  TENRYU_ASSERT(n_nodes >= 0 && n_cells >= 0,
                "axis chain requires non-negative mesh sizes");
  TENRYU_ASSERT(mesh.topo.node_flags.size() ==
                    static_cast<std::size_t>(n_nodes),
                "axis chain requires node_flags for every node");
  TENRYU_ASSERT(
      corner_mass.size() ==
          static_cast<std::size_t>(n_cells) *
              static_cast<std::size_t>(mesh.corner_stride),
      "axis chain requires configured corner masses per cell");
  TENRYU_ASSERT(mesh.node_z != nullptr,
                "axis chain requires mesh.node_z coordinates");

  std::vector<double> host_z(static_cast<std::size_t>(n_nodes), 0.0);
  cuda_check(cudaMemcpy(host_z.data(),
                        mesh.node_z,
                        host_z.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "axis chain cudaMemcpy node_z failed");

  const bool use_cell_nverts =
      mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells) &&
      std::any_of(mesh.cell_nverts.begin(),
                  mesh.cell_nverts.end(),
                  [](const std::uint8_t nverts) { return nverts == 3U; });
  const tenryu::hydro::ReverseCellNodeCSR reverse_csr =
      tenryu::hydro::build_reverse_cell_node_csr(
          mb, n_nodes, use_cell_nverts ? &mesh.cell_nverts : nullptr,
          mesh.corner_stride);
  std::vector<double> lumped_mass(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    const int off = reverse_csr.node_offsets[static_cast<std::size_t>(n)];
    const int next =
        reverse_csr.node_offsets[static_cast<std::size_t>(n) + 1U];
    double sum = 0.0;
    for (int p = off; p < next; ++p) {
      const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
      const int corner =
          reverse_csr.node_corners[static_cast<std::size_t>(p)];
      TENRYU_ASSERT(c >= 0 && c < n_cells,
                    "axis chain reverse CSR cell out of range");
      TENRYU_ASSERT(corner >= 0 && corner < 4,
                    "axis chain reverse CSR corner out of range");
      const double m =
          corner_mass[static_cast<std::size_t>(c) *
                          static_cast<std::size_t>(mesh.corner_stride) +
                      static_cast<std::size_t>(corner)];
      TENRYU_ASSERT(std::isfinite(m) && m >= 0.0,
                    "axis chain corner mass must be finite non-negative");
      sum += m;
    }
    lumped_mass[static_cast<std::size_t>(n)] = sum;
  }

  std::vector<std::pair<double, int>> sorted_nodes;
  sorted_nodes.reserve(static_cast<std::size_t>(n_nodes));
  for (int n = 0; n < n_nodes; ++n) {
    const std::uint8_t flags =
        mesh.topo.node_flags[static_cast<std::size_t>(n)];
    if ((flags & (NODE_AXIS | NODE_POLE_AXIS)) == 0U) {
      continue;
    }
    const double z = host_z[static_cast<std::size_t>(n)];
    TENRYU_ASSERT(std::isfinite(z), "axis chain node z must be finite");
    sorted_nodes.emplace_back(z, n);
  }

  std::sort(sorted_nodes.begin(),
            sorted_nodes.end(),
            [](const std::pair<double, int>& a,
               const std::pair<double, int>& b) {
              if (a.first != b.first) {
                return a.first < b.first;
              }
              return a.second < b.second;
            });

  chain.node_ids.reserve(sorted_nodes.size());
  chain.z.reserve(sorted_nodes.size());
  chain.lumped_mass.reserve(sorted_nodes.size());
  for (const auto& entry : sorted_nodes) {
    const int n = entry.second;
    chain.node_ids.push_back(n);
    chain.z.push_back(entry.first);
    chain.lumped_mass.push_back(lumped_mass[static_cast<std::size_t>(n)]);
  }
  return chain;
}

Mesh::~Mesh() {
  release_persistent_geometry_buffers_();
}

Mesh::Mesh(const Mesh& other)
    : topo(other.topo),
      node_r(other.node_r),
      node_z(other.node_z),
      cell_vol(other.cell_vol),
      cell_area(other.cell_area),
      cell_centroid_r(other.cell_centroid_r),
      cell_centroid_z(other.cell_centroid_z),
      corner_stride(other.corner_stride),
      cell_Svec_r(other.cell_Svec_r),
      cell_Svec_z(other.cell_Svec_z),
      cell_nverts(other.cell_nverts),
      edge_tags(other.edge_tags),
      dim(other.dim),
      geometry_code(other.geometry_code),
      logical(other.logical),
      polar_center_treatment(other.polar_center_treatment),
      multiblock_outer_svec_tangent_balance(
          other.multiblock_outer_svec_tangent_balance),
      button_center(other.button_center) {
  cell_centroid_r_device.copy_from_device(other.cell_centroid_r_device);
  multiblock_cell_node_csr_offsets.copy_from_device(
      other.multiblock_cell_node_csr_offsets);
  multiblock_cell_node_csr_indices.copy_from_device(
      other.multiblock_cell_node_csr_indices);
  multiblock_reverse_csr_node_offsets.copy_from_device(
      other.multiblock_reverse_csr_node_offsets);
  multiblock_reverse_csr_node_cells.copy_from_device(
      other.multiblock_reverse_csr_node_cells);
  multiblock_reverse_csr_node_corners.copy_from_device(
      other.multiblock_reverse_csr_node_corners);
}

Mesh& Mesh::operator=(const Mesh& other) {
  if (this != &other) {
    release_persistent_geometry_buffers_();
    topo = other.topo;
    node_r = other.node_r;
    node_z = other.node_z;
    cell_vol = other.cell_vol;
    cell_area = other.cell_area;
    cell_centroid_r = other.cell_centroid_r;
    cell_centroid_r_device.copy_from_device(other.cell_centroid_r_device);
    cell_centroid_z = other.cell_centroid_z;
    corner_stride = other.corner_stride;
    cell_Svec_r = other.cell_Svec_r;
    cell_Svec_z = other.cell_Svec_z;
    cell_nverts = other.cell_nverts;
    edge_tags = other.edge_tags;
    multiblock_cell_node_csr_offsets.copy_from_device(
        other.multiblock_cell_node_csr_offsets);
    multiblock_cell_node_csr_indices.copy_from_device(
        other.multiblock_cell_node_csr_indices);
    multiblock_reverse_csr_node_offsets.copy_from_device(
        other.multiblock_reverse_csr_node_offsets);
    multiblock_reverse_csr_node_cells.copy_from_device(
        other.multiblock_reverse_csr_node_cells);
    multiblock_reverse_csr_node_corners.copy_from_device(
        other.multiblock_reverse_csr_node_corners);
    dim = other.dim;
    geometry_code = other.geometry_code;
    logical = other.logical;
    polar_center_treatment = other.polar_center_treatment;
    multiblock_outer_svec_tangent_balance =
        other.multiblock_outer_svec_tangent_balance;
    button_center = other.button_center;
  }
  return *this;
}

Mesh::Mesh(Mesh&& other) noexcept
    : topo(std::move(other.topo)),
      node_r(other.node_r),
      node_z(other.node_z),
      cell_vol(std::move(other.cell_vol)),
      cell_area(std::move(other.cell_area)),
      cell_centroid_r(std::move(other.cell_centroid_r)),
      cell_centroid_r_device(std::move(other.cell_centroid_r_device)),
      cell_centroid_z(std::move(other.cell_centroid_z)),
      corner_stride(other.corner_stride),
      cell_Svec_r(std::move(other.cell_Svec_r)),
      cell_Svec_z(std::move(other.cell_Svec_z)),
      cell_nverts(std::move(other.cell_nverts)),
      edge_tags(std::move(other.edge_tags)),
      multiblock_cell_node_csr_offsets(
          std::move(other.multiblock_cell_node_csr_offsets)),
      multiblock_cell_node_csr_indices(
          std::move(other.multiblock_cell_node_csr_indices)),
      multiblock_reverse_csr_node_offsets(
          std::move(other.multiblock_reverse_csr_node_offsets)),
      multiblock_reverse_csr_node_cells(
          std::move(other.multiblock_reverse_csr_node_cells)),
      multiblock_reverse_csr_node_corners(
          std::move(other.multiblock_reverse_csr_node_corners)),
      dim(other.dim),
      geometry_code(other.geometry_code),
      logical(other.logical),
      polar_center_treatment(other.polar_center_treatment),
      multiblock_outer_svec_tangent_balance(
          other.multiblock_outer_svec_tangent_balance),
      button_center(std::move(other.button_center)),
      d_cell_vol_persistent_(other.d_cell_vol_persistent_),
      d_cell_centroid_r_persistent_(other.d_cell_centroid_r_persistent_),
      d_cell_centroid_z_persistent_(other.d_cell_centroid_z_persistent_),
      d_persistent_capacity_(other.d_persistent_capacity_) {
  other.node_r = nullptr;
  other.node_z = nullptr;
  other.d_cell_vol_persistent_ = nullptr;
  other.d_cell_centroid_r_persistent_ = nullptr;
  other.d_cell_centroid_z_persistent_ = nullptr;
  other.d_persistent_capacity_ = 0;
}

Mesh& Mesh::operator=(Mesh&& other) noexcept {
  if (this != &other) {
    release_persistent_geometry_buffers_();
    topo = std::move(other.topo);
    node_r = other.node_r;
    node_z = other.node_z;
    cell_vol = std::move(other.cell_vol);
    cell_area = std::move(other.cell_area);
    cell_centroid_r = std::move(other.cell_centroid_r);
    cell_centroid_r_device = std::move(other.cell_centroid_r_device);
    cell_centroid_z = std::move(other.cell_centroid_z);
    corner_stride = other.corner_stride;
    cell_Svec_r = std::move(other.cell_Svec_r);
    cell_Svec_z = std::move(other.cell_Svec_z);
    cell_nverts = std::move(other.cell_nverts);
    edge_tags = std::move(other.edge_tags);
    multiblock_cell_node_csr_offsets =
        std::move(other.multiblock_cell_node_csr_offsets);
    multiblock_cell_node_csr_indices =
        std::move(other.multiblock_cell_node_csr_indices);
    multiblock_reverse_csr_node_offsets =
        std::move(other.multiblock_reverse_csr_node_offsets);
    multiblock_reverse_csr_node_cells =
        std::move(other.multiblock_reverse_csr_node_cells);
    multiblock_reverse_csr_node_corners =
        std::move(other.multiblock_reverse_csr_node_corners);
    dim = other.dim;
    geometry_code = other.geometry_code;
    logical = other.logical;
    polar_center_treatment = other.polar_center_treatment;
    multiblock_outer_svec_tangent_balance =
        other.multiblock_outer_svec_tangent_balance;
    button_center = std::move(other.button_center);
    d_cell_vol_persistent_ = other.d_cell_vol_persistent_;
    d_cell_centroid_r_persistent_ = other.d_cell_centroid_r_persistent_;
    d_cell_centroid_z_persistent_ = other.d_cell_centroid_z_persistent_;
    d_persistent_capacity_ = other.d_persistent_capacity_;
    other.node_r = nullptr;
    other.node_z = nullptr;
    other.d_cell_vol_persistent_ = nullptr;
    other.d_cell_centroid_r_persistent_ = nullptr;
    other.d_cell_centroid_z_persistent_ = nullptr;
    other.d_persistent_capacity_ = 0;
  }
  return *this;
}

void Mesh::release_persistent_geometry_buffers_() {
  if (d_cell_centroid_z_persistent_ != nullptr) {
    cuda_check(cudaFree(d_cell_centroid_z_persistent_),
               "Mesh persistent geometry cudaFree cell_centroid_z failed");
    d_cell_centroid_z_persistent_ = nullptr;
  }
  if (d_cell_centroid_r_persistent_ != nullptr) {
    cuda_check(cudaFree(d_cell_centroid_r_persistent_),
               "Mesh persistent geometry cudaFree cell_centroid_r failed");
    d_cell_centroid_r_persistent_ = nullptr;
  }
  if (d_cell_vol_persistent_ != nullptr) {
    cuda_check(cudaFree(d_cell_vol_persistent_),
               "Mesh persistent geometry cudaFree cell_vol failed");
    d_cell_vol_persistent_ = nullptr;
  }
  d_persistent_capacity_ = 0;
}

void Mesh::ensure_persistent_geometry_buffers_(const std::size_t n_cells) {
  if (n_cells <= d_persistent_capacity_) {
    return;
  }

  release_persistent_geometry_buffers_();
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_vol_persistent_),
                        n_cells * sizeof(double)),
             "Mesh persistent geometry cudaMalloc cell_vol failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_centroid_r_persistent_),
                        n_cells * sizeof(double)),
             "Mesh persistent geometry cudaMalloc cell_centroid_r failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_centroid_z_persistent_),
                        n_cells * sizeof(double)),
             "Mesh persistent geometry cudaMalloc cell_centroid_z failed");
  d_persistent_capacity_ = n_cells;
}

void Mesh::recompute_geometry_device_only(double* state_vol) {
  TENRYU_ASSERT(dim == 1, "Mesh::recompute_geometry_device_only is 1D-only");
  TENRYU_ASSERT(node_r != nullptr,
                "Mesh.node_r must be bound before geometry recomputation");

  const std::size_t n_cells = static_cast<std::size_t>(topo.n_cells);
  if (n_cells == 0) {
    return;
  }
  TENRYU_ASSERT(state_vol != nullptr,
                "Mesh::recompute_geometry_device_only requires state_vol");

  ensure_persistent_geometry_buffers_(n_cells);
  const int blocks = (topo.nr + 255) / 256;
  recompute_geometry_1d_kernel<<<blocks, 256>>>(
      state_vol, d_cell_centroid_r_persistent_, d_cell_centroid_z_persistent_,
      node_r, topo.nr, geometry_code);
  cuda_check(cudaGetLastError(),
             "Mesh::recompute_geometry_device_only 1D kernel launch failed");
}

MeshGeometryResult Mesh::sync_device_geometry_to_host_checked(
    const double* state_vol,
    const MeshGeometryCheckOptions& opts) {
  TENRYU_ASSERT(dim == 1, "Mesh::sync_device_geometry_to_host is 1D-only");
  TENRYU_ASSERT(node_r != nullptr,
                "Mesh.node_r must be bound before geometry recomputation");

  const std::size_t n_cells = static_cast<std::size_t>(topo.n_cells);
  cell_vol.assign(n_cells, 0.0);
  cell_area.assign(n_cells, 0.0);
  cell_centroid_r.assign(n_cells, 0.0);
  cell_centroid_z.assign(n_cells, 0.0);
  cell_Svec_r.assign(
      n_cells * static_cast<std::size_t>(corner_stride), 0.0);
  cell_Svec_z.assign(
      n_cells * static_cast<std::size_t>(corner_stride), 0.0);

  if (n_cells == 0) {
    cell_centroid_r_device.reset(0);
    return MeshGeometryResult{};
  }
  TENRYU_ASSERT(state_vol != nullptr,
                "Mesh::sync_device_geometry_to_host requires state_vol");
  TENRYU_ASSERT(d_cell_vol_persistent_ != nullptr &&
                    d_cell_centroid_r_persistent_ != nullptr &&
                    d_cell_centroid_z_persistent_ != nullptr &&
                    d_persistent_capacity_ >= n_cells,
                "Mesh::sync_device_geometry_to_host requires device-only geometry first");

  std::vector<double> geo_pack(3 * static_cast<std::size_t>(n_cells));
  const double* srcs[3] = {state_vol, d_cell_centroid_r_persistent_,
                           d_cell_centroid_z_persistent_};
  core::pack_pull_fields(srcs, 3, n_cells, geo_pack.data(),
                         "mesh:sync_device_geometry:pull");
  std::memcpy(cell_vol.data(), geo_pack.data(), n_cells * sizeof(double));
  std::memcpy(cell_centroid_r.data(), geo_pack.data() + n_cells,
              n_cells * sizeof(double));
  std::memcpy(cell_centroid_z.data(), geo_pack.data() + 2 * n_cells,
              n_cells * sizeof(double));
  cell_centroid_r_device.reset(n_cells);
  cuda_check(cudaMemcpy(cell_centroid_r_device.data(), d_cell_centroid_r_persistent_,
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Mesh::sync_device_geometry_to_host centroid_r D2D failed");

  return validate_1d_geometry(*this, opts);
}

void Mesh::sync_device_geometry_to_host(const double* state_vol) {
  MeshGeometryCheckOptions opts{};
  opts.policy = MeshGeometryFailurePolicy::HardAssert;
  (void)sync_device_geometry_to_host_checked(state_vol, opts);
}

MeshGeometryResult Mesh::recompute_geometry_checked(
    const MeshGeometryCheckOptions& opts) {
  TENRYU_ASSERT(node_r != nullptr,
                "Mesh.node_r must be bound before geometry recomputation");

  const std::size_t n_cells = static_cast<std::size_t>(topo.n_cells);
  cell_vol.assign(n_cells, 0.0);
  cell_area.assign(n_cells, 0.0);
  cell_centroid_r.assign(n_cells, 0.0);
  cell_centroid_z.assign(n_cells, 0.0);
  cell_Svec_r.assign(
      n_cells * static_cast<std::size_t>(corner_stride), 0.0);
  cell_Svec_z.assign(
      n_cells * static_cast<std::size_t>(corner_stride), 0.0);

  if (n_cells == 0) {
    cell_centroid_r_device.reset(0);
    return MeshGeometryResult{};
  }
  const bool is_multiblock = topo.multiblock.has_value();

  double* d_cell_vol = nullptr;
  double* d_cell_area = nullptr;
  double* d_cell_centroid_r = nullptr;
  double* d_cell_centroid_z = nullptr;
  double* d_cell_Svec_r = nullptr;
  double* d_cell_Svec_z = nullptr;
  std::uint8_t* d_cell_nverts = nullptr;

  d_cell_vol = static_cast<double*>(core::device_scratch_acquire(
      "mesh:recompute_geometry_checked:d_cell_vol", n_cells * sizeof(double)));
  d_cell_centroid_r = static_cast<double*>(core::device_scratch_acquire(
      "mesh:recompute_geometry_checked:d_cell_centroid_r",
      n_cells * sizeof(double)));
  d_cell_centroid_z = static_cast<double*>(core::device_scratch_acquire(
      "mesh:recompute_geometry_checked:d_cell_centroid_z",
      n_cells * sizeof(double)));

  if (dim == 1) {
    TENRYU_ASSERT(!is_multiblock, "multiblock geometry recompute requires 2D");
    const int blocks = (topo.nr + 255) / 256;
    recompute_geometry_1d_kernel<<<blocks, 256>>>(
        d_cell_vol, d_cell_centroid_r, d_cell_centroid_z, node_r, topo.nr,
        geometry_code);
    cuda_check(cudaGetLastError(), "Mesh::recompute_geometry 1D kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "Mesh::recompute_geometry 1D kernel execution failed");

    cuda_check(cudaMemcpy(cell_vol.data(), d_cell_vol, n_cells * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy cell_vol failed");
    cuda_check(cudaMemcpy(cell_centroid_r.data(), d_cell_centroid_r,
                          n_cells * sizeof(double), cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy cell_centroid_r failed");
    cuda_check(cudaMemcpy(cell_centroid_z.data(), d_cell_centroid_z,
                          n_cells * sizeof(double), cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy cell_centroid_z failed");
    cell_centroid_r_device.reset(n_cells);
    cell_centroid_r_device.copy_from_host(cell_centroid_r);

    return validate_1d_geometry(*this, opts);
  }

  TENRYU_ASSERT(node_z != nullptr,
                "Mesh.node_z must be bound in 2D_RZ geometry recomputation");

  d_cell_area = static_cast<double*>(core::device_scratch_acquire(
      "mesh:recompute_geometry_checked:d_cell_area", n_cells * sizeof(double)));
  d_cell_Svec_r = static_cast<double*>(core::device_scratch_acquire(
      "mesh:recompute_geometry_checked:d_cell_Svec_r",
      n_cells * static_cast<std::size_t>(corner_stride) * sizeof(double)));
  d_cell_Svec_z = static_cast<double*>(core::device_scratch_acquire(
      "mesh:recompute_geometry_checked:d_cell_Svec_z",
      n_cells * static_cast<std::size_t>(corner_stride) * sizeof(double)));

  if (is_multiblock) {
    TENRYU_ASSERT(dim == 2, "multiblock geometry recompute requires 2D");
    TENRYU_ASSERT(topo.multiblock.has_value(),
                  "multiblock geometry recompute requires CSR topology");
    TENRYU_ASSERT(cell_nverts.empty() || cell_nverts.size() == n_cells,
                  "multiblock geometry requires empty or per-cell cell_nverts");
    bool use_cell_nverts = false;
    for (const std::uint8_t nverts : cell_nverts) {
      TENRYU_ASSERT(nverts >= 3U && nverts <= 8U,
                    "multiblock geometry requires 3- to 8-vertex cells");
      use_cell_nverts = use_cell_nverts || nverts != 4U;
    }
    const auto& mb = *topo.multiblock;
    TENRYU_ASSERT(mb.cell_node_csr_offsets.size() == n_cells + 1U,
                  "multiblock cell-node CSR offset size mismatch");
    TENRYU_ASSERT(mb.cell_node_csr_offsets.front() == 0,
                  "multiblock cell-node CSR offsets must start at zero");
    const int n_corner_slots_i = mb.cell_node_csr_offsets.back();
    TENRYU_ASSERT(n_corner_slots_i >= 0,
                  "multiblock cell-node CSR slot count must be non-negative");
    const std::size_t n_corner_slots =
        static_cast<std::size_t>(n_corner_slots_i);
    TENRYU_ASSERT(mb.cell_node_csr_indices.size() == n_corner_slots,
                  "multiblock cell-node CSR index size mismatch");
    TENRYU_ASSERT(mb.cell_block_id.size() == n_cells,
                  "multiblock cell_block_id size mismatch");
    TENRYU_ASSERT(mb.cell_orientation_sign.size() == n_cells,
                  "multiblock cell_orientation_sign size mismatch");
    for (int c = 0; c < topo.n_cells; ++c) {
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int next =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(c + 1)];
      TENRYU_ASSERT(off >= 0 && next >= off,
                    "multiblock cell-node CSR offsets must be monotone");
      TENRYU_ASSERT(next - off == corner_stride,
                    "multiblock cell-node CSR width must match mesh stride");
      TENRYU_ASSERT(static_cast<std::size_t>(next) <= n_corner_slots,
                    "multiblock cell-node CSR offset exceeds index storage");
      const int block_id = mb.cell_block_id[static_cast<std::size_t>(c)];
      TENRYU_ASSERT(block_id >= 0 && block_id < mb.block_count,
                    "multiblock cell block id out of range");
      const int orientation_sign =
          mb.cell_orientation_sign[static_cast<std::size_t>(c)];
      TENRYU_ASSERT(orientation_sign == 1 || orientation_sign == -1,
                    "multiblock cell orientation sign must be +/-1");
      const int active_nverts =
          use_cell_nverts ? mesh_topo_cell_active_nverts(cell_nverts, c)
                          : kMeshTopoCellStorageSlots;
      for (int k = 0; k < active_nverts; ++k) {
        const int n =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        TENRYU_ASSERT(n >= 0 && n < topo.n_nodes,
                      "multiblock cell node index out of range");
      }
    }

    int* d_cell_node_csr_offsets = nullptr;
    int* d_cell_node_csr_indices = nullptr;
    int* d_cell_orientation_sign = nullptr;
    d_cell_node_csr_offsets = static_cast<int*>(core::device_scratch_acquire(
        "mesh:recompute_geometry_checked:d_cell_node_csr_offsets",
        (n_cells + 1U) * sizeof(int)));
    d_cell_node_csr_indices = static_cast<int*>(core::device_scratch_acquire(
        "mesh:recompute_geometry_checked:d_cell_node_csr_indices",
        n_corner_slots * sizeof(int)));
    d_cell_orientation_sign = static_cast<int*>(core::device_scratch_acquire(
        "mesh:recompute_geometry_checked:d_cell_orientation_sign",
        n_cells * sizeof(int)));
    if (use_cell_nverts) {
      d_cell_nverts = static_cast<std::uint8_t*>(core::device_scratch_acquire(
          "mesh:recompute_geometry_checked:d_cell_nverts_multiblock",
          n_cells * sizeof(std::uint8_t)));
      cuda_check(cudaMemcpy(d_cell_nverts,
                            cell_nverts.data(),
                            n_cells * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice),
                 "Mesh::recompute_geometry memcpy multiblock cell_nverts failed");
    }
    cuda_check(cudaMemcpy(d_cell_node_csr_offsets,
                          mb.cell_node_csr_offsets.data(),
                          (n_cells + 1U) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "Mesh::recompute_geometry memcpy multiblock CSR offsets failed");
    cuda_check(cudaMemcpy(d_cell_node_csr_indices,
                          mb.cell_node_csr_indices.data(),
                          n_corner_slots * sizeof(int),
                          cudaMemcpyHostToDevice),
               "Mesh::recompute_geometry memcpy multiblock CSR indices failed");
    cuda_check(cudaMemcpy(d_cell_orientation_sign,
                          mb.cell_orientation_sign.data(),
                          n_cells * sizeof(int),
                          cudaMemcpyHostToDevice),
               "Mesh::recompute_geometry memcpy multiblock cell_orientation_sign failed");

    const int blocks = (topo.n_cells + 255) / 256;
    recompute_geometry_2d_kernel_multiblock_csr<<<blocks, 256>>>(
        d_cell_vol, d_cell_area, d_cell_centroid_r, d_cell_centroid_z,
        d_cell_Svec_r, d_cell_Svec_z, node_r, node_z,
        d_cell_node_csr_offsets, d_cell_node_csr_indices,
        d_cell_orientation_sign, d_cell_nverts,
        topo.n_cells, corner_stride);
    cuda_check(cudaGetLastError(),
               "Mesh::recompute_geometry multiblock kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "Mesh::recompute_geometry multiblock kernel execution failed");

    cuda_check(cudaMemcpy(cell_vol.data(), d_cell_vol, n_cells * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy cell_vol failed");
    cuda_check(cudaMemcpy(cell_area.data(), d_cell_area, n_cells * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy cell_area failed");
    cuda_check(cudaMemcpy(cell_centroid_r.data(), d_cell_centroid_r,
                          n_cells * sizeof(double), cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy cell_centroid_r failed");
    cuda_check(cudaMemcpy(cell_centroid_z.data(), d_cell_centroid_z,
                          n_cells * sizeof(double), cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy cell_centroid_z failed");
    cell_centroid_r_device.reset(n_cells);
    cell_centroid_r_device.copy_from_host(cell_centroid_r);
    cuda_check(cudaMemcpy(cell_Svec_r.data(), d_cell_Svec_r,
                          n_cells * static_cast<std::size_t>(corner_stride) *
                              sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy cell_Svec_r failed");
    cuda_check(cudaMemcpy(cell_Svec_z.data(), d_cell_Svec_z,
                          n_cells * static_cast<std::size_t>(corner_stride) *
                              sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy cell_Svec_z failed");
    std::vector<double> host_node_r(static_cast<std::size_t>(topo.n_nodes), 0.0);
    std::vector<double> host_node_z(static_cast<std::size_t>(topo.n_nodes), 0.0);
    cuda_check(cudaMemcpy(host_node_r.data(), node_r,
                          static_cast<std::size_t>(topo.n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy multiblock node_r failed");
    cuda_check(cudaMemcpy(host_node_z.data(), node_z,
                          static_cast<std::size_t>(topo.n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "Mesh::recompute_geometry memcpy multiblock node_z failed");
    if (logical != LogicalMesh2D::ConeShell &&
        multiblock_outer_svec_tangent_balance) {
      if (!mb.blocks.empty() &&
          mb.blocks.front().role == BlockRole::PENTAGON_BELT) {
        balance_pentagon_belt_shell_svec_tangent(
            topo, mb, host_node_r, host_node_z, cell_Svec_r, cell_Svec_z,
            corner_stride);
      } else {
        balance_multiblock_outer_shell_svec_tangent(
            topo, mb, host_node_r, host_node_z, cell_Svec_r, cell_Svec_z,
            corner_stride);
      }
    }

    return validate_multiblock_geometry(*this, opts);
  }

  const bool use_polar_tri_fan_geometry =
      (logical == LogicalMesh2D::SphericalPolarHalfplane ||
       logical == LogicalMesh2D::PolarInBox) &&
      polar_center_treatment == PolarCenterTreatment::TriFan;
  if (use_polar_tri_fan_geometry) {
    TENRYU_ASSERT(cell_nverts.size() == n_cells,
                  "tri_fan geometry requires cell_nverts size == n_cells");
    d_cell_nverts = static_cast<std::uint8_t*>(core::device_scratch_acquire(
        "mesh:recompute_geometry_checked:d_cell_nverts_trifan",
        n_cells * sizeof(std::uint8_t)));
    cuda_check(cudaMemcpy(d_cell_nverts,
                          cell_nverts.data(),
                          n_cells * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "Mesh::recompute_geometry memcpy cell_nverts failed");
  }

  const int blocks = (topo.n_cells + 255) / 256;
  if (use_polar_tri_fan_geometry) {
    recompute_geometry_2d_kernel_polar_tri_fan<<<blocks, 256>>>(
        d_cell_vol, d_cell_area, d_cell_centroid_r, d_cell_centroid_z,
        d_cell_Svec_r, d_cell_Svec_z, node_r, node_z, d_cell_nverts,
        topo.nr, topo.nz, corner_stride);
  } else if (logical == LogicalMesh2D::SphericalPolarHalfplane) {
    recompute_geometry_2d_kernel_polar<<<blocks, 256>>>(
        d_cell_vol, d_cell_area, d_cell_centroid_r, d_cell_centroid_z,
        d_cell_Svec_r, d_cell_Svec_z, node_r, node_z, topo.nr, topo.nz,
        corner_stride);
  } else if (logical == LogicalMesh2D::PolarInBox) {
    // PolarInBox uses the general RZ quadrilateral geometry. Its logical
    // theta index runs from the upper to the lower axis, opposite to the
    // rectangular z-index orientation.
    recompute_geometry_2d_kernel<-1><<<blocks, 256>>>(
        d_cell_vol, d_cell_area, d_cell_centroid_r, d_cell_centroid_z, d_cell_Svec_r,
        d_cell_Svec_z, node_r, node_z, topo.nr, topo.nz, corner_stride);
  } else {
    recompute_geometry_2d_kernel<+1><<<blocks, 256>>>(
        d_cell_vol, d_cell_area, d_cell_centroid_r, d_cell_centroid_z, d_cell_Svec_r,
        d_cell_Svec_z, node_r, node_z, topo.nr, topo.nz, corner_stride);
  }
  cuda_check(cudaGetLastError(), "Mesh::recompute_geometry 2D kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "Mesh::recompute_geometry 2D kernel execution failed");

  cuda_check(cudaMemcpy(cell_vol.data(), d_cell_vol, n_cells * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Mesh::recompute_geometry memcpy cell_vol failed");
  cuda_check(cudaMemcpy(cell_area.data(), d_cell_area, n_cells * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Mesh::recompute_geometry memcpy cell_area failed");
  cuda_check(cudaMemcpy(cell_centroid_r.data(), d_cell_centroid_r,
                        n_cells * sizeof(double), cudaMemcpyDeviceToHost),
             "Mesh::recompute_geometry memcpy cell_centroid_r failed");
  cuda_check(cudaMemcpy(cell_centroid_z.data(), d_cell_centroid_z,
                        n_cells * sizeof(double), cudaMemcpyDeviceToHost),
             "Mesh::recompute_geometry memcpy cell_centroid_z failed");
  cuda_check(cudaMemcpy(cell_Svec_r.data(), d_cell_Svec_r,
                        n_cells * static_cast<std::size_t>(corner_stride) *
                            sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Mesh::recompute_geometry memcpy cell_Svec_r failed");
  cuda_check(cudaMemcpy(cell_Svec_z.data(), d_cell_Svec_z,
                        n_cells * static_cast<std::size_t>(corner_stride) *
                            sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Mesh::recompute_geometry memcpy cell_Svec_z failed");
  if (logical == LogicalMesh2D::SphericalPolarHalfplane) {
    apply_button_center_geometry(*this);
  }
  cell_centroid_r_device.reset(n_cells);
  cell_centroid_r_device.copy_from_host(cell_centroid_r);

  return validate_2d_geometry(*this, opts);
}

void Mesh::recompute_geometry() {
  MeshGeometryCheckOptions opts{};
  opts.policy = MeshGeometryFailurePolicy::HardAssert;
  (void)recompute_geometry_checked(opts);
}

Mesh create_mesh(const tenryu::core::Config& cfg, tenryu::core::State& state) {
  if (cfg.mesh.topology_scheme ==
      tenryu::core::TopologyScheme::PENTAGON_BELT_SHELL) {
    return create_pentagon_belt_shell_mesh(cfg, state);
  }
  if (cfg.mesh.logical_mesh_2d == "cone_shell") {
    return create_cone_shell_mesh(cfg, state);
  }
  if (mesh_topo_is_multiblock(cfg.mesh)) {
    return create_multiblock_cart_core_polar_shell_mesh(cfg, state);
  }
  TENRYU_ASSERT(cfg.mesh.topology_scheme ==
                    tenryu::core::TopologyScheme::SINGLE_BLOCK,
                "Mesh.topology_scheme must be single_block or supported multiblock");

  Mesh mesh;
  mesh.corner_stride =
      tenryu::core::corner_stride_for_scheme(cfg.mesh.topology_scheme);
  TENRYU_ASSERT(mesh.corner_stride == state.corner_stride,
                "Mesh/state corner stride mismatch");
  mesh.dim = cfg.main.dim;
  const bool legacy_cylindrical_1d =
      cfg.main.dim == 1 && cfg.main.dimension == "1D_CYL";
  mesh.geometry_code =
      (legacy_cylindrical_1d || cfg.mesh.geometry_1d == "cylindrical")
          ? 1
          : ((cfg.mesh.geometry_1d == "planar") ? 2 : 0);
  mesh.logical = parse_logical_mesh_2d(cfg.mesh.logical_mesh_2d);
  mesh.polar_center_treatment =
      parse_polar_center_treatment(cfg.mesh.polar_center_treatment);

  TENRYU_ASSERT(mesh.dim == 1 || mesh.dim == 2,
                "Main.dim must be 1 or 2 for mesh creation");
  TENRYU_ASSERT(mesh.dim == 2 || mesh.logical == LogicalMesh2D::RectangularRZ,
                "Mesh.logical_mesh_2d is only valid for 2D_RZ meshes");
  TENRYU_ASSERT(mesh.logical == LogicalMesh2D::SphericalPolarHalfplane ||
                    mesh.polar_center_treatment == PolarCenterTreatment::Annular ||
                    (mesh.logical == LogicalMesh2D::PolarInBox &&
                     mesh.polar_center_treatment == PolarCenterTreatment::TriFan),
                "non-annular Mesh.polar_center_treatment requires "
                "spherical_polar_halfplane; the current polar_in_box scope uses general quads");
  if (mesh.polar_center_treatment == PolarCenterTreatment::Button) {
    mesh.button_center = ButtonCenterTopology{
        true,
        cfg.mesh.center_button_outer_node_ring,
    };
  } else {
    mesh.button_center.reset();
  }

  mesh.topo.nr = cfg.mesh.nr;
  mesh.topo.nz = (mesh.dim == 2) ? cfg.mesh.nz : 1;
  if (mesh.button_center && mesh.button_center->enabled) {
    TENRYU_ASSERT(mesh.button_center->outer_node_ring >= 1 &&
                      mesh.button_center->outer_node_ring < mesh.topo.nr,
                  "Mesh.center_button_outer_node_ring must satisfy 1 <= I_btn < Mesh.nr");
  }
  const std::int64_t n_cells_ll =
      static_cast<std::int64_t>(mesh.topo.nr) * static_cast<std::int64_t>(mesh.topo.nz);
  TENRYU_ASSERT(n_cells_ll > 0, "Mesh cell count must be positive");
  TENRYU_ASSERT(
      n_cells_ll <= static_cast<std::int64_t>(std::numeric_limits<int>::max()),
      "Mesh cell count exceeds int range");
  mesh.topo.n_cells = static_cast<int>(n_cells_ll);

  std::int64_t n_nodes_ll = 0;
  if (mesh.dim == 2) {
    const std::int64_t nr_nodes = static_cast<std::int64_t>(mesh.topo.nr) + 1;
    const std::int64_t nz_nodes = static_cast<std::int64_t>(mesh.topo.nz) + 1;
    TENRYU_ASSERT(
        nr_nodes > 0 && nz_nodes > 0 &&
            nr_nodes <= (std::numeric_limits<std::int64_t>::max() / nz_nodes),
        "Mesh node count overflow");
    n_nodes_ll = nr_nodes * nz_nodes;
  } else {
    n_nodes_ll = static_cast<std::int64_t>(mesh.topo.nr) + 1;
  }
  TENRYU_ASSERT(n_nodes_ll > 0, "Mesh node count must be positive");
  TENRYU_ASSERT(
      n_nodes_ll <= static_cast<std::int64_t>(std::numeric_limits<int>::max()),
      "Mesh node count exceeds int range");
  mesh.topo.n_nodes = static_cast<int>(n_nodes_ll);
  if (mesh.dim == 2) {
    mesh.cell_nverts.assign(static_cast<std::size_t>(mesh.topo.n_cells), 4U);
    if ((mesh.logical == LogicalMesh2D::SphericalPolarHalfplane ||
         mesh.logical == LogicalMesh2D::PolarInBox) &&
        mesh.polar_center_treatment == PolarCenterTreatment::TriFan) {
      for (int j = 0; j < mesh.topo.nz; ++j) {
        mesh.cell_nverts[static_cast<std::size_t>(mesh.topo.cell_index(0, j))] = 3U;
      }
    }
  } else {
    mesh.cell_nverts.clear();
  }

  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(mesh.topo.n_nodes),
                "State.x_r size does not match mesh node count");
  TENRYU_ASSERT(state.x_z.size() == static_cast<std::size_t>(mesh.topo.n_nodes),
                "State.x_z size does not match mesh node count");

  std::vector<double> r_nodes;
  const bool radial_explicit_nodes = !cfg.mesh.explicit_nodes.empty();
  const bool radial_grid_segments = !cfg.mesh.grid_segments.empty();
  if (radial_explicit_nodes &&
      mesh.logical != LogicalMesh2D::PolarInBox) {
    r_nodes = cfg.mesh.explicit_nodes;
    TENRYU_ASSERT(static_cast<int>(r_nodes.size()) == mesh.topo.nr + 1,
                  "Mesh.explicit_nodes count mismatch: got " +
                      std::to_string(r_nodes.size()) + " nodes, expected " +
                      std::to_string(mesh.topo.nr + 1));
    for (std::size_t k = 1; k < r_nodes.size(); ++k) {
      TENRYU_ASSERT(r_nodes[k] > r_nodes[k - 1],
                    "Mesh.explicit_nodes must be strictly increasing");
    }
  } else if (mesh.dim == 1 || radial_grid_segments) {
    std::vector<GridSegment> segments = cfg.mesh.grid_segments;
    GradingConfig grading = cfg.mesh.grading;
    if (segments.empty()) {
      GridSegment seg;
      seg.r_start = cfg.mesh.r_min;
      seg.r_end = cfg.mesh.r_max;
      seg.nr = mesh.topo.nr;
      segments.push_back(seg);
      grading.edge_ratio = std::nextafter(1.0, 0.0);
    }
    // exact_measure_v2 is 1D-only (validated at parse time); the measure
    // dimension follows the same geometry mapping as mesh.geometry_code.
    const int graded_measure_dim =
        (mesh.dim != 1)
            ? 0
            : ((cfg.main.dimension == "1D_CYL" ||
                cfg.mesh.geometry_1d == "cylindrical")
                   ? 2
                   : ((cfg.mesh.geometry_1d == "planar") ? 1 : 3));
    r_nodes =
        build_graded_nodes(segments, grading, mesh.dim == 2, graded_measure_dim);
    TENRYU_ASSERT(static_cast<int>(r_nodes.size()) == mesh.topo.nr + 1,
                  "Graded node count mismatch: got " +
                      std::to_string(r_nodes.size()) + " nodes, expected " +
                      std::to_string(mesh.topo.nr + 1));
  } else if (mesh.logical != LogicalMesh2D::PolarInBox) {
    r_nodes = build_uniform_nodes(cfg.mesh.r_min, cfg.mesh.r_max, mesh.topo.nr);
  }

  std::vector<double> host_x_r(state.x_r.size(), 0.0);
  std::vector<double> host_x_z(state.x_z.size(), 0.0);

  if (mesh.dim == 1) {
    for (int i = 0; i <= mesh.topo.nr; ++i) {
      const std::size_t idx = static_cast<std::size_t>(i);
      host_x_r[idx] = r_nodes[idx];
      host_x_z[idx] = 0.0;
    }
    set_1d_node_flags(mesh.topo);
    mesh.edge_tags.clear();
  } else if (mesh.logical == LogicalMesh2D::SphericalPolarHalfplane ||
             mesh.logical == LogicalMesh2D::PolarInBox) {
    const bool polar_in_box = mesh.logical == LogicalMesh2D::PolarInBox;
    TENRYU_ASSERT(cfg.main.dimension == "2D_RZ",
                  "polar-family Mesh.logical_mesh_2d requires Main.dimension='2D_RZ'");
    TENRYU_ASSERT(cfg.mesh.spherical_polar_s_max > 0.0,
                  "Mesh.spherical_polar_s_max must be > 0");
    const bool tri_fan =
        mesh.polar_center_treatment == PolarCenterTreatment::TriFan;
    const bool button =
        mesh.polar_center_treatment == PolarCenterTreatment::Button;
    if (!tri_fan && !button) {
      TENRYU_ASSERT(cfg.mesh.spherical_polar_kappa > 0.0,
                    "Mesh.spherical_polar_kappa must be > 0");
    }

    const int ns = polar_in_box ? cfg.mesh.polar_prefix_nr : mesh.topo.nr;
    const int ntheta = mesh.topo.nz;
    const double s_max = cfg.mesh.spherical_polar_s_max;
    const double center_z = polar_in_box ? cfg.mesh.box_center_z : 0.0;
    TENRYU_ASSERT(ns > 0 && ns <= mesh.topo.nr,
                  "polar prefix radial count must be positive and within Mesh.nr");

    std::vector<double> s_nodes(static_cast<std::size_t>(ns + 1), 0.0);
    const bool polar_explicit_nodes = !cfg.mesh.explicit_nodes.empty();
    const bool polar_grid_segments = !cfg.mesh.grid_segments.empty();
    if (polar_explicit_nodes || polar_grid_segments) {
      if (polar_explicit_nodes) {
        TENRYU_ASSERT(static_cast<int>(cfg.mesh.explicit_nodes.size()) == ns + 1,
                      "Mesh.explicit_nodes count mismatch for polar prefix: got " +
                          std::to_string(cfg.mesh.explicit_nodes.size()) +
                          " nodes, expected " + std::to_string(ns + 1));
        s_nodes = cfg.mesh.explicit_nodes;
      } else {
        s_nodes = build_graded_nodes(cfg.mesh.grid_segments, cfg.mesh.grading, true);
        TENRYU_ASSERT(static_cast<int>(s_nodes.size()) == ns + 1,
                      "Graded polar node count mismatch: got " +
                          std::to_string(s_nodes.size()) + " nodes, expected " +
                          std::to_string(ns + 1));
      }
      for (std::size_t k = 1; k < s_nodes.size(); ++k) {
        TENRYU_ASSERT(s_nodes[k] > s_nodes[k - 1],
                      "polar radial nodes must be strictly increasing");
      }
      if (tri_fan || button) {
        TENRYU_ASSERT(s_nodes.front() == 0.0,
                      "graded/explicit polar nodes require s(0) == 0 for "
                      "tri_fan/button center treatment");
      } else {
        TENRYU_ASSERT(s_nodes.front() > 0.0,
                      "graded/explicit polar nodes require s(0) > 0 for "
                      "annular center treatment");
      }
      TENRYU_ASSERT(std::abs(s_nodes.back() - s_max) <=
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
          std::to_string(ns + 1) +
          " s0=" + std::to_string(s_nodes.front()) +
          " s_max=" + std::to_string(s_nodes.back()) +
          " max_adjacent_dr_ratio=" + std::to_string(max_ratio));
    } else if (tri_fan || button) {
      for (int i = 0; i <= ns; ++i) {
        s_nodes[static_cast<std::size_t>(i)] =
            static_cast<double>(i) * s_max / static_cast<double>(ns);
      }
    } else {
      const double kappa = cfg.mesh.spherical_polar_kappa;
      const double delta_s = s_max / (static_cast<double>(ns) + kappa);
      for (int i = 0; i <= ns; ++i) {
        s_nodes[static_cast<std::size_t>(i)] =
            (static_cast<double>(i) + kappa) * delta_s;
      }
    }

    std::vector<double> theta_nodes(static_cast<std::size_t>(ntheta + 1), 0.0);
    const bool theta_zoned = !cfg.mesh.explicit_nodes_theta.empty() ||
                             !cfg.mesh.grid_segments_theta.empty();
    const bool is_symmetric_ladder =
        !theta_zoned && !(cfg.mesh.polar_theta_min > 0.0);
    if (theta_zoned) {
      TENRYU_ASSERT(!cfg.mesh.polar_equal_mu_zoning,
                    "Mesh.grid_theta/explicit_nodes_theta are exclusive with "
                    "polar_equal_mu_zoning");
      if (!cfg.mesh.explicit_nodes_theta.empty()) {
        theta_nodes = cfg.mesh.explicit_nodes_theta;
        TENRYU_ASSERT(static_cast<int>(theta_nodes.size()) == ntheta + 1,
                      "Mesh.explicit_nodes_theta count mismatch: got " +
                          std::to_string(theta_nodes.size()) +
                          " nodes, expected " + std::to_string(ntheta + 1));
      } else {
        theta_nodes = build_graded_nodes(cfg.mesh.grid_segments_theta,
                                         cfg.mesh.grading, true);
        TENRYU_ASSERT(static_cast<int>(theta_nodes.size()) == ntheta + 1,
                      "Graded theta node count mismatch: got " +
                          std::to_string(theta_nodes.size()) +
                          " nodes, expected " + std::to_string(ntheta + 1));
      }
      for (std::size_t k = 1; k < theta_nodes.size(); ++k) {
        TENRYU_ASSERT(theta_nodes[k] > theta_nodes[k - 1],
                      "polar theta nodes must be strictly increasing");
      }
      TENRYU_ASSERT(theta_nodes.front() == cfg.mesh.polar_theta_min,
                    "polar theta nodes must start at Mesh.polar_theta_min");
      TENRYU_ASSERT(std::abs(theta_nodes.back() -
                             spherical_wedge_detail::kPi) <= 1.0e-12,
                    "polar theta nodes must end at theta = pi");
      TENRYU_ASSERT(
          theta_nodes[theta_nodes.size() - 2] < spherical_wedge_detail::kPi,
          "polar theta nodes: second-to-last node must lie below pi");
      theta_nodes.back() = spherical_wedge_detail::kPi;
      double max_ratio = 1.0;
      for (std::size_t k = 2; k < theta_nodes.size(); ++k) {
        const double d0 = theta_nodes[k - 1] - theta_nodes[k - 2];
        const double d1 = theta_nodes[k] - theta_nodes[k - 1];
        const double ratio = std::max(d1 / d0, d0 / d1);
        max_ratio = std::max(max_ratio, ratio);
      }
      tenryu::core::log_info(
          "[mesh-graded] spherical_polar theta nodes: n=" +
          std::to_string(ntheta + 1) +
          " theta0=" + std::to_string(theta_nodes.front()) +
          " theta_end=" + std::to_string(theta_nodes.back()) +
          " max_adjacent_dtheta_ratio=" + std::to_string(max_ratio));
    } else if (cfg.mesh.polar_equal_mu_zoning && tri_fan) {
      if (cfg.mesh.polar_theta_min > 0.0) {
        const double mu_max = std::cos(cfg.mesh.polar_theta_min);
        for (int j = 0; j <= ntheta; ++j) {
          double mu = mu_max - (1.0 + mu_max) * static_cast<double>(j) /
                                    static_cast<double>(ntheta);
          mu = std::clamp(mu, -1.0, 1.0);
          theta_nodes[static_cast<std::size_t>(j)] = std::acos(mu);
        }
        theta_nodes.front() = cfg.mesh.polar_theta_min;
        theta_nodes.back() = spherical_wedge_detail::kPi;
      } else {
        for (int j = 0; j <= ntheta; ++j) {
          double mu = 1.0 - 2.0 * static_cast<double>(j) /
                               static_cast<double>(ntheta);
          mu = std::clamp(mu, -1.0, 1.0);
          theta_nodes[static_cast<std::size_t>(j)] = std::acos(mu);
        }
      }
    } else {
      if (cfg.mesh.polar_theta_min > 0.0) {
        const double theta_span =
            spherical_wedge_detail::kPi - cfg.mesh.polar_theta_min;
        for (int j = 0; j <= ntheta; ++j) {
          theta_nodes[static_cast<std::size_t>(j)] =
              cfg.mesh.polar_theta_min +
              static_cast<double>(j) * theta_span /
                  static_cast<double>(ntheta);
        }
      } else {
        for (int j = 0; j <= ntheta; ++j) {
          theta_nodes[static_cast<std::size_t>(j)] =
              static_cast<double>(j) * spherical_wedge_detail::kPi /
              static_cast<double>(ntheta);
        }
      }
    }

    for (int i = 0; i <= ns; ++i) {
      const double s = s_nodes[static_cast<std::size_t>(i)];
      if (is_symmetric_ladder) {
        const int northern_end = ntheta / 2;
        for (int j = 0; j <= northern_end; ++j) {
          const double theta = theta_nodes[static_cast<std::size_t>(j)];
          const int n = mesh.topo.node_index(i, j);
          host_x_r[static_cast<std::size_t>(n)] =
              (j == 0) ? 0.0 : s * std::sin(theta);
          host_x_z[static_cast<std::size_t>(n)] =
              (2 * j == ntheta) ? center_z + 0.0
                                : center_z + s * std::cos(theta);
        }
        for (int j = northern_end + 1; j <= ntheta; ++j) {
          const int northern_j = ntheta - j;
          const int n = mesh.topo.node_index(i, j);
          const int northern_n = mesh.topo.node_index(i, northern_j);
          host_x_r[static_cast<std::size_t>(n)] =
              host_x_r[static_cast<std::size_t>(northern_n)];
          host_x_z[static_cast<std::size_t>(n)] =
              center_z -
              (host_x_z[static_cast<std::size_t>(northern_n)] - center_z);
        }
      } else {
        for (int j = 0; j <= ntheta; ++j) {
          const double theta = theta_nodes[static_cast<std::size_t>(j)];
          const int n = mesh.topo.node_index(i, j);
          if (!polar_in_box && cfg.mesh.polar_theta_min > 0.0) {
            host_x_r[static_cast<std::size_t>(n)] =
                (j == ntheta) ? 0.0 : s * std::sin(theta);
          } else {
            host_x_r[static_cast<std::size_t>(n)] =
                (j == 0 || j == ntheta) ? 0.0 : s * std::sin(theta);
          }
          host_x_z[static_cast<std::size_t>(n)] =
              center_z + s * std::cos(theta);
        }
      }
    }
    if (polar_in_box) {
      const double theta_tr =
          std::atan2(cfg.mesh.box_r_max,
                     cfg.mesh.box_z_max - cfg.mesh.box_center_z);
      const double theta_br =
          spherical_wedge_detail::kPi -
          std::atan2(cfg.mesh.box_r_max,
                     cfg.mesh.box_center_z - cfg.mesh.box_z_min);
      const int j_tr = nearest_theta_index(theta_nodes, theta_tr);
      const int j_br = nearest_theta_index(theta_nodes, theta_br);
      if (!(j_tr > 0 && j_tr < j_br && j_br < ntheta)) {
        throw core::namelist::ConfigError(
            "polar_in_box corner bands require distinct interior theta indices; increase Mesh.nz");
      }
      mesh.topo.polar_in_box_j_tr = j_tr;
      mesh.topo.polar_in_box_j_br = j_br;

      const PolarInBoxExteriorContext exterior_context =
          build_polar_in_box_exterior_context(
              cfg.mesh, theta_nodes, j_tr, j_br);
      std::vector<RZPoint> lock_nodes;
      lock_nodes.reserve(static_cast<std::size_t>(ntheta + 1));
      std::vector<PolarInBoxExteriorRay> exterior_rays;
      exterior_rays.reserve(static_cast<std::size_t>(ntheta + 1));
      for (int j = 0; j <= ntheta; ++j) {
        const int n = mesh.topo.node_index(ns, j);
        const RZPoint lock_node{
            host_x_r[static_cast<std::size_t>(n)],
            host_x_z[static_cast<std::size_t>(n)]};
        lock_nodes.push_back(lock_node);
        exterior_rays.push_back(polar_in_box_exterior_ray(
            j, lock_node, s_max, exterior_context));
      }
      apply_polar_in_box_cone_postpass(
          lock_nodes, &exterior_rays, exterior_context);

      for (int k = 1; k <= cfg.mesh.morph_rings; ++k) {
        const int i = ns + k;
        const double xi = static_cast<double>(k) /
                          static_cast<double>(cfg.mesh.morph_rings);
        const double angular_blend = delayed_quintic_blend(xi);
        double previous_phi = -1.0;
        for (int j = 0; j <= ntheta; ++j) {
          const RZPoint target =
              exterior_context.collar.morph_target[static_cast<std::size_t>(j)];
          const double target_phi = polar_angle(target, center_z);
          const double phi = theta_nodes[static_cast<std::size_t>(j)] +
                             angular_blend *
                                 (target_phi -
                                  theta_nodes[static_cast<std::size_t>(j)]);
          if (!(j == 0 || phi > previous_phi)) {
            throw core::namelist::ConfigError(
                "polar_in_box morph angular labels are not strictly monotone");
          }
          previous_phi = phi;
          const RZPoint point =
              exterior_rays[static_cast<std::size_t>(j)]
                  .nodes[static_cast<std::size_t>(k - 1)];
          const int n = mesh.topo.node_index(i, j);
          host_x_r[static_cast<std::size_t>(n)] = point.r;
          host_x_z[static_cast<std::size_t>(n)] = point.z;
        }
      }

      for (int k = 1; k <= cfg.mesh.collar_rings; ++k) {
        const int i = ns + cfg.mesh.morph_rings + k;
        for (int j = 0; j <= ntheta; ++j) {
          const RZPoint point =
              exterior_rays[static_cast<std::size_t>(j)]
                  .nodes[static_cast<std::size_t>(
                      cfg.mesh.morph_rings + k - 1)];
          const int n = mesh.topo.node_index(i, j);
          host_x_r[static_cast<std::size_t>(n)] = point.r;
          host_x_z[static_cast<std::size_t>(n)] = point.z;
        }
      }
      set_2d_node_flags(mesh.topo, host_x_r);
      set_rectangular_edge_tags(mesh);
    } else {
      if (cfg.mesh.polar_theta_min > 0.0) {
        set_2d_node_flags(mesh.topo, host_x_r, true, false);
      } else {
        set_2d_node_flags(mesh.topo, host_x_r, true);
      }
    }
    if (!polar_in_box && tri_fan) {
      for (int j = 0; j <= ntheta; ++j) {
        const int n = mesh.topo.node_index(0, j);
        mesh.topo.node_flags[static_cast<std::size_t>(n)] |= NODE_CENTER;
      }
    }
    if (!polar_in_box) {
      if (cfg.mesh.polar_theta_min > 0.0) {
        set_spherical_polar_edge_tags(mesh, true);
      } else {
        set_spherical_polar_edge_tags(mesh);
      }
    }
  } else {
    std::vector<double> z_nodes;
    if (!cfg.mesh.explicit_nodes_z.empty()) {
      z_nodes = cfg.mesh.explicit_nodes_z;
      TENRYU_ASSERT(static_cast<int>(z_nodes.size()) == mesh.topo.nz + 1,
                    "Mesh.explicit_nodes_z count mismatch: got " +
                        std::to_string(z_nodes.size()) + " nodes, expected " +
                        std::to_string(mesh.topo.nz + 1));
      for (std::size_t k = 1; k < z_nodes.size(); ++k) {
        TENRYU_ASSERT(z_nodes[k] > z_nodes[k - 1],
                      "Mesh.explicit_nodes_z must be strictly increasing");
      }
    } else if (!cfg.mesh.grid_segments_z.empty()) {
      z_nodes = build_graded_nodes(cfg.mesh.grid_segments_z, cfg.mesh.grading,
                                   true);
      TENRYU_ASSERT(static_cast<int>(z_nodes.size()) == mesh.topo.nz + 1,
                    "Graded z node count mismatch: got " +
                        std::to_string(z_nodes.size()) + " nodes, expected " +
                        std::to_string(mesh.topo.nz + 1));
    } else {
      z_nodes = build_uniform_nodes(cfg.mesh.z_min, cfg.mesh.z_max,
                                    mesh.topo.nz);
    }
    for (int i = 0; i <= mesh.topo.nr; ++i) {
      for (int j = 0; j <= mesh.topo.nz; ++j) {
        const int n = mesh.topo.node_index(i, j);
        host_x_r[static_cast<std::size_t>(n)] = r_nodes[static_cast<std::size_t>(i)];
        host_x_z[static_cast<std::size_t>(n)] = z_nodes[static_cast<std::size_t>(j)];
      }
    }
    set_2d_node_flags(mesh.topo, host_x_r);
    set_rectangular_edge_tags(mesh);
  }

  // 2026-07-26 kernel review: generation-time discrete no-fold gate for
  // the 2D single-block structured family. Bilinear cell Jacobians are
  // affine in the logical coordinates, so orientation-consistent corner
  // Jacobians imply interior positivity cell-by-cell; a fold appears as a
  // corner sign flip. The spherical-polar family is clockwise-wound in
  // (r, z), so signs are bound to the first active cell's orientation
  // rather than assumed positive. Gate-only: coordinates are not modified.
  if (mesh.dim == 2 && mesh.topo.nz > 0 &&
      !mesh.topo.multiblock.has_value()) {
    const int gate_stride = mesh.topo.nz + 1;
    double reference_sign = 0.0;
    for (int c = 0; c < mesh.topo.n_cells; ++c) {
      if (mesh.is_button_cell(c) || mesh.is_dormant_cell(c)) {
        continue;
      }
      const int ci = c / mesh.topo.nz;
      const int cj = c - ci * mesh.topo.nz;
      const int n0 = ci * gate_stride + cj;
      const int n1 = (ci + 1) * gate_stride + cj;
      const int n2 = (ci + 1) * gate_stride + (cj + 1);
      const int n3 = ci * gate_stride + (cj + 1);
      const double cr[4] = {
          host_x_r[static_cast<std::size_t>(n0)],
          host_x_r[static_cast<std::size_t>(n1)],
          host_x_r[static_cast<std::size_t>(n2)],
          host_x_r[static_cast<std::size_t>(n3)],
      };
      const double cz[4] = {
          host_x_z[static_cast<std::size_t>(n0)],
          host_x_z[static_cast<std::size_t>(n1)],
          host_x_z[static_cast<std::size_t>(n2)],
          host_x_z[static_cast<std::size_t>(n3)],
      };
      const int gate_nverts =
          (!mesh.cell_nverts.empty() &&
           mesh.cell_nverts[static_cast<std::size_t>(c)] == 3U)
              ? 3
              : 4;
      double corner_j[4] = {0.0, 0.0, 0.0, 0.0};
      if (gate_nverts == 3) {
        for (int k = 0; k < 3; ++k) {
          const int kp = (k + 1) % 3;
          const int km = (k + 2) % 3;
          corner_j[k] =
              (cr[kp] - cr[k]) * (cz[km] - cz[k]) -
              (cz[kp] - cz[k]) * (cr[km] - cr[k]);
        }
      } else {
        corner_j[0] = (cr[1] - cr[0]) * (cz[3] - cz[0]) -
                      (cz[1] - cz[0]) * (cr[3] - cr[0]);
        corner_j[1] = (cr[1] - cr[0]) * (cz[2] - cz[1]) -
                      (cz[1] - cz[0]) * (cr[2] - cr[1]);
        corner_j[2] = (cr[2] - cr[3]) * (cz[2] - cz[1]) -
                      (cz[2] - cz[3]) * (cr[2] - cr[1]);
        corner_j[3] = (cr[2] - cr[3]) * (cz[3] - cz[0]) -
                      (cz[2] - cz[3]) * (cr[3] - cr[0]);
      }
      double volume_sum = 0.0;
      for (int k = 0; k < gate_nverts; ++k) {
        const int kp = (k + 1 == gate_nverts) ? 0 : (k + 1);
        volume_sum +=
            (cr[k] + cr[kp]) * (cr[k] * cz[kp] - cr[kp] * cz[k]);
      }
      if (reference_sign == 0.0) {
        if (corner_j[0] == 0.0 || !std::isfinite(corner_j[0])) {
          throw core::namelist::ConfigError(
              "2D mesh generation no-fold gate: first active cell has a "
              "zero or non-finite corner Jacobian (i=" +
              std::to_string(ci) + ", j=" + std::to_string(cj) + ")");
        }
        reference_sign = corner_j[0] > 0.0 ? 1.0 : -1.0;
      }
      for (int k = 0; k < gate_nverts; ++k) {
        if (!(std::isfinite(corner_j[k]) &&
              reference_sign * corner_j[k] > 0.0)) {
          throw core::namelist::ConfigError(
              "2D mesh generation no-fold gate: corner Jacobian sign flip "
              "at cell (i=" + std::to_string(ci) + ", j=" +
              std::to_string(cj) + "), corner " + std::to_string(k) +
              ", value=" + std::to_string(corner_j[k]) +
              ", reference_sign=" + std::to_string(reference_sign));
        }
      }
      if (!(std::isfinite(volume_sum) &&
            reference_sign * volume_sum > 0.0)) {
        throw core::namelist::ConfigError(
            "2D mesh generation no-fold gate: revolved-volume sign flip at "
            "cell (i=" + std::to_string(ci) + ", j=" + std::to_string(cj) +
            "), volume_sum=" + std::to_string(volume_sum) +
            ", reference_sign=" + std::to_string(reference_sign));
      }
    }
  }
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

  mesh.node_r = state.x_r.data();
  mesh.node_z = (mesh.dim == 2) ? state.x_z.data() : nullptr;
  mesh.recompute_geometry();
  if (state.cell_is_void.size() == static_cast<std::size_t>(mesh.topo.n_cells)) {
    std::fill(state.cell_is_void.begin(),
              state.cell_is_void.end(),
              static_cast<std::uint8_t>(0));
    if (mesh.button_center && mesh.button_center->enabled) {
      for (int c = 0; c < mesh.topo.n_cells; ++c) {
        if (mesh.is_dormant_cell(c)) {
          state.cell_is_void[static_cast<std::size_t>(c)] =
              static_cast<std::uint8_t>(1);
        }
      }
    }
  }
  state.cell_vol_initial = mesh.cell_vol;
  return mesh;
}

void set_geometry_check_window(const int begin, const int end) {
  g_geometry_check_begin = begin;
  g_geometry_check_end = end;
}

}  // namespace tenryu::mesh
