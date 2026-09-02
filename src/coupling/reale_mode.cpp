#include "coupling/reale_mode.hpp"
#include "coupling/reale_trigger.hpp"
#include "coupling/staged_dt_gate.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/namelist/errors.hpp"
#include "core/state.hpp"
#include "hydro/anti_hourglass.cuh"
#include "hydro/carrier_projection.hpp"
#include "hydro/pentagon_geometry.cuh"
#include "hydro/reale_remap.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/generator_targets.cuh"
#include "mesh/mesh.hpp"
#include "mesh/poly_geom.cuh"
#include "mesh/ring_pages.hpp"
#include "mesh/tessellation/voronoi_dual.hpp"
#include "mesh/voronoi_rz.cuh"

namespace tenryu::coupling::reale_mode {
namespace {

bool reale_vdiag_enabled() {
  static const bool enabled = [] {
    const char* const v = std::getenv("TENRYU_REALE_VDIAG");
    return v != nullptr && *v == '1';
  }();
  return enabled;
}

bool reale_p3_diag_enabled() {
  static const bool enabled = [] {
    const char* const v = std::getenv("TENRYU_REALE_P3_DIAG");
    return v != nullptr && *v == '1';
  }();
  return enabled;
}

void reale_vdiag_scan(const char* tag, const int step,
                      const std::vector<double>& vr,
                      const std::vector<double>& vz) {
  if (!reale_vdiag_enabled()) return;
  double vmax = 0.0;
  std::size_t imax = 0;
  for (std::size_t n = 0; n < vr.size(); ++n) {
    const double m = std::hypot(vr[n], vz[n]);
    if (m > vmax) {
      vmax = m;
      imax = n;
    }
  }
  std::fprintf(stderr, "[reale_vdiag] %s step=%d vmax=%.6e node=%zu\n",
               tag, step, vmax, imax);
}

void reale_vdiag_rho_scan(const char* tag, const int step,
                          core::State& state) {
  if (!reale_vdiag_enabled()) return;
  std::vector<double> rho;
  state.rho.copy_to_host(rho);
  // Interior reference for the uniform-sphere qualification deck:
  // rho0=0.25. Report the max deviation over cells whose centroid... we
  // do not have centroids here cheaply; report global max/min instead.
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  for (const double v : rho) { lo = std::min(lo, v); hi = std::max(hi, v); }
  std::fprintf(stderr, "[reale_vdiag] %s step=%d rho_min=%.6e rho_max=%.6e\n",
               tag, step, lo, hi);
}

struct HostMesh {
  int n_cells = 0;
  int n_nodes = 0;
  std::vector<int> csr_offsets;
  std::vector<int> csr_indices;
  std::vector<unsigned char> nverts;
  std::vector<double> node_r;
  std::vector<double> node_z;
};

// Mirrors local_face_corners in src/hydro/compatible_force_work_2d.cu.
bool host_local_face_corners(const int active_nverts,
                             const int local,
                             int* corner0,
                             int* corner1) {
  if (active_nverts >= 5) {
    return mesh::mesh_topo_active_local_face_corners(
        active_nverts, local, corner0, corner1);
  }
  if (active_nverts != 3) {
    if (local == 0) {
      *corner0 = 0;
      *corner1 = 3;
    } else if (local == 1) {
      *corner0 = 1;
      *corner1 = 2;
    } else if (local == 2) {
      *corner0 = 0;
      *corner1 = 1;
    } else {
      *corner0 = 3;
      *corner1 = 2;
    }
    return true;
  }
  return mesh::mesh_topo_active_local_face_corners(
      active_nverts, local, corner0, corner1);
}

struct VelocityGridKey {
  std::int64_t r = 0;
  std::int64_t z = 0;

  bool operator==(const VelocityGridKey& other) const {
    return r == other.r && z == other.z;
  }
};

struct VelocityGridKeyHash {
  std::size_t operator()(const VelocityGridKey& key) const noexcept {
    const std::uint64_t r = static_cast<std::uint64_t>(key.r);
    const std::uint64_t z = static_cast<std::uint64_t>(key.z);
    return static_cast<std::size_t>(
        (r * UINT64_C(0x9e3779b97f4a7c15)) ^
        (z + UINT64_C(0x9e3779b97f4a7c15) + (r << 6U) + (r >> 2U)));
  }
};

void limit_remapped_velocities(
    const HostMesh& old_mesh, const HostMesh& new_mesh,
    const std::vector<double>& old_velocity_r,
    const std::vector<double>& old_velocity_z,
    std::vector<double>& new_velocity_r,
    std::vector<double>& new_velocity_z) {
  constexpr double kCellSizeCm = 5.0e-5;
  constexpr double kRadiusCm = 5.0e-5;
  constexpr double kWideRadiusCm = 1.0e-4;
  constexpr double kPadFraction = 0.05;

  const auto grid_coordinate = [](const double coordinate) {
    return static_cast<std::int64_t>(std::floor(coordinate / kCellSizeCm));
  };
  // Fixed deck-scale spacing; the limiter only needs a local neighborhood.
  std::unordered_map<VelocityGridKey, std::vector<std::size_t>,
                     VelocityGridKeyHash>
      grid;
  grid.reserve(static_cast<std::size_t>(old_mesh.n_nodes));
  for (int node = 0; node < old_mesh.n_nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    grid[{grid_coordinate(old_mesh.node_r[index]),
          grid_coordinate(old_mesh.node_z[index])}]
        .push_back(index);
  }

  struct LimiterStats {
    int clamped_nodes = 0;
    int insufficient_neighbors = 0;
    double max_delta_velocity = 0.0;
  } stats;
  std::vector<std::size_t> neighbors;
  for (int node = 0; node < new_mesh.n_nodes; ++node) {
    const std::size_t index = static_cast<std::size_t>(node);
    const double new_r = new_mesh.node_r[index];
    const double new_z = new_mesh.node_z[index];
    const VelocityGridKey center{grid_coordinate(new_r),
                                 grid_coordinate(new_z)};
    const auto gather = [&](const double radius) {
      neighbors.clear();
      const int span = static_cast<int>(std::ceil(radius / kCellSizeCm));
      const double radius_squared = radius * radius;
      for (int dr = -span; dr <= span; ++dr) {
        for (int dz = -span; dz <= span; ++dz) {
          const auto cell = grid.find(
              {center.r + static_cast<std::int64_t>(dr),
               center.z + static_cast<std::int64_t>(dz)});
          if (cell == grid.end()) {
            continue;
          }
          for (const std::size_t old_node : cell->second) {
            const double delta_r = old_mesh.node_r[old_node] - new_r;
            const double delta_z = old_mesh.node_z[old_node] - new_z;
            if (delta_r * delta_r + delta_z * delta_z <= radius_squared) {
              neighbors.push_back(old_node);
            }
          }
        }
      }
    };

    gather(kRadiusCm);
    if (neighbors.size() < 3U) {
      gather(kWideRadiusCm);
    }
    if (neighbors.size() < 3U) {
      ++stats.insufficient_neighbors;
      continue;
    }

    double velocity_min_r = std::numeric_limits<double>::infinity();
    double velocity_max_r = -std::numeric_limits<double>::infinity();
    double velocity_min_z = std::numeric_limits<double>::infinity();
    double velocity_max_z = -std::numeric_limits<double>::infinity();
    for (const std::size_t old_node : neighbors) {
      velocity_min_r = std::min(velocity_min_r, old_velocity_r[old_node]);
      velocity_max_r = std::max(velocity_max_r, old_velocity_r[old_node]);
      velocity_min_z = std::min(velocity_min_z, old_velocity_z[old_node]);
      velocity_max_z = std::max(velocity_max_z, old_velocity_z[old_node]);
    }
    const double pad_r = kPadFraction * (velocity_max_r - velocity_min_r);
    const double pad_z = kPadFraction * (velocity_max_z - velocity_min_z);
    const double old_new_velocity_r = new_velocity_r[index];
    const double old_new_velocity_z = new_velocity_z[index];
    new_velocity_r[index] =
        std::clamp(old_new_velocity_r, velocity_min_r - pad_r,
                   velocity_max_r + pad_r);
    new_velocity_z[index] =
        std::clamp(old_new_velocity_z, velocity_min_z - pad_z,
                   velocity_max_z + pad_z);
    const double delta_velocity_r = new_velocity_r[index] - old_new_velocity_r;
    const double delta_velocity_z = new_velocity_z[index] - old_new_velocity_z;
    if (delta_velocity_r != 0.0 || delta_velocity_z != 0.0) {
      ++stats.clamped_nodes;
      stats.max_delta_velocity =
          std::max(stats.max_delta_velocity,
                   std::hypot(delta_velocity_r, delta_velocity_z));
    }
  }

  if (stats.clamped_nodes > 0 || reale_vdiag_enabled()) {
    std::fprintf(stderr, "[vel_limiter] clamped=%d max_dv=%.3e\n",
                 stats.clamped_nodes, stats.max_delta_velocity);
  }
}

struct StagedMesh : HostMesh {
  std::vector<std::uint8_t> node_flags;
  std::vector<std::uint8_t> node_class;
  std::vector<int> node_domain_vertex;
  std::vector<int> node_carrier_edge;
  std::vector<double> node_lambda;
  std::size_t provenance_conflicts = 0;
};

class DisjointSet {
 public:
  explicit DisjointSet(const int size)
      : parent_(static_cast<std::size_t>(size)) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int find(const int value) {
    int& parent = parent_[static_cast<std::size_t>(value)];
    if (parent != value) {
      parent = find(parent);
    }
    return parent;
  }

  void unite(const int lhs, const int rhs) {
    int lhs_root = find(lhs);
    int rhs_root = find(rhs);
    if (lhs_root == rhs_root) {
      return;
    }
    if (lhs_root > rhs_root) {
      std::swap(lhs_root, rhs_root);
    }
    parent_[static_cast<std::size_t>(rhs_root)] = lhs_root;
  }

 private:
  std::vector<int> parent_;
};

bool skip(core::State& state,
          const char* const stage,
          const std::string& reason) {
  ++state.reale_rezone_skipped;
  core::log_warning(
      "[reale_v2] SKIP step=" + std::to_string(state.step) +
      " stage=" + stage + " reason=" + reason +
      " skipped_total=" + std::to_string(state.reale_rezone_skipped));
  return false;
}

bool close_coordinate(const double lhs, const double rhs) {
  constexpr double kTolerance =
      1024.0 * std::numeric_limits<double>::epsilon();
  return std::abs(lhs - rhs) <=
         kTolerance * std::max({1.0, std::abs(lhs), std::abs(rhs)});
}

bool close_point(const double lhs_r,
                 const double lhs_z,
                 const double rhs_r,
                 const double rhs_z) {
  return close_coordinate(lhs_r, rhs_r) && close_coordinate(lhs_z, rhs_z);
}

bool capture_current_mesh(const core::State& state,
                          HostMesh& host,
                          std::string& reason) {
  const mesh::Mesh& mesh = state.mesh;
  if (!mesh.topo.multiblock.has_value()) {
    reason = "missing_multiblock_topology";
    return false;
  }
  const mesh::MultiBlockTopology& mb = *mesh.topo.multiblock;
  host.n_cells = mesh.topo.n_cells;
  host.n_nodes = mesh.topo.n_nodes;
  if (host.n_cells <= 0 || host.n_nodes <= 0 ||
      mesh.corner_stride != state.corner_stride || state.corner_stride < 8 ||
      mb.cell_node_csr_offsets.size() !=
          static_cast<std::size_t>(host.n_cells + 1) ||
      mb.cell_node_csr_indices.size() !=
          static_cast<std::size_t>(host.n_cells * state.corner_stride) ||
      mesh.cell_nverts.size() != static_cast<std::size_t>(host.n_cells) ||
      (!state.hydro_active.empty() &&
       state.hydro_active.size() != static_cast<std::size_t>(host.n_cells)) ||
      state.x_r.size() != static_cast<std::size_t>(host.n_nodes) ||
      state.x_z.size() != static_cast<std::size_t>(host.n_nodes)) {
    reason = "incomplete_stride8_mesh";
    return false;
  }
  host.csr_offsets = mb.cell_node_csr_offsets;
  host.csr_indices = mb.cell_node_csr_indices;
  host.nverts.assign(mesh.cell_nverts.begin(), mesh.cell_nverts.end());
  state.x_r.copy_to_host(host.node_r);
  state.x_z.copy_to_host(host.node_z);
  for (int cell = 0; cell < host.n_cells; ++cell) {
    const int begin = host.csr_offsets[static_cast<std::size_t>(cell)];
    const int end = host.csr_offsets[static_cast<std::size_t>(cell + 1)];
    const int nverts = host.nverts[static_cast<std::size_t>(cell)];
    if (begin != cell * state.corner_stride ||
        end != begin + state.corner_stride || nverts < 3 ||
        nverts > state.corner_stride) {
      reason = "invalid_stride8_csr";
      return false;
    }
    if (!state.hydro_active.empty() &&
        state.hydro_active[static_cast<std::size_t>(cell)] == 0) {
      reason = "inactive_hydro_cell";
      return false;
    }
    for (int corner = 0; corner < nverts; ++corner) {
      const int node =
          host.csr_indices[static_cast<std::size_t>(begin + corner)];
      if (node < 0 || node >= host.n_nodes) {
        reason = "invalid_csr_node";
        return false;
      }
    }
  }
  for (int node = 0; node < host.n_nodes; ++node) {
    const double r = host.node_r[static_cast<std::size_t>(node)];
    const double z = host.node_z[static_cast<std::size_t>(node)];
    if (!std::isfinite(r) || !std::isfinite(z) || r < 0.0) {
      reason = "invalid_node_coordinate";
      return false;
    }
  }
  return true;
}

struct DisturbedClassification {
  std::vector<unsigned char> dilated;
  std::vector<std::vector<int>> adjacency;
  bool has_face_adjacency = false;
};

struct SubdomainBoundaryEntry {
  int node = -1;
  int cell = -1;
  int local_face = -1;
};

struct SubdomainComponent {
  std::vector<int> cells;
  std::vector<unsigned char> membership;
  std::vector<SubdomainBoundaryEntry> boundary;
};

DisturbedClassification classify_disturbed_cells(
    const core::State& state, const HostMesh& host) {

  constexpr double kMinAltitudeRel = 0.15;
  constexpr double kMaxEdgeRatio = 8.0;
  constexpr double kVelocityFraction = 0.02;
  constexpr double kRhoContrast = 0.02;
  constexpr int kDilationRings = 4;

  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  std::vector<double> rho;
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);
  state.rho.copy_to_host(rho);

  const mesh::MultiBlockTopology& mb = *state.mesh.topo.multiblock;
  bool use_face_adjacency =
      mb.face_adj_csr_offsets.size() ==
          static_cast<std::size_t>(host.n_cells + 1) &&
      !mb.face_adj_csr_offsets.empty() &&
      mb.face_adj_csr_offsets.front() == 0;
  if (use_face_adjacency) {
    for (int cell = 0; cell < host.n_cells; ++cell) {
      const int begin =
          mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
      const int end =
          mb.face_adj_csr_offsets[static_cast<std::size_t>(cell + 1)];
      const int nverts = host.nverts[static_cast<std::size_t>(cell)];
      if (begin < 0 || end < begin || end - begin < nverts ||
          static_cast<std::size_t>(end) >
              mb.face_adj_csr_indices.size()) {
        use_face_adjacency = false;
        break;
      }
    }
  }

  std::vector<std::vector<int>> adjacency(
      static_cast<std::size_t>(host.n_cells));
  if (use_face_adjacency) {
    for (int cell = 0; cell < host.n_cells; ++cell) {
      const int begin =
          mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
      const int end =
          begin + host.nverts[static_cast<std::size_t>(cell)];
      std::vector<int>& neighbors =
          adjacency[static_cast<std::size_t>(cell)];
      for (int entry = begin; entry < end; ++entry) {
        const int neighbor =
            mb.face_adj_csr_indices[static_cast<std::size_t>(entry)];
        if (neighbor >= 0 && neighbor < host.n_cells && neighbor != cell) {
          neighbors.push_back(neighbor);
        }
      }
    }
  } else {
    std::vector<std::vector<int>> node_cells(
        static_cast<std::size_t>(host.n_nodes));
    for (int cell = 0; cell < host.n_cells; ++cell) {
      const int begin = host.csr_offsets[static_cast<std::size_t>(cell)];
      const int nverts = host.nverts[static_cast<std::size_t>(cell)];
      for (int corner = 0; corner < nverts; ++corner) {
        const int node =
            host.csr_indices[static_cast<std::size_t>(begin + corner)];
        node_cells[static_cast<std::size_t>(node)].push_back(cell);
      }
    }
    for (const std::vector<int>& cells : node_cells) {
      for (const int cell : cells) {
        std::vector<int>& neighbors =
            adjacency[static_cast<std::size_t>(cell)];
        for (const int neighbor : cells) {
          if (neighbor != cell) neighbors.push_back(neighbor);
        }
      }
    }
    for (std::vector<int>& neighbors : adjacency) {
      std::sort(neighbors.begin(), neighbors.end());
      neighbors.erase(std::unique(neighbors.begin(), neighbors.end()),
                      neighbors.end());
    }
  }

  double global_max_velocity = 0.0;
  for (int node = 0; node < host.n_nodes; ++node) {
    global_max_velocity = std::max(
        global_max_velocity,
        std::hypot(velocity_r[static_cast<std::size_t>(node)],
                   velocity_z[static_cast<std::size_t>(node)]));
  }
  const double velocity_threshold =
      kVelocityFraction * std::max(global_max_velocity, 1.0);

  double rho_median = 0.0;
  if (!use_face_adjacency) {
    std::vector<double> sorted_rho = rho;
    const std::size_t middle = sorted_rho.size() / 2U;
    std::nth_element(sorted_rho.begin(), sorted_rho.begin() + middle,
                     sorted_rho.end());
    rho_median = sorted_rho[middle];
    if (sorted_rho.size() % 2U == 0U) {
      rho_median =
          0.5 * (rho_median +
                 *std::max_element(sorted_rho.begin(),
                                   sorted_rho.begin() + middle));
    }
  }

  const auto relative_contrast = [](const double lhs, const double rhs) {
    return std::abs(lhs - rhs) /
           std::max({std::abs(lhs), std::abs(rhs), 1.0e-30});
  };

  std::vector<unsigned char> disturbed(
      static_cast<std::size_t>(host.n_cells), 0U);
  int geo_count = 0;
  int kin_count = 0;
  int field_count = 0;
  for (int cell = 0; cell < host.n_cells; ++cell) {
    const int begin = host.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = host.nverts[static_cast<std::size_t>(cell)];
    double twice_area = 0.0;
    double min_edge = std::numeric_limits<double>::infinity();
    double max_edge = 0.0;
    double min_altitude = std::numeric_limits<double>::infinity();
    double cell_max_velocity = 0.0;

    for (int corner = 0; corner < nverts; ++corner) {
      const int next = (corner + 1) % nverts;
      const int node =
          host.csr_indices[static_cast<std::size_t>(begin + corner)];
      const int next_node =
          host.csr_indices[static_cast<std::size_t>(begin + next)];
      const double r = host.node_r[static_cast<std::size_t>(node)];
      const double z = host.node_z[static_cast<std::size_t>(node)];
      const double next_r =
          host.node_r[static_cast<std::size_t>(next_node)];
      const double next_z =
          host.node_z[static_cast<std::size_t>(next_node)];
      twice_area += r * next_z - next_r * z;
      const double edge = std::hypot(next_r - r, next_z - z);
      min_edge = std::min(min_edge, edge);
      max_edge = std::max(max_edge, edge);
      cell_max_velocity = std::max(
          cell_max_velocity,
          std::hypot(velocity_r[static_cast<std::size_t>(node)],
                     velocity_z[static_cast<std::size_t>(node)]));
    }

    for (int corner = 0; corner < nverts; ++corner) {
      const int edge0 = (corner + 1) % nverts;
      const int edge1 = (corner + 2) % nverts;
      const int node =
          host.csr_indices[static_cast<std::size_t>(begin + corner)];
      const int node0 =
          host.csr_indices[static_cast<std::size_t>(begin + edge0)];
      const int node1 =
          host.csr_indices[static_cast<std::size_t>(begin + edge1)];
      const double edge_r =
          host.node_r[static_cast<std::size_t>(node1)] -
          host.node_r[static_cast<std::size_t>(node0)];
      const double edge_z =
          host.node_z[static_cast<std::size_t>(node1)] -
          host.node_z[static_cast<std::size_t>(node0)];
      const double edge_length = std::hypot(edge_r, edge_z);
      if (!(edge_length > 0.0) || !std::isfinite(edge_length)) {
        min_altitude = 0.0;
        break;
      }
      const double point_r =
          host.node_r[static_cast<std::size_t>(node)] -
          host.node_r[static_cast<std::size_t>(node0)];
      const double point_z =
          host.node_z[static_cast<std::size_t>(node)] -
          host.node_z[static_cast<std::size_t>(node0)];
      min_altitude =
          std::min(min_altitude,
                   std::abs(point_r * edge_z - point_z * edge_r) /
                       edge_length);
    }

    const double area = 0.5 * std::abs(twice_area);
    const bool geometric =
        !(area > 0.0) || !std::isfinite(area) ||
        min_altitude / std::sqrt(area) < kMinAltitudeRel ||
        !(min_edge > 0.0) || max_edge / min_edge > kMaxEdgeRatio;
    const bool kinematic = cell_max_velocity > velocity_threshold;
    bool field = false;
    if (use_face_adjacency) {
      for (const int neighbor : adjacency[static_cast<std::size_t>(cell)]) {
        if (relative_contrast(rho[static_cast<std::size_t>(cell)],
                              rho[static_cast<std::size_t>(neighbor)]) >
            kRhoContrast) {
          field = true;
          break;
        }
      }
    } else {
      field = relative_contrast(rho[static_cast<std::size_t>(cell)],
                                rho_median) > kRhoContrast;
    }

    geo_count += geometric ? 1 : 0;
    kin_count += kinematic ? 1 : 0;
    field_count += field ? 1 : 0;
    disturbed[static_cast<std::size_t>(cell)] =
        (geometric || kinematic || field) ? 1U : 0U;
  }

  std::vector<unsigned char> dilated = disturbed;
  std::vector<int> frontier;
  for (int cell = 0; cell < host.n_cells; ++cell) {
    if (disturbed[static_cast<std::size_t>(cell)] != 0U) {
      frontier.push_back(cell);
    }
  }
  for (int ring = 0; ring < kDilationRings; ++ring) {
    std::vector<int> next_frontier;
    for (const int cell : frontier) {
      for (const int neighbor : adjacency[static_cast<std::size_t>(cell)]) {
        if (dilated[static_cast<std::size_t>(neighbor)] == 0U) {
          dilated[static_cast<std::size_t>(neighbor)] = 1U;
          next_frontier.push_back(neighbor);
        }
      }
    }
    frontier = std::move(next_frontier);
  }

  const int disturbed_count = static_cast<int>(
      std::count(disturbed.begin(), disturbed.end(), 1U));
  const int dilated_count =
      static_cast<int>(std::count(dilated.begin(), dilated.end(), 1U));
  const double fraction =
      static_cast<double>(dilated_count) / static_cast<double>(host.n_cells);
  if (reale_p3_diag_enabled()) {
    std::fprintf(stderr,
                 "[p3_diag] step=%d disturbed=%d dilated=%d frac=%.3f "
                 "geo=%d kin=%d fld=%d\n",
                 state.step, disturbed_count, dilated_count, fraction,
                 geo_count, kin_count, field_count);
  }
  return {std::move(dilated), std::move(adjacency),
          use_face_adjacency};
}

std::vector<SubdomainComponent> build_subdomain_components(
    const core::State& state, const HostMesh& host,
    const DisturbedClassification& classification) {
  const mesh::MultiBlockTopology& mb = *state.mesh.topo.multiblock;
  const std::size_t n_cells = static_cast<std::size_t>(host.n_cells);
  std::vector<unsigned char> assigned(n_cells, 0U);
  std::vector<std::vector<unsigned char>> filled_components;

  // The tri-fan center column and its face halo are full-path-only.
  std::vector<unsigned char> tri_fan_protected(n_cells, 0U);
  for (int cell = 0; cell < host.n_cells; ++cell) {
    if (host.nverts[static_cast<std::size_t>(cell)] != 3U) {
      continue;
    }
    tri_fan_protected[static_cast<std::size_t>(cell)] = 1U;
    const int face_begin =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
    const int face_end =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(cell + 1)];
    for (int face = face_begin; face < face_end; ++face) {
      const int neighbor =
          mb.face_adj_csr_indices[static_cast<std::size_t>(face)];
      if (neighbor >= 0 && neighbor < host.n_cells) {
        tri_fan_protected[static_cast<std::size_t>(neighbor)] = 1U;
      }
    }
  }

  std::vector<unsigned char> true_boundary_cell(n_cells, 0U);
  for (const mesh::UniqueOrientedFace& face : mb.boundary_faces) {
    TENRYU_ASSERT(face.cell_a >= 0 && face.cell_a < host.n_cells &&
                      face.cell_b == -1,
                  "subdomain component malformed true boundary face");
    true_boundary_cell[static_cast<std::size_t>(face.cell_a)] = 1U;
  }

  for (int seed = 0; seed < host.n_cells; ++seed) {
    if (classification.dilated[static_cast<std::size_t>(seed)] == 0U ||
        assigned[static_cast<std::size_t>(seed)] != 0U) {
      continue;
    }
    std::vector<unsigned char> component(n_cells, 0U);
    std::vector<int> frontier{seed};
    component[static_cast<std::size_t>(seed)] = 1U;
    assigned[static_cast<std::size_t>(seed)] = 1U;
    while (!frontier.empty()) {
      const int cell = frontier.back();
      frontier.pop_back();
      for (const int neighbor :
           classification.adjacency[static_cast<std::size_t>(cell)]) {
        if (classification.dilated[static_cast<std::size_t>(neighbor)] !=
                0U &&
            assigned[static_cast<std::size_t>(neighbor)] == 0U) {
          component[static_cast<std::size_t>(neighbor)] = 1U;
          assigned[static_cast<std::size_t>(neighbor)] = 1U;
          frontier.push_back(neighbor);
        }
      }
    }

    for (std::size_t cell = 0; cell < n_cells; ++cell) {
      if (tri_fan_protected[cell] != 0U) {
        component[cell] = 0U;
      }
    }
    if (std::none_of(component.begin(), component.end(),
                     [](const unsigned char member) {
                       return member != 0U;
                     })) {
      continue;
    }

    std::vector<unsigned char> outside(n_cells, 0U);
    frontier.clear();
    for (int cell = 0; cell < host.n_cells; ++cell) {
      if (component[static_cast<std::size_t>(cell)] == 0U &&
          true_boundary_cell[static_cast<std::size_t>(cell)] != 0U) {
        outside[static_cast<std::size_t>(cell)] = 1U;
        frontier.push_back(cell);
      }
    }
    while (!frontier.empty()) {
      const int cell = frontier.back();
      frontier.pop_back();
      for (const int neighbor :
           classification.adjacency[static_cast<std::size_t>(cell)]) {
        if (component[static_cast<std::size_t>(neighbor)] == 0U &&
            outside[static_cast<std::size_t>(neighbor)] == 0U) {
          outside[static_cast<std::size_t>(neighbor)] = 1U;
          frontier.push_back(neighbor);
        }
      }
    }
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
      if (component[cell] == 0U && outside[cell] == 0U &&
          tri_fan_protected[cell] == 0U) {
        component[cell] = 1U;
      }
    }
    filled_components.push_back(std::move(component));
  }

  DisjointSet overlapping(
      static_cast<int>(filled_components.size()));
  std::vector<int> first_component(n_cells, -1);
  for (int component = 0;
       component < static_cast<int>(filled_components.size());
       ++component) {
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
      if (filled_components[static_cast<std::size_t>(component)][cell] ==
          0U) {
        continue;
      }
      int& first = first_component[cell];
      if (first < 0) {
        first = component;
      } else {
        overlapping.unite(first, component);
      }
    }
  }

  std::map<int, std::vector<unsigned char>> merged;
  for (int component = 0;
       component < static_cast<int>(filled_components.size());
       ++component) {
    std::vector<unsigned char>& membership =
        merged.try_emplace(overlapping.find(component), n_cells, 0U)
            .first->second;
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
      membership[cell] = static_cast<unsigned char>(
          membership[cell] |
          filled_components[static_cast<std::size_t>(component)][cell]);
    }
  }

  struct BoundaryEdge {
    int node0 = -1;
    int node1 = -1;
    int cell = -1;
    int local_face = -1;
  };
  std::vector<SubdomainComponent> components;
  components.reserve(merged.size());
  int nonsimple_boundary_skip_count = 0;
  for (auto& [root, membership] : merged) {
    (void)root;
    SubdomainComponent component;
    component.membership = std::move(membership);
    std::vector<BoundaryEdge> edges;
    for (int cell = 0; cell < host.n_cells; ++cell) {
      if (component.membership[static_cast<std::size_t>(cell)] == 0U) {
        continue;
      }
      component.cells.push_back(cell);
      const int nverts = host.nverts[static_cast<std::size_t>(cell)];
      const int cell_offset =
          host.csr_offsets[static_cast<std::size_t>(cell)];
      const int face_offset =
          mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
      for (int local_face = 0; local_face < nverts; ++local_face) {
        const int neighbor = mb.face_adj_csr_indices[
            static_cast<std::size_t>(face_offset + local_face)];
        if (neighbor >= 0 &&
            component.membership[static_cast<std::size_t>(neighbor)] !=
                0U) {
          continue;
        }
        int corner0 = -1;
        int corner1 = -1;
        TENRYU_ASSERT(host_local_face_corners(
                          nverts, local_face, &corner0, &corner1),
                      "subdomain component invalid local face");
        edges.push_back({
            host.csr_indices[static_cast<std::size_t>(cell_offset +
                                                      corner0)],
            host.csr_indices[static_cast<std::size_t>(cell_offset +
                                                      corner1)],
            cell, local_face});
      }
    }

    const auto walk_boundary = [&edges](
                                   std::vector<SubdomainBoundaryEntry>&
                                       boundary) {
      std::map<int, int> edge_from_node;
      std::map<int, int> incoming_count;
      for (int edge = 0; edge < static_cast<int>(edges.size()); ++edge) {
        const BoundaryEdge& boundary_edge =
            edges[static_cast<std::size_t>(edge)];
        TENRYU_ASSERT(boundary_edge.node0 >= 0 &&
                          boundary_edge.node1 >= 0 &&
                          boundary_edge.cell >= 0 &&
                          boundary_edge.local_face >= 0,
                      "subdomain boundary edge has negative id");
        if (boundary_edge.node0 == boundary_edge.node1 ||
            !edge_from_node.emplace(boundary_edge.node0, edge).second) {
          return false;
        }
        ++incoming_count[boundary_edge.node1];
      }
      if (edges.size() < 3U || edge_from_node.size() != edges.size()) {
        return false;
      }
      for (const auto& [node, edge] : edge_from_node) {
        (void)edge;
        if (incoming_count[node] != 1) {
          return false;
        }
      }
      const int first_node = edge_from_node.begin()->first;
      int node = first_node;
      std::set<int> visited;
      while (visited.insert(node).second) {
        const auto found = edge_from_node.find(node);
        if (found == edge_from_node.end()) {
          return false;
        }
        const BoundaryEdge& edge =
            edges[static_cast<std::size_t>(found->second)];
        boundary.push_back({edge.node0, edge.cell, edge.local_face});
        node = edge.node1;
      }
      return node == first_node && visited.size() == edges.size();
    };

    if (!walk_boundary(component.boundary)) {
      ++nonsimple_boundary_skip_count;
      std::fprintf(stderr,
                   "[p3_w1] component skipped: nonsimple boundary "
                   "(cells=%d)\n",
                   static_cast<int>(component.cells.size()));
      continue;
    }
    double twice_area = 0.0;
    for (std::size_t vertex = 0; vertex < component.boundary.size();
         ++vertex) {
      const std::size_t next =
          (vertex + 1U) % component.boundary.size();
      const int node0 = component.boundary[vertex].node;
      const int node1 = component.boundary[next].node;
      twice_area +=
          host.node_r[static_cast<std::size_t>(node0)] *
              host.node_z[static_cast<std::size_t>(node1)] -
          host.node_r[static_cast<std::size_t>(node1)] *
              host.node_z[static_cast<std::size_t>(node0)];
    }
    TENRYU_ASSERT(twice_area != 0.0,
                  "subdomain boundary loop has zero area");
    if (twice_area < 0.0) {
      for (BoundaryEdge& edge : edges) {
        std::swap(edge.node0, edge.node1);
      }
      component.boundary.clear();
      TENRYU_ASSERT(walk_boundary(component.boundary),
                    "subdomain reversed boundary walk inconsistency");
    }
    components.push_back(std::move(component));
  }
  (void)nonsimple_boundary_skip_count;
  return components;
}

bool extract_domain_boundary(const core::State& state,
                             const HostMesh& host,
                             mesh::voronoi::DomainBoundary& domain,
                             std::string& reason,
                             std::vector<int>* surviving_nodes = nullptr) {
  const mesh::MultiBlockTopology& mb = *state.mesh.topo.multiblock;
  if (mb.boundary_faces.size() < 3U) {
    reason = "missing_boundary_faces";
    return false;
  }

  std::map<int, int> next_node;
  std::map<int, int> incoming_count;
  for (const mesh::UniqueOrientedFace& face : mb.boundary_faces) {
    if (face.cell_a < 0 || face.cell_a >= host.n_cells ||
        face.cell_b != -1 || face.local_b != -1) {
      reason = "malformed_boundary_face";
      return false;
    }
    const int nverts = host.nverts[static_cast<std::size_t>(face.cell_a)];
    int corner0 = -1;
    int corner1 = -1;
    if (!mesh::mesh_topo_active_local_face_corners(
            nverts, face.local_a, &corner0, &corner1)) {
      reason = "invalid_boundary_local_face";
      return false;
    }
    const int offset =
        host.csr_offsets[static_cast<std::size_t>(face.cell_a)];
    const int node0 =
        host.csr_indices[static_cast<std::size_t>(offset + corner0)];
    const int node1 =
        host.csr_indices[static_cast<std::size_t>(offset + corner1)];
    if (node0 == node1 || !next_node.emplace(node0, node1).second) {
      reason = "non_simple_boundary_loop";
      return false;
    }
    ++incoming_count[node1];
  }
  if (next_node.size() != mb.boundary_faces.size()) {
    reason = "disconnected_boundary_loop";
    return false;
  }
  for (const auto& [node, next] : next_node) {
    (void)next;
    if (incoming_count[node] != 1) {
      reason = "non_simple_boundary_loop";
      return false;
    }
  }

  const int first = next_node.begin()->first;
  int node = first;
  std::set<int> visited;
  std::vector<int> walked_nodes;
  walked_nodes.reserve(next_node.size());
  while (visited.insert(node).second) {
    domain.r.push_back(host.node_r[static_cast<std::size_t>(node)]);
    domain.z.push_back(host.node_z[static_cast<std::size_t>(node)]);
    walked_nodes.push_back(node);
    const auto next = next_node.find(node);
    if (next == next_node.end()) {
      reason = "open_boundary_loop";
      return false;
    }
    node = next->second;
  }
  if (node != first || visited.size() != next_node.size()) {
    reason = "multiple_boundary_loops";
    return false;
  }

  const std::size_t original_vertex_count = domain.r.size();
  std::vector<bool> remove(original_vertex_count, false);
  std::size_t retained_vertex_count = original_vertex_count;
  for (std::size_t vertex = 0; vertex < original_vertex_count; ++vertex) {
    const std::size_t previous =
        (vertex + original_vertex_count - 1U) % original_vertex_count;
    const std::size_t next = (vertex + 1U) % original_vertex_count;
    if (tenryu::mesh::reale::orient2d_sign(
            domain.r[previous], domain.z[previous],
            domain.r[vertex], domain.z[vertex],
            domain.r[next], domain.z[next]) == 0) {
      remove[vertex] = true;
      --retained_vertex_count;
    }
  }
  // consult-26 Q3(c): the axis is one straight boundary object; the origin is bookkeeping, not a hydro vertex.
  if (retained_vertex_count >= 3U &&
      retained_vertex_count < original_vertex_count) {
    std::vector<double> compact_r;
    std::vector<double> compact_z;
    std::vector<int> compact_nodes;
    compact_r.reserve(retained_vertex_count);
    compact_z.reserve(retained_vertex_count);
    compact_nodes.reserve(retained_vertex_count);
    for (std::size_t vertex = 0; vertex < original_vertex_count; ++vertex) {
      if (!remove[vertex]) {
        compact_r.push_back(domain.r[vertex]);
        compact_z.push_back(domain.z[vertex]);
        compact_nodes.push_back(walked_nodes[vertex]);
      }
    }
    domain.r = std::move(compact_r);
    domain.z = std::move(compact_z);
    walked_nodes = std::move(compact_nodes);
  }

  double twice_area = 0.0;
  bool has_axis_edge = false;
  bool axis_spans_origin = false;
  for (std::size_t vertex = 0; vertex < domain.r.size(); ++vertex) {
    const std::size_t next = (vertex + 1U) % domain.r.size();
    const double r = domain.r[vertex];
    const double z = domain.z[vertex];
    if (!std::isfinite(r) || !std::isfinite(z) || r < 0.0) {
      reason = "invalid_domain_coordinate";
      return false;
    }
    twice_area += r * domain.z[next] - domain.r[next] * z;
    has_axis_edge = has_axis_edge ||
                    (r == 0.0 && domain.r[next] == 0.0);
    axis_spans_origin =
        axis_spans_origin ||
        (r == 0.0 && domain.r[next] == 0.0 &&
         std::min(z, domain.z[next]) <= 0.0 &&
         0.0 <= std::max(z, domain.z[next]));
  }
  if (!(std::abs(twice_area) > 0.0) || !has_axis_edge ||
      !axis_spans_origin) {
    reason = "domain_requires_outer_boundary_axis_and_origin";
    return false;
  }
  if (twice_area < 0.0) {
    std::reverse(domain.r.begin(), domain.r.end());
    std::reverse(domain.z.begin(), domain.z.end());
    std::reverse(walked_nodes.begin(), walked_nodes.end());
  }
  if (surviving_nodes != nullptr) {
    *surviving_nodes = std::move(walked_nodes);
  }
  return true;
}

bool build_generators_and_flow(
    const core::State& state,
    const HostMesh& host,
    std::vector<mesh::voronoi::Generator>& generators,
    std::vector<mesh::voronoi::GeneratorFlowSample>& flow,
    std::string& reason,
    const std::vector<int>* selected_cells = nullptr) {
  const mesh::MultiBlockTopology& mb = *state.mesh.topo.multiblock;
  if (mb.cell_id_stable.size() != static_cast<std::size_t>(host.n_cells) ||
      state.mesh.cell_centroid_r.size() !=
          static_cast<std::size_t>(host.n_cells) ||
      state.mesh.cell_centroid_z.size() !=
          static_cast<std::size_t>(host.n_cells) ||
      state.v_r.size() != static_cast<std::size_t>(host.n_nodes) ||
      state.v_z.size() != static_cast<std::size_t>(host.n_nodes)) {
    reason = "incomplete_generator_source";
    return false;
  }

  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);
  std::set<std::uint64_t> ids;
  const std::size_t selected_count =
      selected_cells == nullptr
          ? static_cast<std::size_t>(host.n_cells)
          : selected_cells->size();
  generators.reserve(selected_count);
  flow.reserve(selected_count);
  for (std::size_t selected = 0; selected < selected_count; ++selected) {
    const int cell = selected_cells == nullptr
                         ? static_cast<int>(selected)
                         : (*selected_cells)[selected];
    if (cell < 0 || cell >= host.n_cells) {
      reason = "invalid_selected_generator_cell";
      return false;
    }
    const int stable = mb.cell_id_stable[static_cast<std::size_t>(cell)];
    const double centroid_r =
        state.mesh.cell_centroid_r[static_cast<std::size_t>(cell)];
    const double centroid_z =
        state.mesh.cell_centroid_z[static_cast<std::size_t>(cell)];
    if (stable < 0 || !std::isfinite(centroid_r) ||
        !std::isfinite(centroid_z) || centroid_r < 0.0) {
      reason = "invalid_generator";
      return false;
    }
    const std::uint64_t id = static_cast<std::uint64_t>(stable);
    if (!ids.insert(id).second) {
      reason = "duplicate_stable_cell_id";
      return false;
    }
    generators.push_back({centroid_r, centroid_z, id});

    const int begin = host.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = host.nverts[static_cast<std::size_t>(cell)];
    double average_r = 0.0;
    double average_z = 0.0;
    for (int corner = 0; corner < nverts; ++corner) {
      const int node =
          host.csr_indices[static_cast<std::size_t>(begin + corner)];
      const double vr = velocity_r[static_cast<std::size_t>(node)];
      const double vz = velocity_z[static_cast<std::size_t>(node)];
      if (!std::isfinite(vr) || !std::isfinite(vz)) {
        reason = "invalid_generator_flow";
        return false;
      }
      average_r += vr;
      average_z += vz;
    }
    flow.push_back(
        {average_r / static_cast<double>(nverts),
         average_z / static_cast<double>(nverts)});
  }
  return true;
}

bool build_staged_mesh(const mesh::voronoi::Tessellation& tessellation,
                       const std::vector<mesh::voronoi::Generator>& generators,
                       const std::vector<int>& old_stable_ids,
                       const int corner_stride,
                       StagedMesh& staged,
                       std::vector<double>& generator_r,
                       std::vector<double>& generator_z,
                       std::vector<int>& neighbor_index,
                       std::string& reason) {
  if (!tessellation.valid ||
      tessellation.cells.size() != old_stable_ids.size()) {
    reason = tessellation.reject_reason.empty()
                 ? "tessellation_cell_count_mismatch"
                 : tessellation.reject_reason;
    return false;
  }

  std::map<std::uint64_t, const mesh::voronoi::VoronoiCell*> cell_by_id;
  for (const mesh::voronoi::VoronoiCell& cell : tessellation.cells) {
    if (!cell_by_id.emplace(cell.generator_id, &cell).second) {
      reason = "duplicate_tessellation_cell_id";
      return false;
    }
  }

  std::map<std::uint64_t, const mesh::voronoi::Generator*> generator_by_id;
  for (const mesh::voronoi::Generator& generator : generators) {
    if (!generator_by_id.emplace(generator.id, &generator).second) {
      reason = "duplicate_target_generator_id";
      return false;
    }
  }

  const int n_cells = static_cast<int>(old_stable_ids.size());
  std::vector<const mesh::voronoi::VoronoiCell*> cells(
      static_cast<std::size_t>(n_cells), nullptr);
  std::vector<int> flat_offsets(static_cast<std::size_t>(n_cells + 1), 0);
  for (int cell = 0; cell < n_cells; ++cell) {
    const int stable = old_stable_ids[static_cast<std::size_t>(cell)];
    if (stable < 0) {
      reason = "invalid_stable_cell_id";
      return false;
    }
    const auto found = cell_by_id.find(static_cast<std::uint64_t>(stable));
    if (found == cell_by_id.end()) {
      reason = "missing_tessellation_cell_id";
      return false;
    }
    cells[static_cast<std::size_t>(cell)] = found->second;
    const std::size_t nverts = found->second->r.size();
    if (nverts < 3U || nverts > static_cast<std::size_t>(corner_stride) ||
        found->second->z.size() != nverts ||
        found->second->neighbor_ids.size() != nverts) {
      reason = "tessellation_valence_out_of_stride_range";
      return false;
    }
    flat_offsets[static_cast<std::size_t>(cell + 1)] =
        flat_offsets[static_cast<std::size_t>(cell)] +
        static_cast<int>(nverts);
  }

  const int flat_count = flat_offsets.back();
  DisjointSet vertex_sets(flat_count);
  std::map<std::uint64_t, int> index_by_id;
  for (int cell = 0; cell < n_cells; ++cell) {
    index_by_id.emplace(
        static_cast<std::uint64_t>(
            old_stable_ids[static_cast<std::size_t>(cell)]),
        cell);
  }

  generator_r.resize(static_cast<std::size_t>(n_cells));
  generator_z.resize(static_cast<std::size_t>(n_cells));
  neighbor_index.assign(
      static_cast<std::size_t>(n_cells * corner_stride), -1);
  for (int cell = 0; cell < n_cells; ++cell) {
    const std::uint64_t generator_id =
        static_cast<std::uint64_t>(
            old_stable_ids[static_cast<std::size_t>(cell)]);
    const auto generator_found = generator_by_id.find(generator_id);
    if (generator_found == generator_by_id.end()) {
      reason = "missing_target_generator_id";
      return false;
    }
    generator_r[static_cast<std::size_t>(cell)] =
        generator_found->second->r;
    generator_z[static_cast<std::size_t>(cell)] =
        generator_found->second->z;
  }

  for (int cell = 0; cell < n_cells; ++cell) {
    const mesh::voronoi::VoronoiCell& current =
        *cells[static_cast<std::size_t>(cell)];
    const int nverts = static_cast<int>(current.r.size());
    for (int edge = 0; edge < nverts; ++edge) {
      const std::uint64_t neighbor_id =
          current.neighbor_ids[static_cast<std::size_t>(edge)];
      if (neighbor_id == mesh::voronoi::kBoundaryNeighbor) {
        continue;
      }
      const auto neighbor_found = index_by_id.find(neighbor_id);
      if (neighbor_found == index_by_id.end()) {
        reason = "unknown_tessellation_neighbor";
        return false;
      }
      const int neighbor_cell = neighbor_found->second;
      neighbor_index[static_cast<std::size_t>(
          cell * corner_stride + edge)] = neighbor_cell;
      if (current.generator_id > neighbor_id) {
        continue;
      }
      const mesh::voronoi::VoronoiCell& neighbor =
          *cells[static_cast<std::size_t>(neighbor_cell)];
      int reciprocal_edge = -1;
      for (int candidate = 0;
           candidate < static_cast<int>(neighbor.r.size()); ++candidate) {
        if (neighbor.neighbor_ids[static_cast<std::size_t>(candidate)] !=
            current.generator_id) {
          continue;
        }
        const int edge_next = (edge + 1) % nverts;
        const int candidate_next =
            (candidate + 1) % static_cast<int>(neighbor.r.size());
        if (close_point(current.r[static_cast<std::size_t>(edge)],
                        current.z[static_cast<std::size_t>(edge)],
                        neighbor.r[static_cast<std::size_t>(candidate_next)],
                        neighbor.z[static_cast<std::size_t>(candidate_next)]) &&
            close_point(current.r[static_cast<std::size_t>(edge_next)],
                        current.z[static_cast<std::size_t>(edge_next)],
                        neighbor.r[static_cast<std::size_t>(candidate)],
                        neighbor.z[static_cast<std::size_t>(candidate)])) {
          if (reciprocal_edge >= 0) {
            reason = "ambiguous_reciprocal_tessellation_edge";
            return false;
          }
          reciprocal_edge = candidate;
        }
      }
      if (reciprocal_edge < 0) {
        reason = "missing_reciprocal_tessellation_edge";
        return false;
      }
      const int neighbor_nverts = static_cast<int>(neighbor.r.size());
      vertex_sets.unite(
          flat_offsets[static_cast<std::size_t>(cell)] + edge,
          flat_offsets[static_cast<std::size_t>(neighbor_cell)] +
              (reciprocal_edge + 1) % neighbor_nverts);
      vertex_sets.unite(
          flat_offsets[static_cast<std::size_t>(cell)] + (edge + 1) % nverts,
          flat_offsets[static_cast<std::size_t>(neighbor_cell)] +
              reciprocal_edge);
    }
  }

  std::map<int, int> node_by_root;
  std::vector<std::pair<int, int>> flat_location(
      static_cast<std::size_t>(flat_count));
  for (int cell = 0; cell < n_cells; ++cell) {
    for (int corner = 0;
         corner < static_cast<int>(
                      cells[static_cast<std::size_t>(cell)]->r.size());
         ++corner) {
      flat_location[static_cast<std::size_t>(
          flat_offsets[static_cast<std::size_t>(cell)] + corner)] =
          {cell, corner};
    }
  }
  for (int flat = 0; flat < flat_count; ++flat) {
    const int root = vertex_sets.find(flat);
    if (node_by_root.find(root) != node_by_root.end()) {
      continue;
    }
    const auto [cell, corner] = flat_location[static_cast<std::size_t>(flat)];
    const mesh::voronoi::VoronoiCell& source =
        *cells[static_cast<std::size_t>(cell)];
    const int node = static_cast<int>(staged.node_r.size());
    node_by_root.emplace(root, node);
    staged.node_r.push_back(source.r[static_cast<std::size_t>(corner)]);
    staged.node_z.push_back(source.z[static_cast<std::size_t>(corner)]);
  }

  staged.n_cells = n_cells;
  staged.n_nodes = static_cast<int>(staged.node_r.size());
  staged.csr_offsets.resize(static_cast<std::size_t>(n_cells + 1));
  staged.csr_indices.assign(
      static_cast<std::size_t>(n_cells * corner_stride), -1);
  staged.nverts.resize(static_cast<std::size_t>(n_cells));
  for (int cell = 0; cell < n_cells; ++cell) {
    const int nverts =
        static_cast<int>(cells[static_cast<std::size_t>(cell)]->r.size());
    staged.csr_offsets[static_cast<std::size_t>(cell)] =
        cell * corner_stride;
    staged.nverts[static_cast<std::size_t>(cell)] =
        static_cast<unsigned char>(nverts);
    for (int corner = 0; corner < nverts; ++corner) {
      const int flat =
          flat_offsets[static_cast<std::size_t>(cell)] + corner;
      staged.csr_indices[static_cast<std::size_t>(
          cell * corner_stride + corner)] =
          node_by_root.at(vertex_sets.find(flat));
    }
  }
  staged.csr_offsets[static_cast<std::size_t>(n_cells)] =
      n_cells * corner_stride;

  staged.node_class.assign(
      static_cast<std::size_t>(staged.n_nodes),
      static_cast<std::uint8_t>(
          mesh::voronoi::TessNodeClass::kInterior));
  staged.node_domain_vertex.assign(
      static_cast<std::size_t>(staged.n_nodes), -1);
  staged.node_carrier_edge.assign(
      static_cast<std::size_t>(staged.n_nodes), -1);
  staged.node_lambda.assign(
      static_cast<std::size_t>(staged.n_nodes), 0.0);
  staged.provenance_conflicts = 0;
  if (!tessellation.node_boundary.empty()) {
    std::vector<bool> provenance_assigned(
        static_cast<std::size_t>(staged.n_nodes), false);
    std::vector<bool> provenance_conflicted(
        static_cast<std::size_t>(staged.n_nodes), false);
    for (int cell = 0; cell < n_cells; ++cell) {
      const mesh::voronoi::VoronoiCell& source =
          *cells[static_cast<std::size_t>(cell)];
      const int nverts = static_cast<int>(source.r.size());
      for (int corner = 0; corner < nverts; ++corner) {
        const int staged_node = staged.csr_indices[static_cast<std::size_t>(
            cell * corner_stride + corner)];
        const int tessellation_node =
            source.node_indices[static_cast<std::size_t>(corner)];
        const mesh::voronoi::TessNodeBoundary& boundary =
            tessellation.node_boundary[static_cast<std::size_t>(
                tessellation_node)];
        const std::size_t node_index =
            static_cast<std::size_t>(staged_node);
        if (provenance_conflicted[node_index]) {
          continue;
        }
        const std::uint8_t node_class =
            static_cast<std::uint8_t>(boundary.cls);
        if (!provenance_assigned[node_index]) {
          staged.node_class[node_index] = node_class;
          staged.node_domain_vertex[node_index] = boundary.domain_vertex;
          staged.node_carrier_edge[node_index] = boundary.carrier_edge;
          staged.node_lambda[node_index] = boundary.lambda_cache;
          provenance_assigned[node_index] = true;
          continue;
        }
        const bool different =
            staged.node_class[node_index] != node_class ||
            staged.node_domain_vertex[node_index] != boundary.domain_vertex ||
            staged.node_carrier_edge[node_index] != boundary.carrier_edge ||
            std::bit_cast<std::uint64_t>(staged.node_lambda[node_index]) !=
                std::bit_cast<std::uint64_t>(boundary.lambda_cache);
        if (different) {
          ++staged.provenance_conflicts;
          provenance_conflicted[node_index] = true;
          staged.node_class[node_index] = static_cast<std::uint8_t>(
              mesh::voronoi::TessNodeClass::kInterior);
          staged.node_domain_vertex[node_index] = -1;
          staged.node_carrier_edge[node_index] = -1;
          staged.node_lambda[node_index] = 0.0;
        }
      }
    }
  }

  using EdgeKey = std::pair<int, int>;
  struct EdgeOwner {
    int cell = -1;
    int node0 = -1;
    int node1 = -1;
    std::uint64_t neighbor = mesh::voronoi::kBoundaryNeighbor;
  };
  std::map<EdgeKey, std::vector<EdgeOwner>> edge_owners;
  for (int cell = 0; cell < n_cells; ++cell) {
    const mesh::voronoi::VoronoiCell& source =
        *cells[static_cast<std::size_t>(cell)];
    const int nverts = staged.nverts[static_cast<std::size_t>(cell)];
    for (int edge = 0; edge < nverts; ++edge) {
      const int node0 = staged.csr_indices[static_cast<std::size_t>(
          cell * corner_stride + edge)];
      const int node1 = staged.csr_indices[static_cast<std::size_t>(
          cell * corner_stride + (edge + 1) % nverts)];
      if (node0 == node1) {
        reason = "collapsed_tessellation_edge";
        return false;
      }
      edge_owners[{std::min(node0, node1), std::max(node0, node1)}]
          .push_back({cell, node0, node1,
                      source.neighbor_ids[static_cast<std::size_t>(edge)]});
    }
  }

  staged.node_flags.assign(static_cast<std::size_t>(staged.n_nodes), 0U);
  for (const auto& [key, owners] : edge_owners) {
    (void)key;
    if (owners.size() == 1U) {
      const EdgeOwner& owner = owners.front();
      if (owner.neighbor != mesh::voronoi::kBoundaryNeighbor) {
        reason = "unpaired_internal_tessellation_edge";
        return false;
      }
      std::uint8_t flags = mesh::NODE_BOUNDARY;
      const double r0 = staged.node_r[static_cast<std::size_t>(owner.node0)];
      const double r1 = staged.node_r[static_cast<std::size_t>(owner.node1)];
      if (r0 == 0.0 && r1 == 0.0) {
        flags = static_cast<std::uint8_t>(
            flags | mesh::NODE_AXIS | mesh::NODE_POLE_AXIS);
      } else {
        flags = static_cast<std::uint8_t>(
            flags | mesh::NODE_OUTER_PHYSICAL_BOUNDARY);
      }
      staged.node_flags[static_cast<std::size_t>(owner.node0)] |= flags;
      staged.node_flags[static_cast<std::size_t>(owner.node1)] |= flags;
      continue;
    }
    if (owners.size() != 2U ||
        owners[0].node0 != owners[1].node1 ||
        owners[0].node1 != owners[1].node0 ||
        owners[0].neighbor != static_cast<std::uint64_t>(
                                  old_stable_ids[static_cast<std::size_t>(
                                      owners[1].cell)]) ||
        owners[1].neighbor != static_cast<std::uint64_t>(
                                  old_stable_ids[static_cast<std::size_t>(
                                      owners[0].cell)])) {
      reason = "non_manifold_tessellation_edge";
      return false;
    }
  }

  int center_count = 0;
  for (int node = 0; node < staged.n_nodes; ++node) {
    if (staged.node_r[static_cast<std::size_t>(node)] == 0.0 &&
        staged.node_z[static_cast<std::size_t>(node)] == 0.0) {
      // On a general-polygonal mesh, the origin is an ordinary axis-boundary vertex; NODE_CENTER is a polar-tier structural contract for pinning and origin-force capture.
      staged.node_flags[static_cast<std::size_t>(node)] |=
          static_cast<std::uint8_t>(mesh::NODE_BOUNDARY | mesh::NODE_AXIS |
                                    mesh::NODE_POLE_AXIS);
      ++center_count;
    }
  }
  if (center_count > 1) {
    reason = "tessellation_requires_at_most_one_origin_node";
    return false;
  }
  if (staged.provenance_conflicts > 0) {
    char warning[128];
    std::snprintf(warning, sizeof(warning),
                  "[carrier] provenance conflicts=%zu",
                  staged.provenance_conflicts);
    core::log_warning(warning);
  }
  return true;
}

bool merge_keeps_cells_starshaped(
    const StagedMesh& mesh,
    const int u,
    const int v,
    const int u_root,
    const int v_root,
    const int proposed_representative,
    const std::vector<int>& parent,
    const std::vector<int>& representative,
    const std::vector<int>& incident_cell_offsets,
    const std::vector<int>& incident_cells) {
  const auto find_root = [&parent](int node) {
    while (parent[static_cast<std::size_t>(node)] != node) {
      node = parent[static_cast<std::size_t>(node)];
    }
    return node;
  };
  const auto mapped_node = [&](const int node) {
    const int root = find_root(node);
    if (root == u_root || root == v_root) {
      return proposed_representative;
    }
    return representative[static_cast<std::size_t>(root)];
  };

  std::set<int> affected_cells;
  for (int entry = incident_cell_offsets[static_cast<std::size_t>(u)];
       entry < incident_cell_offsets[static_cast<std::size_t>(u + 1)];
       ++entry) {
    affected_cells.insert(
        incident_cells[static_cast<std::size_t>(entry)]);
  }
  for (int entry = incident_cell_offsets[static_cast<std::size_t>(v)];
       entry < incident_cell_offsets[static_cast<std::size_t>(v + 1)];
       ++entry) {
    affected_cells.insert(
        incident_cells[static_cast<std::size_t>(entry)]);
  }

  std::vector<int> polygon;
  for (const int cell : affected_cells) {
    polygon.clear();
    const int begin = mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = mesh.nverts[static_cast<std::size_t>(cell)];
    polygon.reserve(static_cast<std::size_t>(nverts));
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = mesh.csr_indices[static_cast<std::size_t>(
          begin + corner)];
      const int mapped = mapped_node(node);
      if (polygon.empty() || polygon.back() != mapped) {
        polygon.push_back(mapped);
      }
    }
    if (polygon.size() > 1 && polygon.front() == polygon.back()) {
      polygon.pop_back();
    }
    if (polygon.size() < 3) {
      return false;
    }

    double center_r = 0.0;
    double center_z = 0.0;
    for (const int node : polygon) {
      center_r += mesh.node_r[static_cast<std::size_t>(node)];
      center_z += mesh.node_z[static_cast<std::size_t>(node)];
    }
    center_r /= static_cast<double>(polygon.size());
    center_z /= static_cast<double>(polygon.size());

    double area2 = 0.0;
    for (std::size_t corner = 0; corner < polygon.size(); ++corner) {
      const std::size_t next = (corner + 1) % polygon.size();
      const int node = polygon[corner];
      const int next_node = polygon[next];
      area2 += mesh.node_r[static_cast<std::size_t>(node)] *
                   mesh.node_z[static_cast<std::size_t>(next_node)] -
               mesh.node_r[static_cast<std::size_t>(next_node)] *
                   mesh.node_z[static_cast<std::size_t>(node)];
    }
    const double total_area = 0.5 * area2;
    if (!(total_area > 0.0) && !(total_area < 0.0)) {
      return false;
    }
    const double margin =
        1.0e-6 * std::abs(total_area) /
        static_cast<double>(polygon.size());
    for (std::size_t corner = 0; corner < polygon.size(); ++corner) {
      const std::size_t next = (corner + 1) % polygon.size();
      const int node = polygon[corner];
      const int next_node = polygon[next];
      const double fan_area =
          0.5 * ((mesh.node_r[static_cast<std::size_t>(node)] - center_r) *
                     (mesh.node_z[static_cast<std::size_t>(next_node)] -
                      center_z) -
                 (mesh.node_r[static_cast<std::size_t>(next_node)] -
                  center_r) *
                     (mesh.node_z[static_cast<std::size_t>(node)] -
                      center_z));
      if (total_area > 0.0) {
        if (!(fan_area > margin)) {
          return false;
        }
      } else if (!(fan_area < -margin)) {
        return false;
      }
    }
  }
  return true;
}

void collapse_short_edges(StagedMesh& mesh, double band_rel) {
  if (band_rel <= 0.0) {
    return;
  }

  const int old_n_nodes = mesh.n_nodes;
  const int corner_stride =
      static_cast<int>(mesh.csr_indices.size()) / mesh.n_cells;
  std::vector<double> node_min_area(
      static_cast<std::size_t>(old_n_nodes),
      std::numeric_limits<double>::infinity());
  for (int cell = 0; cell < mesh.n_cells; ++cell) {
    const int begin = mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = mesh.nverts[static_cast<std::size_t>(cell)];
    TENRYU_ASSERT(nverts >= 3,
                  "install collapse requires nondegenerate input cell");
    double area2 = 0.0;
    for (int corner = 0; corner < nverts; ++corner) {
      const int next = (corner + 1) % nverts;
      const int node = mesh.csr_indices[static_cast<std::size_t>(
          begin + corner)];
      const int next_node = mesh.csr_indices[static_cast<std::size_t>(
          begin + next)];
      area2 += mesh.node_r[static_cast<std::size_t>(node)] *
                   mesh.node_z[static_cast<std::size_t>(next_node)] -
               mesh.node_r[static_cast<std::size_t>(next_node)] *
                   mesh.node_z[static_cast<std::size_t>(node)];
    }
    const double area = 0.5 * std::abs(area2);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = mesh.csr_indices[static_cast<std::size_t>(
          begin + corner)];
      double& minimum = node_min_area[static_cast<std::size_t>(node)];
      minimum = std::min(minimum, area);
    }
  }

  std::vector<int> parent(static_cast<std::size_t>(old_n_nodes));
  std::vector<int> representative(static_cast<std::size_t>(old_n_nodes));
  std::iota(parent.begin(), parent.end(), 0);
  std::iota(representative.begin(), representative.end(), 0);
  auto find_root = [&parent](int node) {
    int root = node;
    while (parent[static_cast<std::size_t>(root)] != root) {
      root = parent[static_cast<std::size_t>(root)];
    }
    while (parent[static_cast<std::size_t>(node)] != node) {
      const int next = parent[static_cast<std::size_t>(node)];
      parent[static_cast<std::size_t>(node)] = root;
      node = next;
    }
    return root;
  };
  const auto node_priority = [&mesh](const int node) {
    const std::uint8_t node_class =
        mesh.node_class[static_cast<std::size_t>(node)];
    if (node_class == static_cast<std::uint8_t>(
                          mesh::voronoi::TessNodeClass::kCarrierMaster)) {
      return 2;
    }
    if (node_class == static_cast<std::uint8_t>(
                          mesh::voronoi::TessNodeClass::kCarrierSlave)) {
      return 1;
    }
    return 0;
  };
  const auto boundary_carrier = [&mesh](const int node) {
    const std::uint8_t node_class =
        mesh.node_class[static_cast<std::size_t>(node)];
    return node_class == static_cast<std::uint8_t>(
                             mesh::voronoi::TessNodeClass::kCarrierMaster) ||
           node_class == static_cast<std::uint8_t>(
                             mesh::voronoi::TessNodeClass::kCarrierSlave);
  };

  std::vector<int> incident_cell_offsets(
      static_cast<std::size_t>(old_n_nodes + 1), 0);
  for (int cell = 0; cell < mesh.n_cells; ++cell) {
    const int begin = mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = mesh.nverts[static_cast<std::size_t>(cell)];
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = mesh.csr_indices[static_cast<std::size_t>(
          begin + corner)];
      ++incident_cell_offsets[static_cast<std::size_t>(node + 1)];
    }
  }
  std::partial_sum(incident_cell_offsets.begin(),
                   incident_cell_offsets.end(),
                   incident_cell_offsets.begin());
  std::vector<int> incident_cells(
      static_cast<std::size_t>(incident_cell_offsets.back()));
  std::vector<int> incident_cell_cursors = incident_cell_offsets;
  for (int cell = 0; cell < mesh.n_cells; ++cell) {
    const int begin = mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = mesh.nverts[static_cast<std::size_t>(cell)];
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = mesh.csr_indices[static_cast<std::size_t>(
          begin + corner)];
      const int entry =
          incident_cell_cursors[static_cast<std::size_t>(node)]++;
      incident_cells[static_cast<std::size_t>(entry)] = cell;
    }
  }

  int admitted_pairs = 0;
  int star_rejected = 0;
  const double band2 = band_rel * band_rel;
  for (int cell = 0; cell < mesh.n_cells; ++cell) {
    const int begin = mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = mesh.nverts[static_cast<std::size_t>(cell)];
    for (int corner = 0; corner < nverts; ++corner) {
      const int u = mesh.csr_indices[static_cast<std::size_t>(
          begin + corner)];
      const int v = mesh.csr_indices[static_cast<std::size_t>(
          begin + (corner + 1) % nverts)];
      if (u == v) {
        continue;
      }
      const double dr = mesh.node_r[static_cast<std::size_t>(u)] -
                        mesh.node_r[static_cast<std::size_t>(v)];
      const double dz = mesh.node_z[static_cast<std::size_t>(u)] -
                        mesh.node_z[static_cast<std::size_t>(v)];
      const double area =
          std::min(node_min_area[static_cast<std::size_t>(u)],
                   node_min_area[static_cast<std::size_t>(v)]);
      if (!(dr * dr + dz * dz < band2 * area)) {
        continue;
      }

      const int u_root = find_root(u);
      const int v_root = find_root(v);
      if (u_root == v_root) {
        ++admitted_pairs;
        continue;
      }
      const int u_representative =
          representative[static_cast<std::size_t>(u_root)];
      const int v_representative =
          representative[static_cast<std::size_t>(v_root)];
      if (node_priority(u_representative) == 2 &&
          node_priority(v_representative) == 2) {
        continue;
      }
      if (boundary_carrier(u_representative) &&
          boundary_carrier(v_representative) &&
          mesh.node_carrier_edge[static_cast<std::size_t>(
              u_representative)] !=
              mesh.node_carrier_edge[static_cast<std::size_t>(
                  v_representative)]) {
        continue;
      }

      int winner_root = u_root;
      int loser_root = v_root;
      const int u_priority = node_priority(u_representative);
      const int v_priority = node_priority(v_representative);
      if (v_priority > u_priority ||
          (v_priority == u_priority &&
           v_representative < u_representative)) {
        winner_root = v_root;
        loser_root = u_root;
      }
      const int proposed_representative =
          winner_root == u_root ? u_representative : v_representative;
      if (!merge_keeps_cells_starshaped(
              mesh, u, v, u_root, v_root, proposed_representative,
              parent, representative, incident_cell_offsets,
              incident_cells)) {
        ++star_rejected;
        continue;
      }

      ++admitted_pairs;
      parent[static_cast<std::size_t>(loser_root)] = winner_root;
      representative[static_cast<std::size_t>(winner_root)] =
          proposed_representative;
    }
  }

  std::vector<int> compacted_indices(mesh.csr_indices.size(), -1);
  for (int cell = 0; cell < mesh.n_cells; ++cell) {
    const int old_begin =
        mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int new_begin = cell * corner_stride;
    const int old_nverts = mesh.nverts[static_cast<std::size_t>(cell)];
    int new_nverts = 0;
    for (int corner = 0; corner < old_nverts; ++corner) {
      const int node = mesh.csr_indices[static_cast<std::size_t>(
          old_begin + corner)];
      const int root = find_root(node);
      const int mapped = representative[static_cast<std::size_t>(root)];
      if (new_nverts == 0 ||
          compacted_indices[static_cast<std::size_t>(
              new_begin + new_nverts - 1)] != mapped) {
        compacted_indices[static_cast<std::size_t>(
            new_begin + new_nverts)] = mapped;
        ++new_nverts;
      }
    }
    if (new_nverts > 1 &&
        compacted_indices[static_cast<std::size_t>(new_begin)] ==
            compacted_indices[static_cast<std::size_t>(
                new_begin + new_nverts - 1)]) {
      compacted_indices[static_cast<std::size_t>(
          new_begin + new_nverts - 1)] = -1;
      --new_nverts;
    }
    TENRYU_ASSERT(new_nverts >= 3,
                  "install collapse produced a degenerate cell");
    mesh.csr_offsets[static_cast<std::size_t>(cell)] = new_begin;
    mesh.nverts[static_cast<std::size_t>(cell)] =
        static_cast<unsigned char>(new_nverts);
  }
  mesh.csr_offsets[static_cast<std::size_t>(mesh.n_cells)] =
      mesh.n_cells * corner_stride;

  std::vector<unsigned char> referenced(
      static_cast<std::size_t>(old_n_nodes), 0U);
  for (int cell = 0; cell < mesh.n_cells; ++cell) {
    const int begin = mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = mesh.nverts[static_cast<std::size_t>(cell)];
    for (int corner = 0; corner < nverts; ++corner) {
      referenced[static_cast<std::size_t>(compacted_indices[
          static_cast<std::size_t>(begin + corner)])] = 1U;
    }
  }

  std::vector<int> compact_node_index(
      static_cast<std::size_t>(old_n_nodes), -1);
  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<std::uint8_t> node_flags;
  std::vector<std::uint8_t> node_class;
  std::vector<int> node_domain_vertex;
  std::vector<int> node_carrier_edge;
  std::vector<double> node_lambda;
  node_r.reserve(static_cast<std::size_t>(old_n_nodes));
  node_z.reserve(static_cast<std::size_t>(old_n_nodes));
  node_flags.reserve(static_cast<std::size_t>(old_n_nodes));
  node_class.reserve(static_cast<std::size_t>(old_n_nodes));
  node_domain_vertex.reserve(static_cast<std::size_t>(old_n_nodes));
  node_carrier_edge.reserve(static_cast<std::size_t>(old_n_nodes));
  node_lambda.reserve(static_cast<std::size_t>(old_n_nodes));
  for (int node = 0; node < old_n_nodes; ++node) {
    if (referenced[static_cast<std::size_t>(node)] == 0U) {
      continue;
    }
    compact_node_index[static_cast<std::size_t>(node)] =
        static_cast<int>(node_r.size());
    node_r.push_back(mesh.node_r[static_cast<std::size_t>(node)]);
    node_z.push_back(mesh.node_z[static_cast<std::size_t>(node)]);
    node_flags.push_back(mesh.node_flags[static_cast<std::size_t>(node)]);
    node_class.push_back(mesh.node_class[static_cast<std::size_t>(node)]);
    node_domain_vertex.push_back(
        mesh.node_domain_vertex[static_cast<std::size_t>(node)]);
    node_carrier_edge.push_back(
        mesh.node_carrier_edge[static_cast<std::size_t>(node)]);
    node_lambda.push_back(mesh.node_lambda[static_cast<std::size_t>(node)]);
  }
  for (int cell = 0; cell < mesh.n_cells; ++cell) {
    const int begin = mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = mesh.nverts[static_cast<std::size_t>(cell)];
    for (int corner = 0; corner < nverts; ++corner) {
      int& node = compacted_indices[static_cast<std::size_t>(
          begin + corner)];
      node = compact_node_index[static_cast<std::size_t>(node)];
    }
  }

  mesh.csr_indices = std::move(compacted_indices);
  mesh.node_r = std::move(node_r);
  mesh.node_z = std::move(node_z);
  mesh.node_flags = std::move(node_flags);
  mesh.node_class = std::move(node_class);
  mesh.node_domain_vertex = std::move(node_domain_vertex);
  mesh.node_carrier_edge = std::move(node_carrier_edge);
  mesh.node_lambda = std::move(node_lambda);
  mesh.n_nodes = static_cast<int>(mesh.node_r.size());

  const int removed = old_n_nodes - mesh.n_nodes;
  if (removed > 0) {
    char collapse_log[128];
    std::snprintf(collapse_log, sizeof(collapse_log),
                  "[install_collapse] pairs=%d removed=%d nodes=%d "
                  "star_rejected=%d",
                  admitted_pairs, removed, mesh.n_nodes, star_rejected);
    core::log_info(collapse_log);
  }
}

struct SubdomainTessellationRun {
  mesh::voronoi::DomainBoundary domain;
  std::vector<int> boundary_old_nodes;
  mesh::voronoi::TargetResult targets;
  StagedMesh mesh;
  std::vector<double> generator_r;
  std::vector<double> generator_z;
  std::vector<int> neighbor_index;
  std::vector<int> stable_ids;
};

struct SubdomainSplice {
  StagedMesh mesh;
  std::vector<std::vector<int>> run_node_to_global;
  std::vector<unsigned char> fresh_node;
  std::vector<unsigned char> hanging_cell;
  int rezoned_cells = 0;
  bool has_new_physical_slaves = false;
};

bool build_subdomain_tessellation(
    const core::State& state, const core::Config& cfg,
    const HostMesh& old_mesh, const SubdomainComponent& component,
    const double dt, SubdomainTessellationRun& run,
    std::string& stage, std::string& reason) {
  run.domain.r.reserve(component.boundary.size());
  run.domain.z.reserve(component.boundary.size());
  run.boundary_old_nodes.reserve(component.boundary.size());
  for (const SubdomainBoundaryEntry& entry : component.boundary) {
    run.domain.r.push_back(
        old_mesh.node_r[static_cast<std::size_t>(entry.node)]);
    run.domain.z.push_back(
        old_mesh.node_z[static_cast<std::size_t>(entry.node)]);
    run.boundary_old_nodes.push_back(entry.node);
  }

  std::vector<mesh::voronoi::Generator> generators;
  std::vector<mesh::voronoi::GeneratorFlowSample> flow;
  if (!build_generators_and_flow(state, old_mesh, generators, flow,
                                 reason, &component.cells)) {
    stage = "generators";
    return false;
  }

  std::optional<mesh::tess::DelaunayTriangulation> warm_dt;
  run.targets = mesh::voronoi::build_generator_targets(
      generators, flow, run.domain, dt,
      cfg.numerics.ale.reale_core == "exact", &warm_dt);
  if (!run.targets.valid) {
    stage = "targets";
    reason = run.targets.reject_reason;
    return false;
  }
  const mesh::voronoi::Tessellation& tessellation =
      run.targets.accepted_tessellation;
  if (!tessellation.valid) {
    stage = "voronoi";
    reason = tessellation.reject_reason;
    return false;
  }

  const mesh::MultiBlockTopology& topology =
      *state.mesh.topo.multiblock;
  run.stable_ids.reserve(component.cells.size());
  for (const int cell : component.cells) {
    run.stable_ids.push_back(
        topology.cell_id_stable[static_cast<std::size_t>(cell)]);
  }
  if (!build_staged_mesh(
          tessellation, run.targets.targets, run.stable_ids,
          state.corner_stride, run.mesh, run.generator_r,
          run.generator_z, run.neighbor_index, reason)) {
    stage = "topology";
    return false;
  }
  collapse_short_edges(
      run.mesh, cfg.numerics.ale.reale_short_edge_collapse_rel);

  std::vector<int> master_occurrences(component.boundary.size(), 0);
  for (int node = 0; node < run.mesh.n_nodes; ++node) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    if (run.mesh.node_class[node_index] != static_cast<std::uint8_t>(
            mesh::voronoi::TessNodeClass::kCarrierMaster)) {
      continue;
    }
    const int domain_vertex = run.mesh.node_domain_vertex[node_index];
    if (domain_vertex >= 0 &&
        static_cast<std::size_t>(domain_vertex) <
            master_occurrences.size()) {
      ++master_occurrences[static_cast<std::size_t>(domain_vertex)];
    }
  }
  if (std::any_of(master_occurrences.begin(), master_occurrences.end(),
                  [](const int count) { return count != 1; })) {
    stage = "topology";
    reason = "subdomain_carrier_master_missing";
    return false;
  }
  if (!tessellation.carrier_representability_ok) {
    stage = "topology";
    reason = "subdomain_carrier_representability";
    return false;
  }
  if (!tessellation.carrier_coverage_ok) {
    stage = "topology";
    reason = "subdomain_carrier_coverage";
    return false;
  }
  if (run.mesh.provenance_conflicts > 0) {
    stage = "topology";
    reason = "subdomain_carrier_provenance_conflict";
    return false;
  }
  return true;
}

bool splice_subdomain_mesh(
    const core::State& state, const HostMesh& old_mesh,
    const std::vector<SubdomainComponent>& components,
    const std::vector<SubdomainTessellationRun>& runs,
    SubdomainSplice& splice, std::string& reason) {
  TENRYU_ASSERT(components.size() == runs.size(),
                "subdomain splice component/run count mismatch");
  const int corner_stride = state.corner_stride;
  const mesh::MultiBlockTopology& topology =
      *state.mesh.topo.multiblock;
  StagedMesh& result = splice.mesh;
  result.n_cells = old_mesh.n_cells;
  result.n_nodes = old_mesh.n_nodes;
  result.node_r = old_mesh.node_r;
  result.node_z = old_mesh.node_z;
  TENRYU_ASSERT(state.mesh.topo.node_flags.size() ==
                    static_cast<std::size_t>(old_mesh.n_nodes),
                "subdomain splice node-flag size mismatch");
  result.node_flags = state.mesh.topo.node_flags;
  result.node_class.assign(
      static_cast<std::size_t>(old_mesh.n_nodes),
      static_cast<std::uint8_t>(
          mesh::voronoi::TessNodeClass::kInterior));
  result.node_domain_vertex.assign(
      static_cast<std::size_t>(old_mesh.n_nodes), -1);
  result.node_carrier_edge.assign(
      static_cast<std::size_t>(old_mesh.n_nodes), -1);
  result.node_lambda.assign(
      static_cast<std::size_t>(old_mesh.n_nodes), 0.0);
  const bool has_persistent_node_metadata =
      state.carrier_node_class.size() ==
          static_cast<std::size_t>(old_mesh.n_nodes) &&
      state.carrier_node_edge.size() ==
          static_cast<std::size_t>(old_mesh.n_nodes) &&
      state.carrier_node_lambda.size() ==
          static_cast<std::size_t>(old_mesh.n_nodes);
  TENRYU_ASSERT(
      has_persistent_node_metadata ||
          (state.carrier_node_class.empty() &&
           state.carrier_node_edge.empty() &&
           state.carrier_node_lambda.empty()),
      "subdomain splice carrier metadata size mismatch");
  if (has_persistent_node_metadata) {
    result.node_class = state.carrier_node_class;
    result.node_carrier_edge = state.carrier_node_edge;
    result.node_lambda = state.carrier_node_lambda;
  }

  std::map<int, int> master_by_node;
  if (state.boundary_carrier.valid) {
    for (std::size_t master = 0;
         master < state.boundary_carrier.masters.size(); ++master) {
      const int node = state.boundary_carrier.masters[master].mesh_node;
      TENRYU_ASSERT(node >= 0 && node < old_mesh.n_nodes &&
                        master_by_node.emplace(
                            node, static_cast<int>(master)).second,
                    "subdomain splice carrier master identity");
      result.node_class[static_cast<std::size_t>(node)] =
          static_cast<std::uint8_t>(
              mesh::voronoi::TessNodeClass::kCarrierMaster);
      result.node_domain_vertex[static_cast<std::size_t>(node)] =
          static_cast<int>(master);
    }
  }

  const auto endpoint_carrier_edges = [&](const int node) {
    std::vector<int> edges;
    const auto master = master_by_node.find(node);
    if (master != master_by_node.end()) {
      const int n_masters =
          static_cast<int>(state.boundary_carrier.masters.size());
      edges.push_back((master->second - 1 + n_masters) % n_masters);
      edges.push_back(master->second);
    } else if (result.node_class[static_cast<std::size_t>(node)] ==
               static_cast<std::uint8_t>(
                   mesh::voronoi::TessNodeClass::kCarrierSlave)) {
      edges.push_back(
          result.node_carrier_edge[static_cast<std::size_t>(node)]);
    }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    return edges;
  };
  const auto endpoint_lambda = [&](const int node, const int edge) {
    const auto master = master_by_node.find(node);
    if (master == master_by_node.end()) {
      TENRYU_ASSERT(
          result.node_class[static_cast<std::size_t>(node)] ==
                  static_cast<std::uint8_t>(
                      mesh::voronoi::TessNodeClass::kCarrierSlave) &&
              result.node_carrier_edge[static_cast<std::size_t>(node)] ==
                  edge,
          "subdomain physical segment endpoint carrier mismatch");
      return result.node_lambda[static_cast<std::size_t>(node)];
    }
    const int n_masters =
        static_cast<int>(state.boundary_carrier.masters.size());
    TENRYU_ASSERT(edge == master->second ||
                      (edge + 1) % n_masters == master->second,
                  "subdomain physical segment master edge mismatch");
    return edge == master->second ? 0.0 : 1.0;
  };
  const auto physical_carrier_edge = [&](const int node0,
                                         const int node1) {
    TENRYU_ASSERT(state.boundary_carrier.valid &&
                      state.boundary_carrier.masters.size() >= 3U,
                  "subdomain true boundary requires physical carrier");
    const std::vector<int> edges0 = endpoint_carrier_edges(node0);
    const std::vector<int> edges1 = endpoint_carrier_edges(node1);
    std::vector<int> common;
    std::set_intersection(edges0.begin(), edges0.end(), edges1.begin(),
                          edges1.end(), std::back_inserter(common));
    TENRYU_ASSERT(common.size() == 1U,
                  "subdomain true-boundary endpoints must carry one "
                  "common physical carrier edge");
    return common.front();
  };

  splice.run_node_to_global.resize(runs.size());
  splice.fresh_node.assign(static_cast<std::size_t>(old_mesh.n_nodes), 0U);
  std::vector<std::vector<std::vector<int>>> segment_nodes(runs.size());
  for (std::size_t component_index = 0;
       component_index < components.size(); ++component_index) {
    const SubdomainComponent& component = components[component_index];
    const SubdomainTessellationRun& run = runs[component_index];
    std::vector<int>& node_map =
        splice.run_node_to_global[component_index];
    node_map.assign(static_cast<std::size_t>(run.mesh.n_nodes), -1);
    segment_nodes[component_index].resize(component.boundary.size());

    for (int node = 0; node < run.mesh.n_nodes; ++node) {
      const std::size_t node_index = static_cast<std::size_t>(node);
      if (run.mesh.node_class[node_index] !=
          static_cast<std::uint8_t>(
              mesh::voronoi::TessNodeClass::kCarrierMaster)) {
        continue;
      }
      const int domain_vertex = run.mesh.node_domain_vertex[node_index];
      TENRYU_ASSERT(
          domain_vertex >= 0 &&
              static_cast<std::size_t>(domain_vertex) <
                  run.boundary_old_nodes.size(),
          "subdomain splice temporary master index");
      const int old_node = run.boundary_old_nodes[
          static_cast<std::size_t>(domain_vertex)];
      TENRYU_ASSERT(
          close_point(run.mesh.node_r[node_index],
                      run.mesh.node_z[node_index],
                      old_mesh.node_r[static_cast<std::size_t>(old_node)],
                      old_mesh.node_z[static_cast<std::size_t>(old_node)]),
          "subdomain splice temporary master coordinate");
      node_map[node_index] = old_node;
    }

    for (int node = 0; node < run.mesh.n_nodes; ++node) {
      const std::size_t node_index = static_cast<std::size_t>(node);
      if (node_map[node_index] >= 0) {
        continue;
      }
      const int global_node = result.n_nodes++;
      node_map[node_index] = global_node;
      result.node_r.push_back(run.mesh.node_r[node_index]);
      result.node_z.push_back(run.mesh.node_z[node_index]);
      result.node_flags.push_back(run.mesh.node_flags[node_index]);
      result.node_class.push_back(static_cast<std::uint8_t>(
          mesh::voronoi::TessNodeClass::kInterior));
      result.node_domain_vertex.push_back(-1);
      result.node_carrier_edge.push_back(-1);
      result.node_lambda.push_back(0.0);
      splice.fresh_node.push_back(1U);

      if (run.mesh.node_class[node_index] !=
          static_cast<std::uint8_t>(
              mesh::voronoi::TessNodeClass::kCarrierSlave)) {
        continue;
      }
      const int segment = run.mesh.node_carrier_edge[node_index];
      TENRYU_ASSERT(
          segment >= 0 &&
              static_cast<std::size_t>(segment) <
                  component.boundary.size(),
          "subdomain splice temporary slave edge");
      segment_nodes[component_index][static_cast<std::size_t>(segment)]
          .push_back(node);
      const SubdomainBoundaryEntry& entry =
          component.boundary[static_cast<std::size_t>(segment)];
      const int face_offset = topology.face_adj_csr_offsets[
          static_cast<std::size_t>(entry.cell)];
      const int neighbor = topology.face_adj_csr_indices[
          static_cast<std::size_t>(face_offset + entry.local_face)];
      if (neighbor >= 0) {
        result.node_flags[static_cast<std::size_t>(global_node)] =
            mesh::NODE_NONE;
        continue;
      }

      const int node0 = entry.node;
      const int node1 = component.boundary[
          (static_cast<std::size_t>(segment) + 1U) %
          component.boundary.size()]
                            .node;
      const int carrier_edge = physical_carrier_edge(node0, node1);
      const double segment_length = std::hypot(
          old_mesh.node_r[static_cast<std::size_t>(node1)] -
              old_mesh.node_r[static_cast<std::size_t>(node0)],
          old_mesh.node_z[static_cast<std::size_t>(node1)] -
              old_mesh.node_z[static_cast<std::size_t>(node0)]);
      TENRYU_ASSERT(std::isfinite(segment_length) && segment_length > 0.0,
                    "subdomain physical carrier segment length");
      const double fraction =
          std::hypot(run.mesh.node_r[node_index] -
                         old_mesh.node_r[static_cast<std::size_t>(node0)],
                     run.mesh.node_z[node_index] -
                         old_mesh.node_z[static_cast<std::size_t>(node0)]) /
          segment_length;
      const double lambda0 = endpoint_lambda(node0, carrier_edge);
      const double lambda1 = endpoint_lambda(node1, carrier_edge);
      const double lambda = lambda0 + fraction * (lambda1 - lambda0);
      TENRYU_ASSERT(std::isfinite(lambda) && lambda > 0.0 &&
                        lambda < 1.0,
                    "subdomain physical carrier slave lambda");
      result.node_flags[static_cast<std::size_t>(global_node)] =
          run.mesh.node_flags[node_index];
      TENRYU_ASSERT(
          (result.node_flags[static_cast<std::size_t>(global_node)] &
           mesh::NODE_BOUNDARY) != 0U,
          "subdomain physical carrier slave boundary flags");
      result.node_class[static_cast<std::size_t>(global_node)] =
          static_cast<std::uint8_t>(
              mesh::voronoi::TessNodeClass::kCarrierSlave);
      result.node_carrier_edge[static_cast<std::size_t>(global_node)] =
          carrier_edge;
      result.node_lambda[static_cast<std::size_t>(global_node)] = lambda;
      splice.has_new_physical_slaves = true;
    }
    splice.rezoned_cells += static_cast<int>(component.cells.size());
  }

  std::vector<int> component_of_cell(
      static_cast<std::size_t>(old_mesh.n_cells), -1);
  std::vector<int> local_cell(
      static_cast<std::size_t>(old_mesh.n_cells), -1);
  for (std::size_t component_index = 0;
       component_index < components.size(); ++component_index) {
    const SubdomainComponent& component = components[component_index];
    for (std::size_t local = 0; local < component.cells.size(); ++local) {
      const int cell = component.cells[local];
      TENRYU_ASSERT(component_of_cell[static_cast<std::size_t>(cell)] < 0,
                    "subdomain splice overlapping components");
      component_of_cell[static_cast<std::size_t>(cell)] =
          static_cast<int>(component_index);
      local_cell[static_cast<std::size_t>(cell)] =
          static_cast<int>(local);
    }
  }

  struct CellEdgeInsertion {
    int node0 = -1;
    int node1 = -1;
    std::vector<int> nodes;
  };
  std::vector<std::vector<CellEdgeInsertion>> insertions(
      static_cast<std::size_t>(old_mesh.n_cells));
  splice.hanging_cell.assign(
      static_cast<std::size_t>(old_mesh.n_cells), 0U);
  for (std::size_t component_index = 0;
       component_index < components.size(); ++component_index) {
    const SubdomainComponent& component = components[component_index];
    const SubdomainTessellationRun& run = runs[component_index];
    const std::vector<int>& node_map =
        splice.run_node_to_global[component_index];
    for (std::size_t segment = 0; segment < component.boundary.size();
         ++segment) {
      const SubdomainBoundaryEntry& entry = component.boundary[segment];
      const int face_offset = topology.face_adj_csr_offsets[
          static_cast<std::size_t>(entry.cell)];
      const int neighbor = topology.face_adj_csr_indices[
          static_cast<std::size_t>(face_offset + entry.local_face)];
      if (neighbor < 0) {
        continue;
      }
      TENRYU_ASSERT(
          component.membership[static_cast<std::size_t>(neighbor)] == 0U,
          "subdomain splice interface neighbor membership");
      std::vector<int>& temporary_nodes =
          segment_nodes[component_index][segment];
      std::sort(temporary_nodes.begin(), temporary_nodes.end(),
                [&](const int lhs, const int rhs) {
                  return run.mesh.node_lambda[static_cast<std::size_t>(lhs)] <
                         run.mesh.node_lambda[static_cast<std::size_t>(rhs)];
                });
      if (temporary_nodes.empty()) {
        continue;
      }
      CellEdgeInsertion insertion;
      insertion.node0 = entry.node;
      insertion.node1 = component.boundary[
          (segment + 1U) % component.boundary.size()]
                            .node;
      insertion.nodes.reserve(temporary_nodes.size());
      for (const int node : temporary_nodes) {
        insertion.nodes.push_back(
            node_map[static_cast<std::size_t>(node)]);
      }
      insertions[static_cast<std::size_t>(neighbor)].push_back(
          std::move(insertion));
      splice.hanging_cell[static_cast<std::size_t>(neighbor)] = 1U;
    }
  }

  result.csr_offsets.resize(
      static_cast<std::size_t>(old_mesh.n_cells + 1));
  result.csr_indices.assign(
      static_cast<std::size_t>(old_mesh.n_cells * corner_stride), -1);
  result.nverts.resize(static_cast<std::size_t>(old_mesh.n_cells));
  for (int cell = 0; cell < old_mesh.n_cells; ++cell) {
    const int destination = cell * corner_stride;
    result.csr_offsets[static_cast<std::size_t>(cell)] = destination;
    const int component_index =
        component_of_cell[static_cast<std::size_t>(cell)];
    if (component_index >= 0) {
      const SubdomainTessellationRun& run =
          runs[static_cast<std::size_t>(component_index)];
      const std::vector<int>& node_map =
          splice.run_node_to_global[static_cast<std::size_t>(
              component_index)];
      const int local = local_cell[static_cast<std::size_t>(cell)];
      const int source = run.mesh.csr_offsets[static_cast<std::size_t>(local)];
      const int nverts = run.mesh.nverts[static_cast<std::size_t>(local)];
      result.nverts[static_cast<std::size_t>(cell)] =
          static_cast<unsigned char>(nverts);
      for (int corner = 0; corner < nverts; ++corner) {
        const int run_node = run.mesh.csr_indices[
            static_cast<std::size_t>(source + corner)];
        result.csr_indices[static_cast<std::size_t>(destination + corner)] =
            node_map[static_cast<std::size_t>(run_node)];
      }
      continue;
    }

    const int source = old_mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int old_nverts = old_mesh.nverts[static_cast<std::size_t>(cell)];
    if (insertions[static_cast<std::size_t>(cell)].empty()) {
      result.nverts[static_cast<std::size_t>(cell)] =
          old_mesh.nverts[static_cast<std::size_t>(cell)];
      std::copy_n(old_mesh.csr_indices.begin() + source, corner_stride,
                  result.csr_indices.begin() + destination);
      continue;
    }
    std::size_t required_nverts = static_cast<std::size_t>(old_nverts);
    for (const CellEdgeInsertion& insertion :
         insertions[static_cast<std::size_t>(cell)]) {
      required_nverts += insertion.nodes.size();
    }
    if (required_nverts > static_cast<std::size_t>(corner_stride)) {
      reason = "subdomain_hanging_vertex_exceeds_corner_stride";
      return false;
    }
    int output_corner = 0;
    int matched_insertions = 0;
    for (int corner = 0; corner < old_nverts; ++corner) {
      const int next = (corner + 1) % old_nverts;
      const int node0 = old_mesh.csr_indices[
          static_cast<std::size_t>(source + corner)];
      const int node1 = old_mesh.csr_indices[
          static_cast<std::size_t>(source + next)];
      result.csr_indices[
          static_cast<std::size_t>(destination + output_corner++)] = node0;
      for (const CellEdgeInsertion& insertion :
           insertions[static_cast<std::size_t>(cell)]) {
        const bool forward = insertion.node0 == node0 &&
                             insertion.node1 == node1;
        const bool reverse = insertion.node0 == node1 &&
                             insertion.node1 == node0;
        if (!forward && !reverse) {
          continue;
        }
        ++matched_insertions;
        if (output_corner + static_cast<int>(insertion.nodes.size()) >
            corner_stride) {
          reason = "subdomain_hanging_vertex_exceeds_corner_stride";
          return false;
        }
        if (forward) {
          for (const int node : insertion.nodes) {
            result.csr_indices[static_cast<std::size_t>(
                destination + output_corner++)] = node;
          }
        } else {
          for (auto node = insertion.nodes.rbegin();
               node != insertion.nodes.rend(); ++node) {
            result.csr_indices[static_cast<std::size_t>(
                destination + output_corner++)] = *node;
          }
        }
      }
    }
    TENRYU_ASSERT(
        matched_insertions ==
            static_cast<int>(insertions[static_cast<std::size_t>(cell)].size()),
        "subdomain hanging insertion edge missing");
    if (output_corner > corner_stride) {
      reason = "subdomain_hanging_vertex_exceeds_corner_stride";
      return false;
    }
    result.nverts[static_cast<std::size_t>(cell)] =
        static_cast<unsigned char>(output_corner);
  }
  result.csr_offsets[static_cast<std::size_t>(old_mesh.n_cells)] =
      old_mesh.n_cells * corner_stride;
  return true;
}

hydro::reale::RemapMesh remap_view(const HostMesh& mesh,
                                   const int corner_stride,
                                   const double* generator_r = nullptr,
                                   const double* generator_z = nullptr,
                                   const int* neighbor_index = nullptr) {
  return {mesh.n_cells, corner_stride, mesh.csr_offsets.data(),
          mesh.csr_indices.data(), mesh.nverts.data(), mesh.node_r.data(),
          mesh.node_z.data(), generator_r, generator_z, neighbor_index};
}

// Host replica of compatible_subzonal_pressure.cu::triangle_corner_volumes,
// used after ReALE clears the structured polar-tier topology metadata.
void compute_reale_triangle_corner_volumes(const double r0,
                                           const double z0,
                                           const double r1,
                                           const double z1,
                                           const double r2,
                                           const double z2,
                                           double* volume) {
  const double r[3] = {r0, r1, r2};
  const double z[3] = {z0, z1, z2};
  const double rc = (r[0] + r[1] + r[2]) / 3.0;
  const double zc = (z[0] + z[1] + z[2]) / 3.0;
  for (int corner = 0; corner < 3; ++corner) {
    const int next = (corner + 1) % 3;
    const int previous = (corner + 2) % 3;
    const double sub_r[4] = {
        r[corner], 0.5 * (r[corner] + r[next]), rc,
        0.5 * (r[previous] + r[corner])};
    const double sub_z[4] = {
        z[corner], 0.5 * (z[corner] + z[next]), zc,
        0.5 * (z[previous] + z[corner])};
    volume[corner] =
        std::fabs(hydro::rz::rz_polygon_volume_exact(sub_r, sub_z, 4));
  }
  volume[3] = 0.0;
}

// Host replica of
// compatible_subzonal_pressure.cu::polygon_corner_rz_volumes_slot_cap,
// using the same mean-center/edge-midpoint RZ subpolygons.
void compute_reale_polygon_corner_volumes(
    const hydro::PentagonPoint* points,
    const int nverts,
    double* volume) {
  const hydro::PentagonPoint center = hydro::polygon_center(points, nverts);
  hydro::PentagonPoint
      midpoints[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  hydro::polygon_edge_midpoints_n(points, nverts, midpoints);
  for (int corner = 0; corner < nverts; ++corner) {
    const int previous = (corner == 0) ? nverts - 1 : corner - 1;
    const hydro::PentagonPoint subpolygon[4] = {
        points[corner], midpoints[corner], center, midpoints[previous]};
    volume[corner] = hydro::polygon_rz_volume(subpolygon, 4);
  }
}

void reinitialize_corner_mass_after_remap(
    const StagedMesh& new_mesh,
    const int corner_stride,
    const std::vector<double>& cell_mass,
    std::vector<double>& corner_mass,
    const std::vector<unsigned char>* selected_cells = nullptr) {
  TENRYU_ASSERT(cell_mass.size() ==
                    static_cast<std::size_t>(new_mesh.n_cells),
                "ReALE corner-mass reset cell-mass size mismatch");
  TENRYU_ASSERT(
      corner_mass.size() ==
          static_cast<std::size_t>(new_mesh.n_cells * corner_stride),
      "ReALE corner-mass reset corner-mass size mismatch");
  TENRYU_ASSERT(selected_cells == nullptr ||
                    selected_cells->size() ==
                        static_cast<std::size_t>(new_mesh.n_cells),
                "ReALE corner-mass reset cell-selection size mismatch");
  for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
    if (selected_cells != nullptr &&
        (*selected_cells)[static_cast<std::size_t>(cell)] == 0U) {
      continue;
    }
    const int nverts = new_mesh.nverts[static_cast<std::size_t>(cell)];
    TENRYU_ASSERT(nverts >= 3 &&
                      nverts <= mesh::kMeshTopoCellStorageSlotsMaxGeneral &&
                      nverts <= corner_stride,
                  "ReALE corner-mass reset unsupported cell valence");
    const int offset =
        new_mesh.csr_offsets[static_cast<std::size_t>(cell)];
    double volume[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    if (nverts == 3) {
      const int n0 =
          new_mesh.csr_indices[static_cast<std::size_t>(offset + 0)];
      const int n1 =
          new_mesh.csr_indices[static_cast<std::size_t>(offset + 1)];
      const int n2 =
          new_mesh.csr_indices[static_cast<std::size_t>(offset + 2)];
      compute_reale_triangle_corner_volumes(
          new_mesh.node_r[static_cast<std::size_t>(n0)],
          new_mesh.node_z[static_cast<std::size_t>(n0)],
          new_mesh.node_r[static_cast<std::size_t>(n1)],
          new_mesh.node_z[static_cast<std::size_t>(n1)],
          new_mesh.node_r[static_cast<std::size_t>(n2)],
          new_mesh.node_z[static_cast<std::size_t>(n2)], volume);
    } else if (nverts == 4) {
      const int n0 =
          new_mesh.csr_indices[static_cast<std::size_t>(offset + 0)];
      const int n1 =
          new_mesh.csr_indices[static_cast<std::size_t>(offset + 1)];
      const int n2 =
          new_mesh.csr_indices[static_cast<std::size_t>(offset + 2)];
      const int n3 =
          new_mesh.csr_indices[static_cast<std::size_t>(offset + 3)];
      hydro::rz::compute_quad_corner_volumes_exact_subpolygon(
          new_mesh.node_r[static_cast<std::size_t>(n0)],
          new_mesh.node_z[static_cast<std::size_t>(n0)],
          new_mesh.node_r[static_cast<std::size_t>(n1)],
          new_mesh.node_z[static_cast<std::size_t>(n1)],
          new_mesh.node_r[static_cast<std::size_t>(n2)],
          new_mesh.node_z[static_cast<std::size_t>(n2)],
          new_mesh.node_r[static_cast<std::size_t>(n3)],
          new_mesh.node_z[static_cast<std::size_t>(n3)], volume);
    } else {
      hydro::PentagonPoint
          points[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
      for (int corner = 0; corner < nverts; ++corner) {
        const int node = new_mesh.csr_indices[
            static_cast<std::size_t>(offset + corner)];
        points[corner] = {
            new_mesh.node_r[static_cast<std::size_t>(node)],
            new_mesh.node_z[static_cast<std::size_t>(node)]};
      }
      if (nverts == 5) {
        hydro::pentagon_corner_rz_volumes(points, volume);
      } else {
        compute_reale_polygon_corner_volumes(points, nverts, volume);
      }
    }

    double cell_volume = 0.0;
    for (int corner = 0; corner < nverts; ++corner) {
      TENRYU_ASSERT(std::isfinite(volume[corner]) && volume[corner] >= 0.0,
                    "ReALE corner-mass reset invalid corner volume");
      cell_volume += volume[corner];
    }
    TENRYU_ASSERT(std::isfinite(cell_volume) && cell_volume > 0.0,
                  "ReALE corner-mass reset invalid cell volume");
    const std::size_t base =
        static_cast<std::size_t>(cell * corner_stride);
    for (int corner = 0; corner < nverts; ++corner) {
      corner_mass[base + static_cast<std::size_t>(corner)] =
          cell_mass[static_cast<std::size_t>(cell)] * volume[corner] /
          cell_volume;
    }
    for (int corner = nverts; corner < corner_stride; ++corner) {
      corner_mass[base + static_cast<std::size_t>(corner)] = 0.0;
    }
  }
}

HostMesh extract_cell_subset(const HostMesh& source,
                             const std::vector<int>& cells,
                             const int corner_stride) {
  HostMesh subset;
  subset.n_cells = static_cast<int>(cells.size());
  subset.n_nodes = source.n_nodes;
  subset.node_r = source.node_r;
  subset.node_z = source.node_z;
  subset.csr_offsets.resize(cells.size() + 1U);
  subset.csr_indices.assign(cells.size() *
                                static_cast<std::size_t>(corner_stride),
                            -1);
  subset.nverts.resize(cells.size());
  for (std::size_t local = 0; local < cells.size(); ++local) {
    const int cell = cells[local];
    TENRYU_ASSERT(cell >= 0 && cell < source.n_cells,
                  "subdomain remap subset cell range");
    const int nverts = source.nverts[static_cast<std::size_t>(cell)];
    const int source_offset =
        source.csr_offsets[static_cast<std::size_t>(cell)];
    const int target_offset = static_cast<int>(local) * corner_stride;
    subset.csr_offsets[local] = target_offset;
    subset.nverts[local] = static_cast<unsigned char>(nverts);
    for (int corner = 0; corner < nverts; ++corner) {
      subset.csr_indices[static_cast<std::size_t>(target_offset + corner)] =
          source.csr_indices[
              static_cast<std::size_t>(source_offset + corner)];
    }
  }
  subset.csr_offsets[cells.size()] =
      static_cast<int>(cells.size()) * corner_stride;
  return subset;
}

bool remap_subdomain_components(
    const core::State& state, const core::Config& cfg,
    const HostMesh& old_mesh,
    const std::vector<SubdomainComponent>& components,
    const std::vector<SubdomainTessellationRun>& runs,
    const SubdomainSplice& splice,
    const std::vector<double>& old_mass,
    const std::vector<double>& old_ee,
    const std::vector<double>& old_ei,
    const std::vector<double>& old_corner_mass,
    const std::vector<double>& old_velocity_r,
    const std::vector<double>& old_velocity_z,
    std::vector<double>& new_mass,
    std::vector<double>& new_ee,
    std::vector<double>& new_ei,
    std::vector<double>& new_corner_mass,
    std::vector<double>& new_velocity_r,
    std::vector<double>& new_velocity_z,
    double& remap_ms, std::string& reason) {
  TENRYU_ASSERT(components.size() == runs.size() &&
                    components.size() == splice.run_node_to_global.size(),
                "subdomain remap component/run/splice count mismatch");
  const mesh::MultiBlockTopology& topology =
      *state.mesh.topo.multiblock;
  const int corner_stride = state.corner_stride;
  const auto remap_start = std::chrono::steady_clock::now();
  for (std::size_t component_index = 0;
       component_index < components.size(); ++component_index) {
    const SubdomainComponent& component = components[component_index];
    std::vector<int> donor_cells = component.cells;
    std::set<int> ring_cells;
    for (const int cell : component.cells) {
      const int face_begin = topology.face_adj_csr_offsets[
          static_cast<std::size_t>(cell)];
      const int face_end = topology.face_adj_csr_offsets[
          static_cast<std::size_t>(cell + 1)];
      for (int face = face_begin; face < face_end; ++face) {
        const int neighbor = topology.face_adj_csr_indices[
            static_cast<std::size_t>(face)];
        if (neighbor >= 0 &&
            component.membership[static_cast<std::size_t>(neighbor)] ==
                0U) {
          ring_cells.insert(neighbor);
        }
      }
    }
    donor_cells.insert(donor_cells.end(), ring_cells.begin(),
                       ring_cells.end());
    HostMesh donor_mesh =
        extract_cell_subset(old_mesh, donor_cells, corner_stride);
    HostMesh target_mesh =
        extract_cell_subset(splice.mesh, donor_cells, corner_stride);

    std::vector<double> donor_mass(donor_cells.size());
    std::vector<double> donor_ee(donor_cells.size());
    std::vector<double> donor_ei(donor_cells.size());
    std::vector<double> donor_corner_mass(
        donor_cells.size() * static_cast<std::size_t>(corner_stride),
        0.0);
    for (std::size_t local = 0; local < donor_cells.size(); ++local) {
      const int cell = donor_cells[local];
      donor_mass[local] = old_mass[static_cast<std::size_t>(cell)];
      donor_ee[local] = old_ee[static_cast<std::size_t>(cell)];
      donor_ei[local] = old_ei[static_cast<std::size_t>(cell)];
      for (int corner = 0; corner < corner_stride; ++corner) {
        donor_corner_mass[
            local * static_cast<std::size_t>(corner_stride) +
            static_cast<std::size_t>(corner)] =
            old_corner_mass[static_cast<std::size_t>(
                cell * corner_stride + corner)];
      }
    }

    std::vector<double> target_mass(donor_cells.size(), 0.0);
    std::vector<double> target_ee(donor_cells.size(), 0.0);
    std::vector<double> target_ei(donor_cells.size(), 0.0);
    std::vector<double> target_corner_mass(
        donor_cells.size() * static_cast<std::size_t>(corner_stride),
        0.0);
    std::vector<double> target_velocity_r(
        static_cast<std::size_t>(splice.mesh.n_nodes), 0.0);
    std::vector<double> target_velocity_z(
        static_cast<std::size_t>(splice.mesh.n_nodes), 0.0);
    const hydro::reale::RemapFields donor_fields{
        donor_mass.data(), donor_ee.data(), donor_ei.data(),
        donor_corner_mass.data(), old_velocity_r.data(),
        old_velocity_z.data()};
    const hydro::reale::RemapResult remap =
        hydro::reale::remap_conservative(
            remap_view(donor_mesh, corner_stride),
            remap_view(target_mesh, corner_stride), donor_fields,
            target_mass.data(), target_ee.data(), target_ei.data(),
            target_corner_mass.data(), target_velocity_r.data(),
            target_velocity_z.data(), false,
            cfg.numerics.ale.dvclp_solver_rev,
            cfg.numerics.ale.reale_overlay_additivity_tol);
    if (!remap.ok) {
      reason = remap.reject_reason;
      return false;
    }

    for (std::size_t local = 0; local < component.cells.size(); ++local) {
      const int cell = component.cells[local];
      new_mass[static_cast<std::size_t>(cell)] = target_mass[local];
      new_ee[static_cast<std::size_t>(cell)] = target_ee[local];
      new_ei[static_cast<std::size_t>(cell)] = target_ei[local];
      for (int corner = 0; corner < corner_stride; ++corner) {
        new_corner_mass[static_cast<std::size_t>(
            cell * corner_stride + corner)] =
            target_corner_mass[
                local * static_cast<std::size_t>(corner_stride) +
                static_cast<std::size_t>(corner)];
      }
    }
    const std::vector<int>& node_map =
        splice.run_node_to_global[component_index];
    for (const int global_node : node_map) {
      if (global_node < old_mesh.n_nodes) {
        continue;
      }
      new_velocity_r[static_cast<std::size_t>(global_node)] =
          target_velocity_r[static_cast<std::size_t>(global_node)];
      new_velocity_z[static_cast<std::size_t>(global_node)] =
          target_velocity_z[static_cast<std::size_t>(global_node)];
    }
  }
  remap_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - remap_start)
          .count();
  return true;
}

bool project_subdomain_physical_carrier(
    const core::State& state, const StagedMesh& new_mesh,
    const int corner_stride, std::vector<double>& new_mass,
    std::vector<double>& new_ei, std::vector<double>& new_corner_mass,
    std::vector<double>& new_velocity_r,
    std::vector<double>& new_velocity_z, double& projection_delta_k,
    std::string& reason) {
  TENRYU_ASSERT(state.boundary_carrier.valid &&
                    state.carrier_domain_active,
                "subdomain physical projection requires active carrier");
  const std::size_t n_masters = state.boundary_carrier.masters.size();
  std::vector<int> master_node(n_masters, -1);
  std::vector<unsigned char> master_fix_r(n_masters, 0U);
  std::vector<int> master_occurrences(n_masters, 0);
  std::vector<int> slave_node;
  std::vector<int> slave_edge;
  std::vector<double> slave_lambda;
  for (int node = 0; node < new_mesh.n_nodes; ++node) {
    const std::size_t node_index = static_cast<std::size_t>(node);
    const std::uint8_t node_class = new_mesh.node_class[node_index];
    if (node_class == static_cast<std::uint8_t>(
                          mesh::voronoi::TessNodeClass::kCarrierMaster)) {
      const int domain_vertex = new_mesh.node_domain_vertex[node_index];
      TENRYU_ASSERT(
          domain_vertex >= 0 &&
              static_cast<std::size_t>(domain_vertex) < n_masters,
          "subdomain carrier projection master index out of range");
      const std::size_t master_index =
          static_cast<std::size_t>(domain_vertex);
      ++master_occurrences[master_index];
      master_node[master_index] = node;
      master_fix_r[master_index] =
          state.boundary_carrier.masters[master_index].bc_class ==
                  mesh::CarrierBcClass::kAxis
              ? 1U
              : 0U;
    } else if (node_class == static_cast<std::uint8_t>(
                                 mesh::voronoi::TessNodeClass::
                                     kCarrierSlave)) {
      slave_node.push_back(node);
      slave_edge.push_back(new_mesh.node_carrier_edge[node_index]);
      slave_lambda.push_back(new_mesh.node_lambda[node_index]);
    }
  }
  TENRYU_ASSERT(
      std::all_of(master_occurrences.begin(), master_occurrences.end(),
                  [](const int count) { return count == 1; }),
      "subdomain carrier projection master multiplicity");

  const std::size_t n_nodes = static_cast<std::size_t>(new_mesh.n_nodes);
  const std::size_t n_cells = static_cast<std::size_t>(new_mesh.n_cells);
  std::vector<double> node_mass(n_nodes, 0.0);
  std::vector<int> corner_cell;
  std::vector<int> corner_node;
  std::vector<double> projection_corner_mass;
  corner_cell.reserve(n_cells * static_cast<std::size_t>(corner_stride));
  corner_node.reserve(n_cells * static_cast<std::size_t>(corner_stride));
  projection_corner_mass.reserve(
      n_cells * static_cast<std::size_t>(corner_stride));
  for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
    const int nverts = new_mesh.nverts[static_cast<std::size_t>(cell)];
    for (int corner = 0; corner < nverts; ++corner) {
      const std::size_t flat = static_cast<std::size_t>(
          cell * corner_stride + corner);
      const int node = new_mesh.csr_indices[flat];
      const double mass = new_corner_mass[flat];
      node_mass[static_cast<std::size_t>(node)] += mass;
      corner_cell.push_back(cell);
      corner_node.push_back(node);
      projection_corner_mass.push_back(mass);
    }
  }
  std::vector<double> node_pr(n_nodes, 0.0);
  std::vector<double> node_pz(n_nodes, 0.0);
  for (std::size_t node = 0; node < n_nodes; ++node) {
    node_pr[node] = node_mass[node] * new_velocity_r[node];
    node_pz[node] = node_mass[node] * new_velocity_z[node];
  }

  const hydro::CarrierProjectionInput projection_input{
      static_cast<int>(n_masters),
      master_node.data(),
      master_fix_r.data(),
      static_cast<int>(slave_node.size()),
      slave_node.empty() ? nullptr : slave_node.data(),
      slave_edge.empty() ? nullptr : slave_edge.data(),
      slave_lambda.empty() ? nullptr : slave_lambda.data(),
      new_mesh.n_nodes,
      node_mass.data(),
      node_pr.data(),
      node_pz.data()};
  const hydro::CarrierProjectionResult projection =
      hydro::project_carrier_boundary_velocity(
          projection_input, new_velocity_r.data(), new_velocity_z.data());
  if (!projection.ok) {
    reason = projection.reject_reason;
    return false;
  }

  std::vector<double> delta_e_cell(n_cells, 0.0);
  const hydro::CarrierDefectAllocation allocation =
      hydro::allocate_projection_defect_to_cells(
          projection_input, new_velocity_r.data(), new_velocity_z.data(),
          static_cast<int>(corner_cell.size()), corner_cell.data(),
          corner_node.data(), projection_corner_mass.data(),
          new_mesh.n_cells, delta_e_cell.data());
  if (!allocation.ok) {
    reason = allocation.reject_reason;
    return false;
  }
  TENRYU_ASSERT(
      std::abs(allocation.total - projection.delta_k) <=
          1.0e-9 * std::max(projection.delta_k, 1.0),
      "subdomain carrier projection energy allocation mismatch");
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    if (delta_e_cell[cell] > 0.0) {
      TENRYU_ASSERT(new_mass[cell] > 0.0,
                    "subdomain carrier projection positive cell mass");
      new_ei[cell] += delta_e_cell[cell] / new_mass[cell];
    }
  }
  projection_delta_k = projection.delta_k;
  return true;
}

void install_node_fields(core::State& state,
                         const StagedMesh& staged,
                         const std::vector<double>& velocity_r,
                         const std::vector<double>& velocity_z) {
  state.x_r = staged.node_r;
  state.x_z = staged.node_z;
  state.x_r_initial = staged.node_r;
  state.x_z_initial = staged.node_z;
  state.x_r_reference = staged.node_r;
  state.x_z_reference = staged.node_z;
  state.x_r_shock_target = staged.node_r;
  state.x_z_shock_target = staged.node_z;
  state.v_r = velocity_r;
  state.v_z = velocity_z;
  state.node_planar_mass.reset(static_cast<std::size_t>(staged.n_nodes));
  state.node_planar_mass.fill(0.0);
  state.ring7_last_failed_v_r.reset(0);
  state.ring7_last_failed_v_z.reset(0);
  state.ring7_last_failed_velocity_valid = false;
  state.ref_xi_node.reset(0);
  state.ref_xi_cell.reset(0);
  state.ref_dir0_r.reset(0);
  state.ref_dir0_z.reset(0);
  state.ale_monitor_ref_x_r.reset(0);
  state.ale_monitor_ref_x_z.reset(0);
  state.ale_monitor_ref_corner_fractions.reset(0);
  state.ale_monitor_ref_valid = false;
  state.ale_monitor_ref_post_morph = false;
  ++state.reference_epoch;
  state.trace_mesh_outer_node = -1;
  state.trace_mesh_outer_cell = -1;
}

// One tenryu process is one run, so these slots cannot leak across runs.
std::optional<mesh::tess::DelaunayTriangulation> g_reale_warm_dt;
int g_last_rezone_commit_step = -1;
double g_post_rezone_dt_ref = -1.0;
bool g_post_rezone_dt_pending = false;

bool commit_subdomain_rezone(
    core::State& state, const core::Config& cfg, const HostMesh& old_mesh,
    const std::vector<SubdomainComponent>& components,
    const std::vector<SubdomainTessellationRun>& runs,
    const char* const trigger_reason) {
  std::string reason;
  SubdomainSplice splice;
  const auto splice_start = std::chrono::steady_clock::now();
  if (!splice_subdomain_mesh(state, old_mesh, components, runs, splice,
                             reason)) {
    return skip(state, "topology", reason);
  }
  StagedMesh& new_mesh = splice.mesh;
  const double splice_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - splice_start)
          .count();

  for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
    const int offset = new_mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = new_mesh.nverts[static_cast<std::size_t>(cell)];
    double cell_r[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    double cell_z[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = new_mesh.csr_indices[
          static_cast<std::size_t>(offset + corner)];
      cell_r[corner] = new_mesh.node_r[static_cast<std::size_t>(node)];
      cell_z[corner] = new_mesh.node_z[static_cast<std::size_t>(node)];
    }
    if (!hydro::rz::rz_polygon_volume_certified_positive(
            cell_r, cell_z, nverts) ||
        !hydro::rz::rz_polygon_area2_certified_positive(
            cell_r, cell_z, nverts)) {
      return skip(state, "staged_geometry",
                  "staged_uncertified_cell_volume");
    }
  }

  const std::size_t n_cells = static_cast<std::size_t>(old_mesh.n_cells);
  const std::size_t old_nodes = static_cast<std::size_t>(old_mesh.n_nodes);
  const std::size_t new_nodes = static_cast<std::size_t>(new_mesh.n_nodes);
  if (!state.corner_mass_initialized || state.mass.size() != n_cells ||
      state.ee.size() != n_cells || state.ei.size() != n_cells ||
      state.corner_mass.size() != n_cells * state.corner_stride ||
      state.v_r.size() != old_nodes || state.v_z.size() != old_nodes) {
    return skip(state, "remap", "incomplete_hydro_fields");
  }

  std::vector<double> old_mass;
  std::vector<double> old_ee;
  std::vector<double> old_ei;
  std::vector<double> old_corner_mass;
  std::vector<double> old_velocity_r;
  std::vector<double> old_velocity_z;
  state.mass.copy_to_host(old_mass);
  state.ee.copy_to_host(old_ee);
  state.ei.copy_to_host(old_ei);
  state.corner_mass.copy_to_host(old_corner_mass);
  state.v_r.copy_to_host(old_velocity_r);
  state.v_z.copy_to_host(old_velocity_z);
  reale_vdiag_scan("pre_remap_old", state.step, old_velocity_r,
                   old_velocity_z);

  std::vector<double> new_mass = old_mass;
  std::vector<double> new_ee = old_ee;
  std::vector<double> new_ei = old_ei;
  std::vector<double> new_corner_mass = old_corner_mass;
  std::vector<double> new_velocity_r(new_nodes, 0.0);
  std::vector<double> new_velocity_z(new_nodes, 0.0);
  std::copy(old_velocity_r.begin(), old_velocity_r.end(),
            new_velocity_r.begin());
  std::copy(old_velocity_z.begin(), old_velocity_z.end(),
            new_velocity_z.begin());

  double remap_ms = 0.0;
  if (!remap_subdomain_components(
          state, cfg, old_mesh, components, runs, splice, old_mass,
          old_ee, old_ei, old_corner_mass, old_velocity_r, old_velocity_z,
          new_mass, new_ee, new_ei, new_corner_mass, new_velocity_r,
          new_velocity_z, remap_ms, reason)) {
    return skip(state, "remap", reason);
  }

  std::vector<unsigned char> reset_corner_mass = splice.hanging_cell;
  if (cfg.numerics.ale.reale_corner_mass_reset) {
    for (const SubdomainComponent& component : components) {
      for (const int cell : component.cells) {
        reset_corner_mass[static_cast<std::size_t>(cell)] = 1U;
      }
    }
  }
  reinitialize_corner_mass_after_remap(
      new_mesh, state.corner_stride, new_mass, new_corner_mass,
      &reset_corner_mass);
  reale_vdiag_scan("post_remap", state.step, new_velocity_r,
                   new_velocity_z);
  if (cfg.numerics.ale.reale_velocity_max_principle) {
    limit_remapped_velocities(old_mesh, new_mesh, old_velocity_r,
                              old_velocity_z, new_velocity_r,
                              new_velocity_z);
    std::copy(old_velocity_r.begin(), old_velocity_r.end(),
              new_velocity_r.begin());
    std::copy(old_velocity_z.begin(), old_velocity_z.end(),
              new_velocity_z.begin());
  }

  double projection_delta_k = 0.0;
  if (splice.has_new_physical_slaves &&
      !project_subdomain_physical_carrier(
          state, new_mesh, state.corner_stride, new_mass, new_ei,
          new_corner_mass, new_velocity_r, new_velocity_z,
          projection_delta_k, reason)) {
    return skip(state, "remap", reason);
  }
  reale_vdiag_scan("post_projection", state.step, new_velocity_r,
                   new_velocity_z);

  const double rezone_min_dt = cfg.numerics.ale.rezone_min_dt_s;
  if (rezone_min_dt > 0.0) {
    const StagedDtResult staged_dt = evaluate_staged_hydro_dt(
        new_mesh.csr_offsets.data(), new_mesh.csr_indices.data(),
        new_mesh.nverts.data(), new_mesh.n_cells, new_mesh.n_nodes,
        new_mesh.node_r.data(), new_mesh.node_z.data(),
        new_velocity_r.data(), new_velocity_z.data(), cfg);
    if (staged_dt.min_dt <= rezone_min_dt) {
      char buffer[160];
      std::snprintf(buffer, sizeof(buffer),
                    "staged_dt_%.3e_below_%.1e_cell_%d",
                    staged_dt.min_dt, rezone_min_dt,
                    staged_dt.binding_cell);
      return skip(state, "staged_dt", buffer);
    }
  }

  mesh::RingPageSet new_ring_pages;
  if (state.boundary_carrier.valid) {
    new_ring_pages = mesh::build_ring_pages(
        new_mesh.n_cells, new_mesh.csr_offsets.data(),
        new_mesh.csr_indices.data(), new_mesh.nverts.data(),
        state.corner_stride);
    if (!mesh::validate_ring_pages(new_ring_pages, new_mesh.n_cells)) {
      return skip(state, "topology", "ring_page_invariant");
    }
  }

  const auto install_start = std::chrono::steady_clock::now();
  mesh::Mesh& mesh = state.mesh;
  mesh::MultiBlockTopology& topology = *mesh.topo.multiblock;
  mesh.topo.n_nodes = new_mesh.n_nodes;
  mesh.topo.node_flags = new_mesh.node_flags;
  mesh.cell_nverts.assign(new_mesh.nverts.begin(), new_mesh.nverts.end());
  mesh.geometry_exempt_cells.clear();
  mesh.button_center.reset();
  topology.cell_node_csr_offsets = std::move(new_mesh.csr_offsets);
  topology.cell_node_csr_indices = std::move(new_mesh.csr_indices);
  topology.cell_orientation_sign.assign(n_cells, 1);
  state.mesh.topo.general_polygonal = true;
  state.cell_nverts_host.assign(mesh.cell_nverts.begin(),
                                mesh.cell_nverts.end());
  install_node_fields(state, new_mesh, new_velocity_r, new_velocity_z);
  mesh.node_r = state.x_r.data();
  mesh.node_z = state.x_z.data();

  mesh::rebuild_runtime_topology_caches(state, cfg);
  topology.has_polar_tier = false;
  topology.polar_tier_columns.clear();
  topology.polar_tier_transition_radii.clear();
  topology.dendrite_block_rings.clear();
  topology.south_node_of.clear();
  mesh.multiblock_outer_svec_tangent_balance = false;
  mesh.recompute_geometry();

  if (state.boundary_carrier.valid) {
    state.carrier_ring_pages = std::move(new_ring_pages);
    mesh.ring_pages = &state.carrier_ring_pages;
  } else {
    state.carrier_ring_pages.valid = false;
  }
  state.carrier_node_class = new_mesh.node_class;
  state.carrier_node_edge = new_mesh.node_carrier_edge;
  state.carrier_node_lambda = new_mesh.node_lambda;

  state.mass = new_mass;
  state.ee = new_ee;
  state.ei = new_ei;
  state.corner_mass = new_corner_mass;
  state.corner_mass_initialized = true;
  state.corner_mass_is_lagrangian_invariant =
      hydro::corner_mass_lagrangian_invariant_enabled(cfg);
  state.vol = mesh.cell_vol;
  state.cell_vol_initial = mesh.cell_vol;
  state.vol_prev_hydro = mesh.cell_vol;
  std::vector<double> density(n_cells, 0.0);
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    if (!(mesh.cell_vol[cell] > 0.0) ||
        !std::isfinite(mesh.cell_vol[cell])) {
      core::tenryu_abort("positive ReALE committed volume",
                         "ReALE topology passed staging but committed a "
                         "non-positive cell volume",
                         __FILE__, __LINE__);
    }
    density[cell] = new_mass[cell] / mesh.cell_vol[cell];
  }
  state.rho = density;
  state.dt_prev_hydro = -1.0;
  state.invalidate_cell_material_props();
  state.ale_rezoned = true;
  ++state.ale_rezone_invocations;
  ++state.ale_remaps_applied;
  state.ale_last_applied_step = state.step;
  const double install_ms = splice_ms +
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - install_start)
          .count();

  int axis_masters = 0;
  int axis_intervals = 0;
  if (state.boundary_carrier.valid) {
    for (const mesh::CarrierVertex& master :
         state.boundary_carrier.masters) {
      if (master.bc_class == mesh::CarrierBcClass::kAxis) {
        ++axis_masters;
        ++axis_intervals;
      }
    }
  }
  int folded_cells = 0;
  int folded_cells_dumped = 0;
  for (int cell = 0; cell < mesh.topo.n_cells; ++cell) {
    const int begin =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = mesh.cell_nverts[static_cast<std::size_t>(cell)];
    double ring_twice_area = 0.0;
    double min_edge = std::numeric_limits<double>::infinity();
    for (int vertex = 0; vertex < nverts; ++vertex) {
      const int next = vertex + 1 == nverts ? 0 : vertex + 1;
      const int node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + vertex)];
      const int next_node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + next)];
      const double vertex_r =
          new_mesh.node_r[static_cast<std::size_t>(node)];
      const double vertex_z =
          new_mesh.node_z[static_cast<std::size_t>(node)];
      const double next_r =
          new_mesh.node_r[static_cast<std::size_t>(next_node)];
      const double next_z =
          new_mesh.node_z[static_cast<std::size_t>(next_node)];
      ring_twice_area += vertex_r * next_z - next_r * vertex_z;
      min_edge = std::min(
          min_edge, std::hypot(next_r - vertex_r, next_z - vertex_z));
    }
    const double ring_orientation = ring_twice_area > 0.0 ? 1.0 : -1.0;
    int folds_in_cell = 0;
    for (int vertex = 0; vertex < nverts; ++vertex) {
      const int previous = vertex == 0 ? nverts - 1 : vertex - 1;
      const int next = vertex + 1 == nverts ? 0 : vertex + 1;
      const int previous_node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + previous)];
      const int node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + vertex)];
      const int next_node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + next)];
      const double previous_r =
          new_mesh.node_r[static_cast<std::size_t>(previous_node)];
      const double previous_z =
          new_mesh.node_z[static_cast<std::size_t>(previous_node)];
      const double vertex_r =
          new_mesh.node_r[static_cast<std::size_t>(node)];
      const double vertex_z =
          new_mesh.node_z[static_cast<std::size_t>(node)];
      const double next_r =
          new_mesh.node_r[static_cast<std::size_t>(next_node)];
      const double next_z =
          new_mesh.node_z[static_cast<std::size_t>(next_node)];
      const double fold_cross =
          (vertex_r - previous_r) * (next_z - vertex_z) -
          (vertex_z - previous_z) * (next_r - vertex_r);
      if (ring_orientation * fold_cross < 0.0) {
        ++folds_in_cell;
      }
    }
    if (folds_in_cell > 0) {
      ++folded_cells;
      if (folded_cells_dumped < 3) {
        std::fprintf(stderr,
                     "[mesh_fold_census] cell=%d nverts=%d min_edge=%.3e "
                     "folds_in_cell=%d\n",
                     cell, nverts, min_edge, folds_in_cell);
        ++folded_cells_dumped;
      }
    }
  }

  double target_generation_ms = 0.0;
  double tessellation_ms = 0.0;
  double legacy_assembly_ms = 0.0;
  int weld_contracted = 0;
  int weld_clusters = 0;
  double weld_max_displacement = 0.0;
  int lloyd_iterations = 0;
  for (const SubdomainTessellationRun& run : runs) {
    target_generation_ms += run.targets.target_generation_ms;
    tessellation_ms += run.targets.tessellation_ms;
    legacy_assembly_ms += run.targets.legacy_assembly_ms;
    const mesh::voronoi::Tessellation& tessellation =
        run.targets.accepted_tessellation;
    weld_contracted += tessellation.weld_contracted;
    weld_clusters += tessellation.weld_clusters;
    weld_max_displacement = std::max(
        weld_max_displacement, tessellation.weld_max_displacement);
    lloyd_iterations += run.targets.cvt_iterations_used;
  }
  const double targets_ms = target_generation_ms + tessellation_ms +
                            legacy_assembly_ms;
  char stage_timings[384];
  std::snprintf(
      stage_timings, sizeof(stage_timings),
      " targets_ms=%.0f voronoi_ms=0 remap_ms=%.0f install_ms=%.0f "
      "carrier=%d proj_dk=%.3e pages=%zu axis_masters=%d "
      "axis_intervals=%d weld=%d wclust=%d wdisp=%.1e tg=%.0f "
      "tess=%.0f legacy=%.0f other=0 lloyd=%d",
      targets_ms, remap_ms, install_ms,
      state.carrier_domain_active ? 1 : 0, projection_delta_k,
      state.carrier_ring_pages.pages.size(), axis_masters, axis_intervals,
      weld_contracted, weld_clusters, weld_max_displacement,
      target_generation_ms, tessellation_ms, legacy_assembly_ms,
      lloyd_iterations);
  reale_trigger::snapshot_reference(state);
  g_last_rezone_commit_step = state.step;
  g_post_rezone_dt_pending = true;
  g_post_rezone_dt_ref = -1.0;
  const int component_count = static_cast<int>(components.size());
  core::log_info(
      "[reale_v2] COMMIT step=" + std::to_string(state.step) +
      " cells=" + std::to_string(old_mesh.n_cells) +
      " nodes_old=" + std::to_string(old_mesh.n_nodes) +
      " nodes_new=" + std::to_string(new_mesh.n_nodes) +
      " remaps_total=" + std::to_string(state.ale_remaps_applied) +
      " skipped_total=" + std::to_string(state.reale_rezone_skipped) +
      " subdomain=" + std::to_string(component_count) + "/" +
      std::to_string(component_count) +
      " cells=" + std::to_string(splice.rezoned_cells) + stage_timings +
      " folds=" + std::to_string(folded_cells) +
      " trigger=" + std::string(trigger_reason));
  return true;
}

}  // namespace

void validate_runtime_scope(const core::Config& cfg, const int n_ranks) {
  if (cfg.numerics.ale.mesh_mode == "reale_v2" && n_ranks != 1) {
    throw core::namelist::ConfigError(
        "Numerics.ale.mesh_mode=\"reale_v2\" pilot requires a single MPI "
        "rank");
  }
}

bool rezone_step(core::State& state,
                 const core::Config& cfg,
                 const double dt,
                 const double dt_hydro_bound) {
  if (cfg.numerics.ale.mesh_mode != "reale_v2") {
    return false;
  }
  static const int rezone_interval = [] {
    const char* const value = std::getenv("TENRYU_REALE_REZONE_INTERVAL");
    if (value == nullptr || *value == '\0') {
      return 1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    TENRYU_ASSERT(end != nullptr && *end == '\0' && parsed >= 1,
                  "TENRYU_REALE_REZONE_INTERVAL must be a positive integer");
    if (parsed != 1) {
      core::log_info("[reale_v2] rezone trigger ceiling=" +
                     std::to_string(parsed) +
                     " (TENRYU_REALE_REZONE_INTERVAL)");
    }
    return static_cast<int>(parsed);
  }();
  if (g_post_rezone_dt_pending && dt_hydro_bound > 0.0 &&
      std::isfinite(dt_hydro_bound)) {
    g_post_rezone_dt_ref = dt_hydro_bound;
    g_post_rezone_dt_pending = false;
  }
  const char* trigger_reason = nullptr;
  double trigger_disp_ratio = -1.0;
  if (!reale_trigger::reference_valid(state)) {
    trigger_reason = "first";
  } else {
    const int steps_since = state.step - g_last_rezone_commit_step;
    if (steps_since >= rezone_interval) {
      trigger_reason = "ceiling";
    } else if (g_post_rezone_dt_ref > 0.0 && dt_hydro_bound > 0.0 &&
               std::isfinite(dt_hydro_bound) &&
               dt_hydro_bound <=
                   cfg.numerics.ale.reale_dt_trigger_factor *
                       g_post_rezone_dt_ref &&
               steps_since >= cfg.numerics.ale.reale_dt_trigger_cooldown) {
      trigger_reason = "dt";
    } else {
      trigger_disp_ratio = reale_trigger::max_displacement_ratio(state);
      if (trigger_disp_ratio >= 1.0) {
        trigger_reason = "disp";
      }
    }
  }
  if (trigger_reason == nullptr) {
    return false;
  }
  reale_vdiag_rho_scan("pre_rezone_rho", state.step, state);
  {
    char trigger_log[192];
    std::snprintf(trigger_log, sizeof(trigger_log),
                  "[reale_trigger] step=%d reason=%s steps_since=%d "
                  "dt_hydro=%.3e dt_ref=%.3e disp_ratio=%.3f",
                  state.step, trigger_reason,
                  g_last_rezone_commit_step >= 0
                      ? state.step - g_last_rezone_commit_step
                      : -1,
                  dt_hydro_bound, g_post_rezone_dt_ref, trigger_disp_ratio);
    core::log_info(trigger_log);
  }
  if (!(dt > 0.0) || !std::isfinite(dt)) {
    return skip(state, "capture", "invalid_dt");
  }

  HostMesh old_mesh;
  std::string reason;
  if (!capture_current_mesh(state, old_mesh, reason)) {
    return skip(state, "capture", reason);
  }
  std::optional<DisturbedClassification> disturbed_classification;
  if (cfg.numerics.ale.reale_subdomain_rezone ||
      reale_p3_diag_enabled()) {
    disturbed_classification = classify_disturbed_cells(state, old_mesh);
  }
  const bool use_subdomain_rezone =
      cfg.numerics.ale.reale_subdomain_rezone &&
      disturbed_classification->has_face_adjacency &&
      static_cast<double>(std::count(
          disturbed_classification->dilated.begin(),
          disturbed_classification->dilated.end(), 1U)) /
              static_cast<double>(old_mesh.n_cells) <=
          cfg.numerics.ale.reale_subdomain_frac_max;
  std::vector<SubdomainComponent> subdomain_components;
  if (use_subdomain_rezone) {
    subdomain_components = build_subdomain_components(
        state, old_mesh, *disturbed_classification);
  }
  std::vector<SubdomainTessellationRun> subdomain_runs;
  if (use_subdomain_rezone) {
    mesh::tess::set_tess_gpu_dual_enabled(
        cfg.numerics.ale.tess_gpu_dual);
    mesh::tess::set_tess_gpu_restrict_enabled(
        cfg.numerics.ale.tess_gpu_restrict);
    mesh::voronoi::set_reale_lloyd_max(
        cfg.numerics.ale.reale_lloyd_max);
    mesh::tess::set_reale_short_edge_collapse_rel(0.0);
    subdomain_runs.resize(subdomain_components.size());
    for (std::size_t component = 0;
         component < subdomain_components.size(); ++component) {
      std::string subdomain_stage;
      if (!build_subdomain_tessellation(
              state, cfg, old_mesh, subdomain_components[component], dt,
              subdomain_runs[component], subdomain_stage, reason)) {
        return skip(state, subdomain_stage.c_str(), reason);
      }
    }
  }
  if (use_subdomain_rezone) {
    return commit_subdomain_rezone(
        state, cfg, old_mesh, subdomain_components, subdomain_runs,
        trigger_reason);
  }

  mesh::voronoi::DomainBoundary domain;
  std::vector<int> surviving_nodes;
  if (state.boundary_carrier.valid && state.carrier_domain_active) {
    const auto& masters = state.boundary_carrier.masters;
    domain.r.reserve(masters.size());
    domain.z.reserve(masters.size());
    for (const mesh::CarrierVertex& master : masters) {
      TENRYU_ASSERT(master.mesh_node >= 0 &&
                        master.mesh_node < old_mesh.n_nodes,
                    "carrier master node out of range");
      domain.r.push_back(
          old_mesh.node_r[static_cast<std::size_t>(master.mesh_node)]);
      domain.z.push_back(
          old_mesh.node_z[static_cast<std::size_t>(master.mesh_node)]);
    }
  } else {
    if (!extract_domain_boundary(
            state, old_mesh, domain, reason, &surviving_nodes)) {
      return skip(state, "domain", reason);
    }
  }
  if (!state.boundary_carrier.valid) {
    state.boundary_carrier = mesh::carrier_from_boundary_loop(
        surviving_nodes, old_mesh.node_r, old_mesh.node_z);
    char capture_log[64];
    std::snprintf(capture_log, sizeof(capture_log),
                  "[carrier] captured masters=%zu",
                  state.boundary_carrier.masters.size());
    core::log_info(capture_log);
  }

  std::vector<mesh::voronoi::Generator> generators;
  std::vector<mesh::voronoi::GeneratorFlowSample> flow;
  if (!build_generators_and_flow(
          state, old_mesh, generators, flow, reason)) {
    return skip(state, "generators", reason);
  }
  if (g_reale_warm_dt.has_value() &&
      g_reale_warm_dt->sites.size() != generators.size()) {
    g_reale_warm_dt.reset();
  }
  static bool generators_dumped = false;
  if (const char* const path =
          std::getenv("TENRYU_REALE_DUMP_GENERATORS")) {
    if (!generators_dumped) {
      std::FILE* const output = std::fopen(path, "w");
      TENRYU_ASSERT(output != nullptr, "open ReALE generator dump");
      for (const mesh::voronoi::Generator& generator : generators) {
        std::fprintf(output, "%llu %.17g %.17g\n",
                     static_cast<unsigned long long>(generator.id),
                     generator.r, generator.z);
      }
      TENRYU_ASSERT(std::fclose(output) == 0,
                    "close ReALE generator dump");

      const std::string domain_path = std::string(path) + ".domain";
      std::FILE* const domain_output =
          std::fopen(domain_path.c_str(), "w");
      TENRYU_ASSERT(domain_output != nullptr, "open ReALE domain dump");
      for (std::size_t vertex = 0; vertex < domain.r.size(); ++vertex) {
        std::fprintf(domain_output, "%.17g %.17g\n",
                     domain.r[vertex], domain.z[vertex]);
      }
      TENRYU_ASSERT(std::fclose(domain_output) == 0,
                    "close ReALE domain dump");
      generators_dumped = true;
    }
    return skip(state, "generators", "generator_dump_only");
  }
  mesh::tess::set_tess_gpu_dual_enabled(cfg.numerics.ale.tess_gpu_dual);
  mesh::tess::set_tess_gpu_restrict_enabled(cfg.numerics.ale.tess_gpu_restrict);
  mesh::voronoi::set_reale_lloyd_max(cfg.numerics.ale.reale_lloyd_max);
  // The weld-stage band remains opt-in instrumentation only (dossier §6.6); the namelist drives the install-boundary pass.
  mesh::tess::set_reale_short_edge_collapse_rel(0.0);
  const auto targets_start = std::chrono::steady_clock::now();
  const mesh::voronoi::TargetResult targets =
      mesh::voronoi::build_generator_targets(
          generators, flow, domain, dt,
          cfg.numerics.ale.reale_core == "exact", &g_reale_warm_dt);
  const double targets_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - targets_start)
          .count();
  const double targets_other_ms =
      targets_ms - targets.target_generation_ms -
      targets.tessellation_ms - targets.legacy_assembly_ms;
  if (!targets.valid) {
    return skip(state, "targets", targets.reject_reason);
  }
  const auto voronoi_start = std::chrono::steady_clock::now();
  // The exact core is deterministic, so the retained accepted-state build is
  // bitwise identical to rebuilding from targets.targets here.
  const mesh::voronoi::Tessellation& tessellation =
      targets.accepted_tessellation;
  const double voronoi_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - voronoi_start)
          .count();
  if (!tessellation.valid) {
    return skip(state, "voronoi", tessellation.reject_reason);
  }

  static const bool shadow_core_enabled = [] {
    const char* const value = std::getenv("TENRYU_REALE_SHADOW_CORE");
    return value != nullptr && std::string(value) == "1";
  }();
  if (shadow_core_enabled) {
    std::vector<mesh::tess::Site> sites;
    sites.reserve(targets.targets.size());
    for (const mesh::voronoi::Generator& generator : targets.targets) {
      sites.push_back(
          mesh::tess::Site{generator.r, generator.z, generator.id, 0});
    }
    std::vector<mesh::tess::DomainPoint> domain_points;
    domain_points.reserve(domain.r.size());
    for (std::size_t vertex = 0; vertex < domain.r.size(); ++vertex) {
      domain_points.push_back(
          mesh::tess::DomainPoint{domain.r[vertex], domain.z[vertex]});
    }

    const auto shadow_start = std::chrono::steady_clock::now();
    const mesh::tess::RestrictedVoronoiResult shadow_result =
        mesh::tess::build_restricted_voronoi(
            std::move(sites), std::move(domain_points));
    const double shadow_wall_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - shadow_start)
            .count();
    if (!shadow_result.ok) {
      std::fprintf(stderr,
                   "[shadow_core] BUILD_FAIL code=%s wall_ms=%.1f\n",
                   shadow_result.failure_code.c_str(), shadow_wall_ms);
    } else {
      const mesh::voronoi::Tessellation converted =
          mesh::tess::to_legacy_tessellation(
              shadow_result.dt, shadow_result.dcel);
      const mesh::tess::TessellationComparison comparison =
          mesh::tess::compare_tessellations(tessellation, converted);
      std::fprintf(
          stderr,
          "[shadow_core] topo_eq=%d first_div=\"%s\" coord_dmax=%.3e "
          "vol_dmax=%.3e nv_max=%d nv_gt8=%zu nodes_old=%zu "
          "nodes_new=%zu wall_ms=%.1f\n",
          comparison.topology_equal ? 1 : 0,
          comparison.first_divergence.c_str(), comparison.max_coord_delta,
          comparison.max_volume_delta, comparison.max_valence,
          comparison.nv_over_8, comparison.nodes_old, comparison.nodes_new,
          shadow_wall_ms);
    }
  }

  const mesh::MultiBlockTopology& old_topology =
      *state.mesh.topo.multiblock;
  StagedMesh new_mesh;
  std::vector<double> new_generator_r;
  std::vector<double> new_generator_z;
  std::vector<int> new_neighbor_index;
  const auto staged_mesh_start = std::chrono::steady_clock::now();
  if (!build_staged_mesh(
          tessellation, targets.targets, old_topology.cell_id_stable,
          state.corner_stride, new_mesh, new_generator_r, new_generator_z,
          new_neighbor_index, reason)) {
    return skip(state, "topology", reason);
  }
  collapse_short_edges(
      new_mesh, cfg.numerics.ale.reale_short_edge_collapse_rel);
  if (state.carrier_domain_active) {
    std::vector<int> master_occurrences(
        state.boundary_carrier.masters.size(), 0);
    for (int node = 0; node < new_mesh.n_nodes; ++node) {
      const std::size_t node_index = static_cast<std::size_t>(node);
      if (new_mesh.node_class[node_index] != static_cast<std::uint8_t>(
              mesh::voronoi::TessNodeClass::kCarrierMaster)) {
        continue;
      }
      const int domain_vertex = new_mesh.node_domain_vertex[node_index];
      if (domain_vertex >= 0 &&
          static_cast<std::size_t>(domain_vertex) <
              master_occurrences.size()) {
        ++master_occurrences[static_cast<std::size_t>(domain_vertex)];
      }
    }
    if (std::any_of(master_occurrences.begin(), master_occurrences.end(),
                    [](const int count) { return count != 1; })) {
      return skip(state, "topology", "carrier_master_missing");
    }
    if (!tessellation.carrier_representability_ok) {
      return skip(state, "topology", "carrier_representability");
    }
    if (!tessellation.carrier_coverage_ok) {
      return skip(state, "topology", "carrier_coverage");
    }
    if (new_mesh.provenance_conflicts > 0) {
      return skip(state, "topology", "carrier_provenance_conflict");
    }
  }
  const char* dump_cell_env = std::getenv("TENRYU_REALE_DUMP_CELL");
  const int dump_cell =
      dump_cell_env != nullptr ? std::atoi(dump_cell_env) : -1;
  for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
    const int off = new_mesh.csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = new_mesh.nverts[static_cast<std::size_t>(cell)];
    double cell_r[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    double cell_z[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    for (int k = 0; k < nverts; ++k) {
      const int node =
          new_mesh.csr_indices[static_cast<std::size_t>(off + k)];
      cell_r[k] = new_mesh.node_r[static_cast<std::size_t>(node)];
      cell_z[k] = new_mesh.node_z[static_cast<std::size_t>(node)];
    }
    if (cell == dump_cell) {
      std::fprintf(stderr,
                   "[staged_gate] cell=%d nverts=%d vol=%.17e area2=%.17e\n",
                   cell, nverts,
                   hydro::rz::rz_polygon_volume_exact(cell_r, cell_z, nverts),
                   hydro::rz::rz_polygon_area2_exact(cell_r, cell_z, nverts));
      for (int k = 0; k < nverts; ++k) {
        const int node =
            new_mesh.csr_indices[static_cast<std::size_t>(off + k)];
        std::fprintf(
            stderr,
            "[staged_gate] cell=%d k=%d node=%d r=%.17e z=%.17e\n",
            cell, k, node, cell_r[k], cell_z[k]);
      }
    }
    if (!hydro::rz::rz_polygon_volume_certified_positive(
            cell_r, cell_z, nverts) ||
        !hydro::rz::rz_polygon_area2_certified_positive(
            cell_r, cell_z, nverts)) {
      return skip(state, "staged_geometry",
                  "staged_uncertified_cell_volume");
    }
  }
  if (!tessellation.node_boundary.empty()) {
    mesh::RingPageSet ring_pages = mesh::build_ring_pages(
        new_mesh.n_cells, new_mesh.csr_offsets.data(),
        new_mesh.csr_indices.data(), new_mesh.nverts.data(),
        state.corner_stride);
    if (!mesh::validate_ring_pages(ring_pages, new_mesh.n_cells)) {
      return skip(state, "topology", "ring_page_invariant");
    }
    state.carrier_ring_pages = std::move(ring_pages);
    mesh::Mesh& mesh = state.mesh;
    mesh.ring_pages = &state.carrier_ring_pages;
    // This State member outlives every Mesh use within and across steps, and valid makes a stale pointer inert.
  }
  double install_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - staged_mesh_start)
          .count();

  const std::size_t n_cells = static_cast<std::size_t>(old_mesh.n_cells);
  const std::size_t old_nodes = static_cast<std::size_t>(old_mesh.n_nodes);
  const std::size_t new_nodes = static_cast<std::size_t>(new_mesh.n_nodes);
  if (!state.corner_mass_initialized || state.mass.size() != n_cells ||
      state.ee.size() != n_cells || state.ei.size() != n_cells ||
      state.corner_mass.size() != n_cells * state.corner_stride ||
      state.v_r.size() != old_nodes || state.v_z.size() != old_nodes) {
    state.carrier_ring_pages.valid = false;  // pages describe the discarded topology
    return skip(state, "remap", "incomplete_hydro_fields");
  }

  const auto remap_prep_start = std::chrono::steady_clock::now();
  std::vector<double> old_mass;
  std::vector<double> old_ee;
  std::vector<double> old_ei;
  std::vector<double> old_corner_mass;
  std::vector<double> old_velocity_r;
  std::vector<double> old_velocity_z;
  state.mass.copy_to_host(old_mass);
  state.ee.copy_to_host(old_ee);
  state.ei.copy_to_host(old_ei);
  state.corner_mass.copy_to_host(old_corner_mass);
  state.v_r.copy_to_host(old_velocity_r);
  state.v_z.copy_to_host(old_velocity_z);
  reale_vdiag_scan("pre_remap_old", state.step, old_velocity_r,
                   old_velocity_z);
  const auto remap_d2h_end = std::chrono::steady_clock::now();

  std::vector<double> new_mass(n_cells, 0.0);
  std::vector<double> new_ee(n_cells, 0.0);
  std::vector<double> new_ei(n_cells, 0.0);
  std::vector<double> new_corner_mass(n_cells * state.corner_stride, 0.0);
  std::vector<double> new_velocity_r(new_nodes, 0.0);
  std::vector<double> new_velocity_z(new_nodes, 0.0);
  const auto remap_alloc_end = std::chrono::steady_clock::now();
  const hydro::reale::RemapFields old_fields{
      old_mass.data(), old_ee.data(), old_ei.data(), old_corner_mass.data(),
      old_velocity_r.data(), old_velocity_z.data()};
  if (std::getenv("TENRYU_TESS_TIMING") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_TIMING")) == std::string("1")) {
    std::fprintf(stderr, "[remap_host] d2h_ms=%.1f alloc_ms=%.1f\n",
                 std::chrono::duration<double, std::milli>(
                     remap_d2h_end - remap_prep_start)
                     .count(),
                 std::chrono::duration<double, std::milli>(
                     remap_alloc_end - remap_d2h_end)
                     .count());
  }
  const auto remap_start = std::chrono::steady_clock::now();
  const hydro::reale::RemapResult remap = hydro::reale::remap_conservative(
      remap_view(old_mesh, state.corner_stride),
      remap_view(new_mesh, state.corner_stride, new_generator_r.data(),
                 new_generator_z.data(), new_neighbor_index.data()),
      old_fields,
      new_mass.data(), new_ee.data(), new_ei.data(), new_corner_mass.data(),
      new_velocity_r.data(), new_velocity_z.data(),
      tessellation.weld_contracted == 0,
      cfg.numerics.ale.dvclp_solver_rev,
      cfg.numerics.ale.reale_overlay_additivity_tol);
  const double remap_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - remap_start)
          .count();
  if (!remap.ok) {
    state.carrier_ring_pages.valid = false;  // pages describe the discarded topology
    return skip(state, "remap", remap.reject_reason);
  }
  if (cfg.numerics.ale.reale_corner_mass_reset) {
    reinitialize_corner_mass_after_remap(
        new_mesh, state.corner_stride, new_mass, new_corner_mass);
  }
  reale_vdiag_scan("post_remap", state.step, new_velocity_r, new_velocity_z);
  if (cfg.numerics.ale.reale_velocity_max_principle) {
    limit_remapped_velocities(old_mesh, new_mesh, old_velocity_r,
                              old_velocity_z, new_velocity_r,
                              new_velocity_z);
  }

  double projection_delta_k = 0.0;
  if (state.carrier_domain_active &&
      !tessellation.node_boundary.empty()) {
    const std::size_t n_masters = state.boundary_carrier.masters.size();
    std::vector<int> master_node(n_masters, -1);
    std::vector<unsigned char> master_fix_r(n_masters, 0);
    std::vector<int> master_occurrences(n_masters, 0);
    std::vector<int> slave_node;
    std::vector<int> slave_edge;
    std::vector<double> slave_lambda;
    for (int node = 0; node < new_mesh.n_nodes; ++node) {
      const std::size_t node_index = static_cast<std::size_t>(node);
      const std::uint8_t node_class = new_mesh.node_class[node_index];
      if (node_class == static_cast<std::uint8_t>(
                            mesh::voronoi::TessNodeClass::kCarrierMaster)) {
        const int domain_vertex =
            new_mesh.node_domain_vertex[node_index];
        TENRYU_ASSERT(
            domain_vertex >= 0 &&
                static_cast<std::size_t>(domain_vertex) < n_masters,
            "carrier projection master index out of range");
        const std::size_t master_index =
            static_cast<std::size_t>(domain_vertex);
        ++master_occurrences[master_index];
        master_node[master_index] = node;
        master_fix_r[master_index] =
            state.boundary_carrier.masters[master_index].bc_class ==
                    mesh::CarrierBcClass::kAxis
                ? 1
                : 0;
      } else if (node_class == static_cast<std::uint8_t>(
                                   mesh::voronoi::TessNodeClass::
                                       kCarrierSlave)) {
        slave_node.push_back(node);
        slave_edge.push_back(new_mesh.node_carrier_edge[node_index]);
        slave_lambda.push_back(new_mesh.node_lambda[node_index]);
      }
    }
    TENRYU_ASSERT(
        std::all_of(master_occurrences.begin(), master_occurrences.end(),
                    [](const int count) { return count == 1; }),
        "carrier projection master multiplicity");

    std::vector<double> node_mass(new_nodes, 0.0);
    std::vector<int> corner_cell;
    std::vector<int> corner_node;
    std::vector<double> corner_mass;
    corner_cell.reserve(n_cells * state.corner_stride);
    corner_node.reserve(n_cells * state.corner_stride);
    corner_mass.reserve(n_cells * state.corner_stride);
    for (int cell = 0; cell < new_mesh.n_cells; ++cell) {
      const int nverts = new_mesh.nverts[static_cast<std::size_t>(cell)];
      for (int corner = 0; corner < nverts; ++corner) {
        const std::size_t flat = static_cast<std::size_t>(
            cell * state.corner_stride + corner);
        const int node = new_mesh.csr_indices[flat];
        const double mass = new_corner_mass[flat];
        node_mass[static_cast<std::size_t>(node)] += mass;
        corner_cell.push_back(cell);
        corner_node.push_back(node);
        corner_mass.push_back(mass);
      }
    }
    std::vector<double> node_pr(new_nodes, 0.0);
    std::vector<double> node_pz(new_nodes, 0.0);
    for (std::size_t node = 0; node < new_nodes; ++node) {
      node_pr[node] = node_mass[node] * new_velocity_r[node];
      node_pz[node] = node_mass[node] * new_velocity_z[node];
    }

    const hydro::CarrierProjectionInput projection_input{
        static_cast<int>(n_masters),
        master_node.data(),
        master_fix_r.data(),
        static_cast<int>(slave_node.size()),
        slave_node.empty() ? nullptr : slave_node.data(),
        slave_edge.empty() ? nullptr : slave_edge.data(),
        slave_lambda.empty() ? nullptr : slave_lambda.data(),
        new_mesh.n_nodes,
        node_mass.data(),
        node_pr.data(),
        node_pz.data()};
    const hydro::CarrierProjectionResult projection =
        hydro::project_carrier_boundary_velocity(
            projection_input, new_velocity_r.data(), new_velocity_z.data());
    if (!projection.ok) {
      return skip(state, "remap", projection.reject_reason);
    }

    std::vector<double> delta_e_cell(n_cells, 0.0);
    const hydro::CarrierDefectAllocation allocation =
        hydro::allocate_projection_defect_to_cells(
            projection_input, new_velocity_r.data(), new_velocity_z.data(),
            static_cast<int>(corner_cell.size()), corner_cell.data(),
            corner_node.data(), corner_mass.data(), new_mesh.n_cells,
            delta_e_cell.data());
    if (!allocation.ok) {
      return skip(state, "remap", allocation.reject_reason);
    }
    TENRYU_ASSERT(
        std::abs(allocation.total - projection.delta_k) <=
            1.0e-9 * std::max(projection.delta_k, 1.0),
        "carrier projection energy allocation mismatch");
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
      if (delta_e_cell[cell] > 0.0) {
        TENRYU_ASSERT(new_mass[cell] > 0.0,
                      "carrier projection positive cell mass");
        new_ei[cell] += delta_e_cell[cell] / new_mass[cell];
      }
    }
    projection_delta_k = projection.delta_k;
  }
  reale_vdiag_scan("post_projection", state.step, new_velocity_r,
                   new_velocity_z);

  const double rezone_min_dt = cfg.numerics.ale.rezone_min_dt_s;
  if (rezone_min_dt > 0.0) {
    const StagedDtResult staged_dt = evaluate_staged_hydro_dt(
        new_mesh.csr_offsets.data(), new_mesh.csr_indices.data(),
        new_mesh.nverts.data(), new_mesh.n_cells, new_mesh.n_nodes,
        new_mesh.node_r.data(), new_mesh.node_z.data(),
        new_velocity_r.data(), new_velocity_z.data(), cfg);
    if (staged_dt.min_dt <= rezone_min_dt) {
      char buffer[160];
      std::snprintf(buffer, sizeof(buffer),
                    "staged_dt_%.3e_below_%.1e_cell_%d",
                    staged_dt.min_dt, rezone_min_dt,
                    staged_dt.binding_cell);
      return skip(state, "staged_dt", buffer);
    }
  }

  std::vector<std::pair<double, double>> carrier_old_coordinates;
  if (state.boundary_carrier.valid) {
    carrier_old_coordinates.reserve(state.boundary_carrier.masters.size());
    for (const mesh::CarrierVertex& master :
         state.boundary_carrier.masters) {
      carrier_old_coordinates.emplace_back(
          old_mesh.node_r[static_cast<std::size_t>(master.mesh_node)],
          old_mesh.node_z[static_cast<std::size_t>(master.mesh_node)]);
    }
  }

  const auto install_start = std::chrono::steady_clock::now();
  mesh::Mesh& mesh = state.mesh;
  mesh::MultiBlockTopology& topology = *mesh.topo.multiblock;
  mesh.topo.n_nodes = new_mesh.n_nodes;
  mesh.topo.node_flags = new_mesh.node_flags;
  mesh.cell_nverts.assign(new_mesh.nverts.begin(), new_mesh.nverts.end());
  mesh.geometry_exempt_cells.clear();
  mesh.button_center.reset();
  topology.cell_node_csr_offsets = std::move(new_mesh.csr_offsets);
  topology.cell_node_csr_indices = std::move(new_mesh.csr_indices);
  topology.cell_orientation_sign.assign(n_cells, 1);
  // consult-26 / S2D-ADAPT: structured block indexing is not meaningful on the
  // installed tessellation.
  state.mesh.topo.general_polygonal = true;
  state.cell_nverts_host.assign(
      mesh.cell_nverts.begin(), mesh.cell_nverts.end());
  install_node_fields(state, new_mesh, new_velocity_r, new_velocity_z);
  mesh.node_r = state.x_r.data();
  mesh.node_z = state.x_z.data();

  mesh::rebuild_runtime_topology_caches(state, cfg);
  // The input was a polar-tier mesh, but its structured tier/ring metadata no
  // longer describes the reconnected Voronoi topology.
  topology.has_polar_tier = false;
  topology.polar_tier_columns.clear();
  topology.polar_tier_transition_radii.clear();
  topology.dendrite_block_rings.clear();
  topology.south_node_of.clear();
  mesh.multiblock_outer_svec_tangent_balance = false;
  mesh.recompute_geometry();

  if (state.boundary_carrier.valid) {
    bool refresh_failed = false;
    std::vector<bool> matched(state.boundary_carrier.masters.size(), false);
    if (state.carrier_domain_active &&
        !tessellation.node_boundary.empty()) {
      for (int node = 0; node < new_mesh.n_nodes; ++node) {
        if (new_mesh.node_class[static_cast<std::size_t>(node)] !=
            static_cast<std::uint8_t>(
                mesh::voronoi::TessNodeClass::kCarrierMaster)) {
          continue;
        }
        const int domain_vertex =
            new_mesh.node_domain_vertex[static_cast<std::size_t>(node)];
        TENRYU_ASSERT(
            domain_vertex >= 0 &&
                static_cast<std::size_t>(domain_vertex) <
                    state.boundary_carrier.masters.size(),
            "carrier_master_index_range");
        state.boundary_carrier
            .masters[static_cast<std::size_t>(domain_vertex)]
            .mesh_node = node;
        matched[static_cast<std::size_t>(domain_vertex)] = true;
      }
    } else {
      std::map<std::pair<double, double>, int> master_by_coordinate;
      for (std::size_t master = 0;
           master < carrier_old_coordinates.size(); ++master) {
        const auto [position, inserted] = master_by_coordinate.emplace(
            carrier_old_coordinates[master], static_cast<int>(master));
        (void)position;
        if (!inserted) {
          refresh_failed = true;
        }
      }
      for (int node = 0; node < new_mesh.n_nodes; ++node) {
        const std::pair<double, double> coordinate{
            new_mesh.node_r[static_cast<std::size_t>(node)],
            new_mesh.node_z[static_cast<std::size_t>(node)]};
        const auto master = master_by_coordinate.find(coordinate);
        if (master != master_by_coordinate.end() &&
            master->first.first == coordinate.first &&
            master->first.second == coordinate.second) {
          state.boundary_carrier
              .masters[static_cast<std::size_t>(master->second)]
              .mesh_node = node;
          matched[static_cast<std::size_t>(master->second)] = true;
        }
      }
    }
    if (std::find(matched.begin(), matched.end(), false) != matched.end()) {
      refresh_failed = true;
    }
    if (refresh_failed) {
      state.boundary_carrier.valid = false;
      core::log_warning("[carrier] refresh failed; carrier invalidated");
    }
  }
  if (state.boundary_carrier.valid &&
      !tessellation.node_boundary.empty()) {
    state.carrier_domain_active = true;
  }
  state.carrier_node_class = new_mesh.node_class;
  state.carrier_node_edge = new_mesh.node_carrier_edge;
  state.carrier_node_lambda = new_mesh.node_lambda;

  state.mass = new_mass;
  state.ee = new_ee;
  state.ei = new_ei;
  state.corner_mass = new_corner_mass;
  state.corner_mass_initialized = true;
  state.corner_mass_is_lagrangian_invariant =
      hydro::corner_mass_lagrangian_invariant_enabled(cfg);
  state.vol = mesh.cell_vol;
  state.cell_vol_initial = mesh.cell_vol;
  state.vol_prev_hydro = mesh.cell_vol;
  std::vector<double> density(n_cells, 0.0);
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    if (!(mesh.cell_vol[cell] > 0.0) ||
        !std::isfinite(mesh.cell_vol[cell])) {
      core::tenryu_abort("positive ReALE committed volume",
                         "ReALE topology passed staging but committed a "
                         "non-positive cell volume",
                         __FILE__, __LINE__);
    }
    density[cell] = new_mass[cell] / mesh.cell_vol[cell];
  }
  state.rho = density;
  state.dt_prev_hydro = -1.0;
  state.invalidate_cell_material_props();
  state.ale_rezoned = true;
  ++state.ale_rezone_invocations;
  ++state.ale_remaps_applied;
  state.ale_last_applied_step = state.step;
  install_ms +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - install_start)
          .count();
  int axis_masters = 0;
  int axis_intervals = 0;
  if (state.boundary_carrier.valid) {
    for (const mesh::CarrierVertex& master :
         state.boundary_carrier.masters) {
      if (master.bc_class == mesh::CarrierBcClass::kAxis) {
        ++axis_masters;
        ++axis_intervals;
      }
    }
  }
  int folded_cells = 0;
  int folded_cells_dumped = 0;
  for (int cell = 0; cell < mesh.topo.n_cells; ++cell) {
    const int begin =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = mesh.cell_nverts[static_cast<std::size_t>(cell)];
    double ring_twice_area = 0.0;
    double min_edge = std::numeric_limits<double>::infinity();
    for (int vertex = 0; vertex < nverts; ++vertex) {
      const int next = vertex + 1 == nverts ? 0 : vertex + 1;
      const int node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + vertex)];
      const int next_node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + next)];
      const double vertex_r =
          new_mesh.node_r[static_cast<std::size_t>(node)];
      const double vertex_z =
          new_mesh.node_z[static_cast<std::size_t>(node)];
      const double next_r =
          new_mesh.node_r[static_cast<std::size_t>(next_node)];
      const double next_z =
          new_mesh.node_z[static_cast<std::size_t>(next_node)];
      ring_twice_area += vertex_r * next_z - next_r * vertex_z;
      min_edge = std::min(
          min_edge, std::hypot(next_r - vertex_r, next_z - vertex_z));
    }
    const double ring_orientation = ring_twice_area > 0.0 ? 1.0 : -1.0;
    int folds_in_cell = 0;
    for (int vertex = 0; vertex < nverts; ++vertex) {
      const int previous = vertex == 0 ? nverts - 1 : vertex - 1;
      const int next = vertex + 1 == nverts ? 0 : vertex + 1;
      const int previous_node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + previous)];
      const int node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + vertex)];
      const int next_node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + next)];
      const double previous_r =
          new_mesh.node_r[static_cast<std::size_t>(previous_node)];
      const double previous_z =
          new_mesh.node_z[static_cast<std::size_t>(previous_node)];
      const double vertex_r =
          new_mesh.node_r[static_cast<std::size_t>(node)];
      const double vertex_z =
          new_mesh.node_z[static_cast<std::size_t>(node)];
      const double next_r =
          new_mesh.node_r[static_cast<std::size_t>(next_node)];
      const double next_z =
          new_mesh.node_z[static_cast<std::size_t>(next_node)];
      const double fold_cross =
          (vertex_r - previous_r) * (next_z - vertex_z) -
          (vertex_z - previous_z) * (next_r - vertex_r);
      if (ring_orientation * fold_cross < 0.0) {
        ++folds_in_cell;
      }
    }
    if (folds_in_cell > 0) {
      ++folded_cells;
      if (folded_cells_dumped < 3) {
        std::fprintf(stderr,
                     "[mesh_fold_census] cell=%d nverts=%d min_edge=%.3e "
                     "folds_in_cell=%d\n",
                     cell, nverts, min_edge, folds_in_cell);
        ++folded_cells_dumped;
      }
    }
  }
  char stage_timings[320];
  std::snprintf(stage_timings, sizeof(stage_timings),
                " targets_ms=%.0f voronoi_ms=%.0f remap_ms=%.0f "
                "install_ms=%.0f carrier=%d proj_dk=%.3e pages=%zu"
                " axis_masters=%d axis_intervals=%d"
                " weld=%d wclust=%d wdisp=%.1e"
                " tg=%.0f tess=%.0f legacy=%.0f other=%.0f lloyd=%d",
                targets_ms, voronoi_ms, remap_ms, install_ms,
                state.carrier_domain_active ? 1 : 0,
                projection_delta_k, state.carrier_ring_pages.pages.size(),
                axis_masters, axis_intervals,
                tessellation.weld_contracted, tessellation.weld_clusters,
                tessellation.weld_max_displacement,
                targets.target_generation_ms, targets.tessellation_ms,
                targets.legacy_assembly_ms, targets_other_ms,
                targets.cvt_iterations_used);
  reale_trigger::snapshot_reference(state);
  g_last_rezone_commit_step = state.step;
  g_post_rezone_dt_pending = true;
  g_post_rezone_dt_ref = -1.0;
  core::log_info(
      "[reale_v2] COMMIT step=" + std::to_string(state.step) +
      " cells=" + std::to_string(old_mesh.n_cells) +
      " nodes_old=" + std::to_string(old_mesh.n_nodes) +
      " nodes_new=" + std::to_string(new_mesh.n_nodes) +
      " remaps_total=" + std::to_string(state.ale_remaps_applied) +
      " skipped_total=" + std::to_string(state.reale_rezone_skipped) +
      stage_timings + " folds=" + std::to_string(folded_cells) +
      " trigger=" + std::string(trigger_reason));
  // One-shot installed-mesh dump for offline geometry forensics.
  static bool installed_mesh_dumped = false;
  if (const char* const path =
          std::getenv("TENRYU_REALE_DUMP_INSTALLED")) {
    if (!installed_mesh_dumped) {
      std::FILE* const output = std::fopen(path, "w");
      TENRYU_ASSERT(output != nullptr, "open ReALE installed mesh dump");
      std::vector<double> installed_node_r;
      std::vector<double> installed_node_z;
      state.x_r.copy_to_host(installed_node_r);
      state.x_z.copy_to_host(installed_node_z);
      std::fprintf(output, "nodes %d\n", mesh.topo.n_nodes);
      for (int node = 0; node < mesh.topo.n_nodes; ++node) {
        std::fprintf(output, "%.17g %.17g\n",
                     installed_node_r[static_cast<std::size_t>(node)],
                     installed_node_z[static_cast<std::size_t>(node)]);
      }
      std::fprintf(output, "cells %d\n", mesh.topo.n_cells);
      for (int cell = 0; cell < mesh.topo.n_cells; ++cell) {
        const int begin =
            topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
        const int nverts = mesh.cell_nverts[static_cast<std::size_t>(cell)];
        std::fprintf(output, "%d", nverts);
        for (int corner = 0; corner < nverts; ++corner) {
          std::fprintf(
              output, " %d",
              topology.cell_node_csr_indices[
                  static_cast<std::size_t>(begin + corner)]);
        }
        std::fprintf(output, "\n");
      }
      TENRYU_ASSERT(std::fclose(output) == 0,
                    "close ReALE installed mesh dump");
      installed_mesh_dumped = true;
    }
  }
  reale_vdiag_rho_scan("post_install_rho", state.step, state);
  return true;
}

}  // namespace tenryu::coupling::reale_mode
