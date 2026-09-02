#include "hydro/corner_collapse_ledger.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/pentagon_geometry.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {
namespace {

constexpr int kTransitionBeltRole = 7;
constexpr int kMaximumCorners = 5;
constexpr int kNeighborhoodRings = 2;
constexpr std::size_t kAcceptedStepCapacity = 256U;

static_assert(
    static_cast<int>(mesh::BlockRole::TRANSITION_BELT) ==
        kTransitionBeltRole,
    "collapse ledger numeric transition-belt role changed");

struct WatchCell {
  int cell = -1;
  char pole = '?';
};

struct CellRecord {
  int step = -1;
  CornerCollapseLedgerTrial trial =
      CornerCollapseLedgerTrial::Accepted;
  char pole = '?';
  int cell = -1;
  int nverts = 0;
  double t_n = 0.0;
  double dt = 0.0;
  double area = 0.0;
  double volume = 0.0;
  double q_balance = 0.0;
  double min_j = 0.0;
  double min_jhat = 0.0;
  double volume_crosscheck = 0.0;
  int root_corner = -1;
  double tau_star = 0.0;
  double t_star = 0.0;
  std::array<double, kMaximumCorners> corner_j{};
  std::array<double, kMaximumCorners> corner_jhat{};
  std::array<double, kMaximumCorners> corner_volume{};
  std::array<double, kMaximumCorners> corner_tau{};
  std::array<bool, kMaximumCorners> corner_has_root{};
};

struct StepCapture {
  int step = -1;
  bool emitted = false;
  std::vector<CellRecord> records;
};

struct CornerCollapseLedgerState {
  bool initialized = false;
  int block_id = -1;
  int north_seed = -1;
  int south_seed = -1;
  double block_mean_initial_radius = 0.0;
  std::vector<WatchCell> watch_cells;
  std::deque<StepCapture> accepted_steps;
  std::uint64_t flush_count = 0;
};

bool collapse_ledger_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_COLLAPSE_LEDGER");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

CornerCollapseLedgerState& collapse_ledger_state() {
  static CornerCollapseLedgerState state;
  return state;
}

int active_nverts(const core::State& state, const int cell) {
  return mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, cell);
}

std::vector<int> cell_nodes(const core::State& state, const int cell) {
  const auto& multiblock = *state.mesh.topo.multiblock;
  const int nverts = active_nverts(state, cell);
  const int offset = multiblock.cell_node_csr_offsets[
      static_cast<std::size_t>(cell)];
  const int end = multiblock.cell_node_csr_offsets[
      static_cast<std::size_t>(cell) + 1U];
  TENRYU_ASSERT(offset >= 0 && end - offset >= nverts &&
                    static_cast<std::size_t>(end) <=
                        multiblock.cell_node_csr_indices.size(),
                "collapse ledger cell-node CSR is incomplete");
  std::vector<int> nodes;
  nodes.reserve(static_cast<std::size_t>(nverts));
  for (int corner = 0; corner < nverts; ++corner) {
    nodes.push_back(multiblock.cell_node_csr_indices[
        static_cast<std::size_t>(offset + corner)]);
  }
  return nodes;
}

std::array<double, 2> cell_vertex_mean(
    const core::State& state,
    const int cell,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  const std::vector<int> nodes = cell_nodes(state, cell);
  double sum_r = 0.0;
  double sum_z = 0.0;
  for (const int node : nodes) {
    TENRYU_ASSERT(node >= 0 &&
                      static_cast<std::size_t>(node) < node_r.size() &&
                      static_cast<std::size_t>(node) < node_z.size(),
                  "collapse ledger cell node is out of range");
    sum_r += node_r[static_cast<std::size_t>(node)];
    sum_z += node_z[static_cast<std::size_t>(node)];
  }
  const double inverse_count = 1.0 / static_cast<double>(nodes.size());
  return {sum_r * inverse_count, sum_z * inverse_count};
}

std::vector<int> two_ring_neighborhood(const core::State& state,
                                       const int seed) {
  const auto& multiblock = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  std::vector<unsigned char> selected(static_cast<std::size_t>(n_cells), 0U);
  std::vector<int> frontier{seed};
  selected[static_cast<std::size_t>(seed)] = 1U;
  for (int ring = 0; ring < kNeighborhoodRings; ++ring) {
    std::vector<int> next;
    for (const int cell : frontier) {
      const int offset = multiblock.face_adj_csr_offsets[
          static_cast<std::size_t>(cell)];
      const int end = multiblock.face_adj_csr_offsets[
          static_cast<std::size_t>(cell) + 1U];
      TENRYU_ASSERT(offset >= 0 && end >= offset &&
                        static_cast<std::size_t>(end) <=
                            multiblock.face_adj_csr_indices.size(),
                    "collapse ledger face-adjacency CSR is incomplete");
      for (int entry = offset; entry < end; ++entry) {
        const int neighbor = multiblock.face_adj_csr_indices[
            static_cast<std::size_t>(entry)];
        if (neighbor < 0 || neighbor >= n_cells ||
            selected[static_cast<std::size_t>(neighbor)] != 0U) {
          continue;
        }
        selected[static_cast<std::size_t>(neighbor)] = 1U;
        next.push_back(neighbor);
      }
    }
    frontier = std::move(next);
  }

  std::vector<int> cells;
  for (int cell = 0; cell < n_cells; ++cell) {
    if (selected[static_cast<std::size_t>(cell)] != 0U) {
      cells.push_back(cell);
    }
  }
  return cells;
}

std::string comma_separated_cells(const std::vector<int>& cells) {
  std::ostringstream stream;
  for (std::size_t slot = 0; slot < cells.size(); ++slot) {
    if (slot != 0U) {
      stream << ',';
    }
    stream << cells[slot];
  }
  return stream.str();
}

void initialize_ledger(CornerCollapseLedgerState& ledger,
                       const core::State& state) {
  if (ledger.initialized) {
    return;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "collapse ledger requires multiblock topology");
  const auto& multiblock = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0 &&
                    multiblock.cell_block_id.size() ==
                        static_cast<std::size_t>(n_cells) &&
                    multiblock.cell_orientation_sign.size() ==
                        static_cast<std::size_t>(n_cells) &&
                    multiblock.cell_node_csr_offsets.size() ==
                        static_cast<std::size_t>(n_cells) + 1U &&
                    multiblock.face_adj_csr_offsets.size() ==
                        static_cast<std::size_t>(n_cells) + 1U,
                "collapse ledger requires complete multiblock metadata");
  TENRYU_ASSERT(state.x_r_initial.size() ==
                        static_cast<std::size_t>(n_nodes) &&
                    state.x_z_initial.size() ==
                        static_cast<std::size_t>(n_nodes),
                "collapse ledger requires initial node coordinates");

  std::vector<double> initial_r;
  std::vector<double> initial_z;
  state.x_r_initial.copy_to_host(initial_r);
  state.x_z_initial.copy_to_host(initial_z);

  int outermost_block = -1;
  double outermost_mean_radius =
      -std::numeric_limits<double>::infinity();
  for (int block_id = 0;
       block_id < static_cast<int>(multiblock.blocks.size());
       ++block_id) {
    const mesh::BlockInfo& block =
        multiblock.blocks[static_cast<std::size_t>(block_id)];
    if (static_cast<int>(block.role) != kTransitionBeltRole) {
      continue;
    }
    TENRYU_ASSERT(block.cell_count > 0 && block.cell_begin >= 0 &&
                      block.cell_begin + block.cell_count <= n_cells,
                  "collapse ledger transition block cell range is invalid");
    double radius_sum = 0.0;
    for (int local = 0; local < block.cell_count; ++local) {
      const int cell = block.cell_begin + local;
      const auto center =
          cell_vertex_mean(state, cell, initial_r, initial_z);
      radius_sum += std::hypot(center[0], center[1]);
    }
    const double mean_radius =
        radius_sum / static_cast<double>(block.cell_count);
    if (mean_radius > outermost_mean_radius) {
      outermost_mean_radius = mean_radius;
      outermost_block = block_id;
    }
  }
  TENRYU_ASSERT(outermost_block >= 0,
                "collapse ledger found no numeric role-7 block");

  const mesh::BlockInfo& block =
      multiblock.blocks[static_cast<std::size_t>(outermost_block)];
  int north_seed = -1;
  int south_seed = -1;
  double north_theta = std::numeric_limits<double>::infinity();
  double south_theta = -std::numeric_limits<double>::infinity();
  for (int local = 0; local < block.cell_count; ++local) {
    const int cell = block.cell_begin + local;
    if (active_nverts(state, cell) != 5) {
      continue;
    }
    const auto center = cell_vertex_mean(state, cell, initial_r, initial_z);
    const double theta = std::atan2(center[0], center[1]);
    if (theta < north_theta) {
      north_theta = theta;
      north_seed = cell;
    }
    if (theta > south_theta) {
      south_theta = theta;
      south_seed = cell;
    }
  }
  TENRYU_ASSERT(north_seed >= 0 && south_seed >= 0 &&
                    north_seed != south_seed,
                "collapse ledger outermost role-7 block requires distinct "
                "north/south pentagons");

  const std::vector<int> north_cells =
      two_ring_neighborhood(state, north_seed);
  const std::vector<int> south_cells =
      two_ring_neighborhood(state, south_seed);
  for (const int cell : north_cells) {
    const int nverts = active_nverts(state, cell);
    TENRYU_ASSERT(nverts == 4 || nverts == 5,
                  "collapse ledger north watch set requires quad/pentagon cells");
    ledger.watch_cells.push_back({cell, 'N'});
  }
  for (const int cell : south_cells) {
    const int nverts = active_nverts(state, cell);
    TENRYU_ASSERT(nverts == 4 || nverts == 5,
                  "collapse ledger south watch set requires quad/pentagon cells");
    ledger.watch_cells.push_back({cell, 'S'});
  }

  ledger.block_id = outermost_block;
  ledger.north_seed = north_seed;
  ledger.south_seed = south_seed;
  ledger.block_mean_initial_radius = outermost_mean_radius;
  ledger.initialized = true;

  std::ostringstream manifest;
  manifest << std::scientific << std::setprecision(17)
           << "[collapse-ledger] manifest role=" << kTransitionBeltRole
           << " block_id=" << ledger.block_id
           << " mean_initial_radius=" << ledger.block_mean_initial_radius
           << " north_seed=" << ledger.north_seed
           << " south_seed=" << ledger.south_seed
           << " north_cells=" << comma_separated_cells(north_cells)
           << " south_cells=" << comma_separated_cells(south_cells);
  core::log_info(manifest.str());
}

double determinant(const double ar,
                   const double az,
                   const double br,
                   const double bz) {
  return ar * bz - az * br;
}

bool admissible_root(const double root) {
  return std::isfinite(root) && root > 0.0 && root <= 1.0;
}

bool earliest_quadratic_root(const double j0,
                             const double j1,
                             const double j2,
                             double& root) {
  const double scale = std::max(
      {std::abs(j0), std::abs(j1), std::abs(j2),
       std::numeric_limits<double>::min()});
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * scale;
  if (std::abs(j2) <= tolerance) {
    if (std::abs(j1) <= tolerance) {
      return false;
    }
    const double linear_root = -j0 / j1;
    if (!admissible_root(linear_root)) {
      return false;
    }
    root = linear_root;
    return true;
  }

  double discriminant = j1 * j1 - 4.0 * j2 * j0;
  const double discriminant_scale =
      std::abs(j1 * j1) + std::abs(4.0 * j2 * j0);
  const double discriminant_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      discriminant_scale;
  if (discriminant < 0.0 && discriminant >= -discriminant_tolerance) {
    discriminant = 0.0;
  }
  if (!(discriminant >= 0.0) || !std::isfinite(discriminant)) {
    return false;
  }

  const double square_root = std::sqrt(discriminant);
  const double q = -0.5 * (j1 + std::copysign(square_root, j1));
  const double first = q / j2;
  const double second = q != 0.0 ? j0 / q : first;
  bool found = false;
  double earliest = std::numeric_limits<double>::infinity();
  for (const double candidate : {first, second}) {
    if (admissible_root(candidate) && candidate < earliest) {
      earliest = candidate;
      found = true;
    }
  }
  if (found) {
    root = earliest;
  }
  return found;
}

const char* trial_name(const CornerCollapseLedgerTrial trial) {
  switch (trial) {
    case CornerCollapseLedgerTrial::Accepted:
      return "accepted";
    case CornerCollapseLedgerTrial::RejectedPostrestore:
      return "rejected_postrestore";
  }
  return "unknown";
}

CellRecord capture_cell(const core::State& state,
                        const WatchCell& watched,
                        const int step,
                        const double t_n,
                        const double dt,
                        const CornerCollapseLedgerTrial trial,
                        const std::vector<double>& node_r,
                        const std::vector<double>& node_z,
                        const std::vector<double>& velocity_r,
                        const std::vector<double>& velocity_z,
                        const std::vector<double>& state_volume) {
  const auto& multiblock = *state.mesh.topo.multiblock;
  CellRecord record;
  record.step = step;
  record.trial = trial;
  record.pole = watched.pole;
  record.cell = watched.cell;
  record.nverts = active_nverts(state, watched.cell);
  record.t_n = t_n;
  record.dt = dt;
  TENRYU_ASSERT(record.nverts == 4 || record.nverts == 5,
                "collapse ledger capture requires quad/pentagon cells");
  const int orientation = multiblock.cell_orientation_sign[
      static_cast<std::size_t>(watched.cell)];
  TENRYU_ASSERT(orientation == -1 || orientation == 1,
                "collapse ledger cell orientation must be +/-1");
  const double sign = static_cast<double>(orientation);
  const std::vector<int> nodes = cell_nodes(state, watched.cell);
  std::array<double, kMaximumCorners> r{};
  std::array<double, kMaximumCorners> z{};
  std::array<double, kMaximumCorners> ur{};
  std::array<double, kMaximumCorners> uz{};
  for (int corner = 0; corner < record.nverts; ++corner) {
    const int node = nodes[static_cast<std::size_t>(corner)];
    const std::size_t index = static_cast<std::size_t>(node);
    TENRYU_ASSERT(index < node_r.size() && index < node_z.size() &&
                      index < velocity_r.size() && index < velocity_z.size(),
                  "collapse ledger captured node is out of range");
    r[static_cast<std::size_t>(corner)] = node_r[index];
    z[static_cast<std::size_t>(corner)] = node_z[index];
    ur[static_cast<std::size_t>(corner)] = velocity_r[index];
    uz[static_cast<std::size_t>(corner)] = velocity_z[index];
  }

  double shoelace = 0.0;
  double rz_sum = 0.0;
  record.min_j = std::numeric_limits<double>::infinity();
  record.min_jhat = std::numeric_limits<double>::infinity();
  double max_j = -std::numeric_limits<double>::infinity();
  double earliest_root = std::numeric_limits<double>::infinity();
  for (int corner = 0; corner < record.nverts; ++corner) {
    const int previous =
        (corner + record.nverts - 1) % record.nverts;
    const int next = (corner + 1) % record.nverts;
    const double edge_minus_r =
        r[static_cast<std::size_t>(corner)] -
        r[static_cast<std::size_t>(previous)];
    const double edge_minus_z =
        z[static_cast<std::size_t>(corner)] -
        z[static_cast<std::size_t>(previous)];
    const double edge_plus_r =
        r[static_cast<std::size_t>(next)] -
        r[static_cast<std::size_t>(corner)];
    const double edge_plus_z =
        z[static_cast<std::size_t>(next)] -
        z[static_cast<std::size_t>(corner)];
    const double j0 =
        sign * determinant(edge_minus_r, edge_minus_z,
                           edge_plus_r, edge_plus_z);
    record.corner_j[static_cast<std::size_t>(corner)] = j0;
    record.min_j = std::min(record.min_j, j0);
    max_j = std::max(max_j, j0);

    const double edge_product =
        std::hypot(edge_minus_r, edge_minus_z) *
        std::hypot(edge_plus_r, edge_plus_z);
    const double jhat = j0 / edge_product;
    record.corner_jhat[static_cast<std::size_t>(corner)] = jhat;
    record.min_jhat = std::min(record.min_jhat, jhat);

    const double velocity_minus_r =
        dt * (ur[static_cast<std::size_t>(corner)] -
              ur[static_cast<std::size_t>(previous)]);
    const double velocity_minus_z =
        dt * (uz[static_cast<std::size_t>(corner)] -
              uz[static_cast<std::size_t>(previous)]);
    const double velocity_plus_r =
        dt * (ur[static_cast<std::size_t>(next)] -
              ur[static_cast<std::size_t>(corner)]);
    const double velocity_plus_z =
        dt * (uz[static_cast<std::size_t>(next)] -
              uz[static_cast<std::size_t>(corner)]);
    const double j1 = sign *
        (determinant(velocity_minus_r, velocity_minus_z,
                     edge_plus_r, edge_plus_z) +
         determinant(edge_minus_r, edge_minus_z,
                     velocity_plus_r, velocity_plus_z));
    const double j2 =
        sign * determinant(velocity_minus_r, velocity_minus_z,
                           velocity_plus_r, velocity_plus_z);
    double corner_root = 0.0;
    if (earliest_quadratic_root(j0, j1, j2, corner_root)) {
      record.corner_has_root[static_cast<std::size_t>(corner)] = true;
      record.corner_tau[static_cast<std::size_t>(corner)] = corner_root;
      if (corner_root < earliest_root) {
        earliest_root = corner_root;
        record.root_corner = corner;
      }
    }

    const int edge_next = (corner + 1) % record.nverts;
    const double cross =
        r[static_cast<std::size_t>(corner)] *
            z[static_cast<std::size_t>(edge_next)] -
        r[static_cast<std::size_t>(edge_next)] *
            z[static_cast<std::size_t>(corner)];
    shoelace += cross;
    rz_sum += cross *
              (r[static_cast<std::size_t>(corner)] +
               r[static_cast<std::size_t>(edge_next)]);
  }
  record.area = 0.5 * sign * shoelace;
  record.volume = (kPentagonGeometryPi / 3.0) * sign * rz_sum;
  record.q_balance = record.min_j / max_j;
  if (record.root_corner >= 0) {
    record.tau_star = earliest_root;
    record.t_star = t_n + earliest_root * dt;
  }

  if (record.nverts == 5) {
    PentagonPoint points[5];
    for (int corner = 0; corner < 5; ++corner) {
      points[corner] = {r[static_cast<std::size_t>(corner)],
                        z[static_cast<std::size_t>(corner)]};
    }
    pentagon_corner_rz_volumes(points, record.corner_volume.data());
  } else {
    rz::compute_quad_corner_volumes_exact_subpolygon(
        r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3],
        record.corner_volume.data());
  }

  TENRYU_ASSERT(watched.cell >= 0 &&
                    static_cast<std::size_t>(watched.cell) <
                        state_volume.size(),
                "collapse ledger state volume is out of range");
  const double runtime_volume =
      state_volume[static_cast<std::size_t>(watched.cell)];
  const double volume_scale =
      std::max({std::abs(record.volume), std::abs(runtime_volume),
                std::numeric_limits<double>::min()});
  record.volume_crosscheck =
      std::abs(record.volume - runtime_volume) / volume_scale;
  return record;
}

void log_record(const CellRecord& record) {
  std::ostringstream summary;
  summary << std::scientific << std::setprecision(17)
          << "[collapse-ledger] step=" << record.step
          << " trial=" << trial_name(record.trial)
          << " pole=" << record.pole
          << " cell=" << record.cell
          << " nverts=" << record.nverts
          << " t_n=" << record.t_n
          << " dt=" << record.dt
          << " A_c=" << record.area
          << " V_c_RZ=" << record.volume
          << " q_bal=" << record.q_balance
          << " minJ=" << record.min_j
          << " minJhat=" << record.min_jhat;
  if (record.root_corner >= 0) {
    summary << " root_corner=" << record.root_corner
            << " tau_star=" << record.tau_star
            << " t_star=" << record.t_star;
  } else {
    summary << " root_corner=- tau_star=- t_star=-";
  }
  summary << " vol_xcheck=" << record.volume_crosscheck;
  core::log_info(summary.str());

  std::ostringstream detail;
  detail << std::scientific << std::setprecision(17)
         << "[collapse-ledger-detail] step=" << record.step
         << " trial=" << trial_name(record.trial)
         << " pole=" << record.pole
         << " cell=" << record.cell
         << " J_ck=";
  for (int corner = 0; corner < record.nverts; ++corner) {
    if (corner != 0) {
      detail << ',';
    }
    detail << corner << ':'
           << record.corner_j[static_cast<std::size_t>(corner)];
  }
  detail << " Jhat_ck=";
  for (int corner = 0; corner < record.nverts; ++corner) {
    if (corner != 0) {
      detail << ',';
    }
    detail << corner << ':'
           << record.corner_jhat[static_cast<std::size_t>(corner)];
  }
  detail << " V_ck_RZ=";
  for (int corner = 0; corner < record.nverts; ++corner) {
    if (corner != 0) {
      detail << ',';
    }
    detail << corner << ':'
           << record.corner_volume[static_cast<std::size_t>(corner)];
  }
  detail << " tau_ck=";
  for (int corner = 0; corner < record.nverts; ++corner) {
    if (corner != 0) {
      detail << ',';
    }
    detail << corner << ':';
    if (record.corner_has_root[static_cast<std::size_t>(corner)]) {
      detail << record.corner_tau[static_cast<std::size_t>(corner)];
    } else {
      detail << '-';
    }
  }
  core::log_info(detail.str());
}

void flush_records(CornerCollapseLedgerState& ledger,
                   const StepCapture* rejected) {
  std::vector<const CellRecord*> pending;
  for (const StepCapture& capture : ledger.accepted_steps) {
    if (!capture.emitted) {
      for (const CellRecord& record : capture.records) {
        pending.push_back(&record);
      }
    }
  }
  if (rejected != nullptr) {
    for (const CellRecord& record : rejected->records) {
      pending.push_back(&record);
    }
  }
  if (pending.empty()) {
    return;
  }

  ++ledger.flush_count;
  const CellRecord* maximum = pending.front();
  for (const CellRecord* record : pending) {
    if (record->volume_crosscheck > maximum->volume_crosscheck) {
      maximum = record;
    }
  }
  std::ostringstream crosscheck;
  crosscheck << std::scientific << std::setprecision(17)
             << "[collapse-ledger-xcheck] flush=" << ledger.flush_count
             << " records=" << pending.size()
             << " max_vol_xcheck=" << maximum->volume_crosscheck
             << " step=" << maximum->step
             << " trial=" << trial_name(maximum->trial)
             << " pole=" << maximum->pole
             << " cell=" << maximum->cell;
  core::log_info(crosscheck.str());
  for (const CellRecord* record : pending) {
    log_record(*record);
  }
  for (StepCapture& capture : ledger.accepted_steps) {
    capture.emitted = true;
  }
}

StepCapture capture_step(CornerCollapseLedgerState& ledger,
                         const core::State& state,
                         const int step,
                         const double t_n,
                         const double dt,
                         const CornerCollapseLedgerTrial trial) {
  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt) && std::isfinite(t_n),
                "collapse ledger requires finite t_n and positive dt");
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size() &&
                    state.x_r.size() == state.v_r.size() &&
                    state.x_r.size() == state.v_z.size(),
                "collapse ledger requires node field size agreement");
  TENRYU_ASSERT(state.vol.size() ==
                    static_cast<std::size_t>(state.mesh.topo.n_cells),
                "collapse ledger requires complete state volumes");

  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  std::vector<double> state_volume;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);
  state.vol.copy_to_host(state_volume);

  StepCapture capture;
  capture.step = step;
  capture.records.reserve(ledger.watch_cells.size());
  for (const WatchCell& watched : ledger.watch_cells) {
    capture.records.push_back(capture_cell(
        state, watched, step, t_n, dt, trial, node_r, node_z, velocity_r,
        velocity_z, state_volume));
  }
  // Force-component decomposition is intentionally omitted here; the
  // independent TENRYU_I1B_FORCE_DUMP diagnostic can be enabled alongside.
  return capture;
}

}  // namespace

void corner_collapse_ledger_capture(
    const core::State& state,
    const int step,
    const double t_n,
    const double dt,
    const CornerCollapseLedgerTrial trial) {
  if (!collapse_ledger_enabled()) {
    return;
  }
  CornerCollapseLedgerState& ledger = collapse_ledger_state();
  initialize_ledger(ledger, state);
  StepCapture capture = capture_step(ledger, state, step, t_n, dt, trial);
  if (trial == CornerCollapseLedgerTrial::Accepted) {
    if (!ledger.accepted_steps.empty() &&
        !ledger.accepted_steps.back().emitted &&
        ledger.accepted_steps.back().step == step) {
      StepCapture& current_step = ledger.accepted_steps.back();
      current_step.records.insert(
          current_step.records.end(),
          std::make_move_iterator(capture.records.begin()),
          std::make_move_iterator(capture.records.end()));
    } else {
      ledger.accepted_steps.push_back(std::move(capture));
    }
    while (ledger.accepted_steps.size() > kAcceptedStepCapacity) {
      ledger.accepted_steps.pop_front();
    }
    return;
  }
  flush_records(ledger, &capture);
}

void corner_collapse_ledger_flush() {
  if (!collapse_ledger_enabled()) {
    return;
  }
  CornerCollapseLedgerState& ledger = collapse_ledger_state();
  if (!ledger.initialized) {
    return;
  }
  flush_records(ledger, nullptr);
}

}  // namespace tenryu::hydro
