#include "hydro/axis_cell_ledger.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {
namespace {

constexpr int kColumnStride = 192;
constexpr int kTrackedCellCount = 20;
constexpr int kFirstAxisRow = 275;
constexpr int kLastAxisRow = 287;
constexpr int kCellCornerCount = 4;
constexpr int kBlockSize = 64;
constexpr std::array<int, 5> kTrackedRows = {275, 280, 285, 286, 287};
constexpr std::array<int, 4> kTrackedColumns = {0, 1, 2, 96};

constexpr std::array<int, kTrackedCellCount> make_tracked_cells() {
  std::array<int, kTrackedCellCount> cells{};
  int slot = 0;
  for (const int row : kTrackedRows) {
    for (const int column : kTrackedColumns) {
      cells[static_cast<std::size_t>(slot++)] =
          row * kColumnStride + column;
    }
  }
  return cells;
}

constexpr std::array<int, kTrackedCellCount> kTrackedCells =
    make_tracked_cells();

struct AxisLedgerGate {
  bool enabled = false;
  double t_start = 0.0;
};

struct CellStateCapture {
  double vol = 0.0;
  double rho = 0.0;
  double mass = 0.0;
  double ee = 0.0;
  double ei = 0.0;
  double pressure = 0.0;
  double temperature = 0.0;
  double work_p = 0.0;
  double work_av = 0.0;
  double work_sub = 0.0;
};

struct CellPositionCapture {
  double r[kCellCornerCount]{};
  double z[kCellCornerCount]{};
};

struct NodePosition {
  double r = 0.0;
  double z = 0.0;
};

struct AxisNodeCapture {
  double r_p = 0.0;
  double z_p = 0.0;
  double u_r_p = 0.0;
  double u_z_p = 0.0;
  double r_q = 0.0;
  double z_q = 0.0;
  double u_r_q = 0.0;
  double u_z_q = 0.0;
};

struct AxisNodePartner {
  int axis_node = -1;
  int partner_node = -1;
  int radial_row = -1;
};

struct AxisCellLedgerState {
  core::DeviceArray<int> tracked_cells;
  core::DeviceArray<CellStateCapture> begin_cell_state;
  core::DeviceArray<CellStateCapture> end_cell_state;
  core::DeviceArray<CellPositionCapture> begin_cell_positions;
  core::DeviceArray<CellPositionCapture> end_cell_positions;
  core::DeviceArray<int> axis_nodes;
  core::DeviceArray<int> partner_nodes;
  core::DeviceArray<AxisNodeCapture> end_axis_state;
  std::vector<AxisNodePartner> axis_pairs;
  bool initialized = false;
  bool capture_open = false;
  int begin_step = -1;
  double begin_time = 0.0;
};

const AxisLedgerGate& axis_ledger_gate() {
  static const AxisLedgerGate gate = []() {
    AxisLedgerGate result;
    const char* raw = std::getenv("TENRYU_I1B_AXISCELL_LEDGER");
    if (raw == nullptr || raw[0] == '\0') {
      return result;
    }
    char* end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || *end != '\0' || !std::isfinite(value)) {
      return result;
    }
    result.enabled = true;
    result.t_start = value;
    return result;
  }();
  return gate;
}

AxisCellLedgerState& axis_ledger_state() {
  static AxisCellLedgerState state;
  return state;
}

__global__ void capture_cell_state_kernel(
    CellStateCapture* captures,
    const int* cell_ids,
    const double* vol,
    const double* rho,
    const double* mass,
    const double* ee,
    const double* ei,
    const double* pe,
    const double* pi,
    const double* temperature,
    const double* work_p,
    const double* work_av,
    const double* work_sub,
    const bool capture_work) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kTrackedCellCount) {
    return;
  }
  const int cell = cell_ids[slot];
  CellStateCapture capture;
  capture.vol = vol[cell];
  capture.rho = rho[cell];
  capture.mass = mass[cell];
  capture.ee = ee[cell];
  capture.ei = ei[cell];
  capture.pressure = pe[cell] + pi[cell];
  capture.temperature = temperature[cell];
  if (capture_work) {
    capture.work_p = work_p[cell];
    capture.work_av = work_av[cell];
    capture.work_sub = work_sub[cell];
  }
  captures[slot] = capture;
}

__global__ void capture_cell_positions_kernel(
    CellPositionCapture* captures,
    const int* cell_ids,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const double* x_r,
    const double* x_z) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= kTrackedCellCount) {
    return;
  }
  const int cell = cell_ids[slot];
  const int offset = cell_node_csr_offsets[cell];
  CellPositionCapture capture;
  for (int corner = 0; corner < kCellCornerCount; ++corner) {
    const int node = cell_node_csr_indices[offset + corner];
    capture.r[corner] = x_r[node];
    capture.z[corner] = x_z[node];
  }
  captures[slot] = capture;
}

__global__ void gather_node_positions_kernel(NodePosition* captures,
                                             const int* node_ids,
                                             const int count,
                                             const double* x_r,
                                             const double* x_z) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= count) {
    return;
  }
  const int node = node_ids[slot];
  captures[slot].r = x_r[node];
  captures[slot].z = x_z[node];
}

__global__ void capture_axis_nodes_kernel(AxisNodeCapture* captures,
                                          const int* axis_nodes,
                                          const int* partner_nodes,
                                          const int count,
                                          const double* x_r,
                                          const double* x_z,
                                          const double* v_r,
                                          const double* v_z) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= count) {
    return;
  }
  const int p = axis_nodes[slot];
  const int q = partner_nodes[slot];
  AxisNodeCapture capture;
  capture.r_p = x_r[p];
  capture.z_p = x_z[p];
  capture.u_r_p = v_r[p];
  capture.u_z_p = v_z[p];
  capture.r_q = x_r[q];
  capture.z_q = x_z[q];
  capture.u_r_q = v_r[q];
  capture.u_z_q = v_z[q];
  captures[slot] = capture;
}

void validate_cell_fields(const core::State& state) {
  const std::size_t required =
      static_cast<std::size_t>(kTrackedCells.back()) + 1U;
  TENRYU_ASSERT(state.vol.size() >= required && state.rho.size() >= required &&
                    state.mass.size() >= required && state.ee.size() >= required &&
                    state.ei.size() >= required && state.Pe.size() >= required &&
                    state.Pi.size() >= required && state.Te.size() >= required,
                "axis cell ledger requires all tracked cell state fields");
}

void validate_work_fields(const core::State& state) {
  const std::size_t required =
      static_cast<std::size_t>(kTrackedCells.back()) + 1U;
  TENRYU_ASSERT(state.work_p_per_cell.size() >= required &&
                    state.work_av_per_cell.size() >= required &&
                    state.work_sub_per_cell.size() >= required,
                "axis cell ledger requires all tracked corrector work fields");
}

const NodePosition& node_position(
    const int node,
    const std::vector<int>& node_ids,
    const std::vector<NodePosition>& positions) {
  for (std::size_t slot = 0; slot < node_ids.size(); ++slot) {
    if (node_ids[slot] == node) {
      return positions[slot];
    }
  }
  TENRYU_ASSERT(false, "axis cell ledger node position lookup failed");
  return positions.front();
}

void build_axis_node_map(AxisCellLedgerState& ledger,
                         const core::State& state,
                         const core::Config& cfg) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "axis cell ledger requires multiblock topology");
  const auto& multiblock = *state.mesh.topo.multiblock;
  const int last_cell = kLastAxisRow * kColumnStride;
  TENRYU_ASSERT(static_cast<std::size_t>(last_cell) + 1U <
                    multiblock.cell_node_csr_offsets.size(),
                "axis cell ledger tracked shell cell CSR offset is missing");

  std::vector<int> candidate_nodes;
  for (int row = kFirstAxisRow; row <= kLastAxisRow; ++row) {
    const int cell = row * kColumnStride;
    const int offset =
        multiblock.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int next =
        multiblock.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
    TENRYU_ASSERT(next - offset >= kCellCornerCount,
                  "axis cell ledger tracked shell cell has fewer than four CSR slots");
    TENRYU_ASSERT(offset >= 0 &&
                      static_cast<std::size_t>(offset + kCellCornerCount) <=
                          multiblock.cell_node_csr_indices.size(),
                  "axis cell ledger tracked shell cell CSR indices are missing");
    for (int corner = 0; corner < kCellCornerCount; ++corner) {
      const int node = multiblock.cell_node_csr_indices[
          static_cast<std::size_t>(offset + corner)];
      TENRYU_ASSERT(node >= 0 &&
                        static_cast<std::size_t>(node) < state.x_r.size() &&
                        static_cast<std::size_t>(node) < state.x_z.size(),
                    "axis cell ledger tracked shell node is out of range");
      if (std::find(candidate_nodes.begin(), candidate_nodes.end(), node) ==
          candidate_nodes.end()) {
        candidate_nodes.push_back(node);
      }
    }
  }
  TENRYU_ASSERT(candidate_nodes.size() <= 50U,
                "axis cell ledger candidate node set exceeds the diagnostic bound");

  core::DeviceArray<int> device_candidate_nodes(candidate_nodes.size());
  core::DeviceArray<NodePosition> device_candidate_positions(
      candidate_nodes.size());
  device_candidate_nodes.copy_from_host(candidate_nodes);
  const int candidate_count = static_cast<int>(candidate_nodes.size());
  const int candidate_blocks =
      (candidate_count + kBlockSize - 1) / kBlockSize;
  gather_node_positions_kernel<<<candidate_blocks, kBlockSize>>>(
      device_candidate_positions.data(), device_candidate_nodes.data(),
      candidate_count, state.x_r.data(), state.x_z.data());
  CUDA_CHECK(cudaGetLastError());
  std::vector<NodePosition> candidate_positions;
  device_candidate_positions.copy_to_host(candidate_positions);

  ledger.axis_pairs.clear();
  for (int row = kFirstAxisRow; row <= kLastAxisRow; ++row) {
    const int cell = row * kColumnStride;
    const int offset =
        multiblock.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    std::array<int, kCellCornerCount> nodes{};
    std::array<bool, kCellCornerCount> on_axis{};
    int axis_count = 0;
    for (int corner = 0; corner < kCellCornerCount; ++corner) {
      nodes[static_cast<std::size_t>(corner)] =
          multiblock.cell_node_csr_indices[
              static_cast<std::size_t>(offset + corner)];
      on_axis[static_cast<std::size_t>(corner)] =
          node_position(nodes[static_cast<std::size_t>(corner)],
                        candidate_nodes, candidate_positions)
              .r <= cfg.numerics.axis_eps_cm;
      axis_count += on_axis[static_cast<std::size_t>(corner)] ? 1 : 0;
    }
    TENRYU_ASSERT(axis_count == 2,
                  "axis cell ledger tracked shell cell must have two axis nodes");

    std::array<int, 2> axis_corners{};
    int axis_slot = 0;
    for (int corner = 0; corner < kCellCornerCount; ++corner) {
      if (on_axis[static_cast<std::size_t>(corner)]) {
        axis_corners[static_cast<std::size_t>(axis_slot++)] = corner;
      }
    }
    const NodePosition& first_axis_position = node_position(
        nodes[static_cast<std::size_t>(axis_corners[0])], candidate_nodes,
        candidate_positions);
    const NodePosition& second_axis_position = node_position(
        nodes[static_cast<std::size_t>(axis_corners[1])], candidate_nodes,
        candidate_positions);
    const double first_radius =
        std::hypot(first_axis_position.r, first_axis_position.z);
    const double second_radius =
        std::hypot(second_axis_position.r, second_axis_position.z);
    TENRYU_ASSERT(first_radius != second_radius,
                  "axis cell ledger cannot order the two radial rings");

    for (int local_axis = 0; local_axis < 2; ++local_axis) {
      const int corner = axis_corners[static_cast<std::size_t>(local_axis)];
      const int previous = (corner + kCellCornerCount - 1) % kCellCornerCount;
      const int next = (corner + 1) % kCellCornerCount;
      const bool previous_is_partner =
          !on_axis[static_cast<std::size_t>(previous)];
      const bool next_is_partner = !on_axis[static_cast<std::size_t>(next)];
      TENRYU_ASSERT(previous_is_partner != next_is_partner,
                    "axis cell ledger axis node must have one same-ring off-axis partner");
      const int partner_corner = previous_is_partner ? previous : next;
      const int axis_node = nodes[static_cast<std::size_t>(corner)];
      const int partner_node =
          nodes[static_cast<std::size_t>(partner_corner)];
      const double radius = local_axis == 0 ? first_radius : second_radius;
      const int radial_row =
          radius == std::min(first_radius, second_radius) ? row : row + 1;

      auto existing = std::find_if(
          ledger.axis_pairs.begin(), ledger.axis_pairs.end(),
          [axis_node](const AxisNodePartner& pair) {
            return pair.axis_node == axis_node;
          });
      if (existing == ledger.axis_pairs.end()) {
        ledger.axis_pairs.push_back({axis_node, partner_node, radial_row});
      } else {
        TENRYU_ASSERT(existing->partner_node == partner_node &&
                          existing->radial_row == radial_row,
                      "axis cell ledger deduplicated node has inconsistent topology");
      }
    }
  }
  TENRYU_ASSERT(ledger.axis_pairs.size() <= 50U,
                "axis cell ledger D2 map exceeds the diagnostic bound");

  std::vector<int> axis_nodes;
  std::vector<int> partner_nodes;
  axis_nodes.reserve(ledger.axis_pairs.size());
  partner_nodes.reserve(ledger.axis_pairs.size());
  for (const AxisNodePartner& pair : ledger.axis_pairs) {
    axis_nodes.push_back(pair.axis_node);
    partner_nodes.push_back(pair.partner_node);
  }
  ledger.axis_nodes.reset(axis_nodes.size());
  ledger.partner_nodes.reset(partner_nodes.size());
  ledger.end_axis_state.reset(axis_nodes.size());
  ledger.axis_nodes.copy_from_host(axis_nodes);
  ledger.partner_nodes.copy_from_host(partner_nodes);

  core::log_info("[axis-ledger-D2] map_size=" +
                 std::to_string(ledger.axis_pairs.size()));
}

void initialize_ledger(AxisCellLedgerState& ledger,
                       const core::State& state,
                       const core::Config& cfg) {
  if (ledger.initialized) {
    return;
  }
  validate_cell_fields(state);
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() >
                    static_cast<std::size_t>(kTrackedCells.back()) + 1U,
                "axis cell ledger device CSR offsets are missing");
  TENRYU_ASSERT(state.v_r.size() == state.x_r.size() &&
                    state.v_z.size() == state.x_z.size(),
                "axis cell ledger requires node position/velocity size agreement");
  ledger.tracked_cells.reset(kTrackedCellCount);
  ledger.tracked_cells.copy_from_host(kTrackedCells.data());
  ledger.begin_cell_state.reset(kTrackedCellCount);
  ledger.end_cell_state.reset(kTrackedCellCount);
  ledger.begin_cell_positions.reset(kTrackedCellCount);
  ledger.end_cell_positions.reset(kTrackedCellCount);
  build_axis_node_map(ledger, state, cfg);
  ledger.initialized = true;
}

void capture_cell_state(core::DeviceArray<CellStateCapture>& destination,
                        const AxisCellLedgerState& ledger,
                        const core::State& state,
                        const bool capture_work) {
  const int blocks =
      (kTrackedCellCount + kBlockSize - 1) / kBlockSize;
  capture_cell_state_kernel<<<blocks, kBlockSize>>>(
      destination.data(), ledger.tracked_cells.data(), state.vol.data(),
      state.rho.data(), state.mass.data(), state.ee.data(), state.ei.data(),
      state.Pe.data(), state.Pi.data(), state.Te.data(),
      capture_work ? state.work_p_per_cell.data() : nullptr,
      capture_work ? state.work_av_per_cell.data() : nullptr,
      capture_work ? state.work_sub_per_cell.data() : nullptr, capture_work);
  CUDA_CHECK(cudaGetLastError());
}

void capture_cell_positions(
    core::DeviceArray<CellPositionCapture>& destination,
    const AxisCellLedgerState& ledger,
    const core::State& state) {
  const int blocks =
      (kTrackedCellCount + kBlockSize - 1) / kBlockSize;
  capture_cell_positions_kernel<<<blocks, kBlockSize>>>(
      destination.data(), ledger.tracked_cells.data(),
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(), state.x_r.data(),
      state.x_z.data());
  CUDA_CHECK(cudaGetLastError());
}

void capture_axis_state(AxisCellLedgerState& ledger,
                        const core::State& state) {
  const int count = static_cast<int>(ledger.axis_pairs.size());
  if (count == 0) {
    return;
  }
  const int blocks = (count + kBlockSize - 1) / kBlockSize;
  capture_axis_nodes_kernel<<<blocks, kBlockSize>>>(
      ledger.end_axis_state.data(), ledger.axis_nodes.data(),
      ledger.partner_nodes.data(), count, state.x_r.data(), state.x_z.data(),
      state.v_r.data(), state.v_z.data());
  CUDA_CHECK(cudaGetLastError());
}

void log_cell_lines(const AxisCellLedgerState& ledger, const double dt_op) {
  std::vector<CellStateCapture> begin;
  std::vector<CellStateCapture> end;
  ledger.begin_cell_state.copy_to_host(begin);
  ledger.end_cell_state.copy_to_host(end);
  const double line_time = ledger.begin_time + dt_op;
  const int line_step = ledger.begin_step + 1;
  const double tiny = std::numeric_limits<double>::min();

  for (int slot = 0; slot < kTrackedCellCount; ++slot) {
    const CellStateCapture& before = begin[static_cast<std::size_t>(slot)];
    const CellStateCapture& after = end[static_cast<std::size_t>(slot)];
    const double vdot_geom = (after.vol - before.vol) / dt_op;
    const double vdot_mass =
        (before.mass / after.rho - before.mass / before.rho) / dt_op;
    const double pressure_half = 0.5 * (before.pressure + after.pressure);
    const double vdot_work = -after.work_p / std::max(pressure_half, tiny);
    const double u_before = before.mass * (before.ee + before.ei);
    const double u_after = before.mass * (after.ee + after.ei);
    const double du_rate = (u_after - u_before) / dt_op;
    const double energy_residual =
        du_rate - (after.work_p + after.work_av + after.work_sub);
    const int row = slot / static_cast<int>(kTrackedColumns.size());
    const int column = slot % static_cast<int>(kTrackedColumns.size());

    std::ostringstream stream;
    stream << std::scientific << "[axis-ledger] step=" << line_step
           << " t=" << std::setprecision(6) << line_time
           << " cell=" << kTrackedCells[static_cast<std::size_t>(slot)]
           << " i=" << kTrackedRows[static_cast<std::size_t>(row)]
           << " j=" << kTrackedColumns[static_cast<std::size_t>(column)]
           << " V=" << std::setprecision(6) << after.vol
           << " dVg=" << std::setprecision(3) << vdot_geom
           << " dVm=" << vdot_mass << " dVw=" << vdot_work
           << " Wp=" << after.work_p << " Wav=" << after.work_av
           << " Wsub=" << after.work_sub << " dU=" << du_rate
           << " RE=" << energy_residual << " p=" << pressure_half
           << " Te=" << after.temperature;
    core::log_info(stream.str());
  }
}

void log_d2_line(const AxisCellLedgerState& ledger, const double dt_op) {
  std::vector<AxisNodeCapture> captures;
  ledger.end_axis_state.copy_to_host(captures);
  if (captures.empty()) {
    return;
  }

  std::size_t max_slot = 0;
  double max_ru = captures[0].r_q *
                      (captures[0].u_z_q - captures[0].u_z_p) -
                  (captures[0].z_q - captures[0].z_p) * captures[0].u_r_q;
  for (std::size_t slot = 1; slot < captures.size(); ++slot) {
    const AxisNodeCapture& capture = captures[slot];
    const double ru = capture.r_q * (capture.u_z_q - capture.u_z_p) -
                      (capture.z_q - capture.z_p) * capture.u_r_q;
    if (std::abs(ru) > std::abs(max_ru)) {
      max_ru = ru;
      max_slot = slot;
    }
  }

  const AxisNodeCapture& maximum = captures[max_slot];
  std::ostringstream stream;
  stream << std::scientific << "[axis-ledger-D2] step="
         << ledger.begin_step + 1 << " t=" << std::setprecision(6)
         << ledger.begin_time + dt_op << " maxRu=" << std::setprecision(6)
         << std::abs(max_ru)
         << " at_row=" << ledger.axis_pairs[max_slot].radial_row
         << " uRP=" << std::setprecision(3) << maximum.u_r_p
         << " RP=" << maximum.r_p;
  core::log_info(stream.str());
}

}  // namespace

void axis_cell_ledger_step_begin(core::State& state, const core::Config& cfg) {
  const AxisLedgerGate& gate = axis_ledger_gate();
  if (!gate.enabled || state.t < gate.t_start) {
    return;
  }
  AxisCellLedgerState& ledger = axis_ledger_state();
  initialize_ledger(ledger, state, cfg);
  validate_cell_fields(state);
  capture_cell_state(ledger.begin_cell_state, ledger, state, false);
  capture_cell_positions(ledger.begin_cell_positions, ledger, state);
  ledger.begin_step = state.step;
  ledger.begin_time = state.t;
  ledger.capture_open = true;
}

void axis_cell_ledger_step_end(core::State& state,
                               const core::Config&,
                               const double dt_op) {
  const AxisLedgerGate& gate = axis_ledger_gate();
  if (!gate.enabled) {
    return;
  }
  AxisCellLedgerState& ledger = axis_ledger_state();
  if (!ledger.capture_open) {
    return;
  }
  TENRYU_ASSERT(dt_op > 0.0 && std::isfinite(dt_op),
                "axis cell ledger requires finite positive dt_op");
  TENRYU_ASSERT(ledger.begin_step == state.step,
                "axis cell ledger begin/end step mismatch");
  validate_cell_fields(state);
  validate_work_fields(state);
  capture_cell_state(ledger.end_cell_state, ledger, state, true);
  capture_cell_positions(ledger.end_cell_positions, ledger, state);
  capture_axis_state(ledger, state);
  log_cell_lines(ledger, dt_op);
  log_d2_line(ledger, dt_op);
  ledger.capture_open = false;
}

}  // namespace tenryu::hydro
