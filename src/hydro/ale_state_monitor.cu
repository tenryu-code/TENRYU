#include "hydro/ale_state_monitor.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/button_morph_indexing.hpp"
#include "hydro/cfl.hpp"
#include "hydro/corner_j_predict_cfl.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {

bool fold_diag_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("TENRYU_FOLD_DIAG");
    return value != nullptr && value[0] != '\0' &&
           std::string(value) != "0";
  }();
  return enabled;
}

namespace {

constexpr unsigned long long kDoubleSignBit = 0x8000000000000000ULL;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kMaxTrackedCells = 16;

__device__ unsigned long long ordered_double_bits(const double value) {
  const unsigned long long bits =
      static_cast<unsigned long long>(__double_as_longlong(value));
  return (bits & kDoubleSignBit) != 0ULL ? ~bits : bits ^ kDoubleSignBit;
}

unsigned long long ordered_double_bits_host(const double value) {
  unsigned long long bits = 0ULL;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return (bits & kDoubleSignBit) != 0ULL ? ~bits : bits ^ kDoubleSignBit;
}

double ordered_bits_double_host(const unsigned long long ordered) {
  const unsigned long long bits =
      (ordered & kDoubleSignBit) != 0ULL ? ordered ^ kDoubleSignBit : ~ordered;
  double value = 0.0;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

const std::vector<int>& fold_diag_tracked_cell_ids() {
  static const std::vector<int> ids = []() {
    std::vector<int> parsed_ids;
    const char* raw = std::getenv("TENRYU_FOLD_DIAG_CELLS");
    if (raw == nullptr || raw[0] == '\0') {
      return parsed_ids;
    }

    std::istringstream stream(raw);
    std::string token;
    while (parsed_ids.size() < static_cast<std::size_t>(kMaxTrackedCells) &&
           std::getline(stream, token, ',')) {
      char* end = nullptr;
      const long value = std::strtol(token.c_str(), &end, 10);
      while (end != nullptr && *end != '\0' &&
             std::isspace(static_cast<unsigned char>(*end)) != 0) {
        ++end;
      }
      if (end == token.c_str() || end == nullptr || *end != '\0' ||
          value < static_cast<long>(std::numeric_limits<int>::min()) ||
          value > static_cast<long>(std::numeric_limits<int>::max())) {
        continue;
      }
      parsed_ids.push_back(static_cast<int>(value));
    }
    return parsed_ids;
  }();
  return ids;
}

__device__ bool cell_in_monitor_region(const int cell,
                                       const int shell_cell_offset,
                                       const int ntheta,
                                       const int shell_rows) {
  return cell < shell_cell_offset ||
         (cell >= shell_cell_offset && shell_rows > 0 && ntheta > 0 &&
          (cell - shell_cell_offset) / ntheta < shell_rows);
}

struct CellMonitorValues {
  double q = std::numeric_limits<double>::infinity();
  double h = std::numeric_limits<double>::infinity();
};

__device__ double rz_polygon_volume_4(const double r[4],
                                      const double z[4]) {
  double sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    sum += (r[k] + r[kp]) * (r[k] * z[kp] - r[kp] * z[k]);
  }
  return (kPi / 3.0) * sum;
}

__device__ void quad_corner_subvolumes(const double r[4],
                                       const double z[4],
                                       double volume[4]) {
  const double centroid_r = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double centroid_z = 0.25 * (z[0] + z[1] + z[2] + z[3]);
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    const int km = (k + 3) & 3;
    const double sub_r[4] = {
        r[k], 0.5 * (r[k] + r[kp]), centroid_r,
        0.5 * (r[km] + r[k]),
    };
    const double sub_z[4] = {
        z[k], 0.5 * (z[k] + z[kp]), centroid_z,
        0.5 * (z[km] + z[k]),
    };
    volume[k] = rz_polygon_volume_4(sub_r, sub_z);
  }
}

struct CellFoldValues {
  double r_z = -1.0;
  double h_metric = -1.0;
  double u_theta = 0.0;
  double s_u_theta = 0.0;
};

__device__ CellFoldValues compute_cell_fold_values(
    const int cell,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ ref_corner_fractions) {
  CellFoldValues values;
  const int nv =
      cell_nverts != nullptr ? static_cast<int>(cell_nverts[cell]) : 4;
  if (nv != 4) {
    return values;
  }

  const int offset = cell_node_offsets[cell];
  int nodes[4] = {};
  double r[4] = {};
  double z[4] = {};
  double u_r[4] = {};
  double u_z[4] = {};
  for (int k = 0; k < 4; ++k) {
    nodes[k] = cell_node_indices[offset + k];
    r[k] = x_r[nodes[k]];
    z[k] = x_z[nodes[k]];
    u_r[k] = v_r[nodes[k]];
    u_z[k] = v_z[nodes[k]];
  }

  // Exact for a parallelogram; the alternating corner sum is the standard
  // nonaffine-velocity approximation for a general quadrilateral.
  const double hourglass_r = 0.25 * (u_r[0] - u_r[1] + u_r[2] - u_r[3]);
  const double hourglass_z = 0.25 * (u_z[0] - u_z[1] + u_z[2] - u_z[3]);
  const double r_z = std::hypot(hourglass_r, hourglass_z);
  if (std::isfinite(r_z)) {
    values.r_z = r_z;
  }

  const double centroid_r = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double centroid_z = 0.25 * (z[0] + z[1] + z[2] + z[3]);
  const double velocity_r = 0.25 * (u_r[0] + u_r[1] + u_r[2] + u_r[3]);
  const double velocity_z = 0.25 * (u_z[0] + u_z[1] + u_z[2] + u_z[3]);
  const double theta = std::atan2(centroid_r, centroid_z);
  const double u_theta =
      velocity_r * std::cos(theta) - velocity_z * std::sin(theta);
  const double s = std::hypot(centroid_r, centroid_z);
  if (std::isfinite(u_theta) && std::isfinite(s)) {
    values.u_theta = u_theta;
    values.s_u_theta = s * u_theta;
  }

  double subvolume[4] = {};
  quad_corner_subvolumes(r, z, subvolume);
  double total_volume = 0.0;
  bool valid = true;
  for (int k = 0; k < 4; ++k) {
    const double reference_fraction = ref_corner_fractions[4 * cell + k];
    valid = valid && subvolume[k] > 0.0 && std::isfinite(subvolume[k]) &&
            reference_fraction > 0.0 &&
            std::isfinite(reference_fraction);
    total_volume += subvolume[k];
  }
  valid = valid && total_volume > 0.0 && std::isfinite(total_volume);
  if (valid) {
    double h_metric_squared = 0.0;
    for (int k = 0; k < 4; ++k) {
      const double reference_fraction = ref_corner_fractions[4 * cell + k];
      const double delta =
          (subvolume[k] / total_volume) / reference_fraction - 1.0;
      h_metric_squared += delta * delta;
    }
    if (h_metric_squared >= 0.0 && std::isfinite(h_metric_squared)) {
      values.h_metric = std::sqrt(h_metric_squared);
    }
  }
  return values;
}

__device__ CellMonitorValues compute_cell_monitor_values(
    const int cell,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ ref_x_r,
    const double* __restrict__ ref_x_z,
    const double dt_acoustic_scale) {
  CellMonitorValues values;
  const int offset = cell_node_offsets[cell];
  const int nv =
      cell_nverts != nullptr ? static_cast<int>(cell_nverts[cell]) : 4;

  double shoelace = 0.0;
  double ref_shoelace = 0.0;
  for (int k = 0; k < nv; ++k) {
    const int kp = (k + 1 == nv) ? 0 : k + 1;
    const int node = cell_node_indices[offset + k];
    const int node_p = cell_node_indices[offset + kp];
    shoelace += x_r[node] * x_z[node_p] - x_r[node_p] * x_z[node];
    ref_shoelace += ref_x_r[node] * ref_x_z[node_p] -
                     ref_x_r[node_p] * ref_x_z[node];
  }
  const double orientation = shoelace < 0.0 ? -1.0 : 1.0;
  const double ref_orientation = ref_shoelace < 0.0 ? -1.0 : 1.0;

  for (int k = 0; k < nv; ++k) {
    const int kp = (k + 1 == nv) ? 0 : k + 1;
    const int km = (k == 0) ? nv - 1 : k - 1;
    const int node = cell_node_indices[offset + k];
    const int node_p = cell_node_indices[offset + kp];
    const int node_m = cell_node_indices[offset + km];

    const double e1r = x_r[node_p] - x_r[node];
    const double e1z = x_z[node_p] - x_z[node];
    const double e2r = x_r[node_m] - x_r[node];
    const double e2z = x_z[node_m] - x_z[node];
    const double q_geom = detail::ale_monitor_corner_quality(
        e1r, e1z, e2r, e2z, orientation);

    const double ref_e1r = ref_x_r[node_p] - ref_x_r[node];
    const double ref_e1z = ref_x_z[node_p] - ref_x_z[node];
    const double ref_e2r = ref_x_r[node_m] - ref_x_r[node];
    const double ref_e2z = ref_x_z[node_m] - ref_x_z[node];
    const double q_healthy = detail::ale_monitor_corner_quality(
        ref_e1r, ref_e1z, ref_e2r, ref_e2z, ref_orientation);
    const double q_normalized = q_healthy > 1.0e-14
                                    ? q_geom / q_healthy
                                    : q_geom;
    const double q_clamped =
        std::fmin(1.0e3, std::fmax(-1.0e3, q_normalized));
    values.q = std::fmin(values.q, q_clamped);

    if (!std::isfinite(x_r[node]) || !std::isfinite(x_z[node]) ||
        !std::isfinite(v_r[node]) || !std::isfinite(v_z[node]) ||
        !std::isfinite(x_r[node_p]) || !std::isfinite(x_z[node_p]) ||
        !std::isfinite(v_r[node_p]) || !std::isfinite(v_z[node_p]) ||
        !std::isfinite(x_r[node_m]) || !std::isfinite(x_z[node_m]) ||
        !std::isfinite(v_r[node_m]) || !std::isfinite(v_z[node_m])) {
      continue;
    }

    const double d1r = v_r[node_p] - v_r[node];
    const double d1z = v_z[node_p] - v_z[node];
    const double d2r = v_r[node_m] - v_r[node];
    const double d2z = v_z[node_m] - v_z[node];
    const double j0 = detail::corner_j_predict_cross2(e1r, e1z, e2r, e2z);
    const double b = detail::corner_j_predict_cross2(e1r, e1z, d2r, d2z) +
                     detail::corner_j_predict_cross2(d1r, d1z, e2r, e2z);
    const double a = detail::corner_j_predict_cross2(d1r, d1z, d2r, d2z);
    if (!(orientation * j0 > 0.0) || !std::isfinite(j0) ||
        !std::isfinite(a) || !std::isfinite(b)) {
      continue;
    }

    const double root =
        detail::corner_j_predict_smallest_positive_root(a, b, j0);
    values.h = std::fmin(values.h, root / dt_acoustic_scale);
  }
  return values;
}

__global__ void ale_monitor_reference_fractions_kernel(
    const int n_cells,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    double* __restrict__ reference_fractions) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells) {
    return;
  }

  for (int k = 0; k < 4; ++k) {
    reference_fractions[4 * cell + k] = -1.0;
  }
  const int nv = cell_nverts != nullptr ? static_cast<int>(cell_nverts[cell]) : 4;
  if (nv != 4) {
    return;
  }

  const int offset = cell_node_offsets[cell];
  double r[4] = {};
  double z[4] = {};
  for (int k = 0; k < 4; ++k) {
    const int node = cell_node_indices[offset + k];
    r[k] = x_r[node];
    z[k] = x_z[node];
  }
  double subvolume[4] = {};
  quad_corner_subvolumes(r, z, subvolume);
  const double total_volume =
      subvolume[0] + subvolume[1] + subvolume[2] + subvolume[3];
  if (!(total_volume > 0.0) || !std::isfinite(total_volume)) {
    return;
  }
  for (int k = 0; k < 4; ++k) {
    if (!(subvolume[k] > 0.0) || !std::isfinite(subvolume[k])) {
      return;
    }
  }
  for (int k = 0; k < 4; ++k) {
    reference_fractions[4 * cell + k] = subvolume[k] / total_volume;
  }
}

__global__ void ale_monitor_min_kernel(
    const int n_cells,
    const int shell_cell_offset,
    const int ntheta,
    const int shell_rows,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ ref_x_r,
    const double* __restrict__ ref_x_z,
    const double dt_acoustic_scale,
    unsigned long long* __restrict__ q_min_ordered,
    unsigned long long* __restrict__ h_min_ordered,
    int* __restrict__ evaluated_cells) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells ||
      !cell_in_monitor_region(cell, shell_cell_offset, ntheta, shell_rows)) {
    return;
  }
  const CellMonitorValues values = compute_cell_monitor_values(
      cell, cell_node_offsets, cell_node_indices, cell_nverts, x_r, x_z, v_r,
      v_z, ref_x_r, ref_x_z, dt_acoustic_scale);
  atomicMin(q_min_ordered, ordered_double_bits(values.q));
  atomicMin(h_min_ordered, ordered_double_bits(values.h));
  atomicAdd(evaluated_cells, 1);
}

__global__ void ale_monitor_argmin_kernel(
    const int n_cells,
    const int shell_cell_offset,
    const int ntheta,
    const int shell_rows,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ ref_x_r,
    const double* __restrict__ ref_x_z,
    const double dt_acoustic_scale,
    const unsigned long long q_min_ordered,
    const unsigned long long h_min_ordered,
    int* __restrict__ q_min_cell,
    int* __restrict__ h_min_cell) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells ||
      !cell_in_monitor_region(cell, shell_cell_offset, ntheta, shell_rows)) {
    return;
  }
  const CellMonitorValues values = compute_cell_monitor_values(
      cell, cell_node_offsets, cell_node_indices, cell_nverts, x_r, x_z, v_r,
      v_z, ref_x_r, ref_x_z, dt_acoustic_scale);
  if (ordered_double_bits(values.q) == q_min_ordered) {
    atomicMin(q_min_cell, cell);
  }
  if (ordered_double_bits(values.h) == h_min_ordered) {
    atomicMin(h_min_cell, cell);
  }
}

__global__ void fold_diag_max_kernel(
    const int n_cells,
    const int shell_cell_offset,
    const int ntheta,
    const int shell_rows,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ ref_corner_fractions,
    unsigned long long* __restrict__ r_z_max_ordered,
    unsigned long long* __restrict__ h_metric_max_ordered) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells ||
      !cell_in_monitor_region(cell, shell_cell_offset, ntheta, shell_rows)) {
    return;
  }
  const CellFoldValues values = compute_cell_fold_values(
      cell, cell_node_offsets, cell_node_indices, cell_nverts, x_r, x_z, v_r,
      v_z, ref_corner_fractions);
  if (values.r_z >= 0.0 && std::isfinite(values.r_z)) {
    atomicMax(r_z_max_ordered, ordered_double_bits(values.r_z));
  }
  if (values.h_metric >= 0.0 && std::isfinite(values.h_metric)) {
    atomicMax(h_metric_max_ordered, ordered_double_bits(values.h_metric));
  }
}

__global__ void fold_diag_argmax_kernel(
    const int n_cells,
    const int shell_cell_offset,
    const int ntheta,
    const int shell_rows,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ ref_x_r,
    const double* __restrict__ ref_x_z,
    const double* __restrict__ ref_corner_fractions,
    const double dt_acoustic_scale,
    const unsigned long long r_z_max_ordered,
    const unsigned long long h_metric_max_ordered,
    int* __restrict__ r_z_max_cell,
    int* __restrict__ h_metric_max_cell,
    const int* __restrict__ tracked_cell_ids,
    const int tracked_cell_count,
    AleMonitorTrackedCellResult* __restrict__ tracked_values) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells ||
      !cell_in_monitor_region(cell, shell_cell_offset, ntheta, shell_rows)) {
    return;
  }
  const CellFoldValues fold_values = compute_cell_fold_values(
      cell, cell_node_offsets, cell_node_indices, cell_nverts, x_r, x_z, v_r,
      v_z, ref_corner_fractions);
  if (fold_values.r_z >= 0.0 &&
      ordered_double_bits(fold_values.r_z) == r_z_max_ordered) {
    atomicMin(r_z_max_cell, cell);
  }
  if (fold_values.h_metric >= 0.0 &&
      ordered_double_bits(fold_values.h_metric) == h_metric_max_ordered) {
    atomicMin(h_metric_max_cell, cell);
  }

  for (int slot = 0; slot < tracked_cell_count; ++slot) {
    if (tracked_cell_ids[slot] != cell) {
      continue;
    }
    const CellMonitorValues monitor_values = compute_cell_monitor_values(
        cell, cell_node_offsets, cell_node_indices, cell_nverts, x_r, x_z,
        v_r, v_z, ref_x_r, ref_x_z, dt_acoustic_scale);
    const int offset = cell_node_offsets[cell];
    const int nv =
        cell_nverts != nullptr ? static_cast<int>(cell_nverts[cell]) : 4;
    tracked_values[slot].cell = cell;
    tracked_values[slot].q = monitor_values.q;
    tracked_values[slot].r_z = fold_values.r_z;
    tracked_values[slot].h_metric = fold_values.h_metric;
    tracked_values[slot].u_theta = fold_values.u_theta;
    tracked_values[slot].s_u_theta = fold_values.s_u_theta;
    tracked_values[slot].alt_min =
        detail::csr_polygon_characteristic_length_from_nodes(
            x_r, x_z, cell_node_indices + offset, nv);
  }
}

__global__ void fold_diag_selected_values_kernel(
    const int r_z_max_cell,
    const int h_metric_max_cell,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ ref_corner_fractions,
    CellFoldValues* __restrict__ selected_values) {
  const int slot = threadIdx.x;
  if (slot >= 2) {
    return;
  }
  const int cell = slot == 0 ? r_z_max_cell : h_metric_max_cell;
  if (cell < 0) {
    return;
  }
  selected_values[slot] = compute_cell_fold_values(
      cell, cell_node_offsets, cell_node_indices, cell_nverts, x_r, x_z, v_r,
      v_z, ref_corner_fractions);
}

}  // namespace

void capture_ale_monitor_reference(core::State& state,
                                   const bool post_morph) {
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "ALE monitor reference capture requires matching coordinates");
  state.ale_monitor_ref_x_r.reset(state.x_r.size());
  state.ale_monitor_ref_x_z.reset(state.x_z.size());
  if (!state.x_r.empty()) {
    CUDA_CHECK(cudaMemcpy(state.ale_monitor_ref_x_r.data(),
                          state.x_r.data(),
                          state.x_r.size() * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(state.ale_monitor_ref_x_z.data(),
                          state.x_z.data(),
                          state.x_z.size() * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  }
  if (fold_diag_enabled()) {
    const int n_cells = state.mesh.topo.n_cells;
    TENRYU_ASSERT(state.mesh.dim == 2,
                  "ALE fold diagnostic reference requires a 2D mesh");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "ALE fold diagnostic reference requires CSR offsets");
    state.ale_monitor_ref_corner_fractions.reset(
        4U * static_cast<std::size_t>(std::max(n_cells, 0)));
    if (n_cells > 0) {
      const std::uint8_t* d_cell_nverts = nullptr;
      if (state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)) {
        auto* scratch = static_cast<std::uint8_t*>(core::device_scratch_acquire(
            "ale_state_monitor:cell_nverts",
            static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
        CUDA_CHECK(cudaMemcpy(scratch,
                              state.mesh.cell_nverts.data(),
                              static_cast<std::size_t>(n_cells) *
                                  sizeof(std::uint8_t),
                              cudaMemcpyHostToDevice));
        d_cell_nverts = scratch;
      }
      constexpr int threads = 256;
      const int blocks = (n_cells + threads - 1) / threads;
      ale_monitor_reference_fractions_kernel<<<blocks, threads>>>(
          n_cells,
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts,
          state.x_r.data(),
          state.x_z.data(),
          state.ale_monitor_ref_corner_fractions.data());
      CUDA_CHECK(cudaGetLastError());
    }
  }
  state.ale_monitor_ref_valid = true;
  state.ale_monitor_ref_post_morph = post_morph;
}

AleMonitorResult evaluate_ale_monitor(const core::State& state,
                                      const core::Config& cfg,
                                      const double dt_acoustic_now) {
  AleMonitorResult result;
  const auto indexing = button_morph::make_button_indexing(cfg.mesh);
  const int n_cells = state.mesh.topo.n_cells;
  if (n_cells <= 0) {
    return result;
  }

  TENRYU_ASSERT(state.mesh.dim == 2,
                "ALE monitor requires a 2D mesh");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "ALE monitor requires multiblock topology metadata");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "ALE monitor requires device CSR offsets");
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size() &&
                    state.x_r.size() == state.v_r.size() &&
                    state.x_r.size() == state.v_z.size(),
                "ALE monitor requires matching node field sizes");
  TENRYU_ASSERT(state.ale_monitor_ref_valid &&
                    state.ale_monitor_ref_x_r.size() == state.x_r.size() &&
                    state.ale_monitor_ref_x_z.size() == state.x_z.size(),
                "ALE monitor requires a valid healthy reference snapshot");

  const std::uint8_t* d_cell_nverts = nullptr;
  if (state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)) {
    auto* scratch = static_cast<std::uint8_t*>(core::device_scratch_acquire(
        "ale_state_monitor:cell_nverts",
        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMemcpy(scratch,
                          state.mesh.cell_nverts.data(),
                          static_cast<std::size_t>(n_cells) *
                              sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice));
    d_cell_nverts = scratch;
  }

  auto* d_q_min_ordered = static_cast<unsigned long long*>(
      core::device_scratch_acquire("ale_state_monitor:q_min_ordered",
                                   sizeof(unsigned long long)));
  auto* d_h_min_ordered = static_cast<unsigned long long*>(
      core::device_scratch_acquire("ale_state_monitor:h_min_ordered",
                                   sizeof(unsigned long long)));
  auto* d_q_min_cell = static_cast<int*>(core::device_scratch_acquire(
      "ale_state_monitor:q_min_cell", sizeof(int)));
  auto* d_h_min_cell = static_cast<int*>(core::device_scratch_acquire(
      "ale_state_monitor:h_min_cell", sizeof(int)));
  auto* d_evaluated_cells = static_cast<int*>(core::device_scratch_acquire(
      "ale_state_monitor:evaluated_cells", sizeof(int)));

  const double inf = std::numeric_limits<double>::infinity();
  unsigned long long q_min_ordered = ordered_double_bits_host(inf);
  unsigned long long h_min_ordered = ordered_double_bits_host(inf);
  int evaluated_cells = 0;
  CUDA_CHECK(cudaMemcpy(d_q_min_ordered,
                        &q_min_ordered,
                        sizeof(q_min_ordered),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_h_min_ordered,
                        &h_min_ordered,
                        sizeof(h_min_ordered),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_evaluated_cells,
                        &evaluated_cells,
                        sizeof(evaluated_cells),
                        cudaMemcpyHostToDevice));

  constexpr int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  const double dt_acoustic_scale = std::fmax(dt_acoustic_now, 1.0e-30);
  ale_monitor_min_kernel<<<blocks, threads>>>(
      n_cells,
      indexing.shell_cell_offset,
      indexing.ntheta,
      cfg.numerics.ale.runtime_controller.shell_rows,
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      d_cell_nverts,
      state.x_r.data(),
      state.x_z.data(),
      state.v_r.data(),
      state.v_z.data(),
      state.ale_monitor_ref_x_r.data(),
      state.ale_monitor_ref_x_z.data(),
      dt_acoustic_scale,
      d_q_min_ordered,
      d_h_min_ordered,
      d_evaluated_cells);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&q_min_ordered,
                        d_q_min_ordered,
                        sizeof(q_min_ordered),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_min_ordered,
                        d_h_min_ordered,
                        sizeof(h_min_ordered),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&evaluated_cells,
                        d_evaluated_cells,
                        sizeof(evaluated_cells),
                        cudaMemcpyDeviceToHost));
  if (evaluated_cells <= 0) {
    return result;
  }

  int q_min_cell = std::numeric_limits<int>::max();
  int h_min_cell = std::numeric_limits<int>::max();
  CUDA_CHECK(cudaMemcpy(d_q_min_cell,
                        &q_min_cell,
                        sizeof(q_min_cell),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_h_min_cell,
                        &h_min_cell,
                        sizeof(h_min_cell),
                        cudaMemcpyHostToDevice));
  ale_monitor_argmin_kernel<<<blocks, threads>>>(
      n_cells,
      indexing.shell_cell_offset,
      indexing.ntheta,
      cfg.numerics.ale.runtime_controller.shell_rows,
      state.mesh.multiblock_cell_node_csr_offsets.data(),
      state.mesh.multiblock_cell_node_csr_indices.data(),
      d_cell_nverts,
      state.x_r.data(),
      state.x_z.data(),
      state.v_r.data(),
      state.v_z.data(),
      state.ale_monitor_ref_x_r.data(),
      state.ale_monitor_ref_x_z.data(),
      dt_acoustic_scale,
      q_min_ordered,
      h_min_ordered,
      d_q_min_cell,
      d_h_min_cell);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&q_min_cell,
                        d_q_min_cell,
                        sizeof(q_min_cell),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&h_min_cell,
                        d_h_min_cell,
                        sizeof(h_min_cell),
                        cudaMemcpyDeviceToHost));

  result.q_min = ordered_bits_double_host(q_min_ordered);
  result.h_min = ordered_bits_double_host(h_min_ordered);
  result.q_min_cell = q_min_cell == std::numeric_limits<int>::max()
                          ? -1
                          : q_min_cell;
  result.h_min_cell = h_min_cell == std::numeric_limits<int>::max()
                          ? -1
                          : h_min_cell;
  result.evaluated_cells = evaluated_cells;

  if (fold_diag_enabled()) {
    TENRYU_ASSERT(state.ale_monitor_ref_corner_fractions.size() ==
                      4U * static_cast<std::size_t>(n_cells),
                  "ALE fold diagnostic requires reference corner fractions");

    auto* d_r_z_max_ordered = static_cast<unsigned long long*>(
        core::device_scratch_acquire(
            "ale_state_monitor:fold:r_z_max_ordered",
            sizeof(unsigned long long)));
    auto* d_h_metric_max_ordered = static_cast<unsigned long long*>(
        core::device_scratch_acquire(
            "ale_state_monitor:fold:h_metric_max_ordered",
            sizeof(unsigned long long)));
    auto* d_r_z_max_cell = static_cast<int*>(core::device_scratch_acquire(
        "ale_state_monitor:fold:r_z_max_cell", sizeof(int)));
    auto* d_h_metric_max_cell = static_cast<int*>(
        core::device_scratch_acquire(
            "ale_state_monitor:fold:h_metric_max_cell", sizeof(int)));

    const double negative_inf =
        -std::numeric_limits<double>::infinity();
    unsigned long long r_z_max_ordered =
        ordered_double_bits_host(negative_inf);
    unsigned long long h_metric_max_ordered =
        ordered_double_bits_host(negative_inf);
    CUDA_CHECK(cudaMemcpy(d_r_z_max_ordered,
                          &r_z_max_ordered,
                          sizeof(r_z_max_ordered),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h_metric_max_ordered,
                          &h_metric_max_ordered,
                          sizeof(h_metric_max_ordered),
                          cudaMemcpyHostToDevice));

    fold_diag_max_kernel<<<blocks, threads>>>(
        n_cells,
        indexing.shell_cell_offset,
        indexing.ntheta,
        cfg.numerics.ale.runtime_controller.shell_rows,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts,
        state.x_r.data(),
        state.x_z.data(),
        state.v_r.data(),
        state.v_z.data(),
        state.ale_monitor_ref_corner_fractions.data(),
        d_r_z_max_ordered,
        d_h_metric_max_ordered);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&r_z_max_ordered,
                          d_r_z_max_ordered,
                          sizeof(r_z_max_ordered),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_metric_max_ordered,
                          d_h_metric_max_ordered,
                          sizeof(h_metric_max_ordered),
                          cudaMemcpyDeviceToHost));

    int r_z_max_cell = std::numeric_limits<int>::max();
    int h_metric_max_cell = std::numeric_limits<int>::max();
    CUDA_CHECK(cudaMemcpy(d_r_z_max_cell,
                          &r_z_max_cell,
                          sizeof(r_z_max_cell),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_h_metric_max_cell,
                          &h_metric_max_cell,
                          sizeof(h_metric_max_cell),
                          cudaMemcpyHostToDevice));

    const std::vector<int>& tracked_cell_ids =
        fold_diag_tracked_cell_ids();
    const int tracked_cell_count =
        static_cast<int>(tracked_cell_ids.size());
    int* d_tracked_cell_ids = nullptr;
    AleMonitorTrackedCellResult* d_tracked_values = nullptr;
    if (tracked_cell_count > 0) {
      d_tracked_cell_ids = static_cast<int*>(core::device_scratch_acquire(
          "ale_state_monitor:fold:tracked_cell_ids",
          tracked_cell_ids.size() * sizeof(int)));
      d_tracked_values = static_cast<AleMonitorTrackedCellResult*>(
          core::device_scratch_acquire(
              "ale_state_monitor:fold:tracked_values",
              tracked_cell_ids.size() *
                  sizeof(AleMonitorTrackedCellResult)));
      result.tracked_cells.resize(tracked_cell_ids.size());
      for (std::size_t i = 0; i < tracked_cell_ids.size(); ++i) {
        result.tracked_cells[i].cell = tracked_cell_ids[i];
      }
      CUDA_CHECK(cudaMemcpy(d_tracked_cell_ids,
                            tracked_cell_ids.data(),
                            tracked_cell_ids.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(d_tracked_values,
                            result.tracked_cells.data(),
                            result.tracked_cells.size() *
                                sizeof(AleMonitorTrackedCellResult),
                            cudaMemcpyHostToDevice));
    }

    fold_diag_argmax_kernel<<<blocks, threads>>>(
        n_cells,
        indexing.shell_cell_offset,
        indexing.ntheta,
        cfg.numerics.ale.runtime_controller.shell_rows,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts,
        state.x_r.data(),
        state.x_z.data(),
        state.v_r.data(),
        state.v_z.data(),
        state.ale_monitor_ref_x_r.data(),
        state.ale_monitor_ref_x_z.data(),
        state.ale_monitor_ref_corner_fractions.data(),
        dt_acoustic_scale,
        r_z_max_ordered,
        h_metric_max_ordered,
        d_r_z_max_cell,
        d_h_metric_max_cell,
        d_tracked_cell_ids,
        tracked_cell_count,
        d_tracked_values);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&r_z_max_cell,
                          d_r_z_max_cell,
                          sizeof(r_z_max_cell),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_metric_max_cell,
                          d_h_metric_max_cell,
                          sizeof(h_metric_max_cell),
                          cudaMemcpyDeviceToHost));
    if (tracked_cell_count > 0) {
      CUDA_CHECK(cudaMemcpy(result.tracked_cells.data(),
                            d_tracked_values,
                            result.tracked_cells.size() *
                                sizeof(AleMonitorTrackedCellResult),
                            cudaMemcpyDeviceToHost));
    }

    if (r_z_max_cell != std::numeric_limits<int>::max()) {
      result.r_z_max = ordered_bits_double_host(r_z_max_ordered);
      result.r_z_max_cell = r_z_max_cell;
    }
    if (h_metric_max_cell != std::numeric_limits<int>::max()) {
      result.h_metric_max = ordered_bits_double_host(h_metric_max_ordered);
      result.h_metric_max_cell = h_metric_max_cell;
    }

    if (result.r_z_max_cell >= 0 || result.h_metric_max_cell >= 0) {
      auto* d_selected_values = static_cast<CellFoldValues*>(
          core::device_scratch_acquire(
              "ale_state_monitor:fold:selected_values",
              2U * sizeof(CellFoldValues)));
      CellFoldValues selected_values[2];
      CUDA_CHECK(cudaMemcpy(d_selected_values,
                            selected_values,
                            sizeof(selected_values),
                            cudaMemcpyHostToDevice));
      fold_diag_selected_values_kernel<<<1, 2>>>(
          result.r_z_max_cell,
          result.h_metric_max_cell,
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts,
          state.x_r.data(),
          state.x_z.data(),
          state.v_r.data(),
          state.v_z.data(),
          state.ale_monitor_ref_corner_fractions.data(),
          d_selected_values);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(selected_values,
                            d_selected_values,
                            sizeof(selected_values),
                            cudaMemcpyDeviceToHost));
      if (result.r_z_max_cell >= 0) {
        result.r_z_max_u_theta = selected_values[0].u_theta;
        result.r_z_max_s_u_theta = selected_values[0].s_u_theta;
      }
      if (result.h_metric_max_cell >= 0) {
        result.h_metric_max_u_theta = selected_values[1].u_theta;
        result.h_metric_max_s_u_theta = selected_values[1].s_u_theta;
      }
    }
  }

  result.valid = true;
  return result;
}

}  // namespace tenryu::hydro
