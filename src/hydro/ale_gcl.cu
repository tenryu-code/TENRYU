#include "hydro/ale_gcl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "mesh/mesh.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::ale {
namespace {

constexpr int kInvariantCount = 4;
constexpr int kLinfCount = 3;
constexpr int kMetricCount = kInvariantCount + kLinfCount;
constexpr double kRelativeReferenceFloor = 1.0e-300;
constexpr double kMomentumScaleRelFloor = 1.0e-12;

std::mutex g_reference_mutex;
std::unordered_map<const core::State*, AleVolClosureReference> g_references;

struct VolClosureLocalMetrics {
  double total_mass = 0.0;
  double total_internal_energy = 0.0;
  double total_r_momentum = 0.0;
  double total_z_momentum = 0.0;
  double Te_linf = 0.0;
  double Ti_linf = 0.0;
  double velocity_linf = 0.0;
};

__device__ inline double vol_closure_abs_delta(const double value,
                                               const double reference) {
  const double delta = fabs(value - reference);
  return isfinite(delta) ? delta : INFINITY;
}

__device__ inline double vol_closure_finite_or_infinity(const double value) {
  return isfinite(value) ? value : INFINITY;
}

__device__ inline double atomic_add_double(double* address, const double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        value + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline void atomic_max_double_positive(double* address, const double value) {
  if (!(value >= 0.0)) {
    return;
  }
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  while (__longlong_as_double(static_cast<long long>(old)) < value) {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (old == assumed) {
      break;
    }
  }
}

__global__ void vol_closure_metric_kernel(double* __restrict__ metrics,
                                  const double* __restrict__ rho,
                                  const double* __restrict__ Te,
                                  const double* __restrict__ Ti,
                                  const double* __restrict__ ee,
                                  const double* __restrict__ ei,
                                  const double* __restrict__ vol,
                                  const double* __restrict__ v_r,
                                  const double* __restrict__ v_z,
                                  const int n_cells,
                                  const int nz,
                                  const double Te_ref,
                                  const double Ti_ref,
                                  const double v_r_ref,
                                  const double v_z_ref) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_cells) {
    const int i = idx / nz;
    const int j = idx - i * nz;
    const int stride = nz + 1;
    const int n00 = i * stride + j;
    const int n10 = (i + 1) * stride + j;
    const int n11 = (i + 1) * stride + (j + 1);
    const int n01 = i * stride + (j + 1);
    const double v_r_cell = 0.25 * (v_r[n00] + v_r[n10] + v_r[n11] + v_r[n01]);
    const double v_z_cell = 0.25 * (v_z[n00] + v_z[n10] + v_z[n11] + v_z[n01]);
    const double mass = rho[idx] * vol[idx];
    const double internal_energy = mass * (ee[idx] + ei[idx]);
    const double r_momentum = mass * v_r_cell;
    const double z_momentum = mass * v_z_cell;
    atomic_add_double(metrics + 0, vol_closure_finite_or_infinity(mass));
    atomic_add_double(metrics + 1,
                      vol_closure_finite_or_infinity(internal_energy));
    atomic_add_double(metrics + 2, vol_closure_finite_or_infinity(r_momentum));
    atomic_add_double(metrics + 3, vol_closure_finite_or_infinity(z_momentum));
    atomic_max_double_positive(metrics + 4,
                               vol_closure_abs_delta(Te[idx], Te_ref));
    atomic_max_double_positive(metrics + 5,
                               vol_closure_abs_delta(Ti[idx], Ti_ref));
    const double dv_r = v_r_cell - v_r_ref;
    const double dv_z = v_z_cell - v_z_ref;
    const double dv = sqrt(dv_r * dv_r + dv_z * dv_z);
    atomic_max_double_positive(metrics + 6, isfinite(dv) ? dv : INFINITY);
  }
}

__global__ void vol_closure_metric_kernel_multiblock(
    double* __restrict__ metrics,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ vol,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int corner_stride,
    const int n_cells,
    const double Te_ref,
    const double Ti_ref,
    const double v_r_ref,
    const double v_z_ref) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n_cells) {
    const int off = cell_node_csr_offsets[idx];
    double v_r_cell = 0.0;
    double v_z_cell = 0.0;
    int active_nverts = 0;
    if (cell_nverts != nullptr) {
      active_nverts =
          tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, idx);
    } else {
      for (int k = 0; k < corner_stride; ++k) {
        if (cell_node_csr_indices[off + k] >= 0) {
          ++active_nverts;
        }
      }
    }
    if (active_nverts == tenryu::mesh::kMeshTopoCellStorageSlots) {
      const int n00 = cell_node_csr_indices[off + 0];
      const int n10 = cell_node_csr_indices[off + 1];
      const int n11 = cell_node_csr_indices[off + 2];
      const int n01 = cell_node_csr_indices[off + 3];
      v_r_cell = 0.25 * (v_r[n00] + v_r[n10] + v_r[n11] + v_r[n01]);
      v_z_cell = 0.25 * (v_z[n00] + v_z[n10] + v_z[n11] + v_z[n01]);
    } else {
      double v_r_sum = 0.0;
      double v_z_sum = 0.0;
      const int slot_count =
          cell_nverts != nullptr ? active_nverts : corner_stride;
      for (int k = 0; k < slot_count; ++k) {
        const int node = cell_node_csr_indices[off + k];
        if (node < 0) {
          continue;
        }
        v_r_sum += v_r[node];
        v_z_sum += v_z[node];
      }
      const double inv_nverts = 1.0 / static_cast<double>(active_nverts);
      v_r_cell = inv_nverts * v_r_sum;
      v_z_cell = inv_nverts * v_z_sum;
    }
    const double mass = rho[idx] * vol[idx];
    const double internal_energy = mass * (ee[idx] + ei[idx]);
    const double r_momentum = mass * v_r_cell;
    const double z_momentum = mass * v_z_cell;
    atomic_add_double(metrics + 0, vol_closure_finite_or_infinity(mass));
    atomic_add_double(metrics + 1,
                      vol_closure_finite_or_infinity(internal_energy));
    atomic_add_double(metrics + 2, vol_closure_finite_or_infinity(r_momentum));
    atomic_add_double(metrics + 3, vol_closure_finite_or_infinity(z_momentum));
    atomic_max_double_positive(metrics + 4,
                               vol_closure_abs_delta(Te[idx], Te_ref));
    atomic_max_double_positive(metrics + 5,
                               vol_closure_abs_delta(Ti[idx], Ti_ref));
    const double dv_r = v_r_cell - v_r_ref;
    const double dv_z = v_z_cell - v_z_ref;
    const double dv = sqrt(dv_r * dv_r + dv_z * dv_z);
    atomic_max_double_positive(metrics + 6, isfinite(dv) ? dv : INFINITY);
  }
}

double relative_drift(const double value,
                      const double reference,
                      const double scale_floor) {
  const double value_abs = std::abs(value);
  const double reference_abs = std::abs(reference);
  const double max_scale = value_abs > reference_abs ? value_abs : reference_abs;
  if (max_scale <= scale_floor) {
    return 0.0;
  }
  const double delta = std::abs(value - reference);
  const double denom = reference_abs;
  return denom > kRelativeReferenceFloor ? delta / denom : delta;
}

VolClosureLocalMetrics compute_local_metrics(const core::State& state,
                                             const double Te_ref,
                                             const double Ti_ref,
                                             const double v_r_ref,
                                             const double v_z_ref) {
  const int n_cells = state.mesh.topo.n_cells;
  VolClosureLocalMetrics out;
  if (n_cells <= 0) {
    return out;
  }

  double* d_metrics = nullptr;
  d_metrics = static_cast<double*>(
      core::device_scratch_acquire("ale_vol_closure:local_metrics:d_metrics",
                                   kMetricCount * sizeof(double)));
  CUDA_CHECK(cudaMemset(d_metrics, 0, kMetricCount * sizeof(double)));
  std::uint8_t* d_cell_nverts = nullptr;

  const int blocks = (n_cells + 255) / 256;
  if (state.mesh.topo.multiblock.has_value()) {
    const int corner_stride = state.mesh.corner_stride;
    TENRYU_ASSERT(corner_stride == 4 || corner_stride == 8,
                  "ALE volume-closure audit requires multiblock stride 4 or 8");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "ALE volume-closure audit requires multiblock cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(corner_stride),
                  "ALE volume-closure audit requires multiblock cell-node CSR indices");
    const auto& cell_nverts = state.mesh.cell_nverts;
    const bool use_cell_nverts =
        cell_nverts.size() == static_cast<std::size_t>(n_cells);
    if (use_cell_nverts) {
      TENRYU_ASSERT(
          std::all_of(cell_nverts.begin(), cell_nverts.end(),
                      [corner_stride](const std::uint8_t nverts) {
                        return nverts >= 3U &&
                               static_cast<int>(nverts) <= corner_stride;
                      }),
          "ALE volume-closure audit cell vertex count out of range");
    }
    if (use_cell_nverts) {
      d_cell_nverts = static_cast<std::uint8_t*>(
          core::device_scratch_acquire(
              "ale_vol_closure:local_metrics:d_cell_nverts",
                                       cell_nverts.size() * sizeof(std::uint8_t)));
      CUDA_CHECK(cudaMemcpy(d_cell_nverts,
                            cell_nverts.data(),
                            cell_nverts.size() * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice));
    }
    vol_closure_metric_kernel_multiblock<<<blocks, 256>>>(
        d_metrics,
        state.rho.data(),
        state.Te.data(),
        state.Ti.data(),
        state.ee.data(),
        state.ei.data(),
        state.vol.data(),
        state.v_r.data(),
        state.v_z.data(),
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts,
        corner_stride,
        n_cells,
        Te_ref,
        Ti_ref,
        v_r_ref,
        v_z_ref);
  } else {
    vol_closure_metric_kernel<<<blocks, 256>>>(d_metrics,
                                       state.rho.data(),
                                       state.Te.data(),
                                       state.Ti.data(),
                                       state.ee.data(),
                                       state.ei.data(),
                                       state.vol.data(),
                                       state.v_r.data(),
                                       state.v_z.data(),
                                       n_cells,
                                       state.mesh.topo.nz,
                                       Te_ref,
                                       Ti_ref,
                                       v_r_ref,
                                       v_z_ref);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  double metrics[kMetricCount] = {};
  CUDA_CHECK(cudaMemcpy(metrics,
                        d_metrics,
                        kMetricCount * sizeof(double),
                        cudaMemcpyDeviceToHost));

  out.total_mass = metrics[0];
  out.total_internal_energy = metrics[1];
  out.total_r_momentum = metrics[2];
  out.total_z_momentum = metrics[3];
  out.Te_linf = metrics[4];
  out.Ti_linf = metrics[5];
  out.velocity_linf = metrics[6];
  return out;
}

std::string format_scientific(const double value) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(6) << value;
  return os.str();
}

AleVolClosureReference capture_reference_unlocked(const core::State& state) {
  return capture_ale_vol_closure_reference(state);
}

}  // namespace

AleVolClosureReference capture_ale_vol_closure_reference(
    const core::State& state) {
  AleVolClosureReference reference;
  reference.n_cells = state.mesh.topo.n_cells;
  reference.n_nodes = state.mesh.topo.n_nodes;
  reference.valid = reference.n_cells > 0 && reference.n_nodes > 0;
  if (!reference.valid) {
    return reference;
  }

  TENRYU_ASSERT(state.mesh.dim == 2,
                "ALE volume-closure reference requires a 2D mesh");
  TENRYU_ASSERT(static_cast<int>(state.rho.size()) >= reference.n_cells,
                "ALE volume-closure reference requires rho size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.Te.size()) >= reference.n_cells,
                "ALE volume-closure reference requires Te size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.Ti.size()) >= reference.n_cells,
                "ALE volume-closure reference requires Ti size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.ee.size()) >= reference.n_cells,
                "ALE volume-closure reference requires ee size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.ei.size()) >= reference.n_cells,
                "ALE volume-closure reference requires ei size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.vol.size()) >= reference.n_cells,
                "ALE volume-closure reference requires vol size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.v_r.size()) >= reference.n_nodes,
                "ALE volume-closure reference requires v_r size to cover mesh nodes");
  TENRYU_ASSERT(static_cast<int>(state.v_z.size()) >= reference.n_nodes,
                "ALE volume-closure reference requires v_z size to cover mesh nodes");

  CUDA_CHECK(cudaMemcpy(&reference.Te, state.Te.data(), sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&reference.Ti, state.Ti.data(), sizeof(double),
                        cudaMemcpyDeviceToHost));
  int node_ids[8] = {};
  int active_nverts = mesh::kMeshTopoCellStorageSlots;
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    const int corner_stride = state.mesh.corner_stride;
    TENRYU_ASSERT(corner_stride == 4 || corner_stride == 8,
                  "ALE volume-closure reference requires multiblock stride 4 or 8");
    TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(reference.n_cells) + 1U,
                  "ALE volume-closure reference requires multiblock cell-node CSR offsets");
    TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(reference.n_cells) *
                          static_cast<std::size_t>(corner_stride),
                  "ALE volume-closure reference requires multiblock cell-node CSR indices");
    for (int c = 0; c < reference.n_cells; ++c) {
      const int row_begin =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int row_end =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
      TENRYU_ASSERT(row_end - row_begin == corner_stride,
                    "ALE volume-closure reference CSR width must match corner stride");
    }
    const int off = mb.cell_node_csr_offsets[0];
    if (state.mesh.cell_nverts.size() ==
        static_cast<std::size_t>(reference.n_cells)) {
      const int stored_nverts =
          static_cast<int>(state.mesh.cell_nverts[0]);
      TENRYU_ASSERT(stored_nverts >= 3 && stored_nverts <= corner_stride,
                    "ALE volume-closure reference cell vertex count out of range");
      active_nverts =
          mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, 0);
      for (int k = 0; k < active_nverts; ++k) {
        node_ids[k] =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        TENRYU_ASSERT(node_ids[k] >= 0 && node_ids[k] < reference.n_nodes,
                      "ALE volume-closure reference cell-node CSR index out of range");
      }
    } else {
      active_nverts = 0;
      for (int k = 0; k < corner_stride; ++k) {
        const int node =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        if (node >= 0) {
          TENRYU_ASSERT(node < reference.n_nodes,
                        "ALE volume-closure reference cell-node CSR index out of range");
          node_ids[active_nverts++] = node;
        }
      }
      TENRYU_ASSERT(active_nverts >= 3 && active_nverts <= corner_stride,
                    "ALE volume-closure reference cell vertex count out of range");
    }
  } else {
    const int stride = state.mesh.topo.nz + 1;
    node_ids[0] = 0;
    node_ids[1] = stride;
    node_ids[2] = stride + 1;
    node_ids[3] = 1;
  }
  double v_r_nodes[8] = {};
  double v_z_nodes[8] = {};
  for (int k = 0; k < active_nverts; ++k) {
    CUDA_CHECK(cudaMemcpy(v_r_nodes + k,
                          state.v_r.data() + node_ids[k],
                          sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_z_nodes + k,
                          state.v_z.data() + node_ids[k],
                          sizeof(double),
                          cudaMemcpyDeviceToHost));
  }
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    reference.v_r =
        0.25 * (v_r_nodes[0] + v_r_nodes[1] + v_r_nodes[2] + v_r_nodes[3]);
    reference.v_z =
        0.25 * (v_z_nodes[0] + v_z_nodes[1] + v_z_nodes[2] + v_z_nodes[3]);
  } else {
    double v_r_sum = 0.0;
    double v_z_sum = 0.0;
    for (int k = 0; k < active_nverts; ++k) {
      v_r_sum += v_r_nodes[k];
      v_z_sum += v_z_nodes[k];
    }
    const double inv_nverts = 1.0 / static_cast<double>(active_nverts);
    reference.v_r = inv_nverts * v_r_sum;
    reference.v_z = inv_nverts * v_z_sum;
  }
  const VolClosureLocalMetrics metrics = compute_local_metrics(
      state, reference.Te, reference.Ti, reference.v_r, reference.v_z);
  reference.total_mass = metrics.total_mass;
  reference.total_internal_energy = metrics.total_internal_energy;
  reference.total_r_momentum = metrics.total_r_momentum;
  reference.total_z_momentum = metrics.total_z_momentum;
  return reference;
}

AleVolClosureResidual compute_ale_vol_closure_residual(
    const core::State& state,
    const AleVolClosureReference& reference,
    const parallel::Reduction* reduction) {
  AleVolClosureResidual out;
  if (!reference.valid) {
    return out;
  }

  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(state.mesh.dim == 2,
                "ALE volume-closure residual requires a 2D mesh");
  TENRYU_ASSERT(n_cells == reference.n_cells,
                "ALE volume-closure residual requires unchanged cell count");
  TENRYU_ASSERT(n_nodes == reference.n_nodes,
                "ALE volume-closure residual requires unchanged node count");
  TENRYU_ASSERT(static_cast<int>(state.rho.size()) >= n_cells,
                "ALE volume-closure residual requires rho size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.Te.size()) >= n_cells,
                "ALE volume-closure residual requires Te size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.Ti.size()) >= n_cells,
                "ALE volume-closure residual requires Ti size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.ee.size()) >= n_cells,
                "ALE volume-closure residual requires ee size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.ei.size()) >= n_cells,
                "ALE volume-closure residual requires ei size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.vol.size()) >= n_cells,
                "ALE volume-closure residual requires vol size to cover mesh cells");
  TENRYU_ASSERT(static_cast<int>(state.v_r.size()) >= n_nodes,
                "ALE volume-closure residual requires v_r size to cover mesh nodes");
  TENRYU_ASSERT(static_cast<int>(state.v_z.size()) >= n_nodes,
                "ALE volume-closure residual requires v_z size to cover mesh nodes");

  VolClosureLocalMetrics metrics = compute_local_metrics(
      state, reference.Te, reference.Ti, reference.v_r, reference.v_z);
  double current[kInvariantCount] = {metrics.total_mass,
                                     metrics.total_internal_energy,
                                     metrics.total_r_momentum,
                                     metrics.total_z_momentum};
  double initial[kInvariantCount] = {reference.total_mass,
                                     reference.total_internal_energy,
                                     reference.total_r_momentum,
                                     reference.total_z_momentum};
  if (reduction != nullptr) {
    double totals[2 * kInvariantCount] = {current[0],
                                          current[1],
                                          current[2],
                                          current[3],
                                          initial[0],
                                          initial[1],
                                          initial[2],
                                          initial[3]};
    reduction->allreduce_sum(totals, 2 * kInvariantCount);
    for (int k = 0; k < kInvariantCount; ++k) {
      current[k] = totals[k];
      initial[k] = totals[k + kInvariantCount];
    }
    double linf[kLinfCount] = {metrics.Te_linf, metrics.Ti_linf, metrics.velocity_linf};
    reduction->allreduce_max(linf, kLinfCount);
    metrics.Te_linf = linf[0];
    metrics.Ti_linf = linf[1];
    metrics.velocity_linf = linf[2];
  }

  const double total_mass = initial[0];
  const double thermal_mass =
      total_mass > kRelativeReferenceFloor ? total_mass : kRelativeReferenceFloor;
  const double v_thermal = std::sqrt(2.0 * initial[1] / thermal_mass);
  const double momentum_scale_floor =
      kMomentumScaleRelFloor * total_mass * v_thermal;
  out.total_mass_drift_rel = relative_drift(current[0], initial[0], 0.0);
  out.total_internal_energy_drift_rel =
      relative_drift(current[1], initial[1], 0.0);
  out.total_r_momentum_drift_rel =
      relative_drift(current[2], initial[2], momentum_scale_floor);
  out.total_z_momentum_drift_rel =
      relative_drift(current[3], initial[3], momentum_scale_floor);
  out.Te_linf = metrics.Te_linf;
  out.Ti_linf = metrics.Ti_linf;
  out.velocity_linf = metrics.velocity_linf;
  out.valid = true;
  return out;
}

void capture_ale_vol_closure_reference_if_needed(const core::State& state) {
  std::lock_guard<std::mutex> lock(g_reference_mutex);
  const auto it = g_references.find(&state);
  if (it != g_references.end() &&
      it->second.n_cells == state.mesh.topo.n_cells &&
      it->second.n_nodes == state.mesh.topo.n_nodes) {
    return;
  }
  g_references[&state] = capture_reference_unlocked(state);
}

AleVolClosureResidual log_ale_vol_closure_residual(
    const core::State& state,
    const parallel::Reduction* reduction) {
  AleVolClosureReference reference;
  {
    std::lock_guard<std::mutex> lock(g_reference_mutex);
    auto it = g_references.find(&state);
    if (it == g_references.end() ||
        it->second.n_cells != state.mesh.topo.n_cells ||
        it->second.n_nodes != state.mesh.topo.n_nodes) {
      it = g_references.insert_or_assign(&state, capture_reference_unlocked(state)).first;
    }
    reference = it->second;
  }

  const AleVolClosureResidual residual =
      compute_ale_vol_closure_residual(state, reference, reduction);
  if (residual.valid) {
    core::log_info("[ale-vol-closure] step=" + std::to_string(state.step) +
                   " total_mass_drift_rel=" +
                       format_scientific(residual.total_mass_drift_rel) +
                   " total_internal_energy_drift_rel=" +
                       format_scientific(residual.total_internal_energy_drift_rel) +
                   " total_r_momentum_drift_rel=" +
                       format_scientific(residual.total_r_momentum_drift_rel) +
                   " total_z_momentum_drift_rel=" +
                       format_scientific(residual.total_z_momentum_drift_rel) +
                   " Te_linf=" + format_scientific(residual.Te_linf) +
                   " Ti_linf=" + format_scientific(residual.Ti_linf) +
                   " velocity_linf=" + format_scientific(residual.velocity_linf) +
                   " max_total_drift_rel=" +
                       format_scientific(residual.max_total_drift_rel()) +
                   " max_linf=" + format_scientific(residual.max_linf()));
  }
  return residual;
}

void reset_ale_vol_closure_monitor() {
  std::lock_guard<std::mutex> lock(g_reference_mutex);
  g_references.clear();
}

}  // namespace tenryu::hydro::ale
