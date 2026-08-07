#include "diagnostics/energy_budget.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "hydro/rz_corner_mass.cuh"

namespace tenryu::diagnostics {
namespace {

constexpr int kBlockSize = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

int button_outer_node_ring_or_zero(const core::State& state) {
  return (state.mesh.button_center && state.mesh.button_center->enabled)
             ? state.mesh.button_center->outer_node_ring
             : 0;
}

std::uint8_t* upload_cell_nverts_if_special_2d(const core::State& state) {
  const auto& cell_nverts = state.mesh.cell_nverts;
  if (cell_nverts.size() != state.mass.size()) {
    return nullptr;
  }
  const bool has_special =
      button_outer_node_ring_or_zero(state) > 0 ||
      std::any_of(cell_nverts.begin(), cell_nverts.end(),
                  [](const std::uint8_t nverts) {
                    return nverts != 4U;
                  });
  if (!has_special) {
    return nullptr;
  }
  std::uint8_t* d_cell_nverts = static_cast<std::uint8_t*>(
      core::device_scratch_acquire(
          "energy_budget:2d_cell_nverts",
          cell_nverts.size() * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_nverts,
                        cell_nverts.data(),
                        cell_nverts.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "compute_energy_totals_2d cudaMemcpy cell_nverts failed");
  return d_cell_nverts;
}

__device__ inline void compute_energy_contrib_1d_kernel_body(
    const int c,
    double* __restrict__ contrib_int_e,
    double* __restrict__ contrib_int_i,
    double* __restrict__ contrib_kin,
    double* __restrict__ contrib_kin_nodal,
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ vol,
    const double* __restrict__ mass,
    const double* __restrict__ v_r,
    const int n_cells) {
  contrib_int_e[c] = rho[c] * ee[c] * vol[c];
  contrib_int_i[c] = rho[c] * ei[c] * vol[c];

  const double u = 0.5 * (v_r[c] + v_r[c + 1]);
  contrib_kin[c] = 0.5 * mass[c] * u * u;
  // W-J-2: nodal-form kinetic (half cell mass to each node). The averaged
  // form above under-measures kinetic energy where v varies across the cell.
  contrib_kin_nodal[c] =
      0.25 * mass[c] * (v_r[c] * v_r[c] + v_r[c + 1] * v_r[c + 1]);
}

__global__ void compute_energy_contrib_1d_kernel(
    double* __restrict__ contrib_int_e,
    double* __restrict__ contrib_int_i,
    double* __restrict__ contrib_kin,
    double* __restrict__ contrib_kin_nodal,
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ vol,
    const double* __restrict__ mass,
    const double* __restrict__ v_r,
    const int c_begin,
    const int c_end,
    const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  (void)n_cells;
  if (c >= c_end) {
    return;
  }

  compute_energy_contrib_1d_kernel_body(c, contrib_int_e, contrib_int_i,
                                        contrib_kin, contrib_kin_nodal, rho,
                                        ee, ei, vol, mass, v_r, n_cells);
}

__global__ void compute_energy_contrib_2d_kernel(
    double* __restrict__ contrib_int_e,
    double* __restrict__ contrib_int_i,
    double* __restrict__ contrib_kin,
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ vol,
    const double* __restrict__ mass,
    const double* __restrict__ cached_corner_mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int nr,
    const int nz,
    const int button_outer_node_ring,
    const int c_begin,
    const int c_end,
    const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  contrib_int_e[c] = rho[c] * ee[c] * vol[c];
  contrib_int_i[c] = rho[c] * ei[c] * vol[c];

  if (button_outer_node_ring > 0 && c == 0) {
    const double volume = hydro::rz::button_polygon_volume_from_nodes(
        x_r, x_z, button_outer_node_ring, nz);
    if (!(volume > 0.0) || !isfinite(volume)) {
      contrib_kin[c] = 0.0;
      return;
    }
    double centroid_r = 0.0;
    double centroid_z = 0.0;
    hydro::rz::button_polygon_area_centroid_from_nodes(
        x_r, x_z, button_outer_node_ring, nz, &centroid_r, &centroid_z);
    const double rho_button = fmax(mass[c], 0.0) / volume;
    double kinetic = 0.0;
    for (int k = 0; k <= nz; ++k) {
      const int n = hydro::rz::button_seam_node_index(
          button_outer_node_ring, k, nz);
      const double m_node = fmax(hydro::rz::button_corner_mass_exact_subpolygon(
          rho_button, x_r, x_z, button_outer_node_ring, nz, k,
          centroid_r, centroid_z), 0.0);
      kinetic += m_node * (v_r[n] * v_r[n] + v_z[n] * v_z[n]);
    }
    contrib_kin[c] = 0.5 * kinetic;
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n01 = i * stride + (j + 1);
  const int n11 = (i + 1) * stride + (j + 1);
  const double m_cell = fmax(mass[c], 0.0);
  double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
  if (cached_corner_mass != nullptr) {
    // The diagnostic KE basis must match the DYNAMICAL nodal-mass basis
    // (the cached Lagrangian-invariant corner masses); recomputing from
    // the current geometry repartitions the nodal KE as the mesh deforms.
    const int base = 4 * c;
    m_corner[0] = fmax(cached_corner_mass[base + 0], 0.0);
    m_corner[1] = fmax(cached_corner_mass[base + 1], 0.0);
    m_corner[2] = fmax(cached_corner_mass[base + 2], 0.0);
    m_corner[3] = fmax(cached_corner_mass[base + 3], 0.0);
  } else {
    hydro::rz::compute_rz_corner_masses_from_nodes(c, nz, m_cell, x_r, x_z,
                                                   cell_nverts, m_corner);
  }
  const double m00 = m_corner[0];
  const double m10 = m_corner[1];
  const double m11 = m_corner[2];
  const double m01 = m_corner[3];

  const double vr00 = v_r[n00];
  const double vz00 = v_z[n00];
  const double vr10 = v_r[n10];
  const double vz10 = v_z[n10];
  const double vr11 = v_r[n11];
  const double vz11 = v_z[n11];
  const double vr01 = v_r[n01];
  const double vz01 = v_z[n01];
  contrib_kin[c] = 0.5 * (m00 * (vr00 * vr00 + vz00 * vz00) +
                          m10 * (vr10 * vr10 + vz10 * vz10) +
                          m11 * (vr11 * vr11 + vz11 * vz11) +
                          m01 * (vr01 * vr01 + vz01 * vz01));
}

__global__ void compute_energy_contrib_2d_csr_corner_mass_kernel(
    double* __restrict__ contrib_int_e,
    double* __restrict__ contrib_int_i,
    double* __restrict__ contrib_kin,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ mass,
    const double* __restrict__ corner_mass,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int n_nodes,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const int corner_stride) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  const double m = fmax(mass[c], 0.0);
  contrib_int_e[c] = m * ee[c];
  contrib_int_i[c] = m * ei[c];

  const int base = corner_stride * c;
  const int off = cell_node_csr_offsets[c];
  const int end = cell_node_csr_offsets[c + 1];
  int active_nverts = (cell_nverts != nullptr) ? static_cast<int>(cell_nverts[c])
                                               : 4;
  const int available_nverts = (end > off) ? (end - off) : 0;
  if (active_nverts < 0 || active_nverts > available_nverts) {
    active_nverts = available_nverts;
  }
  if (active_nverts > corner_stride) {
    active_nverts = corner_stride;
  }

  double kinetic = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    if (n < 0 || n >= n_nodes) {
      continue;
    }
    const double cm = fmax(corner_mass[base + k], 0.0);
    const double vr = v_r[n];
    const double vz = v_z[n];
    if (cm > 0.0 && isfinite(cm) && isfinite(vr) && isfinite(vz)) {
      kinetic += 0.5 * cm * (vr * vr + vz * vz);
    }
  }
  contrib_kin[c] = isfinite(kinetic) ? kinetic : 0.0;
}

__device__ inline void reduce_sum_block_kernel_body(
    const int block_idx,
    const int tid,
    double* s,
    const double* __restrict__ in,
    double* __restrict__ block_sums,
    const int n) {
  const int base = 2 * block_idx * blockDim.x;
  const int i0 = base + tid;
  const int i1 = i0 + blockDim.x;

  double value = 0.0;
  if (i0 < n) {
    value += in[i0];
  }
  if (i1 < n) {
    value += in[i1];
  }
  s[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s[tid] += s[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[block_idx] = s[0];
  }
}

__global__ void reduce_sum_block_kernel(const double* __restrict__ in,
                                        double* __restrict__ block_sums,
                                        const int n) {
  __shared__ double s[kBlockSize];
  const int tid = threadIdx.x;
  const int block_idx = blockIdx.x;
  reduce_sum_block_kernel_body(block_idx, tid, s, in, block_sums, n);
}

double reduce_device_sum(const double* d_contrib,
                         const int n,
                         const char* malloc_msg,
                         const char* launch_msg,
                         const char* exec_msg,
                         const char* memcpy_msg,
                         const char* free_msg) {
  if (n <= 0) {
    return 0.0;
  }

  const int blocks = (n + 2 * kBlockSize - 1) / (2 * kBlockSize);
  double* d_block_sums = static_cast<double*>(core::device_scratch_acquire(
      "energy_budget:reduce_block_sums",
      static_cast<std::size_t>(blocks) * sizeof(double)));

  reduce_sum_block_kernel<<<blocks, kBlockSize>>>(d_contrib, d_block_sums, n);
  cuda_check(cudaGetLastError(), launch_msg);
  cuda_check(core::debug_kernel_sync(), exec_msg);

  std::vector<double> host_block_sums(static_cast<std::size_t>(blocks), 0.0);
  cuda_check(cudaMemcpy(host_block_sums.data(),
                        d_block_sums,
                        static_cast<std::size_t>(blocks) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             memcpy_msg);

  long double total = 0.0L;
  for (const double value : host_block_sums) {
    total += static_cast<long double>(value);
  }
  return static_cast<double>(total);
}

// Batched variant: reduces k independent contribution arrays (each length n) with ONE
// device synchronization and ONE D2H transfer. Per-quantity block sums are accumulated
// on the host in long double in ascending block order — bit-identical to k separate
// reduce_device_sum calls (same kernel, same launch geometry, same summation order).
void reduce_device_sums_batched(const double* const* d_contribs,
                                const int k,
                                const int n,
                                double* out,
                                const char* launch_msg,
                                const char* exec_msg,
                                const char* memcpy_msg) {
  if (n <= 0 || k <= 0) {
    for (int q = 0; q < k; ++q) {
      out[q] = 0.0;
    }
    return;
  }

  const int blocks = (n + 2 * kBlockSize - 1) / (2 * kBlockSize);
  double* d_block_sums = static_cast<double*>(core::device_scratch_acquire(
      "energy_budget:reduce_block_sums_batched",
      static_cast<std::size_t>(k) * static_cast<std::size_t>(blocks) * sizeof(double)));

  for (int q = 0; q < k; ++q) {
    reduce_sum_block_kernel<<<blocks, kBlockSize>>>(
        d_contribs[q], d_block_sums + static_cast<std::size_t>(q) * blocks, n);
    cuda_check(cudaGetLastError(), launch_msg);
  }
  cuda_check(core::debug_kernel_sync(), exec_msg);

  double* h_block_sums = static_cast<double*>(core::host_pinned_scratch_acquire(
      "energy_budget:reduce_block_sums_batched_host",
      static_cast<std::size_t>(k) * static_cast<std::size_t>(blocks) *
          sizeof(double)));
  cuda_check(cudaMemcpy(h_block_sums,
                        d_block_sums,
                        static_cast<std::size_t>(k) * static_cast<std::size_t>(blocks) *
                            sizeof(double),
                        cudaMemcpyDeviceToHost),
             memcpy_msg);

  for (int q = 0; q < k; ++q) {
    long double total = 0.0L;
    const double* seg = h_block_sums + static_cast<std::size_t>(q) * blocks;
    for (int b = 0; b < blocks; ++b) {
      total += static_cast<long double>(seg[b]);
    }
    out[q] = static_cast<double>(total);
  }
}

EnergyTotals reduce_totals_from_contrib(const int n_cells,
                                        double* d_int_e,
                                        double* d_int_i,
                                        double* d_kin,
                                        const char* label) {
  EnergyTotals out{};
  out.E_int_e = reduce_device_sum(
      d_int_e,
      n_cells,
      "energy totals reduce int_e malloc failed",
      "energy totals reduce int_e launch failed",
      "energy totals reduce int_e execution failed",
      "energy totals reduce int_e memcpy failed",
      "energy totals reduce int_e free failed");
  out.E_int_i = reduce_device_sum(
      d_int_i,
      n_cells,
      "energy totals reduce int_i malloc failed",
      "energy totals reduce int_i launch failed",
      "energy totals reduce int_i execution failed",
      "energy totals reduce int_i memcpy failed",
      "energy totals reduce int_i free failed");
  out.E_kin = reduce_device_sum(
      d_kin,
      n_cells,
      "energy totals reduce kin malloc failed",
      "energy totals reduce kin launch failed",
      "energy totals reduce kin execution failed",
      "energy totals reduce kin memcpy failed",
      "energy totals reduce kin free failed");
  (void)label;
  return out;
}

}  // namespace

namespace {
// Shared contribution pass of compute_energy_totals_1d: validates the
// state, fills the pooled {int_e, int_i, kin, kin_nodal} contribution
// arrays and returns them via d_contribs_out. Returns false (without
// touching d_contribs_out) for an empty state.
bool compute_energy_totals_1d_contrib(const core::State& state,
                                      double* d_contribs_out[4]) {
  TENRYU_ASSERT(state.mesh.dim == 1, "1D energy totals requires 1D mesh");
  TENRYU_ASSERT(state.mass.size() == state.rho.size(),
                "Energy totals requires mass/rho size match");
  TENRYU_ASSERT(state.vol.size() == state.rho.size(),
                "Energy totals requires vol/rho size match");
  TENRYU_ASSERT(state.ee.size() == state.rho.size(),
                "Energy totals requires ee/rho size match");
  TENRYU_ASSERT(state.ei.size() == state.rho.size(),
                "Energy totals requires ei/rho size match");
  TENRYU_ASSERT(state.v_r.size() == state.rho.size() + 1,
                "Energy totals requires node count = cell count + 1");

  if (state.rho.empty()) {
    return false;
  }

  const int n_cells = static_cast<int>(state.rho.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const int blocks = cw.blocks(kBlockSize);

  double* d_int_e = nullptr;
  double* d_int_i = nullptr;
  double* d_kin = nullptr;
  double* d_kin_nodal = nullptr;

  d_int_e = static_cast<double*>(core::device_scratch_acquire(
      "energy_budget:1d_int_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_int_i = static_cast<double*>(core::device_scratch_acquire(
      "energy_budget:1d_int_i",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_kin = static_cast<double*>(core::device_scratch_acquire(
      "energy_budget:1d_kin",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_kin_nodal = static_cast<double*>(core::device_scratch_acquire(
      "energy_budget:1d_kin_nodal",
      static_cast<std::size_t>(n_cells) * sizeof(double)));

  compute_energy_contrib_1d_kernel<<<blocks, kBlockSize>>>(
      d_int_e,
      d_int_i,
      d_kin,
      d_kin_nodal,
      state.rho.data(),
      state.ee.data(),
      state.ei.data(),
      state.vol.data(),
      state.mass.data(),
      state.v_r.data(),
      cw.begin,
      cw.end,
      n_cells);
  cuda_check(cudaGetLastError(), "compute_energy_totals_1d kernel launch failed");
  cuda_check(core::debug_kernel_sync(),
             "compute_energy_totals_1d kernel execution failed");

  d_contribs_out[0] = d_int_e;
  d_contribs_out[1] = d_int_i;
  d_contribs_out[2] = d_kin;
  d_contribs_out[3] = d_kin_nodal;
  return true;
}
}  // namespace

EnergyTotals compute_energy_totals_1d(const core::State& state) {
  EnergyTotals totals{};
  double* d_contribs[4];
  if (!compute_energy_totals_1d_contrib(state, d_contribs)) {
    return totals;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  // Owned-window sums: reduce only the [cw.begin, cw.end) slice of each
  // contrib array (non-owned entries are never written under n_ranks > 1).
  const double* d_owned_contribs[4] = {
      d_contribs[0] + cw.begin, d_contribs[1] + cw.begin,
      d_contribs[2] + cw.begin, d_contribs[3] + cw.begin};
  double sums[4] = {0.0, 0.0, 0.0, 0.0};
  reduce_device_sums_batched(d_owned_contribs,
                             4,
                             cw.count(),
                             sums,
                             "energy totals 1d batched reduce launch failed",
                             "energy totals 1d batched reduce execution failed",
                             "energy totals 1d batched reduce memcpy failed");
  totals.E_int_e = sums[0];
  totals.E_int_i = sums[1];
  totals.E_kin = sums[2];
  totals.E_kin_nodal = sums[3];

  return totals;
}

namespace {
// Shared contribution pass of compute_energy_totals_2d: validates the
// state, fills the pooled {int_e, int_i, kin} contribution arrays and
// returns them via d_contribs_out. Returns false (without touching
// d_contribs_out) for an empty state.
bool compute_energy_totals_2d_contrib(const core::State& state,
                                      double* d_contribs_out[3]) {
  TENRYU_ASSERT(state.mesh.dim == 2, "2D energy totals requires 2D mesh");
  TENRYU_ASSERT(state.mass.size() == state.rho.size(),
                "Energy totals requires mass/rho size match");
  TENRYU_ASSERT(state.vol.size() == state.rho.size(),
                "Energy totals requires vol/rho size match");
  TENRYU_ASSERT(state.ee.size() == state.rho.size(),
                "Energy totals requires ee/rho size match");
  TENRYU_ASSERT(state.ei.size() == state.rho.size(),
                "Energy totals requires ei/rho size match");
  TENRYU_ASSERT(state.mesh.topo.n_cells == static_cast<int>(state.rho.size()),
                "Energy totals requires mesh cell count match");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == static_cast<int>(state.v_r.size()),
                "Energy totals requires mesh node count match");
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "Energy totals requires matching node radii");
  TENRYU_ASSERT(state.x_z.size() == state.v_r.size(),
                "Energy totals requires matching node coordinates");
  TENRYU_ASSERT(state.v_z.size() == state.v_r.size(),
                "Energy totals requires matching node velocities");

  if (state.rho.empty()) {
    return false;
  }

  const int n_cells = static_cast<int>(state.rho.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const int blocks = cw.blocks(kBlockSize);

  double* d_int_e = nullptr;
  double* d_int_i = nullptr;
  double* d_kin = nullptr;
  std::uint8_t* d_cell_nverts = upload_cell_nverts_if_special_2d(state);
  const int button_outer_node_ring = button_outer_node_ring_or_zero(state);

  d_int_e = static_cast<double*>(core::device_scratch_acquire(
      "energy_budget:2d_int_e",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_int_i = static_cast<double*>(core::device_scratch_acquire(
      "energy_budget:2d_int_i",
      static_cast<std::size_t>(n_cells) * sizeof(double)));
  d_kin = static_cast<double*>(core::device_scratch_acquire(
      "energy_budget:2d_kin",
      static_cast<std::size_t>(n_cells) * sizeof(double)));

  // The diagnostic KE basis must match the DYNAMICAL nodal-mass basis:
  // the compatible scheme uses the CACHED Lagrangian-invariant corner
  // masses, so whenever they exist the energy totals must use them too.
  // The legacy fallback recomputes corner masses from the CURRENT
  // geometry, whose partition drifts away from the dynamical basis as the
  // mesh deforms -- a fake energy non-conservation in the diagnostic
  // (isolated on the G4 5-block smoke: history E_total drifted -2.4e-6
  // while the cached-basis recomputation moved only +5e-8).
  const bool multiblock_csr = state.mesh.topo.multiblock.has_value();
  const bool optionb_ke_basis =
      multiblock_csr &&
      state.corner_mass_initialized &&
      state.corner_mass.size() ==
          static_cast<std::size_t>(n_cells) *
              static_cast<std::size_t>(state.corner_stride) &&
      state.mesh.multiblock_cell_node_csr_offsets.size() ==
          static_cast<std::size_t>(n_cells) + 1U &&
      state.mesh.multiblock_cell_node_csr_indices.size() ==
          static_cast<std::size_t>(n_cells) *
              static_cast<std::size_t>(state.corner_stride);
  // Pre-init belt checkpoints use the quad KE-summary fallback; mid-run corner masses select CSR.
  TENRYU_ASSERT(
      state.corner_stride == 4 || !state.corner_mass_initialized ||
          optionb_ke_basis,
      "energy budget: CSR corner path required for multiblock meshes "
      "(size/stride mismatch)");
  if (optionb_ke_basis) {
    compute_energy_contrib_2d_csr_corner_mass_kernel<<<blocks, kBlockSize>>>(
        d_int_e,
        d_int_i,
        d_kin,
        state.ee.data(),
        state.ei.data(),
        state.mass.data(),
        state.corner_mass.data(),
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts,
        state.v_r.data(),
        state.v_z.data(),
        state.mesh.topo.n_nodes,
        cw.begin,
        cw.end,
        n_cells,
        state.corner_stride);
  } else {
    const bool cached_basis =
        state.corner_mass_initialized &&
        state.corner_mass.size() == static_cast<std::size_t>(n_cells) * 4U;
    compute_energy_contrib_2d_kernel<<<blocks, kBlockSize>>>(
        d_int_e,
        d_int_i,
        d_kin,
        state.rho.data(),
        state.ee.data(),
        state.ei.data(),
        state.vol.data(),
        state.mass.data(),
        cached_basis ? state.corner_mass.data() : nullptr,
        state.x_r.data(),
        state.x_z.data(),
        d_cell_nverts,
        state.v_r.data(),
        state.v_z.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        button_outer_node_ring,
        cw.begin,
        cw.end,
        n_cells);
  }
  cuda_check(cudaGetLastError(), "compute_energy_totals_2d kernel launch failed");
  cuda_check(core::debug_kernel_sync(),
             "compute_energy_totals_2d kernel execution failed");

  d_contribs_out[0] = d_int_e;
  d_contribs_out[1] = d_int_i;
  d_contribs_out[2] = d_kin;
  return true;
}
}  // namespace

EnergyTotals compute_energy_totals_2d(const core::State& state) {
  EnergyTotals totals{};
  double* d_contribs[3];
  if (!compute_energy_totals_2d_contrib(state, d_contribs)) {
    totals.E_kin_nodal = totals.E_kin;
    return totals;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  totals = reduce_totals_from_contrib(cw.count(), d_contribs[0] + cw.begin,
                                      d_contribs[1] + cw.begin,
                                      d_contribs[2] + cw.begin, "2d");
  totals.E_kin_nodal = totals.E_kin;
  return totals;
}

int energy_reduce_blocks(const int n) {
  if (n <= 0) {
    return 0;
  }
  return (n + 2 * kBlockSize - 1) / (2 * kBlockSize);
}

void reduce_device_sums_to_slot(const double* const* d_contribs,
                                const int k,
                                const int n,
                                double* d_slot) {
  const int blocks = energy_reduce_blocks(n);
  if (blocks <= 0 || k <= 0) {
    return;
  }
  for (int q = 0; q < k; ++q) {
    reduce_sum_block_kernel<<<blocks, kBlockSize>>>(
        d_contribs[q], d_slot + static_cast<std::size_t>(q) * blocks, n);
    cuda_check(cudaGetLastError(), "energy totals slot reduce launch failed");
  }
  cuda_check(core::debug_kernel_sync(),
             "energy totals slot reduce execution failed");
}

void materialize_reduced_sums(const double* h_block_sums,
                              const int k,
                              const int blocks,
                              double* out) {
  for (int q = 0; q < k; ++q) {
    long double total = 0.0L;
    const double* seg = h_block_sums + static_cast<std::size_t>(q) * blocks;
    for (int b = 0; b < blocks; ++b) {
      total += static_cast<long double>(seg[b]);
    }
    out[q] = static_cast<double>(total);
  }
}

int compute_energy_totals_1d_to_slot(const core::State& state,
                                     double* d_slot) {
  double* d_contribs[4];
  if (!compute_energy_totals_1d_contrib(state, d_contribs)) {
    return 0;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  // Packet layout in blocks-sized segments: 0=int_e, 1=int_i, 2=kin,
  // 3=kin_nodal.
  reduce_device_sums_to_slot(d_contribs, 4, n_cells, d_slot);
  return energy_reduce_blocks(n_cells);
}

int compute_energy_totals_2d_to_slot(const core::State& state,
                                     double* d_slot) {
  double* d_contribs[3];
  if (!compute_energy_totals_2d_contrib(state, d_contribs)) {
    return 0;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const double* d_owned[3] = {d_contribs[0] + cw.begin,
                              d_contribs[1] + cw.begin,
                              d_contribs[2] + cw.begin};
  reduce_device_sums_to_slot(d_owned, 3, cw.count(), d_slot);
  return energy_reduce_blocks(cw.count());
}

EnergyTotals materialize_energy_totals_1d_from_packet(const double* h_slot,
                                                      const int blocks) {
  EnergyTotals totals{};
  double sums[4] = {0.0, 0.0, 0.0, 0.0};
  // Packet layout in blocks-sized segments: 0=int_e, 1=int_i, 2=kin,
  // 3=kin_nodal.
  materialize_reduced_sums(h_slot, 4, blocks, sums);
  totals.E_int_e = sums[0];
  totals.E_int_i = sums[1];
  totals.E_kin = sums[2];
  totals.E_kin_nodal = sums[3];
  return totals;
}

EnergyTotals materialize_energy_totals_2d(const double* h_slot,
                                          const int blocks) {
  EnergyTotals totals{};
  double sums[3] = {0.0, 0.0, 0.0};
  materialize_reduced_sums(h_slot, 3, blocks, sums);
  totals.E_int_e = sums[0];
  totals.E_int_i = sums[1];
  totals.E_kin = sums[2];
  totals.E_kin_nodal = totals.E_kin;
  return totals;
}

EnergyBudget compute_step_energy_budget(const EnergyBudgetStepInput& input) {
  EnergyBudget out{};
  out.E_int_e = input.after.E_int_e;
  out.E_int_i = input.after.E_int_i;
  out.E_kin = input.after.E_kin;
  out.E_rad = input.E_rad_after;
  out.E_rad_mesh_advection = input.E_rad_mesh_advection;
  out.E_laser_in = input.E_laser_in;
  out.E_Marshak_in = input.E_Marshak_in;
  out.E_volume_in = input.E_volume_in;
  out.E_burn_in = input.E_burn_in;
  out.E_laser_esc = input.E_laser_esc;
  out.E_cbet_iaw = input.E_cbet_iaw;
  out.E_rad_esc = input.E_rad_esc;
  out.E_numerical_loss = input.E_numerical_loss;
  out.E_pdV_bdry = input.E_pdV_bdry;
  out.E_floor = input.E_floor;
  out.E_safety = input.E_safety;
  out.E_redistribution_unresolved = input.E_redistribution_unresolved;
  // Keep signed solver residual for downstream diagnostics output (energy/E_solver).
  out.E_solver = input.E_solver;

  double before_E_int_e = input.before.E_int_e;
  double before_E_int_i = input.before.E_int_i;
  double before_E_kin = input.before.E_kin;
  double before_E_rad = input.E_rad_before;
  if (input.reduction != nullptr) {
    // Laser commanded/escaped/CBET-IAW energies are already global singleton
    // values in current laser plumbing, so only reduce rank-local budget terms here.
    std::array<double, 19> reduced = {
        out.E_int_e,          out.E_int_i,      out.E_kin,      out.E_rad,
        out.E_Marshak_in,     out.E_volume_in,  out.E_burn_in,  out.E_rad_esc,
        out.E_numerical_loss, out.E_pdV_bdry,   out.E_floor,    out.E_safety,
        out.E_redistribution_unresolved, out.E_solver, before_E_int_e,
        before_E_int_i,       before_E_kin,     before_E_rad,
        out.E_rad_mesh_advection,
    };
    input.reduction->allreduce_sum(reduced.data(), static_cast<int>(reduced.size()));
    out.E_int_e = reduced[0];
    out.E_int_i = reduced[1];
    out.E_kin = reduced[2];
    out.E_rad = reduced[3];
    out.E_Marshak_in = reduced[4];
    out.E_volume_in = reduced[5];
    out.E_burn_in = reduced[6];
    out.E_rad_esc = reduced[7];
    out.E_numerical_loss = reduced[8];
    out.E_pdV_bdry = reduced[9];
    out.E_floor = reduced[10];
    out.E_safety = reduced[11];
    out.E_redistribution_unresolved = reduced[12];
    out.E_solver = reduced[13];
    before_E_int_e = reduced[14];
    before_E_int_i = reduced[15];
    before_E_kin = reduced[16];
    before_E_rad = reduced[17];
    out.E_rad_mesh_advection = reduced[18];
  }

  out.E_kinetic = out.E_kin;
  out.E_internal = out.E_int_e + out.E_int_i;
  out.E_total = out.E_internal + out.E_kin + out.E_rad;

  const double E_total_before =
      before_E_int_e + before_E_int_i + before_E_kin + before_E_rad;
  const double E_total_after = out.E_total;
  out.dE_total = E_total_after - E_total_before;

  const double E_solver_source = std::max(out.E_solver, 0.0);
  const double E_solver_sink = std::max(-out.E_solver, 0.0);
  const double E_source =
      out.E_laser_in + out.E_Marshak_in + out.E_volume_in + out.E_burn_in +
      E_solver_source;
  const double E_sink = out.E_laser_esc + out.E_cbet_iaw + out.E_rad_esc +
                        out.E_numerical_loss + out.E_pdV_bdry + E_solver_sink;
  // E_safety can include E_floor contributions in current operator plumbing.
  // Subtract floor once and only the non-overlapping safety remainder.
  const double E_safety_non_floor = std::max(out.E_safety - out.E_floor, 0.0);
  const double E_artificial =
      out.E_floor + E_safety_non_floor + out.E_redistribution_unresolved;
  out.E_denom = std::max({E_total_before, E_source, 1.0e-20});
  // W-J: subtract the tallied radiation-field mesh-advection term (a known,
  // physical dE_total contribution outside every operator tally) so the
  // residual measures genuine bookkeeping error.
  out.epsilon_budget =
      std::abs((out.dE_total - E_artificial - out.E_rad_mesh_advection) -
               (E_source - E_sink)) / out.E_denom;
  return out;
}

}  // namespace tenryu::diagnostics
