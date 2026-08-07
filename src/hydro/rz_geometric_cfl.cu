#include "hydro/cfl.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

constexpr std::uint8_t kGeomCellStructured = 0U;
constexpr std::uint8_t kGeomCellButton = 1U;
constexpr std::uint8_t kGeomCellDormant = 2U;

int button_outer_node_ring_or_zero(const mesh::Mesh& mesh) {
  return (mesh.button_center && mesh.button_center->enabled)
             ? mesh.button_center->outer_node_ring
             : 0;
}

std::uint8_t* upload_button_geom_cell_kind_if_needed(const mesh::Mesh& mesh) {
  const int outer = button_outer_node_ring_or_zero(mesh);
  if (outer <= 0 || mesh.topo.n_cells <= 0) {
    return nullptr;
  }
  std::vector<std::uint8_t> cell_kind(static_cast<std::size_t>(mesh.topo.n_cells),
                                      kGeomCellStructured);
  for (int c = 0; c < mesh.topo.n_cells; ++c) {
    if (mesh.is_button_cell(c)) {
      cell_kind[static_cast<std::size_t>(c)] = kGeomCellButton;
    } else if (mesh.is_dormant_cell(c)) {
      cell_kind[static_cast<std::size_t>(c)] = kGeomCellDormant;
    }
  }

  std::uint8_t* d_cell_kind = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_kind),
                        cell_kind.size() * sizeof(std::uint8_t)),
             "RZ geometric CFL: cudaMalloc cell_kind failed");
  cuda_check(cudaMemcpy(d_cell_kind,
                        cell_kind.data(),
                        cell_kind.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "RZ geometric CFL: copy cell_kind failed");
  return d_cell_kind;
}

__device__ double atomic_min_double(double* address, const double val) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (val < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ double rz_volume4(const double2 x[4]) {
  double s = 0.0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const int l = (k + 1) & 3;
    const double r0 = x[k].x;
    const double z0 = x[k].y;
    const double r1 = x[l].x;
    const double z1 = x[l].y;
    s += (r0 + r1) * (r0 * z1 - r1 * z0);
  }
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  return pi_over_three * s;
}

__device__ double rz_csr_polygon_volume_at_tau(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ cell_nodes,
    const int active_nverts,
    const double orientation_sign,
    const double tau) {
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  double sum = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const int kp = (k + 1) % active_nverts;
    const int n = cell_nodes[k];
    const int np = cell_nodes[kp];
    const double r0 = x_r[n] + tau * v_r[n];
    const double z0 = x_z[n] + tau * v_z[n];
    const double r1 = x_r[np] + tau * v_r[np];
    const double z1 = x_z[np] + tau * v_z[np];
    sum += (r0 + r1) * (r0 * z1 - r1 * z0);
  }
  return orientation_sign * pi_over_three * sum;
}

__device__ bool rz_csr_polygon_geom_ok_with_cumulative(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ cell_nodes,
    const int active_nverts,
    const double orientation_sign,
    const double tau,
    const double V0,
    const double V_initial,
    const double etaV,
    const double etaV_cum,
    const double r_floor) {
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_nodes[k];
    if (x_r[n] + tau * v_r[n] < r_floor) {
      return false;
    }
  }
  const double Vp = rz_csr_polygon_volume_at_tau(
      x_r, x_z, v_r, v_z, cell_nodes, active_nverts, orientation_sign, tau);
  const double floor_per_step = etaV * V0;
  const double floor_cumulative = etaV_cum * V_initial;
  const double floor_effective = fmax(floor_per_step, floor_cumulative);
  return isfinite(Vp) && (Vp >= floor_effective);
}

__device__ double rz_button_volume_at_tau(const double* __restrict__ x_r,
                                          const double* __restrict__ x_z,
                                          const double* __restrict__ v_r,
                                          const double* __restrict__ v_z,
                                          const int outer_ring,
                                          const int nz,
                                          const double tau) {
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  const int nverts = nz + 1;
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    const int n = rz::button_seam_node_index(outer_ring, k, nz);
    const int np = rz::button_seam_node_index(outer_ring, kp, nz);
    const double r0 = x_r[n] + tau * v_r[n];
    const double z0 = x_z[n] + tau * v_z[n];
    const double r1 = x_r[np] + tau * v_r[np];
    const double z1 = x_z[np] + tau * v_z[np];
    sum += (r0 + r1) * (r0 * z1 - r1 * z0);
  }
  return rz::button_orientation_sign() * pi_over_three * sum;
}

__device__ bool rz_geom_ok_with_cumulative(const double2 x0[4],
                                            const double2 vhalf[4],
                                            const double tau,
                                            const double V0,
                                            const double V_initial,
                                            const double etaV,
                                            const double etaV_cum,
                                            const double r_floor) {
  double2 xp[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    xp[k].x = x0[k].x + tau * vhalf[k].x;
    xp[k].y = x0[k].y + tau * vhalf[k].y;
    if (xp[k].x < r_floor) {
      return false;
    }
  }
  const double Vp = rz_volume4(xp);
  const double floor_per_step = etaV * V0;
  const double floor_cumulative = etaV_cum * V_initial;
  const double floor_effective = fmax(floor_per_step, floor_cumulative);
  return isfinite(Vp) && (Vp >= floor_effective);
}

__device__ bool rz_button_geom_ok_with_cumulative(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int outer_ring,
    const int nz,
    const double tau,
    const double V0,
    const double V_initial,
    const double etaV,
    const double etaV_cum) {
  const double Vp =
      rz_button_volume_at_tau(x_r, x_z, v_r, v_z, outer_ring, nz, tau);
  const double floor_per_step = etaV * V0;
  const double floor_cumulative = etaV_cum * V_initial;
  const double floor_effective = fmax(floor_per_step, floor_cumulative);
  return isfinite(Vp) && (Vp >= floor_effective);
}

__global__ void rz_geometric_cfl_kernel(double* __restrict__ min_dt,
                                        const double* __restrict__ x_r,
                                        const double* __restrict__ x_z,
                                        const double* __restrict__ v_r,
                                        const double* __restrict__ v_z,
                                        const double* __restrict__ cell_vol_initial,
                                        const std::uint8_t* __restrict__ cell_kind,
                                        const int* __restrict__ cell_node_csr_offsets,
                                        const int* __restrict__ cell_node_csr_indices,
                                        const int* __restrict__ cell_orientation_sign,
                                        const std::uint8_t* __restrict__ cell_nverts,
                                        const int n_cells,
                                        const int nr,
                                        const int nz,
                                        const int button_outer_node_ring,
                                        const double dt_in,
                                        const double etaV,
                                        const double etaV_cum,
                                        const int cumulative_enabled,
                                        const double r_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const std::uint8_t kind =
      (cell_kind != nullptr) ? cell_kind[c] : kGeomCellStructured;
  if (kind == kGeomCellDormant) {
    return;
  }

  if (kind == kGeomCellButton && button_outer_node_ring > 0 && c == 0) {
    const double V0 =
        rz_button_volume_at_tau(x_r, x_z, v_r, v_z, button_outer_node_ring,
                                nz, 0.0);
    if (!(V0 > 0.0) || !isfinite(V0)) {
      atomic_min_double(min_dt, 0.0);
      return;
    }
    double V_initial = 0.0;
    if (cumulative_enabled != 0) {
      V_initial = cell_vol_initial[c];
      if (!(V_initial > 0.0) || !isfinite(V_initial)) {
        atomic_min_double(min_dt, 0.0);
        return;
      }
    }
    if (rz_button_geom_ok_with_cumulative(
            x_r, x_z, v_r, v_z, button_outer_node_ring, nz, dt_in, V0,
            V_initial, etaV, etaV_cum)) {
      return;
    }

    double lo = 0.0;
    double hi = dt_in;
#pragma unroll
    for (int it = 0; it < 48; ++it) {
      const double mid = 0.5 * (lo + hi);
      if (rz_button_geom_ok_with_cumulative(
              x_r, x_z, v_r, v_z, button_outer_node_ring, nz, mid, V0,
              V_initial, etaV, etaV_cum)) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    atomic_min_double(min_dt, lo);
    return;
  }

  if (cell_node_csr_offsets != nullptr &&
      cell_node_csr_indices != nullptr &&
      cell_orientation_sign != nullptr) {
    const int off = cell_node_csr_offsets[c];
    const int active_nverts =
        mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    const int* cell_nodes = cell_node_csr_indices + off;
    const double orientation_sign =
        static_cast<double>(cell_orientation_sign[c]);
    const double V0 = rz_csr_polygon_volume_at_tau(
        x_r, x_z, v_r, v_z, cell_nodes, active_nverts, orientation_sign, 0.0);
    if (!(V0 > 0.0) || !isfinite(V0)) {
      atomic_min_double(min_dt, 0.0);
      return;
    }
    double V_initial = 0.0;
    if (cumulative_enabled != 0) {
      V_initial = cell_vol_initial[c];
      if (!(V_initial > 0.0) || !isfinite(V_initial)) {
        atomic_min_double(min_dt, 0.0);
        return;
      }
    }
    if (rz_csr_polygon_geom_ok_with_cumulative(
            x_r, x_z, v_r, v_z, cell_nodes, active_nverts, orientation_sign,
            dt_in, V0, V_initial, etaV, etaV_cum, r_floor)) {
      return;
    }

    double lo = 0.0;
    double hi = dt_in;
#pragma unroll
    for (int it = 0; it < 48; ++it) {
      const double mid = 0.5 * (lo + hi);
      if (rz_csr_polygon_geom_ok_with_cumulative(
              x_r, x_z, v_r, v_z, cell_nodes, active_nverts,
              orientation_sign, mid, V0, V_initial, etaV, etaV_cum,
              r_floor)) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    atomic_min_double(min_dt, lo);
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  const double2 x0[4] = {
      make_double2(x_r[n00], x_z[n00]),
      make_double2(x_r[n10], x_z[n10]),
      make_double2(x_r[n11], x_z[n11]),
      make_double2(x_r[n01], x_z[n01]),
  };
  const double2 vh[4] = {
      make_double2(v_r[n00], v_z[n00]),
      make_double2(v_r[n10], v_z[n10]),
      make_double2(v_r[n11], v_z[n11]),
      make_double2(v_r[n01], v_z[n01]),
  };

  const double V0 = rz_volume4(x0);
  if (!(V0 > 0.0) || !isfinite(V0)) {
    atomic_min_double(min_dt, 0.0);
    return;
  }
  double V_initial = 0.0;
  if (cumulative_enabled != 0) {
    V_initial = cell_vol_initial[c];
    if (!(V_initial > 0.0) || !isfinite(V_initial)) {
      atomic_min_double(min_dt, 0.0);
      return;
    }
  }
  if (rz_geom_ok_with_cumulative(x0, vh, dt_in, V0, V_initial, etaV, etaV_cum,
                                 r_floor)) {
    return;
  }

  double lo = 0.0;
  double hi = dt_in;
#pragma unroll
  for (int it = 0; it < 48; ++it) {
    const double mid = 0.5 * (lo + hi);
    if (rz_geom_ok_with_cumulative(x0, vh, mid, V0, V_initial, etaV, etaV_cum,
                                   r_floor)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  atomic_min_double(min_dt, lo);
}

}  // namespace

double compute_rz_geometric_cfl_dt(const core::State& state,
                                   const core::Config& cfg,
                                   const double dt_proposed,
                                   const double* vhalf_r,
                                   const double* vhalf_z) {
  if (!cfg.numerics.hydro.rz_geometric_cfl_enabled ||
      state.mesh.dim != 2 ||
      !(dt_proposed > 0.0) ||
      !std::isfinite(dt_proposed)) {
    return std::numeric_limits<double>::infinity();
  }

  TENRYU_ASSERT(cfg.numerics.hydro.rz_geometric_cfl_etaV > 0.0 &&
                    cfg.numerics.hydro.rz_geometric_cfl_etaV <= 1.0,
                "RZ geometric CFL etaV must be in (0, 1]");
  TENRYU_ASSERT(cfg.numerics.hydro.rz_geometric_cfl_r_floor >= 0.0,
                "RZ geometric CFL r_floor must be >= 0");
  TENRYU_ASSERT(cfg.numerics.hydro.rz_geometric_cfl_v_initial_floor >= 0.0 &&
                    cfg.numerics.hydro.rz_geometric_cfl_v_initial_floor <= 1.0,
                "RZ geometric CFL initial-volume floor must be in [0, 1]");
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "RZ geometric CFL requires matching node position arrays");
  TENRYU_ASSERT(state.v_r.size() == state.v_z.size(),
                "RZ geometric CFL requires matching node velocity arrays");
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "RZ geometric CFL requires matching node position/velocity arrays");
  TENRYU_ASSERT(vhalf_r != nullptr && vhalf_z != nullptr,
                "RZ geometric CFL requires non-null half-step velocity arrays");

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  const bool is_multiblock = state.mesh.topo.multiblock.has_value();
  const bool cumulative_enabled =
      cfg.numerics.hydro.rz_geometric_cfl_cumulative_protection_enabled;
  if (n_cells <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  TENRYU_ASSERT(static_cast<int>(state.x_r.size()) == n_nodes,
                "RZ geometric CFL requires node field size == n_nodes");
  if (is_multiblock) {
    const auto& mb = *state.mesh.topo.multiblock;
    TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "RZ geometric CFL requires multiblock cell-node CSR offsets");
    TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "RZ geometric CFL requires multiblock cell-node CSR indices");
    TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                      static_cast<std::size_t>(n_cells),
                  "RZ geometric CFL requires multiblock orientation signs");
    TENRYU_ASSERT(state.mesh.cell_nverts.empty() ||
                      state.mesh.cell_nverts.size() ==
                          static_cast<std::size_t>(n_cells),
                  "RZ geometric CFL cell_nverts size mismatch");
  } else {
    TENRYU_ASSERT(n_cells == nr * nz,
                  "RZ geometric CFL requires structured cell count");
    TENRYU_ASSERT(n_nodes == (nr + 1) * (nz + 1),
                  "RZ geometric CFL requires structured node count");
  }
  if (cumulative_enabled) {
    TENRYU_ASSERT(static_cast<int>(state.cell_vol_initial.size()) == n_cells,
                  "RZ geometric CFL cumulative protection requires initial cell volumes");
  }

  double* d_min_dt = nullptr;
  d_min_dt = static_cast<double*>(
      core::device_scratch_acquire("rzgcfl:min_dt", sizeof(double)));
  const double inf = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(d_min_dt, &inf, sizeof(double), cudaMemcpyHostToDevice),
             "RZ geometric CFL: init min_dt failed");
  std::uint8_t* d_cell_kind =
      upload_button_geom_cell_kind_if_needed(state.mesh);
  const int button_outer_node_ring =
      button_outer_node_ring_or_zero(state.mesh);
  int* d_cell_orientation_sign = nullptr;
  const std::uint8_t* d_cell_nverts = nullptr;
  core::DeviceArray<std::uint8_t> cell_nverts_storage;
  if (is_multiblock) {
    const auto& mb = *state.mesh.topo.multiblock;
    d_cell_orientation_sign = static_cast<int*>(core::device_scratch_acquire(
        "rzgcfl:cell_orientation_sign",
        static_cast<std::size_t>(n_cells) * sizeof(int)));
    cuda_check(cudaMemcpy(d_cell_orientation_sign,
                          mb.cell_orientation_sign.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "RZ geometric CFL: copy multiblock orientation signs failed");
    if (!state.mesh.cell_nverts.empty()) {
      cell_nverts_storage.reset(static_cast<std::size_t>(n_cells));
      cell_nverts_storage.copy_from_host(state.mesh.cell_nverts);
      d_cell_nverts = cell_nverts_storage.data();
    }
  }

  const int blocks = (n_cells + 255) / 256;
  rz_geometric_cfl_kernel<<<blocks, 256>>>(
      d_min_dt, state.x_r.data(), state.x_z.data(), vhalf_r,
      vhalf_z, cumulative_enabled ? state.cell_vol_initial.data() : nullptr,
      d_cell_kind,
      is_multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data()
                    : nullptr,
      is_multiblock ? state.mesh.multiblock_cell_node_csr_indices.data()
                    : nullptr,
      d_cell_orientation_sign, d_cell_nverts, n_cells, nr, nz,
      button_outer_node_ring, dt_proposed,
      cfg.numerics.hydro.rz_geometric_cfl_etaV,
      cfg.numerics.hydro.rz_geometric_cfl_v_initial_floor,
      cumulative_enabled ? 1 : 0,
      cfg.numerics.hydro.rz_geometric_cfl_r_floor);
  cuda_check(cudaGetLastError(), "RZ geometric CFL kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "RZ geometric CFL kernel execution failed");

  double min_dt = inf;
  cuda_check(cudaMemcpy(&min_dt, d_min_dt, sizeof(double), cudaMemcpyDeviceToHost),
             "RZ geometric CFL: copy min_dt failed");
  if (d_cell_kind != nullptr) {
    cuda_check(cudaFree(d_cell_kind),
               "RZ geometric CFL: cudaFree cell_kind failed");
  }

  if (!std::isfinite(min_dt)) {
    return std::numeric_limits<double>::infinity();
  }
  return std::max(0.0, std::min(dt_proposed, min_dt));
}

double compute_rz_geometric_cfl_dt(const core::State& state,
                                   const core::Config& cfg,
                                   const double dt_proposed) {
  return compute_rz_geometric_cfl_dt(
      state, cfg, dt_proposed, state.v_r.data(), state.v_z.data());
}

}  // namespace tenryu::hydro
