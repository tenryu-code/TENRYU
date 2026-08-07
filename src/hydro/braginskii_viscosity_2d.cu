// 2D RZ Braginskii viscosity kernels, ion + electron channels
// (docs/design/2d_visc_port_spec.md; electron species extension
// docs/design/visc_2d_parity_20260717.md).

#include "hydro/braginskii_viscosity.cuh"
#include "hydro/braginskii_viscosity_device.cuh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::braginskii {

namespace {

void brag_cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " +
                             cudaGetErrorString(err));
  }
}

void brag_sync(const char* msg) {
  brag_cuda_check(cudaGetLastError(), msg);
  brag_cuda_check(cudaDeviceSynchronize(), msg);
}

__device__ inline double atomic_add_double(double* address, const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(
                                  static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline int structured_node_index(const int i,
                                            const int j,
                                            const int nz) {
  return i * (nz + 1) + j;
}

// SPECIES branches are fully separated if constexpr blocks (1D W2
// convention): SPECIES=0 keeps the ion-only load order and expressions
// source-identical to the pre-electron module (bitwise contract).
template <int SPECIES>
__device__ void compute_cell_corner_forces_2d(
    double* __restrict__ corner_visc_r,
    double* __restrict__ corner_visc_z,
    double* __restrict__ heat_rate,
    double* __restrict__ heat_rate_e,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int* nodes,
    const int nverts,
    const DeviceParams p) {
  const int base = c * 4;
  for (int k = 0; k < 4; ++k) {
    corner_visc_r[base + k] = 0.0;
    corner_visc_z[base + k] = 0.0;
  }
  if (heat_rate != nullptr) {
    heat_rate[c] = 0.0;
  }
  if (heat_rate_e != nullptr) {
    heat_rate_e[c] = 0.0;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  double r[4];
  double z[4];
  double vr[4];
  double vz[4];
  for (int k = 0; k < nverts; ++k) {
    const int n = nodes[k];
    r[k] = x_r[n];
    z[k] = x_z[n];
    vr[k] = v_r[n];
    vz[k] = v_z[n];
  }

  double area = 0.0;
  double rsum = 0.0;
  double length = CUDART_INF;
  double a_r[4];
  double a_z[4];
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1) % nverts;
    const int km = (k + nverts - 1) % nverts;
    area += r[k] * z[kp] - r[kp] * z[k];
    a_r[k] = 0.5 * (z[kp] - z[km]);
    a_z[k] = 0.5 * (r[km] - r[kp]);
    rsum += r[k];
    const double dr = r[kp] - r[k];
    const double dz = z[kp] - z[k];
    length = fmin(length, sqrt(dr * dr + dz * dz));
  }
  area = 0.5 * area;

  const double volume = vol[c];
  if (!(fabs(area) > 0.0) || !(rsum > 0.0) || !(volume > 0.0) ||
      !(length > 0.0)) {
    return;
  }

  double sum_vr_ar = 0.0;
  double sum_vr_az = 0.0;
  double sum_vz_ar = 0.0;
  double sum_vz_az = 0.0;
  double sum_vr = 0.0;
  for (int k = 0; k < nverts; ++k) {
    sum_vr_ar += vr[k] * a_r[k];
    sum_vr_az += vr[k] * a_z[k];
    sum_vz_ar += vz[k] * a_r[k];
    sum_vz_az += vz[k] * a_z[k];
    sum_vr += vr[k];
  }

  const double G_rr = sum_vr_ar / area;
  const double G_rz = sum_vr_az / area;
  const double G_zr = sum_vz_ar / area;
  const double G_zz = sum_vz_az / area;
  const double h = sum_vr / rsum;
  const double div3 = G_rr + G_zz + h;
  const double W_rr = 2.0 * G_rr - (2.0 / 3.0) * div3;
  const double W_zz = 2.0 * G_zz - (2.0 / 3.0) * div3;
  const double W_pp = 2.0 * h - (2.0 / 3.0) * div3;
  const double W_rz = G_rz + G_zr;
  if constexpr (SPECIES == 0) {
    const double ti_ev = (Ti != nullptr) ? Ti[c] : p.ti_floor_ev;
    const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
    const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
    const double eta =
        braginskii_eta_poise(rho[c], ti_ev, a_eff, zb, length, p);
    const double pi_rr = -eta * W_rr;
    const double pi_zz = -eta * W_zz;
    const double pi_pp = -eta * W_pp;
    const double pi_rz = -eta * W_rz;

    for (int k = 0; k < nverts; ++k) {
      corner_visc_r[base + k] =
          volume * ((pi_rr * a_r[k] + pi_rz * a_z[k]) / area +
                    pi_pp / rsum);
      corner_visc_z[base + k] =
          volume * ((pi_rz * a_r[k] + pi_zz * a_z[k]) / area);
    }
    if (heat_rate != nullptr) {
      heat_rate[c] =
          0.5 * eta * (W_rr * W_rr + W_zz * W_zz + W_pp * W_pp +
                       2.0 * W_rz * W_rz) *
          volume;
    }
  } else if constexpr (SPECIES == 1) {
    const double te_ev = (Te != nullptr) ? Te[c] : p.te_floor_ev;
    const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
    const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
    const double eta =
        braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, length, p);
    const double pi_rr = -eta * W_rr;
    const double pi_zz = -eta * W_zz;
    const double pi_pp = -eta * W_pp;
    const double pi_rz = -eta * W_rz;

    for (int k = 0; k < nverts; ++k) {
      corner_visc_r[base + k] =
          volume * ((pi_rr * a_r[k] + pi_rz * a_z[k]) / area +
                    pi_pp / rsum);
      corner_visc_z[base + k] =
          volume * ((pi_rz * a_r[k] + pi_zz * a_z[k]) / area);
    }
    if (heat_rate_e != nullptr) {
      heat_rate_e[c] =
          0.5 * eta * (W_rr * W_rr + W_zz * W_zz + W_pp * W_pp +
                       2.0 * W_rz * W_rz) *
          volume;
    }
  } else {
    const double ti_ev = (Ti != nullptr) ? Ti[c] : p.ti_floor_ev;
    const double te_ev = (Te != nullptr) ? Te[c] : p.te_floor_ev;
    const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
    const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
    double eta_i;
    double eta_e;
    if (p.model == 1) {
      // "constant" under species="both": eta_eff = eta_const invariantly,
      // split half/half so all three species settings share the same
      // momentum physics and differ only in heat routing (1D convention).
      eta_i = 0.5 * p.eta_const;
      eta_e = 0.5 * p.eta_const;
    } else {
      eta_i = braginskii_eta_poise(rho[c], ti_ev, a_eff, zb, length, p);
      eta_e = braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, length, p);
    }
    const double eta = eta_i + eta_e;
    const double pi_rr = -eta * W_rr;
    const double pi_zz = -eta * W_zz;
    const double pi_pp = -eta * W_pp;
    const double pi_rz = -eta * W_rz;

    for (int k = 0; k < nverts; ++k) {
      corner_visc_r[base + k] =
          volume * ((pi_rr * a_r[k] + pi_rz * a_z[k]) / area +
                    pi_pp / rsum);
      corner_visc_z[base + k] =
          volume * ((pi_rz * a_r[k] + pi_zz * a_z[k]) / area);
    }
    const double wsq = W_rr * W_rr + W_zz * W_zz + W_pp * W_pp +
                       2.0 * W_rz * W_rz;
    if (heat_rate != nullptr) {
      heat_rate[c] = 0.5 * eta_i * wsq * volume;
    }
    if (heat_rate_e != nullptr) {
      heat_rate_e[c] = 0.5 * eta_e * wsq * volume;
    }
  }
}

template <int SPECIES>
__global__ void corner_forces_structured_kernel(
    double* __restrict__ corner_visc_r,
    double* __restrict__ corner_visc_z,
    double* __restrict__ heat_rate,
    double* __restrict__ heat_rate_e,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const DeviceParams p) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      structured_node_index(i, j, nz),
      structured_node_index(i + 1, j, nz),
      structured_node_index(i + 1, j + 1, nz),
      structured_node_index(i, j + 1, nz),
  };
  compute_cell_corner_forces_2d<SPECIES>(
      corner_visc_r, corner_visc_z, heat_rate, heat_rate_e, x_r, x_z, v_r,
      v_z, rho, Ti, Te, A_eff, zbar, vol, hydro_active, c, nodes, 4, p);
}

template <int SPECIES>
__global__ void corner_forces_multiblock_kernel(
    double* __restrict__ corner_visc_r,
    double* __restrict__ corner_visc_z,
    double* __restrict__ heat_rate,
    double* __restrict__ heat_rate_e,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const DeviceParams p) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int off = cell_node_csr_offsets[c];
  const int nverts =
      (cell_nverts != nullptr && cell_nverts[c] == 3U) ? 3 : 4;
  int nodes[4];
  for (int k = 0; k < nverts; ++k) {
    nodes[k] = cell_node_csr_indices[off + k];
  }
  compute_cell_corner_forces_2d<SPECIES>(
      corner_visc_r, corner_visc_z, heat_rate, heat_rate_e, x_r, x_z, v_r,
      v_z, rho, Ti, Te, A_eff, zbar, vol, hydro_active, c, nodes, nverts, p);
}

__global__ void sum_corner_forces_structured_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ corner_visc_r,
    const double* __restrict__ corner_visc_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      structured_node_index(i, j, nz),
      structured_node_index(i + 1, j, nz),
      structured_node_index(i + 1, j + 1, nz),
      structured_node_index(i, j + 1, nz),
  };
  const int base = c * 4;
  for (int k = 0; k < 4; ++k) {
    atomic_add_double(force_r + nodes[k], corner_visc_r[base + k]);
    atomic_add_double(force_z + nodes[k], corner_visc_z[base + k]);
  }
}

__global__ void sum_corner_forces_multiblock_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ corner_visc_r,
    const double* __restrict__ corner_visc_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ rev_node_offsets,
    const int* __restrict__ rev_node_cells,
    const int* __restrict__ rev_node_corners,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  const int off = rev_node_offsets[n];
  const int end = rev_node_offsets[n + 1];
  double fr = 0.0;
  double fz = 0.0;
  for (int p = off; p < end; ++p) {
    const int c = rev_node_cells[p];
    if (hydro_active != nullptr && hydro_active[c] == 0) {
      continue;
    }
    const int corner = rev_node_corners[p];
    const int idx = c * 4 + corner;
    fr += corner_visc_r[idx];
    fz += corner_visc_z[idx];
  }
  force_r[n] += fr;
  force_z[n] += fz;
}

__global__ void work_visc_structured_kernel(
    double* __restrict__ work_visc,
    const double* __restrict__ corner_visc_r,
    const double* __restrict__ corner_visc_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    work_visc[c] = 0.0;
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      structured_node_index(i, j, nz),
      structured_node_index(i + 1, j, nz),
      structured_node_index(i + 1, j + 1, nz),
      structured_node_index(i, j + 1, nz),
  };
  const int base = c * 4;
  double work = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int n = nodes[k];
    work -= corner_visc_r[base + k] * v_r[n] +
            corner_visc_z[base + k] * v_z[n];
  }
  work_visc[c] = work;
}

__global__ void work_visc_multiblock_kernel(
    double* __restrict__ work_visc,
    const double* __restrict__ corner_visc_r,
    const double* __restrict__ corner_visc_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    work_visc[c] = 0.0;
    return;
  }
  const int off = cell_node_csr_offsets[c];
  const int base = c * 4;
  const int nverts =
      (cell_nverts != nullptr && cell_nverts[c] == 3U) ? 3 : 4;
  double work = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int n = cell_node_csr_indices[off + k];
    work -= corner_visc_r[base + k] * v_r[n] +
            corner_visc_z[base + k] * v_z[n];
  }
  work_visc[c] = work;
}

template <int SPECIES>
__device__ void update_dt_cell_2d(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const int c,
    const int* nodes,
    const int nverts,
    const DeviceParams p,
    const double dt_safety) {
  double r[4];
  double z[4];
  for (int k = 0; k < nverts; ++k) {
    const int n = nodes[k];
    r[k] = x_r[n];
    z[k] = x_z[n];
  }
  double area = 0.0;
  double rsum = 0.0;
  double length = CUDART_INF;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1) % nverts;
    area += r[k] * z[kp] - r[kp] * z[k];
    rsum += r[k];
    const double dr = r[kp] - r[k];
    const double dz = z[kp] - z[k];
    length = fmin(length, sqrt(dr * dr + dz * dz));
  }
  area = 0.5 * area;
  if (!(fabs(area) > 0.0) || !(rsum > 0.0) || !(length > 0.0)) {
    return;
  }
  double eta;
  if constexpr (SPECIES == 0) {
    const double ti_ev = (Ti != nullptr) ? Ti[c] : p.ti_floor_ev;
    const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
    const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
    eta = braginskii_eta_poise(rho[c], ti_ev, a_eff, zb, length, p);
  } else if constexpr (SPECIES == 1) {
    const double te_ev = (Te != nullptr) ? Te[c] : p.te_floor_ev;
    const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
    const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
    eta = braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, length, p);
  } else {
    const double ti_ev = (Ti != nullptr) ? Ti[c] : p.ti_floor_ev;
    const double te_ev = (Te != nullptr) ? Te[c] : p.te_floor_ev;
    const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
    const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
    if (p.model == 1) {
      eta = p.eta_const;
    } else {
      eta = braginskii_eta_poise(rho[c], ti_ev, a_eff, zb, length, p) +
            braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, length, p);
    }
  }
  if (!(eta > 0.0)) {
    return;
  }
  const double dtc =
      dt_safety * rho[c] * length * length / ((8.0 / 3.0) * eta);
  if (dtc > 0.0) {
    atomic_min_double_brag(min_dt, dtc);
  }
}

template <int SPECIES>
__global__ void dt_structured_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const DeviceParams p,
    const double dt_safety) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      structured_node_index(i, j, nz),
      structured_node_index(i + 1, j, nz),
      structured_node_index(i + 1, j + 1, nz),
      structured_node_index(i, j + 1, nz),
  };
  update_dt_cell_2d<SPECIES>(min_dt, x_r, x_z, rho, Ti, Te, A_eff, zbar, c,
                             nodes, 4, p, dt_safety);
}

template <int SPECIES>
__global__ void dt_multiblock_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const DeviceParams p,
    const double dt_safety) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  const int off = cell_node_csr_offsets[c];
  const int nverts =
      (cell_nverts != nullptr && cell_nverts[c] == 3U) ? 3 : 4;
  int nodes[4];
  for (int k = 0; k < nverts; ++k) {
    nodes[k] = cell_node_csr_indices[off + k];
  }
  update_dt_cell_2d<SPECIES>(min_dt, x_r, x_z, rho, Ti, Te, A_eff, zbar, c,
                             nodes, nverts, p, dt_safety);
}

// History-diagnostics device body: per-cell physical channel etas,
// configured effective eta, and configured instantaneous heat rates using
// the 2D strain operator of the stress kernel and the min-edge cap length
// of the dt kernel. Inactive or degenerate cells are marked with
// eta_i_phys = -1 and skipped on the host (1D twin:
// braginskii_history_diag_1d_kernel in braginskii_viscosity.cu).
__device__ void history_diag_cell_2d(
    double* __restrict__ eta_i_phys,
    double* __restrict__ eta_e_phys,
    double* __restrict__ eta_eff,
    double* __restrict__ heat_i,
    double* __restrict__ heat_e,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int* nodes,
    const int nverts,
    const int species,
    const DeviceParams p_run,
    const DeviceParams p_phys) {
  eta_i_phys[c] = -1.0;
  eta_e_phys[c] = 0.0;
  eta_eff[c] = 0.0;
  heat_i[c] = 0.0;
  heat_e[c] = 0.0;
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  if (!active) {
    return;
  }

  double r[4];
  double z[4];
  double vr[4];
  double vz[4];
  for (int k = 0; k < nverts; ++k) {
    const int n = nodes[k];
    r[k] = x_r[n];
    z[k] = x_z[n];
    vr[k] = v_r[n];
    vz[k] = v_z[n];
  }

  double area = 0.0;
  double rsum = 0.0;
  double length = CUDART_INF;
  double a_r[4];
  double a_z[4];
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1) % nverts;
    const int km = (k + nverts - 1) % nverts;
    area += r[k] * z[kp] - r[kp] * z[k];
    a_r[k] = 0.5 * (z[kp] - z[km]);
    a_z[k] = 0.5 * (r[km] - r[kp]);
    rsum += r[k];
    const double dr = r[kp] - r[k];
    const double dz = z[kp] - z[k];
    length = fmin(length, sqrt(dr * dr + dz * dz));
  }
  area = 0.5 * area;

  const double volume = vol[c];
  if (!(fabs(area) > 0.0) || !(rsum > 0.0) || !(volume > 0.0) ||
      !(length > 0.0)) {
    return;
  }

  double sum_vr_ar = 0.0;
  double sum_vr_az = 0.0;
  double sum_vz_ar = 0.0;
  double sum_vz_az = 0.0;
  double sum_vr = 0.0;
  for (int k = 0; k < nverts; ++k) {
    sum_vr_ar += vr[k] * a_r[k];
    sum_vr_az += vr[k] * a_z[k];
    sum_vz_ar += vz[k] * a_r[k];
    sum_vz_az += vz[k] * a_z[k];
    sum_vr += vr[k];
  }

  const double G_rr = sum_vr_ar / area;
  const double G_rz = sum_vr_az / area;
  const double G_zr = sum_vz_ar / area;
  const double G_zz = sum_vz_az / area;
  const double h = sum_vr / rsum;
  const double div3 = G_rr + G_zz + h;
  const double W_rr = 2.0 * G_rr - (2.0 / 3.0) * div3;
  const double W_zz = 2.0 * G_zz - (2.0 / 3.0) * div3;
  const double W_pp = 2.0 * h - (2.0 / 3.0) * div3;
  const double W_rz = G_rz + G_zr;
  const double wsq = W_rr * W_rr + W_zz * W_zz + W_pp * W_pp +
                     2.0 * W_rz * W_rz;
  const double ti_ev = (Ti != nullptr) ? Ti[c] : p_run.ti_floor_ev;
  const double te_ev = (Te != nullptr) ? Te[c] : p_run.te_floor_ev;
  const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
  const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
  // Physical channels (Braginskii formulas regardless of run model).
  eta_i_phys[c] =
      braginskii_eta_poise(rho[c], ti_ev, a_eff, zb, length, p_phys);
  eta_e_phys[c] =
      braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, length, p_phys);
  // Configured effective eta and heat split.
  double e_i = 0.0;
  double e_e = 0.0;
  if (species == 1) {
    e_e = braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, length, p_run);
  } else if (species == 2) {
    if (p_run.model == 1) {
      e_i = 0.5 * p_run.eta_const;
      e_e = 0.5 * p_run.eta_const;
    } else {
      e_i = braginskii_eta_poise(rho[c], ti_ev, a_eff, zb, length, p_run);
      e_e = braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, length, p_run);
    }
  } else {
    e_i = braginskii_eta_poise(rho[c], ti_ev, a_eff, zb, length, p_run);
  }
  eta_eff[c] = e_i + e_e;
  heat_i[c] = 0.5 * e_i * wsq * volume;
  heat_e[c] = 0.5 * e_e * wsq * volume;
}

__global__ void history_diag_2d_structured_kernel(
    double* __restrict__ eta_i_phys,
    double* __restrict__ eta_e_phys,
    double* __restrict__ eta_eff,
    double* __restrict__ heat_i,
    double* __restrict__ heat_e,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const int species,
    const DeviceParams p_run,
    const DeviceParams p_phys) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      structured_node_index(i, j, nz),
      structured_node_index(i + 1, j, nz),
      structured_node_index(i + 1, j + 1, nz),
      structured_node_index(i, j + 1, nz),
  };
  history_diag_cell_2d(eta_i_phys, eta_e_phys, eta_eff, heat_i, heat_e,
                       x_r, x_z, v_r, v_z, rho, Ti, Te, A_eff, zbar, vol,
                       hydro_active, c, nodes, 4, species, p_run, p_phys);
}

__global__ void history_diag_2d_multiblock_kernel(
    double* __restrict__ eta_i_phys,
    double* __restrict__ eta_e_phys,
    double* __restrict__ eta_eff,
    double* __restrict__ heat_i,
    double* __restrict__ heat_e,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const int species,
    const DeviceParams p_run,
    const DeviceParams p_phys) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int off = cell_node_csr_offsets[c];
  const int nverts =
      (cell_nverts != nullptr && cell_nverts[c] == 3U) ? 3 : 4;
  int nodes[4];
  for (int k = 0; k < nverts; ++k) {
    nodes[k] = cell_node_csr_indices[off + k];
  }
  history_diag_cell_2d(eta_i_phys, eta_e_phys, eta_eff, heat_i, heat_e,
                       x_r, x_z, v_r, v_z, rho, Ti, Te, A_eff, zbar, vol,
                       hydro_active, c, nodes, nverts, species, p_run,
                       p_phys);
}

int topo_cell_count(const Topo2D& topo) {
  return topo.multiblock ? topo.n_cells : topo.nr * topo.nz;
}

void validate_topo(const Topo2D& topo) {
  if (topo.multiblock) {
    TENRYU_ASSERT(topo.n_cells > 0 && topo.n_nodes > 0,
                  "braginskii2d multiblock topology requires positive counts");
    TENRYU_ASSERT(topo.cell_node_csr_offsets != nullptr &&
                      topo.cell_node_csr_indices != nullptr,
                  "braginskii2d multiblock topology requires cell-node CSR");
  } else {
    TENRYU_ASSERT(topo.nr > 0 && topo.nz > 0,
                  "braginskii2d structured topology requires nr/nz");
  }
}

void zero_array(double* ptr, const std::size_t n, const char* msg) {
  if (ptr == nullptr || n == 0U) {
    return;
  }
  brag_cuda_check(cudaMemset(ptr, 0, n * sizeof(double)), msg);
}

// Host-side staging shared by compute_dt_braginskii_2d and
// compute_history_diagnostics_2d: identical scratch tags, asserts, and
// copies as the original dt wrapper (byte-neutral refactor).
Topo2D stage_topo_2d_from_state(const core::State& state,
                                const core::Config& cfg,
                                const int n_cells,
                                const int n_nodes) {
  Topo2D topo;
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "braginskii2d requires multiblock topology storage");
    const auto& mb = *state.mesh.topo.multiblock;
    TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "braginskii2d multiblock CSR offset size mismatch");
    TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) * 4U,
                  "braginskii2d multiblock CSR index size mismatch");
    auto* d_offsets = static_cast<int*>(core::device_scratch_acquire(
        "brag2d:cell_node_csr_offsets",
        (static_cast<std::size_t>(n_cells) + 1U) * sizeof(int)));
    auto* d_indices = static_cast<int*>(core::device_scratch_acquire(
        "brag2d:cell_node_csr_indices",
        static_cast<std::size_t>(n_cells) * 4U * sizeof(int)));
    brag_cuda_check(
        cudaMemcpy(d_offsets, mb.cell_node_csr_offsets.data(),
                   (static_cast<std::size_t>(n_cells) + 1U) * sizeof(int),
                   cudaMemcpyHostToDevice),
        "braginskii2d: copy cell-node CSR offsets failed");
    brag_cuda_check(
        cudaMemcpy(d_indices, mb.cell_node_csr_indices.data(),
                   static_cast<std::size_t>(n_cells) * 4U * sizeof(int),
                   cudaMemcpyHostToDevice),
        "braginskii2d: copy cell-node CSR indices failed");
    topo.multiblock = true;
    topo.n_cells = n_cells;
    topo.n_nodes = n_nodes;
    topo.cell_node_csr_offsets = d_offsets;
    topo.cell_node_csr_indices = d_indices;
    if (!state.mesh.cell_nverts.empty()) {
      TENRYU_ASSERT(state.mesh.cell_nverts.size() ==
                        static_cast<std::size_t>(n_cells),
                    "braginskii2d cell_nverts size mismatch");
      auto* d_nverts = static_cast<std::uint8_t*>(
          core::device_scratch_acquire(
              "brag2d:cell_nverts",
              static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
      brag_cuda_check(
          cudaMemcpy(d_nverts, state.mesh.cell_nverts.data(),
                     static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                     cudaMemcpyHostToDevice),
          "braginskii2d: copy cell_nverts failed");
      topo.cell_nverts = d_nverts;
    }
  } else {
    topo.nr = state.mesh.topo.nr;
    topo.nz = state.mesh.topo.nz;
  }
  return topo;
}

std::int8_t* stage_hydro_active_2d(const core::State& state,
                                   const int n_cells) {
  std::int8_t* d_active = nullptr;
  if (!state.hydro_active.empty() &&
      state.hydro_active.size() == static_cast<std::size_t>(n_cells)) {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "brag2d:hydro_active",
        static_cast<std::size_t>(n_cells) * sizeof(std::int8_t)));
    brag_cuda_check(
        cudaMemcpy(d_active, state.hydro_active.data(),
                   static_cast<std::size_t>(n_cells) * sizeof(std::int8_t),
                   cudaMemcpyHostToDevice),
        "braginskii2d: copy hydro_active failed");
  }
  return d_active;
}

}  // namespace

void compute_corner_forces_2d(double* corner_visc_r,
                              double* corner_visc_z,
                              double* heat_rate,
                              double* heat_rate_e,
                              const double* x_r,
                              const double* x_z,
                              const double* v_r,
                              const double* v_z,
                              const double* rho,
                              const double* Ti,
                              const double* Te,
                              const double* A_eff,
                              const double* zbar,
                              const double* vol,
                              const std::int8_t* hydro_active,
                              const Topo2D& topo,
                              const Params& p) {
  const int n_cells = topo_cell_count(topo);
  if (n_cells <= 0) {
    return;
  }
  validate_topo(topo);
  const int blocks = (n_cells + 255) / 256;
  if (topo.multiblock) {
    switch (p.species) {
      case 1:
        corner_forces_multiblock_kernel<1><<<blocks, 256>>>(
            corner_visc_r, corner_visc_z, heat_rate, heat_rate_e, x_r, x_z,
            v_r, v_z, rho, Ti, Te, A_eff, zbar, vol, hydro_active,
            topo.cell_node_csr_offsets, topo.cell_node_csr_indices,
            topo.cell_nverts, n_cells, to_device_params(p));
        break;
      case 2:
        corner_forces_multiblock_kernel<2><<<blocks, 256>>>(
            corner_visc_r, corner_visc_z, heat_rate, heat_rate_e, x_r, x_z,
            v_r, v_z, rho, Ti, Te, A_eff, zbar, vol, hydro_active,
            topo.cell_node_csr_offsets, topo.cell_node_csr_indices,
            topo.cell_nverts, n_cells, to_device_params(p));
        break;
      default:
        corner_forces_multiblock_kernel<0><<<blocks, 256>>>(
            corner_visc_r, corner_visc_z, heat_rate, heat_rate_e, x_r, x_z,
            v_r, v_z, rho, Ti, Te, A_eff, zbar, vol, hydro_active,
            topo.cell_node_csr_offsets, topo.cell_node_csr_indices,
            topo.cell_nverts, n_cells, to_device_params(p));
        break;
    }
  } else {
    switch (p.species) {
      case 1:
        corner_forces_structured_kernel<1><<<blocks, 256>>>(
            corner_visc_r, corner_visc_z, heat_rate, heat_rate_e, x_r, x_z,
            v_r, v_z, rho, Ti, Te, A_eff, zbar, vol, hydro_active, topo.nr,
            topo.nz, to_device_params(p));
        break;
      case 2:
        corner_forces_structured_kernel<2><<<blocks, 256>>>(
            corner_visc_r, corner_visc_z, heat_rate, heat_rate_e, x_r, x_z,
            v_r, v_z, rho, Ti, Te, A_eff, zbar, vol, hydro_active, topo.nr,
            topo.nz, to_device_params(p));
        break;
      default:
        corner_forces_structured_kernel<0><<<blocks, 256>>>(
            corner_visc_r, corner_visc_z, heat_rate, heat_rate_e, x_r, x_z,
            v_r, v_z, rho, Ti, Te, A_eff, zbar, vol, hydro_active, topo.nr,
            topo.nz, to_device_params(p));
        break;
    }
  }
  brag_sync("braginskii2d: corner forces failed");
}

void add_corner_forces_to_nodes_2d(double* force_r,
                                   double* force_z,
                                   const double* corner_visc_r,
                                   const double* corner_visc_z,
                                   const std::int8_t* hydro_active,
                                   const Topo2D& topo) {
  const int n_cells = topo_cell_count(topo);
  if (n_cells <= 0) {
    return;
  }
  validate_topo(topo);
  if (topo.multiblock) {
    TENRYU_ASSERT(topo.rev_node_offsets != nullptr &&
                      topo.rev_node_cells != nullptr &&
                      topo.rev_node_corners != nullptr,
                  "braginskii2d multiblock force sum requires reverse CSR");
    const int blocks = (topo.n_nodes + 255) / 256;
    sum_corner_forces_multiblock_kernel<<<blocks, 256>>>(
        force_r, force_z, corner_visc_r, corner_visc_z, hydro_active,
        topo.rev_node_offsets, topo.rev_node_cells, topo.rev_node_corners,
        topo.n_nodes);
  } else {
    const int blocks = (n_cells + 255) / 256;
    sum_corner_forces_structured_kernel<<<blocks, 256>>>(
        force_r, force_z, corner_visc_r, corner_visc_z, hydro_active,
        topo.nr, topo.nz);
  }
  brag_sync("braginskii2d: add corner forces to nodes failed");
}

void compute_work_visc_2d(double* work_visc,
                          const double* corner_visc_r,
                          const double* corner_visc_z,
                          const double* v_r,
                          const double* v_z,
                          const std::int8_t* hydro_active,
                          const Topo2D& topo) {
  const int n_cells = topo_cell_count(topo);
  if (n_cells <= 0) {
    return;
  }
  validate_topo(topo);
  const int blocks = (n_cells + 255) / 256;
  if (topo.multiblock) {
    work_visc_multiblock_kernel<<<blocks, 256>>>(
        work_visc, corner_visc_r, corner_visc_z, v_r, v_z, hydro_active,
        topo.cell_node_csr_offsets, topo.cell_node_csr_indices,
        topo.cell_nverts, n_cells);
  } else {
    work_visc_structured_kernel<<<blocks, 256>>>(
        work_visc, corner_visc_r, corner_visc_z, v_r, v_z, hydro_active,
        topo.nr, topo.nz);
  }
  brag_sync("braginskii2d: work tally failed");
}

double compute_dt_braginskii_2d_raw(const double* x_r,
                                    const double* x_z,
                                    const double* rho,
                                    const double* Ti,
                                    const double* Te,
                                    const double* A_eff,
                                    const double* zbar,
                                    const std::int8_t* hydro_active,
                                    const Topo2D& topo,
                                    const Params& p) {
  const double inf = std::numeric_limits<double>::infinity();
  const int n_cells = topo_cell_count(topo);
  if (!p.enabled || n_cells <= 0) {
    return inf;
  }
  validate_topo(topo);
  auto* d_min = static_cast<double*>(
      core::device_scratch_acquire("brag2d:dt_min", sizeof(double)));
  brag_cuda_check(
      cudaMemcpy(d_min, &inf, sizeof(double), cudaMemcpyHostToDevice),
      "braginskii2d: init dt failed");
  const int blocks = (n_cells + 255) / 256;
  if (topo.multiblock) {
    switch (p.species) {
      case 1:
        dt_multiblock_kernel<1><<<blocks, 256>>>(
            d_min, x_r, x_z, rho, Ti, Te, A_eff, zbar, hydro_active,
            topo.cell_node_csr_offsets, topo.cell_node_csr_indices,
            topo.cell_nverts, n_cells, to_device_params(p), p.dt_safety);
        break;
      case 2:
        dt_multiblock_kernel<2><<<blocks, 256>>>(
            d_min, x_r, x_z, rho, Ti, Te, A_eff, zbar, hydro_active,
            topo.cell_node_csr_offsets, topo.cell_node_csr_indices,
            topo.cell_nverts, n_cells, to_device_params(p), p.dt_safety);
        break;
      default:
        dt_multiblock_kernel<0><<<blocks, 256>>>(
            d_min, x_r, x_z, rho, Ti, Te, A_eff, zbar, hydro_active,
            topo.cell_node_csr_offsets, topo.cell_node_csr_indices,
            topo.cell_nverts, n_cells, to_device_params(p), p.dt_safety);
        break;
    }
  } else {
    switch (p.species) {
      case 1:
        dt_structured_kernel<1><<<blocks, 256>>>(
            d_min, x_r, x_z, rho, Ti, Te, A_eff, zbar, hydro_active, topo.nr,
            topo.nz, to_device_params(p), p.dt_safety);
        break;
      case 2:
        dt_structured_kernel<2><<<blocks, 256>>>(
            d_min, x_r, x_z, rho, Ti, Te, A_eff, zbar, hydro_active, topo.nr,
            topo.nz, to_device_params(p), p.dt_safety);
        break;
      default:
        dt_structured_kernel<0><<<blocks, 256>>>(
            d_min, x_r, x_z, rho, Ti, Te, A_eff, zbar, hydro_active, topo.nr,
            topo.nz, to_device_params(p), p.dt_safety);
        break;
    }
  }
  brag_sync("braginskii2d: dt failed");
  double out = inf;
  brag_cuda_check(
      cudaMemcpy(&out, d_min, sizeof(double), cudaMemcpyDeviceToHost),
      "braginskii2d: copy dt failed");
  return out;
}

double compute_dt_braginskii_2d(const core::State& state,
                                const core::Config& cfg) {
  const double inf = std::numeric_limits<double>::infinity();
  const Params p = params_from_config(cfg);
  if (!p.enabled) {
    return inf;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  if (n_cells <= 0) {
    return inf;
  }

  const Topo2D topo = stage_topo_2d_from_state(state, cfg, n_cells, n_nodes);
  std::int8_t* d_active = stage_hydro_active_2d(state, n_cells);

  return compute_dt_braginskii_2d_raw(
      state.x_r.data(), state.x_z.data(), state.rho.data(),
      state.Ti.empty() ? nullptr : state.Ti.data(),
      state.Te.empty() ? nullptr : state.Te.data(),
      state.A_eff.empty() ? nullptr : state.A_eff.data(),
      state.zbar.empty() ? nullptr : state.zbar.data(), d_active, topo, p);
}

void ensure_visc_buffers_2d(core::State& state, const int n_cells,
                            const bool with_electron) {
  const std::size_t n_cells_size = static_cast<std::size_t>(n_cells);
  const std::size_t n_corners = n_cells_size * 4U;
  if (state.corner_force_visc_r.size() != n_corners) {
    state.corner_force_visc_r.reset(n_corners);
  }
  if (state.corner_force_visc_z.size() != n_corners) {
    state.corner_force_visc_z.reset(n_corners);
  }
  if (state.work_visc_per_cell.size() != n_cells_size) {
    state.work_visc_per_cell.reset(n_cells_size);
  }
  if (state.visc_heat_rate_per_cell.size() != n_cells_size) {
    state.visc_heat_rate_per_cell.reset(n_cells_size);
  }
  if (with_electron &&
      state.visc_heat_rate_e_per_cell.size() != n_cells_size) {
    state.visc_heat_rate_e_per_cell.reset(n_cells_size);
  }
}

void zero_visc_buffers_2d(core::State& state) {
  zero_array(state.corner_force_visc_r.data(),
             state.corner_force_visc_r.size(),
             "braginskii2d: zero corner_force_visc_r failed");
  zero_array(state.corner_force_visc_z.data(),
             state.corner_force_visc_z.size(),
             "braginskii2d: zero corner_force_visc_z failed");
  zero_array(state.work_visc_per_cell.data(), state.work_visc_per_cell.size(),
             "braginskii2d: zero work_visc_per_cell failed");
  zero_array(state.visc_heat_rate_per_cell.data(),
             state.visc_heat_rate_per_cell.size(),
             "braginskii2d: zero visc_heat_rate_per_cell failed");
  zero_array(state.visc_heat_rate_e_per_cell.data(),
             state.visc_heat_rate_e_per_cell.size(),
             "braginskii2d: zero visc_heat_rate_e_per_cell failed");
}

HistoryDiagnostics compute_history_diagnostics_2d(const core::State& state,
                                                  const core::Config& cfg) {
  HistoryDiagnostics out;
  const Params p = params_from_config(cfg);
  if (!p.enabled || state.mesh.dim != 2) {
    return out;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  if (n_cells <= 0) {
    return out;
  }
  const Topo2D topo = stage_topo_2d_from_state(state, cfg, n_cells, n_nodes);
  std::int8_t* d_active = stage_hydro_active_2d(state, n_cells);

  const std::size_t n = static_cast<std::size_t>(n_cells);
  const std::size_t bytes = n * sizeof(double);
  double* d_buf = nullptr;
  brag_cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_buf), 5 * bytes),
                  "braginskii2d: cudaMalloc diag failed");
  double* d_eta_i = d_buf;
  double* d_eta_e = d_buf + n;
  double* d_eta_eff = d_buf + 2 * n;
  double* d_heat_i = d_buf + 3 * n;
  double* d_heat_e = d_buf + 4 * n;
  DeviceParams p_run = to_device_params(p);
  DeviceParams p_phys = p_run;
  p_phys.model = 0;  // physical channels use the Braginskii formulas
  const int blocks = (n_cells + 255) / 256;
  if (topo.multiblock) {
    history_diag_2d_multiblock_kernel<<<blocks, 256>>>(
        d_eta_i, d_eta_e, d_eta_eff, d_heat_i, d_heat_e, state.x_r.data(),
        state.x_z.data(), state.v_r.data(), state.v_z.data(),
        state.rho.data(),
        state.Ti.empty() ? nullptr : state.Ti.data(),
        state.Te.empty() ? nullptr : state.Te.data(),
        state.A_eff.empty() ? nullptr : state.A_eff.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(), state.vol.data(),
        d_active, topo.cell_node_csr_offsets, topo.cell_node_csr_indices,
        topo.cell_nverts, n_cells, p.species, p_run, p_phys);
  } else {
    history_diag_2d_structured_kernel<<<blocks, 256>>>(
        d_eta_i, d_eta_e, d_eta_eff, d_heat_i, d_heat_e, state.x_r.data(),
        state.x_z.data(), state.v_r.data(), state.v_z.data(),
        state.rho.data(),
        state.Ti.empty() ? nullptr : state.Ti.data(),
        state.Te.empty() ? nullptr : state.Te.data(),
        state.A_eff.empty() ? nullptr : state.A_eff.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(), state.vol.data(),
        d_active, topo.nr, topo.nz, p.species, p_run, p_phys);
  }
  brag_sync("braginskii2d: history diag kernel failed");
  std::vector<double> h(5 * n, 0.0);
  brag_cuda_check(
      cudaMemcpy(h.data(), d_buf, 5 * bytes, cudaMemcpyDeviceToHost),
      "braginskii2d: copy diag failed");
  brag_cuda_check(cudaFree(d_buf), "braginskii2d: cudaFree diag failed");
  std::vector<double> mass_h(n, 0.0);
  brag_cuda_check(cudaMemcpy(mass_h.data(), state.mass.data(), bytes,
                             cudaMemcpyDeviceToHost),
                  "braginskii2d: copy diag mass failed");
  // Deterministic host reduction (fixed cell order, no atomics) — twin of
  // the 1D loop in braginskii_viscosity.cu::compute_history_diagnostics.
  double r_min = std::numeric_limits<double>::infinity();
  double r_max = 0.0;
  double sum_m_lnr = 0.0;
  double sum_m = 0.0;
  bool any_ratio = false;
  for (std::size_t i = 0; i < n; ++i) {
    const double ei = h[i];
    if (ei < 0.0) {
      continue;  // inactive cell
    }
    ++out.n_cells_active;
    const double ee = h[n + i];
    out.eta_i_max = std::fmax(out.eta_i_max, ei);
    out.eta_e_max = std::fmax(out.eta_e_max, ee);
    out.eta_eff_max = std::fmax(out.eta_eff_max, h[2 * n + i]);
    out.heat_rate_i_tot += h[3 * n + i];
    out.heat_rate_e_tot += h[4 * n + i];
    if (ei > 0.0 && ee > 0.0) {
      const double r = ee / ei;
      any_ratio = true;
      r_min = std::fmin(r_min, r);
      r_max = std::fmax(r_max, r);
      const double m = mass_h[i];
      sum_m_lnr += m * std::log(r);
      sum_m += m;
      if (r >= 10.0) {
        ++out.n_cells_e_dom;
      } else if (r <= 0.1) {
        ++out.n_cells_i_dom;
      } else {
        ++out.n_cells_mixed;
      }
    }
  }
  if (any_ratio) {
    out.ratio_min = r_min;
    out.ratio_max = r_max;
    out.ratio_geomean_masswt =
        (sum_m > 0.0) ? std::exp(sum_m_lnr / sum_m) : 0.0;
  }
  out.valid = true;
  return out;
}

}  // namespace tenryu::hydro::braginskii
