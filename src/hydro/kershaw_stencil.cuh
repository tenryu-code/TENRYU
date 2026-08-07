#pragma once

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "hydro/kershaw_boundary.cuh"
#include "hydro/kershaw_geometry.cuh"

namespace tenryu::hydro::kershaw {

constexpr int kStencilSize = 9;
constexpr int kC = 0;
constexpr int kE = 1;
constexpr int kW = 2;
constexpr int kN = 3;
constexpr int kS = 4;
constexpr int kNE = 5;
constexpr int kNW = 6;
constexpr int kSE = 7;
constexpr int kSW = 8;

TENRYU_HOST_DEVICE inline double harmonic_mean(const double a, const double b) {
  if ((a + b) <= 1.0e-30) {
    return 0.0;
  }
  return (2.0 * a * b) / (a + b);
}

TENRYU_HOST_DEVICE inline void apply_mmatrix_repair(double a[kStencilSize],
                                                     bool& repaired) {
  repaired = false;
  for (int k = 1; k < kStencilSize; ++k) {
    if (a[k] > 0.0) {
      a[k] = 0.0;
      repaired = true;
    }
  }
  double sum_off = 0.0;
  for (int k = 1; k < kStencilSize; ++k) {
    sum_off += a[k];
  }
  a[kC] = -sum_off;
}

TENRYU_HOST_DEVICE inline void assemble_kershaw_cell(
    double a[kStencilSize],
    const double* D_eff,
    const double* x_r,
    const double* x_z,
    const int i,
    const int j,
    const int nr,
    const int nz,
    const int bc_r_lo,
    const int bc_r_hi,
    const int bc_z_lo,
    const int bc_z_hi,
    const bool mmatrix_repair,
    int& repaired_flag) {
  for (int k = 0; k < kStencilSize; ++k) {
    a[k] = 0.0;
  }
  repaired_flag = 0;

  const int c = cell_index(i, j, nz);
  const double Dc = D_eff[c];

  const double De = (i + 1 < nr) ? D_eff[cell_index(i + 1, j, nz)] : Dc;
  const double Dw = (i > 0) ? D_eff[cell_index(i - 1, j, nz)] : Dc;
  const double Dn = (j + 1 < nz) ? D_eff[cell_index(i, j + 1, nz)] : Dc;
  const double Ds = (j > 0) ? D_eff[cell_index(i, j - 1, nz)] : Dc;

  double sigma_e = harmonic_mean(Dc, De);
  double sigma_w = harmonic_mean(Dc, Dw);
  double lambda_n = harmonic_mean(Dc, Dn);
  double lambda_s = harmonic_mean(Dc, Ds);

  sigma_e *= r_face_i(x_r, i + 1, j, nz);
  sigma_w *= r_face_i(x_r, i, j, nz);
  lambda_n *= r_face_j(x_r, i, j + 1, nz);
  lambda_s *= r_face_j(x_r, i, j, nz);

  const NodeGeometry g_sw = compute_node_geometry(x_r, x_z, i, j, nr, nz);
  const NodeGeometry g_nw = compute_node_geometry(x_r, x_z, i, j + 1, nr, nz);
  const NodeGeometry g_se = compute_node_geometry(x_r, x_z, i + 1, j, nr, nz);
  const NodeGeometry g_ne = compute_node_geometry(x_r, x_z, i + 1, j + 1, nr, nz);

  a[kE] = -0.25 * sigma_e * (g_se.A2 / g_se.J + g_ne.A2 / g_ne.J);
  a[kW] = -0.25 * sigma_w * (g_sw.A2 / g_sw.J + g_nw.A2 / g_nw.J);
  a[kN] = -0.25 * lambda_n * (g_nw.B2 / g_nw.J + g_ne.B2 / g_ne.J);
  a[kS] = -0.25 * lambda_s * (g_sw.B2 / g_sw.J + g_se.B2 / g_se.J);

  const double rho1_ne = -0.25 * (g_ne.AdotB / g_ne.J);
  const double rho2_nw = 0.25 * (g_nw.AdotB / g_nw.J);
  const double rho2_se = 0.25 * (g_se.AdotB / g_se.J);
  const double rho1_sw = -0.25 * (g_sw.AdotB / g_sw.J);

  a[kNE] = -(sigma_e * rho1_ne + lambda_n * rho1_ne);
  a[kNW] = -(sigma_w * rho2_nw + lambda_n * rho2_nw);
  a[kSE] = -(sigma_e * rho2_se + lambda_s * rho2_se);
  a[kSW] = -(sigma_w * rho1_sw + lambda_s * rho1_sw);

  double sum_off = 0.0;
  for (int k = 1; k < kStencilSize; ++k) {
    sum_off += a[k];
  }
  a[kC] = -sum_off;

  const double dx = fmax(sqrt(cell_area_from_nodes(x_r, x_z, i, j, nz)), 1.0e-30);
  constexpr double d_ext_default = 1.0e30;

  int bc = BC_INTERIOR;
  bc = normal_bc_for_neighbor(i + 1, j, nr, nz, bc_r_lo, bc_r_hi, bc_z_lo, bc_z_hi);
  fold_boundary_coefficient(a[kC],
                            a[kE],
                            i + 1,
                            j,
                            nr,
                            nz,
                            bc,
                            d_ext_default,
                            dx);

  bc = normal_bc_for_neighbor(i - 1, j, nr, nz, bc_r_lo, bc_r_hi, bc_z_lo, bc_z_hi);
  fold_boundary_coefficient(a[kC],
                            a[kW],
                            i - 1,
                            j,
                            nr,
                            nz,
                            bc,
                            d_ext_default,
                            dx);

  bc = normal_bc_for_neighbor(i, j + 1, nr, nz, bc_r_lo, bc_r_hi, bc_z_lo, bc_z_hi);
  fold_boundary_coefficient(a[kC],
                            a[kN],
                            i,
                            j + 1,
                            nr,
                            nz,
                            bc,
                            d_ext_default,
                            dx);

  bc = normal_bc_for_neighbor(i, j - 1, nr, nz, bc_r_lo, bc_r_hi, bc_z_lo, bc_z_hi);
  fold_boundary_coefficient(a[kC],
                            a[kS],
                            i,
                            j - 1,
                            nr,
                            nz,
                            bc,
                            d_ext_default,
                            dx);

  bc = normal_bc_for_neighbor(i + 1, j + 1, nr, nz, bc_r_lo, bc_r_hi, bc_z_lo, bc_z_hi);
  fold_boundary_coefficient(a[kC],
                            a[kNE],
                            i + 1,
                            j + 1,
                            nr,
                            nz,
                            bc,
                            d_ext_default,
                            dx);

  bc = normal_bc_for_neighbor(i - 1, j + 1, nr, nz, bc_r_lo, bc_r_hi, bc_z_lo, bc_z_hi);
  fold_boundary_coefficient(a[kC],
                            a[kNW],
                            i - 1,
                            j + 1,
                            nr,
                            nz,
                            bc,
                            d_ext_default,
                            dx);

  bc = normal_bc_for_neighbor(i + 1, j - 1, nr, nz, bc_r_lo, bc_r_hi, bc_z_lo, bc_z_hi);
  fold_boundary_coefficient(a[kC],
                            a[kSE],
                            i + 1,
                            j - 1,
                            nr,
                            nz,
                            bc,
                            d_ext_default,
                            dx);

  bc = normal_bc_for_neighbor(i - 1, j - 1, nr, nz, bc_r_lo, bc_r_hi, bc_z_lo, bc_z_hi);
  fold_boundary_coefficient(a[kC],
                            a[kSW],
                            i - 1,
                            j - 1,
                            nr,
                            nz,
                            bc,
                            d_ext_default,
                            dx);

  if (mmatrix_repair) {
    bool repaired = false;
    apply_mmatrix_repair(a, repaired);
    repaired_flag = repaired ? 1 : 0;
  }
}

static __global__ __launch_bounds__(256, 2) void kershaw_stencil_build_kernel(
    double* __restrict__ stencil,
    const double* __restrict__ D_eff,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int bc_r_lo,
    const int bc_r_hi,
    const int bc_z_lo,
    const int bc_z_hi,
    const int apply_mmatrix_repair,
    int* __restrict__ mmatrix_fix_count,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  int repaired = 0;
  double a[kStencilSize];
  assemble_kershaw_cell(a,
                        D_eff,
                        x_r,
                        x_z,
                        i,
                        j,
                        nr,
                        nz,
                        bc_r_lo,
                        bc_r_hi,
                        bc_z_lo,
                        bc_z_hi,
                        apply_mmatrix_repair != 0,
                        repaired);

  double* row = stencil + static_cast<std::size_t>(c) * kStencilSize;
  for (int k = 0; k < kStencilSize; ++k) {
    row[k] = a[k];
  }

  if (apply_mmatrix_repair != 0 && repaired != 0 && mmatrix_fix_count != nullptr) {
    atomicAdd(mmatrix_fix_count, 1);
  }
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

__device__ inline double symmetric_pair_power(const double* __restrict__ stencil,
                                              const double* __restrict__ Te,
                                              const double* __restrict__ rho_cv_e,
                                              const std::uint8_t* __restrict__ cell_is_void,
                                              const int c,
                                              const int n,
                                              const int slot_cn,
                                              const int slot_nc) {
  if (n == c) {
    return 0.0;
  }
  if (cell_is_void != nullptr && cell_is_void[n] != static_cast<std::uint8_t>(0)) {
    return 0.0;
  }
  if (!(rho_cv_e[n] > 0.0)) {
    return 0.0;
  }
  const double* row_c = stencil + static_cast<std::size_t>(c) * kStencilSize;
  const double* row_n = stencil + static_cast<std::size_t>(n) * kStencilSize;
  const double G_cn = -row_c[slot_cn];
  const double G_nc = -row_n[slot_nc];
  const double G = 0.5 * (G_cn + G_nc);
  return G * (Te[n] - Te[c]);
}

static __global__ __launch_bounds__(256, 4) void kershaw_alpha_pass_kernel(
    double* __restrict__ alpha_cells,
    const double* __restrict__ Te,
    const double* __restrict__ stencil,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double dt_sub,
    const double T_floor,
    const int nr,
    const int nz,
    const int floor_limiter_mode) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  alpha_cells[c] = 1.0;
  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  const int iE = (i + 1 < nr) ? (i + 1) : i;
  const int iW = (i > 0) ? (i - 1) : i;
  const int jN = (j + 1 < nz) ? (j + 1) : j;
  const int jS = (j > 0) ? (j - 1) : j;

  const int cE = cell_index(iE, j, nz);
  const int cW = cell_index(iW, j, nz);
  const int cN = cell_index(i, jN, nz);
  const int cS = cell_index(i, jS, nz);
  const int cNE = cell_index(iE, jN, nz);
  const int cNW = cell_index(iW, jN, nz);
  const int cSE = cell_index(iE, jS, nz);
  const int cSW = cell_index(iW, jS, nz);

  const double V = fmax(vol[c], 1.0e-30);
  constexpr double k4Pi = 4.0 * 3.141592653589793238462643383279502884;
  const double rho_cv = rho_cv_e[c];
  if (!(rho_cv > 0.0)) {
    return;
  }

  double net_power = 0.0;
  double outflow_sum = 0.0;
  const double pE = symmetric_pair_power(stencil, Te, rho_cv_e, cell_is_void,
                                         c, cE, kE, kW);
  net_power += pE;
  const double pW = symmetric_pair_power(stencil, Te, rho_cv_e, cell_is_void,
                                         c, cW, kW, kE);
  net_power += pW;
  const double pN = symmetric_pair_power(stencil, Te, rho_cv_e, cell_is_void,
                                         c, cN, kN, kS);
  net_power += pN;
  const double pS = symmetric_pair_power(stencil, Te, rho_cv_e, cell_is_void,
                                         c, cS, kS, kN);
  net_power += pS;
  const double pNE = symmetric_pair_power(stencil, Te, rho_cv_e, cell_is_void,
                                          c, cNE, kNE, kSW);
  net_power += pNE;
  const double pNW = symmetric_pair_power(stencil, Te, rho_cv_e, cell_is_void,
                                          c, cNW, kNW, kSE);
  net_power += pNW;
  const double pSE = symmetric_pair_power(stencil, Te, rho_cv_e, cell_is_void,
                                          c, cSE, kSE, kNW);
  net_power += pSE;
  const double pSW = symmetric_pair_power(stencil, Te, rho_cv_e, cell_is_void,
                                          c, cSW, kSW, kNE);
  net_power += pSW;
  if (floor_limiter_mode == 1) {
    outflow_sum += isfinite(pE) ? fmax(0.0, -pE) : 0.0;
    outflow_sum += isfinite(pW) ? fmax(0.0, -pW) : 0.0;
    outflow_sum += isfinite(pN) ? fmax(0.0, -pN) : 0.0;
    outflow_sum += isfinite(pS) ? fmax(0.0, -pS) : 0.0;
    outflow_sum += isfinite(pNE) ? fmax(0.0, -pNE) : 0.0;
    outflow_sum += isfinite(pNW) ? fmax(0.0, -pNW) : 0.0;
    outflow_sum += isfinite(pSE) ? fmax(0.0, -pSE) : 0.0;
    outflow_sum += isfinite(pSW) ? fmax(0.0, -pSW) : 0.0;
  }
  net_power *= k4Pi;

  const double Te_old_c = Te[c];
  constexpr double kDivEpsilon = 1.0e-30;
  double alpha = 1.0;
  if (floor_limiter_mode == 1) {
    outflow_sum *= k4Pi;
    if (!(Te_old_c > T_floor)) {
      alpha = 0.0;
    } else if (dt_sub > 0.0 && outflow_sum > 0.0) {
      const double e_above = rho_cv * V * (Te_old_c - T_floor);
      alpha = fmin(
          1.0, fmax((e_above / fmax(outflow_sum, kDivEpsilon)) / dt_sub, 0.0));
    }
    alpha_cells[c] = alpha;
    return;
  }
  if (dt_sub > 0.0 && Te_old_c > T_floor && net_power < 0.0) {
    const double energy_above_floor = rho_cv * V * (Te_old_c - T_floor);
    const double dt_safe = energy_above_floor / fmax(fabs(net_power), kDivEpsilon);
    alpha = fmin(1.0, fmax(dt_safe / dt_sub, 0.0));
  }
  alpha_cells[c] = alpha;
}

static __global__ __launch_bounds__(256, 4) void kershaw_apply_kernel(
    double* __restrict__ Te_new,
    const double* __restrict__ Te,
    const double* __restrict__ stencil,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double dt_sub,
    const double T_floor,
    const int nr,
    const int nz,
    const double* __restrict__ alpha_cells,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    Te_new[c] = Te[c];
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  const int iE = (i + 1 < nr) ? (i + 1) : i;
  const int iW = (i > 0) ? (i - 1) : i;
  const int jN = (j + 1 < nz) ? (j + 1) : j;
  const int jS = (j > 0) ? (j - 1) : j;

  const int cE = cell_index(iE, j, nz);
  const int cW = cell_index(iW, j, nz);
  const int cN = cell_index(i, jN, nz);
  const int cS = cell_index(i, jS, nz);
  const int cNE = cell_index(iE, jN, nz);
  const int cNW = cell_index(iW, jN, nz);
  const int cSE = cell_index(iE, jS, nz);
  const int cSW = cell_index(iW, jS, nz);
  const double alpha_c = (alpha_cells != nullptr) ? alpha_cells[c] : 1.0;

  const double V = fmax(vol[c], 1.0e-30);
  constexpr double k4Pi = 4.0 * 3.141592653589793238462643383279502884;
  const double rho_cv = rho_cv_e[c];
  if (!(rho_cv > 0.0)) {
    Te_new[c] = Te[c];
    return;
  }

  double net_power = 0.0;
  const double pE = symmetric_pair_power(
      stencil, Te, rho_cv_e, cell_is_void, c, cE, kE, kW);
  const double scaleE = (floor_limiter_mode == 1)
      ? ((pE > 0.0) ? alpha_cells[cE] : alpha_c)
      : fmin(alpha_c, (alpha_cells != nullptr) ? alpha_cells[cE] : 1.0);
  net_power += pE * scaleE;
  const double pW = symmetric_pair_power(
      stencil, Te, rho_cv_e, cell_is_void, c, cW, kW, kE);
  const double scaleW = (floor_limiter_mode == 1)
      ? ((pW > 0.0) ? alpha_cells[cW] : alpha_c)
      : fmin(alpha_c, (alpha_cells != nullptr) ? alpha_cells[cW] : 1.0);
  net_power += pW * scaleW;
  const double pN = symmetric_pair_power(
      stencil, Te, rho_cv_e, cell_is_void, c, cN, kN, kS);
  const double scaleN = (floor_limiter_mode == 1)
      ? ((pN > 0.0) ? alpha_cells[cN] : alpha_c)
      : fmin(alpha_c, (alpha_cells != nullptr) ? alpha_cells[cN] : 1.0);
  net_power += pN * scaleN;
  const double pS = symmetric_pair_power(
      stencil, Te, rho_cv_e, cell_is_void, c, cS, kS, kN);
  const double scaleS = (floor_limiter_mode == 1)
      ? ((pS > 0.0) ? alpha_cells[cS] : alpha_c)
      : fmin(alpha_c, (alpha_cells != nullptr) ? alpha_cells[cS] : 1.0);
  net_power += pS * scaleS;
  const double pNE = symmetric_pair_power(
      stencil, Te, rho_cv_e, cell_is_void, c, cNE, kNE, kSW);
  const double scaleNE = (floor_limiter_mode == 1)
      ? ((pNE > 0.0) ? alpha_cells[cNE] : alpha_c)
      : fmin(alpha_c, (alpha_cells != nullptr) ? alpha_cells[cNE] : 1.0);
  net_power += pNE * scaleNE;
  const double pNW = symmetric_pair_power(
      stencil, Te, rho_cv_e, cell_is_void, c, cNW, kNW, kSE);
  const double scaleNW = (floor_limiter_mode == 1)
      ? ((pNW > 0.0) ? alpha_cells[cNW] : alpha_c)
      : fmin(alpha_c, (alpha_cells != nullptr) ? alpha_cells[cNW] : 1.0);
  net_power += pNW * scaleNW;
  const double pSE = symmetric_pair_power(
      stencil, Te, rho_cv_e, cell_is_void, c, cSE, kSE, kNW);
  const double scaleSE = (floor_limiter_mode == 1)
      ? ((pSE > 0.0) ? alpha_cells[cSE] : alpha_c)
      : fmin(alpha_c, (alpha_cells != nullptr) ? alpha_cells[cSE] : 1.0);
  net_power += pSE * scaleSE;
  const double pSW = symmetric_pair_power(
      stencil, Te, rho_cv_e, cell_is_void, c, cSW, kSW, kNE);
  const double scaleSW = (floor_limiter_mode == 1)
      ? ((pSW > 0.0) ? alpha_cells[cSW] : alpha_c)
      : fmin(alpha_c, (alpha_cells != nullptr) ? alpha_cells[cSW] : 1.0);
  net_power += pSW * scaleSW;
  net_power *= k4Pi;

  const double Te_old_c = Te[c];
  double Te_trial = Te_old_c;
  const double energy_old = rho_cv * V * Te_old_c;
  const double energy_trial = energy_old + dt_sub * net_power;
  Te_trial = energy_trial / (rho_cv * V);

  if (!isfinite(Te_trial) || Te_trial < T_floor) {
    if (rho_cv > 0.0) {
      if (isfinite(Te_trial) && Te_trial < T_floor) {
        const double de = rho_cv * (T_floor - Te_trial) * V;
        atomic_add_double(E_floor, de);
      } else if (!isfinite(Te_trial) && T_floor > 0.0) {
        // Non-finite pre-clamp values cannot form a delta; track floor state directly.
        const double de = rho_cv * T_floor * V;
        atomic_add_double(E_floor, de);
      }
    }
    Te_new[c] = T_floor;
    atomicAdd(clamp_count, 1);
    return;
  }

  Te_new[c] = Te_trial;
}

inline void launch_kershaw_stencil_build(double* d_stencil,
                                         const double* d_coeff,
                                         const double* d_xr,
                                         const double* d_xz,
                                         const int nr,
                                         const int nz,
                                         const int bc_r_lo,
                                         const int bc_r_hi,
                                         const int bc_z_lo,
                                         const int bc_z_hi,
                                         const bool apply_mmatrix_repair,
                                         int* d_fix_count,
                                         cudaStream_t stream = nullptr) {
  const int n_cells = nr * nz;
  const int blocks = (n_cells + 255) / 256;
  kershaw_stencil_build_kernel<<<blocks, 256, 0, stream>>>(d_stencil,
                                                            d_coeff,
                                                            d_xr,
                                                            d_xz,
                                                            bc_r_lo,
                                                            bc_r_hi,
                                                            bc_z_lo,
                                                            bc_z_hi,
                                                            apply_mmatrix_repair ? 1 : 0,
                                                            d_fix_count,
                                                            nr,
                                                            nz);
}

inline void launch_kershaw_alpha_pass(double* d_alpha_cells,
                                      const double* d_te,
                                      const double* d_stencil,
                                      const double* d_rho_cv_e,
                                      const std::uint8_t* d_cell_is_void,
                                      const double* d_vol,
                                      const double dt_sub,
                                      const double te_floor,
                                      const int nr,
                                      const int nz,
                                      const int floor_limiter_mode = 0,
                                      cudaStream_t stream = nullptr) {
  const int n_cells = nr * nz;
  const int blocks = (n_cells + 255) / 256;
  kershaw_alpha_pass_kernel<<<blocks, 256, 0, stream>>>(d_alpha_cells,
                                                         d_te,
                                                         d_stencil,
                                                         d_rho_cv_e,
                                                         d_cell_is_void,
                                                         d_vol,
                                                         dt_sub,
                                                         te_floor,
                                                         nr,
                                                         nz,
                                                         floor_limiter_mode);
}

inline void launch_kershaw_apply(double* d_te_new,
                                 const double* d_te,
                                 const double* d_stencil,
                                 const double* d_rho_cv_e,
                                 const std::uint8_t* d_cell_is_void,
                                 const double* d_vol,
                                 const double dt_sub,
                                 const double te_floor,
                                 const int nr,
                                 const int nz,
                                 int* d_clamp_count,
                                 double* d_e_floor,
                                 const double* d_alpha_cells,
                                 const int floor_limiter_mode = 0,
                                 cudaStream_t stream = nullptr) {
  const int n_cells = nr * nz;
  const int blocks = (n_cells + 255) / 256;
  kershaw_apply_kernel<<<blocks, 256, 0, stream>>>(d_te_new,
                                                    d_te,
                                                    d_stencil,
                                                    d_rho_cv_e,
                                                    d_cell_is_void,
                                                    d_vol,
                                                    dt_sub,
                                                    te_floor,
                                                    nr,
                                                    nz,
                                                    d_alpha_cells,
                                                    d_clamp_count,
                                                    d_e_floor,
                                                    floor_limiter_mode);
}

inline void launch_kershaw_apply(double* d_te_new,
                                 const double* d_te,
                                 const double* d_stencil,
                                 const double* d_rho_cv_e,
                                 const std::uint8_t* d_cell_is_void,
                                 const double* d_vol,
                                 const double dt_sub,
                                 const double te_floor,
                                 const int nr,
                                 const int nz,
                                 int* d_clamp_count,
                                 double* d_e_floor,
                                 cudaStream_t stream = nullptr) {
  launch_kershaw_apply(d_te_new,
                       d_te,
                       d_stencil,
                       d_rho_cv_e,
                       d_cell_is_void,
                       d_vol,
                       dt_sub,
                       te_floor,
                       nr,
                       nz,
                       d_clamp_count,
                       d_e_floor,
                       nullptr,
                       0,
                       stream);
}

}  // namespace tenryu::hydro::kershaw
