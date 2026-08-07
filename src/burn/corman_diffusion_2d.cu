#include "burn/corman_diffusion_2d.cuh"

#include <cmath>
#include <cstddef>
#include <sstream>
#include <string>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "parallel/halo_exchange.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::burn {
namespace {

constexpr int kBlock = 256;
constexpr int kMaxDotBlocks = 4096;
constexpr int kCgIterationCap = 500;
constexpr double kCgRelativeTolerance = 1.0e-10;
constexpr int kTotals = 5;
constexpr int kTotalEscaped = 0;
constexpr int kTotalInflight = 1;
constexpr int kTotalSourced = 2;
constexpr int kTotalDepE = 3;
constexpr int kTotalDepI = 4;
constexpr double kPi = 3.141592653589793238462643383279502884;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__host__ __device__ inline double edge_keV(const CormanParams& p,
                                           const int k) {
  const double x = static_cast<double>(k) / static_cast<double>(p.n_groups);
  return p.E_min_keV * exp(log(p.E_max_keV / p.E_min_keV) * x);
}

__host__ __device__ inline double center_keV(const CormanParams& p,
                                             const int g) {
  return 0.5 * (edge_keV(p, g) + edge_keV(p, g + 1));
}

__host__ __device__ inline double tau_group(const CormanParams& p, const int g,
                                            const double tE,
                                            const double gamma) {
  if (!(tE > 0.0) || !isfinite(tE)) {
    return INFINITY;
  }
  const double E0 = edge_keV(p, g) * corman_detail::kKeVToErg;
  const double E1 = edge_keV(p, g + 1) * corman_detail::kKeVToErg;
  const double gamma_tE = gamma * tE;
  const double lo = gamma_tE + E0 * sqrt(E0);
  const double hi = gamma_tE + E1 * sqrt(E1);
  if (!(lo > 0.0) || !(hi > lo)) {
    return INFINITY;
  }
  return tE * (2.0 / 3.0) * log(hi / lo);
}

__host__ __device__ inline double ion_fraction_group(const CormanParams& p,
                                                     const int g,
                                                     const double tE,
                                                     const double gamma) {
  if (!(tE > 0.0) || !isfinite(tE) || !(gamma > 0.0)) {
    return 0.0;
  }
  const double E0 = edge_keV(p, g) * corman_detail::kKeVToErg;
  const double E1 = edge_keV(p, g + 1) * corman_detail::kKeVToErg;
  constexpr int n = 8;
  const double h = (E1 - E0) / static_cast<double>(n);
  double num = 0.0;
  double den = 0.0;
  for (int i = 0; i <= n; ++i) {
    const double E = E0 + h * static_cast<double>(i);
    const double sqrtE = sqrt(E);
    const double ion = gamma / sqrtE;
    const double F = E / tE + ion;
    const double w = ion / F;
    const double weight = (i == 0 || i == n) ? 1.0 : ((i % 2 == 0) ? 2.0 : 4.0);
    const double invF = 1.0 / F;
    num += weight * w * invF;
    den += weight * invF;
  }
  if (!(den > 0.0)) {
    return 0.0;
  }
  const double f = num / den;
  return (f < 0.0) ? 0.0 : ((f > 1.0) ? 1.0 : f);
}

struct BirthBinning {
  int lo = 0;
  int hi = -1;
  double w_lo = 1.0;
  double w_hi = 0.0;
  double top_excess_erg = 0.0;
};

BirthBinning birth_binning(const CormanParams& p, const double E_birth_keV) {
  BirthBinning b;
  const double c0 = center_keV(p, 0);
  const double ctop = center_keV(p, p.n_groups - 1);
  if (E_birth_keV <= c0) {
    b.lo = 0;
    b.hi = -1;
    b.w_lo = 1.0;
    return b;
  }
  if (E_birth_keV >= ctop) {
    b.lo = p.n_groups - 1;
    b.hi = -1;
    b.w_lo = 1.0;
    b.top_excess_erg =
        (E_birth_keV - ctop) * corman_detail::kKeVToErg;
    return b;
  }
  for (int g = 0; g + 1 < p.n_groups; ++g) {
    const double clo = center_keV(p, g);
    const double chi = center_keV(p, g + 1);
    if (E_birth_keV >= clo && E_birth_keV <= chi) {
      b.lo = g;
      b.hi = g + 1;
      b.w_hi = (E_birth_keV - clo) / (chi - clo);
      b.w_lo = 1.0 - b.w_hi;
      return b;
    }
  }
  return b;
}

__host__ __device__ inline double source_weight(const BirthBinning b,
                                                const int g) {
  double w = 0.0;
  if (g == b.lo) {
    w += b.w_lo;
  }
  if (g == b.hi) {
    w += b.w_hi;
  }
  return w;
}

__host__ __device__ inline double finite_or_zero(const double x) {
  return isfinite(x) ? x : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double x) {
  return isfinite(x) ? fmax(x, 0.0) : 0.0;
}

__host__ __device__ inline int cell_index(const int i, const int j, const int nz) {
  return i * nz + j;
}

__host__ __device__ inline int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

__host__ __device__ inline double burn_edge_area_rz(const double r0,
                                                    const double r1,
                                                    const double z0,
                                                    const double z1) {
  const double r0n = fmax(finite_or_zero(r0), 0.0);
  const double r1n = fmax(finite_or_zero(r1), 0.0);
  const double dr = r1n - r0n;
  const double dz_e = finite_or_zero(z1) - finite_or_zero(z0);
  const double slant = hypot(dr, dz_e);
  return kPi * (r0n + r1n) * slant;
}

struct BurnCellGeometryRZ {
  double r_left = 0.0;
  double r_right = 0.0;
  double z_bottom = 0.0;
  double z_top = 0.0;
  double r_center = 0.0;
  double z_center = 0.0;
  double area_iL = 0.0;
  double area_iR = 0.0;
  double area_jB = 0.0;
  double area_jT = 0.0;
  double V_op = 0.0;
};

__device__ inline BurnCellGeometryRZ burn_rect_cell_geometry_v2(
    const double* __restrict__ x_r, const double* __restrict__ x_z,
    const double* __restrict__ vol, const int i, const int j, const int nz) {
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  const double r00 = finite_or_zero(x_r[n00]);
  const double r10 = finite_or_zero(x_r[n10]);
  const double r11 = finite_or_zero(x_r[n11]);
  const double r01 = finite_or_zero(x_r[n01]);
  const double z00 = finite_or_zero(x_z[n00]);
  const double z10 = finite_or_zero(x_z[n10]);
  const double z11 = finite_or_zero(x_z[n11]);
  const double z01 = finite_or_zero(x_z[n01]);

  BurnCellGeometryRZ geom;
  geom.r_left = fmax(0.5 * (r00 + r01), 0.0);
  geom.r_right = fmax(0.5 * (r10 + r11), 0.0);
  geom.z_bottom = 0.5 * (z00 + z10);
  geom.z_top = 0.5 * (z01 + z11);
  geom.r_center = 0.25 * (r00 + r10 + r11 + r01);
  geom.z_center = 0.25 * (z00 + z10 + z11 + z01);
  geom.area_iL = burn_edge_area_rz(r00, r01, z00, z01);
  geom.area_iR = burn_edge_area_rz(r10, r11, z10, z11);
  geom.area_jB = burn_edge_area_rz(r00, r10, z00, z10);
  geom.area_jT = burn_edge_area_rz(r01, r11, z01, z11);
  const int c = cell_index(i, j, nz);
  const double V_input = nonnegative_finite(vol[c]);
  const double dz = fmax(fabs(geom.z_top - geom.z_bottom), 0.0);
  const double area_z_mid =
      kPi * fmax(geom.r_right * geom.r_right - geom.r_left * geom.r_left, 0.0);
  geom.V_op = (V_input > 0.0) ? V_input : area_z_mid * dz;
  return geom;
}

__global__ void compute_coefficients_kernel(
    CormanParams p, int n_cells, double species_A, double species_Z,
    double E_birth_keV, const double* __restrict__ rho,
    const double* __restrict__ Te_eV, const double* __restrict__ Ti_eV,
    const double* __restrict__ ne, double* __restrict__ tE,
    double* __restrict__ gamma, double* __restrict__ lnL_I) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double lnLe = (p.lnL_e > 0.0) ? p.lnL_e
                                      : corman_electron_log(Te_eV[c], ne[c]);
  const double lnLI = (p.lnL_I > 0.0)
                          ? p.lnL_I
                          : corman_ion_log(species_A, species_Z, E_birth_keV,
                                           rho[c], Te_eV[c], Ti_eV[c], ne[c]);
  tE[c] = corman_tE(species_A, species_Z, Te_eV[c], ne[c], lnLe);
  gamma[c] = corman_gamma(species_A, species_Z, rho[c], lnLI);
  lnL_I[c] = lnLI;
}

__device__ inline double corman2d_face_D(
    const CormanParams& p, const int g, const double species_A,
    const double species_Z, const int iL, const int jL, const int iR,
    const int jR, const int face_n0, const int face_n1, const int nz,
    const int n_cells,
    const double* __restrict__ x_r, const double* __restrict__ x_z,
    const double* __restrict__ vol, const double* __restrict__ rho,
    const double* __restrict__ N_old, const double* __restrict__ lnL_I) {
  const int cL = cell_index(iL, jL, nz);
  const bool boundary = iR < 0 || jR < 0;
  const int cR = boundary ? cL : cell_index(iR, jR, nz);
  const double Nl = N_old[g * n_cells + cL];
  const double Nr = boundary ? Nl : N_old[g * n_cells + cR];
  const double N_f = boundary ? Nl : 0.5 * (Nl + Nr);
  const double rho_f = boundary ? rho[cL] : 0.5 * (rho[cL] + rho[cR]);
  const double lnL_f = boundary ? lnL_I[cL] : 0.5 * (lnL_I[cL] + lnL_I[cR]);
  const double E = center_keV(p, g);
  const double lambda = corman_lambda(species_A, species_Z, E, rho_f, lnL_f);
  const double m_s = species_A * corman_detail::kProtonMassG;
  const double v = sqrt(2.0 * E * corman_detail::kKeVToErg / m_s);
  if (!(lambda > 0.0) || !isfinite(lambda)) {
    return 0.0;
  }
  if (!(N_f > 1.0e-30)) {
    return v * lambda / 3.0;
  }
  double g_n = 0.0;
  double n_hat_r = 0.0;
  double n_hat_z = 0.0;
  if (!boundary) {
    const BurnCellGeometryRZ geomL =
        burn_rect_cell_geometry_v2(x_r, x_z, vol, iL, jL, nz);
    const BurnCellGeometryRZ geomR =
        burn_rect_cell_geometry_v2(x_r, x_z, vol, iR, jR, nz);
    const double dr = geomR.r_center - geomL.r_center;
    const double dz = geomR.z_center - geomL.z_center;
    const double dist = fmax(hypot(dr, dz), 1.0e-300);
    g_n = (Nr - Nl) / dist;
    n_hat_r = dr / dist;
    n_hat_z = dz / dist;
  }
  const double r_f = 0.5 * (x_r[face_n0] + x_r[face_n1]);
  const double z_f = 0.5 * (x_z[face_n0] + x_z[face_n1]);
  const double R_f = fmax(sqrt(r_f * r_f + z_f * z_f), 1.0e-300);
  const double proj = n_hat_r * (r_f / R_f) + n_hat_z * (z_f / R_f);
  const double inv_mubar =
      1.0 + 3.0 * exp(-0.5 * lambda *
                      fabs(g_n * proj / fmax(N_f, 1.0e-30) - 3.6 / R_f));
  const double denom = 3.0 / lambda + fabs(g_n) * inv_mubar / N_f;
  if (!(denom > 0.0)) {
    return v * lambda / 3.0;
  }
  return v / denom;
}

__device__ inline double center_distance(const BurnCellGeometryRZ& a,
                                         const BurnCellGeometryRZ& b) {
  return fmax(hypot(b.r_center - a.r_center, b.z_center - a.z_center),
              1.0e-300);
}

__device__ inline double boundary_sink(
    const CormanParams& p, const int g, const double species_A,
    const double species_Z, const int i, const int j, const int face_n0,
    const int face_n1, const double area, const double dt_s, const int nz,
    const int n_cells,
    const double* __restrict__ x_r, const double* __restrict__ x_z,
    const double* __restrict__ vol, const double* __restrict__ rho,
    const double* __restrict__ N_old, const double* __restrict__ lnL_I,
    const BurnCellGeometryRZ& geom) {
  const int c = cell_index(i, j, nz);
  const double lambda = corman_lambda(species_A, species_Z, center_keV(p, g),
                                      rho[c], lnL_I[c]);
  if (!(lambda > 0.0) || !isfinite(lambda)) {
    return 0.0;
  }
  const double D = corman2d_face_D(
      p, g, species_A, species_Z, i, j, -1, -1, face_n0, face_n1, nz,
      n_cells, x_r, x_z, vol, rho, N_old, lnL_I);
  if (!(D > 0.0)) {
    return 0.0;
  }
  const double r_f = 0.5 * (x_r[face_n0] + x_r[face_n1]);
  const double z_f = 0.5 * (x_z[face_n0] + x_z[face_n1]);
  const double R_f = fmax(sqrt(r_f * r_f + z_f * z_f), 1.0e-300);
  const double dist_half =
      fmax(hypot(r_f - geom.r_center, z_f - geom.z_center), 1.0e-300);
  const double inv_L = 1.0 / (0.71 * lambda) + 1.0 / R_f;
  const double L = 1.0 / inv_L;
  return dt_s * area * D / (dist_half + L);
}

__global__ void assemble_group_kernel(
    CormanParams p, BirthBinning birth, int g, int nr, int nz,
    double species_A, double species_Z, const double* __restrict__ x_r,
    const double* __restrict__ x_z, const double* __restrict__ vol,
    const double* __restrict__ rho, const double* __restrict__ S_birth,
    const double* __restrict__ N_old, const double* __restrict__ N,
    const double* __restrict__ tE, const double* __restrict__ gamma,
    const double* __restrict__ lnL_I, double dt_s, Corman2DBc bc,
    double* __restrict__ diag, double* __restrict__ aL,
    double* __restrict__ aR, double* __restrict__ aB,
    double* __restrict__ aT, double* __restrict__ rhs,
    double* __restrict__ sink_bnd) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c % nz;
  const BurnCellGeometryRZ geom =
      burn_rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
  double d = 0.0;
  double l = 0.0;
  double r = 0.0;
  double btm = 0.0;
  double top = 0.0;
  double sink = 0.0;

  if (i > 0) {
    const BurnCellGeometryRZ other =
        burn_rect_cell_geometry_v2(x_r, x_z, vol, i - 1, j, nz);
    const int n0 = node_index(i, j, nz);
    const int n1 = node_index(i, j + 1, nz);
    const double D = corman2d_face_D(p, g, species_A, species_Z, i - 1, j,
                                     i, j, n0, n1, nz, n_cells, x_r, x_z, vol, rho,
                                     N_old, lnL_I);
    const double a = dt_s * geom.area_iL * D / center_distance(other, geom);
    d += a;
    l = -a;
  } else if (bc.r_inner == 1) {
    sink += boundary_sink(p, g, species_A, species_Z, i, j,
                          node_index(i, j, nz), node_index(i, j + 1, nz),
                          geom.area_iL, dt_s, nz, n_cells, x_r, x_z, vol, rho, N_old,
                          lnL_I, geom);
  }
  if (i + 1 < nr) {
    const BurnCellGeometryRZ other =
        burn_rect_cell_geometry_v2(x_r, x_z, vol, i + 1, j, nz);
    const int n0 = node_index(i + 1, j, nz);
    const int n1 = node_index(i + 1, j + 1, nz);
    const double D = corman2d_face_D(p, g, species_A, species_Z, i, j,
                                     i + 1, j, n0, n1, nz, n_cells, x_r, x_z, vol,
                                     rho, N_old, lnL_I);
    const double a = dt_s * geom.area_iR * D / center_distance(geom, other);
    d += a;
    r = -a;
  } else if (bc.r_outer == 1) {
    sink += boundary_sink(p, g, species_A, species_Z, i, j,
                          node_index(i + 1, j, nz),
                          node_index(i + 1, j + 1, nz), geom.area_iR, dt_s,
                          nz, n_cells, x_r, x_z, vol, rho, N_old, lnL_I, geom);
  }
  if (j > 0) {
    const BurnCellGeometryRZ other =
        burn_rect_cell_geometry_v2(x_r, x_z, vol, i, j - 1, nz);
    const int n0 = node_index(i, j, nz);
    const int n1 = node_index(i + 1, j, nz);
    const double D = corman2d_face_D(p, g, species_A, species_Z, i, j - 1,
                                     i, j, n0, n1, nz, n_cells, x_r, x_z, vol, rho,
                                     N_old, lnL_I);
    const double a = dt_s * geom.area_jB * D / center_distance(other, geom);
    d += a;
    btm = -a;
  } else if (bc.z_bottom == 1) {
    sink += boundary_sink(p, g, species_A, species_Z, i, j,
                          node_index(i, j, nz), node_index(i + 1, j, nz),
                          geom.area_jB, dt_s, nz, n_cells, x_r, x_z, vol, rho, N_old,
                          lnL_I, geom);
  }
  if (j + 1 < nz) {
    const BurnCellGeometryRZ other =
        burn_rect_cell_geometry_v2(x_r, x_z, vol, i, j + 1, nz);
    const int n0 = node_index(i, j + 1, nz);
    const int n1 = node_index(i + 1, j + 1, nz);
    const double D = corman2d_face_D(p, g, species_A, species_Z, i, j,
                                     i, j + 1, n0, n1, nz, n_cells, x_r, x_z, vol,
                                     rho, N_old, lnL_I);
    const double a = dt_s * geom.area_jT * D / center_distance(geom, other);
    d += a;
    top = -a;
  } else if (bc.z_top == 1) {
    sink += boundary_sink(p, g, species_A, species_Z, i, j,
                          node_index(i, j + 1, nz),
                          node_index(i + 1, j + 1, nz), geom.area_jT, dt_s,
                          nz, n_cells, x_r, x_z, vol, rho, N_old, lnL_I, geom);
  }
  d += sink;
  const double tau = tau_group(p, g, tE[c], gamma[c]);
  const double transfer =
      (tau > 0.0 && isfinite(tau)) ? dt_s / tau : 0.0;
  d += geom.V_op * (1.0 + transfer);
  double q = geom.V_op *
             (N_old[g * n_cells + c] +
              dt_s * S_birth[c] * source_weight(birth, g));
  if (g + 1 < p.n_groups) {
    const double tau_up = tau_group(p, g + 1, tE[c], gamma[c]);
    if (tau_up > 0.0 && isfinite(tau_up)) {
      q += geom.V_op * dt_s * N[(g + 1) * n_cells + c] / tau_up;
    }
  }
  diag[c] = d;
  aL[c] = l;
  aR[c] = r;
  aB[c] = btm;
  aT[c] = top;
  rhs[c] = q;
  sink_bnd[c] = sink;
}

__global__ void stencil_matvec_kernel(
    int nr, int nz, const double* __restrict__ diag,
    const double* __restrict__ aL, const double* __restrict__ aR,
    const double* __restrict__ aB, const double* __restrict__ aT,
    const double* __restrict__ x, double* __restrict__ y) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = nr * nz;
  if (c >= n) {
    return;
  }
  const int i = c / nz;
  const int j = c % nz;
  double value = diag[c] * x[c];
  if (i > 0) value += aL[c] * x[c - nz];
  if (i + 1 < nr) value += aR[c] * x[c + nz];
  if (j > 0) value += aB[c] * x[c - 1];
  if (j + 1 < nz) value += aT[c] * x[c + 1];
  y[c] = value;
}

__global__ void residual_kernel(int n, const double* __restrict__ rhs,
                                const double* __restrict__ Ax,
                                double* __restrict__ r) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) r[c] = rhs[c] - Ax[c];
}

__global__ void jacobi_kernel(int n, const double* __restrict__ r,
                              const double* __restrict__ diag,
                              double* __restrict__ z) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) z[c] = r[c] / diag[c];
}

__global__ void copy_kernel(int n, const double* __restrict__ src,
                            double* __restrict__ dst) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) dst[c] = src[c];
}

__global__ void cg_update_kernel(int n, double alpha,
                                 const double* __restrict__ p,
                                 const double* __restrict__ Ap,
                                 double* __restrict__ x,
                                 double* __restrict__ r) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) {
    x[c] += alpha * p[c];
    r[c] -= alpha * Ap[c];
  }
}

__global__ void p_update_kernel(int n, double beta,
                                const double* __restrict__ z,
                                double* __restrict__ p) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n) p[c] = z[c] + beta * p[c];
}

__global__ void dot_blocks_kernel(int n, const double* __restrict__ a,
                                  const double* __restrict__ b,
                                  double* __restrict__ partials) {
  __shared__ double sum[kBlock];
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  sum[threadIdx.x] = (c < n) ? a[c] * b[c] : 0.0;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) partials[blockIdx.x] = sum[0];
}

__global__ void dot_final_kernel(int n, const double* __restrict__ partials,
                                 double* __restrict__ result) {
  __shared__ double sum[kBlock];
  double value = 0.0;
  for (int k = threadIdx.x; k < n; k += kBlock) value += partials[k];
  sum[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) sum[threadIdx.x] += sum[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) result[0] = sum[0];
}

double device_dot(const int n, const double* a, const double* b,
                  double* partials, double* scalar, cudaStream_t stream) {
  const int blocks = (n + kBlock - 1) / kBlock;
  TENRYU_ASSERT(blocks <= kMaxDotBlocks,
                "Corman 2D CG dot reduction exceeds 4096 blocks");
  dot_blocks_kernel<<<blocks, kBlock, 0, stream>>>(n, a, b, partials);
  cuda_check(cudaGetLastError(), "Corman 2D CG dot block kernel launch failed");
  dot_final_kernel<<<1, kBlock, 0, stream>>>(blocks, partials, scalar);
  cuda_check(cudaGetLastError(), "Corman 2D CG dot final kernel launch failed");
  double host = 0.0;
  cuda_check(cudaMemcpyAsync(&host, scalar, sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "Corman 2D CG dot copy failed");
  cuda_check(cudaStreamSynchronize(stream),
             "Corman 2D CG dot synchronize failed");
  return host;
}

void solve_group_cg(
    const int g, const int nr, const int nz, const double species_A,
    const double species_Z, const double* diag, const double* aL,
    const double* aR, const double* aB, const double* aT, const double* rhs,
    const double* N_old, double* x, double* r, double* z, double* pvec,
    double* Ap, double* partials, double* scalar, cudaStream_t stream,
    const parallel::PartitionInfo* part, parallel::CommBuffers* bufs,
    const int c_begin, const int c_end) {
  const int n = nr * nz;
  const int grid = (n + kBlock - 1) / kBlock;
  // Distributed CG (Option C, FLD-2D pattern): dots run over the OWNED
  // contiguous span + Allreduce(SUM) so alpha/beta/convergence are
  // rank-uniform; the search direction's ghost i-planes are exchanged
  // before every matvec (and x once at init), keeping owned rows exact
  // while far rows iterate on garbage-but-finite values nobody reads.
  const bool cg_mpi = part != nullptr && part->n_ranks > 1;
  const int dot_off = cg_mpi ? c_begin : 0;
  const int dot_n = cg_mpi ? (c_end - c_begin) : n;
  const auto gdot = [&](const double* a, const double* b) {
    double v = device_dot(dot_n, a + dot_off, b + dot_off, partials, scalar,
                          stream);
    if (cg_mpi) {
      v = parallel::Reduction(part->n_ranks).allreduce_sum(v);
    }
    return v;
  };
  const double rhs2 = gdot(rhs, rhs);
  if (rhs2 == 0.0) {
    cuda_check(cudaMemsetAsync(x, 0, static_cast<std::size_t>(n) * sizeof(double),
                               stream),
               "Corman 2D CG zero solution failed");
    return;
  }
  copy_kernel<<<grid, kBlock, 0, stream>>>(n, N_old + g * n, x);
  if (cg_mpi) {
    cuda_check(cudaStreamSynchronize(stream),
               "Corman 2D CG x0 sync before exchange failed");
    parallel::exchange_cell_strips_scaled(*part, *bufs, x, 1, 17);
  }
  stencil_matvec_kernel<<<grid, kBlock, 0, stream>>>(
      nr, nz, diag, aL, aR, aB, aT, x, Ap);
  residual_kernel<<<grid, kBlock, 0, stream>>>(n, rhs, Ap, r);
  cuda_check(cudaGetLastError(), "Corman 2D CG initialization failed");
  double r2 = gdot(r, r);
  const double rhs_norm = sqrt(rhs2);
  const double tolerance = kCgRelativeTolerance * rhs_norm;
  if (sqrt(r2) <= tolerance) return;
  jacobi_kernel<<<grid, kBlock, 0, stream>>>(n, r, diag, z);
  copy_kernel<<<grid, kBlock, 0, stream>>>(n, z, pvec);
  cuda_check(cudaGetLastError(), "Corman 2D CG preconditioner launch failed");
  double rz = gdot(r, z);
  bool converged = false;
  for (int iter = 0; iter < kCgIterationCap; ++iter) {
    if (cg_mpi) {
      cuda_check(cudaStreamSynchronize(stream),
                 "Corman 2D CG p sync before exchange failed");
      parallel::exchange_cell_strips_scaled(*part, *bufs, pvec, 1, 18);
    }
    stencil_matvec_kernel<<<grid, kBlock, 0, stream>>>(
        nr, nz, diag, aL, aR, aB, aT, pvec, Ap);
    cuda_check(cudaGetLastError(), "Corman 2D CG matvec launch failed");
    const double pAp = gdot(pvec, Ap);
    const double alpha = rz / pAp;
    cg_update_kernel<<<grid, kBlock, 0, stream>>>(n, alpha, pvec, Ap, x, r);
    cuda_check(cudaGetLastError(), "Corman 2D CG update launch failed");
    r2 = gdot(r, r);
    if (sqrt(r2) <= tolerance) {
      converged = true;
      break;
    }
    jacobi_kernel<<<grid, kBlock, 0, stream>>>(n, r, diag, z);
    cuda_check(cudaGetLastError(), "Corman 2D CG Jacobi launch failed");
    const double rz_new = gdot(r, z);
    const double beta = rz_new / rz;
    p_update_kernel<<<grid, kBlock, 0, stream>>>(n, beta, z, pvec);
    cuda_check(cudaGetLastError(), "Corman 2D CG direction launch failed");
    rz = rz_new;
  }
  if (!converged) {
    std::ostringstream oss;
    oss << "Corman 2D CG failed for group " << g
        << ", relative residual=" << sqrt(r2) / rhs_norm
        << ", species A=" << species_A << " Z=" << species_Z;
    TENRYU_ASSERT(false, oss.str());
  }
}

__global__ void source_tally_kernel(
    CormanParams p, BirthBinning birth, int nr, int nz, double E_birth_keV,
    const double* __restrict__ x_r, const double* __restrict__ x_z,
    const double* __restrict__ vol, const double* __restrict__ S_birth,
    double dt_s, double* __restrict__ dep_e, double* __restrict__ totals) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  (void)p;
  const double E_birth_erg = E_birth_keV * corman_detail::kKeVToErg;
  double sourced = 0.0;
  double dep_e_step = 0.0;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = cell_index(i, j, nz);
      const double V = burn_rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz).V_op;
      const double n_birth = S_birth[c] * dt_s * V;
      sourced += n_birth * E_birth_erg;
      if (birth.top_excess_erg > 0.0) {
        const double e = n_birth * birth.top_excess_erg;
        dep_e[c] += e;
        dep_e_step += e;
      }
    }
  }
  totals[kTotalSourced] += sourced;
  totals[kTotalDepE] += dep_e_step;
}

__global__ void tally_group_kernel(
    CormanParams p, int g, int nr, int nz,
    const double* __restrict__ x_r, const double* __restrict__ x_z,
    const double* __restrict__ vol, const double* __restrict__ N_group,
    const double* __restrict__ tE, const double* __restrict__ gamma,
    const double* __restrict__ sink_bnd, double dt_s,
    double* __restrict__ dep_e,
    double* __restrict__ dep_i, double* __restrict__ totals,
    int i_begin, int i_end) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const double E_g = center_keV(p, g) * corman_detail::kKeVToErg;
  const double E_lower =
      (g > 0) ? center_keV(p, g - 1) * corman_detail::kKeVToErg : 0.0;
  double dep_e_step = 0.0;
  double dep_i_step = 0.0;
  double escaped = 0.0;
  // Owned i-window (Option C): far-region N is stale-finite garbage and
  // must not reach the deposition/escape tallies (owned partials are
  // completed by the driver budget Allreduce).
  for (int i = i_begin; i < i_end; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = cell_index(i, j, nz);
      const double V = burn_rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz).V_op;
      const double tau = tau_group(p, g, tE[c], gamma[c]);
      const double transfer =
          (tau > 0.0 && isfinite(tau)) ? dt_s / tau : 0.0;
      const double n_transfer = N_group[c] * transfer * V;
      if (g > 0) {
        const double dE = E_g - E_lower;
        const double fi = ion_fraction_group(p, g, tE[c], gamma[c]);
        const double e_i = n_transfer * dE * fi;
        const double e_e = n_transfer * dE * (1.0 - fi);
        dep_e[c] += e_e;
        dep_i[c] += e_i;
        dep_e_step += e_e;
        dep_i_step += e_i;
      } else {
        const double e_i = n_transfer * E_g;
        dep_i[c] += e_i;
        dep_i_step += e_i;
      }
    }
  }
  const int n_cells = nr * nz;
  for (int c = 0; c < n_cells; ++c) {
    escaped += N_group[c] * sink_bnd[c] * E_g;
  }
  totals[kTotalEscaped] += escaped;
  totals[kTotalDepE] += dep_e_step;
  totals[kTotalDepI] += dep_i_step;
}

__global__ void inflight_kernel(
    CormanParams p, int nr, int nz, const double* __restrict__ x_r,
    const double* __restrict__ x_z, const double* __restrict__ vol,
    const double* __restrict__ N, double* __restrict__ totals,
    int i_begin, int i_end) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  double inflight = 0.0;
  const int n_cells = nr * nz;
  for (int g = 0; g < p.n_groups; ++g) {
    const double E = center_keV(p, g) * corman_detail::kKeVToErg;
    for (int i = i_begin; i < i_end; ++i) {
      for (int j = 0; j < nz; ++j) {
        const int c = cell_index(i, j, nz);
        const double V = burn_rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz).V_op;
        inflight += N[g * n_cells + c] * E * V;
      }
    }
  }
  totals[kTotalInflight] = inflight;
}

__global__ void scale_rows_by_cell_kernel(double* __restrict__ dst,
                                          const double* __restrict__ src,
                                          const double* __restrict__ rho,
                                          int G, int J, bool divide) {
  const std::size_t total =
      static_cast<std::size_t>(G) * static_cast<std::size_t>(J);
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int j = static_cast<int>(idx % static_cast<std::size_t>(J));
  const double rho_j = rho[j];
  if (divide) {
    dst[idx] = (rho_j > 0.0) ? src[idx] / rho_j : 0.0;
  } else {
    dst[idx] = src[idx] * rho_j;
  }
}

}  // namespace

void corman2d_scale_rows_by_cell(double* dst, const double* src,
                                 const double* rho, const int G, const int J) {
  if (G <= 0 || J <= 0) return;
  TENRYU_ASSERT(dst != nullptr && src != nullptr && rho != nullptr,
                "Corman 2D scale_rows_by_cell received a null pointer");
  const std::size_t total =
      static_cast<std::size_t>(G) * static_cast<std::size_t>(J);
  const int grid = static_cast<int>((total + kBlock - 1U) / kBlock);
  scale_rows_by_cell_kernel<<<grid, kBlock>>>(dst, src, rho, G, J, false);
  cuda_check(cudaGetLastError(),
             "Corman 2D scale_rows_by_cell kernel launch failed");
}

void corman2d_divide_rows_by_cell(double* dst, const double* src,
                                  const double* rho, const int G, const int J) {
  if (G <= 0 || J <= 0) return;
  TENRYU_ASSERT(dst != nullptr && src != nullptr && rho != nullptr,
                "Corman 2D divide_rows_by_cell received a null pointer");
  const std::size_t total =
      static_cast<std::size_t>(G) * static_cast<std::size_t>(J);
  const int grid = static_cast<int>((total + kBlock - 1U) / kBlock);
  scale_rows_by_cell_kernel<<<grid, kBlock>>>(dst, src, rho, G, J, true);
  cuda_check(cudaGetLastError(),
             "Corman 2D divide_rows_by_cell kernel launch failed");
}

CormanStepResult corman_diffusion_2d_step(
    const CormanParams& p_in, const int nr, const int nz,
    const double species_A, const double species_Z,
    const double E_birth_keV, const double* x_r_dev, const double* x_z_dev,
    const double* vol_dev, const double* rho_dev, const double* Te_eV_dev,
    const double* Ti_eV_dev, const double* ne_dev,
    const double* S_birth_dev, const double dt_s, double* N_dev,
    double* dep_e_dev, double* dep_i_dev, const Corman2DBc& bc,
    cudaStream_t stream, const parallel::PartitionInfo* mpi_part,
    parallel::CommBuffers* mpi_bufs, const int mpi_c_begin,
    const int mpi_c_end) {
  CormanStepResult result;
  if (nr <= 0 || nz <= 0) return result;
  const int n_cells = nr * nz;
  const bool corman_mpi = mpi_part != nullptr && mpi_part->n_ranks > 1;
  const int win_i_begin = corman_mpi ? (mpi_c_begin / nz) : 0;
  const int win_i_end = corman_mpi ? (mpi_c_end / nz) : nr;
  CormanParams p = p_in;
  TENRYU_ASSERT(p.n_groups > 0, "Corman 2D n_groups must be positive");
  TENRYU_ASSERT(p.E_min_keV > 0.0 && p.E_max_keV > p.E_min_keV,
                "Corman 2D energy bounds are invalid");
  TENRYU_ASSERT(species_A > 0.0 && species_Z > 0.0,
                "Corman 2D species must have positive A and Z");
  TENRYU_ASSERT(E_birth_keV > 0.0 && dt_s >= 0.0,
                "Corman 2D birth energy and dt are invalid");
  TENRYU_ASSERT(E_birth_keV >= center_keV(p, 0),
                "Corman 2D birth energy below bottom center is unsupported");
  TENRYU_ASSERT(x_r_dev != nullptr && x_z_dev != nullptr &&
                    vol_dev != nullptr && rho_dev != nullptr &&
                    Te_eV_dev != nullptr && Ti_eV_dev != nullptr &&
                    ne_dev != nullptr && S_birth_dev != nullptr &&
                    N_dev != nullptr && dep_e_dev != nullptr &&
                    dep_i_dev != nullptr,
                "Corman 2D received a null device pointer");
  TENRYU_ASSERT((bc.r_inner == 0 || bc.r_inner == 1) &&
                    (bc.r_outer == 0 || bc.r_outer == 1) &&
                    (bc.z_bottom == 0 || bc.z_bottom == 1) &&
                    (bc.z_top == 0 || bc.z_top == 1),
                "Corman 2D boundary class is invalid");

  const std::size_t cells = static_cast<std::size_t>(n_cells);
  const std::size_t all = static_cast<std::size_t>(p.n_groups) * cells;
  auto acquire = [](const char* tag, const std::size_t count) {
    return static_cast<double*>(
        core::device_scratch_acquire(tag, count * sizeof(double)));
  };
  double* N_old = acquire("burn:corman2d:N_old", all);
  double* tE = acquire("burn:corman2d:tE", cells);
  double* gamma = acquire("burn:corman2d:gamma", cells);
  double* lnL_I = acquire("burn:corman2d:lnL_I", cells);
  double* diag = acquire("burn:corman2d:diag", cells);
  double* aL = acquire("burn:corman2d:aL", cells);
  double* aR = acquire("burn:corman2d:aR", cells);
  double* aB = acquire("burn:corman2d:aB", cells);
  double* aT = acquire("burn:corman2d:aT", cells);
  double* rhs = acquire("burn:corman2d:rhs", cells);
  double* sink_bnd = acquire("burn:corman2d:sink_bnd", cells);
  double* r = acquire("burn:corman2d:r", cells);
  double* z = acquire("burn:corman2d:z", cells);
  double* pvec = acquire("burn:corman2d:pvec", cells);
  double* Ap = acquire("burn:corman2d:Ap", cells);
  const int dot_blocks = (n_cells + kBlock - 1) / kBlock;
  TENRYU_ASSERT(dot_blocks <= kMaxDotBlocks,
                "Corman 2D CG dot reduction exceeds 4096 blocks");
  double* partials = acquire("burn:corman2d:dot_partials",
                             static_cast<std::size_t>(dot_blocks));
  double* scalar = acquire("burn:corman2d:dot_scalar", 1U);
  double* totals = acquire("burn:corman2d:totals", kTotals);

  cuda_check(cudaMemsetAsync(totals, 0, kTotals * sizeof(double), stream),
             "Corman 2D zero totals failed");
  cuda_check(cudaMemcpyAsync(N_old, N_dev, all * sizeof(double),
                             cudaMemcpyDeviceToDevice, stream),
             "Corman 2D snapshot N failed");
  const int grid = (n_cells + kBlock - 1) / kBlock;
  compute_coefficients_kernel<<<grid, kBlock, 0, stream>>>(
      p, n_cells, species_A, species_Z, E_birth_keV, rho_dev, Te_eV_dev,
      Ti_eV_dev, ne_dev, tE, gamma, lnL_I);
  cuda_check(cudaGetLastError(), "Corman 2D coefficient kernel launch failed");
  const BirthBinning birth = birth_binning(p, E_birth_keV);
  source_tally_kernel<<<1, 1, 0, stream>>>(
      p, birth, nr, nz, E_birth_keV, x_r_dev, x_z_dev, vol_dev, S_birth_dev,
      dt_s, dep_e_dev, totals);
  cuda_check(cudaGetLastError(), "Corman 2D source tally launch failed");

  for (int g = p.n_groups - 1; g >= 0; --g) {
    assemble_group_kernel<<<grid, kBlock, 0, stream>>>(
        p, birth, g, nr, nz, species_A, species_Z, x_r_dev, x_z_dev, vol_dev,
        rho_dev, S_birth_dev, N_old, N_dev, tE, gamma, lnL_I, dt_s, bc, diag,
        aL, aR, aB, aT, rhs, sink_bnd);
    cuda_check(cudaGetLastError(), "Corman 2D assembly kernel launch failed");
    double* const N_group = N_dev + static_cast<std::size_t>(g) * cells;
    solve_group_cg(g, nr, nz, species_A, species_Z, diag, aL, aR, aB, aT,
                   rhs, N_old, N_group, r, z, pvec, Ap, partials, scalar,
                   stream, mpi_part, mpi_bufs, mpi_c_begin, mpi_c_end);
    tally_group_kernel<<<1, 1, 0, stream>>>(
        p, g, nr, nz, x_r_dev, x_z_dev, vol_dev, N_group, tE, gamma,
        sink_bnd, dt_s, dep_e_dev, dep_i_dev, totals, win_i_begin,
        win_i_end);
    cuda_check(cudaGetLastError(), "Corman 2D tally kernel launch failed");
  }
  inflight_kernel<<<1, 1, 0, stream>>>(p, nr, nz, x_r_dev, x_z_dev, vol_dev,
                                       N_dev, totals, win_i_begin,
                                       win_i_end);
  cuda_check(cudaGetLastError(), "Corman 2D inflight kernel launch failed");
  double totals_h[kTotals] = {0.0, 0.0, 0.0, 0.0, 0.0};
  cuda_check(cudaMemcpyAsync(totals_h, totals, sizeof(totals_h),
                             cudaMemcpyDeviceToHost, stream),
             "Corman 2D copy totals failed");
  cuda_check(cudaStreamSynchronize(stream),
             "Corman 2D stream synchronize failed");
  result.escaped_erg = totals_h[kTotalEscaped];
  result.inflight_erg = totals_h[kTotalInflight];
  result.sourced_erg = totals_h[kTotalSourced];
  return result;
}

}  // namespace tenryu::burn
