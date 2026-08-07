#include "hydro/conduction_snb_2d.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <cub/cub.cuh>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/macros.hpp"
#include "hydro/conduction.cuh"
#include "hydro/kershaw_stencil.cuh"

namespace tenryu::hydro::snb2d {
namespace {

using kershaw::BC_REFLECT;
using kershaw::assemble_kershaw_cell;
using kershaw::cell_index;
using kershaw::kC;
using kershaw::kE;
using kershaw::kN;
using kershaw::kNE;
using kershaw::kNW;
using kershaw::kS;
using kershaw::kSE;
using kershaw::kStencilSize;
using kershaw::kSW;
using kershaw::kW;
using kershaw::node_gradient_from_cells;
using kershaw::symmetric_pair_power;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kFourPi = 12.566370614359172953850573533118;
constexpr double kEvToErg = 1.6022e-12;
constexpr double kElectronMass = 9.1094e-28;
constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
constexpr double kMaxTeForPow = 1.0e8;
constexpr double kMinEffectiveA = 1.0e-12;
constexpr int kBlockSize = 256;
// e^4 in cgs (esu), e = 4.8032e-10 statC (NRL Plasma Formulary) — 1D SNB parity.
constexpr double kEsuCharge4 = 4.8032e-10 * 4.8032e-10 * 4.8032e-10 * 4.8032e-10;
constexpr double kTwoSqrtTwo = 2.8284271247461902909;  // 2*sqrt(2)
constexpr double kBetaMin = 0.1;  // first interior reduced-energy edge (design §3.2)

// Internal solver constants (documented in the design doc, NOT namelist knobs;
// spec docs/design/2d_snb_port_spec.md §3.3 / §8.8).
constexpr double kSnbCgRtol = 1.0e-10;
constexpr int kSnbCgMaxIters = 1000;

// Scalar-pack layout (device doubles) — 1D module parity.
constexpr int kScalarTRef = 0;      // max sanitized Te [eV]
constexpr int kScalarResid = 1;     // max |P_dq - P_dq_prev| over cell-slots
constexpr int kScalarPshMax = 2;    // max |P_sh| over cell-slots
constexpr int kScalarPdqMax = 3;    // max |P_dq| over cell-slots
constexpr int kScalarThetaMin = 4;  // min theta over non-void cells

inline void cuda_check(const cudaError_t err, const char* message) {
  if (err != cudaSuccess) {
    core::log_error(std::string(message) + ": " + cudaGetErrorString(err));
    std::abort();
  }
}

// ---- helpers replicated bit-identically from this tree's conduction.cu ----

// [verbatim conduction.cu:359]
__host__ __device__ inline double sanitize_te_for_pow(const double Te_eV) {
  return (isfinite(Te_eV) && Te_eV > 0.0) ? fmin(Te_eV, kMaxTeForPow) : 0.0;
}

// [verbatim conduction.cu:363]
__host__ __device__ inline double electron_density_formula(const double rho,
                                                           const double zbar,
                                                           const double A) {
  const double rho_pos = fmax(rho, 0.0);
  const double z = fmax(zbar, 0.0);
  if (!(rho_pos > 0.0) || !(z > 0.0) || !(A > 0.0)) {
    return 0.0;
  }
  const double n_e = rho_pos * z / (A * kProtonMass);
  return isfinite(n_e) ? n_e : 0.0;
}

// [verbatim conduction.cu:375]
__host__ __device__ inline double coulomb_log_formula(const double n_e,
                                                      const double Te_eV,
                                                      const double Zbar) {
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(n_e > 0.0) || !(te > 0.0) || !(Zbar > 0.0)) {
    return 2.0;
  }

  double ln_lambda_raw = 0.0;
  if (te >= 10.0 * Zbar * Zbar) {
    ln_lambda_raw = 24.0 - 0.5 * log(n_e) + log(te);
  } else {
    ln_lambda_raw = 23.0 - 0.5 * log(n_e) - log(Zbar) + 1.5 * log(te);
  }
  return fmax(2.0, ln_lambda_raw);
}

// [verbatim conduction.cu:404]
__host__ __device__ inline double vth_e_formula(const double Te_eV) {
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(te > 0.0)) {
    return 0.0;
  }
  const double vth = sqrt(kEvToErg * te / kElectronMass);
  return isfinite(vth) ? vth : 0.0;
}

// [verbatim conduction.cu:414]
__host__ __device__ inline double q_max_formula(const double f_lim,
                                                const double n_e,
                                                const double Te_eV) {
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(f_lim > 0.0) || !(n_e > 0.0) || !(te > 0.0)) {
    return 0.0;
  }
  const double q_max = f_lim * n_e * kEvToErg * te * vth_e_formula(te);
  return isfinite(q_max) ? q_max : 0.0;
}

// [verbatim conduction.cu:563]
__device__ double atomic_min_double(double* address, const double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

// [verbatim conduction.cu:705]
__device__ double atomic_max_double(double* address, const double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value > __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

// ---- SNB spectral/mfp device math (1D-module transcription, design §1) ----

// Normalized cumulative of the SNB source spectrum: P(b) = int_0^b u^4 e^-u du / 24.
__host__ __device__ inline double snb_spectrum_cdf(const double b) {
  if (!(b > 0.0)) {
    return 0.0;
  }
  const double poly = ((((b + 4.0) * b + 12.0) * b + 24.0) * b + 24.0);
  const double value = 1.0 - exp(-b) * poly / 24.0;
  return fmin(fmax(value, 0.0), 1.0);
}

// Group mfp lambda_g(eps_g; cell) [cm] (design §1.4).
// mfp_variant: 0 = geometric_r2 (Sherlock 2017 Eq 1), 1 = original (Schurtz Eq 3+23).
__device__ inline double snb_lambda_g(const double eps_erg,
                                      const double n_e,
                                      const double zbar,
                                      const double ln_lambda,
                                      const int mfp_variant) {
  if (!(n_e > 0.0) || !(zbar > 0.0) || !(ln_lambda > 0.0) || !(eps_erg > 0.0)) {
    return 0.0;
  }
  const double denom_base = kFourPi * n_e * kEsuCharge4 * ln_lambda;
  double z_factor;
  double prefactor;
  if (mfp_variant == 1) {
    z_factor = sqrt(zbar + 1.0);
    prefactor = 2.0;
  } else {
    const double phi = (zbar + 4.2) / (zbar + 0.24);
    z_factor = sqrt(zbar * phi);
    prefactor = kTwoSqrtTwo;
  }
  const double lam = prefactor * eps_erg * eps_erg / (denom_base * z_factor);
  return (isfinite(lam) && lam > 0.0) ? lam : 0.0;
}

// Pair helpers: slot -> neighbor cell (clamped-to-self at domain edges, the
// apply-kernel convention: n==c pairs carry zero power) and mirror slot.
__host__ __device__ inline int slot_neighbor(const int slot, const int i,
                                             const int j, const int nr,
                                             const int nz) {
  const int iE = (i + 1 < nr) ? (i + 1) : i;
  const int iW = (i > 0) ? (i - 1) : i;
  const int jN = (j + 1 < nz) ? (j + 1) : j;
  const int jS = (j > 0) ? (j - 1) : j;
  switch (slot) {
    case kE:
      return cell_index(iE, j, nz);
    case kW:
      return cell_index(iW, j, nz);
    case kN:
      return cell_index(i, jN, nz);
    case kS:
      return cell_index(i, jS, nz);
    case kNE:
      return cell_index(iE, jN, nz);
    case kNW:
      return cell_index(iW, jN, nz);
    case kSE:
      return cell_index(iE, jS, nz);
    default:
      return cell_index(iW, jS, nz);  // kSW
  }
}

__host__ __device__ inline int mirror_slot(const int slot) {
  switch (slot) {
    case kE:
      return kW;
    case kW:
      return kE;
    case kN:
      return kS;
    case kS:
      return kN;
    case kNE:
      return kSW;
    case kNW:
      return kSE;
    case kSE:
      return kNW;
    default:
      return kNE;  // kSW
  }
}

// ---- kernels ----

__global__ void snb2d_max_te_kernel(const double* __restrict__ Te,
                                    const std::uint8_t* __restrict__ cell_is_void,
                                    const int n_cells,
                                    double* __restrict__ scalars) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    return;
  }
  const double te = sanitize_te_for_pow(Te[c]);
  if (te > 0.0) {
    atomic_max_double(&scalars[kScalarTRef], te);
  }
}

// Per-(group, cell) mfp evaluation (lam_tr == lam_abs; the E-field variant is
// fail-closed in 2D v1 by namelist validation).
__global__ void snb2d_lambda_kernel(double* __restrict__ lam_tr,
                                    double* __restrict__ lam_abs,
                                    const double* __restrict__ Te,
                                    const double* __restrict__ rho,
                                    const double* __restrict__ zbar,
                                    const double* __restrict__ A_eff,
                                    const std::uint8_t* __restrict__ cell_is_void,
                                    const double* __restrict__ edges,
                                    const int n_cells,
                                    const int n_groups,
                                    const int mfp_variant) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cells * n_groups) {
    return;
  }
  const int g = idx / n_cells;
  const int c = idx - g * n_cells;
  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    lam_tr[idx] = 0.0;
    lam_abs[idx] = 0.0;
    return;
  }
  const double te = sanitize_te_for_pow(Te[c]);
  const double z = fmax(zbar[c], 0.0);
  const double rho_c = fmax(rho[c], 0.0);
  const double A_c = fmax(A_eff[c], kMinEffectiveA);
  const double n_e = electron_density_formula(rho_c, z, A_c);
  const double ln_lambda = coulomb_log_formula(n_e, te, z);
  const double eps = 0.5 * (edges[g] + edges[g + 1]);  // group CENTER (design §3.2)
  const double lam = snb_lambda_g(eps, n_e, z, ln_lambda, mfp_variant);
  lam_abs[idx] = lam;
  lam_tr[idx] = lam;
}

// D_g[c] = lam_tr[g*n + c] / 3 for the Kershaw builder of group g.
__global__ void snb2d_lam_over3_kernel(double* __restrict__ d_over3,
                                       const double* __restrict__ lam_tr_g,
                                       const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  d_over3[c] = lam_tr_g[c] / 3.0;
}

// Per-group Kershaw stencil build for the H_g operator: all four boundaries
// reflective (paper-standard H BC, design §2), M-matrix repair ON. Reuses the
// header-inline assemble_kershaw_cell — existing TUs untouched.
__global__ void snb2d_h_stencil_build_kernel(double* __restrict__ stencil,
                                             const double* __restrict__ d_over3,
                                             const double* __restrict__ x_r,
                                             const double* __restrict__ x_z,
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
  assemble_kershaw_cell(a, d_over3, x_r, x_z, i, j, nr, nz, BC_REFLECT,
                        BC_REFLECT, BC_REFLECT, BC_REFLECT, true, repaired);
  double* row = stencil + static_cast<std::size_t>(c) * kStencilSize;
  for (int k = 0; k < kStencilSize; ++k) {
    row[k] = a[k];
  }
}

// Symmetrized CSR row for group g from the per-group stencil buffer. The pair
// conductance uses the SAME symmetrization as the apply path
// (Ḡ = 0.5(G_cn + G_nc), kershaw_stencil.cuh symmetric_pair_power), so the
// solved operator and the pair-power flux are one discrete object.
// Row layout: entry 0 = diagonal, entries 1..8 = slots kE..kSW (zero-padded,
// col=row when absent). diag = 4π ΣḠ + V/λ_abs; void or λ_abs<=0 rows are
// identity (H=0).
__global__ void snb2d_symmetrize_csr_kernel(
    double* __restrict__ csr_values,
    int* __restrict__ csr_cols,
    double* __restrict__ diag_inv,
    const double* __restrict__ stencil,
    const double* __restrict__ lam_abs_g,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const int nr,
    const int nz,
    const int g,
    const int n_groups_total_cells /* = n_cells (row base offset uses g) */) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int row = g * n_cells + c;
  double* vals = csr_values + static_cast<std::size_t>(row) * kStencilSize;
  int* cols = csr_cols + static_cast<std::size_t>(row) * kStencilSize;

  const bool is_void =
      (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0));
  const double lam_a = lam_abs_g[c];
  const bool identity_row = is_void || !(lam_a > 0.0) || !(rho_cv_e[c] > 0.0);

  double diag = 0.0;
  double sum_g = 0.0;
  for (int s = 1; s < kStencilSize; ++s) {
    double gbar = 0.0;
    int ncell = c;
    if (!identity_row) {
      ncell = slot_neighbor(s, i, j, nr, nz);
      const bool nvoid = (cell_is_void != nullptr &&
                          cell_is_void[ncell] != static_cast<std::uint8_t>(0));
      if (ncell != c && !nvoid && rho_cv_e[ncell] > 0.0 &&
          lam_abs_g[ncell] > 0.0) {
        const double* row_c = stencil + static_cast<std::size_t>(c) * kStencilSize;
        const double* row_n =
            stencil + static_cast<std::size_t>(ncell) * kStencilSize;
        const double g_cn = -row_c[s];
        const double g_nc = -row_n[mirror_slot(s)];
        gbar = 0.5 * (g_cn + g_nc);
        if (!(gbar > 0.0) || !isfinite(gbar)) {
          gbar = 0.0;  // post-repair coefficients are >= 0; guard NaN/neg
        }
      }
    }
    vals[s] = -kFourPi * gbar;
    cols[s] = (gbar != 0.0) ? (g * n_cells + ncell) : row;
    sum_g += gbar;
  }
  if (identity_row) {
    diag = 1.0;
    for (int s = 1; s < kStencilSize; ++s) {
      vals[s] = 0.0;
      cols[s] = row;
    }
  } else {
    const double V = fmax(vol[c], 1.0e-30);
    diag = kFourPi * sum_g + V / lam_a;
  }
  vals[0] = diag;
  cols[0] = row;
  diag_inv[row] = 1.0 / diag;
}

// RHS for group g: b = Σ_s ξ_{g,cs} · P^sh_cs, with P^sh the OFF T-stencil
// pair power (4π Ḡ^T (Te_n - Te_c), symmetric_pair_power) and ξ the pair-mean
// spectral share (design §3.3). Identity rows get b=0.
__global__ void snb2d_rhs_kernel(double* __restrict__ b,
                                 const double* __restrict__ t_stencil,
                                 const double* __restrict__ Te,
                                 const double* __restrict__ rho_cv_e,
                                 const std::uint8_t* __restrict__ cell_is_void,
                                 const double* __restrict__ lam_abs_g,
                                 const double* __restrict__ edges,
                                 const int nr,
                                 const int nz,
                                 const int g,
                                 const int n_groups) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int row = g * n_cells + c;
  const bool is_void =
      (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0));
  if (is_void || !(lam_abs_g[c] > 0.0) || !(rho_cv_e[c] > 0.0)) {
    b[row] = 0.0;
    return;
  }
  const double te_c = sanitize_te_for_pow(Te[c]);
  double rhs = 0.0;
  for (int s = 1; s < kStencilSize; ++s) {
    const int ncell = slot_neighbor(s, i, j, nr, nz);
    const double p_sh = kFourPi * symmetric_pair_power(t_stencil, Te, rho_cv_e,
                                                       cell_is_void, c, ncell,
                                                       s, mirror_slot(s));
    if (p_sh == 0.0) {
      continue;
    }
    const double te_pair = 0.5 * (te_c + sanitize_te_for_pow(Te[ncell]));
    double xi = 0.0;
    if (te_pair > 0.0) {
      const double kT = kEvToErg * te_pair;
      xi = snb_spectrum_cdf(edges[g + 1] / kT) - snb_spectrum_cdf(edges[g] / kT);
    }
    rhs += xi * p_sh;
  }
  b[row] = isfinite(rhs) ? rhs : 0.0;
}

// ---- CG primitives (deterministic: fixed-order cub reductions, no atomics) --

__global__ void snb2d_spmv_kernel(double* __restrict__ y,
                                  const double* __restrict__ csr_values,
                                  const int* __restrict__ csr_cols,
                                  const double* __restrict__ x,
                                  const int n_rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) {
    return;
  }
  const double* vals = csr_values + static_cast<std::size_t>(row) * kStencilSize;
  const int* cols = csr_cols + static_cast<std::size_t>(row) * kStencilSize;
  double acc = 0.0;
  for (int k = 0; k < kStencilSize; ++k) {
    acc += vals[k] * x[cols[k]];
  }
  y[row] = acc;
}

__global__ void snb2d_elemprod_kernel(double* __restrict__ out,
                                      const double* __restrict__ a,
                                      const double* __restrict__ b,
                                      const int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  out[idx] = a[idx] * b[idx];
}

// y += alpha * x
__global__ void snb2d_axpy_kernel(double* __restrict__ y,
                                  const double* __restrict__ x,
                                  const double alpha,
                                  const int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  y[idx] += alpha * x[idx];
}

// p = z + beta * p
__global__ void snb2d_xpay_kernel(double* __restrict__ p,
                                  const double* __restrict__ z,
                                  const double beta,
                                  const int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  p[idx] = z[idx] + beta * p[idx];
}

// z = Dinv * r
__global__ void snb2d_dinv_kernel(double* __restrict__ z,
                                  const double* __restrict__ diag_inv,
                                  const double* __restrict__ r,
                                  const int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  z[idx] = diag_inv[idx] * r[idx];
}

__global__ void snb2d_zero_kernel(double* __restrict__ x, const int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  x[idx] = 0.0;
}

// Probe export: live T-stencil pair powers, slot-major [c*8 + (slot-1)].
__global__ void snb2d_psh_export_kernel(double* __restrict__ out,
                                        const double* __restrict__ t_stencil,
                                        const double* __restrict__ Te,
                                        const double* __restrict__ rho_cv_e,
                                        const std::uint8_t* __restrict__ cell_is_void,
                                        const int nr,
                                        const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const bool is_void =
      (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0));
  for (int s = 1; s < kStencilSize; ++s) {
    double p = 0.0;
    if (!is_void) {
      const int ncell = slot_neighbor(s, i, j, nr, nz);
      p = kFourPi * symmetric_pair_power(t_stencil, Te, rho_cv_e, cell_is_void,
                                         c, ncell, s, mirror_slot(s));
    }
    out[c * 8 + (s - 1)] = p;
  }
}

// Nonlocal pair correction P_dq and outer cap theta (design §3.4). P_dq is
// read back from the SYMMETRIZED CSR off-diagonals (operator/flux consistency
// is exact by construction): P_dq[c][s] = Σ_g (-csr[s]) (H_n - H_c).
// theta_c caps the cell's total-flux magnitude reconstructed from 4-node
// gradients (the OFF dt-diagnostic convention) against q_max(f_lim).
__global__ void snb2d_pdq_theta_kernel(
    double* __restrict__ p_dq,           // [n*8]
    double* __restrict__ theta,          // [n]
    const double* __restrict__ p_dq_prev,  // [n*8] (nullptr on seed pass)
    const double* __restrict__ csr_values,
    const double* __restrict__ H,        // [G*n]
    const double* __restrict__ lam_tr,   // [G*n]
    const double* __restrict__ t_stencil,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ A_eff,
    const double* __restrict__ kappa_eff,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz,
    const int n_groups,
    const int use_outer_cap,
    const double f_lim,
    double* __restrict__ scalars,
    int* __restrict__ counters) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const bool is_void =
      (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0));

  // Pair sums: P_dq per slot (frozen through the next T-solve) + scalar packs.
  for (int s = 1; s < kStencilSize; ++s) {
    double pdq = 0.0;
    if (!is_void) {
      const int ncell = slot_neighbor(s, i, j, nr, nz);
      if (ncell != c) {
        for (int g = 0; g < n_groups; ++g) {
          const int row = g * n_cells + c;
          const double off =
              csr_values[static_cast<std::size_t>(row) * kStencilSize + s];
          if (off != 0.0) {
            pdq += (-off) * (H[g * n_cells + ncell] - H[g * n_cells + c]);
          }
        }
      }
    }
    if (!isfinite(pdq)) {
      pdq = 0.0;
    }
    p_dq[c * 8 + (s - 1)] = pdq;
    if (p_dq_prev != nullptr) {
      atomic_max_double(&scalars[kScalarResid],
                        fabs(pdq - p_dq_prev[c * 8 + (s - 1)]));
    }
    if (pdq != 0.0) {
      atomic_max_double(&scalars[kScalarPdqMax], fabs(pdq));
    }
    // |P_sh| scale (live pair power on the T-stencil).
    if (!is_void) {
      const int ncell = slot_neighbor(s, i, j, nr, nz);
      const double p_sh =
          kFourPi * symmetric_pair_power(t_stencil, Te, rho_cv_e, cell_is_void,
                                         c, ncell, s, mirror_slot(s));
      if (p_sh != 0.0) {
        atomic_max_double(&scalars[kScalarPshMax], fabs(p_sh));
      }
    }
  }

  double th = 1.0;
  if (is_void) {
    th = 1.0;
  } else if (use_outer_cap != 0) {
    // 4-node-average gradients (OFF dt-diagnostic convention,
    // compute_spitzer_deff_2d_kernel:1382-1389 style, vector form).
    double gr[4];
    double gz[4];
    node_gradient_from_cells(Te, x_r, x_z, i, j, nr, nz, gr[0], gz[0]);
    node_gradient_from_cells(Te, x_r, x_z, i + 1, j, nr, nz, gr[1], gz[1]);
    node_gradient_from_cells(Te, x_r, x_z, i + 1, j + 1, nr, nz, gr[2], gz[2]);
    node_gradient_from_cells(Te, x_r, x_z, i, j + 1, nr, nz, gr[3], gz[3]);
    const double gtr = 0.25 * (gr[0] + gr[1] + gr[2] + gr[3]);
    const double gtz = 0.25 * (gz[0] + gz[1] + gz[2] + gz[3]);
    double q_r = -kappa_eff[c] * gtr;
    double q_z = -kappa_eff[c] * gtz;
    for (int g = 0; g < n_groups; ++g) {
      double hr[4];
      double hz[4];
      const double* Hg = H + static_cast<std::size_t>(g) * n_cells;
      node_gradient_from_cells(Hg, x_r, x_z, i, j, nr, nz, hr[0], hz[0]);
      node_gradient_from_cells(Hg, x_r, x_z, i + 1, j, nr, nz, hr[1], hz[1]);
      node_gradient_from_cells(Hg, x_r, x_z, i + 1, j + 1, nr, nz, hr[2],
                               hz[2]);
      node_gradient_from_cells(Hg, x_r, x_z, i, j + 1, nr, nz, hr[3], hz[3]);
      const double d3 = lam_tr[g * n_cells + c] / 3.0;
      q_r -= d3 * 0.25 * (hr[0] + hr[1] + hr[2] + hr[3]);
      q_z -= d3 * 0.25 * (hz[0] + hz[1] + hz[2] + hz[3]);
    }
    const double q_mag = sqrt(q_r * q_r + q_z * q_z);
    const double te = sanitize_te_for_pow(Te[c]);
    const double z = fmax(zbar[c], 0.0);
    const double A_c = fmax(A_eff[c], kMinEffectiveA);
    const double n_e = electron_density_formula(fmax(rho[c], 0.0), z, A_c);
    const double q_max = q_max_formula(f_lim, n_e, te);
    if (q_max > 0.0) {
      th = 1.0 / (1.0 + q_mag / q_max);
    } else {
      th = 0.0;
    }
  }
  if (!isfinite(th)) {
    th = 0.0;
  }
  th = fmin(1.0, fmax(th, 0.0));
  theta[c] = th;
  if (!is_void) {
    atomic_min_double(&scalars[kScalarThetaMin], th);
    if (th < 0.99) {
      atomicAdd(&counters[0], 1);
    }
    if (th < 0.5) {
      atomicAdd(&counters[1], 1);
    }
  }
}

// SNB stage kernels: pair-structure clones of kershaw_alpha_pass_kernel /
// kershaw_apply_kernel (header-inline originals stay byte-untouched) with the
// pair power replaced by min(theta_c, theta_n) * (P_sh_live + P_dq_frozen).
// P_dq antisymmetry is exact (shared symmetric CSR value x negated dH), so
// telescoping conservation is structural; the alpha floor-throttle acts on
// the capped pair powers, with net and donor modes mirroring the Kershaw rules.
__device__ inline double snb2d_pair_power(const double* __restrict__ t_stencil,
                                          const double* __restrict__ Te,
                                          const double* __restrict__ rho_cv_e,
                                          const std::uint8_t* __restrict__ cell_is_void,
                                          const double* __restrict__ p_dq,
                                          const double* __restrict__ theta,
                                          const int c,
                                          const int ncell,
                                          const int slot) {
  if (ncell == c) {
    return 0.0;
  }
  const double p_sh =
      kFourPi * symmetric_pair_power(t_stencil, Te, rho_cv_e, cell_is_void, c,
                                     ncell, slot, mirror_slot(slot));
  const double p = p_sh + p_dq[c * 8 + (slot - 1)];
  if (p == 0.0) {
    return 0.0;
  }
  const double th = fmin(theta[c], theta[ncell]);
  return th * p;
}

__global__ void snb2d_alpha_pass_kernel(double* __restrict__ alpha_cells,
                                        const double* __restrict__ Te,
                                        const double* __restrict__ t_stencil,
                                        const double* __restrict__ p_dq,
                                        const double* __restrict__ theta,
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
  const double V = fmax(vol[c], 1.0e-30);
  const double rho_cv = rho_cv_e[c];
  if (!(rho_cv > 0.0)) {
    return;
  }
  double net_power = 0.0;
  double outflow_sum = 0.0;
  for (int s = 1; s < kStencilSize; ++s) {
    const double p = snb2d_pair_power(
        t_stencil, Te, rho_cv_e, cell_is_void, p_dq, theta, c,
        slot_neighbor(s, i, j, nr, nz), s);
    net_power += p;
    if (floor_limiter_mode == 1) {
      outflow_sum += isfinite(p) ? fmax(0.0, -p) : 0.0;
    }
  }
  const double Te_old_c = Te[c];
  constexpr double kDivEps = 1.0e-30;
  double alpha = 1.0;
  if (floor_limiter_mode == 1) {
    if (!(Te_old_c > T_floor)) {
      alpha = 0.0;
    } else if (dt_sub > 0.0 && outflow_sum > 0.0) {
      alpha = fmin(
          1.0,
          fmax((rho_cv * V * (Te_old_c - T_floor) /
                fmax(outflow_sum, kDivEps)) /
                   dt_sub,
               0.0));
    }
    alpha_cells[c] = alpha;
    return;
  }
  if (dt_sub > 0.0 && Te_old_c > T_floor && net_power < 0.0) {
    const double energy_above_floor = rho_cv * V * (Te_old_c - T_floor);
    const double dt_safe = energy_above_floor / fmax(fabs(net_power), kDivEps);
    alpha = fmin(1.0, fmax(dt_safe / dt_sub, 0.0));
  }
  alpha_cells[c] = alpha;
}

__global__ void snb2d_apply_kernel(double* __restrict__ Te_new,
                                   const double* __restrict__ Te,
                                   const double* __restrict__ t_stencil,
                                   const double* __restrict__ p_dq,
                                   const double* __restrict__ theta,
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
  const double V = fmax(vol[c], 1.0e-30);
  const double rho_cv = rho_cv_e[c];
  if (!(rho_cv > 0.0)) {
    Te_new[c] = Te[c];
    return;
  }
  const double alpha_c = (alpha_cells != nullptr) ? alpha_cells[c] : 1.0;
  double net_power = 0.0;
  for (int s = 1; s < kStencilSize; ++s) {
    const int ncell = slot_neighbor(s, i, j, nr, nz);
    const double p = snb2d_pair_power(t_stencil, Te, rho_cv_e, cell_is_void,
                                      p_dq, theta, c, ncell, s);
    const double alpha_n = (alpha_cells != nullptr) ? alpha_cells[ncell] : 1.0;
    const double scale = (floor_limiter_mode == 1)
                             ? ((p > 0.0) ? alpha_cells[ncell] : alpha_c)
                             : fmin(alpha_c, alpha_n);
    net_power += p * scale;
  }
  const double Te_old_c = Te[c];
  const double energy_old = rho_cv * V * Te_old_c;
  const double energy_trial = energy_old + dt_sub * net_power;
  const double Te_trial = energy_trial / (rho_cv * V);
  if (!isfinite(Te_trial) || Te_trial < T_floor) {
    if (rho_cv > 0.0) {
      if (isfinite(Te_trial) && Te_trial < T_floor) {
        const double de = rho_cv * (T_floor - Te_trial) * V;
        kershaw::atomic_add_double(E_floor, de);
      } else if (!isfinite(Te_trial) && T_floor > 0.0) {
        const double de = rho_cv * T_floor * V;
        kershaw::atomic_add_double(E_floor, de);
      }
    }
    Te_new[c] = T_floor;
    atomicAdd(clamp_count, 1);
    return;
  }
  Te_new[c] = Te_trial;
}

// ---- host side ----

struct Snb2dWork {
  double* edges = nullptr;       // [G+1] erg
  double* lam_tr = nullptr;      // [G*n]
  double* lam_abs = nullptr;     // [G*n]
  double* d_over3 = nullptr;     // [n] per-group D staging
  double* h_stencil = nullptr;   // [9*n] per-group Kershaw rows
  double* csr_values = nullptr;  // [9*G*n]
  int* csr_cols = nullptr;       // [9*G*n]
  double* diag_inv = nullptr;    // [G*n]
  double* b = nullptr;           // [G*n]
  double* H = nullptr;           // [G*n] CG solution
  double* r = nullptr;           // [G*n]
  double* z = nullptr;           // [G*n]
  double* p = nullptr;           // [G*n]
  double* Ap = nullptr;          // [G*n]
  double* prod = nullptr;        // [G*n] dot scratch
  double* p_dq = nullptr;        // [n*8]
  double* p_dq_prev = nullptr;   // [n*8]
  double* theta = nullptr;       // [n]
  double* te_stash = nullptr;    // [n]
  double* te_tmp = nullptr;      // [n]
  double* alpha = nullptr;       // [n]
  double* scalars = nullptr;     // [8]
  int* counters = nullptr;       // [2]
  double* d_dot = nullptr;       // [1] cub output
};

Snb2dWork snb2d_acquire(const int n, const int G) {
  Snb2dWork w;
  const auto acq = [&](const char* tag, const std::size_t bytes) {
    return core::device_scratch_acquire(tag, bytes);
  };
  const std::size_t gn = static_cast<std::size_t>(G) * n;
  w.edges = static_cast<double*>(acq("conduction:snb2d:edges", (G + 1) * sizeof(double)));
  w.lam_tr = static_cast<double*>(acq("conduction:snb2d:lam_tr", gn * sizeof(double)));
  w.lam_abs = static_cast<double*>(acq("conduction:snb2d:lam_abs", gn * sizeof(double)));
  w.d_over3 = static_cast<double*>(acq("conduction:snb2d:d_over3", n * sizeof(double)));
  w.h_stencil = static_cast<double*>(
      acq("conduction:snb2d:h_stencil", static_cast<std::size_t>(9) * n * sizeof(double)));
  w.csr_values = static_cast<double*>(
      acq("conduction:snb2d:csr_values", 9 * gn * sizeof(double)));
  w.csr_cols = static_cast<int*>(acq("conduction:snb2d:csr_cols", 9 * gn * sizeof(int)));
  w.diag_inv = static_cast<double*>(acq("conduction:snb2d:diag_inv", gn * sizeof(double)));
  w.b = static_cast<double*>(acq("conduction:snb2d:b", gn * sizeof(double)));
  w.H = static_cast<double*>(acq("conduction:snb2d:H", gn * sizeof(double)));
  w.r = static_cast<double*>(acq("conduction:snb2d:r", gn * sizeof(double)));
  w.z = static_cast<double*>(acq("conduction:snb2d:z", gn * sizeof(double)));
  w.p = static_cast<double*>(acq("conduction:snb2d:p", gn * sizeof(double)));
  w.Ap = static_cast<double*>(acq("conduction:snb2d:Ap", gn * sizeof(double)));
  w.prod = static_cast<double*>(acq("conduction:snb2d:prod", gn * sizeof(double)));
  w.p_dq = static_cast<double*>(acq("conduction:snb2d:p_dq",
                                    static_cast<std::size_t>(8) * n * sizeof(double)));
  w.p_dq_prev = static_cast<double*>(
      acq("conduction:snb2d:p_dq_prev", static_cast<std::size_t>(8) * n * sizeof(double)));
  w.theta = static_cast<double*>(acq("conduction:snb2d:theta", n * sizeof(double)));
  w.te_stash = static_cast<double*>(acq("conduction:snb2d:te_stash", n * sizeof(double)));
  w.te_tmp = static_cast<double*>(acq("conduction:snb2d:te_tmp", n * sizeof(double)));
  w.alpha = static_cast<double*>(acq("conduction:snb2d:alpha", n * sizeof(double)));
  w.scalars = static_cast<double*>(acq("conduction:snb2d:scalars", 8 * sizeof(double)));
  w.counters = static_cast<int*>(acq("conduction:snb2d:counters", 2 * sizeof(int)));
  w.d_dot = static_cast<double*>(acq("conduction:snb2d:dot", sizeof(double)));
  return w;
}

double snb2d_dot(const Snb2dWork& w, const double* a, const double* b,
                 const int n_rows) {
  const int blocks = (n_rows + kBlockSize - 1) / kBlockSize;
  snb2d_elemprod_kernel<<<blocks, kBlockSize>>>(w.prod, a, b, n_rows);
  cuda_check(cudaGetLastError(), "SNB2D: elemprod launch failed");
  std::size_t temp_bytes = 0;
  cuda_check(cub::DeviceReduce::Sum(nullptr, temp_bytes, w.prod, w.d_dot, n_rows),
             "SNB2D: cub sum size failed");
  void* temp = core::device_scratch_acquire("conduction:snb2d:cub_temp",
                                            std::max<std::size_t>(temp_bytes, 8));
  cuda_check(cub::DeviceReduce::Sum(temp, temp_bytes, w.prod, w.d_dot, n_rows),
             "SNB2D: cub sum failed");
  double out = 0.0;
  cuda_check(cudaMemcpy(&out, w.d_dot, sizeof(double), cudaMemcpyDeviceToHost),
             "SNB2D: dot readback failed");
  return out;
}

// Jacobi-preconditioned CG over the group-batched block-diagonal SPD system.
// Deterministic: fixed-order cub reductions only. Returns iterations; fills
// resid_out with the final relative residual.
bool snb2d_cg_solve(const Snb2dWork& w, const int n_rows, int* iters_out,
                    double* resid_out) {
  const int blocks = (n_rows + kBlockSize - 1) / kBlockSize;
  snb2d_zero_kernel<<<blocks, kBlockSize>>>(w.H, n_rows);
  cuda_check(cudaMemcpy(w.r, w.b, static_cast<std::size_t>(n_rows) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "SNB2D: r init failed");
  const double norm_b2 = snb2d_dot(w, w.b, w.b, n_rows);
  *iters_out = 0;
  *resid_out = 0.0;
  if (!(norm_b2 > 0.0)) {
    return true;  // zero source -> H = 0 exactly
  }
  const double stop2 = kSnbCgRtol * kSnbCgRtol * norm_b2;
  snb2d_dinv_kernel<<<blocks, kBlockSize>>>(w.z, w.diag_inv, w.r, n_rows);
  cuda_check(cudaMemcpy(w.p, w.z, static_cast<std::size_t>(n_rows) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "SNB2D: p init failed");
  double rz = snb2d_dot(w, w.r, w.z, n_rows);
  double rr = norm_b2;
  for (int it = 1; it <= kSnbCgMaxIters; ++it) {
    snb2d_spmv_kernel<<<blocks, kBlockSize>>>(w.Ap, w.csr_values, w.csr_cols,
                                              w.p, n_rows);
    const double pAp = snb2d_dot(w, w.p, w.Ap, n_rows);
    if (!(pAp > 0.0) || !std::isfinite(pAp)) {
      *iters_out = it;
      *resid_out = std::sqrt(std::max(rr, 0.0) / norm_b2);
      return false;  // SPD violated (should not happen; M-matrix by design)
    }
    const double alpha = rz / pAp;
    snb2d_axpy_kernel<<<blocks, kBlockSize>>>(w.H, w.p, alpha, n_rows);
    snb2d_axpy_kernel<<<blocks, kBlockSize>>>(w.r, w.Ap, -alpha, n_rows);
    rr = snb2d_dot(w, w.r, w.r, n_rows);
    *iters_out = it;
    if (rr <= stop2) {
      *resid_out = std::sqrt(std::max(rr, 0.0) / norm_b2);
      return true;
    }
    snb2d_dinv_kernel<<<blocks, kBlockSize>>>(w.z, w.diag_inv, w.r, n_rows);
    const double rz_new = snb2d_dot(w, w.r, w.z, n_rows);
    const double beta = rz_new / rz;
    rz = rz_new;
    snb2d_xpay_kernel<<<blocks, kBlockSize>>>(w.p, w.z, beta, n_rows);
  }
  *resid_out = std::sqrt(std::max(rr, 0.0) / norm_b2);
  return false;
}

struct Snb2dPassStats {
  double resid = 0.0;
  double psh_max = 0.0;
  double pdq_max = 0.0;
  double theta_min = 1.0;
  int cg_iters = 0;
  double cg_resid = 0.0;
  bool cg_converged = true;
};

// One coefficient pass + H-solve + P_dq/theta refresh on the CURRENT state.Te
// (the 1D snb_pass analog; design §3.6 pseudo-code inner block).
Snb2dPassStats snb2d_pass(core::State& state,
                          const core::Config& cfg,
                          const Snb2dWork& w,
                          const double* d_t_stencil,
                          const double* d_kappa_eff,
                          const double* d_rho_cv_e,
                          const std::uint8_t* d_cell_is_void,
                          const double* d_A_eff,
                          const int nr,
                          const int nz,
                          const bool use_outer_cap,
                          const bool seed_pass) {
  const auto& sc = cfg.numerics.conduction;
  const int n = nr * nz;
  const int G = sc.snb_n_groups;
  const int mfp_variant = (sc.snb_mfp == "original") ? 1 : 0;
  const int cell_blocks = (n + kBlockSize - 1) / kBlockSize;
  const int gc_blocks = (n * G + kBlockSize - 1) / kBlockSize;

  // Reset the per-pass reduction slots (resid/psh/pdq) and counters; keep TRef.
  const double zeros3[3] = {0.0, 0.0, 0.0};
  cuda_check(cudaMemcpy(w.scalars + kScalarResid, zeros3, sizeof(zeros3),
                        cudaMemcpyHostToDevice),
             "SNB2D: scalar reset failed");
  const double theta_min_init = 1.0;
  cuda_check(cudaMemcpy(w.scalars + kScalarThetaMin, &theta_min_init,
                        sizeof(double), cudaMemcpyHostToDevice),
             "SNB2D: theta-min reset failed");
  cuda_check(cudaMemset(w.counters, 0, 2 * sizeof(int)),
             "SNB2D: counter reset failed");

  snb2d_lambda_kernel<<<gc_blocks, kBlockSize>>>(
      w.lam_tr, w.lam_abs, state.Te.data(), state.rho.data(),
      state.zbar.data(), d_A_eff, d_cell_is_void, w.edges, n, G, mfp_variant);
  cuda_check(cudaGetLastError(), "SNB2D: lambda launch failed");

  for (int g = 0; g < G; ++g) {
    snb2d_lam_over3_kernel<<<cell_blocks, kBlockSize>>>(
        w.d_over3, w.lam_tr + static_cast<std::size_t>(g) * n, n);
    snb2d_h_stencil_build_kernel<<<cell_blocks, kBlockSize>>>(
        w.h_stencil, w.d_over3, state.x_r.data(), state.x_z.data(), nr, nz);
    snb2d_symmetrize_csr_kernel<<<cell_blocks, kBlockSize>>>(
        w.csr_values, w.csr_cols, w.diag_inv, w.h_stencil,
        w.lam_abs + static_cast<std::size_t>(g) * n, d_rho_cv_e,
        d_cell_is_void, state.vol.data(), nr, nz, g, n);
    snb2d_rhs_kernel<<<cell_blocks, kBlockSize>>>(
        w.b, d_t_stencil, state.Te.data(), d_rho_cv_e, d_cell_is_void,
        w.lam_abs + static_cast<std::size_t>(g) * n, w.edges, nr, nz, g, G);
  }
  cuda_check(cudaGetLastError(), "SNB2D: assembly launches failed");

  Snb2dPassStats stats;
  stats.cg_converged =
      snb2d_cg_solve(w, G * n, &stats.cg_iters, &stats.cg_resid);
  if (!stats.cg_converged) {
    static bool warned_cg = false;
    if (!warned_cg) {
      core::log_warning(
          "SNB2D H-solve CG did not reach rtol within the iteration cap; "
          "accepting last iterate (further occurrences suppressed)");
      warned_cg = true;
    }
  }

  snb2d_pdq_theta_kernel<<<cell_blocks, kBlockSize>>>(
      w.p_dq, w.theta, seed_pass ? nullptr : w.p_dq_prev, w.csr_values, w.H,
      w.lam_tr, d_t_stencil, state.Te.data(), state.rho.data(),
      state.zbar.data(), d_A_eff, d_kappa_eff, d_rho_cv_e, d_cell_is_void,
      state.x_r.data(), state.x_z.data(), nr, nz, G,
      use_outer_cap ? 1 : 0, cfg.numerics.conduction.f_lim, w.scalars,
      w.counters);
  cuda_check(cudaGetLastError(), "SNB2D: pdq/theta launch failed");

  double host_scalars[5] = {0.0, 0.0, 0.0, 0.0, 1.0};
  cuda_check(cudaMemcpy(host_scalars, w.scalars, sizeof(host_scalars),
                        cudaMemcpyDeviceToHost),
             "SNB2D: scalar readback failed");
  stats.resid = host_scalars[kScalarResid];
  stats.psh_max = host_scalars[kScalarPshMax];
  stats.pdq_max = host_scalars[kScalarPdqMax];
  stats.theta_min = host_scalars[kScalarThetaMin];
  return stats;
}

}  // namespace

std::vector<double> group_edges_beta(const int n_groups, const double e_max_over_te) {
  std::vector<double> edges(static_cast<std::size_t>(n_groups) + 1, 0.0);
  const double bmin = kBetaMin;
  const double bmax = std::max(e_max_over_te, bmin * (1.0 + 1.0e-12));
  for (int g = 1; g <= n_groups; ++g) {
    const double frac = (n_groups > 1)
                            ? (static_cast<double>(g - 1) /
                               static_cast<double>(n_groups - 1))
                            : 1.0;
    edges[static_cast<std::size_t>(g)] = bmin * std::pow(bmax / bmin, frac);
  }
  return edges;
}

void conduction_step_2d_sts_snb(core::State& state,
                                const core::Config& cfg,
                                ConductionResult& result,
                                const double* d_stencil,
                                const double* d_kappa_eff,
                                const double* d_rho_cv_e,
                                const std::uint8_t* d_cell_is_void,
                                const double* d_A_eff,
                                const int nr,
                                const int nz,
                                const std::vector<double>& tau,
                                const int n_sub,
                                const double te_floor,
                                int* d_clamp_count,
                                double* d_e_floor) {
  const auto& sc = cfg.numerics.conduction;
  const int n = nr * nz;
  const int G = sc.snb_n_groups;
  const bool use_outer_cap = sc.f_lim > 0.0;
  const int floor_limiter_mode =
      (cfg.numerics.conduction.sts_floor_limiter == "donor") ? 1 : 0;
  const int cell_blocks = (n + kBlockSize - 1) / kBlockSize;

  Snb2dWork w = snb2d_acquire(n, G);

  result.snb_picard_iters = 0;
  result.snb_converged = true;
  result.snb_picard_resid = 0.0;
  result.snb_cap_faces_99 = 0;
  result.snb_cap_faces_50 = 0;
  result.snb_cap_theta_min = 1.0;
  result.snb_dq_over_qsh_max = 0.0;
  result.snb_solver_iters = 0;
  result.snb_solver_resid = 0.0;

  // Frozen group reference temperature from Te^n (design §3.2).
  cuda_check(cudaMemset(w.scalars, 0, 8 * sizeof(double)),
             "SNB2D: scalars init failed");
  snb2d_max_te_kernel<<<cell_blocks, kBlockSize>>>(
      state.Te.data(), d_cell_is_void, n, w.scalars);
  cuda_check(cudaGetLastError(), "SNB2D: max-Te launch failed");
  double t_ref_ev = 0.0;
  cuda_check(cudaMemcpy(&t_ref_ev, w.scalars + kScalarTRef, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB2D: T_ref readback failed");
  if (!(t_ref_ev > 0.0) || n < 9) {
    return;  // all-cold/void or degenerate mesh: SNB correction is zero
  }

  const std::vector<double> beta_edges = group_edges_beta(G, sc.snb_E_max_over_Te);
  std::vector<double> edges_erg(beta_edges.size(), 0.0);
  for (std::size_t k = 0; k < beta_edges.size(); ++k) {
    edges_erg[k] = beta_edges[k] * kEvToErg * t_ref_ev;
  }
  cuda_check(cudaMemcpy(w.edges, edges_erg.data(),
                        edges_erg.size() * sizeof(double), cudaMemcpyHostToDevice),
             "SNB2D: edges upload failed");

  cuda_check(cudaMemcpy(w.te_stash, state.Te.data(),
                        static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "SNB2D: Te stash failed");
  cuda_check(cudaMemset(w.p_dq, 0, static_cast<std::size_t>(8) * n * sizeof(double)),
             "SNB2D: p_dq init failed");
  cuda_check(cudaMemset(w.p_dq_prev, 0,
                        static_cast<std::size_t>(8) * n * sizeof(double)),
             "SNB2D: p_dq prev init failed");

  // Seed pass: dq(Te^n), theta^0 incl. cap on q_sh alone (1D parity).
  Snb2dPassStats stats =
      snb2d_pass(state, cfg, w, d_stencil, d_kappa_eff, d_rho_cv_e,
                 d_cell_is_void, d_A_eff, nr, nz, use_outer_cap, true);
  if (std::getenv("TENRYU_SNB2D_DEBUG") != nullptr) {
    core::log_info("[snb2d] m=" + std::to_string(0) +
                   " resid=" + std::to_string(stats.resid) +
                   " psh=" + std::to_string(stats.psh_max) +
                   " pdq=" + std::to_string(stats.pdq_max) +
                   " thmin=" + std::to_string(stats.theta_min) +
                   " cg=" + std::to_string(stats.cg_iters) +
                   " cgres=" + std::to_string(stats.cg_resid));
  }

  const int max_iters = std::max(2, sc.snb_picard_max_iters);
  int iters = 0;
  bool converged = false;
  for (int m = 1; m <= max_iters; ++m) {
    if (m > 1) {
      cuda_check(cudaMemcpy(state.Te.data(), w.te_stash,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyDeviceToDevice),
                 "SNB2D: Te restore failed");
    }
    // Accepted-solve-only clamp accounting: reset the pack every iterate.
    const int zero_i = 0;
    const double zero_d = 0.0;
    cuda_check(cudaMemcpy(d_clamp_count, &zero_i, sizeof(int),
                          cudaMemcpyHostToDevice),
               "SNB2D: clamp reset failed");
    cuda_check(cudaMemcpy(d_e_floor, &zero_d, sizeof(double),
                          cudaMemcpyHostToDevice),
               "SNB2D: e_floor reset failed");

    double* te_curr = state.Te.data();
    double* te_next = w.te_tmp;
    for (int sub = 0; sub < n_sub; ++sub) {
      for (const double tau_j : tau) {
        snb2d_alpha_pass_kernel<<<cell_blocks, kBlockSize>>>(
            w.alpha, te_curr, d_stencil, w.p_dq, w.theta, d_rho_cv_e,
            d_cell_is_void, state.vol.data(), tau_j, te_floor, nr, nz,
            floor_limiter_mode);
        snb2d_apply_kernel<<<cell_blocks, kBlockSize>>>(
            te_next, te_curr, d_stencil, w.p_dq, w.theta, d_rho_cv_e,
            d_cell_is_void, state.vol.data(), tau_j, te_floor, nr, nz,
            w.alpha, d_clamp_count, d_e_floor, floor_limiter_mode);
        cuda_check(cudaGetLastError(), "SNB2D: stage launch failed");
        std::swap(te_curr, te_next);
      }
    }
    if (te_curr != state.Te.data()) {
      cuda_check(cudaMemcpy(state.Te.data(), te_curr,
                            static_cast<std::size_t>(n) * sizeof(double),
                            cudaMemcpyDeviceToDevice),
                 "SNB2D: staged Te copy back failed");
    }
    iters = m;

    std::swap(w.p_dq, w.p_dq_prev);
    stats = snb2d_pass(state, cfg, w, d_stencil, d_kappa_eff, d_rho_cv_e,
                       d_cell_is_void, d_A_eff, nr, nz, use_outer_cap, false);
    if (std::getenv("TENRYU_SNB2D_DEBUG") != nullptr) {
      core::log_info("[snb2d] m=" + std::to_string(m) +
                     " resid=" + std::to_string(stats.resid) +
                     " psh=" + std::to_string(stats.psh_max) +
                     " pdq=" + std::to_string(stats.pdq_max) +
                     " thmin=" + std::to_string(stats.theta_min) +
                     " cg=" + std::to_string(stats.cg_iters) +
                     " cgres=" + std::to_string(stats.cg_resid));
    }
    const double scale = stats.psh_max + stats.pdq_max;
    if (m >= 2 && stats.resid <= sc.snb_picard_rtol * std::max(scale, 1.0e-300)) {
      converged = true;
      break;
    }
  }

  if (!converged) {
    static bool warned_snb_nonconv = false;
    if (!warned_snb_nonconv) {
      core::log_warning(
          "SNB2D Picard did not converge within snb_picard_max_iters; "
          "accepting last iterate (further occurrences suppressed)");
      warned_snb_nonconv = true;
    }
  }

  int host_counters[2] = {0, 0};
  cuda_check(cudaMemcpy(host_counters, w.counters, sizeof(host_counters),
                        cudaMemcpyDeviceToHost),
             "SNB2D: counter readback failed");
  result.snb_picard_iters = iters;
  result.snb_converged = converged;
  result.snb_picard_resid = stats.resid;
  result.snb_cap_faces_99 = host_counters[0];
  result.snb_cap_faces_50 = host_counters[1];
  result.snb_cap_theta_min = stats.theta_min;
  result.snb_dq_over_qsh_max =
      (stats.psh_max > 0.0) ? (stats.pdq_max / stats.psh_max) : 0.0;
  result.snb_solver_iters = stats.cg_iters;
  result.snb_solver_resid = stats.cg_resid;
}

Snb2dProbe snb2d_probe_pass(core::State& state,
                            const core::Config& cfg,
                            const double* d_stencil,
                            const double* d_kappa_eff,
                            const double* d_rho_cv_e,
                            const std::uint8_t* d_cell_is_void,
                            const double* d_A_eff,
                            const int nr,
                            const int nz) {
  const auto& sc = cfg.numerics.conduction;
  const int n = nr * nz;
  const int G = sc.snb_n_groups;
  const bool use_outer_cap = sc.f_lim > 0.0;
  const int cell_blocks = (n + kBlockSize - 1) / kBlockSize;

  Snb2dWork w = snb2d_acquire(n, G);
  Snb2dProbe probe;

  cuda_check(cudaMemset(w.scalars, 0, 8 * sizeof(double)),
             "SNB2D probe: scalars init failed");
  snb2d_max_te_kernel<<<cell_blocks, kBlockSize>>>(
      state.Te.data(), d_cell_is_void, n, w.scalars);
  cuda_check(cudaGetLastError(), "SNB2D probe: max-Te launch failed");
  cuda_check(cudaMemcpy(&probe.t_ref_ev, w.scalars + kScalarTRef,
                        sizeof(double), cudaMemcpyDeviceToHost),
             "SNB2D probe: T_ref readback failed");
  if (!(probe.t_ref_ev > 0.0) || n < 9) {
    return probe;
  }
  const std::vector<double> beta_edges = group_edges_beta(G, sc.snb_E_max_over_Te);
  probe.edges_erg.assign(beta_edges.size(), 0.0);
  for (std::size_t k = 0; k < beta_edges.size(); ++k) {
    probe.edges_erg[k] = beta_edges[k] * kEvToErg * probe.t_ref_ev;
  }
  cuda_check(cudaMemcpy(w.edges, probe.edges_erg.data(),
                        probe.edges_erg.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "SNB2D probe: edges upload failed");
  cuda_check(cudaMemset(w.p_dq, 0, static_cast<std::size_t>(8) * n * sizeof(double)),
             "SNB2D probe: p_dq init failed");

  const Snb2dPassStats stats =
      snb2d_pass(state, cfg, w, d_stencil, d_kappa_eff, d_rho_cv_e,
                 d_cell_is_void, d_A_eff, nr, nz, use_outer_cap, true);
  probe.cg_iters = stats.cg_iters;
  probe.cg_resid = stats.cg_resid;
  probe.cg_converged = stats.cg_converged;

  // Host copies: p_dq/theta/lam; p_sh recomputed pairwise on host from the
  // device pair power is not needed — export the live pair power by one more
  // kernel into the prod buffer laid out [n*8].
  probe.p_dq.assign(static_cast<std::size_t>(8) * n, 0.0);
  cuda_check(cudaMemcpy(probe.p_dq.data(), w.p_dq,
                        static_cast<std::size_t>(8) * n * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB2D probe: p_dq readback failed");
  probe.theta.assign(n, 1.0);
  cuda_check(cudaMemcpy(probe.theta.data(), w.theta,
                        static_cast<std::size_t>(n) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB2D probe: theta readback failed");
  probe.lam_tr.assign(static_cast<std::size_t>(G) * n, 0.0);
  probe.lam_abs.assign(static_cast<std::size_t>(G) * n, 0.0);
  cuda_check(cudaMemcpy(probe.lam_tr.data(), w.lam_tr,
                        static_cast<std::size_t>(G) * n * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB2D probe: lam_tr readback failed");
  cuda_check(cudaMemcpy(probe.lam_abs.data(), w.lam_abs,
                        static_cast<std::size_t>(G) * n * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB2D probe: lam_abs readback failed");

  // Live P_sh export (slot-major [c*8 + slot-1]).
  {
    // reuse prod as the [n*8] staging (G*n >= 8*n requires G >= 8; guard).
    double* stage = nullptr;
    const bool prod_big_enough = (static_cast<std::size_t>(G) * n >=
                                  static_cast<std::size_t>(8) * n);
    stage = prod_big_enough
                ? w.prod
                : static_cast<double*>(core::device_scratch_acquire(
                      "conduction:snb2d:psh_stage",
                      static_cast<std::size_t>(8) * n * sizeof(double)));
    snb2d_psh_export_kernel<<<cell_blocks, kBlockSize>>>(
        stage, d_stencil, state.Te.data(), d_rho_cv_e, d_cell_is_void, nr, nz);
    cuda_check(cudaGetLastError(), "SNB2D probe: psh export launch failed");
    probe.p_sh.assign(static_cast<std::size_t>(8) * n, 0.0);
    cuda_check(cudaMemcpy(probe.p_sh.data(), stage,
                          static_cast<std::size_t>(8) * n * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "SNB2D probe: p_sh readback failed");
  }
  return probe;
}

}  // namespace tenryu::hydro::snb2d
