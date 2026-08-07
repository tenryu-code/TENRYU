#include "hydro/conduction_snb_1d.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <cusparse.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/macros.hpp"
#include "hydro/conduction.cuh"
#include "hydro/conduction_bodies.cuh"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro::snb {

__host__ __device__ double snb_lambda_g_ext(const double eps_erg,
                                            const double n_e,
                                            const double zbar,
                                            const double zmom_r2,
                                            const double ln_lambda,
                                            const int mfp_variant) {
  constexpr double four_pi = 12.566370614359172953850573533118;
  constexpr double esu_charge4 =
      4.8032e-10 * 4.8032e-10 * 4.8032e-10 * 4.8032e-10;
  constexpr double two_sqrt_two = 2.8284271247461902909;
  if (!(n_e > 0.0) || !(zbar > 0.0) || !(ln_lambda > 0.0) ||
      !(eps_erg > 0.0)) {
    return 0.0;
  }
  const double denom_base = four_pi * n_e * esu_charge4 * ln_lambda;
  const double z_eff = zbar * zmom_r2;
  double z_factor;
  double prefactor;
  if (mfp_variant == 1) {
    z_factor = sqrt(z_eff + 1.0);
    prefactor = 2.0;
  } else {
    const double phi = (z_eff + 4.2) / (z_eff + 0.24);
    z_factor = sqrt(z_eff * phi);
    prefactor = two_sqrt_two;
  }
  const double lam = prefactor * eps_erg * eps_erg / (denom_base * z_factor);
  return (isfinite(lam) && lam > 0.0) ? lam : 0.0;
}

namespace {

using conduction_bodies::coulomb_log_formula;
using conduction_bodies::electron_density_formula;
using conduction_bodies::harmonic_mean;
using conduction_bodies::kEvToErg;
using conduction_bodies::kirchhoff_face_kappa_spitzer;
using conduction_bodies::q_max_formula;
using conduction_bodies::sanitize_te_for_pow;
using conduction_bodies::cell_center_radius;
using conduction_bodies::atomic_max_double;
using conduction_bodies::atomic_min_double;
using tenryu::mesh::geometry_1d_face_area;

constexpr int kBlockSize = 256;
constexpr double kFourPi = 12.566370614359172953850573533118;
// e^4 in cgs (esu), e = 4.8032e-10 statC (NRL Plasma Formulary).
constexpr double kEsuCharge4 = 4.8032e-10 * 4.8032e-10 * 4.8032e-10 * 4.8032e-10;
constexpr double kTwoSqrtTwo = 2.8284271247461902909;  // 2*sqrt(2)
constexpr double kBetaMin = 0.1;  // first interior reduced-energy edge (design §3.2)

// Scalar-pack layout (device doubles).
constexpr int kScalarTRef = 0;       // max sanitized Te [eV]
constexpr int kScalarResid = 1;      // max |dq - dq_prev|
constexpr int kScalarQshMax = 2;     // max |q_sh|
constexpr int kScalarDqMax = 3;      // max |dq|
constexpr int kScalarThetaMin = 4;   // min theta over interior faces

inline void cuda_check(const cudaError_t err, const char* message) {
  if (err != cudaSuccess) {
    core::log_error(std::string(message) + ": " + cudaGetErrorString(err));
    std::abort();
  }
}

inline void cusparse_check(const cusparseStatus_t status, const char* message) {
  if (status != CUSPARSE_STATUS_SUCCESS) {
    core::log_error(std::string(message) + ": status " +
                    std::to_string(static_cast<int>(status)));
    std::abort();
  }
}

struct SnbCusparseCache {
  cusparseHandle_t handle = nullptr;
  ~SnbCusparseCache() {
    if (handle != nullptr) {
      static_cast<void>(cusparseDestroy(handle));
    }
  }
};

cusparseHandle_t snb_cusparse_handle() {
  static SnbCusparseCache cache;
  if (cache.handle == nullptr) {
    cusparse_check(cusparseCreate(&cache.handle), "SNB cusparseCreate failed");
  }
  return cache.handle;
}

// Normalized cumulative of the SNB source spectrum: P(b) = int_0^b u^4 e^-u du / 24.
__host__ __device__ inline double snb_spectrum_cdf(const double b) {
  if (!(b > 0.0)) {
    return 0.0;
  }
  const double poly = ((((b + 4.0) * b + 12.0) * b + 24.0) * b + 24.0);
  const double value = 1.0 - exp(-b) * poly / 24.0;
  return fmin(fmax(value, 0.0), 1.0);
}

// Group mfp lambda_g(eps_g; cell) [cm] for the adopted variants (design §1.4).
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

__global__ void snb_max_te_kernel(const double* __restrict__ Te,
                                  const std::uint8_t* __restrict__ cell_is_void,
                                  const int n_cells,
                                  double* __restrict__ scalars) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    return;
  }
  const double te = sanitize_te_for_pow(Te[i]);
  if (te > 0.0) {
    atomic_max_double(&scalars[kScalarTRef], te);
  }
}

// Physical SH face flux q_sh = -kappa_face * dTe/dr on faces f in [0, n_cells].
// Face f sits at x_r[f] between cells f-1 and f; boundary faces carry 0.
__global__ void snb_qsh_face_kernel(double* __restrict__ qsh_face,
                                    const double* __restrict__ Te,
                                    const double* __restrict__ kappa_eff,
                                    const double* __restrict__ x_r,
                                    const int n_cells,
                                    const int face_policy_kirchhoff,
                                    double* __restrict__ scalars) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f > n_cells) {
    return;
  }
  if (f == 0 || f == n_cells) {
    qsh_face[f] = 0.0;
    return;
  }
  const int iL = f - 1;
  const int iR = f;
  const double k_face =
      (face_policy_kirchhoff != 0)
          ? kirchhoff_face_kappa_spitzer(kappa_eff[iL], kappa_eff[iR], Te[iL], Te[iR])
          : harmonic_mean(kappa_eff[iL], kappa_eff[iR]);
  const double dr = cell_center_radius(x_r, iR) - cell_center_radius(x_r, iL);
  double q = 0.0;
  if (k_face > 0.0 && dr > 0.0) {
    q = -k_face * (Te[iR] - Te[iL]) / dr;
  }
  qsh_face[f] = q;
  if (q != 0.0) {
    atomic_max_double(&scalars[kScalarQshMax], fabs(q));
  }
}

// Per-(group, cell) mfp evaluation. E-field variant (efield==1) limits the
// transport mfp by the stopping length against the quasi-static Spitzer field
// (design §1.5): 1/lam_tr += |e E|/eps_g with
// |e E| = kB Te [erg] * |dln n_e + gamma(Z) dln Te| [1/cm].
template <bool kZMom>
__global__ void snb_lambda_kernel(double* __restrict__ lam_tr,
                                  double* __restrict__ lam_abs,
                                  const double* __restrict__ Te,
                                  const double* __restrict__ rho,
                                  const double* __restrict__ zbar,
                                  const double* __restrict__ A_eff,
                                  const std::uint8_t* __restrict__ cell_is_void,
                                  const double* __restrict__ x_r,
                                  const double* __restrict__ edges,
                                  const int n_cells,
                                  const int n_groups,
                                  const int mfp_variant,
                                  const int efield_local,
                                  const double* __restrict__ zmom_r2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cells * n_groups) {
    return;
  }
  const int g = idx / n_cells;
  const int i = idx - g * n_cells;
  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    lam_tr[idx] = 0.0;
    lam_abs[idx] = 0.0;
    return;
  }
  const double te = sanitize_te_for_pow(Te[i]);
  const double z = fmax(zbar[i], 0.0);
  const double rho_i = fmax(rho[i], 0.0);
  const double A_i = fmax(A_eff[i], 1.0e-12);
  const double n_e = electron_density_formula(rho_i, z, A_i);
  const double ln_lambda = coulomb_log_formula(n_e, te, z);
  const double eps = 0.5 * (edges[g] + edges[g + 1]);
  double lam;
  if constexpr (kZMom) {
    lam = snb_lambda_g_ext(
        eps, n_e, z, zmom_r2[i], ln_lambda, mfp_variant);
  } else {
    lam = snb_lambda_g(eps, n_e, z, ln_lambda, mfp_variant);
  }
  lam_abs[idx] = lam;
  double lam_t = lam;
  if (efield_local != 0 && lam > 0.0 && te > 0.0 && eps > 0.0) {
    // Central-difference cell gradients of ln(n_e) and ln(Te); one-sided at ends.
    const int im = (i > 0) ? (i - 1) : i;
    const int ip = (i + 1 < n_cells) ? (i + 1) : i;
    if (ip > im) {
      const double dr = cell_center_radius(x_r, ip) - cell_center_radius(x_r, im);
      if (dr > 0.0) {
        const double te_m = sanitize_te_for_pow(Te[im]);
        const double te_p = sanitize_te_for_pow(Te[ip]);
        const double ne_m = electron_density_formula(
            fmax(rho[im], 0.0), fmax(zbar[im], 0.0), fmax(A_eff[im], 1.0e-12));
        const double ne_p = electron_density_formula(
            fmax(rho[ip], 0.0), fmax(zbar[ip], 0.0), fmax(A_eff[ip], 1.0e-12));
        double grad_ln = 0.0;
        if (te_m > 0.0 && te_p > 0.0 && ne_m > 0.0 && ne_p > 0.0) {
          const double gamma_z = 1.0 + 1.5 * (z + 0.477) / (z + 2.15);
          grad_ln = (log(ne_p) - log(ne_m) + gamma_z * (log(te_p) - log(te_m))) / dr;
        }
        const double e_field_term = kEvToErg * te * fabs(grad_ln) / eps;  // [1/cm]
        if (e_field_term > 0.0 && isfinite(e_field_term)) {
          lam_t = 1.0 / (1.0 / lam + e_field_term);
        }
      }
    }
  }
  lam_tr[idx] = lam_t;
}

// Tridiagonal assembly for the volume-integrated group equation (design §3.3):
//   V_i H_i / lam_abs_i - [A D dH]_faces = -(A_{f+1} xi_{f+1} qsh_{f+1} - A_f xi_f qsh_f)
// Void cells carry the identity row H=0. Face D = harmonic(lam_tr_L, lam_tr_R)/3.
template <int GEOM>
__global__ void snb_assemble_kernel(double* __restrict__ lower,
                                    double* __restrict__ diag,
                                    double* __restrict__ upper,
                                    double* __restrict__ rhs,
                                    const double* __restrict__ lam_tr,
                                    const double* __restrict__ lam_abs,
                                    const double* __restrict__ qsh_face,
                                    const double* __restrict__ Te,
                                    const double* __restrict__ edges,
                                    const std::uint8_t* __restrict__ cell_is_void,
                                    const double* __restrict__ x_r,
                                    const double* __restrict__ vol,
                                    const int n_cells,
                                    const int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_cells * n_groups) {
    return;
  }
  const int g = idx / n_cells;
  const int i = idx - g * n_cells;
  const bool is_void =
      (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0));
  const double lam_a = lam_abs[idx];
  if (is_void || !(lam_a > 0.0)) {
    lower[idx] = 0.0;
    diag[idx] = 1.0;
    upper[idx] = 0.0;
    rhs[idx] = 0.0;
    return;
  }
  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }

  double lo = 0.0;
  double up = 0.0;
  double dg = V / lam_a;
  double rh = 0.0;

  // Left face f = i at x_r[i] (absent for i == 0: reflective, zero flux).
  if (i > 0) {
    const double dr = cell_center_radius(x_r, i) - cell_center_radius(x_r, i - 1);
    double area;
    if constexpr (GEOM == 0) {
      area = kFourPi * x_r[i] * x_r[i];
    } else {
      area = geometry_1d_face_area(GEOM, x_r[i]);
    }
    if (dr > 0.0) {
      const double D = harmonic_mean(lam_tr[idx - 1], lam_tr[idx]) / 3.0;
      const double w = area * D / dr;
      lo = -w;
      dg += w;
      const double te_face =
          0.5 * (sanitize_te_for_pow(Te[i - 1]) + sanitize_te_for_pow(Te[i]));
      double xi = 0.0;
      if (te_face > 0.0) {
        const double kT = kEvToErg * te_face;
        xi = snb_spectrum_cdf(edges[g + 1] / kT) - snb_spectrum_cdf(edges[g] / kT);
      }
      rh += area * xi * qsh_face[i];
    }
  }
  // Right face f = i+1 at x_r[i+1] (absent for i == n_cells-1).
  if (i + 1 < n_cells) {
    const double dr = cell_center_radius(x_r, i + 1) - cell_center_radius(x_r, i);
    double area;
    if constexpr (GEOM == 0) {
      area = kFourPi * x_r[i + 1] * x_r[i + 1];
    } else {
      area = geometry_1d_face_area(GEOM, x_r[i + 1]);
    }
    if (dr > 0.0) {
      const double D = harmonic_mean(lam_tr[idx], lam_tr[idx + 1]) / 3.0;
      const double w = area * D / dr;
      up = -w;
      dg += w;
      const double te_face =
          0.5 * (sanitize_te_for_pow(Te[i]) + sanitize_te_for_pow(Te[i + 1]));
      double xi = 0.0;
      if (te_face > 0.0) {
        const double kT = kEvToErg * te_face;
        xi = snb_spectrum_cdf(edges[g + 1] / kT) - snb_spectrum_cdf(edges[g] / kT);
      }
      rh -= area * xi * qsh_face[i + 1];
    }
  }
  // Neighbor coupling into void cells must vanish (their row is identity):
  if (i > 0 && cell_is_void != nullptr &&
      cell_is_void[i - 1] != static_cast<std::uint8_t>(0)) {
    lo = 0.0;
  }
  if (i + 1 < n_cells && cell_is_void != nullptr &&
      cell_is_void[i + 1] != static_cast<std::uint8_t>(0)) {
    up = 0.0;
  }
  lower[idx] = lo;
  diag[idx] = dg;
  upper[idx] = up;
  rhs[idx] = rh;
}

// Nonlocal correction flux and outer cap per face (design §3.4, §2.2).
// dq_f = -sum_g D_gf (H_{g,f} - H_{g,f-1})/dr_f (fixed group order, deterministic).
__global__ void snb_dq_theta_kernel(double* __restrict__ dq_face,
                                    double* __restrict__ theta_face,
                                    const double* __restrict__ dq_face_prev,
                                    const double* __restrict__ H,
                                    const double* __restrict__ lam_tr,
                                    const double* __restrict__ qsh_face,
                                    const double* __restrict__ Te,
                                    const double* __restrict__ rho,
                                    const double* __restrict__ zbar,
                                    const double* __restrict__ A_eff,
                                    const std::uint8_t* __restrict__ cell_is_void,
                                    const double* __restrict__ x_r,
                                    const int n_cells,
                                    const int n_groups,
                                    const int use_outer_cap,
                                    const double f_lim,
                                    double* __restrict__ scalars,
                                    int* __restrict__ counters) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f > n_cells) {
    return;
  }
  if (f == 0 || f == n_cells) {
    dq_face[f] = 0.0;
    theta_face[f] = 1.0;
    return;
  }
  const int iL = f - 1;
  const int iR = f;
  const bool voidL =
      (cell_is_void != nullptr && cell_is_void[iL] != static_cast<std::uint8_t>(0));
  const bool voidR =
      (cell_is_void != nullptr && cell_is_void[iR] != static_cast<std::uint8_t>(0));
  const double dr = cell_center_radius(x_r, iR) - cell_center_radius(x_r, iL);
  double dq = 0.0;
  if (!voidL && !voidR && dr > 0.0) {
    for (int g = 0; g < n_groups; ++g) {
      const double D =
          harmonic_mean(lam_tr[g * n_cells + iL], lam_tr[g * n_cells + iR]) / 3.0;
      dq -= D * (H[g * n_cells + iR] - H[g * n_cells + iL]) / dr;
    }
  }
  if (!isfinite(dq)) {
    dq = 0.0;
  }
  double theta = 1.0;
  if (use_outer_cap != 0) {
    const double te_face =
        0.5 * (sanitize_te_for_pow(Te[iL]) + sanitize_te_for_pow(Te[iR]));
    const double rho_face = 0.5 * (fmax(rho[iL], 0.0) + fmax(rho[iR], 0.0));
    const double z_face = 0.5 * (fmax(zbar[iL], 0.0) + fmax(zbar[iR], 0.0));
    const double A_L = fmax(A_eff[iL], 1.0e-12);
    const double A_R = fmax(A_eff[iR], 1.0e-12);
    const double A_face = (A_L == A_R) ? A_L : fmax(harmonic_mean(A_L, A_R), 1.0e-12);
    const double n_e_face = electron_density_formula(rho_face, z_face, A_face);
    const double q_max = q_max_formula(f_lim, n_e_face, te_face);
    // Free-streaming bound on the nonlocal correction (design §4.4): the
    // SNB dq through a face cannot exceed the local free-streaming flux.
    // Unconverged dq noise on rarefied faces otherwise saturates the
    // outer cap (theta -> 0) and chokes ALL conduction there
    // (2026-07-31 3step band-ringing hydro blast). Non-binding on
    // healthy faces (|dq| << q_max) — bit-identical there.
    if (q_max > 0.0 && fabs(dq) > q_max) {
      dq = (dq > 0.0) ? q_max : -q_max;
      atomicAdd(&counters[2], 1);
    }
    const double q_tot = qsh_face[f] + dq;
    if (q_max > 0.0) {
      theta = 1.0 / (1.0 + fabs(q_tot) / q_max);
    } else {
      theta = 0.0;
    }
  }
  if (!isfinite(theta)) {
    theta = 0.0;
  }
  theta = fmin(1.0, fmax(theta, 0.0));
  dq_face[f] = dq;
  theta_face[f] = theta;
  if (dq_face_prev != nullptr) {
    atomic_max_double(&scalars[kScalarResid], fabs(dq - dq_face_prev[f]));
  }
  if (dq != 0.0) {
    atomic_max_double(&scalars[kScalarDqMax], fabs(dq));
  }
  atomic_min_double(&scalars[kScalarThetaMin], theta);
  if (theta < 0.99) {
    atomicAdd(&counters[0], 1);
  }
  if (theta < 0.5) {
    atomicAdd(&counters[1], 1);
  }
}

// SNB stage kernels: verbatim clones of the historic stage bodies
// (conduction_bodies.cuh conduction_1d_sts_stage_*_kernel_body) with the face
// flux replaced by theta * (k grad Te - dq_phys). Existing kernels/bodies stay
// byte-untouched (bitwise-safety strategy, design §3.5). The floor throttle is
// inherited as-is (per-cell alpha; BUG-15 family fix arrives via merge train).
template <int GEOM, bool KIRCHHOFF>
__global__ void snb_stage_kernel(const double* __restrict__ Te_old,
                                 double* __restrict__ Te_new,
                                 const double* __restrict__ kappa_sh,
                                 const double* __restrict__ theta_face,
                                 const double* __restrict__ dq_face,
                                 const double* __restrict__ rho_cv_e,
                                 const std::uint8_t* __restrict__ cell_is_void,
                                 const double* __restrict__ vol,
                                 const double* __restrict__ x_r,
                                 const int n_cells,
                                 const double tau,
                                 const double Te_floor,
                                 int* __restrict__ clamp_count,
                                 double* __restrict__ E_floor,
                                 const double te_ceiling,
                                 int* __restrict__ ceiling_count,
                                 double* __restrict__ E_ceiling) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    Te_new[i] = Te_old[i];
    return;
  }

  double V;
  if constexpr (GEOM == 2) {
    V = fmax(x_r[i + 1] - x_r[i], 1.0e-30);
  } else {
    V = fmax(vol[i], 1.0e-30);
  }

  double left_term = 0.0;
  if (i > 0) {
    double k_face;
    if constexpr (KIRCHHOFF) {
      k_face = kirchhoff_face_kappa_spitzer(kappa_sh[i - 1], kappa_sh[i],
                                            Te_old[i - 1], Te_old[i]);
    } else {
      k_face = harmonic_mean(kappa_sh[i - 1], kappa_sh[i]);
    }
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    double q_sh_code = 0.0;
    if (k_face > 0.0 && dr > 0.0) {
      q_sh_code = k_face * (Te_old[i] - Te_old[i - 1]) / dr;
    }
    const double q_face = theta_face[i] * (q_sh_code - dq_face[i]);
    double area;
    if constexpr (GEOM == 0) {
      area = kFourPi * x_r[i] * x_r[i];
    } else {
      area = geometry_1d_face_area(GEOM, x_r[i]);
    }
    left_term = area * q_face;
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    double k_face;
    if constexpr (KIRCHHOFF) {
      k_face = kirchhoff_face_kappa_spitzer(kappa_sh[i], kappa_sh[i + 1],
                                            Te_old[i], Te_old[i + 1]);
    } else {
      k_face = harmonic_mean(kappa_sh[i], kappa_sh[i + 1]);
    }
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    double q_sh_code = 0.0;
    if (k_face > 0.0 && dr > 0.0) {
      q_sh_code = k_face * (Te_old[i + 1] - Te_old[i]) / dr;
    }
    const double q_face = theta_face[i + 1] * (q_sh_code - dq_face[i + 1]);
    double area;
    if constexpr (GEOM == 0) {
      area = kFourPi * x_r[i + 1] * x_r[i + 1];
    } else {
      area = geometry_1d_face_area(GEOM, x_r[i + 1]);
    }
    right_term = area * q_face;
  }

  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  double Te_computed = Te_old_i;
  if (rho_cv > 0.0) {
    double alpha = 1.0;
    if (tau > 0.0 && Te_old_i > Te_floor && M < 0.0) {
      const double e_above_floor = rho_cv * (Te_old_i - Te_floor);
      const double dt_safe = e_above_floor / fmax(fabs(M), 1.0e-30);
      alpha = fmin(1.0, fmax(dt_safe / tau, 0.0));
    }
    Te_computed += tau * alpha * M / rho_cv;
  }

  if (!isfinite(Te_computed) || Te_computed < Te_floor) {
    if (rho_cv > 0.0) {
      if (isfinite(Te_computed) && Te_floor > Te_computed) {
        const double de = rho_cv * (Te_floor - Te_computed) * V;
        conduction_bodies::atomic_add_double(E_floor, de);
      } else if (!isfinite(Te_computed) && Te_floor > 0.0) {
        const double de = rho_cv * Te_floor * V;
        conduction_bodies::atomic_add_double(E_floor, de);
      }
    }
    Te_new[i] = Te_floor;
    atomicAdd(clamp_count, 1);
    return;
  }

  // Discrete maximum principle (NUMERICS 4.4): transport alone cannot
  // raise any cell above the entry-state global max Te. A frozen dq into
  // a tiny-rho_cv cell would otherwise integrate without bound
  // (2026-07-30 3step SNB runaway, Te -> 6.8e5 eV at rho = 5e-8).
  if (te_ceiling > 0.0 && Te_computed > te_ceiling) {
    if (rho_cv > 0.0) {
      const double de = rho_cv * (Te_computed - te_ceiling) * V;
      conduction_bodies::atomic_add_double(E_ceiling, de);
    }
    Te_new[i] = te_ceiling;
    atomicAdd(ceiling_count, 1);
    return;
  }

  Te_new[i] = Te_computed;
}

void solve_group_tridiag(const SnbDeviceWork& work, const int n_cells,
                         const int n_groups) {
  cusparseHandle_t handle = snb_cusparse_handle();
  std::size_t buffer_size = 0U;
  cusparse_check(cusparseDgtsv2StridedBatch_bufferSizeExt(
                     handle, n_cells, work.lower, work.diag, work.upper, work.rhs,
                     n_groups, n_cells, &buffer_size),
                 "SNB cusparse buffer size failed");
  void* buffer = core::device_scratch_acquire("conduction:snb:cusparse_buffer",
                                              std::max<std::size_t>(buffer_size, 8U));
  cusparse_check(cusparseDgtsv2StridedBatch(handle, n_cells, work.lower, work.diag,
                                            work.upper, work.rhs, n_groups, n_cells,
                                            buffer),
                 "SNB cusparse tridiagonal solve failed");
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

void snb_fill_t_ref(core::State& state, const std::uint8_t* d_cell_is_void,
                    const int n_cells, double* work_scalars) {
  const int blocks = (n_cells + kBlockSize - 1) / kBlockSize;
  snb_max_te_kernel<<<blocks, kBlockSize>>>(state.Te.data(), d_cell_is_void,
                                            n_cells, work_scalars);
  cuda_check(cudaGetLastError(), "SNB: max-Te launch failed");
}

void snb_pass(core::State& state,
              const core::Config& cfg,
              const SnbDeviceWork& work,
              const double* d_kappa_eff,
              const double* d_A_eff,
              const std::uint8_t* d_cell_is_void,
              const int n_cells,
              const int geom_code,
              const bool face_policy_kirchhoff,
              const bool use_outer_cap,
              const double t_ref_ev,
              double* resid_out,
              double* qsh_scale_out,
              double* dq_ratio_out,
              const double* zmom_r2) {
  const auto& sc = cfg.numerics.conduction;
  const int n_groups = sc.snb_n_groups;
  const int mfp_variant = (sc.snb_mfp == "original") ? 1 : 0;
  const int efield_local = (sc.snb_efield == "local") ? 1 : 0;

  // Host group edges in erg from the frozen reduced ladder.
  const std::vector<double> beta_edges = group_edges_beta(n_groups, sc.snb_E_max_over_Te);
  std::vector<double> edges_erg(beta_edges.size(), 0.0);
  for (std::size_t k = 0; k < beta_edges.size(); ++k) {
    edges_erg[k] = beta_edges[k] * conduction_bodies::kEvToErg * t_ref_ev;
  }
  cuda_check(cudaMemcpy(work.edges, edges_erg.data(),
                        edges_erg.size() * sizeof(double), cudaMemcpyHostToDevice),
             "SNB: edges upload failed");

  // Zero the per-pass reduction slots (resid/qsh/dq) and counters; keep TRef.
  const double zeros3[3] = {0.0, 0.0, 0.0};
  cuda_check(cudaMemcpy(work.scalars + kScalarResid, zeros3, sizeof(zeros3),
                        cudaMemcpyHostToDevice),
             "SNB: scalar reset failed");
  const double theta_min_init = 1.0;
  cuda_check(cudaMemcpy(work.scalars + kScalarThetaMin, &theta_min_init,
                        sizeof(double), cudaMemcpyHostToDevice),
             "SNB: theta-min reset failed");
  cuda_check(cudaMemset(work.counters, 0, 3 * sizeof(int)),
             "SNB: counter reset failed");

  const int face_blocks = (n_cells + 1 + kBlockSize - 1) / kBlockSize;
  snb_qsh_face_kernel<<<face_blocks, kBlockSize>>>(
      work.qsh_face, state.Te.data(), d_kappa_eff, state.x_r.data(), n_cells,
      face_policy_kirchhoff ? 1 : 0, work.scalars);
  cuda_check(cudaGetLastError(), "SNB: qsh face launch failed");

  const int gc_blocks = (n_cells * n_groups + kBlockSize - 1) / kBlockSize;
  if (zmom_r2 != nullptr) {
    snb_lambda_kernel<true><<<gc_blocks, kBlockSize>>>(
        work.lam_tr, work.lam_abs, state.Te.data(), state.rho.data(),
        state.zbar.data(), d_A_eff, d_cell_is_void, state.x_r.data(), work.edges,
        n_cells, n_groups, mfp_variant, efield_local, zmom_r2);
  } else {
    snb_lambda_kernel<false><<<gc_blocks, kBlockSize>>>(
        work.lam_tr, work.lam_abs, state.Te.data(), state.rho.data(),
        state.zbar.data(), d_A_eff, d_cell_is_void, state.x_r.data(), work.edges,
        n_cells, n_groups, mfp_variant, efield_local, nullptr);
  }
  cuda_check(cudaGetLastError(), "SNB: lambda launch failed");

  auto launch_assemble = [&](auto geom_tag) {
    constexpr int GEOM = decltype(geom_tag)::value;
    snb_assemble_kernel<GEOM><<<gc_blocks, kBlockSize>>>(
        work.lower, work.diag, work.upper, work.rhs, work.lam_tr, work.lam_abs,
        work.qsh_face, state.Te.data(), work.edges, d_cell_is_void,
        state.x_r.data(), state.vol.data(), n_cells, n_groups);
  };
  switch (geom_code) {
    case 1:
      launch_assemble(std::integral_constant<int, 1>{});
      break;
    case 2:
      launch_assemble(std::integral_constant<int, 2>{});
      break;
    default:
      launch_assemble(std::integral_constant<int, 0>{});
      break;
  }
  cuda_check(cudaGetLastError(), "SNB: assemble launch failed");

  solve_group_tridiag(work, n_cells, n_groups);

  snb_dq_theta_kernel<<<face_blocks, kBlockSize>>>(
      work.dq_face, work.theta_face, work.dq_face_prev, work.rhs, work.lam_tr,
      work.qsh_face, state.Te.data(), state.rho.data(), state.zbar.data(), d_A_eff,
      d_cell_is_void, state.x_r.data(), n_cells, n_groups, use_outer_cap ? 1 : 0,
      sc.f_lim, work.scalars, work.counters);
  cuda_check(cudaGetLastError(), "SNB: dq/theta launch failed");

  double host_scalars[5] = {0.0, 0.0, 0.0, 0.0, 1.0};
  cuda_check(cudaMemcpy(host_scalars, work.scalars, sizeof(host_scalars),
                        cudaMemcpyDeviceToHost),
             "SNB: scalar readback failed");
  if (resid_out != nullptr) {
    *resid_out = host_scalars[kScalarResid];
  }
  if (qsh_scale_out != nullptr) {
    *qsh_scale_out = host_scalars[kScalarQshMax];
  }
  if (dq_ratio_out != nullptr) {
    const double qsh_max = host_scalars[kScalarQshMax];
    const double dq_max = host_scalars[kScalarDqMax];
    *dq_ratio_out = (qsh_max > 0.0) ? (dq_max / qsh_max) : 0.0;
  }
}

void conduction_step_1d_sts_snb(core::State& state,
                                const core::Config& cfg,
                                ConductionResult& result,
                                const double* d_kappa_eff,
                                const double* d_rho_cv_e,
                                const std::uint8_t* d_cell_is_void,
                                const double* d_A_eff,
                                const int n_cells,
                                const int geom_code,
                                const bool face_policy_kirchhoff,
                                const bool use_outer_cap,
                                const std::vector<double>& tau,
                                const int n_sub,
                                const double te_floor,
                                int* d_clamp_count,
                                double* d_e_floor,
                                const double* zmom_r2) {
  const auto& sc = cfg.numerics.conduction;
  const int n_groups = sc.snb_n_groups;
  const std::size_t nf = static_cast<std::size_t>(n_cells) + 1;
  const std::size_t ngc = static_cast<std::size_t>(n_groups) *
                          static_cast<std::size_t>(n_cells);

  SnbDeviceWork work;
  auto acquire = [&](const char* tag, const std::size_t bytes) {
    return core::device_scratch_acquire(tag, bytes);
  };
  work.edges = static_cast<double*>(acquire("conduction:snb:edges",
                                            (static_cast<std::size_t>(n_groups) + 1) *
                                                sizeof(double)));
  work.lam_tr = static_cast<double*>(acquire("conduction:snb:lam_tr", ngc * sizeof(double)));
  work.lam_abs = static_cast<double*>(acquire("conduction:snb:lam_abs", ngc * sizeof(double)));
  work.lower = static_cast<double*>(acquire("conduction:snb:lower", ngc * sizeof(double)));
  work.diag = static_cast<double*>(acquire("conduction:snb:diag", ngc * sizeof(double)));
  work.upper = static_cast<double*>(acquire("conduction:snb:upper", ngc * sizeof(double)));
  work.rhs = static_cast<double*>(acquire("conduction:snb:rhs", ngc * sizeof(double)));
  work.qsh_face = static_cast<double*>(acquire("conduction:snb:qsh_face", nf * sizeof(double)));
  work.dq_face = static_cast<double*>(acquire("conduction:snb:dq_face", nf * sizeof(double)));
  work.dq_face_prev =
      static_cast<double*>(acquire("conduction:snb:dq_face_prev", nf * sizeof(double)));
  work.theta_face =
      static_cast<double*>(acquire("conduction:snb:theta_face", nf * sizeof(double)));
  work.te_stash = static_cast<double*>(acquire("conduction:snb:te_stash",
                                               static_cast<std::size_t>(n_cells) *
                                                   sizeof(double)));
  work.scalars = static_cast<double*>(acquire("conduction:snb:scalars", 8 * sizeof(double)));
  work.counters = static_cast<int*>(acquire("conduction:snb:counters", 3 * sizeof(int)));
  int* d_ceiling_count = static_cast<int*>(
      acquire("conduction:snb:ceiling_count", sizeof(int)));
  double* d_e_ceiling = static_cast<double*>(
      acquire("conduction:snb:e_ceiling", sizeof(double)));

  // Frozen group reference temperature from Te^n (design §3.2).
  cuda_check(cudaMemset(work.scalars, 0, 8 * sizeof(double)), "SNB: scalars init failed");
  const int cell_blocks = (n_cells + kBlockSize - 1) / kBlockSize;
  snb_fill_t_ref(state, d_cell_is_void, n_cells, work.scalars);
  double t_ref_ev = 0.0;
  cuda_check(cudaMemcpy(&t_ref_ev, work.scalars + kScalarTRef, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB: T_ref readback failed");
  // Max-principle ceiling: entry-state global max Te (0 disables when
  // the state is all-cold or the max sits below the floor).
  const double te_ceiling =
      (t_ref_ev > 0.0 && t_ref_ev >= te_floor) ? t_ref_ev : 0.0;
  // Test-only hook (verify snb_max_principle_1d): override the ceiling to
  // force engagement of the clamp/ledger machinery on a healthy profile.
  // Unset in production; reading getenv per application is host-side and
  // leaves the unset path bit-identical.
  double te_ceiling_eff = te_ceiling;
  if (const char* env = std::getenv("TENRYU_SNB_TEST_CEILING_EV")) {
    const double v = std::atof(env);
    if (v > 0.0) {
      te_ceiling_eff = v;
    }
  }
  result.snb_picard_iters = 0;
  result.snb_converged = true;
  result.snb_picard_resid = 0.0;
  result.snb_cap_faces_99 = 0;
  result.snb_cap_faces_50 = 0;
  result.snb_cap_theta_min = 1.0;
  result.snb_dq_over_qsh_max = 0.0;
  if (!(t_ref_ev > 0.0) || n_cells < 3) {
    // all-cold/void, or too few cells for the batched tridiagonal solver
    // (cusparseDgtsv2StridedBatch requires n >= 3): SNB correction is zero.
    return;
  }

  // Stash Te^n; iterate: restore -> solve(theta^{m-1}, dq^{m-1}) -> refresh dq.
  cuda_check(cudaMemcpy(work.te_stash, state.Te.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "SNB: Te stash failed");
  cuda_check(cudaMemset(work.dq_face, 0, nf * sizeof(double)), "SNB: dq init failed");
  cuda_check(cudaMemset(work.dq_face_prev, 0, nf * sizeof(double)),
             "SNB: dq prev init failed");
  {
    // theta^0 from the SH flux alone (OFF-path-equivalent outer cap).
    double resid0 = 0.0;
    double qsh0 = 0.0;
    double ratio0 = 0.0;
    snb_pass(state, cfg, work, d_kappa_eff, d_A_eff, d_cell_is_void, n_cells,
             geom_code, face_policy_kirchhoff, use_outer_cap, t_ref_ev, &resid0,
             &qsh0, &ratio0, zmom_r2);
    // The pass above also produced dq^0 from H(Te^n); keep it as the first
    // source iterate (Cao's k=1 solve uses S=0; seeding with dq(Te^n) is the
    // same fixed point and saves one iterate).
  }

  double* d_te_tmp = static_cast<double*>(
      acquire("conduction:snb:te_tmp", static_cast<std::size_t>(n_cells) * sizeof(double)));

  const int max_iters = std::max(2, sc.snb_picard_max_iters);
  double resid = std::numeric_limits<double>::infinity();
  double qsh_scale = 0.0;
  double dq_ratio = 0.0;
  int iters = 0;
  bool converged = false;
  for (int m = 1; m <= max_iters; ++m) {
    // Restore Te^n (first iterate starts from the untouched state).
    if (m > 1) {
      cuda_check(cudaMemcpy(state.Te.data(), work.te_stash,
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyDeviceToDevice),
                 "SNB: Te restore failed");
    }
    // Accepted-solve-only clamp accounting: reset the pack every iterate.
    const double clampfloor_init[2] = {0.0, 0.0};
    cuda_check(cudaMemcpy(d_e_floor, clampfloor_init, sizeof(clampfloor_init),
                          cudaMemcpyHostToDevice),
               "SNB: clampfloor reset failed");
    // Accepted-iterate-only accounting, mirroring the floor pack.
    cuda_check(cudaMemset(d_ceiling_count, 0, sizeof(int)),
               "SNB: ceiling count reset failed");
    cuda_check(cudaMemset(d_e_ceiling, 0, sizeof(double)),
               "SNB: ceiling energy reset failed");

    double* te_curr = state.Te.data();
    double* te_next = d_te_tmp;
    auto launch_stage = [&](const double tau_stage) {
      auto launch = [&](auto geom_tag) {
        constexpr int GEOM = decltype(geom_tag)::value;
        if (face_policy_kirchhoff) {
          snb_stage_kernel<GEOM, true><<<cell_blocks, kBlockSize>>>(
              te_curr, te_next, d_kappa_eff, work.theta_face, work.dq_face,
              d_rho_cv_e, d_cell_is_void, state.vol.data(), state.x_r.data(),
              n_cells, tau_stage, te_floor, d_clamp_count, d_e_floor,
              te_ceiling_eff,
              d_ceiling_count, d_e_ceiling);
        } else {
          snb_stage_kernel<GEOM, false><<<cell_blocks, kBlockSize>>>(
              te_curr, te_next, d_kappa_eff, work.theta_face, work.dq_face,
              d_rho_cv_e, d_cell_is_void, state.vol.data(), state.x_r.data(),
              n_cells, tau_stage, te_floor, d_clamp_count, d_e_floor,
              te_ceiling_eff,
              d_ceiling_count, d_e_ceiling);
        }
      };
      switch (geom_code) {
        case 1:
          launch(std::integral_constant<int, 1>{});
          break;
        case 2:
          launch(std::integral_constant<int, 2>{});
          break;
        default:
          launch(std::integral_constant<int, 0>{});
          break;
      }
      cuda_check(cudaGetLastError(), "SNB: stage launch failed");
    };
    for (int sub = 0; sub < n_sub; ++sub) {
      for (const double tau_j : tau) {
        launch_stage(tau_j);
        std::swap(te_curr, te_next);
      }
    }
    if (te_curr != state.Te.data()) {
      cuda_check(cudaMemcpy(state.Te.data(), te_curr,
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyDeviceToDevice),
                 "SNB: staged Te copy back failed");
    }
    iters = m;

    // Refresh dq/theta from the iterate's end state; converged when the source
    // stopped moving (max-norm, order-independent; design §2.3).
    std::swap(work.dq_face, work.dq_face_prev);
    snb_pass(state, cfg, work, d_kappa_eff, d_A_eff, d_cell_is_void, n_cells,
             geom_code, face_policy_kirchhoff, use_outer_cap, t_ref_ev, &resid,
             &qsh_scale, &dq_ratio, zmom_r2);
    const double scale = qsh_scale + dq_ratio * qsh_scale;
    if (m >= 2 && resid <= sc.snb_picard_rtol * std::max(scale, 1.0e-300)) {
      converged = true;
      break;
    }
  }

  if (!converged) {
    static bool warned_snb_nonconv = false;
    if (!warned_snb_nonconv) {
      core::log_warning("SNB Picard did not converge within snb_picard_max_iters; "
                        "accepting last iterate (further occurrences suppressed)");
      warned_snb_nonconv = true;
    }
  }

  int host_counters[3] = {0, 0, 0};
  cuda_check(cudaMemcpy(host_counters, work.counters, sizeof(host_counters),
                        cudaMemcpyDeviceToHost),
             "SNB: counter readback failed");
  double host_scalars[8] = {0.0};
  cuda_check(cudaMemcpy(host_scalars, work.scalars, sizeof(host_scalars),
                        cudaMemcpyDeviceToHost),
             "SNB: final scalar readback failed");
  int host_ceiling_count = 0;
  cuda_check(cudaMemcpy(&host_ceiling_count, d_ceiling_count, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "SNB: ceiling count readback failed");
  double host_e_ceiling = 0.0;
  cuda_check(cudaMemcpy(&host_e_ceiling, d_e_ceiling, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "SNB: ceiling energy readback failed");
  result.snb_picard_iters = iters;
  result.snb_converged = converged;
  result.snb_picard_resid = resid;
  result.snb_cap_faces_99 = host_counters[0];
  result.snb_cap_faces_50 = host_counters[1];
  result.snb_cap_theta_min = host_scalars[kScalarThetaMin];
  result.snb_dq_over_qsh_max = dq_ratio;
  result.snb_ceiling_clamp_count = host_ceiling_count;
  result.snb_E_ceiling_removed = host_e_ceiling;
  static bool warned_snb_dq_clamp = false;
  if (host_counters[2] > 0 && !warned_snb_dq_clamp) {
    core::log_warning(
        "SNB dq free-streaming clamp engaged: " + std::to_string(host_counters[2]) +
        " faces this application (further occurrences suppressed)");
    warned_snb_dq_clamp = true;
  }
  static bool warned_snb_ceiling = false;
  if (host_ceiling_count > 0 && !warned_snb_ceiling) {
    core::log_warning(
        "SNB max-principle ceiling engaged: " + std::to_string(host_ceiling_count) +
        " cells clamped to T_ref=" + std::to_string(te_ceiling_eff) +
        " eV this application, E_removed=" + std::to_string(host_e_ceiling) +
        " erg (further occurrences suppressed)");
    warned_snb_ceiling = true;
  }
}

}  // namespace tenryu::hydro::snb
