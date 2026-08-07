#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro::conduction_bodies {

using tenryu::mesh::geometry_1d_face_area;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kFourPi = 12.566370614359172953850573533118;
constexpr double kEvToErg = 1.6022e-12;
constexpr double kElectronMass = 9.1094e-28;
constexpr double kProtonMass = 1.6726219e-24;
constexpr double kDivEpsilon = 1.0e-30;
constexpr double kMaxTeForPow = 1.0e8;
constexpr double kMinEffectiveA = 1.0e-12;
constexpr double kMinEffectiveGamma = 1.0 + 1.0e-12;

__host__ __device__ inline double sanitize_te_for_pow(const double Te_eV) {
  return (isfinite(Te_eV) && Te_eV > 0.0) ? fmin(Te_eV, kMaxTeForPow) : 0.0;
}

__host__ __device__ inline double harmonic_mean(const double a, const double b) {
  if (!(a > 0.0) || !(b > 0.0)) {
    return 0.0;
  }
  return (2.0 * a * b) / (a + b);
}

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

__host__ __device__ inline double spitzer_es_xi(const double Zbar) {
  const double z = ::fmax(Zbar, 1.0e-6);
  return ((z + 0.24) / (z + 4.2)) * (5.2 / 1.24);
}

__host__ __device__ inline double spitzer_kappa_formula(const double Te_eV,
                                                        const double Zbar,
                                                        const double ln_lambda) {
  constexpr double kappa0 = 3.06e9;
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(te > 0.0) || !(Zbar > 0.0) || !(ln_lambda > 0.0)) {
    return 0.0;
  }
  const double kappa =
      kappa0 * spitzer_es_xi(Zbar) * pow(te, 2.5) / (Zbar * ln_lambda);
  return isfinite(kappa) ? kappa : 0.0;
}

__host__ __device__ inline double spitzer_kappa_formula_ext(
    const double Te_eV,
    const double Zbar,
    const double ln_lambda,
    const double zmom_r2) {
  // Z_eff = Zbar * <Z^2>/Zbar^2 is the mixture collision charge.
  const double Z_coll = Zbar * zmom_r2;
  return spitzer_kappa_formula(Te_eV, Z_coll, ln_lambda);
}

__host__ __device__ inline double vth_e_formula(const double Te_eV) {
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(te > 0.0)) {
    return 0.0;
  }
  const double vth = sqrt(kEvToErg * te / kElectronMass);
  return isfinite(vth) ? vth : 0.0;
}

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

__host__ __device__ inline double electron_collision_frequency_formula(
    const double n_e,
    const double Te_eV,
    const double Zbar,
    const double ln_lambda) {
  const double te = sanitize_te_for_pow(Te_eV);
  if (!(n_e > 0.0) || !(te > 0.0) || !(Zbar > 0.0) || !(ln_lambda > 0.0)) {
    return 0.0;
  }
  const double te32 = te * sqrt(te);
  if (!(te32 > 0.0)) {
    return 0.0;
  }
  const double nu_ei = 2.91e-6 * n_e * Zbar * ln_lambda / te32;
  return isfinite(nu_ei) ? fmax(nu_ei, 0.0) : 0.0;
}

__host__ __device__ inline double mfp_limiter_factor_formula(
    const double n_e,
    const double Te_eV,
    const double Zbar,
    const double ln_lambda,
    const double dl,
    const double mfp_limiter_C) {
  if (!(mfp_limiter_C > 0.0) || !(dl > 0.0)) {
    return 1.0;
  }
  const double nu_ei =
      electron_collision_frequency_formula(n_e, Te_eV, Zbar, ln_lambda);
  const double v_th = vth_e_formula(Te_eV);
  const double lambda_mfp = (nu_ei > 0.0 && v_th > 0.0) ? (v_th / nu_ei) : 1.0e30;
  if (!(lambda_mfp > 0.0) || !isfinite(lambda_mfp)) {
    return 0.0;
  }
  const double ratio = (mfp_limiter_C * dl) / lambda_mfp;
  if (!isfinite(ratio)) {
    return 0.0;
  }
  return fmin(1.0, fmax(ratio, 0.0));
}

__host__ __device__ inline double flux_limiter_formula(const double q_sh,
                                                       const double q_max) {
  if (!(q_max > 0.0)) {
    return 0.0;
  }
  return q_sh / (1.0 + fabs(q_sh) / q_max);
}

__host__ __device__ inline double deff_formula(const double q_limited,
                                               const double rho,
                                               const double cv_e,
                                               const double grad_Te,
                                               const double D_sh,
                                               const double eps_grad) {
  const double g = fabs(grad_Te);
  if (g < eps_grad) {
    return D_sh;
  }
  const double denom = rho * cv_e * g;
  if (!(denom > 0.0)) {
    return D_sh;
  }
  return fabs(q_limited) / denom;
}

__device__ inline double atomic_min_double(double* address, const double value) {
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

__device__ inline double atomic_max_double(double* address, const double value) {
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

__device__ inline double cell_center_radius(const double* x_r, const int i) {
  return 0.5 * (x_r[i] + x_r[i + 1]);
}

__device__ inline double kirchhoff_face_kappa_spitzer(const double kap_l,
                                                      const double kap_r,
                                                      const double te_l_raw,
                                                      const double te_r_raw) {
  const double te_l = sanitize_te_for_pow(te_l_raw);
  const double te_r = sanitize_te_for_pow(te_r_raw);
  if (!(te_l > 0.0) || !(te_r > 0.0) || !(kap_l > 0.0) || !(kap_r > 0.0)) {
    return harmonic_mean(kap_l, kap_r);
  }
  const double kap0_l = kap_l / pow(te_l, 2.5);
  const double kap0_r = kap_r / pow(te_r, 2.5);
  const double ratio = kap0_r / kap0_l;
  if (!(ratio > 0.1 && ratio < 10.0)) {
    return harmonic_mean(kap_l, kap_r);
  }
  const double kap0_f = harmonic_mean(kap0_l, kap0_r);
  const double t_m = 0.5 * (te_l + te_r);
  const double x = (te_r - te_l) / t_m;
  double s52;
  if (fabs(x) < 1.0e-4) {
    s52 = pow(t_m, 2.5) * (1.0 + (2.5 * 1.5 / 24.0) * x * x);
  } else {
    s52 = (2.0 / 7.0) * (pow(te_r, 3.5) - pow(te_l, 3.5)) / (te_r - te_l);
  }
  return kap0_f * s52;
}

template <bool kZMom>
__device__ inline void compute_spitzer_deff_1d_kernel_body(
    const int i,
    double* __restrict__ kappa_eff,
    double* __restrict__ rho_cv_e,
    double* __restrict__ min_ratio,
    double* __restrict__ min_deff,
    double* __restrict__ max_deff,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const std::uint8_t* __restrict__ cell_is_void,
    const double f_lim,
    const double test_kappa,
    const double mfp_limiter_C,
    const double* __restrict__ state_cv_e,
    const double* __restrict__ zmom_r2) {
  const double z = fmax(zbar[i], 0.0);
  const double rho_i = fmax(rho[i], 0.0);
  const double te = sanitize_te_for_pow(Te[i]);
  const double gamma_i = fmax(gamma_eff[i], kMinEffectiveGamma);
  const double A_i = fmax(A_eff[i], kMinEffectiveA);
  const double cv_e_ideal = (gamma_i > 1.0 && A_i > 0.0)
                                ? (z * kEvToErg / (A_i * kProtonMass * (gamma_i - 1.0)))
                                : 0.0;
  const double cv_e =
      (state_cv_e != nullptr && state_cv_e[i] > 0.0) ? state_cv_e[i] : cv_e_ideal;
  const double rho_cv = rho_i * cv_e;

  if (rho_cv_e != nullptr) {
    rho_cv_e[i] = rho_cv;
  }

  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    if (kappa_eff != nullptr) {
      kappa_eff[i] = 0.0;
    }
    return;
  }

  double kappa_local = 0.0;
  double D_eff = 0.0;

  if (test_kappa > 0.0) {
    kappa_local = test_kappa;
    if (rho_cv > 0.0) {
      D_eff = kappa_local / rho_cv;
    }
  } else if (rho_cv > 0.0 && z > 0.0 && te > 0.0) {
    const double n_e = electron_density_formula(rho_i, z, A_i);
    const double ln_lambda = coulomb_log_formula(n_e, te, z);
    double kappa_sh;
    if constexpr (kZMom) {
      const double r2 = (zmom_r2 != nullptr) ? zmom_r2[i] : 1.0;
      kappa_sh = spitzer_kappa_formula_ext(te, z, ln_lambda, r2);
    } else {
      kappa_sh = spitzer_kappa_formula(te, z, ln_lambda);
    }
    kappa_local = fmax(kappa_sh, 0.0);
    if (kappa_local > 0.0 && mfp_limiter_C > 0.0) {
      const double dl_cell = x_r[i + 1] - x_r[i];
      const double limiter =
          mfp_limiter_factor_formula(n_e, te, z, ln_lambda, dl_cell, mfp_limiter_C);
      kappa_local *= limiter;
    }
    const double D_sh = (rho_cv > 0.0) ? (kappa_local / rho_cv) : 0.0;

    double grad = 0.0;
    if (n_cells > 1) {
      if (i == 0) {
        const double rc0 = cell_center_radius(x_r, 0);
        const double rc1 = cell_center_radius(x_r, 1);
        const double dr = rc1 - rc0;
        grad = (dr > 0.0) ? ((Te[1] - Te[0]) / dr) : 0.0;
      } else if (i == n_cells - 1) {
        const double rcl = cell_center_radius(x_r, n_cells - 2);
        const double rcr = cell_center_radius(x_r, n_cells - 1);
        const double dr = rcr - rcl;
        grad = (dr > 0.0) ? ((Te[n_cells - 1] - Te[n_cells - 2]) / dr) : 0.0;
      } else {
        const double rcl = cell_center_radius(x_r, i - 1);
        const double rcr = cell_center_radius(x_r, i + 1);
        const double dr = rcr - rcl;
        grad = (dr > 0.0) ? ((Te[i + 1] - Te[i - 1]) / dr) : 0.0;
      }
    }

    double q_max = q_max_formula(f_lim, n_e, te);
    double te_face = te;
    if (n_cells > 1) {
      if (i == 0) {
        te_face = 0.5 * (te + sanitize_te_for_pow(Te[1]));
        const double rho_face = 0.5 * (rho_i + fmax(rho[1], 0.0));
        const double z_face = 0.5 * (z + fmax(zbar[1], 0.0));
        const double A_1 = fmax(A_eff[1], kMinEffectiveA);
        const double A_face = (A_i == A_1)
                                  ? A_i
                                  : fmax(harmonic_mean(A_i, A_1), kMinEffectiveA);
        q_max =
            q_max_formula(f_lim, electron_density_formula(rho_face, z_face, A_face), te_face);
      } else if (i == n_cells - 1) {
        te_face = 0.5 * (sanitize_te_for_pow(Te[n_cells - 2]) + te);
        const double rho_face = 0.5 * (fmax(rho[n_cells - 2], 0.0) + rho_i);
        const double z_face = 0.5 * (fmax(zbar[n_cells - 2], 0.0) + z);
        const double A_nm1 = fmax(A_eff[n_cells - 2], kMinEffectiveA);
        const double A_face = (A_nm1 == A_i)
                                  ? A_i
                                  : fmax(harmonic_mean(A_nm1, A_i), kMinEffectiveA);
        q_max =
            q_max_formula(f_lim, electron_density_formula(rho_face, z_face, A_face), te_face);
      } else {
        const double te_im1 = sanitize_te_for_pow(Te[i - 1]);
        const double te_ip1 = sanitize_te_for_pow(Te[i + 1]);
        const double te_face_l = 0.5 * (te_im1 + te);
        const double te_face_r = 0.5 * (te + te_ip1);
        const double rho_face_l = 0.5 * (fmax(rho[i - 1], 0.0) + rho_i);
        const double rho_face_r = 0.5 * (rho_i + fmax(rho[i + 1], 0.0));
        const double z_face_l = 0.5 * (fmax(zbar[i - 1], 0.0) + z);
        const double z_face_r = 0.5 * (z + fmax(zbar[i + 1], 0.0));
        const double A_im1 = fmax(A_eff[i - 1], kMinEffectiveA);
        const double A_ip1 = fmax(A_eff[i + 1], kMinEffectiveA);
        const double A_face_l =
            (A_im1 == A_i) ? A_i : fmax(harmonic_mean(A_im1, A_i), kMinEffectiveA);
        const double A_face_r =
            (A_i == A_ip1) ? A_i : fmax(harmonic_mean(A_i, A_ip1), kMinEffectiveA);
        const double q_max_l = q_max_formula(
            f_lim, electron_density_formula(rho_face_l, z_face_l, A_face_l), te_face_l);
        const double q_max_r = q_max_formula(
            f_lim, electron_density_formula(rho_face_r, z_face_r, A_face_r), te_face_r);
        q_max = 0.5 * (q_max_l + q_max_r);
        te_face = 0.5 * (te_face_l + te_face_r);
      }
    }
    const double q_sh = isfinite(kappa_local) ? (-kappa_local * grad) : 0.0;
    const double q_limited = flux_limiter_formula(isfinite(q_sh) ? q_sh : 0.0, q_max);
    const double ell_c = cbrt(fmax(vol[i], 1.0e-30));
    const double eps_grad = fmax(1.0e-10 * te_face / ell_c, 1.0e-30);
    D_eff = deff_formula(q_limited, rho_i, cv_e, grad, D_sh, eps_grad);
    if (!isfinite(D_eff)) {
      D_eff = 0.0;
    }
    D_eff = fmax(D_eff, 0.0);
  }

  if (kappa_eff != nullptr) {
    kappa_eff[i] = kappa_local;
  }

  if (min_ratio != nullptr && D_eff > 0.0) {
    const double dl = x_r[i + 1] - x_r[i];
    if (dl > 0.0) {
      const double ratio = (dl * dl) / D_eff;
      if (ratio > 0.0 && isfinite(ratio)) {
        atomic_min_double(min_ratio, ratio);
      }
    }
  }
  if (min_deff != nullptr && D_eff > 0.0) {
    atomic_min_double(min_deff, D_eff);
  }
  if (max_deff != nullptr && D_eff > 0.0) {
    atomic_max_double(max_deff, D_eff);
  }
}

__device__ inline void compute_powerlaw_test_kappa_deff_1d_kernel_body(
    const int i,
    double* __restrict__ kappa_eff,
    double* __restrict__ rho_cv_e,
    double* __restrict__ min_ratio,
    double* __restrict__ min_deff,
    double* __restrict__ max_deff,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ x_r,
    const int n_cells,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const std::uint8_t* __restrict__ cell_is_void,
    const double test_kappa,
    const double kappa_power,
    const double kappa_rho_power,
    const double* __restrict__ state_cv_e) {
  const double z = fmax(zbar[i], 0.0);
  const double rho_i = fmax(rho[i], 0.0);
  const double te = sanitize_te_for_pow(Te[i]);
  const double gamma_i = fmax(gamma_eff[i], kMinEffectiveGamma);
  const double A_i = fmax(A_eff[i], kMinEffectiveA);
  const double cv_e_ideal = (gamma_i > 1.0 && A_i > 0.0)
                                ? (z * kEvToErg / (A_i * kProtonMass * (gamma_i - 1.0)))
                                : 0.0;
  const double cv_e =
      (state_cv_e != nullptr && state_cv_e[i] > 0.0) ? state_cv_e[i] : cv_e_ideal;
  const double rho_cv = rho_i * cv_e;

  if (rho_cv_e != nullptr) {
    rho_cv_e[i] = rho_cv;
  }

  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    if (kappa_eff != nullptr) {
      kappa_eff[i] = 0.0;
    }
    return;
  }

  double kappa_local = 0.0;
  double D_eff = 0.0;
  if (test_kappa > 0.0) {
    kappa_local = test_kappa * pow(te, kappa_power) * pow(rho_i, kappa_rho_power);
    if (rho_cv > 0.0) {
      D_eff = kappa_local / rho_cv;
    }
  }

  if (kappa_eff != nullptr) {
    kappa_eff[i] = kappa_local;
  }

  if (min_ratio != nullptr && D_eff > 0.0) {
    const double dl = x_r[i + 1] - x_r[i];
    if (dl > 0.0) {
      const double ratio = (dl * dl) / D_eff;
      if (ratio > 0.0 && isfinite(ratio)) {
        atomic_min_double(min_ratio, ratio);
      }
    }
  }
  if (min_deff != nullptr && D_eff > 0.0) {
    atomic_min_double(min_deff, D_eff);
  }
  if (max_deff != nullptr && D_eff > 0.0) {
    atomic_max_double(max_deff, D_eff);
  }
}

__device__ inline void compute_1d_flux_limiter_faces_kernel_body(
    const int face,
    double* __restrict__ flux_limiter_faces,
    const double* __restrict__ Te,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ A_eff,
    const double* __restrict__ x_r,
    const int n_cells,
    const double f_lim,
    const bool kirchhoff_face) {
  const int iL = face;
  const int iR = face + 1;

  // BUG-19: the limiter estimate must use the SAME face-kappa closure as the
  // consuming flux kernel — a harmonic estimate under a Kirchhoff flux leaves
  // sharp fronts effectively unlimited (q_face >> q_max; XC-0b 2026-07-13).
  const double k_face =
      kirchhoff_face
          ? kirchhoff_face_kappa_spitzer(kappa_sh[iL], kappa_sh[iR], Te[iL], Te[iR])
          : harmonic_mean(kappa_sh[iL], kappa_sh[iR]);
  const double rcl = cell_center_radius(x_r, iL);
  const double rcr = cell_center_radius(x_r, iR);
  const double dr = rcr - rcl;

  double q_sh = 0.0;
  if (k_face > 0.0 && dr > 0.0) {
    const double grad = (Te[iR] - Te[iL]) / dr;
    q_sh = k_face * grad;
  }

  double limiter = 1.0;
  if (k_face > 0.0) {
    const double te_face = 0.5 * (sanitize_te_for_pow(Te[iL]) + sanitize_te_for_pow(Te[iR]));
    const double rho_face = 0.5 * (fmax(rho[iL], 0.0) + fmax(rho[iR], 0.0));
    const double z_face = 0.5 * (fmax(zbar[iL], 0.0) + fmax(zbar[iR], 0.0));
    const double A_L = fmax(A_eff[iL], kMinEffectiveA);
    const double A_R = fmax(A_eff[iR], kMinEffectiveA);
    const double A_face =
        (A_L == A_R) ? A_L : fmax(harmonic_mean(A_L, A_R), kMinEffectiveA);
    const double n_e_face = electron_density_formula(rho_face, z_face, A_face);
    const double q_max = q_max_formula(f_lim, n_e_face, te_face);
    if (q_max > 0.0) {
      limiter = 1.0 / (1.0 + fabs(q_sh) / q_max);
    } else {
      limiter = 0.0;
    }
  }

  if (!isfinite(limiter)) {
    limiter = 0.0;
  }
  flux_limiter_faces[face] = fmin(1.0, fmax(limiter, 0.0));
}

template <int GEOM>
__device__ inline void conduction_1d_sts_stage_kernel_body(
    const int i,
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const double* __restrict__ alpha_cells,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode) {
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
    const double k_face = harmonic_mean(kappa_sh[i - 1], kappa_sh[i]);
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
      const double q_sh = k_face * grad;
      const double limiter =
          (flux_limiter_faces != nullptr) ? flux_limiter_faces[i - 1] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      left_term = area * q_face;
    }
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    const double k_face = harmonic_mean(kappa_sh[i], kappa_sh[i + 1]);
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
      const double q_sh = k_face * grad;
      const double limiter = (flux_limiter_faces != nullptr) ? flux_limiter_faces[i] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      right_term = area * q_face;
    }
  }

  const double scale_left = (floor_limiter_mode == 1)
      ? ((i > 0) ? ((left_term > 0.0) ? alpha_cells[i] : alpha_cells[i - 1])
                 : alpha_cells[i])
      : ((i > 0) ? fmin(alpha_cells[i - 1], alpha_cells[i]) : alpha_cells[i]);
  const double scale_right = (floor_limiter_mode == 1)
      ? ((i + 1 < n_cells) ? ((right_term > 0.0) ? alpha_cells[i + 1]
                                                       : alpha_cells[i])
                           : alpha_cells[i])
      : ((i + 1 < n_cells) ? fmin(alpha_cells[i], alpha_cells[i + 1])
                           : alpha_cells[i]);
  left_term *= scale_left;
  right_term *= scale_right;
  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  double Te_computed = Te_old_i;
  if (rho_cv > 0.0) {
    Te_computed += tau * M / rho_cv;
  }

  if (!isfinite(Te_computed) || Te_computed < Te_floor) {
    if (rho_cv > 0.0) {
      if (isfinite(Te_computed) && Te_floor > Te_computed) {
        const double de = rho_cv * (Te_floor - Te_computed) * V;
        atomic_add_double(E_floor, de);
      } else if (!isfinite(Te_computed) && Te_floor > 0.0) {
        const double de = rho_cv * Te_floor * V;
        atomic_add_double(E_floor, de);
      }
    }
    Te_new[i] = Te_floor;
    atomicAdd(clamp_count, 1);
    return;
  }

  Te_new[i] = Te_computed;
}

template <int GEOM>
__device__ inline void conduction_1d_sts_stage_kirchhoff_kernel_body(
    const int i,
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const double* __restrict__ alpha_cells,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode) {
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
    const double k_face =
        kirchhoff_face_kappa_spitzer(kappa_sh[i - 1], kappa_sh[i], Te_old[i - 1], Te_old[i]);
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
      const double q_sh = k_face * grad;
      const double limiter =
          (flux_limiter_faces != nullptr) ? flux_limiter_faces[i - 1] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      left_term = area * q_face;
    }
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    const double k_face =
        kirchhoff_face_kappa_spitzer(kappa_sh[i], kappa_sh[i + 1], Te_old[i], Te_old[i + 1]);
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
      const double q_sh = k_face * grad;
      const double limiter = (flux_limiter_faces != nullptr) ? flux_limiter_faces[i] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      right_term = area * q_face;
    }
  }

  const double scale_left = (floor_limiter_mode == 1)
      ? ((i > 0) ? ((left_term > 0.0) ? alpha_cells[i] : alpha_cells[i - 1])
                 : alpha_cells[i])
      : ((i > 0) ? fmin(alpha_cells[i - 1], alpha_cells[i]) : alpha_cells[i]);
  const double scale_right = (floor_limiter_mode == 1)
      ? ((i + 1 < n_cells) ? ((right_term > 0.0) ? alpha_cells[i + 1]
                                                       : alpha_cells[i])
                           : alpha_cells[i])
      : ((i + 1 < n_cells) ? fmin(alpha_cells[i], alpha_cells[i + 1])
                           : alpha_cells[i]);
  left_term *= scale_left;
  right_term *= scale_right;
  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  double Te_computed = Te_old_i;
  if (rho_cv > 0.0) {
    Te_computed += tau * M / rho_cv;
  }

  if (!isfinite(Te_computed) || Te_computed < Te_floor) {
    if (rho_cv > 0.0) {
      if (isfinite(Te_computed) && Te_floor > Te_computed) {
        const double de = rho_cv * (Te_floor - Te_computed) * V;
        atomic_add_double(E_floor, de);
      } else if (!isfinite(Te_computed) && Te_floor > 0.0) {
        const double de = rho_cv * Te_floor * V;
        atomic_add_double(E_floor, de);
      }
    }
    Te_new[i] = Te_floor;
    atomicAdd(clamp_count, 1);
    return;
  }

  Te_new[i] = Te_computed;
}

template <int GEOM>
__device__ inline void conduction_1d_sts_stage_secant_kernel_body(
    const int i,
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const double* __restrict__ rho,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const double kappa0,
    const double kappa_power,
    const double kappa_rho_power,
    const double* __restrict__ alpha_cells,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor,
    const int floor_limiter_mode) {
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

  const double np1 = kappa_power + 1.0;

  double left_term = 0.0;
  if (i > 0) {
    const double te_l = sanitize_te_for_pow(Te_old[i - 1]);
    const double te_r = sanitize_te_for_pow(Te_old[i]);
    double k_face;
    if (kappa_power == 0.0) {
      k_face = kappa0;
    } else if (te_l == te_r) {
      k_face = kappa0 * pow(te_l, kappa_power);
    } else {
      k_face = kappa0 * (pow(te_r, np1) - pow(te_l, np1)) / (np1 * (te_r - te_l));
    }
    k_face *= pow(0.5 * (fmax(rho[i - 1], 0.0) + fmax(rho[i], 0.0)),
                  kappa_rho_power);
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
      const double q_face = k_face * grad;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      left_term = area * q_face;
    }
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    const double te_l = sanitize_te_for_pow(Te_old[i]);
    const double te_r = sanitize_te_for_pow(Te_old[i + 1]);
    double k_face;
    if (kappa_power == 0.0) {
      k_face = kappa0;
    } else if (te_l == te_r) {
      k_face = kappa0 * pow(te_l, kappa_power);
    } else {
      k_face = kappa0 * (pow(te_r, np1) - pow(te_l, np1)) / (np1 * (te_r - te_l));
    }
    k_face *= pow(0.5 * (fmax(rho[i], 0.0) + fmax(rho[i + 1], 0.0)),
                  kappa_rho_power);
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
      const double q_face = k_face * grad;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      right_term = area * q_face;
    }
  }

  const double scale_left = (floor_limiter_mode == 1)
      ? ((i > 0) ? ((left_term > 0.0) ? alpha_cells[i] : alpha_cells[i - 1])
                 : alpha_cells[i])
      : ((i > 0) ? fmin(alpha_cells[i - 1], alpha_cells[i]) : alpha_cells[i]);
  const double scale_right = (floor_limiter_mode == 1)
      ? ((i + 1 < n_cells) ? ((right_term > 0.0) ? alpha_cells[i + 1]
                                                       : alpha_cells[i])
                           : alpha_cells[i])
      : ((i + 1 < n_cells) ? fmin(alpha_cells[i], alpha_cells[i + 1])
                           : alpha_cells[i]);
  left_term *= scale_left;
  right_term *= scale_right;
  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  double Te_computed = Te_old_i;
  if (rho_cv > 0.0) {
    Te_computed += tau * M / rho_cv;
  }

  if (!isfinite(Te_computed) || Te_computed < Te_floor) {
    if (rho_cv > 0.0) {
      if (isfinite(Te_computed) && Te_floor > Te_computed) {
        const double de = rho_cv * (Te_floor - Te_computed) * V;
        atomic_add_double(E_floor, de);
      } else if (!isfinite(Te_computed) && Te_floor > 0.0) {
        const double de = rho_cv * Te_floor * V;
        atomic_add_double(E_floor, de);
      }
    }
    Te_new[i] = Te_floor;
    atomicAdd(clamp_count, 1);
    return;
  }

  Te_new[i] = Te_computed;
}

template <int GEOM>
__device__ inline void kirchhoff_dt_ratio_1d_kernel_body(
    const int i,
    double* __restrict__ min_ratio,
    const double* __restrict__ kappa_eff,
    const double* __restrict__ rho_cv_e,
    const double* __restrict__ Te,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells) {
  if (cell_is_void != nullptr && cell_is_void[i] != static_cast<std::uint8_t>(0)) {
    return;
  }
  const double rho_cv = rho_cv_e[i];
  if (!(rho_cv > 0.0)) {
    return;
  }

  // BUG-20 / #833: an unset/zero/nonfinite cell extent must not contribute a
  // Gershgorin ratio — the historic 1e-30 floor manufactured a ~1e-27 s dt_exp
  // (STS ~1e9 substeps = effective hang). Healthy states carry positive finite
  // extents, so dt_exp stays bit-identical; degenerate cells fall back to the
  // deff-based estimate from compute_spitzer_deff_1d_kernel_body.
  double V_raw;
  if constexpr (GEOM == 2) {
    V_raw = x_r[i + 1] - x_r[i];
  } else {
    V_raw = vol[i];
  }
  if (!(V_raw > 0.0) || !isfinite(V_raw)) {
    return;
  }
  const double V = fmax(V_raw, 1.0e-30);

  double g_sum = 0.0;
  if (i > 0) {
    const double k_face =
        kirchhoff_face_kappa_spitzer(kappa_eff[i - 1], kappa_eff[i], Te[i - 1], Te[i]);
    const double dr = cell_center_radius(x_r, i) - cell_center_radius(x_r, i - 1);
    double area;
    if constexpr (GEOM == 0) {
      area = kFourPi * x_r[i] * x_r[i];
    } else {
      area = geometry_1d_face_area(GEOM, x_r[i]);
    }
    if (k_face > 0.0 && dr > 0.0) {
      g_sum += area * k_face / dr;
    }
  }
  if (i + 1 < n_cells) {
    const double k_face =
        kirchhoff_face_kappa_spitzer(kappa_eff[i], kappa_eff[i + 1], Te[i], Te[i + 1]);
    const double dr = cell_center_radius(x_r, i + 1) - cell_center_radius(x_r, i);
    double area;
    if constexpr (GEOM == 0) {
      area = kFourPi * x_r[i + 1] * x_r[i + 1];
    } else {
      area = geometry_1d_face_area(GEOM, x_r[i + 1]);
    }
    if (k_face > 0.0 && dr > 0.0) {
      g_sum += area * k_face / dr;
    }
  }

  if (min_ratio != nullptr && g_sum > 0.0 && isfinite(g_sum)) {
    const double ratio = 2.0 * rho_cv * V / g_sum;
    if (ratio > 0.0 && isfinite(ratio)) {
      atomic_min_double(min_ratio, ratio);
    }
  }
}
// PK-C2a-2 megakernel path: the persistent loop was certified (rmtv ON/OFF
// bit-identical) against the pre-BUG-15 single-pass in-body alpha form. The
// standard conduction path moved to the two-pass pair-min alpha (BUG-15) with
// a precomputed alpha_cells field the megakernel does not stage yet; these
// legacy clones pin the certified megakernel numerics until the BUG-15 port
// lands there with its own gate (persistent lane is #56-gated, non-production).
template <int GEOM>
__device__ inline void conduction_1d_sts_stage_legacy_inline_alpha_kernel_body(
    const int i,
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double tau,
    const double Te_floor,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor) {
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
    const double k_face = harmonic_mean(kappa_sh[i - 1], kappa_sh[i]);
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
      const double q_sh = k_face * grad;
      const double limiter =
          (flux_limiter_faces != nullptr) ? flux_limiter_faces[i - 1] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      left_term = area * q_face;
    }
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    const double k_face = harmonic_mean(kappa_sh[i], kappa_sh[i + 1]);
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
      const double q_sh = k_face * grad;
      const double limiter = (flux_limiter_faces != nullptr) ? flux_limiter_faces[i] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      right_term = area * q_face;
    }
  }

  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  double Te_computed = Te_old_i;
  if (rho_cv > 0.0) {
    double alpha = 1.0;
    if (tau > 0.0 && Te_old_i > Te_floor && M < 0.0) {
      const double e_above_floor = rho_cv * (Te_old_i - Te_floor);
      const double dt_safe = e_above_floor / fmax(fabs(M), kDivEpsilon);
      alpha = fmin(1.0, fmax(dt_safe / tau, 0.0));
    }
    Te_computed += tau * alpha * M / rho_cv;
  }

  if (!isfinite(Te_computed) || Te_computed < Te_floor) {
    if (rho_cv > 0.0) {
      if (isfinite(Te_computed) && Te_floor > Te_computed) {
        const double de = rho_cv * (Te_floor - Te_computed) * V;
        atomic_add_double(E_floor, de);
      } else if (!isfinite(Te_computed) && Te_floor > 0.0) {
        const double de = rho_cv * Te_floor * V;
        atomic_add_double(E_floor, de);
      }
    }
    Te_new[i] = Te_floor;
    atomicAdd(clamp_count, 1);
    return;
  }

  Te_new[i] = Te_computed;
}

template <int GEOM>
__device__ inline void conduction_1d_sts_stage_kirchhoff_legacy_inline_alpha_kernel_body(
    const int i,
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ kappa_sh,
    const double* __restrict__ flux_limiter_faces,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const int n_cells,
    const double tau,
    const double Te_floor,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor) {
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
    const double k_face =
        kirchhoff_face_kappa_spitzer(kappa_sh[i - 1], kappa_sh[i], Te_old[i - 1], Te_old[i]);
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
      const double q_sh = k_face * grad;
      const double limiter =
          (flux_limiter_faces != nullptr) ? flux_limiter_faces[i - 1] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      left_term = area * q_face;
    }
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    const double k_face =
        kirchhoff_face_kappa_spitzer(kappa_sh[i], kappa_sh[i + 1], Te_old[i], Te_old[i + 1]);
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
      const double q_sh = k_face * grad;
      const double limiter = (flux_limiter_faces != nullptr) ? flux_limiter_faces[i] : 1.0;
      const double q_face = q_sh * limiter;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      right_term = area * q_face;
    }
  }

  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  double Te_computed = Te_old_i;
  if (rho_cv > 0.0) {
    double alpha = 1.0;
    if (tau > 0.0 && Te_old_i > Te_floor && M < 0.0) {
      const double e_above_floor = rho_cv * (Te_old_i - Te_floor);
      const double dt_safe = e_above_floor / fmax(fabs(M), kDivEpsilon);
      alpha = fmin(1.0, fmax(dt_safe / tau, 0.0));
    }
    Te_computed += tau * alpha * M / rho_cv;
  }

  if (!isfinite(Te_computed) || Te_computed < Te_floor) {
    if (rho_cv > 0.0) {
      if (isfinite(Te_computed) && Te_floor > Te_computed) {
        const double de = rho_cv * (Te_floor - Te_computed) * V;
        atomic_add_double(E_floor, de);
      } else if (!isfinite(Te_computed) && Te_floor > 0.0) {
        const double de = rho_cv * Te_floor * V;
        atomic_add_double(E_floor, de);
      }
    }
    Te_new[i] = Te_floor;
    atomicAdd(clamp_count, 1);
    return;
  }

  Te_new[i] = Te_computed;
}

template <int GEOM>
__device__ inline void conduction_1d_sts_stage_secant_legacy_inline_alpha_kernel_body(
    const int i,
    const double* __restrict__ Te_old,
    double* __restrict__ Te_new,
    const double* __restrict__ rho_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ vol,
    const double* __restrict__ x_r,
    const double* __restrict__ rho,
    const int n_cells,
    const double tau,
    const double Te_floor,
    const double kappa0,
    const double kappa_power,
    const double kappa_rho_power,
    int* __restrict__ clamp_count,
    double* __restrict__ E_floor) {
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

  const double np1 = kappa_power + 1.0;

  double left_term = 0.0;
  if (i > 0) {
    const double te_l = sanitize_te_for_pow(Te_old[i - 1]);
    const double te_r = sanitize_te_for_pow(Te_old[i]);
    double k_face;
    if (kappa_power == 0.0) {
      k_face = kappa0;
    } else if (te_l == te_r) {
      k_face = kappa0 * pow(te_l, kappa_power);
    } else {
      k_face = kappa0 * (pow(te_r, np1) - pow(te_l, np1)) / (np1 * (te_r - te_l));
    }
    k_face *= pow(0.5 * (fmax(rho[i - 1], 0.0) + fmax(rho[i], 0.0)),
                  kappa_rho_power);
    const double rcl = cell_center_radius(x_r, i - 1);
    const double rcr = cell_center_radius(x_r, i);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i] - Te_old[i - 1]) / dr;
      const double q_face = k_face * grad;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i] * x_r[i];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i]);
      }
      left_term = area * q_face;
    }
  }

  double right_term = 0.0;
  if (i + 1 < n_cells) {
    const double te_l = sanitize_te_for_pow(Te_old[i]);
    const double te_r = sanitize_te_for_pow(Te_old[i + 1]);
    double k_face;
    if (kappa_power == 0.0) {
      k_face = kappa0;
    } else if (te_l == te_r) {
      k_face = kappa0 * pow(te_l, kappa_power);
    } else {
      k_face = kappa0 * (pow(te_r, np1) - pow(te_l, np1)) / (np1 * (te_r - te_l));
    }
    k_face *= pow(0.5 * (fmax(rho[i], 0.0) + fmax(rho[i + 1], 0.0)),
                  kappa_rho_power);
    const double rcl = cell_center_radius(x_r, i);
    const double rcr = cell_center_radius(x_r, i + 1);
    const double dr = rcr - rcl;
    if (k_face > 0.0 && dr > 0.0) {
      const double grad = (Te_old[i + 1] - Te_old[i]) / dr;
      const double q_face = k_face * grad;
      double area;
      if constexpr (GEOM == 0) {
        area = kFourPi * x_r[i + 1] * x_r[i + 1];
      } else {
        area = geometry_1d_face_area(GEOM, x_r[i + 1]);
      }
      right_term = area * q_face;
    }
  }

  const double M = (right_term - left_term) / V;
  const double rho_cv = rho_cv_e[i];
  const double Te_old_i = Te_old[i];
  double Te_computed = Te_old_i;
  if (rho_cv > 0.0) {
    double alpha = 1.0;
    if (tau > 0.0 && Te_old_i > Te_floor && M < 0.0) {
      const double e_above_floor = rho_cv * (Te_old_i - Te_floor);
      const double dt_safe = e_above_floor / fmax(fabs(M), kDivEpsilon);
      alpha = fmin(1.0, fmax(dt_safe / tau, 0.0));
    }
    Te_computed += tau * alpha * M / rho_cv;
  }

  if (!isfinite(Te_computed) || Te_computed < Te_floor) {
    if (rho_cv > 0.0) {
      if (isfinite(Te_computed) && Te_floor > Te_computed) {
        const double de = rho_cv * (Te_floor - Te_computed) * V;
        atomic_add_double(E_floor, de);
      } else if (!isfinite(Te_computed) && Te_floor > 0.0) {
        const double de = rho_cv * Te_floor * V;
        atomic_add_double(E_floor, de);
      }
    }
    Te_new[i] = Te_floor;
    atomicAdd(clamp_count, 1);
    return;
  }

  Te_new[i] = Te_computed;
}


__device__ inline void eos_sync_electron_kernel_body(
    const int i,
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const int n_cells,
    const double Te_floor,
    const double* __restrict__ state_cv_e) {
  (void)n_cells;
  const double te = fmax(Te[i], Te_floor);
  const double z = fmax(zbar[i], 0.0);
  const double gamma = fmax(gamma_eff[i], kMinEffectiveGamma);
  const double A = fmax(A_eff[i], kMinEffectiveA);
  const double cv_e_ideal = (gamma > 1.0 && A > 0.0)
                                ? (z * kEvToErg / (A * kProtonMass * (gamma - 1.0)))
                                : 0.0;
  const double cv_e =
      (state_cv_e != nullptr && state_cv_e[i] > 0.0) ? state_cv_e[i] : cv_e_ideal;
  const double e_e = cv_e * te;
  Te[i] = te;
  ee[i] = fmax(e_e, 0.0);
  Pe[i] = (gamma - 1.0) * rho[i] * ee[i];
}

__host__ __device__ inline int ceil_to_int_clamped(const double value) {
  if (!isfinite(value)) {
    return 2147483647;
  }
  if (value <= 1.0) {
    return 1;
  }
  if (value >= 2147483647.0) {
    return 2147483647;
  }
  return static_cast<int>(ceil(value));
}

__host__ __device__ inline int sts_stage_count(const double dt,
                                               const double dt_exp,
                                               const int sts_max_stages) {
  if (!(dt > 0.0) || !(dt_exp > 0.0) || !isfinite(dt_exp)) {
    return 1;
  }
  const double s_raw = ceil(sqrt(2.0 * dt / dt_exp));
  if (sts_max_stages <= 0) {
    if (!isfinite(s_raw) || s_raw > 2147483647.0) {
      return 100000;
    }
    const int s = (static_cast<int>(s_raw) > 1) ? static_cast<int>(s_raw) : 1;
    return (s < 100000) ? s : 100000;
  }
  const int smax = (sts_max_stages < 1) ? 1 : sts_max_stages;
  if (!isfinite(s_raw) || s_raw > 2147483647.0) {
    return smax;
  }
  const int s = (static_cast<int>(s_raw) > 1) ? static_cast<int>(s_raw) : 1;
  return (s < smax) ? s : smax;
}

__host__ __device__ inline double sts_tau_for_stage(const double dt,
                                                    const double dt_exp,
                                                    const int stages,
                                                    const double damping,
                                                    const int stage_index) {
  const double nu = fmin(0.999, fmax(damping, 1.0e-6));
  double tau_sum = 0.0;
  double tau_stage = 0.0;
  for (int j = 1; j <= stages; ++j) {
    const double arg =
        kPi * (2.0 * static_cast<double>(j) - 1.0) /
        (4.0 * static_cast<double>(stages) + 2.0);
    const double c = cos(arg);
    const double denom = nu * nu + (1.0 - nu * nu) * c * c;
    const double tj = dt_exp / denom;
    tau_sum += tj;
    if (j - 1 == stage_index) {
      tau_stage = tj;
    }
  }
  return (tau_sum > 0.0) ? (tau_stage * dt / tau_sum) : 0.0;
}

// Amplification bound of the uniformly rescaled STS ladder (AI kernel review
// 2026-07-26, k07 F-08): max_{lambda in (0, lambda_max]} |prod_j (1 - lambda
// tau_j)|.  The rescale in initialize_sts_taus / sts_tau_for_stage is a
// stability-preserving SHRINK at the shipped defaults (cfl_cond = 0.25,
// conduction.sts_damping = 0.01; measured bound 0.175 up to the subcycle cap),
// but legal namelist settings inflate the ladder past its Chebyshev stability
// interval (e.g. sts_damping = 0.05 with s = 40 at the cap: bound ~ 3e12).
// Callers must reject configurations with bound > 1 before running stages.
// lambda_max is the operator spectral-radius certificate implied by dt_exp:
// dt_exp = cfl_cond * min_ratio with the Gershgorin bound lambda <= 4 /
// min_ratio, hence lambda_max = 4 * cfl_cond / dt_exp (exact for the Kirchhoff
// dt path; the cell-local D_eff estimate can undercover strongly graded meshes
// by up to 2/(1+r), NUMERICS 4.2.1).  Probes every inter-root midpoint (the
// equioscillation bumps) plus a uniform grid; pure host, deterministic.
inline double sts_ladder_amplification_bound(const double* tau,
                                             const int stages,
                                             const double lambda_max) {
  if (tau == nullptr || stages <= 0 || !(lambda_max > 0.0)) {
    return 0.0;
  }
  double bound = 0.0;
  const auto probe = [&](const double lam) {
    if (!(lam > 0.0) || lam > lambda_max) {
      return;
    }
    double g = 1.0;
    for (int j = 0; j < stages; ++j) {
      g *= (1.0 - lam * tau[j]);
    }
    g = fabs(g);
    if (g > bound) {
      bound = g;
    }
  };
  for (int j = 0; j + 1 < stages; ++j) {
    if (tau[j] > 0.0 && tau[j + 1] > 0.0) {
      probe(0.5 * (1.0 / tau[j] + 1.0 / tau[j + 1]));
    }
  }
  const int n_grid = 8 * stages + 8;
  for (int k = 1; k <= n_grid; ++k) {
    probe(lambda_max * static_cast<double>(k) / static_cast<double>(n_grid));
  }
  return bound;
}

}  // namespace tenryu::hydro::conduction_bodies
