#pragma once

#include <cstdint>

#include "hydro/braginskii_viscosity.cuh"

namespace tenryu::hydro::braginskii {

// cgs constants; kEvToErg / kProtonMass match src/hydro/conduction.cu.
constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = 1.6726219e-24;
constexpr double kElectronMass = 9.1093837e-28;
// kTauI0 = 3*sqrt(m_p)*kEvToErg^{3/2} / (4*sqrt(pi)*e^4), e = 4.80320e-10
// esu, so that (NUMERICS plasma-viscosity section; NRL formulary quotes 2.09e7 and
// Manheimer & Colombant 2007 Eq. (2) is the same with n_e = Z n_i):
//   tau_i = kTauI0 * sqrt(A) * Ti_eV^{3/2} / (n_i * Z^4 * lnLambda_ii)  [s]
constexpr double kTauI0 = 2.0852e7;
constexpr double kBraginskiiEta0Coeff = 0.96;
// kTauE0 = 3*sqrt(m_e)*kEvToErg^{3/2} / (4*sqrt(2*pi)*e^4), the Braginskii
// (2.5e) electron collision time written NRL-style (NRL quotes
// nu_e = 2.91e-6, i.e. 1/kTauE0, and conduction.cu uses the same rounded
// coefficient):
//   tau_e = kTauE0 * Te_eV^{3/2} / (n_e * Z * lnLambda_ei)  [s]
constexpr double kTauE0 = 3.4409e5;

struct DeviceParams {
  int model;
  double eta_const;
  double eta0_scale;
  double mfp_cap_cells;
  double lnlambda_fixed;
  double ti_floor_ev;
  double te_floor_ev;
};

inline DeviceParams to_device_params(const Params& p) {
  DeviceParams d;
  d.model = p.model;
  d.eta_const = p.eta_const;
  d.eta0_scale = p.eta0_scale;
  d.mfp_cap_cells = p.mfp_cap_cells;
  d.lnlambda_fixed = p.lnlambda_fixed;
  d.ti_floor_ev = p.ti_floor_ev;
  d.te_floor_ev = p.te_floor_ev;
  return d;
}

__device__ __forceinline__ double atomic_min_double_brag(double* addr,
                                                         double val) {
  unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    const double cur = __longlong_as_double(assumed);
    if (val >= cur) {
      break;
    }
    old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);
  return __longlong_as_double(old);
}

__device__ __forceinline__ double atomic_max_double_brag(double* addr,
                                                         double val) {
  unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_ull;
  unsigned long long assumed;
  do {
    assumed = old;
    const double cur = __longlong_as_double(assumed);
    if (val <= cur) {
      break;
    }
    old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);
  return __longlong_as_double(old);
}

// eta0 [poise]. dr enters only through the mean-free-path cap
// (tau_eff = min(tau_i, mfp_cap_cells * dr / v_th,i), Mason 2014).
__device__ __forceinline__ double braginskii_eta_poise(const double rho,
                                                       const double ti_ev,
                                                       const double a_eff,
                                                       const double zbar,
                                                       const double dr,
                                                       const DeviceParams p) {
  if (p.model == 1) {
    return p.eta_const;
  }
  const double A = fmax(a_eff, 1.0);
  const double n_i = rho / (A * kProtonMass);
  if (!(n_i > 0.0)) {
    return 0.0;
  }
  const double T = fmax(ti_ev, p.ti_floor_ev);
  const double Z = fmax(zbar, 1.0);
  const double t32 = T * sqrt(T);
  double ln_lambda;
  if (p.lnlambda_fixed > 0.0) {
    ln_lambda = p.lnlambda_fixed;
  } else {
    // NRL single-species ion-ion Coulomb log (T in eV, n_i in cm^-3):
    // lnLambda_ii = 23 - ln[Z^3 sqrt(2 n_i) / T^{3/2}], floored at 2 like
    // the e-i convention in eos_device.cuh / conduction.cu.
    ln_lambda = 23.0 - log(Z * Z * Z * sqrt(2.0 * n_i) / t32);
    ln_lambda = fmax(2.0, ln_lambda);
  }
  const double z4 = (Z * Z) * (Z * Z);
  const double tau_i = kTauI0 * sqrt(A) * t32 / (n_i * z4 * ln_lambda);
  double tau_eff = tau_i;
  if (p.mfp_cap_cells > 0.0 && dr > 0.0) {
    const double vth = sqrt(kEvToErg * T / (A * kProtonMass));
    if (vth > 0.0) {
      const double tau_cap = p.mfp_cap_cells * dr / vth;
      tau_eff = fmin(tau_i, tau_cap);
    }
  }
  return p.eta0_scale * kBraginskiiEta0Coeff * n_i * (kEvToErg * T) * tau_eff;
}

__host__ __device__ __forceinline__ double braginskii_eta_poise_ext(
    const double rho,
    const double ti_ev,
    const double a_eff,
    const double zbar,
    const double zmom_r4,
    const double dr,
    const DeviceParams p) {
  if (p.model == 1) {
    return p.eta_const;
  }
  const double A = fmax(a_eff, 1.0);
  const double n_i = rho / (A * kProtonMass);
  if (!(n_i > 0.0)) {
    return 0.0;
  }
  const double T = fmax(ti_ev, p.ti_floor_ev);
  const double Z = fmax(zbar, 1.0);
  const double t32 = T * sqrt(T);
  double ln_lambda;
  if (p.lnlambda_fixed > 0.0) {
    ln_lambda = p.lnlambda_fixed;
  } else {
    // NRL single-species ion-ion Coulomb log (T in eV, n_i in cm^-3):
    // lnLambda_ii = 23 - ln[Z^3 sqrt(2 n_i) / T^{3/2}], floored at 2 like
    // the e-i convention in eos_device.cuh / conduction.cu.
    // lnLambda_ii intentionally keeps the legacy single-species Z^3 form.
    ln_lambda = 23.0 - log(Z * Z * Z * sqrt(2.0 * n_i) / t32);
    ln_lambda = fmax(2.0, ln_lambda);
  }
  const double z4 = (Z * Z) * (Z * Z) * zmom_r4;
  const double tau_i = kTauI0 * sqrt(A) * t32 / (n_i * z4 * ln_lambda);
  double tau_eff = tau_i;
  if (p.mfp_cap_cells > 0.0 && dr > 0.0) {
    const double vth = sqrt(kEvToErg * T / (A * kProtonMass));
    if (vth > 0.0) {
      const double tau_cap = p.mfp_cap_cells * dr / vth;
      tau_eff = fmin(tau_i, tau_cap);
    }
  }
  return p.eta0_scale * kBraginskiiEta0Coeff * n_i * (kEvToErg * T) * tau_eff;
}

// NRL electron-ion Coulomb log; twin of conduction.cu::coulomb_log_formula
// (kept a module-local twin — no cross-module header moves into pre-existing
// TUs; pinned against the conduction twin by
// tests/hydro/test_braginskii_viscosity.cu).
__device__ __forceinline__ double braginskii_coulomb_log_ei(const double n_e,
                                                            const double te_ev,
                                                            const double zbar) {
  if (!(n_e > 0.0) || !(te_ev > 0.0) || !(zbar > 0.0)) {
    return 2.0;
  }
  double ln_lambda_raw = 0.0;
  if (te_ev >= 10.0 * zbar * zbar) {
    ln_lambda_raw = 24.0 - 0.5 * log(n_e) + log(te_ev);
  } else {
    ln_lambda_raw = 23.0 - 0.5 * log(n_e) - log(zbar) + 1.5 * log(te_ev);
  }
  return fmax(2.0, ln_lambda_raw);
}

// eta_e [poise]. Whitney's Z-dependent coefficient (Velikovich, Whitney &
// Thornhill 2001, Eq. (3)): 0.733 at Z=1 (= Braginskii Eq. 2.25), -> 1.81
// as Z -> inf. Z floored at 1 like the ion channel (fit domain Z >= 1);
// n_e = Z n_i enters only through lnLambda_ei (eta_e itself is
// density-independent in the uncapped limit). dr enters only through the
// electron mean-free-path cap (tau_eff = min(tau_e, mfp_cap_cells * dr /
// v_th_e), Mason-2014 rationale shared with the ion cap).
__device__ __forceinline__ double braginskii_eta_e_poise(const double rho,
                                                         const double te_ev,
                                                         const double a_eff,
                                                         const double zbar,
                                                         const double dr,
                                                         const DeviceParams p) {
  if (p.model == 1) {
    return p.eta_const;
  }
  const double A = fmax(a_eff, 1.0);
  const double n_i = rho / (A * kProtonMass);
  if (!(n_i > 0.0)) {
    return 0.0;
  }
  const double T = fmax(te_ev, p.te_floor_ev);
  const double Z = fmax(zbar, 1.0);
  const double n_e = Z * n_i;
  const double t32 = T * sqrt(T);
  double ln_lambda;
  if (p.lnlambda_fixed > 0.0) {
    ln_lambda = p.lnlambda_fixed;
  } else {
    ln_lambda = braginskii_coulomb_log_ei(n_e, T, Z);
  }
  const double tau_e = kTauE0 * t32 / (n_e * Z * ln_lambda);
  double tau_eff = tau_e;
  if (p.mfp_cap_cells > 0.0 && dr > 0.0) {
    const double vth = sqrt(kEvToErg * T / kElectronMass);
    if (vth > 0.0) {
      const double tau_cap = p.mfp_cap_cells * dr / vth;
      tau_eff = fmin(tau_e, tau_cap);
    }
  }
  const double eta00_e = 1.81 * Z * ((Z + 2.82) * Z + 1.343) /
                         (((Z + 4.434) * Z + 5.534) * Z + 1.78);
  return p.eta0_scale * eta00_e * n_e * (kEvToErg * T) * tau_eff;
}

}  // namespace tenryu::hydro::braginskii
