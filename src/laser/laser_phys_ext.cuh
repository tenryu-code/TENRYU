#pragma once

#include <cmath>

#include "core/macros.hpp"
#include "laser/ib_absorption.cuh"

namespace tenryu::laser {

constexpr int kIBExtMaxSpecies = 4;

struct LaserPhysExtOptions {
  // master switch mirrors are handled by callers; the POD carries sub-flags
  int zeff_model = 0;          // 0=off, 1=sequential_strip, 2=table
  int n_species = 0;           // number of entries in z_nuc/x_frac
  double z_nuc[kIBExtMaxSpecies] = {0.0, 0.0, 0.0, 0.0};   // nuclear charges
  double x_frac[kIBExtMaxSpecies] = {0.0, 0.0, 0.0, 0.0};  // number fractions (sum=1)
  const double* zeff_table = nullptr;  // device (or host in host tests) [nD*nT], d*nT+t
  int zt_nd = 0;
  int zt_nt = 0;
  double zt_l10d0 = 0.0;   // log10 of first ni grid point
  double zt_dl10d = 1.0;   // log10 spacing of ni grid
  double zt_l10t0 = 0.0;   // log10 of first T grid point
  double zt_dl10t = 1.0;   // log10 spacing of T grid
  double n_crit_cm3 = 0.0;
  int coulomb_log_model = 0;   // 0=debye (current), 1=laser_frequency
  int langdon_model = 0;       // 0=off, 1=legacy_vacuum_map
  double langdon_zcoll = 1.0;  // representative collision charge for the
                               // Langdon alpha (host fills full-ionization
                               // Z_eff of the composition; corona-valid)
  double langdon_te_min_eV = 100.0;  // f_L=1 below this Te (validity domain)
  int ra_enable = 0;           // resonance absorption master flag (used by later tasks)
  double ra_chi_p = 0.5;       // polarization fraction
  double ra_c = 1.0;           // overall multiplier
  int crit_terminate_deposit = 0;
  // Per-step host-precomputed scalars for RA (filled by later tasks; defaults inert):
  double ra_r_crit_cm = -1.0;  // outermost critical radius; <0 disables RA this step
  double ra_ln_cm = -1.0;      // density scale length |dlnne/dr|^-1 at r_crit; <=0 disables
  double k0_cm_inv = 0.0;      // laser vacuum wavenumber 2*pi/lambda
};

// Bilinear in (log10 ni, log10 Te), clamped to the table edges.
TENRYU_HOST_DEVICE inline double zeff_ratio_from_table(
    const LaserPhysExtOptions& opt, const double ni_cm3, const double Te_eV) {
  if (opt.zeff_table == nullptr || opt.zt_nd < 2 || opt.zt_nt < 2) {
    return 1.0;
  }

  const double u_raw =
      (::log10(::fmax(ni_cm3, 1.0e-30)) - opt.zt_l10d0) / opt.zt_dl10d;
  const double v_raw =
      (::log10(::fmax(Te_eV, 1.0e-30)) - opt.zt_l10t0) / opt.zt_dl10t;
  const double u =
      ::fmin(static_cast<double>(opt.zt_nd - 1), ::fmax(0.0, u_raw));
  const double v =
      ::fmin(static_cast<double>(opt.zt_nt - 1), ::fmax(0.0, v_raw));
  const int d0 = static_cast<int>(::floor(u));
  const int t0 = static_cast<int>(::floor(v));
  const int d1 = (d0 + 1 < opt.zt_nd) ? d0 + 1 : d0;
  const int t1 = (t0 + 1 < opt.zt_nt) ? t0 + 1 : t0;
  const double fu = u - static_cast<double>(d0);
  const double fv = v - static_cast<double>(t0);
  const double r00 = opt.zeff_table[d0 * opt.zt_nt + t0];
  const double r01 = opt.zeff_table[d0 * opt.zt_nt + t1];
  const double r10 = opt.zeff_table[d1 * opt.zt_nt + t0];
  const double r11 = opt.zeff_table[d1 * opt.zt_nt + t1];
  const double r0 = r00 + fv * (r01 - r00);
  const double r1 = r10 + fv * (r11 - r10);
  return r0 + fu * (r1 - r0);
}

// Collision-weighted charge for a multi-species mixture, from the mean
// per-atom ionization zbar_atom and the composition, under the closure that
// species strip sequentially in ascending nuclear charge (light elements
// first; exact in the fully ionized limit).
// The caller must provide species sorted in ascending z_nuc order.
// Returns Z_eff / max(zbar_atom, eps) with Z_eff = sum(x_s z_s^2)/sum(x_s z_s).
// For n_species < 2 or zeff_model==0 returns 1.0.
TENRYU_HOST_DEVICE inline double compute_zeff_ratio_sequential(
    const LaserPhysExtOptions& opt, const double zbar_atom) {
  if (opt.n_species < 2 || opt.zeff_model == 0) {
    return 1.0;
  }

  double remaining = ::fmax(zbar_atom, 0.0);
  double num = 0.0;
  double den = 0.0;
  for (int s = 0; s < opt.n_species; ++s) {
    if (!(opt.x_frac[s] > 0.0)) {
      continue;
    }
    const double z_s = ::fmin(opt.z_nuc[s], remaining / opt.x_frac[s]);
    remaining -= opt.x_frac[s] * z_s;
    num += opt.x_frac[s] * z_s * z_s;
    den += opt.x_frac[s] * z_s;
  }

  if (!(den > 1.0e-12)) {
    return 1.0;
  }
  const double ratio = num / den / ::fmax(zbar_atom, 1.0e-12);
  return ::fmin(10.0, ::fmax(1.0, ratio));
}

// coulomb_log_model==1: laser-frequency cutoff (b_max ~ v_te/omega_0) —
// the Debye-type 1/sqrt(n_hat) of the legacy form is dropped:
//   arg = 1.5e4 * lambda_um * Te_keV^1.5 / Zbar
// Same floor semantics as compute_coulomb_log.
TENRYU_HOST_DEVICE inline double compute_coulomb_log_ext(
    const int coulomb_log_model,
    const double n_hat,
    const double Te_eV,
    const double Zbar,
    const double lambda_cm,
    const double coulomb_log_floor) {
  if (coulomb_log_model != 1) {
    return compute_coulomb_log(n_hat, Te_eV, Zbar, lambda_cm, coulomb_log_floor);
  }

  const double effective_floor = ::fmax(2.0, coulomb_log_floor);
  if (!(Te_eV > 0.0) || !(Zbar > 0.0) || !(lambda_cm > 0.0)) {
    return effective_floor;
  }

  const double lambda_um = lambda_cm * 1.0e4;
  const double Te_keV = Te_eV * 1.0e-3;
  const double Te_32 = Te_keV * ::sqrt(::fmax(Te_keV, 0.0));

  if (!(lambda_um > 0.0) || !(Te_32 > 0.0)) {
    return effective_floor;
  }

  const double arg = 1.5e4 * lambda_um * Te_32 / Zbar;
  const double ln_lambda = (arg > 0.0) ? ::log(arg) : effective_floor;
  return ::fmax(effective_floor, ln_lambda);
}

// alpha = 0.03736 * Z_coll * I14 * lambda_um^2 / Te_keV  (Langdon 1980
// convention v_T^2 = kTe/m). f_L = 1 - 0.553 / (1 + (0.27/alpha)^0.75).
// I_wcm2 <= 0 or langdon_model==0 -> returns 1.0. alpha clamped to
// [1e-6, 1e3] before evaluating f_L; result clamped to [0.447, 1.0].
TENRYU_HOST_DEVICE inline double compute_langdon_factor(
    const int langdon_model,
    const double Z_coll,
    const double I_wcm2,
    const double lambda_cm,
    const double Te_eV,
    const double te_min_eV) {
  if (langdon_model != 1 || !(I_wcm2 > 0.0)) {
    return 1.0;
  }
  if (!(Te_eV > te_min_eV)) {
    return 1.0;
  }

  const double I14 = I_wcm2 * 1.0e-14;
  const double lambda_um = lambda_cm * 1.0e4;
  const double Te_keV = Te_eV * 1.0e-3;
  const double alpha_raw =
      0.03736 * Z_coll * I14 * lambda_um * lambda_um / Te_keV;
  const double alpha = ::fmin(1.0e3, ::fmax(1.0e-6, alpha_raw));
  const double f_langdon =
      1.0 - 0.553 / (1.0 + ::pow(0.27 / alpha, 0.75));
  return ::fmin(1.0, ::fmax(0.447, f_langdon));
}

// Vacuum-map local intensity of a collimated Gaussian or flat-top beam of
// total power P_w [W] and radius w_cm at cylindrical radius rcyl_cm.
// Gaussian (flat_top == 0), with w_cm the 1/e-INTENSITY radius:
//   I = P_w / (pi w^2) * exp(-(rcyl/w)^2)   [W/cm^2]
// Flat-top (flat_top != 0), uniform inside the disc of radius w_cm:
//   I = P_w / (pi w^2) inside, 0 outside   [W/cm^2]
// w_cm <= 0 or P_w <= 0 -> returns 0.
TENRYU_HOST_DEVICE inline double vacuum_map_intensity(
    const double P_w, const double w_cm, const double rcyl_cm,
    const int flat_top) {
  if (!(w_cm > 0.0) || !(P_w > 0.0)) {
    return 0.0;
  }
  constexpr double pi = 3.14159265358979323846;
  if (flat_top != 0) {
    // Uniform disc of radius w_cm (HELIOS-parity spatial model):
    // I = P / (pi w^2) inside, 0 outside.
    return (rcyl_cm < w_cm) ? P_w / (pi * w_cm * w_cm) : 0.0;
  }
  const double radius_ratio = rcyl_cm / w_cm;
  return P_w / (pi * w_cm * w_cm) *
         ::exp(-radius_ratio * radius_ratio);
}

// eta = (k0 Ln)^(2/3) * (b/rc)^2 ; f_p = 1.74098 * eta / sqrt(eta+0.435)
//        * exp(-4/3 * eta^1.5). Returns chi_p * c_ra * f_p, clamped to
// [0, 0.95]. Returns 0 unless ra_enable, 0 < b < rc, Ln > 0, k0 > 0,
// or if eta >= 6 (curve < 1e-8 there).
TENRYU_HOST_DEVICE inline double compute_ra_event_fraction(
    const LaserPhysExtOptions& opt, const double b_cm) {
  if (opt.ra_enable == 0 || !(b_cm > 0.0) ||
      !(b_cm < opt.ra_r_crit_cm) || !(opt.ra_ln_cm > 0.0) ||
      !(opt.k0_cm_inv > 0.0)) {
    return 0.0;
  }

  const double radius_ratio = b_cm / opt.ra_r_crit_cm;
  const double eta =
      ::pow(opt.k0_cm_inv * opt.ra_ln_cm, 2.0 / 3.0) *
      radius_ratio * radius_ratio;
  if (eta >= 6.0) {
    return 0.0;
  }

  const double f_p =
      1.74098 * eta / ::sqrt(eta + 0.435) *
      ::exp(-(4.0 / 3.0) * eta * ::sqrt(eta));
  const double fraction = opt.ra_chi_p * opt.ra_c * f_p;
  return ::fmin(0.95, ::fmax(0.0, fraction));
}

// Extended counterpart of compute_kappa_smooth_factor: same structure, but
// Z prefactor multiplied by the selected Z_eff ratio, ln Lambda from
// compute_coulomb_log_ext, and the result multiplied by compute_langdon_factor.
TENRYU_HOST_DEVICE inline double compute_kappa_smooth_factor_ext(
    const LaserPhysExtOptions& opt,
    const double n_hat,
    const double Te_eV,
    const double Zbar,
    const double lambda_cm,
    const double eps_n,
    const double coulomb_log_floor,
    const double I_wcm2) {
  (void)eps_n;
  if (!(Te_eV > 0.0) || !(Zbar > 0.0) || !(lambda_cm > 0.0)) {
    return 0.0;
  }

  const double n_hat_clamped = ::fmin(1.0, ::fmax(0.0, n_hat));
  const double lambda_um = lambda_cm * 1.0e4;
  const double Te_eV_32 = Te_eV * ::sqrt(::fmax(Te_eV, 0.0));
  const double Te_32 = Te_eV_32 * (1.0e-3 * ::sqrt(1.0e-3));

  if (!(lambda_um > 0.0) || !(Te_32 > 0.0)) {
    return 0.0;
  }

  double zeff_ratio = 1.0;
  if (opt.zeff_model == 1) {
    zeff_ratio = compute_zeff_ratio_sequential(opt, Zbar);
  } else if (opt.zeff_model == 2) {
    const double ni =
        n_hat_clamped * opt.n_crit_cm3 / ::fmax(Zbar, 1.0e-12);
    zeff_ratio = zeff_ratio_from_table(opt, ni, Te_eV);
  }
  const double Z_coll = Zbar * zeff_ratio;
  const double ln_lambda =
      compute_coulomb_log_ext(opt.coulomb_log_model, n_hat_clamped, Te_eV,
                              Zbar, lambda_cm, coulomb_log_floor);
  const double smooth_factor =
      3.4 * Z_coll * ln_lambda / (lambda_um * lambda_um * Te_32);
  const double langdon_factor =
      compute_langdon_factor(opt.langdon_model, Z_coll, I_wcm2, lambda_cm,
                             Te_eV, opt.langdon_te_min_eV);
  return smooth_factor * langdon_factor;
}

}  // namespace tenryu::laser
