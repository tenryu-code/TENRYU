#pragma once

#include <cmath>

#include "core/macros.hpp"

namespace tenryu::materials {

struct ZMomentDeviceTables {
  const double* r2 = nullptr;  // [nd*nt], d*nt+t
  const double* r4 = nullptr;
  int nd = 0;
  int nt = 0;
  double l10d0 = 0.0;
  double dl10d = 1.0;
  double l10t0 = 0.0;
  double dl10t = 1.0;
};

// Clamped bilinear in (log10 ni, log10 Te); returns 1.0 when table absent.
TENRYU_HOST_DEVICE inline double zmoment_lookup(
    const double* tab,
    const ZMomentDeviceTables& t,
    const double ni_cm3,
    const double Te_eV) {
  if (tab == nullptr || t.nd < 2 || t.nt < 2) {
    return 1.0;
  }

  const double u_raw =
      (::log10(::fmax(ni_cm3, 1.0e-30)) - t.l10d0) / t.dl10d;
  const double v_raw =
      (::log10(::fmax(Te_eV, 1.0e-30)) - t.l10t0) / t.dl10t;
  const double u =
      ::fmin(static_cast<double>(t.nd - 1), ::fmax(0.0, u_raw));
  const double v =
      ::fmin(static_cast<double>(t.nt - 1), ::fmax(0.0, v_raw));
  const int d0 = static_cast<int>(::floor(u));
  const int t0 = static_cast<int>(::floor(v));
  const int d1 = (d0 + 1 < t.nd) ? d0 + 1 : d0;
  const int t1 = (t0 + 1 < t.nt) ? t0 + 1 : t0;
  const double fu = u - static_cast<double>(d0);
  const double fv = v - static_cast<double>(t0);
  const double r00 = tab[d0 * t.nt + t0];
  const double r01 = tab[d0 * t.nt + t1];
  const double r10 = tab[d1 * t.nt + t0];
  const double r11 = tab[d1 * t.nt + t1];
  const double r0 = r00 + fv * (r01 - r00);
  const double r1 = r10 + fv * (r11 - r10);
  return r0 + fu * (r1 - r0);
}

TENRYU_HOST_DEVICE inline double zmoment_r2(
    const ZMomentDeviceTables& t,
    const double ni_cm3,
    const double Te_eV) {
  return zmoment_lookup(t.r2, t, ni_cm3, Te_eV);
}

TENRYU_HOST_DEVICE inline double zmoment_r4(
    const ZMomentDeviceTables& t,
    const double ni_cm3,
    const double Te_eV) {
  return zmoment_lookup(t.r4, t, ni_cm3, Te_eV);
}

}  // namespace tenryu::materials
