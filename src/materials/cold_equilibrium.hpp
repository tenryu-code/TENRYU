#pragma once

#include <cmath>
#include <cstdint>

#include "core/macros.hpp"

namespace tenryu::materials {

// Cold-equilibrium constitutive branch (NUMERICS §1 (b), 2026-09-16). Below
// the electron-temperature threshold T_star the electron channel carries the
// potential correction G(v, T_e) = w(T_e) C(v), C(v) = chi(rho) C0(v),
// C0(v) = int_{v0}^{v} [P_N(s) - P_cold(s)] ds, P_cold = P0 + K0 (v0/v - 1),
// so that the reference state (rho0, Te0, Ti0) sits at the total pressure
// P0 with the bulk modulus K0. The correction vanishes with two continuous
// derivatives at T_e = T_star and outside the density support. cgs + eV.
struct ColdEquilibriumParams {
  double rho0 = 0.0;                  // reference density [g/cm^3]; <= 0 disables
  double Te0 = 0.0;                   // reference electron temperature [eV]
  double Ti0 = 0.0;                   // reference ion temperature [eV]
  double P0 = 0.0;                    // reference total pressure [dyn/cm^2]
  double K0 = 0.0;                    // reference bulk modulus [dyn/cm^2]
  double T_star = 0.0;                // end of the temperature transition [eV]
  double T_begin_fraction = 0.5;      // T_a = T_begin_fraction * T_star
  double density_core_ratio = 1.10;   // chi = 1 for |ln(rho/rho0)| <= ln(core)
  double density_outer_ratio = 1.50;  // chi = 0 for |ln(rho/rho0)| >= ln(outer)
  int max_iterations = 80;            // cap of the safeguarded q_e -> T_e inverse
};

// Tabulated primitive C0 on ascending specific-volume knots. Host and device
// share this view; the pointers belong to the owner (ColdEquilibriumTable on
// the host, DeviceColdEquilibriumTable on the device).
struct ColdEquilibriumView {
  const double* v_knots = nullptr;   // [n_knots] ascending [cm^3/g]
  const double* PN_knots = nullptr;  // [n_knots] P_N(v_k) [dyn/cm^2]
  const double* C0_knots = nullptr;  // [n_knots] C0(v_k) [erg/g], C0(v0) = 0
  int n_knots = 0;                   // 0 = branch disabled
  double v0 = 0.0;
  double P0 = 0.0;
  double K0 = 0.0;
  double T_star = 0.0;
  double T_a = 0.0;
  double log_rho0 = 0.0;
  double log_core = 0.0;
  double log_outer = 0.0;
  int max_iterations = 0;            // <= 0: default cap (80) of the q_e -> T_e inverse
};

struct ColdGate {
  double w;
  double dw;   // dw/dT [1/eV]
  double d2w;  // d2w/dT2 [1/eV^2]
};

struct ColdReference {
  double C;    // [erg/g]
  double dC;   // dC/dv [dyn/cm^2]
  double d2C;  // d2C/dv2
  double chi;
};

struct ColdElectronState {
  double Pe;       // corrected electron pressure [dyn/cm^2]
  double qe;       // conserved electron caloric coordinate [erg/g]
  double ee_phys;  // qe + C [erg/g]
  double cv;       // corrected electron heat capacity [erg/(g eV)]
};

TENRYU_HOST_DEVICE inline double cold_s5(const double s) {
  return s * s * s * (10.0 + s * (-15.0 + 6.0 * s));
}
TENRYU_HOST_DEVICE inline double cold_s5_d1(const double s) {
  return 30.0 * s * s * (1.0 - s) * (1.0 - s);
}
TENRYU_HOST_DEVICE inline double cold_s5_d2(const double s) {
  return 60.0 * s * (1.0 - s) * (1.0 - 2.0 * s);
}

TENRYU_HOST_DEVICE inline bool cold_enabled(const ColdEquilibriumView& view) {
  return view.n_knots >= 2 && view.v_knots != nullptr && view.PN_knots != nullptr &&
         view.C0_knots != nullptr;
}

TENRYU_HOST_DEVICE inline ColdGate cold_gate(const ColdEquilibriumView& view, const double T) {
  ColdGate g{1.0, 0.0, 0.0};
  if (!(view.T_star > view.T_a) || T <= view.T_a) {
    return g;
  }
  if (T >= view.T_star) {
    g.w = 0.0;
    return g;
  }
  const double dT = view.T_star - view.T_a;
  const double s = (T - view.T_a) / dT;
  g.w = 1.0 - cold_s5(s);
  g.dw = -cold_s5_d1(s) / dT;
  g.d2w = -cold_s5_d2(s) / (dT * dT);
  return g;
}

// P_cold(v) = P0 + K0 (v0/v - 1).
TENRYU_HOST_DEVICE inline double cold_pressure_branch(const ColdEquilibriumView& view, const double v) {
  return view.P0 + view.K0 * (view.v0 / v - 1.0);
}

// Closed-form integral of [P_N(s) - P_cold(s)] over [v_a, v] inside one
// log-linear segment (P_N(s) = P_a + b ln(s / v_a)).
TENRYU_HOST_DEVICE inline double cold_segment_integral(const double v_a, const double P_a, const double b,
                                                      const double v, const double v0, const double P0,
                                                      const double K0) {
  const double L = log(v / v_a);
  return P_a * (v - v_a) + b * (v * L - v + v_a) -
         (P0 * (v - v_a) + K0 * v0 * L - K0 * (v - v_a));
}

// C0, C0', C0'' at v from the knot table (v clamped into the knot range for
// the segment choice; outside the range the end segment is extended).
struct ColdPrimitive {
  double C0;
  double dC0;
  double d2C0;
};

// Segment index of the primitive: the largest k with v_knots[k] <= v, clamped
// to [0, n-2]. When the caller passes the specific-volume range [v_lo, v_hi] of
// its own table bracket (both > 0), the segment is kept inside that range: at
// a knot shared with the table (or within round-off of it) the neighbouring
// segment would otherwise give the derivatives of the other table cell, and
// the density derivative of the branch would no longer cancel the table's on
// the same cell. v_lo <= 0 or v_hi <= 0 disables the check.
TENRYU_HOST_DEVICE inline int cold_segment_index(const ColdEquilibriumView& view, const double v,
                                                 const double v_lo, const double v_hi) {
  const int n = view.n_knots;
  int lo = 0;
  int hi = n - 1;
  if (v > view.v_knots[0]) {
    while (hi - lo > 1) {
      const int mid = lo + ((hi - lo) >> 1);
      if (view.v_knots[mid] <= v) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
  }
  int k = (lo > n - 2) ? (n - 2) : lo;
  if (v_lo > 0.0 && v_hi > 0.0) {
    constexpr double tol = 1.0e-12;
    if (k > 0 && view.v_knots[k + 1] > v_hi * (1.0 + tol)) {
      k -= 1;
    } else if (k < n - 2 && view.v_knots[k] < v_lo * (1.0 - tol)) {
      k += 1;
    }
  }
  return k;
}

TENRYU_HOST_DEVICE inline ColdPrimitive cold_primitive_bracketed(const ColdEquilibriumView& view,
                                                                 const double v, const double v_lo,
                                                                 const double v_hi) {
  ColdPrimitive out{0.0, 0.0, 0.0};
  const int n = view.n_knots;
  if (n < 2) {
    return out;
  }
  const int k = cold_segment_index(view, v, v_lo, v_hi);
  const double v_a = view.v_knots[k];
  const double v_b = view.v_knots[k + 1];
  const double P_a = view.PN_knots[k];
  const double P_b = view.PN_knots[k + 1];
  const double Lab = log(v_b / v_a);
  const double b = (Lab > 1.0e-12) ? (P_b - P_a) / Lab : 0.0;
  const double PN = P_a + b * log(v / v_a);
  out.C0 = view.C0_knots[k] + cold_segment_integral(v_a, P_a, b, v, view.v0, view.P0, view.K0);
  out.dC0 = PN - cold_pressure_branch(view, v);
  out.d2C0 = b / v + view.K0 * view.v0 / (v * v);
  return out;
}

TENRYU_HOST_DEVICE inline ColdPrimitive cold_primitive(const ColdEquilibriumView& view, const double v) {
  return cold_primitive_bracketed(view, v, 0.0, 0.0);
}

// C, dC/dv, d2C/dv2 and chi at rho. rho_lo / rho_hi (both > 0) are the density
// ends of the caller's table bracket; see cold_segment_index. Zero disables
// the segment alignment (values of C are continuous, so callers that only
// need C or dC may pass zero).
TENRYU_HOST_DEVICE inline ColdReference cold_reference_bracketed(const ColdEquilibriumView& view,
                                                                 const double rho,
                                                                 const double rho_lo,
                                                                 const double rho_hi) {
  ColdReference out{0.0, 0.0, 0.0, 0.0};
  if (!cold_enabled(view) || !(rho > 0.0)) {
    return out;
  }
  const double x = log(rho) - view.log_rho0;
  const double y = fabs(x);
  const double a = view.log_core;
  const double bb = view.log_outer;
  double chi = 0.0;
  double chi_y = 0.0;
  double chi_yy = 0.0;
  if (y <= a) {
    chi = 1.0;
  } else if (y < bb) {
    const double width = bb - a;
    const double s = (y - a) / width;
    chi = 1.0 - cold_s5(s);
    chi_y = -cold_s5_d1(s) / width;
    chi_yy = -cold_s5_d2(s) / (width * width);
  }
  out.chi = chi;
  if (chi == 0.0 && chi_y == 0.0 && chi_yy == 0.0) {
    return out;  // outside the density support: C = C' = C'' = 0
  }
  const double sgn = (x > 0.0) ? 1.0 : ((x < 0.0) ? -1.0 : 0.0);
  const double chi_x = sgn * chi_y;
  const double chi_xx = chi_yy;
  const double v = 1.0 / rho;
  const double chi_v = -chi_x / v;
  const double chi_vv = chi_xx / (v * v) + chi_x / (v * v);
  const double v_lo = (rho_hi > 0.0) ? 1.0 / rho_hi : 0.0;
  const double v_hi = (rho_lo > 0.0) ? 1.0 / rho_lo : 0.0;
  const ColdPrimitive p = cold_primitive_bracketed(view, v, v_lo, v_hi);
  out.C = chi * p.C0;
  out.dC = chi_v * p.C0 + chi * p.dC0;
  out.d2C = chi_vv * p.C0 + 2.0 * chi_v * p.dC0 + chi * p.d2C0;
  return out;
}

TENRYU_HOST_DEVICE inline ColdReference cold_reference(const ColdEquilibriumView& view, const double rho) {
  return cold_reference_bracketed(view, rho, 0.0, 0.0);
}

// Corrected electron quantities from the base table values at (rho, T).
TENRYU_HOST_DEVICE inline ColdElectronState cold_electron_forward(const ColdEquilibriumView& view,
                                                                  const double rho, const double T,
                                                                  const double ee_base,
                                                                  const double Pe_base,
                                                                  const double cv_base) {
  const ColdReference r = cold_reference(view, rho);
  const ColdGate g = cold_gate(view, T);
  ColdElectronState s;
  s.Pe = Pe_base - g.w * r.dC;
  s.qe = ee_base + (g.w - T * g.dw - 1.0) * r.C;
  s.ee_phys = s.qe + r.C;
  s.cv = cv_base - T * g.d2w * r.C;
  return s;
}

struct ColdInverseResult {
  double T;
  int status;      // 0 ok, 1 lower_clamp (target <= F(T_lo)), 2 upper_clamp (target >= F(T_hi)), 3 not converged, 4 invalid input
  int iterations;
};

// Safeguarded Newton/bisection for F(T) = ee_base(T) + (w - T w' - 1) C - qe_target
// on [T_lo, T_hi]. `ee_base(T)` and `cv_base(T)` are callables evaluating the
// base electron table at the fixed density (host: EOSTable::energy/cv, device:
// device_eos_energy/device_eos_cv with a fixed rho bracket).
template <class EBase, class CvBase>
TENRYU_HOST_DEVICE inline ColdInverseResult cold_inverse_Te(const ColdEquilibriumView& view,
                                                            const double rho,
                                                            const double qe_target,
                                                            const double T_lo,
                                                            const double T_hi,
                                                            const double T_hint,
                                                            EBase&& ee_base,
                                                            CvBase&& cv_base) {
  ColdInverseResult out{T_lo, 4, 0};
  if (!(T_hi > T_lo) || !(T_lo > 0.0) || !(qe_target == qe_target) || !(rho > 0.0)) {
    return out;
  }
  const double C = cold_reference(view, rho).C;
  auto F = [&](const double T) {
    const ColdGate g = cold_gate(view, T);
    return ee_base(T) + (g.w - T * g.dw - 1.0) * C - qe_target;
  };
  // An exact root at a bound is a valid solution (status 0); a target beyond
  // the bound is a clamp (status 1 / 2).
  const double F_lo = F(T_lo);
  if (F_lo == 0.0) {
    out.status = 0;
    return out;
  }
  if (F_lo > 0.0) {
    out.status = 1;
    return out;
  }
  const double F_hi = F(T_hi);
  if (F_hi == 0.0) {
    out.T = T_hi;
    out.status = 0;
    return out;
  }
  if (F_hi < 0.0) {
    out.T = T_hi;
    out.status = 2;
    return out;
  }
  // Bracketing iteration in T with Newton acceleration. Convergence is judged
  // on the temperature bracket (relative width 4e-15) or on a Newton step of
  // that size, never on an energy tolerance: a table whose electron energy is
  // nearly flat at low temperature (bound electrons) would otherwise accept
  // any T with a small |F| far from the root. f == 0 counts as the upper side
  // so that an exactly flat stretch of q_e(T) converges to its lowest
  // temperature, the convention of the base table inverse. Two consecutive
  // Newton steps that do not cross the root force a bisection step.
  const int max_it = (view.max_iterations > 0) ? view.max_iterations : 80;
  double L = T_lo;
  double R = T_hi;
  double T = (T_hint == T_hint && T_hint > L && T_hint < R) ? T_hint : sqrt(L * R);
  int stall = 0;
  double sign_prev = 0.0;
  bool newton_prev = false;
  for (int it = 1; it <= max_it; ++it) {
    out.iterations = it;
    const double f = F(T);
    const double sign = (f < 0.0) ? -1.0 : 1.0;
    if (f < 0.0) {
      L = T;
    } else {
      R = T;
    }
    if ((R - L) <= 4.0e-15 * R) {
      out.T = R;
      out.status = 0;
      return out;
    }
    stall = (newton_prev && sign == sign_prev) ? stall + 1 : 0;
    sign_prev = sign;
    double T_new = sqrt(L * R);
    newton_prev = false;
    if (f != 0.0 && stall < 2) {
      const ColdGate g = cold_gate(view, T);
      const double cv = cv_base(T) - T * g.d2w * C;
      if (cv > 0.0) {
        const double T_newton = T - f / cv;
        if (T_newton > L && T_newton < R && T_newton != T) {
          T_new = T_newton;
          newton_prev = true;
        }
      }
    }
    T = T_new;
  }
  out.T = R;
  out.status = 3;
  return out;
}

// Electron pressure to use in the reversible pdV work of one hydro step when
// the cold branch is on: P_e^N + (1 - w) D_C with D_C the divided difference
// of C(v) over the step, so that q_e evolves exactly by -P_e^N dv - (1-w) dC
// while the mechanical part dC = D_C dv is carried by C(v) itself
// (NUMERICS §1 (b)). Pe_half is the corrected (stored) half-step pressure
// P~_e = P_e^N - w C', so P_e^N = Pe_half + w C'(v_half). Without a cold
// branch this returns Pe_half unchanged. v_old / v_new are the specific
// volumes at the beginning and end of the step [cm^3/g].
TENRYU_HOST_DEVICE inline double cold_electron_work_pressure(const ColdEquilibriumView& cold,
                                                             const double rho_half,
                                                             const double Te_half,
                                                             const double Pe_half,
                                                             const double v_old,
                                                             const double v_new) {
  if (!cold_enabled(cold)) {
    return Pe_half;
  }
  const ColdGate g = cold_gate(cold, Te_half);
  const ColdReference r_half = cold_reference(cold, rho_half);
  double D_C = r_half.dC;
  const double dv = v_new - v_old;
  if (fabs(dv) > 1.0e-12 * fmax(fabs(v_old), 1.0e-300) && v_old > 0.0 && v_new > 0.0) {
    const double C_new = cold_reference(cold, 1.0 / v_new).C;
    const double C_old = cold_reference(cold, 1.0 / v_old).C;
    D_C = (C_new - C_old) / dv;
  }
  return Pe_half + g.w * r_half.dC + (1.0 - g.w) * D_C;
}

}  // namespace tenryu::materials
