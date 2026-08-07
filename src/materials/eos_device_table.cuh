#pragma once

#include <cmath>
#include <cstdint>

namespace tenryu::materials {

struct DeviceEOSTableView {
  const double* log_rho_grid;  // [n_rho]
  const double* log_T_grid;    // [n_T]
  const double* P_table;       // [n_T * n_rho], idx = j_T * n_rho + i_rho
  const double* e_table;       // [n_T * n_rho], idx = j_T * n_rho + i_rho
  const double* cv_table;      // [n_T * n_rho], idx = j_T * n_rho + i_rho
  int n_rho;                   // 0 = no table (ideal-gas fallback in caller)
  int n_T;
  double log_rho_min;
  double log_rho_max;
  double log_T_min;
  double log_T_max;
  double d_log_rho_inv;  // 0 = non-uniform -> binary search
  double d_log_T_inv;    // 0 = non-uniform -> binary search
  std::uint8_t supports_rho_e_reclosure;  // table energy is monotone enough for T(rho,e)
};

struct RhoBracket {
  int i0;
  int i1;
  double w;
};

struct DeviceEOSInverseRecloseResult {
  double T;
  double logT;
  double pressure;
  double energy;
  double cv;
  int bracket_failure;
  int fallback_used;
  int lower_clamp;
  int upper_clamp;
};

#ifdef __CUDACC__
__device__ inline int clamp_int(const int v, const int lo, const int hi) {
  if (v < lo) {
    return lo;
  }
  if (v > hi) {
    return hi;
  }
  return v;
}

__device__ inline double clamp01(const double x) {
  return fmin(fmax(x, 0.0), 1.0);
}

__device__ inline double clamp_log_value(double x,
                                         const double x_min,
                                         const double x_max) {
  if (!isfinite(x)) {
    x = x_min;
  }
  if (x < x_min) {
    x = x_min;
  }
  if (x > x_max) {
    x = x_max;
  }
  return x;
}

__device__ inline double table_log_coord(const double* grid,
                                         const int n,
                                         const int idx,
                                         const double x_min,
                                         const double x_max) {
  const int i = clamp_int(idx, 0, (n > 0) ? (n - 1) : 0);
  if (grid != nullptr) {
    return grid[i];
  }
  if (n <= 1) {
    return x_min;
  }
  const double alpha = static_cast<double>(i) / static_cast<double>(n - 1);
  return x_min + alpha * (x_max - x_min);
}

__device__ inline double mixed_e_row(const DeviceEOSTableView& tab,
                                     const int i0,
                                     const int i1,
                                     const double wr,
                                     const int j) {
  const int base = j * tab.n_rho;
  const double e0 = tab.e_table[base + i0];
  const double e1 = tab.e_table[base + i1];
  return e0 + wr * (e1 - e0);
}

__device__ inline RhoBracket find_bracket_from_log(const double* grid,
                                                   const int n,
                                                   const double x_raw,
                                                   const double x_min,
                                                   const double x_max,
                                                   double dlog_inv) {
  if (n <= 0) {
    return RhoBracket{0, 0, 0.0};
  }
  if (n == 1) {
    return RhoBracket{0, 0, 0.0};
  }

  const double x = clamp_log_value(x_raw, x_min, x_max);

  if (!(dlog_inv > 0.0) || !isfinite(dlog_inv)) {
    const double span = x_max - x_min;
    if (grid == nullptr && span > 0.0) {
      dlog_inv = static_cast<double>(n - 1) / span;
    }
  }

  if (dlog_inv > 0.0 && isfinite(dlog_inv)) {
    const double pos = (x - x_min) * dlog_inv;
    int i0 = static_cast<int>(floor(pos));
    i0 = clamp_int(i0, 0, n - 2);
    const int i1 = i0 + 1;
    const double w = clamp01(pos - static_cast<double>(i0));
    return RhoBracket{i0, i1, w};
  }

  if (grid == nullptr) {
    return RhoBracket{0, 1, 0.0};
  }

  if (x <= grid[0]) {
    return RhoBracket{0, 1, 0.0};
  }
  if (x >= grid[n - 1]) {
    return RhoBracket{n - 2, n - 1, 1.0};
  }

  int lo = 0;
  int hi = n - 1;
  while (hi - lo > 1) {
    const int mid = lo + ((hi - lo) >> 1);
    if (grid[mid] <= x) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const double x0 = grid[lo];
  const double x1 = grid[hi];
  const double w = (x1 > x0) ? clamp01((x - x0) / (x1 - x0)) : 0.0;
  return RhoBracket{lo, hi, w};
}

__device__ inline RhoBracket find_T_bracket_from_log(const DeviceEOSTableView& tab,
                                                     const double logT) {
  return find_bracket_from_log(tab.log_T_grid,
                               tab.n_T,
                               logT,
                               tab.log_T_min,
                               tab.log_T_max,
                               tab.d_log_T_inv);
}

__device__ inline RhoBracket find_rho_bracket(const DeviceEOSTableView& tab,
                                              const double rho) {
  if (tab.n_rho <= 0) {
    return RhoBracket{0, 0, 0.0};
  }
  double log_rho = tab.log_rho_min;
  if (rho > 0.0 && isfinite(rho)) {
    log_rho = log(rho);
  }
  return find_bracket_from_log(tab.log_rho_grid,
                               tab.n_rho,
                               log_rho,
                               tab.log_rho_min,
                               tab.log_rho_max,
                               tab.d_log_rho_inv);
}

__device__ inline RhoBracket find_T_bracket(const DeviceEOSTableView& tab,
                                            const double T) {
  if (tab.n_T <= 0) {
    return RhoBracket{0, 0, 0.0};
  }
  double log_T = tab.log_T_min;
  if (T > 0.0 && isfinite(T)) {
    log_T = log(T);
  }
  return find_bracket_from_log(tab.log_T_grid,
                               tab.n_T,
                               log_T,
                               tab.log_T_min,
                               tab.log_T_max,
                               tab.d_log_T_inv);
}

__device__ inline double interp_at(const DeviceEOSTableView& tab,
                                   const double* field,
                                   const RhoBracket& rb,
                                   const double logT) {
  if (tab.n_rho <= 0 || tab.n_T <= 0 || field == nullptr) {
    return 0.0;
  }

  int i0 = 0;
  int i1 = 0;
  double wr = 0.0;
  if (tab.n_rho == 1) {
    i0 = 0;
    i1 = 0;
    wr = 0.0;
  } else {
    i0 = clamp_int(rb.i0, 0, tab.n_rho - 2);
    i1 = clamp_int(rb.i1, i0 + 1, tab.n_rho - 1);
    wr = clamp01(rb.w);
  }

  const RhoBracket tb = find_T_bracket_from_log(tab, logT);
  int j0 = 0;
  int j1 = 0;
  double wt = 0.0;
  if (tab.n_T == 1) {
    j0 = 0;
    j1 = 0;
    wt = 0.0;
  } else {
    j0 = clamp_int(tb.i0, 0, tab.n_T - 2);
    j1 = clamp_int(tb.i1, j0 + 1, tab.n_T - 1);
    wt = clamp01(tb.w);
  }

  const int base0 = j0 * tab.n_rho;
  const int base1 = j1 * tab.n_rho;
  const double v00 = field[base0 + i0];
  const double v10 = field[base0 + i1];
  const double v01 = field[base1 + i0];
  const double v11 = field[base1 + i1];

  const double vx0 = v00 + wr * (v10 - v00);
  const double vx1 = v01 + wr * (v11 - v01);
  return vx0 + wt * (vx1 - vx0);
}

struct AnalyticIdealEos {
  double e_mass;   // [erg/g]
  double cv_mass;  // [erg/g/eV]
  double P;        // [erg/cm^3]
};

__device__ inline AnalyticIdealEos analytic_ideal_electron_eos(
    double rho_gcc,  // mass density [g/cm^3]
    double T_eV,     // temperature [eV]
    double Zbar,     // mean ionization
    double A_amu) {  // atomic weight [amu]
  // Nondegenerate ideal electron gas (CGS+eV).
  // n_e = Zbar * rho / (A * m_p), [cm^-3]
  // cv_mass = (3/2) * Zbar * eV_to_erg / (A * m_p), [erg/g/eV]
  // e_mass = cv_mass * T, [erg/g]
  // P = n_e * T * eV_to_erg = rho * cv_mass * T * (2/3), [erg/cm^3]
  // Note: P_e = (gamma-1)*rho*e_e for ideal monatomic, gamma=5/3, so P =
  // (2/3)*rho*e.
  AnalyticIdealEos out;
  constexpr double kProtonMass = 1.6726219e-24;  // [g]
  constexpr double kEvToErg = 1.6022e-12;        // [erg/eV]
  const double Z_eff = (Zbar > 0.0) ? Zbar : 1.0;
  const double A_eff = (A_amu > 0.0) ? A_amu : 1.0;
  const double T_eff = (T_eV > 0.0) ? T_eV : 1.0e-30;
  out.cv_mass = 1.5 * Z_eff * kEvToErg / (A_eff * kProtonMass);
  out.e_mass = out.cv_mass * T_eff;
  out.P = (2.0 / 3.0) * rho_gcc * out.e_mass;
  return out;
}

__device__ inline bool eos_use_low_density_extrapolation(
    const DeviceEOSTableView& tab,
    bool flag_enabled,
    double rho_gcc) {
  // Returns true when flag is enabled AND rho is below the table minimum density.
  if (!flag_enabled) return false;
  if (tab.n_rho <= 0) return false;
  // Compare positive densities in log-space so rho == rho_min stays in-table.
  if (rho_gcc > 0.0 && isfinite(rho_gcc)) {
    const double log_rho = log(rho_gcc);
    const double tol = 8.0e-15 * fmax(fabs(tab.log_rho_min), 1.0);
    return isfinite(log_rho) && (log_rho < tab.log_rho_min - tol);
  }
  const double rho_min = exp(tab.log_rho_min);
  return (rho_min > 0.0) && (rho_gcc < rho_min);
}

__device__ inline double device_eos_pressure(const DeviceEOSTableView& tab,
                                             const RhoBracket& rb,
                                             const double logT) {
  return interp_at(tab, tab.P_table, rb, logT);
}

__device__ inline double device_eos_energy(const DeviceEOSTableView& tab,
                                           const RhoBracket& rb,
                                           const double logT) {
  return interp_at(tab, tab.e_table, rb, logT);
}

__device__ inline double device_eos_cv(const DeviceEOSTableView& tab,
                                       const RhoBracket& rb,
                                       const double logT) {
  return interp_at(tab, tab.cv_table, rb, logT);
}

__device__ inline double device_eos_pressure_extrap(
    const DeviceEOSTableView& tab,
    const RhoBracket& rb,
    double rho_gcc,
    double T_eV,
    double Zbar,
    double A_amu,
    bool flag_enabled) {
  if (eos_use_low_density_extrapolation(tab, flag_enabled, rho_gcc)) {
    AnalyticIdealEos a = analytic_ideal_electron_eos(rho_gcc, T_eV, Zbar, A_amu);
    return a.P;
  }
  const double logT = log((T_eV > 0.0) ? T_eV : 1.0e-30);
  return device_eos_pressure(tab, rb, logT);
}

__device__ inline double device_eos_energy_extrap(
    const DeviceEOSTableView& tab,
    const RhoBracket& rb,
    double rho_gcc,
    double T_eV,
    double Zbar,
    double A_amu,
    bool flag_enabled) {
  if (eos_use_low_density_extrapolation(tab, flag_enabled, rho_gcc)) {
    AnalyticIdealEos a = analytic_ideal_electron_eos(rho_gcc, T_eV, Zbar, A_amu);
    return a.e_mass;
  }
  const double logT = log((T_eV > 0.0) ? T_eV : 1.0e-30);
  return device_eos_energy(tab, rb, logT);
}

__device__ inline double device_eos_cv_extrap(
    const DeviceEOSTableView& tab,
    const RhoBracket& rb,
    double rho_gcc,
    double T_eV,
    double Zbar,
    double A_amu,
    bool flag_enabled) {
  if (eos_use_low_density_extrapolation(tab, flag_enabled, rho_gcc)) {
    AnalyticIdealEos a = analytic_ideal_electron_eos(rho_gcc, T_eV, Zbar, A_amu);
    return a.cv_mass;
  }
  const double logT = log((T_eV > 0.0) ? T_eV : 1.0e-30);
  return device_eos_cv(tab, rb, logT);
}

__device__ inline double device_eos_T_from_e_monotone(const DeviceEOSTableView& tab,
                                                       const RhoBracket& rb,
                                                       const double e_target) {
  if (tab.n_rho <= 0 || tab.n_T <= 0 || tab.e_table == nullptr) {
    return 0.0;
  }

  int i0 = 0;
  int i1 = 0;
  double wr = 0.0;
  if (tab.n_rho == 1) {
    i0 = 0;
    i1 = 0;
    wr = 0.0;
  } else {
    i0 = clamp_int(rb.i0, 0, tab.n_rho - 2);
    i1 = clamp_int(rb.i1, i0 + 1, tab.n_rho - 1);
    wr = clamp01(rb.w);
  }

  if (tab.n_T == 1) {
    const double logT0 = table_log_coord(
        tab.log_T_grid, tab.n_T, 0, tab.log_T_min, tab.log_T_max);
    return exp(logT0);
  }

  if (!isfinite(e_target)) {
    const double logT0 = table_log_coord(
        tab.log_T_grid, tab.n_T, 0, tab.log_T_min, tab.log_T_max);
    return exp(logT0);
  }

  const double e_min = mixed_e_row(tab, i0, i1, wr, 0);
  if (e_target < e_min) {
    const double logT0 = table_log_coord(
        tab.log_T_grid, tab.n_T, 0, tab.log_T_min, tab.log_T_max);
    return exp(logT0);
  }

  const int j_last = tab.n_T - 1;
  const double e_max = mixed_e_row(tab, i0, i1, wr, j_last);
  if (e_target >= e_max) {
    const double logT1 = table_log_coord(
        tab.log_T_grid, tab.n_T, j_last, tab.log_T_min, tab.log_T_max);
    return exp(logT1);
  }

  int lo = 0;
  int hi = j_last;
  while (hi - lo > 1) {
    const int mid = lo + ((hi - lo) >> 1);
    if (mixed_e_row(tab, i0, i1, wr, mid) <= e_target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const int j0 = lo;
  const int j1 = lo + 1;
  const double e0 = mixed_e_row(tab, i0, i1, wr, j0);
  const double e1 = mixed_e_row(tab, i0, i1, wr, j1);
  const double alpha = (e1 > e0) ? clamp01((e_target - e0) / (e1 - e0)) : 0.0;

  const double logT0 = table_log_coord(
      tab.log_T_grid, tab.n_T, j0, tab.log_T_min, tab.log_T_max);
  const double logT1 = table_log_coord(
      tab.log_T_grid, tab.n_T, j1, tab.log_T_min, tab.log_T_max);
  const double logT = logT0 + alpha * (logT1 - logT0);
  return exp(logT);
}

__device__ inline DeviceEOSInverseRecloseResult device_inverse_reclose(
    const DeviceEOSTableView& tab,
    const double rho,
    const double e_target,
    const double T_floor) {
  DeviceEOSInverseRecloseResult out{};
  out.T = fmax(T_floor, 1.0e-30);
  out.logT = log(out.T);
  out.pressure = 0.0;
  out.energy = 0.0;
  out.cv = 0.0;
  out.bracket_failure = 0;
  out.fallback_used = 0;
  out.lower_clamp = 0;
  out.upper_clamp = 0;

  if (tab.n_rho <= 0 || tab.n_T <= 0 || tab.e_table == nullptr ||
      !isfinite(rho) || !(rho > 0.0) || !isfinite(e_target)) {
    out.bracket_failure = 1;
    return out;
  }

  const RhoBracket rb = find_rho_bracket(tab, rho);
  int i0 = 0;
  int i1 = 0;
  double wr = 0.0;
  if (tab.n_rho == 1) {
    i0 = 0;
    i1 = 0;
    wr = 0.0;
  } else {
    i0 = clamp_int(rb.i0, 0, tab.n_rho - 2);
    i1 = clamp_int(rb.i1, i0 + 1, tab.n_rho - 1);
    wr = clamp01(rb.w);
  }

  const double e_min = mixed_e_row(tab, i0, i1, wr, 0);
  const double e_max = mixed_e_row(tab, i0, i1, wr, tab.n_T - 1);
  const bool bad_bracket = !isfinite(e_min) || !isfinite(e_max) || !(e_max > e_min);
  if (bad_bracket) {
    out.bracket_failure = 1;
  } else if (e_target < e_min) {
    out.lower_clamp = 1;
  } else if (e_target > e_max) {
    out.upper_clamp = 1;
  }

  out.T = fmax(device_eos_T_from_e_monotone(tab, rb, e_target), fmax(T_floor, 1.0e-30));
  out.logT = log(fmax(out.T, 1.0e-30));
  out.pressure = device_eos_pressure(tab, rb, out.logT);
  out.energy = device_eos_energy(tab, rb, out.logT);
  out.cv = fmax(device_eos_cv(tab, rb, out.logT), 0.0);
  return out;
}

__device__ inline DeviceEOSInverseRecloseResult
device_inverse_reclose_with_low_density_extrap(
    const DeviceEOSTableView& tab,
    const double rho,
    const double e_target,
    const double T_floor,
    const double Zbar,
    const double A_amu,
    const bool flag_enabled) {
  if (!eos_use_low_density_extrapolation(tab, flag_enabled, rho)) {
    return device_inverse_reclose(tab, rho, e_target, T_floor);
  }

  DeviceEOSInverseRecloseResult out{};
  out.T = fmax(T_floor, 1.0e-30);
  out.logT = log(out.T);
  out.pressure = 0.0;
  out.energy = 0.0;
  out.cv = 0.0;
  out.bracket_failure = 0;
  out.fallback_used = 0;
  out.lower_clamp = 0;
  out.upper_clamp = 0;

  if (!isfinite(rho) || !(rho > 0.0) || !isfinite(e_target)) {
    out.bracket_failure = 1;
    return out;
  }

  const AnalyticIdealEos unit_T = analytic_ideal_electron_eos(rho, 1.0, Zbar, A_amu);
  if (!isfinite(unit_T.cv_mass) || !(unit_T.cv_mass > 0.0)) {
    out.bracket_failure = 1;
    return out;
  }

  out.T = fmax(e_target / unit_T.cv_mass, fmax(T_floor, 1.0e-30));
  out.logT = log(fmax(out.T, 1.0e-30));
  const AnalyticIdealEos state = analytic_ideal_electron_eos(rho, out.T, Zbar, A_amu);
  out.pressure = state.P;
  out.energy = state.e_mass;
  out.cv = fmax(state.cv_mass, 0.0);
  return out;
}

// ---- BUG-24 high-temperature ideal-tail extension -------------------------
// Beyond the table's temperature ceiling T_top(rho) the tabulated e/P/cv
// saturate, so every table-projecting energy write deletes the excess of a
// super-ceiling cell. These helpers extend the state surface with a
// monotone, C0 ideal tail anchored at the top row:
//   e(T)  = e_top + cv_top * (T - T_top)
//   P(T)  = P_top * (T / T_top)
//   cv(T) = cv_top
// The tail is opt-in (energy-authoritative closure mode); legacy paths are
// untouched.

struct DeviceEOSHighTTailAnchor {
  double T_top = 0.0;
  double e_top = 0.0;
  double P_top = 0.0;
  double cv_top = 0.0;
  int valid = 0;
};

__device__ inline DeviceEOSHighTTailAnchor device_eos_high_t_tail_anchor(
    const DeviceEOSTableView& tab,
    const RhoBracket& rb) {
  DeviceEOSHighTTailAnchor a{};
  if (tab.n_rho <= 0 || tab.n_T <= 0 || tab.e_table == nullptr) {
    return a;
  }
  const double logT_top = tab.log_T_max;
  a.T_top = exp(logT_top);
  a.e_top = device_eos_energy(tab, rb, logT_top);
  a.P_top = device_eos_pressure(tab, rb, logT_top);
  a.cv_top = device_eos_cv(tab, rb, logT_top);
  a.valid = (isfinite(a.T_top) && a.T_top > 0.0 && isfinite(a.e_top) &&
             isfinite(a.P_top) && isfinite(a.cv_top) && a.cv_top > 0.0)
                ? 1
                : 0;
  return a;
}

// Forward evaluation with the tail: below/at the ceiling this is exactly
// the plain table evaluation at log(max(T, floor)); above it, the tail.
struct DeviceEOSTailThermo {
  double e = 0.0;
  double P = 0.0;
  double cv = 0.0;
  int in_tail = 0;
};

__device__ inline DeviceEOSTailThermo device_eos_eval_with_high_t_tail(
    const DeviceEOSTableView& tab,
    const RhoBracket& rb,
    const double T,
    const double T_floor) {
  DeviceEOSTailThermo out{};
  const double T_eff = fmax(T, fmax(T_floor, 1.0e-30));
  const DeviceEOSHighTTailAnchor a = device_eos_high_t_tail_anchor(tab, rb);
  if (a.valid != 0 && T_eff > a.T_top) {
    out.e = a.e_top + a.cv_top * (T_eff - a.T_top);
    out.P = a.P_top * (T_eff / a.T_top);
    out.cv = a.cv_top;
    out.in_tail = 1;
    return out;
  }
  const double logT = log(T_eff);
  out.e = device_eos_energy(tab, rb, logT);
  out.P = device_eos_pressure(tab, rb, logT);
  out.cv = fmax(device_eos_cv(tab, rb, logT), 0.0);
  out.in_tail = 0;
  return out;
}

// Inverse reclosure with the tail: identical to device_inverse_reclose
// except that an upper-clamped target inverts INTO the tail — the returned
// energy equals e_target exactly (energy-authoritative), T extends beyond
// the ceiling, and P/cv come from the tail. lower_clamp / bracket_failure
// results pass through unchanged. upper_clamp stays set as the in-tail
// marker.
__device__ inline DeviceEOSInverseRecloseResult
device_inverse_reclose_with_high_t_tail(
    const DeviceEOSTableView& tab,
    const double rho,
    const double e_target,
    const double T_floor) {
  DeviceEOSInverseRecloseResult out =
      device_inverse_reclose(tab, rho, e_target, T_floor);
  if (out.bracket_failure != 0 || out.upper_clamp == 0) {
    return out;
  }
  const RhoBracket rb = find_rho_bracket(tab, rho);
  const DeviceEOSHighTTailAnchor a = device_eos_high_t_tail_anchor(tab, rb);
  if (a.valid == 0) {
    return out;
  }
  const double T_tail = a.T_top + (e_target - a.e_top) / a.cv_top;
  if (!isfinite(T_tail) || T_tail <= a.T_top) {
    return out;
  }
  out.T = T_tail;
  out.logT = log(T_tail);
  out.energy = e_target;
  out.pressure = a.P_top * (T_tail / a.T_top);
  out.cv = a.cv_top;
  return out;
}

__device__ inline double device_eos_sound_speed(const DeviceEOSTableView& tab,
                                                 const RhoBracket& rb,
                                                 const double logT,
                                                 const double rho,
                                                 const double cv) {
  if (tab.n_rho <= 0 || tab.n_T <= 0 || tab.P_table == nullptr) {
    return 0.0;
  }
  if (!(rho > 0.0) || !isfinite(rho)) {
    return 0.0;
  }

  int i0 = 0;
  int i1 = 0;
  double wr = 0.0;
  if (tab.n_rho == 1) {
    i0 = 0;
    i1 = 0;
    wr = 0.0;
  } else {
    i0 = clamp_int(rb.i0, 0, tab.n_rho - 2);
    i1 = clamp_int(rb.i1, i0 + 1, tab.n_rho - 1);
    wr = clamp01(rb.w);
  }

  const RhoBracket tb = find_T_bracket_from_log(tab, logT);
  int j0 = 0;
  int j1 = 0;
  double wt = 0.0;
  if (tab.n_T == 1) {
    j0 = 0;
    j1 = 0;
    wt = 0.0;
  } else {
    j0 = clamp_int(tb.i0, 0, tab.n_T - 2);
    j1 = clamp_int(tb.i1, j0 + 1, tab.n_T - 1);
    wt = clamp01(tb.w);
  }

  const int base0 = j0 * tab.n_rho;
  const int base1 = j1 * tab.n_rho;
  const double p00 = tab.P_table[base0 + i0];
  const double p10 = tab.P_table[base0 + i1];
  const double p01 = tab.P_table[base1 + i0];
  const double p11 = tab.P_table[base1 + i1];

  const double p_rho0 = p00 + wt * (p01 - p00);
  const double p_rho1 = p10 + wt * (p11 - p10);
  const double p_T0 = p00 + wr * (p10 - p00);
  const double p_T1 = p01 + wr * (p11 - p01);

  const double P = p_T0 + wt * (p_T1 - p_T0);

  const double log_rho0 = table_log_coord(
      tab.log_rho_grid, tab.n_rho, i0, tab.log_rho_min, tab.log_rho_max);
  const double log_rho1 = table_log_coord(
      tab.log_rho_grid, tab.n_rho, i1, tab.log_rho_min, tab.log_rho_max);
  const double rho0 = exp(log_rho0);
  const double rho1 = exp(log_rho1);

  const double log_T0 = table_log_coord(
      tab.log_T_grid, tab.n_T, j0, tab.log_T_min, tab.log_T_max);
  const double log_T1 = table_log_coord(
      tab.log_T_grid, tab.n_T, j1, tab.log_T_min, tab.log_T_max);

  // Analytic local derivatives of the bilinear-in-(ln rho, ln T) interpolant:
  //   dP/drho|_T = [(1-wt)(P10-P00) + wt(P11-P01)] / (rho * dln_rho)
  //   dP/dT|_rho = [(1-wr)(P01-P00) + wr(P11-P10)] / (T * dln_T)
  // evaluated at the query point clamped into the enclosing cell, matching the
  // clamped bracket the interpolation itself uses. Cell-wide linear secants
  // (P1-P0)/(rho1-rho0) would give the mean slope near the cell's logarithmic
  // mean, not the slope at the query point (AI review k11 §3.2).
  const double dln_rho = log_rho1 - log_rho0;
  const double dln_T = log_T1 - log_T0;
  const double rho_q = fmin(fmax(rho, rho0), rho1);
  const double logT_clamped = clamp_log_value(logT, tab.log_T_min, tab.log_T_max);
  const double T = exp(logT_clamped);

  const double dPdrho_T =
      (dln_rho > 0.0 && rho_q > 0.0) ? ((p_rho1 - p_rho0) / (rho_q * dln_rho)) : 0.0;
  const double dPdT_rho = (dln_T > 0.0) ? ((p_T1 - p_T0) / (T * dln_T)) : 0.0;

  const double cv_safe = (cv > 0.0 && isfinite(cv)) ? cv : 0.0;
  double gamma1 = 0.0;
  if (P > 0.0 && isfinite(P)) {
    gamma1 = (rho / P) * dPdrho_T;
    if (cv_safe > 0.0) {
      gamma1 += T * dPdT_rho * dPdT_rho / (rho * P * cv_safe);
    }
  }

  const double cs2 = gamma1 * P / rho;
  const double cs = (cs2 > 0.0) ? sqrt(cs2) : 0.0;
  if (isfinite(cs) && cs > 0.0) {
    return cs;
  }

  const double cs2_fallback = (P > 0.0 && isfinite(P)) ? ((5.0 / 3.0) * P / rho) : 0.0;
  return (cs2_fallback > 0.0) ? sqrt(cs2_fallback) : 0.0;
}
#endif  // __CUDACC__

}  // namespace tenryu::materials
