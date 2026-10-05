#pragma once

#include <cmath>

#include "core/macros.hpp"
#include "materials/cold_equilibrium.hpp"
#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

constexpr double kRecloseCvFloor = 1.0e-3;

TENRYU_HOST_DEVICE inline double reclose_max(const double a, const double b) {
  return (a < b) ? b : a;
}

TENRYU_HOST_DEVICE inline double reclose_min(const double a, const double b) {
  return (b < a) ? b : a;
}

TENRYU_HOST_DEVICE inline double reclose_clamp(const double value,
                                                const double lo,
                                                const double hi) {
  if (value < lo) {
    return lo;
  }
  if (hi < value) {
    return hi;
  }
  return value;
}

struct RecloseBracket {
  int lo;
  int hi;
};

TENRYU_HOST_DEVICE inline RecloseBracket reclose_bracket_index(
    const double* grid,
    const int n,
    const double x) {
  if (n < 2) {
    return {0, 0};
  }
  if (x <= grid[0]) {
    return {0, 1};
  }
  if (x >= grid[n - 1]) {
    return {n - 2, n - 1};
  }

  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = lo + ((hi - lo) >> 1);
    if (x < grid[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return {lo - 1, lo};
}

TENRYU_HOST_DEVICE inline double reclose_log_coordinate(
    const double value,
    const double* log_grid,
    const int n) {
  const double value_min = ::exp(log_grid[0]);
  const double value_max = ::exp(log_grid[n - 1]);
  const double value_safe = std::isnan(value) ? value_min : value;
  const double value_clamped =
      reclose_clamp(value_safe, value_min, value_max);
  return ::log(value_clamped);
}

// The density half of a bilinear table evaluation: the bracket of the log
// density on the grid and the weight within it, shared by every evaluation at
// that density; the temperature half likewise. The temperature inversion below
// takes the density's once and each temperature's once for its energy and heat
// capacity (2026-10-02; it had searched both grids again for each of them).
struct RecloseDensityPoint {
  RecloseBracket rb;
  double tx;
};

struct RecloseTemperaturePoint {
  RecloseBracket tb;
  double ty;
};

TENRYU_HOST_DEVICE inline RecloseDensityPoint reclose_density_point_at_log(
    const DeviceEOSTableView& table, const double x) {
  const RecloseBracket rb =
      reclose_bracket_index(table.log_rho_grid, table.n_rho, x);
  const double x0 = table.log_rho_grid[rb.lo];
  const double x1 = table.log_rho_grid[rb.hi];
  const double tx = (x1 > x0) ? ((x - x0) / (x1 - x0)) : 0.0;
  return {rb, tx};
}

TENRYU_HOST_DEVICE inline RecloseTemperaturePoint reclose_temperature_point_at_log(
    const DeviceEOSTableView& table, const double y) {
  const RecloseBracket tb =
      reclose_bracket_index(table.log_T_grid, table.n_T, y);
  const double y0 = table.log_T_grid[tb.lo];
  const double y1 = table.log_T_grid[tb.hi];
  const double ty = (y1 > y0) ? ((y - y0) / (y1 - y0)) : 0.0;
  return {tb, ty};
}

TENRYU_HOST_DEVICE inline RecloseDensityPoint reclose_density_point(
    const DeviceEOSTableView& table, const double rho) {
  return reclose_density_point_at_log(
      table, reclose_log_coordinate(rho, table.log_rho_grid, table.n_rho));
}

TENRYU_HOST_DEVICE inline RecloseTemperaturePoint reclose_temperature_point(
    const DeviceEOSTableView& table, const double T_eV) {
  return reclose_temperature_point_at_log(
      table, reclose_log_coordinate(T_eV, table.log_T_grid, table.n_T));
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_bilinear_at_points(
    const DeviceEOSTableView& table, const double* values,
    const RecloseDensityPoint& xp, const RecloseTemperaturePoint& yp) {
  const double tx = xp.tx;
  const double ty = yp.ty;
  const int base0 = yp.tb.lo * table.n_rho;
  const int base1 = yp.tb.hi * table.n_rho;
  const double v00 = values[base0 + xp.rb.lo];
  const double v10 = values[base0 + xp.rb.hi];
  const double v01 = values[base1 + xp.rb.lo];
  const double v11 = values[base1 + xp.rb.hi];

#if defined(__CUDA_ARCH__)
  if constexpr (SeparateProducts) {
    const double vx0 = __dadd_rn(v00, __dmul_rn(tx, v10 - v00));
    const double vx1 = __dadd_rn(v01, __dmul_rn(tx, v11 - v01));
    return __dadd_rn(vx0, __dmul_rn(ty, vx1 - vx0));
  }
#endif
  const double vx0 = v00 + tx * (v10 - v00);
  const double vx1 = v01 + tx * (v11 - v01);
  return vx0 + ty * (vx1 - vx0);
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_bilinear_at_log_coordinates(
    const DeviceEOSTableView& table, const double* values,
    const double x, const double y) {
  return reclose_bilinear_at_points<SeparateProducts>(
      table, values, reclose_density_point_at_log(table, x),
      reclose_temperature_point_at_log(table, y));
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_bilinear_log_interp(
    const DeviceEOSTableView& table,
    const double* values,
    const double rho,
    const double T_eV) {
  const double x = reclose_log_coordinate(rho, table.log_rho_grid, table.n_rho);
  const double y = reclose_log_coordinate(T_eV, table.log_T_grid, table.n_T);
  return reclose_bilinear_at_log_coordinates<SeparateProducts>(table, values, x, y);
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_pressure_base(
    const DeviceEOSTableView& table,
    const double rho,
    const double T_eV) {
  return reclose_bilinear_log_interp<SeparateProducts>(table, table.P_table, rho, T_eV);
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_energy_base(
    const DeviceEOSTableView& table,
    const double rho,
    const double T_eV) {
  return reclose_bilinear_log_interp<SeparateProducts>(table, table.e_table, rho, T_eV);
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_cv_base(const DeviceEOSTableView& table,
                                                  const double rho,
                                                  const double T_eV) {
  return reclose_bilinear_log_interp<SeparateProducts>(table, table.cv_table, rho, T_eV);
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_pressure(const DeviceEOSTableView& table,
                                                  const double rho,
                                                  const double T_eV) {
  const double base = reclose_pressure_base<SeparateProducts>(table, rho, T_eV);
  if (!cold_enabled(table.cold)) {
    return base;
  }
  return base - cold_gate(table.cold, T_eV).w * cold_reference(table.cold, rho).dC;
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_energy(const DeviceEOSTableView& table,
                                                const double rho,
                                                const double T_eV) {
  const double base = reclose_energy_base<SeparateProducts>(table, rho, T_eV);
  if (!cold_enabled(table.cold)) {
    return base;
  }
  const ColdGate g = cold_gate(table.cold, T_eV);
  return base + (g.w - T_eV * g.dw - 1.0) * cold_reference(table.cold, rho).C;
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_cv(const DeviceEOSTableView& table,
                                            const double rho,
                                            const double T_eV) {
  const double base = reclose_cv_base<SeparateProducts>(table, rho, T_eV);
  if (!cold_enabled(table.cold)) {
    return base;
  }
  return base - T_eV * cold_gate(table.cold, T_eV).d2w * cold_reference(table.cold, rho).C;
}

// reclose_pressure / reclose_energy / reclose_cv at points of density rho and
// temperature T_eV (the same values).
template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_pressure_at(const DeviceEOSTableView& table,
                                                     const double rho,
                                                     const RecloseDensityPoint& xp,
                                                     const double T_eV,
                                                     const RecloseTemperaturePoint& yp) {
  const double base = reclose_bilinear_at_points<SeparateProducts>(table, table.P_table, xp, yp);
  if (!cold_enabled(table.cold)) {
    return base;
  }
  return base - cold_gate(table.cold, T_eV).w * cold_reference(table.cold, rho).dC;
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_energy_at(const DeviceEOSTableView& table,
                                                   const double rho,
                                                   const RecloseDensityPoint& xp,
                                                   const double T_eV,
                                                   const RecloseTemperaturePoint& yp) {
  const double base = reclose_bilinear_at_points<SeparateProducts>(table, table.e_table, xp, yp);
  if (!cold_enabled(table.cold)) {
    return base;
  }
  const ColdGate g = cold_gate(table.cold, T_eV);
  return base + (g.w - T_eV * g.dw - 1.0) * cold_reference(table.cold, rho).C;
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_cv_at(const DeviceEOSTableView& table,
                                               const double rho,
                                               const RecloseDensityPoint& xp,
                                               const double T_eV,
                                               const RecloseTemperaturePoint& yp) {
  const double base = reclose_bilinear_at_points<SeparateProducts>(table, table.cv_table, xp, yp);
  if (!cold_enabled(table.cold)) {
    return base;
  }
  return base - T_eV * cold_gate(table.cold, T_eV).d2w * cold_reference(table.cold, rho).C;
}

template <bool SeparateProducts = false>
TENRYU_HOST_DEVICE inline double reclose_temperature_from_energy(
    const DeviceEOSTableView& table,
    const double rho,
    const double e) {
  const double T_min = ::exp(table.log_T_grid[0]);
  const double T_max = ::exp(table.log_T_grid[table.n_T - 1]);
  const RecloseDensityPoint xp = reclose_density_point(table, rho);
  if (cold_enabled(table.cold)) {
    const ColdInverseResult inv = cold_inverse_Te(
        table.cold, rho, e, T_min, T_max, 0.0,
        [&](const double T) {
          return reclose_bilinear_at_points<SeparateProducts>(
              table, table.e_table, xp, reclose_temperature_point(table, T));
        },
        [&](const double T) {
          return reclose_bilinear_at_points<SeparateProducts>(
              table, table.cv_table, xp, reclose_temperature_point(table, T));
        },
        table.log_T_grid, table.n_T);
    return inv.T;
  }
  // Without the cold branch the energy and heat capacity are the table's
  // bilinear values (reclose_energy, reclose_cv).
  const double e_min = reclose_bilinear_at_points<SeparateProducts>(
      table, table.e_table, xp, reclose_temperature_point(table, T_min));
  const double e_max = reclose_bilinear_at_points<SeparateProducts>(
      table, table.e_table, xp, reclose_temperature_point(table, T_max));
  if (e <= e_min) {
    return T_min;
  }
  if (e >= e_max) {
    return T_max;
  }

  double T_lo = T_min;
  double T_hi = T_max;
  const double de_span = e_max - e_min;
  const double de_scale =
      reclose_max(reclose_max(::fabs(e_min), ::fabs(e_max)), 1.0);
  if (!(::fabs(de_span) > 1.0e-14 * de_scale)) {
    return T_min;
  }
  const double alpha = (e - e_min) / de_span;
  double T;
#if defined(__CUDA_ARCH__)
  if constexpr (SeparateProducts) {
    T = reclose_clamp(__dadd_rn(T_min, __dmul_rn(alpha, T_max - T_min)), T_min, T_max);
  } else
#endif
  {
    T = reclose_clamp(T_min + alpha * (T_max - T_min), T_min, T_max);
  }

  for (int iter = 0; iter < 24; ++iter) {
    const RecloseTemperaturePoint yp = reclose_temperature_point(table, T);
    const double e_curr = reclose_bilinear_at_points<SeparateProducts>(table, table.e_table, xp, yp);
    const double cv_curr = reclose_max(
        reclose_bilinear_at_points<SeparateProducts>(table, table.cv_table, xp, yp),
        kRecloseCvFloor);
    const double f = e_curr - e;
    if (::fabs(f) <= 1.0e-10 * reclose_max(::fabs(e), 1.0)) {
      break;
    }

    if (f > 0.0) {
      T_hi = reclose_min(T_hi, T);
    } else {
      T_lo = reclose_max(T_lo, T);
    }

    double T_next = T - f / cv_curr;
    if (!(T_next > T_lo && T_next < T_hi)) {
      T_next = 0.5 * (T_lo + T_hi);
    }
    T = reclose_clamp(T_next, T_min, T_max);
  }

  return T;
}

struct RecloseThermo {
  double T;
  double energy;
  double pressure;
  double cv;
};

TENRYU_HOST_DEVICE inline RecloseThermo reclose_thermo_from_energy(
    const DeviceEOSTableView& table,
    const double rho,
    const double energy,
    const double T_floor) {
  const double rho_safe = reclose_max(rho, 1.0e-30);
  const double T_raw =
      reclose_temperature_from_energy(table, rho_safe, reclose_max(energy, 0.0));
  const double T = (std::isfinite(T_raw) && T_raw >= T_floor) ? T_raw : T_floor;
  const RecloseDensityPoint xp = reclose_density_point(table, rho_safe);
  const RecloseTemperaturePoint yp = reclose_temperature_point(table, T);
  return {T,
          reclose_energy_at(table, rho_safe, xp, T, yp),
          reclose_pressure_at(table, rho_safe, xp, T, yp),
          reclose_max(reclose_cv_at(table, rho_safe, xp, T, yp), 0.0)};
}

}  // namespace tenryu::materials
