#include "materials/eos_rho_e_table.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "core/error.hpp"
#include "materials/eos_table.hpp"

namespace tenryu::materials {
namespace {

constexpr int kDefaultNE = 200;
constexpr double kCvFloor = 1.0e-3;
constexpr double kCvMax = 1.0e30;
constexpr double kPressureFloor = 1.0e-30;
constexpr double kSoundSpeed2Floor = 1.0e-30;

struct AxisBracket {
  std::size_t lo = 0;
  std::size_t hi = 0;
  double w = 0.0;
  double x = 0.0;
};

struct FieldEval {
  double value = 0.0;
  double d_dlog_rho = 0.0;
  double d_dlog_e = 0.0;
};

struct SplineAxisEval {
  double A[2] = {1.0, 0.0};
  double B[2] = {0.0, 0.0};
  double dA[2] = {0.0, 0.0};
  double dB[2] = {0.0, 0.0};
};

AxisBracket bracket_log_axis(const std::vector<double>& grid, const double x_raw) {
  TENRYU_ASSERT(!grid.empty(), "EOSRhoETable axis grid must be non-empty");

  AxisBracket out{};
  out.x = std::clamp(x_raw, grid.front(), grid.back());
  if (grid.size() == 1) {
    return out;
  }

  if (out.x <= grid.front()) {
    out.lo = 0;
    out.hi = 1;
    out.w = 0.0;
    return out;
  }
  if (out.x >= grid.back()) {
    out.lo = grid.size() - 2;
    out.hi = grid.size() - 1;
    out.w = 1.0;
    return out;
  }

  const auto it = std::upper_bound(grid.begin(), grid.end(), out.x);
  out.hi = static_cast<std::size_t>(it - grid.begin());
  out.lo = out.hi - 1;
  const double x0 = grid[out.lo];
  const double x1 = grid[out.hi];
  out.w = (x1 > x0) ? ((out.x - x0) / (x1 - x0)) : 0.0;
  return out;
}

double axis_to_physical(const bool use_log_grid, const double value) {
  return use_log_grid ? std::exp(value) : value;
}

double physical_to_axis(const bool use_log_grid,
                        const double value,
                        const double axis_floor) {
  if (use_log_grid) {
    return (std::isfinite(value) && value > 0.0) ? std::log(value) : axis_floor;
  }
  return std::isfinite(value) ? value : axis_floor;
}

double solve_temperature_from_energy_bisection(const EOSTable& rhoT_table,
                                               const double rho,
                                               const double e_target) {
  const double T_min = rhoT_table.T_grid_eV.front();
  const double T_max = rhoT_table.T_grid_eV.back();
  const double e_min = rhoT_table.energy(rho, T_min);
  const double e_max = rhoT_table.energy(rho, T_max);

  TENRYU_ASSERT(e_max >= e_min,
                "EOSRhoETable requires monotone non-decreasing e(rho,T)");

  if (e_target <= e_min) {
    return T_min;
  }
  if (e_target >= e_max) {
    return T_max;
  }

  double logT_lo = rhoT_table.log_T_grid.front();
  double logT_hi = rhoT_table.log_T_grid.back();
  for (int iter = 0; iter < 48; ++iter) {
    const double logT_mid = 0.5 * (logT_lo + logT_hi);
    const double T_mid = std::exp(logT_mid);
    const double e_mid = rhoT_table.energy(rho, T_mid);
    if (e_mid <= e_target) {
      logT_lo = logT_mid;
    } else {
      logT_hi = logT_mid;
    }
  }

  return std::exp(0.5 * (logT_lo + logT_hi));
}

std::vector<double> natural_cubic_second_derivatives(const std::vector<double>& grid,
                                                     const std::vector<double>& values) {
  TENRYU_ASSERT(grid.size() == values.size(),
                "Natural cubic spline requires matching grid/value sizes");

  const std::size_t n = grid.size();
  std::vector<double> second(n, 0.0);
  if (n < 3) {
    return second;
  }

  std::vector<double> work(n, 0.0);
  for (std::size_t i = 1; i + 1 < n; ++i) {
    const double h_lo = grid[i] - grid[i - 1];
    const double h_hi = grid[i + 1] - grid[i];
    TENRYU_ASSERT(h_lo > 0.0 && h_hi > 0.0,
                  "Natural cubic spline requires strictly increasing grid");
    const double span = grid[i + 1] - grid[i - 1];
    const double sig = h_lo / span;
    const double p = sig * second[i - 1] + 2.0;
    second[i] = (sig - 1.0) / p;
    const double slope_lo = (values[i] - values[i - 1]) / h_lo;
    const double slope_hi = (values[i + 1] - values[i]) / h_hi;
    work[i] = (6.0 * (slope_hi - slope_lo) / span - sig * work[i - 1]) / p;
  }

  for (std::size_t k = n - 2; k > 0; --k) {
    second[k] = second[k] * second[k + 1] + work[k];
  }
  return second;
}

std::vector<double> compute_row_second_derivatives(const std::vector<double>& x,
                                                   const std::vector<double>& field,
                                                   const std::size_t nx,
                                                   const std::size_t ny) {
  TENRYU_ASSERT(x.size() == nx, "Row spline derivative x-grid size mismatch");
  TENRYU_ASSERT(field.size() == nx * ny,
                "Row spline derivative field size mismatch");

  std::vector<double> out(field.size(), 0.0);
  std::vector<double> row(nx, 0.0);
  for (std::size_t j = 0; j < ny; ++j) {
    const std::size_t base = j * nx;
    for (std::size_t i = 0; i < nx; ++i) {
      row[i] = field[base + i];
    }
    const auto row_second = natural_cubic_second_derivatives(x, row);
    for (std::size_t i = 0; i < nx; ++i) {
      out[base + i] = row_second[i];
    }
  }
  return out;
}

std::vector<double> compute_column_second_derivatives(const std::vector<double>& y,
                                                      const std::vector<double>& field,
                                                      const std::size_t nx,
                                                      const std::size_t ny) {
  TENRYU_ASSERT(y.size() == ny, "Column spline derivative y-grid size mismatch");
  TENRYU_ASSERT(field.size() == nx * ny,
                "Column spline derivative field size mismatch");

  std::vector<double> out(field.size(), 0.0);
  std::vector<double> col(ny, 0.0);
  for (std::size_t i = 0; i < nx; ++i) {
    for (std::size_t j = 0; j < ny; ++j) {
      col[j] = field[j * nx + i];
    }
    const auto col_second = natural_cubic_second_derivatives(y, col);
    for (std::size_t j = 0; j < ny; ++j) {
      out[j * nx + i] = col_second[j];
    }
  }
  return out;
}

SplineAxisEval build_spline_axis_eval(const std::vector<double>& grid,
                                      const AxisBracket& bracket) {
  SplineAxisEval out{};
  if (grid.size() < 2 || bracket.lo == bracket.hi) {
    return out;
  }

  const double x0 = grid[bracket.lo];
  const double x1 = grid[bracket.hi];
  const double h = std::max(x1 - x0, 1.0e-30);
  const double u = std::clamp((bracket.x - x0) / h, 0.0, 1.0);
  const double one_minus_u = 1.0 - u;

  out.A[0] = one_minus_u;
  out.A[1] = u;
  out.B[0] = (h * h / 6.0) * (one_minus_u * one_minus_u * one_minus_u - one_minus_u);
  out.B[1] = (h * h / 6.0) * (u * u * u - u);
  out.dA[0] = -1.0 / h;
  out.dA[1] = 1.0 / h;
  out.dB[0] = (h / 6.0) * (1.0 - 3.0 * one_minus_u * one_minus_u);
  out.dB[1] = (h / 6.0) * (3.0 * u * u - 1.0);
  return out;
}

FieldEval natural_spline_eval_with_log_derivatives(
    const EOSRhoETable& table,
    const std::vector<double>& field,
    const std::vector<double>& field_xx,
    const std::vector<double>& field_yy,
    const std::vector<double>& field_xxyy,
    const double rho,
    const double e) {
  TENRYU_ASSERT(!table.empty(), "EOSRhoETable evaluation requires a non-empty table");
  TENRYU_ASSERT(field.size() == table.n_rho() * table.n_e(),
                "EOSRhoETable field size mismatch");
  TENRYU_ASSERT(field_xx.size() == field.size() && field_yy.size() == field.size() &&
                    field_xxyy.size() == field.size(),
                "EOSRhoETable spline derivative field size mismatch");

  const double rho_safe = (std::isfinite(rho) && rho > 0.0)
                              ? rho
                              : axis_to_physical(table.use_log_grid(),
                                                 table.log_rho_grid().front());
  const double e_safe = (std::isfinite(e) && e > 0.0)
                            ? e
                            : axis_to_physical(table.use_log_grid(),
                                               table.log_e_grid().front());
  const AxisBracket rb =
      bracket_log_axis(table.log_rho_grid(),
                       physical_to_axis(table.use_log_grid(),
                                        rho_safe,
                                        table.log_rho_grid().front()));
  const AxisBracket eb =
      bracket_log_axis(table.log_e_grid(),
                       physical_to_axis(table.use_log_grid(),
                                        e_safe,
                                        table.log_e_grid().front()));
  const SplineAxisEval x_eval = build_spline_axis_eval(table.log_rho_grid(), rb);
  const SplineAxisEval y_eval = build_spline_axis_eval(table.log_e_grid(), eb);
  const std::size_t ii[2] = {rb.lo, rb.hi};
  const std::size_t jj[2] = {eb.lo, eb.hi};

  FieldEval out{};
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      const std::size_t idx = table.flat_index(ii[a], jj[b]);
      const double q = field[idx];
      const double q_xx = field_xx[idx];
      const double q_yy = field_yy[idx];
      const double q_xxyy = field_xxyy[idx];

      out.value += x_eval.A[a] * y_eval.A[b] * q +
                   x_eval.B[a] * y_eval.A[b] * q_xx +
                   x_eval.A[a] * y_eval.B[b] * q_yy +
                   x_eval.B[a] * y_eval.B[b] * q_xxyy;
      out.d_dlog_rho += x_eval.dA[a] * y_eval.A[b] * q +
                        x_eval.dB[a] * y_eval.A[b] * q_xx +
                        x_eval.dA[a] * y_eval.B[b] * q_yy +
                        x_eval.dB[a] * y_eval.B[b] * q_xxyy;
      out.d_dlog_e += x_eval.A[a] * y_eval.dA[b] * q +
                      x_eval.B[a] * y_eval.dA[b] * q_xx +
                      x_eval.A[a] * y_eval.dB[b] * q_yy +
                      x_eval.B[a] * y_eval.dB[b] * q_xxyy;
    }
  }
  return out;
}

double clamp_positive_floor(const double value, const double floor_value) {
  if (std::isfinite(value) && value > floor_value) {
    return value;
  }
  return floor_value;
}

double safe_cv_from_derivatives(const double dTde, const double dPde) {
  if (!(std::isfinite(dTde) && dTde > 0.0) ||
      !(std::isfinite(dPde) && dPde > 0.0)) {
    return kCvFloor;
  }
  return std::clamp(1.0 / dTde, kCvFloor, kCvMax);
}

}  // namespace

void EOSRhoETable::build_from_rhoT_table(const EOSTable& rhoT_table,
                                         const bool use_linear_grid) {
  TENRYU_ASSERT(!rhoT_table.empty(),
                "EOSRhoETable::build_from_rhoT_table requires a non-empty rho-T table");
  TENRYU_ASSERT(!rhoT_table.log_rho_grid.empty() && !rhoT_table.log_T_grid.empty(),
                "EOSRhoETable::build_from_rhoT_table requires finalized rho-T table");

  use_log_grid_ = !use_linear_grid;
  n_rho_ = static_cast<int>(rhoT_table.n_rho());
  n_e_ = kDefaultNE;
  TENRYU_ASSERT(n_rho_ > 0 && n_e_ > 1,
                "EOSRhoETable::build_from_rhoT_table requires positive grid sizes");

  log_rho_grid_ = use_log_grid_ ? rhoT_table.log_rho_grid : rhoT_table.rho_grid;
  log_e_grid_.assign(static_cast<std::size_t>(n_e_), 0.0);
  P_table_.assign(static_cast<std::size_t>(n_rho_) * static_cast<std::size_t>(n_e_), 0.0);
  T_table_.assign(static_cast<std::size_t>(n_rho_) * static_cast<std::size_t>(n_e_), 0.0);

  double e_min_global = std::numeric_limits<double>::max();
  double e_max_global = 0.0;
  const std::size_t j_T_last = rhoT_table.n_T() - 1;
  for (std::size_t i = 0; i < rhoT_table.n_rho(); ++i) {
    const double e_lo = rhoT_table.e_table[rhoT_table.flat_index(i, 0)];
    const double e_hi = rhoT_table.e_table[rhoT_table.flat_index(i, j_T_last)];
    TENRYU_ASSERT(std::isfinite(e_lo) && std::isfinite(e_hi) && e_lo > 0.0 && e_hi > 0.0,
                  "EOSRhoETable requires positive finite total energy over the source table");
    TENRYU_ASSERT(e_hi >= e_lo,
                  "EOSRhoETable requires monotone non-decreasing row energy in T");
    e_min_global = std::min(e_min_global, e_lo);
    e_max_global = std::max(e_max_global, e_hi);
  }

  TENRYU_ASSERT(e_max_global > e_min_global,
                "EOSRhoETable requires a non-degenerate global energy range");

  if (use_log_grid_) {
    const double log_e_min = std::log(e_min_global);
    const double log_e_max = std::log(e_max_global);
    const double dlog_e =
        (log_e_max - log_e_min) / static_cast<double>(std::max(n_e_ - 1, 1));
    for (int j = 0; j < n_e_; ++j) {
      log_e_grid_[static_cast<std::size_t>(j)] =
          log_e_min + dlog_e * static_cast<double>(j);
    }
  } else {
    const double de = (e_max_global - e_min_global) / static_cast<double>(std::max(n_e_ - 1, 1));
    for (int j = 0; j < n_e_; ++j) {
      log_e_grid_[static_cast<std::size_t>(j)] = e_min_global + de * static_cast<double>(j);
    }
  }

  for (std::size_t i = 0; i < rhoT_table.n_rho(); ++i) {
    const double rho = rhoT_table.rho_grid[i];
    for (int j = 0; j < n_e_; ++j) {
      const double e_target = use_log_grid_
                                  ? std::exp(log_e_grid_[static_cast<std::size_t>(j)])
                                  : log_e_grid_[static_cast<std::size_t>(j)];
      const double T = solve_temperature_from_energy_bisection(rhoT_table, rho, e_target);
      const std::size_t idx = flat_index(i, static_cast<std::size_t>(j));
      T_table_[idx] = T;
      P_table_[idx] = rhoT_table.pressure(rho, T);
    }
  }

  const std::size_t n_rho = static_cast<std::size_t>(n_rho_);
  const std::size_t n_e = static_cast<std::size_t>(n_e_);
  P_xx_ = compute_row_second_derivatives(log_rho_grid_, P_table_, n_rho, n_e);
  P_yy_ = compute_column_second_derivatives(log_e_grid_, P_table_, n_rho, n_e);
  P_xxyy_ = compute_column_second_derivatives(log_e_grid_, P_xx_, n_rho, n_e);

  T_xx_ = compute_row_second_derivatives(log_rho_grid_, T_table_, n_rho, n_e);
  T_yy_ = compute_column_second_derivatives(log_e_grid_, T_table_, n_rho, n_e);
  T_xxyy_ = compute_column_second_derivatives(log_e_grid_, T_xx_, n_rho, n_e);
}

EOSRhoETable::Result EOSRhoETable::eval(const double rho, const double e) const {
  TENRYU_ASSERT(!empty(), "EOSRhoETable::eval requires a non-empty table");

  const FieldEval p_eval =
      natural_spline_eval_with_log_derivatives(*this, P_table_, P_xx_, P_yy_, P_xxyy_, rho, e);
  const FieldEval t_eval =
      natural_spline_eval_with_log_derivatives(*this, T_table_, T_xx_, T_yy_, T_xxyy_, rho, e);

  const double rho_min = axis_to_physical(use_log_grid_, log_rho_grid_.front());
  const double rho_max = axis_to_physical(use_log_grid_, log_rho_grid_.back());
  const double e_min = axis_to_physical(use_log_grid_, log_e_grid_.front());
  const double e_max = axis_to_physical(use_log_grid_, log_e_grid_.back());
  const double rho_safe = (std::isfinite(rho) && rho > 0.0) ? rho : rho_min;
  const double e_safe = (std::isfinite(e) && e > 0.0) ? e : e_min;
  const double rho_clamped = std::clamp(rho_safe, rho_min, rho_max);
  const double e_clamped = std::clamp(e_safe, e_min, e_max);

  const double dPdrho =
      use_log_grid_ ? (p_eval.d_dlog_rho / rho_clamped) : p_eval.d_dlog_rho;
  const double dPde = use_log_grid_ ? (p_eval.d_dlog_e / e_clamped) : p_eval.d_dlog_e;
  const double dTde = use_log_grid_ ? (t_eval.d_dlog_e / e_clamped) : t_eval.d_dlog_e;

  Result out{};
  out.P = clamp_positive_floor(p_eval.value, kPressureFloor);
  out.T = (std::isfinite(t_eval.value) && t_eval.value > 0.0) ? t_eval.value : 0.0;
  out.cv = safe_cv_from_derivatives(dTde, dPde);
  out.cs2 = clamp_positive_floor(
      dPdrho + (out.P / (rho_clamped * rho_clamped)) * dPde, kSoundSpeed2Floor);
  return out;
}

}  // namespace tenryu::materials
