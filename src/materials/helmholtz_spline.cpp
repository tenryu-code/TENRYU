#include "materials/helmholtz_spline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"

namespace tenryu::materials {
namespace {

struct ScalarHermiteEval {
  double q = 0.0;
  double q_x = 0.0;
  double q_y = 0.0;
  double q_xx = 0.0;
  double q_yy = 0.0;
  double q_xy = 0.0;
};

struct Hermite1DEval {
  double q = 0.0;
  double q_x = 0.0;
  double q_xx = 0.0;
};

struct ErrorStats {
  double max_rel = 0.0;
  double rho_at_max = 0.0;
  double T_at_max = 0.0;
  bool initialized = false;

  void update(const double rel, const double rho, const double T) {
    if (!initialized || rel > max_rel) {
      max_rel = rel;
      rho_at_max = rho;
      T_at_max = T;
      initialized = true;
    }
  }
};

struct ContributionStats {
  std::vector<double> samples;
  double max_value = 0.0;
  double rho_at_max = 0.0;
  double T_at_max = 0.0;
  double Pi_at_max = 0.0;
  double Pe_at_max = 0.0;
  double Ptot_at_max = 0.0;
  bool initialized = false;

  void update(const double value,
              const double rho,
              const double T,
              const double Pi,
              const double Pe,
              const double Ptot) {
    samples.push_back(value);
    if (!initialized || value > max_value) {
      max_value = value;
      rho_at_max = rho;
      T_at_max = T;
      Pi_at_max = Pi;
      Pe_at_max = Pe;
      Ptot_at_max = Ptot;
      initialized = true;
    }
  }
};

struct TemperatureInversionStats {
  ErrorStats all{};
  ErrorStats low{};
  std::size_t bracket_failures = 0;
  std::size_t convergence_failures = 0;
};

struct HostTemperatureInversion {
  double T = 0.0;
  bool bracketed = false;
  bool converged = false;
};

[[nodiscard]] inline int signum(const double x) {
  return (x > 0.0) - (x < 0.0);
}

[[nodiscard]] std::size_t nearest_index(const std::vector<double>& grid, const double target) {
  TENRYU_ASSERT(!grid.empty(), "nearest_index requires non-empty grid");
  const auto it = std::lower_bound(grid.begin(), grid.end(), target);
  if (it == grid.begin()) {
    return 0;
  }
  if (it == grid.end()) {
    return grid.size() - 1;
  }
  const std::size_t hi = static_cast<std::size_t>(it - grid.begin());
  const std::size_t lo = hi - 1;
  return (std::fabs(grid[hi] - target) < std::fabs(grid[lo] - target)) ? hi : lo;
}

std::vector<double> monotone_hermite_derivatives_1d(const std::vector<double>& x,
                                                    const std::vector<double>& y) {
  const std::size_t n = x.size();
  TENRYU_ASSERT(y.size() == n, "Hermite spline: x/y size mismatch");
  std::vector<double> d(n, 0.0);
  if (n < 2) {
    return d;
  }
  if (n == 2) {
    const double h = x[1] - x[0];
    TENRYU_ASSERT(h > 0.0, "Hermite spline requires strictly increasing grid");
    const double sec = (y[1] - y[0]) / h;
    d[0] = sec;
    d[1] = sec;
    return d;
  }

  std::vector<double> h(n - 1, 0.0);
  std::vector<double> delta(n - 1, 0.0);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    h[i] = x[i + 1] - x[i];
    TENRYU_ASSERT(h[i] > 0.0, "Hermite spline requires strictly increasing grid");
    delta[i] = (y[i + 1] - y[i]) / h[i];
  }

  d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
  if (signum(d[0]) != signum(delta[0])) {
    d[0] = 0.0;
  } else if (signum(delta[0]) != signum(delta[1]) && std::fabs(d[0]) > 3.0 * std::fabs(delta[0])) {
    d[0] = 3.0 * delta[0];
  }

  for (std::size_t i = 1; i + 1 < n; ++i) {
    if (delta[i - 1] == 0.0 || delta[i] == 0.0 ||
        signum(delta[i - 1]) != signum(delta[i])) {
      d[i] = 0.0;
    } else {
      d[i] = 0.5 * (delta[i - 1] + delta[i]);
    }
  }

  d[n - 1] =
      ((2.0 * h[n - 2] + h[n - 3]) * delta[n - 2] - h[n - 2] * delta[n - 3]) /
      (h[n - 2] + h[n - 3]);
  if (signum(d[n - 1]) != signum(delta[n - 2])) {
    d[n - 1] = 0.0;
  } else if (signum(delta[n - 2]) != signum(delta[n - 3]) &&
             std::fabs(d[n - 1]) > 3.0 * std::fabs(delta[n - 2])) {
    d[n - 1] = 3.0 * delta[n - 2];
  }

  for (std::size_t i = 0; i + 1 < n; ++i) {
    if (delta[i] == 0.0) {
      d[i] = 0.0;
      d[i + 1] = 0.0;
      continue;
    }
    double alpha = d[i] / delta[i];
    double beta = d[i + 1] / delta[i];
    if (alpha < 0.0) {
      d[i] = 0.0;
      alpha = 0.0;
    }
    if (beta < 0.0) {
      d[i + 1] = 0.0;
      beta = 0.0;
    }
    const double norm = alpha * alpha + beta * beta;
    if (norm > 9.0) {
      const double tau = 3.0 / std::sqrt(norm);
      d[i] = tau * alpha * delta[i];
      d[i + 1] = tau * beta * delta[i];
    }
  }

  return d;
}

std::vector<double> compute_row_monotone_derivatives(const std::vector<double>& x,
                                                     const std::vector<double>& field,
                                                     const std::size_t nx,
                                                     const std::size_t ny) {
  TENRYU_ASSERT(field.size() == nx * ny,
                "Hermite spline row derivative field size mismatch");
  std::vector<double> out(field.size(), 0.0);
  std::vector<double> row(nx, 0.0);
  for (std::size_t j = 0; j < ny; ++j) {
    const std::size_t base = j * nx;
    for (std::size_t i = 0; i < nx; ++i) {
      row[i] = field[base + i];
    }
    const auto d = monotone_hermite_derivatives_1d(x, row);
    for (std::size_t i = 0; i < nx; ++i) {
      out[base + i] = d[i];
    }
  }
  return out;
}

std::vector<double> compute_column_monotone_derivatives(const std::vector<double>& y,
                                                        const std::vector<double>& field,
                                                        const std::size_t nx,
                                                        const std::size_t ny) {
  TENRYU_ASSERT(field.size() == nx * ny,
                "Hermite spline column derivative field size mismatch");
  std::vector<double> out(field.size(), 0.0);
  std::vector<double> col(ny, 0.0);
  for (std::size_t i = 0; i < nx; ++i) {
    for (std::size_t j = 0; j < ny; ++j) {
      col[j] = field[j * nx + i];
    }
    const auto d = monotone_hermite_derivatives_1d(y, col);
    for (std::size_t j = 0; j < ny; ++j) {
      out[j * nx + i] = d[j];
    }
  }
  return out;
}

[[nodiscard]] Hermite1DEval eval_hermite_1d(const std::vector<double>& grid,
                                            const std::vector<double>& values,
                                            const std::vector<double>& derivs,
                                            const std::size_t i0,
                                            const double x) {
  TENRYU_ASSERT(i0 + 1 < grid.size(), "Hermite 1D evaluator index out of bounds");
  const double x0 = grid[i0];
  const double x1 = grid[i0 + 1];
  const double h = std::max(x1 - x0, 1.0e-30);
  const double t = std::clamp((x - x0) / h, 0.0, 1.0);

  const double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
  const double h10 = t * t * t - 2.0 * t * t + t;
  const double h01 = -2.0 * t * t * t + 3.0 * t * t;
  const double h11 = t * t * t - t * t;

  const double dh00 = 6.0 * t * t - 6.0 * t;
  const double dh10 = 3.0 * t * t - 4.0 * t + 1.0;
  const double dh01 = -6.0 * t * t + 6.0 * t;
  const double dh11 = 3.0 * t * t - 2.0 * t;

  const double ddh00 = 12.0 * t - 6.0;
  const double ddh10 = 6.0 * t - 4.0;
  const double ddh01 = -12.0 * t + 6.0;
  const double ddh11 = 6.0 * t - 2.0;

  Hermite1DEval out{};
  out.q = h00 * values[i0] + h10 * h * derivs[i0] + h01 * values[i0 + 1] +
          h11 * h * derivs[i0 + 1];
  out.q_x = (dh00 * values[i0] + dh10 * h * derivs[i0] + dh01 * values[i0 + 1] +
             dh11 * h * derivs[i0 + 1]) /
            h;
  out.q_xx = (ddh00 * values[i0] + ddh10 * h * derivs[i0] + ddh01 * values[i0 + 1] +
              ddh11 * h * derivs[i0 + 1]) /
             (h * h);
  return out;
}

[[nodiscard]] ScalarHermiteEval eval_scalar_host(const HelmholtzSpline1T& spline,
                                                 const std::vector<double>& nodes,
                                                 const std::vector<double>& dx_nodes,
                                                 const std::vector<double>& dy_nodes,
                                                 const std::vector<double>& dxdy_nodes,
                                                 const double log_rho_raw,
                                                 const double log_T_raw) {
  ScalarHermiteEval out{};
  const std::size_t nx = spline.n_rho();
  const std::size_t ny = spline.n_T();
  if (nx < 2 || ny < 2 || nodes.empty()) {
    return out;
  }

  const double log_rho =
      std::clamp(log_rho_raw, spline.log_rho_grid.front(), spline.log_rho_grid.back());
  const double log_T =
      std::clamp(log_T_raw, spline.log_T_grid.front(), spline.log_T_grid.back());
  const auto it_x = std::upper_bound(spline.log_rho_grid.begin(), spline.log_rho_grid.end(),
                                     log_rho);
  const auto it_y = std::upper_bound(spline.log_T_grid.begin(), spline.log_T_grid.end(), log_T);
  const std::size_t i0 =
      (it_x == spline.log_rho_grid.begin())
          ? 0
          : std::min<std::size_t>(static_cast<std::size_t>(it_x - spline.log_rho_grid.begin() - 1),
                                  nx - 2);
  const std::size_t j0 =
      (it_y == spline.log_T_grid.begin())
          ? 0
          : std::min<std::size_t>(static_cast<std::size_t>(it_y - spline.log_T_grid.begin() - 1),
                                  ny - 2);
  const std::size_t i1 = i0 + 1;
  const std::size_t j1 = j0 + 1;

  const double x0 = spline.log_rho_grid[i0];
  const double x1 = spline.log_rho_grid[i1];
  const double y0 = spline.log_T_grid[j0];
  const double y1 = spline.log_T_grid[j1];
  const double hx = std::max(x1 - x0, 1.0e-30);
  const double hy = std::max(y1 - y0, 1.0e-30);
  const double u = std::clamp((log_rho - x0) / hx, 0.0, 1.0);
  const double v = std::clamp((log_T - y0) / hy, 0.0, 1.0);

  const double hx_val[2] = {2.0 * u * u * u - 3.0 * u * u + 1.0,
                            -2.0 * u * u * u + 3.0 * u * u};
  const double hx_der[2] = {hx * (u * u * u - 2.0 * u * u + u),
                            hx * (u * u * u - u * u)};
  const double hy_val[2] = {2.0 * v * v * v - 3.0 * v * v + 1.0,
                            -2.0 * v * v * v + 3.0 * v * v};
  const double hy_der[2] = {hy * (v * v * v - 2.0 * v * v + v),
                            hy * (v * v * v - v * v)};

  const double dhx_val_dx[2] = {(6.0 * u * u - 6.0 * u) / hx,
                                (-6.0 * u * u + 6.0 * u) / hx};
  const double dhx_der_dx[2] = {3.0 * u * u - 4.0 * u + 1.0,
                                3.0 * u * u - 2.0 * u};
  const double dhy_val_dy[2] = {(6.0 * v * v - 6.0 * v) / hy,
                                (-6.0 * v * v + 6.0 * v) / hy};
  const double dhy_der_dy[2] = {3.0 * v * v - 4.0 * v + 1.0,
                                3.0 * v * v - 2.0 * v};

  const double ddhx_val_dxx[2] = {(12.0 * u - 6.0) / (hx * hx),
                                  (-12.0 * u + 6.0) / (hx * hx)};
  const double ddhx_der_dxx[2] = {(6.0 * u - 4.0) / hx, (6.0 * u - 2.0) / hx};
  const double ddhy_val_dyy[2] = {(12.0 * v - 6.0) / (hy * hy),
                                  (-12.0 * v + 6.0) / (hy * hy)};
  const double ddhy_der_dyy[2] = {(6.0 * v - 4.0) / hy, (6.0 * v - 2.0) / hy};

  const std::size_t ii[2] = {i0, i1};
  const std::size_t jj[2] = {j0, j1};
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      const std::size_t idx = spline.flat_index(ii[a], jj[b]);
      const double q = nodes[idx];
      const double qx = dx_nodes[idx];
      const double qy = dy_nodes[idx];
      const double qxy = dxdy_nodes[idx];

      out.q += hx_val[a] * hy_val[b] * q + hx_der[a] * hy_val[b] * qx +
               hx_val[a] * hy_der[b] * qy + hx_der[a] * hy_der[b] * qxy;
      out.q_x += dhx_val_dx[a] * hy_val[b] * q + dhx_der_dx[a] * hy_val[b] * qx +
                 dhx_val_dx[a] * hy_der[b] * qy + dhx_der_dx[a] * hy_der[b] * qxy;
      out.q_y += hx_val[a] * dhy_val_dy[b] * q + hx_der[a] * dhy_val_dy[b] * qx +
                 hx_val[a] * dhy_der_dy[b] * qy + hx_der[a] * dhy_der_dy[b] * qxy;
      out.q_xx += ddhx_val_dxx[a] * hy_val[b] * q +
                  ddhx_der_dxx[a] * hy_val[b] * qx +
                  ddhx_val_dxx[a] * hy_der[b] * qy +
                  ddhx_der_dxx[a] * hy_der[b] * qxy;
      out.q_yy += hx_val[a] * ddhy_val_dyy[b] * q +
                  hx_der[a] * ddhy_val_dyy[b] * qx +
                  hx_val[a] * ddhy_der_dyy[b] * qy +
                  hx_der[a] * ddhy_der_dyy[b] * qxy;
      out.q_xy += dhx_val_dx[a] * dhy_val_dy[b] * q +
                  dhx_der_dx[a] * dhy_val_dy[b] * qx +
                  dhx_val_dx[a] * dhy_der_dy[b] * qy +
                  dhx_der_dx[a] * dhy_der_dy[b] * qxy;
    }
  }

  return out;
}

[[nodiscard]] HostTemperatureInversion host_T_from_e(const HelmholtzSpline1T& spline,
                                                     const double rho,
                                                     const double e_target) {
  HostTemperatureInversion out{};
  if (!(rho > 0.0) || spline.n_rho() < 2 || spline.n_T() < 2) {
    return out;
  }

  const double log_rho =
      std::clamp(std::log(rho), spline.log_rho_grid.front(), spline.log_rho_grid.back());
  double y_lo = spline.log_T_grid.front();
  double y_hi = spline.log_T_grid.back();
  double e_lo = 0.0;
  double e_hi = 0.0;

  double y_prev = spline.log_T_grid.front();
  double e_prev =
      eval_scalar_host(spline,
                       spline.e_nodes,
                       spline.e_dx_nodes,
                       spline.e_dy_nodes,
                       spline.e_dxdy_nodes,
                       log_rho,
                       y_prev)
          .q;
  if (e_target <= e_prev) {
    out.T = std::exp(y_prev);
    return out;
  }

  for (std::size_t j = 1; j < spline.n_T(); ++j) {
    const double y_curr = spline.log_T_grid[j];
    const double e_curr =
        eval_scalar_host(spline,
                         spline.e_nodes,
                         spline.e_dx_nodes,
                         spline.e_dy_nodes,
                         spline.e_dxdy_nodes,
                         log_rho,
                         y_curr)
            .q;
    if ((e_target >= e_prev && e_target <= e_curr) ||
        (e_target <= e_prev && e_target >= e_curr)) {
      y_lo = y_prev;
      y_hi = y_curr;
      e_lo = e_prev;
      e_hi = e_curr;
      out.bracketed = true;
      break;
    }
    y_prev = y_curr;
    e_prev = e_curr;
  }

  if (!out.bracketed) {
    out.T = std::exp(y_prev);
    return out;
  }
  if (e_hi < e_lo) {
    std::swap(y_lo, y_hi);
    std::swap(e_lo, e_hi);
  }
  if (e_target <= e_lo) {
    out.T = std::exp(y_lo);
    out.converged = true;
    return out;
  }
  if (e_target >= e_hi) {
    out.T = std::exp(y_hi);
    out.converged = true;
    return out;
  }

  double y = y_lo + (e_target - e_lo) * (y_hi - y_lo) / std::max(e_hi - e_lo, 1.0e-30);
  for (int iter = 0; iter < 32; ++iter) {
    const ScalarHermiteEval e_eval =
        eval_scalar_host(spline,
                         spline.e_nodes,
                         spline.e_dx_nodes,
                         spline.e_dy_nodes,
                         spline.e_dxdy_nodes,
                         log_rho,
                         y);
    const double g = e_eval.q - e_target;
    if (std::fabs(g) <= 1.0e-10 * std::max(std::fabs(e_target), 1.0)) {
      out.T = std::exp(y);
      out.converged = true;
      return out;
    }
    if (g > 0.0) {
      y_hi = y;
    } else {
      y_lo = y;
    }
    const double gp = e_eval.q_y;
    double y_new = 0.5 * (y_lo + y_hi);
    if (std::fabs(gp) > 1.0e-30 && std::isfinite(gp)) {
      const double y_trial = y - g / gp;
      if (y_trial > y_lo && y_trial < y_hi) {
        y_new = y_trial;
      }
    }
    y = y_new;
  }
  out.T = std::exp(y);
  return out;
}

std::string format_error_stat(const ErrorStats& s) {
  std::ostringstream oss;
  if (!s.initialized) {
    oss << "n/a";
  } else {
    oss << s.max_rel << " @ rho=" << s.rho_at_max << ", T=" << s.T_at_max << " eV";
  }
  return oss.str();
}

void log_grid_base_report(const HelmholtzSpline1T& spline,
                          const EOSTable& table,
                          const std::string_view label) {
  if (spline.log_rho_grid.size() < 2 || spline.log_T_grid.size() < 2 ||
      table.rho_grid.size() < 2 || table.T_grid_eV.size() < 2) {
    return;
  }

  const double rho_delta = spline.log_rho_grid[1] - spline.log_rho_grid[0];
  const double T_delta = spline.log_T_grid[1] - spline.log_T_grid[0];
  const double rho_ratio = table.rho_grid[1] / table.rho_grid[0];
  const double T_ratio = table.T_grid_eV[1] / table.T_grid_eV[0];

  std::ostringstream oss;
  oss << "HelmholtzSpline[" << label << "] log-grid base check: "
      << "dlogrho=" << rho_delta
      << ", ln(rho1/rho0)=" << std::log(rho_ratio)
      << ", log10(rho1/rho0)=" << std::log10(rho_ratio)
      << ", dlogT=" << T_delta
      << ", ln(T1/T0)=" << std::log(T_ratio)
      << ", log10(T1/T0)=" << std::log10(T_ratio);
  core::log_info(oss.str());
}

void log_row_consistency_report(const HelmholtzSpline1T& spline, const std::string_view label) {
  const std::size_t nx = spline.n_rho();
  const std::size_t ny = spline.n_T();
  if (nx < 2 || ny == 0) {
    return;
  }

  const std::size_t j = ny / 2;
  const double y = spline.log_T_grid[j];
  std::vector<double> row_values(nx, 0.0);
  std::vector<double> row_derivs(nx, 0.0);
  ErrorStats p_val{};
  ErrorStats p_dx{};
  ErrorStats p_dxx{};
  ErrorStats e_val{};
  ErrorStats e_dx{};
  ErrorStats e_dxx{};

  auto check_field = [&](const std::vector<double>& nodes,
                         const std::vector<double>& dx_nodes,
                         const std::vector<double>& dy_nodes,
                         const std::vector<double>& dxdy_nodes,
                         ErrorStats& val_stats,
                         ErrorStats& dx_stats,
                         ErrorStats& dxx_stats) {
    const std::size_t base = j * nx;
    for (std::size_t i = 0; i < nx; ++i) {
      row_values[i] = nodes[base + i];
      row_derivs[i] = dx_nodes[base + i];
    }
    for (std::size_t i = 0; i + 1 < nx; ++i) {
      const double x_mid = 0.5 * (spline.log_rho_grid[i] + spline.log_rho_grid[i + 1]);
      const double rho_mid = std::exp(x_mid);
      const double T_mid = std::exp(y);
      const Hermite1DEval ref =
          eval_hermite_1d(spline.log_rho_grid, row_values, row_derivs, i, x_mid);
      const ScalarHermiteEval eval =
          eval_scalar_host(spline, nodes, dx_nodes, dy_nodes, dxdy_nodes, x_mid, y);
      val_stats.update(std::fabs(eval.q - ref.q) / std::max(std::fabs(ref.q), 1.0e-30),
                       rho_mid,
                       T_mid);
      dx_stats.update(std::fabs(eval.q_x - ref.q_x) / std::max(std::fabs(ref.q_x), 1.0e-30),
                      rho_mid,
                      T_mid);
      dxx_stats.update(std::fabs(eval.q_xx - ref.q_xx) / std::max(std::fabs(ref.q_xx), 1.0e-30),
                       rho_mid,
                       T_mid);
    }
  };

  check_field(spline.P_nodes,
              spline.P_dx_nodes,
              spline.P_dy_nodes,
              spline.P_dxdy_nodes,
              p_val,
              p_dx,
              p_dxx);
  check_field(spline.e_nodes,
              spline.e_dx_nodes,
              spline.e_dy_nodes,
              spline.e_dxdy_nodes,
              e_val,
              e_dx,
              e_dxx);

  core::log_info("HelmholtzSpline[" + std::string(label) +
                 "] row-consistency (2D vs 1D monotone Hermite, center-row midpoints): "
                 "P[val,dx,dxx]=(" + format_error_stat(p_val) + ", " +
                 format_error_stat(p_dx) + ", " + format_error_stat(p_dxx) +
                 "), e[val,dx,dxx]=(" + format_error_stat(e_val) + ", " +
                 format_error_stat(e_dx) + ", " + format_error_stat(e_dxx) + ")");
}

void log_reconstruction_error_report(const HelmholtzSpline1T& spline,
                                     const EOSTable& table,
                                     const std::string_view label) {
  ErrorStats p_all{};
  ErrorStats e_all{};
  ErrorStats cv_all{};
  ErrorStats p_low{};
  ErrorStats e_low{};
  ErrorStats cv_low{};

  const std::size_t nx = table.n_rho();
  const std::size_t ny = table.n_T();
  for (std::size_t j = 0; j < ny; ++j) {
    const double T = table.T_grid_eV[j];
    const bool is_low_T = (T >= 0.1 && T <= 10.0);
    for (std::size_t i = 0; i < nx; ++i) {
      const std::size_t idx = table.flat_index(i, j);
      const double rho = table.rho_grid[i];
      const double log_rho = table.log_rho_grid[i];
      const double log_T = table.log_T_grid[j];

      const ScalarHermiteEval p_eval =
          eval_scalar_host(spline,
                           spline.P_nodes,
                           spline.P_dx_nodes,
                           spline.P_dy_nodes,
                           spline.P_dxdy_nodes,
                           log_rho,
                           log_T);
      const ScalarHermiteEval e_eval =
          eval_scalar_host(spline,
                           spline.e_nodes,
                           spline.e_dx_nodes,
                           spline.e_dy_nodes,
                           spline.e_dxdy_nodes,
                           log_rho,
                           log_T);

      const double p_ref = table.P_table[idx];
      const double e_ref = table.e_table[idx];
      const double cv_ref = table.cv_table.empty() ? 0.0 : table.cv_table[idx];
      const double p_rel = std::fabs(p_eval.q - p_ref) / std::max(std::fabs(p_ref), 1.0e-30);
      const double e_rel = std::fabs(e_eval.q - e_ref) / std::max(std::fabs(e_ref), 1.0e-30);
      const double cv_fit = std::max(e_eval.q_y / std::max(T, 1.0e-30), 0.0);

      p_all.update(p_rel, rho, T);
      e_all.update(e_rel, rho, T);
      if (is_low_T) {
        p_low.update(p_rel, rho, T);
        e_low.update(e_rel, rho, T);
      }
      if (!table.cv_table.empty()) {
        const double cv_rel =
            std::fabs(cv_fit - cv_ref) / std::max(std::fabs(cv_ref), 1.0e-30);
        cv_all.update(cv_rel, rho, T);
        if (is_low_T) {
          cv_low.update(cv_rel, rho, T);
        }
      }
    }
  }

  core::log_info("HelmholtzSpline[" + std::string(label) +
                 "] reconstruction max relerr: P=" + format_error_stat(p_all) +
                 ", e=" + format_error_stat(e_all) + ", cv=" + format_error_stat(cv_all));
  core::log_info("HelmholtzSpline[" + std::string(label) +
                 "] low-T (0.1-10 eV) max relerr: P=" + format_error_stat(p_low) +
                 ", e=" + format_error_stat(e_low) + ", cv=" + format_error_stat(cv_low));
}

void log_midpoint_compare_report(const HelmholtzSpline1T& spline,
                                 const EOSTable& table,
                                 const std::string_view label) {
  const std::size_t nx = table.n_rho();
  const std::size_t ny = table.n_T();
  if (nx < 2 || ny < 2) {
    return;
  }

  ErrorStats p_all{};
  ErrorStats e_all{};
  ErrorStats p_low{};
  ErrorStats e_low{};
  for (std::size_t j = 0; j + 1 < ny; ++j) {
    const double y_mid = 0.5 * (table.log_T_grid[j] + table.log_T_grid[j + 1]);
    const double T_mid = std::exp(y_mid);
    const bool is_low_T = (T_mid >= 0.1 && T_mid <= 10.0);
    for (std::size_t i = 0; i + 1 < nx; ++i) {
      const double x_mid = 0.5 * (table.log_rho_grid[i] + table.log_rho_grid[i + 1]);
      const double rho_mid = std::exp(x_mid);

      const ScalarHermiteEval p_eval =
          eval_scalar_host(spline,
                           spline.P_nodes,
                           spline.P_dx_nodes,
                           spline.P_dy_nodes,
                           spline.P_dxdy_nodes,
                           x_mid,
                           y_mid);
      const ScalarHermiteEval e_eval =
          eval_scalar_host(spline,
                           spline.e_nodes,
                           spline.e_dx_nodes,
                           spline.e_dy_nodes,
                           spline.e_dxdy_nodes,
                           x_mid,
                           y_mid);

      const double p_bilinear = 0.25 * (table.P_table[table.flat_index(i, j)] +
                                        table.P_table[table.flat_index(i + 1, j)] +
                                        table.P_table[table.flat_index(i, j + 1)] +
                                        table.P_table[table.flat_index(i + 1, j + 1)]);
      const double e_bilinear = 0.25 * (table.e_table[table.flat_index(i, j)] +
                                        table.e_table[table.flat_index(i + 1, j)] +
                                        table.e_table[table.flat_index(i, j + 1)] +
                                        table.e_table[table.flat_index(i + 1, j + 1)]);

      const double p_rel =
          std::fabs(p_eval.q - p_bilinear) / std::max(std::fabs(p_bilinear), 1.0e-30);
      const double e_rel =
          std::fabs(e_eval.q - e_bilinear) / std::max(std::fabs(e_bilinear), 1.0e-30);
      p_all.update(p_rel, rho_mid, T_mid);
      e_all.update(e_rel, rho_mid, T_mid);
      if (is_low_T) {
        p_low.update(p_rel, rho_mid, T_mid);
        e_low.update(e_rel, rho_mid, T_mid);
      }
    }
  }

  core::log_info("HelmholtzSpline[" + std::string(label) +
                 "] midpoint compare vs bilinear: P=" + format_error_stat(p_all) +
                 ", e=" + format_error_stat(e_all));
  core::log_info("HelmholtzSpline[" + std::string(label) +
                 "] midpoint compare vs bilinear low-T (0.1-10 eV): P=" +
                 format_error_stat(p_low) + ", e=" + format_error_stat(e_low));
}

void log_temperature_inversion_report(const HelmholtzSpline1T& spline,
                                      const EOSTable& table,
                                      const std::string_view label) {
  TemperatureInversionStats stats{};
  const std::size_t nx = table.n_rho();
  const std::size_t ny = table.n_T();
  for (std::size_t j = 0; j < ny; ++j) {
    const double T_ref = table.T_grid_eV[j];
    const bool is_low_T = (T_ref >= 0.1 && T_ref <= 10.0);
    for (std::size_t i = 0; i < nx; ++i) {
      const std::size_t idx = table.flat_index(i, j);
      const double rho = table.rho_grid[i];
      const HostTemperatureInversion inv = host_T_from_e(spline, rho, table.e_table[idx]);
      if (!inv.bracketed) {
        ++stats.bracket_failures;
      }
      if (!inv.converged) {
        ++stats.convergence_failures;
      }
      const double rel = std::fabs(inv.T - T_ref) / std::max(std::fabs(T_ref), 1.0e-30);
      stats.all.update(rel, rho, T_ref);
      if (is_low_T) {
        stats.low.update(rel, rho, T_ref);
      }
    }
  }

  core::log_info("HelmholtzSpline[" + std::string(label) +
                 "] T_from_e node inversion: relerr(all)=" + format_error_stat(stats.all) +
                 ", relerr(low-T)=" + format_error_stat(stats.low) +
                 ", bracket_failures=" + std::to_string(stats.bracket_failures) +
                 ", convergence_failures=" + std::to_string(stats.convergence_failures));
}

void log_electron_target_region_report(const HelmholtzSpline1T& spline,
                                       const EOSTable& table,
                                       const std::string_view label) {
  if (label != "electron" || table.empty()) {
    return;
  }

  constexpr double kTargetRho = 0.167;
  constexpr double kTargetT = 0.42;
  const std::size_t nx = table.n_rho();
  const std::size_t ny = table.n_T();
  const std::size_t i_near = nearest_index(table.rho_grid, kTargetRho);
  const std::size_t j_near = nearest_index(table.T_grid_eV, kTargetT);
  const std::size_t icell =
      std::min<std::size_t>((kTargetRho <= table.rho_grid.front()) ? 0
                               : (kTargetRho >= table.rho_grid.back())
                                     ? nx - 2
                                     : nearest_index(table.rho_grid, kTargetRho),
                           nx - 2);
  const std::size_t jcell =
      std::min<std::size_t>((kTargetT <= table.T_grid_eV.front()) ? 0
                               : (kTargetT >= table.T_grid_eV.back())
                                     ? ny - 2
                                     : nearest_index(table.T_grid_eV, kTargetT),
                           ny - 2);
  const std::size_t i0 =
      (table.rho_grid[icell] <= kTargetRho || icell == 0) ? std::min(icell, nx - 2) : icell - 1;
  const std::size_t j0 =
      (table.T_grid_eV[jcell] <= kTargetT || jcell == 0) ? std::min(jcell, ny - 2) : jcell - 1;

  std::ostringstream head;
  head << "HelmholtzSpline[electron] target region: target(rho=" << kTargetRho
       << ", T=" << kTargetT << " eV), nearest_node=(i=" << i_near << ", j=" << j_near
       << ", rho=" << table.rho_grid[i_near] << ", T=" << table.T_grid_eV[j_near]
       << " eV), containing_cell=[i:" << i0 << "->" << (i0 + 1) << ", j:" << j0 << "->"
       << (j0 + 1) << "]";
  core::log_info(head.str());

  const double x_mid = 0.5 * (table.log_rho_grid[i0] + table.log_rho_grid[i0 + 1]);
  const double y_mid = 0.5 * (table.log_T_grid[j0] + table.log_T_grid[j0 + 1]);
  const ScalarHermiteEval p_mid =
      eval_scalar_host(spline,
                       spline.P_nodes,
                       spline.P_dx_nodes,
                       spline.P_dy_nodes,
                       spline.P_dxdy_nodes,
                       x_mid,
                       y_mid);
  const ScalarHermiteEval e_mid =
      eval_scalar_host(spline,
                       spline.e_nodes,
                       spline.e_dx_nodes,
                       spline.e_dy_nodes,
                       spline.e_dxdy_nodes,
                       x_mid,
                       y_mid);
  const double p_bilinear = 0.25 * (table.P_table[table.flat_index(i0, j0)] +
                                    table.P_table[table.flat_index(i0 + 1, j0)] +
                                    table.P_table[table.flat_index(i0, j0 + 1)] +
                                    table.P_table[table.flat_index(i0 + 1, j0 + 1)]);
  const double e_bilinear = 0.25 * (table.e_table[table.flat_index(i0, j0)] +
                                    table.e_table[table.flat_index(i0 + 1, j0)] +
                                    table.e_table[table.flat_index(i0, j0 + 1)] +
                                    table.e_table[table.flat_index(i0 + 1, j0 + 1)]);
  std::ostringstream mid;
  mid << "HelmholtzSpline[electron] target cell midpoint: "
      << "rho_mid=" << std::exp(x_mid) << ", T_mid=" << std::exp(y_mid)
      << " eV, P_spline=" << p_mid.q << ", P_bilinear=" << p_bilinear
      << ", e_spline=" << e_mid.q << ", e_bilinear=" << e_bilinear;
  core::log_info(mid.str());

  const std::size_t i_lo = (i_near == 0) ? 0 : i_near - 1;
  const std::size_t i_hi = std::min(i_near + 1, nx - 1);
  const std::size_t j_lo = (j_near == 0) ? 0 : j_near - 1;
  const std::size_t j_hi = std::min(j_near + 1, ny - 1);
  for (std::size_t j = j_lo; j <= j_hi; ++j) {
    std::ostringstream line;
    line << "HelmholtzSpline[electron] target-neighborhood T[j=" << j << "]="
         << table.T_grid_eV[j] << " eV:";
    for (std::size_t i = i_lo; i <= i_hi; ++i) {
      const std::size_t idx = table.flat_index(i, j);
      line << " [i=" << i << ", rho=" << table.rho_grid[i] << ", P=" << table.P_table[idx]
           << ", e=" << table.e_table[idx] << ", dPx=" << spline.P_dx_nodes[idx]
           << ", dPy=" << spline.P_dy_nodes[idx] << ", dEx=" << spline.e_dx_nodes[idx]
           << ", dEy=" << spline.e_dy_nodes[idx] << "]";
    }
    core::log_info(line.str());
  }

  std::size_t nearest_cross_j = ny;
  double best_cross_dist = std::numeric_limits<double>::infinity();
  for (std::size_t j = 0; j + 1 < ny; ++j) {
    const double p0 = table.P_table[table.flat_index(i_near, j)];
    const double p1 = table.P_table[table.flat_index(i_near, j + 1)];
    if ((p0 <= 0.0 && p1 >= 0.0) || (p0 >= 0.0 && p1 <= 0.0)) {
      const double T_mid_cross = 0.5 * (table.T_grid_eV[j] + table.T_grid_eV[j + 1]);
      const double dist = std::fabs(T_mid_cross - kTargetT);
      if (dist < best_cross_dist) {
        best_cross_dist = dist;
        nearest_cross_j = j;
      }
    }
  }
  if (nearest_cross_j < ny - 1) {
    std::ostringstream cross;
    cross << "HelmholtzSpline[electron] nearest P zero-cross on rho-column i=" << i_near
          << " (rho=" << table.rho_grid[i_near] << "): T="
          << table.T_grid_eV[nearest_cross_j] << " -> " << table.T_grid_eV[nearest_cross_j + 1]
          << " eV, P=" << table.P_table[table.flat_index(i_near, nearest_cross_j)] << " -> "
          << table.P_table[table.flat_index(i_near, nearest_cross_j + 1)];
    core::log_info(cross.str());
  }
}

void log_total_pressure_contribution_report(const EOSTableTriplet& tables) {
  if (tables.ion.empty() || tables.electron.empty() || tables.total.empty()) {
    return;
  }

  ContributionStats stats{};
  const std::size_t nx = tables.total.n_rho();
  const std::size_t ny = tables.total.n_T();
  for (std::size_t j = 0; j < ny; ++j) {
    const double T = tables.total.T_grid_eV[j];
    if (T < 1.0 || T > 10.0) {
      continue;
    }
    for (std::size_t i = 0; i < nx; ++i) {
      const double rho = tables.total.rho_grid[i];
      if (rho < 1.0 || rho > 5.0) {
        continue;
      }
      const std::size_t idx = tables.total.flat_index(i, j);
      const double Pi = tables.ion.P_table[idx];
      const double Pe = tables.electron.P_table[idx];
      const double Ptot = tables.total.P_table[idx];
      const double frac = std::fabs(Pe) / std::max(std::fabs(Ptot), 1.0e-30);
      stats.update(frac, rho, T, Pi, Pe, Ptot);
    }
  }

  if (!stats.initialized) {
    return;
  }
  std::sort(stats.samples.begin(), stats.samples.end());
  const double median = stats.samples[stats.samples.size() / 2];
  std::ostringstream oss;
  oss << "HelmholtzSpline[total] raw pressure contribution in rho=[1,5] g/cc, T=[1,10] eV: "
      << "median |Pe|/|Ptot|=" << median << ", max=" << stats.max_value
      << " @ rho=" << stats.rho_at_max << ", T=" << stats.T_at_max
      << " eV with (Pi,Pe,Ptot)=(" << stats.Pi_at_max << ", " << stats.Pe_at_max << ", "
      << stats.Ptot_at_max << ")";
  core::log_info(oss.str());
}

}  // namespace

void HelmholtzSpline1T::clear() {
  log_rho_grid.clear();
  log_T_grid.clear();
  P_nodes.clear();
  P_dx_nodes.clear();
  P_dy_nodes.clear();
  P_dxdy_nodes.clear();
  e_nodes.clear();
  e_dx_nodes.clear();
  e_dy_nodes.clear();
  e_dxdy_nodes.clear();
}

void HelmholtzSpline1T::build_from_table(const EOSTable& table, const std::string_view label) {
  clear();
  if (table.empty()) {
    return;
  }

  const std::size_t nx = table.n_rho();
  const std::size_t ny = table.n_T();
  TENRYU_ASSERT(nx >= 2 && ny >= 2, "EOS spline backend requires table size >= 2x2");
  TENRYU_ASSERT(table.log_rho_grid.size() == nx && table.log_T_grid.size() == ny,
                "EOS spline backend grid size mismatch");
  TENRYU_ASSERT(table.P_table.size() == nx * ny && table.e_table.size() == nx * ny,
                "EOS spline backend table size mismatch");

  log_rho_grid = table.log_rho_grid;
  log_T_grid = table.log_T_grid;

  P_nodes = table.P_table;
  P_dx_nodes = compute_row_monotone_derivatives(log_rho_grid, P_nodes, nx, ny);
  P_dy_nodes = compute_column_monotone_derivatives(log_T_grid, P_nodes, nx, ny);
  P_dxdy_nodes = compute_column_monotone_derivatives(log_T_grid, P_dx_nodes, nx, ny);

  e_nodes = table.e_table;
  e_dx_nodes = compute_row_monotone_derivatives(log_rho_grid, e_nodes, nx, ny);
  e_dy_nodes = compute_column_monotone_derivatives(log_T_grid, e_nodes, nx, ny);
  e_dxdy_nodes = compute_column_monotone_derivatives(log_T_grid, e_dx_nodes, nx, ny);

  log_grid_base_report(*this, table, label);
  log_row_consistency_report(*this, label);
  log_reconstruction_error_report(*this, table, label);
  log_midpoint_compare_report(*this, table, label);
  log_temperature_inversion_report(*this, table, label);
  log_electron_target_region_report(*this, table, label);
}

HelmholtzSplineTriplet build_helmholtz_spline_triplet(const EOSTableTriplet& tables) {
  HelmholtzSplineTriplet out;
  // Hydro uses only the smooth total-EOS surrogate. Ion/electron inverse/closure for
  // two-temperature bookkeeping remains on the legacy raw-table path.
  out.total.build_from_table(tables.total, "total");
  log_total_pressure_contribution_report(tables);
  return out;
}

}  // namespace tenryu::materials
