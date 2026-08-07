#include "materials/helmholtz_jet.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"

namespace tenryu::materials {
namespace {

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

struct PositivityStats {
  double min_value = std::numeric_limits<double>::infinity();
  double rho_at_min = 0.0;
  double T_at_min = 0.0;
  std::size_t nonpositive = 0;
  bool initialized = false;

  void update(const double value, const double rho, const double T) {
    if (!initialized || value < min_value) {
      min_value = value;
      rho_at_min = rho;
      T_at_min = T;
      initialized = true;
    }
    if (!(value > 0.0) || !std::isfinite(value)) {
      ++nonpositive;
    }
  }
};

struct InversionStats {
  ErrorStats rel{};
  std::size_t bracket_failures = 0;
  std::size_t convergence_failures = 0;
};

[[nodiscard]] std::string format_error_stat(const ErrorStats& s) {
  std::ostringstream oss;
  if (!s.initialized) {
    oss << "n/a";
  } else {
    oss << s.max_rel << " @ rho=" << s.rho_at_max << ", T=" << s.T_at_max << " eV";
  }
  return oss.str();
}

[[nodiscard]] std::string format_pos_stat(const PositivityStats& s) {
  std::ostringstream oss;
  if (!s.initialized) {
    oss << "n/a";
  } else {
    oss << "min=" << s.min_value << " @ rho=" << s.rho_at_min << ", T=" << s.T_at_min
        << " eV, nonpositive=" << s.nonpositive;
  }
  return oss.str();
}

[[nodiscard]] std::size_t find_cell_index(const std::vector<double>& grid, const double x) {
  TENRYU_ASSERT(grid.size() >= 2, "HelmholtzJet requires at least 2 grid points");
  if (x <= grid.front()) {
    return 0;
  }
  if (x >= grid.back()) {
    return grid.size() - 2;
  }
  const auto it = std::upper_bound(grid.begin(), grid.end(), x);
  const std::size_t idx = static_cast<std::size_t>(it - grid.begin() - 1);
  return std::min(idx, grid.size() - 2);
}

[[nodiscard]] double deriv_1d_at(const std::vector<double>& grid,
                                 const std::vector<double>& values,
                                 const std::size_t idx) {
  const std::size_t n = grid.size();
  if (n <= 1) {
    return 0.0;
  }
  if (idx == 0) {
    return (values[1] - values[0]) / std::max(grid[1] - grid[0], 1.0e-30);
  }
  if (idx + 1 >= n) {
    return (values[n - 1] - values[n - 2]) /
           std::max(grid[n - 1] - grid[n - 2], 1.0e-30);
  }
  return (values[idx + 1] - values[idx - 1]) /
         std::max(grid[idx + 1] - grid[idx - 1], 1.0e-30);
}

void basis_quintic_hermite(const double t, double* b, double* db, double* ddb) {
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double t4 = t3 * t;
  const double t5 = t4 * t;

  b[0] = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5;
  b[1] = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5;
  b[2] = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5;
  b[3] = 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
  b[4] = -4.0 * t3 + 7.0 * t4 - 3.0 * t5;
  b[5] = 0.5 * t3 - t4 + 0.5 * t5;

  db[0] = -30.0 * t2 + 60.0 * t3 - 30.0 * t4;
  db[1] = 1.0 - 18.0 * t2 + 32.0 * t3 - 15.0 * t4;
  db[2] = t - 4.5 * t2 + 6.0 * t3 - 2.5 * t4;
  db[3] = 30.0 * t2 - 60.0 * t3 + 30.0 * t4;
  db[4] = -12.0 * t2 + 28.0 * t3 - 15.0 * t4;
  db[5] = 1.5 * t2 - 4.0 * t3 + 2.5 * t4;

  ddb[0] = -60.0 * t + 180.0 * t2 - 120.0 * t3;
  ddb[1] = -36.0 * t + 96.0 * t2 - 60.0 * t3;
  ddb[2] = 1.0 - 9.0 * t + 18.0 * t2 - 10.0 * t3;
  ddb[3] = 60.0 * t - 180.0 * t2 + 120.0 * t3;
  ddb[4] = -24.0 * t + 84.0 * t2 - 60.0 * t3;
  ddb[5] = 3.0 * t - 12.0 * t2 + 10.0 * t3;
}

}  // namespace

void HelmholtzJetEOS::clear() {
  log_rho_grid.clear();
  log_T_grid.clear();
  nodes.clear();
  cell_coeffs.clear();
  n_rho = 0;
  n_T = 0;
}

void HelmholtzJetEOS::build_from_table(const EOSTable& table, const std::string_view label) {
  clear();
  if (table.empty()) {
    return;
  }

  TENRYU_ASSERT(table.log_rho_grid.size() == table.n_rho() &&
                    table.log_T_grid.size() == table.n_T(),
                "HelmholtzJet grid size mismatch");
  TENRYU_ASSERT(table.P_table.size() == table.n_rho() * table.n_T() &&
                    table.e_table.size() == table.n_rho() * table.n_T(),
                "HelmholtzJet table size mismatch");

  log_rho_grid = table.log_rho_grid;
  log_T_grid = table.log_T_grid;
  n_rho = table.n_rho();
  n_T = table.n_T();
  nodes.assign(n_rho * n_T, HelmholtzJetNode{});

  std::vector<double> phi_x(n_rho * n_T, 0.0);
  std::vector<double> phi_y(n_rho * n_T, 0.0);

  for (std::size_t j = 0; j < n_T; ++j) {
    const double T = table.T_grid_eV[j];
    for (std::size_t i = 0; i < n_rho; ++i) {
      const std::size_t idx = flat_index(i, j);
      const double rho = table.rho_grid[i];
      phi_x[idx] = table.P_table[idx] / std::max(rho * T, 1.0e-30);
      phi_y[idx] = -table.e_table[idx] / std::max(T, 1.0e-30);
      nodes[idx].phi_x = phi_x[idx];
      nodes[idx].phi_y = phi_y[idx];
    }
  }

  std::vector<double> phi_xx(n_rho * n_T, 0.0);
  std::vector<double> phi_yy(n_rho * n_T, 0.0);
  std::vector<double> phi_xy_x(n_rho * n_T, 0.0);
  std::vector<double> phi_xy_y(n_rho * n_T, 0.0);

  for (std::size_t j = 0; j < n_T; ++j) {
    std::vector<double> row_x(n_rho, 0.0);
    std::vector<double> row_y(n_rho, 0.0);
    for (std::size_t i = 0; i < n_rho; ++i) {
      row_x[i] = phi_x[flat_index(i, j)];
      row_y[i] = phi_y[flat_index(i, j)];
    }
    for (std::size_t i = 0; i < n_rho; ++i) {
      phi_xx[flat_index(i, j)] = deriv_1d_at(log_rho_grid, row_x, i);
      phi_xy_y[flat_index(i, j)] = deriv_1d_at(log_rho_grid, row_y, i);
    }
  }
  for (std::size_t i = 0; i < n_rho; ++i) {
    std::vector<double> col_x(n_T, 0.0);
    std::vector<double> col_y(n_T, 0.0);
    for (std::size_t j = 0; j < n_T; ++j) {
      col_x[j] = phi_x[flat_index(i, j)];
      col_y[j] = phi_y[flat_index(i, j)];
    }
    for (std::size_t j = 0; j < n_T; ++j) {
      phi_xy_x[flat_index(i, j)] = deriv_1d_at(log_T_grid, col_x, j);
      phi_yy[flat_index(i, j)] = deriv_1d_at(log_T_grid, col_y, j);
    }
  }

  double max_cv = 0.0;
  double max_drho = 0.0;
  for (std::size_t j = 0; j < n_T; ++j) {
    const double T = table.T_grid_eV[j];
    for (std::size_t i = 0; i < n_rho; ++i) {
      const std::size_t idx = flat_index(i, j);
      max_cv = std::max(max_cv, std::fabs(-(phi_y[idx] + phi_yy[idx])));
      max_drho = std::max(max_drho, std::fabs(T * (phi_x[idx] + phi_xx[idx])));
    }
  }
  const double eps_v = std::max(1.0e-6 * std::max(max_cv, 1.0), 1.0e-30);
  const double eps_rho = std::max(1.0e-6 * std::max(max_drho, 1.0), 1.0e-30);

  for (std::size_t j = 0; j < n_T; ++j) {
    const double T = table.T_grid_eV[j];
    for (std::size_t i = 0; i < n_rho; ++i) {
      const std::size_t idx = flat_index(i, j);
      const double phi_xx_min = eps_rho / std::max(T, 1.0e-30) - phi_x[idx];
      const double phi_yy_max = -phi_y[idx] - eps_v;
      nodes[idx].phi_xx = std::max(phi_xx[idx], phi_xx_min);
      nodes[idx].phi_yy = std::min(phi_yy[idx], phi_yy_max);
      nodes[idx].phi_xy = 0.5 * (phi_xy_x[idx] + phi_xy_y[idx]);
    }
  }

  std::vector<double> phi_x_then_y(n_rho * n_T, 0.0);
  std::vector<double> phi_y_then_x(n_rho * n_T, 0.0);
  for (std::size_t i = 1; i < n_rho; ++i) {
    phi_x_then_y[flat_index(i, 0)] =
        phi_x_then_y[flat_index(i - 1, 0)] +
        0.5 * (nodes[flat_index(i - 1, 0)].phi_x + nodes[flat_index(i, 0)].phi_x) *
            (log_rho_grid[i] - log_rho_grid[i - 1]);
  }
  for (std::size_t i = 0; i < n_rho; ++i) {
    for (std::size_t j = 1; j < n_T; ++j) {
      phi_x_then_y[flat_index(i, j)] =
          phi_x_then_y[flat_index(i, j - 1)] +
          0.5 * (nodes[flat_index(i, j - 1)].phi_y + nodes[flat_index(i, j)].phi_y) *
              (log_T_grid[j] - log_T_grid[j - 1]);
    }
  }
  for (std::size_t j = 1; j < n_T; ++j) {
    phi_y_then_x[flat_index(0, j)] =
        phi_y_then_x[flat_index(0, j - 1)] +
        0.5 * (nodes[flat_index(0, j - 1)].phi_y + nodes[flat_index(0, j)].phi_y) *
            (log_T_grid[j] - log_T_grid[j - 1]);
  }
  for (std::size_t j = 0; j < n_T; ++j) {
    for (std::size_t i = 1; i < n_rho; ++i) {
      phi_y_then_x[flat_index(i, j)] =
          phi_y_then_x[flat_index(i - 1, j)] +
          0.5 * (nodes[flat_index(i - 1, j)].phi_x + nodes[flat_index(i, j)].phi_x) *
              (log_rho_grid[i] - log_rho_grid[i - 1]);
    }
  }
  for (std::size_t idx = 0; idx < nodes.size(); ++idx) {
    nodes[idx].phi = 0.5 * (phi_x_then_y[idx] + phi_y_then_x[idx]);
  }

  cell_coeffs.assign((n_rho - 1) * (n_T - 1) * 36, 0.0);
  auto set_coeff = [&](const std::size_t i_cell,
                       const std::size_t j_cell,
                       const int ax,
                       const int ay,
                       const double value) {
    const std::size_t cell = (j_cell * (n_rho - 1) + i_cell) * 36;
    cell_coeffs[cell + static_cast<std::size_t>(ax) * 6 + static_cast<std::size_t>(ay)] = value;
  };

  for (std::size_t j = 0; j + 1 < n_T; ++j) {
    const double hy = log_T_grid[j + 1] - log_T_grid[j];
    for (std::size_t i = 0; i + 1 < n_rho; ++i) {
      const double hx = log_rho_grid[i + 1] - log_rho_grid[i];
      const HelmholtzJetNode& bl = nodes[flat_index(i, j)];
      const HelmholtzJetNode& br = nodes[flat_index(i + 1, j)];
      const HelmholtzJetNode& tl = nodes[flat_index(i, j + 1)];
      const HelmholtzJetNode& tr = nodes[flat_index(i + 1, j + 1)];

      set_coeff(i, j, 0, 0, bl.phi);
      set_coeff(i, j, 1, 0, bl.phi_x * hx);
      set_coeff(i, j, 2, 0, bl.phi_xx * hx * hx);
      set_coeff(i, j, 0, 1, bl.phi_y * hy);
      set_coeff(i, j, 1, 1, bl.phi_xy * hx * hy);
      set_coeff(i, j, 0, 2, bl.phi_yy * hy * hy);

      set_coeff(i, j, 3, 0, br.phi);
      set_coeff(i, j, 4, 0, br.phi_x * hx);
      set_coeff(i, j, 5, 0, br.phi_xx * hx * hx);
      set_coeff(i, j, 3, 1, br.phi_y * hy);
      set_coeff(i, j, 4, 1, br.phi_xy * hx * hy);
      set_coeff(i, j, 3, 2, br.phi_yy * hy * hy);

      set_coeff(i, j, 0, 3, tl.phi);
      set_coeff(i, j, 1, 3, tl.phi_x * hx);
      set_coeff(i, j, 2, 3, tl.phi_xx * hx * hx);
      set_coeff(i, j, 0, 4, tl.phi_y * hy);
      set_coeff(i, j, 1, 4, tl.phi_xy * hx * hy);
      set_coeff(i, j, 0, 5, tl.phi_yy * hy * hy);

      set_coeff(i, j, 3, 3, tr.phi);
      set_coeff(i, j, 4, 3, tr.phi_x * hx);
      set_coeff(i, j, 5, 3, tr.phi_xx * hx * hx);
      set_coeff(i, j, 3, 4, tr.phi_y * hy);
      set_coeff(i, j, 4, 4, tr.phi_xy * hx * hy);
      set_coeff(i, j, 3, 5, tr.phi_yy * hy * hy);
    }
  }

  ErrorStats p_stats{};
  ErrorStats e_stats{};
  PositivityStats cv_stats{};
  PositivityStats drho_stats{};
  PositivityStats cs2_stats{};
  InversionStats inv{};
  for (std::size_t j = 0; j < n_T; ++j) {
    const double T = table.T_grid_eV[j];
    for (std::size_t i = 0; i < n_rho; ++i) {
      const std::size_t idx = flat_index(i, j);
      const double rho = table.rho_grid[i];
      const HelmholtzJetThermo thermo = eval_thermo(rho, T);
      const double p_rel =
          std::fabs(thermo.pressure - table.P_table[idx]) /
          std::max(std::fabs(table.P_table[idx]), 1.0e-30);
      const double e_rel =
          std::fabs(thermo.energy - table.e_table[idx]) /
          std::max(std::fabs(table.e_table[idx]), 1.0e-30);
      p_stats.update(p_rel, rho, T);
      e_stats.update(e_rel, rho, T);
      cv_stats.update(thermo.cv, rho, T);
      drho_stats.update(thermo.dP_drho_T, rho, T);
      cs2_stats.update(thermo.sound_speed * thermo.sound_speed, rho, T);
      const double T_inv = temperature_from_energy(rho, table.e_table[idx]);
      const double t_rel = std::fabs(T_inv - T) / std::max(T, 1.0e-30);
      inv.rel.update(t_rel, rho, T);
      if (!(T_inv > 0.0) || !std::isfinite(T_inv)) {
        ++inv.convergence_failures;
      }
    }
  }

  core::log_info("HelmholtzJet[" + std::string(label) + "] node relerr: P=" +
                 format_error_stat(p_stats) + ", e=" + format_error_stat(e_stats));
  core::log_info("HelmholtzJet[" + std::string(label) + "] positivity: cv{" +
                 format_pos_stat(cv_stats) + "}, dPdrho{" + format_pos_stat(drho_stats) +
                 "}, cs2{" + format_pos_stat(cs2_stats) + "}");
  core::log_info("HelmholtzJet[" + std::string(label) +
                 "] T_from_e node inversion: relerr=" + format_error_stat(inv.rel) +
                 ", bracket_failures=" + std::to_string(inv.bracket_failures) +
                 ", convergence_failures=" + std::to_string(inv.convergence_failures));
}

HelmholtzJetEval HelmholtzJetEOS::eval(const double log_rho_raw, const double log_T_raw) const {
  HelmholtzJetEval out{};
  if (empty()) {
    return out;
  }

  const double log_rho = std::clamp(log_rho_raw, log_rho_grid.front(), log_rho_grid.back());
  const double log_T = std::clamp(log_T_raw, log_T_grid.front(), log_T_grid.back());
  const std::size_t i0 = find_cell_index(log_rho_grid, log_rho);
  const std::size_t j0 = find_cell_index(log_T_grid, log_T);
  const double hx = std::max(log_rho_grid[i0 + 1] - log_rho_grid[i0], 1.0e-30);
  const double hy = std::max(log_T_grid[j0 + 1] - log_T_grid[j0], 1.0e-30);
  const double s = std::clamp((log_rho - log_rho_grid[i0]) / hx, 0.0, 1.0);
  const double t = std::clamp((log_T - log_T_grid[j0]) / hy, 0.0, 1.0);

  double bx[6], dbx[6], ddbx[6];
  double by[6], dby[6], ddby[6];
  basis_quintic_hermite(s, bx, dbx, ddbx);
  basis_quintic_hermite(t, by, dby, ddby);

  const std::size_t cell = (j0 * (n_rho - 1) + i0) * 36;
  for (int ax = 0; ax < 6; ++ax) {
    for (int ay = 0; ay < 6; ++ay) {
      const double c = cell_coeffs[cell + static_cast<std::size_t>(ax) * 6 +
                                   static_cast<std::size_t>(ay)];
      out.phi += c * bx[ax] * by[ay];
      out.phi_x += c * dbx[ax] * by[ay] / hx;
      out.phi_y += c * bx[ax] * dby[ay] / hy;
      out.phi_xx += c * ddbx[ax] * by[ay] / (hx * hx);
      out.phi_yy += c * bx[ax] * ddby[ay] / (hy * hy);
      out.phi_xy += c * dbx[ax] * dby[ay] / (hx * hy);
    }
  }
  return out;
}

HelmholtzJetThermo HelmholtzJetEOS::eval_thermo(const double rho, const double T_eV) const {
  HelmholtzJetThermo out{};
  if (!(rho > 0.0) || !(T_eV > 0.0) || empty()) {
    return out;
  }
  const HelmholtzJetEval phi = eval(std::log(rho), std::log(T_eV));
  const double cv = -(phi.phi_y + phi.phi_yy);
  const double dP_drho_T = T_eV * (phi.phi_x + phi.phi_xx);
  const double dP_dT_rho = rho * (phi.phi_x + phi.phi_xy);
  const double cs2 = dP_drho_T +
                     T_eV * dP_dT_rho * dP_dT_rho /
                         std::max(rho * rho * std::max(cv, 1.0e-30), 1.0e-30);
  out.pressure = rho * T_eV * phi.phi_x;
  out.energy = -T_eV * phi.phi_y;
  out.cv = cv;
  out.dP_drho_T = dP_drho_T;
  out.dP_dT_rho = dP_dT_rho;
  out.sound_speed = (cs2 > 0.0 && std::isfinite(cs2)) ? std::sqrt(cs2) : 0.0;
  return out;
}

double HelmholtzJetEOS::temperature_from_energy(const double rho, const double e_target) const {
  if (!(rho > 0.0) || empty()) {
    return 0.0;
  }
  double y_prev = log_T_grid.front();
  double e_prev = eval_thermo(rho, std::exp(y_prev)).energy;
  if (e_target <= e_prev) {
    return std::exp(y_prev);
  }
  double y_lo = y_prev;
  double y_hi = log_T_grid.back();
  double e_lo = e_prev;
  double e_hi = eval_thermo(rho, std::exp(y_hi)).energy;
  bool found = false;
  for (std::size_t j = 1; j < n_T; ++j) {
    const double y_curr = log_T_grid[j];
    const double e_curr = eval_thermo(rho, std::exp(y_curr)).energy;
    if ((e_target >= e_prev && e_target <= e_curr) ||
        (e_target <= e_prev && e_target >= e_curr)) {
      y_lo = y_prev;
      y_hi = y_curr;
      e_lo = e_prev;
      e_hi = e_curr;
      found = true;
      break;
    }
    y_prev = y_curr;
    e_prev = e_curr;
  }
  if (!found) {
    return std::exp(y_hi);
  }
  if (e_hi < e_lo) {
    std::swap(y_lo, y_hi);
    std::swap(e_lo, e_hi);
  }
  if (e_target <= e_lo) {
    return std::exp(y_lo);
  }
  if (e_target >= e_hi) {
    return std::exp(y_hi);
  }

  double y = y_lo + (e_target - e_lo) * (y_hi - y_lo) / std::max(e_hi - e_lo, 1.0e-30);
  for (int iter = 0; iter < 32; ++iter) {
    const HelmholtzJetThermo thermo = eval_thermo(rho, std::exp(y));
    const double g = thermo.energy - e_target;
    if (std::fabs(g) <= 1.0e-10 * std::max(std::fabs(e_target), 1.0)) {
      return std::exp(y);
    }
    if (g > 0.0) {
      y_hi = y;
    } else {
      y_lo = y;
    }
    const double gp = std::exp(y) * thermo.cv;
    double y_new = 0.5 * (y_lo + y_hi);
    if (std::fabs(gp) > 1.0e-30 && std::isfinite(gp)) {
      const double y_trial = y - g / gp;
      if (y_trial > y_lo && y_trial < y_hi) {
        y_new = y_trial;
      }
    }
    y = y_new;
  }
  return std::exp(0.5 * (y_lo + y_hi));
}

}  // namespace tenryu::materials
