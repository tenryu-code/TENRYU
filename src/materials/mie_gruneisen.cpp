#include "materials/mie_gruneisen.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "materials/eos_table.hpp"

namespace tenryu::materials {
namespace {

constexpr double kGammaFloor = 1.0e-8;
constexpr double kPressureFloor = 1.0e-30;
constexpr double kCs2Floor = 1.0e-30;

int signum(const double x) {
  if (x > 0.0) {
    return 1;
  }
  if (x < 0.0) {
    return -1;
  }
  return 0;
}

std::vector<double> monotone_hermite_derivatives_1d(const std::vector<double>& x,
                                                    const std::vector<double>& y) {
  const std::size_t n = x.size();
  TENRYU_ASSERT(y.size() == n, "MieGruneisenEOS: x/y size mismatch");
  std::vector<double> d(n, 0.0);
  if (n < 2) {
    return d;
  }
  if (n == 2) {
    const double h = x[1] - x[0];
    TENRYU_ASSERT(h > 0.0, "MieGruneisenEOS requires strictly increasing rho grid");
    const double sec = (y[1] - y[0]) / h;
    d[0] = sec;
    d[1] = sec;
    return d;
  }

  std::vector<double> h(n - 1, 0.0);
  std::vector<double> delta(n - 1, 0.0);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    h[i] = x[i + 1] - x[i];
    TENRYU_ASSERT(h[i] > 0.0, "MieGruneisenEOS requires strictly increasing rho grid");
    delta[i] = (y[i + 1] - y[i]) / h[i];
  }

  d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
  if (signum(d[0]) != signum(delta[0])) {
    d[0] = 0.0;
  } else if (signum(delta[0]) != signum(delta[1]) &&
             std::fabs(d[0]) > 3.0 * std::fabs(delta[0])) {
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
    const double sq = alpha * alpha + beta * beta;
    if (sq > 9.0) {
      const double tau = 3.0 / std::sqrt(sq);
      d[i] = tau * alpha * delta[i];
      d[i + 1] = tau * beta * delta[i];
    }
  }
  return d;
}

double clamp_temperature(const EOSTable& table, const double T_ref_eV) {
  TENRYU_ASSERT(!table.T_grid_eV.empty(), "MieGruneisenEOS requires non-empty T grid");
  return std::clamp(T_ref_eV, table.T_grid_eV.front(), table.T_grid_eV.back());
}

double branch_gamma_from_table(const EOSTable& table,
                               const double rho,
                               const double T_ref_eV,
                               const double dT_rel) {
  const double T_min = table.T_grid_eV.front();
  const double T_max = table.T_grid_eV.back();
  const double T_ref = clamp_temperature(table, T_ref_eV);
  const double dT_abs = std::max(std::fabs(T_ref) * dT_rel, 1.0e-6);

  auto gamma_from_pair = [&](const double T_lo, const double T_hi) {
    const double e_lo = table.energy(rho, T_lo);
    const double e_hi = table.energy(rho, T_hi);
    const double p_lo = table.pressure(rho, T_lo);
    const double p_hi = table.pressure(rho, T_hi);
    const double de = e_hi - e_lo;
    if (!(std::isfinite(de) && std::fabs(de) > 1.0e-30 &&
          std::isfinite(p_lo) && std::isfinite(p_hi) && rho > 0.0)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    return (p_hi - p_lo) / (rho * de);
  };

  double gamma = std::numeric_limits<double>::quiet_NaN();
  const double T_lo_c = std::max(T_min, T_ref - dT_abs);
  const double T_hi_c = std::min(T_max, T_ref + dT_abs);
  if (T_hi_c > T_lo_c) {
    gamma = gamma_from_pair(T_lo_c, T_hi_c);
  }
  if (!(std::isfinite(gamma) && gamma > kGammaFloor)) {
    const double T_hi = std::min(T_max, T_ref + dT_abs);
    if (T_hi > T_ref) {
      gamma = gamma_from_pair(T_ref, T_hi);
    }
  }
  if (!(std::isfinite(gamma) && gamma > kGammaFloor)) {
    const double T_lo = std::max(T_min, T_ref - dT_abs);
    if (T_ref > T_lo) {
      gamma = gamma_from_pair(T_lo, T_ref);
    }
  }
  if (!(std::isfinite(gamma) && gamma > kGammaFloor)) {
    const double e_ref = table.energy(rho, T_ref);
    const double p_ref = table.pressure(rho, T_ref);
    if (rho > 0.0 && std::isfinite(e_ref) && std::isfinite(p_ref) && e_ref > 1.0e-30) {
      gamma = p_ref / (rho * e_ref);
    }
  }
  if (!(std::isfinite(gamma) && gamma > kGammaFloor)) {
    gamma = kGammaFloor;
  }
  return gamma;
}

std::size_t find_temperature_window_begin(const std::vector<double>& T_grid) {
  const auto it = std::lower_bound(T_grid.begin(), T_grid.end(), 1.0);
  return static_cast<std::size_t>(it - T_grid.begin());
}

std::size_t find_temperature_window_end(const std::vector<double>& T_grid) {
  const auto it = std::upper_bound(T_grid.begin(), T_grid.end(), 50.0);
  const std::size_t idx = static_cast<std::size_t>(it - T_grid.begin());
  return idx > 0 ? idx : T_grid.size();
}

double format_or_zero(const double value) {
  return std::isfinite(value) ? value : 0.0;
}

std::string format_scientific(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6) << format_or_zero(value);
  return oss.str();
}

double branch_pressure(const double p_ref,
                       const double gamma,
                       const double rho,
                       const double e_specific,
                       const double e_ref) {
  const double pressure = p_ref + gamma * rho * (e_specific - e_ref);
  return (std::isfinite(pressure) && pressure > kPressureFloor) ? pressure : kPressureFloor;
}

double branch_cs2(const double gamma,
                  const double p_ref,
                  const double e_ref,
                  const double dgamma_dlogrho,
                  const double dp_ref_dlogrho,
                  const double de_ref_dlogrho,
                  const double rho,
                  const double e_specific,
                  const double pressure) {
  const double rho_safe = std::max(rho, 1.0e-30);
  const double delta_e = e_specific - e_ref;
  const double dP_dlogrho = dp_ref_dlogrho +
                            dgamma_dlogrho * rho_safe * delta_e +
                            gamma * rho_safe * delta_e -
                            gamma * rho_safe * de_ref_dlogrho;
  const double dPdrho = dP_dlogrho / rho_safe;
  const double cs2 = dPdrho + gamma * pressure / rho_safe;
  return (std::isfinite(cs2) && cs2 > kCs2Floor) ? cs2 : kCs2Floor;
}

}  // namespace

void MieGruneisenEOS::build_from_tables(const EOSTable& ion,
                                        const EOSTable& electron,
                                        const double T_ref_eV,
                                        const double dT_rel,
                                        const std::string& label) {
  TENRYU_ASSERT(!ion.empty() && !electron.empty(),
                "MieGruneisenEOS requires non-empty ion/electron tables");
  TENRYU_ASSERT(ion.n_rho() == electron.n_rho(),
                "MieGruneisenEOS ion/electron rho-grid size mismatch");
  TENRYU_ASSERT(ion.rho_grid == electron.rho_grid,
                "MieGruneisenEOS ion/electron rho-grid mismatch");
  TENRYU_ASSERT(ion.n_T() == electron.n_T(),
                "MieGruneisenEOS ion/electron T-grid size mismatch");
  TENRYU_ASSERT(ion.T_grid_eV == electron.T_grid_eV,
                "MieGruneisenEOS ion/electron T-grid mismatch");
  TENRYU_ASSERT(!ion.log_rho_grid.empty() && !electron.log_rho_grid.empty(),
                "MieGruneisenEOS requires finalized ion/electron tables");
  TENRYU_ASSERT(dT_rel > 0.0, "MieGruneisenEOS requires mg_dT_rel > 0");

  const std::size_t n_rho = ion.n_rho();
  log_rho_grid_ = ion.log_rho_grid;
  gamma_ion_.assign(n_rho, kGammaFloor);
  gamma_ele_.assign(n_rho, kGammaFloor);
  p_ref_ion_.assign(n_rho, kPressureFloor);
  p_ref_ele_.assign(n_rho, kPressureFloor);
  e_ref_ion_.assign(n_rho, 0.0);
  e_ref_ele_.assign(n_rho, 0.0);

  for (std::size_t i = 0; i < n_rho; ++i) {
    const double rho = ion.rho_grid[i];
    const double T_ref_ion = clamp_temperature(ion, T_ref_eV);
    const double T_ref_ele = clamp_temperature(electron, T_ref_eV);

    e_ref_ion_[i] = ion.energy(rho, T_ref_ion);
    e_ref_ele_[i] = electron.energy(rho, T_ref_ele);
    p_ref_ion_[i] = std::max(ion.pressure(rho, T_ref_ion), kPressureFloor);
    p_ref_ele_[i] = std::max(electron.pressure(rho, T_ref_ele), kPressureFloor);
    gamma_ion_[i] = branch_gamma_from_table(ion, rho, T_ref_ion, dT_rel);
    gamma_ele_[i] = branch_gamma_from_table(electron, rho, T_ref_ele, dT_rel);
  }

  dgamma_ion_dlogrho_ = monotone_hermite_derivatives_1d(log_rho_grid_, gamma_ion_);
  dgamma_ele_dlogrho_ = monotone_hermite_derivatives_1d(log_rho_grid_, gamma_ele_);
  dp_ref_ion_dlogrho_ = monotone_hermite_derivatives_1d(log_rho_grid_, p_ref_ion_);
  dp_ref_ele_dlogrho_ = monotone_hermite_derivatives_1d(log_rho_grid_, p_ref_ele_);
  de_ref_ion_dlogrho_ = monotone_hermite_derivatives_1d(log_rho_grid_, e_ref_ion_);
  de_ref_ele_dlogrho_ = monotone_hermite_derivatives_1d(log_rho_grid_, e_ref_ele_);

  const std::size_t j_begin = find_temperature_window_begin(ion.T_grid_eV);
  const std::size_t j_end = find_temperature_window_end(ion.T_grid_eV);
  const std::size_t j_lo = std::min(j_begin, ion.n_T() - 1);
  const std::size_t j_hi = std::max(j_lo + 1, std::min(j_end, ion.n_T()));

  double gamma_ion_min = std::numeric_limits<double>::max();
  double gamma_ion_max = 0.0;
  double gamma_ele_min = std::numeric_limits<double>::max();
  double gamma_ele_max = 0.0;
  double cs2_min_global = std::numeric_limits<double>::max();

  core::log_info("MieGruneisen[" + label + "] build: T_ref_eV=" + format_scientific(T_ref_eV) +
                 " dT_rel=" + format_scientific(dT_rel) +
                 " window_T=[" + format_scientific(ion.T_grid_eV[j_lo]) + "," +
                 format_scientific(ion.T_grid_eV[j_hi - 1]) + "]");

  for (std::size_t i = 0; i < n_rho; ++i) {
    const double rho = ion.rho_grid[i];
    gamma_ion_min = std::min(gamma_ion_min, gamma_ion_[i]);
    gamma_ion_max = std::max(gamma_ion_max, gamma_ion_[i]);
    gamma_ele_min = std::min(gamma_ele_min, gamma_ele_[i]);
    gamma_ele_max = std::max(gamma_ele_max, gamma_ele_[i]);

    double ion_rel_sum = 0.0;
    double ele_rel_sum = 0.0;
    double total_rel_sum = 0.0;
    double ion_rel_max = 0.0;
    double ele_rel_max = 0.0;
    double total_rel_max = 0.0;
    int sample_count = 0;

    for (std::size_t j = j_lo; j < j_hi; ++j) {
      const double T = ion.T_grid_eV[j];
      const double ei = ion.energy(rho, T);
      const double ee = electron.energy(rho, T);
      const double pi_table = std::max(ion.pressure(rho, T), kPressureFloor);
      const double pe_table = std::max(electron.pressure(rho, T), kPressureFloor);
      const double pi_mg =
          branch_pressure(p_ref_ion_[i], gamma_ion_[i], rho, ei, e_ref_ion_[i]);
      const double pe_mg =
          branch_pressure(p_ref_ele_[i], gamma_ele_[i], rho, ee, e_ref_ele_[i]);
      const double pt_table = pi_table + pe_table;
      const double pt_mg = pi_mg + pe_mg;
      const double ion_rel = std::fabs(pi_mg - pi_table) / std::max(std::fabs(pi_table), 1.0e-30);
      const double ele_rel = std::fabs(pe_mg - pe_table) / std::max(std::fabs(pe_table), 1.0e-30);
      const double total_rel =
          std::fabs(pt_mg - pt_table) / std::max(std::fabs(pt_table), 1.0e-30);
      ion_rel_sum += ion_rel;
      ele_rel_sum += ele_rel;
      total_rel_sum += total_rel;
      ion_rel_max = std::max(ion_rel_max, ion_rel);
      ele_rel_max = std::max(ele_rel_max, ele_rel);
      total_rel_max = std::max(total_rel_max, total_rel);
      const double cs2_i = branch_cs2(gamma_ion_[i], p_ref_ion_[i], e_ref_ion_[i],
                                      dgamma_ion_dlogrho_[i], dp_ref_ion_dlogrho_[i],
                                      de_ref_ion_dlogrho_[i], rho, ei, pi_mg);
      const double cs2_e = branch_cs2(gamma_ele_[i], p_ref_ele_[i], e_ref_ele_[i],
                                      dgamma_ele_dlogrho_[i], dp_ref_ele_dlogrho_[i],
                                      de_ref_ele_dlogrho_[i], rho, ee, pe_mg);
      cs2_min_global = std::min(cs2_min_global, cs2_i + cs2_e);
      ++sample_count;
    }

    const double inv_count = (sample_count > 0) ? (1.0 / static_cast<double>(sample_count)) : 0.0;
    core::log_info("MieGruneisen[" + label + "] rho=" + format_scientific(rho) +
                   " Gamma{i,e}=(" + format_scientific(gamma_ion_[i]) + "," +
                   format_scientific(gamma_ele_[i]) + ")" +
                   " relP_mean{i,e,t}=(" + format_scientific(ion_rel_sum * inv_count) + "," +
                   format_scientific(ele_rel_sum * inv_count) + "," +
                   format_scientific(total_rel_sum * inv_count) + ")" +
                   " relP_max{i,e,t}=(" + format_scientific(ion_rel_max) + "," +
                   format_scientific(ele_rel_max) + "," +
                   format_scientific(total_rel_max) + ")");
  }

  core::log_info("MieGruneisen[" + label + "] Gamma range: ion=[" +
                 format_scientific(gamma_ion_min) + "," +
                 format_scientific(gamma_ion_max) + "] electron=[" +
                 format_scientific(gamma_ele_min) + "," +
                 format_scientific(gamma_ele_max) + "] cs2_min=" +
                 format_scientific(cs2_min_global));
}

}  // namespace tenryu::materials
