#include "materials/eos_table.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "materials/ionmix_reader.hpp"
#include "materials/sesame_reader.hpp"

namespace tenryu::materials {
namespace {

constexpr double kCvFloor = 1.0e-3;

void assert_monotonic_positive(const std::vector<double>& grid,
                               const char* grid_name) {
  TENRYU_ASSERT(!grid.empty(), "EOS grid must be non-empty");
  TENRYU_ASSERT(grid.front() > 0.0, "EOS grid values must be positive");
  for (std::size_t i = 1; i < grid.size(); ++i) {
    TENRYU_ASSERT(grid[i] > 0.0, "EOS grid values must be positive");
    TENRYU_ASSERT(grid[i] > grid[i - 1], grid_name);
  }
}

std::pair<std::size_t, std::size_t> bracket_index(const std::vector<double>& grid,
                                                  const double x) {
  if (grid.size() < 2) {
    return {0, 0};
  }
  if (x <= grid.front()) {
    return {0, 1};
  }
  if (x >= grid.back()) {
    return {grid.size() - 2, grid.size() - 1};
  }

  const auto it = std::upper_bound(grid.begin(), grid.end(), x);
  const std::size_t hi = static_cast<std::size_t>(it - grid.begin());
  return {hi - 1, hi};
}

double bilinear_log_interp(const EOSTable& table,
                           const std::vector<double>& values,
                           const double rho,
                           const double T_eV) {
  TENRYU_ASSERT(values.size() == table.n_rho() * table.n_T(),
                "EOS table value array size mismatch");
  TENRYU_ASSERT(!table.log_rho_grid.empty() && !table.log_T_grid.empty(),
                "EOS table log grids are not initialized");

  if (std::isnan(rho) || std::isnan(T_eV)) {
    static bool warned_nan_input = false;
    if (!warned_nan_input) {
      core::log_warning("EOSTable interpolation received NaN rho/Te; clamping to table bounds");
      warned_nan_input = true;
    }
  }
  const double rho_safe = std::isnan(rho) ? table.rho_grid.front() : rho;
  const double T_safe = std::isnan(T_eV) ? table.T_grid_eV.front() : T_eV;
  const double rho_clamped =
      std::clamp(rho_safe, table.rho_grid.front(), table.rho_grid.back());
  const double T_clamped =
      std::clamp(T_safe, table.T_grid_eV.front(), table.T_grid_eV.back());
  const double x = std::log(rho_clamped);
  const double y = std::log(T_clamped);

  const auto [i0, i1] = bracket_index(table.log_rho_grid, x);
  const auto [j0, j1] = bracket_index(table.log_T_grid, y);

  const double x0 = table.log_rho_grid[i0];
  const double x1 = table.log_rho_grid[i1];
  const double y0 = table.log_T_grid[j0];
  const double y1 = table.log_T_grid[j1];

  const double tx = (x1 > x0) ? ((x - x0) / (x1 - x0)) : 0.0;
  const double ty = (y1 > y0) ? ((y - y0) / (y1 - y0)) : 0.0;

  const double v00 = values[table.flat_index(i0, j0)];
  const double v10 = values[table.flat_index(i1, j0)];
  const double v01 = values[table.flat_index(i0, j1)];
  const double v11 = values[table.flat_index(i1, j1)];

  const double vx0 = v00 + tx * (v10 - v00);
  const double vx1 = v01 + tx * (v11 - v01);
  return vx0 + ty * (vx1 - vx0);
}

EOSTable from_sesame_raw(const SesameEOSTableRaw& raw) {
  EOSTable table;
  table.rho_grid = raw.rho_grid;
  table.T_grid_eV = raw.T_grid_eV;
  table.P_table = raw.pressure_dyne_per_cm2;
  table.e_table = raw.energy_erg_per_g;
  table.finalize();
  return table;
}

std::vector<double> load_ascii_values(const std::string& filename) {
  std::ifstream in(filename);
  TENRYU_ASSERT(in.good(), "Failed to open IONMIX file");

  std::vector<double> values;
  double value = 0.0;
  while (in >> value) {
    values.push_back(value);
  }
  TENRYU_ASSERT(!values.empty(), "IONMIX file appears to be empty");
  return values;
}

std::vector<double> make_log_uniform_grid(const std::size_t n,
                                          const double lo,
                                          const double hi) {
  TENRYU_ASSERT(n >= 2, "log-uniform grid requires at least two points");
  TENRYU_ASSERT(lo > 0.0 && hi > lo, "log-uniform grid requires 0 < lo < hi");
  std::vector<double> grid(n, 0.0);
  const double log_lo = std::log(lo);
  const double log_hi = std::log(hi);
  const double inv = 1.0 / static_cast<double>(n - 1U);
  for (std::size_t i = 0; i < n; ++i) {
    const double s = static_cast<double>(i) * inv;
    grid[i] = std::exp(log_lo + s * (log_hi - log_lo));
  }
  return grid;
}

}  // namespace

void EOSTable::finalize() {
  assert_monotonic_positive(rho_grid, "EOS rho grid must be strictly increasing");
  assert_monotonic_positive(T_grid_eV, "EOS temperature grid must be strictly increasing");

  const std::size_t n_points = n_rho() * n_T();
  TENRYU_ASSERT(P_table.size() == n_points, "EOS pressure table size mismatch");
  TENRYU_ASSERT(e_table.size() == n_points, "EOS energy table size mismatch");

  log_rho_grid.resize(n_rho());
  for (std::size_t i = 0; i < n_rho(); ++i) {
    log_rho_grid[i] = std::log(rho_grid[i]);
  }
  log_T_grid.resize(n_T());
  for (std::size_t j = 0; j < n_T(); ++j) {
    log_T_grid[j] = std::log(T_grid_eV[j]);
  }

  cv_table.resize(n_points, kCvFloor);
  if (n_T() == 1) {
    return;
  }

  static bool warned_non_positive_cv = false;
  for (std::size_t i = 0; i < n_rho(); ++i) {
    for (std::size_t j = 0; j < n_T(); ++j) {
      const std::size_t jm = (j == 0) ? 0 : (j - 1);
      const std::size_t jp = (j + 1 >= n_T()) ? (n_T() - 1) : (j + 1);
      const double dT = T_grid_eV[jp] - T_grid_eV[jm];

      double cv = kCvFloor;
      if (dT > 0.0) {
        const double em = e_table[flat_index(i, jm)];
        const double ep = e_table[flat_index(i, jp)];
        cv = (ep - em) / dT;
      }
      if (cv <= 0.0 && !warned_non_positive_cv) {
        core::log_warning("EOSTable::finalize detected non-positive cv; clamping to floor");
        warned_non_positive_cv = true;
      }
      cv_table[flat_index(i, j)] = std::max(cv, kCvFloor);
    }
  }
}

double EOSTable::pressure(const double rho, const double T_eV) const {
  TENRYU_ASSERT(!empty(), "EOSTable::pressure requires non-empty table");
  return bilinear_log_interp(*this, P_table, rho, T_eV);
}

double EOSTable::energy(const double rho, const double T_eV) const {
  TENRYU_ASSERT(!empty(), "EOSTable::energy requires non-empty table");
  return bilinear_log_interp(*this, e_table, rho, T_eV);
}

double EOSTable::cv(const double rho, const double T_eV) const {
  TENRYU_ASSERT(!empty(), "EOSTable::cv requires non-empty table");
  return bilinear_log_interp(*this, cv_table, rho, T_eV);
}

double EOSTable::temperature_from_energy(const double rho, const double e) const {
  TENRYU_ASSERT(!empty(), "EOSTable::temperature_from_energy requires non-empty table");

  const double T_min = T_grid_eV.front();
  const double T_max = T_grid_eV.back();
  const double e_min = energy(rho, T_min);
  const double e_max = energy(rho, T_max);
  if (e <= e_min) {
    return T_min;
  }
  if (e >= e_max) {
    return T_max;
  }

  double T_lo = T_min;
  double T_hi = T_max;
  const double de_span = e_max - e_min;
  const double de_scale = std::max({std::abs(e_min), std::abs(e_max), 1.0});
  if (!(std::abs(de_span) > 1.0e-14 * de_scale)) {
    // Flat/degenerate table column: no meaningful inverse map in energy.
    return T_min;
  }
  const double alpha = (e - e_min) / de_span;
  double T = std::clamp(T_min + alpha * (T_max - T_min), T_min, T_max);

  for (int iter = 0; iter < 24; ++iter) {
    const double e_curr = energy(rho, T);
    const double cv_curr = std::max(cv(rho, T), kCvFloor);
    const double f = e_curr - e;
    if (std::abs(f) <= 1.0e-10 * std::max(std::abs(e), 1.0)) {
      break;
    }

    if (f > 0.0) {
      T_hi = std::min(T_hi, T);
    } else {
      T_lo = std::max(T_lo, T);
    }

    double T_next = T - f / cv_curr;
    if (!(T_next > T_lo && T_next < T_hi)) {
      T_next = 0.5 * (T_lo + T_hi);
    }
    T = std::clamp(T_next, T_min, T_max);
  }

  return T;
}

double EOSTable::sound_speed(const double rho, const double P) const {
  constexpr double kRhoFloor = 1.0e-30;
  if (!(rho > kRhoFloor)) {
    static bool warned_rho_floor = false;
    if (!warned_rho_floor) {
      core::log_warning("EOSTable::sound_speed: rho <= rho_floor encountered; "
                        "returning c_s=0 for vacuum/near-vacuum cells");
      warned_rho_floor = true;
    }
    return 0.0;
  }
  const double P_safe = std::max(P, 0.0);
  if (!(P_safe > 0.0)) {
    return 0.0;
  }

  const double T_min = T_grid_eV.front();
  const double T_max = T_grid_eV.back();
  const double P_min = pressure(rho, T_min);
  const double P_max = pressure(rho, T_max);

  double T_eval = T_min;
  if (!(std::isfinite(P_min) && std::isfinite(P_max))) {
    static bool warned_nonfinite_bounds = false;
    if (!warned_nonfinite_bounds) {
      core::log_warning("EOSTable::sound_speed encountered non-finite pressure bounds; "
                        "falling back to Tmin energy for gamma_eff");
      warned_nonfinite_bounds = true;
    }
  } else {
    const bool increasing = (P_max >= P_min);
    const bool below_min = increasing ? (P_safe <= P_min) : (P_safe >= P_min);
    const bool above_max = increasing ? (P_safe >= P_max) : (P_safe <= P_max);
    if (below_min) {
      T_eval = T_min;
    } else if (above_max) {
      T_eval = T_max;
    } else {
      double T_lo = T_min;
      double T_hi = T_max;
      for (int iter = 0; iter < 40; ++iter) {
        const double T_mid = 0.5 * (T_lo + T_hi);
        const double P_mid = pressure(rho, T_mid);
        if (!std::isfinite(P_mid)) {
          break;
        }
        if (increasing) {
          if (P_mid < P_safe) {
            T_lo = T_mid;
          } else {
            T_hi = T_mid;
          }
        } else {
          if (P_mid > P_safe) {
            T_lo = T_mid;
          } else {
            T_hi = T_mid;
          }
        }
      }
      T_eval = 0.5 * (T_lo + T_hi);
    }
  }

  const double e_internal = energy(rho, T_eval);
  if (!(e_internal > 0.0)) {
    // Shifted EOS reference: gamma_eff formula inapplicable, use isothermal fallback.
    static bool warned_fallback = false;
    if (!warned_fallback) {
      core::log_warning("EOSTable::sound_speed: e_internal <= 0 from table; "
                        "falling back to isothermal cs=sqrt(P/rho)");
      warned_fallback = true;
    }
    return std::sqrt(std::max(P_safe / rho, 0.0));
  }
  const double gamma_eff_raw = 1.0 + P_safe / (rho * e_internal);
  const double gamma_eff = std::clamp(gamma_eff_raw, 1.0, 10.0);
  if (gamma_eff != gamma_eff_raw) {
    static bool warned_gamma_clamp = false;
    if (!warned_gamma_clamp) {
      core::log_warning("EOSTable::sound_speed: gamma_eff clamped to [1,10]");
      warned_gamma_clamp = true;
    }
  }
  const double cs2 = gamma_eff * P_safe / rho;
  return std::sqrt(std::max(cs2, 0.0));
}

EOSTable build_sesame_ion_table(const EOSTable& total, const EOSTable& electron) {
  TENRYU_ASSERT(!total.empty() && !electron.empty(),
                "build_sesame_ion_table requires non-empty total/electron tables");
  TENRYU_ASSERT(total.P_table.size() == total.n_rho() * total.n_T() &&
                    total.e_table.size() == total.n_rho() * total.n_T(),
                "build_sesame_ion_table total table size mismatch");

  EOSTable ion;
  ion.rho_grid = total.rho_grid;
  ion.T_grid_eV = total.T_grid_eV;
  const std::size_t n = total.P_table.size();
  ion.P_table.resize(n);
  ion.e_table.resize(n);

  // Identical grids take the node-wise fast path so the difference is exact
  // (the interpolated path reproduces top-edge nodes only to ~1 ulp).
  const bool same_grid =
      total.rho_grid.size() == electron.rho_grid.size() &&
      total.T_grid_eV.size() == electron.T_grid_eV.size() &&
      std::equal(total.rho_grid.begin(), total.rho_grid.end(),
                 electron.rho_grid.begin()) &&
      std::equal(total.T_grid_eV.begin(), total.T_grid_eV.end(),
                 electron.T_grid_eV.begin());

  std::size_t negative_P_nodes = 0;
  std::size_t negative_e_nodes = 0;
  for (std::size_t j = 0; j < total.n_T(); ++j) {
    for (std::size_t i = 0; i < total.n_rho(); ++i) {
      const std::size_t k = total.flat_index(i, j);
      const double rho = total.rho_grid[i];
      const double T = total.T_grid_eV[j];
      const double P_e =
          same_grid ? electron.P_table[k] : electron.pressure(rho, T);
      const double e_e =
          same_grid ? electron.e_table[k] : electron.energy(rho, T);
      const double P_i = total.P_table[k] - P_e;
      const double e_i = total.e_table[k] - e_e;
      if (P_i < 0.0) {
        ++negative_P_nodes;
      }
      if (e_i < 0.0) {
        ++negative_e_nodes;
      }
      ion.P_table[k] = P_i;
      ion.e_table[k] = e_i;
    }
  }
  if (negative_P_nodes > 0 || negative_e_nodes > 0) {
    core::log_warning(
        "SESAME ion table (301 - 304) has negative nodes: P<0 at " +
        std::to_string(negative_P_nodes) + ", e<0 at " +
        std::to_string(negative_e_nodes) + " of " + std::to_string(n) +
        " nodes (energy references of 301/304 may differ; values pass through "
        "unclamped, monotonicity certification gates the inverse)");
  }

  ion.finalize();
  return ion;
}

EOSTablePair load_sesame(const std::string& filename,
                         const int mat_id,
                         const double zbar_hint) {
  const SesameData sesame = read_xsesame(filename, mat_id);

  EOSTable total = from_sesame_raw(*sesame.table_301_total);
  EOSTable electron;
  if (sesame.table_304_electron.has_value()) {
    electron = from_sesame_raw(*sesame.table_304_electron);
  } else {
    // Approximation for 304-missing materials: split total EOS into electron/ion
    // pieces with a constant zbar_hint ratio. This is a 1T fallback only.
    electron = total;
    const double z_clamped = std::max(zbar_hint, 0.0);
    const double frac = z_clamped / (1.0 + z_clamped);
    for (double& p : electron.P_table) {
      p *= frac;
    }
    for (double& e : electron.e_table) {
      e *= frac;
    }
    electron.finalize();
  }

  return {std::move(total), std::move(electron)};
}

EOSTablePair ionmix_eos_to_table_pair(const IonmixEOSData& eos, const double A) {
  if (!(eos.ntemp > 0 && eos.ndens > 0)) {
    throw std::runtime_error("ionmix_eos_to_table_pair requires ntemp>0 and ndens>0");
  }
  if (!(A > 0.0) || !std::isfinite(A)) {
    throw std::runtime_error("ionmix_eos_to_table_pair requires finite A>0");
  }

  const std::size_t n_rho = static_cast<std::size_t>(eos.ndens);
  const std::size_t n_T = static_cast<std::size_t>(eos.ntemp);
  if (n_rho > (std::numeric_limits<std::size_t>::max() / n_T)) {
    throw std::runtime_error("ionmix_eos_to_table_pair dimensions overflow");
  }
  const std::size_t n_points = n_rho * n_T;
  if (eos.temps_eV.size() != n_T || eos.numdens_cm3.size() != n_rho ||
      eos.P_i_cgs.size() != n_points || eos.P_e_cgs.size() != n_points ||
      eos.e_i_cgs.size() != n_points || eos.e_e_cgs.size() != n_points) {
    throw std::runtime_error("ionmix_eos_to_table_pair input size mismatch");
  }

  EOSTable total;
  EOSTable electron;
  total.rho_grid.resize(n_rho, 0.0);
  electron.rho_grid.resize(n_rho, 0.0);
  for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
    const double rho = eos.numdens_cm3[i_rho] * A * core::constants::proton_mass;
    if (!(rho > 0.0) || !std::isfinite(rho)) {
      throw std::runtime_error("ionmix_eos_to_table_pair encountered non-positive rho grid");
    }
    if (i_rho > 0 && !(rho > total.rho_grid[i_rho - 1])) {
      throw std::runtime_error("ionmix_eos_to_table_pair rho grid is not strictly increasing");
    }
    total.rho_grid[i_rho] = rho;
    electron.rho_grid[i_rho] = rho;
  }

  total.T_grid_eV = eos.temps_eV;
  electron.T_grid_eV = eos.temps_eV;
  for (std::size_t j_T = 0; j_T < n_T; ++j_T) {
    const double T = total.T_grid_eV[j_T];
    if (!(T > 0.0) || !std::isfinite(T)) {
      throw std::runtime_error("ionmix_eos_to_table_pair encountered non-positive T grid");
    }
    if (j_T > 0 && !(T > total.T_grid_eV[j_T - 1])) {
      throw std::runtime_error("ionmix_eos_to_table_pair T grid is not strictly increasing");
    }
  }

  total.P_table.resize(n_points, 0.0);
  total.e_table.resize(n_points, 0.0);
  electron.P_table.resize(n_points, 0.0);
  electron.e_table.resize(n_points, 0.0);
  for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
    for (std::size_t j_T = 0; j_T < n_T; ++j_T) {
      const std::size_t src = i_rho * n_T + j_T;      // IONMIX: density-major
      const std::size_t dst = j_T * n_rho + i_rho;    // EOSTable: temperature-major
      const double P_i = eos.P_i_cgs[src];
      const double P_e = eos.P_e_cgs[src];
      const double e_i = eos.e_i_cgs[src];
      const double e_e = eos.e_e_cgs[src];
      if (!std::isfinite(P_i) || !std::isfinite(P_e) ||
          !std::isfinite(e_i) || !std::isfinite(e_e)) {
        throw std::runtime_error("ionmix_eos_to_table_pair encountered non-finite EOS value");
      }
      total.P_table[dst] = P_i + P_e;
      total.e_table[dst] = e_i + e_e;
      electron.P_table[dst] = P_e;
      electron.e_table[dst] = e_e;
    }
  }

  total.finalize();
  electron.finalize();
  return {std::move(total), std::move(electron)};
}

EOSTableTriplet ionmix_eos_to_table_triplet(const IonmixEOSData& eos, const double A) {
  if (!(eos.ntemp > 0 && eos.ndens > 0)) {
    throw std::runtime_error("ionmix_eos_to_table_triplet requires ntemp>0 and ndens>0");
  }
  if (!(A > 0.0) || !std::isfinite(A)) {
    throw std::runtime_error("ionmix_eos_to_table_triplet requires finite A>0");
  }

  const std::size_t n_rho = static_cast<std::size_t>(eos.ndens);
  const std::size_t n_T = static_cast<std::size_t>(eos.ntemp);
  if (n_rho > (std::numeric_limits<std::size_t>::max() / n_T)) {
    throw std::runtime_error("ionmix_eos_to_table_triplet dimensions overflow");
  }
  const std::size_t n_points = n_rho * n_T;
  if (eos.temps_eV.size() != n_T || eos.numdens_cm3.size() != n_rho ||
      eos.P_i_cgs.size() != n_points || eos.P_e_cgs.size() != n_points ||
      eos.e_i_cgs.size() != n_points || eos.e_e_cgs.size() != n_points) {
    throw std::runtime_error("ionmix_eos_to_table_triplet input size mismatch");
  }

  EOSTable ion;
  EOSTable electron;
  EOSTable total;
  ion.rho_grid.resize(n_rho, 0.0);
  electron.rho_grid.resize(n_rho, 0.0);
  total.rho_grid.resize(n_rho, 0.0);
  for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
    const double rho = eos.numdens_cm3[i_rho] * A * core::constants::proton_mass;
    if (!(rho > 0.0) || !std::isfinite(rho)) {
      throw std::runtime_error("ionmix_eos_to_table_triplet encountered non-positive rho grid");
    }
    if (i_rho > 0 && !(rho > total.rho_grid[i_rho - 1])) {
      throw std::runtime_error(
          "ionmix_eos_to_table_triplet rho grid is not strictly increasing");
    }
    ion.rho_grid[i_rho] = rho;
    electron.rho_grid[i_rho] = rho;
    total.rho_grid[i_rho] = rho;
  }

  ion.T_grid_eV = eos.temps_eV;
  electron.T_grid_eV = eos.temps_eV;
  total.T_grid_eV = eos.temps_eV;
  for (std::size_t j_T = 0; j_T < n_T; ++j_T) {
    const double T = total.T_grid_eV[j_T];
    if (!(T > 0.0) || !std::isfinite(T)) {
      throw std::runtime_error("ionmix_eos_to_table_triplet encountered non-positive T grid");
    }
    if (j_T > 0 && !(T > total.T_grid_eV[j_T - 1])) {
      throw std::runtime_error("ionmix_eos_to_table_triplet T grid is not strictly increasing");
    }
  }

  ion.P_table.resize(n_points, 0.0);
  ion.e_table.resize(n_points, 0.0);
  electron.P_table.resize(n_points, 0.0);
  electron.e_table.resize(n_points, 0.0);
  total.P_table.resize(n_points, 0.0);
  total.e_table.resize(n_points, 0.0);
  for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
    for (std::size_t j_T = 0; j_T < n_T; ++j_T) {
      const std::size_t src = i_rho * n_T + j_T;   // IONMIX: density-major
      const std::size_t dst = j_T * n_rho + i_rho;  // EOSTable: temperature-major
      const double P_i = eos.P_i_cgs[src];
      const double P_e = eos.P_e_cgs[src];
      const double e_i = eos.e_i_cgs[src];
      const double e_e = eos.e_e_cgs[src];
      if (!std::isfinite(P_i) || !std::isfinite(P_e) ||
          !std::isfinite(e_i) || !std::isfinite(e_e)) {
        throw std::runtime_error("ionmix_eos_to_table_triplet encountered non-finite EOS value");
      }
      ion.P_table[dst] = P_i;
      ion.e_table[dst] = e_i;
      electron.P_table[dst] = P_e;
      electron.e_table[dst] = e_e;
      total.P_table[dst] = P_i + P_e;
      total.e_table[dst] = e_i + e_e;
    }
  }

  ion.finalize();
  electron.finalize();
  total.finalize();
  return {std::move(ion), std::move(electron), std::move(total)};
}

EOSTableTriplet build_power_law_te_tables(const double A,
                                          const double ideal_gas_gamma,
                                          const double f_erg_g,
                                          const double beta,
                                          const double mu_rho,
                                          const double gamma_p,
                                          const double step_D_erg_g_eV,
                                          const double step_Tc_eV,
                                          const double step_w_eV) {
  if (!(A > 0.0) || !std::isfinite(A)) {
    throw std::runtime_error("build_power_law_te_tables requires finite A>0");
  }
  if (!(ideal_gas_gamma > 1.0) || !std::isfinite(ideal_gas_gamma)) {
    throw std::runtime_error("build_power_law_te_tables requires finite ideal_gas_gamma>1");
  }
  if (!(f_erg_g > 0.0) || !std::isfinite(f_erg_g)) {
    throw std::runtime_error("build_power_law_te_tables requires finite f_erg_g>0");
  }
  if (!(beta > 0.0) || !std::isfinite(beta)) {
    throw std::runtime_error("build_power_law_te_tables requires finite beta>0");
  }
  if (!std::isfinite(mu_rho)) {
    throw std::runtime_error("build_power_law_te_tables requires finite mu_rho");
  }
  if (!(gamma_p > 1.0) || !std::isfinite(gamma_p)) {
    throw std::runtime_error("build_power_law_te_tables requires finite gamma_p>1");
  }
  if (!std::isfinite(step_D_erg_g_eV) || step_D_erg_g_eV < 0.0) {
    throw std::runtime_error(
        "build_power_law_te_tables requires finite step_D_erg_g_eV>=0");
  }
  if (step_D_erg_g_eV > 0.0 &&
      (!(step_Tc_eV > 0.0) || !std::isfinite(step_Tc_eV) ||
       !(step_w_eV > 0.0) || !std::isfinite(step_w_eV))) {
    throw std::runtime_error(
        "build_power_law_te_tables: step_D>0 requires step_Tc>0 and step_w>0");
  }

  constexpr std::size_t n_rho = 64;
  constexpr std::size_t n_T = 512;
  constexpr double rho_min = 0.02;
  constexpr double rho_max = 2.0;
  constexpr double T_min_eV = 0.05;
  constexpr double T_max_eV = 2000.0;

  EOSTable ion;
  EOSTable electron;
  EOSTable total;
  ion.rho_grid = make_log_uniform_grid(n_rho, rho_min, rho_max);
  electron.rho_grid = ion.rho_grid;
  total.rho_grid = ion.rho_grid;
  ion.T_grid_eV = make_log_uniform_grid(n_T, T_min_eV, T_max_eV);
  electron.T_grid_eV = ion.T_grid_eV;
  total.T_grid_eV = ion.T_grid_eV;

  const std::size_t n_points = n_rho * n_T;
  ion.P_table.assign(n_points, 0.0);
  ion.e_table.assign(n_points, 0.0);
  electron.P_table.assign(n_points, 0.0);
  electron.e_table.assign(n_points, 0.0);
  total.P_table.assign(n_points, 0.0);
  total.e_table.assign(n_points, 0.0);

  const double gm1_i = ideal_gas_gamma - 1.0;
  const double cv_i =
      core::constants::eV_to_erg / (A * core::constants::proton_mass * gm1_i);
  const double gm1_e = gamma_p - 1.0;
  for (std::size_t i_rho = 0; i_rho < n_rho; ++i_rho) {
    const double rho = ion.rho_grid[i_rho];
    const double rho_factor = std::pow(rho, -mu_rho);
    for (std::size_t j_T = 0; j_T < n_T; ++j_T) {
      const double T = ion.T_grid_eV[j_T];
      const std::size_t dst = j_T * n_rho + i_rho;
      const double e_i = cv_i * T;
      const double P_i = gm1_i * rho * e_i;
      double e_e = f_erg_g * std::pow(T, beta) * rho_factor;
      if (step_D_erg_g_eV > 0.0) {
        // Stable softplus: c_v gains a smooth step of height D across
        // (Tc, w); the energy integral is exact so the table generator and
        // the python reference ODE share the same closed form.
        const double x = (T - step_Tc_eV) / step_w_eV;
        const double softplus =
            std::max(x, 0.0) + std::log1p(std::exp(-std::abs(x)));
        e_e += step_D_erg_g_eV * step_w_eV * softplus;
      }
      const double P_e = gm1_e * rho * e_e;
      ion.e_table[dst] = e_i;
      ion.P_table[dst] = P_i;
      electron.e_table[dst] = e_e;
      electron.P_table[dst] = P_e;
      total.e_table[dst] = e_i + e_e;
      total.P_table[dst] = P_i + P_e;
    }
  }

  ion.finalize();
  electron.finalize();
  total.finalize();
  return {std::move(ion), std::move(electron), std::move(total)};
}

EOSTablePair load_ionmix(const std::string& filename) {
  // Deprecated ASCII-only fallback. Runtime/config paths should use
  // load_ionmix_binary_eos() + ionmix_eos_to_table_pair().
  // Minimal host-side reader for IONMIX v4 ASCII.
  // Assumed token layout (SPECIFICATION §IONMIX):
  //   n_T n_rho Z A
  //   T_grid[n_T]                        (eV)
  //   rho_grid[n_rho]                    (g/cc)
  //   Zbar[n_T*n_rho]
  //   e_ion[n_T*n_rho]
  //   e_electron[n_T*n_rho]
  //   P_ion[n_T*n_rho]
  //   P_electron[n_T*n_rho]
  // Optional opacity blocks (Planck/Rosseland) may follow and are ignored here.
  const std::vector<double> values = load_ascii_values(filename);
  TENRYU_ASSERT(values.size() >= 4, "IONMIX file missing header n_T n_rho Z A");

  std::size_t cursor = 0;
  const int n_T = static_cast<int>(values[cursor++]);
  const int n_rho = static_cast<int>(values[cursor++]);
  const double Z_header = values[cursor++];
  const double A_header = values[cursor++];
  TENRYU_ASSERT(n_T > 0 && n_rho > 0, "IONMIX shape must be positive");
  TENRYU_ASSERT(std::isfinite(Z_header) && std::isfinite(A_header),
                "IONMIX header Z/A must be finite");

  const std::size_t n_points =
      static_cast<std::size_t>(n_T) * static_cast<std::size_t>(n_rho);
  const std::size_t expected = 4ULL + static_cast<std::size_t>(n_T) +
                               static_cast<std::size_t>(n_rho) + 5ULL * n_points;
  TENRYU_ASSERT(values.size() >= expected,
                "IONMIX file does not contain enough values");

  EOSTable total;
  EOSTable electron;
  total.rho_grid.resize(static_cast<std::size_t>(n_rho));
  total.T_grid_eV.resize(static_cast<std::size_t>(n_T));
  electron.rho_grid.resize(static_cast<std::size_t>(n_rho));
  electron.T_grid_eV.resize(static_cast<std::size_t>(n_T));

  for (int j = 0; j < n_T; ++j) {
    const double T = values[cursor++];
    total.T_grid_eV[static_cast<std::size_t>(j)] = T;
    electron.T_grid_eV[static_cast<std::size_t>(j)] = T;
  }
  for (int i = 0; i < n_rho; ++i) {
    const double rho = values[cursor++];
    total.rho_grid[static_cast<std::size_t>(i)] = rho;
    electron.rho_grid[static_cast<std::size_t>(i)] = rho;
  }

  std::vector<double> zbar_table(n_points, 0.0);
  std::vector<double> e_ion(n_points, 0.0);
  std::vector<double> e_electron(n_points, 0.0);
  std::vector<double> P_ion(n_points, 0.0);
  std::vector<double> P_electron(n_points, 0.0);
  for (std::size_t k = 0; k < n_points; ++k) {
    zbar_table[k] = values[cursor++];
  }
  (void)zbar_table;
  for (std::size_t k = 0; k < n_points; ++k) {
    e_ion[k] = values[cursor++];
  }
  for (std::size_t k = 0; k < n_points; ++k) {
    e_electron[k] = values[cursor++];
  }
  for (std::size_t k = 0; k < n_points; ++k) {
    P_ion[k] = values[cursor++];
  }
  for (std::size_t k = 0; k < n_points; ++k) {
    P_electron[k] = values[cursor++];
  }

  total.P_table.resize(n_points);
  total.e_table.resize(n_points);
  electron.P_table.resize(n_points);
  electron.e_table.resize(n_points);
  for (std::size_t k = 0; k < n_points; ++k) {
    electron.P_table[k] = P_electron[k];
    electron.e_table[k] = e_electron[k];
    total.P_table[k] = P_ion[k] + P_electron[k];
    total.e_table[k] = e_ion[k] + e_electron[k];
  }

  total.finalize();
  electron.finalize();
  return {std::move(total), std::move(electron)};
}

}  // namespace tenryu::materials
