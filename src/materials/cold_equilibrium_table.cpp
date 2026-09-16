#include "materials/cold_equilibrium_table.hpp"

#include <algorithm>
#include <cmath>

#include "core/error.hpp"
#include "materials/eos_table.hpp"

namespace tenryu::materials {

ColdEquilibriumView ColdEquilibriumTable::view() const {
  ColdEquilibriumView out{};
  if (empty()) {
    return out;
  }
  out.v_knots = v_knots.data();
  out.PN_knots = PN_knots.data();
  out.C0_knots = C0_knots.data();
  out.n_knots = static_cast<int>(v_knots.size());
  out.v0 = 1.0 / params.rho0;
  out.P0 = params.P0;
  out.K0 = params.K0;
  out.T_star = params.T_star;
  out.T_a = params.T_begin_fraction * params.T_star;
  out.log_rho0 = std::log(params.rho0);
  out.log_core = std::log(params.density_core_ratio);
  out.log_outer = std::log(params.density_outer_ratio);
  out.max_iterations = params.max_iterations;
  return out;
}

ColdEquilibriumTable build_cold_equilibrium_table(const EOSTable& ion,
                                                  const EOSTable& electron,
                                                  const ColdEquilibriumParams& params) {
  TENRYU_ASSERT(params.rho0 > 0.0, "cold-equilibrium rho0 must be positive");
  TENRYU_ASSERT(params.K0 > 0.0, "cold-equilibrium K0 must be positive");
  TENRYU_ASSERT(params.T_star > 0.0, "cold-equilibrium T_star must be positive");
  TENRYU_ASSERT(params.T_begin_fraction > 0.0 && params.T_begin_fraction < 1.0,
                "cold-equilibrium T_begin_fraction must lie in (0, 1)");
  TENRYU_ASSERT(params.density_core_ratio > 1.0 &&
                    params.density_core_ratio < params.density_outer_ratio,
                "cold-equilibrium density ratios must satisfy 1 < core < outer");
  TENRYU_ASSERT(!ion.empty() && !electron.empty(),
                "cold-equilibrium requires non-empty ion and electron tables");

  const double rho_lo = params.rho0 / (params.density_outer_ratio * 1.05);
  const double rho_hi = params.rho0 * params.density_outer_ratio * 1.05;
  std::vector<double> densities{params.rho0};
  double below = 0.0;
  double above = 0.0;
  for (const double rho : ion.rho_grid) {
    if (rho >= rho_lo && rho <= rho_hi) {
      densities.push_back(rho);
    }
    if (rho < rho_lo && rho > below) {
      below = rho;
    }
    if (rho > rho_hi && (above == 0.0 || rho < above)) {
      above = rho;
    }
  }
  if (below > 0.0) {
    densities.push_back(below);
  }
  if (above > 0.0) {
    densities.push_back(above);
  }

  ColdEquilibriumTable out;
  out.params = params;
  out.v_knots.reserve(densities.size());
  for (const double rho : densities) {
    out.v_knots.push_back(1.0 / rho);
  }
  std::sort(out.v_knots.begin(), out.v_knots.end());
  // Merge knots that coincide up to round-off (relative spacing below 1e-9):
  // the log-linear segment slope (P_b - P_a) / ln(v_b / v_a) of such a pair is
  // ill-defined. The reference knot v0 is always kept.
  {
    const double v0_ref = 1.0 / params.rho0;
    std::vector<double> merged;
    merged.reserve(out.v_knots.size());
    for (const double v : out.v_knots) {
      if (!merged.empty() && std::abs(v - merged.back()) <= 1.0e-9 * merged.back()) {
        if (v == v0_ref) {
          merged.back() = v;  // keep the exact reference knot
        }
        continue;
      }
      merged.push_back(v);
    }
    out.v_knots.swap(merged);
  }

  out.PN_knots.resize(out.v_knots.size());
  out.C0_knots.resize(out.v_knots.size());
  for (std::size_t k = 0; k < out.v_knots.size(); ++k) {
    const double rho = 1.0 / out.v_knots[k];
    out.PN_knots[k] = ion.pressure(rho, params.Ti0) + electron.pressure(rho, params.Te0);
  }

  const double v0 = 1.0 / params.rho0;
  const auto it0 = std::find(out.v_knots.begin(), out.v_knots.end(), v0);
  TENRYU_ASSERT(it0 != out.v_knots.end(), "cold-equilibrium reference knot not found");
  const std::size_t k0 = static_cast<std::size_t>(it0 - out.v_knots.begin());
  out.C0_knots[k0] = 0.0;
  for (std::size_t k = k0; k + 1 < out.v_knots.size(); ++k) {
    const double b = (out.PN_knots[k + 1] - out.PN_knots[k]) /
                     std::log(out.v_knots[k + 1] / out.v_knots[k]);
    out.C0_knots[k + 1] = out.C0_knots[k] +
                           cold_segment_integral(out.v_knots[k], out.PN_knots[k], b,
                                                 out.v_knots[k + 1], v0, params.P0, params.K0);
  }
  for (std::size_t k = k0; k > 0; --k) {
    const double b = (out.PN_knots[k] - out.PN_knots[k - 1]) /
                     std::log(out.v_knots[k] / out.v_knots[k - 1]);
    out.C0_knots[k - 1] = out.C0_knots[k] -
                           cold_segment_integral(out.v_knots[k - 1], out.PN_knots[k - 1], b,
                                                 out.v_knots[k], v0, params.P0, params.K0);
  }
  return out;
}

}  // namespace tenryu::materials
