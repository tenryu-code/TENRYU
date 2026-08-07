#include "radiation/energy_budget.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

double total_internal_electron(const core::State& s) {
  std::vector<double> rho(s.rho.size(), 0.0);
  std::vector<double> ee(s.ee.size(), 0.0);
  std::vector<double> vol(s.vol.size(), 0.0);
  s.rho.copy_to_host(rho.data());
  s.ee.copy_to_host(ee.data());
  s.vol.copy_to_host(vol.data());

  long double sum = 0.0L;
  for (std::size_t i = 0; i < rho.size(); ++i) {
    sum += static_cast<long double>(rho[i]) * static_cast<long double>(ee[i]) *
           static_cast<long double>(vol[i]);
  }
  return static_cast<double>(sum);
}

double total_internal_ion(const core::State& s) {
  std::vector<double> rho(s.rho.size(), 0.0);
  std::vector<double> ei(s.ei.size(), 0.0);
  std::vector<double> vol(s.vol.size(), 0.0);
  s.rho.copy_to_host(rho.data());
  s.ei.copy_to_host(ei.data());
  s.vol.copy_to_host(vol.data());

  long double sum = 0.0L;
  for (std::size_t i = 0; i < rho.size(); ++i) {
    sum += static_cast<long double>(rho[i]) * static_cast<long double>(ei[i]) *
           static_cast<long double>(vol[i]);
  }
  return static_cast<double>(sum);
}

double total_radiation_field(const core::State& s) {
  std::vector<double> rad(s.rad_E.size(), 0.0);
  std::vector<double> vol(s.vol.size(), 0.0);
  rad.assign(s.rad_E.size(), 0.0);
  s.rad_E.copy_to_host(rad.data());
  s.vol.copy_to_host(vol.data());

  if (!s.vol.empty()) {
    TENRYU_ASSERT(s.rad_E.size() % s.vol.size() == 0,
                  "total_radiation_field requires rad_E size divisible by volume size");
  }
  const int n_groups = (s.vol.empty() ? 1 : static_cast<int>(s.rad_E.size() / s.vol.size()));
  long double sum = 0.0L;
  for (std::size_t c = 0; c < vol.size(); ++c) {
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = c * static_cast<std::size_t>(n_groups) + static_cast<std::size_t>(g);
      sum += static_cast<long double>(rad[idx]) * static_cast<long double>(vol[c]);
    }
  }
  return static_cast<double>(sum);
}

}  // namespace

// Deprecated legacy helper: kept for compatibility, not used by the current
// coupling driver energy-budget path.
EnergyBudget compute_energy_budget(const core::State& before,
                                   const core::State& after,
                                   const double E_marshak_in,
                                   const double E_rad_esc,
                                   const double E_floor,
                                   const double E_safety) {
  EnergyBudget out;
  const double Eint_e_before = total_internal_electron(before);
  const double Eint_e_after = total_internal_electron(after);
  const double Eint_i_before = total_internal_ion(before);
  const double Eint_i_after = total_internal_ion(after);
  const double Erad_before = total_radiation_field(before);
  const double Erad_after = total_radiation_field(after);

  out.dE_int_e = Eint_e_after - Eint_e_before;
  out.dE_int_i = Eint_i_after - Eint_i_before;
  out.dE_kin = 0.0;
  out.dE_rad = Erad_after - Erad_before;
  out.E_marshak_in = E_marshak_in;
  out.E_rad_esc = E_rad_esc;
  out.E_floor = E_floor;
  out.E_safety = E_safety;

  const double lhs = out.dE_int_e + out.dE_int_i + out.dE_kin + out.dE_rad;
  const double rhs = out.E_marshak_in - out.E_rad_esc + out.E_floor + out.E_safety;
  const double denom = std::max({std::abs(Eint_e_before + Eint_i_before + Erad_before),
                                 std::abs(out.E_marshak_in),
                                 1.0e-20});
  out.epsilon_budget = std::abs(lhs - rhs) / denom;
  return out;
}

}  // namespace tenryu::radiation
