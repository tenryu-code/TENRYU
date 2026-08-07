#include "burn/burn_stage_2d.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include "burn/deposition.cuh"
#include "burn/screening.hpp"

namespace tenryu::burn {
namespace {

constexpr double kProtonMassG = 1.6726219e-24;

int product_slot(const int reaction, const int product_index) {
  constexpr int slots[kNumReactions][2] = {
      {0, -1}, {1, 2}, {3, -1}, {4, 5}};
  return slots[reaction][product_index];
}

double clamp01(const double x) {
  return (x < 0.0) ? 0.0 : ((x > 1.0) ? 1.0 : x);
}

}  // namespace

BurnStage2DResult compute_burn_step_2d(const BurnStage2DInputs& in,
                                       const BurnStage2DParams& p,
                                       const PartitionTable& table,
                                       std::vector<double>& burn_y,
                                       std::vector<double>& dE_e,
                                       std::vector<double>& dE_i,
                                       std::vector<double>& rate_diag,
                                       std::vector<double>& Qe_diag,
                                       std::vector<double>& Qi_diag,
                                       std::vector<double>* S_birth) {
  const int n_cells = in.n_cells;
  const std::size_t n = static_cast<std::size_t>(n_cells);
  dE_e.assign(n, 0.0);
  dE_i.assign(n, 0.0);
  rate_diag.assign(n, 0.0);
  Qe_diag.assign(n, 0.0);
  Qi_diag.assign(n, 0.0);
  if (p.scheme == 1 && S_birth != nullptr) {
    S_birth->assign(6U * n, 0.0);
  }

  BurnStage2DResult result;
  result.dt_limit_s = std::numeric_limits<double>::infinity();

  // Owned window (Option C): (0, 0) = full range for serial and 1D
  // -replicated callers; the windowed loop leaves far-region inventories
  // and dE entries untouched (zeros), so the downstream sums are owned
  // partials by construction.
  const int c_lo = (in.c_end > in.c_begin) ? in.c_begin : 0;
  const int c_hi = (in.c_end > in.c_begin) ? in.c_end : n_cells;
  for (int c = c_lo; c < c_hi; ++c) {
    double fuel_frac = 0.0;
    for (int i = 0; i < in.n_fuel_mat; ++i) {
      const int m = in.fuel_mat[i];
      fuel_frac += in.volFrac[c * in.n_mat + m];
    }
    if (!(fuel_frac > p.vf_threshold)) {
      continue;
    }
    if (in.rho[c] <= 0.0) {
      continue;
    }

    const double T_keV = in.Ti_eV[c] * 1.0e-3;
    if (T_keV < p.T_floor_keV) {
      continue;
    }

    double local_n[kNumSpecies];
    for (int s = 0; s < kNumSpecies; ++s) {
      local_n[s] = burn_y[static_cast<std::size_t>(c * kNumSpecies + s)] *
                   in.rho[c];
    }

    const double ne_c = in.zbar[c] * in.rho[c] / (in.A_eff[c] * kProtonMassG);
    double F[kNumReactions];
    const double* screen = nullptr;
    const ScreeningMode screening_mode =
        static_cast<ScreeningMode>(p.screening_mode);
    if (screening_mode != ScreeningMode::kNone) {
      burn_screening_factors(screening_mode, in.Ti_eV[c], in.Te_eV[c], ne_c,
                             local_n, F);
      screen = F;
    }

    double counts[kNumReactions];
    const int M = burn_network_step(local_n, T_keV, p.dt_s, p.channels,
                                    p.eps_deplete, p.subcycle_max, counts,
                                    screen);
    result.max_substeps = std::max(result.max_substeps, M);

    for (int s = 0; s < kNumSpecies; ++s) {
      burn_y[static_cast<std::size_t>(c * kNumSpecies + s)] =
          local_n[s] / in.rho[c];
    }

    bool any_counts = false;
    for (int k = 0; k < kNumReactions; ++k) {
      any_counts = any_counts || (counts[k] != 0.0);
    }
    if (!any_counts) {
      continue;
    }

    const double Te_keV_c = in.Te_eV[c] * 1.0e-3;
    const double Ti_keV_c =
        ((in.Ti_eV[c] > 0.0) ? in.Ti_eV[c] : 0.0) * 1.0e-3;

    for (int k = 0; k < kNumReactions; ++k) {
      if (!(counts[k] > 0.0)) {
        continue;
      }

      const double E_rel_neutron = counts[k] * in.vol[c] * neutron_MeV(k) *
                                   kMeVToErg;
      result.released_neutron += E_rel_neutron;
      const double n_reactions = counts[k] * in.vol[c];
      if (k == kDT) {
        result.n_neutrons_dt += n_reactions;
      } else if (k == kDDn) {
        result.n_neutrons_dd += n_reactions;
      }

      for (int i = 0; i < 2; ++i) {
        const int sp = charged_species(k, i);
        if (sp < 0) {
          continue;
        }

        const double E_MeV = charged_MeV(k, i);
        const double E_rel = counts[k] * in.vol[c] * E_MeV * kMeVToErg;
        result.released_charged += E_rel;
        if (p.scheme == 1) {
          if (S_birth != nullptr) {
            const int slot = product_slot(k, i);
            if (slot >= 0) {
              (*S_birth)[static_cast<std::size_t>(slot) * n +
                         static_cast<std::size_t>(c)] += counts[k] / p.dt_s;
            }
          }
          continue;
        }
        const double f_ion =
            p.use_fraley_partition
                ? fraley_ion_fraction(Te_keV_c)
                : partition_f_ion(table, product_slot(k, i), Te_keV_c,
                                  Ti_keV_c, ne_c);
        dE_i[static_cast<std::size_t>(c)] += f_ion * E_rel;
        dE_e[static_cast<std::size_t>(c)] += (1.0 - f_ion) * E_rel;
      }
    }

    const double rate_total = (counts[0] + counts[1] + counts[2] + counts[3]) /
                              p.dt_s;
    rate_diag[static_cast<std::size_t>(c)] = rate_total;
    Qe_diag[static_cast<std::size_t>(c)] =
        dE_e[static_cast<std::size_t>(c)] / (in.vol[c] * p.dt_s);
    Qi_diag[static_cast<std::size_t>(c)] =
        dE_i[static_cast<std::size_t>(c)] / (in.vol[c] * p.dt_s);

    const double P_dep_c =
        (dE_e[static_cast<std::size_t>(c)] +
         dE_i[static_cast<std::size_t>(c)]) /
        p.dt_s;
    if (P_dep_c > 0.0) {
      const double e_cell =
          in.rho[c] * in.vol[c] * std::max(in.ee[c] + in.ei[c], 0.0);
      if (e_cell > 0.0) {
        const double cand = p.explicit_source_limit * e_cell / P_dep_c;
        result.dt_limit_s = std::min(result.dt_limit_s, cand);
      }
    }
  }

  for (int c = 0; c < n_cells; ++c) {
    result.dep_e += dE_e[static_cast<std::size_t>(c)];
  }
  for (int c = 0; c < n_cells; ++c) {
    result.dep_i += dE_i[static_cast<std::size_t>(c)];
  }
  result.esc_charged =
      (p.scheme == 1) ? 0.0 : (result.released_charged - result.dep_e - result.dep_i);
  result.esc_neutron = result.released_neutron;

  return result;
}

}  // namespace tenryu::burn
