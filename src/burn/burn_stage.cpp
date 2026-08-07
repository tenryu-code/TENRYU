#include "burn/burn_stage.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "burn/burn_stage_gpu.cuh"
#include "burn/deposition.cuh"
#include "burn/neutron_heating.hpp"
#include "burn/screening.hpp"
#include "core/error.hpp"

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

BurnStageResult compute_burn_step_1d_host(const BurnStageInputs& in,
                                          const BurnStageParams& p,
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

  BurnStageResult result;
  result.dt_limit_s = std::numeric_limits<double>::infinity();
  double dt_limit_subcycle = std::numeric_limits<double>::infinity();

  std::vector<double> nh_emit;
  if (p.neutron_heating) {
    nh_emit.assign(n * static_cast<std::size_t>(kNumNeutronLines), 0.0);
  }

  int first = -1;
  int last = -1;
  for (int c = 0; c < n_cells; ++c) {
    double fuel_frac = 0.0;
    for (int i = 0; i < in.n_fuel_mat; ++i) {
      const int m = in.fuel_mat[i];
      fuel_frac += in.volFrac[c * in.n_mat + m];
    }
    if (fuel_frac > p.vf_threshold) {
      if (first < 0) {
        first = c;
      }
      last = c;
    }
  }

  if (first < 0) {
    return result;
  }

  result.burn_region_first = first;
  result.burn_region_last = last;

  std::vector<double> r_center(n, 0.0);
  std::vector<double> dr(n, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    r_center[static_cast<std::size_t>(c)] =
        0.5 * (in.r_node[c] + in.r_node[c + 1]);
    dr[static_cast<std::size_t>(c)] = in.r_node[c + 1] - in.r_node[c];
  }

  const double R_b = in.r_node[last + 1];

  std::vector<double> S_out(n, 0.0);
  S_out[static_cast<std::size_t>(last)] =
      0.5 * in.rho[last] * dr[static_cast<std::size_t>(last)];
  // v1 uses one single fuel interval: threshold gaps between first and last
  // stay in the burn region.
  for (int c = last - 1; c >= first; --c) {
    S_out[static_cast<std::size_t>(c)] =
        S_out[static_cast<std::size_t>(c + 1)] +
        0.5 * in.rho[c] * dr[static_cast<std::size_t>(c)] +
        0.5 * in.rho[c + 1] * dr[static_cast<std::size_t>(c + 1)];
  }

  for (int c = first; c <= last; ++c) {
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
    int M_req = 0;
    const int M = burn_network_step(local_n, T_keV, p.dt_s, p.channels,
                                    p.eps_deplete, p.subcycle_max, counts,
                                    screen, &M_req);
    result.max_substeps_required =
        std::max(result.max_substeps_required, M_req);
    if (M_req > p.subcycle_max) {
      ++result.subcycle_saturated_cells;
      const double dt_shrink =
          0.9 * p.dt_s * static_cast<double>(p.subcycle_max) /
          static_cast<double>(M_req);
      if (dt_shrink > 0.0) {
        dt_limit_subcycle = std::min(dt_limit_subcycle, dt_shrink);
        result.dt_limit_s = std::min(result.dt_limit_s, dt_shrink);
      }
    }
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

    const double rate_dt = counts[kDT] / p.dt_s;
    const double rate_dd = counts[kDDn] / p.dt_s;
    const double vol_c = in.vol[c];
    const double vr_c = (in.v_r != nullptr) ? in.v_r[c] : 0.0;
    const double vr2_c = vr_c * vr_c;
    result.w_dt += rate_dt * vol_c;
    result.w_dd += rate_dd * vol_c;
    result.wTi_dt += rate_dt * vol_c * in.Ti_eV[c];
    result.wTi_dd += rate_dd * vol_c * in.Ti_eV[c];
    result.wvr2_dt += rate_dt * vol_c * vr2_c;
    result.wvr2_dd += rate_dd * vol_c * vr2_c;

    const double rc = r_center[static_cast<std::size_t>(c)];
    const double u = clamp01(rc / R_b);
    const double span = R_b - rc;
    const double rho_bar =
        (span > 1.0e-9 * R_b)
            ? (S_out[static_cast<std::size_t>(c)] / span)
            : in.rho[c];
    const double Te_keV_c = in.Te_eV[c] * 1.0e-3;
    const double Ti_keV_c =
        ((in.Ti_eV[c] > 0.0) ? in.Ti_eV[c] : 0.0) * 1.0e-3;
    const double rho_lam_alpha = alpha_rho_lambda(Te_keV_c, in.rho[c]);

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
      if (p.neutron_heating && E_rel_neutron > 0.0) {
        const int line = (k == kDT) ? 0 : 1;
        nh_emit[static_cast<std::size_t>(c) *
                    static_cast<std::size_t>(kNumNeutronLines) +
                static_cast<std::size_t>(line)] += E_rel_neutron;
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
        const double rho_lam_s =
            rho_lam_alpha * range_scale_factor(sp, E_MeV);
        const double tau =
            (rho_lam_s > 0.0) ? (R_b * rho_bar / rho_lam_s) : 1.0e30;
        const double f_dep = point_sphere_deposited_fraction(u, tau);
        const double f_ion =
            p.use_fraley_partition
                ? fraley_ion_fraction(Te_keV_c)
                : partition_f_ion(table, product_slot(k, i), Te_keV_c,
                                  Ti_keV_c, ne_c);

        dE_i[static_cast<std::size_t>(c)] += f_ion * f_dep * E_rel;
        dE_e[static_cast<std::size_t>(c)] +=
            (1.0 - f_ion) * f_dep * E_rel;
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

  if (p.neutron_heating && result.released_neutron > 0.0) {
    NeutronHeatingParams nh_p;
    nh_p.n_mu = p.neutron_heating_n_mu;
    nh_p.use_fraley_partition = p.use_fraley_partition;
    const NeutronHeatingResult nh = deposit_neutron_heating_1d(
        n_cells, in.r_node, in.rho, in.Te_eV, in.Ti_eV, in.zbar, in.A_eff,
        burn_y, nh_emit, nh_p, table, dE_e, dE_i);
    result.nh_dep_e = nh.dep_e;
    result.nh_dep_i = nh.dep_i;
    result.nh_degraded = nh.degraded;
    result.nh_escaped = nh.escaped;
    result.nh_conservation_resid = nh.conservation_resid;
    result.dep_e += nh.dep_e;
    result.dep_i += nh.dep_i;
    result.esc_neutron = result.released_neutron - nh.dep_e - nh.dep_i;
    // Refresh the per-cell source diagnostics and the explicit-source dt
    // guard with the neutron contribution included — preheated cells
    // OUTSIDE the burn region now carry sources too.
    result.dt_limit_s = std::numeric_limits<double>::infinity();
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t cs = static_cast<std::size_t>(c);
      if (!(in.vol[c] > 0.0)) {
        continue;
      }
      Qe_diag[cs] = dE_e[cs] / (in.vol[c] * p.dt_s);
      Qi_diag[cs] = dE_i[cs] / (in.vol[c] * p.dt_s);
      const double P_dep_c = (dE_e[cs] + dE_i[cs]) / p.dt_s;
      if (P_dep_c > 0.0) {
        const double e_cell =
            in.rho[c] * in.vol[c] * std::max(in.ee[c] + in.ei[c], 0.0);
        if (e_cell > 0.0) {
          const double cand = p.explicit_source_limit * e_cell / P_dep_c;
          result.dt_limit_s = std::min(result.dt_limit_s, cand);
        }
      }
    }
    result.dt_limit_s = std::min(result.dt_limit_s, dt_limit_subcycle);
  }

  if (result.subcycle_saturated_cells > 0) {
    static int warned_subcycle_saturation = 0;
    ++warned_subcycle_saturation;
    if (warned_subcycle_saturation == 1 ||
        warned_subcycle_saturation % 100 == 0) {
      core::log_warning(
          "burn: subcycle_max saturated in " +
          std::to_string(result.subcycle_saturated_cells) +
          " cells (required " +
          std::to_string(result.max_substeps_required) +
          " substeps); eps_deplete accuracy not met this step, burn dt limit "
          "engaged");
    }
  }

  return result;
}

BurnStageResult compute_burn_step_1d(const BurnStageInputs& in,
                                     const BurnStageParams& p,
                                     const PartitionTable& table,
                                     std::vector<double>& burn_y,
                                     std::vector<double>& dE_e,
                                     std::vector<double>& dE_i,
                                     std::vector<double>& rate_diag,
                                     std::vector<double>& Qe_diag,
                                     std::vector<double>& Qi_diag,
                                     std::vector<double>* S_birth) {
  const char* const host_stage_env = std::getenv("TENRYU_BURN_HOST_STAGE");
  const bool force_host =
      host_stage_env != nullptr && std::strcmp(host_stage_env, "1") == 0;
  if (force_host || in.n_cells <= 0) {
    return compute_burn_step_1d_host(in, p, table, burn_y, dE_e, dE_i,
                                     rate_diag, Qe_diag, Qi_diag, S_birth);
  }

  std::vector<double> nh_emit;
  double dt_limit_subcycle = std::numeric_limits<double>::infinity();
  unsigned int screening_warning_flags = 0U;
  BurnStageResult result = compute_burn_step_1d_device_stage(
      in, p, table, burn_y, dE_e, dE_i, rate_diag, Qe_diag, Qi_diag,
      S_birth, nh_emit, dt_limit_subcycle, screening_warning_flags);
  burn_screening_emit_warnings(screening_warning_flags);

  if (result.subcycle_saturated_cells > 0) {
    static int warned_subcycle_saturation = 0;
    ++warned_subcycle_saturation;
    if (warned_subcycle_saturation == 1 ||
        warned_subcycle_saturation % 100 == 0) {
      core::log_warning(
          "burn: subcycle_max saturated in " +
          std::to_string(result.subcycle_saturated_cells) +
          " cells (required " +
          std::to_string(result.max_substeps_required) +
          " substeps); eps_deplete accuracy not met this step, burn dt limit "
          "engaged");
    }
  }

  return result;
}

}  // namespace tenryu::burn
