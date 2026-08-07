#pragma once

#include "burn/reactivity.cuh"
#include "core/macros.hpp"

namespace tenryu::burn {

struct BurnChannels {
  bool dt;
  bool dd;
  bool d3he;
};

TENRYU_HOST_DEVICE inline void burn_reaction_rates(
    const double n[kNumSpecies], const double sv1, const double sv2,
    const double sv3, const double sv4, double r[kNumReactions]) {
  r[kDT] = sym_factor(kDT) * n[kD] * n[kT] * sv1;
  r[kDDp] = sym_factor(kDDp) * n[kD] * n[kD] * sv2;
  r[kDDn] = sym_factor(kDDn) * n[kD] * n[kD] * sv3;
  r[kD3He] = sym_factor(kD3He) * n[kD] * n[kHe3] * sv4;
}

TENRYU_HOST_DEVICE inline void burn_stoich_delta_from_events(
    const double events[kNumReactions], double dn[kNumSpecies]) {
  for (int s = 0; s < kNumSpecies; ++s) {
    dn[s] = 0.0;
    for (int k = 0; k < kNumReactions; ++k) {
      dn[s] += stoich(k, s) * events[k];
    }
  }
}

TENRYU_HOST_DEVICE inline double burn_scale_back_theta(
    const double n[kNumSpecies], const double dn[kNumSpecies]) {
  double theta = 1.0;
  for (int s = 0; s < kNumSpecies; ++s) {
    if (dn[s] < 0.0 && n[s] + dn[s] < 0.0) {
      const double theta_s = n[s] / (-dn[s]);
      if (theta_s < theta) {
        theta = theta_s;
      }
    }
  }
  return theta;
}

// Per-cell frozen-temperature depletion step (design sections 3.1, 6.1):
//  - reactivities evaluated ONCE from T_keV (frozen over the step),
//  - M = clamp(ceil(dt * max_rel_rate / eps_deplete), 1, subcycle_max)
//    substeps, RK2 (explicit midpoint),
//  - deterministic scale-back positivity limiter: if a candidate substep
//    would drive any consumed species negative, scale the WHOLE substep's
//    rates by theta = min_s( n_s / (-dn_s) ) over violating species
//    (theta in [0,1]); reaction counts are scaled first, and inventories are
//    rebuilt from the ledgered events. The counts ledger is the primary object
//    downstream energy bookkeeping uses; the stoichiometric identity holds up
//    to one FP addition rounding per substep per species and the sub-ulp zero
//    clamp.
// Optional screen[kNumReactions] multiplies the frozen reactivities; nullptr
// is the v1 path. Outputs: n[] updated in place; counts[kNumReactions] =
// reaction events per cm^3 accumulated over the step (RK2-consistent:
// counts_k += h*r_k(midpoint), the identical FP products used for the
// inventory update); returns substeps used (int).
TENRYU_HOST_DEVICE inline int burn_network_step(
    double n[kNumSpecies], double T_keV, double dt_s, BurnChannels ch,
    double eps_deplete, int subcycle_max, double counts[kNumReactions],
    const double* screen = nullptr, int* substeps_required = nullptr) {
  for (int k = 0; k < kNumReactions; ++k) {
    counts[k] = 0.0;
  }

  if (!(T_keV > 0.0) || (!(ch.dt) && !(ch.dd) && !(ch.d3he)) || !(dt_s > 0.0)) {
    if (substeps_required != nullptr) {
      *substeps_required = 0;
    }
    return 0;
  }

  const double s0 = screen ? screen[kDT] : 1.0;
  const double s1 = screen ? screen[kDDp] : 1.0;
  const double s2 = screen ? screen[kDDn] : 1.0;
  const double s3 = screen ? screen[kD3He] : 1.0;
  const double sv1 = ch.dt ? bosch_hale_sv(kDT, T_keV) * s0 : 0.0;
  const double sv2 = ch.dd ? bosch_hale_sv(kDDp, T_keV) * s1 : 0.0;
  const double sv3 = ch.dd ? bosch_hale_sv(kDDn, T_keV) * s2 : 0.0;
  const double sv4 = ch.d3he ? bosch_hale_sv(kD3He, T_keV) * s3 : 0.0;

  // Substep controller: resolve every REACTANT of an enabled channel —
  // consumption AND growth. A trace reactant growing fast (bred T feeding
  // the DT channel) needs resolution even though it is net-produced; pure
  // products (He4, p; He3 with the D3He channel off) do not feed back and
  // are integrated accurately by resolving their sources. Scale floor
  // 1e-9*n_tot keeps near-empty reactants from demanding infinite M while
  // still forcing subcycle_max when they grow from ~zero. Controlled on
  // max(gross production, gross consumption) per reactant; the net rate hides
  // balanced breeding/consumption turnover (AI review k14 B-2, 2026-07-26).
  double r0[kNumReactions];
  burn_reaction_rates(n, sv1, sv2, sv3, sv4, r0);
  double n_tot = 0.0;
  for (int s = 0; s < kNumSpecies; ++s) {
    n_tot += n[s];
  }
  const double n_floor = 1.0e-9 * n_tot;
  const bool reactant[kNumSpecies] = {true, ch.dt, ch.d3he, false, false};
  double max_rel = 0.0;
  for (int s = 0; s < kNumSpecies; ++s) {
    if (!reactant[s]) {
      continue;
    }
    double production = 0.0;
    double consumption = 0.0;
    for (int k = 0; k < kNumReactions; ++k) {
      const double contrib = stoich(k, s) * r0[k];
      if (contrib > 0.0) {
        production += contrib;
      } else {
        consumption -= contrib;
      }
    }
    const double mag = (production > consumption) ? production : consumption;
    if (mag > 0.0) {
      const double denom_floor = (n[s] > n_floor) ? n[s] : n_floor;
      const double denom = (denom_floor > 1.0) ? denom_floor : 1.0;
      const double rel = mag / denom;
      if (rel > max_rel) {
        max_rel = rel;
      }
    }
  }

  const double M_real = ceil(dt_s * max_rel / eps_deplete);
  int M = 1;
  if (M_real >= static_cast<double>(subcycle_max)) {
    M = subcycle_max;
  } else if (M_real > 1.0) {
    M = static_cast<int>(M_real);
  }
  if (substeps_required != nullptr) {
    const double m_req =
        (M_real > 1.0) ? ((M_real < 1.0e9) ? M_real : 1.0e9) : 1.0;
    *substeps_required = static_cast<int>(m_req);
  }

  const double h = dt_s / static_cast<double>(M);
  for (int m = 0; m < M; ++m) {
    double r_a[kNumReactions];
    burn_reaction_rates(n, sv1, sv2, sv3, sv4, r_a);

    double half_events[kNumReactions];
    for (int k = 0; k < kNumReactions; ++k) {
      half_events[k] = 0.5 * h * r_a[k];
    }
    double dn_half[kNumSpecies];
    burn_stoich_delta_from_events(half_events, dn_half);
    const double theta_half = burn_scale_back_theta(n, dn_half);

    double n_mid[kNumSpecies];
    for (int s = 0; s < kNumSpecies; ++s) {
      n_mid[s] = n[s] + theta_half * dn_half[s];
      if (n_mid[s] < 0.0) {
        n_mid[s] = 0.0;
      }
    }

    double r_m[kNumReactions];
    burn_reaction_rates(n_mid, sv1, sv2, sv3, sv4, r_m);
    double events[kNumReactions];
    for (int k = 0; k < kNumReactions; ++k) {
      events[k] = h * r_m[k];
    }
    double dn_trial[kNumSpecies];
    burn_stoich_delta_from_events(events, dn_trial);
    const double theta = burn_scale_back_theta(n, dn_trial);

    double events_applied[kNumReactions];
    for (int k = 0; k < kNumReactions; ++k) {
      events_applied[k] = theta * events[k];
    }
    double dn[kNumSpecies];
    burn_stoich_delta_from_events(events_applied, dn);
    for (int s = 0; s < kNumSpecies; ++s) {
      n[s] += dn[s];
      if (n[s] < 0.0) {
        n[s] = 0.0;
      }
    }
    for (int k = 0; k < kNumReactions; ++k) {
      counts[k] += events_applied[k];
    }
  }

  return M;
}

}  // namespace tenryu::burn
