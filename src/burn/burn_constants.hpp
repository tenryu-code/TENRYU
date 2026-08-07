#pragma once

#include "core/macros.hpp"

namespace tenryu::burn {

// Species indices (cell-major inventory layout n[c*kNumSpecies + s]).
enum Species : int { kD = 0, kT = 1, kHe3 = 2, kHe4 = 3, kP = 4 };
inline constexpr int kNumSpecies = 5;

// Reaction indices.
enum Reaction : int { kDT = 0, kDDp = 1, kDDn = 2, kD3He = 3 };
inline constexpr int kNumReactions = 4;

// Species masses in PROTON-MASS units (repo n_i convention, constants.hpp
// proton_mass = 1.6726219e-24 g; see design doc unit-trap table).
TENRYU_HOST_DEVICE inline double species_A(int s) {
  constexpr double t[kNumSpecies] = {2.0141, 3.0160, 3.0160, 4.0026, 1.0073};
  return t[s];
}

inline constexpr double kMeVToErg = 1.602176634e-6;

// Reaction energetics [MeV]; kinematic splits sum to Q exactly (design section 3.2).
// R1 DT:   D+T   -> He4(3.540) + n(14.049), Q=17.589
// R2 DDp:  D+D   -> T(1.010)   + p(3.023),  Q=4.033
// R3 DDn:  D+D   -> He3(0.820) + n(2.449),  Q=3.269
// R4 D3He: D+He3 -> He4(3.690) + p(14.663), Q=18.353
TENRYU_HOST_DEVICE inline double reaction_Q_MeV(int k) {
  constexpr double t[kNumReactions] = {17.589, 4.033, 3.269, 18.353};
  return t[k];
}

TENRYU_HOST_DEVICE inline double neutron_MeV(int k) {
  constexpr double t[kNumReactions] = {14.049, 0.0, 2.449, 0.0};
  return t[k];
}

// Up to two charged products per reaction: species id (-1 = none) and E [MeV].
TENRYU_HOST_DEVICE inline int charged_species(int k, int i) {
  constexpr int t[kNumReactions][2] = {
      {kHe4, -1}, {kT, kP}, {kHe3, -1}, {kHe4, kP}};
  return t[k][i];
}

TENRYU_HOST_DEVICE inline double charged_MeV(int k, int i) {
  constexpr double t[kNumReactions][2] = {
      {3.540, 0.0}, {1.010, 3.023}, {0.820, 0.0}, {3.690, 14.663}};
  return t[k][i];
}

// Stoichiometry: change in each species per single reaction event.
// rows = reaction, cols = species (D,T,He3,He4,p).
TENRYU_HOST_DEVICE inline double stoich(int k, int s) {
  constexpr double t[kNumReactions][kNumSpecies] = {
      {-1.0, -1.0, 0.0, +1.0, 0.0},    // DT
      {-2.0, +1.0, 0.0, 0.0, +1.0},    // DDp
      {-2.0, 0.0, +1.0, 0.0, 0.0},     // DDn
      {-1.0, 0.0, -1.0, +1.0, +1.0},   // D3He
  };
  return t[k][s];
}
// Like-particle symmetry factor 1/(1+delta_ij) applied inside the rate:
TENRYU_HOST_DEVICE inline double sym_factor(int k) {
  constexpr double t[kNumReactions] = {1.0, 0.5, 0.5, 1.0};
  return t[k];
}

// Bosch-Hale 1992 Table VII reactivity-fit coefficients (NF 32, 611).
// Order: DT, DDp, DDn, D3He rows to match Reaction enum? NO -- keep the
// Reaction enum order exactly: index by Reaction, values below.
struct BoschHaleFit {
  double B_G;        // sqrt(keV)
  double mrc2_keV;
  double C1, C2, C3, C4, C5, C6, C7;
  double T_min_keV;  // fit validity floor (below: reactivity treated as 0)
  double T_max_keV;  // fit ceiling (above: evaluated at ceiling, clamped)
};
TENRYU_HOST_DEVICE inline BoschHaleFit bosch_hale_fit(int k) {
  constexpr BoschHaleFit t[kNumReactions] = {
      // DT: T(d,n)4He
      {34.3827, 1.124656e6, 1.17302e-9, 1.51361e-2, 7.51886e-2, 4.60643e-3,
       1.35000e-2, -1.06750e-4, 1.36600e-5, 0.2, 100.0},
      // DDp: D(d,p)T
      {31.3970, 9.37814e5, 5.65718e-12, 3.41267e-3, 1.99167e-3, 0.0,
       1.05060e-5, 0.0, 0.0, 0.2, 100.0},
      // DDn: D(d,n)3He
      {31.3970, 9.37814e5, 5.43360e-12, 5.85778e-3, 7.68222e-3, 0.0,
       -2.96400e-6, 0.0, 0.0, 0.2, 100.0},
      // D3He: 3He(d,p)4He
      {68.7508, 1.124572e6, 5.51036e-10, 6.41918e-3, -2.02896e-3, -1.91080e-5,
       1.35776e-4, 0.0, 0.0, 0.5, 190.0},
  };
  return t[k];
}

// v2-E neutron in-flight heating: frozen elastic cross sections and mean
// first-collision transfer fractions per (line, target). Lines follow
// neutron_MeV's kinematic splits: line 0 = DT-n 14.049 MeV, line 1 =
// DD-n 2.449 MeV. Targets: 0 = D, 1 = T (subset of Species; He3/He4/p
// scattering is a documented v2 omission). Provenance and uncertainty
// bands: docs/design/burn_kernel_v2_20260710.md section E.2 (Navratil
// LLNL-TR-423504 appendix table; Frenje PRL 107, 122502; Miller PRC 106,
// 024001 figure-derived; Hale PRC 42, 438). sigma in barns; convert at
// the use site (1 b = 1e-24 cm^2).
inline constexpr int kNumNeutronLines = 2;

TENRYU_HOST_DEVICE inline double neutron_line_MeV(int l) {
  constexpr double t[kNumNeutronLines] = {14.049, 2.449};
  return t[l];
}

TENRYU_HOST_DEVICE inline double neutron_sigma_el_barn(int l, int tgt) {
  constexpr double t[kNumNeutronLines][2] = {
      {0.63, 0.94},  // 14.049 MeV on D (derived +-10%), T (evaluation <5%)
      {2.31, 2.30},  // 2.449 MeV on D (+-5%), T (+-4%)
  };
  return t[l][tgt];
}

// f = 2A/(A+1)^2 (1 - <cos th_CM>), A = m_i/m_n (atomic masses):
// A_D = 1.9968 -> 0.4447 isotropic; A_T = 2.9901 -> 0.3756 isotropic.
// <cos th_CM>: 14 MeV on T = 0.4535 (Navratil table integration),
// on D = +0.29 (Miller 12->14.1 MeV derived); 2.449 MeV = 0.0 for both
// (isotropic approximation, documented).
TENRYU_HOST_DEVICE inline double neutron_f_transfer(int l, int tgt) {
  constexpr double t[kNumNeutronLines][2] = {
      {0.3157, 0.2053},
      {0.4447, 0.3756},
  };
  return t[l][tgt];
}

// Partition-table product slots of the mean-energy elastic recoils
// (partition.hpp slots 6..9): index [line][target].
TENRYU_HOST_DEVICE inline int neutron_recoil_slot(int l, int tgt) {
  constexpr int t[kNumNeutronLines][2] = {{6, 7}, {8, 9}};
  return t[l][tgt];
}

}  // namespace tenryu::burn
