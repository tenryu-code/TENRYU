#pragma once

#include <vector>

namespace tenryu::burn {

struct PartitionTable {
  // f_ion(species, Te, Ti, ne): fraction of the charged product's birth energy
  // deposited on IONS, integrated over LP slowing from E0 to
  // max(1.5 kTe, 1e-3 E0); electron field evaluated at Te, ion fields
  // (D/T/He3) at Ti (2026-07-26 review fix).
  // Grid: log-uniform Te [0.1, 200] keV x log-uniform ne [1e20, 1e28] cm^-3.
  int n_te = 64;
  int n_ti = 16;
  int n_ne = 16;
  double te_min_keV = 0.1, te_max_keV = 200.0;
  double ti_min_keV = 0.1, ti_max_keV = 200.0;
  double ne_min = 1.0e20, ne_max = 1.0e28;
  // layout [((product_slot * n_te + i_te) * n_ti + i_ti) * n_ne + i_ne]
  // product slots: 0=DT-alpha, 1=DDp-T, 2=DDp-p, 3=DDn-He3, 4=D3He-alpha,
  // 5=D3He-p, 6=n14-recoil-D (4.435 MeV), 7=n14-recoil-T (2.884 MeV),
  // 8=n2.45-recoil-D (1.089 MeV), 9=n2.45-recoil-T (0.920 MeV) [v2-E]
  static constexpr int kNumProductSlots = 10;
  std::vector<double> f_ion;
  double x_D = 0.5, x_T = 0.5, x_He3 = 0.0;
};

// Build at init for the declared fuel composition (field ions D/T/He3 with
// number fractions x_*; field electrons ne). Deterministic (fixed loops).
PartitionTable build_partition_table(double x_D, double x_T, double x_He3);

// Fraley 1974 Eq. 4 partition (DT alpha only; gate/diagnostic knob):
double fraley_ion_fraction(double Te_keV);

// Trilinear lookup in (log Te, log Ti, log ne), clamped at the grid edges.
double partition_f_ion(const PartitionTable& t, int product_slot,
                       double Te_keV, double Ti_keV, double ne_cm3);

}  // namespace tenryu::burn
