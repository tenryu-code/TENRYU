#pragma once

namespace tenryu::burn {

enum class ScreeningMode : int {
  kNone = 0,
  kSalpeter = 1,
  kChugunovDeWitt = 2,
};

// Per-cell, per-reaction reactivity enhancement factors F_k >= 1
// (F = exp(h)). Inputs: Ti_eV/Te_eV (repo eV), ne_cm3, and the cell's ion
// species densities n_s [cm^-3] (kNumSpecies array: D,T,He3,He4,p) used for
// the mixture moments <Z>, <Z^2>. Writes F[kNumReactions]. mode==kNone
// writes all 1.0. If the total ion density is <= 0, writes all 1.0.
void burn_screening_factors(ScreeningMode mode, double Ti_eV, double Te_eV,
                            double ne_cm3, const double n_species[],
                            double F[]);

// Emit the host warn-once messages represented by screening warning flags.
void burn_screening_emit_warnings(unsigned int warning_flags);

}  // namespace tenryu::burn
