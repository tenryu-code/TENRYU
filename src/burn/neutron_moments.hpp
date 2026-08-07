#pragma once

namespace tenryu::burn {

// Brysk 1973 neutron spectral moments from burn-averaged inputs.
// Channel: 0 = DT (E_n0 = 14.049 MeV base), 1 = DDn (2.449 MeV).
struct BryskMoments {
  double mean_shift_keV = 0.0;       // <E_n> - (m_frac)*Q, Eq. 30
  double sigma_thermal_keV = 0.0;    // Gaussian sigma from Eq. 35 width
  double sigma_total_keV = 0.0;      // + isotropic fluid broadening (design F)
};

BryskMoments brysk_moments(int channel, double Ti_burn_keV,
                           double vr2_burn_cm2s2);

}  // namespace tenryu::burn

