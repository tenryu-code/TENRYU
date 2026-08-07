#pragma once

#include <cmath>

#include "burn/burn_constants.hpp"
#include "core/macros.hpp"

namespace tenryu::burn {

// rho*lambda for the 3.540 MeV DT alpha [g/cm^2]; Te in keV, rho in g/cm^3.
// Fraley 1974 fit 3d (+-2% vs the integrated stopping, 1-100 keV, at solid
// DT rho0=0.213) times the electron-term Coulomb-log density scaling from
// Fraley 3b; reproduces Fraley's stated range enhancement delta(rho)=1->3
// for solid->1e4 g/cm^3 above 10 keV (2.92 at 10 keV, 1e4 g/cm^3).
TENRYU_HOST_DEVICE inline double alpha_rho_lambda(double Te_keV,
                                                   double rho_gcc) {
  const double kRho0 = 0.213;
  double Te = (Te_keV > 0.1) ? Te_keV : 0.1;
  const double Te54 = pow(Te, 1.25);
  const double base = 1.5e-2 * Te54 / (1.0 + 8.2e-3 * Te54);
  const double rho = (rho_gcc > 1.0e-12) ? rho_gcc : 1.0e-12;
  double num = 1.0 + 0.17 * log(Te);
  double den = 1.0 + 0.17 * log(Te * sqrt(kRho0 / rho));
  num = (num > 0.15) ? num : 0.15;
  den = (den > 0.15) ? den : 0.15;
  return base * (num / den);
}

// lambda_s = lambda_alpha * sqrt(m_s E_s / (m_alpha E_alpha)) * (Z_alpha/Z_s)^2.
// species_Z: D,T,He3,He4,p -> 1,1,2,2,1.
TENRYU_HOST_DEVICE inline double species_Z(int s) {
  constexpr double t[kNumSpecies] = {1.0, 1.0, 2.0, 2.0, 1.0};
  return t[s];
}

TENRYU_HOST_DEVICE inline double range_scale_factor(int species,
                                                     double E_MeV) {
  const double mE_alpha = species_A(kHe4) * charged_MeV(kDT, 0);
  const double z_ratio = species_Z(kHe4) / species_Z(species);
  return sqrt(species_A(species) * E_MeV / mE_alpha) * z_ratio * z_ratio;
}

TENRYU_HOST_DEVICE inline double point_sphere_IL(double m, double u,
                                                  double a2, double sa) {
  const double s = sqrt(a2 + u * u * m * m);
  return -0.5 * u * m * m + 0.5 * m * s +
         (a2 / (2.0 * u)) * asinh(u * m / sa);
}

TENRYU_HOST_DEVICE inline double point_sphere_IL2(double m, double u,
                                                   double a2) {
  const double s2 = a2 + u * u * m * m;
  return a2 * m + (2.0 / 3.0) * u * u * m * m * m -
         (2.0 / (3.0 * u)) * s2 * sqrt(s2);
}

TENRYU_HOST_DEVICE inline double point_sphere_deposited_fraction(double u,
                                                                  double tau) {
  // clamps: u in [0,1], tau > 0
  u = (u < 0.0) ? 0.0 : ((u > 1.0) ? 1.0 : u);
  if (!(tau > 0.0)) return 0.0;
  const double kUEps = 1.0e-6;
  if (u < kUEps) {
    // center: every chord has length R (units of R: L=1); column = tau.
    const double x = (tau < 1.0) ? tau : 1.0;
    return 1.0 - (1.0 - x) * (1.0 - x);
  }
  const double lam = 1.0 / tau;
  const double a2 = 1.0 - u * u;
  double mustar = (1.0 - u * u - lam * lam) / (2.0 * lam * u);
  double m0 = (mustar < -1.0) ? -1.0 : ((mustar > 1.0) ? 1.0 : mustar);
  // antiderivatives over mu of L and L^2 (L(mu) = -u*mu + sqrt(a2+u^2*mu^2)):
  //   IL(m)  = -u*m^2/2 + (m/2)*sqrt(a2+u^2 m^2) + (a2/(2u))*asinh(u*m/sqrt(a2))
  //   IL2(m) = a2*m + (2/3)*u^2*m^3 - (2/(3u))*(a2+u^2 m^2)^{3/2}
  // (a2 > 0 is guaranteed by u<1; at u==1, a2==0 and asinh(term) -> 0 works
  //  since a2/(2u)*asinh(u*m/sqrt(a2)) -> 0 as a2->0; guard sqrt(a2) with
  //  a tiny floor 1e-300 to avoid 0/0 producing NaN.)
  const double sa = sqrt((a2 > 1.0e-300) ? a2 : 1.0e-300);
  const double full = 0.5 * (m0 + 1.0);
  const double quad =
      0.5 * (2.0 * tau *
                 (point_sphere_IL(1.0, u, a2, sa) -
                  point_sphere_IL(m0, u, a2, sa)) -
             tau * tau *
                 (point_sphere_IL2(1.0, u, a2) -
                  point_sphere_IL2(m0, u, a2)));
  double f = full + quad;
  return (f < 0.0) ? 0.0 : ((f > 1.0) ? 1.0 : f);
}

}  // namespace tenryu::burn
