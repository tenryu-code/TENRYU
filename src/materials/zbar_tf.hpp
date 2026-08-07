#pragma once

namespace tenryu::materials {

[[nodiscard]] double compute_zbar_tf(double rho,
                                     double Te_eV,
                                     double Z_nuc,
                                     double A_amu);

[[nodiscard]] double compute_dzbar_dTe(double rho,
                                       double Te_eV,
                                       double Z_nuc,
                                       double A_amu,
                                       double temperature_floor_eV = 1.0e-3);

}  // namespace tenryu::materials
