#pragma once

namespace tenryu::verification {

double noh_shock_radius(double t, double v0);

double noh_density_plateau(double rho0, double gamma);

// Planar Noh: a single shock compression, no geometric pre-compression.
double noh_density_plateau_planar(double rho0, double gamma);

// Cylindrical Noh: geometric pre-compression (1 power) x Hugoniot 4:1.
double noh_density_plateau_cylindrical(double rho0, double gamma);

}  // namespace tenryu::verification
