#pragma once

#include <vector>

namespace tenryu::radiation {

struct SNTransport1DConfig {
  int n_angles = 8;         // S_N order; must be even
  int max_iterations = 200;  // Source iteration limit
  double convergence_tol = 1.0e-6;
  bool origin_parity_only = false;
};

struct SNTransport1DResult {
  std::vector<double> chi;   // [n_cells * n_groups], P_rr/E
  std::vector<double> E_sn;  // [n_cells * n_groups], [erg/cm^3]
  int iterations = 0;
  double convergence_error = 0.0;
  bool converged = false;
};

SNTransport1DResult solve_sn_transport_1d(
    const double* sigma_a,       // [n_cells * n_groups], [1/cm]
    const double* sigma_s,       // [n_cells * n_groups], [1/cm]
    const double* planck_source, // [n_cells * n_groups], [erg/cm^3/s]
    const double* node_r,        // [n_cells + 1], [cm]
    const double* vol,           // [n_cells], [cm^3]
    int n_cells,
    int n_groups,
    const SNTransport1DConfig& config);

}  // namespace tenryu::radiation
