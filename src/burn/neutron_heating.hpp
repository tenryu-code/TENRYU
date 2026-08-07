#pragma once

#include <vector>

#include "burn/partition.hpp"

namespace tenryu::burn {

struct NeutronHeatingParams {
  int n_mu = 16;                       // even Gauss-Legendre order over mu in [-1,1]
  bool use_fraley_partition = false;   // else LP table (recoil slots 6..9)
};

struct NeutronHeatingResult {  // all energies in erg, this step
  double dep_e = 0.0;
  double dep_i = 0.0;
  double degraded = 0.0;   // scattered-neutron remainder (single flight: untracked)
  double escaped = 0.0;    // uncollided chord tails
  double emitted = 0.0;
  double conservation_resid = 0.0;  // |emitted - sum(parts)| / max(emitted, 1e-300)
};

// v2-E: single-flight first-collision deposit of the two neutron lines
// (burn_constants neutron_line_MeV) through the spherical 1D mesh.
// emit_erg: [n_cells * kNumNeutronLines] cell-major, total erg emitted this
// step per (cell, line). Attenuation targets are the fuel D/T inventories
// (burn_y * rho); cells with zero D/T are transparent (fuel self-heating
// tier; shell/hohlraum neutron kerma is a documented v3 item). Adds into
// dE_e/dE_i [erg per cell]. Full-sphere and annular (r_node[0] > 0) meshes
// are both handled; deterministic fixed-order host loops.
NeutronHeatingResult deposit_neutron_heating_1d(
    int n_cells,
    const double* r_node,   // n_cells+1 edges [cm]
    const double* rho,      // g/cm^3
    const double* Te_eV,
    const double* Ti_eV,
    const double* zbar,
    const double* A_eff,    // proton-mass units
    const std::vector<double>& burn_y,    // [n_cells*kNumSpecies], 1/g
    const std::vector<double>& emit_erg,  // [n_cells*kNumNeutronLines]
    const NeutronHeatingParams& p,
    const PartitionTable& table,
    std::vector<double>& dE_e,
    std::vector<double>& dE_i);

}  // namespace tenryu::burn
