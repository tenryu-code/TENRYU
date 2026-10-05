#pragma once

// Parts of the 1D S_N solver (radiation/sn_transport_1d_gpu.cu) shared with
// the linear-discontinuous scheme (radiation/sn_ld_1d_gpu.cu).

#include "core/config.hpp"
#include "radiation/sn_ld_1d_gpu.cuh"

namespace tenryu::core {
struct State;
}

namespace tenryu::radiation {
class PlanckTable;
}

namespace tenryu::radiation::sn1d_internal {

// Per-cell group opacities at state.Te: sn_sigma_a (absorption), sn_sigma_pe
// (emission), sn_sigma_s (physical scattering) and sn_eta (the cell-constant
// emission of the step-characteristic scheme), every material model.
void evaluate_opacity(core::State& state, const core::Config& cfg, const PlanckTable& planck,
                      const core::Config::MaterialsConfig::MatDef& mat, int n_cells,
                      int n_groups, double dt);

// The device quadrature of the 1D sweeps (Gauss-Legendre with the
// weighted-diamond factors for the sphere and the slab, the product
// quadrature of the cylinder) as chains for the linear-discontinuous sweep.
sn_ld::QuadratureView quadrature(int n_angles, int geom);

// Allocates the S_N state arrays of the given sizes (sigma, eta, face fluxes,
// diagnostics, reduction scratch).
void ensure_buffers(core::State& state, int n_cells, int n_groups, int n_angles);

// The external volume source of the step (Radiation.volume_source_rate / volume_source_x_max), on
// the device: source[c] = rate where the cell center 0.5 (x_r[c] + x_r[c + 1]) <= x_max, else 0;
// returns the energy it injects over dt, the sum over those cells of dt max(V, 0) rate (a
// non-finite V counts 0) in cell order (one copy to the host). x_r, vol, source: device arrays.
double volume_source(const double* x_r, const double* vol, int n_cells, double rate,
                     double x_max, double dt, double* source);

}  // namespace tenryu::radiation::sn1d_internal
