#pragma once

#include <cuda_runtime.h>

#include "core/state.hpp"

namespace tenryu::radiation {

void ensure_sn_dsa_1d_gpu_buffers(core::State& state,
                                  int n_cells,
                                  int n_groups);

// Diffusion correction of the swept scalar flux, consistent with the sweep's
// spatial closure (NUMERICS 6.8): mu and weight are the angular quadrature
// [n_angles], theta the sweep's spatial weight per cell, group and angle
// (psi_m = theta psi_downstream + (1 - theta) psi_upstream; nullptr: 1/2).
// geom is the 1D geometry code (mesh::Geometry1D) of the face areas.
void apply_sn_dsa_1d_gpu(core::State& state,
                         int n_cells,
                         int n_groups,
                         double dt,
                         int geom,
                         cudaStream_t stream,
                         const double* mu,
                         const double* weight,
                         int n_angles,
                         const double* theta);

// The DSA's scratch (its address identifies the buffers of a captured
// inner-iteration graph).
const void* sn_dsa_1d_gpu_scratch(int n_cells, int n_groups);

}  // namespace tenryu::radiation
