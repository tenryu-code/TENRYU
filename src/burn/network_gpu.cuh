#pragma once

#include <cuda_runtime.h>

#include "burn/network.cuh"

namespace tenryu::burn {

// n_dev: [n_cells*kNumSpecies] cell-major; Ti_eV_dev: [n_cells] (repo eV);
// counts_dev: [n_cells*kNumReactions] cell-major (overwritten);
// substeps_dev: [n_cells] (int, overwritten; 0 for skipped cells).
// T_floor_keV: cells with Ti below it are skipped entirely (design section 2).
void burn_network_apply_1d(int n_cells, double* n_dev, const double* Ti_eV_dev,
                           double dt_s, BurnChannels ch, double T_floor_keV,
                           double eps_deplete, int subcycle_max,
                           double* counts_dev, int* substeps_dev,
                           cudaStream_t stream);

}  // namespace tenryu::burn
