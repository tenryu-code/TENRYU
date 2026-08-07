#include "burn/network_gpu.cuh"

namespace tenryu::burn {
namespace {

__global__ void burn_network_apply_1d_kernel(
    const int n_cells, double* __restrict__ n_dev,
    const double* __restrict__ Ti_eV_dev, const double dt_s, const BurnChannels ch,
    const double T_floor_keV, const double eps_deplete, const int subcycle_max,
    double* __restrict__ counts_dev, int* __restrict__ substeps_dev) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  double* const n_cell = n_dev + c * kNumSpecies;
  double* const counts_cell = counts_dev + c * kNumReactions;
  const double T_keV = Ti_eV_dev[c] * 1.0e-3;
  if (T_keV < T_floor_keV) {
    for (int k = 0; k < kNumReactions; ++k) {
      counts_cell[k] = 0.0;
    }
    substeps_dev[c] = 0;
    return;
  }

  double n_local[kNumSpecies];
  for (int s = 0; s < kNumSpecies; ++s) {
    n_local[s] = n_cell[s];
  }
  double counts_local[kNumReactions];
  const int substeps = burn_network_step(
      n_local, T_keV, dt_s, ch, eps_deplete, subcycle_max, counts_local);
  for (int s = 0; s < kNumSpecies; ++s) {
    n_cell[s] = n_local[s];
  }
  for (int k = 0; k < kNumReactions; ++k) {
    counts_cell[k] = counts_local[k];
  }
  substeps_dev[c] = substeps;
}

}  // namespace

void burn_network_apply_1d(int n_cells, double* n_dev, const double* Ti_eV_dev,
                           double dt_s, BurnChannels ch, double T_floor_keV,
                           double eps_deplete, int subcycle_max,
                           double* counts_dev, int* substeps_dev,
                           cudaStream_t stream) {
  if (n_cells <= 0) {
    return;
  }

  constexpr int kBlock = 128;
  const int grid = (n_cells + kBlock - 1) / kBlock;
  burn_network_apply_1d_kernel<<<grid, kBlock, 0, stream>>>(
      n_cells, n_dev, Ti_eV_dev, dt_s, ch, T_floor_keV, eps_deplete,
      subcycle_max, counts_dev, substeps_dev);
}

}  // namespace tenryu::burn
