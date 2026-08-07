#pragma once

#include <cuda_runtime.h>

namespace tenryu::burn {

struct McParams {
  double E_min_keV = 20.0;      // thermalization cut (shared with diffusion)
  int particles_per_cell = 16;  // per slot per step
  unsigned long long seed = 12345ULL;
  double dE_frac = 0.1;         // max relative CSDA energy change per substep
};

struct McStepResult {
  double sourced_erg = 0.0;
  double escaped_erg = 0.0;
  double inflight_erg = 0.0;    // after the step
  int live_particles = 0;
  bool overflow = false;
};

__host__ __device__ inline unsigned long long mc_transport_global_id(
    const int cell, const int slot, const int sample,
    const int particles_per_cell) {
  return ((static_cast<unsigned long long>(cell) * 6ULL +
           static_cast<unsigned long long>(slot)) *
          static_cast<unsigned long long>(particles_per_cell)) +
         static_cast<unsigned long long>(sample);
}

McStepResult mc_transport_step(
    const McParams& p, int n_cells, long long step_index,
    const double* r_node_dev, const double* rho_dev,
    const double* Te_eV_dev, const double* Ti_eV_dev, const double* ne_dev,
    const double* S_birth_dev,   // [6*n_cells], particles/cm^3/s, slot-major
    const double* vol_dev, double dt_s,
    double* r_p, double* mu_p, double* E_p, double* w_p, int* slot_p,
    unsigned char* alive_p, int capacity, int* live_count_inout,
    double* dep_e_dev, double* dep_i_dev,   // [n_cells] erg +=
    cudaStream_t stream);

}  // namespace tenryu::burn
