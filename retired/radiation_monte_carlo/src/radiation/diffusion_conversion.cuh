#pragma once

#include <cstdint>

#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {

struct DiffusionConversionStats {
  std::uint64_t n_particles = 0;
  double energy = 0.0;
};

DiffusionConversionStats fold_entering_diffusion_particles_cuda(
    PhotonPool& pool,
    int n_particles,
    const std::uint8_t* diff_cell,
    const std::uint8_t* diff_cell_prev,
    const double* vol,
    double* diff_E,
    double* entry_E,
    int n_cells,
    int n_groups);

void fill_diffusion_exit_phase_space_1d_cuda(PhotonPool& pool,
                                             int start,
                                             int n_new,
                                             const double* node_r,
                                             int n_cells,
                                             std::uint64_t user_seed,
                                             std::uint64_t step_number,
                                             double dt);

void zero_exiting_diffusion_energy_cuda(double* diff_E,
                                        const std::uint8_t* diff_cell,
                                        const std::uint8_t* diff_cell_prev,
                                        int n_cells,
                                        int n_groups);

}  // namespace tenryu::radiation
