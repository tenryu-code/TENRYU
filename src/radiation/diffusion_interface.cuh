#pragma once

#include <cstdint>

#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {

struct DiffusionInterfaceResult {
  int n_spawned = 0;
  double E_spawned = 0.0;
  double E_leaked_vacuum = 0.0;
};

DiffusionInterfaceResult spawn_imc_from_diffusion_faces(
    PhotonPool& pool,
    double* diff_E,
    const double* sigma_R,
    const double* vol,
    const double* node_r,
    const std::uint8_t* diff_cell,
    double* face_current_out,
    int n_cells,
    int n_groups,
    double dt,
    int particles_per_face_group,
    std::uint64_t global_id_base,
    std::uint64_t seed,
    std::uint64_t step,
    int max_pool_size);

double deposit_diffusion_face_current_in(
    double* diff_E,
    const double* face_current_in,
    const double* vol,
    const std::uint8_t* diff_cell,
    int n_cells,
    int n_groups);

double diffusion_face_current_in_energy(const double* face_current_in,
                                        const std::uint8_t* diff_cell,
                                        int n_cells,
                                        int n_groups);

}  // namespace tenryu::radiation
