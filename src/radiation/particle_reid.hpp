#pragma once

#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {

struct ParticleReIDStats {
  int checked = 0;
  int kept = 0;
  int updated = 0;
  int skipped_dead = 0;
  int skipped_ddmc = 0;
  int skipped_nan = 0;
  int binary_search = 0;
  int clamped = 0;
};

ParticleReIDStats reidentify_finite_position_particles_1d_cuda(
    PhotonPool& pool,
    const double* node_r,
    int n_cells,
    int n_particles = -1);

}  // namespace tenryu::radiation
