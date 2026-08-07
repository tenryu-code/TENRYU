#pragma once

#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {

void compute_weight_stats_device(const PhotonPool& pool,
                                 double* weight_min,
                                 double* weight_mean,
                                 double* weight_max);

void reserve_pool_stats_scratch(int capacity);
void release_pool_stats_scratch();

}  // namespace tenryu::radiation
