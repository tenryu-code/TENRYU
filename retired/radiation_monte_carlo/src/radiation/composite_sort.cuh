#pragma once

#include <cstdint>

#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {

struct CompositeSortResult {
  int n_alive = 0;
  int n_imc = 0;
  int n_ddmc = 0;
  int n_rw = 0;
};

CompositeSortResult composite_sort_and_partition(PhotonPool& pool,
                                                 int n_particles,
                                                 int n_cells,
                                                 int n_groups,
                                                 double* E_numerical_loss = nullptr);
CompositeSortResult compact_alive_only(PhotonPool& pool,
                                       int n_particles,
                                       int n_cells,
                                       int n_groups,
                                       double* E_numerical_loss = nullptr);
void reserve_compact_alive_scratch(int capacity);
void release_compact_alive_scratch();

}  // namespace tenryu::radiation
