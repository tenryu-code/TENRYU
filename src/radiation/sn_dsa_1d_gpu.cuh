#pragma once

#include <cuda_runtime.h>

#include "core/state.hpp"

namespace tenryu::radiation {

void ensure_sn_dsa_1d_gpu_buffers(core::State& state,
                                  int n_cells,
                                  int n_groups);

void apply_sn_dsa_1d_gpu(core::State& state,
                         int n_cells,
                         int n_groups,
                         double dt,
                         cudaStream_t stream = nullptr);

}  // namespace tenryu::radiation
