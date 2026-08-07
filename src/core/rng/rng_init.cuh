#pragma once

#include <cstdint>

#include <curand_kernel.h>

namespace tenryu::core::rng {

__device__ inline void init_philox(curandStatePhilox4_32_10_t* state,
                                   const uint64_t global_id,
                                   const uint32_t rng_counter,
                                   const uint64_t user_seed,
                                   const uint64_t step_number) {
  curand_init(global_id ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(rng_counter),
              state);
}

__global__ inline void rng_init_kernel(curandStatePhilox4_32_10_t* states,
                                       const uint64_t* global_id,
                                       const uint32_t* rng_counter,
                                       const uint64_t user_seed,
                                       const uint64_t step_number,
                                       const int n_particles) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles) {
    return;
  }

  init_philox(&states[tid],
              global_id[tid],
              rng_counter[tid],
              user_seed,
              step_number);
}

}  // namespace tenryu::core::rng
