#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_types.cuh"

namespace tenryu::hydro::ale1d {

std::vector<Ale1dFeature> compute_features(
    const core::State& state,
    const core::Config& cfg,
    double dt_step,
    cudaStream_t stream = nullptr);

}  // namespace tenryu::hydro::ale1d
