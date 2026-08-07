#pragma once

#include <cuda_runtime.h>

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::materials {

void zmoment_upload_tables(core::State& state, const core::Config& cfg);

void zmoment_fill_fields(core::State& state, cudaStream_t stream);

}  // namespace tenryu::materials
