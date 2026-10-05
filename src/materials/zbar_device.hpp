#pragma once

#include <memory>

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::materials {

struct ZbarDeviceContext {
  struct Impl;
  std::unique_ptr<Impl> impl;
  ZbarDeviceContext();
  ~ZbarDeviceContext();
  ZbarDeviceContext(const ZbarDeviceContext&) = delete;
  ZbarDeviceContext& operator=(const ZbarDeviceContext&) = delete;
};

// How the device evaluates the Thomas-Fermi fit: with CUDA's pow and exp, or in the host's
// rounding (glibc's pow and exp reproduced, each operation rounded separately), which gives the
// host's values bit for bit (zbar_tf_host_rounding.cuh).
enum class ZbarTfRounding { cuda_math, host };

void update_zbar_fields_device(ZbarDeviceContext& context,
                               core::State& state, const core::Config& cfg,
                               ZbarTfRounding tf_rounding = ZbarTfRounding::cuda_math);

}  // namespace tenryu::materials
