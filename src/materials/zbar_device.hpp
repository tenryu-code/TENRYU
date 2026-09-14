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

void update_zbar_fields_device(ZbarDeviceContext& context,
                               core::State& state, const core::Config& cfg);

}  // namespace tenryu::materials
