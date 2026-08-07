#pragma once

#include <stdexcept>

namespace tenryu::core::namelist {

class ConfigError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class ValueError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace tenryu::core::namelist
