#include "core/version.hpp"

namespace tenryu::core {

std::string tenryu_version_string() {
  return "0.0.1";
}

int tenryu_version_major() {
  return TENRYU_VERSION_MAJOR;
}

int tenryu_version_minor() {
  return TENRYU_VERSION_MINOR;
}

int tenryu_version_patch() {
  return TENRYU_VERSION_PATCH;
}

}  // namespace tenryu::core
