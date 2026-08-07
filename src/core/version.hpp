#pragma once

#include <string>

namespace tenryu::core {

inline constexpr int TENRYU_VERSION_MAJOR = 0;
inline constexpr int TENRYU_VERSION_MINOR = 0;
inline constexpr int TENRYU_VERSION_PATCH = 1;

std::string tenryu_version_string();
int tenryu_version_major();
int tenryu_version_minor();
int tenryu_version_patch();

}  // namespace tenryu::core
