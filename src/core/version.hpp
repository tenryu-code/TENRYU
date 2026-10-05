#pragma once

#include <string>

namespace tenryu::core {

inline constexpr int TENRYU_VERSION_MAJOR = 1;
inline constexpr int TENRYU_VERSION_MINOR = 0;
inline constexpr int TENRYU_VERSION_PATCH = 0;

// "1.0.0-beta.1": the version written into the frozen configuration
// (_tenryu_version). The source revision is not part of it (it is shown only
// by `tenryu --version`, drivers/version_banner.hpp), so the outputs of builds
// of different commits compare equal.
std::string tenryu_version_string();
int tenryu_version_major();
int tenryu_version_minor();
int tenryu_version_patch();

}  // namespace tenryu::core
