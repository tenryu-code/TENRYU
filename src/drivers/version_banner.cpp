#include "drivers/version_banner.hpp"

#include "core/version.hpp"
#include "tenryu_source_revision.hpp"

namespace tenryu::drivers {

std::string version_banner() {
  const std::string version = core::tenryu_version_string();
  const std::string revision = TENRYU_SOURCE_REVISION;
  if (revision.empty() || revision == "unknown") {
    return version;
  }
  return version + " (source " + revision + ")";
}

}  // namespace tenryu::drivers
