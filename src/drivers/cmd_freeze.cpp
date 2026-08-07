#include "drivers/cli.hpp"

#include <filesystem>
#include <iostream>

#include "core/error.hpp"
#if TENRYU_ENABLE_PYTHON
#include "core/namelist/freeze.hpp"
#include "core/namelist/runtime.hpp"
#endif

namespace tenryu::drivers {
namespace {

std::string format_cli_error(const std::string& message) {
  return "TENRYU ERROR [namelist]: " + message;
}

std::filesystem::path default_freeze_output(const std::string& namelist_path) {
  const std::filesystem::path input_path(namelist_path);
  return std::filesystem::path(input_path.stem().string() + "_frozen.json");
}

}  // namespace

int cmd_freeze(const std::string& namelist_path, const std::string& output_path) {
#if TENRYU_ENABLE_PYTHON
  try {
    tenryu::core::namelist::PythonGuard python_guard;
    tenryu::core::namelist::Runtime runtime;
    runtime.execute(namelist_path);

    const std::filesystem::path freeze_path =
        output_path.empty() ? default_freeze_output(namelist_path)
                            : std::filesystem::path(output_path);
    tenryu::core::namelist::Freeze::write_pretty(runtime.config(), freeze_path);

    const auto source = runtime.resolved_namelist_path();
    const auto copy_target = freeze_path.parent_path() / source.filename();
    std::error_code same_ec;
    const bool same_path = std::filesystem::equivalent(source, copy_target, same_ec);
    if (!same_path || same_ec) {
      std::error_code ec;
      std::filesystem::copy_file(
          source, copy_target, std::filesystem::copy_options::overwrite_existing, ec);
      if (ec) {
        tenryu::core::log_warning("Failed to copy source namelist next to frozen JSON: " +
                                  ec.message());
      }
    }

    tenryu::core::log_info("[TENRYU] Frozen config saved: " + freeze_path.string());
    return 0;
  } catch (const std::exception& e) {
    std::cerr << format_cli_error(e.what()) << '\n';
    return 1;
  }
#else
  (void)namelist_path;
  (void)output_path;
  tenryu::core::log_error("TENRYU was built without Python support (TENRYU_ENABLE_PYTHON=OFF)");
  return 1;
#endif
}

}  // namespace tenryu::drivers
