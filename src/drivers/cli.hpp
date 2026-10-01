#pragma once

#include <optional>
#include <string>

#include <CLI/CLI.hpp>

namespace tenryu::core {
struct Config;
#if TENRYU_ENABLE_PYTHON
namespace namelist {
class Builder;
}
#endif
}

namespace tenryu::drivers {

struct CliOptions {
  bool verbose = false;
  bool quiet = false;
};

void add_common_cli_options(CLI::App& app, CliOptions& options);
void configure_logging(const CliOptions& options);
void setup_file_logging(const std::string& log_dir);

void validate_s2_multiblock_runtime_features(const tenryu::core::Config& cfg);

#if TENRYU_ENABLE_PYTHON
std::string build_mesh_requirement_json_for_config(
    const tenryu::core::Config& cfg,
    const tenryu::core::namelist::Builder& builder,
    bool* violated_out,
    std::string* violation_message_out);
#endif

// Run control of `tenryu run` for a restart: the end time and the step limit of the continued run. They override
// Main.t_end and Main.max_steps of the deck, which stays unchanged, so the checkpoint's frozen configuration still
// matches it (SPECIFICATION §7.4). Refused on a fresh run.
struct RunControlOverrides {
  std::optional<double> t_end;
  std::optional<int> max_steps;
};

int cmd_run(const std::string& namelist_path,
            const std::string& restart_prefix = "",
            const std::string& output_dir_override = "",
            const RunControlOverrides& run_control = {});
int cmd_validate(const std::string& namelist_path, bool mesh_preview = false);
int cmd_freeze(const std::string& namelist_path, const std::string& output_path);
int cmd_verify(const std::string& test_name, bool generate_golden = false);

}  // namespace tenryu::drivers
