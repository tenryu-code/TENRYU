#pragma once

#include <string>

#include <CLI/CLI.hpp>

namespace tenryu::core {
struct Config;
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

int cmd_run(const std::string& namelist_path,
            const std::string& restart_prefix = "",
            const std::string& output_dir_override = "");
int cmd_validate(const std::string& namelist_path, bool mesh_preview = false);
int cmd_freeze(const std::string& namelist_path, const std::string& output_path);
int cmd_verify(const std::string& test_name, bool generate_golden = false);

}  // namespace tenryu::drivers
