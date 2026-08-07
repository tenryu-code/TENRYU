#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include "core/config.hpp"

namespace tenryu::core::namelist {

using NamelistConfig = tenryu::core::Config;

struct FreezeGeometrySummary {
  bool has_rho = false;
  double rho_min = 0.0;
  double rho_max = 0.0;
  double rho_mean = 0.0;

  bool has_Te = false;
  double Te_min = 0.0;
  double Te_max = 0.0;

  bool has_Ti = false;
  double Ti_min = 0.0;
  double Ti_max = 0.0;

  std::map<std::string, double> material_volume;
};

struct FreezeTableSummary {
  double t_min = 0.0;
  double t_max = 0.0;
  int n_points = 0;
  double peak_value = 0.0;
  double integrated_value = 0.0;
};

struct FreezeExtras {
  std::optional<FreezeGeometrySummary> geometry;
  std::map<std::string, FreezeTableSummary> tables;
};

class Freeze {
 public:
  static std::string to_checkpoint_json(const NamelistConfig& config);
  static bool configs_equivalent(const std::string& json_a,
                                 const std::string& json_b);
  static std::string to_json(const NamelistConfig& config,
                             const FreezeExtras* extras = nullptr);
  static void write(const NamelistConfig& config,
                    const std::filesystem::path& output_path,
                    const FreezeExtras* extras = nullptr);
  static void write_pretty(const NamelistConfig& config,
                           const std::filesystem::path& output_path,
                           const FreezeExtras* extras = nullptr);
};

}  // namespace tenryu::core::namelist
