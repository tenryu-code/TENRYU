#pragma once

#include <limits>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/namelist/frozen_table.hpp"
#include "core/state.hpp"

namespace tenryu::laser {

struct Beam {
  std::string name;
  double dir_x = 0.0;
  double dir_y = 0.0;
  double dir_z = 1.0;
  double focus_x = 0.0;
  double focus_y = 0.0;
  double focus_z = 0.0;
  double focus_lab_z = 0.0;
  double f_number = 8.0;
  double defocus_DR = 0.0;
  double delta_lambda_nm = 0.0;
  std::string profile_model = "gaussian";
  double profile_w0_cm = 0.0;
  int profile_m = 2;
  std::vector<double> profile_r_cm;
  std::vector<double> profile_I;
  int wave_id = -1;
  core::namelist::FrozenTable1D power_table;

  [[nodiscard]] double get_power(double t) const;
  [[nodiscard]] double profile(double R_cm) const;
};

struct Beams {
  std::vector<Beam> items;

  [[nodiscard]] double total_power(double t) const;
};

Beams create_from_config(const core::Config::LaserConfig& laser,
                         const core::State& state,
                         double target_radius_cm,
                         double z_center_override = std::numeric_limits<double>::quiet_NaN());

}  // namespace tenryu::laser
