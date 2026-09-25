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

  // 1D: position of the focus on the beam axis relative to the target centre,
  // positive toward the source (the launch side of the 1D laser mesh),
  // -(focus . d_hat). The 1D target is spherically symmetric, so every beam is
  // traced as one that propagates toward -Z; for the direction (0, 0, -1) this
  // is the lab z of the focus.
  [[nodiscard]] double axial_focus_1d() const {
    return -(focus_x * dir_x + focus_y * dir_y + focus_lab_z * dir_z);
  }
  [[nodiscard]] double get_power(double t) const;
  // Power averaged over the step [t0, t1] (exact integral of the waveform).
  [[nodiscard]] double get_average_power(double t0, double t1) const;
  [[nodiscard]] double profile(double R_cm) const;
};

struct Beams {
  std::vector<Beam> items;

  [[nodiscard]] double total_power(double t) const;
  [[nodiscard]] double total_average_power(double t0, double t1) const;
};

Beams create_from_config(const core::Config::LaserConfig& laser,
                         const core::State& state,
                         double target_radius_cm,
                         double z_center_override = std::numeric_limits<double>::quiet_NaN());

}  // namespace tenryu::laser
