#include "laser/beams.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

#include "core/error.hpp"
#include "laser/coordinate_transform.cuh"
#include "laser/waveform_table.cuh"

namespace tenryu::laser {
namespace {

core::namelist::FrozenTable1D make_zero_power_table(const double t_end) {
  core::namelist::FrozenTable1D table;
  table.x = {0.0, t_end};
  table.y = {0.0, 0.0};
  table.n_points = 2;
  table.x_min = 0.0;
  table.x_max = t_end;
  table.zero_outside = true;
  return table;
}

}  // namespace

double Beam::get_power(const double t) const {
  // Namelist power callables return Watts (SI); convert to erg/s (cgs).
  constexpr double W_to_erg_per_s = 1.0e7;
  return waveform_power_at_time(power_table, t) * W_to_erg_per_s;
}

double Beam::profile(const double R_cm) const {
  const double w0 = std::max(profile_w0_cm, 1.0e-30);
  const double x = R_cm / w0;

  if (profile_model == "super_gaussian") {
    const int m = std::max(profile_m, 1);
    const double p = std::pow(std::abs(x), 2.0 * static_cast<double>(m));
    return std::exp(-2.0 * p);
  }
  if (profile_model == "flat_top") {
    return (R_cm <= w0) ? 1.0 : 0.0;
  }
  if (profile_model == "table") {
    if (profile_r_cm.size() < 2) {
      return 0.0;
    }
    if (R_cm <= profile_r_cm.front()) {
      return profile_I.front();
    }
    if (R_cm >= profile_r_cm.back()) {
      return 0.0;
    }
    const auto it =
        std::upper_bound(profile_r_cm.begin(), profile_r_cm.end(), R_cm);
    const std::size_t hi = static_cast<std::size_t>(it - profile_r_cm.begin());
    const std::size_t lo = hi - 1;
    const double span = profile_r_cm[hi] - profile_r_cm[lo];
    const double a = (span > 0.0) ? (R_cm - profile_r_cm[lo]) / span : 0.0;
    return profile_I[lo] + a * (profile_I[hi] - profile_I[lo]);
  }
  // gaussian + custom fallback
  return std::exp(-2.0 * x * x);
}

double Beams::total_power(const double t) const {
  double sum = 0.0;
  for (const Beam& beam : items) {
    sum += std::max(0.0, beam.get_power(t));
  }
  return sum;
}

Beams create_from_config(const core::Config::LaserConfig& laser,
                         const core::State& state,
                         const double target_radius_cm,
                         const double z_center_override) {
  Beams beams;
  if (!laser.enabled) {
    return beams;
  }

  beams.items.reserve(laser.beams.size());
  std::array<double, 3> target_center{0.0, 0.0, 0.0};
  if (std::isfinite(z_center_override)) {
    target_center[2] = z_center_override;
  } else if (state.mesh.dim == 2 && !state.mesh.cell_centroid_z.empty()) {
    auto [z_min_it, z_max_it] =
        std::minmax_element(state.mesh.cell_centroid_z.begin(), state.mesh.cell_centroid_z.end());
    target_center[2] = 0.5 * (*z_min_it + *z_max_it);
  }
  for (std::size_t i = 0; i < laser.beams.size(); ++i) {
    const auto& in = laser.beams[i];

    Beam out;
    out.name = in.name;

    const auto direction = resolve_beam_direction(in);
    out.dir_x = direction[0];
    out.dir_y = direction[1];
    out.dir_z = direction[2];

    const auto focus = resolve_focus_1d(in, target_radius_cm, target_center);
    out.focus_x = focus[0];
    out.focus_y = focus[1];
    out.focus_lab_z = focus[2];
    // Ray initialization is performed in a beam-local (R,Z) frame.
    // Store the axial focus coordinate as projection onto beam direction.
    out.focus_z = focus[0] * out.dir_x + focus[1] * out.dir_y + focus[2] * out.dir_z;

    out.f_number = in.f_number;
    out.defocus_DR = in.defocus_DR;
    out.delta_lambda_nm = in.delta_lambda_nm;
    out.profile_model = in.profile_model.empty() ? laser.profile_model : in.profile_model;

    const double w0_um = (in.profile_w0_um > 0.0) ? in.profile_w0_um : laser.profile_w0_um;
    out.profile_w0_cm = (w0_um > 0.0) ? (w0_um * 1.0e-4)
                                      : (0.5 * target_radius_cm / std::max(in.f_number, 1.0));
    out.profile_m = (in.profile_m > 0) ? in.profile_m : laser.profile_m;
    out.profile_r_cm = in.profile_r_cm.empty() ? laser.profile_r_cm : in.profile_r_cm;
    out.profile_I = in.profile_I.empty() ? laser.profile_I : in.profile_I;
    out.wave_id = static_cast<int>(i);

    if (i < state.laser_waveforms.size() && state.laser_waveforms[i].n_points > 0) {
      out.power_table = state.laser_waveforms[i];
    } else {
      out.power_table = make_zero_power_table(state.t + std::max(state.dt, 1.0));
      core::log_warning("Laser waveform table missing for beam index " + std::to_string(i) +
                        "; using zero-power fallback");
    }

    if (out.profile_model == "custom") {
      core::log_warning("Laser profile_model='custom' is not implemented in M12; falling back "
                        "to gaussian");
      out.profile_model = "gaussian";
    }

    if (out.profile_model != "gaussian" && out.profile_model != "super_gaussian" &&
        out.profile_model != "flat_top" && out.profile_model != "table") {
      core::log_warning("Unknown Laser profile_model='" + out.profile_model +
                        "'; falling back to gaussian");
      out.profile_model = "gaussian";
    }
    if (out.profile_model == "table" && out.profile_r_cm.size() < 2) {
      core::log_warning("laser beam '" + out.name +
                        "': table profile has no data; falling back to gaussian");
      out.profile_model = "gaussian";
    }

    beams.items.push_back(std::move(out));
  }

  return beams;
}

}  // namespace tenryu::laser
