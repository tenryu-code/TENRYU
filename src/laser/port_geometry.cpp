#include "laser/port_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "laser/hot_electron_1d.cuh"

namespace tenryu::laser::port_geom {
namespace {

constexpr double kPi = 3.14159265358979323846;

double dot(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double norm(const std::array<double, 3>& v) {
  return std::sqrt(dot(v, v));
}

std::array<double, 3> cross(const std::array<double, 3>& a,
                            const std::array<double, 3>& b) {
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

std::array<double, 3> normalized(const std::array<double, 3>& v) {
  const double magnitude = norm(v);
  return {v[0] / magnitude, v[1] / magnitude, v[2] / magnitude};
}

double interpolate_profile(const BeamAngularProfile& profile, const double mu) {
  if (mu <= profile.mu.front()) {
    return profile.I.front();
  }
  if (mu >= profile.mu.back()) {
    return profile.I.back();
  }
  const auto upper = std::upper_bound(profile.mu.begin(), profile.mu.end(), mu);
  const std::size_t hi =
      static_cast<std::size_t>(upper - profile.mu.begin());
  const std::size_t lo = hi - 1;
  const double fraction =
      (mu - profile.mu[lo]) / (profile.mu[hi] - profile.mu[lo]);
  return profile.I[lo] + fraction * (profile.I[hi] - profile.I[lo]);
}

bool is_duplicate_axis(
    const std::array<double, 3>& candidate,
    const std::vector<std::array<double, 3>>& accepted) {
  for (const auto& axis : accepted) {
    if (std::abs(dot(candidate, axis)) > 1.0 - 1.0e-10) {
      return true;
    }
  }
  return false;
}

}  // namespace

PortFrame build_port_frame(const std::array<double, 3>& axis_unit,
                           const double roll_deg) {
  const double axis_norm = norm(axis_unit);
  TENRYU_ASSERT(std::abs(axis_norm - 1.0) <= 1.0e-12,
                "port_geometry axis must be unit length");

  const std::array<double, 3> aux =
      (std::abs(axis_unit[2]) < 0.9)
          ? std::array<double, 3>{0.0, 0.0, 1.0}
          : std::array<double, 3>{1.0, 0.0, 0.0};
  const double projection = dot(aux, axis_unit);
  const std::array<double, 3> transverse{
      aux[0] - projection * axis_unit[0],
      aux[1] - projection * axis_unit[1],
      aux[2] - projection * axis_unit[2]};
  const std::array<double, 3> e1 = normalized(transverse);
  const std::array<double, 3> e2 = cross(axis_unit, e1);

  const double roll_rad = roll_deg * kPi / 180.0;
  const double cos_roll = std::cos(roll_rad);
  const double sin_roll = std::sin(roll_rad);
  PortFrame frame;
  for (int component = 0; component < 3; ++component) {
    const auto k = static_cast<std::size_t>(component);
    frame.e1[k] = cos_roll * e1[k] + sin_roll * e2[k];
    frame.e2[k] = -sin_roll * e1[k] + cos_roll * e2[k];
  }
  frame.b = axis_unit;
  return frame;
}

PortTable build_port_table(std::vector<Port> ports, const double lambda0_nm) {
  std::sort(ports.begin(), ports.end(),
            [](const Port& a, const Port& b) {
              return a.port_id < b.port_id;
            });
  for (std::size_t i = 1; i < ports.size(); ++i) {
    TENRYU_ASSERT(ports[i - 1].port_id != ports[i].port_id,
                  "port_geometry port_id values must be unique");
  }

  PortTable table;
  table.ports = std::move(ports);
  table.frames.reserve(table.ports.size());
  for (const auto& port : table.ports) {
    table.frames.push_back(build_port_frame(port.axis, port.roll_deg));
  }

  const std::size_t pair_count =
      table.ports.size() * (table.ports.size() - (table.ports.empty() ? 0 : 1)) /
      2;
  table.pairs.reserve(pair_count);
  table.delta_omega.reserve(pair_count);
  table.axis_dot.reserve(pair_count);
  for (std::size_t i = 0; i < table.ports.size(); ++i) {
    for (std::size_t j = i + 1; j < table.ports.size(); ++j) {
      table.pairs.emplace_back(static_cast<int>(i), static_cast<int>(j));
      const double lambda_i_cm =
          (lambda0_nm + table.ports[i].delta_lambda_nm) * 1.0e-7;
      const double lambda_j_cm =
          (lambda0_nm + table.ports[j].delta_lambda_nm) * 1.0e-7;
      table.delta_omega.push_back(
          2.0 * kPi * core::constants::c_light *
          (1.0 / lambda_j_cm - 1.0 / lambda_i_cm));
      table.axis_dot.push_back(
          dot(table.frames[i].b, table.frames[j].b));
    }
  }
  return table;
}

AngularGrid build_angular_grid(const int n_mu, const int n_phi) {
  AngularGrid grid;
  hot_electron::gauss_legendre(n_mu, grid.mu, grid.wmu);
  grid.phi.reserve(static_cast<std::size_t>(n_phi));
  for (int j = 0; j < n_phi; ++j) {
    grid.phi.push_back(2.0 * kPi * (static_cast<double>(j) + 0.5) /
                       static_cast<double>(n_phi));
  }
  return grid;
}

IlluminationResult illumination_metrics(const PortTable& table,
                                        const BeamAngularProfile& profile,
                                        const AngularGrid& grid,
                                        const double I_cut) {
  TENRYU_ASSERT(profile.mu.size() >= 2,
                "port_geometry angular profile requires at least two nodes");
  TENRYU_ASSERT(profile.mu.size() == profile.I.size(),
                "port_geometry angular profile size mismatch");
  for (std::size_t i = 1; i < profile.mu.size(); ++i) {
    TENRYU_ASSERT(profile.mu[i - 1] < profile.mu[i],
                  "port_geometry angular profile mu must be ascending");
  }
  TENRYU_ASSERT(grid.mu.size() == grid.wmu.size(),
                "port_geometry angular grid size mismatch");

  double integral_I = 0.0;
  double integral_I2 = 0.0;
  double union_solid_angle = 0.0;
  double I_max = -std::numeric_limits<double>::infinity();
  const double phi_weight = 2.0 * kPi / static_cast<double>(grid.phi.size());

  for (std::size_t i_mu = 0; i_mu < grid.mu.size(); ++i_mu) {
    const double mu = grid.mu[i_mu];
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    const double weight = grid.wmu[i_mu] * phi_weight;
    for (const double phi : grid.phi) {
      const std::array<double, 3> omega{
          sin_theta * std::cos(phi), sin_theta * std::sin(phi), mu};
      double I_tot = 0.0;
      for (std::size_t i_port = 0; i_port < table.ports.size(); ++i_port) {
        const double mu_axis = dot(table.frames[i_port].b, omega);
        I_tot += table.ports[i_port].power_weight *
                 interpolate_profile(profile, mu_axis);
      }
      integral_I += weight * I_tot;
      integral_I2 += weight * I_tot * I_tot;
      if (I_tot > I_cut) {
        union_solid_angle += weight;
      }
      I_max = std::max(I_max, I_tot);
    }
  }

  const double sphere_area = 4.0 * kPi;
  return IlluminationResult{
      integral_I * integral_I / (sphere_area * integral_I2),
      union_solid_angle / sphere_area,
      integral_I / sphere_area,
      I_max};
}

CommonWaveResult common_wave_cluster(
    const CommonWaveInput& input,
    const std::vector<std::array<double, 3>>& candidate_axes,
    const double delta_theta_deg) {
  TENRYU_ASSERT(input.k_hat.size() == input.I.size(),
                "port_geometry common-wave input size mismatch");
  TENRYU_ASSERT(!input.I.empty(),
                "port_geometry common-wave input must not be empty");
  TENRYU_ASSERT(!candidate_axes.empty(),
                "port_geometry common-wave candidate axes must not be empty");

  CommonWaveResult result;
  result.I_cw = -std::numeric_limits<double>::infinity();
  result.I_lower = *std::max_element(input.I.begin(), input.I.end());
  result.I_upper = 0.0;
  for (const double intensity : input.I) {
    result.I_upper += intensity;
  }
  result.n_sigma = 0;
  result.axis_index = 0;

  const double delta_theta = delta_theta_deg * kPi / 180.0;
  const double dedup_tolerance = delta_theta * 1.0e-3;
  for (std::size_t axis_index = 0; axis_index < candidate_axes.size();
       ++axis_index) {
    std::vector<double> theta;
    theta.reserve(input.k_hat.size());
    for (const auto& k_hat : input.k_hat) {
      theta.push_back(
          std::acos(std::clamp(dot(k_hat, candidate_axes[axis_index]),
                              -1.0, 1.0)));
    }

    std::vector<double> sorted_theta = theta;
    std::sort(sorted_theta.begin(), sorted_theta.end());
    std::vector<double> cone_angles;
    cone_angles.reserve(sorted_theta.size());
    for (const double angle : sorted_theta) {
      if (cone_angles.empty() ||
          std::abs(angle - cone_angles.back()) > dedup_tolerance) {
        cone_angles.push_back(angle);
      }
    }

    for (const double cone_angle : cone_angles) {
      double cluster_intensity = 0.0;
      int member_count = 0;
      for (std::size_t i = 0; i < theta.size(); ++i) {
        if (std::abs(theta[i] - cone_angle) <= delta_theta) {
          cluster_intensity += input.I[i];
          ++member_count;
        }
      }
      if (cluster_intensity > result.I_cw) {
        result.I_cw = cluster_intensity;
        result.n_sigma = member_count;
        result.axis_index = static_cast<int>(axis_index);
      }
    }
  }

  return result;
}

std::vector<std::array<double, 3>> build_candidate_axes(
    const CommonWaveInput& input, const std::array<double, 3>& r_hat) {
  TENRYU_ASSERT(input.k_hat.size() == input.I.size(),
                "port_geometry common-wave input size mismatch");

  std::vector<std::array<double, 3>> axes;
  axes.push_back(r_hat);
  for (std::size_t i = 0; i < input.k_hat.size(); ++i) {
    for (std::size_t j = i + 1; j < input.k_hat.size(); ++j) {
      const std::array<double, 3> sum{
          input.k_hat[i][0] + input.k_hat[j][0],
          input.k_hat[i][1] + input.k_hat[j][1],
          input.k_hat[i][2] + input.k_hat[j][2]};
      if (norm(sum) < 1.0e-6) {
        continue;
      }
      const std::array<double, 3> candidate = normalized(sum);
      if (!is_duplicate_axis(candidate, axes)) {
        axes.push_back(candidate);
      }
    }
  }
  return axes;
}

}  // namespace tenryu::laser::port_geom
