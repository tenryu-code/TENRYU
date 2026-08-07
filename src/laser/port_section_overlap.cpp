#include "laser/port_section_overlap.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "core/error.hpp"

namespace tenryu::laser::port_section {
namespace {

constexpr double kPi = 3.14159265358979323846;

double dot(const std::array<double, 3>& a,
           const std::array<double, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

std::array<double, 3> to_lab(const port_geom::PortFrame& frame,
                             const std::array<double, 3>& local) {
  return {
      frame.e1[0] * local[0] + frame.e2[0] * local[1] +
          frame.b[0] * local[2],
      frame.e1[1] * local[0] + frame.e2[1] * local[1] +
          frame.b[1] * local[2],
      frame.e1[2] * local[0] + frame.e2[2] * local[1] +
          frame.b[2] * local[2]};
}

// Duplicated from the file-local helper in port_section_chi.cpp.
std::array<double, 3> lab_dir(const port_geom::PortFrame& frame,
                              const double theta, const double phi,
                              const double alpha, const int sheet) {
  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  const double cos_phi = std::cos(phi);
  const double sin_phi = std::sin(phi);
  const std::array<double, 3> r_hat{
      sin_theta * cos_phi, sin_theta * sin_phi, cos_theta};
  const std::array<double, 3> theta_hat{
      cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta};
  const std::array<double, 2> meridional =
      sector_ps::meridional_direction(alpha, sheet);
  const std::array<double, 3> local{
      meridional[0] * r_hat[0] + meridional[1] * theta_hat[0],
      meridional[0] * r_hat[1] + meridional[1] * theta_hat[1],
      meridional[0] * r_hat[2] + meridional[1] * theta_hat[2]};
  return to_lab(frame, local);
}

double profile_value(const port_geom::BeamAngularProfile& profile,
                     const double mu) {
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

}  // namespace

port_geom::BeamAngularProfile beam_profile_at_shell(
    const sector_ps::PhaseSpaceTable& table, const int shell_index,
    const int n_mu_profile) {
  TENRYU_ASSERT(n_mu_profile >= 2,
                "port_section angular profile requires at least two nodes");

  port_geom::BeamAngularProfile profile;
  profile.mu.resize(static_cast<std::size_t>(n_mu_profile));
  profile.I.assign(static_cast<std::size_t>(n_mu_profile), 0.0);
  std::vector<double> weighted_intensity(
      static_cast<std::size_t>(n_mu_profile), 0.0);
  std::vector<double> power(static_cast<std::size_t>(n_mu_profile), 0.0);

  for (int k = 0; k < n_mu_profile; ++k) {
    profile.mu[static_cast<std::size_t>(k)] =
        -1.0 + 2.0 * static_cast<double>(k) /
                   static_cast<double>(n_mu_profile - 1);
  }

  for (int sheet = 0; sheet < 2; ++sheet) {
    for (const sector_ps::CrossingView& crossing :
         sector_ps::crossings(table, shell_index, sheet)) {
      if (crossing.in_limiter_zone) {
        continue;
      }
      const double mu = std::clamp(std::cos(crossing.theta), -1.0, 1.0);
      const double scaled =
          0.5 * (mu + 1.0) * static_cast<double>(n_mu_profile - 1);
      const std::size_t bin = static_cast<std::size_t>(
          std::clamp(static_cast<int>(std::lround(scaled)), 0,
                     n_mu_profile - 1));
      const double intensity = crossing.P / crossing.area;
      weighted_intensity[bin] += crossing.P * intensity;
      power[bin] += crossing.P;
    }
  }

  std::vector<std::size_t> populated;
  populated.reserve(static_cast<std::size_t>(n_mu_profile));
  for (std::size_t k = 0; k < profile.I.size(); ++k) {
    if (power[k] != 0.0) {
      profile.I[k] = weighted_intensity[k] / power[k];
      populated.push_back(k);
    }
  }
  if (populated.empty()) {
    return profile;
  }

  const std::size_t first = populated.front();
  const std::size_t last = populated.back();
  std::fill(profile.I.begin(), profile.I.begin() + first, profile.I[first]);
  std::fill(profile.I.begin() + last + 1, profile.I.end(), profile.I[last]);
  for (std::size_t n = 1; n < populated.size(); ++n) {
    const std::size_t lo = populated[n - 1];
    const std::size_t hi = populated[n];
    for (std::size_t k = lo + 1; k < hi; ++k) {
      const double fraction =
          (profile.mu[k] - profile.mu[lo]) /
          (profile.mu[hi] - profile.mu[lo]);
      profile.I[k] =
          profile.I[lo] + fraction * (profile.I[hi] - profile.I[lo]);
    }
  }
  return profile;
}

port_geom::IlluminationResult illumination_at_shell(
    const IlluminationInput& input) {
  const port_geom::BeamAngularProfile profile =
      beam_profile_at_shell(*input.table, input.shell_index,
                            input.n_mu_profile);
  const port_geom::AngularGrid grid =
      port_geom::build_angular_grid(input.n_mu_grid, input.n_phi_grid);
  const port_geom::IlluminationResult first_pass =
      port_geom::illumination_metrics(*input.ports, profile, grid, 0.0);
  return port_geom::illumination_metrics(
      *input.ports, profile, grid,
      input.I_cut_fraction * first_pass.I_max);
}

CommonWaveDriveResult common_wave_drive(
    const CommonWaveDriveInput& input,
    CommonWaveDriveGridOutput* const grid_output) {
  const port_geom::BeamAngularProfile built_profile =
      input.profile_override == nullptr
          ? beam_profile_at_shell(*input.table, input.shell_index,
                                  input.n_mu_profile)
          : port_geom::BeamAngularProfile{};
  const port_geom::BeamAngularProfile& profile =
      input.profile_override == nullptr ? built_profile
                                        : *input.profile_override;
  const port_geom::AngularGrid grid =
      port_geom::build_angular_grid(input.n_mu_grid, input.n_phi_grid);
  if (grid_output != nullptr) {
    grid_output->mu = grid.mu;
    grid_output->phi = grid.phi;
    const std::size_t n_grid = grid.mu.size() * grid.phi.size();
    grid_output->I_tot.assign(n_grid, 0.0);
    grid_output->I_cw.assign(n_grid, 0.0);
    grid_output->n_sigma.assign(n_grid, 0.0);
  }

  double integral_I_cw = 0.0;
  double integral_I_cw2 = 0.0;
  double integral_lower = 0.0;
  double integral_upper = 0.0;
  std::vector<int> sigma_frequency(input.ports->ports.size() + 1, 0);
  const double phi_weight = 2.0 * kPi /
                            static_cast<double>(grid.phi.size());

  for (std::size_t i_mu = 0; i_mu < grid.mu.size(); ++i_mu) {
    const double mu = grid.mu[i_mu];
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    const double weight = grid.wmu[i_mu] * phi_weight;
    for (std::size_t i_phi = 0; i_phi < grid.phi.size(); ++i_phi) {
      const double phi = grid.phi[i_phi];
      const std::array<double, 3> omega{
          sin_theta * std::cos(phi), sin_theta * std::sin(phi), mu};
      port_geom::CommonWaveInput cluster_input;
      cluster_input.k_hat.reserve(input.ports->ports.size());
      cluster_input.I.reserve(input.ports->ports.size());
      for (std::size_t i_port = 0; i_port < input.ports->ports.size();
           ++i_port) {
        const port_geom::PortFrame& frame = input.ports->frames[i_port];
        const double mu_axis =
            std::clamp(dot(frame.b, omega), -1.0, 1.0);
        const double theta = std::acos(mu_axis);
        const double phi_port =
            std::atan2(dot(omega, frame.e2), dot(omega, frame.e1));
        const sector_ps::PumpLookup lookup = sector_ps::lookup(
            *input.table, input.shell_index, 0, theta);
        cluster_input.k_hat.push_back(
            lab_dir(frame, theta, phi_port, lookup.alpha, 0));
        cluster_input.I.push_back(
            lookup.valid
                ? input.ports->ports[i_port].power_weight *
                      profile_value(profile, mu_axis)
                : 0.0);
      }

      const std::vector<std::array<double, 3>> candidate_axes =
          port_geom::build_candidate_axes(cluster_input, omega);
      const port_geom::CommonWaveResult cluster =
          port_geom::common_wave_cluster(cluster_input, candidate_axes,
                                         input.delta_theta_deg);
      if (grid_output != nullptr) {
        const std::size_t index = i_mu * grid.phi.size() + i_phi;
        grid_output->I_tot[index] = cluster.I_upper;
        grid_output->I_cw[index] = cluster.I_cw;
        grid_output->n_sigma[index] =
            static_cast<double>(cluster.n_sigma);
      }

      // The shell-area factor r^2 is common to every term and cancels from
      // I_drive. The lower and upper diagnostics are solid-angle means.
      integral_I_cw += weight * cluster.I_cw;
      integral_I_cw2 += weight * cluster.I_cw * cluster.I_cw;
      integral_lower += weight * cluster.I_lower;
      integral_upper += weight * cluster.I_upper;
      if (cluster.I_cw > 0.0) {
        ++sigma_frequency[static_cast<std::size_t>(cluster.n_sigma)];
      }
    }
  }

  int n_sigma_mode = 0;
  int mode_frequency = 0;
  for (std::size_t n_sigma = 0; n_sigma < sigma_frequency.size();
       ++n_sigma) {
    if (sigma_frequency[n_sigma] > mode_frequency) {
      mode_frequency = sigma_frequency[n_sigma];
      n_sigma_mode = static_cast<int>(n_sigma);
    }
  }

  const double sphere_area = 4.0 * kPi;
  return CommonWaveDriveResult{
      integral_I_cw == 0.0 ? 0.0 : integral_I_cw2 / integral_I_cw,
      integral_lower / sphere_area,
      integral_upper / sphere_area,
      n_sigma_mode};
}

}  // namespace tenryu::laser::port_section
