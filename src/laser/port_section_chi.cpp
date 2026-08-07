#include "laser/port_section_chi.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

#include "core/constants.hpp"
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

std::array<double, 3> lab_pos_dir(const port_geom::PortFrame& frame,
                                  const double theta, const double phi) {
  const double sin_theta = std::sin(theta);
  const std::array<double, 3> local{
      sin_theta * std::cos(phi),
      sin_theta * std::sin(phi),
      std::cos(theta)};
  return to_lab(frame, local);
}

std::array<double, 3> lab_dir(const port_geom::PortFrame& frame,
                              const double theta, const double phi,
                              const double alpha, const int sheet) {
  const double sin_theta = std::sin(theta);
  const double cos_theta = std::cos(theta);
  const double cos_phi = std::cos(phi);
  const double sin_phi = std::sin(phi);
  // r_hat is the position direction. theta_hat = d(r_hat)/d(theta)
  // is already normalized, and meridional rays have no phi_hat component.
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

}  // namespace

ChiBuildResult build_chi_ps(const ChiBuildInput& input) {
  const std::int64_t G_ref =
      2 * static_cast<std::int64_t>(input.n_impact_bins);
  const std::int64_t G_ps_64 =
      static_cast<std::int64_t>(input.ports->ports.size()) * G_ref;
  const std::int64_t n_pairs_64 = G_ps_64 * (G_ps_64 - 1) / 2;
  const std::string pair_cap_message =
      "port_section v1 supports at most 65536 expanded pairs (got " +
      std::to_string(n_pairs_64) +
      "): reduce ports or bins (OMEGA/NIF-scale needs the S2+ "
      "class/tiling path)";
  TENRYU_ASSERT(n_pairs_64 <= 65536, pair_cap_message);

  ChiBuildResult result;
  result.G_ps = static_cast<int>(G_ps_64);
  result.n_pairs = static_cast<int>(n_pairs_64);
  result.pairs_with_seed = 0;
  result.pairs_with_pump = 0;
  result.chi.assign(static_cast<std::size_t>(input.n_cells) *
                        static_cast<std::size_t>(result.n_pairs),
                    0.0);
  result.omega_state.resize(static_cast<std::size_t>(result.G_ps));
  result.weight_state.resize(static_cast<std::size_t>(result.G_ps));

  for (int state = 0; state < result.G_ps; ++state) {
    const int port_index = state / static_cast<int>(G_ref);
    const double lambda_cm =
        (input.lambda0_nm +
         input.ports->ports[static_cast<std::size_t>(port_index)]
             .delta_lambda_nm) *
        1.0e-7;
    result.omega_state[static_cast<std::size_t>(state)] =
        2.0 * kPi * core::constants::c_light / lambda_cm;
    result.weight_state[static_cast<std::size_t>(state)] =
        input.ports->ports[static_cast<std::size_t>(port_index)].power_weight;
  }

  for (int cell = 0; cell < input.n_cells; ++cell) {
    if (input.cell_mask[cell] == 0) {
      continue;
    }
    int pair = 0;
    for (int p = 0; p < result.G_ps; ++p) {
      const int port_p = p / static_cast<int>(G_ref);
      const int reference_p = p % static_cast<int>(G_ref);
      const int sheet_p = reference_p / input.n_impact_bins;
      const int bin_p = reference_p % input.n_impact_bins;
      const auto& seed_crossings =
          sector_ps::crossings(*input.table, cell, sheet_p);
      double seed_power = 0.0;
      bool has_seed = false;
      for (const sector_ps::CrossingView& crossing : seed_crossings) {
        if (crossing.in_limiter_zone) {
          continue;
        }
        if (input.ray_bin[crossing.ray_index] == bin_p) {
          has_seed = true;
          seed_power += crossing.P;
        }
      }

      for (int q = p + 1; q < result.G_ps; ++q) {
        const int port_q = q / static_cast<int>(G_ref);
        const int reference_q = q % static_cast<int>(G_ref);
        const int sheet_q = reference_q / input.n_impact_bins;
        double numerator = 0.0;
        double denominator = 0.0;
        bool has_pump = false;

        if (has_seed) {
          ++result.pairs_with_seed;
        }

        if (seed_power > 0.0) {
          for (const sector_ps::CrossingView& crossing : seed_crossings) {
            if (crossing.in_limiter_zone) {
              continue;
            }
            if (input.ray_bin[crossing.ray_index] != bin_p) {
              continue;
            }
            const double seed_weight = crossing.P / seed_power;
            for (int m = 0; m < input.n_section_phi; ++m) {
              const double phi =
                  (static_cast<double>(m) + 0.5) * 2.0 * kPi /
                  static_cast<double>(input.n_section_phi);
              const auto& frame_p =
                  input.ports->frames[static_cast<std::size_t>(port_p)];
              const auto& frame_q =
                  input.ports->frames[static_cast<std::size_t>(port_q)];
              const std::array<double, 3> position =
                  lab_pos_dir(frame_p, crossing.theta, phi);
              const std::array<double, 3> seed_direction =
                  lab_dir(frame_p, crossing.theta, phi, crossing.alpha,
                          sheet_p);

              const double v1 = dot(position, frame_q.e1);
              const double v2 = dot(position, frame_q.e2);
              const double vb = dot(position, frame_q.b);
              const double theta_q =
                  std::acos(std::clamp(vb, -1.0, 1.0));
              const double phi_q = std::atan2(v2, v1);
              const sector_ps::PumpLookup pump =
                  sector_ps::lookup(*input.table, cell, sheet_q, theta_q);
              if (!pump.valid) {
                continue;
              }
              has_pump = true;
              denominator += seed_weight * pump.I;

              const std::array<double, 3> pump_direction =
                  lab_dir(frame_q, theta_q, phi_q, pump.alpha, sheet_q);
              const double cos_psi = dot(seed_direction, pump_direction);
              const double mu_p = dot(seed_direction, position);
              const double mu_q = dot(pump_direction, position);
              const double ka2 = std::max(2.0 * (1.0 - cos_psi), 0.0);
              const double ka = input.cell_k_bar[cell] * std::sqrt(ka2);
              if (ka < input.k_a_floor * input.cell_k_bar[cell] ||
                  ka <= 0.0) {
                continue;
              }

              const double g =
                  (result.omega_state[static_cast<std::size_t>(q)] -
                   result.omega_state[static_cast<std::size_t>(p)] -
                   input.cell_k_bar[cell] * input.cell_u_r[cell] *
                       (mu_q - mu_p)) /
                  (ka * std::max(input.cell_c_a[cell], 1.0e-30));
              const double ga = g * input.alpha_iaw;
              const double one_g2 = 1.0 - g * g;
              const double Pg =
                  ga / (ga * ga + one_g2 * one_g2);
              const double polarization =
                  0.25 * input.f_cbet * (1.0 + cos_psi * cos_psi);
              numerator +=
                  seed_weight * pump.I * polarization * Pg;
            }
          }
        }

        if (has_pump) {
          ++result.pairs_with_pump;
        }
        if (denominator > 0.0) {
          result.chi[static_cast<std::size_t>(cell) *
                         static_cast<std::size_t>(result.n_pairs) +
                     static_cast<std::size_t>(pair)] =
              input.cell_chi_pref[cell] * numerator / denominator;
        }
        ++pair;
      }
    }
  }

  return result;
}

}  // namespace tenryu::laser::port_section
