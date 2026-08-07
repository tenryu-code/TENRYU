#include "radiation/sn_transport_1d.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kFourPi = 12.56637061435917295385;
constexpr double kFourPiOverThree = 4.18879020478639098462;
constexpr double kEnergyFloor = 1.0e-300;
constexpr double kSigmaFloor = 0.0;

[[nodiscard]] bool finite(const double value) {
  return std::isfinite(value);
}

[[nodiscard]] double finite_or_zero(const double value) {
  return finite(value) ? value : 0.0;
}

[[nodiscard]] double nonnegative_finite(const double value) {
  return finite(value) ? std::max(value, 0.0) : 0.0;
}

[[nodiscard]] std::size_t cell_group_index(const int cell,
                                           const int group,
                                           const int n_groups) {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(group);
}

[[nodiscard]] std::size_t psi_index(const int group,
                                    const int angle,
                                    const int cell,
                                    const int n_angles,
                                    const int n_cells) {
  return (static_cast<std::size_t>(group) * static_cast<std::size_t>(n_angles) +
          static_cast<std::size_t>(angle)) *
             static_cast<std::size_t>(n_cells) +
         static_cast<std::size_t>(cell);
}

[[nodiscard]] double safe_radius(const double r) {
  return std::max(finite_or_zero(r), 0.0);
}

[[nodiscard]] double face_area(const double r) {
  const double rr = safe_radius(r);
  return kFourPi * rr * rr;
}

[[nodiscard]] double shell_volume(const double r_in, const double r_out) {
  const double rin = safe_radius(r_in);
  const double rout = safe_radius(r_out);
  const double volume = kFourPiOverThree * (rout * rout * rout - rin * rin * rin);
  return std::max(volume, 0.0);
}

void compute_gauss_legendre(const int n,
                            std::vector<double>& mu,
                            std::vector<double>& weight) {
  TENRYU_ASSERT(n > 0 && (n % 2) == 0,
                "SN transport requires a positive even number of angles");
  mu.assign(static_cast<std::size_t>(n), 0.0);
  weight.assign(static_cast<std::size_t>(n), 0.0);

  constexpr double pi = 3.141592653589793238462643383279502884;
  constexpr double tol = 1.0e-15;
  const int half = n / 2;
  for (int k = 0; k < half; ++k) {
    double x = std::cos(pi * (static_cast<double>(k) + 0.75) /
                        (static_cast<double>(n) + 0.5));
    double p1 = 0.0;
    double p2 = 0.0;
    for (int iter = 0; iter < 64; ++iter) {
      p1 = 1.0;
      p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * static_cast<double>(j) - 1.0) * x * p2 -
              (static_cast<double>(j) - 1.0) * p3) /
             static_cast<double>(j);
      }
      const double pp = static_cast<double>(n) * (x * p1 - p2) / (x * x - 1.0);
      const double dx = p1 / pp;
      x -= dx;
      if (std::abs(dx) <= tol) {
        break;
      }
    }

    p1 = 1.0;
    p2 = 0.0;
    for (int j = 1; j <= n; ++j) {
      const double p3 = p2;
      p2 = p1;
      p1 = ((2.0 * static_cast<double>(j) - 1.0) * x * p2 -
            (static_cast<double>(j) - 1.0) * p3) /
           static_cast<double>(j);
    }
    const double pp = static_cast<double>(n) * (x * p1 - p2) / (x * x - 1.0);
    const double w = 2.0 / ((1.0 - x * x) * pp * pp);
    mu[static_cast<std::size_t>(k)] = -x;
    mu[static_cast<std::size_t>(n - 1 - k)] = x;
    weight[static_cast<std::size_t>(k)] = w;
    weight[static_cast<std::size_t>(n - 1 - k)] = w;
  }
}

[[nodiscard]] std::vector<double> angular_coefficients(const std::vector<double>& mu,
                                                       const std::vector<double>& weight) {
  const int n_angles = static_cast<int>(mu.size());
  std::vector<double> alpha(static_cast<std::size_t>(n_angles + 1), 0.0);
  for (int n = 0; n < n_angles; ++n) {
    alpha[static_cast<std::size_t>(n + 1)] =
        alpha[static_cast<std::size_t>(n)] - mu[static_cast<std::size_t>(n)] *
                                                weight[static_cast<std::size_t>(n)];
  }
  alpha.front() = 0.0;
  alpha.back() = 0.0;
  for (double& value : alpha) {
    if (std::abs(value) < 1.0e-14) {
      value = 0.0;
    }
    value = std::max(value, 0.0);
  }
  return alpha;
}

}  // namespace

SNTransport1DResult solve_sn_transport_1d(const double* const sigma_a,
                                          const double* const sigma_s,
                                          const double* const planck_source,
                                          const double* const node_r,
                                          const double* const vol,
                                          const int n_cells,
                                          const int n_groups,
                                          const SNTransport1DConfig& config) {
  TENRYU_ASSERT(sigma_a != nullptr, "SN transport requires sigma_a");
  TENRYU_ASSERT(sigma_s != nullptr, "SN transport requires sigma_s");
  TENRYU_ASSERT(planck_source != nullptr, "SN transport requires planck_source");
  TENRYU_ASSERT(node_r != nullptr, "SN transport requires node_r");
  TENRYU_ASSERT(vol != nullptr, "SN transport requires cell volumes");
  TENRYU_ASSERT(n_cells >= 0, "SN transport requires n_cells >= 0");
  TENRYU_ASSERT(n_groups >= 0, "SN transport requires n_groups >= 0");
  TENRYU_ASSERT(config.n_angles > 0 && (config.n_angles % 2) == 0,
                "SN transport n_angles must be positive and even");
  TENRYU_ASSERT(config.max_iterations >= 0,
                "SN transport max_iterations must be >= 0");
  TENRYU_ASSERT(config.convergence_tol >= 0.0,
                "SN transport convergence_tol must be >= 0");

  SNTransport1DResult result{};
  const std::size_t n_cells_us = static_cast<std::size_t>(n_cells);
  const std::size_t n_groups_us = static_cast<std::size_t>(n_groups);
  const std::size_t n_total = n_cells_us * n_groups_us;
  result.chi.assign(n_total, 1.0 / 3.0);
  result.E_sn.assign(n_total, 0.0);
  if (n_cells == 0 || n_groups == 0) {
    result.converged = true;
    return result;
  }

  std::vector<double> mu;
  std::vector<double> weight;
  compute_gauss_legendre(config.n_angles, mu, weight);
  const std::vector<double> alpha = angular_coefficients(mu, weight);

  std::vector<double> area_in(n_cells_us, 0.0);
  std::vector<double> area_out(n_cells_us, 0.0);
  std::vector<double> volume(n_cells_us, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    area_in[c_us] = face_area(node_r[c]);
    area_out[c_us] = face_area(node_r[c + 1]);
    const double V_input = nonnegative_finite(vol[c]);
    volume[c_us] = (V_input > 0.0) ? V_input : shell_volume(node_r[c], node_r[c + 1]);
  }

  std::vector<double> phi(n_total, 0.0);
  std::vector<double> phi_old(n_total, 0.0);
  std::vector<double> psi_bar(static_cast<std::size_t>(config.n_angles) * n_total, 0.0);
  std::vector<double> angular_edge(n_cells_us, 0.0);
  std::vector<double> inner_boundary(static_cast<std::size_t>(config.n_angles), 0.0);
  std::vector<double> phi_next(n_total, 0.0);

  const int max_iterations = std::max(config.max_iterations, 1);
  for (int iter = 0; iter < max_iterations; ++iter) {
    phi_old = phi;
    std::fill(phi_next.begin(), phi_next.end(), 0.0);
    std::fill(psi_bar.begin(), psi_bar.end(), 0.0);

    for (int g = 0; g < n_groups; ++g) {
      std::fill(angular_edge.begin(), angular_edge.end(), 0.0);
      std::fill(inner_boundary.begin(), inner_boundary.end(), 0.0);

      for (int n = 0; n < config.n_angles; ++n) {
        const double mu_n = mu[static_cast<std::size_t>(n)];
        if (mu_n == 0.0) {
          continue;
        }
        const double abs_mu = std::abs(mu_n);
        const double alpha_prev = alpha[static_cast<std::size_t>(n)];
        const double alpha_next = alpha[static_cast<std::size_t>(n + 1)];

        if (mu_n < 0.0) {
          double incoming = 0.0;
          for (int c = n_cells - 1; c >= 0; --c) {
            const std::size_t c_us = static_cast<std::size_t>(c);
            const std::size_t cg = cell_group_index(c, g, n_groups);
            const double V = volume[c_us];
            const double sigma_t =
                std::max(nonnegative_finite(sigma_a[cg]) +
                             nonnegative_finite(sigma_s[cg]),
                         kSigmaFloor);
            const double source =
                0.5 * nonnegative_finite(planck_source[cg]) +
                0.5 * nonnegative_finite(sigma_s[cg]) *
                    std::max(finite_or_zero(phi_old[cg]), 0.0);
            const double edge_in = std::max(angular_edge[c_us], 0.0);
            const double denom =
                2.0 * abs_mu * area_in[c_us] + 2.0 * alpha_next + sigma_t * V;
            double average = 0.0;
            if (denom > 0.0 && finite(denom)) {
              const double numer =
                  V * source + abs_mu * (area_in[c_us] + area_out[c_us]) * incoming +
                  (alpha_prev + alpha_next) * edge_in;
              average = numer / denom;
            }
            average = std::max(finite_or_zero(average), 0.0);
            double outgoing = 2.0 * average - incoming;
            double edge_out = 2.0 * average - edge_in;
            if (!finite(outgoing) || outgoing < 0.0) {
              outgoing = 0.0;
            }
            if (!finite(edge_out) || edge_out < 0.0) {
              edge_out = 0.0;
            }
            psi_bar[psi_index(g, n, c, config.n_angles, n_cells)] = average;
            phi_next[cg] += weight[static_cast<std::size_t>(n)] * average;
            angular_edge[c_us] = edge_out;
            incoming = outgoing;
          }
          inner_boundary[static_cast<std::size_t>(n)] = incoming;
        } else {
          const int reflected = config.n_angles - 1 - n;
          double incoming =
              (reflected >= 0 && reflected < config.n_angles)
                  ? std::max(inner_boundary[static_cast<std::size_t>(reflected)], 0.0)
                  : 0.0;
          for (int c = 0; c < n_cells; ++c) {
            const std::size_t c_us = static_cast<std::size_t>(c);
            const std::size_t cg = cell_group_index(c, g, n_groups);
            const double V = volume[c_us];
            const double sigma_t =
                std::max(nonnegative_finite(sigma_a[cg]) +
                             nonnegative_finite(sigma_s[cg]),
                         kSigmaFloor);
            const double source =
                0.5 * nonnegative_finite(planck_source[cg]) +
                0.5 * nonnegative_finite(sigma_s[cg]) *
                    std::max(finite_or_zero(phi_old[cg]), 0.0);
            const double edge_in = std::max(angular_edge[c_us], 0.0);
            const double denom =
                2.0 * mu_n * area_out[c_us] + 2.0 * alpha_next + sigma_t * V;
            double average = 0.0;
            if (denom > 0.0 && finite(denom)) {
              const double numer =
                  V * source + mu_n * (area_in[c_us] + area_out[c_us]) * incoming +
                  (alpha_prev + alpha_next) * edge_in;
              average = numer / denom;
            }
            average = std::max(finite_or_zero(average), 0.0);
            double outgoing = 2.0 * average - incoming;
            double edge_out = 2.0 * average - edge_in;
            if (!finite(outgoing) || outgoing < 0.0) {
              outgoing = 0.0;
            }
            if (!finite(edge_out) || edge_out < 0.0) {
              edge_out = 0.0;
            }
            psi_bar[psi_index(g, n, c, config.n_angles, n_cells)] = average;
            phi_next[cg] += weight[static_cast<std::size_t>(n)] * average;
            angular_edge[c_us] = edge_out;
            incoming = outgoing;
          }
        }
      }
    }

    double max_error = 0.0;
    for (std::size_t i = 0; i < n_total; ++i) {
      const double denom = std::max(std::abs(phi_next[i]), kEnergyFloor);
      max_error =
          std::max(max_error, std::abs(phi_next[i] - phi_old[i]) / denom);
    }
    phi.swap(phi_next);
    result.iterations = iter + 1;
    result.convergence_error = max_error;
    result.converged = result.convergence_error <= config.convergence_tol;
    if (result.converged) {
      break;
    }
  }

  for (int c = 0; c < n_cells; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t cg = cell_group_index(c, g, n_groups);
      double moment0 = 0.0;
      double moment2 = 0.0;
      for (int n = 0; n < config.n_angles; ++n) {
        const double psi =
            std::max(psi_bar[psi_index(g, n, c, config.n_angles, n_cells)], 0.0);
        const double w = weight[static_cast<std::size_t>(n)];
        const double mu_n = mu[static_cast<std::size_t>(n)];
        moment0 += w * psi;
        moment2 += w * mu_n * mu_n * psi;
      }
      const double E = moment0 / core::constants::c_light;
      const double P_rr = moment2 / core::constants::c_light;
      result.E_sn[cg] = finite(E) ? std::max(E, 0.0) : 0.0;
      const double denom = std::max(result.E_sn[cg], kEnergyFloor);
      const double chi = P_rr / denom;
      result.chi[cg] = finite(chi) ? std::clamp(chi, 0.0, 1.0) : (1.0 / 3.0);
    }
  }

  return result;
}

}  // namespace tenryu::radiation
