#include "laser/sector_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

namespace tenryu::laser::sector_adapter {
namespace {

constexpr double kPi = 3.14159265358979323846;

struct PathWork {
  sector_ps::RayPath path;
  std::size_t turning_node = 0;
};

double record_alpha(const float mu_value) {
  const double mu =
      std::clamp(static_cast<double>(mu_value), -1.0, 1.0);
  return std::asin(std::sqrt(std::max(0.0, 1.0 - mu * mu)));
}

double transverse_over_radius(const double mu_value, const double radius) {
  if (!(radius > 0.0)) {
    return 0.0;
  }
  const double mu = std::clamp(mu_value, -1.0, 1.0);
  return std::sqrt(std::max(0.0, 1.0 - mu * mu)) / radius;
}

std::size_t turning_node(const std::vector<double>& radius) {
  return static_cast<std::size_t>(
      std::distance(radius.begin(),
                    std::min_element(radius.begin(), radius.end())));
}

std::optional<double> interpolate_theta_on_leg(
    const PathWork& neighbor, const double radius, const int leg) {
  const std::size_t first = (leg == 0) ? 0 : neighbor.turning_node;
  const std::size_t last =
      (leg == 0) ? neighbor.turning_node : neighbor.path.r.size() - 1;

  for (std::size_t node = first; node <= last; ++node) {
    if (neighbor.path.r[node] == radius) {
      return neighbor.path.theta[node];
    }
  }
  for (std::size_t node = first; node < last; ++node) {
    const double r0 = neighbor.path.r[node];
    const double r1 = neighbor.path.r[node + 1];
    if (r0 == r1 || radius < std::min(r0, r1) ||
        radius > std::max(r0, r1)) {
      continue;
    }
    const double weight = (radius - r0) / (r1 - r0);
    return neighbor.path.theta[node] +
           weight * (neighbor.path.theta[node + 1] -
                     neighbor.path.theta[node]);
  }
  return std::nullopt;
}

}  // namespace

std::vector<sector_ps::RayPath> build_ray_paths(
    const AdapterInput& input,
    std::vector<int>* source_ray_indices) {
  if (source_ray_indices != nullptr) {
    source_ray_indices->clear();
  }
  std::vector<PathWork> work;
  if (input.n_rays <= 0 || input.ray_rec_offset == nullptr ||
      input.rec_cell == nullptr || input.rec_mu == nullptr ||
      input.rec_ds == nullptr || input.ray_P0 == nullptr ||
      input.ray_impact_parameter == nullptr ||
      input.r_edges == nullptr || input.n_cells <= 0) {
    return {};
  }

  work.reserve(static_cast<std::size_t>(input.n_rays));
  int zero_record_rays = 0;
  for (int ray = 0; ray < input.n_rays; ++ray) {
    const std::int64_t begin = input.ray_rec_offset[ray];
    const std::int64_t end = input.ray_rec_offset[ray + 1];
    if (end <= begin) {
      ++zero_record_rays;
      continue;
    }

    bool valid_cells = true;
    for (std::int64_t record = begin; record < end; ++record) {
      const int cell = input.rec_cell[record];
      if (cell < 0 || cell >= input.n_cells) {
        valid_cells = false;
        break;
      }
    }
    if (!valid_cells) {
      continue;
    }

    std::vector<std::int64_t> geometric_records;
    geometric_records.reserve(static_cast<std::size_t>(end - begin));
    for (std::int64_t record = begin; record < end; ++record) {
      if (input.rec_ds[record] > 0.0) {
        geometric_records.push_back(record);
      }
    }
    if (geometric_records.empty()) {
      ++zero_record_rays;
      continue;
    }

    const std::size_t n_records = geometric_records.size();
    PathWork entry;
    sector_ps::RayPath& path = entry.path;
    path.ray_index = static_cast<int>(work.size());
    path.impact_parameter =
        std::max(0.0, input.ray_impact_parameter[ray]);
    path.r.resize(n_records + 1);
    path.theta.resize(n_records + 1);
    path.P.resize(n_records + 1);
    path.area.assign(n_records + 1, 0.0);
    path.alpha.resize(n_records + 1);

    std::vector<double> node_mu(n_records + 1, 0.0);
    const std::int64_t first_record = geometric_records.front();
    const int first_cell = input.rec_cell[first_record];
    if (n_records > 1) {
      const int next_cell = input.rec_cell[geometric_records[1]];
      if (next_cell < first_cell) {
        path.r[0] = input.r_edges[first_cell + 1];
      } else if (next_cell > first_cell) {
        path.r[0] = input.r_edges[first_cell];
      } else {
        path.r[0] = (input.rec_mu[first_record] < 0.0f)
                        ? input.r_edges[first_cell + 1]
                        : input.r_edges[first_cell];
      }
    } else {
      path.r[0] = (input.rec_mu[first_record] < 0.0f)
                      ? input.r_edges[first_cell + 1]
                      : input.r_edges[first_cell];
    }
    node_mu[0] = static_cast<double>(input.rec_mu[first_record]);
    for (std::size_t k = 0; k < n_records; ++k) {
      const std::int64_t record = geometric_records[k];
      const int cell = input.rec_cell[record];
      if (k + 1 < n_records) {
        const int next_cell = input.rec_cell[geometric_records[k + 1]];
        if (next_cell < cell) {
          path.r[k + 1] = input.r_edges[cell];
        } else if (next_cell > cell) {
          path.r[k + 1] = input.r_edges[cell + 1];
        } else {
          path.r[k + 1] = (input.rec_mu[record] < 0.0f)
                              ? input.r_edges[cell]
                              : input.r_edges[cell + 1];
        }
      } else {
        path.r[k + 1] = (input.rec_mu[record] < 0.0f)
                            ? input.r_edges[cell]
                            : input.r_edges[cell + 1];
      }
      node_mu[k + 1] = static_cast<double>(input.rec_mu[record]);
    }

    const double r0 = path.r.front();
    const double entry_ratio =
        (r0 > 0.0) ? std::min(1.0, path.impact_parameter / r0) : 1.0;
    path.theta[0] = std::asin(entry_ratio);
    for (std::size_t k = 0; k < n_records; ++k) {
      const std::int64_t record = geometric_records[k];
      const double r_mid = 0.5 * (path.r[k] + path.r[k + 1]);
      const double mu_mid = 0.5 * (node_mu[k] + node_mu[k + 1]);
      const double dtheta =
          input.rec_ds[record] / 6.0 *
          (transverse_over_radius(node_mu[k], path.r[k]) +
           4.0 * transverse_over_radius(mu_mid, r_mid) +
           transverse_over_radius(node_mu[k + 1], path.r[k + 1]));
      path.theta[k + 1] = path.theta[k] + dtheta;
    }

    for (std::size_t node = 0; node <= n_records; ++node) {
      path.alpha[node] = record_alpha(static_cast<float>(node_mu[node]));
    }

    double power = input.ray_P0[ray];
    if (!std::isfinite(power) || power < 0.0) {
      power = 0.0;
    }
    path.P[0] = power;
    std::size_t geometric_node = 0;
    for (std::int64_t record = begin; record < end; ++record) {
      if (input.attenuate != nullptr) {
        power =
            input.attenuate(power, record, input.attenuation_context);
      }
      if (input.rec_ds[record] > 0.0) {
        ++geometric_node;
      }
      path.P[geometric_node] = power;
    }

    entry.turning_node = turning_node(path.r);
    work.push_back(std::move(entry));
    if (source_ray_indices != nullptr) {
      source_ray_indices->push_back(ray);
    }
  }
  static_cast<void>(zero_record_rays);

  for (std::size_t ray = 0; ray < work.size(); ++ray) {
    sector_ps::RayPath& path = work[ray].path;
    for (std::size_t node = 0; node < path.r.size(); ++node) {
      const int leg = (node <= work[ray].turning_node) ? 0 : 1;
      std::optional<double> inner_theta;
      std::optional<double> outer_theta;
      if (ray > 0) {
        inner_theta =
            interpolate_theta_on_leg(work[ray - 1], path.r[node], leg);
      }
      if (ray + 1 < work.size()) {
        outer_theta =
            interpolate_theta_on_leg(work[ray + 1], path.r[node], leg);
      }

      double dtheta_eff = 0.0;
      if (inner_theta.has_value() && outer_theta.has_value()) {
        dtheta_eff =
            0.5 * std::abs(*outer_theta - *inner_theta);
      } else if (inner_theta.has_value()) {
        dtheta_eff =
            std::abs(path.theta[node] - *inner_theta);
      } else if (outer_theta.has_value()) {
        dtheta_eff =
            std::abs(*outer_theta - path.theta[node]);
      }
      dtheta_eff = std::max(dtheta_eff, 1.0e-12);

      const double sin_eff =
          std::max(std::sin(path.theta[node]),
                   std::sin(0.5 * dtheta_eff));
      const double cos_alpha =
          std::max(std::cos(path.alpha[node]), 1.0e-6);
      path.area[node] =
          2.0 * kPi * path.r[node] * path.r[node] *
          dtheta_eff * sin_eff * cos_alpha;
    }
  }

  std::vector<sector_ps::RayPath> result;
  result.reserve(work.size());
  for (PathWork& entry : work) {
    result.push_back(std::move(entry.path));
  }
  return result;
}

}  // namespace tenryu::laser::sector_adapter
