#include "hydro/reale_cvt_patch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tenryu::hydro {
namespace {

struct Assignment {
  std::vector<double> centroid_r;
  std::vector<double> centroid_z;
  std::vector<double> monitor_mass;
  std::vector<int> sample_count;
  int n_empty = 0;
};

double movable_coefficient_of_variation(
    const std::vector<double>& monitor_mass,
    const std::vector<int>& sample_count,
    const std::vector<std::uint8_t>& movable) {
  double mass_sum = 0.0;
  int n_nonempty = 0;
  for (std::size_t i = 0; i < monitor_mass.size(); ++i) {
    if (movable[i] == 0 || sample_count[i] == 0) {
      continue;
    }
    mass_sum += monitor_mass[i];
    ++n_nonempty;
  }
  if (n_nonempty == 0) {
    return 0.0;
  }

  const double mean = mass_sum / static_cast<double>(n_nonempty);
  if (mean == 0.0) {
    return 0.0;
  }

  double squared_deviation_sum = 0.0;
  for (std::size_t i = 0; i < monitor_mass.size(); ++i) {
    if (movable[i] == 0 || sample_count[i] == 0) {
      continue;
    }
    const double deviation = monitor_mass[i] - mean;
    squared_deviation_sum += deviation * deviation;
  }
  const double population_sigma =
      std::sqrt(squared_deviation_sum / static_cast<double>(n_nonempty));
  return population_sigma / mean;
}

Assignment assign_samples(
    const std::vector<double>& gen_r,
    const std::vector<double>& gen_z,
    const std::function<double(double, double)>& monitor,
    const CvtDomain& domain,
    const int grid_nr,
    const int grid_nz) {
  const std::size_t n_generators = gen_r.size();
  Assignment assignment;
  assignment.centroid_r.assign(n_generators, 0.0);
  assignment.centroid_z.assign(n_generators, 0.0);
  assignment.monitor_mass.assign(n_generators, 0.0);
  assignment.sample_count.assign(n_generators, 0);

  const double dr =
      (domain.r1 - domain.r0) / static_cast<double>(grid_nr);
  const double dz =
      (domain.z1 - domain.z0) / static_cast<double>(grid_nz);
  for (int ir = 0; ir < grid_nr; ++ir) {
    const double r =
        domain.r0 + (static_cast<double>(ir) + 0.5) * dr;
    for (int iz = 0; iz < grid_nz; ++iz) {
      const double z =
          domain.z0 + (static_cast<double>(iz) + 0.5) * dz;
      const double weight = monitor(r, z);
      if (!std::isfinite(weight) || weight <= 0.0) {
        throw std::invalid_argument(
            "CVT monitor values must be finite and positive");
      }

      std::size_t nearest = 0;
      double nearest_distance2 =
          (r - gen_r[0]) * (r - gen_r[0]) +
          (z - gen_z[0]) * (z - gen_z[0]);
      for (std::size_t generator = 1; generator < n_generators;
           ++generator) {
        const double distance2 =
            (r - gen_r[generator]) * (r - gen_r[generator]) +
            (z - gen_z[generator]) * (z - gen_z[generator]);
        if (distance2 < nearest_distance2) {
          nearest = generator;
          nearest_distance2 = distance2;
        }
      }

      assignment.monitor_mass[nearest] += weight;
      assignment.centroid_r[nearest] += weight * r;
      assignment.centroid_z[nearest] += weight * z;
      ++assignment.sample_count[nearest];
    }
  }

  for (std::size_t generator = 0; generator < n_generators; ++generator) {
    if (assignment.sample_count[generator] == 0) {
      assignment.centroid_r[generator] = gen_r[generator];
      assignment.centroid_z[generator] = gen_z[generator];
      ++assignment.n_empty;
      continue;
    }
    assignment.centroid_r[generator] /=
        assignment.monitor_mass[generator];
    assignment.centroid_z[generator] /=
        assignment.monitor_mass[generator];
  }
  return assignment;
}

void validate_inputs(const std::vector<double>& gen_r,
                     const std::vector<double>& gen_z,
                     const std::vector<std::uint8_t>& movable,
                     const CvtDomain& domain,
                     const int n_iters,
                     const int grid_nr,
                     const int grid_nz,
                     const double damping) {
  if (gen_r.empty() || gen_r.size() != gen_z.size()) {
    throw std::invalid_argument(
        "CVT generator coordinate arrays must be non-empty and equal-sized");
  }
  if (movable.size() != gen_r.size()) {
    throw std::invalid_argument(
        "CVT patch mask must match the generator array size");
  }
  bool has_movable = false;
  bool has_frozen = false;
  for (const std::uint8_t flag : movable) {
    has_movable = has_movable || flag != 0;
    has_frozen = has_frozen || flag == 0;
  }
  if (!has_movable || !has_frozen) {
    throw std::invalid_argument(
        "CVT patch must contain movable and frozen generators");
  }
  if (n_iters < 0) {
    throw std::invalid_argument(
        "CVT relocation iteration count must be non-negative");
  }
  if (grid_nr < 8 || grid_nz < 8) {
    throw std::invalid_argument(
        "CVT quadrature grid dimensions must each be at least eight");
  }
  if (damping <= 0.0 || damping > 1.0) {
    throw std::invalid_argument(
        "CVT relocation damping must be in the interval (0, 1]");
  }
  if (domain.r1 <= domain.r0 || domain.z1 <= domain.z0) {
    throw std::invalid_argument(
        "CVT domain must have positive extent in both coordinates");
  }
}

}  // namespace

CvtPatchResult reale_cvt_patch_relax(
    std::vector<double>& gen_r,
    std::vector<double>& gen_z,
    const std::vector<std::uint8_t>& movable,
    const std::function<double(double, double)>& monitor,
    const CvtDomain& domain,
    const int n_iters,
    const int grid_nr,
    const int grid_nz,
    const double damping) {
  validate_inputs(
      gen_r, gen_z, movable, domain, n_iters, grid_nr, grid_nz, damping);

  const std::vector<double> initial_r = gen_r;
  const std::vector<double> initial_z = gen_z;
  Assignment assignment =
      assign_samples(gen_r, gen_z, monitor, domain, grid_nr, grid_nz);
  const std::vector<double> initial_monitor_mass =
      assignment.monitor_mass;

  CvtPatchResult result{};
  result.cv_initial = movable_coefficient_of_variation(
      assignment.monitor_mass, assignment.sample_count, movable);

  for (int iteration = 0; iteration < n_iters; ++iteration) {
    for (std::size_t generator = 0; generator < gen_r.size();
         ++generator) {
      if (movable[generator] == 0) {
        continue;
      }
      gen_r[generator] = std::clamp(
          gen_r[generator] +
              damping *
                  (assignment.centroid_r[generator] - gen_r[generator]),
          domain.r0, domain.r1);
      gen_z[generator] = std::clamp(
          gen_z[generator] +
              damping *
                  (assignment.centroid_z[generator] - gen_z[generator]),
          domain.z0, domain.z1);
    }
    assignment =
        assign_samples(gen_r, gen_z, monitor, domain, grid_nr, grid_nz);
  }

  result.n_empty = assignment.n_empty;
  result.cv_final = movable_coefficient_of_variation(
      assignment.monitor_mass, assignment.sample_count, movable);
  for (std::size_t generator = 0; generator < gen_r.size();
       ++generator) {
    if (movable[generator] != 0) {
      const double displacement =
          std::hypot(gen_r[generator] - initial_r[generator],
                     gen_z[generator] - initial_z[generator]);
      if (displacement > 0.0) {
        ++result.n_moved;
      }
      result.max_displacement =
          std::max(result.max_displacement, displacement);
      continue;
    }

    const double initial_mass = initial_monitor_mass[generator];
    const double mass_change =
        std::abs(assignment.monitor_mass[generator] - initial_mass);
    const double relative_drift =
        initial_mass > 0.0
            ? mass_change / initial_mass
            : (mass_change == 0.0
                   ? 0.0
                   : std::numeric_limits<double>::infinity());
    result.boundary_drift =
        std::max(result.boundary_drift, relative_drift);
  }
  return result;
}

}  // namespace tenryu::hydro
