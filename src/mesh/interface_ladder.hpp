#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "core/error.hpp"

namespace tenryu::mesh::assembly {

// One side's sizing request on an interface interval (design doc §0.2).
struct SizingProposal {
  double soft_target = 0.0;
  double hard_min = 0.0;
  double hard_max = 0.0;
  double weight = 1.0;
};

// Result of merging two proposals at one point.
struct MergedSizing {
  bool feasible = false;
  double h = 0.0;
  double lo = 0.0;
  double hi = 0.0;
};

inline MergedSizing merge_sizing(const SizingProposal& a,
                                 const SizingProposal& b) {
  const auto validate = [](const SizingProposal& proposal) {
    TENRYU_ASSERT(std::isfinite(proposal.soft_target) &&
                      proposal.soft_target > 0.0,
                  "sizing soft target must be positive and finite");
    TENRYU_ASSERT(std::isfinite(proposal.hard_min) &&
                      proposal.hard_min >= 0.0,
                  "sizing hard minimum must be non-negative and finite");
    TENRYU_ASSERT(std::isfinite(proposal.hard_max) &&
                      proposal.hard_max > 0.0,
                  "sizing hard maximum must be positive and finite");
    TENRYU_ASSERT(proposal.hard_min <= proposal.hard_max,
                  "sizing hard minimum must not exceed hard maximum");
    TENRYU_ASSERT(std::isfinite(proposal.weight) && proposal.weight > 0.0,
                  "sizing weight must be positive and finite");
  };
  validate(a);
  validate(b);

  MergedSizing merged;
  merged.lo = std::max(a.hard_min, b.hard_min);
  merged.hi = std::min(a.hard_max, b.hard_max);
  if (merged.lo > merged.hi) {
    return merged;
  }

  const double log_mean =
      (a.weight * std::log(a.soft_target) +
       b.weight * std::log(b.soft_target)) /
      (a.weight + b.weight);
  const double value = std::exp(log_mean);
  merged.feasible = true;
  merged.h = std::clamp(value, merged.lo, merged.hi);
  return merged;
}

namespace detail {

inline std::vector<double> cumulative_monitor(
    const double length,
    const std::function<double(double)>& h_of_s,
    const int n_quad) {
  TENRYU_ASSERT(length > 0.0, "monitor length must be positive");
  TENRYU_ASSERT(n_quad >= 1, "monitor quadrature count must be positive");

  const double panel_width = length / static_cast<double>(n_quad);
  std::vector<double> cumulative(static_cast<std::size_t>(n_quad) + 1U,
                                 0.0);
  for (int k = 0; k < n_quad; ++k) {
    const double midpoint =
        (static_cast<double>(k) + 0.5) * panel_width;
    const double h = h_of_s(midpoint);
    TENRYU_ASSERT(std::isfinite(h) && h > 0.0,
                  "monitor sizing samples must be positive and finite");
    const double panel_mass = panel_width / h;
    TENRYU_ASSERT(std::isfinite(panel_mass),
                  "monitor panel mass must be finite");
    cumulative[static_cast<std::size_t>(k) + 1U] =
        cumulative[static_cast<std::size_t>(k)] + panel_mass;
    TENRYU_ASSERT(
        std::isfinite(cumulative[static_cast<std::size_t>(k) + 1U]),
        "cumulative monitor mass must be finite");
  }
  return cumulative;
}

inline double interpolate_cumulative(const std::vector<double>& cumulative,
                                     const double length,
                                     const double s) {
  if (s <= 0.0) {
    return 0.0;
  }
  if (s >= length) {
    return cumulative.back();
  }

  const std::size_t n_quad = cumulative.size() - 1U;
  const double panel_width = length / static_cast<double>(n_quad);
  const double scaled = s / panel_width;
  const std::size_t panel = std::min(
      static_cast<std::size_t>(scaled), n_quad - 1U);
  const double panel_left = static_cast<double>(panel) * panel_width;
  const double fraction = (s - panel_left) / panel_width;
  return cumulative[panel] +
         fraction * (cumulative[panel + 1U] - cumulative[panel]);
}

}  // namespace detail

// Monitor-equidistributed ladder on [0, length] for target size h(s).
inline std::vector<double> equidistribute_monitor(
    const double length,
    const std::function<double(double)>& h_of_s,
    const int n_intervals,
    const int n_quad = 4096) {
  TENRYU_ASSERT(length > 0.0, "monitor length must be positive");
  TENRYU_ASSERT(n_intervals >= 1,
                "monitor interval count must be positive");
  TENRYU_ASSERT(n_quad >= n_intervals,
                "monitor quadrature count must cover all intervals");

  const std::vector<double> cumulative =
      detail::cumulative_monitor(length, h_of_s, n_quad);
  const double total_mass = cumulative.back();
  TENRYU_ASSERT(total_mass > 0.0,
                "total monitor mass must be positive");

  std::vector<double> coordinates(
      static_cast<std::size_t>(n_intervals) + 1U, 0.0);
  coordinates.back() = length;
  for (int j = 1; j < n_intervals; ++j) {
    const double target =
        total_mass * static_cast<double>(j) /
        static_cast<double>(n_intervals);
    double lo = 0.0;
    double hi = length;
    for (int iteration = 0; iteration < 64; ++iteration) {
      const double mid = 0.5 * (lo + hi);
      if (detail::interpolate_cumulative(cumulative, length, mid) < target) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    coordinates[static_cast<std::size_t>(j)] = 0.5 * (lo + hi);
  }

  for (std::size_t j = 1; j < coordinates.size(); ++j) {
    TENRYU_ASSERT(coordinates[j - 1U] < coordinates[j],
                  "monitor coordinates must be strictly increasing");
  }
  return coordinates;
}

// Ideal (real-valued) count of an interval under sizing h(s).
inline double monitor_mass(
    const double length,
    const std::function<double(double)>& h_of_s,
    const int n_quad = 4096) {
  return detail::cumulative_monitor(length, h_of_s, n_quad).back();
}

// Largest-remainder apportionment of a total count over ideal masses.
inline std::vector<int> apportion_counts(
    const std::vector<double>& interval_masses,
    const int total_count,
    const int min_per_interval) {
  TENRYU_ASSERT(!interval_masses.empty(),
                "apportionment requires at least one interval");
  TENRYU_ASSERT(min_per_interval >= 0,
                "minimum count per interval must be non-negative");
  TENRYU_ASSERT(
      static_cast<long long>(total_count) >=
          static_cast<long long>(min_per_interval) *
              static_cast<long long>(interval_masses.size()),
      "total count cannot satisfy the per-interval minimum");

  double mass_sum = 0.0;
  for (const double mass : interval_masses) {
    TENRYU_ASSERT(std::isfinite(mass) && mass > 0.0,
                  "apportionment masses must be positive and finite");
    mass_sum += mass;
    TENRYU_ASSERT(std::isfinite(mass_sum),
                  "apportionment mass sum must be finite");
  }

  const std::size_t n_intervals = interval_masses.size();
  std::vector<int> counts(n_intervals, 0);
  std::vector<double> remainders(n_intervals, 0.0);
  std::vector<std::size_t> order(n_intervals, 0U);
  long long allocated = 0;
  for (std::size_t i = 0; i < n_intervals; ++i) {
    const double ideal =
        static_cast<double>(total_count) * interval_masses[i] / mass_sum;
    const double floored = std::floor(ideal);
    counts[i] = std::max(min_per_interval, static_cast<int>(floored));
    remainders[i] = ideal - floored;
    order[i] = i;
    allocated += counts[i];
  }

  if (allocated > total_count) {
    std::sort(order.begin(), order.end(),
              [&remainders](const std::size_t lhs, const std::size_t rhs) {
                if (remainders[lhs] != remainders[rhs]) {
                  return remainders[lhs] < remainders[rhs];
                }
                return lhs < rhs;
              });
    long long excess = allocated - total_count;
    for (const std::size_t i : order) {
      const long long available = counts[i] - min_per_interval;
      const long long reduction = std::min(excess, available);
      counts[i] -= static_cast<int>(reduction);
      excess -= reduction;
      if (excess == 0) {
        break;
      }
    }
    TENRYU_ASSERT(excess == 0,
                  "apportionment could not reduce to the requested total");
    allocated = total_count;
  }

  if (allocated < total_count) {
    std::sort(order.begin(), order.end(),
              [&remainders](const std::size_t lhs, const std::size_t rhs) {
                if (remainders[lhs] != remainders[rhs]) {
                  return remainders[lhs] > remainders[rhs];
                }
                return lhs < rhs;
              });
    long long remaining = total_count - allocated;
    for (const std::size_t i : order) {
      if (remaining == 0) {
        break;
      }
      ++counts[i];
      --remaining;
    }
    TENRYU_ASSERT(remaining == 0,
                  "apportionment could not reach the requested total");
  }

  long long count_sum = 0;
  for (const int count : counts) {
    TENRYU_ASSERT(count >= min_per_interval,
                  "apportioned count violates the interval minimum");
    count_sum += count;
  }
  TENRYU_ASSERT(count_sum == total_count,
                "apportioned counts must sum to the requested total");
  return counts;
}

// Landmark-split ladder with independently equidistributed sub-intervals.
inline std::vector<double> equidistribute_with_landmarks(
    const double length,
    const std::function<double(double)>& h_of_s,
    const std::vector<double>& interior_landmarks,
    const int n_intervals_total,
    const int min_per_interval = 1,
    const int n_quad_per_interval = 4096) {
  TENRYU_ASSERT(length > 0.0, "monitor length must be positive");
  double previous = 0.0;
  for (const double landmark : interior_landmarks) {
    TENRYU_ASSERT(landmark > previous && landmark < length,
                  "landmarks must be strictly increasing and interior");
    previous = landmark;
  }

  std::vector<double> endpoints;
  endpoints.reserve(interior_landmarks.size() + 2U);
  endpoints.push_back(0.0);
  endpoints.insert(endpoints.end(), interior_landmarks.begin(),
                   interior_landmarks.end());
  endpoints.push_back(length);

  std::vector<double> masses;
  masses.reserve(endpoints.size() - 1U);
  for (std::size_t interval = 0; interval + 1U < endpoints.size();
       ++interval) {
    const double start = endpoints[interval];
    const double interval_length = endpoints[interval + 1U] - start;
    const auto local_h = [&h_of_s, start](const double local_s) {
      return h_of_s(start + local_s);
    };
    masses.push_back(
        monitor_mass(interval_length, local_h, n_quad_per_interval));
  }

  const std::vector<int> counts =
      apportion_counts(masses, n_intervals_total, min_per_interval);
  std::vector<double> coordinates;
  coordinates.reserve(static_cast<std::size_t>(n_intervals_total) + 1U);
  coordinates.push_back(0.0);
  for (std::size_t interval = 0; interval < counts.size(); ++interval) {
    const double start = endpoints[interval];
    const double end = endpoints[interval + 1U];
    const double interval_length = end - start;
    const auto local_h = [&h_of_s, start](const double local_s) {
      return h_of_s(start + local_s);
    };
    const std::vector<double> local = equidistribute_monitor(
        interval_length, local_h, counts[interval], n_quad_per_interval);
    for (int j = 1; j <= counts[interval]; ++j) {
      if (j == counts[interval]) {
        coordinates.push_back(end);
      } else {
        coordinates.push_back(start + local[static_cast<std::size_t>(j)]);
      }
    }
  }

  TENRYU_ASSERT(
      coordinates.size() ==
          static_cast<std::size_t>(n_intervals_total) + 1U,
      "landmark ladder interval count must match the requested total");
  return coordinates;
}

}  // namespace tenryu::mesh::assembly
