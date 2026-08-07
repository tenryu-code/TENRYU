#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace tenryu::core {

namespace radiation_group_detail {

inline std::vector<double> log_spaced_segment(const double lo,
                                              const double hi,
                                              const int n_groups) {
  std::vector<double> out;
  if (!(hi > lo) || n_groups <= 0) {
    return out;
  }
  out.reserve(static_cast<std::size_t>(n_groups + 1));
  const double log_lo = std::log(lo);
  const double log_hi = std::log(hi);
  for (int i = 0; i <= n_groups; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(n_groups);
    out.push_back(std::exp((1.0 - u) * log_lo + u * log_hi));
  }
  out.front() = lo;
  out.back() = hi;
  return out;
}

inline void normalize_counts(std::vector<int>& counts, const int n_groups) {
  int total = 0;
  for (const int count : counts) {
    total += count;
  }
  while (total < n_groups) {
    auto it = std::max_element(counts.begin(), counts.end());
    if (it == counts.end()) {
      break;
    }
    ++(*it);
    ++total;
  }
  while (total > n_groups) {
    auto it = std::max_element(counts.begin(), counts.end());
    if (it == counts.end() || *it <= 1) {
      break;
    }
    --(*it);
    --total;
  }
}

}  // namespace radiation_group_detail

inline std::vector<double> repack_radiation_group_bounds_for_hard_xray(
    const int n_groups,
    const std::vector<double>& compute_T_range_eV) {
  if (n_groups <= 0 || compute_T_range_eV.size() != 2U) {
    return {};
  }
  const double E_min = compute_T_range_eV[0];
  const double E_max = compute_T_range_eV[1];
  if (!(E_min > 0.0 && E_max > E_min)) {
    return {};
  }
  if (n_groups == 1) {
    return {E_min, E_max};
  }

  const std::vector<double> base_edges = {10.0, 200.0, 1000.0, 2000.0, 5000.0};
  const std::vector<int> desired = {15, 20, 5, 5, 20, 15};

  std::vector<double> edges;
  edges.push_back(E_min);
  for (const double edge : base_edges) {
    if (edge > E_min && edge < E_max) {
      edges.push_back(edge);
    }
  }
  edges.push_back(E_max);

  std::vector<int> counts;
  counts.reserve(edges.size() - 1U);
  for (std::size_t i = 0; i + 1U < edges.size(); ++i) {
    const double mid = std::sqrt(edges[i] * edges[i + 1U]);
    int count = desired.back();
    if (mid < 10.0) {
      count = desired[0];
    } else if (mid < 200.0) {
      count = desired[1];
    } else if (mid < 1000.0) {
      count = desired[2];
    } else if (mid < 2000.0) {
      count = desired[3];
    } else if (mid < 5000.0) {
      count = desired[4];
    }
    counts.push_back(std::max(1, count));
  }
  radiation_group_detail::normalize_counts(counts, n_groups);

  std::vector<double> bounds;
  bounds.reserve(static_cast<std::size_t>(n_groups + 1));
  bounds.push_back(E_min);
  for (std::size_t i = 0; i + 1U < edges.size(); ++i) {
    const auto segment =
        radiation_group_detail::log_spaced_segment(edges[i], edges[i + 1U], counts[i]);
    for (std::size_t j = 1; j < segment.size(); ++j) {
      bounds.push_back(segment[j]);
    }
  }
  if (static_cast<int>(bounds.size()) != n_groups + 1) {
    return {};
  }
  bounds.front() = E_min;
  bounds.back() = E_max;
  return bounds;
}

inline int count_radiation_groups_inside_energy_band(const std::vector<double>& bounds_eV,
                                                     const double lo_eV,
                                                     const double hi_eV) {
  if (!(hi_eV > lo_eV) || bounds_eV.size() < 2U) {
    return 0;
  }
  int count = 0;
  constexpr double eps = 1.0e-10;
  for (std::size_t g = 0; g + 1U < bounds_eV.size(); ++g) {
    const double lo = bounds_eV[g];
    const double hi = bounds_eV[g + 1U];
    if (lo >= lo_eV * (1.0 - eps) && hi <= hi_eV * (1.0 + eps)) {
      ++count;
    }
  }
  return count;
}

}  // namespace tenryu::core
