#include "mesh/carrier_coverage.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mesh/tessellation/exact_types.hpp"

namespace tenryu::mesh {

namespace {

ExactRational rational_zero() {
  return ExactRational{{0.0}, {1.0}};
}

ExactRational rational_one() {
  return ExactRational{{1.0}, {1.0}};
}

bool rational_is_malformed(const ExactRational& rational) {
  return rational.num.empty() || rational.den.empty() ||
         tess::expansion_sign(rational.den) <= 0;
}

double approx(const ExactRational& rational) {
  return tess::approx_of_expansion(rational.num).value /
         tess::approx_of_expansion(rational.den).value;
}

}  // namespace

ExactRational make_exact_rational(tess::Expansion num, tess::Expansion den) {
  if (num.empty() || den.empty()) {
    throw std::invalid_argument("carrier_coverage: empty expansion");
  }
  const int sign = tess::expansion_sign(den);
  if (sign == 0) {
    throw std::invalid_argument("carrier_coverage: zero denominator");
  }
  if (sign < 0) {
    num = tess::expansion_negate(num);
    den = tess::expansion_negate(den);
  }
  return ExactRational{std::move(num), std::move(den)};
}

ExactRational exact_rational_ratio(const double num, const double den) {
  if (!std::isfinite(num) || !std::isfinite(den)) {
    throw std::invalid_argument("carrier_coverage: non-finite ratio");
  }
  return make_exact_rational(tess::Expansion{num}, tess::Expansion{den});
}

int compare_exact_rational(const ExactRational& a, const ExactRational& b) {
  return tess::compare_event_parameters_exact(a.num, a.den, b.num, b.den);
}

CoverageResult check_carrier_coverage(
    const int n_edges,
    const std::vector<CarrierEdgeInterval>& intervals) {
  if (n_edges <= 0) {
    return CoverageResult{
        false, CoverageViolation::kInvalidInput, -1, -1, 0.0};
  }
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    if (rational_is_malformed(intervals[i].lo) ||
        rational_is_malformed(intervals[i].hi)) {
      return CoverageResult{false,
                            CoverageViolation::kInvalidRational,
                            intervals[i].edge,
                            static_cast<int>(i),
                            0.0};
    }
  }
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    if (intervals[i].edge < 0 || intervals[i].edge >= n_edges) {
      return CoverageResult{false,
                            CoverageViolation::kEdgeOutOfRange,
                            intervals[i].edge,
                            static_cast<int>(i),
                            0.0};
    }
  }

  std::vector<std::vector<int>> per_edge(
      static_cast<std::size_t>(n_edges));
  for (std::size_t i = 0; i < intervals.size(); ++i) {
    per_edge[static_cast<std::size_t>(intervals[i].edge)].push_back(
        static_cast<int>(i));
  }

  const ExactRational zero = rational_zero();
  const ExactRational one = rational_one();
  for (int edge = 0; edge < n_edges; ++edge) {
    std::vector<int>& bucket = per_edge[static_cast<std::size_t>(edge)];
    if (bucket.empty()) {
      return CoverageResult{
          false, CoverageViolation::kEmptyEdge, edge, -1, 0.0};
    }
    std::stable_sort(bucket.begin(), bucket.end(), [&](const int a, const int b) {
      const int comparison =
          compare_exact_rational(intervals[a].lo, intervals[b].lo);
      if (comparison != 0) {
        return comparison < 0;
      }
      return compare_exact_rational(intervals[a].hi, intervals[b].hi) < 0;
    });

    for (std::size_t k = 0; k < bucket.size(); ++k) {
      const CarrierEdgeInterval& current = intervals[bucket[k]];
      if (compare_exact_rational(current.lo, current.hi) >= 0) {
        return CoverageResult{false,
                              CoverageViolation::kDegenerate,
                              edge,
                              static_cast<int>(k),
                              approx(current.lo)};
      }
      if (k == 0 && compare_exact_rational(current.lo, zero) != 0) {
        return CoverageResult{false,
                              CoverageViolation::kFirstNotZero,
                              edge,
                              0,
                              approx(current.lo)};
      }
      if (k > 0) {
        const CarrierEdgeInterval& previous = intervals[bucket[k - 1]];
        const int comparison =
            compare_exact_rational(previous.hi, current.lo);
        if (comparison < 0) {
          return CoverageResult{false,
                                CoverageViolation::kGap,
                                edge,
                                static_cast<int>(k),
                                approx(current.lo)};
        }
        if (comparison > 0) {
          return CoverageResult{false,
                                CoverageViolation::kOverlap,
                                edge,
                                static_cast<int>(k),
                                approx(current.lo)};
        }
      }
    }

    const CarrierEdgeInterval& last = intervals[bucket.back()];
    if (compare_exact_rational(last.hi, one) != 0) {
      return CoverageResult{false,
                            CoverageViolation::kLastNotOne,
                            edge,
                            static_cast<int>(bucket.size() - 1),
                            approx(last.hi)};
    }
  }

  return CoverageResult{true, CoverageViolation::kNone, -1, -1, 0.0};
}

}  // namespace tenryu::mesh
