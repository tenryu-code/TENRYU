#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <utility>
#include <vector>

#include "core/error.hpp"

namespace tenryu::mesh::assembly {

struct CountVariable {
  int id = -1;
  double ideal = 0.0;
  double weight = 1.0;
  int lo = 1;
  int hi = 1 << 20;
  bool parity_even = false;
  int fixed = -1;
};

struct CountEquality {
  std::vector<int> ids;
};

struct CountSpanSum {
  int whole = -1;
  std::vector<int> parts;
};

struct CountProblem {
  std::vector<CountVariable> variables;
  std::vector<CountEquality> equalities;
  std::vector<CountSpanSum> span_sums;
  double lambda = 0.0;
};

struct CountSolution {
  bool feasible = false;
  std::vector<int> counts;
  double objective = 0.0;
};

namespace detail {

class CountUnionFind {
 public:
  explicit CountUnionFind(const int size)
      : parent_(static_cast<std::size_t>(size), 0) {
    for (int id = 0; id < size; ++id) {
      parent_[static_cast<std::size_t>(id)] = id;
    }
  }

  int find(const int id) {
    int root = id;
    while (parent_[static_cast<std::size_t>(root)] != root) {
      root = parent_[static_cast<std::size_t>(root)];
    }
    int current = id;
    while (parent_[static_cast<std::size_t>(current)] != current) {
      const int next = parent_[static_cast<std::size_t>(current)];
      parent_[static_cast<std::size_t>(current)] = root;
      current = next;
    }
    return root;
  }

  void unite(const int lhs, const int rhs) {
    const int lhs_root = find(lhs);
    const int rhs_root = find(rhs);
    if (lhs_root == rhs_root) {
      return;
    }
    const int root = std::min(lhs_root, rhs_root);
    const int child = std::max(lhs_root, rhs_root);
    parent_[static_cast<std::size_t>(child)] = root;
  }

 private:
  std::vector<int> parent_;
};

struct CountClassState {
  bool active = false;
  bool derived = false;
  int root = -1;
  int lo = 1;
  int hi = std::numeric_limits<int>::max();
  bool parity_even = false;
  int fixed = -1;
  int multiplicity = 0;
  double weight = 0.0;
  double weighted_log_ideal = 0.0;
  double ideal = 0.0;
};

struct CanonicalCountSpanSum {
  int whole = -1;
  std::vector<int> parts;
};

inline double count_class_objective(const CountClassState& count_class,
                                    const int count,
                                    const double lambda) {
  const double log_ratio =
      std::log(static_cast<double>(count)) - std::log(count_class.ideal);
  return count_class.weight * log_ratio * log_ratio +
         lambda * static_cast<double>(count_class.multiplicity) *
             static_cast<double>(count);
}

}  // namespace detail

inline CountSolution solve_counts(const CountProblem& problem) {
  TENRYU_ASSERT(std::isfinite(problem.lambda) && problem.lambda >= 0.0,
                "count objective lambda must be non-negative and finite");
  TENRYU_ASSERT(
      problem.variables.size() <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "count variable set is too large");

  const int variable_count = static_cast<int>(problem.variables.size());
  std::vector<CountVariable> variables = problem.variables;
  for (const CountVariable& variable : variables) {
    TENRYU_ASSERT(variable.id >= 0 && variable.id < variable_count,
                  "count variable ids must be in [0, n)");
  }
  std::sort(variables.begin(), variables.end(),
            [](const CountVariable& lhs, const CountVariable& rhs) {
              return lhs.id < rhs.id;
            });
  for (int id = 0; id < variable_count; ++id) {
    const CountVariable& variable =
        variables[static_cast<std::size_t>(id)];
    TENRYU_ASSERT(variable.id == id,
                  "count variable ids must be dense and unique");
    TENRYU_ASSERT(std::isfinite(variable.ideal) && variable.ideal > 0.0,
                  "count ideal must be positive and finite");
    TENRYU_ASSERT(std::isfinite(variable.weight) && variable.weight > 0.0,
                  "count weight must be positive and finite");
    TENRYU_ASSERT(variable.lo >= 1,
                  "count lower bound must be at least one");
    TENRYU_ASSERT(variable.hi >= variable.lo,
                  "count upper bound must not be below its lower bound");
    TENRYU_ASSERT(variable.fixed == -1 ||
                      (variable.fixed >= variable.lo &&
                       variable.fixed <= variable.hi),
                  "fixed count must be -1 or lie within its bounds");
    TENRYU_ASSERT(variable.fixed == -1 || !variable.parity_even ||
                      variable.fixed % 2 == 0,
                  "fixed count must satisfy its parity constraint");
  }

  detail::CountUnionFind union_find(variable_count);
  for (const CountEquality& equality : problem.equalities) {
    for (const int id : equality.ids) {
      TENRYU_ASSERT(id >= 0 && id < variable_count,
                    "count equality id must be in [0, n)");
    }
    for (std::size_t index = 1; index < equality.ids.size(); ++index) {
      union_find.unite(equality.ids.front(), equality.ids[index]);
    }
  }

  std::vector<detail::CountClassState> classes(
      static_cast<std::size_t>(variable_count));
  bool class_constraints_feasible = true;
  for (const CountVariable& variable : variables) {
    const int root = union_find.find(variable.id);
    detail::CountClassState& count_class =
        classes[static_cast<std::size_t>(root)];
    count_class.active = true;
    count_class.root = root;
    count_class.lo = std::max(count_class.lo, variable.lo);
    count_class.hi = std::min(count_class.hi, variable.hi);
    count_class.parity_even =
        count_class.parity_even || variable.parity_even;
    if (variable.fixed >= 0) {
      if (count_class.fixed >= 0 && count_class.fixed != variable.fixed) {
        class_constraints_feasible = false;
      } else {
        count_class.fixed = variable.fixed;
      }
    }
    ++count_class.multiplicity;
    count_class.weight += variable.weight;
    count_class.weighted_log_ideal +=
        variable.weight * std::log(variable.ideal);
  }

  std::vector<int> root_order;
  root_order.reserve(classes.size());
  for (int root = 0; root < variable_count; ++root) {
    detail::CountClassState& count_class =
        classes[static_cast<std::size_t>(root)];
    if (!count_class.active) {
      continue;
    }
    root_order.push_back(root);
    TENRYU_ASSERT(std::isfinite(count_class.weight) &&
                      count_class.weight > 0.0 &&
                      std::isfinite(count_class.weighted_log_ideal),
                  "merged count class objective must be finite");
    count_class.ideal = std::exp(count_class.weighted_log_ideal /
                                 count_class.weight);
    TENRYU_ASSERT(std::isfinite(count_class.ideal) &&
                      count_class.ideal > 0.0,
                  "merged count class ideal must be positive and finite");
    if (count_class.lo > count_class.hi ||
        (count_class.fixed >= 0 &&
         (count_class.fixed < count_class.lo ||
          count_class.fixed > count_class.hi ||
          (count_class.parity_even && count_class.fixed % 2 != 0)))) {
      class_constraints_feasible = false;
    }
  }
  if (!class_constraints_feasible) {
    return {};
  }

  std::vector<detail::CanonicalCountSpanSum> span_sums;
  span_sums.reserve(problem.span_sums.size());
  std::vector<int> class_span(static_cast<std::size_t>(variable_count), -1);
  std::vector<int> class_role(static_cast<std::size_t>(variable_count), 0);
  for (const CountSpanSum& span_sum : problem.span_sums) {
    TENRYU_ASSERT(span_sum.whole >= 0 &&
                      span_sum.whole < variable_count,
                  "count span whole id must be in [0, n)");
    TENRYU_ASSERT(!span_sum.parts.empty(),
                  "count span sum must contain at least one part");
    for (const int id : span_sum.parts) {
      TENRYU_ASSERT(id >= 0 && id < variable_count,
                    "count span part id must be in [0, n)");
    }

    detail::CanonicalCountSpanSum canonical;
    canonical.whole = union_find.find(span_sum.whole);
    canonical.parts.reserve(span_sum.parts.size());
    for (const int id : span_sum.parts) {
      const int part = union_find.find(id);
      TENRYU_ASSERT(canonical.whole != part,
                    "count span whole class must differ from every part");
      canonical.parts.push_back(part);
    }
    std::sort(canonical.parts.begin(), canonical.parts.end());

    const int span_index = static_cast<int>(span_sums.size());
    TENRYU_ASSERT(
        class_span[static_cast<std::size_t>(canonical.whole)] == -1,
        "count class may appear in at most one span sum and not as both "
        "whole and part");
    class_span[static_cast<std::size_t>(canonical.whole)] = span_index;
    class_role[static_cast<std::size_t>(canonical.whole)] = 1;
    for (const int part : canonical.parts) {
      const std::size_t part_index = static_cast<std::size_t>(part);
      if (class_span[part_index] == -1) {
        class_span[part_index] = span_index;
        class_role[part_index] = 2;
      } else {
        // Repeated part classes in one span are retained as multiplicity.
        TENRYU_ASSERT(
            class_span[part_index] == span_index &&
                class_role[part_index] == 2,
            "count class may appear in at most one span sum and not as both "
            "whole and part");
      }
    }
    classes[static_cast<std::size_t>(canonical.whole)].derived = true;
    span_sums.push_back(std::move(canonical));
  }
  std::sort(span_sums.begin(), span_sums.end(),
            [](const detail::CanonicalCountSpanSum& lhs,
               const detail::CanonicalCountSpanSum& rhs) {
              return lhs.whole < rhs.whole;
            });

  std::vector<int> free_roots;
  std::vector<std::vector<int>> domains;
  for (const int root : root_order) {
    const detail::CountClassState& count_class =
        classes[static_cast<std::size_t>(root)];
    if (count_class.derived) {
      continue;
    }
    free_roots.push_back(root);
    std::vector<int> domain;
    if (count_class.fixed >= 0) {
      domain.push_back(count_class.fixed);
    } else {
      int domain_lo = count_class.lo;
      int domain_hi = count_class.hi;
      const double ideal_lo = std::floor(count_class.ideal / 2.0);
      const double ideal_hi = std::ceil(count_class.ideal * 2.0);
      bool window_empty = ideal_lo > static_cast<double>(count_class.hi) ||
                          ideal_hi < static_cast<double>(count_class.lo);
      if (!window_empty) {
        if (ideal_lo > static_cast<double>(domain_lo)) {
          domain_lo = static_cast<int>(ideal_lo);
        }
        if (ideal_hi < static_cast<double>(domain_hi)) {
          domain_hi = static_cast<int>(ideal_hi);
        }
        window_empty = domain_lo > domain_hi;
      }
      if (window_empty) {
        domain_lo = count_class.lo;
        domain_hi = static_cast<int>(std::min(
            static_cast<long long>(count_class.hi),
            static_cast<long long>(count_class.lo) + 63LL));
      }
      for (long long candidate = domain_lo; candidate <= domain_hi;
           ++candidate) {
        if (!count_class.parity_even || candidate % 2LL == 0LL) {
          domain.push_back(static_cast<int>(candidate));
        }
      }
    }
    TENRYU_ASSERT(!domain.empty() && domain.size() <= 64U,
                  "count problem is out of the v1 solver envelope");
    domains.push_back(std::move(domain));
  }

  std::vector<int> class_counts(static_cast<std::size_t>(variable_count),
                                -1);
  std::vector<int> best_class_counts;
  double best_objective = std::numeric_limits<double>::infinity();
  bool found = false;

  // Merged equality-class log terms omit the count-independent additive
  // constant, which is acceptable for the v1 objective.
  const auto lexicographically_smaller = [&root_order](
                                             const std::vector<int>& lhs,
                                             const std::vector<int>& rhs) {
    for (const int root : root_order) {
      const int lhs_count = lhs[static_cast<std::size_t>(root)];
      const int rhs_count = rhs[static_cast<std::size_t>(root)];
      if (lhs_count != rhs_count) {
        return lhs_count < rhs_count;
      }
    }
    return false;
  };

  std::function<void(std::size_t, double)> search;
  search = [&](const std::size_t free_index,
               const double partial_objective) {
    if (free_index < free_roots.size()) {
      if (found && partial_objective >= best_objective) {
        return;
      }
      const int root = free_roots[free_index];
      const detail::CountClassState& count_class =
          classes[static_cast<std::size_t>(root)];
      for (const int candidate : domains[free_index]) {
        class_counts[static_cast<std::size_t>(root)] = candidate;
        const double candidate_objective =
            partial_objective + detail::count_class_objective(
                                    count_class, candidate, problem.lambda);
        search(free_index + 1U, candidate_objective);
      }
      return;
    }

    double objective = partial_objective;
    for (const detail::CanonicalCountSpanSum& span_sum : span_sums) {
      long long derived_count = 0;
      for (const int part : span_sum.parts) {
        derived_count += class_counts[static_cast<std::size_t>(part)];
      }
      const detail::CountClassState& whole =
          classes[static_cast<std::size_t>(span_sum.whole)];
      if (derived_count < whole.lo || derived_count > whole.hi ||
          (whole.parity_even && derived_count % 2LL != 0LL) ||
          (whole.fixed >= 0 && derived_count != whole.fixed)) {
        return;
      }
      const int count = static_cast<int>(derived_count);
      class_counts[static_cast<std::size_t>(span_sum.whole)] = count;
      objective +=
          detail::count_class_objective(whole, count, problem.lambda);
    }

    if (!found || objective < best_objective ||
        (objective == best_objective &&
         lexicographically_smaller(class_counts, best_class_counts))) {
      found = true;
      best_objective = objective;
      best_class_counts = class_counts;
    }
  };
  search(0U, 0.0);

  if (!found) {
    return {};
  }

  CountSolution solution;
  solution.feasible = true;
  solution.counts.resize(static_cast<std::size_t>(variable_count));
  solution.objective = best_objective;
  for (int id = 0; id < variable_count; ++id) {
    const int root = union_find.find(id);
    solution.counts[static_cast<std::size_t>(id)] =
        best_class_counts[static_cast<std::size_t>(root)];
  }
  return solution;
}

}  // namespace tenryu::mesh::assembly
