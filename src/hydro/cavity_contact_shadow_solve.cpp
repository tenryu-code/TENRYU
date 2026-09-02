#include "hydro/cavity_contact_shadow_solve.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/error.hpp"

namespace tenryu::hydro {
namespace {

struct NodeWeight {
  int node = -1;
  double r = 0.0;
  double z = 0.0;
};

std::array<NodeWeight, 3> row_weights(const ShadowContactRow& row) {
  return {{{row.node_s, row.normal_r, row.normal_z},
           {row.node_a,
            -(1.0 - row.xi) * row.normal_r,
            -(1.0 - row.xi) * row.normal_z},
           {row.node_b,
            -row.xi * row.normal_r,
            -row.xi * row.normal_z}}};
}

void fill_active_flags(const std::vector<std::size_t>& active_rows,
                       ShadowContactSolveResult& result) {
  std::fill(result.active.begin(), result.active.end(), 0U);
  for (const std::size_t row : active_rows) {
    result.active[row] = 1U;
  }
}

}  // namespace

ShadowContactSolveResult solve_shadow_contact_lcp(
    std::vector<ShadowContactRow>& rows,
    const std::vector<double>& node_mass,
    const std::vector<double>& v_r,
    const std::vector<double>& v_z,
    const double dt,
    const double gap_floor,
    const GapTermMode gap_term_mode) {
  TENRYU_ASSERT(
      std::is_sorted(rows.begin(), rows.end(),
                     [](const ShadowContactRow& lhs,
                        const ShadowContactRow& rhs) {
                       return lhs.constraint_id < rhs.constraint_id;
                     }),
      "shadow contact rows are not in canonical order");
  TENRYU_ASSERT(node_mass.size() == v_r.size() &&
                    node_mass.size() == v_z.size() && dt > 0.0,
                "shadow contact solver input size mismatch");

  ShadowContactSolveResult result;
  const std::size_t row_count = rows.size();
  result.lambda.assign(row_count, 0.0);
  result.active.assign(row_count, 0U);
  if (row_count == 0U) {
    return result;
  }

  std::vector<double> q(row_count, 0.0);
  std::vector<std::array<NodeWeight, 3>> weights;
  weights.reserve(row_count);
  for (std::size_t alpha = 0; alpha < row_count; ++alpha) {
    const ShadowContactRow& row = rows[alpha];
    TENRYU_ASSERT(row.node_s >= 0 && row.node_a >= 0 && row.node_b >= 0 &&
                      static_cast<std::size_t>(row.node_s) < v_r.size() &&
                      static_cast<std::size_t>(row.node_a) < v_r.size() &&
                      static_cast<std::size_t>(row.node_b) < v_r.size(),
                  "shadow contact row node index mismatch");
    const std::size_t node_s = static_cast<std::size_t>(row.node_s);
    const std::size_t node_a = static_cast<std::size_t>(row.node_a);
    const std::size_t node_b = static_cast<std::size_t>(row.node_b);
    const double gap_term =
        gap_term_mode == GapTermMode::kFull
            ? row.gap - gap_floor
            : std::max(row.gap - gap_floor, 0.0);
    q[alpha] = row.normal_r * v_r[node_s] +
               row.normal_z * v_z[node_s] -
               (1.0 - row.xi) *
                   (row.normal_r * v_r[node_a] +
                    row.normal_z * v_z[node_a]) -
               row.xi * (row.normal_r * v_r[node_b] +
                         row.normal_z * v_z[node_b]) +
               gap_term / dt;
    weights.push_back(row_weights(row));
  }

  std::vector<double> W(row_count * row_count, 0.0);
  for (std::size_t alpha = 0; alpha < row_count; ++alpha) {
    for (std::size_t beta = 0; beta <= alpha; ++beta) {
      double value = 0.0;
      for (const NodeWeight& alpha_weight : weights[alpha]) {
        for (const NodeWeight& beta_weight : weights[beta]) {
          if (alpha_weight.node != beta_weight.node) {
            continue;
          }
          const double mass =
              node_mass[static_cast<std::size_t>(alpha_weight.node)];
          if (mass <= 0.0) {
            if (alpha == beta) {
              result.singular = true;
            }
            continue;
          }
          value += (alpha_weight.r * beta_weight.r +
                    alpha_weight.z * beta_weight.z) /
                   mass;
        }
      }
      W[alpha * row_count + beta] = value;
      W[beta * row_count + alpha] = value;
    }
  }

  std::vector<std::size_t> active_rows;
  active_rows.reserve(row_count);
  for (std::size_t alpha = 0; alpha < row_count; ++alpha) {
    if (q[alpha] < 0.0 || rows[alpha].legacy_engaged) {
      active_rows.push_back(alpha);
    }
  }

  constexpr int kMaxIterations = 64;
  constexpr double kRankTolerance = 1.0e-30;
  constexpr double kComplementarityTolerance = 1.0e-14;
  bool converged = false;
  for (int iteration = 0; iteration < kMaxIterations; ++iteration) {
    result.iterations = iteration + 1;
    std::fill(result.lambda.begin(), result.lambda.end(), 0.0);

    const std::size_t active_count = active_rows.size();
    bool rank_guard_tripped = false;
    if (active_count > 0U) {
      double max_diagonal = 0.0;
      for (const std::size_t row : active_rows) {
        max_diagonal =
            std::max(max_diagonal, std::abs(W[row * row_count + row]));
      }

      std::vector<double> L(active_count * active_count, 0.0);
      std::vector<double> D(active_count, 0.0);
      std::size_t dropped_local_row = active_count;
      for (std::size_t k = 0; k < active_count; ++k) {
        L[k * active_count + k] = 1.0;
        double pivot =
            W[active_rows[k] * row_count + active_rows[k]];
        for (std::size_t j = 0; j < k; ++j) {
          const double l_kj = L[k * active_count + j];
          pivot -= l_kj * l_kj * D[j];
        }
        if (max_diagonal == 0.0 ||
            std::abs(pivot) < kRankTolerance * max_diagonal) {
          dropped_local_row = k;
          break;
        }
        D[k] = pivot;
        for (std::size_t i = k + 1U; i < active_count; ++i) {
          double entry =
              W[active_rows[i] * row_count + active_rows[k]];
          for (std::size_t j = 0; j < k; ++j) {
            entry -= L[i * active_count + j] *
                     L[k * active_count + j] * D[j];
          }
          L[i * active_count + k] = entry / D[k];
        }
      }

      if (dropped_local_row != active_count) {
        active_rows.erase(active_rows.begin() +
                          static_cast<std::ptrdiff_t>(dropped_local_row));
        ++result.released_rows;
        result.singular = true;
        rank_guard_tripped = true;
      } else {
        std::vector<double> y(active_count, 0.0);
        std::vector<double> z(active_count, 0.0);
        std::vector<double> active_lambda(active_count, 0.0);
        for (std::size_t i = 0; i < active_count; ++i) {
          double rhs = -q[active_rows[i]];
          for (std::size_t j = 0; j < i; ++j) {
            rhs -= L[i * active_count + j] * y[j];
          }
          y[i] = rhs;
          z[i] = y[i] / D[i];
        }
        for (std::size_t reverse = active_count; reverse > 0U; --reverse) {
          const std::size_t i = reverse - 1U;
          double value = z[i];
          for (std::size_t j = i + 1U; j < active_count; ++j) {
            value -= L[j * active_count + i] * active_lambda[j];
          }
          active_lambda[i] = value;
          result.lambda[active_rows[i]] = value;
        }
      }
    }
    if (rank_guard_tripped) {
      continue;
    }

    double lambda_scale = 1.0;
    for (const std::size_t row : active_rows) {
      lambda_scale = std::max(lambda_scale, std::abs(result.lambda[row]));
    }
    std::size_t release_local_row = active_rows.size();
    double most_negative_normalized =
        std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < active_rows.size(); ++i) {
      const double normalized = result.lambda[active_rows[i]] / lambda_scale;
      if (normalized < -kComplementarityTolerance &&
          normalized < most_negative_normalized) {
        most_negative_normalized = normalized;
        release_local_row = i;
      }
    }
    if (release_local_row != active_rows.size()) {
      active_rows.erase(active_rows.begin() +
                        static_cast<std::ptrdiff_t>(release_local_row));
      ++result.released_rows;
      continue;
    }

    std::vector<std::uint8_t> is_active(row_count, 0U);
    for (const std::size_t row : active_rows) {
      is_active[row] = 1U;
    }
    std::size_t add_row = row_count;
    double most_violated_normalized =
        std::numeric_limits<double>::infinity();
    for (std::size_t alpha = 0; alpha < row_count; ++alpha) {
      if (is_active[alpha] != 0U) {
        continue;
      }
      double w = q[alpha];
      for (const std::size_t beta : active_rows) {
        w += W[alpha * row_count + beta] * result.lambda[beta];
      }
      const double normalized = w / std::max(std::abs(q[alpha]), 1.0);
      if (normalized < -kComplementarityTolerance &&
          normalized < most_violated_normalized) {
        most_violated_normalized = normalized;
        add_row = alpha;
      }
    }
    if (add_row != row_count) {
      const auto insertion =
          std::lower_bound(active_rows.begin(), active_rows.end(), add_row);
      active_rows.insert(insertion, add_row);
      continue;
    }

    converged = true;
    break;
  }

  if (!converged) {
    result.singular = true;
  }
  fill_active_flags(active_rows, result);
  for (std::size_t alpha = 0; alpha < row_count; ++alpha) {
    if (result.active[alpha] == 0U) {
      result.lambda[alpha] = 0.0;
    } else {
      result.lambda[alpha] = std::max(result.lambda[alpha], 0.0);
    }
  }
  return result;
}

}  // namespace tenryu::hydro
