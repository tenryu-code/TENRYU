#include "hydro/band_ale.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <math_constants.h>
#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "core/state.hpp"
#include "hydro/euler_window_blend.hpp"
#include "hydro/refinement_tracker.hpp"
#include "hydro/rezone_objective.cuh"
#include "mesh/mesh.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::band_ale {

std::vector<double> closure_catchment_xi(
    const std::vector<double>& Hstar) {
  std::vector<double> xi;
  if (Hstar.size() < 2U) {
    return xi;
  }
  double total = 0.0;
  for (const double width : Hstar) {
    total += width;
  }
  xi.reserve(Hstar.size() - 1U);
  double cumulative = 0.0;
  for (std::size_t k = 0; k + 1U < Hstar.size(); ++k) {
    cumulative += Hstar[k];
    xi.push_back(cumulative / total);
  }
  return xi;
}

bool closure_catchment_redistribute(
    const std::vector<double>& birth_spacings,
    const double L,
    const double floor_cm,
    const double ratio_max,
    std::vector<double>* out_spacings) {
  if (out_spacings == nullptr) {
    return false;
  }
  out_spacings->clear();
  const std::size_t n = birth_spacings.size();
  if (n == 0U || !std::isfinite(L) || !(L > 0.0) ||
      !std::isfinite(floor_cm) || !(floor_cm > 0.0) ||
      !std::isfinite(ratio_max) || !(ratio_max > 1.0) ||
      L < static_cast<double>(n) * floor_cm) {
    return false;
  }
  double birth_sum = 0.0;
  for (const double spacing : birth_spacings) {
    if (!std::isfinite(spacing) || !(spacing > 0.0)) {
      return false;
    }
    birth_sum += spacing;
  }
  if (!std::isfinite(birth_sum) || !(birth_sum > 0.0)) {
    return false;
  }

  std::vector<double> spacings(n, 0.0);
  const double scale = L / birth_sum;
  for (std::size_t i = 0; i < n; ++i) {
    spacings[i] = birth_spacings[i] * scale;
  }

  for (int iteration = 0; iteration < 8; ++iteration) {
    std::vector<std::uint8_t> clamped(n, 0U);
    for (std::size_t i = 0; i < n; ++i) {
      if (spacings[i] < floor_cm) {
        spacings[i] = floor_cm;
        clamped[i] = 1U;
      }
    }
    for (std::size_t i = 0; i + 1U < n; ++i) {
      const double lower = spacings[i] / ratio_max;
      const double upper = spacings[i] * ratio_max;
      spacings[i + 1U] =
          std::min(std::max(spacings[i + 1U], lower), upper);
    }
    for (std::size_t i = n - 1U; i > 0U; --i) {
      const double lower = spacings[i] / ratio_max;
      const double upper = spacings[i] * ratio_max;
      spacings[i - 1U] =
          std::min(std::max(spacings[i - 1U], lower), upper);
    }

    double spacing_sum = 0.0;
    double adjustable_sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      spacing_sum += spacings[i];
      if (clamped[i] == 0U) {
        adjustable_sum += spacings[i];
      }
    }
    const double residual = L - spacing_sum;
    if (residual != 0.0) {
      if (!(adjustable_sum > 0.0) || !std::isfinite(adjustable_sum)) {
        return false;
      }
      for (std::size_t i = 0; i < n; ++i) {
        if (clamped[i] == 0U) {
          spacings[i] += residual * spacings[i] / adjustable_sum;
        }
      }
    }
  }

  double spacing_sum = 0.0;
  for (const double spacing : spacings) {
    if (!std::isfinite(spacing) || spacing < floor_cm) {
      return false;
    }
    spacing_sum += spacing;
  }
  if (std::abs(spacing_sum - L) > 1.0e-12 * L) {
    return false;
  }
  const double ratio_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * ratio_max;
  for (std::size_t i = 0; i + 1U < n; ++i) {
    const double ratio = spacings[i + 1U] / spacings[i];
    if (ratio < 1.0 / ratio_max - ratio_tolerance ||
        ratio > ratio_max + ratio_tolerance) {
      return false;
    }
  }
  *out_spacings = std::move(spacings);
  return true;
}

double band_significance_channel(const double value,
                                 const double reference,
                                 const double y0,
                                 const double y1) {
  if (value <= 0.0 || reference <= 0.0) {
    return 0.0;
  }
  return band_smoothstep5(
      (std::log(value / reference) - std::log(y0)) /
      (std::log(y1) - std::log(y0)));
}

double band_significance_weight(const double w_rho,
                                const double w_m,
                                const double w_z,
                                const double w_pv) {
  return 1.0 - (1.0 - w_rho) * (1.0 - w_m) *
                   (1.0 - w_z) * (1.0 - w_pv);
}

PoleThetaCellMetrics pole_theta_cell_metrics(const double r[4],
                                             const double z[4],
                                             const double h_theta_star) {
  constexpr int radial_neighbor[4] = {1, 0, 3, 2};
  constexpr int angular_neighbor[4] = {3, 2, 1, 0};
  PoleThetaCellMetrics metrics;
  metrics.h_theta_min = std::numeric_limits<double>::infinity();
  metrics.min_altitude_abs = std::numeric_limits<double>::infinity();

  for (int corner = 0; corner < 4; ++corner) {
    const int radial = radial_neighbor[corner];
    const int angular = angular_neighbor[corner];
    const double es_r = r[radial] - r[corner];
    const double es_z = z[radial] - z[corner];
    const double et_r = r[angular] - r[corner];
    const double et_z = z[angular] - z[corner];
    const double det = es_r * et_z - es_z * et_r;
    const double len_s = std::hypot(es_r, es_z);
    const double len_s_safe = std::max(len_s, 1.0e-300);
    const double h_theta_corner = std::abs(det) / len_s_safe;
    metrics.h_theta_min = std::min(metrics.h_theta_min, h_theta_corner);
    metrics.min_altitude_abs =
        std::min(metrics.min_altitude_abs, h_theta_corner);

    const double h_theta_star_safe = std::max(h_theta_star, 1.0e-300);
    const double u_r = es_r / len_s_safe;
    const double u_z = es_z / len_s_safe;
    const double v_r = et_r / h_theta_star_safe;
    const double v_z = et_z / h_theta_star_safe;
    const double g11 = u_r * u_r + u_z * u_z;
    const double g12 = u_r * v_r + u_z * v_z;
    const double g22 = v_r * v_r + v_z * v_z;
    const double discriminant =
        std::sqrt((g11 - g22) * (g11 - g22) + 4.0 * g12 * g12);
    const double lambda_plus = 0.5 * (g11 + g22 + discriminant);
    const double lambda_minus = 0.5 * (g11 + g22 - discriminant);
    const double kappa_corner = std::sqrt(
        std::max(lambda_plus, 0.0) / std::max(lambda_minus, 1.0e-300));
    metrics.kappa_max = std::max(metrics.kappa_max, kappa_corner);
  }

  return metrics;
}

double pole_theta_cell_excess(const double H,
                              const double kappa,
                              const double h_hard,
                              const double kappa_hard) {
  double qH = 0.0;
  if (H < h_hard) {
    qH = std::max(
        0.0, std::log(h_hard / std::max(H, 1.0e-6)));
  }
  double qk = 0.0;
  if (kappa > kappa_hard) {
    qk = std::max(
        0.0,
        std::log(std::max(kappa, kappa_hard) / kappa_hard));
  }
  return std::sqrt(qH * qH + qk * qk);
}

struct PoleThetaTrialCellGeometry {
  std::array<double, 4> pre_r = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> pre_z = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> post_r = {{0.0, 0.0, 0.0, 0.0}};
  std::array<double, 4> post_z = {{0.0, 0.0, 0.0, 0.0}};
  double weight = 0.0;
  double h_theta_star = 0.0;
  double frozen_q_pre = std::numeric_limits<double>::quiet_NaN();
  double frozen_min_altitude_pre =
      std::numeric_limits<double>::quiet_NaN();
};

struct PoleThetaTrialAggregate {
  double q8_pre = 0.0;
  double q8_post = 0.0;
  double min_altitude_pre = std::numeric_limits<double>::infinity();
  double min_altitude_post = std::numeric_limits<double>::infinity();
};

PoleThetaTrialAggregate pole_theta_trial_aggregate(
    const std::vector<PoleThetaTrialCellGeometry>& cells,
    const double h_hard,
    const double kappa_hard) {
  PoleThetaTrialAggregate aggregate;
  double q8_pre_sum = 0.0;
  double q8_post_sum = 0.0;
  for (const auto& cell : cells) {
    PoleThetaCellMetrics pre;
    const bool need_pre_metrics =
        !std::isfinite(cell.frozen_q_pre) ||
        (cell.weight > 0.5 &&
         !std::isfinite(cell.frozen_min_altitude_pre));
    if (need_pre_metrics) {
      pre = pole_theta_cell_metrics(
          cell.pre_r.data(), cell.pre_z.data(), cell.h_theta_star);
    }
    const PoleThetaCellMetrics post = pole_theta_cell_metrics(
        cell.post_r.data(), cell.post_z.data(), cell.h_theta_star);
    const double q_pre = std::isfinite(cell.frozen_q_pre)
                             ? cell.frozen_q_pre
                             : cell.weight * pole_theta_cell_excess(
                                   pre.h_theta_min / cell.h_theta_star,
                                   pre.kappa_max,
                                   h_hard,
                                   kappa_hard);
    const double q_post = cell.weight * pole_theta_cell_excess(
        post.h_theta_min / cell.h_theta_star,
        post.kappa_max,
        h_hard,
        kappa_hard);
    q8_pre_sum += std::pow(q_pre, 8.0);
    q8_post_sum += std::pow(q_post, 8.0);
    if (cell.weight > 0.5) {
      aggregate.min_altitude_pre = std::min(
          aggregate.min_altitude_pre,
          std::isfinite(cell.frozen_min_altitude_pre)
              ? cell.frozen_min_altitude_pre
              : pre.min_altitude_abs);
      aggregate.min_altitude_post = std::min(
          aggregate.min_altitude_post, post.min_altitude_abs);
    }
  }
  if (!cells.empty()) {
    const double inverse_count =
        1.0 / static_cast<double>(cells.size());
    aggregate.q8_pre = std::pow(q8_pre_sum * inverse_count, 1.0 / 8.0);
    aggregate.q8_post = std::pow(q8_post_sum * inverse_count, 1.0 / 8.0);
  }
  return aggregate;
}

std::pair<double, double> pole_theta_row_trial_excess(
    const std::vector<double>& theta_row,
    const std::vector<double>& applied,
    const std::vector<double>& weights,
    const double h_hard,
    const double kappa_hard,
    const double h_theta_star) {
  if (theta_row.size() < 2U || applied.size() != theta_row.size() ||
      weights.size() + 1U != theta_row.size() ||
      !(h_theta_star > 0.0)) {
    return {0.0, 0.0};
  }
  std::vector<PoleThetaTrialCellGeometry> cells;
  cells.reserve(weights.size());
  const double inner_radius = 1.0;
  const double outer_radius = inner_radius + h_theta_star;
  for (std::size_t cell_index = 0; cell_index < weights.size();
       ++cell_index) {
    PoleThetaTrialCellGeometry cell;
    cell.weight = weights[cell_index];
    cell.h_theta_star = h_theta_star;
    const std::array<double, 4> radius = {{
        inner_radius, outer_radius, outer_radius, inner_radius}};
    const std::array<double, 4> theta_pre = {{
        theta_row[cell_index], theta_row[cell_index],
        theta_row[cell_index + 1U], theta_row[cell_index + 1U]}};
    const std::array<double, 4> theta_post = {{
        theta_row[cell_index] + applied[cell_index],
        theta_row[cell_index] + applied[cell_index],
        theta_row[cell_index + 1U] + applied[cell_index + 1U],
        theta_row[cell_index + 1U] + applied[cell_index + 1U]}};
    for (int corner = 0; corner < 4; ++corner) {
      const std::size_t c = static_cast<std::size_t>(corner);
      cell.pre_r[c] = radius[c] * std::sin(theta_pre[c]);
      cell.pre_z[c] = radius[c] * std::cos(theta_pre[c]);
      cell.post_r[c] = radius[c] * std::sin(theta_post[c]);
      cell.post_z[c] = radius[c] * std::cos(theta_post[c]);
    }
    cells.push_back(cell);
  }
  const PoleThetaTrialAggregate aggregate = pole_theta_trial_aggregate(
      cells, h_hard, kappa_hard);
  return {aggregate.q8_pre, aggregate.q8_post};
}

PoleThetaSector pole_theta_grow_sector(const std::vector<double>& H,
                                       const std::vector<double>& kappa,
                                       const int n_columns,
                                       const bool from_north,
                                       const double mark_h,
                                       const double mark_k,
                                       const double stop_h,
                                       const double stop_k,
                                       const int halo_columns) {
  const int pole_column = from_north ? 0 : n_columns - 1;
  const int direction = from_north ? 1 : -1;
  const auto marked = [&](const int column) {
    return H[static_cast<std::size_t>(column)] < mark_h ||
           kappa[static_cast<std::size_t>(column)] > mark_k;
  };
  const auto healthy = [&](const int column) {
    return H[static_cast<std::size_t>(column)] > stop_h &&
           kappa[static_cast<std::size_t>(column)] < stop_k;
  };

  if (!marked(pole_column)) {
    return {-1, -1};
  }

  int last_marked = pole_column;
  int consecutive_healthy = 0;
  for (int column = pole_column + direction;
       column >= 0 && column < n_columns;
       column += direction) {
    if (marked(column)) {
      last_marked = column;
    }
    if (healthy(column)) {
      ++consecutive_healthy;
      if (consecutive_healthy == 2) {
        break;
      }
    } else {
      consecutive_healthy = 0;
    }
  }

  const int halo_extent = from_north
                              ? std::min(halo_columns,
                                         n_columns - 1 - last_marked)
                              : std::min(halo_columns, last_marked);
  const int equator_end = last_marked + direction * halo_extent;
  return {std::min(pole_column, equator_end),
          std::max(pole_column, equator_end)};
}

PoleThetaSector pole_theta_grow_sector_q(
    const std::vector<double>& q_half,
    const int half_size,
    const double q_stop,
    const int halo_columns) {
  if (half_size <= 0 || !(q_half[0] > q_stop)) {
    return {-1, -1};
  }

  int last_violating = 0;
  for (int col = 1; col < half_size; ++col) {
    if (!(q_half[static_cast<std::size_t>(col)] > q_stop)) {
      break;
    }
    last_violating = col;
  }
  return {0, std::min(last_violating + halo_columns, half_size - 1)};
}

std::vector<double> pole_theta_delta_map(
    const std::vector<double>& theta,
    const int j0,
    const int j1,
    const double alpha,
    const double deadband_frac,
    const double move_limit_frac,
    const double dtheta_star,
    const std::vector<double>& taper_w) {
  std::vector<double> dtheta(theta.size(), 0.0);
  if (theta.size() != taper_w.size() || j0 < 0 || j0 >= j1 ||
      static_cast<std::size_t>(j1) >= theta.size() ||
      !(dtheta_star > 0.0)) {
    return dtheta;
  }
  for (int j = j0 + 1; j <= j1; ++j) {
    if (!(theta[static_cast<std::size_t>(j)] >
          theta[static_cast<std::size_t>(j - 1)])) {
      return dtheta;
    }
  }

  const double theta_span = theta[static_cast<std::size_t>(j1)] -
                            theta[static_cast<std::size_t>(j0)];
  const double deadband = deadband_frac * dtheta_star;
  const double move_limit = move_limit_frac * dtheta_star;
  for (int j = j0 + 1; j < j1; ++j) {
    const double fraction =
        static_cast<double>(j - j0) / static_cast<double>(j1 - j0);
    const double theta_equal =
        theta[static_cast<std::size_t>(j0)] + fraction * theta_span;
    const double displacement =
        theta_equal - theta[static_cast<std::size_t>(j)];
    if (std::abs(displacement) < deadband) {
      continue;
    }
    dtheta[static_cast<std::size_t>(j)] = std::clamp(
        alpha * taper_w[static_cast<std::size_t>(j)] * displacement,
        -move_limit, move_limit);
  }

  const double epsilon = 0.02 * dtheta_star;
  double previous_theta = theta[static_cast<std::size_t>(j0)];
  for (int j = j0 + 1; j < j1; ++j) {
    double projected_theta = theta[static_cast<std::size_t>(j)] +
                             dtheta[static_cast<std::size_t>(j)];
    projected_theta = std::max(projected_theta, previous_theta + epsilon);
    projected_theta = std::min(
        projected_theta,
        theta[static_cast<std::size_t>(j1)] -
            epsilon * static_cast<double>(j1 - j));
    dtheta[static_cast<std::size_t>(j)] =
        projected_theta - theta[static_cast<std::size_t>(j)];
    previous_theta = projected_theta;
  }

  return dtheta;
}

double pole_theta_parity_gate(
    const std::vector<double>& d_north,
    const std::vector<double>& d_south,
    const int ntheta,
    const double deadband_frac,
    const double dtheta_star,
    std::vector<double>* out_north_applied,
    std::vector<double>* out_south_applied) {
  if (out_north_applied == nullptr || out_south_applied == nullptr) {
    return 0.0;
  }
  out_north_applied->assign(d_north.size(), 0.0);
  out_south_applied->assign(d_south.size(), 0.0);
  const std::size_t expected_size =
      ntheta >= 0 ? static_cast<std::size_t>(ntheta) + 1U : 0U;
  if (d_north.size() != expected_size ||
      d_south.size() != expected_size ||
      !(deadband_frac > 0.0) || !(dtheta_star > 0.0)) {
    return 0.0;
  }

  std::vector<double> d_sym(expected_size, 0.0);
  std::vector<double> d_break(expected_size, 0.0);
  double max_break = 0.0;
  for (int spoke = 0; spoke <= ntheta; ++spoke) {
    const std::size_t north_index = static_cast<std::size_t>(spoke);
    const std::size_t south_index =
        static_cast<std::size_t>(ntheta - spoke);
    d_sym[north_index] =
        0.5 * (d_north[north_index] - d_south[south_index]);
    d_break[north_index] =
        0.5 * (d_north[north_index] + d_south[south_index]);
    max_break = std::max(max_break, std::abs(d_break[north_index]));
  }
  const double a_odd = band_activation_up(
      max_break / (deadband_frac * dtheta_star), 2.0, 4.0);
  for (int spoke = 0; spoke <= ntheta; ++spoke) {
    const std::size_t north_index = static_cast<std::size_t>(spoke);
    const std::size_t south_index =
        static_cast<std::size_t>(ntheta - spoke);
    if (spoke * 2 == ntheta) {
      const double fixed_point_applied =
          a_odd * d_break[north_index];
      (*out_north_applied)[north_index] = fixed_point_applied;
      (*out_south_applied)[south_index] = fixed_point_applied;
      continue;
    }
    if (a_odd == 0.0) {
      (*out_north_applied)[north_index] = d_sym[north_index];
      (*out_south_applied)[south_index] = -d_sym[north_index];
    } else if (a_odd == 1.0) {
      (*out_north_applied)[north_index] = d_north[north_index];
      (*out_south_applied)[south_index] = d_south[south_index];
    } else {
      (*out_north_applied)[north_index] =
          d_sym[north_index] + a_odd * d_break[north_index];
      (*out_south_applied)[south_index] =
          -d_sym[north_index] + a_odd * d_break[north_index];
    }
  }
  return a_odd;
}

bool pole_theta_legendre_fit(const std::vector<double>& theta,
                             const std::vector<double>& y,
                             const int order,
                             std::vector<double>* coeffs) {
  if (coeffs == nullptr) {
    return false;
  }
  coeffs->clear();
  if (theta.size() != y.size() || theta.size() < 2U || order < 0 ||
      static_cast<std::size_t>(order) >= theta.size()) {
    return false;
  }
  for (std::size_t j = 0; j < theta.size(); ++j) {
    if (!std::isfinite(theta[j]) || !std::isfinite(y[j]) ||
        theta[j] < 0.0 || theta[j] > CUDART_PI ||
        (j > 0U && !(theta[j] > theta[j - 1U]))) {
      return false;
    }
  }

  const std::size_t n_coeffs = static_cast<std::size_t>(order) + 1U;
  std::vector<double> basis(theta.size() * n_coeffs, 0.0);
  std::vector<double> normal(n_coeffs * n_coeffs, 0.0);
  std::vector<double> rhs(n_coeffs, 0.0);
  for (std::size_t j = 0; j < theta.size(); ++j) {
    const double mu = std::cos(theta[j]);
    double* row = basis.data() + j * n_coeffs;
    row[0] = 1.0;
    if (n_coeffs > 1U) {
      row[1] = mu;
    }
    for (std::size_t ell = 2U; ell < n_coeffs; ++ell) {
      row[ell] =
          ((2.0 * static_cast<double>(ell) - 1.0) * mu * row[ell - 1U] -
           (static_cast<double>(ell) - 1.0) * row[ell - 2U]) /
          static_cast<double>(ell);
    }

    const double theta_left =
        j == 0U ? theta.front() : 0.5 * (theta[j - 1U] + theta[j]);
    const double theta_right =
        j + 1U == theta.size()
            ? theta.back()
            : 0.5 * (theta[j] + theta[j + 1U]);
    const double weight =
        std::abs(std::cos(theta_right) - std::cos(theta_left));
    for (std::size_t ell = 0; ell < n_coeffs; ++ell) {
      rhs[ell] += weight * row[ell] * y[j];
      for (std::size_t mode = 0; mode < n_coeffs; ++mode) {
        normal[ell * n_coeffs + mode] +=
            weight * row[ell] * row[mode];
      }
    }
  }

  for (std::size_t column = 0; column < n_coeffs; ++column) {
    std::size_t pivot_row = column;
    double pivot_magnitude =
        std::abs(normal[column * n_coeffs + column]);
    for (std::size_t row = column + 1U; row < n_coeffs; ++row) {
      const double magnitude =
          std::abs(normal[row * n_coeffs + column]);
      if (magnitude > pivot_magnitude) {
        pivot_magnitude = magnitude;
        pivot_row = row;
      }
    }
    if (!(pivot_magnitude > 0.0) || !std::isfinite(pivot_magnitude)) {
      return false;
    }
    if (pivot_row != column) {
      for (std::size_t mode = 0; mode < n_coeffs; ++mode) {
        std::swap(normal[column * n_coeffs + mode],
                  normal[pivot_row * n_coeffs + mode]);
      }
      std::swap(rhs[column], rhs[pivot_row]);
    }

    const double pivot = normal[column * n_coeffs + column];
    for (std::size_t row = column + 1U; row < n_coeffs; ++row) {
      const double factor = normal[row * n_coeffs + column] / pivot;
      normal[row * n_coeffs + column] = 0.0;
      for (std::size_t mode = column + 1U; mode < n_coeffs; ++mode) {
        normal[row * n_coeffs + mode] -=
            factor * normal[column * n_coeffs + mode];
      }
      rhs[row] -= factor * rhs[column];
    }
  }

  std::vector<double> solution(n_coeffs, 0.0);
  for (std::size_t reverse = n_coeffs; reverse > 0U; --reverse) {
    const std::size_t row = reverse - 1U;
    double residual = rhs[row];
    for (std::size_t mode = row + 1U; mode < n_coeffs; ++mode) {
      residual -= normal[row * n_coeffs + mode] * solution[mode];
    }
    const double pivot = normal[row * n_coeffs + row];
    if (!(std::abs(pivot) > 0.0) || !std::isfinite(pivot)) {
      return false;
    }
    solution[row] = residual / pivot;
    if (!std::isfinite(solution[row])) {
      return false;
    }
  }
  *coeffs = std::move(solution);
  return true;
}

double pole_theta_legendre_eval(const std::vector<double>& coeffs,
                                const double theta) {
  if (coeffs.empty()) {
    return 0.0;
  }
  const double mu = std::cos(theta);
  double value = coeffs[0];
  if (coeffs.size() == 1U) {
    return value;
  }
  double p_nm2 = 1.0;
  double p_nm1 = mu;
  value += coeffs[1] * p_nm1;
  for (std::size_t ell = 2U; ell < coeffs.size(); ++ell) {
    const double p_n =
        ((2.0 * static_cast<double>(ell) - 1.0) * mu * p_nm1 -
         (static_cast<double>(ell) - 1.0) * p_nm2) /
        static_cast<double>(ell);
    value += coeffs[ell] * p_n;
    p_nm2 = p_nm1;
    p_nm1 = p_n;
  }
  return value;
}

namespace {

bool pole_theta_pinned_basis(const std::vector<double>& theta,
                             const int lc,
                             int* lc_eff,
                             std::vector<double>* basis) {
  if (lc_eff == nullptr || basis == nullptr || theta.size() < 2U ||
      lc < 0) {
    return false;
  }
  const std::size_t ntheta = theta.size() - 1U;
  *lc_eff = std::min(lc, static_cast<int>(ntheta / 4U));
  const std::size_t stride = static_cast<std::size_t>(*lc_eff) + 1U;
  basis->assign(theta.size() * stride, 0.0);

  std::vector<double> endpoint_begin(stride, 0.0);
  std::vector<double> endpoint_end(stride, 0.0);
  endpoint_begin[0] = 1.0;
  endpoint_end[0] = 1.0;
  if (*lc_eff >= 1) {
    endpoint_begin[1] = std::cos(theta.front());
    endpoint_end[1] = std::cos(theta.back());
  }
  for (int ell = 2; ell <= *lc_eff; ++ell) {
    const auto mode = static_cast<std::size_t>(ell);
    endpoint_begin[mode] =
        ((2.0 * static_cast<double>(ell) - 1.0) *
             endpoint_begin[mode - 1U] * endpoint_begin[1] -
         (static_cast<double>(ell) - 1.0) *
             endpoint_begin[mode - 2U]) /
        static_cast<double>(ell);
    endpoint_end[mode] =
        ((2.0 * static_cast<double>(ell) - 1.0) *
             endpoint_end[mode - 1U] * endpoint_end[1] -
         (static_cast<double>(ell) - 1.0) *
             endpoint_end[mode - 2U]) /
        static_cast<double>(ell);
  }

  const double theta_span = theta.back() - theta.front();
  for (std::size_t j = 1U; j + 1U < theta.size(); ++j) {
    const double fraction = (theta[j] - theta.front()) / theta_span;
    double* row = basis->data() + j * stride;
    std::vector<double> legendre(stride, 0.0);
    legendre[0] = 1.0;
    if (*lc_eff >= 1) {
      legendre[1] = std::cos(theta[j]);
    }
    for (int ell = 2; ell <= *lc_eff; ++ell) {
      const auto mode = static_cast<std::size_t>(ell);
      legendre[mode] =
          ((2.0 * static_cast<double>(ell) - 1.0) *
               legendre[mode - 1U] * legendre[1] -
           (static_cast<double>(ell) - 1.0) * legendre[mode - 2U]) /
          static_cast<double>(ell);
    }
    for (int ell = 1; ell <= *lc_eff; ++ell) {
      const auto mode = static_cast<std::size_t>(ell);
      const double ramp = endpoint_begin[mode] +
          fraction * (endpoint_end[mode] - endpoint_begin[mode]);
      row[mode] = legendre[mode] - ramp;
    }
  }
  return true;
}

}  // namespace

std::vector<double> pole_theta_pinned_band_fit(
    const std::vector<double>& theta,
    const int lc,
    const std::vector<double>& correction) {
  if (theta.size() != correction.size() || theta.size() < 2U || lc < 0) {
    return {};
  }
  for (std::size_t j = 0; j < theta.size(); ++j) {
    if (!std::isfinite(theta[j]) || !std::isfinite(correction[j]) ||
        theta[j] < 0.0 || theta[j] > CUDART_PI ||
        (j > 0U && !(theta[j] > theta[j - 1U]))) {
      return {};
    }
  }

  int lc_eff = 0;
  std::vector<double> basis;
  if (!pole_theta_pinned_basis(theta, lc, &lc_eff, &basis)) {
    return {};
  }
  std::vector<double> coeffs(
      static_cast<std::size_t>(lc_eff) + 1U, 0.0);
  if (lc_eff == 0) {
    return coeffs;
  }

  const std::size_t mode_count = static_cast<std::size_t>(lc_eff);
  const std::size_t stride = mode_count + 1U;
  std::vector<double> normal(mode_count * mode_count, 0.0);
  std::vector<double> rhs(mode_count, 0.0);
  for (std::size_t j = 0; j < theta.size(); ++j) {
    const double weight = std::max(std::sin(theta[j]), 1.0e-12);
    const double* row = basis.data() + j * stride;
    for (std::size_t ell = 0; ell < mode_count; ++ell) {
      const double phi_ell = row[ell + 1U];
      rhs[ell] += weight * phi_ell * correction[j];
      for (std::size_t mode = 0; mode < mode_count; ++mode) {
        normal[ell * mode_count + mode] +=
            weight * phi_ell * row[mode + 1U];
      }
    }
  }

  double max_diagonal = 0.0;
  for (std::size_t mode = 0; mode < mode_count; ++mode) {
    max_diagonal = std::max(
        max_diagonal, normal[mode * mode_count + mode]);
  }
  if (!(max_diagonal > 0.0) || !std::isfinite(max_diagonal)) {
    return coeffs;
  }

  const double pivot_floor = 1.0e-28 * max_diagonal;
  std::vector<double> cholesky(mode_count * mode_count, 0.0);
  std::vector<std::uint8_t> retained(mode_count, 0U);
  for (std::size_t column = 0; column < mode_count; ++column) {
    double pivot = normal[column * mode_count + column];
    for (std::size_t previous = 0; previous < column; ++previous) {
      const double value = cholesky[column * mode_count + previous];
      pivot -= value * value;
    }
    if (!std::isfinite(pivot) || pivot < pivot_floor) {
      continue;
    }
    retained[column] = 1U;
    cholesky[column * mode_count + column] = std::sqrt(pivot);
    for (std::size_t row = column + 1U; row < mode_count; ++row) {
      double residual = normal[row * mode_count + column];
      for (std::size_t previous = 0; previous < column; ++previous) {
        residual -= cholesky[row * mode_count + previous] *
                    cholesky[column * mode_count + previous];
      }
      cholesky[row * mode_count + column] =
          residual / cholesky[column * mode_count + column];
    }
  }

  std::vector<double> forward(mode_count, 0.0);
  for (std::size_t row = 0; row < mode_count; ++row) {
    if (retained[row] == 0U) {
      continue;
    }
    double residual = rhs[row];
    for (std::size_t column = 0; column < row; ++column) {
      residual -= cholesky[row * mode_count + column] * forward[column];
    }
    forward[row] = residual / cholesky[row * mode_count + row];
  }
  for (std::size_t reverse = mode_count; reverse > 0U; --reverse) {
    const std::size_t row = reverse - 1U;
    if (retained[row] == 0U) {
      continue;
    }
    double residual = forward[row];
    for (std::size_t column = row + 1U; column < mode_count; ++column) {
      residual -= cholesky[column * mode_count + row] *
                  coeffs[column + 1U];
    }
    coeffs[row + 1U] =
        residual / cholesky[row * mode_count + row];
  }
  return coeffs;
}

double pole_theta_row_defect(
    const std::vector<double>& theta,
    const int phys_lp,
    const int phys_lc,
    std::vector<double>* dtheta_correction) {
  if (dtheta_correction != nullptr) {
    dtheta_correction->assign(theta.size(), 0.0);
  }
  if (theta.size() < 2U || phys_lp < 0 || phys_lp >= phys_lc) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  for (std::size_t j = 0; j < theta.size(); ++j) {
    if (!std::isfinite(theta[j]) || theta[j] < 0.0 ||
        theta[j] > CUDART_PI ||
        (j > 0U && !(theta[j] > theta[j - 1U]))) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }

  const std::size_t ntheta = theta.size() - 1U;
  const double nominal_pitch =
      (theta.back() - theta.front()) / static_cast<double>(ntheta);
  const double symmetry_tolerance =
      8.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::abs(nominal_pitch));
  bool uniform_row = true;
  for (std::size_t g = 0; g < ntheta; ++g) {
    if (std::abs((theta[g + 1U] - theta[g]) - nominal_pitch) >
        symmetry_tolerance) {
      uniform_row = false;
      break;
    }
  }
  if (uniform_row) {
    return 0.0;
  }

  std::vector<double> dth(ntheta, 0.0);
  std::vector<double> thm(ntheta, 0.0);
  for (std::size_t g = 0; g < ntheta; ++g) {
    dth[g] = theta[g + 1U] - theta[g];
    thm[g] = 0.5 * (theta[g] + theta[g + 1U]);
  }

  const int pitch_fit_order = std::min(
      phys_lc, static_cast<int>(ntheta / 4U));
  std::vector<double> pitch_coeffs;
  if (!pole_theta_legendre_fit(
          thm, dth, pitch_fit_order, &pitch_coeffs)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const auto taper = [phys_lp, phys_lc](const int ell) {
    if (ell <= phys_lp) {
      return 1.0;
    }
    if (ell >= phys_lc) {
      return 0.0;
    }
    return band_smoothstep5(
        static_cast<double>(phys_lc - ell) /
        static_cast<double>(phys_lc - phys_lp));
  };
  for (std::size_t ell = 0; ell < pitch_coeffs.size(); ++ell) {
    pitch_coeffs[ell] *= taper(static_cast<int>(ell));
  }

  const double pitch_floor =
      0.05 * CUDART_PI / static_cast<double>(ntheta);
  std::vector<double> dth_phys(ntheta, 0.0);
  std::vector<double> defect(ntheta, 0.0);
  bool all_defects_zero = true;
  double eighth_sum = 0.0;
  double weight_sum = 0.0;
  for (std::size_t g = 0; g < ntheta; ++g) {
    dth_phys[g] = std::max(
        pole_theta_legendre_eval(pitch_coeffs, thm[g]), pitch_floor);
    defect[g] = std::log(dth[g] / dth_phys[g]);
    all_defects_zero = all_defects_zero && defect[g] == 0.0;
    const double weight = std::max(
        std::sin(thm[g]) * CUDART_PI / static_cast<double>(ntheta),
        1.0e-12);
    eighth_sum += weight * std::pow(std::abs(defect[g]), 8.0);
    weight_sum += weight;
  }
  const double E = std::pow(eighth_sum / weight_sum, 1.0 / 8.0);
  if (dtheta_correction == nullptr || all_defects_zero) {
    return E;
  }

  std::vector<double> dtheta(theta.size(), 0.0);
  double cumulative = 0.0;
  for (std::size_t j = 1U; j <= ntheta; ++j) {
    cumulative += dth_phys[j - 1U] - dth[j - 1U];
    dtheta[j] = cumulative;
  }
  const double total = cumulative;
  for (std::size_t j = 0; j <= ntheta; ++j) {
    dtheta[j] -= static_cast<double>(j) /
                 static_cast<double>(ntheta) * total;
  }

  for (std::size_t j = 0; j <= ntheta; ++j) {
    const double adjacent_pitch =
        j == 0U ? dth_phys.front()
                : (j == ntheta
                       ? dth_phys.back()
                       : 0.5 * (dth_phys[j - 1U] + dth_phys[j]));
    const double cap = 0.35 * adjacent_pitch;
    if (cap > 0.0) {
      dtheta[j] = cap * std::tanh(dtheta[j] / cap);
    }
  }

  const std::vector<double> correction_coeffs =
      pole_theta_pinned_band_fit(theta, phys_lc, dtheta);
  if (correction_coeffs.empty()) {
    return E;
  }
  int correction_lc_eff = 0;
  std::vector<double> correction_basis;
  if (!pole_theta_pinned_basis(
          theta, phys_lc, &correction_lc_eff, &correction_basis)) {
    return E;
  }
  const std::size_t correction_stride =
      static_cast<std::size_t>(correction_lc_eff) + 1U;
  for (std::size_t j = 0; j <= ntheta; ++j) {
    const double* row =
        correction_basis.data() + j * correction_stride;
    for (int ell = 1; ell <= correction_lc_eff; ++ell) {
      dtheta[j] -= taper(ell) *
                   correction_coeffs[static_cast<std::size_t>(ell)] *
                   row[static_cast<std::size_t>(ell)];
    }
  }
  *dtheta_correction = std::move(dtheta);
  return E;
}

namespace {

double pole_theta_eval_plain_row_curve(const PoleThetaRowCurve& curve,
                                       const double theta) {
  if (curve.theta.size() < 2U || curve.s.size() != curve.theta.size() ||
      curve.tangents.size() != curve.theta.size() ||
      !std::isfinite(theta)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (theta <= curve.theta.front()) {
    return curve.s.front();
  }
  if (theta >= curve.theta.back()) {
    return curve.s.back();
  }
  const auto upper =
      std::upper_bound(curve.theta.begin(), curve.theta.end(), theta);
  const std::size_t right =
      static_cast<std::size_t>(upper - curve.theta.begin());
  const std::size_t left = right - 1U;
  const double h = curve.theta[right] - curve.theta[left];
  const double t = (theta - curve.theta[left]) / h;
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
  const double h10 = t3 - 2.0 * t2 + t;
  const double h01 = -2.0 * t3 + 3.0 * t2;
  const double h11 = t3 - t2;
  return h00 * curve.s[left] + h10 * h * curve.tangents[left] +
         h01 * curve.s[right] + h11 * h * curve.tangents[right];
}

}  // namespace

bool pole_theta_build_row_curve(const std::vector<double>& theta,
                                const std::vector<double>& s,
                                const int fit_order,
                                const std::vector<int>& protected_modes,
                                PoleThetaRowCurve* out) {
  if (out == nullptr) {
    return false;
  }
  *out = PoleThetaRowCurve{};
  if (theta.size() != s.size() || theta.size() < 2U || fit_order < 0) {
    return false;
  }
  for (std::size_t j = 0; j < theta.size(); ++j) {
    if (!std::isfinite(theta[j]) || !std::isfinite(s[j]) ||
        theta[j] < 0.0 || theta[j] > CUDART_PI ||
        (j > 0U && !(theta[j] > theta[j - 1U]))) {
      return false;
    }
  }

  PoleThetaRowCurve curve;
  curve.theta = theta;
  curve.s = s;
  curve.tangents.assign(theta.size(), 0.0);
  std::vector<double> intervals(theta.size() - 1U, 0.0);
  std::vector<double> secants(theta.size() - 1U, 0.0);
  for (std::size_t j = 0; j + 1U < theta.size(); ++j) {
    intervals[j] = theta[j + 1U] - theta[j];
    secants[j] = (s[j + 1U] - s[j]) / intervals[j];
  }

  if (theta.size() == 2U) {
    curve.tangents[0] = secants[0];
    curve.tangents[1] = secants[0];
  } else {
    for (std::size_t j = 1U; j + 1U < theta.size(); ++j) {
      const double left = secants[j - 1U];
      const double right = secants[j];
      if (left == 0.0 || right == 0.0 ||
          std::signbit(left) != std::signbit(right)) {
        curve.tangents[j] = 0.0;
      } else {
        const double w1 = 2.0 * intervals[j] + intervals[j - 1U];
        const double w2 = intervals[j] + 2.0 * intervals[j - 1U];
        curve.tangents[j] =
            (w1 + w2) / (w1 / left + w2 / right);
      }
    }

    const auto endpoint_tangent = [](const double h0,
                                     const double h1,
                                     const double d0,
                                     const double d1) {
      double tangent = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
      if (tangent == 0.0 || std::signbit(tangent) != std::signbit(d0)) {
        return 0.0;
      }
      if (std::signbit(d0) != std::signbit(d1) &&
          std::abs(tangent) > 3.0 * std::abs(d0)) {
        tangent = 3.0 * d0;
      }
      return tangent;
    };
    curve.tangents.front() = endpoint_tangent(
        intervals[0], intervals[1], secants[0], secants[1]);
    curve.tangents.back() = endpoint_tangent(
        intervals[intervals.size() - 1U],
        intervals[intervals.size() - 2U],
        secants[secants.size() - 1U],
        secants[secants.size() - 2U]);
  }

  const int effective_fit_order = std::min(
      fit_order, static_cast<int>(theta.size() / 4U));
  curve.mode_correction.assign(
      static_cast<std::size_t>(effective_fit_order) + 1U, 0.0);
  bool have_protected_mode = false;
  for (const int mode : protected_modes) {
    if (mode >= 0 && mode <= effective_fit_order) {
      have_protected_mode = true;
      break;
    }
  }
  if (have_protected_mode) {
    std::vector<double> data_coeffs;
    if (!pole_theta_legendre_fit(
            theta, s, effective_fit_order, &data_coeffs)) {
      return false;
    }

    const std::size_t dense_size = 4U * theta.size() + 1U;
    std::vector<double> dense_theta(dense_size, 0.0);
    std::vector<double> dense_s(dense_size, 0.0);
    for (std::size_t j = 0; j < dense_size; ++j) {
      const double fraction =
          static_cast<double>(j) / static_cast<double>(dense_size - 1U);
      dense_theta[j] =
          theta.front() + fraction * (theta.back() - theta.front());
      dense_s[j] = pole_theta_eval_plain_row_curve(curve, dense_theta[j]);
    }
    std::vector<double> interpolant_coeffs;
    if (!pole_theta_legendre_fit(
            dense_theta, dense_s, effective_fit_order,
            &interpolant_coeffs)) {
      return false;
    }
    for (const int mode : protected_modes) {
      if (mode >= 0 && mode <= effective_fit_order) {
        const std::size_t index = static_cast<std::size_t>(mode);
        curve.mode_correction[index] =
            data_coeffs[index] - interpolant_coeffs[index];
      }
    }
  }

  std::vector<double> sorted_s = s;
  std::sort(sorted_s.begin(), sorted_s.end());
  const std::size_t middle = sorted_s.size() / 2U;
  const double median_s = sorted_s.size() % 2U == 0U
      ? 0.5 * (sorted_s[middle - 1U] + sorted_s[middle])
      : sorted_s[middle];
  double max_abs_correction = 0.0;
  for (const double theta_node : theta) {
    max_abs_correction = std::max(
        max_abs_correction,
        std::abs(pole_theta_legendre_eval(
            curve.mode_correction, theta_node)));
  }
  if (max_abs_correction > 0.5 * median_s) {
    return false;
  }

  *out = std::move(curve);
  return true;
}

double pole_theta_eval_row_curve(const PoleThetaRowCurve& curve,
                                 const double theta) {
  if (curve.theta.size() < 2U || curve.s.size() != curve.theta.size() ||
      curve.tangents.size() != curve.theta.size() ||
      !std::isfinite(theta)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double clamped_theta =
      std::clamp(theta, curve.theta.front(), curve.theta.back());
  return pole_theta_eval_plain_row_curve(curve, clamped_theta) +
         pole_theta_legendre_eval(curve.mode_correction, clamped_theta);
}

bool pole_theta_s_target_admissible(const double s_target,
                                    const double s_min_row,
                                    const double s_max_row) {
  return std::isfinite(s_target) && s_target > 0.0 &&
         0.5 * s_min_row <= s_target &&
         s_target <= 2.0 * s_max_row;
}

double band_respace_capped_move(const double radius,
                                const double target_radius,
                                const double chi,
                                const double local_gap,
                                const double cap_frac) {
  const double full_move = chi * (target_radius - radius);
  const double cap = cap_frac * local_gap;
  return std::isfinite(cap) && cap > 0.0
      ? std::clamp(full_move, -cap, cap)
      : full_move;
}

namespace {

enum class BandKind : std::uint8_t {
  Belt = 0,
  Axis = 1,
  AxisRepair = 2,
  Shell = 3,
  Estimator = 4,
  ClosureCatchment = 5,
  PoleTheta = 6,
};

enum class PcDecision : std::uint8_t {
  Hold = 0,
  Noop = 1,
  Reject = 2,
  Commit = 3,
  Pending = 255,
};

constexpr std::uint8_t kPcClean = 0U;
constexpr std::uint8_t kPcTraveling = 1U;
constexpr std::uint8_t kPcAmbiguous = 2U;
constexpr std::uint8_t kPcPersistent = 3U;

// Center-only target v1 hard-wired parameters.
constexpr double kCenterTargetLambda = 0.1;
constexpr double kCenterTargetMu = 1.0e-3;
constexpr double kCenterTargetAreaFloorFraction = 1.0e-4;
constexpr int kCenterTargetNewtonIterations = 8;
constexpr int kCenterTargetBacktrackingHalvings = 4;
constexpr int kAxisRepairPatchRowHalfwidth = 2;
constexpr int kAxisRepairSweeps = 4;
constexpr int kAxisRepairCooldownSteps = 25;
constexpr double kClosureCatchmentArmSpeed = 1.0e5;
constexpr double kClosureCatchmentOperatingFloorFraction = 0.5;

struct CenterPentagon {
  int center_node = -1;
  std::array<int, 5> boundary_nodes = {{-1, -1, -1, -1, -1}};
  std::array<double, 5> initial_area_fraction = {{0.0, 0.0, 0.0, 0.0, 0.0}};
};

struct AxisRepairCell {
  int cell = -1;
  int block_id = -1;
  int row = -1;
  std::vector<int> row_cells;
  std::vector<double> row_initial_area;
  double initial_area = 0.0;
  double current_area = 0.0;
  double scale = 1.0;
  double ring_scale = 1.0;
  bool engaged = false;
};

struct PcRidgeSample {
  double time = 0.0;
  double radius = 0.0;
  double mass_coordinate = 0.0;
  double max_error = 0.0;
  double pressure_jump = 0.0;
  bool valid = false;
};

struct BandState {
  std::string name;
  BandKind kind = BandKind::Belt;
  int block_id = -1;
  double sort_radius = 0.0;
  std::vector<int> cells;
  std::vector<std::vector<int>> lines;
  std::vector<int> center_nodes;
  std::vector<CenterPentagon> center_pentagons;
  std::vector<AxisRepairCell> axis_repair_cells;
  std::vector<std::vector<int>> shell_rings;
  std::vector<double> shell_initial_area;
  double shell_dr0 = 0.0;
  double shell_min_spacing = std::numeric_limits<double>::infinity();
  bool estimator_polar_geometry = true;
  bool estimator_layer_axis_is_j = false;
  std::vector<std::uint8_t> pinned_nodes;
  core::DeviceArray<int> d_cells;
  core::DeviceArray<double> d_initial_area;
  double min_ratio = std::numeric_limits<double>::infinity();
  double eta_h = std::numeric_limits<double>::quiet_NaN();
  int argmin_cell = -1;
  bool engaged = false;
  std::uint64_t engaged_steps = 0;
  std::int64_t last_counted_step = -1;
  std::uint64_t engagements = 0;
  std::uint64_t skipped_inadmissible = 0;
  std::uint64_t applied_partial_sigma = 0;
  double min_sigma_applied = 1.0;
  int axis_repair_engaged_count = 0;
  std::vector<int> axis_repair_patch_cells;
  int axis_repair_cooldown_steps_remaining = 0;
  std::int64_t axis_repair_cooldown_armed_step = -1;

  // Per-column estimator temporal stitch: the previously selected raw
  // component interval [first cell row, last cell row] for each column.
  std::vector<std::array<int, 2>> pc_previous_intervals;
  // Spherical-dispatch shadow stitch, deliberately independent of the
  // production per-column temporal history.
  std::vector<std::array<int, 2>> pc_shadow_previous_intervals;
  // Per-spoke pinned endpoint node ids for the conservative shared-node
  // closure; -1 denotes an inactive spoke.
  std::vector<int> pc_spoke_inner_nodes;
  std::vector<int> pc_spoke_outer_nodes;
  // Radial ring indices paired with the pinned endpoint ids above.
  std::vector<int> pc_spoke_i0;
  std::vector<int> pc_spoke_i1;
  // Per-spoke activity after column closure and conservative window
  // intersection.
  std::vector<std::uint8_t> pc_spoke_active;
  // Number of estimator evaluations still required before another committed
  // per-column transaction may be proposed.
  int pc_cooldown_events_remaining = 0;
  // Chi used by the last accepted per-column transaction; this is the rate
  // limiter's temporal state.
  double pc_previous_chi = 0.0;
  // Sigma reported for the last accepted per-column transaction; a value
  // below the configured floor triggers retrospective starvation handling.
  double pc_last_committed_sigma = 1.0;
  // Runtime-only four-sample ridge history. run_start replaces the entire
  // runtime, so these buffers reset both at run start and after an epoch swap.
  std::vector<std::array<PcRidgeSample, 4>> pc_ridge_history;
  std::vector<std::uint8_t> pc_ridge_history_count;

  std::vector<int> pc_i0;
  std::vector<int> pc_i1;
  std::vector<std::uint8_t> pc_well_formed;
  std::vector<std::uint8_t> pc_reason;
  std::vector<std::uint8_t> pc_classification;
  std::vector<int> pc_tube_i0;
  std::vector<int> pc_tube_i1;
  std::vector<std::uint8_t> pc_tube_mask;
  std::vector<std::uint8_t> pc_column_active;
  PerColumnStateForTest pc_state_for_test;
  std::vector<double> pc_eta;
  std::vector<double> pc_cached_target_r;
  std::vector<double> pc_cached_target_z;
  double pc_coverage = 0.0;
  double pc_eta_p = 0.0;
  double pc_chi = 0.0;
  int pc_held_columns = 0;
  int pc_active_spokes = 0;
  int pc_patch_count = 0;
  int pc_tube_columns = 0;
  int pc_ambiguous_columns = 0;
  std::uint8_t pc_escalation_reason = 0U;
  bool pc_phase_b_active = false;
  int pc_release_evaluations = 0;
  int pc_sigma_starvation_events_remaining = 0;
  PcDecision pc_predecision = PcDecision::Pending;
  std::string pc_reject_reason;
  bool pc_transaction_pending = false;
  bool pc_transaction_rejected = false;
  double pc_pending_sigma = 1.0;
  double pc_pending_chi = 0.0;
  bool pc_shadow_engaged = false;
  int pc_shadow_release_evaluations = 0;
  bool pc_shadow_hold = false;
  bool pc_legacy_hold = false;
  std::uint64_t pc_shadow_hold_xor_legacy_only = 0;
  std::uint64_t pc_shadow_hold_xor_pc_only = 0;
  bool pc_shadow_engage_decision = false;
  int pc_shadow_first_mismatch = -1;

  int closure_core_block_id = -1;
  int closure_bridge_block_id = -1;
  int closure_columns = 0;
  int closure_held_columns = 0;
  bool closure_topology_supported = false;
  bool closure_no_pool_logged = false;
  bool closure_armed = false;
  bool closure_arm_hold_logged = false;
  bool closure_engaged_latch = false;
  // Runtime-only birth data. run_start resets it, so restart processes
  // intentionally re-birth the closure catchment grading in increment 1.
  bool closure_birth_recorded = false;
  std::vector<std::vector<double>> closure_birth_spacings;
  std::vector<std::vector<int>> closure_birth_outer_nodes;
  std::vector<double> closure_cached_target_r;
  std::vector<double> closure_cached_target_z;
  std::vector<int> closure_target_nodes;
  double closure_min_h = std::numeric_limits<double>::infinity();
  double closure_min_eta = std::numeric_limits<double>::infinity();
  double closure_max_nu = 0.0;
  double closure_prescale = 1.0;
  int closure_pending_absorb_rows = 0;

  // v2 common-mode controller state.
  double closure_lambda_h_ema = 0.0;
  double closure_lambda_m_ema = 0.0;
  double closure_eta_h_prev = 0.0;
  double closure_eta_m_prev = 0.0;
  double closure_prev_eval_t = -1.0;
  double closure_next_eval_t = -1.0;
  double closure_accum_chi = 0.0;
  double closure_rearm_eta_h = 0.0;
  bool closure_rearmed = true;

  // --- PoleTheta controller state (one band, two parity-coupled poles) ---
  struct PoleThetaPole {
    bool armed = false;
    bool fired_this_step = false;
    bool hard = false;
    int consecutive_fire_evals = 0;
    double last_fire_t = -std::numeric_limits<double>::infinity();
    double h_min_at_fire = std::numeric_limits<double>::infinity();
    bool awaiting_improvement = false;
    int widen_halo_extra = 0;
    bool sector_unreachable_logged = false;
    std::vector<std::pair<double, double>> h_history;  // (t, H_min)
    std::vector<double> last_fire_row_a;
    std::vector<std::pair<int, int>> last_fire_row_ids;
    int last_fire_row_count = 0;
  };
  PoleThetaPole pole_north;
  PoleThetaPole pole_south;
  bool pole_routine_acted = false;
  // Fixed 0.5 ps cadence for the first v2 routine-lane release.
  double pole_routine_next_eval_t = -1.0;
  int pole_routine_log_countdown = 0;
  // Runtime-only hard-lane futility state. Restart intentionally resets the
  // debt to zero; checkpoint persistence is out of scope for this release.
  double pole_hard_debt = 0.0;
  double pole_hard_t_last = -1.0e30;
  double pole_hard_t_last_eval = -1.0e30;
  std::set<std::pair<int, int>> pole_hard_signature;
  std::set<std::pair<int, int>> pole_hard_emergency_signature;
  bool pole_hard_emergency_used = false;
  std::uint64_t pole_hard_futile_suppressions = 0;
  std::uint64_t pole_hard_refractory_suppressions = 0;
  int pole_shell_block_id = -1;
  int pole_tier_block_id = -1;
  int pole_bridge_block_id = -1;
};

struct RuntimeState {
  bool initialized = false;
  int rank = 0;
  int n_cells = 0;
  int n_nodes = 0;
  std::uint64_t skipped_uncrosschecked_belts = 0;
  std::uint64_t skipped_shell_axis_blocks = 0;
  std::uint64_t skipped_trivial_axis_blocks = 0;
  std::uint64_t center_fallbacks = 0;
  std::uint64_t axis_repair_total_transactions = 0;
  bool axis_repair_initial_area_finalized = false;
  std::vector<BandState> bands;
  // Per-step pole-theta composition map: node-indexed delta-theta (radians),
  // empty when the controller did not fire this step.
  std::vector<double> pole_theta_dtheta;
  // Positive entries are curve-preserving target radii.
  std::vector<double> pole_theta_target_s;
  bool pole_theta_consumed = false;
  bool pole_theta_composed_last_build = false;
  std::vector<double> axis_repair_initial_area;
  std::vector<double> axis_repair_current_area;
  std::vector<std::uint8_t> axis_repair_axis_nodes;
  std::vector<std::vector<int>> axis_repair_node_incident_cells;
  core::DeviceArray<double> d_current_area;
  core::DeviceArray<double> d_min_ratio;
  core::DeviceArray<int> d_argmin_cell;
};

RuntimeState* g_runtime = nullptr;

__global__ void band_min_ratio_kernel(
    const int* __restrict__ cells,
    const double* __restrict__ initial_area,
    const double* __restrict__ current_area,
    const int n_band_cells,
    const int owned_begin,
    const int owned_end,
    double* __restrict__ min_ratio) {
  __shared__ double block_min[256];
  double local_min = CUDART_INF;
  for (int index = threadIdx.x; index < n_band_cells; index += blockDim.x) {
    const int cell = cells[index];
    if (cell < owned_begin || cell >= owned_end) {
      continue;
    }
    const double ratio = current_area[cell] / initial_area[index];
    local_min = fmin(local_min, isfinite(ratio) ? ratio : -CUDART_INF);
  }
  block_min[threadIdx.x] = local_min;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      block_min[threadIdx.x] =
          fmin(block_min[threadIdx.x], block_min[threadIdx.x + offset]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *min_ratio = block_min[0];
  }
}

__global__ void band_min_ratio_argmin_kernel(
    const int* __restrict__ cells,
    const double* __restrict__ initial_area,
    const double* __restrict__ current_area,
    const int n_band_cells,
    const int owned_begin,
    const int owned_end,
    double* __restrict__ min_ratio,
    int* __restrict__ argmin_cell) {
  __shared__ double block_min[256];
  __shared__ int block_argmin[256];
  double local_min = CUDART_INF;
  int local_argmin = INT_MAX;
  for (int index = threadIdx.x; index < n_band_cells; index += blockDim.x) {
    const int cell = cells[index];
    if (cell < owned_begin || cell >= owned_end) {
      continue;
    }
    const double ratio = current_area[cell] / initial_area[index];
    const double candidate = isfinite(ratio) ? ratio : -CUDART_INF;
    if (candidate < local_min ||
        (candidate == local_min && cell < local_argmin)) {
      local_min = candidate;
      local_argmin = cell;
    }
  }
  block_min[threadIdx.x] = local_min;
  block_argmin[threadIdx.x] = local_argmin;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      const double candidate = block_min[threadIdx.x + offset];
      const int candidate_argmin =
          block_argmin[threadIdx.x + offset];
      if (candidate < block_min[threadIdx.x] ||
          (candidate == block_min[threadIdx.x] &&
           candidate_argmin < block_argmin[threadIdx.x])) {
        block_min[threadIdx.x] = candidate;
        block_argmin[threadIdx.x] = candidate_argmin;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    *min_ratio = block_min[0];
    *argmin_cell = block_argmin[0];
  }
}

std::vector<int> block_row_cells(const mesh::BlockInfo& block,
                                 const int row) {
  TENRYU_ASSERT(
      row >= 0 && row < block.n_i_cells,
      "band ALE structured row index is out of range");
  std::vector<int> cells;
  cells.reserve(static_cast<std::size_t>(block.n_j_cells));
  const int begin = block.cell_begin + row * block.n_j_cells;
  for (int j = 0; j < block.n_j_cells; ++j) {
    cells.push_back(begin + j);
  }
  return cells;
}

void append_cells(std::vector<int>& destination,
                  const std::vector<int>& source) {
  destination.insert(destination.end(), source.begin(), source.end());
}

std::vector<std::vector<int>> structured_node_rings(
    const mesh::MultiBlockTopology& topology,
    const mesh::BlockInfo& block,
    const ale::m1::ring_detail::BlockCornerRoles* borrowed = nullptr) {
  const auto detected =
      ale::m1::ring_detail::detect_block_corner_roles(topology, block);
  const auto* roles = &detected;
  if (!detected.valid) {
    if (block.n_i_cells != 1 || block.n_j_cells < 2 ||
        borrowed == nullptr || !borrowed->valid) {
      return {};
    }
    roles = borrowed;
  }
  std::vector<std::vector<int>> rings;
  rings.reserve(static_cast<std::size_t>(block.n_i_cells) + 1U);
  rings.push_back(ale::m1::ring_detail::block_cell_ring_roles(
      topology, block, *roles, 0, false));
  for (int row = 0; row < block.n_i_cells; ++row) {
    rings.push_back(ale::m1::ring_detail::block_cell_ring_roles(
        topology, block, *roles, row, true));
  }
  return rings;
}

struct BeltEdge {
  int node_a = -1;
  int node_b = -1;
  int count = 0;
};

struct SeamNeighbor {
  int block_id = -1;
  mesh::BlockSide side = mesh::BlockSide::I_MINUS;
  int index_count = 0;
};

bool detect_borrowed_belt_roles(
    const mesh::MultiBlockTopology& topology,
    ale::m1::ring_detail::BlockCornerRoles& roles) {
  const auto detect_role = [&](const mesh::BlockRole role) {
    for (const auto& block : topology.blocks) {
      if (block.role != role || block.n_i_cells < 2 ||
          block.n_j_cells < 2) {
        continue;
      }
      const auto candidate =
          ale::m1::ring_detail::detect_block_corner_roles(
              topology, block);
      if (candidate.valid) {
        roles = candidate;
        return true;
      }
    }
    return false;
  };
  return detect_role(mesh::BlockRole::POLAR_SHELL) ||
         detect_role(mesh::BlockRole::POLAR_TIER);
}

bool find_unique_seam_neighbor(
    const mesh::MultiBlockTopology& topology,
    const int belt_block,
    const mesh::BlockSide belt_side,
    SeamNeighbor& neighbor) {
  int matches = 0;
  for (const auto& seam : topology.seams) {
    if (seam.block_a == belt_block && seam.side_a == belt_side) {
      neighbor = {seam.block_b, seam.side_b, seam.index_count};
      ++matches;
    }
    if (seam.block_b == belt_block && seam.side_b == belt_side) {
      neighbor = {seam.block_a, seam.side_a, seam.index_count};
      ++matches;
    }
  }
  return matches == 1;
}

bool try_structured_boundary_ring(
    const mesh::MultiBlockTopology& topology,
    const mesh::BlockInfo& block,
    const mesh::BlockSide side,
    std::vector<int>& ring,
    const ale::m1::ring_detail::BlockCornerRoles* borrowed = nullptr,
    bool* used_borrowed = nullptr) {
  if (used_borrowed != nullptr) {
    *used_borrowed = false;
  }
  if (block.n_i_cells < 1 || block.n_j_cells < 2 ||
      (side != mesh::BlockSide::I_MINUS &&
       side != mesh::BlockSide::I_PLUS)) {
    return false;
  }
  const auto detected =
      ale::m1::ring_detail::detect_block_corner_roles(topology, block);
  if (detected.valid) {
    const int row =
        side == mesh::BlockSide::I_MINUS ? 0 : block.n_i_cells - 1;
    const bool upper = side == mesh::BlockSide::I_PLUS;
    ring = ale::m1::ring_detail::block_cell_ring_roles(
        topology, block, detected, row, upper);
    return !ring.empty();
  }
  if (block.n_i_cells != 1 ||
      borrowed == nullptr || !borrowed->valid) {
    return false;
  }
  if (used_borrowed != nullptr) {
    *used_borrowed = true;
  }
  // The belt CSR check disambiguates this candidate from the opposite ring.
  const bool upper = side == mesh::BlockSide::I_PLUS;
  ring = ale::m1::ring_detail::block_cell_ring_roles(
      topology, block, *borrowed, 0, upper);
  return !ring.empty();
}

bool try_belt_boundary_edges(
    const mesh::MultiBlockTopology& topology,
    const mesh::BlockInfo& belt_block,
    const std::vector<std::uint8_t>& cell_nverts,
    const ale::m1::ring_detail::BlockCornerRoles& roles,
    std::vector<BeltEdge>& boundary_edges) {
  if (!roles.valid || belt_block.n_i_cells != 1 ||
      belt_block.n_j_cells <= 0 ||
      belt_block.cell_count != belt_block.n_j_cells ||
      roles.in_hi < 0 || roles.in_hi >= 4 ||
      roles.out_hi < 0 || roles.out_hi >= 4 ||
      roles.in_hi == roles.out_hi) {
    return false;
  }

  std::vector<BeltEdge> edges;
  for (int local_cell = 0;
       local_cell < belt_block.cell_count;
       ++local_cell) {
    const int cell = belt_block.cell_begin + local_cell;
    if (cell < 0 ||
        cell >= static_cast<int>(cell_nverts.size()) ||
        static_cast<std::size_t>(cell) + 1U >=
            topology.cell_node_csr_offsets.size() ||
        mesh::mesh_topo_cell_active_nverts(cell_nverts, cell) != 3) {
      return false;
    }
    const int begin =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int end =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
    if (begin < 0 || end - begin < 4 ||
        static_cast<std::size_t>(end) >
            topology.cell_node_csr_indices.size()) {
      return false;
    }
    int node_a = topology.cell_node_csr_indices[
        static_cast<std::size_t>(begin + roles.in_hi)];
    int node_b = topology.cell_node_csr_indices[
        static_cast<std::size_t>(begin + roles.out_hi)];
    if (node_a < 0 || node_b < 0 || node_a == node_b) {
      return false;
    }
    if (node_a > node_b) {
      std::swap(node_a, node_b);
    }
    auto edge = std::find_if(
        edges.begin(),
        edges.end(),
        [node_a, node_b](const BeltEdge& candidate) {
          return candidate.node_a == node_a &&
                 candidate.node_b == node_b;
        });
    if (edge == edges.end()) {
      edges.push_back(BeltEdge{node_a, node_b, 1});
    } else {
      ++edge->count;
      if (edge->count > 2) {
        return false;
      }
    }
  }

  for (const auto& edge : edges) {
    if (edge.count == 1) {
      boundary_edges.push_back(edge);
    }
  }
  return !boundary_edges.empty();
}

bool ring_matches_forward_or_reverse(
    const std::vector<int>& actual,
    const std::vector<int>& expected) {
  return actual == expected ||
         (actual.size() == expected.size() &&
          std::equal(actual.begin(), actual.end(), expected.rbegin()));
}

bool extract_checked_belt_ring(
    const std::vector<BeltEdge>& boundary_edges,
    const std::vector<int>& expected_ring,
    const int n_nodes,
    std::vector<int>& belt_ring) {
  if (expected_ring.size() < 2U) {
    return false;
  }
  std::vector<std::uint8_t> expected(
      static_cast<std::size_t>(n_nodes), 0U);
  for (const int node : expected_ring) {
    if (node < 0 || node >= n_nodes ||
        expected[static_cast<std::size_t>(node)] != 0U) {
      return false;
    }
    expected[static_cast<std::size_t>(node)] = 1U;
  }

  std::vector<std::vector<int>> adjacency(
      static_cast<std::size_t>(n_nodes));
  int included_edges = 0;
  for (const auto& edge : boundary_edges) {
    if (edge.node_a < 0 || edge.node_a >= n_nodes ||
        edge.node_b < 0 || edge.node_b >= n_nodes) {
      return false;
    }
    if (expected[static_cast<std::size_t>(edge.node_a)] == 0U ||
        expected[static_cast<std::size_t>(edge.node_b)] == 0U) {
      continue;
    }
    adjacency[static_cast<std::size_t>(edge.node_a)].push_back(
        edge.node_b);
    adjacency[static_cast<std::size_t>(edge.node_b)].push_back(
        edge.node_a);
    ++included_edges;
  }
  if (included_edges != static_cast<int>(expected_ring.size()) - 1) {
    return false;
  }

  int first = -1;
  int endpoints = 0;
  for (const int node : expected_ring) {
    const auto degree =
        adjacency[static_cast<std::size_t>(node)].size();
    if (degree == 1U) {
      first = node;
      ++endpoints;
    } else if (degree != 2U) {
      return false;
    }
  }
  if (endpoints != 2) {
    return false;
  }

  int previous = -1;
  int current = first;
  while (current >= 0) {
    belt_ring.push_back(current);
    int next = -1;
    for (const int neighbor :
         adjacency[static_cast<std::size_t>(current)]) {
      if (neighbor != previous) {
        next = neighbor;
        break;
      }
    }
    previous = current;
    current = next;
  }
  return belt_ring.size() == expected_ring.size() &&
         ring_matches_forward_or_reverse(belt_ring, expected_ring);
}

double block_mean_radius(const mesh::MultiBlockTopology& topology,
                         const mesh::BlockInfo& block,
                         const std::vector<double>& node_r,
                         const std::vector<double>& node_z,
                         const std::vector<std::uint8_t>& cell_nverts) {
  std::vector<int> nodes;
  for (int cell = block.cell_begin;
       cell < block.cell_begin + block.cell_count;
       ++cell) {
    const int begin =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts =
        mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
    for (int local = 0; local < nverts; ++local) {
      nodes.push_back(topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + local)]);
    }
  }
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  TENRYU_ASSERT(!nodes.empty(), "band ALE belt block must contain nodes");
  double radius_sum = 0.0;
  for (const int node : nodes) {
    radius_sum += std::hypot(
        node_r[static_cast<std::size_t>(node)],
        node_z[static_cast<std::size_t>(node)]);
  }
  return radius_sum / static_cast<double>(nodes.size());
}

double ring_mean_radius(const std::vector<int>& ring,
                        const std::vector<double>& node_r,
                        const std::vector<double>& node_z) {
  TENRYU_ASSERT(!ring.empty(), "band ALE ring must be non-empty");
  double radius_sum = 0.0;
  for (const int node : ring) {
    TENRYU_ASSERT(
        node >= 0 &&
            static_cast<std::size_t>(node) < node_r.size() &&
            static_cast<std::size_t>(node) < node_z.size(),
        "band ALE ring node id is out of range");
    radius_sum += std::hypot(
        node_r[static_cast<std::size_t>(node)],
        node_z[static_cast<std::size_t>(node)]);
  }
  return radius_sum / static_cast<double>(ring.size());
}

std::vector<double> ring_mean_radii(
    const std::vector<std::vector<int>>& rings,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  std::vector<double> mean_radius;
  mean_radius.reserve(rings.size());
  for (const auto& ring : rings) {
    mean_radius.push_back(ring_mean_radius(ring, node_r, node_z));
  }
  return mean_radius;
}

double ring_spacing_nonuniformity(
    const std::vector<std::vector<int>>& rings,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  TENRYU_ASSERT(
      rings.size() >= 3U,
      "band ALE spacing nonuniformity requires at least three rings");
  const std::vector<double> mean_radius =
      ring_mean_radii(rings, node_r, node_z);
  std::vector<double> spacing;
  spacing.reserve(mean_radius.size() - 1U);
  for (std::size_t ring = 0; ring + 1U < mean_radius.size(); ++ring) {
    spacing.push_back(
        std::abs(mean_radius[ring] - mean_radius[ring + 1U]));
  }
  std::sort(spacing.begin(), spacing.end());
  const std::size_t middle = spacing.size() / 2U;
  const double median =
      spacing.size() % 2U == 0U
          ? 0.5 * (spacing[middle - 1U] + spacing[middle])
          : spacing[middle];
  return spacing.back() / median;
}

double layer_spacing_nonuniformity(
    const std::vector<std::vector<int>>& lines,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const bool layer_axis_is_j) {
  TENRYU_ASSERT(
      lines.size() >= 3U,
      "band ALE spacing nonuniformity requires at least three lines");
  std::vector<double> mean_coordinate;
  mean_coordinate.reserve(lines.size());
  for (const auto& line : lines) {
    TENRYU_ASSERT(
        !line.empty(),
        "band ALE spacing line must be non-empty");
    double coordinate_sum = 0.0;
    for (const int node : line) {
      TENRYU_ASSERT(
          node >= 0 && static_cast<std::size_t>(node) < node_r.size() &&
              static_cast<std::size_t>(node) < node_z.size(),
          "band ALE spacing line node id is out of range");
      coordinate_sum += layer_axis_is_j
                            ? node_z[static_cast<std::size_t>(node)]
                            : node_r[static_cast<std::size_t>(node)];
    }
    mean_coordinate.push_back(
        coordinate_sum / static_cast<double>(line.size()));
  }
  std::vector<double> spacing;
  spacing.reserve(mean_coordinate.size() - 1U);
  for (std::size_t line = 0; line + 1U < mean_coordinate.size(); ++line) {
    spacing.push_back(
        std::abs(mean_coordinate[line] - mean_coordinate[line + 1U]));
  }
  std::sort(spacing.begin(), spacing.end());
  const std::size_t middle = spacing.size() / 2U;
  const double median =
      spacing.size() % 2U == 0U
          ? 0.5 * (spacing[middle - 1U] + spacing[middle])
          : spacing[middle];
  return spacing.back() / median;
}

std::vector<std::vector<int>> build_node_incident_cells(
    const core::State& state) {
  const auto& topology = *state.mesh.topo.multiblock;
  std::vector<std::vector<int>> incident_cells(
      static_cast<std::size_t>(state.mesh.topo.n_nodes));
  for (int cell = 0; cell < state.mesh.topo.n_cells; ++cell) {
    const int begin =
        topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts =
        mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, cell);
    for (int local = 0; local < nverts; ++local) {
      const int node = topology.cell_node_csr_indices[
          static_cast<std::size_t>(begin + local)];
      TENRYU_ASSERT(
          node >= 0 && node < state.mesh.topo.n_nodes,
          "band ALE center incident map node id is out of range");
      incident_cells[static_cast<std::size_t>(node)].push_back(cell);
    }
  }
  return incident_cells;
}

struct CenterTriangleEdge {
  int node_a = -1;
  int node_b = -1;
};

double oriented_triangle_area(const double center_r,
                              const double center_z,
                              const double node_a_r,
                              const double node_a_z,
                              const double node_b_r,
                              const double node_b_z) {
  return 0.5 *
         ((node_a_r - center_r) * (node_b_z - center_z) -
          (node_a_z - center_z) * (node_b_r - center_r));
}

void initialize_center_pentagons(
    const core::State& state,
    const std::vector<double>& initial_r,
    const std::vector<double>& initial_z,
    const std::vector<std::vector<int>>& incident_cells,
    BandState& band) {
  if (band.center_nodes.empty()) {
    return;
  }
  const auto& topology = *state.mesh.topo.multiblock;
  std::sort(band.center_nodes.begin(), band.center_nodes.end());
  TENRYU_ASSERT(
      std::adjacent_find(
          band.center_nodes.begin(), band.center_nodes.end()) ==
          band.center_nodes.end(),
      "band ALE center chain contains duplicate node ids");

  std::vector<std::uint8_t> line_nodes(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  for (const auto& line : band.lines) {
    for (const int node : line) {
      line_nodes[static_cast<std::size_t>(node)] = 1U;
    }
  }

  band.center_pentagons.reserve(band.center_nodes.size());
  for (const int center : band.center_nodes) {
    TENRYU_ASSERT(
        center >= 0 && center < state.mesh.topo.n_nodes,
        "band ALE center node id is out of range");
    const auto& cells =
        incident_cells[static_cast<std::size_t>(center)];
    TENRYU_ASSERT(
        cells.size() == 5U,
        "band ALE center node must have exactly five incident cells");

    std::array<CenterTriangleEdge, 5> triangle_edges;
    std::vector<int> boundary_nodes;
    boundary_nodes.reserve(5U);
    for (std::size_t triangle = 0; triangle < cells.size(); ++triangle) {
      const int cell = cells[triangle];
      TENRYU_ASSERT(
          mesh::mesh_topo_cell_active_nverts(
              state.mesh.cell_nverts, cell) == 3,
          "band ALE center pentagon incident cell must be a triangle");
      const int begin =
          topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
      CenterTriangleEdge edge;
      for (int local = 0; local < 3; ++local) {
        const int node = topology.cell_node_csr_indices[
            static_cast<std::size_t>(begin + local)];
        if (node == center) {
          continue;
        }
        if (edge.node_a < 0) {
          edge.node_a = node;
        } else if (edge.node_b < 0) {
          edge.node_b = node;
        } else {
          TENRYU_ASSERT(
              false,
              "band ALE center triangle has more than two boundary nodes");
        }
      }
      TENRYU_ASSERT(
          edge.node_a >= 0 && edge.node_b >= 0 &&
              edge.node_a != edge.node_b,
          "band ALE center triangle boundary edge is invalid");
      triangle_edges[triangle] = edge;
      boundary_nodes.push_back(edge.node_a);
      boundary_nodes.push_back(edge.node_b);
    }

    std::sort(boundary_nodes.begin(), boundary_nodes.end());
    boundary_nodes.erase(
        std::unique(boundary_nodes.begin(), boundary_nodes.end()),
        boundary_nodes.end());
    TENRYU_ASSERT(
        boundary_nodes.size() == 5U,
        "band ALE center incident triangles do not form a pentagon");
    std::array<std::vector<int>, 5> adjacency;
    for (const auto& edge : triangle_edges) {
      const auto node_a =
          std::lower_bound(
              boundary_nodes.begin(), boundary_nodes.end(), edge.node_a) -
          boundary_nodes.begin();
      const auto node_b =
          std::lower_bound(
              boundary_nodes.begin(), boundary_nodes.end(), edge.node_b) -
          boundary_nodes.begin();
      TENRYU_ASSERT(
          node_a < static_cast<std::ptrdiff_t>(boundary_nodes.size()) &&
              node_b < static_cast<std::ptrdiff_t>(boundary_nodes.size()),
          "band ALE center boundary node lookup failed");
      adjacency[static_cast<std::size_t>(node_a)].push_back(edge.node_b);
      adjacency[static_cast<std::size_t>(node_b)].push_back(edge.node_a);
    }
    for (auto& neighbors : adjacency) {
      std::sort(neighbors.begin(), neighbors.end());
      TENRYU_ASSERT(
          neighbors.size() == 2U && neighbors[0] != neighbors[1],
          "band ALE center incident triangles do not form one cycle");
    }

    CenterPentagon pentagon;
    pentagon.center_node = center;
    pentagon.boundary_nodes[0] = boundary_nodes.front();
    int previous = -1;
    int current = pentagon.boundary_nodes[0];
    for (std::size_t position = 1;
         position < pentagon.boundary_nodes.size();
         ++position) {
      const auto current_index =
          std::lower_bound(
              boundary_nodes.begin(), boundary_nodes.end(), current) -
          boundary_nodes.begin();
      const auto& neighbors =
          adjacency[static_cast<std::size_t>(current_index)];
      const int next =
          neighbors[0] != previous ? neighbors[0] : neighbors[1];
      TENRYU_ASSERT(
          next != pentagon.boundary_nodes[0] &&
              std::find(
                  pentagon.boundary_nodes.begin(),
                  pentagon.boundary_nodes.begin() +
                      static_cast<std::ptrdiff_t>(position),
                  next) ==
                  pentagon.boundary_nodes.begin() +
                      static_cast<std::ptrdiff_t>(position),
          "band ALE center boundary cycle closes early");
      pentagon.boundary_nodes[position] = next;
      previous = current;
      current = next;
    }
    const auto last_index =
        std::lower_bound(
            boundary_nodes.begin(), boundary_nodes.end(), current) -
        boundary_nodes.begin();
    TENRYU_ASSERT(
        std::find(
            adjacency[static_cast<std::size_t>(last_index)].begin(),
            adjacency[static_cast<std::size_t>(last_index)].end(),
            pentagon.boundary_nodes[0]) !=
            adjacency[static_cast<std::size_t>(last_index)].end(),
        "band ALE center boundary cycle does not close");

    double initial_polygon_area = 0.0;
    for (std::size_t edge = 0;
         edge < pentagon.boundary_nodes.size();
         ++edge) {
      const auto next = (edge + 1U) % pentagon.boundary_nodes.size();
      const auto node_a =
          static_cast<std::size_t>(pentagon.boundary_nodes[edge]);
      const auto node_b =
          static_cast<std::size_t>(pentagon.boundary_nodes[next]);
      initial_polygon_area +=
          initial_r[node_a] * initial_z[node_b] -
          initial_z[node_a] * initial_r[node_b];
    }
    if (initial_polygon_area < 0.0) {
      std::reverse(
          pentagon.boundary_nodes.begin() + 1,
          pentagon.boundary_nodes.end());
      initial_polygon_area = -initial_polygon_area;
    }
    TENRYU_ASSERT(
        std::isfinite(initial_polygon_area) &&
            initial_polygon_area > 0.0,
        "band ALE initial center boundary polygon must have positive area");

    double initial_pentagon_area = 0.0;
    std::array<double, 5> initial_triangle_area = {{
        0.0, 0.0, 0.0, 0.0, 0.0}};
    const auto center_index = static_cast<std::size_t>(center);
    for (std::size_t triangle = 0;
         triangle < pentagon.boundary_nodes.size();
         ++triangle) {
      const auto next =
          (triangle + 1U) % pentagon.boundary_nodes.size();
      const int node_a = pentagon.boundary_nodes[triangle];
      const int node_b = pentagon.boundary_nodes[next];
      const auto matching_edge = std::find_if(
          triangle_edges.begin(),
          triangle_edges.end(),
          [node_a, node_b](const CenterTriangleEdge& edge) {
            return (edge.node_a == node_a && edge.node_b == node_b) ||
                   (edge.node_a == node_b && edge.node_b == node_a);
          });
      TENRYU_ASSERT(
          matching_edge != triangle_edges.end(),
          "band ALE center boundary edge has no incident triangle");
      const double area = oriented_triangle_area(
          initial_r[center_index],
          initial_z[center_index],
          initial_r[static_cast<std::size_t>(node_a)],
          initial_z[static_cast<std::size_t>(node_a)],
          initial_r[static_cast<std::size_t>(node_b)],
          initial_z[static_cast<std::size_t>(node_b)]);
      TENRYU_ASSERT(
          std::isfinite(area) && area > 0.0,
          "band ALE initial center triangle must have positive area");
      initial_triangle_area[triangle] = area;
      initial_pentagon_area += area;
      TENRYU_ASSERT(
          line_nodes[static_cast<std::size_t>(node_a)] != 0U,
          "band ALE center boundary node must have a ring-line target");
    }
    TENRYU_ASSERT(
        std::isfinite(initial_pentagon_area) &&
            initial_pentagon_area > 0.0,
        "band ALE initial center pentagon must have positive area");
    for (std::size_t triangle = 0;
         triangle < pentagon.initial_area_fraction.size();
         ++triangle) {
      pentagon.initial_area_fraction[triangle] =
          initial_triangle_area[triangle] / initial_pentagon_area;
    }
    band.center_pentagons.push_back(pentagon);
  }
}

bool make_belt_band(
    const core::State& state,
    const int belt_block_id,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const ale::m1::ring_detail::BlockCornerRoles& borrowed_roles,
    BandState& band) {
  const auto& topology = *state.mesh.topo.multiblock;
  const auto& belt_block =
      topology.blocks[static_cast<std::size_t>(belt_block_id)];
  SeamNeighbor outer_neighbor;
  SeamNeighbor inner_neighbor;
  if (!find_unique_seam_neighbor(
          topology,
          belt_block_id,
          mesh::BlockSide::I_MINUS,
          outer_neighbor) ||
      !find_unique_seam_neighbor(
          topology,
          belt_block_id,
          mesh::BlockSide::I_PLUS,
          inner_neighbor)) {
    return false;
  }
  const std::string crosscheck_message =
      "band ALE transition belt block " +
      std::to_string(belt_block_id) +
      " borrowed corner-role boundary rings do not match adjacent tier "
      "seam rings";
  if (outer_neighbor.block_id < 0 ||
      outer_neighbor.block_id >=
          static_cast<int>(topology.blocks.size()) ||
      inner_neighbor.block_id < 0 ||
      inner_neighbor.block_id >=
          static_cast<int>(topology.blocks.size()) ||
      outer_neighbor.index_count <= 0 ||
      inner_neighbor.index_count <= 0) {
    return false;
  }
  const auto& outer_tier =
      topology.blocks[static_cast<std::size_t>(outer_neighbor.block_id)];
  const auto& inner_tier =
      topology.blocks[static_cast<std::size_t>(inner_neighbor.block_id)];
  TENRYU_ASSERT(
      outer_neighbor.side == mesh::BlockSide::I_PLUS &&
          inner_neighbor.side == mesh::BlockSide::I_MINUS &&
          outer_tier.role == mesh::BlockRole::POLAR_TIER &&
          inner_tier.role == mesh::BlockRole::POLAR_TIER,
      crosscheck_message);

  std::vector<int> outer_seam_ring;
  std::vector<int> inner_seam_ring;
  bool outer_used_borrowed = false;
  bool inner_used_borrowed = false;
  if (!try_structured_boundary_ring(
          topology,
          outer_tier,
          outer_neighbor.side,
          outer_seam_ring,
          &borrowed_roles,
          &outer_used_borrowed) ||
      !try_structured_boundary_ring(
          topology,
          inner_tier,
          inner_neighbor.side,
          inner_seam_ring,
          &borrowed_roles,
          &inner_used_borrowed)) {
    return false;
  }
  TENRYU_ASSERT(
      (outer_used_borrowed ||
       outer_neighbor.index_count + 1 ==
           static_cast<int>(outer_seam_ring.size())) &&
          (inner_used_borrowed ||
           inner_neighbor.index_count + 1 ==
               static_cast<int>(inner_seam_ring.size())),
      crosscheck_message);

  std::vector<BeltEdge> belt_boundary_edges;
  if (!try_belt_boundary_edges(
          topology,
          belt_block,
          state.mesh.cell_nverts,
          borrowed_roles,
          belt_boundary_edges)) {
    return false;
  }
  std::vector<int> belt_outer_ring;
  std::vector<int> belt_inner_ring;
  bool outer_flipped = false;
  bool inner_flipped = false;
  const auto crosscheck_tier_ring =
      [&](const mesh::BlockInfo& tier,
          const mesh::BlockSide side,
          const bool used_borrowed,
          std::vector<int>& seam_ring,
          std::vector<int>& belt_ring,
          bool& flipped) {
        if (!used_borrowed) {
          return extract_checked_belt_ring(
              belt_boundary_edges,
              seam_ring,
              state.mesh.topo.n_nodes,
              belt_ring);
        }
        const bool upper = side == mesh::BlockSide::I_PLUS;
        auto alternate_ring =
            ale::m1::ring_detail::block_cell_ring_roles(
                topology, tier, borrowed_roles, 0, !upper);
        if (extract_checked_belt_ring(
                belt_boundary_edges,
                seam_ring,
                state.mesh.topo.n_nodes,
                belt_ring)) {
          return true;
        }
        belt_ring.clear();
        if (!extract_checked_belt_ring(
                belt_boundary_edges,
                alternate_ring,
                state.mesh.topo.n_nodes,
                belt_ring)) {
          return false;
        }
        seam_ring = std::move(alternate_ring);
        flipped = true;
        return true;
      };
  const bool outer_matches = crosscheck_tier_ring(
      outer_tier,
      outer_neighbor.side,
      outer_used_borrowed,
      outer_seam_ring,
      belt_outer_ring,
      outer_flipped);
  const bool inner_matches = crosscheck_tier_ring(
      inner_tier,
      inner_neighbor.side,
      inner_used_borrowed,
      inner_seam_ring,
      belt_inner_ring,
      inner_flipped);
  TENRYU_ASSERT(
      (outer_used_borrowed || outer_matches) &&
          (inner_used_borrowed || inner_matches),
      crosscheck_message);
  if (!outer_matches || !inner_matches ||
      (outer_used_borrowed &&
       outer_neighbor.index_count + 1 !=
           static_cast<int>(outer_seam_ring.size())) ||
      (inner_used_borrowed &&
       inner_neighbor.index_count + 1 !=
           static_cast<int>(inner_seam_ring.size()))) {
    return false;
  }
  if (belt_outer_ring != outer_seam_ring) {
    std::reverse(belt_outer_ring.begin(), belt_outer_ring.end());
  }
  if (belt_inner_ring != inner_seam_ring) {
    std::reverse(belt_inner_ring.begin(), belt_inner_ring.end());
  }

  auto outer_rings =
      structured_node_rings(topology, outer_tier, &borrowed_roles);
  auto inner_rings =
      structured_node_rings(topology, inner_tier, &borrowed_roles);
  if (outer_rings.empty() || inner_rings.empty()) {
    return false;
  }
  if (outer_flipped) {
    std::reverse(outer_rings.begin(), outer_rings.end());
  }
  if (inner_flipped) {
    std::reverse(inner_rings.begin(), inner_rings.end());
  }

  band.kind = BandKind::Belt;
  band.block_id = belt_block_id;
  band.sort_radius = block_mean_radius(
      topology, belt_block, node_r, node_z, state.mesh.cell_nverts);
  append_cells(
      band.cells,
      block_row_cells(outer_tier, outer_tier.n_i_cells - 1));
  for (int cell = belt_block.cell_begin;
       cell < belt_block.cell_begin + belt_block.cell_count;
       ++cell) {
    band.cells.push_back(cell);
  }
  append_cells(band.cells, block_row_cells(inner_tier, 0));
  band.lines.push_back(
      outer_rings[static_cast<std::size_t>(outer_tier.n_i_cells - 1)]);
  band.lines.push_back(std::move(belt_outer_ring));
  auto belt_interior_rings =
      ale::m1::ring_detail::transition_belt_interior_rings(belt_block);
  if (belt_block.n_i_cells == 1) {
    TENRYU_ASSERT(
        belt_interior_rings.size() == 1U,
        "band ALE one-row transition belt must have one center chain");
    band.center_nodes = belt_interior_rings.front();
  }
  for (auto& ring : belt_interior_rings) {
    band.lines.push_back(std::move(ring));
  }
  band.lines.push_back(std::move(belt_inner_ring));
  band.lines.push_back(inner_rings[1]);
  band.pinned_nodes.assign(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  for (const int node : band.lines.front()) {
    band.pinned_nodes[static_cast<std::size_t>(node)] = 1U;
  }
  for (const int node : band.lines.back()) {
    band.pinned_nodes[static_cast<std::size_t>(node)] = 1U;
  }
  return true;
}

std::vector<BandState> make_dendrite_belt_bands(
    const core::State& state,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  const auto& topology = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(
      topology.dendrite_block_rings.size() == topology.blocks.size(),
      "band ALE dendrite ring table must match the block table");

  int shell_block = -1;
  std::vector<int> tier_blocks;
  std::vector<int> transition_blocks;
  for (int block_id = 0;
       block_id < static_cast<int>(topology.blocks.size());
       ++block_id) {
    const auto role =
        topology.blocks[static_cast<std::size_t>(block_id)].role;
    if (role == mesh::BlockRole::POLAR_SHELL) {
      TENRYU_ASSERT(
          shell_block < 0,
          "band ALE dendrite requires exactly one shell block");
      shell_block = block_id;
    } else if (role == mesh::BlockRole::POLAR_TIER) {
      tier_blocks.push_back(block_id);
    } else if (role == mesh::BlockRole::TRANSITION_BELT) {
      transition_blocks.push_back(block_id);
    }
  }
  TENRYU_ASSERT(
      shell_block >= 0 && tier_blocks.size() == 6U &&
          transition_blocks.size() == 6U,
      "band ALE dendrite block descriptor cardinality mismatch");

  const auto& shell_rings =
      topology.dendrite_block_rings[
          static_cast<std::size_t>(shell_block)];
  const auto tier_rings = [&](const std::size_t tier) -> const auto& {
    return topology.dendrite_block_rings[
        static_cast<std::size_t>(tier_blocks[tier])];
  };
  const auto transition_rings =
      [&](const std::size_t transition) -> const auto& {
    return topology.dendrite_block_rings[
        static_cast<std::size_t>(transition_blocks[transition])];
  };

  TENRYU_ASSERT(
      !shell_rings.inner_adjacent_ring.empty() &&
          !shell_rings.inner_ring.empty(),
      "band ALE dendrite shell ring descriptors are incomplete");
  for (std::size_t tier = 0; tier < tier_blocks.size(); ++tier) {
    const auto& rings = tier_rings(tier);
    TENRYU_ASSERT(
        !rings.outer_ring.empty() &&
            !rings.outer_adjacent_ring.empty() &&
            !rings.inner_adjacent_ring.empty() &&
            !rings.inner_ring.empty(),
        "band ALE dendrite tier ring descriptors are incomplete");
  }
  for (std::size_t transition = 0;
       transition < transition_blocks.size();
       ++transition) {
    const auto& rings = transition_rings(transition);
    TENRYU_ASSERT(
        !rings.outer_ring.empty() && !rings.inner_ring.empty(),
        "band ALE dendrite transition ring descriptors are incomplete");
  }

  TENRYU_ASSERT(
      shell_rings.inner_ring == tier_rings(0).outer_ring &&
          tier_rings(0).inner_ring ==
              transition_rings(0).outer_ring &&
          transition_rings(0).inner_ring ==
              tier_rings(1).outer_ring &&
          tier_rings(1).inner_ring ==
              transition_rings(1).outer_ring &&
          transition_rings(1).inner_ring ==
              transition_rings(2).outer_ring &&
          transition_rings(2).inner_ring ==
              tier_rings(2).outer_ring,
      "band ALE dendrite outer transition chain mismatch");
  for (std::size_t tier = 2; tier + 1U < tier_blocks.size(); ++tier) {
    const std::size_t transition = tier + 1U;
    TENRYU_ASSERT(
        tier_rings(tier).inner_ring ==
                transition_rings(transition).outer_ring &&
            transition_rings(transition).inner_ring ==
                tier_rings(tier + 1U).outer_ring,
        "band ALE dendrite transition chain mismatch");
  }

  const auto append_block_cells =
      [&](std::vector<int>& cells, const int block_id) {
        const auto& block =
            topology.blocks[static_cast<std::size_t>(block_id)];
        for (int cell = block.cell_begin;
             cell < block.cell_begin + block.cell_count;
             ++cell) {
          cells.push_back(cell);
        }
      };
  const auto make_band =
      [&](const int characteristic_block,
          std::vector<int> cells,
          std::vector<std::vector<int>> lines,
          std::vector<int> center_nodes) {
        BandState band;
        band.kind = BandKind::Belt;
        band.block_id = characteristic_block;
        band.sort_radius = block_mean_radius(
            topology,
            topology.blocks[
                static_cast<std::size_t>(characteristic_block)],
            node_r,
            node_z,
            state.mesh.cell_nverts);
        band.cells = std::move(cells);
        band.lines = std::move(lines);
        band.lines.erase(
            std::remove_if(
                band.lines.begin(), band.lines.end(),
                [](const auto& line) { return line.empty(); }),
            band.lines.end());
        band.center_nodes = std::move(center_nodes);
        band.pinned_nodes.assign(
            static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
        for (const int node : band.lines.front()) {
          band.pinned_nodes[static_cast<std::size_t>(node)] = 1U;
        }
        for (const int node : band.lines.back()) {
          band.pinned_nodes[static_cast<std::size_t>(node)] = 1U;
        }
        return band;
      };

  std::vector<BandState> bands;
  bands.reserve(5U);

  std::vector<int> cells;
  const auto& tier1a =
      topology.blocks[static_cast<std::size_t>(tier_blocks[0])];
  append_cells(
      cells, block_row_cells(tier1a, tier1a.n_i_cells - 1));
  append_block_cells(cells, transition_blocks[0]);
  append_cells(
      cells,
      block_row_cells(
          topology.blocks[
              static_cast<std::size_t>(tier_blocks[1])],
          0));
  bands.push_back(make_band(
      transition_blocks[0],
      std::move(cells),
      {
          tier_rings(0).inner_adjacent_ring,
          tier_rings(0).inner_ring,
          transition_rings(0).center_chain,
          transition_rings(0).inner_ring,
          tier_rings(1).outer_adjacent_ring,
      },
      transition_rings(0).center_chain));

  cells.clear();
  const auto& tier1b =
      topology.blocks[static_cast<std::size_t>(tier_blocks[1])];
  append_cells(
      cells, block_row_cells(tier1b, tier1b.n_i_cells - 1));
  append_block_cells(cells, transition_blocks[1]);
  append_block_cells(cells, transition_blocks[2]);
  append_cells(
      cells,
      block_row_cells(
          topology.blocks[
              static_cast<std::size_t>(tier_blocks[2])],
          0));
  bands.push_back(make_band(
      transition_blocks[1],
      std::move(cells),
      {
          tier_rings(1).inner_adjacent_ring,
          tier_rings(1).inner_ring,
          transition_rings(1).center_chain,
          transition_rings(1).inner_ring,
          transition_rings(2).center_chain,
          transition_rings(2).inner_ring,
          tier_rings(2).outer_adjacent_ring,
      },
      [&] {
        std::vector<int> center_nodes =
            transition_rings(1).center_chain;
        center_nodes.insert(
            center_nodes.end(),
            transition_rings(2).center_chain.begin(),
            transition_rings(2).center_chain.end());
        return center_nodes;
      }()));

  for (std::size_t outer_tier = 2; outer_tier < 5U;
       ++outer_tier) {
    cells.clear();
    const auto& outer_block =
        topology.blocks[
            static_cast<std::size_t>(tier_blocks[outer_tier])];
    append_cells(
        cells,
        block_row_cells(
            outer_block, outer_block.n_i_cells - 1));
    const std::size_t transition = outer_tier + 1U;
    append_block_cells(cells, transition_blocks[transition]);
    append_cells(
        cells,
        block_row_cells(
            topology.blocks[
                static_cast<std::size_t>(
                    tier_blocks[outer_tier + 1U])],
            0));
    bands.push_back(make_band(
        transition_blocks[transition],
        std::move(cells),
        {
            tier_rings(outer_tier).inner_adjacent_ring,
            tier_rings(outer_tier).inner_ring,
            transition_rings(transition).center_chain,
            transition_rings(transition).inner_ring,
            tier_rings(outer_tier + 1U).outer_adjacent_ring,
        },
        transition_rings(transition).center_chain));
  }

  return bands;
}

// §18.6: the estimator band's lattice abstraction. Multiblock wraps the
// POLAR_SHELL block (identical indexing to the historic code — bit-for-bit);
// single-block wraps the structured nr x nz lattice with a configurable
// layer axis.
struct BandLatticeView {
  bool single_block = false;
  bool layer_axis_is_j = false;   // single-block only; false = layers along i
  bool polar_geometry = true;     // radius-based respace/spacing vs coordinate
  int cell_begin = 0;
  int n_layers = 0;               // rows the band scans/windows over
  int n_across = 0;               // cells per row
  int nr = 0, nz = 0;             // single-block lattice extents
};

BandLatticeView make_multiblock_band_view(
    const mesh::MultiBlockTopology& topology,
    const mesh::BlockInfo& block) {
  TENRYU_ASSERT(
      block.role == mesh::BlockRole::POLAR_SHELL,
      "band ALE multiblock lattice view requires the polar shell block");
  TENRYU_ASSERT(
      &block >= topology.blocks.data() &&
          &block < topology.blocks.data() + topology.blocks.size(),
      "band ALE multiblock lattice view block is not in the topology");
  BandLatticeView view;
  view.cell_begin = block.cell_begin;
  view.n_layers = block.n_i_cells;
  view.n_across = block.n_j_cells;
  return view;
}

BandLatticeView make_single_block_band_view(
    const core::State& state,
    const core::Config& cfg) {
  TENRYU_ASSERT(
      state.mesh.dim == 2 && !state.mesh.topo.multiblock.has_value() &&
          state.mesh.topo.nr >= 5 && state.mesh.topo.nz >= 5 &&
          state.mesh.topo.n_cells ==
              state.mesh.topo.nr * state.mesh.topo.nz,
      "band ALE estimator requires a structured single-block 2D lattice "
      "with nr,nz >= 5");
  BandLatticeView view;
  view.single_block = true;
  view.polar_geometry =
      cfg.mesh.logical_mesh_2d.rfind("spherical_polar", 0) == 0;
  const std::string& configured_axis =
      cfg.numerics.ale.band_ale.estimator_band_axis;
  TENRYU_ASSERT(
      configured_axis == "auto" || configured_axis == "i" ||
          configured_axis == "j",
      "band ALE estimator lattice axis must be auto, i, or j");
  view.layer_axis_is_j =
      configured_axis == "j" ||
      (configured_axis == "auto" && !view.polar_geometry);
  view.nr = state.mesh.topo.nr;
  view.nz = state.mesh.topo.nz;
  view.n_layers = view.layer_axis_is_j ? view.nz : view.nr;
  view.n_across = view.layer_axis_is_j ? view.nr : view.nz;
  return view;
}

int view_cell(const BandLatticeView& view,
              const int layer,
              const int across) {
  TENRYU_ASSERT(
      layer >= 0 && layer < view.n_layers && across >= 0 &&
          across < view.n_across,
      "band ALE estimator lattice cell index is out of range");
  if (!view.single_block) {
    return view.cell_begin + layer * view.n_across + across;
  }
  return view.layer_axis_is_j ? across * view.nz + layer
                              : layer * view.nz + across;
}

std::size_t view_local_index(const BandLatticeView& view,
                             const int layer,
                             const int across) {
  return static_cast<std::size_t>(layer) *
             static_cast<std::size_t>(view.n_across) +
         static_cast<std::size_t>(across);
}

std::vector<int> view_node_line(
    const BandLatticeView& view,
    const int line_index,
    const std::vector<std::vector<int>>* multiblock_shell_rings = nullptr) {
  TENRYU_ASSERT(
      line_index >= 0 && line_index <= view.n_layers,
      "band ALE estimator lattice node line is out of range");
  if (!view.single_block) {
    TENRYU_ASSERT(
        multiblock_shell_rings != nullptr &&
            multiblock_shell_rings->size() ==
                static_cast<std::size_t>(view.n_layers) + 1U,
        "band ALE multiblock lattice view requires precomputed shell rings");
    return (*multiblock_shell_rings)[static_cast<std::size_t>(line_index)];
  }
  std::vector<int> nodes;
  nodes.reserve(static_cast<std::size_t>(view.n_across) + 1U);
  if (view.layer_axis_is_j) {
    for (int i = 0; i <= view.nr; ++i) {
      nodes.push_back(i * (view.nz + 1) + line_index);
    }
  } else {
    for (int j = 0; j <= view.nz; ++j) {
      nodes.push_back(line_index * (view.nz + 1) + j);
    }
  }
  return nodes;
}

double view_line_mean_coordinate(
    const BandLatticeView& view,
    const std::vector<int>& line,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  if (view.polar_geometry) {
    return ring_mean_radius(line, node_r, node_z);
  }
  TENRYU_ASSERT(
      !line.empty(),
      "band ALE estimator lattice node line must be non-empty");
  double coordinate_sum = 0.0;
  for (const int node : line) {
    TENRYU_ASSERT(
        node >= 0 && static_cast<std::size_t>(node) < node_r.size() &&
            static_cast<std::size_t>(node) < node_z.size(),
        "band ALE estimator lattice node id is out of range");
    coordinate_sum += view.layer_axis_is_j
                          ? node_z[static_cast<std::size_t>(node)]
                          : node_r[static_cast<std::size_t>(node)];
  }
  return coordinate_sum / static_cast<double>(line.size());
}

BandState make_shell_band(const core::State& state,
                          const std::vector<double>& initial_r,
                          const std::vector<double>& initial_z) {
  const auto& topology = *state.mesh.topo.multiblock;
  const auto& block =
      mesh::mesh_topo_multiblock_polar_shell_block(topology);
  const int block_id =
      static_cast<int>(&block - topology.blocks.data());

  BandState band;
  band.kind = BandKind::Shell;
  band.block_id = block_id;
  band.name = "shell";
  band.shell_rings = structured_node_rings(topology, block);
  TENRYU_ASSERT(
      band.shell_rings.size() ==
          static_cast<std::size_t>(block.n_i_cells) + 1U,
      "band ALE shell requires complete structured node rings");

  std::vector<double> initial_mean_radius;
  initial_mean_radius.reserve(band.shell_rings.size());
  for (const auto& ring : band.shell_rings) {
    initial_mean_radius.push_back(
        ring_mean_radius(ring, initial_r, initial_z));
  }
  TENRYU_ASSERT(
      initial_mean_radius.size() >= 2U &&
          initial_mean_radius.back() > initial_mean_radius.front(),
      "band ALE shell initial radial span must be positive");
  band.shell_dr0 =
      (initial_mean_radius.back() - initial_mean_radius.front()) /
      static_cast<double>(block.n_i_cells);
  TENRYU_ASSERT(
      std::isfinite(band.shell_dr0) && band.shell_dr0 > 0.0,
      "band ALE shell initial radial spacing must be finite and positive");

  band.shell_initial_area.reserve(
      static_cast<std::size_t>(block.cell_count));
  for (int local_cell = 0; local_cell < block.cell_count; ++local_cell) {
    const double area = state.mesh.cell_area[
        static_cast<std::size_t>(block.cell_begin + local_cell)];
    TENRYU_ASSERT(
        std::isfinite(area) && area > 0.0,
        "band ALE shell initial planar area must be finite and positive");
    band.shell_initial_area.push_back(area);
  }
  return band;
}

BandState make_estimator_band(const core::State& state,
                              const core::Config& cfg,
                              const std::vector<double>& initial_r,
                              const std::vector<double>& initial_z) {
  if (state.mesh.topo.multiblock.has_value()) {
    BandState band = make_shell_band(state, initial_r, initial_z);
    band.kind = BandKind::Estimator;
    band.name = "estimator1";
    return band;
  }

  const BandLatticeView view =
      make_single_block_band_view(state, cfg);
  BandState band;
  band.kind = BandKind::Estimator;
  band.block_id = -1;
  band.name = "estimator1";
  band.estimator_polar_geometry = view.polar_geometry;
  band.estimator_layer_axis_is_j = view.layer_axis_is_j;
  band.shell_rings.reserve(
      static_cast<std::size_t>(view.n_layers) + 1U);
  for (int line = 0; line <= view.n_layers; ++line) {
    band.shell_rings.push_back(view_node_line(view, line));
  }

  const double first_coordinate = view_line_mean_coordinate(
      view, band.shell_rings.front(), initial_r, initial_z);
  const double last_coordinate = view_line_mean_coordinate(
      view, band.shell_rings.back(), initial_r, initial_z);
  TENRYU_ASSERT(
      std::isfinite(first_coordinate) && std::isfinite(last_coordinate) &&
          last_coordinate > first_coordinate,
      "band ALE estimator initial layer-axis span must be positive");
  band.shell_dr0 =
      (last_coordinate - first_coordinate) /
      static_cast<double>(view.n_layers);
  TENRYU_ASSERT(
      std::isfinite(band.shell_dr0) && band.shell_dr0 > 0.0,
      "band ALE estimator initial layer spacing must be finite and positive");

  band.shell_initial_area.reserve(
      static_cast<std::size_t>(view.n_layers) *
      static_cast<std::size_t>(view.n_across));
  for (int layer = 0; layer < view.n_layers; ++layer) {
    for (int across = 0; across < view.n_across; ++across) {
      const double area = state.mesh.cell_area[
          static_cast<std::size_t>(view_cell(view, layer, across))];
      TENRYU_ASSERT(
          std::isfinite(area) && area > 0.0,
          "band ALE estimator initial planar area must be finite and positive");
      band.shell_initial_area.push_back(area);
    }
  }
  return band;
}

BandState make_closure_catchment_band(const core::State& state) {
  const auto& topology = *state.mesh.topo.multiblock;
  BandState band;
  band.kind = BandKind::ClosureCatchment;
  band.name = "closure_catchment";
  int core_count = 0;
  int bridge_count = 0;
  for (int block_id = 0;
       block_id < static_cast<int>(topology.blocks.size());
       ++block_id) {
    const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
    if (block.role == mesh::BlockRole::CENTRAL_CORE) {
      ++core_count;
      band.closure_core_block_id = block_id;
    } else if (block.role == mesh::BlockRole::BRIDGE) {
      ++bridge_count;
      band.closure_bridge_block_id = block_id;
    }
  }
  if (bridge_count != 1) {
    core::log_info(
        "[closure-catchment] no-op: topology does not have a unique "
        "BRIDGE block");
    return band;
  }
  if (core_count != 1) {
    core::log_info(
        "[closure-catchment] no-op: topology does not have a unique "
        "CENTRAL_CORE block");
    return band;
  }
  const auto& core_block = topology.blocks[
      static_cast<std::size_t>(band.closure_core_block_id)];
  const auto& bridge_block = topology.blocks[
      static_cast<std::size_t>(band.closure_bridge_block_id)];
  if (core_block.n_j_cells < 8 ||
      core_block.n_j_cells != bridge_block.n_j_cells) {
    core::log_info(
        "[closure-catchment] no-op: CENTRAL_CORE and BRIDGE must have the "
        "same number of angular columns (at least 8)");
    return band;
  }
  band.closure_columns = core_block.n_j_cells;
  band.closure_topology_supported = true;
  return band;
}

BandState make_pole_theta_band(const core::State& state) {
  const auto& topology = *state.mesh.topo.multiblock;
  BandState band;
  band.kind = BandKind::PoleTheta;
  band.name = "pole_theta";

  int shell_block_id = -1;
  int tier_block_id = -1;
  int bridge_block_id = -1;
  int shell_count = 0;
  int tier_count = 0;
  int bridge_count = 0;
  for (int block_id = 0;
       block_id < static_cast<int>(topology.blocks.size());
       ++block_id) {
    const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
    if (block.role == mesh::BlockRole::POLAR_SHELL) {
      ++shell_count;
      shell_block_id = block_id;
    } else if (block.role == mesh::BlockRole::POLAR_TIER) {
      ++tier_count;
      tier_block_id = block_id;
    } else if (block.role == mesh::BlockRole::BRIDGE) {
      ++bridge_count;
      bridge_block_id = block_id;
    }
  }

  if (shell_count != 1 || tier_count > 1 || bridge_count != 1) {
    core::log_info(
        "[pole-theta] no-op: topology lacks unique POLAR_SHELL/BRIDGE");
    return band;
  }

  const int ntheta = topology.blocks[
      static_cast<std::size_t>(shell_block_id)].n_j_cells;
  if (ntheta <= 0 ||
      topology.blocks[
          static_cast<std::size_t>(bridge_block_id)].n_j_cells != ntheta ||
      (tier_block_id >= 0 &&
       topology.blocks[
           static_cast<std::size_t>(tier_block_id)].n_j_cells != ntheta)) {
    core::log_info(
        "[pole-theta] no-op: topology lacks unique POLAR_SHELL/BRIDGE");
    return band;
  }

  band.pole_shell_block_id = shell_block_id;
  band.pole_tier_block_id = tier_block_id;
  band.pole_bridge_block_id = bridge_block_id;
  return band;
}

struct PoleThetaOwnedRow {
  int block_id = -1;
  int node_row = -1;  // within the block's owned node-row range
};

// Decompose a node id into (block, owned node row). Returns false when the
// node does not belong to a scanned block or the block's owned node count
// is not a multiple of (ntheta + 1).
bool pole_theta_owned_row_of_node(
    const mesh::MultiBlockTopology& topology,
    const std::vector<int>& scan_block_ids,
    const int node,
    const int ntheta,
    PoleThetaOwnedRow* out) {
  for (const int block_id : scan_block_ids) {
    const auto& block =
        topology.blocks[static_cast<std::size_t>(block_id)];
    if (node >= block.owned_node_begin &&
        node < block.owned_node_begin + block.owned_node_count &&
        block.owned_node_count % (ntheta + 1) == 0) {
      out->block_id = block_id;
      out->node_row =
          (node - block.owned_node_begin) / (ntheta + 1);
      return true;
    }
  }
  return false;
}

struct PoleThetaCompressionScan {
  std::vector<double> eighth_sum;
  std::vector<double> weight_sum;
};

struct PoleThetaHardAuditCell {
  int cell = -1;
  int block_id = -1;
  int row = -1;
  int col = -1;
  std::array<int, 4> quad = {{-1, -1, -1, -1}};
  double weight = 0.0;
  double rho_ref = 0.0;
  double h_theta_star = 0.0;
  double q_pre = 0.0;
  double min_altitude_pre = 0.0;
  double area = 0.0;
};

double pole_theta_jaccard_distance(
    const std::set<std::pair<int, int>>& lhs,
    const std::set<std::pair<int, int>>& rhs) {
  std::size_t intersection_size = 0U;
  auto left = lhs.begin();
  auto right = rhs.begin();
  while (left != lhs.end() && right != rhs.end()) {
    if (*left < *right) {
      ++left;
    } else if (*right < *left) {
      ++right;
    } else {
      ++intersection_size;
      ++left;
      ++right;
    }
  }
  const std::size_t union_size =
      lhs.size() + rhs.size() - intersection_size;
  if (union_size == 0U) {
    return 0.0;
  }
  return 1.0 - static_cast<double>(intersection_size) /
                   static_cast<double>(union_size);
}

void pole_theta_routine_residual(
    const core::State& state,
    const core::Config& cfg,
    BandState& band,
    const int rank,
    const mesh::MultiBlockTopology& topology,
    const std::vector<int>& scan_block_ids,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::map<int, PoleThetaCompressionScan>& compression_scan,
    const bool curve_preserving_enabled,
    const int fit_order,
    const std::vector<int>& protected_modes) {
  if (!cfg.numerics.ale.band_ale.pole_theta_routine_enabled) {
    return;
  }
  if (state.t < band.pole_routine_next_eval_t) {
    return;
  }
  band.pole_routine_next_eval_t = state.t + 0.5e-12;

  const auto& pole_theta = cfg.numerics.ale.band_ale;
  const bool hold_enabled = pole_theta.pole_theta_shock_hold > 0.0;
  const int ntheta = topology.blocks[
      static_cast<std::size_t>(band.pole_shell_block_id)].n_j_cells;
  int rows_acted = 0;
  int gated_rows = 0;
  double max_E = 0.0;
  double max_a = 0.0;
  for (const int block_id : scan_block_ids) {
    const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
    if (block.owned_node_count % (ntheta + 1) != 0) {
      continue;
    }
    const int node_row_count = block.owned_node_count / (ntheta + 1);
    for (int node_row = 0; node_row < node_row_count; ++node_row) {
      std::vector<double> theta_row(
          static_cast<std::size_t>(ntheta + 1), 0.0);
      std::vector<double> s_row;
      if (curve_preserving_enabled) {
        s_row.assign(static_cast<std::size_t>(ntheta + 1), 0.0);
      }
      std::vector<int> nodes(static_cast<std::size_t>(ntheta + 1), -1);
      for (int spoke = 0; spoke <= ntheta; ++spoke) {
        const int node = block.owned_node_begin +
                         node_row * (ntheta + 1) + spoke;
        const auto node_index = static_cast<std::size_t>(node);
        nodes[static_cast<std::size_t>(spoke)] = node;
        theta_row[static_cast<std::size_t>(spoke)] = std::atan2(
            std::max(node_r[node_index], 0.0), node_z[node_index]);
        if (curve_preserving_enabled) {
          s_row[static_cast<std::size_t>(spoke)] =
              std::hypot(node_r[node_index], node_z[node_index]);
        }
      }

      std::vector<double> dtheta;
      const double E = pole_theta_row_defect(
          theta_row,
          pole_theta.pole_theta_phys_lp,
          pole_theta.pole_theta_phys_lc,
          &dtheta);
      if (!std::isfinite(E)) {
        continue;
      }
      max_E = std::max(max_E, E);
      const double a_low = band_activation_up(
          E / pole_theta.pole_theta_noise_floor, 2.0, 4.0);
      const double activation = pole_theta_routine_activation(
          E,
          pole_theta.pole_theta_noise_floor,
          pole_theta.pole_theta_noise_ceiling);
      if (a_low > 0.0 && activation == 0.0) {
        ++gated_rows;
      }
      max_a = std::max(max_a, activation);
      if (activation <= 0.0) {
        continue;
      }

      double hold_activation = 0.0;
      if (hold_enabled) {
        const auto scan_it = compression_scan.find(block_id);
        if (scan_it != compression_scan.end()) {
          const std::size_t row_index =
              static_cast<std::size_t>(node_row);
          const double eighth_sum = scan_it->second.eighth_sum[row_index];
          const double weight_sum = scan_it->second.weight_sum[row_index];
          if (weight_sum > 0.0) {
            const double row_M =
                std::pow(eighth_sum / weight_sum, 1.0 / 8.0);
            hold_activation = band_activation_up(
                row_M,
                pole_theta.pole_theta_shock_hold - 0.05,
                pole_theta.pole_theta_shock_hold + 0.10);
          }
        }
      }

      const double scale = activation * (1.0 - hold_activation);
      bool row_acted = false;
      for (double& delta : dtheta) {
        delta *= scale;
        row_acted = row_acted || delta != 0.0;
      }
      if (!row_acted) {
        continue;
      }

      PoleThetaRowCurve curve;
      const bool curve_ok =
          curve_preserving_enabled &&
          pole_theta_build_row_curve(
              theta_row, s_row, fit_order, protected_modes, &curve);
      double s_min_row = 0.0;
      double s_max_row = 0.0;
      if (curve_ok) {
        const auto row_envelope =
            std::minmax_element(s_row.begin(), s_row.end());
        s_min_row = *row_envelope.first;
        s_max_row = *row_envelope.second;
      }
      for (int spoke = 0; spoke <= ntheta; ++spoke) {
        const double delta = dtheta[static_cast<std::size_t>(spoke)];
        const auto node_index = static_cast<std::size_t>(
            nodes[static_cast<std::size_t>(spoke)]);
        if (!state.central_pseudo_core.boundary_node_mask.empty() &&
            node_index <
                state.central_pseudo_core.boundary_node_mask.size() &&
            state.central_pseudo_core.boundary_node_mask[node_index] != 0U) {
          continue;
        }
        g_runtime->pole_theta_dtheta[node_index] += delta;
        if (curve_ok && delta != 0.0) {
          const double s_target = pole_theta_eval_row_curve(
              curve,
              theta_row[static_cast<std::size_t>(spoke)] + delta);
          if (pole_theta_s_target_admissible(
                  s_target, s_min_row, s_max_row)) {
            g_runtime->pole_theta_target_s[node_index] = s_target;
          }
        }
      }
      ++rows_acted;
      band.pole_routine_acted = true;
    }
  }

  if (rows_acted > 0) {
    if (band.pole_routine_log_countdown == 0) {
      if (rank == 0) {
        std::ostringstream log;
        log << "[pole-theta] routine rows=" << rows_acted
            << " gated_rows=" << gated_rows
            << " maxE=" << std::scientific << std::setprecision(6) << max_E
            << " maxA=" << max_a;
        core::log_info(log.str());
      }
      band.pole_routine_log_countdown = 20;
    } else {
      --band.pole_routine_log_countdown;
    }
  }
}

void evaluate_pole_theta(core::State& state,
                         const core::Config& cfg,
                         BandState& band,
                         const int rank) {
  if (band.pole_shell_block_id < 0 ||
      band.pole_bridge_block_id < 0) {
    return;
  }

  band.pole_north.fired_this_step = false;
  band.pole_south.fired_this_step = false;
  band.pole_routine_acted = false;
  if (g_runtime->pole_theta_consumed) {
    g_runtime->pole_theta_dtheta.clear();
    g_runtime->pole_theta_target_s.clear();
  }

  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  const bool hold_enabled =
      cfg.numerics.ale.band_ale.pole_theta_shock_hold > 0.0;
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> cs_host;
  std::vector<double> rho_host;
  std::vector<double> mass_host;
  std::vector<double> pressure_e_host;
  std::vector<double> pressure_i_host;
  state.cs.copy_to_host(cs_host);
  state.rho.copy_to_host(rho_host);
  state.mass.copy_to_host(mass_host);
  state.Pe.copy_to_host(pressure_e_host);
  state.Pi.copy_to_host(pressure_i_host);
  if (hold_enabled) {
    state.v_r.copy_to_host(v_r);
    state.v_z.copy_to_host(v_z);
  }

  const auto& topology = *state.mesh.topo.multiblock;
  const auto& shell = topology.blocks[
      static_cast<std::size_t>(band.pole_shell_block_id)];
  const int ntheta = shell.n_j_cells;
  const double dtheta_star = CUDART_PI / static_cast<double>(ntheta);
  std::vector<double> H(
      static_cast<std::size_t>(ntheta),
      std::numeric_limits<double>::infinity());
  std::vector<double> kappa(static_cast<std::size_t>(ntheta), 0.0);
  std::vector<double> q_eighth_sum(
      static_cast<std::size_t>(ntheta), 0.0);
  std::vector<int> q_count(static_cast<std::size_t>(ntheta), 0);
  std::array<std::vector<int>, 2> marked_cell_nodes;
  std::array<bool, 2> pole_row_shock = {{false, false}};
  std::vector<PoleThetaHardAuditCell> hard_audit_scan;

  const std::array<int, 3> block_ids = {{
      band.pole_shell_block_id,
      band.pole_tier_block_id,
      band.pole_bridge_block_id,
  }};
  std::vector<int> scan_block_ids;
  std::map<int, PoleThetaCompressionScan> compression_scan;
  for (const int block_id : block_ids) {
    if (block_id < 0) {
      continue;
    }
    scan_block_ids.push_back(block_id);
    const auto& block =
        topology.blocks[static_cast<std::size_t>(block_id)];
    std::vector<int> attribution_block_ids;
    if (hold_enabled) {
      attribution_block_ids.push_back(block_id);
      PoleThetaCompressionScan scan;
      const std::size_t scan_size =
          block.owned_node_count % (ntheta + 1) == 0
              ? static_cast<std::size_t>(
                    block.owned_node_count / (ntheta + 1))
              : 0U;
      scan.eighth_sum.assign(scan_size, 0.0);
      scan.weight_sum.assign(scan_size, 0.0);
      compression_scan.emplace(block_id, std::move(scan));
    }
    for (int row = 0; row < block.n_i_cells; ++row) {
      double ring_mass_sum = 0.0;
      double ring_rho_weighted_sum = 0.0;
      double ring_z_weighted_sum = 0.0;
      double ring_pv_weighted_sum = 0.0;
      int ring_cell_count = 0;
      for (int col = 0; col < ntheta; ++col) {
        const int cell = block.cell_begin + row * ntheta + col;
        const auto cell_index = static_cast<std::size_t>(cell);
        if (!state.central_pseudo_core.member_mask.empty() &&
            cell_index < state.central_pseudo_core.member_mask.size() &&
            state.central_pseudo_core.member_mask[cell_index] != 0U) {
          continue;
        }
        const double cell_mass = mass_host[cell_index];
        const double cell_density = rho_host[cell_index];
        const double cell_pressure =
            pressure_e_host[cell_index] + pressure_i_host[cell_index];
        const double cell_volume = state.mesh.cell_area[cell_index];
        ring_mass_sum += cell_mass;
        ring_rho_weighted_sum += cell_mass * cell_density;
        ring_z_weighted_sum +=
            cell_mass * cell_density * cs_host[cell_index];
        ring_pv_weighted_sum +=
            cell_mass * cell_pressure * cell_volume;
        ++ring_cell_count;
      }
      const bool have_ring_reference =
          ring_cell_count > 0 && ring_mass_sum > 0.0;
      const double rho_ref =
          have_ring_reference
              ? ring_rho_weighted_sum / ring_mass_sum
              : 0.0;
      const double z_ref =
          have_ring_reference
              ? ring_z_weighted_sum / ring_mass_sum
              : 0.0;
      const double pv_ref =
          have_ring_reference
              ? ring_pv_weighted_sum / ring_mass_sum
              : 0.0;
      const double m_ref =
          ring_cell_count > 0
              ? ring_mass_sum / static_cast<double>(ring_cell_count)
              : 0.0;
      for (int col = 0; col < ntheta; ++col) {
        const int cell =
            block.cell_begin + row * ntheta + col;
        const auto cell_index = static_cast<std::size_t>(cell);
        const bool pool_member =
            !state.central_pseudo_core.member_mask.empty() &&
            cell_index < state.central_pseudo_core.member_mask.size() &&
            state.central_pseudo_core.member_mask[cell_index] != 0U;
        if ((cell_index <
                 state.central_pseudo_core.inactive_member_mask.size() &&
             state.central_pseudo_core
                     .inactive_member_mask[cell_index] != 0U) ||
            (cell_index <
                 state.pole_angular_derefine.inactive_member_mask.size() &&
             state.pole_angular_derefine
                     .inactive_member_mask[cell_index] != 0U) ||
            (!state.hydro_active.empty() &&
             cell_index < state.hydro_active.size() &&
             state.hydro_active[cell_index] == 0)) {
          continue;
        }
        const int begin = topology.cell_node_csr_offsets[cell_index];
        int q[4];
        bool quad_ok = true;
        for (int k = 0; k < 4; ++k) {
          q[k] = topology.cell_node_csr_indices[
              static_cast<std::size_t>(begin + k)];
          if (q[k] < 0 ||
              q[k] >= state.mesh.topo.n_nodes) {
            quad_ok = false;
          }
        }
        if (!quad_ok) {
          continue;
        }
        const auto edge_evidence = [&](const int a, const int b) {
          const int diff = std::abs(q[a] - q[b]);
          if (diff == 1) {
            return +1;  // angular
          }
          if (diff == ntheta + 1) {
            return -1;  // radial
          }
          return 0;  // seam or irregular: no evidence
        };
        int evidence_a =
            edge_evidence(0, 1) + edge_evidence(2, 3);
        int evidence_b =
            edge_evidence(1, 2) + edge_evidence(3, 0);
        // Positive evidence = pair is angular. The two pairs must not
        // claim the same type; a quad always has at least one definite
        // edge (each cell has an angular edge on an owned node row).
        bool pair_a_angular;
        if (evidence_a > 0 || evidence_b < 0) {
          pair_a_angular = true;
        } else if (evidence_b > 0 || evidence_a < 0) {
          pair_a_angular = false;
        } else {
          continue;  // no topological evidence; skip the cell
        }
        int quad[4];
        if (pair_a_angular) {
          // CSR order is angular-first: (a0, a1, b1, b0) -> reorder.
          quad[0] = q[0];
          quad[1] = q[3];
          quad[2] = q[2];
          quad[3] = q[1];
        } else {
          quad[0] = q[0];
          quad[1] = q[1];
          quad[2] = q[2];
          quad[3] = q[3];
        }
        if (hold_enabled) {
          double ir = 0.5 * (node_r[quad[0]] + node_r[quad[3]]);
          double iz = 0.5 * (node_z[quad[0]] + node_z[quad[3]]);
          double orr = 0.5 * (node_r[quad[1]] + node_r[quad[2]]);
          double oz = 0.5 * (node_z[quad[1]] + node_z[quad[2]]);
          double ivr = 0.5 * (v_r[quad[0]] + v_r[quad[3]]);
          double ivz = 0.5 * (v_z[quad[0]] + v_z[quad[3]]);
          double ovr = 0.5 * (v_r[quad[1]] + v_r[quad[2]]);
          double ovz = 0.5 * (v_z[quad[1]] + v_z[quad[2]]);
          if (std::hypot(ir, iz) > std::hypot(orr, oz)) {
            std::swap(ir, orr);
            std::swap(iz, oz);
            std::swap(ivr, ovr);
            std::swap(ivz, ovz);
          }
          double sr = orr - ir;
          double sz = oz - iz;
          const double sl = std::hypot(sr, sz);
          if (sl > 0.0) {
            sr /= sl;
            sz /= sl;
            const double u_in = ivr * sr + ivz * sz;
            const double u_out = ovr * sr + ovz * sz;
            const double cs_cell =
                cell_index < cs_host.size() ? cs_host[cell_index] : 0.0;
            const double compression_mach = std::max(
                0.0,
                (u_in - u_out) /
                    std::max(cs_cell, std::numeric_limits<double>::min()));
            if (cs_cell > 0.0 &&
                compression_mach >
                    cfg.numerics.ale.band_ale.pole_theta_shock_hold) {
              pole_row_shock[static_cast<std::size_t>(
                  col < ntheta / 2 ? 0 : 1)] = true;
            }
            double centroid_r = 0.0;
            double centroid_z = 0.0;
            for (const int node : quad) {
              centroid_r += node_r[static_cast<std::size_t>(node)];
              centroid_z += node_z[static_cast<std::size_t>(node)];
            }
            centroid_r *= 0.25;
            centroid_z *= 0.25;
            const double theta_center =
                std::atan2(centroid_r, centroid_z);
            const double weight =
                std::sin(theta_center) * dtheta_star;
            const double contribution =
                weight * std::pow(compression_mach, 8.0);
            std::array<int, 2> owned_rows = {{-1, -1}};
            int owned_row_count = 0;
            for (const int node : quad) {
              PoleThetaOwnedRow owned_row;
              if (!pole_theta_owned_row_of_node(
                      topology,
                      attribution_block_ids,
                      node,
                      ntheta,
                      &owned_row)) {
                continue;
              }
              bool duplicate = false;
              for (int index = 0; index < owned_row_count; ++index) {
                duplicate = duplicate ||
                    owned_rows[static_cast<std::size_t>(index)] ==
                        owned_row.node_row;
              }
              if (!duplicate && owned_row_count < 2) {
                owned_rows[static_cast<std::size_t>(owned_row_count)] =
                    owned_row.node_row;
                ++owned_row_count;
              }
            }
            auto& scan = compression_scan.at(block_id);
            for (int index = 0; index < owned_row_count; ++index) {
              const std::size_t owned_row = static_cast<std::size_t>(
                  owned_rows[static_cast<std::size_t>(index)]);
              scan.eighth_sum[owned_row] += contribution;
              scan.weight_sum[owned_row] += weight;
            }
          }
        }
        double rq[4];
        double zq[4];
        double s_bar = 0.0;
        for (int corner = 0; corner < 4; ++corner) {
          const int node = quad[corner];
          const auto node_index = static_cast<std::size_t>(node);
          rq[corner] = node_r[node_index];
          zq[corner] = node_z[node_index];
          s_bar += std::hypot(rq[corner], zq[corner]);
        }
        s_bar *= 0.25;
        const double h_theta_star_cell = s_bar * dtheta_star;
        if (!(h_theta_star_cell > 0.0)) {
          continue;
        }

        const PoleThetaCellMetrics metrics =
            pole_theta_cell_metrics(rq, zq, h_theta_star_cell);
        const double H_cell =
            metrics.h_theta_min / h_theta_star_cell;
        double significance_weight = 1.0;
        if (pool_member) {
          significance_weight = 0.0;
        } else if (have_ring_reference) {
          // Fixed ruling constants: two-decade logarithmic transitions.
          const double w_rho = band_significance_channel(
              rho_host[cell_index], rho_ref, 1.0e-5, 1.0e-3);
          const double w_m = band_significance_channel(
              mass_host[cell_index], m_ref, 1.0e-5, 1.0e-3);
          const double w_z = band_significance_channel(
              rho_host[cell_index] * cs_host[cell_index],
              z_ref,
              1.0e-4,
              1.0e-2);
          const double cell_pv =
              (pressure_e_host[cell_index] +
               pressure_i_host[cell_index]) *
              state.mesh.cell_area[cell_index];
          const double w_pv = band_significance_channel(
              cell_pv, pv_ref, 1.0e-4, 1.0e-2);
          significance_weight = band_significance_weight(
              w_rho, w_m, w_z, w_pv);
        }
        const double cell_excess = pole_theta_cell_excess(
            H_cell,
            metrics.kappa_max,
            cfg.numerics.ale.band_ale.pole_theta_h_hard,
            cfg.numerics.ale.band_ale.pole_theta_kappa_hard);
        PoleThetaHardAuditCell audit_cell;
        audit_cell.cell = cell;
        audit_cell.block_id = block_id;
        audit_cell.row = row;
        audit_cell.col = col;
        for (int corner = 0; corner < 4; ++corner) {
          audit_cell.quad[static_cast<std::size_t>(corner)] = quad[corner];
        }
        audit_cell.weight = significance_weight;
        audit_cell.rho_ref = rho_ref;
        audit_cell.h_theta_star = h_theta_star_cell;
        audit_cell.q_pre = significance_weight * cell_excess;
        audit_cell.min_altitude_pre = metrics.min_altitude_abs;
        audit_cell.area = state.mesh.cell_area[cell_index];
        hard_audit_scan.push_back(audit_cell);
        q_eighth_sum[static_cast<std::size_t>(col)] +=
            std::pow(significance_weight * cell_excess, 8.0);
        ++q_count[static_cast<std::size_t>(col)];
        H[static_cast<std::size_t>(col)] = std::min(
            H[static_cast<std::size_t>(col)], H_cell);
        kappa[static_cast<std::size_t>(col)] = std::max(
            kappa[static_cast<std::size_t>(col)], metrics.kappa_max);
        if (H_cell < cfg.numerics.ale.band_ale.pole_theta_h_release ||
            metrics.kappa_max > 2.0) {
          auto& pole_marked_cell_nodes = marked_cell_nodes[
              static_cast<std::size_t>(col < ntheta / 2 ? 0 : 1)];
          for (int k = 0; k < 4; ++k) {
            pole_marked_cell_nodes.push_back(q[k]);
          }
        }
      }
    }
  }

  std::vector<double> q(static_cast<std::size_t>(ntheta), 0.0);
  for (int col = 0; col < ntheta; ++col) {
    const std::size_t column = static_cast<std::size_t>(col);
    if (q_count[column] > 0) {
      q[column] = std::pow(
          q_eighth_sum[column] /
              static_cast<double>(q_count[column]),
          1.0 / 8.0);
    }
  }

  const bool curve_preserving_enabled =
      cfg.numerics.ale.band_ale.pole_theta_curve_preserving;
  std::vector<int> protected_modes;
  if (curve_preserving_enabled) {
    const std::string& protected_modes_text =
        cfg.numerics.ale.band_ale.pole_theta_protected_modes;
    std::istringstream stream(protected_modes_text);
    std::string token;
    while (std::getline(stream, token, ',')) {
      protected_modes.push_back(std::stoi(token));
    }
    if (std::find(protected_modes.begin(), protected_modes.end(), 0) ==
        protected_modes.end()) {
      protected_modes.insert(protected_modes.begin(), 0);
    }
    if (std::find(protected_modes.begin(), protected_modes.end(), 1) ==
        protected_modes.end()) {
      protected_modes.insert(protected_modes.begin(), 1);
    }
    std::sort(protected_modes.begin(), protected_modes.end());
    protected_modes.erase(
        std::unique(protected_modes.begin(), protected_modes.end()),
        protected_modes.end());
  }
  const int fit_order =
      curve_preserving_enabled
          ? std::min(
                cfg.numerics.ale.band_ale.pole_theta_fit_order,
                ntheta / 4)
          : 0;

  pole_theta_routine_residual(
      state,
      cfg,
      band,
      rank,
      topology,
      scan_block_ids,
      node_r,
      node_z,
      compression_scan,
      curve_preserving_enabled,
      fit_order,
      protected_modes);

  const int half_size = ntheta / 2;
  std::array<std::vector<double>, 2> H_half;
  std::array<std::vector<double>, 2> kappa_half;
  std::array<std::vector<double>, 2> q_half;
  for (int pole_index = 0; pole_index < 2; ++pole_index) {
    H_half[static_cast<std::size_t>(pole_index)].assign(
        static_cast<std::size_t>(half_size),
        std::numeric_limits<double>::infinity());
    kappa_half[static_cast<std::size_t>(pole_index)].assign(
        static_cast<std::size_t>(half_size), 0.0);
    q_half[static_cast<std::size_t>(pole_index)].assign(
        static_cast<std::size_t>(half_size), 0.0);
  }
  std::array<double, 2> status_H = {{
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
  }};
  std::array<double, 2> status_kappa = {{0.0, 0.0}};
  std::array<bool, 2> have_finite_pole = {{false, false}};
  const std::array<bool, 2> shock_hold = {{
      pole_row_shock[0],
      pole_row_shock[1],
  }};
  for (int pole_index = 0; pole_index < 2; ++pole_index) {
    const bool from_north = pole_index == 0;
    auto& pole = from_north ? band.pole_north : band.pole_south;
    if (curve_preserving_enabled && pole.last_fire_row_count > 0) {
      double max_rel_dev = 0.0;
      const std::size_t modes_per_row = protected_modes.size();
      const int audit_row_count = std::min(
          pole.last_fire_row_count,
          static_cast<int>(pole.last_fire_row_ids.size()));
      for (int row_index = 0; row_index < audit_row_count; ++row_index) {
        const auto& row_id = pole.last_fire_row_ids[
            static_cast<std::size_t>(row_index)];
        const auto& block = topology.blocks[
            static_cast<std::size_t>(row_id.first)];
        std::vector<double> theta_row(
            static_cast<std::size_t>(ntheta + 1), 0.0);
        std::vector<double> s_row(
            static_cast<std::size_t>(ntheta + 1), 0.0);
        for (int spoke = 0; spoke <= ntheta; ++spoke) {
          const int node = block.owned_node_begin +
                           row_id.second * (ntheta + 1) + spoke;
          const auto node_index = static_cast<std::size_t>(node);
          theta_row[static_cast<std::size_t>(spoke)] = std::atan2(
              std::max(node_r[node_index], 0.0), node_z[node_index]);
          s_row[static_cast<std::size_t>(spoke)] = std::hypot(
              node_r[node_index], node_z[node_index]);
        }
        std::vector<double> current_a;
        if (!pole_theta_legendre_fit(
                theta_row, s_row, fit_order, &current_a)) {
          continue;
        }
        const std::size_t row_offset =
            static_cast<std::size_t>(row_index) * modes_per_row;
        if (row_offset + modes_per_row > pole.last_fire_row_a.size() ||
            !std::isfinite(pole.last_fire_row_a[row_offset])) {
          continue;
        }
        const double denominator = std::max(
            std::abs(pole.last_fire_row_a[row_offset]),
            std::numeric_limits<double>::min());
        for (std::size_t mode_index = 0;
             mode_index < modes_per_row;
             ++mode_index) {
          const int mode = protected_modes[mode_index];
          const double pre_a =
              pole.last_fire_row_a[row_offset + mode_index];
          if (mode < 2 || mode > fit_order || !std::isfinite(pre_a)) {
            continue;
          }
          max_rel_dev = std::max(
              max_rel_dev,
              std::abs(current_a[static_cast<std::size_t>(mode)] - pre_a) /
                  denominator);
        }
      }
      if (rank == 0) {
        std::ostringstream log;
        log << "[pole-theta] mode audit pole="
            << (from_north ? "N" : "S")
            << " max_rel_dev=" << std::scientific
            << std::setprecision(6) << max_rel_dev;
        core::log_info(log.str());
      }
      pole.last_fire_row_a.clear();
      pole.last_fire_row_ids.clear();
      pole.last_fire_row_count = 0;
    }

    for (int col = 0; col < half_size; ++col) {
      const int global_col =
          from_north ? col : ntheta - 1 - col;
      H_half[static_cast<std::size_t>(pole_index)]
            [static_cast<std::size_t>(col)] =
          H[static_cast<std::size_t>(global_col)];
      kappa_half[static_cast<std::size_t>(pole_index)]
                [static_cast<std::size_t>(col)] =
          kappa[static_cast<std::size_t>(global_col)];
      q_half[static_cast<std::size_t>(pole_index)]
            [static_cast<std::size_t>(col)] =
          q[static_cast<std::size_t>(global_col)];
    }

    double H_min = std::numeric_limits<double>::infinity();
    double kappa_max = 0.0;
    for (int col = 0; col < half_size; ++col) {
      const double H_column =
          H_half[static_cast<std::size_t>(pole_index)]
                [static_cast<std::size_t>(col)];
      if (!std::isfinite(H_column)) {
        continue;
      }
      have_finite_pole[static_cast<std::size_t>(pole_index)] = true;
      H_min = std::min(H_min, H_column);
      kappa_max = std::max(
          kappa_max,
          kappa_half[static_cast<std::size_t>(pole_index)]
                    [static_cast<std::size_t>(col)]);
    }
    if (!have_finite_pole[static_cast<std::size_t>(pole_index)]) {
      continue;
    }
    status_H[static_cast<std::size_t>(pole_index)] = H_min;
    status_kappa[static_cast<std::size_t>(pole_index)] = kappa_max;

    if (shock_hold[static_cast<std::size_t>(pole_index)]) {
      if (rank == 0 && (state.step + 1) % 500 == 0) {
        core::log_info(std::string("[pole-theta] shock hold pole=") +
                       (from_north ? "N" : "S"));
      }
      continue;
    }

    if (H_min < cfg.numerics.ale.band_ale.pole_theta_h_arm ||
        kappa_max > cfg.numerics.ale.band_ale.pole_theta_kappa_arm) {
      pole.armed = true;
    } else if (
        H_min > cfg.numerics.ale.band_ale.pole_theta_h_release &&
        kappa_max < 2.0) {
      pole.armed = false;
    }
  }

  double coupled_H_min = std::numeric_limits<double>::infinity();
  double coupled_kappa_max = 0.0;
  for (int pole_index = 0; pole_index < 2; ++pole_index) {
    if (shock_hold[static_cast<std::size_t>(pole_index)] ||
        !have_finite_pole[static_cast<std::size_t>(pole_index)]) {
      continue;
    }
    coupled_H_min = std::min(
        coupled_H_min, status_H[static_cast<std::size_t>(pole_index)]);
    coupled_kappa_max = std::max(
        coupled_kappa_max,
        status_kappa[static_cast<std::size_t>(pole_index)]);
  }
  std::vector<double> request_col(
      static_cast<std::size_t>(half_size), 0.0);
  double coupled_q_max = 0.0;
  for (int col = 0; col < half_size; ++col) {
    const std::size_t column = static_cast<std::size_t>(col);
    for (int pole_index = 0; pole_index < 2; ++pole_index) {
      if (!shock_hold[static_cast<std::size_t>(pole_index)]) {
        request_col[column] = std::max(
            request_col[column],
            q_half[static_cast<std::size_t>(pole_index)][column]);
      }
    }
    coupled_q_max = std::max(coupled_q_max, request_col[column]);
  }
  const bool hard = coupled_q_max > 0.0;
  if (hard) {
    std::set<std::pair<int, int>> owned_rows;
    for (int pole_index = 0; pole_index < 2; ++pole_index) {
      if (shock_hold[static_cast<std::size_t>(pole_index)]) {
        continue;
      }
      for (const int node :
           marked_cell_nodes[static_cast<std::size_t>(pole_index)]) {
        PoleThetaOwnedRow owned_row;
        if (pole_theta_owned_row_of_node(
                topology,
                scan_block_ids,
                node,
                ntheta,
                &owned_row)) {
          owned_rows.emplace(owned_row.block_id, owned_row.node_row);
        }
      }
    }

    const double cooldown_base =
        cfg.numerics.ale.band_ale.pole_theta_cooldown_base_s;
    const bool futility_control_enabled = cooldown_base > 0.0;
    double authority = 1.0;
    if (futility_control_enabled) {
      const double novelty_distance = pole_theta_jaccard_distance(
          owned_rows, band.pole_hard_signature);
      const double novelty = band_activation_up(
          novelty_distance, 0.20, 0.40);
      if (novelty > 0.0) {
        // Novelty acts only on accumulated futility; it never bypasses the
        // physical-time cooldown between committed fires.
        band.pole_hard_debt *= 1.0 - novelty;
      }

      if (band.pole_hard_emergency_used &&
          pole_theta_jaccard_distance(
              owned_rows, band.pole_hard_emergency_signature) > 0.40) {
        band.pole_hard_emergency_used = false;
      }
      // H_min < 0.05 is the v1 imminent-collapse proxy. A predicted
      // inversion horizon is the eventual emergency criterion.
      const bool emergency =
          coupled_H_min < 0.05 && !band.pole_hard_emergency_used;
      if (emergency) {
        band.pole_hard_emergency_signature = owned_rows;
        band.pole_hard_emergency_used = true;
      }

      // Futility memory is 8 ps; debt doubles the cooldown per unit, with
      // the physical cooldown capped at 16 ps.
      const double tau_cd = std::min(
          cooldown_base * std::exp2(band.pole_hard_debt), 16.0e-12);
      const double G_cool = band_activation_up(
          state.t - band.pole_hard_t_last, tau_cd, 1.1 * tau_cd);
      const double G_cap = pole_theta_capability(band.pole_hard_debt);
      authority = emergency ? 1.0 : G_cool * G_cap;
    }

    if (authority == 0.0) {
      ++band.pole_hard_refractory_suppressions;
      if (rank == 0 &&
          band.pole_hard_refractory_suppressions % 50U == 0U) {
        std::ostringstream log;
        log << "[pole-theta] refractory debt=" << std::scientific
            << std::setprecision(6) << band.pole_hard_debt
            << " dt_since=" << state.t - band.pole_hard_t_last;
        core::log_info(log.str());
      }
    } else {
      const int halo_columns =
          cfg.numerics.ale.band_ale.pole_theta_halo_columns;
      const PoleThetaSector sector = pole_theta_grow_sector_q(
          request_col, half_size, 0.0, halo_columns);
      if (sector.col_begin >= 0) {
        const PoleThetaSector core_sector = pole_theta_grow_sector_q(
            request_col, half_size, 0.0, 0);
        const int north_sector_begin = sector.col_begin;
        const int north_sector_end = sector.col_end;
        const int south_sector_begin = ntheta - 1 - sector.col_end;
        const int south_sector_end = ntheta - 1 - sector.col_begin;
        const int north_j0 = 0;
        const int north_j1 = north_sector_end + 1;
        const int south_j0 = south_sector_begin;
        const int south_j1 = ntheta;
        std::array<std::vector<double>, 2> taper_w;
        taper_w[0].assign(
            static_cast<std::size_t>(ntheta + 1), 0.0);
        taper_w[1].assign(
            static_cast<std::size_t>(ntheta + 1), 0.0);
        for (int spoke = north_j0 + 1; spoke < north_j1; ++spoke) {
          taper_w[0][static_cast<std::size_t>(spoke)] =
              spoke <= core_sector.col_end ? 1.0 : 0.5;
        }
        const int south_core_begin =
            ntheta - 1 - core_sector.col_end;
        for (int spoke = south_j0 + 1; spoke < south_j1; ++spoke) {
          taper_w[1][static_cast<std::size_t>(spoke)] =
              spoke > south_core_begin ? 1.0 : 0.5;
        }

        std::set<std::pair<int, int>> expanded_owned_rows;
        for (const auto& owned_row : owned_rows) {
          const auto& block = topology.blocks[
              static_cast<std::size_t>(owned_row.first)];
          const int node_row_count =
              block.owned_node_count / (ntheta + 1);
          const int row_begin = std::max(
              0,
              owned_row.second -
                  cfg.numerics.ale.band_ale.pole_theta_halo_rows);
          const int row_end = std::min(
              node_row_count - 1,
              owned_row.second +
                  cfg.numerics.ale.band_ale.pole_theta_halo_rows);
          for (int node_row = row_begin; node_row <= row_end; ++node_row) {
            expanded_owned_rows.emplace(owned_row.first, node_row);
          }
        }

        struct HardCandidateRow {
          std::pair<int, int> row_id = {-1, -1};
          std::vector<double> theta;
          std::vector<int> nodes;
          std::vector<double> applied;
          PoleThetaRowCurve curve;
          bool curve_ok = false;
          double s_min = 0.0;
          double s_max = 0.0;
        };
        std::vector<HardCandidateRow> candidate_rows;
        candidate_rows.reserve(expanded_owned_rows.size());
        std::size_t anchor_skipped_nodes = 0U;
        double a_odd_max = 0.0;
        bool curve_failure_logged = false;
        for (const auto& owned_row : expanded_owned_rows) {
          const auto& block = topology.blocks[
              static_cast<std::size_t>(owned_row.first)];
          const int node_row = owned_row.second;
          std::vector<double> theta_row(
              static_cast<std::size_t>(ntheta + 1), 0.0);
          std::vector<double> s_row;
          if (curve_preserving_enabled) {
            s_row.assign(static_cast<std::size_t>(ntheta + 1), 0.0);
          }
          std::vector<int> nodes(
              static_cast<std::size_t>(ntheta + 1), -1);
          for (int spoke = 0; spoke <= ntheta; ++spoke) {
            const int node = block.owned_node_begin +
                             node_row * (ntheta + 1) + spoke;
            const auto node_index = static_cast<std::size_t>(node);
            nodes[static_cast<std::size_t>(spoke)] = node;
            theta_row[static_cast<std::size_t>(spoke)] = std::atan2(
                std::max(node_r[node_index], 0.0), node_z[node_index]);
            if (curve_preserving_enabled) {
              s_row[static_cast<std::size_t>(spoke)] = std::hypot(
                  node_r[node_index], node_z[node_index]);
            }
          }
          PoleThetaRowCurve curve;
          const bool curve_ok =
              !curve_preserving_enabled
                  ? false
                  : pole_theta_build_row_curve(
                        theta_row, s_row, fit_order, protected_modes, &curve);
          double s_min_row = 0.0;
          double s_max_row = 0.0;
          if (curve_ok) {
            const auto row_envelope =
                std::minmax_element(s_row.begin(), s_row.end());
            s_min_row = *row_envelope.first;
            s_max_row = *row_envelope.second;
          }
          if (curve_preserving_enabled && !curve_ok && rank == 0 &&
              !curve_failure_logged) {
            std::ostringstream log;
            log << "[pole-theta] curve reconstruction failed coupled=1"
                << " row=(" << owned_row.first << "," << owned_row.second
                << ") — radius-preserving fallback";
            core::log_info(log.str());
            curve_failure_logged = true;
          }

          const std::vector<double> d_north = pole_theta_delta_map(
              theta_row,
              north_j0,
              north_j1,
              cfg.numerics.ale.band_ale.pole_theta_alpha_hard,
              cfg.numerics.ale.band_ale.pole_theta_deadband_frac,
              cfg.numerics.ale.band_ale.pole_theta_move_limit_hard_frac,
              dtheta_star,
              taper_w[0]);
          const std::vector<double> d_south = pole_theta_delta_map(
              theta_row,
              south_j0,
              south_j1,
              cfg.numerics.ale.band_ale.pole_theta_alpha_hard,
              cfg.numerics.ale.band_ale.pole_theta_deadband_frac,
              cfg.numerics.ale.band_ale.pole_theta_move_limit_hard_frac,
              dtheta_star,
              taper_w[1]);
          std::vector<double> d_north_applied;
          std::vector<double> d_south_applied;
          const double a_odd = pole_theta_parity_gate(
              d_north,
              d_south,
              ntheta,
              cfg.numerics.ale.band_ale.pole_theta_deadband_frac,
              dtheta_star,
              &d_north_applied,
              &d_south_applied);
          a_odd_max = std::max(a_odd_max, a_odd);

          HardCandidateRow candidate;
          candidate.row_id = owned_row;
          candidate.theta = theta_row;
          candidate.nodes = nodes;
          candidate.applied.assign(
              static_cast<std::size_t>(ntheta + 1), 0.0);
          candidate.curve = std::move(curve);
          candidate.curve_ok = curve_ok;
          candidate.s_min = s_min_row;
          candidate.s_max = s_max_row;
          const std::array<int, 2> j0 = {{north_j0, south_j0}};
          const std::array<int, 2> j1 = {{north_j1, south_j1}};
          const std::array<const std::vector<double>*, 2> applied = {{
              &d_north_applied,
              &d_south_applied,
          }};
          for (int pole_index = 0; pole_index < 2; ++pole_index) {
            if (shock_hold[static_cast<std::size_t>(pole_index)]) {
              continue;
            }
            for (int spoke = j0[static_cast<std::size_t>(pole_index)];
                 spoke <= j1[static_cast<std::size_t>(pole_index)];
                 ++spoke) {
              const double delta =
                  (*applied[static_cast<std::size_t>(pole_index)])
                      [static_cast<std::size_t>(spoke)];
              const auto node_index = static_cast<std::size_t>(
                  nodes[static_cast<std::size_t>(spoke)]);
              if (!state.central_pseudo_core.boundary_node_mask.empty() &&
                  node_index <
                      state.central_pseudo_core.boundary_node_mask.size() &&
                  state.central_pseudo_core.boundary_node_mask[node_index] !=
                      0U) {
                ++anchor_skipped_nodes;
                continue;
              }
              candidate.applied[static_cast<std::size_t>(spoke)] = delta;
            }
          }
          candidate_rows.push_back(std::move(candidate));
        }

        // The effectiveness audit always uses radius-preserving trial
        // positions, including when the committed curve variant changes radii
        // by design.
        std::map<int, std::pair<double, double>> trial_positions;
        std::map<int, double> applied_by_node;
        for (const auto& candidate : candidate_rows) {
          for (int spoke = 0; spoke <= ntheta; ++spoke) {
            const std::size_t spoke_index = static_cast<std::size_t>(spoke);
            const int node = candidate.nodes[spoke_index];
            const std::size_t node_index = static_cast<std::size_t>(node);
            const double delta = candidate.applied[spoke_index];
            const double theta_trial = candidate.theta[spoke_index] + delta;
            const double radius = std::hypot(
                node_r[node_index], node_z[node_index]);
            trial_positions[node] = {
                radius * std::sin(theta_trial),
                radius * std::cos(theta_trial)};
            applied_by_node[node] = delta;
          }
        }

        std::vector<PoleThetaTrialCellGeometry> trial_cells;
        std::vector<std::size_t> affected_scan_indices;
        std::map<int, std::size_t> affected_cells_by_id;
        double damage_denominator = 0.0;
        for (std::size_t scan_index = 0;
             scan_index < hard_audit_scan.size();
             ++scan_index) {
          const auto& scanned = hard_audit_scan[scan_index];
          bool affected = true;
          for (const int node : scanned.quad) {
            affected = affected && trial_positions.find(node) !=
                                       trial_positions.end();
          }
          if (!affected) {
            continue;
          }
          PoleThetaTrialCellGeometry trial;
          trial.weight = scanned.weight;
          trial.h_theta_star = scanned.h_theta_star;
          trial.frozen_q_pre = scanned.q_pre;
          trial.frozen_min_altitude_pre = scanned.min_altitude_pre;
          for (int corner = 0; corner < 4; ++corner) {
            const std::size_t c = static_cast<std::size_t>(corner);
            const int node = scanned.quad[c];
            const std::size_t node_index = static_cast<std::size_t>(node);
            trial.pre_r[c] = node_r[node_index];
            trial.pre_z[c] = node_z[node_index];
            trial.post_r[c] = trial_positions.at(node).first;
            trial.post_z[c] = trial_positions.at(node).second;
          }
          affected_cells_by_id.emplace(scanned.cell, scan_index);
          affected_scan_indices.push_back(scan_index);
          damage_denominator += scanned.weight * scanned.area;
          trial_cells.push_back(trial);
        }

        const PoleThetaTrialAggregate audit = pole_theta_trial_aggregate(
            trial_cells,
            cfg.numerics.ale.band_ale.pole_theta_h_hard,
            cfg.numerics.ale.band_ale.pole_theta_kappa_hard);
        const double R_Q =
            (audit.q8_pre - audit.q8_post) /
            std::max(audit.q8_pre, 1.0e-12);
        double R_dt = 0.0;
        if (std::isfinite(audit.min_altitude_pre) &&
            std::isfinite(audit.min_altitude_post)) {
          R_dt =
              (audit.min_altitude_post - audit.min_altitude_pre) /
              std::max(audit.min_altitude_pre, 1.0e-30);
        }
        const double benefit = pole_theta_hard_benefit(R_Q, R_dt);
        // A predicted time-to-inversion benefit channel is deferred.

        double damage_numerator = 0.0;
        for (const std::size_t scan_index : affected_scan_indices) {
          const auto& left = hard_audit_scan[scan_index];
          if (left.col + 1 >= ntheta) {
            continue;
          }
          const auto& block = topology.blocks[
              static_cast<std::size_t>(left.block_id)];
          const int right_cell =
              block.cell_begin + left.row * ntheta + left.col + 1;
          const auto right_it = affected_cells_by_id.find(right_cell);
          if (right_it == affected_cells_by_id.end()) {
            continue;
          }
          const auto& right = hard_audit_scan[right_it->second];
          const double shared_displacement = 0.5 * (
              applied_by_node.at(left.quad[3]) +
              applied_by_node.at(left.quad[2]));
          const double sweep =
              std::abs(shared_displacement) * 0.5 *
              (left.area + right.area);
          const double rho_star = 1.0e-12 *
              std::max(left.rho_ref, right.rho_ref);
          const double rho_left =
              rho_host[static_cast<std::size_t>(left.cell)];
          const double rho_right =
              rho_host[static_cast<std::size_t>(right.cell)];
          const double L_f = std::min(
              std::abs(std::log(
                  (rho_left + rho_star) / (rho_right + rho_star))),
              std::log(1.0e8));
          const double w_f = std::max(left.weight, right.weight);
          damage_numerator += sweep * w_f * L_f;
        }
        const double D_rho = damage_numerator /
            std::max(damage_denominator, 1.0e-300);
        const double damage_gate = pole_theta_damage_gate(D_rho);

        if (futility_control_enabled) {
          const double debt_decayed = band.pole_hard_debt * std::exp(
              -(state.t - band.pole_hard_t_last_eval) / 8.0e-12);
          const double futility = 1.0 - band_activation_up(
              benefit, 0.25, 0.75);
          // Hardcoded futility-debt ceiling b_max = 4 (consult 7).
          band.pole_hard_debt = std::min(
              4.0, debt_decayed + futility);
          band.pole_hard_t_last_eval = state.t;
        }
        // Novelty compares against the last evaluated coupled fire,
        // regardless of whether that evaluation was effective or futile.
        band.pole_hard_signature = owned_rows;

        if (benefit == 0.0 || damage_gate == 0.0) {
          ++band.pole_hard_futile_suppressions;
          if (rank == 0 &&
              band.pole_hard_futile_suppressions % 50U == 0U) {
            std::ostringstream log;
            log << "[pole-theta] futile B=" << std::scientific
                << std::setprecision(6) << benefit
                << " R_Q=" << R_Q
                << " R_dt=" << R_dt
                << " D_rho=" << D_rho
                << " debt=" << band.pole_hard_debt;
            core::log_info(log.str());
          }
        } else {
          const double applied_scale = authority * benefit * damage_gate;
          std::size_t added_nodes = 0U;
          for (const auto& candidate : candidate_rows) {
            for (int spoke = 0; spoke <= ntheta; ++spoke) {
              const std::size_t spoke_index = static_cast<std::size_t>(spoke);
              const std::size_t node_index = static_cast<std::size_t>(
                  candidate.nodes[spoke_index]);
              const double delta =
                  applied_scale * candidate.applied[spoke_index];
              if (delta != 0.0 &&
                  g_runtime->pole_theta_dtheta[node_index] == 0.0) {
                ++added_nodes;
              }
              g_runtime->pole_theta_dtheta[node_index] = delta;
              if (candidate.curve_ok && delta != 0.0) {
                const double s_target = pole_theta_eval_row_curve(
                    candidate.curve,
                    candidate.theta[spoke_index] + delta);
                if (pole_theta_s_target_admissible(
                        s_target, candidate.s_min, candidate.s_max)) {
                  g_runtime->pole_theta_target_s[node_index] = s_target;
                }
              }
            }
          }

          for (int pole_index = 0; pole_index < 2; ++pole_index) {
            if (shock_hold[static_cast<std::size_t>(pole_index)]) {
              continue;
            }
            auto& pole =
                pole_index == 0 ? band.pole_north : band.pole_south;
            pole.fired_this_step = true;
            if (curve_preserving_enabled) {
              pole.last_fire_row_ids.clear();
              for (auto row_it = expanded_owned_rows.begin();
                   row_it != expanded_owned_rows.end() &&
                   pole.last_fire_row_ids.size() < 4U;
                   ++row_it) {
                pole.last_fire_row_ids.push_back(*row_it);
              }
              pole.last_fire_row_count =
                  static_cast<int>(pole.last_fire_row_ids.size());
              const std::size_t modes_per_row = protected_modes.size();
              pole.last_fire_row_a.assign(
                  pole.last_fire_row_ids.size() * modes_per_row,
                  std::numeric_limits<double>::quiet_NaN());
              for (std::size_t row_index = 0;
                   row_index < pole.last_fire_row_ids.size();
                   ++row_index) {
                const auto& row_id = pole.last_fire_row_ids[row_index];
                const auto& block = topology.blocks[
                    static_cast<std::size_t>(row_id.first)];
                std::vector<double> theta_row(
                    static_cast<std::size_t>(ntheta + 1), 0.0);
                std::vector<double> s_row(
                    static_cast<std::size_t>(ntheta + 1), 0.0);
                for (int spoke = 0; spoke <= ntheta; ++spoke) {
                  const int node = block.owned_node_begin +
                                   row_id.second * (ntheta + 1) + spoke;
                  const auto node_index = static_cast<std::size_t>(node);
                  theta_row[static_cast<std::size_t>(spoke)] = std::atan2(
                      std::max(node_r[node_index], 0.0), node_z[node_index]);
                  s_row[static_cast<std::size_t>(spoke)] = std::hypot(
                      node_r[node_index], node_z[node_index]);
                }
                std::vector<double> pre_a;
                if (!pole_theta_legendre_fit(
                        theta_row, s_row, fit_order, &pre_a)) {
                  continue;
                }
                const std::size_t row_offset = row_index * modes_per_row;
                for (std::size_t mode_index = 0;
                     mode_index < modes_per_row;
                     ++mode_index) {
                  const int mode = protected_modes[mode_index];
                  if (mode <= fit_order) {
                    pole.last_fire_row_a[row_offset + mode_index] =
                        pre_a[static_cast<std::size_t>(mode)];
                  }
                }
              }
            }
          }

          if (futility_control_enabled) {
            band.pole_hard_t_last = state.t;
          }
          if (rank == 0) {
            std::ostringstream log;
            log << "[pole-theta] fire coupled=1"
                << " step=" << state.step + 1
                << " q_max=" << std::scientific << std::setprecision(6)
                << coupled_q_max
                << " H_min=" << std::scientific << std::setprecision(6)
                << coupled_H_min
                << " kappa_max=" << coupled_kappa_max
                << " H_min_raw=" << coupled_H_min
                << " kappa_max_raw=" << coupled_kappa_max
                << " sector=[" << north_sector_begin
                << "," << north_sector_end << "]"
                << " mirror_sector=[" << south_sector_begin
                << "," << south_sector_end << "]"
                << " nodes=" << added_nodes
                << " a_odd_max=" << a_odd_max
                << " anchor_skipped_nodes=" << anchor_skipped_nodes;
            core::log_info(log.str());
          }
        }
      }
    }
  }

  band.min_ratio = std::min(status_H[0], status_H[1]);
  if (rank == 0 && (state.step + 1) % 200 == 0 &&
      (band.pole_north.armed || band.pole_south.armed)) {
    std::ostringstream log;
    log << "[pole-theta] armed step=" << state.step + 1
        << " N_H_min=" << std::scientific << std::setprecision(6)
        << status_H[0]
        << " N_kappa_max=" << status_kappa[0]
        << " S_H_min=" << status_H[1]
        << " S_kappa_max=" << status_kappa[1];
    core::log_info(log.str());
  }
}

double shell_front_radius(const mesh::BlockInfo& block,
                          const std::vector<double>& ring_radius,
                          const std::vector<double>& rho) {
  if (block.n_i_cells < 3) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  TENRYU_ASSERT(
      ring_radius.size() ==
          static_cast<std::size_t>(block.n_i_cells) + 1U,
      "band ALE shell front requires one radius per node ring");

  std::vector<double> row_radius(
      static_cast<std::size_t>(block.n_i_cells), 0.0);
  std::vector<double> ring_mean_rho(
      static_cast<std::size_t>(block.n_i_cells), 0.0);
  for (int row = 0; row < block.n_i_cells; ++row) {
    const auto row_index = static_cast<std::size_t>(row);
    row_radius[row_index] =
        0.5 * (ring_radius[row_index] + ring_radius[row_index + 1U]);
    const int row_begin = block.cell_begin + row * block.n_j_cells;
    for (int j = 0; j < block.n_j_cells; ++j) {
      const int cell = row_begin + j;
      TENRYU_ASSERT(
          cell >= 0 && static_cast<std::size_t>(cell) < rho.size() &&
              std::isfinite(rho[static_cast<std::size_t>(cell)]),
          "band ALE shell front density sample is unavailable");
      ring_mean_rho[row_index] += rho[static_cast<std::size_t>(cell)];
    }
    ring_mean_rho[row_index] /=
        static_cast<double>(block.n_j_cells);
  }

  int front_row = 1;
  double maximum_gradient = -1.0;
  for (int row = 1; row + 1 < block.n_i_cells; ++row) {
    const auto row_index = static_cast<std::size_t>(row);
    const double radial_span =
        row_radius[row_index + 1U] - row_radius[row_index - 1U];
    TENRYU_ASSERT(
        std::isfinite(radial_span) && radial_span > 0.0,
        "band ALE shell centered-difference radial span must be positive");
    const double gradient = std::abs(
        (ring_mean_rho[row_index + 1U] -
         ring_mean_rho[row_index - 1U]) /
        radial_span);
    TENRYU_ASSERT(
        std::isfinite(gradient),
        "band ALE shell density gradient must be finite");
    if (gradient > maximum_gradient) {
      maximum_gradient = gradient;
      front_row = row;
    }
  }
  return row_radius[static_cast<std::size_t>(front_row)];
}

void update_shell_band(const core::State& state,
                       const core::Config& cfg,
                       BandState& band) {
  TENRYU_ASSERT(
      band.kind == BandKind::Shell,
      "band ALE shell update requires a shell band");
  const auto& topology = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(
      band.block_id >= 0 &&
          band.block_id < static_cast<int>(topology.blocks.size()),
      "band ALE shell block id is out of range");
  const auto& block =
      topology.blocks[static_cast<std::size_t>(band.block_id)];
  TENRYU_ASSERT(
      block.role == mesh::BlockRole::POLAR_SHELL &&
          band.shell_rings.size() ==
              static_cast<std::size_t>(block.n_i_cells) + 1U &&
          band.shell_initial_area.size() ==
              static_cast<std::size_t>(block.cell_count),
      "band ALE shell metadata is incomplete");

  std::vector<double> current_r;
  std::vector<double> current_z;
  std::vector<double> rho;
  state.x_r.copy_to_host(current_r);
  state.x_z.copy_to_host(current_z);
  state.rho.copy_to_host(rho);

  std::vector<double> current_mean_radius;
  current_mean_radius.reserve(band.shell_rings.size());
  for (const auto& ring : band.shell_rings) {
    current_mean_radius.push_back(
        ring_mean_radius(ring, current_r, current_z));
  }
  for (std::size_t ring = 1; ring < current_mean_radius.size(); ++ring) {
    TENRYU_ASSERT(
        std::isfinite(current_mean_radius[ring]) &&
            current_mean_radius[ring] > current_mean_radius[ring - 1U],
        "band ALE shell mean ring radii must be strictly increasing");
  }

  band.lines.clear();
  band.cells.clear();
  band.pinned_nodes.clear();
  band.shell_min_spacing = std::numeric_limits<double>::infinity();
  double r_front;
  if (cfg.numerics.ale.band_ale.shell_front_metric == "min_spacing") {
    std::size_t front_pair = 0;
    double minimum_spacing = std::numeric_limits<double>::infinity();
    for (std::size_t ring = 0; ring + 1U < current_mean_radius.size();
         ++ring) {
      const double spacing =
          current_mean_radius[ring + 1U] - current_mean_radius[ring];
      if (spacing < minimum_spacing) {
        minimum_spacing = spacing;
        front_pair = ring;
      }
    }
    r_front = 0.5 * (current_mean_radius[front_pair] +
                     current_mean_radius[front_pair + 1U]);
  } else {
    r_front = shell_front_radius(block, current_mean_radius, rho);
  }
  if (!std::isfinite(r_front)) {
    return;
  }
  band.sort_radius = r_front;
  const double window_inner =
      r_front -
      static_cast<double>(cfg.numerics.ale.band_ale.shell_window_in_rows) *
          band.shell_dr0;
  const double window_outer =
      r_front +
      static_cast<double>(cfg.numerics.ale.band_ale.shell_window_out_rows) *
          band.shell_dr0;

  int first_ring = -1;
  int last_ring = -1;
  const int outermost_allowed_ring =
      block.n_i_cells -
      cfg.numerics.ale.band_ale.shell_boundary_guard_rows;
  for (int ring = 0;
       ring < static_cast<int>(current_mean_radius.size());
       ++ring) {
    const double radius =
        current_mean_radius[static_cast<std::size_t>(ring)];
    if (ring <= outermost_allowed_ring &&
        radius >= window_inner && radius <= window_outer) {
      if (first_ring < 0) {
        first_ring = ring;
      }
      last_ring = ring;
    }
  }
  if (first_ring < 0 || last_ring - first_ring + 1 < 3) {
    return;
  }

  for (int ring = first_ring + 1; ring <= last_ring; ++ring) {
    band.shell_min_spacing = std::min(
        band.shell_min_spacing,
        current_mean_radius[static_cast<std::size_t>(ring)] -
            current_mean_radius[static_cast<std::size_t>(ring - 1)]);
  }

  band.lines.reserve(
      static_cast<std::size_t>(last_ring - first_ring + 1));
  for (int ring = last_ring; ring >= first_ring; --ring) {
    band.lines.push_back(
        band.shell_rings[static_cast<std::size_t>(ring)]);
  }
  std::vector<double> initial_area;
  initial_area.reserve(
      static_cast<std::size_t>(last_ring - first_ring) *
      static_cast<std::size_t>(block.n_j_cells));
  for (int row = first_ring; row < last_ring; ++row) {
    const std::vector<int> row_cells = block_row_cells(block, row);
    append_cells(band.cells, row_cells);
    for (const int cell : row_cells) {
      initial_area.push_back(
          band.shell_initial_area[
              static_cast<std::size_t>(cell - block.cell_begin)]);
    }
  }
  band.pinned_nodes.assign(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  for (const int node : band.lines.front()) {
    band.pinned_nodes[static_cast<std::size_t>(node)] = 1U;
  }
  for (const int node : band.lines.back()) {
    band.pinned_nodes[static_cast<std::size_t>(node)] = 1U;
  }
  band.d_cells.reset(band.cells.size());
  band.d_cells.copy_from_host(band.cells);
  band.d_initial_area.reset(initial_area.size());
  band.d_initial_area.copy_from_host(initial_area);
}

struct PcColumnWindow {
  int component_first = -1;
  int component_last = -1;
  int i0 = -1;
  int i1 = -1;
  bool well_formed = false;
  bool front_inside = false;
  std::uint8_t reason = 0U;
  double eta = 0.0;
};

struct PcEvaluation {
  std::vector<PcColumnWindow> columns;
  std::vector<std::uint8_t> classification;
  std::vector<int> tube_i0;
  std::vector<int> tube_i1;
  std::vector<std::uint8_t> tube_mask;
  std::vector<std::uint8_t> patch_eligible;
  double coverage = 0.0;
  double eta_p = 0.0;
  int held_columns = 0;
  int patch_count = 0;
  int tube_columns = 0;
  int ambiguous_columns = 0;
  std::uint8_t escalation_reason = 0U;
  bool phase_b = false;
  bool hold = false;
};

struct LegacyEstimatorDecision {
  int i0 = -1;
  int i1 = -1;
  bool hold = false;
};

int reflected_column_index(int column, const int n_columns) {
  TENRYU_ASSERT(n_columns > 0,
                "band ALE per-column reflection requires columns");
  while (column < 0 || column >= n_columns) {
    if (column < 0) {
      column = -column - 1;
    } else {
      column = 2 * n_columns - column - 1;
    }
  }
  return column;
}

double column_ring_radius(const BandState& band,
                          const int ring,
                          const int column,
                          const std::vector<double>& node_r,
                          const std::vector<double>& node_z) {
  TENRYU_ASSERT(
      ring >= 0 && ring < static_cast<int>(band.shell_rings.size()) &&
          column >= 0 &&
          column + 1 <
              static_cast<int>(band.shell_rings[static_cast<std::size_t>(ring)]
                                   .size()),
      "band ALE per-column ring sample is out of range");
  const auto& nodes = band.shell_rings[static_cast<std::size_t>(ring)];
  const int node0 = nodes[static_cast<std::size_t>(column)];
  const int node1 = nodes[static_cast<std::size_t>(column + 1)];
  return 0.5 * (
      std::hypot(node_r[static_cast<std::size_t>(node0)],
                 node_z[static_cast<std::size_t>(node0)]) +
      std::hypot(node_r[static_cast<std::size_t>(node1)],
                 node_z[static_cast<std::size_t>(node1)]));
}

double column_edge_spacing(const BandState& band,
                           const int ring,
                           const int column,
                           const std::vector<double>& node_r,
                           const std::vector<double>& node_z) {
  const int last_ring = static_cast<int>(band.shell_rings.size()) - 1;
  double spacing = std::numeric_limits<double>::infinity();
  const double radius =
      column_ring_radius(band, ring, column, node_r, node_z);
  if (ring > 0) {
    spacing = std::min(
        spacing,
        std::abs(radius - column_ring_radius(
                              band, ring - 1, column, node_r, node_z)));
  }
  if (ring < last_ring) {
    spacing = std::min(
        spacing,
        std::abs(column_ring_radius(
                     band, ring + 1, column, node_r, node_z) -
                 radius));
  }
  return spacing;
}

std::vector<std::pair<int, int>> per_column_components(
    const core::State& state,
    const mesh::BlockInfo& block,
    const int column,
    const double cut,
    bool* any_component) {
  std::vector<std::pair<int, int>> components;
  *any_component = false;
  int row = 0;
  while (row < block.n_i_cells) {
    const int cell = block.cell_begin + row * block.n_j_cells + column;
    const double error = state.refine_error[static_cast<std::size_t>(cell)];
    TENRYU_ASSERT(
        std::isfinite(error),
        "band ALE per-column estimator sample is unavailable");
    if (error < cut) {
      ++row;
      continue;
    }
    const int first = row;
    int last = row;
    ++row;
    while (row < block.n_i_cells) {
      const int next_cell =
          block.cell_begin + row * block.n_j_cells + column;
      const double next_error =
          state.refine_error[static_cast<std::size_t>(next_cell)];
      TENRYU_ASSERT(
          std::isfinite(next_error),
          "band ALE per-column estimator sample is unavailable");
      if (next_error >= cut) {
        last = row;
        ++row;
        continue;
      }
      if (row + 1 < block.n_i_cells &&
          next_error >= 0.60) {
        const int after_gap =
            block.cell_begin + (row + 1) * block.n_j_cells + column;
        const double after_gap_error =
            state.refine_error[static_cast<std::size_t>(after_gap)];
        TENRYU_ASSERT(
            std::isfinite(after_gap_error),
            "band ALE per-column estimator sample is unavailable");
        if (after_gap_error >= cut) {
          last = row + 1;
          row += 2;
          continue;
        }
      }
      break;
    }
    *any_component = true;
    if (last - first + 1 >= 3) {
      components.emplace_back(first, last);
    }
  }
  return components;
}

double interval_overlap(const std::pair<int, int>& current,
                        const std::array<int, 2>& previous) {
  if (previous[0] < 0 || previous[1] < previous[0]) {
    return 0.0;
  }
  const int intersection =
      std::max(0, std::min(current.second, previous[1]) -
                      std::max(current.first, previous[0]) + 1);
  if (intersection == 0) {
    return 0.0;
  }
  const int union_size =
      std::max(current.second, previous[1]) -
      std::min(current.first, previous[0]) + 1;
  return static_cast<double>(intersection) /
         static_cast<double>(union_size);
}

bool component_window_holds(const core::State& state,
                            const core::Config& cfg,
                            const mesh::BlockInfo& block,
                            const int column,
                            const std::pair<int, int>& component,
                            int* i0,
                            int* i1) {
  *i0 = std::max(
      0, component.first -
             cfg.numerics.ale.band_ale.estimator_band_in_rows);
  const int outermost_allowed_ring =
      block.n_i_cells -
      cfg.numerics.ale.band_ale.shell_boundary_guard_rows;
  *i1 = std::min(
      outermost_allowed_ring,
      component.second +
          cfg.numerics.ale.band_ale.estimator_band_out_rows + 1);
  if (*i1 - *i0 + 1 < 3) {
    return false;
  }
  for (int row = *i0; row < *i1; ++row) {
    const int cell = block.cell_begin + row * block.n_j_cells + column;
    if (state.refine_error[static_cast<std::size_t>(cell)] >=
        cfg.numerics.ale.band_ale.estimator_band_shock_hold) {
      return true;
    }
  }
  return false;
}

int nearest_column_ring(const BandState& band,
                        const int column,
                        const int last_ring,
                        const double target_radius,
                        const std::vector<double>& node_r,
                        const std::vector<double>& node_z) {
  int nearest = 0;
  double nearest_distance = std::numeric_limits<double>::infinity();
  for (int ring = 0; ring <= last_ring; ++ring) {
    const double distance = std::abs(
        column_ring_radius(band, ring, column, node_r, node_z) -
        target_radius);
    if (distance < nearest_distance) {
      nearest_distance = distance;
      nearest = ring;
    }
  }
  return nearest;
}

void regularize_column_edge(
    const core::Config& cfg,
    const BandState& band,
    const mesh::BlockInfo& block,
    const bool inner,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    std::vector<PcColumnWindow>& columns,
    const std::vector<std::uint8_t>& filter_mask,
    std::vector<std::uint8_t>& valid) {
  const int n_columns = block.n_j_cells;
  const int q =
      cfg.numerics.ale.band_ale.estimator_band_pc_filter_halfwidth;
  const double dtheta = CUDART_PI / static_cast<double>(n_columns);
  std::vector<double> raw(static_cast<std::size_t>(n_columns), 0.0);
  std::vector<double> repaired(static_cast<std::size_t>(n_columns), 0.0);
  std::vector<double> filtered(static_cast<std::size_t>(n_columns), 0.0);
  for (int j = 0; j < n_columns; ++j) {
    if (filter_mask[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    const auto& column = columns[static_cast<std::size_t>(j)];
    const int ring = inner ? column.i0 : column.i1;
    raw[static_cast<std::size_t>(j)] =
        column_ring_radius(band, ring, j, node_r, node_z);
  }
  repaired = raw;
  for (int j = 0; j < n_columns; ++j) {
    if (filter_mask[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    const int jm = reflected_column_index(j - 1, n_columns);
    const int jp = reflected_column_index(j + 1, n_columns);
    if (filter_mask[static_cast<std::size_t>(jm)] == 0U ||
        filter_mask[static_cast<std::size_t>(jp)] == 0U) {
      continue;
    }
    const int ring = inner ? columns[static_cast<std::size_t>(j)].i0
                           : columns[static_cast<std::size_t>(j)].i1;
    const double h =
        column_edge_spacing(band, ring, j, node_r, node_z);
    const double neighbor_mean = 0.5 * (
        raw[static_cast<std::size_t>(jm)] +
        raw[static_cast<std::size_t>(jp)]);
    if (std::abs(raw[static_cast<std::size_t>(j)] - neighbor_mean) >
            1.5 * h &&
        std::abs(raw[static_cast<std::size_t>(jm)] -
                 raw[static_cast<std::size_t>(jp)]) <
            0.5 * h) {
      repaired[static_cast<std::size_t>(j)] = neighbor_mean;
    }
  }

  const int kernel_denominator = 1 << (2 * q);
  for (int j = 0; j < n_columns; ++j) {
    if (filter_mask[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    double numerator = 0.0;
    double mass = 0.0;
    int coefficient = 1;
    for (int slot = 0; slot <= 2 * q; ++slot) {
      if (slot > 0) {
        coefficient = coefficient * (2 * q - slot + 1) / slot;
      }
      const int k = slot - q;
      const int sample = reflected_column_index(j + k, n_columns);
      const double weight =
          static_cast<double>(coefficient) /
          static_cast<double>(kernel_denominator);
      if (filter_mask[static_cast<std::size_t>(sample)] != 0U) {
        numerator += weight * repaired[static_cast<std::size_t>(sample)];
        mass += weight;
      }
    }
    if (mass < 0.75) {
      valid[static_cast<std::size_t>(j)] = 0U;
      columns[static_cast<std::size_t>(j)].reason = 3U;
    } else {
      filtered[static_cast<std::size_t>(j)] = numerator / mass;
    }
  }

  // Filter against immutable raw membership before limiting; filter-mass,
  // slope/curvature, and snapping invalidations must not feed back into the mask.
  const std::vector<double> slope_input = filtered;
  for (int j = 0; j < n_columns; ++j) {
    if (valid[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    double maximum_slope = 0.0;
    double lower = -std::numeric_limits<double>::infinity();
    double upper = std::numeric_limits<double>::infinity();
    const double alpha =
        0.5 * cfg.numerics.ale.band_ale.estimator_band_pc_slope_limit *
        dtheta;
    const double ratio_min =
        alpha < 1.0 ? (1.0 - alpha) / (1.0 + alpha) : 0.0;
    const double ratio_max =
        alpha < 1.0
            ? (1.0 + alpha) / (1.0 - alpha)
            : std::numeric_limits<double>::infinity();
    for (const int offset : {-1, 1}) {
      const int neighbor = reflected_column_index(j + offset, n_columns);
      if (neighbor == j ||
          valid[static_cast<std::size_t>(neighbor)] == 0U) {
        continue;
      }
      const double neighbor_radius =
          slope_input[static_cast<std::size_t>(neighbor)];
      const double mean_radius = 0.5 * (
          slope_input[static_cast<std::size_t>(j)] + neighbor_radius);
      maximum_slope = std::max(
          maximum_slope,
          std::abs(slope_input[static_cast<std::size_t>(j)] -
                   neighbor_radius) /
              (mean_radius * dtheta));
      lower = std::max(lower, ratio_min * neighbor_radius);
      upper = std::min(upper, ratio_max * neighbor_radius);
    }
    if (maximum_slope >
        cfg.numerics.ale.band_ale.estimator_band_pc_slope_reject) {
      valid[static_cast<std::size_t>(j)] = 0U;
      columns[static_cast<std::size_t>(j)].reason = 4U;
      continue;
    }
    if (maximum_slope >
        cfg.numerics.ale.band_ale.estimator_band_pc_slope_limit) {
      const auto& column = columns[static_cast<std::size_t>(j)];
      const int conservative_ring =
          inner ? column.component_first : column.component_last + 1;
      const double conservative_radius = column_ring_radius(
          band, conservative_ring, j, node_r, node_z);
      if (inner) {
        upper = std::min(upper, conservative_radius);
      } else {
        lower = std::max(lower, conservative_radius);
      }
      if (lower > upper) {
        valid[static_cast<std::size_t>(j)] = 0U;
        columns[static_cast<std::size_t>(j)].reason = 4U;
      } else {
        filtered[static_cast<std::size_t>(j)] = std::clamp(
            filtered[static_cast<std::size_t>(j)], lower, upper);
      }
    }
  }

  const std::vector<double> curvature_input = filtered;
  for (int j = 0; j < n_columns; ++j) {
    if (valid[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    const int jm = reflected_column_index(j - 1, n_columns);
    const int jp = reflected_column_index(j + 1, n_columns);
    if (valid[static_cast<std::size_t>(jm)] == 0U ||
        valid[static_cast<std::size_t>(jp)] == 0U) {
      continue;
    }
    const auto& column = columns[static_cast<std::size_t>(j)];
    const int edge_ring = inner ? column.i0 : column.i1;
    const double h =
        column_edge_spacing(band, edge_ring, j, node_r, node_z);
    const double neighbor_mean = 0.5 * (
        curvature_input[static_cast<std::size_t>(jm)] +
        curvature_input[static_cast<std::size_t>(jp)]);
    const double limit =
        0.5 * cfg.numerics.ale.band_ale.estimator_band_pc_curvature_limit *
        h;
    if (std::abs(curvature_input[static_cast<std::size_t>(j)] -
                 neighbor_mean) > limit) {
      double lower = neighbor_mean - limit;
      double upper = neighbor_mean + limit;
      const int conservative_ring =
          inner ? column.component_first : column.component_last + 1;
      const double conservative_radius = column_ring_radius(
          band, conservative_ring, j, node_r, node_z);
      if (inner) {
        upper = std::min(upper, conservative_radius);
      } else {
        lower = std::max(lower, conservative_radius);
      }
      if (lower > upper) {
        valid[static_cast<std::size_t>(j)] = 0U;
        columns[static_cast<std::size_t>(j)].reason = 5U;
      } else {
        filtered[static_cast<std::size_t>(j)] = std::clamp(
            filtered[static_cast<std::size_t>(j)], lower, upper);
      }
    }
  }

  const int last_ring =
      block.n_i_cells -
      cfg.numerics.ale.band_ale.shell_boundary_guard_rows;
  const std::vector<double> final_physical = filtered;
  for (int j = 0; j < n_columns; ++j) {
    if (valid[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    const int jm = reflected_column_index(j - 1, n_columns);
    const int jp = reflected_column_index(j + 1, n_columns);
    for (const int neighbor : {jm, jp}) {
      if (neighbor == j ||
          valid[static_cast<std::size_t>(neighbor)] == 0U) {
        continue;
      }
      const double mean_radius = 0.5 * (
          final_physical[static_cast<std::size_t>(j)] +
          final_physical[static_cast<std::size_t>(neighbor)]);
      const double slope = std::abs(
          final_physical[static_cast<std::size_t>(j)] -
          final_physical[static_cast<std::size_t>(neighbor)]) /
          (mean_radius * dtheta);
      if (slope >
          cfg.numerics.ale.band_ale.estimator_band_pc_slope_limit) {
        valid[static_cast<std::size_t>(j)] = 0U;
        columns[static_cast<std::size_t>(j)].reason = 4U;
        break;
      }
    }
    if (valid[static_cast<std::size_t>(j)] == 0U ||
        valid[static_cast<std::size_t>(jm)] == 0U ||
        valid[static_cast<std::size_t>(jp)] == 0U) {
      continue;
    }
    const auto& column = columns[static_cast<std::size_t>(j)];
    const int edge_ring = inner ? column.i0 : column.i1;
    const double h =
        column_edge_spacing(band, edge_ring, j, node_r, node_z);
    if (std::abs(
            final_physical[static_cast<std::size_t>(jp)] -
            2.0 * final_physical[static_cast<std::size_t>(j)] +
            final_physical[static_cast<std::size_t>(jm)]) >
        cfg.numerics.ale.band_ale.estimator_band_pc_curvature_limit * h) {
      valid[static_cast<std::size_t>(j)] = 0U;
      columns[static_cast<std::size_t>(j)].reason = 5U;
    }
  }
  std::vector<int> snapped(static_cast<std::size_t>(n_columns), -1);
  for (int j = 0; j < n_columns; ++j) {
    if (valid[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    const auto& column = columns[static_cast<std::size_t>(j)];
    int ring = nearest_column_ring(
        band, j, last_ring, filtered[static_cast<std::size_t>(j)],
        node_r, node_z);
    if (inner) {
      ring = std::min(ring, column.component_first);
    } else {
      ring = std::max(ring, column.component_last + 1);
    }
    snapped[static_cast<std::size_t>(j)] = ring;
  }

  const std::vector<int> snap_input = snapped;
  for (int j = 0; j < n_columns; ++j) {
    if (valid[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    int lower = 0;
    int upper = last_ring;
    for (const int offset : {-1, 1}) {
      const int neighbor = reflected_column_index(j + offset, n_columns);
      if (neighbor == j ||
          valid[static_cast<std::size_t>(neighbor)] == 0U) {
        continue;
      }
      lower = std::max(
          lower, snap_input[static_cast<std::size_t>(neighbor)] - 1);
      upper = std::min(
          upper, snap_input[static_cast<std::size_t>(neighbor)] + 1);
    }
    const auto& column = columns[static_cast<std::size_t>(j)];
    if (inner) {
      upper = std::min(upper, column.component_first);
    } else {
      lower = std::max(lower, column.component_last + 1);
    }
    if (lower > upper) {
      valid[static_cast<std::size_t>(j)] = 0U;
      columns[static_cast<std::size_t>(j)].reason = 6U;
    } else {
      snapped[static_cast<std::size_t>(j)] = std::clamp(
          snapped[static_cast<std::size_t>(j)], lower, upper);
    }
  }
  for (int j = 0; j < n_columns; ++j) {
    if (valid[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    for (const int offset : {-1, 1}) {
      const int neighbor = reflected_column_index(j + offset, n_columns);
      if (neighbor != j &&
          valid[static_cast<std::size_t>(neighbor)] != 0U &&
          std::abs(snapped[static_cast<std::size_t>(j)] -
                   snapped[static_cast<std::size_t>(neighbor)]) > 1) {
        valid[static_cast<std::size_t>(j)] = 0U;
        columns[static_cast<std::size_t>(j)].reason = 6U;
      }
    }
  }
  for (int j = 0; j < n_columns; ++j) {
    if (valid[static_cast<std::size_t>(j)] == 0U) {
      continue;
    }
    if (inner) {
      columns[static_cast<std::size_t>(j)].i0 =
          snapped[static_cast<std::size_t>(j)];
    } else {
      columns[static_cast<std::size_t>(j)].i1 =
          snapped[static_cast<std::size_t>(j)];
    }
  }
}

double median_value(std::vector<double> values) {
  TENRYU_ASSERT(!values.empty(),
                "band ALE per-column median requires samples");
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  return values.size() % 2U == 0U
             ? 0.5 * (values[middle - 1U] + values[middle])
             : values[middle];
}

void reset_ridge_history_column(BandState& band, const int column) {
  if (column < 0 ||
      static_cast<std::size_t>(column) >= band.pc_ridge_history.size()) {
    return;
  }
  band.pc_ridge_history[static_cast<std::size_t>(column)] = {};
  band.pc_ridge_history_count[static_cast<std::size_t>(column)] = 0U;
}

void push_ridge_sample(BandState& band,
                       const int column,
                       const PcRidgeSample& sample) {
  auto& history = band.pc_ridge_history[static_cast<std::size_t>(column)];
  auto& count =
      band.pc_ridge_history_count[static_cast<std::size_t>(column)];
  if (count > 0U) {
    const double last_time = history[static_cast<std::size_t>(count - 1U)].time;
    if (sample.time == last_time) {
      history[static_cast<std::size_t>(count - 1U)] = sample;
      return;
    }
    if (sample.time < last_time) {
      reset_ridge_history_column(band, column);
    }
  }
  if (count < history.size()) {
    history[static_cast<std::size_t>(count)] = sample;
    ++count;
    return;
  }
  for (std::size_t slot = 1; slot < history.size(); ++slot) {
    history[slot - 1U] = history[slot];
  }
  history.back() = sample;
}

bool ridge_speed(const BandState& band,
                 const int column,
                 double* speed) {
  const auto count =
      band.pc_ridge_history_count[static_cast<std::size_t>(column)];
  if (count < 3U) {
    return false;
  }
  const auto& history =
      band.pc_ridge_history[static_cast<std::size_t>(column)];
  double mean_time = 0.0;
  double mean_radius = 0.0;
  for (std::uint8_t slot = 0U; slot < count; ++slot) {
    const auto& sample = history[static_cast<std::size_t>(slot)];
    if (!sample.valid) {
      return false;
    }
    mean_time += sample.time;
    mean_radius += sample.radius;
  }
  mean_time /= static_cast<double>(count);
  mean_radius /= static_cast<double>(count);
  double numerator = 0.0;
  double denominator = 0.0;
  for (std::uint8_t slot = 0U; slot < count; ++slot) {
    const auto& sample = history[static_cast<std::size_t>(slot)];
    const double dt = sample.time - mean_time;
    numerator += dt * (sample.radius - mean_radius);
    denominator += dt * dt;
  }
  if (!(std::isfinite(numerator) && std::isfinite(denominator) &&
        denominator > 0.0)) {
    return false;
  }
  *speed = numerator / denominator;
  return std::isfinite(*speed);
}

double cell_radial_velocity(
    const mesh::MultiBlockTopology& topology,
    const mesh::Mesh& mesh_state,
    const int cell,
    const std::vector<double>& velocity_r,
    const std::vector<double>& velocity_z) {
  const double centroid_r =
      mesh_state.cell_centroid_r[static_cast<std::size_t>(cell)];
  const double centroid_z =
      mesh_state.cell_centroid_z[static_cast<std::size_t>(cell)];
  const double radius = std::max(
      std::hypot(centroid_r, centroid_z), 1.0e-300);
  const int begin =
      topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int end =
      topology.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
  if (begin < 0 || end <= begin ||
      end > static_cast<int>(topology.cell_node_csr_indices.size())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double sum = 0.0;
  for (int offset = begin; offset < end; ++offset) {
    const int node = topology.cell_node_csr_indices[
        static_cast<std::size_t>(offset)];
    if (node < 0 ||
        static_cast<std::size_t>(node) >= velocity_r.size() ||
        static_cast<std::size_t>(node) >= velocity_z.size()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    sum +=
        (velocity_r[static_cast<std::size_t>(node)] * centroid_r +
         velocity_z[static_cast<std::size_t>(node)] * centroid_z) /
        radius;
  }
  return sum / static_cast<double>(end - begin);
}

bool node_incident_cells_disjoint_from_tube(
    const std::vector<std::uint8_t>& tube,
    const int n_rows,
    const int n_columns,
    const int ring,
    const int spoke) {
  for (const int row : {ring - 1, ring}) {
    if (row < 0 || row >= n_rows) {
      continue;
    }
    for (const int column : {spoke - 1, spoke}) {
      if (column < 0 || column >= n_columns) {
        continue;
      }
      if (tube[static_cast<std::size_t>(row) *
                   static_cast<std::size_t>(n_columns) +
               static_cast<std::size_t>(column)] != 0U) {
        return false;
      }
    }
  }
  return true;
}

PcEvaluation evaluate_estimator_columns(
    const core::State& state,
    const core::Config& cfg,
    const BandState& band,
    std::vector<std::array<int, 2>>& previous_intervals,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  const auto& topology = *state.mesh.topo.multiblock;
  const auto& block =
      topology.blocks[static_cast<std::size_t>(band.block_id)];
  const int n_columns = block.n_j_cells;
  if (previous_intervals.size() != static_cast<std::size_t>(n_columns)) {
    previous_intervals.assign(
        static_cast<std::size_t>(n_columns),
        std::array<int, 2>{{-1, -1}});
  }
  PcEvaluation evaluation;
  evaluation.columns.resize(static_cast<std::size_t>(n_columns));
  std::vector<std::uint8_t> valid(
      static_cast<std::size_t>(n_columns), 0U);
  if (state.refine_error.size() !=
      static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    for (auto& column : evaluation.columns) {
      column.reason = 8U;
    }
    return evaluation;
  }

  const double cut = cfg.numerics.ale.band_ale.estimator_band_cut;
  for (int j = 0; j < n_columns; ++j) {
    bool any_component = false;
    const auto components =
        per_column_components(state, block, j, cut, &any_component);
    if (components.empty()) {
      evaluation.columns[static_cast<std::size_t>(j)].reason =
          any_component ? 2U : 1U;
      continue;
    }
    int selected = -1;
    double best_overlap = 0.0;
    for (int k = 0; k < static_cast<int>(components.size()); ++k) {
      const double overlap = interval_overlap(
          components[static_cast<std::size_t>(k)],
          previous_intervals[static_cast<std::size_t>(j)]);
      if (overlap > best_overlap ||
          (overlap == best_overlap && overlap > 0.0 &&
           (selected < 0 ||
            components[static_cast<std::size_t>(k)].second >
                components[static_cast<std::size_t>(selected)].second))) {
        best_overlap = overlap;
        selected = k;
      }
    }
    if (best_overlap == 0.0) {
      bool hold_row_in_window = false;
      bool fewer_than_3_rings = false;
      for (int k = static_cast<int>(components.size()) - 1;
           k >= 0;
           --k) {
        int candidate_i0 = -1;
        int candidate_i1 = -1;
        const bool holds = component_window_holds(
            state, cfg, block, j,
            components[static_cast<std::size_t>(k)],
            &candidate_i0, &candidate_i1);
        hold_row_in_window = hold_row_in_window || holds;
        fewer_than_3_rings =
            fewer_than_3_rings || candidate_i1 - candidate_i0 + 1 < 3;
        if (!holds && candidate_i1 - candidate_i0 + 1 >= 3) {
          selected = k;
          break;
        }
      }
      if (selected < 0) {
        evaluation.columns[static_cast<std::size_t>(j)].front_inside = true;
        evaluation.columns[static_cast<std::size_t>(j)].reason =
            hold_row_in_window
                ? 7U
                : (fewer_than_3_rings ? 2U : 8U);
        continue;
      }
    }

    auto& column = evaluation.columns[static_cast<std::size_t>(j)];
    const auto component = components[static_cast<std::size_t>(selected)];
    column.component_first = component.first;
    column.component_last = component.second;
    const bool holds = component_window_holds(
        state, cfg, block, j, component, &column.i0, &column.i1);
    if (column.i1 - column.i0 + 1 < 3) {
      column.reason = 2U;
      continue;
    }
    column.well_formed = true;
    valid[static_cast<std::size_t>(j)] = 1U;
    previous_intervals[static_cast<std::size_t>(j)] =
        {{component.first, component.second}};
    (void)holds;
  }

  const std::vector<std::uint8_t> filter_mask = valid;
  regularize_column_edge(
      cfg, band, block, true, node_r, node_z,
      evaluation.columns, filter_mask, valid);
  regularize_column_edge(
      cfg, band, block, false, node_r, node_z,
      evaluation.columns, filter_mask, valid);

  for (int j = 0; j < n_columns; ++j) {
    auto& column = evaluation.columns[static_cast<std::size_t>(j)];
    if (valid[static_cast<std::size_t>(j)] == 0U ||
        column.i1 - column.i0 + 1 < 3) {
      if (column.reason == 0U) {
        column.reason =
            column.i1 - column.i0 + 1 < 3 ? 2U : 8U;
      }
      column.well_formed = false;
      column.i0 = -1;
      column.i1 = -1;
      continue;
    }
    std::vector<double> spacing;
    spacing.reserve(static_cast<std::size_t>(column.i1 - column.i0));
    bool spacing_valid = true;
    for (int ring = column.i0; ring < column.i1; ++ring) {
      const double h =
          column_ring_radius(band, ring + 1, j, node_r, node_z) -
          column_ring_radius(band, ring, j, node_r, node_z);
      if (!(std::isfinite(h) && h > 0.0)) {
        spacing_valid = false;
        break;
      }
      spacing.push_back(h);
    }
    if (!spacing_valid || spacing.empty()) {
      column.well_formed = false;
      column.reason = 8U;
      column.i0 = -1;
      column.i1 = -1;
      continue;
    }
    const double median = median_value(spacing);
    column.eta =
        *std::max_element(spacing.begin(), spacing.end()) / median;
    if (!std::isfinite(column.eta)) {
      column.well_formed = false;
      column.reason = 8U;
      column.i0 = -1;
      column.i1 = -1;
    }
  }

  std::vector<std::pair<double, int>> eta_samples;
  int well_formed = 0;
  for (int j = 0; j < n_columns; ++j) {
    const auto& column = evaluation.columns[static_cast<std::size_t>(j)];
    if (column.front_inside) {
      ++evaluation.held_columns;
    }
    if (!column.well_formed) {
      continue;
    }
    ++well_formed;
    eta_samples.emplace_back(column.eta, j);
    bool column_holds = false;
    for (int row = column.i0; row < column.i1; ++row) {
      const int cell = block.cell_begin + row * block.n_j_cells + j;
      if (state.refine_error[static_cast<std::size_t>(cell)] >=
          cfg.numerics.ale.band_ale.estimator_band_shock_hold) {
        column_holds = true;
        break;
      }
    }
    if (column_holds) {
      ++evaluation.held_columns;
    }
  }
  evaluation.coverage =
      static_cast<double>(well_formed) / static_cast<double>(n_columns);
  evaluation.hold = evaluation.held_columns > 0;
  if (!eta_samples.empty()) {
    std::stable_sort(
        eta_samples.begin(), eta_samples.end(),
        [](const auto& lhs, const auto& rhs) {
          return lhs.first < rhs.first ||
                 (lhs.first == rhs.first && lhs.second < rhs.second);
        });
    const std::size_t rank =
        static_cast<std::size_t>(
            std::ceil(0.90 * static_cast<double>(eta_samples.size()))) - 1U;
    evaluation.eta_p = eta_samples[rank].first;
  }
  return evaluation;
}

PcEvaluation evaluate_estimator_columns_phase_b(
    const core::State& state,
    const core::Config& cfg,
    BandState& band,
    std::vector<std::array<int, 2>>& previous_intervals,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  PcEvaluation evaluation = evaluate_estimator_columns(
      state, cfg, band, previous_intervals, node_r, node_z);
  evaluation.phase_b = true;
  const auto& topology = *state.mesh.topo.multiblock;
  const auto& block =
      topology.blocks[static_cast<std::size_t>(band.block_id)];
  const int n_rows = block.n_i_cells;
  const int n_columns = block.n_j_cells;
  const std::size_t n_cells =
      static_cast<std::size_t>(state.mesh.topo.n_cells);
  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  evaluation.classification.assign(
      static_cast<std::size_t>(n_columns), kPcClean);
  evaluation.tube_i0.assign(static_cast<std::size_t>(n_columns), -1);
  evaluation.tube_i1.assign(static_cast<std::size_t>(n_columns), -1);
  evaluation.patch_eligible.assign(
      static_cast<std::size_t>(n_columns), 0U);
  if (band.pc_ridge_history.size() !=
          static_cast<std::size_t>(n_columns) ||
      band.pc_ridge_history_count.size() !=
          static_cast<std::size_t>(n_columns)) {
    band.pc_ridge_history.assign(
        static_cast<std::size_t>(n_columns), {});
    band.pc_ridge_history_count.assign(
        static_cast<std::size_t>(n_columns), 0U);
  }

  const bool field_input_missing =
      state.refine_error.size() != n_cells ||
      state.Pe.size() != n_cells || state.Pi.size() != n_cells ||
      state.rho.size() != n_cells || state.mass.size() != n_cells ||
      state.v_r.size() != n_nodes || state.v_z.size() != n_nodes ||
      state.mesh.cell_centroid_r.size() != n_cells ||
      state.mesh.cell_centroid_z.size() != n_cells ||
      topology.cell_node_csr_offsets.size() != n_cells + 1U;
  bool missing_input = field_input_missing;
  std::vector<double> pressure_e;
  std::vector<double> pressure_i;
  std::vector<double> density;
  std::vector<double> mass;
  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  if (!field_input_missing) {
    state.Pe.copy_to_host(pressure_e);
    state.Pi.copy_to_host(pressure_i);
    state.rho.copy_to_host(density);
    state.mass.copy_to_host(mass);
    state.v_r.copy_to_host(velocity_r);
    state.v_z.copy_to_host(velocity_z);
  }

  std::vector<double> radial_velocity(
      static_cast<std::size_t>(block.cell_count),
      std::numeric_limits<double>::quiet_NaN());
  if (!field_input_missing) {
    for (int local_cell = 0; local_cell < block.cell_count; ++local_cell) {
      const int cell = block.cell_begin + local_cell;
      radial_velocity[static_cast<std::size_t>(local_cell)] =
          cell_radial_velocity(
              topology, state.mesh, cell, velocity_r, velocity_z);
    }
  }

  std::vector<std::array<int, 2>> traveling_components(
      static_cast<std::size_t>(n_columns),
      std::array<int, 2>{{-1, -1}});
  const double log_p_travel = std::log(1.30);
  const double log_rho_travel = std::log(1.15);
  const double log_p_persistent = std::log(1.10);
  for (int j = 0; j < n_columns; ++j) {
    if (field_input_missing) {
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcAmbiguous;
      reset_ridge_history_column(band, j);
      continue;
    }

    bool column_missing = false;
    std::vector<std::pair<double, int>> radial_profile;
    radial_profile.reserve(static_cast<std::size_t>(n_rows));
    for (int row = 0; row < n_rows; ++row) {
      const int cell = block.cell_begin + row * n_columns + j;
      const double radius = std::hypot(
          state.mesh.cell_centroid_r[static_cast<std::size_t>(cell)],
          state.mesh.cell_centroid_z[static_cast<std::size_t>(cell)]);
      const double pressure = pressure_e[static_cast<std::size_t>(cell)] +
                              pressure_i[static_cast<std::size_t>(cell)];
      if (!(std::isfinite(radius) && std::isfinite(pressure) &&
            pressure > 0.0)) {
        column_missing = true;
        break;
      }
      radial_profile.emplace_back(radius, row);
    }
    if (column_missing || radial_profile.size() < 2U) {
      missing_input = true;
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcAmbiguous;
      reset_ridge_history_column(band, j);
      continue;
    }
    std::stable_sort(
        radial_profile.begin(), radial_profile.end(),
        [](const auto& lhs, const auto& rhs) {
          return lhs.first < rhs.first ||
                 (lhs.first == rhs.first && lhs.second < rhs.second);
        });
    int ridge_pair = -1;
    double maximum_gradient = -1.0;
    for (int pair = 0;
         pair + 1 < static_cast<int>(radial_profile.size());
         ++pair) {
      const int cell0 =
          block.cell_begin + radial_profile[static_cast<std::size_t>(pair)].second *
                                 n_columns + j;
      const int cell1 = block.cell_begin +
                        radial_profile[static_cast<std::size_t>(pair + 1)].second *
                            n_columns +
                        j;
      const double p0 = pressure_e[static_cast<std::size_t>(cell0)] +
                        pressure_i[static_cast<std::size_t>(cell0)];
      const double p1 = pressure_e[static_cast<std::size_t>(cell1)] +
                        pressure_i[static_cast<std::size_t>(cell1)];
      const double dr =
          radial_profile[static_cast<std::size_t>(pair + 1)].first -
          radial_profile[static_cast<std::size_t>(pair)].first;
      if (!(std::isfinite(dr) && dr > 0.0)) {
        column_missing = true;
        break;
      }
      const double gradient = std::abs(std::log(p1) - std::log(p0)) / dr;
      if (gradient > maximum_gradient) {
        maximum_gradient = gradient;
        ridge_pair = pair;
      }
    }
    if (column_missing || ridge_pair < 0) {
      missing_input = true;
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcAmbiguous;
      reset_ridge_history_column(band, j);
      continue;
    }
    const int ridge_row =
        radial_profile[static_cast<std::size_t>(ridge_pair)].second;
    const double ridge_radius = 0.5 * (
        radial_profile[static_cast<std::size_t>(ridge_pair)].first +
        radial_profile[static_cast<std::size_t>(ridge_pair + 1)].first);

    bool any_component = false;
    const auto components = per_column_components(
        state, block, j,
        cfg.numerics.ale.band_ale.estimator_band_cut,
        &any_component);
    int ridge_component = -1;
    for (int component = 0;
         component < static_cast<int>(components.size());
         ++component) {
      const auto interval = components[static_cast<std::size_t>(component)];
      if (ridge_row >= interval.first && ridge_row <= interval.second) {
        ridge_component = component;
        break;
      }
    }
    if (ridge_component < 0) {
      evaluation.classification[static_cast<std::size_t>(j)] = kPcClean;
      reset_ridge_history_column(band, j);
      continue;
    }
    const auto component =
        components[static_cast<std::size_t>(ridge_component)];

    std::vector<double> p_inner;
    std::vector<double> p_outer;
    std::vector<double> rho_inner;
    std::vector<double> rho_outer;
    std::vector<double> u_inner;
    std::vector<double> u_outer;
    for (int row = std::max(0, component.first - 4);
         row < component.first;
         ++row) {
      const int cell = block.cell_begin + row * n_columns + j;
      const double pressure = pressure_e[static_cast<std::size_t>(cell)] +
                              pressure_i[static_cast<std::size_t>(cell)];
      const double rho = density[static_cast<std::size_t>(cell)];
      const double velocity = radial_velocity[
          static_cast<std::size_t>(cell - block.cell_begin)];
      if (!(std::isfinite(pressure) && pressure > 0.0 &&
            std::isfinite(rho) && rho > 0.0 &&
            std::isfinite(velocity))) {
        column_missing = true;
        break;
      }
      p_inner.push_back(pressure);
      rho_inner.push_back(rho);
      u_inner.push_back(velocity);
    }
    for (int row = component.second + 1;
         !column_missing &&
         row <= std::min(n_rows - 1, component.second + 4);
         ++row) {
      const int cell = block.cell_begin + row * n_columns + j;
      const double pressure = pressure_e[static_cast<std::size_t>(cell)] +
                              pressure_i[static_cast<std::size_t>(cell)];
      const double rho = density[static_cast<std::size_t>(cell)];
      const double velocity = radial_velocity[
          static_cast<std::size_t>(cell - block.cell_begin)];
      if (!(std::isfinite(pressure) && pressure > 0.0 &&
            std::isfinite(rho) && rho > 0.0 &&
            std::isfinite(velocity))) {
        column_missing = true;
        break;
      }
      p_outer.push_back(pressure);
      rho_outer.push_back(rho);
      u_outer.push_back(velocity);
    }
    if (column_missing || p_inner.size() < 2U || p_outer.size() < 2U) {
      missing_input = true;
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcAmbiguous;
      reset_ridge_history_column(band, j);
      continue;
    }
    const double plateau_p_inner = median_value(p_inner);
    const double plateau_p_outer = median_value(p_outer);
    const double plateau_rho_inner = median_value(rho_inner);
    const double plateau_rho_outer = median_value(rho_outer);
    const double plateau_u_inner = median_value(u_inner);
    const double plateau_u_outer = median_value(u_outer);
    if (!(std::isfinite(plateau_p_inner) && plateau_p_inner > 0.0 &&
          std::isfinite(plateau_p_outer) && plateau_p_outer > 0.0 &&
          std::isfinite(plateau_rho_inner) && plateau_rho_inner > 0.0 &&
          std::isfinite(plateau_rho_outer) && plateau_rho_outer > 0.0 &&
          std::isfinite(plateau_u_inner) &&
          std::isfinite(plateau_u_outer))) {
      missing_input = true;
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcAmbiguous;
      reset_ridge_history_column(band, j);
      continue;
    }

    double total_mass = 0.0;
    double cumulative_mass = 0.0;
    double ridge_mass = 0.0;
    for (int row = 0; row < n_rows; ++row) {
      const int cell = block.cell_begin + row * n_columns + j;
      const double cell_mass = mass[static_cast<std::size_t>(cell)];
      if (!(std::isfinite(cell_mass) && cell_mass >= 0.0)) {
        column_missing = true;
        break;
      }
      total_mass += cell_mass;
      if (row <= ridge_row) {
        cumulative_mass += cell_mass;
      }
      if (row == ridge_row) {
        ridge_mass = cell_mass;
      }
    }
    if (column_missing ||
        !(std::isfinite(total_mass) && total_mass > 0.0)) {
      missing_input = true;
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcAmbiguous;
      reset_ridge_history_column(band, j);
      continue;
    }

    double maximum_error = 0.0;
    for (int row = component.first; row <= component.second; ++row) {
      const int cell = block.cell_begin + row * n_columns + j;
      maximum_error = std::max(
          maximum_error,
          state.refine_error[static_cast<std::size_t>(cell)]);
    }
    const double pressure_jump =
        std::abs(std::log(plateau_p_outer / plateau_p_inner));
    const double mass_coordinate = cumulative_mass / total_mass;
    const double ridge_mass_fraction = ridge_mass / total_mass;
    push_ridge_sample(
        band, j,
        PcRidgeSample{
            state.t,
            ridge_radius,
            mass_coordinate,
            maximum_error,
            pressure_jump,
            true});

    double speed = 0.0;
    if (!ridge_speed(band, j, &speed)) {
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcAmbiguous;
      continue;
    }
    const double pressure_scale =
        std::max(std::abs(plateau_p_inner), std::abs(plateau_p_outer));
    if (std::abs(plateau_p_inner - plateau_p_outer) <=
        1.0e-12 * pressure_scale) {
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcAmbiguous;
      continue;
    }

    const bool inner_is_upstream = plateau_p_inner < plateau_p_outer;
    const double pressure_upstream =
        inner_is_upstream ? plateau_p_inner : plateau_p_outer;
    const double density_upstream =
        inner_is_upstream ? plateau_rho_inner : plateau_rho_outer;
    const double density_downstream =
        inner_is_upstream ? plateau_rho_outer : plateau_rho_inner;
    const double velocity_upstream =
        inner_is_upstream ? plateau_u_inner : plateau_u_outer;
    const double velocity_downstream =
        inner_is_upstream ? plateau_u_outer : plateau_u_inner;
    const double density_jump =
        std::abs(std::log(plateau_rho_outer / plateau_rho_inner));
    // Ideal-gas estimate is sufficient for the M_rel >= 0.5 discriminator.
    const double sound_speed = std::sqrt(
        (5.0 / 3.0) * pressure_upstream / density_upstream);
    const double relative_mach =
        std::abs(speed - velocity_upstream) / sound_speed;
    const double upstream_flux =
        density_upstream * (speed - velocity_upstream);
    const double downstream_flux =
        density_downstream * (speed - velocity_downstream);
    const double mass_flux_residual =
        std::abs(upstream_flux - downstream_flux) /
        (std::abs(upstream_flux) + std::abs(downstream_flux) + 1.0e-300);
    if (!(std::isfinite(relative_mach) &&
          std::isfinite(mass_flux_residual))) {
      missing_input = true;
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcAmbiguous;
      reset_ridge_history_column(band, j);
      continue;
    }

    int traveling_tests = 0;
    traveling_tests += pressure_jump >= log_p_travel ? 1 : 0;
    traveling_tests += density_jump >= log_rho_travel ? 1 : 0;
    traveling_tests += relative_mach >= 0.50 ? 1 : 0;
    traveling_tests += mass_flux_residual <= 0.30 ? 1 : 0;
    const bool traveling =
        maximum_error >= 0.90 ||
        (maximum_error >= 0.85 && traveling_tests >= 3);
    if (traveling) {
      evaluation.classification[static_cast<std::size_t>(j)] =
          kPcTraveling;
      traveling_components[static_cast<std::size_t>(j)] =
          {{component.first, component.second}};
      continue;
    }

    bool persistent = false;
    const auto history_count =
        band.pc_ridge_history_count[static_cast<std::size_t>(j)];
    if (history_count == 4U && relative_mach <= 0.15) {
      persistent = true;
      const auto& history =
          band.pc_ridge_history[static_cast<std::size_t>(j)];
      for (const auto& sample : history) {
        persistent = persistent && sample.valid &&
                     sample.max_error <= 0.86 &&
                     sample.pressure_jump <= log_p_persistent;
      }
      const double drift_rows =
          std::abs(mass_coordinate - history.front().mass_coordinate) /
          std::max(ridge_mass_fraction, 1.0e-300);
      persistent = persistent && std::isfinite(drift_rows) &&
                   drift_rows < 0.5;
    }
    evaluation.classification[static_cast<std::size_t>(j)] =
        persistent ? kPcPersistent : kPcAmbiguous;
  }

  evaluation.ambiguous_columns = static_cast<int>(std::count(
      evaluation.classification.begin(),
      evaluation.classification.end(),
      kPcAmbiguous));
  evaluation.held_columns = 0;
  for (const auto classification : evaluation.classification) {
    if (classification == kPcTraveling ||
        classification == kPcAmbiguous) {
      ++evaluation.held_columns;
    }
  }

  std::vector<std::uint8_t> tube_seed(
      static_cast<std::size_t>(n_rows) *
          static_cast<std::size_t>(n_columns),
      0U);
  if (state.refine_error.size() == n_cells) {
    for (int row = 0; row < n_rows; ++row) {
      for (int j = 0; j < n_columns; ++j) {
        const int cell = block.cell_begin + row * n_columns + j;
        if (state.refine_error[static_cast<std::size_t>(cell)] >=
            cfg.numerics.ale.band_ale.estimator_band_shock_hold) {
          tube_seed[static_cast<std::size_t>(row) *
                        static_cast<std::size_t>(n_columns) +
                    static_cast<std::size_t>(j)] = 1U;
        }
      }
    }
  }
  for (int j = 0; j < n_columns; ++j) {
    const auto interval =
        traveling_components[static_cast<std::size_t>(j)];
    for (int row = interval[0]; row >= 0 && row <= interval[1]; ++row) {
      tube_seed[static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(n_columns) +
                static_cast<std::size_t>(j)] = 1U;
    }
  }
  const auto dilate = [&](const std::vector<std::uint8_t>& source,
                          const int radial_margin,
                          const int angular_margin) {
    std::vector<std::uint8_t> result(source.size(), 0U);
    for (int row = 0; row < n_rows; ++row) {
      for (int j = 0; j < n_columns; ++j) {
        if (source[static_cast<std::size_t>(row) *
                       static_cast<std::size_t>(n_columns) +
                   static_cast<std::size_t>(j)] == 0U) {
          continue;
        }
        for (int target_row = std::max(0, row - radial_margin);
             target_row <= std::min(n_rows - 1, row + radial_margin);
             ++target_row) {
          for (int target_column = std::max(0, j - angular_margin);
               target_column <=
                   std::min(n_columns - 1, j + angular_margin);
               ++target_column) {
            result[static_cast<std::size_t>(target_row) *
                       static_cast<std::size_t>(n_columns) +
                   static_cast<std::size_t>(target_column)] = 1U;
          }
        }
      }
    }
    return result;
  };
  const int base_angular_margin =
      cfg.numerics.ale.band_ale.estimator_band_pc_filter_halfwidth + 2;
  evaluation.tube_mask = dilate(
      tube_seed,
      cfg.numerics.ale.band_ale.estimator_band_pc_tube_dilate_rows,
      base_angular_margin);
  bool tube_touches_axis = false;
  for (int row = 0; row < n_rows; ++row) {
    tube_touches_axis = tube_touches_axis ||
        evaluation.tube_mask[static_cast<std::size_t>(row) *
                                 static_cast<std::size_t>(n_columns)] != 0U ||
        evaluation.tube_mask[static_cast<std::size_t>(row) *
                                 static_cast<std::size_t>(n_columns) +
                             static_cast<std::size_t>(n_columns - 1)] != 0U;
  }
  if (tube_touches_axis &&
      cfg.numerics.ale.band_ale.estimator_band_pc_tube_dilate_cols_extra > 0) {
    evaluation.tube_mask = dilate(
        evaluation.tube_mask,
        0,
        cfg.numerics.ale.band_ale.estimator_band_pc_tube_dilate_cols_extra);
  }
  for (int j = 0; j < n_columns; ++j) {
    for (int row = 0; row < n_rows; ++row) {
      if (evaluation.tube_mask[
              static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(n_columns) +
              static_cast<std::size_t>(j)] == 0U) {
        continue;
      }
      if (evaluation.tube_i0[static_cast<std::size_t>(j)] < 0) {
        evaluation.tube_i0[static_cast<std::size_t>(j)] = row;
      }
      evaluation.tube_i1[static_cast<std::size_t>(j)] = row;
    }
    if (evaluation.tube_i0[static_cast<std::size_t>(j)] >= 0) {
      ++evaluation.tube_columns;
    }
  }

  const double ambiguous_fraction =
      static_cast<double>(evaluation.ambiguous_columns) /
      static_cast<double>(n_columns);
  bool cap_junction = false;
  const int cap_rows = std::min(
      n_rows, cfg.numerics.ale.band_ale.estimator_band_in_rows);
  for (int j = 0; j < n_columns && !cap_junction; ++j) {
    for (int row = 0; row < cap_rows; ++row) {
      if (evaluation.tube_mask[
              static_cast<std::size_t>(row) *
                  static_cast<std::size_t>(n_columns) +
              static_cast<std::size_t>(j)] != 0U) {
        cap_junction = true;
        break;
      }
    }
  }
  if (ambiguous_fraction >
      cfg.numerics.ale.band_ale.estimator_band_pc_ambiguous_hold_fraction) {
    evaluation.escalation_reason = 1U;
  } else if (cap_junction) {
    evaluation.escalation_reason = 2U;
  } else if (missing_input) {
    evaluation.escalation_reason = 3U;
  }
  evaluation.hold = evaluation.escalation_reason != 0U;

  const int angular_margin =
      base_angular_margin +
      (tube_touches_axis
           ? cfg.numerics.ale.band_ale.estimator_band_pc_tube_dilate_cols_extra
           : 0);
  std::vector<std::uint8_t> held_column(
      static_cast<std::size_t>(n_columns), 0U);
  for (int j = 0; j < n_columns; ++j) {
    const auto classification =
        evaluation.classification[static_cast<std::size_t>(j)];
    if (classification != kPcTraveling &&
        classification != kPcAmbiguous) {
      continue;
    }
    for (int blocked = std::max(0, j - angular_margin);
         blocked <= std::min(n_columns - 1, j + angular_margin);
         ++blocked) {
      held_column[static_cast<std::size_t>(blocked)] = 1U;
    }
  }
  std::vector<std::uint8_t> candidate(
      static_cast<std::size_t>(n_columns), 0U);
  for (int j = 0; j < n_columns; ++j) {
    const auto& column = evaluation.columns[static_cast<std::size_t>(j)];
    if (!column.well_formed ||
        held_column[static_cast<std::size_t>(j)] != 0U) {
      continue;
    }
    bool contained = true;
    for (int ring = column.i0 + 1;
         ring < column.i1 && contained;
         ++ring) {
      contained =
          node_incident_cells_disjoint_from_tube(
              evaluation.tube_mask, n_rows, n_columns, ring, j) &&
          node_incident_cells_disjoint_from_tube(
              evaluation.tube_mask, n_rows, n_columns, ring, j + 1);
    }
    if (contained) {
      candidate[static_cast<std::size_t>(j)] = 1U;
    }
  }
  const int minimum_patch_width =
      2 * (cfg.numerics.ale.band_ale.estimator_band_pc_filter_halfwidth + 1) +
      3;
  int patch_begin = 0;
  while (patch_begin < n_columns) {
    if (candidate[static_cast<std::size_t>(patch_begin)] == 0U) {
      ++patch_begin;
      continue;
    }
    int patch_end = patch_begin;
    while (patch_end + 1 < n_columns &&
           candidate[static_cast<std::size_t>(patch_end + 1)] != 0U) {
      ++patch_end;
    }
    if (patch_end - patch_begin + 1 >= minimum_patch_width) {
      ++evaluation.patch_count;
      for (int j = patch_begin; j <= patch_end; ++j) {
        evaluation.patch_eligible[static_cast<std::size_t>(j)] = 1U;
      }
    }
    patch_begin = patch_end + 1;
  }
  if (evaluation.hold) {
    std::fill(
        evaluation.patch_eligible.begin(),
        evaluation.patch_eligible.end(),
        0U);
    evaluation.patch_count = 0;
  }
  return evaluation;
}

static bool estimator_band_window_has_compression(
    const mesh::BlockInfo& block,
    const std::vector<std::vector<int>>& shell_rings,
    const int first_ring,
    const int last_ring,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<double>& velocity_r,
    const std::vector<double>& velocity_z,
    const std::vector<double>& cs_host,
    const double hold_mach) {
  for (int row = first_ring; row < last_ring; ++row) {
    const auto& inner_nodes =
        shell_rings[static_cast<std::size_t>(row)];
    const auto& outer_nodes =
        shell_rings[static_cast<std::size_t>(row + 1)];
    for (int j = 0; j < block.n_j_cells; ++j) {
      const auto column = static_cast<std::size_t>(j);
      const auto inner0 =
          static_cast<std::size_t>(inner_nodes[column]);
      const auto inner1 =
          static_cast<std::size_t>(inner_nodes[column + 1U]);
      const auto outer0 =
          static_cast<std::size_t>(outer_nodes[column]);
      const auto outer1 =
          static_cast<std::size_t>(outer_nodes[column + 1U]);
      const double sr =
          0.5 * (node_r[outer0] + node_r[outer1] -
                 node_r[inner0] - node_r[inner1]);
      const double sz =
          0.5 * (node_z[outer0] + node_z[outer1] -
                 node_z[inner0] - node_z[inner1]);
      const double sl = std::hypot(sr, sz);
      if (sl <= 0.0) {
        continue;
      }
      const double u_in =
          0.5 * (velocity_r[inner0] + velocity_r[inner1]) * sr / sl +
          0.5 * (velocity_z[inner0] + velocity_z[inner1]) * sz / sl;
      const double u_out =
          0.5 * (velocity_r[outer0] + velocity_r[outer1]) * sr / sl +
          0.5 * (velocity_z[outer0] + velocity_z[outer1]) * sz / sl;
      const int cell =
          block.cell_begin + row * block.n_j_cells + j;
      if (cell < 0 || static_cast<std::size_t>(cell) >= cs_host.size()) {
        continue;
      }
      const double cs = cs_host[static_cast<std::size_t>(cell)];
      if (cs <= 0.0) {
        continue;
      }
      const double compression_mach = (u_in - u_out) / cs;
      if (compression_mach > hold_mach) {
        return true;
      }
    }
  }
  return false;
}

LegacyEstimatorDecision evaluate_legacy_estimator_decision(
    const core::State& state,
    const core::Config& cfg,
    const mesh::BlockInfo& block,
    const std::vector<std::vector<int>>& shell_rings) {
  LegacyEstimatorDecision decision;
  if (state.refine_error.size() !=
      static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    return decision;
  }
  std::vector<double> mean(
      static_cast<std::size_t>(block.n_i_cells), 0.0);
  for (int row = 0; row < block.n_i_cells; ++row) {
    for (int j = 0; j < block.n_j_cells; ++j) {
      const int cell = block.cell_begin + row * block.n_j_cells + j;
      mean[static_cast<std::size_t>(row)] +=
          state.refine_error[static_cast<std::size_t>(cell)];
    }
    mean[static_cast<std::size_t>(row)] /=
        static_cast<double>(block.n_j_cells);
  }
  int last_flagged = -1;
  for (int row = block.n_i_cells - 1; row >= 0; --row) {
    if (mean[static_cast<std::size_t>(row)] >=
        cfg.numerics.ale.band_ale.estimator_band_cut) {
      last_flagged = row;
      break;
    }
  }
  if (last_flagged < 0) {
    return decision;
  }
  int first_flagged = last_flagged;
  while (first_flagged > 0 &&
         mean[static_cast<std::size_t>(first_flagged - 1)] >=
             cfg.numerics.ale.band_ale.estimator_band_cut) {
    --first_flagged;
  }
  decision.i0 = std::max(
      0, first_flagged -
             cfg.numerics.ale.band_ale.estimator_band_in_rows);
  decision.i1 = std::min(
      block.n_i_cells -
          cfg.numerics.ale.band_ale.shell_boundary_guard_rows,
      last_flagged +
          cfg.numerics.ale.band_ale.estimator_band_out_rows + 1);
  if (decision.i1 - decision.i0 + 1 < 3) {
    decision.i0 = -1;
    decision.i1 = -1;
    return decision;
  }
  const double maximum = *std::max_element(
      mean.begin() + decision.i0, mean.begin() + decision.i1);
  decision.hold = maximum >=
      cfg.numerics.ale.band_ale.estimator_band_shock_hold;
  if (decision.hold) {
    return decision;
  }
  const double hold_mach =
      cfg.numerics.ale.band_ale.estimator_band_hold_mach;
  if (hold_mach > 0.0) {
    std::vector<double> node_r;
    std::vector<double> node_z;
    std::vector<double> velocity_r;
    std::vector<double> velocity_z;
    std::vector<double> cs_host;
    state.x_r.copy_to_host(node_r);
    state.x_z.copy_to_host(node_z);
    state.v_r.copy_to_host(velocity_r);
    state.v_z.copy_to_host(velocity_z);
    state.cs.copy_to_host(cs_host);
    if (estimator_band_window_has_compression(
            block, shell_rings, decision.i0, decision.i1,
            node_r, node_z, velocity_r, velocity_z, cs_host,
            hold_mach)) {
      decision.hold = true;
    }
  }
  return decision;
}

// Keep the maintenance band on the persistent outer feature: select only the
// outermost connected flagged interval so a traveling shock is never inside
// a respace band after it separates and moves inward.
void update_estimator_band_legacy(const core::State& state,
                                  const core::Config& cfg,
                                  BandState& band) {
  TENRYU_ASSERT(
      band.kind == BandKind::Estimator,
      "band ALE estimator update requires an estimator band");
  BandLatticeView view;
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& topology = *state.mesh.topo.multiblock;
    TENRYU_ASSERT(
        band.block_id >= 0 &&
            band.block_id < static_cast<int>(topology.blocks.size()),
        "band ALE estimator block id is out of range");
    const auto& block =
        topology.blocks[static_cast<std::size_t>(band.block_id)];
    TENRYU_ASSERT(
        block.role == mesh::BlockRole::POLAR_SHELL &&
            band.shell_rings.size() ==
                static_cast<std::size_t>(block.n_i_cells) + 1U &&
            band.shell_initial_area.size() ==
                static_cast<std::size_t>(block.cell_count),
        "band ALE estimator metadata is incomplete");
    view = make_multiblock_band_view(topology, block);
  } else {
    view = make_single_block_band_view(state, cfg);
    TENRYU_ASSERT(
        band.block_id == -1 &&
            band.shell_rings.size() ==
                static_cast<std::size_t>(view.n_layers) + 1U &&
            band.shell_initial_area.size() ==
                static_cast<std::size_t>(view.n_layers) *
                    static_cast<std::size_t>(view.n_across) &&
            band.estimator_polar_geometry == view.polar_geometry &&
            band.estimator_layer_axis_is_j == view.layer_axis_is_j,
        "band ALE estimator metadata is incomplete");
  }

  band.lines.clear();
  band.cells.clear();
  band.pinned_nodes.clear();
  band.shell_min_spacing = std::numeric_limits<double>::infinity();
  if (state.refine_error.size() !=
      static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    return;
  }

  std::vector<double> row_mean_error(
      static_cast<std::size_t>(view.n_layers), 0.0);
  for (int row = 0; row < view.n_layers; ++row) {
    const auto row_index = static_cast<std::size_t>(row);
    for (int across = 0; across < view.n_across; ++across) {
      const int cell = view_cell(view, row, across);
      TENRYU_ASSERT(
          cell >= 0 &&
              static_cast<std::size_t>(cell) < state.refine_error.size() &&
              std::isfinite(state.refine_error[static_cast<std::size_t>(cell)]),
          "band ALE estimator sample is unavailable");
      row_mean_error[row_index] +=
          state.refine_error[static_cast<std::size_t>(cell)];
    }
    row_mean_error[row_index] /= static_cast<double>(view.n_across);
  }

  const double cut = cfg.numerics.ale.band_ale.estimator_band_cut;
  int last_flagged_row = -1;
  for (int row = view.n_layers - 1; row >= 0; --row) {
    if (row_mean_error[static_cast<std::size_t>(row)] >= cut) {
      last_flagged_row = row;
      break;
    }
  }
  if (last_flagged_row < 0) {
    return;
  }
  int first_flagged_row = last_flagged_row;
  while (first_flagged_row > 0 &&
         row_mean_error[static_cast<std::size_t>(first_flagged_row - 1)] >=
             cut) {
    --first_flagged_row;
  }

  const int first_ring = std::max(
      0,
      first_flagged_row -
          cfg.numerics.ale.band_ale.estimator_band_in_rows);
  const int outermost_allowed_ring =
      view.n_layers -
      cfg.numerics.ale.band_ale.shell_boundary_guard_rows;
  const int last_ring = std::min(
      outermost_allowed_ring,
      last_flagged_row +
          cfg.numerics.ale.band_ale.estimator_band_out_rows + 1);
  if (last_ring - first_ring + 1 < 3) {
    return;
  }

  const double max_window_row_mean = *std::max_element(
      row_mean_error.begin() + first_ring,
      row_mean_error.begin() + last_ring);
  const bool window_held =
      max_window_row_mean >=
      cfg.numerics.ale.band_ale.estimator_band_shock_hold;
  {
    static int last_logged_first = -1;
    static int last_logged_last = -1;
    static bool last_logged_held = false;
    if (first_ring != last_logged_first || last_ring != last_logged_last ||
        window_held != last_logged_held) {
      last_logged_first = first_ring;
      last_logged_last = last_ring;
      last_logged_held = window_held;
      std::ostringstream window_log;
      window_log << "[band-ale-est-window] first_ring=" << first_ring
                 << " last_ring=" << last_ring
                 << " first_flagged=" << first_flagged_row
                 << " last_flagged=" << last_flagged_row
                 << std::scientific << std::setprecision(3)
                 << " max_row_mean=" << max_window_row_mean
                 << " held=" << (window_held ? 1 : 0);
      core::log_info(window_log.str());
    }
  }
  // A traveling shock inside the window shows shock-level row means (measured
  // 0.93-0.99) while the persistent feature stays <= ~0.84; maintenance must
  // never respace across a front (A452), so the band holds until the shock
  // leaves.
  if (window_held) {
    return;
  }

  int front_shell_row = -1;
  double front_coverage = 0.0;
  // The tracker getter is multiblock-only until the tracker generalization
  // wave. On single-block it returns false, so the static shock hold governs.
  const bool front_tracked =
      refinement_tracker_front_shell_row(&front_shell_row, &front_coverage);
  const double margin_rows =
      cfg.numerics.ale.band_ale.estimator_band_front_hold_margin_rows;
  const bool front_hold =
      front_tracked &&
      static_cast<double>(front_shell_row) >=
          static_cast<double>(first_ring) - margin_rows &&
      static_cast<double>(front_shell_row) <=
          static_cast<double>(last_ring) + margin_rows;
  if (front_hold) {
    static long long front_hold_logged = -1;
    if (front_hold_logged != static_cast<long long>(first_ring) * 100000LL +
                                 last_ring) {
      front_hold_logged =
          static_cast<long long>(first_ring) * 100000LL + last_ring;
      std::ostringstream hold_log;
      hold_log << "[band-ale-est-window] front_hold=1 front_shell_row="
               << front_shell_row << " first_ring=" << first_ring
               << " last_ring=" << last_ring
               << std::scientific << std::setprecision(3)
               << " coverage=" << front_coverage;
      core::log_info(hold_log.str());
    }
    return;
  }

  const double hold_mach =
      cfg.numerics.ale.band_ale.estimator_band_hold_mach;
  if (hold_mach > 0.0 && state.mesh.topo.multiblock.has_value()) {
    const auto& topology = *state.mesh.topo.multiblock;
    const auto& block =
        mesh::mesh_topo_multiblock_polar_shell_block(topology);
    std::vector<double> node_r;
    std::vector<double> node_z;
    std::vector<double> velocity_r;
    std::vector<double> velocity_z;
    std::vector<double> cs_host;
    state.x_r.copy_to_host(node_r);
    state.x_z.copy_to_host(node_z);
    state.v_r.copy_to_host(velocity_r);
    state.v_z.copy_to_host(velocity_z);
    state.cs.copy_to_host(cs_host);
    if (estimator_band_window_has_compression(
            block, band.shell_rings, first_ring, last_ring,
            node_r, node_z, velocity_r, velocity_z, cs_host,
            hold_mach)) {
      return;
    }
  }

  band.lines.reserve(
      static_cast<std::size_t>(last_ring - first_ring + 1));
  for (int ring = last_ring; ring >= first_ring; --ring) {
    band.lines.push_back(
        band.shell_rings[static_cast<std::size_t>(ring)]);
  }
  std::vector<double> initial_area;
  initial_area.reserve(
      static_cast<std::size_t>(last_ring - first_ring) *
      static_cast<std::size_t>(view.n_across));
  for (int row = first_ring; row < last_ring; ++row) {
    for (int across = 0; across < view.n_across; ++across) {
      const int cell = view_cell(view, row, across);
      band.cells.push_back(cell);
      initial_area.push_back(
          band.shell_initial_area[
              view_local_index(view, row, across)]);
    }
  }
  band.pinned_nodes.assign(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  for (const int node : band.lines.front()) {
    band.pinned_nodes[static_cast<std::size_t>(node)] = 1U;
  }
  for (const int node : band.lines.back()) {
    band.pinned_nodes[static_cast<std::size_t>(node)] = 1U;
  }
  band.d_cells.reset(band.cells.size());
  band.d_cells.copy_from_host(band.cells);
  band.d_initial_area.reset(initial_area.size());
  band.d_initial_area.copy_from_host(initial_area);
}

void finalize_per_column_transaction(const core::Config& cfg,
                                     BandState& band) {
  if (!band.pc_transaction_pending) {
    return;
  }
  if (!band.pc_transaction_rejected) {
    band.pc_last_committed_sigma = band.pc_pending_sigma;
    band.pc_previous_chi = band.pc_pending_chi;
    band.pc_cooldown_events_remaining =
        cfg.numerics.ale.band_ale.estimator_band_pc_cooldown_events;
    if (band.pc_last_committed_sigma <
        cfg.numerics.ale.band_ale.estimator_band_pc_sigma_floor) {
      ++band.skipped_inadmissible;
      band.pc_sigma_starvation_events_remaining =
          cfg.numerics.ale.band_ale.estimator_band_pc_cooldown_events;
      band.pc_previous_chi =
          cfg.numerics.ale.band_ale.estimator_band_pc_chi_step;
    }
  }
  band.pc_transaction_pending = false;
  band.pc_transaction_rejected = false;
  band.pc_pending_sigma = 1.0;
  band.pc_pending_chi = 0.0;
}

void install_per_column_evaluation(const core::State& state,
                                   const core::Config& cfg,
                                   BandState& band,
                                   const PcEvaluation& evaluation) {
  const auto& block = state.mesh.topo.multiblock->blocks[
      static_cast<std::size_t>(band.block_id)];
  const int n_columns = block.n_j_cells;
  const int n_spokes = n_columns + 1;
  band.lines.clear();
  band.cells.clear();
  band.shell_min_spacing = std::numeric_limits<double>::infinity();
  band.pinned_nodes.assign(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  band.pc_i0.assign(static_cast<std::size_t>(n_columns), -1);
  band.pc_i1.assign(static_cast<std::size_t>(n_columns), -1);
  band.pc_well_formed.assign(
      static_cast<std::size_t>(n_columns), 0U);
  band.pc_reason.assign(static_cast<std::size_t>(n_columns), 0U);
  band.pc_classification = evaluation.classification;
  if (band.pc_classification.empty()) {
    band.pc_classification.assign(
        static_cast<std::size_t>(n_columns), kPcClean);
  }
  band.pc_tube_i0 = evaluation.tube_i0;
  band.pc_tube_i1 = evaluation.tube_i1;
  if (band.pc_tube_i0.empty()) {
    band.pc_tube_i0.assign(static_cast<std::size_t>(n_columns), -1);
    band.pc_tube_i1.assign(static_cast<std::size_t>(n_columns), -1);
  }
  band.pc_tube_mask = evaluation.tube_mask;
  band.pc_column_active.assign(
      static_cast<std::size_t>(n_columns), 0U);
  band.pc_eta.assign(static_cast<std::size_t>(n_columns), 0.0);
  band.pc_spoke_inner_nodes.assign(
      static_cast<std::size_t>(n_spokes), -1);
  band.pc_spoke_outer_nodes.assign(
      static_cast<std::size_t>(n_spokes), -1);
  band.pc_spoke_i0.assign(static_cast<std::size_t>(n_spokes), -1);
  band.pc_spoke_i1.assign(static_cast<std::size_t>(n_spokes), -1);
  band.pc_spoke_active.assign(
      static_cast<std::size_t>(n_spokes), 0U);
  band.pc_coverage = evaluation.coverage;
  band.pc_eta_p = evaluation.eta_p;
  band.pc_chi = 0.0;
  band.pc_held_columns = evaluation.held_columns;
  band.pc_active_spokes = 0;
  band.pc_patch_count = evaluation.patch_count;
  band.pc_tube_columns = evaluation.tube_columns;
  band.pc_ambiguous_columns = evaluation.ambiguous_columns;
  band.pc_escalation_reason = evaluation.escalation_reason;
  band.pc_phase_b_active = evaluation.phase_b;
  band.pc_predecision = PcDecision::Pending;
  band.pc_reject_reason.clear();
  for (int j = 0; j < n_columns; ++j) {
    const auto& column = evaluation.columns[static_cast<std::size_t>(j)];
    band.pc_reason[static_cast<std::size_t>(j)] = column.reason;
    if (!column.well_formed) {
      continue;
    }
    band.pc_i0[static_cast<std::size_t>(j)] = column.i0;
    band.pc_i1[static_cast<std::size_t>(j)] = column.i1;
    band.pc_well_formed[static_cast<std::size_t>(j)] = 1U;
    band.pc_eta[static_cast<std::size_t>(j)] = column.eta;
    band.pc_column_active[static_cast<std::size_t>(j)] =
        evaluation.phase_b
            ? evaluation.patch_eligible[static_cast<std::size_t>(j)]
            : 1U;
  }
  if (evaluation.hold) {
    band.pc_predecision = PcDecision::Hold;
  } else if (evaluation.coverage <
             cfg.numerics.ale.band_ale.estimator_band_pc_coverage_min) {
    band.pc_predecision = PcDecision::Noop;
    band.pc_reject_reason = "coverage";
  } else if (evaluation.phase_b && evaluation.patch_count == 0) {
    band.pc_predecision = PcDecision::Noop;
    band.pc_reject_reason = "patch-width";
  }

  if (band.pc_predecision == PcDecision::Pending) {
    for (int spoke = 0; spoke < n_spokes; ++spoke) {
      int i0 = -1;
      int i1 = -1;
      if (spoke == 0) {
        if (band.pc_column_active[0] != 0U) {
          i0 = band.pc_i0[0];
          i1 = band.pc_i1[0];
        }
      } else if (spoke == n_columns) {
        const auto adjacent = static_cast<std::size_t>(n_columns - 1);
        if (band.pc_column_active[adjacent] != 0U) {
          i0 = band.pc_i0[adjacent];
          i1 = band.pc_i1[adjacent];
        }
      } else {
        const auto left = static_cast<std::size_t>(spoke - 1);
        const auto right = static_cast<std::size_t>(spoke);
        if (band.pc_column_active[left] != 0U &&
            band.pc_column_active[right] != 0U) {
          i0 = std::max(band.pc_i0[left], band.pc_i0[right]);
          i1 = std::min(band.pc_i1[left], band.pc_i1[right]);
        }
      }
      if (i0 < 0 || i1 - i0 + 1 < 3) {
        continue;
      }
      const int inner_node =
          band.shell_rings[static_cast<std::size_t>(i0)]
                          [static_cast<std::size_t>(spoke)];
      const int outer_node =
          band.shell_rings[static_cast<std::size_t>(i1)]
                          [static_cast<std::size_t>(spoke)];
      band.pc_spoke_inner_nodes[static_cast<std::size_t>(spoke)] =
          inner_node;
      band.pc_spoke_outer_nodes[static_cast<std::size_t>(spoke)] =
          outer_node;
      band.pc_spoke_i0[static_cast<std::size_t>(spoke)] = i0;
      band.pc_spoke_i1[static_cast<std::size_t>(spoke)] = i1;
      band.pc_spoke_active[static_cast<std::size_t>(spoke)] = 1U;
      band.pinned_nodes[static_cast<std::size_t>(inner_node)] = 1U;
      band.pinned_nodes[static_cast<std::size_t>(outer_node)] = 1U;
      ++band.pc_active_spokes;
    }
    if (band.pc_active_spokes == 0) {
      band.pc_predecision = PcDecision::Noop;
      band.pc_reject_reason = "closure";
    }
  }

  std::vector<double> initial_area;
  if (band.pc_predecision == PcDecision::Pending) {
    for (int j = 0; j < n_columns; ++j) {
      if (band.pc_column_active[static_cast<std::size_t>(j)] == 0U) {
        continue;
      }
      for (int row = band.pc_i0[static_cast<std::size_t>(j)];
           row < band.pc_i1[static_cast<std::size_t>(j)];
           ++row) {
        const int cell = block.cell_begin + row * block.n_j_cells + j;
        band.cells.push_back(cell);
        initial_area.push_back(
            band.shell_initial_area[
                static_cast<std::size_t>(cell - block.cell_begin)]);
      }
    }
  }
  band.d_cells.reset(band.cells.size());
  band.d_cells.copy_from_host(band.cells);
  band.d_initial_area.reset(initial_area.size());
  band.d_initial_area.copy_from_host(initial_area);
}

bool per_column_trigger_condition(const BandState& band,
                                  const core::Config& cfg) {
  if (band.pc_eta_p <
      cfg.numerics.ale.band_ale.estimator_band_eta_on) {
    return false;
  }
  int consecutive = 0;
  for (std::size_t j = 0; j < band.pc_eta.size(); ++j) {
    if (band.pc_column_active[j] != 0U &&
        band.pc_eta[j] >=
            cfg.numerics.ale.band_ale.estimator_band_eta_on) {
      ++consecutive;
      if (consecutive >= 3) {
        return true;
      }
    } else {
      consecutive = 0;
    }
  }
  return false;
}

void update_per_column_state_for_test(BandState& band) {
  band.pc_state_for_test = PerColumnStateForTest{
      band.pc_i0,
      band.pc_i1,
      band.pc_well_formed,
      band.pc_reason,
      band.pc_spoke_active,
      band.pc_classification,
      band.pc_tube_i0,
      band.pc_tube_i1,
      band.pc_patch_count,
      band.pc_escalation_reason,
      band.pc_predecision == PcDecision::Hold,
      band.engaged,
      band.pc_eta_p,
      band.pc_coverage};
}

void update_shadow_trigger(const core::Config& cfg,
                           const PcEvaluation& evaluation,
                           BandState& band) {
  if (evaluation.hold ||
      evaluation.coverage <
          cfg.numerics.ale.band_ale.estimator_band_pc_coverage_min) {
    band.pc_shadow_engaged = false;
    band.pc_shadow_release_evaluations = 0;
    band.pc_shadow_engage_decision = false;
    return;
  }
  int consecutive = 0;
  bool trigger = false;
  for (std::size_t j = 0; j < evaluation.columns.size(); ++j) {
    const auto& column = evaluation.columns[j];
    const bool active = evaluation.phase_b
                            ? evaluation.patch_eligible[j] != 0U
                            : column.well_formed;
    if (active &&
        column.eta >= cfg.numerics.ale.band_ale.estimator_band_eta_on) {
      ++consecutive;
      trigger = trigger || consecutive >= 3;
    } else {
      consecutive = 0;
    }
  }
  trigger = trigger &&
            evaluation.eta_p >=
                cfg.numerics.ale.band_ale.estimator_band_eta_on;
  if (!band.pc_shadow_engaged && trigger) {
    band.pc_shadow_engaged = true;
    band.pc_shadow_release_evaluations = 0;
  } else if (band.pc_shadow_engaged &&
             evaluation.eta_p <=
                 cfg.numerics.ale.band_ale.estimator_band_eta_off) {
    ++band.pc_shadow_release_evaluations;
    if (band.pc_shadow_release_evaluations >= 2) {
      band.pc_shadow_engaged = false;
      band.pc_shadow_release_evaluations = 0;
    }
  } else {
    band.pc_shadow_release_evaluations = 0;
  }
  band.pc_shadow_engage_decision = band.pc_shadow_engaged;
}

void update_estimator_band(const core::State& state,
                           const core::Config& cfg,
                           BandState& band) {
  if (!cfg.numerics.ale.band_ale.estimator_band_per_column) {
    update_estimator_band_legacy(state, cfg, band);
    return;
  }
  const auto& block = state.mesh.topo.multiblock->blocks[
      static_cast<std::size_t>(band.block_id)];
  std::vector<double> current_r;
  std::vector<double> current_z;
  state.x_r.copy_to_host(current_r);
  state.x_z.copy_to_host(current_z);
  if (!cfg.numerics.hydro.pressure_drive_perturbation.enabled) {
    const LegacyEstimatorDecision legacy =
        evaluate_legacy_estimator_decision(
            state, cfg, block, band.shell_rings);
    update_estimator_band_legacy(state, cfg, band);
    const PcEvaluation shadow =
        cfg.numerics.ale.band_ale.estimator_band_pc_phase_b
            ? evaluate_estimator_columns_phase_b(
                  state, cfg, band, band.pc_shadow_previous_intervals,
                  current_r, current_z)
            : evaluate_estimator_columns(
                  state, cfg, band, band.pc_shadow_previous_intervals,
                  current_r, current_z);
    update_shadow_trigger(cfg, shadow, band);
    band.pc_shadow_hold = shadow.hold;
    band.pc_legacy_hold = legacy.hold;
    if (legacy.hold && !shadow.hold) {
      ++band.pc_shadow_hold_xor_legacy_only;
    } else if (!legacy.hold && shadow.hold) {
      ++band.pc_shadow_hold_xor_pc_only;
    }
    band.pc_shadow_first_mismatch = -1;
    for (int j = 0; j < block.n_j_cells; ++j) {
      const auto& column = shadow.columns[static_cast<std::size_t>(j)];
      if (column.well_formed &&
          (column.i0 != legacy.i0 || column.i1 != legacy.i1)) {
        band.pc_shadow_first_mismatch = j;
        break;
      }
    }
    return;
  }

  finalize_per_column_transaction(cfg, band);
  const PcEvaluation evaluation =
      cfg.numerics.ale.band_ale.estimator_band_pc_phase_b
          ? evaluate_estimator_columns_phase_b(
                state, cfg, band, band.pc_previous_intervals,
                current_r, current_z)
          : evaluate_estimator_columns(
                state, cfg, band, band.pc_previous_intervals,
                current_r, current_z);
  install_per_column_evaluation(state, cfg, band, evaluation);
}

struct ClosureCatchmentRow {
  int block_id = -1;
  int row = -1;
  double mean_centroid_radius = 0.0;
  std::vector<int> inner_nodes;
  std::vector<int> outer_nodes;
};

int closure_block_cell_id(const mesh::BlockInfo& block,
                          const int i,
                          const int j) {
  TENRYU_ASSERT(
      i >= 0 && i < block.n_i_cells &&
          j >= 0 && j < block.n_j_cells,
      "closure catchment structured cell index is out of range");
  return block.cell_begin + i * block.n_j_cells + j;
}

bool closure_cell_nodes(const mesh::MultiBlockTopology& topology,
                        const int cell,
                        std::vector<int>* nodes) {
  const int begin = topology.cell_node_csr_offsets[
      static_cast<std::size_t>(cell)];
  const int end = topology.cell_node_csr_offsets[
      static_cast<std::size_t>(cell) + 1U];
  if (end - begin < 4) {
    return false;
  }
  nodes->assign(
      topology.cell_node_csr_indices.begin() + begin,
      topology.cell_node_csr_indices.begin() + end);
  return true;
}

std::vector<int> closure_shared_nodes(
    const std::vector<int>& lhs,
    const std::vector<int>& rhs) {
  std::vector<int> shared;
  for (const int lhs_node : lhs) {
    if (std::find(rhs.begin(), rhs.end(), lhs_node) != rhs.end()) {
      shared.push_back(lhs_node);
    }
  }
  std::sort(shared.begin(), shared.end());
  shared.erase(std::unique(shared.begin(), shared.end()), shared.end());
  return shared;
}

std::vector<int> closure_other_nodes(
    const std::vector<int>& nodes,
    const std::vector<int>& shared) {
  std::vector<int> other;
  for (const int node : nodes) {
    if (std::find(shared.begin(), shared.end(), node) == shared.end()) {
      other.push_back(node);
    }
  }
  std::sort(other.begin(), other.end());
  other.erase(std::unique(other.begin(), other.end()), other.end());
  return other;
}

bool closure_classify_radial_pair(
    const std::vector<int>& pair,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    int* inner,
    int* outer) {
  if (pair.size() != 2U) {
    return false;
  }
  const int first = pair[0];
  const int second = pair[1];
  const double first_radius = std::hypot(
      node_r[static_cast<std::size_t>(first)],
      node_z[static_cast<std::size_t>(first)]);
  const double second_radius = std::hypot(
      node_r[static_cast<std::size_t>(second)],
      node_z[static_cast<std::size_t>(second)]);
  if (!std::isfinite(first_radius) || !std::isfinite(second_radius) ||
      first_radius == second_radius) {
    return false;
  }
  if (first_radius < second_radius) {
    *inner = first;
    *outer = second;
  } else {
    *inner = second;
    *outer = first;
  }
  return true;
}

bool closure_row_node_rings(
    const mesh::MultiBlockTopology& topology,
    const mesh::BlockInfo& block,
    const int row,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    std::vector<int>* inner_nodes,
    std::vector<int>* outer_nodes) {
  if (block.n_j_cells < 2) {
    return false;
  }
  std::vector<std::vector<int>> cell_nodes(
      static_cast<std::size_t>(block.n_j_cells));
  for (int j = 0; j < block.n_j_cells; ++j) {
    if (!closure_cell_nodes(
            topology,
            closure_block_cell_id(block, row, j),
            &cell_nodes[static_cast<std::size_t>(j)])) {
      return false;
    }
  }

  inner_nodes->assign(
      static_cast<std::size_t>(block.n_j_cells) + 1U, -1);
  outer_nodes->assign(
      static_cast<std::size_t>(block.n_j_cells) + 1U, -1);
  const auto assign_spoke = [&](const int spoke,
                                const std::vector<int>& pair) {
    int inner = -1;
    int outer = -1;
    if (!closure_classify_radial_pair(
            pair, node_r, node_z, &inner, &outer)) {
      return false;
    }
    const auto spoke_index = static_cast<std::size_t>(spoke);
    if ((*inner_nodes)[spoke_index] >= 0 &&
        ((*inner_nodes)[spoke_index] != inner ||
         (*outer_nodes)[spoke_index] != outer)) {
      return false;
    }
    (*inner_nodes)[spoke_index] = inner;
    (*outer_nodes)[spoke_index] = outer;
    return true;
  };

  for (int j = 0; j < block.n_j_cells; ++j) {
    const auto& current = cell_nodes[static_cast<std::size_t>(j)];
    const std::vector<int> left =
        j == 0
            ? closure_other_nodes(
                  current,
                  closure_shared_nodes(
                      current, cell_nodes[static_cast<std::size_t>(1)]))
            : closure_shared_nodes(
                  cell_nodes[static_cast<std::size_t>(j - 1)], current);
    const std::vector<int> right =
        j == block.n_j_cells - 1
            ? closure_other_nodes(
                  current,
                  closure_shared_nodes(
                      cell_nodes[static_cast<std::size_t>(j - 1)], current))
            : closure_shared_nodes(
                  current, cell_nodes[static_cast<std::size_t>(j + 1)]);
    if (!assign_spoke(j, left) || !assign_spoke(j + 1, right)) {
      return false;
    }
  }
  return std::none_of(
      inner_nodes->begin(), inner_nodes->end(),
      [](const int node) { return node < 0; });
}

double closure_cell_centroid_radius(
    const mesh::MultiBlockTopology& topology,
    const int cell,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  std::vector<int> nodes;
  if (!closure_cell_nodes(topology, cell, &nodes)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  for (const int node : nodes) {
    centroid_r += node_r[static_cast<std::size_t>(node)];
    centroid_z += node_z[static_cast<std::size_t>(node)];
  }
  const double inverse_count = 1.0 / static_cast<double>(nodes.size());
  return std::hypot(
      inverse_count * centroid_r, inverse_count * centroid_z);
}

void closure_log_hold(const int step,
                      const int rank,
                      const std::string& reason) {
  if (rank == 0) {
    core::log_info(
        "[closure-catchment] step=" + std::to_string(step) +
        " hold=" + reason);
  }
}

bool prepare_closure_catchment_target(
    const core::State& state,
    const core::Config& cfg,
    BandState& band,
    const int rank) {
  const int step = state.step + 1;
  band.cells.clear();
  band.closure_held_columns = 0;
  if (!band.closure_topology_supported) {
    return false;
  }
  if (state.central_pseudo_core.member_cells.empty()) {
    if (!band.closure_no_pool_logged && rank == 0) {
      core::log_info(
          "[closure-catchment] no-op: pooled member set is empty");
      band.closure_no_pool_logged = true;
    }
    return false;
  }

  const auto& topology = *state.mesh.topo.multiblock;
  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);
  const auto& catchment = cfg.numerics.ale.band_ale;

  const auto& core_block = topology.blocks[
      static_cast<std::size_t>(band.closure_core_block_id)];
  const auto& bridge_block = topology.blocks[
      static_cast<std::size_t>(band.closure_bridge_block_id)];
  const int member_rows = state.central_pseudo_core.member_ring_count;
  const int total_rows = core_block.n_i_cells + bridge_block.n_i_cells;
  if (member_rows < 0 || member_rows >= total_rows) {
    closure_log_hold(step, rank, "pooled-row-count-out-of-range");
    return false;
  }
  const int core_first = std::min(member_rows, core_block.n_i_cells);
  const int bridge_first = std::max(0, member_rows - core_block.n_i_cells);

  std::vector<ClosureCatchmentRow> rows;
  rows.reserve(static_cast<std::size_t>(total_rows - member_rows));
  const auto append_rows = [&](const int block_id, const int first_row) {
    const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
    for (int i = first_row; i < block.n_i_cells; ++i) {
      ClosureCatchmentRow row;
      row.block_id = block_id;
      row.row = i;
      for (int j = 0; j < block.n_j_cells; ++j) {
        row.mean_centroid_radius += closure_cell_centroid_radius(
            topology, closure_block_cell_id(block, i, j), node_r, node_z);
      }
      row.mean_centroid_radius /= static_cast<double>(block.n_j_cells);
      rows.push_back(std::move(row));
    }
  };
  append_rows(band.closure_core_block_id, core_first);
  append_rows(band.closure_bridge_block_id, bridge_first);
  if (std::any_of(
          rows.begin(), rows.end(),
          [](const ClosureCatchmentRow& row) {
            return !std::isfinite(row.mean_centroid_radius);
          })) {
    closure_log_hold(step, rank, "nonfinite-row-centroid-radius");
    return false;
  }
  std::sort(
      rows.begin(), rows.end(),
      [](const ClosureCatchmentRow& lhs, const ClosureCatchmentRow& rhs) {
        if (lhs.mean_centroid_radius != rhs.mean_centroid_radius) {
          return lhs.mean_centroid_radius < rhs.mean_centroid_radius;
        }
        if (lhs.block_id != rhs.block_id) {
          return lhs.block_id < rhs.block_id;
        }
        return lhs.row < rhs.row;
      });
  rows.erase(
      std::lower_bound(
          rows.begin(), rows.end(),
          catchment.closure_catchment_s_protect_cm,
          [](const ClosureCatchmentRow& row, const double radius) {
            return row.mean_centroid_radius < radius;
          }),
      rows.end());

  if (rows.size() < 8U) {
    band.closure_held_columns = band.closure_columns;
    closure_log_hold(step, rank, "fewer-than-eight-catchment-rows");
    return false;
  }

  for (auto& row : rows) {
    const auto& block = topology.blocks[
        static_cast<std::size_t>(row.block_id)];
    if (!std::isfinite(row.mean_centroid_radius) ||
        !closure_row_node_rings(
            topology, block, row.row, node_r, node_z,
            &row.inner_nodes, &row.outer_nodes)) {
      closure_log_hold(step, rank, "structured-row-classification-failed");
      return false;
    }
  }
  for (std::size_t q = 0; q + 1U < rows.size(); ++q) {
    if (rows[q].outer_nodes != rows[q + 1U].inner_nodes) {
      closure_log_hold(step, rank, "radial-row-continuity-failed");
      return false;
    }
  }

  if (catchment.closure_catchment_forced_active &&
      !band.closure_armed) {
    const auto& innermost_row = rows.front();
    double row_max_speed = 0.0;
    for (const int node : innermost_row.inner_nodes) {
      const auto node_index = static_cast<std::size_t>(node);
      row_max_speed = std::max(
          row_max_speed,
          std::hypot(velocity_r[node_index], velocity_z[node_index]));
    }
    for (const int node : innermost_row.outer_nodes) {
      const auto node_index = static_cast<std::size_t>(node);
      row_max_speed = std::max(
          row_max_speed,
          std::hypot(velocity_r[node_index], velocity_z[node_index]));
    }
    if (row_max_speed >= kClosureCatchmentArmSpeed) {
      band.closure_armed = true;
    } else {
      if (!band.closure_arm_hold_logged && rank == 0) {
        core::log_info(
            "[closure-catchment] holding until piston-adjacent flow arrives");
        band.closure_arm_hold_logged = true;
      }
      return false;
    }
  }

  const std::size_t n_collar =
      std::max<std::size_t>(4U, (rows.size() + 3U) / 4U);
  std::size_t full_rows = rows.size() - n_collar;
  for (std::size_t q = 0; q < full_rows; ++q) {
    if (rows[q].mean_centroid_radius >
        catchment.closure_catchment_s_catch_cm) {
      full_rows = q;
      break;
    }
  }
  for (const auto& row : rows) {
    const auto& block = topology.blocks[
        static_cast<std::size_t>(row.block_id)];
    for (int j = 0; j < block.n_j_cells; ++j) {
      band.cells.push_back(closure_block_cell_id(block, row.row, j));
    }
  }
  if (full_rows == 0U) {
    closure_log_hold(step, rank, "full-respace-zone-is-empty-or-unanchored");
    return false;
  }

  const int n_columns = band.closure_columns;
  const std::size_t n_spokes = static_cast<std::size_t>(n_columns) + 1U;
  std::vector<std::vector<int>> spoke_nodes(n_spokes);
  std::vector<std::vector<double>> current_spacings(n_spokes);
  for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
    std::vector<int> radial_chain;
    radial_chain.reserve(rows.size() + 1U);
    radial_chain.push_back(rows.front().inner_nodes[spoke]);
    for (const auto& row : rows) {
      radial_chain.push_back(row.outer_nodes[spoke]);
    }
    spoke_nodes[spoke] = std::move(radial_chain);
    auto& spacings = current_spacings[spoke];
    spacings.reserve(rows.size());
    for (std::size_t q = 0; q < rows.size(); ++q) {
      const int inner_node = spoke_nodes[spoke][q];
      const int outer_node = spoke_nodes[spoke][q + 1U];
      const double inner_radius = std::hypot(
          node_r[static_cast<std::size_t>(inner_node)],
          node_z[static_cast<std::size_t>(inner_node)]);
      const double outer_radius = std::hypot(
          node_r[static_cast<std::size_t>(outer_node)],
          node_z[static_cast<std::size_t>(outer_node)]);
      const double spacing = outer_radius - inner_radius;
      if (!std::isfinite(spacing) || !(spacing > 0.0)) {
        closure_log_hold(step, rank, "nonpositive-radial-spacing");
        return false;
      }
      spacings.push_back(spacing);
    }
  }
  band.closure_min_h = std::numeric_limits<double>::infinity();
  for (const auto& spacings : current_spacings) {
    for (const double spacing : spacings) {
      band.closure_min_h = std::min(band.closure_min_h, spacing);
    }
  }

  const std::size_t n_full = full_rows;
  const bool record_birth_on_engagement = !band.closure_birth_recorded;
  std::vector<std::vector<double>> birth_spacings(n_spokes);
  if (record_birth_on_engagement) {
    birth_spacings = current_spacings;
  } else {
    if (band.closure_birth_spacings.size() != n_spokes ||
        band.closure_birth_outer_nodes.size() != n_spokes) {
      closure_log_hold(step, rank, "birth-spoke-count-mismatch");
      return false;
    }
    for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
      auto& birth = band.closure_birth_spacings[spoke];
      auto& birth_outer = band.closure_birth_outer_nodes[spoke];
      if (birth.size() != birth_outer.size()) {
        closure_log_hold(step, rank, "birth-node-map-size-mismatch");
        return false;
      }
      auto& selected = birth_spacings[spoke];
      selected.reserve(current_spacings[spoke].size());
      for (std::size_t q = 0; q < current_spacings[spoke].size(); ++q) {
        const int outer_node = spoke_nodes[spoke][q + 1U];
        const auto match = std::find(
            birth_outer.begin(), birth_outer.end(), outer_node);
        if (match == birth_outer.end()) {
          birth_outer.push_back(outer_node);
          birth.push_back(current_spacings[spoke][q]);
          selected.push_back(current_spacings[spoke][q]);
          continue;
        }
        selected.push_back(
            birth[static_cast<std::size_t>(match - birth_outer.begin())]);
      }
    }
  }

  double min_full_birth_spacing = std::numeric_limits<double>::infinity();
  for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
    for (std::size_t q = 0; q < full_rows; ++q) {
      min_full_birth_spacing =
          std::min(min_full_birth_spacing, birth_spacings[spoke][q]);
    }
  }
  const double h_op = std::max(
      catchment.closure_catchment_spacing_floor_cm,
      kClosureCatchmentOperatingFloorFraction * min_full_birth_spacing);

  double min_h0 = std::numeric_limits<double>::infinity();
  int consecutive_compressed_rows = 0;
  for (std::size_t q = 0; q < rows.size(); ++q) {
    double min_spacing = std::numeric_limits<double>::infinity();
    for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
      min_spacing = std::min(min_spacing, current_spacings[spoke][q]);
    }
    if (q == 0U) {
      min_h0 = min_spacing;
    }
    if (!(min_spacing < h_op)) {
      break;
    }
    ++consecutive_compressed_rows;
  }
  const int rows_needed = std::clamp(consecutive_compressed_rows, 0, 4);
  if (rows_needed >= 1 &&
      state.central_pseudo_core.built &&
      state.central_pseudo_core.valid &&
      state.mesh.topo.multiblock.has_value()) {
    band.closure_pending_absorb_rows = rows_needed;
    if (rank == 0) {
      std::ostringstream log;
      log << "[closure-catchment] proactive absorption request rows=+"
          << rows_needed
          << " min_h0=" << std::scientific << std::setprecision(3)
          << min_h0
          << " floor=" << h_op;
      core::log_info(log.str());
    }
    return false;
  }

  if (catchment.closure_catchment_forced_active) {
    // Legacy v1 forced-active target construction. Keep this block isolated
    // from the routine v2 common-mode controller below.
    const double health = h_op > 0.0
                              ? band.closure_min_h / h_op
                              : std::numeric_limits<double>::infinity();
    band.closure_cached_target_r = node_r;
    band.closure_cached_target_z = node_z;
    band.closure_target_nodes.clear();
    std::vector<double> proposed_radius(node_r.size(), 0.0);
    std::vector<double> proposed_theta(node_r.size(), 0.0);
    for (std::size_t node = 0; node < node_r.size(); ++node) {
      proposed_radius[node] = std::hypot(node_r[node], node_z[node]);
      proposed_theta[node] = std::atan2(node_r[node], node_z[node]);
    }
    band.closure_min_eta = std::numeric_limits<double>::infinity();
    std::size_t uniform_fallback_count = 0U;

    for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
      const auto& birth = birth_spacings[spoke];
      const auto& current = current_spacings[spoke];
      const auto& nodes = spoke_nodes[spoke];
      const double theta_ref =
          static_cast<double>(spoke) *
          (CUDART_PI / static_cast<double>(n_columns));
      if (birth.size() != current.size() || birth.size() < full_rows) {
        closure_log_hold(step, rank, "birth-spacing-count-mismatch");
        return false;
      }
      for (std::size_t q = 0; q < current.size(); ++q) {
        band.closure_min_eta =
            std::min(band.closure_min_eta, current[q] / birth[q]);
      }

      std::vector<double> full_birth(
          birth.begin(), birth.begin() + static_cast<std::ptrdiff_t>(full_rows));
      const double piston_radius = std::hypot(
          node_r[static_cast<std::size_t>(nodes.front())],
          node_z[static_cast<std::size_t>(nodes.front())]);
      const double full_outer_radius = std::hypot(
          node_r[static_cast<std::size_t>(nodes[full_rows])],
          node_z[static_cast<std::size_t>(nodes[full_rows])]);
      const double L = full_outer_radius - piston_radius;
      std::vector<double> redistributed;
      if (L < static_cast<double>(n_full) * h_op) {
        redistributed.assign(n_full, L / static_cast<double>(n_full));
        ++uniform_fallback_count;
      } else {
        if (!closure_catchment_redistribute(
                full_birth,
                L,
                h_op,
                catchment.closure_catchment_ratio_max,
                &redistributed)) {
          redistributed.assign(n_full, L / static_cast<double>(n_full));
          ++uniform_fallback_count;
        }
      }

      std::vector<double> respace_radius(nodes.size(), piston_radius);
      for (std::size_t q = 0; q < full_rows; ++q) {
        respace_radius[q + 1U] = respace_radius[q] + redistributed[q];
      }
      double continued_spacing = redistributed.back();
      for (std::size_t q = full_rows; q < current.size(); ++q) {
        const double birth_ratio = birth[q] / birth[q - 1U];
        const double bounded_ratio = std::min(
            std::max(
                birth_ratio,
                1.0 / catchment.closure_catchment_ratio_max),
            catchment.closure_catchment_ratio_max);
        continued_spacing = std::max(
            catchment.closure_catchment_spacing_floor_cm,
            continued_spacing * bounded_ratio);
        respace_radius[q + 1U] =
            respace_radius[q] + continued_spacing;
      }
      for (std::size_t q = 1U; q <= full_rows; ++q) {
        const auto node_index = static_cast<std::size_t>(nodes[q]);
        proposed_radius[node_index] = respace_radius[q];
        proposed_theta[node_index] = theta_ref;
      }
      const std::size_t collar_nodes = current.size() - full_rows;
      for (std::size_t k = 1U; k < collar_nodes; ++k) {
        const std::size_t q = full_rows + k;
        const double zeta =
            static_cast<double>(k) / static_cast<double>(collar_nodes);
        const double blend =
            1.0 - 3.0 * zeta * zeta + 2.0 * zeta * zeta * zeta;
        const int node = nodes[q];
        const double current_radius = std::hypot(
            node_r[static_cast<std::size_t>(node)],
            node_z[static_cast<std::size_t>(node)]);
        const auto node_index = static_cast<std::size_t>(node);
        const double current_theta =
            std::atan2(node_r[node_index], node_z[node_index]);
        proposed_radius[node_index] =
            current_radius + blend * (respace_radius[q] - current_radius);
        proposed_theta[node_index] =
            current_theta + blend * (theta_ref - current_theta);
      }
    }

    band.closure_max_nu = 0.0;
    for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
      const auto& nodes = spoke_nodes[spoke];
      const auto& current = current_spacings[spoke];
      const bool boundary_spoke = spoke == 0U || spoke + 1U == n_spokes;
      for (std::size_t q = 0; q < current.size(); ++q) {
        const int inner_node = nodes[q];
        const int outer_node = nodes[q + 1U];
        const auto inner_index = static_cast<std::size_t>(inner_node);
        const auto outer_index = static_cast<std::size_t>(outer_node);
        const double proposed_inner_r =
            node_r[inner_index] == 0.0 || boundary_spoke
                ? 0.0
                : proposed_radius[inner_index] *
                      std::sin(proposed_theta[inner_index]);
        const double proposed_inner_z =
            proposed_radius[inner_index] *
            std::cos(proposed_theta[inner_index]);
        const double proposed_outer_r =
            node_r[outer_index] == 0.0 || boundary_spoke
                ? 0.0
                : proposed_radius[outer_index] *
                      std::sin(proposed_theta[outer_index]);
        const double proposed_outer_z =
            proposed_radius[outer_index] *
            std::cos(proposed_theta[outer_index]);
        const double displacement = std::max(
            std::hypot(
                proposed_inner_r - node_r[inner_index],
                proposed_inner_z - node_z[inner_index]),
            std::hypot(
                proposed_outer_r - node_r[outer_index],
                proposed_outer_z - node_z[outer_index]));
        band.closure_max_nu =
            std::max(band.closure_max_nu, displacement / current[q]);
      }
    }
    band.closure_prescale =
        band.closure_max_nu > catchment.closure_catchment_nu_max
            ? catchment.closure_catchment_nu_max / band.closure_max_nu
            : 1.0;

    for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
      const bool boundary_spoke = spoke == 0U || spoke + 1U == n_spokes;
      for (const int node : spoke_nodes[spoke]) {
        const auto node_index = static_cast<std::size_t>(node);
        const double current_radius =
            std::hypot(node_r[node_index], node_z[node_index]);
        const double current_theta =
            std::atan2(node_r[node_index], node_z[node_index]);
        const double target_radius =
            current_radius + band.closure_prescale *
                (proposed_radius[node_index] - current_radius);
        const double target_theta = proposed_theta[node_index];
        if (target_radius == current_radius && target_theta == current_theta) {
          continue;
        }
        band.closure_cached_target_r[node_index] =
            node_r[node_index] == 0.0 || boundary_spoke
                ? 0.0
                : target_radius * std::sin(target_theta);
        band.closure_cached_target_z[node_index] =
            target_radius * std::cos(target_theta);
        band.closure_target_nodes.push_back(node);
      }
    }
    std::sort(
        band.closure_target_nodes.begin(), band.closure_target_nodes.end());
    band.closure_target_nodes.erase(
        std::unique(
            band.closure_target_nodes.begin(), band.closure_target_nodes.end()),
        band.closure_target_nodes.end());

    if (record_birth_on_engagement) {
      band.closure_birth_spacings = current_spacings;
      band.closure_birth_outer_nodes.assign(n_spokes, {});
      for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
        auto& birth_outer = band.closure_birth_outer_nodes[spoke];
        birth_outer.reserve(current_spacings[spoke].size());
        for (std::size_t q = 0; q < current_spacings[spoke].size(); ++q) {
          birth_outer.push_back(spoke_nodes[spoke][q + 1U]);
        }
      }
      band.closure_birth_recorded = true;
    }

    if (rank == 0) {
      std::ostringstream log;
      log << "[closure-catchment] step=" << step
          << " cols=" << band.closure_columns
          << " held=" << band.closure_held_columns
          << " uniform_fb=" << uniform_fallback_count
          << " h_op=" << std::scientific << std::setprecision(3)
          << h_op
          << " min_h="
          << band.closure_min_h
          << " health=" << std::scientific << std::setprecision(3)
          << health
          << " engaged="
          << (catchment.closure_catchment_forced_active ||
                      band.closure_engaged_latch
                  ? 1
                  : 0)
          << " shock_hold=0"
          << " min_eta=" << std::fixed << std::setprecision(3)
          << band.closure_min_eta
          << " max_nu=" << band.closure_max_nu
          << " prescale=" << band.closure_prescale
          << " theta_norm=1"
          << " armed=1";
      core::log_info(log.str());
    }
    return true;
  }

  const std::size_t support_core_rows = static_cast<std::size_t>(
      catchment.closure_catchment_support_core_rows);
  const std::size_t support_taper_rows = static_cast<std::size_t>(
      catchment.closure_catchment_support_taper_rows);
  const std::size_t support_requested =
      support_core_rows + support_taper_rows;
  const std::size_t support_rows = std::min(rows.size(), support_requested);
  std::vector<double> support_weight(support_rows, 0.0);
  for (std::size_t k = 0; k < support_rows; ++k) {
    if (k < support_core_rows) {
      support_weight[k] = 1.0;
    } else {
      support_weight[k] = band_smoothstep5(
          static_cast<double>(support_requested - k) /
          static_cast<double>(support_taper_rows));
    }
  }

  const double dtheta_nominal =
      CUDART_PI / static_cast<double>(n_columns);
  const std::size_t metric_rows = std::max<std::size_t>(support_rows, 3U);
  std::vector<std::vector<double>> solid_angle_weight(
      metric_rows,
      std::vector<double>(static_cast<std::size_t>(n_columns), 0.0));
  for (std::size_t k = 0; k < metric_rows; ++k) {
    const auto& row = rows[k];
    for (int j = 0; j < n_columns; ++j) {
      const std::array<int, 4> cell_nodes = {{
          row.inner_nodes[static_cast<std::size_t>(j)],
          row.outer_nodes[static_cast<std::size_t>(j)],
          row.outer_nodes[static_cast<std::size_t>(j) + 1U],
          row.inner_nodes[static_cast<std::size_t>(j) + 1U]}};
      double centroid_r = 0.0;
      double centroid_z = 0.0;
      for (const int node : cell_nodes) {
        centroid_r += node_r[static_cast<std::size_t>(node)];
        centroid_z += node_z[static_cast<std::size_t>(node)];
      }
      centroid_r *= 0.25;
      centroid_z *= 0.25;
      const double theta_center = std::atan2(centroid_r, centroid_z);
      // Simple robust cell solid-angle weight requested for the common-mode
      // lane: sin(theta_center) times the nominal angular pitch.
      solid_angle_weight[k][static_cast<std::size_t>(j)] =
          std::sin(theta_center) * dtheta_nominal;
    }
  }

  std::vector<double> cs_host;
  state.cs.copy_to_host(cs_host);
  double cs_weighted_sum = 0.0;
  double cs_weight_sum = 0.0;
  for (std::size_t k = 0; k < support_rows; ++k) {
    const auto& row = rows[k];
    const auto& block = topology.blocks[
        static_cast<std::size_t>(row.block_id)];
    for (int j = 0; j < n_columns; ++j) {
      const auto column = static_cast<std::size_t>(j);
      const auto cell = static_cast<std::size_t>(
          closure_block_cell_id(block, row.row, j));
      const double weight = solid_angle_weight[k][column];
      cs_weighted_sum += weight * cs_host[cell];
      cs_weight_sum += weight;
    }
  }
  const double cs_bar = cs_weighted_sum / cs_weight_sum;
  const double L_band =
      rows[support_rows - 1U].mean_centroid_radius -
      rows.front().mean_centroid_radius;
  const double tau_p = std::clamp(
      2.0 * L_band / cs_bar, 0.5e-12, 5.0e-12);
  const double dt_eval = 0.25 * tau_p;
  if (state.t < band.closure_next_eval_t) {
    if (step % 500 == 0) {
      closure_log_hold(step, rank, "cadence-idle");
    }
    return false;
  }
  band.closure_next_eval_t = state.t + dt_eval;

  const double tiny = std::numeric_limits<double>::min();
  const double grading_max = 1.5;
  double eta_h_sum = 0.0;
  double eta_h_weight = 0.0;
  double min_h_gamma = std::numeric_limits<double>::infinity();
  for (int j = 0; j < n_columns; ++j) {
    const auto column = static_cast<std::size_t>(j);
    const double h_gamma = current_spacings[column][0U];
    const double h1 = current_spacings[column][1U];
    const double h2 = current_spacings[column][2U];
    const double h_ref_unclamped = h1 * h1 / std::max(h2, tiny);
    const double h_ref = std::clamp(
        h_ref_unclamped, h1 / grading_max, h1 * grading_max);
    const double weight = solid_angle_weight[0U][column];
    eta_h_sum += weight * h_gamma / h_ref;
    eta_h_weight += weight;
    min_h_gamma = std::min(min_h_gamma, h_gamma);
  }
  const double eta_h = eta_h_sum / eta_h_weight;

  std::vector<double> mass_host;
  state.mass.copy_to_host(mass_host);
  double eta_m_sum = 0.0;
  double eta_m_weight = 0.0;
  const auto& interface_row = rows[0U];
  const auto& reference_row = rows[1U];
  const auto& interface_block = topology.blocks[
      static_cast<std::size_t>(interface_row.block_id)];
  const auto& reference_block = topology.blocks[
      static_cast<std::size_t>(reference_row.block_id)];
  for (int j = 0; j < n_columns; ++j) {
    const auto column = static_cast<std::size_t>(j);
    const auto interface_cell = static_cast<std::size_t>(
        closure_block_cell_id(interface_block, interface_row.row, j));
    const auto reference_cell = static_cast<std::size_t>(
        closure_block_cell_id(reference_block, reference_row.row, j));
    const double interface_weight = solid_angle_weight[0U][column];
    const double reference_weight = solid_angle_weight[1U][column];
    // Use the next-row mass per solid angle directly as the comoving mass
    // reference; no density-based reconstruction is needed.
    const double interface_mass_per_omega =
        mass_host[interface_cell] / std::max(interface_weight, tiny);
    const double reference_mass_per_omega =
        mass_host[reference_cell] / std::max(reference_weight, tiny);
    eta_m_sum += interface_weight * interface_mass_per_omega /
                 std::max(reference_mass_per_omega, tiny);
    eta_m_weight += interface_weight;
  }
  const double eta_m = std::clamp(eta_m_sum / eta_m_weight, 0.0, 4.0);

  const double dt_since = state.t - band.closure_prev_eval_t;
  if (band.closure_prev_eval_t >= 0.0 && dt_since > 0.0) {
    const double alpha =
        1.0 - std::exp(-dt_since / (0.25 * tau_p));
    if (eta_h > 0.0 && band.closure_eta_h_prev > 0.0) {
      const double lambda_h_inst =
          -(std::log(eta_h) - std::log(band.closure_eta_h_prev)) /
          dt_since;
      band.closure_lambda_h_ema =
          (1.0 - alpha) * band.closure_lambda_h_ema +
          alpha * lambda_h_inst;
    }
    if (eta_m > 0.0 && band.closure_eta_m_prev > 0.0) {
      const double lambda_m_inst =
          -(std::log(eta_m) - std::log(band.closure_eta_m_prev)) /
          dt_since;
      band.closure_lambda_m_ema =
          (1.0 - alpha) * band.closure_lambda_m_ema +
          alpha * lambda_m_inst;
    }
  }
  band.closure_eta_h_prev = eta_h;
  band.closure_eta_m_prev = eta_m;
  band.closure_prev_eval_t = state.t;

  const double lam_h_full = std::log(
      catchment.closure_catchment_eta_h_arm /
      catchment.closure_catchment_eta_h_full) / tau_p;
  const double eta_h_pred = eta_h * std::exp(
      -std::max(band.closure_lambda_h_ema, 0.0) * tau_p);
  const double a_h = band_activation_up(
                         band.closure_lambda_h_ema, 0.0, lam_h_full) *
                     band_activation_down(
                         eta_h_pred,
                         catchment.closure_catchment_eta_h_arm,
                         catchment.closure_catchment_eta_h_full);
  const double lam_m_full = std::log(
      catchment.closure_catchment_eta_m_arm /
      catchment.closure_catchment_eta_m_full) / tau_p;
  const double eta_m_pred = eta_m * std::exp(
      -std::max(band.closure_lambda_m_ema, 0.0) * tau_p);
  const double a_m = band_activation_up(
                         band.closure_lambda_m_ema, 0.0, lam_m_full) *
                     band_activation_down(
                         eta_m_pred,
                         catchment.closure_catchment_eta_m_arm,
                         catchment.closure_catchment_eta_m_full);
  double a_C = 1.0 - (1.0 - a_h) * (1.0 - a_m);

  const bool hard_signal = min_h_gamma < h_op;
  if (hard_signal) {
    band.closure_rearmed = true;
  } else if (!band.closure_rearmed) {
    if (eta_h < band.closure_rearm_eta_h *
                    (1.0 - catchment.closure_catchment_rearm_drop)) {
      band.closure_rearmed = true;
    } else {
      a_C = 0.0;
    }
  }

  if (catchment.closure_catchment_shock_hold > 0.0) {
    double shock_M = 0.0;
    for (std::size_t k = 0; k < support_rows; ++k) {
      const auto& row = rows[k];
      const auto& block = topology.blocks[
          static_cast<std::size_t>(row.block_id)];
      double mach_eighth_sum = 0.0;
      double mach_weight_sum = 0.0;
      for (int j = 0; j < n_columns; ++j) {
        const auto column = static_cast<std::size_t>(j);
        const auto inner_node = static_cast<std::size_t>(
            row.inner_nodes[column]);
        const auto outer_node = static_cast<std::size_t>(
            row.outer_nodes[column]);
        double sr = node_r[outer_node] - node_r[inner_node];
        double sz = node_z[outer_node] - node_z[inner_node];
        const double spacing = std::hypot(sr, sz);
        if (!(spacing > 0.0)) {
          continue;
        }
        sr /= spacing;
        sz /= spacing;
        const double u_in =
            velocity_r[inner_node] * sr + velocity_z[inner_node] * sz;
        const double u_out =
            velocity_r[outer_node] * sr + velocity_z[outer_node] * sz;
        const auto cell = static_cast<std::size_t>(
            closure_block_cell_id(block, row.row, j));
        const double compression_mach = std::max(
            0.0, (u_in - u_out) / std::max(cs_host[cell], tiny));
        const double weight = solid_angle_weight[k][column];
        mach_eighth_sum +=
            weight * std::pow(compression_mach, 8.0);
        mach_weight_sum += weight;
      }
      const double row_M = std::pow(
          mach_eighth_sum / mach_weight_sum, 1.0 / 8.0);
      shock_M = std::max(shock_M, row_M);
    }
    a_C *= 1.0 - band_activation_up(
                       shock_M,
                       catchment.closure_catchment_shock_hold - 0.05,
                       catchment.closure_catchment_shock_hold + 0.10);
  }

  if (a_C <= 0.0) {
    if (step % 500 == 0) {
      closure_log_hold(step, rank, "routine-inactive");
    }
    return false;
  }

  std::vector<double> Hbar(support_rows, 0.0);
  double Hbar_sum = 0.0;
  for (std::size_t k = 0; k < support_rows; ++k) {
    double weighted_width = 0.0;
    double weight_sum = 0.0;
    for (int j = 0; j < n_columns; ++j) {
      const auto column = static_cast<std::size_t>(j);
      const double weight = solid_angle_weight[k][column];
      weighted_width += weight * current_spacings[column][k];
      weight_sum += weight;
    }
    Hbar[k] = weighted_width / weight_sum;
    Hbar_sum += Hbar[k];
  }
  const double Hbar_uniform =
      Hbar_sum / static_cast<double>(support_rows);
  std::vector<double> Hstar(support_rows, 0.0);
  for (std::size_t k = 0; k < support_rows; ++k) {
    const double reset_pull =
        (1.0 - catchment.closure_catchment_reset_eta) * support_weight[k];
    Hstar[k] = (1.0 - reset_pull) * Hbar[k] +
               reset_pull * Hbar_uniform;
  }
  const std::vector<double> xi = closure_catchment_xi(Hstar);

  band.closure_cached_target_r = node_r;
  band.closure_cached_target_z = node_z;
  band.closure_target_nodes.clear();
  double chi = 0.0;
  for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
    const auto& nodes = spoke_nodes[spoke];
    const auto anchor_a = static_cast<std::size_t>(nodes[0U]);
    const auto anchor_b = static_cast<std::size_t>(nodes[support_rows]);
    for (std::size_t k = 1U; k < support_rows; ++k) {
      const auto node = static_cast<std::size_t>(nodes[k]);
      const double fraction = xi[k - 1U];
      const double star_r =
          (1.0 - fraction) * node_r[anchor_a] +
          fraction * node_r[anchor_b];
      const double star_z =
          (1.0 - fraction) * node_z[anchor_a] +
          fraction * node_z[anchor_b];
      const double target_r =
          node_r[node] + a_C * (star_r - node_r[node]);
      const double target_z =
          node_z[node] + a_C * (star_z - node_z[node]);
      const double displacement =
          std::hypot(target_r - node_r[node], target_z - node_z[node]);
      chi = std::max(chi, displacement / std::max(Hbar_uniform, tiny));
      if (target_r == node_r[node] && target_z == node_z[node]) {
        continue;
      }
      band.closure_cached_target_r[node] = target_r;
      band.closure_cached_target_z[node] = target_z;
      band.closure_target_nodes.push_back(static_cast<int>(node));
    }
  }
  std::sort(
      band.closure_target_nodes.begin(), band.closure_target_nodes.end());
  band.closure_target_nodes.erase(
      std::unique(
          band.closure_target_nodes.begin(),
          band.closure_target_nodes.end()),
      band.closure_target_nodes.end());

  band.closure_accum_chi += chi;
  if (band.closure_accum_chi < catchment.closure_catchment_accum_frac &&
      !hard_signal) {
    if (step % 500 == 0) {
      closure_log_hold(step, rank, "accumulating");
    }
    return false;
  }
  band.closure_accum_chi = 0.0;
  band.closure_rearmed = false;
  band.closure_rearm_eta_h = eta_h;
  band.closure_min_eta = std::min(eta_h, eta_m);
  band.closure_max_nu = chi;
  band.closure_prescale = 1.0;

  if (record_birth_on_engagement) {
    band.closure_birth_spacings = current_spacings;
    band.closure_birth_outer_nodes.assign(n_spokes, {});
    for (std::size_t spoke = 0; spoke < n_spokes; ++spoke) {
      auto& birth_outer = band.closure_birth_outer_nodes[spoke];
      birth_outer.reserve(current_spacings[spoke].size());
      for (std::size_t q = 0; q < current_spacings[spoke].size(); ++q) {
        birth_outer.push_back(spoke_nodes[spoke][q + 1U]);
      }
    }
    band.closure_birth_recorded = true;
  }

  if (rank == 0) {
    std::ostringstream log;
    log << "[closure-catchment] step=" << step
        << " mode=common-p0"
        << " eta_h=" << std::fixed << std::setprecision(3) << eta_h
        << " eta_m=" << eta_m
        << " activation=" << a_C
        << " chi=" << chi
        << " hard=" << (hard_signal ? 1 : 0);
    core::log_info(log.str());
  }
  return true;
}

BandState make_axis_band(const core::State& state,
                         const int block_id,
                         const bool high_side,
                         const std::vector<std::vector<int>>& rings) {
  const auto& topology = *state.mesh.topo.multiblock;
  const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
  TENRYU_ASSERT(
      block.n_j_cells >= 2,
      "band ALE axis bands require at least two cell columns");

  const int first_cell_column = high_side ? block.n_j_cells - 2 : 0;
  BandState band;
  band.kind = BandKind::Axis;
  band.block_id = block_id;
  band.name = "axis" + std::to_string(block_id) +
              (high_side ? "hi" : "lo");
  for (int row = 0; row < block.n_i_cells; ++row) {
    const int row_begin = block.cell_begin + row * block.n_j_cells;
    band.cells.push_back(row_begin + first_cell_column);
    band.cells.push_back(row_begin + first_cell_column + 1);
  }

  const int first_node_column = high_side ? block.n_j_cells - 2 : 0;
  band.lines.resize(3);
  for (int local_column = 0; local_column < 3; ++local_column) {
    auto& line = band.lines[static_cast<std::size_t>(local_column)];
    line.reserve(static_cast<std::size_t>(block.n_i_cells) + 1U);
    const int node_column = first_node_column + local_column;
    for (const auto& ring : rings) {
      TENRYU_ASSERT(
          node_column >= 0 &&
              node_column < static_cast<int>(ring.size()),
          "band ALE axis node column is out of range");
      line.push_back(ring[static_cast<std::size_t>(node_column)]);
    }
  }

  band.pinned_nodes.assign(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  for (const auto& line : band.lines) {
    TENRYU_ASSERT(
        line.size() >= 2U,
        "band ALE axis radial line requires two endpoints");
    band.pinned_nodes[static_cast<std::size_t>(line.front())] = 1U;
    band.pinned_nodes[static_cast<std::size_t>(line.back())] = 1U;
  }
  const std::size_t far_line = high_side ? 0U : 2U;
  for (const int node : band.lines[far_line]) {
    band.pinned_nodes[static_cast<std::size_t>(node)] = 1U;
  }
  return band;
}

BandState make_axis_repair_band(const core::State& state,
                                const int block_id) {
  const auto& topology = *state.mesh.topo.multiblock;
  const auto& block = topology.blocks[static_cast<std::size_t>(block_id)];
  TENRYU_ASSERT(
      block.n_i_cells >= 1 && block.n_j_cells >= 1 &&
          block.cell_count == block.n_i_cells * block.n_j_cells,
      "band ALE axis repair requires a non-empty structured tier block");

  BandState band;
  band.kind = BandKind::AxisRepair;
  band.block_id = block_id;
  band.name = "axis-repair" + std::to_string(block_id);
  for (int row = 0; row < block.n_i_cells; ++row) {
    const std::vector<int> row_cells = block_row_cells(block, row);
    const auto append_cell = [&](const int cell) {
      AxisRepairCell repair;
      repair.cell = cell;
      repair.block_id = block_id;
      repair.row = row;
      repair.row_cells = row_cells;
      band.cells.push_back(cell);
      band.axis_repair_cells.push_back(std::move(repair));
    };
    append_cell(row_cells.front());
    if (row_cells.back() != row_cells.front()) {
      append_cell(row_cells.back());
    }
  }
  TENRYU_ASSERT(
      std::is_sorted(band.cells.begin(), band.cells.end()),
      "band ALE axis repair cells must be ordered by ascending cell id");
  return band;
}

std::vector<int> build_axis_repair_patch_cells(
    const BandState& band,
    const mesh::BlockInfo& block) {
  TENRYU_ASSERT(
      block.n_i_cells >= 1 && block.n_j_cells >= 1 &&
          block.cell_count == block.n_i_cells * block.n_j_cells,
      "band ALE axis repair patch requires a structured tier block");
  std::vector<int> patch_cells;
  for (const auto& repair : band.axis_repair_cells) {
    if (!repair.engaged) {
      continue;
    }
    TENRYU_ASSERT(
        block.n_j_cells >= 2,
        "band ALE axis repair engaged tier block requires an adjacent "
        "angular column");
    const int local_cell = repair.cell - block.cell_begin;
    TENRYU_ASSERT(
        local_cell >= 0 && local_cell < block.cell_count &&
            local_cell / block.n_j_cells == repair.row,
        "band ALE axis repair monitored cell is outside its structured row");
    const int column = local_cell % block.n_j_cells;
    const bool low_side = column == 0;
    TENRYU_ASSERT(
        low_side || column == block.n_j_cells - 1,
        "band ALE axis repair monitored cell is not axis-adjacent");
    const int adjacent_column = low_side ? column + 1 : column - 1;
    const int row_lo =
        std::max(0, repair.row - kAxisRepairPatchRowHalfwidth);
    const int row_hi =
        std::min(
            block.n_i_cells - 1,
            repair.row + kAxisRepairPatchRowHalfwidth);
    for (int row = row_lo; row <= row_hi; ++row) {
      const int row_begin = block.cell_begin + row * block.n_j_cells;
      patch_cells.push_back(row_begin + column);
      patch_cells.push_back(row_begin + adjacent_column);
    }
  }
  std::sort(patch_cells.begin(), patch_cells.end());
  patch_cells.erase(
      std::unique(patch_cells.begin(), patch_cells.end()),
      patch_cells.end());
  return patch_cells;
}

double axis_repair_ring_scale_for_row(const BandState& band,
                                      const int row) {
  for (const auto& repair : band.axis_repair_cells) {
    if (repair.row == row) {
      TENRYU_ASSERT(
          std::isfinite(repair.ring_scale) && repair.ring_scale > 0.0,
          "band ALE axis repair row scale must be finite and positive");
      return repair.ring_scale;
    }
  }
  TENRYU_ASSERT(
      false,
      "band ALE axis repair patch row has no ring-scale monitor");
  return 1.0;
}

void initialize_runtime(core::State& state,
                        const core::Config& cfg,
                        RuntimeState& runtime) {
  const bool has_multiblock = state.mesh.topo.multiblock.has_value();
  const bool single_block_estimator_admitted =
      !has_multiblock &&
      cfg.numerics.ale.band_ale.bands == "estimator" &&
      state.mesh.dim == 2 && state.mesh.topo.nr >= 5 &&
      state.mesh.topo.nz >= 5 &&
      state.mesh.topo.n_cells ==
          state.mesh.topo.nr * state.mesh.topo.nz;
  TENRYU_ASSERT(
      has_multiblock || single_block_estimator_admitted,
      "band ALE requires multiblock topology unless an estimator band is "
      "running on an admitted structured single-block lattice");
  const mesh::MultiBlockTopology* topology =
      has_multiblock ? &*state.mesh.topo.multiblock : nullptr;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(
      state.mesh.cell_area.size() == static_cast<std::size_t>(n_cells),
      "band ALE requires current planar cell areas");
  TENRYU_ASSERT(
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells),
      "band ALE requires per-cell vertex counts");
  TENRYU_ASSERT(
      state.mesh.topo.node_flags.size() == static_cast<std::size_t>(n_nodes),
      "band ALE requires node flags");

  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);

  const bool axis_core_mode_active =
      cfg.numerics.ale.euler_window.enabled &&
      cfg.numerics.ale.euler_window.role == "axis_survival_core";
  TENRYU_ASSERT(
      has_multiblock || !axis_core_mode_active,
      "single-block estimator band must skip multiblock axis-core construction");
  std::vector<std::uint8_t> axis_core_excluded_cells;
  std::vector<std::uint8_t> axis_core_pinned_nodes;
  if (axis_core_mode_active) {
    const auto observation = axis_core_disk_observe(state);
    axis_core_excluded_cells.assign(
        static_cast<std::size_t>(n_cells), 0U);
    axis_core_pinned_nodes.assign(
        static_cast<std::size_t>(n_nodes), 0U);
    const auto mark_cells = [&](const std::vector<int>& cells) {
      for (const int cell : cells) {
        TENRYU_ASSERT(
            cell >= 0 && cell < n_cells,
            "axis-core exclusion cell id is out of range");
        axis_core_excluded_cells[static_cast<std::size_t>(cell)] = 1U;
      }
    };
    mark_cells(observation.plateau_cells);
    mark_cells(observation.feather_cells);
    mark_cells(observation.guard_cells);
    TENRYU_ASSERT(
        topology->cell_node_csr_offsets.size() ==
            static_cast<std::size_t>(n_cells) + 1U,
        "axis-core exclusion requires complete cell-node CSR offsets");
    for (int cell = 0; cell < n_cells; ++cell) {
      if (axis_core_excluded_cells[static_cast<std::size_t>(cell)] == 0U) {
        continue;
      }
      const int begin = topology->cell_node_csr_offsets[
          static_cast<std::size_t>(cell)];
      const int end = topology->cell_node_csr_offsets[
          static_cast<std::size_t>(cell) + 1U];
      TENRYU_ASSERT(
          begin >= 0 && begin <= end &&
              end <= static_cast<int>(
                  topology->cell_node_csr_indices.size()),
          "axis-core exclusion cell-node CSR range is invalid");
      for (int offset = begin; offset < end; ++offset) {
        const int node = topology->cell_node_csr_indices[
            static_cast<std::size_t>(offset)];
        TENRYU_ASSERT(
            node >= 0 && node < n_nodes,
            "axis-core exclusion node id is out of range");
        axis_core_pinned_nodes[static_cast<std::size_t>(node)] = 1U;
      }
    }
  }

  const bool include_belts =
      cfg.numerics.ale.band_ale.bands == "belts" ||
      cfg.numerics.ale.band_ale.bands == "belts_axis" ||
      cfg.numerics.ale.band_ale.bands == "belts_axis_shell";
  const bool include_axis =
      cfg.numerics.ale.band_ale.bands == "axis" ||
      cfg.numerics.ale.band_ale.bands == "belts_axis" ||
      cfg.numerics.ale.band_ale.bands == "belts_axis_shell";
  const bool include_shell =
      cfg.numerics.ale.band_ale.bands == "shell" ||
      cfg.numerics.ale.band_ale.bands == "belts_axis_shell";
  const bool include_estimator =
      cfg.numerics.ale.band_ale.bands == "estimator";
  const bool include_axis_repair =
      cfg.numerics.ale.band_ale.axis_repair_enabled;
  const bool include_closure_catchment =
      cfg.numerics.ale.band_ale.closure_catchment_enabled;
  TENRYU_ASSERT(
      has_multiblock ||
          (include_estimator && !include_belts && !include_axis &&
           !include_shell && !include_axis_repair),
      "single-block band ALE must construct only the estimator band");

  if (include_belts) {
    if (!topology->dendrite_block_rings.empty()) {
      auto bands = make_dendrite_belt_bands(state, node_r, node_z);
      for (auto& band : bands) {
        runtime.bands.push_back(std::move(band));
      }
    } else {
      ale::m1::ring_detail::BlockCornerRoles borrowed_roles;
      const bool have_borrowed_roles =
          detect_borrowed_belt_roles(*topology, borrowed_roles);
      for (int block_id = 0;
           block_id < static_cast<int>(topology->blocks.size());
           ++block_id) {
        if (topology->blocks[static_cast<std::size_t>(block_id)].role ==
            mesh::BlockRole::TRANSITION_BELT) {
          BandState band;
          if (!have_borrowed_roles ||
              !make_belt_band(
                  state,
                  block_id,
                  node_r,
                  node_z,
                  borrowed_roles,
                  band)) {
            ++runtime.skipped_uncrosschecked_belts;
            continue;
          }
          runtime.bands.push_back(std::move(band));
        }
      }
    }
    std::sort(
        runtime.bands.begin(),
        runtime.bands.end(),
        [](const BandState& lhs, const BandState& rhs) {
          if (lhs.sort_radius != rhs.sort_radius) {
            return lhs.sort_radius > rhs.sort_radius;
          }
          return lhs.block_id < rhs.block_id;
        });
    for (std::size_t index = 0; index < runtime.bands.size(); ++index) {
      runtime.bands[index].name = "belt" + std::to_string(index + 1U);
    }
  }

  if (include_axis) {
    for (int block_id = 0;
         block_id < static_cast<int>(topology->blocks.size());
         ++block_id) {
      const auto role =
          topology->blocks[static_cast<std::size_t>(block_id)].role;
      if (role == mesh::BlockRole::POLAR_SHELL &&
          !cfg.numerics.ale.band_ale.axis_shell_block_enabled) {
        ++runtime.skipped_shell_axis_blocks;
        continue;
      }
      if (role != mesh::BlockRole::POLAR_TIER &&
          role != mesh::BlockRole::POLAR_SHELL) {
        continue;
      }
      const auto& block =
          topology->blocks[static_cast<std::size_t>(block_id)];
      // A one-row axis column has only pinned endpoints and no movable
      // interior.
      if (block.n_i_cells < 2) {
        ++runtime.skipped_trivial_axis_blocks;
        continue;
      }
      const auto rings = structured_node_rings(*topology, block);
      if (rings.empty()) {
        ++runtime.skipped_trivial_axis_blocks;
        continue;
      }
      runtime.bands.push_back(
          make_axis_band(state, block_id, false, rings));
      runtime.bands.push_back(
          make_axis_band(state, block_id, true, rings));
    }
  }

  if (include_axis_repair) {
    for (int block_id = 0;
         block_id < static_cast<int>(topology->blocks.size());
         ++block_id) {
      const auto& block =
          topology->blocks[static_cast<std::size_t>(block_id)];
      if (block.role == mesh::BlockRole::POLAR_TIER &&
          block.n_i_cells >= 1) {
        runtime.bands.push_back(
            make_axis_repair_band(state, block_id));
      }
    }

    runtime.axis_repair_initial_area.assign(
        static_cast<std::size_t>(n_cells), 0.0);
    runtime.axis_repair_current_area.assign(
        static_cast<std::size_t>(n_cells), 0.0);
    const int owned_begin = state.owned_cells_begin_or(0);
    const int owned_end = state.owned_cells_end_or(n_cells);
    TENRYU_ASSERT(
        owned_begin >= 0 && owned_begin <= owned_end &&
            owned_end <= n_cells,
        "band ALE axis repair owned cell range is invalid");
    for (int cell = owned_begin; cell < owned_end; ++cell) {
      const double area =
          state.mesh.cell_area[static_cast<std::size_t>(cell)];
      TENRYU_ASSERT(
          std::isfinite(area) && area > 0.0,
          "band ALE axis repair initial area must be finite and positive");
      runtime.axis_repair_initial_area[static_cast<std::size_t>(cell)] =
          area;
    }

    std::vector<double> initial_r;
    state.x_r_initial.copy_to_host(initial_r);
    TENRYU_ASSERT(
        initial_r.size() == static_cast<std::size_t>(n_nodes),
        "band ALE axis repair requires initial radial coordinates");
    runtime.axis_repair_axis_nodes.assign(
        static_cast<std::size_t>(n_nodes), 0U);
    for (int node = 0; node < n_nodes; ++node) {
      if (initial_r[static_cast<std::size_t>(node)] == 0.0) {
        runtime.axis_repair_axis_nodes[static_cast<std::size_t>(node)] =
            1U;
      }
    }
    runtime.axis_repair_node_incident_cells =
        build_node_incident_cells(state);
  }

  if (axis_core_mode_active) {
    std::vector<BandState> surviving_bands;
    surviving_bands.reserve(runtime.bands.size());
    for (auto& band : runtime.bands) {
      const bool intersects_axis_core = std::any_of(
          band.cells.begin(),
          band.cells.end(),
          [&](const int cell) {
            TENRYU_ASSERT(
                cell >= 0 && cell < n_cells,
                "band ALE cell id is out of range");
            return axis_core_excluded_cells[
                       static_cast<std::size_t>(cell)] != 0U;
          });
      if (intersects_axis_core) {
        core::log_info(
            "[band-ale] band " + band.name +
            " suppressed (axis-core exclusion)");
        continue;
      }
      TENRYU_ASSERT(
          band.pinned_nodes.empty() ||
              band.pinned_nodes.size() ==
                  static_cast<std::size_t>(n_nodes),
          "band ALE pinned-node mask size mismatch");
      if (band.pinned_nodes.empty()) {
        band.pinned_nodes.assign(
            static_cast<std::size_t>(n_nodes), 0U);
      }
      for (int node = 0; node < n_nodes; ++node) {
        const auto node_index = static_cast<std::size_t>(node);
        if (axis_core_pinned_nodes[node_index] != 0U) {
          band.pinned_nodes[node_index] = 1U;
        }
      }
      surviving_bands.push_back(std::move(band));
    }
    runtime.bands = std::move(surviving_bands);
  }

  if (include_shell) {
    std::vector<double> initial_r;
    std::vector<double> initial_z;
    state.x_r_initial.copy_to_host(initial_r);
    state.x_z_initial.copy_to_host(initial_z);
    TENRYU_ASSERT(
        initial_r.size() == static_cast<std::size_t>(n_nodes) &&
            initial_z.size() == static_cast<std::size_t>(n_nodes),
        "band ALE shell requires initial node coordinates");
    runtime.bands.push_back(
        make_shell_band(state, initial_r, initial_z));
  }

  if (include_estimator) {
    std::vector<double> initial_r;
    std::vector<double> initial_z;
    state.x_r_initial.copy_to_host(initial_r);
    state.x_z_initial.copy_to_host(initial_z);
    TENRYU_ASSERT(
        initial_r.size() == static_cast<std::size_t>(n_nodes) &&
            initial_z.size() == static_cast<std::size_t>(n_nodes),
        "band ALE estimator requires initial node coordinates");
    runtime.bands.push_back(
        make_estimator_band(state, cfg, initial_r, initial_z));
  }

  if (include_closure_catchment) {
    runtime.bands.push_back(make_closure_catchment_band(state));
  }

  if (cfg.numerics.ale.band_ale.pole_theta_enabled &&
      state.mesh.topo.multiblock.has_value()) {
    runtime.bands.push_back(make_pole_theta_band(state));
  }

  if (cfg.numerics.ale.band_ale.center_target == "volume_fraction") {
    const bool have_center_chains = std::any_of(
        runtime.bands.begin(),
        runtime.bands.end(),
        [](const BandState& band) {
          return !band.center_nodes.empty();
        });
    if (have_center_chains) {
      std::vector<double> initial_r;
      std::vector<double> initial_z;
      state.x_r_initial.copy_to_host(initial_r);
      state.x_z_initial.copy_to_host(initial_z);
      TENRYU_ASSERT(
          initial_r.size() == static_cast<std::size_t>(n_nodes) &&
              initial_z.size() == static_cast<std::size_t>(n_nodes),
          "band ALE center targets require initial node coordinates");
      const auto incident_cells = build_node_incident_cells(state);
      for (auto& band : runtime.bands) {
        initialize_center_pentagons(
            state, initial_r, initial_z, incident_cells, band);
      }
    }
  }

  for (auto& band : runtime.bands) {
    if (band.kind == BandKind::Shell ||
        band.kind == BandKind::Estimator ||
        band.kind == BandKind::ClosureCatchment ||
        band.kind == BandKind::PoleTheta) {
      continue;
    }
    TENRYU_ASSERT(
        !band.cells.empty(),
        "band ALE band cell list must be non-empty");
    if (band.kind == BandKind::AxisRepair) {
      continue;
    }
    std::vector<double> initial_area;
    initial_area.reserve(band.cells.size());
    for (const int cell : band.cells) {
      TENRYU_ASSERT(
          cell >= 0 && cell < n_cells,
          "band ALE cell id is out of range");
      const double area =
          state.mesh.cell_area[static_cast<std::size_t>(cell)];
      TENRYU_ASSERT(
          std::isfinite(area) && area > 0.0,
          "band ALE initial planar area must be finite and positive");
      initial_area.push_back(area);
    }
    band.d_cells.reset(band.cells.size());
    band.d_cells.copy_from_host(band.cells);
    band.d_initial_area.reset(initial_area.size());
    band.d_initial_area.copy_from_host(initial_area);
  }

  runtime.n_cells = n_cells;
  runtime.n_nodes = n_nodes;
  runtime.d_current_area.reset(static_cast<std::size_t>(n_cells));
  runtime.d_min_ratio.reset(1U);
  if (include_axis) {
    runtime.d_argmin_cell.reset(1U);
  }
  runtime.initialized = true;
}

RuntimeState& require_runtime(core::State& state, const core::Config& cfg) {
  if (g_runtime == nullptr) {
    g_runtime = new RuntimeState;
  }
  if (!g_runtime->initialized) {
    initialize_runtime(state, cfg, *g_runtime);
  }
  TENRYU_ASSERT(
      g_runtime->n_cells == state.mesh.topo.n_cells &&
          g_runtime->n_nodes == state.mesh.topo.n_nodes,
      "band ALE runtime topology changed during a run");
  return *g_runtime;
}

void finalize_axis_repair_initial_area(
    RuntimeState& runtime,
    const parallel::Reduction* reduction) {
  if (runtime.axis_repair_initial_area.empty() ||
      runtime.axis_repair_initial_area_finalized) {
    return;
  }
  if (reduction != nullptr) {
    reduction->allreduce_sum(
        runtime.axis_repair_initial_area.data(), runtime.n_cells);
  }
  for (auto& band : runtime.bands) {
    if (band.kind != BandKind::AxisRepair) {
      continue;
    }
    for (auto& repair : band.axis_repair_cells) {
      const double initial_area =
          runtime.axis_repair_initial_area[
              static_cast<std::size_t>(repair.cell)];
      TENRYU_ASSERT(
          std::isfinite(initial_area) && initial_area > 0.0,
          "band ALE axis repair monitored initial area is unavailable");
      repair.initial_area = initial_area;
      repair.row_initial_area.clear();
      repair.row_initial_area.reserve(repair.row_cells.size());
      for (const int cell : repair.row_cells) {
        const double row_initial_area =
            runtime.axis_repair_initial_area[
                static_cast<std::size_t>(cell)];
        TENRYU_ASSERT(
            std::isfinite(row_initial_area) && row_initial_area > 0.0,
            "band ALE axis repair ring initial area is unavailable");
        repair.row_initial_area.push_back(row_initial_area);
      }
    }
  }
  runtime.axis_repair_initial_area_finalized = true;
}

struct CenterTargetEvaluation {
  double objective = 0.0;
  double gradient_r = 0.0;
  double gradient_z = 0.0;
  double hessian_rr = 0.0;
  double hessian_rz = 0.0;
  double hessian_zz = 0.0;
};

bool optimize_center_target(
    const CenterPentagon& pentagon,
    const std::vector<double>& current_r,
    const std::vector<double>& current_z,
    std::vector<double>& ideal_r,
    std::vector<double>& ideal_z) {
  double pentagon_area = 0.0;
  std::array<double, 5> area_gradient_r = {{0.0, 0.0, 0.0, 0.0, 0.0}};
  std::array<double, 5> area_gradient_z = {{0.0, 0.0, 0.0, 0.0, 0.0}};
  for (std::size_t triangle = 0;
       triangle < pentagon.boundary_nodes.size();
       ++triangle) {
    const auto next =
        (triangle + 1U) % pentagon.boundary_nodes.size();
    const auto node_a =
        static_cast<std::size_t>(pentagon.boundary_nodes[triangle]);
    const auto node_b =
        static_cast<std::size_t>(pentagon.boundary_nodes[next]);
    pentagon_area +=
        ideal_r[node_a] * ideal_z[node_b] -
        ideal_z[node_a] * ideal_r[node_b];
    area_gradient_r[triangle] =
        0.5 * (ideal_z[node_a] - ideal_z[node_b]);
    area_gradient_z[triangle] =
        0.5 * (ideal_r[node_b] - ideal_r[node_a]);
  }
  pentagon_area *= 0.5;
  if (!(std::isfinite(pentagon_area) && pentagon_area > 0.0)) {
    return false;
  }

  const auto center_index =
      static_cast<std::size_t>(pentagon.center_node);
  const double lagrangian_r = current_r[center_index];
  const double lagrangian_z = current_z[center_index];
  const auto evaluate =
      [&](const double center_r,
          const double center_z,
          CenterTargetEvaluation& evaluation) {
        evaluation = CenterTargetEvaluation{};
        for (std::size_t triangle = 0;
             triangle < pentagon.boundary_nodes.size();
             ++triangle) {
          const auto next =
              (triangle + 1U) % pentagon.boundary_nodes.size();
          const auto node_a =
              static_cast<std::size_t>(
                  pentagon.boundary_nodes[triangle]);
          const auto node_b =
              static_cast<std::size_t>(
                  pentagon.boundary_nodes[next]);
          const double area = oriented_triangle_area(
              center_r,
              center_z,
              ideal_r[node_a],
              ideal_z[node_a],
              ideal_r[node_b],
              ideal_z[node_b]);
          const double target_area =
              pentagon.initial_area_fraction[triangle] *
              pentagon_area;
          const double area_floor =
              kCenterTargetAreaFloorFraction * target_area;
          if (!(std::isfinite(area) && area > area_floor)) {
            return false;
          }
          const double log_ratio = std::log(area / target_area);
          const double area_above_floor = area - area_floor;
          evaluation.objective +=
              0.5 * log_ratio * log_ratio -
              kCenterTargetMu *
                  std::log(area_above_floor / target_area);
          const double gradient_coefficient =
              log_ratio / area -
              kCenterTargetMu / area_above_floor;
          const double hessian_coefficient =
              (1.0 - log_ratio) / (area * area) +
              kCenterTargetMu /
                  (area_above_floor * area_above_floor);
          const double area_r = area_gradient_r[triangle];
          const double area_z = area_gradient_z[triangle];
          evaluation.gradient_r +=
              gradient_coefficient * area_r;
          evaluation.gradient_z +=
              gradient_coefficient * area_z;
          evaluation.hessian_rr +=
              hessian_coefficient * area_r * area_r;
          evaluation.hessian_rz +=
              hessian_coefficient * area_r * area_z;
          evaluation.hessian_zz +=
              hessian_coefficient * area_z * area_z;
        }
        const double delta_r = center_r - lagrangian_r;
        const double delta_z = center_z - lagrangian_z;
        const double regularization =
            kCenterTargetLambda / pentagon_area;
        evaluation.objective +=
            0.5 * regularization *
            (delta_r * delta_r + delta_z * delta_z);
        evaluation.gradient_r += regularization * delta_r;
        evaluation.gradient_z += regularization * delta_z;
        evaluation.hessian_rr += regularization;
        evaluation.hessian_zz += regularization;
        return
            std::isfinite(evaluation.objective) &&
            std::isfinite(evaluation.gradient_r) &&
            std::isfinite(evaluation.gradient_z) &&
            std::isfinite(evaluation.hessian_rr) &&
            std::isfinite(evaluation.hessian_rz) &&
            std::isfinite(evaluation.hessian_zz);
      };

  double center_r = ideal_r[center_index];
  double center_z = ideal_z[center_index];
  for (int iteration = 0;
       iteration < kCenterTargetNewtonIterations;
       ++iteration) {
    CenterTargetEvaluation evaluation;
    if (!evaluate(center_r, center_z, evaluation)) {
      continue;
    }
    const double determinant =
        evaluation.hessian_rr * evaluation.hessian_zz -
        evaluation.hessian_rz * evaluation.hessian_rz;
    if (!std::isfinite(determinant) || determinant == 0.0) {
      continue;
    }
    const double step_r =
        (-evaluation.hessian_zz * evaluation.gradient_r +
         evaluation.hessian_rz * evaluation.gradient_z) /
        determinant;
    const double step_z =
        (evaluation.hessian_rz * evaluation.gradient_r -
         evaluation.hessian_rr * evaluation.gradient_z) /
        determinant;
    if (!std::isfinite(step_r) || !std::isfinite(step_z)) {
      continue;
    }

    double step_scale = 1.0;
    for (int halving = 0;
         halving <= kCenterTargetBacktrackingHalvings;
         ++halving) {
      const double candidate_r = center_r + step_scale * step_r;
      const double candidate_z = center_z + step_scale * step_z;
      CenterTargetEvaluation candidate;
      if (evaluate(candidate_r, candidate_z, candidate) &&
          candidate.objective < evaluation.objective) {
        center_r = candidate_r;
        center_z = candidate_z;
        break;
      }
      step_scale *= 0.5;
    }
  }
  ideal_r[center_index] = center_r;
  ideal_z[center_index] = center_z;
  return true;
}

void apply_radial_respace_target(
    const core::State& state,
    const BandState& band,
    const std::vector<double>& current_r,
    const std::vector<double>& current_z,
    const double chi,
    const double move_cap_frac,
    const bool uniform_current_fractions,
    std::vector<double>& target_r,
    std::vector<double>& target_z) {
  std::vector<double> initial_r;
  std::vector<double> initial_z;
  state.x_r_initial.copy_to_host(initial_r);
  state.x_z_initial.copy_to_host(initial_z);
  const std::vector<double> mean_radius =
      ring_mean_radii(band.lines, current_r, current_z);
  const std::vector<double> initial_mean_radius =
      ring_mean_radii(band.lines, initial_r, initial_z);
  const double initial_span =
      initial_mean_radius.front() - initial_mean_radius.back();
  const double current_inner_radius = mean_radius.back();
  const double current_span =
      mean_radius.front() - current_inner_radius;
  const std::size_t nodes_per_line = band.lines.front().size();
  const bool uniform_line_lengths = std::all_of(
      band.lines.begin(), band.lines.end(),
      [nodes_per_line](const std::vector<int>& nodes) {
        return nodes.size() == nodes_per_line;
      });
  if (!uniform_line_lengths) {
    // Per-line homothety scaling (the pre-cap algorithm): dendrite and
    // other unequal-length line sets have no cross-line node pairing, so
    // the gap-capped path below cannot apply; move_cap_frac is not used
    // on this branch.
    std::vector<double> initial_r;
    std::vector<double> initial_z;
    state.x_r_initial.copy_to_host(initial_r);
    state.x_z_initial.copy_to_host(initial_z);
    const std::vector<double> mean_radius =
        ring_mean_radii(band.lines, current_r, current_z);
    const std::vector<double> initial_mean_radius =
        ring_mean_radii(band.lines, initial_r, initial_z);
    const double initial_span =
        initial_mean_radius.front() - initial_mean_radius.back();
    const double current_inner_radius = mean_radius.back();
    const double current_span =
        mean_radius.front() - current_inner_radius;
    for (std::size_t line = 0; line < band.lines.size(); ++line) {
      double target_radius;
      if (uniform_current_fractions) {
        const double fraction =
            static_cast<double>(line) /
            static_cast<double>(band.lines.size() - 1U);
        target_radius =
            mean_radius.front() +
            fraction * (mean_radius.back() - mean_radius.front());
      } else {
        const double fraction =
            (initial_mean_radius[line] - initial_mean_radius.back()) /
            initial_span;
        target_radius =
            current_inner_radius + fraction * current_span;
      }
      for (const int node : band.lines[line]) {
        const auto node_index = static_cast<std::size_t>(node);
        if (band.pinned_nodes[node_index] != 0U) {
          continue;
        }
        const double radius =
            std::hypot(current_r[node_index], current_z[node_index]);
        if (!(radius > 0.0)) {
          continue;
        }
        const double scale = target_radius / radius;
        const double ideal_r = current_r[node_index] * scale;
        const double ideal_z = current_z[node_index] * scale;
        if ((state.mesh.topo.node_flags[node_index] &
             mesh::NODE_POLE_AXIS) == 0U) {
          target_r[node_index] =
              current_r[node_index] +
              chi * (ideal_r - current_r[node_index]);
        }
        target_z[node_index] =
            current_z[node_index] +
            chi * (ideal_z - current_z[node_index]);
      }
    }
    return;
  }
  for (std::size_t line = 0; line < band.lines.size(); ++line) {
    double target_radius;
    if (uniform_current_fractions) {
      const double fraction =
          static_cast<double>(line) /
          static_cast<double>(band.lines.size() - 1U);
      target_radius =
          mean_radius.front() +
          fraction * (mean_radius.back() - mean_radius.front());
    } else {
      const double fraction =
          (initial_mean_radius[line] - initial_mean_radius.back()) /
          initial_span;
      target_radius =
          current_inner_radius + fraction * current_span;
    }
    for (std::size_t k = 0; k < band.lines[line].size(); ++k) {
      const int node = band.lines[line][k];
      const auto node_index = static_cast<std::size_t>(node);
      if (band.pinned_nodes[node_index] != 0U) {
        continue;
      }
      const double radius =
          std::hypot(current_r[node_index], current_z[node_index]);
      if (!(radius > 0.0)) {
        continue;
      }
      const auto prev = line > 0U
          ? static_cast<std::size_t>(band.lines[line - 1U][k])
          : node_index;
      const auto next = line + 1U < band.lines.size()
          ? static_cast<std::size_t>(band.lines[line + 1U][k])
          : node_index;
      const double gap_out = line > 0U
          ? std::abs(std::hypot(current_r[prev], current_z[prev]) - radius)
          : std::numeric_limits<double>::infinity();
      const double gap_in = line + 1U < band.lines.size()
          ? std::abs(radius -
                     std::hypot(current_r[next], current_z[next]))
          : std::numeric_limits<double>::infinity();
      const double local_gap = std::min(gap_out, gap_in);
      const double scale = target_radius / radius;
      const double ideal_r = current_r[node_index] * scale;
      const double ideal_z = current_z[node_index] * scale;
      const double full_move = chi * (target_radius - radius);
      const double move = band_respace_capped_move(
          radius, target_radius, chi, local_gap, move_cap_frac);
      if (std::memcmp(&move, &full_move, sizeof(double)) == 0) {
        if ((state.mesh.topo.node_flags[node_index] &
             mesh::NODE_POLE_AXIS) == 0U) {
          target_r[node_index] =
              current_r[node_index] +
              chi * (ideal_r - current_r[node_index]);
        }
        target_z[node_index] =
            current_z[node_index] +
            chi * (ideal_z - current_z[node_index]);
      } else {
        const double capped_scale = (radius + move) / radius;
        if ((state.mesh.topo.node_flags[node_index] &
             mesh::NODE_POLE_AXIS) == 0U) {
          target_r[node_index] = current_r[node_index] * capped_scale;
        }
        target_z[node_index] = current_z[node_index] * capped_scale;
      }
    }
  }
}

void apply_layer_respace_target(
    const core::State& state,
    const BandState& band,
    const std::vector<double>& current_r,
    const std::vector<double>& current_z,
    const double chi,
    std::vector<double>& target_r,
    std::vector<double>& target_z) {
  TENRYU_ASSERT(
      band.lines.size() >= 2U,
      "band ALE layer respace requires at least two node lines");
  std::vector<double> mean_coordinate;
  mean_coordinate.reserve(band.lines.size());
  for (const auto& line : band.lines) {
    TENRYU_ASSERT(
        !line.empty(),
        "band ALE layer respace node line must be non-empty");
    double coordinate_sum = 0.0;
    for (const int node : line) {
      const auto node_index = static_cast<std::size_t>(node);
      TENRYU_ASSERT(
          node >= 0 && node_index < current_r.size() &&
              node_index < current_z.size(),
          "band ALE layer respace node id is out of range");
      coordinate_sum += band.estimator_layer_axis_is_j
                            ? current_z[node_index]
                            : current_r[node_index];
    }
    mean_coordinate.push_back(
        coordinate_sum / static_cast<double>(line.size()));
  }

  for (std::size_t line = 0; line < band.lines.size(); ++line) {
    const double fraction =
        static_cast<double>(line) /
        static_cast<double>(band.lines.size() - 1U);
    const double target_coordinate =
        mean_coordinate.front() +
        fraction * (mean_coordinate.back() - mean_coordinate.front());
    for (const int node : band.lines[line]) {
      const auto node_index = static_cast<std::size_t>(node);
      if (band.pinned_nodes[node_index] != 0U) {
        continue;
      }
      if (band.estimator_layer_axis_is_j) {
        target_z[node_index] =
            current_z[node_index] +
            chi * (target_coordinate - current_z[node_index]);
      } else if ((state.mesh.topo.node_flags[node_index] &
                  mesh::NODE_POLE_AXIS) == 0U) {
        target_r[node_index] =
            current_r[node_index] +
            chi * (target_coordinate - current_r[node_index]);
      }
    }
  }
}

double spoke_ring_radius(const BandState& band,
                         const int ring,
                         const int spoke,
                         const std::vector<double>& node_r,
                         const std::vector<double>& node_z) {
  const int node = band.shell_rings[static_cast<std::size_t>(ring)]
                                  [static_cast<std::size_t>(spoke)];
  return std::hypot(node_r[static_cast<std::size_t>(node)],
                    node_z[static_cast<std::size_t>(node)]);
}

double spoke_node_radial_gap(const BandState& band,
                             const int ring,
                             const int spoke,
                             const std::vector<double>& node_r,
                             const std::vector<double>& node_z) {
  const int last_ring = static_cast<int>(band.shell_rings.size()) - 1;
  double gap = std::numeric_limits<double>::infinity();
  const double radius =
      spoke_ring_radius(band, ring, spoke, node_r, node_z);
  if (ring > 0) {
    gap = std::min(
        gap,
        std::abs(radius - spoke_ring_radius(
                              band, ring - 1, spoke, node_r, node_z)));
  }
  if (ring < last_ring) {
    gap = std::min(
        gap,
        std::abs(spoke_ring_radius(
                     band, ring + 1, spoke, node_r, node_z) -
                 radius));
  }
  return gap;
}

bool prepare_per_column_target(const core::State& state,
                               const core::Config& cfg,
                               BandState& band,
                               const bool first_engagement,
                               std::string* reject_reason) {
  std::vector<double> current_r;
  std::vector<double> current_z;
  state.x_r.copy_to_host(current_r);
  state.x_z.copy_to_host(current_z);
  const int n_spokes = static_cast<int>(band.pc_spoke_active.size());
  const int n_rings = static_cast<int>(band.shell_rings.size());
  TENRYU_ASSERT(
      n_spokes >= 2 &&
          band.pc_spoke_i0.size() == static_cast<std::size_t>(n_spokes) &&
          band.pc_spoke_i1.size() == static_cast<std::size_t>(n_spokes),
      "band ALE per-column target metadata is incomplete");
  std::vector<double> taper(static_cast<std::size_t>(n_spokes), 1.0);
  const int taper_width =
      cfg.numerics.ale.band_ale.estimator_band_pc_filter_halfwidth + 1;
  int start = 0;
  while (start < n_spokes) {
    if (band.pc_spoke_active[static_cast<std::size_t>(start)] == 0U) {
      ++start;
      continue;
    }
    int end = start;
    while (end + 1 < n_spokes &&
           band.pc_spoke_active[static_cast<std::size_t>(end + 1)] != 0U) {
      ++end;
    }
    if (start > 0) {
      for (int k = 1; k <= taper_width && start + k - 1 <= end; ++k) {
        const double tau = 0.5 *
            (1.0 - std::cos(CUDART_PI * static_cast<double>(k) /
                            static_cast<double>(taper_width + 1)));
        taper[static_cast<std::size_t>(start + k - 1)] = std::min(
            taper[static_cast<std::size_t>(start + k - 1)], tau);
      }
    }
    if (end + 1 < n_spokes) {
      for (int k = 1; k <= taper_width && end - k + 1 >= start; ++k) {
        const double tau = 0.5 *
            (1.0 - std::cos(CUDART_PI * static_cast<double>(k) /
                            static_cast<double>(taper_width + 1)));
        taper[static_cast<std::size_t>(end - k + 1)] = std::min(
            taper[static_cast<std::size_t>(end - k + 1)], tau);
      }
    }
    start = end + 1;
  }

  std::vector<double> displacement(
      static_cast<std::size_t>(n_rings) *
          static_cast<std::size_t>(n_spokes),
      0.0);
  const auto displacement_at = [&](const int ring, const int spoke)
      -> double& {
    return displacement[
        static_cast<std::size_t>(ring) *
            static_cast<std::size_t>(n_spokes) +
        static_cast<std::size_t>(spoke)];
  };
  for (int spoke = 0; spoke < n_spokes; ++spoke) {
    if (band.pc_spoke_active[static_cast<std::size_t>(spoke)] == 0U) {
      continue;
    }
    const int i0 = band.pc_spoke_i0[static_cast<std::size_t>(spoke)];
    const int i1 = band.pc_spoke_i1[static_cast<std::size_t>(spoke)];
    const double inner_radius =
        spoke_ring_radius(band, i0, spoke, current_r, current_z);
    const double outer_radius =
        spoke_ring_radius(band, i1, spoke, current_r, current_z);
    for (int ring = i0 + 1; ring < i1; ++ring) {
      if (band.pc_phase_b_active &&
          !node_incident_cells_disjoint_from_tube(
              band.pc_tube_mask,
              n_rings - 1,
              n_spokes - 1,
              ring,
              spoke)) {
        continue;
      }
      const double fraction =
          static_cast<double>(ring - i0) /
          static_cast<double>(i1 - i0);
      const double target_radius =
          inner_radius + fraction * (outer_radius - inner_radius);
      displacement_at(ring, spoke) =
          taper[static_cast<std::size_t>(spoke)] *
          (target_radius -
           spoke_ring_radius(band, ring, spoke, current_r, current_z));
    }
  }

  const double dtheta =
      CUDART_PI / static_cast<double>(n_spokes - 1);
  for (int spoke = 0; spoke + 1 < n_spokes; ++spoke) {
    for (int ring = 1; ring + 1 < n_rings; ++ring) {
      const double d0 = displacement_at(ring, spoke);
      const double d1 = displacement_at(ring, spoke + 1);
      if (d0 == 0.0 && d1 == 0.0) {
        continue;
      }
      const double h = std::min(
          spoke_node_radial_gap(
              band, ring, spoke, current_r, current_z),
          spoke_node_radial_gap(
              band, ring, spoke + 1, current_r, current_z));
      const double radius = std::min(
          spoke_ring_radius(band, ring, spoke, current_r, current_z),
          spoke_ring_radius(band, ring, spoke + 1, current_r, current_z));
      if (std::abs(d1 - d0) >
          0.25 * std::min(h, radius * dtheta)) {
        *reject_reason = "smoothness-1";
        return false;
      }
    }
  }
  for (int spoke = 1; spoke + 1 < n_spokes; ++spoke) {
    for (int ring = 1; ring + 1 < n_rings; ++ring) {
      const double dm = displacement_at(ring, spoke - 1);
      const double d0 = displacement_at(ring, spoke);
      const double dp = displacement_at(ring, spoke + 1);
      if (dm == 0.0 && d0 == 0.0 && dp == 0.0) {
        continue;
      }
      const double h =
          spoke_node_radial_gap(
              band, ring, spoke, current_r, current_z);
      if (std::abs(dp - 2.0 * d0 + dm) > 0.20 * h) {
        *reject_reason = "smoothness-2";
        return false;
      }
    }
  }

  for (int spoke = 0; spoke < n_spokes; ++spoke) {
    if (band.pc_spoke_active[static_cast<std::size_t>(spoke)] == 0U) {
      continue;
    }
    const int i0 = band.pc_spoke_i0[static_cast<std::size_t>(spoke)];
    const int i1 = band.pc_spoke_i1[static_cast<std::size_t>(spoke)];
    std::vector<double> current_spacing;
    std::vector<double> target_spacing;
    current_spacing.reserve(static_cast<std::size_t>(i1 - i0));
    target_spacing.reserve(static_cast<std::size_t>(i1 - i0));
    for (int ring = i0; ring < i1; ++ring) {
      const double current0 =
          spoke_ring_radius(band, ring, spoke, current_r, current_z);
      const double current1 =
          spoke_ring_radius(band, ring + 1, spoke, current_r, current_z);
      const double current_h = current1 - current0;
      if (!(std::isfinite(current_h) && current_h > 0.0)) {
        *reject_reason = "spacing-positive";
        return false;
      }
      current_spacing.push_back(current_h);
      target_spacing.push_back(
          current1 + displacement_at(ring + 1, spoke) -
          current0 - displacement_at(ring, spoke));
    }
    const double spacing_floor = 0.35 * median_value(current_spacing);
    for (std::size_t k = 0; k < target_spacing.size(); ++k) {
      if (!(target_spacing[k] > 0.0)) {
        *reject_reason = "spacing-positive";
        return false;
      }
      if (target_spacing[k] < spacing_floor) {
        *reject_reason = "spacing-floor";
        return false;
      }
      if (k > 0U) {
        const double ratio = target_spacing[k] / target_spacing[k - 1U];
        if (ratio < 2.0 / 3.0 || ratio > 3.0 / 2.0) {
          *reject_reason = "spacing-ratio";
          return false;
        }
      }
    }
  }

  double displacement_bound = std::numeric_limits<double>::infinity();
  for (int spoke = 0; spoke < n_spokes; ++spoke) {
    if (band.pc_spoke_active[static_cast<std::size_t>(spoke)] == 0U) {
      continue;
    }
    const int i0 = band.pc_spoke_i0[static_cast<std::size_t>(spoke)];
    const int i1 = band.pc_spoke_i1[static_cast<std::size_t>(spoke)];
    for (int ring = i0 + 1; ring < i1; ++ring) {
      const double d = std::abs(displacement_at(ring, spoke));
      const double h =
          spoke_node_radial_gap(
              band, ring, spoke, current_r, current_z);
      displacement_bound = std::min(
          displacement_bound, 0.20 * h / (d + 1.0e-300));
    }
  }
  // Phase A recorded divergence from consult Eq. (36): the swept-volume
  // limiter term is deferred to Phase B.
  const double rate_bound =
      first_engagement
          ? cfg.numerics.ale.band_ale.estimator_band_pc_chi_step
          : band.pc_previous_chi +
                cfg.numerics.ale.band_ale.estimator_band_pc_chi_step;
  band.pc_chi = std::min(
      {cfg.numerics.ale.band_ale.estimator_band_pc_chi_max,
       rate_bound,
       displacement_bound});
  TENRYU_ASSERT(
      std::isfinite(band.pc_chi) && band.pc_chi > 0.0,
      "band ALE per-column chi bound must be finite and positive");

  band.pc_cached_target_r = current_r;
  band.pc_cached_target_z = current_z;
  for (int spoke = 0; spoke < n_spokes; ++spoke) {
    if (band.pc_spoke_active[static_cast<std::size_t>(spoke)] == 0U) {
      continue;
    }
    const int i0 = band.pc_spoke_i0[static_cast<std::size_t>(spoke)];
    const int i1 = band.pc_spoke_i1[static_cast<std::size_t>(spoke)];
    for (int ring = i0 + 1; ring < i1; ++ring) {
      const int node =
          band.shell_rings[static_cast<std::size_t>(ring)]
                          [static_cast<std::size_t>(spoke)];
      const auto node_index = static_cast<std::size_t>(node);
      const double radius = std::hypot(
          current_r[node_index], current_z[node_index]);
      if (!(radius > 0.0)) {
        continue;
      }
      const double target_radius =
          radius + band.pc_chi * displacement_at(ring, spoke);
      const double scale = target_radius / radius;
      if ((state.mesh.topo.node_flags[node_index] &
           mesh::NODE_POLE_AXIS) == 0U) {
        band.pc_cached_target_r[node_index] =
            current_r[node_index] * scale;
      } else {
        band.pc_cached_target_r[node_index] = 0.0;
      }
      band.pc_cached_target_z[node_index] =
          current_z[node_index] * scale;
    }
  }
  return true;
}

void fnv1a_byte(std::uint64_t* hash, const std::uint8_t byte) {
  *hash ^= static_cast<std::uint64_t>(byte);
  *hash *= 1099511628211ULL;
}

template <typename UInt>
void fnv1a_little_endian(std::uint64_t* hash, const UInt value) {
  for (std::size_t byte = 0; byte < sizeof(UInt); ++byte) {
    fnv1a_byte(
        hash,
        static_cast<std::uint8_t>(value >> (8U * byte)));
  }
}

std::uint64_t per_column_decision_hash(const BandState& band,
                                       const PcDecision decision) {
  // Deterministic FNV-1a byte layout: for each column, little-endian int32
  // i0, int32 i1, uint8 well_formed; then little-endian int64
  // llround(eta_P*1e6), int64 llround(chi*1e6), and uint8 decision code.
  std::uint64_t hash = 14695981039346656037ULL;
  for (std::size_t j = 0; j < band.pc_well_formed.size(); ++j) {
    const std::int32_t i0 = band.pc_well_formed[j] != 0U
                                ? static_cast<std::int32_t>(band.pc_i0[j])
                                : -1;
    const std::int32_t i1 = band.pc_well_formed[j] != 0U
                                ? static_cast<std::int32_t>(band.pc_i1[j])
                                : -1;
    fnv1a_little_endian(
        &hash, static_cast<std::uint32_t>(i0));
    fnv1a_little_endian(
        &hash, static_cast<std::uint32_t>(i1));
    fnv1a_byte(&hash, band.pc_well_formed[j]);
  }
  const std::int64_t eta_scaled =
      std::llround(band.pc_eta_p * 1.0e6);
  const std::int64_t chi_scaled =
      std::llround(band.pc_chi * 1.0e6);
  fnv1a_little_endian(
      &hash, static_cast<std::uint64_t>(eta_scaled));
  fnv1a_little_endian(
      &hash, static_cast<std::uint64_t>(chi_scaled));
  fnv1a_byte(&hash, static_cast<std::uint8_t>(decision));
  return hash;
}

void log_per_column_decision(const std::int64_t step,
                             const BandState& band,
                             const PcDecision decision,
                             const int rank) {
  if (rank != 0) {
    return;
  }
  std::ostringstream log;
  log << "[band-ale-pc] step=" << step << " decision=";
  if (decision == PcDecision::Hold) {
    log << "hold";
  } else if (decision == PcDecision::Noop) {
    log << "noop";
  } else if (decision == PcDecision::Reject) {
    log << "reject:" << band.pc_reject_reason;
  } else {
    log << "commit";
  }
  log << " hash=" << std::hex << std::setw(16) << std::setfill('0')
      << per_column_decision_hash(band, decision)
      << std::dec << " held_cols=" << band.pc_held_columns
      << " patches=" << band.pc_patch_count
      << " tube_cols=" << band.pc_tube_columns
      << " ambiguous=" << band.pc_ambiguous_columns;
  core::log_info(log.str());
}

void log_per_column_shadow(const std::int64_t step,
                           const BandState& band,
                           const int rank) {
  if (rank != 0) {
    return;
  }
  const bool legacy_commit = band.engaged && !band.pc_legacy_hold;
  const bool pc_commit =
      band.pc_shadow_engage_decision && !band.pc_shadow_hold;
  // Shadow agreement is decision equivalence; hold races alone do not matter.
  const bool agree =
      legacy_commit == pc_commit &&
      band.pc_shadow_engage_decision == band.engaged &&
      (!(legacy_commit && pc_commit) ||
       band.pc_shadow_first_mismatch < 0);
  std::ostringstream log;
  log << "[band-ale-pc-shadow] step=" << step
      << " agree=" << (agree ? 1 : 0)
      << " hold_legacy=" << (band.pc_legacy_hold ? 1 : 0)
      << " hold_pc=" << (band.pc_shadow_hold ? 1 : 0)
      << " engage_legacy=" << (band.engaged ? 1 : 0)
      << " engage_pc=" << (band.pc_shadow_engage_decision ? 1 : 0)
      << " mismatch_j=" << band.pc_shadow_first_mismatch
      << " hold_xor_legacy_only="
      << band.pc_shadow_hold_xor_legacy_only
      << " hold_xor_pc_only=" << band.pc_shadow_hold_xor_pc_only;
  core::log_info(log.str());
}

void log_engagement_change(const int step,
                           const BandState& band,
                           const double min_ratio,
                           const int rank) {
  if (rank != 0) {
    return;
  }
  std::ostringstream log;
  log << "[band-ale] step=" << step
      << " band=" << band.name
      << " engaged=" << (band.engaged ? 1 : 0)
      << " min_ratio=" << std::scientific << std::setprecision(3)
      << min_ratio;
  if (band.kind == BandKind::Estimator) {
    log << " eta_h=" << band.eta_h;
    if (!band.pc_i0.empty()) {
      log << " C=" << band.pc_coverage
          << " held_cols=" << band.pc_held_columns
          << " active_spokes=" << band.pc_active_spokes
          << " eta_P=" << band.pc_eta_p
          << " chi=" << band.pc_chi;
    }
  }
  core::log_info(log.str());
}

}  // namespace

void run_start(core::State& state, const core::Config& cfg) {
  delete g_runtime;
  g_runtime = nullptr;
  if (cfg.numerics.ale.band_ale.enabled) {
    g_runtime = new RuntimeState;
    initialize_runtime(state, cfg, *g_runtime);
  }
}

std::vector<ActiveBand> begin_step(
    core::State& state,
    const core::Config& cfg,
    const parallel::Reduction* reduction,
    const int rank) {
  static const int diag_every = [] {
    const char* s = std::getenv("TENRYU_BANDALE_DIAG");
    return s ? std::atoi(s) : 0;
  }();
  RuntimeState& runtime = require_runtime(state, cfg);
  runtime.rank = rank;
  const bool have_pole_theta = std::any_of(
      runtime.bands.begin(),
      runtime.bands.end(),
      [](const BandState& band) {
        return band.kind == BandKind::PoleTheta;
      });
  if (have_pole_theta) {
    runtime.pole_theta_dtheta.assign(
        static_cast<std::size_t>(runtime.n_nodes), 0.0);
    runtime.pole_theta_target_s.assign(
        static_cast<std::size_t>(runtime.n_nodes), 0.0);
    runtime.pole_theta_consumed = false;
    runtime.pole_theta_composed_last_build = false;
  }
  for (auto& band : runtime.bands) {
    if (band.kind == BandKind::Shell) {
      update_shell_band(state, cfg, band);
    } else if (band.kind == BandKind::Estimator) {
      update_estimator_band(state, cfg, band);
    }
  }
  runtime.d_current_area.copy_from_host(state.mesh.cell_area);
  const int owned_begin =
      state.owned_cells_begin_or(0);
  const int owned_end =
      state.owned_cells_end_or(runtime.n_cells);
  if (!runtime.axis_repair_initial_area.empty()) {
    finalize_axis_repair_initial_area(runtime, reduction);
    runtime.axis_repair_current_area.assign(
        static_cast<std::size_t>(runtime.n_cells), 0.0);
    for (int cell = owned_begin; cell < owned_end; ++cell) {
      runtime.axis_repair_current_area[static_cast<std::size_t>(cell)] =
          state.mesh.cell_area[static_cast<std::size_t>(cell)];
    }
    if (reduction != nullptr) {
      reduction->allreduce_sum(
          runtime.axis_repair_current_area.data(), runtime.n_cells);
    }
  }
  std::vector<ActiveBand> active;
  active.reserve(runtime.bands.size());
  for (std::size_t index = 0; index < runtime.bands.size(); ++index) {
    auto& band = runtime.bands[index];
    const std::int64_t this_step = state.step + 1;
    if (band.kind == BandKind::PoleTheta) {
      evaluate_pole_theta(state, cfg, band, rank);
      band.engaged = band.pole_north.fired_this_step ||
                     band.pole_south.fired_this_step ||
                     band.pole_routine_acted;
      band.argmin_cell = -1;
      if (band.engaged) {
        active.push_back(
            ActiveBand{index, band.name, band.min_ratio, -1});
      }
      continue;
    }
    if (band.kind == BandKind::ClosureCatchment) {
      const bool was_engaged = band.engaged;
      band.engaged = prepare_closure_catchment_target(
          state, cfg, band, rank);
      if (band.closure_pending_absorb_rows > 0) {
        state.central_pseudo_core.min_member_ring_count =
            state.central_pseudo_core.member_ring_count +
            band.closure_pending_absorb_rows;
        band.closure_pending_absorb_rows = 0;
      }
      band.min_ratio = band.closure_min_eta;
      band.argmin_cell = -1;
      if (band.engaged) {
        if (!was_engaged) {
          ++band.engagements;
        }
        if (band.last_counted_step != this_step) {
          ++band.engaged_steps;
        }
        active.push_back(
            ActiveBand{index, band.name, band.min_ratio, -1});
      }
      band.last_counted_step = this_step;
      continue;
    }
    const bool per_column_production =
        band.kind == BandKind::Estimator &&
        cfg.numerics.ale.band_ale.estimator_band_per_column &&
        cfg.numerics.hydro.pressure_drive_perturbation.enabled;
    const bool per_column_shadow =
        band.kind == BandKind::Estimator &&
        cfg.numerics.ale.band_ale.estimator_band_per_column &&
        !cfg.numerics.hydro.pressure_drive_perturbation.enabled;
    if (per_column_production &&
        band.pc_predecision != PcDecision::Pending) {
      const bool was_engaged = band.engaged;
      band.engaged = false;
      band.pc_release_evaluations = 0;
      band.min_ratio = std::numeric_limits<double>::infinity();
      band.eta_h = band.pc_eta_p;
      band.argmin_cell = -1;
      band.pc_chi = 0.0;
      PcDecision decision = band.pc_predecision;
      if (band.pc_sigma_starvation_events_remaining > 0) {
        decision = PcDecision::Hold;
        --band.pc_sigma_starvation_events_remaining;
      }
      if (band.pc_cooldown_events_remaining > 0) {
        --band.pc_cooldown_events_remaining;
      }
      if (band.last_counted_step != this_step && was_engaged) {
        log_engagement_change(
            this_step, band, band.min_ratio, rank);
      }
      log_per_column_decision(this_step, band, decision, rank);
      update_per_column_state_for_test(band);
      band.last_counted_step = this_step;
      continue;
    }
    if ((band.kind == BandKind::Shell ||
         band.kind == BandKind::Estimator) &&
        !per_column_production &&
        band.lines.size() < 3U) {
      TENRYU_ASSERT(
          band.cells.empty(),
          "band ALE inactive dynamic window must not contain cells");
      const bool was_engaged = band.engaged;
      band.engaged = false;
      band.min_ratio = std::numeric_limits<double>::infinity();
      band.eta_h = std::numeric_limits<double>::quiet_NaN();
      band.argmin_cell = -1;
      if (band.last_counted_step != this_step && was_engaged) {
        log_engagement_change(
            this_step, band, band.min_ratio, rank);
      }
      if (per_column_shadow) {
        log_per_column_shadow(this_step, band, rank);
      }
      band.last_counted_step = this_step;
      continue;
    }
    if (band.kind == BandKind::AxisRepair) {
      const bool new_step =
          band.last_counted_step != this_step;
      const int previous_engaged_count =
          band.axis_repair_engaged_count;
      int engaged_count = 0;
      double min_eta = std::numeric_limits<double>::infinity();
      for (auto& repair : band.axis_repair_cells) {
        const double current_area =
            runtime.axis_repair_current_area[
                static_cast<std::size_t>(repair.cell)];
        TENRYU_ASSERT(
            std::isfinite(current_area) && current_area > 0.0,
            "band ALE axis repair monitored area must be finite and positive");
        repair.current_area = current_area;
        repair.scale = std::sqrt(current_area / repair.initial_area);

        std::vector<double> ring_scale;
        ring_scale.reserve(repair.row_cells.size());
        for (std::size_t q = 0; q < repair.row_cells.size(); ++q) {
          const double row_area =
              runtime.axis_repair_current_area[
                  static_cast<std::size_t>(repair.row_cells[q])];
          TENRYU_ASSERT(
              std::isfinite(row_area) && row_area > 0.0,
              "band ALE axis repair ring area must be finite and positive");
          ring_scale.push_back(
              std::sqrt(row_area / repair.row_initial_area[q]));
        }
        std::sort(ring_scale.begin(), ring_scale.end());
        TENRYU_ASSERT(
            !ring_scale.empty(),
            "band ALE axis repair ring scale list must be non-empty");
        repair.ring_scale =
            ring_scale[ring_scale.size() / 2U];
        const double eta = repair.scale / repair.ring_scale;
        TENRYU_ASSERT(
            std::isfinite(eta) && eta > 0.0,
            "band ALE axis repair eta must be finite and positive");
        min_eta = std::min(min_eta, eta);

        if (!repair.engaged &&
            eta < cfg.numerics.ale.band_ale.axis_repair_eta_on) {
          repair.engaged = true;
          ++band.engagements;
        } else if (
            repair.engaged &&
            eta > cfg.numerics.ale.band_ale.axis_repair_eta_off) {
          repair.engaged = false;
        }
        if (repair.engaged) {
          ++engaged_count;
        }
      }
      band.axis_repair_engaged_count = engaged_count;
      band.engaged = engaged_count > 0;
      band.min_ratio = min_eta;
      const auto& block =
          state.mesh.topo.multiblock->blocks[
              static_cast<std::size_t>(band.block_id)];
      band.axis_repair_patch_cells =
          build_axis_repair_patch_cells(band, block);
      const int cooldown_before =
          band.axis_repair_cooldown_steps_remaining;
      const bool on_cooldown =
          cooldown_before > 0 &&
          this_step > band.axis_repair_cooldown_armed_step;
      if (new_step && on_cooldown) {
        --band.axis_repair_cooldown_steps_remaining;
      }
      if (new_step &&
          (engaged_count != previous_engaged_count ||
           (on_cooldown &&
            cooldown_before == kAxisRepairCooldownSteps)) &&
          rank == 0) {
        core::log_info(
            "[axis-repair] step=" + std::to_string(this_step) +
            " block=" + std::to_string(band.block_id) +
            " engaged=" + std::to_string(engaged_count) +
            " sweeps=" + std::to_string(kAxisRepairSweeps) +
            " patch_cells=" +
            std::to_string(band.axis_repair_patch_cells.size()) +
            (on_cooldown ? " cooldown" : ""));
      }
      if (band.engaged && !on_cooldown) {
        if (new_step) {
          ++band.engaged_steps;
          ++runtime.axis_repair_total_transactions;
        }
        active.push_back(
            ActiveBand{index, band.name, min_eta, -1});
      }
      band.last_counted_step = this_step;
      continue;
    }
    double min_ratio = std::numeric_limits<double>::infinity();
    int argmin_cell = -1;
    if (band.kind == BandKind::Axis) {
      band_min_ratio_argmin_kernel<<<1, 256>>>(
          band.d_cells.data(),
          band.d_initial_area.data(),
          runtime.d_current_area.data(),
          static_cast<int>(band.cells.size()),
          owned_begin,
          owned_end,
          runtime.d_min_ratio.data(),
          runtime.d_argmin_cell.data());
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(
          &min_ratio,
          runtime.d_min_ratio.data(),
          sizeof(double),
          cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(
          &argmin_cell,
          runtime.d_argmin_cell.data(),
          sizeof(int),
          cudaMemcpyDeviceToHost));
      if (reduction != nullptr) {
        const double local_min_ratio = min_ratio;
        min_ratio = reduction->allreduce_min(local_min_ratio);
        const bool owns_global_min =
            local_min_ratio == min_ratio &&
            argmin_cell != std::numeric_limits<int>::max();
        const std::uint64_t local_argmin =
            owns_global_min
                ? static_cast<std::uint64_t>(argmin_cell)
                : std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t global_argmin =
            reduction->allreduce_min(local_argmin);
        TENRYU_ASSERT(
            global_argmin <=
                static_cast<std::uint64_t>(
                    std::numeric_limits<int>::max()),
            "band ALE axis argmin cell was not owned by any rank");
        argmin_cell = static_cast<int>(global_argmin);
      }
    } else {
      band_min_ratio_kernel<<<1, 256>>>(
          band.d_cells.data(),
          band.d_initial_area.data(),
          runtime.d_current_area.data(),
          static_cast<int>(band.cells.size()),
          owned_begin,
          owned_end,
          runtime.d_min_ratio.data());
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(
          &min_ratio,
          runtime.d_min_ratio.data(),
          sizeof(double),
          cudaMemcpyDeviceToHost));
      if (reduction != nullptr) {
        min_ratio = reduction->allreduce_min(min_ratio);
      }
    }
    band.min_ratio = min_ratio;
    band.argmin_cell = argmin_cell;

    if (band.kind == BandKind::Estimator && !per_column_production) {
      std::vector<double> current_r;
      std::vector<double> current_z;
      state.x_r.copy_to_host(current_r);
      state.x_z.copy_to_host(current_z);
      if (band.estimator_polar_geometry) {
        band.eta_h = ring_spacing_nonuniformity(
            band.lines, current_r, current_z);
      } else {
        band.eta_h = layer_spacing_nonuniformity(
            band.lines,
            current_r,
            current_z,
            band.estimator_layer_axis_is_j);
      }
      if (rank == 0) {
        std::ostringstream eta_log;
        eta_log << "[band-ale-est-eta] band=" << band.name
                << " eta_h=" << std::fixed << std::setprecision(4) << band.eta_h
                << " eta_on="
                << cfg.numerics.ale.band_ale.estimator_band_eta_on
                << " eta_off="
                << cfg.numerics.ale.band_ale.estimator_band_eta_off
                // Pre-update state; transition is logged by log_engagement_change.
                << " engaged=" << (band.engaged ? 1 : 0);
        core::log_info(eta_log.str());
      }
    } else if (per_column_production) {
      band.eta_h = band.pc_eta_p;
    }

    const bool spacing_triggered =
        band.kind == BandKind::Shell &&
        band.shell_min_spacing <
            cfg.numerics.ale.band_ale.shell_min_spacing_frac *
                band.shell_dr0;
    const bool was_engaged = band.engaged;
    if (per_column_production) {
      bool first_engagement = false;
      if (!band.engaged && per_column_trigger_condition(band, cfg)) {
        band.engaged = true;
        first_engagement = true;
        band.pc_release_evaluations = 0;
        ++band.engagements;
      } else if (
          band.engaged &&
          band.pc_eta_p <=
              cfg.numerics.ale.band_ale.estimator_band_eta_off) {
        ++band.pc_release_evaluations;
        if (band.pc_release_evaluations >= 2) {
          band.engaged = false;
          band.pc_release_evaluations = 0;
        }
      } else {
        band.pc_release_evaluations = 0;
      }

      PcDecision decision = PcDecision::Noop;
      band.pc_chi = 0.0;
      if (band.pc_sigma_starvation_events_remaining > 0) {
        band.engaged = false;
        band.pc_release_evaluations = 0;
        decision = PcDecision::Hold;
        --band.pc_sigma_starvation_events_remaining;
        if (band.pc_cooldown_events_remaining > 0) {
          --band.pc_cooldown_events_remaining;
        }
      } else if (!band.engaged) {
        if (band.pc_cooldown_events_remaining > 0) {
          --band.pc_cooldown_events_remaining;
        }
      } else if (band.pc_cooldown_events_remaining > 0) {
        --band.pc_cooldown_events_remaining;
      } else {
        std::string reject_reason;
        if (!prepare_per_column_target(
                state, cfg, band, first_engagement, &reject_reason)) {
          band.pc_reject_reason = reject_reason;
          ++band.skipped_inadmissible;
          decision = PcDecision::Reject;
        } else {
          band.pc_transaction_pending = true;
          band.pc_transaction_rejected = false;
          band.pc_pending_sigma = 1.0;
          band.pc_pending_chi = band.pc_chi;
          decision = PcDecision::Commit;
        }
      }
      if (band.last_counted_step != this_step &&
          band.engaged != was_engaged) {
        log_engagement_change(this_step, band, min_ratio, rank);
      }
      if (band.engaged && band.last_counted_step != this_step) {
        ++band.engaged_steps;
      }
      if (decision == PcDecision::Commit) {
        active.push_back(
            ActiveBand{index, band.name, min_ratio, argmin_cell});
      }
      log_per_column_decision(this_step, band, decision, rank);
      update_per_column_state_for_test(band);
      band.last_counted_step = this_step;
      continue;
    }
    if (band.kind == BandKind::Estimator) {
      if (!band.engaged &&
          band.eta_h >=
              cfg.numerics.ale.band_ale.estimator_band_eta_on) {
        band.engaged = true;
        ++band.engagements;
      } else if (
          band.engaged &&
          band.eta_h <=
              cfg.numerics.ale.band_ale.estimator_band_eta_off) {
        band.engaged = false;
      }
    } else {
      if (!band.engaged &&
          (min_ratio < cfg.numerics.ale.band_ale.aspect_trigger ||
           spacing_triggered)) {
        band.engaged = true;
        ++band.engagements;
      } else if (
          band.engaged &&
          min_ratio >
              cfg.numerics.ale.band_ale.aspect_trigger *
                  cfg.numerics.ale.band_ale.release_hysteresis &&
          (band.kind != BandKind::Shell ||
           band.shell_min_spacing >
               cfg.numerics.ale.band_ale.shell_min_spacing_frac *
                   band.shell_dr0 *
                   cfg.numerics.ale.band_ale.release_hysteresis)) {
        band.engaged = false;
      }
    }
    if (band.last_counted_step != this_step &&
        band.engaged != was_engaged) {
      log_engagement_change(this_step, band, min_ratio, rank);
    }
    if (per_column_shadow) {
      log_per_column_shadow(this_step, band, rank);
    }
    if (diag_every > 0 && rank == 0 &&
        (state.step + 1) % diag_every == 0) {
      std::ostringstream log;
      log << "[band-ale-diag] step=" << state.step + 1
          << " band=" << band.name
          << " min_ratio=" << std::scientific << std::setprecision(6)
          << min_ratio
          << " engaged=" << (band.engaged ? 1 : 0);
      core::log_info(log.str());
    }
    if (band.engaged) {
      if (band.last_counted_step != this_step) {
        ++band.engaged_steps;
      }
      active.push_back(
          ActiveBand{index, band.name, min_ratio, argmin_cell});
    }
    band.last_counted_step = this_step;
  }
  const bool have_non_pole_theta_active = std::any_of(
      active.begin(),
      active.end(),
      [&runtime](const ActiveBand& active_band) {
        return runtime.bands[active_band.index].kind !=
               BandKind::PoleTheta;
      });
  if (have_non_pole_theta_active) {
    active.erase(
        std::remove_if(
            active.begin(),
            active.end(),
            [&runtime](const ActiveBand& active_band) {
              return runtime.bands[active_band.index].kind ==
                     BandKind::PoleTheta;
            }),
        active.end());
  }
  return active;
}

static void build_target_impl(const core::State& state,
                              const core::Config& cfg,
                              const std::size_t band_index,
                              std::vector<double>& target_r,
                              std::vector<double>& target_z) {
  TENRYU_ASSERT(
      g_runtime != nullptr && g_runtime->initialized &&
          band_index < g_runtime->bands.size(),
      "band ALE target requested for invalid runtime band");
  BandState& band = g_runtime->bands[band_index];
  std::vector<double> current_r;
  std::vector<double> current_z;
  state.x_r.copy_to_host(current_r);
  state.x_z.copy_to_host(current_z);
  target_r = current_r;
  target_z = current_z;
  const double chi = cfg.numerics.ale.band_ale.chi;

  if (band.kind == BandKind::ClosureCatchment) {
    if (cfg.numerics.ale.band_ale.closure_catchment_forced_active &&
        cfg.numerics.ale.band_ale.closure_catchment_max_bites > 1) {
      band.engaged = prepare_closure_catchment_target(
          state, cfg, band, g_runtime->rank);
      if (!band.engaged) {
        return;
      }
    }
    TENRYU_ASSERT(
        band.closure_cached_target_r.size() == current_r.size() &&
            band.closure_cached_target_z.size() == current_z.size(),
        "closure catchment cached target is unavailable");
    for (const int node : band.closure_target_nodes) {
      const auto node_index = static_cast<std::size_t>(node);
      target_r[node_index] = band.closure_cached_target_r[node_index];
      target_z[node_index] = band.closure_cached_target_z[node_index];
    }
    return;
  }

  if (band.kind == BandKind::AxisRepair) {
    const auto& topology = *state.mesh.topo.multiblock;
    const std::size_t n_nodes =
        static_cast<std::size_t>(state.mesh.topo.n_nodes);
    TENRYU_ASSERT(
        g_runtime->axis_repair_axis_nodes.size() == n_nodes &&
            g_runtime->axis_repair_node_incident_cells.size() == n_nodes,
        "band ALE axis repair node metadata size mismatch");
    TENRYU_ASSERT(
        g_runtime->axis_repair_current_area.size() ==
            static_cast<std::size_t>(state.mesh.topo.n_cells),
        "band ALE axis repair current-area field size mismatch");
    TENRYU_ASSERT(
        !band.axis_repair_patch_cells.empty(),
        "band ALE axis repair engaged block has an empty patch");

    std::vector<double> initial_r;
    std::vector<double> initial_z;
    state.x_r_initial.copy_to_host(initial_r);
    state.x_z_initial.copy_to_host(initial_z);
    TENRYU_ASSERT(
        initial_r.size() == n_nodes && initial_z.size() == n_nodes,
        "band ALE axis repair requires initial node coordinates");

    struct CornerTarget {
      double winv00 = 0.0;
      double winv01 = 0.0;
      double winv10 = 0.0;
      double winv11 = 0.0;
    };
    struct PatchCell {
      int cell = -1;
      int nverts = 0;
      std::array<int, 4> nodes = {{-1, -1, -1, -1}};
      std::array<CornerTarget, 4> corners;
    };

    const auto& block = topology.blocks[
        static_cast<std::size_t>(band.block_id)];
    std::vector<PatchCell> patch_cells;
    patch_cells.reserve(band.axis_repair_patch_cells.size());
    std::vector<int> patch_nodes;
    for (const int cell : band.axis_repair_patch_cells) {
      TENRYU_ASSERT(
          cell >= block.cell_begin &&
              cell < block.cell_begin + block.cell_count,
          "band ALE axis repair patch cell is outside its tier block");
      PatchCell patch;
      patch.cell = cell;
      patch.nverts =
          mesh::mesh_topo_cell_active_nverts(
              state.mesh.cell_nverts, cell);
      TENRYU_ASSERT(
          patch.nverts >= 3 && patch.nverts <= 4,
          "band ALE axis repair patch requires three- or four-node cells");
      const int begin =
          topology.cell_node_csr_offsets[
              static_cast<std::size_t>(cell)];
      for (int local = 0; local < patch.nverts; ++local) {
        const int node = topology.cell_node_csr_indices[
            static_cast<std::size_t>(begin + local)];
        TENRYU_ASSERT(
            node >= 0 && node < state.mesh.topo.n_nodes,
            "band ALE axis repair patch node id is out of range");
        patch.nodes[static_cast<std::size_t>(local)] = node;
        patch_nodes.push_back(node);
      }
      const int row =
          (cell - block.cell_begin) / block.n_j_cells;
      const double ring_scale =
          axis_repair_ring_scale_for_row(band, row);
      for (int corner = 0; corner < patch.nverts; ++corner) {
        // Match the production corner-Jacobian convention: the columns are
        // the next-corner and previous-corner edge vectors, respectively.
        const int next = (corner + 1) % patch.nverts;
        const int previous =
            (corner + patch.nverts - 1) % patch.nverts;
        const auto corner_node = static_cast<std::size_t>(
            patch.nodes[static_cast<std::size_t>(corner)]);
        const auto next_node = static_cast<std::size_t>(
            patch.nodes[static_cast<std::size_t>(next)]);
        const auto previous_node = static_cast<std::size_t>(
            patch.nodes[static_cast<std::size_t>(previous)]);
        const double w00 =
            ring_scale *
            (initial_r[next_node] - initial_r[corner_node]);
        const double w01 =
            ring_scale *
            (initial_r[previous_node] - initial_r[corner_node]);
        const double w10 =
            ring_scale *
            (initial_z[next_node] - initial_z[corner_node]);
        const double w11 =
            ring_scale *
            (initial_z[previous_node] - initial_z[corner_node]);
        const double determinant = w00 * w11 - w01 * w10;
        TENRYU_ASSERT(
            std::isfinite(determinant) && determinant != 0.0,
            "band ALE axis repair initial corner Jacobian is singular");
        auto& target =
            patch.corners[static_cast<std::size_t>(corner)];
        target.winv00 = w11 / determinant;
        target.winv01 = -w01 / determinant;
        target.winv10 = -w10 / determinant;
        target.winv11 = w00 / determinant;
        TENRYU_ASSERT(
            std::isfinite(target.winv00) &&
                std::isfinite(target.winv01) &&
                std::isfinite(target.winv10) &&
                std::isfinite(target.winv11),
            "band ALE axis repair inverse target Jacobian is non-finite");
      }
      patch_cells.push_back(std::move(patch));
    }
    std::sort(patch_nodes.begin(), patch_nodes.end());
    patch_nodes.erase(
        std::unique(patch_nodes.begin(), patch_nodes.end()),
        patch_nodes.end());

    std::vector<std::uint8_t> patch_cell_mask(
        static_cast<std::size_t>(state.mesh.topo.n_cells), 0U);
    for (const int cell : band.axis_repair_patch_cells) {
      patch_cell_mask[static_cast<std::size_t>(cell)] = 1U;
    }
    std::vector<std::uint8_t> fixed_node(n_nodes, 0U);
    std::vector<double> node_displacement_cap(n_nodes, 0.0);
    for (const int node : patch_nodes) {
      const auto node_index = static_cast<std::size_t>(node);
      if (!band.pinned_nodes.empty() &&
          band.pinned_nodes[node_index] != 0U) {
        fixed_node[node_index] = 1U;
      }
      double minimum_area = std::numeric_limits<double>::infinity();
      for (const int incident_cell :
           g_runtime->axis_repair_node_incident_cells[node_index]) {
        if (patch_cell_mask[
                static_cast<std::size_t>(incident_cell)] == 0U) {
          fixed_node[node_index] = 1U;
          continue;
        }
        const double area =
            g_runtime->axis_repair_current_area[
                static_cast<std::size_t>(incident_cell)];
        TENRYU_ASSERT(
            std::isfinite(area) && area > 0.0,
            "band ALE axis repair patch-cell area must be finite and positive");
        minimum_area = std::min(minimum_area, area);
      }
      TENRYU_ASSERT(
          std::isfinite(minimum_area) && minimum_area > 0.0,
          "band ALE axis repair node has no incident patch-cell area");
      node_displacement_cap[node_index] =
          cfg.numerics.ale.band_ale.axis_repair_cap_rel *
          std::sqrt(minimum_area);
    }

    for (int sweep = 0; sweep < kAxisRepairSweeps; ++sweep) {
      for (const int node : patch_nodes) {
        const auto node_index = static_cast<std::size_t>(node);
        if (fixed_node[node_index] != 0U) {
          continue;
        }
        double gradient_r = 0.0;
        double gradient_z = 0.0;
        double hessian_rr = 0.0;
        double hessian_rz = 0.0;
        double hessian_zz = 0.0;
        for (const auto& patch : patch_cells) {
          int node_local = -1;
          for (int local = 0; local < patch.nverts; ++local) {
            if (patch.nodes[static_cast<std::size_t>(local)] == node) {
              node_local = local;
              break;
            }
          }
          if (node_local < 0) {
            continue;
          }
          for (int corner = 0; corner < patch.nverts; ++corner) {
            const int next = (corner + 1) % patch.nverts;
            const int previous =
                (corner + patch.nverts - 1) % patch.nverts;
            const auto corner_node = static_cast<std::size_t>(
                patch.nodes[static_cast<std::size_t>(corner)]);
            const auto next_node = static_cast<std::size_t>(
                patch.nodes[static_cast<std::size_t>(next)]);
            const auto previous_node = static_cast<std::size_t>(
                patch.nodes[static_cast<std::size_t>(previous)]);
            const double a00 =
                target_r[next_node] - target_r[corner_node];
            const double a01 =
                target_r[previous_node] - target_r[corner_node];
            const double a10 =
                target_z[next_node] - target_z[corner_node];
            const double a11 =
                target_z[previous_node] - target_z[corner_node];
            const auto& reference =
                patch.corners[static_cast<std::size_t>(corner)];
            const double residual00 =
                a00 * reference.winv00 +
                a01 * reference.winv10 - 1.0;
            const double residual01 =
                a00 * reference.winv01 +
                a01 * reference.winv11;
            const double residual10 =
                a10 * reference.winv00 +
                a11 * reference.winv10;
            const double residual11 =
                a10 * reference.winv01 +
                a11 * reference.winv11 - 1.0;
            const double derivative_next =
                (next == node_local ? 1.0 : 0.0) -
                (corner == node_local ? 1.0 : 0.0);
            const double derivative_previous =
                (previous == node_local ? 1.0 : 0.0) -
                (corner == node_local ? 1.0 : 0.0);
            const double derivative_0 =
                derivative_next * reference.winv00 +
                derivative_previous * reference.winv10;
            const double derivative_1 =
                derivative_next * reference.winv01 +
                derivative_previous * reference.winv11;
            gradient_r +=
                2.0 * residual00 * derivative_0;
            gradient_r +=
                2.0 * residual01 * derivative_1;
            gradient_z +=
                2.0 * residual10 * derivative_0;
            gradient_z +=
                2.0 * residual11 * derivative_1;
            hessian_rr +=
                2.0 * derivative_0 * derivative_0;
            hessian_rr +=
                2.0 * derivative_1 * derivative_1;
            hessian_zz +=
                2.0 * derivative_0 * derivative_0;
            hessian_zz +=
                2.0 * derivative_1 * derivative_1;
          }
        }
        TENRYU_ASSERT(
            std::isfinite(gradient_r) &&
                std::isfinite(gradient_z) &&
                std::isfinite(hessian_rr) &&
                std::isfinite(hessian_rz) &&
                std::isfinite(hessian_zz) &&
                hessian_rr > 0.0 && hessian_zz > 0.0,
            "band ALE axis repair local Gauss-Newton system is invalid");

        double step_r = 0.0;
        double step_z = 0.0;
        if (g_runtime->axis_repair_axis_nodes[node_index] != 0U) {
          step_z = -gradient_z / hessian_zz;
        } else {
          const double determinant =
              hessian_rr * hessian_zz -
              hessian_rz * hessian_rz;
          TENRYU_ASSERT(
              std::isfinite(determinant) && determinant > 0.0,
              "band ALE axis repair local Gauss-Newton Hessian is singular");
          step_r =
              (-hessian_zz * gradient_r +
               hessian_rz * gradient_z) /
              determinant;
          step_z =
              (hessian_rz * gradient_r -
               hessian_rr * gradient_z) /
              determinant;
        }
        TENRYU_ASSERT(
            std::isfinite(step_r) && std::isfinite(step_z),
            "band ALE axis repair Newton step is non-finite");

        double displacement_r =
            target_r[node_index] + step_r - current_r[node_index];
        double displacement_z =
            target_z[node_index] + step_z - current_z[node_index];
        if (g_runtime->axis_repair_axis_nodes[node_index] != 0U) {
          displacement_r = 0.0;
        }
        const double displacement_norm =
            std::hypot(displacement_r, displacement_z);
        const double cap = node_displacement_cap[node_index];
        if (displacement_norm > cap) {
          const double scale = cap / displacement_norm;
          displacement_r *= scale;
          displacement_z *= scale;
        }
        target_r[node_index] =
            current_r[node_index] + displacement_r;
        target_z[node_index] =
            current_z[node_index] + displacement_z;
      }
    }
    // The driver reports only rejection. Arm provisionally here and cancel
    // in record_inadmissible(); a surviving arm therefore means acceptance.
    band.axis_repair_cooldown_steps_remaining =
        kAxisRepairCooldownSteps;
    band.axis_repair_cooldown_armed_step = state.step + 1;
    return;
  }

  if (band.kind == BandKind::Shell) {
    TENRYU_ASSERT(
        cfg.numerics.ale.band_ale.shell_target == "respace",
        "band ALE shell target must be respace");
    apply_radial_respace_target(
        state,
        band,
        current_r,
        current_z,
        chi,
        cfg.numerics.ale.band_ale.respace_move_cap_frac,
        false,
        target_r,
        target_z);
    return;
  }

  if (band.kind == BandKind::Estimator) {
    TENRYU_ASSERT(
        cfg.numerics.ale.band_ale.bands == "estimator",
        "band ALE estimator target must be respace");
    if (cfg.numerics.ale.band_ale.estimator_band_per_column &&
        cfg.numerics.hydro.pressure_drive_perturbation.enabled) {
      TENRYU_ASSERT(
          band.pc_cached_target_r.size() == current_r.size() &&
              band.pc_cached_target_z.size() == current_z.size(),
          "band ALE per-column cached target is unavailable");
      target_r = band.pc_cached_target_r;
      target_z = band.pc_cached_target_z;
      return;
    }
    if (band.estimator_polar_geometry) {
      apply_radial_respace_target(
          state,
          band,
          current_r,
          current_z,
          chi,
          cfg.numerics.ale.band_ale.respace_move_cap_frac,
          true,
          target_r,
          target_z);
    } else {
      apply_layer_respace_target(
          state,
          band,
          current_r,
          current_z,
          chi,
          target_r,
          target_z);
    }
    return;
  }

  if (band.kind == BandKind::Belt) {
    if (cfg.numerics.ale.band_ale.center_target == "line") {
      if (cfg.numerics.ale.band_ale.belt_target == "respace") {
        apply_radial_respace_target(
            state,
            band,
            current_r,
            current_z,
            chi,
            cfg.numerics.ale.band_ale.respace_move_cap_frac,
            false,
            target_r,
            target_z);
        return;
      }
      for (const auto& ring : band.lines) {
        TENRYU_ASSERT(
            !ring.empty(),
            "band ALE belt ring must be non-empty");
        double radius_sum = 0.0;
        for (const int node : ring) {
          radius_sum += std::hypot(
              current_r[static_cast<std::size_t>(node)],
              current_z[static_cast<std::size_t>(node)]);
        }
        const double mean_radius =
            radius_sum / static_cast<double>(ring.size());
        for (const int node : ring) {
          const auto node_index = static_cast<std::size_t>(node);
          if (band.pinned_nodes[node_index] != 0U) {
            continue;
          }
          const double radius =
              std::hypot(current_r[node_index], current_z[node_index]);
          if (!(radius > 0.0)) {
            continue;
          }
          const double scale = mean_radius / radius;
          const double ideal_r = current_r[node_index] * scale;
          const double ideal_z = current_z[node_index] * scale;
          if ((state.mesh.topo.node_flags[node_index] &
               mesh::NODE_POLE_AXIS) == 0U) {
            target_r[node_index] =
                current_r[node_index] + chi * (ideal_r - current_r[node_index]);
          }
          target_z[node_index] =
              current_z[node_index] + chi * (ideal_z - current_z[node_index]);
        }
      }
      return;
    }

    std::vector<double> ideal_r = current_r;
    std::vector<double> ideal_z = current_z;
    if (cfg.numerics.ale.band_ale.belt_target == "respace") {
      std::vector<double> initial_r;
      std::vector<double> initial_z;
      state.x_r_initial.copy_to_host(initial_r);
      state.x_z_initial.copy_to_host(initial_z);
      std::vector<double> mean_radius(band.lines.size(), 0.0);
      std::vector<double> initial_mean_radius(band.lines.size(), 0.0);
      for (std::size_t line = 0; line < band.lines.size(); ++line) {
        const auto& ring = band.lines[line];
        TENRYU_ASSERT(
            !ring.empty(),
            "band ALE belt ring must be non-empty");
        for (const int node : ring) {
          const auto node_index = static_cast<std::size_t>(node);
          mean_radius[line] +=
              std::hypot(current_r[node_index], current_z[node_index]);
          initial_mean_radius[line] +=
              std::hypot(initial_r[node_index], initial_z[node_index]);
        }
        mean_radius[line] /= static_cast<double>(ring.size());
        initial_mean_radius[line] /= static_cast<double>(ring.size());
      }
      const double initial_span =
          initial_mean_radius.front() - initial_mean_radius.back();
      const double current_inner_radius = mean_radius.back();
      const double current_span =
          mean_radius.front() - current_inner_radius;
      for (std::size_t line = 0; line < band.lines.size(); ++line) {
        const double fraction =
            (initial_mean_radius[line] - initial_mean_radius.back()) /
            initial_span;
        const double target_radius =
            current_inner_radius + fraction * current_span;
        for (const int node : band.lines[line]) {
          const auto node_index = static_cast<std::size_t>(node);
          if (band.pinned_nodes[node_index] != 0U) {
            continue;
          }
          const double radius =
              std::hypot(current_r[node_index], current_z[node_index]);
          if (!(radius > 0.0)) {
            continue;
          }
          const double scale = target_radius / radius;
          const double line_target_r = current_r[node_index] * scale;
          const double line_target_z = current_z[node_index] * scale;
          if ((state.mesh.topo.node_flags[node_index] &
               mesh::NODE_POLE_AXIS) == 0U) {
            ideal_r[node_index] = line_target_r;
          }
          ideal_z[node_index] = line_target_z;
        }
      }
    } else {
      for (const auto& ring : band.lines) {
        TENRYU_ASSERT(
            !ring.empty(),
            "band ALE belt ring must be non-empty");
        double radius_sum = 0.0;
        for (const int node : ring) {
          radius_sum += std::hypot(
              current_r[static_cast<std::size_t>(node)],
              current_z[static_cast<std::size_t>(node)]);
        }
        const double mean_radius =
            radius_sum / static_cast<double>(ring.size());
        for (const int node : ring) {
          const auto node_index = static_cast<std::size_t>(node);
          if (band.pinned_nodes[node_index] != 0U) {
            continue;
          }
          const double radius =
              std::hypot(current_r[node_index], current_z[node_index]);
          if (!(radius > 0.0)) {
            continue;
          }
          const double scale = mean_radius / radius;
          const double line_target_r = current_r[node_index] * scale;
          const double line_target_z = current_z[node_index] * scale;
          if ((state.mesh.topo.node_flags[node_index] &
               mesh::NODE_POLE_AXIS) == 0U) {
            ideal_r[node_index] = line_target_r;
          }
          ideal_z[node_index] = line_target_z;
        }
      }
    }

    // volume_fraction measured net-negative during shock transit
    // (champion 3.216 vs 3.229); retained as an experiment surface.
    for (const auto& pentagon : band.center_pentagons) {
      if (!optimize_center_target(
              pentagon,
              current_r,
              current_z,
              ideal_r,
              ideal_z)) {
        ++g_runtime->center_fallbacks;
      }
    }
    for (const auto& ring : band.lines) {
      for (const int node : ring) {
        const auto node_index = static_cast<std::size_t>(node);
        if (band.pinned_nodes[node_index] != 0U) {
          continue;
        }
        if ((state.mesh.topo.node_flags[node_index] &
             mesh::NODE_POLE_AXIS) == 0U) {
          target_r[node_index] =
              current_r[node_index] +
              chi * (ideal_r[node_index] - current_r[node_index]);
        }
        target_z[node_index] =
            current_z[node_index] +
            chi * (ideal_z[node_index] - current_z[node_index]);
      }
    }
    return;
  }

  if (band.kind == BandKind::PoleTheta) {
    return;
  }

  TENRYU_ASSERT(
      band.cells.size() % 2U == 0U,
      "band ALE axis cell list must contain row-major cell pairs");
  const auto argmin =
      std::find(band.cells.begin(), band.cells.end(), band.argmin_cell);
  TENRYU_ASSERT(
      argmin != band.cells.end(),
      "band ALE axis argmin cell is absent from the band cell list");
  const std::size_t argmin_position =
      static_cast<std::size_t>(argmin - band.cells.begin());
  const std::size_t argmin_row = argmin_position / 2U;
  const std::size_t n_rows = band.cells.size() / 2U;
  const std::size_t halfwidth = static_cast<std::size_t>(
      cfg.numerics.ale.band_ale.axis_segment_halfwidth);
  const std::size_t row_lo =
      argmin_row > halfwidth ? argmin_row - halfwidth : 0U;
  const std::size_t row_hi =
      std::min(n_rows - 1U, argmin_row + halfwidth);
  const std::size_t point_lo = row_lo;
  const std::size_t point_hi = row_hi + 1U;

  if (cfg.numerics.ale.band_ale.axis_target == "respace") {
    std::vector<double> initial_r;
    std::vector<double> initial_z;
    state.x_r_initial.copy_to_host(initial_r);
    state.x_z_initial.copy_to_host(initial_z);
    for (const auto& line : band.lines) {
      TENRYU_ASSERT(
          line.size() == n_rows + 1U,
          "band ALE axis line length does not match its cell rows");
      const auto first_index =
          static_cast<std::size_t>(line[point_lo]);
      const auto last_index =
          static_cast<std::size_t>(line[point_hi]);
      const double radius_first =
          std::hypot(current_r[first_index], current_z[first_index]);
      const double radius_last =
          std::hypot(current_r[last_index], current_z[last_index]);
      const double initial_radius_first =
          std::hypot(initial_r[first_index], initial_z[first_index]);
      const double initial_radius_last =
          std::hypot(initial_r[last_index], initial_z[last_index]);
      const double initial_span =
          initial_radius_first - initial_radius_last;
      const double current_span = radius_first - radius_last;
      for (std::size_t point = point_lo + 1U;
           point < point_hi;
           ++point) {
        const auto node_index =
            static_cast<std::size_t>(line[point]);
        if (band.pinned_nodes[node_index] != 0U) {
          continue;
        }
        const double radius =
            std::hypot(current_r[node_index], current_z[node_index]);
        if (!(radius > 0.0)) {
          continue;
        }
        const double initial_radius =
            std::hypot(initial_r[node_index], initial_z[node_index]);
        const double fraction =
            (initial_radius - initial_radius_last) / initial_span;
        const double target_radius =
            radius_last + fraction * current_span;
        const double scale = target_radius / radius;
        const double ideal_r = current_r[node_index] * scale;
        const double ideal_z = current_z[node_index] * scale;
        if ((state.mesh.topo.node_flags[node_index] &
             mesh::NODE_POLE_AXIS) == 0U) {
          target_r[node_index] =
              current_r[node_index] +
              chi * (ideal_r - current_r[node_index]);
        }
        target_z[node_index] =
            current_z[node_index] +
            chi * (ideal_z - current_z[node_index]);
      }
    }
    return;
  }

  for (const auto& line : band.lines) {
    TENRYU_ASSERT(
        line.size() == n_rows + 1U,
        "band ALE axis line length does not match its cell rows");
    for (std::size_t point = point_lo + 1U;
         point < point_hi;
         ++point) {
      const auto node_index =
          static_cast<std::size_t>(line[point]);
      if (band.pinned_nodes[node_index] != 0U) {
        continue;
      }
      const double ideal_z =
          0.5 *
          (current_z[static_cast<std::size_t>(line[point - 1U])] +
           current_z[static_cast<std::size_t>(line[point + 1U])]);
      target_z[node_index] =
          current_z[node_index] + chi * (ideal_z - current_z[node_index]);
    }
  }
}

void build_target(const core::State& state,
                  const core::Config& cfg,
                  const std::size_t band_index,
                  std::vector<double>& target_r,
                  std::vector<double>& target_z) {
  build_target_impl(state, cfg, band_index, target_r, target_z);
  compose_pole_theta_target(state, cfg, target_r, target_z);
}

bool is_closure_catchment(const std::size_t band_index) {
  TENRYU_ASSERT(
      g_runtime != nullptr && g_runtime->initialized &&
          band_index < g_runtime->bands.size(),
      "band ALE closure catchment query requested for invalid runtime band");
  const auto& band = g_runtime->bands[band_index];
  return band.kind == BandKind::ClosureCatchment;
}

bool is_pole_theta(const std::size_t band_index) {
  TENRYU_ASSERT(
      g_runtime != nullptr && g_runtime->initialized &&
          band_index < g_runtime->bands.size(),
      "band ALE pole-theta query requested for invalid runtime band");
  const auto& band = g_runtime->bands[band_index];
  return band.kind == BandKind::PoleTheta;
}

bool wants_reapply(const std::size_t band_index) {
  TENRYU_ASSERT(
      g_runtime != nullptr && g_runtime->initialized &&
          band_index < g_runtime->bands.size(),
      "band ALE reapply query requested for invalid runtime band");
  const auto& band = g_runtime->bands[band_index];
  return band.kind == BandKind::ClosureCatchment &&
         band.engaged && band.closure_prescale < 1.0;
}

void compose_pole_theta_target(const core::State& state,
                               const core::Config& cfg,
                               std::vector<double>& target_r,
                               std::vector<double>& target_z) {
  if (g_runtime == nullptr || !g_runtime->initialized) {
    return;
  }
  g_runtime->pole_theta_composed_last_build = false;
  const bool have_pole_theta = std::any_of(
      g_runtime->bands.begin(),
      g_runtime->bands.end(),
      [](const BandState& band) {
        return band.kind == BandKind::PoleTheta;
      });
  if (!have_pole_theta || g_runtime->pole_theta_dtheta.empty() ||
      g_runtime->pole_theta_consumed) {
    return;
  }

  const bool have_curve_targets =
      cfg.numerics.ale.band_ale.pole_theta_curve_preserving &&
      std::any_of(
          g_runtime->pole_theta_target_s.begin(),
          g_runtime->pole_theta_target_s.end(),
          [](const double target_s) { return target_s > 0.0; });
  std::vector<double> current_r;
  std::vector<double> current_z;
  if (have_curve_targets) {
    state.x_r.copy_to_host(current_r);
    state.x_z.copy_to_host(current_z);
  }

  bool composed = false;
  for (std::size_t node = 0;
       node < g_runtime->pole_theta_dtheta.size();
       ++node) {
    const double dtheta = g_runtime->pole_theta_dtheta[node];
    if (dtheta == 0.0) {
      continue;
    }
    composed = true;
    const double rt = target_r[node];
    const double zt = target_z[node];
    const double st_composed = std::hypot(rt, zt);
    const double st_curve =
        node < g_runtime->pole_theta_target_s.size()
            ? g_runtime->pole_theta_target_s[node]
            : 0.0;
    double st = st_composed;
    if (st_curve > 0.0) {
      const double st_current = std::hypot(current_r[node], current_z[node]);
      if (std::fabs(st_composed - st_current) < 0.1 * st_current) {
        st = st_curve;
      }
    }
    const double theta =
        std::atan2(std::max(rt, 0.0), zt) + dtheta;
    target_r[node] = st * std::sin(theta);
    target_z[node] = st * std::cos(theta);
  }
  g_runtime->pole_theta_composed_last_build = composed;
}

void note_pole_theta_consumed() {
  if (g_runtime != nullptr && g_runtime->initialized) {
    g_runtime->pole_theta_consumed = true;
  }
}

bool pole_theta_composed_last_build() {
  return g_runtime != nullptr && g_runtime->initialized &&
         g_runtime->pole_theta_composed_last_build;
}

void record_inadmissible(const std::size_t band_index) {
  TENRYU_ASSERT(
      g_runtime != nullptr && g_runtime->initialized &&
          band_index < g_runtime->bands.size(),
      "band ALE inadmissible counter requested for invalid runtime band");
  auto& band = g_runtime->bands[band_index];
  ++band.skipped_inadmissible;
  if (band.kind == BandKind::Estimator &&
      band.pc_transaction_pending) {
    band.pc_transaction_rejected = true;
  }
  if (band.kind == BandKind::AxisRepair) {
    band.axis_repair_cooldown_steps_remaining = 0;
    band.axis_repair_cooldown_armed_step = -1;
  }
}

void record_partial_sigma(const std::size_t band_index,
                          const double sigma) {
  TENRYU_ASSERT(
      g_runtime != nullptr && g_runtime->initialized &&
          band_index < g_runtime->bands.size(),
      "band ALE partial sigma counter requested for invalid runtime band");
  auto& band = g_runtime->bands[band_index];
  ++band.applied_partial_sigma;
  band.min_sigma_applied = std::min(band.min_sigma_applied, sigma);
  if (band.kind == BandKind::Estimator &&
      band.pc_transaction_pending) {
    band.pc_pending_sigma = sigma;
  }
}

const std::vector<int>* band_ale_band_cells_for_test(
    const char* band_name) {
  if (g_runtime == nullptr || !g_runtime->initialized ||
      band_name == nullptr) {
    return nullptr;
  }
  const auto band = std::find_if(
      g_runtime->bands.begin(),
      g_runtime->bands.end(),
      [band_name](const BandState& candidate) {
        return candidate.name == band_name;
      });
  return band == g_runtime->bands.end() ? nullptr : &band->cells;
}

bool band_ale_band_engaged_for_test(const char* band_name) {
  if (g_runtime == nullptr || !g_runtime->initialized ||
      band_name == nullptr) {
    return false;
  }
  const auto band = std::find_if(
      g_runtime->bands.begin(),
      g_runtime->bands.end(),
      [band_name](const BandState& candidate) {
        return candidate.name == band_name;
      });
  return band != g_runtime->bands.end() && band->engaged;
}

double band_ale_band_eta_h_for_test(const char* band_name) {
  if (g_runtime == nullptr || !g_runtime->initialized ||
      band_name == nullptr) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const auto band = std::find_if(
      g_runtime->bands.begin(),
      g_runtime->bands.end(),
      [band_name](const BandState& candidate) {
        return candidate.name == band_name;
      });
  return band == g_runtime->bands.end()
             ? std::numeric_limits<double>::quiet_NaN()
             : band->eta_h;
}

const PerColumnStateForTest* band_ale_per_column_state_for_test() {
  if (g_runtime == nullptr || !g_runtime->initialized) {
    return nullptr;
  }
  const auto band = std::find_if(
      g_runtime->bands.begin(),
      g_runtime->bands.end(),
      [](const BandState& candidate) {
        return candidate.kind == BandKind::Estimator &&
               !candidate.pc_state_for_test.i0.empty();
      });
  return band == g_runtime->bands.end()
             ? nullptr
             : &band->pc_state_for_test;
}

void run_end(const core::Config& cfg, const int rank) {
  if (!cfg.numerics.ale.band_ale.enabled || g_runtime == nullptr) {
    return;
  }
  if (rank == 0 && g_runtime->initialized) {
    core::log_info(
        "[band-ale-summary] skipped_uncrosschecked_belts=" +
        std::to_string(g_runtime->skipped_uncrosschecked_belts) +
        " skipped_shell_axis_blocks=" +
        std::to_string(g_runtime->skipped_shell_axis_blocks) +
        " skipped_trivial_axis_blocks=" +
        std::to_string(g_runtime->skipped_trivial_axis_blocks) +
        " center_fallbacks=" +
        std::to_string(g_runtime->center_fallbacks));
    if (cfg.numerics.ale.band_ale.axis_repair_enabled) {
      core::log_info(
          "[axis-repair-summary] total=" +
          std::to_string(g_runtime->axis_repair_total_transactions));
    }
    for (const auto& band : g_runtime->bands) {
      if (band.kind == BandKind::AxisRepair) {
        continue;
      }
      std::ostringstream log;
      log << "[band-ale-summary] band=" << band.name
          << " engaged_steps=" << band.engaged_steps
          << " engagements=" << band.engagements
          << " skipped_inadmissible=" << band.skipped_inadmissible
          << " applied_partial_sigma=" << band.applied_partial_sigma
          << " min_sigma=" << std::scientific << std::setprecision(6)
          << band.min_sigma_applied;
      core::log_info(log.str());
    }
  }
  delete g_runtime;
  g_runtime = nullptr;
}

}  // namespace tenryu::hydro::band_ale
