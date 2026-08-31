#include "core/zoning_intent.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tenryu::core {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kPositionToleranceScale = 1.0e-12;
constexpr double kMuFloor = 1.0e-300;
constexpr double kIntegralTolerance = 1.0e-12;
constexpr double kBinTolerance = 1.0e-10;
constexpr double kCapacityRoundoff = 1.0e-12;
constexpr double kEnvelopeLogGuard = 100.0;
constexpr double kProjectionCheckTolerance = 1.0e-12;
constexpr double kFeasibilityWindowTolerance = 1.0e-12;
constexpr double kWidthCheckTolerance = 1.0e-10;
constexpr double kRatioCheckTolerance = 1.0e-6;
constexpr double kBandOverlapTolerance = 1.0e-9;
constexpr double kClosureCheckTolerance = 1.0e-8;
constexpr double kMaxAnchorLogAmplitude = 9.210340371976184;  // ln(1e4)
constexpr double kMaxAnchorLogAggregate = 13.815510557964274;  // ln(1e6)
constexpr int kInitialQuadratureBins = 8;
constexpr int kMaximumQuadratureBins = 16384;
constexpr int kNewtonIterations = 3;
constexpr int kBisectionIterations = 200;

enum class IntegrandKind {
  kMeasure,
  kMonitor,
};

struct SamplePair {
  double measure = 0.0;
  double monitor = 0.0;
};

struct IntegralTable {
  std::vector<double> values;
  std::vector<double> mid_values;
  std::vector<double> cumulative;
  double total = 0.0;
  double max_simpson_trapezoid_difference = 0.0;
};

struct PanelTable {
  double r_begin = 0.0;
  double r_end = 0.0;
  int n_bins = 0;
  std::vector<double> coordinates;
  IntegralTable measure;
  IntegralTable monitor;
};

struct SegmentData {
  double r_begin = 0.0;
  double r_end = 0.0;
  std::vector<PanelTable> panels;
  double measure_total = 0.0;
  double monitor_total = 0.0;
  int n_min = 0;
  int n_max = 0;
  int n_cells = 0;
};

struct IntegrandEvaluator {
  ZoningMeasure measure = ZoningMeasure::kWidth;
  const std::vector<ZoningProfilePoint>* profile = nullptr;
  const std::vector<ZoningAnchor>* anchors = nullptr;
  const std::function<double(double)>* rho0 = nullptr;
  bool density_invalid = false;
  double invalid_coordinate = 0.0;
  double invalid_density = 0.0;

  double preferred_weight(const double r) const {
    double log_weight = 0.0;
    if (!profile->empty()) {
      if (r <= profile->front().r) {
        log_weight = std::log(profile->front().w);
      } else if (r >= profile->back().r) {
        log_weight = std::log(profile->back().w);
      } else {
        const auto upper = std::upper_bound(
            profile->begin(), profile->end(), r,
            [](const double value, const ZoningProfilePoint& point) {
              return value < point.r;
            });
        const auto lower = upper - 1;
        const double fraction = (r - lower->r) / (upper->r - lower->r);
        log_weight = std::log(lower->w) +
                     fraction * (std::log(upper->w) - std::log(lower->w));
      }
    }
    for (const ZoningAnchor& anchor : *anchors) {
      const double u = (r - anchor.r) / anchor.half_width;
      if (std::abs(u) < 1.0) {
        log_weight +=
            anchor.log_amplitude * 0.5 * (1.0 + std::cos(kPi * u));
      }
    }
    return std::exp(log_weight);
  }

  double measure_density(const double r) {
    if (measure == ZoningMeasure::kWidth) {
      return 1.0;
    }

    const double density = (*rho0)(r);
    if (!std::isfinite(density) || density < 0.0) {
      if (!density_invalid) {
        density_invalid = true;
        invalid_coordinate = r;
        invalid_density = density;
      }
      return 0.0;
    }

    switch (measure) {
      case ZoningMeasure::kArealMass:
        return density;
      case ZoningMeasure::kCylindricalLineMass:
        return density * 2.0 * kPi * r;
      case ZoningMeasure::kSphericalCellMass:
        return density * 4.0 * kPi * r * r;
      case ZoningMeasure::kWidth:
        return 1.0;
    }
    return 0.0;
  }

  SamplePair sample(const double r) {
    SamplePair pair;
    pair.measure = measure_density(r);
    pair.monitor = pair.measure / preferred_weight(r);
    return pair;
  }
};

std::string format_double(const double value) {
  std::ostringstream stream;
  stream << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
  return stream.str();
}

ZoningResult failure_result(const ZoningStatus status,
                            std::string code,
                            std::string message,
                            const std::vector<std::string>& warnings = {}) {
  ZoningResult result;
  result.diag.status = status;
  result.diag.code = std::move(code);
  result.diag.message = std::move(message);
  result.diag.warnings = warnings;
  return result;
}

ZoningResult density_failure(const IntegrandEvaluator& evaluator,
                             const std::vector<std::string>& warnings) {
  return failure_result(
      ZoningStatus::kInvalidInput, "MESH_DENSITY_INVALID",
      "rho0 sample at r=" + format_double(evaluator.invalid_coordinate) +
          " is non-finite or negative: " + format_double(evaluator.invalid_density),
      warnings);
}

bool bitwise_equal(const double lhs, const double rhs) {
  return std::bit_cast<std::uint64_t>(lhs) == std::bit_cast<std::uint64_t>(rhs);
}

IntegralTable integrate_samples(const std::vector<double>& coordinates,
                                const std::vector<double>& values,
                                const std::vector<double>& mid_values) {
  IntegralTable table;
  table.values = values;
  table.mid_values = mid_values;
  table.cumulative.resize(coordinates.size(), 0.0);
  for (std::size_t i = 0; i + 1 < coordinates.size(); ++i) {
    const double h = coordinates[i + 1] - coordinates[i];
    const double simpson =
        (h / 6.0) * (values[i] + 4.0 * mid_values[i] + values[i + 1]);
    const double trapezoid = (h / 2.0) * (values[i] + values[i + 1]);
    table.cumulative[i + 1] = table.cumulative[i] + simpson;
    table.max_simpson_trapezoid_difference = std::max(
        table.max_simpson_trapezoid_difference, std::abs(simpson - trapezoid));
  }
  table.total = table.cumulative.back();
  return table;
}

PanelTable build_panel(const double r_begin,
                       const double r_end,
                       const int n_bins,
                       IntegrandEvaluator& evaluator) {
  PanelTable panel;
  panel.r_begin = r_begin;
  panel.r_end = r_end;
  panel.n_bins = n_bins;
  panel.coordinates.resize(static_cast<std::size_t>(n_bins) + 1U);
  std::vector<double> measure_values(static_cast<std::size_t>(n_bins) + 1U);
  std::vector<double> monitor_values(static_cast<std::size_t>(n_bins) + 1U);
  std::vector<double> measure_mid_values(static_cast<std::size_t>(n_bins));
  std::vector<double> monitor_mid_values(static_cast<std::size_t>(n_bins));

  const double h = (r_end - r_begin) / static_cast<double>(n_bins);
  for (int k = 0; k <= n_bins; ++k) {
    double coordinate = r_begin + static_cast<double>(k) * h;
    if (k == 0) {
      coordinate = r_begin;
    } else if (k == n_bins) {
      coordinate = r_end;
    }
    panel.coordinates[static_cast<std::size_t>(k)] = coordinate;

    double sample_coordinate = coordinate;
    if (k == 0) {
      sample_coordinate = std::nextafter(r_begin, r_end);
    } else if (k == n_bins) {
      sample_coordinate = std::nextafter(r_end, r_begin);
    }
    const SamplePair pair = evaluator.sample(sample_coordinate);
    measure_values[static_cast<std::size_t>(k)] = pair.measure;
    monitor_values[static_cast<std::size_t>(k)] = pair.monitor;
  }

  for (int k = 0; k < n_bins; ++k) {
    const double midpoint =
        (panel.coordinates[static_cast<std::size_t>(k)] +
         panel.coordinates[static_cast<std::size_t>(k + 1)]) /
        2.0;
    const SamplePair pair = evaluator.sample(midpoint);
    measure_mid_values[static_cast<std::size_t>(k)] = pair.measure;
    monitor_mid_values[static_cast<std::size_t>(k)] = pair.monitor;
  }

  panel.measure = integrate_samples(panel.coordinates, measure_values, measure_mid_values);
  panel.monitor = integrate_samples(panel.coordinates, monitor_values, monitor_mid_values);
  return panel;
}

bool integral_converged(const IntegralTable& current, const IntegralTable& previous) {
  const bool total_converged =
      std::abs(current.total - previous.total) <=
      kIntegralTolerance * std::max(std::abs(current.total), kMuFloor);
  const bool bins_converged =
      current.max_simpson_trapezoid_difference <=
      kBinTolerance * std::max(current.total, kMuFloor);
  return total_converged && bins_converged;
}

bool build_refined_panel(const double r_begin,
                         const double r_end,
                         IntegrandEvaluator& evaluator,
                         PanelTable& output) {
  PanelTable previous =
      build_panel(r_begin, r_end, kInitialQuadratureBins / 2, evaluator);
  if (evaluator.density_invalid) {
    return false;
  }

  for (int n_bins = kInitialQuadratureBins;
       n_bins <= kMaximumQuadratureBins;
       n_bins *= 2) {
    PanelTable current = build_panel(r_begin, r_end, n_bins, evaluator);
    if (evaluator.density_invalid) {
      return false;
    }
    if (integral_converged(current.measure, previous.measure) &&
        integral_converged(current.monitor, previous.monitor)) {
      output = std::move(current);
      return true;
    }
    if (n_bins == kMaximumQuadratureBins) {
      return false;
    }
    previous = std::move(current);
  }
  return false;
}

double quadratic_value(const IntegralTable& table,
                       const std::size_t bin,
                       const double fraction) {
  const double f0 = table.values[bin];
  const double fm = table.mid_values[bin];
  const double f1 = table.values[bin + 1U];
  const double coefficient_a = 2.0 * (f1 + f0 - 2.0 * fm);
  const double coefficient_b = 4.0 * fm - 3.0 * f0 - f1;
  return (coefficient_a * fraction + coefficient_b) * fraction + f0;
}

double quadratic_integral(const IntegralTable& table,
                          const std::size_t bin,
                          const double fraction,
                          const double width) {
  const double f0 = table.values[bin];
  const double fm = table.mid_values[bin];
  const double f1 = table.values[bin + 1U];
  const double coefficient_a = 2.0 * (f1 + f0 - 2.0 * fm);
  const double coefficient_b = 4.0 * fm - 3.0 * f0 - f1;
  return width *
         (f0 * fraction + 0.5 * coefficient_b * fraction * fraction +
          (coefficient_a / 3.0) * fraction * fraction * fraction);
}

const IntegralTable& select_table(const PanelTable& panel, const IntegrandKind kind) {
  return kind == IntegrandKind::kMeasure ? panel.measure : panel.monitor;
}

double segment_total(const SegmentData& segment, const IntegrandKind kind) {
  return kind == IntegrandKind::kMeasure ? segment.measure_total : segment.monitor_total;
}

double evaluate_cumulative(const SegmentData& segment,
                           const double r,
                           const IntegrandKind kind) {
  if (r <= segment.r_begin) {
    return 0.0;
  }
  if (r >= segment.r_end) {
    return segment_total(segment, kind);
  }

  double offset = 0.0;
  for (const PanelTable& panel : segment.panels) {
    const IntegralTable& table = select_table(panel, kind);
    if (r >= panel.r_end) {
      offset += table.total;
      continue;
    }
    if (r <= panel.r_begin) {
      return offset;
    }

    const auto upper = std::upper_bound(panel.coordinates.begin(),
                                        panel.coordinates.end(), r);
    const std::size_t bin =
        static_cast<std::size_t>(upper - panel.coordinates.begin() - 1);
    const double bin_begin = panel.coordinates[bin];
    const double width = panel.coordinates[bin + 1U] - bin_begin;
    const double fraction = (r - bin_begin) / width;
    return offset + table.cumulative[bin] +
           quadratic_integral(table, bin, fraction, width);
  }
  return segment_total(segment, kind);
}

double invert_cumulative(const SegmentData& segment,
                         const double target,
                         const IntegrandKind kind) {
  if (target <= 0.0) {
    return segment.r_begin;
  }
  if (target >= segment_total(segment, kind)) {
    return segment.r_end;
  }

  std::vector<double> panel_cumulative;
  panel_cumulative.reserve(segment.panels.size());
  double running_total = 0.0;
  for (const PanelTable& panel : segment.panels) {
    running_total += select_table(panel, kind).total;
    panel_cumulative.push_back(running_total);
  }
  const auto panel_endpoint =
      std::lower_bound(panel_cumulative.begin(), panel_cumulative.end(), target);
  const std::size_t panel_index =
      static_cast<std::size_t>(panel_endpoint - panel_cumulative.begin());
  const PanelTable& panel = segment.panels[panel_index];
  const IntegralTable& table = select_table(panel, kind);
  const double offset = panel_index == 0U ? 0.0 : panel_cumulative[panel_index - 1U];
  const double local_target = target - offset;
  const auto endpoint = std::lower_bound(table.cumulative.begin() + 1,
                                         table.cumulative.end(), local_target);
  const std::size_t bin =
      static_cast<std::size_t>(endpoint - table.cumulative.begin() - 1);
  const double bin_integral = table.cumulative[bin + 1U] - table.cumulative[bin];
  const double linear_fraction =
      std::clamp((local_target - table.cumulative[bin]) / bin_integral, 0.0, 1.0);
  const double bin_begin = panel.coordinates[bin];
  const double bin_end = panel.coordinates[bin + 1U];
  const double width = bin_end - bin_begin;
  const double linear_estimate = bin_begin + linear_fraction * width;
  double estimate = linear_estimate;

  for (int iteration = 0; iteration < kNewtonIterations; ++iteration) {
    const double fraction = (estimate - bin_begin) / width;
    const double derivative = std::max(quadratic_value(table, bin, fraction), 0.0);
    if (derivative < kMuFloor) {
      return linear_estimate;
    }
    const double residual =
        table.cumulative[bin] +
        quadratic_integral(table, bin, fraction, width) - local_target;
    estimate = std::clamp(estimate - residual / derivative, bin_begin, bin_end);
  }
  return estimate;
}

enum class ProjectionOutcome {
  kOk = 0,
  kInfeasible = 1,        // certificate in failure_message
  kNumericalFailure = 2,  // post-check violation in failure_message
};

ProjectionOutcome project_cell_measures(const SegmentData& segment,
                                        const ZoningIntentConfig& cfg,
                                        const std::vector<double>& lo_bounds,
                                        const std::vector<double>& hi_bounds,
                                        const std::vector<double>& preferred,
                                        std::vector<double>& projected,
                                        std::string& failure_message) {
  for (std::size_t i = 0; i < preferred.size(); ++i) {
    if (!(preferred[i] > 0.0) || !std::isfinite(preferred[i])) {
      failure_message = "cell " + std::to_string(i) +
                        " has non-positive measure " + format_double(preferred[i]);
      return ProjectionOutcome::kNumericalFailure;
    }
  }

  const std::size_t n = preferred.size();
  const double total_measure = segment.measure_total;
  const double scale = total_measure / static_cast<double>(n);
  const double ratio_limit = std::log(cfg.ratio_hard_max);
  std::vector<double> x0(n, 0.0);
  std::vector<double> envelope_lo(n, 0.0);
  std::vector<double> envelope_hi(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    x0[i] = std::log(preferred[i] / scale);
    envelope_lo[i] = lo_bounds[i] > 0.0
                         ? std::log(lo_bounds[i] / scale)
                         : -kEnvelopeLogGuard;
    envelope_hi[i] = hi_bounds[i] > 0.0
                         ? std::log(hi_bounds[i] / scale)
                         : kEnvelopeLogGuard;
  }

  for (std::size_t i = 1U; i < n; ++i) {
    envelope_lo[i] = std::max(envelope_lo[i], envelope_lo[i - 1U] - ratio_limit);
    envelope_hi[i] = std::min(envelope_hi[i], envelope_hi[i - 1U] + ratio_limit);
  }
  for (std::size_t i = n - 1U; i-- > 0U;) {
    envelope_lo[i] = std::max(envelope_lo[i], envelope_lo[i + 1U] - ratio_limit);
    envelope_hi[i] = std::min(envelope_hi[i], envelope_hi[i + 1U] + ratio_limit);
  }

  for (std::size_t i = 0; i < n; ++i) {
    if (envelope_lo[i] > envelope_hi[i]) {
      failure_message =
          "cell " + std::to_string(i) +
          ": the ratio chain and the cell-measure bounds admit no value "
          "(envelope empty)";
      return ProjectionOutcome::kInfeasible;
    }
  }

  double minimum_sum = 0.0;
  double maximum_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    minimum_sum += std::exp(envelope_lo[i]);
    maximum_sum += std::exp(envelope_hi[i]);
  }
  minimum_sum *= scale;
  maximum_sum *= scale;
  if (total_measure < minimum_sum * (1.0 - kFeasibilityWindowTolerance) ||
      total_measure > maximum_sum * (1.0 + kFeasibilityWindowTolerance)) {
    failure_message =
        "total measure " + format_double(total_measure) +
        " lies outside the feasible window [" + format_double(minimum_sum) +
        ", " + format_double(maximum_sum) +
        "] of the ratio chain and cell-measure bounds";
    return ProjectionOutcome::kInfeasible;
  }

  std::vector<double> x(n, 0.0);
  const auto shifted_sum = [&](const double sigma) {
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double interval_lo = envelope_lo[i];
      double interval_hi = envelope_hi[i];
      if (i > 0U) {
        interval_lo = std::max(interval_lo, x[i - 1U] - ratio_limit);
        interval_hi = std::min(interval_hi, x[i - 1U] + ratio_limit);
      }
      x[i] = std::clamp(x0[i] + sigma, interval_lo, interval_hi);
      sum += std::exp(x[i]);
    }
    return sum * scale;
  };

  double sigma_lo = envelope_lo[0] - x0[0];
  double sigma_hi = envelope_hi[0] - x0[0];
  for (std::size_t i = 1U; i < n; ++i) {
    sigma_lo = std::min(sigma_lo, envelope_lo[i] - x0[i]);
    sigma_hi = std::max(sigma_hi, envelope_hi[i] - x0[i]);
  }
  for (int iteration = 0; iteration < kBisectionIterations; ++iteration) {
    const double sigma_mid = 0.5 * (sigma_lo + sigma_hi);
    if (shifted_sum(sigma_mid) < total_measure) {
      sigma_lo = sigma_mid;
    } else {
      sigma_hi = sigma_mid;
    }
  }
  shifted_sum(0.5 * (sigma_lo + sigma_hi));
  projected.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    projected[i] = std::exp(x[i]) * scale;
  }

  double worst_violation = 0.0;
  std::string worst_detail = "none";
  for (std::size_t i = 0; i + 1U < projected.size(); ++i) {
    const double ratio = std::max(projected[i], projected[i + 1U]) /
                         std::min(projected[i], projected[i + 1U]);
    const double violation = ratio / cfg.ratio_hard_max - 1.0;
    if (!std::isfinite(ratio) || violation > worst_violation) {
      worst_violation = std::isfinite(violation)
                            ? violation
                            : std::numeric_limits<double>::infinity();
      worst_detail = "adjacent ratio at edge " + std::to_string(i + 1U) +
                     " is " + format_double(ratio);
    }
  }
  for (std::size_t i = 0; i < projected.size(); ++i) {
    if (lo_bounds[i] > 0.0) {
      const double violation = lo_bounds[i] / projected[i] - 1.0;
      if (!std::isfinite(projected[i]) || violation > worst_violation) {
        worst_violation = std::isfinite(violation)
                              ? violation
                              : std::numeric_limits<double>::infinity();
        worst_detail = "cell-measure lower-bound value in cell " +
                       std::to_string(i) + " is " + format_double(projected[i]);
      }
    }
  }
  for (std::size_t i = 0; i < projected.size(); ++i) {
    if (hi_bounds[i] > 0.0) {
      const double violation = projected[i] / hi_bounds[i] - 1.0;
      if (!std::isfinite(projected[i]) || violation > worst_violation) {
        worst_violation = std::isfinite(violation)
                              ? violation
                              : std::numeric_limits<double>::infinity();
        worst_detail = "cell-measure upper-bound value in cell " +
                       std::to_string(i) + " is " + format_double(projected[i]);
      }
    }
  }
  const double projected_sum =
      std::accumulate(projected.begin(), projected.end(), 0.0);
  const double sum_violation =
      std::abs(projected_sum - segment.measure_total) / segment.measure_total;
  if (!std::isfinite(sum_violation) || sum_violation > worst_violation) {
    worst_violation = std::isfinite(sum_violation)
                          ? sum_violation
                          : std::numeric_limits<double>::infinity();
    worst_detail = "relative measure-sum residual is " + format_double(sum_violation);
  }

  if (worst_violation > kProjectionCheckTolerance) {
    failure_message = "projection constraint verification failed; worst violation: " +
                      worst_detail;
    return ProjectionOutcome::kNumericalFailure;
  }
  return ProjectionOutcome::kOk;
}

PanelTable build_verification_panel(const PanelTable& source,
                                    IntegrandEvaluator& evaluator) {
  PanelTable panel;
  panel.r_begin = source.r_begin;
  panel.r_end = source.r_end;
  panel.n_bins = 2 * source.n_bins;
  panel.coordinates.resize(static_cast<std::size_t>(panel.n_bins) + 1U);
  std::vector<double> values(static_cast<std::size_t>(panel.n_bins) + 1U);
  std::vector<double> mid_values(static_cast<std::size_t>(panel.n_bins));

  const double h =
      (panel.r_end - panel.r_begin) / static_cast<double>(panel.n_bins);
  for (int k = 0; k <= panel.n_bins; ++k) {
    double coordinate = panel.r_begin + static_cast<double>(k) * h;
    if (k == 0) {
      coordinate = panel.r_begin;
    } else if (k == panel.n_bins) {
      coordinate = panel.r_end;
    }
    panel.coordinates[static_cast<std::size_t>(k)] = coordinate;

    double sample_coordinate = coordinate;
    if (k == 0) {
      sample_coordinate = std::nextafter(panel.r_begin, panel.r_end);
    } else if (k == panel.n_bins) {
      sample_coordinate = std::nextafter(panel.r_end, panel.r_begin);
    }
    values[static_cast<std::size_t>(k)] =
        evaluator.measure_density(sample_coordinate);
  }
  for (int k = 0; k < panel.n_bins; ++k) {
    const double midpoint =
        (panel.coordinates[static_cast<std::size_t>(k)] +
         panel.coordinates[static_cast<std::size_t>(k + 1)]) /
        2.0;
    mid_values[static_cast<std::size_t>(k)] =
        evaluator.measure_density(midpoint);
  }
  panel.measure = integrate_samples(panel.coordinates, values, mid_values);
  return panel;
}

}  // namespace

ZoningResult compute_zoning_intent_nodes(
    const double r_min, const double r_max,
    const ZoningIntentConfig& cfg,
    const std::function<double(double)>& rho0) {
  if (!std::isfinite(r_min) || !std::isfinite(r_max) || !(r_max > r_min)) {
    return failure_result(
        ZoningStatus::kInvalidInput, "MESH_DOMAIN_EMPTY",
        "domain requires finite r_min < r_max; got r_min=" + format_double(r_min) +
            ", r_max=" + format_double(r_max));
  }
  const double domain_length = r_max - r_min;
  const double position_tolerance = kPositionToleranceScale * domain_length;
  if (cfg.n_cells < 1) {
    return failure_result(ZoningStatus::kInvalidInput, "MESH_BUDGET_NONPOSITIVE",
                          "n_cells must be >= 1; got " + std::to_string(cfg.n_cells));
  }
  if (cfg.min_cells_per_segment < 1) {
    return failure_result(
        ZoningStatus::kInvalidInput, "MESH_MIN_CELLS_NONPOSITIVE",
        "min_cells_per_segment must be >= 1; got " +
            std::to_string(cfg.min_cells_per_segment));
  }
  if (!std::isfinite(cfg.dr_min) || cfg.dr_min < 0.0) {
    return failure_result(ZoningStatus::kInvalidInput, "MESH_DR_MIN_NEGATIVE",
                          "dr_min must be finite and >= 0; got " +
                              format_double(cfg.dr_min));
  }
  if (!std::isfinite(cfg.cell_measure_min) || cfg.cell_measure_min < 0.0 ||
      !std::isfinite(cfg.cell_measure_max) || cfg.cell_measure_max < 0.0 ||
      (cfg.cell_measure_min > 0.0 && cfg.cell_measure_max > 0.0 &&
       cfg.cell_measure_min > cfg.cell_measure_max)) {
    return failure_result(
        ZoningStatus::kInvalidInput, "MESH_CELL_MEASURE_BOX_INVALID",
        "cell_measure_min and cell_measure_max must be finite and >= 0, and "
        "cell_measure_min must not exceed cell_measure_max when both are enabled; "
        "got cell_measure_min=" +
            format_double(cfg.cell_measure_min) + ", cell_measure_max=" +
            format_double(cfg.cell_measure_max));
  }
  if (!std::isfinite(cfg.ratio_hard_max) || !(cfg.ratio_hard_max > 1.0) ||
      cfg.ratio_hard_max > 2.0) {
    return failure_result(
        ZoningStatus::kInvalidInput, "MESH_RATIO_CAP_OUT_OF_RANGE",
        "ratio_hard_max must be in (1.0, 2.0]; 2.0 is the immutable solver policy "
        "ceiling; got " +
            format_double(cfg.ratio_hard_max));
  }

  for (std::size_t i = 0; i < cfg.pins.size(); ++i) {
    const double r = cfg.pins[i].r;
    if (!std::isfinite(r) || !(r > r_min + position_tolerance) ||
        !(r < r_max - position_tolerance)) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_PIN_OUT_OF_DOMAIN",
          "pin " + std::to_string(i) + " at r=" + format_double(r) +
              " must lie strictly inside (" +
              format_double(r_min + position_tolerance) + ", " +
              format_double(r_max - position_tolerance) + ")");
    }
  }
  for (std::size_t i = 1; i < cfg.pins.size(); ++i) {
    if (!(cfg.pins[i].r > cfg.pins[i - 1U].r)) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_PIN_NOT_SORTED",
          "pins must be strictly increasing; pin " + std::to_string(i - 1U) +
              " is " + format_double(cfg.pins[i - 1U].r) + ", pin " +
              std::to_string(i) + " is " + format_double(cfg.pins[i].r));
    }
  }
  for (std::size_t i = 1; i < cfg.pins.size(); ++i) {
    const double separation = cfg.pins[i].r - cfg.pins[i - 1U].r;
    if (separation < position_tolerance) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_PIN_DUPLICATE",
          "adjacent pins " + std::to_string(i - 1U) + " and " +
              std::to_string(i) + " are separated by " + format_double(separation) +
              ", below tol_r=" + format_double(position_tolerance));
    }
  }

  for (std::size_t i = 0; i < cfg.profile.size(); ++i) {
    const double r = cfg.profile[i].r;
    if (!std::isfinite(r) || r < r_min || r > r_max) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_PROFILE_OUT_OF_DOMAIN",
          "profile point " + std::to_string(i) + " at r=" + format_double(r) +
              " must lie in [" + format_double(r_min) + ", " +
              format_double(r_max) + "]");
    }
  }
  for (std::size_t i = 1; i < cfg.profile.size(); ++i) {
    if (!(cfg.profile[i].r > cfg.profile[i - 1U].r)) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_PROFILE_COORD_NOT_SORTED",
          "profile coordinates must be strictly increasing; point " +
              std::to_string(i - 1U) + " is " +
              format_double(cfg.profile[i - 1U].r) + ", point " +
              std::to_string(i) + " is " + format_double(cfg.profile[i].r));
    }
  }
  for (std::size_t i = 1; i < cfg.profile.size(); ++i) {
    const double separation = cfg.profile[i].r - cfg.profile[i - 1U].r;
    if (separation <= position_tolerance) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_PROFILE_COORD_DUPLICATE",
          "profile coordinates " + std::to_string(i - 1U) + " and " +
              std::to_string(i) + " are separated by " + format_double(separation) +
              ", within tol_r=" + format_double(position_tolerance));
    }
  }
  for (std::size_t i = 0; i < cfg.profile.size(); ++i) {
    if (!std::isfinite(cfg.profile[i].w) || !(cfg.profile[i].w > 0.0)) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_PROFILE_VALUE_NONPOSITIVE",
          "profile point " + std::to_string(i) + " has w=" +
              format_double(cfg.profile[i].w) + "; w must be finite and > 0");
    }
  }
  for (std::size_t i = 0; i < cfg.anchors.size(); ++i) {
    const ZoningAnchor& anchor = cfg.anchors[i];
    if (!std::isfinite(anchor.r) || anchor.r < r_min || anchor.r > r_max) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_ANCHOR_OUT_OF_DOMAIN",
          "anchor " + std::to_string(i) + " at r=" + format_double(anchor.r) +
              " must lie in [" + format_double(r_min) + ", " +
              format_double(r_max) + "]");
    }
    if (!std::isfinite(anchor.half_width) || !(anchor.half_width > 0.0)) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_ANCHOR_WIDTH_NONPOSITIVE",
          "anchor " + std::to_string(i) + " has half_width=" +
              format_double(anchor.half_width) +
              "; half_width must be finite and > 0");
    }
    if (!std::isfinite(anchor.log_amplitude) ||
        std::abs(anchor.log_amplitude) > kMaxAnchorLogAmplitude) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_ANCHOR_AMPLITUDE_INVALID",
          "anchor " + std::to_string(i) + " has log_amplitude=" +
              format_double(anchor.log_amplitude) + "; |log_amplitude| must be <= " +
              format_double(kMaxAnchorLogAmplitude));
    }
  }
  if (!cfg.anchors.empty()) {
    std::vector<double> anchor_check_points;
    anchor_check_points.reserve(3U * cfg.anchors.size());
    for (const ZoningAnchor& anchor : cfg.anchors) {
      anchor_check_points.push_back(
          std::clamp(anchor.r - anchor.half_width, r_min, r_max));
      anchor_check_points.push_back(anchor.r);
      anchor_check_points.push_back(
          std::clamp(anchor.r + anchor.half_width, r_min, r_max));
    }
    double worst_coordinate = anchor_check_points.front();
    double worst_absolute_sum = 0.0;
    for (const double check_point : anchor_check_points) {
      double sum = 0.0;
      for (const ZoningAnchor& anchor : cfg.anchors) {
        const double u = (check_point - anchor.r) / anchor.half_width;
        if (std::abs(u) < 1.0) {
          sum += anchor.log_amplitude * 0.5 * (1.0 + std::cos(kPi * u));
        }
      }
      const double absolute_sum = std::abs(sum);
      if (absolute_sum > worst_absolute_sum) {
        worst_absolute_sum = absolute_sum;
        worst_coordinate = check_point;
      }
    }
    if (worst_absolute_sum > kMaxAnchorLogAggregate) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_ANCHOR_AGGREGATE_EXCESSIVE",
          "anchor aggregate at r=" + format_double(worst_coordinate) +
              " has |S|=" + format_double(worst_absolute_sum) +
              ", above the limit " + format_double(kMaxAnchorLogAggregate));
    }
  }
  for (std::size_t i = 0; i < cfg.bands.size(); ++i) {
    const ZoningBand& band = cfg.bands[i];
    if (!std::isfinite(band.measure_frac_begin) ||
        !std::isfinite(band.measure_frac_end) ||
        !std::isfinite(band.cell_measure_min) ||
        !std::isfinite(band.cell_measure_max) ||
        band.measure_frac_begin < 0.0 ||
        !(band.measure_frac_begin < band.measure_frac_end) ||
        band.measure_frac_end > 1.0) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_BAND_RANGE_INVALID",
          "band " + std::to_string(i) + " has measure_frac_begin=" +
              format_double(band.measure_frac_begin) +
              ", measure_frac_end=" + format_double(band.measure_frac_end));
    }
    if (band.cell_measure_min < 0.0 || band.cell_measure_max < 0.0 ||
        (band.cell_measure_min > 0.0 && band.cell_measure_max > 0.0 &&
         band.cell_measure_min > band.cell_measure_max)) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_BAND_BOUNDS_INVALID",
          "band " + std::to_string(i) + " has cell_measure_min=" +
              format_double(band.cell_measure_min) +
              ", cell_measure_max=" + format_double(band.cell_measure_max));
    }
  }
  if (cfg.measure != ZoningMeasure::kWidth && !rho0) {
    return failure_result(
        ZoningStatus::kInvalidInput, "MESH_MEASURE_NEEDS_DENSITY",
        "the selected mass measure requires a rho0(r) function");
  }

  double lo_measure = cfg.cell_measure_min;
  const double hi_measure = cfg.cell_measure_max;
  if (cfg.measure == ZoningMeasure::kWidth && cfg.dr_min > 0.0) {
    lo_measure = std::max(lo_measure, cfg.dr_min);
  }

  std::vector<std::string> warnings;
  std::vector<double> segment_edges;
  segment_edges.reserve(cfg.pins.size() + 2U);
  segment_edges.push_back(r_min);
  for (const ZoningIntentPin& pin : cfg.pins) {
    segment_edges.push_back(pin.r);
  }
  segment_edges.push_back(r_max);

  std::vector<double> usable_events;
  usable_events.reserve(cfg.extra_events.size());
  for (const double event : cfg.extra_events) {
    if (!std::isfinite(event) || event < r_min || event > r_max) {
      warnings.push_back("extra_event outside domain ignored: " + format_double(event));
    } else if (event > r_min && event < r_max) {
      usable_events.push_back(event);
    }
  }

  IntegrandEvaluator evaluator;
  evaluator.measure = cfg.measure;
  evaluator.profile = &cfg.profile;
  evaluator.anchors = &cfg.anchors;
  evaluator.rho0 = &rho0;

  std::vector<SegmentData> segments;
  segments.reserve(segment_edges.size() - 1U);
  for (std::size_t segment_index = 0;
       segment_index + 1U < segment_edges.size(); ++segment_index) {
    SegmentData segment;
    segment.r_begin = segment_edges[segment_index];
    segment.r_end = segment_edges[segment_index + 1U];

    std::vector<double> panel_edges = {segment.r_begin, segment.r_end};
    for (const ZoningProfilePoint& point : cfg.profile) {
      if (point.r > segment.r_begin && point.r < segment.r_end) {
        panel_edges.push_back(point.r);
      }
    }
    for (const ZoningAnchor& anchor : cfg.anchors) {
      const double support_begin = anchor.r - anchor.half_width;
      if (support_begin > segment.r_begin && support_begin < segment.r_end) {
        panel_edges.push_back(support_begin);
      }
      if (anchor.r > segment.r_begin && anchor.r < segment.r_end) {
        panel_edges.push_back(anchor.r);
      }
      const double support_end = anchor.r + anchor.half_width;
      if (support_end > segment.r_begin && support_end < segment.r_end) {
        panel_edges.push_back(support_end);
      }
    }
    for (const double event : usable_events) {
      if (event > segment.r_begin && event < segment.r_end) {
        panel_edges.push_back(event);
      }
    }
    std::sort(panel_edges.begin(), panel_edges.end());
    std::vector<double> deduplicated_edges;
    deduplicated_edges.reserve(panel_edges.size());
    for (const double edge : panel_edges) {
      if (deduplicated_edges.empty() ||
          edge - deduplicated_edges.back() > position_tolerance) {
        deduplicated_edges.push_back(edge);
      }
    }
    if (!bitwise_equal(deduplicated_edges.back(), segment.r_end)) {
      deduplicated_edges.back() = segment.r_end;
    }

    for (std::size_t panel_index = 0;
         panel_index + 1U < deduplicated_edges.size(); ++panel_index) {
      PanelTable panel;
      const double panel_begin = deduplicated_edges[panel_index];
      const double panel_end = deduplicated_edges[panel_index + 1U];
      if (!build_refined_panel(panel_begin, panel_end, evaluator, panel)) {
        if (evaluator.density_invalid) {
          return density_failure(evaluator, warnings);
        }
        return failure_result(
            ZoningStatus::kNumericalFailure, "MESH_QUADRATURE_NOT_CONVERGED",
            "quadrature did not converge on panel [" + format_double(panel_begin) +
                ", " + format_double(panel_end) +
                "]; undeclared discontinuities must be declared via pins/extra_events",
            warnings);
      }
      segment.measure_total += panel.measure.total;
      segment.monitor_total += panel.monitor.total;
      segment.panels.push_back(std::move(panel));
    }

    if (cfg.measure != ZoningMeasure::kWidth && !(segment.monitor_total > 0.0)) {
      return failure_result(
          ZoningStatus::kInvalidInput, "MESH_DENSITY_INVALID",
          "segment [" + format_double(segment.r_begin) + ", " +
              format_double(segment.r_end) +
              "] has zero measure; zero-density (void) regions are not representable in "
              "this measure — use kWidth or declare the void boundary as a pin",
          warnings);
    }
    segments.push_back(std::move(segment));
  }

  long long minimum_sum = 0;
  long long maximum_sum = 0;
  long long minimum_sum_without_box = 0;
  long long maximum_sum_without_box = 0;
  double width_capacity_sum = 0.0;
  for (std::size_t i = 0; i < segments.size(); ++i) {
    SegmentData& segment = segments[i];
    const double segment_length = segment.r_end - segment.r_begin;
    segment.n_min = cfg.min_cells_per_segment;
    if (cfg.dr_min > 0.0) {
      const double raw_capacity =
          std::floor(segment_length / cfg.dr_min * (1.0 + kCapacityRoundoff));
      segment.n_max = raw_capacity >= static_cast<double>(std::numeric_limits<int>::max())
                          ? std::numeric_limits<int>::max()
                          : static_cast<int>(raw_capacity);
      width_capacity_sum += segment_length / cfg.dr_min;
    } else {
      segment.n_max = std::numeric_limits<int>::max();
    }

    minimum_sum_without_box += segment.n_min;
    maximum_sum_without_box = std::min<long long>(
        std::numeric_limits<int>::max(),
        maximum_sum_without_box + segment.n_max);

    bool n_min_box_binding = false;
    bool n_max_box_binding = false;
    if (hi_measure > 0.0) {
      const double raw_minimum =
          std::ceil(segment.measure_total / hi_measure *
                    (1.0 - kCapacityRoundoff));
      const int n_min_box =
          raw_minimum >= static_cast<double>(std::numeric_limits<int>::max())
              ? std::numeric_limits<int>::max()
              : std::max(1, static_cast<int>(raw_minimum));
      n_min_box_binding = n_min_box > segment.n_min;
      segment.n_min = std::max(segment.n_min, n_min_box);
    }
    if (lo_measure > 0.0) {
      const double raw_maximum =
          std::floor(segment.measure_total / lo_measure *
                     (1.0 + kCapacityRoundoff));
      const int n_max_box =
          raw_maximum >= static_cast<double>(std::numeric_limits<int>::max())
              ? std::numeric_limits<int>::max()
              : static_cast<int>(raw_maximum);
      const bool explicit_lower_box_is_stricter =
          cfg.cell_measure_min > 0.0 &&
          (cfg.measure != ZoningMeasure::kWidth ||
           cfg.cell_measure_min > cfg.dr_min);
      n_max_box_binding =
          explicit_lower_box_is_stricter && n_max_box < segment.n_max;
      segment.n_max = std::min(segment.n_max, n_max_box);
    }

    if (segment.n_min > segment.n_max) {
      if (n_min_box_binding || n_max_box_binding) {
        return failure_result(
            ZoningStatus::kInfeasible, "MESH_CELL_MEASURE_BOX_INFEASIBLE",
            "segment " + std::to_string(i) + " has Q_s=" +
                format_double(segment.measure_total) + ", lo_measure=" +
                format_double(lo_measure) + ", hi_measure=" +
                format_double(hi_measure) + ", N_min_s=" +
                std::to_string(segment.n_min) + ", N_max_s=" +
                std::to_string(segment.n_max),
            warnings);
      }
      return failure_result(
          ZoningStatus::kInfeasible, "MESH_SEGMENT_BUDGET_CONFLICT",
          "segment " + std::to_string(i) + " has L_s=" +
              format_double(segment_length) + ", dr_min=" +
              format_double(cfg.dr_min) + ", N_min_s=" +
              std::to_string(segment.n_min) + ", N_max_s=" +
              std::to_string(segment.n_max),
          warnings);
    }
    minimum_sum += segment.n_min;
    maximum_sum = std::min<long long>(
        std::numeric_limits<int>::max(), maximum_sum + segment.n_max);
  }

  if (minimum_sum > cfg.n_cells) {
    if (minimum_sum_without_box <= cfg.n_cells) {
      return failure_result(
          ZoningStatus::kInfeasible, "MESH_CELL_MEASURE_BOX_INFEASIBLE",
          "sum of box-constrained N_min_s=" + std::to_string(minimum_sum) +
              " exceeds n_cells=" + std::to_string(cfg.n_cells) +
              "; lo_measure=" + format_double(lo_measure) +
              ", hi_measure=" + format_double(hi_measure),
          warnings);
    }
    return failure_result(
        ZoningStatus::kInfeasible, "MESH_SEGMENT_MIN_COUNT_INFEASIBLE",
        "sum(N_min_s)=" + std::to_string(minimum_sum) +
            " exceeds n_cells=" + std::to_string(cfg.n_cells),
        warnings);
  }
  if (maximum_sum < cfg.n_cells) {
    if (maximum_sum_without_box >= cfg.n_cells) {
      return failure_result(
          ZoningStatus::kInfeasible, "MESH_CELL_MEASURE_BOX_INFEASIBLE",
          "sum of box-constrained N_max_s=" + std::to_string(maximum_sum) +
              " is below n_cells=" + std::to_string(cfg.n_cells) +
              "; lo_measure=" + format_double(lo_measure) +
              ", hi_measure=" + format_double(hi_measure),
          warnings);
    }
    return failure_result(
        ZoningStatus::kInfeasible, "MESH_DR_MIN_COUNT_INFEASIBLE",
        "sum(L_s/dr_min) capacity=" + format_double(width_capacity_sum) +
            " is below n_cells=" + std::to_string(cfg.n_cells),
        warnings);
  }

  double total_measure = 0.0;
  double total_monitor = 0.0;
  for (const SegmentData& segment : segments) {
    total_measure += segment.measure_total;
    total_monitor += segment.monitor_total;
  }
  std::vector<double> ideal_shares(segments.size(), 0.0);
  int allocated_cells = 0;
  for (std::size_t i = 0; i < segments.size(); ++i) {
    ideal_shares[i] = static_cast<double>(cfg.n_cells) *
                      (segments[i].monitor_total / total_monitor);
    segments[i].n_cells = segments[i].n_min;
    allocated_cells += segments[i].n_cells;
  }
  while (allocated_cells < cfg.n_cells) {
    std::size_t selected = 0;
    double largest_deficit = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < segments.size(); ++i) {
      if (segments[i].n_cells >= segments[i].n_max) {
        continue;
      }
      const double deficit = ideal_shares[i] - segments[i].n_cells;
      if (deficit > largest_deficit) {
        largest_deficit = deficit;
        selected = i;
      }
    }
    ++segments[selected].n_cells;
    ++allocated_cells;
  }

  std::vector<double> nodes;
  nodes.reserve(static_cast<std::size_t>(cfg.n_cells) + 1U);
  double segment_measure_offset = 0.0;
  for (std::size_t segment_index = 0;
       segment_index < segments.size(); ++segment_index) {
    const SegmentData& segment = segments[segment_index];
    const int n_cells = segment.n_cells;
    std::vector<double> preferred_nodes(static_cast<std::size_t>(n_cells) + 1U);
    preferred_nodes.front() = segment.r_begin;
    preferred_nodes.back() = segment.r_end;
    for (int j = 1; j < n_cells; ++j) {
      const double target =
          (static_cast<double>(j) / static_cast<double>(n_cells)) *
          segment.monitor_total;
      preferred_nodes[static_cast<std::size_t>(j)] =
          invert_cumulative(segment, target, IntegrandKind::kMonitor);
    }

    std::vector<double> preferred_measures(static_cast<std::size_t>(n_cells));
    for (int i = 0; i < n_cells; ++i) {
      preferred_measures[static_cast<std::size_t>(i)] =
          evaluate_cumulative(segment, preferred_nodes[static_cast<std::size_t>(i + 1)],
                              IntegrandKind::kMeasure) -
          evaluate_cumulative(segment, preferred_nodes[static_cast<std::size_t>(i)],
                              IntegrandKind::kMeasure);
    }

    std::vector<double> projected_measures;
    std::string projection_failure;
    const auto band_coverage = [&](const std::vector<double>& candidate) {
      std::vector<std::vector<bool>> coverage(
          cfg.bands.size(),
          std::vector<bool>(static_cast<std::size_t>(n_cells), false));
      double prefix = segment_measure_offset;
      for (int i = 0; i < n_cells; ++i) {
        const double span_begin = prefix / total_measure;
        prefix += candidate[static_cast<std::size_t>(i)];
        const double span_end = prefix / total_measure;
        for (std::size_t k = 0; k < cfg.bands.size(); ++k) {
          const ZoningBand& band = cfg.bands[k];
          coverage[k][static_cast<std::size_t>(i)] =
              span_begin < band.measure_frac_end &&
              span_end > band.measure_frac_begin;
        }
      }
      return coverage;
    };

    std::vector<std::vector<bool>> coverage_acc =
        band_coverage(preferred_measures);
    const std::size_t maximum_reconciliation_rounds =
        static_cast<std::size_t>(n_cells) * cfg.bands.size() + 1U;
    std::size_t reconciliation_rounds = 0;
    bool coverage_reconciled = false;
    // Coverage only ever grows: a cell once inside a band stays bounded even if
    // its final span leaves the band. Conservative, and guarantees termination.
    // Bands bound cell measures only; they do not steer the integer allocation.
    for (; reconciliation_rounds < maximum_reconciliation_rounds;) {
      ++reconciliation_rounds;
      std::vector<double> lo_bounds(static_cast<std::size_t>(n_cells), lo_measure);
      std::vector<double> hi_bounds(static_cast<std::size_t>(n_cells), hi_measure);
      for (std::size_t k = 0; k < cfg.bands.size(); ++k) {
        const ZoningBand& band = cfg.bands[k];
        for (int i = 0; i < n_cells; ++i) {
          const std::size_t cell = static_cast<std::size_t>(i);
          if (!coverage_acc[k][cell]) {
            continue;
          }
          if (band.cell_measure_min > 0.0) {
            lo_bounds[cell] = std::max(lo_bounds[cell], band.cell_measure_min);
          }
          if (band.cell_measure_max > 0.0) {
            hi_bounds[cell] = hi_bounds[cell] > 0.0
                                  ? std::min(hi_bounds[cell], band.cell_measure_max)
                                  : band.cell_measure_max;
          }
        }
      }

      if (!cfg.bands.empty()) {
        double lower_sum = 0.0;
        double upper_sum = 0.0;
        bool every_upper_bound_enabled = true;
        for (int i = 0; i < n_cells; ++i) {
          const std::size_t cell = static_cast<std::size_t>(i);
          if (lo_bounds[cell] > 0.0 && hi_bounds[cell] > 0.0 &&
              lo_bounds[cell] > hi_bounds[cell]) {
            return failure_result(
                ZoningStatus::kInfeasible, "MESH_BAND_BOX_INFEASIBLE",
                "segment " + std::to_string(segment_index) + ", cell " +
                    std::to_string(i) + " has lo_i=" +
                    format_double(lo_bounds[cell]) + ", hi_i=" +
                    format_double(hi_bounds[cell]),
                warnings);
          }
          if (lo_bounds[cell] > 0.0) {
            lower_sum += lo_bounds[cell];
          }
          if (hi_bounds[cell] > 0.0) {
            upper_sum += hi_bounds[cell];
          } else {
            every_upper_bound_enabled = false;
          }
        }
        if (lower_sum > segment.measure_total) {
          return failure_result(
              ZoningStatus::kInfeasible, "MESH_BAND_BOX_INFEASIBLE",
              "segment " + std::to_string(segment_index) +
                  " has sum(enabled lo_i)=" + format_double(lower_sum) +
                  ", above measure_total=" +
                  format_double(segment.measure_total),
              warnings);
        }
        if (every_upper_bound_enabled && upper_sum < segment.measure_total) {
          return failure_result(
              ZoningStatus::kInfeasible, "MESH_BAND_BOX_INFEASIBLE",
              "segment " + std::to_string(segment_index) +
                  " has sum(enabled hi_i)=" + format_double(upper_sum) +
                  ", below measure_total=" +
                  format_double(segment.measure_total),
              warnings);
        }
      }

      const ProjectionOutcome projection_outcome = project_cell_measures(
          segment, cfg, lo_bounds, hi_bounds, preferred_measures,
          projected_measures, projection_failure);
      switch (projection_outcome) {
        case ProjectionOutcome::kOk:
          break;
        case ProjectionOutcome::kInfeasible:
          return failure_result(
              ZoningStatus::kInfeasible, "MESH_CHAIN_SUM_INFEASIBLE",
              "segment " + std::to_string(segment_index) + ": " +
                  projection_failure,
              warnings);
        case ProjectionOutcome::kNumericalFailure:
          return failure_result(
              ZoningStatus::kNumericalFailure, "MESH_PROJECTION_STAGNATED",
              "segment " + std::to_string(segment_index) + ": " +
                  projection_failure,
              warnings);
      }

      const std::vector<std::vector<bool>> projected_coverage =
          band_coverage(projected_measures);
      bool coverage_grew = false;
      for (std::size_t k = 0; k < cfg.bands.size(); ++k) {
        for (int i = 0; i < n_cells; ++i) {
          const std::size_t cell = static_cast<std::size_t>(i);
          if (projected_coverage[k][cell] && !coverage_acc[k][cell]) {
            coverage_acc[k][cell] = true;
            coverage_grew = true;
          }
        }
      }
      if (!coverage_grew) {
        coverage_reconciled = true;
        break;
      }
    }
    if (!coverage_reconciled) {
      return failure_result(
          ZoningStatus::kNumericalFailure, "MESH_PROJECTION_STAGNATED",
          "band coverage reconciliation exceeded its bound",
          warnings);
    }
    if (reconciliation_rounds > 1U) {
      warnings.push_back(
          "band coverage grew during projection; " +
          std::to_string(reconciliation_rounds) + " reconciliation rounds");
    }

    const double projected_sum =
        std::accumulate(projected_measures.begin(), projected_measures.end(), 0.0);
    std::vector<double> segment_nodes(static_cast<std::size_t>(n_cells) + 1U);
    segment_nodes.front() = segment.r_begin;
    segment_nodes.back() = segment.r_end;
    double prefix = 0.0;
    for (int j = 1; j < n_cells; ++j) {
      prefix += projected_measures[static_cast<std::size_t>(j - 1)];
      if (cfg.measure == ZoningMeasure::kWidth) {
        segment_nodes[static_cast<std::size_t>(j)] =
            segment.r_begin + prefix *
                                  ((segment.r_end - segment.r_begin) / projected_sum);
      } else {
        const double target = prefix * (segment.measure_total / projected_sum);
        segment_nodes[static_cast<std::size_t>(j)] =
            invert_cumulative(segment, target, IntegrandKind::kMeasure);
      }
    }

    if (segment_index == 0U) {
      nodes.insert(nodes.end(), segment_nodes.begin(), segment_nodes.end());
    } else {
      nodes.insert(nodes.end(), segment_nodes.begin() + 1, segment_nodes.end());
    }
    segment_measure_offset += segment.measure_total;
  }

  std::vector<SegmentData> verification_segments;
  verification_segments.reserve(segments.size());
  for (const SegmentData& segment : segments) {
    SegmentData verification;
    verification.r_begin = segment.r_begin;
    verification.r_end = segment.r_end;
    verification.n_cells = segment.n_cells;
    for (const PanelTable& panel : segment.panels) {
      PanelTable verification_panel = build_verification_panel(panel, evaluator);
      if (evaluator.density_invalid) {
        return density_failure(evaluator, warnings);
      }
      verification.measure_total += verification_panel.measure.total;
      verification.panels.push_back(std::move(verification_panel));
    }
    verification_segments.push_back(std::move(verification));
  }

  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (!std::isfinite(nodes[i])) {
      return failure_result(
          ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_NONFINITE",
          "node " + std::to_string(i) + " is non-finite: " +
              format_double(nodes[i]),
          warnings);
    }
  }
  for (std::size_t i = 0; i + 1U < nodes.size(); ++i) {
    if (!(nodes[i + 1U] > nodes[i])) {
      return failure_result(
          ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_MONOTONE",
          "nodes " + std::to_string(i) + " and " + std::to_string(i + 1U) +
              " are not strictly increasing: " + format_double(nodes[i]) + ", " +
              format_double(nodes[i + 1U]),
          warnings);
    }
  }
  if (!bitwise_equal(nodes.front(), r_min) || !bitwise_equal(nodes.back(), r_max)) {
    return failure_result(
        ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_PIN_DRIFT",
        "domain endpoint drifted from its exact input value", warnings);
  }
  for (const ZoningIntentPin& pin : cfg.pins) {
    const bool found = std::any_of(nodes.begin(), nodes.end(), [&](const double node) {
      return bitwise_equal(node, pin.r);
    });
    if (!found) {
      return failure_result(
          ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_PIN_DRIFT",
          "pin at r=" + format_double(pin.r) + " is not bitwise present in the nodes",
          warnings);
    }
  }

  double minimum_width = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i + 1U < nodes.size(); ++i) {
    const double width = nodes[i + 1U] - nodes[i];
    minimum_width = std::min(minimum_width, width);
  }
  if (cfg.dr_min > 0.0 &&
      minimum_width < cfg.dr_min * (1.0 - kWidthCheckTolerance)) {
    std::string message =
        "minimum width " + format_double(minimum_width) + " is below dr_min=" +
        format_double(cfg.dr_min);
    if (cfg.measure != ZoningMeasure::kWidth) {
      message +=
          "; joint mass-measure + width-floor projection is a planned upgrade; use a "
          "larger dr_min-compatible budget or profile change";
    }
    return failure_result(ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_WIDTH",
                          std::move(message), warnings);
  }

  std::vector<double> verification_measures;
  verification_measures.reserve(static_cast<std::size_t>(cfg.n_cells));
  double verification_total = 0.0;
  for (const SegmentData& segment : verification_segments) {
    verification_total += segment.measure_total;
  }
  std::vector<double> verification_fraction_nodes(
      static_cast<std::size_t>(cfg.n_cells) + 1U, 0.0);
  std::size_t node_offset = 0;
  double verification_measure_offset = 0.0;
  for (const SegmentData& segment : verification_segments) {
    for (int i = 0; i < segment.n_cells; ++i) {
      const std::size_t cell = node_offset + static_cast<std::size_t>(i);
      const double local_begin = evaluate_cumulative(
          segment, nodes[cell], IntegrandKind::kMeasure);
      const double local_end = evaluate_cumulative(
          segment, nodes[cell + 1U], IntegrandKind::kMeasure);
      verification_measures.push_back(local_end - local_begin);
      verification_fraction_nodes[cell] =
          (verification_measure_offset + local_begin) / verification_total;
      verification_fraction_nodes[cell + 1U] =
          (verification_measure_offset + local_end) / verification_total;
    }
    node_offset += static_cast<std::size_t>(segment.n_cells);
    verification_measure_offset += segment.measure_total;
  }

  std::vector<std::size_t> cross_pin_edges;
  cross_pin_edges.reserve(cfg.pins.size());
  std::size_t cumulative_cells = 0;
  for (std::size_t i = 0; i + 1U < segments.size(); ++i) {
    cumulative_cells += static_cast<std::size_t>(segments[i].n_cells);
    cross_pin_edges.push_back(cumulative_cells);
  }

  double ratio_max = 1.0;
  double ratio_sum = 0.0;
  int soft_exceed_count = 0;
  for (std::size_t i = 0; i + 1U < verification_measures.size(); ++i) {
    double ratio = std::numeric_limits<double>::infinity();
    if (verification_measures[i] > 0.0 && verification_measures[i + 1U] > 0.0 &&
        std::isfinite(verification_measures[i]) &&
        std::isfinite(verification_measures[i + 1U])) {
      ratio = std::max(verification_measures[i], verification_measures[i + 1U]) /
              std::min(verification_measures[i], verification_measures[i + 1U]);
    }
    const auto pin_edge =
        std::find(cross_pin_edges.begin(), cross_pin_edges.end(), i + 1U);
    const bool across_pin = pin_edge != cross_pin_edges.end();
    if (ratio > cfg.ratio_hard_max * (1.0 + kRatioCheckTolerance)) {
      if (across_pin) {
        const std::size_t pin_index =
            static_cast<std::size_t>(pin_edge - cross_pin_edges.begin());
        if (!cfg.pins[pin_index].ratio_jump_allowed) {
          return failure_result(
              ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_RATIO_CROSS_PIN",
              "pin at r=" + format_double(cfg.pins[pin_index].r) +
                  " has achieved adjacent-cell-measure ratio " + format_double(ratio) +
                  "; set ratio_jump_allowed on the pin, or rebalance the intent so the "
                  "per-side cell measures meet at the pin",
              warnings);
        }
      } else {
        return failure_result(
            ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_RATIO",
            "interior edge " + std::to_string(i + 1U) +
                " has achieved adjacent-cell-measure ratio " + format_double(ratio) +
                ", above ratio_hard_max=" + format_double(cfg.ratio_hard_max),
            warnings);
      }
    }
    ratio_max = std::max(ratio_max, ratio);
    ratio_sum += ratio;
    if (ratio > cfg.preferred_ratio) {
      ++soft_exceed_count;
    }
  }

  for (std::size_t i = 0; i < verification_measures.size(); ++i) {
    const double measure = verification_measures[i];
    if (lo_measure > 0.0 &&
        (!std::isfinite(measure) ||
         measure < lo_measure * (1.0 - kRatioCheckTolerance))) {
      return failure_result(
          ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_CELL_MEASURE_BOX",
          "cell " + std::to_string(i) + " has measure " +
              format_double(measure) + ", below lo_measure=" +
              format_double(lo_measure),
          warnings);
    }
    if (hi_measure > 0.0 &&
        (!std::isfinite(measure) ||
         measure > hi_measure * (1.0 + kRatioCheckTolerance))) {
      return failure_result(
          ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_CELL_MEASURE_BOX",
          "cell " + std::to_string(i) + " has measure " +
              format_double(measure) + ", above hi_measure=" +
              format_double(hi_measure),
          warnings);
    }
  }

  std::vector<double> band_cell_measure_max_achieved(cfg.bands.size(), 0.0);
  std::vector<double> band_cell_measure_min_achieved(
      cfg.bands.size(), std::numeric_limits<double>::infinity());
  std::vector<bool> band_covers_cell(cfg.bands.size(), false);
  for (std::size_t i = 0; i < verification_measures.size(); ++i) {
    const double span_begin = verification_fraction_nodes[i];
    const double span_end = verification_fraction_nodes[i + 1U];
    const double measure = verification_measures[i];
    for (std::size_t k = 0; k < cfg.bands.size(); ++k) {
      const ZoningBand& band = cfg.bands[k];
      const double overlap =
          std::min(span_end, band.measure_frac_end) -
          std::max(span_begin, band.measure_frac_begin);
      if (!(overlap > kBandOverlapTolerance)) {
        continue;
      }
      if (band.cell_measure_min > 0.0 &&
          (!std::isfinite(measure) ||
           measure < band.cell_measure_min * (1.0 - kRatioCheckTolerance))) {
        return failure_result(
            ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_BAND",
            "band " + std::to_string(k) + ", cell " + std::to_string(i) +
                " has measure " + format_double(measure) +
                ", below cell_measure_min=" +
                format_double(band.cell_measure_min) +
                "; cells that moved into the band during projection are bounded only "
                "at verification time — widen the band, relax the bound, or change "
                "the budget/profile so the preferred and final cell spans agree",
            warnings);
      }
      if (band.cell_measure_max > 0.0 &&
          (!std::isfinite(measure) ||
           measure > band.cell_measure_max * (1.0 + kRatioCheckTolerance))) {
        return failure_result(
            ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_BAND",
            "band " + std::to_string(k) + ", cell " + std::to_string(i) +
                " has measure " + format_double(measure) +
                ", above cell_measure_max=" +
                format_double(band.cell_measure_max) +
                "; cells that moved into the band during projection are bounded only "
                "at verification time — widen the band, relax the bound, or change "
                "the budget/profile so the preferred and final cell spans agree",
            warnings);
      }
      band_covers_cell[k] = true;
      band_cell_measure_max_achieved[k] =
          std::max(band_cell_measure_max_achieved[k], measure);
      band_cell_measure_min_achieved[k] =
          std::min(band_cell_measure_min_achieved[k], measure);
    }
  }
  for (std::size_t k = 0; k < cfg.bands.size(); ++k) {
    if (!band_covers_cell[k]) {
      band_cell_measure_min_achieved[k] = 0.0;
    }
  }

  const double verification_sum = std::accumulate(
      verification_measures.begin(), verification_measures.end(), 0.0);
  const double closure_residual =
      std::abs(verification_sum - verification_total);
  const double closure_relative =
      closure_residual / std::max(std::abs(verification_total), kMuFloor);
  if (closure_residual > kClosureCheckTolerance * verification_total) {
    return failure_result(
        ZoningStatus::kNumericalFailure, "MESH_POSTCHECK_MEASURE_CLOSURE",
        "verification measure closure residual " + format_double(closure_residual) +
            " exceeds 1e-8 of total measure " + format_double(verification_total),
        warnings);
  }

  ZoningResult result;
  result.ok = true;
  result.nodes = std::move(nodes);
  result.diag.status = ZoningStatus::kOk;
  result.diag.ratio_max_achieved = ratio_max;
  result.diag.ratio_mean_achieved =
      verification_measures.size() > 1U
          ? ratio_sum / static_cast<double>(verification_measures.size() - 1U)
          : 1.0;
  result.diag.n_ratio_soft_exceed = soft_exceed_count;
  result.diag.width_min_achieved = minimum_width;
  const auto measure_extrema = std::minmax_element(
      verification_measures.begin(), verification_measures.end());
  result.diag.cell_measure_min_achieved = *measure_extrema.first;
  result.diag.cell_measure_max_achieved = *measure_extrema.second;
  result.diag.band_cell_measure_max_achieved =
      std::move(band_cell_measure_max_achieved);
  result.diag.band_cell_measure_min_achieved =
      std::move(band_cell_measure_min_achieved);
  result.diag.quadrature_rel_residual = closure_relative;
  result.diag.warnings = std::move(warnings);
  result.diag.cells_per_segment.reserve(segments.size());
  for (const SegmentData& segment : segments) {
    result.diag.cells_per_segment.push_back(segment.n_cells);
  }
  return result;
}

}  // namespace tenryu::core
