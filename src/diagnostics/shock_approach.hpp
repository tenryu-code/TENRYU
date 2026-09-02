#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/error.hpp"

namespace tenryu::diagnostics {

namespace shock_approach_detail {

inline double determinant_3x3(const double a00,
                              const double a01,
                              const double a02,
                              const double a10,
                              const double a11,
                              const double a12,
                              const double a20,
                              const double a21,
                              const double a22) {
  return a00 * (a11 * a22 - a12 * a21) -
         a01 * (a10 * a22 - a12 * a20) +
         a02 * (a10 * a21 - a11 * a20);
}

}  // namespace shock_approach_detail

// Radius of the inward-moving shock ridge from a binned radial profile.
// Inputs: bin-center radii s[i] (strictly increasing, size n >= 8) and the
// theta-averaged cell pressure p[i] (finite, > 0). The ridge is located at the
// maximum of |dp/ds| computed with central differences on the (possibly
// nonuniform) bins; the returned radius refines the argmax by a 3-point
// parabolic fit of |dp/ds| over (i-1, i, i+1) in the s coordinate (clamped to
// [s[i-1], s[i+1]]; falls back to s[i] when the parabola degenerates,
// |curvature| <= 1e-300, or the argmax touches the array ends).
inline double shock_ridge_radius(const double* s, const double* p, const int n) {
  TENRYU_ASSERT(s != nullptr, "shock ridge radii must not be null");
  TENRYU_ASSERT(p != nullptr, "shock ridge pressures must not be null");
  TENRYU_ASSERT(n >= 8, "shock ridge profile requires at least eight bins");

  for (int i = 0; i < n; ++i) {
    TENRYU_ASSERT(std::isfinite(s[i]),
                  "shock ridge radii must be finite");
    TENRYU_ASSERT(std::isfinite(p[i]) && p[i] > 0.0,
                  "shock ridge pressures must be positive and finite");
    if (i > 0) {
      TENRYU_ASSERT(s[i - 1] < s[i],
                    "shock ridge radii must be strictly increasing");
    }
  }

  std::vector<double> gradient(static_cast<std::size_t>(n), 0.0);
  int ridge_index = 1;
  double ridge_gradient = -1.0;
  for (int i = 1; i < n - 1; ++i) {
    const double h_left = s[i] - s[i - 1];
    const double h_right = s[i + 1] - s[i];
    const double derivative =
        -h_right * p[i - 1] / (h_left * (h_left + h_right)) +
        (h_right - h_left) * p[i] / (h_left * h_right) +
        h_left * p[i + 1] / (h_right * (h_left + h_right));
    gradient[static_cast<std::size_t>(i)] = std::abs(derivative);
    if (gradient[static_cast<std::size_t>(i)] > ridge_gradient) {
      ridge_gradient = gradient[static_cast<std::size_t>(i)];
      ridge_index = i;
    }
  }

  if (ridge_index == 1 || ridge_index == n - 2) {
    return s[ridge_index];
  }

  const double x0 = s[ridge_index - 1];
  const double x1 = s[ridge_index];
  const double x2 = s[ridge_index + 1];
  const double y0 = gradient[static_cast<std::size_t>(ridge_index - 1)];
  const double y1 = gradient[static_cast<std::size_t>(ridge_index)];
  const double y2 = gradient[static_cast<std::size_t>(ridge_index + 1)];
  const double slope_left = (y1 - y0) / (x1 - x0);
  const double slope_right = (y2 - y1) / (x2 - x1);
  const double curvature =
      2.0 * (slope_right - slope_left) / (x2 - x0);
  if (std::abs(curvature) <= 1.0e-300) {
    return x1;
  }

  const double vertex =
      0.5 * (x0 + x1) - slope_left / curvature;
  return std::clamp(vertex, x0, x2);
}

// Rolling arrival predictor. Deterministic, fixed-capacity ring buffer of the
// last `capacity` (t, s_ridge) samples (default 16, min 4).
class ShockApproachTracker {
 public:
  explicit ShockApproachTracker(const int capacity = 16)
      : capacity_(capacity) {
    TENRYU_ASSERT(capacity >= 4,
                  "shock approach tracker capacity must be at least four");
    times_.resize(static_cast<std::size_t>(capacity), 0.0);
    radii_.resize(static_cast<std::size_t>(capacity), 0.0);
  }

  void push(const double t, const double s_ridge) {
    TENRYU_ASSERT(std::isfinite(t),
                  "shock approach sample time must be finite");
    TENRYU_ASSERT(std::isfinite(s_ridge),
                  "shock approach sample radius must be finite");
    if (size_ > 0) {
      const int newest = (next_ + capacity_ - 1) % capacity_;
      TENRYU_ASSERT(times_[static_cast<std::size_t>(newest)] < t,
                    "shock approach sample times must be strictly increasing");
    }

    times_[static_cast<std::size_t>(next_)] = t;
    radii_[static_cast<std::size_t>(next_)] = s_ridge;
    next_ = (next_ + 1) % capacity_;
    if (size_ < capacity_) {
      ++size_;
    }
  }

  int size() const { return size_; }

  // Least-squares quadratic fit s(t) = c0 + c1*t + c2*t^2 over the stored
  // samples (all of them; requires size() >= 3; with exactly 3 the fit is
  // exact). Returns false when size() < 3 or the normal-equation solve is
  // degenerate (|det| <= 1e-300 after centering; see below).
  // Numerical conditioning: center time as tau = t - t_last before forming
  // the 3x3 normal equations (Cramer solve), so coefficients are evaluated
  // in the tau frame; convert derived quantities accordingly.
  bool fit(double* c0, double* c1, double* c2) const {
    TENRYU_ASSERT(c0 != nullptr, "shock approach fit c0 must not be null");
    TENRYU_ASSERT(c1 != nullptr, "shock approach fit c1 must not be null");
    TENRYU_ASSERT(c2 != nullptr, "shock approach fit c2 must not be null");
    if (size_ < 3) {
      return false;
    }

    const int newest = (next_ + capacity_ - 1) % capacity_;
    const double t_last = times_[static_cast<std::size_t>(newest)];
    double sum_tau = 0.0;
    double sum_tau2 = 0.0;
    double sum_tau3 = 0.0;
    double sum_tau4 = 0.0;
    double sum_s = 0.0;
    double sum_tau_s = 0.0;
    double sum_tau2_s = 0.0;
    for (int k = 0; k < size_; ++k) {
      const int index = (next_ + capacity_ - size_ + k) % capacity_;
      const double tau = times_[static_cast<std::size_t>(index)] - t_last;
      const double tau2 = tau * tau;
      const double radius = radii_[static_cast<std::size_t>(index)];
      sum_tau += tau;
      sum_tau2 += tau2;
      sum_tau3 += tau2 * tau;
      sum_tau4 += tau2 * tau2;
      sum_s += radius;
      sum_tau_s += tau * radius;
      sum_tau2_s += tau2 * radius;
    }

    const double count = static_cast<double>(size_);
    const double det = shock_approach_detail::determinant_3x3(
        count, sum_tau, sum_tau2, sum_tau, sum_tau2, sum_tau3,
        sum_tau2, sum_tau3, sum_tau4);
    if (std::abs(det) <= 1.0e-300) {
      return false;
    }

    const double det_c0 = shock_approach_detail::determinant_3x3(
        sum_s, sum_tau, sum_tau2, sum_tau_s, sum_tau2, sum_tau3,
        sum_tau2_s, sum_tau3, sum_tau4);
    const double det_c1 = shock_approach_detail::determinant_3x3(
        count, sum_s, sum_tau2, sum_tau, sum_tau_s, sum_tau3,
        sum_tau2, sum_tau2_s, sum_tau4);
    const double det_c2 = shock_approach_detail::determinant_3x3(
        count, sum_tau, sum_s, sum_tau, sum_tau2, sum_tau_s,
        sum_tau2, sum_tau3, sum_tau2_s);
    *c0 = det_c0 / det;
    *c1 = det_c1 / det;
    *c2 = det_c2 / det;
    return true;
  }

  // RMS residual of the current quadratic fit over the stored samples
  // (same fit as fit(); false when fit() fails).
  bool fit_residual_rms(double* rms) const {
    TENRYU_ASSERT(rms != nullptr,
                  "shock approach residual output must not be null");
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    if (!fit(&c0, &c1, &c2)) {
      return false;
    }

    const int newest = (next_ + capacity_ - 1) % capacity_;
    const double t_last = times_[static_cast<std::size_t>(newest)];
    double sum_squared_residual = 0.0;
    for (int k = 0; k < size_; ++k) {
      const int index = (next_ + capacity_ - size_ + k) % capacity_;
      const double tau = times_[static_cast<std::size_t>(index)] - t_last;
      const double fitted = c0 + c1 * tau + c2 * tau * tau;
      const double residual =
          radii_[static_cast<std::size_t>(index)] - fitted;
      sum_squared_residual += residual * residual;
    }
    *rms = std::sqrt(sum_squared_residual / static_cast<double>(size_));
    return true;
  }

  // Current front speed toward the center (positive when moving inward):
  //   v = -(ds/dt) at the newest sample. False when fit() fails.
  bool speed(double* v_inward) const {
    TENRYU_ASSERT(v_inward != nullptr,
                  "shock approach speed output must not be null");
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    if (!fit(&c0, &c1, &c2)) {
      return false;
    }
    *v_inward = -c1;
    return true;
  }

  // Predicted time at which s(t) reaches target_s (smallest tau >= 0 root of
  // the centered quadratic c0 + c1*tau + c2*tau^2 = target_s, converted to
  // absolute t). Handles the linear fallback |c2| <= 1e-300. Returns false
  // when fit() fails, when no root with tau >= 0 exists, or when the front is
  // not approaching the target (v_inward <= 0 while s_last > target_s).
  bool predict_arrival(const double target_s, double* t_arrival) const {
    TENRYU_ASSERT(t_arrival != nullptr,
                  "shock approach arrival output must not be null");
    double c0 = 0.0;
    double c1 = 0.0;
    double c2 = 0.0;
    if (!fit(&c0, &c1, &c2)) {
      return false;
    }

    const int newest = (next_ + capacity_ - 1) % capacity_;
    const double t_last = times_[static_cast<std::size_t>(newest)];
    const double s_last = radii_[static_cast<std::size_t>(newest)];
    const double v_inward = -c1;
    if (v_inward <= 0.0 && s_last > target_s) {
      return false;
    }

    const double constant = c0 - target_s;
    double tau = 0.0;
    if (std::abs(c2) <= 1.0e-300) {
      if (std::abs(c1) <= 1.0e-300) {
        if (std::abs(constant) > 1.0e-300) {
          return false;
        }
      } else {
        tau = -constant / c1;
        if (tau < 0.0) {
          return false;
        }
      }
    } else {
      const double discriminant = c1 * c1 - 4.0 * c2 * constant;
      if (discriminant < 0.0) {
        return false;
      }
      const double root = std::sqrt(discriminant);
      const double q = -0.5 * (c1 + std::copysign(root, c1));
      const double tau_a = q / c2;
      const double tau_b = q == 0.0 ? 0.0 : constant / q;
      const bool a_valid = tau_a >= 0.0;
      const bool b_valid = tau_b >= 0.0;
      if (!a_valid && !b_valid) {
        return false;
      }
      if (a_valid && b_valid) {
        tau = std::min(tau_a, tau_b);
      } else {
        tau = a_valid ? tau_a : tau_b;
      }
    }

    *t_arrival = t_last + tau;
    return true;
  }

 private:
  int capacity_ = 0;
  std::vector<double> times_;
  std::vector<double> radii_;
  int size_ = 0;
  int next_ = 0;
};

struct SectorFrontSample {
  double s_ridge;
  bool valid;
};

class SectorShockState {
 public:
  explicit SectorShockState(const int sectors)
      : n_sec(sectors) {
    TENRYU_ASSERT(sectors >= 4 && sectors % 2 == 0,
                  "sector shock state requires an even sector count >= 4");
    trackers.assign(static_cast<std::size_t>(sectors),
                    ShockApproachTracker(16));
    last_sigma_t.assign(static_cast<std::size_t>(sectors), 0.0);
  }

  int n_sec;
  std::vector<ShockApproachTracker> trackers;
  std::vector<double> last_sigma_t;
  double committed_deadline_s = std::numeric_limits<double>::infinity();
  double last_call_t = -1.0;
};

// Weighted least-squares Legendre fit of y_k ~ sum_{l=0..l_max} b_l P_l(mu_k)
// with weights w_k (solid angle x confidence). Deterministic dense normal
// equations solved by Gaussian elimination with partial pivoting on the
// (l_max+1)^2 system. Returns false when fewer than (l_max+1) samples have
// w_k > 0 or the system is singular (|pivot| <= 1e-300).
inline bool legendre_fit(const double* mu,
                         const double* y,
                         const double* w,
                         const int n,
                         const int l_max,
                         double* b_out) {
  TENRYU_ASSERT(mu != nullptr, "Legendre fit mu must not be null");
  TENRYU_ASSERT(y != nullptr, "Legendre fit values must not be null");
  TENRYU_ASSERT(w != nullptr, "Legendre fit weights must not be null");
  TENRYU_ASSERT(b_out != nullptr, "Legendre fit output must not be null");
  TENRYU_ASSERT(l_max >= 0 && l_max <= 4,
                "Legendre fit maximum degree must be in [0, 4]");

  const int order = l_max + 1;
  std::array<std::array<double, 6>, 5> augmented{};
  int positive_weight_samples = 0;
  for (int k = 0; k < n; ++k) {
    if (!(w[k] > 0.0)) {
      continue;
    }
    ++positive_weight_samples;
    std::array<double, 5> p{};
    p[0] = 1.0;
    if (l_max >= 1) {
      p[1] = mu[k];
    }
    for (int l = 1; l < l_max; ++l) {
      p[static_cast<std::size_t>(l + 1)] =
          ((2.0 * static_cast<double>(l) + 1.0) * mu[k] *
               p[static_cast<std::size_t>(l)] -
           static_cast<double>(l) * p[static_cast<std::size_t>(l - 1)]) /
          static_cast<double>(l + 1);
    }
    for (int row = 0; row < order; ++row) {
      for (int col = 0; col < order; ++col) {
        augmented[static_cast<std::size_t>(row)]
                 [static_cast<std::size_t>(col)] +=
            w[k] * p[static_cast<std::size_t>(row)] *
            p[static_cast<std::size_t>(col)];
      }
      augmented[static_cast<std::size_t>(row)]
               [static_cast<std::size_t>(order)] +=
          w[k] * p[static_cast<std::size_t>(row)] * y[k];
    }
  }
  if (positive_weight_samples < order) {
    return false;
  }

  for (int col = 0; col < order; ++col) {
    int pivot_row = col;
    double pivot_magnitude =
        std::abs(augmented[static_cast<std::size_t>(col)]
                          [static_cast<std::size_t>(col)]);
    for (int row = col + 1; row < order; ++row) {
      const double candidate =
          std::abs(augmented[static_cast<std::size_t>(row)]
                            [static_cast<std::size_t>(col)]);
      if (candidate > pivot_magnitude) {
        pivot_magnitude = candidate;
        pivot_row = row;
      }
    }
    if (pivot_magnitude <= 1.0e-300) {
      return false;
    }
    if (pivot_row != col) {
      std::swap(augmented[static_cast<std::size_t>(pivot_row)],
                augmented[static_cast<std::size_t>(col)]);
    }
    for (int row = col + 1; row < order; ++row) {
      const double factor =
          augmented[static_cast<std::size_t>(row)]
                   [static_cast<std::size_t>(col)] /
          augmented[static_cast<std::size_t>(col)]
                   [static_cast<std::size_t>(col)];
      for (int entry = col; entry <= order; ++entry) {
        augmented[static_cast<std::size_t>(row)]
                 [static_cast<std::size_t>(entry)] -=
            factor * augmented[static_cast<std::size_t>(col)]
                              [static_cast<std::size_t>(entry)];
      }
    }
  }

  std::array<double, 5> solution{};
  for (int row = order - 1; row >= 0; --row) {
    double rhs = augmented[static_cast<std::size_t>(row)]
                          [static_cast<std::size_t>(order)];
    for (int col = row + 1; col < order; ++col) {
      rhs -= augmented[static_cast<std::size_t>(row)]
                      [static_cast<std::size_t>(col)] *
             solution[static_cast<std::size_t>(col)];
    }
    const double pivot = augmented[static_cast<std::size_t>(row)]
                                  [static_cast<std::size_t>(row)];
    if (std::abs(pivot) <= 1.0e-300) {
      return false;
    }
    solution[static_cast<std::size_t>(row)] = rhs / pivot;
  }
  for (int l = 0; l < order; ++l) {
    b_out[l] = solution[static_cast<std::size_t>(l)];
  }
  return true;
}

// Earliest confidence-bounded deadline over valid sectors:
//   min_k [t_arr_k - nu*sigma_k - n_guard*h_cell/|v_k|] - dt_scan.
// Sectors with have_arr==false or v_k <= 0 are skipped. Returns false when
// no sector qualifies.
inline bool earliest_confidence_deadline(const double* t_arr,
                                         const double* sigma_t,
                                         const double* v_inward,
                                         const bool* valid,
                                         const int n,
                                         const double nu,
                                         const double n_guard,
                                         const double h_cell,
                                         const double dt_scan,
                                         double* t_end_out) {
  TENRYU_ASSERT(t_arr != nullptr, "sector arrival times must not be null");
  TENRYU_ASSERT(sigma_t != nullptr, "sector arrival sigmas must not be null");
  TENRYU_ASSERT(v_inward != nullptr, "sector speeds must not be null");
  TENRYU_ASSERT(valid != nullptr, "sector valid flags must not be null");
  TENRYU_ASSERT(t_end_out != nullptr, "sector deadline output must not be null");

  double earliest = std::numeric_limits<double>::infinity();
  bool have_deadline = false;
  for (int k = 0; k < n; ++k) {
    if (!valid[k] || v_inward[k] <= 0.0) {
      continue;
    }
    const double candidate =
        t_arr[k] - nu * sigma_t[k] -
        n_guard * h_cell / std::abs(v_inward[k]);
    earliest = std::min(earliest, candidate);
    have_deadline = true;
  }
  if (!have_deadline) {
    return false;
  }
  *t_end_out = earliest - dt_scan;
  return true;
}

// Move-earlier-only committed deadline with maturity/future gating.
// Returns true when the candidate was committed into *committed.
inline bool commit_sector_deadline(double* committed,
                                   const double candidate,
                                   const double t_now,
                                   const int max_tracker_size,
                                   bool* held_out) {
  TENRYU_ASSERT(committed != nullptr,
                "committed sector deadline must not be null");
  TENRYU_ASSERT(held_out != nullptr,
                "sector deadline held output must not be null");
  *held_out = false;
  if (!(candidate > t_now) || max_tracker_size < 8) {
    return false;
  }
  if (candidate > *committed) {
    *held_out = true;
    return false;
  }
  *committed = std::min(*committed, candidate);
  return true;
}

// Crossing-time lead: number of shock cell-crossing times between now and the
// predicted arrival: N_lead = (t_arrival - t_now) * v_inward / h_cell.
// Returns false when v_inward <= 0 or h_cell <= 0.
inline bool crossing_time_lead(const double t_now,
                               const double t_arrival,
                               const double v_inward,
                               const double h_cell,
                               double* n_lead) {
  TENRYU_ASSERT(n_lead != nullptr,
                "shock crossing-time lead output must not be null");
  if (v_inward <= 0.0 || h_cell <= 0.0) {
    return false;
  }
  *n_lead = (t_arrival - t_now) * v_inward / h_cell;
  return true;
}

}  // namespace tenryu::diagnostics
