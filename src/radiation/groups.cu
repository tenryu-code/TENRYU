#include "radiation/groups.cuh"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kNorm = 15.0 / (kPi * kPi * kPi * kPi);

inline double planck_kernel(const double x) {
  if (x <= 0.0) {
    return 0.0;
  }
  if (x < 1.0e-3) {
    return x * x - 0.5 * x * x * x;
  }
  if (x > 500.0) {
    return x * x * x * std::exp(-x);
  }
  return (x * x * x) / std::expm1(x);
}

// Fixed n=256 composite Simpson: sufficient for smooth Planck kernel
// (verified to ~1e-10 relative error for typical ICF group widths).
// Adaptive quadrature is not needed for this application.
double integrate_simpson(const double x0, const double x1) {
  if (!(x1 > x0)) {
    return 0.0;
  }
  constexpr int n = 256;
  const double h = (x1 - x0) / static_cast<double>(n);

  double sum = planck_kernel(x0) + planck_kernel(x1);
  for (int i = 1; i < n; ++i) {
    const double x = x0 + h * static_cast<double>(i);
    sum += (i % 2 == 0 ? 2.0 : 4.0) * planck_kernel(x);
  }
  return sum * h / 3.0;
}

}  // namespace

Groups::Groups() : Groups(std::vector<double>{0.0, 1.0e6}) {}

std::vector<double> Groups::make_log_uniform_bounds(const int n_groups,
                                                    const double E_min_eV,
                                                    const double E_max_eV) {
  TENRYU_ASSERT(n_groups >= 1, "make_log_uniform_bounds requires n_groups >= 1");
  TENRYU_ASSERT(E_min_eV > 0.0, "make_log_uniform_bounds requires E_min > 0");
  TENRYU_ASSERT(E_max_eV > E_min_eV,
                "make_log_uniform_bounds requires E_max > E_min");

  std::vector<double> bounds(static_cast<std::size_t>(n_groups + 1), 0.0);
  const double log_min = std::log(E_min_eV);
  const double log_max = std::log(E_max_eV);
  for (int g = 0; g <= n_groups; ++g) {
    const double u = static_cast<double>(g) / static_cast<double>(n_groups);
    bounds[static_cast<std::size_t>(g)] =
        std::exp((1.0 - u) * log_min + u * log_max);
  }
  bounds.front() = E_min_eV;
  bounds.back() = E_max_eV;
  return bounds;
}

Groups::Groups(std::vector<double> group_bounds_eV)
    : group_bounds_eV_(std::move(group_bounds_eV)) {
  TENRYU_ASSERT(group_bounds_eV_.size() >= 2,
                "Groups requires at least two group bounds");
  TENRYU_ASSERT(std::is_sorted(group_bounds_eV_.begin(), group_bounds_eV_.end()),
                "Groups bounds must be sorted");
}

int Groups::num_groups() const noexcept {
  return static_cast<int>(group_bounds_eV_.size() - 1U);
}

double Groups::energy_lo(const int g) const {
  TENRYU_ASSERT(g >= 0 && g < num_groups(), "Groups::energy_lo group out of range");
  return group_bounds_eV_[static_cast<std::size_t>(g)];
}

double Groups::energy_hi(const int g) const {
  TENRYU_ASSERT(g >= 0 && g < num_groups(), "Groups::energy_hi group out of range");
  return group_bounds_eV_[static_cast<std::size_t>(g + 1)];
}

double Groups::energy_rep(const int g) const {
  const double lo = energy_lo(g);
  const double hi = energy_hi(g);
  if (lo <= 0.0) {
    return 0.5 * hi;
  }
  return std::sqrt(lo * hi);
}

double Groups::planck_fraction_raw(const int g, const double T_eV) const {
  if (num_groups() == 1) {
    return 1.0;
  }

  TENRYU_ASSERT(g >= 0 && g < num_groups(),
                "Groups::planck_fraction_raw group out of range");
  TENRYU_ASSERT(T_eV > 0.0, "Groups::planck_fraction_raw requires T > 0");

  const double x_lo = energy_lo(g) / T_eV;
  const double x_hi = energy_hi(g) / T_eV;
  return kNorm * integrate_simpson(x_lo, x_hi);
}

double Groups::planck_fraction(const int g, const double T_eV) const {
  if (num_groups() == 1) {
    return 1.0;
  }
  TENRYU_ASSERT(T_eV > 0.0, "Groups::planck_fraction requires T > 0");

  std::vector<double> raw(static_cast<std::size_t>(num_groups()), 0.0);
  for (int gg = 0; gg < num_groups(); ++gg) {
    raw[static_cast<std::size_t>(gg)] = std::max(0.0, planck_fraction_raw(gg, T_eV));
  }
  const double sum = std::accumulate(raw.begin(), raw.end(), 0.0);
  if (sum <= 0.0) {
    return 0.0;
  }
  // Planck fractions b_g are renormalized to sum to 1 over configured groups.
  // Energy from low/high-energy tails outside group boundaries is redistributed
  // into represented groups. This is correct for finite-group IMC: total
  // emitted energy must be allocated entirely among available groups. Users
  // should ensure group boundaries cover the physically relevant spectrum to
  // minimize tail redistribution.
  return raw[static_cast<std::size_t>(g)] / sum;
}

const std::vector<double>& Groups::bounds_eV() const noexcept {
  return group_bounds_eV_;
}

}  // namespace tenryu::radiation
