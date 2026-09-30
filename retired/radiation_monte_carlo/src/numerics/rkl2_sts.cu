#include "numerics/rkl2_sts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/error.hpp"

namespace tenryu::numerics {
namespace {

struct LegendreValues {
  double p = 0.0;
  double dp = 0.0;
  double ddp = 0.0;
};

std::vector<LegendreValues> compute_legendre_values(const int s,
                                                    const double x) {
  std::vector<LegendreValues> values(static_cast<std::size_t>(s + 1));
  values[0].p = 1.0;
  if (s == 0) {
    return values;
  }

  values[1].p = x;
  values[1].dp = 1.0;
  values[1].ddp = 0.0;
  for (int j = 2; j <= s; ++j) {
    const double alpha = static_cast<double>(2 * j - 1) / static_cast<double>(j);
    const double beta = static_cast<double>(j - 1) / static_cast<double>(j);
    const auto& jm1 = values[static_cast<std::size_t>(j - 1)];
    const auto& jm2 = values[static_cast<std::size_t>(j - 2)];
    auto& out = values[static_cast<std::size_t>(j)];
    out.p = alpha * x * jm1.p - beta * jm2.p;
    out.dp = alpha * (jm1.p + x * jm1.dp) - beta * jm2.dp;
    out.ddp = alpha * (2.0 * jm1.dp + x * jm1.ddp) - beta * jm2.ddp;
  }
  return values;
}

}  // namespace

RKL2Coefficients compute_rkl2_coefficients(const int s,
                                           const double damping) {
  TENRYU_ASSERT(s >= 2, "compute_rkl2_coefficients requires s >= 2");
  TENRYU_ASSERT(std::isfinite(damping) && damping >= 0.0,
                "compute_rkl2_coefficients requires non-negative damping");

  const double damp = std::max(damping, 0.0);
  const double w0 = 1.0 + 2.0 * damp /
                             (static_cast<double>(s) * static_cast<double>(s + 1));
  const std::vector<LegendreValues> legendre =
      compute_legendre_values(s, w0);
  const double dp_s = legendre[static_cast<std::size_t>(s)].dp;
  const double ddp_s = legendre[static_cast<std::size_t>(s)].ddp;
  TENRYU_ASSERT(std::isfinite(dp_s) && std::isfinite(ddp_s) &&
                    std::abs(ddp_s) > 0.0,
                "compute_rkl2_coefficients invalid Legendre derivatives");
  const double w1 = dp_s / ddp_s;

  std::vector<double> a(static_cast<std::size_t>(s + 1), 0.0);
  std::vector<double> b(static_cast<std::size_t>(s + 1), 1.0 / 3.0);
  for (int j = 2; j <= s; ++j) {
    const auto& v = legendre[static_cast<std::size_t>(j)];
    TENRYU_ASSERT(std::isfinite(v.dp) && std::isfinite(v.ddp) &&
                      std::abs(v.dp) > 0.0,
                  "compute_rkl2_coefficients invalid stage derivative");
    b[static_cast<std::size_t>(j)] = v.ddp / (v.dp * v.dp);
  }
  for (int j = 0; j <= s; ++j) {
    a[static_cast<std::size_t>(j)] =
        1.0 - b[static_cast<std::size_t>(j)] *
                  legendre[static_cast<std::size_t>(j)].p;
  }

  RKL2Coefficients coeff;
  coeff.s = s;
  coeff.mu.assign(static_cast<std::size_t>(s + 1), 0.0);
  coeff.nu.assign(static_cast<std::size_t>(s + 1), 0.0);
  coeff.mu_tilde.assign(static_cast<std::size_t>(s + 1), 0.0);
  coeff.gamma_tilde.assign(static_cast<std::size_t>(s + 1), 0.0);

  coeff.mu_tilde[1] = b[1] * w1;
  for (int j = 2; j <= s; ++j) {
    const std::size_t ju = static_cast<std::size_t>(j);
    const double alpha = static_cast<double>(2 * j - 1) /
                         static_cast<double>(j);
    coeff.mu[ju] = alpha * b[ju] / b[ju - 1U] * w0;
    coeff.nu[ju] = -static_cast<double>(j - 1) / static_cast<double>(j) *
                   b[ju] / b[ju - 2U];
    coeff.mu_tilde[ju] = alpha * b[ju] / b[ju - 1U] * w1;
    coeff.gamma_tilde[ju] = -a[ju - 1U] * coeff.mu_tilde[ju];
  }

  return coeff;
}

int estimate_rkl2_stages(const double dt,
                         const double dt_explicit,
                         const double safety) {
  if (!(dt > 0.0) || !std::isfinite(dt) ||
      !(dt_explicit > 0.0) || !std::isfinite(dt_explicit)) {
    return 2;
  }
  const double safe = (std::isfinite(safety) && safety > 0.0) ? safety : 1.0;
  const double ratio = dt / (dt_explicit * safe);
  const double required_real = std::ceil(
      0.5 * (std::sqrt(9.0 + 16.0 * std::max(ratio, 0.0)) - 1.0));
  if (!std::isfinite(required_real) ||
      required_real >= static_cast<double>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  const int required = static_cast<int>(required_real);
  return std::max(required, 2);
}

int estimate_rkl2_subcycles(const double dt,
                            const double dt_explicit,
                            const int max_stages,
                            const double safety) {
  if (!(dt > 0.0) || !std::isfinite(dt) ||
      !(dt_explicit > 0.0) || !std::isfinite(dt_explicit) ||
      max_stages <= 0) {
    return 1;
  }
  const int s_max = std::max(max_stages, 2);
  if (estimate_rkl2_stages(dt, dt_explicit, safety) <= s_max) {
    return 1;
  }
  const double safe = (std::isfinite(safety) && safety > 0.0) ? safety : 1.0;
  const double s_max_real = static_cast<double>(s_max);
  const double capacity =
      (s_max_real * s_max_real + s_max_real - 2.0) / 4.0;
  const double dt_sub_max = safe * capacity * dt_explicit;
  if (!(dt_sub_max > 0.0) || !std::isfinite(dt_sub_max)) {
    return std::numeric_limits<int>::max();
  }
  const double n_sub_real = std::ceil(dt / dt_sub_max);
  if (!std::isfinite(n_sub_real) ||
      n_sub_real >= static_cast<double>(std::numeric_limits<int>::max())) {
    return std::numeric_limits<int>::max();
  }
  return std::max(1, static_cast<int>(n_sub_real));
}

}  // namespace tenryu::numerics
