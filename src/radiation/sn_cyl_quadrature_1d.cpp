#include "radiation/sn_cyl_quadrature_1d.hpp"

#include <cmath>
#include <cstddef>

#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// Gauss-Legendre nodes/weights on (-1, 1), ascending nodes. Same Newton
// iteration as compute_gauss_legendre in sn_transport_1d_gpu.cu.
void gauss_legendre_full(const int n,
                         std::vector<double>& x,
                         std::vector<double>& w) {
  x.assign(static_cast<std::size_t>(n), 0.0);
  w.assign(static_cast<std::size_t>(n), 0.0);
  constexpr double tol = 1.0e-15;
  const int half = n / 2;
  for (int k = 0; k < half; ++k) {
    double xk = std::cos(kPi * (static_cast<double>(k) + 0.75) /
                         (static_cast<double>(n) + 0.5));
    double p1 = 0.0;
    double p2 = 0.0;
    for (int iter = 0; iter < 64; ++iter) {
      p1 = 1.0;
      p2 = 0.0;
      for (int j = 1; j <= n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * static_cast<double>(j) - 1.0) * xk * p2 -
              (static_cast<double>(j) - 1.0) * p3) /
             static_cast<double>(j);
      }
      const double pp =
          static_cast<double>(n) * (xk * p1 - p2) / (xk * xk - 1.0);
      const double dx = p1 / pp;
      xk -= dx;
      if (std::abs(dx) <= tol) {
        break;
      }
    }
    p1 = 1.0;
    p2 = 0.0;
    for (int j = 1; j <= n; ++j) {
      const double p3 = p2;
      p2 = p1;
      p1 = ((2.0 * static_cast<double>(j) - 1.0) * xk * p2 -
            (static_cast<double>(j) - 1.0) * p3) /
           static_cast<double>(j);
    }
    const double pp =
        static_cast<double>(n) * (xk * p1 - p2) / (xk * xk - 1.0);
    const double wk = 2.0 / ((1.0 - xk * xk) * pp * pp);
    x[static_cast<std::size_t>(k)] = -xk;
    x[static_cast<std::size_t>(n - 1 - k)] = xk;
    w[static_cast<std::size_t>(k)] = wk;
    w[static_cast<std::size_t>(n - 1 - k)] = wk;
  }
}

}  // namespace

int sn_cyl_levels_for(const int n_angles) {
  if (n_angles < 8 || (n_angles % 2) != 0) {
    return 0;
  }
  const int half = n_angles / 2;
  const int L = static_cast<int>(
      std::llround(std::sqrt(static_cast<double>(half))));
  return (L >= 2 && L * L == half) ? L : 0;
}

SnCylQuadrature1D build_sn_cyl_quadrature_1d(const int n_angles) {
  const int L = sn_cyl_levels_for(n_angles);
  TENRYU_ASSERT(L >= 2,
                "cylindrical S_N requires n_angles = 2*L*L with integer "
                "L >= 2 (8, 18, 32, 50, 72, ...)");
  const int M = 2 * L;
  const int n_flat = L * M;

  SnCylQuadrature1D q;
  q.n_levels = L;
  q.n_azim = M;
  q.mu.assign(static_cast<std::size_t>(n_flat), 0.0);
  q.weight.assign(static_cast<std::size_t>(n_flat), 0.0);
  q.alpha_half.assign(static_cast<std::size_t>(n_flat + 1), 0.0);
  q.tau_wd.assign(static_cast<std::size_t>(n_flat), 0.5);
  q.sd_mu.assign(static_cast<std::size_t>(L), 0.0);

  // Polar levels: positive-half nodes of GL(2L) on (-1,1), ascending xi.
  // Their GL weights sum to exactly 1 over the half by symmetry.
  std::vector<double> gl_x;
  std::vector<double> gl_w;
  gauss_legendre_full(2 * L, gl_x, gl_w);

  // Azimuthal pattern (unit level, sin(theta) == 1): Chebyshev midpoint
  // cosines with exact +-pair symmetry. c_k = cos((2k+1)pi/(2M)) descends
  // with k; the ascending-mu pattern is (-c_0 ... -c_{M/2-1}, +c_{M/2-1}
  // ... +c_0) so the axis-reflection partner of m is exactly M-1-m with
  // bit-equal |mu|.
  const int Mh = M / 2;
  std::vector<double> c_pos(static_cast<std::size_t>(Mh), 0.0);
  for (int k = 0; k < Mh; ++k) {
    c_pos[static_cast<std::size_t>(k)] =
        std::cos(kPi * (2.0 * static_cast<double>(k) + 1.0) /
                 (2.0 * static_cast<double>(M)));
  }
  std::vector<double> cos_mid(static_cast<std::size_t>(M), 0.0);
  for (int m = 0; m < Mh; ++m) {
    cos_mid[static_cast<std::size_t>(m)] = -c_pos[static_cast<std::size_t>(m)];
    cos_mid[static_cast<std::size_t>(M - 1 - m)] =
        c_pos[static_cast<std::size_t>(m)];
  }

  // Azimuthal cell-edge cosines (M&M Eq. A4 with equal weights: omega edges
  // uniform on (pi -> 0), edge cosines cos(omega_edge)), built antisymmetric
  // so ce[j] == -ce[M-j] exactly in floating point.
  std::vector<double> ce(static_cast<std::size_t>(M + 1), 0.0);
  ce[0] = -1.0;
  ce[static_cast<std::size_t>(M)] = 1.0;
  for (int j = 1; j < M; ++j) {
    if (2 * j == M) {
      ce[static_cast<std::size_t>(j)] = 0.0;
    } else if (2 * j < M) {
      ce[static_cast<std::size_t>(j)] =
          -std::cos(kPi * static_cast<double>(j) / static_cast<double>(M));
    } else {
      ce[static_cast<std::size_t>(j)] =
          std::cos(kPi * static_cast<double>(M - j) / static_cast<double>(M));
    }
  }

  // Weighted-diamond factor (M&M Eqs. A1-A3): tau_m = (mu_m - mu_{m-1/2}) /
  // (mu_{m+1/2} - mu_{m-1/2}); sin(theta_l) cancels, so tau is
  // level-independent (M-periodic in the flattened array).
  std::vector<double> tau_pattern(static_cast<std::size_t>(M), 0.5);
  for (int m = 0; m < M; ++m) {
    const double lo = ce[static_cast<std::size_t>(m)];
    const double hi = ce[static_cast<std::size_t>(m + 1)];
    const double denom = hi - lo;
    TENRYU_ASSERT(denom > 0.0,
                  "cylindrical S_N azimuthal edge cosines must ascend");
    const double t = (cos_mid[static_cast<std::size_t>(m)] - lo) / denom;
    TENRYU_ASSERT(t > 0.0 && t < 1.0,
                  "cylindrical S_N weighted-diamond tau must lie in (0,1)");
    tau_pattern[static_cast<std::size_t>(m)] = t;
  }

  const double u_azim = 2.0 / static_cast<double>(M);
  for (int l = 0; l < L; ++l) {
    const double xi = gl_x[static_cast<std::size_t>(L + l)];
    const double v_l = gl_w[static_cast<std::size_t>(L + l)];
    const double sin_theta = std::sqrt((1.0 - xi) * (1.0 + xi));
    TENRYU_ASSERT(sin_theta > 0.0 && sin_theta < 1.0,
                  "cylindrical S_N level sine out of range");
    q.sd_mu[static_cast<std::size_t>(l)] = sin_theta;
    const int base = l * M;
    for (int m = 0; m < M; ++m) {
      const std::size_t p = static_cast<std::size_t>(base + m);
      q.mu[p] = sin_theta * cos_mid[static_cast<std::size_t>(m)];
      q.weight[p] = v_l * u_azim;
      q.tau_wd[p] = tau_pattern[static_cast<std::size_t>(m)];
    }
    // Per-level Carlson recursion; both level edges pinned to exactly 0.0
    // (the recursion end lands on 0 up to roundoff by +-mu pairing). Same
    // snap/clamp policy as the spherical angular_coefficients.
    q.alpha_half[static_cast<std::size_t>(base)] = 0.0;
    for (int m = 0; m < M; ++m) {
      q.alpha_half[static_cast<std::size_t>(base + m + 1)] =
          q.alpha_half[static_cast<std::size_t>(base + m)] -
          q.mu[static_cast<std::size_t>(base + m)] *
              q.weight[static_cast<std::size_t>(base + m)];
    }
    q.alpha_half[static_cast<std::size_t>(base + M)] = 0.0;
    for (int m = 1; m < M; ++m) {
      double& value = q.alpha_half[static_cast<std::size_t>(base + m)];
      value = (std::abs(value) < 1.0e-14) ? 0.0 : std::max(value, 0.0);
      // Interior edges must stay strictly positive: the sweep kernels treat
      // alpha == 0 as a level-chain delimiter.
      TENRYU_ASSERT(value > 0.0,
                    "cylindrical S_N interior alpha edge collapsed to zero");
    }
  }
  return q;
}

}  // namespace tenryu::radiation
