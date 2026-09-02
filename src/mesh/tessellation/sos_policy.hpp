#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "core/error.hpp"
#include "mesh/tessellation/exact_types.hpp"

namespace tenryu::mesh::tess {

// Site::rank is the dense rank in ascending stable_id order and is computed
// once by the caller for each tessellation build.

constexpr double kShewchukEpsilon =
    0.5 * std::numeric_limits<double>::epsilon();
// Shewchuk's unit roundoff is DBL_EPSILON / 2.  His orient2d "A" bound is
// (3 + 16 epsilon) epsilon times |acx bcy| + |acy bcx|, and his incircle
// "A" bound is (10 + 96 epsilon) epsilon times the permanent below.
constexpr double kOrientErrBound =
    (3.0 + 16.0 * kShewchukEpsilon) * kShewchukEpsilon;
constexpr double kIncircleErrBound =
    (10.0 + 96.0 * kShewchukEpsilon) * kShewchukEpsilon;
// The build does not impose a TU-wide FP-contraction policy.  Widen the
// published bounds by 2x so a compiler-contracted filter expression cannot
// turn an uncertain determinant into a claimed-certain sign.
constexpr double kFilterContractionSafety = 2.0;

namespace detail {

struct PredicateCounters {
  std::size_t incircle_fast = 0;
  std::size_t incircle_exact = 0;
  std::size_t orient_fast = 0;
  std::size_t orient_exact = 0;
  std::size_t flips = 0;
  std::size_t locate_steps = 0;
};

extern thread_local PredicateCounters* g_predicate_counters;
// Cheap non-TLS gate for the counter helpers: the thread-local
// lookup itself is measurable in predicate-hot loops. Maintained by
// PredicateCounterReporter alongside the pointer (single-host-thread
// diagnostics, like the tessellation itself).
extern bool g_predicate_counters_active;

inline void count_incircle_fast() {
  if (!g_predicate_counters_active) {
    return;
  }
  if (g_predicate_counters != nullptr) {
    ++g_predicate_counters->incircle_fast;
  }
}

inline void count_incircle_exact() {
  if (!g_predicate_counters_active) {
    return;
  }
  if (g_predicate_counters != nullptr) {
    ++g_predicate_counters->incircle_exact;
  }
}

inline void count_orient_fast() {
  if (!g_predicate_counters_active) {
    return;
  }
  if (g_predicate_counters != nullptr) {
    ++g_predicate_counters->orient_fast;
  }
}

inline void count_orient_exact() {
  if (!g_predicate_counters_active) {
    return;
  }
  if (g_predicate_counters != nullptr) {
    ++g_predicate_counters->orient_exact;
  }
}

inline void count_flip() {
  if (!g_predicate_counters_active) {
    return;
  }
  if (g_predicate_counters != nullptr) {
    ++g_predicate_counters->flips;
  }
}

inline void count_locate_step() {
  if (!g_predicate_counters_active) {
    return;
  }
  if (g_predicate_counters != nullptr) {
    ++g_predicate_counters->locate_steps;
  }
}

// Test-only hook (legacy precedent: voronoi_detail::set_slow_path_for_tests):
// when set, the filtered wrappers route every call through the forced-exact
// expansions so the cutover gate can certify filter-bound correctness by byte
// identity. Never enabled in production paths.
extern bool g_forced_exact_for_tests;

inline void set_forced_exact_for_tests(const bool enabled) {
  g_forced_exact_for_tests = enabled;
}

}  // namespace detail

inline int orient2d_sign_forced_exact(const Site& a,
                                      const Site& b,
                                      const Site& c) {
  detail::count_orient_exact();
  return orient2d_exact_sign(a.r, a.z, b.r, b.z, c.r, c.z);
}

inline int incircle_sign_forced_exact(const Site& a,
                                      const Site& b,
                                      const Site& c,
                                      const Site& d) {
  detail::count_incircle_exact();
  return incircle_exact_sign(a.r, a.z, b.r, b.z, c.r, c.z, d.r, d.z);
}

inline int orient2d_sign_tess(const Site& a,
                              const Site& b,
                              const Site& c) {
  if (detail::g_forced_exact_for_tests) {
    return orient2d_sign_forced_exact(a, b, c);
  }
  const double acx = a.r - c.r;
  const double acy = a.z - c.z;
  const double bcx = b.r - c.r;
  const double bcy = b.z - c.z;
  const double left = acx * bcy;
  const double right = acy * bcx;
  const double determinant = left - right;
  const double error_bound =
      kFilterContractionSafety * kOrientErrBound *
      (std::abs(left) + std::abs(right));
  if (std::abs(determinant) > error_bound) {
    detail::count_orient_fast();
    return determinant > 0.0 ? 1 : -1;
  }
  return orient2d_sign_forced_exact(a, b, c);
}

inline int incircle_sign_tess(const Site& a,
                              const Site& b,
                              const Site& c,
                              const Site& d) {
  if (detail::g_forced_exact_for_tests) {
    return incircle_sign_forced_exact(a, b, c, d);
  }
  const double adx = a.r - d.r;
  const double ady = a.z - d.z;
  const double bdx = b.r - d.r;
  const double bdy = b.z - d.z;
  const double cdx = c.r - d.r;
  const double cdy = c.z - d.z;

  const double bdxcdy = bdx * cdy;
  const double cdxbdy = cdx * bdy;
  const double cdxady = cdx * ady;
  const double adxcdy = adx * cdy;
  const double adxbdy = adx * bdy;
  const double bdxady = bdx * ady;
  const double alift = adx * adx + ady * ady;
  const double blift = bdx * bdx + bdy * bdy;
  const double clift = cdx * cdx + cdy * cdy;
  const double determinant =
      alift * (bdxcdy - cdxbdy) +
      blift * (cdxady - adxcdy) +
      clift * (adxbdy - bdxady);
  const double permanent =
      (std::abs(bdxcdy) + std::abs(cdxbdy)) * alift +
      (std::abs(cdxady) + std::abs(adxcdy)) * blift +
      (std::abs(adxbdy) + std::abs(bdxady)) * clift;
  const double error_bound =
      kFilterContractionSafety * kIncircleErrBound * permanent;
  if (std::abs(determinant) > error_bound) {
    detail::count_incircle_fast();
    return determinant > 0.0 ? 1 : -1;
  }
  return incircle_sign_forced_exact(a, b, c, d);
}

inline int incircle_sos(const Site& a,
                        const Site& b,
                        const Site& c,
                        const Site& d) {
  TENRYU_ASSERT(orient2d_sign_tess(a, b, c) > 0,
                "incircle_sos requires an exact CCW triangle");

  const int exact_sign = incircle_sign_tess(a, b, c, d);
  if (exact_sign != 0) {
    return exact_sign;
  }

  struct CofactorCoefficient {
    const Site* site;
    int sign;
  };
  std::array<CofactorCoefficient, 4> coefficients{{
      {&a, -orient2d_sign_forced_exact(b, c, d)},
      {&b, orient2d_sign_forced_exact(a, c, d)},
      {&c, -orient2d_sign_forced_exact(a, b, d)},
      {&d, orient2d_sign_forced_exact(a, b, c)},
  }};
  std::sort(coefficients.begin(), coefficients.end(),
            [](const CofactorCoefficient& lhs,
               const CofactorCoefficient& rhs) {
              return lhs.site->stable_id < rhs.site->stable_id;
            });
  for (const CofactorCoefficient& coefficient : coefficients) {
    if (coefficient.sign != 0) {
      return coefficient.sign;
    }
  }

  TENRYU_ASSERT(false, "sos_unresolved");
  return 1;
}

}  // namespace tenryu::mesh::tess
