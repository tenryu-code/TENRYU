#pragma once

// sqrt(a^2 + b^2) rounded to double on the device, for the emission GMRES's Givens rotations
// (sn_ld_1d_gpu.cu), whose host loop called std::hypot: after scaling by a power of two, the sum
// of the squares as a double-double (exact products), its square root corrected once (the residual
// of the rounded root is exact by fma). This is the correctly rounded value but for a true value
// within about 2^-54 of a unit in the last place of a midpoint between two doubles; std::hypot of
// glibc 2.39 is one unit in the last place off the correctly rounded value in about 0.1 % of random
// pairs (test_rounded_hypot), so the two differ there. An infinite argument gives +inf and a NaN
// (with no infinity) NaN, as std::hypot. The __d*_rn intrinsics keep every step rounded on its own
// whatever the including file's multiply-add contraction.

#include <math_constants.h>

namespace tenryu::radiation {

__device__ inline double rounded_hypot(const double a, const double b) {
  const double x = fabs(a);
  const double y = fabs(b);
  if (isinf(x) || isinf(y)) {
    return CUDART_INF;
  }
  if (isnan(x) || isnan(y)) {
    return __dadd_rn(x, y);
  }
  const double hi = (x < y) ? y : x;
  const double lo = (x < y) ? x : y;
  if (hi == 0.0) {
    return 0.0;
  }
  const int e = ilogb(hi);
  const double xs = scalbn(hi, -e);
  const double ys = scalbn(lo, -e);
  const double p = __dmul_rn(xs, xs);
  const double pe = __fma_rn(xs, xs, -p);
  const double q = __dmul_rn(ys, ys);
  const double qe = __fma_rn(ys, ys, -q);
  const double s = __dadd_rn(p, q);
  const double bb = __dsub_rn(s, p);
  const double se = __dadd_rn(__dsub_rn(p, __dsub_rn(s, bb)), __dsub_rn(q, bb));
  const double s_lo = __dadd_rn(se, __dadd_rn(pe, qe));
  const double sh = __dadd_rn(s, s_lo);
  const double sl = __dsub_rn(s_lo, __dsub_rn(sh, s));
  const double r0 = sqrt(sh);
  const double d = __dadd_rn(__fma_rn(-r0, r0, sh), sl);
  return scalbn(__dadd_rn(r0, __ddiv_rn(d, __dmul_rn(2.0, r0))), e);
}

}  // namespace tenryu::radiation
