#pragma once

// 1D spherical laser rays integrated along their characteristics
// (Laser.raytrace.integrator = "characteristic", NUMERICS 5.3.6).
//
// Same medium, events and outputs as ray_trace_1d_sph_body (the march): the
// radial profile n_hat (clipped), n_hat_raw, the smooth kappa factor and T_e
// are linear in r between radial nodes, and deposits go to hydro cells. The
// march steps dx/dxi = v, dv/dxi = -grad(n_hat)/2 with |v|^2 + n_hat = 1. In
// the spherically symmetric field the angular momentum B = |x x v| is
// conserved, and with eps = 1 - n_hat and Q(r) = r^2 eps(r) - B^2 (>= 0 on the
// path)
//   ds / |dr| = sqrt(eps) r / sqrt(Q),   dphi / |dr| = B / (r sqrt(Q)),
// so the path is integrated piece by piece between the breakpoints shared by
// all rays of a trace (radial nodes, hydro faces, the critical-adjacent split
// radius; CharacteristicPieces) and the ray's event radii: arc length,
// polar-angle increment and optical depth by adaptive Gauss-Kronrod (3, 7)
// quadrature in u = sqrt(r - r_0) when a root r_0 of Q lies at or just below
// the piece (the 1/sqrt(Q) of a turning point becomes a regular integrand),
// else in r. The inward leg ends at the first of: the turning point (largest
// root of the piecewise cubic Q below the entry), the critical radius
// (n_hat_raw = 1 - eps_crit), the march's analytic critical-layer tail
// closure, or the centre (B = 0); the outward leg retraces the pieces to the
// profile edge. Hot-electron capture fires at the first upward crossing of
// each threshold with the power after the absorption up to the crossing;
// resonance absorption fires once at the turning point with b = B; the Langdon
// factor multiplies each traversed piece's optical depth, evaluated at its
// midpoint with the vacuum-map intensity at the ray's cylindrical radius
// r |sin(phi)|.
//
// kTile lanes trace one ray, kTile pieces at a time: each lane evaluates one
// piece (entry checks, events, quadrature, Langdon factor), then lane-parallel
// scans give the polar angle and the power at each piece's entry (prefix sum
// and product), the first event ends the chunk, and the deposits, tau
// diagnostics and CBET records are summed over runs of equal cells (segmented
// sums) and written by one lane each. The events that end a leg are handled by
// all lanes alike, lane 0 writing. kTile = 1 (one thread per ray) is the
// sequential form; kTile = 32 (one warp) and 64 .. 256 (several warps of a
// block, chosen by the launcher from the ray count and the GPU) differ from it
// only in the rounding order of those sums and products.

#include <cmath>
#include <cstring>

#include "laser/ray_trace_bodies.cuh"

namespace tenryu::laser::ray_trace_bodies {

namespace characteristic_detail {

// Gauss-Kronrod (3, 7) on [-1, 1]: Kronrod nodes 0, +-kKronrodX1..3; the
// 3-point Gauss rule uses 0 and +-kKronrodX2. (Scalars: namespace-scope
// constexpr arrays are not usable in device code.)
constexpr double kKronrodX1 = 0.960491268708020283423507092629;
constexpr double kKronrodX2 = 0.774596669241483377035853079956;
constexpr double kKronrodX3 = 0.434243749346802558002071502844;
constexpr double kKronrodW0 = 0.450916538658474142345110087045;  // x = 0
constexpr double kKronrodW1 = 0.104656226026467265193823857192;
constexpr double kKronrodW2 = 0.268488089868333440728569280667;
constexpr double kKronrodW3 = 0.401397414775962222905051818618;
constexpr double kGaussW0 = 0.888888888888888888888888888889;  // x = 0
constexpr double kGaussW2 = 0.555555555555555555555555555556;  // x = +-kKronrodX2
// A panel is accepted when |K7 - G3| of arc length, polar angle and optical
// depth are each below the tolerance times the panel's share of the piece.
constexpr double kQuadRelTol = 1.0e-7;
constexpr double kPhiAbsTol = 1.0e-12;
constexpr double kTauAbsTol = 1.0e-12;
constexpr int kMaxPanelDepth = 20;
constexpr int kMaxPanels = 64;
constexpr int kRootIterations = 80;
constexpr double kPi = 3.141592653589793238462643383279502884;

// f(r) = v0 + s (r - x0) on one radial interval (linear interpolation between
// the interval's nodes, the march's interpolate_radial_field), anchored at the
// interval's first node x0. The intercept form f0 + f1 r lost |f1 r| times the
// rounding unit to cancellation: on a profile interval of 2e-17 cm at r = 0.01
// cm it put errors of 1e-3 on n_hat (2026-09-24).
struct LinearField {
  double x0 = 0.0;
  double v0 = 0.0;
  double s = 0.0;

  __device__ __forceinline__ double at(const double r) const { return v0 + s * (r - x0); }
  // The r where the field takes `value` (s != 0).
  __device__ __forceinline__ double where(const double value) const {
    return x0 + (value - v0) / s;
  }
};

__device__ __forceinline__ LinearField linear_field(const double* node_r,
                                                    const double* field,
                                                    const int j) {
  LinearField out;
  if (field == nullptr) {
    return out;
  }
  const double r0 = node_r[j];
  const double r1 = node_r[j + 1];
  out.x0 = r0;
  out.v0 = field[j];
  const double dr = r1 - r0;
  if (dr > 0.0) {
    out.s = (field[j + 1] - field[j]) / dr;
  }
  return out;
}

// A ray's conserved quantities in the 1D geometry (Mesh.geometry_1d, NUMERICS
// 5.3.6 (g)): B = |x x v| in the plane of the trace (the orbital plane of a
// sphere, the cross-section of a cylinder; 0 for a slab) and V2 = the squared
// velocity component that the gradient never changes (along the cylinder
// axis; parallel to a slab), with |v|^2 = 1 - n_hat. The path coordinate r is
// the radius (sphere, cylinder) or the height x above the slab's inner wall.
// Q(r) = r^2 (1 - n(r) - V2) - B^2 (curved) or 1 - n(r) - V2 (slab) is the
// squared radial velocity times r^2 (curved) or 1 (slab).
struct RayInvariants {
  double B = 0.0;
  double B2 = 0.0;
  double V2 = 0.0;
  double V = 0.0;
  bool planar = false;
};

// Q(r) with n linear on the interval.
__device__ __forceinline__ double q_of_r(const LinearField& n, const RayInvariants& inv,
                                         const double r) {
  const double e = 1.0 - n.at(r) - inv.V2;
  return inv.planar ? e : r * r * e - inv.B2;
}

__device__ __forceinline__ double dq_dr(const LinearField& n, const RayInvariants& inv,
                                        const double r) {
  // d/dr [r^2 (1 - n(r) - V2)] = 2 r (1 - n(r) - V2) - r^2 n'; slab: -n'
  return inv.planar ? -n.s : 2.0 * r * (1.0 - n.at(r) - inv.V2) - n.s * r * r;
}

// The root of Q in [lo, hi] given Q(lo) < 0 <= Q(hi): Q has at most one
// stationary point in r > 0 (none in a slab), so the root is unique there.
// Bisection with a safeguarded Newton step (deterministic, fixed iteration
// cap); the result has Q >= 0.
__device__ inline double bracketed_root(const LinearField& n,
                                        const RayInvariants& inv,
                                        double lo,
                                        double hi) {
  double r = hi;
  for (int it = 0; it < kRootIterations; ++it) {
    const double q = q_of_r(n, inv, r);
    if (q >= 0.0) {
      hi = r;
    } else {
      lo = r;
    }
    if (!(hi - lo > 1.0e-15 * hi)) {
      break;
    }
    const double dq = dq_dr(n, inv, r);
    double r_next = (dq != 0.0) ? r - q / dq : 0.5 * (lo + hi);
    if (!(r_next > lo && r_next < hi)) {
      r_next = 0.5 * (lo + hi);
    }
    r = r_next;
  }
  return hi;
}

// Turning point in [lo, hi] for a ray entering at hi with Q(hi) > 0: the
// largest root of Q there (cubic; linear in a slab), or -1 when Q stays
// positive.
__device__ inline double turning_point_in(const LinearField& n,
                                          const RayInvariants& inv,
                                          const double lo,
                                          const double hi) {
  if (q_of_r(n, inv, lo) < 0.0) {
    return bracketed_root(n, inv, lo, hi);
  }
  if (inv.planar) {
    return -1.0;  // Q linear: no interior minimum
  }
  // Q(lo) >= 0: an interior negative minimum still turns the ray. With
  // n = n0 + n1 r and e0 = 1 - V2 - n0, Q' = 0 at r* = 2 e0 / (3 n1) (the
  // other stationary point is r = 0); r* is a minimum only when e0 < 0, and a
  // maximum never makes Q negative. Anchored: r* = x0 + (2 (1 - V2 - v0) -
  // s x0) / (3 s).
  if (n.s != 0.0) {
    const double r_star =
        n.x0 + (2.0 * (1.0 - inv.V2 - n.v0) - n.s * n.x0) / (3.0 * n.s);
    if (r_star > lo && r_star < hi && q_of_r(n, inv, r_star) < 0.0) {
      return bracketed_root(n, inv, r_star, hi);
    }
  }
  return -1.0;
}

// Root to regularise the quadrature on [a, b] when no exact turning point is
// known: the Newton extrapolation of Q from a, when Q grows outward from a and
// the extrapolated root lies within one piece width below a; else -1 (plain r).
__device__ __forceinline__ double regularising_root(const LinearField& n,
                                                    const RayInvariants& inv,
                                                    const double a,
                                                    const double b) {
  const double q_a = q_of_r(n, inv, a);
  const double dq_a = dq_dr(n, inv, a);
  if (!(q_a > 0.0) || !(dq_a > 0.0)) {
    return -1.0;
  }
  const double d = q_a / dq_a;
  if (!(d < (b - a))) {
    return -1.0;
  }
  return ::fmax(a - d, 0.0);
}

struct QuadratureContext {
  LinearField n;
  LinearField smooth;
  RayInvariants inv;
  double eps_n = 0.0;
  double test_kappa_cm_inv = 0.0;
  bool use_u = false;  // variable u = sqrt(r - r0), else r
  double r0 = 0.0;
  // With u: Q(r0 + w) = c0 + w (c1 + w (c2 + w c3)), the cubic expanded about
  // r0, so Q / w near a root r0 has no cancellation (r^2 eps - B^2 there is a
  // difference of nearly equal numbers and can round to <= 0).
  double c0 = 0.0;
  double c1 = 0.0;
  double c2 = 0.0;
  double c3 = 0.0;
};

// Integrands of arc length, polar angle and optical depth (without the
// Langdon factor) per unit quadrature variable at x.
__device__ __forceinline__ bool evaluate_integrands(const QuadratureContext& ctx,
                                                    const double x,
                                                    double& f_ds,
                                                    double& f_phi,
                                                    double& f_tau) {
  double r = x;
  double eps = 0.0;
  double n_hat = 0.0;
  double jac_over_sqrt_q = 0.0;  // (dr/dx) / sqrt(Q)
  if (ctx.use_u) {
    const double w = x * x;
    r = ctx.r0 + w;
    n_hat = ctx.n.at(r);
    eps = 1.0 - n_hat;
    // (2u) / sqrt(Q) = 2 / sqrt(Q / w)
    const double q_over_w = ctx.c0 / w + ctx.c1 + w * (ctx.c2 + w * ctx.c3);
    if (!(w > 0.0) || !(q_over_w > 0.0)) {
      return false;
    }
    jac_over_sqrt_q = 2.0 / ::sqrt(q_over_w);
  } else {
    n_hat = ctx.n.at(r);
    eps = 1.0 - n_hat;
    const double q = q_of_r(ctx.n, ctx.inv, r);
    if (!(q > 0.0)) {
      return false;
    }
    jac_over_sqrt_q = 1.0 / ::sqrt(q);
  }
  if ((!ctx.inv.planar && !(r > 0.0)) || !(eps > 0.0)) {
    return false;
  }
  // Curved: ds/|dr| = sqrt(eps) r / sqrt(Q), dphi/|dr| = B / (r sqrt(Q)).
  // Slab: ds/|dx| = sqrt(eps) / sqrt(Q) and the lateral drift dy/|dx| =
  // V / sqrt(Q) takes the angle's place.
  if (ctx.inv.planar) {
    f_ds = ::sqrt(eps) * jac_over_sqrt_q;
    f_phi = ctx.inv.V * jac_over_sqrt_q;
  } else {
    f_ds = ::sqrt(eps) * r * jac_over_sqrt_q;
    f_phi = ctx.inv.B * jac_over_sqrt_q / r;
  }
  const double kappa = (ctx.test_kappa_cm_inv > 0.0)
                           ? ctx.test_kappa_cm_inv
                           : compute_kappa_from_smooth(ctx.smooth.at(r), n_hat, ctx.eps_n);
  f_tau = f_ds * kappa;
  return true;
}

struct PanelEstimate {
  double k_ds = 0.0;
  double k_phi = 0.0;
  double k_tau = 0.0;
  double k_taux = 0.0;  // Kronrod sum of the optical-depth integrand times x
  double g_ds = 0.0;
  double g_phi = 0.0;
  double g_tau = 0.0;
  bool ok = true;
};

__device__ inline PanelEstimate estimate_panel(const QuadratureContext& ctx,
                                               const double x0,
                                               const double x1) {
  PanelEstimate e;
  const double half = 0.5 * (x1 - x0);
  const double mid = 0.5 * (x1 + x0);
  double f_ds = 0.0;
  double f_phi = 0.0;
  double f_tau = 0.0;
  if (!evaluate_integrands(ctx, mid, f_ds, f_phi, f_tau)) {
    e.ok = false;
    return e;
  }
  e.k_ds = kKronrodW0 * f_ds;
  e.k_phi = kKronrodW0 * f_phi;
  e.k_tau = kKronrodW0 * f_tau;
  e.k_taux = kKronrodW0 * f_tau * mid;
  e.g_ds = kGaussW0 * f_ds;
  e.g_phi = kGaussW0 * f_phi;
  e.g_tau = kGaussW0 * f_tau;
  // Symmetric node pair +-x (half-width units): sums of the three integrands.
  const auto pair_sums = [&](const double x, double& s_ds, double& s_phi,
                             double& s_tau, double& s_taux) -> bool {
    const double dx = half * x;
    double m_ds = 0.0;
    double m_phi = 0.0;
    double m_tau = 0.0;
    double p_ds = 0.0;
    double p_phi = 0.0;
    double p_tau = 0.0;
    if (!evaluate_integrands(ctx, mid - dx, m_ds, m_phi, m_tau) ||
        !evaluate_integrands(ctx, mid + dx, p_ds, p_phi, p_tau)) {
      return false;
    }
    s_ds = m_ds + p_ds;
    s_phi = m_phi + p_phi;
    s_tau = m_tau + p_tau;
    s_taux = m_tau * (mid - dx) + p_tau * (mid + dx);
    return true;
  };
  double s_ds = 0.0;
  double s_phi = 0.0;
  double s_tau = 0.0;
  double s_taux = 0.0;
  if (!pair_sums(kKronrodX1, s_ds, s_phi, s_tau, s_taux)) {
    e.ok = false;
    return e;
  }
  e.k_ds += kKronrodW1 * s_ds;
  e.k_phi += kKronrodW1 * s_phi;
  e.k_tau += kKronrodW1 * s_tau;
  e.k_taux += kKronrodW1 * s_taux;
  if (!pair_sums(kKronrodX2, s_ds, s_phi, s_tau, s_taux)) {
    e.ok = false;
    return e;
  }
  e.k_ds += kKronrodW2 * s_ds;
  e.k_phi += kKronrodW2 * s_phi;
  e.k_tau += kKronrodW2 * s_tau;
  e.k_taux += kKronrodW2 * s_taux;
  e.g_ds += kGaussW2 * s_ds;
  e.g_phi += kGaussW2 * s_phi;
  e.g_tau += kGaussW2 * s_tau;
  if (!pair_sums(kKronrodX3, s_ds, s_phi, s_tau, s_taux)) {
    e.ok = false;
    return e;
  }
  e.k_ds += kKronrodW3 * s_ds;
  e.k_phi += kKronrodW3 * s_phi;
  e.k_tau += kKronrodW3 * s_tau;
  e.k_taux += kKronrodW3 * s_taux;
  e.k_ds *= half;
  e.k_taux *= half;
  e.k_phi *= half;
  e.k_tau *= half;
  e.g_ds *= half;
  e.g_phi *= half;
  e.g_tau *= half;
  e.ok = ::isfinite(e.k_ds) && ::isfinite(e.k_phi) && ::isfinite(e.k_tau) &&
         ::isfinite(e.g_ds) && ::isfinite(e.g_phi) && ::isfinite(e.g_tau);
  return e;
}

// Langdon factor inputs (kPhysExt with langdon_model != 0). The factor
// multiplies each traversed piece's optical depth, evaluated at the piece
// midpoint: radius, polar angle and T_e (linear on the radial interval).
struct LangdonContext {
  bool active = false;
  int model = 0;
  double zcoll = 0.0;
  // Multi-material decks: the collision charge per radial node (linear on
  // the interval, as T_e), else zcoll for every piece (2026-09-24).
  const double* zcoll_radial = nullptr;
  double I0_wcm2 = 0.0;
  double w_cm = 0.0;
  int profile_kind = 0;
  double sg_two_m = 0.0;
  double te_min_eV = 0.0;
  double lambda_cm = 0.0;
};

__device__ __forceinline__ double langdon_factor_at(const LangdonContext& langdon,
                                                    const LinearField& te,
                                                    const double r,
                                                    const double phi,
                                                    const LinearField& zcoll_field = {}) {
  if (!langdon.active) {
    return 1.0;
  }
  const double I_vac =
      vacuum_map_intensity(langdon.I0_wcm2, langdon.w_cm, ::fabs(r * ::sin(phi)),
                           langdon.profile_kind, langdon.sg_two_m);
  const double zcoll =
      (langdon.zcoll_radial != nullptr) ? zcoll_field.at(r) : langdon.zcoll;
  return compute_langdon_factor(langdon.model, zcoll, I_vac, langdon.lambda_cm,
                                te.at(r), langdon.te_min_eV);
}

struct PieceIntegrals {
  double ds = 0.0;    // arc length [cm]
  double dphi = 0.0;  // polar-angle increment magnitude [rad]
  double tau0 = 0.0;  // optical depth without the Langdon factor
  // Where the piece absorbs: the optical-depth-weighted mean of the
  // quadrature variable, as a radius and as the fraction of [lo, hi] in that
  // variable (the polar angle is interpolated linearly in it).
  double r_centroid = 0.0;
  double frac_centroid = 0.5;
  int panels = 0;
  bool unresolved = false;  // a panel was accepted at the depth or count cap
  bool valid = true;
};

// Integrals over [a, b] (a < b). r0 >= 0 selects the variable
// u = sqrt(r - r0) (r0 <= a), else r. Panels are processed depth-first,
// lower half first (a fixed order: the sums are deterministic).
__device__ inline PieceIntegrals integrate_piece(const double a,
                                                 const double b,
                                                 const double r0,
                                                 const LinearField& n,
                                                 const LinearField& smooth,
                                                 const RayInvariants& inv,
                                                 const double eps_n,
                                                 const double test_kappa_cm_inv) {
  PieceIntegrals out;
  if (!(b > a)) {
    return out;
  }
  QuadratureContext ctx;
  ctx.n = n;
  ctx.smooth = smooth;
  ctx.inv = inv;
  ctx.eps_n = eps_n;
  ctx.test_kappa_cm_inv = test_kappa_cm_inv;
  ctx.use_u = r0 >= 0.0 && r0 <= a;
  ctx.r0 = ctx.use_u ? r0 : 0.0;
  if (ctx.use_u) {
    // A piece that starts at r0 starts at a root of Q (the turning point, or
    // the centre with B = 0): Q(r0) there is the root finder's rounding
    // residual, which would put an artificial layer of width sqrt(Q(r0)/Q'(r0))
    // at u = 0 and defeat the regularisation; it is taken as 0.
    // (r0 + w)^2 (e0 - s w) - B^2 with e0 = 1 - n(r0) - V2; slab: e0 - s w.
    const double e0 = 1.0 - n.at(r0) - inv.V2;
    ctx.c0 = (a == r0) ? 0.0 : q_of_r(n, inv, r0);
    if (inv.planar) {
      ctx.c1 = -n.s;
      ctx.c2 = 0.0;
      ctx.c3 = 0.0;
    } else {
      ctx.c1 = 2.0 * r0 * e0 - n.s * r0 * r0;
      ctx.c2 = e0 - 2.0 * n.s * r0;
      ctx.c3 = -n.s;
    }
  }
  const double x_lo = ctx.use_u ? ::sqrt(::fmax(a - r0, 0.0)) : a;
  const double x_hi = ctx.use_u ? ::sqrt(::fmax(b - r0, 0.0)) : b;
  const double width = x_hi - x_lo;
  if (!(width > 0.0)) {
    return out;
  }

  double stack_x0[kMaxPanelDepth + 1];
  double stack_x1[kMaxPanelDepth + 1];
  int stack_depth[kMaxPanelDepth + 1];
  stack_x0[0] = x_lo;
  stack_x1[0] = x_hi;
  stack_depth[0] = 0;
  int top = 1;
  double tol_ds = 0.0;
  double tol_phi = 0.0;
  double tol_tau = 0.0;
  double taux = 0.0;
  while (top > 0) {
    --top;
    const double x0 = stack_x0[top];
    const double x1 = stack_x1[top];
    const int depth = stack_depth[top];
    const PanelEstimate e = estimate_panel(ctx, x0, x1);
    ++out.panels;
    if (!e.ok) {
      out.valid = false;
      return out;
    }
    if (out.panels == 1) {
      tol_ds = kQuadRelTol * ::fabs(e.k_ds);
      tol_phi = ::fmax(kQuadRelTol * ::fabs(e.k_phi), kPhiAbsTol);
      tol_tau = ::fmax(kQuadRelTol * ::fabs(e.k_tau), kTauAbsTol);
    }
    const double share = (x1 - x0) / width;
    const bool converged = ::fabs(e.k_ds - e.g_ds) <= tol_ds * share &&
                           ::fabs(e.k_phi - e.g_phi) <= tol_phi * share &&
                           ::fabs(e.k_tau - e.g_tau) <= tol_tau * share;
    if (!converged && depth < kMaxPanelDepth && out.panels + top + 2 <= kMaxPanels) {
      const double xm = 0.5 * (x0 + x1);
      stack_x0[top] = xm;
      stack_x1[top] = x1;
      stack_depth[top] = depth + 1;
      ++top;
      stack_x0[top] = x0;
      stack_x1[top] = xm;
      stack_depth[top] = depth + 1;
      ++top;
      continue;
    }
    if (!converged) {
      out.unresolved = true;
    }
    out.ds += e.k_ds;
    out.dphi += e.k_phi;
    out.tau0 += e.k_tau;
    taux += e.k_taux;
  }
  if (!(::isfinite(out.ds) && ::isfinite(out.dphi) && ::isfinite(out.tau0))) {
    out.valid = false;
  }
  double x_c = 0.5 * (x_lo + x_hi);
  if (out.tau0 > 0.0 && ::isfinite(taux)) {
    x_c = ::fmin(::fmax(taux / out.tau0, x_lo), x_hi);
  }
  out.r_centroid = ctx.use_u ? ctx.r0 + x_c * x_c : x_c;
  out.frac_centroid = (x_c - x_lo) / width;
  return out;
}

// Deposit cell of a piece lying in hydro cell c (cells split at the
// critical-adjacent radius as accumulate_deposit_1d does).
__device__ __forceinline__ int piece_cell(const int c,
                                          const double r_mid,
                                          const double* hydro_r_edges,
                                          const int allowed_supercritical_cell,
                                          const int critical_adjacent_subcritical_cell,
                                          const double critical_adjacent_split_r) {
  if (critical_adjacent_subcritical_cell == c && allowed_supercritical_cell == c - 1 &&
      critical_adjacent_split_r > hydro_r_edges[c] &&
      critical_adjacent_split_r < hydro_r_edges[c + 1] && r_mid < critical_adjacent_split_r) {
    return allowed_supercritical_cell;
  }
  return c;
}

// Number of v[i] < x (strict) and <= x in an ascending array.
__device__ __forceinline__ int count_less(const double* v, const int n, const double x) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (v[mid] < x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ __forceinline__ int count_less_equal(const double* v, const int n, const double x) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (v[mid] <= x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

}  // namespace characteristic_detail

// Breakpoints of the characteristic pieces, shared by all rays of one trace
// (NUMERICS 5.3.6 (b)): the radial nodes, the hydro faces strictly inside
// (r_0, r_max) (r_0 = 0 for a sphere or cylinder, a slab's inner wall) and the
// critical-adjacent split radius when it lies there,
// ascending (equal values ordered node, face, split, which leaves a zero-length
// piece between them); piece k = [r[k], r[k+1]] with its radial interval and
// deposit cell.
struct CharacteristicPieces {
  const double* r = nullptr;
  const int* interval = nullptr;
  const int* cell = nullptr;
  const int* count = nullptr;  // number of pieces (device scalar)
};

// Pieces are at most (radial nodes) + (hydro faces) + (split) - 1.
__host__ __device__ inline int characteristic_piece_capacity(const int n_radial_nodes,
                                                             const int n_hydro_cells) {
  return n_radial_nodes + n_hydro_cells + 1;
}

// Item t of the breakpoint scatter (t in [0, n_nodes + n_cells + 2)): radial
// node t, then hydro face t - n_nodes, then the split radius; each value goes
// to its rank in the merged order. The last item writes the piece count.
__device__ inline void characteristic_breakpoint_body(const int t,
                                                      const double* __restrict__ node_r,
                                                      const int n_nodes,
                                                      const double* __restrict__ edges,
                                                      const int n_cells,
                                                      const double split_r,
                                                      double* __restrict__ out_r,
                                                      int* __restrict__ out_count) {
  using characteristic_detail::count_less;
  using characteristic_detail::count_less_equal;
  const double r_max = node_r[n_nodes - 1];
  const int n_edges = n_cells + 1;
  const int e_lo = count_less_equal(edges, n_edges, node_r[0]);  // first face > node_r[0]
  const int e_hi = count_less(edges, n_edges, r_max);      // first face >= r_max
  const bool split_in = split_r > node_r[0] && split_r < r_max;
  const auto inside_faces_less = [&](const double v) {
    return ::min(::max(count_less(edges, n_edges, v), e_lo), e_hi) - e_lo;
  };
  const auto inside_faces_less_equal = [&](const double v) {
    return ::min(::max(count_less_equal(edges, n_edges, v), e_lo), e_hi) - e_lo;
  };
  if (t < n_nodes) {
    const double v = node_r[t];
    out_r[t + inside_faces_less(v) + ((split_in && split_r < v) ? 1 : 0)] = v;
  } else if (t < n_nodes + n_edges) {
    const int e = t - n_nodes;
    if (e < e_lo || e >= e_hi) {
      return;
    }
    const double v = edges[e];
    out_r[count_less_equal(node_r, n_nodes, v) + (e - e_lo) +
          ((split_in && split_r < v) ? 1 : 0)] = v;
  } else if (t == n_nodes + n_edges) {
    if (split_in) {
      out_r[count_less_equal(node_r, n_nodes, split_r) + inside_faces_less_equal(split_r)] =
          split_r;
    }
    *out_count = n_nodes + ::max(e_hi - e_lo, 0) + (split_in ? 1 : 0) - 1;
  }
}

// Radial interval and deposit cell of piece k (k < count).
__device__ inline void characteristic_piece_info_body(const int k,
                                                      const double* __restrict__ piece_r,
                                                      const int* __restrict__ count,
                                                      const double* __restrict__ node_r,
                                                      const int n_nodes,
                                                      const double* __restrict__ edges,
                                                      const int n_cells,
                                                      const int allowed_supercritical_cell,
                                                      const int critical_adjacent_subcritical_cell,
                                                      const double critical_adjacent_split_r,
                                                      int* __restrict__ out_interval,
                                                      int* __restrict__ out_cell) {
  using characteristic_detail::count_less_equal;
  if (k >= *count) {
    return;
  }
  const double lo = piece_r[k];
  const double hi = piece_r[k + 1];
  const double mid = 0.5 * (lo + hi);
  const int j = ::min(::max(count_less_equal(node_r, n_nodes, lo) - 1, 0), n_nodes - 2);
  int c = 0;
  if (mid >= edges[n_cells]) {
    c = n_cells - 1;
  } else if (mid > edges[0]) {
    c = ::min(::max(count_less_equal(edges, n_cells + 1, mid) - 1, 0), n_cells - 1);
  }
  out_interval[k] = j;
  out_cell[k] = characteristic_detail::piece_cell(c, mid, edges, allowed_supercritical_cell,
                                                  critical_adjacent_subcritical_cell,
                                                  critical_adjacent_split_r);
}

namespace characteristic_detail {

// Inward events of a piece: the ray turns at the top (tangent, Q <= 0 there),
// at the largest root of Q inside, reaches the critical radius, or passes the
// centre.
enum PieceEvent : int {
  kEventNone = 0,
  kEventTangent = 1,
  kEventTurn = 2,
  kEventCritical = 3,
  kEventCentre = 4,
};

// One lane's prefetched piece. Everything that does not depend on the ray's
// power is computed here, in parallel over the lanes: geometry, the step-entry
// checks at the piece's entry radius (critical termination, the march's tail
// closure trigger), the inward event, the quadrature over the traversed part
// [r_stop, b] (inward) or [a, b], the Langdon factor at the traversed part's
// midpoint and the absorbed fraction -expm1(-tau).
struct PrefetchedPiece {
  double a = 0.0;
  double b = 0.0;
  double r_stop = 0.0;
  double r0 = -1.0;
  double ds = 0.0;
  double dphi = 0.0;
  double tau0 = 0.0;
  double r_centroid = 0.0;
  double frac_centroid = 0.5;  // of [lo, hi] in the quadrature variable
  double tau = 0.0;
  double absorbed_fraction = 0.0;
  int j = 0;
  int cell = 0;
  int event = kEventNone;
  int panels = 0;
  int valid = 1;
  int unresolved = 0;      // quadrature accepted at the panel cap
  int entry_invalid = 0;   // non-finite n_hat at the entry
  int entry_critical = 0;  // n_hat >= 1 - eps_crit or n_hat_raw >= the critical value there
  int entry_trigger = 0;   // the march's tail closure fires at the entry
};

__device__ inline PrefetchedPiece prefetch_piece(const CharacteristicPieces& pieces,
                                                 const int k,
                                                 const bool inward,
                                                 const bool first_lane,
                                                 const double r_cur,
                                                 const double r_turn,
                                                 const double* __restrict__ radial_node_r,
                                                 const double* __restrict__ radial_n_hat,
                                                 const double* __restrict__ radial_n_hat_raw,
                                                 const double* __restrict__ radial_smooth_kappa,
                                                 const double* __restrict__ radial_dn_dr,
                                                 const RayInvariants& inv,
                                                 const double eps_crit,
                                                 const double eps_n,
                                                 const double test_kappa_cm_inv,
                                                 const bool check_trigger,
                                                 const bool check_entry_critical = true) {
  PrefetchedPiece out;
  out.j = pieces.interval[k];
  out.cell = pieces.cell[k];
  out.a = (!inward && first_lane) ? r_cur : pieces.r[k];
  out.b = (inward && first_lane) ? r_cur : pieces.r[k + 1];
  const LinearField nf = linear_field(radial_node_r, radial_n_hat, out.j);
  const LinearField nf_raw = linear_field(radial_node_r, radial_n_hat_raw, out.j);
  const LinearField sf = linear_field(radial_node_r, radial_smooth_kappa, out.j);
  const double nh_crit = 1.0 - eps_crit;

  // Step-entry checks at the entry radius.
  const double r_in = inward ? out.b : out.a;
  const double nh_in = nf.at(r_in);
  const double nh_in_raw = nf_raw.at(r_in);
  if (!::isfinite(nh_in) || !::isfinite(nh_in_raw)) {
    out.entry_invalid = 1;
    return out;
  }
  if (check_entry_critical && (nh_in >= 1.0 - eps_crit || nh_in_raw >= nh_crit)) {
    out.entry_critical = 1;
    return out;
  }
  const double q_in = q_of_r(nf, inv, r_in);
  if (check_trigger && nh_in_raw >= kCritLayerHandoffNhatRaw) {
    const double dn_dr = radial_dn_dr[out.j];
    const double w_interval = radial_node_r[out.j + 1] - radial_node_r[out.j];
    const RadialInterval c_in{
        out.j, (w_interval > 0.0)
                   ? clamp_unit_interval((r_in - radial_node_r[out.j]) / w_interval)
                   : 0.0};
    const double kappa_in = (test_kappa_cm_inv > 0.0)
                                ? test_kappa_cm_inv
                                : compute_kappa_from_smooth(sf.at(r_in), nh_in, eps_n);
    const TailClosureEntryData entry = compute_tail_closure_entry_1d(
        c_in, radial_smooth_kappa, radial_n_hat_raw, test_kappa_cm_inv, nh_in, kappa_in);
    const double v_radial = (inward ? -1.0 : 1.0) * ::sqrt(::fmax(q_in, 0.0)) /
                            (inv.planar ? 1.0 : ::fmax(r_in, 1.0e-300));
    const double g_mag = ::fabs(dn_dr);
    if (::isfinite(g_mag) && g_mag > 0.0 &&
        should_trigger_tail_closure(nh_in_raw, entry.A_entry, g_mag, dn_dr * v_radial,
                                    ::fmax(1.0 - nh_in, 0.0))) {
      out.entry_trigger = 1;
      return out;
    }
  }

  double lo = out.a;
  if (inward) {
    out.r_stop = out.a;
    if (!(q_in > 0.0)) {
      out.r_stop = out.b;
      out.event = kEventTangent;
    } else {
      const double r_t = turning_point_in(nf, inv, out.a, out.b);
      if (r_t >= 0.0 && r_t < out.b) {
        out.r_stop = r_t;
        out.event = kEventTurn;
      }
      // n_hat_raw is linear on the piece and below the critical value at the
      // entry: it reaches it before r_stop when it does at r_stop.
      if (nf_raw.at(out.r_stop) >= nh_crit && nf_raw.s != 0.0) {
        out.r_stop = ::fmin(::fmax(nf_raw.where(nh_crit), out.r_stop), out.b);
        out.event = kEventCritical;
      }
      // The centre (r = 0) of a sphere or cylinder; a slab's inner wall
      // (the profile's first node).
      if (out.event == kEventNone && !(out.a > radial_node_r[0])) {
        out.event = kEventCentre;
      }
    }
    lo = out.r_stop;
    if (out.event == kEventTurn) {
      out.r0 = out.r_stop;
    } else if (out.b > lo) {
      out.r0 = regularising_root(nf, inv, lo, out.b);
    }
  } else {
    out.r_stop = out.b;
    // The substitution centre: the ray's turning radius, unless this
    // interval's own Q has a near-root just below a closer to a (the ray
    // left its turning point in a flatter interval below and Q is still near
    // zero at a; Q then grows steeply in this interval and 1 / sqrt(Q) is
    // near-singular at a — critical-layer pieces of GXII hit the panel cap
    // with the turning radius 6e-6 cm below a and the near-root 1.6e-10 cm
    // below it).
    out.r0 = (r_turn >= 0.0) ? r_turn : -1.0;
    const double r_near = (out.b > out.a) ? regularising_root(nf, inv, out.a, out.b) : -1.0;
    if (r_near >= 0.0 && (out.r0 < 0.0 || r_near > out.r0)) {
      out.r0 = r_near;
    }
  }
  const PieceIntegrals pi =
      integrate_piece(lo, out.b, out.r0, nf, sf, inv, eps_n, test_kappa_cm_inv);
  out.ds = pi.ds;
  out.dphi = pi.dphi;
  out.tau0 = pi.tau0;
  out.r_centroid = pi.r_centroid;
  out.frac_centroid = pi.frac_centroid;
  out.panels = pi.panels;
  out.valid = pi.valid ? 1 : 0;
  out.unresolved = pi.unresolved ? 1 : 0;
  return out;
}

// ---- Tile primitives: kTile lanes trace one ray ----
//
// kTile = 1: one thread (each primitive reduces to the identity); kTile = 32:
// one warp (shuffles and ballots); kTile = 64 .. 256: several warps of one
// block, which exchange their warp results through shared memory and meet at
// a named barrier of their own (id 1 + the tile's index in the block, so the
// tiles of a block do not wait for each other). Every primitive is collective:
// all lanes of the tile call it, which the trace guarantees (the lanes of a
// ray hold identical copies of its state and take the same branches). The
// sums and products combine each warp's scan with the totals of the warps
// before it in warp order: deterministic, with a rounding order that depends
// on kTile.

constexpr int kTileMaxWarpsPerBlock = 32;
constexpr int kTileMaxTilesPerBlock = 8;

// Shared scratch of the multi-warp primitives (one instance per block): per
// warp two 8-byte slots and two ints, per tile one 8-byte broadcast slot.
struct TileScratch {
  unsigned long long w0[kTileMaxWarpsPerBlock];
  unsigned long long w1[kTileMaxWarpsPerBlock];
  int i0[kTileMaxWarpsPerBlock];
  int i1[kTileMaxWarpsPerBlock];
  unsigned long long bcast[kTileMaxTilesPerBlock];
};

__device__ __forceinline__ TileScratch& tile_scratch() {
  __shared__ TileScratch scratch;
  return scratch;
}

template <typename T>
__device__ __forceinline__ unsigned long long tile_to_bits(const T value) {
  static_assert(sizeof(T) <= sizeof(unsigned long long), "tile slots hold 8 bytes");
  unsigned long long bits = 0ULL;
  memcpy(&bits, &value, sizeof(T));
  return bits;
}

template <typename T>
__device__ __forceinline__ T tile_from_bits(const unsigned long long bits) {
  T value;
  memcpy(&value, &bits, sizeof(T));
  return value;
}

template <int kTile>
__device__ __forceinline__ int tile_lane() {
  if constexpr (kTile == 1) {
    return 0;
  } else {
    return static_cast<int>(threadIdx.x) & (kTile - 1);
  }
}

// Warps of a multi-warp tile: the block warp index of its first warp and this
// lane's warp within the tile.
template <int kTile>
__device__ __forceinline__ int tile_first_warp() {
  return (static_cast<int>(threadIdx.x) / kTile) * (kTile / 32);
}

template <int kTile>
__device__ __forceinline__ int tile_warp() {
  return (static_cast<int>(threadIdx.x) & (kTile - 1)) >> 5;
}

__device__ __forceinline__ int tile_warp_lane() {
  return static_cast<int>(threadIdx.x) & 31;
}

template <int kTile>
__device__ __forceinline__ void tile_sync() {
  if constexpr (kTile == 1) {
  } else if constexpr (kTile == 32) {
    __syncwarp();
  } else {
    static_assert(kTile % 32 == 0 && kTile / 32 <= kTileMaxWarpsPerBlock,
                  "a multi-warp tile is a whole number of warps");
    const unsigned barrier = 1u + threadIdx.x / static_cast<unsigned>(kTile);
    asm volatile("bar.sync %0, %1;" : : "r"(barrier), "n"(kTile) : "memory");
  }
}

template <int kTile, typename T>
__device__ __forceinline__ T tile_broadcast(const T value, const int src_lane) {
  if constexpr (kTile == 1) {
    (void)src_lane;
    return value;
  } else if constexpr (kTile == 32) {
    return __shfl_sync(0xffffffffu, value, src_lane, kTile);
  } else {
    TileScratch& sh = tile_scratch();
    const int tile = static_cast<int>(threadIdx.x) / kTile;
    if (tile_lane<kTile>() == src_lane) {
      sh.bcast[tile] = tile_to_bits(value);
    }
    tile_sync<kTile>();
    const T out = tile_from_bits<T>(sh.bcast[tile]);
    tile_sync<kTile>();
    return out;
  }
}

// Lowest lane of the tile with pred set, or kTile when none has it.
template <int kTile>
__device__ __forceinline__ int tile_first_true(const bool pred) {
  if constexpr (kTile == 1) {
    return pred ? 0 : 1;
  } else if constexpr (kTile == 32) {
    const unsigned mask = __ballot_sync(0xffffffffu, pred);
    return (mask == 0u) ? kTile : (__ffs(static_cast<int>(mask)) - 1);
  } else {
    TileScratch& sh = tile_scratch();
    const unsigned mask = __ballot_sync(0xffffffffu, pred);
    const int w0 = tile_first_warp<kTile>();
    if (tile_warp_lane() == 0) {
      sh.i0[w0 + tile_warp<kTile>()] = (mask == 0u) ? 32 : (__ffs(static_cast<int>(mask)) - 1);
    }
    tile_sync<kTile>();
    int first = kTile;
    for (int w = 0; w < kTile / 32; ++w) {
      const int v = sh.i0[w0 + w];
      if (v < 32) {
        first = 32 * w + v;
        break;
      }
    }
    tile_sync<kTile>();
    return first;
  }
}

// Highest lane of the tile with pred set, or -1 when none has it.
template <int kTile>
__device__ __forceinline__ int tile_last_true(const bool pred) {
  if constexpr (kTile == 1) {
    return pred ? 0 : -1;
  } else if constexpr (kTile == 32) {
    const unsigned mask = __ballot_sync(0xffffffffu, pred);
    return 31 - __clz(static_cast<int>(mask));
  } else {
    TileScratch& sh = tile_scratch();
    const unsigned mask = __ballot_sync(0xffffffffu, pred);
    const int w0 = tile_first_warp<kTile>();
    if (tile_warp_lane() == 0) {
      sh.i0[w0 + tile_warp<kTile>()] = 31 - __clz(static_cast<int>(mask));
    }
    tile_sync<kTile>();
    int last = -1;
    for (int w = kTile / 32 - 1; w >= 0; --w) {
      const int v = sh.i0[w0 + w];
      if (v >= 0) {
        last = 32 * w + v;
        break;
      }
    }
    tile_sync<kTile>();
    return last;
  }
}

// Number of lanes of the tile with pred set (inclusive: those up to this lane).
template <int kTile>
__device__ __forceinline__ int tile_count(const bool pred, const bool inclusive_of_lane = false) {
  if constexpr (kTile == 1) {
    return pred ? 1 : 0;
  } else if constexpr (kTile == 32) {
    const unsigned mask = __ballot_sync(0xffffffffu, pred);
    if (!inclusive_of_lane) {
      return __popc(mask);
    }
    const int lane = tile_warp_lane();
    const unsigned upto = (lane >= 31) ? 0xffffffffu : ((2u << lane) - 1u);
    return __popc(mask & upto);
  } else {
    TileScratch& sh = tile_scratch();
    const unsigned mask = __ballot_sync(0xffffffffu, pred);
    const int w0 = tile_first_warp<kTile>();
    const int my_warp = tile_warp<kTile>();
    if (tile_warp_lane() == 0) {
      sh.i1[w0 + my_warp] = __popc(mask);
    }
    tile_sync<kTile>();
    int count = 0;
    const int n_warps = inclusive_of_lane ? my_warp : kTile / 32;
    for (int w = 0; w < n_warps; ++w) {
      count += sh.i1[w0 + w];
    }
    tile_sync<kTile>();
    if (inclusive_of_lane) {
      const int lane = tile_warp_lane();
      const unsigned upto = (lane >= 31) ? 0xffffffffu : ((2u << lane) - 1u);
      count += __popc(mask & upto);
    }
    return count;
  }
}

template <int kTile>
__device__ __forceinline__ bool tile_any(const bool pred) {
  return tile_first_true<kTile>(pred) < kTile;
}

// The value of the next (previous) lane, fill at the tile's last (first) lane.
template <int kTile, typename T>
__device__ __forceinline__ T tile_next(const T value, const T fill, const int lane) {
  if constexpr (kTile == 1) {
    (void)value;
    (void)lane;
    return fill;
  } else if constexpr (kTile == 32) {
    const T v = __shfl_down_sync(0xffffffffu, value, 1, kTile);
    return (lane == kTile - 1) ? fill : v;
  } else {
    TileScratch& sh = tile_scratch();
    T v = __shfl_down_sync(0xffffffffu, value, 1, 32);
    const int warp = tile_first_warp<kTile>() + tile_warp<kTile>();
    if (tile_warp_lane() == 0) {
      sh.w0[warp] = tile_to_bits(value);
    }
    tile_sync<kTile>();
    if (tile_warp_lane() == 31 && lane != kTile - 1) {
      v = tile_from_bits<T>(sh.w0[warp + 1]);
    }
    tile_sync<kTile>();
    return (lane == kTile - 1) ? fill : v;
  }
}

template <int kTile, typename T>
__device__ __forceinline__ T tile_previous(const T value, const T fill, const int lane) {
  if constexpr (kTile == 1) {
    (void)value;
    (void)lane;
    return fill;
  } else if constexpr (kTile == 32) {
    const T v = __shfl_up_sync(0xffffffffu, value, 1, kTile);
    return (lane == 0) ? fill : v;
  } else {
    TileScratch& sh = tile_scratch();
    T v = __shfl_up_sync(0xffffffffu, value, 1, 32);
    const int warp = tile_first_warp<kTile>() + tile_warp<kTile>();
    if (tile_warp_lane() == 31) {
      sh.w0[warp] = tile_to_bits(value);
    }
    tile_sync<kTile>();
    if (tile_warp_lane() == 0 && lane != 0) {
      v = tile_from_bits<T>(sh.w0[warp - 1]);
    }
    tile_sync<kTile>();
    return (lane == 0) ? fill : v;
  }
}

// Inclusive scan within a warp (Hillis-Steele over the 32 lanes).
template <typename T, typename Op>
__device__ __forceinline__ T warp_inclusive_scan(const T value, const Op op) {
  T inclusive = value;
  const int lane = tile_warp_lane();
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const T y = __shfl_up_sync(0xffffffffu, inclusive, d, 32);
    if (lane >= d) {
      inclusive = op(inclusive, y);
    }
  }
  return inclusive;
}

// Inclusive scan over a multi-warp tile: each warp's scan combined with the
// totals of the tile's warps before it, in warp order.
template <int kTile, typename T, typename Op>
__device__ __forceinline__ T tile_inclusive_scan(const T value, const T identity, const Op op) {
  TileScratch& sh = tile_scratch();
  const T inclusive = warp_inclusive_scan(value, op);
  const int w0 = tile_first_warp<kTile>();
  const int my_warp = tile_warp<kTile>();
  if (tile_warp_lane() == 31) {
    sh.w1[w0 + my_warp] = tile_to_bits(inclusive);
  }
  tile_sync<kTile>();
  T prefix = identity;
  for (int w = 0; w < my_warp; ++w) {
    prefix = op(prefix, tile_from_bits<T>(sh.w1[w0 + w]));
  }
  tile_sync<kTile>();
  return (my_warp == 0) ? inclusive : op(prefix, inclusive);
}

// Exclusive prefix sum over the lanes (lane 0 gets 0).
template <int kTile, typename T>
__device__ __forceinline__ T tile_exclusive_sum(const T value, const int lane) {
  if constexpr (kTile == 1) {
    (void)value;
    (void)lane;
    return T(0);
  } else if constexpr (kTile == 32) {
    T inclusive = value;
#pragma unroll
    for (int d = 1; d < kTile; d <<= 1) {
      const T y = __shfl_up_sync(0xffffffffu, inclusive, d, kTile);
      if (lane >= d) {
        inclusive += y;
      }
    }
    return tile_previous<kTile>(inclusive, T(0), lane);
  } else {
    const T inclusive =
        tile_inclusive_scan<kTile>(value, T(0), [](const T a, const T b) { return a + b; });
    return tile_previous<kTile>(inclusive, T(0), lane);
  }
}

// Exclusive prefix product over the lanes (lane 0 gets 1).
template <int kTile>
__device__ __forceinline__ double tile_exclusive_product(const double value, const int lane) {
  if constexpr (kTile == 1) {
    (void)value;
    (void)lane;
    return 1.0;
  } else if constexpr (kTile == 32) {
    double inclusive = value;
#pragma unroll
    for (int d = 1; d < kTile; d <<= 1) {
      const double y = __shfl_up_sync(0xffffffffu, inclusive, d, kTile);
      if (lane >= d) {
        inclusive *= y;
      }
    }
    return tile_previous<kTile>(inclusive, 1.0, lane);
  } else {
    const double inclusive = tile_inclusive_scan<kTile>(
        value, 1.0, [](const double a, const double b) { return a * b; });
    return tile_previous<kTile>(inclusive, 1.0, lane);
  }
}

// Inclusive sum within runs of equal keys (the keys of a run are contiguous
// lanes: deposit cells and radial intervals are monotone along the path, and
// lanes outside every run carry keys of their own). A multi-warp tile adds to
// the first run of each warp the part of that run in the warps before it
// (their last runs' sums, in warp order, back to the warp where the run
// starts).
template <int kTile>
__device__ __forceinline__ double tile_segmented_inclusive_sum(const double value,
                                                              const int key,
                                                              const int lane) {
  if constexpr (kTile == 1) {
    (void)key;
    (void)lane;
    return value;
  } else {
    double inclusive = value;
    const int width = (kTile == 32) ? kTile : 32;
    const int wl = (kTile == 32) ? lane : tile_warp_lane();
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const double y = __shfl_up_sync(0xffffffffu, inclusive, d, width);
      const int k = __shfl_up_sync(0xffffffffu, key, d, width);
      if (wl >= d && k == key) {
        inclusive += y;
      }
    }
    if constexpr (kTile == 32) {
      return inclusive;
    } else {
      TileScratch& sh = tile_scratch();
      const int first_key = __shfl_sync(0xffffffffu, key, 0, 32);
      const int last_key = __shfl_sync(0xffffffffu, key, 31, 32);
      const double last_sum = __shfl_sync(0xffffffffu, inclusive, 31, 32);
      const int w0 = tile_first_warp<kTile>();
      const int my_warp = tile_warp<kTile>();
      if (wl == 0) {
        sh.i0[w0 + my_warp] = first_key;
        sh.i1[w0 + my_warp] = last_key;
        sh.w1[w0 + my_warp] = tile_to_bits(last_sum);
      }
      tile_sync<kTile>();
      double carry = 0.0;
      bool has_carry = false;
      if (key == first_key && my_warp > 0 && sh.i1[w0 + my_warp - 1] == key) {
        // The run continues from earlier warps: the warps it spans entirely,
        // back to the one where it starts.
        int w_start = my_warp - 1;
        while (w_start > 0 && sh.i0[w0 + w_start] == key && sh.i1[w0 + w_start - 1] == key) {
          --w_start;
        }
        for (int w = w_start; w < my_warp; ++w) {
          const double part = tile_from_bits<double>(sh.w1[w0 + w]);
          carry = has_carry ? carry + part : part;
          has_carry = true;
        }
      }
      tile_sync<kTile>();
      return has_carry ? carry + inclusive : inclusive;
    }
  }
}

// Lane-event codes of a chunk, in the march's order at a step: the step cap
// and the entry checks come before the piece, the piece's own events after.
enum LaneEvent : int {
  kLaneNone = 0,
  kLaneGuard = 1,          // max_steps reached at the entry
  kLaneInvalid = 2,        // non-finite entry state or quadrature
  kLaneEntryCritical = 3,  // critical at the entry
  kLaneTrigger = 4,        // the march's tail closure fires at the entry
  kLaneCritical = 5,       // the critical radius inside the piece
  kLaneTurn = 6,           // turning point (or tangent) inside the piece
  kLaneCentre = 7,         // through the centre
  kLaneCutoff = 8,         // the power fell below the cutoff over the piece
};

}  // namespace characteristic_detail

template <int kTile, bool kCbetRecord, bool kHotECapture, bool kPhysExt>
__device__ inline void ray_trace_1d_characteristic_body(
    const int ray,
    const CharacteristicPieces pieces,
    double* __restrict__ deposit_1d,
    double* __restrict__ deposit_per_ray,
    double* __restrict__ unabsorbed_per_ray,
    double* __restrict__ tail_power_per_ray,
    const double* __restrict__ radial_node_r,
    const double* __restrict__ radial_n_hat,
    const double* __restrict__ radial_n_hat_raw,
    const double* __restrict__ radial_smooth_kappa,
    const double* __restrict__ radial_dn_dr,
    const double* __restrict__ hydro_r_edges,
    const int allowed_supercritical_cell,
    const int critical_adjacent_subcritical_cell,
    const double critical_adjacent_split_r,
    const double* __restrict__ ray_R0,
    const double* __restrict__ ray_Z0,
    const double* __restrict__ ray_vR0,
    const double* __restrict__ ray_vZ0,
    const double* __restrict__ ray_power,
    const double* __restrict__ ray_power0,
    const double eps_n,
    const double eps_crit,
    const double lambda_cm,
    const double test_kappa_cm_inv,
    const double intensity_cutoff,
    const int max_ray_steps,
    const int n_radial_nodes,
    const int n_hydro_cells,
    const int n_rays,
    double* __restrict__ traj_pos_R,
    double* __restrict__ traj_pos_Z,
    double* __restrict__ traj_power,
    int* __restrict__ traj_step_count,
    const int n_output_rays,
    const int output_stride,
    const int traj_max_steps,
    int* __restrict__ step_histogram,
    int* __restrict__ step_count,
    int* __restrict__ ray_steps_out,
    double* __restrict__ P_unabsorbed,
    unsigned long long* __restrict__ tail_closure_count,
    double* __restrict__ tail_closure_absorbed_power,
    unsigned long long* __restrict__ critical_surface_hit_count,
    core::DeviceErrorFlags* __restrict__ error_flags,
    const CbetRecordDeviceArgs cbet_args,
    const HotECaptureParams hot_e_params,
    double* __restrict__ hot_e_capture,
    const laser::LaserPhysExtOptions phys_opt,
    const double* __restrict__ radial_T_e,
    double* __restrict__ ra_per_ray,
    double* __restrict__ tau_shell_out,
    const int per_ray_row_base = 0,
    const int reflect_at_critical = 0,
    const int geometry = 0,
    const double* __restrict__ ray_vA0 = nullptr) {
  using namespace characteristic_detail;
  // Mesh.geometry_1d (mesh::Geometry1D): 0 sphere, 1 cylinder, 2 slab.
  // (R, Z) is the ray's position in the plane of the trace: the orbital plane
  // through the centre of a sphere (R the distance from the beam axis), the
  // cross-section of a cylinder, or (lateral, height) for a slab, whose
  // profile coordinate is Z. ray_vA0 is the velocity component out of that
  // plane (along the cylinder axis; the second lateral direction of a slab).
  const bool planar = geometry == 2;
  static_assert(kTile == 1 || kTile == 32 || kTile == 64 || kTile == 128 || kTile == 256,
                "the characteristic trace runs on 1 lane or 32, 64, 128 or 256 lanes per ray");

  // The kTile lanes of a ray hold identical copies of the ray state. Each chunk
  // of kTile pieces along the path is processed with lane-parallel scans; the
  // events that end a leg are handled by all lanes alike, lane 0 writing.
  const int lane = tile_lane<kTile>();
  const bool writer = (lane == 0);
  const int tid = ray;
  // Row of this ray in the per-ray tallies (deposit, unabsorbed, tail power):
  // the launcher traces the rays in batches when all rows do not fit.
  const int row = ray - per_ray_row_base;
  const int output_idx = (output_stride > 0) ? (tid / output_stride) : -1;
  const bool traj_ray = (traj_pos_R != nullptr) && (output_stride > 0) &&
                        (tid % output_stride == 0) && (output_idx < n_output_rays);
  int traj_stored = 0;
  // Work counter for the step statistics and the max_steps guard: one per
  // piece plus one per extra quadrature panel.
  int n_steps = 0;
  RayStepHistogramGuard histogram_guard(writer ? step_histogram : nullptr, &n_steps,
                                        writer ? step_count : nullptr,
                                        writer ? ray_steps_out : nullptr, tid, n_rays);
  // This ray's deposit row (private: lanes add to different cells) or the
  // shared deposit array (atomics).
  double* const deposit_row =
      (deposit_per_ray != nullptr)
          ? deposit_per_ray + static_cast<std::size_t>(row) * static_cast<std::size_t>(n_hydro_cells)
          : nullptr;
  const auto add_deposit = [&](const int cell, const double power) {
    if (!(power > 0.0) || cell < 0 || cell >= n_hydro_cells) {
      return;
    }
    if (deposit_row != nullptr) {
      deposit_row[cell] += power;
    } else if (deposit_1d != nullptr) {
      atomic_add_double(&deposit_1d[cell], power);
    }
  };
  double* tail_slot =
      (writer && tail_power_per_ray != nullptr) ? &tail_power_per_ray[row] : nullptr;
  int hydro_hint = n_hydro_cells / 2;

  double R = ray_R0[tid];
  double Z = ray_Z0[tid];
  double vR = ray_vR0[tid];
  double vZ = ray_vZ0[tid];
  double vA = (ray_vA0 != nullptr) ? ray_vA0[tid] : 0.0;
  double I = ray_power[tid];
  const double I0 = ray_power0[tid];
  [[maybe_unused]] unsigned hot_e_captured_mask = 0u;

  // CBET records (the CbetRecordCursor semantics: one record per visit of a
  // cell, same-cell pieces merged, w = the power at the visit's entry).
  const bool rec_active = kCbetRecord && cbet_args.rec_cell != nullptr && cbet_args.cap_per_ray > 0;
  const long long rec_gray = cbet_args.ray_offset + tid;
  const long long rec_base = rec_gray * static_cast<long long>(cbet_args.cap_per_ray);
  int rec_count = 0;
  bool rec_overflow = false;
  int open_cell = -1;
  double open_ds = 0.0;
  double open_S = 0.0;
  double open_muds = 0.0;
  double open_w = 0.0;
  const auto rec_write = [&](const int idx, const bool write_head, const bool write_tail,
                             const int cell, const double w, const double muds,
                             const double ds, const double S) {
    const long long slot = rec_base + idx;
    if (write_head) {
      cbet_args.rec_cell[slot] = cell;
      cbet_args.rec_w[slot] = w;
    }
    if (write_tail) {
      const double mu = (ds > 0.0) ? ::fmin(1.0, ::fmax(-1.0, muds / ds)) : 0.0;
      cbet_args.rec_mu[slot] = static_cast<float>(mu);
      cbet_args.rec_ds[slot] = ds;
      cbet_args.rec_S[slot] = S;
    }
  };
  const auto rec_mark_overflow = [&]() {
    rec_overflow = true;
    if (writer && cbet_args.ray_overflow != nullptr) {
      cbet_args.ray_overflow[rec_gray] = 1;
    }
    open_cell = -1;
    open_ds = 0.0;
    open_S = 0.0;
    open_muds = 0.0;
  };
  const auto rec_flush_open = [&]() {
    if (!rec_active || rec_overflow || open_cell < 0) {
      return;
    }
    if (rec_count >= cbet_args.cap_per_ray) {
      rec_mark_overflow();
      return;
    }
    if (writer) {
      rec_write(rec_count, true, true, open_cell, open_w, open_muds, open_ds, open_S);
      cbet_args.rec_count[rec_gray] = rec_count + 1;
    }
    ++rec_count;
    open_cell = -1;
    open_ds = 0.0;
    open_S = 0.0;
    open_muds = 0.0;
  };
  // One piece added in the walk order (the serial paths of the events).
  const auto rec_add_segment = [&](const int cell, const double muds, const double ds,
                                   const double S, const double w_entry) {
    if (!rec_active || rec_overflow || cell < 0) {
      return;
    }
    if (cell != open_cell) {
      rec_flush_open();
      if (rec_overflow) {
        return;
      }
      open_cell = cell;
      open_w = w_entry;
    }
    open_ds += ds;
    open_S += S;
    open_muds += muds;
  };
  [[maybe_unused]] const auto rec_add_terminal = [&](const int cell, const double tau) {
    if (!rec_active) {
      return;
    }
    rec_flush_open();
    if (rec_overflow || cell < 0) {
      return;
    }
    if (rec_count >= cbet_args.cap_per_ray) {
      rec_mark_overflow();
      return;
    }
    if (writer) {
      rec_write(rec_count, true, true, cell, 0.0, 0.0, 0.0, tau);
      cbet_args.rec_count[rec_gray] = rec_count + 1;
    }
    ++rec_count;
  };

  const auto book_unabsorbed = [&](const double power) {
    if (!writer || !(power > 0.0)) {
      return;
    }
    if (unabsorbed_per_ray != nullptr) {
      unabsorbed_per_ray[row] += power;
    } else {
      atomic_add_double(P_unabsorbed, power);
    }
  };
  // The points written (write_traj keeps at most traj_max_steps of them).
  const auto finish_traj = [&]() {
    if (writer && traj_ray) {
      traj_step_count[output_idx] = min(traj_stored, traj_max_steps);
    }
  };
  // Trajectory point number `offset` after the stored ones, written by the
  // calling lane, in the (R >= 0, Z) meridional half-plane.
  const auto write_traj = [&](const int offset, const double R_pos, const double Z_pos,
                              const double power, const int rec_idx) {
    const int n = traj_stored + offset;
    if (!traj_ray || n >= traj_max_steps) {
      return;
    }
    const int idx = output_idx * traj_max_steps + n;
    traj_pos_R[idx] = R_pos;
    traj_pos_Z[idx] = Z_pos;
    traj_power[idx] = power;
    if (cbet_args.traj_rec_idx != nullptr) {
      cbet_args.traj_rec_idx[idx] = rec_idx;
    }
  };
  // Trajectory coordinates of a path point: (|r sin phi|, r cos phi) in the
  // plane of a sphere or cylinder; (lateral drift, height) for a slab, whose
  // "phi" accumulates the lateral position.
  const auto traj_R = [&](const double r_pos, const double phi_pos) {
    return planar ? phi_pos : ::fabs(r_pos * ::sin(phi_pos));
  };
  const auto traj_Z = [&](const double r_pos, const double phi_pos) {
    return planar ? r_pos : r_pos * ::cos(phi_pos);
  };
  const auto store_traj = [&](const double r_pos, const double phi_pos, const double power) {
    if (writer) {
      write_traj(0, traj_R(r_pos, phi_pos), traj_Z(r_pos, phi_pos), power, rec_count);
    }
    if (traj_ray) {
      ++traj_stored;
    }
  };
  // Serial deposits of the event paths (lane 0; the chunk's lane writes are
  // ordered before them).
  const auto event_deposit = [&](const int cell, const double power) {
    tile_sync<kTile>();
    if (writer) {
      add_deposit(cell, power);
    }
    tile_sync<kTile>();
  };
  // Critical termination: the remaining power goes to the critical-adjacent
  // cell with terminate_mode="deposit", else to the unabsorbed ledger.
  const auto terminate_critical = [&]() {
    rec_flush_open();
    if constexpr (kPhysExt) {
      if (phys_opt.crit_terminate_deposit != 0) {
        int dep_cell = critical_adjacent_subcritical_cell;
        if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
          dep_cell = allowed_supercritical_cell;
        }
        if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
          event_deposit(dep_cell, I);
          finish_traj();
          return;
        }
      }
    }
    book_unabsorbed(I);
    finish_traj();
  };
  const auto fail_invalid = [&](const bool nan_particle) {
    if (writer && error_flags != nullptr) {
      if (nan_particle) {
        atomicExch(&error_flags->nan_particle, 1);
      } else {
        atomicExch(&error_flags->invalid_cell, 1);
      }
    }
    rec_flush_open();
    book_unabsorbed(I);
    finish_traj();
  };

  if (!(::isfinite(R) && ::isfinite(Z) && ::isfinite(vR) && ::isfinite(vZ) && ::isfinite(vA) &&
        ::isfinite(I) && ::isfinite(I0))) {
    if (writer && error_flags != nullptr) {
      atomicExch(&error_flags->nan_particle, 1);
    }
    if (::isfinite(I) && I > 0.0) {
      book_unabsorbed(I);
    }
    finish_traj();
    return;
  }
  if (!(I > 0.0)) {
    finish_traj();
    return;
  }
  const int n_pieces = *pieces.count;
  if (n_radial_nodes < 2 || n_hydro_cells <= 0 || (!planar && !(radial_node_r[0] == 0.0)) ||
      n_pieces < 1) {
    fail_invalid(false);
    return;
  }

  const double r_max = radial_node_r[n_radial_nodes - 1];
  if (planar) {
    if (writer) {
      write_traj(0, R, Z, I, 0);
    }
    if (traj_ray) {
      ++traj_stored;
    }
    // Straight to the slab's outer end Z = r_max (the launch height is at or
    // above it).
    if (Z > r_max) {
      if (!(vZ < 0.0)) {
        book_unabsorbed(I);
        finish_traj();
        return;
      }
      R += vR * ((Z - r_max) / (-vZ));
      Z = r_max;
      if (writer) {
        write_traj(0, R, Z, I, 0);
      }
      if (traj_ray) {
        ++traj_stored;
      }
    }
  } else {
    reflect_axis_if_needed(&R, &vR);
    if (writer) {
      write_traj(0, R, Z, I, 0);
    }
    if (traj_ray) {
      ++traj_stored;
    }
    if (outside_radial_profile(R, Z, radial_node_r, n_radial_nodes)) {
      if (!advance_to_radial_profile_entry(&R, &Z, &vR, vZ, radial_node_r, n_radial_nodes)) {
        book_unabsorbed(I);
        finish_traj();
        return;
      }
      if (writer) {
        write_traj(0, R, Z, I, 0);
      }
      if (traj_ray) {
        ++traj_stored;
      }
    }
  }

  // Entry state.
  const double r_entry = planar ? ::fmin(Z, r_max) : ::fmin(radial_distance(R, Z), r_max);
  {
    const RadialInterval c_entry =
        locate_radial_interval(radial_node_r, n_radial_nodes, r_entry);
    const double nh_entry = interpolate_radial_field(radial_n_hat, c_entry);
    const double nh_entry_raw = interpolate_radial_field(radial_n_hat_raw, c_entry);
    if (!::isfinite(nh_entry) || !::isfinite(nh_entry_raw)) {
      fail_invalid(false);
      return;
    }
    if (nh_entry >= 1.0 - eps_crit) {
      terminate_critical();
      return;
    }
    // H = 1: |v| = sqrt(1 - n_hat) at the entry (the march's launch-speed rule).
    const double v_scale = ::sqrt(::fmax(0.0, 1.0 - nh_entry)) /
                           ::fmax(::sqrt(vR * vR + vZ * vZ + vA * vA), 1.0e-300);
    vR *= v_scale;
    vZ *= v_scale;
    vA *= v_scale;
  }
  RayInvariants inv;
  inv.planar = planar;
  double phi_sign = 1.0;
  double phi = 0.0;
  if (planar) {
    // Slab: the lateral velocity is conserved; "phi" is the lateral position.
    inv.V2 = vR * vR + vA * vA;
    inv.V = ::sqrt(inv.V2);
    phi_sign = (vR < 0.0) ? -1.0 : 1.0;
    phi = R;
  } else {
    const double L_signed = R * vZ - Z * vR;  // dphi/dxi = -L / r^2
    inv.B = ::fabs(L_signed);
    inv.B2 = inv.B * inv.B;
    inv.V2 = vA * vA;  // along the cylinder axis (0 in a sphere's orbital plane)
    inv.V = ::fabs(vA);
    phi_sign = (L_signed > 0.0) ? -1.0 : 1.0;
    phi = ::atan2(R, Z);
  }
  const double B = inv.B;
  if (!(::isfinite(inv.B) && ::isfinite(inv.V2) && ::isfinite(phi))) {
    fail_invalid(true);
    return;
  }

  LangdonContext langdon;
  if constexpr (kPhysExt) {
    // The vacuum-map intensity is the round beam's on a sphere (NUMERICS 5.4):
    // spheres only (the builder turns the Langdon factor off elsewhere).
    if (test_kappa_cm_inv <= 0.0 && phys_opt.langdon_model != 0 && radial_T_e != nullptr &&
        geometry == 0) {
      langdon.active = true;
      langdon.model = phys_opt.langdon_model;
      langdon.zcoll = phys_opt.langdon_zcoll;
      langdon.zcoll_radial = phys_opt.langdon_zcoll_radial;
      langdon.I0_wcm2 = phys_opt.langdon_I0_wcm2;
      langdon.w_cm = phys_opt.langdon_w_cm;
      langdon.profile_kind = phys_opt.langdon_profile_kind;
      langdon.sg_two_m = phys_opt.langdon_sg_two_m;
      langdon.te_min_eV = phys_opt.langdon_te_min_eV;
      langdon.lambda_cm = lambda_cm;
    }
  }

  const double cutoff_power = intensity_cutoff * I0;
  const int n_breakpoints = n_pieces + 1;
  double r_cur = r_entry;
  bool inward = planar ? (vZ < 0.0) : ((R * vR + Z * vZ) < 0.0);
  double r_turn = -1.0;          // exact root for the outward leg's u-variable
  bool skip_first_trigger = false;  // a trigger that did not close is not re-evaluated
  // Laser.absorption.critical_handling.terminate = False (reflect_at_critical):
  // no analytic critical-layer tail closure; a ray reaching the critical
  // radius reflects there and is traced outward like a ray at a turning
  // point. The outward leg does not re-test its starting point for the
  // critical value (it sits on the critical radius).
  const bool reflect_critical = reflect_at_critical != 0;
  bool skip_first_entry_critical = false;

  // Values at r_cur on radial interval jl for the tail closure (the march's
  // step-entry inputs).
  const auto entry_state = [&](const int jl, double& nh_cur, double& nh_cur_raw,
                               double& kappa_cur, RadialInterval& c_cur, double& dn_dr,
                               double& v_mag2, double& q_cur) {
    const LinearField nf = linear_field(radial_node_r, radial_n_hat, jl);
    const LinearField nf_raw = linear_field(radial_node_r, radial_n_hat_raw, jl);
    const LinearField sf = linear_field(radial_node_r, radial_smooth_kappa, jl);
    nh_cur = nf.at(r_cur);
    nh_cur_raw = nf_raw.at(r_cur);
    q_cur = q_of_r(nf, inv, r_cur);
    dn_dr = radial_dn_dr[jl];
    v_mag2 = ::fmax(1.0 - nh_cur, 0.0);
    const double w_interval = radial_node_r[jl + 1] - radial_node_r[jl];
    c_cur = RadialInterval{
        jl, (w_interval > 0.0) ? clamp_unit_interval((r_cur - radial_node_r[jl]) / w_interval)
                               : 0.0};
    kappa_cur = (test_kappa_cm_inv > 0.0)
                    ? test_kappa_cm_inv
                    : compute_kappa_from_smooth(sf.at(r_cur), nh_cur, eps_n);
  };
  // Analytic critical-layer tail closure at r_cur (the march's rule and
  // inputs); lane 0 deposits and counts, every lane takes the outcome.
  const auto try_tail = [&](const int jl, const TailClosureMode mode,
                            const bool use_v_dot_g) -> int {
    double nh_cur = 0.0;
    double nh_cur_raw = 0.0;
    double kappa_cur = 0.0;
    RadialInterval c_cur{};
    double dn_dr = 0.0;
    double v_mag2 = 0.0;
    double q_cur = 0.0;
    entry_state(jl, nh_cur, nh_cur_raw, kappa_cur, c_cur, dn_dr, v_mag2, q_cur);
    const double v_radial = (inward ? -1.0 : 1.0) * ::sqrt(::fmax(q_cur, 0.0)) /
                            (planar ? 1.0 : ::fmax(r_cur, 1.0e-300));
    const double v_dot_g = use_v_dot_g ? dn_dr * v_radial : 1.0;
    int status = static_cast<int>(TailClosureStatus::kNoClosure);
    double I_tail = I;
    tile_sync<kTile>();
    if (writer) {
      DepositCellCacheGuard cache(deposit_row != nullptr ? deposit_row : deposit_1d);
      status = static_cast<int>(try_tail_closure_1d(
          cache, c_cur, radial_smooth_kappa, radial_n_hat_raw, hydro_r_edges, n_hydro_cells,
          &hydro_hint, allowed_supercritical_cell, critical_adjacent_subcritical_cell,
          critical_adjacent_split_r, test_kappa_cm_inv, nh_cur, nh_cur_raw, kappa_cur,
          ::fabs(dn_dr), v_dot_g, v_mag2, r_cur, I, mode, tail_closure_count,
          tail_closure_absorbed_power, tail_slot, error_flags, I_tail));
    }
    tile_sync<kTile>();
    status = tile_broadcast<kTile>(status, 0);
    I_tail = tile_broadcast<kTile>(I_tail, 0);
    if (status == static_cast<int>(TailClosureStatus::kClosed)) {
      if constexpr (kCbetRecord) {
        int tail_cell = -1;
        double tau_tail = 0.0;
        if (rec_active &&
            cbet_probe_tail_1d(c_cur, radial_smooth_kappa, radial_n_hat_raw, hydro_r_edges,
                               n_hydro_cells, allowed_supercritical_cell,
                               critical_adjacent_subcritical_cell, critical_adjacent_split_r,
                               test_kappa_cm_inv, nh_cur, kappa_cur, ::fabs(dn_dr), r_cur,
                               &tail_cell, &tau_tail)) {
          rec_add_terminal(tail_cell, tau_tail);
        }
      }
      I = I_tail;
      store_traj(r_cur, phi, I);
    }
    return status;
  };

  // Resonance absorption at the turning point (or the critical reflection),
  // b = B on a sphere (the march's b_eff = sqrt(1 - n_hat) r at its last
  // inward step): b / r_crit is the sine of the vacuum incidence angle there.
  // Cylinder: sin^2 = (B / r_crit)^2 + V2; slab: sin = V.
  const auto turning_point_resonance_absorption = [&]() {
    if constexpr (kPhysExt) {
      if (phys_opt.ra_enable != 0) {
        const double r_c = phys_opt.ra_r_crit_cm;
        const double b_ra = planar ? r_c * inv.V
                                   : ((geometry == 1) ? ::sqrt(inv.B2 + inv.V2 * r_c * r_c) : B);
        const double f_ra = compute_ra_event_fraction(phys_opt, b_ra);
        if (f_ra > 0.0 && I > 0.0 && ra_per_ray != nullptr &&
            critical_adjacent_subcritical_cell >= 0 &&
            critical_adjacent_subcritical_cell < n_hydro_cells) {
          const double dP = f_ra * I;
          I -= dP;
          if (writer) {
            ra_per_ray[tid] += dP;
          }
          event_deposit(critical_adjacent_subcritical_cell, dP);
        }
      }
    }
  };
  // Reflection at the critical radius r_cur (reflect_critical). Q(r_cur) > 0
  // there (the ray has not reached its turning point), so the outward leg has
  // no regularising root at r_cur (r_turn = -1: each piece looks for its own).
  const auto reflect_at_critical_radius = [&]() {
    if (writer) {
      record_critical_surface_hit(critical_surface_hit_count);
    }
    inward = false;
    r_turn = -1.0;
    skip_first_entry_critical = true;
    turning_point_resonance_absorption();
  };

  while (true) {
    // ---- Chunk: up to kTile consecutive pieces along the path from r_cur ----
    const int k_first = inward ? count_less(pieces.r, n_breakpoints, r_cur) - 1
                               : count_less_equal(pieces.r, n_breakpoints, r_cur) - 1;
    const int n_avail = inward ? ::min(kTile, k_first + 1) : ::min(kTile, n_pieces - k_first);
    if (n_avail <= 0) {
      if (!inward) {
        // r_cur is at the profile edge.
        rec_flush_open();
        book_unabsorbed(I);
        finish_traj();
        return;
      }
      fail_invalid(false);
      return;
    }
    const bool active = lane < n_avail;
    PrefetchedPiece mine;
    if (active) {
      mine = prefetch_piece(pieces, inward ? k_first - lane : k_first + lane, inward, lane == 0,
                            r_cur, r_turn, radial_node_r, radial_n_hat, radial_n_hat_raw,
                            radial_smooth_kappa, radial_dn_dr, inv, eps_crit, eps_n,
                            test_kappa_cm_inv,
                            !reflect_critical && !(lane == 0 && skip_first_trigger),
                            !(lane == 0 && skip_first_entry_critical));
    }
    skip_first_trigger = false;
    skip_first_entry_critical = false;

    // Lane events in the march's order at a step.
    const int work = active ? 1 + ((mine.panels > 1) ? mine.panels - 1 : 0) : 0;
    const int work_before = n_steps + tile_exclusive_sum<kTile, int>(work, lane);
    int lane_event = kLaneNone;
    if (active) {
      if (work_before >= max_ray_steps) {
        lane_event = kLaneGuard;
      } else if (mine.entry_invalid != 0) {
        lane_event = kLaneInvalid;
      } else if (mine.entry_critical != 0) {
        lane_event = kLaneEntryCritical;
      } else if (mine.entry_trigger != 0) {
        lane_event = kLaneTrigger;
      } else if (mine.valid == 0) {
        lane_event = kLaneInvalid;
      } else if (inward && mine.event == kEventCritical) {
        lane_event = kLaneCritical;
      } else if (inward && (mine.event == kEventTurn || mine.event == kEventTangent)) {
        lane_event = kLaneTurn;
      } else if (inward && mine.event == kEventCentre) {
        lane_event = kLaneCentre;
      }
    }
    const int s1 = tile_first_true<kTile>(lane_event != kLaneNone);
    const int s1_event = (s1 < kTile) ? tile_broadcast<kTile>(lane_event, s1) : kLaneNone;
    const bool s1_traversed = (s1_event == kLaneTurn || s1_event == kLaneCentre);
    const bool traversed = active && (lane < s1 || (lane == s1 && s1_traversed));

    // Polar angle at each lane's entry, Langdon factor, optical depth.
    const double r_from = inward ? mine.b : mine.a;  // entry radius of the traversed part
    const double r_to = mine.r_stop;                  // exit radius
    const double dphi_signed = traversed ? phi_sign * mine.dphi : 0.0;
    const double phi_in = phi + tile_exclusive_sum<kTile, double>(dphi_signed, lane);
    [[maybe_unused]] const LinearField te_lane =
        (active && radial_T_e != nullptr) ? linear_field(radial_node_r, radial_T_e, mine.j)
                                          : LinearField{};
    [[maybe_unused]] const LinearField zcoll_lane =
        (active && langdon.zcoll_radial != nullptr)
            ? linear_field(radial_node_r, langdon.zcoll_radial, mine.j)
            : LinearField{};
    double tau_lane = 0.0;
    double absorbed_fraction = 0.0;
    double transmission = 1.0;
    if (traversed && mine.tau0 > 0.0) {
      // Along the path from the entry: 1 - frac inward (the variable grows
      // with r), frac outward.
      const double s_travel = inward ? 1.0 - mine.frac_centroid : mine.frac_centroid;
      tau_lane = mine.tau0 * langdon_factor_at(langdon, te_lane, mine.r_centroid,
                                               phi_in + s_travel * dphi_signed, zcoll_lane);
      absorbed_fraction = -::expm1(-tau_lane);
      transmission = ::exp(-tau_lane);
    }

    // Hot-electron captures: for each uncaptured channel, the first traversed
    // lane crossing its threshold upward in n_hat_raw; that lane splits its
    // piece at the crossing radius (sub-pieces integrated again).
    [[maybe_unused]] int n_sub = 1;
    [[maybe_unused]] double sub_r_end[HotECaptureParams::kMaxChannels + 1];
    [[maybe_unused]] double sub_dphi[HotECaptureParams::kMaxChannels + 1];
    [[maybe_unused]] double sub_ds[HotECaptureParams::kMaxChannels + 1];
    [[maybe_unused]] double sub_tau[HotECaptureParams::kMaxChannels + 1];
    [[maybe_unused]] double sub_frac[HotECaptureParams::kMaxChannels + 1];
    [[maybe_unused]] int sub_ch[HotECaptureParams::kMaxChannels + 1];
    [[maybe_unused]] unsigned chunk_captures = 0u;
    if constexpr (kHotECapture) {
      const LinearField nf_raw =
          active ? linear_field(radial_node_r, radial_n_hat_raw, mine.j) : LinearField{};
      double split_r[HotECaptureParams::kMaxChannels];
      int split_ch[HotECaptureParams::kMaxChannels];
      int n_split = 0;
      for (int he_ch = 0; he_ch < hot_e_params.n_channels; ++he_ch) {
        if ((hot_e_captured_mask & (1u << he_ch)) != 0u) {
          continue;
        }
        const double thr = hot_e_params.threshold_nhat[he_ch];
        const bool crosses = traversed && nf_raw.s != 0.0 && nf_raw.at(r_from) < thr &&
                             nf_raw.at(r_to) >= thr;
        const int c_lane = tile_first_true<kTile>(crosses);
        if (c_lane < kTile) {
          chunk_captures |= (1u << he_ch);
        }
        if (lane == c_lane) {
          split_r[n_split] = ::fmin(::fmax(nf_raw.where(thr), ::fmin(r_from, r_to)),
                                    ::fmax(r_from, r_to));
          split_ch[n_split] = he_ch;
          ++n_split;
        }
      }
      if (n_split > 0) {
        // Channels are sorted by threshold host-side: path order.
        const LinearField nf = linear_field(radial_node_r, radial_n_hat, mine.j);
        const LinearField sf = linear_field(radial_node_r, radial_smooth_kappa, mine.j);
        n_sub = n_split + 1;
        double a = r_from;
        double phi_sub = phi_in;
        transmission = 1.0;
        tau_lane = 0.0;
        for (int s = 0; s < n_sub; ++s) {
          const double b_end = (s < n_split) ? split_r[s] : r_to;
          const PieceIntegrals sub = integrate_piece(::fmin(a, b_end), ::fmax(a, b_end), mine.r0,
                                                     nf, sf, inv, eps_n, test_kappa_cm_inv);
          double tau = sub.valid ? sub.tau0 : 0.0;
          if (tau > 0.0) {
            const double s_travel = (b_end < a) ? 1.0 - sub.frac_centroid : sub.frac_centroid;
            tau *= langdon_factor_at(langdon, te_lane, sub.r_centroid,
                                     phi_sub + phi_sign * s_travel * sub.dphi, zcoll_lane);
          }
          sub_r_end[s] = b_end;
          sub_dphi[s] = sub.valid ? sub.dphi : 0.0;
          sub_ds[s] = sub.valid ? sub.ds : 0.0;
          sub_tau[s] = tau;
          sub_frac[s] = (tau > 0.0) ? -::expm1(-tau) : 0.0;
          sub_ch[s] = (s < n_split) ? split_ch[s] : -1;
          transmission *= ::exp(-tau);
          if (s < n_split) {
            transmission *= hot_e_params.one_minus_eta[split_ch[s]];
          }
          tau_lane += tau;
          phi_sub += phi_sign * sub_dphi[s];
          a = b_end;
        }
      }
    }

    // Power at each lane's entry and exit; the cutoff.
    const double P_in = I * tile_exclusive_product<kTile>(traversed ? transmission : 1.0, lane);
    const double P_out = P_in * transmission;
    const int u = tile_first_true<kTile>(traversed && P_out < cutoff_power);
    int e_lane = n_avail - 1;
    int e_event = kLaneNone;
    if (u < kTile && (u < s1 || (u == s1 && s1_traversed))) {
      e_lane = u;
      e_event = kLaneCutoff;
    } else if (s1 < kTile) {
      e_lane = s1;
      e_event = s1_event;
    }
    const bool apply = traversed && lane <= e_lane;
    const int n_applied = tile_count<kTile>(apply);
    if (apply && mine.unresolved != 0 && error_flags != nullptr) {
      atomicAdd(&error_flags->unresolved_quadrature, 1);
    }

    // Deposits, per-cell runs summed across the lanes (the run's last lane adds).
    double dP_lane = 0.0;
    if (apply) {
      if (n_sub == 1) {
        if (P_in > 0.0 && tau_lane > 0.0) {
          dP_lane = ::fmin(::fmax(P_in * absorbed_fraction, 0.0), P_in);
        }
      } else if constexpr (kHotECapture) {
        double P = P_in;
        for (int s = 0; s < n_sub; ++s) {
          double dP = 0.0;
          if (P > 0.0 && sub_tau[s] > 0.0) {
            dP = ::fmin(::fmax(P * sub_frac[s], 0.0), P);
          }
          dP_lane += dP;
          P -= dP;
          if (sub_ch[s] >= 0) {
            const double r_s = sub_r_end[s];
            const LinearField nf = linear_field(radial_node_r, radial_n_hat, mine.j);
            const double q_s = ::fmax(q_of_r(nf, inv, r_s), 0.0);
            const double v_mag = ::sqrt(::fmax(1.0 - nf.at(r_s), 0.0));
            double mu = 0.0;
            if (v_mag > 0.0 && (planar || r_s > 0.0)) {
              mu = ::fmin(1.0, ::fmax(-1.0, (inward ? -1.0 : 1.0) * ::sqrt(q_s) /
                                                 ((planar ? 1.0 : r_s) * v_mag)));
            }
            const std::size_t he_base =
                (static_cast<std::size_t>(tid) * hot_e_params.n_channels + sub_ch[s]) * 4;
            hot_e_capture[he_base + 0] = 1.0;
            hot_e_capture[he_base + 1] = r_s;
            hot_e_capture[he_base + 2] = mu;
            hot_e_capture[he_base + 3] = P;
            P *= hot_e_params.one_minus_eta[sub_ch[s]];
          }
        }
      }
    }
    {
      const int key = apply ? mine.cell : -2 - lane;
      const double run_sum = tile_segmented_inclusive_sum<kTile>(apply ? dP_lane : 0.0, key, lane);
      const int next_key = tile_next<kTile>(key, -1, lane);
      if (apply && next_key != key) {
        add_deposit(mine.cell, run_sum);
      }
      if (tau_shell_out != nullptr) {
        const int jkey = apply ? mine.j : -2 - lane;
        const double tau_sum =
            tile_segmented_inclusive_sum<kTile>(apply ? tau_lane : 0.0, jkey, lane);
        // Every lane takes part in the shuffle (a shuffle behind a lane
        // condition waits for the absent lanes forever).
        const int next_jkey = tile_next<kTile>(jkey, -1, lane);
        if (apply && next_jkey != jkey) {
          tau_shell_out[static_cast<std::size_t>(tid) *
                            static_cast<std::size_t>(n_radial_nodes - 1) +
                        static_cast<std::size_t>(mine.j)] += tau_sum;
        }
      }
    }

    // CBET records of the chunk's cell visits.
    int rec_count_chunk = rec_count;  // records flushed before this chunk's visits
    int rec_ordinal = 0;              // this lane's visit within the chunk
    if constexpr (kCbetRecord) {
      if (rec_active && !rec_overflow && n_applied > 0) {
        const int cell0 = tile_broadcast<kTile>(mine.cell, 0);
        if (open_cell >= 0 && open_cell != cell0) {
          rec_flush_open();
        }
        if (!rec_overflow) {
          rec_count_chunk = rec_count;
          const bool merge = (open_cell >= 0 && open_cell == cell0);
          double c_ds = apply ? ((n_sub == 1) ? mine.ds : 0.0) : 0.0;
          double c_S = apply ? tau_lane : 0.0;
          double c_muds = apply ? (r_to - r_from) : 0.0;
          if constexpr (kHotECapture) {
            if (apply && n_sub > 1) {
              for (int s = 0; s < n_sub; ++s) {
                c_ds += sub_ds[s];
              }
            }
          }
          double c_w = P_in;
          if (lane == 0 && merge) {
            c_ds += open_ds;
            c_S += open_S;
            c_muds += open_muds;
            c_w = open_w;
          }
          const int key = apply ? mine.cell : -2 - lane;
          const double s_ds = tile_segmented_inclusive_sum<kTile>(c_ds, key, lane);
          const double s_S = tile_segmented_inclusive_sum<kTile>(c_S, key, lane);
          const double s_muds = tile_segmented_inclusive_sum<kTile>(c_muds, key, lane);
          const int prev_key = tile_previous<kTile>(key, -1, lane);
          const bool is_head = apply && (lane == 0 || prev_key != key);
          const int n_visits = tile_count<kTile>(is_head);
          rec_ordinal = tile_count<kTile>(is_head, true) - 1;
          const int next_key = tile_next<kTile>(key, -1, lane);
          const bool is_tail = apply && next_key != key;
          const int n_complete = n_visits - 1;  // the last visit stays open
          const bool complete = apply && rec_ordinal < n_complete;
          const int idx = rec_count + rec_ordinal;
          if (complete && idx < cbet_args.cap_per_ray) {
            rec_write(idx, is_head, is_tail, mine.cell, c_w, s_muds, s_ds, s_S);
          }
          // The open (last) visit: its sums from the last applied lane, its w
          // from its first lane.
          const int last_lane = n_applied - 1;
          const int open_head = tile_last_true<kTile>(is_head);
          if (rec_count + n_complete > cbet_args.cap_per_ray) {
            if (writer) {
              cbet_args.rec_count[rec_gray] = cbet_args.cap_per_ray;
            }
            rec_count = cbet_args.cap_per_ray;
            rec_mark_overflow();
          } else {
            rec_count += n_complete;
            if (writer && n_complete > 0) {
              cbet_args.rec_count[rec_gray] = rec_count;
            }
            open_cell = tile_broadcast<kTile>(mine.cell, last_lane);
            open_ds = tile_broadcast<kTile>(s_ds, last_lane);
            open_S = tile_broadcast<kTile>(s_S, last_lane);
            open_muds = tile_broadcast<kTile>(s_muds, last_lane);
            open_w = tile_broadcast<kTile>(c_w, open_head);
          }
        }
      }
    }

    // Trajectory points at the traversed pieces' exits.
    if (traj_ray) {
      const int n_points = apply ? n_sub : 0;
      const int offset = tile_exclusive_sum<kTile, int>(n_points, lane);
      if (apply) {
        if (n_sub == 1) {
          const double phi_out = phi_in + dphi_signed;
          write_traj(offset, traj_R(r_to, phi_out), traj_Z(r_to, phi_out), P_out,
                     rec_count_chunk + rec_ordinal);
        } else if constexpr (kHotECapture) {
          double P = P_in;
          double phi_p = phi_in;
          for (int s = 0; s < n_sub; ++s) {
            double dP = 0.0;
            if (P > 0.0 && sub_tau[s] > 0.0) {
              dP = ::fmin(::fmax(P * sub_frac[s], 0.0), P);
            }
            P -= dP;
            phi_p += phi_sign * sub_dphi[s];
            write_traj(offset + s, traj_R(sub_r_end[s], phi_p), traj_Z(sub_r_end[s], phi_p), P,
                       rec_count_chunk + rec_ordinal);
            if (sub_ch[s] >= 0) {
              P *= hot_e_params.one_minus_eta[sub_ch[s]];
            }
          }
        }
      }
      traj_stored += tile_broadcast<kTile>(offset + n_points, (n_applied > 0) ? n_applied - 1 : 0) *
                     ((n_applied > 0) ? 1 : 0);
    }

    // State after the applied lanes; the event of the chunk's end lane.
    if constexpr (kHotECapture) {
      // Captures in lanes past the end lane did not happen.
      for (int he_ch = 0; he_ch < hot_e_params.n_channels; ++he_ch) {
        if ((chunk_captures & (1u << he_ch)) == 0u) {
          continue;
        }
        bool has = false;
        if (apply && n_sub > 1) {
          for (int s = 0; s < n_sub; ++s) {
            has = has || sub_ch[s] == he_ch;
          }
        }
        if (tile_any<kTile>(has)) {
          hot_e_captured_mask |= (1u << he_ch);
        }
      }
    }
    const double e_P_in = tile_broadcast<kTile>(P_in, e_lane);
    const double e_P_out = tile_broadcast<kTile>(P_out, e_lane);
    const double e_phi_in = tile_broadcast<kTile>(phi_in, e_lane);
    const double e_dphi = tile_broadcast<kTile>(dphi_signed, e_lane);
    const double e_r_from = tile_broadcast<kTile>(r_from, e_lane);
    const double e_r_to = tile_broadcast<kTile>(r_to, e_lane);
    const int e_j = tile_broadcast<kTile>(mine.j, e_lane);
    const int e_work_before = tile_broadcast<kTile>(work_before, e_lane);
    const int e_work = tile_broadcast<kTile>(work, e_lane);
    tile_sync<kTile>();

    if (e_event == kLaneGuard) {
      I = e_P_in;
      phi = e_phi_in;
      n_steps = e_work_before;
      rec_flush_open();
      book_unabsorbed(I);
      if (writer && error_flags != nullptr) {
        atomicAdd(&error_flags->infinite_loop, 1);
      }
      finish_traj();
      return;
    }
    n_steps = e_work_before + e_work;
    const bool e_traversed = (e_event == kLaneNone || e_event == kLaneCutoff ||
                              e_event == kLaneTurn || e_event == kLaneCentre);
    if (e_traversed) {
      I = e_P_out;
      phi = e_phi_in + e_dphi;
      r_cur = e_r_to;
    } else {
      I = e_P_in;
      phi = e_phi_in;
      r_cur = e_r_from;
    }
    switch (e_event) {
      case kLaneCutoff:
        rec_flush_open();
        book_unabsorbed(I);
        finish_traj();
        return;
      case kLaneInvalid:
        fail_invalid(false);
        return;
      case kLaneEntryCritical:
        if (reflect_critical && inward) {
          reflect_at_critical_radius();
          continue;
        }
        terminate_critical();
        return;
      case kLaneTrigger: {
        const int status = try_tail(e_j, TailClosureMode::kRequireTrigger, true);
        if (status == static_cast<int>(TailClosureStatus::kInvalid)) {
          fail_invalid(true);
          return;
        }
        if (status == static_cast<int>(TailClosureStatus::kClosed)) {
          terminate_critical();
          return;
        }
        // Not closed after all: the piece is traversed from r_cur.
        skip_first_trigger = true;
        continue;
      }
      case kLaneCritical: {
        // The march's critical-crossing segment: staged captures at the entry
        // power, then the tail closure without the trigger gate; failing that,
        // absorption to the critical radius and termination there.
        const double e_stop = tile_broadcast<kTile>(mine.r_stop, e_lane);
        const double e_ds = tile_broadcast<kTile>(mine.ds, e_lane);
        const double e_dphi_abs = tile_broadcast<kTile>(mine.dphi, e_lane);
        const double e_tau0 = tile_broadcast<kTile>(mine.tau0, e_lane);
        const double e_r_centroid = tile_broadcast<kTile>(mine.r_centroid, e_lane);
        const double e_frac_centroid = tile_broadcast<kTile>(mine.frac_centroid, e_lane);
        const int e_cell = tile_broadcast<kTile>(mine.cell, e_lane);
        if constexpr (kHotECapture) {
          const LinearField nf = linear_field(radial_node_r, radial_n_hat, e_j);
          const LinearField nf_raw = linear_field(radial_node_r, radial_n_hat_raw, e_j);
          for (int he_ch = 0; he_ch < hot_e_params.n_channels; ++he_ch) {
            if ((hot_e_captured_mask & (1u << he_ch)) != 0u) {
              continue;
            }
            const double thr = hot_e_params.threshold_nhat[he_ch];
            if (!(nf_raw.at(r_cur) < thr && nf_raw.at(e_stop) >= thr) || nf_raw.s == 0.0) {
              continue;
            }
            hot_e_captured_mask |= (1u << he_ch);
            const double r_s = ::fmin(::fmax(nf_raw.where(thr), ::fmin(e_stop, r_cur)),
                                      ::fmax(e_stop, r_cur));
            const double q_s = ::fmax(q_of_r(nf, inv, r_s), 0.0);
            const double v_mag = ::sqrt(::fmax(1.0 - nf.at(r_s), 0.0));
            double mu = 0.0;
            if (v_mag > 0.0 && (planar || r_s > 0.0)) {
              mu = ::fmin(1.0, ::fmax(-1.0, -::sqrt(q_s) / ((planar ? 1.0 : r_s) * v_mag)));
            }
            if (writer) {
              const std::size_t he_base =
                  (static_cast<std::size_t>(tid) * hot_e_params.n_channels + he_ch) * 4;
              hot_e_capture[he_base + 0] = 1.0;
              hot_e_capture[he_base + 1] = r_s;
              hot_e_capture[he_base + 2] = mu;
              hot_e_capture[he_base + 3] = I;
            }
            I *= hot_e_params.one_minus_eta[he_ch];
          }
        }
        if (!reflect_critical) {
          const int status = try_tail(e_j, TailClosureMode::kCriticalCrossing, false);
          if (status == static_cast<int>(TailClosureStatus::kClosed)) {
            terminate_critical();
            return;
          }
          if (writer) {
            record_critical_surface_hit(critical_surface_hit_count);
          }
        }
        if (::fabs(e_stop - r_cur) > 0.0) {
          double tau = e_tau0;
          if (tau > 0.0) {
            const LinearField te = (radial_T_e != nullptr)
                                       ? linear_field(radial_node_r, radial_T_e, e_j)
                                       : LinearField{};
            const LinearField zc = (langdon.zcoll_radial != nullptr)
                                       ? linear_field(radial_node_r, langdon.zcoll_radial, e_j)
                                       : LinearField{};
            tau *= langdon_factor_at(langdon, te, e_r_centroid,
                                     phi + phi_sign * (1.0 - e_frac_centroid) * e_dphi_abs, zc);
          }
          if (writer && tau_shell_out != nullptr) {
            tau_shell_out[static_cast<std::size_t>(tid) *
                              static_cast<std::size_t>(n_radial_nodes - 1) +
                          static_cast<std::size_t>(e_j)] += tau;
          }
          double I_next = I;
          const double dP = absorbed_power_expm1(I, tau, I_next);
          event_deposit(e_cell, dP);
          if constexpr (kCbetRecord) {
            if (e_ds > 0.0) {
              rec_add_segment(e_cell, e_stop - r_cur, e_ds, tau, I);
            }
          }
          phi += phi_sign * e_dphi_abs;
          I = I_next;
          store_traj(e_stop, phi, I);
        }
        if (reflect_critical) {
          r_cur = e_stop;
          reflect_at_critical_radius();
          continue;
        }
        terminate_critical();
        return;
      }
      case kLaneTurn:
        inward = false;
        r_turn = r_cur;
        turning_point_resonance_absorption();
        break;
      case kLaneCentre:
        inward = false;
        if (planar) {
          // The slab's inner wall (the hydro boundary's rigid wall, a mirror
          // plane): reflected, Q > 0 there (no regularising root).
          r_turn = -1.0;
        } else {
          // Through the centre (B = 0): the ray continues on the opposite side.
          r_turn = 0.0;
          phi += kPi;
        }
        break;
      default:
        if (!inward && r_cur >= r_max) {
          rec_flush_open();
          book_unabsorbed(I);
          finish_traj();
          return;
        }
        break;
    }
  }
}

}  // namespace tenryu::laser::ray_trace_bodies
