#include "mesh/tessellation/lloyd_skip.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

#include "mesh/tessellation/sos_policy.hpp"

namespace tenryu::mesh::tess {
namespace {

enum class CertOutcome { kOk, kSlackNonpositive, kMarginExceeded };

CertOutcome orient_certificate(const Site& a, const Site& b, const Site& c,
                               double dra, double dza, double drb, double dzb,
                               double drc, double dzc) {
  // Filter quantities exactly as orient2d_sign_tess computes them.
  const double acx = a.r - c.r;
  const double acy = a.z - c.z;
  const double bcx = b.r - c.r;
  const double bcy = b.z - c.z;
  const double left = acx * bcy;
  const double right = acy * bcx;
  const double determinant = left - right;
  const double error_bound =
      kFilterContractionSafety * kOrientErrBound *
      (std::fabs(left) + std::fabs(right));
  const double slack = std::fabs(determinant) - error_bound;
  if (!(slack > 0.0)) {
    return CertOutcome::kSlackNonpositive;
  }
  // (U, Delta) interval propagation mirroring the same expression tree
  // (design doc section 4): for a product, Delta = U_x*Delta_y + U_y*Delta_x.
  const double d_acx = dra + drc;
  const double u_acx = std::fabs(acx) + d_acx;
  const double d_acy = dza + dzc;
  const double u_acy = std::fabs(acy) + d_acy;
  const double d_bcx = drb + drc;
  const double u_bcx = std::fabs(bcx) + d_bcx;
  const double d_bcy = dzb + dzc;
  const double u_bcy = std::fabs(bcy) + d_bcy;
  const double d_left = u_acx * d_bcy + u_bcy * d_acx;
  const double d_right = u_acy * d_bcx + u_bcx * d_acy;
  const double d_det = d_left + d_right;
  if (!(2.0 * d_det <= slack)) {
    return CertOutcome::kMarginExceeded;
  }
  return CertOutcome::kOk;
}

CertOutcome incircle_certificate(const Site& a, const Site& b, const Site& c,
                                 const Site& d, double dra, double dza,
                                 double drb, double dzb, double drc,
                                 double dzc, double drd, double dzd) {
  // Filter quantities exactly as incircle_sign_tess computes them.
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
      (std::fabs(bdxcdy) + std::fabs(cdxbdy)) * alift +
      (std::fabs(cdxady) + std::fabs(adxcdy)) * blift +
      (std::fabs(adxbdy) + std::fabs(bdxady)) * clift;
  const double error_bound =
      kFilterContractionSafety * kIncircleErrBound * permanent;
  const double slack = std::fabs(determinant) - error_bound;
  if (!(slack > 0.0)) {
    return CertOutcome::kSlackNonpositive;
  }
  // (U, Delta) leaves.
  const double d_adx = dra + drd;
  const double u_adx = std::fabs(adx) + d_adx;
  const double d_ady = dza + dzd;
  const double u_ady = std::fabs(ady) + d_ady;
  const double d_bdx = drb + drd;
  const double u_bdx = std::fabs(bdx) + d_bdx;
  const double d_bdy = dzb + dzd;
  const double u_bdy = std::fabs(bdy) + d_bdy;
  const double d_cdx = drc + drd;
  const double u_cdx = std::fabs(cdx) + d_cdx;
  const double d_cdy = dzc + dzd;
  const double u_cdy = std::fabs(cdy) + d_cdy;
  // Pairwise products.
  const double d_bdxcdy = u_bdx * d_cdy + u_cdy * d_bdx;
  const double u_bdxcdy = u_bdx * u_cdy;
  const double d_cdxbdy = u_cdx * d_bdy + u_bdy * d_cdx;
  const double u_cdxbdy = u_cdx * u_bdy;
  const double d_cdxady = u_cdx * d_ady + u_ady * d_cdx;
  const double u_cdxady = u_cdx * u_ady;
  const double d_adxcdy = u_adx * d_cdy + u_cdy * d_adx;
  const double u_adxcdy = u_adx * u_cdy;
  const double d_adxbdy = u_adx * d_bdy + u_bdy * d_adx;
  const double u_adxbdy = u_adx * u_bdy;
  const double d_bdxady = u_bdx * d_ady + u_ady * d_bdx;
  const double u_bdxady = u_bdx * u_ady;
  // Lifts (square = product with both factors equal).
  const double d_alift = 2.0 * u_adx * d_adx + 2.0 * u_ady * d_ady;
  const double u_alift = u_adx * u_adx + u_ady * u_ady;
  const double d_blift = 2.0 * u_bdx * d_bdx + 2.0 * u_bdy * d_bdy;
  const double u_blift = u_bdx * u_bdx + u_bdy * u_bdy;
  const double d_clift = 2.0 * u_cdx * d_cdx + 2.0 * u_cdy * d_cdy;
  const double u_clift = u_cdx * u_cdx + u_cdy * u_cdy;
  // Cross-term pair sums.
  const double d_cross_a = d_bdxcdy + d_cdxbdy;
  const double u_cross_a = u_bdxcdy + u_cdxbdy;
  const double d_cross_b = d_cdxady + d_adxcdy;
  const double u_cross_b = u_cdxady + u_adxcdy;
  const double d_cross_c = d_adxbdy + d_bdxady;
  const double u_cross_c = u_adxbdy + u_bdxady;
  // Lifted terms and the determinant's change bound.
  const double d_term_a = u_alift * d_cross_a + u_cross_a * d_alift;
  const double d_term_b = u_blift * d_cross_b + u_cross_b * d_blift;
  const double d_term_c = u_clift * d_cross_c + u_cross_c * d_clift;
  const double d_det = d_term_a + d_term_b + d_term_c;
  if (!(2.0 * d_det <= slack)) {
    return CertOutcome::kMarginExceeded;
  }
  return CertOutcome::kOk;
}

bool valid_site(const DelaunayTriangulation& dt, const SiteId site) {
  return site >= 0 && site < static_cast<SiteId>(dt.sites.size());
}

bool valid_half_edge(const DelaunayTriangulation& dt,
                     const HalfEdgeId edge) {
  return edge >= 0 &&
         edge < static_cast<HalfEdgeId>(dt.he_origin.size());
}

}  // namespace

LloydSkipReport certify_lloyd_noop(const DelaunayTriangulation& dt,
                                   const std::vector<double>& delta_r,
                                   const std::vector<double>& delta_z) {
  LloydSkipReport report;
  const std::size_t site_count = dt.sites.size();
  const std::size_t half_edge_count = dt.he_origin.size();
  if (site_count < 3 || delta_r.size() != site_count ||
      delta_z.size() != site_count || dt.tri_edge.empty() ||
      dt.he_twin.size() != half_edge_count ||
      dt.he_next.size() != half_edge_count ||
      dt.he_face.size() != half_edge_count ||
      half_edge_count != 3 * dt.tri_edge.size()) {
    report.refusal = "degenerate_structure";
    return report;
  }

  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    const HalfEdgeId e0 = dt.tri_edge[triangle];
    if (!valid_half_edge(dt, e0)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const HalfEdgeId e1 = dt.he_next[e0];
    if (!valid_half_edge(dt, e1)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const HalfEdgeId e2 = dt.he_next[e1];
    if (!valid_half_edge(dt, e2) || dt.he_next[e2] != e0) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const SiteId v0 = dt.he_origin[e0];
    const SiteId v1 = dt.he_origin[e1];
    const SiteId v2 = dt.he_origin[e2];
    if (!valid_site(dt, v0) || !valid_site(dt, v1) ||
        !valid_site(dt, v2)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const CertOutcome outcome = orient_certificate(
        dt.sites[v0], dt.sites[v1], dt.sites[v2], delta_r[v0], delta_z[v0],
        delta_r[v1], delta_z[v1], delta_r[v2], delta_z[v2]);
    ++report.orient_checked;
    ++report.certificates_checked;
    if (outcome == CertOutcome::kSlackNonpositive) {
      report.refusal = "slack_nonpositive_orient";
      return report;
    }
    if (outcome == CertOutcome::kMarginExceeded) {
      report.refusal = "margin_exceeded_orient";
      return report;
    }
  }

  for (HalfEdgeId e = 0;
       e < static_cast<HalfEdgeId>(half_edge_count); ++e) {
    const HalfEdgeId t = dt.he_twin[e];
    if (t == kInvalidId) {
      continue;
    }
    if (!valid_half_edge(dt, t)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    if (e > t) {
      continue;
    }
    const HalfEdgeId e1 = dt.he_next[e];
    if (!valid_half_edge(dt, e1)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const HalfEdgeId e2 = dt.he_next[e1];
    if (!valid_half_edge(dt, e2)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const HalfEdgeId t1 = dt.he_next[t];
    if (!valid_half_edge(dt, t1)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const HalfEdgeId t2 = dt.he_next[t1];
    if (!valid_half_edge(dt, t2)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const SiteId a = dt.he_origin[e];
    const SiteId b = dt.he_origin[e1];
    const SiteId c = dt.he_origin[e2];
    const SiteId d = dt.he_origin[t2];
    if (!valid_site(dt, a) || !valid_site(dt, b) ||
        !valid_site(dt, c) || !valid_site(dt, d)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const CertOutcome outcome = incircle_certificate(
        dt.sites[a], dt.sites[b], dt.sites[c], dt.sites[d],
        delta_r[a], delta_z[a], delta_r[b], delta_z[b],
        delta_r[c], delta_z[c], delta_r[d], delta_z[d]);
    ++report.incircle_checked;
    ++report.certificates_checked;
    if (outcome == CertOutcome::kSlackNonpositive) {
      report.refusal = "slack_nonpositive_incircle";
      return report;
    }
    if (outcome == CertOutcome::kMarginExceeded) {
      report.refusal = "margin_exceeded_incircle";
      return report;
    }
  }

  std::vector<HalfEdgeId> boundary_out(site_count, kInvalidId);
  std::size_t boundary_count = 0;
  for (HalfEdgeId e = 0;
       e < static_cast<HalfEdgeId>(half_edge_count); ++e) {
    if (dt.he_twin[e] != kInvalidId) {
      continue;
    }
    const SiteId origin = dt.he_origin[e];
    if (!valid_site(dt, origin) || boundary_out[origin] != kInvalidId) {
      report.refusal = "degenerate_structure";
      return report;
    }
    boundary_out[origin] = e;
    ++boundary_count;
  }
  if (boundary_count < 3) {
    report.refusal = "degenerate_structure";
    return report;
  }

  for (HalfEdgeId e = 0;
       e < static_cast<HalfEdgeId>(half_edge_count); ++e) {
    if (dt.he_twin[e] != kInvalidId) {
      continue;
    }
    const SiteId p = dt.he_origin[e];
    if (!valid_site(dt, p)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const HalfEdgeId e1 = dt.he_next[e];
    if (!valid_half_edge(dt, e1)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const SiteId q = dt.he_origin[e1];
    if (!valid_site(dt, q)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const HalfEdgeId s = boundary_out[q];
    if (s == kInvalidId || !valid_half_edge(dt, s)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const HalfEdgeId s1 = dt.he_next[s];
    if (!valid_half_edge(dt, s1)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const SiteId r = dt.he_origin[s1];
    if (!valid_site(dt, r)) {
      report.refusal = "degenerate_structure";
      return report;
    }
    const CertOutcome outcome = orient_certificate(
        dt.sites[p], dt.sites[q], dt.sites[r], delta_r[p], delta_z[p],
        delta_r[q], delta_z[q], delta_r[r], delta_z[r]);
    ++report.hull_checked;
    ++report.certificates_checked;
    if (outcome == CertOutcome::kSlackNonpositive) {
      report.refusal = "slack_nonpositive_hull";
      return report;
    }
    if (outcome == CertOutcome::kMarginExceeded) {
      report.refusal = "margin_exceeded_hull";
      return report;
    }
  }

  report.certified = true;
  report.refusal = nullptr;
  return report;
}

}  // namespace tenryu::mesh::tess
