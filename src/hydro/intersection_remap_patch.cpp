#include "hydro/intersection_remap_patch.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace tenryu::hydro::ixremap {
namespace {

constexpr double kPiOverThree =
    1.0471975511965977461542144610931676280657231331250352736615;
constexpr double kAxisCanonicalThreshold = 1.0e-300;
constexpr double kClipRelativeEpsilon =
    64.0 * std::numeric_limits<double>::epsilon();

double canonical_r(const double r) {
  return std::abs(r) < kAxisCanonicalThreshold ? 0.0 : r;
}

double cross(const double ar,
             const double az,
             const double br,
             const double bz) {
  return ar * bz - az * br;
}

double polygon_twice_area(const Polygon& p) {
  double twice_area = 0.0;
  for (std::size_t k = 0; k < p.r.size(); ++k) {
    const std::size_t kp = (k + 1U) % p.r.size();
    twice_area += canonical_r(p.r[k]) * p.z[kp] -
                  canonical_r(p.r[kp]) * p.z[k];
  }
  return twice_area;
}

void append_vertex(Polygon& polygon, const double r, const double z) {
  const double canonical = canonical_r(r);
  if (!polygon.r.empty() && polygon.r.back() == canonical &&
      polygon.z.back() == z) {
    return;
  }
  polygon.r.push_back(canonical);
  polygon.z.push_back(z);
}

void remove_closing_duplicate(Polygon& polygon) {
  if (polygon.r.size() > 1U && polygon.r.front() == polygon.r.back() &&
      polygon.z.front() == polygon.z.back()) {
    polygon.r.pop_back();
    polygon.z.pop_back();
  }
}

double edge_side(const double ar,
                 const double az,
                 const double br,
                 const double bz,
                 const double qr,
                 const double qz) {
  return cross(br - ar, bz - az, qr - ar, qz - az);
}

double edge_tolerance(const Polygon& polygon,
                      const double ar,
                      const double az,
                      const double br,
                      const double bz) {
  double scale = std::max(std::abs(br - ar), std::abs(bz - az));
  for (std::size_t k = 0; k < polygon.r.size(); ++k) {
    scale = std::max(scale, std::abs(polygon.r[k] - ar));
    scale = std::max(scale, std::abs(polygon.z[k] - az));
  }
  return kClipRelativeEpsilon * scale * scale;
}

void append_line_intersection(Polygon& polygon,
                              const double sr,
                              const double sz,
                              const double er,
                              const double ez,
                              const double side_s,
                              const double side_e) {
  const double denominator = side_s - side_e;
  if (denominator == 0.0 || !std::isfinite(denominator)) {
    return;
  }
  const double t = side_s / denominator;
  if (!(t >= 0.0 && t <= 1.0)) {
    return;
  }
  if (t == 0.0) {
    append_vertex(polygon, sr, sz);
  } else if (t == 1.0) {
    append_vertex(polygon, er, ez);
  } else {
    append_vertex(polygon,
                  sr + t * (er - sr),
                  sz + t * (ez - sz));
  }
}

double corner_jacobian(const double r[4],
                       const double z[4],
                       const int corner) {
  const int next = (corner + 1) & 3;
  const int previous = (corner + 3) & 3;
  return cross(r[next] - r[corner], z[next] - z[corner],
               r[previous] - r[corner], z[previous] - z[corner]);
}

Polygon make_triangle(const double r[4],
                      const double z[4],
                      const int a,
                      const int b,
                      const int c) {
  Polygon triangle{{canonical_r(r[a]), canonical_r(r[b]), canonical_r(r[c])},
                   {z[a], z[b], z[c]}};
  if (polygon_twice_area(triangle) < 0.0) {
    std::swap(triangle.r[1], triangle.r[2]);
    std::swap(triangle.z[1], triangle.z[2]);
  }
  return triangle;
}

Polygon make_quad(const double r[4], const double z[4]) {
  Polygon quad;
  quad.r.reserve(4U);
  quad.z.reserve(4U);
  for (int k = 0; k < 4; ++k) {
    quad.r.push_back(canonical_r(r[k]));
    quad.z.push_back(z[k]);
  }
  return quad;
}

void unpack_quad(const std::array<double, 8>& packed,
                 double r[4],
                 double z[4]) {
  for (int k = 0; k < 4; ++k) {
    r[k] = canonical_r(packed[static_cast<std::size_t>(k)]);
    z[k] = packed[static_cast<std::size_t>(k + 4)];
  }
}

double split_quad_volume(const double r[4], const double z[4]) {
  const QuadSplit split = split_quad_deterministic(r, z);
  double volume = 0.0;
  for (int k = 0; k < split.n; ++k) {
    volume += polygon_rz_volume(split.tri[k]);
  }
  return volume;
}

bool strictly_convex_quad(const double r[4], const double z[4]) {
  for (int k = 0; k < 4; ++k) {
    if (!(corner_jacobian(r, z, k) > 0.0)) {
      return false;
    }
  }
  return true;
}

}  // namespace

double polygon_rz_volume(const Polygon& p) {
  if (p.r.size() != p.z.size() || p.r.size() < 3U) {
    return 0.0;
  }

  const double twice_area = polygon_twice_area(p);
  assert(twice_area >= 0.0);
  if (!(twice_area > 0.0)) {
    return 0.0;
  }

  double sum = 0.0;
  for (std::size_t k = 0; k < p.r.size(); ++k) {
    const std::size_t kp = (k + 1U) % p.r.size();
    const double rk = canonical_r(p.r[k]);
    const double rkp = canonical_r(p.r[kp]);
    sum += (p.z[kp] - p.z[k]) *
           (rk * rk + rk * rkp + rkp * rkp);
  }
  const double volume = kPiOverThree * sum;
  return volume > 0.0 ? volume : 0.0;
}

Polygon clip_convex(const Polygon& subject, const Polygon& clip) {
  if (subject.r.size() != subject.z.size() ||
      clip.r.size() != clip.z.size() ||
      subject.r.size() < 3U || clip.r.size() < 3U) {
    return {};
  }
  if (!(polygon_twice_area(subject) > 0.0) ||
      !(polygon_twice_area(clip) > 0.0)) {
    return {};
  }

  Polygon input;
  input.r.reserve(subject.r.size());
  input.z.reserve(subject.z.size());
  for (std::size_t k = 0; k < subject.r.size(); ++k) {
    append_vertex(input, subject.r[k], subject.z[k]);
  }
  remove_closing_duplicate(input);
  if (input.r.size() < 3U) {
    return {};
  }

  for (std::size_t edge = 0; edge < clip.r.size(); ++edge) {
    const std::size_t edge_next = (edge + 1U) % clip.r.size();
    const double ar = canonical_r(clip.r[edge]);
    const double az = clip.z[edge];
    const double br = canonical_r(clip.r[edge_next]);
    const double bz = clip.z[edge_next];
    const double tolerance = edge_tolerance(input, ar, az, br, bz);
    Polygon output;
    output.r.reserve(input.r.size() + 2U);
    output.z.reserve(input.z.size() + 2U);

    double sr = input.r.back();
    double sz = input.z.back();
    double side_s = edge_side(ar, az, br, bz, sr, sz);
    bool inside_s = side_s >= -tolerance;
    for (std::size_t k = 0; k < input.r.size(); ++k) {
      const double er = input.r[k];
      const double ez = input.z[k];
      const double side_e = edge_side(ar, az, br, bz, er, ez);
      const bool inside_e = side_e >= -tolerance;
      if (inside_s != inside_e) {
        append_line_intersection(output, sr, sz, er, ez, side_s, side_e);
      }
      if (inside_e) {
        append_vertex(output, er, ez);
      }
      sr = er;
      sz = ez;
      side_s = side_e;
      inside_s = inside_e;
    }
    remove_closing_duplicate(output);
    if (output.r.size() < 3U) {
      return {};
    }
    input = std::move(output);
  }
  return input;
}

QuadSplit split_quad_deterministic(const double r[4], const double z[4]) {
  QuadSplit result;
  double canonical[4];
  int reflex_count = 0;
  int reflex_corner = -1;
  bool degenerate = false;
  for (int k = 0; k < 4; ++k) {
    canonical[k] = canonical_r(r[k]);
    if (!std::isfinite(canonical[k]) || !std::isfinite(z[k])) {
      return result;
    }
  }
  for (int k = 0; k < 4; ++k) {
    const double jacobian = corner_jacobian(canonical, z, k);
    if (jacobian < 0.0) {
      ++reflex_count;
      reflex_corner = k;
    } else if (jacobian == 0.0) {
      degenerate = true;
    }
  }
  if (reflex_count >= 2 || degenerate) {
    return result;
  }

  const Polygon quad = make_quad(canonical, z);
  if (!(polygon_twice_area(quad) > 0.0)) {
    return result;
  }

  const int diagonal_corner = reflex_count == 1 ? reflex_corner : 0;
  const int k1 = (diagonal_corner + 1) & 3;
  const int k2 = (diagonal_corner + 2) & 3;
  const int k3 = (diagonal_corner + 3) & 3;
  result.tri[0] =
      make_triangle(canonical, z, diagonal_corner, k1, k2);
  result.tri[1] =
      make_triangle(canonical, z, diagonal_corner, k2, k3);
  result.n = 2;
  return result;
}

double quad_pair_overlap_rz_volume(const double ra[4],
                                   const double za[4],
                                   const double rc[4],
                                   const double zc[4]) {
  double canonical_target_r[4];
  for (int k = 0; k < 4; ++k) {
    canonical_target_r[k] = canonical_r(rc[k]);
  }
  const bool target_is_convex = strictly_convex_quad(canonical_target_r, zc);
  assert(target_is_convex);
  if (!target_is_convex) {
    return 0.0;
  }

  const QuadSplit source = split_quad_deterministic(ra, za);
  if (source.n == 0) {
    return 0.0;
  }
  const Polygon target = make_quad(canonical_target_r, zc);
  double overlap = 0.0;
  for (int k = 0; k < source.n; ++k) {
    overlap += polygon_rz_volume(clip_convex(source.tri[k], target));
  }
  return overlap;
}

PatchIntersection build_patch_intersection(
    const std::vector<std::array<double, 8>>& src_quads,
    const std::vector<std::array<double, 8>>& dst_quads) {
  PatchIntersection result;
  result.n_src = static_cast<int>(src_quads.size());
  result.n_dst = static_cast<int>(dst_quads.size());
  result.overlap.assign(src_quads.size() * dst_quads.size(), 0.0);
  result.src_vol.assign(src_quads.size(), 0.0);
  result.dst_vol.assign(dst_quads.size(), 0.0);

  for (std::size_t a = 0; a < src_quads.size(); ++a) {
    double r[4];
    double z[4];
    unpack_quad(src_quads[a], r, z);
    if (split_quad_deterministic(r, z).n == 0) {
      ++result.split_failures;
    }
    result.src_vol[a] = split_quad_volume(r, z);
  }
  for (std::size_t c = 0; c < dst_quads.size(); ++c) {
    double r[4];
    double z[4];
    unpack_quad(dst_quads[c], r, z);
    const bool target_is_convex = strictly_convex_quad(r, z);
    assert(target_is_convex);
    if (target_is_convex) {
      result.dst_vol[c] = split_quad_volume(r, z);
    }
  }

  for (std::size_t a = 0; a < src_quads.size(); ++a) {
    double ra[4];
    double za[4];
    unpack_quad(src_quads[a], ra, za);
    for (std::size_t c = 0; c < dst_quads.size(); ++c) {
      double rc[4];
      double zc[4];
      unpack_quad(dst_quads[c], rc, zc);
      result.overlap[a * dst_quads.size() + c] =
          quad_pair_overlap_rz_volume(ra, za, rc, zc);
    }
  }

  constexpr double tiny = std::numeric_limits<double>::min();
  for (std::size_t a = 0; a < src_quads.size(); ++a) {
    double row_sum = 0.0;
    for (std::size_t c = 0; c < dst_quads.size(); ++c) {
      row_sum += result.overlap[a * dst_quads.size() + c];
    }
    const double defect =
        std::abs(row_sum - result.src_vol[a]) /
        std::max(result.src_vol[a], tiny);
    if (a == 0U || defect > result.max_row_defect) {
      result.worst_row_index = static_cast<int>(a);
    }
    result.max_row_defect = std::max(result.max_row_defect, defect);
  }
  for (std::size_t c = 0; c < dst_quads.size(); ++c) {
    double column_sum = 0.0;
    for (std::size_t a = 0; a < src_quads.size(); ++a) {
      column_sum += result.overlap[a * dst_quads.size() + c];
    }
    const double defect =
        std::abs(column_sum - result.dst_vol[c]) /
        std::max(result.dst_vol[c], tiny);
    if (c == 0U || defect > result.max_col_defect) {
      result.worst_col_index = static_cast<int>(c);
    }
    result.max_col_defect = std::max(result.max_col_defect, defect);
  }
  return result;
}

void transfer_piecewise_constant(
    const PatchIntersection& ix,
    const std::vector<double>& src_extensive,
    std::vector<double>& dst_extensive) {
  if (ix.n_dst <= 0) {
    dst_extensive.clear();
    return;
  }
  dst_extensive.assign(static_cast<std::size_t>(ix.n_dst), 0.0);
  if (ix.n_src <= 0) {
    return;
  }

  const std::size_t n_src = std::min(
      {static_cast<std::size_t>(ix.n_src), src_extensive.size(),
       ix.src_vol.size()});
  const std::size_t n_dst = static_cast<std::size_t>(ix.n_dst);
  for (std::size_t a = 0; a < n_src; ++a) {
    if (!(ix.src_vol[a] > 0.0)) {
      continue;
    }
    const double intensive = src_extensive[a] / ix.src_vol[a];
    for (std::size_t c = 0; c < n_dst; ++c) {
      const std::size_t index = a * static_cast<std::size_t>(ix.n_dst) + c;
      if (index < ix.overlap.size()) {
        dst_extensive[c] += intensive * ix.overlap[index];
      }
    }
  }
}

void transfer_renormalized(
    const PatchIntersection& ix,
    const std::vector<double>& src_extensive,
    std::vector<double>& dst_extensive,
    std::vector<std::uint8_t>* orphaned) {
  const std::size_t n_src =
      ix.n_src > 0 ? static_cast<std::size_t>(ix.n_src) : 0U;
  const std::size_t n_dst =
      ix.n_dst > 0 ? static_cast<std::size_t>(ix.n_dst) : 0U;
  dst_extensive.assign(n_dst, 0.0);
  if (orphaned != nullptr) {
    orphaned->assign(n_src, 0U);
  }

  const std::size_t available_src = std::min(n_src, src_extensive.size());
  for (std::size_t a = 0; a < available_src; ++a) {
    double row_sum = 0.0;
    for (std::size_t c = 0; c < n_dst; ++c) {
      const std::size_t index = a * n_dst + c;
      if (index < ix.overlap.size()) {
        row_sum += ix.overlap[index];
      }
    }
    if (row_sum <= 0.0) {
      if (src_extensive[a] != 0.0 && orphaned != nullptr) {
        (*orphaned)[a] = 1U;
      }
      continue;
    }
    for (std::size_t c = 0; c < n_dst; ++c) {
      const std::size_t index = a * n_dst + c;
      if (index < ix.overlap.size()) {
        dst_extensive[c] +=
            src_extensive[a] * ix.overlap[index] / row_sum;
      }
    }
  }
}

void zonal_momentum_and_ke(const double corner_masses[4],
                           const double vr[4],
                           const double vz[4],
                           double& Pr,
                           double& Pz,
                           double& K) {
  Pr = 0.0;
  Pz = 0.0;
  K = 0.0;
  for (int k = 0; k < 4; ++k) {
    Pr += corner_masses[k] * vr[k];
    Pz += corner_masses[k] * vz[k];
    K += 0.5 * corner_masses[k] *
         (vr[k] * vr[k] + vz[k] * vz[k]);
  }
}

void distribute_corner_masses_and_momenta(
    const double m_new,
    const double Pr_new,
    const double Pz_new,
    const double corner_weights[4],
    double corner_masses_out[4],
    double corner_pr_out[4],
    double corner_pz_out[4]) {
  for (int k = 0; k < 4; ++k) {
    corner_masses_out[k] = m_new * corner_weights[k];
    corner_pr_out[k] = Pr_new * corner_weights[k];
    corner_pz_out[k] = Pz_new * corner_weights[k];
  }
}

}  // namespace tenryu::hydro::ixremap
