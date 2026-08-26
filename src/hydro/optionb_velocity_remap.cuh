#pragma once

#include <cmath>
#include <cstdint>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "hydro/optionb_subcell_remap.cuh"

namespace tenryu::hydro::optionb {

enum class NodeVelocityProjector : std::uint8_t {
  FREE = 0,
  RZ_AXIS = 1,
  PINNED_OR_NODE_CENTER = 2,
};

enum class VelocityRemapStatus : std::uint8_t {
  OK = 0,
  NEEDS_EXPANDED_STENCIL = 1,
  INVALID_INPUT = 2,
};

struct VelocityMomentumPacketFctResult {
  VelocityRemapStatus status;
  double alpha;
  double alpha_mass;
  double alpha_momentum_r;
  double alpha_momentum_z;
};

enum class AffineHourglassFilterStatus : std::uint8_t {
  OK = 0,
  INVALID_INPUT = 1,
  DEGENERATE_BASIS = 2,
};

enum class AffineHourglassBasis : std::uint8_t {
  FULL_AFFINE = 0,
  RADIAL_ODD = 1,
  Z_LINEAR = 2,
  NONE = 3,
};

constexpr double kAffineHourglassAlphaMax = 0.25;
constexpr double kAffineHourglassC = 0.5;
constexpr double kAffineHourglassEta0 = 0.05;
constexpr double kAffineHourglassEps = 1.0e-30;

struct AffineHourglassComponentResult {
  AffineHourglassFilterStatus status;
  AffineHourglassBasis basis;
  double beta0;
  double beta_r;
  double beta_z;
  double k_hg;
  double k_aff;
  double eta;
  double alpha_raw;
  double alpha_bound;
  double alpha;
  double residual_mass;
  double residual_r;
  double residual_z;
};

struct AffineHourglassFilterResult {
  AffineHourglassComponentResult ur;
  AffineHourglassComponentResult uz;
};

namespace detail {

__host__ __device__ inline AffineHourglassComponentResult
make_affine_hourglass_component_result(
    const AffineHourglassFilterStatus status,
    const AffineHourglassBasis basis) {
  AffineHourglassComponentResult result;
  result.status = status;
  result.basis = basis;
  result.beta0 = 0.0;
  result.beta_r = 0.0;
  result.beta_z = 0.0;
  result.k_hg = 0.0;
  result.k_aff = 0.0;
  result.eta = 0.0;
  result.alpha_raw = 0.0;
  result.alpha_bound = 0.0;
  result.alpha = 0.0;
  result.residual_mass = 0.0;
  result.residual_r = 0.0;
  result.residual_z = 0.0;
  return result;
}

__host__ __device__ inline VelocityMomentumPacketFctResult
make_velocity_momentum_packet_fct_result(const VelocityRemapStatus status) {
  VelocityMomentumPacketFctResult result;
  result.status = status;
  result.alpha = 0.0;
  result.alpha_mass = 0.0;
  result.alpha_momentum_r = 0.0;
  result.alpha_momentum_z = 0.0;
  return result;
}

__host__ __device__ inline NodeVelocityProjector node_projector_at(
    const NodeVelocityProjector* projector,
    const int i) {
  return projector == nullptr ? NodeVelocityProjector::FREE : projector[i];
}

__host__ __device__ inline bool valid_cell_nverts(const int nverts) {
  return nverts >= 3;
}

__host__ __device__ inline bool point_in_convex_hull(const double* r,
                                                     const double* z,
                                                     const int nverts,
                                                     const double rq,
                                                     const double zq) {
  if (!valid_cell_nverts(nverts) ||
      !tenryu::hydro::rz::finite_double(rq) ||
      !tenryu::hydro::rz::finite_double(zq)) {
    return false;
  }

  const double area2 = tenryu::hydro::rz::rz_polygon_area2_exact(r, z, nverts);
  const double scale = polygon_length_scale(r, z, nverts, rq, zq);
  const double cross_tol = 1024.0 * kDoubleEps * scale * scale;
  if (!(fabs(area2) > cross_tol) ||
      !tenryu::hydro::rz::finite_double(area2)) {
    return false;
  }

  const double sign = area2 >= 0.0 ? 1.0 : -1.0;
  for (int i = 0; i < nverts; ++i) {
    if (!tenryu::hydro::rz::finite_double(r[i]) ||
        !tenryu::hydro::rz::finite_double(z[i])) {
      return false;
    }
    const int ip = (i + 1 == nverts) ? 0 : i + 1;
    const double er = r[ip] - r[i];
    const double ez = z[ip] - z[i];
    const double qr = rq - r[i];
    const double qz = zq - z[i];
    const double cross = er * qz - ez * qr;
    if (sign * cross < -cross_tol) {
      return false;
    }
  }
  return true;
}

__host__ __device__ inline bool barycentric_query_ok(const double* r,
                                                     const double* z,
                                                     const int nverts,
                                                     const double rq,
                                                     const double zq,
                                                     const double* lambda) {
  double sum = 0.0;
  double r_recon = 0.0;
  double z_recon = 0.0;
  for (int i = 0; i < nverts; ++i) {
    if (!tenryu::hydro::rz::finite_double(lambda[i])) {
      return false;
    }
    sum += lambda[i];
    r_recon += lambda[i] * r[i];
    z_recon += lambda[i] * z[i];
  }

  const double scale = polygon_length_scale(r, z, nverts, rq, zq);
  const double sum_tol = 1024.0 * kDoubleEps;
  const double coord_tol = 1.0e-12 * max2(1.0, scale);
  return fabs(sum - 1.0) <= sum_tol &&
         fabs(r_recon - rq) <= coord_tol &&
         fabs(z_recon - zq) <= coord_tol;
}

__host__ __device__ inline void projected_velocity(
    const double ur_in,
    const double uz_in,
    const NodeVelocityProjector projector,
    double* ur_out,
    double* uz_out) {
  *ur_out = ur_in;
  *uz_out = uz_in;
  switch (projector) {
    case NodeVelocityProjector::RZ_AXIS:
      *ur_out = 0.0;
      break;
    case NodeVelocityProjector::PINNED_OR_NODE_CENTER:
      *ur_out = 0.0;
      *uz_out = 0.0;
      break;
    case NodeVelocityProjector::FREE:
      break;
  }
}

__host__ __device__ inline bool solve_spd3_cholesky(
    const double g00,
    const double g01,
    const double g02,
    const double g11,
    const double g12,
    const double g22,
    const double b0,
    const double b1,
    const double b2,
    double* x0,
    double* x1,
    double* x2) {
  const double scale =
      max2(max2(max2(fabs(g00), fabs(g01)), max2(fabs(g02), fabs(g11))),
           max2(fabs(g12), fabs(g22)));
  const double tol = 1024.0 * kDoubleEps * max2(scale, 1.0e-300);
  if (!(g00 > tol) || !tenryu::hydro::rz::finite_double(g00)) {
    return false;
  }

  const double l00 = sqrt(g00);
  const double l10 = g01 / l00;
  const double l20 = g02 / l00;
  const double d11 = g11 - l10 * l10;
  if (!(d11 > tol) || !tenryu::hydro::rz::finite_double(d11)) {
    return false;
  }
  const double l11 = sqrt(d11);
  const double l21 = (g12 - l20 * l10) / l11;
  const double d22 = g22 - l20 * l20 - l21 * l21;
  if (!(d22 > tol) || !tenryu::hydro::rz::finite_double(d22)) {
    return false;
  }
  const double l22 = sqrt(d22);

  const double y0 = b0 / l00;
  const double y1 = (b1 - l10 * y0) / l11;
  const double y2 = (b2 - l20 * y0 - l21 * y1) / l22;

  *x2 = y2 / l22;
  *x1 = (y1 - l21 * (*x2)) / l11;
  *x0 = (y0 - l10 * (*x1) - l20 * (*x2)) / l00;
  return tenryu::hydro::rz::finite_double(*x0) &&
         tenryu::hydro::rz::finite_double(*x1) &&
         tenryu::hydro::rz::finite_double(*x2);
}

__host__ __device__ inline bool solve_radial_odd_fit(const double* r,
                                                     const double* m_corner,
                                                     const double* v,
                                                     const int nverts,
                                                     double* beta_r) {
  double g = 0.0;
  double b = 0.0;
  for (int a = 0; a < nverts; ++a) {
    if (m_corner[a] > 0.0 && tenryu::hydro::rz::finite_double(m_corner[a])) {
      g += m_corner[a] * r[a] * r[a];
      b += m_corner[a] * r[a] * v[a];
    }
  }
  if (!(g > 0.0) || !tenryu::hydro::rz::finite_double(g) ||
      !tenryu::hydro::rz::finite_double(b)) {
    return false;
  }
  *beta_r = b / g;
  return tenryu::hydro::rz::finite_double(*beta_r);
}

__host__ __device__ inline bool solve_z_linear_fit(const double* z,
                                                   const double* m_corner,
                                                   const double* v,
                                                   const int nverts,
                                                   double* beta0,
                                                   double* beta_z) {
  double g00 = 0.0;
  double g01 = 0.0;
  double g11 = 0.0;
  double b0 = 0.0;
  double b1 = 0.0;
  for (int a = 0; a < nverts; ++a) {
    if (m_corner[a] > 0.0 && tenryu::hydro::rz::finite_double(m_corner[a])) {
      g00 += m_corner[a];
      g01 += m_corner[a] * z[a];
      g11 += m_corner[a] * z[a] * z[a];
      b0 += m_corner[a] * v[a];
      b1 += m_corner[a] * z[a] * v[a];
    }
  }

  const double det = g00 * g11 - g01 * g01;
  const double scale = max2(fabs(g00 * g11), fabs(g01 * g01));
  const double tol = 1024.0 * kDoubleEps * max2(scale, 1.0e-300);
  if (!(det > tol) || !tenryu::hydro::rz::finite_double(det)) {
    return false;
  }
  *beta0 = (b0 * g11 - b1 * g01) / det;
  *beta_z = (g00 * b1 - g01 * b0) / det;
  return tenryu::hydro::rz::finite_double(*beta0) &&
         tenryu::hydro::rz::finite_double(*beta_z);
}

__host__ __device__ inline AffineHourglassComponentResult
fit_affine_hourglass_component(const double* r,
                               const double* z,
                               const int nverts,
                               const double* m_corner,
                               const double* p,
                               const bool radial_component,
                               const double K_floor,
                               const double alpha_max,
                               const double C_hg,
                               const double eta0,
                               const double eps,
                               double* v,
                               double* v_aff,
                               double* h) {
  AffineHourglassComponentResult result =
      make_affine_hourglass_component_result(
          AffineHourglassFilterStatus::OK, AffineHourglassBasis::NONE);
  if (!valid_cell_nverts(nverts) || r == nullptr || z == nullptr ||
      m_corner == nullptr || p == nullptr || v == nullptr ||
      v_aff == nullptr || h == nullptr || !(K_floor >= 0.0) ||
      !(alpha_max >= 0.0) || !(C_hg >= 0.0) || !(eta0 >= 0.0) ||
      !(eps > 0.0) || !tenryu::hydro::rz::finite_double(K_floor) ||
      !tenryu::hydro::rz::finite_double(alpha_max) ||
      !tenryu::hydro::rz::finite_double(C_hg) ||
      !tenryu::hydro::rz::finite_double(eta0) ||
      !tenryu::hydro::rz::finite_double(eps)) {
    return make_affine_hourglass_component_result(
        AffineHourglassFilterStatus::INVALID_INPUT, AffineHourglassBasis::NONE);
  }

  double g00 = 0.0;
  double g01 = 0.0;
  double g02 = 0.0;
  double g11 = 0.0;
  double g12 = 0.0;
  double g22 = 0.0;
  double b0 = 0.0;
  double b1 = 0.0;
  double b2 = 0.0;
  double m_sum = 0.0;
  double mv_sum = 0.0;
  int active_count = 0;

  for (int a = 0; a < nverts; ++a) {
    v[a] = 0.0;
    v_aff[a] = 0.0;
    h[a] = 0.0;
    if (!tenryu::hydro::rz::finite_double(r[a]) ||
        !tenryu::hydro::rz::finite_double(z[a]) ||
        !tenryu::hydro::rz::finite_double(m_corner[a]) ||
        !tenryu::hydro::rz::finite_double(p[a])) {
      return make_affine_hourglass_component_result(
          AffineHourglassFilterStatus::INVALID_INPUT,
          AffineHourglassBasis::NONE);
    }
    if (!(m_corner[a] > 0.0)) {
      continue;
    }

    v[a] = p[a] / m_corner[a];
    if (!tenryu::hydro::rz::finite_double(v[a])) {
      return make_affine_hourglass_component_result(
          AffineHourglassFilterStatus::INVALID_INPUT,
          AffineHourglassBasis::NONE);
    }
    ++active_count;
    m_sum += m_corner[a];
    mv_sum += m_corner[a] * v[a];
    g00 += m_corner[a];
    g01 += m_corner[a] * r[a];
    g02 += m_corner[a] * z[a];
    g11 += m_corner[a] * r[a] * r[a];
    g12 += m_corner[a] * r[a] * z[a];
    g22 += m_corner[a] * z[a] * z[a];
    b0 += m_corner[a] * v[a];
    b1 += m_corner[a] * r[a] * v[a];
    b2 += m_corner[a] * z[a] * v[a];
  }

  if (active_count <= 0 || !(m_sum > 0.0) ||
      !tenryu::hydro::rz::finite_double(m_sum)) {
    return make_affine_hourglass_component_result(
        AffineHourglassFilterStatus::DEGENERATE_BASIS,
        AffineHourglassBasis::NONE);
  }

  if (solve_spd3_cholesky(g00,
                          g01,
                          g02,
                          g11,
                          g12,
                          g22,
                          b0,
                          b1,
                          b2,
                          &result.beta0,
                          &result.beta_r,
                          &result.beta_z)) {
    result.basis = AffineHourglassBasis::FULL_AFFINE;
  } else if (radial_component) {
    result.beta0 = 0.0;
    result.beta_z = 0.0;
    if (!solve_radial_odd_fit(r, m_corner, v, nverts, &result.beta_r)) {
      return make_affine_hourglass_component_result(
          AffineHourglassFilterStatus::DEGENERATE_BASIS,
          AffineHourglassBasis::NONE);
    }
    result.basis = AffineHourglassBasis::RADIAL_ODD;
  } else {
    result.beta_r = 0.0;
    if (!solve_z_linear_fit(
            z, m_corner, v, nverts, &result.beta0, &result.beta_z)) {
      return make_affine_hourglass_component_result(
          AffineHourglassFilterStatus::DEGENERATE_BASIS,
          AffineHourglassBasis::NONE);
    }
    result.basis = AffineHourglassBasis::Z_LINEAR;
  }

  const double vbar = mv_sum / m_sum;
  for (int a = 0; a < nverts; ++a) {
    if (!(m_corner[a] > 0.0)) {
      continue;
    }
    v_aff[a] = result.beta0 + result.beta_r * r[a] + result.beta_z * z[a];
    h[a] = v[a] - v_aff[a];
    if (active_count == 3 &&
        result.basis == AffineHourglassBasis::FULL_AFFINE) {
      v_aff[a] = v[a];
      h[a] = 0.0;
    }
    result.k_hg += 0.5 * m_corner[a] * h[a] * h[a];
    const double dv_aff = v_aff[a] - vbar;
    result.k_aff += 0.5 * m_corner[a] * dv_aff * dv_aff;
    result.residual_mass += m_corner[a] * h[a];
    result.residual_r += m_corner[a] * r[a] * h[a];
    result.residual_z += m_corner[a] * z[a] * h[a];
  }

  if (!(result.k_hg > 0.0)) {
    return result;
  }

  const double denom = result.k_aff + K_floor;
  if (denom > 0.0 && tenryu::hydro::rz::finite_double(denom)) {
    result.eta = result.k_hg / denom;
  } else {
    result.eta = 1.0e300;
  }
  if (!tenryu::hydro::rz::finite_double(result.eta)) {
    result.eta = 1.0e300;
  }
  if (result.eta > eta0) {
    const double alpha =
        C_hg * (result.eta - eta0) / (result.eta + eps);
    result.alpha_raw = min2(alpha_max, max2(0.0, alpha));
  }
  return result;
}

__host__ __device__ inline double component_alpha_bound(
    const int nverts,
    const double* m_corner,
    const double* v,
    const double* h,
    const double* v_min,
    const double* v_max) {
  double cell_min = 0.0;
  double cell_max = 0.0;
  bool have_cell_bound = false;
  if (v_min == nullptr || v_max == nullptr) {
    for (int a = 0; a < nverts; ++a) {
      if (!(m_corner[a] > 0.0)) {
        continue;
      }
      if (!have_cell_bound) {
        cell_min = v[a];
        cell_max = v[a];
        have_cell_bound = true;
      } else {
        cell_min = min2(cell_min, v[a]);
        cell_max = max2(cell_max, v[a]);
      }
    }
  }
  if (!have_cell_bound && (v_min == nullptr || v_max == nullptr)) {
    return 0.0;
  }

  double alpha_bound = 1.0e300;
  for (int a = 0; a < nverts; ++a) {
    if (!(m_corner[a] > 0.0)) {
      continue;
    }
    const double lo = v_min == nullptr ? cell_min : v_min[a];
    const double hi = v_max == nullptr ? cell_max : v_max[a];
    if (!tenryu::hydro::rz::finite_double(lo) ||
        !tenryu::hydro::rz::finite_double(hi) || lo > hi) {
      return 0.0;
    }
    const double bound_tol =
        1024.0 * kDoubleEps * max2(1.0, max2(fabs(lo), fabs(hi)));
    if (v[a] < lo - bound_tol || v[a] > hi + bound_tol) {
      return 0.0;
    }
    if (h[a] > 0.0) {
      alpha_bound = min2(alpha_bound, max2(0.0, (v[a] - lo) / h[a]));
    } else if (h[a] < 0.0) {
      alpha_bound = min2(alpha_bound, max2(0.0, (hi - v[a]) / (-h[a])));
    }
  }
  return alpha_bound;
}

__host__ __device__ inline AffineHourglassComponentResult
apply_affine_hourglass_component(const double* r,
                                 const double* z,
                                 const int nverts,
                                 const double* m_corner,
                                 double* p,
                                 const bool radial_component,
                                 const double K_floor,
                                 const double* v_min,
                                 const double* v_max,
                                 const double alpha_max,
                                 const double C_hg,
                                 const double eta0,
                                 const double eps,
                                 double* v,
                                 double* v_aff,
                                 double* h) {
  AffineHourglassComponentResult result =
      fit_affine_hourglass_component(r,
                                     z,
                                     nverts,
                                     m_corner,
                                     p,
                                     radial_component,
                                     K_floor,
                                     alpha_max,
                                     C_hg,
                                     eta0,
                                     eps,
                                     v,
                                     v_aff,
                                     h);
  if (result.status != AffineHourglassFilterStatus::OK ||
      !(result.alpha_raw > 0.0)) {
    return result;
  }

  result.alpha_bound =
      component_alpha_bound(nverts, m_corner, v, h, v_min, v_max);
  result.alpha = min2(result.alpha_raw, result.alpha_bound);
  if (!(result.alpha > 0.0)) {
    return result;
  }
  for (int a = 0; a < nverts; ++a) {
    if (m_corner[a] > 0.0) {
      p[a] = m_corner[a] * (v[a] - result.alpha * h[a]);
    }
  }
  return result;
}

__host__ __device__ inline double clamp_to_bounds(const double value,
                                                  const double lo,
                                                  const double hi) {
  return max2(lo, min2(hi, value));
}

__host__ __device__ inline bool packet_low_order_value_admissible(
    const double value,
    const double lo,
    const double hi) {
  const double tol =
      1024.0 * kDoubleEps * max2(1.0, max2(fabs(lo), fabs(hi)));
  return tenryu::hydro::rz::finite_double(value) &&
         tenryu::hydro::rz::finite_double(lo) &&
         tenryu::hydro::rz::finite_double(hi) && lo <= hi &&
         fabs(value - clamp_to_bounds(value, lo, hi)) <= tol;
}

__host__ __device__ inline bool fct_receiver_velocity_alpha_bound(
    const double m_final,
    const double p_lo,
    const double p_antidiffusive,
    const double v_min,
    const double v_max,
    double* alpha_bound) {
  if (!tenryu::hydro::rz::finite_double(m_final) ||
      !tenryu::hydro::rz::finite_double(p_lo) ||
      !tenryu::hydro::rz::finite_double(p_antidiffusive) ||
      !tenryu::hydro::rz::finite_double(v_min) ||
      !tenryu::hydro::rz::finite_double(v_max) || v_min > v_max) {
    return false;
  }
  if (!(m_final > 0.0)) {
    return fabs(p_lo) <=
           1024.0 * kDoubleEps * max2(1.0, fabs(p_lo));
  }

  const double p_min = v_min * m_final;
  const double p_max = v_max * m_final;
  const double tol =
      1024.0 * kDoubleEps * max2(1.0, max2(fabs(p_min), fabs(p_max)));
  if (p_lo < p_min - tol || p_lo > p_max + tol) {
    return false;
  }

  if (p_antidiffusive > 0.0) {
    const double bound = (p_max - p_lo) / p_antidiffusive;
    if (!tenryu::hydro::rz::finite_double(bound)) {
      return false;
    }
    *alpha_bound = min2(*alpha_bound, max2(0.0, bound));
  } else if (p_antidiffusive < 0.0) {
    const double bound = (p_lo - p_min) / (-p_antidiffusive);
    if (!tenryu::hydro::rz::finite_double(bound)) {
      return false;
    }
    *alpha_bound = min2(*alpha_bound, max2(0.0, bound));
  }
  return true;
}

}  // namespace detail

__host__ __device__ inline void apply_node_velocity_projector(
    const NodeVelocityProjector projector,
    double* ur,
    double* uz) {
  double projected_ur = 0.0;
  double projected_uz = 0.0;
  detail::projected_velocity(*ur, *uz, projector, &projected_ur, &projected_uz);
  *ur = projected_ur;
  *uz = projected_uz;
}

__host__ __device__ inline void gather_corner_momentum(
    const double m_cell,
    const double* r,
    const double* z,
    const int nverts,
    const double* ur,
    const double* uz,
    const NodeVelocityProjector* projector,
    double* m_corner,
    double* p_r,
    double* p_z) {
  if (!detail::valid_cell_nverts(nverts)) {
    return;
  }

  first_moment_corner_masses(m_cell, r, z, nverts, m_corner);
  for (int i = 0; i < nverts; ++i) {
    double ui_r = 0.0;
    double ui_z = 0.0;
    detail::projected_velocity(ur[i],
                               uz[i],
                               detail::node_projector_at(projector, i),
                               &ui_r,
                               &ui_z);
    p_r[i] = m_corner[i] * ui_r;
    p_z[i] = m_corner[i] * ui_z;
  }
}

// If v_min/v_max are omitted, the current cell's own corner
// velocity range as a placeholder bound; expanded donor-stencil bounds are
// expected to replace that at wiring time.
__host__ __device__ inline AffineHourglassFilterResult
apply_affine_orthogonal_hourglass_filter(
    const double* r,
    const double* z,
    const int nverts,
    const double* m_corner,
    double* p_r,
    double* p_z,
    const double K_floor,
    double* v_scratch,
    double* v_aff_scratch,
    double* h_scratch,
    const double* v_min_r = nullptr,
    const double* v_max_r = nullptr,
    const double* v_min_z = nullptr,
    const double* v_max_z = nullptr,
    const double alpha_max = kAffineHourglassAlphaMax,
    const double C_hg = kAffineHourglassC,
    const double eta0 = kAffineHourglassEta0,
    const double eps = kAffineHourglassEps) {
  AffineHourglassFilterResult result;
  result.ur = detail::apply_affine_hourglass_component(r,
                                                       z,
                                                       nverts,
                                                       m_corner,
                                                       p_r,
                                                       true,
                                                       K_floor,
                                                       v_min_r,
                                                       v_max_r,
                                                       alpha_max,
                                                       C_hg,
                                                       eta0,
                                                       eps,
                                                       v_scratch,
                                                       v_aff_scratch,
                                                       h_scratch);
  result.uz = detail::apply_affine_hourglass_component(r,
                                                       z,
                                                       nverts,
                                                       m_corner,
                                                       p_z,
                                                       false,
                                                       K_floor,
                                                       v_min_z,
                                                       v_max_z,
                                                       alpha_max,
                                                       C_hg,
                                                       eta0,
                                                       eps,
                                                       v_scratch,
                                                       v_aff_scratch,
                                                       h_scratch);
  return result;
}

__host__ __device__ inline VelocityRemapStatus remap_velocity_momentum_packet(
    const double dm_q,
    const double rq,
    const double zq,
    const double* donor_r,
    const double* donor_z,
    const int donor_nverts,
    const double* donor_ur,
    const double* donor_uz,
    const NodeVelocityProjector* donor_projector,
    double* donor_m_corner,
    double* donor_p_r,
    double* donor_p_z,
    const double* receiver_r,
    const double* receiver_z,
    const int receiver_nverts,
    double* receiver_m_corner,
    double* receiver_p_r,
    double* receiver_p_z,
    double* lambda_d_packet,
    double* lambda_r_packet,
    double* lambda_d_vertex,
    double* u_d_r_at_receiver,
    double* u_d_z_at_receiver) {
  if (!(dm_q > 0.0) || !tenryu::hydro::rz::finite_double(dm_q) ||
      !detail::valid_cell_nverts(donor_nverts) ||
      !detail::valid_cell_nverts(receiver_nverts) ||
      lambda_d_packet == nullptr || lambda_r_packet == nullptr ||
      lambda_d_vertex == nullptr || u_d_r_at_receiver == nullptr ||
      u_d_z_at_receiver == nullptr) {
    return VelocityRemapStatus::INVALID_INPUT;
  }

  if (!detail::point_in_convex_hull(donor_r, donor_z, donor_nverts, rq, zq) ||
      !detail::point_in_convex_hull(
          receiver_r, receiver_z, receiver_nverts, rq, zq)) {
    return VelocityRemapStatus::NEEDS_EXPANDED_STENCIL;
  }

  for (int b = 0; b < receiver_nverts; ++b) {
    if (!detail::point_in_convex_hull(donor_r,
                                      donor_z,
                                      donor_nverts,
                                      receiver_r[b],
                                      receiver_z[b])) {
      return VelocityRemapStatus::NEEDS_EXPANDED_STENCIL;
    }
  }

  barycentric_weights(
      donor_r, donor_z, donor_nverts, rq, zq, lambda_d_packet);
  if (!detail::barycentric_query_ok(
          donor_r, donor_z, donor_nverts, rq, zq, lambda_d_packet)) {
    return VelocityRemapStatus::INVALID_INPUT;
  }
  barycentric_weights(
      receiver_r, receiver_z, receiver_nverts, rq, zq, lambda_r_packet);
  if (!detail::barycentric_query_ok(
          receiver_r, receiver_z, receiver_nverts, rq, zq, lambda_r_packet)) {
    return VelocityRemapStatus::INVALID_INPUT;
  }

  double u_q_r = 0.0;
  double u_q_z = 0.0;
  for (int a = 0; a < donor_nverts; ++a) {
    double ui_r = 0.0;
    double ui_z = 0.0;
    detail::projected_velocity(donor_ur[a],
                               donor_uz[a],
                               detail::node_projector_at(donor_projector, a),
                               &ui_r,
                               &ui_z);
    u_q_r += lambda_d_packet[a] * ui_r;
    u_q_z += lambda_d_packet[a] * ui_z;
  }

  double u_avg_r = 0.0;
  double u_avg_z = 0.0;
  for (int b = 0; b < receiver_nverts; ++b) {
    barycentric_weights(donor_r,
                        donor_z,
                        donor_nverts,
                        receiver_r[b],
                        receiver_z[b],
                        lambda_d_vertex);
    if (!detail::barycentric_query_ok(donor_r,
                                      donor_z,
                                      donor_nverts,
                                      receiver_r[b],
                                      receiver_z[b],
                                      lambda_d_vertex)) {
      return VelocityRemapStatus::INVALID_INPUT;
    }

    double u_db_r = 0.0;
    double u_db_z = 0.0;
    for (int a = 0; a < donor_nverts; ++a) {
      double ui_r = 0.0;
      double ui_z = 0.0;
      detail::projected_velocity(donor_ur[a],
                                 donor_uz[a],
                                 detail::node_projector_at(donor_projector, a),
                                 &ui_r,
                                 &ui_z);
      u_db_r += lambda_d_vertex[a] * ui_r;
      u_db_z += lambda_d_vertex[a] * ui_z;
    }
    u_d_r_at_receiver[b] = u_db_r;
    u_d_z_at_receiver[b] = u_db_z;
    u_avg_r += lambda_r_packet[b] * u_db_r;
    u_avg_z += lambda_r_packet[b] * u_db_z;
  }

  const double correction_r = u_q_r - u_avg_r;
  const double correction_z = u_q_z - u_avg_z;

  for (int a = 0; a < donor_nverts; ++a) {
    double ui_r = 0.0;
    double ui_z = 0.0;
    detail::projected_velocity(donor_ur[a],
                               donor_uz[a],
                               detail::node_projector_at(donor_projector, a),
                               &ui_r,
                               &ui_z);
    const double dm_a = dm_q * lambda_d_packet[a];
    donor_m_corner[a] -= dm_a;
    donor_p_r[a] -= dm_a * ui_r;
    donor_p_z[a] -= dm_a * ui_z;
  }

  for (int b = 0; b < receiver_nverts; ++b) {
    const double dm_b = dm_q * lambda_r_packet[b];
    receiver_m_corner[b] += dm_b;
    receiver_p_r[b] += dm_b * (u_d_r_at_receiver[b] + correction_r);
    receiver_p_z[b] += dm_b * (u_d_z_at_receiver[b] + correction_z);
  }

  return VelocityRemapStatus::OK;
}

__host__ __device__ inline VelocityMomentumPacketFctResult
remap_velocity_momentum_packet_fct(
    const double dm_q,
    const double rq,
    const double zq,
    const double* donor_r,
    const double* donor_z,
    const int donor_nverts,
    const double* donor_ur,
    const double* donor_uz,
    const NodeVelocityProjector* donor_projector,
    double* donor_m_corner,
    double* donor_p_r,
    double* donor_p_z,
    const double* receiver_r,
    const double* receiver_z,
    const int receiver_nverts,
    double* receiver_m_corner,
    double* receiver_p_r,
    double* receiver_p_z,
    double* lambda_d_packet,
    double* lambda_r_packet,
    double* lambda_d_vertex,
    double* u_d_r_at_receiver,
    double* u_d_z_at_receiver,
    const bool disable_fct_limiter = false,
    // Basis-coherent donor distribution (TENRYU_I1B_OPTIONB_COHERENT):
    // when non-null, the donor-side mass removal uses these FIXED pre-remap
    // basis corner shares (sum 1) instead of the packet-centroid barycentric
    // shares. Combined with the per-cell outgoing mass-flux scale this makes
    // donor corner masses unconditionally non-negative under basis seeding.
    // The packet's exchanged momentum stays dm_q*u_q (u_q is still the
    // lambda_d-interpolated donor velocity); the per-corner removal carries
    // a uniform velocity-content correction (u_q - u_tilde) so every donor
    // corner velocity shifts identically and uniform flow is preserved.
    const double* donor_sigma = nullptr) {
  if (!(dm_q > 0.0) || !tenryu::hydro::rz::finite_double(dm_q) ||
      donor_r == nullptr || donor_z == nullptr || donor_ur == nullptr ||
      donor_uz == nullptr || donor_m_corner == nullptr ||
      donor_p_r == nullptr || donor_p_z == nullptr ||
      receiver_r == nullptr || receiver_z == nullptr ||
      receiver_m_corner == nullptr || receiver_p_r == nullptr ||
      receiver_p_z == nullptr || !detail::valid_cell_nverts(donor_nverts) ||
      !detail::valid_cell_nverts(receiver_nverts) ||
      lambda_d_packet == nullptr || lambda_r_packet == nullptr ||
      lambda_d_vertex == nullptr || u_d_r_at_receiver == nullptr ||
      u_d_z_at_receiver == nullptr) {
    return detail::make_velocity_momentum_packet_fct_result(
        VelocityRemapStatus::INVALID_INPUT);
  }

  if (!detail::point_in_convex_hull(donor_r, donor_z, donor_nverts, rq, zq) ||
      !detail::point_in_convex_hull(
          receiver_r, receiver_z, receiver_nverts, rq, zq)) {
    return detail::make_velocity_momentum_packet_fct_result(
        VelocityRemapStatus::NEEDS_EXPANDED_STENCIL);
  }

  for (int b = 0; b < receiver_nverts; ++b) {
    if (!detail::point_in_convex_hull(donor_r,
                                      donor_z,
                                      donor_nverts,
                                      receiver_r[b],
                                      receiver_z[b])) {
      return detail::make_velocity_momentum_packet_fct_result(
          VelocityRemapStatus::NEEDS_EXPANDED_STENCIL);
    }
  }

  barycentric_weights(
      donor_r, donor_z, donor_nverts, rq, zq, lambda_d_packet);
  if (!detail::barycentric_query_ok(
          donor_r, donor_z, donor_nverts, rq, zq, lambda_d_packet)) {
    return detail::make_velocity_momentum_packet_fct_result(
        VelocityRemapStatus::INVALID_INPUT);
  }
  barycentric_weights(
      receiver_r, receiver_z, receiver_nverts, rq, zq, lambda_r_packet);
  if (!detail::barycentric_query_ok(
          receiver_r, receiver_z, receiver_nverts, rq, zq, lambda_r_packet)) {
    return detail::make_velocity_momentum_packet_fct_result(
        VelocityRemapStatus::INVALID_INPUT);
  }

  double u_q_r = 0.0;
  double u_q_z = 0.0;
  double u_tilde_r = 0.0;
  double u_tilde_z = 0.0;
  double donor_min_ur = 0.0;
  double donor_max_ur = 0.0;
  double donor_min_uz = 0.0;
  double donor_max_uz = 0.0;
  for (int a = 0; a < donor_nverts; ++a) {
    if (!tenryu::hydro::rz::finite_double(donor_m_corner[a]) ||
        !tenryu::hydro::rz::finite_double(donor_p_r[a]) ||
        !tenryu::hydro::rz::finite_double(donor_p_z[a])) {
      return detail::make_velocity_momentum_packet_fct_result(
          VelocityRemapStatus::INVALID_INPUT);
    }
    double ui_r = 0.0;
    double ui_z = 0.0;
    detail::projected_velocity(donor_ur[a],
                               donor_uz[a],
                               detail::node_projector_at(donor_projector, a),
                               &ui_r,
                               &ui_z);
    if (!tenryu::hydro::rz::finite_double(ui_r) ||
        !tenryu::hydro::rz::finite_double(ui_z)) {
      return detail::make_velocity_momentum_packet_fct_result(
          VelocityRemapStatus::INVALID_INPUT);
    }
    if (a == 0) {
      donor_min_ur = ui_r;
      donor_max_ur = ui_r;
      donor_min_uz = ui_z;
      donor_max_uz = ui_z;
    } else {
      donor_min_ur = detail::min2(donor_min_ur, ui_r);
      donor_max_ur = detail::max2(donor_max_ur, ui_r);
      donor_min_uz = detail::min2(donor_min_uz, ui_z);
      donor_max_uz = detail::max2(donor_max_uz, ui_z);
    }
    u_q_r += lambda_d_packet[a] * ui_r;
    u_q_z += lambda_d_packet[a] * ui_z;
    if (donor_sigma != nullptr) {
      if (!tenryu::hydro::rz::finite_double(donor_sigma[a]) ||
          donor_sigma[a] < 0.0) {
        return detail::make_velocity_momentum_packet_fct_result(
            VelocityRemapStatus::INVALID_INPUT);
      }
      u_tilde_r += donor_sigma[a] * ui_r;
      u_tilde_z += donor_sigma[a] * ui_z;
    }
  }

  if (!detail::packet_low_order_value_admissible(
          u_q_r, donor_min_ur, donor_max_ur) ||
      !detail::packet_low_order_value_admissible(
          u_q_z, donor_min_uz, donor_max_uz)) {
    return detail::make_velocity_momentum_packet_fct_result(
        VelocityRemapStatus::INVALID_INPUT);
  }

  double u_avg_r = 0.0;
  double u_avg_z = 0.0;
  for (int b = 0; b < receiver_nverts; ++b) {
    if (!tenryu::hydro::rz::finite_double(receiver_m_corner[b]) ||
        !tenryu::hydro::rz::finite_double(receiver_p_r[b]) ||
        !tenryu::hydro::rz::finite_double(receiver_p_z[b])) {
      return detail::make_velocity_momentum_packet_fct_result(
          VelocityRemapStatus::INVALID_INPUT);
    }
    barycentric_weights(donor_r,
                        donor_z,
                        donor_nverts,
                        receiver_r[b],
                        receiver_z[b],
                        lambda_d_vertex);
    if (!detail::barycentric_query_ok(donor_r,
                                      donor_z,
                                      donor_nverts,
                                      receiver_r[b],
                                      receiver_z[b],
                                      lambda_d_vertex)) {
      return detail::make_velocity_momentum_packet_fct_result(
          VelocityRemapStatus::INVALID_INPUT);
    }

    double u_db_r = 0.0;
    double u_db_z = 0.0;
    for (int a = 0; a < donor_nverts; ++a) {
      double ui_r = 0.0;
      double ui_z = 0.0;
      detail::projected_velocity(donor_ur[a],
                                 donor_uz[a],
                                 detail::node_projector_at(donor_projector, a),
                                 &ui_r,
                                 &ui_z);
      u_db_r += lambda_d_vertex[a] * ui_r;
      u_db_z += lambda_d_vertex[a] * ui_z;
    }
    u_d_r_at_receiver[b] = u_db_r;
    u_d_z_at_receiver[b] = u_db_z;
    u_avg_r += lambda_r_packet[b] * u_db_r;
    u_avg_z += lambda_r_packet[b] * u_db_z;
  }

  const double correction_r = u_q_r - u_avg_r;
  const double correction_z = u_q_z - u_avg_z;

  VelocityMomentumPacketFctResult result;
  result.status = VelocityRemapStatus::OK;
  result.alpha = 1.0;
  result.alpha_mass = 1.0;
  result.alpha_momentum_r = 1.0;
  result.alpha_momentum_z = 1.0;

  for (int a = 0; a < donor_nverts; ++a) {
    const double dm_a =
        dm_q *
        (donor_sigma != nullptr ? donor_sigma[a] : lambda_d_packet[a]);
    const double m_final = donor_m_corner[a] - dm_a;
    const double tol =
        1024.0 * detail::kDoubleEps *
        detail::max2(1.0, fabs(donor_m_corner[a]) + fabs(dm_a));
    if (!tenryu::hydro::rz::finite_double(dm_a) ||
        !tenryu::hydro::rz::finite_double(m_final) ||
        m_final < -tol) {
      return detail::make_velocity_momentum_packet_fct_result(
          VelocityRemapStatus::INVALID_INPUT);
    }
  }

  for (int b = 0; b < receiver_nverts; ++b) {
    const double dm_b = dm_q * lambda_r_packet[b];
    const double m_final = receiver_m_corner[b] + dm_b;
    const double tol =
        1024.0 * detail::kDoubleEps *
        detail::max2(1.0, fabs(receiver_m_corner[b]) + fabs(dm_b));
    if (!tenryu::hydro::rz::finite_double(dm_b) ||
        !tenryu::hydro::rz::finite_double(m_final) ||
        m_final < -tol) {
      return detail::make_velocity_momentum_packet_fct_result(
          VelocityRemapStatus::INVALID_INPUT);
    }

    const double u_hi_r = u_d_r_at_receiver[b] + correction_r;
    const double u_hi_z = u_d_z_at_receiver[b] + correction_z;
    const double p_lo_r = receiver_p_r[b] + dm_b * u_q_r;
    const double p_lo_z = receiver_p_z[b] + dm_b * u_q_z;
    const double p_anti_r = dm_b * (u_hi_r - u_q_r);
    const double p_anti_z = dm_b * (u_hi_z - u_q_z);

    if (!disable_fct_limiter) {
      if (!detail::fct_receiver_velocity_alpha_bound(m_final,
                                                     p_lo_r,
                                                     p_anti_r,
                                                     donor_min_ur,
                                                     donor_max_ur,
                                                     &result.alpha_momentum_r) ||
          !detail::fct_receiver_velocity_alpha_bound(m_final,
                                                     p_lo_z,
                                                     p_anti_z,
                                                     donor_min_uz,
                                                     donor_max_uz,
                                                     &result.alpha_momentum_z)) {
        return detail::make_velocity_momentum_packet_fct_result(
            VelocityRemapStatus::INVALID_INPUT);
      }
    }
  }

  result.alpha = detail::min2(result.alpha_mass,
                              detail::min2(result.alpha_momentum_r,
                                           result.alpha_momentum_z));
  result.alpha = detail::max2(0.0, detail::min2(1.0, result.alpha));
  if (!tenryu::hydro::rz::finite_double(result.alpha)) {
    return detail::make_velocity_momentum_packet_fct_result(
        VelocityRemapStatus::INVALID_INPUT);
  }

  for (int a = 0; a < donor_nverts; ++a) {
    double ui_r = 0.0;
    double ui_z = 0.0;
    detail::projected_velocity(donor_ur[a],
                               donor_uz[a],
                               detail::node_projector_at(donor_projector, a),
                               &ui_r,
                               &ui_z);
    if (donor_sigma != nullptr) {
      // Sum over corners of dm_a*(u_a + u_q - u_tilde) is exactly
      // dm_q*u_q: the exchanged momentum is identical to the
      // lambda_d path; only the intra-donor partition differs.
      const double dm_a = dm_q * donor_sigma[a];
      donor_m_corner[a] -= dm_a;
      donor_p_r[a] -= dm_a * (ui_r + (u_q_r - u_tilde_r));
      donor_p_z[a] -= dm_a * (ui_z + (u_q_z - u_tilde_z));
    } else {
      const double dm_a = dm_q * lambda_d_packet[a];
      donor_m_corner[a] -= dm_a;
      donor_p_r[a] -= dm_a * ui_r;
      donor_p_z[a] -= dm_a * ui_z;
    }
  }

  for (int b = 0; b < receiver_nverts; ++b) {
    const double dm_b = dm_q * lambda_r_packet[b];
    const double u_hi_r = u_d_r_at_receiver[b] + correction_r;
    const double u_hi_z = u_d_z_at_receiver[b] + correction_z;
    receiver_m_corner[b] += dm_b;
    if (result.alpha == 1.0) {
      receiver_p_r[b] += dm_b * u_hi_r;
      receiver_p_z[b] += dm_b * u_hi_z;
    } else {
      receiver_p_r[b] += dm_b * (u_q_r + result.alpha * (u_hi_r - u_q_r));
      receiver_p_z[b] += dm_b * (u_q_z + result.alpha * (u_hi_z - u_q_z));
    }
  }

  return result;
}

__host__ __device__ inline void scatter_nodal_velocity(
    const int n_corners,
    const int* corner_node,
    const double* m_corner,
    const double* p_r,
    const double* p_z,
    const int n_nodes,
    const NodeVelocityProjector* node_projector,
    double* node_mass,
    double* node_p_r,
    double* node_p_z,
    double* ur,
    double* uz) {
  if (n_corners <= 0 || n_nodes <= 0) {
    return;
  }

  for (int i = 0; i < n_nodes; ++i) {
    node_mass[i] = 0.0;
    node_p_r[i] = 0.0;
    node_p_z[i] = 0.0;
    ur[i] = 0.0;
    uz[i] = 0.0;
  }

  for (int c = 0; c < n_corners; ++c) {
    const int node = corner_node[c];
    if (node < 0 || node >= n_nodes) {
      continue;
    }
    node_mass[node] += m_corner[c];
    node_p_r[node] += p_r[c];
    node_p_z[node] += p_z[c];
  }

  for (int i = 0; i < n_nodes; ++i) {
    if (node_mass[i] > 0.0 &&
        tenryu::hydro::rz::finite_double(node_mass[i])) {
      ur[i] = node_p_r[i] / node_mass[i];
      uz[i] = node_p_z[i] / node_mass[i];
    }
    apply_node_velocity_projector(
        detail::node_projector_at(node_projector, i), &ur[i], &uz[i]);
  }
}

}  // namespace tenryu::hydro::optionb
