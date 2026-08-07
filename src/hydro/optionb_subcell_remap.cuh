#pragma once

#include <cmath>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "hydro/rz_corner_mass.cuh"

namespace tenryu::hydro::optionb {
namespace detail {

constexpr double kDoubleEps = 2.220446049250313080847263336181640625e-16;

__host__ __device__ inline double max2(const double a, const double b) {
  return a > b ? a : b;
}

__host__ __device__ inline double min2(const double a, const double b) {
  return a < b ? a : b;
}

__host__ __device__ inline double clamp01(const double x) {
  return max2(0.0, min2(1.0, x));
}

__host__ __device__ inline void fill_uniform(const int nverts, double* lambda) {
  const double w = nverts > 0 ? 1.0 / static_cast<double>(nverts) : 0.0;
  for (int i = 0; i < nverts; ++i) {
    lambda[i] = w;
  }
}

__host__ __device__ inline void fill_vertex(const int nverts,
                                            const int vertex,
                                            double* lambda) {
  for (int i = 0; i < nverts; ++i) {
    lambda[i] = (i == vertex) ? 1.0 : 0.0;
  }
}

__host__ __device__ inline void fill_edge(const int nverts,
                                          const int i0,
                                          const int i1,
                                          const double t,
                                          double* lambda) {
  for (int i = 0; i < nverts; ++i) {
    lambda[i] = 0.0;
  }
  const double tc = clamp01(t);
  lambda[i0] = 1.0 - tc;
  lambda[i1] = tc;
}

__host__ __device__ inline double polygon_length_scale(const double* r,
                                                       const double* z,
                                                       const int nverts,
                                                       const double rq,
                                                       const double zq) {
  double scale = max2(fabs(rq), fabs(zq));
  for (int i = 0; i < nverts; ++i) {
    scale = max2(scale, max2(fabs(r[i]), fabs(z[i])));
    const int ip = (i + 1 == nverts) ? 0 : i + 1;
    const double er = r[ip] - r[i];
    const double ez = z[ip] - z[i];
    scale = max2(scale, sqrt(er * er + ez * ez));
  }
  return max2(scale, 1.0e-300);
}

__host__ __device__ inline bool handle_vertex_or_edge_query(const double* r,
                                                            const double* z,
                                                            const int nverts,
                                                            const double rq,
                                                            const double zq,
                                                            double* lambda) {
  const double scale = polygon_length_scale(r, z, nverts, rq, zq);
  const double vertex_tol = 64.0 * kDoubleEps * scale;
  const double vertex_tol2 = vertex_tol * vertex_tol;
  for (int i = 0; i < nverts; ++i) {
    const double dr = rq - r[i];
    const double dz = zq - z[i];
    if (dr * dr + dz * dz <= vertex_tol2) {
      fill_vertex(nverts, i, lambda);
      return true;
    }
  }

  for (int i = 0; i < nverts; ++i) {
    const int ip = (i + 1 == nverts) ? 0 : i + 1;
    const double er = r[ip] - r[i];
    const double ez = z[ip] - z[i];
    const double len2 = er * er + ez * ez;
    if (!(len2 > 0.0) || !tenryu::hydro::rz::finite_double(len2)) {
      continue;
    }
    const double qr = rq - r[i];
    const double qz = zq - z[i];
    const double cross = er * qz - ez * qr;
    const double dot = er * qr + ez * qz;
    const double cross_tol = 128.0 * kDoubleEps * len2;
    const double dot_tol = 128.0 * kDoubleEps * len2;
    if (fabs(cross) <= cross_tol && dot >= -dot_tol &&
        dot <= len2 + dot_tol) {
      fill_edge(nverts, i, ip, dot / len2, lambda);
      return true;
    }
  }
  return false;
}

__host__ __device__ inline double triangle_area_tol(const double* r,
                                                    const double* z) {
  double edge2_max = 0.0;
  for (int i = 0; i < 3; ++i) {
    const int ip = (i + 1 == 3) ? 0 : i + 1;
    const double er = r[ip] - r[i];
    const double ez = z[ip] - z[i];
    edge2_max = max2(edge2_max, er * er + ez * ez);
  }
  return 256.0 * kDoubleEps * max2(edge2_max, 1.0e-300);
}

__host__ __device__ inline double tan_half_angle(const double ar,
                                                 const double az,
                                                 const double br,
                                                 const double bz) {
  const double la2 = ar * ar + az * az;
  const double lb2 = br * br + bz * bz;
  if (!(la2 > 0.0) || !(lb2 > 0.0)) {
    return 0.0;
  }
  const double la = sqrt(la2);
  const double lb = sqrt(lb2);
  const double cross = ar * bz - az * br;
  const double dot = ar * br + az * bz;
  const double scale = la * lb;
  const double denom = scale + dot;
  const double tol = 128.0 * kDoubleEps * scale;
  if (fabs(denom) > tol) {
    return cross / denom;
  }
  if (cross > 0.0) {
    return 1.0e100;
  }
  if (cross < 0.0) {
    return -1.0e100;
  }
  return 0.0;
}

}  // namespace detail

__host__ __device__ inline void rz_mass_centroid(const double* r,
                                                 const double* z,
                                                 const int nverts,
                                                 double* rbar,
                                                 double* zbar) {
  *rbar = 0.0;
  *zbar = 0.0;
  if (nverts <= 0) {
    return;
  }

  double r_ref = 0.0;
  double z_ref = 0.0;
  for (int i = 0; i < nverts; ++i) {
    r_ref += r[i];
    z_ref += z[i];
  }
  r_ref /= static_cast<double>(nverts);
  z_ref /= static_cast<double>(nverts);

  double area2 = 0.0;
  double i_u_sum = 0.0;
  double i_v_sum = 0.0;
  double i_uu_sum = 0.0;
  double i_uv_sum = 0.0;
  for (int i = 0; i < nverts; ++i) {
    const int ip = (i + 1 == nverts) ? 0 : i + 1;
    const double u0 = r[i] - r_ref;
    const double v0 = z[i] - z_ref;
    const double u1 = r[ip] - r_ref;
    const double v1 = z[ip] - z_ref;
    const double cross = u0 * v1 - u1 * v0;
    area2 += cross;
    i_u_sum += (u0 + u1) * cross;
    i_v_sum += (v0 + v1) * cross;
    i_uu_sum += (u0 * u0 + u0 * u1 + u1 * u1) * cross;
    i_uv_sum +=
        (2.0 * u0 * v0 + u0 * v1 + u1 * v0 + 2.0 * u1 * v1) * cross;
  }

  const double area = 0.5 * area2;
  const double i_u = i_u_sum / 6.0;
  const double i_v = i_v_sum / 6.0;
  const double i_uu = i_uu_sum / 12.0;
  const double i_uv = i_uv_sum / 24.0;

  const double i_r = r_ref * area + i_u;
  const double i_rr = r_ref * r_ref * area + 2.0 * r_ref * i_u + i_uu;
  const double i_rz =
      r_ref * z_ref * area + r_ref * i_v + z_ref * i_u + i_uv;
  const double i_r_tol =
      128.0 * detail::kDoubleEps * (fabs(r_ref * area) + fabs(i_u) + 1.0e-300);

  if (fabs(i_r) > i_r_tol && tenryu::hydro::rz::finite_double(i_r) &&
      tenryu::hydro::rz::finite_double(i_rr) &&
      tenryu::hydro::rz::finite_double(i_rz)) {
    const double rb = i_rr / i_r;
    const double zb = i_rz / i_r;
    if (tenryu::hydro::rz::finite_double(rb) &&
        tenryu::hydro::rz::finite_double(zb)) {
      *rbar = rb;
      *zbar = zb;
      return;
    }
  }

  double r_avg = 0.0;
  double z_avg = 0.0;
  for (int i = 0; i < nverts; ++i) {
    r_avg += r[i];
    z_avg += z[i];
  }
  *rbar = r_avg / static_cast<double>(nverts);
  *zbar = z_avg / static_cast<double>(nverts);
}

__host__ __device__ inline void barycentric_weights(const double* r,
                                                    const double* z,
                                                    const int nverts,
                                                    const double rq,
                                                    const double zq,
                                                    double* lambda) {
  if (nverts <= 0) {
    return;
  }
  for (int i = 0; i < nverts; ++i) {
    lambda[i] = 0.0;
  }
  if (detail::handle_vertex_or_edge_query(r, z, nverts, rq, zq, lambda)) {
    return;
  }

  if (nverts == 3) {
    const double area2 = tenryu::hydro::rz::rz_polygon_area2_exact(r, z, 3);
    if (!(fabs(area2) > detail::triangle_area_tol(r, z)) ||
        !tenryu::hydro::rz::finite_double(area2)) {
      detail::fill_uniform(nverts, lambda);
      return;
    }
    const double q1r = r[1] - rq;
    const double q1z = z[1] - zq;
    const double q2r = r[2] - rq;
    const double q2z = z[2] - zq;
    const double q0r = r[0] - rq;
    const double q0z = z[0] - zq;
    lambda[0] = (q1r * q2z - q1z * q2r) / area2;
    lambda[1] = (q2r * q0z - q2z * q0r) / area2;
    lambda[2] = 1.0 - lambda[0] - lambda[1];
    return;
  }

  double w_sum = 0.0;
  for (int i = 0; i < nverts; ++i) {
    const int im = (i == 0) ? nverts - 1 : i - 1;
    const int ip = (i + 1 == nverts) ? 0 : i + 1;
    const double vir = r[i] - rq;
    const double viz = z[i] - zq;
    const double dim2 = vir * vir + viz * viz;
    if (!(dim2 > 0.0) || !tenryu::hydro::rz::finite_double(dim2)) {
      detail::fill_vertex(nverts, i, lambda);
      return;
    }
    const double vimr = r[im] - rq;
    const double vimz = z[im] - zq;
    const double vipr = r[ip] - rq;
    const double vipz = z[ip] - zq;
    const double tan_prev = detail::tan_half_angle(vimr, vimz, vir, viz);
    const double tan_next = detail::tan_half_angle(vir, viz, vipr, vipz);
    lambda[i] = (tan_prev + tan_next) / sqrt(dim2);
    w_sum += lambda[i];
  }

  const double w_tol =
      128.0 * detail::kDoubleEps * (fabs(w_sum) + 1.0e-300);
  if (!(fabs(w_sum) > w_tol) || !tenryu::hydro::rz::finite_double(w_sum)) {
    // Stage 1 does not define production out-of-hull behavior; keep results
    // finite for degenerate/outside queries and let later remap stages reject.
    detail::fill_uniform(nverts, lambda);
    return;
  }

  for (int i = 0; i < nverts; ++i) {
    lambda[i] /= w_sum;
    if (!tenryu::hydro::rz::finite_double(lambda[i])) {
      detail::fill_uniform(nverts, lambda);
      return;
    }
  }
}

__host__ __device__ inline void first_moment_corner_masses(
    const double m_cell,
    const double* r,
    const double* z,
    const int nverts,
    double* m_corner) {
  if (nverts <= 0) {
    return;
  }
  double rbar = 0.0;
  double zbar = 0.0;
  rz_mass_centroid(r, z, nverts, &rbar, &zbar);
  barycentric_weights(r, z, nverts, rbar, zbar, m_corner);
  for (int i = 0; i < nverts; ++i) {
    m_corner[i] *= m_cell;
  }
}

}  // namespace tenryu::hydro::optionb
