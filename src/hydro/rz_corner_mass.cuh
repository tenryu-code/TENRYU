#pragma once

#include <cmath>
#include <cstdint>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "mesh/mesh.hpp"

namespace tenryu::hydro::rz {

struct CornerMassFallbackProbe {
  int fired;
  double vals[5];
};

struct CornerMassFallbackRecord {
  int cell;
  int stage;
  int orient;
  unsigned int nan;
  double vals[5];
};

struct CornerMassFallbackRecorder {
  int claimed;
  int fire_count_bbsw;
  int fire_count_subpolygon;
  CornerMassFallbackRecord first;
};

// F-09 production-stage mapping:
//  1 hydro structured corner mass; 2 hydro multiblock exact-subpolygon;
//  3/4 HLLC pre/post corner mass; 5 CSR node projection;
//  6/7 CSR pre/post KE; 8/9/10 CSR total-energy scale/build/recover;
//  11/12 structured physical-KE build/recover;
//  13/14/15 CSR physical-KE scale/build/recover;
//  16/17 ALE driver pre/post corner mass.
constexpr int kCornerMassFallbackStageHydroStructured = 1;
constexpr int kCornerMassFallbackStageHydroMultiblock = 2;
constexpr int kCornerMassFallbackStageHllcPre = 3;
constexpr int kCornerMassFallbackStageHllcPost = 4;
constexpr int kCornerMassFallbackStageCsrNodeProjection = 5;
constexpr int kCornerMassFallbackStageCsrKePre = 6;
constexpr int kCornerMassFallbackStageCsrKePost = 7;
constexpr int kCornerMassFallbackStageCsrTotalEnergyScale = 8;
constexpr int kCornerMassFallbackStageCsrTotalEnergyBuild = 9;
constexpr int kCornerMassFallbackStageCsrTotalEnergyRecover = 10;
constexpr int kCornerMassFallbackStageStructuredPhysicalKeBuild = 11;
constexpr int kCornerMassFallbackStageStructuredPhysicalKeRecover = 12;
constexpr int kCornerMassFallbackStageCsrPhysicalKeScale = 13;
constexpr int kCornerMassFallbackStageCsrPhysicalKeBuild = 14;
constexpr int kCornerMassFallbackStageCsrPhysicalKeRecover = 15;
constexpr int kCornerMassFallbackStageAleDriverPre = 16;
constexpr int kCornerMassFallbackStageAleDriverPost = 17;

constexpr int kCornerMassConventionBbswRadialV0 = 0;
constexpr int kCornerMassConventionKinematicBasisRzV1 = 1;

void corner_mass_fallback_run_start();
CornerMassFallbackRecorder* corner_mass_fallback_device_recorder();
CornerMassFallbackRecorder corner_mass_fallback_copy_to_host();

__device__ inline void record_corner_mass_fallback(
    CornerMassFallbackRecorder* recorder,
    const CornerMassFallbackProbe& probe,
    const bool bbsw,
    const int cell,
    const int stage,
    const int orient) {
#if defined(__CUDA_ARCH__)
  if (recorder == nullptr || probe.fired != 1) {
    return;
  }
  atomicAdd(bbsw ? &recorder->fire_count_bbsw
                 : &recorder->fire_count_subpolygon,
            1);
  if (atomicCAS(&recorder->claimed, 0, 1) == 0) {
    recorder->first.cell = cell;
    recorder->first.stage = stage;
    recorder->first.orient = orient;
    unsigned int nan = 0U;
    for (int k = 0; k < 5; ++k) {
      recorder->first.vals[k] = probe.vals[k];
      const double value = probe.vals[k];
      if (!(value == value && value != INFINITY && value != -INFINITY)) {
        nan |= 1U << k;
      }
    }
    recorder->first.nan = nan;
  }
#else
  (void)recorder;
  (void)probe;
  (void)bbsw;
  (void)cell;
  (void)stage;
  (void)orient;
#endif
}

__host__ __device__ inline int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

__host__ __device__ inline bool finite_double(const double x) {
  return x == x && x != INFINITY && x != -INFINITY;
}

__host__ __device__ inline double rz_polygon_volume_exact(const double* r,
                                                          const double* z,
                                                          const int nverts) {
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    sum += (r[k] * z[kp] - r[kp] * z[k]) * (r[k] + r[kp]);
  }
  return pi_over_three * sum;
}

__host__ __device__ inline double rz_polygon_area2_exact(const double* r,
                                                         const double* z,
                                                         const int nverts) {
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    sum += r[k] * z[kp] - r[kp] * z[k];
  }
  return sum;
}

__host__ __device__ inline int button_seam_node_index(const int outer_ring,
                                                      const int j,
                                                      const int nz) {
  return node_index(outer_ring, j, nz);
}

__host__ __device__ inline double button_orientation_sign() {
  return -1.0;
}

__host__ __device__ inline double
button_polygon_volume_from_nodes(const double* x_r,
                                 const double* x_z,
                                 const int outer_ring,
                                 const int nz) {
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  const int nverts = nz + 1;
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    const int n = button_seam_node_index(outer_ring, k, nz);
    const int np = button_seam_node_index(outer_ring, kp, nz);
    sum += (x_r[n] * x_z[np] - x_r[np] * x_z[n]) * (x_r[n] + x_r[np]);
  }
  return button_orientation_sign() * pi_over_three * sum;
}

__host__ __device__ inline double button_seam_swept_volume_from_nodes(
    const double* x_r_lag,
    const double* x_z_lag,
    const double* x_r_ref,
    const double* x_z_ref,
    const int outer_ring,
    const int nz,
    const int j) {
  if (j == nz) {
    return 0.0;
  }
  const int n0 = button_seam_node_index(outer_ring, j, nz);
  const int n1 = button_seam_node_index(outer_ring, j + 1, nz);
  const double r[4] = {x_r_lag[n0], x_r_ref[n0], x_r_ref[n1], x_r_lag[n1]};
  const double z[4] = {x_z_lag[n0], x_z_ref[n0], x_z_ref[n1], x_z_lag[n1]};
  return button_orientation_sign() * -rz_polygon_volume_exact(r, z, 4);
}

__host__ __device__ inline double
button_polygon_area2_from_nodes(const double* x_r,
                                const double* x_z,
                                const int outer_ring,
                                const int nz) {
  const int nverts = nz + 1;
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    const int n = button_seam_node_index(outer_ring, k, nz);
    const int np = button_seam_node_index(outer_ring, kp, nz);
    sum += x_r[n] * x_z[np] - x_r[np] * x_z[n];
  }
  return sum;
}

__host__ __device__ inline void
button_polygon_area_centroid_from_nodes(const double* x_r,
                                        const double* x_z,
                                        const int outer_ring,
                                        const int nz,
                                        double* centroid_r,
                                        double* centroid_z) {
  const int nverts = nz + 1;
  double cross_sum = 0.0;
  double cr_sum = 0.0;
  double cz_sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    const int n = button_seam_node_index(outer_ring, k, nz);
    const int np = button_seam_node_index(outer_ring, kp, nz);
    const double cross = x_r[n] * x_z[np] - x_r[np] * x_z[n];
    cross_sum += cross;
    cr_sum += (x_r[n] + x_r[np]) * cross;
    cz_sum += (x_z[n] + x_z[np]) * cross;
  }
  if (fabs(cross_sum) > 0.0) {
    const double inv = 1.0 / (3.0 * cross_sum);
    *centroid_r = cr_sum * inv;
    *centroid_z = cz_sum * inv;
    return;
  }

  double r_avg = 0.0;
  double z_avg = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int n = button_seam_node_index(outer_ring, k, nz);
    r_avg += x_r[n];
    z_avg += x_z[n];
  }
  *centroid_r = r_avg / static_cast<double>(nverts);
  *centroid_z = z_avg / static_cast<double>(nverts);
}

__host__ __device__ inline void rz_polygon_area_centroid_exact(
    const double* r,
    const double* z,
    const int nverts,
    double* centroid_r,
    double* centroid_z) {
  double cross_sum = 0.0;
  double cr_sum = 0.0;
  double cz_sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    const double cross = r[k] * z[kp] - r[kp] * z[k];
    cross_sum += cross;
    cr_sum += (r[k] + r[kp]) * cross;
    cz_sum += (z[k] + z[kp]) * cross;
  }
  if (fabs(cross_sum) > 0.0) {
    const double inv = 1.0 / (3.0 * cross_sum);
    *centroid_r = cr_sum * inv;
    *centroid_z = cz_sum * inv;
    return;
  }
  double r_avg = 0.0;
  double z_avg = 0.0;
  for (int k = 0; k < nverts; ++k) {
    r_avg += r[k];
    z_avg += z[k];
  }
  *centroid_r = r_avg / static_cast<double>(nverts);
  *centroid_z = z_avg / static_cast<double>(nverts);
}

__host__ __device__ inline void rz_polygon_svec_exact(
    const double* r,
    const double* z,
    const int nverts,
    const int k,
    const double centroid_r,
    const double centroid_z,
    const double orientation_sign,
    double* sr_out,
    double* sz_out) {
  const int km1 = (k + nverts - 1) % nverts;
  const int kp1 = (k + 1) % nverts;
  const double trkm1 = r[km1] - centroid_r;
  const double trk = r[k] - centroid_r;
  const double trkp1 = r[kp1] - centroid_r;
  const double tzkm1 = z[km1] - centroid_z;
  const double tzk = z[k] - centroid_z;
  const double tzkp1 = z[kp1] - centroid_z;

  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  const double sr =
      pi_over_three *
      (2.0 * trk * (tzkp1 - tzkm1) + trkp1 * (tzkp1 - tzk) +
       trkm1 * (tzk - tzkm1) + 3.0 * centroid_r * (tzkp1 - tzkm1));
  const double sz = pi_over_three *
                    (trkm1 * (trkm1 + trk) - trkp1 * (trk + trkp1) +
                     3.0 * centroid_r * (trkm1 - trkp1));
  *sr_out = orientation_sign * sr;
  *sz_out = orientation_sign * sz;
}

__host__ __device__ inline void
button_polygon_svec_from_nodes(const double* x_r,
                               const double* x_z,
                               const int outer_ring,
                               const int nz,
                               const int k,
                               const double centroid_r,
                               const double centroid_z,
                               double* sr_out,
                               double* sz_out) {
  const int nverts = nz + 1;
  const int km1 = (k + nverts - 1) % nverts;
  const int kp1 = (k + 1) % nverts;
  const int nm1 = button_seam_node_index(outer_ring, km1, nz);
  const int n = button_seam_node_index(outer_ring, k, nz);
  const int np1 = button_seam_node_index(outer_ring, kp1, nz);

  const double trkm1 = x_r[nm1] - centroid_r;
  const double trk = x_r[n] - centroid_r;
  const double trkp1 = x_r[np1] - centroid_r;
  const double tzkm1 = x_z[nm1] - centroid_z;
  const double tzk = x_z[n] - centroid_z;
  const double tzkp1 = x_z[np1] - centroid_z;

  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  const double sr =
      pi_over_three *
      (2.0 * trk * (tzkp1 - tzkm1) + trkp1 * (tzkp1 - tzk) +
       trkm1 * (tzk - tzkm1) + 3.0 * centroid_r * (tzkp1 - tzkm1));
  const double sz = pi_over_three *
                    (trkm1 * (trkm1 + trk) - trkp1 * (trk + trkp1) +
                     3.0 * centroid_r * (trkm1 - trkp1));
  *sr_out = button_orientation_sign() * sr;
  *sz_out = button_orientation_sign() * sz;
}

__host__ __device__ inline double
button_corner_volume_exact_subpolygon(const double* x_r,
                                      const double* x_z,
                                      const int outer_ring,
                                      const int nz,
                                      const int k,
                                      const double centroid_r,
                                      const double centroid_z) {
  const int nverts = nz + 1;
  const int km1 = (k + nverts - 1) % nverts;
  const int kp1 = (k + 1) % nverts;
  const int nm1 = button_seam_node_index(outer_ring, km1, nz);
  const int n = button_seam_node_index(outer_ring, k, nz);
  const int np1 = button_seam_node_index(outer_ring, kp1, nz);

  const double r_sub[4] = {
      x_r[n],
      0.5 * (x_r[n] + x_r[np1]),
      centroid_r,
      0.5 * (x_r[nm1] + x_r[n]),
  };
  const double z_sub[4] = {
      x_z[n],
      0.5 * (x_z[n] + x_z[np1]),
      centroid_z,
      0.5 * (x_z[nm1] + x_z[n]),
  };
  return button_orientation_sign() * rz_polygon_volume_exact(r_sub, z_sub, 4);
}

__host__ __device__ inline double
button_corner_mass_exact_subpolygon(const double rho,
                                    const double* x_r,
                                    const double* x_z,
                                    const int outer_ring,
                                    const int nz,
                                    const int k,
                                    const double centroid_r,
                                    const double centroid_z) {
  return rho * button_corner_volume_exact_subpolygon(
                   x_r, x_z, outer_ring, nz, k, centroid_r, centroid_z);
}

__host__ __device__ inline void compute_quad_corner_masses_bbsw(
    const double m_cell,
    const double r00,
    const double r10,
    const double r11,
    const double r01,
    double* m_corner,
    CornerMassFallbackProbe* probe = nullptr) {
  const double R_L = 0.5 * (r00 + r01);
  const double R_R = 0.5 * (r10 + r11);
  const double denom = (R_L + R_R) * 6.0;
  if (!(denom > 0.0) || !finite_double(denom)) {
    if (probe != nullptr) {
      probe->fired = 1;
      probe->vals[0] = r00;
      probe->vals[1] = r10;
      probe->vals[2] = r11;
      probe->vals[3] = r01;
      probe->vals[4] = m_cell;
    }
    const double m_quarter = 0.25 * m_cell;
    m_corner[0] = m_quarter;
    m_corner[1] = m_quarter;
    m_corner[2] = m_quarter;
    m_corner[3] = m_quarter;
    return;
  }

  const double w_L = (2.0 * R_L + R_R) / denom;
  const double w_R = (R_L + 2.0 * R_R) / denom;
  m_corner[0] = w_L * m_cell;
  m_corner[1] = w_R * m_cell;
  m_corner[2] = w_R * m_cell;
  m_corner[3] = w_L * m_cell;
}

__host__ __device__ inline void compute_quad_corner_masses_kinematic_rz_v1(
    const double m_cell,
    const double r00,
    const double z00,
    const double r10,
    const double z10,
    const double r11,
    const double z11,
    const double r01,
    const double z01,
    double* m_corner,
    CornerMassFallbackProbe* probe = nullptr) {
  const double gauss_nodes[2] = {
      -0.5773502691896257,
      0.5773502691896257,
  };
  const double r[4] = {r00, r10, r11, r01};
  const double z[4] = {z00, z10, z11, z01};
  double numerator[4] = {0.0, 0.0, 0.0, 0.0};
  double denominator = 0.0;

  for (int ixi = 0; ixi < 2; ++ixi) {
    const double xi = gauss_nodes[ixi];
    for (int ieta = 0; ieta < 2; ++ieta) {
      const double eta = gauss_nodes[ieta];
      const double shape[4] = {
          0.25 * (1.0 - xi) * (1.0 - eta),
          0.25 * (1.0 + xi) * (1.0 - eta),
          0.25 * (1.0 + xi) * (1.0 + eta),
          0.25 * (1.0 - xi) * (1.0 + eta),
      };
      const double dshape_dxi[4] = {
          -0.25 * (1.0 - eta),
          0.25 * (1.0 - eta),
          0.25 * (1.0 + eta),
          -0.25 * (1.0 + eta),
      };
      const double dshape_deta[4] = {
          -0.25 * (1.0 - xi),
          -0.25 * (1.0 + xi),
          0.25 * (1.0 + xi),
          0.25 * (1.0 - xi),
      };

      double radius = 0.0;
      double dr_dxi = 0.0;
      double dr_deta = 0.0;
      double dz_dxi = 0.0;
      double dz_deta = 0.0;
      for (int k = 0; k < 4; ++k) {
        radius += shape[k] * r[k];
        dr_dxi += dshape_dxi[k] * r[k];
        dr_deta += dshape_deta[k] * r[k];
        dz_dxi += dshape_dxi[k] * z[k];
        dz_deta += dshape_deta[k] * z[k];
      }
      const double jacobian =
          dr_dxi * dz_deta - dr_deta * dz_dxi;
      const double weighted_jacobian = radius * jacobian;
      denominator += weighted_jacobian;
      for (int k = 0; k < 4; ++k) {
        numerator[k] += shape[k] * weighted_jacobian;
      }
    }
  }

  if (!(denominator > 0.0) || !finite_double(denominator)) {
    if (probe != nullptr) {
      probe->fired = 1;
      probe->vals[0] = r00;
      probe->vals[1] = r10;
      probe->vals[2] = r11;
      probe->vals[3] = r01;
      probe->vals[4] = m_cell;
    }
    const double m_quarter = 0.25 * m_cell;
    m_corner[0] = m_quarter;
    m_corner[1] = m_quarter;
    m_corner[2] = m_quarter;
    m_corner[3] = m_quarter;
    return;
  }

  for (int k = 0; k < 4; ++k) {
    m_corner[k] = m_cell * numerator[k] / denominator;
  }
}

__host__ __device__ inline void compute_tri_corner_masses_kinematic_rz_v1(
    const double m_cell,
    const double r0,
    const double z0,
    const double r1,
    const double z1,
    const double r2,
    const double z2,
    double* m3,
    CornerMassFallbackProbe* probe = nullptr) {
  (void)z0;
  (void)z1;
  (void)z2;
  const double S = r0 + r1 + r2;
  if (!(S > 0.0) || !finite_double(S)) {
    if (probe != nullptr) {
      probe->fired = 1;
      probe->vals[0] = r0;
      probe->vals[1] = r1;
      probe->vals[2] = r2;
      probe->vals[3] = S;
      probe->vals[4] = m_cell;
    }
    const double m_third = m_cell / 3.0;
    m3[0] = m_third;
    m3[1] = m_third;
    m3[2] = m_third;
    return;
  }

  const double denominator = 4.0 * S;
  m3[0] = m_cell * ((r0 + S) / denominator);
  m3[1] = m_cell * ((r1 + S) / denominator);
  m3[2] = m_cell * ((r2 + S) / denominator);
}

__host__ __device__ inline void compute_quad_corner_signed_volumes_subpolygon(
    const double r00,
    const double z00,
    const double r10,
    const double z10,
    const double r11,
    const double z11,
    const double r01,
    const double z01,
    double* v_corner) {
  const double rc = 0.25 * (r00 + r10 + r11 + r01);
  const double zc = 0.25 * (z00 + z10 + z11 + z01);
  const double r01m = 0.5 * (r00 + r10);
  const double z01m = 0.5 * (z00 + z10);
  const double r12m = 0.5 * (r10 + r11);
  const double z12m = 0.5 * (z10 + z11);
  const double r23m = 0.5 * (r11 + r01);
  const double z23m = 0.5 * (z11 + z01);
  const double r30m = 0.5 * (r01 + r00);
  const double z30m = 0.5 * (z01 + z00);

  const double r_sub0[4] = {r00, r01m, rc, r30m};
  const double z_sub0[4] = {z00, z01m, zc, z30m};
  const double r_sub1[4] = {r01m, r10, r12m, rc};
  const double z_sub1[4] = {z01m, z10, z12m, zc};
  const double r_sub2[4] = {rc, r12m, r11, r23m};
  const double z_sub2[4] = {zc, z12m, z11, z23m};
  const double r_sub3[4] = {r30m, rc, r23m, r01};
  const double z_sub3[4] = {z30m, zc, z23m, z01};

  v_corner[0] = rz_polygon_volume_exact(r_sub0, z_sub0, 4);
  v_corner[1] = rz_polygon_volume_exact(r_sub1, z_sub1, 4);
  v_corner[2] = rz_polygon_volume_exact(r_sub2, z_sub2, 4);
  v_corner[3] = rz_polygon_volume_exact(r_sub3, z_sub3, 4);
}

__host__ __device__ inline void compute_quad_corner_volumes_exact_subpolygon(
    const double r00,
    const double z00,
    const double r10,
    const double z10,
    const double r11,
    const double z11,
    const double r01,
    const double z01,
    double* v_corner) {
  const double rc = 0.25 * (r00 + r10 + r11 + r01);
  const double zc = 0.25 * (z00 + z10 + z11 + z01);
  const double r01m = 0.5 * (r00 + r10);
  const double z01m = 0.5 * (z00 + z10);
  const double r12m = 0.5 * (r10 + r11);
  const double z12m = 0.5 * (z10 + z11);
  const double r23m = 0.5 * (r11 + r01);
  const double z23m = 0.5 * (z11 + z01);
  const double r30m = 0.5 * (r01 + r00);
  const double z30m = 0.5 * (z01 + z00);

  const double r_sub0[4] = {r00, r01m, rc, r30m};
  const double z_sub0[4] = {z00, z01m, zc, z30m};
  const double r_sub1[4] = {r01m, r10, r12m, rc};
  const double z_sub1[4] = {z01m, z10, z12m, zc};
  const double r_sub2[4] = {rc, r12m, r11, r23m};
  const double z_sub2[4] = {zc, z12m, z11, z23m};
  const double r_sub3[4] = {r30m, rc, r23m, r01};
  const double z_sub3[4] = {z30m, zc, z23m, z01};

  v_corner[0] = fabs(rz_polygon_volume_exact(r_sub0, z_sub0, 4));
  v_corner[1] = fabs(rz_polygon_volume_exact(r_sub1, z_sub1, 4));
  v_corner[2] = fabs(rz_polygon_volume_exact(r_sub2, z_sub2, 4));
  v_corner[3] = fabs(rz_polygon_volume_exact(r_sub3, z_sub3, 4));
}

__host__ __device__ inline void compute_quad_corner_pressure_force_subpolygon(
    const double pressure,
    const double r00,
    const double z00,
    const double r10,
    const double z10,
    const double r11,
    const double z11,
    const double r01,
    const double z01,
    const double orientation_sign,
    double* force_r,
    double* force_z) {
  const double r[4] = {r00, r10, r11, r01};
  const double z[4] = {z00, z10, z11, z01};
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  rz_polygon_area_centroid_exact(r, z, 4, &centroid_r, &centroid_z);
  for (int k = 0; k < 4; ++k) {
    double sr = 0.0;
    double sz = 0.0;
    rz_polygon_svec_exact(r, z, 4, k, centroid_r, centroid_z,
                          orientation_sign, &sr, &sz);
    force_r[k] = pressure * sr;
    force_z[k] = pressure * sz;
  }
}

__host__ __device__ inline void compute_quad_corner_masses_exact_subpolygon(
    const double m_cell,
    const double r00,
    const double z00,
    const double r10,
    const double z10,
    const double r11,
    const double z11,
    const double r01,
    const double z01,
    double* m_corner,
    CornerMassFallbackProbe* probe = nullptr) {
  const double full_r[4] = {r00, r10, r11, r01};
  const double full_z[4] = {z00, z10, z11, z01};
  const double v_total = fabs(rz_polygon_volume_exact(full_r, full_z, 4));
  if (!(v_total > 0.0) || !finite_double(v_total)) {
    if (probe != nullptr) {
      double v_corner_signed[4] = {0.0, 0.0, 0.0, 0.0};
      compute_quad_corner_signed_volumes_subpolygon(r00,
                                                    z00,
                                                    r10,
                                                    z10,
                                                    r11,
                                                    z11,
                                                    r01,
                                                    z01,
                                                    v_corner_signed);
      probe->fired = 1;
      probe->vals[0] = v_corner_signed[0];
      probe->vals[1] = v_corner_signed[1];
      probe->vals[2] = v_corner_signed[2];
      probe->vals[3] = v_corner_signed[3];
      probe->vals[4] = v_total;
    }
    const double m_quarter = 0.25 * m_cell;
    m_corner[0] = m_quarter;
    m_corner[1] = m_quarter;
    m_corner[2] = m_quarter;
    m_corner[3] = m_quarter;
    return;
  }

  double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
  compute_quad_corner_volumes_exact_subpolygon(r00,
                                               z00,
                                               r10,
                                               z10,
                                               r11,
                                               z11,
                                               r01,
                                               z01,
                                               v_corner);
  m_corner[0] = m_cell * (v_corner[0] / v_total);
  m_corner[1] = m_cell * (v_corner[1] / v_total);
  m_corner[2] = m_cell * (v_corner[2] / v_total);
  m_corner[3] = m_cell * (v_corner[3] / v_total);
}

__host__ __device__ inline void
compute_quad_corner_masses_partitioned_subpolygon(const double m_cell,
                                                  const double r00,
                                                  const double z00,
                                                  const double r10,
                                                  const double z10,
                                                  const double r11,
                                                  const double z11,
                                                  const double r01,
                                                  const double z01,
                                                  double* m_corner) {
  double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
  compute_quad_corner_volumes_exact_subpolygon(r00,
                                               z00,
                                               r10,
                                               z10,
                                               r11,
                                               z11,
                                               r01,
                                               z01,
                                               v_corner);
  const double v_sum = v_corner[0] + v_corner[1] + v_corner[2] + v_corner[3];
  if (!(v_sum > 0.0) || !finite_double(v_sum)) {
    const double m_quarter = 0.25 * m_cell;
    m_corner[0] = m_quarter;
    m_corner[1] = m_quarter;
    m_corner[2] = m_quarter;
    m_corner[3] = m_quarter;
    return;
  }
  m_corner[0] = m_cell * (v_corner[0] / v_sum);
  m_corner[1] = m_cell * (v_corner[1] / v_sum);
  m_corner[2] = m_cell * (v_corner[2] / v_sum);
  m_corner[3] = m_cell * (v_corner[3] / v_sum);
}

__host__ __device__ inline bool
compute_quad_corner_masses_rz_volume_no_planar_fallback(
    const double m_cell,
    const double r00,
    const double z00,
    const double r10,
    const double z10,
    const double r11,
    const double z11,
    const double r01,
    const double z01,
    double* m_corner) {
  m_corner[0] = 0.0;
  m_corner[1] = 0.0;
  m_corner[2] = 0.0;
  m_corner[3] = 0.0;
  if (!(m_cell >= 0.0) || !finite_double(m_cell)) {
    return false;
  }

  double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
  compute_quad_corner_volumes_exact_subpolygon(r00,
                                               z00,
                                               r10,
                                               z10,
                                               r11,
                                               z11,
                                               r01,
                                               z01,
                                               v_corner);
  const double v_sum = v_corner[0] + v_corner[1] + v_corner[2] + v_corner[3];
  if (!(v_sum > 0.0) || !finite_double(v_sum)) {
    return false;
  }
  m_corner[0] = m_cell * (v_corner[0] / v_sum);
  m_corner[1] = m_cell * (v_corner[1] / v_sum);
  m_corner[2] = m_cell * (v_corner[2] / v_sum);
  m_corner[3] = m_cell * (v_corner[3] / v_sum);
  return finite_double(m_corner[0]) && finite_double(m_corner[1]) &&
         finite_double(m_corner[2]) && finite_double(m_corner[3]);
}

__host__ __device__ inline void compute_triangle_corner_masses_exact(
    const double m_cell,
    const double r0,
    const double z0,
    const double r1,
    const double z1,
    const double r2,
    const double z2,
    double* m_corner) {
  const double r[3] = {r0, r1, r2};
  const double z[3] = {z0, z1, z2};
  const double total = rz_polygon_volume_exact(r, z, 3);
  if (!(total != 0.0) || !finite_double(total)) {
    const double m_third = m_cell / 3.0;
    m_corner[0] = m_third;
    m_corner[1] = m_third;
    m_corner[2] = m_third;
    m_corner[3] = 0.0;
    return;
  }

  const double rc = (r0 + r1 + r2) / 3.0;
  const double zc = (z0 + z1 + z2) / 3.0;
  for (int k = 0; k < 3; ++k) {
    const int kp = (k + 1) % 3;
    const int km = (k + 2) % 3;
    const double sr[4] = {
        r[k],
        0.5 * (r[k] + r[kp]),
        rc,
        0.5 * (r[km] + r[k]),
    };
    const double sz[4] = {
        z[k],
        0.5 * (z[k] + z[kp]),
        zc,
        0.5 * (z[km] + z[k]),
    };
    m_corner[k] = m_cell * (rz_polygon_volume_exact(sr, sz, 4) / total);
  }
  m_corner[3] = 0.0;
}

__host__ __device__ inline void compute_rz_corner_masses_from_nodes(
    const int c,
    const int nz,
    const double m_cell,
    const double* x_r,
    const double* x_z,
    const std::uint8_t* cell_nverts,
    double* m_corner,
    CornerMassFallbackProbe* probe = nullptr,
    const int corner_mass_convention =
        kCornerMassConventionBbswRadialV0) {
  const int i = c / nz;
  const int j = c - i * nz;
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  if (tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c) == 3) {
    if (corner_mass_convention ==
        kCornerMassConventionKinematicBasisRzV1) {
      compute_tri_corner_masses_kinematic_rz_v1(m_cell,
                                                x_r[n00],
                                                x_z[n00],
                                                x_r[n10],
                                                x_z[n10],
                                                x_r[n11],
                                                x_z[n11],
                                                m_corner,
                                                probe);
      m_corner[3] = 0.0;
    } else {
      compute_triangle_corner_masses_exact(m_cell,
                                           x_r[n00],
                                           x_z[n00],
                                           x_r[n10],
                                           x_z[n10],
                                           x_r[n11],
                                           x_z[n11],
                                           m_corner);
    }
    return;
  }
  if (corner_mass_convention ==
      kCornerMassConventionKinematicBasisRzV1) {
    compute_quad_corner_masses_kinematic_rz_v1(
        m_cell,
        x_r[n00],
        x_z[n00],
        x_r[n10],
        x_z[n10],
        x_r[n11],
        x_z[n11],
        x_r[n01],
        x_z[n01],
        m_corner,
        probe);
  } else {
    compute_quad_corner_masses_bbsw(
        m_cell, x_r[n00], x_r[n10], x_r[n11], x_r[n01], m_corner, probe);
  }
}

__host__ __device__ inline void compute_rz_corner_masses_for_cell(
    const int c,
    const int nz,
    const double m_cell,
    const double* x_r,
    const double* x_z,
    const std::uint8_t* cell_nverts,
    double* corner_mass,
    const int corner_stride,
    CornerMassFallbackProbe* probe = nullptr,
    const int corner_mass_convention =
        kCornerMassConventionBbswRadialV0) {
  double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
  compute_rz_corner_masses_from_nodes(
      c, nz, m_cell, x_r, x_z, cell_nverts, m_corner, probe,
      corner_mass_convention);
  corner_mass[c * corner_stride + 0] = m_corner[0];
  corner_mass[c * corner_stride + 1] = m_corner[1];
  corner_mass[c * corner_stride + 2] = m_corner[2];
  corner_mass[c * corner_stride + 3] = m_corner[3];
}

__host__ __device__ inline void compute_rz_corner_masses_for_cell(
    const int c,
    const int nz,
    const double m_cell,
    const double* x_r,
    double* corner_mass,
    const int corner_stride,
    CornerMassFallbackProbe* probe = nullptr) {
  const int i = c / nz;
  const int j = c - i * nz;
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
  compute_quad_corner_masses_bbsw(
      m_cell, x_r[n00], x_r[n10], x_r[n11], x_r[n01], m_corner, probe);
  corner_mass[c * corner_stride + 0] = m_corner[0];
  corner_mass[c * corner_stride + 1] = m_corner[1];
  corner_mass[c * corner_stride + 2] = m_corner[2];
  corner_mass[c * corner_stride + 3] = m_corner[3];
}

}  // namespace tenryu::hydro::rz
