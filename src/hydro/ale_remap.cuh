#pragma once

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "hydro/ale_remap_2d_rz.hpp"
#include "hydro/rz_quad_volume.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale {
namespace detail {

__host__ __device__ inline double edge_pappus_contribution(const double R_a,
                                                           const double Z_a,
                                                           const double R_b,
                                                           const double Z_b) {
  return (R_a + R_b) * (R_a * Z_b - R_b * Z_a);
}

__host__ __device__ inline double rz_polygon_volume_4(const double R[4],
                                                      const double Z[4]) {
  double sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int k1 = (k + 1) & 3;
    sum += edge_pappus_contribution(R[k], Z[k], R[k1], Z[k1]);
  }
  return (kPi / 3.0) * sum;
}

__host__ __device__ inline double face_swept_volume_outward(
    const double R_a_old,
    const double Z_a_old,
    const double R_b_old,
    const double Z_b_old,
    const double R_a_new,
    const double Z_a_new,
    const double R_b_new,
    const double Z_b_new) {
  return (kPi / 3.0) *
         (edge_pappus_contribution(R_a_new, Z_a_new, R_b_new, Z_b_new) -
          edge_pappus_contribution(R_a_old, Z_a_old, R_b_old, Z_b_old));
}

__host__ __device__ inline int cell_index(const int i, const int j, const int nz) {
  return i * nz + j;
}

__host__ __device__ inline int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

__device__ inline int clampi(const int v, const int lo, const int hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

__device__ inline double cell_center_r(const double* x_r,
                                       const int i,
                                       const int j,
                                       const int nz,
                                       const std::uint8_t* cell_nverts = nullptr) {
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  const int c = cell_index(i, j, nz);
  if (cell_nverts != nullptr && cell_nverts[c] == 3U) {
    return (x_r[n00] + x_r[n10] + x_r[n11]) / 3.0;
  }
  return 0.25 * (x_r[n00] + x_r[n10] + x_r[n11] + x_r[n01]);
}

__device__ inline double cell_center_z(const double* x_z,
                                       const int i,
                                       const int j,
                                       const int nz,
                                       const std::uint8_t* cell_nverts = nullptr) {
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  const int c = cell_index(i, j, nz);
  if (cell_nverts != nullptr && cell_nverts[c] == 3U) {
    return (x_z[n00] + x_z[n10] + x_z[n11]) / 3.0;
  }
  return 0.25 * (x_z[n00] + x_z[n10] + x_z[n11] + x_z[n01]);
}

__device__ inline double van_leer_phi(const double r) {
  return (r + fabs(r)) / (1.0 + fabs(r));
}

// First moment of r over the R-Z polygon revolved around z-axis:
//   integral_P 2*pi*r*r dA, evaluated from oriented triangles (0,1,2) and (0,2,3).
__host__ __device__ inline double rz_signed_quad_moment_r(const double r0,
                                                          const double z0,
                                                          const double r1,
                                                          const double z1,
                                                          const double r2,
                                                          const double z2,
                                                          const double r3,
                                                          const double z3) {
  const double A012 = 0.5 * ((r1 - r0) * (z2 - z0) - (r2 - r0) * (z1 - z0));
  const double A023 = 0.5 * ((r2 - r0) * (z3 - z0) - (r3 - r0) * (z2 - z0));

  const double rr012 =
      (A012 / 6.0) * (r0 * r0 + r1 * r1 + r2 * r2 + r0 * r1 + r1 * r2 + r2 * r0);
  const double rr023 =
      (A023 / 6.0) * (r0 * r0 + r2 * r2 + r3 * r3 + r0 * r2 + r2 * r3 + r3 * r0);

  return 2.0 * kPi * (rr012 + rr023);
}

// First moment of z over the R-Z polygon revolved around z-axis:
//   integral_P 2*pi*r*z dA, evaluated from oriented triangles (0,1,2) and (0,2,3).
__host__ __device__ inline double rz_signed_quad_moment_z(const double r0,
                                                          const double z0,
                                                          const double r1,
                                                          const double z1,
                                                          const double r2,
                                                          const double z2,
                                                          const double r3,
                                                          const double z3) {
  const double A012 = 0.5 * ((r1 - r0) * (z2 - z0) - (r2 - r0) * (z1 - z0));
  const double A023 = 0.5 * ((r2 - r0) * (z3 - z0) - (r3 - r0) * (z2 - z0));

  const double rz012 = (A012 / 6.0) * (r0 * z0 + r1 * z1 + r2 * z2) +
                       (A012 / 12.0) *
                           (r0 * z1 + r1 * z0 + r1 * z2 + r2 * z1 + r2 * z0 + r0 * z2);
  const double rz023 = (A023 / 6.0) * (r0 * z0 + r2 * z2 + r3 * z3) +
                       (A023 / 12.0) *
                           (r0 * z2 + r2 * z0 + r2 * z3 + r3 * z2 + r3 * z0 + r0 * z3);

  return 2.0 * kPi * (rz012 + rz023);
}

template <bool FixedSign>
__host__ __device__ inline double swept_volume_r_face_t(const double* x_r_old,
                                                        const double* x_z_old,
                                                        const double* x_r_new,
                                                        const double* x_z_new,
                                                        const int i_face,
                                                        const int j,
                                                        const int nz) {
  // R-face at (i_face, j .. j+1). Positive volume is low-i to high-i flux.
  const int n0 = node_index(i_face, j, nz);
  const int n1 = node_index(i_face, j + 1, nz);

  const double raw = rz_signed_quad_volume(x_r_old[n0],
                                           x_z_old[n0],
                                           x_r_new[n0],
                                           x_z_new[n0],
                                           x_r_new[n1],
                                           x_z_new[n1],
                                           x_r_old[n1],
                                           x_z_old[n1]);
  return FixedSign ? -raw : raw;
}

__host__ __device__ inline double swept_volume_r_face(const double* x_r_old,
                                                      const double* x_z_old,
                                                      const double* x_r_new,
                                                      const double* x_z_new,
                                                      const int i_face,
                                                      const int j,
                                                      const int nz) {
  return swept_volume_r_face_t<false>(
      x_r_old, x_z_old, x_r_new, x_z_new, i_face, j, nz);
}

template <bool FixedSign>
__host__ __device__ inline double swept_volume_z_face_t(const double* x_r_old,
                                                        const double* x_z_old,
                                                        const double* x_r_new,
                                                        const double* x_z_new,
                                                        const int i,
                                                        const int j_face,
                                                        const int nz) {
  // Z-face at (i .. i+1, j_face). Positive volume is low-j to high-j flux.
  const int n0 = node_index(i, j_face, nz);
  const int n1 = node_index(i + 1, j_face, nz);

  const double raw = rz_signed_quad_volume(x_r_old[n0],
                                           x_z_old[n0],
                                           x_r_old[n1],
                                           x_z_old[n1],
                                           x_r_new[n1],
                                           x_z_new[n1],
                                           x_r_new[n0],
                                           x_z_new[n0]);
  return FixedSign ? -raw : raw;
}

__host__ __device__ inline double swept_volume_z_face(const double* x_r_old,
                                                      const double* x_z_old,
                                                      const double* x_r_new,
                                                      const double* x_z_new,
                                                      const int i,
                                                      const int j_face,
                                                      const int nz) {
  return swept_volume_z_face_t<false>(
      x_r_old, x_z_old, x_r_new, x_z_new, i, j_face, nz);
}

template <bool FixedSign>
__device__ inline double flux_r_face_t(const double* field,
                                       const double* x_r_old,
                                       const double* x_z_old,
                                       const double* x_r_new,
                                       const double* x_z_new,
                                       const int i_face,
                                       const int j,
                                       const int nr,
                                       const int nz,
                                       const RemapDispatchAuditDeviceView audit = {},
                                       const std::uint8_t* geometry_policy_exempt = nullptr) {
  if (i_face <= 0 || i_face >= nr) {
    return 0.0;
  }

  const int i_left = i_face - 1;
  const int i_right = i_face;
  const int c_left = cell_index(i_left, j, nz);
  const int c_right = cell_index(i_right, j, nz);
  if (geometry_policy_exempt != nullptr &&
      (geometry_policy_exempt[c_left] != 0 ||
       geometry_policy_exempt[c_right] != 0)) {
    return 0.0;
  }

  const double deltaV =
      swept_volume_r_face_t<FixedSign>(x_r_old, x_z_old, x_r_new, x_z_new, i_face, j, nz);
  // FixedSign flips the stored primitive sign only; physics formulas below are written in the legacy convention (bit-invariant contract, test #1375).
  const double dV_legacy = FixedSign ? -deltaV : deltaV;
  if (fabs(dV_legacy) < 1.0e-30) {
    return 0.0;
  }

  const int i_donor = (dV_legacy > 0.0) ? i_left : i_right;
  remap_dispatch_audit_count(
      audit,
      RemapDispatchAuditCounter::LegacySweptVolume,
      cell_index(i_donor, j, nz));

  const int i_m = clampi(i_donor - 1, 0, nr - 1);
  const int i_p = clampi(i_donor + 1, 0, nr - 1);
  const int c_m = cell_index(i_m, j, nz);
  const int c_d = cell_index(i_donor, j, nz);
  const int c_p = cell_index(i_p, j, nz);

  if (geometry_policy_exempt != nullptr &&
      (geometry_policy_exempt[c_m] != 0 ||
       geometry_policy_exempt[c_p] != 0)) {
    return field[c_d] * dV_legacy;
  }

  const double q_m = field[c_m];
  const double q_d = field[c_d];
  const double q_p = field[c_p];

  const double den = q_p - q_d;
  double r_ratio = 0.0;
  if (fabs(den) >= 1.0e-30) {
    r_ratio = (q_d - q_m) / den;
  }
  const double phi = van_leer_phi(r_ratio);

  const int i_dx0 = clampi(i_donor - 1, 0, nr - 1);
  const int i_dx1 = clampi(i_donor + 1, 0, nr - 1);
  const double rc0 = cell_center_r(x_r_old, i_dx0, j, nz);
  const double rc1 = cell_center_r(x_r_old, i_dx1, j, nz);
  const double dx = fmax(fabs(rc1 - rc0), 1.0e-30);

  const double grad = phi * den / dx;
  const double d_cf = (dV_legacy > 0.0 ? 0.5 : -0.5) * dx;
  const double q_face = q_d + grad * d_cf;

  return q_face * dV_legacy;
}

template <bool FixedSign>
__device__ inline double flux_z_face_t(const double* field,
                                       const double* x_r_old,
                                       const double* x_z_old,
                                       const double* x_r_new,
                                       const double* x_z_new,
                                       const int i,
                                       const int j_face,
                                       const int nr,
                                       const int nz,
                                       const RemapDispatchAuditDeviceView audit = {},
                                       const std::uint8_t* geometry_policy_exempt = nullptr) {
  if (j_face <= 0 || j_face >= nz) {
    return 0.0;
  }

  const int j_low = j_face - 1;
  const int j_high = j_face;
  const int c_low = cell_index(i, j_low, nz);
  const int c_high = cell_index(i, j_high, nz);
  if (geometry_policy_exempt != nullptr &&
      (geometry_policy_exempt[c_low] != 0 ||
       geometry_policy_exempt[c_high] != 0)) {
    return 0.0;
  }

  const double deltaV =
      swept_volume_z_face_t<FixedSign>(x_r_old, x_z_old, x_r_new, x_z_new, i, j_face, nz);
  // FixedSign flips the stored primitive sign only; physics formulas below are written in the legacy convention (bit-invariant contract, test #1375).
  const double dV_legacy = FixedSign ? -deltaV : deltaV;
  if (fabs(dV_legacy) < 1.0e-30) {
    return 0.0;
  }

  const int j_donor = (dV_legacy > 0.0) ? j_low : j_high;
  remap_dispatch_audit_count(
      audit,
      RemapDispatchAuditCounter::LegacySweptVolume,
      cell_index(i, j_donor, nz));

  const int j_m = clampi(j_donor - 1, 0, nz - 1);
  const int j_p = clampi(j_donor + 1, 0, nz - 1);
  const int c_m = cell_index(i, j_m, nz);
  const int c_d = cell_index(i, j_donor, nz);
  const int c_p = cell_index(i, j_p, nz);

  if (geometry_policy_exempt != nullptr &&
      (geometry_policy_exempt[c_m] != 0 ||
       geometry_policy_exempt[c_p] != 0)) {
    return field[c_d] * dV_legacy;
  }

  const double q_m = field[c_m];
  const double q_d = field[c_d];
  const double q_p = field[c_p];

  const double den = q_p - q_d;
  double r_ratio = 0.0;
  if (fabs(den) >= 1.0e-30) {
    r_ratio = (q_d - q_m) / den;
  }
  const double phi = van_leer_phi(r_ratio);

  const int j_dx0 = clampi(j_donor - 1, 0, nz - 1);
  const int j_dx1 = clampi(j_donor + 1, 0, nz - 1);
  const double zc0 = cell_center_z(x_z_old, i, j_dx0, nz);
  const double zc1 = cell_center_z(x_z_old, i, j_dx1, nz);
  const double dx = fmax(fabs(zc1 - zc0), 1.0e-30);

  const double grad = phi * den / dx;
  const double d_cf = (dV_legacy > 0.0 ? 0.5 : -0.5) * dx;
  const double q_face = q_d + grad * d_cf;

  return q_face * dV_legacy;
}

struct RemapDamageResult {
  double D_rho_max = 0.0;
  double A_axis_max = 0.0;
  int axis_max_j = -1;
  std::vector<double> axis_inflow_this_event;
};

inline double remap_damage_face_factor(const double deltaV,
                                       const double rho_l,
                                       const double rho_r,
                                       const double vol_l,
                                       const double vol_r,
                                       const double rho_floor) {
  const double v_min = std::min(vol_l, vol_r);
  const double denom = rho_l + rho_r + rho_floor;
  if (!(std::isfinite(deltaV) && std::isfinite(v_min) && v_min > 0.0 &&
        std::isfinite(denom) && denom > 0.0)) {
    return 0.0;
  }
  return (std::abs(deltaV) / v_min) * (std::abs(rho_r - rho_l) / denom);
}

template <bool FixedSign>
inline RemapDamageResult check_remap_damage_per_face_t(
    const double* d_rho,
    const double* d_vol_old,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_xr_new,
    const double* d_xz_new,
    const int nr,
    const int nz,
    const double rho_floor,
    const std::vector<double>* axis_mass_initial = nullptr) {
  RemapDamageResult out;
  if (nz > 0) {
    out.axis_inflow_this_event.assign(static_cast<std::size_t>(nz), 0.0);
  }
  if (nr <= 0 || nz <= 0) {
    return out;
  }
  TENRYU_ASSERT(d_rho != nullptr, "ALE remap damage check requires rho");
  TENRYU_ASSERT(d_vol_old != nullptr, "ALE remap damage check requires old volumes");
  TENRYU_ASSERT(d_xr_old != nullptr && d_xz_old != nullptr,
                "ALE remap damage check requires old mesh coordinates");
  TENRYU_ASSERT(d_xr_new != nullptr && d_xz_new != nullptr,
                "ALE remap damage check requires trial mesh coordinates");
  if (axis_mass_initial != nullptr && !axis_mass_initial->empty()) {
    TENRYU_ASSERT(static_cast<int>(axis_mass_initial->size()) == nz,
                  "ALE remap damage axis mass size mismatch");
  }

  const std::size_t n_cells = static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz);
  const std::size_t n_nodes =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  std::vector<double> rho(n_cells, 0.0);
  std::vector<double> vol_old(n_cells, 0.0);
  std::vector<double> xr_old(n_nodes, 0.0);
  std::vector<double> xz_old(n_nodes, 0.0);
  std::vector<double> xr_new(n_nodes, 0.0);
  std::vector<double> xz_new(n_nodes, 0.0);

  CUDA_CHECK(cudaMemcpy(rho.data(), d_rho, n_cells * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(vol_old.data(), d_vol_old, n_cells * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(xr_old.data(), d_xr_old, n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(xz_old.data(), d_xz_old, n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(xr_new.data(), d_xr_new, n_nodes * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(
      cudaMemcpy(xz_new.data(), d_xz_new, n_nodes * sizeof(double), cudaMemcpyDeviceToHost));

  for (int i_face = 1; i_face < nr; ++i_face) {
    for (int j = 0; j < nz; ++j) {
      const int c_l = cell_index(i_face - 1, j, nz);
      const int c_r = cell_index(i_face, j, nz);
      const double deltaV =
          swept_volume_r_face_t<FixedSign>(xr_old.data(), xz_old.data(), xr_new.data(), xz_new.data(),
                                           i_face, j, nz);
      // FixedSign flips the stored primitive sign only; physics formulas below are written in the legacy convention (bit-invariant contract, test #1375).
      const double dV_legacy = FixedSign ? -deltaV : deltaV;
      out.D_rho_max = std::max(out.D_rho_max,
                               remap_damage_face_factor(dV_legacy,
                                                        rho[static_cast<std::size_t>(c_l)],
                                                        rho[static_cast<std::size_t>(c_r)],
                                                        vol_old[static_cast<std::size_t>(c_l)],
                                                        vol_old[static_cast<std::size_t>(c_r)],
                                                        rho_floor));
    }
  }

  for (int i = 0; i < nr; ++i) {
    for (int j_face = 1; j_face < nz; ++j_face) {
      const int c_l = cell_index(i, j_face - 1, nz);
      const int c_r = cell_index(i, j_face, nz);
      const double deltaV =
          swept_volume_z_face_t<FixedSign>(xr_old.data(), xz_old.data(), xr_new.data(), xz_new.data(),
                                           i, j_face, nz);
      // FixedSign flips the stored primitive sign only; physics formulas below are written in the legacy convention (bit-invariant contract, test #1375).
      const double dV_legacy = FixedSign ? -deltaV : deltaV;
      out.D_rho_max = std::max(out.D_rho_max,
                               remap_damage_face_factor(dV_legacy,
                                                        rho[static_cast<std::size_t>(c_l)],
                                                        rho[static_cast<std::size_t>(c_r)],
                                                        vol_old[static_cast<std::size_t>(c_l)],
                                                        vol_old[static_cast<std::size_t>(c_r)],
                                                        rho_floor));
    }
  }

  if (nr >= 2) {
    const bool has_axis_mass_initial =
        (axis_mass_initial != nullptr && !axis_mass_initial->empty());
    for (int j = 0; j < nz; ++j) {
      const int c0 = cell_index(0, j, nz);
      const int c1 = cell_index(1, j, nz);
      const double deltaV =
          swept_volume_r_face_t<FixedSign>(xr_old.data(), xz_old.data(), xr_new.data(), xz_new.data(),
                                           1, j, nz);
      // FixedSign flips the stored primitive sign only; physics formulas below are written in the legacy convention (bit-invariant contract, test #1375).
      const double dV_legacy = FixedSign ? -deltaV : deltaV;
      const double inflow_volume = std::max(0.0, -dV_legacy);
      const double density_excess =
          std::max(0.0, rho[static_cast<std::size_t>(c1)] -
                            rho[static_cast<std::size_t>(c0)]);
      const double inflow = inflow_volume * density_excess;
      out.axis_inflow_this_event[static_cast<std::size_t>(j)] = inflow;

      const double mass_base =
          has_axis_mass_initial
              ? (*axis_mass_initial)[static_cast<std::size_t>(j)]
              : rho[static_cast<std::size_t>(c0)] * vol_old[static_cast<std::size_t>(c0)];
      const double mass_floor = rho_floor * std::max(0.0, vol_old[static_cast<std::size_t>(c0)]);
      const double denom = mass_base + mass_floor;
      const double A = (denom > 0.0)
                           ? (inflow / denom)
                           : ((inflow > 0.0) ? std::numeric_limits<double>::infinity() : 0.0);
      if (A > out.A_axis_max) {
        out.A_axis_max = A;
        out.axis_max_j = j;
      }
    }
  }

  return out;
}

inline RemapDamageResult compute_remap_damage(
    const double* d_rho,
    const double* d_vol_old,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_xr_new,
    const double* d_xz_new,
    const int nr,
    const int nz,
    const double rho_floor,
    const std::vector<double>* axis_mass_initial,
    const bool swept_volume_sign_fixed) {
  if (swept_volume_sign_fixed) {
    return check_remap_damage_per_face_t<true>(d_rho,
                                               d_vol_old,
                                               d_xr_old,
                                               d_xz_old,
                                               d_xr_new,
                                               d_xz_new,
                                               nr,
                                               nz,
                                               rho_floor,
                                               axis_mass_initial);
  }
  return check_remap_damage_per_face_t<false>(d_rho,
                                              d_vol_old,
                                              d_xr_old,
                                              d_xz_old,
                                              d_xr_new,
                                              d_xz_new,
                                              nr,
                                              nz,
                                              rho_floor,
                                              axis_mass_initial);
}

inline bool remap_damage_axis_budget_exceeded(
    const RemapDamageResult& dmg,
    const std::vector<double>& axis_inflow_budget,
    const std::vector<double>& axis_mass_initial,
    const double budget_factor,
    int* fail_j = nullptr) {
  if (fail_j != nullptr) {
    *fail_j = -1;
  }
  const std::size_t n = dmg.axis_inflow_this_event.size();
  TENRYU_ASSERT(axis_inflow_budget.size() == n,
                "ALE remap damage axis budget size mismatch");
  TENRYU_ASSERT(axis_mass_initial.size() == n,
                "ALE remap damage axis initial mass size mismatch");
  for (std::size_t j = 0; j < n; ++j) {
    const double trial_budget = axis_inflow_budget[j] + dmg.axis_inflow_this_event[j];
    const double budget_cap = budget_factor * axis_mass_initial[j];
    if (trial_budget > budget_cap) {
      if (fail_j != nullptr) {
        *fail_j = static_cast<int>(j);
      }
      return true;
    }
  }
  return false;
}

}  // namespace detail

template <bool FixedSign>
static __global__ __launch_bounds__(256, 4) void conservative_remap_kernel_t(
    double* __restrict__ field_new,
    const double* __restrict__ field_old,
    const double* __restrict__ vol_old,
    const double* __restrict__ vol_new,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int sweep_direction,
    const int nr,
    const int nz,
    const std::uint8_t* __restrict__ geometry_policy_exempt,
    const RemapDispatchAuditDeviceView audit) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c] != 0) {
    field_new[c] = field_old[c];
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  const double qbar_old = field_old[c] * vol_old[c];

  double F_plus = 0.0;
  double F_minus = 0.0;
  if (sweep_direction == 0) {
    F_plus = detail::flux_r_face_t<FixedSign>(field_old,
                                              x_r_old,
                                              x_z_old,
                                              x_r_new,
                                              x_z_new,
                                              i + 1,
                                              j,
                                              nr,
                                              nz,
                                              audit,
                                              geometry_policy_exempt);
    F_minus = detail::flux_r_face_t<FixedSign>(field_old,
                                               x_r_old,
                                               x_z_old,
                                               x_r_new,
                                               x_z_new,
                                               i,
                                               j,
                                               nr,
                                               nz,
                                               audit,
                                               geometry_policy_exempt);
  } else {
    F_plus = detail::flux_z_face_t<FixedSign>(field_old,
                                              x_r_old,
                                              x_z_old,
                                              x_r_new,
                                              x_z_new,
                                              i,
                                              j + 1,
                                              nr,
                                              nz,
                                              audit,
                                              geometry_policy_exempt);
    F_minus = detail::flux_z_face_t<FixedSign>(field_old,
                                               x_r_old,
                                               x_z_old,
                                               x_r_new,
                                               x_z_new,
                                               i,
                                               j,
                                               nr,
                                               nz,
                                               audit,
                                               geometry_policy_exempt);
  }

  const double qbar_new = qbar_old - F_plus + F_minus;
  field_new[c] = qbar_new / fmax(vol_new[c], 1.0e-30);
}

template <bool FixedSign>
static __global__ __launch_bounds__(256, 4) void compute_intermediate_volume_kernel_t(
    double* __restrict__ vol_mid,
    int* __restrict__ nonpositive_count,
    const std::uint8_t* __restrict__ geometry_policy_exempt,
    int* __restrict__ d_first_nonpositive_cell,
    const double* __restrict__ vol_old,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int sweep_direction,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c] != 0) {
    vol_mid[c] = vol_old[c];
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  const double deltaV_plus =
      (sweep_direction == 0)
          ? detail::swept_volume_r_face_t<FixedSign>(
                x_r_old, x_z_old, x_r_new, x_z_new, i + 1, j, nz)
          : detail::swept_volume_z_face_t<FixedSign>(
                x_r_old, x_z_old, x_r_new, x_z_new, i, j + 1, nz);
  // FixedSign flips the stored primitive sign only; physics formulas below are written in the legacy convention (bit-invariant contract, test #1375).
  const double dV_plus_legacy = FixedSign ? -deltaV_plus : deltaV_plus;
  const double deltaV_minus =
      (sweep_direction == 0)
          ? detail::swept_volume_r_face_t<FixedSign>(
                x_r_old, x_z_old, x_r_new, x_z_new, i, j, nz)
          : detail::swept_volume_z_face_t<FixedSign>(
                x_r_old, x_z_old, x_r_new, x_z_new, i, j, nz);
  // FixedSign flips the stored primitive sign only; physics formulas below are written in the legacy convention (bit-invariant contract, test #1375).
  const double dV_minus_legacy = FixedSign ? -deltaV_minus : deltaV_minus;

  const double vol = vol_old[c] - dV_plus_legacy + dV_minus_legacy;
  const bool invalid_vol = (!isfinite(vol)) || !(vol > 0.0);
  vol_mid[c] = invalid_vol ? 0.0 : vol;
  if (invalid_vol) {
    atomicAdd(nonpositive_count, 1);
    atomicMin(d_first_nonpositive_cell, c);
  }
}

inline void launch_conservative_remap(double* d_field_new,
                                      const double* d_field_old,
                                      const double* d_vol_old,
                                      const double* d_vol_new,
                                      const double* d_xr_old,
                                      const double* d_xz_old,
                                      const double* d_xr_new,
                                      const double* d_xz_new,
                                      const int sweep_direction,
                                      const int nr,
                                      const int nz,
                                      const bool swept_volume_sign_fixed,
                                      const RemapDispatchAuditDeviceView audit = {},
                                      cudaStream_t stream = nullptr,
                                      const std::uint8_t* d_geometry_policy_exempt = nullptr) {
  const int n_cells = nr * nz;
  const int blocks = (n_cells + 255) / 256;
  if (swept_volume_sign_fixed) {
    conservative_remap_kernel_t<true><<<blocks, 256, 0, stream>>>(d_field_new,
                                                                   d_field_old,
                                                                   d_vol_old,
                                                                   d_vol_new,
                                                                   d_xr_old,
                                                                   d_xz_old,
                                                                   d_xr_new,
                                                                   d_xz_new,
                                                                   sweep_direction,
                                                                   nr,
                                                                   nz,
                                                                   d_geometry_policy_exempt,
                                                                   audit);
  } else {
    conservative_remap_kernel_t<false><<<blocks, 256, 0, stream>>>(d_field_new,
                                                                    d_field_old,
                                                                    d_vol_old,
                                                                    d_vol_new,
                                                                    d_xr_old,
                                                                    d_xz_old,
                                                                    d_xr_new,
                                                                    d_xz_new,
                                                                    sweep_direction,
                                                                    nr,
                                                                    nz,
                                                                    d_geometry_policy_exempt,
                                                                    audit);
  }
}

inline bool launch_remap_strang(double* d_field_inout,
                                double* d_field_tmp,
                                double* d_vol_mid,
                                const double* d_vol_old,
                                const double* d_vol_new,
                                const double* d_xr_old,
                                const double* d_xz_old,
                                const double* d_xr_new,
                                const double* d_xz_new,
                                const int nr,
                                const int nz,
                                const int step_number,
                                const bool swept_volume_sign_fixed,
                                const std::uint8_t* d_geometry_policy_exempt = nullptr,
                                const RemapDispatchAuditDeviceView audit = {},
                                cudaStream_t stream = nullptr) {
  const int first_dir = (step_number % 2 == 0) ? 0 : 1;
  const int second_dir = 1 - first_dir;
  const int n_cells = nr * nz;
  const int blocks = (n_cells + 255) / 256;
  int* d_nonpositive_count = nullptr;
  int* d_first_nonpositive_cell = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nonpositive_count), sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_first_nonpositive_cell), sizeof(int)));
  CUDA_CHECK(cudaMemsetAsync(d_nonpositive_count, 0, sizeof(int), stream));
  int first_nonpositive_cell = INT_MAX;
  CUDA_CHECK(cudaMemcpyAsync(d_first_nonpositive_cell,
                             &first_nonpositive_cell,
                             sizeof(int),
                             cudaMemcpyHostToDevice,
                             stream));

  if (swept_volume_sign_fixed) {
    compute_intermediate_volume_kernel_t<true><<<blocks, 256, 0, stream>>>(d_vol_mid,
                                                                            d_nonpositive_count,
                                                                            d_geometry_policy_exempt,
                                                                            d_first_nonpositive_cell,
                                                                            d_vol_old,
                                                                            d_xr_old,
                                                                            d_xz_old,
                                                                            d_xr_new,
                                                                            d_xz_new,
                                                                            first_dir,
                                                                            nr,
                                                                            nz);
  } else {
    compute_intermediate_volume_kernel_t<false><<<blocks, 256, 0, stream>>>(d_vol_mid,
                                                                             d_nonpositive_count,
                                                                             d_geometry_policy_exempt,
                                                                             d_first_nonpositive_cell,
                                                                             d_vol_old,
                                                                             d_xr_old,
                                                                             d_xz_old,
                                                                             d_xr_new,
                                                                             d_xz_new,
                                                                             first_dir,
                                                                             nr,
                                                                             nz);
  }
  CUDA_CHECK(cudaGetLastError());

  int nonpositive_count = 0;
  if (stream == nullptr) {
    CUDA_CHECK(cudaMemcpy(&nonpositive_count,
                          d_nonpositive_count,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
  } else {
    CUDA_CHECK(cudaMemcpyAsync(&nonpositive_count,
                               d_nonpositive_count,
                               sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  if (nonpositive_count > 0) {
    if (stream == nullptr) {
      CUDA_CHECK(cudaMemcpy(&first_nonpositive_cell,
                            d_first_nonpositive_cell,
                            sizeof(int),
                            cudaMemcpyDeviceToHost));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(&first_nonpositive_cell,
                                 d_first_nonpositive_cell,
                                 sizeof(int),
                                 cudaMemcpyDeviceToHost,
                                 stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  CUDA_CHECK(cudaFree(d_first_nonpositive_cell));
  CUDA_CHECK(cudaFree(d_nonpositive_count));
  if (nonpositive_count > 0) {
    core::log_warning(
        "[ale-remap] transport aborted: non-positive intermediate volume at cell=" +
        std::to_string(first_nonpositive_cell) +
        " (i=" + std::to_string(first_nonpositive_cell / nz) +
        ", j=" + std::to_string(first_nonpositive_cell % nz) +
        ") count=" + std::to_string(nonpositive_count));
    return false;
  }

  launch_conservative_remap(d_field_tmp,
                            d_field_inout,
                            d_vol_old,
                            d_vol_mid,
                            d_xr_old,
                            d_xz_old,
                            d_xr_new,
                            d_xz_new,
                            first_dir,
                            nr,
                            nz,
                            swept_volume_sign_fixed,
                            audit,
                            stream,
                            d_geometry_policy_exempt);
  CUDA_CHECK(cudaGetLastError());
  if (stream == nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  launch_conservative_remap(d_field_inout,
                            d_field_tmp,
                            d_vol_mid,
                            d_vol_new,
                            d_xr_old,
                            d_xz_old,
                            d_xr_new,
                            d_xz_new,
                            second_dir,
                            nr,
                            nz,
                            swept_volume_sign_fixed,
                            audit,
                            stream,
                            d_geometry_policy_exempt);
  CUDA_CHECK(cudaGetLastError());
  if (stream == nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  return true;
}

enum class RemapMs2Limiter {
  VanLeer = 0,
  BarthJespersen = 1,
};

namespace detail {

__device__ inline void ms2_accumulate_lstsq_neighbor(const double* field,
                                                     const double* x_r,
                                                     const double* x_z,
                                                     double& a00,
                                                     double& a01,
                                                     double& a11,
                                                     double& b0,
                                                     double& b1,
                                                     const double q_c,
                                                     const double r_c,
                                                     const double z_c,
                                                     const int i_n,
                                                     const int j_n,
                                                     const int nz) {
  const int c_n = cell_index(i_n, j_n, nz);
  const double dr = cell_center_r(x_r, i_n, j_n, nz) - r_c;
  const double dz = cell_center_z(x_z, i_n, j_n, nz) - z_c;
  const double dist = fmax(sqrt(dr * dr + dz * dz), 1.0e-30);
  const double w = 1.0 / dist;
  const double dq = field[c_n] - q_c;

  a00 += w * dr * dr;
  a01 += w * dr * dz;
  a11 += w * dz * dz;
  b0 += w * dr * dq;
  b1 += w * dz * dq;
}

__device__ inline double ms2_van_leer_limiter_ratio(const double q_c,
                                                    const double q_neighbor,
                                                    const double q_f) {
  const double dq_f = q_f - q_c;
  if (fabs(dq_f) <= 1.0e-30) {
    return 1.0;
  }
  return van_leer_phi((q_neighbor - q_c) / dq_f);
}

__device__ inline double ms2_barth_jespersen_limiter_ratio(const double q_c,
                                                           const double q_min,
                                                           const double q_max,
                                                           const double q_f) {
  const double dq = q_f - q_c;
  if (q_f > q_max && fabs(dq) > 1.0e-30) {
    return (q_max - q_c) / dq;
  }
  if (q_f < q_min && fabs(dq) > 1.0e-30) {
    return (q_min - q_c) / dq;
  }
  return 1.0;
}

__device__ inline double ms2_van_leer_limiter_factor(const double* field,
                                                     const double* x_r,
                                                     const double* x_z,
                                                     const double grad_r,
                                                     const double grad_z,
                                                     const int i,
                                                     const int j,
                                                     const int nr,
                                                     const int nz,
                                                     const std::uint8_t* geometry_policy_exempt) {
  const int c = cell_index(i, j, nz);
  const double q_c = field[c];
  double q_neighbor[4] = {q_c, q_c, q_c, q_c};
  bool use_neighbor[4] = {true, true, true, true};

  if (i > 0) {
    const int c_n = cell_index(i - 1, j, nz);
    if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c_n] != 0) {
      use_neighbor[0] = false;
    } else {
      q_neighbor[0] = field[c_n];
    }
  }
  if (i + 1 < nr) {
    const int c_n = cell_index(i + 1, j, nz);
    if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c_n] != 0) {
      use_neighbor[1] = false;
    } else {
      q_neighbor[1] = field[c_n];
    }
  }
  if (j > 0) {
    const int c_n = cell_index(i, j - 1, nz);
    if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c_n] != 0) {
      use_neighbor[2] = false;
    } else {
      q_neighbor[2] = field[c_n];
    }
  }
  if (j + 1 < nz) {
    const int c_n = cell_index(i, j + 1, nz);
    if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c_n] != 0) {
      use_neighbor[3] = false;
    } else {
      q_neighbor[3] = field[c_n];
    }
  }

  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  const double r_c = 0.25 * (x_r[n00] + x_r[n10] + x_r[n11] + x_r[n01]);
  const double z_c = 0.25 * (x_z[n00] + x_z[n10] + x_z[n11] + x_z[n01]);

  double psi = 1.0;
  const double r_face[4] = {
      0.5 * (x_r[n00] + x_r[n01]),
      0.5 * (x_r[n10] + x_r[n11]),
      0.5 * (x_r[n00] + x_r[n10]),
      0.5 * (x_r[n01] + x_r[n11])};
  const double z_face[4] = {
      0.5 * (x_z[n00] + x_z[n01]),
      0.5 * (x_z[n10] + x_z[n11]),
      0.5 * (x_z[n00] + x_z[n10]),
      0.5 * (x_z[n01] + x_z[n11])};

  for (int f = 0; f < 4; ++f) {
    if (!use_neighbor[f]) {
      continue;
    }
    const double q_f = q_c + grad_r * (r_face[f] - r_c) + grad_z * (z_face[f] - z_c);
    psi = fmin(psi, ms2_van_leer_limiter_ratio(q_c, q_neighbor[f], q_f));
  }

  if (!isfinite(psi)) {
    return 0.0;
  }
  return fmin(1.0, fmax(0.0, psi));
}

__device__ inline double ms2_barth_jespersen_limiter_factor(const double* field,
                                                           const double* x_r,
                                                           const double* x_z,
                                                           const double grad_r,
                                                           const double grad_z,
                                                           const int i,
                                                           const int j,
                                                           const int nr,
                                                           const int nz,
                                                           const std::uint8_t* geometry_policy_exempt) {
  const int c = cell_index(i, j, nz);
  const double q_c = field[c];
  double q_min = q_c;
  double q_max = q_c;

  if (i > 0) {
    const int c_n = cell_index(i - 1, j, nz);
    if (geometry_policy_exempt == nullptr || geometry_policy_exempt[c_n] == 0) {
      const double q = field[c_n];
      q_min = fmin(q_min, q);
      q_max = fmax(q_max, q);
    }
  }
  if (i + 1 < nr) {
    const int c_n = cell_index(i + 1, j, nz);
    if (geometry_policy_exempt == nullptr || geometry_policy_exempt[c_n] == 0) {
      const double q = field[c_n];
      q_min = fmin(q_min, q);
      q_max = fmax(q_max, q);
    }
  }
  if (j > 0) {
    const int c_n = cell_index(i, j - 1, nz);
    if (geometry_policy_exempt == nullptr || geometry_policy_exempt[c_n] == 0) {
      const double q = field[c_n];
      q_min = fmin(q_min, q);
      q_max = fmax(q_max, q);
    }
  }
  if (j + 1 < nz) {
    const int c_n = cell_index(i, j + 1, nz);
    if (geometry_policy_exempt == nullptr || geometry_policy_exempt[c_n] == 0) {
      const double q = field[c_n];
      q_min = fmin(q_min, q);
      q_max = fmax(q_max, q);
    }
  }

  if (q_max - q_min <= 1.0e-30) {
    return 0.0;
  }

  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  const double r_c = 0.25 * (x_r[n00] + x_r[n10] + x_r[n11] + x_r[n01]);
  const double z_c = 0.25 * (x_z[n00] + x_z[n10] + x_z[n11] + x_z[n01]);

  double psi = 1.0;
  const double r_face[4] = {
      0.5 * (x_r[n00] + x_r[n01]),
      0.5 * (x_r[n10] + x_r[n11]),
      0.5 * (x_r[n00] + x_r[n10]),
      0.5 * (x_r[n01] + x_r[n11])};
  const double z_face[4] = {
      0.5 * (x_z[n00] + x_z[n01]),
      0.5 * (x_z[n10] + x_z[n11]),
      0.5 * (x_z[n00] + x_z[n10]),
      0.5 * (x_z[n01] + x_z[n11])};

  for (int f = 0; f < 4; ++f) {
    const double q_f = q_c + grad_r * (r_face[f] - r_c) + grad_z * (z_face[f] - z_c);
    psi = fmin(psi, ms2_barth_jespersen_limiter_ratio(q_c, q_min, q_max, q_f));
  }

  if (!isfinite(psi)) {
    return 0.0;
  }
  return fmin(1.0, fmax(0.0, psi));
}

template <bool FixedSign>
__device__ inline double ms2_swept_moment_r_face_t(const double* x_r_old,
                                                   const double* x_z_old,
                                                   const double* x_r_new,
                                                   const double* x_z_new,
                                                   const int i_face,
                                                   const int j,
                                                   const int nz) {
  const int n0 = node_index(i_face, j, nz);
  const int n1 = node_index(i_face, j + 1, nz);
  const double raw = rz_signed_quad_moment_r(x_r_old[n0],
                                             x_z_old[n0],
                                             x_r_new[n0],
                                             x_z_new[n0],
                                             x_r_new[n1],
                                             x_z_new[n1],
                                             x_r_old[n1],
                                             x_z_old[n1]);
  return FixedSign ? -raw : raw;
}

__device__ inline double ms2_swept_moment_r_face(const double* x_r_old,
                                                 const double* x_z_old,
                                                 const double* x_r_new,
                                                 const double* x_z_new,
                                                 const int i_face,
                                                 const int j,
                                                 const int nz) {
  return ms2_swept_moment_r_face_t<false>(
      x_r_old, x_z_old, x_r_new, x_z_new, i_face, j, nz);
}

template <bool FixedSign>
__device__ inline double ms2_swept_moment_z_face_t(const double* x_r_old,
                                                   const double* x_z_old,
                                                   const double* x_r_new,
                                                   const double* x_z_new,
                                                   const int i,
                                                   const int j_face,
                                                   const int nz) {
  const int n0 = node_index(i, j_face, nz);
  const int n1 = node_index(i + 1, j_face, nz);
  const double raw = rz_signed_quad_moment_z(x_r_old[n0],
                                             x_z_old[n0],
                                             x_r_old[n1],
                                             x_z_old[n1],
                                             x_r_new[n1],
                                             x_z_new[n1],
                                             x_r_new[n0],
                                             x_z_new[n0]);
  return FixedSign ? -raw : raw;
}

__device__ inline double ms2_swept_moment_z_face(const double* x_r_old,
                                                 const double* x_z_old,
                                                 const double* x_r_new,
                                                 const double* x_z_new,
                                                 const int i,
                                                 const int j_face,
                                                 const int nz) {
  return ms2_swept_moment_z_face_t<false>(
      x_r_old, x_z_old, x_r_new, x_z_new, i, j_face, nz);
}

template <bool FixedSign>
__device__ inline double ms2_swept_moment_r_z_face_t(const double* x_r_old,
                                                     const double* x_z_old,
                                                     const double* x_r_new,
                                                     const double* x_z_new,
                                                     const int i,
                                                     const int j_face,
                                                     const int nz) {
  const int n0 = node_index(i, j_face, nz);
  const int n1 = node_index(i + 1, j_face, nz);
  const double raw = rz_signed_quad_moment_r(x_r_old[n0],
                                             x_z_old[n0],
                                             x_r_old[n1],
                                             x_z_old[n1],
                                             x_r_new[n1],
                                             x_z_new[n1],
                                             x_r_new[n0],
                                             x_z_new[n0]);
  return FixedSign ? -raw : raw;
}

__device__ inline double ms2_swept_moment_r_z_face(const double* x_r_old,
                                                   const double* x_z_old,
                                                   const double* x_r_new,
                                                   const double* x_z_new,
                                                   const int i,
                                                   const int j_face,
                                                   const int nz) {
  return ms2_swept_moment_r_z_face_t<false>(
      x_r_old, x_z_old, x_r_new, x_z_new, i, j_face, nz);
}

template <bool FixedSign>
__device__ inline double ms2_swept_moment_z_r_face_t(const double* x_r_old,
                                                     const double* x_z_old,
                                                     const double* x_r_new,
                                                     const double* x_z_new,
                                                     const int i_face,
                                                     const int j,
                                                     const int nz) {
  const int n0 = node_index(i_face, j, nz);
  const int n1 = node_index(i_face, j + 1, nz);
  const double raw = rz_signed_quad_moment_z(x_r_old[n0],
                                             x_z_old[n0],
                                             x_r_new[n0],
                                             x_z_new[n0],
                                             x_r_new[n1],
                                             x_z_new[n1],
                                             x_r_old[n1],
                                             x_z_old[n1]);
  return FixedSign ? -raw : raw;
}

__device__ inline double ms2_swept_moment_z_r_face(const double* x_r_old,
                                                   const double* x_z_old,
                                                   const double* x_r_new,
                                                   const double* x_z_new,
                                                   const int i_face,
                                                   const int j,
                                                   const int nz) {
  return ms2_swept_moment_z_r_face_t<false>(
      x_r_old, x_z_old, x_r_new, x_z_new, i_face, j, nz);
}

__host__ __device__ inline int csr_canonical_edge_for_local_face(
    const int local_face) {
  return (local_face == 0) ? 3 : ((local_face == 1) ? 1 : ((local_face == 2) ? 0 : 2));
}

__host__ __device__ inline bool csr_active_local_face_corners(
    const std::uint8_t* cell_nverts,
    const int cell,
    const int local_face,
    int* corner0,
    int* corner1) {
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
  return tenryu::mesh::mesh_topo_active_local_face_corners(
      active_nverts, local_face, corner0, corner1);
}

__host__ __device__ inline double csr_cell_orientation_sign(
    const int cell,
    const int* __restrict__ cell_orientation_sign) {
  return static_cast<double>(cell_orientation_sign[cell]);
}

enum class CsrFaceSweptMomentsStatus : std::uint8_t {
  OK = 0,
  SKIP = 1,
};

__host__ __device__ inline double csr_cell_center_r(
    const double* x_r,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const int cell,
    const std::uint8_t* cell_nverts = nullptr) {
  const int off = cell_node_csr_offsets[cell];
  if (cell_nverts == nullptr) {
    double r = 0.0;
    for (int k = 0; k < 4; ++k) {
      r += x_r[cell_node_csr_indices[off + k]];
    }
    return 0.25 * r;
  }
  double r = 0.0;
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
  for (int k = 0; k < active_nverts; ++k) {
    r += x_r[cell_node_csr_indices[off + k]];
  }
  return r / static_cast<double>(active_nverts);
}

__host__ __device__ inline double csr_cell_center_z(
    const double* x_z,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const int cell,
    const std::uint8_t* cell_nverts = nullptr) {
  const int off = cell_node_csr_offsets[cell];
  if (cell_nverts == nullptr) {
    double z = 0.0;
    for (int k = 0; k < 4; ++k) {
      z += x_z[cell_node_csr_indices[off + k]];
    }
    return 0.25 * z;
  }
  double z = 0.0;
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
  for (int k = 0; k < active_nverts; ++k) {
    z += x_z[cell_node_csr_indices[off + k]];
  }
  return z / static_cast<double>(active_nverts);
}

__host__ __device__ inline bool csr_face_center(
    const double* x_r,
    const double* x_z,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const int cell,
    const int local_face,
    double& r_face,
    double& z_face,
    const std::uint8_t* cell_nverts = nullptr) {
  int corner0 = 0;
  int corner1 = 0;
  if (!csr_active_local_face_corners(
          cell_nverts, cell, local_face, &corner0, &corner1)) {
    r_face = 0.0;
    z_face = 0.0;
    return false;
  }
  const int off = cell_node_csr_offsets[cell];
  const int na = cell_node_csr_indices[off + corner0];
  const int nb = cell_node_csr_indices[off + corner1];
  r_face = 0.5 * (x_r[na] + x_r[nb]);
  z_face = 0.5 * (x_z[na] + x_z[nb]);
  return true;
}

__host__ __device__ inline bool csr_face_swept_node_indices(
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const int cell,
    const int local_face,
    int* na,
    int* nb) {
  int corner0 = 0;
  int corner1 = 0;
  if (!csr_active_local_face_corners(
          cell_nverts, cell, local_face, &corner0, &corner1)) {
    *na = -1;
    *nb = -1;
    return false;
  }
  const int off = cell_node_csr_offsets[cell];
  *na = cell_node_csr_indices[off + corner0];
  *nb = cell_node_csr_indices[off + corner1];
  return true;
}

__host__ __device__ inline double csr_face_swept_volume_outward(
    const double* x_r_old,
    const double* x_z_old,
    const double* x_r_new,
    const double* x_z_new,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const int cell,
    const int local_face,
    const std::uint8_t* cell_nverts = nullptr) {
  int na = -1;
  int nb = -1;
  if (!csr_face_swept_node_indices(cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_nverts,
                                   cell,
                                   local_face,
                                   &na,
                                   &nb)) {
    return 0.0;
  }
  if (x_r_old[na] == x_r_new[na] && x_z_old[na] == x_z_new[na] &&
      x_r_old[nb] == x_r_new[nb] && x_z_old[nb] == x_z_new[nb]) {
    return 0.0;
  }
  return csr_cell_orientation_sign(cell, cell_orientation_sign) *
         face_swept_volume_outward(x_r_old[na],
                                   x_z_old[na],
                                   x_r_old[nb],
                                   x_z_old[nb],
                                   x_r_new[na],
                                   x_z_new[na],
                                   x_r_new[nb],
                                   x_z_new[nb]);
}

__host__ __device__ inline double csr_face_swept_moments_outward(
    const double* x_r_old,
    const double* x_z_old,
    const double* x_r_new,
    const double* x_z_new,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const int cell,
    const int local_face,
    const std::uint8_t* cell_nverts,
    double* rq,
    double* zq,
    CsrFaceSweptMomentsStatus* status,
    bool* exact_moments = nullptr) {
  if (rq != nullptr) {
    *rq = 0.0;
  }
  if (zq != nullptr) {
    *zq = 0.0;
  }
  if (status != nullptr) {
    *status = CsrFaceSweptMomentsStatus::SKIP;
  }
  if (exact_moments != nullptr) {
    *exact_moments = false;
  }

  int na = -1;
  int nb = -1;
  if (!csr_face_swept_node_indices(cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_nverts,
                                   cell,
                                   local_face,
                                   &na,
                                   &nb)) {
    return 0.0;
  }
  if (x_r_old[na] == x_r_new[na] && x_z_old[na] == x_z_new[na] &&
      x_r_old[nb] == x_r_new[nb] && x_z_old[nb] == x_z_new[nb]) {
    return 0.0;
  }

  const double dV_csr = csr_face_swept_volume_outward(x_r_old,
                                                      x_z_old,
                                                      x_r_new,
                                                      x_z_new,
                                                      cell_node_csr_offsets,
                                                      cell_node_csr_indices,
                                                      cell_orientation_sign,
                                                      cell,
                                                      local_face,
                                                      cell_nverts);
  if (!isfinite(dV_csr) || dV_csr == 0.0) {
    return dV_csr;
  }

  // rq/zq is the swept-quad centroid, not moment/dV_csr. Affine exactness
  // only needs shared dm_q debit/credit and linear-precision weights at a
  // finite admissible packet location.
  const double r0 = x_r_old[na];
  const double z0 = x_z_old[na];
  const double r1 = x_r_old[nb];
  const double z1 = x_z_old[nb];
  const double r2 = x_r_new[nb];
  const double z2 = x_z_new[nb];
  const double r3 = x_r_new[na];
  const double z3 = x_z_new[na];
  const double quad_volume =
      rz_signed_quad_volume(r0, z0, r1, z1, r2, z2, r3, z3);
  if (isfinite(quad_volume) && quad_volume != 0.0) {
    const double rq_candidate =
        rz_signed_quad_moment_r(r0, z0, r1, z1, r2, z2, r3, z3) /
        quad_volume;
    const double zq_candidate =
        rz_signed_quad_moment_z(r0, z0, r1, z1, r2, z2, r3, z3) /
        quad_volume;
    if (isfinite(rq_candidate) && isfinite(zq_candidate)) {
      if (rq != nullptr) {
        *rq = rq_candidate;
      }
      if (zq != nullptr) {
        *zq = zq_candidate;
      }
      if (status != nullptr) {
        *status = CsrFaceSweptMomentsStatus::OK;
      }
      if (exact_moments != nullptr) {
        *exact_moments = true;
      }
      return dV_csr;
    }
  }

  if (rq != nullptr) {
    *rq = 0.25 * (r0 + r1 + r2 + r3);
  }
  if (zq != nullptr) {
    *zq = 0.25 * (z0 + z1 + z2 + z3);
  }
  if (status != nullptr) {
    *status = CsrFaceSweptMomentsStatus::OK;
  }
  return dV_csr;
}

__device__ inline double csr_barth_jespersen_limiter_ratio(
    const double q_c,
    const double q_min,
    const double q_max,
    const double q_face) {
  const double dq = q_face - q_c;
  if (fabs(dq) <= 1.0e-300 || !isfinite(dq)) {
    return 1.0;
  }
  const double eps_tol = 1.0e-14 * fmax(fabs(q_max - q_min), 1.0);
  if (dq > eps_tol) {
    return fmin(1.0, fmax(0.0, (q_max - q_c) / dq));
  }
  if (dq < -eps_tol) {
    return fmin(1.0, fmax(0.0, (q_min - q_c) / dq));
  }
  return 1.0;
}

__device__ inline double csr_limited_reconstruction(
    const double* field,
    const double* grad_r,
    const double* grad_z,
    const double* x_r,
    const double* x_z,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const int cell,
    const double r_face,
    const double z_face,
    const double floor_value,
    const std::uint8_t* cell_nverts = nullptr,
    const RemapDispatchAuditDeviceView audit = {}) {
  const double q = field[cell];
  const double r_c =
      csr_cell_center_r(
          x_r, cell_node_csr_offsets, cell_node_csr_indices, cell, cell_nverts);
  const double z_c =
      csr_cell_center_z(
          x_z, cell_node_csr_offsets, cell_node_csr_indices, cell, cell_nverts);
  double q_face = q + grad_r[cell] * (r_face - r_c) +
                  grad_z[cell] * (z_face - z_c);
  if (!isfinite(q_face)) {
    remap_dispatch_audit_count(
        audit,
        RemapDispatchAuditCounter::ReconstructionNonfiniteFallback,
        cell);
    q_face = q;
  }
  if (floor_value >= 0.0) {
    q_face = fmax(q_face, floor_value);
  }
  return q_face;
}

template <bool FixedSign>
__device__ inline double ms2_flux_r_face_t(const double* field,
                                           const double* grad_r,
                                           const double* grad_z,
                                           const double* x_r_old,
                                           const double* x_z_old,
                                           const double* x_r_new,
                                           const double* x_z_new,
                                           const int i_face,
                                           const int j,
                                           const int nr,
                                           const int nz,
                                           const RemapDispatchAuditDeviceView audit = {},
                                           const std::uint8_t* geometry_policy_exempt = nullptr) {
  if (i_face <= 0 || i_face >= nr) {
    return 0.0;
  }

  const int c_left = cell_index(i_face - 1, j, nz);
  const int c_right = cell_index(i_face, j, nz);
  if (geometry_policy_exempt != nullptr &&
      (geometry_policy_exempt[c_left] != 0 ||
       geometry_policy_exempt[c_right] != 0)) {
    return 0.0;
  }

  const double deltaV =
      swept_volume_r_face_t<FixedSign>(x_r_old, x_z_old, x_r_new, x_z_new, i_face, j, nz);
  if (fabs(deltaV) < 1.0e-30) {
    return 0.0;
  }

  const int i_donor = (deltaV > 0.0) ? (i_face - 1) : i_face;
  const int c_d = cell_index(i_donor, j, nz);
  remap_dispatch_audit_count(
      audit, RemapDispatchAuditCounter::ExactSweptMoment, c_d);
  const double r_c = cell_center_r(x_r_old, i_donor, j, nz);
  const double z_c = cell_center_z(x_z_old, i_donor, j, nz);
  const double m_r =
      ms2_swept_moment_r_face_t<FixedSign>(x_r_old, x_z_old, x_r_new, x_z_new, i_face, j, nz);
  const double m_z =
      ms2_swept_moment_z_r_face_t<FixedSign>(x_r_old, x_z_old, x_r_new, x_z_new, i_face, j, nz);

  return field[c_d] * deltaV + grad_r[c_d] * (m_r - r_c * deltaV) +
         grad_z[c_d] * (m_z - z_c * deltaV);
}

template <bool FixedSign>
__device__ inline double ms2_flux_z_face_t(const double* field,
                                           const double* grad_r,
                                           const double* grad_z,
                                           const double* x_r_old,
                                           const double* x_z_old,
                                           const double* x_r_new,
                                           const double* x_z_new,
                                           const int i,
                                           const int j_face,
                                           const int nr,
                                           const int nz,
                                           const RemapDispatchAuditDeviceView audit = {},
                                           const std::uint8_t* geometry_policy_exempt = nullptr) {
  if (j_face <= 0 || j_face >= nz) {
    return 0.0;
  }

  const int c_low = cell_index(i, j_face - 1, nz);
  const int c_high = cell_index(i, j_face, nz);
  if (geometry_policy_exempt != nullptr &&
      (geometry_policy_exempt[c_low] != 0 ||
       geometry_policy_exempt[c_high] != 0)) {
    return 0.0;
  }

  const double deltaV =
      swept_volume_z_face_t<FixedSign>(x_r_old, x_z_old, x_r_new, x_z_new, i, j_face, nz);
  if (fabs(deltaV) < 1.0e-30) {
    return 0.0;
  }

  const int j_donor = (deltaV > 0.0) ? (j_face - 1) : j_face;
  const int c_d = cell_index(i, j_donor, nz);
  remap_dispatch_audit_count(
      audit, RemapDispatchAuditCounter::ExactSweptMoment, c_d);
  const double r_c = cell_center_r(x_r_old, i, j_donor, nz);
  const double z_c = cell_center_z(x_z_old, i, j_donor, nz);
  const double m_r =
      ms2_swept_moment_r_z_face_t<FixedSign>(x_r_old, x_z_old, x_r_new, x_z_new, i, j_face, nz);
  const double m_z =
      ms2_swept_moment_z_face_t<FixedSign>(x_r_old, x_z_old, x_r_new, x_z_new, i, j_face, nz);

  return field[c_d] * deltaV + grad_r[c_d] * (m_r - r_c * deltaV) +
         grad_z[c_d] * (m_z - z_c * deltaV);
}

}  // namespace detail

static __global__ __launch_bounds__(256, 4) void compute_cell_slopes_lstsq_2d_rz_kernel(
    double* __restrict__ grad_r,
    double* __restrict__ grad_z,
    const double* __restrict__ field,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz,
    const RemapDispatchAuditDeviceView audit,
    const std::uint8_t* __restrict__ geometry_policy_exempt = nullptr) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c] != 0) {
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const double q_c = field[c];
  const double r_c = detail::cell_center_r(x_r, i, j, nz);
  const double z_c = detail::cell_center_z(x_z, i, j, nz);

  double a00 = 0.0;
  double a01 = 0.0;
  double a11 = 0.0;
  double b0 = 0.0;
  double b1 = 0.0;

  if (i > 0) {
    const int c_n = detail::cell_index(i - 1, j, nz);
    if (geometry_policy_exempt == nullptr || geometry_policy_exempt[c_n] == 0) {
      detail::ms2_accumulate_lstsq_neighbor(
          field, x_r, x_z, a00, a01, a11, b0, b1, q_c, r_c, z_c, i - 1, j, nz);
    }
  }
  if (i + 1 < nr) {
    const int c_n = detail::cell_index(i + 1, j, nz);
    if (geometry_policy_exempt == nullptr || geometry_policy_exempt[c_n] == 0) {
      detail::ms2_accumulate_lstsq_neighbor(
          field, x_r, x_z, a00, a01, a11, b0, b1, q_c, r_c, z_c, i + 1, j, nz);
    }
  }
  if (j > 0) {
    const int c_n = detail::cell_index(i, j - 1, nz);
    if (geometry_policy_exempt == nullptr || geometry_policy_exempt[c_n] == 0) {
      detail::ms2_accumulate_lstsq_neighbor(
          field, x_r, x_z, a00, a01, a11, b0, b1, q_c, r_c, z_c, i, j - 1, nz);
    }
  }
  if (j + 1 < nz) {
    const int c_n = detail::cell_index(i, j + 1, nz);
    if (geometry_policy_exempt == nullptr || geometry_policy_exempt[c_n] == 0) {
      detail::ms2_accumulate_lstsq_neighbor(
          field, x_r, x_z, a00, a01, a11, b0, b1, q_c, r_c, z_c, i, j + 1, nz);
    }
  }

  const double det = a00 * a11 - a01 * a01;
  const double scale = fabs(a00 * a11) + fabs(a01 * a01) + 1.0e-300;
  if (fabs(det) > 1.0e-24 * scale) {
    grad_r[c] = (a11 * b0 - a01 * b1) / det;
    grad_z[c] = (-a01 * b0 + a00 * b1) / det;
  } else {
    remap_dispatch_audit_count(
        audit,
        RemapDispatchAuditCounter::Ms2DegenerateGradientFallback,
        c);
    grad_r[c] = (fabs(a00) > 1.0e-300) ? (b0 / a00) : 0.0;
    grad_z[c] = (fabs(a11) > 1.0e-300) ? (b1 / a11) : 0.0;
  }
}

static __global__ __launch_bounds__(256, 4) void apply_slope_limiter_van_leer_kernel(
    double* __restrict__ grad_r,
    double* __restrict__ grad_z,
    const double* __restrict__ field,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz,
    const RemapDispatchAuditDeviceView audit,
    const std::uint8_t* __restrict__ geometry_policy_exempt = nullptr) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c] != 0) {
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const double psi = detail::ms2_van_leer_limiter_factor(
      field,
      x_r,
      x_z,
      grad_r[c],
      grad_z[c],
      i,
      j,
      nr,
      nz,
      geometry_policy_exempt);
  if (psi < 1.0) {
    remap_dispatch_audit_count(
        audit, RemapDispatchAuditCounter::LimiterActivation, c);
  }
  grad_r[c] *= psi;
  grad_z[c] *= psi;
}

static __global__ __launch_bounds__(256, 4)
void apply_slope_limiter_barth_jespersen_kernel(double* __restrict__ grad_r,
                                                double* __restrict__ grad_z,
                                                const double* __restrict__ field,
                                                const double* __restrict__ x_r,
                                                const double* __restrict__ x_z,
                                                const int nr,
                                                const int nz,
                                                const RemapDispatchAuditDeviceView audit,
                                                const std::uint8_t* __restrict__
                                                    geometry_policy_exempt = nullptr) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c] != 0) {
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const double psi = detail::ms2_barth_jespersen_limiter_factor(
      field,
      x_r,
      x_z,
      grad_r[c],
      grad_z[c],
      i,
      j,
      nr,
      nz,
      geometry_policy_exempt);
  if (psi < 1.0) {
    remap_dispatch_audit_count(
        audit, RemapDispatchAuditCounter::LimiterActivation, c);
  }
  grad_r[c] *= psi;
  grad_z[c] *= psi;
}

static __global__ __launch_bounds__(256, 4)
void csr_compute_lsq_gradients_kernel(double* __restrict__ grad_r,
                                      double* __restrict__ grad_z,
                                      const double* __restrict__ field,
                                      const double* __restrict__ x_r,
                                      const double* __restrict__ x_z,
                                      const int* __restrict__ face_adj_csr_offsets,
                                      const int* __restrict__ face_adj_csr_indices,
                                      const int* __restrict__ cell_node_csr_offsets,
                                      const int* __restrict__ cell_node_csr_indices,
                                      const std::uint8_t* __restrict__ cell_nverts,
                                      const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double q_c = field[c];
  const double r_c =
      detail::csr_cell_center_r(
          x_r, cell_node_csr_offsets, cell_node_csr_indices, c, cell_nverts);
  const double z_c =
      detail::csr_cell_center_z(
          x_z, cell_node_csr_offsets, cell_node_csr_indices, c, cell_nverts);

  double a00 = 0.0;
  double a01 = 0.0;
  double a11 = 0.0;
  double b0 = 0.0;
  double b1 = 0.0;
  int valid_neighbors = 0;

  const int off = face_adj_csr_offsets[c];
  const int end = face_adj_csr_offsets[c + 1];
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  for (int p = off; p < end; ++p) {
    const int local = p - off;
    if (!tenryu::mesh::mesh_topo_local_face_is_active(active_nverts, local)) {
      continue;
    }
    const int nb = face_adj_csr_indices[p];
    if (nb < 0 || nb >= n_cells) {
      continue;
    }
    const double dr =
        detail::csr_cell_center_r(
            x_r, cell_node_csr_offsets, cell_node_csr_indices, nb, cell_nverts) -
        r_c;
    const double dz =
        detail::csr_cell_center_z(
            x_z, cell_node_csr_offsets, cell_node_csr_indices, nb, cell_nverts) -
        z_c;
    const double dist2 = dr * dr + dz * dz;
    if (!(dist2 > 1.0e-300) || !isfinite(dist2)) {
      continue;
    }
    const double w = 1.0 / dist2;
    const double dq = field[nb] - q_c;
    a00 += w * dr * dr;
    a01 += w * dr * dz;
    a11 += w * dz * dz;
    b0 += w * dr * dq;
    b1 += w * dz * dq;
    ++valid_neighbors;
  }

  if (valid_neighbors < 2) {
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }

  const double det = a00 * a11 - a01 * a01;
  const double scale = fabs(a00 * a11) + fabs(a01 * a01) + 1.0e-300;
  if (fabs(det) > 1.0e-24 * scale && isfinite(det)) {
    grad_r[c] = (a11 * b0 - a01 * b1) / det;
    grad_z[c] = (-a01 * b0 + a00 * b1) / det;
  } else {
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
  }
}

static __global__ __launch_bounds__(256, 4)
void csr_apply_barth_jespersen_limiter_kernel(
    double* __restrict__ grad_r,
    double* __restrict__ grad_z,
    const double* __restrict__ field,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double q_c = field[c];
  double q_min = q_c;
  double q_max = q_c;
  const int adj_off = face_adj_csr_offsets[c];
  const int adj_end = face_adj_csr_offsets[c + 1];
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  for (int p = adj_off; p < adj_end; ++p) {
    const int local = p - adj_off;
    if (!tenryu::mesh::mesh_topo_local_face_is_active(active_nverts, local)) {
      continue;
    }
    const int nb = face_adj_csr_indices[p];
    if (nb < 0 || nb >= n_cells) {
      continue;
    }
    const double q = field[nb];
    q_min = fmin(q_min, q);
    q_max = fmax(q_max, q);
  }

  if (!(q_max > q_min) || !isfinite(grad_r[c]) || !isfinite(grad_z[c])) {
    grad_r[c] = 0.0;
    grad_z[c] = 0.0;
    return;
  }

  const double r_c =
      detail::csr_cell_center_r(
          x_r, cell_node_csr_offsets, cell_node_csr_indices, c, cell_nverts);
  const double z_c =
      detail::csr_cell_center_z(
          x_z, cell_node_csr_offsets, cell_node_csr_indices, c, cell_nverts);
  double alpha = 1.0;
  for (int local = 0; local < 4; ++local) {
    double r_f = 0.0;
    double z_f = 0.0;
    if (!detail::csr_face_center(x_r,
                                 x_z,
                                 cell_node_csr_offsets,
                                 cell_node_csr_indices,
                                 c,
                                 local,
                                 r_f,
                                 z_f,
                                 cell_nverts)) {
      continue;
    }
    const double q_face =
        q_c + grad_r[c] * (r_f - r_c) + grad_z[c] * (z_f - z_c);
    alpha = fmin(alpha,
                 detail::csr_barth_jespersen_limiter_ratio(
                     q_c, q_min, q_max, q_face));
  }

  if (!isfinite(alpha)) {
    alpha = 0.0;
  }
  alpha = fmin(1.0, fmax(0.0, alpha));
  grad_r[c] *= alpha;
  grad_z[c] *= alpha;
}

template <bool FixedSign>
static __global__ __launch_bounds__(256, 4) void conservative_remap_ms2_kernel_t(
    double* __restrict__ field_new,
    const double* __restrict__ field_old,
    const double* __restrict__ grad_r,
    const double* __restrict__ grad_z,
    const double* __restrict__ vol_old,
    const double* __restrict__ vol_new,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const int sweep_direction,
    const int nr,
    const int nz,
    const std::uint8_t* __restrict__ geometry_policy_exempt,
    const RemapDispatchAuditDeviceView audit) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  if (geometry_policy_exempt != nullptr && geometry_policy_exempt[c] != 0) {
    field_new[c] = field_old[c];
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;

  const double qbar_old = field_old[c] * vol_old[c];
  double F_plus = 0.0;
  double F_minus = 0.0;
  if (sweep_direction == 0) {
    F_plus = detail::ms2_flux_r_face_t<FixedSign>(field_old,
                                                  grad_r,
                                                  grad_z,
                                                  x_r_old,
                                                  x_z_old,
                                                  x_r_new,
                                                  x_z_new,
                                                  i + 1,
                                                  j,
                                                  nr,
                                                  nz,
                                                  audit,
                                                  geometry_policy_exempt);
    F_minus = detail::ms2_flux_r_face_t<FixedSign>(field_old,
                                                   grad_r,
                                                   grad_z,
                                                   x_r_old,
                                                   x_z_old,
                                                   x_r_new,
                                                   x_z_new,
                                                   i,
                                                   j,
                                                   nr,
                                                   nz,
                                                   audit,
                                                   geometry_policy_exempt);
  } else {
    F_plus = detail::ms2_flux_z_face_t<FixedSign>(field_old,
                                                  grad_r,
                                                  grad_z,
                                                  x_r_old,
                                                  x_z_old,
                                                  x_r_new,
                                                  x_z_new,
                                                  i,
                                                  j + 1,
                                                  nr,
                                                  nz,
                                                  audit,
                                                  geometry_policy_exempt);
    F_minus = detail::ms2_flux_z_face_t<FixedSign>(field_old,
                                                   grad_r,
                                                   grad_z,
                                                   x_r_old,
                                                   x_z_old,
                                                   x_r_new,
                                                   x_z_new,
                                                   i,
                                                   j,
                                                   nr,
                                                   nz,
                                                   audit,
                                                   geometry_policy_exempt);
  }

  const double qbar_new = qbar_old - F_plus + F_minus;
  field_new[c] = qbar_new / fmax(vol_new[c], 1.0e-30);
}

inline void launch_conservative_remap_ms2(double* d_field_new,
                                          const double* d_field_old,
                                          double* d_grad_r,
                                          double* d_grad_z,
                                          const double* d_vol_old,
                                          const double* d_vol_new,
                                          const double* d_xr_old,
                                          const double* d_xz_old,
                                          const double* d_xr_new,
                                          const double* d_xz_new,
                                          const int sweep_direction,
                                          const int nr,
                                          const int nz,
                                          const RemapMs2Limiter limiter,
                                          const bool swept_volume_sign_fixed = false,
                                          const RemapDispatchAuditDeviceView audit = {},
                                          cudaStream_t stream = nullptr,
                                          const std::uint8_t* d_geometry_policy_exempt = nullptr) {
  const int n_cells = nr * nz;
  const int blocks = (n_cells + 255) / 256;
  compute_cell_slopes_lstsq_2d_rz_kernel<<<blocks, 256, 0, stream>>>(
      d_grad_r,
      d_grad_z,
      d_field_old,
      d_xr_old,
      d_xz_old,
      nr,
      nz,
      audit,
      d_geometry_policy_exempt);
  CUDA_CHECK(cudaGetLastError());
  if (limiter == RemapMs2Limiter::BarthJespersen) {
    apply_slope_limiter_barth_jespersen_kernel<<<blocks, 256, 0, stream>>>(
        d_grad_r,
        d_grad_z,
        d_field_old,
        d_xr_old,
        d_xz_old,
        nr,
        nz,
        audit,
        d_geometry_policy_exempt);
  } else {
    apply_slope_limiter_van_leer_kernel<<<blocks, 256, 0, stream>>>(
        d_grad_r,
        d_grad_z,
        d_field_old,
        d_xr_old,
        d_xz_old,
        nr,
        nz,
        audit,
        d_geometry_policy_exempt);
  }
  CUDA_CHECK(cudaGetLastError());
  if (swept_volume_sign_fixed) {
    conservative_remap_ms2_kernel_t<true><<<blocks, 256, 0, stream>>>(d_field_new,
                                                                       d_field_old,
                                                                       d_grad_r,
                                                                       d_grad_z,
                                                                       d_vol_old,
                                                                       d_vol_new,
                                                                       d_xr_old,
                                                                       d_xz_old,
                                                                       d_xr_new,
                                                                       d_xz_new,
                                                                       sweep_direction,
                                                                       nr,
                                                                       nz,
                                                                       d_geometry_policy_exempt,
                                                                       audit);
  } else {
    conservative_remap_ms2_kernel_t<false><<<blocks, 256, 0, stream>>>(d_field_new,
                                                                        d_field_old,
                                                                        d_grad_r,
                                                                        d_grad_z,
                                                                        d_vol_old,
                                                                        d_vol_new,
                                                                        d_xr_old,
                                                                        d_xz_old,
                                                                        d_xr_new,
                                                                        d_xz_new,
                                                                        sweep_direction,
                                                                        nr,
                                                                        nz,
                                                                        d_geometry_policy_exempt,
                                                                        audit);
  }
}

inline bool launch_remap_strang_ms2(double* d_field_inout,
                                    double* d_field_tmp,
                                    double* d_vol_mid,
                                    const double* d_vol_old,
                                    const double* d_vol_new,
                                    const double* d_xr_old,
                                    const double* d_xz_old,
                                    const double* d_xr_new,
                                    const double* d_xz_new,
                                    const int nr,
                                    const int nz,
                                    const int step_number,
                                    const RemapMs2Limiter limiter,
                                    const bool swept_volume_sign_fixed = false,
                                    const std::uint8_t* d_geometry_policy_exempt = nullptr,
                                    const RemapDispatchAuditDeviceView audit = {},
                                    cudaStream_t stream = nullptr) {
  const int first_dir = (step_number % 2 == 0) ? 0 : 1;
  const int second_dir = 1 - first_dir;
  const int n_cells = nr * nz;
  const int blocks = (n_cells + 255) / 256;
  int* d_nonpositive_count = nullptr;
  int* d_first_nonpositive_cell = nullptr;
  double* d_grad_r = nullptr;
  double* d_grad_z = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_nonpositive_count), sizeof(int)));
  CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void**>(&d_first_nonpositive_cell), sizeof(int)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_grad_r),
                        static_cast<std::size_t>(n_cells) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_grad_z),
                        static_cast<std::size_t>(n_cells) * sizeof(double)));
  CUDA_CHECK(cudaMemsetAsync(d_nonpositive_count, 0, sizeof(int), stream));
  int first_nonpositive_cell = INT_MAX;
  CUDA_CHECK(cudaMemcpyAsync(d_first_nonpositive_cell,
                             &first_nonpositive_cell,
                             sizeof(int),
                             cudaMemcpyHostToDevice,
                             stream));

  if (swept_volume_sign_fixed) {
    compute_intermediate_volume_kernel_t<true><<<blocks, 256, 0, stream>>>(d_vol_mid,
                                                                            d_nonpositive_count,
                                                                            d_geometry_policy_exempt,
                                                                            d_first_nonpositive_cell,
                                                                            d_vol_old,
                                                                            d_xr_old,
                                                                            d_xz_old,
                                                                            d_xr_new,
                                                                            d_xz_new,
                                                                            first_dir,
                                                                            nr,
                                                                            nz);
  } else {
    compute_intermediate_volume_kernel_t<false><<<blocks, 256, 0, stream>>>(d_vol_mid,
                                                                             d_nonpositive_count,
                                                                             d_geometry_policy_exempt,
                                                                             d_first_nonpositive_cell,
                                                                             d_vol_old,
                                                                             d_xr_old,
                                                                             d_xz_old,
                                                                             d_xr_new,
                                                                             d_xz_new,
                                                                             first_dir,
                                                                             nr,
                                                                             nz);
  }
  CUDA_CHECK(cudaGetLastError());

  int nonpositive_count = 0;
  if (stream == nullptr) {
    CUDA_CHECK(cudaMemcpy(&nonpositive_count,
                          d_nonpositive_count,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
  } else {
    CUDA_CHECK(cudaMemcpyAsync(&nonpositive_count,
                               d_nonpositive_count,
                               sizeof(int),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  if (nonpositive_count > 0) {
    if (stream == nullptr) {
      CUDA_CHECK(cudaMemcpy(&first_nonpositive_cell,
                            d_first_nonpositive_cell,
                            sizeof(int),
                            cudaMemcpyDeviceToHost));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(&first_nonpositive_cell,
                                 d_first_nonpositive_cell,
                                 sizeof(int),
                                 cudaMemcpyDeviceToHost,
                                 stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
  }
  CUDA_CHECK(cudaFree(d_first_nonpositive_cell));
  CUDA_CHECK(cudaFree(d_nonpositive_count));
  if (nonpositive_count > 0) {
    CUDA_CHECK(cudaFree(d_grad_z));
    CUDA_CHECK(cudaFree(d_grad_r));
    core::log_warning(
        "[ale-remap] transport aborted: non-positive intermediate volume at cell=" +
        std::to_string(first_nonpositive_cell) +
        " (i=" + std::to_string(first_nonpositive_cell / nz) +
        ", j=" + std::to_string(first_nonpositive_cell % nz) +
        ") count=" + std::to_string(nonpositive_count));
    return false;
  }

  launch_conservative_remap_ms2(d_field_tmp,
                                d_field_inout,
                                d_grad_r,
                                d_grad_z,
                                d_vol_old,
                                d_vol_mid,
                                d_xr_old,
                                d_xz_old,
                                d_xr_new,
                                d_xz_new,
                                first_dir,
                                nr,
                                nz,
                                limiter,
                                swept_volume_sign_fixed,
                                audit,
                                stream,
                                d_geometry_policy_exempt);
  CUDA_CHECK(cudaGetLastError());
  if (stream == nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  launch_conservative_remap_ms2(d_field_inout,
                                d_field_tmp,
                                d_grad_r,
                                d_grad_z,
                                d_vol_mid,
                                d_vol_new,
                                d_xr_old,
                                d_xz_old,
                                d_xr_new,
                                d_xz_new,
                                second_dir,
                                nr,
                                nz,
                                limiter,
                                swept_volume_sign_fixed,
                                audit,
                                stream,
                                d_geometry_policy_exempt);
  CUDA_CHECK(cudaGetLastError());
  if (stream == nullptr) {
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  CUDA_CHECK(cudaFree(d_grad_z));
  CUDA_CHECK(cudaFree(d_grad_r));
  return true;
}

}  // namespace tenryu::hydro::ale
