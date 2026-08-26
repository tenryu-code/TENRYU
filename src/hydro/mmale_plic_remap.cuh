#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "hydro/ale_remap.cuh"
#include "hydro/plic_geometry.cuh"
#include "hydro/rz_corner_mass.cuh"

namespace tenryu::hydro::mmale {

/*
MMALE PLIC-in-CSR remap MVP design note
---------------------------------------
Scope: standalone explicit-parameter primitives for a 2-material ideal-gas
ALE remap in TENRYU 2D RZ.  Nothing in this header is wired by default.

Geometry and units:
  * Coordinates are cgs (r,z in cm).  Swept volumes are exact RZ volumes
    dV = 2*pi*int_S r dA [cm^3], evaluated by rz_polygon_volume_exact().
  * A PLIC interface stores material `material_below` in the half-plane
    n_r*r + n_z*z <= alpha.  The other MVP material receives the complement.
  * CSR integration should pass the same signed swept volume used by
    csr_face_swept_volume_outward().  The clipped polygon is used for material
    fractions; dV_1 is computed as dV_total - dV_0 so the per-face split closes
    to round-off under the existing CSR debit/credit sign convention.

Conservative update sketch for integration:
  1. For mixed donor cells, call reconstruct_plic_plane_from_stencil_2mat()
     on an explicit local volume-fraction stencil and store n/alpha.  The
     reconstruction mirrors the existing Youngs-seeded fixed-angle LVIRA path
     in plic_normal.cu, but keeps Config/State out of this module.
  2. In the CSR face loop, build/pass the swept polygon S_f and signed dV_to_cell.
     call split_swept_volume_by_plic_2mat(S_f, plane, dV_to_cell, donor_f0,...).
  3. Convert the split volumes to mass/Ee/Ei fluxes with
     compute_face_flux_2mat(), then accumulate them to cell-major material
     extents with accumulate_face_flux_2mat().
  4. After all face fluxes, call pressure_equilibrate_ideal_gas_2mat() per cell
     to recover rho_k, e_k, and volfrac_k with volfrac_0 + volfrac_1 == 1.

MVP exclusions from the grounded spec remain explicit integration gates:
  * total_energy_remap_2d_rz and compatible corner/subzonal mass need separate
    per-material paths before being enabled with this header;
  * staggered/nodal momentum reconstruction is a follow-on.  This header only
    supplies a conservative cell-momentum hook using the same material mass flux.
*/

constexpr int kMmaleMaterials = 2;
constexpr double kTiny = 1.0e-300;
constexpr double kGeomTiny = 1.0e-30;

struct PlicPlane2Mat {
  double n_r = 0.0;
  double n_z = 0.0;
  double alpha = 0.0;
  double volume_fraction = 0.0;
  double residual_l2 = 0.0;
  double grad_F_magnitude = 0.0;
  int material_below = 0;
  bool mixed = false;
  bool valid = false;
  bool ill_conditioned = true;
};

struct PlicSplit2Mat {
  double dV[kMmaleMaterials] = {0.0, 0.0};
  double fraction[kMmaleMaterials] = {0.0, 0.0};
  double residual_abs = 0.0;
  bool used_plic = false;
};

struct MaterialFaceFlux2Mat {
  double mass[kMmaleMaterials] = {0.0, 0.0};
  double Ee[kMmaleMaterials] = {0.0, 0.0};
  double Ei[kMmaleMaterials] = {0.0, 0.0};
  double mom_r[kMmaleMaterials] = {0.0, 0.0};
  double mom_z[kMmaleMaterials] = {0.0, 0.0};
};

struct IdealGasClosure2Mat {
  double volfrac[kMmaleMaterials] = {1.0, 0.0};
  double rho[kMmaleMaterials] = {0.0, 0.0};
  double ee[kMmaleMaterials] = {0.0, 0.0};
  double ei[kMmaleMaterials] = {0.0, 0.0};
  double pressure = 0.0;
  bool pressure_equilibrated = false;
  bool used_mass_fraction_fallback = false;
};

__host__ __device__ inline bool finite_value(const double x) {
  return x == x && x != INFINITY && x != -INFINITY;
}

__host__ __device__ inline double clamp01(const double x) {
  return fmin(1.0, fmax(0.0, x));
}

__host__ __device__ inline int stencil_index(const int i,
                                             const int j,
                                             const int stencil_size) {
  return i * stencil_size + j;
}

__host__ __device__ inline bool normalize(double& r, double& z) {
  const double norm = sqrt(r * r + z * z);
  if (!(norm > kGeomTiny) || !finite_value(norm)) {
    r = 0.0;
    z = 0.0;
    return false;
  }
  r /= norm;
  z /= norm;
  return true;
}

__host__ __device__ inline tenryu::hydro::plic::Polygon make_polygon(
    const double* r,
    const double* z,
    const int nverts) {
  tenryu::hydro::plic::Polygon p{};
  p.n = 0;
  const int n = (nverts <= tenryu::hydro::plic::kPlicMaxPolygonVertices)
                    ? nverts
                    : tenryu::hydro::plic::kPlicMaxPolygonVertices;
  for (int k = 0; k < n; ++k) {
    tenryu::hydro::plic::append_vertex(p, r[k], z[k]);
  }
  return p;
}

__host__ __device__ inline double rz_volume(
    const tenryu::hydro::plic::Polygon& p) {
  if (p.n < 3) {
    return 0.0;
  }
  return tenryu::hydro::rz::rz_polygon_volume_exact(p.r, p.z, p.n);
}

__host__ __device__ inline double positive_rz_volume(
    const tenryu::hydro::plic::Polygon& p) {
  return fabs(rz_volume(p));
}

__host__ __device__ inline double polygon_meridional_area(
    const tenryu::hydro::plic::Polygon& p) {
  if (p.n < 3) {
    return 0.0;
  }
  double cross = 0.0;
  for (int k = 0; k < p.n; ++k) {
    const int kp = (k + 1 == p.n) ? 0 : (k + 1);
    cross += p.r[k] * p.z[kp] - p.r[kp] * p.z[k];
  }
  return 0.5 * fabs(cross);
}

__host__ __device__ inline void polygon_bounds(
    const tenryu::hydro::plic::Polygon& p,
    double& r_min,
    double& r_max,
    double& z_min,
    double& z_max) {
  if (p.n <= 0) {
    r_min = r_max = z_min = z_max = 0.0;
    return;
  }
  r_min = r_max = p.r[0];
  z_min = z_max = p.z[0];
  for (int k = 1; k < p.n; ++k) {
    r_min = fmin(r_min, p.r[k]);
    r_max = fmax(r_max, p.r[k]);
    z_min = fmin(z_min, p.z[k]);
    z_max = fmax(z_max, p.z[k]);
  }
}

struct AlphaSolveResult {
  double alpha = 0.0;
  double residual_rel = 1.0;
  int iterations = 0;
  bool converged = false;
};

__host__ __device__ inline AlphaSolveResult find_alpha_for_volume_fraction(
    const tenryu::hydro::plic::Polygon& cell,
    const double n_r,
    const double n_z,
    const double target_volfrac,
    const int max_iter,
    const double tol_rel) {
  AlphaSolveResult out{};
  const double v_cell = positive_rz_volume(cell);
  const double f_target = clamp01(target_volfrac);
  if (!(cell.n >= 3) || !(v_cell > kTiny) || f_target <= 0.0 ||
      f_target >= 1.0) {
    out.alpha = 0.0;
    out.residual_rel = 0.0;
    out.converged = (f_target <= 0.0 || f_target >= 1.0);
    return out;
  }

  double a_lo = n_r * cell.r[0] + n_z * cell.z[0];
  double a_hi = a_lo;
  for (int k = 1; k < cell.n; ++k) {
    const double a = n_r * cell.r[k] + n_z * cell.z[k];
    a_lo = fmin(a_lo, a);
    a_hi = fmax(a_hi, a);
  }
  if (!(a_hi > a_lo)) {
    out.alpha = a_lo;
    return out;
  }

  const double target_v = f_target * v_cell;
  double last_residual = 1.0;
  for (int it = 0; it < max_iter; ++it) {
    const double a_mid = 0.5 * (a_lo + a_hi);
    tenryu::hydro::plic::Polygon clipped{};
    tenryu::hydro::plic::clip_polygon_by_halfplane(
        cell, n_r, n_z, a_mid, clipped);
    const double v_mid = positive_rz_volume(clipped);
    if (v_mid < target_v) {
      a_lo = a_mid;
    } else {
      a_hi = a_mid;
    }
    last_residual = fabs(v_mid - target_v) / fmax(target_v, kGeomTiny);
    out.iterations = it + 1;
    if (last_residual < tol_rel) {
      out.alpha = a_mid;
      out.residual_rel = last_residual;
      out.converged = true;
      return out;
    }
  }

  out.alpha = 0.5 * (a_lo + a_hi);
  tenryu::hydro::plic::Polygon clipped{};
  tenryu::hydro::plic::clip_polygon_by_halfplane(
      cell, n_r, n_z, out.alpha, clipped);
  const double v = positive_rz_volume(clipped);
  out.residual_rel = fabs(v - target_v) / fmax(target_v, kGeomTiny);
  out.converged = out.residual_rel < tol_rel || last_residual < tol_rel;
  return out;
}

__host__ __device__ inline double one_sided_or_centered_slope(
    const double f_m,
    const double f_0,
    const double f_p,
    const double x_m,
    const double x_0,
    const double x_p,
    const bool has_m,
    const bool has_p) {
  if (has_m && has_p && fabs(x_p - x_m) > kGeomTiny) {
    return (f_p - f_m) / (x_p - x_m);
  }
  if (has_p && fabs(x_p - x_0) > kGeomTiny) {
    return (f_p - f_0) / (x_p - x_0);
  }
  if (has_m && fabs(x_0 - x_m) > kGeomTiny) {
    return (f_0 - f_m) / (x_0 - x_m);
  }
  return 0.0;
}

struct YoungsSeed {
  double n_r = 0.0;
  double n_z = 0.0;
  double grad_norm = 0.0;
  bool valid = false;
  bool ill_conditioned = true;
};

__host__ __device__ inline YoungsSeed compute_youngs_seed(
    const double* volfrac_stencil,
    const double* r_centroid_stencil,
    const double* z_centroid_stencil,
    const int target_i,
    const int target_j,
    const int stencil_size,
    const double h_eff) {
  YoungsSeed seed{};
  if (volfrac_stencil == nullptr || r_centroid_stencil == nullptr ||
      z_centroid_stencil == nullptr || stencil_size <= 0 ||
      target_i < 0 || target_i >= stencil_size ||
      target_j < 0 || target_j >= stencil_size) {
    return seed;
  }

  const int c = stencil_index(target_i, target_j, stencil_size);
  const bool has_im = target_i > 0;
  const bool has_ip = target_i + 1 < stencil_size;
  const bool has_jm = target_j > 0;
  const bool has_jp = target_j + 1 < stencil_size;

  const int im = stencil_index(has_im ? target_i - 1 : target_i,
                               target_j,
                               stencil_size);
  const int ip = stencil_index(has_ip ? target_i + 1 : target_i,
                               target_j,
                               stencil_size);
  const int jm = stencil_index(target_i,
                               has_jm ? target_j - 1 : target_j,
                               stencil_size);
  const int jp = stencil_index(target_i,
                               has_jp ? target_j + 1 : target_j,
                               stencil_size);
  const double grad_r = one_sided_or_centered_slope(
      volfrac_stencil[im],
      volfrac_stencil[c],
      volfrac_stencil[ip],
      r_centroid_stencil[im],
      r_centroid_stencil[c],
      r_centroid_stencil[ip],
      has_im,
      has_ip);
  const double grad_z = one_sided_or_centered_slope(
      volfrac_stencil[jm],
      volfrac_stencil[c],
      volfrac_stencil[jp],
      z_centroid_stencil[jm],
      z_centroid_stencil[c],
      z_centroid_stencil[jp],
      has_jm,
      has_jp);

  seed.grad_norm = sqrt(grad_r * grad_r + grad_z * grad_z);
  const double scaled = h_eff * seed.grad_norm;
  const double stencil_n = static_cast<double>(stencil_size * stencil_size);
  const double threshold = fmax(1.0e-8, 1.0e-9 * sqrt(stencil_n));
  seed.ill_conditioned = scaled < threshold;
  if (seed.grad_norm > kGeomTiny) {
    seed.n_r = -grad_r / seed.grad_norm;
    seed.n_z = -grad_z / seed.grad_norm;
    seed.valid = true;
  }
  return seed;
}

struct CandidateScore {
  double n_r = 0.0;
  double n_z = 0.0;
  double alpha = 0.0;
  double residual_l2 = INFINITY;
  bool valid = false;
};

__host__ __device__ inline void neighbor_rect(
    const double rc,
    const double zc,
    const double dr,
    const double dz,
    tenryu::hydro::plic::Polygon& p) {
  p.n = 0;
  tenryu::hydro::plic::append_vertex(p, rc - 0.5 * dr, zc - 0.5 * dz);
  tenryu::hydro::plic::append_vertex(p, rc + 0.5 * dr, zc - 0.5 * dz);
  tenryu::hydro::plic::append_vertex(p, rc + 0.5 * dr, zc + 0.5 * dz);
  tenryu::hydro::plic::append_vertex(p, rc - 0.5 * dr, zc + 0.5 * dz);
}

__host__ __device__ inline CandidateScore score_plic_normal_candidate(
    const tenryu::hydro::plic::Polygon& cell,
    const double* volfrac_stencil,
    const double* r_centroid_stencil,
    const double* z_centroid_stencil,
    const double* volume_stencil,
    const int target_i,
    const int target_j,
    const int stencil_size,
    const int alpha_max_iter,
    const double alpha_tol_rel,
    double n_r,
    double n_z) {
  CandidateScore score{};
  if (!normalize(n_r, n_z) || volfrac_stencil == nullptr ||
      r_centroid_stencil == nullptr || z_centroid_stencil == nullptr) {
    return score;
  }
  const int c = stencil_index(target_i, target_j, stencil_size);
  const double f_target = clamp01(volfrac_stencil[c]);
  if (!(cell.n >= 3) || f_target <= 0.0 || f_target >= 1.0) {
    return score;
  }

  const AlphaSolveResult alpha = find_alpha_for_volume_fraction(
      cell, n_r, n_z, f_target, alpha_max_iter, alpha_tol_rel);
  if (!alpha.converged && !(alpha.residual_rel < 1.0e-10)) {
    return score;
  }

  double r_min = 0.0;
  double r_max = 0.0;
  double z_min = 0.0;
  double z_max = 0.0;
  polygon_bounds(cell, r_min, r_max, z_min, z_max);
  const double dr = fmax(r_max - r_min, kGeomTiny);
  const double dz = fmax(z_max - z_min, kGeomTiny);

  double err = 0.0;
  double wsum = 0.0;
  for (int i = 0; i < stencil_size; ++i) {
    for (int j = 0; j < stencil_size; ++j) {
      const int idx = stencil_index(i, j, stencil_size);
      tenryu::hydro::plic::Polygon nb{};
      neighbor_rect(r_centroid_stencil[idx], z_centroid_stencil[idx], dr, dz, nb);
      double v_neighbor =
          (volume_stencil != nullptr) ? fabs(volume_stencil[idx]) : 0.0;
      if (!(v_neighbor > kTiny)) {
        v_neighbor = positive_rz_volume(nb);
      }
      if (!(v_neighbor > kTiny)) {
        continue;
      }
      tenryu::hydro::plic::Polygon clipped{};
      tenryu::hydro::plic::clip_polygon_by_halfplane(
          nb, n_r, n_z, alpha.alpha, clipped);
      const double f_pred = clamp01(positive_rz_volume(clipped) / v_neighbor);
      const double diff = f_pred - clamp01(volfrac_stencil[idx]);
      const double weight =
          2.0 * tenryu::hydro::ale::detail::kPi *
          fmax(r_centroid_stencil[idx], 0.0);
      err += weight * diff * diff;
      wsum += weight;
    }
  }

  score.n_r = n_r;
  score.n_z = n_z;
  score.alpha = alpha.alpha;
  score.residual_l2 = (wsum > 0.0) ? sqrt(err / wsum) : sqrt(err);
  score.valid = finite_value(score.residual_l2);
  return score;
}

__host__ __device__ inline void fixed_lvira_angle(
    const int candidate,
    double& n_r,
    double& n_z) {
  switch (candidate) {
    case 0:
      n_r = 1.0;
      n_z = 0.0;
      break;
    case 1:
      n_r = 0.86602540378443864676;
      n_z = 0.5;
      break;
    case 2:
      n_r = 0.70710678118654752440;
      n_z = 0.70710678118654752440;
      break;
    case 3:
      n_r = 0.5;
      n_z = 0.86602540378443864676;
      break;
    case 4:
      n_r = 0.0;
      n_z = 1.0;
      break;
    case 5:
      n_r = -0.5;
      n_z = 0.86602540378443864676;
      break;
    case 6:
      n_r = -0.70710678118654752440;
      n_z = 0.70710678118654752440;
      break;
    case 7:
      n_r = -0.86602540378443864676;
      n_z = 0.5;
      break;
    case 8:
      n_r = -1.0;
      n_z = 0.0;
      break;
    case 9:
      n_r = -0.86602540378443864676;
      n_z = -0.5;
      break;
    case 10:
      n_r = -0.70710678118654752440;
      n_z = -0.70710678118654752440;
      break;
    case 11:
      n_r = -0.5;
      n_z = -0.86602540378443864676;
      break;
    case 12:
      n_r = 0.0;
      n_z = -1.0;
      break;
    case 13:
      n_r = 0.5;
      n_z = -0.86602540378443864676;
      break;
    case 14:
      n_r = 0.70710678118654752440;
      n_z = -0.70710678118654752440;
      break;
    default:
      n_r = 0.86602540378443864676;
      n_z = -0.5;
      break;
  }
}

__host__ __device__ inline PlicPlane2Mat reconstruct_plic_plane_from_stencil_2mat(
    const double* cell_r,
    const double* cell_z,
    const int cell_nverts,
    const double* volfrac_stencil,
    const double* r_centroid_stencil,
    const double* z_centroid_stencil,
    const double* volume_stencil,
    const int target_i,
    const int target_j,
    const int stencil_size,
    const int material_below,
    const double mixed_threshold_min,
    const double mixed_threshold_max,
    const int alpha_max_iter,
    const double alpha_tol_rel,
    const bool use_lvira_candidates) {
  PlicPlane2Mat out{};
  out.material_below = (material_below == 1) ? 1 : 0;
  if (cell_r == nullptr || cell_z == nullptr || volfrac_stencil == nullptr ||
      r_centroid_stencil == nullptr || z_centroid_stencil == nullptr ||
      cell_nverts < 3 || stencil_size <= 0 || target_i < 0 ||
      target_i >= stencil_size || target_j < 0 || target_j >= stencil_size) {
    return out;
  }

  const int c = stencil_index(target_i, target_j, stencil_size);
  const double f = clamp01(volfrac_stencil[c]);
  out.volume_fraction = f;
  out.mixed = f >= mixed_threshold_min && f <= mixed_threshold_max;
  if (!out.mixed) {
    out.ill_conditioned = false;
    return out;
  }

  const tenryu::hydro::plic::Polygon cell = make_polygon(cell_r, cell_z, cell_nverts);
  const double area = polygon_meridional_area(cell);
  const YoungsSeed seed = compute_youngs_seed(volfrac_stencil,
                                              r_centroid_stencil,
                                              z_centroid_stencil,
                                              target_i,
                                              target_j,
                                              stencil_size,
                                              sqrt(fmax(area, 0.0)));
  out.grad_F_magnitude = seed.grad_norm;
  CandidateScore best{};

  if (seed.valid) {
    best = score_plic_normal_candidate(cell,
                                       volfrac_stencil,
                                       r_centroid_stencil,
                                       z_centroid_stencil,
                                       volume_stencil,
                                       target_i,
                                       target_j,
                                       stencil_size,
                                       alpha_max_iter,
                                       alpha_tol_rel,
                                       seed.n_r,
                                       seed.n_z);
  }

  if (use_lvira_candidates) {
    for (int a = 0; a < 16; ++a) {
      double n_r = 0.0;
      double n_z = 0.0;
      fixed_lvira_angle(a, n_r, n_z);
      const CandidateScore cand = score_plic_normal_candidate(cell,
                                                              volfrac_stencil,
                                                              r_centroid_stencil,
                                                              z_centroid_stencil,
                                                              volume_stencil,
                                                              target_i,
                                                              target_j,
                                                              stencil_size,
                                                              alpha_max_iter,
                                                              alpha_tol_rel,
                                                              n_r,
                                                              n_z);
      if (cand.valid && (!best.valid || cand.residual_l2 < best.residual_l2)) {
        best = cand;
      }
    }
  }

  if (best.valid) {
    out.n_r = best.n_r;
    out.n_z = best.n_z;
    out.alpha = best.alpha;
    out.residual_l2 = best.residual_l2;
    out.valid = true;
    out.ill_conditioned = seed.ill_conditioned;
  }
  return out;
}

__host__ __device__ inline PlicSplit2Mat split_swept_volume_by_plic_2mat(
    const tenryu::hydro::plic::Polygon& swept_polygon,
    const PlicPlane2Mat& plane,
    const double signed_swept_volume_to_cell,
    const double donor_volfrac_material0) {
  PlicSplit2Mat out{};
  if (!finite_value(signed_swept_volume_to_cell) ||
      signed_swept_volume_to_cell == 0.0) {
    return out;
  }

  double f0 = clamp01(donor_volfrac_material0);
  if (plane.valid && plane.mixed && swept_polygon.n >= 3) {
    const double v_poly = rz_volume(swept_polygon);
    if (finite_value(v_poly) && fabs(v_poly) > kTiny) {
      tenryu::hydro::plic::Polygon clipped{};
      tenryu::hydro::plic::clip_polygon_by_halfplane(
          swept_polygon, plane.n_r, plane.n_z, plane.alpha, clipped);
      const double v_clip = rz_volume(clipped);
      double f_below = clamp01(v_clip / v_poly);
      if (!finite_value(f_below)) {
        f_below = clamp01(plane.volume_fraction);
      }
      if (plane.material_below == 0) {
        f0 = f_below;
      } else {
        f0 = 1.0 - f_below;
      }
      out.used_plic = true;
    }
  }

  out.fraction[0] = clamp01(f0);
  out.fraction[1] = 1.0 - out.fraction[0];
  out.dV[0] = signed_swept_volume_to_cell * out.fraction[0];
  out.dV[1] = signed_swept_volume_to_cell - out.dV[0];
  out.residual_abs = fabs((out.dV[0] + out.dV[1]) - signed_swept_volume_to_cell);
  return out;
}

__host__ __device__ inline bool csr_swept_polygon_outward(
    const double* x_r_old,
    const double* x_z_old,
    const double* x_r_new,
    const double* x_z_new,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const int cell,
    const int local_face,
    tenryu::hydro::plic::Polygon& swept_polygon) {
  swept_polygon.n = 0;
  int na = -1;
  int nb = -1;
  if (!tenryu::hydro::ale::detail::csr_face_swept_node_indices(
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_nverts,
          cell,
          local_face,
          &na,
          &nb)) {
    return false;
  }
  tenryu::hydro::plic::append_vertex(swept_polygon, x_r_old[na], x_z_old[na]);
  tenryu::hydro::plic::append_vertex(swept_polygon, x_r_old[nb], x_z_old[nb]);
  tenryu::hydro::plic::append_vertex(swept_polygon, x_r_new[nb], x_z_new[nb]);
  tenryu::hydro::plic::append_vertex(swept_polygon, x_r_new[na], x_z_new[na]);
  return true;
}

__host__ __device__ inline double csr_signed_swept_volume_outward(
    const double* x_r_old,
    const double* x_z_old,
    const double* x_r_new,
    const double* x_z_new,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const int* cell_orientation_sign,
    const std::uint8_t* cell_nverts,
    const int cell,
    const int local_face) {
  return tenryu::hydro::ale::detail::csr_face_swept_volume_outward(
      x_r_old,
      x_z_old,
      x_r_new,
      x_z_new,
      cell_node_csr_offsets,
      cell_node_csr_indices,
      cell_orientation_sign,
      cell,
      local_face,
      cell_nverts);
}

__host__ __device__ inline void material_primitives_from_extents_2mat(
    const double cell_volume,
    const double mass0,
    const double mass1,
    const double Ee0,
    const double Ee1,
    const double Ei0,
    const double Ei1,
    const double volfrac0,
    double rho[kMmaleMaterials],
    double ee[kMmaleMaterials],
    double ei[kMmaleMaterials]) {
  const double vf0 = clamp01(volfrac0);
  const double vf1 = 1.0 - vf0;
  const double V = fmax(cell_volume, kTiny);
  const double V0 = fmax(vf0 * V, kTiny);
  const double V1 = fmax(vf1 * V, kTiny);
  rho[0] = (mass0 > 0.0 && finite_value(mass0)) ? mass0 / V0 : 0.0;
  rho[1] = (mass1 > 0.0 && finite_value(mass1)) ? mass1 / V1 : 0.0;
  ee[0] = (mass0 > 0.0 && finite_value(Ee0)) ? fmax(Ee0, 0.0) / mass0 : 0.0;
  ee[1] = (mass1 > 0.0 && finite_value(Ee1)) ? fmax(Ee1, 0.0) / mass1 : 0.0;
  ei[0] = (mass0 > 0.0 && finite_value(Ei0)) ? fmax(Ei0, 0.0) / mass0 : 0.0;
  ei[1] = (mass1 > 0.0 && finite_value(Ei1)) ? fmax(Ei1, 0.0) / mass1 : 0.0;
}

__host__ __device__ inline MaterialFaceFlux2Mat compute_face_flux_2mat(
    const double dV_material[kMmaleMaterials],
    const double donor_rho_material[kMmaleMaterials],
    const double donor_ee_material[kMmaleMaterials],
    const double donor_ei_material[kMmaleMaterials],
    const double donor_v_r,
    const double donor_v_z) {
  MaterialFaceFlux2Mat flux{};
  for (int m = 0; m < kMmaleMaterials; ++m) {
    const double rho_m =
        finite_value(donor_rho_material[m]) ? fmax(donor_rho_material[m], 0.0) : 0.0;
    const double dm = rho_m * dV_material[m];
    flux.mass[m] = dm;
    flux.Ee[m] = dm * (finite_value(donor_ee_material[m])
                           ? fmax(donor_ee_material[m], 0.0)
                           : 0.0);
    flux.Ei[m] = dm * (finite_value(donor_ei_material[m])
                           ? fmax(donor_ei_material[m], 0.0)
                           : 0.0);
    flux.mom_r[m] = dm * donor_v_r;
    flux.mom_z[m] = dm * donor_v_z;
  }
  return flux;
}

__host__ __device__ inline void add_value(double* address, const double value) {
  if (address == nullptr || value == 0.0 || !finite_value(value)) {
    return;
  }
#if defined(__CUDA_ARCH__)
  atomicAdd(address, value);
#else
  *address += value;
#endif
}

__host__ __device__ inline void accumulate_face_flux_2mat(
    double* mass_new_mat,
    double* Ee_new_mat,
    double* Ei_new_mat,
    double* mom_r_new,
    double* mom_z_new,
    const int cell,
    const int n_mat,
    const MaterialFaceFlux2Mat& flux) {
  if (cell < 0 || n_mat < kMmaleMaterials) {
    return;
  }
  double mom_r_total = 0.0;
  double mom_z_total = 0.0;
  for (int m = 0; m < kMmaleMaterials; ++m) {
    const int idx = cell * n_mat + m;
    add_value(mass_new_mat != nullptr ? mass_new_mat + idx : nullptr, flux.mass[m]);
    add_value(Ee_new_mat != nullptr ? Ee_new_mat + idx : nullptr, flux.Ee[m]);
    add_value(Ei_new_mat != nullptr ? Ei_new_mat + idx : nullptr, flux.Ei[m]);
    mom_r_total += flux.mom_r[m];
    mom_z_total += flux.mom_z[m];
  }
  add_value(mom_r_new != nullptr ? mom_r_new + cell : nullptr, mom_r_total);
  add_value(mom_z_new != nullptr ? mom_z_new + cell : nullptr, mom_z_total);
}

__host__ __device__ inline IdealGasClosure2Mat pressure_equilibrate_ideal_gas_2mat(
    const double cell_volume,
    const double mass[kMmaleMaterials],
    const double Ee[kMmaleMaterials],
    const double Ei[kMmaleMaterials],
    const double gamma[kMmaleMaterials]) {
  IdealGasClosure2Mat out{};
  const double V = fmax(cell_volume, kTiny);
  double m_sum = 0.0;
  double p_numerator = 0.0;
  bool all_active_have_positive_energy = true;
  for (int m = 0; m < kMmaleMaterials; ++m) {
    const double mm = (mass[m] > 0.0 && finite_value(mass[m])) ? mass[m] : 0.0;
    const double E =
        fmax((finite_value(Ee[m]) ? Ee[m] : 0.0) +
                 (finite_value(Ei[m]) ? Ei[m] : 0.0),
             0.0);
    const double gm1 = (gamma[m] > 1.0 && finite_value(gamma[m]))
                           ? gamma[m] - 1.0
                           : 0.0;
    m_sum += mm;
    if (mm > 0.0) {
      all_active_have_positive_energy = all_active_have_positive_energy &&
                                        E > 0.0 && gm1 > 0.0;
      p_numerator += gm1 * E;
    }
  }

  if (m_sum <= 0.0 || !finite_value(m_sum)) {
    out.volfrac[0] = 1.0;
    out.volfrac[1] = 0.0;
    out.used_mass_fraction_fallback = true;
  } else if (p_numerator > 0.0 && finite_value(p_numerator) &&
             all_active_have_positive_energy) {
    out.pressure = p_numerator / V;
    double vf0 = 0.0;
    for (int m = 0; m < kMmaleMaterials; ++m) {
      const double E =
          fmax((finite_value(Ee[m]) ? Ee[m] : 0.0) +
                   (finite_value(Ei[m]) ? Ei[m] : 0.0),
               0.0);
      const double gm1 = gamma[m] - 1.0;
      const double mm =
          (mass[m] > 0.0 && finite_value(mass[m])) ? mass[m] : 0.0;
      out.volfrac[m] = (mm > 0.0) ? (gm1 * E) / fmax(p_numerator, kTiny) : 0.0;
    }
    vf0 = clamp01(out.volfrac[0]);
    out.volfrac[0] = vf0;
    out.volfrac[1] = 1.0 - vf0;
    out.pressure_equilibrated = true;
  } else {
    out.volfrac[0] = clamp01(mass[0] / m_sum);
    out.volfrac[1] = 1.0 - out.volfrac[0];
    out.used_mass_fraction_fallback = true;
  }

  for (int m = 0; m < kMmaleMaterials; ++m) {
    const double mm = (mass[m] > 0.0 && finite_value(mass[m])) ? mass[m] : 0.0;
    const double Vm = fmax(out.volfrac[m] * V, kTiny);
    out.rho[m] = (mm > 0.0) ? mm / Vm : 0.0;
    out.ee[m] = (mm > 0.0 && finite_value(Ee[m])) ? fmax(Ee[m], 0.0) / mm : 0.0;
    out.ei[m] = (mm > 0.0 && finite_value(Ei[m])) ? fmax(Ei[m], 0.0) / mm : 0.0;
  }
  return out;
}

__host__ __device__ inline void write_closure_to_cell_arrays_2mat(
    double* volfrac_cm,
    double* rho_mat_cm,
    double* ee_mat_cm,
    double* ei_mat_cm,
    double* rho_cell,
    double* ee_cell,
    double* ei_cell,
    const int cell,
    const int n_mat,
    const double cell_volume,
    const double mass[kMmaleMaterials],
    const double Ee[kMmaleMaterials],
    const double Ei[kMmaleMaterials],
    const IdealGasClosure2Mat& closure) {
  if (cell < 0 || n_mat < kMmaleMaterials) {
    return;
  }
  for (int m = 0; m < kMmaleMaterials; ++m) {
    const int idx = cell * n_mat + m;
    if (volfrac_cm != nullptr) {
      volfrac_cm[idx] = closure.volfrac[m];
    }
    if (rho_mat_cm != nullptr) {
      rho_mat_cm[idx] = closure.rho[m];
    }
    if (ee_mat_cm != nullptr) {
      ee_mat_cm[idx] = closure.ee[m];
    }
    if (ei_mat_cm != nullptr) {
      ei_mat_cm[idx] = closure.ei[m];
    }
  }

  const double m_total = fmax(mass[0], 0.0) + fmax(mass[1], 0.0);
  const double V = fmax(cell_volume, kTiny);
  if (rho_cell != nullptr) {
    rho_cell[cell] = m_total / V;
  }
  if (ee_cell != nullptr) {
    ee_cell[cell] = (m_total > 0.0) ? (fmax(Ee[0], 0.0) + fmax(Ee[1], 0.0)) /
                                         m_total
                                   : 0.0;
  }
  if (ei_cell != nullptr) {
    ei_cell[cell] = (m_total > 0.0) ? (fmax(Ei[0], 0.0) + fmax(Ei[1], 0.0)) /
                                         m_total
                                   : 0.0;
  }
}

}  // namespace tenryu::hydro::mmale
