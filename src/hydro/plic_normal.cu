#include "hydro/plic_normal.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/plic_geometry.cuh"

namespace tenryu::hydro::plic {
namespace {

constexpr double kTiny = 1.0e-30;
constexpr double kVolfracEps = 1.0e-10;
constexpr double kTwoPi = 2.0 * kPi;

int grid_index(const int i, const int j, const int n) {
  return i * n + j;
}

double clamp01(const double x) {
  return std::clamp(x, 0.0, 1.0);
}

bool normalize(double& r, double& z) {
  const double norm = std::sqrt(r * r + z * z);
  if (!(norm > kTiny)) {
    r = 0.0;
    z = 0.0;
    return false;
  }
  r /= norm;
  z /= norm;
  return true;
}

double meridional_area(const double r0,
                       const double z0,
                       const double r1,
                       const double z1,
                       const double r2,
                       const double z2,
                       const double r3,
                       const double z3) {
  const double cross =
      r0 * z1 - r1 * z0 + r1 * z2 - r2 * z1 + r2 * z3 - r3 * z2 +
      r3 * z0 - r0 * z3;
  return 0.5 * std::abs(cross);
}

double cell_volume(const double r0,
                   const double z0,
                   const double r1,
                   const double z1,
                   const double r2,
                   const double z2,
                   const double r3,
                   const double z3) {
  return std::abs(tenryu::hydro::ale::detail::rz_signed_quad_volume(
      r0, z0, r1, z1, r2, z2, r3, z3));
}

double target_cell_volfrac(const double* vf,
                           const int target_i,
                           const int target_j,
                           const int stencil_size) {
  return clamp01(vf[grid_index(target_i, target_j, stencil_size)]);
}

double one_sided_or_centered_slope(const double f_m,
                                   const double f_0,
                                   const double f_p,
                                   const double x_m,
                                   const double x_0,
                                   const double x_p,
                                   const bool has_m,
                                   const bool has_p) {
  if (has_m && has_p && std::abs(x_p - x_m) > kTiny) {
    return (f_p - f_m) / (x_p - x_m);
  }
  if (has_p && std::abs(x_p - x_0) > kTiny) {
    return (f_p - f_0) / (x_p - x_0);
  }
  if (has_m && std::abs(x_0 - x_m) > kTiny) {
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

YoungsSeed compute_youngs_seed(const double* vf,
                               const double* r_centroid,
                               const double* z_centroid,
                               const int target_i,
                               const int target_j,
                               const int stencil_size,
                               const double h_eff) {
  const int c = grid_index(target_i, target_j, stencil_size);
  const bool has_im = target_i > 0;
  const bool has_ip = target_i + 1 < stencil_size;
  const bool has_jm = target_j > 0;
  const bool has_jp = target_j + 1 < stencil_size;

  double grad_r = 0.0;
  if (target_i != 0) {
    const int im = grid_index(std::max(0, target_i - 1), target_j, stencil_size);
    const int ip = grid_index(std::min(stencil_size - 1, target_i + 1),
                              target_j, stencil_size);
    grad_r = one_sided_or_centered_slope(vf[im], vf[c], vf[ip],
                                         r_centroid[im], r_centroid[c],
                                         r_centroid[ip], has_im, has_ip);
  }

  const int jm = grid_index(target_i, std::max(0, target_j - 1), stencil_size);
  const int jp = grid_index(target_i, std::min(stencil_size - 1, target_j + 1),
                            stencil_size);
  const double grad_z = one_sided_or_centered_slope(vf[jm], vf[c], vf[jp],
                                                    z_centroid[jm],
                                                    z_centroid[c],
                                                    z_centroid[jp],
                                                    has_jm, has_jp);

  YoungsSeed seed;
  seed.grad_norm = std::sqrt(grad_r * grad_r + grad_z * grad_z);
  const double g_scaled = h_eff * seed.grad_norm;
  const double threshold =
      std::max(1.0e-8,
               10.0 * kVolfracEps *
                   std::sqrt(static_cast<double>(stencil_size * stencil_size)));
  seed.ill_conditioned = (g_scaled < threshold);
  if (seed.grad_norm > kTiny) {
    seed.n_r = -grad_r / seed.grad_norm;
    seed.n_z = -grad_z / seed.grad_norm;
    seed.valid = true;
  }
  return seed;
}

double stencil_range(const double* vf, const int stencil_size) {
  const int n = stencil_size * stencil_size;
  double lo = std::numeric_limits<double>::infinity();
  double hi = -std::numeric_limits<double>::infinity();
  for (int k = 0; k < n; ++k) {
    lo = std::min(lo, vf[k]);
    hi = std::max(hi, vf[k]);
  }
  return hi - lo;
}

struct CandidateScore {
  double n_r = 0.0;
  double n_z = 0.0;
  double alpha = 0.0;
  double residual = std::numeric_limits<double>::infinity();
  double alpha_residual = 1.0;
  bool valid = false;
};

void neighbor_rect(const double rc,
                   const double zc,
                   const double dr,
                   const double dz,
                   double& r0,
                   double& z0,
                   double& r1,
                   double& z1,
                   double& r2,
                   double& z2,
                   double& r3,
                   double& z3) {
  r0 = rc - 0.5 * dr;
  z0 = zc - 0.5 * dz;
  r1 = rc + 0.5 * dr;
  z1 = z0;
  r2 = r1;
  z2 = zc + 0.5 * dz;
  r3 = r0;
  z3 = z2;
}

CandidateScore score_candidate(const double cell_r0,
                               const double cell_z0,
                               const double cell_r1,
                               const double cell_z1,
                               const double cell_r2,
                               const double cell_z2,
                               const double cell_r3,
                               const double cell_z3,
                               const double* vf,
                               const double* r_centroid,
                               const double* z_centroid,
                               const double* v_cell_grid,
                               const int target_i,
                               const int target_j,
                               const int stencil_size,
                               double n_r,
                               double n_z) {
  CandidateScore score;
  if (!normalize(n_r, n_z)) {
    return score;
  }

  const double v_target = cell_volume(cell_r0, cell_z0, cell_r1, cell_z1,
                                      cell_r2, cell_z2, cell_r3, cell_z3);
  const double f_target = target_cell_volfrac(vf, target_i, target_j, stencil_size);
  if (!(v_target > 0.0) || f_target <= 0.0 || f_target >= 1.0) {
    return score;
  }

  const AlphaFinderResult alpha = find_alpha_for_volume_fraction(
      cell_r0, cell_z0, cell_r1, cell_z1, cell_r2, cell_z2, cell_r3, cell_z3,
      n_r, n_z, v_target, f_target, 50, 1.0e-12);
  if (!alpha.converged && !(alpha.volume_fraction_residual < 1.0e-10)) {
    return score;
  }

  const double dr = 0.5 * (std::abs(cell_r1 - cell_r0) +
                           std::abs(cell_r2 - cell_r3));
  const double dz = 0.5 * (std::abs(cell_z3 - cell_z0) +
                           std::abs(cell_z2 - cell_z1));
  double err = 0.0;
  double wsum = 0.0;
  for (int i = 0; i < stencil_size; ++i) {
    for (int j = 0; j < stencil_size; ++j) {
      const int idx = grid_index(i, j, stencil_size);
      double nr0 = 0.0;
      double nz0 = 0.0;
      double nr1 = 0.0;
      double nz1 = 0.0;
      double nr2 = 0.0;
      double nz2 = 0.0;
      double nr3 = 0.0;
      double nz3 = 0.0;
      neighbor_rect(r_centroid[idx], z_centroid[idx], dr, dz, nr0, nz0, nr1,
                    nz1, nr2, nz2, nr3, nz3);
      double v_neighbor = v_cell_grid != nullptr ? v_cell_grid[idx] : 0.0;
      if (!(v_neighbor > 0.0)) {
        v_neighbor = cell_volume(nr0, nz0, nr1, nz1, nr2, nz2, nr3, nz3);
      }
      if (!(v_neighbor > 0.0)) {
        continue;
      }
      const double v_pred =
          v_below_in_cell(nr0, nz0, nr1, nz1, nr2, nz2, nr3, nz3, n_r, n_z,
                          alpha.alpha);
      const double f_pred = clamp01(v_pred / v_neighbor);
      const double diff = f_pred - clamp01(vf[idx]);
      const double weight = kTwoPi * std::max(r_centroid[idx], 0.0);
      err += weight * diff * diff;
      wsum += weight;
    }
  }

  score.n_r = n_r;
  score.n_z = n_z;
  score.alpha = alpha.alpha;
  score.alpha_residual = alpha.volume_fraction_residual;
  score.residual = (wsum > 0.0) ? std::sqrt(err / wsum) : std::sqrt(err);
  score.valid = std::isfinite(score.residual);
  return score;
}

NormalEstimatorResult estimate_from_arrays(const double cell_r0,
                                           const double cell_z0,
                                           const double cell_r1,
                                           const double cell_z1,
                                           const double cell_r2,
                                           const double cell_z2,
                                           const double cell_r3,
                                           const double cell_z3,
                                           const double* vf,
                                           const double* r_centroid,
                                           const double* z_centroid,
                                           const double* v_cell_grid,
                                           const int target_i,
                                           const int target_j,
                                           const int stencil_size,
                                           const std::string& mode) {
  NormalEstimatorResult result;
  if (vf == nullptr || r_centroid == nullptr || z_centroid == nullptr ||
      stencil_size <= 0 || target_i < 0 || target_i >= stencil_size ||
      target_j < 0 || target_j >= stencil_size) {
    return result;
  }

  const double area = meridional_area(cell_r0, cell_z0, cell_r1, cell_z1,
                                      cell_r2, cell_z2, cell_r3, cell_z3);
  const YoungsSeed seed =
      compute_youngs_seed(vf, r_centroid, z_centroid, target_i, target_j,
                          stencil_size, std::sqrt(std::max(area, 0.0)));
  result.grad_F_magnitude = seed.grad_norm;

  const bool pure_lvira = (mode == "LVIRA" || mode == "lvira");
  const bool youngs_only = (mode == "youngs" || mode == "Youngs");
  const bool use_lvira = !youngs_only;

  if (!pure_lvira && seed.valid) {
    CandidateScore youngs_score =
        score_candidate(cell_r0, cell_z0, cell_r1, cell_z1, cell_r2, cell_z2,
                        cell_r3, cell_z3, vf, r_centroid, z_centroid,
                        v_cell_grid, target_i, target_j, stencil_size,
                        seed.n_r, seed.n_z);
    if (youngs_score.valid) {
      result.n_r = youngs_score.n_r;
      result.n_z = youngs_score.n_z;
      result.alpha = youngs_score.alpha;
      result.residual_l2 = youngs_score.residual;
      result.valid = true;
      result.ill_conditioned = seed.ill_conditioned;
      if (!use_lvira) {
        return result;
      }
    } else if (!use_lvira) {
      return result;
    }
  }

  if (!use_lvira) {
    return result;
  }

  constexpr std::array<double, 16> kAnglesDeg = {
      0.0,   30.0,  45.0,  60.0,  90.0,  120.0, 135.0, 150.0,
      180.0, 210.0, 225.0, 240.0, 270.0, 300.0, 315.0, 330.0};
  CandidateScore best;
  if (result.valid) {
    best.n_r = result.n_r;
    best.n_z = result.n_z;
    best.alpha = result.alpha;
    best.residual = result.residual_l2;
    best.valid = true;
  }

  for (const double angle_deg : kAnglesDeg) {
    const double angle = angle_deg * kPi / 180.0;
    CandidateScore score =
        score_candidate(cell_r0, cell_z0, cell_r1, cell_z1, cell_r2, cell_z2,
                        cell_r3, cell_z3, vf, r_centroid, z_centroid,
                        v_cell_grid, target_i, target_j, stencil_size,
                        std::cos(angle), std::sin(angle));
    if (score.valid && (!best.valid || score.residual < best.residual)) {
      best = score;
    }
  }

  if (best.valid) {
    result.n_r = best.n_r;
    result.n_z = best.n_z;
    result.alpha = best.alpha;
    result.residual_l2 = best.residual;
    result.valid = true;
    result.ill_conditioned =
        seed.ill_conditioned && stencil_range(vf, stencil_size) < 1.0e-12;
  }
  return result;
}

}  // namespace

NormalEstimatorResult estimate_normal_for_test(
    const double cell_r0,
    const double cell_z0,
    const double cell_r1,
    const double cell_z1,
    const double cell_r2,
    const double cell_z2,
    const double cell_r3,
    const double cell_z3,
    const double* neighbor_volfrac_grid,
    const double* neighbor_r_centroid,
    const double* neighbor_z_centroid,
    const double* neighbor_v_cell,
    const int target_i,
    const int target_j,
    const int stencil_size,
    const std::string& mode) {
  return estimate_from_arrays(cell_r0, cell_z0, cell_r1, cell_z1, cell_r2,
                              cell_z2, cell_r3, cell_z3,
                              neighbor_volfrac_grid, neighbor_r_centroid,
                              neighbor_z_centroid, neighbor_v_cell, target_i,
                              target_j, stencil_size, mode);
}

NormalEstimatorResult largest_face_jump_normal_for_test(
    const double cell_r0,
    const double cell_z0,
    const double cell_r1,
    const double cell_z1,
    const double cell_r2,
    const double cell_z2,
    const double cell_r3,
    const double cell_z3,
    const double* neighbor_volfrac_grid,
    const double* neighbor_v_cell,
    const int target_i,
    const int target_j,
    const int stencil_size) {
  NormalEstimatorResult result;
  if (neighbor_volfrac_grid == nullptr || stencil_size <= 0) {
    return result;
  }
  const int c = grid_index(target_i, target_j, stencil_size);
  const double f0 = neighbor_volfrac_grid[c];
  struct Face {
    int di;
    int dj;
    double nr;
    double nz;
  };
  constexpr std::array<Face, 4> faces = {
      Face{-1, 0, -1.0, 0.0}, Face{1, 0, 1.0, 0.0},
      Face{0, -1, 0.0, -1.0}, Face{0, 1, 0.0, 1.0}};

  double best_jump = 0.0;
  double best_nr = 0.0;
  double best_nz = 0.0;
  for (const Face& face : faces) {
    const int ni = target_i + face.di;
    const int nj = target_j + face.dj;
    if (ni < 0 || ni >= stencil_size || nj < 0 || nj >= stencil_size) {
      continue;
    }
    const double fn = neighbor_volfrac_grid[grid_index(ni, nj, stencil_size)];
    const double jump = std::abs(fn - f0);
    if (jump > best_jump) {
      best_jump = jump;
      const double sign = (f0 >= fn) ? 1.0 : -1.0;
      best_nr = sign * face.nr;
      best_nz = sign * face.nz;
    }
  }
  if (!(best_jump > kVolfracEps)) {
    return result;
  }

  const double v = cell_volume(cell_r0, cell_z0, cell_r1, cell_z1, cell_r2,
                               cell_z2, cell_r3, cell_z3);
  const double f = clamp01(f0);
  const AlphaFinderResult alpha = find_alpha_for_volume_fraction(
      cell_r0, cell_z0, cell_r1, cell_z1, cell_r2, cell_z2, cell_r3, cell_z3,
      best_nr, best_nz, v, f, 50, 1.0e-12);
  result.n_r = best_nr;
  result.n_z = best_nz;
  result.alpha = alpha.alpha;
  result.residual_l2 = best_jump;
  result.ill_conditioned = false;
  result.valid = alpha.converged || alpha.volume_fraction_residual < 1.0e-10;
  result.fallback_used = "largest_face_jump";
  (void)neighbor_v_cell;
  return result;
}

Case2FallbackResult resolve_case2_fallback_for_test(
    const double cell_r0,
    const double cell_z0,
    const double cell_r1,
    const double cell_z1,
    const double cell_r2,
    const double cell_z2,
    const double cell_r3,
    const double cell_z3,
    const double* neighbor_volfrac_grid,
    const double* neighbor_r_centroid,
    const double* neighbor_z_centroid,
    const double* neighbor_v_cell,
    const int target_i,
    const int target_j,
    const int stencil_size,
    double prev_normal_r,
    double prev_normal_z,
    const bool prev_normal_valid,
    const double cell_volfrac_now,
    const double cell_volfrac_at_last_plic,
    const double freshness_volfrac_threshold) {
  Case2FallbackResult out;
  const bool fresh =
      std::abs(cell_volfrac_now - cell_volfrac_at_last_plic) <
      freshness_volfrac_threshold;
  if (prev_normal_valid && fresh && normalize(prev_normal_r, prev_normal_z)) {
    const double v = cell_volume(cell_r0, cell_z0, cell_r1, cell_z1, cell_r2,
                                 cell_z2, cell_r3, cell_z3);
    const AlphaFinderResult alpha = find_alpha_for_volume_fraction(
        cell_r0, cell_z0, cell_r1, cell_z1, cell_r2, cell_z2, cell_r3,
        cell_z3, prev_normal_r, prev_normal_z, v, clamp01(cell_volfrac_now),
        50, 1.0e-12);
    out.normal.n_r = prev_normal_r;
    out.normal.n_z = prev_normal_z;
    out.normal.alpha = alpha.alpha;
    out.normal.valid = alpha.converged || alpha.volume_fraction_residual < 1.0e-10;
    out.normal.ill_conditioned = false;
    out.normal.fallback_used = "previous_normal";
    out.reconstructed = out.normal.valid;
    return out;
  }
  out.normal.prev_normal_invalidated = prev_normal_valid && !fresh;

  NormalEstimatorResult lvira =
      estimate_normal_for_test(cell_r0, cell_z0, cell_r1, cell_z1, cell_r2,
                               cell_z2, cell_r3, cell_z3,
                               neighbor_volfrac_grid, neighbor_r_centroid,
                               neighbor_z_centroid, neighbor_v_cell, target_i,
                               target_j, stencil_size, "LVIRA");
  if (lvira.valid && !lvira.ill_conditioned) {
    lvira.fallback_used = "lvira_fixed_angles";
    lvira.prev_normal_invalidated = out.normal.prev_normal_invalidated;
    out.normal = lvira;
    out.reconstructed = true;
    return out;
  }

  NormalEstimatorResult face_jump =
      largest_face_jump_normal_for_test(cell_r0, cell_z0, cell_r1, cell_z1,
                                        cell_r2, cell_z2, cell_r3, cell_z3,
                                        neighbor_volfrac_grid, neighbor_v_cell,
                                        target_i, target_j, stencil_size);
  if (face_jump.valid) {
    face_jump.prev_normal_invalidated = out.normal.prev_normal_invalidated;
    out.normal = face_jump;
    out.reconstructed = true;
    return out;
  }

  out.normal.valid = false;
  out.normal.ill_conditioned = true;
  out.normal.fallback_used = "scalar_volfrac_advection_skip";
  out.degradation_event_required = true;
  return out;
}

double compute_eta_E_per_cell(const double* face_mass_flux,
                              const double* face_e_closure,
                              const double e_cell,
                              const double e_floor,
                              const double E_floor,
                              const int n_faces,
                              const int n_mat) {
  if (face_mass_flux == nullptr || face_e_closure == nullptr || n_faces <= 0 ||
      n_mat <= 0) {
    return 0.0;
  }

  double numerator = 0.0;
  double denominator = 0.0;
  const double e_ref = std::max(e_cell, e_floor);
  for (int f = 0; f < n_faces; ++f) {
    for (int m = 0; m < n_mat; ++m) {
      const int idx = f * n_mat + m;
      const double w = std::abs(face_mass_flux[idx]);
      numerator += w * std::abs(e_cell - face_e_closure[idx]);
      denominator += w * e_ref;
    }
  }
  return numerator / std::max(denominator, E_floor);
}

NormalEstimatorResult estimate_normal_youngs_seeded_lvira(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const int cell_idx,
    const int material_idx,
    double prev_normal_r,
    double prev_normal_z,
    const bool prev_normal_valid) {
  (void)prev_normal_r;
  (void)prev_normal_z;
  (void)prev_normal_valid;

  const int nr = cfg.mesh.nr;
  const int nz = (cfg.main.dim == 2) ? cfg.mesh.nz : 1;
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (cfg.main.dim != 2 || nr <= 0 || nz <= 0 || cell_idx < 0 ||
      cell_idx >= nr * nz || material_idx < 0 || material_idx >= n_mat) {
    return {};
  }

  const int ci = cell_idx / nz;
  const int cj = cell_idx - ci * nz;
  if (ci == 0) {
    NormalEstimatorResult axis;
    axis.ill_conditioned = true;
    axis.fallback_used = "axis_exempt";
    return axis;
  }

  std::vector<double> vf_host(state.volFrac.size(), 0.0);
  state.volFrac.copy_to_host(vf_host.data());
  std::vector<double> node_r(state.x_r.size(), 0.0);
  std::vector<double> node_z(state.x_z.size(), 0.0);
  state.x_r.copy_to_host(node_r.data());
  state.x_z.copy_to_host(node_z.data());

  constexpr int stencil = 3;
  std::array<double, stencil * stencil> vf{};
  std::array<double, stencil * stencil> rc{};
  std::array<double, stencil * stencil> zc{};
  std::array<double, stencil * stencil> vc{};
  for (int si = 0; si < stencil; ++si) {
    for (int sj = 0; sj < stencil; ++sj) {
      const int ii = std::clamp(ci + si - 1, 0, nr - 1);
      const int jj = std::clamp(cj + sj - 1, 0, nz - 1);
      const int sc = grid_index(si, sj, stencil);
      const int c = ii * nz + jj;
      vf[sc] = vf_host[static_cast<std::size_t>(c * n_mat + material_idx)];
      rc[sc] = state.mesh.cell_centroid_r[static_cast<std::size_t>(c)];
      zc[sc] = state.mesh.cell_centroid_z[static_cast<std::size_t>(c)];
      vc[sc] = state.mesh.cell_vol[static_cast<std::size_t>(c)];
    }
  }

  const int stride = nz + 1;
  const int n00 = ci * stride + cj;
  const int n10 = (ci + 1) * stride + cj;
  const int n11 = (ci + 1) * stride + (cj + 1);
  const int n01 = ci * stride + (cj + 1);

  return estimate_normal_for_test(
      node_r[static_cast<std::size_t>(n00)],
      node_z[static_cast<std::size_t>(n00)],
      node_r[static_cast<std::size_t>(n10)],
      node_z[static_cast<std::size_t>(n10)],
      node_r[static_cast<std::size_t>(n11)],
      node_z[static_cast<std::size_t>(n11)],
      node_r[static_cast<std::size_t>(n01)],
      node_z[static_cast<std::size_t>(n01)], vf.data(), rc.data(), zc.data(),
      vc.data(), 1, 1, stencil, cfg.numerics.plic.normal_estimator);
}

}  // namespace tenryu::hydro::plic
