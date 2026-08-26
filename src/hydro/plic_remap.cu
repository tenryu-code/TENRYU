#include "hydro/plic_remap.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "core/error.hpp"
#include "coupling/profile_observability.hpp"
#include "hydro/ale_remap.cuh"
#include "hydro/eos_context.hpp"
#include "hydro/oriented_swept_volume.cuh"
#include "hydro/plic_geometry.cuh"
#include "materials/eos_device_table.cuh"

namespace tenryu::hydro::plic {
namespace {

constexpr double kTiny = 1.0e-30;
constexpr int kCountsInterfaceCells = 0;
constexpr int kCountsActiveCells = 1;
constexpr int kCountsAttempts = 2;
constexpr int kCountsSuccesses = 3;
constexpr int kCountsAxisExempt = 4;
constexpr int kCountsClassD = 5;
constexpr int kCountsRepairs = 6;
constexpr int kCountsSize = 7;
constexpr int kMetricResidual = 0;
constexpr int kMetricDrift = 1;
constexpr int kMetricSwept = 2;
constexpr int kMetricsSize = 3;
constexpr int kMaxPlicMaterials = 8;
constexpr double kProtonMass = 1.6726219e-24;
constexpr double kEvToErg = 1.6022e-12;

struct Cf6MaterialParams {
  int n_mat = 0;
  int two_temperature = 1;
  double A[kMaxPlicMaterials] = {};
  double Z[kMaxPlicMaterials] = {};
};

__device__ inline double atomic_max_double(double* address, const double value) {
  if (!isfinite(value) || value <= 0.0) {
    return *address;
  }
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0ULL;
  do {
    assumed = old;
    const double old_value = __longlong_as_double(static_cast<long long>(assumed));
    if (old_value >= value) {
      break;
    }
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
}

__host__ __device__ inline int cm_index(const int c, const int m, const int n_mat) {
  return c * n_mat + m;
}

__host__ __device__ inline int mc_index(const int m, const int c, const int n_cells) {
  return m * n_cells + c;
}

__host__ __device__ inline double clamp01_device(const double x) {
  return fmin(1.0, fmax(0.0, x));
}

__device__ inline bool eos_table_valid(const tenryu::materials::DeviceEOSTableView& tab) {
  return tab.n_rho > 0 && tab.n_T > 0 && tab.P_table != nullptr;
}

__device__ inline double table_pressure_at_rho_T(
    const tenryu::materials::DeviceEOSTableView& tab,
    const double rho,
    const double T) {
  const auto rb = tenryu::materials::find_rho_bracket(tab, rho);
  const double logT = log(fmax(T, kTiny));
  return tenryu::materials::device_eos_pressure(tab, rb, logT);
}

__device__ inline double table_pressure_sum_at_rho(
    const tenryu::materials::DeviceEOSTableView& ion,
    const tenryu::materials::DeviceEOSTableView& electron,
    const tenryu::materials::DeviceEOSTableView& total,
    const double rho,
    const double Te,
    const double Ti,
    const int two_temperature) {
  const bool has_ion = eos_table_valid(ion);
  const bool has_electron = eos_table_valid(electron);
  const bool has_total = eos_table_valid(total);
  double p = 0.0;
  if (!two_temperature && has_total) {
    p += table_pressure_at_rho_T(total, rho, Te);
  } else {
    if (has_electron) {
      p += table_pressure_at_rho_T(electron, rho, Te);
    }
    if (has_ion) {
      p += table_pressure_at_rho_T(ion, rho, two_temperature ? Ti : Te);
    }
  }
  return p;
}

__device__ inline double table_rho_from_pressure(
    const tenryu::materials::DeviceEOSTableView& ion,
    const tenryu::materials::DeviceEOSTableView& electron,
    const tenryu::materials::DeviceEOSTableView& total,
    const double pressure,
    const double Te,
    const double Ti,
    const int two_temperature) {
  const bool has_ion = eos_table_valid(ion);
  const bool has_electron = eos_table_valid(electron);
  const bool has_total = eos_table_valid(total);
  if (!(pressure > 0.0) || (!has_ion && !has_electron && !has_total)) {
    return 0.0;
  }

  double log_rho_lo = -1.0e300;
  double log_rho_hi = 1.0e300;
  if (has_electron) {
    log_rho_lo = fmax(log_rho_lo, electron.log_rho_min);
    log_rho_hi = fmin(log_rho_hi, electron.log_rho_max);
  }
  if (has_ion && (two_temperature || !has_total)) {
    log_rho_lo = fmax(log_rho_lo, ion.log_rho_min);
    log_rho_hi = fmin(log_rho_hi, ion.log_rho_max);
  }
  if (has_total && !two_temperature) {
    log_rho_lo = fmax(log_rho_lo, total.log_rho_min);
    log_rho_hi = fmin(log_rho_hi, total.log_rho_max);
  }
  if (!(log_rho_hi >= log_rho_lo) || !isfinite(log_rho_lo) || !isfinite(log_rho_hi)) {
    return 0.0;
  }

  double lo = log_rho_lo;
  double hi = log_rho_hi;
  const double p_lo =
      table_pressure_sum_at_rho(ion, electron, total, exp(lo), Te, Ti, two_temperature);
  const double p_hi =
      table_pressure_sum_at_rho(ion, electron, total, exp(hi), Te, Ti, two_temperature);
  if (!isfinite(p_lo) || !isfinite(p_hi)) {
    return 0.0;
  }
  if (pressure <= p_lo) {
    return exp(lo);
  }
  if (pressure >= p_hi) {
    return exp(hi);
  }
  for (int iter = 0; iter < 36; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double p_mid =
        table_pressure_sum_at_rho(ion, electron, total, exp(mid), Te, Ti, two_temperature);
    if (!isfinite(p_mid)) {
      return 0.0;
    }
    if (p_mid < pressure) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return exp(0.5 * (lo + hi));
}

__device__ inline double ideal_gas_rho_from_pressure(const Cf6MaterialParams& params,
                                                     const int m,
                                                     const double pressure,
                                                     const double Te,
                                                     const double Ti) {
  if (m < 0 || m >= params.n_mat || !(pressure > 0.0)) {
    return 0.0;
  }
  const double A = fmax(params.A[m], 1.0e-30);
  const double Z = fmax(params.Z[m], 0.0);
  const double ion_T = params.two_temperature ? Ti : Te;
  const double coeff = kEvToErg * (Z * fmax(Te, kTiny) + fmax(ion_T, kTiny)) /
                       (A * kProtonMass);
  const double rho = (coeff > 0.0) ? pressure / coeff : 0.0;
  return (isfinite(rho) && rho > 0.0) ? rho : 0.0;
}

__device__ inline double local_material_density_proxy(
    const double* __restrict__ rho_old,
    const double* __restrict__ volfrac_mat,
    const int mat,
    const int donor_i,
    const int donor_j,
    const int nr,
    const int nz,
    const int n_cells) {
  double best_vf = -1.0;
  double best_rho = rho_old[ale::detail::cell_index(donor_i, donor_j, nz)];
  for (int di = -1; di <= 1; ++di) {
    const int ii = donor_i + di;
    if (ii < 0 || ii >= nr) {
      continue;
    }
    for (int dj = -1; dj <= 1; ++dj) {
      const int jj = donor_j + dj;
      if (jj < 0 || jj >= nz) {
        continue;
      }
      const int c = ale::detail::cell_index(ii, jj, nz);
      const double f = volfrac_mat[mc_index(mat, c, n_cells)];
      const double r = rho_old[c];
      if (f > best_vf && r > 0.0 && isfinite(r)) {
        best_vf = f;
        best_rho = r;
      }
    }
  }
  return (best_rho > 0.0 && isfinite(best_rho)) ? best_rho : 0.0;
}

__device__ inline bool plic_interface_cell(const double* __restrict__ volfrac_mat,
                                           const int c,
                                           const int n_cells,
                                           const int n_mat,
                                           const double threshold_min,
                                           const double threshold_max) {
  int active_count = 0;
  for (int m = 0; m < n_mat; ++m) {
    const double f = volfrac_mat[mc_index(m, c, n_cells)];
    if (f >= threshold_min && f <= threshold_max) {
      ++active_count;
    }
  }
  return active_count > 0;
}

__device__ inline double cf6_effective_density_for_donor(
    const double* __restrict__ rho_old,
    const double* __restrict__ volfrac_mat,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const tenryu::materials::DeviceEOSTableView* __restrict__ ion_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ total_views,
    const Cf6MaterialParams params,
    const int donor,
    const int donor_i,
    const int donor_j,
    const int nr,
    const int nz,
    const int n_cells,
    const int n_mat,
    const double threshold_min,
    const double threshold_max) {
  const double cell_mean_rho = rho_old[donor];
  if (!plic_interface_cell(
          volfrac_mat, donor, n_cells, n_mat, threshold_min, threshold_max)) {
    return cell_mean_rho;
  }
  if (n_mat > kMaxPlicMaterials || params.n_mat < n_mat) {
    return cell_mean_rho;
  }

  double vf[kMaxPlicMaterials];
  double rho_m[kMaxPlicMaterials];
  const double Te_c = (Te != nullptr) ? fmax(Te[donor], kTiny) : 1.0;
  const double Ti_c = (Ti != nullptr) ? fmax(Ti[donor], kTiny) : Te_c;
  const double pressure =
      fmax((Pe != nullptr ? Pe[donor] : 0.0) + (Pi != nullptr ? Pi[donor] : 0.0), 0.0);
  for (int m = 0; m < n_mat; ++m) {
    vf[m] = volfrac_mat[mc_index(m, donor, n_cells)];
    double rho_est = 0.0;
    if (pressure > 0.0 && ion_views != nullptr && electron_views != nullptr &&
        total_views != nullptr) {
      rho_est = table_rho_from_pressure(
          ion_views[m], electron_views[m], total_views[m], pressure, Te_c, Ti_c,
          params.two_temperature);
    }
    if (!(rho_est > 0.0) || !isfinite(rho_est)) {
      rho_est =
          ideal_gas_rho_from_pressure(params, m, pressure, Te_c, Ti_c);
    }
    if (!(rho_est > 0.0) || !isfinite(rho_est)) {
      rho_est = local_material_density_proxy(
          rho_old, volfrac_mat, m, donor_i, donor_j, nr, nz, n_cells);
    }
    rho_m[m] = (rho_est > 0.0 && isfinite(rho_est)) ? rho_est : cell_mean_rho;
  }
  return rho_material_aware_donor_density(
      cell_mean_rho, vf, rho_m, n_mat, true);
}

__device__ inline bool normalize2(double& r, double& z) {
  const double n = sqrt(r * r + z * z);
  if (!(n > kTiny) || !isfinite(n)) {
    r = 0.0;
    z = 0.0;
    return false;
  }
  r /= n;
  z /= n;
  return true;
}

__device__ inline double abs_cell_volume(const double* xr,
                                         const double* xz,
                                         const int i,
                                         const int j,
                                         const int nz) {
  const int n00 = ale::detail::node_index(i, j, nz);
  const int n10 = ale::detail::node_index(i + 1, j, nz);
  const int n11 = ale::detail::node_index(i + 1, j + 1, nz);
  const int n01 = ale::detail::node_index(i, j + 1, nz);
  return fabs(ale::detail::rz_signed_quad_volume(xr[n00],
                                                  xz[n00],
                                                  xr[n10],
                                                  xz[n10],
                                                  xr[n11],
                                                  xz[n11],
                                                  xr[n01],
                                                  xz[n01]));
}

__device__ inline double cell_span(const double* xr,
                                   const double* xz,
                                   const int i,
                                   const int j,
                                   const int nz) {
  const int n00 = ale::detail::node_index(i, j, nz);
  const int n10 = ale::detail::node_index(i + 1, j, nz);
  const int n11 = ale::detail::node_index(i + 1, j + 1, nz);
  const int n01 = ale::detail::node_index(i, j + 1, nz);
  double r_min = fmin(fmin(xr[n00], xr[n10]), fmin(xr[n11], xr[n01]));
  double r_max = fmax(fmax(xr[n00], xr[n10]), fmax(xr[n11], xr[n01]));
  double z_min = fmin(fmin(xz[n00], xz[n10]), fmin(xz[n11], xz[n01]));
  double z_max = fmax(fmax(xz[n00], xz[n10]), fmax(xz[n11], xz[n01]));
  return fmax(fmax(r_max - r_min, z_max - z_min), kTiny);
}

__device__ inline double centered_or_one_sided_slope(const double fm,
                                                     const double f0,
                                                     const double fp,
                                                     const double xm,
                                                     const double x0,
                                                     const double xp,
                                                     const bool has_m,
                                                     const bool has_p) {
  if (has_m && has_p && fabs(xp - xm) > kTiny) {
    return (fp - fm) / (xp - xm);
  }
  if (has_p && fabs(xp - x0) > kTiny) {
    return (fp - f0) / (xp - x0);
  }
  if (has_m && fabs(x0 - xm) > kTiny) {
    return (f0 - fm) / (x0 - xm);
  }
  return 0.0;
}

__device__ inline bool interface_segment_centroid(const double r0,
                                                  const double z0,
                                                  const double r1,
                                                  const double z1,
                                                  const double r2,
                                                  const double z2,
                                                  const double r3,
                                                  const double z3,
                                                  const double nr,
                                                  const double nz,
                                                  const double alpha,
                                                  double& rc,
                                                  double& zc) {
  double ir[4] = {0.0, 0.0, 0.0, 0.0};
  double iz[4] = {0.0, 0.0, 0.0, 0.0};
  const double vr[4] = {r0, r1, r2, r3};
  const double vz[4] = {z0, z1, z2, z3};
  int n = 0;
  for (int e = 0; e < 4; ++e) {
    const int ep = (e + 1) & 3;
    const double s0 = nr * vr[e] + nz * vz[e] - alpha;
    const double s1 = nr * vr[ep] + nz * vz[ep] - alpha;
    if (fabs(s0) <= 1.0e-14 && n < 4) {
      ir[n] = vr[e];
      iz[n] = vz[e];
      ++n;
    }
    if (s0 * s1 < 0.0 && n < 4) {
      const double t = s0 / (s0 - s1);
      ir[n] = vr[e] + t * (vr[ep] - vr[e]);
      iz[n] = vz[e] + t * (vz[ep] - vz[e]);
      ++n;
    }
  }
  if (n < 2) {
    rc = 0.25 * (r0 + r1 + r2 + r3);
    zc = 0.25 * (z0 + z1 + z2 + z3);
    return false;
  }
  rc = 0.5 * (ir[0] + ir[1]);
  zc = 0.5 * (iz[0] + iz[1]);
  return isfinite(rc) && isfinite(zc);
}

__global__ void build_interface_mask_kernel(double* __restrict__ interface_mask,
                                            const double* __restrict__ volfrac,
                                            int* __restrict__ counts,
                                            const int n_cells,
                                            const int n_mat,
                                            const double threshold_min,
                                            const double threshold_max) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  bool interface_cell = false;
  for (int m = 0; m < n_mat; ++m) {
    const double f = volfrac[cm_index(c, m, n_mat)];
    if (f >= threshold_min && f <= threshold_max) {
      interface_cell = true;
    }
  }
  interface_mask[c] = interface_cell ? 1.0 : 0.0;
  if (interface_cell) {
    atomicAdd(counts + kCountsInterfaceCells, 1);
  }
}

__global__ void expand_active_mask_kernel(double* __restrict__ active_mask,
                                          const double* __restrict__ interface_mask,
                                          int* __restrict__ counts,
                                          const int nr,
                                          const int nz,
                                          const int radius) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  bool active = false;
  for (int di = -radius; di <= radius && !active; ++di) {
    const int ii = i + di;
    if (ii < 0 || ii >= nr) {
      continue;
    }
    for (int dj = -radius; dj <= radius; ++dj) {
      const int jj = j + dj;
      if (jj < 0 || jj >= nz) {
        continue;
      }
      if (interface_mask[ale::detail::cell_index(ii, jj, nz)] > 0.5) {
        active = true;
        break;
      }
    }
  }
  active_mask[c] = active ? 1.0 : 0.0;
  if (active) {
    atomicAdd(counts + kCountsActiveCells, 1);
  }
}

__global__ void reconstruct_kernel(double* __restrict__ valid,
                                   double* __restrict__ normal_r,
                                   double* __restrict__ normal_z,
                                   double* __restrict__ alpha,
                                   double* __restrict__ centroid_r,
                                   double* __restrict__ centroid_z,
                                   double* __restrict__ last_centroid_r,
                                   double* __restrict__ last_centroid_z,
                                   const double* __restrict__ active_mask,
                                   const double* __restrict__ volfrac,
                                   const double* __restrict__ xr,
                                   const double* __restrict__ xz,
                                   const double* __restrict__ vol,
                                   int* __restrict__ counts,
                                   double* __restrict__ metrics,
                                   const int nr,
                                   const int nz,
                                   const int n_mat,
                                   const double threshold_min,
                                   const double threshold_max,
                                   const int alpha_max_iter,
                                   const double alpha_tol_rel,
                                   const int has_previous) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int total = n_cells * n_mat;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_mat;
  const int m = idx - c * n_mat;
  valid[idx] = 0.0;
  normal_r[idx] = 0.0;
  normal_z[idx] = 0.0;
  alpha[idx] = 0.0;
  if (active_mask[c] <= 0.5) {
    return;
  }

  const double f = volfrac[idx];
  if (f <= threshold_min || f >= threshold_max) {
    return;
  }

  atomicAdd(counts + kCountsAttempts, 1);
  const int i = c / nz;
  const int j = c - i * nz;
  if (i == 0) {
    atomicAdd(counts + kCountsAxisExempt, 1);
    return;
  }

  const int im = max(i - 1, 0);
  const int ip = min(i + 1, nr - 1);
  const int jm = max(j - 1, 0);
  const int jp = min(j + 1, nz - 1);
  const int c_im = ale::detail::cell_index(im, j, nz);
  const int c_ip = ale::detail::cell_index(ip, j, nz);
  const int c_jm = ale::detail::cell_index(i, jm, nz);
  const int c_jp = ale::detail::cell_index(i, jp, nz);

  const double rc = ale::detail::cell_center_r(xr, i, j, nz);
  const double zc = ale::detail::cell_center_z(xz, i, j, nz);
  const double grad_r = centered_or_one_sided_slope(
      volfrac[cm_index(c_im, m, n_mat)],
      f,
      volfrac[cm_index(c_ip, m, n_mat)],
      ale::detail::cell_center_r(xr, im, j, nz),
      rc,
      ale::detail::cell_center_r(xr, ip, j, nz),
      im != i,
      ip != i);
  const double grad_z = centered_or_one_sided_slope(
      volfrac[cm_index(c_jm, m, n_mat)],
      f,
      volfrac[cm_index(c_jp, m, n_mat)],
      ale::detail::cell_center_z(xz, i, jm, nz),
      zc,
      ale::detail::cell_center_z(xz, i, jp, nz),
      jm != j,
      jp != j);
  double nr_hat = -grad_r;
  double nz_hat = -grad_z;
  const double grad_mag = sqrt(grad_r * grad_r + grad_z * grad_z);
  const double h_eff = cell_span(xr, xz, i, j, nz);
  const double grad_threshold = fmax(1.0e-8, 30.0 * threshold_min);
  if (!normalize2(nr_hat, nz_hat) || h_eff * grad_mag < grad_threshold) {
    atomicAdd(counts + kCountsClassD, 1);
    return;
  }

  const int n00 = ale::detail::node_index(i, j, nz);
  const int n10 = ale::detail::node_index(i + 1, j, nz);
  const int n11 = ale::detail::node_index(i + 1, j + 1, nz);
  const int n01 = ale::detail::node_index(i, j + 1, nz);
  const double v_cell = vol != nullptr ? vol[c] : abs_cell_volume(xr, xz, i, j, nz);
  const AlphaFinderResult alpha_result = find_alpha_for_volume_fraction(
      xr[n00],
      xz[n00],
      xr[n10],
      xz[n10],
      xr[n11],
      xz[n11],
      xr[n01],
      xz[n01],
      nr_hat,
      nz_hat,
      fmax(v_cell, kTiny),
      clamp01_device(f),
      alpha_max_iter,
      alpha_tol_rel);
  if (!alpha_result.converged &&
      !(alpha_result.volume_fraction_residual <= 10.0 * alpha_tol_rel)) {
    atomicAdd(counts + kCountsClassD, 1);
    return;
  }

  double iface_r = rc;
  double iface_z = zc;
  (void)interface_segment_centroid(xr[n00],
                                   xz[n00],
                                   xr[n10],
                                   xz[n10],
                                   xr[n11],
                                   xz[n11],
                                   xr[n01],
                                   xz[n01],
                                   nr_hat,
                                   nz_hat,
                                   alpha_result.alpha,
                                   iface_r,
                                   iface_z);
  valid[idx] = 1.0;
  normal_r[idx] = nr_hat;
  normal_z[idx] = nz_hat;
  alpha[idx] = alpha_result.alpha;
  centroid_r[idx] = iface_r;
  centroid_z[idx] = iface_z;
  atomicAdd(counts + kCountsSuccesses, 1);
  atomic_max_double(metrics + kMetricResidual, alpha_result.volume_fraction_residual);
  if (has_previous) {
    const double old_r = last_centroid_r[idx];
    const double old_z = last_centroid_z[idx];
    if (isfinite(old_r) && isfinite(old_z)) {
      const double drift =
          sqrt((iface_r - old_r) * (iface_r - old_r) + (iface_z - old_z) * (iface_z - old_z)) /
          cell_span(xr, xz, i, j, nz);
      atomic_max_double(metrics + kMetricDrift, drift);
    }
  }
  last_centroid_r[idx] = iface_r;
  last_centroid_z[idx] = iface_z;
}

__device__ inline double sign_clamped_flux(double raw, const double delta_v) {
  if (!(fabs(delta_v) > kTiny) || !isfinite(raw)) {
    return 0.0;
  }
  const double frac = clamp01_device(raw / delta_v);
  return frac * delta_v;
}

__device__ inline double material_flux_from_reconstruction(
    const double vf,
    const int valid,
    const double nr_hat,
    const double nz_hat,
    const double alpha,
    const double delta_v,
    const double r0,
    const double z0,
    const double r1,
    const double z1,
    const double r2,
    const double z2,
    const double r3,
    const double z3,
    const double threshold_min,
    const double threshold_max) {
  if (vf <= threshold_min) {
    return 0.0;
  }
  if (vf >= threshold_max) {
    return delta_v;
  }
  if (valid) {
    const double clipped =
        v_below_in_cell(r0, z0, r1, z1, r2, z2, r3, z3, nr_hat, nz_hat, alpha);
    return sign_clamped_flux(clipped, delta_v);
  }
  return clamp01_device(vf) * delta_v;
}

template <int MaxMaterials>
__device__ inline void close_face_flux(double* raw,
                                       const double* vf,
                                       const int n_mat,
                                       const double delta_v) {
  double sum = 0.0;
  int largest = 0;
  double largest_vf = -1.0;
  for (int m = 0; m < n_mat; ++m) {
    sum += raw[m];
    if (vf[m] > largest_vf) {
      largest_vf = vf[m];
      largest = m;
    }
  }
  raw[largest] += delta_v - sum;
}

template <bool DonorSignFixed>
__global__ void compute_r_face_flux_kernel(double* __restrict__ flux_r,
                                           const double* __restrict__ volfrac_mat,
                                           const double* __restrict__ recon_valid,
                                           const double* __restrict__ normal_r,
                                           const double* __restrict__ normal_z,
                                           const double* __restrict__ alpha,
                                           const double* __restrict__ vol_old,
                                           const double* __restrict__ xr_old,
                                           const double* __restrict__ xz_old,
                                           const double* __restrict__ xr_new,
                                           const double* __restrict__ xz_new,
                                           double* __restrict__ metrics,
                                           const int nr,
                                           const int nz,
                                           const int n_mat,
                                           const double threshold_min,
                                           const double threshold_max) {
  constexpr int kMaxMaterials = 8;
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_faces = (nr + 1) * nz;
  const int n_cells = nr * nz;
  if (face >= n_faces) {
    return;
  }
  for (int m = 0; m < n_mat; ++m) {
    flux_r[face * n_mat + m] = 0.0;
  }
  const int i_face = face / nz;
  const int j = face - i_face * nz;
  if (i_face <= 0 || i_face >= nr || n_mat > kMaxMaterials) {
    return;
  }
  const double delta_v =
      ale::detail::swept_volume_r_face(xr_old, xz_old, xr_new, xz_new, i_face, j, nz);
  if (!(fabs(delta_v) > kTiny) || !isfinite(delta_v)) {
    return;
  }
  const int donor_i = DonorSignFixed ? ((delta_v > 0.0) ? i_face : i_face - 1)
                                     : ((delta_v > 0.0) ? i_face - 1 : i_face);
  const int donor = ale::detail::cell_index(donor_i, j, nz);
  const int n0 = ale::detail::node_index(i_face, j, nz);
  const int n1 = ale::detail::node_index(i_face, j + 1, nz);
  double raw[kMaxMaterials];
  double vf[kMaxMaterials];
  for (int m = 0; m < n_mat; ++m) {
    vf[m] = volfrac_mat[mc_index(m, donor, n_cells)];
    const int ridx = cm_index(donor, m, n_mat);
    raw[m] = material_flux_from_reconstruction(vf[m],
                                               recon_valid[ridx] > 0.5,
                                               normal_r[ridx],
                                               normal_z[ridx],
                                               alpha[ridx],
                                               delta_v,
                                               xr_old[n0],
                                               xz_old[n0],
                                               xr_new[n0],
                                               xz_new[n0],
                                               xr_new[n1],
                                               xz_new[n1],
                                               xr_old[n1],
                                               xz_old[n1],
                                               threshold_min,
                                               threshold_max);
  }
  close_face_flux<kMaxMaterials>(raw, vf, n_mat, delta_v);
  // Legacy stores +geometric sweep; fixed stores conservative outflow from
  // the low-index cell, i.e. the negated geometric sweep.
  for (int m = 0; m < n_mat; ++m) {
    flux_r[face * n_mat + m] = DonorSignFixed ? -raw[m] : raw[m];
  }
  const int left = ale::detail::cell_index(i_face - 1, j, nz);
  const int right = ale::detail::cell_index(i_face, j, nz);
  const double denom = fmax(fmin(vol_old[left], vol_old[right]), kTiny);
  atomic_max_double(metrics + kMetricSwept, fabs(delta_v) / denom);
}

template <bool DonorSignFixed>
__global__ void compute_z_face_flux_kernel(double* __restrict__ flux_z,
                                           const double* __restrict__ volfrac_mat,
                                           const double* __restrict__ recon_valid,
                                           const double* __restrict__ normal_r,
                                           const double* __restrict__ normal_z,
                                           const double* __restrict__ alpha,
                                           const double* __restrict__ vol_old,
                                           const double* __restrict__ xr_old,
                                           const double* __restrict__ xz_old,
                                           const double* __restrict__ xr_new,
                                           const double* __restrict__ xz_new,
                                           double* __restrict__ metrics,
                                           const int nr,
                                           const int nz,
                                           const int n_mat,
                                           const double threshold_min,
                                           const double threshold_max) {
  constexpr int kMaxMaterials = 8;
  const int face = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_faces = nr * (nz + 1);
  const int n_cells = nr * nz;
  if (face >= n_faces) {
    return;
  }
  for (int m = 0; m < n_mat; ++m) {
    flux_z[face * n_mat + m] = 0.0;
  }
  const int i = face / (nz + 1);
  const int j_face = face - i * (nz + 1);
  if (j_face <= 0 || j_face >= nz || n_mat > kMaxMaterials) {
    return;
  }
  const double delta_v =
      ale::detail::swept_volume_z_face(xr_old, xz_old, xr_new, xz_new, i, j_face, nz);
  if (!(fabs(delta_v) > kTiny) || !isfinite(delta_v)) {
    return;
  }
  const int donor_j = DonorSignFixed ? ((delta_v > 0.0) ? j_face : j_face - 1)
                                     : ((delta_v > 0.0) ? j_face - 1 : j_face);
  const int donor = ale::detail::cell_index(i, donor_j, nz);
  const int n0 = ale::detail::node_index(i, j_face, nz);
  const int n1 = ale::detail::node_index(i + 1, j_face, nz);
  double raw[kMaxMaterials];
  double vf[kMaxMaterials];
  for (int m = 0; m < n_mat; ++m) {
    vf[m] = volfrac_mat[mc_index(m, donor, n_cells)];
    const int ridx = cm_index(donor, m, n_mat);
    raw[m] = material_flux_from_reconstruction(vf[m],
                                               recon_valid[ridx] > 0.5,
                                               normal_r[ridx],
                                               normal_z[ridx],
                                               alpha[ridx],
                                               delta_v,
                                               xr_old[n0],
                                               xz_old[n0],
                                               xr_old[n1],
                                               xz_old[n1],
                                               xr_new[n1],
                                               xz_new[n1],
                                               xr_new[n0],
                                               xz_new[n0],
                                               threshold_min,
                                               threshold_max);
  }
  close_face_flux<kMaxMaterials>(raw, vf, n_mat, delta_v);
  // Legacy stores +geometric sweep; fixed stores conservative outflow from
  // the low-index cell, i.e. the negated geometric sweep.
  for (int m = 0; m < n_mat; ++m) {
    flux_z[face * n_mat + m] = DonorSignFixed ? -raw[m] : raw[m];
  }
  const int low = ale::detail::cell_index(i, j_face - 1, nz);
  const int high = ale::detail::cell_index(i, j_face, nz);
  const double denom = fmax(fmin(vol_old[low], vol_old[high]), kTiny);
  atomic_max_double(metrics + kMetricSwept, fabs(delta_v) / denom);
}

__device__ inline double cf6_r_density_flux(
    const int face,
    const double* __restrict__ rho_old,
    const double* __restrict__ volfrac_mat,
    const double* __restrict__ flux_r,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const tenryu::materials::DeviceEOSTableView* __restrict__ ion_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ total_views,
    const Cf6MaterialParams params,
    const int nr,
    const int nz,
    const int n_mat,
    const double threshold_min,
    const double threshold_max) {
  const int n_cells = nr * nz;
  const int i_face = face / nz;
  const int j = face - i_face * nz;
  if (i_face <= 0 || i_face >= nr) {
    return 0.0;
  }
  // PLIC face flux is consumed as stored: legacy producers close to the raw
  // geometric sweep, fixed producers close to conservative outflow from the
  // low-index cell. The donor is intentionally selected from this stored sign.
  double delta_v = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    delta_v += flux_r[face * n_mat + m];
  }
  if (!(fabs(delta_v) > kTiny) || !isfinite(delta_v)) {
    return 0.0;
  }
  const int donor_i = (delta_v > 0.0) ? i_face - 1 : i_face;
  const int donor = ale::detail::cell_index(donor_i, j, nz);
  const double rho_eff = cf6_effective_density_for_donor(rho_old,
                                                         volfrac_mat,
                                                         Te,
                                                         Ti,
                                                         Pe,
                                                         Pi,
                                                         ion_views,
                                                         electron_views,
                                                         total_views,
                                                         params,
                                                         donor,
                                                         donor_i,
                                                         j,
                                                         nr,
                                                         nz,
                                                         n_cells,
                                                         n_mat,
                                                         threshold_min,
                                                         threshold_max);
  return delta_v * rho_eff;
}

__device__ inline double cf6_z_density_flux(
    const int face,
    const double* __restrict__ rho_old,
    const double* __restrict__ volfrac_mat,
    const double* __restrict__ flux_z,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const tenryu::materials::DeviceEOSTableView* __restrict__ ion_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ total_views,
    const Cf6MaterialParams params,
    const int nr,
    const int nz,
    const int n_mat,
    const double threshold_min,
    const double threshold_max) {
  const int n_cells = nr * nz;
  const int i = face / (nz + 1);
  const int j_face = face - i * (nz + 1);
  if (j_face <= 0 || j_face >= nz) {
    return 0.0;
  }
  // PLIC face flux is consumed as stored: legacy producers close to the raw
  // geometric sweep, fixed producers close to conservative outflow from the
  // low-index cell. The donor is intentionally selected from this stored sign.
  double delta_v = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    delta_v += flux_z[face * n_mat + m];
  }
  if (!(fabs(delta_v) > kTiny) || !isfinite(delta_v)) {
    return 0.0;
  }
  const int donor_j = (delta_v > 0.0) ? j_face - 1 : j_face;
  const int donor = ale::detail::cell_index(i, donor_j, nz);
  const double rho_eff = cf6_effective_density_for_donor(rho_old,
                                                         volfrac_mat,
                                                         Te,
                                                         Ti,
                                                         Pe,
                                                         Pi,
                                                         ion_views,
                                                         electron_views,
                                                         total_views,
                                                         params,
                                                         donor,
                                                         i,
                                                         donor_j,
                                                         nr,
                                                         nz,
                                                         n_cells,
                                                         n_mat,
                                                         threshold_min,
                                                         threshold_max);
  return delta_v * rho_eff;
}

__global__ void remap_density_cf6_kernel(
    double* __restrict__ rho,
    const double* __restrict__ rho_old,
    const double* __restrict__ volfrac_mat,
    const double* __restrict__ flux_r,
    const double* __restrict__ flux_z,
    const double* __restrict__ vol_old,
    const double* __restrict__ vol_new,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const tenryu::materials::DeviceEOSTableView* __restrict__ ion_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ total_views,
    const Cf6MaterialParams params,
    int* __restrict__ counts,
    const int nr,
    const int nz,
    const int n_mat,
    const double threshold_min,
    const double threshold_max) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int r_plus = (i + 1) * nz + j;
  const int r_minus = i * nz + j;
  const int z_plus = i * (nz + 1) + (j + 1);
  const int z_minus = i * (nz + 1) + j;
  // CF6 consumes the stored signed conservative face flux unchanged; the sign
  // convention is fixed by PLIC face-flux production.
  const double q_old = fmax(rho_old[c], 0.0) * fmax(vol_old[c], 0.0);
  const double q_new =
      q_old -
      cf6_r_density_flux(r_plus,
                         rho_old,
                         volfrac_mat,
                         flux_r,
                         Te,
                         Ti,
                         Pe,
                         Pi,
                         ion_views,
                         electron_views,
                         total_views,
                         params,
                         nr,
                         nz,
                         n_mat,
                         threshold_min,
                         threshold_max) +
      cf6_r_density_flux(r_minus,
                         rho_old,
                         volfrac_mat,
                         flux_r,
                         Te,
                         Ti,
                         Pe,
                         Pi,
                         ion_views,
                         electron_views,
                         total_views,
                         params,
                         nr,
                         nz,
                         n_mat,
                         threshold_min,
                         threshold_max) -
      cf6_z_density_flux(z_plus,
                         rho_old,
                         volfrac_mat,
                         flux_z,
                         Te,
                         Ti,
                         Pe,
                         Pi,
                         ion_views,
                         electron_views,
                         total_views,
                         params,
                         nr,
                         nz,
                         n_mat,
                         threshold_min,
                         threshold_max) +
      cf6_z_density_flux(z_minus,
                         rho_old,
                         volfrac_mat,
                         flux_z,
                         Te,
                         Ti,
                         Pe,
                         Pi,
                         ion_views,
                         electron_views,
                         total_views,
                         params,
                         nr,
                         nz,
                         n_mat,
                         threshold_min,
                         threshold_max);
  double rho_new = q_new / fmax(vol_new[c], kTiny);
  if (!isfinite(rho_new) || rho_new < 0.0) {
    rho_new = 0.0;
    atomicAdd(counts + kCountsRepairs, 1);
  }
  rho[c] = rho_new;
}

__global__ void gather_material_volume_kernel(double* __restrict__ volfrac_mat,
                                              const double* __restrict__ flux_r,
                                              const double* __restrict__ flux_z,
                                              const double* __restrict__ vol_old,
                                              const double* __restrict__ vol_new,
                                              int* __restrict__ counts,
                                              const int nr,
                                              const int nz,
                                              const int n_mat) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int total = n_cells * n_mat;
  if (idx >= total) {
    return;
  }
  const int m = idx / n_cells;
  const int c = idx - m * n_cells;
  const int i = c / nz;
  const int j = c - i * nz;
  const double old_qbar = fmax(volfrac_mat[idx], 0.0) * fmax(vol_old[c], 0.0);
  const int r_plus = ((i + 1) * nz + j) * n_mat + m;
  const int r_minus = (i * nz + j) * n_mat + m;
  const int z_plus = (i * (nz + 1) + (j + 1)) * n_mat + m;
  const int z_minus = (i * (nz + 1) + j) * n_mat + m;
  // Material volumes consume the stored PLIC face flux unchanged. In fixed
  // mode positive values are conservative outflow from the low-index cell.
  const double qbar = old_qbar - flux_r[r_plus] + flux_r[r_minus] -
                      flux_z[z_plus] + flux_z[z_minus];
  double vf = qbar / fmax(vol_new[c], kTiny);
  if (!isfinite(vf)) {
    vf = 0.0;
    atomicAdd(counts + kCountsRepairs, 1);
  }
  if (vf < -1.0e-12 || vf > 1.0 + 1.0e-12) {
    atomicAdd(counts + kCountsRepairs, 1);
  }
  volfrac_mat[idx] = clamp01_device(vf);
}

__global__ void snap_material_residual_kernel(double* __restrict__ volfrac_mat,
                                              double* __restrict__ cell_residual,
                                              int* __restrict__ counts,
                                              double* __restrict__ metrics,
                                              const int n_cells,
                                              const int n_mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double sum = 0.0;
  int largest = 0;
  double largest_vf = -1.0;
  for (int m = 0; m < n_mat; ++m) {
    const double vf = volfrac_mat[mc_index(m, c, n_cells)];
    sum += vf;
    if (vf > largest_vf) {
      largest_vf = vf;
      largest = m;
    }
  }
  const double residual = 1.0 - sum;
  cell_residual[c] = residual;
  atomic_max_double(metrics + kMetricResidual, fabs(residual));
  const int idx = mc_index(largest, c, n_cells);
  double adjusted = volfrac_mat[idx] + residual;
  if (adjusted < -1.0e-12 || adjusted > 1.0 + 1.0e-12) {
    atomicAdd(counts + kCountsRepairs, 1);
  }
  volfrac_mat[idx] = clamp01_device(adjusted);
}

PlicRemapStatus status_from_device(const int* d_counts,
                                   const double* d_metrics,
                                   cudaStream_t stream) {
  int counts[kCountsSize] = {};
  double metrics[kMetricsSize] = {};
  CUDA_CHECK(cudaMemcpyAsync(counts,
                             d_counts,
                             sizeof(counts),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(metrics,
                             d_metrics,
                             sizeof(metrics),
                             cudaMemcpyDeviceToHost,
                             stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  PlicRemapStatus status;
  status.active = true;
  status.interface_cells = counts[kCountsInterfaceCells];
  status.active_cells = counts[kCountsActiveCells];
  status.reconstruction_attempts = counts[kCountsAttempts];
  status.reconstruction_successes = counts[kCountsSuccesses];
  status.axis_exempt_cells = counts[kCountsAxisExempt];
  status.class_d_events = counts[kCountsClassD];
  status.repair_events = counts[kCountsRepairs];
  status.max_volume_fraction_residual = metrics[kMetricResidual];
  status.max_interface_centroid_drift_relative = metrics[kMetricDrift];
  status.max_swept_fraction = metrics[kMetricSwept];
  return status;
}

void reset_device_counters(int*& d_counts, double*& d_metrics, cudaStream_t stream) {
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_counts), kCountsSize * sizeof(int)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_metrics), kMetricsSize * sizeof(double)));
  CUDA_CHECK(cudaMemsetAsync(d_counts, 0, kCountsSize * sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_metrics, 0, kMetricsSize * sizeof(double), stream));
}

Cf6MaterialParams make_cf6_material_params(const tenryu::core::Config& cfg,
                                           const int n_mat) {
  Cf6MaterialParams params;
  params.n_mat = std::min(n_mat, kMaxPlicMaterials);
  params.two_temperature = cfg.main.two_temperature ? 1 : 0;
  for (int m = 0; m < params.n_mat; ++m) {
    const auto& mat = cfg.materials.materials[static_cast<std::size_t>(m)];
    params.A[m] = mat.A;
    params.Z[m] = mat.Z;
  }
  return params;
}

void update_observability(const PlicRemapStatus& status,
                          tenryu::coupling::ProfileObservability* observability) {
  if (observability == nullptr || !status.active) {
    return;
  }
  observability->interface_cells_observed += status.interface_cells;
  observability->interface_reconstruction_attempt_count +=
      status.reconstruction_attempts;
  observability->interface_reconstruction_success_count +=
      status.reconstruction_successes;
  observability->axis_exempt_cells_count += status.axis_exempt_cells;
  observability->plic_max_volume_fraction_residual_observed =
      std::max(observability->plic_max_volume_fraction_residual_observed,
               status.max_volume_fraction_residual);
  if (status.class_d_events > 0) {
    observability->class_d_runtime_fires_matrix[2][1] += status.class_d_events;
  }
  if (status.repair_events > 0) {
    observability->class_d_runtime_fires_matrix[2][2] += status.repair_events;
  }
}

PlicRemapStatus launch_reconstruction_impl(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr,
    const double* d_xz,
    const double* d_vol,
    const int nr,
    const int nz,
    tenryu::coupling::ProfileObservability* observability,
    const bool update_obs,
    cudaStream_t stream) {
  PlicRemapStatus empty;
  if (!plic_runtime_active(state, cfg)) {
    return empty;
  }
  const int n_cells = nr * nz;
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (n_cells <= 0 || n_mat <= 0) {
    return empty;
  }
  ensure_plic_remap_scratch(state, cfg, n_cells, n_mat, nr, nz);

  int* d_counts = nullptr;
  double* d_metrics = nullptr;
  reset_device_counters(d_counts, d_metrics, stream);
  const int blocks_cells = (n_cells + 255) / 256;
  const int total_cm = n_cells * n_mat;
  const int blocks_cm = (total_cm + 255) / 256;
  CUDA_CHECK(cudaMemsetAsync(state.plic_interface_mask.data(),
                             0,
                             state.plic_interface_mask.size() * sizeof(double),
                             stream));
  CUDA_CHECK(cudaMemsetAsync(state.plic_active_mask.data(),
                             0,
                             state.plic_active_mask.size() * sizeof(double),
                             stream));
  build_interface_mask_kernel<<<blocks_cells, 256, 0, stream>>>(
      state.plic_interface_mask.data(),
      state.volFrac.data(),
      d_counts,
      n_cells,
      n_mat,
      cfg.numerics.plic.fast_path_threshold_min,
      cfg.numerics.plic.fast_path_threshold_max);
  CUDA_CHECK(cudaGetLastError());
  expand_active_mask_kernel<<<blocks_cells, 256, 0, stream>>>(
      state.plic_active_mask.data(),
      state.plic_interface_mask.data(),
      d_counts,
      nr,
      nz,
      cfg.numerics.plic.fast_path_halo_radius_cells);
  CUDA_CHECK(cudaGetLastError());
  const int has_previous = state.plic_last_reconstruction_step >= 0 ? 1 : 0;
  reconstruct_kernel<<<blocks_cm, 256, 0, stream>>>(
      state.plic_reconstruction_valid.data(),
      state.plic_normal_r.data(),
      state.plic_normal_z.data(),
      state.plic_alpha.data(),
      state.plic_centroid_r.data(),
      state.plic_centroid_z.data(),
      state.plic_last_centroid_r.data(),
      state.plic_last_centroid_z.data(),
      state.plic_active_mask.data(),
      state.volFrac.data(),
      d_xr,
      d_xz,
      d_vol,
      d_counts,
      d_metrics,
      nr,
      nz,
      n_mat,
      cfg.numerics.plic.fast_path_threshold_min,
      cfg.numerics.plic.fast_path_threshold_max,
      cfg.numerics.plic.alpha_solver_max_iter,
      cfg.numerics.plic.alpha_tolerance_rel,
      has_previous);
  CUDA_CHECK(cudaGetLastError());
  PlicRemapStatus status = status_from_device(d_counts, d_metrics, stream);
  CUDA_CHECK(cudaFree(d_metrics));
  CUDA_CHECK(cudaFree(d_counts));
  status.drift_triggered =
      status.max_interface_centroid_drift_relative >
      cfg.numerics.plic.drift_sensor_max_relative;
  state.plic_last_reconstruction_step = state.step;
  if (update_obs) {
    update_observability(status, observability);
  }
  return status;
}

}  // namespace

bool plic_runtime_active(const tenryu::core::State& state,
                         const tenryu::core::Config& cfg) {
  return cfg.numerics.plic.enabled && !cfg.numerics.plic.in_run_disabled &&
         !state.plic_remap_sticky_fallback;
}

std::size_t plic_cell_material_scratch_size(const tenryu::core::Config& cfg,
                                            const int n_cells) {
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (!cfg.numerics.plic.enabled || cfg.numerics.plic.in_run_disabled ||
      n_cells <= 0 || n_mat <= 0) {
    return 0;
  }
  return static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
}

void ensure_plic_remap_scratch(tenryu::core::State& state,
                               const tenryu::core::Config& cfg,
                               const int n_cells,
                               const int n_mat,
                               const int nr,
                               const int nz) {
  if (!plic_runtime_active(state, cfg) || n_cells <= 0 || n_mat <= 0) {
    return;
  }
  const std::size_t n_cell = static_cast<std::size_t>(n_cells);
  const std::size_t n_cm = n_cell * static_cast<std::size_t>(n_mat);
  const std::size_t n_r_faces =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz);
  const std::size_t n_z_faces =
      static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz + 1);
  state.plic_interface_mask.reset(n_cell);
  state.plic_active_mask.reset(n_cell);
  state.plic_reconstruction_valid.reset(n_cm);
  state.plic_normal_r.reset(n_cm);
  state.plic_normal_z.reset(n_cm);
  state.plic_alpha.reset(n_cm);
  state.plic_centroid_r.reset(n_cm);
  state.plic_centroid_z.reset(n_cm);
  state.plic_last_centroid_r.reset(n_cm);
  state.plic_last_centroid_z.reset(n_cm);
  state.plic_face_flux_r.reset(n_r_faces * static_cast<std::size_t>(n_mat));
  state.plic_face_flux_z.reset(n_z_faces * static_cast<std::size_t>(n_mat));
  state.plic_cell_residual.reset(n_cell);
}

void reset_plic_runtime(tenryu::core::State& state) {
  state.plic_remap_sticky_fallback = false;
  state.plic_consecutive_drift_triggers = 0;
  state.plic_last_reconstruction_step = -1;
  state.plic_interface_mask.reset(0);
  state.plic_active_mask.reset(0);
  state.plic_reconstruction_valid.reset(0);
  state.plic_normal_r.reset(0);
  state.plic_normal_z.reset(0);
  state.plic_alpha.reset(0);
  state.plic_centroid_r.reset(0);
  state.plic_centroid_z.reset(0);
  state.plic_last_centroid_r.reset(0);
  state.plic_last_centroid_z.reset(0);
  state.plic_face_flux_r.reset(0);
  state.plic_face_flux_z.reset(0);
  state.plic_cell_residual.reset(0);
}

PlicRemapStatus launch_plic_reconstruction(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr,
    const double* d_xz,
    const double* d_vol,
    const int nr,
    const int nz,
    tenryu::coupling::ProfileObservability* observability,
    cudaStream_t stream) {
  return launch_reconstruction_impl(
      state, cfg, d_xr, d_xz, d_vol, nr, nz, observability, true, stream);
}

PlicRemapStatus launch_plic_drift_sensor(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr,
    const double* d_xz,
    const double* d_vol,
    const int nr,
    const int nz,
    tenryu::coupling::ProfileObservability* observability,
    cudaStream_t stream) {
  PlicRemapStatus status = launch_reconstruction_impl(
      state, cfg, d_xr, d_xz, d_vol, nr, nz, observability, false, stream);
  status.drift_triggered =
      status.max_interface_centroid_drift_relative >
      cfg.numerics.plic.drift_sensor_max_relative;
  if (status.drift_triggered) {
    tenryu::core::log_warning(
        "[plic_remap] out-of-cycle interface reconstruction refresh triggered "
        "drift sensor; updating last interface centroids");
  }
  return status;
}

PlicRemapStatus launch_plic_material_volume_remap(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    double* d_volfrac_mat,
    const double* d_vol_old,
    const double* d_vol_new,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_xr_new,
    const double* d_xz_new,
    const int nr,
    const int nz,
    tenryu::coupling::ProfileObservability* observability,
    const tenryu::hydro::HydroEOSContext* eos_ctx,
    cudaStream_t stream) {
  PlicRemapStatus status = launch_reconstruction_impl(
      state, cfg, d_xr_old, d_xz_old, d_vol_old, nr, nz, observability, false, stream);
  if (!status.active) {
    return status;
  }
  if (status.reconstruction_attempts == 0) {
    status.fallback_engaged = true;
    return status;
  }
  const int n_cells = nr * nz;
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (n_mat > 8) {
    status.fallback_engaged = true;
    status.class_d_events += 1;
    tenryu::core::log_warning(
        "[plic_remap] scalar fallback engaged because the PLIC face kernel supports "
        "at most 8 materials");
    return status;
  }

  int* d_counts = nullptr;
  double* d_metrics = nullptr;
  reset_device_counters(d_counts, d_metrics, stream);
  const int n_r_faces = (nr + 1) * nz;
  const int n_z_faces = nr * (nz + 1);
  const int blocks_r = (n_r_faces + 255) / 256;
  const int blocks_z = (n_z_faces + 255) / 256;
  const int blocks_cm = (n_cells * n_mat + 255) / 256;
  const int blocks_cells = (n_cells + 255) / 256;
  double* d_rho_cf6_old = nullptr;
  const bool effective_donor_sign_fixed =
      swept_volume_convention_from_flag(
          cfg.numerics.ale.swept_volume_sign_fixed) ==
      SweptVolumeConvention::OrientedLowToHighV1;
  if (effective_donor_sign_fixed) {
    compute_r_face_flux_kernel<true><<<blocks_r, 256, 0, stream>>>(
        state.plic_face_flux_r.data(),
        d_volfrac_mat,
        state.plic_reconstruction_valid.data(),
        state.plic_normal_r.data(),
        state.plic_normal_z.data(),
        state.plic_alpha.data(),
        d_vol_old,
        d_xr_old,
        d_xz_old,
        d_xr_new,
        d_xz_new,
        d_metrics,
        nr,
        nz,
        n_mat,
        cfg.numerics.plic.fast_path_threshold_min,
        cfg.numerics.plic.fast_path_threshold_max);
  } else {
    compute_r_face_flux_kernel<false><<<blocks_r, 256, 0, stream>>>(
        state.plic_face_flux_r.data(),
        d_volfrac_mat,
        state.plic_reconstruction_valid.data(),
        state.plic_normal_r.data(),
        state.plic_normal_z.data(),
        state.plic_alpha.data(),
        d_vol_old,
        d_xr_old,
        d_xz_old,
        d_xr_new,
        d_xz_new,
        d_metrics,
        nr,
        nz,
        n_mat,
        cfg.numerics.plic.fast_path_threshold_min,
        cfg.numerics.plic.fast_path_threshold_max);
  }
  CUDA_CHECK(cudaGetLastError());
  if (effective_donor_sign_fixed) {
    compute_z_face_flux_kernel<true><<<blocks_z, 256, 0, stream>>>(
        state.plic_face_flux_z.data(),
        d_volfrac_mat,
        state.plic_reconstruction_valid.data(),
        state.plic_normal_r.data(),
        state.plic_normal_z.data(),
        state.plic_alpha.data(),
        d_vol_old,
        d_xr_old,
        d_xz_old,
        d_xr_new,
        d_xz_new,
        d_metrics,
        nr,
        nz,
        n_mat,
        cfg.numerics.plic.fast_path_threshold_min,
        cfg.numerics.plic.fast_path_threshold_max);
  } else {
    compute_z_face_flux_kernel<false><<<blocks_z, 256, 0, stream>>>(
        state.plic_face_flux_z.data(),
        d_volfrac_mat,
        state.plic_reconstruction_valid.data(),
        state.plic_normal_r.data(),
        state.plic_normal_z.data(),
        state.plic_alpha.data(),
        d_vol_old,
        d_xr_old,
        d_xz_old,
        d_xr_new,
        d_xz_new,
        d_metrics,
        nr,
        nz,
        n_mat,
        cfg.numerics.plic.fast_path_threshold_min,
        cfg.numerics.plic.fast_path_threshold_max);
  }
  CUDA_CHECK(cudaGetLastError());
  const bool cf6_active = cfg.numerics.plic.rho_material_aware_donor &&
                          !cfg.numerics.materials.per_material_conservation_enabled;
  if (cf6_active) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rho_cf6_old),
                          static_cast<std::size_t>(n_cells) * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(d_rho_cf6_old,
                               state.rho.data(),
                               static_cast<std::size_t>(n_cells) * sizeof(double),
                               cudaMemcpyDeviceToDevice,
                               stream));
    const Cf6MaterialParams cf6_params = make_cf6_material_params(cfg, n_mat);
    const tenryu::materials::DeviceEOSTableView* ion_views =
        (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_ion_views
                                                              : nullptr;
    const tenryu::materials::DeviceEOSTableView* electron_views =
        (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat)
            ? eos_ctx->d_electron_views
            : nullptr;
    const tenryu::materials::DeviceEOSTableView* total_views =
        (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_total_views
                                                              : nullptr;
    remap_density_cf6_kernel<<<blocks_cells, 256, 0, stream>>>(
        state.rho.data(),
        d_rho_cf6_old,
        d_volfrac_mat,
        state.plic_face_flux_r.data(),
        state.plic_face_flux_z.data(),
        d_vol_old,
        d_vol_new,
        state.Te.data(),
        state.Ti.data(),
        state.Pe.data(),
        state.Pi.data(),
        ion_views,
        electron_views,
        total_views,
        cf6_params,
        d_counts,
        nr,
        nz,
        n_mat,
        cfg.numerics.plic.fast_path_threshold_min,
        cfg.numerics.plic.fast_path_threshold_max);
    CUDA_CHECK(cudaGetLastError());
    status.cf6_density_remap_used = true;
  }
  gather_material_volume_kernel<<<blocks_cm, 256, 0, stream>>>(
      d_volfrac_mat,
      state.plic_face_flux_r.data(),
      state.plic_face_flux_z.data(),
      d_vol_old,
      d_vol_new,
      d_counts,
      nr,
      nz,
      n_mat);
  CUDA_CHECK(cudaGetLastError());
  snap_material_residual_kernel<<<blocks_cells, 256, 0, stream>>>(
      d_volfrac_mat,
      state.plic_cell_residual.data(),
      d_counts,
      d_metrics,
      n_cells,
      n_mat);
  CUDA_CHECK(cudaGetLastError());
  PlicRemapStatus remap_status = status_from_device(d_counts, d_metrics, stream);
  if (d_rho_cf6_old != nullptr) {
    CUDA_CHECK(cudaFree(d_rho_cf6_old));
  }
  CUDA_CHECK(cudaFree(d_metrics));
  CUDA_CHECK(cudaFree(d_counts));

  status.repair_events += remap_status.repair_events;
  status.max_volume_fraction_residual =
      std::max(status.max_volume_fraction_residual,
               remap_status.max_volume_fraction_residual);
  status.max_swept_fraction = remap_status.max_swept_fraction;
  status.drift_triggered =
      status.max_interface_centroid_drift_relative >
          cfg.numerics.plic.drift_sensor_max_relative ||
      status.max_swept_fraction > cfg.numerics.plic.drift_sensor_max_swept_fraction;
  update_observability(status, observability);
  return status;
}

void apply_plic_fallback_policy(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const PlicRemapStatus& status,
    tenryu::coupling::ProfileObservability* observability) {
  if (!cfg.numerics.plic.enabled || cfg.numerics.plic.in_run_disabled) {
    return;
  }
  if (status.drift_triggered) {
    ++state.plic_consecutive_drift_triggers;
  } else {
    state.plic_consecutive_drift_triggers = 0;
  }
  if (state.plic_consecutive_drift_triggers > 5 &&
      !state.plic_remap_sticky_fallback) {
    state.plic_remap_sticky_fallback = true;
    tenryu::core::log_warning(
        "[plic_remap] sticky scalar fallback engaged after repeated PLIC drift "
        "sensor triggers");
    if (observability != nullptr) {
      observability->plic_remap_fallback_engaged = true;
      observability->class_d_runtime_fires_matrix[2][1] += 1;
      tenryu::core::log_warning(
          "[plic_degradation] PLIC remap scalar fallback: "
          "repeated_interface_centroid_drift");
    }
  }
  if (observability != nullptr && state.plic_remap_sticky_fallback) {
    observability->plic_remap_fallback_engaged = true;
  }
}

}  // namespace tenryu::hydro::plic
