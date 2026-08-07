#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace tenryu::hydro::ale::remap_repair {

#if defined(__CUDACC__)
#define TENRYU_REMAP_REPAIR_HD __host__ __device__
#else
#define TENRYU_REMAP_REPAIR_HD
#endif

/**
 * Conservative bound-preserving remap repair (FCT).
 *
 * This module is intentionally cfg-free and geometry-free.  The RZ swept
 * geometry has already entered the caller's face fluxes through
 * dV = 2*pi*r*dr*dz integrals and any exact RZ moments.  All quantities here are
 * conservative cell-integrated amounts, or densities with a consistent volume
 * convention chosen by the caller.  No unit conversion is performed; TENRYU's
 * cgs+eV convention is preserved by passing same-unit bounds and floors.
 *
 * For each shared face f, let F_f^L be the low-order donor flux and F_f^H the
 * high-order limited flux, both with the same sign convention: positive flux is
 * added to face_cell_a[f] and subtracted from face_cell_b[f].  The caller forms
 *
 *   A_f = F_f^H - F_f^L .
 *
 * Starting from a bound-preserving low-order state U_c^L, the repaired state is
 *
 *   U_c^{n+1} = U_c^L + sum_f s_cf theta_f A_f ,
 *
 * where s_cf is +1 for face_cell_a, -1 for face_cell_b, and theta_f in [0,1].
 * Zalesak's limiter computes per-cell antidiffusive budgets
 *
 *   P_c^+ = sum max(s_cf A_f, 0),     P_c^- = sum min(s_cf A_f, 0),
 *   Q_c^+ = U_c^max - U_c^L,          Q_c^- = U_c^min - U_c^L,
 *
 * then
 *
 *   R_c^+ = min(1, Q_c^+ / P_c^+) when P_c^+ > 0, else 1,
 *   R_c^- = min(1, Q_c^- / P_c^-) when P_c^- < 0, else 1.
 *
 * The shared face coefficient is the same for both cells:
 *
 *   theta_f = min(R_a^sign(A_f), R_b^sign(-A_f)).
 *
 * The single symmetric theta_f preserves pairwise conservation because the
 * accepted antidiffusive correction is added to one cell and subtracted from the
 * other.  A remap loop wires this module by:
 *
 *   1. computing and applying F^L into U_low;
 *   2. computing A_f = F^H - F^L for each conserved field;
 *   3. initializing theta_f = 1;
 *   4. limiting theta by scalar bounds for mass, material fractions, etc.;
 *   5. limiting theta by the coupled rho/momentum/total-energy guard if used;
 *   6. applying theta_f A_f to U_low with equal-and-opposite face updates.
 *
 * Volume fractions use the same theta_f for every material component.  If the
 * low-order alpha sums to one per cell and the per-face alpha corrections sum to
 * zero over materials, the common theta keeps sum(alpha)=1 while the scalar
 * bounds keep each alpha in [0,1].
 */

static constexpr double kTinyPositive = 1.0e-300;
static constexpr double kHugeBound = 1.0e300;

TENRYU_REMAP_REPAIR_HD inline bool finite_value(const double x) {
#if defined(__CUDA_ARCH__)
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

TENRYU_REMAP_REPAIR_HD inline double min_value(const double a,
                                               const double b) {
#if defined(__CUDA_ARCH__)
  return fmin(a, b);
#else
  return std::fmin(a, b);
#endif
}

TENRYU_REMAP_REPAIR_HD inline double max_value(const double a,
                                               const double b) {
#if defined(__CUDA_ARCH__)
  return fmax(a, b);
#else
  return std::fmax(a, b);
#endif
}

TENRYU_REMAP_REPAIR_HD inline double abs_value(const double x) {
#if defined(__CUDA_ARCH__)
  return fabs(x);
#else
  return std::fabs(x);
#endif
}

TENRYU_REMAP_REPAIR_HD inline double sqrt_value(const double x) {
#if defined(__CUDA_ARCH__)
  return sqrt(x);
#else
  return std::sqrt(x);
#endif
}

TENRYU_REMAP_REPAIR_HD inline double clamp01(const double x) {
  if (!finite_value(x)) {
    return 0.0;
  }
  return min_value(1.0, max_value(0.0, x));
}

TENRYU_REMAP_REPAIR_HD inline bool valid_cell_index(const int c,
                                                    const int n_cells) {
  return c >= 0 && c < n_cells;
}

TENRYU_REMAP_REPAIR_HD inline double zalesak_positive_ratio(
    const double q_plus,
    const double p_plus) {
  if (!finite_value(p_plus) || !finite_value(q_plus)) {
    return 0.0;
  }
  if (!(p_plus > 0.0)) {
    return 1.0;
  }
  if (!(q_plus > 0.0)) {
    return 0.0;
  }
  return clamp01(q_plus / p_plus);
}

TENRYU_REMAP_REPAIR_HD inline double zalesak_negative_ratio(
    const double q_minus,
    const double p_minus) {
  if (!finite_value(p_minus) || !finite_value(q_minus)) {
    return 0.0;
  }
  if (!(p_minus < 0.0)) {
    return 1.0;
  }
  if (!(q_minus < 0.0)) {
    return 0.0;
  }
  return clamp01(q_minus / p_minus);
}

TENRYU_REMAP_REPAIR_HD inline double bounded_floor_value(
    const double* values,
    const double scalar_value,
    const int c) {
  const double v = (values != nullptr) ? values[c] : scalar_value;
  return (v > 0.0 && finite_value(v)) ? v : kTinyPositive;
}

TENRYU_REMAP_REPAIR_HD inline double internal_energy_amount(
    const double rho,
    const double mom_r,
    const double mom_z,
    const double total_energy) {
  if (!(rho > 0.0) || !finite_value(rho) || !finite_value(mom_r) ||
      !finite_value(mom_z) || !finite_value(total_energy)) {
    return -kHugeBound;
  }
  return total_energy - 0.5 * (mom_r * mom_r + mom_z * mom_z) / rho;
}

TENRYU_REMAP_REPAIR_HD inline double coupled_state_budget_margin(
    const double rho_low,
    const double mom_abs_low,
    const double total_energy_low,
    const double rho_negative_sum,
    const double momentum_abs_sum,
    const double total_energy_negative_sum,
    const double rho_floor,
    const double internal_energy_floor,
    const double theta) {
  const double rho_bound =
      max_value(rho_floor, rho_low + theta * rho_negative_sum);
  if (!(rho_bound > 0.0) || !finite_value(rho_bound)) {
    return -kHugeBound;
  }
  const double p_bound = mom_abs_low + theta * momentum_abs_sum;
  const double E_bound = total_energy_low + theta * total_energy_negative_sum;
  return E_bound - 0.5 * p_bound * p_bound / rho_bound -
         internal_energy_floor;
}

TENRYU_REMAP_REPAIR_HD inline double coupled_state_budget_guard_ratio(
    const double rho_low,
    const double mom_r_low,
    const double mom_z_low,
    const double total_energy_low,
    const double rho_negative_sum,
    const double momentum_abs_sum,
    const double total_energy_negative_sum,
    const double rho_floor,
    const double internal_energy_floor) {
  if (!(rho_low > 0.0) || !finite_value(rho_low) ||
      !finite_value(mom_r_low) || !finite_value(mom_z_low) ||
      !finite_value(total_energy_low)) {
    return 0.0;
  }

  const double rho_min =
      (rho_floor > 0.0 && finite_value(rho_floor)) ? rho_floor : kTinyPositive;
  const double e_floor = (internal_energy_floor > 0.0 &&
                          finite_value(internal_energy_floor))
                             ? internal_energy_floor
                             : kTinyPositive;
  const double e_low =
      internal_energy_amount(rho_low, mom_r_low, mom_z_low, total_energy_low);
  if (!(e_low > e_floor) || !finite_value(e_low)) {
    return 0.0;
  }

  double rho_ratio = 1.0;
  const double d_rho_neg =
      (rho_negative_sum < 0.0 && finite_value(rho_negative_sum))
          ? rho_negative_sum
          : 0.0;
  if (rho_low + d_rho_neg < rho_min && d_rho_neg < 0.0) {
    rho_ratio = clamp01((rho_min - rho_low) / d_rho_neg);
  }

  const double p0 =
      sqrt_value(mom_r_low * mom_r_low + mom_z_low * mom_z_low);
  const double dp_abs =
      (momentum_abs_sum > 0.0 && finite_value(momentum_abs_sum))
          ? momentum_abs_sum
          : 0.0;
  const double dE_neg =
      (total_energy_negative_sum < 0.0 &&
       finite_value(total_energy_negative_sum))
          ? total_energy_negative_sum
          : 0.0;

  const double hi0 = rho_ratio;
  if (!(hi0 > 0.0)) {
    return 0.0;
  }
  if (coupled_state_budget_margin(rho_low,
                                  p0,
                                  total_energy_low,
                                  d_rho_neg,
                                  dp_abs,
                                  dE_neg,
                                  rho_min,
                                  e_floor,
                                  hi0) >= 0.0) {
    return clamp01(hi0);
  }
  if (coupled_state_budget_margin(rho_low,
                                  p0,
                                  total_energy_low,
                                  d_rho_neg,
                                  dp_abs,
                                  dE_neg,
                                  rho_min,
                                  e_floor,
                                  0.0) < 0.0) {
    return 0.0;
  }

  double lo = 0.0;
  double hi = hi0;
  for (int iter = 0; iter < 48; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (coupled_state_budget_margin(rho_low,
                                    p0,
                                    total_energy_low,
                                    d_rho_neg,
                                    dp_abs,
                                    dE_neg,
                                    rho_min,
                                    e_floor,
                                    mid) >= 0.0) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return clamp01(lo);
}

TENRYU_REMAP_REPAIR_HD inline double volume_fraction_sum_residual(
    const double* alpha,
    const int cell,
    const int n_materials,
    const int stride) {
  if (alpha == nullptr || cell < 0 || n_materials <= 0 || stride <= 0) {
    return kHugeBound;
  }
  double sum = 0.0;
  for (int m = 0; m < n_materials; ++m) {
    const double a = alpha[m * stride + cell];
    if (!finite_value(a)) {
      return kHugeBound;
    }
    sum += a;
  }
  return sum - 1.0;
}

TENRYU_REMAP_REPAIR_HD inline bool volume_fraction_unit_sum_admissible(
    const double* alpha,
    const int cell,
    const int n_materials,
    const int stride,
    const double tolerance) {
  if (alpha == nullptr || cell < 0 || n_materials <= 0 || stride <= 0) {
    return false;
  }
  for (int m = 0; m < n_materials; ++m) {
    const double a = alpha[m * stride + cell];
    if (!finite_value(a) || a < -tolerance || a > 1.0 + tolerance) {
      return false;
    }
  }
  return abs_value(volume_fraction_sum_residual(alpha, cell, n_materials, stride)) <=
         tolerance;
}

#if defined(__CUDACC__)

namespace detail {

static constexpr int kThreadsPerBlock = 256;

inline int blocks_for(const int n) {
  return (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
}

__device__ inline void accumulate_cell_scalar_budget(
    double* __restrict__ positive_sum,
    double* __restrict__ negative_sum,
    const int c,
    const int n_cells,
    const double delta) {
  if (!valid_cell_index(c, n_cells) || !finite_value(delta)) {
    return;
  }
  if (delta > 0.0) {
    atomicAdd(positive_sum + c, delta);
  } else if (delta < 0.0) {
    atomicAdd(negative_sum + c, delta);
  }
}

__device__ inline void accumulate_cell_coupled_budget(
    double* __restrict__ rho_negative_sum,
    double* __restrict__ momentum_abs_sum,
    double* __restrict__ total_energy_negative_sum,
    const int c,
    const int n_cells,
    const double d_rho,
    const double d_mom_r,
    const double d_mom_z,
    const double d_total_energy) {
  if (!valid_cell_index(c, n_cells) || !finite_value(d_rho) ||
      !finite_value(d_mom_r) || !finite_value(d_mom_z) ||
      !finite_value(d_total_energy)) {
    return;
  }
  if (rho_negative_sum != nullptr && d_rho < 0.0) {
    atomicAdd(rho_negative_sum + c, d_rho);
  }
  if (momentum_abs_sum != nullptr) {
    const double dp = sqrt_value(d_mom_r * d_mom_r + d_mom_z * d_mom_z);
    if (dp > 0.0 && finite_value(dp)) {
      atomicAdd(momentum_abs_sum + c, dp);
    }
  }
  if (total_energy_negative_sum != nullptr && d_total_energy < 0.0) {
    atomicAdd(total_energy_negative_sum + c, d_total_energy);
  }
}

}  // namespace detail

__global__ void conservative_remap_repair_fill_kernel(double* values,
                                                      const double value,
                                                      const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    values[i] = value;
  }
}

__global__ void conservative_remap_repair_compute_antidiffusive_flux_kernel(
    double* __restrict__ antidiffusive_flux_to_a,
    const double* __restrict__ high_order_flux_to_a,
    const double* __restrict__ low_order_flux_to_a,
    const int n_faces) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const double hi = high_order_flux_to_a[f];
  const double lo = low_order_flux_to_a[f];
  antidiffusive_flux_to_a[f] =
      (finite_value(hi) && finite_value(lo)) ? (hi - lo) : 0.0;
}

__global__ void conservative_remap_repair_accumulate_scalar_sums_kernel(
    double* __restrict__ positive_sum,
    double* __restrict__ negative_sum,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const double* __restrict__ antidiffusive_flux_to_a,
    const int n_faces,
    const int n_cells) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const double A = antidiffusive_flux_to_a[f];
  if (!finite_value(A)) {
    return;
  }
  const int cell_a = face_cell_a[f];
  const int cell_b = (face_cell_b != nullptr) ? face_cell_b[f] : -1;
  detail::accumulate_cell_scalar_budget(
      positive_sum, negative_sum, cell_a, n_cells, A);
  detail::accumulate_cell_scalar_budget(
      positive_sum, negative_sum, cell_b, n_cells, -A);
}

__global__ void conservative_remap_repair_compute_scalar_ratios_kernel(
    double* __restrict__ positive_ratio,
    double* __restrict__ negative_ratio,
    const double* __restrict__ low_state,
    const double* __restrict__ lower_bound,
    const double* __restrict__ upper_bound,
    const double* __restrict__ positive_sum,
    const double* __restrict__ negative_sum,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double u = low_state[c];
  const double lo = (lower_bound != nullptr) ? lower_bound[c] : -kHugeBound;
  const double hi = (upper_bound != nullptr) ? upper_bound[c] : kHugeBound;
  if (!finite_value(u) || !finite_value(lo) || !finite_value(hi) || lo > hi) {
    positive_ratio[c] = 0.0;
    negative_ratio[c] = 0.0;
    return;
  }
  positive_ratio[c] =
      zalesak_positive_ratio(hi - u, positive_sum[c]);
  negative_ratio[c] =
      zalesak_negative_ratio(lo - u, negative_sum[c]);
}

__global__ void conservative_remap_repair_compute_scalar_uniform_ratios_kernel(
    double* __restrict__ positive_ratio,
    double* __restrict__ negative_ratio,
    const double* __restrict__ low_state,
    const double lower_bound,
    const double upper_bound,
    const double* __restrict__ positive_sum,
    const double* __restrict__ negative_sum,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double u = low_state[c];
  if (!finite_value(u) || !finite_value(lower_bound) ||
      !finite_value(upper_bound) || lower_bound > upper_bound) {
    positive_ratio[c] = 0.0;
    negative_ratio[c] = 0.0;
    return;
  }
  positive_ratio[c] =
      zalesak_positive_ratio(upper_bound - u, positive_sum[c]);
  negative_ratio[c] =
      zalesak_negative_ratio(lower_bound - u, negative_sum[c]);
}

__global__ void conservative_remap_repair_limit_theta_by_scalar_kernel(
    double* __restrict__ theta,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const double* __restrict__ antidiffusive_flux_to_a,
    const double* __restrict__ positive_ratio,
    const double* __restrict__ negative_ratio,
    const int n_faces,
    const int n_cells) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double limited = clamp01(theta[f]);
  const double A = antidiffusive_flux_to_a[f];
  if (!finite_value(A)) {
    theta[f] = 0.0;
    return;
  }
  const int cell_a = face_cell_a[f];
  const int cell_b = (face_cell_b != nullptr) ? face_cell_b[f] : -1;
  if (valid_cell_index(cell_a, n_cells)) {
    limited = min_value(
        limited, (A >= 0.0) ? positive_ratio[cell_a] : negative_ratio[cell_a]);
  }
  if (valid_cell_index(cell_b, n_cells)) {
    limited = min_value(
        limited, (-A >= 0.0) ? positive_ratio[cell_b] : negative_ratio[cell_b]);
  }
  theta[f] = clamp01(limited);
}

__global__ void conservative_remap_repair_copy_cell_values_kernel(
    double* __restrict__ dst,
    const double* __restrict__ src,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n_cells) {
    dst[c] = src[c];
  }
}

__global__ void conservative_remap_repair_apply_scalar_kernel(
    double* __restrict__ repaired_state,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const double* __restrict__ antidiffusive_flux_to_a,
    const double* __restrict__ theta,
    const int n_faces,
    const int n_cells) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const double A = antidiffusive_flux_to_a[f];
  const double t = clamp01(theta[f]);
  if (!finite_value(A) || !finite_value(t)) {
    return;
  }
  const double dU_a = t * A;
  const int cell_a = face_cell_a[f];
  const int cell_b = (face_cell_b != nullptr) ? face_cell_b[f] : -1;
  if (valid_cell_index(cell_a, n_cells)) {
    atomicAdd(repaired_state + cell_a, dU_a);
  }
  if (valid_cell_index(cell_b, n_cells)) {
    atomicAdd(repaired_state + cell_b, -dU_a);
  }
}

__global__ void conservative_remap_repair_accumulate_coupled_sums_kernel(
    double* __restrict__ rho_negative_sum,
    double* __restrict__ momentum_abs_sum,
    double* __restrict__ total_energy_negative_sum,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const double* __restrict__ rho_antidiffusive_flux_to_a,
    const double* __restrict__ mom_r_antidiffusive_flux_to_a,
    const double* __restrict__ mom_z_antidiffusive_flux_to_a,
    const double* __restrict__ total_energy_antidiffusive_flux_to_a,
    const int n_faces,
    const int n_cells) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  const double d_rho = rho_antidiffusive_flux_to_a[f];
  const double d_mom_r =
      (mom_r_antidiffusive_flux_to_a != nullptr)
          ? mom_r_antidiffusive_flux_to_a[f]
          : 0.0;
  const double d_mom_z =
      (mom_z_antidiffusive_flux_to_a != nullptr)
          ? mom_z_antidiffusive_flux_to_a[f]
          : 0.0;
  const double d_E = total_energy_antidiffusive_flux_to_a[f];
  const int cell_a = face_cell_a[f];
  const int cell_b = (face_cell_b != nullptr) ? face_cell_b[f] : -1;
  detail::accumulate_cell_coupled_budget(rho_negative_sum,
                                         momentum_abs_sum,
                                         total_energy_negative_sum,
                                         cell_a,
                                         n_cells,
                                         d_rho,
                                         d_mom_r,
                                         d_mom_z,
                                         d_E);
  detail::accumulate_cell_coupled_budget(rho_negative_sum,
                                         momentum_abs_sum,
                                         total_energy_negative_sum,
                                         cell_b,
                                         n_cells,
                                         -d_rho,
                                         -d_mom_r,
                                         -d_mom_z,
                                         -d_E);
}

__global__ void conservative_remap_repair_compute_coupled_guard_kernel(
    double* __restrict__ cell_guard,
    const double* __restrict__ rho_low,
    const double* __restrict__ mom_r_low,
    const double* __restrict__ mom_z_low,
    const double* __restrict__ total_energy_low,
    const double* __restrict__ rho_negative_sum,
    const double* __restrict__ momentum_abs_sum,
    const double* __restrict__ total_energy_negative_sum,
    const double* __restrict__ rho_floor,
    const double rho_floor_scalar,
    const double* __restrict__ internal_energy_floor,
    const double internal_energy_floor_scalar,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double rho_min = bounded_floor_value(rho_floor, rho_floor_scalar, c);
  const double e_min =
      bounded_floor_value(internal_energy_floor, internal_energy_floor_scalar, c);
  cell_guard[c] =
      coupled_state_budget_guard_ratio(rho_low[c],
                                       mom_r_low[c],
                                       mom_z_low[c],
                                       total_energy_low[c],
                                       rho_negative_sum[c],
                                       momentum_abs_sum[c],
                                       total_energy_negative_sum[c],
                                       rho_min,
                                       e_min);
}

__global__ void conservative_remap_repair_limit_theta_by_cell_guard_kernel(
    double* __restrict__ theta,
    const int* __restrict__ face_cell_a,
    const int* __restrict__ face_cell_b,
    const double* __restrict__ cell_guard,
    const int n_faces,
    const int n_cells) {
  const int f = blockIdx.x * blockDim.x + threadIdx.x;
  if (f >= n_faces) {
    return;
  }
  double limited = clamp01(theta[f]);
  const int cell_a = face_cell_a[f];
  const int cell_b = (face_cell_b != nullptr) ? face_cell_b[f] : -1;
  if (valid_cell_index(cell_a, n_cells)) {
    limited = min_value(limited, cell_guard[cell_a]);
  }
  if (valid_cell_index(cell_b, n_cells)) {
    limited = min_value(limited, cell_guard[cell_b]);
  }
  theta[f] = clamp01(limited);
}

inline void launch_fill(double* values,
                        const double value,
                        const int n,
                        cudaStream_t stream = nullptr) {
  if (n <= 0) {
    return;
  }
  conservative_remap_repair_fill_kernel<<<detail::blocks_for(n),
                                          detail::kThreadsPerBlock,
                                          0,
                                          stream>>>(values, value, n);
}

inline void launch_compute_antidiffusive_flux(
    double* antidiffusive_flux_to_a,
    const double* high_order_flux_to_a,
    const double* low_order_flux_to_a,
    const int n_faces,
    cudaStream_t stream = nullptr) {
  if (n_faces <= 0) {
    return;
  }
  conservative_remap_repair_compute_antidiffusive_flux_kernel<<<
      detail::blocks_for(n_faces),
      detail::kThreadsPerBlock,
      0,
      stream>>>(antidiffusive_flux_to_a,
                high_order_flux_to_a,
                low_order_flux_to_a,
                n_faces);
}

inline void launch_accumulate_scalar_sums(
    double* positive_sum,
    double* negative_sum,
    const int* face_cell_a,
    const int* face_cell_b,
    const double* antidiffusive_flux_to_a,
    const int n_faces,
    const int n_cells,
    cudaStream_t stream = nullptr) {
  if (n_faces <= 0) {
    return;
  }
  conservative_remap_repair_accumulate_scalar_sums_kernel<<<
      detail::blocks_for(n_faces),
      detail::kThreadsPerBlock,
      0,
      stream>>>(positive_sum,
                negative_sum,
                face_cell_a,
                face_cell_b,
                antidiffusive_flux_to_a,
                n_faces,
                n_cells);
}

inline void launch_compute_scalar_ratios(
    double* positive_ratio,
    double* negative_ratio,
    const double* low_state,
    const double* lower_bound,
    const double* upper_bound,
    const double* positive_sum,
    const double* negative_sum,
    const int n_cells,
    cudaStream_t stream = nullptr) {
  if (n_cells <= 0) {
    return;
  }
  conservative_remap_repair_compute_scalar_ratios_kernel<<<
      detail::blocks_for(n_cells),
      detail::kThreadsPerBlock,
      0,
      stream>>>(positive_ratio,
                negative_ratio,
                low_state,
                lower_bound,
                upper_bound,
                positive_sum,
                negative_sum,
                n_cells);
}

inline void launch_compute_scalar_uniform_ratios(
    double* positive_ratio,
    double* negative_ratio,
    const double* low_state,
    const double lower_bound,
    const double upper_bound,
    const double* positive_sum,
    const double* negative_sum,
    const int n_cells,
    cudaStream_t stream = nullptr) {
  if (n_cells <= 0) {
    return;
  }
  conservative_remap_repair_compute_scalar_uniform_ratios_kernel<<<
      detail::blocks_for(n_cells),
      detail::kThreadsPerBlock,
      0,
      stream>>>(positive_ratio,
                negative_ratio,
                low_state,
                lower_bound,
                upper_bound,
                positive_sum,
                negative_sum,
                n_cells);
}

inline void launch_limit_theta_by_scalar(
    double* theta,
    const int* face_cell_a,
    const int* face_cell_b,
    const double* antidiffusive_flux_to_a,
    const double* positive_ratio,
    const double* negative_ratio,
    const int n_faces,
    const int n_cells,
    cudaStream_t stream = nullptr) {
  if (n_faces <= 0) {
    return;
  }
  conservative_remap_repair_limit_theta_by_scalar_kernel<<<
      detail::blocks_for(n_faces),
      detail::kThreadsPerBlock,
      0,
      stream>>>(theta,
                face_cell_a,
                face_cell_b,
                antidiffusive_flux_to_a,
                positive_ratio,
                negative_ratio,
                n_faces,
                n_cells);
}

inline void launch_limit_theta_by_scalar_bounds(
    double* theta,
    const double* low_state,
    const double* lower_bound,
    const double* upper_bound,
    const int* face_cell_a,
    const int* face_cell_b,
    const double* antidiffusive_flux_to_a,
    double* positive_sum,
    double* negative_sum,
    double* positive_ratio,
    double* negative_ratio,
    const int n_cells,
    const int n_faces,
    cudaStream_t stream = nullptr) {
  launch_fill(positive_sum, 0.0, n_cells, stream);
  launch_fill(negative_sum, 0.0, n_cells, stream);
  launch_accumulate_scalar_sums(positive_sum,
                                negative_sum,
                                face_cell_a,
                                face_cell_b,
                                antidiffusive_flux_to_a,
                                n_faces,
                                n_cells,
                                stream);
  launch_compute_scalar_ratios(positive_ratio,
                               negative_ratio,
                               low_state,
                               lower_bound,
                               upper_bound,
                               positive_sum,
                               negative_sum,
                               n_cells,
                               stream);
  launch_limit_theta_by_scalar(theta,
                               face_cell_a,
                               face_cell_b,
                               antidiffusive_flux_to_a,
                               positive_ratio,
                               negative_ratio,
                               n_faces,
                               n_cells,
                               stream);
}

inline void launch_limit_theta_by_uniform_bounds(
    double* theta,
    const double* low_state,
    const double lower_bound,
    const double upper_bound,
    const int* face_cell_a,
    const int* face_cell_b,
    const double* antidiffusive_flux_to_a,
    double* positive_sum,
    double* negative_sum,
    double* positive_ratio,
    double* negative_ratio,
    const int n_cells,
    const int n_faces,
    cudaStream_t stream = nullptr) {
  launch_fill(positive_sum, 0.0, n_cells, stream);
  launch_fill(negative_sum, 0.0, n_cells, stream);
  launch_accumulate_scalar_sums(positive_sum,
                                negative_sum,
                                face_cell_a,
                                face_cell_b,
                                antidiffusive_flux_to_a,
                                n_faces,
                                n_cells,
                                stream);
  launch_compute_scalar_uniform_ratios(positive_ratio,
                                       negative_ratio,
                                       low_state,
                                       lower_bound,
                                       upper_bound,
                                       positive_sum,
                                       negative_sum,
                                       n_cells,
                                       stream);
  launch_limit_theta_by_scalar(theta,
                               face_cell_a,
                               face_cell_b,
                               antidiffusive_flux_to_a,
                               positive_ratio,
                               negative_ratio,
                               n_faces,
                               n_cells,
                               stream);
}

inline void launch_limit_theta_by_volume_fraction_bounds(
    double* theta,
    const double* alpha_low,
    const double* alpha_antidiffusive_flux_to_a,
    const int n_materials,
    const int cell_stride,
    const int face_stride,
    const int* face_cell_a,
    const int* face_cell_b,
    double* positive_sum,
    double* negative_sum,
    double* positive_ratio,
    double* negative_ratio,
    const int n_cells,
    const int n_faces,
    cudaStream_t stream = nullptr) {
  if (n_materials <= 0 || cell_stride <= 0 || face_stride <= 0) {
    return;
  }
  for (int m = 0; m < n_materials; ++m) {
    launch_limit_theta_by_uniform_bounds(
        theta,
        alpha_low + m * cell_stride,
        0.0,
        1.0,
        face_cell_a,
        face_cell_b,
        alpha_antidiffusive_flux_to_a + m * face_stride,
        positive_sum,
        negative_sum,
        positive_ratio,
        negative_ratio,
        n_cells,
        n_faces,
        stream);
  }
}

inline void launch_copy_cell_values(double* dst,
                                    const double* src,
                                    const int n_cells,
                                    cudaStream_t stream = nullptr) {
  if (n_cells <= 0) {
    return;
  }
  conservative_remap_repair_copy_cell_values_kernel<<<
      detail::blocks_for(n_cells),
      detail::kThreadsPerBlock,
      0,
      stream>>>(dst, src, n_cells);
}

inline void launch_apply_scalar(double* repaired_state,
                                const int* face_cell_a,
                                const int* face_cell_b,
                                const double* antidiffusive_flux_to_a,
                                const double* theta,
                                const int n_faces,
                                const int n_cells,
                                cudaStream_t stream = nullptr) {
  if (n_faces <= 0) {
    return;
  }
  conservative_remap_repair_apply_scalar_kernel<<<detail::blocks_for(n_faces),
                                                  detail::kThreadsPerBlock,
                                                  0,
                                                  stream>>>(
      repaired_state,
      face_cell_a,
      face_cell_b,
      antidiffusive_flux_to_a,
      theta,
      n_faces,
      n_cells);
}

inline void launch_accumulate_coupled_sums(
    double* rho_negative_sum,
    double* momentum_abs_sum,
    double* total_energy_negative_sum,
    const int* face_cell_a,
    const int* face_cell_b,
    const double* rho_antidiffusive_flux_to_a,
    const double* mom_r_antidiffusive_flux_to_a,
    const double* mom_z_antidiffusive_flux_to_a,
    const double* total_energy_antidiffusive_flux_to_a,
    const int n_faces,
    const int n_cells,
    cudaStream_t stream = nullptr) {
  if (n_faces <= 0) {
    return;
  }
  conservative_remap_repair_accumulate_coupled_sums_kernel<<<
      detail::blocks_for(n_faces),
      detail::kThreadsPerBlock,
      0,
      stream>>>(rho_negative_sum,
                momentum_abs_sum,
                total_energy_negative_sum,
                face_cell_a,
                face_cell_b,
                rho_antidiffusive_flux_to_a,
                mom_r_antidiffusive_flux_to_a,
                mom_z_antidiffusive_flux_to_a,
                total_energy_antidiffusive_flux_to_a,
                n_faces,
                n_cells);
}

inline void launch_compute_coupled_guard(
    double* cell_guard,
    const double* rho_low,
    const double* mom_r_low,
    const double* mom_z_low,
    const double* total_energy_low,
    const double* rho_negative_sum,
    const double* momentum_abs_sum,
    const double* total_energy_negative_sum,
    const double* rho_floor,
    const double rho_floor_scalar,
    const double* internal_energy_floor,
    const double internal_energy_floor_scalar,
    const int n_cells,
    cudaStream_t stream = nullptr) {
  if (n_cells <= 0) {
    return;
  }
  conservative_remap_repair_compute_coupled_guard_kernel<<<
      detail::blocks_for(n_cells),
      detail::kThreadsPerBlock,
      0,
      stream>>>(cell_guard,
                rho_low,
                mom_r_low,
                mom_z_low,
                total_energy_low,
                rho_negative_sum,
                momentum_abs_sum,
                total_energy_negative_sum,
                rho_floor,
                rho_floor_scalar,
                internal_energy_floor,
                internal_energy_floor_scalar,
                n_cells);
}

inline void launch_limit_theta_by_cell_guard(
    double* theta,
    const int* face_cell_a,
    const int* face_cell_b,
    const double* cell_guard,
    const int n_faces,
    const int n_cells,
    cudaStream_t stream = nullptr) {
  if (n_faces <= 0) {
    return;
  }
  conservative_remap_repair_limit_theta_by_cell_guard_kernel<<<
      detail::blocks_for(n_faces),
      detail::kThreadsPerBlock,
      0,
      stream>>>(theta, face_cell_a, face_cell_b, cell_guard, n_faces, n_cells);
}

inline void launch_limit_theta_by_coupled_positivity(
    double* theta,
    const double* rho_low,
    const double* mom_r_low,
    const double* mom_z_low,
    const double* total_energy_low,
    const int* face_cell_a,
    const int* face_cell_b,
    const double* rho_antidiffusive_flux_to_a,
    const double* mom_r_antidiffusive_flux_to_a,
    const double* mom_z_antidiffusive_flux_to_a,
    const double* total_energy_antidiffusive_flux_to_a,
    double* rho_negative_sum,
    double* momentum_abs_sum,
    double* total_energy_negative_sum,
    double* cell_guard,
    const double* rho_floor,
    const double rho_floor_scalar,
    const double* internal_energy_floor,
    const double internal_energy_floor_scalar,
    const int n_cells,
    const int n_faces,
    cudaStream_t stream = nullptr) {
  launch_fill(rho_negative_sum, 0.0, n_cells, stream);
  launch_fill(momentum_abs_sum, 0.0, n_cells, stream);
  launch_fill(total_energy_negative_sum, 0.0, n_cells, stream);
  launch_accumulate_coupled_sums(rho_negative_sum,
                                 momentum_abs_sum,
                                 total_energy_negative_sum,
                                 face_cell_a,
                                 face_cell_b,
                                 rho_antidiffusive_flux_to_a,
                                 mom_r_antidiffusive_flux_to_a,
                                 mom_z_antidiffusive_flux_to_a,
                                 total_energy_antidiffusive_flux_to_a,
                                 n_faces,
                                 n_cells,
                                 stream);
  launch_compute_coupled_guard(cell_guard,
                               rho_low,
                               mom_r_low,
                               mom_z_low,
                               total_energy_low,
                               rho_negative_sum,
                               momentum_abs_sum,
                               total_energy_negative_sum,
                               rho_floor,
                               rho_floor_scalar,
                               internal_energy_floor,
                               internal_energy_floor_scalar,
                               n_cells,
                               stream);
  launch_limit_theta_by_cell_guard(theta,
                                   face_cell_a,
                                   face_cell_b,
                                   cell_guard,
                                   n_faces,
                                   n_cells,
                                   stream);
}

#endif  // defined(__CUDACC__)

#undef TENRYU_REMAP_REPAIR_HD

}  // namespace tenryu::hydro::ale::remap_repair
