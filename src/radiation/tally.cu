#include "radiation/tally.cuh"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "parallel/comm_buffers.hpp"
#include "radiation/imc.hpp"
#include "radiation/boundary.cuh"
#include "mesh/cell_geometry_2d.cuh"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

constexpr double kRefDiagMaxTemperatureForT4 = 1.0e6;

struct ReferenceDiagCellScratch {
  int eligible;
  int active;
  int strong;
  int hybrid_suppressed;
  double W;
  double tau;
  double chi;
  double sigma_cell;
  double E_total;
  double center;
  double center_z;
  double E_ref_total;
};

struct ReferenceDiagFaceScratch {
  double reduced_flux;
  double knudsen;
  double grad_Te;
  double grad_rho;
};

struct ReferenceDiagCellBlockStats {
  long long eligible;
  long long active;
  long long strong;
  long long hybrid_suppressed;
  double W_sum;
  double tau_sum;
  double chi_sum;
  double W_min;
  double W_max;
  double tau_min;
  double tau_max;
  double chi_max;
  double E_ref_total;
};

struct ReferenceDiagFaceBlockStats {
  double reduced_flux_max;
  double knudsen_max;
  double grad_Te_max;
  double grad_rho_max;
};

double ref_diag_clamped_temperature_for_t4_host(const double temperature_eV) {
  if (!std::isfinite(temperature_eV) || temperature_eV <= 0.0) {
    return 0.0;
  }
  return std::min(temperature_eV, kRefDiagMaxTemperatureForT4);
}

double ref_diag_safe_temperature_pow4_host(const double temperature_eV) {
  const double t = ref_diag_clamped_temperature_for_t4_host(temperature_eV);
  return t * t * t * t;
}

__device__ inline double ref_diag_min(const double a, const double b) {
  return (b < a) ? b : a;
}

__device__ inline double ref_diag_max(const double a, const double b) {
  return (a < b) ? b : a;
}

__device__ inline double ref_diag_clamp(const double value,
                                        const double low,
                                        const double high) {
  return (value < low) ? low : ((high < value) ? high : value);
}

__device__ inline double ref_diag_finite_or_zero(const double value) {
  return isfinite(value) ? value : 0.0;
}

__device__ inline double ref_diag_clamped_temperature_for_t4(const double temperature_eV) {
  if (!isfinite(temperature_eV) || temperature_eV <= 0.0) {
    return 0.0;
  }
  return ref_diag_min(temperature_eV, kRefDiagMaxTemperatureForT4);
}

__device__ inline double ref_diag_safe_temperature_pow4(const double temperature_eV) {
  const double t = ref_diag_clamped_temperature_for_t4(temperature_eV);
  return t * t * t * t;
}

__host__ __device__ inline ReferenceDiagCellBlockStats ref_diag_cell_stats_zero() {
  ReferenceDiagCellBlockStats stats{};
  stats.W_min = 1.0e300;
  stats.tau_min = 1.0e300;
  return stats;
}

__host__ __device__ inline void ref_diag_cell_stats_combine(
    ReferenceDiagCellBlockStats* dst,
    const ReferenceDiagCellBlockStats& src) {
  dst->eligible += src.eligible;
  dst->active += src.active;
  dst->strong += src.strong;
  dst->hybrid_suppressed += src.hybrid_suppressed;
  dst->W_sum += src.W_sum;
  dst->tau_sum += src.tau_sum;
  dst->chi_sum += src.chi_sum;
  dst->W_min = (src.W_min < dst->W_min) ? src.W_min : dst->W_min;
  dst->W_max = (dst->W_max < src.W_max) ? src.W_max : dst->W_max;
  dst->tau_min = (src.tau_min < dst->tau_min) ? src.tau_min : dst->tau_min;
  dst->tau_max = (dst->tau_max < src.tau_max) ? src.tau_max : dst->tau_max;
  dst->chi_max = (dst->chi_max < src.chi_max) ? src.chi_max : dst->chi_max;
  dst->E_ref_total += src.E_ref_total;
}

__host__ __device__ inline ReferenceDiagFaceBlockStats ref_diag_face_stats_zero() {
  return {};
}

__host__ __device__ inline void ref_diag_face_stats_combine(
    ReferenceDiagFaceBlockStats* dst,
    const ReferenceDiagFaceBlockStats& src) {
  dst->reduced_flux_max = (dst->reduced_flux_max < src.reduced_flux_max)
                              ? src.reduced_flux_max
                              : dst->reduced_flux_max;
  dst->knudsen_max =
      (dst->knudsen_max < src.knudsen_max) ? src.knudsen_max : dst->knudsen_max;
  dst->grad_Te_max =
      (dst->grad_Te_max < src.grad_Te_max) ? src.grad_Te_max : dst->grad_Te_max;
  dst->grad_rho_max =
      (dst->grad_rho_max < src.grad_rho_max) ? src.grad_rho_max : dst->grad_rho_max;
}

__device__ inline double ref_diag_harmonic_pair(const double a,
                                                const double b,
                                                const double floor) {
  const double aa = ref_diag_max(a, floor);
  const double bb = ref_diag_max(b, floor);
  return 2.0 * aa * bb / ref_diag_max(aa + bb, floor);
}

__device__ inline double ref_diag_reference_derivative_weight(
    const PlanckTableDeviceView planck,
    const int g,
    const double T) {
  const double dT = ref_diag_max(T * 0.01, 1.0e-3);
  const double T_hi = T + dT;
  const double T_lo = ref_diag_max(T - dT, 1.0e-6);
  const double t4_hi = T_hi * T_hi * T_hi * T_hi;
  const double t4_lo = T_lo * T_lo * T_lo * T_lo;
  const double b_hi = ref_diag_max(planck.interpolate_b(g, T_hi), 0.0);
  const double b_lo = ref_diag_max(planck.interpolate_b(g, T_lo), 0.0);
  return ref_diag_max((t4_hi * b_hi - t4_lo * b_lo) /
                          ref_diag_max(T_hi - T_lo, 1.0e-30),
                      0.0);
}

__device__ inline double ref_diag_difference_reference_cell_sigma(
    const double weight_sum,
    const double inverse_sigma_weight_sum,
    const double sigma_max) {
  return (weight_sum > 0.0 && inverse_sigma_weight_sum > 0.0)
             ? (weight_sum / inverse_sigma_weight_sum)
             : sigma_max;
}

__device__ inline double ref_diag_difference_reference_weight(const double W_max,
                                                              const double tau,
                                                              const double tau0,
                                                              const double chi,
                                                              const double chi0) {
  const double W_max_cfg = ref_diag_clamp(W_max, 0.0, 1.0);
  const double tau0_sq = tau0 * tau0;
  const double tau_sq = tau * tau;
  const double w_tau = (tau_sq > 0.0) ? (tau_sq / (tau_sq + tau0_sq)) : 0.0;
  const double chi_arg = chi / ref_diag_max(chi0, 1.0e-300);
  const double w_lte =
      (chi_arg > 1.0e75) ? 0.0 : (1.0 / (1.0 + chi_arg * chi_arg * chi_arg * chi_arg));
  return ref_diag_clamp(W_max_cfg * w_tau * w_lte, 0.0, W_max_cfg);
}

__global__ void zero_tallies_kernel(double* __restrict__ rad_dep,
                                    double* __restrict__ rad_E_tally,
                                    double* __restrict__ E_escape,
                                    double* __restrict__ holo_Prr_tally,
                                    double* __restrict__ holo_Prr_coverage_tally,
                                    const int n_cells,
                                    const int n_groups) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid < n_total) {
    rad_dep[tid] = 0.0;
    rad_E_tally[tid] = 0.0;
    if (holo_Prr_tally != nullptr) {
      holo_Prr_tally[tid] = 0.0;
    }
    if (holo_Prr_coverage_tally != nullptr) {
      holo_Prr_coverage_tally[tid] = 0.0;
    }
  }
  if (tid < n_groups) {
    E_escape[tid] = 0.0;
  }
}

__global__ void preseed_reference_absorption_kernel(
    double* __restrict__ rad_dep,
    const double* __restrict__ sigma_a_eff,
    const double* __restrict__ E_ref,
    const double* __restrict__ vol,
    const int n_cells,
    const int n_groups,
    const double dt) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }

  const int c = tid / n_groups;
  const double dep = tenryu::core::constants::c_light *
                     fmax(sigma_a_eff[tid], 0.0) * E_ref[tid] *
                     fmax(vol[c], 0.0) * dt;
  if (isfinite(dep) && dep != 0.0) {
    rad_dep[tid] += dep;
  }
}

__global__ void reference_field_diag_cells_kernel(
    ReferenceDiagCellScratch* __restrict__ cell_out,
    double* __restrict__ E_ref_start,
    double* __restrict__ W_ref_cell,
    const double* __restrict__ sigma_R,
    const double* __restrict__ node_r,
    const double* __restrict__ Te,
    const double* __restrict__ vol,
    const double* __restrict__ physical_E_density,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::uint8_t* __restrict__ diffusion_cell,
    const PlanckTableDeviceView planck,
    const int n_cells,
    const int n_groups,
    const double Te_floor,
    const double e_floor,
    const double W_max_cfg,
    const double tau0,
    const double chi0) {
  constexpr double kSigmaFloor = 1.0e-30;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  ReferenceDiagCellScratch out{};
  const int base = c * n_groups;
  out.center = 0.5 * (node_r[c] + node_r[c + 1]);
  W_ref_cell[c] = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    E_ref_start[base + g] = 0.0;
  }

  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    cell_out[c] = out;
    return;
  }

  out.eligible = 1;
  const double dx = ref_diag_max(node_r[c + 1] - node_r[c], 0.0);
  const double T =
      ref_diag_clamped_temperature_for_t4(ref_diag_max(Te[c], Te_floor));
  const double T4 = ref_diag_safe_temperature_pow4(T);

  double weight_sum = 0.0;
  double denom_sigma = 0.0;
  double sigma_max = 0.0;
  double chi_num = 0.0;
  double chi_den = e_floor;
  double B_total = 0.0;
  double E_phys_total = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const int idx = base + g;
    const double sigma_g =
        ref_diag_max(ref_diag_finite_or_zero(sigma_R[idx]), 0.0);
    sigma_max = ref_diag_max(sigma_max, sigma_g);
    const double w_g =
        (n_groups > 1) ? ref_diag_reference_derivative_weight(planck, g, T) : 1.0;
    weight_sum += w_g;
    denom_sigma += w_g / ref_diag_max(sigma_g, kSigmaFloor);

    const double b_g =
        (n_groups == 1) ? 1.0 : ref_diag_max(planck.interpolate_b(g, T), 0.0);
    const double B_g = tenryu::core::constants::a_eV * T4 * b_g;
    const double E_g = physical_E_density != nullptr
                           ? ref_diag_finite_or_zero(physical_E_density[idx])
                           : 0.0;
    B_total += B_g;
    E_phys_total += ref_diag_max(E_g, 0.0);
    chi_num += fabs(E_g - B_g);
    chi_den += ref_diag_max(E_g, 0.0) + B_g;
  }

  const double sigma_mean =
      ref_diag_difference_reference_cell_sigma(weight_sum, denom_sigma, sigma_max);
  out.sigma_cell = sigma_mean;
  out.E_total = E_phys_total;
  out.tau = sigma_mean * dx;
  out.chi = (chi_den > 0.0) ? (chi_num / chi_den) : 0.0;
  double W =
      ref_diag_difference_reference_weight(W_max_cfg, out.tau, tau0, out.chi, chi0);

  if (diffusion_cell != nullptr && diffusion_cell[c] != 0U) {
    W = 0.0;
    out.hybrid_suppressed = 1;
  }

  out.W = W;
  out.active = (W > 0.0) ? 1 : 0;
  out.strong = (W >= 0.5) ? 1 : 0;
  W_ref_cell[c] = W;
  out.E_ref_total = W * B_total * ref_diag_max(vol[c], 0.0);
  if (W != 0.0) {
    for (int g = 0; g < n_groups; ++g) {
      const double b_g =
          (n_groups == 1) ? 1.0 : ref_diag_max(planck.interpolate_b(g, T), 0.0);
      E_ref_start[base + g] = W * tenryu::core::constants::a_eV * T4 * b_g;
    }
  }
  cell_out[c] = out;
}

__global__ void reference_field_diag_cells_2d_kernel(
    ReferenceDiagCellScratch* __restrict__ cell_out,
    double* __restrict__ E_ref_start,
    double* __restrict__ W_ref_cell,
    const double* __restrict__ sigma_R,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ Te,
    const double* __restrict__ vol,
    const double* __restrict__ physical_E_density,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::uint8_t* __restrict__ diffusion_cell,
    const PlanckTableDeviceView planck,
    const int n_cells,
    const int n_groups,
    const int nr,
    const int nz,
    const double Te_floor,
    const double e_floor,
    const double W_max_cfg,
    const double tau0,
    const double chi0) {
  constexpr double kSigmaFloor = 1.0e-30;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  ReferenceDiagCellScratch out{};
  const int base = c * n_groups;
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  out.center = 0.25 * (node_r[n00] + node_r[n10] + node_r[n11] + node_r[n01]);
  out.center_z = 0.25 * (node_z[n00] + node_z[n10] + node_z[n11] + node_z[n01]);
  W_ref_cell[c] = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    E_ref_start[base + g] = 0.0;
  }

  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    cell_out[c] = out;
    return;
  }

  out.eligible = 1;
  const tenryu::mesh::CellWidths2D widths =
      tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz, c);
  const double h_eff = ref_diag_max(ref_diag_min(widths.h_R, widths.h_Z), 0.0);
  const double T =
      ref_diag_clamped_temperature_for_t4(ref_diag_max(Te[c], Te_floor));
  const double T4 = ref_diag_safe_temperature_pow4(T);

  double weight_sum = 0.0;
  double denom_sigma = 0.0;
  double sigma_max = 0.0;
  double chi_num = 0.0;
  double chi_den = e_floor;
  double B_total = 0.0;
  double E_phys_total = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const int idx = base + g;
    const double sigma_g =
        ref_diag_max(ref_diag_finite_or_zero(sigma_R[idx]), 0.0);
    sigma_max = ref_diag_max(sigma_max, sigma_g);
    const double w_g =
        (n_groups > 1) ? ref_diag_reference_derivative_weight(planck, g, T) : 1.0;
    weight_sum += w_g;
    denom_sigma += w_g / ref_diag_max(sigma_g, kSigmaFloor);

    const double b_g =
        (n_groups == 1) ? 1.0 : ref_diag_max(planck.interpolate_b(g, T), 0.0);
    const double B_g = tenryu::core::constants::a_eV * T4 * b_g;
    const double E_g = physical_E_density != nullptr
                           ? ref_diag_finite_or_zero(physical_E_density[idx])
                           : 0.0;
    B_total += B_g;
    E_phys_total += ref_diag_max(E_g, 0.0);
    chi_num += fabs(E_g - B_g);
    chi_den += ref_diag_max(E_g, 0.0) + B_g;
  }

  const double sigma_mean =
      ref_diag_difference_reference_cell_sigma(weight_sum, denom_sigma, sigma_max);
  out.sigma_cell = sigma_mean;
  out.E_total = E_phys_total;
  out.tau = sigma_mean * h_eff;
  out.chi = (chi_den > 0.0) ? (chi_num / chi_den) : 0.0;
  double W =
      ref_diag_difference_reference_weight(W_max_cfg, out.tau, tau0, out.chi, chi0);

  if (diffusion_cell != nullptr && diffusion_cell[c] != 0U) {
    W = 0.0;
    out.hybrid_suppressed = 1;
  }

  out.W = W;
  out.active = (W > 0.0) ? 1 : 0;
  out.strong = (W >= 0.5) ? 1 : 0;
  W_ref_cell[c] = W;
  out.E_ref_total = W * B_total * ref_diag_max(vol[c], 0.0);
  if (W != 0.0) {
    for (int g = 0; g < n_groups; ++g) {
      const double b_g =
          (n_groups == 1) ? 1.0 : ref_diag_max(planck.interpolate_b(g, T), 0.0);
      E_ref_start[base + g] = W * tenryu::core::constants::a_eV * T4 * b_g;
    }
  }
  cell_out[c] = out;
}

__global__ void reference_field_diag_faces_kernel(
    ReferenceDiagFaceScratch* __restrict__ face_out,
    const ReferenceDiagCellScratch* __restrict__ cell_data,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_cells,
    const double e_floor) {
  constexpr double kSigmaFloor = 1.0e-30;
  constexpr double kLogFloor = 1.0e-300;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c + 1 >= n_cells) {
    return;
  }

  ReferenceDiagFaceScratch out{};
  const int l = c;
  const int r = c + 1;
  if (cell_is_void != nullptr &&
      (cell_is_void[l] != 0U || cell_is_void[r] != 0U)) {
    face_out[c] = out;
    return;
  }

  const double dr = ref_diag_max(cell_data[r].center - cell_data[l].center, 1.0e-30);
  const double sigma_face =
      ref_diag_harmonic_pair(cell_data[l].sigma_cell, cell_data[r].sigma_cell, kSigmaFloor);
  const double E_face =
      ref_diag_max(0.5 * (cell_data[l].E_total + cell_data[r].E_total), e_floor);
  const double knudsen =
      fabs(cell_data[r].E_total - cell_data[l].E_total) /
      (ref_diag_max(sigma_face, kSigmaFloor) * E_face * dr);
  if (isfinite(knudsen)) {
    out.knudsen = knudsen;
    out.reduced_flux = knudsen / 3.0;
  }

  const double dln_Te =
      fabs(log(ref_diag_max(Te[r], kLogFloor) / ref_diag_max(Te[l], kLogFloor)));
  const double dln_rho =
      fabs(log(ref_diag_max(rho[r], kLogFloor) / ref_diag_max(rho[l], kLogFloor)));
  if (isfinite(dln_Te)) {
    out.grad_Te = dln_Te;
  }
  if (isfinite(dln_rho)) {
    out.grad_rho = dln_rho;
  }
  face_out[c] = out;
}

__device__ inline void ref_diag_accumulate_face_pair_2d(
    ReferenceDiagFaceBlockStats* local,
    const ReferenceDiagCellScratch* __restrict__ cell_data,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const std::uint8_t* __restrict__ cell_is_void,
    const int l,
    const int r,
    const double e_floor) {
  constexpr double kSigmaFloor = 1.0e-30;
  constexpr double kLogFloor = 1.0e-300;
  if (cell_is_void != nullptr &&
      (cell_is_void[l] != 0U || cell_is_void[r] != 0U)) {
    return;
  }

  const double dr = cell_data[r].center - cell_data[l].center;
  const double dz = cell_data[r].center_z - cell_data[l].center_z;
  const double ds = ref_diag_max(sqrt(dr * dr + dz * dz), 1.0e-30);
  const double sigma_face =
      ref_diag_harmonic_pair(cell_data[l].sigma_cell, cell_data[r].sigma_cell, kSigmaFloor);
  const double E_face =
      ref_diag_max(0.5 * (cell_data[l].E_total + cell_data[r].E_total), e_floor);
  const double knudsen =
      fabs(cell_data[r].E_total - cell_data[l].E_total) /
      (ref_diag_max(sigma_face, kSigmaFloor) * E_face * ds);
  if (isfinite(knudsen)) {
    local->knudsen_max = ref_diag_max(local->knudsen_max, knudsen);
    local->reduced_flux_max = ref_diag_max(local->reduced_flux_max, knudsen / 3.0);
  }

  const double dln_Te =
      fabs(log(ref_diag_max(Te[r], kLogFloor) / ref_diag_max(Te[l], kLogFloor)));
  const double dln_rho =
      fabs(log(ref_diag_max(rho[r], kLogFloor) / ref_diag_max(rho[l], kLogFloor)));
  if (isfinite(dln_Te)) {
    local->grad_Te_max = ref_diag_max(local->grad_Te_max, dln_Te);
  }
  if (isfinite(dln_rho)) {
    local->grad_rho_max = ref_diag_max(local->grad_rho_max, dln_rho);
  }
}

__global__ void reference_field_diag_faces_2d_reduce_kernel(
    ReferenceDiagFaceBlockStats* __restrict__ block_stats,
    const ReferenceDiagCellScratch* __restrict__ cell_data,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_cells,
    const int nr,
    const int nz,
    const double e_floor) {
  extern __shared__ unsigned char ref_diag_shared_raw[];
  auto* shared = reinterpret_cast<ReferenceDiagFaceBlockStats*>(ref_diag_shared_raw);
  const int tid = threadIdx.x;
  const int c = blockIdx.x * blockDim.x + tid;
  ReferenceDiagFaceBlockStats local = ref_diag_face_stats_zero();
  if (c < n_cells) {
    const int i = c / nz;
    const int j = c - i * nz;
    if (i + 1 < nr) {
      ref_diag_accumulate_face_pair_2d(&local,
                                       cell_data,
                                       Te,
                                       rho,
                                       cell_is_void,
                                       c,
                                       c + nz,
                                       e_floor);
    }
    if (j + 1 < nz) {
      ref_diag_accumulate_face_pair_2d(&local,
                                       cell_data,
                                       Te,
                                       rho,
                                       cell_is_void,
                                       c,
                                       c + 1,
                                       e_floor);
    }
  }
  shared[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      ref_diag_face_stats_combine(&shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_stats[blockIdx.x] = shared[0];
  }
}

__global__ void reference_field_diag_reduce_cells_kernel(
    const ReferenceDiagCellScratch* __restrict__ cell_data,
    ReferenceDiagCellBlockStats* __restrict__ block_stats,
    const int n_cells) {
  extern __shared__ unsigned char ref_diag_shared_raw[];
  auto* shared = reinterpret_cast<ReferenceDiagCellBlockStats*>(ref_diag_shared_raw);
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  ReferenceDiagCellBlockStats local = ref_diag_cell_stats_zero();
  if (idx < n_cells) {
    const ReferenceDiagCellScratch cell = cell_data[idx];
    if (cell.eligible != 0) {
      local.eligible = 1;
      local.active = cell.active;
      local.strong = cell.strong;
      local.hybrid_suppressed = cell.hybrid_suppressed;
      local.W_sum = cell.W;
      local.tau_sum = cell.tau;
      local.chi_sum = cell.chi;
      local.W_min = cell.W;
      local.W_max = cell.W;
      local.tau_min = cell.tau;
      local.tau_max = cell.tau;
      local.chi_max = cell.chi;
      local.E_ref_total = cell.E_ref_total;
    }
  }
  shared[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      ref_diag_cell_stats_combine(&shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_stats[blockIdx.x] = shared[0];
  }
}

__global__ void reference_field_diag_reduce_faces_kernel(
    const ReferenceDiagFaceScratch* __restrict__ face_data,
    ReferenceDiagFaceBlockStats* __restrict__ block_stats,
    const int n_faces) {
  extern __shared__ unsigned char ref_diag_shared_raw[];
  auto* shared = reinterpret_cast<ReferenceDiagFaceBlockStats*>(ref_diag_shared_raw);
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  ReferenceDiagFaceBlockStats local = ref_diag_face_stats_zero();
  if (idx < n_faces) {
    const ReferenceDiagFaceScratch face = face_data[idx];
    local.reduced_flux_max = face.reduced_flux;
    local.knudsen_max = face.knudsen;
    local.grad_Te_max = face.grad_Te;
    local.grad_rho_max = face.grad_rho;
  }
  shared[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      ref_diag_face_stats_combine(&shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_stats[blockIdx.x] = shared[0];
  }
}

__device__ inline double reference_harmonic_pair(const double a,
                                                 const double b,
                                                 const double floor) {
  const double aa = fmax(a, floor);
  const double bb = fmax(b, floor);
  return (2.0 * aa * bb) / fmax(aa + bb, floor);
}

__device__ inline double reference_ap_limiter(const double tau) {
  const double x = 0.75 * fmax(tau, 0.0);
  if (x <= 1.0e-12) {
    return 1.0;
  }
  return tanh(x) / x;
}

__device__ inline double reference_face_area_1d(const double r_face) {
  constexpr double kPiDevice = 3.141592653589793238462643383279502884;
  const double r = fmax(r_face, 0.0);
  return 4.0 * kPiDevice * r * r;
}

__device__ inline double reference_face_Q_1d(
    const int face,
    const int group,
    const double* __restrict__ E_ref_start,
    const double* __restrict__ sigma_R,
    const double* __restrict__ node_r,
    const double* __restrict__ boundary_E_inner,
    const double* __restrict__ boundary_E_outer,
    const int n_cells,
    const int n_groups,
    const double dt,
    const int bc_inner,
    const int bc_outer) {
  constexpr double kSigmaFloor = 1.0e-30;
  const double area = reference_face_area_1d(node_r[face]);
  if (!(area > 0.0) || !(dt > 0.0)) {
    return 0.0;
  }

  double E_left = 0.0;
  double E_right = 0.0;
  double sigma_face = 0.0;
  double dr = 0.0;

  if (face == 0) {
    if (!is_escape_boundary(bc_inner)) {
      return 0.0;
    }
    const int right = group;
    E_left = (boundary_E_inner != nullptr) ? fmax(boundary_E_inner[group], 0.0) : 0.0;
    E_right = fmax(E_ref_start[right], 0.0);
    sigma_face = fmax(sigma_R[right], kSigmaFloor);
    dr = 0.5 * fmax(node_r[1] - node_r[0], 0.0);
  } else if (face == n_cells) {
    if (!is_escape_boundary(bc_outer)) {
      return 0.0;
    }
    const int left = (n_cells - 1) * n_groups + group;
    E_left = fmax(E_ref_start[left], 0.0);
    E_right = (boundary_E_outer != nullptr) ? fmax(boundary_E_outer[group], 0.0) : 0.0;
    sigma_face = fmax(sigma_R[left], kSigmaFloor);
    dr = 0.5 * fmax(node_r[n_cells] - node_r[n_cells - 1], 0.0);
  } else {
    const int left = (face - 1) * n_groups + group;
    const int right = face * n_groups + group;
    E_left = fmax(E_ref_start[left], 0.0);
    E_right = fmax(E_ref_start[right], 0.0);
    sigma_face = reference_harmonic_pair(sigma_R[left], sigma_R[right], kSigmaFloor);
    dr = 0.5 * fmax(node_r[face + 1] - node_r[face - 1], 0.0);
  }

  const double tau = sigma_face * dr;
  const double psi = reference_ap_limiter(tau);
  const double F = 0.25 * tenryu::core::constants::c_light * psi * (E_left - E_right);
  const double Q = area * F * dt;
  return isfinite(Q) ? Q : 0.0;
}

__global__ void reference_face_transport_1d_kernel(
    double* __restrict__ U_ref_end,
    double* __restrict__ ref_face_delta_U,
    double* __restrict__ E_ref_avg,
    const double* __restrict__ E_ref_start,
    const double* __restrict__ sigma_R,
    const double* __restrict__ node_r,
    const double* __restrict__ vol,
    const double* __restrict__ boundary_E_inner,
    const double* __restrict__ boundary_E_outer,
    const int n_cells,
    const int n_groups,
    const double dt,
    const int bc_inner,
    const int bc_outer) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }

  const int c = tid / n_groups;
  const int g = tid - c * n_groups;
  const double Q_left = reference_face_Q_1d(c,
                                            g,
                                            E_ref_start,
                                            sigma_R,
                                            node_r,
                                            boundary_E_inner,
                                            boundary_E_outer,
                                            n_cells,
                                            n_groups,
                                            dt,
                                            bc_inner,
                                            bc_outer);
  const double Q_right = reference_face_Q_1d(c + 1,
                                             g,
                                             E_ref_start,
                                             sigma_R,
                                             node_r,
                                             boundary_E_inner,
                                             boundary_E_outer,
                                             n_cells,
                                             n_groups,
                                             dt,
                                             bc_inner,
                                             bc_outer);
  const double delta_U = Q_left - Q_right;
  const double V = fmax(vol[c], 0.0);
  const double U_start = fmax(E_ref_start[tid], 0.0) * V;
  ref_face_delta_U[tid] = delta_U;
  U_ref_end[tid] = U_start + delta_U;
  E_ref_avg[tid] = (V > 1.0e-30) ? (E_ref_start[tid] + 0.5 * delta_U / V) : 0.0;
}

__global__ void reference_face_current_1d_kernel(
    double* __restrict__ ref_face_current,
    const double* __restrict__ E_ref_start,
    const double* __restrict__ sigma_R,
    const double* __restrict__ node_r,
    const double* __restrict__ boundary_E_inner,
    const double* __restrict__ boundary_E_outer,
    const int n_cells,
    const int n_groups,
    const double dt,
    const int bc_inner,
    const int bc_outer) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_faces = n_cells + 1;
  const int n_total = n_faces * n_groups;
  if (tid >= n_total) {
    return;
  }

  const int face = tid / n_groups;
  const int g = tid - face * n_groups;
  ref_face_current[tid] = reference_face_Q_1d(face,
                                              g,
                                              E_ref_start,
                                              sigma_R,
                                              node_r,
                                              boundary_E_inner,
                                              boundary_E_outer,
                                              n_cells,
                                              n_groups,
                                              dt,
                                              bc_inner,
                                              bc_outer);
}

__global__ void reference_boundary_accounting_1d_kernel(
    double* __restrict__ boundary_escape,
    double* __restrict__ boundary_source,
    const double* __restrict__ E_ref_start,
    const double* __restrict__ sigma_R,
    const double* __restrict__ node_r,
    const double* __restrict__ boundary_E_inner,
    const double* __restrict__ boundary_E_outer,
    const int n_cells,
    const int n_groups,
    const double dt,
    const int bc_inner,
    const int bc_outer) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }

  double E_escape = 0.0;
  double E_source = 0.0;

  const double Q_inner = reference_face_Q_1d(0,
                                            g,
                                            E_ref_start,
                                            sigma_R,
                                            node_r,
                                            boundary_E_inner,
                                            boundary_E_outer,
                                            n_cells,
                                            n_groups,
                                            dt,
                                            bc_inner,
                                            bc_outer);
  if (Q_inner < 0.0) {
    E_escape += -Q_inner;
  } else {
    E_source += Q_inner;
  }

  const double Q_outer = reference_face_Q_1d(n_cells,
                                            g,
                                            E_ref_start,
                                            sigma_R,
                                            node_r,
                                            boundary_E_inner,
                                            boundary_E_outer,
                                            n_cells,
                                            n_groups,
                                            dt,
                                            bc_inner,
                                            bc_outer);
  if (Q_outer > 0.0) {
    E_escape += Q_outer;
  } else {
    E_source += -Q_outer;
  }

  boundary_escape[g] = isfinite(E_escape) ? E_escape : 0.0;
  boundary_source[g] = isfinite(E_source) ? E_source : 0.0;
}

__global__ void tally_finalize_kernel(double* __restrict__ rad_E,
                                      const double* __restrict__ rad_E_tally,
                                      const double* __restrict__ diff_E,
                                      const std::uint8_t* __restrict__ diff_cell,
                                      const double* __restrict__ E_ref_avg,
                                      double* __restrict__ residual_E,
                                      const double* __restrict__ vol,
                                      const int n_cells,
                                      const int n_groups,
                                      const double dt,
                                      unsigned long long* __restrict__ negative_count) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }

  const int c = tid / n_groups;
  if (diff_E != nullptr && diff_cell != nullptr && diff_cell[c] != 0U) {
    rad_E[tid] = diff_E[tid];
    if (residual_E != nullptr) {
      const double ref = (E_ref_avg != nullptr) ? E_ref_avg[tid] : 0.0;
      residual_E[tid] = diff_E[tid] - ref;
    }
    return;
  }

  const double V = vol[c];
  if (V <= 1.0e-30 || dt <= 0.0) {
    rad_E[tid] = 0.0;
    if (residual_E != nullptr) {
      residual_E[tid] = 0.0;
    }
    return;
  }

  const double residual = rad_E_tally[tid] / (V * tenryu::core::constants::c_light * dt);
  if (residual_E != nullptr) {
    residual_E[tid] = residual;
  }
  const double ref = (E_ref_avg != nullptr) ? E_ref_avg[tid] : 0.0;
  double E = ref + residual;
  if (E < 0.0) {
    E = 0.0;
    if (negative_count != nullptr) {
      atomicAdd(negative_count, 1ULL);
    }
  }
  rad_E[tid] = E;
}

__global__ void holo_accept_radiation_kernel(double* __restrict__ rad_E,
                                             const double* __restrict__ E_LO,
                                             const std::uint8_t* __restrict__ core_mask,
                                             const int n_cells,
                                             const int n_groups) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }

  const int c = tid / n_groups;
  if (core_mask == nullptr || core_mask[c] == 0U) {
    return;
  }
  const double E_LO_raw = E_LO[tid];
  const double E_lo = isfinite(E_LO_raw) ? E_LO_raw : 0.0;
  rad_E[tid] = fmax(E_lo, 0.0);
}

__global__ void holo_set_reference_U_kernel(double* __restrict__ previous_reference_U,
                                            const double* __restrict__ rad_E,
                                            const double* __restrict__ vol,
                                            const std::uint8_t* __restrict__ core_mask,
                                            const int n_cells,
                                            const int n_groups) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }

  const int c = tid / n_groups;
  if (core_mask == nullptr || core_mask[c] == 0U) {
    return;
  }
  const double E = isfinite(rad_E[tid]) ? rad_E[tid] : 0.0;
  const double V = isfinite(vol[c]) ? fmax(vol[c], 0.0) : 0.0;
  previous_reference_U[tid] = fmax(E, 0.0) * V;
}

__device__ inline double holo_finite_or_zero(const double value) {
  return isfinite(value) ? value : 0.0;
}

__device__ inline double holo_nonnegative_finite(const double value) {
  return isfinite(value) ? fmax(value, 0.0) : 0.0;
}

__device__ inline double holo_harmonic_pair(const double a,
                                            const double b,
                                            const double floor) {
  const double aa = fmax(a, floor);
  const double bb = fmax(b, floor);
  return (2.0 * aa * bb) / fmax(aa + bb, floor);
}

__device__ inline double holo_cell_center(const double* __restrict__ node_r,
                                          const int c) {
  return 0.5 * (holo_finite_or_zero(node_r[c]) +
                holo_finite_or_zero(node_r[c + 1]));
}

__device__ inline double holo_face_area(const double* __restrict__ node_r,
                                        const int face) {
  constexpr double kPiDevice = 3.141592653589793238462643383279502884;
  const double r = fmax(holo_finite_or_zero(node_r[face]), 0.0);
  return 4.0 * kPiDevice * r * r;
}

__device__ inline double holo_diffusive_face_current(
    const double* __restrict__ E,
    const double* __restrict__ sigma_R,
    const double* __restrict__ node_r,
    const int face,
    const int group,
    const int n_cells,
    const int n_groups) {
  constexpr double kSigmaFloor = 1.0e-30;
  if (face <= 0 || face >= n_cells) {
    return 0.0;
  }
  const int left = face - 1;
  const int right = face;
  const double dr = holo_cell_center(node_r, right) - holo_cell_center(node_r, left);
  if (!(dr > 0.0)) {
    return 0.0;
  }
  const int left_key = left * n_groups + group;
  const int right_key = right * n_groups + group;
  const double sigma_left =
      fmax(holo_nonnegative_finite(sigma_R[left_key]), kSigmaFloor);
  const double sigma_right =
      fmax(holo_nonnegative_finite(sigma_R[right_key]), kSigmaFloor);
  const double D_left = tenryu::core::constants::c_light / (3.0 * sigma_left);
  const double D_right = tenryu::core::constants::c_light / (3.0 * sigma_right);
  const double D_face = holo_harmonic_pair(D_left, D_right, 0.0);
  const double coupling = holo_face_area(node_r, face) * D_face / dr;
  const double E_left = holo_finite_or_zero(E[left_key]);
  const double E_right = holo_finite_or_zero(E[right_key]);
  const double J = coupling * (E_right - E_left);
  return isfinite(J) ? J : 0.0;
}

__device__ inline double holo_step_face_current_rate(
    const double* __restrict__ face_current_step,
    const double* __restrict__ reference_face_current,
    const int face,
    const int group,
    const int n_cells,
    const int n_groups,
    const double dt) {
  if (!(dt > 0.0) || face < 0 || face > n_cells ||
      group < 0 || group >= n_groups) {
    return 0.0;
  }
  const int key = face * n_groups + group;
  double J_step = 0.0;
  if (face_current_step != nullptr && isfinite(face_current_step[key])) {
    J_step += face_current_step[key];
  }
  if (reference_face_current != nullptr && isfinite(reference_face_current[key])) {
    J_step += reference_face_current[key];
  }
  return J_step / dt;
}

__device__ inline double holo_flux_defect(
    const double* __restrict__ E_HO,
    const double* __restrict__ face_current_step,
    const double* __restrict__ reference_face_current,
    const double* __restrict__ sigma_R,
    const double* __restrict__ node_r,
    const double* __restrict__ lo_weight,
    const int face,
    const int group,
    const int n_cells,
    const int n_groups,
    const double dt) {
  if (face <= 0 || face >= n_cells) {
    return 0.0;
  }
  if (face_current_step == nullptr && reference_face_current == nullptr) {
    return 0.0;
  }
  const double wl = lo_weight[face - 1];
  const double wr = lo_weight[face];
  if (!isfinite(wl) || !isfinite(wr) || wl <= 0.0 || wr <= 0.0) {
    return 0.0;
  }
  const double J_ho = holo_step_face_current_rate(face_current_step,
                                                  reference_face_current,
                                                  face,
                                                  group,
                                                  n_cells,
                                                  n_groups,
                                                  dt);
  const double J_diff = holo_diffusive_face_current(E_HO,
                                                    sigma_R,
                                                    node_r,
                                                    face,
                                                    group,
                                                    n_cells,
                                                    n_groups);
  const double defect = J_ho - J_diff;
  return isfinite(defect) ? defect : 0.0;
}

__global__ void compute_holo_consistency_kernel(
    double* __restrict__ consistency_source,
    const double* __restrict__ E_HO,
    const double* __restrict__ E_LO_pred,
    const double* __restrict__ E_old,
    const double* __restrict__ face_current_step,
    const double* __restrict__ reference_face_current,
    const double* __restrict__ sigma_R,
    const double* __restrict__ vol,
    const double* __restrict__ node_r,
    const double* __restrict__ lo_weight,
    const int n_cells,
    const int n_groups,
    const double dt) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }

  const int c = tid / n_groups;
  const int group = tid - c * n_groups;
  const double w = lo_weight[c];
  if (!isfinite(w) || w <= 0.0) {
    consistency_source[tid] = 0.0;
    return;
  }
  const double eho = holo_finite_or_zero(E_HO[tid]);
  const double elo = holo_finite_or_zero(E_LO_pred[tid]);
  const double eold = holo_finite_or_zero(E_old[tid]);
  const double V = isfinite(vol[c]) ? fmax(vol[c], 0.0) : 0.0;
  const double temporal =
      (dt > 0.0) ? (V / dt) * ((eho - eold) - (elo - eold)) : 0.0;
  const double gamma_left = holo_flux_defect(E_HO,
                                             face_current_step,
                                             reference_face_current,
                                             sigma_R,
                                             node_r,
                                             lo_weight,
                                             c,
                                             group,
                                             n_cells,
                                             n_groups,
                                             dt);
  const double gamma_right = holo_flux_defect(E_HO,
                                              face_current_step,
                                              reference_face_current,
                                              sigma_R,
                                              node_r,
                                              lo_weight,
                                              c + 1,
                                              group,
                                              n_cells,
                                              n_groups,
                                              dt);
  const double source = temporal + gamma_right - gamma_left;
  consistency_source[tid] = isfinite(source) ? source : 0.0;
}

__global__ void difference_reproject_residual_kernel(double* __restrict__ residual_E,
                                                     const double* __restrict__ rad_E,
                                                     const double* __restrict__ E_ref,
                                                     const std::uint8_t* __restrict__ core_mask,
                                                     const int n_cells,
                                                     const int n_groups) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }

  const int c = tid / n_groups;
  if (core_mask != nullptr && core_mask[c] == 0U) {
    return;
  }
  const double E_acc = isfinite(rad_E[tid]) ? rad_E[tid] : 0.0;
  const double E_reference = isfinite(E_ref[tid]) ? E_ref[tid] : 0.0;
  residual_E[tid] = E_acc - E_reference;
}

__global__ void holo_prr_finalize_kernel(double* __restrict__ Prr,
                                         double* __restrict__ chi,
                                         double* __restrict__ coverage,
                                         const double* __restrict__ Prr_tally,
                                         const double* __restrict__ coverage_tally,
                                         const double* __restrict__ rad_E_tally,
                                         const double* __restrict__ vol,
                                         const std::uint8_t* __restrict__ holo_core,
                                         const int n_cells,
                                         const int n_groups,
                                         const double dt,
                                         const double E_floor) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }

  const int c = tid / n_groups;
  if (holo_core == nullptr || holo_core[c] == 0U || vol[c] <= 1.0e-30 || dt <= 0.0) {
    Prr[tid] = 0.0;
    chi[tid] = 0.0;
    coverage[tid] = 0.0;
    return;
  }

  const double norm = vol[c] * tenryu::core::constants::c_light * dt;
  const double prr = Prr_tally[tid] / norm;
  const double E_ho = rad_E_tally[tid] / norm;
  const double denom = fmax(E_ho, fmax(E_floor, 0.0));
  const double raw_coverage =
      coverage_tally[tid] / fmax(fabs(rad_E_tally[tid]), 1.0e-300);

  Prr[tid] = isfinite(prr) ? prr : 0.0;
  chi[tid] = (denom > 0.0 && isfinite(prr)) ? (prr / denom) : 0.0;
  coverage[tid] = isfinite(raw_coverage) ? fmin(fmax(raw_coverage, 0.0), 1.0) : 0.0;
}

}  // namespace

void zero_tallies_cuda(double* rad_dep,
                       double* rad_E_tally,
                       double* E_escape,
                       const int n_cells,
                       const int n_groups,
                       double* holo_Prr_tally,
                       double* holo_Prr_coverage_tally) {
  TENRYU_ASSERT(rad_dep != nullptr, "zero_tallies_cuda requires rad_dep");
  TENRYU_ASSERT(rad_E_tally != nullptr, "zero_tallies_cuda requires rad_E_tally");
  TENRYU_ASSERT(E_escape != nullptr, "zero_tallies_cuda requires E_escape");

  constexpr int kBlock = 256;
  const int n_total = n_cells * n_groups;
  const int n_work = (n_total > n_groups) ? n_total : n_groups;
  const int grid = (n_work + kBlock - 1) / kBlock;
  if (grid > 0) {
    zero_tallies_kernel<<<grid, kBlock>>>(rad_dep,
                                          rad_E_tally,
                                          E_escape,
                                          holo_Prr_tally,
                                          holo_Prr_coverage_tally,
                                          n_cells,
                                          n_groups);
    cuda_check(cudaGetLastError(), "zero_tallies kernel launch failed");
    cuda_check(cudaDeviceSynchronize(), "zero_tallies kernel execution failed");
  }
}

void preseed_reference_absorption_cuda(double* rad_dep,
                                       const double* sigma_a_eff,
                                       const double* E_ref,
                                       const double* vol,
                                       const int n_cells,
                                       const int n_groups,
                                       const double dt) {
  TENRYU_ASSERT(rad_dep != nullptr,
                "preseed_reference_absorption_cuda requires rad_dep");
  TENRYU_ASSERT(sigma_a_eff != nullptr,
                "preseed_reference_absorption_cuda requires sigma_a_eff");
  TENRYU_ASSERT(E_ref != nullptr,
                "preseed_reference_absorption_cuda requires E_ref");
  TENRYU_ASSERT(vol != nullptr,
                "preseed_reference_absorption_cuda requires vol");

  constexpr int kBlock = 256;
  const int n_total = n_cells * n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid > 0) {
    preseed_reference_absorption_kernel<<<grid, kBlock>>>(
        rad_dep, sigma_a_eff, E_ref, vol, n_cells, n_groups, dt);
    cuda_check(cudaGetLastError(),
               "preseed_reference_absorption kernel launch failed");
  }
}

ReferenceFaceTransportResult reference_face_transport_1d_cuda(
    double* U_ref_end,
    double* ref_face_delta_U,
    double* E_ref_avg,
    const double* E_ref_start,
    const double* sigma_R,
    const double* node_r,
    const double* vol,
    const double* boundary_E_inner,
    const double* boundary_E_outer,
    const int n_cells,
    const int n_groups,
    const double dt,
    const int bc_inner,
    const int bc_outer,
    parallel::DeviceArray* boundary_workspace,
    double* ref_face_current) {
  TENRYU_ASSERT(U_ref_end != nullptr,
                "reference_face_transport_1d_cuda requires U_ref_end");
  TENRYU_ASSERT(ref_face_delta_U != nullptr,
                "reference_face_transport_1d_cuda requires ref_face_delta_U");
  TENRYU_ASSERT(E_ref_avg != nullptr,
                "reference_face_transport_1d_cuda requires E_ref_avg");
  TENRYU_ASSERT(E_ref_start != nullptr,
                "reference_face_transport_1d_cuda requires E_ref_start");
  TENRYU_ASSERT(sigma_R != nullptr,
                "reference_face_transport_1d_cuda requires sigma_R");
  TENRYU_ASSERT(node_r != nullptr,
                "reference_face_transport_1d_cuda requires node_r");
  TENRYU_ASSERT(vol != nullptr,
                "reference_face_transport_1d_cuda requires vol");

  ReferenceFaceTransportResult result{};
  if (n_cells <= 0 || n_groups <= 0 || !(dt > 0.0)) {
    return result;
  }

  constexpr int kBlock = 256;
  const int n_total = n_cells * n_groups;
  const int grid_cells = (n_total + kBlock - 1) / kBlock;
  reference_face_transport_1d_kernel<<<grid_cells, kBlock>>>(U_ref_end,
                                                             ref_face_delta_U,
                                                             E_ref_avg,
                                                             E_ref_start,
                                                             sigma_R,
                                                             node_r,
                                                             vol,
                                                             boundary_E_inner,
                                                             boundary_E_outer,
                                                             n_cells,
                                                             n_groups,
                                                             dt,
                                                             bc_inner,
                                                             bc_outer);
  cuda_check(cudaGetLastError(), "reference_face_transport_1d kernel launch failed");

  if (ref_face_current != nullptr) {
    const int n_faces_total = (n_cells + 1) * n_groups;
    const int grid_faces = (n_faces_total + kBlock - 1) / kBlock;
    reference_face_current_1d_kernel<<<grid_faces, kBlock>>>(ref_face_current,
                                                             E_ref_start,
                                                             sigma_R,
                                                             node_r,
                                                             boundary_E_inner,
                                                             boundary_E_outer,
                                                             n_cells,
                                                             n_groups,
                                                             dt,
                                                             bc_inner,
                                                             bc_outer);
    cuda_check(cudaGetLastError(),
               "reference_face_current_1d kernel launch failed");
  }

  const std::size_t group_bytes = sizeof(double) * static_cast<std::size_t>(n_groups);
  parallel::DeviceArray local_boundary_workspace;
  parallel::DeviceArray& boundary_storage =
      (boundary_workspace != nullptr) ? *boundary_workspace : local_boundary_workspace;
  boundary_storage.resize(group_bytes * 2U);
  double* d_boundary_escape = boundary_storage.as<double>();
  double* d_boundary_source =
      reinterpret_cast<double*>(static_cast<std::uint8_t*>(boundary_storage.ptr) +
                                group_bytes);
  const int grid_groups = (n_groups + kBlock - 1) / kBlock;
  reference_boundary_accounting_1d_kernel<<<grid_groups, kBlock>>>(d_boundary_escape,
                                                                   d_boundary_source,
                                                                   E_ref_start,
                                                                   sigma_R,
                                                                   node_r,
                                                                   boundary_E_inner,
                                                                   boundary_E_outer,
                                                                   n_cells,
                                                                   n_groups,
                                                                   dt,
                                                                   bc_inner,
                                                                   bc_outer);
  cuda_check(cudaGetLastError(),
             "reference_boundary_accounting_1d kernel launch failed");

  std::vector<double> host_escape(static_cast<std::size_t>(n_groups), 0.0);
  std::vector<double> host_source(static_cast<std::size_t>(n_groups), 0.0);
  cuda_check(cudaMemcpy(host_escape.data(),
                        d_boundary_escape,
                        group_bytes,
                        cudaMemcpyDeviceToHost),
             "reference_face_transport_1d copy boundary_escape failed");
  cuda_check(cudaMemcpy(host_source.data(),
                        d_boundary_source,
                        group_bytes,
                        cudaMemcpyDeviceToHost),
             "reference_face_transport_1d copy boundary_source failed");

  long double escape_total = 0.0L;
  long double source_total = 0.0L;
  for (int g = 0; g < n_groups; ++g) {
    const double E_escape = host_escape[static_cast<std::size_t>(g)];
    const double E_source = host_source[static_cast<std::size_t>(g)];
    if (std::isfinite(E_escape) && E_escape > 0.0) {
      escape_total += static_cast<long double>(E_escape);
    }
    if (std::isfinite(E_source) && E_source > 0.0) {
      source_total += static_cast<long double>(E_source);
    }
  }
  result.E_escape = static_cast<double>(escape_total);
  result.E_source = static_cast<double>(source_total);
  return result;
}

ReferenceFieldDiagnostics compute_reference_field_diagnostics_cuda(
    const core::State& state,
    const core::Config& cfg,
    const PlanckTableDeviceView planck,
    const double* sigma_R,
    const double* physical_E_density,
    const std::uint8_t* diffusion_cell,
    const int n_cells,
    const int n_groups,
    double* E_ref_start,
    double* W_ref_cell,
    parallel::DeviceArray& cell_workspace,
    parallel::DeviceArray& face_workspace,
    parallel::DeviceArray& void_workspace) {
  ReferenceFieldDiagnostics diag{};
  if (!cfg.radiation.imc.difference.enabled) {
    return diag;
  }
  const bool is_1d = (state.mesh.dim == 1);
  const bool is_2d = (state.mesh.dim == 2);
  TENRYU_ASSERT(is_1d || is_2d,
                "difference reference diagnostics require 1D_SPH or 2D_RZ");
  const std::size_t n_cells_us = static_cast<std::size_t>(std::max(n_cells, 0));
  const std::size_t n_groups_us = static_cast<std::size_t>(std::max(n_groups, 0));
  const std::size_t n_cell_groups_us = n_cells_us * n_groups_us;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(state.Te.size() == n_cells_us,
                "compute_reference_field_diagnostics_cuda Te size mismatch");
  TENRYU_ASSERT(state.rho.size() == n_cells_us,
                "compute_reference_field_diagnostics_cuda rho size mismatch");
  TENRYU_ASSERT(state.vol.size() == n_cells_us,
                "compute_reference_field_diagnostics_cuda vol size mismatch");
  if (is_1d) {
    TENRYU_ASSERT(state.x_r.size() == n_cells_us + 1U,
                  "compute_reference_field_diagnostics_cuda node_r size mismatch");
  } else {
    TENRYU_ASSERT(nr > 0 && nz > 0,
                  "compute_reference_field_diagnostics_cuda requires valid 2D topology");
    TENRYU_ASSERT(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz) ==
                      n_cells_us,
                  "compute_reference_field_diagnostics_cuda requires nr*nz cells");
    const std::size_t n_nodes =
        static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
    TENRYU_ASSERT(state.x_r.size() == n_nodes,
                  "compute_reference_field_diagnostics_cuda node_r size mismatch");
    TENRYU_ASSERT(state.x_z.size() == n_nodes,
                  "compute_reference_field_diagnostics_cuda node_z size mismatch");
  }
  TENRYU_ASSERT(state.cell_is_void.size() == n_cells_us,
                "compute_reference_field_diagnostics_cuda void mask size mismatch");
  TENRYU_ASSERT(planck.n_groups == n_groups,
                "compute_reference_field_diagnostics_cuda group count mismatch");

  diag.valid = true;
  if (n_cells <= 0 || n_groups <= 0) {
    return diag;
  }
  TENRYU_ASSERT(sigma_R != nullptr,
                "compute_reference_field_diagnostics_cuda requires sigma_R");
  TENRYU_ASSERT(E_ref_start != nullptr,
                "compute_reference_field_diagnostics_cuda requires E_ref_start");
  TENRYU_ASSERT(W_ref_cell != nullptr,
                "compute_reference_field_diagnostics_cuda requires W_ref_cell");

  constexpr int kBlock = 128;
  const int grid_cells = (n_cells + kBlock - 1) / kBlock;
  const int n_face_items = is_1d ? std::max(n_cells - 1, 0) : n_cells;
  const int grid_faces = (n_face_items + kBlock - 1) / kBlock;
  const bool have_void_cells =
      std::any_of(state.cell_is_void.begin(),
                  state.cell_is_void.end(),
                  [](const std::uint8_t v) { return v != 0U; });
  const auto align_up = [](const std::size_t value, const std::size_t alignment) {
    return ((value + alignment - 1U) / alignment) * alignment;
  };
  const std::size_t void_bytes = have_void_cells ? sizeof(std::uint8_t) * n_cells_us : 0U;
  const std::size_t cell_stats_offset =
      align_up(void_bytes, alignof(ReferenceDiagCellBlockStats));
  const std::size_t cell_stats_bytes =
      sizeof(ReferenceDiagCellBlockStats) * static_cast<std::size_t>(grid_cells);
  const std::size_t face_stats_offset =
      align_up(cell_stats_offset + cell_stats_bytes,
               alignof(ReferenceDiagFaceBlockStats));
  const std::size_t face_stats_bytes =
      sizeof(ReferenceDiagFaceBlockStats) * static_cast<std::size_t>(grid_faces);
  void_workspace.resize(face_stats_offset + face_stats_bytes);
  auto* workspace_raw = static_cast<std::uint8_t*>(void_workspace.ptr);
  std::uint8_t* d_void = nullptr;
  if (have_void_cells) {
    d_void = workspace_raw;
    cuda_check(cudaMemcpy(d_void,
                          state.cell_is_void.data(),
                          void_bytes,
                          cudaMemcpyHostToDevice),
               "compute_reference_field_diagnostics_cuda copy void mask failed");
  }
  auto* d_cell_block_stats =
      reinterpret_cast<ReferenceDiagCellBlockStats*>(workspace_raw + cell_stats_offset);
  auto* d_face_block_stats =
      reinterpret_cast<ReferenceDiagFaceBlockStats*>(workspace_raw + face_stats_offset);

  cell_workspace.resize(sizeof(ReferenceDiagCellScratch) * n_cells_us);
  auto* d_cells = cell_workspace.as<ReferenceDiagCellScratch>();
  const std::size_t n_faces_us =
      (is_1d && n_cells_us > 0U) ? (n_cells_us - 1U) : 0U;
  face_workspace.resize(sizeof(ReferenceDiagFaceScratch) * n_faces_us);
  auto* d_faces = face_workspace.as<ReferenceDiagFaceScratch>();

  const double* d_physical_E = physical_E_density;
  if (d_physical_E == nullptr && state.rad_E.size() == n_cell_groups_us) {
    d_physical_E = state.rad_E.data();
  }

  const double e_floor =
      std::max(core::constants::a_eV *
                   ref_diag_safe_temperature_pow4_host(
                       std::max(cfg.numerics.floors.Te, 1.0e-12)),
               1.0e-300);
  const auto& dc = cfg.radiation.imc.difference;

  if (is_1d) {
    reference_field_diag_cells_kernel<<<grid_cells, kBlock>>>(d_cells,
                                                              E_ref_start,
                                                              W_ref_cell,
                                                              sigma_R,
                                                              state.x_r.data(),
                                                              state.Te.data(),
                                                              state.vol.data(),
                                                              d_physical_E,
                                                              d_void,
                                                              diffusion_cell,
                                                              planck,
                                                              n_cells,
                                                              n_groups,
                                                              cfg.numerics.floors.Te,
                                                              e_floor,
                                                              dc.W_max,
                                                              dc.tau0,
                                                              dc.chi0);
  } else {
    reference_field_diag_cells_2d_kernel<<<grid_cells, kBlock>>>(d_cells,
                                                                 E_ref_start,
                                                                 W_ref_cell,
                                                                 sigma_R,
                                                                 state.x_r.data(),
                                                                 state.x_z.data(),
                                                                 state.Te.data(),
                                                                 state.vol.data(),
                                                                 d_physical_E,
                                                                 d_void,
                                                                 diffusion_cell,
                                                                 planck,
                                                                 n_cells,
                                                                 n_groups,
                                                                 nr,
                                                                 nz,
                                                                 cfg.numerics.floors.Te,
                                                                 e_floor,
                                                                 dc.W_max,
                                                                 dc.tau0,
                                                                 dc.chi0);
  }
  cuda_check(cudaGetLastError(),
             "compute_reference_field_diagnostics_cuda cell kernel launch failed");

  if (is_1d && n_face_items > 0) {
    reference_field_diag_faces_kernel<<<grid_faces, kBlock>>>(d_faces,
                                                              d_cells,
                                                              state.Te.data(),
                                                              state.rho.data(),
                                                              d_void,
                                                              n_cells,
                                                              e_floor);
    cuda_check(cudaGetLastError(),
               "compute_reference_field_diagnostics_cuda face kernel launch failed");
  } else if (is_2d && n_face_items > 0) {
    reference_field_diag_faces_2d_reduce_kernel
        <<<grid_faces,
           kBlock,
           sizeof(ReferenceDiagFaceBlockStats) * static_cast<std::size_t>(kBlock)>>>(
            d_face_block_stats,
            d_cells,
            state.Te.data(),
            state.rho.data(),
            d_void,
            n_cells,
            nr,
            nz,
            e_floor);
    cuda_check(cudaGetLastError(),
               "compute_reference_field_diagnostics_cuda 2D face reduction launch failed");
  }
  reference_field_diag_reduce_cells_kernel
      <<<grid_cells,
         kBlock,
         sizeof(ReferenceDiagCellBlockStats) * static_cast<std::size_t>(kBlock)>>>(
          d_cells,
          d_cell_block_stats,
          n_cells);
  cuda_check(cudaGetLastError(),
             "compute_reference_field_diagnostics_cuda cell reduction launch failed");
  if (is_1d && n_face_items > 0) {
    reference_field_diag_reduce_faces_kernel
        <<<grid_faces,
           kBlock,
           sizeof(ReferenceDiagFaceBlockStats) * static_cast<std::size_t>(kBlock)>>>(
            d_faces,
            d_face_block_stats,
            n_face_items);
    cuda_check(cudaGetLastError(),
               "compute_reference_field_diagnostics_cuda face reduction launch failed");
  }

  std::vector<ReferenceDiagCellBlockStats> host_cell_blocks(
      static_cast<std::size_t>(grid_cells));
  cuda_check(cudaMemcpy(host_cell_blocks.data(),
                        d_cell_block_stats,
                        cell_stats_bytes,
                        cudaMemcpyDeviceToHost),
             "compute_reference_field_diagnostics_cuda copy cell reduction failed");
  std::vector<ReferenceDiagFaceBlockStats> host_face_blocks(
      static_cast<std::size_t>(grid_faces));
  if (grid_faces > 0) {
    cuda_check(cudaMemcpy(host_face_blocks.data(),
                          d_face_block_stats,
                          face_stats_bytes,
                          cudaMemcpyDeviceToHost),
               "compute_reference_field_diagnostics_cuda copy face reduction failed");
  }

  ReferenceDiagCellBlockStats cell_total = ref_diag_cell_stats_zero();
  for (const ReferenceDiagCellBlockStats& block : host_cell_blocks) {
    ref_diag_cell_stats_combine(&cell_total, block);
  }
  diag.eligible_cells = cell_total.eligible;
  diag.active_cells = cell_total.active;
  diag.strong_cells = cell_total.strong;
  diag.hybrid_suppressed_cells = cell_total.hybrid_suppressed;
  diag.E_ref_total = cell_total.E_ref_total;

  if (diag.eligible_cells > 0) {
    const double inv_count = 1.0 / static_cast<double>(diag.eligible_cells);
    diag.W_min = std::isfinite(cell_total.W_min) ? cell_total.W_min : 0.0;
    diag.W_mean = cell_total.W_sum * inv_count;
    diag.W_max = cell_total.W_max;
    diag.tau_min = std::isfinite(cell_total.tau_min) ? cell_total.tau_min : 0.0;
    diag.tau_mean = cell_total.tau_sum * inv_count;
    diag.tau_max = cell_total.tau_max;
    diag.chi_mean = cell_total.chi_sum * inv_count;
    diag.chi_max = cell_total.chi_max;
  }

  ReferenceDiagFaceBlockStats face_total = ref_diag_face_stats_zero();
  for (const ReferenceDiagFaceBlockStats& block : host_face_blocks) {
    ref_diag_face_stats_combine(&face_total, block);
  }
  diag.knudsen_max = face_total.knudsen_max;
  diag.reduced_flux_max = face_total.reduced_flux_max;
  diag.front_grad_Te_max = face_total.grad_Te_max;
  diag.front_grad_rho_max = face_total.grad_rho_max;
  return diag;
}

void tally_finalize_cuda(double* rad_E,
                         const double* rad_E_tally,
                         const double* vol,
                         const int n_cells,
                         const int n_groups,
                         const double dt,
                         const double* diff_E,
                         const std::uint8_t* diff_cell,
                         const double* E_ref_avg,
                         double* residual_E) {
  TENRYU_ASSERT(rad_E != nullptr, "tally_finalize_cuda requires rad_E");
  TENRYU_ASSERT(rad_E_tally != nullptr, "tally_finalize_cuda requires rad_E_tally");
  TENRYU_ASSERT(vol != nullptr, "tally_finalize_cuda requires vol");

  constexpr int kBlock = 256;
  const int n_total = n_cells * n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid > 0) {
    unsigned long long* d_negative_count = nullptr;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_negative_count), sizeof(unsigned long long)),
               "tally_finalize cudaMalloc negative_count failed");
    cuda_check(cudaMemset(d_negative_count, 0, sizeof(unsigned long long)),
               "tally_finalize cudaMemset negative_count failed");
    tally_finalize_kernel<<<grid, kBlock>>>(rad_E,
                                            rad_E_tally,
                                            diff_E,
                                            diff_cell,
                                            E_ref_avg,
                                            residual_E,
                                            vol,
                                            n_cells,
                                            n_groups,
                                            dt,
                                            d_negative_count);
    cuda_check(cudaGetLastError(), "tally_finalize kernel launch failed");
    cuda_check(cudaDeviceSynchronize(), "tally_finalize kernel execution failed");
    unsigned long long host_negative_count = 0ULL;
    cuda_check(cudaMemcpy(&host_negative_count,
                          d_negative_count,
                          sizeof(host_negative_count),
                          cudaMemcpyDeviceToHost),
               "tally_finalize copy negative_count failed");
    cuda_check(cudaFree(d_negative_count), "tally_finalize cudaFree negative_count failed");
    if (host_negative_count > 0ULL) {
      static std::atomic<int> warn_count{0};
      const int n = warn_count.fetch_add(1);
      if (n == 0) {
        core::log_warning("tally_finalize: clamped negative rad_E values to 0 (count=" +
                          std::to_string(host_negative_count) + "); "
                          "this is expected with difference formulation");
      }
    }
  }
}

void holo_accept_radiation_cuda(double* rad_E,
                                const double* E_LO,
                                const std::uint8_t* core_mask,
                                const int n_cells,
                                const int n_groups,
                                const double* vol,
                                double* previous_reference_U) {
  TENRYU_ASSERT(rad_E != nullptr, "holo_accept_radiation_cuda requires rad_E");
  TENRYU_ASSERT(E_LO != nullptr, "holo_accept_radiation_cuda requires E_LO");
  TENRYU_ASSERT(core_mask != nullptr, "holo_accept_radiation_cuda requires core_mask");
  TENRYU_ASSERT(vol == nullptr && previous_reference_U == nullptr,
                "holo_accept_radiation_cuda does not retarget previous_reference_U");

  constexpr int kBlock = 256;
  const int n_total = n_cells * n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid > 0) {
    holo_accept_radiation_kernel<<<grid, kBlock>>>(rad_E,
                                                   E_LO,
                                                   core_mask,
                                                   n_cells,
                                                   n_groups);
    cuda_check(cudaGetLastError(), "holo_accept_radiation kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "holo_accept_radiation kernel execution failed");
  }
}

void holo_set_reference_U_cuda(double* previous_reference_U,
                               const double* rad_E,
                               const double* vol,
                               const std::uint8_t* core_mask,
                               const int n_cells,
                               const int n_groups) {
  TENRYU_ASSERT(previous_reference_U != nullptr,
                "holo_set_reference_U_cuda requires previous_reference_U");
  TENRYU_ASSERT(rad_E != nullptr, "holo_set_reference_U_cuda requires rad_E");
  TENRYU_ASSERT(vol != nullptr, "holo_set_reference_U_cuda requires vol");
  TENRYU_ASSERT(core_mask != nullptr, "holo_set_reference_U_cuda requires core_mask");

  constexpr int kBlock = 256;
  const int n_total = n_cells * n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid > 0) {
    holo_set_reference_U_kernel<<<grid, kBlock>>>(previous_reference_U,
                                                  rad_E,
                                                  vol,
                                                  core_mask,
                                                  n_cells,
                                                  n_groups);
    cuda_check(cudaGetLastError(), "holo_set_reference_U kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "holo_set_reference_U kernel execution failed");
  }
}

void compute_holo_consistency_cuda(double* consistency_source,
                                   const double* E_HO,
                                   const double* E_LO_pred,
                                   const double* E_old,
                                   const double* face_current_step,
                                   const double* reference_face_current,
                                   const double* sigma_R,
                                   const double* vol,
                                   const double* node_r,
                                   const double* lo_weight,
                                   const int n_cells,
                                   const int n_groups,
                                   const double dt) {
  TENRYU_ASSERT(consistency_source != nullptr,
                "compute_holo_consistency_cuda requires consistency_source");
  TENRYU_ASSERT(E_HO != nullptr, "compute_holo_consistency_cuda requires E_HO");
  TENRYU_ASSERT(E_LO_pred != nullptr,
                "compute_holo_consistency_cuda requires E_LO_pred");
  TENRYU_ASSERT(E_old != nullptr, "compute_holo_consistency_cuda requires E_old");
  TENRYU_ASSERT(sigma_R != nullptr, "compute_holo_consistency_cuda requires sigma_R");
  TENRYU_ASSERT(vol != nullptr, "compute_holo_consistency_cuda requires vol");
  TENRYU_ASSERT(node_r != nullptr, "compute_holo_consistency_cuda requires node_r");
  TENRYU_ASSERT(lo_weight != nullptr, "compute_holo_consistency_cuda requires lo_weight");
  TENRYU_ASSERT(dt > 0.0, "compute_holo_consistency_cuda requires dt > 0");

  constexpr int kBlock = 256;
  const int n_total = n_cells * n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid > 0) {
    compute_holo_consistency_kernel<<<grid, kBlock>>>(consistency_source,
                                                      E_HO,
                                                      E_LO_pred,
                                                      E_old,
                                                      face_current_step,
                                                      reference_face_current,
                                                      sigma_R,
                                                      vol,
                                                      node_r,
                                                      lo_weight,
                                                      n_cells,
                                                      n_groups,
                                                      dt);
    cuda_check(cudaGetLastError(), "compute_holo_consistency kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "compute_holo_consistency kernel execution failed");
  }
}

void difference_reproject_residual_cuda(double* residual_E,
                                        const double* rad_E,
                                        const double* E_ref,
                                        const int n_cells,
                                        const int n_groups,
                                        const std::uint8_t* core_mask) {
  TENRYU_ASSERT(residual_E != nullptr,
                "difference_reproject_residual_cuda requires residual_E");
  TENRYU_ASSERT(rad_E != nullptr, "difference_reproject_residual_cuda requires rad_E");
  TENRYU_ASSERT(E_ref != nullptr, "difference_reproject_residual_cuda requires E_ref");

  constexpr int kBlock = 256;
  const int n_total = n_cells * n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid > 0) {
    difference_reproject_residual_kernel<<<grid, kBlock>>>(residual_E,
                                                           rad_E,
                                                           E_ref,
                                                           core_mask,
                                                           n_cells,
                                                           n_groups);
    cuda_check(cudaGetLastError(),
               "difference_reproject_residual kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "difference_reproject_residual kernel execution failed");
  }
}

void holo_prr_finalize_cuda(double* Prr,
                            double* chi,
                            double* coverage,
                            const double* Prr_tally,
                            const double* coverage_tally,
                            const double* rad_E_tally,
                            const double* vol,
                            const std::uint8_t* holo_core,
                            const int n_cells,
                            const int n_groups,
                            const double dt,
                            const double E_floor) {
  TENRYU_ASSERT(Prr != nullptr, "holo_prr_finalize_cuda requires Prr");
  TENRYU_ASSERT(chi != nullptr, "holo_prr_finalize_cuda requires chi");
  TENRYU_ASSERT(coverage != nullptr, "holo_prr_finalize_cuda requires coverage");
  TENRYU_ASSERT(Prr_tally != nullptr, "holo_prr_finalize_cuda requires Prr_tally");
  TENRYU_ASSERT(coverage_tally != nullptr,
                "holo_prr_finalize_cuda requires coverage_tally");
  TENRYU_ASSERT(rad_E_tally != nullptr,
                "holo_prr_finalize_cuda requires rad_E_tally");
  TENRYU_ASSERT(vol != nullptr, "holo_prr_finalize_cuda requires vol");

  constexpr int kBlock = 256;
  const int n_total = n_cells * n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid > 0) {
    holo_prr_finalize_kernel<<<grid, kBlock>>>(Prr,
                                               chi,
                                               coverage,
                                               Prr_tally,
                                               coverage_tally,
                                               rad_E_tally,
                                               vol,
                                               holo_core,
                                               n_cells,
                                               n_groups,
                                               dt,
                                               E_floor);
    cuda_check(cudaGetLastError(), "holo_prr_finalize kernel launch failed");
    cuda_check(cudaDeviceSynchronize(), "holo_prr_finalize kernel execution failed");
  }
}

}  // namespace tenryu::radiation
