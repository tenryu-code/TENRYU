#include "radiation/imc_transport_persistent.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "parallel/partition.hpp"
#include "radiation/interface.hpp"

namespace tenryu::radiation {
namespace {

constexpr unsigned kFullMask = 0xFFFFFFFFu;
constexpr double kGeomEps = 1.0e-12;
constexpr int kMaxEvents = 10000;
constexpr double kInf = INFINITY;
constexpr double kRngEps = 1.0e-16;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

struct RankBoundaryParams1D {
  int ghost_layers = 0;
  int nr_local = 0;
  std::int8_t has_left_boundary = 0;
  std::int8_t has_right_boundary = 0;
};

inline RankBoundaryParams1D make_rank_boundary_params_1d(
    const tenryu::parallel::PartitionInfo* part) {
  RankBoundaryParams1D out{};
  if (part == nullptr) {
    return out;
  }
  out.ghost_layers = std::max(part->ghost_layers, 0);
  out.nr_local = std::max(part->nr_local, 0);
  out.has_left_boundary = part->has_left_boundary() ? static_cast<std::int8_t>(1)
                                                     : static_cast<std::int8_t>(0);
  out.has_right_boundary = part->has_right_boundary() ? static_cast<std::int8_t>(1)
                                                       : static_cast<std::int8_t>(0);
  return out;
}

__device__ inline double rng_uniform(curandStatePhilox4_32_10_t* rng,
                                     std::uint32_t* counter) {
  *counter += 1U;
  return fmin(fmax(curand_uniform_double(rng), kRngEps), 1.0 - kRngEps);
}

__device__ inline double compute_tau_saturated_device(const double sigma,
                                                      const double length) {
  const double sigma_pos = fmax(sigma, 0.0);
  const double length_pos = fmax(length, 0.0);
  if (sigma_pos <= 0.0 || length_pos <= 0.0) {
    return 0.0;
  }
  return fmin(sigma_pos * length_pos, DBL_MAX);
}

__device__ inline double sphere_intersection_distance(const double r,
                                                      const double mu,
                                                      const double r_face) {
  const double b = mu * r;
  double D = b * b - (r * r - r_face * r_face);
  if (D < -kGeomEps * kGeomEps) {
    return kInf;
  }
  D = fmax(D, 0.0);
  const double sqrtD = sqrt(D);

  double s1 = kInf;
  double s2 = kInf;
  const double sign_b = (b >= 0.0) ? 1.0 : -1.0;
  const double q = -(b + sign_b * sqrtD);
  if (fabs(q) > 0.0) {
    s1 = q;
    s2 = (r * r - r_face * r_face) / q;
  } else {
    s1 = sqrtD;
  }

  double s = kInf;
  if (s1 > kGeomEps) {
    s = s1;
  }
  if (s2 > kGeomEps) {
    s = fmin(s, s2);
  }
  return s;
}

__device__ inline double boundary_distance_1d(const double r,
                                              const double mu,
                                              const double r_lo,
                                              const double r_hi,
                                              int* hit_face) {
  *hit_face = 1;
  double s_outer = sphere_intersection_distance(r, mu, r_hi);
  double s_inner = kInf;
  if (r_lo > 0.0 && mu < 0.0) {
    s_inner = sphere_intersection_distance(r, mu, r_lo);
  }

  if (s_inner < s_outer) {
    *hit_face = 0;
    return s_inner;
  }
  return s_outer;
}

__device__ inline double atomic_add_double(double* address, const double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* address_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_ull;
  unsigned long long int assumed = 0ULL;
  do {
    assumed = old;
    old = atomicCAS(address_ull,
                    assumed,
                    __double_as_longlong(
                        value + __longlong_as_double(static_cast<long long>(assumed))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline void tally_face_current(double* __restrict__ face_current_step,
                                          const int face_idx,
                                          const int group,
                                          const int n_cells,
                                          const int n_groups,
                                          const double signed_energy) {
  if (face_current_step == nullptr || face_idx < 0 || face_idx > n_cells ||
      group < 0 || group >= n_groups || !isfinite(signed_energy)) {
    return;
  }
  atomic_add_double(&face_current_step[face_idx * n_groups + group], signed_energy);
}

__device__ inline bool destination_is_diffusion_cell(
    const std::uint8_t* __restrict__ diff_cell,
    const int dest_cell,
    const int n_cells) {
  return diff_cell != nullptr && dest_cell >= 0 && dest_cell < n_cells &&
         diff_cell[dest_cell] != 0U;
}

__device__ inline void tally_diffusion_entry(
    double* __restrict__ diff_face_current_in,
    double* __restrict__ face_current_step,
    unsigned long long* __restrict__ diffusion_interface_kills,
    const int face_idx,
    const int group,
    const int n_cells,
    const int n_groups,
    const double packet_energy,
    const double signed_energy) {
  if (face_idx < 0 || face_idx > n_cells || group < 0 || group >= n_groups ||
      !isfinite(packet_energy) || !(packet_energy > 0.0)) {
    return;
  }
  if (diff_face_current_in != nullptr) {
    atomic_add_double(&diff_face_current_in[face_idx * n_groups + group], packet_energy);
  }
  tally_face_current(face_current_step, face_idx, group, n_cells, n_groups, signed_energy);
  if (diffusion_interface_kills != nullptr) {
    atomicAdd(diffusion_interface_kills, 1ULL);
  }
}

__device__ inline void atomic_inc_counter(unsigned long long* counter) {
  if (counter != nullptr) {
    atomicAdd(counter, 1ULL);
  }
}

__device__ inline void sample_isotropic_half_space(curandStatePhilox4_32_10_t* rng,
                                                   std::uint32_t* counter,
                                                   const double normal_sign,
                                                   double* mu,
                                                   double* dir_z,
                                                   double* dir_phi) {
  const double xi_mu = rng_uniform(rng, counter);
  const double mu_abs = sqrt(fmax(xi_mu, 0.0));
  const double xi_phi = rng_uniform(rng, counter);
  const double phi = 2.0 * 3.14159265358979323846 * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu_abs * mu_abs));
  *mu = normal_sign * mu_abs;
  *dir_z = sin_theta * cos(phi);
  *dir_phi = sin_theta * sin(phi);
}

__device__ inline void sample_isotropic_direction(curandStatePhilox4_32_10_t* rng,
                                                  std::uint32_t* counter,
                                                  double* mu,
                                                  double* dir_z,
                                                  double* dir_phi) {
  const double xi_mu = rng_uniform(rng, counter);
  const double xi_phi = rng_uniform(rng, counter);
  const double mu_new = 2.0 * xi_mu - 1.0;
  const double phi = 2.0 * 3.14159265358979323846 * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu_new * mu_new));
  *mu = mu_new;
  *dir_z = sin_theta * cos(phi);
  *dir_phi = sin_theta * sin(phi);
}

__device__ inline double pgrw_inverse_cdf_lookup(
    const double xi,
    const double* __restrict__ theta_table,
    const double* __restrict__ xi_table,
    const int table_size,
    const double theta_max) {
  if (theta_table == nullptr || xi_table == nullptr || table_size <= 0) {
    return 0.0;
  }
  if (!(xi > 0.0)) {
    return 0.0;
  }

  const double xi_last = xi_table[table_size - 1];
  if (xi >= xi_last) {
    return fmin(theta_max, theta_table[table_size - 1]);
  }

  const double xi_first = xi_table[0];
  const double theta_first = theta_table[0];
  if (xi <= xi_first) {
    if (!(xi_first > 0.0) || !(theta_first > 0.0)) {
      return 0.0;
    }
    return fmin(theta_first * xi / xi_first, theta_first);
  }

  int lo = 0;
  int hi = table_size - 1;
  while (hi - lo > 1) {
    const int mid = lo + (hi - lo) / 2;
    if (xi_table[mid] >= xi) {
      hi = mid;
    } else {
      lo = mid;
    }
  }

  const double xi0 = xi_table[lo];
  const double xi1 = xi_table[hi];
  const double theta0 = theta_table[lo];
  const double theta1 = theta_table[hi];
  const double denom = xi1 - xi0;
  if (!(denom > 0.0)) {
    return theta0;
  }
  const double w = (xi - xi0) / denom;
  return theta0 + w * (theta1 - theta0);
}

__device__ inline double pgrw_inverse_position_row(
    const double xi,
    const double* __restrict__ row_cdf,
    const int rho_bins) {
  if (row_cdf == nullptr || rho_bins <= 1) {
    return 0.0;
  }

  const double denom_bins = static_cast<double>(rho_bins - 1);
  if (!(xi > 0.0)) {
    return 0.0;
  }
  if (xi >= row_cdf[rho_bins - 1]) {
    return 1.0;
  }
  if (xi <= row_cdf[0]) {
    return 0.0;
  }

  int lo = 0;
  int hi = rho_bins - 1;
  while (hi - lo > 1) {
    const int mid = lo + (hi - lo) / 2;
    if (row_cdf[mid] >= xi) {
      hi = mid;
    } else {
      lo = mid;
    }
  }

  const double c0 = row_cdf[lo];
  const double c1 = row_cdf[hi];
  const double rho0 = static_cast<double>(lo) / denom_bins;
  const double rho1 = static_cast<double>(hi) / denom_bins;
  const double denom = c1 - c0;
  if (!(denom > 0.0)) {
    return rho0;
  }
  const double w = (xi - c0) / denom;
  return rho0 + w * (rho1 - rho0);
}

__device__ inline double pgrw_sample_position(
    curandStatePhilox4_32_10_t* rng,
    std::uint32_t* counter,
    const double t_event,
    const double R0,
    const double D,
    const double* __restrict__ pos_cdf,
    const int theta_bins,
    const int rho_bins,
    const double theta_max) {
  if (pos_cdf == nullptr || theta_bins <= 1 || rho_bins <= 1 || !(t_event > 0.0) ||
      !(R0 > 0.0) || !(D > 0.0) || !(theta_max > 0.0)) {
    return 0.0;
  }

  const double theta =
      fmin(theta_max,
           tenryu::core::constants::c_light * D * t_event / fmax(R0 * R0, 1.0e-300));
  const double theta_step = theta_max / static_cast<double>(theta_bins - 1);
  if (!(theta > theta_step)) {
    return 0.0;
  }

  const double xi = rng_uniform(rng, counter);
  if (theta >= theta_max) {
    const double rho_frac = pgrw_inverse_position_row(
        xi, pos_cdf + static_cast<std::size_t>(theta_bins - 1) * static_cast<std::size_t>(rho_bins),
        rho_bins);
    return fmin(fmax(rho_frac, 0.0), 1.0) * R0;
  }

  int theta_lo = static_cast<int>(theta / theta_step);
  if (theta_lo < 1) {
    theta_lo = 1;
  }
  if (theta_lo > theta_bins - 2) {
    theta_lo = theta_bins - 2;
  }
  const int theta_hi = theta_lo + 1;
  const double theta0 = theta_step * static_cast<double>(theta_lo);
  const double w = (theta - theta0) / theta_step;
  const double rho_lo = pgrw_inverse_position_row(
      xi,
      pos_cdf + static_cast<std::size_t>(theta_lo) * static_cast<std::size_t>(rho_bins),
      rho_bins);
  const double rho_hi = pgrw_inverse_position_row(
      xi,
      pos_cdf + static_cast<std::size_t>(theta_hi) * static_cast<std::size_t>(rho_bins),
      rho_bins);
  const double rho_frac = rho_lo + w * (rho_hi - rho_lo);
  return fmin(fmax(rho_frac, 0.0), 1.0) * R0;
}

__device__ inline void warp_tally(double* rad_dep,
                                  double* rad_E_tally,
                                  double* holo_Prr_tally,
                                  double* holo_Prr_coverage_tally,
                                  const int key,
                                  const double dep,
                                  const double tl,
                                  const double prr,
                                  const double prr_coverage) {
#if __CUDA_ARCH__ >= 700
  const unsigned mask = __activemask();
  const int lane = threadIdx.x & 31;
  const unsigned peers = __match_any_sync(mask, key);
  const unsigned lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
  const int peer_rank = __popc(peers & lane_mask);
  const int peer_count = __popc(peers);

  double sum_dep = dep;
  double sum_tl = tl;
  double sum_prr = prr;
  double sum_prr_coverage = prr_coverage;
  // Reduce by peer rank, not physical lane distance, so sparse peer masks are safe.
  for (int offset = 1; offset < peer_count; offset <<= 1) {
    const int src_rank = peer_rank + offset;
    const bool has_src = src_rank < peer_count;
    unsigned src_lane = static_cast<unsigned>(lane);
    if (has_src) {
      const unsigned mapped = __fns(peers, 0u, src_rank + 1);
      src_lane = (mapped < 32u) ? mapped : static_cast<unsigned>(lane);
    }
    const double tmp_dep = __shfl_sync(peers, sum_dep, static_cast<int>(src_lane));
    const double tmp_tl = __shfl_sync(peers, sum_tl, static_cast<int>(src_lane));
    const double tmp_prr = __shfl_sync(peers, sum_prr, static_cast<int>(src_lane));
    const double tmp_prr_coverage =
        __shfl_sync(peers, sum_prr_coverage, static_cast<int>(src_lane));
    if (has_src) {
      sum_dep += tmp_dep;
      sum_tl += tmp_tl;
      sum_prr += tmp_prr;
      sum_prr_coverage += tmp_prr_coverage;
    }
  }

  if (peer_rank == 0) {
    if (sum_dep != 0.0) {
      atomic_add_double(&rad_dep[key], sum_dep);
    }
    if (sum_tl != 0.0) {
      atomic_add_double(&rad_E_tally[key], sum_tl);
    }
    if (holo_Prr_tally != nullptr && sum_prr != 0.0) {
      atomic_add_double(&holo_Prr_tally[key], sum_prr);
    }
    if (holo_Prr_coverage_tally != nullptr && sum_prr_coverage != 0.0) {
      atomic_add_double(&holo_Prr_coverage_tally[key], sum_prr_coverage);
    }
  }
#else
  if (dep != 0.0) {
    atomic_add_double(&rad_dep[key], dep);
  }
  if (tl != 0.0) {
    atomic_add_double(&rad_E_tally[key], tl);
  }
  if (holo_Prr_tally != nullptr && prr != 0.0) {
    atomic_add_double(&holo_Prr_tally[key], prr);
  }
  if (holo_Prr_coverage_tally != nullptr && prr_coverage != 0.0) {
    atomic_add_double(&holo_Prr_coverage_tally[key], prr_coverage);
  }
#endif
}

__device__ inline bool has_non_finite_imc_state_1d(const double r,
                                                    const double z,
                                                    const double mu,
                                                    const double dir_z,
                                                    const double dir_phi,
                                                    const double E) {
  return !isfinite(r) || !isfinite(z) || !isfinite(mu) || !isfinite(dir_z) ||
         !isfinite(dir_phi) || !isfinite(E);
}

__device__ inline void mark_nan_particle(tenryu::core::DeviceErrorFlags* error_flags,
                                         double* E_numerical_loss,
                                         const std::int8_t sign,
                                         double* E) {
  if (error_flags != nullptr) {
    atomicExch(&error_flags->nan_particle, 1);
  }
  if (E_numerical_loss != nullptr && isfinite(*E) && *E > 0.0) {
    atomic_add_double(E_numerical_loss, static_cast<double>(sign) * (*E));
  }
  *E = 0.0;
}

__device__ inline int hydro_owner_from_rad_position_1d(
    const int rad_cell,
    const double pos_r,
    const double* __restrict__ hydro_node_r,
    const int32_t* __restrict__ rad_h_begin,
    const int32_t* __restrict__ rad_h_end) {
  if (hydro_node_r == nullptr || rad_h_begin == nullptr || rad_h_end == nullptr || rad_cell < 0) {
    return -1;
  }

  const int h_begin = rad_h_begin[rad_cell];
  const int h_end = rad_h_end[rad_cell];
  if (h_begin < 0 || h_end <= h_begin) {
    return -1;
  }
  if (h_end - h_begin <= 1) {
    return h_begin;
  }
  if (!isfinite(pos_r)) {
    return h_begin;
  }

  int lo = h_begin;
  int hi = h_end - 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (hydro_node_r[mid + 1] <= pos_r) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  if (lo < h_begin) {
    lo = h_begin;
  }
  if (lo >= h_end) {
    lo = h_end - 1;
  }
  return lo;
}

__device__ inline std::uint16_t sample_group_from_cdf_device(
    const double* __restrict__ cdf,
    const int n_groups,
    const double xi_g) {
  int g_new = n_groups - 1;
  const double cdf_first = fmin(fmax(cdf[0], 0.0), 1.0);
  const double cdf_last = fmin(fmax(cdf[n_groups - 1], 0.0), 1.0);
  if (xi_g <= cdf_first) {
    g_new = 0;
  } else if (xi_g >= cdf_last) {
    g_new = n_groups - 1;
  } else {
    int lo = 0;
    int hi = n_groups - 1;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      const double cdf_mid = fmin(fmax(cdf[mid], 0.0), 1.0);
      if (cdf_mid >= xi_g) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    g_new = lo;
  }
  return static_cast<std::uint16_t>(g_new);
}

__device__ inline double cdf_row_probability_bias(const double* __restrict__ row_cdf,
                                                  const int group) {
  const double hi = fmin(fmax(row_cdf[group], 0.0), 1.0);
  const double lo =
      (group > 0) ? fmin(fmax(row_cdf[group - 1], 0.0), 1.0) : 0.0;
  return fmax(hi - lo, 0.0);
}

__device__ inline bool sample_group_from_cdf_range_device(
    const double* __restrict__ cdf,
    const int n_groups,
    const int g_begin,
    const double xi_g,
    std::uint16_t* g_new) {
  if (cdf == nullptr || g_new == nullptr || n_groups <= 0 || g_begin < 0 ||
      g_begin >= n_groups) {
    return false;
  }

  const double cdf_base =
      (g_begin > 0) ? fmin(fmax(cdf[g_begin - 1], 0.0), 1.0) : 0.0;
  const double cdf_last = fmin(fmax(cdf[n_groups - 1], 0.0), 1.0);
  if (!(cdf_last > cdf_base)) {
    return false;
  }

  const double xi_scaled = cdf_base + (cdf_last - cdf_base) * fmin(fmax(xi_g, 0.0), 1.0);
  int lo = g_begin;
  int hi = n_groups - 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    const double cdf_mid = fmin(fmax(cdf[mid], 0.0), 1.0);
    if (cdf_mid >= xi_scaled) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  *g_new = static_cast<std::uint16_t>(lo);
  return true;
}

template <bool kEnableInterfaceConversion>
__global__ __launch_bounds__(128, 8) void imc_transport_kernel(
    double* __restrict__ pos_r_arr,
    double* __restrict__ pos_z_arr,
    double* __restrict__ dir_r_arr,
    double* __restrict__ dir_z_arr,
    double* __restrict__ dir_phi_arr,
    double* __restrict__ energy_arr,
    double* __restrict__ birth_energy_arr,
    double* __restrict__ time_remain_arr,
    const std::int8_t* __restrict__ sign_arr,
    std::uint64_t* __restrict__ global_id_arr,
    std::uint32_t* __restrict__ rng_counter_arr,
    std::int32_t* __restrict__ cell_id_arr,
    std::uint16_t* __restrict__ group_id_arr,
    std::uint8_t* __restrict__ mode_arr,
    std::uint8_t* __restrict__ alive_arr,
    const double* __restrict__ sigma_a_eff,
    const double* __restrict__ sigma_s_eff,
    const double* __restrict__ Te,
    const double* __restrict__ vol,
    double* __restrict__ sloc_abs_wr,
    double* __restrict__ sloc_abs_wr2,
    double* __restrict__ sloc_abs_E,
    const double* __restrict__ node_r,
    const TransportMode* __restrict__ ddmc_mode,
    const int ddmc_zero_flux_interfaces,
    const int n_groups_for_mode,
    const int emissivity_preserving,
    const double* __restrict__ sigma_R,
    const double* __restrict__ cell_dx,
    const double* __restrict__ fleck_f,
    const double* __restrict__ sigma_a_raw,
    const double* __restrict__ eta_cdf,
    const double* __restrict__ eta_cdf_hydro,
    const double* __restrict__ scatter_bias_cdf,
    const double* __restrict__ hydro_node_r,
    const int32_t* __restrict__ rad_h_begin,
    const int32_t* __restrict__ rad_h_end,
    const int* __restrict__ g_diff_end,
    const double* __restrict__ sigma_a_bar,
    const double* __restrict__ sigma_t_bar,
    const double* __restrict__ D_pgrw,
    const double* __restrict__ gamma_pgrw,
    const double* __restrict__ pgrw_leak_inv_cdf,
    const double* __restrict__ pgrw_leak_cdf_xi,
    const double* __restrict__ pgrw_pos_cdf,
    const int pgrw_leak_table_size,
    const int pgrw_pos_theta_bins,
    const int pgrw_pos_rho_bins,
    const double pgrw_theta_max,
    const double pgrw_tau_rw,
    const PlanckTableDeviceView planck,
    const int inelastic_scatter,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_E_tally,
    double* __restrict__ holo_Prr_tally,
    double* __restrict__ holo_Prr_coverage_tally,
    double* __restrict__ face_current_step,
    const std::uint8_t* __restrict__ diff_cell,
    double* __restrict__ diff_face_current_in,
    double* __restrict__ E_escape,
    double* __restrict__ E_numerical_loss,
    unsigned long long* __restrict__ imc_absorbed,
    unsigned long long* __restrict__ imc_escaped,
    unsigned long long* __restrict__ diffusion_interface_kills,
    unsigned long long* __restrict__ interface_transitions,
    unsigned long long* __restrict__ interface_reflections,
    unsigned long long* __restrict__ conversion_prob_violations,
    unsigned long long* __restrict__ cnt_boundary,
    unsigned long long* __restrict__ cnt_scatter,
    unsigned long long* __restrict__ cnt_census,
    unsigned long long* __restrict__ cnt_absorb_kill,
    unsigned long long* __restrict__ cnt_absorb_survive,
    unsigned long long* __restrict__ cnt_roulette_kill,
    int* __restrict__ global_counter,
    const int n_cells,
    const int n_groups,
    const int n_imc,
    const double dt,
    const double E_avg,
    const double w_cutoff,
    const double p_survival,
    const double f_cutoff,
    const int tail_pass,
    const int bc_inner,
    const int bc_outer,
    const std::uint64_t step_number,
    const std::uint64_t user_seed,
    const int ghost_layers,
    const int nr_local,
    const std::int8_t has_left_boundary,
    const std::int8_t has_right_boundary,
    tenryu::core::DeviceErrorFlags* __restrict__ error_flags) {
  const int lane = threadIdx.x & 31;

  int my_idx = -1;
  bool active = false;
  int events = 0;

  double r_l = 0.0;
  double z_l = 0.0;
  double mu_l = 0.0;
  double dir_z_l = 0.0;
  double dir_phi_l = 0.0;
  double E_l = 0.0;
  double birth_E_l = 0.0;
  double t_remain_l = 0.0;
  double tau_scatter_remain = 0.0;
  std::int32_t cell_l = -1;
  std::int8_t sign_l = 1;
  // group_l is uint16_t for register pressure; n_groups > 65535 is not supported.
  // Config validation should enforce this limit.
  std::uint16_t group_l = 0;
  std::uint8_t alive_l = kAlive;
  std::uint64_t gid_l = 0;
  std::uint32_t rng_counter_l = 0;
  std::uint32_t local_cnt_boundary = 0;
  std::uint32_t local_cnt_scatter = 0;
  std::uint32_t local_cnt_census = 0;
  std::uint32_t local_cnt_absorb_kill = 0;
  std::uint32_t local_cnt_absorb_survive = 0;
  std::uint32_t local_cnt_roulette_kill = 0;
  curandStatePhilox4_32_10_t rng;

  while (true) {
    // Refill inactive lanes.
    const unsigned need = __ballot_sync(kFullMask, !active);
    const int n_need = __popc(need);

    int base = 0;
    if (lane == 0 && n_need > 0) {
      base = atomicAdd(global_counter, n_need);
    }
    base = __shfl_sync(kFullMask, base, 0);

    if (!active) {
      const unsigned lane_mask = (lane == 0) ? 0u : ((1u << lane) - 1u);
      const int offset = __popc(need & lane_mask);
      const int idx = base + offset;
      if (offset < n_need && idx < n_imc) {
        my_idx = idx;
        local_cnt_boundary = 0;
        local_cnt_scatter = 0;
        local_cnt_census = 0;
        local_cnt_absorb_kill = 0;
        local_cnt_absorb_survive = 0;
        local_cnt_roulette_kill = 0;
        r_l = pos_r_arr[my_idx];
        z_l = pos_z_arr[my_idx];
        mu_l = dir_r_arr[my_idx];
        dir_z_l = dir_z_arr[my_idx];
        dir_phi_l = dir_phi_arr[my_idx];
        E_l = energy_arr[my_idx];
        birth_E_l = birth_energy_arr[my_idx];
        t_remain_l = time_remain_arr[my_idx];
        sign_l = sign_arr[my_idx];
        if (tail_pass != 0 && t_remain_l <= 0.0) {
          alive_l = alive_arr[my_idx];
          active = false;
        } else if (t_remain_l <= 0.0) {
          t_remain_l = dt;
        }
        gid_l = global_id_arr[my_idx];
        rng_counter_l = rng_counter_arr[my_idx];
        cell_l = cell_id_arr[my_idx];
        group_l = group_id_arr[my_idx];
        alive_l = alive_arr[my_idx];
        if (alive_l != kAlive || (tail_pass != 0 && t_remain_l <= 0.0)) {
          alive_l = kDead;
          active = false;
        } else {
          curand_init(gid_l ^ user_seed,
                      static_cast<unsigned long long>(step_number),
                      static_cast<unsigned long long>(rng_counter_l),
                      &rng);
          tau_scatter_remain = -log(rng_uniform(&rng, &rng_counter_l));
          events = 0;
          active = true;
        }
      }
    }

    const bool any_active = __any_sync(kFullMask, active);
    if (!any_active) {
      const unsigned active_mask = __activemask();
      const int consumed = atomicAdd(global_counter, 0);
      const unsigned done_mask = __ballot_sync(active_mask, consumed >= n_imc);
      if (done_mask == active_mask) {
        break;
      }
    }

    if (active) {
      events += 1;
      if (events >= kMaxEvents) {
        if (error_flags != nullptr) {
          atomicAdd(&error_flags->infinite_loop, 1);
        }
        if (E_l > 0.0 && E_numerical_loss != nullptr) {
          atomic_add_double(E_numerical_loss, static_cast<double>(sign_l) * E_l);
        }
        if (E_l > 0.0) {
          E_l = 0.0;
        }
        t_remain_l = 0.0;
        alive_l = kDead;
        if (cnt_boundary != nullptr && local_cnt_boundary > 0) {
          atomicAdd(cnt_boundary, static_cast<unsigned long long>(local_cnt_boundary));
        }
        if (cnt_scatter != nullptr && local_cnt_scatter > 0) {
          atomicAdd(cnt_scatter, static_cast<unsigned long long>(local_cnt_scatter));
        }
        if (cnt_census != nullptr && local_cnt_census > 0) {
          atomicAdd(cnt_census, static_cast<unsigned long long>(local_cnt_census));
        }
        if (cnt_absorb_kill != nullptr && local_cnt_absorb_kill > 0) {
          atomicAdd(cnt_absorb_kill, static_cast<unsigned long long>(local_cnt_absorb_kill));
        }
        if (cnt_absorb_survive != nullptr && local_cnt_absorb_survive > 0) {
          atomicAdd(cnt_absorb_survive,
                    static_cast<unsigned long long>(local_cnt_absorb_survive));
        }
        if (cnt_roulette_kill != nullptr && local_cnt_roulette_kill > 0) {
          atomicAdd(cnt_roulette_kill, static_cast<unsigned long long>(local_cnt_roulette_kill));
        }
        pos_r_arr[my_idx] = r_l;
        pos_z_arr[my_idx] = z_l;
        dir_r_arr[my_idx] = mu_l;
        dir_z_arr[my_idx] = dir_z_l;
        dir_phi_arr[my_idx] = dir_phi_l;
        energy_arr[my_idx] = E_l;
        time_remain_arr[my_idx] = t_remain_l;
        birth_energy_arr[my_idx] = birth_E_l;
        cell_id_arr[my_idx] = cell_l;
        group_id_arr[my_idx] = group_l;
        alive_arr[my_idx] = alive_l;
        rng_counter_arr[my_idx] = rng_counter_l;
        my_idx = -1;
        active = false;
      }
    }

    if (active) {
      if (cell_l < 0 || cell_l >= n_cells ||
          group_l >= static_cast<unsigned>(n_groups)) {
        if (error_flags != nullptr) {
          atomicExch(&error_flags->invalid_cell, 1);
        }
        if (E_l > 0.0 && E_numerical_loss != nullptr) {
          atomic_add_double(E_numerical_loss, static_cast<double>(sign_l) * E_l);
        }
        E_l = 0.0;
        t_remain_l = 0.0;
        alive_l = kDead;
        if (cnt_boundary != nullptr && local_cnt_boundary > 0) {
          atomicAdd(cnt_boundary, static_cast<unsigned long long>(local_cnt_boundary));
        }
        if (cnt_scatter != nullptr && local_cnt_scatter > 0) {
          atomicAdd(cnt_scatter, static_cast<unsigned long long>(local_cnt_scatter));
        }
        if (cnt_census != nullptr && local_cnt_census > 0) {
          atomicAdd(cnt_census, static_cast<unsigned long long>(local_cnt_census));
        }
        if (cnt_absorb_kill != nullptr && local_cnt_absorb_kill > 0) {
          atomicAdd(cnt_absorb_kill, static_cast<unsigned long long>(local_cnt_absorb_kill));
        }
        if (cnt_absorb_survive != nullptr && local_cnt_absorb_survive > 0) {
          atomicAdd(cnt_absorb_survive,
                    static_cast<unsigned long long>(local_cnt_absorb_survive));
        }
        if (cnt_roulette_kill != nullptr && local_cnt_roulette_kill > 0) {
          atomicAdd(cnt_roulette_kill, static_cast<unsigned long long>(local_cnt_roulette_kill));
        }
        pos_r_arr[my_idx] = r_l;
        pos_z_arr[my_idx] = z_l;
        dir_r_arr[my_idx] = mu_l;
        dir_z_arr[my_idx] = dir_z_l;
        dir_phi_arr[my_idx] = dir_phi_l;
        energy_arr[my_idx] = E_l;
        time_remain_arr[my_idx] = t_remain_l;
        birth_energy_arr[my_idx] = birth_E_l;
        cell_id_arr[my_idx] = cell_l;
        group_id_arr[my_idx] = group_l;
        alive_arr[my_idx] = alive_l;
        rng_counter_arr[my_idx] = rng_counter_l;
        my_idx = -1;
        active = false;
      }
    }

    if (active) {
      const int key = cell_l * n_groups + static_cast<int>(group_l);
      const double sigma_a = sigma_a_eff[key];
      const double sigma_s = sigma_s_eff[key];
      int hit_face = 1;
      const double r_lo = node_r[cell_l];
      const double r_hi = node_r[cell_l + 1];
      const double s_bdry = boundary_distance_1d(r_l, mu_l, r_lo, r_hi, &hit_face);
      const double s_cen = tenryu::core::constants::c_light * t_remain_l;
      const double s_scatter = (sigma_s > 0.0)
                                   ? ((tau_scatter_remain > 1.0e-14)
                                          ? (tau_scatter_remain / sigma_s)
                                          : 0.0)
                                   : kInf;
      bool used_pgrw = false;
      if (pgrw_tau_rw > 0.0 && fleck_f != nullptr && g_diff_end != nullptr &&
          sigma_a_bar != nullptr && sigma_t_bar != nullptr && D_pgrw != nullptr &&
          gamma_pgrw != nullptr && pgrw_leak_inv_cdf != nullptr &&
          pgrw_leak_cdf_xi != nullptr && pgrw_pos_cdf != nullptr &&
          static_cast<int>(group_l) < g_diff_end[cell_l] && g_diff_end[cell_l] > 0) {
        const double R0 = (r_lo > 0.0) ? fmin(r_l - r_lo, r_hi - r_l) : (r_hi - r_l);
        const double sigma_t_cell = sigma_t_bar[cell_l];
        const double D_cell = D_pgrw[cell_l];
        const double tau_sphere = sigma_t_cell * R0;
        const double sigma_t_group = fmax(sigma_a + sigma_s, 0.0);
        const double mfp = 1.0 / fmax(sigma_t_group, 1.0e-30);
        if (R0 > kGeomEps && sigma_t_cell > 0.0 && D_cell > 0.0 &&
            R0 > mfp && tau_sphere >= pgrw_tau_rw &&
            tenryu::core::constants::c_light * t_remain_l > R0) {
          used_pgrw = true;

          double t_up = kInf;
          const double sigma_a_cell = fmax(sigma_a_bar[cell_l], 0.0);
          const double gamma_cell = clamp01(gamma_pgrw[cell_l]);
          const double f_cell = clamp01(fleck_f[cell_l]);
          if (g_diff_end[cell_l] < n_groups && gamma_cell < 1.0 - 1.0e-12) {
            const double rate_up = (1.0 - f_cell) * (1.0 - gamma_cell) * sigma_a_cell *
                                   tenryu::core::constants::c_light;
            if (rate_up > 0.0) {
              t_up = -log(rng_uniform(&rng, &rng_counter_l)) / rate_up;
            }
          }

          const double theta_leak =
              pgrw_inverse_cdf_lookup(rng_uniform(&rng, &rng_counter_l),
                                      pgrw_leak_inv_cdf,
                                      pgrw_leak_cdf_xi,
                                      pgrw_leak_table_size,
                                      pgrw_theta_max);
          const double t_leak = theta_leak * R0 * R0 /
                                (tenryu::core::constants::c_light * fmax(D_cell, 1.0e-30));

          enum PgrwEventType { kPgrwUp = 0, kPgrwCensus = 1, kPgrwLeak = 2 };
          int event_type = kPgrwLeak;
          double t_event = t_leak;
          if (t_up <= t_remain_l && t_up <= t_leak) {
            event_type = kPgrwUp;
            t_event = t_up;
          } else if (t_remain_l <= t_leak) {
            event_type = kPgrwCensus;
            t_event = t_remain_l;
          }

          const double tau_abs = tenryu::core::constants::c_light * f_cell * sigma_a_cell *
                                 fmax(t_event, 0.0);
          const double r_before = r_l;
          const double E_old = E_l;
          double dep = 0.0;
          if (tau_abs < 1.0e-6) {
            dep = E_old * tau_abs * (1.0 - 0.5 * tau_abs);
          } else {
            dep = -E_old * expm1(-tau_abs);
          }
          dep = fmin(fmax(dep, 0.0), E_old);
          E_l = E_old - dep;

          const double sigma_dep = f_cell * sigma_a_cell;
          const double tl_pgrw =
              (sigma_dep > 0.0) ? (dep / sigma_dep)
                                : (E_old * tenryu::core::constants::c_light * t_event);
          if (dep != 0.0 || tl_pgrw != 0.0) {
            const double sign_d = static_cast<double>(sign_l);
            warp_tally(rad_dep,
                       rad_E_tally,
                       holo_Prr_tally,
                       holo_Prr_coverage_tally,
                       key,
                       sign_d * dep,
                       sign_d * tl_pgrw,
                       0.0,
                       0.0);
          }

          if (event_type == kPgrwLeak) {
            const double r_center = r_l;
            const double mu_loc = 2.0 * rng_uniform(&rng, &rng_counter_l) - 1.0;
            const double r_new_sq = r_l * r_l + R0 * R0 + 2.0 * r_l * R0 * mu_loc;
            const double r_new = sqrt(fmax(r_new_sq, 0.0));
            const double leak_normal_sign = (r_new >= r_center) ? 1.0 : -1.0;
            const double r_lo_clamp = r_lo + kGeomEps;
            const double r_hi_clamp = r_hi - kGeomEps;
            r_l = r_new;
            if (r_hi_clamp > r_lo_clamp) {
              r_l = fmax(r_l, r_lo_clamp);
              r_l = fmin(r_l, r_hi_clamp);
            } else {
              r_l = 0.5 * (r_lo + r_hi);
            }
            sample_isotropic_half_space(
                &rng, &rng_counter_l, leak_normal_sign, &mu_l, &dir_z_l, &dir_phi_l);
            tau_scatter_remain = -log(rng_uniform(&rng, &rng_counter_l));
            t_remain_l = fmax(t_remain_l - t_event, 0.0);
          } else {
            const double rho_event =
                pgrw_sample_position(&rng,
                                     &rng_counter_l,
                                     t_event,
                                     R0,
                                     D_cell,
                                     pgrw_pos_cdf,
                                     pgrw_pos_theta_bins,
                                     pgrw_pos_rho_bins,
                                     pgrw_theta_max);
            const double mu_loc = 2.0 * rng_uniform(&rng, &rng_counter_l) - 1.0;
            const double r_new_sq =
                r_l * r_l + rho_event * rho_event + 2.0 * r_l * rho_event * mu_loc;
            const double r_lo_clamp = r_lo + kGeomEps;
            const double r_hi_clamp = r_hi - kGeomEps;
            r_l = sqrt(fmax(r_new_sq, 0.0));
            if (r_hi_clamp > r_lo_clamp) {
              r_l = fmax(r_l, r_lo_clamp);
              r_l = fmin(r_l, r_hi_clamp);
            } else {
              r_l = 0.5 * (r_lo + r_hi);
            }
            sample_isotropic_direction(&rng, &rng_counter_l, &mu_l, &dir_z_l, &dir_phi_l);
            if (event_type == kPgrwCensus) {
              ++local_cnt_census;
              t_remain_l = 0.0;
              alive_l = kAlive;
              active = false;
            } else {
              const int g_transport_begin = g_diff_end[cell_l];
              const int n_transport = n_groups - g_diff_end[cell_l];
              if (n_transport > 0) {
                bool sampled_transport_group = false;
                const int cell_base = cell_l * n_groups;
                if (scatter_bias_cdf != nullptr) {
                  std::uint16_t g_new = group_l;
                  sampled_transport_group = sample_group_from_cdf_range_device(
                      scatter_bias_cdf + cell_base,
                      n_groups,
                      g_transport_begin,
                      rng_uniform(&rng, &rng_counter_l),
                      &g_new);
                  if (sampled_transport_group) {
                    const double q_bias =
                        cdf_row_probability_bias(scatter_bias_cdf + cell_base, g_new);
                    const double cdf_range_hi =
                        fmin(fmax(scatter_bias_cdf[cell_base + n_groups - 1], 0.0), 1.0);
                    const double cdf_range_lo =
                        (g_transport_begin > 0)
                            ? fmin(fmax(scatter_bias_cdf[cell_base + g_transport_begin - 1], 0.0),
                                   1.0)
                            : 0.0;
                    const double q_bias_range =
                        (cdf_range_hi > cdf_range_lo) ? (q_bias / (cdf_range_hi - cdf_range_lo))
                                                      : 0.0;

                    const double Te_c = fmax(Te[cell_l], 1.0e-12);
                    const double* sigma_a_src =
                        (sigma_a_raw != nullptr) ? sigma_a_raw : sigma_a_eff;
                    double total_phys = 0.0;
                    for (int gg = g_transport_begin; gg < n_groups; ++gg) {
                      const double b_g =
                          (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(gg, Te_c), 0.0);
                      total_phys +=
                          fmax(sigma_a_src[cell_base + gg], 0.0) * b_g;
                    }
                    double p_phys = 0.0;
                    if (total_phys > 0.0) {
                      const double b_g =
                          (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g_new, Te_c), 0.0);
                      p_phys =
                          (fmax(sigma_a_src[cell_base + g_new], 0.0) * b_g) / total_phys;
                    }

                    if (q_bias_range > 1.0e-300 && p_phys > 0.0) {
                      E_l *= p_phys / q_bias_range;
                    }
                    group_l = g_new;
                  }
                } else if (eta_cdf != nullptr) {
                  std::uint16_t g_new = group_l;
                  sampled_transport_group = sample_group_from_cdf_range_device(
                      eta_cdf + cell_base,
                      n_groups,
                      g_transport_begin,
                      rng_uniform(&rng, &rng_counter_l),
                      &g_new);
                  if (sampled_transport_group) {
                    group_l = g_new;
                  }
                } else if (eta_cdf_hydro != nullptr && hydro_node_r != nullptr &&
                           rad_h_begin != nullptr && rad_h_end != nullptr) {
                  const int hydro_owner = hydro_owner_from_rad_position_1d(
                      cell_l, r_l, hydro_node_r, rad_h_begin, rad_h_end);
                  if (hydro_owner >= 0) {
                    std::uint16_t g_new = group_l;
                    const int hydro_base = hydro_owner * n_groups;
                    sampled_transport_group = sample_group_from_cdf_range_device(
                        eta_cdf_hydro + hydro_base,
                        n_groups,
                        g_transport_begin,
                        rng_uniform(&rng, &rng_counter_l),
                        &g_new);
                    if (sampled_transport_group) {
                      group_l = g_new;
                    }
                  }
                }

                if (!sampled_transport_group) {
                  const double Te_c = fmax(Te[cell_l], 1.0e-12);
                  double weight_total = 0.0;
                  for (int g = g_transport_begin; g < n_groups; ++g) {
                    const double sigma_abs_g = (sigma_a_raw != nullptr)
                                                   ? fmax(sigma_a_raw[cell_base + g], 0.0)
                                                   : fmax(sigma_a_eff[cell_base + g], 0.0);
                    const double b_g =
                        (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, Te_c), 0.0);
                    weight_total += sigma_abs_g * b_g;
                  }

                  if (weight_total > 0.0) {
                    const double xi_g = rng_uniform(&rng, &rng_counter_l) * weight_total;
                    double cdf = 0.0;
                    int g_new = g_transport_begin;
                    for (int g = g_transport_begin; g < n_groups; ++g) {
                      const double sigma_abs_g = (sigma_a_raw != nullptr)
                                                     ? fmax(sigma_a_raw[cell_base + g], 0.0)
                                                     : fmax(sigma_a_eff[cell_base + g], 0.0);
                      const double b_g =
                          (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, Te_c), 0.0);
                      cdf += sigma_abs_g * b_g;
                      if (xi_g <= cdf) {
                        g_new = g;
                        break;
                      }
                    }
                    group_l = static_cast<std::uint16_t>(g_new);
                    sampled_transport_group = true;
                  }
                }

                if (!sampled_transport_group) {
                  group_l = static_cast<std::uint16_t>(g_transport_begin);
                }
              }
              tau_scatter_remain = -log(rng_uniform(&rng, &rng_counter_l));
              t_remain_l = fmax(t_remain_l - t_event, 0.0);
            }
          }
          if (sloc_abs_wr != nullptr && sloc_abs_wr2 != nullptr && sloc_abs_E != nullptr) {
            const double dE = E_old - E_l;
            if (dE > 0.0) {
              const double r_mid = 0.5 * (r_before + r_l);
              int sloc_cell = cell_l;
              if (hydro_node_r != nullptr && rad_h_begin != nullptr && rad_h_end != nullptr) {
                sloc_cell = hydro_owner_from_rad_position_1d(cell_l,
                                                             r_mid,
                                                             hydro_node_r,
                                                             rad_h_begin,
                                                             rad_h_end);
              }
              if (sloc_cell >= 0) {
                const double signed_dE = static_cast<double>(sign_l) * dE;
                atomic_add_double(&sloc_abs_wr[sloc_cell], r_mid * signed_dE);
                atomic_add_double(&sloc_abs_wr2[sloc_cell], r_mid * r_mid * signed_dE);
                atomic_add_double(&sloc_abs_E[sloc_cell], signed_dE);
              }
            }
          }
          if (dep > 0.0) {
            ++local_cnt_absorb_survive;
          }

          if (E_l <= 0.0) {
            E_l = 0.0;
            alive_l = kDead;
            active = false;
            ++local_cnt_absorb_kill;
            atomic_inc_counter(imc_absorbed);
          }
        }
      }

      if (!used_pgrw) {
        const double s_min = fmin(s_cen, fmin(s_bdry, s_scatter));

        const double r_before = r_l;
        const double E_old = E_l;
        const double tau = compute_tau_saturated_device(sigma_a, s_min);
        double dep = 0.0;
        if (tau < 1.0e-6) {
          dep = E_old * tau * (1.0 - 0.5 * tau);
        } else {
          dep = -E_old * expm1(-tau);
        }
        dep = fmin(fmax(dep, 0.0), E_old);
        E_l = E_old - dep;

        const double tl = (sigma_a > 0.0) ? (dep / sigma_a) : (E_old * s_min);
        if (dep != 0.0 || tl != 0.0) {
          const double sign_d = static_cast<double>(sign_l);
          const double mu2 = fmin(1.0, fmax(0.0, mu_l * mu_l));
          warp_tally(rad_dep,
                     rad_E_tally,
                     holo_Prr_tally,
                     holo_Prr_coverage_tally,
                     key,
                     sign_d * dep,
                     sign_d * tl,
                     sign_d * tl * mu2,
                     tl);
        }

        const double r_old = r_l;
        const double r2_new =
            fmax(0.0, r_old * r_old + 2.0 * mu_l * r_old * s_min + s_min * s_min);
        r_l = sqrt(r2_new);
        if (r_l > 1.0e-30) {
          mu_l = (mu_l * r_old + s_min) / r_l;
          mu_l = fmin(1.0, fmax(-1.0, mu_l));
        } else {
          mu_l = 1.0;
        }
        t_remain_l -= s_min / tenryu::core::constants::c_light;
        if (sigma_s > 0.0) {
          tau_scatter_remain -= sigma_s * s_min;
        }
        if (sloc_abs_wr != nullptr && sloc_abs_wr2 != nullptr && sloc_abs_E != nullptr) {
          const double dE = E_old - E_l;
          if (dE > 0.0) {
            const double r_mid = 0.5 * (r_before + r_l);
            int sloc_cell = cell_l;
            if (hydro_node_r != nullptr && rad_h_begin != nullptr && rad_h_end != nullptr) {
              sloc_cell = hydro_owner_from_rad_position_1d(cell_l,
                                                           r_mid,
                                                           hydro_node_r,
                                                           rad_h_begin,
                                                           rad_h_end);
            }
            if (sloc_cell >= 0) {
              const double signed_dE = static_cast<double>(sign_l) * dE;
              atomic_add_double(&sloc_abs_wr[sloc_cell], r_mid * signed_dE);
              atomic_add_double(&sloc_abs_wr2[sloc_cell], r_mid * r_mid * signed_dE);
              atomic_add_double(&sloc_abs_E[sloc_cell], signed_dE);
            }
          }
        }
        if (dep > 0.0) {
          ++local_cnt_absorb_survive;
        }

        if (E_l <= 0.0) {
          E_l = 0.0;
          alive_l = kDead;
          active = false;
          ++local_cnt_absorb_kill;
          atomic_inc_counter(imc_absorbed);
        }

        if (active) {
          const bool is_census =
              (t_remain_l <= 1.0e-10 * dt) ||
              (s_cen <= s_bdry && s_cen <= s_scatter);
          if (is_census) {
            ++local_cnt_census;
            t_remain_l = 0.0;
            active = false;
            alive_l = kAlive;
          } else {
            const bool is_boundary = (s_bdry <= s_scatter);
            if (is_boundary) {
              ++local_cnt_boundary;
              if (hit_face == 0) {
                const int neighbor_cell = cell_l - 1;
                const bool ghost_check_enabled = (ghost_layers > 0) && (nr_local > 0);
                const bool enters_left_ghost =
                    ghost_check_enabled && (neighbor_cell < ghost_layers);
                const bool treat_as_left_boundary =
                    (cell_l == 0) || (enters_left_ghost && has_left_boundary != 0);
                if (enters_left_ghost && has_left_boundary == 0) {
                  tally_face_current(face_current_step,
                                     cell_l,
                                     static_cast<int>(group_l),
                                     n_cells,
                                     n_groups,
                                     -E_l);
                  cell_l = -(100 + 0);
                  alive_l = kAlive;
                  active = false;
                } else if (treat_as_left_boundary) {
                  if (is_escape_boundary(bc_inner)) {
                    tally_face_current(face_current_step,
                                       cell_l,
                                       static_cast<int>(group_l),
                                       n_cells,
                                       n_groups,
                                       -E_l);
                    atomic_add_double(&E_escape[group_l], static_cast<double>(sign_l) * E_l);
                    E_l = 0.0;
                    alive_l = kDead;
                    active = false;
                    atomic_inc_counter(imc_escaped);
                  } else {
                    mu_l = -mu_l;
                    r_l = fmax(r_l, kGeomEps);
                  }
                } else if (destination_is_diffusion_cell(diff_cell, neighbor_cell, n_cells)) {
                  tally_diffusion_entry(diff_face_current_in,
                                        face_current_step,
                                        diffusion_interface_kills,
                                        cell_l,
                                        static_cast<int>(group_l),
                                        n_cells,
                                        n_groups,
                                        E_l,
                                        -E_l);
                  E_l = 0.0;
                  t_remain_l = 0.0;
                  alive_l = kDead;
                  mode_arr[my_idx] = kModeIMC;
                  active = false;
                } else {
                  if constexpr (kEnableInterfaceConversion) {
                    bool neighbor_is_ddmc = false;
                    if (ddmc_mode != nullptr && neighbor_cell >= 0 &&
                        neighbor_cell < n_cells &&
                        static_cast<int>(group_l) < n_groups_for_mode) {
                      const int mode_idx =
                          neighbor_cell * n_groups_for_mode + static_cast<int>(group_l);
                      neighbor_is_ddmc = (ddmc_mode[mode_idx] == TransportMode::DDMC);
                    }

                    if (neighbor_is_ddmc && ddmc_zero_flux_interfaces != 0) {
                      sample_isotropic_half_space(&rng,
                                                  &rng_counter_l,
                                                  +1.0,
                                                  &mu_l,
                                                  &dir_z_l,
                                                  &dir_phi_l);
                      const double r_lo_curr = node_r[cell_l] + kGeomEps;
                      const double r_hi_curr = node_r[cell_l + 1] - kGeomEps;
                      if (r_hi_curr > r_lo_curr) {
                        r_l = fmax(r_l, r_lo_curr);
                        r_l = fmin(r_l, r_hi_curr);
                      } else {
                        r_l = 0.5 * (node_r[cell_l] + node_r[cell_l + 1]);
                      }
                      atomic_inc_counter(interface_reflections);
                    } else if (neighbor_is_ddmc) {
                      const int sigma_idx =
                          neighbor_cell * n_groups + static_cast<int>(group_l);
                      const double sigma_r =
                          (sigma_R != nullptr) ? fmax(sigma_R[sigma_idx], 0.0) : 0.0;

                      double delta_x = 0.0;
                      if (cell_dx != nullptr) {
                        delta_x = fmax(cell_dx[neighbor_cell], 0.0);
                      }
                      if (delta_x <= 0.0) {
                        delta_x = compute_delta_x_m_1d(node_r[neighbor_cell],
                                                       node_r[neighbor_cell + 1]);
                      }

                      double omega = 0.0;
                      if (fleck_f != nullptr) {
                        omega = clamp01(1.0 - fleck_f[neighbor_cell]);
                      } else if (sigma_R != nullptr && sigma_a_raw != nullptr &&
                                 sigma_r > 0.0) {
                        const double sigma_a_c = fmax(sigma_a_raw[sigma_idx], 0.0);
                        omega = clamp01(1.0 - sigma_a_c / sigma_r);
                      }

                      const double mu_face = clamp01(fabs(mu_l));
                      InterfaceProbabilityResult p_result{};
                      if (emissivity_preserving != 0) {
                        p_result = compute_P_hat_emissivity(sigma_r, delta_x, omega, mu_face);
                      } else {
                        p_result.probability = compute_standard_p_mu(sigma_r, delta_x, mu_face);
                      }
                      if (p_result.used_standard_fallback) {
                        atomic_inc_counter(conversion_prob_violations);
                      }

                      if (rng_uniform(&rng, &rng_counter_l) < p_result.probability) {
                        tally_face_current(face_current_step,
                                           cell_l,
                                           static_cast<int>(group_l),
                                           n_cells,
                                           n_groups,
                                           -E_l);
                        cell_l = neighbor_cell;
                        mode_arr[my_idx] = kModeDDMC;
                        t_remain_l = 0.0;
                        const double nan = NAN;
                        r_l = nan;
                        z_l = nan;
                        mu_l = nan;
                        dir_z_l = nan;
                        dir_phi_l = nan;
                        atomic_inc_counter(interface_transitions);
                        alive_l = kAlive;
                        active = false;
                      } else {
                        sample_isotropic_half_space(&rng,
                                                    &rng_counter_l,
                                                    +1.0,
                                                    &mu_l,
                                                    &dir_z_l,
                                                    &dir_phi_l);
                        const double r_lo_curr = node_r[cell_l] + kGeomEps;
                        const double r_hi_curr = node_r[cell_l + 1] - kGeomEps;
                        if (r_hi_curr > r_lo_curr) {
                          r_l = fmax(r_l, r_lo_curr);
                          r_l = fmin(r_l, r_hi_curr);
                        } else {
                          r_l = 0.5 * (node_r[cell_l] + node_r[cell_l + 1]);
                        }
                        atomic_inc_counter(interface_reflections);
                      }
                    } else {
                      tally_face_current(face_current_step,
                                         cell_l,
                                         static_cast<int>(group_l),
                                         n_cells,
                                         n_groups,
                                         -E_l);
                      cell_l = neighbor_cell;
                      if (cell_l >= 0 && cell_l < n_cells) {
                        const double r_lo_new = node_r[cell_l] + kGeomEps;
                        const double r_hi_new = node_r[cell_l + 1] - kGeomEps;
                        if (r_hi_new > r_lo_new) {
                          r_l = fmin(r_l, r_hi_new);
                          r_l = fmax(r_l, r_lo_new);
                        } else {
                          r_l = 0.5 * (node_r[cell_l] + node_r[cell_l + 1]);
                        }
                      }
                    }
                  } else {
                    tally_face_current(face_current_step,
                                       cell_l,
                                       static_cast<int>(group_l),
                                       n_cells,
                                       n_groups,
                                       -E_l);
                    cell_l = neighbor_cell;
                    if (cell_l >= 0 && cell_l < n_cells) {
                      const double r_lo_new = node_r[cell_l] + kGeomEps;
                      const double r_hi_new = node_r[cell_l + 1] - kGeomEps;
                      if (r_hi_new > r_lo_new) {
                        r_l = fmin(r_l, r_hi_new);
                        r_l = fmax(r_l, r_lo_new);
                      } else {
                        r_l = 0.5 * (node_r[cell_l] + node_r[cell_l + 1]);
                      }
                    }
                  }
                }
              } else {
                const int neighbor_cell = cell_l + 1;
                const bool ghost_check_enabled = (ghost_layers > 0) && (nr_local > 0);
                const bool enters_right_ghost =
                    ghost_check_enabled &&
                    (neighbor_cell >= (ghost_layers + nr_local));
                const bool treat_as_right_boundary =
                    (cell_l == n_cells - 1) ||
                    (enters_right_ghost && has_right_boundary != 0);
                if (enters_right_ghost && has_right_boundary == 0) {
                  tally_face_current(face_current_step,
                                     cell_l + 1,
                                     static_cast<int>(group_l),
                                     n_cells,
                                     n_groups,
                                     E_l);
                  cell_l = -(100 + 1);
                  alive_l = kAlive;
                  active = false;
                } else if (treat_as_right_boundary) {
                  if (is_escape_boundary(bc_outer)) {
                    tally_face_current(face_current_step,
                                       cell_l + 1,
                                       static_cast<int>(group_l),
                                       n_cells,
                                       n_groups,
                                       E_l);
                    atomic_add_double(&E_escape[group_l], static_cast<double>(sign_l) * E_l);
                    E_l = 0.0;
                    alive_l = kDead;
                    active = false;
                    atomic_inc_counter(imc_escaped);
                  } else {
                    mu_l = -mu_l;
                  }
                } else if (destination_is_diffusion_cell(diff_cell, neighbor_cell, n_cells)) {
                  tally_diffusion_entry(diff_face_current_in,
                                        face_current_step,
                                        diffusion_interface_kills,
                                        cell_l + 1,
                                        static_cast<int>(group_l),
                                        n_cells,
                                        n_groups,
                                        E_l,
                                        E_l);
                  E_l = 0.0;
                  t_remain_l = 0.0;
                  alive_l = kDead;
                  mode_arr[my_idx] = kModeIMC;
                  active = false;
                } else {
                  if constexpr (kEnableInterfaceConversion) {
                    bool neighbor_is_ddmc = false;
                    if (ddmc_mode != nullptr && neighbor_cell >= 0 &&
                        neighbor_cell < n_cells &&
                        static_cast<int>(group_l) < n_groups_for_mode) {
                      const int mode_idx =
                          neighbor_cell * n_groups_for_mode + static_cast<int>(group_l);
                      neighbor_is_ddmc = (ddmc_mode[mode_idx] == TransportMode::DDMC);
                    }

                    if (neighbor_is_ddmc && ddmc_zero_flux_interfaces != 0) {
                      sample_isotropic_half_space(&rng,
                                                  &rng_counter_l,
                                                  -1.0,
                                                  &mu_l,
                                                  &dir_z_l,
                                                  &dir_phi_l);
                      const double r_lo_curr = node_r[cell_l] + kGeomEps;
                      const double r_hi_curr = node_r[cell_l + 1] - kGeomEps;
                      if (r_hi_curr > r_lo_curr) {
                        r_l = fmax(r_l, r_lo_curr);
                        r_l = fmin(r_l, r_hi_curr);
                      } else {
                        r_l = 0.5 * (node_r[cell_l] + node_r[cell_l + 1]);
                      }
                      atomic_inc_counter(interface_reflections);
                    } else if (neighbor_is_ddmc) {
                      const int sigma_idx =
                          neighbor_cell * n_groups + static_cast<int>(group_l);
                      const double sigma_r =
                          (sigma_R != nullptr) ? fmax(sigma_R[sigma_idx], 0.0) : 0.0;

                      double delta_x = 0.0;
                      if (cell_dx != nullptr) {
                        delta_x = fmax(cell_dx[neighbor_cell], 0.0);
                      }
                      if (delta_x <= 0.0) {
                        delta_x = compute_delta_x_m_1d(node_r[neighbor_cell],
                                                       node_r[neighbor_cell + 1]);
                      }

                      double omega = 0.0;
                      if (fleck_f != nullptr) {
                        omega = clamp01(1.0 - fleck_f[neighbor_cell]);
                      } else if (sigma_R != nullptr && sigma_a_raw != nullptr &&
                                 sigma_r > 0.0) {
                        const double sigma_a_c = fmax(sigma_a_raw[sigma_idx], 0.0);
                        omega = clamp01(1.0 - sigma_a_c / sigma_r);
                      }

                      const double mu_face = clamp01(fabs(mu_l));
                      InterfaceProbabilityResult p_result{};
                      if (emissivity_preserving != 0) {
                        p_result = compute_P_hat_emissivity(sigma_r, delta_x, omega, mu_face);
                      } else {
                        p_result.probability = compute_standard_p_mu(sigma_r, delta_x, mu_face);
                      }
                      if (p_result.used_standard_fallback) {
                        atomic_inc_counter(conversion_prob_violations);
                      }

                      if (rng_uniform(&rng, &rng_counter_l) < p_result.probability) {
                        tally_face_current(face_current_step,
                                           cell_l + 1,
                                           static_cast<int>(group_l),
                                           n_cells,
                                           n_groups,
                                           E_l);
                        cell_l = neighbor_cell;
                        mode_arr[my_idx] = kModeDDMC;
                        t_remain_l = 0.0;
                        const double nan = NAN;
                        r_l = nan;
                        z_l = nan;
                        mu_l = nan;
                        dir_z_l = nan;
                        dir_phi_l = nan;
                        atomic_inc_counter(interface_transitions);
                        alive_l = kAlive;
                        active = false;
                      } else {
                        sample_isotropic_half_space(&rng,
                                                    &rng_counter_l,
                                                    -1.0,
                                                    &mu_l,
                                                    &dir_z_l,
                                                    &dir_phi_l);
                        const double r_lo_curr = node_r[cell_l] + kGeomEps;
                        const double r_hi_curr = node_r[cell_l + 1] - kGeomEps;
                        if (r_hi_curr > r_lo_curr) {
                          r_l = fmax(r_l, r_lo_curr);
                          r_l = fmin(r_l, r_hi_curr);
                        } else {
                          r_l = 0.5 * (node_r[cell_l] + node_r[cell_l + 1]);
                        }
                        atomic_inc_counter(interface_reflections);
                      }
                    } else {
                      tally_face_current(face_current_step,
                                         cell_l + 1,
                                         static_cast<int>(group_l),
                                         n_cells,
                                         n_groups,
                                         E_l);
                      cell_l = neighbor_cell;
                      if (cell_l >= 0 && cell_l < n_cells) {
                        const double r_lo_new = node_r[cell_l] + kGeomEps;
                        const double r_hi_new = node_r[cell_l + 1] - kGeomEps;
                        if (r_hi_new > r_lo_new) {
                          r_l = fmax(r_l, r_lo_new);
                          r_l = fmin(r_l, r_hi_new);
                        } else {
                          r_l = 0.5 * (node_r[cell_l] + node_r[cell_l + 1]);
                        }
                      }
                    }
                  } else {
                    tally_face_current(face_current_step,
                                       cell_l + 1,
                                       static_cast<int>(group_l),
                                       n_cells,
                                       n_groups,
                                       E_l);
                    cell_l = neighbor_cell;
                    if (cell_l >= 0 && cell_l < n_cells) {
                      const double r_lo_new = node_r[cell_l] + kGeomEps;
                      const double r_hi_new = node_r[cell_l + 1] - kGeomEps;
                      if (r_hi_new > r_lo_new) {
                        r_l = fmax(r_l, r_lo_new);
                        r_l = fmin(r_l, r_hi_new);
                      } else {
                        r_l = 0.5 * (node_r[cell_l] + node_r[cell_l + 1]);
                      }
                    }
                  }
                }
              }
            } else {
              ++local_cnt_scatter;
              tau_scatter_remain = -log(rng_uniform(&rng, &rng_counter_l));
              sample_isotropic_direction(&rng, &rng_counter_l, &mu_l, &dir_z_l, &dir_phi_l);

              if (inelastic_scatter != 0 && n_groups > 1) {
                const int cell_base = cell_l * n_groups;
                if (scatter_bias_cdf != nullptr) {
                  const double xi_g = rng_uniform(&rng, &rng_counter_l);
                  const int g_new = static_cast<int>(
                      sample_group_from_cdf_device(scatter_bias_cdf + cell_base, n_groups, xi_g));
                  const double q_bias =
                      cdf_row_probability_bias(scatter_bias_cdf + cell_base, g_new);
                  if (q_bias > 1.0e-300) {
                    double p_phys = 0.0;
                    if (eta_cdf != nullptr) {
                      p_phys = cdf_row_probability_bias(eta_cdf + cell_base, g_new);
                    } else if (eta_cdf_hydro != nullptr && hydro_node_r != nullptr &&
                               rad_h_begin != nullptr && rad_h_end != nullptr) {
                      const int hydro_owner = hydro_owner_from_rad_position_1d(
                          cell_l, r_l, hydro_node_r, rad_h_begin, rad_h_end);
                      if (hydro_owner >= 0) {
                        const int hydro_base = hydro_owner * n_groups;
                        p_phys =
                            cdf_row_probability_bias(eta_cdf_hydro + hydro_base, g_new);
                      }
                    } else {
                      const double Te_c = fmax(Te[cell_l], 1.0e-12);
                      double total = 0.0;
                      for (int gg = 0; gg < n_groups; ++gg) {
                        total += fmax(sigma_a_eff[cell_base + gg], 0.0) *
                                 fmax(planck.interpolate_b(gg, Te_c), 0.0);
                      }
                      if (total > 0.0) {
                        p_phys =
                            (fmax(sigma_a_eff[cell_base + g_new], 0.0) *
                             fmax(planck.interpolate_b(g_new, Te_c), 0.0)) /
                            total;
                      }
                    }
                    if (p_phys > 0.0) {
                      E_l *= p_phys / q_bias;
                    }
                  }
                  group_l = static_cast<std::uint16_t>(g_new);
                } else if (eta_cdf != nullptr) {
                  const double xi_g = rng_uniform(&rng, &rng_counter_l);
                  group_l = sample_group_from_cdf_device(eta_cdf + cell_base, n_groups, xi_g);
                } else if (eta_cdf_hydro != nullptr && hydro_node_r != nullptr &&
                           rad_h_begin != nullptr && rad_h_end != nullptr) {
                  const int hydro_owner = hydro_owner_from_rad_position_1d(
                      cell_l, r_l, hydro_node_r, rad_h_begin, rad_h_end);
                  if (hydro_owner >= 0) {
                    const double xi_g = rng_uniform(&rng, &rng_counter_l);
                    const int hydro_base = hydro_owner * n_groups;
                    group_l =
                        sample_group_from_cdf_device(eta_cdf_hydro + hydro_base, n_groups, xi_g);
                  }
                } else {
                  const double Te_c = fmax(Te[cell_l], 1.0e-12);

                  double cdf_total = 0.0;
                  for (int g = 0; g < n_groups; ++g) {
                    const double sigma_g = fmax(sigma_a_eff[cell_base + g], 0.0);
                    const double b_g = fmax(planck.interpolate_b(g, Te_c), 0.0);
                    cdf_total += sigma_g * b_g;
                  }

                  if (cdf_total > 0.0) {
                    const double xi_g = rng_uniform(&rng, &rng_counter_l) * cdf_total;
                    double cdf = 0.0;
                    int g_new = static_cast<int>(group_l);
                    for (int g = 0; g < n_groups; ++g) {
                      const double sigma_g = fmax(sigma_a_eff[cell_base + g], 0.0);
                      const double b_g = fmax(planck.interpolate_b(g, Te_c), 0.0);
                      cdf += sigma_g * b_g;
                      if (xi_g <= cdf) {
                        g_new = g;
                        break;
                      }
                    }
                    group_l = static_cast<std::uint16_t>(g_new);
                  }
                }
              }
            }
          }
        }
      }

      if (active && f_cutoff > 0.0 && E_l < f_cutoff * birth_E_l) {
        const int key_cut = cell_l * n_groups + static_cast<int>(group_l);
        warp_tally(rad_dep,
                   rad_E_tally,
                   holo_Prr_tally,
                   holo_Prr_coverage_tally,
                   key_cut,
                   static_cast<double>(sign_l) * E_l,
                   0.0,
                   0.0,
                   0.0);
        E_l = 0.0;
        alive_l = kDead;
        active = false;
        atomic_inc_counter(imc_absorbed);
      }

      if (active && E_l < w_cutoff * fmax(E_avg, 1.0e-300)) {
        if (p_survival > 0.0 && rng_uniform(&rng, &rng_counter_l) < p_survival) {
          E_l /= p_survival;
        } else {
          const int key_rr = cell_l * n_groups + static_cast<int>(group_l);
          warp_tally(rad_dep,
                     rad_E_tally,
                     holo_Prr_tally,
                     holo_Prr_coverage_tally,
                     key_rr,
                     static_cast<double>(sign_l) * E_l,
                     0.0,
                     0.0,
                     0.0);
          E_l = 0.0;
          alive_l = kDead;
          active = false;
          ++local_cnt_roulette_kill;
          atomic_inc_counter(imc_absorbed);
          if (error_flags != nullptr) {
            atomicExch(&error_flags->roulette_kill, 1);
          }
        }
      }

      if (active && has_non_finite_imc_state_1d(r_l, z_l, mu_l, dir_z_l, dir_phi_l, E_l)) {
        mark_nan_particle(error_flags, E_numerical_loss, sign_l, &E_l);
        r_l = 0.0;
        z_l = 0.0;
        mu_l = 1.0;
        dir_z_l = 0.0;
        dir_phi_l = 0.0;
        t_remain_l = 0.0;
        alive_l = kDead;
        active = false;
      }

      if (!active) {
        if (cnt_boundary != nullptr && local_cnt_boundary > 0) {
          atomicAdd(cnt_boundary, static_cast<unsigned long long>(local_cnt_boundary));
        }
        if (cnt_scatter != nullptr && local_cnt_scatter > 0) {
          atomicAdd(cnt_scatter, static_cast<unsigned long long>(local_cnt_scatter));
        }
        if (cnt_census != nullptr && local_cnt_census > 0) {
          atomicAdd(cnt_census, static_cast<unsigned long long>(local_cnt_census));
        }
        if (cnt_absorb_kill != nullptr && local_cnt_absorb_kill > 0) {
          atomicAdd(cnt_absorb_kill, static_cast<unsigned long long>(local_cnt_absorb_kill));
        }
        if (cnt_absorb_survive != nullptr && local_cnt_absorb_survive > 0) {
          atomicAdd(cnt_absorb_survive,
                    static_cast<unsigned long long>(local_cnt_absorb_survive));
        }
        if (cnt_roulette_kill != nullptr && local_cnt_roulette_kill > 0) {
          atomicAdd(cnt_roulette_kill, static_cast<unsigned long long>(local_cnt_roulette_kill));
        }
        const bool is_ddmc_sentinel_state =
            (alive_l == kAlive) && (mode_arr[my_idx] == kModeDDMC);
        if (!is_ddmc_sentinel_state &&
            has_non_finite_imc_state_1d(r_l, z_l, mu_l, dir_z_l, dir_phi_l, E_l)) {
          mark_nan_particle(error_flags, E_numerical_loss, sign_l, &E_l);
          r_l = 0.0;
          z_l = 0.0;
          mu_l = 1.0;
          dir_z_l = 0.0;
          dir_phi_l = 0.0;
          t_remain_l = 0.0;
          alive_l = kDead;
        }

        pos_r_arr[my_idx] = r_l;
        pos_z_arr[my_idx] = z_l;
        dir_r_arr[my_idx] = mu_l;
        dir_z_arr[my_idx] = dir_z_l;
        dir_phi_arr[my_idx] = dir_phi_l;
        energy_arr[my_idx] = E_l;
        time_remain_arr[my_idx] = t_remain_l;
        birth_energy_arr[my_idx] = birth_E_l;
        cell_id_arr[my_idx] = cell_l;
        group_id_arr[my_idx] = group_l;
        alive_arr[my_idx] = alive_l;
        if (alive_l == kAlive) {
          if constexpr (kEnableInterfaceConversion) {
            if (mode_arr[my_idx] != kModeDDMC) {
              mode_arr[my_idx] = kModeIMC;
            }
          } else {
            mode_arr[my_idx] = kModeIMC;
          }
        }
        rng_counter_arr[my_idx] = rng_counter_l;
      }
    }
  }

  if (active && my_idx >= 0) {
    if (has_non_finite_imc_state_1d(r_l, z_l, mu_l, dir_z_l, dir_phi_l, E_l)) {
      mark_nan_particle(error_flags, E_numerical_loss, sign_l, &E_l);
      r_l = 0.0;
      z_l = 0.0;
      mu_l = 1.0;
      dir_z_l = 0.0;
      dir_phi_l = 0.0;
      t_remain_l = 0.0;
      alive_l = kDead;
    }

    pos_r_arr[my_idx] = r_l;
    pos_z_arr[my_idx] = z_l;
    dir_r_arr[my_idx] = mu_l;
    dir_z_arr[my_idx] = dir_z_l;
    dir_phi_arr[my_idx] = dir_phi_l;
    energy_arr[my_idx] = E_l;
    time_remain_arr[my_idx] = t_remain_l;
    birth_energy_arr[my_idx] = birth_E_l;
    cell_id_arr[my_idx] = cell_l;
    group_id_arr[my_idx] = group_l;
    alive_arr[my_idx] = alive_l;
    if constexpr (kEnableInterfaceConversion) {
      if (mode_arr[my_idx] != kModeDDMC) {
        mode_arr[my_idx] = kModeIMC;
      }
    } else {
      mode_arr[my_idx] = kModeIMC;
    }
    rng_counter_arr[my_idx] = rng_counter_l;
  }
}

__global__ void update_imc_weight_kernel(const int n_imc,
                                         const double* __restrict__ energy_arr,
                                         const double* __restrict__ birth_energy_arr,
                                         double* __restrict__ weight_arr) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_imc) {
    return;
  }
  const double birth = birth_energy_arr[idx];
  const double energy = fmax(energy_arr[idx], 0.0);
  weight_arr[idx] = (birth > 0.0) ? (energy / birth) : 0.0;
}

}  // namespace

void imc_transport_persistent_cuda(const TransportInputs& in,
                                   const tenryu::parallel::PartitionInfo* part) {
  TENRYU_ASSERT(in.pool != nullptr, "imc_transport requires pool");
  TENRYU_ASSERT(in.sigma_a_eff != nullptr, "imc_transport requires sigma_a_eff");
  TENRYU_ASSERT(in.sigma_s_eff != nullptr, "imc_transport requires sigma_s_eff");
  TENRYU_ASSERT(in.Te != nullptr, "imc_transport requires Te");
  TENRYU_ASSERT(in.vol != nullptr, "imc_transport requires vol");
  TENRYU_ASSERT(in.node_r != nullptr, "imc_transport requires node_r");
  TENRYU_ASSERT(in.rad_dep != nullptr, "imc_transport requires rad_dep");
  TENRYU_ASSERT(in.rad_E_tally != nullptr, "imc_transport requires rad_E_tally");
  TENRYU_ASSERT(in.E_escape != nullptr, "imc_transport requires E_escape");
  TENRYU_ASSERT(in.E_numerical_loss != nullptr,
                "imc_transport requires E_numerical_loss");
  TENRYU_ASSERT(in.imc_absorbed != nullptr, "imc_transport requires imc_absorbed");
  TENRYU_ASSERT(in.imc_escaped != nullptr, "imc_transport requires imc_escaped");
  TENRYU_ASSERT(in.n_cells >= 0, "imc_transport requires n_cells >= 0");
  TENRYU_ASSERT(in.n_groups >= 1, "imc_transport requires n_groups >= 1");
  TENRYU_ASSERT(in.n_imc >= 0, "imc_transport requires n_imc >= 0");
  if (in.ddmc_mode != nullptr) {
    TENRYU_ASSERT(in.n_groups_for_mode >= 1,
                  "imc_transport requires n_groups_for_mode >= 1 when ddmc_mode is set");
    TENRYU_ASSERT(in.n_groups_for_mode == in.n_groups,
                  "imc_transport requires n_groups_for_mode == n_groups when ddmc_mode is set");
    TENRYU_ASSERT(in.sigma_R != nullptr,
                  "imc_transport requires sigma_R when ddmc_mode is set");
    TENRYU_ASSERT(in.fleck_f != nullptr || in.sigma_a != nullptr,
                  "imc_transport requires fleck_f or sigma_a when ddmc_mode is set");
  }
  if (in.pgrw_tau_rw > 0.0) {
    TENRYU_ASSERT(in.g_diff_end != nullptr, "imc_transport requires g_diff_end for PGRW");
    TENRYU_ASSERT(in.sigma_a_bar != nullptr,
                  "imc_transport requires sigma_a_bar for PGRW");
    TENRYU_ASSERT(in.sigma_t_bar != nullptr,
                  "imc_transport requires sigma_t_bar for PGRW");
    TENRYU_ASSERT(in.D_pgrw != nullptr, "imc_transport requires D_pgrw for PGRW");
    TENRYU_ASSERT(in.gamma_pgrw != nullptr,
                  "imc_transport requires gamma_pgrw for PGRW");
    TENRYU_ASSERT(in.pgrw_leak_inv_cdf != nullptr,
                  "imc_transport requires leak inverse CDF for PGRW");
    TENRYU_ASSERT(in.pgrw_leak_cdf_xi != nullptr,
                  "imc_transport requires leak xi table for PGRW");
    TENRYU_ASSERT(in.pgrw_pos_cdf != nullptr,
                  "imc_transport requires position CDF table for PGRW");
    TENRYU_ASSERT(in.pgrw_leak_table_size > 0,
                  "imc_transport requires positive PGRW leak table size");
    TENRYU_ASSERT(in.pgrw_pos_theta_bins > 1,
                  "imc_transport requires at least two PGRW theta bins");
    TENRYU_ASSERT(in.pgrw_pos_rho_bins > 1,
                  "imc_transport requires at least two PGRW rho bins");
  }

  if (in.n_imc == 0) {
    return;
  }

  TENRYU_ASSERT(in.pool->pos_r != nullptr, "imc_transport requires pool->pos_r");
  TENRYU_ASSERT(in.pool->pos_z != nullptr, "imc_transport requires pool->pos_z");
  TENRYU_ASSERT(in.pool->dir_r != nullptr, "imc_transport requires pool->dir_r");
  TENRYU_ASSERT(in.pool->dir_z != nullptr, "imc_transport requires pool->dir_z");
  TENRYU_ASSERT(in.pool->dir_phi != nullptr, "imc_transport requires pool->dir_phi");
  TENRYU_ASSERT(in.pool->energy != nullptr, "imc_transport requires pool->energy");
  TENRYU_ASSERT(in.pool->birth_energy != nullptr,
                "imc_transport requires pool->birth_energy");
  TENRYU_ASSERT(in.pool->sign != nullptr, "imc_transport requires pool->sign");
  TENRYU_ASSERT(in.pool->time_remain != nullptr,
                "imc_transport requires pool->time_remain");
  TENRYU_ASSERT(in.pool->global_id != nullptr, "imc_transport requires pool->global_id");
  TENRYU_ASSERT(in.pool->rng_counter != nullptr, "imc_transport requires pool->rng_counter");
  TENRYU_ASSERT(in.pool->cell_id != nullptr, "imc_transport requires pool->cell_id");
  TENRYU_ASSERT(in.pool->group_id != nullptr, "imc_transport requires pool->group_id");
  TENRYU_ASSERT(in.pool->mode != nullptr, "imc_transport requires pool->mode");
  TENRYU_ASSERT(in.pool->alive != nullptr, "imc_transport requires pool->alive");
  TENRYU_ASSERT(in.pool->weight != nullptr, "imc_transport requires pool->weight");

  int* d_counter = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_counter), sizeof(int)),
             "imc_transport cudaMalloc counter failed");
  cuda_check(cudaMemset(d_counter, 0, sizeof(int)),
             "imc_transport cudaMemset counter failed");

  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, 0),
             "imc_transport cudaGetDeviceProperties failed");

  constexpr int kBlock = 128;
  constexpr int kBlocksPerSm = 8;
  const int grid = std::max(1, prop.multiProcessorCount * kBlocksPerSm);
  const RankBoundaryParams1D rank_boundary = make_rank_boundary_params_1d(part);
  const double* scatter_bias_cdf = in.scatter_bias_cdf;

  if (in.ddmc_mode != nullptr) {
    imc_transport_kernel<true><<<grid, kBlock>>>(in.pool->pos_r,
                                                 in.pool->pos_z,
                                                 in.pool->dir_r,
                                                 in.pool->dir_z,
                                                 in.pool->dir_phi,
                                                 in.pool->energy,
                                                 in.pool->birth_energy,
                                                 in.pool->time_remain,
                                                 in.pool->sign,
                                                 in.pool->global_id,
                                                 in.pool->rng_counter,
                                                 in.pool->cell_id,
                                                 in.pool->group_id,
                                                 in.pool->mode,
                                                 in.pool->alive,
                                                 in.sigma_a_eff,
                                                 in.sigma_s_eff,
                                                 in.Te,
                                                 in.vol,
                                                 in.sloc_abs_wr,
                                                 in.sloc_abs_wr2,
                                                 in.sloc_abs_E,
                                                 in.node_r,
                                                 in.ddmc_mode,
                                                 in.ddmc_zero_flux_interfaces ? 1 : 0,
                                                 in.n_groups_for_mode,
                                                 in.emissivity_preserving ? 1 : 0,
                                                 in.sigma_R,
                                                 in.cell_dx,
                                                 in.fleck_f,
                                                 in.sigma_a,
                                                 in.eta_cdf,
                                                 in.eta_cdf_hydro,
                                                 scatter_bias_cdf,
                                                 in.hydro_node_r,
                                                 in.rad_h_begin,
                                                 in.rad_h_end,
                                                 in.g_diff_end,
                                                 in.sigma_a_bar,
                                                 in.sigma_t_bar,
                                                 in.D_pgrw,
                                                 in.gamma_pgrw,
                                                 in.pgrw_leak_inv_cdf,
                                                 in.pgrw_leak_cdf_xi,
                                                 in.pgrw_pos_cdf,
                                                 in.pgrw_leak_table_size,
                                                 in.pgrw_pos_theta_bins,
                                                 in.pgrw_pos_rho_bins,
                                                 in.pgrw_theta_max,
                                                 in.pgrw_tau_rw,
                                                 in.planck,
                                                 in.inelastic_scatter ? 1 : 0,
                                                 in.rad_dep,
                                                 in.rad_E_tally,
                                                 in.holo_Prr_tally,
                                                 in.holo_Prr_coverage_tally,
                                                 in.face_current_step,
                                                 in.diff_cell,
                                                 in.diff_face_current_in,
                                                 in.E_escape,
                                                 in.E_numerical_loss,
                                                 in.imc_absorbed,
                                                 in.imc_escaped,
                                                 in.diffusion_interface_kills,
                                                 in.interface_transitions,
                                                 in.interface_reflections,
                                                 in.conversion_prob_violations,
                                                 in.cnt_boundary,
                                                 in.cnt_scatter,
                                                 in.cnt_census,
                                                 in.cnt_absorb_kill,
                                                 in.cnt_absorb_survive,
                                                 in.cnt_roulette_kill,
                                                 d_counter,
                                                 in.n_cells,
                                                 in.n_groups,
                                                 in.n_imc,
                                                 in.dt,
                                                 in.E_avg,
                                                 in.w_cutoff,
                                                 in.p_survival,
                                                 in.f_cutoff,
                                                 in.tail_pass ? 1 : 0,
                                                 in.bc_inner,
                                                 in.bc_outer,
                                                 in.step_number,
                                                 in.user_seed,
                                                 rank_boundary.ghost_layers,
                                                 rank_boundary.nr_local,
                                                 rank_boundary.has_left_boundary,
                                                 rank_boundary.has_right_boundary,
                                                 in.error_flags);
  } else {
    imc_transport_kernel<false><<<grid, kBlock>>>(in.pool->pos_r,
                                                  in.pool->pos_z,
                                                  in.pool->dir_r,
                                                  in.pool->dir_z,
                                                  in.pool->dir_phi,
                                                  in.pool->energy,
                                                  in.pool->birth_energy,
                                                  in.pool->time_remain,
                                                  in.pool->sign,
                                                  in.pool->global_id,
                                                  in.pool->rng_counter,
                                                  in.pool->cell_id,
                                                  in.pool->group_id,
                                                  in.pool->mode,
                                                  in.pool->alive,
                                                  in.sigma_a_eff,
                                                  in.sigma_s_eff,
                                                  in.Te,
                                                  in.vol,
                                                  in.sloc_abs_wr,
                                                  in.sloc_abs_wr2,
                                                  in.sloc_abs_E,
                                                  in.node_r,
                                                  in.ddmc_mode,
                                                  in.ddmc_zero_flux_interfaces ? 1 : 0,
                                                  in.n_groups_for_mode,
                                                  in.emissivity_preserving ? 1 : 0,
                                                  in.sigma_R,
                                                  in.cell_dx,
                                                  in.fleck_f,
                                                  in.sigma_a,
                                                  in.eta_cdf,
                                                  in.eta_cdf_hydro,
                                                  scatter_bias_cdf,
                                                  in.hydro_node_r,
                                                  in.rad_h_begin,
                                                  in.rad_h_end,
                                                  in.g_diff_end,
                                                  in.sigma_a_bar,
                                                  in.sigma_t_bar,
                                                  in.D_pgrw,
                                                  in.gamma_pgrw,
                                                  in.pgrw_leak_inv_cdf,
                                                  in.pgrw_leak_cdf_xi,
                                                  in.pgrw_pos_cdf,
                                                  in.pgrw_leak_table_size,
                                                  in.pgrw_pos_theta_bins,
                                                  in.pgrw_pos_rho_bins,
                                                  in.pgrw_theta_max,
                                                  in.pgrw_tau_rw,
                                                  in.planck,
                                                  in.inelastic_scatter ? 1 : 0,
                                                  in.rad_dep,
                                                  in.rad_E_tally,
                                                  in.holo_Prr_tally,
                                                  in.holo_Prr_coverage_tally,
                                                  in.face_current_step,
                                                  in.diff_cell,
                                                  in.diff_face_current_in,
                                                  in.E_escape,
                                                  in.E_numerical_loss,
                                                  in.imc_absorbed,
                                                  in.imc_escaped,
                                                  in.diffusion_interface_kills,
                                                  in.interface_transitions,
                                                  in.interface_reflections,
                                                  in.conversion_prob_violations,
                                                  in.cnt_boundary,
                                                  in.cnt_scatter,
                                                  in.cnt_census,
                                                  in.cnt_absorb_kill,
                                                  in.cnt_absorb_survive,
                                                  in.cnt_roulette_kill,
                                                  d_counter,
                                                  in.n_cells,
                                                  in.n_groups,
                                                  in.n_imc,
                                                  in.dt,
                                                  in.E_avg,
                                                  in.w_cutoff,
                                                  in.p_survival,
                                                  in.f_cutoff,
                                                  in.tail_pass ? 1 : 0,
                                                  in.bc_inner,
                                                  in.bc_outer,
                                                  in.step_number,
                                                  in.user_seed,
                                                  rank_boundary.ghost_layers,
                                                  rank_boundary.nr_local,
                                                  rank_boundary.has_left_boundary,
                                                  rank_boundary.has_right_boundary,
                                                  in.error_flags);
  }

  cuda_check(cudaGetLastError(), "imc_transport kernel launch failed");
  const int weight_grid = (in.n_imc + kBlock - 1) / kBlock;
  update_imc_weight_kernel<<<weight_grid, kBlock>>>(in.n_imc,
                                                    in.pool->energy,
                                                    in.pool->birth_energy,
                                                    in.pool->weight);
  cuda_check(cudaGetLastError(), "imc_transport weight kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "imc_transport kernel execution failed");
  cuda_check(cudaFree(d_counter), "imc_transport cudaFree counter failed");
}

void imc_transport_persistent_cuda(const TransportInputs& in) {
  imc_transport_persistent_cuda(in, nullptr);
}

void imc_transport_persistent_cuda(const TransportInputs& in,
                                   const tenryu::parallel::PartitionInfo& part) {
  imc_transport_persistent_cuda(in, &part);
}

}  // namespace tenryu::radiation
