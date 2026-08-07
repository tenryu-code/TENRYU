#include "laser/cbet.cuh"

#include <algorithm>
#include <cmath>
#include <string>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::laser {

namespace {

constexpr double kPi = 3.14159265358979323846;
// Must match src/laser/refraction.cu and laser_mesh.cu conventions.
constexpr double kElectronMass = 9.1094e-28;      // g
constexpr double kElementaryCharge = 4.8032e-10;  // statC
constexpr double kProtonMass = 1.6726219e-24;     // g

inline void cbet_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void ensure_device_capacity(T** ptr, std::size_t* cap, const std::size_t needed,
                            const char* what) {
  if (needed <= *cap && *ptr != nullptr) {
    return;
  }
  if (*ptr != nullptr) {
    static_cast<void>(cudaFree(*ptr));
    *ptr = nullptr;
  }
  *cap = 0;
  if (needed == 0) {
    return;
  }
  cbet_check(cudaMalloc(reinterpret_cast<void**>(ptr), needed * sizeof(T)), what);
  *cap = needed;
}

template <typename T>
void free_device_pointer(T** ptr) {
  if (*ptr != nullptr) {
    static_cast<void>(cudaFree(*ptr));
    *ptr = nullptr;
  }
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------

__global__ void cbet_build_keys_kernel(const std::int32_t* __restrict__ rec_cell,
                                       const float* __restrict__ rec_mu,
                                       const std::int32_t* __restrict__ rec_count,
                                       const std::int32_t* __restrict__ ray_group_base,
                                       const std::int64_t* __restrict__ ray_rec_offset,
                                       std::int64_t* __restrict__ sort_key,
                                       std::int32_t* __restrict__ sort_slot,
                                       const int cap_per_ray,
                                       const int n_bins,
                                       const int G,
                                       const int n_rays) {
  const int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= n_rays) {
    return;
  }
  const int cnt = rec_count[ray];
  const std::int64_t out_base = ray_rec_offset[ray];
  const std::int64_t in_base =
      static_cast<std::int64_t>(ray) * static_cast<std::int64_t>(cap_per_ray);
  const int base_g = ray_group_base[ray];
  for (int k = 0; k < cnt; ++k) {
    const std::int64_t slot = in_base + k;
    const int g = base_g + ((rec_mu[slot] > 0.0f) ? n_bins : 0);
    const std::int64_t key =
        static_cast<std::int64_t>(rec_cell[slot]) * static_cast<std::int64_t>(G) +
        static_cast<std::int64_t>(g);
    sort_key[out_base + k] = key;
    sort_slot[out_base + k] = static_cast<std::int32_t>(slot);
  }
}

__global__ void cbet_seg_offsets_kernel(const std::int64_t* __restrict__ sort_key,
                                        const std::int64_t n_live,
                                        std::int64_t* __restrict__ seg_offsets,
                                        const std::int64_t n_segments) {
  const std::int64_t s = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (s > n_segments) {
    return;
  }
  if (s == n_segments) {
    seg_offsets[s] = n_live;
    return;
  }
  // lower_bound of s in sort_key[0..n_live)
  std::int64_t lo = 0;
  std::int64_t hi = n_live;
  while (lo < hi) {
    const std::int64_t mid = (lo + hi) >> 1;
    if (sort_key[mid] < s) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  seg_offsets[s] = lo;
}

__global__ void cbet_tally_kernel(const std::int64_t* __restrict__ seg_offsets,
                                  const std::int32_t* __restrict__ sort_slot,
                                  const double* __restrict__ rec_w,
                                  const double* __restrict__ rec_ds,
                                  const float* __restrict__ rec_mu,
                                  double* __restrict__ L,
                                  double* __restrict__ Mmu,
                                  double* __restrict__ ds_max_gc,
                                  const std::int64_t n_segments) {
  const std::int64_t s = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (s >= n_segments) {
    return;
  }
  double l = 0.0;
  double m = 0.0;
  double dmax = 0.0;
  const std::int64_t k0 = seg_offsets[s];
  const std::int64_t k1 = seg_offsets[s + 1];
  for (std::int64_t k = k0; k < k1; ++k) {
    const std::int32_t slot = sort_slot[k];
    const double ds = rec_ds[slot];
    const double wd = rec_w[slot] * ds;
    l += wd;
    m += wd * static_cast<double>(rec_mu[slot]);
    dmax = ::fmax(dmax, ds);
  }
  L[s] = l;
  Mmu[s] = m;
  ds_max_gc[s] = dmax;
}

__global__ void cbet_init_rec_w_ps_kernel(
    const double* __restrict__ rec_w,
    const double* __restrict__ port_weight,
    double* __restrict__ rec_w_ps,
    const std::int64_t n_records_capacity,
    const int n_ports) {
  const std::int64_t idx =
      blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  const std::int64_t total =
      static_cast<std::int64_t>(n_ports) * n_records_capacity;
  if (idx >= total) {
    return;
  }
  const int port = static_cast<int>(idx / n_records_capacity);
  const std::int64_t slot = idx % n_records_capacity;
  rec_w_ps[idx] = port_weight[port] * rec_w[slot];
}

// ps-mode remap twin of cbet_tally_kernel; keep in lockstep.
__global__ void cbet_tally_ps_kernel(
    const std::int64_t* __restrict__ seg_offsets,
    const std::int32_t* __restrict__ sort_slot,
    const double* __restrict__ rec_w_ps,
    const double* __restrict__ rec_ds,
    const float* __restrict__ rec_mu,
    double* __restrict__ L,
    double* __restrict__ Mmu,
    double* __restrict__ ds_max_gc,
    const std::int64_t n_segments,
    const std::int64_t n_records_capacity,
    const int port,
    const int G_ref,
    const int G_ps) {
  const std::int64_t s =
      blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (s >= n_segments) {
    return;
  }
  double l = 0.0;
  double m = 0.0;
  double dmax = 0.0;
  const std::int64_t k0 = seg_offsets[s];
  const std::int64_t k1 = seg_offsets[s + 1];
  const double* const rec_w_port =
      rec_w_ps + static_cast<std::int64_t>(port) * n_records_capacity;
  for (std::int64_t k = k0; k < k1; ++k) {
    const std::int32_t slot = sort_slot[k];
    const double ds = rec_ds[slot];
    const double wd = rec_w_port[slot] * ds;
    l += wd;
    m += wd * static_cast<double>(rec_mu[slot]);
    dmax = ::fmax(dmax, ds);
  }
  const int c = static_cast<int>(s / G_ref);
  const int g = static_cast<int>(s % G_ref);
  const std::int64_t out =
      static_cast<std::int64_t>(c) * G_ps + port * G_ref + g;
  L[out] = l;
  Mmu[out] = m;
  ds_max_gc[out] = dmax;
}

// Signed pair coupling for orientation "p gains from q when positive".
// Exactly odd under (p<->q) because every term negates exactly.
__device__ double cbet_pair_coupling(const double mu_p,
                                     const double mu_q,
                                     const double omega_p,
                                     const double omega_q,
                                     const double chi_pref,
                                     const double c_a,
                                     const double u_r,
                                     const double k_bar,
                                     const double f_cbet,
                                     const double alpha_iaw,
                                     const int n_phi,
                                     const double k_a_floor,
                                     const double test_chi) {
  if (test_chi > 0.0) {
    return test_chi;
  }
  const double s_p = ::sqrt(::fmax(0.0, 1.0 - mu_p * mu_p));
  const double s_q = ::sqrt(::fmax(0.0, 1.0 - mu_q * mu_q));
  const double domega = omega_q - omega_p;
  const double kdotu = k_bar * u_r * (mu_q - mu_p);
  const double ca_safe = ::fmax(c_a, 1.0e-30);
  double acc = 0.0;
  for (int k = 0; k < n_phi; ++k) {
    const double phi = (static_cast<double>(k) + 0.5) * kPi / static_cast<double>(n_phi);
    const double cospsi = mu_p * mu_q + s_p * s_q * ::cos(phi);
    const double ka2 = ::fmax(2.0 * (1.0 - cospsi), 0.0);
    const double ka = k_bar * ::sqrt(ka2);
    if (!(ka >= k_a_floor * k_bar) || !(ka > 0.0)) {
      continue;
    }
    const double g = (domega - kdotu) / (ka * ca_safe);
    const double ga = g * alpha_iaw;
    const double one_g2 = 1.0 - g * g;
    const double denom = ga * ga + one_g2 * one_g2;
    const double Pg = ga / denom;
    const double pol = 0.25 * f_cbet * (1.0 + cospsi * cospsi);
    acc += pol * Pg;
  }
  return chi_pref * (acc / static_cast<double>(n_phi));
}

__global__ void cbet_chi_kernel(const double* __restrict__ L,
                                const double* __restrict__ Mmu,
                                const std::int16_t* __restrict__ pair_p,
                                const std::int16_t* __restrict__ pair_q,
                                const double* __restrict__ omega_group,
                                const double* __restrict__ cell_chi_pref,
                                const double* __restrict__ cell_c_a,
                                const double* __restrict__ cell_u_r,
                                const double* __restrict__ cell_k_bar,
                                const double* __restrict__ cell_vol,
                                const std::uint8_t* __restrict__ cell_mask,
                                double* __restrict__ chi,
                                const double f_cbet,
                                const double alpha_iaw,
                                const int n_phi,
                                const double k_a_floor,
                                const double test_chi,
                                const int G,
                                const int n_pairs,
                                const int n_cells) {
  const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(n_cells) * n_pairs;
  if (idx >= total) {
    return;
  }
  const int c = static_cast<int>(idx / n_pairs);
  const int pair = static_cast<int>(idx % n_pairs);
  double value = 0.0;
  if (cell_mask[c] != 0 && cell_vol[c] > 0.0) {
    const int p = pair_p[pair];
    const int q = pair_q[pair];
    const double Lp = L[static_cast<std::int64_t>(c) * G + p];
    const double Lq = L[static_cast<std::int64_t>(c) * G + q];
    if (Lp > 0.0 && Lq > 0.0) {
      const double mu_p = Mmu[static_cast<std::int64_t>(c) * G + p] / Lp;
      const double mu_q = Mmu[static_cast<std::int64_t>(c) * G + q] / Lq;
      value = cbet_pair_coupling(mu_p, mu_q, omega_group[p], omega_group[q],
                                 cell_chi_pref[c], cell_c_a[c], cell_u_r[c],
                                 cell_k_bar[c], f_cbet, alpha_iaw, n_phi, k_a_floor,
                                 test_chi);
    }
  }
  chi[idx] = value;
}

__global__ void cbet_losscap_kernel(const double* __restrict__ L,
                                    const double* __restrict__ ds_max_gc,
                                    const double* __restrict__ chi,
                                    const std::int32_t* __restrict__ pair_index,
                                    const double* __restrict__ cell_vol,
                                    const std::uint8_t* __restrict__ cell_mask,
                                    double* __restrict__ fcap,
                                    const double theta_cap,
                                    const int G,
                                    const int n_pairs,
                                    const int n_cells) {
  const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(n_cells) * G;
  if (idx >= total) {
    return;
  }
  const int c = static_cast<int>(idx / G);
  const int g = static_cast<int>(idx % G);
  double f = 1.0;
  const double Lg = L[idx];
  if (cell_mask[c] != 0 && Lg > 0.0 && cell_vol[c] > 0.0) {
    const double inv_vol = 1.0 / cell_vol[c];
    const std::int64_t row = static_cast<std::int64_t>(c) * G;
    double loss = 0.0;
    for (int gp = 0; gp < G; ++gp) {
      if (gp == g) {
        continue;
      }
      const int pair = pair_index[g * G + gp];
      const double chi_pq = chi[static_cast<std::int64_t>(c) * n_pairs + pair];
      // Canonical (min,max) operand order so both orientations of the pair
      // round the product identically — exact FP antisymmetry of the pair
      // amount, matching the NUMERICS 5.10.3 claim (AI review M-01/S-04).
      const double Lmin = L[row + ::min(g, gp)];
      const double Lmax = L[row + ::max(g, gp)];
      const double amt = chi_pq * Lmin * Lmax * inv_vol;  // >0: pair_p gains
      // from this group's perspective: negative contribution means loss
      const double signed_amt = (g < gp) ? amt : -amt;
      loss += ::fmax(0.0, -signed_amt);
    }
    const double ds_max = ds_max_gc[idx];
    if (loss > 0.0 && ds_max > 0.0) {
      const double cap = theta_cap * Lg / ds_max;
      f = ::fmin(1.0, cap / loss);
    }
  }
  fcap[idx] = f;
}

// S4 kinetic-response hook. Future nonlinear IAW models may modify the
// receiver response here; v1 preserves the wave-action frequency ratio.
__device__ __forceinline__ double cbet_kinetic_response_hook(
    const int g,
    const double omega_ratio,
    const double amt) {
  static_cast<void>(g);
  static_cast<void>(amt);
  return omega_ratio;
}

__global__ void cbet_dq_kernel(const double* __restrict__ L,
                               const double* __restrict__ chi,
                               const double* __restrict__ fcap,
                               const std::int32_t* __restrict__ pair_index,
                               const double* __restrict__ cell_vol,
                               const std::uint8_t* __restrict__ cell_mask,
                               const double* __restrict__ omega_group,
                               double* __restrict__ dQ,
                               double* __restrict__ iaw_partial,
                               double* __restrict__ exch_partial,
                               unsigned long long* __restrict__ capped_pairs,
                               const int action_mode,
                               const int G,
                               const int n_pairs,
                               const int n_cells) {
  const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(n_cells) * G;
  if (idx >= total) {
    return;
  }
  const int c = static_cast<int>(idx / G);
  const int g = static_cast<int>(idx % G);
  double net = 0.0;
  double iaw = 0.0;
  double exch = 0.0;
  const double Lg = L[idx];
  if (cell_mask[c] != 0 && Lg > 0.0 && cell_vol[c] > 0.0) {
    const double inv_vol = 1.0 / cell_vol[c];
    const std::int64_t row = static_cast<std::int64_t>(c) * G;
    const double f_g = fcap[idx];
    for (int gp = 0; gp < G; ++gp) {
      if (gp == g) {
        continue;
      }
      const int pair = pair_index[g * G + gp];
      const double chi_pq = chi[static_cast<std::int64_t>(c) * n_pairs + pair];
      const double f_gp = fcap[row + gp];
      const double f = ::fmin(f_g, f_gp);
      // Canonical (min,max) operand order: chi is a single cached evaluation
      // and fmin is symmetric, so with this ordering the applied pair amount
      // is bitwise identical from both orientations (AI review M-01/S-04).
      const double Lmin = L[row + ::min(g, gp)];
      const double Lmax = L[row + ::max(g, gp)];
      const double amt = chi_pq * Lmin * Lmax * inv_vol * f;
      const double signed_amt = (g < gp) ? amt : -amt;
      if (omega_group != nullptr && action_mode != 0) {
        const bool g_gains = signed_amt > 0.0;
        if (g_gains) {
          const double transfer_amt = signed_amt;
          const double ratio = cbet_kinetic_response_hook(
              g, omega_group[g] / omega_group[gp], transfer_amt);
          // For finite nonzero equal omegas, x/x is exactly 1.0:
          // transfer_amt*ratio is bitwise transfer_amt and the IAW term is
          // exactly 0.0, preserving the zero-detuning path bitwise.
          net += transfer_amt * ratio;
          iaw += transfer_amt * (1.0 - ratio);
        } else {
          net += signed_amt;
        }
      } else {
        net += signed_amt;
      }
      if (g < gp) {
        exch += ::fabs(amt);
        if (f < 1.0 && amt != 0.0) {
          atomicAdd(capped_pairs, 1ULL);
        }
      }
    }
  }
  dQ[idx] = net;
  iaw_partial[idx] = iaw;
  exch_partial[idx] = exch;
}

__global__ void cbet_iaw_cell_reduce_kernel(
    const double* __restrict__ iaw_partial,
    double* __restrict__ iaw_cell,
    const int G,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double sum = 0.0;
  const std::int64_t row = static_cast<std::int64_t>(c) * G;
  for (int g = 0; g < G; ++g) {
    sum += iaw_partial[row + g];
  }
  iaw_cell[c] = sum;
}

__global__ void cbet_propagate_kernel(const std::int32_t* __restrict__ rec_cell,
                                      const float* __restrict__ rec_mu,
                                      const double* __restrict__ rec_ds,
                                      const double* __restrict__ rec_S,
                                      double* __restrict__ rec_w,
                                      const std::int32_t* __restrict__ rec_count,
                                      const std::int32_t* __restrict__ ray_group_base,
                                      const double* __restrict__ ray_P0,
                                      const std::uint8_t* __restrict__ ray_overflow,
                                      const double* __restrict__ dQ,
                                      const double* __restrict__ L,
                                      const std::uint8_t* __restrict__ cell_mask,
                                      double* __restrict__ dep_rows,
                                      double* __restrict__ unabs_rows,
                                      unsigned long long* __restrict__ clamp_count,
                                      double* __restrict__ clamp_rows,
                                      const int cap_per_ray,
                                      const int n_bins,
                                      const int G,
                                      const int n_cells,
                                      const int n_rays,
                                      const int final_pass) {
  const int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= n_rays) {
    return;
  }
  const int cnt = rec_count[ray];
  double P = ray_P0[ray];
  if (!::isfinite(P) || P < 0.0) {
    P = 0.0;
  }
  double* dep_row = (final_pass != 0)
                        ? dep_rows + static_cast<std::int64_t>(ray) * n_cells
                        : nullptr;
  const std::int64_t base =
      static_cast<std::int64_t>(ray) * static_cast<std::int64_t>(cap_per_ray);
  const int base_g = ray_group_base[ray];
  const bool cbet_active_ray = (ray_overflow[ray] == 0);
  // Clamp-injected power is tallied per ray (exclusive row) and reduced in
  // fixed order afterwards — no atomicAdd(double), bitwise deterministic.
  double clamp_acc = 0.0;
  for (int k = 0; k < cnt; ++k) {
    const std::int64_t slot = base + k;
    const int c = rec_cell[slot];
    const double S_half = 0.5 * rec_S[slot];
    // IB first half
    {
      const double dP = -P * ::expm1(-S_half);
      if (dep_row != nullptr && c >= 0 && c < n_cells) {
        dep_row[c] += dP;
      }
      P -= dP;
    }
    // CBET exchange (share of the group's net dQ by previous-iteration weight)
    const double ds = rec_ds[slot];
    if (cbet_active_ray && ds > 0.0 && c >= 0 && c < n_cells && cell_mask[c] != 0) {
      const int g = base_g + ((rec_mu[slot] > 0.0f) ? n_bins : 0);
      const std::int64_t gc = static_cast<std::int64_t>(c) * G + g;
      const double Lgc = L[gc];
      if (Lgc > 0.0) {
        const double dP_cbet = dQ[gc] * (rec_w[slot] * ds / Lgc);
        P += dP_cbet;
        if (P < 0.0) {
          clamp_acc += -P;
          P = 0.0;
          atomicAdd(clamp_count, 1ULL);
        }
      }
    }
    rec_w[slot] = P;
    // IB second half
    {
      const double dP = -P * ::expm1(-S_half);
      if (dep_row != nullptr && c >= 0 && c < n_cells) {
        dep_row[c] += dP;
      }
      P -= dP;
    }
  }
  if (final_pass != 0) {
    unabs_rows[ray] = (::isfinite(P) && P > 0.0) ? P : 0.0;
    if (clamp_rows != nullptr) {
      clamp_rows[ray] = clamp_acc;
    }
  }
}

// ps-mode remap twin of cbet_propagate_kernel; keep in lockstep.
__global__ void cbet_propagate_ps_kernel(
    const std::int32_t* __restrict__ rec_cell,
    const float* __restrict__ rec_mu,
    const double* __restrict__ rec_ds,
    const double* __restrict__ rec_S,
    double* __restrict__ rec_w_ps,
    const std::int32_t* __restrict__ rec_count,
    const std::int32_t* __restrict__ ray_group_base,
    const double* __restrict__ ray_P0,
    const std::uint8_t* __restrict__ ray_overflow,
    const double* __restrict__ dQ,
    const double* __restrict__ L,
    const std::uint8_t* __restrict__ cell_mask,
    const double* __restrict__ port_weight,
    double* __restrict__ dep_rows,
    double* __restrict__ unabs_rows,
    unsigned long long* __restrict__ clamp_count,
    double* __restrict__ clamp_rows,
    const double* __restrict__ ps_cell_nhat,
    const double* __restrict__ ps_capture_thresh,
    const std::int32_t* __restrict__ ps_capture_order,
    const double* __restrict__ ps_one_minus_eta,
    double* __restrict__ ps_capture_stage,
    double* __restrict__ traj_rec_ratio,
    const int ps_n_channels,
    const int cap_per_ray,
    const int n_bins,
    const int G_ref,
    const int G_ps,
    const int n_cells,
    const int n_rays,
    const int port,
    const long long traj_ray_offset,
    const int traj_n_output_rays,
    const int traj_output_stride,
    const int viz_port,
    const int final_pass) {
  const int ray = blockIdx.x * blockDim.x + threadIdx.x;
  (void)ps_capture_order;
  if (ray >= n_rays) {
    return;
  }
  const int cnt = rec_count[ray];
  double P = port_weight[port] * ray_P0[ray];
  if (!::isfinite(P) || P < 0.0) {
    P = 0.0;
  }
  double P_ref = P;
  double* dep_row = (final_pass != 0)
                        ? dep_rows + static_cast<std::int64_t>(ray) * n_cells
                        : nullptr;
  const std::int64_t n_records_capacity =
      static_cast<std::int64_t>(n_rays) * cap_per_ray;
  const std::int64_t base =
      static_cast<std::int64_t>(port) * n_records_capacity +
      static_cast<std::int64_t>(ray) * static_cast<std::int64_t>(cap_per_ray);
  const int base_g = ray_group_base[ray];
  const bool cbet_active_ray = (ray_overflow[ray] == 0);
  // Clamp-injected power is tallied per ray (exclusive row) and reduced in
  // fixed order afterwards — no atomicAdd(double), bitwise deterministic.
  double clamp_acc = 0.0;
  int next_capture = 0;
  const bool track_ref =
      (final_pass != 0 && port == viz_port &&
       traj_rec_ratio != nullptr && traj_n_output_rays > 0 &&
       traj_output_stride > 0);
  int output_slot = -1;
  if (track_ref) {
    const long long local_ray = static_cast<long long>(ray) - traj_ray_offset;
    if (local_ray >= 0 && local_ray % traj_output_stride == 0) {
      const long long candidate = local_ray / traj_output_stride;
      if (candidate < traj_n_output_rays) {
        output_slot = static_cast<int>(candidate);
      }
    }
  }
  for (int k = 0; k < cnt; ++k) {
    const std::int64_t slot = base + k;
    const std::int64_t ref_slot =
        static_cast<std::int64_t>(ray) * static_cast<std::int64_t>(cap_per_ray) + k;
    const int c = rec_cell[ref_slot];
    const double S_half = 0.5 * rec_S[ref_slot];
    if (output_slot >= 0) {
      traj_rec_ratio[static_cast<std::int64_t>(output_slot) * cap_per_ray + k] =
          (P_ref > 0.0) ? (P / P_ref) : 1.0;
    }
    if (final_pass != 0 && ps_n_channels > 0 && c >= 0 &&
        c < n_cells) {
      // port_section records are the available event granularity: capture at
      // record entry, before this record's IB halves and CBET exchange.
      while (next_capture < ps_n_channels &&
             ps_cell_nhat[c] >= ps_capture_thresh[next_capture]) {
        double* const capture =
            ps_capture_stage +
            ((static_cast<std::int64_t>(port) * n_rays + ray) *
                 ps_n_channels +
             next_capture) *
                4;
        capture[0] = 1.0;
        capture[1] = P;
        capture[2] = static_cast<double>(rec_mu[ref_slot]);
        capture[3] = static_cast<double>(c);
        P *= ps_one_minus_eta[next_capture];
        if (track_ref) {
          P_ref *= ps_one_minus_eta[next_capture];
        }
        ++next_capture;
      }
    }
    // IB first half
    {
      const double dP = -P * ::expm1(-S_half);
      if (dep_row != nullptr && c >= 0 && c < n_cells) {
        dep_row[c] += dP;
      }
      P -= dP;
    }
    if (track_ref) {
      const double dP_ref = -P_ref * ::expm1(-S_half);
      P_ref -= dP_ref;
    }
    // CBET exchange (share of the group's net dQ by previous-iteration weight)
    const double ds = rec_ds[ref_slot];
    if (cbet_active_ray && ds > 0.0 && c >= 0 && c < n_cells &&
        cell_mask[c] != 0) {
      const int g = base_g + ((rec_mu[ref_slot] > 0.0f) ? n_bins : 0);
      const std::int64_t gc =
          static_cast<std::int64_t>(c) * G_ps + port * G_ref + g;
      const double Lgc = L[gc];
      if (Lgc > 0.0) {
        const double dP_cbet = dQ[gc] * (rec_w_ps[slot] * ds / Lgc);
        P += dP_cbet;
        if (P < 0.0) {
          clamp_acc += -P;
          P = 0.0;
          atomicAdd(clamp_count, 1ULL);
        }
      }
    }
    rec_w_ps[slot] = P;
    // IB second half
    {
      const double dP = -P * ::expm1(-S_half);
      if (dep_row != nullptr && c >= 0 && c < n_cells) {
        dep_row[c] += dP;
      }
      P -= dP;
    }
    if (track_ref) {
      const double dP_ref = -P_ref * ::expm1(-S_half);
      P_ref -= dP_ref;
    }
  }
  if (final_pass != 0) {
    unabs_rows[ray] += (::isfinite(P) && P > 0.0) ? P : 0.0;
    if (clamp_rows != nullptr) {
      clamp_rows[ray] += clamp_acc;
    }
  }
}

__global__ void cbet_scale_traj_power_kernel(
    double* __restrict__ traj_power,
    const std::int32_t* __restrict__ traj_rec_idx,
    const int* __restrict__ traj_step_count,
    const double* __restrict__ traj_rec_ratio,
    const std::int32_t* __restrict__ rec_count,
    const long long traj_ray_offset,
    const int traj_n_output_rays,
    const int traj_output_stride,
    const int traj_max_steps,
    const int cap_per_ray,
    const int n_rays) {
  const std::int64_t linear =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t capacity =
      static_cast<std::int64_t>(traj_n_output_rays) * traj_max_steps;
  if (linear >= capacity) {
    return;
  }
  const int output_slot = static_cast<int>(linear / traj_max_steps);
  const int point = static_cast<int>(linear % traj_max_steps);
  if (point >= traj_step_count[output_slot]) {
    return;
  }
  const long long ray_ll =
      traj_ray_offset +
      static_cast<long long>(output_slot) * traj_output_stride;
  if (ray_ll < 0 || ray_ll >= n_rays) {
    return;
  }
  const int cnt = rec_count[static_cast<int>(ray_ll)];
  if (cnt <= 0) {
    return;
  }
  int rec = traj_rec_idx[linear];
  if (rec < 0) {
    rec = 0;
  } else if (rec >= cnt) {
    rec = cnt - 1;
  }
  traj_power[linear] *=
      traj_rec_ratio[static_cast<std::int64_t>(output_slot) * cap_per_ray + rec];
}

// Fused deterministic reductions over the [n_cells*G] tallies.
// out[0] = sum |L - L_prev|, out[1] = sum L_prev, out[2] = sum dQ,
// out[3] = sum |dQ|, out[4] = sum exch_partial.
__global__ void cbet_iter_stats_kernel(const double* __restrict__ L,
                                       const double* __restrict__ L_prev,
                                       const double* __restrict__ dQ,
                                       const double* __restrict__ exch_partial,
                                       double* __restrict__ out,
                                       const std::int64_t n) {
  __shared__ double sh[5][256];
  const int t = threadIdx.x;
  double a0 = 0.0;
  double a1 = 0.0;
  double a2 = 0.0;
  double a3 = 0.0;
  double a4 = 0.0;
  for (std::int64_t i = t; i < n; i += 256) {
    a0 += ::fabs(L[i] - L_prev[i]);
    a1 += L_prev[i];
    a2 += dQ[i];
    a3 += ::fabs(dQ[i]);
    a4 += exch_partial[i];
  }
  sh[0][t] = a0;
  sh[1][t] = a1;
  sh[2][t] = a2;
  sh[3][t] = a3;
  sh[4][t] = a4;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (t < stride) {
      sh[0][t] += sh[0][t + stride];
      sh[1][t] += sh[1][t + stride];
      sh[2][t] += sh[2][t + stride];
      sh[3][t] += sh[3][t + stride];
      sh[4][t] += sh[4][t + stride];
    }
    __syncthreads();
  }
  if (t == 0) {
    out[0] = sh[0][0];
    out[1] = sh[1][0];
    out[2] = sh[2][0];
    out[3] = sh[3][0];
    out[4] = sh[4][0];
  }
}

__device__ __forceinline__ double cbet_atomic_add_double(double* address, const double value) {
  return atomicAdd(address, value);
}

__global__ void cbet_build_keys_2d_kernel(const std::int32_t* __restrict__ rec_cell,
                                          const float* __restrict__ rec_mu,
                                          const float* __restrict__ rec_c,
                                          const std::int32_t* __restrict__ rec_count,
                                          const std::int32_t* __restrict__ ray_group_base,
                                          const std::int64_t* __restrict__ ray_rec_offset,
                                          std::int64_t* __restrict__ sort_key,
                                          std::int32_t* __restrict__ sort_slot,
                                          const int cap_per_ray,
                                          const int n_bins,
                                          const int G,
                                          const int n_rays) {
  const int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= n_rays) {
    return;
  }
  const int cnt = rec_count[ray];
  const std::int64_t out_base = ray_rec_offset[ray];
  const std::int64_t in_base =
      static_cast<std::int64_t>(ray) * static_cast<std::int64_t>(cap_per_ray);
  const int base_g = ray_group_base[ray];
  for (int k = 0; k < cnt; ++k) {
    const std::int64_t slot = in_base + k;
    const int branch = ((rec_mu[slot] > 0.0f) ? 1 : 0) + ((rec_c[slot] > 0.0f) ? 2 : 0);
    const int g = base_g + branch * n_bins;
    const std::int64_t key =
        static_cast<std::int64_t>(rec_cell[slot]) * static_cast<std::int64_t>(G) +
        static_cast<std::int64_t>(g);
    sort_key[out_base + k] = key;
    sort_slot[out_base + k] = static_cast<std::int32_t>(slot);
  }
}

__global__ void cbet_tally_2d_kernel(const std::int64_t* __restrict__ seg_offsets,
                                     const std::int32_t* __restrict__ sort_slot,
                                     const double* __restrict__ rec_w,
                                     const double* __restrict__ rec_ds,
                                     const float* __restrict__ rec_mu,
                                     const float* __restrict__ rec_c,
                                     double* __restrict__ L,
                                     double* __restrict__ Mmu,
                                     double* __restrict__ Mc,
                                     double* __restrict__ ds_max_gc,
                                     const std::int64_t n_segments) {
  const std::int64_t s = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  if (s >= n_segments) {
    return;
  }
  double l = 0.0;
  double ma = 0.0;
  double mc = 0.0;
  double dmax = 0.0;
  const std::int64_t k0 = seg_offsets[s];
  const std::int64_t k1 = seg_offsets[s + 1];
  for (std::int64_t k = k0; k < k1; ++k) {
    const std::int32_t slot = sort_slot[k];
    const double ds = rec_ds[slot];
    const double wd = rec_w[slot] * ds;
    l += wd;
    ma += wd * static_cast<double>(rec_mu[slot]);
    mc += wd * static_cast<double>(rec_c[slot]);
    dmax = ::fmax(dmax, ds);
  }
  L[s] = l;
  Mmu[s] = ma;
  Mc[s] = mc;
  ds_max_gc[s] = dmax;
}

__global__ void cbet_losscap_2d_kernel(const double* __restrict__ L,
                                       const double* __restrict__ Mmu,
                                       const double* __restrict__ Mc,
                                       const double* __restrict__ ds_max_gc,
                                       const double* __restrict__ omega_group,
                                       const double* __restrict__ cell_chi_pref,
                                       const double* __restrict__ cell_c_a,
                                       const double* __restrict__ cell_u_r,
                                       const double* __restrict__ cell_u_z,
                                       const double* __restrict__ cell_k_bar,
                                       const double* __restrict__ cell_vol,
                                       const std::uint8_t* __restrict__ cell_mask,
                                       double* __restrict__ fcap,
                                       const double theta_cap,
                                       const double f_cbet,
                                       const double alpha_iaw,
                                       const double k_a_floor,
                                       const double test_chi,
                                       const int G,
                                       const int n_cells) {
  const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(n_cells) * G;
  if (idx >= total) {
    return;
  }
  const int c = static_cast<int>(idx / G);
  const int g = static_cast<int>(idx % G);
  double f = 1.0;
  const double Lg = L[idx];
  if (cell_mask[c] != 0 && Lg > 0.0 && cell_vol[c] > 0.0) {
    const double inv_vol = 1.0 / cell_vol[c];
    const std::int64_t row = static_cast<std::int64_t>(c) * G;
    double loss = 0.0;
    for (int gp = 0; gp < G; ++gp) {
      if (gp == g) {
        continue;
      }
      const double Lgp = L[row + gp];
      if (!(Lgp > 0.0)) {
        continue;
      }
      // Evaluate the coupling with (p,q) = (min,max) group ids so both orientations
      // compute bitwise-identical values (exact pairwise antisymmetry).
      const int gmin = ::min(g, gp);
      const int gmax = ::max(g, gp);
      const std::int64_t imin = row + gmin;
      const std::int64_t imax = row + gmax;
      const double Lmin = L[imin];
      const double Lmax = L[imax];
      const double chi_pq = cbet_pair_coupling_2d(
          Mmu[imin] / Lmin, Mc[imin] / Lmin, Mmu[imax] / Lmax, Mc[imax] / Lmax,
          omega_group[gmin], omega_group[gmax], cell_chi_pref[c], cell_c_a[c],
          cell_u_r[c], cell_u_z[c], cell_k_bar[c], f_cbet, alpha_iaw, k_a_floor,
          test_chi);
      // Canonical (min,max) product order — see the 1D kernels (M-01/S-04).
      const double amt = chi_pq * Lmin * Lmax * inv_vol;
      const double signed_amt = (g < gp) ? amt : -amt;
      loss += ::fmax(0.0, -signed_amt);
    }
    const double ds_max = ds_max_gc[idx];
    if (loss > 0.0 && ds_max > 0.0) {
      const double cap = theta_cap * Lg / ds_max;
      f = ::fmin(1.0, cap / loss);
    }
  }
  fcap[idx] = f;
}

__global__ void cbet_dq_2d_kernel(const double* __restrict__ L,
                                  const double* __restrict__ Mmu,
                                  const double* __restrict__ Mc,
                                  const double* __restrict__ fcap,
                                  const double* __restrict__ omega_group,
                                  const double* __restrict__ cell_chi_pref,
                                  const double* __restrict__ cell_c_a,
                                  const double* __restrict__ cell_u_r,
                                  const double* __restrict__ cell_u_z,
                                  const double* __restrict__ cell_k_bar,
                                  const double* __restrict__ cell_vol,
                                  const std::uint8_t* __restrict__ cell_mask,
                                  double* __restrict__ dQ,
                                  double* __restrict__ exch_partial,
                                  unsigned long long* __restrict__ capped_pairs,
                                  const double f_cbet,
                                  const double alpha_iaw,
                                  const double k_a_floor,
                                  const double test_chi,
                                  const int G,
                                  const int n_cells) {
  const std::int64_t idx = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(n_cells) * G;
  if (idx >= total) {
    return;
  }
  const int c = static_cast<int>(idx / G);
  const int g = static_cast<int>(idx % G);
  double net = 0.0;
  double exch = 0.0;
  const double Lg = L[idx];
  if (cell_mask[c] != 0 && Lg > 0.0 && cell_vol[c] > 0.0) {
    const double inv_vol = 1.0 / cell_vol[c];
    const std::int64_t row = static_cast<std::int64_t>(c) * G;
    const double f_g = fcap[idx];
    for (int gp = 0; gp < G; ++gp) {
      if (gp == g) {
        continue;
      }
      const double Lgp = L[row + gp];
      if (!(Lgp > 0.0)) {
        continue;
      }
      const int gmin = ::min(g, gp);
      const int gmax = ::max(g, gp);
      const std::int64_t imin = row + gmin;
      const std::int64_t imax = row + gmax;
      const double Lmin = L[imin];
      const double Lmax = L[imax];
      const double chi_pq = cbet_pair_coupling_2d(
          Mmu[imin] / Lmin, Mc[imin] / Lmin, Mmu[imax] / Lmax, Mc[imax] / Lmax,
          omega_group[gmin], omega_group[gmax], cell_chi_pref[c], cell_c_a[c],
          cell_u_r[c], cell_u_z[c], cell_k_bar[c], f_cbet, alpha_iaw, k_a_floor,
          test_chi);
      const double f_gp = fcap[row + gp];
      const double fmin_pair = ::fmin(f_g, f_gp);
      // Canonical (min,max) product order — see the 1D kernels (M-01/S-04).
      const double amt = chi_pq * Lmin * Lmax * inv_vol * fmin_pair;
      const double signed_amt = (g < gp) ? amt : -amt;
      net += signed_amt;
      if (g < gp) {
        exch += ::fabs(amt);
        if (fmin_pair < 1.0 && amt != 0.0) {
          atomicAdd(capped_pairs, 1ULL);
        }
      }
    }
  }
  dQ[idx] = net;
  exch_partial[idx] = exch;
}

__global__ void cbet_propagate_2d_kernel(const std::int32_t* __restrict__ rec_cell,
                                         const float* __restrict__ rec_mu,
                                         const float* __restrict__ rec_c,
                                         const float* __restrict__ rec_w00,
                                         const float* __restrict__ rec_w10,
                                         const float* __restrict__ rec_w01,
                                         const double* __restrict__ rec_ds,
                                         const double* __restrict__ rec_S,
                                         double* __restrict__ rec_w,
                                         const std::int32_t* __restrict__ rec_count,
                                         const std::int32_t* __restrict__ ray_group_base,
                                         const double* __restrict__ ray_P0,
                                         const std::uint8_t* __restrict__ ray_overflow,
                                         const double* __restrict__ dQ,
                                         const double* __restrict__ L,
                                         const std::uint8_t* __restrict__ cell_mask,
                                         double* __restrict__ dep_nodes,
                                         double* __restrict__ unabs_rows,
                                         unsigned long long* __restrict__ clamp_count,
                                         double* __restrict__ clamp_rows,
                                         const int cap_per_ray,
                                         const int n_bins,
                                         const int n_branches,
                                         const int G,
                                         const int n_cells,
                                         const int nz_cells,
                                         const int n_nodes_z,
                                         const int n_nodes,
                                         const int n_rays,
                                         const int final_pass) {
  const int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= n_rays) {
    return;
  }
  const int cnt = rec_count[ray];
  double P = ray_P0[ray];
  if (!::isfinite(P) || P < 0.0) {
    P = 0.0;
  }
  const int base_g = ray_group_base[ray];
  const int beam = base_g / (n_branches * n_bins);
  double* dep_row = (final_pass != 0)
                        ? dep_nodes + static_cast<std::int64_t>(beam) * n_nodes
                        : nullptr;
  const std::int64_t base =
      static_cast<std::int64_t>(ray) * static_cast<std::int64_t>(cap_per_ray);
  const bool cbet_active_ray = (ray_overflow[ray] == 0);
  // Per-ray clamp tally, reduced in fixed order afterwards (no double atomic).
  double clamp_acc = 0.0;
  for (int k = 0; k < cnt; ++k) {
    const std::int64_t slot = base + k;
    const int c = rec_cell[slot];
    const double S_half = 0.5 * rec_S[slot];
    double dep_acc = 0.0;
    // IB first half
    {
      const double dP = -P * ::expm1(-S_half);
      dep_acc += dP;
      P -= dP;
    }
    // CBET exchange (share of the group's net dQ by previous-iteration weight)
    const double ds = rec_ds[slot];
    if (cbet_active_ray && ds > 0.0 && c >= 0 && c < n_cells && cell_mask[c] != 0) {
      const int branch = ((rec_mu[slot] > 0.0f) ? 1 : 0) + ((rec_c[slot] > 0.0f) ? 2 : 0);
      const int g = base_g + branch * n_bins;
      const std::int64_t gc = static_cast<std::int64_t>(c) * G + g;
      const double Lgc = L[gc];
      if (Lgc > 0.0) {
        const double dP_cbet = dQ[gc] * (rec_w[slot] * ds / Lgc);
        P += dP_cbet;
        if (P < 0.0) {
          clamp_acc += -P;
          P = 0.0;
          atomicAdd(clamp_count, 1ULL);
        }
      }
    }
    rec_w[slot] = P;
    // IB second half
    {
      const double dP = -P * ::expm1(-S_half);
      dep_acc += dP;
      P -= dP;
    }
    if (dep_row != nullptr && dep_acc > 0.0 && c >= 0 && c < n_cells) {
      const int ci = c / nz_cells;
      const int cj = c % nz_cells;
      const int n00 = ci * n_nodes_z + cj;
      const int n10 = (ci + 1) * n_nodes_z + cj;
      const int n01 = ci * n_nodes_z + (cj + 1);
      const int n11 = (ci + 1) * n_nodes_z + (cj + 1);
      const double w00 = static_cast<double>(rec_w00[slot]);
      const double w10 = static_cast<double>(rec_w10[slot]);
      const double w01 = static_cast<double>(rec_w01[slot]);
      const double w11 = ::fmax(0.0, 1.0 - w00 - w10 - w01);
      if (w00 != 0.0) cbet_atomic_add_double(&dep_row[n00], dep_acc * w00);
      if (w10 != 0.0) cbet_atomic_add_double(&dep_row[n10], dep_acc * w10);
      if (w01 != 0.0) cbet_atomic_add_double(&dep_row[n01], dep_acc * w01);
      if (w11 != 0.0) cbet_atomic_add_double(&dep_row[n11], dep_acc * w11);
    }
  }
  if (final_pass != 0) {
    unabs_rows[ray] = (::isfinite(P) && P > 0.0) ? P : 0.0;
    if (clamp_rows != nullptr) {
      clamp_rows[ray] = clamp_acc;
    }
  }
}

__global__ void cbet_sum_rows_kernel(const double* __restrict__ rows,
                                     const int first,
                                     const int count,
                                     double* __restrict__ out_add) {
  __shared__ double sh[256];
  const int t = threadIdx.x;
  double a = 0.0;
  for (int i = t; i < count; i += 256) {
    a += rows[first + i];
  }
  sh[t] = a;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (t < stride) {
      sh[t] += sh[t + stride];
    }
    __syncthreads();
  }
  if (t == 0) {
    out_add[0] += sh[0];
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// workspace management
// ---------------------------------------------------------------------------

CbetWorkspace::~CbetWorkspace() {
  release();
}

void CbetWorkspace::release() {
  auto free_ptr = [](auto** p) {
    if (*p != nullptr) {
      static_cast<void>(cudaFree(*p));
      *p = nullptr;
    }
  };
  free_ptr(&rec_cell);
  free_ptr(&rec_mu);
  free_ptr(&rec_c);
  free_ptr(&rec_w00);
  free_ptr(&rec_w10);
  free_ptr(&rec_w01);
  free_ptr(&rec_ds);
  free_ptr(&rec_S);
  free_ptr(&rec_w);
  free_ptr(&rec_w_ps);
  free_ptr(&ps_cell_nhat);
  free_ptr(&ps_capture_thresh);
  free_ptr(&ps_capture_order);
  free_ptr(&ps_one_minus_eta);
  free_ptr(&ps_capture_stage);
  free_ptr(&rec_count);
  free_ptr(&ray_group_base);
  free_ptr(&ray_P0);
  free_ptr(&ray_overflow);
  free_ptr(&ray_rec_offset);
  free_ptr(&sort_key);
  free_ptr(&sort_slot);
  free_ptr(&seg_offsets);
  free_ptr(&L);
  free_ptr(&L_prev);
  free_ptr(&Mmu);
  free_ptr(&Mc);
  free_ptr(&ds_max_gc);
  free_ptr(&fcap);
  free_ptr(&dQ);
  free_ptr(&iaw_partial);
  free_ptr(&exch_partial);
  free_ptr(&iaw_cell);
  free_ptr(&chi);
  free_ptr(&pair_p);
  free_ptr(&pair_q);
  free_ptr(&pair_index);
  free_ptr(&omega_group);
  free_ptr(&port_weight);
  free_ptr(&cell_chi_pref);
  free_ptr(&cell_c_a);
  free_ptr(&cell_u_r);
  free_ptr(&cell_u_z);
  free_ptr(&cell_k_bar);
  free_ptr(&cell_vol);
  free_ptr(&cell_mask);
  free_ptr(&dep_rows);
  free_ptr(&dep_nodes);
  free_ptr(&unabs_rows);
  free_ptr(&clamp_rows);
  free_ptr(&d_scalars);
  free_ptr(&d_iaw_sum);
  free_ptr(&d_counters);
  cap_records = cap_rays = cap_gc = cap_chi = cap_cells = cap_rows = 0;
  cap_pair_list = cap_pair_index = cap_omega = 0;
  cap_dep_nodes = 0;
  cap_rec_w_ps = cap_port_weight = 0;
  cap_ps_cell_nhat = cap_ps_capture_thresh = cap_ps_capture_order = 0;
  cap_ps_one_minus_eta = cap_ps_capture_stage = 0;
  n_rays_total = cap_per_ray = n_cells = n_beams = n_bins = n_groups = n_pairs = 0;
  n_branches = 2;
  ps_mode = false;
  n_ports = 0;
  G_ref = 0;
  ps_n_channels = 0;
  dim2d = false;
  n_nodes = 0;
  nz_cells = 0;
  n_nodes_z = 0;
  n_records_live = 0;
  beam_ray_offset.clear();
  beam_ray_count.clear();
  beam_omega.clear();
}

CbetWorkspace& global_cbet_workspace() {
  // Deliberately leaked (mirrors global_skip_cache): CUDA teardown order at
  // process exit makes destructor-time cudaFree unsafe.
  static CbetWorkspace* ws = new CbetWorkspace();
  return *ws;
}

void invalidate_global_cbet_workspace() {
  global_cbet_workspace().release();
}

void cbet_workspace_prepare(CbetWorkspace& ws,
                            const int n_rays_total,
                            const int cap_per_ray,
                            const int n_cells,
                            const int n_beams,
                            const int n_impact_bins,
                            cudaStream_t stream,
                            const int n_branches,
                            const bool dim2d,
                            const int n_nodes,
                            const int nz_cells,
                            const int n_nodes_z,
                            const int n_ports,
                            const int ps_n_channels) {
  TENRYU_ASSERT(n_rays_total > 0, "cbet_workspace_prepare requires n_rays_total > 0");
  TENRYU_ASSERT(cap_per_ray > 0, "cbet_workspace_prepare requires cap_per_ray > 0");
  TENRYU_ASSERT(n_cells > 0, "cbet_workspace_prepare requires n_cells > 0");
  TENRYU_ASSERT(n_beams > 0, "cbet_workspace_prepare requires n_beams > 0");
  TENRYU_ASSERT(n_impact_bins > 0, "cbet_workspace_prepare requires n_impact_bins > 0");
  TENRYU_ASSERT(n_branches > 0, "cbet_workspace_prepare requires n_branches > 0");
  TENRYU_ASSERT(n_ports >= 0, "cbet_workspace_prepare requires n_ports >= 0");
  TENRYU_ASSERT(
      ps_n_channels >= 0 &&
          ps_n_channels <=
              core::Config::LaserConfig::HotElectronConfig::kMaxSources,
      "cbet_workspace_prepare ps_n_channels exceeds the hot-electron channel cap");
  TENRYU_ASSERT(!dim2d || (n_nodes > 0 && nz_cells > 0 && n_nodes_z > 0),
                "cbet_workspace_prepare 2D requires n_nodes, nz_cells, and n_nodes_z > 0");
  TENRYU_ASSERT(n_ports == 0 || !dim2d, "port_section is 1D-only");
  TENRYU_ASSERT(n_ports == 0 || n_beams == 1,
                "port_section requires exactly one reference beam");
  TENRYU_ASSERT(ps_n_channels == 0 || n_ports > 0,
                "ps hot-electron capture requires port_section mode");
  ws.n_rays_total = n_rays_total;
  ws.cap_per_ray = cap_per_ray;
  ws.n_cells = n_cells;
  ws.n_beams = n_beams;
  ws.n_bins = n_impact_bins;
  ws.n_branches = n_branches;
  ws.dim2d = dim2d;
  ws.n_nodes = n_nodes;
  ws.nz_cells = nz_cells;
  ws.n_nodes_z = n_nodes_z;
  ws.ps_mode = n_ports > 0;
  ws.n_ports = n_ports;
  ws.G_ref = ws.ps_mode ? n_branches * n_impact_bins : 0;
  ws.ps_n_channels = ps_n_channels;
  // pair_p/pair_q are int16_t and n_pairs = G(G-1)/2 must fit in int, so the
  // group count is hard-bounded; evaluate the product in 64-bit before the
  // int assignment (AI review C-01 companion check).
  const long long n_groups_ref_64 = static_cast<long long>(n_beams) *
                                    static_cast<long long>(n_branches) *
                                    static_cast<long long>(n_impact_bins);
  const long long n_groups_64 =
      ws.ps_mode ? static_cast<long long>(n_ports) * ws.G_ref
                 : n_groups_ref_64;
  TENRYU_ASSERT(n_groups_64 <= 32767,
                "cbet group count n_beams*n_branches*n_impact_bins exceeds 32767 "
                "(int16 pair-table limit); reduce Laser.cbet.n_impact_bins");
  ws.n_groups = static_cast<int>(n_groups_64);
  ws.n_pairs = ws.n_groups * (ws.n_groups - 1) / 2;
  TENRYU_ASSERT(ws.n_groups >= 2, "cbet requires at least 2 groups");
  TENRYU_ASSERT(!ws.ps_mode || ws.n_pairs <= 65536,
                "port_section v1 supports at most 65536 expanded pairs");

  // Grow-only allocation groups (no per-solve malloc when sizes are stable —
  // per-call cudaMalloc is the measured W-F host-cost antipattern).
  const std::size_t n_rec = static_cast<std::size_t>(n_rays_total) *
                            static_cast<std::size_t>(cap_per_ray);
  const std::size_t n_rec_ps =
      static_cast<std::size_t>(n_ports) * n_rec;
  const std::size_t n_ps_capture =
      static_cast<std::size_t>(n_ports) *
      static_cast<std::size_t>(n_rays_total) *
      static_cast<std::size_t>(ps_n_channels) * 4U;
  const std::size_t n_ray = static_cast<std::size_t>(n_rays_total);
  const std::size_t n_gc = static_cast<std::size_t>(ws.n_groups) *
                           static_cast<std::size_t>(n_cells);
  const std::size_t n_chi = static_cast<std::size_t>(n_cells) *
                            static_cast<std::size_t>(ws.n_pairs);
  const std::size_t n_pair_tbl = static_cast<std::size_t>(ws.n_groups) * ws.n_groups;
  const std::size_t n_cell = static_cast<std::size_t>(n_cells);
  const std::size_t n_row = n_ray * n_cell;
  const std::size_t n_dep_nodes = static_cast<std::size_t>(n_beams) *
                                  static_cast<std::size_t>(n_nodes);
  const bool records_need_realloc = (n_rec > ws.cap_records || ws.rec_cell == nullptr);
  const bool gc_need_realloc = (n_gc > ws.cap_gc || ws.L == nullptr);
  const bool cells_need_realloc = (n_cell > ws.cap_cells || ws.cell_chi_pref == nullptr);
  if (dim2d) {
    const auto bytes = [](const std::size_t n, const std::size_t elem) {
      return static_cast<unsigned long long>(n) * static_cast<unsigned long long>(elem);
    };
    unsigned long long projected = 0ULL;
    projected += bytes(n_rec, sizeof(std::int32_t));        // rec_cell
    projected += bytes(n_rec, 5ULL * sizeof(float));        // rec_mu, rec_c, rec_w00/10/01
    projected += bytes(n_rec, 3ULL * sizeof(double));       // rec_ds, rec_S, rec_w
    projected += bytes(n_rec, sizeof(std::int64_t));        // sort_key
    projected += bytes(n_rec, sizeof(std::int32_t));        // sort_slot
    projected += bytes(n_ray, 2ULL * sizeof(std::int32_t)); // rec_count, ray_group_base
    projected += bytes(n_ray, sizeof(double));              // ray_P0
    projected += bytes(n_ray, sizeof(std::uint8_t));        // ray_overflow
    projected += bytes(n_ray + 1U, sizeof(std::int64_t));   // ray_rec_offset
    projected += bytes(n_gc, 9ULL * sizeof(double));        // L/L_prev/Mmu/Mc/ds/fcap/dQ/iaw/exch
    projected += bytes(n_gc + 1U, sizeof(std::int64_t));    // seg_offsets
    projected += bytes(static_cast<std::size_t>(ws.n_pairs), 2ULL * sizeof(std::int16_t));
    projected += bytes(n_pair_tbl, sizeof(std::int32_t));
    projected += bytes(static_cast<std::size_t>(ws.n_groups), sizeof(double));
    projected += bytes(n_cell, 7ULL * sizeof(double));      // cell fields + iaw_cell
    projected += bytes(n_cell, sizeof(std::uint8_t));       // cell_mask
    projected += bytes(n_dep_nodes, sizeof(double));        // dep_nodes
    projected += bytes(n_ray, sizeof(double));              // unabs_rows
    projected += bytes(9U, sizeof(double));
    projected += bytes(4U, sizeof(unsigned long long));
    const double projected_gib =
        static_cast<double>(projected) / static_cast<double>(1ULL << 30);
    TENRYU_ASSERT(
        projected <= (8ULL << 30),
        std::string("cbet 2D workspace projected size ") + std::to_string(projected_gib) +
            " GiB exceeds 8 GiB; reduce Laser.rays_per_beam / "
            "Laser.cbet.n_impact_bins / Laser.cbet.max_segments_per_ray "
            "or coarsen Laser.lasermesh");
    if (records_need_realloc || gc_need_realloc) {
      const double projected_mib =
          static_cast<double>(projected) / static_cast<double>(1ULL << 20);
      core::log_info("[cbet_2d_workspace] projected=" + std::to_string(projected_mib) +
                     " MiB");
    }
  }
  if (records_need_realloc) {
    std::size_t cap = ws.cap_records;
    ensure_device_capacity(&ws.rec_cell, &cap, n_rec, "cbet rec_cell alloc");
    cap = ws.cap_records;
    ensure_device_capacity(&ws.rec_mu, &cap, n_rec, "cbet rec_mu alloc");
    cap = ws.cap_records;
    ensure_device_capacity(&ws.rec_ds, &cap, n_rec, "cbet rec_ds alloc");
    cap = ws.cap_records;
    ensure_device_capacity(&ws.rec_S, &cap, n_rec, "cbet rec_S alloc");
    cap = ws.cap_records;
    ensure_device_capacity(&ws.rec_w, &cap, n_rec, "cbet rec_w alloc");
    cap = ws.cap_records;
    ensure_device_capacity(&ws.sort_key, &cap, n_rec, "cbet sort_key alloc");
    cap = ws.cap_records;
    ensure_device_capacity(&ws.sort_slot, &cap, n_rec, "cbet sort_slot alloc");
    ws.cap_records = n_rec;
  }
  if (ws.ps_mode) {
    ensure_device_capacity(&ws.rec_w_ps, &ws.cap_rec_w_ps, n_rec_ps,
                           "cbet rec_w_ps alloc");
    ensure_device_capacity(&ws.port_weight, &ws.cap_port_weight,
                           static_cast<std::size_t>(n_ports),
                           "cbet port_weight alloc");
    if (ps_n_channels > 0) {
      ensure_device_capacity(&ws.ps_cell_nhat, &ws.cap_ps_cell_nhat,
                             n_cell, "cbet ps_cell_nhat alloc");
      ensure_device_capacity(&ws.ps_capture_thresh,
                             &ws.cap_ps_capture_thresh,
                             static_cast<std::size_t>(ps_n_channels),
                             "cbet ps_capture_thresh alloc");
      ensure_device_capacity(&ws.ps_capture_order,
                             &ws.cap_ps_capture_order,
                             static_cast<std::size_t>(ps_n_channels),
                             "cbet ps_capture_order alloc");
      ensure_device_capacity(&ws.ps_one_minus_eta,
                             &ws.cap_ps_one_minus_eta,
                             static_cast<std::size_t>(ps_n_channels),
                             "cbet ps_one_minus_eta alloc");
      ensure_device_capacity(&ws.ps_capture_stage,
                             &ws.cap_ps_capture_stage, n_ps_capture,
                             "cbet ps_capture_stage alloc");
    }
  }
  if (dim2d) {
    const std::size_t optional_cap = records_need_realloc ? 0U : ws.cap_records;
    std::size_t cap = optional_cap;
    ensure_device_capacity(&ws.rec_c, &cap, n_rec, "cbet rec_c alloc");
    cap = optional_cap;
    ensure_device_capacity(&ws.rec_w00, &cap, n_rec, "cbet rec_w00 alloc");
    cap = optional_cap;
    ensure_device_capacity(&ws.rec_w10, &cap, n_rec, "cbet rec_w10 alloc");
    cap = optional_cap;
    ensure_device_capacity(&ws.rec_w01, &cap, n_rec, "cbet rec_w01 alloc");
  } else {
    free_device_pointer(&ws.rec_c);
    free_device_pointer(&ws.rec_w00);
    free_device_pointer(&ws.rec_w10);
    free_device_pointer(&ws.rec_w01);
  }
  if (n_ray > ws.cap_rays || ws.rec_count == nullptr) {
    std::size_t cap = ws.cap_rays;
    ensure_device_capacity(&ws.rec_count, &cap, n_ray, "cbet rec_count alloc");
    cap = ws.cap_rays;
    ensure_device_capacity(&ws.ray_group_base, &cap, n_ray, "cbet group_base alloc");
    cap = ws.cap_rays;
    ensure_device_capacity(&ws.ray_P0, &cap, n_ray, "cbet ray_P0 alloc");
    cap = ws.cap_rays;
    ensure_device_capacity(&ws.ray_overflow, &cap, n_ray, "cbet overflow alloc");
    cap = ws.cap_rays;
    ensure_device_capacity(&ws.ray_rec_offset, &cap, n_ray + 1,
                           "cbet ray_rec_offset alloc");
    cap = ws.cap_rays;
    ensure_device_capacity(&ws.unabs_rows, &cap, n_ray, "cbet unabs alloc");
    cap = ws.cap_rays;
    ensure_device_capacity(&ws.clamp_rows, &cap, n_ray, "cbet clamp_rows alloc");
    ws.cap_rays = n_ray;
  }
  if (gc_need_realloc) {
    std::size_t cap = ws.cap_gc;
    ensure_device_capacity(&ws.L, &cap, n_gc, "cbet L alloc");
    cap = ws.cap_gc;
    ensure_device_capacity(&ws.L_prev, &cap, n_gc, "cbet L_prev alloc");
    cap = ws.cap_gc;
    ensure_device_capacity(&ws.Mmu, &cap, n_gc, "cbet Mmu alloc");
    cap = ws.cap_gc;
    ensure_device_capacity(&ws.ds_max_gc, &cap, n_gc, "cbet ds_max alloc");
    cap = ws.cap_gc;
    ensure_device_capacity(&ws.fcap, &cap, n_gc, "cbet fcap alloc");
    cap = ws.cap_gc;
    ensure_device_capacity(&ws.dQ, &cap, n_gc, "cbet dQ alloc");
    cap = ws.cap_gc;
    ensure_device_capacity(&ws.iaw_partial, &cap, n_gc,
                           "cbet iaw_partial alloc");
    cap = ws.cap_gc;
    ensure_device_capacity(&ws.exch_partial, &cap, n_gc, "cbet exch_partial alloc");
    cap = ws.cap_gc;
    ensure_device_capacity(&ws.seg_offsets, &cap, n_gc + 1, "cbet seg_offsets alloc");
    ws.cap_gc = n_gc;
  }
  if (dim2d) {
    const std::size_t optional_cap = gc_need_realloc ? 0U : ws.cap_gc;
    std::size_t cap = optional_cap;
    ensure_device_capacity(&ws.Mc, &cap, n_gc, "cbet Mc alloc");
  } else {
    free_device_pointer(&ws.Mc);
  }
  if (dim2d) {
    free_device_pointer(&ws.chi);
    ws.cap_chi = 0;
  } else {
    ensure_device_capacity(&ws.chi, &ws.cap_chi, n_chi, "cbet chi alloc");
  }
  // Per-array capacities: ensure_device_capacity is a no-op when the tracked
  // capacity already covers the request, so no outer size gate is needed (and
  // a shared gate was the C-01 overrun: pair_p/pair_q/omega_group could be
  // skipped while their element counts grew).
  {
    std::size_t cap_q = ws.cap_pair_list;
    ensure_device_capacity(&ws.pair_p, &ws.cap_pair_list,
                           static_cast<std::size_t>(ws.n_pairs), "cbet pair_p alloc");
    ensure_device_capacity(&ws.pair_q, &cap_q, static_cast<std::size_t>(ws.n_pairs),
                           "cbet pair_q alloc");
    TENRYU_ASSERT(cap_q == ws.cap_pair_list,
                  "cbet pair_p/pair_q capacity tracking diverged");
    ensure_device_capacity(&ws.pair_index, &ws.cap_pair_index, n_pair_tbl,
                           "cbet pair_index alloc");
    ensure_device_capacity(&ws.omega_group, &ws.cap_omega,
                           static_cast<std::size_t>(ws.n_groups), "cbet omega alloc");
  }
  if (cells_need_realloc) {
    std::size_t cap = ws.cap_cells;
    ensure_device_capacity(&ws.cell_chi_pref, &cap, n_cell, "cbet chi_pref alloc");
    cap = ws.cap_cells;
    ensure_device_capacity(&ws.cell_c_a, &cap, n_cell, "cbet c_a alloc");
    cap = ws.cap_cells;
    ensure_device_capacity(&ws.cell_u_r, &cap, n_cell, "cbet u_r alloc");
    cap = ws.cap_cells;
    ensure_device_capacity(&ws.cell_k_bar, &cap, n_cell, "cbet k_bar alloc");
    cap = ws.cap_cells;
    ensure_device_capacity(&ws.cell_vol, &cap, n_cell, "cbet vol alloc");
    cap = ws.cap_cells;
    ensure_device_capacity(&ws.cell_mask, &cap, n_cell, "cbet mask alloc");
    cap = ws.cap_cells;
    ensure_device_capacity(&ws.iaw_cell, &cap, n_cell, "cbet iaw_cell alloc");
    ws.cap_cells = n_cell;
  }
  if (dim2d) {
    const std::size_t optional_cap = cells_need_realloc ? 0U : ws.cap_cells;
    std::size_t cap = optional_cap;
    ensure_device_capacity(&ws.cell_u_z, &cap, n_cell, "cbet u_z alloc");
  } else {
    free_device_pointer(&ws.cell_u_z);
  }
  if (dim2d) {
    free_device_pointer(&ws.dep_rows);
    ws.cap_rows = 0;
    ensure_device_capacity(&ws.dep_nodes, &ws.cap_dep_nodes, n_dep_nodes,
                           "cbet dep_nodes alloc");
  } else {
    free_device_pointer(&ws.dep_nodes);
    ws.cap_dep_nodes = 0;
    ensure_device_capacity(&ws.dep_rows, &ws.cap_rows, n_row, "cbet dep_rows alloc");
  }
  if (ws.d_scalars == nullptr) {
    std::size_t cap = 0;
    ensure_device_capacity(&ws.d_scalars, &cap, static_cast<std::size_t>(8),
                           "cbet scalars alloc");
    cap = 0;
    ensure_device_capacity(&ws.d_counters, &cap, static_cast<std::size_t>(4),
                           "cbet counters alloc");
  }
  if (ws.d_iaw_sum == nullptr) {
    std::size_t cap = 0;
    ensure_device_capacity(&ws.d_iaw_sum, &cap, static_cast<std::size_t>(1),
                           "cbet iaw sum alloc");
  }

  if (!ws.ps_mode) {
    // static pair tables (host build, H2D)
    std::vector<std::int16_t> h_pair_p;
    std::vector<std::int16_t> h_pair_q;
    h_pair_p.reserve(static_cast<std::size_t>(ws.n_pairs));
    h_pair_q.reserve(static_cast<std::size_t>(ws.n_pairs));
    std::vector<std::int32_t> h_pair_index(
        static_cast<std::size_t>(ws.n_groups) * ws.n_groups, -1);
    int pair = 0;
    for (int p = 0; p < ws.n_groups; ++p) {
      for (int q = p + 1; q < ws.n_groups; ++q) {
        h_pair_p.push_back(static_cast<std::int16_t>(p));
        h_pair_q.push_back(static_cast<std::int16_t>(q));
        h_pair_index[static_cast<std::size_t>(p) * ws.n_groups + q] = pair;
        h_pair_index[static_cast<std::size_t>(q) * ws.n_groups + p] = pair;
        ++pair;
      }
    }
    TENRYU_ASSERT(pair == ws.n_pairs, "cbet pair table construction mismatch");
    cbet_check(cudaMemcpyAsync(ws.pair_p, h_pair_p.data(),
                               h_pair_p.size() * sizeof(std::int16_t),
                               cudaMemcpyHostToDevice, stream),
               "cbet pair_p H2D failed");
    cbet_check(cudaMemcpyAsync(ws.pair_q, h_pair_q.data(),
                               h_pair_q.size() * sizeof(std::int16_t),
                               cudaMemcpyHostToDevice, stream),
               "cbet pair_q H2D failed");
    cbet_check(cudaMemcpyAsync(ws.pair_index, h_pair_index.data(),
                               h_pair_index.size() * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream),
               "cbet pair_index H2D failed");
  }
  cbet_check(cudaMemsetAsync(ws.rec_count, 0,
                             static_cast<std::size_t>(n_rays_total) * sizeof(std::int32_t),
                             stream),
             "cbet rec_count memset failed");
  cbet_check(cudaMemsetAsync(ws.ray_overflow, 0,
                             static_cast<std::size_t>(n_rays_total) * sizeof(std::uint8_t),
                             stream),
             "cbet overflow memset failed");
  cbet_check(cudaMemsetAsync(ws.ray_P0, 0,
                             static_cast<std::size_t>(n_rays_total) * sizeof(double),
                             stream),
             "cbet ray_P0 memset failed");
  cbet_check(cudaMemsetAsync(ws.ray_group_base, 0,
                             static_cast<std::size_t>(n_rays_total) *
                                 sizeof(std::int32_t),
                             stream),
             "cbet ray_group_base memset failed");
  // pair tables H2D are consumed by later kernels on the same stream; the host
  // vectors go out of scope here, so block until the copies land.
  cbet_check(cudaStreamSynchronize(stream), "cbet prepare stream sync failed");
  ws.beam_ray_offset.assign(static_cast<std::size_t>(n_beams), 0);
  ws.beam_ray_count.assign(static_cast<std::size_t>(n_beams), 0);
  ws.beam_omega.assign(static_cast<std::size_t>(n_beams), 0.0);
  ws.n_records_live = 0;
}

void cbet_stage_ray_meta(CbetWorkspace& ws,
                         const int beam_index,
                         const int ray_offset,
                         const int count,
                         const double omega_beam,
                         const double* d_ray_power,
                         cudaStream_t stream) {
  TENRYU_ASSERT(beam_index >= 0 && beam_index < ws.n_beams,
                "cbet_stage_ray_meta beam_index out of range");
  TENRYU_ASSERT(ray_offset >= 0 && count >= 0 &&
                    ray_offset + count <= ws.n_rays_total,
                "cbet_stage_ray_meta ray range out of bounds");
  ws.beam_ray_offset[static_cast<std::size_t>(beam_index)] = ray_offset;
  ws.beam_ray_count[static_cast<std::size_t>(beam_index)] = count;
  ws.beam_omega[static_cast<std::size_t>(beam_index)] = omega_beam;
  if (count == 0) {
    return;
  }
  std::vector<std::int32_t> h_base(static_cast<std::size_t>(count));
  const int base = beam_index * ws.n_branches * ws.n_bins;
  for (int k = 0; k < count; ++k) {
    const int bin = std::min(ws.n_bins - 1,
                             static_cast<int>((static_cast<long long>(k) * ws.n_bins) /
                                              std::max(count, 1)));
    h_base[static_cast<std::size_t>(k)] = base + bin;
  }
  cbet_check(cudaMemcpyAsync(ws.ray_group_base + ray_offset, h_base.data(),
                             h_base.size() * sizeof(std::int32_t),
                             cudaMemcpyHostToDevice, stream),
             "cbet group_base H2D failed");
  cbet_check(cudaMemcpyAsync(ws.ray_P0 + ray_offset, d_ray_power,
                             static_cast<std::size_t>(count) * sizeof(double),
                             cudaMemcpyDeviceToDevice, stream),
             "cbet ray_P0 D2D failed");
  cbet_check(cudaStreamSynchronize(stream), "cbet stage_ray_meta stream sync failed");
}

void cbet_stage_cell_fields(CbetWorkspace& ws,
                            const core::State& state,
                            const HydroMirror1D& mirror,
                            const core::Config::LaserConfig& laser,
                            const double lambda0_cm,
                            cudaStream_t stream) {
  const int n_cells = ws.n_cells;
  TENRYU_ASSERT(static_cast<int>(mirror.rho.size()) == n_cells &&
                    static_cast<int>(mirror.zbar.size()) == n_cells &&
                    static_cast<int>(mirror.A_eff.size()) == n_cells,
                "cbet_stage_cell_fields mirror size mismatch");
  const double c = core::constants::c_light;
  const double eV = core::constants::eV_to_erg;
  const double omega0 = 2.0 * kPi * c / lambda0_cm;
  const double n_crit = kElectronMass * omega0 * omega0 /
                        (4.0 * kPi * kElementaryCharge * kElementaryCharge);
  const double lam_pref = lambda0_cm * kElementaryCharge * kElementaryCharge /
                          (c * c * c * kElectronMass);
  const double eps_n = laser.absorption.eps_n;
  const double cutoff = laser.cbet.ne_frac_cutoff;

  std::vector<double> h_Ti(static_cast<std::size_t>(n_cells));
  state.Ti.copy_to_host(h_Ti);
  std::vector<double> h_Te(static_cast<std::size_t>(n_cells));
  state.Te.copy_to_host(h_Te);
  std::vector<double> h_vr;
  state.v_r.copy_to_host(h_vr);
  std::vector<double> h_vol(static_cast<std::size_t>(n_cells));
  state.vol.copy_to_host(h_vol);
  TENRYU_ASSERT(static_cast<int>(h_vr.size()) >= n_cells + 1,
                "cbet_stage_cell_fields node velocity size mismatch");

  std::vector<double> h_pref(static_cast<std::size_t>(n_cells));
  std::vector<double> h_ca(static_cast<std::size_t>(n_cells));
  std::vector<double> h_ur(static_cast<std::size_t>(n_cells));
  std::vector<double> h_kbar(static_cast<std::size_t>(n_cells));
  std::vector<std::uint8_t> h_mask(static_cast<std::size_t>(n_cells));
  for (int i = 0; i < n_cells; ++i) {
    const std::size_t si = static_cast<std::size_t>(i);
    const double rho = mirror.rho[si];
    const double zbar = std::max(mirror.zbar[si], 0.0);
    const double A_eff = std::max(mirror.A_eff[si], 1.0e-30);
    const double Te_erg = std::max(h_Te[si], 0.0) * eV;
    const double Ti_erg = std::max(h_Ti[si], 0.0) * eV;
    const double u_r_cell = 0.5 * (h_vr[si] + h_vr[si + 1]);
    const double ne = rho * zbar / (A_eff * kProtonMass);
    const double nh_raw = ne / n_crit;
    const bool is_void = (!mirror.cell_is_void.empty() && mirror.cell_is_void[si] != 0);
    // NaN in any staged field silently poisons chi_pref/c_a/g for the cell
    // (std::max(NaN, 0.0) keeps the NaN; Ti and u_r were previously unchecked
    // — AI review M-12). Non-finite cells are masked out like rho<=0 cells.
    const bool finite_fields = std::isfinite(rho) && std::isfinite(zbar) &&
                               std::isfinite(A_eff) && std::isfinite(Te_erg) &&
                               std::isfinite(Ti_erg) && std::isfinite(u_r_cell) &&
                               std::isfinite(h_vol[si]);
    const bool active = finite_fields && !is_void && rho > 0.0 && zbar > 0.0 &&
                        Te_erg > 0.0 && nh_raw > 0.0 && nh_raw < cutoff &&
                        h_vol[si] > 0.0;
    h_mask[si] = active ? 1 : 0;
    if (active) {
      const double denom_T = zbar * Te_erg + 3.0 * Ti_erg;
      const double one_minus = std::max(1.0 - nh_raw, eps_n);
      h_pref[si] = lam_pref * (nh_raw / one_minus) * (zbar / std::max(denom_T, 1.0e-300));
      h_ca[si] = std::sqrt(std::max(denom_T, 0.0) / (A_eff * kProtonMass));
      h_kbar[si] = std::sqrt(std::max(1.0 - nh_raw, eps_n)) * omega0 / c;
      h_ur[si] = u_r_cell;
    } else {
      h_pref[si] = 0.0;
      h_ca[si] = 0.0;
      h_ur[si] = 0.0;
      h_kbar[si] = 0.0;
    }
  }
  cbet_check(cudaMemcpyAsync(ws.cell_chi_pref, h_pref.data(),
                             h_pref.size() * sizeof(double), cudaMemcpyHostToDevice,
                             stream),
             "cbet chi_pref H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_c_a, h_ca.data(), h_ca.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "cbet c_a H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_u_r, h_ur.data(), h_ur.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "cbet u_r H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_k_bar, h_kbar.data(),
                             h_kbar.size() * sizeof(double), cudaMemcpyHostToDevice,
                             stream),
             "cbet k_bar H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_vol, h_vol.data(), h_vol.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "cbet vol H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_mask, h_mask.data(),
                             h_mask.size() * sizeof(std::uint8_t),
                             cudaMemcpyHostToDevice, stream),
             "cbet mask H2D failed");
  cbet_check(cudaStreamSynchronize(stream), "cbet stage_cell_fields stream sync failed");
}

void cbet_stage_cell_fields_2d(CbetWorkspace& ws,
                               const CbetLmFields& lm,
                               const core::Config::LaserConfig& laser,
                               const double lambda0_cm,
                               cudaStream_t stream) {
  TENRYU_ASSERT(ws.dim2d, "cbet_stage_cell_fields_2d requires 2D workspace");
  const int n_nodes_r = static_cast<int>(lm.node_R.size());
  const int n_nodes_z = static_cast<int>(lm.node_Z.size());
  TENRYU_ASSERT(n_nodes_r >= 2 && n_nodes_z >= 2,
                "cbet_stage_cell_fields_2d requires at least 2x2 nodes");
  const int n_nodes = n_nodes_r * n_nodes_z;
  const int nz_cells = n_nodes_z - 1;
  const int n_cells = (n_nodes_r - 1) * nz_cells;
  TENRYU_ASSERT(n_cells == ws.n_cells, "cbet_stage_cell_fields_2d cell count mismatch");
  TENRYU_ASSERT(nz_cells == ws.nz_cells, "cbet_stage_cell_fields_2d nz_cells mismatch");
  TENRYU_ASSERT(ws.n_nodes == n_nodes, "cbet_stage_cell_fields_2d node count mismatch");
  TENRYU_ASSERT(ws.n_nodes_z == n_nodes_z, "cbet_stage_cell_fields_2d node stride mismatch");
  TENRYU_ASSERT(static_cast<int>(lm.n_hat_raw.size()) == n_nodes &&
                    static_cast<int>(lm.Te.size()) == n_nodes &&
                    static_cast<int>(lm.Zbar.size()) == n_nodes &&
                    static_cast<int>(lm.Ti.size()) == n_nodes &&
                    static_cast<int>(lm.u_R.size()) == n_nodes &&
                    static_cast<int>(lm.u_Z.size()) == n_nodes &&
                    static_cast<int>(lm.A_eff.size()) == n_nodes &&
                    static_cast<int>(lm.covered.size()) == n_nodes,
                "cbet_stage_cell_fields_2d node field size mismatch");

  const double c = core::constants::c_light;
  const double eV = core::constants::eV_to_erg;
  const double omega0 = 2.0 * kPi * c / lambda0_cm;
  const double lam_pref = lambda0_cm * kElementaryCharge * kElementaryCharge /
                          (c * c * c * kElectronMass);
  const double eps_n = laser.absorption.eps_n;
  const double cutoff = laser.cbet.ne_frac_cutoff;

  std::vector<double> h_pref(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> h_ca(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> h_ur(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> h_uz(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> h_kbar(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> h_vol(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<std::uint8_t> h_mask(static_cast<std::size_t>(n_cells), 0U);
  auto mean4 = [](const std::vector<double>& f,
                  const int n00,
                  const int n10,
                  const int n01,
                  const int n11) {
    return 0.25 * (f[static_cast<std::size_t>(n00)] +
                   f[static_cast<std::size_t>(n10)] +
                   f[static_cast<std::size_t>(n01)] +
                   f[static_cast<std::size_t>(n11)]);
  };
  for (int ci = 0; ci < n_nodes_r - 1; ++ci) {
    for (int cj = 0; cj < n_nodes_z - 1; ++cj) {
      const int cidx = ci * nz_cells + cj;
      const int n00 = ci * n_nodes_z + cj;
      const int n10 = (ci + 1) * n_nodes_z + cj;
      const int n01 = ci * n_nodes_z + (cj + 1);
      const int n11 = (ci + 1) * n_nodes_z + (cj + 1);
      const std::size_t sc = static_cast<std::size_t>(cidx);
      const bool covered_cell = lm.covered[static_cast<std::size_t>(n00)] != 0U &&
                                lm.covered[static_cast<std::size_t>(n10)] != 0U &&
                                lm.covered[static_cast<std::size_t>(n01)] != 0U &&
                                lm.covered[static_cast<std::size_t>(n11)] != 0U;
      const double nh_raw = mean4(lm.n_hat_raw, n00, n10, n01, n11);
      const double zbar = mean4(lm.Zbar, n00, n10, n01, n11);
      const double Te_erg = std::max(mean4(lm.Te, n00, n10, n01, n11), 0.0) * eV;
      const double Ti_erg = std::max(mean4(lm.Ti, n00, n10, n01, n11), 0.0) * eV;
      const double A_eff = mean4(lm.A_eff, n00, n10, n01, n11);
      const double u_R = mean4(lm.u_R, n00, n10, n01, n11);
      const double u_Z = mean4(lm.u_Z, n00, n10, n01, n11);
      const double R0 = lm.node_R[static_cast<std::size_t>(ci)];
      const double R1 = lm.node_R[static_cast<std::size_t>(ci + 1)];
      const double Z0 = lm.node_Z[static_cast<std::size_t>(cj)];
      const double Z1 = lm.node_Z[static_cast<std::size_t>(cj + 1)];
      const double V_c = kPi * (R1 * R1 - R0 * R0) * std::fabs(Z1 - Z0);
      // Mirror the 1D staging guard: non-finite node means (incl. Ti/u which
      // feed c_a and kdotu) must not enter the gain kernels (AI review M-12).
      const bool finite_fields = std::isfinite(nh_raw) && std::isfinite(zbar) &&
                                 std::isfinite(Te_erg) && std::isfinite(Ti_erg) &&
                                 std::isfinite(A_eff) && std::isfinite(u_R) &&
                                 std::isfinite(u_Z) && std::isfinite(V_c);
      const bool active = finite_fields && covered_cell && zbar > 0.0 &&
                          Te_erg > 0.0 && nh_raw > 0.0 && nh_raw < cutoff &&
                          V_c > 0.0 && A_eff > 0.0;
      if (!active) {
        continue;
      }
      const double denom_T = zbar * Te_erg + 3.0 * Ti_erg;
      const double one_minus = std::max(1.0 - nh_raw, eps_n);
      const double A_eff_safe = std::max(A_eff, 1.0e-30);
      h_pref[sc] = lam_pref * (nh_raw / one_minus) *
                   (zbar / std::max(denom_T, 1.0e-300));
      h_ca[sc] = std::sqrt(std::max(denom_T, 0.0) / (A_eff_safe * kProtonMass));
      h_ur[sc] = u_R;
      h_uz[sc] = u_Z;
      h_kbar[sc] = std::sqrt(std::max(1.0 - nh_raw, eps_n)) * omega0 / c;
      h_vol[sc] = V_c;
      h_mask[sc] = 1U;
    }
  }

  cbet_check(cudaMemcpyAsync(ws.cell_chi_pref, h_pref.data(),
                             h_pref.size() * sizeof(double), cudaMemcpyHostToDevice,
                             stream),
             "cbet 2D chi_pref H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_c_a, h_ca.data(), h_ca.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "cbet 2D c_a H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_u_r, h_ur.data(), h_ur.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "cbet 2D u_r H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_u_z, h_uz.data(), h_uz.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "cbet 2D u_z H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_k_bar, h_kbar.data(),
                             h_kbar.size() * sizeof(double), cudaMemcpyHostToDevice,
                             stream),
             "cbet 2D k_bar H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_vol, h_vol.data(), h_vol.size() * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "cbet 2D vol H2D failed");
  cbet_check(cudaMemcpyAsync(ws.cell_mask, h_mask.data(),
                             h_mask.size() * sizeof(std::uint8_t),
                             cudaMemcpyHostToDevice, stream),
             "cbet 2D mask H2D failed");
  cbet_check(cudaStreamSynchronize(stream), "cbet stage_cell_fields_2d stream sync failed");
}

CbetSolveResult cbet_solve_and_deposit(CbetWorkspace& ws,
                                       const core::Config::LaserConfig::CbetConfig& cfg,
                                       cudaStream_t stream,
                                       const CbetPortSectionArgs* ps,
                                       const bool collect_dq) {
  CbetSolveResult result;
  TENRYU_ASSERT((ps != nullptr) == ws.ps_mode,
                "cbet port_section arguments/workspace mode mismatch");
  if (ps != nullptr) {
    TENRYU_ASSERT(!ws.dim2d, "port_section is 1D-only");
    TENRYU_ASSERT(ps->n_ports == ws.n_ports,
                  "cbet port_section port count mismatch");
    TENRYU_ASSERT(ws.n_groups == ws.n_ports * ws.G_ref,
                  "cbet port_section expanded group count mismatch");
  }
  const int G = ws.n_groups;
  const int G_ref = (ps != nullptr) ? ws.G_ref : G;
  const int n_cells = ws.n_cells;
  const std::int64_t n_gc = static_cast<std::int64_t>(G) * n_cells;
  const std::int64_t n_gc_ref =
      static_cast<std::int64_t>(G_ref) * n_cells;
  const int n_rays = ws.n_rays_total;
  result.dq_G = G;
  result.dq_n_bins = ws.n_bins;
  result.dq_n_branches = ws.n_branches;
  if (ps != nullptr) {
    result.dq_G_ref = G_ref;
    result.dq_n_ports = ws.n_ports;
  }

  if (ps != nullptr) {
    const std::size_t n_chi =
        static_cast<std::size_t>(n_cells) * ws.n_pairs;
    const std::size_t n_pair_index =
        static_cast<std::size_t>(G) * G;
    if (ps->chi_device != nullptr) {
      cbet_check(cudaMemcpyAsync(ws.chi, ps->chi_device,
                                 n_chi * sizeof(double),
                                 cudaMemcpyDeviceToDevice, stream),
                 "cbet port_section chi D2D failed");
    } else {
      cbet_check(cudaMemcpyAsync(ws.chi, ps->chi_host,
                                 n_chi * sizeof(double),
                                 cudaMemcpyHostToDevice, stream),
                 "cbet port_section chi H2D failed");
    }
    cbet_check(cudaMemcpyAsync(ws.omega_group, ps->omega_state_host,
                               static_cast<std::size_t>(G) * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "cbet port_section omega H2D failed");
    cbet_check(cudaMemcpyAsync(ws.port_weight, ps->port_weight_host,
                               static_cast<std::size_t>(ws.n_ports) *
                                   sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "cbet port_section weight H2D failed");
    cbet_check(cudaMemcpyAsync(ws.pair_p, ps->pair_p,
                               static_cast<std::size_t>(ws.n_pairs) *
                                   sizeof(std::int16_t),
                               cudaMemcpyHostToDevice, stream),
               "cbet port_section pair_p H2D failed");
    cbet_check(cudaMemcpyAsync(ws.pair_q, ps->pair_q,
                               static_cast<std::size_t>(ws.n_pairs) *
                                   sizeof(std::int16_t),
                               cudaMemcpyHostToDevice, stream),
               "cbet port_section pair_q H2D failed");
    cbet_check(cudaMemcpyAsync(ws.pair_index, ps->pair_index,
                               n_pair_index * sizeof(std::int32_t),
                               cudaMemcpyHostToDevice, stream),
               "cbet port_section pair_index H2D failed");
    cbet_check(cudaStreamSynchronize(stream),
               "cbet port_section table stage sync failed");
  } else {
    // omega per group from beam bookkeeping
    std::vector<double> h_omega_g(static_cast<std::size_t>(G), 0.0);
    for (int g = 0; g < G; ++g) {
      const int b = g / (ws.n_branches * ws.n_bins);
      h_omega_g[static_cast<std::size_t>(g)] =
          ws.beam_omega[static_cast<std::size_t>(b)];
    }
    cbet_check(cudaMemcpyAsync(ws.omega_group, h_omega_g.data(),
                               h_omega_g.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "cbet omega_group H2D failed");
    cbet_check(cudaStreamSynchronize(stream), "cbet omega stage sync failed");
  }

  // exclusive scan of rec_count -> ray_rec_offset, live record count
  {
    thrust::device_ptr<const std::int32_t> cnt(ws.rec_count);
    thrust::device_ptr<std::int64_t> off(ws.ray_rec_offset);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), cnt, cnt + n_rays, off,
                           static_cast<std::int64_t>(0));
    std::int64_t last_off = 0;
    std::int32_t last_cnt = 0;
    cbet_check(cudaMemcpyAsync(&last_off, ws.ray_rec_offset + (n_rays - 1),
                               sizeof(std::int64_t), cudaMemcpyDeviceToHost, stream),
               "cbet scan tail D2H failed");
    cbet_check(cudaMemcpyAsync(&last_cnt, ws.rec_count + (n_rays - 1),
                               sizeof(std::int32_t), cudaMemcpyDeviceToHost, stream),
               "cbet count tail D2H failed");
    cbet_check(cudaStreamSynchronize(stream), "cbet scan sync failed");
    ws.n_records_live = last_off + last_cnt;
    const std::int64_t total = ws.n_records_live;
    cbet_check(cudaMemcpyAsync(ws.ray_rec_offset + n_rays, &total,
                               sizeof(std::int64_t), cudaMemcpyHostToDevice, stream),
               "cbet scan total H2D failed");
  }

  if (ws.n_records_live > 0) {
    const int block = 128;
    const int grid_rays = (n_rays + block - 1) / block;
    if (ws.dim2d) {
      cbet_build_keys_2d_kernel<<<grid_rays, block, 0, stream>>>(
          ws.rec_cell, ws.rec_mu, ws.rec_c, ws.rec_count, ws.ray_group_base,
          ws.ray_rec_offset, ws.sort_key, ws.sort_slot, ws.cap_per_ray, ws.n_bins, G,
          n_rays);
    } else {
      cbet_build_keys_kernel<<<grid_rays, block, 0, stream>>>(
          ws.rec_cell, ws.rec_mu, ws.rec_count, ws.ray_group_base, ws.ray_rec_offset,
          ws.sort_key, ws.sort_slot, ws.cap_per_ray, ws.n_bins, G_ref, n_rays);
    }
    cbet_check(cudaGetLastError(), "cbet build_keys launch failed");
    thrust::device_ptr<std::int64_t> keys(ws.sort_key);
    thrust::device_ptr<std::int32_t> slots(ws.sort_slot);
    thrust::sort_by_key(thrust::cuda::par.on(stream), keys,
                        keys + ws.n_records_live, slots);
    const std::int64_t grid_seg =
        (n_gc_ref + 1 + block - 1) / block;
    cbet_seg_offsets_kernel<<<static_cast<unsigned int>(grid_seg), block, 0, stream>>>(
        ws.sort_key, ws.n_records_live, ws.seg_offsets, n_gc_ref);
    cbet_check(cudaGetLastError(), "cbet seg_offsets launch failed");
  } else {
    cbet_check(cudaMemsetAsync(ws.seg_offsets, 0,
                               (static_cast<std::size_t>(n_gc_ref) + 1) *
                                   sizeof(std::int64_t),
                               stream),
               "cbet seg_offsets memset failed");
  }

  cbet_check(cudaMemsetAsync(ws.d_counters, 0, 4 * sizeof(unsigned long long), stream),
             "cbet counters memset failed");
  cbet_check(cudaMemsetAsync(ws.L_prev, 0,
                             static_cast<std::size_t>(n_gc) * sizeof(double), stream),
             "cbet L_prev memset failed");

  const int block = 128;
  const std::int64_t grid_gc = (n_gc + block - 1) / block;
  const std::int64_t grid_gc_ref =
      (n_gc_ref + block - 1) / block;
  const std::int64_t n_cp = static_cast<std::int64_t>(n_cells) * ws.n_pairs;
  const std::int64_t grid_cp = (n_cp + block - 1) / block;
  const int grid_rays = (n_rays + block - 1) / block;
  const std::int64_t n_records_capacity =
      static_cast<std::int64_t>(n_rays) * ws.cap_per_ray;
  if (ps != nullptr) {
    const std::int64_t n_rec_ps =
        static_cast<std::int64_t>(ws.n_ports) * n_records_capacity;
    const std::int64_t grid_rec_ps =
        (n_rec_ps + block - 1) / block;
    cbet_init_rec_w_ps_kernel<<<static_cast<unsigned int>(grid_rec_ps), block,
                               0, stream>>>(
        ws.rec_w, ws.port_weight, ws.rec_w_ps, n_records_capacity,
        ws.n_ports);
    cbet_check(cudaGetLastError(),
               "cbet port_section rec_w init launch failed");
  }

  double h_stats[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  int m = 0;
  bool converged = false;
  const int max_iters = std::max(cfg.max_iters, 1);
  for (m = 1; m <= max_iters; ++m) {
    if (m > 1) {
      cbet_check(cudaMemcpyAsync(ws.L_prev, ws.L,
                                 static_cast<std::size_t>(n_gc) * sizeof(double),
                                 cudaMemcpyDeviceToDevice, stream),
                 "cbet L_prev copy failed");
    }
    if (ps != nullptr) {
      for (int port = 0; port < ws.n_ports; ++port) {
        cbet_tally_ps_kernel<<<static_cast<unsigned int>(grid_gc_ref), block,
                               0, stream>>>(
            ws.seg_offsets, ws.sort_slot, ws.rec_w_ps, ws.rec_ds, ws.rec_mu,
            ws.L, ws.Mmu, ws.ds_max_gc, n_gc_ref, n_records_capacity, port,
            G_ref, G);
        cbet_check(cudaGetLastError(),
                   "cbet port_section tally launch failed");
      }
      cbet_losscap_kernel<<<static_cast<unsigned int>(grid_gc), block, 0,
                            stream>>>(
          ws.L, ws.ds_max_gc, ws.chi, ws.pair_index, ws.cell_vol,
          ws.cell_mask, ws.fcap, cfg.theta_cap, G, ws.n_pairs, n_cells);
      cbet_check(cudaGetLastError(), "cbet losscap launch failed");
    } else if (ws.dim2d) {
      cbet_tally_2d_kernel<<<static_cast<unsigned int>(grid_gc), block, 0, stream>>>(
          ws.seg_offsets, ws.sort_slot, ws.rec_w, ws.rec_ds, ws.rec_mu, ws.rec_c, ws.L,
          ws.Mmu, ws.Mc, ws.ds_max_gc, n_gc);
      cbet_check(cudaGetLastError(), "cbet tally launch failed");
      cbet_losscap_2d_kernel<<<static_cast<unsigned int>(grid_gc), block, 0, stream>>>(
          ws.L, ws.Mmu, ws.Mc, ws.ds_max_gc, ws.omega_group, ws.cell_chi_pref,
          ws.cell_c_a, ws.cell_u_r, ws.cell_u_z, ws.cell_k_bar, ws.cell_vol,
          ws.cell_mask, ws.fcap, cfg.theta_cap, cfg.f_cbet, cfg.alpha_iaw,
          cfg.k_a_floor, cfg.test_chi, G, n_cells);
      cbet_check(cudaGetLastError(), "cbet losscap launch failed");
    } else {
      cbet_tally_kernel<<<static_cast<unsigned int>(grid_gc), block, 0, stream>>>(
          ws.seg_offsets, ws.sort_slot, ws.rec_w, ws.rec_ds, ws.rec_mu, ws.L, ws.Mmu,
          ws.ds_max_gc, n_gc);
      cbet_check(cudaGetLastError(), "cbet tally launch failed");
      cbet_chi_kernel<<<static_cast<unsigned int>(grid_cp), block, 0, stream>>>(
          ws.L, ws.Mmu, ws.pair_p, ws.pair_q, ws.omega_group, ws.cell_chi_pref,
          ws.cell_c_a, ws.cell_u_r, ws.cell_k_bar, ws.cell_vol, ws.cell_mask, ws.chi,
          cfg.f_cbet, cfg.alpha_iaw, cfg.n_phi, cfg.k_a_floor, cfg.test_chi, G,
          ws.n_pairs, n_cells);
      cbet_check(cudaGetLastError(), "cbet chi launch failed");
      cbet_losscap_kernel<<<static_cast<unsigned int>(grid_gc), block, 0, stream>>>(
          ws.L, ws.ds_max_gc, ws.chi, ws.pair_index, ws.cell_vol, ws.cell_mask, ws.fcap,
          cfg.theta_cap, G, ws.n_pairs, n_cells);
      cbet_check(cudaGetLastError(), "cbet losscap launch failed");
    }
    cbet_check(cudaMemsetAsync(ws.d_counters + 1, 0, sizeof(unsigned long long), stream),
               "cbet capped counter memset failed");
    if (ps != nullptr) {
      cbet_check(cudaMemsetAsync(
                     ws.iaw_partial, 0,
                     static_cast<std::size_t>(n_gc) * sizeof(double), stream),
                 "cbet port_section iaw_partial memset failed");
    }
    if (ws.dim2d) {
      cbet_dq_2d_kernel<<<static_cast<unsigned int>(grid_gc), block, 0, stream>>>(
          ws.L, ws.Mmu, ws.Mc, ws.fcap, ws.omega_group, ws.cell_chi_pref, ws.cell_c_a,
          ws.cell_u_r, ws.cell_u_z, ws.cell_k_bar, ws.cell_vol, ws.cell_mask, ws.dQ,
          ws.exch_partial, ws.d_counters + 1, cfg.f_cbet, cfg.alpha_iaw,
          cfg.k_a_floor, cfg.test_chi, G, n_cells);
    } else {
      cbet_dq_kernel<<<static_cast<unsigned int>(grid_gc), block, 0, stream>>>(
          ws.L, ws.chi, ws.fcap, ws.pair_index, ws.cell_vol, ws.cell_mask,
          ws.omega_group, ws.dQ, ws.iaw_partial, ws.exch_partial,
          ws.d_counters + 1, ps != nullptr ? 1 : 0, G, ws.n_pairs, n_cells);
    }
    cbet_check(cudaGetLastError(), "cbet dQ launch failed");
    cbet_iter_stats_kernel<<<1, 256, 0, stream>>>(ws.L, ws.L_prev, ws.dQ,
                                                  ws.exch_partial, ws.d_scalars, n_gc);
    cbet_check(cudaGetLastError(), "cbet stats launch failed");
    cbet_check(cudaMemcpyAsync(h_stats, ws.d_scalars, 5 * sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "cbet stats D2H failed");
    cbet_check(cudaStreamSynchronize(stream), "cbet stats sync failed");
    if (m > 1) {
      const double denom = std::max(h_stats[1], 1.0e-300);
      result.conv_final = h_stats[0] / denom;
      if (result.conv_final < cfg.tol) {
        converged = true;
        break;
      }
    }
    if (m == max_iters) {
      // Do not overwrite rec_w on the last iteration: the final depositing pass
      // must apply dQ (computed from the tally of the CURRENT rec_w) with the
      // SAME rec_w in the share numerators, otherwise the share sums drift off
      // one and the applied exchange silently violates conservation.
      break;
    }
    if (ps != nullptr) {
      for (int port = 0; port < ws.n_ports; ++port) {
        cbet_propagate_ps_kernel<<<grid_rays, block, 0, stream>>>(
            ws.rec_cell, ws.rec_mu, ws.rec_ds, ws.rec_S, ws.rec_w_ps,
            ws.rec_count, ws.ray_group_base, ws.ray_P0, ws.ray_overflow,
            ws.dQ, ws.L, ws.cell_mask, ws.port_weight, ws.dep_rows,
            ws.unabs_rows, ws.d_counters, ws.clamp_rows, ws.ps_cell_nhat,
            ws.ps_capture_thresh, ws.ps_capture_order,
            ws.ps_one_minus_eta,
            ws.ps_capture_stage, ps->traj_rec_ratio, ws.ps_n_channels,
            ws.cap_per_ray, ws.n_bins, G_ref, G, n_cells, n_rays, port,
            ps->traj_ray_offset, ps->traj_n_output_rays,
            ps->traj_output_stride, ps->viz_port, 0);
        cbet_check(cudaGetLastError(),
                   "cbet port_section propagate launch failed");
      }
    } else if (ws.dim2d) {
      cbet_propagate_2d_kernel<<<grid_rays, block, 0, stream>>>(
          ws.rec_cell, ws.rec_mu, ws.rec_c, ws.rec_w00, ws.rec_w10, ws.rec_w01,
          ws.rec_ds, ws.rec_S, ws.rec_w, ws.rec_count, ws.ray_group_base, ws.ray_P0,
          ws.ray_overflow, ws.dQ, ws.L, ws.cell_mask, ws.dep_nodes, ws.unabs_rows,
          ws.d_counters, ws.clamp_rows, ws.cap_per_ray, ws.n_bins, ws.n_branches,
          G, n_cells, ws.nz_cells, ws.n_nodes_z, ws.n_nodes, n_rays, 0);
    } else {
      cbet_propagate_kernel<<<grid_rays, block, 0, stream>>>(
          ws.rec_cell, ws.rec_mu, ws.rec_ds, ws.rec_S, ws.rec_w, ws.rec_count,
          ws.ray_group_base, ws.ray_P0, ws.ray_overflow, ws.dQ, ws.L, ws.cell_mask,
          ws.dep_rows, ws.unabs_rows, ws.d_counters, ws.clamp_rows, ws.cap_per_ray,
          ws.n_bins, G, n_cells, n_rays, 0);
    }
    if (ps == nullptr) {
      cbet_check(cudaGetLastError(), "cbet propagate launch failed");
    }
  }
  result.iterations = std::min(m, max_iters);
  result.converged = converged;
  if (!converged) {
    core::log_warning("cbet_solve_and_deposit did not converge within max_iters; "
                      "using last iterate (conservative by construction)");
  }
  if (ps != nullptr) {
    const int grid_cells = (n_cells + block - 1) / block;
    cbet_iaw_cell_reduce_kernel<<<grid_cells, block, 0, stream>>>(
        ws.iaw_partial, ws.iaw_cell, G, n_cells);
    cbet_check(cudaGetLastError(),
               "cbet port_section iaw cell reduce launch failed");
    cbet_check(cudaMemsetAsync(ws.d_iaw_sum, 0, sizeof(double), stream),
               "cbet port_section iaw sum memset failed");
    cbet_sum_rows_kernel<<<1, 256, 0, stream>>>(
        ws.iaw_cell, 0, n_cells, ws.d_iaw_sum);
    cbet_check(cudaGetLastError(),
               "cbet port_section iaw total reduce launch failed");
  }
  if (collect_dq) {
    result.dq_host.resize(static_cast<std::size_t>(n_gc));
    cbet_check(cudaMemcpyAsync(result.dq_host.data(), ws.dQ,
                               result.dq_host.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "cbet final-iteration dQ D2H failed");
    cbet_check(cudaStreamSynchronize(stream),
               "cbet final-iteration dQ sync failed");
  }

  // final depositing pass
  if (ws.dim2d) {
    cbet_check(cudaMemsetAsync(ws.dep_nodes, 0,
                               static_cast<std::size_t>(ws.n_beams) *
                                   static_cast<std::size_t>(ws.n_nodes) * sizeof(double),
                               stream),
               "cbet dep_nodes memset failed");
  } else {
    cbet_check(cudaMemsetAsync(ws.dep_rows, 0,
                               static_cast<std::size_t>(n_rays) *
                                   static_cast<std::size_t>(n_cells) * sizeof(double),
                               stream),
               "cbet dep_rows memset failed");
  }
  cbet_check(cudaMemsetAsync(ws.unabs_rows, 0,
                             static_cast<std::size_t>(n_rays) * sizeof(double), stream),
             "cbet unabs memset failed");
  cbet_check(cudaMemsetAsync(ws.clamp_rows, 0,
                             static_cast<std::size_t>(n_rays) * sizeof(double), stream),
             "cbet clamp_rows memset failed");
  cbet_check(cudaMemsetAsync(ws.d_scalars + 5, 0, sizeof(double), stream),
             "cbet clamped_power memset failed");
  if (ps != nullptr) {
    if (ws.ps_n_channels > 0) {
      const std::size_t capture_doubles =
          static_cast<std::size_t>(ws.n_ports) *
          static_cast<std::size_t>(n_rays) *
          static_cast<std::size_t>(ws.ps_n_channels) * 4U;
      cbet_check(cudaMemsetAsync(
                     ws.ps_capture_stage, 0,
                     capture_doubles * sizeof(double), stream),
                 "cbet port_section capture stage memset failed");
    }
    for (int port = 0; port < ws.n_ports; ++port) {
      cbet_propagate_ps_kernel<<<grid_rays, block, 0, stream>>>(
          ws.rec_cell, ws.rec_mu, ws.rec_ds, ws.rec_S, ws.rec_w_ps,
          ws.rec_count, ws.ray_group_base, ws.ray_P0, ws.ray_overflow,
          ws.dQ, ws.L, ws.cell_mask, ws.port_weight, ws.dep_rows,
          ws.unabs_rows, ws.d_counters, ws.clamp_rows, ws.ps_cell_nhat,
          ws.ps_capture_thresh, ws.ps_capture_order,
          ws.ps_one_minus_eta,
          ws.ps_capture_stage, ps->traj_rec_ratio, ws.ps_n_channels,
          ws.cap_per_ray, ws.n_bins, G_ref, G, n_cells, n_rays, port,
          ps->traj_ray_offset, ps->traj_n_output_rays,
          ps->traj_output_stride, ps->viz_port, 1);
      cbet_check(cudaGetLastError(),
                 "cbet port_section final propagate launch failed");
    }
  } else if (ws.dim2d) {
    cbet_propagate_2d_kernel<<<grid_rays, block, 0, stream>>>(
        ws.rec_cell, ws.rec_mu, ws.rec_c, ws.rec_w00, ws.rec_w10, ws.rec_w01,
        ws.rec_ds, ws.rec_S, ws.rec_w, ws.rec_count, ws.ray_group_base, ws.ray_P0,
        ws.ray_overflow, ws.dQ, ws.L, ws.cell_mask, ws.dep_nodes, ws.unabs_rows,
        ws.d_counters, ws.clamp_rows, ws.cap_per_ray, ws.n_bins, ws.n_branches, G,
        n_cells, ws.nz_cells, ws.n_nodes_z, ws.n_nodes, n_rays, 1);
  } else {
    cbet_propagate_kernel<<<grid_rays, block, 0, stream>>>(
        ws.rec_cell, ws.rec_mu, ws.rec_ds, ws.rec_S, ws.rec_w, ws.rec_count,
        ws.ray_group_base, ws.ray_P0, ws.ray_overflow, ws.dQ, ws.L, ws.cell_mask,
        ws.dep_rows, ws.unabs_rows, ws.d_counters, ws.clamp_rows, ws.cap_per_ray,
        ws.n_bins, G, n_cells, n_rays, 1);
  }
  if (ps == nullptr) {
    cbet_check(cudaGetLastError(), "cbet final propagate launch failed");
  }
  if (ps != nullptr && ps->traj_power != nullptr &&
      ps->traj_rec_idx != nullptr && ps->traj_step_count != nullptr &&
      ps->traj_rec_ratio != nullptr && ps->traj_n_output_rays > 0 &&
      ps->traj_output_stride > 0 && ps->traj_max_steps > 0) {
    const std::int64_t traj_capacity =
        static_cast<std::int64_t>(ps->traj_n_output_rays) *
        ps->traj_max_steps;
    const int traj_block = 256;
    const int traj_grid =
        static_cast<int>((traj_capacity + traj_block - 1) / traj_block);
    cbet_scale_traj_power_kernel<<<traj_grid, traj_block, 0, stream>>>(
        ps->traj_power, ps->traj_rec_idx, ps->traj_step_count,
        ps->traj_rec_ratio, ws.rec_count, ps->traj_ray_offset,
        ps->traj_n_output_rays, ps->traj_output_stride,
        ps->traj_max_steps, ws.cap_per_ray, n_rays);
    cbet_check(cudaGetLastError(),
               "cbet port_section trajectory scale launch failed");
  }
  // Deterministic fixed-order fold of the per-ray clamp tallies (S-03: keeps
  // the clamped_power diagnostic bitwise stable when multiple clamps fire).
  cbet_sum_rows_kernel<<<1, 256, 0, stream>>>(ws.clamp_rows, 0, n_rays,
                                              ws.d_scalars + 5);
  cbet_check(cudaGetLastError(), "cbet clamp reduce launch failed");

  unsigned long long h_counters[4] = {0ULL, 0ULL, 0ULL, 0ULL};
  double h_iaw_rate = 0.0;
  cbet_check(cudaMemcpyAsync(h_counters, ws.d_counters, 4 * sizeof(unsigned long long),
                             cudaMemcpyDeviceToHost, stream),
             "cbet counters D2H failed");
  if (ps != nullptr) {
    cbet_check(cudaMemcpyAsync(&h_iaw_rate, ws.d_iaw_sum, sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "cbet iaw rate D2H failed");
  }
  cbet_check(cudaStreamSynchronize(stream), "cbet final sync failed");
  result.clamp_count = h_counters[0];
  result.capped_pairs = h_counters[1];
  // rec_w [erg/s] and ds [cm] give L [erg cm/s]; chi [cm s/erg]
  // times L_l*L_g divided by cell volume [cm^3] gives each dQ and IAW
  // cell ledger directly in erg/s. The deterministic cell total therefore
  // needs no volume factor before conversion to step energy in laser.cu.
  result.E_iaw_rate = h_iaw_rate;
  result.exchanged_power = h_stats[4];
  result.ledger_residual_rel =
      (ps != nullptr)
          ? std::fabs(h_stats[2] + h_iaw_rate) /
                std::max(h_stats[3] + std::fabs(h_iaw_rate), 1.0e-300)
          : std::fabs(h_stats[2]) / std::max(h_stats[3], 1.0e-300);
  {
    thrust::device_ptr<const std::uint8_t> ov(ws.ray_overflow);
    result.overflow_rays = thrust::reduce(thrust::cuda::par.on(stream), ov, ov + n_rays,
                                          static_cast<long long>(0),
                                          thrust::plus<long long>());
  }
  // Record-capacity overflow (AI review C-02/S-02, requantified
  // 2026-07-31): an overflowed ray keeps its prefix records (their IB
  // deposit replays normally), its truncated tail is classified
  // unabsorbed by the replay's final power (energy conserving, deposit
  // location approximated), and the ray is excluded from the CBET
  // exchange entirely (cbet_active_ray gating). With the theta-limiter
  // march, physically ducted/whispering-gallery rays can legitimately
  // exceed any n_cells-scaled capacity (measured: trapped orbits with
  // thousands of cell transitions on the GXII shelf at t~2.5e-10), so
  // a small overflow count is a documented approximation, not
  // corruption. A LARGE fraction still indicates a sizing bug and
  // aborts.
  if (result.overflow_rays > 0) {
    const double overflow_frac =
        static_cast<double>(result.overflow_rays) /
        static_cast<double>(std::max(n_rays, 1));
    core::log_warning(
        "cbet: " + std::to_string(result.overflow_rays) +
        " ray(s) exceeded the per-ray record capacity (frac " +
        std::to_string(overflow_frac) +
        ") — prefix deposits retained, tails classified unabsorbed, rays "
        "excluded from CBET exchange (ducted/trapped orbits); raise "
        "Laser.cbet.max_segments_per_ray to audit");
    TENRYU_ASSERT(overflow_frac <= 0.01,
                  std::string("cbet: ") +
                      std::to_string(result.overflow_rays) +
                      " ray(s) (>1%) overflowed the per-ray record "
                      "capacity — capacity sizing bug; raise "
                      "Laser.cbet.max_segments_per_ray (0 = auto)");
  }
  {
    double h_clamped = 0.0;
    cbet_check(cudaMemcpyAsync(&h_clamped, ws.d_scalars + 5, sizeof(double),
                               cudaMemcpyDeviceToHost, stream),
               "cbet clamped_power D2H failed");
    cbet_check(cudaStreamSynchronize(stream), "cbet clamped_power sync failed");
    result.clamped_power = h_clamped;
  }
  return result;
}

void cbet_sum_rows_add(const double* d_rows,
                       const int first,
                       const int count,
                       double* d_out,
                       cudaStream_t stream) {
  if (count <= 0) {
    return;
  }
  cbet_sum_rows_kernel<<<1, 256, 0, stream>>>(d_rows, first, count, d_out);
  cbet_check(cudaGetLastError(), "cbet sum_rows launch failed");
}

}  // namespace tenryu::laser
