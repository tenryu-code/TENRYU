#include "laser/fast_trace_1d.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "laser/ib_absorption.cuh"
#include "laser/laser_phys_ext.cuh"
#include "laser/ray_trace_bodies.cuh"

namespace tenryu::laser {
namespace {

constexpr int kBlockSize = 128;
constexpr int kReductionBlockSize = 256;
constexpr std::size_t kPerRayShellBytesCap = 512ULL * 1024ULL * 1024ULL;
constexpr int kRayBatchSize = 8192;
constexpr int kReducedScalarCount = 4;
constexpr int kUnabsorbedScalar = 0;
constexpr int kRaScalar = 1;
constexpr int kCriticalHitsScalar = 2;
constexpr int kInvalidScalar = 3;
constexpr int kMaxShellOverlaps = 8;

// Bouguer-invariant reduction for the CC perf wave-3 S1 design (2026-08-04).
// In a shell with constant refractive index n, a = b / n is constant and
// ds = sqrt(r2^2 - a^2) - sqrt(r1^2 - a^2) is the exact leg length.
__global__ void fast_trace_1d_kernel(
    const double* __restrict__ radial_node_r,
    const double* __restrict__ radial_n_hat,
    const double* __restrict__ radial_n_hat_raw,
    const double* __restrict__ radial_smooth_kappa,
    const double* __restrict__ radial_dn_dr,
    const double* __restrict__ radial_T_e,
    const double* __restrict__ ray_R0,
    const double* __restrict__ ray_Z0,
    const double* __restrict__ ray_vR0,
    const double* __restrict__ ray_vZ0,
    const double* __restrict__ ray_power,
    const double* __restrict__ ray_power0,
    const int ray_offset,
    const int batch_rays,
    const int batch_capacity,
    const int n_shells,
    const double eps_n,
    const double eps_crit,
    const double test_kappa_cm_inv,
    const double lambda_cm,
    const LaserPhysExtOptions phys_opt,
    const int phys_ext_active,
    const double beam_P_w,
    const double beam_w_cm,
    double* __restrict__ per_ray_shell,
    double* __restrict__ per_ray_unabsorbed,
    double* __restrict__ per_ray_critical_hits,
    double* __restrict__ per_ray_invalid,
    double* __restrict__ per_ray_ra,
    double* __restrict__ tau_shell_out,
    double* __restrict__ tail_stats) {
  const int local_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_ray >= batch_rays) {
    return;
  }

  const int ray = ray_offset + local_ray;
  const double R0 = ray_R0[ray];
  const double Z0 = ray_Z0[ray];
  const double vR0 = ray_vR0[ray];
  const double vZ0 = ray_vZ0[ray];
  double P = ray_power[ray];
  const double P0 = ray_power0[ray];
  if (!(::isfinite(R0) && ::isfinite(Z0) && ::isfinite(vR0) &&
        ::isfinite(vZ0) && ::isfinite(P) && ::isfinite(P0))) {
    if (::isfinite(P) && P > 0.0) {
      per_ray_unabsorbed[local_ray] = P;
    }
    per_ray_invalid[local_ray] = 1.0;
    return;
  }
  if (!(P > 0.0)) {
    return;
  }

  const double v_norm = ::hypot(vR0, vZ0);
  if (!(::isfinite(v_norm) && v_norm > 0.0)) {
    per_ray_unabsorbed[local_ray] = P;
    per_ray_invalid[local_ray] = 1.0;
    return;
  }
  const double b = ::abs(R0 * vZ0 - Z0 * vR0) / v_norm;
  const double I_vac =
      (phys_ext_active != 0 && phys_opt.langdon_model != 0)
          ? vacuum_map_intensity(beam_P_w, beam_w_cm, ::abs(R0), 1)
          : 0.0;

  int turning_shell = -1;
  int last_subcritical_shell = -1;
  bool terminated_at_critical = false;
  for (int j = n_shells - 1; j >= 0; --j) {
    const double r_lower = radial_node_r[j];
    const double r_upper = radial_node_r[j + 1];
    const double n_hat = radial_n_hat[j];
    const double n_hat_raw = radial_n_hat_raw[j];
    const double dn_dr = radial_dn_dr[j];
    if (!(::isfinite(r_lower) && ::isfinite(r_upper) && r_upper >= r_lower &&
          ::isfinite(n_hat) && ::isfinite(n_hat_raw) && ::isfinite(dn_dr))) {
      per_ray_unabsorbed[local_ray] = P;
      per_ray_invalid[local_ray] = 1.0;
      return;
    }

    if (n_hat >= 1.0 - eps_crit || !(1.0 - n_hat > 0.0)) {
      const int target_shell =
          (last_subcritical_shell >= 0) ? last_subcritical_shell : j;
      if (last_subcritical_shell >= 0) {
        per_ray_critical_hits[local_ray] = 1.0;
      }
      if (phys_ext_active != 0 && phys_opt.crit_terminate_deposit != 0 &&
          target_shell >= 0) {
        per_ray_shell[static_cast<std::size_t>(target_shell) * batch_capacity +
                      local_ray] += P;
        P = 0.0;
      } else {
        per_ray_unabsorbed[local_ray] = P;
      }
      terminated_at_critical = true;
      break;
    }
    last_subcritical_shell = j;

    double kappa = test_kappa_cm_inv;
    if (!(kappa > 0.0)) {
      kappa = compute_kappa_from_smooth(radial_smooth_kappa[j], n_hat, eps_n);
      if (phys_ext_active != 0 && phys_opt.langdon_model != 0 && kappa > 0.0 &&
          radial_T_e != nullptr) {
        kappa *= compute_langdon_factor(
            phys_opt.langdon_model, phys_opt.langdon_zcoll, I_vac, lambda_cm,
            radial_T_e[j], phys_opt.langdon_te_min_eV);
      }
    }
    if (!(::isfinite(kappa) && kappa >= 0.0)) {
      per_ray_unabsorbed[local_ray] = P;
      per_ray_invalid[local_ray] = 1.0;
      return;
    }

    if (n_hat_raw >= ray_trace_bodies::kCritLayerHandoffNhatRaw &&
        1.0 - n_hat > 0.0) {
      const double v_mag2 = 1.0 - n_hat;
      const double v_mag = ::sqrt(v_mag2);
      const double sin_theta2 =
          ::fmin(1.0, b * b / ::fmax(v_mag2 * r_upper * r_upper, 1.0e-300));
      const double cos_theta = ::sqrt(::fmax(0.0, 1.0 - sin_theta2));
      const double v_radial = -cos_theta * v_mag;
      const double g_mag = ::fabs(dn_dr);
      const double v_dot_g = dn_dr * v_radial;
      const ray_trace_bodies::RadialInterval closure_interval{j, 0.0};
      double P_tail = P;
      ray_trace_bodies::TailClosureStatus tail_status;
      {
        // Expose this ray's shell slot as a one-cell deposit field so the
        // march closure can keep its native DepositCellCacheGuard path.
        double* const tail_deposit =
            per_ray_shell + static_cast<std::size_t>(j) * batch_capacity +
            local_ray;
        ray_trace_bodies::DepositCellCacheGuard tail_deposit_cache(tail_deposit);
        int tail_hint = 0;
        tail_status = ray_trace_bodies::try_tail_closure_1d(
            tail_deposit_cache, closure_interval, radial_smooth_kappa,
            radial_n_hat_raw, radial_node_r + j, 1, &tail_hint, -1, -1, -1.0,
            test_kappa_cm_inv, n_hat, n_hat_raw, kappa, g_mag, v_dot_g,
            v_mag2, r_upper, P,
            ray_trace_bodies::TailClosureMode::kRequireTrigger, nullptr,
            nullptr, nullptr, nullptr, P_tail);
      }
      if (tail_status == ray_trace_bodies::TailClosureStatus::kInvalid) {
        per_ray_unabsorbed[local_ray] = P;
        per_ray_invalid[local_ray] = 1.0;
        return;
      }
      if (tail_status == ray_trace_bodies::TailClosureStatus::kClosed) {
        constexpr int kMaxTailClosureShells = 32;
        const double dP = P - P_tail;
        const double L = ::fmax(0.0, (1.0 - n_hat_raw) / g_mag);
        int tail_shells[kMaxTailClosureShells];
        double tail_fractions[kMaxTailClosureShells];
        double tail_weights[kMaxTailClosureShells];
        int tail_shell_count = 0;
        bool redistribute_tail = L > 0.0 && ::isfinite(L);
        if (redistribute_tail) {
          const double r_hi = r_upper;
          const double r_lo_unclamped = r_upper - L;
          const double r_lo_span =
              ::fmax(radial_node_r[0], r_lo_unclamped);
          const double represented_length = r_hi - r_lo_span;
          const double fraction_length =
              (r_lo_span > r_lo_unclamped) ? represented_length : L;
          redistribute_tail = fraction_length > 0.0;
          for (int k = j;
               redistribute_tail && k >= 0 &&
               tail_shell_count < kMaxTailClosureShells;
               --k) {
            const double overlap =
                ::fmax(0.0, ::fmin(radial_node_r[k + 1], r_hi) -
                                    ::fmax(radial_node_r[k], r_lo_span));
            if (overlap > 0.0) {
              tail_shells[tail_shell_count] = k;
              tail_fractions[tail_shell_count] = overlap / fraction_length;
              ++tail_shell_count;
            }
            if (radial_node_r[k] <= r_lo_span) {
              break;
            }
          }

          const double tau_tail =
              -::log(::fmax(P_tail / ::fmax(P, 1.0e-300), 1.0e-300));
          redistribute_tail =
              tail_shell_count > 0 && ::isfinite(tau_tail);
          if (redistribute_tail) {
            const double absorption_fraction = 1.0 - ::exp(-tau_tail);
            double cumulative_fraction = 0.0;
            double tau_previous = 0.0;
            for (int idx = 0; idx < tail_shell_count; ++idx) {
              cumulative_fraction += tail_fractions[idx];
              const double tau_current =
                  (idx + 1 == tail_shell_count)
                      ? tau_tail
                      : tau_tail * cumulative_fraction;
              tail_weights[idx] =
                  (::exp(-tau_previous) - ::exp(-tau_current)) /
                  absorption_fraction;
              if (!::isfinite(tail_weights[idx])) {
                redistribute_tail = false;
                break;
              }
              tau_previous = tau_current;
            }
          }
        }
        if (redistribute_tail) {
          per_ray_shell[static_cast<std::size_t>(j) * batch_capacity +
                        local_ray] -= dP;
          // The normalized weights sum to one, so redistribution conserves dP.
          for (int idx = 0; idx < tail_shell_count; ++idx) {
            per_ray_shell[static_cast<std::size_t>(tail_shells[idx]) *
                              batch_capacity +
                          local_ray] += dP * tail_weights[idx];
          }
        }
        if (tail_stats != nullptr) {
          if (redistribute_tail) {
            atomicAdd(&tail_stats[0], 1.0);
            atomicAdd(&tail_stats[2], dP);
          } else {
            atomicAdd(&tail_stats[1], 1.0);
            atomicAdd(&tail_stats[3], dP);
          }
        }
        P = P_tail;
        if (phys_ext_active != 0 && phys_opt.crit_terminate_deposit != 0) {
          per_ray_shell[static_cast<std::size_t>(j) * batch_capacity +
                        local_ray] += P;
        } else {
          per_ray_unabsorbed[local_ray] = P;
        }
        return;
      }
    }

    const double a2 = b * b / (1.0 - n_hat);
    const double r_lower2 = r_lower * r_lower;
    const double r_upper2 = r_upper * r_upper;
    double ds = 0.0;
    if (r_lower2 <= a2) {
      const double r_turn =
          ::fmin(r_upper, ::fmax(r_lower, ::sqrt(::fmax(a2, 0.0))));
      ds = ::sqrt(::fmax(r_upper2 - r_turn * r_turn, 0.0));
      turning_shell = j;
    } else {
      ds = ::sqrt(::fmax(r_upper2 - a2, 0.0)) -
           ::sqrt(::fmax(r_lower2 - a2, 0.0));
    }
    if (!(::isfinite(ds) && ds >= 0.0)) {
      per_ray_unabsorbed[local_ray] = P;
      per_ray_invalid[local_ray] = 1.0;
      return;
    }
    const double tau = kappa * ds;
    if (tau_shell_out != nullptr) {
      // Shell j is radial interval [radial_node_r[j], radial_node_r[j + 1]].
      tau_shell_out[static_cast<std::size_t>(ray) * n_shells + j] += tau;
    }
    const double P_after = P * ::exp(-tau);
    per_ray_shell[static_cast<std::size_t>(j) * batch_capacity + local_ray] +=
        P - P_after;
    P = P_after;
    if (turning_shell >= 0) {
      break;
    }
  }

  if (terminated_at_critical) {
    return;
  }
  if (turning_shell < 0) {
    per_ray_unabsorbed[local_ray] = P;
    return;
  }

  if (phys_ext_active != 0 && phys_opt.ra_enable != 0) {
    const double f_ra = compute_ra_event_fraction(phys_opt, b);
    const double dP = f_ra * P;
    P -= dP;
    per_ray_ra[local_ray] += dP;
  }

  for (int j = turning_shell; j < n_shells; ++j) {
    const double r_lower = radial_node_r[j];
    const double r_upper = radial_node_r[j + 1];
    const double n_hat = radial_n_hat[j];
    double kappa = test_kappa_cm_inv;
    if (!(kappa > 0.0)) {
      kappa = compute_kappa_from_smooth(radial_smooth_kappa[j], n_hat, eps_n);
      if (phys_ext_active != 0 && phys_opt.langdon_model != 0 && kappa > 0.0 &&
          radial_T_e != nullptr) {
        kappa *= compute_langdon_factor(
            phys_opt.langdon_model, phys_opt.langdon_zcoll, I_vac, lambda_cm,
            radial_T_e[j], phys_opt.langdon_te_min_eV);
      }
    }
    const double a2 = b * b / (1.0 - n_hat);
    const double r_lower2 = r_lower * r_lower;
    const double r_upper2 = r_upper * r_upper;
    const double ds = (j == turning_shell)
                          ? ::sqrt(::fmax(r_upper2 - a2, 0.0))
                          : ::sqrt(::fmax(r_upper2 - a2, 0.0)) -
                                ::sqrt(::fmax(r_lower2 - a2, 0.0));
    const double tau = kappa * ds;
    if (tau_shell_out != nullptr) {
      tau_shell_out[static_cast<std::size_t>(ray) * n_shells + j] += tau;
    }
    const double P_after = P * ::exp(-tau);
    per_ray_shell[static_cast<std::size_t>(j) * batch_capacity + local_ray] +=
        P - P_after;
    P = P_after;
  }
  per_ray_unabsorbed[local_ray] = P;
}

__global__ void precompute_fast_trace_shell_overlaps_kernel(
    const double* __restrict__ radial_node_r,
    const double* __restrict__ hydro_r_edges,
    const int n_shells,
    const int n_hydro_cells,
    const int allowed_supercritical_cell,
    const int critical_adjacent_subcritical_cell,
    const double critical_adjacent_split_r,
    int* __restrict__ targets,
    double* __restrict__ weights) {
  const int shell = blockIdx.x * blockDim.x + threadIdx.x;
  if (shell >= n_shells) {
    return;
  }

  const int base = shell * kMaxShellOverlaps;
  for (int k = 0; k < kMaxShellOverlaps; ++k) {
    targets[base + k] = -1;
    weights[base + k] = 0.0;
  }

  const double r_lo = radial_node_r[shell];
  const double r_up = radial_node_r[shell + 1];
  const double r_lo3 = r_lo * r_lo * r_lo;
  const double r_up3 = r_up * r_up * r_up;
  const double shell_volume = r_up3 - r_lo3;
  if (r_up <= r_lo || shell_volume == 0.0) {
    const double r_mid = 0.5 * (r_lo + r_up);
    targets[base] = ray_trace_bodies::cbet_locate_deposit_cell_1d(
        hydro_r_edges, n_hydro_cells, allowed_supercritical_cell,
        critical_adjacent_subcritical_cell, critical_adjacent_split_r, r_mid);
    weights[base] = 1.0;
    return;
  }

  int lo = 0;
  int hi = n_hydro_cells + 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (hydro_r_edges[mid] <= r_lo) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  int n_overlaps = 0;
  double a = r_lo;
  int edge = lo;
  while (a < r_up) {
    const double b = (edge <= n_hydro_cells)
                         ? ::fmin(r_up, hydro_r_edges[edge])
                         : r_up;
    if (b > a) {
      const double b3 = b * b * b;
      const double a3 = a * a * a;
      const double weight = (b3 - a3) / shell_volume;
      const int slot = (n_overlaps < kMaxShellOverlaps)
                           ? n_overlaps
                           : kMaxShellOverlaps - 1;
      if (n_overlaps < kMaxShellOverlaps) {
        targets[base + slot] =
            ray_trace_bodies::cbet_locate_deposit_cell_1d(
                hydro_r_edges, n_hydro_cells, allowed_supercritical_cell,
                critical_adjacent_subcritical_cell, critical_adjacent_split_r,
                0.5 * (a + b));
      }
      weights[base + slot] += weight;
      ++n_overlaps;
    }
    a = b;
    ++edge;
  }

  const int n_entries =
      (n_overlaps < kMaxShellOverlaps) ? n_overlaps : kMaxShellOverlaps;
  double weight_sum = 0.0;
  for (int k = 0; k < n_entries; ++k) {
    weight_sum += weights[base + k];
  }
  for (int k = 0; k < n_entries; ++k) {
    weights[base + k] /= weight_sum;
  }
}

__global__ void reduce_fast_trace_per_shell_kernel(
    const double* __restrict__ per_ray_shell,
    const double* __restrict__ per_ray_unabsorbed,
    const double* __restrict__ per_ray_critical_hits,
    const double* __restrict__ per_ray_invalid,
    const double* __restrict__ per_ray_ra,
    const int batch_rays,
    const int batch_capacity,
    const int n_shells,
    double* __restrict__ shell_sum,
    double* __restrict__ reduced_scalars) {
  __shared__ double partial[kReductionBlockSize];

  const int output = blockIdx.x;
  const int lane = threadIdx.x;
  double acc = 0.0;
  if (output < n_shells) {
    const double* const shell_values =
        per_ray_shell + static_cast<std::size_t>(output) * batch_capacity;
    for (int ray = lane; ray < batch_rays; ray += blockDim.x) {
      acc += shell_values[ray];
    }
  } else {
    const int scalar = output - n_shells;
    const double* values = nullptr;
    if (scalar == kUnabsorbedScalar) {
      values = per_ray_unabsorbed;
    } else if (scalar == kRaScalar) {
      values = per_ray_ra;
    } else if (scalar == kCriticalHitsScalar) {
      values = per_ray_critical_hits;
    } else {
      values = per_ray_invalid;
    }
    for (int ray = lane; ray < batch_rays; ray += blockDim.x) {
      acc += values[ray];
    }
  }

  partial[lane] = acc;
  __syncthreads();
  // This fixed tree does NOT preserve the march's ray-ascending single-thread
  // order. The fast path has its own deterministic order; run-repeat bit
  // identity is the contract.
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) {
      partial[lane] += partial[lane + offset];
    }
    __syncthreads();
  }
  if (lane == 0) {
    if (output < n_shells) {
      shell_sum[output] = partial[0];
    } else {
      reduced_scalars[output - n_shells] = partial[0];
    }
  }
}

__global__ void sum_fast_trace_absorbed_per_ray_kernel(
    const double* __restrict__ per_ray_shell,
    const double* __restrict__ per_ray_ra,
    const int ray_offset,
    const int batch_rays,
    const int batch_capacity,
    const int n_shells,
    double* __restrict__ pabs_per_ray_out) {
  const int local_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (local_ray >= batch_rays) {
    return;
  }

  double absorbed = 0.0;
  for (int shell = 0; shell < n_shells; ++shell) {
    absorbed += per_ray_shell[static_cast<std::size_t>(shell) * batch_capacity +
                              local_ray];
  }
  // Tail closure and critical termination are already in per_ray_shell.
  // Fast-side RA remains separate until the hydro-cell gather.
  absorbed += per_ray_ra[local_ray];
  pabs_per_ray_out[ray_offset + local_ray] = absorbed;
}

__global__ void gather_fast_trace_tallies_kernel(
    const double* __restrict__ shell_sum,
    const double* __restrict__ reduced_scalars,
    const int* __restrict__ targets,
    const double* __restrict__ weights,
    const int n_shells,
    const int n_hydro_cells,
    const int critical_adjacent_subcritical_cell,
    double* __restrict__ deposit_1d,
    double* __restrict__ P_unabsorbed,
    double* __restrict__ d_ra_power_total,
    unsigned long long* __restrict__ critical_surface_hit_count,
    core::DeviceErrorFlags* __restrict__ error_flags) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell > n_hydro_cells) {
    return;
  }

  if (cell < n_hydro_cells) {
    double acc = 0.0;
    for (int shell = 0; shell < n_shells; ++shell) {
      const int base = shell * kMaxShellOverlaps;
      for (int k = 0; k < kMaxShellOverlaps; ++k) {
        if (targets[base + k] == cell) {
          acc += weights[base + k] * shell_sum[shell];
        }
      }
    }
    if (cell == critical_adjacent_subcritical_cell) {
      acc += reduced_scalars[kRaScalar];
    }
    deposit_1d[cell] += acc;
    return;
  }

  if (P_unabsorbed != nullptr) {
    *P_unabsorbed += reduced_scalars[kUnabsorbedScalar];
  }
  if (d_ra_power_total != nullptr) {
    *d_ra_power_total += reduced_scalars[kRaScalar];
  }
  if (critical_surface_hit_count != nullptr) {
    *critical_surface_hit_count += static_cast<unsigned long long>(
        reduced_scalars[kCriticalHitsScalar]);
  }
  if (reduced_scalars[kInvalidScalar] != 0.0 && error_flags != nullptr) {
    error_flags->invalid_cell = 1;
  }
}

}  // namespace

cudaError_t launch_fast_trace_1d(
    const RayArray1D& rays,
    const LaserMesh& mesh,
    const core::Config::LaserConfig& laser_cfg,
    const double lambda_cm,
    const double* d_hydro_r_edges,
    const int n_hydro_cells,
    const int allowed_supercritical_cell,
    const int critical_adjacent_subcritical_cell,
    const double critical_adjacent_split_r,
    double* d_deposit_1d,
    double* d_unabsorbed,
    core::DeviceErrorFlags* d_error_flags,
    unsigned long long* d_critical_surface_hit_count,
    cudaStream_t stream,
    int* d_step_histogram,
    int* d_step_count,
    int* d_traj_step_count,
    const int n_output_rays,
    const LaserPhysExtOptions* phys_ext,
    const double* d_radial_T_e,
    const double beam_P_w,
    const double beam_w_cm,
    double* d_ra_power_total,
    double* d_tau_shell_out,
    double* d_pabs_per_ray_out) {
  if (rays.n_rays <= 0) {
    return cudaSuccess;
  }
  TENRYU_ASSERT(mesh.radial_n_nodes >= 2,
                "fast_trace_1d requires at least 2 radial nodes");
  TENRYU_ASSERT(d_hydro_r_edges != nullptr,
                "fast_trace_1d requires hydro_r_edges");
  TENRYU_ASSERT(n_hydro_cells > 0,
                "fast_trace_1d requires positive hydro cell count");
  TENRYU_ASSERT(d_deposit_1d != nullptr,
                "fast_trace_1d requires deposit_1d");
  static double* const d_tail_stats = [] {
    double* p = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&p), 4 * sizeof(double)) !=
        cudaSuccess) {
      return static_cast<double*>(nullptr);
    }
    if (cudaMemset(p, 0, 4 * sizeof(double)) != cudaSuccess) {
      cudaFree(p);
      return static_cast<double*>(nullptr);
    }
    return p;
  }();
  static unsigned long long tail_stats_calls = 0;
  ++tail_stats_calls;

  if (d_step_histogram != nullptr) {
    cudaMemsetAsync(d_step_histogram, 0,
                    LaserMesh::kTraceStepHistSize * sizeof(int), stream);
  }
  if (d_step_count != nullptr) {
    cudaMemsetAsync(d_step_count, 0,
                    static_cast<std::size_t>(rays.n_rays) * sizeof(int), stream);
  }
  if (d_traj_step_count != nullptr && n_output_rays > 0) {
    cudaMemsetAsync(d_traj_step_count, 0,
                    static_cast<std::size_t>(n_output_rays) * sizeof(int), stream);
  }

  const int n_shells = mesh.radial_n_nodes - 1;
  const std::size_t cap_doubles = kPerRayShellBytesCap / sizeof(double);
  const bool use_batches =
      static_cast<std::size_t>(rays.n_rays) > cap_doubles /
                                                   static_cast<std::size_t>(n_shells);
  // Above the 512 MiB per-ray-shell threshold, preserve the same fixed order
  // in sequential batches of at most 8192 rays.
  const int batch_capacity = use_batches ? std::min(kRayBatchSize, rays.n_rays)
                                         : rays.n_rays;
  const std::size_t shell_doubles =
      static_cast<std::size_t>(batch_capacity) * n_shells;
  const std::size_t slab_doubles =
      shell_doubles + 4ULL * static_cast<std::size_t>(batch_capacity);
  const std::size_t reduction_doubles =
      static_cast<std::size_t>(n_shells) + kReducedScalarCount;
  const std::size_t scratch_bytes =
      (slab_doubles + reduction_doubles) * sizeof(double);
  double* const slab = static_cast<double*>(core::device_scratch_acquire(
      "fast_trace_1d:per_ray_shell", scratch_bytes));
  double* const shell_sum = slab + slab_doubles;
  double* const reduced_scalars = shell_sum + n_shells;
  const std::size_t overlap_slots =
      static_cast<std::size_t>(n_shells) * kMaxShellOverlaps;
  int* const overlap_targets = static_cast<int*>(core::device_scratch_acquire(
      "fast_trace_1d:shell_overlap_targets", overlap_slots * sizeof(int)));
  double* const overlap_weights =
      static_cast<double*>(core::device_scratch_acquire(
          "fast_trace_1d:shell_overlap_weights",
          overlap_slots * sizeof(double)));
  const int shell_grid = (n_shells + kBlockSize - 1) / kBlockSize;
  precompute_fast_trace_shell_overlaps_kernel<<<shell_grid, kBlockSize, 0,
                                                stream>>>(
      mesh.radial_node_r, d_hydro_r_edges, n_shells, n_hydro_cells,
      allowed_supercritical_cell, critical_adjacent_subcritical_cell,
      critical_adjacent_split_r, overlap_targets, overlap_weights);

  const LaserPhysExtOptions phys =
      (phys_ext != nullptr) ? *phys_ext : LaserPhysExtOptions{};
  for (int ray_offset = 0; ray_offset < rays.n_rays;
       ray_offset += batch_capacity) {
    const int batch_rays = std::min(batch_capacity, rays.n_rays - ray_offset);
    cudaMemset2DAsync(slab, static_cast<std::size_t>(batch_capacity) *
                                sizeof(double),
                      0, static_cast<std::size_t>(batch_rays) * sizeof(double),
                      n_shells, stream);
    double* const per_ray_unabsorbed = slab + shell_doubles;
    double* const per_ray_critical_hits = per_ray_unabsorbed + batch_rays;
    double* const per_ray_invalid = per_ray_critical_hits + batch_rays;
    double* const per_ray_ra = per_ray_invalid + batch_rays;
    cudaMemsetAsync(per_ray_unabsorbed, 0,
                    4ULL * static_cast<std::size_t>(batch_rays) * sizeof(double),
                    stream);

    const int ray_grid = (batch_rays + kBlockSize - 1) / kBlockSize;
    fast_trace_1d_kernel<<<ray_grid, kBlockSize, 0, stream>>>(
        mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
        mesh.radial_smooth_kappa, mesh.radial_dn_dr, d_radial_T_e, rays.R0,
        rays.Z0, rays.vR0, rays.vZ0, rays.power, rays.power0, ray_offset,
        batch_rays, batch_capacity, n_shells, laser_cfg.absorption.eps_n,
        laser_cfg.raytrace.eps_crit, laser_cfg.raytrace.test_kappa, lambda_cm,
        phys, phys_ext != nullptr ? 1 : 0, beam_P_w, beam_w_cm, slab,
        per_ray_unabsorbed, per_ray_critical_hits, per_ray_invalid, per_ray_ra,
        d_tau_shell_out, d_tail_stats);
    if (d_pabs_per_ray_out != nullptr) {
      sum_fast_trace_absorbed_per_ray_kernel<<<ray_grid, kBlockSize, 0, stream>>>(
          slab, per_ray_ra, ray_offset, batch_rays, batch_capacity, n_shells,
          d_pabs_per_ray_out);
    }
    reduce_fast_trace_per_shell_kernel<<<
        n_shells + kReducedScalarCount, kReductionBlockSize, 0, stream>>>(
        slab, per_ray_unabsorbed, per_ray_critical_hits, per_ray_invalid,
        per_ray_ra, batch_rays, batch_capacity, n_shells, shell_sum,
        reduced_scalars);
    // Gather each batch into the final tallies in fixed sequential batch order.
    const int reduce_total = n_hydro_cells + 1;
    const int reduce_grid = (reduce_total + kBlockSize - 1) / kBlockSize;
    gather_fast_trace_tallies_kernel<<<reduce_grid, kBlockSize, 0, stream>>>(
        shell_sum, reduced_scalars, overlap_targets, overlap_weights, n_shells,
        n_hydro_cells, critical_adjacent_subcritical_cell, d_deposit_1d,
        d_unabsorbed, d_ra_power_total, d_critical_surface_hit_count,
        d_error_flags);
  }
  if (d_tail_stats != nullptr && (tail_stats_calls % 5000ULL) == 0ULL) {
    double h_tail_stats[4] = {0.0, 0.0, 0.0, 0.0};
    const cudaError_t copy_status =
        cudaMemcpyAsync(h_tail_stats, d_tail_stats, sizeof(h_tail_stats),
                        cudaMemcpyDeviceToHost, stream);
    if (copy_status == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess) {
      char msg[256];
      std::snprintf(
          msg, sizeof(msg),
          "[laser][fast-tail] calls=%llu redistributed=%llu fallback=%llu "
          "dP_redist_total=%.6e dP_lump_total=%.6e",
          tail_stats_calls,
          static_cast<unsigned long long>(h_tail_stats[0]),
          static_cast<unsigned long long>(h_tail_stats[1]),
          h_tail_stats[2], h_tail_stats[3]);
      core::log_info(msg);
    }
  }
  return cudaGetLastError();
}

}  // namespace tenryu::laser
