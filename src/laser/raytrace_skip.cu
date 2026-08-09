#include "laser/raytrace_skip.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::laser {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ inline void ray_skip_check_kernel_body(
    const int tid,
    double* __restrict__ delta_max_cell,
    int* __restrict__ crit_hit,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ Zbar,
    const double* __restrict__ volFrac,
    const double* __restrict__ A_mat,
    const int n_mat,
    const double* __restrict__ rho_cached,
    const double* __restrict__ Te_cached,
    const double* __restrict__ Zbar_cached,
    const double rho_floor,
    const double Te_floor,
    const double Zbar_floor,
    const double n_crit,
    const double n_hat_margin,
    const double crit_guard,
    const double A_eff_uniform,
    const int use_l2_relative,
    const int n_cells) {
  const double rho_now = rho[tid];
  const double Te_now = Te[tid];
  const double Zbar_now = Zbar[tid];

  const double rho_ref = rho_cached[tid];
  const double Te_ref = Te_cached[tid];
  const double Zbar_ref = Zbar_cached[tid];

  const double abs_drho = fabs(rho_now - rho_ref);
  const double abs_dTe = fabs(Te_now - Te_ref);
  const double abs_dZ = fabs(Zbar_now - Zbar_ref);
  if (use_l2_relative != 0) {
    const double denom_rho = fmax(fabs(rho_now), rho_floor);
    const double denom_Te = fmax(fabs(Te_now), Te_floor);
    const double denom_Z = fmax(fabs(Zbar_now), Zbar_floor);
    const double d_rho = abs_drho / denom_rho;
    const double d_Te = abs_dTe / denom_Te;
    const double d_Z = abs_dZ / denom_Z;
    delta_max_cell[tid] = d_rho * d_rho + d_Te * d_Te + d_Z * d_Z;
  } else {
    const double denom_rho = fmax(fabs(rho_ref), rho_floor);
    const double denom_Te = fmax(fabs(Te_ref), Te_floor);
    const double denom_Z = fmax(fabs(Zbar_ref), Zbar_floor);
    const double d_rho = abs_drho / denom_rho;
    const double d_Te = abs_dTe / denom_Te;
    const double d_Z = abs_dZ / denom_Z;
    delta_max_cell[tid] = fmax(d_rho, fmax(d_Te, d_Z));
  }

  double A_eff = fmax(A_eff_uniform, 1.0e-30);
  if (A_mat != nullptr && n_mat > 0) {
    if (n_mat == 1) {
      A_eff = fmax(A_mat[0], 1.0e-30);
    } else if (volFrac != nullptr) {
      double inv_A_eff = 0.0;
      const int base = tid * n_mat;
      for (int m = 0; m < n_mat; ++m) {
        const double f = fmax(volFrac[base + m], 0.0);
        const double A = fmax(A_mat[m], 1.0e-30);
        inv_A_eff += f / A;
      }
      if (inv_A_eff > 1.0e-30) {
        A_eff = 1.0 / inv_A_eff;
      }
    }
  }
  const double ne_now = fmax(0.0, rho_now) * fmax(0.0, Zbar_now) /
                        (fmax(A_eff, 1.0e-30) * core::constants::proton_mass);
  const double ne_cached = fmax(0.0, rho_ref) * fmax(0.0, Zbar_ref) /
                           (fmax(A_eff, 1.0e-30) * core::constants::proton_mass);
  const double band = n_hat_margin - crit_guard;
  const double inv_ncrit = 1.0 / fmax(n_crit, 1.0e-30);
  const bool above_now = ne_now * inv_ncrit > band;
  const bool above_cached = ne_cached * inv_ncrit > band;
  if (above_now != above_cached) {
    atomicExch(crit_hit, 1);
  }
}

__global__ void ray_skip_check_kernel(double* __restrict__ delta_max_cell,
                                      int* __restrict__ crit_hit,
                                      const double* __restrict__ rho,
                                      const double* __restrict__ Te,
                                      const double* __restrict__ Zbar,
                                      const double* __restrict__ volFrac,
                                      const double* __restrict__ A_mat,
                                      const int n_mat,
                                      const double* __restrict__ rho_cached,
                                      const double* __restrict__ Te_cached,
                                      const double* __restrict__ Zbar_cached,
                                      const double rho_floor,
                                      const double Te_floor,
                                      const double Zbar_floor,
                                      const double n_crit,
                                      const double n_hat_margin,
                                      const double crit_guard,
                                      const double A_eff_uniform,
                                      const int use_l2_relative,
                                      const int n_cells) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_cells) {
    return;
  }

  ray_skip_check_kernel_body(tid, delta_max_cell, crit_hit, rho, Te, Zbar,
                             volFrac, A_mat, n_mat, rho_cached, Te_cached,
                             Zbar_cached, rho_floor, Te_floor, Zbar_floor,
                             n_crit, n_hat_margin, crit_guard, A_eff_uniform,
                             use_l2_relative, n_cells);
}

__global__ void reconstruct_laser_dep_kernel(double* __restrict__ laser_dep,
                                             const double* __restrict__ f_hat,
                                             const double* __restrict__ P_g_dt,
                                             const int n_groups,
                                             const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double sum = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    sum += f_hat[g * n_cells + c] * P_g_dt[g];
  }
  laser_dep[c] = sum;
}

}  // namespace

RaytraceSkipCache::~RaytraceSkipCache() {
  release();
}

void RaytraceSkipCache::release() {
  if (rho_cached != nullptr) {
    cuda_check(cudaFree(rho_cached), "RaytraceSkipCache::release cudaFree rho_cached failed");
    rho_cached = nullptr;
  }
  if (Te_cached != nullptr) {
    cuda_check(cudaFree(Te_cached), "RaytraceSkipCache::release cudaFree Te_cached failed");
    Te_cached = nullptr;
  }
  if (Zbar_cached != nullptr) {
    cuda_check(cudaFree(Zbar_cached), "RaytraceSkipCache::release cudaFree Zbar_cached failed");
    Zbar_cached = nullptr;
  }
  if (f_hat != nullptr) {
    cuda_check(cudaFree(f_hat), "RaytraceSkipCache::release cudaFree f_hat failed");
    f_hat = nullptr;
  }
  if (delta != nullptr) {
    cuda_check(cudaFree(delta), "RaytraceSkipCache::release cudaFree delta failed");
    delta = nullptr;
  }
  if (d_crit_hit != nullptr) {
    cuda_check(cudaFree(d_crit_hit), "RaytraceSkipCache::release cudaFree d_crit_hit failed");
    d_crit_hit = nullptr;
  }

  n_cells = 0;
  n_groups = 0;
  valid = false;
  consecutive_skip_count = 0;
  cached_total_power = 0.0;
  cached_group_powers.clear();
  cached_beam_dirs.clear();
  cached_beam_focuses.clear();
  cached_beam_defocus.clear();
}

void RaytraceSkipCache::ensure_capacity(const int cells, const int groups) {
  if (cells <= 0 || groups <= 0) {
    release();
    return;
  }
  if (cells == n_cells && groups == n_groups && rho_cached != nullptr && Te_cached != nullptr &&
      Zbar_cached != nullptr && f_hat != nullptr && delta != nullptr && d_crit_hit != nullptr) {
    return;
  }

  release();
  n_cells = cells;
  n_groups = groups;

  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  const std::size_t fhat_bytes = static_cast<std::size_t>(n_cells) *
                                 static_cast<std::size_t>(n_groups) * sizeof(double);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&rho_cached), cell_bytes),
             "RaytraceSkipCache::ensure_capacity cudaMalloc rho_cached failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&Te_cached), cell_bytes),
             "RaytraceSkipCache::ensure_capacity cudaMalloc Te_cached failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&Zbar_cached), cell_bytes),
             "RaytraceSkipCache::ensure_capacity cudaMalloc Zbar_cached failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&f_hat), fhat_bytes),
             "RaytraceSkipCache::ensure_capacity cudaMalloc f_hat failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&delta), cell_bytes),
             "RaytraceSkipCache::ensure_capacity cudaMalloc delta failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_crit_hit), sizeof(int)),
             "RaytraceSkipCache::ensure_capacity cudaMalloc d_crit_hit failed");
  cuda_check(cudaMemset(f_hat, 0, fhat_bytes),
             "RaytraceSkipCache::ensure_capacity memset f_hat failed");

  valid = false;
  consecutive_skip_count = 0;
  cached_total_power = 0.0;
  cached_group_powers.assign(static_cast<std::size_t>(n_groups), 0.0);
  cached_beam_dirs.clear();
  cached_beam_focuses.clear();
  cached_beam_defocus.clear();
}

void RaytraceSkipCache::invalidate() {
  valid = false;
  consecutive_skip_count = 0;
  cached_total_power = 0.0;
  std::fill(cached_group_powers.begin(), cached_group_powers.end(), 0.0);
  cached_beam_dirs.clear();
  cached_beam_focuses.clear();
  cached_beam_defocus.clear();
}

bool RaytraceSkipCache::should_skip(const core::State& state,
                                    const core::Config::LaserConfig& config,
                                    const double n_crit,
                                    const double n_hat_margin,
                                    const std::vector<double>& material_A,
                                    const double A_eff_uniform,
                                    const std::vector<double>& group_powers,
                                    const std::vector<Vec3>& beam_dirs,
                                    const std::vector<Vec3>& beam_focuses,
                                    const std::vector<double>& beam_defocus,
                                    const bool ale_rezoned,
                                    cudaStream_t stream,
                                    const double rho_floor,
                                    const double Te_floor,
                                    double* metric_out,
                                    bool* eligible_out,
                                    bool* crit_hit_out,
                                    const bool cbet_active) {
  if (metric_out != nullptr) {
    *metric_out = std::numeric_limits<double>::infinity();
  }
  if (eligible_out != nullptr) {
    *eligible_out = false;
  }
  if (crit_hit_out != nullptr) {
    *crit_hit_out = false;
  }
  const auto& skip_cfg = config.raytrace_skip_config;
  if (!skip_cfg.enabled) {
    consecutive_skip_count = 0;
    return false;
  }
  ++ctr_calls;
  if ((ctr_calls % 5000) == 0) {
    core::log_info("[laser][skip] calls=" + std::to_string(ctr_calls) +
                   " fires=" + std::to_string(ctr_fires) +
                   " veto{warmup=" + std::to_string(ctr_veto_warmup) +
                   " maxc=" + std::to_string(ctr_veto_maxconsec) +
                   " ale=" + std::to_string(ctr_veto_ale) +
                   " power=" + std::to_string(ctr_veto_power) +
                   " geom=" + std::to_string(ctr_veto_geometry) +
                   " crit=" + std::to_string(ctr_veto_crit) +
                   " metric=" + std::to_string(ctr_veto_metric) + "}");
  }
  if (!valid || n_cells <= 0 || n_groups <= 0 || state.step == 0) {
    ++ctr_veto_warmup;
    consecutive_skip_count = 0;
    return false;
  }
  if (consecutive_skip_count >= skip_cfg.max_consecutive) {
    ++ctr_veto_maxconsec;
    consecutive_skip_count = 0;
    return false;
  }
  if (ale_rezoned) {
    ++ctr_veto_ale;
    consecutive_skip_count = 0;
    return false;
  }

  const double total_power =
      std::accumulate(group_powers.begin(), group_powers.end(), 0.0, [](double a, double b) {
        return a + std::max(0.0, b);
      });
  const bool now_zero = !(total_power > 0.0);
  const bool cached_zero = !(cached_total_power > 0.0);
  if (now_zero != cached_zero) {
    ++ctr_veto_power;
    consecutive_skip_count = 0;
    return false;
  }
  const double power_rel_change =
      std::abs(total_power - cached_total_power) / std::max(cached_total_power, 1.0e-30);
  if (power_rel_change > 0.01) {
    ++ctr_veto_power;
    consecutive_skip_count = 0;
    return false;
  }

  if (group_powers.size() != cached_group_powers.size()) {
    ++ctr_veto_power;
    consecutive_skip_count = 0;
    return false;
  }
  const int n_compare_groups = static_cast<int>(group_powers.size());
  for (int g = 0; g < n_compare_groups; ++g) {
    const bool now_g_zero = !(group_powers[static_cast<std::size_t>(g)] > 0.0);
    const bool cached_g_zero = !(cached_group_powers[static_cast<std::size_t>(g)] > 0.0);
    if (now_g_zero != cached_g_zero) {
      ++ctr_veto_power;
      consecutive_skip_count = 0;
      return false;
    }
  }
  if (cbet_active) {
    for (int g = 0; g < n_compare_groups; ++g) {
      const double now_p = group_powers[static_cast<std::size_t>(g)];
      const double was_p = cached_group_powers[static_cast<std::size_t>(g)];
      const double rel_g =
          std::abs(now_p - was_p) / std::max(std::abs(was_p), 1.0e-30);
      if (rel_g > 0.01) {
        // CBET couples beams nonlinearly; per-beam power scaling is no longer
        // exact, so any per-group drift forces a full re-trace.
        ++ctr_veto_power;
        consecutive_skip_count = 0;
        return false;
      }
    }
  }
  if (beam_dirs.size() != cached_beam_dirs.size()) {
    ++ctr_veto_geometry;
    consecutive_skip_count = 0;
    return false;
  }
  if (beam_focuses.size() != cached_beam_focuses.size() ||
      beam_defocus.size() != cached_beam_defocus.size()) {
    ++ctr_veto_geometry;
    consecutive_skip_count = 0;
    return false;
  }
  constexpr double kBeamDirL2Tol = 1.0e-10;
  for (std::size_t i = 0; i < beam_dirs.size(); ++i) {
    const double dx = beam_dirs[i].x - cached_beam_dirs[i].x;
    const double dy = beam_dirs[i].y - cached_beam_dirs[i].y;
    const double dz = beam_dirs[i].z - cached_beam_dirs[i].z;
    const double l2 = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (l2 > kBeamDirL2Tol) {
      ++ctr_veto_geometry;
      consecutive_skip_count = 0;
      return false;
    }
  }
  constexpr double kBeamFocusL2Tol = 1.0e-10;
  for (std::size_t i = 0; i < beam_focuses.size(); ++i) {
    const double dx = beam_focuses[i].x - cached_beam_focuses[i].x;
    const double dy = beam_focuses[i].y - cached_beam_focuses[i].y;
    const double dz = beam_focuses[i].z - cached_beam_focuses[i].z;
    const double l2 = std::sqrt(dx * dx + dy * dy + dz * dz);
    if (l2 > kBeamFocusL2Tol) {
      ++ctr_veto_geometry;
      consecutive_skip_count = 0;
      return false;
    }
  }
  constexpr double kBeamDefocusTol = 1.0e-12;
  for (std::size_t i = 0; i < beam_defocus.size(); ++i) {
    if (std::abs(beam_defocus[i] - cached_beam_defocus[i]) > kBeamDefocusTol) {
      ++ctr_veto_geometry;
      consecutive_skip_count = 0;
      return false;
    }
  }
  if (eligible_out != nullptr) {
    *eligible_out = true;
  }

  const int block = 256;
  const int grid = (n_cells + block - 1) / block;
  const double rho_floor_use = std::max(rho_floor, 0.0);
  const double Te_floor_use = std::max(Te_floor, 0.0);
  const double Zbar_floor = 1.0e-2;
  const int use_l2_relative = (skip_cfg.norm == "l2_relative") ? 1 : 0;
  const int n_mat_total = static_cast<int>(material_A.size());
  const bool use_multimat_Aeff =
      (n_mat_total > 1) &&
      (state.volFrac.size() ==
       static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat_total));
  const int n_mat_kernel = use_multimat_Aeff ? n_mat_total : std::min(1, n_mat_total);
  const double* volfrac_ptr = use_multimat_Aeff ? state.volFrac.data() : nullptr;
  double* d_A_mat = nullptr;
  if (n_mat_kernel > 0) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_A_mat),
                          static_cast<std::size_t>(n_mat_kernel) * sizeof(double)),
               "RaytraceSkipCache::should_skip cudaMalloc d_A_mat failed");
    cuda_check(cudaMemcpyAsync(d_A_mat, material_A.data(),
                               static_cast<std::size_t>(n_mat_kernel) * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "RaytraceSkipCache::should_skip memcpyAsync d_A_mat failed");
  }

  cuda_check(cudaMemsetAsync(d_crit_hit, 0, sizeof(int), stream),
             "RaytraceSkipCache::should_skip memset d_crit_hit failed");
  ray_skip_check_kernel<<<grid, block, 0, stream>>>(delta, d_crit_hit, state.rho.data(),
                                                     state.Te.data(), state.zbar.data(),
                                                     volfrac_ptr, d_A_mat, n_mat_kernel,
                                                     rho_cached, Te_cached, Zbar_cached,
                                                     rho_floor_use, Te_floor_use, Zbar_floor,
                                                     n_crit, n_hat_margin, skip_cfg.crit_guard,
                                                     A_eff_uniform, use_l2_relative, n_cells);
  cuda_check(cudaGetLastError(), "RaytraceSkipCache::should_skip kernel launch failed");

  std::vector<double> h_delta(static_cast<std::size_t>(n_cells), 0.0);
  int crit_hit = 0;
  cuda_check(cudaMemcpyAsync(h_delta.data(), delta, h_delta.size() * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "RaytraceSkipCache::should_skip memcpy delta D2H failed");
  cuda_check(cudaMemcpyAsync(&crit_hit, d_crit_hit, sizeof(int), cudaMemcpyDeviceToHost, stream),
             "RaytraceSkipCache::should_skip memcpy crit_hit D2H failed");
  cuda_check(cudaStreamSynchronize(stream),
             "RaytraceSkipCache::should_skip stream synchronize failed");
  if (d_A_mat != nullptr) {
    cuda_check(cudaFree(d_A_mat), "RaytraceSkipCache::should_skip cudaFree d_A_mat failed");
  }

  if (crit_hit != 0) {
    if (crit_hit_out != nullptr) {
      *crit_hit_out = true;
    }
    ++ctr_veto_crit;
    consecutive_skip_count = 0;
    return false;
  }

  double metric = 0.0;
  if (use_l2_relative != 0) {
    long double sum2 = 0.0L;
    for (const double d : h_delta) {
      if (!std::isfinite(d)) {
        metric = std::numeric_limits<double>::infinity();
        break;
      }
      sum2 += static_cast<long double>(d);
    }
    if (std::isfinite(metric)) {
      metric = std::sqrt(static_cast<double>(sum2 / (3.0L * std::max(1, n_cells))));
    }
  } else {
    for (const double d : h_delta) {
      if (!std::isfinite(d)) {
        metric = std::numeric_limits<double>::infinity();
        break;
      }
      metric = std::max(metric, d);
    }
  }
  if (metric_out != nullptr) {
    *metric_out = metric;
  }

  if (metric < skip_cfg.threshold) {
    ++ctr_fires;
    ++consecutive_skip_count;
    return true;
  }

  ++ctr_veto_metric;
  consecutive_skip_count = 0;
  return false;
}

void RaytraceSkipCache::scale_deposit(core::State& state,
                                      const std::vector<double>& group_powers,
                                      const double dt,
                                      cudaStream_t stream) const {
  if (!valid || n_cells <= 0 || n_groups <= 0 || !(dt > 0.0)) {
    state.laser_dep.fill(0.0);
    return;
  }

  std::vector<double> P_g_dt(static_cast<std::size_t>(n_groups), 0.0);
  for (int g = 0; g < n_groups; ++g) {
    const double P = (g < static_cast<int>(group_powers.size()))
                         ? std::max(0.0, group_powers[static_cast<std::size_t>(g)])
                         : 0.0;
    P_g_dt[static_cast<std::size_t>(g)] = P * dt;
  }

  double* d_P_g_dt = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_P_g_dt),
                        static_cast<std::size_t>(n_groups) * sizeof(double)),
             "RaytraceSkipCache::scale_deposit cudaMalloc d_P_g_dt failed");
  cuda_check(cudaMemcpyAsync(d_P_g_dt, P_g_dt.data(),
                             static_cast<std::size_t>(n_groups) * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "RaytraceSkipCache::scale_deposit memcpyAsync d_P_g_dt failed");

  const int block = 256;
  const int grid = (n_cells + block - 1) / block;
  reconstruct_laser_dep_kernel<<<grid, block, 0, stream>>>(state.laser_dep.data(), f_hat,
                                                            d_P_g_dt, n_groups, n_cells);
  cuda_check(cudaGetLastError(),
             "RaytraceSkipCache::scale_deposit reconstruct kernel launch failed");
  cuda_check(cudaFree(d_P_g_dt), "RaytraceSkipCache::scale_deposit cudaFree d_P_g_dt failed");
}

void RaytraceSkipCache::update_cache(const core::State& state,
                                     const std::vector<std::vector<double>>& f_hat_group_host,
                                     const std::vector<double>& group_powers,
                                     const std::vector<Vec3>& beam_dirs,
                                     const std::vector<Vec3>& beam_focuses,
                                     const std::vector<double>& beam_defocus,
                                     cudaStream_t stream) {
  if (n_cells <= 0 || n_groups <= 0) {
    invalidate();
    return;
  }

  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  cuda_check(cudaMemcpyAsync(rho_cached, state.rho.data(), cell_bytes, cudaMemcpyDeviceToDevice,
                             stream),
             "RaytraceSkipCache::update_cache memcpy rho D2D failed");
  cuda_check(cudaMemcpyAsync(Te_cached, state.Te.data(), cell_bytes, cudaMemcpyDeviceToDevice,
                             stream),
             "RaytraceSkipCache::update_cache memcpy Te D2D failed");
  cuda_check(cudaMemcpyAsync(Zbar_cached, state.zbar.data(), cell_bytes, cudaMemcpyDeviceToDevice,
                             stream),
             "RaytraceSkipCache::update_cache memcpy Zbar D2D failed");

  std::vector<double> flat_fhat(static_cast<std::size_t>(n_cells) *
                                    static_cast<std::size_t>(n_groups),
                                0.0);
  for (int g = 0; g < n_groups; ++g) {
    if (g >= static_cast<int>(f_hat_group_host.size())) {
      continue;
    }
    const auto& fg = f_hat_group_host[static_cast<std::size_t>(g)];
    if (fg.size() != static_cast<std::size_t>(n_cells)) {
      continue;
    }
    std::copy(fg.begin(), fg.end(), flat_fhat.begin() + static_cast<std::size_t>(g) * n_cells);
  }

  cuda_check(cudaMemcpyAsync(f_hat, flat_fhat.data(),
                             flat_fhat.size() * sizeof(double), cudaMemcpyHostToDevice, stream),
             "RaytraceSkipCache::update_cache memcpy f_hat H2D failed");
  cuda_check(cudaStreamSynchronize(stream),
             "RaytraceSkipCache::update_cache stream synchronize failed");

  cached_group_powers.assign(static_cast<std::size_t>(n_groups), 0.0);
  for (int g = 0; g < n_groups; ++g) {
    if (g < static_cast<int>(group_powers.size())) {
      cached_group_powers[static_cast<std::size_t>(g)] =
          std::max(0.0, group_powers[static_cast<std::size_t>(g)]);
    }
  }
  cached_total_power =
      std::accumulate(cached_group_powers.begin(), cached_group_powers.end(), 0.0);
  cached_beam_dirs = beam_dirs;
  cached_beam_focuses = beam_focuses;
  cached_beam_defocus = beam_defocus;
  valid = true;
  consecutive_skip_count = 0;
}

}  // namespace tenryu::laser
