#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "laser/coordinate_transform.cuh"

namespace tenryu::laser {

struct RaytraceSkipCache {
  double* rho_cached = nullptr;
  double* Te_cached = nullptr;
  double* Zbar_cached = nullptr;
  double* f_hat = nullptr;   // [n_groups * n_cells]
  double* delta = nullptr;   // [n_cells]
  int* d_crit_hit = nullptr; // [1]

  int n_cells = 0;
  int n_groups = 0;
  bool valid = false;
  int consecutive_skip_count = 0;
  double cached_total_power = 0.0;
  std::vector<double> cached_group_powers;
  std::vector<Vec3> cached_beam_dirs;
  std::vector<Vec3> cached_beam_focuses;
  std::vector<double> cached_beam_defocus;

  RaytraceSkipCache() = default;
  ~RaytraceSkipCache();
  RaytraceSkipCache(const RaytraceSkipCache&) = delete;
  RaytraceSkipCache& operator=(const RaytraceSkipCache&) = delete;
  RaytraceSkipCache(RaytraceSkipCache&&) = delete;
  RaytraceSkipCache& operator=(RaytraceSkipCache&&) = delete;

  void release();
  void ensure_capacity(int cells, int groups);
  void invalidate();

  [[nodiscard]] bool should_skip(const core::State& state,
                                 const core::Config::LaserConfig& config,
                                 double n_crit,
                                 double n_hat_margin,
                                 const std::vector<double>& material_A,
                                 double A_eff_uniform,
                                 const std::vector<double>& group_powers,
                                 const std::vector<Vec3>& beam_dirs,
                                 const std::vector<Vec3>& beam_focuses,
                                 const std::vector<double>& beam_defocus,
                                 bool ale_rezoned,
                                 cudaStream_t stream = nullptr,
                                 double rho_floor = 1.0e-10,
                                 double Te_floor = 1.0e-3,
                                 double* metric_out = nullptr,
                                 bool* eligible_out = nullptr,
                                 bool* crit_hit_out = nullptr,
                                 bool cbet_active = false);

  void scale_deposit(core::State& state,
                     const std::vector<double>& group_powers,
                     double dt,
                     cudaStream_t stream = nullptr) const;

  void update_cache(const core::State& state,
                    const std::vector<std::vector<double>>& f_hat_group_host,
                    const std::vector<double>& group_powers,
                    const std::vector<Vec3>& beam_dirs,
                    const std::vector<Vec3>& beam_focuses,
                    const std::vector<double>& beam_defocus,
                    cudaStream_t stream = nullptr);
};

}  // namespace tenryu::laser
