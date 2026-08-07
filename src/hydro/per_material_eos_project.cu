#include "hydro/per_material_eos_project.cuh"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/eos_context.hpp"
#include "hydro/per_material_eos_accessors.cuh"

namespace tenryu::hydro::per_material {
namespace {

enum CounterIndex : int {
  kCounterEOSInverse = kPerMaterialCounterEOSInverse,
  kCounterLazyCacheHit = kPerMaterialCounterLazyCacheHit,
  kCounterLazyCacheMiss = kPerMaterialCounterLazyCacheMiss,
  kCounterMixtureProjection = kPerMaterialCounterMixtureProjection,
  kCounterEOSTableValidityViolation = kPerMaterialCounterEOSTableValidityViolation,
  kCounterPresenceAbsent = kPerMaterialCounterPresenceAbsent,
  kCounterCount = kPerMaterialCounterCount,
};

struct MaterialEOSParams {
  double A = 1.0;
  double Zbar = 1.0;
  double gamma = 5.0 / 3.0;
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline int checked_int(const std::size_t value, const char* message) {
  TENRYU_ASSERT(value <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                message);
  return static_cast<int>(value);
}

bool material_requires_eos_context(const tenryu::core::Config::MaterialsConfig::MatDef& mat) {
  return mat.eos_tables != nullptr || mat.eos_model != "ideal_gas";
}

bool eos_bisection_profile_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("TENRYU_PROFILE_EOS_BISECTION");
    return value != nullptr && value[0] != '\0' && std::string(value) != "0";
  }();
  return enabled;
}

std::vector<MaterialEOSParams> make_material_params(const core::Config& cfg) {
  std::vector<MaterialEOSParams> params(cfg.materials.materials.size());
  for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
    const auto& mat = cfg.materials.materials[m];
    MaterialEOSParams p{};
    p.A = (mat.A > 0.0) ? mat.A : 1.0;
    p.gamma = (mat.ideal_gas_gamma > 1.0) ? mat.ideal_gas_gamma : (5.0 / 3.0);
    if (cfg.materials.zbar.model == "fixed" && cfg.materials.zbar.fixed_value >= 0.0) {
      p.Zbar = cfg.materials.zbar.fixed_value;
    } else {
      p.Zbar = (mat.Z > 0.0) ? mat.Z : 1.0;
    }
    if (mat.is_void) {
      p.Zbar = 0.0;
    }
    params[m] = p;
  }
  return params;
}

__device__ inline double atomic_add_double(double* address, const double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        value + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline void atomic_max_double_positive(double* address, const double value) {
  if (!(value > 0.0) || !isfinite(value)) {
    return;
  }
#if __CUDA_ARCH__ >= 600
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    const double current = __longlong_as_double(static_cast<long long>(assumed));
    if (current >= value) {
      return;
    }
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
  } while (assumed != old);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    const double current = __longlong_as_double(static_cast<long long>(assumed));
    if (current >= value) {
      return;
    }
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
  } while (assumed != old);
#endif
}

__global__ void project_per_material_thermo_kernel(
    PerMaterialAccessorView view,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ ion_views,
    const MaterialEOSParams* __restrict__ params,
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    double* __restrict__ cs,
    double* __restrict__ zbar,
    double* __restrict__ cv_e,
    double* __restrict__ cv_i,
    const double te_floor,
    const double ti_floor,
    const bool low_density_extrap) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = view.n_cells * view.n_mat;
  if (idx >= total || view.n_cells <= 0 || view.n_mat <= 0) {
    return;
  }

  const int c = idx / view.n_mat;
  const int m = idx - c * view.n_mat;
  const double vf = view.volfrac[idx];
  if (!(vf > view.presence_threshold_volfrac) || !isfinite(vf)) {
    record_per_material_counter(view, kCounterPresenceAbsent);
    return;
  }

  const double mass_m = view.mass_per_material[idx];
  const double V = view.vol[c];
  if (!(mass_m > 0.0) || !(V > 0.0) || !isfinite(mass_m) || !isfinite(V)) {
    record_per_material_counter(view, kCounterPresenceAbsent);
    return;
  }
  const double rho_m = mass_m / (vf * V);
  if (!(rho_m > view.presence_threshold_mass_density_g_per_cc) || !isfinite(rho_m)) {
    record_per_material_counter(view, kCounterPresenceAbsent);
    return;
  }

  const MaterialEOSParams p = params[m];
  const tenryu::materials::DeviceEOSTableView electron_view =
      (electron_views != nullptr) ? electron_views[m] : tenryu::materials::DeviceEOSTableView{};
  const tenryu::materials::DeviceEOSTableView ion_view =
      (ion_views != nullptr) ? ion_views[m] : tenryu::materials::DeviceEOSTableView{};

  const PerMaterialThermo electron = get_electron_thermo_per_material(view,
                                                                       electron_view,
                                                                       c,
                                                                       m,
                                                                       p.Zbar,
                                                                       p.A,
                                                                       te_floor,
                                                                       low_density_extrap,
                                                                       p.gamma);
  const PerMaterialThermo ion = get_ion_thermo_per_material(view,
                                                            ion_view,
                                                            c,
                                                            m,
                                                            p.A,
                                                            ti_floor,
                                                            low_density_extrap,
                                                            p.gamma);
  const double cs_m = get_cs(electron, ion);
  atomic_add_double(Te + c, mass_m * get_te(electron));
  atomic_add_double(Ti + c, mass_m * get_ti(ion));
  atomic_add_double(Pe + c, vf * get_pe(electron));
  atomic_add_double(Pi + c, vf * get_pi(ion));
  atomic_max_double_positive(cs + c, cs_m);
  atomic_add_double(zbar + c, mass_m * get_zbar_per_material(p.Zbar));
  atomic_add_double(cv_e + c, mass_m * get_cv_e(electron));
  atomic_add_double(cv_i + c, mass_m * get_cv_i(ion));
}

__global__ void normalize_per_material_projection_kernel(double* __restrict__ Te,
                                                         double* __restrict__ Ti,
                                                         double* __restrict__ ee,
                                                         double* __restrict__ ei,
                                                         double* __restrict__ zbar,
                                                         double* __restrict__ cv_e,
                                                         double* __restrict__ cv_i,
                                                         const double* __restrict__ mass,
                                                         const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double m = mass[c];
  if (m > 0.0 && isfinite(m)) {
    Te[c] /= m;
    Ti[c] /= m;
    ee[c] /= m;
    ei[c] /= m;
    zbar[c] /= m;
    cv_e[c] /= m;
    cv_i[c] /= m;
  } else {
    Te[c] = 0.0;
    Ti[c] = 0.0;
    ee[c] = 0.0;
    ei[c] = 0.0;
    zbar[c] = 0.0;
    cv_e[c] = 0.0;
    cv_i[c] = 0.0;
  }
}

__global__ void reduce_per_material_specific_energy_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ Ee_per_material,
    const double* __restrict__ Ei_per_material,
    const int n_cells,
    const int n_mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double Ee_sum = 0.0;
  double Ei_sum = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    const int idx = c * n_mat + m;
    const double Ee_m = Ee_per_material[idx];
    const double Ei_m = Ei_per_material[idx];
    Ee_sum += (isfinite(Ee_m) && Ee_m > 0.0) ? Ee_m : 0.0;
    Ei_sum += (isfinite(Ei_m) && Ei_m > 0.0) ? Ei_m : 0.0;
  }
  ee[c] = Ee_sum;
  ei[c] = Ei_sum;
}

}  // namespace

void refresh_per_material_derived_cell_fields(core::State& state,
                                              const core::Config& cfg,
                                              const HydroEOSContext* eos_ctx,
                                              const bool force_invalidate_all) {
  if (!cfg.numerics.materials.per_material_conservation_enabled) {
    return;
  }

  const std::size_t n_cells_size = state.rho.size();
  const std::size_t n_mat_size = cfg.materials.materials.size();
  if (n_cells_size == 0 || n_mat_size == 0) {
    return;
  }
  const std::size_t n_cell_mat = n_cells_size * n_mat_size;
  const int n_cells = checked_int(n_cells_size, "per-material EOS n_cells overflow");
  const int n_mat = checked_int(n_mat_size, "per-material EOS n_mat overflow");

  bool any_table_backed = false;
  for (const auto& mat : cfg.materials.materials) {
    any_table_backed = any_table_backed || material_requires_eos_context(mat);
  }
  if (any_table_backed) {
    TENRYU_ASSERT(eos_ctx != nullptr,
                  "per-material EOS requires HydroEOSContext for table-backed materials");
    TENRYU_ASSERT(eos_ctx->n_materials >= n_mat,
                  "per-material EOS context material count is smaller than Config material count");
  }

  TENRYU_ASSERT(state.mass_per_material.size() == n_cell_mat,
                "per-material EOS refresh requires mass_per_material size n_cells*n_materials");
  TENRYU_ASSERT(state.Ee_per_material.size() == n_cell_mat,
                "per-material EOS refresh requires Ee_per_material size n_cells*n_materials");
  TENRYU_ASSERT(state.Ei_per_material.size() == n_cell_mat,
                "per-material EOS refresh requires Ei_per_material size n_cells*n_materials");
  TENRYU_ASSERT(state.volFrac.size() == n_cell_mat,
                "per-material EOS refresh requires volFrac size n_cells*n_materials");
  TENRYU_ASSERT(state.vol.size() == n_cells_size,
                "per-material EOS refresh requires vol size n_cells");
  TENRYU_ASSERT(state.mass.size() == n_cells_size,
                "per-material EOS refresh requires mass size n_cells");

  if (state.cv_e.size() != n_cells_size) {
    state.cv_e.reset(n_cells_size);
  }
  if (state.cv_i.size() != n_cells_size) {
    state.cv_i.reset(n_cells_size);
  }
  if (state.cs.size() != n_cells_size) {
    state.cs.reset(n_cells_size);
  }

  const bool lazy_cache_enabled = cfg.numerics.materials.lazy_cache_te_m_enabled;
  if (lazy_cache_enabled) {
    TENRYU_ASSERT(state.Te_per_material.size() == n_cell_mat,
                  "per-material EOS lazy cache requires Te_per_material size n_cells*n_materials");
    TENRYU_ASSERT(state.Ti_per_material.size() == n_cell_mat,
                  "per-material EOS lazy cache requires Ti_per_material size n_cells*n_materials");
    TENRYU_ASSERT(state.Te_per_material_valid.size() == n_cell_mat,
                  "per-material EOS lazy cache requires Te valid flags size n_cells*n_materials");
    TENRYU_ASSERT(state.Ti_per_material_valid.size() == n_cell_mat,
                  "per-material EOS lazy cache requires Ti valid flags size n_cells*n_materials");
    if (force_invalidate_all) {
      invalidate_per_material_caches_all(state);
    }
  }
  const bool cache_active_this_refresh = lazy_cache_enabled && !force_invalidate_all;

  const bool profile_refresh = eos_bisection_profile_enabled();
  cudaEvent_t profile_start = nullptr;
  cudaEvent_t profile_stop = nullptr;
  if (profile_refresh) {
    cuda_check(cudaEventCreate(&profile_start),
               "per-material EOS profile cudaEventCreate start failed");
    cuda_check(cudaEventCreate(&profile_stop),
               "per-material EOS profile cudaEventCreate stop failed");
    cuda_check(cudaEventRecord(profile_start),
               "per-material EOS profile cudaEventRecord start failed");
  }

  const std::size_t cell_bytes = n_cells_size * sizeof(double);
  cuda_check(cudaMemset(state.Te.data(), 0, cell_bytes),
             "per-material EOS cudaMemset Te failed");
  cuda_check(cudaMemset(state.Ti.data(), 0, cell_bytes),
             "per-material EOS cudaMemset Ti failed");
  cuda_check(cudaMemset(state.Pe.data(), 0, cell_bytes),
             "per-material EOS cudaMemset Pe failed");
  cuda_check(cudaMemset(state.Pi.data(), 0, cell_bytes),
             "per-material EOS cudaMemset Pi failed");
  cuda_check(cudaMemset(state.ee.data(), 0, cell_bytes),
             "per-material EOS cudaMemset ee failed");
  cuda_check(cudaMemset(state.ei.data(), 0, cell_bytes),
             "per-material EOS cudaMemset ei failed");
  cuda_check(cudaMemset(state.cs.data(), 0, cell_bytes),
             "per-material EOS cudaMemset cs failed");
  cuda_check(cudaMemset(state.zbar.data(), 0, cell_bytes),
             "per-material EOS cudaMemset zbar failed");
  cuda_check(cudaMemset(state.cv_e.data(), 0, cell_bytes),
             "per-material EOS cudaMemset cv_e failed");
  cuda_check(cudaMemset(state.cv_i.data(), 0, cell_bytes),
             "per-material EOS cudaMemset cv_i failed");

  unsigned long long* d_counts = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_counts),
                        kCounterCount * sizeof(unsigned long long)),
             "per-material EOS cudaMalloc d_counts failed");
  cuda_check(cudaMemset(d_counts, 0, kCounterCount * sizeof(unsigned long long)),
             "per-material EOS cudaMemset d_counts failed");

  std::uint8_t* d_te_valid = nullptr;
  std::uint8_t* d_ti_valid = nullptr;
  if (cache_active_this_refresh) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_te_valid),
                          n_cell_mat * sizeof(std::uint8_t)),
               "per-material EOS cudaMalloc Te valid mirror failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ti_valid),
                          n_cell_mat * sizeof(std::uint8_t)),
               "per-material EOS cudaMalloc Ti valid mirror failed");
    cuda_check(cudaMemcpy(d_te_valid,
                          state.Te_per_material_valid.data(),
                          n_cell_mat * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "per-material EOS cudaMemcpy Te valid H2D failed");
    cuda_check(cudaMemcpy(d_ti_valid,
                          state.Ti_per_material_valid.data(),
                          n_cell_mat * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "per-material EOS cudaMemcpy Ti valid H2D failed");
  }

  const std::vector<MaterialEOSParams> h_params = make_material_params(cfg);
  MaterialEOSParams* d_params = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_params),
                        h_params.size() * sizeof(MaterialEOSParams)),
             "per-material EOS cudaMalloc material params failed");
  cuda_check(cudaMemcpy(d_params,
                        h_params.data(),
                        h_params.size() * sizeof(MaterialEOSParams),
                        cudaMemcpyHostToDevice),
             "per-material EOS cudaMemcpy material params failed");

  PerMaterialAccessorView view{};
  view.mass_per_material = state.mass_per_material.data();
  view.Ee_per_material = state.Ee_per_material.data();
  view.Ei_per_material = state.Ei_per_material.data();
  view.volfrac = state.volFrac.data();
  view.vol = state.vol.data();
  view.Te_per_material = cache_active_this_refresh ? state.Te_per_material.data() : nullptr;
  view.Ti_per_material = cache_active_this_refresh ? state.Ti_per_material.data() : nullptr;
  view.Te_per_material_valid = d_te_valid;
  view.Ti_per_material_valid = d_ti_valid;
  view.lazy_cache_te_m_enabled = cache_active_this_refresh;
  view.presence_threshold_volfrac = cfg.numerics.materials.presence_threshold_volfrac;
  view.presence_threshold_mass_density_g_per_cc =
      cfg.numerics.materials.presence_threshold_mass_density_g_per_cc;
  view.d_counts = d_counts;
  view.n_cells = n_cells;
  view.n_mat = n_mat;

  const auto* d_electron_views =
      (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_electron_views
                                                            : nullptr;
  const auto* d_ion_views =
      (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_ion_views
                                                            : nullptr;
  const int blocks_cm = (n_cells * n_mat + 255) / 256;
  project_per_material_thermo_kernel<<<blocks_cm, 256>>>(view,
                                                         d_electron_views,
                                                         d_ion_views,
                                                         d_params,
                                                         state.Te.data(),
                                                         state.Ti.data(),
                                                         state.Pe.data(),
                                                         state.Pi.data(),
                                                         state.cs.data(),
                                                         state.zbar.data(),
                                                         state.cv_e.data(),
                                                         state.cv_i.data(),
                                                         cfg.numerics.floors.Te,
                                                         cfg.numerics.floors.Ti,
                                                         cfg.materials.low_density_extrapolation);
  cuda_check(cudaGetLastError(), "per-material EOS projection kernel launch failed");

  const int blocks_cells = (n_cells + 255) / 256;
  reduce_per_material_specific_energy_kernel<<<blocks_cells, 256>>>(
      state.ee.data(),
      state.ei.data(),
      state.Ee_per_material.data(),
      state.Ei_per_material.data(),
      n_cells,
      n_mat);
  cuda_check(cudaGetLastError(), "per-material EOS energy reduction kernel launch failed");
  normalize_per_material_projection_kernel<<<blocks_cells, 256>>>(state.Te.data(),
                                                                  state.Ti.data(),
                                                                  state.ee.data(),
                                                                  state.ei.data(),
                                                                  state.zbar.data(),
                                                                  state.cv_e.data(),
                                                                  state.cv_i.data(),
                                                                  state.mass.data(),
                                                                  n_cells);
  cuda_check(cudaGetLastError(), "per-material EOS normalize kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "per-material EOS refresh synchronize failed");

  if (cache_active_this_refresh) {
    cuda_check(cudaMemcpy(state.Te_per_material_valid.data(),
                          d_te_valid,
                          n_cell_mat * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "per-material EOS cudaMemcpy Te valid D2H failed");
    cuda_check(cudaMemcpy(state.Ti_per_material_valid.data(),
                          d_ti_valid,
                          n_cell_mat * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "per-material EOS cudaMemcpy Ti valid D2H failed");
  }

  unsigned long long h_counts[kCounterCount] = {};
  cuda_check(cudaMemcpy(h_counts,
                        d_counts,
                        kCounterCount * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost),
             "per-material EOS cudaMemcpy counts failed");
  state.dispatch_counters.eos_inverse_call_count.fetch_add(
      static_cast<std::uint64_t>(h_counts[kCounterEOSInverse]), std::memory_order_relaxed);
  state.dispatch_counters.lazy_cache_te_m_hit_count.fetch_add(
      static_cast<std::uint64_t>(h_counts[kCounterLazyCacheHit]), std::memory_order_relaxed);
  state.dispatch_counters.lazy_cache_te_m_miss_count.fetch_add(
      static_cast<std::uint64_t>(h_counts[kCounterLazyCacheMiss]), std::memory_order_relaxed);
  state.dispatch_counters.mixture_projection_call_count.fetch_add(
      static_cast<std::uint64_t>(h_counts[kCounterMixtureProjection]),
      std::memory_order_relaxed);
  state.dispatch_counters.eos_table_validity_violations.fetch_add(
      static_cast<std::uint64_t>(h_counts[kCounterEOSTableValidityViolation]),
      std::memory_order_relaxed);
  state.dispatch_counters.presence_absent_events.fetch_add(
      static_cast<std::uint64_t>(h_counts[kCounterPresenceAbsent]),
      std::memory_order_relaxed);
  state.invalidate_cell_material_props();

  if (profile_refresh) {
    cuda_check(cudaEventRecord(profile_stop),
               "per-material EOS profile cudaEventRecord stop failed");
    cuda_check(cudaEventSynchronize(profile_stop),
               "per-material EOS profile cudaEventSynchronize stop failed");
    float elapsed_ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&elapsed_ms, profile_start, profile_stop),
               "per-material EOS profile cudaEventElapsedTime failed");
    core::log_info("[per_material_eos_refresh_timing] step=" +
                   std::to_string(state.step) +
                   " n_cells=" + std::to_string(n_cells) +
                   " n_mat=" + std::to_string(n_mat) +
                   " force_invalidate_all=" +
                   std::to_string(force_invalidate_all ? 1 : 0) +
                   " lazy_cache_active=" +
                   std::to_string(cache_active_this_refresh ? 1 : 0) +
                   " eos_inverse_calls=" +
                   std::to_string(h_counts[kCounterEOSInverse]) +
                   " lazy_cache_hits=" +
                   std::to_string(h_counts[kCounterLazyCacheHit]) +
                   " lazy_cache_misses=" +
                   std::to_string(h_counts[kCounterLazyCacheMiss]) +
                   " elapsed_ms=" + std::to_string(static_cast<double>(elapsed_ms)));
    cuda_check(cudaEventDestroy(profile_stop),
               "per-material EOS profile cudaEventDestroy stop failed");
    cuda_check(cudaEventDestroy(profile_start),
               "per-material EOS profile cudaEventDestroy start failed");
  }

  if (d_params != nullptr) {
    cuda_check(cudaFree(d_params), "per-material EOS cudaFree material params failed");
  }
  if (d_ti_valid != nullptr) {
    cuda_check(cudaFree(d_ti_valid), "per-material EOS cudaFree Ti valid failed");
  }
  if (d_te_valid != nullptr) {
    cuda_check(cudaFree(d_te_valid), "per-material EOS cudaFree Te valid failed");
  }
  cuda_check(cudaFree(d_counts), "per-material EOS cudaFree d_counts failed");
}

void invalidate_per_material_caches(core::State& state, const int c, const int m) {
  if (state.Te_per_material_valid.empty() && state.Ti_per_material_valid.empty()) {
    return;
  }
  state.dispatch_counters.lazy_cache_te_m_invalidation_count.fetch_add(
      1, std::memory_order_relaxed);
  TENRYU_ASSERT(state.rho.size() > 0, "per-material cache invalidation requires cells");
  TENRYU_ASSERT(state.volFrac.size() % state.rho.size() == 0,
                "per-material cache invalidation cannot infer n_mat");
  const std::size_t n_mat = state.volFrac.size() / state.rho.size();
  TENRYU_ASSERT(c >= 0 && static_cast<std::size_t>(c) < state.rho.size(),
                "per-material cache invalidation cell index out of range");
  TENRYU_ASSERT(m >= 0 && static_cast<std::size_t>(m) < n_mat,
                "per-material cache invalidation material index out of range");
  const std::size_t idx = static_cast<std::size_t>(c) * n_mat + static_cast<std::size_t>(m);
  if (!state.Te_per_material_valid.empty()) {
    TENRYU_ASSERT(idx < state.Te_per_material_valid.size(),
                  "per-material Te cache invalidation flag index out of range");
    state.Te_per_material_valid[idx] = 0u;
  }
  if (!state.Ti_per_material_valid.empty()) {
    TENRYU_ASSERT(idx < state.Ti_per_material_valid.size(),
                  "per-material Ti cache invalidation flag index out of range");
    state.Ti_per_material_valid[idx] = 0u;
  }
}

void invalidate_per_material_caches_all(core::State& state) {
  if (!state.Te_per_material_valid.empty() || !state.Ti_per_material_valid.empty()) {
    state.dispatch_counters.lazy_cache_te_m_invalidation_count.fetch_add(
        1, std::memory_order_relaxed);
  }
  std::fill(state.Te_per_material_valid.begin(), state.Te_per_material_valid.end(), 0u);
  std::fill(state.Ti_per_material_valid.begin(), state.Ti_per_material_valid.end(), 0u);
}

}  // namespace tenryu::hydro::per_material
