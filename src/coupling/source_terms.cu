#include "coupling/source_terms.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "hydro/eos_context.hpp"
#include "hydro/per_material_eos_accessors.cuh"
#include "hydro/per_material_eos_project.cuh"
#include "materials/eos_device.cuh"
#include "materials/eos_device_table.hpp"
#include "materials/eos_table.hpp"
#include "mesh/cell_geometry_2d.cuh"

namespace tenryu::coupling {
namespace {

constexpr double kMinEffectiveA = 1.0e-12;
constexpr double kMinEffectiveGamma = 1.0 + 1.0e-12;
constexpr double active_W_for_smoothing = 0.5;
constexpr std::int8_t kTransportModeDiffusion = 3;

// Packed staging for inject_laser_source_terms: one D2H gather and one
// H2D scatter per call instead of ~17 blocking per-field copies
// (2026-07-31 perf lane A; values and host math bit-identical).
struct LaserInjectStaging {
  double* d_pack = nullptr;          // device pack [n_slots * n]
  std::vector<double> h_pack;        // host mirror
  std::size_t capacity_cells = 0;
};
LaserInjectStaging& laser_inject_staging() {
  static LaserInjectStaging s;
  return s;
}

__global__ void pack_fields_kernel(double* __restrict__ pack,
                                   const double* const* __restrict__ srcs,
                                   const int n_fields,
                                   const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_fields * n) return;
  const int f = i / n;
  pack[i] = srcs[f][i - f * n];
}

__global__ void unpack_fields_kernel(double* const* __restrict__ dsts,
                                     const double* __restrict__ pack,
                                     const int n_fields,
                                     const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_fields * n) return;
  const int f = i / n;
  dsts[f][i - f * n] = pack[i];
}

inline bool use_exact_ideal_gas_hydro_backend(
    const core::Config::MaterialsConfig::MatDef& mat) {
  return mat.hydro_eos_backend == "exact_ideal_gas";
}

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ inline double atomic_add_double(double* address, const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

void accumulate_floor_and_clamp(double* E_floor_injected,
                                int* clamp_count,
                                const double local_floor,
                                const int local_clamp) {
  if (E_floor_injected != nullptr) {
    *E_floor_injected += std::max(local_floor, 0.0);
  }
  if (clamp_count != nullptr) {
    *clamp_count += std::max(local_clamp, 0);
  }
}

struct SourceEOSTableViews {
  materials::DeviceEOSTableView ion{};
  materials::DeviceEOSTableView electron{};
};

struct SourceMaterialParams {
  double A = 1.0;
  double Zbar = 1.0;
  double gamma = 5.0 / 3.0;
};

std::vector<SourceMaterialParams> make_source_material_params(
    const core::Config& cfg) {
  std::vector<SourceMaterialParams> params(cfg.materials.materials.size());
  for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
    const auto& mat = cfg.materials.materials[m];
    SourceMaterialParams p{};
    p.A = std::max(mat.A, kMinEffectiveA);
    p.gamma = std::max(mat.ideal_gas_gamma, kMinEffectiveGamma);
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

class SourceEOSTableCache {
 public:
  SourceEOSTableViews views_for(const materials::EOSTableTriplet& tables) {
    if (tables_ != &tables) {
      ion_.upload(tables.ion);
      electron_.upload(tables.electron);
      tables_ = &tables;
    }
    SourceEOSTableViews views;
    views.ion = ion_.view();
    views.electron = electron_.view();
    return views;
  }

 private:
  const materials::EOSTableTriplet* tables_ = nullptr;
  materials::DeviceEOSTable ion_;
  materials::DeviceEOSTable electron_;
};

SourceEOSTableCache& source_eos_table_cache() {
  static SourceEOSTableCache cache;
  return cache;
}

SourceEOSTableViews select_source_eos_table_views(
    const core::Config::MaterialsConfig::MatDef& mat,
    const int material_index,
    const hydro::HydroEOSContext* eos_ctx) {
  SourceEOSTableViews views;
  if (mat.eos_tables == nullptr) {
    return views;
  }

  if (eos_ctx != nullptr && material_index >= 0 &&
      material_index < eos_ctx->n_materials) {
    views.ion = eos_ctx->ion_view(material_index);
    views.electron = eos_ctx->electron_view(material_index);
    if (views.ion.n_rho > 0 && views.electron.n_rho > 0) {
      return views;
    }
  }

  return source_eos_table_cache().views_for(*mat.eos_tables);
}

void assert_common_source_state_sizes(const core::State& state,
                                      const char* caller) {
  const std::size_t n_cells = state.rho.size();
  TENRYU_ASSERT(state.mass.size() == n_cells,
                std::string(caller) + " requires mass/rho size match");
  TENRYU_ASSERT(state.zbar.size() == n_cells,
                std::string(caller) + " requires zbar/rho size match");
  TENRYU_ASSERT(state.vol.size() == n_cells,
                std::string(caller) + " requires vol/rho size match");
  TENRYU_ASSERT(state.ee.size() == n_cells,
                std::string(caller) + " requires ee/rho size match");
  TENRYU_ASSERT(state.Te.size() == n_cells,
                std::string(caller) + " requires Te/rho size match");
  TENRYU_ASSERT(state.ei.size() == n_cells,
                std::string(caller) + " requires ei/rho size match");
  TENRYU_ASSERT(state.Ti.size() == n_cells,
                std::string(caller) + " requires Ti/rho size match");
  TENRYU_ASSERT(state.Pe.size() == n_cells,
                std::string(caller) + " requires Pe/rho size match");
  TENRYU_ASSERT(state.Pi.size() == n_cells,
                std::string(caller) + " requires Pi/rho size match");
  TENRYU_ASSERT(state.cell_is_void.size() == n_cells,
                std::string(caller) + " requires cell_is_void/rho size match");
}

void compute_effective_A_gamma(const core::Config& cfg,
                               const core::State& state,
                               const int n_cells,
                               std::vector<double>& A_eff,
                               std::vector<double>& gamma_eff,
                               std::vector<int>* dominant_material = nullptr) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "compute_effective_A_gamma requires at least one material");

  const auto& materials = cfg.materials.materials;
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "compute_effective_A_gamma requires at least one non-void material");
  const auto& mat0 = materials[static_cast<std::size_t>(first_nonvoid)];
  const double A0 = std::max(mat0.A, kMinEffectiveA);
  const double gamma0 = std::max(mat0.ideal_gas_gamma, kMinEffectiveGamma);
  A_eff.assign(static_cast<std::size_t>(std::max(n_cells, 0)), A0);
  gamma_eff.assign(static_cast<std::size_t>(std::max(n_cells, 0)), gamma0);
  if (dominant_material != nullptr) {
    dominant_material->assign(static_cast<std::size_t>(std::max(n_cells, 0)),
                              first_nonvoid);
  }
  if (n_cells <= 0) {
    return;
  }

  const int n_mat = static_cast<int>(materials.size());
  if (n_mat <= 1) {
    return;
  }

  const std::size_t expected =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  if (state.volFrac.size() != expected) {
    static bool warned_volfrac_size_mismatch = false;
    if (!warned_volfrac_size_mismatch) {
      core::log_warning("Multi-material mixing: volFrac size mismatch (" +
                        std::to_string(state.volFrac.size()) + " vs expected " +
                        std::to_string(expected) +
                        "); falling back to first non-void material.");
      warned_volfrac_size_mismatch = true;
    }
    return;
  }

  std::vector<double> volfrac(expected, 0.0);
  state.volFrac.copy_to_host(volfrac.data());

  std::vector<double> inv_A_m(static_cast<std::size_t>(n_mat), 0.0);
  std::vector<double> gamma_m(static_cast<std::size_t>(n_mat), gamma0);
  for (int m = 0; m < n_mat; ++m) {
    const auto& mat = materials[static_cast<std::size_t>(m)];
    const double A_m = std::max(mat.A, kMinEffectiveA);
    inv_A_m[static_cast<std::size_t>(m)] = 1.0 / A_m;
    gamma_m[static_cast<std::size_t>(m)] =
        std::max(mat.ideal_gas_gamma, kMinEffectiveGamma);
  }

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t base = static_cast<std::size_t>(c) * static_cast<std::size_t>(n_mat);
    double frac_sum = 0.0;
    double inv_A_c = 0.0;
    double gamma_c = 0.0;
    double best_frac = -1.0;
    int best_mat = first_nonvoid;
    for (int m = 0; m < n_mat; ++m) {
      const std::size_t m_idx = static_cast<std::size_t>(m);
      if (materials[m_idx].is_void) {
        continue;
      }
      const double frac_raw = volfrac[base + m_idx];
      const double frac =
          (std::isfinite(frac_raw) && frac_raw > 0.0) ? frac_raw : 0.0;
      frac_sum += frac;
      inv_A_c += frac * inv_A_m[m_idx];
      gamma_c += frac * gamma_m[m_idx];
      if (frac > best_frac) {
        best_frac = frac;
        best_mat = m;
      }
    }
    if (frac_sum > 1.0e-30) {
      inv_A_c /= frac_sum;
      gamma_c /= frac_sum;
    }
    if (std::isfinite(inv_A_c) && inv_A_c > 1.0e-30) {
      A_eff[static_cast<std::size_t>(c)] = std::max(1.0 / inv_A_c, kMinEffectiveA);
    }
    if (std::isfinite(gamma_c) && gamma_c > 0.0) {
      gamma_eff[static_cast<std::size_t>(c)] =
          std::max(gamma_c, kMinEffectiveGamma);
    }
    if (dominant_material != nullptr) {
      (*dominant_material)[static_cast<std::size_t>(c)] = best_mat;
    }
  }
}

void compute_raw_net_radiation_source_terms(const std::vector<double>& rad_dep,
                                            const std::vector<double>& rad_emit,
                                            const int n_cells,
                                            const int n_groups,
                                            std::vector<double>& raw_delta_E) {
  raw_delta_E.assign(static_cast<std::size_t>(std::max(n_cells, 0)), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cell_base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
    double dep_sum = 0.0;
    double emit_sum = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = cell_base + static_cast<std::size_t>(g);
      dep_sum += rad_dep[idx];
      emit_sum += rad_emit[idx];
    }
    raw_delta_E[static_cast<std::size_t>(c)] = dep_sum - emit_sum;
  }
}

struct NetESourceSmoothingDiagnostics {
  int faces_total = 0;
  int faces_active = 0;
  double alpha_mean = 0.0;
  double alpha_min = 0.0;
};

NetESourceSmoothingDiagnostics compute_gradient_adaptive_smoothing_diagnostics(
    const core::Config::RadiationConfig::ImcConfig::NetElectronSourceSmoothingConfig&
        smoothing_cfg,
    const std::vector<double>& mass,
    const std::vector<double>& node_r,
    const std::vector<double>& sigma_R_max,
    const std::vector<int>& dominant_material,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& Te,
    const std::vector<double>& rho) {
  constexpr double kMassEps = 1.0e-30;
  constexpr double kLogFloor = 1.0e-30;
  const int n_cells = static_cast<int>(mass.size());
  NetESourceSmoothingDiagnostics diag;
  diag.faces_total = std::max(n_cells - 1, 0);
  for (int left = 0; left + 1 < n_cells; ++left) {
    const int right = left + 1;
    if (cell_is_void[static_cast<std::size_t>(left)] != static_cast<std::uint8_t>(0) ||
        cell_is_void[static_cast<std::size_t>(right)] != static_cast<std::uint8_t>(0)) {
      continue;
    }
    if (dominant_material[static_cast<std::size_t>(left)] !=
        dominant_material[static_cast<std::size_t>(right)]) {
      continue;
    }
    const double mass_left = mass[static_cast<std::size_t>(left)];
    const double mass_right = mass[static_cast<std::size_t>(right)];
    if (!(mass_left > kMassEps) || !(mass_right > kMassEps)) {
      continue;
    }

    const double dx_left =
        std::fmax(node_r[static_cast<std::size_t>(left + 1)] -
                     node_r[static_cast<std::size_t>(left)],
                 0.0);
    const double dx_right =
        std::fmax(node_r[static_cast<std::size_t>(right + 1)] -
                     node_r[static_cast<std::size_t>(right)],
                 0.0);
    const double tau_left =
        std::fmax(sigma_R_max[static_cast<std::size_t>(left)], 0.0) * dx_left;
    const double tau_right =
        std::fmax(sigma_R_max[static_cast<std::size_t>(right)], 0.0) * dx_right;
    if (std::fmin(tau_left, tau_right) < smoothing_cfg.tau_threshold) {
      continue;
    }

    const double dln_Te = std::fabs(std::log(
        std::fmax(Te[static_cast<std::size_t>(right)], kLogFloor) /
        std::fmax(Te[static_cast<std::size_t>(left)], kLogFloor)));
    const double dln_rho = std::fabs(std::log(
        std::fmax(rho[static_cast<std::size_t>(right)], kLogFloor) /
        std::fmax(rho[static_cast<std::size_t>(left)], kLogFloor)));
    const double Te_arg = dln_Te / smoothing_cfg.grad_Te_scale;
    const double rho_arg = dln_rho / smoothing_cfg.grad_rho_scale;
    const double alpha_face =
        smoothing_cfg.alpha * std::exp(-(Te_arg * Te_arg)) *
        std::exp(-(rho_arg * rho_arg));
    diag.alpha_mean += alpha_face;
    if (diag.faces_active == 0 || alpha_face < diag.alpha_min) {
      diag.alpha_min = alpha_face;
    }
    ++diag.faces_active;
  }
  if (diag.faces_active > 0) {
    diag.alpha_mean /= static_cast<double>(diag.faces_active);
  } else {
    diag.alpha_mean = 0.0;
    diag.alpha_min = 0.0;
  }
  return diag;
}

__device__ __forceinline__ double smooth_face_flux_1d_device(
    const double* __restrict__ H_raw,
    const double* __restrict__ mass,
    const double* __restrict__ node_r,
    const double* __restrict__ sigma_R_max,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const int* __restrict__ dominant_material,
    const std::uint8_t* __restrict__ cell_is_void,
    const int left,
    const int right,
    const int n_cells,
    const double alpha,
    const double tau_threshold,
    const double grad_Te_scale,
    const double grad_rho_scale,
    const bool gradient_adaptive) {
  constexpr double kMassEps = 1.0e-30;
  constexpr double kLogFloor = 1.0e-30;
  if (left < 0 || right >= n_cells) {
    return 0.0;
  }
  if (cell_is_void[left] != static_cast<std::uint8_t>(0) ||
      cell_is_void[right] != static_cast<std::uint8_t>(0)) {
    return 0.0;
  }
  if (dominant_material[left] != dominant_material[right]) {
    return 0.0;
  }

  const double mass_left = mass[left];
  const double mass_right = mass[right];
  if (!(mass_left > kMassEps) || !(mass_right > kMassEps)) {
    return 0.0;
  }

  const double dx_left = fmax(node_r[left + 1] - node_r[left], 0.0);
  const double dx_right = fmax(node_r[right + 1] - node_r[right], 0.0);
  const double tau_left = fmax(sigma_R_max[left], 0.0) * dx_left;
  const double tau_right = fmax(sigma_R_max[right], 0.0) * dx_right;
  if (fmin(tau_left, tau_right) < tau_threshold) {
    return 0.0;
  }

  double alpha_face = alpha;
  if (gradient_adaptive) {
    const double dln_Te =
        fabs(log(fmax(Te[right], kLogFloor) / fmax(Te[left], kLogFloor)));
    const double dln_rho =
        fabs(log(fmax(rho[right], kLogFloor) / fmax(rho[left], kLogFloor)));
    const double Te_arg = dln_Te / grad_Te_scale;
    const double rho_arg = dln_rho / grad_rho_scale;
    alpha_face = alpha * exp(-(Te_arg * Te_arg)) * exp(-(rho_arg * rho_arg));
  }

  const double m_face =
      2.0 * mass_left * mass_right / fmax(mass_left + mass_right, kMassEps);
  const double e_left = H_raw[left] / mass_left;
  const double e_right = H_raw[right] / mass_right;
  return alpha_face * m_face * (e_left - e_right);
}

__device__ __forceinline__ double smooth_face_flux_2d_device(
    const double* __restrict__ H_raw,
    const double* __restrict__ mass,
    const double* __restrict__ sigma_R_max,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const int* __restrict__ dominant_material,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ difference_W,
    const int left,
    const int right,
    const int n_cells,
    const double alpha,
    const double tau_threshold,
    const double grad_Te_scale,
    const double grad_rho_scale,
    const bool gradient_adaptive,
    const double tau_left,
    const double tau_right) {
  (void)sigma_R_max;
  constexpr double kDifferenceWBarrier = 0.5;
  constexpr double kMassEps = 1.0e-30;
  constexpr double kLogFloor = 1.0e-30;
  if (left < 0 || right >= n_cells) {
    return 0.0;
  }
  if (cell_is_void[left] != static_cast<std::uint8_t>(0) ||
      cell_is_void[right] != static_cast<std::uint8_t>(0)) {
    return 0.0;
  }
  if (difference_W != nullptr &&
      (!(difference_W[left] < kDifferenceWBarrier) ||
       !(difference_W[right] < kDifferenceWBarrier))) {
    return 0.0;
  }
  if (dominant_material[left] != dominant_material[right]) {
    return 0.0;
  }

  const double mass_left = mass[left];
  const double mass_right = mass[right];
  if (!(mass_left > kMassEps) || !(mass_right > kMassEps)) {
    return 0.0;
  }

  if (fmin(tau_left, tau_right) < tau_threshold) {
    return 0.0;
  }

  double alpha_face = alpha;
  if (gradient_adaptive) {
    const double dln_Te =
        fabs(log(fmax(Te[right], kLogFloor) / fmax(Te[left], kLogFloor)));
    const double dln_rho =
        fabs(log(fmax(rho[right], kLogFloor) / fmax(rho[left], kLogFloor)));
    const double Te_arg = dln_Te / grad_Te_scale;
    const double rho_arg = dln_rho / grad_rho_scale;
    alpha_face = alpha * exp(-(Te_arg * Te_arg)) * exp(-(rho_arg * rho_arg));
  }

  const double m_face =
      2.0 * mass_left * mass_right / fmax(mass_left + mass_right, kMassEps);
  const double e_left = H_raw[left] / mass_left;
  const double e_right = H_raw[right] / mass_right;
  return alpha_face * m_face * (e_left - e_right);
}

__global__ void smooth_net_electron_source_terms_1d_kernel(
    double* __restrict__ H_apply,
    const double* __restrict__ H_raw,
    const double* __restrict__ mass,
    const double* __restrict__ node_r,
    const double* __restrict__ sigma_R_max,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const int* __restrict__ dominant_material,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_cells,
    const double alpha,
    const double tau_threshold,
    const double grad_Te_scale,
    const double grad_rho_scale,
    const bool gradient_adaptive) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double F_left =
      (c > 0) ? smooth_face_flux_1d_device(H_raw, mass, node_r, sigma_R_max,
                                           Te, rho, dominant_material, cell_is_void,
                                           c - 1, c, n_cells, alpha,
                                           tau_threshold, grad_Te_scale,
                                           grad_rho_scale, gradient_adaptive)
              : 0.0;
  const double F_right =
      (c + 1 < n_cells)
          ? smooth_face_flux_1d_device(H_raw, mass, node_r, sigma_R_max,
                                       Te, rho, dominant_material, cell_is_void,
                                       c, c + 1, n_cells, alpha, tau_threshold,
                                       grad_Te_scale, grad_rho_scale,
                                       gradient_adaptive)
          : 0.0;
  H_apply[c] = H_raw[c] + F_left - F_right;
}

__global__ void conservative_smooth_delta_E_1d_kernel(
    double* __restrict__ H_out,
    const double* __restrict__ H_in,
    const double* __restrict__ mass,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_cells,
    const double alpha) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (cell_is_void[c] != 0u) {
    H_out[c] = H_in[c];
    return;
  }

  constexpr double kMassEps = 1.0e-30;
  const double mass_c = fmax(mass[c], kMassEps);
  const double e_c = H_in[c] / mass_c;

  double F_left = 0.0;
  if (c > 0 && cell_is_void[c - 1] == 0u) {
    const double mass_l = fmax(mass[c - 1], kMassEps);
    const double m_face = 2.0 * mass_l * mass_c / (mass_l + mass_c);
    F_left = alpha * m_face * (H_in[c - 1] / mass_l - e_c);
  }
  double F_right = 0.0;
  if (c + 1 < n_cells && cell_is_void[c + 1] == 0u) {
    const double mass_r = fmax(mass[c + 1], kMassEps);
    const double m_face = 2.0 * mass_c * mass_r / (mass_c + mass_r);
    F_right = alpha * m_face * (e_c - H_in[c + 1] / mass_r);
  }
  H_out[c] = H_in[c] + F_left - F_right;
}

__global__ void smooth_net_electron_source_terms_2d_kernel(
    double* __restrict__ H_apply,
    const double* __restrict__ H_raw,
    const double* __restrict__ mass,
    const double* __restrict__ vol,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ sigma_R_max,
    const double* __restrict__ Te,
    const double* __restrict__ rho,
    const int* __restrict__ dominant_material,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ difference_W,
    const int nr,
    const int nz,
    const double alpha,
    const double tau_threshold,
    const double grad_Te_scale,
    const double grad_rho_scale,
    const bool gradient_adaptive) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const tenryu::mesh::CellWidths2D widths =
      tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz, c);
  const double sigma_this = fmax(sigma_R_max[c], 0.0);
  const double tau_R_this = sigma_this * fmax(widths.h_R, 0.0);
  const double tau_Z_this = sigma_this * fmax(widths.h_Z, 0.0);

  double flux_sum = 0.0;
  if (i > 0) {
    const int neighbor = c - nz;
    const tenryu::mesh::CellWidths2D neighbor_widths =
        tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz,
                                             neighbor);
    const double tau_R_neighbor =
        fmax(sigma_R_max[neighbor], 0.0) * fmax(neighbor_widths.h_R, 0.0);
    flux_sum += smooth_face_flux_2d_device(
        H_raw, mass, sigma_R_max, Te, rho, dominant_material, cell_is_void,
        difference_W, neighbor, c, n_cells, alpha, tau_threshold, grad_Te_scale,
        grad_rho_scale, gradient_adaptive, tau_R_neighbor, tau_R_this);
  }
  if (i + 1 < nr) {
    const int neighbor = c + nz;
    const tenryu::mesh::CellWidths2D neighbor_widths =
        tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz,
                                             neighbor);
    const double tau_R_neighbor =
        fmax(sigma_R_max[neighbor], 0.0) * fmax(neighbor_widths.h_R, 0.0);
    flux_sum -= smooth_face_flux_2d_device(
        H_raw, mass, sigma_R_max, Te, rho, dominant_material, cell_is_void,
        difference_W, c, neighbor, n_cells, alpha, tau_threshold, grad_Te_scale,
        grad_rho_scale, gradient_adaptive, tau_R_this, tau_R_neighbor);
  }
  if (j > 0) {
    const int neighbor = c - 1;
    const tenryu::mesh::CellWidths2D neighbor_widths =
        tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz,
                                             neighbor);
    const double tau_Z_neighbor =
        fmax(sigma_R_max[neighbor], 0.0) * fmax(neighbor_widths.h_Z, 0.0);
    flux_sum += smooth_face_flux_2d_device(
        H_raw, mass, sigma_R_max, Te, rho, dominant_material, cell_is_void,
        difference_W, neighbor, c, n_cells, alpha, tau_threshold, grad_Te_scale,
        grad_rho_scale, gradient_adaptive, tau_Z_neighbor, tau_Z_this);
  }
  if (j + 1 < nz) {
    const int neighbor = c + 1;
    const tenryu::mesh::CellWidths2D neighbor_widths =
        tenryu::mesh::compute_cell_widths_2d(node_r, node_z, vol, nr, nz,
                                             neighbor);
    const double tau_Z_neighbor =
        fmax(sigma_R_max[neighbor], 0.0) * fmax(neighbor_widths.h_Z, 0.0);
    flux_sum -= smooth_face_flux_2d_device(
        H_raw, mass, sigma_R_max, Te, rho, dominant_material, cell_is_void,
        difference_W, c, neighbor, n_cells, alpha, tau_threshold, grad_Te_scale,
        grad_rho_scale, gradient_adaptive, tau_Z_this, tau_Z_neighbor);
  }

  H_apply[c] = H_raw[c] + flux_sum;
}

__global__ void inject_radiation_source_terms_kernel(
    double* __restrict__ ee,
    double* __restrict__ Te,
    double* __restrict__ Pe,
    double* __restrict__ ei,
    double* __restrict__ Ti,
    double* __restrict__ Pi,
    double* __restrict__ delta_E_rad_prev,
    const double* __restrict__ rho,
    const double* __restrict__ mass,
    const double* __restrict__ zbar,
    const double* __restrict__ cv_e,
    const double* __restrict__ applied_delta_E,
    const double* __restrict__ A_eff,
    const double* __restrict__ gamma_eff,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::uint8_t* __restrict__ diffusion_cell,
    const std::uint8_t* __restrict__ holo_source_cell,
    const int n_cells,
    const double te_floor,
    const double ti_floor,
    const bool use_two_temp,
    const bool has_table_eos,
    const bool use_first_cv_override,
    const double cv_e_override,
    const double eos_T_ref_eV,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele,
    double* __restrict__ floor_energy,
    double* __restrict__ skipped_energy,
    int* __restrict__ clamp_count) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double delta_E = applied_delta_E[c];
  delta_E_rad_prev[c] = delta_E;
  if (holo_source_cell != nullptr &&
      holo_source_cell[c] != static_cast<std::uint8_t>(0)) {
    return;
  }
  if (diffusion_cell[c] != static_cast<std::uint8_t>(0)) {
    delta_E_rad_prev[c] = 0.0;
    return;
  }
  if (cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    atomic_add_double(skipped_energy, delta_E);
    return;
  }

  const double rho_c = rho[c];
  const double mass_c = mass[c];
  if (mass_c < 1.0e-30) {
    atomic_add_double(skipped_energy, delta_E);
    return;
  }

  const double A_c = A_eff[c];
  const double gm1_c = gamma_eff[c] - 1.0;
  const double cv_mass_i =
      fmax(tenryu::core::constants::eV_to_erg /
               (A_c * tenryu::core::constants::proton_mass * gm1_c),
           1.0e-30);
  if (!use_two_temp) {
    ee[c] += ei[c];
  }
  ee[c] += delta_E / mass_c;
  const double ee_before_floor = ee[c];

  const double rho_safe = fmax(rho_c, 1.0e-30);
  const bool use_table_eos_closure =
      use_two_temp && has_table_eos && tab_ele.n_rho > 0;
  double Te_raw = 0.0;
  double Te_new = te_floor;
  if (use_table_eos_closure) {
    const auto rb_e = tenryu::materials::find_rho_bracket(tab_ele, rho_safe);
    Te_raw = tenryu::materials::device_eos_T_from_e_monotone(
        tab_ele, rb_e, ee[c]);
    if (!isfinite(Te_raw) || Te_raw < te_floor) {
      Te_new = te_floor;
      const double logTe = log(fmax(Te_new, 1.0e-300));
      ee[c] = tenryu::materials::device_eos_energy(tab_ele, rb_e, logTe);
    } else {
      Te_new = Te_raw;
    }
    const double logTe = log(fmax(Te_new, 1.0e-300));
    Pe[c] = tenryu::materials::device_eos_pressure(tab_ele, rb_e, logTe);
  } else if (use_first_cv_override && eos_T_ref_eV > 0.0) {
    const double T_ref3 = eos_T_ref_eV * eos_T_ref_eV * eos_T_ref_eV;
    const double alpha0 = cv_e_override / (4.0 * T_ref3);
    const double arg = ee[c] * rho_safe / alpha0;
    Te_raw = (arg > 0.0) ? pow(arg, 0.25) : 0.0;
    if (!isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = fmax(Te_raw, te_floor);
    const double Te2 = Te_new * Te_new;
    ee[c] = alpha0 * Te2 * Te2 / rho_safe;
  } else {
    const double z = fmax(zbar[c], 0.0);
    double cv_mass_e = 0.0;
    if (use_first_cv_override) {
      cv_mass_e = cv_e_override / rho_safe;
    } else if (cv_e != nullptr && cv_e[c] > 0.0) {
      cv_mass_e = cv_e[c];
    } else {
      cv_mass_e = z * tenryu::core::constants::eV_to_erg /
                  (A_c * tenryu::core::constants::proton_mass * gm1_c);
    }
    cv_mass_e = fmax(cv_mass_e, 1.0e-30);
    const double cv_mass_total =
        use_two_temp ? cv_mass_e
                     : (use_first_cv_override
                            ? cv_mass_e
                            : fmax(cv_mass_e + cv_mass_i, 1.0e-30));
    Te_raw = ee[c] / cv_mass_total;
    if (!isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = fmax(Te_raw, te_floor);
    ee[c] = cv_mass_total * Te_new;
  }

  if (Te_new > Te_raw) {
    atomicAdd(clamp_count, 1);
  }
  const double de_floor_e = ee[c] - ee_before_floor;
  if (de_floor_e > 0.0) {
    atomic_add_double(floor_energy, mass_c * de_floor_e);
  }

  Te[c] = Te_new;
  if (!use_table_eos_closure) {
    Pe[c] = gm1_c * rho_c * ee[c];
  }
  if (use_two_temp) {
    if (has_table_eos && tab_ion.n_rho > 0) {
      const auto rb_i = tenryu::materials::find_rho_bracket(tab_ion, rho_safe);
      const double Ti_raw_table =
          tenryu::materials::device_eos_T_from_e_monotone(tab_ion, rb_i, ei[c]);
      const double ei_before_floor = ei[c];
      if (!isfinite(Ti_raw_table) || Ti_raw_table < ti_floor) {
        atomicAdd(clamp_count, 1);
        Ti[c] = ti_floor;
        const double logTi = log(fmax(ti_floor, 1.0e-300));
        ei[c] = tenryu::materials::device_eos_energy(tab_ion, rb_i, logTi);
      } else {
        Ti[c] = Ti_raw_table;
      }
      const double logTi = log(fmax(Ti[c], 1.0e-300));
      Pi[c] = tenryu::materials::device_eos_pressure(tab_ion, rb_i, logTi);
      const double de_floor_i = ei[c] - ei_before_floor;
      if (de_floor_i > 0.0) {
        atomic_add_double(floor_energy, mass_c * de_floor_i);
      }
    } else {
      const double Ti_prev = isfinite(Ti[c]) ? Ti[c] : 0.0;
      if (Ti_prev < ti_floor) {
        atomicAdd(clamp_count, 1);
        const double ei_before_floor = ei[c];
        Ti[c] = ti_floor;
        ei[c] = cv_mass_i * ti_floor;
        Pi[c] = gm1_c * rho_c * ei[c];
        const double de_floor_i = ei[c] - ei_before_floor;
        if (de_floor_i > 0.0) {
          atomic_add_double(floor_energy, mass_c * de_floor_i);
        }
      }
    }
  } else {
    Ti[c] = Te_new;
    ei[c] = 0.0;
    Pi[c] = 0.0;
  }
}

__global__ void qei_coupling_substep_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    double* __restrict__ cv_e,
    double* __restrict__ cv_i,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ A_eff,
    const double* __restrict__ gamma_eff,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double dt,
    const double te_floor,
    const double ti_floor,
    const bool has_table_eos,
    const bool use_first_cv_override,
    const double cv_e_override,
    const double eos_T_ref_eV,
    const double qei_multiplier,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || !(dt > 0.0)) {
    return;
  }
  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const double rho_c = rho[c];
  const double rho_safe = fmax(rho_c, 1.0e-30);
  const double A_c = fmax(A_eff[c], 1.0e-12);
  const double gamma_c = fmax(gamma_eff[c], 1.0 + 1.0e-12);
  const double gm1_c = fmax(gamma_c - 1.0, 1.0e-30);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 0.0;
  const double cv_mass_i_fallback =
      fmax(tenryu::core::constants::eV_to_erg /
               (A_c * tenryu::core::constants::proton_mass * gm1_c),
           1.0e-30);
  double cv_mass_e_qei = 0.0;
  if (cv_e != nullptr && cv_e[c] > 0.0) {
    cv_mass_e_qei = cv_e[c];
  } else if (use_first_cv_override) {
    cv_mass_e_qei = cv_e_override / rho_safe;
  } else {
    cv_mass_e_qei =
        z * tenryu::core::constants::eV_to_erg /
        (A_c * tenryu::core::constants::proton_mass * gm1_c);
  }
  cv_mass_e_qei = fmax(cv_mass_e_qei, 1.0e-30);
  const double cv_mass_i_qei =
      (cv_i != nullptr && cv_i[c] > 0.0) ? fmax(cv_i[c], 1.0e-30)
                                         : cv_mass_i_fallback;

  double qei_term = 0.0;
  if (cv_e != nullptr && cv_i != nullptr && cv_e[c] > 0.0 && cv_i[c] > 0.0) {
    qei_term = tenryu::materials::compute_qei_term_with_cv(
        fmax(rho_c, 0.0), fmax(Te[c], 0.0), fmax(Ti[c], 0.0), z, A_c,
        cv_mass_e_qei, cv_mass_i_qei, dt, qei_multiplier);
  } else {
    qei_term = tenryu::materials::compute_qei_term_analytical(
        fmax(rho_c, 0.0), fmax(Te[c], 0.0), fmax(Ti[c], 0.0), z, A_c,
        gamma_c, dt, qei_multiplier);
  }

  // AI review k15 1.4/P0-4 (2026-07-26): bracket the transfer into the
  // physically admissible interval instead of clamping each side
  // independently — the old independent fmax floors created or destroyed
  // pair energy whenever one side hit zero (reachable for table-EOS cells
  // where the frozen-cv transfer overshoots the stored energy). One shared
  // applied transfer keeps ee + ei exactly conserved and both sides
  // nonnegative; cells where the floors never engaged are bit-identical.
  const double qei_hi = fmax(ee[c], 0.0);   // most the electrons can give
  const double qei_lo = -fmax(ei[c], 0.0);  // most the ions can give
  const double qei_applied =
      isfinite(qei_term) ? fmin(fmax(qei_term, qei_lo), qei_hi) : 0.0;
  ee[c] -= qei_applied;
  ei[c] += qei_applied;

  const bool use_table_eos_closure =
      has_table_eos && tab_ele.n_rho > 0 && tab_ion.n_rho > 0;
  double Te_raw = 0.0;
  double Te_new = te_floor;
  if (use_table_eos_closure) {
    const auto rb_e = tenryu::materials::find_rho_bracket(tab_ele, rho_safe);
    Te_raw = tenryu::materials::device_eos_T_from_e_monotone(tab_ele, rb_e, ee[c]);
    if (!isfinite(Te_raw) || Te_raw < te_floor) {
      Te_new = te_floor;
      const double logTe = log(fmax(Te_new, 1.0e-300));
      ee[c] = tenryu::materials::device_eos_energy(tab_ele, rb_e, logTe);
    } else {
      Te_new = Te_raw;
    }
    const double logTe = log(fmax(Te_new, 1.0e-300));
    Pe[c] = tenryu::materials::device_eos_pressure(tab_ele, rb_e, logTe);
    if (cv_e != nullptr) {
      cv_e[c] = fmax(tenryu::materials::device_eos_cv(tab_ele, rb_e, logTe), 0.0);
    }
  } else if (use_first_cv_override && eos_T_ref_eV > 0.0) {
    const double T_ref3 = eos_T_ref_eV * eos_T_ref_eV * eos_T_ref_eV;
    const double alpha0 = cv_e_override / (4.0 * T_ref3);
    const double arg = ee[c] * rho_safe / alpha0;
    Te_raw = (arg > 0.0) ? pow(arg, 0.25) : 0.0;
    if (!isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = fmax(Te_raw, te_floor);
    const double Te2 = Te_new * Te_new;
    ee[c] = alpha0 * Te2 * Te2 / rho_safe;
    Pe[c] = gm1_c * rho_c * ee[c];
    if (cv_e != nullptr) {
      cv_e[c] = fmax(cv_mass_e_qei, 0.0);
    }
  } else {
    Te_raw = ee[c] / cv_mass_e_qei;
    if (!isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = fmax(Te_raw, te_floor);
    ee[c] = cv_mass_e_qei * Te_new;
    Pe[c] = gm1_c * rho_c * ee[c];
    if (cv_e != nullptr) {
      cv_e[c] = fmax(cv_mass_e_qei, 0.0);
    }
  }
  Te[c] = Te_new;

  if (use_table_eos_closure) {
    const auto rb_i = tenryu::materials::find_rho_bracket(tab_ion, rho_safe);
    const double Ti_raw_table =
        tenryu::materials::device_eos_T_from_e_monotone(tab_ion, rb_i, ei[c]);
    if (!isfinite(Ti_raw_table) || Ti_raw_table < ti_floor) {
      Ti[c] = ti_floor;
      const double logTi = log(fmax(ti_floor, 1.0e-300));
      ei[c] = tenryu::materials::device_eos_energy(tab_ion, rb_i, logTi);
    } else {
      Ti[c] = Ti_raw_table;
    }
    const double logTi = log(fmax(Ti[c], 1.0e-300));
    Pi[c] = tenryu::materials::device_eos_pressure(tab_ion, rb_i, logTi);
    if (cv_i != nullptr) {
      cv_i[c] = fmax(tenryu::materials::device_eos_cv(tab_ion, rb_i, logTi), 0.0);
    }
  } else {
    double Ti_raw = ei[c] / cv_mass_i_qei;
    if (!isfinite(Ti_raw)) {
      Ti_raw = 0.0;
    }
    Ti[c] = fmax(Ti_raw, ti_floor);
    ei[c] = cv_mass_i_qei * Ti[c];
    Pi[c] = gm1_c * rho_c * ei[c];
    if (cv_i != nullptr) {
      cv_i[c] = fmax(cv_mass_i_qei, 0.0);
    }
  }
}

__global__ void qei_coupling_substep_kernel_per_material(
    double* __restrict__ Ee_per_material,
    double* __restrict__ Ei_per_material,
    const double* __restrict__ mass_per_material,
    const double* __restrict__ volfrac,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::int8_t* __restrict__ hydro_active,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ ion_views,
    const SourceMaterialParams* __restrict__ params,
    double* __restrict__ Te_per_material,
    double* __restrict__ Ti_per_material,
    std::uint8_t* __restrict__ Te_per_material_valid,
    std::uint8_t* __restrict__ Ti_per_material_valid,
    unsigned long long* __restrict__ counts,
    const int n_cells,
    const int n_mat,
    const double dt,
    const double te_floor,
    const double ti_floor,
    const double presence_threshold_volfrac,
    const double presence_threshold_mass_density,
    const bool lazy_cache_enabled,
    const double qei_multiplier,
    const bool low_density_extrap) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_mat;
  if (idx >= total || !(dt > 0.0)) {
    return;
  }

  const int c = idx / n_mat;
  const int m = idx - c * n_mat;
  if (cell_is_void != nullptr && cell_is_void[c] != static_cast<std::uint8_t>(0)) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const double vf = volfrac[idx];
  if (!(vf > presence_threshold_volfrac) || !isfinite(vf)) {
    if (counts != nullptr) {
      atomicAdd(counts + tenryu::hydro::per_material::kPerMaterialCounterPresenceAbsent,
                1ULL);
    }
    return;
  }
  const double mass_m = mass_per_material[idx];
  const double V = vol[c];
  if (!(mass_m > 0.0) || !(V > 0.0) || !isfinite(mass_m) || !isfinite(V)) {
    if (counts != nullptr) {
      atomicAdd(counts + tenryu::hydro::per_material::kPerMaterialCounterPresenceAbsent,
                1ULL);
    }
    return;
  }
  const double rho_m = mass_m / (vf * V);
  if (!(rho_m > presence_threshold_mass_density) || !isfinite(rho_m)) {
    if (counts != nullptr) {
      atomicAdd(counts + tenryu::hydro::per_material::kPerMaterialCounterPresenceAbsent,
                1ULL);
    }
    return;
  }

  tenryu::hydro::per_material::PerMaterialAccessorView view{};
  view.mass_per_material = mass_per_material;
  view.Ee_per_material = Ee_per_material;
  view.Ei_per_material = Ei_per_material;
  view.volfrac = volfrac;
  view.vol = vol;
  view.Te_per_material = lazy_cache_enabled ? Te_per_material : nullptr;
  view.Ti_per_material = lazy_cache_enabled ? Ti_per_material : nullptr;
  view.Te_per_material_valid = lazy_cache_enabled ? Te_per_material_valid : nullptr;
  view.Ti_per_material_valid = lazy_cache_enabled ? Ti_per_material_valid : nullptr;
  view.lazy_cache_te_m_enabled = lazy_cache_enabled;
  view.presence_threshold_volfrac = presence_threshold_volfrac;
  view.presence_threshold_mass_density_g_per_cc = presence_threshold_mass_density;
  view.d_counts = counts;
  view.n_cells = n_cells;
  view.n_mat = n_mat;

  const SourceMaterialParams p = params[m];
  const auto electron_view =
      (electron_views != nullptr) ? electron_views[m] : tenryu::materials::DeviceEOSTableView{};
  const auto ion_view =
      (ion_views != nullptr) ? ion_views[m] : tenryu::materials::DeviceEOSTableView{};
  const auto electron = tenryu::hydro::per_material::get_electron_thermo_per_material(
      view, electron_view, c, m, p.Zbar, p.A, te_floor, low_density_extrap, p.gamma);
  const auto ion = tenryu::hydro::per_material::get_ion_thermo_per_material(
      view, ion_view, c, m, p.A, ti_floor, low_density_extrap, p.gamma);

  const double cv_e_m = fmax(tenryu::hydro::per_material::get_cv_e(electron), 0.0);
  const double cv_i_m = fmax(tenryu::hydro::per_material::get_cv_i(ion), 0.0);
  const double qei_specific = tenryu::materials::compute_qei_term_with_cv(
      rho_m,
      fmax(tenryu::hydro::per_material::get_te(electron), 0.0),
      fmax(tenryu::hydro::per_material::get_ti(ion), 0.0),
      fmax(p.Zbar, 0.0),
      fmax(p.A, kMinEffectiveA),
      cv_e_m,
      cv_i_m,
      dt,
      qei_multiplier);
  const double dE = qei_specific * mass_m;
  if (isfinite(dE)) {
    // AI review k15 1.4/P0-4 (2026-07-26): shared bracketed transfer —
    // exact per-material pair conservation (see qei_coupling_substep_kernel).
    const double dE_hi = fmax(Ee_per_material[idx], 0.0);
    const double dE_lo = -fmax(Ei_per_material[idx], 0.0);
    const double dE_applied = fmin(fmax(dE, dE_lo), dE_hi);
    Ee_per_material[idx] -= dE_applied;
    Ei_per_material[idx] += dE_applied;
  }
  if (Te_per_material_valid != nullptr) {
    Te_per_material_valid[idx] = 0u;
  }
  if (Ti_per_material_valid != nullptr) {
    Ti_per_material_valid[idx] = 0u;
  }
}

void smooth_net_electron_source_terms_1d(
    const core::Config::RadiationConfig::ImcConfig::NetElectronSourceSmoothingConfig&
        smoothing_cfg,
    const core::State& state,
    const std::vector<double>& sigma_R_max,
    const std::vector<int>& dominant_material,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& raw_delta_E,
    std::vector<double>& applied_delta_E) {
  const int n_cells = static_cast<int>(raw_delta_E.size());
  applied_delta_E = raw_delta_E;
  TENRYU_ASSERT(smoothing_cfg.passes >= 0,
                "smooth_net_electron_source_terms_1d passes must be >= 0");
  const int smooth_passes = smoothing_cfg.passes;
  if (n_cells < 2 || !(smoothing_cfg.alpha > 0.0) || smooth_passes == 0) {
    return;
  }

  TENRYU_ASSERT(static_cast<int>(sigma_R_max.size()) == n_cells,
                "smooth_net_electron_source_terms_1d sigma_R_max size mismatch");
  TENRYU_ASSERT(static_cast<int>(dominant_material.size()) == n_cells,
                "smooth_net_electron_source_terms_1d dominant_material size mismatch");
  TENRYU_ASSERT(static_cast<int>(cell_is_void.size()) == n_cells,
                "smooth_net_electron_source_terms_1d cell_is_void size mismatch");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_cells + 1),
                "smooth_net_electron_source_terms_1d x_r size mismatch");
  TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_1d mass size mismatch");
  TENRYU_ASSERT(state.Te.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_1d Te size mismatch");
  TENRYU_ASSERT(state.rho.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_1d rho size mismatch");
  if (smoothing_cfg.gradient_adaptive) {
    TENRYU_ASSERT(smoothing_cfg.grad_Te_scale > 0.0,
                  "smooth_net_electron_source_terms_1d grad_Te_scale must be > 0");
    TENRYU_ASSERT(smoothing_cfg.grad_rho_scale > 0.0,
                  "smooth_net_electron_source_terms_1d grad_rho_scale must be > 0");
  }

  NetESourceSmoothingDiagnostics smoothing_diag;
  if (smoothing_cfg.gradient_adaptive) {
    std::vector<double> host_mass(static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> host_node_r(static_cast<std::size_t>(n_cells + 1), 0.0);
    std::vector<double> host_Te(static_cast<std::size_t>(n_cells), 0.0);
    std::vector<double> host_rho(static_cast<std::size_t>(n_cells), 0.0);
    state.mass.copy_to_host(host_mass.data());
    state.x_r.copy_to_host(host_node_r.data());
    state.Te.copy_to_host(host_Te.data());
    state.rho.copy_to_host(host_rho.data());
    smoothing_diag = compute_gradient_adaptive_smoothing_diagnostics(
        smoothing_cfg, host_mass, host_node_r, sigma_R_max, dominant_material,
        cell_is_void, host_Te, host_rho);
  }

  double* d_H_raw = nullptr;
  double* d_H_apply = nullptr;
  double* d_sigma_R_max = nullptr;
  int* d_dominant_material = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;

  d_H_raw = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_H_raw",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_H_apply = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_H_apply",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_sigma_R_max = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_sigma_R_max",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_dominant_material = static_cast<int*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_dominant_material",
      sizeof(int) * static_cast<std::size_t>(n_cells)));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_1d:d_cell_is_void",
      sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells)));

  cuda_check(cudaMemcpy(d_H_raw, raw_delta_E.data(),
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_1d copy H_raw failed");
  cuda_check(cudaMemcpy(d_sigma_R_max, sigma_R_max.data(),
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_1d copy sigma_R_max failed");
  cuda_check(cudaMemcpy(d_dominant_material, dominant_material.data(),
                        sizeof(int) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_1d copy dominant_material failed");
  cuda_check(cudaMemcpy(d_cell_is_void, cell_is_void.data(),
                        sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_1d copy cell_is_void failed");

  const int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  for (int pass = 0; pass < smooth_passes; ++pass) {
    const double* const d_H_in = (pass % 2 == 0) ? d_H_raw : d_H_apply;
    double* const d_H_out = (pass % 2 == 0) ? d_H_apply : d_H_raw;
    smooth_net_electron_source_terms_1d_kernel<<<blocks, threads>>>(
        d_H_out, d_H_in, state.mass.data(), state.x_r.data(), d_sigma_R_max,
        state.Te.data(), state.rho.data(), d_dominant_material, d_cell_is_void,
        n_cells, smoothing_cfg.alpha, smoothing_cfg.tau_threshold,
        smoothing_cfg.grad_Te_scale, smoothing_cfg.grad_rho_scale,
        smoothing_cfg.gradient_adaptive);
    cuda_check(cudaGetLastError(),
               "smooth_net_electron_source_terms_1d kernel launch failed");
  }

  const double* const d_H_result =
      (smooth_passes % 2 == 0) ? d_H_raw : d_H_apply;
  cuda_check(cudaMemcpy(applied_delta_E.data(), d_H_result,
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyDeviceToHost),
             "smooth_net_electron_source_terms_1d copy H_apply failed");

  if (smoothing_cfg.gradient_adaptive) {
    std::ostringstream oss;
    oss << "[net_e_smooth] step=" << (state.step + 1)
        << " faces_total=" << smoothing_diag.faces_total
        << " faces_active=" << smoothing_diag.faces_active
        << " alpha_mean=" << smoothing_diag.alpha_mean
        << " alpha_min=" << smoothing_diag.alpha_min;
    core::log_info(oss.str());
  }
}

void smooth_net_electron_source_terms_2d(
    const core::Config::RadiationConfig::ImcConfig::NetElectronSourceSmoothingConfig&
        smoothing_cfg,
    const core::State& state,
    const std::vector<double>& sigma_R_max,
    const std::vector<int>& dominant_material,
    const std::vector<std::uint8_t>& cell_is_void,
    const double* difference_W,
    const std::vector<double>& raw_delta_E,
    std::vector<double>& applied_delta_E) {
  const int n_cells = static_cast<int>(raw_delta_E.size());
  applied_delta_E = raw_delta_E;
  TENRYU_ASSERT(smoothing_cfg.passes >= 0,
                "smooth_net_electron_source_terms_2d passes must be >= 0");
  const int smooth_passes = smoothing_cfg.passes;
  if (n_cells < 2 || !(smoothing_cfg.alpha > 0.0) || smooth_passes == 0) {
    return;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(nr > 0 && nz > 0,
                "smooth_net_electron_source_terms_2d requires positive topology");
  TENRYU_ASSERT(n_cells == nr * nz,
                "smooth_net_electron_source_terms_2d topology size mismatch");
  TENRYU_ASSERT(static_cast<int>(sigma_R_max.size()) == n_cells,
                "smooth_net_electron_source_terms_2d sigma_R_max size mismatch");
  TENRYU_ASSERT(static_cast<int>(dominant_material.size()) == n_cells,
                "smooth_net_electron_source_terms_2d dominant_material size mismatch");
  TENRYU_ASSERT(static_cast<int>(cell_is_void.size()) == n_cells,
                "smooth_net_electron_source_terms_2d cell_is_void size mismatch");
  const std::size_t n_nodes =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  TENRYU_ASSERT(state.x_r.size() == n_nodes,
                "smooth_net_electron_source_terms_2d x_r size mismatch");
  TENRYU_ASSERT(state.x_z.size() == n_nodes,
                "smooth_net_electron_source_terms_2d x_z size mismatch");
  TENRYU_ASSERT(state.vol.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_2d vol size mismatch");
  TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_2d mass size mismatch");
  TENRYU_ASSERT(state.Te.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_2d Te size mismatch");
  TENRYU_ASSERT(state.rho.size() == static_cast<std::size_t>(n_cells),
                "smooth_net_electron_source_terms_2d rho size mismatch");
  if (smoothing_cfg.gradient_adaptive) {
    TENRYU_ASSERT(smoothing_cfg.grad_Te_scale > 0.0,
                  "smooth_net_electron_source_terms_2d grad_Te_scale must be > 0");
    TENRYU_ASSERT(smoothing_cfg.grad_rho_scale > 0.0,
                  "smooth_net_electron_source_terms_2d grad_rho_scale must be > 0");
  }

  double* d_H_raw = nullptr;
  double* d_H_apply = nullptr;
  double* d_sigma_R_max = nullptr;
  int* d_dominant_material = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;

  d_H_raw = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_H_raw",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_H_apply = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_H_apply",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_sigma_R_max = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_sigma_R_max",
      sizeof(double) * static_cast<std::size_t>(n_cells)));
  d_dominant_material = static_cast<int*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_dominant_material",
      sizeof(int) * static_cast<std::size_t>(n_cells)));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:smooth_net_electron_source_terms_2d:d_cell_is_void",
      sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells)));

  cuda_check(cudaMemcpy(d_H_raw, raw_delta_E.data(),
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_2d copy H_raw failed");
  cuda_check(cudaMemcpy(d_sigma_R_max, sigma_R_max.data(),
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_2d copy sigma_R_max failed");
  cuda_check(cudaMemcpy(d_dominant_material, dominant_material.data(),
                        sizeof(int) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_2d copy dominant_material failed");
  cuda_check(cudaMemcpy(d_cell_is_void, cell_is_void.data(),
                        sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyHostToDevice),
             "smooth_net_electron_source_terms_2d copy cell_is_void failed");

  const int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  for (int pass = 0; pass < smooth_passes; ++pass) {
    const double* const d_H_in = (pass % 2 == 0) ? d_H_raw : d_H_apply;
    double* const d_H_out = (pass % 2 == 0) ? d_H_apply : d_H_raw;
    smooth_net_electron_source_terms_2d_kernel<<<blocks, threads>>>(
        d_H_out, d_H_in, state.mass.data(), state.vol.data(), state.x_r.data(),
        state.x_z.data(), d_sigma_R_max, state.Te.data(), state.rho.data(),
        d_dominant_material, d_cell_is_void, difference_W, nr, nz,
        smoothing_cfg.alpha, smoothing_cfg.tau_threshold,
        smoothing_cfg.grad_Te_scale, smoothing_cfg.grad_rho_scale,
        smoothing_cfg.gradient_adaptive);
    cuda_check(cudaGetLastError(),
               "smooth_net_electron_source_terms_2d kernel launch failed");
  }

  const double* const d_H_result =
      (smooth_passes % 2 == 0) ? d_H_raw : d_H_apply;
  cuda_check(cudaMemcpy(applied_delta_E.data(), d_H_result,
                        sizeof(double) * static_cast<std::size_t>(n_cells),
                        cudaMemcpyDeviceToHost),
             "smooth_net_electron_source_terms_2d copy H_apply failed");

}

void conservative_smooth_delta_E_1d(
    const core::Config::RadiationConfig::ImcConfig::ConservativeSmootherConfig&
        smoothing_cfg,
    const core::State& state,
    const std::vector<std::uint8_t>& cell_is_void,
    std::vector<double>& applied_delta_E) {
  const int n_cells = static_cast<int>(applied_delta_E.size());
  TENRYU_ASSERT(smoothing_cfg.passes >= 0,
                "conservative_smooth_delta_E_1d passes must be >= 0");
  const int smooth_passes = smoothing_cfg.passes;
  if (n_cells < 2 || !(smoothing_cfg.alpha > 0.0) || smooth_passes == 0) {
    return;
  }

  TENRYU_ASSERT(static_cast<int>(cell_is_void.size()) == n_cells,
                "conservative_smooth_delta_E_1d cell_is_void size mismatch");
  TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells),
                "conservative_smooth_delta_E_1d mass size mismatch");

  double* d_H_a = nullptr;
  double* d_H_b = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;

  const std::size_t cell_bytes =
      sizeof(double) * static_cast<std::size_t>(n_cells);
  const std::size_t mask_bytes =
      sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
  d_H_a = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:conservative_smooth_delta_E_1d:d_H_a", cell_bytes));
  d_H_b = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:conservative_smooth_delta_E_1d:d_H_b", cell_bytes));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:conservative_smooth_delta_E_1d:d_cell_is_void",
      mask_bytes));

  cuda_check(cudaMemcpy(d_H_a, applied_delta_E.data(), cell_bytes,
                        cudaMemcpyHostToDevice),
             "conservative_smooth_delta_E_1d copy H_a failed");
  cuda_check(cudaMemcpy(d_cell_is_void, cell_is_void.data(), mask_bytes,
                        cudaMemcpyHostToDevice),
             "conservative_smooth_delta_E_1d copy cell_is_void failed");

  const int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  for (int pass = 0; pass < smooth_passes; ++pass) {
    const double* const d_H_in = (pass % 2 == 0) ? d_H_a : d_H_b;
    double* const d_H_out = (pass % 2 == 0) ? d_H_b : d_H_a;
    conservative_smooth_delta_E_1d_kernel<<<blocks, threads>>>(
        d_H_out, d_H_in, state.mass.data(), d_cell_is_void, n_cells,
        smoothing_cfg.alpha);
    cuda_check(cudaGetLastError(),
               "conservative_smooth_delta_E_1d kernel launch failed");
  }

  const double* const d_H_result =
      (smooth_passes % 2 == 0) ? d_H_a : d_H_b;
  cuda_check(cudaMemcpy(applied_delta_E.data(), d_H_result, cell_bytes,
                        cudaMemcpyDeviceToHost),
             "conservative_smooth_delta_E_1d copy H_result failed");

}

}  // namespace

double inject_radiation_source_terms_impl(core::State& state,
                                          const core::Config& cfg,
                                          const double dt,
                                          double* E_floor_injected,
                                          int* clamp_count,
                                          const std::vector<double>* sigma_R_max,
                                          const hydro::HydroEOSContext* eos_ctx) {
  const bool verbose_subphase_timing = (cfg.main.verbosity == "verbose");
  using Clock = std::chrono::steady_clock;
  const auto t_inject_start =
      verbose_subphase_timing ? Clock::now() : Clock::time_point{};
  auto t_inject_subphase = t_inject_start;
  double setup_ms = 0.0;
  double diff_cell_ms = 0.0;
  double blocked_ms = 0.0;
  double d2h_ms = 0.0;
  double raw_net_ms = 0.0;
  double smooth_ms = 0.0;
  double gpu_ms = 0.0;
  double reduction_ms = 0.0;
  double finalize_ms = 0.0;
  const auto mark_subphase = [&](double& value) {
    if (verbose_subphase_timing) {
      const auto t_now = Clock::now();
      value += std::chrono::duration<double, std::milli>(
                   t_now - t_inject_subphase)
                   .count();
      t_inject_subphase = t_now;
    }
  };
  const auto log_subphase = [&]() {
    if (verbose_subphase_timing) {
      const double total_ms = setup_ms + diff_cell_ms + blocked_ms + d2h_ms +
                              raw_net_ms + smooth_ms + gpu_ms +
                              reduction_ms + finalize_ms;
      std::ostringstream oss;
      oss << "[inject_subphase] step=" << state.step
          << std::fixed << std::setprecision(2)
          << " setup=" << setup_ms
          << " diff_cell=" << diff_cell_ms
          << " blocked=" << blocked_ms
          << " d2h=" << d2h_ms
          << " raw_net=" << raw_net_ms
          << " smooth=" << smooth_ms
          << " gpu=" << gpu_ms
          << " reduction=" << reduction_ms
          << " finalize=" << finalize_ms
          << " total=" << total_ms << " ms";
      core::log_info(oss.str());
    }
  };

  if (state.rad_dep.empty() || state.rho.empty() || dt <= 0.0) {
    mark_subphase(setup_ms);
    log_subphase();
    return 0.0;
  }
  if (cfg.materials.materials.empty()) {
    mark_subphase(setup_ms);
    log_subphase();
    return 0.0;
  }
  assert_common_source_state_sizes(state, "inject_radiation_source_terms");
  TENRYU_ASSERT(state.rad_emit.empty() || state.rad_emit.size() == state.rad_dep.size(),
                "inject_radiation_source_terms requires rad_emit size == rad_dep size when present");

  const auto& materials = cfg.materials.materials;
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "inject_radiation_source_terms requires at least one non-void material");
  const auto& mat0 = materials[static_cast<std::size_t>(first_nonvoid)];
  const bool has_table_eos =
      mat0.eos_tables != nullptr && !use_exact_ideal_gas_hydro_backend(mat0);

  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<double> A_eff;
  std::vector<double> gamma_eff;
  std::vector<int> dominant_material;
  compute_effective_A_gamma(cfg, state, n_cells, A_eff, gamma_eff,
                            &dominant_material);
  TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_radiation_source_terms A_eff size mismatch");
  TENRYU_ASSERT(gamma_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_radiation_source_terms gamma_eff size mismatch");
  TENRYU_ASSERT(dominant_material.size() == static_cast<std::size_t>(n_cells),
                "inject_radiation_source_terms dominant_material size mismatch");

  bool any_cv_e_override = false;
  for (const auto& mat : materials) {
    if (mat.cv_e_override > 0.0) {
      any_cv_e_override = true;
      break;
    }
  }
  const bool use_first_cv_override = any_cv_e_override && mat0.cv_e_override > 0.0;
  mark_subphase(setup_ms);

  TENRYU_ASSERT(state.rad_dep.size() % state.rho.size() == 0,
                "inject_radiation_source_terms requires rad_dep size divisible by rho size");
  const int n_groups =
      (n_cells > 0) ? static_cast<int>(state.rad_dep.size() / state.rho.size()) : 1;
  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  std::vector<std::uint8_t> diffusion_cell(static_cast<std::size_t>(n_cells), 0U);
  if (state.ddmc_mode_map_valid) {
    TENRYU_ASSERT(state.ddmc_mode_map.size() == n_cell_groups,
                  "inject_radiation_source_terms ddmc_mode_map size mismatch");
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t base =
          static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
      for (int g = 0; g < n_groups; ++g) {
        if (state.ddmc_mode_map[base + static_cast<std::size_t>(g)] ==
            kTransportModeDiffusion) {
          diffusion_cell[static_cast<std::size_t>(c)] = 1U;
          break;
        }
      }
    }
  }
  mark_subphase(diff_cell_ms);

  std::vector<std::uint8_t> source_apply_blocked = state.cell_is_void;
  for (int c = 0; c < n_cells; ++c) {
    if (diffusion_cell[static_cast<std::size_t>(c)] != 0U) {
      source_apply_blocked[static_cast<std::size_t>(c)] = 1U;
    }
  }
  const bool smoothing_requested = cfg.radiation.imc.net_e_source_smoothing.enabled;
  if (smoothing_requested && cfg.radiation.imc.difference.enabled &&
      state.difference_W.size() == static_cast<std::size_t>(n_cells)) {
    std::vector<double> difference_W(static_cast<std::size_t>(n_cells), 0.0);
    state.difference_W.copy_to_host(difference_W.data());
    for (int c = 0; c < n_cells; ++c) {
      const double W = difference_W[static_cast<std::size_t>(c)];
      if (!std::isfinite(W) || W >= active_W_for_smoothing) {
        source_apply_blocked[static_cast<std::size_t>(c)] = 1U;
      }
    }
  } else if (smoothing_requested && cfg.radiation.imc.difference.enabled) {
    std::fill(source_apply_blocked.begin(), source_apply_blocked.end(), 1U);
  }

  std::vector<double> rad_dep(state.rad_dep.size(), 0.0);
  std::vector<double> rad_emit(state.rad_dep.size(), 0.0);
  std::vector<double> holo_rad_dep;
  std::vector<double> holo_rad_emit;
  std::vector<double> raw_delta_E;
  std::vector<double> applied_delta_E;

  bool holo_core_present = false;
  bool holo_patch_present = false;
  bool holo_source_owner = false;
  if (cfg.radiation.holo.enabled && state.holo_core_mask_valid) {
    TENRYU_ASSERT(state.holo_core_mask.size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms requires holo_core_mask/rho size match");
    TENRYU_ASSERT(state.holo_patch_mask.size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms requires holo_patch_mask/rho size match");
    TENRYU_ASSERT(state.holo_lo_weight.size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms requires holo_lo_weight/rho size match");
    holo_core_present =
        std::any_of(state.holo_core_mask.begin(),
                    state.holo_core_mask.end(),
                    [](const std::uint8_t value) { return value != 0U; });
    holo_patch_present =
        std::any_of(state.holo_patch_mask.begin(),
                    state.holo_patch_mask.end(),
                    [](const std::uint8_t value) { return value != 0U; });
    if (holo_core_present || holo_patch_present) {
      TENRYU_ASSERT(state.holo_rad_dep.size() == state.rad_dep.size(),
                    "inject_radiation_source_terms requires holo_rad_dep size == rad_dep size");
      TENRYU_ASSERT(state.holo_rad_emit.size() == state.rad_dep.size(),
                    "inject_radiation_source_terms requires holo_rad_emit size == rad_dep size");
    }
  }

  const bool smoothing_supported =
      smoothing_requested && (state.mesh.dim == 1 || state.mesh.dim == 2);
  if (smoothing_supported) {
    TENRYU_ASSERT(sigma_R_max != nullptr,
                  "inject_radiation_source_terms requires sigma_R_max when "
                  "Radiation.imc.net_e_source_smoothing.enabled");
    TENRYU_ASSERT(sigma_R_max->size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms sigma_R_max size mismatch");
  }
  mark_subphase(blocked_ms);

  state.rad_dep.copy_to_host(rad_dep.data());
  if (state.rad_emit.size() == state.rad_dep.size() && !state.rad_emit.empty()) {
    state.rad_emit.copy_to_host(rad_emit.data());
  }
  if (holo_core_present || holo_patch_present) {
    holo_rad_dep.assign(state.rad_dep.size(), 0.0);
    holo_rad_emit.assign(state.rad_dep.size(), 0.0);
    state.holo_rad_dep.copy_to_host(holo_rad_dep.data());
    state.holo_rad_emit.copy_to_host(holo_rad_emit.data());
    holo_source_owner = state.holo_lo_source_valid;
    if (!holo_source_owner) {
      for (int c = 0; c < n_cells && !holo_source_owner; ++c) {
        const std::size_t c_idx = static_cast<std::size_t>(c);
        if (state.holo_core_mask[c_idx] == 0U &&
            state.holo_patch_mask[c_idx] == 0U) {
          continue;
        }
        const std::size_t cell_base =
            c_idx * static_cast<std::size_t>(n_groups);
        for (int g = 0; g < n_groups; ++g) {
          const std::size_t idx = cell_base + static_cast<std::size_t>(g);
          if (holo_rad_dep[idx] != 0.0 || holo_rad_emit[idx] != 0.0) {
            holo_source_owner = true;
            break;
          }
        }
      }
    }
  }
  const bool has_table_cv_e = !state.cv_e.empty();

  const double te_floor = cfg.numerics.floors.Te;
  const double ti_floor = cfg.numerics.floors.Ti;
  const bool use_two_temp = cfg.main.two_temperature;
  const bool sn_material_coupling = cfg.radiation.holo.sn_material_coupling;
  const bool sn_qd_lo_updates_material_directly =
      sn_material_coupling && state.mesh.dim == 1 &&
      cfg.radiation.holo.solver == "quasidiffusion_1d";
  const bool holo_updates_material_directly =
      holo_source_owner && (!sn_material_coupling ||
                            sn_qd_lo_updates_material_directly);
  mark_subphase(d2h_ms);

  compute_raw_net_radiation_source_terms(rad_dep, rad_emit, n_cells, n_groups,
                                         raw_delta_E);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_idx = static_cast<std::size_t>(c);
    const bool holo_core_cell =
        holo_source_owner && state.holo_core_mask[c_idx] != 0U;
    const bool holo_patch_cell =
        holo_source_owner && state.holo_patch_mask[c_idx] != 0U;
    if (holo_core_cell || holo_patch_cell) {
      if (diffusion_cell[c_idx] != 0U) {
        TENRYU_ASSERT(false,
                      "HOLO source ownership cannot overlap deterministic diffusion cells");
      }
      double holo_delta_E = 0.0;
      const std::size_t cell_base =
          c_idx * static_cast<std::size_t>(n_groups);
      for (int g = 0; g < n_groups; ++g) {
        const std::size_t idx = cell_base + static_cast<std::size_t>(g);
        holo_delta_E += holo_rad_dep[idx] - holo_rad_emit[idx];
      }
      if (holo_core_cell) {
        raw_delta_E[c_idx] = holo_delta_E;
      } else {
        const double w_raw = state.holo_lo_weight[c_idx];
        const double w = std::isfinite(w_raw) ? std::clamp(w_raw, 0.0, 1.0) : 0.0;
        raw_delta_E[c_idx] =
            w * holo_delta_E + (1.0 - w) * raw_delta_E[c_idx];
      }
      source_apply_blocked[c_idx] = 1U;
      continue;
    }
    if (diffusion_cell[static_cast<std::size_t>(c)] != 0U) {
      raw_delta_E[static_cast<std::size_t>(c)] = 0.0;
    }
  }
  applied_delta_E = raw_delta_E;
  mark_subphase(raw_net_ms);

  if (smoothing_supported) {
    if (state.mesh.dim == 1) {
      smooth_net_electron_source_terms_1d(
          cfg.radiation.imc.net_e_source_smoothing, state, *sigma_R_max,
          dominant_material, source_apply_blocked, raw_delta_E, applied_delta_E);
    } else if (state.mesh.dim == 2) {
      const double* const difference_W =
          (cfg.radiation.imc.difference.enabled &&
           state.difference_W.size() == static_cast<std::size_t>(n_cells))
              ? state.difference_W.data()
              : nullptr;
      smooth_net_electron_source_terms_2d(
          cfg.radiation.imc.net_e_source_smoothing, state, *sigma_R_max,
          dominant_material, source_apply_blocked, difference_W, raw_delta_E,
          applied_delta_E);
    }
  }
  const auto& cons_smooth = cfg.radiation.imc.conservative_smoother;
  if (cons_smooth.enabled && cons_smooth.passes > 0 &&
      cons_smooth.alpha > 0.0 && n_cells >= 2 && state.mesh.dim == 1) {
    std::vector<std::uint8_t> conservative_source_blocked = state.cell_is_void;
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_idx = static_cast<std::size_t>(c);
      if (diffusion_cell[c_idx] != 0U ||
          (holo_source_owner &&
           (state.holo_core_mask[c_idx] != 0U ||
            state.holo_patch_mask[c_idx] != 0U))) {
        conservative_source_blocked[c_idx] = 1U;
      }
    }
    conservative_smooth_delta_E_1d(cons_smooth, state,
                                   conservative_source_blocked, applied_delta_E);
  }
  mark_subphase(smooth_ms);

  SourceEOSTableViews table_views;
  if (use_two_temp && has_table_eos) {
    table_views = select_source_eos_table_views(mat0, first_nonvoid, eos_ctx);
    TENRYU_ASSERT(table_views.electron.n_rho > 0 && table_views.ion.n_rho > 0,
                  "inject_radiation_source_terms requires non-empty device EOS tables");
  }
  if (has_table_cv_e) {
    TENRYU_ASSERT(state.cv_e.size() == static_cast<std::size_t>(n_cells),
                  "inject_radiation_source_terms cv_e size mismatch");
  }
  if (state.delta_E_rad_prev.size() != static_cast<std::size_t>(n_cells)) {
    state.delta_E_rad_prev.reset(static_cast<std::size_t>(n_cells));
  }

  double* d_applied_delta_E = nullptr;
  double* d_A_eff = nullptr;
  double* d_gamma_eff = nullptr;
  double* d_reduction = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;
  std::uint8_t* d_diffusion_cell = nullptr;
  std::uint8_t* d_holo_source_cell = nullptr;
  int* d_clamp_count = nullptr;

  const std::size_t cell_bytes =
      sizeof(double) * static_cast<std::size_t>(n_cells);
  const std::size_t mask_bytes =
      sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
  d_applied_delta_E = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_applied_delta_E",
      cell_bytes));
  d_A_eff = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_A_eff", cell_bytes));
  d_gamma_eff = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_gamma_eff",
      cell_bytes));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_cell_is_void",
      mask_bytes));
  d_diffusion_cell = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_diffusion_cell",
      mask_bytes));
  if (holo_updates_material_directly) {
    d_holo_source_cell = static_cast<std::uint8_t*>(core::device_scratch_acquire(
        "source_terms:inject_radiation_source_terms_impl:d_holo_source_cell",
        mask_bytes));
  }
  d_reduction = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_reduction",
      2 * sizeof(double)));
  d_clamp_count = static_cast<int*>(core::device_scratch_acquire(
      "source_terms:inject_radiation_source_terms_impl:d_clamp_count",
      sizeof(int)));

  cuda_check(cudaMemcpy(d_applied_delta_E, applied_delta_E.data(), cell_bytes,
                        cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy applied_delta_E failed");
  cuda_check(cudaMemcpy(d_A_eff, A_eff.data(), cell_bytes, cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy A_eff failed");
  cuda_check(cudaMemcpy(d_gamma_eff, gamma_eff.data(), cell_bytes,
                        cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy gamma_eff failed");
  cuda_check(cudaMemcpy(d_cell_is_void, state.cell_is_void.data(), mask_bytes,
                        cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy cell_is_void failed");
  cuda_check(cudaMemcpy(d_diffusion_cell, diffusion_cell.data(), mask_bytes,
                        cudaMemcpyHostToDevice),
             "inject_radiation_source_terms copy diffusion_cell failed");
  if (holo_updates_material_directly) {
    cuda_check(cudaMemcpy(d_holo_source_cell, state.holo_core_mask.data(), mask_bytes,
                          cudaMemcpyHostToDevice),
               "inject_radiation_source_terms copy holo_source_cell failed");
  }
  cuda_check(cudaMemset(d_reduction, 0, 2 * sizeof(double)),
             "inject_radiation_source_terms zero reduction failed");
  cuda_check(cudaMemset(d_clamp_count, 0, sizeof(int)),
             "inject_radiation_source_terms zero clamp_count failed");

  const int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  inject_radiation_source_terms_kernel<<<blocks, threads>>>(
      state.ee.data(), state.Te.data(), state.Pe.data(), state.ei.data(),
      state.Ti.data(), state.Pi.data(), state.delta_E_rad_prev.data(),
      state.rho.data(), state.mass.data(), state.zbar.data(),
      has_table_cv_e ? state.cv_e.data() : nullptr, d_applied_delta_E, d_A_eff,
      d_gamma_eff, d_cell_is_void, d_diffusion_cell, d_holo_source_cell, n_cells, te_floor,
      ti_floor, use_two_temp, has_table_eos, use_first_cv_override,
      mat0.cv_e_override, mat0.eos_T_ref_eV, table_views.ion,
      table_views.electron, &d_reduction[0], &d_reduction[1], d_clamp_count);
  cuda_check(cudaGetLastError(),
             "inject_radiation_source_terms kernel launch failed");
  if (verbose_subphase_timing) {
    cuda_check(core::debug_kernel_sync(),
               "inject_radiation_source_terms kernel synchronize failed");
  }
  mark_subphase(gpu_ms);

  double host_reduction[2] = {0.0, 0.0};
  int local_clamp_count = 0;
  cuda_check(cudaMemcpy(host_reduction, d_reduction, 2 * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "inject_radiation_source_terms copy reduction failed");
  cuda_check(cudaMemcpy(&local_clamp_count, d_clamp_count, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "inject_radiation_source_terms copy clamp_count failed");
  mark_subphase(reduction_ms);

  const double floor_energy = host_reduction[0];
  const double skipped_energy = host_reduction[1];
  accumulate_floor_and_clamp(E_floor_injected, clamp_count, floor_energy, local_clamp_count);
  mark_subphase(finalize_ms);
  log_subphase();
  return skipped_energy;
}

double inject_radiation_source_terms(core::State& state,
                                     const core::Config& cfg,
                                     const double dt,
                                     double* E_floor_injected,
                                     int* clamp_count,
                                     const std::vector<double>* sigma_R_max) {
  return inject_radiation_source_terms_impl(
      state, cfg, dt, E_floor_injected, clamp_count, sigma_R_max, nullptr);
}

double inject_radiation_source_terms(core::State& state,
                                     const core::Config& cfg,
                                     const double dt,
                                     double* E_floor_injected,
                                     int* clamp_count,
                                     const std::vector<double>* sigma_R_max,
                                     const hydro::HydroEOSContext* eos_ctx) {
  return inject_radiation_source_terms_impl(
      state, cfg, dt, E_floor_injected, clamp_count, sigma_R_max, eos_ctx);
}

void apply_qei_coupling_substep(core::State& state,
                                const core::Config& cfg,
                                const double dt_sub,
                                const hydro::HydroEOSContext* eos_ctx) {
  if (!cfg.main.two_temperature || !(dt_sub > 0.0) || state.rho.empty()) {
    return;
  }
  if (cfg.materials.materials.empty()) {
    return;
  }
  assert_common_source_state_sizes(state, "apply_qei_coupling_substep");

  const int n_cells = static_cast<int>(state.rho.size());
  const auto& materials = cfg.materials.materials;
  const int n_mat = static_cast<int>(materials.size());
  const std::size_t n_cell_mat =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    TENRYU_ASSERT(n_mat > 0, "per-material Q_ei requires at least one material");
    TENRYU_ASSERT(state.mass_per_material.size() == n_cell_mat,
                  "per-material Q_ei requires mass_per_material size n_cells*n_materials");
    TENRYU_ASSERT(state.Ee_per_material.size() == n_cell_mat,
                  "per-material Q_ei requires Ee_per_material size n_cells*n_materials");
    TENRYU_ASSERT(state.Ei_per_material.size() == n_cell_mat,
                  "per-material Q_ei requires Ei_per_material size n_cells*n_materials");
    TENRYU_ASSERT(state.volFrac.size() == n_cell_mat,
                  "per-material Q_ei requires volFrac size n_cells*n_materials");
    TENRYU_ASSERT(state.vol.size() == static_cast<std::size_t>(n_cells),
                  "per-material Q_ei requires vol size n_cells");
    TENRYU_ASSERT(state.hydro_active.empty() ||
                      state.hydro_active.size() == static_cast<std::size_t>(n_cells),
                  "per-material Q_ei hydro_active size mismatch");

    bool any_table_backed = false;
    for (const auto& mat : materials) {
      any_table_backed =
          any_table_backed || mat.eos_tables != nullptr || mat.eos_model != "ideal_gas";
    }
    if (any_table_backed) {
      TENRYU_ASSERT(eos_ctx != nullptr,
                    "per-material Q_ei requires HydroEOSContext for table-backed materials");
      TENRYU_ASSERT(eos_ctx->n_materials >= n_mat,
                    "per-material Q_ei EOS context material count mismatch");
    }

    std::uint8_t* d_cell_is_void = nullptr;
    std::int8_t* d_hydro_active = nullptr;
    SourceMaterialParams* d_params = nullptr;
    std::uint8_t* d_te_valid = nullptr;
    std::uint8_t* d_ti_valid = nullptr;
    unsigned long long* d_counts = nullptr;

    const std::size_t void_bytes =
        sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
    d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
        "source_terms:apply_qei_coupling_substep:d_cell_is_void_per_material",
        void_bytes));
    cuda_check(cudaMemcpy(d_cell_is_void, state.cell_is_void.data(), void_bytes,
                          cudaMemcpyHostToDevice),
               "per-material Q_ei copy cell_is_void failed");
    d_hydro_active = const_cast<std::int8_t*>(state.hydro_active_device_ptr());

    const std::vector<SourceMaterialParams> h_params = make_source_material_params(cfg);
    d_params = static_cast<SourceMaterialParams*>(core::device_scratch_acquire(
        "source_terms:apply_qei_coupling_substep:d_params_per_material",
        h_params.size() * sizeof(SourceMaterialParams)));
    cuda_check(cudaMemcpy(d_params,
                          h_params.data(),
                          h_params.size() * sizeof(SourceMaterialParams),
                          cudaMemcpyHostToDevice),
               "per-material Q_ei copy params failed");

    const bool lazy_cache_enabled = cfg.numerics.materials.lazy_cache_te_m_enabled;
    if (lazy_cache_enabled) {
      TENRYU_ASSERT(state.Te_per_material.size() == n_cell_mat,
                    "per-material Q_ei lazy cache requires Te_per_material");
      TENRYU_ASSERT(state.Ti_per_material.size() == n_cell_mat,
                    "per-material Q_ei lazy cache requires Ti_per_material");
      TENRYU_ASSERT(state.Te_per_material_valid.size() == n_cell_mat,
                    "per-material Q_ei lazy cache requires Te valid flags");
      TENRYU_ASSERT(state.Ti_per_material_valid.size() == n_cell_mat,
                    "per-material Q_ei lazy cache requires Ti valid flags");
      d_te_valid = static_cast<std::uint8_t*>(core::device_scratch_acquire(
          "source_terms:apply_qei_coupling_substep:d_te_valid",
          n_cell_mat * sizeof(std::uint8_t)));
      d_ti_valid = static_cast<std::uint8_t*>(core::device_scratch_acquire(
          "source_terms:apply_qei_coupling_substep:d_ti_valid",
          n_cell_mat * sizeof(std::uint8_t)));
      cuda_check(cudaMemcpy(d_te_valid,
                            state.Te_per_material_valid.data(),
                            n_cell_mat * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice),
                 "per-material Q_ei copy Te valid failed");
      cuda_check(cudaMemcpy(d_ti_valid,
                            state.Ti_per_material_valid.data(),
                            n_cell_mat * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice),
                 "per-material Q_ei copy Ti valid failed");
    }
    d_counts = static_cast<unsigned long long*>(core::device_scratch_acquire(
        "source_terms:apply_qei_coupling_substep:d_counts",
        hydro::per_material::kPerMaterialCounterCount *
            sizeof(unsigned long long)));
    cuda_check(cudaMemset(d_counts,
                          0,
                          hydro::per_material::kPerMaterialCounterCount *
                              sizeof(unsigned long long)),
               "per-material Q_ei cudaMemset counts failed");

    const auto* d_electron_views =
        (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_electron_views
                                                              : nullptr;
    const auto* d_ion_views =
        (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_ion_views
                                                              : nullptr;
    const int threads = 256;
    const int blocks =
        (static_cast<int>(n_cell_mat) + threads - 1) / threads;
    qei_coupling_substep_kernel_per_material<<<blocks, threads>>>(
        state.Ee_per_material.data(),
        state.Ei_per_material.data(),
        state.mass_per_material.data(),
        state.volFrac.data(),
        state.vol.data(),
        d_cell_is_void,
        d_hydro_active,
        d_electron_views,
        d_ion_views,
        d_params,
        lazy_cache_enabled ? state.Te_per_material.data() : nullptr,
        lazy_cache_enabled ? state.Ti_per_material.data() : nullptr,
        d_te_valid,
        d_ti_valid,
        d_counts,
        n_cells,
        n_mat,
        dt_sub,
        cfg.numerics.floors.Te,
        cfg.numerics.floors.Ti,
        cfg.numerics.materials.presence_threshold_volfrac,
        cfg.numerics.materials.presence_threshold_mass_density_g_per_cc,
        lazy_cache_enabled,
        cfg.numerics.hydro.qei_multiplier,
        cfg.materials.low_density_extrapolation);
    cuda_check(cudaGetLastError(),
               "per-material Q_ei kernel launch failed");
    cuda_check(core::debug_kernel_sync(),
               "per-material Q_ei kernel synchronize failed");
    state.dispatch_counters.per_material_kernel_call_count.fetch_add(
        1, std::memory_order_relaxed);

    if (lazy_cache_enabled) {
      cuda_check(cudaMemcpy(state.Te_per_material_valid.data(),
                            d_te_valid,
                            n_cell_mat * sizeof(std::uint8_t),
                            cudaMemcpyDeviceToHost),
                 "per-material Q_ei copy Te valid back failed");
      cuda_check(cudaMemcpy(state.Ti_per_material_valid.data(),
                            d_ti_valid,
                            n_cell_mat * sizeof(std::uint8_t),
                            cudaMemcpyDeviceToHost),
                 "per-material Q_ei copy Ti valid back failed");
    }
    unsigned long long h_counts[hydro::per_material::kPerMaterialCounterCount] = {};
    cuda_check(cudaMemcpy(h_counts,
                          d_counts,
                          hydro::per_material::kPerMaterialCounterCount *
                              sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "per-material Q_ei copy counts failed");
    state.dispatch_counters.eos_inverse_call_count.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[hydro::per_material::kPerMaterialCounterEOSInverse]),
        std::memory_order_relaxed);
    state.dispatch_counters.lazy_cache_te_m_hit_count.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[hydro::per_material::kPerMaterialCounterLazyCacheHit]),
        std::memory_order_relaxed);
    state.dispatch_counters.lazy_cache_te_m_miss_count.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[hydro::per_material::kPerMaterialCounterLazyCacheMiss]),
        std::memory_order_relaxed);
    state.dispatch_counters.eos_table_validity_violations.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[hydro::per_material::kPerMaterialCounterEOSTableValidityViolation]),
        std::memory_order_relaxed);
    state.dispatch_counters.presence_absent_events.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[hydro::per_material::kPerMaterialCounterPresenceAbsent]),
        std::memory_order_relaxed);

    hydro::per_material::refresh_per_material_derived_cell_fields(
        state, cfg, eos_ctx, true);

    return;
  }

  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "apply_qei_coupling_substep requires at least one non-void material");
  const auto& mat0 = materials[static_cast<std::size_t>(first_nonvoid)];
  const bool has_table_eos =
      mat0.eos_tables != nullptr && !use_exact_ideal_gas_hydro_backend(mat0);

  std::vector<double> A_eff;
  std::vector<double> gamma_eff;
  compute_effective_A_gamma(cfg, state, n_cells, A_eff, gamma_eff, nullptr);
  TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                "apply_qei_coupling_substep A_eff size mismatch");
  TENRYU_ASSERT(gamma_eff.size() == static_cast<std::size_t>(n_cells),
                "apply_qei_coupling_substep gamma_eff size mismatch");

  bool any_cv_e_override = false;
  for (const auto& mat : materials) {
    if (mat.cv_e_override > 0.0) {
      any_cv_e_override = true;
      break;
    }
  }
  const bool use_first_cv_override =
      any_cv_e_override && mat0.cv_e_override > 0.0;

  SourceEOSTableViews table_views;
  if (has_table_eos) {
    table_views = source_eos_table_cache().views_for(*mat0.eos_tables);
    TENRYU_ASSERT(table_views.electron.n_rho > 0 && table_views.ion.n_rho > 0,
                  "apply_qei_coupling_substep requires non-empty device EOS tables");
  }

  TENRYU_ASSERT(state.cv_e.empty() ||
                    state.cv_e.size() == static_cast<std::size_t>(n_cells),
                "apply_qei_coupling_substep cv_e size mismatch");
  TENRYU_ASSERT(state.cv_i.empty() ||
                    state.cv_i.size() == static_cast<std::size_t>(n_cells),
                "apply_qei_coupling_substep cv_i size mismatch");
  TENRYU_ASSERT(state.hydro_active.empty() ||
                    state.hydro_active.size() == static_cast<std::size_t>(n_cells),
                "apply_qei_coupling_substep hydro_active size mismatch");

  double* d_A_eff = nullptr;
  double* d_gamma_eff = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;
  std::int8_t* d_hydro_active = nullptr;

  const std::size_t cell_bytes =
      sizeof(double) * static_cast<std::size_t>(n_cells);
  const std::size_t void_bytes =
      sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
  d_A_eff = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:apply_qei_coupling_substep:d_A_eff", cell_bytes));
  d_gamma_eff = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:apply_qei_coupling_substep:d_gamma_eff", cell_bytes));
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source_terms:apply_qei_coupling_substep:d_cell_is_void", void_bytes));
  cuda_check(cudaMemcpy(d_A_eff, A_eff.data(), cell_bytes, cudaMemcpyHostToDevice),
             "apply_qei_coupling_substep copy A_eff failed");
  cuda_check(cudaMemcpy(d_gamma_eff, gamma_eff.data(), cell_bytes,
                        cudaMemcpyHostToDevice),
             "apply_qei_coupling_substep copy gamma_eff failed");
  cuda_check(cudaMemcpy(d_cell_is_void, state.cell_is_void.data(), void_bytes,
                        cudaMemcpyHostToDevice),
             "apply_qei_coupling_substep copy cell_is_void failed");

  d_hydro_active = const_cast<std::int8_t*>(state.hydro_active_device_ptr());

  const int threads = 256;
  const int blocks = (n_cells + threads - 1) / threads;
  qei_coupling_substep_kernel<<<blocks, threads>>>(
      state.ee.data(), state.ei.data(), state.Te.data(), state.Ti.data(),
      state.Pe.data(), state.Pi.data(),
      state.cv_e.empty() ? nullptr : state.cv_e.data(),
      state.cv_i.empty() ? nullptr : state.cv_i.data(), state.rho.data(),
      state.zbar.data(), d_A_eff, d_gamma_eff, d_cell_is_void, d_hydro_active,
      n_cells, dt_sub, cfg.numerics.floors.Te, cfg.numerics.floors.Ti,
      has_table_eos, use_first_cv_override, mat0.cv_e_override,
      mat0.eos_T_ref_eV, cfg.numerics.hydro.qei_multiplier,
      table_views.ion, table_views.electron);
  cuda_check(cudaGetLastError(),
             "apply_qei_coupling_substep kernel launch failed");
  cuda_check(core::debug_kernel_sync(),
             "apply_qei_coupling_substep kernel synchronize failed");

}

double inject_laser_source_terms(core::State& state,
                                 const core::Config& cfg,
                                 const double dt,
                                 double* E_floor_injected,
                                 int* clamp_count) {
  if (state.laser_dep.empty() || state.rho.empty() || dt <= 0.0) {
    return 0.0;
  }
  if (cfg.materials.materials.empty()) {
    return 0.0;
  }
  assert_common_source_state_sizes(state, "inject_laser_source_terms");
  TENRYU_ASSERT(state.laser_dep.size() == state.rho.size(),
                "inject_laser_source_terms requires laser_dep/rho size match");

  const auto& materials = cfg.materials.materials;
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "inject_laser_source_terms requires at least one non-void material");
  const auto& mat0 = materials[static_cast<std::size_t>(first_nonvoid)];
  const bool has_table_eos =
      mat0.eos_tables != nullptr && !use_exact_ideal_gas_hydro_backend(mat0);
  bool any_cv_e_override = false;
  for (const auto& mat : materials) {
    if (mat.cv_e_override > 0.0) {
      any_cv_e_override = true;
      break;
    }
  }
  const bool use_first_cv_override = any_cv_e_override && mat0.cv_e_override > 0.0;

  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<double> A_eff;
  std::vector<double> gamma_eff;
  compute_effective_A_gamma(cfg, state, n_cells, A_eff, gamma_eff, nullptr);
  TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_laser_source_terms A_eff size mismatch");
  TENRYU_ASSERT(gamma_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_laser_source_terms gamma_eff size mismatch");

  const double te_floor = cfg.numerics.floors.Te;
  const double ti_floor = cfg.numerics.floors.Ti;

  std::vector<double> rho(state.rho.size(), 0.0);
  std::vector<double> zbar(state.zbar.size(), 0.0);
  std::vector<double> vol(state.vol.size(), 0.0);
  std::vector<double> ee(state.ee.size(), 0.0);
  std::vector<double> Te(state.Te.size(), 0.0);
  std::vector<double> ei(state.ei.size(), 0.0);
  std::vector<double> Ti(state.Ti.size(), 0.0);
  std::vector<double> Pe(state.Pe.size(), 0.0);
  std::vector<double> Pi(state.Pi.size(), 0.0);
  std::vector<double> laser_dep(state.laser_dep.size(), 0.0);
  std::vector<double> host_cv_e;

  const bool has_table_cv_e = !state.cv_e.empty();
  if (has_table_cv_e) {
    host_cv_e.resize(state.cv_e.size());
  }

  // Pack order: ee, Te, Pe, ei, Ti, Pi, rho, zbar, vol, laser_dep,
  // and conditional cv_e last. The first six entries are reused as the
  // writeback destination table.
  constexpr int kLaserInjectPointerSlots = 16;
  constexpr int kLaserInjectWritebackSlots = 6;
  const int n_pack_slots = has_table_cv_e ? 11 : 10;
  const std::size_t n_cells_size = static_cast<std::size_t>(n_cells);
  const std::size_t cell_bytes = n_cells_size * sizeof(double);
  const std::size_t pack_values =
      static_cast<std::size_t>(n_pack_slots) * n_cells_size;
  auto& staging = laser_inject_staging();
  staging.d_pack = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:laser_inject_pack", pack_values * sizeof(double)));
  if (n_cells_size > staging.capacity_cells) {
    staging.capacity_cells = n_cells_size;
  }
  if (staging.h_pack.size() < pack_values) {
    staging.h_pack.resize(pack_values);
  }

  double* host_ptrs[kLaserInjectPointerSlots] = {};
  host_ptrs[0] = state.ee.data();
  host_ptrs[1] = state.Te.data();
  host_ptrs[2] = state.Pe.data();
  host_ptrs[3] = state.ei.data();
  host_ptrs[4] = state.Ti.data();
  host_ptrs[5] = state.Pi.data();
  host_ptrs[6] = state.rho.data();
  host_ptrs[7] = state.zbar.data();
  host_ptrs[8] = state.vol.data();
  host_ptrs[9] = state.laser_dep.data();
  if (has_table_cv_e) {
    host_ptrs[10] = state.cv_e.data();
  }
  auto** const d_ptrs = static_cast<double**>(core::device_scratch_acquire(
      "source_terms:laser_inject_ptrs",
      kLaserInjectPointerSlots * sizeof(double*)));
  cuda_check(cudaMemcpy(d_ptrs, host_ptrs, sizeof(host_ptrs),
                        cudaMemcpyHostToDevice),
             "inject_laser_source_terms copy pointer table failed");

  const int threads = 256;
  const int pack_blocks =
      (n_pack_slots * n_cells + threads - 1) / threads;
  pack_fields_kernel<<<pack_blocks, threads>>>(
      staging.d_pack, reinterpret_cast<const double* const*>(d_ptrs),
      n_pack_slots, n_cells);
  cuda_check(cudaGetLastError(),
             "inject_laser_source_terms pack kernel launch failed");
  cuda_check(cudaMemcpy(staging.h_pack.data(), staging.d_pack,
                        pack_values * sizeof(double), cudaMemcpyDeviceToHost),
             "inject_laser_source_terms packed D2H failed");

  std::memcpy(ee.data(), staging.h_pack.data() + 0 * n_cells_size, cell_bytes);
  std::memcpy(Te.data(), staging.h_pack.data() + 1 * n_cells_size, cell_bytes);
  std::memcpy(Pe.data(), staging.h_pack.data() + 2 * n_cells_size, cell_bytes);
  std::memcpy(ei.data(), staging.h_pack.data() + 3 * n_cells_size, cell_bytes);
  std::memcpy(Ti.data(), staging.h_pack.data() + 4 * n_cells_size, cell_bytes);
  std::memcpy(Pi.data(), staging.h_pack.data() + 5 * n_cells_size, cell_bytes);
  std::memcpy(rho.data(), staging.h_pack.data() + 6 * n_cells_size, cell_bytes);
  std::memcpy(zbar.data(), staging.h_pack.data() + 7 * n_cells_size, cell_bytes);
  std::memcpy(vol.data(), staging.h_pack.data() + 8 * n_cells_size, cell_bytes);
  std::memcpy(laser_dep.data(), staging.h_pack.data() + 9 * n_cells_size,
              cell_bytes);
  if (has_table_cv_e) {
    std::memcpy(host_cv_e.data(),
                staging.h_pack.data() + 10 * n_cells_size, cell_bytes);
  }

  const bool use_two_temp = cfg.main.two_temperature;
  double floor_energy = 0.0;
  int local_clamp_count = 0;
  double skipped_energy = 0.0;
  int redirected_void_cells_with_laser = 0;
  double redirected_void_energy = 0.0;
  double unrecoverable_void_energy = 0.0;
  static bool warned_negative_laser_dep = false;

  for (int c = 0; c < n_cells; ++c) {
    double dep = laser_dep[static_cast<std::size_t>(c)];
    if (dep < 0.0) {
      if (!warned_negative_laser_dep) {
        core::log_warning("inject_laser_source_terms: negative laser_dep detected (cell=" +
                          std::to_string(c) + ", value=" + std::to_string(dep) +
                          "); clamping to 0.");
        warned_negative_laser_dep = true;
      }
      dep = 0.0;
    }
    laser_dep[static_cast<std::size_t>(c)] = dep;
  }

  // Redirect laser deposition from void cells to nearest inward non-void cell.
  double carry_dep = 0.0;
  for (int c = n_cells - 1; c >= 0; --c) {
    const std::size_t c_idx = static_cast<std::size_t>(c);
    const double dep = laser_dep[c_idx];
    if (state.cell_is_void[c_idx] != 0U) {
      if (dep > 0.0) {
        ++redirected_void_cells_with_laser;
        redirected_void_energy += dep;
      }
      carry_dep += dep;
      laser_dep[c_idx] = 0.0;
      continue;
    }
    if (carry_dep > 0.0) {
      laser_dep[c_idx] += carry_dep;
      carry_dep = 0.0;
    }
  }
  if (carry_dep > 0.0) {
    unrecoverable_void_energy = carry_dep;
    skipped_energy += carry_dep;
  }

  for (int c = 0; c < n_cells; ++c) {
    const double dep = laser_dep[static_cast<std::size_t>(c)];
    const std::size_t c_idx = static_cast<std::size_t>(c);
    if (state.cell_is_void[c_idx] != 0U) {
      continue;
    }
    const double rho_c = rho[c_idx];
    const double vol_c = vol[c_idx];
    const double denom = rho_c * vol_c;
    if (!(denom > 1.0e-30)) {
      skipped_energy += dep;
      continue;
    }

    const double A_c = A_eff[c_idx];
    const double gm1_c = gamma_eff[c_idx] - 1.0;
    const double cv_mass_i =
        std::max(core::constants::eV_to_erg /
                     (A_c * core::constants::proton_mass * gm1_c),
                 1.0e-30);
    if (!use_two_temp) {
      // 1T closure: ee stores total internal energy.
      ee[c_idx] += ei[c_idx];
    }
    ee[c_idx] += dep / denom;
    const double ee_before_floor = ee[c_idx];

    const double rho_safe = std::max(rho_c, 1.0e-30);
    const bool use_table_eos_closure = use_two_temp && has_table_eos;
    double Te_raw = 0.0;
    double Te_new = te_floor;
    if (use_table_eos_closure) {
      Te_raw = mat0.eos_tables->electron.temperature_from_energy(rho_safe, ee[c_idx]);
      if (!std::isfinite(Te_raw) || Te_raw < te_floor) {
        Te_new = te_floor;
        ee[c_idx] = mat0.eos_tables->electron.energy(rho_safe, Te_new);
      } else {
        Te_new = Te_raw;
      }
      Pe[c_idx] = mat0.eos_tables->electron.pressure(rho_safe, Te_new);
    } else if (use_first_cv_override && mat0.eos_T_ref_eV > 0.0) {
      const double T_ref = mat0.eos_T_ref_eV;
      const double T_ref3 = T_ref * T_ref * T_ref;
      const double alpha0 = mat0.cv_e_override / (4.0 * T_ref3);
      const double arg = ee[c_idx] * rho_safe / alpha0;
      Te_raw = (arg > 0.0) ? std::pow(arg, 0.25) : 0.0;
      if (!std::isfinite(Te_raw)) {
        Te_raw = 0.0;
      }
      Te_new = std::max(Te_raw, te_floor);
      const double Te4 = Te_new * Te_new * Te_new * Te_new;
      ee[c_idx] = alpha0 * Te4 / rho_safe;
    } else {
      const double z = std::max(zbar[c_idx], 0.0);
      double cv_mass_e = 0.0;
      if (use_first_cv_override) {
        cv_mass_e = mat0.cv_e_override / rho_safe;
      } else if (has_table_cv_e && host_cv_e[c_idx] > 0.0) {
        cv_mass_e = host_cv_e[c_idx];
      } else {
        cv_mass_e = z * core::constants::eV_to_erg /
                    (A_c * core::constants::proton_mass * gm1_c);
      }
      cv_mass_e = std::max(cv_mass_e, 1.0e-30);
      const double cv_mass_total = use_two_temp
                                       ? cv_mass_e
                                       : (use_first_cv_override
                                              ? cv_mass_e
                                              : std::max(cv_mass_e + cv_mass_i,
                                                         1.0e-30));
      Te_raw = ee[c_idx] / cv_mass_total;
      if (!std::isfinite(Te_raw)) {
        Te_raw = 0.0;
      }
      Te_new = std::max(Te_raw, te_floor);
      ee[c_idx] = cv_mass_total * Te_new;
    }

    if (Te_new > Te_raw) {
      ++local_clamp_count;
    }
    const double de_floor_e = ee[c_idx] - ee_before_floor;
    if (de_floor_e > 0.0) {
      floor_energy += rho_c * vol_c * de_floor_e;
    }

    Te[c_idx] = Te_new;
    if (!use_table_eos_closure) {
      Pe[c_idx] = gm1_c * rho_c * ee[c_idx];
    }
    if (use_two_temp) {
      if (has_table_eos) {
        const double Ti_raw_table =
            mat0.eos_tables->ion.temperature_from_energy(rho_safe, ei[c_idx]);
        const double ei_before_floor = ei[c_idx];
        if (!std::isfinite(Ti_raw_table) || Ti_raw_table < ti_floor) {
          ++local_clamp_count;
          Ti[c_idx] = ti_floor;
          ei[c_idx] = mat0.eos_tables->ion.energy(rho_safe, ti_floor);
        } else {
          Ti[c_idx] = Ti_raw_table;
        }
        Pi[c_idx] = mat0.eos_tables->ion.pressure(rho_safe, Ti[c_idx]);
        const double de_floor_i = ei[c_idx] - ei_before_floor;
        if (de_floor_i > 0.0) {
          floor_energy += rho_c * vol_c * de_floor_i;
        }
      } else {
        const double Ti_prev = std::isfinite(Ti[c_idx]) ? Ti[c_idx] : 0.0;
        if (Ti_prev < ti_floor) {
          ++local_clamp_count;
          const double ei_before_floor = ei[c_idx];
          Ti[c_idx] = ti_floor;
          ei[c_idx] = cv_mass_i * ti_floor;
          Pi[c_idx] = gm1_c * rho_c * ei[c_idx];
          const double de_floor_i = ei[c_idx] - ei_before_floor;
          if (de_floor_i > 0.0) {
            floor_energy += rho_c * vol_c * de_floor_i;
          }
        }
      }
    } else {
      // 1T closure convention: ee is total internal energy, ei/Pi are unused.
      Ti[c_idx] = Te_new;
      ei[c_idx] = 0.0;
      Pi[c_idx] = 0.0;
    }
  }

  if (redirected_void_cells_with_laser > 0 || unrecoverable_void_energy > 0.0) {
    // Compute total laser deposition (after redirect) for fraction check.
    double total_laser_dep = redirected_void_energy;
    for (int c = 0; c < n_cells; ++c) {
      total_laser_dep += laser_dep[static_cast<std::size_t>(c)];
    }
    const double frac =
        (total_laser_dep > 0.0) ? (redirected_void_energy / total_laser_dep) : 0.0;
    if (frac >= 0.05 || unrecoverable_void_energy > 0.0) {
      core::log_warning(
          "WARNING: laser energy in void cells redirected from " +
          std::to_string(redirected_void_cells_with_laser) + " cells (" +
          std::to_string(frac * 100.0) + "% of total), unrecoverable=" +
          std::to_string(unrecoverable_void_energy) + " erg");
    }
  }

  std::memcpy(staging.h_pack.data() + 0 * n_cells_size, ee.data(), cell_bytes);
  std::memcpy(staging.h_pack.data() + 1 * n_cells_size, Te.data(), cell_bytes);
  std::memcpy(staging.h_pack.data() + 2 * n_cells_size, Pe.data(), cell_bytes);
  std::memcpy(staging.h_pack.data() + 3 * n_cells_size, ei.data(), cell_bytes);
  std::memcpy(staging.h_pack.data() + 4 * n_cells_size, Ti.data(), cell_bytes);
  std::memcpy(staging.h_pack.data() + 5 * n_cells_size, Pi.data(), cell_bytes);
  cuda_check(cudaMemcpy(staging.d_pack, staging.h_pack.data(),
                        kLaserInjectWritebackSlots * cell_bytes,
                        cudaMemcpyHostToDevice),
             "inject_laser_source_terms packed H2D failed");
  const int unpack_blocks =
      (kLaserInjectWritebackSlots * n_cells + threads - 1) / threads;
  unpack_fields_kernel<<<unpack_blocks, threads>>>(
      d_ptrs, staging.d_pack, kLaserInjectWritebackSlots, n_cells);
  cuda_check(cudaGetLastError(),
             "inject_laser_source_terms unpack kernel launch failed");
  // No device sync: subsequent default-stream work observes this unpack in order.
  accumulate_floor_and_clamp(E_floor_injected, clamp_count, floor_energy, local_clamp_count);
  return skipped_energy;
}

// Deposit burn dE_e/dE_i [erg/cell] with laser host-mirror closure; returns skipped energy.
double inject_burn_source_terms(core::State& state,
                                const core::Config& cfg,
                                const std::vector<double>& dE_e,
                                const std::vector<double>& dE_i,
                                double* E_floor_injected,
                                int* clamp_count) {
  TENRYU_ASSERT(dE_e.size() == state.rho.size(),
                "inject_burn_source_terms requires dE_e/rho size match");
  TENRYU_ASSERT(dE_i.size() == state.rho.size(),
                "inject_burn_source_terms requires dE_i/rho size match");
  if (state.rho.empty()) {
    return 0.0;
  }
  if (cfg.materials.materials.empty()) {
    return 0.0;
  }
  assert_common_source_state_sizes(state, "inject_burn_source_terms");

  const auto& materials = cfg.materials.materials;
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "inject_burn_source_terms requires at least one non-void material");
  const auto& mat0 = materials[static_cast<std::size_t>(first_nonvoid)];
  const bool has_table_eos =
      mat0.eos_tables != nullptr && !use_exact_ideal_gas_hydro_backend(mat0);
  bool any_cv_e_override = false;
  for (const auto& mat : materials) {
    if (mat.cv_e_override > 0.0) {
      any_cv_e_override = true;
      break;
    }
  }
  const bool use_first_cv_override = any_cv_e_override && mat0.cv_e_override > 0.0;

  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<double> A_eff;
  std::vector<double> gamma_eff;
  compute_effective_A_gamma(cfg, state, n_cells, A_eff, gamma_eff, nullptr);
  TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_burn_source_terms A_eff size mismatch");
  TENRYU_ASSERT(gamma_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_burn_source_terms gamma_eff size mismatch");

  const double te_floor = cfg.numerics.floors.Te;
  const double ti_floor = cfg.numerics.floors.Ti;

  std::vector<double> rho(state.rho.size(), 0.0);
  std::vector<double> zbar(state.zbar.size(), 0.0);
  std::vector<double> vol(state.vol.size(), 0.0);
  std::vector<double> ee(state.ee.size(), 0.0);
  std::vector<double> Te(state.Te.size(), 0.0);
  std::vector<double> ei(state.ei.size(), 0.0);
  std::vector<double> Ti(state.Ti.size(), 0.0);
  std::vector<double> Pe(state.Pe.size(), 0.0);
  std::vector<double> Pi(state.Pi.size(), 0.0);
  std::vector<double> host_cv_e;

  state.rho.copy_to_host(rho.data());
  state.zbar.copy_to_host(zbar.data());
  state.vol.copy_to_host(vol.data());
  state.ee.copy_to_host(ee.data());
  state.Te.copy_to_host(Te.data());
  state.ei.copy_to_host(ei.data());
  state.Ti.copy_to_host(Ti.data());
  state.Pe.copy_to_host(Pe.data());
  state.Pi.copy_to_host(Pi.data());
  const bool has_table_cv_e = !state.cv_e.empty();
  if (has_table_cv_e) {
    host_cv_e.resize(state.cv_e.size());
    state.cv_e.copy_to_host(host_cv_e.data());
  }
  const std::vector<double> ee_entry = ee;
  const std::vector<double> ei_entry = ei;

  const bool use_two_temp = cfg.main.two_temperature;
  double floor_energy = 0.0;
  int local_clamp_count = 0;
  double skipped_energy = 0.0;

  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_idx = static_cast<std::size_t>(c);
    const double rho_c = rho[c_idx];
    const double vol_c = vol[c_idx];
    const double denom = rho_c * vol_c;
    const double dep_e = dE_e[c_idx];
    const double dep_i = dE_i[c_idx];
    if (state.cell_is_void[c_idx] != 0U || !(denom > 1.0e-30)) {
      skipped_energy += dep_e + dep_i;
      continue;
    }

    const double A_c = A_eff[c_idx];
    const double gm1_c = gamma_eff[c_idx] - 1.0;
    const double cv_mass_i =
        std::max(core::constants::eV_to_erg /
                     (A_c * core::constants::proton_mass * gm1_c),
                 1.0e-30);
    if (!use_two_temp) {
      // 1T closure: ee stores total internal energy.
      ee[c_idx] += ei[c_idx];
    }
    ee[c_idx] += (use_two_temp ? dep_e : (dep_e + dep_i)) / denom;
    const double ee_before_floor = ee[c_idx];

    const double rho_safe = std::max(rho_c, 1.0e-30);
    const bool use_table_eos_closure = use_two_temp && has_table_eos;
    double Te_raw = 0.0;
    double Te_new = te_floor;
    if (use_table_eos_closure) {
      Te_raw = mat0.eos_tables->electron.temperature_from_energy(rho_safe, ee[c_idx]);
      if (!std::isfinite(Te_raw) || Te_raw < te_floor) {
        Te_new = te_floor;
        ee[c_idx] = mat0.eos_tables->electron.energy(rho_safe, Te_new);
      } else {
        Te_new = Te_raw;
      }
      Pe[c_idx] = mat0.eos_tables->electron.pressure(rho_safe, Te_new);
    } else if (use_first_cv_override && mat0.eos_T_ref_eV > 0.0) {
      const double T_ref = mat0.eos_T_ref_eV;
      const double T_ref3 = T_ref * T_ref * T_ref;
      const double alpha0 = mat0.cv_e_override / (4.0 * T_ref3);
      const double arg = ee[c_idx] * rho_safe / alpha0;
      Te_raw = (arg > 0.0) ? std::pow(arg, 0.25) : 0.0;
      if (!std::isfinite(Te_raw)) {
        Te_raw = 0.0;
      }
      Te_new = std::max(Te_raw, te_floor);
      const double Te4 = Te_new * Te_new * Te_new * Te_new;
      ee[c_idx] = alpha0 * Te4 / rho_safe;
    } else {
      const double z = std::max(zbar[c_idx], 0.0);
      double cv_mass_e = 0.0;
      if (use_first_cv_override) {
        cv_mass_e = mat0.cv_e_override / rho_safe;
      } else if (has_table_cv_e && host_cv_e[c_idx] > 0.0) {
        cv_mass_e = host_cv_e[c_idx];
      } else {
        cv_mass_e = z * core::constants::eV_to_erg /
                    (A_c * core::constants::proton_mass * gm1_c);
      }
      cv_mass_e = std::max(cv_mass_e, 1.0e-30);
      const double cv_mass_total = use_two_temp
                                       ? cv_mass_e
                                       : (use_first_cv_override
                                              ? cv_mass_e
                                              : std::max(cv_mass_e + cv_mass_i,
                                                         1.0e-30));
      Te_raw = ee[c_idx] / cv_mass_total;
      if (!std::isfinite(Te_raw)) {
        Te_raw = 0.0;
      }
      Te_new = std::max(Te_raw, te_floor);
      ee[c_idx] = cv_mass_total * Te_new;
    }

    if (Te_new > Te_raw) {
      ++local_clamp_count;
    }
    const double de_floor_e = ee[c_idx] - ee_before_floor;
    if (de_floor_e > 0.0) {
      floor_energy += rho_c * vol_c * de_floor_e;
    }

    Te[c_idx] = Te_new;
    if (!use_table_eos_closure) {
      Pe[c_idx] = gm1_c * rho_c * ee[c_idx];
    }
    if (use_two_temp) {
      ei[c_idx] += dep_i / denom;
      if (has_table_eos) {
        const double Ti_raw_table =
            mat0.eos_tables->ion.temperature_from_energy(rho_safe, ei[c_idx]);
        const double ei_before_floor = ei[c_idx];
        if (!std::isfinite(Ti_raw_table) || Ti_raw_table < ti_floor) {
          ++local_clamp_count;
          Ti[c_idx] = ti_floor;
          ei[c_idx] = mat0.eos_tables->ion.energy(rho_safe, ti_floor);
        } else {
          Ti[c_idx] = Ti_raw_table;
        }
        Pi[c_idx] = mat0.eos_tables->ion.pressure(rho_safe, Ti[c_idx]);
        const double de_floor_i = ei[c_idx] - ei_before_floor;
        if (de_floor_i > 0.0) {
          floor_energy += rho_c * vol_c * de_floor_i;
        }
      } else {
        const double Ti_raw = ei[c_idx] / cv_mass_i;
        const double Ti_new = std::max(Ti_raw, ti_floor);
        const double ei_before_floor = ei[c_idx];
        Ti[c_idx] = Ti_new;
        if (Ti_new > Ti_raw) {
          ++local_clamp_count;
          ei[c_idx] = cv_mass_i * Ti_new;
          const double de_floor_i = ei[c_idx] - ei_before_floor;
          if (de_floor_i > 0.0) {
            floor_energy += rho_c * vol_c * de_floor_i;
          }
        }
        Pi[c_idx] = gm1_c * rho_c * ei[c_idx];
      }
    } else {
      // 1T closure convention: ee is total internal energy, ei/Pi are unused.
      Ti[c_idx] = Te_new;
      ei[c_idx] = 0.0;
      Pi[c_idx] = 0.0;
    }
  }

  const std::size_t n_mat_sz = materials.size();
  const bool per_material_deposit =
      cfg.numerics.materials.per_material_conservation_enabled &&
      cfg.main.two_temperature &&
      state.Ee_per_material.size() ==
          static_cast<std::size_t>(n_cells) * n_mat_sz &&
      state.Ei_per_material.size() ==
          static_cast<std::size_t>(n_cells) * n_mat_sz &&
      state.mass_per_material.size() ==
          static_cast<std::size_t>(n_cells) * n_mat_sz;
  if (per_material_deposit) {
    const std::size_t n_cell_mat = static_cast<std::size_t>(n_cells) * n_mat_sz;
    std::vector<double> mass_pm(n_cell_mat, 0.0);
    std::vector<double> Ee_pm(n_cell_mat, 0.0);
    std::vector<double> Ei_pm(n_cell_mat, 0.0);
    state.mass_per_material.copy_to_host(mass_pm.data());
    state.Ee_per_material.copy_to_host(Ee_pm.data());
    state.Ei_per_material.copy_to_host(Ei_pm.data());
    for (int c = 0; c < n_cells; ++c) {
      const std::size_t c_idx = static_cast<std::size_t>(c);
      const double mass_c = rho[c_idx] * vol[c_idx];
      if (!(mass_c > 1.0e-30)) {
        continue;
      }
      const double dE_cell_e = mass_c * (ee[c_idx] - ee_entry[c_idx]);
      const double dE_cell_i = mass_c * (ei[c_idx] - ei_entry[c_idx]);
      if (dE_cell_e == 0.0 && dE_cell_i == 0.0) {
        continue;
      }
      double sum_mass_m = 0.0;
      for (std::size_t m = 0; m < n_mat_sz; ++m) {
        const double mass_m = mass_pm[c_idx * n_mat_sz + m];
        if (mass_m > 0.0 && std::isfinite(mass_m)) {
          sum_mass_m += mass_m;
        }
      }
      if (!(sum_mass_m > 0.0)) {
        continue;
      }
      for (std::size_t m = 0; m < n_mat_sz; ++m) {
        const std::size_t idx = c_idx * n_mat_sz + m;
        const double mass_m = mass_pm[idx];
        if (!(mass_m > 0.0) || !std::isfinite(mass_m)) {
          continue;
        }
        const double share = mass_m / sum_mass_m;
        Ee_pm[idx] = std::max(Ee_pm[idx] + dE_cell_e * share, 0.0);
        Ei_pm[idx] = std::max(Ei_pm[idx] + dE_cell_i * share, 0.0);
      }
    }
    state.Ee_per_material.copy_from_host(Ee_pm.data());
    state.Ei_per_material.copy_from_host(Ei_pm.data());
    state.Te_per_material_valid.assign(n_cell_mat, static_cast<std::uint8_t>(0));
    state.Ti_per_material_valid.assign(n_cell_mat, static_cast<std::uint8_t>(0));
  }
  state.ee.copy_from_host(ee.data());
  state.Te.copy_from_host(Te.data());
  state.Pe.copy_from_host(Pe.data());
  state.ei.copy_from_host(ei.data());
  state.Ti.copy_from_host(Ti.data());
  state.Pi.copy_from_host(Pi.data());
  accumulate_floor_and_clamp(E_floor_injected, clamp_count, floor_energy, local_clamp_count);
  return skipped_energy;
}

}  // namespace tenryu::coupling
