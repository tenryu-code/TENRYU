#include "materials/opacity.cuh"

#include <cmath>

#include <cuda_runtime.h>

#include "core/device_error_flags.cuh"
#include "core/error.hpp"
#include "materials/opacity_eval.cuh"

namespace tenryu::materials {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

struct OpacityFlagsCache {
  tenryu::core::DeviceErrorFlags* d_flags = nullptr;

  ~OpacityFlagsCache() {
    if (d_flags != nullptr) {
      static_cast<void>(cudaFree(d_flags));
      d_flags = nullptr;
    }
  }
};

OpacityFlagsCache& opacity_flags_cache() {
  static OpacityFlagsCache cache;
  return cache;
}

template <typename Fn>
double integrate_simpson_log(const double x0, const double x1, Fn&& fn) {
  if (!(x1 > x0) || !(x0 > 0.0)) {
    return 0.0;
  }
  constexpr int n = 512;
  const double log_x0 = std::log(x0);
  const double log_x1 = std::log(x1);
  const double h = (log_x1 - log_x0) / static_cast<double>(n);
  double sum = 0.0;
  for (int i = 0; i <= n; ++i) {
    const double z = log_x0 + h * static_cast<double>(i);
    const double x = std::exp(z);
    const double w = (i == 0 || i == n) ? 1.0 : ((i & 1) ? 4.0 : 2.0);
    sum += w * fn(x) * x;
  }
  return sum * h / 3.0;
}

double planck_weight_host(const double nu_eV, const double T_eV) {
  if (!(nu_eV > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double x = nu_eV / T_eV;
  if (x > 700.0) {
    return nu_eV * nu_eV * nu_eV * std::exp(-x);
  }
  const double denom = std::expm1(x);
  if (!(denom > 0.0)) {
    return 0.0;
  }
  return (nu_eV * nu_eV * nu_eV) / denom;
}

double rosseland_weight_host(const double nu_eV, const double T_eV) {
  if (!(nu_eV > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double x = nu_eV / T_eV;
  const double T2 = T_eV * T_eV;
  if (x > 80.0) {
    return (nu_eV * nu_eV * nu_eV * nu_eV) * std::exp(-x) / T2;
  }
  const double ex = std::exp(x);
  const double em1 = std::expm1(x);
  if (!(em1 > 0.0)) {
    return 0.0;
  }
  return (nu_eV * nu_eV * nu_eV * nu_eV) * ex / (T2 * em1 * em1);
}

tenryu::core::DeviceErrorFlags copy_and_check_flags(
    const tenryu::core::DeviceErrorFlags* d_flags,
    const char* context) {
  tenryu::core::DeviceErrorFlags host_flags{};
  cuda_check(cudaMemcpy(&host_flags,
                        d_flags,
                        sizeof(host_flags),
                        cudaMemcpyDeviceToHost),
             context);
  if (host_flags.opacity_out_of_range != 0) {
    tenryu::core::log_warning(
        "compute_opacities: detected out-of-range rho; non-positive rho was clamped");
  }
  return host_flags;
}

__global__ void eval_opacity_power_law_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    double* __restrict__ sigma_a,
    double* __restrict__ sigma_R,
    const double power_law_kappa0,
    const double power_law_alpha_T,
    const double power_law_lambda_rho,
    const double power_law_T_ref_eV,
    const double power_law_rho_ref,
    const double kappa_floor,
    const double kappa_cap,
    const int n_cells,
    const int n_groups,
    tenryu::core::DeviceErrorFlags* __restrict__ error_flags) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double rho_c = rho[c];
  if (!(rho_c >= 0.0)) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->opacity_out_of_range, 1);
    }
  }

  const double rho_safe = (rho_c >= 0.0) ? rho_c : 0.0;
  const double sigma_min = rho_safe * fmax(kappa_floor, 0.0);
  const double sigma_max = rho_safe * fmax(kappa_cap, 0.0);
  double kappa = 0.0;
  if (rho_safe > 0.0) {
    kappa = power_law_kappa0 *
            pow(Te[c] / power_law_T_ref_eV, -power_law_alpha_T) *
            pow(rho_safe / power_law_rho_ref, power_law_lambda_rho);
  }
  const double sigma_c = apply_sigma_bounds(rho_safe * kappa, sigma_min, sigma_max);
  const int base = c * n_groups;
  for (int g = 0; g < n_groups; ++g) {
    sigma_a[base + g] = sigma_c;
    sigma_R[base + g] = sigma_c;
  }
}

}  // namespace

double FrequencyDependentOpacity::sigma_nu(const double nu_eV, const double T_eV) {
  if (!(nu_eV > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double x = nu_eV / T_eV;
  const double one_minus_exp = (x > 700.0) ? 1.0 : (-std::expm1(-x));
  return (27.0e9 / (nu_eV * nu_eV * nu_eV)) * one_minus_exp;
}

double FrequencyDependentOpacity::energy_lo(const int group) const {
  TENRYU_ASSERT(group >= 0 &&
                    group + 1 < static_cast<int>(bounds_eV_.size()),
                "FrequencyDependentOpacity group out of range");
  return bounds_eV_[static_cast<std::size_t>(group)];
}

double FrequencyDependentOpacity::energy_hi(const int group) const {
  TENRYU_ASSERT(group >= 0 &&
                    group + 1 < static_cast<int>(bounds_eV_.size()),
                "FrequencyDependentOpacity group out of range");
  return bounds_eV_[static_cast<std::size_t>(group + 1)];
}

double FrequencyDependentOpacity::kappa_planck(const int group,
                                               const double rho,
                                               const double T_eV) const {
  if (!(rho > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double lo = energy_lo(group);
  const double hi = energy_hi(group);
  const auto sigma_weighted = [T_eV](const double E) {
    return sigma_nu(E, T_eV) * planck_weight_host(E, T_eV);
  };
  const auto weight = [T_eV](const double E) { return planck_weight_host(E, T_eV); };
  const double num = integrate_simpson_log(lo, hi, sigma_weighted);
  const double den = integrate_simpson_log(lo, hi, weight);
  const double sigma_P = (den > 0.0) ? (num / den) : 0.0;
  return std::max(sigma_P, 0.0) / rho;
}

double FrequencyDependentOpacity::kappa_rosseland(const int group,
                                                  const double rho,
                                                  const double T_eV) const {
  if (!(rho > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double lo = energy_lo(group);
  const double hi = energy_hi(group);
  const auto wr = [T_eV](const double E) { return rosseland_weight_host(E, T_eV); };
  const auto wr_over_sigma = [T_eV](const double E) {
    const double sigma = sigma_nu(E, T_eV);
    return (sigma > 0.0) ? (rosseland_weight_host(E, T_eV) / sigma) : 0.0;
  };
  const double den = integrate_simpson_log(lo, hi, wr_over_sigma);
  const double num = integrate_simpson_log(lo, hi, wr);
  const double sigma_R = (den > 0.0) ? (num / den) : 0.0;
  return std::max(sigma_R, 0.0) / rho;
}

void evaluate_opacity_cuda(const OpacityEvalView& view,
                           tenryu::core::DeviceErrorFlags* host_flags_out) {
  if (view.opacity_model == kOpacityModelFreqDepMarshak) {
    evaluate_frequency_dependent_opacity_cuda(view, host_flags_out);
    return;
  }
  if (view.opacity_model == kOpacityModelPowerLaw) {
    evaluate_power_law_opacity_cuda(view, host_flags_out);
    return;
  }
  TENRYU_ASSERT(view.opacity_model != kOpacityModelTableNLTE,
                "evaluate_opacity_cuda does not support table_nlte; use NLTE coefficient path");
  evaluate_constant_opacity_cuda(view, host_flags_out);
}

void evaluate_constant_opacity_cuda(const OpacityEvalView& view,
                                    tenryu::core::DeviceErrorFlags* host_flags_out) {
  TENRYU_ASSERT(view.rho != nullptr, "evaluate_constant_opacity_cuda requires rho");
  TENRYU_ASSERT(view.Te != nullptr, "evaluate_constant_opacity_cuda requires Te");
  TENRYU_ASSERT(view.sigma_a != nullptr, "evaluate_constant_opacity_cuda requires sigma_a");
  TENRYU_ASSERT(view.sigma_R != nullptr, "evaluate_constant_opacity_cuda requires sigma_R");
  TENRYU_ASSERT(view.n_cells >= 0, "evaluate_constant_opacity_cuda requires n_cells >= 0");
  TENRYU_ASSERT(view.n_groups >= 1, "evaluate_constant_opacity_cuda requires n_groups >= 1");

  auto& cache = opacity_flags_cache();
  if (cache.d_flags == nullptr) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&cache.d_flags),
                          sizeof(*cache.d_flags)),
               "evaluate_constant_opacity_cuda cudaMalloc flags failed");
  }
  cuda_check(cudaMemset(cache.d_flags, 0, sizeof(*cache.d_flags)),
             "evaluate_constant_opacity_cuda cudaMemset flags failed");

  constexpr int kBlock = 256;
  const int grid = (view.n_cells + kBlock - 1) / kBlock;
  if (grid > 0) {
    eval_opacity_constant_kernel<<<grid, kBlock>>>(view.rho,
                                                   view.Te,
                                                   view.sigma_a,
                                                   view.sigma_R,
                                                   view.kappa_planck_const,
                                                   view.kappa_rosseland_const,
                                                   view.kappa_planck_cell,
                                                   view.kappa_rosseland_cell,
                                                   view.kappa_floor,
                                                   view.kappa_cap,
                                                   view.n_cells,
                                                   view.n_groups,
                                                   cache.d_flags);
    cuda_check(cudaGetLastError(),
               "evaluate_constant_opacity_cuda kernel launch failed");
  }

  const tenryu::core::DeviceErrorFlags host_flags =
      copy_and_check_flags(cache.d_flags,
                           "evaluate_constant_opacity_cuda copy flags failed");
  if (host_flags_out != nullptr) {
    *host_flags_out = host_flags;
  }
}

void evaluate_power_law_opacity_cuda(const OpacityEvalView& view,
                                     tenryu::core::DeviceErrorFlags* host_flags_out) {
  TENRYU_ASSERT(view.rho != nullptr, "evaluate_power_law_opacity_cuda requires rho");
  TENRYU_ASSERT(view.Te != nullptr, "evaluate_power_law_opacity_cuda requires Te");
  TENRYU_ASSERT(view.sigma_a != nullptr, "evaluate_power_law_opacity_cuda requires sigma_a");
  TENRYU_ASSERT(view.sigma_R != nullptr, "evaluate_power_law_opacity_cuda requires sigma_R");
  TENRYU_ASSERT(view.n_cells >= 0, "evaluate_power_law_opacity_cuda requires n_cells >= 0");
  TENRYU_ASSERT(view.n_groups >= 1, "evaluate_power_law_opacity_cuda requires n_groups >= 1");

  auto& cache = opacity_flags_cache();
  if (cache.d_flags == nullptr) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&cache.d_flags),
                          sizeof(*cache.d_flags)),
               "evaluate_power_law_opacity_cuda cudaMalloc flags failed");
  }
  cuda_check(cudaMemset(cache.d_flags, 0, sizeof(*cache.d_flags)),
             "evaluate_power_law_opacity_cuda cudaMemset flags failed");

  constexpr int kBlock = 256;
  const int grid = (view.n_cells + kBlock - 1) / kBlock;
  if (grid > 0) {
    eval_opacity_power_law_kernel<<<grid, kBlock>>>(view.rho,
                                                   view.Te,
                                                   view.sigma_a,
                                                   view.sigma_R,
                                                   view.power_law_kappa0,
                                                   view.power_law_alpha_T,
                                                   view.power_law_lambda_rho,
                                                   view.power_law_T_ref_eV,
                                                   view.power_law_rho_ref,
                                                   view.kappa_floor,
                                                   view.kappa_cap,
                                                   view.n_cells,
                                                   view.n_groups,
                                                   cache.d_flags);
    cuda_check(cudaGetLastError(),
               "evaluate_power_law_opacity_cuda kernel launch failed");
  }

  const tenryu::core::DeviceErrorFlags host_flags =
      copy_and_check_flags(cache.d_flags,
                           "evaluate_power_law_opacity_cuda copy flags failed");
  if (host_flags_out != nullptr) {
    *host_flags_out = host_flags;
  }
}

void evaluate_frequency_dependent_opacity_cuda(
    const OpacityEvalView& view,
    tenryu::core::DeviceErrorFlags* host_flags_out) {
  TENRYU_ASSERT(view.kappa_planck_cell == nullptr &&
                    view.kappa_rosseland_cell == nullptr,
                "freq_dep_marshak does not support per-material blending (I4 spec §5b guard)");
  TENRYU_ASSERT(view.rho != nullptr,
                "evaluate_frequency_dependent_opacity_cuda requires rho");
  TENRYU_ASSERT(view.Te != nullptr,
                "evaluate_frequency_dependent_opacity_cuda requires Te");
  TENRYU_ASSERT(view.group_bounds_eV != nullptr,
                "evaluate_frequency_dependent_opacity_cuda requires group bounds");
  TENRYU_ASSERT(view.sigma_a != nullptr,
                "evaluate_frequency_dependent_opacity_cuda requires sigma_a");
  TENRYU_ASSERT(view.sigma_R != nullptr,
                "evaluate_frequency_dependent_opacity_cuda requires sigma_R");
  TENRYU_ASSERT(view.n_cells >= 0,
                "evaluate_frequency_dependent_opacity_cuda requires n_cells >= 0");
  TENRYU_ASSERT(view.n_groups >= 1,
                "evaluate_frequency_dependent_opacity_cuda requires n_groups >= 1");

  auto& cache = opacity_flags_cache();
  if (cache.d_flags == nullptr) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&cache.d_flags),
                          sizeof(*cache.d_flags)),
               "evaluate_frequency_dependent_opacity_cuda cudaMalloc flags failed");
  }
  cuda_check(cudaMemset(cache.d_flags, 0, sizeof(*cache.d_flags)),
             "evaluate_frequency_dependent_opacity_cuda cudaMemset flags failed");

  constexpr int kBlock = 128;
  const int grid = (view.n_cells + kBlock - 1) / kBlock;
  if (grid > 0) {
    eval_opacity_freq_dep_marshak_kernel<<<grid, kBlock>>>(view.rho,
                                                           view.Te,
                                                           view.group_bounds_eV,
                                                           view.sigma_a,
                                                           view.sigma_R,
                                                           view.kappa_floor,
                                                           view.kappa_cap,
                                                           view.n_cells,
                                                           view.n_groups,
                                                           cache.d_flags);
    cuda_check(cudaGetLastError(),
               "evaluate_frequency_dependent_opacity_cuda kernel launch failed");
  }

  const tenryu::core::DeviceErrorFlags host_flags = copy_and_check_flags(
      cache.d_flags, "evaluate_frequency_dependent_opacity_cuda copy flags failed");
  if (host_flags_out != nullptr) {
    *host_flags_out = host_flags;
  }
}

}  // namespace tenryu::materials
