#include "radiation/diffusion_source_solve.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr int kBlock = 256;
constexpr int kMaxEquilibriumNewtonIterations = 30;
constexpr int kMaxBacktracks = 8;
constexpr double kNewtonRelTol = 1.0e-10;
constexpr double kConservationRelTol = 1.0e-6;
constexpr double kDerivativeFloor = 1.0e-30;
constexpr double kEnergyScaleFloor = 1.0e-30;

struct SourceResidual {
  double R = 0.0;
  double dR_dT = 0.0;
  double scale = kEnergyScaleFloor;
  bool valid = false;
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ inline double finite_or_zero(const double x) {
  return isfinite(x) ? x : 0.0;
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

__device__ inline double planck_b_nonnegative(const PlanckTableDeviceView& planck,
                                              const int n_groups,
                                              const int g,
                                              const double T) {
  return (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T), 0.0);
}

__device__ inline SourceResidual evaluate_equilibrium_residual(const double m,
                                                               const double V,
                                                               const double cv_e,
                                                               const double T_old,
                                                               const double E_old_sum,
                                                               const double T) {
  SourceResidual out{};
  const double T2 = T * T;
  const double T3 = T2 * T;
  const double T4 = T2 * T2;
  const double matter_term = m * cv_e * (T - T_old);
  const double rad_term = V * (tenryu::core::constants::a_eV * T4 - E_old_sum);
  const double dR_dT =
      m * cv_e + V * 4.0 * tenryu::core::constants::a_eV * T3;

  if (!isfinite(T4) || !isfinite(matter_term) || !isfinite(rad_term) ||
      !(dR_dT > kDerivativeFloor) || !isfinite(dR_dT)) {
    return out;
  }

  out.R = matter_term + rad_term;
  const double residual_scale = fmax(fabs(matter_term), fabs(rad_term));
  const double reference_scale =
      fmax(m * cv_e * fmax(T, T_old), V * fmax(fabs(E_old_sum), 1e-30));
  out.scale = fmax(fmax(residual_scale, reference_scale), kEnergyScaleFloor);
  out.dR_dT = dR_dT;
  out.valid = isfinite(out.R) && isfinite(out.scale);
  return out;
}

__device__ inline void restore_source_cell_state(
    const DiffusionSourceSolveInputs& in,
    const int c,
    const double* __restrict__ diff_E_save,
    const double* __restrict__ ee_save,
    const double* __restrict__ Te_save,
    const double* __restrict__ Pe_save) {
  in.Te[c] = Te_save[c];
  in.ee[c] = ee_save[c];
  in.Pe[c] = Pe_save[c];
  for (int g = 0; g < in.n_groups; ++g) {
    const int idx = c * in.n_groups + g;
    in.diff_E[idx] = diff_E_save[idx];
  }
}

__global__ void diffusion_source_solve_kernel(DiffusionSourceSolveInputs in,
                                              const double* __restrict__ diff_E_save,
                                              const double* __restrict__ ee_save,
                                              const double* __restrict__ Te_save,
                                              const double* __restrict__ Pe_save,
                                              double* __restrict__ matter_delta_total,
                                              double* __restrict__ rad_delta_total,
                                              int* __restrict__ max_iter,
                                              int* __restrict__ n_failures) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= in.n_cells || in.diff_cell[c] == 0U) {
    return;
  }

  const double rho_c = finite_or_zero(in.rho[c]);
  const double V = finite_or_zero(in.vol[c]);
  const double m_from_state = (in.mass != nullptr) ? finite_or_zero(in.mass[c]) : 0.0;
  const double m = (m_from_state > 0.0) ? m_from_state : rho_c * V;
  double cv_e = (in.cv_e != nullptr) ? finite_or_zero(in.cv_e[c]) : 0.0;
  if (!(cv_e > 0.0)) {
    cv_e = finite_or_zero(in.cv_e_const);
  }

  const double T_floor = fmax(finite_or_zero(in.temperature_floor_eV), 1.0e-12);
  if (!(rho_c > 0.0) || !(V > 0.0) || !(m > 0.0) || !(cv_e > 0.0) ||
      !(in.dt_s > 0.0)) {
    restore_source_cell_state(in, c, diff_E_save, ee_save, Te_save, Pe_save);
    atomicAdd(n_failures, 1);
    return;
  }

  const double ee_old = finite_or_zero(ee_save[c]);
  const double T_old = finite_or_zero(Te_save[c]);
  double T = fmax(T_old, T_floor);
  double E_old_sum = 0.0;
  for (int g = 0; g < in.n_groups; ++g) {
    const int idx = c * in.n_groups + g;
    const double E_old = fmax(finite_or_zero(diff_E_save[idx]), 0.0);
    E_old_sum += E_old;
  }
  if (!isfinite(E_old_sum)) {
    restore_source_cell_state(in, c, diff_E_save, ee_save, Te_save, Pe_save);
    atomicAdd(n_failures, 1);
    return;
  }

  bool converged = false;
  int local_iters = 0;

  for (int iter = 0; iter < kMaxEquilibriumNewtonIterations; ++iter) {
    local_iters = iter + 1;

    const SourceResidual residual =
        evaluate_equilibrium_residual(m, V, cv_e, T_old, E_old_sum, T);
    if (!residual.valid) {
      break;
    }
    if (fabs(residual.R) <= kNewtonRelTol * residual.scale) {
      converged = true;
      break;
    }

    double delta_T = -residual.R / residual.dR_dT;
    delta_T = fmax(delta_T, -0.5 * fmax(T, T_floor));
    if (!isfinite(delta_T)) {
      break;
    }

    double T_next = fmax(T + delta_T, T_floor);
    SourceResidual trial =
        evaluate_equilibrium_residual(m, V, cv_e, T_old, E_old_sum, T_next);
    int backtracks = 0;
    while ((!trial.valid || fabs(trial.R) > fabs(residual.R)) &&
           backtracks < kMaxBacktracks) {
      delta_T *= 0.5;
      T_next = fmax(T + delta_T, T_floor);
      trial =
          evaluate_equilibrium_residual(m, V, cv_e, T_old, E_old_sum, T_next);
      ++backtracks;
    }
    if (!trial.valid || fabs(trial.R) > fabs(residual.R)) {
      break;
    }
    T = T_next;
    if (fabs(trial.R) <= kNewtonRelTol * trial.scale) {
      converged = true;
      break;
    }
  }

  if (!converged) {
    restore_source_cell_state(in, c, diff_E_save, ee_save, Te_save, Pe_save);
    atomicAdd(n_failures, 1);
    atomicMax(max_iter, local_iters);
    return;
  }
  atomicMax(max_iter, local_iters);

  const double ee_new = ee_old + cv_e * (T - T_old);
  const double matter_delta = m * (ee_new - ee_old);
  double rad_delta = 0.0;

  const double T2 = T * T;
  const double T4 = T2 * T2;
  double E_new_sum = 0.0;
  for (int g = 0; g < in.n_groups; ++g) {
    const int idx = c * in.n_groups + g;
    const double b_g = planck_b_nonnegative(in.planck, in.n_groups, g, T);
    const double B_g = tenryu::core::constants::a_eV * T4 * b_g;
    const double E_new = fmax(B_g, 0.0);
    in.diff_E[idx] = E_new;
    E_new_sum += E_new;
    if (!isfinite(B_g) || !isfinite(E_new) || !isfinite(E_new_sum)) {
      restore_source_cell_state(in, c, diff_E_save, ee_save, Te_save, Pe_save);
      atomicAdd(n_failures, 1);
      return;
    }
  }

  const double target_sum = tenryu::core::constants::a_eV * T4;
  if (!isfinite(target_sum)) {
    restore_source_cell_state(in, c, diff_E_save, ee_save, Te_save, Pe_save);
    atomicAdd(n_failures, 1);
    return;
  }
  if (E_new_sum > kEnergyScaleFloor && target_sum > kEnergyScaleFloor) {
    const double rescale = target_sum / E_new_sum;
    if (!isfinite(rescale)) {
      restore_source_cell_state(in, c, diff_E_save, ee_save, Te_save, Pe_save);
      atomicAdd(n_failures, 1);
      return;
    }
    for (int g = 0; g < in.n_groups; ++g) {
      in.diff_E[c * in.n_groups + g] *= rescale;
    }
  }

  rad_delta = 0.0;
  for (int g = 0; g < in.n_groups; ++g) {
    const int idx = c * in.n_groups + g;
    const double E_old = fmax(finite_or_zero(diff_E_save[idx]), 0.0);
    const double E_new = fmax(finite_or_zero(in.diff_E[idx]), 0.0);
    rad_delta += V * (E_new - E_old);
    if (!isfinite(E_new) || !isfinite(rad_delta)) {
      restore_source_cell_state(in, c, diff_E_save, ee_save, Te_save, Pe_save);
      atomicAdd(n_failures, 1);
      return;
    }
  }

  const double balance = matter_delta + rad_delta;
  const double balance_scale =
      fmax(fmax(fabs(matter_delta), fabs(rad_delta)), kEnergyScaleFloor);
  if (!isfinite(balance) || !isfinite(balance_scale) ||
      fabs(balance) > kConservationRelTol * balance_scale) {
    restore_source_cell_state(in, c, diff_E_save, ee_save, Te_save, Pe_save);
    atomicAdd(n_failures, 1);
    return;
  }

  in.Te[c] = T;
  in.ee[c] = ee_new;
  in.Pe[c] = fmax(in.pressure_gamma_minus_one, 0.0) * rho_c * ee_new;

  for (int g = 0; g < in.n_groups; ++g) {
    const int idx = c * in.n_groups + g;
    const double sigma_P = fmax(finite_or_zero(in.sigma_P[idx]), 0.0);
    const double b_g = planck_b_nonnegative(in.planck, in.n_groups, g, T);
    const double B_g = tenryu::core::constants::a_eV * T4 * b_g;
    const double E_new = fmax(finite_or_zero(in.diff_E[idx]), 0.0);
    const double rate = in.dt_s * tenryu::core::constants::c_light * sigma_P;
    in.diff_E[idx] = E_new;
    in.rad_dep[idx] += rate * E_new * V;
    in.rad_emit[idx] += rate * B_g * V;
  }
  atomic_add_double(matter_delta_total, matter_delta);
  atomic_add_double(rad_delta_total, rad_delta);
}

}  // namespace

DiffusionSourceSolveResult diffusion_source_solve_cuda(
    const DiffusionSourceSolveInputs& in) {
  DiffusionSourceSolveResult out{};
  if (in.n_cells <= 0) {
    return out;
  }

  TENRYU_ASSERT(!in.use_table_eos,
                "diffusion_source_solve_cuda table EOS path is deferred");
  TENRYU_ASSERT(in.n_groups > 0,
                "diffusion_source_solve_cuda requires n_groups > 0");
  TENRYU_ASSERT(in.dt_s > 0.0,
                "diffusion_source_solve_cuda requires dt_s > 0");
  TENRYU_ASSERT(in.diff_E != nullptr,
                "diffusion_source_solve_cuda requires diff_E");
  TENRYU_ASSERT(in.ee != nullptr, "diffusion_source_solve_cuda requires ee");
  TENRYU_ASSERT(in.Te != nullptr, "diffusion_source_solve_cuda requires Te");
  TENRYU_ASSERT(in.Pe != nullptr, "diffusion_source_solve_cuda requires Pe");
  TENRYU_ASSERT(in.sigma_P != nullptr,
                "diffusion_source_solve_cuda requires sigma_P");
  TENRYU_ASSERT(in.vol != nullptr, "diffusion_source_solve_cuda requires vol");
  TENRYU_ASSERT(in.rho != nullptr, "diffusion_source_solve_cuda requires rho");
  TENRYU_ASSERT(in.diff_cell != nullptr,
                "diffusion_source_solve_cuda requires diff_cell");
  TENRYU_ASSERT(in.rad_dep != nullptr,
                "diffusion_source_solve_cuda requires rad_dep");
  TENRYU_ASSERT(in.rad_emit != nullptr,
                "diffusion_source_solve_cuda requires rad_emit");

  const std::size_t n_cells = static_cast<std::size_t>(in.n_cells);
  const std::size_t n_groups = static_cast<std::size_t>(in.n_groups);
  const std::size_t n_total = n_cells * n_groups;
  const std::size_t cell_bytes = sizeof(double) * n_cells;
  const std::size_t energy_bytes = sizeof(double) * n_total;
  double* d_diff_E_save = nullptr;
  double* d_ee_save = nullptr;
  double* d_Te_save = nullptr;
  double* d_Pe_save = nullptr;
  double* d_matter_delta = nullptr;
  double* d_rad_delta = nullptr;
  int* d_max_iter = nullptr;
  int* d_failures = nullptr;
  d_diff_E_save = static_cast<double*>(core::device_scratch_acquire(
      "diffusion_source_solve:diff_E_save", energy_bytes));
  d_ee_save = static_cast<double*>(
      core::device_scratch_acquire("diffusion_source_solve:ee_save", cell_bytes));
  d_Te_save = static_cast<double*>(
      core::device_scratch_acquire("diffusion_source_solve:Te_save", cell_bytes));
  d_Pe_save = static_cast<double*>(
      core::device_scratch_acquire("diffusion_source_solve:Pe_save", cell_bytes));
  d_matter_delta = static_cast<double*>(
      core::device_scratch_acquire("diffusion_source_solve:matter_delta", sizeof(double)));
  d_rad_delta = static_cast<double*>(
      core::device_scratch_acquire("diffusion_source_solve:rad_delta", sizeof(double)));
  d_max_iter = static_cast<int*>(
      core::device_scratch_acquire("diffusion_source_solve:max_iter", sizeof(int)));
  d_failures = static_cast<int*>(
      core::device_scratch_acquire("diffusion_source_solve:failures", sizeof(int)));
  cuda_check(cudaMemcpy(d_diff_E_save, in.diff_E, energy_bytes, cudaMemcpyDeviceToDevice),
             "diffusion_source_solve_cuda copy diff_E save failed");
  cuda_check(cudaMemcpy(d_ee_save, in.ee, cell_bytes, cudaMemcpyDeviceToDevice),
             "diffusion_source_solve_cuda copy ee save failed");
  cuda_check(cudaMemcpy(d_Te_save, in.Te, cell_bytes, cudaMemcpyDeviceToDevice),
             "diffusion_source_solve_cuda copy Te save failed");
  cuda_check(cudaMemcpy(d_Pe_save, in.Pe, cell_bytes, cudaMemcpyDeviceToDevice),
             "diffusion_source_solve_cuda copy Pe save failed");
  cuda_check(cudaMemset(d_matter_delta, 0, sizeof(double)),
             "diffusion_source_solve_cuda zero matter delta failed");
  cuda_check(cudaMemset(d_rad_delta, 0, sizeof(double)),
             "diffusion_source_solve_cuda zero rad delta failed");
  cuda_check(cudaMemset(d_max_iter, 0, sizeof(int)),
             "diffusion_source_solve_cuda zero max_iter failed");
  cuda_check(cudaMemset(d_failures, 0, sizeof(int)),
             "diffusion_source_solve_cuda zero failures failed");

  const int grid = (in.n_cells + kBlock - 1) / kBlock;
  diffusion_source_solve_kernel<<<grid, kBlock>>>(in,
                                                  d_diff_E_save,
                                                  d_ee_save,
                                                  d_Te_save,
                                                  d_Pe_save,
                                                  d_matter_delta,
                                                  d_rad_delta,
                                                  d_max_iter,
                                                  d_failures);
  cuda_check(cudaGetLastError(),
             "diffusion_source_solve_cuda kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "diffusion_source_solve_cuda kernel execution failed");

  cuda_check(cudaMemcpy(&out.max_newton_iter,
                        d_max_iter,
                        sizeof(out.max_newton_iter),
                        cudaMemcpyDeviceToHost),
             "diffusion_source_solve_cuda copy max_iter failed");
  cuda_check(cudaMemcpy(&out.n_failures,
                        d_failures,
                        sizeof(out.n_failures),
                        cudaMemcpyDeviceToHost),
             "diffusion_source_solve_cuda copy failures failed");
  cuda_check(cudaMemcpy(&out.matter_delta,
                        d_matter_delta,
                        sizeof(out.matter_delta),
                        cudaMemcpyDeviceToHost),
             "diffusion_source_solve_cuda copy matter delta failed");
  cuda_check(cudaMemcpy(&out.rad_delta,
                        d_rad_delta,
                        sizeof(out.rad_delta),
                        cudaMemcpyDeviceToHost),
             "diffusion_source_solve_cuda copy rad delta failed");

  if (out.n_failures > 0) {
    core::log_warning("diffusion_source_solve_cuda: non-converged, "
                      "non-conserved, or invalid diffusion cells=" +
                      std::to_string(out.n_failures));
  }
  return out;
}

}  // namespace tenryu::radiation
