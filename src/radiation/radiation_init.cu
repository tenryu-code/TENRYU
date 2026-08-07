#include "radiation/radiation_init.cuh"

#include <cmath>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr int kBlock = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ double finite_or_zero(const double value) {
  return isfinite(value) ? value : 0.0;
}

__global__ void initialize_radiation_field_equilibrium_kernel(
    const double* __restrict__ Te,
    double* __restrict__ rad_E,
    double* __restrict__ rad_E_old,
    PlanckTableDeviceView planck,
    int n_cells,
    int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const double T = fmax(finite_or_zero(Te[c]), 0.0);
  const double T2 = T * T;
  const double b = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T), 0.0);
  const double E = core::constants::a_eV * T2 * T2 * b;
  rad_E[idx] = E;
  rad_E_old[idx] = E;
}

__global__ void initialize_radiation_field_planck_kernel(
    const double Tr_eV,
    double* __restrict__ rad_E,
    double* __restrict__ rad_E_old,
    PlanckTableDeviceView planck,
    int n_cells,
    int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  (void)c;
  const double T = fmax(finite_or_zero(Tr_eV), 0.0);
  const double T2 = T * T;
  const double b = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T), 0.0);
  const double E = core::constants::a_eV * T2 * T2 * b;
  rad_E[idx] = E;
  rad_E_old[idx] = E;
}

}  // namespace

void initialize_radiation_field_equilibrium_gpu(
    core::State& state,
    const PlanckTable& planck) {
  const int n_cells = static_cast<int>(state.Te.size());
  const int n_groups = planck.n_groups();
  TENRYU_ASSERT(n_cells >= 0,
                "initialize_radiation_field_equilibrium_gpu requires valid cell count");
  TENRYU_ASSERT(n_groups > 0,
                "initialize_radiation_field_equilibrium_gpu requires n_groups > 0");
  const std::size_t expected =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  TENRYU_ASSERT(state.rad_E.size() == expected,
                "initialize_radiation_field_equilibrium_gpu rad_E size mismatch");
  TENRYU_ASSERT(state.rad_E_old.size() == expected,
                "initialize_radiation_field_equilibrium_gpu rad_E_old size mismatch");
  if (expected == 0U) {
    return;
  }

  const int total = static_cast<int>(expected);
  const int blocks = (total + kBlock - 1) / kBlock;
  initialize_radiation_field_equilibrium_kernel<<<blocks, kBlock>>>(
      state.Te.data(),
      state.rad_E.data(),
      state.rad_E_old.data(),
      planck.device_view(),
      n_cells,
      n_groups);
  cuda_check(cudaGetLastError(),
             "initialize_radiation_field_equilibrium_gpu launch failed");
}

void initialize_radiation_field_planck_gpu(
    core::State& state,
    const PlanckTable& planck,
    const double Tr_eV) {
  const int n_cells = static_cast<int>(state.Te.size());
  const int n_groups = planck.n_groups();
  TENRYU_ASSERT(n_cells >= 0,
                "initialize_radiation_field_planck_gpu requires valid cell count");
  TENRYU_ASSERT(n_groups > 0,
                "initialize_radiation_field_planck_gpu requires n_groups > 0");
  TENRYU_ASSERT(Tr_eV > 0.0,
                "initialize_radiation_field_planck_gpu requires Tr_eV > 0");
  const std::size_t expected =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  TENRYU_ASSERT(state.rad_E.size() == expected,
                "initialize_radiation_field_planck_gpu rad_E size mismatch");
  TENRYU_ASSERT(state.rad_E_old.size() == expected,
                "initialize_radiation_field_planck_gpu rad_E_old size mismatch");
  if (expected == 0U) {
    return;
  }

  const int total = static_cast<int>(expected);
  const int blocks = (total + kBlock - 1) / kBlock;
  initialize_radiation_field_planck_kernel<<<blocks, kBlock>>>(
      Tr_eV,
      state.rad_E.data(),
      state.rad_E_old.data(),
      planck.device_view(),
      n_cells,
      n_groups);
  cuda_check(cudaGetLastError(),
             "initialize_radiation_field_planck_gpu launch failed");
}

void apply_initial_radiation_field(core::State& state,
                                   const core::Config& cfg) {
  if (!cfg.radiation.enabled) {
    return;
  }
  const std::string& field = cfg.geometry.radiation_field;
  if (field != "equilibrium" && field != "planck") {
    return;
  }
  auto planck_table = build_planck_table_from_config(cfg);
  if (field == "planck") {
    initialize_radiation_field_planck_gpu(state, planck_table,
                                          cfg.geometry.radiation_field_Tr_eV);
  } else {
    initialize_radiation_field_equilibrium_gpu(state, planck_table);
  }
}

}  // namespace tenryu::radiation
