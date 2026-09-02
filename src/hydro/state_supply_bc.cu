#include "hydro/state_supply_bc.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"

namespace tenryu::hydro {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ bool supply_for_cell(const int j,
                                const int nz,
                                const int bottom_active,
                                const int top_active,
                                const double bottom_rho,
                                const double bottom_uz,
                                const double bottom_T,
                                const double top_rho,
                                const double top_uz,
                                const double top_T,
                                double* rho,
                                double* uz,
                                double* T) {
  if (bottom_active != 0 && j == 0) {
    *rho = bottom_rho;
    *uz = bottom_uz;
    *T = bottom_T;
    return true;
  }
  if (top_active != 0 && j == nz - 1) {
    *rho = top_rho;
    *uz = top_uz;
    *T = top_T;
    return true;
  }
  return false;
}

__device__ void restore_supply_face_vz_for_cell(double* __restrict__ v_z,
                                                const int i,
                                                const int j,
                                                const int nr,
                                                const int nz,
                                                const double supply_uz) {
  const int stride = nz + 1;
  if (j == 0) {
    v_z[i * stride] = supply_uz;
    if (i == nr - 1) {
      v_z[(i + 1) * stride] = supply_uz;
    }
  } else if (j == nz - 1) {
    v_z[i * stride + nz] = supply_uz;
    if (i == nr - 1) {
      v_z[(i + 1) * stride + nz] = supply_uz;
    }
  }
}

__global__ void restore_state_supply_material_vz_kernel(
    double* __restrict__ v_z,
    const int nr,
    const int nz,
    const int bottom_active,
    const int top_active,
    const double bottom_uz,
    const double top_uz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nr) {
    return;
  }

  const int stride = nz + 1;
  if (bottom_active != 0) {
    v_z[i * stride] = bottom_uz;
  }
  if (top_active != 0) {
    v_z[i * stride + nz] = top_uz;
  }
}

__global__ void override_state_supply_kernel(
    const std::int8_t* __restrict__ mask,
    double* __restrict__ rho,
    double* __restrict__ mass,
    double* __restrict__ Te,
    double* __restrict__ Ti,
    const double* __restrict__ vol,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    double* __restrict__ v_z,
    double* __restrict__ pre_rho,
    double* __restrict__ pre_mass,
    double* __restrict__ pre_ee,
    double* __restrict__ pre_ei,
    double* __restrict__ pre_uz,
    double* __restrict__ totals,
    const int nr,
    const int nz,
    const int bottom_active,
    const int top_active,
    const double bottom_rho,
    const double bottom_uz,
    const double bottom_T,
    const double top_rho,
    const double top_uz,
    const double top_T) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells || mask[c] == 0) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  double supply_rho = 0.0;
  double supply_uz = 0.0;
  double supply_T = 0.0;
  if (!supply_for_cell(j, nz, bottom_active, top_active,
                       bottom_rho, bottom_uz, bottom_T,
                       top_rho, top_uz, top_T,
                       &supply_rho, &supply_uz, &supply_T)) {
    return;
  }

  const double mass_pre = mass[c];
  const double uz_pre = supply_uz;
  pre_rho[c] = rho[c];
  pre_mass[c] = mass_pre;
  pre_ee[c] = ee[c];
  pre_ei[c] = ei[c];
  pre_uz[c] = uz_pre;

  const double mass_post = supply_rho * vol[c];
  rho[c] = supply_rho;
  mass[c] = mass_post;
  Te[c] = supply_T;
  Ti[c] = supply_T;
  restore_supply_face_vz_for_cell(v_z, i, j, nr, nz, supply_uz);

  atomicAdd(totals + 0, mass_post - mass_pre);
  atomicAdd(totals + 1, mass_post * supply_uz - mass_pre * uz_pre);
}

__global__ void tally_state_supply_energy_kernel(
    const std::int8_t* __restrict__ mask,
    const double* __restrict__ mass,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ pre_mass,
    const double* __restrict__ pre_ee,
    const double* __restrict__ pre_ei,
    const double* __restrict__ pre_uz,
    double* __restrict__ total,
    const int nr,
    const int nz,
    const int bottom_active,
    const int top_active,
    const double bottom_uz,
    const double top_uz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells || mask[c] == 0) {
    return;
  }

  const int j = c - (c / nz) * nz;
  double supply_uz = 0.0;
  if (bottom_active != 0 && j == 0) {
    supply_uz = bottom_uz;
  } else if (top_active != 0 && j == nz - 1) {
    supply_uz = top_uz;
  } else {
    return;
  }

  const double e_post = ee[c] + ei[c] + 0.5 * supply_uz * supply_uz;
  const double e_pre = pre_ee[c] + pre_ei[c] + 0.5 * pre_uz[c] * pre_uz[c];
  atomicAdd(total, mass[c] * e_post - pre_mass[c] * e_pre);
}

std::int8_t* upload_mask(const char* pool_tag, const std::vector<std::int8_t>& mask) {
  if (mask.empty()) {
    return nullptr;
  }
  std::int8_t* d_mask = nullptr;
  const std::size_t mask_bytes = mask.size() * sizeof(std::int8_t);
  if (pool_tag != nullptr) {
    d_mask = static_cast<std::int8_t*>(core::device_scratch_acquire(pool_tag, mask_bytes));
  } else {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask), mask_bytes),
               "state_supply upload mask cudaMalloc failed");
  }
  cuda_check(cudaMemcpy(d_mask, mask.data(), mask.size() * sizeof(std::int8_t),
                        cudaMemcpyHostToDevice),
             "state_supply upload mask cudaMemcpy failed");
  return d_mask;
}

void ensure_state_supply_storage(core::State& state) {
  const std::size_t n_cells = state.rho.size();
  if (state.state_supply_mask.size() != n_cells) {
    state.state_supply_mask.assign(n_cells, static_cast<std::int8_t>(0));
  }
  if (state.state_supply_pre_rho.size() != n_cells) {
    state.state_supply_pre_rho.reset(n_cells);
  }
  if (state.state_supply_pre_mass.size() != n_cells) {
    state.state_supply_pre_mass.reset(n_cells);
  }
  if (state.state_supply_pre_ee.size() != n_cells) {
    state.state_supply_pre_ee.reset(n_cells);
  }
  if (state.state_supply_pre_ei.size() != n_cells) {
    state.state_supply_pre_ei.reset(n_cells);
  }
  if (state.state_supply_pre_uz.size() != n_cells) {
    state.state_supply_pre_uz.reset(n_cells);
  }
}

}  // namespace

bool has_state_supply_bc(const core::Config& cfg) {
  return cfg.main.dim == 2 && cfg.numerics.hydro.boundary_2d.has_any_state_supply();
}

void init_state_supply_mask(core::State& state, const core::Config& cfg) {
  ensure_state_supply_storage(state);
  std::fill(state.state_supply_mask.begin(), state.state_supply_mask.end(),
            static_cast<std::int8_t>(0));
  if (!has_state_supply_bc(cfg)) {
    return;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(nr > 0 && nz > 0, "state_supply requires positive 2D mesh topology");
  TENRYU_ASSERT(state.state_supply_mask.size() == static_cast<std::size_t>(nr * nz),
                "state_supply mask size mismatch");

  const auto& b = cfg.numerics.hydro.boundary_2d;
  for (int i = 0; i < nr; ++i) {
    if (b.z_bottom_cfg.is_state_supply()) {
      state.state_supply_mask[static_cast<std::size_t>(i * nz)] = 1;
    }
    if (b.z_top_cfg.is_state_supply()) {
      state.state_supply_mask[static_cast<std::size_t>(i * nz + (nz - 1))] = 1;
    }
  }
}

void reset_state_supply_step_tallies(core::State& state) {
  state.state_supply_dM_step = 0.0;
  state.state_supply_dE_step = 0.0;
  state.state_supply_dPz_step = 0.0;
}

void reset_state_supply_all_tallies(core::State& state) {
  reset_state_supply_step_tallies(state);
  state.state_supply_dM_cumulative = 0.0;
  state.state_supply_dE_cumulative = 0.0;
  state.state_supply_dPz_cumulative = 0.0;
}

void apply_state_supply_zonal_override(core::State& state,
                                       const core::Config& cfg,
                                       const bool accumulate_tally) {
  if (!has_state_supply_bc(cfg) || state.rho.empty()) {
    return;
  }
  init_state_supply_mask(state, cfg);

  std::int8_t* d_mask = upload_mask("ssbc:mask_override", state.state_supply_mask);
  double* d_totals = nullptr;
  d_totals =
      static_cast<double*>(core::device_scratch_acquire("ssbc:totals", 2 * sizeof(double)));
  cuda_check(cudaMemset(d_totals, 0, 2 * sizeof(double)),
             "state_supply totals cudaMemset failed");

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  const int blocks = (n_cells + 255) / 256;
  const auto& b = cfg.numerics.hydro.boundary_2d;
  override_state_supply_kernel<<<blocks, 256>>>(
      d_mask, state.rho.data(), state.mass.data(), state.Te.data(), state.Ti.data(),
      state.vol.data(), state.ee.data(), state.ei.data(), state.v_z.data(),
      state.state_supply_pre_rho.data(), state.state_supply_pre_mass.data(),
      state.state_supply_pre_ee.data(), state.state_supply_pre_ei.data(),
      state.state_supply_pre_uz.data(), d_totals, nr, nz,
      b.z_bottom_cfg.supply_active(state.t) ? 1 : 0,
      b.z_top_cfg.supply_active(state.t) ? 1 : 0,
      b.z_bottom_cfg.supply_rho_g_per_cc,
      b.z_bottom_cfg.supply_u_z_cm_per_s,
      b.z_bottom_cfg.supply_T_eV,
      b.z_top_cfg.supply_rho_g_per_cc,
      b.z_top_cfg.supply_u_z_cm_per_s,
      b.z_top_cfg.supply_T_eV);
  cuda_check(cudaGetLastError(), "state_supply override kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "state_supply override kernel execution failed");

  std::array<double, 2> totals{0.0, 0.0};
  cuda_check(cudaMemcpy(totals.data(), d_totals, 2 * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "state_supply totals cudaMemcpy failed");

  if (accumulate_tally) {
    state.state_supply_dM_step += totals[0];
    state.state_supply_dPz_step += totals[1];
    state.state_supply_dM_cumulative += totals[0];
    state.state_supply_dPz_cumulative += totals[1];
  }
}

void restore_state_supply_material_velocity(core::State& state,
                                            const core::Config& cfg) {
  if (!has_state_supply_bc(cfg) || state.v_z.empty()) {
    return;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int blocks = (nr + 1 + 255) / 256;
  const auto& b = cfg.numerics.hydro.boundary_2d;
  restore_state_supply_material_vz_kernel<<<blocks, 256>>>(
      state.v_z.data(), nr, nz,
      b.z_bottom_cfg.supply_active(state.t) ? 1 : 0,
      b.z_top_cfg.supply_active(state.t) ? 1 : 0,
      b.z_bottom_cfg.supply_u_z_cm_per_s,
      b.z_top_cfg.supply_u_z_cm_per_s);
  cuda_check(cudaGetLastError(), "state_supply material v_z kernel launch failed");
  cuda_check(core::debug_kernel_sync(),
             "state_supply material v_z kernel execution failed");
}

void tally_state_supply_delta_E(core::State& state,
                                const core::Config& cfg,
                                const bool accumulate_tally) {
  if (!has_state_supply_bc(cfg) || state.rho.empty()) {
    return;
  }

  std::int8_t* d_mask = upload_mask("ssbc:mask_tally", state.state_supply_mask);
  double* d_total = nullptr;
  d_total =
      static_cast<double*>(core::device_scratch_acquire("ssbc:total_e", sizeof(double)));
  cuda_check(cudaMemset(d_total, 0, sizeof(double)),
             "state_supply dE cudaMemset failed");

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  const int blocks = (n_cells + 255) / 256;
  const auto& b = cfg.numerics.hydro.boundary_2d;
  tally_state_supply_energy_kernel<<<blocks, 256>>>(
      d_mask, state.mass.data(), state.ee.data(), state.ei.data(),
      state.state_supply_pre_mass.data(), state.state_supply_pre_ee.data(),
      state.state_supply_pre_ei.data(), state.state_supply_pre_uz.data(),
      d_total, nr, nz,
      b.z_bottom_cfg.supply_active(state.t) ? 1 : 0,
      b.z_top_cfg.supply_active(state.t) ? 1 : 0,
      b.z_bottom_cfg.supply_u_z_cm_per_s,
      b.z_top_cfg.supply_u_z_cm_per_s);
  cuda_check(cudaGetLastError(), "state_supply dE kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "state_supply dE kernel execution failed");

  double dE = 0.0;
  cuda_check(cudaMemcpy(&dE, d_total, sizeof(double), cudaMemcpyDeviceToHost),
             "state_supply dE cudaMemcpy failed");

  if (accumulate_tally) {
    state.state_supply_dE_step += dE;
    state.state_supply_dE_cumulative += dE;
  }
}

}  // namespace tenryu::hydro
