#include "hydro/ale_axis_band_controller.cuh"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/axis_band_guard.hpp"
#include "hydro/axis_band_margin.cuh"
#include "hydro/axis_band_remap.cuh"
#include "hydro/eos_context.hpp"
#include "materials/eos_device_table.cuh"
#include "parallel/halo_exchange.hpp"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::ale {
namespace {

constexpr double kTinyRho = 1.0e-30;
constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = 1.6726219e-24;

__global__ void axis_band_eos_reclosure_kernel(
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ cv_e_out,
    double* __restrict__ cv_i_out,
    const double* __restrict__ mass,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const int K,
    const int nz,
    const double gamma,
    const double A,
    const double rho_floor,
    const double te_floor,
    const double ti_floor,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele,
    const bool low_density_extrapolation) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  const int band_cells = K * nz;
  if (b >= band_cells) {
    return;
  }
  const int c = b;

  const double z = fmax(zbar[c], 0.0);
  const double cv_i = kEvToErg / (A * kProtonMass * (gamma - 1.0));
  const double cv_e = z * kEvToErg / (A * kProtonMass * (gamma - 1.0));

  const double rho_c = fmax(rho[c], rho_floor);
  double e_i = fmax(ei[c], 0.0);
  double e_e = fmax(ee[c], 0.0);

  const bool use_ion_table =
      tab_ion.n_rho > 0 && tab_ion.n_T > 0 && tab_ion.e_table != nullptr &&
      tab_ion.P_table != nullptr && tab_ion.cv_table != nullptr &&
      tab_ion.supports_rho_e_reclosure != 0u;
  const bool use_ele_table =
      tab_ele.n_rho > 0 && tab_ele.n_T > 0 && tab_ele.e_table != nullptr &&
      tab_ele.P_table != nullptr && tab_ele.cv_table != nullptr &&
      tab_ele.supports_rho_e_reclosure != 0u;

  double Ti_c = ti_floor;
  double Te_c = te_floor;
  double Pi_c = 0.0;
  double Pe_c = 0.0;
  double cv_i_c = cv_i;
  double cv_e_c = cv_e;

  if (use_ion_table) {
    const auto inv = tenryu::materials::device_inverse_reclose_with_low_density_extrap(
        tab_ion, rho_c, e_i, ti_floor, 1.0, A, low_density_extrapolation);
    Ti_c = inv.T;
    Pi_c = inv.pressure;
    e_i = inv.energy;
    cv_i_c = inv.cv;
  } else if (cv_i > 0.0) {
    Ti_c = fmax(e_i / cv_i, ti_floor);
    e_i = cv_i * Ti_c;
    Pi_c = (gamma - 1.0) * rho_c * e_i;
  } else {
    e_i = 0.0;
    cv_i_c = 0.0;
  }

  if (use_ele_table) {
    const auto inv = tenryu::materials::device_inverse_reclose_with_low_density_extrap(
        tab_ele, rho_c, e_e, te_floor, z, A, low_density_extrapolation);
    Te_c = inv.T;
    Pe_c = inv.pressure;
    e_e = inv.energy;
    cv_e_c = inv.cv;
  } else if (cv_e > 0.0) {
    Te_c = fmax(e_e / cv_e, te_floor);
    e_e = cv_e * Te_c;
    Pe_c = (gamma - 1.0) * rho_c * e_e;
  } else {
    e_e = 0.0;
    cv_e_c = 0.0;
  }

  (void)mass;
  Te[c] = Te_c;
  Ti[c] = Ti_c;
  Pe[c] = Pe_c;
  Pi[c] = Pi_c;
  ee[c] = e_e;
  ei[c] = e_i;
  if (cv_e_out != nullptr) {
    cv_e_out[c] = cv_e_c;
  }
  if (cv_i_out != nullptr) {
    cv_i_out[c] = cv_i_c;
  }
}

int max_axis_band_K(const core::State& state, const core::Config& cfg) {
  if (state.mesh.dim != 2 || state.mesh.topo.nr < 2) {
    return 0;
  }
  return std::min(cfg.numerics.ale.axis_band_managed_remap_max_width,
                  state.mesh.topo.nr - 1);
}

void reclose_axis_band_eos(core::State& state,
                           const core::Config& cfg,
                           const HydroEOSContext* eos_ctx,
                           const int K) {
  if (K <= 0 || state.mesh.topo.nz <= 0 || cfg.materials.materials.empty()) {
    return;
  }

  const int n_cells = state.mesh.topo.nr * state.mesh.topo.nz;
  TENRYU_ASSERT(static_cast<int>(state.rho.size()) >= n_cells,
                "axis band controller EOS reclosure requires rho");
  TENRYU_ASSERT(static_cast<int>(state.mass.size()) >= n_cells,
                "axis band controller EOS reclosure requires mass");
  TENRYU_ASSERT(static_cast<int>(state.ee.size()) >= n_cells,
                "axis band controller EOS reclosure requires ee");
  TENRYU_ASSERT(static_cast<int>(state.ei.size()) >= n_cells,
                "axis band controller EOS reclosure requires ei");
  TENRYU_ASSERT(static_cast<int>(state.Te.size()) >= n_cells,
                "axis band controller EOS reclosure requires Te");
  TENRYU_ASSERT(static_cast<int>(state.Ti.size()) >= n_cells,
                "axis band controller EOS reclosure requires Ti");
  TENRYU_ASSERT(static_cast<int>(state.Pe.size()) >= n_cells,
                "axis band controller EOS reclosure requires Pe");
  TENRYU_ASSERT(static_cast<int>(state.Pi.size()) >= n_cells,
                "axis band controller EOS reclosure requires Pi");
  TENRYU_ASSERT(static_cast<int>(state.zbar.size()) >= n_cells,
                "axis band controller EOS reclosure requires zbar");

  const auto& mat = cfg.materials.materials.front();
  if (!(mat.ideal_gas_gamma > 1.0) || !(mat.A > 0.0)) {
    return;
  }

  tenryu::materials::DeviceEOSTableView ion_eos{};
  tenryu::materials::DeviceEOSTableView ele_eos{};
  if (eos_ctx != nullptr && eos_ctx->n_materials > 0) {
    ion_eos = eos_ctx->ion_view(0);
    ele_eos = eos_ctx->electron_view(0);
  }

  const int band_cells = K * state.mesh.topo.nz;
  const int blocks = (band_cells + 255) / 256;
  axis_band_eos_reclosure_kernel<<<blocks, 256>>>(
      state.Te.data(),
      state.Ti.data(),
      state.Pe.data(),
      state.Pi.data(),
      state.ee.data(),
      state.ei.data(),
      state.cv_e.empty() ? nullptr : state.cv_e.data(),
      state.cv_i.empty() ? nullptr : state.cv_i.data(),
      state.mass.data(),
      state.rho.data(),
      state.zbar.data(),
      K,
      state.mesh.topo.nz,
      mat.ideal_gas_gamma,
      mat.A,
      std::max(cfg.numerics.floors.rho, kTinyRho),
      cfg.numerics.floors.Te,
      cfg.numerics.floors.Ti,
      ion_eos,
      ele_eos,
      cfg.materials.low_density_extrapolation);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void exchange_axis_band_fields(core::State& state,
                               const parallel::PartitionInfo& part,
                               parallel::CommBuffers* bufs) {
  if (part.n_ranks <= 1 || bufs == nullptr) {
    return;
  }

  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  double* node_ptrs[4] = {
      state.x_r.data(), state.x_z.data(), state.v_r.data(), state.v_z.data()};
  parallel::exchange_node_fields(part, *bufs, node_ptrs, 4, n_nodes, nullptr, 6);

  std::vector<double*> cell_ptrs;
  cell_ptrs.reserve(9);
  if (static_cast<int>(state.vol.size()) == n_cells) {
    cell_ptrs.push_back(state.vol.data());
  }
  if (static_cast<int>(state.rho.size()) == n_cells) {
    cell_ptrs.push_back(state.rho.data());
  }
  if (static_cast<int>(state.mass.size()) == n_cells) {
    cell_ptrs.push_back(state.mass.data());
  }
  if (static_cast<int>(state.ee.size()) == n_cells) {
    cell_ptrs.push_back(state.ee.data());
  }
  if (static_cast<int>(state.ei.size()) == n_cells) {
    cell_ptrs.push_back(state.ei.data());
  }
  if (static_cast<int>(state.Te.size()) == n_cells) {
    cell_ptrs.push_back(state.Te.data());
  }
  if (static_cast<int>(state.Ti.size()) == n_cells) {
    cell_ptrs.push_back(state.Ti.data());
  }
  if (static_cast<int>(state.Pe.size()) == n_cells) {
    cell_ptrs.push_back(state.Pe.data());
  }
  if (static_cast<int>(state.Pi.size()) == n_cells) {
    cell_ptrs.push_back(state.Pi.data());
  }
  if (!cell_ptrs.empty()) {
    parallel::exchange_cell_fields(
        part, *bufs, cell_ptrs.data(), static_cast<int>(cell_ptrs.size()),
        n_cells, nullptr, 6);
  }
}

void copy_remap_diagnostics(AxisBandResult& out,
                            const AxisBandRemapResult& remap) {
  out.K_used = remap.K;
  out.mass_delta_rel = remap.mass_delta_rel;
  out.E_int_e_delta_rel = remap.E_int_e_delta_rel;
  out.E_int_i_delta_rel = remap.E_int_i_delta_rel;
  out.E_kin_delta_rel = remap.E_kin_delta_rel;
  out.E_rad_delta_rel = remap.E_rad_delta_rel;
}

}  // namespace

AxisBandDecision evaluate_axis_band_need(
    const core::State& state,
    const core::Config& cfg,
    const parallel::Reduction* reduction) {
  AxisBandDecision decision;
  decision.recommended_K = cfg.numerics.ale.axis_band_managed_remap_width;
  decision.margin_trigger =
      cfg.numerics.ale.axis_band_managed_remap_margin_trigger;

  if (!cfg.numerics.ale.axis_band_managed_remap_enabled ||
      !cfg.numerics.ale.axis_band_managed_remap_equal_volume ||
      !cfg.numerics.has_physical_rz_axis ||
      state.mesh.dim != 2) {
    return decision;
  }

  const int width = cfg.numerics.ale.axis_band_managed_remap_width;
  const int K_max = max_axis_band_K(state, cfg);
  if (width > K_max) {
    return decision;
  }

  const AxisBandKSelection sel =
      select_axis_band_K(state,
                         width,
                         K_max,
                         cfg.numerics.ale.axis_band_managed_remap_margin_trigger,
                         reduction);
  if (!sel.per_K_margins.empty()) {
    decision.row_K_margin = sel.per_K_margins.front();
  }
  if (!sel.any_K_healthy) {
    return decision;
  }

  decision.recommended_K = sel.K_chosen;
  decision.needed = decision.row_K_margin < 1.0;
  return decision;
}

AxisBandResult apply_axis_band_managed_remap(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const HydroEOSContext* eos_ctx,
    const int K_initial) {
  AxisBandResult out;
  if (!cfg.numerics.ale.axis_band_managed_remap_enabled ||
      !cfg.numerics.ale.axis_band_managed_remap_equal_volume ||
      !cfg.numerics.has_physical_rz_axis) {
    return out;
  }
  out.applied = true;
  out.skip_reason = AxisBandResult::SkipReason::AllKFailed;

  const int K_max = max_axis_band_K(state, cfg);
  if (K_max <= 0 || K_initial > K_max) {
    out.applied = false;
    out.skip_reason = AxisBandResult::SkipReason::NoOp;
    return out;
  }

  const int K_start = std::max(1, K_initial);
  AxisBandRemapResult last_remap;
  for (int K = K_start; K <= K_max; ++K) {
    AxisBandGuard guard;
    guard.capture(
        state, K,
        cfg.numerics.ale.axis_band_managed_remap_include_radiation_groups, nullptr);
    AxisBandRemapResult remap =
        apply_axis_band_remap(state, cfg, K, reduction);
    last_remap = remap;
    copy_remap_diagnostics(out, remap);
    if (remap.succeeded) {
      if (cfg.numerics.ale.transaction_failure_inject_point == K) {
        core::log_warning(
            std::string("[axis_band_inject] forced rollback at K=") + std::to_string(K));
        guard.restore(state, nullptr);
        out.skip_reason = AxisBandResult::SkipReason::RemapFailedPositivity;
        continue;
      }
      reclose_axis_band_eos(state, cfg, eos_ctx, K);
      exchange_axis_band_fields(state, part, bufs);
      const AxisBandMarginResult margin =
          compute_row_K_margin(state, K, reduction);
      out.remap_succeeded = true;
      out.K_used = K;
      out.min_axis_volume_rel_after = margin.valid ? margin.row_K_margin : 0.0;
      out.skip_reason = AxisBandResult::SkipReason::NoOp;
      guard.accept();
      return out;
    }

    guard.restore(state, nullptr);
    out.skip_reason = AxisBandResult::SkipReason::RemapFailedPositivity;
  }

  copy_remap_diagnostics(out, last_remap);
  out.remap_succeeded = false;
  out.skip_reason = AxisBandResult::SkipReason::AllKFailed;
  return out;
}

}  // namespace tenryu::hydro::ale
