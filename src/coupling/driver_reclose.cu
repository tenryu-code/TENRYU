#include "coupling/driver_reclose.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "core/state.hpp"
#include "hydro/eos_context.hpp"
#include "materials/eos_table.hpp"
#include "materials/eos_table_reclose.cuh"

namespace tenryu::coupling {
namespace {

constexpr int kRecloseBlockSize = 256;

void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__global__ void tabular_eos_reclose_kernel(
    const materials::DeviceEOSTableView electron_table,
    const materials::DeviceEOSTableView ion_table,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::size_t cell_is_void_size,
    const int n_cells,
    const double* __restrict__ rho,
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    double* __restrict__ cv_e,
    double* __restrict__ cv_i,
    const double te_floor,
    const double ti_floor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  const std::size_t i_us = static_cast<std::size_t>(i);
  if (i_us < cell_is_void_size && cell_is_void[i_us] != 0U) {
    return;
  }

  const materials::RecloseThermo electron = materials::reclose_thermo_from_energy(
      electron_table, rho[i], ee[i], te_floor);
  Te[i] = electron.T;
  ee[i] = electron.energy;
  Pe[i] = electron.pressure;
  if (cv_e != nullptr) {
    cv_e[i] = electron.cv;
  }

  const materials::RecloseThermo ion =
      materials::reclose_thermo_from_energy(ion_table, rho[i], ei[i], ti_floor);
  Ti[i] = ion.T;
  ei[i] = ion.energy;
  Pi[i] = ion.pressure;
  if (cv_i != nullptr) {
    cv_i[i] = ion.cv;
  }
}

// Preserve the driver closure's existing conversion, without changing its value.
constexpr double kConductionRecloseEvToErg = 1.6022e-12;

struct ConductionEOSDeviceParams {
  materials::DeviceEOSTableView electron;
  materials::DeviceEOSTableView ion;
  materials::DeviceEOSTableView total;
  double electron_T_top;
  double ion_T_top;
  double total_T_top;
  double te_floor;
  double ti_floor;
  double gamma;
  double A;
  double cv_override;
  double T_ref;
  double cv_i_mass;
  bool two_temp;
  bool exact_ideal;
  bool energy_authoritative;
  bool use_total;
  // Per-cell dominant-material table selection and the matching per-material
  // table ceilings (nullptr => first-material tables/ceilings, see below).
  materials::CellEOSTableSelector cell_tables;
  const double* electron_T_top_by_material;
  const double* ion_T_top_by_material;
  const double* total_T_top_by_material;
};

struct ConductionTailThermo {
  double e;
  double P;
  double cv;
};

__device__ ConductionTailThermo conduction_thermo_with_tail(
    const materials::DeviceEOSTableView table, const double T_top,
    const double rho, const double T, const bool energy_authoritative) {
  if (energy_authoritative && std::isfinite(T_top) && T_top > 0.0 && T > T_top) {
    const double e_top = materials::reclose_energy<true>(table, rho, T_top);
    const double P_top = materials::reclose_pressure<true>(table, rho, T_top);
    const double cv_top = materials::reclose_max(materials::reclose_cv<true>(table, rho, T_top), 0.0);
    if (std::isfinite(e_top) && std::isfinite(P_top) && std::isfinite(cv_top) && cv_top > 0.0) {
      return {__dadd_rn(e_top, __dmul_rn(cv_top, T - T_top)), P_top * (T / T_top), cv_top};
    }
  }
  return {materials::reclose_energy<true>(table, rho, T),
          materials::reclose_pressure<true>(table, rho, T),
          materials::reclose_max(materials::reclose_cv<true>(table, rho, T), 0.0)};
}

__global__ void sync_conduction_eos_from_temperature_kernel(
    const ConductionEOSDeviceParams params, const int n,
    const std::uint8_t* cell_is_void, const std::size_t mask_size,
    const double* rho, const double* zbar, const double* Te, const double* Ti,
    double* ee, double* ei, double* Pe, double* Pi, double* cv_e, double* cv_i) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double e = 0.0, ion_e = 0.0, P = 0.0, ion_P = 0.0, cve = 0.0, cvi = 0.0;
  if (!(static_cast<std::size_t>(i) < mask_size && cell_is_void[i] != 0U)) {
    const double rho_safe = materials::reclose_max(rho[i], 1.0e-30);
    const double te = materials::reclose_max(Te[i], params.te_floor);
    const double ti = params.two_temp ? materials::reclose_max(Ti[i], params.ti_floor) : 0.0;
    if (params.exact_ideal) {
      const double z = materials::reclose_max(zbar[i], 0.0);
      const double cv_e_mass = params.cv_override > 0.0
          ? params.cv_override / rho_safe
          : z * kConductionRecloseEvToErg /
                (params.A * core::constants::proton_mass * (params.gamma - 1.0));
      const double cv_e_cell = materials::reclose_max(cv_e_mass, 0.0);
      const double cv_i_cell = materials::reclose_max(params.cv_i_mass, 0.0);
      cve = params.two_temp ? cv_e_cell : cv_e_cell + cv_i_cell;
      cvi = params.two_temp ? cv_i_cell : 0.0;
      if (params.two_temp) {
        e = cve * te;
        ion_e = cvi * ti;
        P = (params.gamma - 1.0) * rho[i] * e;
        ion_P = (params.gamma - 1.0) * rho[i] * ion_e;
      } else {
        if (params.T_ref > 0.0 && params.cv_override > 0.0) {
          const double T_ref3 = params.T_ref * params.T_ref * params.T_ref;
          const double alpha0 = params.cv_override / (4.0 * T_ref3);
          const double T4 = te * te * te * te;
          e = alpha0 * T4 / rho_safe;
        } else {
          e = (cve + cvi) * te;
        }
        P = (params.gamma - 1.0) * rho[i] * e;
      }
    } else {
      // Per-cell dominant-material table (multi-material closure): a cell
      // whose material has its own table uses it together with that table's
      // ceiling; otherwise the first non-void material's table and ceiling
      // (the historic behaviour) are kept.
      const int m = params.cell_tables.material_of(i);
      const materials::DeviceEOSTableView* e_views =
          params.use_total ? params.cell_tables.total_views
                           : params.cell_tables.electron_views;
      const double* e_tops = params.use_total ? params.total_T_top_by_material
                                              : params.electron_T_top_by_material;
      const bool e_per_cell =
          (m >= 0 && e_views != nullptr && e_tops != nullptr && e_views[m].n_rho > 0);
      const materials::DeviceEOSTableView table =
          e_per_cell ? e_views[m] : (params.use_total ? params.total : params.electron);
      const double T_top =
          e_per_cell ? e_tops[m]
                     : (params.use_total ? params.total_T_top : params.electron_T_top);
      const auto electron = conduction_thermo_with_tail(
          table, T_top, rho_safe, te, params.energy_authoritative);
      e = electron.e;
      P = electron.P;
      cve = electron.cv;
      if (params.two_temp) {
        const bool i_per_cell =
            (m >= 0 && params.cell_tables.ion_views != nullptr &&
             params.ion_T_top_by_material != nullptr &&
             params.cell_tables.ion_views[m].n_rho > 0);
        const materials::DeviceEOSTableView ion_table =
            i_per_cell ? params.cell_tables.ion_views[m] : params.ion;
        const double ion_T_top =
            i_per_cell ? params.ion_T_top_by_material[m] : params.ion_T_top;
        const auto ion = conduction_thermo_with_tail(
            ion_table, ion_T_top, rho_safe, ti, params.energy_authoritative);
        ion_e = ion.e;
        ion_P = ion.P;
        cvi = ion.cv;
      }
    }
  }
  ee[i] = e;
  Pe[i] = P;
  if (cv_e != nullptr) cv_e[i] = cve;
  if (cv_i != nullptr) cv_i[i] = cvi;
  if (params.two_temp) {
    ei[i] = ion_e;
    Pi[i] = ion_P;
  }
}

__global__ void apply_conduction_energy_increment_kernel(
    const materials::DeviceEOSTableView table_first, const double T_top_first,
    const materials::CellEOSTableSelector cell_tables, const bool use_total,
    const double* T_top_by_material,
    const double te_floor, const int n,
    const std::uint8_t* cell_is_void, const std::size_t mask_size,
    const double* rho, const double* Te_old, const double* cv_old,
    double* ee, double* Te, double* Pe, double* cv_e) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  Pe[i] = 0.0;
  if (cv_e != nullptr) cv_e[i] = 0.0;
  if (static_cast<std::size_t>(i) < mask_size && cell_is_void[i] != 0U) return;
  // Per-cell dominant-material table and ceiling (multi-material closure);
  // falls back to the first non-void material's table when unavailable.
  const int m = cell_tables.material_of(i);
  const materials::DeviceEOSTableView* views =
      use_total ? cell_tables.total_views : cell_tables.electron_views;
  const bool per_cell = (m >= 0 && views != nullptr && T_top_by_material != nullptr &&
                         views[m].n_rho > 0);
  const materials::DeviceEOSTableView table = per_cell ? views[m] : table_first;
  const double T_top = per_cell ? T_top_by_material[m] : T_top_first;
  const double rho_safe = materials::reclose_max(rho[i], 1.0e-30);
  const double cv_used = cv_old != nullptr && cv_old[i] > 0.0
      ? cv_old[i]
      : materials::reclose_max(materials::reclose_cv<true>(
            table, rho_safe, materials::reclose_max(Te_old[i], te_floor)), 0.0);
  // Conservative booking of the solve's energy motion in its own metric.
  // Signed table energies (negative cold-curve values) are valid: the former
  // max(., 0) clamp replaced every negative electron energy by zero and the
  // inversion below then set Te to the zero crossing of e_e(T) (2026-09-14).
  ee[i] = __dadd_rn(ee[i], __dmul_rn(cv_used, Te[i] - Te_old[i]));
  double T_inv = materials::reclose_temperature_from_energy<true>(table, rho_safe, ee[i]);
  bool in_tail = false;
  if (std::isfinite(T_top) && T_top > 0.0) {
    const double e_top = materials::reclose_energy<true>(table, rho_safe, T_top);
    const double cv_top = materials::reclose_max(materials::reclose_cv<true>(table, rho_safe, T_top), 0.0);
    if (std::isfinite(e_top) && cv_top > 0.0 && ee[i] > e_top) {
      T_inv = T_top + (ee[i] - e_top) / cv_top;
      in_tail = true;
    }
  }
  if (!std::isfinite(T_inv) || T_inv < te_floor) T_inv = te_floor;
  Te[i] = T_inv;
  if (in_tail) {
    const double P_top = materials::reclose_pressure<true>(table, rho_safe, T_top);
    Pe[i] = P_top * (T_inv / T_top);
    if (cv_e != nullptr) cv_e[i] = materials::reclose_max(materials::reclose_cv<true>(table, rho_safe, T_top), 0.0);
  } else {
    Pe[i] = materials::reclose_pressure<true>(table, rho_safe, T_inv);
    if (cv_e != nullptr) cv_e[i] = materials::reclose_max(materials::reclose_cv<true>(table, rho_safe, T_inv), 0.0);
  }
}

void refresh_cell_is_void_cache(
    DriverRecloseContext& context,
    const std::vector<std::uint8_t>& cell_is_void) {
  if (context.cached_cell_is_void == cell_is_void) {
    return;
  }
  if (cell_is_void.size() > context.cell_is_void_capacity) {
    if (context.d_cell_is_void != nullptr) {
      cuda_check(cudaFree(context.d_cell_is_void),
                 "driver reclose cell_is_void cudaFree failed");
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&context.d_cell_is_void),
                          cell_is_void.size() * sizeof(std::uint8_t)),
               "driver reclose cell_is_void cudaMalloc failed");
    context.cell_is_void_capacity = cell_is_void.size();
  }
  if (!cell_is_void.empty()) {
    cuda_check(cudaMemcpy(context.d_cell_is_void,
                          cell_is_void.data(),
                          cell_is_void.size() * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "driver reclose cell_is_void H2D failed");
  }
  context.cached_cell_is_void = cell_is_void;
}

// Upload the per-material table temperature ceilings once (Config tables are
// immutable for the run). Materials without a table get 0.0.
void ensure_material_T_top_uploaded(DriverRecloseContext& context,
                                    const core::Config& cfg) {
  const auto& mats = cfg.materials.materials;
  const std::size_t n_mat = mats.size();
  std::vector<const materials::EOSTableTriplet*> keys(n_mat, nullptr);
  for (std::size_t m = 0; m < n_mat; ++m) {
    keys[m] = mats[m].eos_tables.get();
  }
  if (keys == context.material_T_top_keys &&
      context.electron_T_top_by_material.size() == n_mat) {
    return;
  }
  std::vector<double> electron_top(n_mat, 0.0);
  std::vector<double> ion_top(n_mat, 0.0);
  std::vector<double> total_top(n_mat, 0.0);
  const auto top = [](const materials::EOSTable& table) {
    return table.T_grid_eV.empty() ? 0.0 : table.T_grid_eV.back();
  };
  for (std::size_t m = 0; m < n_mat; ++m) {
    if (!mats[m].eos_tables) {
      continue;
    }
    electron_top[m] = top(mats[m].eos_tables->electron);
    ion_top[m] = top(mats[m].eos_tables->ion);
    total_top[m] = top(mats[m].eos_tables->total);
  }
  context.electron_T_top_by_material.reset(n_mat);
  context.ion_T_top_by_material.reset(n_mat);
  context.total_T_top_by_material.reset(n_mat);
  if (n_mat > 0) {
    context.electron_T_top_by_material.copy_from_host(electron_top.data());
    context.ion_T_top_by_material.copy_from_host(ion_top.data());
    context.total_T_top_by_material.copy_from_host(total_top.data());
  }
  context.material_T_top_keys = keys;
}

// Per-cell table selector for a launch: the context's per-material view
// arrays plus the state's dominant-material index (null when not sized).
materials::CellEOSTableSelector cell_table_selector(
    const hydro::HydroEOSContext& eos_context, const core::State& state) {
  const std::size_t n = state.rho.size();
  return materials::make_cell_eos_table_selector(
      eos_context.d_ion_views, eos_context.d_electron_views, eos_context.d_total_views,
      eos_context.n_materials,
      (n > 0 && state.cell_material_index.size() == n) ? state.cell_material_index.data()
                                                        : nullptr);
}

}  // namespace

DriverRecloseContext::~DriverRecloseContext() {
  if (d_cell_is_void != nullptr) {
    static_cast<void>(cudaFree(d_cell_is_void));
  }
}

void launch_tabular_eos_reclose(
    DriverRecloseContext& context,
    const materials::DeviceEOSTableView electron_table,
    const materials::DeviceEOSTableView ion_table,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::size_t n_cells,
    const double* rho,
    double* ee,
    double* ei,
    double* Te,
    double* Ti,
    double* Pe,
    double* Pi,
    double* cv_e,
    double* cv_i,
    const double te_floor,
    const double ti_floor) {
  TENRYU_ASSERT(n_cells <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "driver reclose cell count exceeds kernel limit");
  if (n_cells == 0) {
    return;
  }

  refresh_cell_is_void_cache(context, cell_is_void);
  const int n = static_cast<int>(n_cells);
  const int blocks = (n + kRecloseBlockSize - 1) / kRecloseBlockSize;
  tabular_eos_reclose_kernel<<<blocks, kRecloseBlockSize>>>(
      electron_table,
      ion_table,
      context.d_cell_is_void,
      cell_is_void.size(),
      n,
      rho,
      ee,
      ei,
      Te,
      Ti,
      Pe,
      Pi,
      cv_e,
      cv_i,
      te_floor,
      ti_floor);
  cuda_check(cudaGetLastError(), "driver tabular EOS reclose kernel launch failed");
}


void capture_conduction_fields_device(DriverRecloseContext& context, const core::State& state) {
  const std::size_t n = state.Te.size();
  context.conduction_Te_old.reset(n);
  context.conduction_cv_old.reset(state.cv_e.empty() ? 0 : n);
  if (n == 0) return;
  cuda_check(cudaMemcpyAsync(context.conduction_Te_old.data(), state.Te.data(),
                             n * sizeof(double), cudaMemcpyDeviceToDevice),
             "conduction old Te D2D failed");
  if (!state.cv_e.empty()) {
    TENRYU_ASSERT(state.cv_e.size() == n, "conduction old cv/Te size mismatch");
    cuda_check(cudaMemcpyAsync(context.conduction_cv_old.data(), state.cv_e.data(),
                               n * sizeof(double), cudaMemcpyDeviceToDevice),
               "conduction old cv D2D failed");
  }
}

bool sync_ee_from_Te_device(core::State& state, const core::Config& cfg,
                            const hydro::HydroEOSContext& eos_context,
                            DriverRecloseContext& context) {
  const int first = cfg.materials.first_nonvoid_material_index();
  if (first < 0) return false;
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(first)];
  const bool exact = mat.hydro_eos_backend == "exact_ideal_gas";
  if (!mat.eos_tables && !exact) return false;
  const std::size_t n = state.rho.size();
  if (n == 0) return true;
  state.ensure_cell_material_props(cfg);
  TENRYU_ASSERT(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "conduction EOS cell count exceeds kernel limit");
  ConductionEOSDeviceParams params{};
  params.te_floor = cfg.numerics.floors.Te;
  params.ti_floor = cfg.numerics.floors.Ti;
  params.gamma = mat.ideal_gas_gamma;
  params.A = mat.A;
  params.cv_override = mat.cv_e_override;
  params.T_ref = mat.eos_T_ref_eV;
  params.cv_i_mass = kConductionRecloseEvToErg /
      (mat.A * core::constants::proton_mass * (mat.ideal_gas_gamma - 1.0));
  params.two_temp = cfg.main.two_temperature;
  params.exact_ideal = exact;
  params.energy_authoritative = cfg.numerics.hydro.eos_closure_mode == "energy_authoritative";
  if (!exact) {
    params.electron = eos_context.electron_view(first);
    params.ion = eos_context.ion_view(first);
    params.total = eos_context.total_view(first);
    const auto& tables = *mat.eos_tables;
    const auto top = [](const materials::EOSTable& table) {
      return table.T_grid_eV.empty() ? 0.0 : table.T_grid_eV.back();
    };
    params.electron_T_top = top(tables.electron);
    params.ion_T_top = top(tables.ion);
    params.total_T_top = top(tables.total);
    params.use_total = !params.two_temp && !tables.total.empty();
    if ((params.use_total ? params.total.n_rho : params.electron.n_rho) == 0 ||
        (params.two_temp && params.ion.n_rho == 0)) return false;
    ensure_material_T_top_uploaded(context, cfg);
    params.cell_tables = cell_table_selector(eos_context, state);
    params.electron_T_top_by_material = context.electron_T_top_by_material.data();
    params.ion_T_top_by_material = context.ion_T_top_by_material.data();
    params.total_T_top_by_material = context.total_T_top_by_material.data();
  }
  refresh_cell_is_void_cache(context, state.cell_is_void);
  const int cells = static_cast<int>(n);
  const int blocks = (cells + kRecloseBlockSize - 1) / kRecloseBlockSize;
  sync_conduction_eos_from_temperature_kernel<<<blocks, kRecloseBlockSize>>>(
      params, cells, context.d_cell_is_void, state.cell_is_void.size(),
      state.rho.data(), state.zbar.data(), state.Te.data(), state.Ti.data(),
      state.ee.data(), state.ei.data(), state.Pe.data(), state.Pi.data(),
      state.cv_e.empty() ? nullptr : state.cv_e.data(),
      state.cv_i.empty() ? nullptr : state.cv_i.data());
  cuda_check(cudaGetLastError(), "sync_conduction_eos_from_temperature_kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "sync_conduction_eos_from_temperature_kernel failed");
  return true;
}

bool apply_conduction_energy_increment_device(
    core::State& state, const core::Config& cfg,
    const hydro::HydroEOSContext& eos_context, DriverRecloseContext& context) {
  const int first = cfg.materials.first_nonvoid_material_index();
  if (first < 0) return false;
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(first)];
  if (!mat.eos_tables) return false;
  const auto& table = cfg.main.two_temperature ? mat.eos_tables->electron : mat.eos_tables->total;
  const std::size_t n = state.rho.size();
  if (table.empty() || table.T_grid_eV.empty() || context.conduction_Te_old.size() != n) return false;
  if (n == 0) return true;
  state.ensure_cell_material_props(cfg);
  ensure_material_T_top_uploaded(context, cfg);
  const bool use_total = !cfg.main.two_temperature;
  const materials::CellEOSTableSelector cell_tables = cell_table_selector(eos_context, state);
  const double* T_top_by_material = use_total ? context.total_T_top_by_material.data()
                                              : context.electron_T_top_by_material.data();
  const auto view = cfg.main.two_temperature ? eos_context.electron_view(first) : eos_context.total_view(first);
  if (view.n_rho == 0 || view.n_T == 0) return false;
  TENRYU_ASSERT(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "conduction increment cell count exceeds kernel limit");
  refresh_cell_is_void_cache(context, state.cell_is_void);
  const int cells = static_cast<int>(n);
  const int blocks = (cells + kRecloseBlockSize - 1) / kRecloseBlockSize;
  apply_conduction_energy_increment_kernel<<<blocks, kRecloseBlockSize>>>(
      view, table.T_grid_eV.back(), cell_tables, use_total, T_top_by_material,
      cfg.numerics.floors.Te, cells,
      context.d_cell_is_void, state.cell_is_void.size(), state.rho.data(),
      context.conduction_Te_old.data(), context.conduction_cv_old.data(),
      state.ee.data(), state.Te.data(), state.Pe.data(),
      state.cv_e.empty() ? nullptr : state.cv_e.data());
  cuda_check(cudaGetLastError(), "apply_conduction_energy_increment_kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "apply_conduction_energy_increment_kernel failed");
  return true;
}

}  // namespace tenryu::coupling
