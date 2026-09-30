#include "coupling/source_terms.hpp"
#include "core/nvtx_range.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/deterministic_sum.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "core/launch_shape.hpp"
#include "hydro/eos_context.hpp"
#include "hydro/per_material_eos_accessors.cuh"
#include "hydro/per_material_eos_project.cuh"
#include "materials/eos_device.cuh"
#include "materials/eos_device_table.hpp"
#include "materials/eos_cell_table_selector.cuh"
#include "materials/material_closure.hpp"
#include "materials/eos_table.hpp"
#include "materials/eos_table_reclose.cuh"
#include "mesh/cell_geometry_2d.cuh"

namespace tenryu::coupling {
namespace {

constexpr double kMinEffectiveA = 1.0e-12;
constexpr double kMinEffectiveGamma = 1.0 + 1.0e-12;

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
  materials::DeviceEOSTableView total{};
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
      total_.upload(tables.total);
      tables_ = &tables;
    }
    SourceEOSTableViews views;
    views.ion = ion_.view();
    views.electron = electron_.view();
    views.total = total_.view();
    return views;
  }

 private:
  const materials::EOSTableTriplet* tables_ = nullptr;
  materials::DeviceEOSTable ion_;
  materials::DeviceEOSTable electron_;
  materials::DeviceEOSTable total_;
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
    views.total = eos_ctx->total_view(material_index);
    if (views.ion.n_rho > 0 && views.electron.n_rho > 0) {
      return views;
    }
  }

  return source_eos_table_cache().views_for(*mat.eos_tables);
}

// Per-cell dominant-material table selector of a source-term launch
// (materials/eos_cell_table_selector.cuh). A 1D cell whose material has no
// table closes with the ideal gas of its material, as the 1D hydro closure
// does; the 2D closures still evaluate every cell with one material's tables,
// so a 2D launch keeps lending the first non-void material's table to such a
// cell (2026-09-23).
materials::CellEOSTableSelector source_cell_table_selector(
    core::State& state, const core::Config& cfg, const hydro::HydroEOSContext* eos_ctx) {
  state.ensure_cell_material_props(cfg);
  const std::size_t n = state.rho.size();
  materials::CellEOSTableSelector selector = materials::make_cell_eos_table_selector(
      eos_ctx != nullptr ? eos_ctx->d_ion_views : nullptr,
      eos_ctx != nullptr ? eos_ctx->d_electron_views : nullptr,
      eos_ctx != nullptr ? eos_ctx->d_total_views : nullptr,
      eos_ctx != nullptr ? eos_ctx->n_materials : 0,
      (n > 0 && state.cell_material_index.size() == n) ? state.cell_material_index.data()
                                                        : nullptr);
  selector.lend_fallback_to_tableless = (state.mesh.dim != 1);
  selector.closure_params = (eos_ctx != nullptr) ? eos_ctx->d_closure_params : nullptr;
  return selector;
}

// Material of the source-term closures' table fallback view, and whether the
// table closures run at all (the per-cell table selection then decides table
// or ideal gas per cell): the tables' reference material
// (Config::eos_table_reference_material_index), in 1D the first non-void
// material with tables, so an ideal gas listed first no longer switches the
// table closures off for the tabled cells (2026-09-23); in 2D the first
// non-void material. The exact ideal-gas backend of the first non-void
// material closes every cell with the ideal gas.
int source_table_reference_material(const core::Config& cfg) {
  return cfg.materials.eos_table_reference_material_index(cfg.main.dim);
}

bool source_table_eos_enabled(const core::Config& cfg,
                              const core::Config::MaterialsConfig::MatDef& mat0) {
  const int ref = source_table_reference_material(cfg);
  // The reference material is never an exact ideal gas (Config), and each
  // cell's exact ideal-gas material has empty views: a first material on
  // the exact ideal-gas backend no longer switches the table closures off
  // for the tabled cells of other materials (2026-09-24).
  static_cast<void>(mat0);
  return ref >= 0 && cfg.materials.materials[static_cast<std::size_t>(ref)].eos_tables;
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

// Energy-authoritative closure above the table temperature ceiling: the
// linear ideal-gas tail anchored at T_max (e = e_top + cv_top (T - T_top),
// P = P_top T / T_top, cv = cv_top), identical to
// materials::device_inverse_reclose_with_high_t_tail. in_tail = 0 below the
// ceiling, in legacy closure mode, or when the anchor is not usable.
struct SubstepTailClosure {
  double T = 0.0;
  double P = 0.0;
  double cv = 0.0;
  int in_tail = 0;
};

__device__ inline SubstepTailClosure substep_high_t_tail_closure(
    const tenryu::materials::DeviceEOSTableView& tab,
    const tenryu::materials::RhoBracket& rb,
    const double e_target,
    const int energy_authoritative) {
  SubstepTailClosure out{};
  if (energy_authoritative == 0 || !isfinite(e_target)) {
    return out;
  }
  const tenryu::materials::DeviceEOSHighTTailAnchor a =
      tenryu::materials::device_eos_high_t_tail_anchor(tab, rb);
  if (a.valid == 0 || !(e_target > a.e_top)) {
    return out;
  }
  const double T_tail = a.T_top + (e_target - a.e_top) / a.cv_top;
  if (!isfinite(T_tail) || !(T_tail > a.T_top)) {
    return out;
  }
  out.T = T_tail;
  out.P = a.P_top * (T_tail / a.T_top);
  out.cv = a.cv_top;
  out.in_tail = 1;
  return out;
}

// Host twin of substep_high_t_tail_closure for the host reference paths
// (host table evaluation at the ceiling; agrees with the device anchor to
// interpolation round-off).
SubstepTailClosure host_high_t_tail_closure(const materials::EOSTable& table,
                                            const double rho,
                                            const double e_target,
                                            const bool energy_authoritative) {
  SubstepTailClosure out{};
  if (!energy_authoritative || !std::isfinite(e_target) || table.empty() ||
      table.T_grid_eV.empty()) {
    return out;
  }
  const double T_top = table.T_grid_eV.back();
  const double e_top = table.energy(rho, T_top);
  const double P_top = table.pressure(rho, T_top);
  const double cv_top = table.cv(rho, T_top);
  if (!(std::isfinite(T_top) && T_top > 0.0 && std::isfinite(e_top) &&
        std::isfinite(P_top) && std::isfinite(cv_top) && cv_top > 0.0) ||
      !(e_target > e_top)) {
    return out;
  }
  const double T_tail = T_top + (e_target - e_top) / cv_top;
  if (!std::isfinite(T_tail) || !(T_tail > T_top)) {
    return out;
  }
  out.T = T_tail;
  out.P = P_top * (T_tail / T_top);
  out.cv = cv_top;
  out.in_tail = 1;
  return out;
}

struct LaserInjectionMaterial {
  double inv_A;
  double gamma;
  bool is_void;

  bool operator==(const LaserInjectionMaterial& other) const {
    return inv_A == other.inv_A && gamma == other.gamma && is_void == other.is_void;
  }
};

struct LaserInjectionDeviceCache {
  core::DeviceBuffer<std::uint8_t> cell_is_void;
  std::vector<std::uint8_t> cached_cell_is_void;
  core::DeviceBuffer<LaserInjectionMaterial> materials;
  std::vector<LaserInjectionMaterial> cached_materials;
  std::shared_ptr<const materials::EOSTableTriplet> eos_tables;
};

struct LaserInjectionDeviceParams {
  SourceEOSTableViews tables;
  double A0;
  double gamma0;
  double te_floor;
  double ti_floor;
  double cv_e_override;
  double T_ref;
  int n_materials;
  bool use_two_temp;
  bool has_table_eos;
  bool use_first_cv_override;
  // Per-cell dominant-material table selection (multi-material closure).
  materials::CellEOSTableSelector cell_tables;
  // eos_closure_mode == "energy_authoritative": sub-floor energies are kept
  // (only the temperature is floored) instead of being rewritten (2026-09-14).
  bool energy_authoritative;
};

struct LaserInjectionCellLedger {
  double floor_e;
  double floor_i;
  double skipped_energy;
  int clamp_count;
  bool skipped;
};

struct LaserInjectionSummary {
  double floor_energy = 0.0;
  double skipped_energy = 0.0;
  double redirected_void_energy = 0.0;
  double unrecoverable_void_energy = 0.0;
  double redirected_fraction = 0.0;
  double negative_deposit = 0.0;
  int clamp_count = 0;
  int redirected_void_cells = 0;
  int first_negative_cell = -1;
};

// One thread, cells in order (the backward carry is serial); the restrict
// qualifiers let the loads run ahead of the stores.
__global__ void prepare_laser_injection_kernel(
    const double* __restrict__ deposition, const std::uint8_t* __restrict__ cell_is_void,
    double* __restrict__ redirected, const int n, LaserInjectionSummary* __restrict__ summary) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  LaserInjectionSummary result;
  for (int c = 0; c < n; ++c) {
    double dep = deposition[c];
    if (dep < 0.0) {
      if (result.first_negative_cell < 0) {
        result.first_negative_cell = c;
        result.negative_deposit = dep;
      }
      dep = 0.0;
    }
    redirected[c] = dep;
  }
  double carry = 0.0;
  for (int c = n - 1; c >= 0; --c) {
    const double dep = redirected[c];
    if (cell_is_void[c] != 0U) {
      if (dep > 0.0) {
        ++result.redirected_void_cells;
        result.redirected_void_energy += dep;
      }
      carry += dep;
      redirected[c] = 0.0;
    } else if (carry > 0.0) {
      redirected[c] += carry;
      carry = 0.0;
    }
  }
  if (carry > 0.0) {
    result.unrecoverable_void_energy = carry;
    result.skipped_energy += carry;
  }
  if (result.redirected_void_cells > 0 || result.unrecoverable_void_energy > 0.0) {
    double total = result.redirected_void_energy;
    for (int c = 0; c < n; ++c) total += redirected[c];
    result.redirected_fraction = total > 0.0 ? result.redirected_void_energy / total : 0.0;
  }
  *summary = result;
}

__device__ void laser_injection_material_properties(
    const int c, const LaserInjectionDeviceParams& params,
    const double* volfrac, const LaserInjectionMaterial* materials,
    double& A, double& gamma) {
  if (params.n_materials <= 1 || volfrac == nullptr) return;
  double frac_sum = 0.0;
  double inv_A = 0.0;
  double gamma_sum = 0.0;
  for (int m = 0; m < params.n_materials; ++m) {
    if (materials[m].is_void) continue;
    const double raw = volfrac[static_cast<std::size_t>(c) * params.n_materials + m];
    const double frac = (::isfinite(raw) && raw > 0.0) ? raw : 0.0;
    frac_sum += frac;
    // Match the host's separately rounded products, without CUDA contraction.
    inv_A = __dadd_rn(inv_A, __dmul_rn(frac, materials[m].inv_A));
    gamma_sum = __dadd_rn(gamma_sum, __dmul_rn(frac, materials[m].gamma));
  }
  if (frac_sum > 1.0e-30) {
    inv_A /= frac_sum;
    gamma_sum /= frac_sum;
  }
  if (::isfinite(inv_A) && inv_A > 1.0e-30) {
    A = materials::reclose_max(1.0 / inv_A, kMinEffectiveA);
  }
  if (::isfinite(gamma_sum) && gamma_sum > 0.0) {
    gamma = materials::reclose_max(gamma_sum, kMinEffectiveGamma);
  }
}

__global__ void inject_laser_source_cells_kernel(
    const LaserInjectionDeviceParams params, const int n,
    const std::uint8_t* cell_is_void, const LaserInjectionMaterial* material_params,
    const double* volfrac, const double* rho, const double* vol, const double* zbar,
    const double* cv_e, const double* laser_dep,
    double* ee, double* Te, double* Pe, double* ei, double* Ti, double* Pi,
    LaserInjectionCellLedger* ledger) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  ledger[c] = {};
  int local_clamp_count = 0;
  const double dep = laser_dep[static_cast<std::size_t>(c)];
  const std::size_t c_idx = static_cast<std::size_t>(c);
  if (cell_is_void[c_idx] != 0U) {
    return;
  }
  const double rho_c = rho[c_idx];
  const double vol_c = vol[c_idx];
  const double denom = rho_c * vol_c;
  if (!(denom > 1.0e-30)) {
    ledger[c].skipped_energy = dep;
    ledger[c].skipped = true;
    return;
  }

  // Per-cell dominant-material tables (multi-material closure, 2026-09-14);
  // a 1D cell whose material has no table closes with the ideal gas
  // (source_cell_table_selector).
  const materials::DeviceEOSTableView tab_ele =
      params.cell_tables.electron(c, params.tables.electron);
  const materials::DeviceEOSTableView tab_ion = params.cell_tables.ion(c, params.tables.ion);
  const bool cell_has_tables =
      params.has_table_eos && tab_ele.n_rho > 0 && tab_ion.n_rho > 0;
  // 1T: the total energy closes on the cell's total table, as in the 1T
  // hydro closure (the 1T injection used the ideal-gas branch for every cell,
  // 2026-09-23).
  const materials::DeviceEOSTableView tab_total =
      params.cell_tables.total(c, params.tables.total);
  const materials::DeviceEOSTableView tab_closure = params.use_two_temp ? tab_ele : tab_total;

  double A_c = params.A0;
  double gamma_c = params.gamma0;
  laser_injection_material_properties(c, params, volfrac, material_params, A_c, gamma_c);
  const double gm1_c = gamma_c - 1.0;
  const double cv_mass_i =
      materials::reclose_max(core::constants::eV_to_erg /
                   (A_c * core::constants::proton_mass * gm1_c),
               1.0e-30);
  if (!params.use_two_temp) {
    // 1T closure: ee stores total internal energy.
    ee[c_idx] += ei[c_idx];
  }
  ee[c_idx] += dep / denom;
  const double ee_before_floor = ee[c_idx];

  const double rho_safe = materials::reclose_max(rho_c, 1.0e-30);
  // Heat-capacity override of the cell's material when the materials'
  // closure parameters differ (per-cell 1D runs), else the run-level
  // (first non-void material's) one (2026-09-24).
  const materials::MaterialClosureParams* cp_c = params.cell_tables.closure_of(c);
  const bool use_cv_override_c =
      (cp_c != nullptr) ? (cp_c->cv_e_override > 0.0) : params.use_first_cv_override;
  const double cv_override_c = (cp_c != nullptr) ? cp_c->cv_e_override : params.cv_e_override;
  const double T_ref_c = (cp_c != nullptr) ? cp_c->eos_T_ref_eV : params.T_ref;
  const bool use_table_eos_closure =
      params.use_two_temp ? cell_has_tables : (params.has_table_eos && tab_total.n_rho > 0);
  double Te_raw = 0.0;
  double Te_new = params.te_floor;
  if (use_table_eos_closure) {
    // Energy-authoritative: an energy above the table ceiling closes on the
    // high-temperature tail (2026-09-23; the inversion capped Te at T_max).
    const SubstepTailClosure tail_e = substep_high_t_tail_closure(
        tab_closure, materials::find_rho_bracket(tab_closure, rho_safe), ee[c_idx],
        params.energy_authoritative ? 1 : 0);
    if (tail_e.in_tail != 0) {
      Te_raw = tail_e.T;
      Te_new = tail_e.T;
      Pe[c_idx] = tail_e.P;
    } else {
      Te_raw = materials::reclose_temperature_from_energy<true>(tab_closure, rho_safe, ee[c_idx]);
      if (!::isfinite(Te_raw) || Te_raw < params.te_floor) {
        Te_new = params.te_floor;
        if (!params.energy_authoritative || !::isfinite(Te_raw)) {
          ee[c_idx] = materials::reclose_energy<true>(tab_closure, rho_safe, Te_new);
        }
      } else {
        Te_new = Te_raw;
      }
      Pe[c_idx] = materials::reclose_pressure<true>(tab_closure, rho_safe, Te_new);
    }
  } else if (use_cv_override_c && T_ref_c > 0.0) {
    const double T_ref = T_ref_c;
    const double T_ref3 = T_ref * T_ref * T_ref;
    const double alpha0 = cv_override_c / (4.0 * T_ref3);
    const double arg = ee[c_idx] * rho_safe / alpha0;
    Te_raw = (arg > 0.0) ? ::pow(arg, 0.25) : 0.0;
    if (!::isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = materials::reclose_max(Te_raw, params.te_floor);
    const double Te4 = Te_new * Te_new * Te_new * Te_new;
    ee[c_idx] = alpha0 * Te4 / rho_safe;
  } else {
    const double z = materials::reclose_max(zbar[c_idx], 0.0);
    double cv_mass_e = 0.0;
    bool cv_mass_e_is_total = false;
    if (use_cv_override_c) {
      cv_mass_e = cv_override_c / rho_safe;
    } else if (cv_e != nullptr && cv_e[c_idx] > 0.0) {
      cv_mass_e = cv_e[c_idx];
      // In 1T the state heat capacity is the total one (the 1T hydro closure
      // stores it there); the ion part used to be added a second time.
      cv_mass_e_is_total = !params.use_two_temp;
    } else {
      cv_mass_e = z * core::constants::eV_to_erg /
                  (A_c * core::constants::proton_mass * gm1_c);
    }
    cv_mass_e = materials::reclose_max(cv_mass_e, 1.0e-30);
    const double cv_mass_total = params.use_two_temp
                                     ? cv_mass_e
                                     : ((use_cv_override_c || cv_mass_e_is_total)
                                            ? cv_mass_e
                                            : materials::reclose_max(cv_mass_e + cv_mass_i,
                                                       1.0e-30));
    Te_raw = ee[c_idx] / cv_mass_total;
    if (!::isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = materials::reclose_max(Te_raw, params.te_floor);
    ee[c_idx] = cv_mass_total * Te_new;
  }

  if (Te_new > Te_raw) {
    ++local_clamp_count;
  }
  const double de_floor_e = ee[c_idx] - ee_before_floor;
  if (de_floor_e > 0.0) {
    ledger[c].floor_e = __dmul_rn(__dmul_rn(rho_c, vol_c), de_floor_e);
  }

  Te[c_idx] = Te_new;
  if (!use_table_eos_closure) {
    Pe[c_idx] = gm1_c * rho_c * ee[c_idx];
  }
  if (params.use_two_temp) {
    if (cell_has_tables) {
      const double ei_before_floor = ei[c_idx];
      const SubstepTailClosure tail_i = substep_high_t_tail_closure(
          tab_ion, materials::find_rho_bracket(tab_ion, rho_safe), ei[c_idx],
          params.energy_authoritative ? 1 : 0);
      if (tail_i.in_tail != 0) {
        Ti[c_idx] = tail_i.T;
        Pi[c_idx] = tail_i.P;
      } else {
        const double Ti_raw_table =
            materials::reclose_temperature_from_energy<true>(tab_ion, rho_safe, ei[c_idx]);
        if (!::isfinite(Ti_raw_table) || Ti_raw_table < params.ti_floor) {
          ++local_clamp_count;
          Ti[c_idx] = params.ti_floor;
          if (!params.energy_authoritative || !::isfinite(Ti_raw_table)) {
            ei[c_idx] = materials::reclose_energy<true>(tab_ion, rho_safe, params.ti_floor);
          }
        } else {
          Ti[c_idx] = Ti_raw_table;
        }
        Pi[c_idx] = materials::reclose_pressure<true>(tab_ion, rho_safe, Ti[c_idx]);
      }
      const double de_floor_i = ei[c_idx] - ei_before_floor;
      if (de_floor_i > 0.0) {
        ledger[c].floor_i = __dmul_rn(__dmul_rn(rho_c, vol_c), de_floor_i);
      }
    } else {
      const double Ti_prev = ::isfinite(Ti[c_idx]) ? Ti[c_idx] : 0.0;
      if (Ti_prev < params.ti_floor) {
        ++local_clamp_count;
        const double ei_before_floor = ei[c_idx];
        Ti[c_idx] = params.ti_floor;
        ei[c_idx] = cv_mass_i * params.ti_floor;
        Pi[c_idx] = gm1_c * rho_c * ei[c_idx];
        const double de_floor_i = ei[c_idx] - ei_before_floor;
        if (de_floor_i > 0.0) {
          ledger[c].floor_i = __dmul_rn(__dmul_rn(rho_c, vol_c), de_floor_i);
        }
      }
    }
  } else {
    // 1T closure convention: ee is total internal energy, ei/Pi are unused.
    Ti[c_idx] = Te_new;
    ei[c_idx] = 0.0;
    Pi[c_idx] = 0.0;
  }
  ledger[c].clamp_count = local_clamp_count;
}

__global__ void fold_laser_injection_ledger_kernel(
    const LaserInjectionCellLedger* __restrict__ ledger, const int n,
    LaserInjectionSummary* __restrict__ summary) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  // Fixed cell order, electron then ion, exactly as in the host ledger. The
  // sums are held in registers from the summary's values (the same additions
  // in the same order): with the summary updated in memory every cell, each
  // cell's loads waited for the previous store.
  double floor_energy = summary->floor_energy;
  double skipped_energy = summary->skipped_energy;
  int clamp_count = summary->clamp_count;
  for (int c = 0; c < n; ++c) {
    if (ledger[c].floor_e > 0.0) floor_energy += ledger[c].floor_e;
    if (ledger[c].floor_i > 0.0) floor_energy += ledger[c].floor_i;
    if (ledger[c].skipped) skipped_energy += ledger[c].skipped_energy;
    clamp_count += ledger[c].clamp_count;
  }
  summary->floor_energy = floor_energy;
  summary->skipped_energy = skipped_energy;
  summary->clamp_count = clamp_count;
}

double inject_laser_source_terms_device(
    core::State& state, const core::Config& cfg,
    const core::Config::MaterialsConfig::MatDef& mat0,
    const bool has_table_eos, const bool use_first_cv_override,
    double* E_floor_injected, int* clamp_count,
    const hydro::HydroEOSContext* eos_ctx) {
  static LaserInjectionDeviceCache cache;
  const std::size_t n = state.rho.size();
  if (cache.cached_cell_is_void != state.cell_is_void) {
    cache.cell_is_void.reset(n);
    cache.cell_is_void.copy_from_host(state.cell_is_void);
    cache.cached_cell_is_void = state.cell_is_void;
  }
  std::vector<LaserInjectionMaterial> material_params;
  material_params.reserve(cfg.materials.materials.size());
  for (const auto& mat : cfg.materials.materials) {
    material_params.push_back({1.0 / std::max(mat.A, kMinEffectiveA),
                              std::max(mat.ideal_gas_gamma, kMinEffectiveGamma),
                              mat.is_void});
  }
  if (cache.cached_materials != material_params) {
    cache.materials.reset(material_params.size());
    cache.materials.copy_from_host(material_params);
    cache.cached_materials = material_params;
  }
  const bool valid_volfrac = state.volFrac.size() == n * material_params.size();
  if (!valid_volfrac && material_params.size() > 1) {
    static bool warned_volfrac_size_mismatch = false;
    if (!warned_volfrac_size_mismatch) {
      core::log_warning("Multi-material mixing: volFrac size mismatch (" +
                        std::to_string(state.volFrac.size()) + " vs expected " +
                        std::to_string(n * material_params.size()) +
                        "); falling back to first non-void material.");
      warned_volfrac_size_mismatch = true;
    }
  }
  LaserInjectionDeviceParams params{};
  if (has_table_eos) {
    const int ref = source_table_reference_material(cfg);
    const auto& ref_mat = cfg.materials.materials[static_cast<std::size_t>(ref)];
    params.tables = select_source_eos_table_views(ref_mat, ref, nullptr);
    // Keep the table cache's identity alive across independent driver runs.
    cache.eos_tables = ref_mat.eos_tables;
  }
  params.A0 = std::max(mat0.A, kMinEffectiveA);
  params.gamma0 = std::max(mat0.ideal_gas_gamma, kMinEffectiveGamma);
  params.te_floor = cfg.numerics.floors.Te;
  params.ti_floor = cfg.numerics.floors.Ti;
  params.cv_e_override = mat0.cv_e_override;
  params.T_ref = mat0.eos_T_ref_eV;
  params.n_materials = static_cast<int>(material_params.size());
  params.use_two_temp = cfg.main.two_temperature;
  params.has_table_eos = has_table_eos;
  params.use_first_cv_override = use_first_cv_override;
  params.cell_tables = source_cell_table_selector(state, cfg, eos_ctx);
  params.energy_authoritative =
      (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
  auto* redirected = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:laser_redirected", n * sizeof(double)));
  auto* ledger = static_cast<LaserInjectionCellLedger*>(core::device_scratch_acquire(
      "source_terms:laser_cell_ledger", n * sizeof(LaserInjectionCellLedger)));
  auto* summary = static_cast<LaserInjectionSummary*>(core::device_scratch_acquire(
      "source_terms:laser_summary", sizeof(LaserInjectionSummary)));
  prepare_laser_injection_kernel<<<1, 1>>>(
      state.laser_dep.data(), cache.cell_is_void.data(), redirected,
      static_cast<int>(n), summary);
  cuda_check(cudaGetLastError(), "prepare_laser_injection_kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "prepare_laser_injection_kernel failed");
  const int threads = core::serial_cell_block_size(static_cast<int>(n));
  inject_laser_source_cells_kernel<<<core::serial_cell_blocks(static_cast<int>(n)), threads>>>(
      params, static_cast<int>(n), cache.cell_is_void.data(), cache.materials.data(),
      valid_volfrac ? state.volFrac.data() : nullptr,
      state.rho.data(), state.vol.data(), state.zbar.data(),
      state.cv_e.empty() ? nullptr : state.cv_e.data(), redirected,
      state.ee.data(), state.Te.data(), state.Pe.data(), state.ei.data(),
      state.Ti.data(), state.Pi.data(), ledger);
  cuda_check(cudaGetLastError(), "inject_laser_source_cells_kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "inject_laser_source_cells_kernel failed");
  fold_laser_injection_ledger_kernel<<<1, 1>>>(ledger, static_cast<int>(n), summary);
  cuda_check(cudaGetLastError(), "fold_laser_injection_ledger_kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "fold_laser_injection_ledger_kernel failed");
  LaserInjectionSummary result;
  cuda_check(cudaMemcpy(&result, summary, sizeof(result), cudaMemcpyDeviceToHost),
             "inject_laser_source_terms_device summary D2H failed");
  static bool warned_negative_laser_dep = false;
  if (result.first_negative_cell >= 0 && !warned_negative_laser_dep) {
    core::log_warning("inject_laser_source_terms: negative laser_dep detected (cell=" +
                      std::to_string(result.first_negative_cell) + ", value=" +
                      std::to_string(result.negative_deposit) + "); clamping to 0.");
    warned_negative_laser_dep = true;
  }
  if ((result.redirected_void_cells > 0 || result.unrecoverable_void_energy > 0.0) &&
      (result.redirected_fraction >= 0.05 || result.unrecoverable_void_energy > 0.0)) {
    core::log_warning("WARNING: laser energy in void cells redirected from " +
                      std::to_string(result.redirected_void_cells) + " cells (" +
                      std::to_string(result.redirected_fraction * 100.0) +
                      "% of total), unrecoverable=" +
                      std::to_string(result.unrecoverable_void_energy) + " erg");
  }
  accumulate_floor_and_clamp(E_floor_injected, clamp_count,
                            result.floor_energy, result.clamp_count);
  return result.skipped_energy;
}

// Burn energy deposit (NUMERICS §14.5). A cell without a deposit is left as
// it is (re-closing it only perturbed Te by the inversion tolerance and, above
// the table ceiling, capped it at T_max); a deposit cell takes dep_e/(rho V)
// into ee and dep_i/(rho V) into ei (1T: both into ee) and closes with the
// laser-deposit closure (host-matching table inversion, cv_override or ideal
// gas), except that an energy-authoritative table energy above the table
// ceiling closes on the high-temperature tail. Void or massless cells book the
// deposit as skipped. The ledger is folded in cell order.
struct BurnInjectionCellLedger {
  double floor_e;
  double floor_i;
  double skipped_energy;
  int clamp_count;
};

struct BurnInjectionSummary {
  double floor_energy = 0.0;
  double skipped_energy = 0.0;
  int clamp_count = 0;
};

__global__ void inject_burn_source_cells_kernel(
    const LaserInjectionDeviceParams params, const int n,
    const std::uint8_t* cell_is_void, const LaserInjectionMaterial* material_params,
    const double* volfrac, const double* rho, const double* vol, const double* zbar,
    const double* cv_e, const double* dep_e_cell, const double* dep_i_cell,
    double* ee, double* Te, double* Pe, double* ei, double* Ti, double* Pi,
    BurnInjectionCellLedger* ledger) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n) return;
  ledger[c] = {};
  const std::size_t c_idx = static_cast<std::size_t>(c);
  const double dep_e = dep_e_cell[c_idx];
  const double dep_i = dep_i_cell[c_idx];
  const double rho_c = rho[c_idx];
  const double vol_c = vol[c_idx];
  const double denom = rho_c * vol_c;
  if (cell_is_void[c_idx] != 0U || !(denom > 1.0e-30)) {
    ledger[c].skipped_energy = dep_e + dep_i;
    return;
  }
  if (dep_e == 0.0 && dep_i == 0.0) {
    return;
  }
  int local_clamp_count = 0;

  // Per-cell dominant-material tables; a 1D cell whose material has no table
  // closes with the ideal gas (source_cell_table_selector).
  const materials::DeviceEOSTableView tab_ele =
      params.cell_tables.electron(c, params.tables.electron);
  const materials::DeviceEOSTableView tab_ion = params.cell_tables.ion(c, params.tables.ion);
  const bool cell_has_tables =
      params.has_table_eos && tab_ele.n_rho > 0 && tab_ion.n_rho > 0;
  // 1T: the total energy closes on the cell's total table, as in the 1T
  // hydro closure (the 1T injection used the ideal-gas branch for every cell,
  // 2026-09-23).
  const materials::DeviceEOSTableView tab_total =
      params.cell_tables.total(c, params.tables.total);
  const materials::DeviceEOSTableView tab_closure = params.use_two_temp ? tab_ele : tab_total;

  double A_c = params.A0;
  double gamma_c = params.gamma0;
  laser_injection_material_properties(c, params, volfrac, material_params, A_c, gamma_c);
  const double gm1_c = gamma_c - 1.0;
  const double cv_mass_i =
      materials::reclose_max(core::constants::eV_to_erg /
                   (A_c * core::constants::proton_mass * gm1_c),
               1.0e-30);
  if (!params.use_two_temp) {
    // 1T closure: ee stores total internal energy.
    ee[c_idx] += ei[c_idx];
  }
  ee[c_idx] += (params.use_two_temp ? dep_e : (dep_e + dep_i)) / denom;
  const double ee_before_floor = ee[c_idx];

  const double rho_safe = materials::reclose_max(rho_c, 1.0e-30);
  // Heat-capacity override of the cell's material when the materials'
  // closure parameters differ (per-cell 1D runs), else the run-level
  // (first non-void material's) one (2026-09-24).
  const materials::MaterialClosureParams* cp_c = params.cell_tables.closure_of(c);
  const bool use_cv_override_c =
      (cp_c != nullptr) ? (cp_c->cv_e_override > 0.0) : params.use_first_cv_override;
  const double cv_override_c = (cp_c != nullptr) ? cp_c->cv_e_override : params.cv_e_override;
  const double T_ref_c = (cp_c != nullptr) ? cp_c->eos_T_ref_eV : params.T_ref;
  const bool use_table_eos_closure =
      params.use_two_temp ? cell_has_tables : (params.has_table_eos && tab_total.n_rho > 0);
  const int energy_authoritative = params.energy_authoritative ? 1 : 0;
  double Te_raw = 0.0;
  double Te_new = params.te_floor;
  if (use_table_eos_closure) {
    const SubstepTailClosure tail_e = substep_high_t_tail_closure(
        tab_closure, materials::find_rho_bracket(tab_closure, rho_safe), ee[c_idx],
        energy_authoritative);
    if (tail_e.in_tail != 0) {
      Te_raw = tail_e.T;
      Te_new = tail_e.T;
      Pe[c_idx] = tail_e.P;
    } else {
      Te_raw = materials::reclose_temperature_from_energy<true>(tab_closure, rho_safe, ee[c_idx]);
      if (!::isfinite(Te_raw) || Te_raw < params.te_floor) {
        Te_new = params.te_floor;
        if (!params.energy_authoritative || !::isfinite(Te_raw)) {
          ee[c_idx] = materials::reclose_energy<true>(tab_closure, rho_safe, Te_new);
        }
      } else {
        Te_new = Te_raw;
      }
      Pe[c_idx] = materials::reclose_pressure<true>(tab_closure, rho_safe, Te_new);
    }
  } else if (use_cv_override_c && T_ref_c > 0.0) {
    const double T_ref = T_ref_c;
    const double T_ref3 = T_ref * T_ref * T_ref;
    const double alpha0 = cv_override_c / (4.0 * T_ref3);
    const double arg = ee[c_idx] * rho_safe / alpha0;
    Te_raw = (arg > 0.0) ? ::pow(arg, 0.25) : 0.0;
    if (!::isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = materials::reclose_max(Te_raw, params.te_floor);
    const double Te4 = Te_new * Te_new * Te_new * Te_new;
    ee[c_idx] = alpha0 * Te4 / rho_safe;
  } else {
    const double z = materials::reclose_max(zbar[c_idx], 0.0);
    double cv_mass_e = 0.0;
    bool cv_mass_e_is_total = false;
    if (use_cv_override_c) {
      cv_mass_e = cv_override_c / rho_safe;
    } else if (cv_e != nullptr && cv_e[c_idx] > 0.0) {
      cv_mass_e = cv_e[c_idx];
      // In 1T the state heat capacity is the total one (the 1T hydro closure
      // stores it there); the ion part used to be added a second time.
      cv_mass_e_is_total = !params.use_two_temp;
    } else {
      cv_mass_e = z * core::constants::eV_to_erg /
                  (A_c * core::constants::proton_mass * gm1_c);
    }
    cv_mass_e = materials::reclose_max(cv_mass_e, 1.0e-30);
    const double cv_mass_total = params.use_two_temp
                                     ? cv_mass_e
                                     : ((use_cv_override_c || cv_mass_e_is_total)
                                            ? cv_mass_e
                                            : materials::reclose_max(cv_mass_e + cv_mass_i,
                                                       1.0e-30));
    Te_raw = ee[c_idx] / cv_mass_total;
    if (!::isfinite(Te_raw)) {
      Te_raw = 0.0;
    }
    Te_new = materials::reclose_max(Te_raw, params.te_floor);
    ee[c_idx] = cv_mass_total * Te_new;
  }

  if (Te_new > Te_raw) {
    ++local_clamp_count;
  }
  const double de_floor_e = ee[c_idx] - ee_before_floor;
  if (de_floor_e > 0.0) {
    ledger[c].floor_e = __dmul_rn(__dmul_rn(rho_c, vol_c), de_floor_e);
  }

  Te[c_idx] = Te_new;
  if (!use_table_eos_closure) {
    Pe[c_idx] = gm1_c * rho_c * ee[c_idx];
  }
  if (params.use_two_temp) {
    ei[c_idx] += dep_i / denom;
    if (cell_has_tables) {
      const double ei_before_floor = ei[c_idx];
      const SubstepTailClosure tail_i = substep_high_t_tail_closure(
          tab_ion, materials::find_rho_bracket(tab_ion, rho_safe), ei[c_idx],
          energy_authoritative);
      if (tail_i.in_tail != 0) {
        Ti[c_idx] = tail_i.T;
        Pi[c_idx] = tail_i.P;
      } else {
        const double Ti_raw_table =
            materials::reclose_temperature_from_energy<true>(tab_ion, rho_safe, ei[c_idx]);
        if (!::isfinite(Ti_raw_table) || Ti_raw_table < params.ti_floor) {
          ++local_clamp_count;
          Ti[c_idx] = params.ti_floor;
          if (!params.energy_authoritative || !::isfinite(Ti_raw_table)) {
            ei[c_idx] = materials::reclose_energy<true>(tab_ion, rho_safe, params.ti_floor);
          }
        } else {
          Ti[c_idx] = Ti_raw_table;
        }
        Pi[c_idx] = materials::reclose_pressure<true>(tab_ion, rho_safe, Ti[c_idx]);
      }
      const double de_floor_i = ei[c_idx] - ei_before_floor;
      if (de_floor_i > 0.0) {
        ledger[c].floor_i = __dmul_rn(__dmul_rn(rho_c, vol_c), de_floor_i);
      }
    } else {
      const double Ti_raw = ei[c_idx] / cv_mass_i;
      const double Ti_new = materials::reclose_max(Ti_raw, params.ti_floor);
      const double ei_before_floor = ei[c_idx];
      Ti[c_idx] = Ti_new;
      if (Ti_new > Ti_raw) {
        ++local_clamp_count;
        ei[c_idx] = cv_mass_i * Ti_new;
        const double de_floor_i = ei[c_idx] - ei_before_floor;
        if (de_floor_i > 0.0) {
          ledger[c].floor_i = __dmul_rn(__dmul_rn(rho_c, vol_c), de_floor_i);
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
  ledger[c].clamp_count = local_clamp_count;
}

__global__ void fold_burn_injection_ledger_kernel(
    const BurnInjectionCellLedger* ledger, const int n, BurnInjectionSummary* summary) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  BurnInjectionSummary result;
  // Fixed cell order, electron then ion, as the host ledger summed them.
  for (int c = 0; c < n; ++c) {
    if (ledger[c].floor_e > 0.0) result.floor_energy += ledger[c].floor_e;
    if (ledger[c].floor_i > 0.0) result.floor_energy += ledger[c].floor_i;
    result.skipped_energy += ledger[c].skipped_energy;
    result.clamp_count += ledger[c].clamp_count;
  }
  *summary = result;
}

double inject_burn_source_terms_device(
    core::State& state, const core::Config& cfg,
    const core::Config::MaterialsConfig::MatDef& mat0,
    const bool has_table_eos, const bool use_first_cv_override,
    const std::vector<double>& dE_e, const std::vector<double>& dE_i,
    double* E_floor_injected, int* clamp_count,
    const hydro::HydroEOSContext* eos_ctx) {
  static LaserInjectionDeviceCache cache;
  const std::size_t n = state.rho.size();
  if (cache.cached_cell_is_void != state.cell_is_void) {
    cache.cell_is_void.reset(n);
    cache.cell_is_void.copy_from_host(state.cell_is_void);
    cache.cached_cell_is_void = state.cell_is_void;
  }
  std::vector<LaserInjectionMaterial> material_params;
  material_params.reserve(cfg.materials.materials.size());
  for (const auto& mat : cfg.materials.materials) {
    material_params.push_back({1.0 / std::max(mat.A, kMinEffectiveA),
                              std::max(mat.ideal_gas_gamma, kMinEffectiveGamma),
                              mat.is_void});
  }
  if (cache.cached_materials != material_params) {
    cache.materials.reset(material_params.size());
    cache.materials.copy_from_host(material_params);
    cache.cached_materials = material_params;
  }
  const bool valid_volfrac = state.volFrac.size() == n * material_params.size();
  LaserInjectionDeviceParams params{};
  if (has_table_eos) {
    const int ref = source_table_reference_material(cfg);
    const auto& ref_mat = cfg.materials.materials[static_cast<std::size_t>(ref)];
    params.tables = select_source_eos_table_views(ref_mat, ref, nullptr);
    cache.eos_tables = ref_mat.eos_tables;
  }
  params.A0 = std::max(mat0.A, kMinEffectiveA);
  params.gamma0 = std::max(mat0.ideal_gas_gamma, kMinEffectiveGamma);
  params.te_floor = cfg.numerics.floors.Te;
  params.ti_floor = cfg.numerics.floors.Ti;
  params.cv_e_override = mat0.cv_e_override;
  params.T_ref = mat0.eos_T_ref_eV;
  params.n_materials = static_cast<int>(material_params.size());
  params.use_two_temp = cfg.main.two_temperature;
  params.has_table_eos = has_table_eos;
  params.use_first_cv_override = use_first_cv_override;
  params.cell_tables = source_cell_table_selector(state, cfg, eos_ctx);
  params.energy_authoritative =
      (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");

  auto* d_dep = static_cast<double*>(core::device_scratch_acquire(
      "source_terms:burn_deposit", 2 * n * sizeof(double)));
  cuda_check(cudaMemcpy(d_dep, dE_e.data(), n * sizeof(double), cudaMemcpyHostToDevice),
             "inject_burn_source_terms dE_e upload failed");
  cuda_check(cudaMemcpy(d_dep + n, dE_i.data(), n * sizeof(double), cudaMemcpyHostToDevice),
             "inject_burn_source_terms dE_i upload failed");
  auto* ledger = static_cast<BurnInjectionCellLedger*>(core::device_scratch_acquire(
      "source_terms:burn_cell_ledger", n * sizeof(BurnInjectionCellLedger)));
  auto* summary = static_cast<BurnInjectionSummary*>(core::device_scratch_acquire(
      "source_terms:burn_summary", sizeof(BurnInjectionSummary)));
  const int threads = core::serial_cell_block_size(static_cast<int>(n));
  inject_burn_source_cells_kernel<<<core::serial_cell_blocks(static_cast<int>(n)), threads>>>(
      params, static_cast<int>(n), cache.cell_is_void.data(), cache.materials.data(),
      valid_volfrac ? state.volFrac.data() : nullptr, state.rho.data(), state.vol.data(),
      state.zbar.data(), state.cv_e.empty() ? nullptr : state.cv_e.data(), d_dep, d_dep + n,
      state.ee.data(), state.Te.data(), state.Pe.data(), state.ei.data(), state.Ti.data(),
      state.Pi.data(), ledger);
  cuda_check(cudaGetLastError(), "inject_burn_source_cells_kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "inject_burn_source_cells_kernel failed");
  fold_burn_injection_ledger_kernel<<<1, 1>>>(ledger, static_cast<int>(n), summary);
  cuda_check(cudaGetLastError(), "fold_burn_injection_ledger_kernel launch failed");
  BurnInjectionSummary result;
  cuda_check(cudaMemcpy(&result, summary, sizeof(result), cudaMemcpyDeviceToHost),
             "inject_burn_source_terms summary D2H failed");
  accumulate_floor_and_clamp(E_floor_injected, clamp_count, result.floor_energy,
                            result.clamp_count);
  return result.skipped_energy;
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
    const LaserInjectionDeviceParams mix_params,
    const LaserInjectionMaterial* __restrict__ material_params,
    const double* __restrict__ volfrac,
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
    const tenryu::materials::DeviceEOSTableView tab_ion_first,
    const tenryu::materials::DeviceEOSTableView tab_ele_first,
    const tenryu::materials::CellEOSTableSelector cell_tables,
    const int energy_authoritative,
    const double* __restrict__ zmom_r2) {
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

  // Per-cell dominant-material tables (multi-material closure, 2026-09-14);
  // a 1D cell whose material has no table closes with the ideal gas
  // (source_cell_table_selector).
  const tenryu::materials::DeviceEOSTableView tab_ion = cell_tables.ion(c, tab_ion_first);
  const tenryu::materials::DeviceEOSTableView tab_ele =
      cell_tables.electron(c, tab_ele_first);

  // Heat-capacity override of the cell's material when the materials'
  // closure parameters differ (per-cell 1D runs), else the run-level one
  // (2026-09-24).
  const materials::MaterialClosureParams* cp_c = cell_tables.closure_of(c);
  const bool use_cv_override_c =
      (cp_c != nullptr) ? (cp_c->cv_e_override > 0.0) : use_first_cv_override;
  const double cv_override_c = (cp_c != nullptr) ? cp_c->cv_e_override : cv_e_override;
  const double T_ref_c = (cp_c != nullptr) ? cp_c->eos_T_ref_eV : eos_T_ref_eV;
  const double rho_c = rho[c];
  const double rho_safe = fmax(rho_c, 1.0e-30);
  // Effective A and gamma of the cell's material mix, formed here with the
  // host-matching products of compute_effective_A_gamma (the former host
  // evaluation and upload every substep gave the same values, 2026-09-23).
  double A_mix = mix_params.A0;
  double gamma_mix = mix_params.gamma0;
  laser_injection_material_properties(c, mix_params, volfrac, material_params, A_mix,
                                      gamma_mix);
  const double A_c = fmax(A_mix, 1.0e-12);
  const double gamma_c = fmax(gamma_mix, 1.0 + 1.0e-12);
  const double gm1_c = fmax(gamma_c - 1.0, 1.0e-30);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : 0.0;
  const double cv_mass_i_fallback =
      fmax(tenryu::core::constants::eV_to_erg /
               (A_c * tenryu::core::constants::proton_mass * gm1_c),
           1.0e-30);
  double cv_mass_e_qei = 0.0;
  if (cv_e != nullptr && cv_e[c] > 0.0) {
    cv_mass_e_qei = cv_e[c];
  } else if (use_cv_override_c) {
    cv_mass_e_qei = cv_override_c / rho_safe;
  } else {
    cv_mass_e_qei =
        z * tenryu::core::constants::eV_to_erg /
        (A_c * tenryu::core::constants::proton_mass * gm1_c);
  }
  cv_mass_e_qei = fmax(cv_mass_e_qei, 1.0e-30);
  const double cv_mass_i_qei =
      (cv_i != nullptr && cv_i[c] > 0.0) ? fmax(cv_i[c], 1.0e-30)
                                         : cv_mass_i_fallback;

  // zmom_r2 (the cell's <Z^2>/<Z>^2 while the moment tables are active): the relaxation time takes
  // <Z^2> = Zbar^2 r2, as on the hydro update paths.
  double qei_term = 0.0;
  if (cv_e != nullptr && cv_i != nullptr && cv_e[c] > 0.0 && cv_i[c] > 0.0) {
    qei_term = (zmom_r2 != nullptr)
                   ? tenryu::materials::compute_qei_term_with_cv_ext(
                         fmax(rho_c, 0.0), fmax(Te[c], 0.0), fmax(Ti[c], 0.0), z, zmom_r2[c], A_c,
                         cv_mass_e_qei, cv_mass_i_qei, dt, qei_multiplier)
                   : tenryu::materials::compute_qei_term_with_cv(
                         fmax(rho_c, 0.0), fmax(Te[c], 0.0), fmax(Ti[c], 0.0), z, A_c,
                         cv_mass_e_qei, cv_mass_i_qei, dt, qei_multiplier);
  } else {
    qei_term = (zmom_r2 != nullptr)
                   ? tenryu::materials::compute_qei_term_analytical_ext(
                         fmax(rho_c, 0.0), fmax(Te[c], 0.0), fmax(Ti[c], 0.0), z, zmom_r2[c], A_c,
                         gamma_c, dt, qei_multiplier)
                   : tenryu::materials::compute_qei_term_analytical(
                         fmax(rho_c, 0.0), fmax(Te[c], 0.0), fmax(Ti[c], 0.0), z, A_c,
                         gamma_c, dt, qei_multiplier);
  }

  // 2026-07-26 review: bracket the transfer into the
  // physically admissible interval instead of clamping each side
  // independently — the old independent fmax floors created or destroyed
  // pair energy whenever one side hit zero (reachable for table-EOS cells
  // where the frozen-cv transfer overshoots the stored energy). One shared
  // applied transfer keeps ee + ei exactly conserved and both sides
  // nonnegative; cells where the floors never engaged are bit-identical.
  double qei_hi = fmax(ee[c], 0.0);   // most the electrons can give
  double qei_lo = -fmax(ei[c], 0.0);  // most the ions can give
  // Energy-authoritative table cells: the admissible domain is
  // [e(rho, T_floor), inf) and signed (negative cold-curve) table energies are
  // valid, so bound the transfer by the energy above the floor (2026-09-14).
  if (energy_authoritative != 0 && has_table_eos && tab_ele.n_rho > 0 &&
      tab_ion.n_rho > 0 && te_floor > 0.0 && ti_floor > 0.0) {
    const auto rb_e_floor = tenryu::materials::find_rho_bracket(tab_ele, rho_safe);
    const auto rb_i_floor = tenryu::materials::find_rho_bracket(tab_ion, rho_safe);
    const double e_e_floor = tenryu::materials::device_eos_energy(
        tab_ele, rb_e_floor, log(fmax(te_floor, 1.0e-300)));
    const double e_i_floor = tenryu::materials::device_eos_energy(
        tab_ion, rb_i_floor, log(fmax(ti_floor, 1.0e-300)));
    qei_hi = fmax(ee[c] - e_e_floor, 0.0);
    qei_lo = fmin(-(ei[c] - e_i_floor), 0.0);
  }
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
    // Energy-authoritative: an energy above the table ceiling inverts into the
    // ideal-gas tail, as in the hydro closures, the FLD matter update and the
    // conduction increment; capping Te at T_max here handed the conduction
    // solve that follows a flattened Te profile (2026-09-23).
    const SubstepTailClosure tail_e = substep_high_t_tail_closure(
        tab_ele, rb_e, ee[c], energy_authoritative);
    if (tail_e.in_tail != 0) {
      Te_new = tail_e.T;
      Pe[c] = tail_e.P;
      if (cv_e != nullptr) {
        cv_e[c] = tail_e.cv;
      }
    } else {
      if (!isfinite(Te_raw) || Te_raw < te_floor) {
        Te_new = te_floor;
        // Energy-authoritative: keep the sub-floor energy (only the temperature
        // is floored); the table value is written only for non-finite input.
        if (energy_authoritative == 0 || !isfinite(Te_raw)) {
          const double logTe = log(fmax(Te_new, 1.0e-300));
          ee[c] = tenryu::materials::device_eos_energy(tab_ele, rb_e, logTe);
        }
      } else {
        Te_new = Te_raw;
      }
      const double logTe = log(fmax(Te_new, 1.0e-300));
      Pe[c] = tenryu::materials::device_eos_pressure(tab_ele, rb_e, logTe);
      if (cv_e != nullptr) {
        cv_e[c] = fmax(tenryu::materials::device_eos_cv(tab_ele, rb_e, logTe), 0.0);
      }
    }
  } else if (use_cv_override_c && T_ref_c > 0.0) {
    const double T_ref3 = T_ref_c * T_ref_c * T_ref_c;
    const double alpha0 = cv_override_c / (4.0 * T_ref3);
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
    const SubstepTailClosure tail_i = substep_high_t_tail_closure(
        tab_ion, rb_i, ei[c], energy_authoritative);
    if (tail_i.in_tail != 0) {
      Ti[c] = tail_i.T;
      Pi[c] = tail_i.P;
      if (cv_i != nullptr) {
        cv_i[c] = tail_i.cv;
      }
    } else {
      if (!isfinite(Ti_raw_table) || Ti_raw_table < ti_floor) {
        Ti[c] = ti_floor;
        if (energy_authoritative == 0 || !isfinite(Ti_raw_table)) {
          const double logTi = log(fmax(ti_floor, 1.0e-300));
          ei[c] = tenryu::materials::device_eos_energy(tab_ion, rb_i, logTi);
        }
      } else {
        Ti[c] = Ti_raw_table;
      }
      const double logTi = log(fmax(Ti[c], 1.0e-300));
      Pi[c] = tenryu::materials::device_eos_pressure(tab_ion, rb_i, logTi);
      if (cv_i != nullptr) {
        cv_i[c] = fmax(tenryu::materials::device_eos_cv(tab_ion, rb_i, logTi), 0.0);
      }
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
    // 2026-07-26 review: shared bracketed transfer —
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

}  // namespace

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
    d_hydro_active = (cfg.numerics.hydro.T_start_inactive_cells == "rigid_wall")
                         ? nullptr
                         : const_cast<std::int8_t*>(state.hydro_active_device_ptr());

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
  const bool has_table_eos = source_table_eos_enabled(cfg, mat0);

  // Material table and void mask cached on the device (uploaded when they
  // change); the kernel forms each cell's effective A and gamma itself.
  static LaserInjectionDeviceCache qei_cache;
  const std::size_t n_sz = static_cast<std::size_t>(n_cells);
  if (qei_cache.cached_cell_is_void != state.cell_is_void) {
    qei_cache.cell_is_void.reset(n_sz);
    qei_cache.cell_is_void.copy_from_host(state.cell_is_void);
    qei_cache.cached_cell_is_void = state.cell_is_void;
  }
  std::vector<LaserInjectionMaterial> material_params;
  material_params.reserve(materials.size());
  for (const auto& mat : materials) {
    material_params.push_back({1.0 / std::max(mat.A, kMinEffectiveA),
                              std::max(mat.ideal_gas_gamma, kMinEffectiveGamma),
                              mat.is_void});
  }
  if (qei_cache.cached_materials != material_params) {
    qei_cache.materials.reset(material_params.size());
    qei_cache.materials.copy_from_host(material_params);
    qei_cache.cached_materials = material_params;
  }
  const bool valid_volfrac = state.volFrac.size() == n_sz * material_params.size();
  if (!valid_volfrac && material_params.size() > 1) {
    static bool warned_volfrac_size_mismatch = false;
    if (!warned_volfrac_size_mismatch) {
      core::log_warning("Multi-material mixing: volFrac size mismatch (" +
                        std::to_string(state.volFrac.size()) + " vs expected " +
                        std::to_string(n_sz * material_params.size()) +
                        "); falling back to first non-void material.");
      warned_volfrac_size_mismatch = true;
    }
  }
  LaserInjectionDeviceParams mix_params{};
  mix_params.A0 = std::max(mat0.A, kMinEffectiveA);
  mix_params.gamma0 = std::max(mat0.ideal_gas_gamma, kMinEffectiveGamma);
  mix_params.n_materials = static_cast<int>(material_params.size());

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
    const int ref = source_table_reference_material(cfg);
    table_views = source_eos_table_cache().views_for(
        *materials[static_cast<std::size_t>(ref)].eos_tables);
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

  std::int8_t* d_hydro_active = nullptr;
  d_hydro_active = (cfg.numerics.hydro.T_start_inactive_cells == "rigid_wall")
                       ? nullptr
                       : const_cast<std::int8_t*>(state.hydro_active_device_ptr());

  const materials::CellEOSTableSelector cell_tables =
      source_cell_table_selector(state, cfg, eos_ctx);
  const int threads = core::serial_cell_block_size(n_cells);
  const int blocks = core::serial_cell_blocks(n_cells);
  qei_coupling_substep_kernel<<<blocks, threads>>>(
      state.ee.data(), state.ei.data(), state.Te.data(), state.Ti.data(),
      state.Pe.data(), state.Pi.data(),
      state.cv_e.empty() ? nullptr : state.cv_e.data(),
      state.cv_i.empty() ? nullptr : state.cv_i.data(), state.rho.data(),
      state.zbar.data(), mix_params, qei_cache.materials.data(),
      valid_volfrac ? state.volFrac.data() : nullptr, qei_cache.cell_is_void.data(),
      d_hydro_active,
      n_cells, dt_sub, cfg.numerics.floors.Te, cfg.numerics.floors.Ti,
      has_table_eos, use_first_cv_override, mat0.cv_e_override,
      mat0.eos_T_ref_eV, cfg.numerics.hydro.qei_multiplier,
      table_views.ion, table_views.electron, cell_tables,
      (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative") ? 1 : 0,
      (state.zmom_active && state.zmom_r2.size() == static_cast<std::size_t>(n_cells))
          ? state.zmom_r2.data()
          : nullptr);
  cuda_check(cudaGetLastError(),
             "apply_qei_coupling_substep kernel launch failed");
  cuda_check(core::debug_kernel_sync(),
             "apply_qei_coupling_substep kernel synchronize failed");

}

double inject_laser_source_terms(core::State& state,
                                 const core::Config& cfg,
                                 const double dt,
                                 double* E_floor_injected,
                                 int* clamp_count,
                                 const hydro::HydroEOSContext* eos_ctx) {
  const core::NvtxRange nvtx_range("source.laser_injection");
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
  const bool has_table_eos = source_table_eos_enabled(cfg, mat0);
  bool any_cv_e_override = false;
  for (const auto& mat : materials) {
    if (mat.cv_e_override > 0.0) {
      any_cv_e_override = true;
      break;
    }
  }
  const bool use_first_cv_override = any_cv_e_override && mat0.cv_e_override > 0.0;

  const int n_cells = static_cast<int>(state.rho.size());
  const char* host_path = std::getenv("TENRYU_LASER_SOURCE_HOST");
  // CUDA pow(x, 0.25) does not reproduce the legacy host libm bitwise.
  // Keep that analytic override on its original path; tabular 2T takes priority.
  bool any_t4_closure = use_first_cv_override && mat0.eos_T_ref_eV > 0.0;
  if (materials::material_closure_params_vary(cfg)) {
    for (const auto& mat : materials) {
      any_t4_closure = any_t4_closure || (!mat.is_void && mat.cv_e_override > 0.0 &&
                                          mat.eos_T_ref_eV > 0.0);
    }
  }
  const bool legacy_t4_closure = any_t4_closure &&
                                 !(cfg.main.two_temperature && has_table_eos);
  if (cfg.main.dimension == "1D_SPH" &&
      !legacy_t4_closure &&
      !(host_path != nullptr && host_path[0] == '1')) {
    return inject_laser_source_terms_device(
        state, cfg, mat0, has_table_eos, use_first_cv_override,
        E_floor_injected, clamp_count, eos_ctx);
  }
  std::vector<double> A_eff;
  std::vector<double> gamma_eff;
  compute_effective_A_gamma(cfg, state, n_cells, A_eff, gamma_eff, nullptr);
  TENRYU_ASSERT(A_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_laser_source_terms A_eff size mismatch");
  TENRYU_ASSERT(gamma_eff.size() == static_cast<std::size_t>(n_cells),
                "inject_laser_source_terms gamma_eff size mismatch");

  const double te_floor = cfg.numerics.floors.Te;
  const double ti_floor = cfg.numerics.floors.Ti;
  const bool energy_authoritative =
      (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");

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

  // Per-cell dominant-material tables (multi-material closure, 2026-09-14).
  // In 1D a cell whose material has no table (not both the electron and the
  // ion table) closes with the ideal gas of its material, as the device
  // kernel and the 1D hydro closure do (2026-09-23). The 2D closures still
  // evaluate every cell with one material's tables, so in 2D such a cell
  // keeps the first non-void material's table.
  state.ensure_cell_material_props(cfg);
  std::vector<int> cell_mat_h(static_cast<std::size_t>(n_cells), first_nonvoid);
  if (state.cell_material_index.size() == static_cast<std::size_t>(n_cells)) {
    state.cell_material_index.copy_to_host(cell_mat_h);
  }
  // Per-material closure parameters (per-cell 1D runs, 2026-09-24).
  const bool per_cell_params = materials::material_closure_params_vary(cfg);
  const std::vector<materials::MaterialClosureParams> closure_params =
      materials::material_closure_params(cfg);
  const auto cell_tables_of = [&](const std::size_t i)
      -> const materials::EOSTableTriplet* {
    const int m = cell_mat_h[i];
    if (m >= 0 && static_cast<std::size_t>(m) < materials.size() &&
        materials[static_cast<std::size_t>(m)].eos_tables &&
        // An exact ideal-gas material closes with the ideal gas (the device
        // selector's views are empty for it).
        !(state.mesh.dim == 1 &&
          use_exact_ideal_gas_hydro_backend(materials[static_cast<std::size_t>(m)]))) {
      return materials[static_cast<std::size_t>(m)].eos_tables.get();
    }
    return nullptr;
  };
  // Fallback tables: the tables' reference material (source_table_eos_enabled).
  const int ref_material = source_table_reference_material(cfg);
  const materials::EOSTableTriplet* ref_tables =
      (ref_material >= 0)
          ? materials[static_cast<std::size_t>(ref_material)].eos_tables.get()
          : nullptr;
  const auto electron_table_of = [&](const std::size_t i) -> const materials::EOSTable& {
    const auto* tabs = cell_tables_of(i);
    return (tabs != nullptr && !tabs->electron.empty()) ? tabs->electron
                                                        : ref_tables->electron;
  };
  const auto ion_table_of = [&](const std::size_t i) -> const materials::EOSTable& {
    const auto* tabs = cell_tables_of(i);
    return (tabs != nullptr && !tabs->ion.empty()) ? tabs->ion : ref_tables->ion;
  };
  const bool tableless_cells_ideal = (state.mesh.dim == 1);
  const auto cell_has_tables = [&](const std::size_t i) {
    if (!has_table_eos) {
      return false;
    }
    if (!tableless_cells_ideal) {
      return true;
    }
    const auto* tabs = cell_tables_of(i);
    return tabs != nullptr && !tabs->electron.empty() && !tabs->ion.empty();
  };
  // 1T: the total energy closes on the cell's total table, as in the 1T hydro
  // closure and the device kernel (2026-09-23; the 1T injection used the
  // ideal-gas branch for every cell).
  const auto total_table_of = [&](const std::size_t i) -> const materials::EOSTable& {
    const auto* tabs = cell_tables_of(i);
    return (tabs != nullptr && !tabs->total.empty()) ? tabs->total : ref_tables->total;
  };
  const auto cell_has_total_table = [&](const std::size_t i) {
    if (!has_table_eos) {
      return false;
    }
    if (!tableless_cells_ideal) {
      return !total_table_of(i).empty();
    }
    const auto* tabs = cell_tables_of(i);
    return tabs != nullptr && !tabs->total.empty();
  };
  const auto closure_table_of = [&](const std::size_t i) -> const materials::EOSTable& {
    return use_two_temp ? electron_table_of(i) : total_table_of(i);
  };

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
    const bool cell_tables_present = cell_has_tables(c_idx);
    const bool use_table_eos_closure =
        use_two_temp ? cell_tables_present : cell_has_total_table(c_idx);
    // Heat-capacity override of the cell's material when the materials'
    // closure parameters differ (per-cell 1D runs, 2026-09-24).
    const int cm_c = cell_mat_h[c_idx];
    const materials::MaterialClosureParams* cp_c =
        (per_cell_params && cm_c >= 0 && static_cast<std::size_t>(cm_c) < closure_params.size())
            ? &closure_params[static_cast<std::size_t>(cm_c)]
            : nullptr;
    const bool use_cv_override_c =
        (cp_c != nullptr) ? (cp_c->cv_e_override > 0.0) : use_first_cv_override;
    const double cv_override_c = (cp_c != nullptr) ? cp_c->cv_e_override : mat0.cv_e_override;
    const double T_ref_c = (cp_c != nullptr) ? cp_c->eos_T_ref_eV : mat0.eos_T_ref_eV;
    double Te_raw = 0.0;
    double Te_new = te_floor;
    if (use_table_eos_closure) {
      const SubstepTailClosure tail_e = host_high_t_tail_closure(
          closure_table_of(c_idx), rho_safe, ee[c_idx], energy_authoritative);
      if (tail_e.in_tail != 0) {
        Te_raw = tail_e.T;
        Te_new = tail_e.T;
        Pe[c_idx] = tail_e.P;
      } else {
        Te_raw = closure_table_of(c_idx).temperature_from_energy(rho_safe, ee[c_idx]);
        if (!std::isfinite(Te_raw) || Te_raw < te_floor) {
          Te_new = te_floor;
          if (!energy_authoritative || !std::isfinite(Te_raw)) {
            ee[c_idx] = closure_table_of(c_idx).energy(rho_safe, Te_new);
          }
        } else {
          Te_new = Te_raw;
        }
        Pe[c_idx] = closure_table_of(c_idx).pressure(rho_safe, Te_new);
      }
    } else if (use_cv_override_c && T_ref_c > 0.0) {
      const double T_ref = T_ref_c;
      const double T_ref3 = T_ref * T_ref * T_ref;
      const double alpha0 = cv_override_c / (4.0 * T_ref3);
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
      bool cv_mass_e_is_total = false;
      if (use_cv_override_c) {
        cv_mass_e = cv_override_c / rho_safe;
      } else if (has_table_cv_e && host_cv_e[c_idx] > 0.0) {
        cv_mass_e = host_cv_e[c_idx];
        // In 1T the state heat capacity is the total one (the 1T hydro
        // closure stores it there); the ion part used to be added a second
        // time.
        cv_mass_e_is_total = !use_two_temp;
      } else {
        cv_mass_e = z * core::constants::eV_to_erg /
                    (A_c * core::constants::proton_mass * gm1_c);
      }
      cv_mass_e = std::max(cv_mass_e, 1.0e-30);
      const double cv_mass_total = use_two_temp
                                       ? cv_mass_e
                                       : ((use_cv_override_c || cv_mass_e_is_total)
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
      if (cell_tables_present) {
        const double ei_before_floor = ei[c_idx];
        const SubstepTailClosure tail_i = host_high_t_tail_closure(
            ion_table_of(c_idx), rho_safe, ei[c_idx], energy_authoritative);
        if (tail_i.in_tail != 0) {
          Ti[c_idx] = tail_i.T;
          Pi[c_idx] = tail_i.P;
        } else {
          const double Ti_raw_table =
              ion_table_of(c_idx).temperature_from_energy(rho_safe, ei[c_idx]);
          if (!std::isfinite(Ti_raw_table) || Ti_raw_table < ti_floor) {
            ++local_clamp_count;
            Ti[c_idx] = ti_floor;
            if (!energy_authoritative || !std::isfinite(Ti_raw_table)) {
              ei[c_idx] = ion_table_of(c_idx).energy(rho_safe, ti_floor);
            }
          } else {
            Ti[c_idx] = Ti_raw_table;
          }
          Pi[c_idx] = ion_table_of(c_idx).pressure(rho_safe, Ti[c_idx]);
        }
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
                                int* clamp_count,
                                const hydro::HydroEOSContext* eos_ctx) {
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
  const bool has_table_eos = source_table_eos_enabled(cfg, mat0);
  bool any_cv_e_override = false;
  for (const auto& mat : materials) {
    if (mat.cv_e_override > 0.0) {
      any_cv_e_override = true;
      break;
    }
  }
  const bool use_first_cv_override = any_cv_e_override && mat0.cv_e_override > 0.0;

  const int n_cells = static_cast<int>(state.rho.size());
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
  std::vector<double> ee_entry;
  std::vector<double> ei_entry;
  if (per_material_deposit) {
    ee_entry.resize(state.ee.size());
    ei_entry.resize(state.ei.size());
    state.ee.copy_to_host(ee_entry.data());
    state.ei.copy_to_host(ei_entry.data());
  }

  const double skipped_energy = inject_burn_source_terms_device(
      state, cfg, mat0, has_table_eos, use_first_cv_override, dE_e, dE_i,
      E_floor_injected, clamp_count, eos_ctx);

  if (per_material_deposit) {
    std::vector<double> rho(state.rho.size(), 0.0);
    std::vector<double> vol(state.vol.size(), 0.0);
    std::vector<double> ee(state.ee.size(), 0.0);
    std::vector<double> ei(state.ei.size(), 0.0);
    state.rho.copy_to_host(rho.data());
    state.vol.copy_to_host(vol.data());
    state.ee.copy_to_host(ee.data());
    state.ei.copy_to_host(ei.data());
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
  return skipped_energy;
}

}  // namespace tenryu::coupling
