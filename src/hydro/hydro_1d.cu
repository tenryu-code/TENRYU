#include "hydro/hydro_1d.hpp"
#include "hydro/braginskii_viscosity.cuh"
#include "hydro/hydro_1d_bodies.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <initializer_list>
#include <sstream>
#include <string>
#include <utility>
#include <atomic>
#include <vector>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>

#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/fancy_iterators.cuh"
#include "core/kernel_guard.hpp"
#include "core/namelist/errors.hpp"
#include "diagnostics/diagnostics.hpp"
#include "hydro/adaptive_av_gate.hpp"
#include "hydro/artificial_viscosity.hpp"
#include "hydro/boundary.hpp"
#include "hydro/eos_context.hpp"
#include "hydro/shock_tracker.hpp"
#include "materials/eos_device.cuh"
#include "materials/eos_device_table.cuh"
#include "materials/eos_rho_e_device.cuh"
#include "materials/helmholtz_jet_device.cuh"
#include "materials/helmholtz_spline_device.cuh"
#include "materials/mie_gruneisen_device.cuh"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro {
namespace {

using tenryu::mesh::geometry_1d_face_area;
using tenryu::mesh::geometry_1d_shell_volume;

constexpr double kFourPi = 12.566370614359172953850573533118;
constexpr double kEvToErg = tenryu::core::constants::eV_to_erg;
constexpr double kProtonMass = tenryu::core::constants::proton_mass;
constexpr double kCheckerboardFilterBeta = 0.15;
constexpr double kCheckerboardFilterEps = 1.0e-30;
constexpr double kOddEvenWeightEps = 1.0e-30;
constexpr double kElectronOddEvenGradFrac = 0.1;
constexpr double kElectronOddEvenEps = 1.0e-30;
constexpr double kIonArtHeatEps = 1.0e-30;
constexpr double kIonArtHeatPressureFloor = 1.0e-30;
constexpr double kHighKVelocityDamperEps = 1.0e-30;
constexpr double kHighKVelocityDamperNoiseMin = 0.5;
constexpr double kPostShockThresholdFrac = 0.01;
constexpr double kPostShockVelocityCoreFrac = 0.2;
constexpr double kPostShockSensorEps = 1.0e-30;
constexpr double kShockTimeInactive = -1.0e300;
using persistent_1d::apply_exact_sound_speed_override;
using persistent_1d::diagnostic_cv_e_mass;
using persistent_1d::diagnostic_cv_i_mass;
using persistent_1d::enable_table_eos_writeback;
using persistent_1d::energy_inverse_input;
using persistent_1d::energy_needs_repair;
using persistent_1d::ideal_gas_cv_e_mass;
using persistent_1d::ideal_gas_cv_i_mass;
using persistent_1d::kDiagnosticIdealGasGamma;
using persistent_1d::kExactOverrideCv;
using persistent_1d::kExactOverrideNone;
using persistent_1d::kExactOverrideNoWriteback;
using persistent_1d::kExactOverridePTCs;
using persistent_1d::kExactOverridePressure;
using persistent_1d::kExactOverridePressureAndCs;
using persistent_1d::kExactOverrideSoundSpeed;
using persistent_1d::kExactOverrideTempReclosure;
using persistent_1d::kExactOverrideTemperature;
using persistent_1d::record_inverse_reclose_status;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline void sync_kernel(const char* message) {
  cuda_check(cudaGetLastError(), message);
  if (core::sync_every_kernel_enabled()) {
    cuda_check(cudaDeviceSynchronize(), message);
  }
}

inline double checkerboard_filter_beta_from_odd_even_c(const double C_oe) {
  return kCheckerboardFilterBeta * std::min(std::max(C_oe, 0.0), 1.0);
}

struct HydroTableViews {
  tenryu::materials::DeviceEOSTableView tab_ion{};
  tenryu::materials::DeviceEOSTableView tab_ele{};
  tenryu::materials::DeviceEOSTableView tab_total{};
  tenryu::materials::EOSRhoEDeviceView tab_total_rho_e{};
  tenryu::materials::HelmholtzSplineDeviceView spline_total{};
  tenryu::materials::HelmholtzJetDeviceView jet_total{};
  tenryu::materials::MieGruneisenDeviceView mie_gruneisen{};
  std::uint8_t hydro_backend_kind = 0u;
  bool supports_rho_e_reclosure = false;
};

inline bool use_helmholtz_spline_backend(const HydroTableViews& views) {
  return views.hydro_backend_kind == 1u;
}

inline bool use_helmholtz_jet_backend(const HydroTableViews& views) {
  return views.hydro_backend_kind == 2u;
}

inline bool use_exact_ideal_gas_backend(const HydroTableViews& views) {
  return views.hydro_backend_kind == 3u;
}

inline bool use_rho_e_table_backend(const HydroTableViews& views) {
  return views.hydro_backend_kind == 4u;
}

inline bool use_mie_gruneisen_backend(const HydroTableViews& views) {
  return views.hydro_backend_kind == 5u;
}

inline HydroTableViews select_hydro_table_views(const HydroEOSContext* eos_ctx) {
  HydroTableViews views{};
  if (eos_ctx == nullptr || eos_ctx->n_materials <= 0) {
    return views;
  }
  views.tab_ion = eos_ctx->ion_view(0);
  views.tab_ele = eos_ctx->electron_view(0);
  views.tab_total = eos_ctx->total_view(0);
  views.tab_total_rho_e = eos_ctx->total_rho_e_view(0);
  views.hydro_backend_kind = eos_ctx->material_hydro_backend_kind(0);
  views.supports_rho_e_reclosure = eos_ctx->material_supports_rho_e_reclosure(0);
  if (views.hydro_backend_kind == 1u) {
    views.spline_total = eos_ctx->total_helmholtz_view(0);
  } else if (views.hydro_backend_kind == 2u) {
    views.jet_total = eos_ctx->total_helmholtz_jet_view(0);
  } else if (views.hydro_backend_kind == 5u) {
    views.mie_gruneisen = eos_ctx->mie_gruneisen_view(0);
  }
  return views;
}

inline bool hydro_has_table_eos_backend_data(const HydroTableViews& views) {
  return views.tab_total.n_rho > 0 || views.tab_ion.n_rho > 0 || views.tab_ele.n_rho > 0 ||
         views.tab_total_rho_e.n_rho > 0 || views.spline_total.n_rho > 0 ||
         views.jet_total.n_rho > 0 || views.mie_gruneisen.n_rho > 0;
}

void ensure_table_cv_fields(core::State& state,
                            const HydroTableViews& views,
                            const bool required) {
  if (!required) {
    return;
  }
  const std::size_t n = state.rho.size();
  if (n == 0) {
    return;
  }
  if (state.cv_e.size() != n) {
    state.cv_e.reset(n);
  }
  if (state.cv_i.size() != n) {
    state.cv_i.reset(n);
  }
}

void validate_compatible_energy_eos_support(const core::Config& cfg,
                                            const HydroTableViews& views) {
  if (!cfg.numerics.hydro.compatible_energy) {
    return;
  }
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "Hydro1D requires at least one material");
  const auto& hydro_mat = cfg.materials.materials.front();
  const bool requires_uploaded_table =
      hydro_mat.eos_model == "tmat" && hydro_mat.hydro_eos_backend == "legacy";
  if (requires_uploaded_table && !hydro_has_table_eos_backend_data(views)) {
    throw core::namelist::ConfigError(
        "Numerics.hydro.compatible_energy=True with TMAT legacy EOS requires an "
        "initialized HydroEOSContext with uploaded TMAT tables");
  }
  const bool eos_supported =
      !hydro_has_table_eos_backend_data(views) || views.supports_rho_e_reclosure;
  if (!eos_supported) {
    throw core::namelist::ConfigError(
        "Numerics.hydro.compatible_energy=True requires an EOS backend with "
        "rho-e inverse reclosure support (ideal_gas or TMAT legacy table)");
  }
}

inline int select_exact_override_kind(const core::Config& cfg, const HydroTableViews& views) {
  if (use_exact_ideal_gas_backend(views) || use_mie_gruneisen_backend(views) ||
      !hydro_has_table_eos_backend_data(views)) {
    return kExactOverrideNone;
  }
  const std::string& exact_override = cfg.numerics.hydro.exact_override;
  if (exact_override == "pressure") {
    return kExactOverridePressure;
  }
  if (exact_override == "sound_speed") {
    return kExactOverrideSoundSpeed;
  }
  if (exact_override == "temperature") {
    return kExactOverrideTemperature;
  }
  if (exact_override == "cv") {
    return kExactOverrideCv;
  }
  if (exact_override == "temp_reclosure") {
    return kExactOverrideTempReclosure;
  }
  if (exact_override == "pressure_and_cs") {
    return kExactOverridePressureAndCs;
  }
  if (exact_override == "p_t_cs") {
    return kExactOverridePTCs;
  }
  if (exact_override == "no_writeback") {
    return kExactOverrideNoWriteback;
  }
  return kExactOverrideNone;
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

// NOH-DET intra-step probe (docs/design/nohdet_reproducer_sweep_design_20260707.md):
// env-gated, read-only order-independent XOR-fold of device arrays at bracket
// points inside the 1D Lagrangian step. TENRYU_HYDRO_STEP_HASH=1 enables;
// TENRYU_HYDRO_STEP_HASH_MIN / _MAX (inclusive, 1-based like [state_hash])
// restrict the step window. Default-inert.
struct HydroStepHashGate {
  bool enabled = false;
  long long min_step = -1;
  long long max_step = -1;
};
inline const HydroStepHashGate& hydro_step_hash_gate() {
  static const HydroStepHashGate gate = [] {
    HydroStepHashGate g;
    const char* s = std::getenv("TENRYU_HYDRO_STEP_HASH");
    g.enabled = (s != nullptr && s[0] == '1');
    if (const char* m = std::getenv("TENRYU_HYDRO_STEP_HASH_MIN")) {
      g.min_step = std::atoll(m);
    }
    if (const char* m = std::getenv("TENRYU_HYDRO_STEP_HASH_MAX")) {
      g.max_step = std::atoll(m);
    }
    return g;
  }();
  return gate;
}
inline bool hydro_step_hash_active(const long long step_1based) {
  const HydroStepHashGate& g = hydro_step_hash_gate();
  if (!g.enabled) {
    return false;
  }
  if (g.min_step >= 0 && step_1based < g.min_step) {
    return false;
  }
  if (g.max_step >= 0 && step_1based > g.max_step) {
    return false;
  }
  return true;
}
inline unsigned long long hydro_step_hash_fold(const double* device_ptr,
                                               const std::size_t n) {
  if (device_ptr == nullptr || n == 0) {
    return 0ULL;
  }
  std::vector<double> host(n);
  cudaMemcpy(host.data(), device_ptr, n * sizeof(double),
             cudaMemcpyDeviceToHost);
  unsigned long long acc = 0ULL;
  for (std::size_t i = 0; i < n; ++i) {
    unsigned long long bits;
    std::memcpy(&bits, &host[i], sizeof(bits));
    acc ^= bits + 0x9e3779b97f4a7c15ULL * static_cast<unsigned long long>(i + 1);
  }
  return acc;
}
inline void hydro_step_probe(const long long step_1based,
                             const char* tag,
                             std::initializer_list<
                                 std::pair<const char*, std::pair<const double*, std::size_t>>>
                                 fields) {
  std::ostringstream oss;
  oss << "[hydro_hash] step=" << step_1based << " tag=" << tag << std::hex;
  for (const auto& f : fields) {
    oss << " " << f.first << "="
        << hydro_step_hash_fold(f.second.first, f.second.second);
  }
  tenryu::core::log_info(oss.str());
}

__device__ inline double atomic_max_double(double* address, const double val) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (__longlong_as_double(static_cast<long long>(old)) < val) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ inline void apply_exact_override_1t(const int exact_override_kind,
                                               const double rho,
                                               const double A,
                                               const double Z,
                                               const double te_floor,
                                               double* ee,
                                               double* ei,
                                               double* Te,
                                               double* Ti,
                                               double* Pe,
                                               double* Pi,
                                               const tenryu::materials::DeviceEOSTableView tab = {},
                                               double* cv_e_out = nullptr) {
  if (exact_override_kind == kExactOverridePressure) {
    *Pe = (kDiagnosticIdealGasGamma - 1.0) * rho * (*ee);
    *Pi = 0.0;
    return;
  }
  if (exact_override_kind == kExactOverridePressureAndCs) {
    *Pe = (kDiagnosticIdealGasGamma - 1.0) * rho * (*ee);
    *Pi = 0.0;
    // cs is handled separately in the sound speed kernel
    return;
  }
  if (exact_override_kind == kExactOverridePTCs) {
    // P + T + cs exact, cv stays from table
    *Pe = (kDiagnosticIdealGasGamma - 1.0) * rho * (*ee);
    *Pi = 0.0;
    const double cv_total = diagnostic_cv_i_mass(A) + diagnostic_cv_e_mass(A, Z);
    double te = te_floor;
    if (*ee > 0.0 && cv_total > 0.0) {
      te = fmax((*ee) / cv_total, te_floor);
    }
    *Te = te;
    *Ti = te;
    // cs is handled in sound speed kernel via kExactOverridePTCs
    return;
  }
  if (exact_override_kind == kExactOverrideTemperature) {
    const double cv_total = diagnostic_cv_i_mass(A) + diagnostic_cv_e_mass(A, Z);
    double te = te_floor;
    if (*ee > 0.0 && cv_total > 0.0) {
      te = fmax((*ee) / cv_total, te_floor);
    }
    *Te = te;
    *Ti = te;
  }
  if (exact_override_kind == kExactOverrideTempReclosure && tab.n_rho > 0) {
    // Exact T from ideal gas, then consistent P from table at that T
    const double cv_total = diagnostic_cv_i_mass(A) + diagnostic_cv_e_mass(A, Z);
    double te = te_floor;
    if (*ee > 0.0 && cv_total > 0.0) {
      te = fmax((*ee) / cv_total, te_floor);
    }
    *Te = te;
    *Ti = te;
    // Re-evaluate P from table at exact T (consistent reclosure)
    const auto rb = tenryu::materials::find_rho_bracket(tab, rho);
    const double logT = log(fmax(te, 1.0e-30));
    *Pe = tenryu::materials::device_eos_pressure(tab, rb, logT);
    *Pi = 0.0;
    // Also update cv from table at exact T
    if (cv_e_out != nullptr) {
      *cv_e_out = fmax(tenryu::materials::device_eos_cv(tab, rb, logT), 0.0);
    }
    // Update stored e to match table at exact T (full consistency)
    *ee = tenryu::materials::device_eos_energy(tab, rb, logT);
    *ei = 0.0;
  }
}

// §6o.4j lesson: every debug-dump call gets a global sequence number so two
// program points can never alias the same file.
static int h1d_debug_seq_next() {
  static std::atomic<int> seq{0};
  return seq.fetch_add(1);
}
__global__ void compute_density_kernel(double* __restrict__ rho,
                                       const double* __restrict__ mass,
                                       const double* __restrict__ vol,
                                       const std::uint8_t* __restrict__ cell_is_void,
                                       const int c_begin,
                                       const int c_end,
                                       const int own_begin,
                                       const int own_end,
                                       const int n_cells,
                                       const double rho_floor,
                                       int* __restrict__ rho_clamp_count) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  persistent_1d::compute_density_kernel_body(
      i, rho, mass, vol, cell_is_void, n_cells, rho_floor,
      (i >= own_begin && i < own_end) ? rho_clamp_count : nullptr);
}

__global__ void enforce_1t_closure_kernel(double* __restrict__ ee,
                                          double* __restrict__ ei,
                                          double* __restrict__ Te,
                                          double* __restrict__ Ti,
                                          double* __restrict__ Pe,
                                          double* __restrict__ Pi,
                                          const double* __restrict__ rho,
                                          const double* __restrict__ zbar,
                                          const int c_begin,
                                          const int c_end,
                                          const int own_begin,
                                          const int own_end,
                                          const int n_cells,
                                          const double* __restrict__ gamma_eff,
                                          const double* __restrict__ A_eff,
                                          const double fallback_z,
                                          const double cv_e_override,
                                          const double te_floor,
                                          const tenryu::materials::DeviceEOSTableView tab_total,
                                          const tenryu::materials::EOSRhoEDeviceView
                                              tab_total_rho_e,
                                          const tenryu::materials::HelmholtzSplineDeviceView
                                              spline_total,
                                          const tenryu::materials::HelmholtzJetDeviceView
                                              jet_total,
                                          const bool use_helmholtz,
                                          const bool use_helmholtz_jet,
                                          const bool use_rho_e_table,
                                          const bool use_exact_ideal_gas,
                                          const bool eos_writeback,
                                          const bool preserve_table_energy,
                                          const bool energy_authoritative,
                                          int* __restrict__ inverse_clamp_count,
                                          int* __restrict__ inverse_failure_count,
                                          const int exact_override_kind,
                                          double* __restrict__ cv_e_out,
                                          double* __restrict__ cv_i_out) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  const double rho_i = fmax(rho[i], 1.0e-30);
  // Per-cell effective ideal-gas properties (see State::ensure_cell_material_props).
  const double gamma = fmax(gamma_eff[i], 1.0 + 1.0e-12);
  const double A = fmax(A_eff[i], 1.0e-12);
  const bool allow_table_energy_writeback =
      !preserve_table_energy && enable_table_eos_writeback(eos_writeback, exact_override_kind);
  if (use_exact_ideal_gas) {
    persistent_1d::enforce_1t_closure_ideal_kernel_body(
        i, ee, ei, Te, Ti, Pe, Pi, rho, zbar, gamma_eff, A_eff, fallback_z,
        cv_e_override, te_floor, cv_e_out, cv_i_out);
    return;
  }
  ee[i] += ei[i];
  if (use_helmholtz && spline_total.n_rho > 0 && rho_i >= exp(spline_total.log_rho_min)) {
    const double e_raw = ee[i];
    const double e = energy_inverse_input(e_raw);
    const bool repair_energy = energy_needs_repair(e_raw);
    const double T =
        fmax(tenryu::materials::device_helmholtz_T_from_e(spline_total, rho_i, e), te_floor);
    const double logT = log(fmax(T, 1.0e-30));
    const auto thermo =
        tenryu::materials::device_helmholtz_eval_thermo(spline_total, rho_i, logT);
    if (allow_table_energy_writeback || repair_energy) {
      ee[i] = thermo.energy;
    }
    ei[i] = 0.0;
    Te[i] = T;
    Ti[i] = T;
    Pe[i] = thermo.pressure;
    Pi[i] = 0.0;
    if (cv_e_out != nullptr) {
      cv_e_out[i] = thermo.cv;
    }
    if (cv_i_out != nullptr) {
      cv_i_out[i] = 0.0;
    }
    apply_exact_override_1t(exact_override_kind, rho[i], A, fallback_z, te_floor, &ee[i], &ei[i],
                            &Te[i], &Ti[i], &Pe[i], &Pi[i]);
    return;
  }
  if (use_helmholtz_jet && jet_total.n_rho > 0 && rho_i >= exp(jet_total.log_rho_min)) {
    const double e_raw = ee[i];
    const double e = energy_inverse_input(e_raw);
    const bool repair_energy = energy_needs_repair(e_raw);
    const double T =
        fmax(tenryu::materials::device_helmholtz_jet_T_from_e(jet_total, rho_i, e), te_floor);
    const double logT = log(fmax(T, 1.0e-30));
    const auto thermo =
        tenryu::materials::device_helmholtz_jet_eval_thermo(jet_total, rho_i, logT);
    if (allow_table_energy_writeback || repair_energy) {
      ee[i] = thermo.energy;
    }
    ei[i] = 0.0;
    Te[i] = T;
    Ti[i] = T;
    Pe[i] = thermo.pressure;
    Pi[i] = 0.0;
    if (cv_e_out != nullptr) {
      cv_e_out[i] = thermo.cv;
    }
    if (cv_i_out != nullptr) {
      cv_i_out[i] = 0.0;
    }
    apply_exact_override_1t(exact_override_kind, rho[i], A, fallback_z, te_floor, &ee[i], &ei[i],
                            &Te[i], &Ti[i], &Pe[i], &Pi[i]);
    return;
  }
  if (use_rho_e_table && tab_total_rho_e.n_rho > 0 &&
      rho_i >= tenryu::materials::eos_rho_e_rho_min(tab_total_rho_e)) {
    const double e_raw = ee[i];
    const bool repair_energy = energy_needs_repair(e_raw);
    const auto thermo = tenryu::materials::device_eos_rho_e_eval(
        tab_total_rho_e, rho_i, energy_inverse_input(e_raw));
    if (isfinite(thermo.P) && isfinite(thermo.T) && thermo.T >= te_floor) {
      if (allow_table_energy_writeback || repair_energy) {
        ee[i] = thermo.e_clamped;
      }
      ei[i] = 0.0;
      Te[i] = thermo.T;
      Ti[i] = thermo.T;
      Pe[i] = thermo.P;
      Pi[i] = 0.0;
      if (cv_e_out != nullptr) {
        cv_e_out[i] = fmax(thermo.cv, 0.0);
      }
      if (cv_i_out != nullptr) {
        cv_i_out[i] = 0.0;
      }
      apply_exact_override_1t(exact_override_kind, rho[i], A, fallback_z, te_floor, &ee[i],
                              &ei[i], &Te[i], &Ti[i], &Pe[i], &Pi[i]);
      return;
    }
    if (tab_total.n_rho > 0 && rho_i >= exp(tab_total.log_rho_min)) {
      const auto rb = tenryu::materials::find_rho_bracket(tab_total, rho_i);
      const double T = fmax(te_floor, 1.0e-30);
      const double logT = log(T);
      ee[i] = tenryu::materials::device_eos_energy(tab_total, rb, logT);
      ei[i] = 0.0;
      Te[i] = T;
      Ti[i] = T;
      Pe[i] = tenryu::materials::device_eos_pressure(tab_total, rb, logT);
      Pi[i] = 0.0;
      const double cv = fmax(tenryu::materials::device_eos_cv(tab_total, rb, logT), 0.0);
      if (cv_e_out != nullptr) {
        cv_e_out[i] = cv;
      }
      if (cv_i_out != nullptr) {
        cv_i_out[i] = 0.0;
      }
      apply_exact_override_1t(exact_override_kind, rho[i], A, fallback_z, te_floor, &ee[i],
                              &ei[i], &Te[i], &Ti[i], &Pe[i], &Pi[i]);
      return;
    }
  }
  if (tab_total.n_rho > 0 && rho_i >= exp(tab_total.log_rho_min)) {
    const double e_raw = ee[i];
    const double e = energy_inverse_input(e_raw);
    const bool repair_energy = energy_needs_repair(e_raw);
    const auto inv =
        energy_authoritative
            ? tenryu::materials::device_inverse_reclose_with_high_t_tail(
                  tab_total, rho_i, e, te_floor)
            : tenryu::materials::device_inverse_reclose(tab_total, rho_i, e,
                                                        te_floor);
    record_inverse_reclose_status(
        inv, (i >= own_begin && i < own_end) ? inverse_clamp_count : nullptr,
        (i >= own_begin && i < own_end) ? inverse_failure_count : nullptr);
    const bool clamp_veto = energy_authoritative && !repair_energy &&
        inv.bracket_failure == 0 &&
        (inv.lower_clamp != 0 || inv.upper_clamp != 0);
    if (!clamp_veto &&
        (allow_table_energy_writeback || repair_energy ||
         inv.bracket_failure != 0 || inv.lower_clamp != 0 ||
         inv.upper_clamp != 0)) {
      ee[i] = inv.energy;
    }
    ei[i] = 0.0;
    Te[i] = inv.T;
    Ti[i] = inv.T;
    Pe[i] = inv.pressure;
    Pi[i] = 0.0;
    if (cv_e_out != nullptr) {
      cv_e_out[i] = inv.cv;
    }
    if (cv_i_out != nullptr) {
      cv_i_out[i] = 0.0;
    }
    apply_exact_override_1t(exact_override_kind, rho[i], A, fallback_z, te_floor, &ee[i], &ei[i],
                            &Te[i], &Ti[i], &Pe[i], &Pi[i], tab_total,
                            cv_e_out != nullptr ? &cv_e_out[i] : nullptr);
    return;
  }

  double cv_total = 0.0;
  if (cv_e_override > 0.0) {
    cv_total = cv_e_override / rho_i;
  } else {
    const double z = (zbar != nullptr) ? fmax(zbar[i], 0.0) : fallback_z;
    cv_total = (1.0 + z) * kEvToErg / (A * kProtonMass * (gamma - 1.0));
  }

  const double e = fmax(ee[i], 0.0);
  ee[i] = e;
  ei[i] = 0.0;

  double te = te_floor;
  if (e > 0.0 && cv_total > 0.0) {
    te = fmax(e / cv_total, te_floor);
  }

  Te[i] = te;
  Ti[i] = te;
  Pi[i] = 0.0;
  Pe[i] = (gamma - 1.0) * rho[i] * e;
  if (cv_e_out != nullptr) {
    cv_e_out[i] = cv_total;
  }
  if (cv_i_out != nullptr) {
    cv_i_out[i] = 0.0;
  }
  apply_exact_override_1t(exact_override_kind, rho[i], A, fallback_z, te_floor, &ee[i], &ei[i],
                          &Te[i], &Ti[i], &Pe[i], &Pi[i]);
}

__global__ void enforce_2t_closure_kernel(double* __restrict__ ee,
                                          double* __restrict__ ei,
                                          double* __restrict__ Te,
                                          double* __restrict__ Ti,
                                          double* __restrict__ Pe,
                                          double* __restrict__ Pi,
                                          const double* __restrict__ rho,
                                          const double* __restrict__ zbar,
                                          const int c_begin,
                                          const int c_end,
                                          const int own_begin,
                                          const int own_end,
                                          const int n_cells,
                                          const double* __restrict__ gamma_eff,
                                          const double* __restrict__ A_eff,
                                          const double fallback_z,
                                          const double cv_e_override,
                                          const double te_floor,
                                          const double ti_floor,
                                          const tenryu::materials::DeviceEOSTableView tab_ion,
                                          const tenryu::materials::DeviceEOSTableView tab_ele,
                                          const tenryu::materials::EOSRhoEDeviceView
                                              tab_total_rho_e,
                                          const tenryu::materials::HelmholtzSplineDeviceView
                                              spline_total,
                                          const tenryu::materials::HelmholtzJetDeviceView
                                              jet_total,
                                          const tenryu::materials::MieGruneisenDeviceView
                                              mie_gruneisen,
                                          const bool use_helmholtz,
                                          const bool use_helmholtz_jet,
                                          const bool use_rho_e_table,
                                          const bool use_mie_gruneisen,
                                          const bool use_exact_ideal_gas,
                                          const bool eos_writeback,
                                          const bool preserve_table_energy,
                                          const bool energy_authoritative,
                                          int* __restrict__ inverse_clamp_count,
                                          int* __restrict__ inverse_failure_count,
                                          const int exact_override_kind,
                                          double* __restrict__ cv_e_out,
                                          double* __restrict__ cv_i_out) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  persistent_1d::enforce_2t_closure_kernel_body(
      i, ee, ei, Te, Ti, Pe, Pi, rho, zbar, n_cells, gamma_eff, A_eff,
      fallback_z, cv_e_override, te_floor, ti_floor, tab_ion, tab_ele,
      tab_total_rho_e, spline_total, jet_total, mie_gruneisen, use_helmholtz,
      use_helmholtz_jet, use_rho_e_table, use_mie_gruneisen, use_exact_ideal_gas,
      eos_writeback, preserve_table_energy, energy_authoritative,
      (i >= own_begin && i < own_end) ? inverse_clamp_count : nullptr,
      (i >= own_begin && i < own_end) ? inverse_failure_count : nullptr,
      exact_override_kind, cv_e_out, cv_i_out);
}

__global__ void compute_sound_speed_1t_kernel(double* __restrict__ cs,
                                              const double* __restrict__ ee,
                                              const double* __restrict__ Te,
                                              const double* __restrict__ rho,
                                              const double* __restrict__ Pe,
                                              const double* __restrict__ Pi,
                                              const tenryu::materials::DeviceEOSTableView tab_total,
                                              const tenryu::materials::EOSRhoEDeviceView
                                                  tab_total_rho_e,
                                              const tenryu::materials::HelmholtzSplineDeviceView
                                                  spline_total,
                                              const tenryu::materials::HelmholtzJetDeviceView
                                                  jet_total,
                                              const double* __restrict__ cv_arr,
                                              const int c_begin,
                                              const int c_end,
                                              const int n_cells,
                                              const bool use_helmholtz,
                                              const bool use_helmholtz_jet,
                                              const bool use_rho_e_table,
                                              const bool use_exact_ideal_gas,
                                              const int exact_override_kind,
                                              const double* __restrict__ gamma_eff) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  const double rho_i = fmax(rho[i], 1.0e-30);
  const double gamma = fmax(gamma_eff[i], 1.0 + 1.0e-12);
  if (use_exact_ideal_gas) {
    persistent_1d::compute_sound_speed_1t_ideal_kernel_body(
        i, cs, ee, gamma_eff);
    return;
  }
  if (use_helmholtz && spline_total.n_rho > 0 && rho_i >= exp(spline_total.log_rho_min)) {
    const double T = (Te != nullptr)
                         ? fmax(Te[i], 1.0e-30)
                         : fmax(tenryu::materials::device_helmholtz_T_from_e(
                                    spline_total, rho_i, ee[i]),
                                1.0e-30);
    cs[i] = tenryu::materials::device_helmholtz_eval_thermo(spline_total, rho_i, log(T))
                .sound_speed;
    cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
    return;
  }
  if (use_helmholtz_jet && jet_total.n_rho > 0 && rho_i >= exp(jet_total.log_rho_min)) {
    const double T = (Te != nullptr)
                         ? fmax(Te[i], 1.0e-30)
                         : fmax(tenryu::materials::device_helmholtz_jet_T_from_e(
                                    jet_total, rho_i, ee[i]),
                                1.0e-30);
    cs[i] = tenryu::materials::device_helmholtz_jet_eval_thermo(jet_total, rho_i, log(T))
                .sound_speed;
    cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
    return;
  }
  if (use_rho_e_table && tab_total_rho_e.n_rho > 0 &&
      rho_i >= tenryu::materials::eos_rho_e_rho_min(tab_total_rho_e)) {
    const auto thermo = tenryu::materials::device_eos_rho_e_eval(tab_total_rho_e, rho_i, ee[i]);
    cs[i] = sqrt(fmax(thermo.cs2, 0.0));
    cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
    return;
  }
  if (tab_total.n_rho > 0 && rho_i >= exp(tab_total.log_rho_min)) {
    const auto rb = tenryu::materials::find_rho_bracket(tab_total, rho_i);
    const double e = (tab_total.n_rho > 0) ? ee[i] : fmax(ee[i], 0.0);
    const double T =
        fmax(tenryu::materials::device_eos_T_from_e_monotone(tab_total, rb, e), 1.0e-30);
    const double logT = log(T);
    const double cv =
        (cv_arr != nullptr)
            ? fmax(cv_arr[i], 0.0)
            : fmax(tenryu::materials::device_eos_cv(tab_total, rb, logT), 0.0);
    cs[i] = tenryu::materials::device_eos_sound_speed(tab_total, rb, logT, rho_i, cv);
    cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
    return;
  }

  const double e = (tab_total.n_rho > 0) ? ee[i] : fmax(ee[i], 0.0);
  cs[i] = sqrt(gamma * (gamma - 1.0) * e);
  cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
}

__global__ void compute_sound_speed_2t_kernel(double* __restrict__ cs,
                                              const double* __restrict__ rho,
                                              const double* __restrict__ ee,
                                              const double* __restrict__ ei,
                                              const double* __restrict__ Pe,
                                              const double* __restrict__ Pi,
                                              const double* __restrict__ Te,
                                              const double* __restrict__ Ti,
                                              const tenryu::materials::DeviceEOSTableView tab_ion,
                                              const tenryu::materials::DeviceEOSTableView tab_ele,
                                              const tenryu::materials::EOSRhoEDeviceView
                                                  tab_total_rho_e,
                                              const tenryu::materials::HelmholtzSplineDeviceView
                                                  spline_total,
                                              const tenryu::materials::HelmholtzJetDeviceView
                                                  jet_total,
                                              const tenryu::materials::MieGruneisenDeviceView
                                                  mie_gruneisen,
                                              const double* __restrict__ cv_i_arr,
                                              const double* __restrict__ cv_e_arr,
                                              const int c_begin,
                                              const int c_end,
                                              const int n_cells,
                                              const bool use_helmholtz,
                                              const bool use_helmholtz_jet,
                                              const bool use_rho_e_table,
                                              const bool use_mie_gruneisen,
                                              const bool use_exact_ideal_gas,
                                              const int exact_override_kind,
                                              const double* __restrict__ gamma_eff) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  persistent_1d::compute_sound_speed_2t_kernel_body(
      i, cs, rho, ee, ei, Pe, Pi, Te, Ti, tab_ion, tab_ele,
      tab_total_rho_e, spline_total, jet_total, mie_gruneisen, cv_i_arr,
      cv_e_arr, n_cells, use_helmholtz, use_helmholtz_jet, use_rho_e_table,
      use_mie_gruneisen, use_exact_ideal_gas, exact_override_kind, gamma_eff);
}

__global__ void compute_node_activity_kernel(std::uint8_t* __restrict__ node_active,
                                             const std::int8_t* __restrict__ hydro_active,
                                             const int c_begin,
                                             const int c_end,
                                             const int n_cells) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  persistent_1d::compute_node_activity_kernel_body(
      i, node_active, hydro_active, n_cells);
}

__global__ void zero_center_node_kernel(std::uint8_t* __restrict__ node_active,
                                        const int n_nodes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i != 0 || n_nodes <= 0) {
    return;
  }

  persistent_1d::zero_center_node_kernel_body(i, node_active, n_nodes);
}

template <bool kHasExtra>
__global__ void build_cell_pq_kernel(double* __restrict__ pq,
                                     const double* __restrict__ Pe,
                                     const double* __restrict__ Pi,
                                     const double* __restrict__ Qvisc,
                                     const double* __restrict__ p_extra,
                                     const std::int8_t* __restrict__ hydro_active,
                                     const int c_begin,
                                     const int c_end,
                                     const int n_cells) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  persistent_1d::build_cell_pq_kernel_body<kHasExtra>(
      i, pq, Pe, Pi, Qvisc, p_extra, hydro_active, n_cells);
}

__global__ void filter_pq_checkerboard_1d_kernel(
    double* __restrict__ pq_filtered,
    const double* __restrict__ pq_raw,
    const double* __restrict__ rho,
    const double* __restrict__ Qvisc,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double beta_cb) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (!active) {
    pq_filtered[i] = pq_raw[i];
    return;
  }

  if (i <= 1 || i >= n_cells - 2) {
    pq_filtered[i] = pq_raw[i];
    return;
  }

  if (hydro_active != nullptr &&
      (hydro_active[i - 2] == 0 || hydro_active[i - 1] == 0 ||
       hydro_active[i + 1] == 0 || hydro_active[i + 2] == 0)) {
    pq_filtered[i] = pq_raw[i];
    return;
  }

  // Keep the shock-supporting cell pressure intact, but still allow the
  // adjacent Qvisc=0 cells to smooth a persistent odd-even pattern.
  if (Qvisc[i] > 0.0) {
    pq_filtered[i] = pq_raw[i];
    return;
  }

  const double dpLL = pq_raw[i - 1] - pq_raw[i - 2];
  const double dpL = pq_raw[i] - pq_raw[i - 1];
  const double dpR = pq_raw[i + 1] - pq_raw[i];
  const double dpRR = pq_raw[i + 2] - pq_raw[i + 1];
  const double drhoLL = rho[i - 1] - rho[i - 2];
  const double drhoL = rho[i] - rho[i - 1];
  const double drhoR = rho[i + 1] - rho[i];
  const double drhoRR = rho[i + 2] - rho[i + 1];
  const bool center_is_extremum = (dpL * dpR < 0.0) && (drhoL * drhoR < 0.0);
  const bool left_neighbor_is_extremum =
      (dpLL * dpL < 0.0) && (drhoLL * drhoL < 0.0);
  const bool right_neighbor_is_extremum =
      (dpR * dpRR < 0.0) && (drhoR * drhoRR < 0.0);
  if (center_is_extremum && left_neighbor_is_extremum &&
      right_neighbor_is_extremum) {
    const double osc_p_left =
        fmin(fabs(dpLL), fabs(dpL)) /
        fmax(fmax(fabs(dpLL), fabs(dpL)), kCheckerboardFilterEps);
    const double osc_p_center =
        fmin(fabs(dpL), fabs(dpR)) /
        fmax(fmax(fabs(dpL), fabs(dpR)), kCheckerboardFilterEps);
    const double osc_p_right =
        fmin(fabs(dpR), fabs(dpRR)) /
        fmax(fmax(fabs(dpR), fabs(dpRR)), kCheckerboardFilterEps);
    const double osc_rho_left =
        fmin(fabs(drhoLL), fabs(drhoL)) /
        fmax(fmax(fabs(drhoLL), fabs(drhoL)), kCheckerboardFilterEps);
    const double osc_rho_center =
        fmin(fabs(drhoL), fabs(drhoR)) /
        fmax(fmax(fabs(drhoL), fabs(drhoR)), kCheckerboardFilterEps);
    const double osc_rho_right =
        fmin(fabs(drhoR), fabs(drhoRR)) /
        fmax(fmax(fabs(drhoR), fabs(drhoRR)), kCheckerboardFilterEps);
    const double W_cb = fmin(
        fmin(fmin(osc_p_left, osc_p_center), osc_p_right),
        fmin(fmin(osc_rho_left, osc_rho_center), osc_rho_right));
    pq_filtered[i] =
        pq_raw[i] +
        beta_cb * W_cb *
            (0.5 * (pq_raw[i - 1] - 2.0 * pq_raw[i] + pq_raw[i + 1]));
    return;
  }

  pq_filtered[i] = pq_raw[i];
}

__global__ void compute_odd_even_weight_1d_kernel(
    double* __restrict__ W_oe,
    const double* __restrict__ rho,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  if (i == 0 || i == n_cells - 1) {
    W_oe[i] = 0.0;
    return;
  }

  if (hydro_active != nullptr) {
    if (hydro_active[i - 1] == 0 || hydro_active[i] == 0 ||
        hydro_active[i + 1] == 0) {
      W_oe[i] = 0.0;
      return;
    }
  }

  const double s_im1 = 1.0 / fmax(rho[i - 1], kOddEvenWeightEps);
  const double s_i = 1.0 / fmax(rho[i], kOddEvenWeightEps);
  const double s_ip1 = 1.0 / fmax(rho[i + 1], kOddEvenWeightEps);

  const double dsL = s_i - s_im1;
  const double dsR = s_ip1 - s_i;
  if (dsL * dsR < 0.0) {
    W_oe[i] = fmin(fabs(dsL), fabs(dsR)) /
              fmax(fmax(fabs(dsL), fabs(dsR)), kOddEvenWeightEps);
    return;
  }

  W_oe[i] = 0.0;
}

template <int GEOM>
__device__ inline double odd_even_mu_1d_device(
    const int i,
    const double* __restrict__ W_oe,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const double* __restrict__ node_r,
    const double* __restrict__ shock_time,
    const double* __restrict__ Qvisc,
    const double t_current,
    const double decay_cells,
    const double C_oe,
    const double C_psv,
    const double* __restrict__ C_psv_cell,
    const double qvisc_max) {
  if (i < 0 || rho[i] <= 0.0 || cs[i] <= 0.0) {
    return 0.0;
  }

  double damping_weight = 0.0;
  if (C_oe > 0.0 && W_oe != nullptr && W_oe[i] > 0.0) {
    damping_weight += C_oe * W_oe[i];
  }
  const double C_psv_eff =
      (C_psv_cell != nullptr) ? C_psv_cell[i] : C_psv;
  if (C_psv_eff > 0.0 && shock_time != nullptr) {
    const double q_core_threshold =
        kPostShockVelocityCoreFrac * fmax(qvisc_max, 0.0);
    if (!(qvisc_max > 0.0) || Qvisc == nullptr || Qvisc[i] <= q_core_threshold) {
      const double dr_i = fmax(node_r[i + 1] - node_r[i], 0.0);
      const double tau_decay =
          decay_cells * dr_i / fmax(cs[i], kPostShockSensorEps);
      if (tau_decay > 0.0) {
        const double age = fmax(t_current - shock_time[i], 0.0);
        damping_weight += C_psv_eff * exp(-age / tau_decay);
      }
    }
  }

  if (!(damping_weight > 0.0)) {
    return 0.0;
  }

  double area_avg;
  if constexpr (GEOM == 0) {
    area_avg =
        0.5 * kFourPi * (node_r[i] * node_r[i] + node_r[i + 1] * node_r[i + 1]);
  } else {
    area_avg = 0.5 * (geometry_1d_face_area(GEOM, node_r[i]) +
                      geometry_1d_face_area(GEOM, node_r[i + 1]));
  }
  return damping_weight * rho[i] * cs[i] * area_avg;
}

template <int GEOM>
__global__ void apply_odd_even_damping_1d_kernel(
    double* __restrict__ accel,
    double* __restrict__ heat_oe,
    double* __restrict__ damping_pair_force,
    const double* __restrict__ W_oe,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const double* __restrict__ mass,
    const double* __restrict__ shock_time,
    const double* __restrict__ Qvisc,
    const std::uint8_t* __restrict__ node_active,
    const int n_begin,
    const int n_end,
    const int n_cells,
    const double t_current,
    const double decay_cells,
    const double C_oe,
    const double C_psv,
    const double* __restrict__ C_psv_cell,
    const double qvisc_max,
    const int bc_type) {
  const int j = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = n_cells + 1;
  if (j >= n_end) {
    return;
  }

  if (j < n_cells) {
    const double mu = odd_even_mu_1d_device<GEOM>(
        j, W_oe, rho, cs, node_r, shock_time, Qvisc, t_current, decay_cells,
        C_oe, C_psv, C_psv_cell, qvisc_max);
    const double du = node_u[j + 1] - node_u[j];
    if (heat_oe != nullptr) {
      heat_oe[j] = mu * du * du;
    }
    if (damping_pair_force != nullptr) {
      damping_pair_force[j] = mu * du;
    }
  }

  if (node_active[j] == 0u) {
    return;
  }

  if (j == n_nodes - 1 &&
      (bc_type == static_cast<int>(HydroBoundaryType::FIXED) ||
       bc_type == static_cast<int>(HydroBoundaryType::REFLECT))) {
    return;
  }

  double force = 0.0;
  if (j > 0) {
    const double mu_left = odd_even_mu_1d_device<GEOM>(
        j - 1, W_oe, rho, cs, node_r, shock_time, Qvisc, t_current, decay_cells,
        C_oe, C_psv, C_psv_cell, qvisc_max);
    force -= mu_left * (node_u[j] - node_u[j - 1]);
  }
  if (j < n_cells) {
    const double mu_right = odd_even_mu_1d_device<GEOM>(
        j, W_oe, rho, cs, node_r, shock_time, Qvisc, t_current, decay_cells,
        C_oe, C_psv, C_psv_cell, qvisc_max);
    force += mu_right * (node_u[j + 1] - node_u[j]);
  }

  double node_mass = 0.0;
  if (j == 0) {
    node_mass = 0.5 * mass[0];
  } else if (j < n_cells) {
    node_mass = 0.5 * (mass[j - 1] + mass[j]);
  } else {
    node_mass = 0.5 * mass[n_cells - 1];
  }

  if (node_mass > 0.0) {
    accel[j] += force / node_mass;
  }
}

template <int GEOM>
__global__ void compute_acceleration_1d_kernel(double* __restrict__ accel,
                                               const double* __restrict__ mass,
                                               const double* __restrict__ x_r,
                                               const double* __restrict__ cell_pq,
                                               const std::uint8_t* __restrict__ node_active,
                                               const int n_begin,
                                               const int n_end,
                                               const int n_cells,
                                               const double* boundary_pe,
                                               const double* boundary_pi,
                                               const double* boundary_qvisc,
                                               const double boundary_pressure,
                                               const int bc_type) {
  const int j = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = n_cells + 1;
  if (j >= n_end) {
    return;
  }

  const double pe = boundary_pe[n_cells - 1];
  const double pi = boundary_pi[n_cells - 1];
  const double qvisc = boundary_qvisc[n_cells - 1];
  double ghost_pq = qvisc;
  if (bc_type == static_cast<int>(HydroBoundaryType::FIXED) ||
      bc_type == static_cast<int>(HydroBoundaryType::REFLECT)) {
    ghost_pq = pe + pi + qvisc;
  } else if (bc_type == static_cast<int>(HydroBoundaryType::PRESSURE)) {
    ghost_pq = boundary_pressure + qvisc;
  }

  persistent_1d::compute_acceleration_1d_kernel_body<GEOM>(
      j, accel, mass, x_r, cell_pq, node_active, n_cells, ghost_pq, bc_type);
}

template <int GEOM>
__global__ void gamma_r_force_work_1d_kernel(
    double* __restrict__ w_r,
    const double* __restrict__ p_extra,
    const double* __restrict__ r_half,
    const double* __restrict__ u_old,
    const double* __restrict__ u_new,
    const std::uint8_t* __restrict__ node_active,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  persistent_1d::gamma_r_force_work_1d_kernel_body<GEOM>(
      c, w_r, p_extra, r_half, u_old, u_new, node_active, n_cells, dt);
}

__global__ void predictor_update_kernel(double* __restrict__ v_r,
                                        double* __restrict__ x_r,
                                        double* __restrict__ u_half,
                                        const double* __restrict__ u_old,
                                        const double* __restrict__ r_old,
                                        const double* __restrict__ accel,
                                        const std::uint8_t* __restrict__ node_active,
                                        const int n_begin,
                                        const int n_end,
                                        const int n_nodes,
                                        const double dt) {
  const int j = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_end) {
    return;
  }

  persistent_1d::predictor_update_kernel_body(
      j, v_r, x_r, u_half, u_old, r_old, accel, node_active, n_nodes, dt);
}

// k01 P0-2 (AI review 2026-07-26, time_integrator="midpoint_v2"): advance the
// internal energies to the half step with the t^n pressure/AV over the
// predictor's geometric half-step volume change, so the mid-step EOS closure
// evaluates a genuine time-centered P^{n+1/2} = P(rho^{n+1/2}, e^{n+1/2})
// instead of the stale-energy P(rho^{n+1/2}, e^n). This is a stage state
// only: every corrector energy path re-bases on e_old, so the stage values
// are discarded after the midpoint force/pressure evaluation (no
// conservation ledger entry; the fmax(.,0) stage clamp is transient too).
__global__ void predictor_energy_half_step_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ vol_half,
    const double* __restrict__ vol_old,
    const double* __restrict__ mass,
    const double* __restrict__ Pe_old,
    const double* __restrict__ Pi_old,
    const double* __restrict__ Q_old,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_cells,
    const int use_two_temp,
    const int q_heat_to_electron) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[i] == 0) {
    return;
  }
  if (cell_is_void != nullptr && cell_is_void[i] != 0u) {
    return;
  }
  const double m = mass[i];
  if (!(m > 0.0)) {
    return;
  }
  const double dV = vol_half[i] - vol_old[i];
  if (use_two_temp != 0) {
    double de_e = -Pe_old[i] * dV / m;
    double de_i = -Pi_old[i] * dV / m;
    const double q_work = -Q_old[i] * dV / m;
    if (q_heat_to_electron != 0) {
      de_e += q_work;
    } else {
      de_i += q_work;
    }
    ee[i] = fmax(ee[i] + de_e, 0.0);
    ei[i] = fmax(ei[i] + de_i, 0.0);
  } else {
    const double de = -(Pe_old[i] + Pi_old[i] + Q_old[i]) * dV / m;
    ee[i] = fmax(ee[i] + de, 0.0);
  }
}

__global__ void corrector_update_kernel(double* __restrict__ v_r,
                                        double* __restrict__ x_r,
                                        const double* __restrict__ u_old,
                                        const double* __restrict__ r_old,
                                        const double* __restrict__ u_half,
                                        const double* __restrict__ a_half,
                                        const std::uint8_t* __restrict__ node_active,
                                        const int n_begin,
                                        const int n_end,
                                        const int n_nodes,
                                        const double dt) {
  const int j = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_end) {
    return;
  }

  persistent_1d::corrector_update_kernel_body(
      j, v_r, x_r, u_old, r_old, u_half, a_half, node_active, n_nodes, dt);
}

__global__ void propagate_void_node_displacement_kernel(
    double* __restrict__ x_r,
    double* __restrict__ v_r,
    const double* __restrict__ r_old,
    const std::uint8_t* __restrict__ node_active,
    const int n_nodes) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i != 0) {
    return;
  }

  persistent_1d::propagate_void_node_displacement_kernel_body(
      i, x_r, v_r, r_old, node_active, n_nodes);
}

__global__ void copy_array_kernel(double* __restrict__ dst,
                                  const double* __restrict__ src,
                                  const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  persistent_1d::copy_array_kernel_body(i, dst, src, n);
}

__global__ void add_arrays_inplace_kernel(double* __restrict__ dst,
                                          const double* __restrict__ src,
                                          const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  dst[i] += src[i];
}

__global__ void average_arrays_kernel(double* __restrict__ out,
                                      const double* __restrict__ a,
                                      const double* __restrict__ b,
                                      const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  persistent_1d::average_arrays_kernel_body(i, out, a, b, n);
}

template <int GEOM>
__device__ inline double compatible_volume_change_1d(
    const double* __restrict__ r_new,
    const double* __restrict__ r_old,
    const double* __restrict__ u_half,
    const int i,
    const double dt) {
  const double r_lo_half = 0.5 * (r_old[i] + r_new[i]);
  const double r_hi_half = 0.5 * (r_old[i + 1] + r_new[i + 1]);
  double area_lo;
  if constexpr (GEOM == 0) {
    area_lo = kFourPi * r_lo_half * r_lo_half;
  } else {
    area_lo = geometry_1d_face_area(GEOM, r_lo_half);
  }
  double area_hi;
  if constexpr (GEOM == 0) {
    area_hi = kFourPi * r_hi_half * r_hi_half;
  } else {
    area_hi = geometry_1d_face_area(GEOM, r_hi_half);
  }
  return dt * (area_hi * u_half[i + 1] - area_lo * u_half[i]);
}

template <int GEOM>
__global__ void energy_update_with_old_volume_kernel(
    double* __restrict__ ee,
    const double* __restrict__ e_old,
    const double* __restrict__ r_new,
    const double* __restrict__ r_old,
    const double* __restrict__ u_half,
    const double* __restrict__ vol,
    const double* __restrict__ vol_old,
    const double* __restrict__ mass,
    const double* __restrict__ P_half,
    const double* __restrict__ Q_half,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt,
    const int compatible_energy,
    double* __restrict__ eta_compatible,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  persistent_1d::energy_update_with_old_volume_kernel_body<GEOM>(
      i, ee, e_old, r_new, r_old, u_half, vol, vol_old, mass, P_half,
      Q_half, hydro_active, n_cells, dt, compatible_energy, eta_compatible,
      E_floor_injected, clamp_count);
}

template <int GEOM>
__global__ void energy_update_with_old_volume_2t_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ ee_old,
    const double* __restrict__ ei_old,
    const double* __restrict__ rho_half,
    const double* __restrict__ Te_half,
    const double* __restrict__ Ti_half,
    const double* __restrict__ r_new,
    const double* __restrict__ r_old,
    const double* __restrict__ u_half,
    const double* __restrict__ vol,
    const double* __restrict__ vol_old,
    const double* __restrict__ mass,
    const double* __restrict__ Pe_half,
    const double* __restrict__ Pi_half,
    const double* __restrict__ Q_half,
    const double* __restrict__ zbar,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const double fallback_z,
    const double qei_multiplier,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele,
    const double* __restrict__ cv_e_arr,
    const double* __restrict__ cv_i_arr,
    const int q_heat_to_electron,
    const int compatible_energy,
    double* __restrict__ eta_compatible,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  persistent_1d::energy_update_with_old_volume_2t_kernel_body<GEOM>(
      i, ee, ei, ee_old, ei_old, rho_half, Te_half, Ti_half, r_new, r_old,
      u_half, vol, vol_old, mass, Pe_half, Pi_half, Q_half, zbar,
      hydro_active, n_cells, dt, gamma_eff, A_eff, fallback_z, tab_ion,
      tab_ele, cv_e_arr, cv_i_arr, q_heat_to_electron, compatible_energy,
      eta_compatible, E_floor_injected, clamp_count);
}

template <int GEOM>
__global__ void compatible_energy_update_1d_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ v_old,
    const double* __restrict__ v_new,
    const double* __restrict__ pq_half,
    const double* __restrict__ Pe_half,
    const double* __restrict__ Pi_half,
    const double* __restrict__ Q_half,
    const double* __restrict__ node_r_half,
    const double* __restrict__ odd_even_pair_force_half,
    const double* __restrict__ p_extra_half,
    const double* __restrict__ mass,
    const std::uint8_t* __restrict__ cell_is_void,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt,
    const int use_two_temp,
    const int q_heat_to_electron,
    const double ghost_pq_half,
    const int free_outer_boundary,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count,
    double* __restrict__ residual_sums) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (cell_is_void != nullptr && cell_is_void[c] != 0u) {
    return;
  }

  const int jL = c;
  const int jR = c + 1;
  const double r_L = node_r_half[jL];
  const double r_R = node_r_half[jR];
  double A_L;
  if constexpr (GEOM == 0) {
    A_L = kFourPi * r_L * r_L;
  } else {
    A_L = geometry_1d_face_area(GEOM, r_L);
  }
  double A_R;
  if constexpr (GEOM == 0) {
    A_R = kFourPi * r_R * r_R;
  } else {
    A_R = geometry_1d_face_area(GEOM, r_R);
  }
  // gamma_r v3 split: the p_r component of the work belongs to the radiation
  // field (paid by the driver's work update), not to matter internal energy.
  const double p = (p_extra_half != nullptr) ? (pq_half[c] - p_extra_half[c])
                                             : pq_half[c];
  double p_R = p;
  if (free_outer_boundary != 0 && c == n_cells - 1) {
    p_R -= ghost_pq_half;
  }
  const double ubar_L = 0.5 * (v_old[jL] + v_new[jL]);
  const double ubar_R = 0.5 * (v_old[jR] + v_new[jR]);

  const double W_L = dt * (-A_L * p) * ubar_L;
  const double W_R = dt * (A_R * p_R) * ubar_R;
  const double dE_pq = -(W_L + W_R);
  double dE_oe = 0.0;
  if (odd_even_pair_force_half != nullptr) {
    const double F_oe_L = odd_even_pair_force_half[c];
    dE_oe = -dt * F_oe_L * (ubar_L - ubar_R);
  }
  const double dE = dE_pq + dE_oe;

  const double m = mass[c];
  if (m > 0.0) {
    if (use_two_temp != 0) {
      const double p_raw = fmax(fabs(p), 1.0e-30);
      double f_e = fmax(Pe_half[c], 0.0) / p_raw;
      double f_iQ = fmax(Pi_half[c] + Q_half[c], 0.0) / p_raw;
      const double f_sum = f_e + f_iQ;
      if (f_sum > 0.0 && isfinite(f_sum)) {
        f_e /= f_sum;
        f_iQ /= f_sum;
      } else {
        f_e = 0.5;
        f_iQ = 0.5;
      }
      double de_e = f_e * dE_pq / m;
      double de_i = f_iQ * dE_pq / m;
      if (q_heat_to_electron != 0) {
        de_e += dE_oe / m;
      } else {
        de_i += dE_oe / m;
      }
      const double ee_raw = ee[c] + de_e;
      const double ei_raw = ei[c] + de_i;
      const double ee_new = fmax(ee_raw, 0.0);
      const double ei_new = fmax(ei_raw, 0.0);
      ee[c] = ee_new;
      ei[c] = ei_new;
      if (ee_raw < 0.0) {
        if (E_floor_injected != nullptr) {
          atomic_add_double(E_floor_injected, m * (ee_new - ee_raw));
        }
        if (clamp_count != nullptr) {
          atomicAdd(clamp_count, 1);
        }
      }
      if (ei_raw < 0.0) {
        if (E_floor_injected != nullptr) {
          atomic_add_double(E_floor_injected, m * (ei_new - ei_raw));
        }
        if (clamp_count != nullptr) {
          atomicAdd(clamp_count, 1);
        }
      }
    } else {
      const double ee_raw = ee[c] + dE / m;
      const double ee_new = fmax(ee_raw, 0.0);
      ee[c] = ee_new;
      if (ee_raw < 0.0) {
        if (E_floor_injected != nullptr) {
          atomic_add_double(E_floor_injected, m * (ee_new - ee_raw));
        }
        if (clamp_count != nullptr) {
          atomicAdd(clamp_count, 1);
        }
      }
    }
  }

  if (residual_sums != nullptr) {
    atomic_add_double(&residual_sums[0], dE);
  }
}

__global__ void compatible_energy_kinetic_residual_1d_kernel(
    const double* __restrict__ mass,
    const double* __restrict__ v_old,
    const double* __restrict__ v_new,
    const std::uint8_t* __restrict__ node_active,
    const int n_begin,
    const int n_end,
    const int n_cells,
    double* __restrict__ residual_sums) {
  const int j = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_end) {
    return;
  }
  if (node_active != nullptr && node_active[j] == 0u) {
    return;
  }

  double node_mass = 0.0;
  if (j == 0) {
    node_mass = 0.5 * mass[0];
  } else if (j == n_cells) {
    node_mass = 0.5 * mass[n_cells - 1];
  } else {
    node_mass = 0.5 * (mass[j - 1] + mass[j]);
  }
  const double dK =
      0.5 * node_mass * (v_new[j] * v_new[j] - v_old[j] * v_old[j]);
  if (residual_sums != nullptr) {
    atomic_add_double(&residual_sums[1], dK);
  }
}

template <bool kZMom>
__global__ void apply_qei_transfer_2t_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const double* __restrict__ mass,
    const double* __restrict__ zbar,
    const double* __restrict__ cv_e,
    const double* __restrict__ cv_i,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const double fallback_z,
    const double qei_multiplier,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count,
    const double* __restrict__ zmom_r2) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[i] == 0) {
    return;
  }

  const double gamma = fmax(gamma_eff[i], 1.0 + 1.0e-12);
  const double A = fmax(A_eff[i], 1.0e-12);
  const double z = (zbar != nullptr) ? fmax(zbar[i], 0.0) : fallback_z;
  double qei_term;
  if constexpr (kZMom) {
    qei_term =
        (cv_e != nullptr && cv_i != nullptr)
            ? tenryu::materials::compute_qei_term_with_cv_ext(
                  fmax(rho[i], 0.0), fmax(Te[i], 0.0), fmax(Ti[i], 0.0), z,
                  zmom_r2[i], A, fmax(cv_e[i], 0.0), fmax(cv_i[i], 0.0), dt,
                  qei_multiplier)
            : tenryu::materials::compute_qei_term_analytical_ext(
                  fmax(rho[i], 0.0), fmax(Te[i], 0.0), fmax(Ti[i], 0.0), z,
                  zmom_r2[i], A, gamma, dt, qei_multiplier);
  } else {
    qei_term =
        (cv_e != nullptr && cv_i != nullptr)
            ? tenryu::materials::compute_qei_term_with_cv(
                  fmax(rho[i], 0.0), fmax(Te[i], 0.0), fmax(Ti[i], 0.0), z, A,
                  fmax(cv_e[i], 0.0), fmax(cv_i[i], 0.0), dt, qei_multiplier)
            : tenryu::materials::compute_qei_term_analytical(
                  fmax(rho[i], 0.0), fmax(Te[i], 0.0), fmax(Ti[i], 0.0), z, A,
                  gamma, dt, qei_multiplier);
  }
  // AI review k15 1.4/P0-4 (2026-07-26): bracket the transfer into the
  // admissible interval instead of clamping ee/ei independently — the old
  // floors created pair energy whenever one side hit zero (tallied into
  // E_floor_injected). One shared applied transfer keeps ee + ei exactly
  // conserved with both sides nonnegative, so this kernel no longer injects
  // floor energy at all; a clipped transfer is still counted. Cells where
  // the floors never engaged are bit-identical.
  const double qei_hi = fmax(ee[i], 0.0);   // most the electrons can give
  const double qei_lo = -fmax(ei[i], 0.0);  // most the ions can give
  const double qei_applied =
      isfinite(qei_term) ? fmin(fmax(qei_term, qei_lo), qei_hi) : 0.0;
  ee[i] -= qei_applied;
  ei[i] += qei_applied;
  if (qei_applied != qei_term && clamp_count != nullptr) {
    atomicAdd(clamp_count, 1);
  }
  (void)E_floor_injected;
  (void)mass;
}

__global__ void apply_artificial_heat_kernel(
    double* __restrict__ e,
    const double* __restrict__ heat_rate,
    const double* __restrict__ mass,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt,
    const int clamp_to_zero,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (!active) {
    return;
  }

  const double m = mass[i];
  if (m <= 0.0) {
    return;
  }

  const double e_raw = e[i] + dt * heat_rate[i] / m;
  if (clamp_to_zero == 0) {
    e[i] = e_raw;
    return;
  }

  const double e_new = fmax(e_raw, 0.0);
  e[i] = e_new;
  if (e_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (e_new - e_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
}

__device__ __forceinline__ bool hydro_cell_active_1d(
    const std::int8_t* __restrict__ hydro_active,
    const int i) {
  return hydro_active == nullptr || hydro_active[i] != 0;
}

__device__ __forceinline__ double clamp01_1d(const double x) {
  return fmin(1.0, fmax(0.0, x));
}

template <int GEOM>
__device__ __forceinline__ double compute_ion_artificial_heat_face_power_1d(
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ cs,
    const double* __restrict__ cv_i,
    const double* __restrict__ q2,
    const double* __restrict__ div_u,
    const double* __restrict__ node_r,
    const double* __restrict__ mass,
    const std::int8_t* __restrict__ hydro_active,
    const int left_cell,
    const int right_cell,
    const double dt,
    const double heat_c,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff) {
  if (!(heat_c > 0.0) || !(dt > 0.0) ||
      !hydro_cell_active_1d(hydro_active, left_cell) ||
      !hydro_cell_active_1d(hydro_active, right_cell)) {
    return 0.0;
  }

  const double dr_left = node_r[left_cell + 1] - node_r[left_cell];
  const double dr_right = node_r[right_cell + 1] - node_r[right_cell];
  const double dn_face = 0.5 * (dr_left + dr_right);
  if (!(dr_left > 0.0) || !(dr_right > 0.0) || !(dn_face > 0.0)) {
    return 0.0;
  }

  const double rho_face = 0.5 * (rho[left_cell] + rho[right_cell]);
  const double cs_left = fmax(cs[left_cell], 0.0);
  const double cs_right = fmax(cs[right_cell], 0.0);
  const double cs_face = 0.5 * (cs_left + cs_right);
  const double cs2_face = 0.5 * (cs_left * cs_left + cs_right * cs_right);
  const double cv_left =
      (cv_i != nullptr)
          ? fmax(cv_i[left_cell], 0.0)
          : fmax(ideal_gas_cv_i_mass(fmax(gamma_eff[left_cell], 1.0 + 1.0e-12),
                                     fmax(A_eff[left_cell], 1.0e-12)),
                 0.0);
  const double cv_right =
      (cv_i != nullptr)
          ? fmax(cv_i[right_cell], 0.0)
          : fmax(ideal_gas_cv_i_mass(fmax(gamma_eff[right_cell], 1.0 + 1.0e-12),
                                     fmax(A_eff[right_cell], 1.0e-12)),
                 0.0);
  const double cv_face = 0.5 * (cv_left + cv_right);
  if (!(rho_face > 0.0) || !(cs_face > 0.0) || !(cv_face > 0.0)) {
    return 0.0;
  }

  const double compression = fmax(fmax(0.0, -div_u[left_cell]),
                                  fmax(0.0, -div_u[right_cell]));
  const double q2_face = fmax(fmax(q2[left_cell], 0.0), fmax(q2[right_cell], 0.0));
  if (!(compression > 0.0) || !(q2_face > 0.0)) {
    return 0.0;
  }

  const double Mc = dn_face * compression / fmax(cs_face, kIonArtHeatEps);
  const double S_comp = clamp01_1d((Mc - 0.05) / 0.10);
  if (!(S_comp > 0.0)) {
    return 0.0;
  }

  const double dp_th = (Pe[right_cell] + Pi[right_cell]) -
                       (Pe[left_cell] + Pi[left_cell]);
  const double drho = rho[right_cell] - rho[left_cell];
  const double abs_dp = fabs(dp_th);
  const double S_contact =
      abs_dp / (abs_dp + cs2_face * fabs(drho) + kIonArtHeatPressureFloor);
  if (!(S_contact > 0.0)) {
    return 0.0;
  }

  const double r_face = node_r[right_cell];
  double area_face;
  if constexpr (GEOM == 0) {
    area_face = kFourPi * r_face * r_face;
  } else {
    area_face = geometry_1d_face_area(GEOM, r_face);
  }
  if (!(area_face > 0.0)) {
    return 0.0;
  }

  const double beta = q2_face / fmax(compression, kIonArtHeatEps);
  const double kappa_raw = heat_c * S_comp * S_contact * beta * cv_face;
  // k01 §4.4 / k03 F-14 (AI review 2026-07-26): the former slab-style cap
  // 0.25 rho_f cv_f dn^2 / dt ignores the actual face area and the smaller
  // neighbor heat capacity, so on graded/spherical meshes the explicit
  // exchange could exceed the light cell's stability limit. Bound the pair
  // conductance instead: dt (A kappa / dn)(1/C_L + 1/C_R) <= 0.25 with
  // C_k = m_k cv_k the cell heat capacities.
  const double C_left = fmax(mass[left_cell], 0.0) * cv_left;
  const double C_right = fmax(mass[right_cell], 0.0) * cv_right;
  if (!(C_left > 0.0) || !(C_right > 0.0)) {
    return 0.0;
  }
  const double C_pair = C_left * C_right / (C_left + C_right);
  const double kappa_cap = 0.25 * dn_face * C_pair / (area_face * dt);
  const double kappa = fmin(kappa_raw, kappa_cap);
  if (!(kappa > 0.0) || !isfinite(kappa)) {
    return 0.0;
  }

  return -area_face * kappa * (Ti[right_cell] - Ti[left_cell]) / dn_face;
}

template <int GEOM>
__global__ void compute_ion_artificial_heat_1d_kernel(
    double* __restrict__ heat_rate,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ cs,
    const double* __restrict__ cv_i,
    const double* __restrict__ q2,
    const double* __restrict__ div_u,
    const double* __restrict__ node_r,
    const double* __restrict__ mass,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt,
    const double heat_c,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  if (!hydro_cell_active_1d(hydro_active, i) || !(heat_c > 0.0)) {
    heat_rate[i] = 0.0;
    return;
  }

  const double left_power =
      (i > 0) ? compute_ion_artificial_heat_face_power_1d<GEOM>(
                    rho, Ti, Pe, Pi, cs, cv_i, q2, div_u, node_r, mass,
                    hydro_active,
                    i - 1, i, dt, heat_c, gamma_eff, A_eff)
              : 0.0;
  const double right_power =
      (i + 1 < n_cells) ? compute_ion_artificial_heat_face_power_1d<GEOM>(
                              rho, Ti, Pe, Pi, cs, cv_i, q2, div_u, node_r,
                              mass, hydro_active, i, i + 1, dt, heat_c,
                              gamma_eff, A_eff)
                        : 0.0;
  heat_rate[i] = left_power - right_power;
}

__device__ __forceinline__ bool electron_odd_even_cell_eligible_1d(
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int i,
    const int n_cells) {
  return i >= 0 && i < n_cells && hydro_cell_active_1d(hydro_active, i) &&
         (cell_is_void == nullptr || cell_is_void[i] == static_cast<std::uint8_t>(0));
}

__device__ __forceinline__ double electron_odd_even_weight_1d_device(
    const double* __restrict__ ee,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int i,
    const int n_cells) {
  if (i <= 0 || i >= n_cells - 1) {
    return 0.0;
  }
  if (!electron_odd_even_cell_eligible_1d(hydro_active, cell_is_void, i - 1, n_cells) ||
      !electron_odd_even_cell_eligible_1d(hydro_active, cell_is_void, i, n_cells) ||
      !electron_odd_even_cell_eligible_1d(hydro_active, cell_is_void, i + 1, n_cells)) {
    return 0.0;
  }

  const double de_left = ee[i] - ee[i - 1];
  const double de_right = ee[i + 1] - ee[i];
  if (!(de_left * de_right < 0.0)) {
    return 0.0;
  }

  const double d2 = ee[i + 1] - 2.0 * ee[i] + ee[i - 1];
  const double grad = 0.5 * fabs(ee[i + 1] - ee[i - 1]);
  if (!(fabs(d2) > kElectronOddEvenGradFrac * grad + kElectronOddEvenEps)) {
    return 0.0;
  }

  return fmin(fabs(de_left), fabs(de_right)) /
         fmax(fmax(fabs(de_left), fabs(de_right)), kElectronOddEvenEps);
}

__device__ inline void compute_electron_odd_even_flux_1d_kernel_body(
    const int j,
    double* __restrict__ flux_face,
    const double* __restrict__ ee,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ mass,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_cells,
    const double C_ee_oe) {
  flux_face[j] = 0.0;
  if (!(C_ee_oe > 0.0) || j <= 1 || j >= n_cells - 1) {
    return;
  }

  const int left = j - 1;
  const int right = j;
  if (!electron_odd_even_cell_eligible_1d(hydro_active, cell_is_void, left - 1, n_cells) ||
      !electron_odd_even_cell_eligible_1d(hydro_active, cell_is_void, left, n_cells) ||
      !electron_odd_even_cell_eligible_1d(hydro_active, cell_is_void, right, n_cells) ||
      !electron_odd_even_cell_eligible_1d(hydro_active, cell_is_void, right + 1, n_cells)) {
    return;
  }

  const double w_left =
      electron_odd_even_weight_1d_device(ee, hydro_active, cell_is_void, left, n_cells);
  const double w_right =
      electron_odd_even_weight_1d_device(ee, hydro_active, cell_is_void, right, n_cells);
  if (!(w_left > 0.0) || !(w_right > 0.0)) {
    return;
  }

  const double d2_left = ee[right] - 2.0 * ee[left] + ee[left - 1];
  const double d2_right = ee[right + 1] - 2.0 * ee[right] + ee[left];
  if (!(d2_left * d2_right < 0.0)) {
    return;
  }

  const double vol_left = fmax(vol[left], 0.0);
  const double vol_right = fmax(vol[right], 0.0);
  const double rho_left = fmax(rho[left], 0.0);
  const double rho_right = fmax(rho[right], 0.0);
  if (!(vol_left > 0.0) || !(vol_right > 0.0) ||
      !(rho_left > 0.0) || !(rho_right > 0.0)) {
    return;
  }

  const double vol_face =
      2.0 * vol_left * vol_right / fmax(vol_left + vol_right, kElectronOddEvenEps);
  const double rho_face = 0.5 * (rho_left + rho_right);
  const double exchange_mass = rho_face * vol_face;
  const double sensor = fmin(w_left, w_right);
  double flux = C_ee_oe * sensor * exchange_mass * (ee[right] - ee[left]);
  // k01 §4.1 (AI review 2026-07-26): donor/hull cap. With strong mass
  // grading rho_face * vol_face can exceed the light cell's own mass, so
  // the uncapped one-shot exchange could push its e_e past the neighbor
  // value (and below zero). Capping the transfer at the reduced-mass value
  // mu |de| keeps both cells inside the [e_L, e_R] hull (positivity and
  // monotonicity), and never binds for near-uniform masses with
  // C_ee_oe * sensor <= 1/2.
  const double m_left = fmax(mass[left], 0.0);
  const double m_right = fmax(mass[right], 0.0);
  if (!(m_left > 0.0) || !(m_right > 0.0)) {
    return;
  }
  const double mu_pair = m_left * m_right / (m_left + m_right);
  const double cap = mu_pair * fabs(ee[right] - ee[left]);
  if (fabs(flux) > cap) {
    flux = copysign(cap, flux);
  }
  flux_face[j] = flux;
}

__global__ void compute_electron_odd_even_flux_1d_kernel(
    double* __restrict__ flux_face,
    const double* __restrict__ ee,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ mass,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_begin,
    const int n_end,
    const int n_cells,
    const double C_ee_oe) {
  const int j = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_end) {
    return;
  }

  compute_electron_odd_even_flux_1d_kernel_body(
      j, flux_face, ee, rho, vol, mass, hydro_active, cell_is_void, n_cells,
      C_ee_oe);
}

__device__ inline void apply_electron_odd_even_flux_1d_kernel_body(
    const int i,
    double* __restrict__ ee,
    const double* __restrict__ flux_face,
    const double* __restrict__ mass,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int n_cells) {
  if (!electron_odd_even_cell_eligible_1d(hydro_active, cell_is_void, i, n_cells)) {
    return;
  }

  const double m = mass[i];
  if (!(m > 0.0)) {
    return;
  }

  ee[i] += (flux_face[i + 1] - flux_face[i]) / m;
}

__global__ void apply_electron_odd_even_flux_1d_kernel(
    double* __restrict__ ee,
    const double* __restrict__ flux_face,
    const double* __restrict__ mass,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int c_begin,
    const int c_end,
    const int n_cells) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }

  apply_electron_odd_even_flux_1d_kernel_body(
      i, ee, flux_face, mass, hydro_active, cell_is_void, n_cells);
}

__device__ __forceinline__ bool hk_velocity_damper_cell_eligible_1d(
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int c,
    const int n_cells) {
  return c >= 0 && c < n_cells && hydro_cell_active_1d(hydro_active, c) &&
         (cell_is_void == nullptr || cell_is_void[c] == static_cast<std::uint8_t>(0));
}

__device__ __forceinline__ bool hk_velocity_damper_same_material_1d(
    const int* __restrict__ dominant_material,
    const int left,
    const int right) {
  return dominant_material == nullptr || dominant_material[left] == dominant_material[right];
}

__device__ __forceinline__ double hk_velocity_damper_node_mass_1d(
    const double* __restrict__ mass,
    const int j,
    const int n_cells) {
  if (j == 0) {
    return 0.5 * mass[0];
  }
  if (j < n_cells) {
    return 0.5 * (mass[j - 1] + mass[j]);
  }
  return 0.5 * mass[n_cells - 1];
}

__host__ __device__ __forceinline__ double hk_velocity_damper_log_jump(
    const double left,
    const double right) {
  const double l = fmax(left, kHighKVelocityDamperEps);
  const double r = fmax(right, kHighKVelocityDamperEps);
  return fabs(log(r / l));
}

__device__ __forceinline__ double hk_velocity_damper_cell_tau_1d(
    const double* __restrict__ node_r,
    const double* __restrict__ sigma_R_max,
    const int c) {
  const double dr = fmax(node_r[c + 1] - node_r[c], 0.0);
  return fmax(sigma_R_max[c], 0.0) * dr;
}

__device__ inline void compute_hk_velocity_residual_1d_kernel_body(
    const int j,
    double* __restrict__ dv,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const std::uint8_t* __restrict__ node_active,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int* __restrict__ dominant_material,
    const int n_cells) {
  dv[j] = 0.0;
  if (j <= 0 || j >= n_cells || node_active[j] == 0u) {
    return;
  }

  const int left_cell = j - 1;
  const int right_cell = j;
  if (!hk_velocity_damper_cell_eligible_1d(hydro_active, cell_is_void, left_cell, n_cells) ||
      !hk_velocity_damper_cell_eligible_1d(hydro_active, cell_is_void, right_cell, n_cells) ||
      !hk_velocity_damper_same_material_1d(dominant_material, left_cell, right_cell)) {
    return;
  }

  const double r_j = node_r[j];
  const double x_l = node_r[j - 1] - r_j;
  const double x_r = node_r[j + 1] - r_j;
  if (!(x_l < 0.0) || !(x_r > 0.0)) {
    return;
  }

  const double v_l = node_u[j - 1];
  const double v_c = node_u[j];
  const double v_r = node_u[j + 1];
  const double sum_x = x_l + x_r;
  const double sum_xx = x_l * x_l + x_r * x_r;
  const double sum_v = v_l + v_c + v_r;
  const double sum_vx = v_l * x_l + v_r * x_r;
  const double det = 3.0 * sum_xx - sum_x * sum_x;
  if (!(fabs(det) > kHighKVelocityDamperEps) || !isfinite(det)) {
    return;
  }

  const double vbar = (sum_v * sum_xx - sum_vx * sum_x) / det;
  if (isfinite(vbar)) {
    dv[j] = v_c - vbar;
  }
}

__global__ void compute_hk_velocity_residual_1d_kernel(
    double* __restrict__ dv,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const std::uint8_t* __restrict__ node_active,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int* __restrict__ dominant_material,
    const int n_begin,
    const int n_end,
    const int n_cells) {
  const int j = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_end) {
    return;
  }

  compute_hk_velocity_residual_1d_kernel_body(
      j, dv, node_r, node_u, node_active, hydro_active, cell_is_void,
      dominant_material, n_cells);
}

__global__ void apply_hk_velocity_damper_impulse_1d_kernel(
    double* __restrict__ node_u,
    double* __restrict__ E_damp,
    const double* __restrict__ dv,
    const double* __restrict__ node_r,
    const double* __restrict__ mass,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ Qvisc,
    const double* __restrict__ sigma_R_max,
    const std::uint8_t* __restrict__ node_active,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_is_void,
    const int* __restrict__ dominant_material,
    const std::uint8_t* __restrict__ near_front,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const int color,
    const double C_damp,
    const double tau_min,
    const double grad_Te_max,
    const double grad_rho_max,
    int* __restrict__ active_pairs,
    int* __restrict__ front_blocked,
    int* __restrict__ tau_blocked,
    double* __restrict__ KE_removed) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end || (i & 1) != color) {
    return;
  }

  if (!(C_damp > 0.0) || i <= 0 || i >= n_cells - 1) {
    return;
  }
  if (node_active[i] == 0u || node_active[i + 1] == 0u) {
    return;
  }
  if (near_front != nullptr && near_front[i] != 0u) {
    if (front_blocked != nullptr) {
      atomicAdd(front_blocked, 1);
    }
    return;
  }

  if (!hk_velocity_damper_cell_eligible_1d(hydro_active, cell_is_void, i, n_cells)) {
    return;
  }
  const int mat_center = (dominant_material != nullptr) ? dominant_material[i] : 0;
  for (int c = i - 1; c <= i + 1; ++c) {
    if (!hk_velocity_damper_cell_eligible_1d(hydro_active, cell_is_void, c, n_cells)) {
      return;
    }
    if (dominant_material != nullptr && dominant_material[c] != mat_center) {
      return;
    }
    if (Qvisc[c] > 0.0) {
      if (front_blocked != nullptr) {
        atomicAdd(front_blocked, 1);
      }
      return;
    }
  }

  if (tau_min > 0.0) {
    if (sigma_R_max == nullptr) {
      if (tau_blocked != nullptr) {
        atomicAdd(tau_blocked, 1);
      }
      return;
    }
    const double tau_left = fmax(hk_velocity_damper_cell_tau_1d(node_r, sigma_R_max, i - 1),
                                 hk_velocity_damper_cell_tau_1d(node_r, sigma_R_max, i));
    const double tau_right = fmax(hk_velocity_damper_cell_tau_1d(node_r, sigma_R_max, i),
                                  hk_velocity_damper_cell_tau_1d(node_r, sigma_R_max, i + 1));
    if (!(fmin(tau_left, tau_right) >= tau_min)) {
      if (tau_blocked != nullptr) {
        atomicAdd(tau_blocked, 1);
      }
      return;
    }
  }

  for (int c = i - 1; c <= i; ++c) {
    const double dlnTe = hk_velocity_damper_log_jump(Te[c], Te[c + 1]);
    const double dlnrho = hk_velocity_damper_log_jump(rho[c], rho[c + 1]);
    if (!(dlnTe <= grad_Te_max) || !(dlnrho <= grad_rho_max)) {
      if (front_blocked != nullptr) {
        atomicAdd(front_blocked, 1);
      }
      return;
    }
  }

  const double qr = dv[i] - dv[i + 1];
  const double smooth =
      (node_u[i] - dv[i]) - (node_u[i + 1] - dv[i + 1]);
  const double qr2 = qr * qr;
  const double smooth2 = smooth * smooth;
  const double S_noise = qr2 / fmax(qr2 + smooth2, kHighKVelocityDamperEps);
  if (!(S_noise >= kHighKVelocityDamperNoiseMin)) {
    return;
  }

  const double m_left = hk_velocity_damper_node_mass_1d(mass, i, n_cells);
  const double m_right = hk_velocity_damper_node_mass_1d(mass, i + 1, n_cells);
  if (!(m_left > 0.0) || !(m_right > 0.0)) {
    return;
  }
  const double inv_mass_sum = 1.0 / m_left + 1.0 / m_right;
  const double mu = 1.0 / inv_mass_sum;
  const double u_left_old = node_u[i];
  const double u_right_old = node_u[i + 1];
  const double q = u_left_old - u_right_old;
  double impulse = C_damp * S_noise * mu * qr;
  if (!(impulse * q > 0.0) || !isfinite(impulse)) {
    return;
  }

  const double impulse_limit = fabs(q) / fmax(inv_mass_sum, kHighKVelocityDamperEps);
  if (fabs(impulse) > impulse_limit) {
    impulse = copysign(impulse_limit, impulse);
  }

  const double u_left_new = u_left_old - impulse / m_left;
  const double u_right_new = u_right_old + impulse / m_right;
  const double E_loss =
      0.5 * m_left * (u_left_old * u_left_old - u_left_new * u_left_new) +
      0.5 * m_right * (u_right_old * u_right_old - u_right_new * u_right_new);
  if (!(E_loss > 0.0) || !isfinite(E_loss)) {
    return;
  }

  node_u[i] = u_left_new;
  node_u[i + 1] = u_right_new;
  E_damp[i] += E_loss;
  if (active_pairs != nullptr) {
    atomicAdd(active_pairs, 1);
  }
  if (KE_removed != nullptr) {
    atomic_add_double(KE_removed, E_loss);
  }
}

__global__ void apply_hk_velocity_damper_heat_1d_kernel(
    double* __restrict__ e,
    const double* __restrict__ E_damp,
    const double* __restrict__ mass,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end || !hydro_cell_active_1d(hydro_active, i)) {
    return;
  }
  const double m = mass[i];
  if (!(m > 0.0)) {
    return;
  }
  const double E = fmax(E_damp[i], 0.0);
  if (E > 0.0) {
    e[i] += E / m;
  }
}

__device__ inline void reduce_active_max_kernel_body(
    const int i,
    const int /*tid*/,
    double* __restrict__ /*shared*/,
    double* __restrict__ max_value,
    const double* __restrict__ values,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells) {
  if (!hydro_cell_active_1d(hydro_active, i)) {
    return;
  }
  const double value = fmax(values[i], 0.0);
  if (value > 0.0) {
    atomic_max_double(max_value, value);
  }
}

__global__ void reduce_active_max_kernel(double* __restrict__ max_value,
                                         const double* __restrict__ values,
                                         const std::int8_t* __restrict__ hydro_active,
                                         const int c_begin,
                                         const int c_end,
                                         const int n_cells) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ double shared[];
  if (i >= c_end) {
    return;
  }

  reduce_active_max_kernel_body(i, tid, shared, max_value, values, hydro_active,
                                n_cells);
}

__global__ void update_shock_time_kernel(double* __restrict__ shock_time,
                                         const double* __restrict__ Qvisc,
                                         const std::int8_t* __restrict__ hydro_active,
                                         const int c_begin,
                                         const int c_end,
                                         const int n_cells,
                                         const double threshold,
                                         const double t_current) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }
  if (!hydro_cell_active_1d(hydro_active, i)) {
    return;
  }
  if (Qvisc[i] > threshold) {
    shock_time[i] = t_current;
  }
}

__device__ __forceinline__ double shock_history_weight_1d(
    const double t_current,
    const double shock_time,
    const double tau_decay) {
  if (!(tau_decay > 0.0)) {
    return 0.0;
  }
  const double age = fmax(t_current - shock_time, 0.0);
  return exp(-age / tau_decay);
}

template <int GEOM>
__device__ __forceinline__ double compute_post_shock_face_power_1d(
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ cs,
    const double* __restrict__ shock_time,
    const double* __restrict__ node_r,
    const std::int8_t* __restrict__ hydro_active,
    const int left_cell,
    const int right_cell,
    const double t_current,
    const double heat_c,
    const double decay_cells) {
  if (!hydro_cell_active_1d(hydro_active, left_cell) ||
      !hydro_cell_active_1d(hydro_active, right_cell)) {
    return 0.0;
  }

  const double dr_left = node_r[left_cell + 1] - node_r[left_cell];
  const double dr_right = node_r[right_cell + 1] - node_r[right_cell];
  const double dr_face = 0.5 * (dr_left + dr_right);
  if (!(dr_left > 0.0) || !(dr_right > 0.0) || !(dr_face > 0.0)) {
    return 0.0;
  }

  const double cs_left = fmax(cs[left_cell], 0.0);
  const double cs_right = fmax(cs[right_cell], 0.0);
  const double cs_face = 0.5 * (cs_left + cs_right);
  const double rho_face = 0.5 * (rho[left_cell] + rho[right_cell]);
  if (!(cs_face > 0.0) || !(rho_face > 0.0)) {
    return 0.0;
  }

  const double tau_left =
      decay_cells * dr_left / fmax(cs_left, kPostShockSensorEps);
  const double tau_right =
      decay_cells * dr_right / fmax(cs_right, kPostShockSensorEps);
  const double psi_left = shock_history_weight_1d(t_current, shock_time[left_cell], tau_left);
  const double psi_right =
      shock_history_weight_1d(t_current, shock_time[right_cell], tau_right);
  const double psi_face = 0.5 * (psi_left + psi_right);
  if (!(psi_face > 0.0)) {
    return 0.0;
  }

  const double e_left = ee[left_cell] + ((ei != nullptr) ? ei[left_cell] : 0.0);
  const double e_right = ee[right_cell] + ((ei != nullptr) ? ei[right_cell] : 0.0);
  const double grad_e = (e_right - e_left) / dr_face;
  const double flux = -heat_c * rho_face * cs_face * dr_face * psi_face * grad_e;
  const double r_face = node_r[right_cell];
  double area_face;
  if constexpr (GEOM == 0) {
    area_face = kFourPi * r_face * r_face;
  } else {
    area_face = geometry_1d_face_area(GEOM, r_face);
  }
  return area_face * flux;
}

template <int GEOM>
__global__ void apply_post_shock_heat_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const double* __restrict__ node_r,
    const double* __restrict__ mass,
    const double* __restrict__ Pe_split,
    const double* __restrict__ Pi_split,
    const double* __restrict__ shock_time,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt,
    const double t_current,
    const double heat_c,
    const double decay_cells,
    const int two_temperature,
    const int clamp_to_zero,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= c_end) {
    return;
  }
  if (!hydro_cell_active_1d(hydro_active, i)) {
    return;
  }

  const double m = mass[i];
  if (!(m > 0.0)) {
    return;
  }

  const double left_power =
      (i > 0)
          ? compute_post_shock_face_power_1d<GEOM>(
                rho, ee, ei, cs, shock_time, node_r, hydro_active, i - 1, i,
                t_current, heat_c, decay_cells)
          : 0.0;
  const double right_power =
      (i + 1 < n_cells)
          ? compute_post_shock_face_power_1d<GEOM>(
                rho, ee, ei, cs, shock_time, node_r, hydro_active, i, i + 1,
                t_current, heat_c, decay_cells)
          : 0.0;
  const double de_total = dt * (left_power - right_power) / m;
  if (de_total == 0.0) {
    return;
  }

  if (two_temperature == 0) {
    const double ee_raw = ee[i] + de_total;
    if (clamp_to_zero == 0) {
      ee[i] = ee_raw;
      return;
    }
    const double ee_new = fmax(ee_raw, 0.0);
    ee[i] = ee_new;
    if (ee_raw < 0.0) {
      if (E_floor_injected != nullptr) {
        atomic_add_double(E_floor_injected, m * (ee_new - ee_raw));
      }
      if (clamp_count != nullptr) {
        atomicAdd(clamp_count, 1);
      }
    }
    return;
  }

  double electron_frac = 0.5;
  const double p_total = Pe_split[i] + Pi_split[i];
  if (isfinite(p_total) && fabs(p_total) > 0.0 && isfinite(Pe_split[i])) {
    electron_frac = Pe_split[i] / p_total;
  }

  const double ee_raw = ee[i] + electron_frac * de_total;
  const double ei_raw = ei[i] + (1.0 - electron_frac) * de_total;
  if (clamp_to_zero == 0) {
    ee[i] = ee_raw;
    ei[i] = ei_raw;
    return;
  }

  const double ee_new = fmax(ee_raw, 0.0);
  const double ei_new = fmax(ei_raw, 0.0);
  ee[i] = ee_new;
  ei[i] = ei_new;
  if (ee_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (ee_new - ee_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
  if (ei_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (ei_new - ei_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
}

// NOH-DET #17: the renorm sums must be fixed-order. The former
// atomic_add_double accumulation made the 1T-renorm scale depend on the
// inter-block atomic arrival order (run-to-run bifurcation at
// rounding-boundary steps; VERIFICATION 3.2r).
struct ActiveMassOp {
  const double* mass = nullptr;
  const std::int8_t* active = nullptr;

  __host__ __device__ double operator()(const int i) const {
    const bool is_active = (active == nullptr) || (active[i] != 0);
    return is_active ? mass[i] : 0.0;
  }
};

struct ActiveInternalOp {
  const double* mass = nullptr;
  const double* ee = nullptr;
  const std::int8_t* active = nullptr;

  __host__ __device__ double operator()(const int i) const {
    const bool is_active = (active == nullptr) || (active[i] != 0);
    return is_active ? mass[i] * fmax(ee[i], 0.0) : 0.0;
  }
};

__global__ void scale_active_energy_kernel(double* __restrict__ ee,
                                           const std::int8_t* __restrict__ hydro_active,
                                           const int n_cells,
                                           const double scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  persistent_1d::scale_active_energy_kernel_body(i, ee, hydro_active, scale);
}

__global__ void shift_active_energy_kernel(double* __restrict__ ee,
                                           const std::int8_t* __restrict__ hydro_active,
                                           const int n_cells,
                                           const double de) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  persistent_1d::shift_active_energy_kernel_body(i, ee, hydro_active, de);
}

__global__ void add_first_active_residual_kernel(double* __restrict__ ee,
                                                 const double* __restrict__ mass,
                                                 const int first_active,
                                                 const double residual) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    persistent_1d::add_first_active_residual_kernel_body(
        ee, mass, first_active, residual);
  }
}

std::string format_scientific(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << value;
  return oss.str();
}

double copy_cell_value_from_device(const core::CellField1D& field,
                                   const int cell,
                                   const char* message) {
  TENRYU_ASSERT(cell >= 0 && static_cast<std::size_t>(cell) < field.size(),
                "copy_cell_value_from_device index out of range");
  double value = 0.0;
  cuda_check(cudaMemcpy(&value,
                        field.data() + cell,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             message);
  return value;
}

void log_exact_ideal_gas_step0_diagnostic(const core::State& state,
                                          const core::Config& cfg,
                                          const HydroTableViews& eos_views) {
  static bool logged = false;
  if (logged || state.step != 0 || !use_exact_ideal_gas_backend(eos_views) ||
      state.rho.empty()) {
    return;
  }
  logged = true;

  const int diag_cell = std::min(150, static_cast<int>(state.rho.size()) - 1);
  const auto& mat = cfg.materials.materials.front();
  const double rho = copy_cell_value_from_device(
      state.rho, diag_cell, "Hydro1D: copy exact_ideal_gas diag rho failed");
  const double z = copy_cell_value_from_device(
      state.zbar, diag_cell, "Hydro1D: copy exact_ideal_gas diag zbar failed");
  const double ee = copy_cell_value_from_device(
      state.ee, diag_cell, "Hydro1D: copy exact_ideal_gas diag ee failed");
  const double ei = copy_cell_value_from_device(
      state.ei, diag_cell, "Hydro1D: copy exact_ideal_gas diag ei failed");
  const double Te = copy_cell_value_from_device(
      state.Te, diag_cell, "Hydro1D: copy exact_ideal_gas diag Te failed");
  const double Ti = copy_cell_value_from_device(
      state.Ti, diag_cell, "Hydro1D: copy exact_ideal_gas diag Ti failed");
  const double Pe = copy_cell_value_from_device(
      state.Pe, diag_cell, "Hydro1D: copy exact_ideal_gas diag Pe failed");
  const double Pi = copy_cell_value_from_device(
      state.Pi, diag_cell, "Hydro1D: copy exact_ideal_gas diag Pi failed");
  const double cs = copy_cell_value_from_device(
      state.cs, diag_cell, "Hydro1D: copy exact_ideal_gas diag cs failed");

  const double gamma =
      (state.gamma_eff.size() == state.rho.size())
          ? copy_cell_value_from_device(
                state.gamma_eff, diag_cell,
                "Hydro1D: copy exact_ideal_gas diag gamma_eff failed")
          : mat.ideal_gas_gamma;
  const double A_cell =
      (state.A_eff.size() == state.rho.size())
          ? copy_cell_value_from_device(
                state.A_eff, diag_cell,
                "Hydro1D: copy exact_ideal_gas diag A_eff failed")
          : mat.A;
  const double rho_safe = fmax(rho, 1.0e-30);
  const double z_safe = fmax(z, 0.0);
  const double cv_i = ideal_gas_cv_i_mass(gamma, A_cell);
  const double cv_e = ideal_gas_cv_e_mass(gamma, A_cell, z_safe, mat.cv_e_override, rho_safe);

  double ee_native = 0.0;
  double ei_native = 0.0;
  double Pe_native = 0.0;
  double Pi_native = 0.0;
  double cs_native = 0.0;
  if (cfg.main.two_temperature) {
    const double Te_native = fmax(Te, cfg.numerics.floors.Te);
    const double Ti_native = fmax(Ti, cfg.numerics.floors.Ti);
    ee_native = cv_e * Te_native;
    ei_native = cv_i * Ti_native;
    Pe_native = (gamma - 1.0) * rho * ee_native;
    Pi_native = (gamma - 1.0) * rho * ei_native;
    cs_native = sqrt(fmax(gamma * (Pe_native + Pi_native) / rho_safe, 0.0));
  } else {
    const double T_native = fmax(Te, cfg.numerics.floors.Te);
    if (mat.eos_T_ref_eV > 0.0 && mat.cv_e_override > 0.0) {
      const double T_ref = mat.eos_T_ref_eV;
      const double T_ref3 = T_ref * T_ref * T_ref;
      const double alpha0 = mat.cv_e_override / (4.0 * T_ref3);
      const double T4 = T_native * T_native * T_native * T_native;
      ee_native = alpha0 * T4 / rho_safe;
    } else {
      ee_native = (cv_e + cv_i) * T_native;
    }
    ei_native = 0.0;
    Pe_native = (gamma - 1.0) * rho * ee_native;
    Pi_native = 0.0;
    cs_native = sqrt(fmax(gamma * (gamma - 1.0) * ee_native, 0.0));
  }

  core::log_info("[exact_ideal_gas_diag] step=0 cell=" + std::to_string(diag_cell) +
                 " rho=" + format_scientific(rho) +
                 " ee=" + format_scientific(ee) +
                 " ei=" + format_scientific(ei) +
                 " Te=" + format_scientific(Te) +
                 " Ti=" + format_scientific(Ti) +
                 " Pe=" + format_scientific(Pe) +
                 " Pi=" + format_scientific(Pi) +
                 " Ptotal=" + format_scientific(Pe + Pi) +
                 " cs=" + format_scientific(cs));
  core::log_info("[exact_ideal_gas_diag_native] step=0 cell=" + std::to_string(diag_cell) +
                 " rho=" + format_scientific(rho) +
                 " ee=" + format_scientific(ee_native) +
                 " ei=" + format_scientific(ei_native) +
                 " Te=" + format_scientific(Te) +
                 " Ti=" + format_scientific(Ti) +
                 " Pe=" + format_scientific(Pe_native) +
                 " Pi=" + format_scientific(Pi_native) +
                 " Ptotal=" + format_scientific(Pe_native + Pi_native) +
                 " cs=" + format_scientific(cs_native));
}

std::uint8_t* upload_uint8_array(const std::vector<std::uint8_t>& values,
                                 const char* name) {
  if (values.empty()) {
    return nullptr;
  }
  const std::string tag = std::string("hydro_1d:upload:") + name;
  auto* d_values = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      tag.c_str(), values.size() * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_values, values.data(),
                        values.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             (std::string("Hydro1D: cudaMemcpy ") + name + " failed").c_str());
  return d_values;
}

std::uint8_t* upload_cell_is_void(const std::vector<std::uint8_t>& cell_is_void,
                                  const char* name) {
  return upload_uint8_array(cell_is_void, name);
}

int* upload_int_array(const std::vector<int>& values, const char* name) {
  if (values.empty()) {
    return nullptr;
  }
  const std::string tag = std::string("hydro_1d:upload:") + name;
  auto* d_values = static_cast<int*>(core::device_scratch_acquire(
      tag.c_str(), values.size() * sizeof(int)));
  cuda_check(cudaMemcpy(d_values, values.data(), values.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
             (std::string("Hydro1D: cudaMemcpy ") + name + " failed").c_str());
  return d_values;
}

std::vector<int> compute_dominant_material_1d(const core::State& state,
                                              const core::Config& cfg,
                                              const int n_cells) {
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (n_cells <= 0 || n_mat <= 1) {
    return {};
  }
  const int first_nonvoid = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(first_nonvoid >= 0,
                "Hydro1D high-k velocity damper requires a non-void material");
  const std::size_t expected =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  TENRYU_ASSERT(state.volFrac.size() == expected,
                "Hydro1D high-k velocity damper requires volFrac size consistency");

  std::vector<double> volfrac(expected, 0.0);
  state.volFrac.copy_to_host(volfrac.data());
  std::vector<int> dominant(static_cast<std::size_t>(n_cells), first_nonvoid);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_mat);
    double best_frac = -1.0;
    int best_mat = first_nonvoid;
    for (int m = 0; m < n_mat; ++m) {
      const auto& mat = cfg.materials.materials[static_cast<std::size_t>(m)];
      if (mat.is_void) {
        continue;
      }
      const double frac_raw = volfrac[base + static_cast<std::size_t>(m)];
      const double frac =
          (std::isfinite(frac_raw) && frac_raw > 0.0) ? frac_raw : 0.0;
      if (frac > best_frac) {
        best_frac = frac;
        best_mat = m;
      }
    }
    dominant[static_cast<std::size_t>(c)] = best_mat;
  }
  return dominant;
}

std::vector<std::uint8_t> compute_hk_velocity_near_front_mask_1d(
    const core::State& state,
    core::ConstCellField1DView Qvisc,
    const int n_cells,
    const double grad_Te_max,
    const double grad_rho_max,
    const int guard_cells) {
  TENRYU_ASSERT(n_cells >= 0, "Hydro1D high-k velocity damper requires n_cells >= 0");
  TENRYU_ASSERT(state.rho.size() == static_cast<std::size_t>(n_cells),
                "Hydro1D high-k velocity damper requires rho size consistency");
  TENRYU_ASSERT(state.Te.size() == static_cast<std::size_t>(n_cells),
                "Hydro1D high-k velocity damper requires Te size consistency");
  TENRYU_ASSERT(Qvisc.size() == static_cast<std::size_t>(n_cells),
                "Hydro1D high-k velocity damper requires Qvisc size consistency");

  std::vector<std::uint8_t> near_front(static_cast<std::size_t>(n_cells), 0u);
  if (n_cells == 0) {
    return near_front;
  }

  std::vector<double> rho;
  std::vector<double> Te;
  std::vector<double> qvisc;
  state.rho.copy_to_host(rho);
  state.Te.copy_to_host(Te);
  qvisc.resize(Qvisc.size());
  cuda_check(cudaMemcpy(qvisc.data(), Qvisc.data(), Qvisc.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Hydro1D high-k velocity damper copy Qvisc failed");

  std::vector<std::uint8_t> is_front(static_cast<std::size_t>(n_cells), 0u);
  for (int c = 0; c < n_cells; ++c) {
    bool front = qvisc[static_cast<std::size_t>(c)] > 0.0;
    if (!front && static_cast<std::size_t>(c) < state.cell_is_void.size()) {
      front = state.cell_is_void[static_cast<std::size_t>(c)] != 0u;
    }
    if (!front && c > 0) {
      const double dlnTe = hk_velocity_damper_log_jump(
          Te[static_cast<std::size_t>(c - 1)], Te[static_cast<std::size_t>(c)]);
      const double dlnrho = hk_velocity_damper_log_jump(
          rho[static_cast<std::size_t>(c - 1)], rho[static_cast<std::size_t>(c)]);
      front = dlnTe > grad_Te_max || dlnrho > grad_rho_max;
    }
    if (!front && c + 1 < n_cells) {
      const double dlnTe = hk_velocity_damper_log_jump(
          Te[static_cast<std::size_t>(c)], Te[static_cast<std::size_t>(c + 1)]);
      const double dlnrho = hk_velocity_damper_log_jump(
          rho[static_cast<std::size_t>(c)], rho[static_cast<std::size_t>(c + 1)]);
      front = dlnTe > grad_Te_max || dlnrho > grad_rho_max;
    }
    is_front[static_cast<std::size_t>(c)] = front ? 1u : 0u;
  }

  const int guard = std::max(guard_cells, 0);
  for (int c = 0; c < n_cells; ++c) {
    if (is_front[static_cast<std::size_t>(c)] == 0u) {
      continue;
    }
    const int lo = std::max(0, c - guard);
    const int hi = std::min(n_cells - 1, c + guard);
    for (int j = lo; j <= hi; ++j) {
      near_front[static_cast<std::size_t>(j)] = 1u;
    }
  }

  return near_front;
}

__global__ void find_nonpositive_volume_1d_kernel(
    const double* __restrict__ vol,
    int* __restrict__ first_failing_cell,
    const int c_begin,
    const int c_end,
    const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  persistent_1d::find_nonpositive_volume_1d_kernel_body(
      c, vol, first_failing_cell, n_cells);
}

void refresh_geometry_and_density(core::State& state,
                                  const core::Config& cfg,
                                  const std::uint8_t* d_cell_is_void,
                                  int* rho_clamp_count = nullptr,
                                  const bool sync_host_geometry = false) {
  state.mesh.recompute_geometry_device_only(state.vol.data());

  TENRYU_ASSERT(state.mass.size() == state.vol.size(),
                "Hydro1D requires mass/volume size consistency");
  TENRYU_ASSERT(state.cell_is_void.size() == state.mass.size(),
                "Hydro1D requires cell_is_void/mass size consistency");
  const int n_cells = static_cast<int>(state.mass.size());
  TENRYU_ASSERT(d_cell_is_void != nullptr || n_cells == 0,
                "Hydro1D requires device cell_is_void for density refresh");
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  // Ghost-inclusive compute window (§6o.4, the 2D §6m prescription): ghost
  // rho must track the owner bitwise; the clamp counter stays owned.
  const core::State::LaunchWindow dw = state.owned_cell_window_ghost(n_cells, 1);
  compute_density_kernel<<<dw.blocks(), 256>>>(state.rho.data(), state.mass.data(),
                                          state.vol.data(), d_cell_is_void,
                                          dw.begin, dw.end, cw.begin, cw.end, n_cells,
                                          cfg.numerics.floors.rho,
                                          rho_clamp_count);
  sync_kernel("Hydro1D compute_density kernel failed");
  if (sync_host_geometry) {
    state.mesh.sync_device_geometry_to_host(state.vol.data());
  }
}

void enforce_1t_closure(
    core::State& state,
    const core::Config& cfg,
    const HydroTableViews& eos_views = HydroTableViews{},
    const bool preserve_table_energy = false,
    const bool energy_authoritative = false,
    int* inverse_clamp_count = nullptr,
    int* inverse_failure_count = nullptr) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "Hydro1D requires at least one material");
  const auto& mat = cfg.materials.materials.front();
  const int n_cells = static_cast<int>(state.rho.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  // Ghost-inclusive (§6o.4, 2D §6m ported): ghost closure outputs must
  // track the owner bitwise (table EOS re-eval != init values); counters
  // stay owned-windowed via cw.
  const core::State::LaunchWindow fw =
      state.owned_cell_window_ghost(n_cells, 1);
  const int exact_override_kind = select_exact_override_kind(cfg, eos_views);
  double* cv_e_ptr = state.cv_e.empty() ? nullptr : state.cv_e.data();
  double* cv_i_ptr = state.cv_i.empty() ? nullptr : state.cv_i.data();
  enforce_1t_closure_kernel<<<fw.blocks(), 256>>>(
      state.ee.data(), state.ei.data(), state.Te.data(), state.Ti.data(),
      state.Pe.data(), state.Pi.data(), state.rho.data(), state.zbar.data(),
      fw.begin, fw.end, cw.begin, cw.end, n_cells,
      state.gamma_eff.data(), state.A_eff.data(), mat.Z, mat.cv_e_override,
      cfg.numerics.floors.Te,
      eos_views.tab_total, eos_views.tab_total_rho_e, eos_views.spline_total,
      eos_views.jet_total,
      use_helmholtz_spline_backend(eos_views), use_helmholtz_jet_backend(eos_views),
      use_rho_e_table_backend(eos_views),
      use_exact_ideal_gas_backend(eos_views), cfg.numerics.hydro.eos_writeback,
      preserve_table_energy, energy_authoritative, inverse_clamp_count,
      inverse_failure_count,
      exact_override_kind, cv_e_ptr, cv_i_ptr);
  sync_kernel("Hydro1D enforce_1t_closure kernel failed");
  // cv override: replace table cv with exact ideal gas cv after closure
  if (exact_override_kind == kExactOverrideCv && cv_e_ptr != nullptr) {
    const double cv_ig = diagnostic_cv_i_mass(mat.A) + diagnostic_cv_e_mass(mat.A, mat.Z);
    thrust::fill(thrust::device, cv_e_ptr, cv_e_ptr + n_cells, cv_ig);
    if (cv_i_ptr != nullptr) {
      thrust::fill(thrust::device, cv_i_ptr, cv_i_ptr + n_cells, diagnostic_cv_i_mass(mat.A));
    }
  }
}

void enforce_2t_closure(
    core::State& state,
    const core::Config& cfg,
    const HydroTableViews& eos_views = HydroTableViews{},
    const bool preserve_table_energy = false,
    const bool energy_authoritative = false,
    int* inverse_clamp_count = nullptr,
    int* inverse_failure_count = nullptr) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "Hydro1D requires at least one material");
  const auto& mat = cfg.materials.materials.front();
  const int n_cells = static_cast<int>(state.rho.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  // Ghost-inclusive (§6o.4, 2D §6m ported): ghost closure outputs must
  // track the owner bitwise (table EOS re-eval != init values); counters
  // stay owned-windowed via cw.
  const core::State::LaunchWindow fw =
      state.owned_cell_window_ghost(n_cells, 1);
  const int exact_override_kind = select_exact_override_kind(cfg, eos_views);
  double* cv_e_ptr = state.cv_e.empty() ? nullptr : state.cv_e.data();
  double* cv_i_ptr = state.cv_i.empty() ? nullptr : state.cv_i.data();
  enforce_2t_closure_kernel<<<fw.blocks(), 256>>>(
      state.ee.data(), state.ei.data(), state.Te.data(), state.Ti.data(),
      state.Pe.data(), state.Pi.data(), state.rho.data(), state.zbar.data(),
      fw.begin, fw.end, cw.begin, cw.end, n_cells,
      state.gamma_eff.data(), state.A_eff.data(), mat.Z, mat.cv_e_override,
      cfg.numerics.floors.Te,
      cfg.numerics.floors.Ti,
      eos_views.tab_ion, eos_views.tab_ele, eos_views.tab_total_rho_e, eos_views.spline_total,
      eos_views.jet_total, eos_views.mie_gruneisen,
      use_helmholtz_spline_backend(eos_views), use_helmholtz_jet_backend(eos_views),
      use_rho_e_table_backend(eos_views), use_mie_gruneisen_backend(eos_views),
      use_exact_ideal_gas_backend(eos_views), cfg.numerics.hydro.eos_writeback,
      preserve_table_energy, energy_authoritative, inverse_clamp_count,
      inverse_failure_count,
      exact_override_kind, cv_e_ptr, cv_i_ptr);
  sync_kernel("Hydro1D enforce_2t_closure kernel failed");
  // cv override: replace table cv with exact ideal gas cv after 2T closure
  if (exact_override_kind == kExactOverrideCv) {
    if (cv_e_ptr != nullptr) {
      const double cv_e_ig = diagnostic_cv_e_mass(mat.A, mat.Z);
      thrust::fill(thrust::device, cv_e_ptr, cv_e_ptr + n_cells, cv_e_ig);
    }
    if (cv_i_ptr != nullptr) {
      const double cv_i_ig = diagnostic_cv_i_mass(mat.A);
      thrust::fill(thrust::device, cv_i_ptr, cv_i_ptr + n_cells, cv_i_ig);
    }
  }
}

void enforce_eos_closure(core::State& state,
                         const core::Config& cfg,
                         const bool use_two_temp,
                         const HydroTableViews& eos_views = HydroTableViews{},
                         const bool preserve_table_energy = false,
                         const bool report_inverse_reclosure = false) {
  // Single choke point: every closure consumer gets consistent per-cell
  // effective properties (BUG-1 fix — no more materials[0]-for-all-cells).
  state.ensure_cell_material_props(cfg);
  const bool energy_authoritative_closure =
      (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
  int* d_inverse_clamp_count = nullptr;
  int* d_inverse_failure_count = nullptr;
  if (report_inverse_reclosure) {
    int* d_inverse_pack = static_cast<int*>(core::device_scratch_acquire(
        "hydro_1d:enforce_eos_closure:inverse_pack", 2 * sizeof(int)));
    d_inverse_clamp_count = d_inverse_pack + 0;
    d_inverse_failure_count = d_inverse_pack + 1;
    cuda_check(cudaMemset(d_inverse_pack, 0, 2 * sizeof(int)),
               "Hydro1D: memset inverse pack failed");
  }

  if (use_two_temp) {
    enforce_2t_closure(state, cfg, eos_views, preserve_table_energy,
                       energy_authoritative_closure, d_inverse_clamp_count,
                       d_inverse_failure_count);
  } else {
    enforce_1t_closure(state, cfg, eos_views, preserve_table_energy,
                       energy_authoritative_closure, d_inverse_clamp_count,
                       d_inverse_failure_count);
  }

  if (report_inverse_reclosure) {
    int inverse_counts[2] = {0, 0};
    cuda_check(cudaMemcpy(inverse_counts, d_inverse_clamp_count, sizeof(inverse_counts),
                          cudaMemcpyDeviceToHost),
               "Hydro1D: copy inverse pack failed");
    const int inverse_clamps = inverse_counts[0];
    const int inverse_failures = inverse_counts[1];
    if (inverse_failures > 0) {
      core::log_warning("Hydro1D compatible TMAT inverse reclosure failure: "
                        "clamp=" + std::to_string(inverse_clamps) +
                        " failure=" + std::to_string(inverse_failures));
    } else if (inverse_clamps > 0) {
      core::log_debug("Hydro1D compatible TMAT inverse reclosure boundary-edge clamp: "
                      "clamp=" + std::to_string(inverse_clamps));
    }
  }
}

void compute_cell_sound_speed(core::CellField1DView cs,
                              const core::State& state,
                              const core::Config& cfg,
                              const bool use_two_temp,
                              const HydroTableViews& eos_views = HydroTableViews{}) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "Hydro1D requires at least one material");
  TENRYU_ASSERT(state.gamma_eff.size() == state.rho.size(),
                "compute_cell_sound_speed requires ensure_cell_material_props "
                "to have run (closure precedes sound speed)");
  TENRYU_ASSERT(cs.size() == state.rho.size(),
                "compute_cell_sound_speed requires cs size consistency");

  const int n_cells = static_cast<int>(state.rho.size());
  // Ghost-inclusive (§6o.4n root): the odd-even damping kernel evaluates
  // interface NODES on both ranks and reads cs at the neighbor-side cell,
  // so ghost cs must equal the owner's bitwise (inputs rho/Pe/Te are
  // ghost-fresh). Serial identity: pw == [0, n).
  const core::State::LaunchWindow cw = state.owned_cell_window_ghost(n_cells, 1);
  const int exact_override_kind = select_exact_override_kind(cfg, eos_views);
  const double* cv_e_ptr = state.cv_e.empty() ? nullptr : state.cv_e.data();
  const double* cv_i_ptr = state.cv_i.empty() ? nullptr : state.cv_i.data();
  if (use_two_temp) {
    compute_sound_speed_2t_kernel<<<cw.blocks(), 256>>>(
        cs.data(), state.rho.data(), state.ee.data(), state.ei.data(), state.Pe.data(),
        state.Pi.data(), state.Te.data(), state.Ti.data(), eos_views.tab_ion, eos_views.tab_ele,
        eos_views.tab_total_rho_e, eos_views.spline_total, eos_views.jet_total,
        eos_views.mie_gruneisen, cv_i_ptr,
        cv_e_ptr, cw.begin, cw.end, n_cells,
        use_helmholtz_spline_backend(eos_views),
        use_helmholtz_jet_backend(eos_views), use_rho_e_table_backend(eos_views),
        use_mie_gruneisen_backend(eos_views),
        use_exact_ideal_gas_backend(eos_views), exact_override_kind,
        state.gamma_eff.data());
  } else {
    compute_sound_speed_1t_kernel<<<cw.blocks(), 256>>>(
        cs.data(), state.ee.data(), state.Te.data(), state.rho.data(), state.Pe.data(),
        state.Pi.data(), eos_views.tab_total, eos_views.tab_total_rho_e, eos_views.spline_total,
        eos_views.jet_total, cv_e_ptr, cw.begin, cw.end, n_cells,
        use_helmholtz_spline_backend(eos_views), use_helmholtz_jet_backend(eos_views),
        use_rho_e_table_backend(eos_views),
        use_exact_ideal_gas_backend(eos_views), exact_override_kind,
        state.gamma_eff.data());
  }
  sync_kernel("Hydro1D compute_sound_speed kernel failed");
}

bool has_pressure_boundary(const core::Config& cfg) {
  return parse_boundary_type_1d(cfg) == HydroBoundaryType::PRESSURE;
}

void compute_odd_even_weight_1d_impl(core::CellField1DView W_oe,
                                     core::ConstCellField1DView rho,
                                     const std::int8_t* hydro_active,
                                     const int c_begin,
                                     const int c_end) {
  TENRYU_ASSERT(W_oe.size() == rho.size(),
                "compute_odd_even_weight_1d requires output and rho sizes to match");
  if (rho.empty()) {
    return;
  }

  const int blocks = ((c_end - c_begin) + 255) / 256;
  compute_odd_even_weight_1d_kernel<<<blocks, 256>>>(
      W_oe.data(), rho.data(), hydro_active, c_begin, c_end,
      static_cast<int>(rho.size()));
  sync_kernel("Hydro1D: compute_odd_even_weight_1d kernel failed");
}

void filter_pq_checkerboard_1d_impl(core::CellField1DView pq_filtered,
                                    core::ConstCellField1DView pq_raw,
                                    core::ConstCellField1DView rho,
                                    core::ConstCellField1DView Qvisc,
                                    const std::int8_t* hydro_active,
                                    const double beta_cb,
                                    const int c_begin,
                                    const int c_end) {
  TENRYU_ASSERT(pq_raw.size() == rho.size(),
                "filter_pq_checkerboard_1d requires pq_raw and rho sizes to match");
  TENRYU_ASSERT(pq_raw.size() == Qvisc.size(),
                "filter_pq_checkerboard_1d requires pq_raw and Qvisc sizes to match");
  TENRYU_ASSERT(pq_filtered.size() == pq_raw.size(),
                "filter_pq_checkerboard_1d requires output and pq_raw sizes to match");
  if (pq_raw.empty()) {
    return;
  }

  const int blocks = ((c_end - c_begin) + 255) / 256;
  if (beta_cb <= 0.0) {
    copy_array_kernel<<<blocks, 256>>>(pq_filtered.data() + c_begin,
                                       pq_raw.data() + c_begin,
                                       c_end - c_begin);
    sync_kernel("Hydro1D: copy pq_raw kernel failed");
    return;
  }

  filter_pq_checkerboard_1d_kernel<<<blocks, 256>>>(
      pq_filtered.data(), pq_raw.data(), rho.data(), Qvisc.data(), hydro_active,
      c_begin, c_end, static_cast<int>(pq_raw.size()), beta_cb);
  sync_kernel("Hydro1D: filter_pq_checkerboard_1d kernel failed");
}

}  // namespace

void filter_pq_checkerboard_1d(core::CellField1D& pq_filtered,
                               const core::CellField1D& pq_raw,
                               const core::CellField1D& rho,
                               const core::CellField1D& Qvisc) {
  pq_filtered.reset(pq_raw.size());
  filter_pq_checkerboard_1d_impl(pq_filtered, pq_raw, rho, Qvisc, nullptr,
                                 kCheckerboardFilterBeta, 0,
                                 static_cast<int>(pq_raw.size()));
}

void compute_odd_even_weight_1d(core::CellField1D& W_oe,
                                const core::CellField1D& rho) {
  W_oe.reset(rho.size());
  compute_odd_even_weight_1d_impl(W_oe, rho, nullptr, 0,
                                  static_cast<int>(rho.size()));
}

void run_compatible_energy_update_1d(
    core::State& state,
    const core::Config& cfg,
    const double dt,
    const double* v_old,
    const double* v_new,
    const double* pq_half,
    const double* Pe_half,
    const double* Pi_half,
    const double* Q_half,
    const double* node_r_half,
    const double* odd_even_pair_force_half,
    const double* p_extra_half,
    const int n_cells,
    const double ghost_pq_half,
    const int free_outer_boundary,
    double* E_floor_injected,
    int* clamp_count) {
  if (n_cells <= 0) {
    return;
  }

  std::uint8_t* d_cell_is_void =
      upload_cell_is_void(state.cell_is_void, "cell_is_void_compat");
  double* d_residual_sums = nullptr;
  d_residual_sums = static_cast<double*>(core::device_scratch_acquire(
      "hydro_1d:run_compatible_energy_update_1d:d_residual_sums",
      2 * sizeof(double)));
  cuda_check(cudaMemset(d_residual_sums, 0, 2 * sizeof(double)),
             "Hydro1D: cudaMemset compatible residual failed");

  const int n_nodes = n_cells + 1;
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  const int q_heat_to_electron =
      (cfg.numerics.hydro.av_heat_to == "electron") ? 1 : 0;
  const int geom_code = state.mesh.geometry_code;
  switch (geom_code) {
    case 1:
      compatible_energy_update_1d_kernel<1><<<cw.blocks(), 256>>>(
          state.ee.data(), state.ei.data(), v_old, v_new, pq_half, Pe_half, Pi_half,
          Q_half, node_r_half, odd_even_pair_force_half, p_extra_half, state.mass.data(),
          d_cell_is_void, cw.begin, cw.end, n_cells, dt,
          cfg.main.two_temperature ? 1 : 0, q_heat_to_electron,
          ghost_pq_half, free_outer_boundary,
          E_floor_injected, clamp_count, d_residual_sums);
      break;
    case 2:
      compatible_energy_update_1d_kernel<2><<<cw.blocks(), 256>>>(
          state.ee.data(), state.ei.data(), v_old, v_new, pq_half, Pe_half, Pi_half,
          Q_half, node_r_half, odd_even_pair_force_half, p_extra_half, state.mass.data(),
          d_cell_is_void, cw.begin, cw.end, n_cells, dt,
          cfg.main.two_temperature ? 1 : 0, q_heat_to_electron,
          ghost_pq_half, free_outer_boundary,
          E_floor_injected, clamp_count, d_residual_sums);
      break;
    default:
      compatible_energy_update_1d_kernel<0><<<cw.blocks(), 256>>>(
          state.ee.data(), state.ei.data(), v_old, v_new, pq_half, Pe_half, Pi_half,
          Q_half, node_r_half, odd_even_pair_force_half, p_extra_half, state.mass.data(),
          d_cell_is_void, cw.begin, cw.end, n_cells, dt,
          cfg.main.two_temperature ? 1 : 0, q_heat_to_electron,
          ghost_pq_half, free_outer_boundary,
          E_floor_injected, clamp_count, d_residual_sums);
      break;
  }
  sync_kernel("Hydro1D: compatible energy update kernel failed");

  compatible_energy_kinetic_residual_1d_kernel<<<nw.blocks(), 256>>>(
      state.mass.data(), v_old, v_new, nullptr, nw.begin, nw.end, n_cells,
      d_residual_sums);
  sync_kernel("Hydro1D: compatible energy kinetic residual kernel failed");

  double residual_sums[2] = {0.0, 0.0};
  cuda_check(cudaMemcpy(residual_sums, d_residual_sums, 2 * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Hydro1D: copy compatible residual failed");
  {
    int world_rank = 0;
    int world_size = 1;
    parallel::Partition::query_world(&world_rank, &world_size);
    if (world_size > 1) {
      const parallel::Reduction reduction(world_size);
      reduction.allreduce_sum(residual_sums, 2);
    }
  }

  const double residual = residual_sums[0] + residual_sums[1];
  const double denom = std::max(std::abs(residual_sums[0]), std::abs(residual_sums[1]));
  constexpr double kCompatibleEnergyResidualWarnRel = 1.0e-6;
  if (denom > 0.0 && std::abs(residual) > kCompatibleEnergyResidualWarnRel * denom) {
    core::log_warning("Hydro1D compatible_energy residual exceeded tolerance: R=" +
                      format_scientific(residual) +
                      " sum_dE=" + format_scientific(residual_sums[0]) +
                      " sum_dK=" + format_scientific(residual_sums[1]));
  }
}

template <int GEOM>
__global__ void follow_void_nodes_1d_kernel(double* __restrict__ x_r,
                                            double* __restrict__ v_r,
                                            double* __restrict__ mass,
                                            const int first_void_cell,
                                            const int n_cells,
                                            const double rho_void) {
  // Void cells carry no forces, so their interior nodes never move on their
  // own and a Lagrangian edge expanding into the void region inverts cells
  // (BUG-5). Re-space the void-interior nodes uniformly between the plasma
  // edge node (owned by the last real cell — never touched here) and the
  // outer boundary node (owned by the free-boundary physics). Void masses
  // are re-pinned to rho_void * V so the placeholder cells keep their floor
  // density; they are excluded from physics, so no conserved quantity of a
  // real cell is affected.
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int edge_node = first_void_cell;      // outer node of last real cell
  const int last_node = n_cells;              // outer boundary node
  const int n_span = last_node - edge_node;   // number of void cells
  if (n_span <= 1) {
    return;  // zero or one void cell: no interior nodes to manage
  }
  const double edge = x_r[edge_node];
  const double outer = x_r[last_node];
  const double width = outer - edge;
  if (!(width > 0.0)) {
    return;  // edge reached the boundary; free-boundary physics takes over
  }
  const int interior = edge_node + 1 + j;
  if (interior >= last_node) {
    return;
  }
  const double frac =
      static_cast<double>(j + 1) / static_cast<double>(n_span);
  x_r[interior] = edge + frac * width;
  v_r[interior] = 0.0;
  // Re-pin the void cell masses on both sides of this node lazily: each
  // thread owns cell (interior - 1); thread j == n_span - 2 also owns the
  // outermost void cell.
  const double r0 = edge + (static_cast<double>(j) / n_span) * width;
  const double r1 = x_r[interior];
  double vol0;
  if constexpr (GEOM == 0) {
    vol0 =
        (kFourPi / 3.0) * (r1 - r0) * (r1 * r1 + r1 * r0 + r0 * r0);
  } else {
    vol0 = geometry_1d_shell_volume(GEOM, r0, r1);
  }
  mass[interior - 1] = rho_void * fmax(vol0, 0.0);
  if (j == n_span - 2) {
    const double r2 = outer;
    double vol1;
    if constexpr (GEOM == 0) {
      vol1 =
          (kFourPi / 3.0) * (r2 - r1) * (r2 * r2 + r2 * r1 + r1 * r1);
    } else {
      vol1 = geometry_1d_shell_volume(GEOM, r1, r2);
    }
    mass[interior] = rho_void * fmax(vol1, 0.0);
  }
}

void follow_void_region_nodes_1d(core::State& state, const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0 || state.cell_is_void.size() != static_cast<std::size_t>(n_cells)) {
    return;
  }
  // Contiguous trailing void region only (the 1D exterior-vacuum layout).
  int first_void = n_cells;
  for (int c = n_cells - 1; c >= 0; --c) {
    if (state.cell_is_void[static_cast<std::size_t>(c)] != 0U) {
      first_void = c;
    } else {
      break;
    }
  }
  if (first_void >= n_cells) {
    return;  // no trailing void region
  }
  const int n_span = n_cells - first_void;
  if (n_span <= 1) {
    return;
  }
  const int blocks = (n_span + 255) / 256;
  const int geom_code = state.mesh.geometry_code;
  switch (geom_code) {
    case 1:
      follow_void_nodes_1d_kernel<1><<<blocks, 256>>>(
          state.x_r.data(), state.v_r.data(), state.mass.data(), first_void,
          n_cells, cfg.materials.void_config.rho);
      break;
    case 2:
      follow_void_nodes_1d_kernel<2><<<blocks, 256>>>(
          state.x_r.data(), state.v_r.data(), state.mass.data(), first_void,
          n_cells, cfg.materials.void_config.rho);
      break;
    default:
      follow_void_nodes_1d_kernel<0><<<blocks, 256>>>(
          state.x_r.data(), state.v_r.data(), state.mass.data(), first_void,
          n_cells, cfg.materials.void_config.rho);
      break;
  }
  sync_kernel("Hydro1D: follow_void_nodes kernel failed");
}

void Hydro1D::prepare_initial_sound_speed(core::State& state,
                                          const core::Config& cfg,
                                          const HydroEOSContext* eos_ctx) const {
  if (state.rho.empty()) {
    return;
  }

  const HydroTableViews eos_views = select_hydro_table_views(eos_ctx);
  ensure_table_cv_fields(state, eos_views,
                         cfg.numerics.hydro.compatible_energy ||
                             !cfg.main.two_temperature);
  validate_compatible_energy_eos_support(cfg, eos_views);

  const bool use_two_temp = cfg.main.two_temperature;
  const auto h1d_dump5 = [&](const char* phase) {
    const char* dbg = std::getenv("TENRYU_H1D_DEBUG");
    if (dbg == nullptr) {
      return;
    }
    int dbg_rank = 0;
    if (const char* re = std::getenv("OMPI_COMM_WORLD_RANK")) {
      dbg_rank = std::atoi(re);
    }
    const int nc = static_cast<int>(state.rho.size());
    const auto dump_one = [&](const char* tag, const double* dptr) {
      if (dptr == nullptr) {
        return;
      }
      std::vector<double> h(static_cast<std::size_t>(nc));
      cuda_check(cudaMemcpy(h.data(), dptr,
                            static_cast<std::size_t>(nc) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H1D debug dump5 D2H failed");
      const std::string path = std::string(dbg) + "_" + phase + tag + "_q" + std::to_string(h1d_debug_seq_next()) + "_r" +
                               std::to_string(dbg_rank) + ".bin";
      if (FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    };
    dump_one("ee", state.ee.data());
    dump_one("te", state.Te.data());
    dump_one("rho", state.rho.data());
  };
  h1d_dump5("i0");
  enforce_eos_closure(state, cfg, use_two_temp, eos_views);
  h1d_dump5("i1");

  core::CellField1D cs_n;
  cs_n.reset(state.rho.size());
  compute_cell_sound_speed(cs_n, state, cfg, use_two_temp, eos_views);
  state.cs = std::move(cs_n);
  log_exact_ideal_gas_step0_diagnostic(state, cfg, eos_views);
}

tenryu::coupling::HydroStepResult Hydro1D::lagrangian_step(
    core::State& state,
    const double dt,
    const core::Config& cfg,
    const double t_op,
    double* E_floor_injected,
    int* clamp_count,
    int* rho_clamp_count,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    cudaStream_t stream,
    const HydroEOSContext* eos_ctx,
    const std::vector<double>* sigma_R_max,
    const core::CellField1D* p_extra,
    core::CellField1D* p_extra_work) const {
  tenryu::coupling::HydroStepResult step_result{};
  TENRYU_ASSERT(dt > 0.0, "Hydro1D::lagrangian_step requires dt > 0");
  // W-K: optional isotropic extra pressure (radiation p_r) joins the
  // FORCE-side pq only; the energy-side P_half assembly is deliberately
  // untouched (the radiation field pays its own work via the gamma_r=4/3
  // volume update in the driver — verdict D2 double-count avoidance).
  const double* pq_extra_ptr =
      (p_extra != nullptr && !p_extra->empty()) ? p_extra->data() : nullptr;
  TENRYU_ASSERT(cfg.main.dimension == "1D_SPH" ||
                    cfg.main.dimension == "1D_CYL",
                "Hydro1D requires 1D_SPH or 1D_CYL geometry");
  TENRYU_ASSERT(state.mesh.dim == 1, "Hydro1D requires 1D mesh");
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "Hydro1D requires matching node arrays");
  TENRYU_ASSERT(state.x_r.size() == state.rho.size() + 1,
                "Hydro1D requires node count = cell count + 1");
  const int geom_code = state.mesh.geometry_code;
  const std::string& av_type = cfg.numerics.hydro.av_type;
  TENRYU_ASSERT(av_type == "vnr" || av_type == "riemann" ||
                    av_type == "riemann_compatible" || av_type == "csw",
                "Numerics.hydro.av_type must be \"vnr\", \"riemann\", "
                "\"riemann_compatible\", or \"csw\"");
  const bool use_vnr_av = av_type == "vnr";
  const bool use_riemann_av = av_type == "riemann";
  const bool use_riemann_compatible_av = av_type == "riemann_compatible";
  const bool use_csw_av = av_type == "csw";
  const bool use_two_temp = cfg.main.two_temperature;
  const std::string& av_heat_to = cfg.numerics.hydro.av_heat_to;
  TENRYU_ASSERT(av_heat_to == "ion" || av_heat_to == "electron",
                "Numerics.hydro.av_heat_to must be \"ion\" or \"electron\" in v1.0");
  const auto& adaptive_av_cfg = cfg.numerics.hydro.adaptive_av;
  const bool adaptive_av_enabled = adaptive_av_cfg.enabled && use_vnr_av;
  const bool adaptive_av_has_heat =
      adaptive_av_enabled && (adaptive_av_cfg.base.heat_C > 0.0 ||
                              adaptive_av_cfg.primary.heat_C > 0.0 ||
                              adaptive_av_cfg.rebound.heat_C > 0.0);
  const bool adaptive_av_has_cpsv =
      adaptive_av_enabled && (adaptive_av_cfg.base.Cpsv > 0.0 ||
                              adaptive_av_cfg.primary.Cpsv > 0.0 ||
                              adaptive_av_cfg.rebound.Cpsv > 0.0);
  const bool adaptive_av_has_bulk =
      adaptive_av_enabled && (adaptive_av_cfg.base.cbulk > 0.0 ||
                              adaptive_av_cfg.primary.cbulk > 0.0 ||
                              adaptive_av_cfg.rebound.cbulk > 0.0);
  const double av_heat_c =
      (use_riemann_av || use_riemann_compatible_av)
          ? 0.0
          : cfg.numerics.hydro.av_heat_C;
  const bool av_heat_enabled = av_heat_c > 0.0 || adaptive_av_has_heat;
  const double ion_art_heat_c = cfg.numerics.hydro.ion_art_heat_C;
  const bool ion_art_heat_enabled = ion_art_heat_c > 0.0;
  TENRYU_ASSERT(!ion_art_heat_enabled || use_two_temp,
                "Numerics.hydro.ion_art_heat_C > 0 requires 2T hydro");
  TENRYU_ASSERT(!ion_art_heat_enabled || use_vnr_av,
                "Numerics.hydro.ion_art_heat_C > 0 requires VNR AV");
  const double bulk_viscosity_c = cfg.numerics.hydro.bulk_viscosity_C;
  const bool bulk_viscosity_enabled = bulk_viscosity_c > 0.0 || adaptive_av_has_bulk;
  const bool post_shock_heat_enabled =
      cfg.numerics.hydro.post_shock_heat &&
      cfg.numerics.hydro.post_shock_heat_C > 0.0;
  const double post_shock_heat_c = cfg.numerics.hydro.post_shock_heat_C;
  const double post_shock_heat_decay = cfg.numerics.hydro.post_shock_heat_decay;
  const double odd_even_damping_c = cfg.numerics.hydro.odd_even_damping_C;
  const double post_shock_velocity_damping_c =
      cfg.numerics.hydro.post_shock_velocity_damping_C;
  const bool post_shock_velocity_damping_enabled =
      post_shock_velocity_damping_c > 0.0 || adaptive_av_has_cpsv;
  const bool shock_history_enabled =
      post_shock_heat_enabled || post_shock_velocity_damping_enabled;
  const bool nodal_velocity_damping_enabled =
      odd_even_damping_c > 0.0 || post_shock_velocity_damping_enabled;
  const double checkerboard_filter_beta =
      checkerboard_filter_beta_from_odd_even_c(odd_even_damping_c);
  const double ee_odd_even_c =
      use_two_temp ? cfg.numerics.hydro.ee_odd_even_C : 0.0;
  const bool electron_odd_even_enabled = ee_odd_even_c > 0.0;
  const double hk_velocity_damper_c = cfg.numerics.hydro.hk_velocity_damper_C;
  const bool hk_velocity_damper_enabled = hk_velocity_damper_c > 0.0;
  const int q_heat_to_electron = (av_heat_to == "electron") ? 1 : 0;
  // k01 P0-2/P0-3 (AI review 2026-07-26): opt-in midpoint_v2 integrator —
  // stage-advance the energies to t^{n+1/2} before the mid-step closure and
  // use the stage P/Q directly (no (P^n + P^{n+1/2})/2 quarter-step average)
  // in the corrector energy update. Default legacy_pc is bit-preserving.
  const bool midpoint_v2 =
      cfg.numerics.hydro.time_integrator == "midpoint_v2";
  double local_E_floor = 0.0;
  int local_clamp_count = 0;
  int local_rho_clamp_count = 0;

  if (state.rho.empty()) {
    return step_result;
  }

  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  const int blocks_cells = (n_cells + 255) / 256;
  if (shock_history_enabled && state.shock_time.size() != state.rho.size()) {
    state.shock_time.reset(state.rho.size());
    state.shock_time.fill(kShockTimeInactive);
  }
  if (adaptive_av_enabled && state.adaptive_av_gate.size() != state.rho.size()) {
    state.adaptive_av_gate.reset(state.rho.size());
    state.adaptive_av_gate.fill(0.0);
  }
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  const core::State::LaunchWindow pw = state.owned_cell_window_ghost(n_cells, 1);
  const double t_half = t_op + 0.5 * dt;

  const HydroTableViews eos_views = select_hydro_table_views(eos_ctx);
  const bool compatible_energy_requested = cfg.numerics.hydro.compatible_energy;
  ensure_table_cv_fields(state, eos_views,
                         compatible_energy_requested || !cfg.main.two_temperature);
  validate_compatible_energy_eos_support(cfg, eos_views);
  const bool compatible_energy_eos_supported =
      !hydro_has_table_eos_backend_data(eos_views) || eos_views.supports_rho_e_reclosure;
  const bool has_incompatible_velocity_damping_forces =
      cfg.numerics.hydro.post_shock_velocity_damping_C > 0.0 ||
      adaptive_av_has_cpsv;
  const bool run_compatible_energy =
      compatible_energy_requested && compatible_energy_eos_supported &&
      !has_incompatible_velocity_damping_forces;
  const int legacy_volume_compatible_energy = 0;
  if (compatible_energy_requested && state.step == 0) {
    if (has_incompatible_velocity_damping_forces) {
      core::log_warning(
          "compatible_energy disabled: post-shock velocity damping active");
    }
  }

  core::ScratchCellField1D hk_sigma_R_max;
  const double* hk_sigma_R_max_ptr = nullptr;
  bool hk_velocity_damper_active = hk_velocity_damper_enabled;
  if (hk_velocity_damper_active && cfg.numerics.hydro.hk_velocity_damper_tau_min > 0.0) {
    if (sigma_R_max != nullptr &&
        sigma_R_max->size() == static_cast<std::size_t>(n_cells)) {
      hk_sigma_R_max.reset("hydro1d_hk_sigma_R_max", n_cells);
      hk_sigma_R_max.copy_from_host(*sigma_R_max);
      hk_sigma_R_max_ptr = hk_sigma_R_max.data();
    } else {
      static bool warned_missing_sigma = false;
      if (!warned_missing_sigma) {
        core::log_warning(
            "Hydro1D high-k velocity damper disabled until sigma_R_max is available");
        warned_missing_sigma = true;
      }
      hk_velocity_damper_active = false;
    }
  }
  std::vector<int> hk_dominant_material_host =
      hk_velocity_damper_active ? compute_dominant_material_1d(state, cfg, n_cells)
                                : std::vector<int>{};
  int* d_hk_dominant_material =
      upload_int_array(hk_dominant_material_host, "hk_dominant_material");

  std::int8_t* d_hydro_active =
      const_cast<std::int8_t*>(state.hydro_active_device_ptr());
  std::uint8_t* d_cell_is_void =
      upload_cell_is_void(state.cell_is_void, "cell_is_void_step");
  const auto h1d_dump6 = [&](const char* phase) {
    const char* dbg = std::getenv("TENRYU_H1D_DEBUG");
    if (dbg == nullptr) {
      return;
    }
    int dbg_rank = 0;
    if (const char* re = std::getenv("OMPI_COMM_WORLD_RANK")) {
      dbg_rank = std::atoi(re);
    }
    const auto dump_one = [&](const char* tag, const double* dptr) {
      std::vector<double> h(static_cast<std::size_t>(n_cells));
      cuda_check(cudaMemcpy(h.data(), dptr,
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H1D debug dump6 D2H failed");
      const std::string path = std::string(dbg) + "_" + phase + tag + "_q" + std::to_string(h1d_debug_seq_next()) + "_r" +
                               std::to_string(dbg_rank) + ".bin";
      if (FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    };
    dump_one("ee", state.ee.data());
    dump_one("te", state.Te.data());
  };
  h1d_dump6("x0");
  if (part.n_ranks > 1 && bufs != nullptr) {
    // OPEN-GXII-COND/§6o.4: rho/Pe/Pi/Te/Ti/zbar joined so ghost cells
    // enter the step owner-bitwise; the ghost-inclusive closure/density
    // recompute keeps them owner-consistent within the step (the 2D §6m
    // prescription ported to 1D — table-EOS decks exposed the staleness).
    double* field_ptrs[13] = {nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr, nullptr, nullptr,
                              nullptr, nullptr, nullptr};
    int n_field_ptrs = 0;
    field_ptrs[n_field_ptrs++] = state.ee.data();
    field_ptrs[n_field_ptrs++] = state.ei.data();
    field_ptrs[n_field_ptrs++] = state.mass.data();
    field_ptrs[n_field_ptrs++] = state.Qvisc.data();
    field_ptrs[n_field_ptrs++] = state.rho.data();
    field_ptrs[n_field_ptrs++] = state.Pe.data();
    field_ptrs[n_field_ptrs++] = state.Pi.data();
    field_ptrs[n_field_ptrs++] = state.Te.data();
    field_ptrs[n_field_ptrs++] = state.Ti.data();
    field_ptrs[n_field_ptrs++] = state.zbar.data();
    if (adaptive_av_enabled && !state.adaptive_av_gate.empty()) {
      field_ptrs[n_field_ptrs++] = state.adaptive_av_gate.data();
    }
    if (shock_history_enabled && !state.shock_time.empty()) {
      field_ptrs[n_field_ptrs++] = state.shock_time.data();
    }
    if (hk_sigma_R_max_ptr != nullptr) {
      field_ptrs[n_field_ptrs++] = hk_sigma_R_max.data();
    }
    parallel::exchange_cell_fields(part, *bufs, field_ptrs, n_field_ptrs, n_cells,
                                   stream, 1);
    h1d_dump6("x1");
    if (d_hydro_active != nullptr) {
      std::int8_t* hydro_active_ptr = d_hydro_active;
      parallel::exchange_int8_fields(part, *bufs, &hydro_active_ptr, 1, n_cells,
                                     stream, 1);
      // Halo exchange mutated the device mirror in place; the D2H refresh below re-syncs the host copy, so the mirror version is intentionally NOT bumped.
      cuda_check(cudaMemcpy(state.hydro_active.data(), hydro_active_ptr,
                            state.hydro_active.size() * sizeof(std::int8_t),
                            cudaMemcpyDeviceToHost),
                 "Hydro1D: cudaMemcpy hydro_active D2H failed");
    }
  }
  // Option-C staged intra-step ghost refreshes (design doc
  // mpi_w2_hydro1d_conversion_spec.md): derived caches are exchanged after
  // they are produced so every downstream stencil reads exact <=2-deep
  // ghosts. No-ops at n_ranks==1.
  const auto exchange_cell_ghosts = [&](std::initializer_list<double*> fields) {
    if (part.n_ranks <= 1 || bufs == nullptr) {
      return;
    }
    double* ptrs[8];
    int n_ptrs = 0;
    for (double* ptr : fields) {
      if (ptr != nullptr) {
        ptrs[n_ptrs++] = ptr;
      }
    }
    if (n_ptrs > 0) {
      parallel::exchange_cell_fields(part, *bufs, ptrs, n_ptrs, n_cells, stream, 1);
    }
  };
  if (part.n_ranks > 1) {
    // v1 multi-rank limitations (documented in the design doc §6b.4): the
    // void-region node scan is a global serial dependency and the adaptive
    // AV leading-shock tracker reduces over the full line; both defer to
    // the W3 1D line-gather. Production gate decks use neither.
    const bool any_void =
        std::any_of(state.cell_is_void.begin(), state.cell_is_void.end(),
                    [](const std::uint8_t v) { return v != 0u; });
    TENRYU_ASSERT(!any_void,
                  "MPI v1: 1D void regions are not supported multi-rank yet "
                  "(global void-run scan; W3 line-gather)");
    TENRYU_ASSERT(!adaptive_av_enabled,
                  "MPI v1: adaptive AV is not supported multi-rank yet "
                  "(global leading-shock tracker; W3 line-gather)");
    TENRYU_ASSERT(!braginskii::params_from_config(cfg).enabled,
                  "MPI v1: Braginskii viscosity is not windowed multi-rank "
                  "yet (own conversion wave)");
  }
  const auto allreduce_sum_scalar = [&](const double value) -> double {
    if (part.n_ranks <= 1) {
      return value;
    }
    const parallel::Reduction reduction(part.n_ranks);
    return reduction.allreduce_sum(value);
  };
  auto reduce_active_max = [&](const double* values,
                               const char* reduce_message,
                               const char* copy_message) -> double {
    double* d_max_value = nullptr;
    const double zero = 0.0;
    d_max_value = static_cast<double*>(core::device_scratch_acquire(
        "hydro_1d:reduce_active_max:d_max_value", sizeof(double)));
    cuda_check(cudaMemcpy(d_max_value, &zero, sizeof(double), cudaMemcpyHostToDevice),
               "Hydro1D: init active max failed");
    reduce_active_max_kernel<<<cw.blocks(), 256>>>(d_max_value, values, d_hydro_active,
                                                   cw.begin, cw.end, n_cells);
    sync_kernel(reduce_message);
    double max_value = 0.0;
    cuda_check(cudaMemcpy(&max_value, d_max_value, sizeof(double), cudaMemcpyDeviceToHost),
               copy_message);
    if (part.n_ranks > 1) {
      const parallel::Reduction reduction(part.n_ranks);
      max_value = reduction.allreduce_max(max_value);
    }
    return max_value;
  };
  double* d_floors_pack = static_cast<double*>(core::device_scratch_acquire(
      "hydro_1d:lagrangian_step:floors_pack", 2 * sizeof(double)));
  double* d_hydro_floor = d_floors_pack + 0;
  int* d_hydro_clamp_count =
      reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(d_floors_pack) + 8);
  int* d_rho_clamp_count =
      reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(d_floors_pack) + 12);
  {
    const double h_floors_init[2] = {0.0, 0.0};
    cuda_check(cudaMemcpy(d_floors_pack, h_floors_init, sizeof(h_floors_init),
                          cudaMemcpyHostToDevice),
               "Hydro1D: init floors pack failed");
  }

  const long long probe_step = static_cast<long long>(state.step) + 1;
  const bool probe_on = hydro_step_hash_active(probe_step);
  refresh_geometry_and_density(state, cfg, d_cell_is_void, d_rho_clamp_count);
  // TODO(#C-05): still blocked. Boundary PdV is accumulated in coupling/driver.cpp
  // for diagnostics, but this renormalization path uses compute_energy_budget_1d()
  // (internal+kinetic only), so re-enabling it for pressure BC would erase physical
  // pressure-drive work from the hydro state update.
  const bool apply_1t_energy_renorm =
      !use_two_temp && !has_pressure_boundary(cfg) && !run_compatible_energy;
  // TENRYU_HYDRO_EBAL=1: log the matter-energy delta of the entry EOS
  // closure projection (velocities untouched by the closure, so the
  // E_total difference isolates the internal-energy rewrite).
  static const bool hydro_ebal_enabled = [] {
    const char* s = std::getenv("TENRYU_HYDRO_EBAL");
    return s != nullptr && std::atoi(s) != 0;
  }();
  const double ebal_U_pre =
      hydro_ebal_enabled ? diagnostics::compute_energy_budget_1d(state).E_total
                         : 0.0;
  const auto h1d_dump4 = [&](const char* phase) {
    const char* h1dbg4 = std::getenv("TENRYU_H1D_DEBUG");
    if (h1dbg4 == nullptr) {
      return;
    }
    int dbg_rank = 0;
    if (const char* re = std::getenv("OMPI_COMM_WORLD_RANK")) {
      dbg_rank = std::atoi(re);
    }
    const auto dump_one = [&](const char* tag, const double* dptr) {
      if (dptr == nullptr) {
        return;
      }
      std::vector<double> h(static_cast<std::size_t>(n_cells));
      cuda_check(cudaMemcpy(h.data(), dptr,
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H1D debug dump4 D2H failed");
      const std::string path = std::string(h1dbg4) + "_" + phase + tag +
                               "_q" + std::to_string(h1d_debug_seq_next()) + "_r" + std::to_string(dbg_rank) + ".bin";
      if (FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    };
    dump_one("ee", state.ee.data());
    dump_one("te", state.Te.data());
    dump_one("rho", state.rho.data());
    dump_one("pe", state.Pe.data());
  };
  h1d_dump4("e0");
  enforce_eos_closure(state, cfg, use_two_temp, eos_views);
  if (hydro_ebal_enabled) {
    const double ebal_U_post =
        diagnostics::compute_energy_budget_1d(state).E_total;
    std::ostringstream ebal_os;
    ebal_os << std::scientific << std::setprecision(9)
            << "[hydro_ebal] t=" << t_op
            << " entry_eos_dU=" << (ebal_U_post - ebal_U_pre)
            << " U_pre=" << ebal_U_pre << " U_post=" << ebal_U_post;
    core::log_info(ebal_os.str());
  }
  h1d_dump4("e1");
  const double E_before = apply_1t_energy_renorm
                              ? allreduce_sum_scalar(
                                    diagnostics::compute_energy_budget_1d(state).E_total)
                              : 0.0;

  const bool av_eos_aware_active =
      use_vnr_av && cfg.numerics.hydro.av_eos_aware &&
      (eos_views.tab_total.n_rho > 0 || eos_views.tab_ion.n_rho > 0 ||
       eos_views.tab_ele.n_rho > 0 || eos_views.spline_total.n_rho > 0 ||
       eos_views.jet_total.n_rho > 0);
  const double av_c1_scalar =
      adaptive_av_enabled ? adaptive_av_cfg.base.c1
                          : (use_csw_av ? cfg.numerics.hydro.csw_C1
                                        : cfg.numerics.hydro.av_linear);
  const double av_c2_scalar =
      adaptive_av_enabled ? adaptive_av_cfg.base.c2
                          : (use_csw_av ? cfg.numerics.hydro.csw_C2
                                        : cfg.numerics.hydro.av_quadratic);
  const double av_heat_scalar =
      adaptive_av_enabled ? adaptive_av_cfg.base.heat_C : av_heat_c;
  const double bulk_viscosity_scalar =
      adaptive_av_enabled ? adaptive_av_cfg.base.cbulk : bulk_viscosity_c;
  const CswLimiter csw_limiter =
      (cfg.numerics.hydro.csw_limiter == "bj") ? CswLimiter::BarthJespersen
                                               : CswLimiter::VanLeer;
  const ArtificialViscosityType av_kind =
      use_riemann_av ? ArtificialViscosityType::Riemann
                     : (use_riemann_compatible_av
                            ? ArtificialViscosityType::RiemannCompatible
                            : (use_csw_av ? ArtificialViscosityType::Csw
                                          : ArtificialViscosityType::Vnr));
  ArtificialViscosity av(av_c1_scalar, av_c2_scalar,
                         cfg.numerics.hydro.av_limiter_J, av_heat_scalar,
                         av_eos_aware_active, cfg.numerics.hydro.av_eos_gamma1_ref,
                         cfg.numerics.hydro.av_eos_boost_max, av_kind,
                         bulk_viscosity_scalar, csw_limiter,
                         cfg.numerics.hydro.csw_shock_limiter_floor,
                         cfg.numerics.hydro.csw_zero_uniform_compression,
                         cfg.numerics.hydro.csw_diagnostics);
  // reset() zero-fills; keep that contract without the per-step realloc (perf lane B).
  if (cs_pingpong_.size() != n_cells) {
    cs_pingpong_.reset(n_cells);
  } else {
    cuda_check(cudaMemset(cs_pingpong_.data(), 0,
                          static_cast<std::size_t>(n_cells) * sizeof(double)),
               "Hydro1D: cs_pingpong_ zero reset failed");
  }
  compute_cell_sound_speed(cs_pingpong_, state, cfg, use_two_temp, eos_views);
  exchange_cell_ghosts({state.rho.data(), state.vol.data(), state.Te.data(),
                        state.Ti.data(), state.Pe.data(), state.Pi.data(),
                        cs_pingpong_.data()});
  if (probe_on) {
    hydro_step_probe(probe_step, "P1_eos",
                     {{"pe", {state.Pe.data(), state.Pe.size()}},
                      {"pi", {state.Pi.data(), state.Pi.size()}},
                      {"cs", {cs_pingpong_.data(), cs_pingpong_.size()}}});
  }
  AdaptiveAVFields adaptive_av_fields;
  core::ConstCellField1DView adaptive_c1;
  core::ConstCellField1DView adaptive_c2;
  core::ConstCellField1DView adaptive_heat_C;
  const core::CellField1D* adaptive_Cpsv = nullptr;
  core::ConstCellField1DView adaptive_cbulk;
  if (adaptive_av_enabled) {
    if (q_probe_persist_.size() != n_cells) {
      q_probe_persist_.reset(n_cells);
    } else {
      cuda_check(cudaMemset(q_probe_persist_.data(), 0,
                            static_cast<std::size_t>(n_cells) * sizeof(double)),
                 "Hydro1D: q_probe_persist_ zero reset failed");
    }
    if (div_u_probe_persist_.size() != n_cells) {
      div_u_probe_persist_.reset(n_cells);
    } else {
      cuda_check(
          cudaMemset(div_u_probe_persist_.data(), 0,
                     static_cast<std::size_t>(n_cells) * sizeof(double)),
          "Hydro1D: div_u_probe_persist_ zero reset failed");
    }
    av.compute_Q_1d(q_probe_persist_, state.rho, state.vol, state.x_r, state.v_r,
                    state.Pe, state.Pi, cs_pingpong_,
                    state.hydro_active_device_ptr(), geom_code, {}, {},
                    div_u_probe_persist_);
    const LeadingShockState shock = update_leading_shock_tracker_1d(
        state, cfg, q_probe_persist_, div_u_probe_persist_, t_op, dt);
    build_adaptive_av_fields_1d(state, cfg, shock, dt, adaptive_av_fields);
    adaptive_c1 = adaptive_av_fields.c1;
    adaptive_c2 = adaptive_av_fields.c2;
    adaptive_heat_C = adaptive_av_fields.heat_C;
    adaptive_Cpsv = &adaptive_av_fields.Cpsv;
    adaptive_cbulk = adaptive_av_fields.cbulk;
  }
  av.compute_Q_1d(state.Qvisc, state.rho, state.vol, state.x_r, state.v_r, state.Pe,
                  state.Pi, cs_pingpong_, state.hydro_active_device_ptr(), geom_code,
                  {}, {}, {}, adaptive_c1, adaptive_c2);
  if (bulk_viscosity_enabled) {
    av.add_bulk_viscosity_1d(state.Qvisc, state.rho, state.vol, state.x_r, state.v_r,
                             cs_pingpong_, state.hydro_active_device_ptr(),
                             adaptive_cbulk);
  }
  if (probe_on) {
    hydro_step_probe(probe_step, "P2_qn",
                     {{"q", {state.Qvisc.data(), state.Qvisc.size()}}});
  }
  std::swap(state.cs, cs_pingpong_);
  log_exact_ideal_gas_step0_diagnostic(state, cfg, eos_views);

  core::ScratchNodeField1D r_old;
  core::ScratchNodeField1D u_old;
  core::ScratchCellField1D V_old;
  core::ScratchCellField1D e_old;
  core::ScratchCellField1D ei_old;
  core::ScratchCellField1D Pe_old;
  core::ScratchCellField1D Pi_old;
  core::ScratchCellField1D Q_old;
  r_old.reset("hydro1d_r_old", state.x_r.size());
  u_old.reset("hydro1d_u_old", state.v_r.size());
  V_old.reset("hydro1d_V_old", state.vol.size());
  e_old.reset("hydro1d_e_old", state.ee.size());
  Pe_old.reset("hydro1d_Pe_old", n_cells);
  Pi_old.reset("hydro1d_Pi_old", n_cells);
  Q_old.reset("hydro1d_Q_old", n_cells);
  if (use_two_temp) {
    ei_old.reset("hydro1d_ei_old", state.ei.size());
  }

  cuda_check(cudaMemcpy(r_old.data(), state.x_r.data(), n_nodes * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro1D: copy r_old failed");
  cuda_check(cudaMemcpy(u_old.data(), state.v_r.data(), n_nodes * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro1D: copy u_old failed");
  cuda_check(cudaMemcpy(V_old.data(), state.vol.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro1D: copy V_old failed");
  cuda_check(cudaMemcpy(e_old.data(), state.ee.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro1D: copy e_old failed");
  cuda_check(cudaMemcpy(Pe_old.data(), state.Pe.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro1D: copy Pe_old failed");
  cuda_check(cudaMemcpy(Pi_old.data(), state.Pi.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro1D: copy Pi_old failed");
  cuda_check(cudaMemcpy(Q_old.data(), state.Qvisc.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro1D: copy Q_old failed");
  if (use_two_temp) {
    cuda_check(cudaMemcpy(ei_old.data(), state.ei.data(), n_cells * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro1D: copy ei_old failed");
  }

  std::uint8_t* d_node_active = nullptr;
  d_node_active = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "hydro_1d:lagrangian_step:d_node_active",
      n_nodes * sizeof(std::uint8_t)));
  cuda_check(cudaMemset(d_node_active, 0, n_nodes * sizeof(std::uint8_t)),
             "Hydro1D: cudaMemset node_active failed");
  compute_node_activity_kernel<<<cw.blocks(), 256>>>(d_node_active, d_hydro_active,
                                                      cw.begin, cw.end, n_cells);
  zero_center_node_kernel<<<1, 1>>>(d_node_active, n_nodes);
  sync_kernel("Hydro1D: compute_node_activity kernel failed");

  core::ScratchCellField1D pq_n;
  pq_n.reset("hydro1d_pq_n", n_cells);
  if (pq_extra_ptr != nullptr) {
    build_cell_pq_kernel<true><<<pw.blocks(), 256>>>(
        pq_n.data(), state.Pe.data(), state.Pi.data(), state.Qvisc.data(),
        pq_extra_ptr, d_hydro_active, pw.begin, pw.end, n_cells);
  } else {
    build_cell_pq_kernel<false><<<pw.blocks(), 256>>>(
        pq_n.data(), state.Pe.data(), state.Pi.data(), state.Qvisc.data(),
        nullptr, d_hydro_active, pw.begin, pw.end, n_cells);
  }
  sync_kernel("Hydro1D: build_cell_pq n kernel failed");

  core::ScratchCellField1D pq_n_filtered;
  const double* pq_n_for_accel = pq_n.data();
  if (checkerboard_filter_beta > 0.0) {
    // Filter the cell-centered pq source before acceleration so the same
    // checkerboard mode is not regenerated every hydro substep.
    pq_n_filtered.reset("hydro1d_pq_n_filtered", n_cells);
    filter_pq_checkerboard_1d_impl(pq_n_filtered, pq_n, state.rho, state.Qvisc,
                                   d_hydro_active, checkerboard_filter_beta,
                                   cw.begin, cw.end);
    exchange_cell_ghosts({pq_n_filtered.data()});
    pq_n_for_accel = pq_n_filtered.data();
  }
  if (probe_on) {
    hydro_step_probe(probe_step, "P3_pqn",
                     {{"pq", {pq_n_for_accel, static_cast<std::size_t>(n_cells)}}});
  }

  core::ScratchNodeField1D a_n;
  a_n.reset("hydro1d_a_n", n_nodes);
  const auto bc = parse_boundary_type_1d(cfg);
  const double boundary_pressure_n = pressure_ghost_1d(state, cfg, bc, t_op);
  switch (geom_code) {
    case 1:
      compute_acceleration_1d_kernel<1><<<nw.blocks(), 256>>>(
          a_n.data(), state.mass.data(), state.x_r.data(), pq_n_for_accel, d_node_active,
          nw.begin, nw.end, n_cells, state.Pe.data(), state.Pi.data(),
          state.Qvisc.data(), boundary_pressure_n, static_cast<int>(bc));
      break;
    case 2:
      compute_acceleration_1d_kernel<2><<<nw.blocks(), 256>>>(
          a_n.data(), state.mass.data(), state.x_r.data(), pq_n_for_accel, d_node_active,
          nw.begin, nw.end, n_cells, state.Pe.data(), state.Pi.data(),
          state.Qvisc.data(), boundary_pressure_n, static_cast<int>(bc));
      break;
    default:
      compute_acceleration_1d_kernel<0><<<nw.blocks(), 256>>>(
          a_n.data(), state.mass.data(), state.x_r.data(), pq_n_for_accel, d_node_active,
          nw.begin, nw.end, n_cells, state.Pe.data(), state.Pi.data(),
          state.Qvisc.data(), boundary_pressure_n, static_cast<int>(bc));
      break;
  }
  sync_kernel("Hydro1D: compute_acceleration n kernel failed");
  if (probe_on) {
    hydro_step_probe(probe_step, "P4_an",
                     {{"a", {a_n.data(), a_n.size()}}});
  }
  // W-H Braginskii ion viscosity (config-gated, default-inert, env diagnostic
  // overrides): trace-free stress force added to the pressure/AV acceleration
  // at both stages, before the damping filters, so it is time-centered like
  // the pressure.
  const braginskii::Params brag_params = braginskii::params_from_config(cfg);
  const int brag_skip_outer =
      (bc == HydroBoundaryType::FIXED || bc == HydroBoundaryType::REFLECT)
          ? 1
          : 0;
  core::ScratchCellField1D brag_pi_rr;
  core::ScratchCellField1D brag_pi_tt;
  core::ScratchCellField1D H_brag_half;
  core::ScratchCellField1D H_brag_e_half;
  if (brag_params.enabled) {
    brag_pi_rr.reset("hydro1d_brag_pi_rr", n_cells);
    brag_pi_tt.reset("hydro1d_brag_pi_tt", n_cells);
    braginskii::compute_cell_stress_1d(
        brag_pi_rr.data(), brag_pi_tt.data(), nullptr, nullptr,
        state.x_r.data(), state.v_r.data(), state.rho.data(),
        state.Ti.empty() ? nullptr : state.Ti.data(),
        state.Te.empty() ? nullptr : state.Te.data(),
        state.A_eff.empty() ? nullptr : state.A_eff.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(), state.vol.data(),
        d_hydro_active, n_cells, geom_code, brag_params,
        state.zmom_active ? state.zmom_r4.data() : nullptr);
    braginskii::add_node_accel_1d(
        a_n.data(), state.mass.data(), state.x_r.data(), state.vol.data(),
        brag_pi_rr.data(), brag_pi_tt.data(), d_node_active, n_cells,
        geom_code, brag_skip_outer, brag_params);
  }
  double qvisc_max_n = 0.0;
  if (post_shock_velocity_damping_enabled) {
    qvisc_max_n = reduce_active_max(state.Qvisc.data(),
                                    "Hydro1D: reduce qvisc_max n kernel failed",
                                    "Hydro1D: copy qvisc_max n failed");
  }
  if (nodal_velocity_damping_enabled) {
    core::ScratchCellField1D W_oe_n;
    W_oe_n.reset("hydro1d_W_oe_n", n_cells);
    compute_odd_even_weight_1d_impl(W_oe_n, state.rho, d_hydro_active,
                                    cw.begin, cw.end);
    exchange_cell_ghosts({W_oe_n.data(), state.Qvisc.data()});
    switch (geom_code) {
      case 1:
        apply_odd_even_damping_1d_kernel<1><<<nw.blocks(), 256>>>(
            a_n.data(), nullptr, nullptr, W_oe_n.data(), state.rho.data(), state.cs.data(),
            state.x_r.data(), state.v_r.data(), state.mass.data(),
            state.shock_time.empty() ? nullptr : state.shock_time.data(),
            state.Qvisc.data(), d_node_active, nw.begin, nw.end, n_cells,
            t_op, post_shock_heat_decay,
            odd_even_damping_c, post_shock_velocity_damping_c,
            adaptive_Cpsv == nullptr ? nullptr : adaptive_Cpsv->data(), qvisc_max_n,
            static_cast<int>(bc));
        break;
      case 2:
        apply_odd_even_damping_1d_kernel<2><<<nw.blocks(), 256>>>(
            a_n.data(), nullptr, nullptr, W_oe_n.data(), state.rho.data(), state.cs.data(),
            state.x_r.data(), state.v_r.data(), state.mass.data(),
            state.shock_time.empty() ? nullptr : state.shock_time.data(),
            state.Qvisc.data(), d_node_active, nw.begin, nw.end, n_cells,
            t_op, post_shock_heat_decay,
            odd_even_damping_c, post_shock_velocity_damping_c,
            adaptive_Cpsv == nullptr ? nullptr : adaptive_Cpsv->data(), qvisc_max_n,
            static_cast<int>(bc));
        break;
      default:
        apply_odd_even_damping_1d_kernel<0><<<nw.blocks(), 256>>>(
            a_n.data(), nullptr, nullptr, W_oe_n.data(), state.rho.data(), state.cs.data(),
            state.x_r.data(), state.v_r.data(), state.mass.data(),
            state.shock_time.empty() ? nullptr : state.shock_time.data(),
            state.Qvisc.data(), d_node_active, nw.begin, nw.end, n_cells,
            t_op, post_shock_heat_decay,
            odd_even_damping_c, post_shock_velocity_damping_c,
            adaptive_Cpsv == nullptr ? nullptr : adaptive_Cpsv->data(), qvisc_max_n,
            static_cast<int>(bc));
        break;
    }
    sync_kernel("Hydro1D: apply_odd_even_damping n kernel failed");
  }

  core::ScratchNodeField1D u_half;
  u_half.reset("hydro1d_u_half", n_nodes);
  predictor_update_kernel<<<nw.blocks(), 256>>>(
      state.v_r.data(), state.x_r.data(), u_half.data(), u_old.data(), r_old.data(),
      a_n.data(), d_node_active, nw.begin, nw.end, n_nodes, dt);
  sync_kernel("Hydro1D: predictor_update kernel failed");

  propagate_void_node_displacement_kernel<<<1, 1>>>(
      state.x_r.data(), state.v_r.data(), r_old.data(), d_node_active, n_nodes);
  sync_kernel("Hydro1D: propagate_void_node predictor failed");

  apply_boundary_1d(state, cfg);
  follow_void_region_nodes_1d(state, cfg);
  copy_array_kernel<<<nw.blocks(), 256>>>(u_half.data() + nw.begin,
                                          state.v_r.data() + nw.begin,
                                          nw.count());
  sync_kernel("Hydro1D: copy u_half kernel failed");
  if (const char* h1dbg = std::getenv("TENRYU_H1D_DEBUG")) {
    // Diagnostic-only (§6o.4c localization): raw per-rank dumps of the
    // predictor-stage intermediates, overwritten each step.
    int dbg_rank = 0;
    const char* rank_env = std::getenv("OMPI_COMM_WORLD_RANK");
    if (rank_env != nullptr) {
      dbg_rank = std::atoi(rank_env);
    }
    const auto dump_arr = [&](const char* tag, const double* dptr,
                              const int count) {
      std::vector<double> h(static_cast<std::size_t>(count));
      cuda_check(cudaMemcpy(h.data(), dptr,
                            static_cast<std::size_t>(count) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H1D debug dump D2H failed");
      const std::string path = std::string(h1dbg) + "_" + tag + "_q" + std::to_string(h1d_debug_seq_next()) + "_r" +
                               std::to_string(dbg_rank) + ".bin";
      if (FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    };
    dump_arr("an", a_n.data(), n_nodes);
    dump_arr("uh", u_half.data(), n_nodes);
    dump_arr("xr", state.x_r.data(), n_nodes);
    dump_arr("pq", pq_n_for_accel, n_cells);
    dump_arr("cs", state.cs.data(), static_cast<int>(state.cs.size()));
  }
  if (probe_on) {
    hydro_step_probe(probe_step, "P5_pred",
                     {{"vr", {state.v_r.data(), state.v_r.size()}},
                      {"xr", {state.x_r.data(), state.x_r.size()}},
                      {"uh", {u_half.data(), u_half.size()}}});
  }

  if (part.n_ranks > 1 && bufs != nullptr) {
    // Post-predictor ghost-node refresh: the corrector-phase AV evaluates
    // ghost-cell Q from node positions/velocities the predictor just moved;
    // without this the interface pq_half is built from stale ghost nodes
    // (measured 1e-13-level corrector divergence on the I1 radshock deck).
    double* mid_node_ptrs[2] = {state.x_r.data(), state.v_r.data()};
    parallel::exchange_node_fields(part, *bufs, mid_node_ptrs, 2, n_nodes,
                                   stream, 2);
  }
  refresh_geometry_and_density(state, cfg, d_cell_is_void, d_rho_clamp_count);
  if (midpoint_v2) {
    predictor_energy_half_step_kernel<<<blocks_cells, 256>>>(
        state.ee.data(), state.ei.data(), state.vol.data(), V_old.data(),
        state.mass.data(), Pe_old.data(), Pi_old.data(), Q_old.data(),
        d_hydro_active, d_cell_is_void, n_cells, use_two_temp ? 1 : 0,
        q_heat_to_electron);
    sync_kernel("Hydro1D: predictor_energy_half_step kernel failed");
  }
  enforce_eos_closure(state, cfg, use_two_temp, eos_views);
  if (const char* h1dbg3 = std::getenv("TENRYU_H1D_DEBUG")) {
    // Diagnostic-only (§6o.4e): half-step closure outputs and inputs.
    int dbg_rank = 0;
    if (const char* re = std::getenv("OMPI_COMM_WORLD_RANK")) {
      dbg_rank = std::atoi(re);
    }
    const auto dump_arr3 = [&](const char* tag, const double* dptr,
                               const int count) {
      if (dptr == nullptr || count <= 0) {
        return;
      }
      std::vector<double> h(static_cast<std::size_t>(count));
      cuda_check(cudaMemcpy(h.data(), dptr,
                            static_cast<std::size_t>(count) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H1D debug dump3 D2H failed");
      const std::string path = std::string(h1dbg3) + "_h" + tag + "_q" + std::to_string(h1d_debug_seq_next()) + "_r" +
                               std::to_string(dbg_rank) + ".bin";
      if (FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    };
    dump_arr3("pe", state.Pe.data(), n_cells);
    dump_arr3("rho", state.rho.data(), n_cells);
    dump_arr3("ee", state.ee.data(), n_cells);
    dump_arr3("zb", state.zbar.data(), n_cells);
    dump_arr3("ge", state.gamma_eff.empty() ? nullptr : state.gamma_eff.data(),
              n_cells);
    dump_arr3("ae", state.A_eff.empty() ? nullptr : state.A_eff.data(),
              n_cells);
    dump_arr3("vol", state.vol.data(), n_cells);
  }

  core::ScratchCellField1D E_hk_velocity_damp;
  bool h_half_valid = false;
  core::ScratchCellField1D H_ion_art_half;
  core::ScratchCellField1D chi_half;
  core::ScratchCellField1D q2_half;
  core::ScratchCellField1D div_u_half;
  core::ScratchCellField1D cs_half;
  cs_half.reset("hydro1d_cs_half", n_cells);
  compute_cell_sound_speed(cs_half, state, cfg, use_two_temp, eos_views);
  exchange_cell_ghosts({state.rho.data(), state.vol.data(), state.Te.data(),
                        state.Ti.data(), state.Pe.data(), state.Pi.data(),
                        cs_half.data()});
  if (av_heat_enabled || ion_art_heat_enabled) {
    core::CellField1DView chi_out;
    core::CellField1DView q2_out;
    core::CellField1DView div_u_out;
    if (av_heat_enabled) {
      chi_half.reset("hydro1d_chi_half", n_cells);
      chi_out = chi_half;
    }
    if (ion_art_heat_enabled) {
      q2_half.reset("hydro1d_q2_half", n_cells);
      div_u_half.reset("hydro1d_div_u_half", n_cells);
      q2_out = q2_half;
      div_u_out = div_u_half;
    }
    av.compute_Q_1d(state.Qvisc, state.rho, state.vol, state.x_r, state.v_r, state.Pe,
                    state.Pi, cs_half, state.hydro_active_device_ptr(), geom_code, chi_out,
                    q2_out, div_u_out, adaptive_c1, adaptive_c2);
  } else {
    av.compute_Q_1d(state.Qvisc, state.rho, state.vol, state.x_r, state.v_r, state.Pe,
                    state.Pi, cs_half, state.hydro_active_device_ptr(), geom_code, {},
                    {}, {}, adaptive_c1, adaptive_c2);
  }
  if (bulk_viscosity_enabled) {
    av.add_bulk_viscosity_1d(state.Qvisc, state.rho, state.vol, state.x_r, state.v_r,
                             cs_half, state.hydro_active_device_ptr(), adaptive_cbulk);
  }
  double qvisc_max_half = 0.0;
  if (shock_history_enabled && !state.shock_time.empty()) {
    qvisc_max_half = reduce_active_max(state.Qvisc.data(),
                                       "Hydro1D: reduce qvisc_max kernel failed",
                                       "Hydro1D: copy qvisc_max failed");
    if (qvisc_max_half > 0.0) {
      update_shock_time_kernel<<<cw.blocks(), 256>>>(
          state.shock_time.data(), state.Qvisc.data(), d_hydro_active,
          cw.begin, cw.end, n_cells,
          kPostShockThresholdFrac * qvisc_max_half, t_half);
      sync_kernel("Hydro1D: update_shock_time kernel failed");
    }
  }

  core::ScratchCellField1D P_half;
  core::ScratchCellField1D Pi_half;
  core::ScratchCellField1D Q_half;
  core::ScratchCellField1D rho_half;
  core::ScratchCellField1D Te_half;
  core::ScratchCellField1D Ti_half;
  P_half.reset("hydro1d_P_half", n_cells);
  Pi_half.reset("hydro1d_Pi_half", n_cells);
  Q_half.reset("hydro1d_Q_half", n_cells);
  if (use_two_temp) {
    rho_half.reset("hydro1d_rho_half", n_cells);
    Te_half.reset("hydro1d_Te_half", n_cells);
    Ti_half.reset("hydro1d_Ti_half", n_cells);
  }
  if (midpoint_v2) {
    // midpoint_v2: the stage closure already sits at t^{n+1/2}; averaging it
    // with t^n values would re-introduce the legacy quarter-step bias
    // (k01 P0-3), so the corrector energy uses the stage values directly.
    copy_array_kernel<<<blocks_cells, 256>>>(P_half.data(), state.Pe.data(),
                                             n_cells);
    copy_array_kernel<<<blocks_cells, 256>>>(Pi_half.data(), state.Pi.data(),
                                             n_cells);
    copy_array_kernel<<<blocks_cells, 256>>>(Q_half.data(), state.Qvisc.data(),
                                             n_cells);
  } else {
    // Ghost-inclusive (§6o.4l -> real defect): the odd-even pair-force
    // stencil inside the compatible energy update reads _half values at
    // i±1, so interface-adjacent owned cells consumed the ZERO-initialized
    // ghost P_half/Q_half (2-ulp ee flips from step 1's second half-step).
    // Inputs (Pe_old snapshot, closure outputs) are ghost-fresh, so the
    // ghost averages equal the owner's bitwise. Serial: pw == [0, n).
    average_arrays_kernel<<<pw.blocks(), 256>>>(
        P_half.data() + pw.begin, Pe_old.data() + pw.begin,
        state.Pe.data() + pw.begin, pw.count());
    average_arrays_kernel<<<pw.blocks(), 256>>>(
        Pi_half.data() + pw.begin, Pi_old.data() + pw.begin,
        state.Pi.data() + pw.begin, pw.count());
    average_arrays_kernel<<<pw.blocks(), 256>>>(
        Q_half.data() + pw.begin, Q_old.data() + pw.begin,
        state.Qvisc.data() + pw.begin, pw.count());
  }
  sync_kernel("Hydro1D: build P_half/Pi_half/Q_half kernels failed");
  if (probe_on) {
    hydro_step_probe(probe_step, "P6_qhalf",
                     {{"qh", {Q_half.data(), Q_half.size()}}});
  }
  if (av_heat_enabled) {
    const core::CellField1D* heat_source = use_two_temp ? &state.ei : &state.ee;
    if (h_half_pingpong_.size() != n_cells) {
      h_half_pingpong_.reset(n_cells);
    } else {
      cuda_check(cudaMemset(h_half_pingpong_.data(), 0,
                            static_cast<std::size_t>(n_cells) * sizeof(double)),
                 "Hydro1D: h_half_pingpong_ zero reset failed");
    }
    av.compute_H_1d(h_half_pingpong_, state.rho, *heat_source, chi_half,
                    state.x_r, state.hydro_active_device_ptr(), geom_code,
                    adaptive_heat_C);
    h_half_valid = true;
  }
  if (ion_art_heat_enabled) {
    H_ion_art_half.reset("hydro1d_H_ion_art_half", n_cells);
    const double* cv_i_ptr = state.cv_i.empty() ? nullptr : state.cv_i.data();
    switch (geom_code) {
      case 1:
        compute_ion_artificial_heat_1d_kernel<1><<<cw.blocks(), 256>>>(
            H_ion_art_half.data(), state.rho.data(), state.Ti.data(), state.Pe.data(),
            state.Pi.data(), cs_half.data(), cv_i_ptr, q2_half.data(), div_u_half.data(),
            state.x_r.data(), state.mass.data(), d_hydro_active,
            cw.begin, cw.end, n_cells, dt, ion_art_heat_c,
            state.gamma_eff.data(), state.A_eff.data());
        break;
      case 2:
        compute_ion_artificial_heat_1d_kernel<2><<<cw.blocks(), 256>>>(
            H_ion_art_half.data(), state.rho.data(), state.Ti.data(), state.Pe.data(),
            state.Pi.data(), cs_half.data(), cv_i_ptr, q2_half.data(), div_u_half.data(),
            state.x_r.data(), state.mass.data(), d_hydro_active,
            cw.begin, cw.end, n_cells, dt, ion_art_heat_c,
            state.gamma_eff.data(), state.A_eff.data());
        break;
      default:
        compute_ion_artificial_heat_1d_kernel<0><<<cw.blocks(), 256>>>(
            H_ion_art_half.data(), state.rho.data(), state.Ti.data(), state.Pe.data(),
            state.Pi.data(), cs_half.data(), cv_i_ptr, q2_half.data(), div_u_half.data(),
            state.x_r.data(), state.mass.data(), d_hydro_active,
            cw.begin, cw.end, n_cells, dt, ion_art_heat_c,
            state.gamma_eff.data(), state.A_eff.data());
        break;
    }
    sync_kernel("Hydro1D: ion_artificial_heat kernel failed");
  }
  if (use_two_temp) {
    cuda_check(cudaMemcpy(rho_half.data(), state.rho.data(), n_cells * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro1D: copy rho_half failed");
    cuda_check(cudaMemcpy(Te_half.data(), state.Te.data(), n_cells * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro1D: copy Te_half failed");
    cuda_check(cudaMemcpy(Ti_half.data(), state.Ti.data(), n_cells * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro1D: copy Ti_half failed");
  }

  core::ScratchCellField1D pq_half;
  pq_half.reset("hydro1d_pq_half", n_cells);
  if (pq_extra_ptr != nullptr) {
    build_cell_pq_kernel<true><<<pw.blocks(), 256>>>(
        pq_half.data(), state.Pe.data(), state.Pi.data(), state.Qvisc.data(),
        pq_extra_ptr, d_hydro_active, pw.begin, pw.end, n_cells);
  } else {
    build_cell_pq_kernel<false><<<pw.blocks(), 256>>>(
        pq_half.data(), state.Pe.data(), state.Pi.data(), state.Qvisc.data(),
        nullptr, d_hydro_active, pw.begin, pw.end, n_cells);
  }
  sync_kernel("Hydro1D: build_cell_pq half kernel failed");

  core::ScratchCellField1D pq_half_filtered;
  const double* pq_half_for_accel = pq_half.data();
  if (checkerboard_filter_beta > 0.0) {
    pq_half_filtered.reset("hydro1d_pq_half_filtered", n_cells);
    filter_pq_checkerboard_1d_impl(pq_half_filtered, pq_half, state.rho,
                                   state.Qvisc, d_hydro_active,
                                   checkerboard_filter_beta, cw.begin, cw.end);
    exchange_cell_ghosts({pq_half_filtered.data()});
    pq_half_for_accel = pq_half_filtered.data();
  }

  core::ScratchNodeField1D a_half;
  a_half.reset("hydro1d_a_half", n_nodes);
  const double boundary_pressure_half =
      pressure_ghost_1d(state, cfg, bc, t_op + 0.5 * dt);
  switch (geom_code) {
    case 1:
      compute_acceleration_1d_kernel<1><<<nw.blocks(), 256>>>(
          a_half.data(), state.mass.data(), state.x_r.data(), pq_half_for_accel,
          d_node_active, nw.begin, nw.end, n_cells, state.Pe.data(),
          state.Pi.data(), state.Qvisc.data(), boundary_pressure_half,
          static_cast<int>(bc));
      break;
    case 2:
      compute_acceleration_1d_kernel<2><<<nw.blocks(), 256>>>(
          a_half.data(), state.mass.data(), state.x_r.data(), pq_half_for_accel,
          d_node_active, nw.begin, nw.end, n_cells, state.Pe.data(),
          state.Pi.data(), state.Qvisc.data(), boundary_pressure_half,
          static_cast<int>(bc));
      break;
    default:
      compute_acceleration_1d_kernel<0><<<nw.blocks(), 256>>>(
          a_half.data(), state.mass.data(), state.x_r.data(), pq_half_for_accel,
          d_node_active, nw.begin, nw.end, n_cells, state.Pe.data(),
          state.Pi.data(), state.Qvisc.data(), boundary_pressure_half,
          static_cast<int>(bc));
      break;
  }
  sync_kernel("Hydro1D: compute_acceleration half kernel failed");
  if (probe_on) {
    hydro_step_probe(probe_step, "P7_ahalf",
                     {{"ah", {a_half.data(), a_half.size()}},
                      {"pqh", {pq_half.data(), pq_half.size()}}});
  }
  if (brag_params.enabled) {
    H_brag_half.reset("hydro1d_H_brag_half", n_cells);
    if (brag_params.species != 0) {
      H_brag_e_half.reset("hydro1d_H_brag_e_half", n_cells);
    }
    braginskii::compute_cell_stress_1d(
        brag_pi_rr.data(), brag_pi_tt.data(), H_brag_half.data(),
        H_brag_e_half.empty() ? nullptr : H_brag_e_half.data(),
        state.x_r.data(), state.v_r.data(), state.rho.data(),
        state.Ti.empty() ? nullptr : state.Ti.data(),
        state.Te.empty() ? nullptr : state.Te.data(),
        state.A_eff.empty() ? nullptr : state.A_eff.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(), state.vol.data(),
        d_hydro_active, n_cells, geom_code, brag_params,
        state.zmom_active ? state.zmom_r4.data() : nullptr);
    braginskii::add_node_accel_1d(
        a_half.data(), state.mass.data(), state.x_r.data(), state.vol.data(),
        brag_pi_rr.data(), brag_pi_tt.data(), d_node_active, n_cells,
        geom_code, brag_skip_outer, brag_params);
  }
  core::ScratchCellField1D odd_even_pair_force_half;
  if (nodal_velocity_damping_enabled) {
    core::ScratchCellField1D W_oe_half;
    W_oe_half.reset("hydro1d_W_oe_half", n_cells);
    compute_odd_even_weight_1d_impl(W_oe_half, state.rho, d_hydro_active,
                                    cw.begin, cw.end);
    exchange_cell_ghosts({W_oe_half.data(), state.Qvisc.data(),
                          state.shock_time.empty() ? nullptr
                                                   : state.shock_time.data()});
    if (run_compatible_energy) {
      odd_even_pair_force_half.reset("hydro1d_odd_even_pair_force_half", n_cells);
    } else {
      if (heat_oe_half_pingpong_.size() != n_cells) {
        heat_oe_half_pingpong_.reset(n_cells);
      } else {
        cuda_check(
            cudaMemset(heat_oe_half_pingpong_.data(), 0,
                       static_cast<std::size_t>(n_cells) * sizeof(double)),
            "Hydro1D: heat_oe_half_pingpong_ zero reset failed");
      }
    }
    switch (geom_code) {
      case 1:
        apply_odd_even_damping_1d_kernel<1><<<nw.blocks(), 256>>>(
            a_half.data(),
            run_compatible_energy ? nullptr : heat_oe_half_pingpong_.data(),
            run_compatible_energy ? odd_even_pair_force_half.data() : nullptr,
            W_oe_half.data(), state.rho.data(), cs_half.data(), state.x_r.data(),
            state.v_r.data(), state.mass.data(),
            state.shock_time.empty() ? nullptr : state.shock_time.data(),
            state.Qvisc.data(), d_node_active, nw.begin, nw.end, n_cells,
            t_half, post_shock_heat_decay,
            odd_even_damping_c, post_shock_velocity_damping_c,
            adaptive_Cpsv == nullptr ? nullptr : adaptive_Cpsv->data(), qvisc_max_half,
            static_cast<int>(bc));
        break;
      case 2:
        apply_odd_even_damping_1d_kernel<2><<<nw.blocks(), 256>>>(
            a_half.data(),
            run_compatible_energy ? nullptr : heat_oe_half_pingpong_.data(),
            run_compatible_energy ? odd_even_pair_force_half.data() : nullptr,
            W_oe_half.data(), state.rho.data(), cs_half.data(), state.x_r.data(),
            state.v_r.data(), state.mass.data(),
            state.shock_time.empty() ? nullptr : state.shock_time.data(),
            state.Qvisc.data(), d_node_active, nw.begin, nw.end, n_cells,
            t_half, post_shock_heat_decay,
            odd_even_damping_c, post_shock_velocity_damping_c,
            adaptive_Cpsv == nullptr ? nullptr : adaptive_Cpsv->data(), qvisc_max_half,
            static_cast<int>(bc));
        break;
      default:
        apply_odd_even_damping_1d_kernel<0><<<nw.blocks(), 256>>>(
            a_half.data(),
            run_compatible_energy ? nullptr : heat_oe_half_pingpong_.data(),
            run_compatible_energy ? odd_even_pair_force_half.data() : nullptr,
            W_oe_half.data(), state.rho.data(), cs_half.data(), state.x_r.data(),
            state.v_r.data(), state.mass.data(),
            state.shock_time.empty() ? nullptr : state.shock_time.data(),
            state.Qvisc.data(), d_node_active, nw.begin, nw.end, n_cells,
            t_half, post_shock_heat_decay,
            odd_even_damping_c, post_shock_velocity_damping_c,
            adaptive_Cpsv == nullptr ? nullptr : adaptive_Cpsv->data(), qvisc_max_half,
            static_cast<int>(bc));
        break;
    }
    sync_kernel("Hydro1D: apply_odd_even_damping half kernel failed");
    if (const char* h1dbga = std::getenv("TENRYU_H1D_DEBUG")) {
      int dbg_rank = 0;
      if (const char* re = std::getenv("OMPI_COMM_WORLD_RANK")) {
        dbg_rank = std::atoi(re);
      }
      std::vector<double> h(static_cast<std::size_t>(n_nodes));
      cuda_check(cudaMemcpy(h.data(), a_half.data(),
                            static_cast<std::size_t>(n_nodes) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H1D debug dump a D2H failed");
      const std::string path = std::string(h1dbga) + "_ahalf_q" +
                               std::to_string(h1d_debug_seq_next()) + "_r" +
                               std::to_string(dbg_rank) + ".bin";
      if (FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    }
  }
  if (nodal_velocity_damping_enabled && !run_compatible_energy) {
    if (!h_half_valid) {
      std::swap(h_half_pingpong_, heat_oe_half_pingpong_);
      h_half_valid = true;
    } else {
      add_arrays_inplace_kernel<<<cw.blocks(), 256>>>(
          h_half_pingpong_.data() + cw.begin,
          heat_oe_half_pingpong_.data() + cw.begin, cw.count());
      sync_kernel("Hydro1D: add odd_even heat kernel failed");
    }
  }

  core::ScratchNodeField1D r_half_compat;
  if (run_compatible_energy) {
    r_half_compat.reset("hydro1d_r_half_compat", n_nodes);
    cuda_check(cudaMemcpy(r_half_compat.data(), state.x_r.data(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro1D: copy compatible r_half failed");
  }
  core::ScratchNodeField1D r_half_gamma;
  if (pq_extra_ptr != nullptr && p_extra_work != nullptr) {
    r_half_gamma.reset("hydro1d_r_half_gamma", n_nodes);
    cuda_check(cudaMemcpy(r_half_gamma.data(), state.x_r.data(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro1D: copy gamma_r r_half failed");
  }

  corrector_update_kernel<<<nw.blocks(), 256>>>(
      state.v_r.data(), state.x_r.data(), u_old.data(), r_old.data(), u_half.data(),
      a_half.data(), d_node_active, nw.begin, nw.end, n_nodes, dt);
  sync_kernel("Hydro1D: corrector_update kernel failed");

  if (pq_extra_ptr != nullptr && p_extra_work != nullptr && !p_extra_work->empty()) {
    switch (geom_code) {
      case 1:
        gamma_r_force_work_1d_kernel<1><<<cw.blocks(), 256>>>(
            p_extra_work->data(), pq_extra_ptr,
            r_half_gamma.data(), u_old.data(), state.v_r.data(),
            d_node_active, cw.begin, cw.end, n_cells, dt);
        break;
      case 2:
        gamma_r_force_work_1d_kernel<2><<<cw.blocks(), 256>>>(
            p_extra_work->data(), pq_extra_ptr,
            r_half_gamma.data(), u_old.data(), state.v_r.data(),
            d_node_active, cw.begin, cw.end, n_cells, dt);
        break;
      default:
        gamma_r_force_work_1d_kernel<0><<<cw.blocks(), 256>>>(
            p_extra_work->data(), pq_extra_ptr,
            r_half_gamma.data(), u_old.data(), state.v_r.data(),
            d_node_active, cw.begin, cw.end, n_cells, dt);
        break;
    }
    sync_kernel("Hydro1D: gamma_r force work kernel failed");
  }

  propagate_void_node_displacement_kernel<<<1, 1>>>(
      state.x_r.data(), state.v_r.data(), r_old.data(), d_node_active, n_nodes);
  sync_kernel("Hydro1D: propagate_void_node corrector failed");

  apply_boundary_1d(state, cfg);
  follow_void_region_nodes_1d(state, cfg);
  if (probe_on) {
    hydro_step_probe(probe_step, "P8_corr",
                     {{"vr", {state.v_r.data(), state.v_r.size()}},
                      {"xr", {state.x_r.data(), state.x_r.size()}}});
  }
  core::ScratchNodeField1D v_corrected_compat;
  if (run_compatible_energy) {
    v_corrected_compat.reset("hydro1d_v_corrected_compat", n_nodes);
    cuda_check(cudaMemcpy(v_corrected_compat.data(), state.v_r.data(),
                          n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro1D: copy compatible corrected velocity failed");
  }
  if (hk_velocity_damper_active) {
    const std::vector<std::uint8_t> hk_near_front_host =
        compute_hk_velocity_near_front_mask_1d(
            state, Q_half, n_cells,
            cfg.numerics.hydro.hk_velocity_damper_grad_Te_max,
            cfg.numerics.hydro.hk_velocity_damper_grad_rho_max,
            cfg.numerics.hydro.hk_velocity_damper_guard_cells);
    std::uint8_t* d_hk_near_front =
        upload_uint8_array(hk_near_front_host, "hk_near_front");
    int* d_hk_active_pairs = nullptr;
    int* d_hk_front_blocked = nullptr;
    int* d_hk_tau_blocked = nullptr;
    double* d_hk_ke_removed = nullptr;
    double* d_hk_pack = static_cast<double*>(core::device_scratch_acquire(
        "hydro_1d:lagrangian_step:hk_pack", 4 * sizeof(double)));
    d_hk_ke_removed = d_hk_pack + 0;
    d_hk_active_pairs = reinterpret_cast<int*>(d_hk_pack + 1);
    d_hk_front_blocked = reinterpret_cast<int*>(d_hk_pack + 2);
    d_hk_tau_blocked = reinterpret_cast<int*>(d_hk_pack + 3);
    cuda_check(cudaMemset(d_hk_pack, 0, 4 * sizeof(double)),
               "Hydro1D: cudaMemset hk pack failed");

    core::ScratchNodeField1D hk_velocity_dv;
    hk_velocity_dv.reset("hydro1d_hk_velocity_dv", n_nodes);
    E_hk_velocity_damp.reset("hydro1d_E_hk_velocity_damp", n_cells);
    if (part.n_ranks > 1 && bufs != nullptr) {
      // The residual stencil reads v_r at +-1 nodes; refresh ghost nodes
      // updated by the corrector before evaluating it.
      double* hk_node_ptrs[2] = {state.x_r.data(), state.v_r.data()};
      parallel::exchange_node_fields(part, *bufs, hk_node_ptrs, 2, n_nodes,
                                     stream, 2);
    }
    compute_hk_velocity_residual_1d_kernel<<<nw.blocks(), 256>>>(
        hk_velocity_dv.data(), state.x_r.data(), state.v_r.data(), d_node_active,
        d_hydro_active, d_cell_is_void, d_hk_dominant_material,
        nw.begin, nw.end, n_cells);
    sync_kernel("Hydro1D: high-k velocity damper residual kernel failed");
    for (int color = 0; color < 2; ++color) {
      apply_hk_velocity_damper_impulse_1d_kernel<<<cw.blocks(), 256>>>(
          state.v_r.data(), E_hk_velocity_damp.data(), hk_velocity_dv.data(),
          state.x_r.data(), state.mass.data(), state.rho.data(), state.Te.data(),
          Q_half.data(), hk_sigma_R_max_ptr, d_node_active, d_hydro_active,
          d_cell_is_void, d_hk_dominant_material, d_hk_near_front,
          cw.begin, cw.end, n_cells, color,
          hk_velocity_damper_c, cfg.numerics.hydro.hk_velocity_damper_tau_min,
          cfg.numerics.hydro.hk_velocity_damper_grad_Te_max,
          cfg.numerics.hydro.hk_velocity_damper_grad_rho_max,
          d_hk_active_pairs, d_hk_front_blocked, d_hk_tau_blocked,
          d_hk_ke_removed);
      sync_kernel("Hydro1D: high-k velocity damper impulse kernel failed");
    }

    int hk_active_pairs = 0;
    int hk_front_blocked = 0;
    int hk_tau_blocked = 0;
    double hk_ke_removed = 0.0;
    double h_hk[4] = {0.0, 0.0, 0.0, 0.0};
    cuda_check(cudaMemcpy(h_hk, d_hk_pack, sizeof(h_hk), cudaMemcpyDeviceToHost),
               "Hydro1D: copy hk pack failed");
    hk_ke_removed = h_hk[0];
    std::memcpy(&hk_active_pairs, &h_hk[1], sizeof(int));
    std::memcpy(&hk_front_blocked, &h_hk[2], sizeof(int));
    std::memcpy(&hk_tau_blocked, &h_hk[3], sizeof(int));
    if (part.n_ranks > 1) {
      double hk_diag[4] = {static_cast<double>(hk_active_pairs),
                           static_cast<double>(hk_front_blocked),
                           static_cast<double>(hk_tau_blocked),
                           hk_ke_removed};
      const parallel::Reduction reduction(part.n_ranks);
      reduction.allreduce_sum(hk_diag, 4);
      hk_active_pairs = static_cast<int>(hk_diag[0]);
      hk_front_blocked = static_cast<int>(hk_diag[1]);
      hk_tau_blocked = static_cast<int>(hk_diag[2]);
      hk_ke_removed = hk_diag[3];
    }
    core::log_debug("[hk_vdamp] step=" + std::to_string(state.step) +
                    " active_pairs=" + std::to_string(hk_active_pairs) +
                    " front_blocked=" + std::to_string(hk_front_blocked) +
                    " tau_blocked=" + std::to_string(hk_tau_blocked) +
                    " KE_removed=" + format_scientific(hk_ke_removed));
  }
  // Host-mirror sync is deferred to the end of the step: the checked sync
  // hard-asserts on a non-positive volume, which would bypass the soft-fail
  // retry detection below (nothing between here and the tail moves nodes).
  refresh_geometry_and_density(state, cfg, d_cell_is_void, d_rho_clamp_count);

  if (run_compatible_energy) {
    double ghost_pq_half = 0.0;
    if (bc == HydroBoundaryType::PRESSURE) {
      TENRYU_ASSERT(state.pressure_drive_1d.has_value(),
                    "pressure boundary requires initialized pressure_drive_1d table");
      double qvisc_last = 0.0;
      cuda_check(cudaMemcpy(&qvisc_last, state.Qvisc.data() + n_cells - 1,
                            sizeof(double), cudaMemcpyDeviceToHost),
                 "Hydro1D: copy compatible Qvisc boundary failed");
      ghost_pq_half =
          state.pressure_drive_1d->eval(t_op + 0.5 * dt) + qvisc_last;
    } else if (bc == HydroBoundaryType::FIXED ||
               bc == HydroBoundaryType::REFLECT) {
      double pe_last = 0.0;
      double pi_last = 0.0;
      double qvisc_last = 0.0;
      cuda_check(cudaMemcpy(&pe_last, state.Pe.data() + n_cells - 1,
                            sizeof(double), cudaMemcpyDeviceToHost),
                 "Hydro1D: copy compatible Pe boundary failed");
      cuda_check(cudaMemcpy(&pi_last, state.Pi.data() + n_cells - 1,
                            sizeof(double), cudaMemcpyDeviceToHost),
                 "Hydro1D: copy compatible Pi boundary failed");
      cuda_check(cudaMemcpy(&qvisc_last, state.Qvisc.data() + n_cells - 1,
                            sizeof(double), cudaMemcpyDeviceToHost),
                 "Hydro1D: copy compatible Qvisc boundary failed");
      ghost_pq_half = pe_last + pi_last + qvisc_last;
    }
    cuda_check(cudaMemcpy(state.ee.data(), e_old.data(), n_cells * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro1D: restore compatible ee_old failed");
    if (use_two_temp) {
      cuda_check(cudaMemcpy(state.ei.data(), ei_old.data(), n_cells * sizeof(double),
                            cudaMemcpyDeviceToDevice),
                 "Hydro1D: restore compatible ei_old failed");
    }
    if (const char* h1dbg2 = std::getenv("TENRYU_H1D_DEBUG")) {
      // Diagnostic-only (§6o.4d): corrector-stage _half intermediates just
      // before the compatible energy update, overwritten each step.
      int dbg_rank = 0;
      if (const char* re = std::getenv("OMPI_COMM_WORLD_RANK")) {
        dbg_rank = std::atoi(re);
      }
      const auto dump_arr2 = [&](const char* tag, const double* dptr,
                                 const int count) {
        if (dptr == nullptr) {
          return;
        }
        std::vector<double> h(static_cast<std::size_t>(count));
        cuda_check(cudaMemcpy(h.data(), dptr,
                              static_cast<std::size_t>(count) * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "H1D debug dump2 D2H failed");
        const std::string path = std::string(h1dbg2) + "_c" + tag + "_q" + std::to_string(h1d_debug_seq_next()) + "_r" +
                                 std::to_string(dbg_rank) + ".bin";
        if (FILE* fp = std::fopen(path.c_str(), "wb")) {
          std::fwrite(h.data(), sizeof(double), h.size(), fp);
          std::fclose(fp);
        }
      };
      dump_arr2("ph", P_half.data(), n_cells);
      dump_arr2("pih", Pi_half.data(), n_cells);
      dump_arr2("qh", Q_half.data(), n_cells);
      dump_arr2("pqh", pq_half_for_accel, n_cells);
      dump_arr2("vc", v_corrected_compat.data(),
                static_cast<int>(v_corrected_compat.size()));
      dump_arr2("rh", r_half_compat.data(),
                static_cast<int>(r_half_compat.size()));
      dump_arr2("uo", u_old.data(), n_nodes);
    }
    run_compatible_energy_update_1d(
        state, cfg, dt, u_old.data(), v_corrected_compat.data(), pq_half_for_accel,
        P_half.data(), Pi_half.data(), Q_half.data(), r_half_compat.data(),
        odd_even_pair_force_half.empty() ? nullptr : odd_even_pair_force_half.data(),
        pq_extra_ptr,
        n_cells, ghost_pq_half, bc == HydroBoundaryType::FREE ? 1 : 0,
        d_hydro_floor, d_hydro_clamp_count);
    if (const char* h1dbgw = std::getenv("TENRYU_H1D_DEBUG")) {
      // Diagnostic-only (§6o.4l audit lane): ee right after the compatible
      // energy update, sequence-numbered like the rest of the H1D dumps.
      int dbg_rank = 0;
      if (const char* re = std::getenv("OMPI_COMM_WORLD_RANK")) {
        dbg_rank = std::atoi(re);
      }
      std::vector<double> h(static_cast<std::size_t>(n_cells));
      cuda_check(cudaMemcpy(h.data(), state.ee.data(),
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H1D debug dump w D2H failed");
      const std::string path = std::string(h1dbgw) + "_wee_q" +
                               std::to_string(h1d_debug_seq_next()) + "_r" +
                               std::to_string(dbg_rank) + ".bin";
      if (FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    }
    if (use_two_temp) {
      const auto& mat = cfg.materials.materials.front();
      const bool use_table_cv_qei = hydro_has_table_eos_backend_data(eos_views);
      const double* cv_e_ptr =
          (use_table_cv_qei && !state.cv_e.empty()) ? state.cv_e.data() : nullptr;
      const double* cv_i_ptr =
          (use_table_cv_qei && !state.cv_i.empty()) ? state.cv_i.data() : nullptr;
      if (state.zmom_active) {
        apply_qei_transfer_2t_kernel<true><<<cw.blocks(), 256>>>(
            state.ee.data(), state.ei.data(), rho_half.data(), Te_half.data(),
            Ti_half.data(), state.mass.data(), state.zbar.data(), cv_e_ptr, cv_i_ptr,
            d_hydro_active, cw.begin, cw.end, n_cells, dt,
            state.gamma_eff.data(), state.A_eff.data(), mat.Z,
            cfg.numerics.hydro.qei_multiplier,
            d_hydro_floor, d_hydro_clamp_count, state.zmom_r2.data());
      } else {
        apply_qei_transfer_2t_kernel<false><<<cw.blocks(), 256>>>(
            state.ee.data(), state.ei.data(), rho_half.data(), Te_half.data(),
            Ti_half.data(), state.mass.data(), state.zbar.data(), cv_e_ptr, cv_i_ptr,
            d_hydro_active, cw.begin, cw.end, n_cells, dt,
            state.gamma_eff.data(), state.A_eff.data(), mat.Z,
            cfg.numerics.hydro.qei_multiplier,
            d_hydro_floor, d_hydro_clamp_count, nullptr);
      }
      sync_kernel("Hydro1D: compatible qei transfer kernel failed");
    }
  } else if (use_two_temp) {
    const auto& mat = cfg.materials.materials.front();
    switch (geom_code) {
      case 1:
        energy_update_with_old_volume_2t_kernel<1><<<cw.blocks(), 256>>>(
            state.ee.data(), state.ei.data(), e_old.data(), ei_old.data(), rho_half.data(),
            Te_half.data(), Ti_half.data(), state.x_r.data(), r_old.data(), u_half.data(),
            state.vol.data(), V_old.data(), state.mass.data(), P_half.data(), Pi_half.data(),
            Q_half.data(), state.zbar.data(), d_hydro_active,
            cw.begin, cw.end, n_cells, dt,
            state.gamma_eff.data(), state.A_eff.data(), mat.Z,
            cfg.numerics.hydro.qei_multiplier,
            eos_views.tab_ion, eos_views.tab_ele,
            nullptr, nullptr, q_heat_to_electron, legacy_volume_compatible_energy,
            state.eta_compatible.empty() ? nullptr : state.eta_compatible.data(),
            d_hydro_floor, d_hydro_clamp_count);
        break;
      case 2:
        energy_update_with_old_volume_2t_kernel<2><<<cw.blocks(), 256>>>(
            state.ee.data(), state.ei.data(), e_old.data(), ei_old.data(), rho_half.data(),
            Te_half.data(), Ti_half.data(), state.x_r.data(), r_old.data(), u_half.data(),
            state.vol.data(), V_old.data(), state.mass.data(), P_half.data(), Pi_half.data(),
            Q_half.data(), state.zbar.data(), d_hydro_active,
            cw.begin, cw.end, n_cells, dt,
            state.gamma_eff.data(), state.A_eff.data(), mat.Z,
            cfg.numerics.hydro.qei_multiplier,
            eos_views.tab_ion, eos_views.tab_ele,
            nullptr, nullptr, q_heat_to_electron, legacy_volume_compatible_energy,
            state.eta_compatible.empty() ? nullptr : state.eta_compatible.data(),
            d_hydro_floor, d_hydro_clamp_count);
        break;
      default:
        energy_update_with_old_volume_2t_kernel<0><<<cw.blocks(), 256>>>(
            state.ee.data(), state.ei.data(), e_old.data(), ei_old.data(), rho_half.data(),
            Te_half.data(), Ti_half.data(), state.x_r.data(), r_old.data(), u_half.data(),
            state.vol.data(), V_old.data(), state.mass.data(), P_half.data(), Pi_half.data(),
            Q_half.data(), state.zbar.data(), d_hydro_active,
            cw.begin, cw.end, n_cells, dt,
            state.gamma_eff.data(), state.A_eff.data(), mat.Z,
            cfg.numerics.hydro.qei_multiplier,
            eos_views.tab_ion, eos_views.tab_ele,
            nullptr, nullptr, q_heat_to_electron, legacy_volume_compatible_energy,
            state.eta_compatible.empty() ? nullptr : state.eta_compatible.data(),
            d_hydro_floor, d_hydro_clamp_count);
        break;
    }
  } else {
    switch (geom_code) {
      case 1:
        energy_update_with_old_volume_kernel<1><<<cw.blocks(), 256>>>(
            state.ee.data(), e_old.data(), state.x_r.data(), r_old.data(), u_half.data(),
            state.vol.data(), V_old.data(), state.mass.data(), P_half.data(), Q_half.data(),
            d_hydro_active, cw.begin, cw.end, n_cells, dt,
            legacy_volume_compatible_energy,
            state.eta_compatible.empty() ? nullptr : state.eta_compatible.data(),
            d_hydro_floor, d_hydro_clamp_count);
        break;
      case 2:
        energy_update_with_old_volume_kernel<2><<<cw.blocks(), 256>>>(
            state.ee.data(), e_old.data(), state.x_r.data(), r_old.data(), u_half.data(),
            state.vol.data(), V_old.data(), state.mass.data(), P_half.data(), Q_half.data(),
            d_hydro_active, cw.begin, cw.end, n_cells, dt,
            legacy_volume_compatible_energy,
            state.eta_compatible.empty() ? nullptr : state.eta_compatible.data(),
            d_hydro_floor, d_hydro_clamp_count);
        break;
      default:
        energy_update_with_old_volume_kernel<0><<<cw.blocks(), 256>>>(
            state.ee.data(), e_old.data(), state.x_r.data(), r_old.data(), u_half.data(),
            state.vol.data(), V_old.data(), state.mass.data(), P_half.data(), Q_half.data(),
            d_hydro_active, cw.begin, cw.end, n_cells, dt,
            legacy_volume_compatible_energy,
            state.eta_compatible.empty() ? nullptr : state.eta_compatible.data(),
            d_hydro_floor, d_hydro_clamp_count);
        break;
    }
  }
  sync_kernel("Hydro1D: energy_update kernel failed");
  if (!E_hk_velocity_damp.empty()) {
    double* const heat_target = use_two_temp ? state.ei.data() : state.ee.data();
    apply_hk_velocity_damper_heat_1d_kernel<<<cw.blocks(), 256>>>(
        heat_target, E_hk_velocity_damp.data(), state.mass.data(), d_hydro_active,
        cw.begin, cw.end, n_cells);
    sync_kernel("Hydro1D: high-k velocity damper heat kernel failed");
  }
  if (h_half_valid) {
    double* const heat_target = use_two_temp ? state.ei.data() : state.ee.data();
    const int clamp_to_zero =
        use_two_temp
            ? (((use_helmholtz_spline_backend(eos_views) || use_helmholtz_jet_backend(eos_views)) ||
                eos_views.tab_ion.n_rho > 0)
                   ? 0
                   : 1)
            : 1;
    apply_artificial_heat_kernel<<<cw.blocks(), 256>>>(
        heat_target, h_half_pingpong_.data(), state.mass.data(), d_hydro_active,
        cw.begin, cw.end, n_cells, dt, clamp_to_zero,
        d_hydro_floor, d_hydro_clamp_count);
    sync_kernel("Hydro1D: artificial_heat kernel failed");
  }
  if (!H_ion_art_half.empty()) {
    const int clamp_to_zero =
        (((use_helmholtz_spline_backend(eos_views) || use_helmholtz_jet_backend(eos_views)) ||
          eos_views.tab_ion.n_rho > 0)
             ? 0
             : 1);
    apply_artificial_heat_kernel<<<cw.blocks(), 256>>>(
        state.ei.data(), H_ion_art_half.data(), state.mass.data(), d_hydro_active,
        cw.begin, cw.end, n_cells, dt, clamp_to_zero,
        d_hydro_floor, d_hydro_clamp_count);
    sync_kernel("Hydro1D: ion_artificial_heat apply kernel failed");
  }
  if (!H_brag_half.empty()) {
    double* const heat_target =
        use_two_temp ? state.ei.data() : state.ee.data();
    const int clamp_to_zero =
        use_two_temp
            ? (((use_helmholtz_spline_backend(eos_views) || use_helmholtz_jet_backend(eos_views)) ||
                eos_views.tab_ion.n_rho > 0)
                   ? 0
                   : 1)
            : 1;
    apply_artificial_heat_kernel<<<cw.blocks(), 256>>>(
        heat_target, H_brag_half.data(), state.mass.data(), d_hydro_active,
        cw.begin, cw.end, n_cells, dt, clamp_to_zero,
        d_hydro_floor, d_hydro_clamp_count);
    sync_kernel("Hydro1D: braginskii_heat apply kernel failed");
  }
  if (!H_brag_e_half.empty()) {
    // Electron-viscous heating goes to the electron energy in 2T and to the
    // single (total) energy in 1T — state.ee in both cases.
    const int clamp_to_zero =
        use_two_temp
            ? (((use_helmholtz_spline_backend(eos_views) || use_helmholtz_jet_backend(eos_views)) ||
                eos_views.tab_ion.n_rho > 0)
                   ? 0
                   : 1)
            : 1;
    apply_artificial_heat_kernel<<<cw.blocks(), 256>>>(
        state.ee.data(), H_brag_e_half.data(), state.mass.data(),
        d_hydro_active, cw.begin, cw.end, n_cells, dt, clamp_to_zero,
        d_hydro_floor, d_hydro_clamp_count);
    sync_kernel("Hydro1D: braginskii_electron_heat apply kernel failed");
  }
  if (post_shock_heat_enabled && !state.shock_time.empty()) {
    const int clamp_to_zero =
        use_two_temp
            ? (((use_helmholtz_spline_backend(eos_views) || use_helmholtz_jet_backend(eos_views)) ||
                eos_views.tab_ion.n_rho > 0)
                   ? 0
                   : 1)
            : 1;
    switch (geom_code) {
      case 1:
        apply_post_shock_heat_kernel<1><<<cw.blocks(), 256>>>(
            state.ee.data(), use_two_temp ? state.ei.data() : nullptr, state.rho.data(),
            cs_half.data(), state.x_r.data(), state.mass.data(), P_half.data(),
            Pi_half.data(), state.shock_time.data(), d_hydro_active,
            cw.begin, cw.end, n_cells, dt, t_half,
            post_shock_heat_c, post_shock_heat_decay, use_two_temp ? 1 : 0,
            clamp_to_zero, d_hydro_floor, d_hydro_clamp_count);
        break;
      case 2:
        apply_post_shock_heat_kernel<2><<<cw.blocks(), 256>>>(
            state.ee.data(), use_two_temp ? state.ei.data() : nullptr, state.rho.data(),
            cs_half.data(), state.x_r.data(), state.mass.data(), P_half.data(),
            Pi_half.data(), state.shock_time.data(), d_hydro_active,
            cw.begin, cw.end, n_cells, dt, t_half,
            post_shock_heat_c, post_shock_heat_decay, use_two_temp ? 1 : 0,
            clamp_to_zero, d_hydro_floor, d_hydro_clamp_count);
        break;
      default:
        apply_post_shock_heat_kernel<0><<<cw.blocks(), 256>>>(
            state.ee.data(), use_two_temp ? state.ei.data() : nullptr, state.rho.data(),
            cs_half.data(), state.x_r.data(), state.mass.data(), P_half.data(),
            Pi_half.data(), state.shock_time.data(), d_hydro_active,
            cw.begin, cw.end, n_cells, dt, t_half,
            post_shock_heat_c, post_shock_heat_decay, use_two_temp ? 1 : 0,
            clamp_to_zero, d_hydro_floor, d_hydro_clamp_count);
        break;
    }
    sync_kernel("Hydro1D: post_shock_heat kernel failed");
  }
  if (electron_odd_even_enabled) {
    core::ScratchNodeField1D ee_oe_flux;
    ee_oe_flux.reset("hydro1d_ee_oe_flux", n_nodes);
    compute_electron_odd_even_flux_1d_kernel<<<nw.blocks(), 256>>>(
        ee_oe_flux.data(), state.ee.data(), state.rho.data(), state.vol.data(),
        state.mass.data(), d_hydro_active, d_cell_is_void,
        nw.begin, nw.end, n_cells, ee_odd_even_c);
    sync_kernel("Hydro1D: electron_odd_even_flux kernel failed");
    apply_electron_odd_even_flux_1d_kernel<<<cw.blocks(), 256>>>(
        state.ee.data(), ee_oe_flux.data(), state.mass.data(), d_hydro_active,
        d_cell_is_void, cw.begin, cw.end, n_cells);
    sync_kernel("Hydro1D: electron_odd_even_apply kernel failed");
  }

  {
    double h_floors[2] = {0.0, 0.0};
    cuda_check(cudaMemcpy(h_floors, d_floors_pack, sizeof(h_floors),
                          cudaMemcpyDeviceToHost),
               "Hydro1D: copy floors pack failed");
    local_E_floor = h_floors[0];
    const auto* floors_bytes = reinterpret_cast<const unsigned char*>(h_floors);
    std::memcpy(&local_clamp_count, floors_bytes + 8, sizeof(int));
    std::memcpy(&local_rho_clamp_count, floors_bytes + 12, sizeof(int));
  }

  if (probe_on) {
    hydro_step_probe(probe_step, "P9_preren",
                     {{"ee", {state.ee.data(), state.ee.size()}}});
  }
  if (apply_1t_energy_renorm) {
    const double E_after =
        allreduce_sum_scalar(diagnostics::compute_energy_budget_1d(state).E_total);
    const double delta_E = E_before - E_after;
    if (std::abs(E_before) > 0.0) {
      const double rel_corr = std::abs(delta_E) / std::abs(E_before);
      if (rel_corr > 1.0e-12) {
        tenryu::core::log_debug("Hydro1D energy correction rel=" +
                                format_scientific(rel_corr));
      }
    }

    if (std::abs(delta_E) > 0.0) {
      double* d_renorm_sums = static_cast<double*>(core::device_scratch_acquire(
          "hydro_1d:lagrangian_step:d_renorm_sums", 2 * sizeof(double)));
      using CountingIt = core::CountingInputIterator<int>;
      CountingIt indices(cw.begin);
      auto mass_it = core::TransformInputIterator<double, ActiveMassOp, CountingIt>(
          indices, ActiveMassOp{state.mass.data(), d_hydro_active});
      auto internal_it =
          core::TransformInputIterator<double, ActiveInternalOp, CountingIt>(
              indices, ActiveInternalOp{state.mass.data(), state.ee.data(),
                                        d_hydro_active});
      std::size_t mass_bytes = 0U;
      std::size_t internal_bytes = 0U;
      cuda_check(cub::DeviceReduce::Sum(nullptr, mass_bytes, mass_it,
                                        d_renorm_sums, cw.count()),
                 "Hydro1D: renorm mass reduce size query failed");
      cuda_check(cub::DeviceReduce::Sum(nullptr, internal_bytes, internal_it,
                                        d_renorm_sums + 1, cw.count()),
                 "Hydro1D: renorm internal reduce size query failed");
      const std::size_t renorm_temp_bytes = std::max(mass_bytes, internal_bytes);
      void* d_renorm_temp = core::device_scratch_acquire(
          "hydro_1d:lagrangian_step:d_renorm_reduce_temp", renorm_temp_bytes);
      std::size_t mass_temp_bytes = renorm_temp_bytes;
      cuda_check(cub::DeviceReduce::Sum(d_renorm_temp, mass_temp_bytes, mass_it,
                                        d_renorm_sums, cw.count()),
                 "Hydro1D: renorm mass reduce failed");
      std::size_t internal_temp_bytes = renorm_temp_bytes;
      cuda_check(cub::DeviceReduce::Sum(d_renorm_temp, internal_temp_bytes,
                                        internal_it, d_renorm_sums + 1, cw.count()),
                 "Hydro1D: renorm internal reduce failed");
      sync_kernel("Hydro1D: renorm reductions failed");

      double h_renorm[2] = {0.0, 0.0};
      cuda_check(cudaMemcpy(h_renorm, d_renorm_sums, sizeof(h_renorm),
                            cudaMemcpyDeviceToHost),
                 "Hydro1D: memcpy renorm sums failed");
      const double active_mass = allreduce_sum_scalar(h_renorm[0]);
      const double active_internal = allreduce_sum_scalar(h_renorm[1]);

      if (active_internal > 0.0) {
        const double scale = (active_internal + delta_E) / active_internal;
        scale_active_energy_kernel<<<cw.blocks(), 256>>>(state.ee.data() + cw.begin,
                                                          d_hydro_active + cw.begin,
                                                          cw.count(), scale);
        sync_kernel("Hydro1D: scale_active_energy kernel failed");
      } else if (active_mass > 0.0) {
        const double de = delta_E / active_mass;
        shift_active_energy_kernel<<<cw.blocks(), 256>>>(state.ee.data() + cw.begin,
                                                          d_hydro_active + cw.begin,
                                                          cw.count(), de);
        sync_kernel("Hydro1D: shift_active_energy kernel failed");
      }

      int first_active = -1;
      if (state.hydro_active.empty()) {
        first_active = 0;
      } else {
        for (std::size_t i = static_cast<std::size_t>(cw.begin);
             i < static_cast<std::size_t>(cw.end) &&
             i < state.hydro_active.size();
             ++i) {
          if (state.hydro_active[i] != 0) {
            first_active = static_cast<int>(i);
            break;
          }
        }
      }
      if (part.n_ranks > 1) {
        const parallel::Reduction reduction(part.n_ranks);
        double key = first_active >= 0 ? static_cast<double>(first_active)
                                       : 1.0e18;
        key = reduction.allreduce_min(key);
        first_active = key < 1.0e18 ? static_cast<int>(key) : -1;
      }

      if (first_active >= 0) {
        const double E_corrected =
            allreduce_sum_scalar(diagnostics::compute_energy_budget_1d(state).E_total);
        const double residual = E_before - E_corrected;
        // The residual patch targets one global cell; only its owner holds
        // valid ee/mass and applies the kernel (all ranks agree on the
        // residual value via the global sums above).
        const bool owns_first_active =
            first_active >= cw.begin && first_active < cw.end;
        if (std::abs(residual) > 0.0 && owns_first_active) {
          double ee_first = 0.0;
          double m_first = 0.0;
          cuda_check(cudaMemcpy(&ee_first,
                                state.ee.data() + first_active,
                                sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "Hydro1D: copy first_active ee failed");
          cuda_check(cudaMemcpy(&m_first,
                                state.mass.data() + first_active,
                                sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "Hydro1D: copy first_active mass failed");
          if (m_first > 0.0) {
            const double delta_ee = residual / m_first;
            const double rel_delta =
                std::abs(delta_ee) / std::max(std::abs(ee_first), 1.0e-30);
            if (rel_delta > 0.1) {
              tenryu::core::log_warning(
                  "Hydro1D residual correction concentrated in first active cell: rel_delta_ee=" +
                  format_scientific(rel_delta));
            }
          }
          add_first_active_residual_kernel<<<1, 1>>>(
              state.ee.data(), state.mass.data(), first_active, residual);
          sync_kernel("Hydro1D: add_first_active_residual kernel failed");
        }
      }
    }
  }

  if (probe_on) {
    hydro_step_probe(probe_step, "P10_postren",
                     {{"ee", {state.ee.data(), state.ee.size()}}});
  }
  const bool compatible_table_reclosure =
      run_compatible_energy && hydro_has_table_eos_backend_data(eos_views);
  enforce_eos_closure(state, cfg, use_two_temp, eos_views,
                      compatible_table_reclosure, compatible_table_reclosure);
  if (cs_pingpong_.size() != n_cells) {
    cs_pingpong_.reset(n_cells);
  } else {
    cuda_check(cudaMemset(cs_pingpong_.data(), 0,
                          static_cast<std::size_t>(n_cells) * sizeof(double)),
               "Hydro1D: cs_pingpong_ zero reset failed");
  }
  compute_cell_sound_speed(cs_pingpong_, state, cfg, use_two_temp, eos_views);
  // ex1.5c: the post-corrector refresh/closure recomputed rho/vol/EOS on the
  // owned window only; downstream operators (conduction reads ghost rho/vol
  // for the interface-face coefficients) need the FINAL ghosts, not the
  // mid-step values left by ex1.5b (measured: 17-cell ulp STS spread seeded
  // at the interface on the coupled radshock probe).
  // OPEN-HYDRO-CORNER tail: both exchanges must ALSO precede the end-of-step
  // AV recompute below — its limiter stencil at interface-adjacent owned
  // cells reads ghost-cell node velocities, so exchanging after Q left the
  // stored Qvisc computed from one-motion-stale coords (measured: gate
  // Qvisc bit mismatch at the interface cell while every state field stayed
  // bitwise).
  exchange_cell_ghosts({state.rho.data(), state.vol.data(), state.Te.data(),
                        state.Ti.data(), state.Pe.data(), state.Pi.data(),
                        cs_pingpong_.data()});
  if (part.n_ranks > 1 && bufs != nullptr) {
    double* node_field_ptrs[2] = {state.x_r.data(), state.v_r.data()};
    parallel::exchange_node_fields(part, *bufs, node_field_ptrs, 2, n_nodes, stream,
                                   2);
  }
  av.compute_Q_1d(state.Qvisc, state.rho, state.vol, state.x_r, state.v_r, state.Pe,
                  state.Pi, cs_pingpong_, state.hydro_active_device_ptr(), geom_code,
                  {}, {}, {}, adaptive_c1, adaptive_c2);
  if (bulk_viscosity_enabled) {
    av.add_bulk_viscosity_1d(state.Qvisc, state.rho, state.vol, state.x_r, state.v_r,
                             cs_pingpong_, state.hydro_active_device_ptr(),
                             adaptive_cbulk);
  }
  std::swap(state.cs, cs_pingpong_);

  if (const char* h1dbgz = std::getenv("TENRYU_H1D_DEBUG")) {
    int dbg_rank = 0;
    if (const char* re = std::getenv("OMPI_COMM_WORLD_RANK")) {
      dbg_rank = std::atoi(re);
    }
    const auto dump_z = [&](const char* tag, const double* dptr) {
      std::vector<double> h(static_cast<std::size_t>(n_cells));
      cuda_check(cudaMemcpy(h.data(), dptr,
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H1D debug dumpz D2H failed");
      const std::string path = std::string(h1dbgz) + "_z" + tag + "_q" +
                               std::to_string(h1d_debug_seq_next()) + "_r" +
                               std::to_string(dbg_rank) + ".bin";
      if (FILE* fp = std::fopen(path.c_str(), "wb")) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    };
    dump_z("ee", state.ee.data());
    dump_z("te", state.Te.data());
    dump_z("rho", state.rho.data());
    dump_z("pe", state.Pe.data());
    dump_z("zb", state.zbar.data());
    if (!state.cv_e.empty()) {
      dump_z("cve", state.cv_e.data());
    }
  }
  if (E_floor_injected != nullptr) {
    *E_floor_injected += std::max(local_E_floor, 0.0);
  }
  if (clamp_count != nullptr) {
    *clamp_count += std::max(local_clamp_count, 0);
  }
  if (rho_clamp_count != nullptr) {
    *rho_clamp_count += std::max(local_rho_clamp_count, 0);
  }

  // W-B robustness: detect compression-driven node crossing (non-positive
  // cell volume) at the end of the Lagrangian sub-step. With the driver
  // full-step retry enabled this becomes a soft failure (restore + dt/2 via
  // the standard snapshot machinery); otherwise fail hard here, earlier and
  // with better context than the downstream laser/diagnostics geometry
  // recompute that used to trip on it.
  {
    int* d_failing_cell = nullptr;
    d_failing_cell = static_cast<int*>(core::device_scratch_acquire(
        "hydro_1d:lagrangian_step:d_failing_cell", sizeof(int)));
    const int init_sentinel = n_cells;
    cuda_check(cudaMemcpy(d_failing_cell, &init_sentinel, sizeof(int),
                          cudaMemcpyHostToDevice),
               "Hydro1D: init failing_cell failed");
    find_nonpositive_volume_1d_kernel<<<cw.blocks(), 256>>>(
        state.vol.data(), d_failing_cell, cw.begin, cw.end, n_cells);
    sync_kernel("Hydro1D find_nonpositive_volume kernel failed");
    int failing_cell = n_cells;
    cuda_check(cudaMemcpy(&failing_cell, d_failing_cell, sizeof(int),
                          cudaMemcpyDeviceToHost),
               "Hydro1D: read failing_cell failed");
    if (failing_cell < n_cells) {
      double failing_vol = 0.0;
      cuda_check(cudaMemcpy(&failing_vol,
                            state.vol.data() + failing_cell,
                            sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Hydro1D: read failing volume failed");
      if (cfg.numerics.hydro.driver_full_step_retry_enabled) {
        step_result.retry_required = true;
        step_result.reason = "non_positive_volume_1d";
        step_result.first_failing_cell = failing_cell;
        step_result.failing_value = failing_vol;
        core::log_warning(
            "Hydro1D: non-positive cell volume at cell=" +
            std::to_string(failing_cell) + " (vol=" +
            std::to_string(failing_vol) +
            "); requesting driver full-step retry");
        return step_result;
      }
      TENRYU_ASSERT(false,
                    "Hydro1D: non-positive cell volume at cell=" +
                        std::to_string(failing_cell) + " (vol=" +
                        std::to_string(failing_vol) +
                        "); enable Numerics.hydro.driver_full_step_retry"
                        " for soft retry");
    }
  }
  // k13 F-05/F-09 (AI kernel review 2026-07-26): end-of-step viscous
  // stability audit. The scalar dt formula bounds only the longitudinal
  // diffusion at the step-start state; the exact Gershgorin row-sum of the
  // assembled operator additionally sees the spherical/cylindrical hoop
  // stiffness (up to ~2.5x at the origin) and the eta ~ T^{5/2} growth over
  // the step (predictor shock heating can raise the corrector stiffness
  // ~5.7x per temperature doubling). At the default dt_safety = 0.3 the
  // committed dt always clears this bound (bit-neutral); a violation means
  // the step ran outside the explicit stability certificate and is retried
  // (soft) or aborted, mirroring the non-positive-volume guard above.
  const auto brag_fmt_sci = [](const double v) {
    char buf[32] = {};
    std::snprintf(buf, sizeof(buf), "%.6e", v);
    return std::string(buf);
  };
  if (brag_params.enabled) {
    const double lambda_g = braginskii::compute_viscous_gershgorin_lambda_1d(
        state.x_r.data(), state.rho.data(),
        state.Ti.empty() ? nullptr : state.Ti.data(),
        state.Te.empty() ? nullptr : state.Te.data(),
        state.A_eff.empty() ? nullptr : state.A_eff.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(), state.mass.data(),
        state.vol.data(), d_hydro_active, d_node_active, n_cells, geom_code,
        brag_skip_outer, brag_params,
        state.zmom_active ? state.zmom_r4.data() : nullptr);
    const double dt_limit = (lambda_g > 0.0)
                                ? (2.0 / lambda_g)
                                : std::numeric_limits<double>::infinity();
    if (!(dt <= dt_limit * (1.0 + 1.0e-12))) {
      if (cfg.numerics.hydro.driver_full_step_retry_enabled) {
        step_result.retry_required = true;
        step_result.reason = "braginskii_viscous_dt_1d";
        step_result.suggested_dt = brag_params.dt_safety * dt_limit;
        core::log_warning(
            "Hydro1D: viscous Gershgorin stability audit failed (dt=" +
            brag_fmt_sci(dt) + " > 2/lambda_G=" + brag_fmt_sci(dt_limit) +
            "); requesting driver full-step retry");
        return step_result;
      }
      TENRYU_ASSERT(false,
                    "Hydro1D: viscous Gershgorin stability audit failed (dt=" +
                        brag_fmt_sci(dt) + " > 2/lambda_G=" +
                        brag_fmt_sci(dt_limit) +
                        "); enable Numerics.hydro.driver_full_step_retry for "
                        "soft retry or reduce plasma_viscosity.dt_safety");
    }
  }
  // Deferred from the final geometry refresh (see above): publish the final
  // node/volume geometry to the host mirror only after the step is known
  // admissible. On the soft-failure path the mirror keeps the last good
  // geometry and the driver restores the snapshot anyway.
  state.mesh.sync_device_geometry_to_host(state.vol.data());
  return step_result;
}

}  // namespace tenryu::hydro
