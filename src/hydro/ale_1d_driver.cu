#include "hydro/ale_1d_driver.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "hydro/ale_1d_diagnostics.cuh"
#include "hydro/ale_1d_remap.cuh"
#include "hydro/ale_1d_rezone.cuh"
#include "hydro/ale_1d_sensor.cuh"
#include "hydro/ale_1d_velocity_project.cuh"
#include "hydro/boundary.hpp"
#include "hydro/eos_context.hpp"
#include "hydro/hydro_1d_bodies.cuh"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro::ale1d {
namespace {

constexpr int kBlockSize = 256;
constexpr double kFourPiOverThree =
    4.188790204786390984616857844372670512262892532500141094646;
constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

int blocks_for(const int n) {
  return (n + kBlockSize - 1) / kBlockSize;
}

bool closure_dump_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("TENRYU_ALE1D_CLOSURE_DUMP");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
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

__host__ __device__ double volume_coordinate(const double r, const int geom) {
  return (geom == 0) ? (kFourPiOverThree * r * r * r)
                     : tenryu::mesh::geometry_1d_shell_volume_cubes(geom, 0.0, r);
}

bool uses_particle_radiation_mode(const core::Config& cfg) {
  return cfg.radiation.mode == core::RadiationMode::ImcDdmc ||
         cfg.radiation.imc.enabled ||
         cfg.radiation.ddmc.enabled ||
         cfg.radiation.imc.difference.enabled ||
         cfg.radiation.holo.enabled;
}

int effective_cell_count(const core::State& state, const core::Config& cfg) {
  if (state.mesh.topo.n_cells > 0) {
    return state.mesh.topo.n_cells;
  }
  return cfg.mesh.nr;
}

bool outer_boundary_is_fixed(const core::Config& cfg) {
  return cfg.numerics.hydro.boundary_1d == "fixed" ||
         cfg.numerics.hydro.boundary_1d == "reflect";
}

double current_max_dr_ratio(const core::State& state, const int n_cells) {
  if (n_cells <= 1 || state.x_r.size() < static_cast<std::size_t>(n_cells + 1)) {
    return 1.0;
  }
  std::vector<double> r(static_cast<std::size_t>(n_cells + 1), 0.0);
  state.x_r.copy_to_host(r.data());
  double ratio = 1.0;
  for (int i = 0; i + 1 < n_cells; ++i) {
    const double a = r[static_cast<std::size_t>(i + 1)] -
                     r[static_cast<std::size_t>(i)];
    const double b = r[static_cast<std::size_t>(i + 2)] -
                     r[static_cast<std::size_t>(i + 1)];
    if (!(a > 0.0) || !(b > 0.0)) {
      return std::numeric_limits<double>::infinity();
    }
    ratio = std::max(ratio, std::max(a / b, b / a));
  }
  return ratio;
}

void enforce_boundary_candidate(std::vector<double>& r_candidate,
                                NodeConstraintMask& node_mask,
                                const core::Config& cfg) {
  if (r_candidate.empty()) {
    return;
  }
  r_candidate.front() = cfg.mesh.r_min;
  if (outer_boundary_is_fixed(cfg)) {
    r_candidate.back() = cfg.mesh.r_max;
  }
  if (node_mask.pinned.size() == r_candidate.size()) {
    node_mask.pinned.front() = true;
    if (outer_boundary_is_fixed(cfg)) {
      node_mask.pinned.back() = true;
    }
    node_mask.n_protected_nodes =
        static_cast<int>(std::count(node_mask.pinned.begin(),
                                    node_mask.pinned.end(),
                                    true));
  }
}

bool validate_candidate_geometry(const std::vector<double>& r_candidate,
                                 const int n_cells,
                                 const int geom) {
  if (r_candidate.size() != static_cast<std::size_t>(n_cells + 1)) {
    return false;
  }
  for (int i = 0; i < n_cells; ++i) {
    const double r0 = r_candidate[static_cast<std::size_t>(i)];
    const double r1 = r_candidate[static_cast<std::size_t>(i + 1)];
    if (!std::isfinite(r0) || !std::isfinite(r1) || !(r1 > r0)) {
      return false;
    }
    const double dx = r1 - r0;
    const double vol = volume_coordinate(r1, geom) - volume_coordinate(r0, geom);
    if (!(dx > 0.0) || !(vol > 0.0) || !std::isfinite(vol)) {
      return false;
    }
  }
  return true;
}

std::vector<int> protected_faces_from_features(
    const std::vector<Ale1dFeature>& features,
    const int n_cells) {
  std::vector<int> faces;
  faces.reserve(features.size());
  for (const Ale1dFeature& feature : features) {
    if (feature.peak_cell_or_face < 0) {
      continue;
    }
    const int face = std::clamp(feature.peak_cell_or_face, 0, n_cells);
    if (std::find(faces.begin(), faces.end(), face) == faces.end()) {
      faces.push_back(face);
    }
  }
  return faces;
}

template <typename T>
void copy_device_array_to_state_field(core::Field1D<T>& dst,
                                      const DeviceArray<double>& src,
                                      const char* label) {
  TENRYU_ASSERT(dst.size() == src.size(), label);
  if (src.empty()) {
    return;
  }
  cuda_check(cudaMemcpy(dst.data(), src.data(), src.size() * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             label);
}

__global__ void recompute_density_kernel(double* __restrict__ rho,
                                         const double* __restrict__ mass,
                                         const double* __restrict__ vol,
                                         const int n_cells) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  rho[i] = (vol[i] > 0.0) ? mass[i] / vol[i] : 0.0;
}

// Post-remap per-species table reclosure honoring the energy-authoritative closure policy
// (Numerics.hydro.eos_closure_mode). Under "energy_authoritative" the evolved
// energy is kept on table-edge inverse clamps (writeback veto) and
// super-ceiling targets invert into the high-T ideal tail; "legacy" keeps the
// historic always-project behavior bit-for-bit. Nonfinite/negative raw inputs
// bypass the veto (repair), mirroring the hydro closure in
// hydro_1d_bodies.cuh. T-floor injections accumulate into de_floor_specific.
__device__ inline void reclose_species_table_1d(
    const tenryu::materials::DeviceEOSTableView& tab,
    const double rho_c,
    const double e_raw,
    const double t_floor,
    const double zbar,
    const double A,
    const bool low_density_extrapolation,
    const bool energy_authoritative,
    double* e_inout,
    double* T_out,
    double* P_out,
    double* cv_out,
    double* de_floor_specific) {
  const double e_in = *e_inout;
  tenryu::materials::DeviceEOSInverseRecloseResult inv;
  if (tenryu::materials::eos_use_low_density_extrapolation(
          tab, low_density_extrapolation, rho_c)) {
    inv = tenryu::materials::device_inverse_reclose_with_low_density_extrap(
        tab, rho_c, e_in, t_floor, zbar, A, low_density_extrapolation);
  } else if (energy_authoritative) {
    inv = tenryu::materials::device_inverse_reclose_with_high_t_tail(
        tab, rho_c, e_in, t_floor);
  } else {
    inv = tenryu::materials::device_inverse_reclose(tab, rho_c, e_in, t_floor);
  }
  *T_out = inv.T;
  *P_out = inv.pressure;
  *cv_out = inv.cv;
  const bool repair = !isfinite(e_raw) || e_raw < 0.0;
  const bool clamp_veto = energy_authoritative && !repair &&
                          inv.bracket_failure == 0 &&
                          (inv.lower_clamp != 0 || inv.upper_clamp != 0);
  if (!clamp_veto) {
    if (inv.T <= fmax(t_floor, 1.0e-30) && inv.energy > e_in) {
      *de_floor_specific += inv.energy - e_in;
    }
    *e_inout = inv.energy;
  }
}

__global__ void eos_reclosure_kernel(double* __restrict__ Te,
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
                                     const int n_cells,
                                     const double gamma,
                                     const double A,
                                     const double rho_floor,
                                     const double te_floor,
                                     const double ti_floor,
                                     const tenryu::materials::DeviceEOSTableView tab_ion,
                                     const tenryu::materials::DeviceEOSTableView tab_ele,
                                     const bool low_density_extrapolation,
                                     const bool energy_authoritative,
                                     double* __restrict__ E_floor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double z = fmax(zbar[c], 0.0);
  const double cv_i = kEvToErg / (A * kProtonMass * (gamma - 1.0));
  const double cv_e = z * kEvToErg / (A * kProtonMass * (gamma - 1.0));
  const double rho_c = fmax(rho[c], rho_floor);
  double e_i = ei[c];
  double e_e = ee[c];

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
  double de_floor_specific = 0.0;

  if (use_ion_table) {
    reclose_species_table_1d(tab_ion, rho_c, ei[c], ti_floor, 1.0, A,
                             low_density_extrapolation, energy_authoritative,
                             &e_i, &Ti_c, &Pi_c, &cv_i_c, &de_floor_specific);
  } else if (cv_i > 0.0) {
    e_i = fmax(e_i, 0.0);
    const double Ti_old = e_i / cv_i;
    Ti_c = fmax(Ti_old, ti_floor);
    if (Ti_c > Ti_old) {
      de_floor_specific += cv_i * (Ti_c - Ti_old);
    }
    e_i = cv_i * Ti_c;
    Pi_c = (gamma - 1.0) * rho_c * e_i;
  } else {
    e_i = 0.0;
    cv_i_c = 0.0;
  }

  if (use_ele_table) {
    reclose_species_table_1d(tab_ele, rho_c, ee[c], te_floor, z, A,
                             low_density_extrapolation, energy_authoritative,
                             &e_e, &Te_c, &Pe_c, &cv_e_c, &de_floor_specific);
  } else if (cv_e > 0.0) {
    e_e = fmax(e_e, 0.0);
    const double Te_old = e_e / cv_e;
    Te_c = fmax(Te_old, te_floor);
    if (Te_c > Te_old) {
      de_floor_specific += cv_e * (Te_c - Te_old);
    }
    e_e = cv_e * Te_c;
    Pe_c = (gamma - 1.0) * rho_c * e_e;
  } else {
    e_e = 0.0;
    cv_e_c = 0.0;
  }

  if (E_floor != nullptr && de_floor_specific > 0.0) {
    atomic_add_double(E_floor, de_floor_specific * fmax(mass[c], 0.0));
  }

  Ti[c] = Ti_c;
  Te[c] = Te_c;
  ei[c] = e_i;
  ee[c] = e_e;
  Pe[c] = Pe_c;
  Pi[c] = Pi_c;
  if (cv_i_out != nullptr) {
    cv_i_out[c] = cv_i_c;
  }
  if (cv_e_out != nullptr) {
    cv_e_out[c] = cv_e_c;
  }
}

__global__ void compute_sound_speed_kernel(double* __restrict__ cs,
                                           const double* __restrict__ rho,
                                           const double* __restrict__ ee,
                                           const double* __restrict__ ei,
                                           const double* __restrict__ Pe,
                                           const double* __restrict__ Pi,
                                           const double* __restrict__ Te,
                                           const double* __restrict__ Ti,
                                           const tenryu::materials::DeviceEOSTableView tab_ion,
                                           const tenryu::materials::DeviceEOSTableView tab_ele,
                                           const double* __restrict__ cv_i_arr,
                                           const double* __restrict__ cv_e_arr,
                                           const int n_cells,
                                           const bool use_table_eos,
                                           const double* __restrict__ gamma_eff,
                                           const double gamma) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  if (use_table_eos) {
    persistent_1d::compute_sound_speed_2t_kernel_body(
        i, cs, rho, ee, ei, Pe, Pi, Te, Ti, tab_ion, tab_ele,
        tenryu::materials::EOSRhoEDeviceView{},
        tenryu::materials::HelmholtzSplineDeviceView{},
        tenryu::materials::HelmholtzJetDeviceView{},
        tenryu::materials::MieGruneisenDeviceView{}, cv_i_arr, cv_e_arr,
        n_cells, false, false, false, false, false,
        persistent_1d::kExactOverrideNone, gamma_eff);
    return;
  }
  const double rho_i = fmax(rho[i], 1.0e-300);
  cs[i] = sqrt(fmax(gamma * (Pe[i] + Pi[i]) / rho_i, 0.0));
}

struct Ale1dDriverScratch {
  Ale1dRemapScratch remap;
  Ale1dVelocityProjectScratch velocity;
  // 1-slot device accumulator for the reclosure floor ledger (drained into
  // State::E_floor_injected after every commit).
  DeviceArray<double> e_floor_ledger;
  int n_cells = -1;
  int n_groups = -1;
  int n_materials = -1;

  void ensure_size(const int n,
                   const int groups,
                   const int materials,
                   const bool ke_conservation_closure) {
    if (e_floor_ledger.size() != 1U) {
      e_floor_ledger.resize(1U);
    }
    if (n == n_cells && groups == n_groups && materials == n_materials &&
        remap.size_matches(n, groups, materials, ke_conservation_closure) &&
        velocity.size_matches(n)) {
      return;
    }
    remap.resize(n, groups, materials, ke_conservation_closure);
    velocity.resize(n);
    n_cells = n;
    n_groups = groups;
    n_materials = materials;
  }
};

Ale1dDriverScratch& driver_scratch() {
  static std::unique_ptr<Ale1dDriverScratch> scratch;
  if (!scratch) {
    scratch = std::make_unique<Ale1dDriverScratch>();
  }
  return *scratch;
}

void commit_scratch(core::State& state,
                    const core::Config& cfg,
                    const std::vector<double>& r_candidate,
                    const Ale1dRemapScratch& remap,
                    const Ale1dVelocityProjectScratch& velocity,
                    const HydroEOSContext* eos_ctx) {
  const int n_cells = effective_cell_count(state, cfg);
  const int n_groups = std::max(0, cfg.radiation.groups);
  const int n_materials = static_cast<int>(cfg.materials.materials.size());
  const int blocks = blocks_for(n_cells);

  TENRYU_ASSERT(state.x_r.size() == r_candidate.size(),
                "ALE1D commit x_r size mismatch");
  state.x_r.copy_from_host(r_candidate);
  copy_device_array_to_state_field(state.mass, remap.mass_new,
                                   "ALE1D commit mass size mismatch");
  copy_device_array_to_state_field(state.ee, remap.ee_new,
                                   "ALE1D commit ee size mismatch");
  copy_device_array_to_state_field(state.ei, remap.ei_new,
                                   "ALE1D commit ei size mismatch");
  if (n_groups > 0) {
    copy_device_array_to_state_field(state.rad_E, remap.rad_E_new,
                                     "ALE1D commit rad_E size mismatch");
  }
  if (n_materials > 0) {
    copy_device_array_to_state_field(state.volFrac, remap.volFrac_new,
                                     "ALE1D commit volFrac size mismatch");
  }
  copy_device_array_to_state_field(state.vol, remap.vol_new,
                                   "ALE1D commit vol size mismatch");
  copy_device_array_to_state_field(state.v_r, velocity.v_new_node,
                                   "ALE1D commit v_r size mismatch");
  recompute_density_kernel<<<blocks, kBlockSize>>>(state.rho.data(),
                                                   state.mass.data(),
                                                   state.vol.data(),
                                                   n_cells);
  cuda_check(cudaGetLastError(), "ALE1D commit density kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "ALE1D commit density kernel failed");

  state.ale_last_applied_step = state.step;
  apply_boundary_1d(state, cfg);
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;
  recompute_density_kernel<<<blocks, kBlockSize>>>(state.rho.data(),
                                                   state.mass.data(),
                                                   state.vol.data(),
                                                   n_cells);
  cuda_check(cudaGetLastError(),
             "ALE1D post-geometry density kernel launch failed");

  // ALE1D post-remap EOS closure is intentionally restricted to material 0.
  const auto& mat = cfg.materials.materials.front();
  tenryu::materials::DeviceEOSTableView ion_eos{};
  tenryu::materials::DeviceEOSTableView ele_eos{};
  if (eos_ctx != nullptr && eos_ctx->n_materials > 0) {
    ion_eos = eos_ctx->ion_view(0);
    ele_eos = eos_ctx->electron_view(0);
  }
  const bool energy_authoritative =
      (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
  DeviceArray<double>& e_floor_ledger = driver_scratch().e_floor_ledger;
  if (e_floor_ledger.size() != 1U) {
    e_floor_ledger.resize(1U);
  }
  cuda_check(cudaMemset(e_floor_ledger.data(), 0, sizeof(double)),
             "ALE1D floor ledger memset failed");
  eos_reclosure_kernel<<<blocks, kBlockSize>>>(
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
      n_cells,
      mat.ideal_gas_gamma,
      mat.A,
      cfg.numerics.floors.rho,
      cfg.numerics.floors.Te,
      cfg.numerics.floors.Ti,
      ion_eos,
      ele_eos,
      cfg.materials.low_density_extrapolation,
      energy_authoritative,
      e_floor_ledger.data());
  cuda_check(cudaGetLastError(), "ALE1D EOS reclosure kernel launch failed");

  if (state.cs.size() != static_cast<std::size_t>(n_cells)) {
    state.cs.reset(static_cast<std::size_t>(n_cells));
  }
  const bool use_table_eos =
      mat.eos_model != "ideal_gas" && eos_ctx != nullptr &&
      eos_ctx->material_supports_rho_e_reclosure(0);
  compute_sound_speed_kernel<<<blocks, kBlockSize>>>(state.cs.data(),
                                                     state.rho.data(),
                                                     state.ee.data(),
                                                     state.ei.data(),
                                                     state.Pe.data(),
                                                     state.Pi.data(),
                                                     state.Te.data(),
                                                     state.Ti.data(),
                                                     ion_eos,
                                                     ele_eos,
                                                     state.cv_i.empty()
                                                         ? nullptr
                                                         : state.cv_i.data(),
                                                     state.cv_e.empty()
                                                         ? nullptr
                                                         : state.cv_e.data(),
                                                     n_cells,
                                                     use_table_eos,
                                                     state.gamma_eff.data(),
                                                     mat.ideal_gas_gamma);
  cuda_check(cudaGetLastError(), "ALE1D sound speed kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "ALE1D post-commit kernels failed");

  std::vector<double> e_floor_host(1, 0.0);
  e_floor_ledger.copy_to_host(e_floor_host);
  if (e_floor_host[0] > 0.0) {
    state.E_floor_injected += e_floor_host[0];
  }

  state.holo_ale_invalidated = true;
  state.ale_rezoned = true;
  state.particle_sort_cache_invalidated = true;
  state.rad_dep.fill(0.0);
  state.rad_emit.fill(0.0);
  state.holo_rad_dep.fill(0.0);
  state.holo_rad_emit.fill(0.0);
  state.Qvisc.fill(0.0);
}

}  // namespace

const char* to_string(const Ale1dSkipReason r) {
  switch (r) {
    case Ale1dSkipReason::None:
      return "None";
    case Ale1dSkipReason::Disabled:
      return "Disabled";
    case Ale1dSkipReason::WrongGeometry:
      return "WrongGeometry";
    case Ale1dSkipReason::ParticleModeUnsupported:
      return "ParticleModeUnsupported";
    case Ale1dSkipReason::NTooSmall:
      return "NTooSmall";
    case Ale1dSkipReason::ProtectedFractionTooHigh:
      return "ProtectedFractionTooHigh";
    case Ale1dSkipReason::MovableSegmentTooSmall:
      return "MovableSegmentTooSmall";
    case Ale1dSkipReason::BenefitTooSmall:
      return "BenefitTooSmall";
    case Ale1dSkipReason::CandidateInvalid:
      return "CandidateInvalid";
    case Ale1dSkipReason::ConservationRejected:
      return "ConservationRejected";
    case Ale1dSkipReason::DtPenaltyTooLarge:
      return "DtPenaltyTooLarge";
  }
  return "Unknown";
}

namespace {

Ale1dStepResult apply_ale_1d_attempt(core::State& state,
                                     const core::Config& cfg,
                                     const HydroEOSContext* eos_ctx) {
  Ale1dStepResult out;
  const auto& ale = cfg.numerics.ale1d;

  if (!ale.enabled) {
    out.skip_reason = Ale1dSkipReason::Disabled;
    return out;
  }
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "ALE1D requires at least one material");
  const auto& mat0 = cfg.materials.materials.front();
  if (mat0.eos_model != "ideal_gas") {
    TENRYU_ASSERT(
        eos_ctx != nullptr && eos_ctx->n_materials > 0 &&
            eos_ctx->material_supports_rho_e_reclosure(0),
        "ALE1D table EOS requires a material-0 HydroEOSContext backend with "
        "rho-e reclosure support");
  }
  if (cfg.main.dimension != "1D_SPH" || cfg.main.dim != 1 || state.mesh.dim != 1) {
    out.skip_reason = Ale1dSkipReason::WrongGeometry;
    return out;
  }
  if (uses_particle_radiation_mode(cfg)) {
    out.skip_reason = Ale1dSkipReason::ParticleModeUnsupported;
    return out;
  }

  const int n_cells = effective_cell_count(state, cfg);
  if (n_cells < ale.min_cells) {
    out.skip_reason = Ale1dSkipReason::NTooSmall;
    return out;
  }

  out.max_dr_ratio = current_max_dr_ratio(state, n_cells);
  out.cadence_triggered =
      state.step > 0 && ale.every_n_steps > 0 &&
      (state.step % ale.every_n_steps) == 0;
  const double emergency_threshold = std::max(1.0, ale.candidate_dt_penalty_max);
  out.quality_triggered =
      ale.emergency_enabled && out.max_dr_ratio > emergency_threshold;
  std::vector<double> floor_r_nodes;
  if (ale.min_width_floor.enabled) {
    if (state.ale1d_floor_cooldown_remaining > 0) {
      // Cooling down after a rejected floor-triggered attempt: skip the
      // floor-trigger evaluation itself. Cadence/quality triggers are
      // unaffected; any applied rezone resets the cooldown (see wrapper).
      --state.ale1d_floor_cooldown_remaining;
    } else {
      floor_r_nodes.assign(static_cast<std::size_t>(n_cells + 1), 0.0);
      state.x_r.copy_to_host(floor_r_nodes.data());
      double min_dl = std::numeric_limits<double>::infinity();
      for (int i = 0; i < n_cells; ++i) {
        min_dl = std::min(
            min_dl,
            floor_r_nodes[static_cast<std::size_t>(i + 1)] -
                floor_r_nodes[static_cast<std::size_t>(i)]);
      }
      out.floor_triggered = min_dl < ale.min_width_floor.floor_cm;
    }
  }
  if (!out.cadence_triggered && !out.quality_triggered &&
      !out.floor_triggered) {
    return out;
  }
  if (state.ale_last_applied_step >= 0 &&
      state.step < state.ale_last_applied_step + ale.min_steps_between_ale) {
    out.skip_reason = Ale1dSkipReason::BenefitTooSmall;
    return out;
  }

  std::vector<Ale1dFeature> features = compute_features(state, cfg, state.dt);
  Ale1dRezoneResult rezone_result;
  if (out.floor_triggered) {
    std::vector<bool> pinned(static_cast<std::size_t>(n_cells + 1), false);
    pinned.front() = true;
    pinned.back() = true;
    for (const Ale1dFeature& feature : features) {
      const int face = feature.peak_cell_or_face;
      if (feature.pinned_face && face >= 0 && face <= n_cells) {
        pinned[static_cast<std::size_t>(face)] = true;
      }
    }
    std::vector<double> rho_host(static_cast<std::size_t>(n_cells), 0.0);
    state.rho.copy_to_host(rho_host.data());
    std::vector<bool> eligible(static_cast<std::size_t>(n_cells), false);
    // Never remap density-floored corona cells: near-zero extensive fields fail remap validators, and relief there is physically meaningless.
    for (int i = 0; i < n_cells; ++i) {
      eligible[static_cast<std::size_t>(i)] =
          rho_host[static_cast<std::size_t>(i)] >
          100.0 * cfg.numerics.floors.rho;
    }
    MinWidthFloorCandidateResult floor_result =
        build_min_width_floor_candidate(
            floor_r_nodes,
            pinned,
            eligible,
            ale.min_width_floor.floor_cm,
            ale.min_width_floor.target_factor,
            ale.min_width_floor.relief_halfwidth_cells,
            ale.min_width_floor.max_growth_factor);
    if (!floor_result.success) {
      out.skip_reason = Ale1dSkipReason::CandidateInvalid;
      return out;
    }
    if (floor_result.no_relief_available) {
      out.skip_reason = Ale1dSkipReason::BenefitTooSmall;
      return out;
    }
    rezone_result.r_candidate = std::move(floor_result.r_candidate);
    rezone_result.node_mask.pinned = std::move(pinned);
    rezone_result.node_mask.n_protected_nodes = static_cast<int>(std::count(
        rezone_result.node_mask.pinned.begin(),
        rezone_result.node_mask.pinned.end(), true));
    rezone_result.success = true;
  } else {
    rezone_result = rezone(state, cfg, features);
  }
  if (!rezone_result.success) {
    out.skip_reason = rezone_result.skip_reason;
    return out;
  }
  enforce_boundary_candidate(rezone_result.r_candidate, rezone_result.node_mask, cfg);
  out.n_protected_nodes = rezone_result.node_mask.n_protected_nodes;
  if (!validate_candidate_geometry(rezone_result.r_candidate,
                                   n_cells,
                                   state.mesh.geometry_code)) {
    out.skip_reason = Ale1dSkipReason::CandidateInvalid;
    return out;
  }

  const int n_groups = std::max(0, cfg.radiation.groups);
  const int n_materials = static_cast<int>(cfg.materials.materials.size());
  TENRYU_ASSERT(n_materials > 0, "ALE1D requires at least one material");
  const bool ke_conservation_closure =
      cfg.numerics.ale1d.ke_conservation_closure;
  Ale1dDriverScratch& scratch = driver_scratch();
  scratch.ensure_size(
      n_cells, n_groups, n_materials, ke_conservation_closure);

  const std::vector<int> protected_faces =
      protected_faces_from_features(features, n_cells);
  const Ale1dRemapResult remap_result =
      remap_v3(state,
               cfg,
               rezone_result.r_candidate,
               rezone_result.node_mask,
               protected_faces,
               scratch.remap);
  out.mass_conservation_rel_err = remap_result.mass_conservation_rel_err;
  out.radiation_conservation_rel_err =
      remap_result.radiation_conservation_rel_err;
  if (!remap_result.success) {
    out.remap_rejected = true;
    out.skip_reason = remap_result.skip_reason == Ale1dSkipReason::CandidateInvalid
                          ? Ale1dSkipReason::CandidateInvalid
                          : Ale1dSkipReason::ConservationRejected;
    return out;
  }

  std::vector<double> mass_new;
  std::vector<double> delta_y;
  std::vector<double> phi_face;
  std::vector<int> donor;
  scratch.remap.mass_new.copy_to_host(mass_new);
  scratch.remap.delta_Y.copy_to_host(delta_y);
  scratch.remap.phi_face.copy_to_host(phi_face);
  scratch.remap.donor.copy_to_host(donor);
  const Ale1dVelocityProjectResult velocity_result =
      project_velocity(state,
                       mass_new,
                       delta_y,
                       donor,
                       phi_face,
                       ke_conservation_closure,
                       cfg.main.two_temperature,
                       ke_conservation_closure
                           ? scratch.remap.ke_remap.data()
                           : nullptr,
                       ke_conservation_closure
                           ? scratch.remap.ee_new.data()
                           : nullptr,
                       ke_conservation_closure
                           ? scratch.remap.ei_new.data()
                           : nullptr,
                       scratch.velocity);
  out.kinetic_energy_drift_rel = velocity_result.kinetic_energy_drift_rel;
  if (!velocity_result.success) {
    out.skip_reason = Ale1dSkipReason::ConservationRejected;
    return out;
  }

  const Ale1dDiagnosticsResult diagnostics =
      compute_diagnostics(state, cfg, scratch.remap, remap_result, velocity_result);
  out.mass_conservation_rel_err = diagnostics.mass_conservation_rel_err;
  out.energy_conservation_rel_err = diagnostics.global_total_energy_rel_err;
  out.radiation_conservation_rel_err =
      diagnostics.radiation_conservation_rel_err;
  out.kinetic_energy_drift_rel = diagnostics.kinetic_energy_drift_rel;
  if (!diagnostics.hard_tolerance_passed) {
    static int closure_dump_reject_count = 0;
    if (closure_dump_enabled() && closure_dump_reject_count < 2) {
      const int invocation = ++closure_dump_reject_count;
      std::ostringstream message;
      message << std::scientific << std::setprecision(6)
              << "[ale1d-ke-closure-reject] invocation=" << invocation
              << " KE_old=" << velocity_result.kinetic_energy_old
              << " KE_new=" << velocity_result.kinetic_energy_new
              << " deposited_total="
              << velocity_result.ke_closure_deposited;
      core::log_info(message.str());
    }
    out.remap_rejected = true;
    out.skip_reason = Ale1dSkipReason::ConservationRejected;
    return out;
  }

  commit_scratch(state,
                 cfg,
                 rezone_result.r_candidate,
                 scratch.remap,
                 scratch.velocity,
                 eos_ctx);
  out.applied = true;
  out.skip_reason = Ale1dSkipReason::None;
  return out;
}

}  // namespace

Ale1dStepResult apply_ale_1d(core::State& state,
                              const core::Config& cfg,
                              const HydroEOSContext* eos_ctx) {
  Ale1dStepResult out = apply_ale_1d_attempt(state, cfg, eos_ctx);
  const int cooldown_steps =
      cfg.numerics.ale1d.min_width_floor.retrigger_cooldown_steps;
  if (out.applied) {
    state.ale1d_floor_cooldown_remaining = 0;
  } else if (out.floor_triggered && cooldown_steps > 0) {
    state.ale1d_floor_cooldown_remaining = cooldown_steps;
  }
  return out;
}

}  // namespace tenryu::hydro::ale1d
