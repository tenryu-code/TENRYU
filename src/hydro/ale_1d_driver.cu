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
#include "hydro/hydro_1d.hpp"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro::ale1d {
namespace {

constexpr int kBlockSize = 256;
constexpr double kFourPiOverThree =
    4.188790204786390984616857844372670512262892532500141094646;

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

struct Ale1dDriverScratch {
  Ale1dRemapScratch remap;
  Ale1dVelocityProjectScratch velocity;
  int n_cells = -1;
  int n_groups = -1;
  int n_materials = -1;

  void ensure_size(const int n,
                   const int groups,
                   const int materials,
                   const bool ke_conservation_closure) {
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

  // The remap moved volume fractions between cells: rebuild the per-cell
  // material properties, then close the EOS and the sound speed with the
  // hydro's closure (1T or 2T, every EOS backend), the state the next
  // Lagrangian step's entry closure would produce. The ALE used its own
  // per-species reclosure, which closed 1T runs with the 2T formulas
  // (T_i at the floor with the floor energy put into e_i, T_e from the
  // electron heat capacity alone, c_v,e without the ion part; 2026-09-23).
  state.invalidate_cell_material_props();
  Hydro1D{}.close_eos_and_sound_speed(state, cfg, eos_ctx);

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
    case Ale1dSkipReason::TooSoon:
      return "TooSoon";
  }
  return "Unknown";
}

Ale1dAcousticDtBounds acoustic_dt_bounds(const std::vector<double>& r_current,
                                         const std::vector<double>& cs_current,
                                         const std::vector<double>& r_candidate) {
  Ale1dAcousticDtBounds bounds;
  const std::size_t n = cs_current.size();
  if (n == 0 || r_current.size() != n + 1 || r_candidate.size() != n + 1) {
    return bounds;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const double dr = r_current[i + 1] - r_current[i];
    if (cs_current[i] > 0.0 && dr > 0.0) {
      bounds.current = std::min(bounds.current, dr / cs_current[i]);
    }
  }
  // Both meshes are ordered: the current cells a candidate cell [a, b]
  // overlaps start at the first k with r_{k+1} > a.
  std::size_t k = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double a = r_candidate[i];
    const double b = r_candidate[i + 1];
    if (!(b > a)) {
      continue;
    }
    while (k + 1 < n && r_current[k + 1] <= a) {
      ++k;
    }
    double c_max = 0.0;
    for (std::size_t m = k; m < n && r_current[m] < b; ++m) {
      c_max = std::max(c_max, cs_current[m]);
    }
    if (c_max > 0.0) {
      bounds.candidate = std::min(bounds.candidate, (b - a) / c_max);
    }
  }
  return bounds;
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
  if (cfg.main.dimension != "1D_SPH" || cfg.main.dim != 1 || state.mesh.dim != 1) {
    out.skip_reason = Ale1dSkipReason::WrongGeometry;
    return out;
  }
  if (uses_particle_radiation_mode(cfg)) {
    out.skip_reason = Ale1dSkipReason::ParticleModeUnsupported;
    return out;
  }
  // Preconditions of an attempt, checked after the skip conditions (a 2D or
  // particle-radiation configuration skips before them; test_ale_1d_skeleton
  // aborted here with a material-less 2D configuration).
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "ALE1D requires at least one material");
  const auto& mat0 = cfg.materials.materials.front();
  if (mat0.eos_model != "ideal_gas") {
    // The commit closes the EOS with the hydro's closure, which evaluates the
    // material's tables from the context.
    TENRYU_ASSERT(eos_ctx != nullptr && eos_ctx->n_materials > 0,
                  "ALE1D with a table EOS requires the HydroEOSContext");
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
  out.quality_triggered =
      ale.emergency_enabled && out.max_dr_ratio > ale.emergency_max_dr_ratio;
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
    out.skip_reason = Ale1dSkipReason::TooSoon;
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

  // Candidate gates on the acoustic time-step bound (NUMERICS §3.4.1): no
  // candidate may lower it by more than candidate_dt_penalty_max, and an
  // attempt triggered by the cadence alone must raise it by
  // benefit_min_dt_gain when the benefit gate is on (the mesh-quality and
  // min-width-floor triggers respond to a mesh defect and are exempt). The
  // parameters were read but never evaluated (2026-09-23).
  {
    std::vector<double> r_current(static_cast<std::size_t>(n_cells + 1), 0.0);
    state.x_r.copy_to_host(r_current.data());
    std::vector<double> cs_current;
    if (state.cs.size() == static_cast<std::size_t>(n_cells)) {
      cs_current.assign(static_cast<std::size_t>(n_cells), 0.0);
      state.cs.copy_to_host(cs_current.data());
    }
    const Ale1dAcousticDtBounds bounds =
        acoustic_dt_bounds(r_current, cs_current, rezone_result.r_candidate);
    const bool bounded =
        std::isfinite(bounds.current) && std::isfinite(bounds.candidate) &&
        bounds.current > 0.0 && bounds.candidate > 0.0;
    out.candidate_dt_gain = bounded ? bounds.candidate / bounds.current : 1.0;
    if (bounded && bounds.current / bounds.candidate > ale.candidate_dt_penalty_max) {
      out.skip_reason = Ale1dSkipReason::DtPenaltyTooLarge;
      return out;
    }
    const bool rescue_trigger = out.quality_triggered || out.floor_triggered;
    if (ale.enable_benefit_gate && !rescue_trigger &&
        out.candidate_dt_gain < ale.benefit_min_dt_gain) {
      out.skip_reason = Ale1dSkipReason::BenefitTooSmall;
      return out;
    }
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

  const Ale1dVelocityProjectResult velocity_result =
      project_velocity(state,
                       scratch.remap.mass_new.data(),
                       scratch.remap.mass_flux.data(),
                       scratch.remap.phi_face.data(),
                       cfg.numerics.ale1d.remap.limiter_theta,
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
