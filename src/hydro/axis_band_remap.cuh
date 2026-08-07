#pragma once

#include <cstdint>

namespace tenryu::core {
struct State;
struct Config;
}  // namespace tenryu::core

namespace tenryu::parallel {
class Reduction;
}  // namespace tenryu::parallel

namespace tenryu::hydro::ale {

// Failure modes for the band-only remap. Hard gates trip the rollback in
// W2-D. Conservation residuals for mass, internal energies, radiation
// energy, and band cell momentum are hard gates (AI review k04 R14);
// the node-projection kinetic-energy delta remains diagnostic-only because
// the mass-weighted projection is not KE-conserving by construction.
enum class AxisBandRemapFailure : std::uint8_t {
  None = 0,
  IntermediatePositivityViolation,  // post-R-sweep intermediate volume <= 0
  FinalPositivityViolation,         // post-Z-sweep final volume <= 0
  RemappedMassNegative,             // any cell mass <= 0 after both sweeps
  RemappedInternalEnergyNegative,   // any cell rho*ee or rho*ei <= 0 after both sweeps
  TargetMeshInfeasible,             // R_K <= 0 or dz_j <= 0 at any j
  ScratchAllocationFailed,
  RemappedRadiationNegative,        // any radiation group extensive energy < 0 after a sweep
  ConservationViolation,            // |conservation residual| > 1e-8 hard gate
};

inline const char* axis_band_remap_failure_name(const AxisBandRemapFailure r) {
  switch (r) {
    case AxisBandRemapFailure::None:
      return "none";
    case AxisBandRemapFailure::IntermediatePositivityViolation:
      return "intermediate_positivity";
    case AxisBandRemapFailure::FinalPositivityViolation:
      return "final_positivity";
    case AxisBandRemapFailure::RemappedMassNegative:
      return "remapped_mass_negative";
    case AxisBandRemapFailure::RemappedInternalEnergyNegative:
      return "remapped_internal_energy_negative";
    case AxisBandRemapFailure::TargetMeshInfeasible:
      return "target_mesh_infeasible";
    case AxisBandRemapFailure::ScratchAllocationFailed:
      return "scratch_allocation_failed";
    case AxisBandRemapFailure::RemappedRadiationNegative:
      return "remapped_radiation_negative";
    case AxisBandRemapFailure::ConservationViolation:
      return "conservation_violation";
  }
  return "unknown";
}

struct AxisBandRemapResult {
  bool succeeded = false;
  AxisBandRemapFailure failure = AxisBandRemapFailure::None;
  int K = 0;
  // Conservation diagnostics (relative deltas, signed). Populated whenever
  // the remap completes both sweeps even if positivity later fails.
  double mass_delta_rel = 0.0;
  double E_int_e_delta_rel = 0.0;  // sum (rho*ee)*V over band cells
  double E_int_i_delta_rel = 0.0;
  double E_kin_delta_rel = 0.0;  // node-centric kinetic energy over band cells
  double E_rad_delta_rel = 0.0;  // sum rad_E_g * V over band cells
  // Band cell-momentum deltas across both sweeps, normalized by the physical
  // scale max(|P_r_before|, |P_z_before|, M_band * max|v|, tiny) so that a
  // symmetry-zero total momentum does not inflate the residual (k04 R14).
  double mom_r_delta_scaled = 0.0;
  double mom_z_delta_scaled = 0.0;
  // Min volumes (diagnostic).
  double min_v_intermediate = 0.0;
  double min_v_final = 0.0;
  // Min cell mass / specific internal energy after remap.
  double min_cell_mass = 0.0;
  double min_specific_e = 0.0;  // min over band cells of min(ee, ei)
};

// Band-only swept-volume remap. Source: current State (axis-band slice).
// Target: equal-RZ-volume mesh r_i = R_K(j) sqrt(i/K), z_i = Z_K(j) for
// i in [0, K], j in [0, nz]. Both R-sweep and Z-sweep run; cells in
// [i in [0, K-1], j in [0, nz-1]] are conservatively remapped.
AxisBandRemapResult apply_axis_band_remap(
    core::State& state,
    const core::Config& cfg,
    int K,
    const parallel::Reduction* reduction);

}  // namespace tenryu::hydro::ale
