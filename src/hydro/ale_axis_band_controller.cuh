#pragma once

#include <cstdint>

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::parallel {
struct CommBuffers;
struct PartitionInfo;
class Reduction;
}  // namespace tenryu::parallel

namespace tenryu::hydro {
struct HydroEOSContext;
}

namespace tenryu::hydro::ale {

struct AxisBandDecision {
  bool needed = false;
  int recommended_K = 3;
  double row_K_margin = 0.0;
  double margin_trigger = 0.0;
};

struct AxisBandResult {
  bool applied = false;
  bool remap_succeeded = false;
  int K_used = 0;
  double min_axis_volume_rel_after = 0.0;
  double mass_delta_rel = 0.0;
  double E_int_e_delta_rel = 0.0;
  double E_int_i_delta_rel = 0.0;
  double E_kin_delta_rel = 0.0;
  double E_rad_delta_rel = 0.0;

  enum class SkipReason : std::uint8_t {
    Disabled,
    NotNeeded,
    AllKFailed,
    RemapFailedPositivity,
    NoOp
  };
  SkipReason skip_reason = SkipReason::Disabled;
};

AxisBandDecision evaluate_axis_band_need(
    const core::State& state,
    const core::Config& cfg,
    const parallel::Reduction* reduction);

AxisBandResult apply_axis_band_managed_remap(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const HydroEOSContext* eos_ctx,
    int K_initial = 3);

}  // namespace tenryu::hydro::ale
