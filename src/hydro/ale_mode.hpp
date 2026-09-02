#pragma once

#include <cstdint>

#include "hydro/mesh_regime.hpp"

namespace tenryu::hydro::ale {

enum class AleMode : std::uint8_t {
  ScheduledDefault = 0,
  FullWinslow,
  AxisSpineOnly,
  AxisSpinePlusLocal,
  BoundaryPatchProjection,
  CdLocalWinslow,
  InteriorMultiNodeProjection,
  AxisVariationalProjection,
  FlankTangentialStrip,
};

struct AleRequest {
  AleMode mode = AleMode::ScheduledDefault;
  bool force_rezone = false;
  int focus_cell = -1;
  int patch_radius_i = 2;
  int patch_radius_j = 2;
  MeshFailureRegime source_regime = MeshFailureRegime::Unknown;
};

struct RepairPlan {
  AleMode ale_mode = AleMode::ScheduledDefault;
  int focus_cell = -1;
  int retry_attempt = 0;
  double dt_retry = 0.0;
  bool is_same_dt_strategy_retry = false;
};

}  // namespace tenryu::hydro::ale
