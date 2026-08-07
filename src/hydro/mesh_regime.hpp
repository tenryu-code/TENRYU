#pragma once

#include <cstdint>

namespace tenryu::hydro {

enum class MeshTopoTag : std::uint8_t {
  Interior = 0,
  AxisFace,
  AxisBand,
  RadialOuterBoundary,
  ZBottomBoundary,
  ZTopBoundary,
  CornerBoundary,
  Inactive,
};

enum class MeshFailureRegime : std::uint8_t {
  Unknown = 0,
  AxisFace,
  AxisBand,
  DomainBoundary,
  InteriorCD,
  InteriorSmooth,
  VoidOrInactive,
};

struct CellRegime {
  std::uint8_t topo_tag = static_cast<std::uint8_t>(MeshTopoTag::Interior);
  float cd_score = 0.0F;
  float compression_score = 0.0F;
  float boundary_score = 0.0F;
  float trigger_scale_threshold = 0.5F;
  std::uint8_t primary_regime =
      static_cast<std::uint8_t>(MeshFailureRegime::Unknown);
};

}  // namespace tenryu::hydro
