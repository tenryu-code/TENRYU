#pragma once

#include <cstdint>
#include <cstring>
#include <string>

namespace tenryu::mesh {

enum class MeshGeometryFailureKind : std::uint8_t {
  None = 0,
  NonPositiveCellVolume,
  NonPositiveCellArea,
  NonFiniteCellVolume,
  NonFiniteCellArea,
  NonFiniteNodeCoordinate,
  TopologyInvariantViolation,
  NegativeRadialCoordinate,
};

enum class MeshGeometryFailurePolicy : std::uint8_t {
  HardAssert = 0,
  SoftReturn,
};

struct MeshGeometryResult {
  bool admissible = true;
  bool recoverable = false;
  MeshGeometryFailureKind kind = MeshGeometryFailureKind::None;
  int failing_cell = -1;
  int failing_i = -1;
  int failing_j = -1;
  double failing_value = 0.0;
  double min_cell_vol = 0.0;
  double min_cell_area = 0.0;
  const char* reason = "";
  // Axis-failure telemetry fields. Populated only when
  // TENRYU_STAGE24_TELEMETRY=1; otherwise left at defaults.
  bool dt_sensitive = false;
  bool state_sensitive = false;
  double margin_rel = 0.0;
  double dt_star_est = 0.0;
  enum class PreferredRoute : std::uint8_t {
    None = 0,
    ReduceDt,
    RepairAxisBand,
    RepairAxisFace,
    RepairInteriorCd,
    AbortStateSensitive,
  };
  PreferredRoute preferred_route = PreferredRoute::None;
};

struct MeshGeometryCheckOptions {
  MeshGeometryFailurePolicy policy = MeshGeometryFailurePolicy::HardAssert;
  bool check_active_cells_only = false;
  const std::int8_t* d_hydro_active = nullptr;
  const std::uint8_t* inactive_cell_mask = nullptr;
  double volume_floor = 0.0;
  double area_floor = 0.0;
};

inline const char* mesh_geometry_failure_kind_name(
    const MeshGeometryFailureKind kind) {
  switch (kind) {
    case MeshGeometryFailureKind::None:
      return "none";
    case MeshGeometryFailureKind::NonPositiveCellVolume:
      return "non_positive_cell_volume";
    case MeshGeometryFailureKind::NonPositiveCellArea:
      return "non_positive_cell_area";
    case MeshGeometryFailureKind::NonFiniteCellVolume:
      return "non_finite_cell_volume";
    case MeshGeometryFailureKind::NonFiniteCellArea:
      return "non_finite_cell_area";
    case MeshGeometryFailureKind::NonFiniteNodeCoordinate:
      return "non_finite_node_coordinate";
    case MeshGeometryFailureKind::TopologyInvariantViolation:
      return "topology_invariant_violation";
    case MeshGeometryFailureKind::NegativeRadialCoordinate:
      return "negative_radial_coordinate";
  }
  return "unknown";
}

inline const char* mesh_geometry_preferred_route_name(
    const MeshGeometryResult::PreferredRoute route) {
  switch (route) {
    case MeshGeometryResult::PreferredRoute::None:
      return "none";
    case MeshGeometryResult::PreferredRoute::ReduceDt:
      return "reduce_dt";
    case MeshGeometryResult::PreferredRoute::RepairAxisBand:
      return "repair_axis_band";
    case MeshGeometryResult::PreferredRoute::RepairAxisFace:
      return "repair_axis_face";
    case MeshGeometryResult::PreferredRoute::RepairInteriorCd:
      return "repair_interior_cd";
    case MeshGeometryResult::PreferredRoute::AbortStateSensitive:
      return "abort_state_sensitive";
  }
  return "unknown";
}

inline std::string format_legacy_failure_message(
    const MeshGeometryResult& result) {
  const bool reason_area =
      result.reason != nullptr && std::strcmp(result.reason, "cell_area") == 0;
  const bool is_area_failure =
      reason_area ||
      result.kind == MeshGeometryFailureKind::NonPositiveCellArea ||
      result.kind == MeshGeometryFailureKind::NonFiniteCellArea;
  const bool is_2d = result.failing_i >= 0 && result.failing_j >= 0;
  if (is_area_failure) {
    return "Encountered non-positive 2D RZ cell area at cell=" +
           std::to_string(result.failing_cell) + ", area=" +
           std::to_string(result.failing_value);
  }
  if (is_2d) {
    return "Encountered non-positive 2D RZ cell volume at cell=" +
           std::to_string(result.failing_cell) + ", vol=" +
           std::to_string(result.failing_value);
  }
  return "Encountered non-positive 1D cell volume at cell=" +
         std::to_string(result.failing_cell) + ", vol=" +
         std::to_string(result.failing_value);
}

}  // namespace tenryu::mesh
