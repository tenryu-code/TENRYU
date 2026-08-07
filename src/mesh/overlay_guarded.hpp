#pragma once

#include "core/field_measure.hpp"
#include "core/namelist/errors.hpp"
#include "mesh/overlay_transfer.cuh"

namespace tenryu::mesh::overlay {

// Registry-guarded transfer entry (design F2 enforcement): looks up the field's
// contract, fail-louds when the field is undeclared (startup failure semantics),
// when its transfer kind is not kOverlayIntegrate (kForbidden/kRebuildFromBasis/
// kRecomputeDerived fields must never reach the overlay engine), or when its
// measure is not a volume measure (kPhysicalRZVolume or kPerRadianVolume -- the
// per-radian gather below integrates rho * r dA). On a passing contract,
// forwards to overlay_gather_cell.
inline OverlayCellResult guarded_overlay_gather_cell(
    const tenryu::core::FieldMeasureRegistry& registry,
    const std::string& field_name,
    const double* recv_r, const double* recv_z, int recv_n,
    const double* donor_r, const double* donor_z, const int* donor_n,
    int donor_stride, const double* rho, int n_donors,
    double rtol, double abs_floor) {
  const tenryu::core::FieldContract& contract = registry.require(field_name);

  const auto transfer_kind_name = [](const tenryu::core::TransferKind kind) {
    switch (kind) {
      case tenryu::core::TransferKind::kOverlayIntegrate:
        return "kOverlayIntegrate";
      case tenryu::core::TransferKind::kRebuildFromBasis:
        return "kRebuildFromBasis";
      case tenryu::core::TransferKind::kRecomputeDerived:
        return "kRecomputeDerived";
      case tenryu::core::TransferKind::kForbidden:
        return "kForbidden";
    }
    return "unknown TransferKind";
  };
  if (contract.transfer != tenryu::core::TransferKind::kOverlayIntegrate) {
    throw tenryu::core::namelist::ConfigError(
        "FieldMeasureRegistry contract '" + field_name +
        "' has transfer kind " + transfer_kind_name(contract.transfer) +
        "; overlay transfer requires kOverlayIntegrate");
  }

  const auto field_measure_name = [](const tenryu::core::FieldMeasure measure) {
    switch (measure) {
      case tenryu::core::FieldMeasure::kPlanarArea:
        return "kPlanarArea";
      case tenryu::core::FieldMeasure::kPhysicalRZVolume:
        return "kPhysicalRZVolume";
      case tenryu::core::FieldMeasure::kPhysicalFaceArea:
        return "kPhysicalFaceArea";
      case tenryu::core::FieldMeasure::kPerRadianVolume:
        return "kPerRadianVolume";
      case tenryu::core::FieldMeasure::kNodalKinematic:
        return "kNodalKinematic";
    }
    return "unknown FieldMeasure";
  };
  if (contract.measure != tenryu::core::FieldMeasure::kPhysicalRZVolume &&
      contract.measure != tenryu::core::FieldMeasure::kPerRadianVolume) {
    throw tenryu::core::namelist::ConfigError(
        "FieldMeasureRegistry contract '" + field_name + "' has measure " +
        field_measure_name(contract.measure) +
        "; overlay transfer requires kPhysicalRZVolume or kPerRadianVolume");
  }

  return overlay_gather_cell(
      recv_r, recv_z, recv_n, donor_r, donor_z, donor_n, donor_stride, rho,
      n_donors, rtol, abs_floor);
}

}  // namespace tenryu::mesh::overlay
