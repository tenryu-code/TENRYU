#include "core/field_measure.hpp"

#include <cmath>
#include <utility>

#include "core/namelist/errors.hpp"

namespace tenryu::core {

void FieldMeasureRegistry::register_contract(FieldContract c) {
  if (c.name.empty()) {
    throw namelist::ConfigError(
        "FieldMeasureRegistry contract name must not be empty");
  }
  if (!(c.lower_bound <= c.upper_bound)) {
    throw namelist::ConfigError(
        "FieldMeasureRegistry contract '" + c.name +
        "' has lower_bound greater than upper_bound");
  }
  if (c.reconstruction_order != 0 && c.reconstruction_order != 1) {
    throw namelist::ConfigError(
        "FieldMeasureRegistry contract '" + c.name +
        "' has reconstruction_order outside {0, 1}");
  }
  if (index_.contains(c.name)) {
    throw namelist::ConfigError(
        "FieldMeasureRegistry duplicate contract '" + c.name + "'");
  }

  const std::size_t index = contracts_.size();
  index_.emplace(c.name, index);
  contracts_.push_back(std::move(c));
}

bool FieldMeasureRegistry::has(const std::string& name) const {
  return index_.contains(name);
}

const FieldContract& FieldMeasureRegistry::require(
    const std::string& name) const {
  const auto it = index_.find(name);
  if (it == index_.end()) {
    throw namelist::ConfigError(
        "FieldMeasureRegistry missing contract '" + name + "'");
  }
  return contracts_[it->second];
}

std::vector<std::string> FieldMeasureRegistry::audit_missing(
    const std::vector<std::string>& live_fields) const {
  std::vector<std::string> missing;
  for (const std::string& name : live_fields) {
    if (!has(name)) {
      missing.push_back(name);
    }
  }
  return missing;
}

std::size_t FieldMeasureRegistry::size() const {
  return contracts_.size();
}

bool transfer_allowed(const FieldContract& contract) {
  return contract.transfer != TransferKind::kForbidden;
}

FieldMeasureRegistry make_core_field_registry() {
  FieldMeasureRegistry registry;

  registry.register_contract({
      "subcell_mass",
      FieldSupport::kCornerSubcell,
      FieldMeasure::kPhysicalRZVolume,
      ConservationLaw::kExtensiveConserved,
      TransferKind::kOverlayIntegrate,
      0.0,
      HUGE_VAL,
      1,
      EpochDeps{true, false, false},
      "state.hpp corner_mass: physical subcell mass m(cn); NEVER rebuilt from "
      "a basis (design §3); Lagrangian-invariant between events."});
  registry.register_contract({
      "kinematic_node_mass",
      FieldSupport::kNode,
      FieldMeasure::kNodalKinematic,
      ConservationLaw::kNonConserved,
      TransferKind::kRebuildFromBasis,
      0.0,
      HUGE_VAL,
      0,
      EpochDeps{true, false, false},
      "declarative (P0B binding): m^K rebuilt from the kinematic basis at "
      "events; gated by the four proofs (design §3)."});
  registry.register_contract({
      "node_momentum_r",
      FieldSupport::kNode,
      FieldMeasure::kNodalKinematic,
      ConservationLaw::kExtensiveConserved,
      TransferKind::kOverlayIntegrate,
      -HUGE_VAL,
      HUGE_VAL,
      1,
      EpochDeps{true, false, false},
      "declarative (P0B binding): radial node momentum uses "
      "gather-remap-scatter with KE closure (design §3)."});
  registry.register_contract({
      "node_momentum_z",
      FieldSupport::kNode,
      FieldMeasure::kNodalKinematic,
      ConservationLaw::kExtensiveConserved,
      TransferKind::kOverlayIntegrate,
      -HUGE_VAL,
      HUGE_VAL,
      1,
      EpochDeps{true, false, false},
      "declarative (P0B binding): axial node momentum uses "
      "gather-remap-scatter with KE closure (design §3)."});
  registry.register_contract({
      "internal_energy",
      FieldSupport::kMaterialPerCell,
      FieldMeasure::kPhysicalRZVolume,
      ConservationLaw::kExtensiveConserved,
      TransferKind::kOverlayIntegrate,
      0.0,
      HUGE_VAL,
      1,
      EpochDeps{true, false, false},
      "state.hpp Ee_per_material and Ei_per_material: extensive m*e per "
      "material; KE-closure deposits land here."});
  registry.register_contract({
      "radiation_group_energy",
      FieldSupport::kRadiationGroupPerCell,
      FieldMeasure::kPhysicalRZVolume,
      ConservationLaw::kExtensiveConserved,
      TransferKind::kOverlayIntegrate,
      0.0,
      HUGE_VAL,
      1,
      EpochDeps{true, false, false},
      "state.hpp rad_E: radiation group energy."});
  registry.register_contract({
      "passive_tracer",
      FieldSupport::kCellCenter,
      FieldMeasure::kPhysicalRZVolume,
      ConservationLaw::kExtensiveConserved,
      TransferKind::kOverlayIntegrate,
      0.0,
      HUGE_VAL,
      1,
      EpochDeps{true, false, false},
      "state.hpp gas_tracer_Y: passive tracer."});
  registry.register_contract({
      "planar_inertia",
      FieldSupport::kCellCenter,
      FieldMeasure::kPlanarArea,
      ConservationLaw::kIntensive,
      TransferKind::kRecomputeDerived,
      0.0,
      HUGE_VAL,
      0,
      EpochDeps{true, true, false},
      "declarative (P0B binding): mesh-objective weight; the FIX-2 "
      "planar-measure example -- NOT interchangeable with subcell_mass."});
  registry.register_contract({
      "face_flux_area",
      FieldSupport::kFace,
      FieldMeasure::kPhysicalFaceArea,
      ConservationLaw::kNonConserved,
      TransferKind::kRecomputeDerived,
      0.0,
      HUGE_VAL,
      0,
      EpochDeps{true, false, false},
      "declarative (P0B binding): physical face-area flux measure."});
  registry.register_contract({
      "eos_branch_metadata",
      FieldSupport::kMaterialPerCell,
      FieldMeasure::kPerRadianVolume,
      ConservationLaw::kNonConserved,
      TransferKind::kForbidden,
      -HUGE_VAL,
      HUGE_VAL,
      0,
      EpochDeps{true, false, true},
      "declarative (P0B binding): snapshot/restore by byte copy only (Layer-T "
      "§3); transfer across topology change is FORBIDDEN -- must be "
      "re-closed."});
  registry.register_contract({
      "controller_regime_state",
      FieldSupport::kCellCenter,
      FieldMeasure::kPlanarArea,
      ConservationLaw::kNonConserved,
      TransferKind::kForbidden,
      -HUGE_VAL,
      HUGE_VAL,
      0,
      EpochDeps{true, true, false},
      "declarative (P0B binding): explicit persisted controller state "
      "(design §6) -- restart-serialized, never remapped."});

  return registry;
}

}  // namespace tenryu::core
