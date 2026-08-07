#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tenryu::core {

enum class FieldSupport : std::uint8_t {
  kCellCenter,
  kNode,
  kCornerSubcell,
  kFace,
  kMaterialPerCell,
  kRadiationGroupPerCell
};

enum class FieldMeasure : std::uint8_t {
  kPlanarArea,        // dA = dr dz
  kPhysicalRZVolume,  // dV = 2*pi*r dA
  kPhysicalFaceArea,  // dS = 2*pi*r ds
  kPerRadianVolume,   // dV1 = r dA
  kNodalKinematic     // scheme-defined nodal mass measure
};

enum class ConservationLaw : std::uint8_t {
  kExtensiveConserved,  // sum over support must be preserved by transfer
  kIntensive,           // transferred as density against the declared measure
  kNonConserved         // diagnostic / controller state
};

enum class TransferKind : std::uint8_t {
  kOverlayIntegrate,  // exact overlay integration of reconstruction
  kRebuildFromBasis,  // rebuilt from the kinematic basis; never summed
  kRecomputeDerived,  // invalidated and recomputed from primaries
  kForbidden          // must never be transferred; fail if attempted
};

struct EpochDeps {
  // Epoch counters that invalidate this field.
  bool topology = false;
  bool reference = false;
  bool solution = true;
};

struct FieldContract {
  std::string name;  // canonical field name
  FieldSupport support;
  FieldMeasure measure;
  ConservationLaw law;
  TransferKind transfer;
  double lower_bound;        // -inf if unbounded
  double upper_bound;        // +inf if unbounded
  int reconstruction_order;  // 0 or 1; declarative in P0A
  EpochDeps epoch_deps;
  std::string note;  // state.hpp member mapping / rationale
};

class FieldMeasureRegistry {
 public:
  // Duplicate names fail loudly.
  void register_contract(FieldContract c);
  bool has(const std::string& name) const;
  // Missing names fail loudly.
  const FieldContract& require(const std::string& name) const;
  // Returns names without contracts in the order given in live_fields.
  std::vector<std::string> audit_missing(
      const std::vector<std::string>& live_fields) const;
  std::size_t size() const;

 private:
  // Vector storage preserves deterministic registration order.
  std::vector<FieldContract> contracts_;
  std::unordered_map<std::string, std::size_t> index_;
};

bool transfer_allowed(const FieldContract& contract);

// Returns the core seed table.
FieldMeasureRegistry make_core_field_registry();

}  // namespace tenryu::core
