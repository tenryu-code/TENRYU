#pragma once

#include <cstdint>

#include "core/config.hpp"
#include "core/state.hpp"
#include "mesh/candidate_mesh_admissibility.hpp"

namespace tenryu::hydro {

struct ReferenceBarrierAleResult {
  bool engaged = false;
  bool succeeded = false;
  double lambda_accepted = 0.0;
  int linesearch_iters = 0;
  tenryu::mesh::CandidateMeshQuality final_quality{};
  std::uint64_t trigger_reason_mask = 0;
  double mass_floor_delta = 0.0;
  double E_floor_injected = 0.0;
  double E_redistribution_unresolved = 0.0;

  enum class TriggerReason : std::uint64_t {
    AxisMargin = 1ULL << 0,
    CornerJRatio = 1ULL << 1,
  };
};

struct ReferenceBarrierScope {
  const std::uint8_t* d_active_cell_mask = nullptr;   // 1 = participates
  const std::uint8_t* d_frozen_velocity_node_mask = nullptr;  // 1 = frozen
};

std::uint64_t evaluate_reference_barrier_trigger(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg);

void build_reference_target_mesh(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    double* d_target_r,
    double* d_target_z);

ReferenceBarrierAleResult apply_reference_barrier_rezone(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_target_r,
    const double* d_target_z,
    const ReferenceBarrierScope* scope = nullptr);

}  // namespace tenryu::hydro
