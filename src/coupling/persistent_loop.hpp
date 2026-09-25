#pragma once

#include <vector>

#include "core/namelist/frozen_table_device.cuh"
#include "diagnostics/energy_budget.hpp"
#include "materials/ionmix_reader.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu { namespace core { struct State; struct Config; } }
namespace tenryu { namespace laser { struct LaserMesh; } }

namespace tenryu::coupling {

// One step of a chunk, for the driver's per-step checks and history rows:
// the step energy budget (diagnostics::compute_step_energy_budget of the
// step's energy totals and ledgers, as the multi-kernel step builds it) and
// the step's floor clamp count.
struct PersistentStepSummary {
  int step = 0;  // state.step after the step
  double t_after = 0.0;
  double dt = 0.0;
  int clamp_count = 0;
  diagnostics::EnergyBudget budget{};
};

struct PersistentChunkResult {
  int steps_advanced = 0;
  double t_after = 0.0;
  double dt_after = 0.0;
  int exit_reason = 0;
  int error_code = 0;
  std::vector<PersistentStepSummary> steps;
};

bool persistent_loop_supported_c1(const core::State& state,
                                  const core::Config& cfg,
                                  const laser::LaserMesh* laser_mesh = nullptr);

void prepare_persistent_laser_entry(core::State& state,
                                    const core::Config& cfg,
                                    laser::LaserMesh& laser_mesh);

PersistentChunkResult run_persistent_chunk(core::State& state,
                                           const core::Config& cfg,
                                           laser::LaserMesh& laser_mesh,
                                           radiation::PlanckTableDeviceView planck,
                                           materials::IonmixOpacityDeviceView
                                               nlte_opacity,
                                           core::namelist::FrozenTable1DDeviceView
                                               laser_waveform,
                                           double t_end,
                                           double t_next_output,
                                           int max_steps_remaining);

}  // namespace tenryu::coupling
