#pragma once

#include "core/namelist/frozen_table_device.cuh"
#include "materials/ionmix_reader.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu { namespace core { struct State; struct Config; } }
namespace tenryu { namespace laser { struct LaserMesh; } }

namespace tenryu::coupling {

struct PersistentChunkResult {
  int steps_advanced = 0;
  double t_after = 0.0;
  double dt_after = 0.0;
  int exit_reason = 0;
  int error_code = 0;
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
