#pragma once

#include <string>

#include "core/config.hpp"
#include "core/state.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::io {

enum class PerMaterialCheckpointReadStatus {
  MissingGroupDisabled,
  MissingGroupEnabled,
  PresentDisabled,
  PresentEnabled,
  PresentEnabledIncomplete,
};

struct CheckpointData {
  tenryu::core::State state;
  tenryu::radiation::PhotonPool photon_pool;
  PerMaterialCheckpointReadStatus per_material_checkpoint_status =
      PerMaterialCheckpointReadStatus::MissingGroupEnabled;
};

PerMaterialCheckpointReadStatus read_per_material_checkpoint_status(
    const std::string& h5_file_path,
    bool per_material_enabled);

PerMaterialCheckpointReadStatus read_per_material_checkpoint_status(
    const std::string& h5_file_path);

class HDF5Reader {
 public:
  HDF5Reader() = default;

  CheckpointData read_checkpoint(const tenryu::core::Config& cfg,
                                 const std::string& checkpoint_prefix,
                                 int rank = 0) const;
};

}  // namespace tenryu::io
