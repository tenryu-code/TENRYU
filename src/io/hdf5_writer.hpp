#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "core/config.hpp"
#include "core/state.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::io {

struct MaterialInterfaceAttributes {
  bool plic_enabled = false;
  int plic_schema_version = 0;
  std::string plic_reconstruction_engine_version;
  std::string plic_normal_estimator;
  std::string t0_volume_cut_method;
  std::string plic_reconstruction_method;
};

enum class MaterialInterfaceReadStatus {
  MissingGroupPlicDisabled,
  Present,
  InconsistentMissingGroupForProductionComparable,
};

MaterialInterfaceReadStatus read_material_interface_status(
    const std::string& h5_file_path);

std::optional<MaterialInterfaceAttributes> read_material_interface_attributes(
    const std::string& h5_file_path);

struct PerMaterialConservationResiduals {
  double mass_max_abs_residual = 0.0;
  double mass_max_rel_residual = 0.0;
  double Ee_max_abs_residual = 0.0;
  double Ee_max_rel_residual = 0.0;
  double Ei_max_abs_residual = 0.0;
  double Ei_max_rel_residual = 0.0;
};

PerMaterialConservationResiduals compute_per_material_conservation_residuals(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg);

std::uint64_t dispatch_counters_regression_hash(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg);

class HDF5Writer {
 public:
  HDF5Writer() = default;

  // Writes the snapshot and returns once it is published (the file renamed
  // from .tmp; any snapshot still pending from write_snapshot_in_background is
  // published first).
  void write_snapshot(const tenryu::core::State& state,
                      const tenryu::core::Config& cfg,
                      int file_index,
                      int step,
                      double t,
                      const std::string& output_dir,
                      const std::string& case_name,
                      int rank = 0) const;

  // The same snapshot, finished by a worker thread: the call creates the file,
  // writes the groups, attributes and small datasets and queues the large
  // datasets' compression (the data is copied before it returns); the worker
  // waits for the compression, writes the chunks, closes the file and
  // publishes it, in the order of the calls, at most two pending. The time
  // loop's snapshots (OutputManager::write_snapshot) take this path.
  void write_snapshot_in_background(const tenryu::core::State& state,
                                    const tenryu::core::Config& cfg,
                                    int file_index,
                                    int step,
                                    double t,
                                    const std::string& output_dir,
                                    const std::string& case_name,
                                    int rank = 0) const;

  std::string write_checkpoint(
      const tenryu::core::State& state,
      const tenryu::core::Config& cfg,
      const tenryu::radiation::PhotonPool& photon_pool,
      int file_index,
      int step,
      double t,
      const std::string& output_dir,
      const std::string& case_name) const;
  // Waits until every snapshot of write_snapshot_in_background so far is
  // published and rethrows the first error of the worker.
  static void wait_for_snapshot_writes();
};

}  // namespace tenryu::io
