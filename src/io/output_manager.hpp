#pragma once

#include <string>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::io {

class OutputManager {
 public:
  std::string output_dir;
  std::string config_dir;
  std::string checkpoint_dir;
  std::string results_dir;
  std::string log_dir;

  // Under MPI pass the rank: only rank 0 claims/creates the output tree
  // (every writer entry is already rank-0-gated); rank > 0 sets the base
  // path strings without touching the filesystem — this removes the
  // per-rank directory-index race (run_p2 vs run_p2_001).
  OutputManager() = default;
  OutputManager(const OutputManager&) = default;
  OutputManager& operator=(const OutputManager&) = default;
  OutputManager(OutputManager&&) = default;
  OutputManager& operator=(OutputManager&&) = default;
  // Waits for the snapshots still being published (a safety net; the driver
  // waits at the end of a run).
  ~OutputManager();

  void init(const tenryu::core::Config& cfg, int rank = 0);
  // write_snapshot publishes the snapshot on the HDF5 writer's worker thread
  // (HDF5Writer::write_snapshot_in_background): waits until every snapshot
  // written so far is on disk, rethrowing a publication error.
  void wait_for_snapshots() const;
  void set_termination_reason(std::string reason);
  int last_checkpoint_step() const;
  const std::string& last_checkpoint_path() const;

  void write_run_info(const tenryu::core::State& state,
                      const tenryu::core::Config& cfg) const;
  void write_mesh_requirement(const std::string& json) const;
  void write_frozen_config(const std::string& case_name,
                           const std::string& frozen_json) const;
  // Both bring the host copies of the 1D burn arrays up to date first
  // (State::sync_burn_arrays_to_host).
  void write_snapshot(tenryu::core::State& state,
                      const tenryu::core::Config& cfg,
                      int step,
                      double t,
                      const std::string& case_name,
                      int rank = 0);
  void write_checkpoint(tenryu::core::State& state,
                        const tenryu::core::Config& cfg,
                        int step,
                        double t,
                        const std::string& case_name,
                        int rank = 0);

  [[nodiscard]] bool should_plot(int step,
                                 double t,
                                 const tenryu::core::State& state,
                                 const tenryu::core::Config& cfg) const;
  [[nodiscard]] bool should_history(int step,
                                    double t,
                                    const tenryu::core::State& state,
                                    const tenryu::core::Config& cfg) const;
  [[nodiscard]] bool should_checkpoint(int step,
                                       double t,
                                       const tenryu::core::State& state,
                                       const tenryu::core::Config& cfg) const;

 private:
  void rotate_checkpoints(const tenryu::core::Config& cfg,
                          const std::string& case_name,
                          int rank = 0) const;
  std::string termination_reason_ = "running";
  int snapshot_count_ = 0;
  int checkpoint_count_ = 0;
  int last_checkpoint_step_ = -1;
  std::string last_checkpoint_path_;
};

}  // namespace tenryu::io
