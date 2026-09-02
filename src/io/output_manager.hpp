#pragma once

#include <string>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::radiation {
struct PhotonPool;
}

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
  void init(const tenryu::core::Config& cfg, int rank = 0);
  void set_termination_reason(std::string reason);
  int last_checkpoint_step() const;
  const std::string& last_checkpoint_path() const;

  void write_run_info(const tenryu::core::State& state,
                      const tenryu::core::Config& cfg) const;
  void write_frozen_config(const std::string& case_name,
                           const std::string& frozen_json) const;
  void write_snapshot(const tenryu::core::State& state,
                      const tenryu::core::Config& cfg,
                      int step,
                      double t,
                      const std::string& case_name,
                      int rank = 0);
  void write_checkpoint(const tenryu::core::State& state,
                        const tenryu::core::Config& cfg,
                        const tenryu::radiation::PhotonPool& photon_pool,
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
