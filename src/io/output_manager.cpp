#include "io/output_manager.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#if TENRYU_ENABLE_MPI
#include <mpi.h>
#endif
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "core/error.hpp"
#include "io/hdf5_writer.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::io {
namespace {

double time_epsilon(const double t, const double interval) {
  return 1.0e-14 * std::max(std::abs(t), interval);
}

bool by_time_interval(const double interval,
                      const double t_next,
                      const double t) {
  if (interval <= 0.0 || t_next < 0.0) {
    return false;
  }
  const double eps = time_epsilon(t, interval);
  return t >= (t_next - eps);
}

std::string escape_json_string(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (const char ch : input) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

std::string solver_requested(const tenryu::core::Config& cfg) {
  const auto& fld = cfg.radiation.multigroup_diffusion;
  return fld.linear_solver_2d_requested.empty() ? fld.linear_solver_2d
                                                : fld.linear_solver_2d_requested;
}

std::string solver_resolved(const tenryu::core::Config& cfg) {
  const auto& fld = cfg.radiation.multigroup_diffusion;
  return fld.linear_solver_2d_resolved.empty() ? fld.linear_solver_2d
                                               : fld.linear_solver_2d_resolved;
}

}  // namespace

void OutputManager::init(const tenryu::core::Config& cfg, const int rank) {
  TENRYU_ASSERT(!cfg.output.directory.empty(), "Output.directory must not be empty");
  if (rank > 0) {
    // §6q blocker 1: rank 0 resolves the (possibly sequence-numbered)
    // output dir and BROADCASTS it — deriving a base path locally gave
    // every non-zero rank a DIFFERENT directory view than rank 0
    // (the namelist stage pre-creates the base dir, so rank 0 lands on
    // "_001" while the others hold the unsuffixed base), and any
    // non-rank-0-gated writer then split-brained the run (SQUID P>=4
    // H5Fcreate abort). No mkdir on this rank.
    std::string resolved;
#if TENRYU_ENABLE_MPI
    int mpi_up = 0;
    MPI_Initialized(&mpi_up);
    if (mpi_up != 0) {
      int len = 0;
      MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
      resolved.resize(static_cast<std::size_t>(len));
      if (len > 0) {
        MPI_Bcast(resolved.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
      }
    }
#endif
    if (resolved.empty()) {
      std::filesystem::path rp(cfg.output.directory);
      if (rp.filename().empty()) {
        rp = rp.parent_path();
      }
      resolved = rp.string();
    }
    const std::filesystem::path rp(resolved);
    output_dir = rp.string();
    config_dir = (rp / "config").string();
    checkpoint_dir = (rp / "checkpoints").string();
    results_dir = (rp / "results").string();
    log_dir = (rp / "log").string();
    return;
  }
  std::filesystem::path base_path(cfg.output.directory);
  if (base_path.filename().empty()) {
    base_path = base_path.parent_path();
  }
  const std::filesystem::path parent_dir = base_path.parent_path();
  const std::string base_stem = base_path.stem().string();
  TENRYU_ASSERT(!base_stem.empty(), "Output.directory must include a directory name");

  int max_output_index = -1;
  const std::filesystem::path scan_dir =
      parent_dir.empty() ? std::filesystem::path(".") : parent_dir;
  if (std::filesystem::exists(scan_dir)) {
    const std::string prefix = base_stem + "_";
    for (const auto& entry : std::filesystem::directory_iterator(scan_dir)) {
      if (!entry.is_directory()) {
        continue;
      }
      const std::string dirname = entry.path().filename().string();
      if (dirname.size() != prefix.size() + 3) {
        continue;
      }
      if (dirname.rfind(prefix, 0) != 0) {
        continue;
      }
      int index = 0;
      bool valid = true;
      for (std::size_t i = prefix.size(); i < dirname.size(); ++i) {
        const char ch = dirname[i];
        if (ch < '0' || ch > '9') {
          valid = false;
          break;
        }
        index = index * 10 + (ch - '0');
      }
      if (!valid) {
        continue;
      }
      max_output_index = std::max(max_output_index, index);
    }
  }

  if (!std::filesystem::exists(base_path)) {
    output_dir = base_path.string();
  } else {
    const int next_output_index = std::max(max_output_index, 0) + 1;
    TENRYU_ASSERT(next_output_index <= 999, "Output directory index exceeds 999");
    std::ostringstream numbered_dirname;
    numbered_dirname << base_stem << '_' << std::setw(3) << std::setfill('0')
                     << next_output_index;
    const std::filesystem::path numbered_output_dir =
        parent_dir / numbered_dirname.str();
    output_dir = numbered_output_dir.string();
  }
  if (std::getenv("TENRYU_IO_DEBUG") != nullptr) {
    std::fprintf(stderr, "[io-debug] init rank=%d pid=%d output_dir=%s\n",
                 rank, static_cast<int>(getpid()), output_dir.c_str());
  }
  std::filesystem::create_directories(output_dir);
#if TENRYU_ENABLE_MPI
  {
    int mpi_up = 0;
    MPI_Initialized(&mpi_up);
    if (mpi_up != 0) {
      int n_ranks_bcast = 1;
      MPI_Comm_size(MPI_COMM_WORLD, &n_ranks_bcast);
      if (n_ranks_bcast > 1) {
        int len = static_cast<int>(output_dir.size());
        MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(output_dir.data(), len, MPI_CHAR, 0, MPI_COMM_WORLD);
      }
    }
  }
#endif
  config_dir = (std::filesystem::path(output_dir) / "config").string();
  checkpoint_dir = (std::filesystem::path(output_dir) / "checkpoints").string();
  results_dir = (std::filesystem::path(output_dir) / "results").string();
  log_dir = (std::filesystem::path(output_dir) / "log").string();
  std::filesystem::create_directories(config_dir);
  std::filesystem::create_directories(checkpoint_dir);
  std::filesystem::create_directories(results_dir);
  std::filesystem::create_directories(log_dir);

  int max_snapshot_index = -1;
  int max_checkpoint_index = -1;
  auto scan_h5_indices = [&](const std::string& dir_path,
                             const bool checkpoints) {
    if (!std::filesystem::exists(dir_path)) {
      return;
    }
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
      if (!entry.is_regular_file()) {
        continue;
      }
      const std::filesystem::path path = entry.path();
      if (path.extension() != ".h5") {
        continue;
      }
      const std::string filename = path.filename().string();
      const bool is_checkpoint = filename.find("_ckpt_") != std::string::npos;
      if (checkpoints != is_checkpoint) {
        continue;
      }
      const std::size_t suffix_pos = filename.rfind(".h5");
      if (suffix_pos == std::string::npos) {
        continue;
      }
      const std::size_t underscore_pos = filename.rfind('_', suffix_pos);
      if (underscore_pos == std::string::npos) {
        continue;
      }
      if (suffix_pos != underscore_pos + 5) {
        continue;
      }
      const std::string index_str = filename.substr(underscore_pos + 1, 4);
      int index = 0;
      bool valid = true;
      for (const char ch : index_str) {
        if (ch < '0' || ch > '9') {
          valid = false;
          break;
        }
        index = index * 10 + (ch - '0');
      }
      if (!valid) {
        continue;
      }
      if (checkpoints) {
        max_checkpoint_index = std::max(max_checkpoint_index, index);
      } else {
        max_snapshot_index = std::max(max_snapshot_index, index);
      }
    }
  };

  scan_h5_indices(results_dir, false);
  scan_h5_indices(checkpoint_dir, true);
  snapshot_count_ = max_snapshot_index + 1;
  checkpoint_count_ = max_checkpoint_index + 1;
}

void OutputManager::set_termination_reason(std::string reason) {
  termination_reason_ = std::move(reason);
}

void OutputManager::write_run_info(const tenryu::core::State& state,
                                   const tenryu::core::Config& cfg) const {
  const std::filesystem::path out_path =
      std::filesystem::path(output_dir) / "run_info.json";
  std::ofstream ofs(out_path, std::ios::binary | std::ios::trunc);
  TENRYU_ASSERT(ofs.good(), "Failed to open run_info.json for writing");

  ofs << std::setprecision(17);
  ofs << "{\n";
  ofs << "  \"step\": " << state.step << ",\n";
  ofs << "  \"t\": " << state.t << ",\n";
  ofs << "  \"dt\": " << state.dt << ",\n";
  ofs << "  \"termination_reason\": \""
      << escape_json_string(termination_reason_) << "\",\n";
  ofs << "  \"solver_requested\": \""
      << escape_json_string(solver_requested(cfg)) << "\",\n";
  ofs << "  \"solver_resolved\": \""
      << escape_json_string(solver_resolved(cfg)) << "\",\n";
  ofs << "  \"cg_cap_exit_unconverged\": "
      << state.cg_cap_exit_unconverged << ",\n";
  ofs << "  \"newton_cap_exit_unconverged\": "
      << state.newton_cap_exit_unconverged << "\n";
  ofs << "}\n";
  TENRYU_ASSERT(ofs.good(), "Failed while writing run_info.json");
  ofs.flush();
  TENRYU_ASSERT(ofs.good(), "Failed to flush run_info.json");
  ofs.close();
  TENRYU_ASSERT(!ofs.fail(), "Failed to close run_info.json");
}

void OutputManager::write_frozen_config(const std::string& case_name,
                                        const std::string& frozen_json) const {
  const std::filesystem::path out_path =
      std::filesystem::path(config_dir) / (case_name + "_frozen.json");
  std::ofstream ofs(out_path, std::ios::binary | std::ios::trunc);
  TENRYU_ASSERT(ofs.good(), "Failed to open frozen config JSON for writing");

  ofs << frozen_json;
  if (!frozen_json.empty() && frozen_json.back() != '\n') {
    ofs << '\n';
  }
  TENRYU_ASSERT(ofs.good(), "Failed while writing frozen config JSON");
  ofs.flush();
  TENRYU_ASSERT(ofs.good(), "Failed to flush frozen config JSON");
  ofs.close();
  TENRYU_ASSERT(!ofs.fail(), "Failed to close frozen config JSON");
}

void OutputManager::write_snapshot(const tenryu::core::State& state,
                                   const tenryu::core::Config& cfg,
                                   const int step,
                                   const double t,
                                   const std::string& case_name,
                                   const int rank) {
  if (cfg.output.format != "hdf5") {
    return;
  }
  const int file_index = snapshot_count_++;
  HDF5Writer writer;
  writer.write_snapshot(state, cfg, file_index, step, t, results_dir, case_name, rank);
}

void OutputManager::write_checkpoint(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const tenryu::radiation::PhotonPool& photon_pool,
    const int step,
    const double t,
    const std::string& case_name,
    const int rank) {
  if (cfg.output.format != "hdf5") {
    return;
  }
  const int file_index = checkpoint_count_++;
  HDF5Writer writer;
  writer.write_checkpoint(state, cfg, photon_pool, file_index, step, t, checkpoint_dir,
                          case_name);
  rotate_checkpoints(cfg, case_name, rank);
}

void OutputManager::rotate_checkpoints(const tenryu::core::Config& cfg,
                                       const std::string& case_name,
                                       const int rank) const {
  if (rank != 0) {
    return;
  }
  if (cfg.output.checkpoint_keep_last <= 0) {
    return;
  }

  std::vector<std::filesystem::path> checkpoints;
  const std::string prefix = case_name + "_ckpt_";
  for (const auto& entry : std::filesystem::directory_iterator(checkpoint_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string filename = entry.path().filename().string();
    if (filename.rfind(prefix, 0) != 0) {
      continue;
    }
    if (entry.path().extension() != ".h5") {
      continue;
    }
    checkpoints.push_back(entry.path());
  }

  if (static_cast<int>(checkpoints.size()) <= cfg.output.checkpoint_keep_last) {
    return;
  }

  std::sort(checkpoints.begin(), checkpoints.end());
  const int remove_count =
      static_cast<int>(checkpoints.size()) - cfg.output.checkpoint_keep_last;
  for (int i = 0; i < remove_count; ++i) {
    std::error_code ec;
    std::filesystem::remove(checkpoints[static_cast<std::size_t>(i)], ec);
    if (ec) {
      core::log_warning("Failed to remove old checkpoint '" +
                        checkpoints[static_cast<std::size_t>(i)].string() +
                        "': " + ec.message());
    }
  }
}

bool OutputManager::should_plot(const int step,
                                const double t,
                                const tenryu::core::State& state,
                                const tenryu::core::Config& cfg) const {
  const bool by_step = (cfg.output.plot_every > 0) && (step % cfg.output.plot_every == 0);
  const bool by_time =
      by_time_interval(cfg.output.plot_every_s, state.t_next_plot, t);
  return by_step || by_time;
}

bool OutputManager::should_history(const int step,
                                   const double t,
                                   const tenryu::core::State& state,
                                   const tenryu::core::Config& cfg) const {
  const bool by_step =
      (cfg.output.history_every > 0) && (step % cfg.output.history_every == 0);
  const bool by_time =
      by_time_interval(cfg.output.history_every_s, state.t_next_history, t);
  return by_step || by_time;
}

bool OutputManager::should_checkpoint(const int step,
                                      const double t,
                                      const tenryu::core::State& state,
                                      const tenryu::core::Config& cfg) const {
  const bool by_step =
      (cfg.output.checkpoint_every > 0) &&
      (step % cfg.output.checkpoint_every == 0);
  const bool by_time = by_time_interval(cfg.output.checkpoint_every_s,
                                        state.t_next_checkpoint, t);
  return by_step || by_time;
}

}  // namespace tenryu::io
