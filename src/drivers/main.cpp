#include "drivers/cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "core/version.hpp"

#if TENRYU_ENABLE_MPI
#include <cuda_runtime.h>
#include <mpi.h>
#endif

namespace {

#if TENRYU_ENABLE_MPI
void log_mpi_error(const char* api, const int code) {
  if (code != MPI_SUCCESS) {
    std::cerr << "MPI error: " << api << " returned code " << code << '\n';
  }
}

// 1 MPI rank = 1 GPU (NUMERICS §12): bind each rank to a device by its
// node-local rank before any CUDA context is created. With a single visible
// device (or single rank) this selects device 0, the CUDA default, so
// single-rank behavior is unchanged.
void bind_rank_local_gpu() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    return;
  }
  MPI_Comm shared_comm = MPI_COMM_NULL;
  const int split_err = MPI_Comm_split_type(
      MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shared_comm);
  log_mpi_error("MPI_Comm_split_type", split_err);
  if (split_err != MPI_SUCCESS || shared_comm == MPI_COMM_NULL) {
    return;
  }
  int local_rank = 0;
  const int rank_err = MPI_Comm_rank(shared_comm, &local_rank);
  log_mpi_error("MPI_Comm_rank(shared)", rank_err);
  MPI_Comm_free(&shared_comm);
  if (rank_err != MPI_SUCCESS) {
    return;
  }
  const cudaError_t set_err = cudaSetDevice(local_rank % device_count);
  if (set_err != cudaSuccess) {
    std::cerr << "CUDA error: cudaSetDevice(" << (local_rank % device_count)
              << ") failed: " << cudaGetErrorString(set_err) << '\n';
  }
}

class MpiSession {
 public:
  explicit MpiSession(bool enabled) {
    if (!enabled) {
      return;
    }

    int mpi_initialized = 0;
    const int init_query_err = MPI_Initialized(&mpi_initialized);
    log_mpi_error("MPI_Initialized", init_query_err);
    if (init_query_err != MPI_SUCCESS) {
      return;
    }
    if (!mpi_initialized) {
      int argc = 0;
      char** argv = nullptr;
      const int init_err = MPI_Init(&argc, &argv);
      log_mpi_error("MPI_Init", init_err);
      if (init_err != MPI_SUCCESS) {
        return;
      }
      owns_mpi_ = true;
    }
    bind_rank_local_gpu();
  }

  ~MpiSession() {
    if (!owns_mpi_) {
      return;
    }

    int mpi_finalized = 0;
    const int finalized_query_err = MPI_Finalized(&mpi_finalized);
    log_mpi_error("MPI_Finalized", finalized_query_err);
    if (finalized_query_err != MPI_SUCCESS) {
      return;
    }
    if (!mpi_finalized) {
      const int finalize_err = MPI_Finalize();
      log_mpi_error("MPI_Finalize", finalize_err);
    }
  }

 private:
  bool owns_mpi_ = false;
};
#else
class MpiSession {
 public:
  explicit MpiSession(bool enabled) {
    (void)enabled;
  }
};
#endif

}  // namespace

int main(int argc, char** argv) {
  // TENRYU_LOG_FULLBUF_MB: size (MiB) of a full stdout buffer replacing
  // the libc default (~4 KB). Batches per-step diagnostic log lines so
  // network filesystems (FUSE/Lustre) see large writes instead of a
  // per-step flush storm (measured 10x wall on RunPod). 0 disables.
  // Crash paths flush explicitly (core/error.cpp); normal exit flushes
  // via stdio shutdown. Default 4 MiB.
  {
    double log_buf_mb = 4.0;
    if (const char* s = std::getenv("TENRYU_LOG_FULLBUF_MB")) {
      log_buf_mb = std::atof(s);
    }
    if (log_buf_mb > 0.0) {
      const std::size_t n = static_cast<std::size_t>(log_buf_mb * 1048576.0);
      static std::vector<char> log_buf;
      log_buf.resize(n);
      std::setvbuf(stdout, log_buf.data(), _IOFBF, n);
    }
  }
  CLI::App app{"TENRYU radiation-hydrodynamics simulation code", "tenryu"};
  app.require_subcommand(0, 1);
  app.set_version_flag("--version", tenryu::core::tenryu_version_string(),
                       "Show version and exit");

  tenryu::drivers::CliOptions common_options;
  tenryu::drivers::add_common_cli_options(app, common_options);

  std::string run_input;
  std::string run_restart;
  std::string run_output_dir;
  auto* run_cmd = app.add_subcommand("run", "Run TENRYU simulation");
  run_cmd->add_option("namelist.py", run_input, "Path to input namelist")
      ->required();
  run_cmd->add_option("--restart", run_restart,
                      "Checkpoint prefix (<dir>/<case>_ckpt_NNNN) to restart from");
  run_cmd->add_option("--output-dir", run_output_dir,
                      "Override Output.directory from the deck (fresh runs only)");

  std::string verify_test;
  bool verify_generate_golden = false;
  auto* verify_cmd = app.add_subcommand("verify", "Run verification suites");
  verify_cmd->add_option("test_name", verify_test,
                         "Verification test name or 'all'");
  verify_cmd->add_flag(
      "--generate-golden",
      verify_generate_golden,
      "Generate/update golden references for supported verification tests");

  std::string validate_input;
  bool validate_mesh_preview = false;
  auto* validate_cmd = app.add_subcommand("validate", "Validate namelist input");
  validate_cmd->add_option("namelist.py", validate_input,
                           "Path to input namelist")
      ->required();
  validate_cmd->add_flag("--mesh-preview", validate_mesh_preview,
                         "Emit a one-line TENRYU-MESH-PREVIEW JSON after successful validation");

  std::string freeze_input;
  std::string freeze_output;
  auto* freeze_cmd = app.add_subcommand("freeze", "Freeze namelist to JSON");
  freeze_cmd->add_option("namelist.py", freeze_input, "Path to input namelist")
      ->required();
  freeze_cmd->add_option("-o,--output", freeze_output, "Path to output JSON");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  tenryu::drivers::configure_logging(common_options);

  const bool needs_mpi = run_cmd->parsed() || verify_cmd->parsed();
  MpiSession mpi_session(needs_mpi);

  if (run_cmd->parsed()) {
    return tenryu::drivers::cmd_run(run_input, run_restart, run_output_dir);
  }
  if (verify_cmd->parsed()) {
    if (verify_test.empty()) {
      std::cerr << "TENRYU ERROR [verify]: test_name is required\n";
      std::cerr << verify_cmd->help() << '\n';
      return 2;
    }
    return tenryu::drivers::cmd_verify(verify_test, verify_generate_golden);
  }
  if (validate_cmd->parsed()) {
    return tenryu::drivers::cmd_validate(validate_input, validate_mesh_preview);
  }
  if (freeze_cmd->parsed()) {
    return tenryu::drivers::cmd_freeze(freeze_input, freeze_output);
  }

  std::cout << app.help() << '\n';
  return 0;
}
