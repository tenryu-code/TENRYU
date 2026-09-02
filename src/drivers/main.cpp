#include "drivers/cli.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "core/version.hpp"
#include "drivers/cmd_checkpoint_swap_center.hpp"

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

std::optional<int> parse_optional_int(const std::string& text,
                                      const char* flag) {
  if (text.empty()) {
    return std::nullopt;
  }
  std::size_t consumed = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(flag) +
                                " requires an integer value");
  }
  if (consumed != text.size() ||
      value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string(flag) +
                                " requires an in-range integer value");
  }
  return static_cast<int>(value);
}

std::optional<double> parse_optional_double(const std::string& text,
                                            const char* flag) {
  if (text.empty()) {
    return std::nullopt;
  }
  std::size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception&) {
    throw std::invalid_argument(std::string(flag) +
                                " requires a floating-point value");
  }
  if (consumed != text.size() || !std::isfinite(value)) {
    throw std::invalid_argument(std::string(flag) +
                                " requires a finite floating-point value");
  }
  return value;
}

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

  tenryu::drivers::CheckpointSwapCenterOptions swap_options;
  std::string swap_cut_ring;
  std::string swap_r_c;
  std::string swap_core_cells;
  std::string swap_bridge_layers;
  std::string swap_max_speed;
  std::string swap_max_mach;
  std::string swap_max_rho_relative;
  std::string swap_max_temperature_relative;
  std::string swap_seam_displacement_floor_cm;
  std::string swap_mirror_z_floor_cm;
  std::string swap_mirror_v_floor_cm_s;
  std::string swap_mirror_tolerance;
  std::string swap_legendre_tolerance;
  std::string swap_ledger_kappa;
  std::string swap_conservation_v_floor_cm_s;
  auto* swap_cmd = app.add_subcommand(
      "checkpoint-swap-center",
      "Replace a quiescent polar-tier center and write a segment-2 checkpoint");
  swap_cmd->add_option("namelist.py", swap_options.namelist_path,
                       "Segment-2 namelist (configuration authority)")
      ->required();
  swap_cmd->add_option("checkpoint-a", swap_options.input_checkpoint,
                       "Parent polar-tier checkpoint prefix or .h5 file")
      ->required();
  swap_cmd->add_option("-o,--output", swap_options.output_checkpoint,
                       "Output checkpoint B .h5 path")
      ->required();
  swap_cmd->add_option("--cut-ring", swap_cut_ring,
                       "Assert the deck's polar-tier cut ring");
  swap_cmd->add_option("--r-c", swap_r_c,
                       "Assert the deck's Cartesian-core half-width [cm]");
  swap_cmd->add_option("--core-cells", swap_core_cells,
                       "Assert the derived Cartesian half-core cell count");
  swap_cmd->add_option("--bridge-layers", swap_bridge_layers,
                       "Assert the deck's bridge layer count");
  swap_cmd->add_option("--max-speed", swap_max_speed,
                       "Maximum replaced-region speed [cm/s]");
  swap_cmd->add_option("--max-mach", swap_max_mach,
                       "Maximum replaced-region Mach number");
  swap_cmd->add_option("--max-rho-rel", swap_max_rho_relative,
                       "Maximum replaced-region relative density departure");
  swap_cmd->add_option(
      "--max-temperature-rel",
      swap_max_temperature_relative,
      "Maximum replaced-region relative Te/Ti departure");
  swap_cmd->add_option(
      "--seam-displacement-floor-cm",
      swap_seam_displacement_floor_cm,
      "Maximum seam displacement from construction coordinates [cm]");
  swap_cmd->add_option("--mirror-z-floor-cm", swap_mirror_z_floor_cm,
                       "Absolute floor for mirror z comparisons [cm]");
  swap_cmd->add_option("--mirror-v-floor-cm-s", swap_mirror_v_floor_cm_s,
                       "Absolute floor for mirror velocity comparisons [cm/s]");
  swap_cmd->add_option("--mirror-tol", swap_mirror_tolerance,
                       "North/south mirror relative tolerance");
  swap_cmd->add_option("--legendre-tol", swap_legendre_tolerance,
                       "Maximum normalized Legendre moment");
  swap_cmd->add_option("--ledger-kappa", swap_ledger_kappa,
                       "Multiplier on gamma_d conservation bounds");
  swap_cmd->add_option(
      "--conservation-v-floor-cm-s",
      swap_conservation_v_floor_cm_s,
      "Velocity floor for momentum conservation bounds [cm/s]");
  swap_cmd->add_flag("--dry-run", swap_options.dry_run,
                     "Run construction and all gates without writing checkpoint B");
  swap_cmd->add_flag("--identity-noop", swap_options.identity_noop,
                     "Test-only byte-preserving hybrid-to-identical-hybrid copy");

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
  if (swap_cmd->parsed()) {
    try {
      swap_options.assert_cut_ring =
          parse_optional_int(swap_cut_ring, "--cut-ring");
      swap_options.assert_r_c_cm =
          parse_optional_double(swap_r_c, "--r-c");
      swap_options.assert_core_cells_per_half =
          parse_optional_int(swap_core_cells, "--core-cells");
      swap_options.assert_bridge_layers =
          parse_optional_int(swap_bridge_layers, "--bridge-layers");
      const auto apply_double = [](const std::string& text,
                                   const char* flag,
                                   double& destination) {
        if (const auto value = parse_optional_double(text, flag)) {
          destination = *value;
        }
      };
      apply_double(swap_max_speed,
                   "--max-speed",
                   swap_options.thresholds.max_speed_cm_s);
      apply_double(swap_max_mach,
                   "--max-mach",
                   swap_options.thresholds.max_mach);
      apply_double(swap_max_rho_relative,
                   "--max-rho-rel",
                   swap_options.thresholds.max_rho_relative);
      apply_double(swap_max_temperature_relative,
                   "--max-temperature-rel",
                   swap_options.thresholds.max_temperature_relative);
      apply_double(swap_seam_displacement_floor_cm,
                   "--seam-displacement-floor-cm",
                   swap_options.thresholds.seam_displacement_floor_cm);
      apply_double(swap_mirror_z_floor_cm,
                   "--mirror-z-floor-cm",
                   swap_options.thresholds.mirror_z_floor_cm);
      apply_double(swap_mirror_v_floor_cm_s,
                   "--mirror-v-floor-cm-s",
                   swap_options.thresholds.mirror_v_floor_cm_s);
      apply_double(swap_mirror_tolerance,
                   "--mirror-tol",
                   swap_options.thresholds.mirror_tolerance);
      apply_double(swap_legendre_tolerance,
                   "--legendre-tol",
                   swap_options.thresholds.legendre_tolerance);
      apply_double(swap_ledger_kappa,
                   "--ledger-kappa",
                   swap_options.thresholds.ledger_kappa);
      apply_double(swap_conservation_v_floor_cm_s,
                   "--conservation-v-floor-cm-s",
                   swap_options.thresholds.conservation_v_floor_cm_s);
    } catch (const std::exception& error) {
      std::cerr << "TENRYU ERROR [checkpoint-swap-center]: " << error.what()
                << '\n';
      return 2;
    }
    return tenryu::drivers::cmd_checkpoint_swap_center(swap_options);
  }

  std::cout << app.help() << '\n';
  return 0;
}
