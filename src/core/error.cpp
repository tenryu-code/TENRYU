#include "core/error.hpp"

#include <cstdio>
#include <cstdlib>

#include <spdlog/spdlog.h>

#if TENRYU_ENABLE_MPI
#include <mpi.h>
#endif

namespace tenryu::core {

void log_fatal(const std::string& message) {
  spdlog::critical(message);
}

void log_error(const std::string& message) {
  spdlog::error(message);
}

void log_warning(const std::string& message) {
  spdlog::warn(message);
}

void log_info(const std::string& message) {
  spdlog::info(message);
}

void log_debug(const std::string& message) {
  spdlog::debug(message);
}

[[noreturn]] void tenryu_abort(const char* condition,
                               std::string_view message,
                               const char* file,
                               int line) {
  spdlog::critical("TENRYU_ASSERT failed: ({}) at {}:{} - {}", condition, file, line,
                   message);

#if TENRYU_ENABLE_MPI
  int mpi_initialized = 0;
  const int initialized_err = MPI_Initialized(&mpi_initialized);
  if (initialized_err != MPI_SUCCESS) {
    spdlog::critical("MPI_Initialized failed during TENRYU abort path (code={})",
                     initialized_err);
  }
  if (initialized_err == MPI_SUCCESS && mpi_initialized) {
    int mpi_finalized = 0;
    const int finalized_err = MPI_Finalized(&mpi_finalized);
    if (finalized_err != MPI_SUCCESS) {
      spdlog::critical("MPI_Finalized failed during TENRYU abort path (code={})",
                       finalized_err);
    } else if (!mpi_finalized) {
      const int abort_err = MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
      if (abort_err != MPI_SUCCESS) {
        spdlog::critical("MPI_Abort failed during TENRYU abort path (code={})", abort_err);
      }
    }
  }
#endif

  std::fflush(stdout);
  std::fflush(stderr);
  std::abort();
}

}  // namespace tenryu::core
