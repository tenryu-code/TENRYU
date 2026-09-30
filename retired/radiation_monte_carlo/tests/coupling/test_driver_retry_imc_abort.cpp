#include <csignal>
#include <cstdlib>
#include <sys/wait.h>
#include <unistd.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "coupling/driver.hpp"

namespace {

void run_retry_imc_driver() {
  tenryu::core::Config cfg;
  cfg.numerics.hydro.driver_full_step_retry_enabled = true;
  cfg.radiation.mode = tenryu::core::RadiationMode::ImcDdmc;

  tenryu::core::State state;
  tenryu::coupling::Driver driver;
  driver.run(state, cfg);
}

}  // namespace

int main() {
  const pid_t pid = fork();
  if (pid < 0) {
    return EXIT_FAILURE;
  }
  if (pid == 0) {
    run_retry_imc_driver();
    _exit(EXIT_SUCCESS);
  }

  int status = 0;
  if (waitpid(pid, &status, 0) < 0) {
    return EXIT_FAILURE;
  }
  if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) {
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
