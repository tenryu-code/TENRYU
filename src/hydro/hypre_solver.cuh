#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::hydro {

struct HypreSolver {
  [[noreturn]] static void fail_stub() {
    TENRYU_ASSERT(
        false,
        "Conduction solver='hypre' selected, but HypreSolver is a stub implementation");
  }

  bool initialized = false;
  int n_local_cells = 0;
  int stencil_width = 0;

  void init(const int n_local_cells_in, const int stencil_width_in) {
    (void)n_local_cells_in;
    (void)stencil_width_in;
    fail_stub();
  }

  void update_matrix(const double* stencil_9pt,
                     const double* rho_Cv,
                     const double dt,
                     const int n_cells,
                     cudaStream_t stream) {
    (void)stencil_9pt;
    (void)rho_Cv;
    (void)dt;
    (void)n_cells;
    (void)stream;
    fail_stub();
  }

  int solve(double* Te_out, const double* Te_in, cudaStream_t stream) {
    (void)Te_out;
    (void)Te_in;
    (void)stream;
    fail_stub();
    return -1;
  }

  void destroy() {
    initialized = false;
    n_local_cells = 0;
    stencil_width = 0;
  }
};

}  // namespace tenryu::hydro
