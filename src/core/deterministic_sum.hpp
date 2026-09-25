#pragma once

#include <cuda_runtime.h>

namespace tenryu::core {

// Sum of d_values[0, n) in a fixed order, written to (or, with accumulate,
// added to) *d_out on the device: one block, each thread sums the indices
// t, t + B, t + 2B, ... in index order, then a fixed-order block tree. The
// result is bitwise identical from run to run (unlike an atomicAdd
// accumulation, whose order follows the thread schedule). For the energy
// ledgers of the 1D operators: every kernel writes its contribution to a
// per-cell slot and the ledger is this sum (2026-09-24).
void deterministic_sum(const double* d_values, int n, double* d_out, bool accumulate,
                       cudaStream_t stream = nullptr);

}  // namespace tenryu::core
