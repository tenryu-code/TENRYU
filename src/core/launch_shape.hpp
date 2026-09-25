#pragma once

namespace tenryu::core {

// Threads per block for a kernel with one thread per cell whose threads each
// run long serial FP64 work (table EOS inversions and closures). The time of
// such a kernel is set by the per-thread latency and by the FP64 units of the
// SMs its blocks occupy, not by memory traffic: with 256 threads per block a
// 1D deck of a few hundred cells ran on two SMs. Grids up to 32768 cells take
// one warp per block, spread over as many SMs as there are warps; larger grids
// keep 256 threads (at 32768 cells both shapes give the same warps per SM on a
// 128-SM GPU). Every thread computes the same values either way.
inline int serial_cell_block_size(const int n_cells) {
  return (n_cells <= 32768) ? 32 : 256;
}

inline int serial_cell_blocks(const int n_cells) {
  const int block = serial_cell_block_size(n_cells);
  return (n_cells + block - 1) / block;
}

}  // namespace tenryu::core
