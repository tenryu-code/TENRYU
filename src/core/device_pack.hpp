#pragma once

#include <cstddef>

namespace tenryu::core {

// Gather up to 8 equal-length device double arrays into one staged device
// buffer and pull them to host with a single blocking cudaMemcpy.
// h_dst must have room for k*n doubles; layout: array i occupies [i*n, (i+1)*n).
// tag: device-scratch pool tag (grow-only persistent staging).
void pack_pull_fields(const double* const* d_srcs, int k, int n, double* h_dst,
                      const char* tag);

// Inverse: push k equal-length arrays from one host buffer (same layout) with a
// single blocking cudaMemcpy into staged device memory, then scatter to the
// destination arrays with one kernel.
void pack_push_fields(const double* h_src, int k, int n, double* const* d_dsts,
                      const char* tag);

// Pull one double from each of two device addresses with one transfer.
void pull_two_scalars(const double* d_a, const double* d_b, double* h_out2,
                      const char* tag);

}  // namespace tenryu::core
