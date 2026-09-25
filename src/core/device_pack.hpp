#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

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

// d_dst[i] += d_src[i] for 0 <= i < n, one kernel on the legacy default
// stream (ordered after earlier default-stream work, no host synchronization).
void add_device_array(double* d_dst, const double* d_src, std::size_t n);

// d_dsts[a][j] = d_srcs[a][j] for 0 <= j < counts[a] and 0 <= a < k (k <= 8):
// the device-to-device copies of k arrays in one kernel on the legacy default
// stream (ordered after earlier default-stream work, no host synchronization),
// in place of k cudaMemcpy calls. The arrays must not overlap.
void copy_device_arrays(double* const* d_dsts, const double* const* d_srcs, const int* counts,
                        int k);

// Device copy of a host cell mask (State::cell_is_void) for kernels that only
// read it. The mask changes only when cells become or stop being void, so it
// is uploaded when its content differs from the last upload rather than on
// every call; the upload is ordered after earlier default-stream work. The
// pointer is valid until a call with a different mask.
const std::uint8_t* device_cell_is_void(const std::vector<std::uint8_t>& host_mask);

}  // namespace tenryu::core
