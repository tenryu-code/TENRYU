#pragma once

#include <cstddef>
#include <cstdint>

namespace tenryu::mesh::tess {

// Per-triangle exact circumcenter batch computed on the GPU.
// Term sequences are byte-identical to the host circumcenter_exact
// (exact_construct.cpp) by construction; any triangle the fixed-capacity
// device path cannot certify carries fallback != 0 and MUST be
// recomputed on the host by the caller.
inline constexpr int kGpuCcXYCap = 48;  // numerator term capacity
inline constexpr int kGpuCcWCap = 16;   // denominator term capacity

struct GpuCircumcenterBatchView {
  // Strides: x/y terms kGpuCcXYCap per triangle, w terms kGpuCcWCap.
  const double* x_terms = nullptr;
  const std::int32_t* x_count = nullptr;
  const double* y_terms = nullptr;
  const std::int32_t* y_count = nullptr;
  const double* w_terms = nullptr;
  const std::int32_t* w_count = nullptr;
  const double* constructed = nullptr;   // 2 per triangle (r, z)
  const std::uint8_t* fallback = nullptr;  // nonzero => host recompute
  std::size_t count = 0;
  double gpu_ms = 0.0;   // H2D + kernel + sync wall time
  double d2h_ms = 0.0;   // D2H wall time
};

// Triangle count of the most recent gpu_circumcenter_batch in this
// process (0 before any batch). Lets dependent kernels assert the
// resident device buffers match their tessellation.
std::size_t gpu_circumcenter_last_batch_count();

// sorted_site_triplets: 3*n_triangles site indices (each triangle's three
// sites sorted ascending by stable_id — the operand order is load-bearing).
// site_rz: 2*n_sites interleaved (r, z).
// The returned pointers reference pinned scratch owned by the core
// device-scratch pool; they stay valid until the next call on this thread.
// Fails loud (TENRYU_ASSERT) on any CUDA error: no silent host downgrade.
GpuCircumcenterBatchView gpu_circumcenter_batch(
    std::size_t n_triangles,
    const std::int32_t* sorted_site_triplets,
    const double* site_rz,
    std::size_t n_sites);

}  // namespace tenryu::mesh::tess
