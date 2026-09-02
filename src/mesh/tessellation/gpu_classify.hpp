#pragma once

#include <cstddef>
#include <cstdint>

namespace tenryu::mesh::tess {

// Per-dual-node certified domain classification, computed on the GPU
// from the device-resident circumcenter expansions left by
// gpu_circumcenter_batch. Status semantics:
//   0 = certified outside (winding zero, no boundary contact possible)
//   1 = certified inside  (winding nonzero, no boundary contact)
//   2 = uncertain — the caller MUST run the original host
//       classification for this node
// Certified answers are exact: every per-segment sign is accepted only
// when its rounded-arithmetic error bound proves it, so the winding
// number equals the exact one and no segment can touch the point.
struct GpuClassifyView {
  const std::uint8_t* status = nullptr;
  std::size_t count = 0;
  double gpu_ms = 0.0;
  std::size_t certified_inside = 0;
  std::size_t certified_outside = 0;
  std::size_t uncertain = 0;
};

// node_rep_triangle: per dual node, the batch slot (representative
// triangle id) holding its exact point's term sequences.
// segments_rz: 4 doubles per domain segment (a.r, a.z, b.r, b.z), in
// domain-segment order after preprocess_domain.
// Fails loud (TENRYU_ASSERT) on CUDA errors or when the resident batch
// count does not cover the referenced triangles.
GpuClassifyView gpu_classify_batch(std::size_t n_nodes,
                                   const std::int32_t* node_rep_triangle,
                                   std::size_t n_triangles,
                                   const double* segments_rz,
                                   std::size_t n_segments);

}  // namespace tenryu::mesh::tess
