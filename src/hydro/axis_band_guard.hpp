#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "core/mesh_transaction.hpp"

namespace tenryu::core {
struct State;
}

namespace tenryu::hydro::ale {

// Band-scoped rollback guard over core::ShadowTransaction. The axis band (rows i=0..K-1
// of cells, i=0..K of nodes) occupies CONTIGUOUS PREFIXES of the row-major State fields,
// so the guard captures prefix byte ranges device-to-device (byte-exact, replacing the
// old host snapshot's D2H/H2D vectors). Single-use per attempt: capture ->
// (failure) restore | (success) accept.
class AxisBandGuard {
 public:
  void capture(core::State& state, int K, bool include_radiation,
               cudaStream_t stream);
  void restore(core::State& state, cudaStream_t stream);
  void accept();
  bool captured() const;

 private:
  core::ShadowTransaction tx_;
  int K_ = 0;
  int nz_ = 0;
  int n_materials_ = 0;
  int n_groups_ = 0;
  bool include_radiation_ = false;
  std::vector<std::size_t> captured_bytes_;
  std::vector<void*> captured_ptrs_;
};

}  // namespace tenryu::hydro::ale
