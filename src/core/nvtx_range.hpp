#pragma once

#include <cstdlib>

#if TENRYU_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace tenryu::core {

// Opt-in host ranges for Nsight Systems; no CUDA completion is requested.
class NvtxRange {
 public:
  explicit NvtxRange(const char* name) {
#if TENRYU_ENABLE_NVTX
    if (enabled()) {
      nvtxRangePushA(name);
    }
#else
    (void)name;
#endif
  }
  ~NvtxRange() {
#if TENRYU_ENABLE_NVTX
    if (enabled()) {
      nvtxRangePop();
    }
#endif
  }
  void reset(const char* name) {
#if TENRYU_ENABLE_NVTX
    if (enabled()) {
      nvtxRangePop();
      nvtxRangePushA(name);
    }
#else
    (void)name;
#endif
  }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;

 private:
#if TENRYU_ENABLE_NVTX
  static bool enabled() {
    static const bool value = [] {
      const char* raw = std::getenv("TENRYU_NVTX_RANGES");
      return raw != nullptr && raw[0] == '1';
    }();
    return value;
  }
#endif
};

}  // namespace tenryu::core
