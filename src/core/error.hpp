#pragma once

#include <string>
#include <string_view>

namespace tenryu::core {

void log_fatal(const std::string& message);
void log_error(const std::string& message);
void log_warning(const std::string& message);
void log_info(const std::string& message);
void log_debug(const std::string& message);

[[noreturn]] void tenryu_abort(const char* condition,
                               std::string_view message,
                               const char* file,
                               int line);

}  // namespace tenryu::core

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define CUDA_CHECK(expr)                                                          \
  do {                                                                            \
    const cudaError_t tenryu_cuda_err__ = (expr);                                 \
    if (tenryu_cuda_err__ != cudaSuccess) {                                       \
      ::tenryu::core::tenryu_abort(#expr, cudaGetErrorString(tenryu_cuda_err__),  \
                                   __FILE__, __LINE__);                           \
    }                                                                             \
  } while (0)
#else
#define CUDA_CHECK(expr) ((void)0)
#endif

// Fail-fast by design: TENRYU_ASSERT aborts immediately to preserve post-mortem state.
// Future architecture work may add an emergency checkpoint hook before tenryu_abort().
#define TENRYU_ASSERT(cond, msg) \
  do {                            \
    if (!(cond)) {                \
      ::tenryu::core::tenryu_abort(#cond, (msg), __FILE__, __LINE__); \
    }                             \
  } while (0)
