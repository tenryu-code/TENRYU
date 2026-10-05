#pragma once

#include <cstdio>

#if defined(__GLIBC__)
#include <gnu/libc-version.h>
#endif

namespace tenryu::core::glibc_libm {

// Whether this process's exp, log and pow are the build that glibc_libm_device.cuh reproduces:
// glibc 2.28 or later (the algorithms there) on an x86-64 processor with FMA and AVX2, where the
// library's ifunc selects its FMA build. Comparisons of device results with host results computed
// by these functions hold bit for bit only then; tests skip otherwise.
inline bool host_has_reproduced_build() {
#if defined(__x86_64__) && defined(__GLIBC__)
  int major = 0;
  int minor = 0;
  if (std::sscanf(gnu_get_libc_version(), "%d.%d", &major, &minor) != 2) {
    return false;
  }
  if (major < 2 || (major == 2 && minor < 28)) {
    return false;
  }
  return __builtin_cpu_supports("fma") && __builtin_cpu_supports("avx2");
#else
  return false;
#endif
}

}  // namespace tenryu::core::glibc_libm
