#pragma once

#if defined(__CUDACC__)
#define TENRYU_MB_THETA_HOST_DEVICE __host__ __device__
#else
#define TENRYU_MB_THETA_HOST_DEVICE
#endif

namespace tenryu::mesh {

TENRYU_MB_THETA_HOST_DEVICE inline double multiblock_theta_node(
    const int k,
    const int ntheta,
    const double cap_widen_factor) {
  constexpr double kPi =
      3.1415926535897932384626433832795028841971693993751;
  if (cap_widen_factor == 1.0) {
    return static_cast<double>(k) * kPi / static_cast<double>(ntheta);
  }
  if (k == 0) {
    return 0.0;
  }
  if (k == ntheta) {
    return kPi;
  }
  const double interior_width =
      kPi / (static_cast<double>(ntheta - 2) + 2.0 * cap_widen_factor);
  return cap_widen_factor * interior_width +
         static_cast<double>(k - 1) * interior_width;
}

}  // namespace tenryu::mesh

#undef TENRYU_MB_THETA_HOST_DEVICE
