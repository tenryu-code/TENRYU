#pragma once

#include "core/glibc_libm_device.cuh"
#include "materials/eos_table_reclose.cuh"
#include "materials/zbar_math.hpp"

namespace tenryu::materials {

// zbar_tf_value (zbar_math.hpp) on the device with the host's rounding: the same operations in
// the same order, each rounded separately (the host build does not fuse a multiply and an add),
// pow and exp from core::glibc_libm (the results of x86-64 glibc), and the correctly rounded
// square root. The result has the bits of the host's compute_zbar_tf on x86-64 processors where
// glibc selects its FMA build of pow and exp (FMA and AVX2); a NaN result is a NaN of possibly
// other bits. The explicit intrinsics keep the arithmetic separate in translation units compiled
// with contraction.
__device__ inline double zbar_tf_value_host_rounding(const double rho, const double Te_eV,
                                                     const double Z_nuc, const double A_amu) {
  namespace libm = core::glibc_libm;
  using namespace zbar_tf_fit;
  const double Z = reclose_max(Z_nuc, 0.0);
  const double A = reclose_max(A_amu, 1.0e-30);
  if (!(Z > 0.0)) return 0.0;
  const double rho_c = (rho > 0.0 && rho < 1.0e300) ? rho : 1.0e-30;
  const double T_c = (Te_eV > 0.0 && Te_eV < 1.0e300) ? Te_eV : 0.0;
  const double R = reclose_max(__ddiv_rn(rho_c, __dmul_rn(Z, A)), 1.0e-300);
  const double T0 = __ddiv_rn(T_c, libm::pow(Z, 4.0 / 3.0));
  const double TF = __ddiv_rn(T0, __dadd_rn(1.0, T0));
  const double TF2 = __dmul_rn(TF, TF);
  const double TF7 = __dmul_rn(__dmul_rn(__dmul_rn(TF2, TF2), TF2), TF);
  const double A_fit = __dadd_rn(__dmul_rn(kA1, libm::pow(T0, kA2)),
                                 __dmul_rn(kA3, libm::pow(T0, kA4)));
  const double B_fit =
      -libm::exp(__dadd_rn(__dadd_rn(kB0, __dmul_rn(kB1, TF)), __dmul_rn(kB2, TF7)));
  const double C_fit = __dadd_rn(__dmul_rn(kC1, TF), kC2);
  const double Q1 = __dmul_rn(A_fit, libm::pow(R, B_fit));
  const double Q = libm::pow(__dadd_rn(libm::pow(R, C_fit), libm::pow(Q1, C_fit)),
                             __ddiv_rn(1.0, C_fit));
  const double x = __dmul_rn(kAlpha, libm::pow(Q, kBeta));
  if (!(x < 1.0e300)) return Z;
  const double zbar = __ddiv_rn(
      __dmul_rn(Z, x),
      __dadd_rn(__dadd_rn(1.0, x), __dsqrt_rn(__dadd_rn(1.0, __dmul_rn(2.0, x)))));
  return reclose_clamp(zbar, 0.0, Z);
}

}  // namespace tenryu::materials
