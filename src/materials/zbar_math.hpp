#pragma once

#include <cmath>

#include <cuda_runtime.h>

#include "core/macros.hpp"
#include "materials/eos_table_reclose.cuh"

namespace tenryu::materials {

// Preserve the host's multiply-then-add rounding when CUDA contracts expressions.
TENRYU_HOST_DEVICE inline double zbar_separate_add_product(
    const double a, const double b, const double c) {
#if defined(__CUDA_ARCH__)
  return __dadd_rn(a, __dmul_rn(b, c));
#else
  return a + b * c;
#endif
}

// Thomas-Fermi mean ionization, R. M. More, Adv. At. Mol. Phys. 21, 305
// (1985), Table IV ("an approximate fit to" the TF ionization state), with the
// TF scaling variables R = rho/(Z A) [g/cm^3] and T0 = T/Z^{4/3} [eV]:
//   T_F = T0/(1+T0),  A = a1 T0^a2 + a3 T0^a4,  B = -exp(b0 + b1 T_F + b2 T_F^7),
//   C = c1 T_F + c2,  Q1 = A R^B,  Q = (R^C + Q1^C)^{1/C},  x = alpha Q^beta,
//   Z* = Z x / (1 + x + sqrt(1 + 2 x)).
// (2026-09-23: replaces a formula that scaled density by rho/A and approached
// full ionization as 1 - sqrt(T0/T) — D at 0.17 g/cc gave 0.86 at 100 eV and
// 0.946 at 1 keV instead of 0.968 / 0.997.) Z and A are the material's mean
// nuclear charge and mass; rho > 0 and T >= 0 are the only guards.
TENRYU_HOST_DEVICE inline double zbar_tf_value(
    const double rho, const double Te_eV, const double Z_nuc, const double A_amu) {
  const double Z = reclose_max(Z_nuc, 0.0);
  const double A = reclose_max(A_amu, 1.0e-30);
  if (!(Z > 0.0)) return 0.0;
  constexpr double kAlpha = 14.3139;
  constexpr double kBeta = 0.6624;
  constexpr double kA1 = 0.003323;
  constexpr double kA2 = 0.9718;
  constexpr double kA3 = 9.26148e-5;
  constexpr double kA4 = 3.10165;
  constexpr double kB0 = -1.7630;
  constexpr double kB1 = 1.43175;
  constexpr double kB2 = 0.31546;
  constexpr double kC1 = -0.366667;
  constexpr double kC2 = 0.983333;
  const double rho_c = (rho > 0.0 && rho < 1.0e300) ? rho : 1.0e-30;
  const double T_c = (Te_eV > 0.0 && Te_eV < 1.0e300) ? Te_eV : 0.0;
  const double R = reclose_max(rho_c / (Z * A), 1.0e-300);
  const double T0 = T_c / ::pow(Z, 4.0 / 3.0);
  const double TF = T0 / (1.0 + T0);
  const double TF2 = TF * TF;
  const double TF7 = TF2 * TF2 * TF2 * TF;
  const double A_fit = kA1 * ::pow(T0, kA2) + kA3 * ::pow(T0, kA4);
  const double B_fit = -::exp(kB0 + kB1 * TF + kB2 * TF7);
  const double C_fit = kC1 * TF + kC2;
  const double Q1 = A_fit * ::pow(R, B_fit);
  const double Q = ::pow(::pow(R, C_fit) + ::pow(Q1, C_fit), 1.0 / C_fit);
  const double x = kAlpha * ::pow(Q, kBeta);
  if (!(x < 1.0e300)) return Z;
  const double zbar = Z * x / (1.0 + x + ::sqrt(1.0 + 2.0 * x));
  return reclose_clamp(zbar, 0.0, Z);
}

struct ZbarTableView {
  DeviceEOSTableView table{};
  double rho_min = 0.0;
  double rho_max = 0.0;
  double T_min = 0.0;
  double T_max = 0.0;
};

struct ZbarClampedInput {
  double rho;
  double T;
  bool clamped;
};

TENRYU_HOST_DEVICE inline ZbarClampedInput zbar_clamp_input(
    const ZbarTableView& table, const double rho, const double T) {
  double rho_safe = std::isfinite(rho) ? rho : table.rho_min;
  double T_safe = std::isfinite(T) ? T : table.T_min;
  bool clamped = false;
  if (!(rho_safe > 0.0)) { rho_safe = table.rho_min; clamped = true; }
  if (!(T_safe > 0.0)) { T_safe = table.T_min; clamped = true; }
  if (rho_safe < table.rho_min || rho_safe > table.rho_max) clamped = true;
  if (T_safe < table.T_min || T_safe > table.T_max) clamped = true;
  return {reclose_clamp(rho_safe, table.rho_min, table.rho_max),
          reclose_clamp(T_safe, table.T_min, table.T_max), clamped};
}

TENRYU_HOST_DEVICE inline double zbar_tabular_value(
    const ZbarTableView& view, const ZbarClampedInput& input) {
  const auto& table = view.table;
  const double x = input.rho == view.rho_min ? table.log_rho_grid[0] :
                   input.rho == view.rho_max ? table.log_rho_grid[table.n_rho - 1] :
                   ::log(input.rho);
  const double y = input.T == view.T_min ? table.log_T_grid[0] :
                   input.T == view.T_max ? table.log_T_grid[table.n_T - 1] :
                   ::log(input.T);
  return reclose_max(0.0, reclose_bilinear_at_log_coordinates<true>(
      table, table.e_table, x, y));
}

}  // namespace tenryu::materials
