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

TENRYU_HOST_DEVICE inline double zbar_tf_value(
    const double rho, const double Te_eV, const double Z_nuc, const double A_amu) {
  const double Z = reclose_max(Z_nuc, 0.0);
  const double A = reclose_max(A_amu, 1.0e-30);
  if (Z <= 0.0) return 0.0;
  const double rho_c = reclose_clamp(rho, 1.0e-3, 1.0e4);
  const double Te_c = reclose_clamp(Te_eV, 1.0e-2, 1.0e4);
  const double rho_H = rho_c / A;
  const double Te_H = Te_c / ::pow(Z, 4.0 / 3.0);
  const double eta = ::sqrt(reclose_max(rho_H / 0.148, 0.0));
  const double z0_H = eta / (1.0 + eta);
  double T0_exp_arg = 6.98 * ::pow(reclose_max(rho_H, 0.0), 0.075);
  if (!std::isfinite(T0_exp_arg)) T0_exp_arg = 700.0;
  T0_exp_arg = reclose_min(T0_exp_arg, 700.0);
  const double T0_H = 0.0327 * ::exp(T0_exp_arg);
  const double Y = 1.0 / (1.0 + ::sqrt(reclose_max(T0_H / reclose_max(Te_H, 1.0e-30), 0.0)));
  const double zbar = Z * zbar_separate_add_product(z0_H, 1.0 - z0_H, Y);
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
