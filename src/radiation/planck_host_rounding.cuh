#pragma once

#include "core/device_ordered_sum.cuh"
#include "core/glibc_libm_device.cuh"
#include "radiation/planck_table.cuh"

// PlanckTable::interpolate_b_host on the device, in the host's rounding: the same interval search
// and branches, each operation rounded separately in the host's order (the host build does not
// fuse a multiply and an add), the host's log and exp (core::glibc_libm) and fmax with the x86-64
// glibc rules. The value has the bits of interpolate_b_host on x86-64 processors where glibc
// selects its FMA build of log and exp (FMA and AVX2). For host computations of the step moved to
// the device without changing their results (the S_N outer Marshak boundary, 2026-10-02).

namespace tenryu::radiation {

// planck_log_hermite in the host's rounding.
__device__ inline double planck_log_hermite_host_rounding(const double* ln_x,
                                                          const double* dln_x,
                                                          const int i0,
                                                          const int i1,
                                                          const double du,
                                                          const double w) {
  const double w2 = __dmul_rn(w, w);
  const double w3 = __dmul_rn(w2, w);
  const double h00 = __dadd_rn(__dsub_rn(__dmul_rn(2.0, w3), __dmul_rn(3.0, w2)), 1.0);
  const double h10 = __dadd_rn(__dsub_rn(w3, __dmul_rn(2.0, w2)), w);
  const double h01 = __dadd_rn(__dmul_rn(-2.0, w3), __dmul_rn(3.0, w2));
  const double h11 = __dsub_rn(w3, w2);
  double sum = __dmul_rn(h00, ln_x[i0]);
  sum = __dadd_rn(sum, __dmul_rn(__dmul_rn(h10, du), dln_x[i0]));
  sum = __dadd_rn(sum, __dmul_rn(h01, ln_x[i1]));
  sum = __dadd_rn(sum, __dmul_rn(__dmul_rn(h11, du), dln_x[i1]));
  return core::glibc_libm::exp(sum);
}

// planck_fraction_in_interval in the host's rounding.
__device__ inline double planck_fraction_in_interval_host_rounding(const double* cdf,
                                                                   const double* tail,
                                                                   const double* ln_cdf,
                                                                   const double* ln_tail,
                                                                   const double* dln_cdf,
                                                                   const double* dln_tail,
                                                                   const int n_groups,
                                                                   const int lo,
                                                                   const int hi,
                                                                   const double du,
                                                                   const double w,
                                                                   const int g) {
  const int base0 = lo * n_groups;
  const int base1 = hi * n_groups;
  const auto value = [&](const double* x, const double* ln_x, const double* dln_x,
                         const int k) {
    if (w <= 0.0) {
      return x[base0 + k];
    }
    if (w >= 1.0) {
      return x[base1 + k];
    }
    return planck_log_hermite_host_rounding(ln_x, dln_x, base0 + k, base1 + k, du, w);
  };
  const auto cumulative_side = [&](const int k) {
    return k < 0 || (k < n_groups - 1 && cdf[base0 + k] < 0.5 && cdf[base1 + k] < 0.5);
  };
  const bool g_cum = cumulative_side(g);
  const bool gm1_cum = cumulative_side(g - 1);
  double b = 0.0;
  if (g_cum) {
    const double C_g = value(cdf, ln_cdf, dln_cdf, g);
    const double C_gm1 = (g == 0) ? 0.0 : value(cdf, ln_cdf, dln_cdf, g - 1);
    b = __dsub_rn(C_g, C_gm1);
  } else if (!gm1_cum) {
    const double D_g = (g == n_groups - 1) ? 0.0 : value(tail, ln_tail, dln_tail, g);
    const double D_gm1 = value(tail, ln_tail, dln_tail, g - 1);
    b = __dsub_rn(D_gm1, D_g);
  } else {
    const double D_g = (g == n_groups - 1) ? 0.0 : value(tail, ln_tail, dln_tail, g);
    const double C_gm1 = (g == 0) ? 0.0 : value(cdf, ln_cdf, dln_cdf, g - 1);
    b = __dsub_rn(__dsub_rn(1.0, D_g), C_gm1);
  }
  return core::device_ordered::fmax_like_glibc(b, 0.0);
}

// interpolate_b_host(g, T_eV) without its out-of-range warning (the caller reports it on the host,
// PlanckTable::warn_if_outside_range).
__device__ inline double planck_fraction_host_rounding(const PlanckTableDeviceView& table,
                                                       const int g,
                                                       const double T_eV) {
  if (table.n_groups <= 1) {
    return 1.0;
  }
  if (g < 0 || g >= table.n_groups) {
    return 0.0;
  }
  if (table.n_T <= 1 || table.constant_in_T != 0) {
    return table.b_g[g];
  }
  if (T_eV <= table.T_grid[0]) {
    return table.b_g[g];
  }
  if (T_eV >= table.T_grid[table.n_T - 1]) {
    return table.b_g[(table.n_T - 1) * table.n_groups + g];
  }
  int lo = 0;
  int hi = table.n_T - 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) / 2;
    if (table.T_grid[mid] <= T_eV) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const double dl = __dsub_rn(table.ln_T_grid[hi], table.ln_T_grid[lo]);
  if (!(dl > 0.0)) {
    return table.b_g[lo * table.n_groups + g];
  }
  const double w = __ddiv_rn(__dsub_rn(core::glibc_libm::log(T_eV), table.ln_T_grid[lo]), dl);
  return planck_fraction_in_interval_host_rounding(table.cdf_g, table.tail_g, table.ln_cdf_g,
                                                   table.ln_tail_g, table.dln_cdf_g,
                                                   table.dln_tail_g, table.n_groups, lo, hi, dl,
                                                   w, g);
}

}  // namespace tenryu::radiation
