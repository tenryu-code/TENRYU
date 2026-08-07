#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "core/macros.hpp"

namespace tenryu::burn {

struct PartitionTableDeviceView {
  int n_te;
  int n_ti;
  int n_ne;
  double te_min_keV;
  double te_max_keV;
  double ti_min_keV;
  double ti_max_keV;
  double ne_min;
  double ne_max;
  const double* f_ion;
};

TENRYU_HOST_DEVICE inline std::size_t partition_index_device(
    const PartitionTableDeviceView& t, const int product_slot,
    const int i_te, const int i_ti, const int i_ne) {
  return static_cast<std::size_t>(
      ((product_slot * t.n_te + i_te) * t.n_ti + i_ti) * t.n_ne + i_ne);
}

TENRYU_HOST_DEVICE inline double partition_clamp_to_range(
    const double x, const double lo, const double hi) {
  return (x > lo) ? ((x < hi) ? x : hi) : lo;
}

TENRYU_HOST_DEVICE inline double fraley_ion_fraction_device(
    const double Te_keV) {
  return 1.0 / (1.0 + 32.0 / Te_keV);
}

TENRYU_HOST_DEVICE inline double partition_f_ion_device(
    const PartitionTableDeviceView& t, const int product_slot,
    const double Te_keV, const double Ti_keV, const double ne_cm3) {
  const double Te =
      partition_clamp_to_range(Te_keV, t.te_min_keV, t.te_max_keV);
  const double Ti =
      partition_clamp_to_range(Ti_keV, t.ti_min_keV, t.ti_max_keV);
  const double ne = partition_clamp_to_range(ne_cm3, t.ne_min, t.ne_max);
  const double te_coord =
      (std::log(Te) - std::log(t.te_min_keV)) /
      (std::log(t.te_max_keV) - std::log(t.te_min_keV)) *
      static_cast<double>(t.n_te - 1);
  const double ti_coord =
      (std::log(Ti) - std::log(t.ti_min_keV)) /
      (std::log(t.ti_max_keV) - std::log(t.ti_min_keV)) *
      static_cast<double>(t.n_ti - 1);
  const double ne_coord =
      (std::log(ne) - std::log(t.ne_min)) /
      (std::log(t.ne_max) - std::log(t.ne_min)) *
      static_cast<double>(t.n_ne - 1);
  const int i_te0 =
      std::min(static_cast<int>(std::floor(te_coord)), t.n_te - 2);
  const int i_ti0 =
      std::min(static_cast<int>(std::floor(ti_coord)), t.n_ti - 2);
  const int i_ne0 =
      std::min(static_cast<int>(std::floor(ne_coord)), t.n_ne - 2);
  const double w_te = te_coord - static_cast<double>(i_te0);
  const double w_ti = ti_coord - static_cast<double>(i_ti0);
  const double w_ne = ne_coord - static_cast<double>(i_ne0);

  const double f000 =
      t.f_ion[partition_index_device(t, product_slot, i_te0, i_ti0, i_ne0)];
  const double f100 = t.f_ion[partition_index_device(
      t, product_slot, i_te0 + 1, i_ti0, i_ne0)];
  const double f010 = t.f_ion[partition_index_device(
      t, product_slot, i_te0, i_ti0 + 1, i_ne0)];
  const double f110 = t.f_ion[partition_index_device(
      t, product_slot, i_te0 + 1, i_ti0 + 1, i_ne0)];
  const double f001 = t.f_ion[partition_index_device(
      t, product_slot, i_te0, i_ti0, i_ne0 + 1)];
  const double f101 = t.f_ion[partition_index_device(
      t, product_slot, i_te0 + 1, i_ti0, i_ne0 + 1)];
  const double f011 = t.f_ion[partition_index_device(
      t, product_slot, i_te0, i_ti0 + 1, i_ne0 + 1)];
  const double f111 = t.f_ion[partition_index_device(
      t, product_slot, i_te0 + 1, i_ti0 + 1, i_ne0 + 1)];
  const double f00 = (1.0 - w_te) * f000 + w_te * f100;
  const double f10 = (1.0 - w_te) * f010 + w_te * f110;
  const double f0 = (1.0 - w_ti) * f00 + w_ti * f10;
  const double f01 = (1.0 - w_te) * f001 + w_te * f101;
  const double f11 = (1.0 - w_te) * f011 + w_te * f111;
  const double f1 = (1.0 - w_ti) * f01 + w_ti * f11;
  return (1.0 - w_ne) * f0 + w_ne * f1;
}

}  // namespace tenryu::burn
