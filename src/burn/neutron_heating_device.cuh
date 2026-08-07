#pragma once

#include <algorithm>
#include <cmath>

#include "burn/partition_device.cuh"
#include "core/macros.hpp"

namespace tenryu::burn {

namespace neutron_heating_device_detail {

constexpr double kProtonMassG = 1.6726219e-24;
constexpr double kPi = 3.141592653589793238462643383279502884;

TENRYU_HOST_DEVICE inline void gauss_legendre_node_weight(
    const int n, const int k, double* const mu_negative,
    double* const mu_positive, double* const weight) {
  constexpr double tol = 1.0e-15;
  double x = std::cos(kPi * (static_cast<double>(k) + 0.75) /
                      (static_cast<double>(n) + 0.5));
  double p1 = 0.0;
  double p2 = 0.0;
  for (int iter = 0; iter < 64; ++iter) {
    p1 = 1.0;
    p2 = 0.0;
    for (int j = 1; j <= n; ++j) {
      const double p3 = p2;
      p2 = p1;
      p1 = ((2.0 * static_cast<double>(j) - 1.0) * x * p2 -
            (static_cast<double>(j) - 1.0) * p3) /
           static_cast<double>(j);
    }
    const double pp =
        static_cast<double>(n) * (x * p1 - p2) / (x * x - 1.0);
    const double dx = p1 / pp;
    x -= dx;
    if (std::abs(dx) <= tol) {
      break;
    }
  }
  p1 = 1.0;
  p2 = 0.0;
  for (int j = 1; j <= n; ++j) {
    const double p3 = p2;
    p2 = p1;
    p1 = ((2.0 * static_cast<double>(j) - 1.0) * x * p2 -
          (static_cast<double>(j) - 1.0) * p3) /
         static_cast<double>(j);
  }
  const double pp =
      static_cast<double>(n) * (x * p1 - p2) / (x * x - 1.0);
  *mu_negative = -x;
  *mu_positive = x;
  *weight = 2.0 / ((1.0 - x * x) * pp * pp);
}

// Spherical chord walk from radius r0 (> 0) with direction cosine mu against
// +r, through shells r_node[0..n_cells]. Emits (ds, cell) segments in flight
// order. GL interior nodes give |mu| < 1, so with r0 > 0 the impact
// parameter b = r0 sqrt(1-mu^2) is > 0 and the exact center is never hit.
// Annular meshes (r_node[0] > 0): the central hole is vacuum — traversed
// with no segment emitted.
TENRYU_HOST_DEVICE inline double chord_s_out(
    const double r0, const double mu, const double b2, const double R) {
  return -r0 * mu + std::sqrt(std::max(R * R - b2, 0.0));
}

TENRYU_HOST_DEVICE inline double chord_s_in(
    const double r0, const double mu, const double b2, const double R) {
  return -r0 * mu - std::sqrt(std::max(R * R - b2, 0.0));
}

template <typename F>
TENRYU_HOST_DEVICE inline void walk_spherical_chord(
    const double r0, const double mu, const double* const r_node,
    const int n_cells, F&& segment) {
  const double b2 = r0 * r0 * (1.0 - mu * mu);
  int c = 0;
  while (c + 1 < n_cells && r_node[c + 1] < r0) {
    ++c;
  }
  double s_cur = 0.0;
  bool inbound = (mu < 0.0);
  while (true) {
    if (inbound) {
      const double R_inner = r_node[c];
      if (c > 0 && R_inner * R_inner > b2) {
        const double s_next = chord_s_in(r0, mu, b2, R_inner);
        segment(s_next - s_cur, c);
        s_cur = s_next;
        --c;
      } else {
        if (c == 0 && r_node[0] > 0.0 && b2 < r_node[0] * r_node[0]) {
          // annular mesh: march to the inner boundary, skip the vacuum
          // hole, re-enter outbound at the same radius.
          segment(chord_s_in(r0, mu, b2, r_node[0]) - s_cur, 0);
          s_cur = chord_s_out(r0, mu, b2, r_node[0]);
        }
        const double s_next = chord_s_out(r0, mu, b2, r_node[c + 1]);
        segment(s_next - s_cur, c);
        s_cur = s_next;
        ++c;
        inbound = false;
        if (c >= n_cells) {
          return;
        }
      }
    } else {
      const double s_next = chord_s_out(r0, mu, b2, r_node[c + 1]);
      segment(s_next - s_cur, c);
      s_cur = s_next;
      ++c;
      if (c >= n_cells) {
        return;
      }
    }
  }
}

struct SegmentDeposit {
  double dep_e_D = 0.0;
  double dep_i_D = 0.0;
  double degraded_D = 0.0;
  double dep_e_T = 0.0;
  double dep_i_T = 0.0;
  double degraded_T = 0.0;
  double transmission = 0.0;
};

TENRYU_HOST_DEVICE inline SegmentDeposit deposit_segment(
    const double ds, const int c, const double E_ch, const double T_rem,
    const double sD, const double sT, const double fD, const double fT,
    const int slotD, const int slotT, const double* const nD,
    const double* const nT, const double* const rho,
    const double* const Te_eV, const double* const Ti_eV,
    const double* const zbar, const double* const A_eff,
    const bool use_fraley_partition,
    const PartitionTableDeviceView& table) {
  SegmentDeposit out;
  out.transmission = T_rem;
  if (!(T_rem > 0.0) || !(ds > 0.0)) {
    return out;
  }
  const double SD = nD[c] * sD;  // 1/cm
  const double ST = nT[c] * sT;
  const double Sigma = SD + ST;
  if (!(Sigma > 0.0)) {
    return out;
  }
  const double dtau = Sigma * ds;
  const double P_col = T_rem * (-std::expm1(-dtau));
  const double wD = SD / Sigma;
  const double wT = ST / Sigma;
  const double Te_keV = ((Te_eV[c] > 0.0) ? Te_eV[c] : 0.0) * 1.0e-3;
  const double Ti_keV = ((Ti_eV[c] > 0.0) ? Ti_eV[c] : 0.0) * 1.0e-3;
  double ne = 0.0;
  if (A_eff[c] > 0.0 && rho[c] > 0.0 && zbar[c] > 0.0) {
    ne = rho[c] * zbar[c] / (A_eff[c] * kProtonMassG);
  }
  if (!(ne > 0.0)) {
    ne = table.ne_min;
  }
  const double E_col = E_ch * P_col;
  if (wD > 0.0) {
    const double f_ion =
        use_fraley_partition
            ? fraley_ion_fraction_device(Te_keV)
            : partition_f_ion_device(table, slotD, Te_keV, Ti_keV, ne);
    out.dep_i_D = E_col * wD * fD * f_ion;
    out.dep_e_D = E_col * wD * fD * (1.0 - f_ion);
    out.degraded_D = E_col * wD * (1.0 - fD);
  }
  if (wT > 0.0) {
    const double f_ion =
        use_fraley_partition
            ? fraley_ion_fraction_device(Te_keV)
            : partition_f_ion_device(table, slotT, Te_keV, Ti_keV, ne);
    out.dep_i_T = E_col * wT * fT * f_ion;
    out.dep_e_T = E_col * wT * fT * (1.0 - f_ion);
    out.degraded_T = E_col * wT * (1.0 - fT);
  }
  out.transmission *= std::exp(-dtau);
  return out;
}

}  // namespace neutron_heating_device_detail

}  // namespace tenryu::burn
