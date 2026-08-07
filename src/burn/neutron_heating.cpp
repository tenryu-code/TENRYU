#include "burn/neutron_heating.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "burn/burn_constants.hpp"
#include "burn/neutron_heating_device.cuh"
#include "core/error.hpp"

namespace tenryu::burn {
namespace {

constexpr double kBarnToCm2 = 1.0e-24;

// Gauss-Legendre nodes/weights on [-1,1] (even n). Local mirror of the SN
// builder in sn_transport_1d.cpp, copied to keep tenryu_burn free of a
// radiation dependency; same Newton-on-P_n algorithm and tolerances.
void gauss_legendre(const int n, std::vector<double>& mu,
                    std::vector<double>& weight) {
  mu.assign(static_cast<std::size_t>(n), 0.0);
  weight.assign(static_cast<std::size_t>(n), 0.0);
  const int half = n / 2;
  for (int k = 0; k < half; ++k) {
    double w = 0.0;
    neutron_heating_device_detail::gauss_legendre_node_weight(
        n, k, &mu[static_cast<std::size_t>(k)],
        &mu[static_cast<std::size_t>(n - 1 - k)], &w);
    weight[static_cast<std::size_t>(k)] = w;
    weight[static_cast<std::size_t>(n - 1 - k)] = w;
  }
}

}  // namespace

NeutronHeatingResult deposit_neutron_heating_1d(
    const int n_cells, const double* r_node, const double* rho,
    const double* Te_eV, const double* Ti_eV, const double* zbar,
    const double* A_eff, const std::vector<double>& burn_y,
    const std::vector<double>& emit_erg, const NeutronHeatingParams& p,
    const PartitionTable& table, std::vector<double>& dE_e,
    std::vector<double>& dE_i) {
  NeutronHeatingResult out;
  TENRYU_ASSERT(p.n_mu >= 2 && (p.n_mu % 2) == 0,
                "neutron_heating requires an even n_mu >= 2");
  TENRYU_ASSERT(emit_erg.size() ==
                    static_cast<std::size_t>(n_cells) *
                        static_cast<std::size_t>(kNumNeutronLines),
                "neutron_heating emit_erg size mismatch");

  std::vector<double> mu;
  std::vector<double> w;
  gauss_legendre(p.n_mu, mu, w);

  // Post-network composition: number densities [1/cm^3] of the two elastic
  // targets over the whole mesh.
  std::vector<double> nD(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> nT(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    if (rho[c] > 0.0) {
      nD[static_cast<std::size_t>(c)] =
          burn_y[static_cast<std::size_t>(c * kNumSpecies + kD)] * rho[c];
      nT[static_cast<std::size_t>(c)] =
          burn_y[static_cast<std::size_t>(c * kNumSpecies + kT)] * rho[c];
    }
  }
  const PartitionTableDeviceView table_view{
      table.n_te,
      table.n_ti,
      table.n_ne,
      table.te_min_keV,
      table.te_max_keV,
      table.ti_min_keV,
      table.ti_max_keV,
      table.ne_min,
      table.ne_max,
      table.f_ion.data()};

  for (int l = 0; l < kNumNeutronLines; ++l) {
    const double sD = neutron_sigma_el_barn(l, 0) * kBarnToCm2;  // cm^2
    const double sT = neutron_sigma_el_barn(l, 1) * kBarnToCm2;
    const double fD = neutron_f_transfer(l, 0);
    const double fT = neutron_f_transfer(l, 1);
    const int slotD = neutron_recoil_slot(l, 0);
    const int slotT = neutron_recoil_slot(l, 1);
    for (int j = 0; j < n_cells; ++j) {
      const double E_cell =
          emit_erg[static_cast<std::size_t>(j) *
                       static_cast<std::size_t>(kNumNeutronLines) +
                   static_cast<std::size_t>(l)];
      if (!(E_cell > 0.0)) {
        continue;
      }
      out.emitted += E_cell;
      const double r0 = 0.5 * (r_node[j] + r_node[j + 1]);
      for (int q = 0; q < p.n_mu; ++q) {
        // isotropic emission: GL weights sum to 2 over [-1,1].
        const double E_ch = E_cell * 0.5 * w[static_cast<std::size_t>(q)];
        double T_rem = 1.0;
        neutron_heating_device_detail::walk_spherical_chord(
            r0, mu[static_cast<std::size_t>(q)], r_node, n_cells,
            [&](const double ds, const int c) {
              const std::size_t cs = static_cast<std::size_t>(c);
              const neutron_heating_device_detail::SegmentDeposit dep =
                  neutron_heating_device_detail::deposit_segment(
                      ds, c, E_ch, T_rem, sD, sT, fD, fT, slotD, slotT,
                      nD.data(), nT.data(), rho, Te_eV, Ti_eV, zbar, A_eff,
                      p.use_fraley_partition, table_view);
              dE_i[cs] += dep.dep_i_D;
              dE_e[cs] += dep.dep_e_D;
              out.dep_i += dep.dep_i_D;
              out.dep_e += dep.dep_e_D;
              out.degraded += dep.degraded_D;
              dE_i[cs] += dep.dep_i_T;
              dE_e[cs] += dep.dep_e_T;
              out.dep_i += dep.dep_i_T;
              out.dep_e += dep.dep_e_T;
              out.degraded += dep.degraded_T;
              T_rem = dep.transmission;
            });
        out.escaped += E_ch * T_rem;
      }
    }
  }

  out.conservation_resid =
      std::abs(out.emitted -
               (out.dep_e + out.dep_i + out.degraded + out.escaped)) /
      std::max(out.emitted, 1.0e-300);
  return out;
}

}  // namespace tenryu::burn
