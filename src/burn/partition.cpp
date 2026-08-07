#include "burn/partition.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <future>

#include <cuda_runtime.h>

#include "burn/burn_constants.hpp"
#include "burn/deposition.cuh"
#include "burn/partition_device.cuh"

namespace tenryu::burn {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kElectronChargeEsu = 4.80320425e-10;
constexpr double kElectronMassG = 9.1093837015e-28;
constexpr double kProtonMassG = 1.6726219e-24;
constexpr double kHbarErgS = 1.054571817e-27;
constexpr double kKeVToErg = 1.602176634e-9;
constexpr double kMeVToErgLocal = 1.602176634e-6;
constexpr int kIntegralPoints = 4000;

struct ProductSpec {
  int species;
  double E0_MeV;
};

struct StoppingComponents {
  double ions;
  double total;
};

ProductSpec product_spec(const int slot) {
  // Slots 6-9: v2-E neutron elastic recoils at the mean first-collision
  // energy f_transfer * E_line (design doc section E.2/E.3).
  constexpr ProductSpec t[PartitionTable::kNumProductSlots] = {
      {kHe4, 3.540}, {kT, 1.010},   {kP, 3.023},
      {kHe3, 0.820}, {kHe4, 3.690}, {kP, 14.663},
      {kD, 4.435},   {kT, 2.884},   {kD, 1.089},  {kT, 0.920},
  };
  return t[slot];
}

double lp_dedx_field(const int test_species, const double E_erg,
                     const double Te_erg, const double Tf_erg,
                     const double ne, const double n_f, const double m_f,
                     const double Z_f) {
  if (!(n_f > 0.0)) {
    return 0.0;
  }

  const double Z_t = species_Z(test_species);
  const double m_t = species_A(test_species) * kProtonMassG;
  const double vt2 = 2.0 * E_erg / m_t;
  const double vt = std::sqrt(vt2);
  const double vf2 = 2.0 * Tf_erg / m_f;
  const double x = vt2 / vf2;
  const double sx = std::sqrt(x);
  const double ex = std::exp(-x);
  const double mu = std::erf(sx) - 2.0 * std::sqrt(x / kPi) * ex;
  const double dmu = 2.0 * std::sqrt(x / kPi) * ex;
  const double wpf2 =
      4.0 * kPi * n_f * (Z_f * kElectronChargeEsu) *
      (Z_f * kElectronChargeEsu) / m_f;
  const double m_r = m_t * m_f / (m_t + m_f);
  const double u2 = vt2 + vf2;
  const double u = std::sqrt(u2);
  const double p_perp =
      Z_t * Z_f * kElectronChargeEsu * kElectronChargeEsu / (m_r * u2);
  const double p_min =
      std::sqrt(p_perp * p_perp +
                (kHbarErgS / (2.0 * m_r * u)) *
                    (kHbarErgS / (2.0 * m_r * u)));
  const double lam_De =
      std::sqrt(Te_erg / (4.0 * kPi * ne * kElectronChargeEsu *
                          kElectronChargeEsu));
  const double lnLb = std::max(2.0, std::log(lam_De / p_min));
  const double G =
      mu - (m_f / m_t) * (dmu - (mu + dmu) / lnLb);
  const double collective = (x > 1.0) ? std::log(1.123 * std::sqrt(x)) : 0.0;
  return (Z_t * kElectronChargeEsu) * (Z_t * kElectronChargeEsu) / vt2 *
         wpf2 * (G * lnLb + collective);
}

StoppingComponents lp_stopping_components(
    const int test_species, const double E_erg, const double Te_erg,
    const double Ti_erg, const double ne, const double x_D,
    const double x_T, const double x_He3) {
  const double zbar = x_D + x_T + 2.0 * x_He3;
  const double n_i_total = ne / zbar;
  const double dEdx_e =
      lp_dedx_field(test_species, E_erg, Te_erg, Te_erg, ne, ne,
                    kElectronMassG, 1.0);
  double dEdx_ions = 0.0;
  dEdx_ions += lp_dedx_field(test_species, E_erg, Te_erg, Ti_erg, ne,
                             x_D * n_i_total, species_A(kD) * kProtonMassG,
                             species_Z(kD));
  dEdx_ions += lp_dedx_field(test_species, E_erg, Te_erg, Ti_erg, ne,
                             x_T * n_i_total, species_A(kT) * kProtonMassG,
                             species_Z(kT));
  dEdx_ions += lp_dedx_field(test_species, E_erg, Te_erg, Ti_erg, ne,
                             x_He3 * n_i_total,
                             species_A(kHe3) * kProtonMassG, species_Z(kHe3));
  return {dEdx_ions, dEdx_ions + dEdx_e};
}

double lp_ion_fraction(const int test_species, const double E0_MeV,
                       const double Te_keV, const double Ti_keV,
                       const double ne, const double x_D, const double x_T,
                       const double x_He3) {
  const double E0 = E0_MeV * kMeVToErgLocal;
  const double Te_erg = Te_keV * kKeVToErg;
  const double Ti_erg = Ti_keV * kKeVToErg;
  const double E_min = std::max(1.5 * Te_erg, 1.0e-3 * E0);
  const double dE = (E0 - E_min) / static_cast<double>(kIntegralPoints - 1);
  double numerator = 0.0;
  double denominator = 0.0;

  StoppingComponents prev =
      lp_stopping_components(test_species, E0, Te_erg, Ti_erg, ne, x_D, x_T,
                             x_He3);
  for (int i = 1; i < kIntegralPoints; ++i) {
    const double E = E0 - dE * static_cast<double>(i);
    const StoppingComponents cur = lp_stopping_components(
        test_species, E, Te_erg, Ti_erg, ne, x_D, x_T, x_He3);
    if (prev.total > 0.0 && cur.total > 0.0) {
      const double r0 = prev.ions / prev.total;
      const double r1 = cur.ions / cur.total;
      numerator += 0.5 * (r0 + r1) * dE;
      denominator += dE;
    }
    prev = cur;
  }

  return (denominator > 0.0) ? (numerator / denominator) : 0.0;
}

std::size_t partition_index(const PartitionTable& t, const int product_slot,
                            const int i_te, const int i_ti, const int i_ne) {
  return static_cast<std::size_t>(
      ((product_slot * t.n_te + i_te) * t.n_ti + i_ti) * t.n_ne + i_ne);
}

double log_grid_value(const double v_min, const double v_max, const int n,
                      const int i) {
  const double a = std::log(v_min);
  const double b = std::log(v_max);
  return std::exp(a + (b - a) * static_cast<double>(i) /
                          static_cast<double>(n - 1));
}

}  // namespace

PartitionTable build_partition_table(double x_D, double x_T, double x_He3) {
  PartitionTable t;
  t.x_D = x_D;
  t.x_T = x_T;
  t.x_He3 = x_He3;
  t.f_ion.assign(static_cast<std::size_t>(PartitionTable::kNumProductSlots) *
                     t.n_te * t.n_ti * t.n_ne,
                 0.0);

  std::vector<std::future<void>> futures;
  futures.reserve(PartitionTable::kNumProductSlots);
  for (int slot = 0; slot < PartitionTable::kNumProductSlots; ++slot) {
    futures.emplace_back(std::async(std::launch::async, [&, slot]() {
      const ProductSpec p = product_spec(slot);
      for (int i_te = 0; i_te < t.n_te; ++i_te) {
        const double Te_keV =
            log_grid_value(t.te_min_keV, t.te_max_keV, t.n_te, i_te);
        for (int i_ti = 0; i_ti < t.n_ti; ++i_ti) {
          const double Ti_keV =
              log_grid_value(t.ti_min_keV, t.ti_max_keV, t.n_ti, i_ti);
          for (int i_ne = 0; i_ne < t.n_ne; ++i_ne) {
            const double ne =
                log_grid_value(t.ne_min, t.ne_max, t.n_ne, i_ne);
            t.f_ion[partition_index(t, slot, i_te, i_ti, i_ne)] =
                lp_ion_fraction(p.species, p.E0_MeV, Te_keV, Ti_keV, ne, x_D,
                                x_T, x_He3);
          }
        }
      }
    }));
  }
  for (auto& future : futures) {
    future.get();
  }

  return t;
}

double fraley_ion_fraction(double Te_keV) {
  return fraley_ion_fraction_device(Te_keV);
}

double partition_f_ion(const PartitionTable& t, int product_slot,
                       double Te_keV, double Ti_keV, double ne_cm3) {
  const PartitionTableDeviceView view{
      t.n_te,       t.n_ti,       t.n_ne,  t.te_min_keV, t.te_max_keV,
      t.ti_min_keV, t.ti_max_keV, t.ne_min, t.ne_max,    t.f_ion.data()};
  return partition_f_ion_device(view, product_slot, Te_keV, Ti_keV, ne_cm3);
}

}  // namespace tenryu::burn
