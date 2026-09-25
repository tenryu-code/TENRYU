#include "burn/partition.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <future>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

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

// Li-Petrasso stopping of a test particle on one field species, split by what
// the terms depend on so that the table build evaluates each of them once:
// lp_velocity_terms depends on the test particle's energy and the field
// temperature only, lp_dedx_from_terms adds the field density and the
// Coulomb logarithm (the Debye length of the field electrons). Every
// quantity is the expression of the combined per-point formula, evaluated in
// the same order, so the table is bitwise that of evaluating the formula at
// every point.
struct LpVelocityTerms {
  double mu;
  double dmu;
  double p_min;
  double collective;
  double pre;  // (Z_t e)^2 / v_t^2
};

LpVelocityTerms lp_velocity_terms(const int test_species, const double E_erg,
                                  const double Tf_erg, const double m_f,
                                  const double Z_f) {
  const double Z_t = species_Z(test_species);
  const double m_t = species_A(test_species) * kProtonMassG;
  const double vt2 = 2.0 * E_erg / m_t;
  const double vf2 = 2.0 * Tf_erg / m_f;
  const double x = vt2 / vf2;
  const double sx = std::sqrt(x);
  const double ex = std::exp(-x);
  LpVelocityTerms v;
  v.mu = std::erf(sx) - 2.0 * std::sqrt(x / kPi) * ex;
  v.dmu = 2.0 * std::sqrt(x / kPi) * ex;
  const double m_r = m_t * m_f / (m_t + m_f);
  const double u2 = vt2 + vf2;
  const double u = std::sqrt(u2);
  const double p_perp =
      Z_t * Z_f * kElectronChargeEsu * kElectronChargeEsu / (m_r * u2);
  v.p_min =
      std::sqrt(p_perp * p_perp +
                (kHbarErgS / (2.0 * m_r * u)) *
                    (kHbarErgS / (2.0 * m_r * u)));
  v.collective = (x > 1.0) ? std::log(1.123 * std::sqrt(x)) : 0.0;
  v.pre = (Z_t * kElectronChargeEsu) * (Z_t * kElectronChargeEsu) / vt2;
  return v;
}

double lp_debye_length(const double Te_erg, const double ne) {
  return std::sqrt(Te_erg / (4.0 * kPi * ne * kElectronChargeEsu *
                             kElectronChargeEsu));
}

// Squared plasma frequency of a field species.
double lp_field_wp2(const double n_f, const double m_f, const double Z_f) {
  return 4.0 * kPi * n_f * (Z_f * kElectronChargeEsu) *
         (Z_f * kElectronChargeEsu) / m_f;
}

// dE/dx on a field species with n_f > 0 from its velocity terms, its squared
// plasma frequency, m_f / m_t and the Debye length of the field electrons.
double lp_dedx_from_terms(const LpVelocityTerms& v, const double lam_De,
                          const double wpf2, const double mass_ratio) {
  const double lnLb = std::max(2.0, std::log(lam_De / v.p_min));
  const double G = v.mu - mass_ratio * (v.dmu - (v.mu + v.dmu) / lnLb);
  return v.pre * wpf2 * (G * lnLb + v.collective);
}

// Field ions of the fuel: D, T, He3 in the order their stopping is summed.
struct FieldIon {
  double x;
  double m;
  double Z;
};

// The table entries of one product slot and one electron temperature: the
// ion fraction of the slowing-down from E0 to max(1.5 kTe, 1e-3 E0) by the
// trapezoid rule on kIntegralPoints energies, for every (Ti, ne). The
// electron stopping does not depend on Ti and the velocity terms of an ion
// field not on ne, so each is evaluated once per energy and reused; a field
// without ions adds 0 to the ion stopping, as its formula returns 0.
void build_partition_entries(PartitionTable& t, const int slot, const int i_te) {
  const ProductSpec p = product_spec(slot);
  const double m_t = species_A(p.species) * kProtonMassG;
  const double Te_keV = log_grid_value(t.te_min_keV, t.te_max_keV, t.n_te, i_te);
  const double E0 = p.E0_MeV * kMeVToErgLocal;
  const double Te_erg = Te_keV * kKeVToErg;
  const double E_min = std::max(1.5 * Te_erg, 1.0e-3 * E0);
  const double dE = (E0 - E_min) / static_cast<double>(kIntegralPoints - 1);
  const std::size_t n_E = static_cast<std::size_t>(kIntegralPoints);
  std::vector<double> E(n_E);
  E[0] = E0;
  for (std::size_t i = 1; i < n_E; ++i) {
    E[i] = E0 - dE * static_cast<double>(i);
  }
  const FieldIon ions[3] = {{t.x_D, species_A(kD) * kProtonMassG, species_Z(kD)},
                            {t.x_T, species_A(kT) * kProtonMassG, species_Z(kT)},
                            {t.x_He3, species_A(kHe3) * kProtonMassG, species_Z(kHe3)}};
  const double ion_mass_ratio[3] = {ions[0].m / m_t, ions[1].m / m_t, ions[2].m / m_t};
  const double zbar = t.x_D + t.x_T + 2.0 * t.x_He3;
  const std::size_t n_ne = static_cast<std::size_t>(t.n_ne);

  // Electron stopping for every (ne, E); the Debye length for every ne.
  std::vector<double> lam_De(n_ne);
  std::vector<double> dEdx_e(n_ne * n_E);
  {
    std::vector<LpVelocityTerms> ve(n_E);
    for (std::size_t k = 0; k < n_E; ++k) {
      ve[k] = lp_velocity_terms(p.species, E[k], Te_erg, kElectronMassG, 1.0);
    }
    const double electron_mass_ratio = kElectronMassG / m_t;
    for (std::size_t i_ne = 0; i_ne < n_ne; ++i_ne) {
      const double ne = log_grid_value(t.ne_min, t.ne_max, t.n_ne, static_cast<int>(i_ne));
      lam_De[i_ne] = lp_debye_length(Te_erg, ne);
      const double wpe2 = lp_field_wp2(ne, kElectronMassG, 1.0);
      double* row = dEdx_e.data() + i_ne * n_E;
      if (!(ne > 0.0)) {
        std::fill(row, row + n_E, 0.0);
        continue;
      }
      for (std::size_t k = 0; k < n_E; ++k) {
        row[k] = lp_dedx_from_terms(ve[k], lam_De[i_ne], wpe2, electron_mass_ratio);
      }
    }
  }

  std::vector<LpVelocityTerms> vi(3 * n_E);
  std::vector<double> dEdx_ions(n_E);
  std::vector<double> ratio(n_E);
  std::vector<unsigned char> positive(n_E);
  for (int i_ti = 0; i_ti < t.n_ti; ++i_ti) {
    const double Ti_keV = log_grid_value(t.ti_min_keV, t.ti_max_keV, t.n_ti, i_ti);
    const double Ti_erg = Ti_keV * kKeVToErg;
    for (int f = 0; f < 3; ++f) {
      if (!(ions[f].x > 0.0)) {
        continue;
      }
      for (std::size_t k = 0; k < n_E; ++k) {
        vi[static_cast<std::size_t>(f) * n_E + k] =
            lp_velocity_terms(p.species, E[k], Ti_erg, ions[f].m, ions[f].Z);
      }
    }
    for (std::size_t i_ne = 0; i_ne < n_ne; ++i_ne) {
      const double ne = log_grid_value(t.ne_min, t.ne_max, t.n_ne, static_cast<int>(i_ne));
      const double n_i_total = ne / zbar;
      std::fill(dEdx_ions.begin(), dEdx_ions.end(), 0.0);
      for (int f = 0; f < 3; ++f) {
        const double n_f = ions[f].x * n_i_total;
        if (!(n_f > 0.0)) {
          for (std::size_t k = 0; k < n_E; ++k) {
            dEdx_ions[k] += 0.0;
          }
          continue;
        }
        const double wpf2 = lp_field_wp2(n_f, ions[f].m, ions[f].Z);
        const LpVelocityTerms* vf = vi.data() + static_cast<std::size_t>(f) * n_E;
        for (std::size_t k = 0; k < n_E; ++k) {
          dEdx_ions[k] += lp_dedx_from_terms(vf[k], lam_De[i_ne], wpf2, ion_mass_ratio[f]);
        }
      }
      const double* e_row = dEdx_e.data() + i_ne * n_E;
      for (std::size_t k = 0; k < n_E; ++k) {
        const double total = dEdx_ions[k] + e_row[k];
        positive[k] = (total > 0.0) ? 1U : 0U;
        ratio[k] = positive[k] ? dEdx_ions[k] / total : 0.0;
      }
      double numerator = 0.0;
      double denominator = 0.0;
      for (std::size_t k = 1; k < n_E; ++k) {
        if (positive[k - 1] && positive[k]) {
          numerator += 0.5 * (ratio[k - 1] + ratio[k]) * dE;
          denominator += dE;
        }
      }
      t.f_ion[partition_index(t, slot, i_te, i_ti, static_cast<int>(i_ne))] =
          (denominator > 0.0) ? (numerator / denominator) : 0.0;
    }
  }
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

  // One task per (product slot, electron temperature), taken in order by a
  // pool of worker threads; every task writes its own entries.
  const int n_tasks = PartitionTable::kNumProductSlots * t.n_te;
  const int n_workers = std::max(
      1, std::min(n_tasks, static_cast<int>(std::thread::hardware_concurrency())));
  std::atomic<int> next_task{0};
  std::vector<std::future<void>> futures;
  futures.reserve(static_cast<std::size_t>(n_workers));
  for (int w = 0; w < n_workers; ++w) {
    futures.emplace_back(std::async(std::launch::async, [&]() {
#if defined(__linux__)
      // The driver builds the table while the time loop runs: its workers
      // take only CPU time that nothing else wants.
      sched_param idle_param{};
      pthread_setschedparam(pthread_self(), SCHED_IDLE, &idle_param);
#endif
      for (int task = next_task.fetch_add(1); task < n_tasks;
           task = next_task.fetch_add(1)) {
        build_partition_entries(t, task / t.n_te, task % t.n_te);
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
