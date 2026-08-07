#pragma once

#include <cmath>
#include <cstdint>

#include "materials/eos_device_table.cuh"

#ifdef __CUDACC__
#define TENRYU_PM_HD __host__ __device__ inline
#define TENRYU_PM_DEVICE __device__ inline
#else
#define TENRYU_PM_HD inline
#define TENRYU_PM_DEVICE inline
#endif

namespace tenryu::hydro::per_material {

struct PerMaterialAccessorView {
  const double* mass_per_material = nullptr;
  const double* Ee_per_material = nullptr;
  const double* Ei_per_material = nullptr;
  const double* volfrac = nullptr;
  const double* vol = nullptr;

  double* Te_per_material = nullptr;
  double* Ti_per_material = nullptr;
  std::uint8_t* Te_per_material_valid = nullptr;
  std::uint8_t* Ti_per_material_valid = nullptr;
  bool lazy_cache_te_m_enabled = false;

  double presence_threshold_volfrac = 1.0e-10;
  double presence_threshold_mass_density_g_per_cc = 1.0e-12;

  unsigned long long* d_counts = nullptr;

  int n_cells = 0;
  int n_mat = 0;
};

enum PerMaterialCounterIndex : int {
  kPerMaterialCounterEOSInverse = 0,
  kPerMaterialCounterLazyCacheHit = 1,
  kPerMaterialCounterLazyCacheMiss = 2,
  kPerMaterialCounterMixtureProjection = 3,
  kPerMaterialCounterEOSTableValidityViolation = 4,
  kPerMaterialCounterPresenceAbsent = 5,
  kPerMaterialCounterCount = 6,
};

TENRYU_PM_HD void record_per_material_counter(const PerMaterialAccessorView& v,
                                              const int counter_index) {
#if defined(__CUDA_ARCH__)
  if (v.d_counts != nullptr && counter_index >= 0) {
    atomicAdd(v.d_counts + counter_index, 1ULL);
  }
#else
  (void)v;
  (void)counter_index;
#endif
}

struct PerMaterialThermo {
  double T = 0.0;
  double pressure = 0.0;
  double cv = 0.0;
  double cs = 0.0;
  double energy = 0.0;
  double logT = 0.0;
  bool clamped = false;
  bool failed = false;
};

TENRYU_PM_HD double get_rho_per_material(const PerMaterialAccessorView& v,
                                         const int c,
                                         const int m) {
  if (c < 0 || m < 0 || c >= v.n_cells || m >= v.n_mat ||
      v.mass_per_material == nullptr || v.volfrac == nullptr || v.vol == nullptr) {
    return 0.0;
  }
  const int idx = c * v.n_mat + m;
  const double vf = v.volfrac[idx];
  const double V = v.vol[c];
  const double mass_m = v.mass_per_material[idx];
  if (!(vf > v.presence_threshold_volfrac) || !(V > 0.0) || !(mass_m > 0.0) ||
      !isfinite(vf) || !isfinite(V) || !isfinite(mass_m)) {
    record_per_material_counter(v, kPerMaterialCounterPresenceAbsent);
    return 0.0;
  }
  const double rho_m = mass_m / (vf * V);
  if (!(rho_m > v.presence_threshold_mass_density_g_per_cc) || !isfinite(rho_m)) {
    record_per_material_counter(v, kPerMaterialCounterPresenceAbsent);
    return 0.0;
  }
  return rho_m;
}

TENRYU_PM_HD double get_te(const PerMaterialThermo& electron) {
  return electron.T;
}

TENRYU_PM_HD double get_ti(const PerMaterialThermo& ion) {
  return ion.T;
}

TENRYU_PM_HD double get_pe(const PerMaterialThermo& electron) {
  return electron.pressure;
}

TENRYU_PM_HD double get_pi(const PerMaterialThermo& ion) {
  return ion.pressure;
}

TENRYU_PM_HD double get_cs(const PerMaterialThermo& electron,
                          const PerMaterialThermo& ion) {
  const double cs2 = electron.cs * electron.cs + ion.cs * ion.cs;
  return (cs2 > 0.0 && isfinite(cs2)) ? sqrt(cs2) : 0.0;
}

TENRYU_PM_HD double get_cv_e(const PerMaterialThermo& electron) {
  return electron.cv;
}

TENRYU_PM_HD double get_cv_i(const PerMaterialThermo& ion) {
  return ion.cv;
}

TENRYU_PM_HD double get_zbar_per_material(const double zbar_m) {
  return zbar_m;
}

TENRYU_PM_HD double ideal_species_cv(const double Zbar,
                                     const double A_amu,
                                     const double gamma) {
  constexpr double kProtonMass = 1.6726219e-24;
  constexpr double kEvToErg = 1.6022e-12;
  const double gm1 = gamma - 1.0;
  if (!(A_amu > 0.0) || !(gm1 > 0.0)) {
    return 0.0;
  }
  return Zbar * kEvToErg / (A_amu * kProtonMass * gm1);
}

TENRYU_PM_HD PerMaterialThermo ideal_thermo(const double rho,
                                            const double e_target,
                                            const double T_floor,
                                            const double Zbar,
                                            const double A_amu,
                                            const double gamma) {
  PerMaterialThermo out{};
  const double cv = ideal_species_cv(Zbar, A_amu, gamma);
  const double T_old = (cv > 0.0) ? (e_target / cv) : 0.0;
  out.T = fmax(T_old, fmax(T_floor, 1.0e-30));
  out.logT = log(out.T);
  out.cv = cv;
  out.energy = cv * out.T;
  out.pressure = (gamma - 1.0) * rho * out.energy;
  const double cs2 =
      (rho > 0.0 && out.pressure > 0.0) ? (gamma * out.pressure / rho) : 0.0;
  out.cs = (cs2 > 0.0) ? sqrt(cs2) : 0.0;
  out.clamped = out.T > T_old;
  out.failed = false;
  return out;
}

#ifdef __CUDACC__
TENRYU_PM_DEVICE bool use_table_reclosure(
    const materials::DeviceEOSTableView& tab) {
  return tab.n_rho > 0 && tab.n_T > 0 && tab.e_table != nullptr &&
         tab.P_table != nullptr && tab.cv_table != nullptr &&
         tab.supports_rho_e_reclosure != 0u;
}

TENRYU_PM_DEVICE bool table_forward_query_out_of_range(
    const materials::DeviceEOSTableView& tab,
    const double rho_m,
    const double T_m) {
  if (!use_table_reclosure(tab) || !(rho_m > 0.0) || !(T_m > 0.0) ||
      !isfinite(rho_m) || !isfinite(T_m)) {
    return false;
  }
  const double log_rho = log(rho_m);
  const double log_T = log(T_m);
  return log_rho < tab.log_rho_min || log_rho > tab.log_rho_max ||
         log_T < tab.log_T_min || log_T > tab.log_T_max;
}

TENRYU_PM_DEVICE PerMaterialThermo table_forward_thermo(
    const materials::DeviceEOSTableView& table,
    const double rho_m,
    const double T_m,
    const double T_floor,
    const double Zbar_m,
    const double A_m,
    const bool low_density_extrap) {
  PerMaterialThermo out{};
  out.T = fmax(T_m, fmax(T_floor, 1.0e-30));
  out.logT = log(out.T);
  const tenryu::materials::RhoBracket rb =
      tenryu::materials::find_rho_bracket(table, rho_m);
  out.pressure = tenryu::materials::device_eos_pressure_extrap(
      table, rb, rho_m, out.T, Zbar_m, A_m, low_density_extrap);
  out.energy = tenryu::materials::device_eos_energy_extrap(
      table, rb, rho_m, out.T, Zbar_m, A_m, low_density_extrap);
  out.cv = tenryu::materials::device_eos_cv_extrap(
      table, rb, rho_m, out.T, Zbar_m, A_m, low_density_extrap);
  out.cs = tenryu::materials::device_eos_sound_speed(
      table, rb, out.logT, rho_m, out.cv);
  out.clamped = false;
  out.failed = false;
  return out;
}

TENRYU_PM_DEVICE PerMaterialThermo table_inverse_thermo(
    const materials::DeviceEOSTableView& table,
    const double rho_m,
    const double e_m,
    const double T_floor,
    const double Zbar_m,
    const double A_m,
    const bool low_density_extrap) {
  const auto inv = tenryu::materials::device_inverse_reclose_with_low_density_extrap(
      table, rho_m, e_m, T_floor, Zbar_m, A_m, low_density_extrap);
  PerMaterialThermo out{};
  out.T = inv.T;
  out.logT = inv.logT;
  out.pressure = inv.pressure;
  out.energy = inv.energy;
  out.cv = inv.cv;
  const tenryu::materials::RhoBracket rb =
      tenryu::materials::find_rho_bracket(table, rho_m);
  out.cs = tenryu::materials::device_eos_sound_speed(
      table, rb, inv.logT, rho_m, inv.cv);
  out.clamped = inv.lower_clamp != 0 || inv.upper_clamp != 0;
  out.failed = inv.bracket_failure != 0;
  return out;
}

TENRYU_PM_DEVICE PerMaterialThermo get_electron_thermo_per_material(
    const PerMaterialAccessorView& v,
    const materials::DeviceEOSTableView& electron_view_m,
    const int c,
    const int m,
    const double Zbar_m,
    const double A_m,
    const double Te_floor,
    const bool low_density_extrap,
    const double ideal_gas_gamma = 5.0 / 3.0) {
  const int idx = c * v.n_mat + m;
  const double rho_m = get_rho_per_material(v, c, m);
  const double mass_m = v.mass_per_material[idx];
  const double e_m =
      (mass_m > 0.0 && v.Ee_per_material != nullptr) ? (v.Ee_per_material[idx] / mass_m) : 0.0;
  const bool cache_active =
      v.lazy_cache_te_m_enabled && v.Te_per_material != nullptr &&
      v.Te_per_material_valid != nullptr;
  if (cache_active && v.Te_per_material_valid[idx] != 0u) {
    if (v.d_counts != nullptr) {
      atomicAdd(v.d_counts + kPerMaterialCounterLazyCacheHit, 1ULL);
    }
    const double cached_T = v.Te_per_material[idx];
    if (use_table_reclosure(electron_view_m)) {
      if (table_forward_query_out_of_range(electron_view_m, rho_m, cached_T)) {
        record_per_material_counter(v, kPerMaterialCounterEOSTableValidityViolation);
      }
      return table_forward_thermo(electron_view_m,
                                  rho_m,
                                  cached_T,
                                  Te_floor,
                                  Zbar_m,
                                  A_m,
                                  low_density_extrap);
    }
    return ideal_thermo(rho_m, e_m, Te_floor, Zbar_m, A_m, ideal_gas_gamma);
  }

  if (cache_active && v.d_counts != nullptr) {
    atomicAdd(v.d_counts + kPerMaterialCounterLazyCacheMiss, 1ULL);
  }
  if (v.d_counts != nullptr) {
    atomicAdd(v.d_counts + kPerMaterialCounterEOSInverse, 1ULL);
  }

  PerMaterialThermo thermo{};
  if (use_table_reclosure(electron_view_m)) {
    thermo = table_inverse_thermo(
        electron_view_m, rho_m, e_m, Te_floor, Zbar_m, A_m, low_density_extrap);
    if (thermo.clamped || thermo.failed) {
      record_per_material_counter(v, kPerMaterialCounterEOSTableValidityViolation);
    }
  } else {
    thermo = ideal_thermo(rho_m, e_m, Te_floor, Zbar_m, A_m, ideal_gas_gamma);
  }
  if (cache_active) {
    v.Te_per_material[idx] = thermo.T;
    v.Te_per_material_valid[idx] = 1u;
  }
  return thermo;
}

TENRYU_PM_DEVICE PerMaterialThermo get_ion_thermo_per_material(
    const PerMaterialAccessorView& v,
    const materials::DeviceEOSTableView& ion_view_m,
    const int c,
    const int m,
    const double A_m,
    const double Ti_floor,
    const bool low_density_extrap,
    const double ideal_gas_gamma = 5.0 / 3.0) {
  const int idx = c * v.n_mat + m;
  const double rho_m = get_rho_per_material(v, c, m);
  const double mass_m = v.mass_per_material[idx];
  const double e_m =
      (mass_m > 0.0 && v.Ei_per_material != nullptr) ? (v.Ei_per_material[idx] / mass_m) : 0.0;
  const bool cache_active =
      v.lazy_cache_te_m_enabled && v.Ti_per_material != nullptr &&
      v.Ti_per_material_valid != nullptr;
  if (cache_active && v.Ti_per_material_valid[idx] != 0u) {
    const double cached_T = v.Ti_per_material[idx];
    if (use_table_reclosure(ion_view_m)) {
      if (table_forward_query_out_of_range(ion_view_m, rho_m, cached_T)) {
        record_per_material_counter(v, kPerMaterialCounterEOSTableValidityViolation);
      }
      return table_forward_thermo(
          ion_view_m, rho_m, cached_T, Ti_floor, 1.0, A_m, low_density_extrap);
    }
    return ideal_thermo(rho_m, e_m, Ti_floor, 1.0, A_m, ideal_gas_gamma);
  }

  PerMaterialThermo thermo{};
  if (use_table_reclosure(ion_view_m)) {
    thermo = table_inverse_thermo(
        ion_view_m, rho_m, e_m, Ti_floor, 1.0, A_m, low_density_extrap);
    if (thermo.clamped || thermo.failed) {
      record_per_material_counter(v, kPerMaterialCounterEOSTableValidityViolation);
    }
  } else {
    thermo = ideal_thermo(rho_m, e_m, Ti_floor, 1.0, A_m, ideal_gas_gamma);
  }
  if (cache_active) {
    v.Ti_per_material[idx] = thermo.T;
    v.Ti_per_material_valid[idx] = 1u;
  }
  return thermo;
}
#endif

}  // namespace tenryu::hydro::per_material

#undef TENRYU_PM_HD
#undef TENRYU_PM_DEVICE
