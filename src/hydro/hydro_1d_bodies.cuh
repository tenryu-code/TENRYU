#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "core/constants.hpp"
#include "hydro/boundary.hpp"
#include "materials/eos_device.cuh"
#include "materials/eos_device_table.cuh"
#include "materials/eos_rho_e_device.cuh"
#include "materials/helmholtz_jet_device.cuh"
#include "materials/helmholtz_spline_device.cuh"
#include "materials/mie_gruneisen_device.cuh"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro::persistent_1d {

using tenryu::mesh::geometry_1d_face_area;
using tenryu::mesh::geometry_1d_shell_volume;

constexpr double kFourPi = 12.566370614359172953850573533118;
constexpr double kFourPiOverThree =
    4.188790204786390984616857844372670512262892532500141094646;
constexpr double kEvToErg = tenryu::core::constants::eV_to_erg;
constexpr double kProtonMass = tenryu::core::constants::proton_mass;
constexpr double kShockPressureJumpThreshold = 0.3;
constexpr double kShockDensityJumpThreshold = 0.05;
constexpr double kShockRhConsistencyThreshold = 0.5;
constexpr double kCompMachScale = 0.05;
constexpr double kOscillationThreshold = 0.2;
constexpr double kShockSupportFloor = 0.25;
constexpr double kMildCompressionAlpha = 0.25;
constexpr double kSensorEps = 1.0e-30;
constexpr double kCflSensorEps = 1.0e-30;
constexpr double kDiagnosticIdealGasGamma = 5.0 / 3.0;
constexpr int kExactOverrideNone = 0;
constexpr int kExactOverridePressure = 1;
constexpr int kExactOverrideSoundSpeed = 2;
constexpr int kExactOverrideTemperature = 3;
constexpr int kExactOverrideCv = 4;
constexpr int kExactOverrideTempReclosure = 5;
constexpr int kExactOverridePressureAndCs = 6;
constexpr int kExactOverridePTCs = 7;  // P+T+cs exact, cv from table
constexpr int kExactOverrideNoWriteback = 8;  // skip ee writeback

__host__ __device__ inline bool energy_needs_repair(const double e) {
  return !isfinite(e) || e < 0.0;
}

__host__ __device__ inline double energy_inverse_input(const double e) {
  return isfinite(e) ? e : 0.0;
}

__host__ __device__ inline bool enable_table_eos_writeback(
    const bool eos_writeback,
    const int exact_override_kind) {
  return eos_writeback && exact_override_kind != kExactOverrideNoWriteback;
}

__host__ __device__ inline double ideal_gas_cv_i_mass(const double gamma,
                                                      const double A) {
  if (!(gamma > 1.0) || !(A > 0.0)) {
    return 0.0;
  }
  return kEvToErg / (A * kProtonMass * (gamma - 1.0));
}

__host__ __device__ inline double ideal_gas_cv_e_mass(const double gamma,
                                                      const double A,
                                                      const double Z,
                                                      const double cv_e_override,
                                                      const double rho) {
  if (cv_e_override > 0.0) {
    return cv_e_override / fmax(rho, 1.0e-30);
  }
  return fmax(Z, 0.0) * ideal_gas_cv_i_mass(gamma, A);
}

__host__ __device__ inline double diagnostic_cv_i_mass(const double A) {
  if (!(A > 0.0)) {
    return 0.0;
  }
  return 1.5 * kEvToErg / (A * kProtonMass);
}

__host__ __device__ inline double diagnostic_cv_e_mass(const double A,
                                                       const double Z) {
  return fmax(Z, 0.0) * diagnostic_cv_i_mass(A);
}

__device__ inline double atomic_add_double(double* address, const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(
                                  static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline void record_inverse_reclose_status(
    const tenryu::materials::DeviceEOSInverseRecloseResult& inv,
    int* __restrict__ inverse_clamp_count,
    int* __restrict__ inverse_failure_count) {
  if (inv.bracket_failure != 0 && inverse_failure_count != nullptr) {
    atomicAdd(inverse_failure_count, 1);
  }
  if ((inv.lower_clamp != 0 || inv.upper_clamp != 0) &&
      inverse_clamp_count != nullptr) {
    atomicAdd(inverse_clamp_count, 1);
  }
}

__device__ inline void apply_exact_override_2t(
    const int exact_override_kind,
    const double rho,
    const double A,
    const double Z,
    const double te_floor,
    const double ti_floor,
    const double ee,
    const double ei,
    double* Te,
    double* Ti,
    double* Pe,
    double* Pi) {
  if (exact_override_kind == kExactOverridePressure) {
    *Pe = (kDiagnosticIdealGasGamma - 1.0) * rho * ee;
    *Pi = (kDiagnosticIdealGasGamma - 1.0) * rho * ei;
    return;
  }
  if (exact_override_kind == kExactOverridePressureAndCs) {
    *Pe = (kDiagnosticIdealGasGamma - 1.0) * rho * ee;
    *Pi = (kDiagnosticIdealGasGamma - 1.0) * rho * ei;
    return;
  }
  if (exact_override_kind == kExactOverridePTCs) {
    *Pe = (kDiagnosticIdealGasGamma - 1.0) * rho * ee;
    *Pi = (kDiagnosticIdealGasGamma - 1.0) * rho * ei;
    const double cv_i = diagnostic_cv_i_mass(A);
    const double cv_e = diagnostic_cv_e_mass(A, Z);
    *Te = (ee > 0.0 && cv_e > 0.0) ? fmax(ee / cv_e, te_floor) : te_floor;
    *Ti = (ei > 0.0 && cv_i > 0.0) ? fmax(ei / cv_i, ti_floor) : ti_floor;
    return;
  }
  if (exact_override_kind == kExactOverrideTemperature) {
    const double cv_i = diagnostic_cv_i_mass(A);
    const double cv_e = diagnostic_cv_e_mass(A, Z);
    double te = te_floor;
    double ti = ti_floor;
    if (ee > 0.0 && cv_e > 0.0) {
      te = fmax(ee / cv_e, te_floor);
    }
    if (ei > 0.0 && cv_i > 0.0) {
      ti = fmax(ei / cv_i, ti_floor);
    }
    *Te = te;
    *Ti = ti;
  }
}

__device__ inline double apply_exact_sound_speed_override(
    const int exact_override_kind,
    const double cs_value,
    const double rho,
    const double Pe,
    const double Pi) {
  if (exact_override_kind != kExactOverrideSoundSpeed &&
      exact_override_kind != kExactOverridePressureAndCs &&
      exact_override_kind != kExactOverridePTCs) {
    return cs_value;
  }
  const double rho_safe = fmax(rho, 1.0e-30);
  const double cs2 = kDiagnosticIdealGasGamma * (Pe + Pi) / rho_safe;
  return sqrt(fmax(cs2, 0.0));
}

__device__ __forceinline__ double clamp01_device(const double x) {
  return fmin(1.0, fmax(0.0, x));
}

__device__ inline void recompute_geometry_1d_kernel_body(
    const int i,
    double* __restrict__ cell_vol,
    double* __restrict__ cell_centroid_r,
    double* __restrict__ cell_centroid_z,
    const double* __restrict__ node_r,
    const int /*nr*/,
    const int geom) {
  const double r0 = node_r[i];
  const double r1 = node_r[i + 1];
  const double dr = r1 - r0;
  const double vol =
      (geom == 0)
          ? (kFourPiOverThree * dr * (r1 * r1 + r1 * r0 + r0 * r0))
          : geometry_1d_shell_volume(geom, r0, r1);

  cell_vol[i] = vol;
  cell_centroid_r[i] = 0.5 * (r0 + r1);
  cell_centroid_z[i] = 0.0;
}

__device__ inline void compute_density_kernel_body(
    const int i,
    double* __restrict__ rho,
    const double* __restrict__ mass,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ cell_is_void,
    const int /*n_cells*/,
    const double rho_floor,
    int* __restrict__ rho_clamp_count) {
  const double rho_raw = (vol[i] > 0.0) ? (mass[i] / vol[i]) : 0.0;
  const double floor = fmax(rho_floor, 0.0);
  const double rho_new = fmax(rho_raw, floor);
  rho[i] = rho_new;
  const bool is_void = (cell_is_void != nullptr &&
                        cell_is_void[i] != static_cast<std::uint8_t>(0));
  if (!is_void && rho_clamp_count != nullptr && rho_raw < floor) {
    atomicAdd(rho_clamp_count, 1);
  }
}

__device__ inline void enforce_1t_closure_ideal_kernel_body(
    const int i,
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const double fallback_z,
    const double cv_e_override,
    const double te_floor,
    double* __restrict__ cv_e_out,
    double* __restrict__ cv_i_out) {
  const double rho_i = fmax(rho[i], 1.0e-30);
  const double gamma = fmax(gamma_eff[i], 1.0 + 1.0e-12);
  const double A = fmax(A_eff[i], 1.0e-12);
  ee[i] += ei[i];
  const double z = (zbar != nullptr) ? fmax(zbar[i], 0.0) : fmax(fallback_z, 0.0);
  const double cv_i = ideal_gas_cv_i_mass(gamma, A);
  const double cv_e = ideal_gas_cv_e_mass(gamma, A, z, cv_e_override, rho_i);
  const double cv_total = cv_i + cv_e;
  const double e = fmax(ee[i], 0.0);
  ee[i] = e;
  ei[i] = 0.0;

  double te = te_floor;
  if (e > 0.0 && cv_total > 0.0) {
    te = fmax(e / cv_total, te_floor);
  }

  Te[i] = te;
  Ti[i] = te;
  Pe[i] = (gamma - 1.0) * rho[i] * e;
  Pi[i] = 0.0;
  if (cv_e_out != nullptr) {
    cv_e_out[i] = cv_total;
  }
  if (cv_i_out != nullptr) {
    cv_i_out[i] = 0.0;
  }
}

__device__ inline void enforce_2t_closure_kernel_body(
    const int i,
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const int n_cells,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const double fallback_z,
    const double cv_e_override,
    const double te_floor,
    const double ti_floor,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele,
    const tenryu::materials::EOSRhoEDeviceView tab_total_rho_e,
    const tenryu::materials::HelmholtzSplineDeviceView spline_total,
    const tenryu::materials::HelmholtzJetDeviceView jet_total,
    const tenryu::materials::MieGruneisenDeviceView mie_gruneisen,
    const bool use_helmholtz,
    const bool use_helmholtz_jet,
    const bool use_rho_e_table,
    const bool use_mie_gruneisen,
    const bool use_exact_ideal_gas,
    const bool eos_writeback,
    const bool preserve_table_energy,
    const bool energy_authoritative,
    int* __restrict__ inverse_clamp_count,
    int* __restrict__ inverse_failure_count,
    const int exact_override_kind,
    double* __restrict__ cv_e_out,
    double* __restrict__ cv_i_out) {
  const double rho_i = fmax(rho[i], 1.0e-30);
  // Per-cell effective ideal-gas properties (State::ensure_cell_material_props
  // is the single source of truth; historic behavior used materials[0] for
  // every cell, which poisoned multi-material closures — see BUG-1).
  const double gamma = fmax(gamma_eff[i], 1.0 + 1.0e-12);
  const double A = fmax(A_eff[i], 1.0e-12);
  const bool allow_table_energy_writeback =
      !preserve_table_energy && enable_table_eos_writeback(eos_writeback, exact_override_kind);
  if (use_exact_ideal_gas) {
    const double z = (zbar != nullptr) ? fmax(zbar[i], 0.0) : fmax(fallback_z, 0.0);
    const double cv_i = ideal_gas_cv_i_mass(gamma, A);
    const double cv_e = ideal_gas_cv_e_mass(gamma, A, z, cv_e_override, rho_i);
    double e_i = fmax(ei[i], 0.0);
    double e_e = fmax(ee[i], 0.0);
    double ti = ti_floor;
    double te = te_floor;

    if (cv_i > 0.0) {
      ti = fmax(e_i / cv_i, ti_floor);
    } else {
      e_i = 0.0;
    }
    if (cv_e > 0.0) {
      te = fmax(e_e / cv_e, te_floor);
    } else {
      e_e = 0.0;
    }

    ei[i] = e_i;
    ee[i] = e_e;
    Ti[i] = ti;
    Te[i] = te;
    Pi[i] = (gamma - 1.0) * rho[i] * e_i;
    Pe[i] = (gamma - 1.0) * rho[i] * e_e;
    if (cv_i_out != nullptr) {
      cv_i_out[i] = cv_i;
    }
    if (cv_e_out != nullptr) {
      cv_e_out[i] = cv_e;
    }
    return;
  }
  if (use_mie_gruneisen && mie_gruneisen.n_rho > 0) {
    const double e_i = fmax(energy_inverse_input(ei[i]), 0.0);
    const double e_e = fmax(energy_inverse_input(ee[i]), 0.0);
    ei[i] = e_i;
    ee[i] = e_e;
    Pi[i] = tenryu::materials::device_mie_gruneisen_pressure(mie_gruneisen, rho_i, e_i, false);
    Pe[i] = tenryu::materials::device_mie_gruneisen_pressure(mie_gruneisen, rho_i, e_e, true);
    return;
  }
  if (tab_ion.n_rho > 0 && rho_i >= exp(tab_ion.log_rho_min)) {
    const double e_i_raw = ei[i];
    const double e_e_raw = ee[i];
    const bool repair_i = energy_needs_repair(e_i_raw);
    const bool repair_e = energy_needs_repair(e_e_raw);
    const double e_i = energy_inverse_input(e_i_raw);
    const double e_e = energy_inverse_input(e_e_raw);
    const auto inv_i =
        energy_authoritative
            ? tenryu::materials::device_inverse_reclose_with_high_t_tail(
                  tab_ion, rho_i, e_i, ti_floor)
            : tenryu::materials::device_inverse_reclose(tab_ion, rho_i, e_i,
                                                        ti_floor);
    const auto inv_e =
        energy_authoritative
            ? tenryu::materials::device_inverse_reclose_with_high_t_tail(
                  tab_ele, rho_i, e_e, te_floor)
            : tenryu::materials::device_inverse_reclose(tab_ele, rho_i, e_e,
                                                        te_floor);
    record_inverse_reclose_status(inv_i, inverse_clamp_count, inverse_failure_count);
    record_inverse_reclose_status(inv_e, inverse_clamp_count, inverse_failure_count);
    const bool clamp_veto_i = energy_authoritative && !repair_i &&
        inv_i.bracket_failure == 0 &&
        (inv_i.lower_clamp != 0 || inv_i.upper_clamp != 0);
    const bool clamp_veto_e = energy_authoritative && !repair_e &&
        inv_e.bracket_failure == 0 &&
        (inv_e.lower_clamp != 0 || inv_e.upper_clamp != 0);
    if (!clamp_veto_i &&
        (allow_table_energy_writeback || repair_i ||
         inv_i.bracket_failure != 0 || inv_i.lower_clamp != 0 ||
         inv_i.upper_clamp != 0)) {
      ei[i] = inv_i.energy;
    }
    if (!clamp_veto_e &&
        (allow_table_energy_writeback || repair_e ||
         inv_e.bracket_failure != 0 || inv_e.lower_clamp != 0 ||
         inv_e.upper_clamp != 0)) {
      ee[i] = inv_e.energy;
    }
    Ti[i] = inv_i.T;
    Te[i] = inv_e.T;
    const double pi_legacy = inv_i.pressure;
    const double pe_legacy = inv_e.pressure;
    if (use_helmholtz && spline_total.n_rho > 0 &&
        rho_i >= exp(spline_total.log_rho_min)) {
      const double e_total = ei[i] + ee[i];
      const double T_total =
          fmax(tenryu::materials::device_helmholtz_T_from_e(spline_total, rho_i, e_total),
               fmax(te_floor, ti_floor));
      const double p_total =
          tenryu::materials::device_helmholtz_eval_thermo(spline_total,
                                                          rho_i,
                                                          log(fmax(T_total, 1.0e-30)))
              .pressure;
      const double p_legacy = pi_legacy + pe_legacy;
      if (fabs(p_legacy) > 1.0e-30 && isfinite(p_legacy)) {
        const double scale = p_total / p_legacy;
        Pi[i] = scale * pi_legacy;
        Pe[i] = scale * pe_legacy;
      } else {
        Pi[i] = 0.0;
        Pe[i] = p_total;
      }
    } else if (use_helmholtz_jet && jet_total.n_rho > 0 &&
               rho_i >= exp(jet_total.log_rho_min)) {
      const double e_total = ei[i] + ee[i];
      const double T_total =
          fmax(tenryu::materials::device_helmholtz_jet_T_from_e(jet_total, rho_i, e_total),
               fmax(te_floor, ti_floor));
      const double p_total =
          tenryu::materials::device_helmholtz_jet_eval_thermo(
              jet_total, rho_i, log(fmax(T_total, 1.0e-30)))
              .pressure;
      const double p_legacy = pi_legacy + pe_legacy;
      if (fabs(p_legacy) > 1.0e-30 && isfinite(p_legacy)) {
        const double scale = p_total / p_legacy;
        Pi[i] = scale * pi_legacy;
        Pe[i] = scale * pe_legacy;
      } else {
        Pi[i] = 0.0;
        Pe[i] = p_total;
      }
    } else if (use_rho_e_table && tab_total_rho_e.n_rho > 0 &&
               rho_i >= tenryu::materials::eos_rho_e_rho_min(tab_total_rho_e)) {
      const double e_total = ei[i] + ee[i];
      const double p_total =
          tenryu::materials::device_eos_rho_e_eval(tab_total_rho_e, rho_i, e_total).P;
      const double p_legacy = pi_legacy + pe_legacy;
      if (isfinite(p_total) && fabs(p_legacy) > 1.0e-30 && isfinite(p_legacy)) {
        const double scale = p_total / p_legacy;
        Pi[i] = scale * pi_legacy;
        Pe[i] = scale * pe_legacy;
      } else if (isfinite(p_total)) {
        Pi[i] = 0.0;
        Pe[i] = p_total;
      } else {
        Pi[i] = pi_legacy;
        Pe[i] = pe_legacy;
      }
    } else {
      Pi[i] = pi_legacy;
      Pe[i] = pe_legacy;
    }
    if (cv_i_out != nullptr) {
      cv_i_out[i] = inv_i.cv;
    }
    if (cv_e_out != nullptr) {
      cv_e_out[i] = inv_e.cv;
    }
    apply_exact_override_2t(exact_override_kind, rho[i], A, fallback_z, te_floor, ti_floor,
                            ee[i], ei[i], &Te[i], &Ti[i], &Pe[i], &Pi[i]);
    return;
  }

  const double z = (zbar != nullptr) ? fmax(zbar[i], 0.0) : fmax(fallback_z, 0.0);
  const double cv_i = ideal_gas_cv_i_mass(gamma, A);
  const double cv_e = ideal_gas_cv_e_mass(gamma, A, z, cv_e_override, rho_i);

  double e_i = fmax(ei[i], 0.0);
  double e_e = fmax(ee[i], 0.0);
  double ti = ti_floor;
  double te = te_floor;

  if (cv_i > 0.0) {
    ti = fmax(tenryu::materials::eos_temperature_ion(e_i, cv_i), ti_floor);
  } else {
    e_i = 0.0;
  }
  if (cv_e > 0.0) {
    te = fmax(tenryu::materials::eos_temperature_electron(e_e, cv_e), te_floor);
  } else {
    e_e = 0.0;
  }

  ei[i] = e_i;
  ee[i] = e_e;
  Ti[i] = ti;
  Te[i] = te;
  Pi[i] = tenryu::materials::eos_pressure_ion(rho[i], e_i, gamma);
  Pe[i] = tenryu::materials::eos_pressure_electron(rho[i], e_e, gamma);
  if (cv_i_out != nullptr) {
    cv_i_out[i] = cv_i;
  }
  if (cv_e_out != nullptr) {
    cv_e_out[i] = cv_e;
  }
  apply_exact_override_2t(exact_override_kind, rho[i], A, fallback_z, te_floor, ti_floor, ee[i],
                          ei[i], &Te[i], &Ti[i], &Pe[i], &Pi[i]);
}

__device__ inline void compute_sound_speed_1t_ideal_kernel_body(
    const int i,
    double* __restrict__ cs,
    const double* __restrict__ ee,
    const double* __restrict__ gamma_eff) {
  const double gamma = fmax(gamma_eff[i], 1.0 + 1.0e-12);
  const double e = fmax(ee[i], 0.0);
  cs[i] = sqrt(fmax(gamma * (gamma - 1.0) * e, 0.0));
}

__device__ inline void compute_sound_speed_2t_kernel_body(
    const int i,
    double* __restrict__ cs,
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele,
    const tenryu::materials::EOSRhoEDeviceView tab_total_rho_e,
    const tenryu::materials::HelmholtzSplineDeviceView spline_total,
    const tenryu::materials::HelmholtzJetDeviceView jet_total,
    const tenryu::materials::MieGruneisenDeviceView mie_gruneisen,
    const double* __restrict__ cv_i_arr,
    const double* __restrict__ cv_e_arr,
    const int n_cells,
    const bool use_helmholtz,
    const bool use_helmholtz_jet,
    const bool use_rho_e_table,
    const bool use_mie_gruneisen,
    const bool use_exact_ideal_gas,
    const int exact_override_kind,
    const double* __restrict__ gamma_eff) {
  const double rho_i = fmax(rho[i], 1.0e-30);
  const double gamma = fmax(gamma_eff[i], 1.0 + 1.0e-12);
  if (use_exact_ideal_gas) {
    const double cs2 = gamma * (Pe[i] + Pi[i]) / rho_i;
    cs[i] = sqrt(fmax(cs2, 0.0));
    return;
  }
  if (use_mie_gruneisen && mie_gruneisen.n_rho > 0) {
    cs[i] = tenryu::materials::device_mie_gruneisen_sound_speed_2t(
        mie_gruneisen, rho_i, fmax(energy_inverse_input(ei[i]), 0.0),
        fmax(energy_inverse_input(ee[i]), 0.0), Pi[i], Pe[i]);
    return;
  }
  if (use_helmholtz && spline_total.n_rho > 0 && rho_i >= exp(spline_total.log_rho_min)) {
    const double e_total = ee[i] + ei[i];
    const double T_total = fmax(
        tenryu::materials::device_helmholtz_T_from_e(spline_total, rho_i, e_total), 1.0e-30);
    cs[i] = tenryu::materials::device_helmholtz_eval_thermo(spline_total,
                                                            rho_i,
                                                            log(T_total))
                .sound_speed;
    cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
    return;
  }
  if (use_helmholtz_jet && jet_total.n_rho > 0 && rho_i >= exp(jet_total.log_rho_min)) {
    const double e_total = ee[i] + ei[i];
    const double T_total = fmax(
        tenryu::materials::device_helmholtz_jet_T_from_e(jet_total, rho_i, e_total), 1.0e-30);
    cs[i] = tenryu::materials::device_helmholtz_jet_eval_thermo(jet_total,
                                                                rho_i,
                                                                log(T_total))
                .sound_speed;
    cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
    return;
  }
  if (use_rho_e_table && tab_total_rho_e.n_rho > 0 &&
      rho_i >= tenryu::materials::eos_rho_e_rho_min(tab_total_rho_e)) {
    const double e_total = ee[i] + ei[i];
    const auto thermo =
        tenryu::materials::device_eos_rho_e_eval(tab_total_rho_e, rho_i, e_total);
    cs[i] = sqrt(fmax(thermo.cs2, 0.0));
    cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
    return;
  }
  if (tab_ion.n_rho > 0 && rho_i >= exp(tab_ion.log_rho_min)) {
    const auto rb_ion = tenryu::materials::find_rho_bracket(tab_ion, rho_i);
    const auto rb_ele = tenryu::materials::find_rho_bracket(tab_ele, rho_i);
    const double logTi = log(fmax(Ti[i], 1.0e-30));
    const double logTe = log(fmax(Te[i], 1.0e-30));
    const double cv_i =
        (cv_i_arr != nullptr)
            ? fmax(cv_i_arr[i], 0.0)
            : fmax(tenryu::materials::device_eos_cv(tab_ion, rb_ion, logTi), 0.0);
    const double cv_e =
        (cv_e_arr != nullptr)
            ? fmax(cv_e_arr[i], 0.0)
            : fmax(tenryu::materials::device_eos_cv(tab_ele, rb_ele, logTe), 0.0);
    const double cs_ion =
        tenryu::materials::device_eos_sound_speed(tab_ion, rb_ion, logTi, rho_i, cv_i);
    const double cs_ele =
        tenryu::materials::device_eos_sound_speed(tab_ele, rb_ele, logTe, rho_i, cv_e);
    cs[i] = sqrt(cs_ion * cs_ion + cs_ele * cs_ele);
    cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
    return;
  }

  cs[i] = tenryu::materials::eos_sound_speed_2t(gamma, Pe[i], gamma, Pi[i], rho[i]);
  cs[i] = apply_exact_sound_speed_override(exact_override_kind, cs[i], rho_i, Pe[i], Pi[i]);
}

__device__ __forceinline__ double total_pressure_device(
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const int i) {
  return Pe[i] + Pi[i];
}

__device__ __forceinline__ double interface_developed_shock_weight_1d_device(
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int left_cell,
    const int right_cell) {
  const double p_left = total_pressure_device(Pe, Pi, left_cell);
  const double p_right = total_pressure_device(Pe, Pi, right_cell);
  const double rho_left = rho[left_cell];
  const double rho_right = rho[right_cell];
  const double dp = p_right - p_left;
  const double drho = rho_right - rho_left;
  const double Jp = fabs(dp) / fmax(fmin(p_left, p_right), kSensorEps);
  const double Jrho =
      fabs(drho) / fmax(fmin(rho_left, rho_right), kSensorEps);
  const double cs2_face =
      0.5 * (cs[left_cell] * cs[left_cell] + cs[right_cell] * cs[right_cell]);
  const double Z = fabs(dp) / fmax(cs2_face * fabs(drho), kSensorEps);
  const bool same_sign = (dp * drho > 0.0);
  return (same_sign && Jp >= kShockPressureJumpThreshold &&
          Jrho >= kShockDensityJumpThreshold &&
          Z >= kShockRhConsistencyThreshold)
             ? 1.0
             : 0.0;
}

__device__ __forceinline__ double interface_shock_support_weight_1d_device(
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int left_cell,
    const int right_cell) {
  if (interface_developed_shock_weight_1d_device(Pe, Pi, rho, cs, left_cell, right_cell) >
      0.0) {
    return 1.0;
  }

  const double p_left = total_pressure_device(Pe, Pi, left_cell);
  const double p_right = total_pressure_device(Pe, Pi, right_cell);
  const double rho_left = rho[left_cell];
  const double rho_right = rho[right_cell];
  const double dp = p_right - p_left;
  const double drho = rho_right - rho_left;
  const double Jp = fabs(dp) / fmax(fmin(p_left, p_right), kSensorEps);
  const double Jrho =
      fabs(drho) / fmax(fmin(rho_left, rho_right), kSensorEps);
  const double cs2_face =
      0.5 * (cs[left_cell] * cs[left_cell] + cs[right_cell] * cs[right_cell]);
  const double Z = fabs(dp) / fmax(cs2_face * fabs(drho), kSensorEps);
  return (Jp >= kShockPressureJumpThreshold &&
          Jrho < kShockDensityJumpThreshold &&
          Z >= kShockRhConsistencyThreshold)
             ? 1.0
             : 0.0;
}

__device__ __forceinline__ double interface_mild_compression_weight_1d_device(
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int left_cell,
    const int right_cell) {
  const double p_left = total_pressure_device(Pe, Pi, left_cell);
  const double p_right = total_pressure_device(Pe, Pi, right_cell);
  const double rho_left = rho[left_cell];
  const double rho_right = rho[right_cell];
  const double dp = p_right - p_left;
  const double drho = rho_right - rho_left;
  const double Jp = fabs(dp) / fmax(fmin(p_left, p_right), kSensorEps);
  const double cs2_face =
      0.5 * (cs[left_cell] * cs[left_cell] + cs[right_cell] * cs[right_cell]);
  const double Z = fabs(dp) / fmax(cs2_face * fabs(drho), kSensorEps);
  const bool same_sign = (dp * drho > 0.0);
  const double W_rh =
      (same_sign && Z >= kShockRhConsistencyThreshold) ? 1.0 : 0.0;
  const double W_jp_soft =
      clamp01_device(Jp / kShockPressureJumpThreshold);
  return W_rh * W_jp_soft;
}

__device__ __forceinline__ double compute_node_sigma_1d_device(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const int j,
    const int n_nodes,
    const double J) {
  double r_left = 0.0;
  double u_left = 0.0;
  double r_right = 0.0;
  double u_right = 0.0;
  if (j == 0) {
    r_left = -node_r[1];
    u_left = -node_u[1];
  } else {
    r_left = node_r[j - 1];
    u_left = node_u[j - 1];
  }
  if (j == n_nodes - 1) {
    r_right = 2.0 * node_r[j] - node_r[j - 1];
    u_right = 2.0 * node_u[j] - node_u[j - 1];
  } else {
    r_right = node_r[j + 1];
    u_right = node_u[j + 1];
  }

  const double r_j = node_r[j];
  const double u_j = node_u[j];
  const double dr_left = r_j - r_left;
  const double dr_right = r_right - r_j;
  if (dr_left <= 0.0 || dr_right <= 0.0) {
    return 0.0;
  }

  const double SL = J * (u_j - u_left) / dr_left;
  const double SR = J * (u_right - u_j) / dr_right;
  if (SL * SR <= 0.0) {
    return 0.0;
  }

  const double SC = ((dr_right / dr_left) * (u_j - u_left) +
                     (dr_left / dr_right) * (u_right - u_j)) /
                    (r_right - r_left);
  return copysign(fmin(fmin(fabs(SL), fabs(SC)), fabs(SR)), SL);
}

__device__ inline void compute_node_sigma_1d_kernel_body(
    const int j,
    double* __restrict__ sigma,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const int n_nodes,
    const double J) {
  sigma[j] = compute_node_sigma_1d_device(node_r, node_u, j, n_nodes, J);
}

__device__ __forceinline__ double compute_eos_aware_boost_device(
    const bool eos_aware,
    const double rho,
    const double total_pressure,
    const double cs,
    const double gamma1_ref,
    const double boost_max) {
  if (!eos_aware || !(rho > 0.0) || !(total_pressure > kSensorEps) || !(cs > 0.0)) {
    return 1.0;
  }
  const double gamma1_local = rho * cs * cs / total_pressure;
  if (!(gamma1_local > 0.0) || !isfinite(gamma1_local)) {
    return 1.0;
  }
  return fmin(fmax(gamma1_ref / gamma1_local, 1.0), boost_max);
}

template <int GEOM>
__device__ inline void compute_q_1d_kernel_body(
    const int i,
    double* __restrict__ Q,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ cs,
    const double* __restrict__ sigma,
    double* __restrict__ chi_out,
    double* __restrict__ q2_out,
    double* __restrict__ div_u_out,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double c1,
    const double c2,
    const double* __restrict__ c1_eff_field,
    const double* __restrict__ c2_eff_field,
    const bool eos_aware,
    const double gamma1_ref,
    const double boost_max) {
  if (hydro_active != nullptr && hydro_active[i] == 0) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    if (div_u_out != nullptr) {
      div_u_out[i] = 0.0;
    }
    return;
  }

  const double V = vol[i];
  const double r0 = node_r[i];
  const double r1 = node_r[i + 1];
  const double dr = r1 - r0;
  if (V <= 0.0 || dr <= 0.0) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    if (div_u_out != nullptr) {
      div_u_out[i] = 0.0;
    }
    return;
  }

  double A0;
  double A1;
  if constexpr (GEOM == 0) {
    A0 = kFourPi * r0 * r0;
    A1 = kFourPi * r1 * r1;
  } else {
    A0 = geometry_1d_face_area(GEOM, r0);
    A1 = geometry_1d_face_area(GEOM, r1);
  }
  const double dVdt = A1 * node_u[i + 1] - A0 * node_u[i];
  const double div_u = dVdt / V;
  if (div_u_out != nullptr) {
    div_u_out[i] = div_u;
  }

  if (div_u >= 0.0) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    return;
  }

  const double u_right = node_u[i + 1] - 0.5 * dr * sigma[i + 1];
  const double u_left = node_u[i] + 0.5 * dr * sigma[i];
  const double du_lim = u_right - u_left;
  const double chi = fmax(0.0, -du_lim / dr);
  if (chi_out != nullptr) {
    chi_out[i] = chi;
  }
  if (chi <= 0.0) {
    Q[i] = 0.0;
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    return;
  }

  const double boost = compute_eos_aware_boost_device(
      eos_aware, rho[i], total_pressure_device(Pe, Pi, i), cs[i],
      gamma1_ref, boost_max);
  const double c1_local = (c1_eff_field != nullptr) ? c1_eff_field[i] : c1;
  const double c2_local = (c2_eff_field != nullptr) ? c2_eff_field[i] : c2;
  const double c1_eff = c1_local * boost;
  const double c2_eff = c2_local * boost;
  const double q2_limited = rho[i] * (c2_eff * c2_eff) * dr * dr * chi * chi;
  if (q2_out != nullptr) {
    q2_out[i] = q2_limited;
  }

  const double left_shock = (i == 0)
                                ? 1.0
                                : interface_shock_support_weight_1d_device(
                                      Pe, Pi, rho, cs, i - 1, i);
  const double right_shock =
      (i == n_cells - 1)
          ? 1.0
          : interface_shock_support_weight_1d_device(Pe, Pi, rho, cs, i, i + 1);
  const double W_shock = fmax(left_shock, right_shock);
  const double M_comp = dr * chi / fmax(cs[i], kSensorEps);
  const double W_comp = clamp01_device(M_comp / kCompMachScale);
  const double Q_base =
      q2_limited + rho[i] * (c1_eff * dr * cs[i] * chi);

  if (W_shock > 0.0) {
    const double left_developed =
        (i == 0)
            ? 1.0
            : interface_developed_shock_weight_1d_device(Pe, Pi, rho, cs, i - 1, i);
    const double right_developed =
        (i == n_cells - 1)
            ? 1.0
            : interface_developed_shock_weight_1d_device(Pe, Pi, rho, cs, i, i + 1);

    double W_osc = 1.0;
    if (i > 0 && i + 1 < n_cells) {
      const double p_left = total_pressure_device(Pe, Pi, i - 1);
      const double p_center = total_pressure_device(Pe, Pi, i);
      const double p_right = total_pressure_device(Pe, Pi, i + 1);
      const double dpL = p_center - p_left;
      const double dpR = p_right - p_center;
      if (dpL * dpR < 0.0) {
        const double osc =
            fmin(fabs(dpL), fabs(dpR)) / fmax(fmax(fabs(dpL), fabs(dpR)), kSensorEps);
        if (osc > kOscillationThreshold &&
            !(left_developed > 0.0 && right_developed > 0.0)) {
          W_osc = 0.0;
        }
      }
    }

    const double phi = W_shock * fmax(kShockSupportFloor, W_comp * W_osc);
    Q[i] = phi * Q_base;
    return;
  }

  const double left_mild =
      (i == 0)
          ? 0.0
          : interface_mild_compression_weight_1d_device(Pe, Pi, rho, cs, i - 1, i);
  const double right_mild =
      (i == n_cells - 1)
          ? 0.0
          : interface_mild_compression_weight_1d_device(Pe, Pi, rho, cs, i, i + 1);
  const double left_neighbor_shock =
      (i > 1)
          ? interface_shock_support_weight_1d_device(Pe, Pi, rho, cs, i - 2, i - 1)
          : 0.0;
  const double right_neighbor_shock =
      (i + 2 < n_cells)
          ? interface_shock_support_weight_1d_device(Pe, Pi, rho, cs, i + 1, i + 2)
          : 0.0;
  if (left_neighbor_shock > 0.0 || right_neighbor_shock > 0.0) {
    Q[i] = 0.0;
    return;
  }
  const double W_mild = fmax(left_mild, right_mild);
  Q[i] = kMildCompressionAlpha * W_mild * W_comp *
         rho[i] * (c1_eff * dr * cs[i] * chi);
}

__device__ inline void compute_node_activity_kernel_body(
    const int i,
    std::uint8_t* __restrict__ node_active,
    const std::int8_t* __restrict__ hydro_active,
    const int /*n_cells*/) {
  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (!active) {
    return;
  }

  node_active[i] = 1u;
  node_active[i + 1] = 1u;
}

__device__ inline void zero_center_node_kernel_body(
    const int /*i*/,
    std::uint8_t* __restrict__ node_active,
    const int /*n_nodes*/) {
  node_active[0] = 0u;
}

template <bool kHasExtra>
__device__ inline void build_cell_pq_kernel_body(
    const int i,
    double* __restrict__ pq,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ Qvisc,
    const double* __restrict__ p_extra,
    const std::int8_t* __restrict__ hydro_active,
    const int /*n_cells*/) {
  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (!active) {
    pq[i] = 0.0;
    return;
  }
  if constexpr (kHasExtra) {
    pq[i] = Pe[i] + Pi[i] + Qvisc[i] + p_extra[i];
  } else {
    pq[i] = Pe[i] + Pi[i] + Qvisc[i];
  }
}

template <int GEOM>
__device__ inline void compute_acceleration_1d_kernel_body(
    const int j,
    double* __restrict__ accel,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ cell_pq,
    const std::uint8_t* __restrict__ node_active,
    const int n_cells,
    const double ghost_pq,
    const int bc_type) {
  if (j == 0) {
    accel[j] = 0.0;
    return;
  }

  if (node_active[j] == 0u) {
    accel[j] = 0.0;
    return;
  }

  if (j < n_cells) {
    const double node_mass = 0.5 * (mass[j - 1] + mass[j]);
    double area;
    if constexpr (GEOM == 0) {
      area = kFourPi * x_r[j] * x_r[j];
    } else {
      area = geometry_1d_face_area(GEOM, x_r[j]);
    }
    const double dp = cell_pq[j] - cell_pq[j - 1];
    accel[j] = (node_mass > 0.0) ? (-(area / node_mass) * dp) : 0.0;
    return;
  }

  const double node_mass = 0.5 * mass[n_cells - 1];
  double area;
  if constexpr (GEOM == 0) {
    area = kFourPi * x_r[n_cells] * x_r[n_cells];
  } else {
    area = geometry_1d_face_area(GEOM, x_r[n_cells]);
  }
  const double dp = ghost_pq - cell_pq[n_cells - 1];
  double a = (node_mass > 0.0) ? (-(area / node_mass) * dp) : 0.0;
  if (bc_type == static_cast<int>(HydroBoundaryType::FIXED) ||
      bc_type == static_cast<int>(HydroBoundaryType::REFLECT)) {
    a = 0.0;
  }
  accel[j] = a;
}

// gamma_r v3 (verdict 2026-07-06): per-cell radiation-pressure force work,
// exactly conjugate to compute_acceleration_1d_kernel — same half-position
// face areas, same node-active masking, ubar = (u_old + u_new)/2.
template <int GEOM>
__device__ inline void gamma_r_force_work_1d_kernel_body(
    const int c,
    double* __restrict__ w_r,
    const double* __restrict__ p_extra,
    const double* __restrict__ r_half,
    const double* __restrict__ u_old,
    const double* __restrict__ u_new,
    const std::uint8_t* __restrict__ node_active,
    const int n_cells,
    const double dt) {
  (void)n_cells;
  const int jL = c;
  const int jR = c + 1;
  double term_L = 0.0;
  double term_R = 0.0;
  if (jL > 0 && node_active[jL] != 0u) {
    double area_L;
    if constexpr (GEOM == 0) {
      area_L = kFourPi * r_half[jL] * r_half[jL];
    } else {
      area_L = geometry_1d_face_area(GEOM, r_half[jL]);
    }
    term_L = area_L * 0.5 * (u_old[jL] + u_new[jL]);
  }
  if (node_active[jR] != 0u) {
    double area_R;
    if constexpr (GEOM == 0) {
      area_R = kFourPi * r_half[jR] * r_half[jR];
    } else {
      area_R = geometry_1d_face_area(GEOM, r_half[jR]);
    }
    term_R = area_R * 0.5 * (u_old[jR] + u_new[jR]);
  }
  w_r[c] = p_extra[c] * dt * (term_R - term_L);
}

__device__ inline void predictor_update_kernel_body(
    const int j,
    double* __restrict__ v_r,
    double* __restrict__ x_r,
    double* __restrict__ u_half,
    const double* __restrict__ u_old,
    const double* __restrict__ r_old,
    const double* __restrict__ accel,
    const std::uint8_t* __restrict__ node_active,
    const int /*n_nodes*/,
    const double dt) {
  if (node_active[j] != 0u) {
    u_half[j] = u_old[j] + 0.5 * dt * accel[j];
    x_r[j] = r_old[j] + 0.5 * dt * u_half[j];
  } else {
    u_half[j] = u_old[j];
    x_r[j] = r_old[j];
  }
  v_r[j] = u_half[j];
}

__device__ inline void corrector_update_kernel_body(
    const int j,
    double* __restrict__ v_r,
    double* __restrict__ x_r,
    const double* __restrict__ u_old,
    const double* __restrict__ r_old,
    const double* __restrict__ u_half,
    const double* __restrict__ a_half,
    const std::uint8_t* __restrict__ node_active,
    const int /*n_nodes*/,
    const double dt) {
  if (node_active[j] != 0u) {
    v_r[j] = u_old[j] + dt * a_half[j];
    x_r[j] = r_old[j] + dt * u_half[j];
  } else {
    v_r[j] = u_old[j];
    x_r[j] = r_old[j];
  }
}

__device__ inline void propagate_void_node_displacement_kernel_body(
    const int /*i*/,
    double* __restrict__ x_r,
    double* __restrict__ v_r,
    const double* __restrict__ r_old,
    const std::uint8_t* __restrict__ node_active,
    const int n_nodes) {
  for (int j = 1; j < n_nodes; ++j) {
    if (node_active[j] != 0u || node_active[j - 1] == 0u) {
      continue;
    }

    const double delta_x = x_r[j - 1] - r_old[j - 1];
    const double boundary_v = v_r[j - 1];
    int k = j;
    while (k < n_nodes && node_active[k] == 0u) {
      x_r[k] = r_old[k] + delta_x;
      v_r[k] = boundary_v;
      ++k;
    }
    j = k - 1;
  }
}

// PERSISTENT-PATH-ONLY duplicate: the production kernel body lives inline in
// hydro_1d.cu; keep this in sync manually when void-node following changes.
template <int GEOM>
__device__ inline void follow_void_nodes_1d_body_persistent(
    const int j,
    double* __restrict__ x_r,
    double* __restrict__ v_r,
    double* __restrict__ mass,
    const int first_void_cell,
    const int n_cells,
    const double rho_void) {
  // Void cells carry no forces, so their interior nodes never move on their
  // own and a Lagrangian edge expanding into the void region inverts cells
  // (BUG-5). Re-space the void-interior nodes uniformly between the plasma
  // edge node (owned by the last real cell — never touched here) and the
  // outer boundary node (owned by the free-boundary physics). Void masses
  // are re-pinned to rho_void * V so the placeholder cells keep their floor
  // density; they are excluded from physics, so no conserved quantity of a
  // real cell is affected.
  const int edge_node = first_void_cell;      // outer node of last real cell
  const int last_node = n_cells;              // outer boundary node
  const int n_span = last_node - edge_node;   // number of void cells
  if (n_span <= 1) {
    return;  // zero or one void cell: no interior nodes to manage
  }
  const double edge = x_r[edge_node];
  const double outer = x_r[last_node];
  const double width = outer - edge;
  if (!(width > 0.0)) {
    return;  // edge reached the boundary; free-boundary physics takes over
  }
  const int interior = edge_node + 1 + j;
  if (interior >= last_node) {
    return;
  }
  const double frac =
      static_cast<double>(j + 1) / static_cast<double>(n_span);
  x_r[interior] = edge + frac * width;
  v_r[interior] = 0.0;
  // Re-pin the void cell masses on both sides of this node lazily: each
  // thread owns cell (interior - 1); thread j == n_span - 2 also owns the
  // outermost void cell.
  const double r0 = edge + (static_cast<double>(j) / n_span) * width;
  const double r1 = x_r[interior];
  double vol0;
  if constexpr (GEOM == 0) {
    vol0 =
        (kFourPi / 3.0) * (r1 - r0) * (r1 * r1 + r1 * r0 + r0 * r0);
  } else {
    vol0 = geometry_1d_shell_volume(GEOM, r0, r1);
  }
  mass[interior - 1] = rho_void * fmax(vol0, 0.0);
  if (j == n_span - 2) {
    const double r2 = outer;
    double vol1;
    if constexpr (GEOM == 0) {
      vol1 =
          (kFourPi / 3.0) * (r2 - r1) * (r2 * r2 + r2 * r1 + r1 * r1);
    } else {
      vol1 = geometry_1d_shell_volume(GEOM, r1, r2);
    }
    mass[interior] = rho_void * fmax(vol1, 0.0);
  }
}

__device__ inline void copy_array_kernel_body(
    const int i,
    double* __restrict__ dst,
    const double* __restrict__ src,
    const int /*n*/) {
  dst[i] = src[i];
}

__device__ inline void average_arrays_kernel_body(
    const int i,
    double* __restrict__ out,
    const double* __restrict__ a,
    const double* __restrict__ b,
    const int /*n*/) {
  out[i] = 0.5 * (a[i] + b[i]);
}

template <int GEOM>
__device__ inline double compatible_volume_change_1d(
    const double* __restrict__ r_new,
    const double* __restrict__ r_old,
    const double* __restrict__ u_half,
    const int i,
    const double dt) {
  const double r_lo_half = 0.5 * (r_old[i] + r_new[i]);
  const double r_hi_half = 0.5 * (r_old[i + 1] + r_new[i + 1]);
  double area_lo;
  if constexpr (GEOM == 0) {
    area_lo = kFourPi * r_lo_half * r_lo_half;
  } else {
    area_lo = geometry_1d_face_area(GEOM, r_lo_half);
  }
  double area_hi;
  if constexpr (GEOM == 0) {
    area_hi = kFourPi * r_hi_half * r_hi_half;
  } else {
    area_hi = geometry_1d_face_area(GEOM, r_hi_half);
  }
  return dt * (area_hi * u_half[i + 1] - area_lo * u_half[i]);
}

template <int GEOM>
__device__ inline void energy_update_with_old_volume_kernel_body(
    const int i,
    double* __restrict__ ee,
    const double* __restrict__ e_old,
    const double* __restrict__ r_new,
    const double* __restrict__ r_old,
    const double* __restrict__ u_half,
    const double* __restrict__ vol,
    const double* __restrict__ vol_old,
    const double* __restrict__ mass,
    const double* __restrict__ P_half,
    const double* __restrict__ Q_half,
    const std::int8_t* __restrict__ hydro_active,
    const int /*n_cells*/,
    const double dt,
    const int compatible_energy,
    double* __restrict__ eta_compatible,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (!active) {
    ee[i] = e_old[i];
    if (eta_compatible != nullptr) {
      eta_compatible[i] = 0.0;
    }
    return;
  }

  const double m = mass[i];
  if (m <= 0.0) {
    ee[i] = e_old[i];
    if (eta_compatible != nullptr) {
      eta_compatible[i] = 0.0;
    }
    return;
  }

  const double dV_geom = vol[i] - vol_old[i];
  const double dV =
      (compatible_energy != 0)
          ? compatible_volume_change_1d<GEOM>(r_new, r_old, u_half, i, dt)
          : dV_geom;
  if (eta_compatible != nullptr) {
    eta_compatible[i] = dV_geom - dV;
  }
  const double e_raw = e_old[i] - (P_half[i] + Q_half[i]) * dV / m;
  const double e_new = fmax(e_raw, 0.0);
  ee[i] = e_new;
  if (e_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (e_new - e_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
}

template <int GEOM>
__device__ inline void energy_update_with_old_volume_2t_kernel_body(
    const int i,
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ ee_old,
    const double* __restrict__ ei_old,
    const double* __restrict__ rho_half,
    const double* __restrict__ Te_half,
    const double* __restrict__ Ti_half,
    const double* __restrict__ r_new,
    const double* __restrict__ r_old,
    const double* __restrict__ u_half,
    const double* __restrict__ vol,
    const double* __restrict__ vol_old,
    const double* __restrict__ mass,
    const double* __restrict__ Pe_half,
    const double* __restrict__ Pi_half,
    const double* __restrict__ Q_half,
    const double* __restrict__ zbar,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double dt,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const double fallback_z,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele,
    const double* __restrict__ cv_e_arr,
    const double* __restrict__ cv_i_arr,
    const int q_heat_to_electron,
    const int compatible_energy,
    double* __restrict__ eta_compatible,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (!active) {
    ee[i] = ee_old[i];
    ei[i] = ei_old[i];
    if (eta_compatible != nullptr) {
      eta_compatible[i] = 0.0;
    }
    return;
  }

  const double gamma = fmax(gamma_eff[i], 1.0 + 1.0e-12);
  const double A = fmax(A_eff[i], 1.0e-12);
  const double m = mass[i];
  if (m <= 0.0) {
    ee[i] = ee_old[i];
    ei[i] = ei_old[i];
    if (eta_compatible != nullptr) {
      eta_compatible[i] = 0.0;
    }
    return;
  }

  const double dV_geom = vol[i] - vol_old[i];
  const double dV =
      (compatible_energy != 0)
          ? compatible_volume_change_1d<GEOM>(r_new, r_old, u_half, i, dt)
          : dV_geom;
  if (eta_compatible != nullptr) {
    eta_compatible[i] = dV_geom - dV;
  }
  const double rho_qei = fmax(rho_half[i], 0.0);
  const double z = (zbar != nullptr) ? fmax(zbar[i], 0.0) : fallback_z;
  double qei_term = 0.0;
  if (tab_ion.n_rho > 0 && tab_ele.n_rho > 0 &&
      cv_e_arr != nullptr && cv_i_arr != nullptr) {
    qei_term = tenryu::materials::compute_qei_term_with_cv(
        rho_qei, fmax(Te_half[i], 0.0), fmax(Ti_half[i], 0.0), z, A,
        fmax(cv_e_arr[i], 0.0), fmax(cv_i_arr[i], 0.0), dt);
  } else {
    qei_term = tenryu::materials::compute_qei_term_analytical(
        rho_qei, fmax(Te_half[i], 0.0), fmax(Ti_half[i], 0.0), z, A, gamma, dt);
  }

  // Q_ei applied as conservative transfer: +Q_ei to ions, -Q_ei to electrons
  // (see NUMERICS §1.1.3).
  double de_i = qei_term;
  double de_e = -qei_term;
  const double q_work = -Q_half[i] * dV / m;
  if (compatible_energy != 0) {
    const double p_total = Pe_half[i] + Pi_half[i];
    double electron_frac = 0.5;
    if (fabs(p_total) > 0.0) {
      electron_frac = Pe_half[i] / p_total;
    }
    const double pdv_total = -p_total * dV / m;
    de_e += electron_frac * pdv_total;
    de_i += (1.0 - electron_frac) * pdv_total;
  } else {
    de_i += -Pi_half[i] * dV / m;
    de_e += -Pe_half[i] * dV / m;
  }
  if (q_heat_to_electron != 0) {
    de_e += q_work;
  } else {
    de_i += q_work;
  }

  const double ei_raw = ei_old[i] + de_i;
  const double ee_raw = ee_old[i] + de_e;
  const bool use_table = (tab_ion.n_rho > 0);
  const double ei_new = use_table ? ei_raw : fmax(ei_raw, 0.0);
  const double ee_new = use_table ? ee_raw : fmax(ee_raw, 0.0);
  ei[i] = ei_new;
  ee[i] = ee_new;

  if (!use_table && ei_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (ei_new - ei_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
  if (!use_table && ee_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (ee_new - ee_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
}

__device__ inline void scale_active_energy_kernel_body(
    const int i,
    double* __restrict__ ee,
    const std::int8_t* __restrict__ hydro_active,
    const double scale) {
  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (!active) {
    return;
  }
  ee[i] = fmax(ee[i] * scale, 0.0);
}

__device__ inline void shift_active_energy_kernel_body(
    const int i,
    double* __restrict__ ee,
    const std::int8_t* __restrict__ hydro_active,
    const double de) {
  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (!active) {
    return;
  }
  ee[i] = fmax(ee[i] + de, 0.0);
}

__device__ inline void add_first_active_residual_kernel_body(
    double* __restrict__ ee,
    const double* __restrict__ mass,
    const int first_active,
    const double residual) {
  if (first_active >= 0) {
    const double m = mass[first_active];
    if (m > 0.0) {
      ee[first_active] += residual / m;
    }
  }
}

__device__ inline void find_nonpositive_volume_1d_kernel_body(
    const int c,
    const double* __restrict__ vol,
    int* __restrict__ first_failing_cell,
    const int /*n_cells*/) {
  if (!(vol[c] > 0.0)) {
    atomicMin(first_failing_cell, c);
  }
}

__device__ inline void apply_boundary_1d_kernel_body(
    const int idx,
    double* __restrict__ x_r,
    double* __restrict__ v_r,
    const int n_nodes,
    const int apply_outer,
    const double r_min,
    const double r_max) {
  if (idx == 0 && n_nodes > 0) {
    v_r[0] = 0.0;
    x_r[0] = r_min;
  }

  if (idx == 1 && n_nodes > 1 && apply_outer != 0) {
    const int j_outer = n_nodes - 1;
    v_r[j_outer] = 0.0;
    x_r[j_outer] = r_max;
  }
}

__device__ __forceinline__ double cfl_compute_chi_1d(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const int i,
    const int n_cells,
    const double J) {
  const double r0 = node_r[i];
  const double r1 = node_r[i + 1];
  const double dr = r1 - r0;
  if (dr <= 0.0) {
    return 0.0;
  }

  const int n_nodes = n_cells + 1;
  const double sigma_left = compute_node_sigma_1d_device(node_r, node_u, i, n_nodes, J);
  const double sigma_right = compute_node_sigma_1d_device(node_r, node_u, i + 1, n_nodes, J);
  const double u_right = node_u[i + 1] - 0.5 * dr * sigma_right;
  const double u_left = node_u[i] + 0.5 * dr * sigma_left;
  const double du_lim = u_right - u_left;
  return fmax(0.0, -du_lim / dr);
}

__device__ inline double cfl_1d_candidate_body(
    const int i,
    const double* __restrict__ x_r,
    const double* __restrict__ v_r,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ cs_arr,
    const double* __restrict__ shock_time,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double t_current,
    const double gamma,
    const double c1,
    const double c2,
    const int use_riemann_av,
    const int post_shock_heat_enabled,
    const double post_shock_heat_c,
    const double post_shock_heat_decay,
    const double J,
    double* __restrict__ post_shock_dt,
    int* __restrict__ have_active) {
  if (hydro_active != nullptr && hydro_active[i] == 0) {
    return CUDART_INF;
  }

  *have_active = 1;

  const double dr = x_r[i + 1] - x_r[i];
  if (dr <= 0.0) {
    return CUDART_INF;
  }

  const double e_e = ee[i] > 0.0 ? ee[i] : 0.0;
  const double e_i = (ei != nullptr && ei[i] > 0.0) ? ei[i] : 0.0;
  const double cs_val =
      (cs_arr != nullptr) ? cs_arr[i] : sqrt(gamma * (gamma - 1.0) * (e_e + e_i));
  const double cs = fmax(cs_val, 0.0);
  double denom = cs;
  if (use_riemann_av != 0) {
    return CUDART_INF;
  } else {
    const double chi = cfl_compute_chi_1d(x_r, v_r, i, n_cells, J);
    const double compression_speed = dr * chi;
    denom += c1 * cs + c2 * compression_speed;
  }
  if (denom <= 0.0) {
    return CUDART_INF;
  }

  const double dt = dr / denom;

  if (post_shock_heat_enabled != 0 && post_shock_dt != nullptr &&
      shock_time != nullptr && post_shock_heat_c > 0.0 && cs > 0.0) {
    const double tau_decay =
        post_shock_heat_decay * dr / fmax(cs, kCflSensorEps);
    if (tau_decay > 0.0) {
      const double age = fmax(t_current - shock_time[i], 0.0);
      const double psi = exp(-age / tau_decay);
      const double denom_heat = post_shock_heat_c * psi * cs;
      if (denom_heat > 0.0) {
        *post_shock_dt = 0.5 * dr / denom_heat;
      }
    }
  }

  return dt;
}

}  // namespace tenryu::hydro::persistent_1d
