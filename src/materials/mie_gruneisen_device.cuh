#pragma once

#include <cmath>

#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

constexpr double kMieGruneisenGammaFloor = 1.0e-8;
constexpr double kMieGruneisenPressureFloor = 1.0e-30;
constexpr double kMieGruneisenCs2Floor = 1.0e-30;

struct MieGruneisenDeviceView {
  const double* log_rho_grid;
  const double* gamma_ion;
  const double* gamma_ele;
  const double* p_ref_ion;
  const double* p_ref_ele;
  const double* e_ref_ion;
  const double* e_ref_ele;
  const double* dgamma_ion_dlogrho;
  const double* dgamma_ele_dlogrho;
  const double* dp_ref_ion_dlogrho;
  const double* dp_ref_ele_dlogrho;
  const double* de_ref_ion_dlogrho;
  const double* de_ref_ele_dlogrho;
  int n_rho;
  double log_rho_min;
  double log_rho_max;
  double d_log_rho_inv;
};

#ifdef __CUDACC__

struct MieGruneisenFieldEval {
  double value = 0.0;
  double d_dlogrho = 0.0;
};

struct MieGruneisenBranchState {
  double gamma = kMieGruneisenGammaFloor;
  double p_ref = kMieGruneisenPressureFloor;
  double e_ref = 0.0;
  double dgamma_dlogrho = 0.0;
  double dp_ref_dlogrho = 0.0;
  double de_ref_dlogrho = 0.0;
};

__device__ inline MieGruneisenFieldEval device_eval_mie_gruneisen_field(
    const MieGruneisenDeviceView& view,
    const double* values,
    const double* slopes,
    const double rho) {
  MieGruneisenFieldEval out{};
  if (view.n_rho <= 0 || values == nullptr) {
    return out;
  }
  if (view.n_rho == 1) {
    out.value = values[0];
    if (slopes != nullptr) {
      out.d_dlogrho = slopes[0];
    }
    return out;
  }

  double log_rho = view.log_rho_min;
  if (rho > 0.0 && isfinite(rho)) {
    log_rho = log(rho);
  }
  const RhoBracket rb = find_bracket_from_log(view.log_rho_grid,
                                              view.n_rho,
                                              log_rho,
                                              view.log_rho_min,
                                              view.log_rho_max,
                                              view.d_log_rho_inv);
  const int i0 = clamp_int(rb.i0, 0, view.n_rho - 2);
  const int i1 = clamp_int(rb.i1, i0 + 1, view.n_rho - 1);
  const double x0 = table_log_coord(view.log_rho_grid, view.n_rho, i0,
                                    view.log_rho_min, view.log_rho_max);
  const double x1 = table_log_coord(view.log_rho_grid, view.n_rho, i1,
                                    view.log_rho_min, view.log_rho_max);
  const double h = fmax(x1 - x0, 1.0e-30);
  const double t = clamp01((clamp_log_value(log_rho, view.log_rho_min, view.log_rho_max) - x0) / h);

  const double h00 = 2.0 * t * t * t - 3.0 * t * t + 1.0;
  const double h10 = t * t * t - 2.0 * t * t + t;
  const double h01 = -2.0 * t * t * t + 3.0 * t * t;
  const double h11 = t * t * t - t * t;
  const double dh00 = 6.0 * t * t - 6.0 * t;
  const double dh10 = 3.0 * t * t - 4.0 * t + 1.0;
  const double dh01 = -6.0 * t * t + 6.0 * t;
  const double dh11 = 3.0 * t * t - 2.0 * t;

  const double y0 = values[i0];
  const double y1 = values[i1];
  const double m0 = (slopes != nullptr) ? slopes[i0] : 0.0;
  const double m1 = (slopes != nullptr) ? slopes[i1] : 0.0;

  out.value = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
  out.d_dlogrho =
      (dh00 * y0 + dh10 * h * m0 + dh01 * y1 + dh11 * h * m1) / h;
  return out;
}

__device__ inline MieGruneisenBranchState device_mie_gruneisen_branch_state(
    const MieGruneisenDeviceView& view,
    const double rho,
    const bool electron_branch) {
  MieGruneisenBranchState out{};
  if (view.n_rho <= 0) {
    return out;
  }

  const MieGruneisenFieldEval gamma_eval =
      device_eval_mie_gruneisen_field(view,
                                      electron_branch ? view.gamma_ele : view.gamma_ion,
                                      electron_branch ? view.dgamma_ele_dlogrho
                                                      : view.dgamma_ion_dlogrho,
                                      rho);
  const MieGruneisenFieldEval pref_eval =
      device_eval_mie_gruneisen_field(view,
                                      electron_branch ? view.p_ref_ele : view.p_ref_ion,
                                      electron_branch ? view.dp_ref_ele_dlogrho
                                                      : view.dp_ref_ion_dlogrho,
                                      rho);
  const MieGruneisenFieldEval eref_eval =
      device_eval_mie_gruneisen_field(view,
                                      electron_branch ? view.e_ref_ele : view.e_ref_ion,
                                      electron_branch ? view.de_ref_ele_dlogrho
                                                      : view.de_ref_ion_dlogrho,
                                      rho);

  out.gamma = (isfinite(gamma_eval.value) && gamma_eval.value > kMieGruneisenGammaFloor)
                  ? gamma_eval.value
                  : kMieGruneisenGammaFloor;
  out.p_ref = (isfinite(pref_eval.value) && pref_eval.value > kMieGruneisenPressureFloor)
                  ? pref_eval.value
                  : kMieGruneisenPressureFloor;
  out.e_ref = isfinite(eref_eval.value) ? eref_eval.value : 0.0;
  out.dgamma_dlogrho = isfinite(gamma_eval.d_dlogrho) ? gamma_eval.d_dlogrho : 0.0;
  out.dp_ref_dlogrho = isfinite(pref_eval.d_dlogrho) ? pref_eval.d_dlogrho : 0.0;
  out.de_ref_dlogrho = isfinite(eref_eval.d_dlogrho) ? eref_eval.d_dlogrho : 0.0;
  return out;
}

__device__ inline double device_mie_gruneisen_pressure_from_state(
    const MieGruneisenBranchState& branch,
    const double rho,
    const double e_specific) {
  const double rho_safe = fmax(rho, 1.0e-30);
  const double e_safe = isfinite(e_specific) ? e_specific : 0.0;
  const double pressure = branch.p_ref + branch.gamma * rho_safe * (e_safe - branch.e_ref);
  return (isfinite(pressure) && pressure > kMieGruneisenPressureFloor)
             ? pressure
             : kMieGruneisenPressureFloor;
}

__device__ inline double device_mie_gruneisen_pressure(
    const MieGruneisenDeviceView& view,
    const double rho,
    const double e_specific,
    const bool electron_branch) {
  const MieGruneisenBranchState branch =
      device_mie_gruneisen_branch_state(view, rho, electron_branch);
  return device_mie_gruneisen_pressure_from_state(branch, rho, e_specific);
}

__device__ inline double device_mie_gruneisen_branch_cs2(
    const MieGruneisenBranchState& branch,
    const double rho,
    const double e_specific,
    const double pressure) {
  const double rho_safe = fmax(rho, 1.0e-30);
  const double e_safe = isfinite(e_specific) ? e_specific : 0.0;
  const double delta_e = e_safe - branch.e_ref;
  const double dP_dlogrho = branch.dp_ref_dlogrho +
                            branch.dgamma_dlogrho * rho_safe * delta_e +
                            branch.gamma * rho_safe * delta_e -
                            branch.gamma * rho_safe * branch.de_ref_dlogrho;
  const double dPdrho = dP_dlogrho / rho_safe;
  const double cs2 = dPdrho + branch.gamma * pressure / rho_safe;
  return (isfinite(cs2) && cs2 > kMieGruneisenCs2Floor)
             ? cs2
             : kMieGruneisenCs2Floor;
}

__device__ inline double device_mie_gruneisen_sound_speed_2t(
    const MieGruneisenDeviceView& view,
    const double rho,
    const double ei,
    const double ee,
    const double Pi,
    const double Pe) {
  const MieGruneisenBranchState ion =
      device_mie_gruneisen_branch_state(view, rho, false);
  const MieGruneisenBranchState ele =
      device_mie_gruneisen_branch_state(view, rho, true);
  const double cs2_ion = device_mie_gruneisen_branch_cs2(ion, rho, ei, Pi);
  const double cs2_ele = device_mie_gruneisen_branch_cs2(ele, rho, ee, Pe);
  return sqrt(fmax(cs2_ion + cs2_ele, kMieGruneisenCs2Floor));
}

#endif  // __CUDACC__

}  // namespace tenryu::materials
