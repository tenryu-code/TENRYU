#pragma once

#include <cmath>

#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

constexpr double kEOSRhoECvFloor = 1.0e-3;
constexpr double kEOSRhoECvMax = 1.0e30;
constexpr double kEOSRhoEPressureFloor = 1.0e-30;
constexpr double kEOSRhoECs2Floor = 1.0e-30;

struct EOSRhoEDeviceView {
  const double* log_rho_grid;  // [n_rho], axis is log(rho) or rho depending on use_log_grid
  const double* log_e_grid;    // [n_e], axis is log(e) or e depending on use_log_grid
  const double* P_table;       // [n_e * n_rho], idx = j_e * n_rho + i_rho
  const double* T_table;       // [n_e * n_rho], idx = j_e * n_rho + i_rho
  const double* P_xx;          // [n_e * n_rho], idx = j_e * n_rho + i_rho
  const double* P_yy;          // [n_e * n_rho], idx = j_e * n_rho + i_rho
  const double* P_xxyy;        // [n_e * n_rho], idx = j_e * n_rho + i_rho
  const double* T_xx;          // [n_e * n_rho], idx = j_e * n_rho + i_rho
  const double* T_yy;          // [n_e * n_rho], idx = j_e * n_rho + i_rho
  const double* T_xxyy;        // [n_e * n_rho], idx = j_e * n_rho + i_rho
  bool use_log_grid;           // true -> (log rho, log e); false -> (rho, e)
  int n_rho;                   // 0 = no rho-e table
  int n_e;
  double log_rho_min;
  double log_rho_max;
  double log_e_min;
  double log_e_max;
  double d_log_rho_inv;  // 0 = non-uniform -> binary search
  double d_log_e_inv;    // 0 = non-uniform -> binary search
};

__host__ __device__ inline double eos_rho_e_axis_to_physical(const bool use_log_grid,
                                                             const double axis_value) {
  return use_log_grid ? exp(axis_value) : axis_value;
}

__host__ __device__ inline double eos_rho_e_rho_min(const EOSRhoEDeviceView& view) {
  return eos_rho_e_axis_to_physical(view.use_log_grid, view.log_rho_min);
}

struct EOSRhoEResult {
  double P = 0.0;
  double T = 0.0;
  double cv = 0.0;
  double cs2 = 0.0;
  double e_clamped = 0.0;
};

#ifdef __CUDACC__
struct EOSRhoEQuery {
  int i0 = 0;
  int i1 = 0;
  int j0 = 0;
  int j1 = 0;
  double log_rho = 0.0;
  double log_e = 0.0;
  double rho = 0.0;
  double e = 0.0;
};

struct EOSRhoESplineAxisEval {
  double A[2] = {1.0, 0.0};
  double B[2] = {0.0, 0.0};
  double dA[2] = {0.0, 0.0};
  double dB[2] = {0.0, 0.0};
};

__device__ inline EOSRhoEQuery prepare_eos_rho_e_query(const EOSRhoEDeviceView& view,
                                                       const double rho,
                                                       const double e_specific) {
  EOSRhoEQuery q{};
  if (view.n_rho <= 0 || view.n_e <= 0) {
    return q;
  }

  double log_rho = view.log_rho_min;
  if (rho > 0.0 && isfinite(rho)) {
    log_rho = view.use_log_grid ? log(rho) : rho;
  }
  double log_e = view.log_e_min;
  if (e_specific > 0.0 && isfinite(e_specific)) {
    log_e = view.use_log_grid ? log(e_specific) : e_specific;
  }

  const RhoBracket rb = find_bracket_from_log(view.log_rho_grid,
                                              view.n_rho,
                                              log_rho,
                                              view.log_rho_min,
                                              view.log_rho_max,
                                              view.d_log_rho_inv);
  const RhoBracket eb = find_bracket_from_log(view.log_e_grid,
                                              view.n_e,
                                              log_e,
                                              view.log_e_min,
                                              view.log_e_max,
                                              view.d_log_e_inv);

  q.log_rho = clamp_log_value(log_rho, view.log_rho_min, view.log_rho_max);
  q.log_e = clamp_log_value(log_e, view.log_e_min, view.log_e_max);

  if (view.n_rho == 1) {
    q.i0 = 0;
    q.i1 = 0;
  } else {
    q.i0 = clamp_int(rb.i0, 0, view.n_rho - 2);
    q.i1 = clamp_int(rb.i1, q.i0 + 1, view.n_rho - 1);
  }

  if (view.n_e == 1) {
    q.j0 = 0;
    q.j1 = 0;
  } else {
    q.j0 = clamp_int(eb.i0, 0, view.n_e - 2);
    q.j1 = clamp_int(eb.i1, q.j0 + 1, view.n_e - 1);
  }

  q.rho = eos_rho_e_axis_to_physical(view.use_log_grid, q.log_rho);
  q.e = eos_rho_e_axis_to_physical(view.use_log_grid, q.log_e);
  return q;
}

__device__ inline EOSRhoESplineAxisEval build_eos_rho_e_spline_axis_eval(
    const double* grid,
    const int n,
    const int i0,
    const int i1,
    const double x,
    const double x_min,
    const double x_max) {
  EOSRhoESplineAxisEval out{};
  if (n < 2 || i0 == i1) {
    return out;
  }

  const double x0 = table_log_coord(grid, n, i0, x_min, x_max);
  const double x1 = table_log_coord(grid, n, i1, x_min, x_max);
  const double h = fmax(x1 - x0, 1.0e-30);
  const double u = clamp01((x - x0) / h);
  const double one_minus_u = 1.0 - u;

  out.A[0] = one_minus_u;
  out.A[1] = u;
  out.B[0] = (h * h / 6.0) * (one_minus_u * one_minus_u * one_minus_u - one_minus_u);
  out.B[1] = (h * h / 6.0) * (u * u * u - u);
  out.dA[0] = -1.0 / h;
  out.dA[1] = 1.0 / h;
  out.dB[0] = (h / 6.0) * (1.0 - 3.0 * one_minus_u * one_minus_u);
  out.dB[1] = (h / 6.0) * (3.0 * u * u - 1.0);
  return out;
}

__device__ inline void eval_rho_e_field_with_log_derivatives(
    const EOSRhoEDeviceView& view,
    const double* field,
    const double* field_xx,
    const double* field_yy,
    const double* field_xxyy,
    const EOSRhoEQuery& q,
    double* value,
    double* dfdlogrho,
    double* dfdloge) {
  if (value != nullptr) {
    *value = 0.0;
  }
  if (dfdlogrho != nullptr) {
    *dfdlogrho = 0.0;
  }
  if (dfdloge != nullptr) {
    *dfdloge = 0.0;
  }
  if (view.n_rho <= 0 || view.n_e <= 0 || field == nullptr || field_xx == nullptr ||
      field_yy == nullptr || field_xxyy == nullptr) {
    return;
  }

  const EOSRhoESplineAxisEval x_eval = build_eos_rho_e_spline_axis_eval(view.log_rho_grid,
                                                                        view.n_rho,
                                                                        q.i0,
                                                                        q.i1,
                                                                        q.log_rho,
                                                                        view.log_rho_min,
                                                                        view.log_rho_max);
  const EOSRhoESplineAxisEval y_eval = build_eos_rho_e_spline_axis_eval(view.log_e_grid,
                                                                        view.n_e,
                                                                        q.j0,
                                                                        q.j1,
                                                                        q.log_e,
                                                                        view.log_e_min,
                                                                        view.log_e_max);
  const int ii[2] = {q.i0, q.i1};
  const int jj[2] = {q.j0, q.j1};
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      const int idx = jj[b] * view.n_rho + ii[a];
      const double q_field = field[idx];
      const double q_xx = field_xx[idx];
      const double q_yy = field_yy[idx];
      const double q_xxyy = field_xxyy[idx];

      if (value != nullptr) {
        *value += x_eval.A[a] * y_eval.A[b] * q_field +
                  x_eval.B[a] * y_eval.A[b] * q_xx +
                  x_eval.A[a] * y_eval.B[b] * q_yy +
                  x_eval.B[a] * y_eval.B[b] * q_xxyy;
      }
      if (dfdlogrho != nullptr) {
        *dfdlogrho += x_eval.dA[a] * y_eval.A[b] * q_field +
                      x_eval.dB[a] * y_eval.A[b] * q_xx +
                      x_eval.dA[a] * y_eval.B[b] * q_yy +
                      x_eval.dB[a] * y_eval.B[b] * q_xxyy;
      }
      if (dfdloge != nullptr) {
        *dfdloge += x_eval.A[a] * y_eval.dA[b] * q_field +
                    x_eval.B[a] * y_eval.dA[b] * q_xx +
                    x_eval.A[a] * y_eval.dB[b] * q_yy +
                    x_eval.B[a] * y_eval.dB[b] * q_xxyy;
      }
    }
  }
}

__device__ inline double safe_eos_rho_e_cv(const double dTde, const double dPde) {
  if (!(isfinite(dTde) && dTde > 0.0) || !(isfinite(dPde) && dPde > 0.0)) {
    return kEOSRhoECvFloor;
  }
  return fmin(fmax(1.0 / dTde, kEOSRhoECvFloor), kEOSRhoECvMax);
}

__device__ inline EOSRhoEResult device_eos_rho_e_eval(const EOSRhoEDeviceView& view,
                                                      const double rho,
                                                      const double e_specific) {
  EOSRhoEResult out{};
  if (view.n_rho <= 0 || view.n_e <= 0 || view.P_table == nullptr || view.T_table == nullptr ||
      view.P_xx == nullptr || view.P_yy == nullptr || view.P_xxyy == nullptr ||
      view.T_xx == nullptr || view.T_yy == nullptr || view.T_xxyy == nullptr) {
    return out;
  }

  const EOSRhoEQuery q = prepare_eos_rho_e_query(view, rho, e_specific);

  double P = 0.0;
  double dP_dlogrho = 0.0;
  double dP_dloge = 0.0;
  eval_rho_e_field_with_log_derivatives(
      view, view.P_table, view.P_xx, view.P_yy, view.P_xxyy, q, &P, &dP_dlogrho, &dP_dloge);

  double T = 0.0;
  double dT_dloge = 0.0;
  eval_rho_e_field_with_log_derivatives(
      view, view.T_table, view.T_xx, view.T_yy, view.T_xxyy, q, &T, nullptr, &dT_dloge);

  const double dPdrho =
      view.use_log_grid ? ((q.rho > 0.0) ? (dP_dlogrho / q.rho) : 0.0) : dP_dlogrho;
  const double dPde =
      view.use_log_grid ? ((q.e > 0.0) ? (dP_dloge / q.e) : 0.0) : dP_dloge;
  const double dTde =
      view.use_log_grid ? ((q.e > 0.0) ? (dT_dloge / q.e) : 0.0) : dT_dloge;
  const double P_safe =
      (isfinite(P) && P > kEOSRhoEPressureFloor) ? P : kEOSRhoEPressureFloor;
  const double cs2 = dPdrho + ((q.rho > 0.0) ? ((P_safe / (q.rho * q.rho)) * dPde) : 0.0);

  out.P = P_safe;
  out.T = (isfinite(T) && T > 0.0) ? T : 0.0;
  out.cv = safe_eos_rho_e_cv(dTde, dPde);
  out.cs2 = (isfinite(cs2) && cs2 > kEOSRhoECs2Floor) ? cs2 : kEOSRhoECs2Floor;
  out.e_clamped = q.e;
  return out;
}
#endif  // __CUDACC__

}  // namespace tenryu::materials
