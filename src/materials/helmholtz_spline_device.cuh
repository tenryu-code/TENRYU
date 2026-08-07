#pragma once

#include <cmath>

#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

struct HelmholtzSplineDeviceView {
  const double* log_rho_grid;   // [n_rho]
  const double* log_T_grid;     // [n_T]

  const double* P_nodes;        // [n_T * n_rho]
  const double* P_dx_nodes;     // [n_T * n_rho]
  const double* P_dy_nodes;     // [n_T * n_rho]
  const double* P_dxdy_nodes;   // [n_T * n_rho]

  const double* e_nodes;        // [n_T * n_rho]
  const double* e_dx_nodes;     // [n_T * n_rho]
  const double* e_dy_nodes;     // [n_T * n_rho]
  const double* e_dxdy_nodes;   // [n_T * n_rho]

  int n_rho;
  int n_T;
  double log_rho_min;
  double log_rho_max;
  double log_T_min;
  double log_T_max;
  double d_log_rho_inv;
  double d_log_T_inv;
};

struct HelmholtzSplineEval {
  double q;
  double q_x;
  double q_y;
  double q_xx;
  double q_yy;
  double q_xy;
};

struct HelmholtzSplineThermo {
  double pressure;
  double energy;
  double cv;
  double sound_speed;
  double dP_drho_T;
  double dP_dT_rho;
};

#ifdef __CUDACC__

__device__ inline int helmholtz_flat_index(const HelmholtzSplineDeviceView& spline,
                                           const int i_rho,
                                           const int j_T) {
  return j_T * spline.n_rho + i_rho;
}

__device__ inline HelmholtzSplineEval device_helmholtz_eval_scalar(
    const HelmholtzSplineDeviceView& spline,
    const double log_rho_raw,
    const double log_T_raw,
    const double* nodes,
    const double* dx_nodes,
    const double* dy_nodes,
    const double* dxdy_nodes) {
  HelmholtzSplineEval out{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (spline.n_rho < 2 || spline.n_T < 2 || nodes == nullptr) {
    return out;
  }

  const double log_rho = clamp_log_value(log_rho_raw, spline.log_rho_min, spline.log_rho_max);
  const double log_T = clamp_log_value(log_T_raw, spline.log_T_min, spline.log_T_max);
  const RhoBracket rb_x = find_bracket_from_log(spline.log_rho_grid,
                                                spline.n_rho,
                                                log_rho,
                                                spline.log_rho_min,
                                                spline.log_rho_max,
                                                spline.d_log_rho_inv);
  const RhoBracket rb_y = find_bracket_from_log(spline.log_T_grid,
                                                spline.n_T,
                                                log_T,
                                                spline.log_T_min,
                                                spline.log_T_max,
                                                spline.d_log_T_inv);

  const int i0 = clamp_int(rb_x.i0, 0, spline.n_rho - 2);
  const int i1 = i0 + 1;
  const int j0 = clamp_int(rb_y.i0, 0, spline.n_T - 2);
  const int j1 = j0 + 1;
  const double x0 = table_log_coord(spline.log_rho_grid,
                                    spline.n_rho,
                                    i0,
                                    spline.log_rho_min,
                                    spline.log_rho_max);
  const double x1 = table_log_coord(spline.log_rho_grid,
                                    spline.n_rho,
                                    i1,
                                    spline.log_rho_min,
                                    spline.log_rho_max);
  const double y0 = table_log_coord(spline.log_T_grid,
                                    spline.n_T,
                                    j0,
                                    spline.log_T_min,
                                    spline.log_T_max);
  const double y1 = table_log_coord(spline.log_T_grid,
                                    spline.n_T,
                                    j1,
                                    spline.log_T_min,
                                    spline.log_T_max);
  const double hx = fmax(x1 - x0, 1.0e-30);
  const double hy = fmax(y1 - y0, 1.0e-30);
  const double u = clamp01((log_rho - x0) / hx);
  const double v = clamp01((log_T - y0) / hy);

  const double hx_val[2] = {2.0 * u * u * u - 3.0 * u * u + 1.0,
                            -2.0 * u * u * u + 3.0 * u * u};
  const double hx_der[2] = {hx * (u * u * u - 2.0 * u * u + u),
                            hx * (u * u * u - u * u)};
  const double hy_val[2] = {2.0 * v * v * v - 3.0 * v * v + 1.0,
                            -2.0 * v * v * v + 3.0 * v * v};
  const double hy_der[2] = {hy * (v * v * v - 2.0 * v * v + v),
                            hy * (v * v * v - v * v)};

  const double dhx_val_dx[2] = {(6.0 * u * u - 6.0 * u) / hx,
                                (-6.0 * u * u + 6.0 * u) / hx};
  const double dhx_der_dx[2] = {3.0 * u * u - 4.0 * u + 1.0,
                                3.0 * u * u - 2.0 * u};
  const double dhy_val_dy[2] = {(6.0 * v * v - 6.0 * v) / hy,
                                (-6.0 * v * v + 6.0 * v) / hy};
  const double dhy_der_dy[2] = {3.0 * v * v - 4.0 * v + 1.0,
                                3.0 * v * v - 2.0 * v};

  const double ddhx_val_dxx[2] = {(12.0 * u - 6.0) / (hx * hx),
                                  (-12.0 * u + 6.0) / (hx * hx)};
  const double ddhx_der_dxx[2] = {(6.0 * u - 4.0) / hx, (6.0 * u - 2.0) / hx};
  const double ddhy_val_dyy[2] = {(12.0 * v - 6.0) / (hy * hy),
                                  (-12.0 * v + 6.0) / (hy * hy)};
  const double ddhy_der_dyy[2] = {(6.0 * v - 4.0) / hy, (6.0 * v - 2.0) / hy};

  const int ii[2] = {i0, i1};
  const int jj[2] = {j0, j1};
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      const int idx = helmholtz_flat_index(spline, ii[a], jj[b]);
      const double q = nodes[idx];
      const double qx = dx_nodes[idx];
      const double qy = dy_nodes[idx];
      const double qxy = dxdy_nodes[idx];

      out.q += hx_val[a] * hy_val[b] * q + hx_der[a] * hy_val[b] * qx +
               hx_val[a] * hy_der[b] * qy + hx_der[a] * hy_der[b] * qxy;
      out.q_x += dhx_val_dx[a] * hy_val[b] * q + dhx_der_dx[a] * hy_val[b] * qx +
                 dhx_val_dx[a] * hy_der[b] * qy + dhx_der_dx[a] * hy_der[b] * qxy;
      out.q_y += hx_val[a] * dhy_val_dy[b] * q + hx_der[a] * dhy_val_dy[b] * qx +
                 hx_val[a] * dhy_der_dy[b] * qy + hx_der[a] * dhy_der_dy[b] * qxy;
      out.q_xx += ddhx_val_dxx[a] * hy_val[b] * q +
                  ddhx_der_dxx[a] * hy_val[b] * qx +
                  ddhx_val_dxx[a] * hy_der[b] * qy +
                  ddhx_der_dxx[a] * hy_der[b] * qxy;
      out.q_yy += hx_val[a] * ddhy_val_dyy[b] * q +
                  hx_der[a] * ddhy_val_dyy[b] * qx +
                  hx_val[a] * ddhy_der_dyy[b] * qy +
                  hx_der[a] * ddhy_der_dyy[b] * qxy;
      out.q_xy += dhx_val_dx[a] * dhy_val_dy[b] * q +
                  dhx_der_dx[a] * dhy_val_dy[b] * qx +
                  dhx_val_dx[a] * dhy_der_dy[b] * qy +
                  dhx_der_dx[a] * dhy_der_dy[b] * qxy;
    }
  }
  return out;
}

__device__ inline HelmholtzSplineThermo device_helmholtz_eval_thermo(
    const HelmholtzSplineDeviceView& spline,
    const double rho,
    const double logT) {
  HelmholtzSplineThermo out{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (!(rho > 0.0) || spline.n_rho < 2 || spline.n_T < 2) {
    return out;
  }

  const double log_rho = log(rho);
  const HelmholtzSplineEval p_eval = device_helmholtz_eval_scalar(spline,
                                                                  log_rho,
                                                                  logT,
                                                                  spline.P_nodes,
                                                                  spline.P_dx_nodes,
                                                                  spline.P_dy_nodes,
                                                                  spline.P_dxdy_nodes);
  const HelmholtzSplineEval e_eval = device_helmholtz_eval_scalar(spline,
                                                                  log_rho,
                                                                  logT,
                                                                  spline.e_nodes,
                                                                  spline.e_dx_nodes,
                                                                  spline.e_dy_nodes,
                                                                  spline.e_dxdy_nodes);
  const double T = exp(logT);
  const double cv = fmax(e_eval.q_y / fmax(T, 1.0e-30), 1.0e-30);
  const double dP_drho_T = p_eval.q_x / fmax(rho, 1.0e-30);
  const double dP_dT_rho = p_eval.q_y / fmax(T, 1.0e-30);
  const double cs2 =
      dP_drho_T + T * dP_dT_rho * dP_dT_rho / fmax(rho * rho * cv, 1.0e-30);

  out.pressure = p_eval.q;
  out.energy = e_eval.q;
  out.cv = cv;
  out.dP_drho_T = dP_drho_T;
  out.dP_dT_rho = dP_dT_rho;
  out.sound_speed = (cs2 > 0.0 && isfinite(cs2)) ? sqrt(cs2) : 0.0;
  return out;
}

__device__ inline double device_helmholtz_T_from_e(
    const HelmholtzSplineDeviceView& spline,
    const double rho,
    const double e_target) {
  if (!(rho > 0.0) || spline.n_rho < 2 || spline.n_T < 2) {
    return 0.0;
  }

  const double log_rho = clamp_log_value(log(rho), spline.log_rho_min, spline.log_rho_max);
  double y_lo = spline.log_T_min;
  double y_hi = spline.log_T_max;
  double e_lo = 0.0;
  double e_hi = 0.0;
  bool found = false;

  double y_prev = spline.log_T_grid[0];
  double e_prev = device_helmholtz_eval_scalar(spline,
                                               log_rho,
                                               y_prev,
                                               spline.e_nodes,
                                               spline.e_dx_nodes,
                                               spline.e_dy_nodes,
                                               spline.e_dxdy_nodes)
                      .q;
  if (e_target <= e_prev) {
    return exp(y_prev);
  }

  for (int j = 1; j < spline.n_T; ++j) {
    const double y_curr = spline.log_T_grid[j];
    const double e_curr = device_helmholtz_eval_scalar(spline,
                                                       log_rho,
                                                       y_curr,
                                                       spline.e_nodes,
                                                       spline.e_dx_nodes,
                                                       spline.e_dy_nodes,
                                                       spline.e_dxdy_nodes)
                              .q;
    if ((e_target >= e_prev && e_target <= e_curr) ||
        (e_target <= e_prev && e_target >= e_curr)) {
      y_lo = y_prev;
      y_hi = y_curr;
      e_lo = e_prev;
      e_hi = e_curr;
      found = true;
      break;
    }
    y_prev = y_curr;
    e_prev = e_curr;
  }
  if (!found) {
    return exp(y_prev);
  }
  if (e_hi < e_lo) {
    const double y_tmp = y_lo;
    y_lo = y_hi;
    y_hi = y_tmp;
    const double e_tmp = e_lo;
    e_lo = e_hi;
    e_hi = e_tmp;
  }
  if (e_target <= e_lo) {
    return exp(y_lo);
  }
  if (e_target >= e_hi) {
    return exp(y_hi);
  }

  double y = y_lo + (e_target - e_lo) * (y_hi - y_lo) / fmax(e_hi - e_lo, 1.0e-30);
  for (int iter = 0; iter < 32; ++iter) {
    const HelmholtzSplineEval e_eval = device_helmholtz_eval_scalar(spline,
                                                                    log_rho,
                                                                    y,
                                                                    spline.e_nodes,
                                                                    spline.e_dx_nodes,
                                                                    spline.e_dy_nodes,
                                                                    spline.e_dxdy_nodes);
    const double g = e_eval.q - e_target;
    if (fabs(g) <= 1.0e-10 * fmax(fabs(e_target), 1.0)) {
      return exp(y);
    }
    if (g > 0.0) {
      y_hi = y;
    } else {
      y_lo = y;
    }
    const double gp = e_eval.q_y;
    double y_new = 0.5 * (y_lo + y_hi);
    if (fabs(gp) > 1.0e-30 && isfinite(gp)) {
      const double y_trial = y - g / gp;
      if (y_trial > y_lo && y_trial < y_hi) {
        y_new = y_trial;
      }
    }
    y = y_new;
  }
  return exp(y);
}

#endif  // __CUDACC__

}  // namespace tenryu::materials
