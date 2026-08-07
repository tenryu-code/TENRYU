#pragma once

#include <cmath>

#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

struct HelmholtzJetDeviceView {
  const double* log_rho_grid;   // [n_rho]
  const double* log_T_grid;     // [n_T]
  const double* cell_coeffs;    // [(n_rho-1)*(n_T-1)*36]
  int n_rho;
  int n_T;
  double log_rho_min;
  double log_rho_max;
  double log_T_min;
  double log_T_max;
  double d_log_rho_inv;
  double d_log_T_inv;
};

struct HelmholtzJetDeviceEval {
  double phi;
  double phi_x;
  double phi_y;
  double phi_xx;
  double phi_yy;
  double phi_xy;
};

struct HelmholtzJetDeviceThermo {
  double pressure;
  double energy;
  double cv;
  double dP_drho_T;
  double dP_dT_rho;
  double sound_speed;
};

#ifdef __CUDACC__

__device__ inline void helmholtz_jet_basis(const double t,
                                           double* b,
                                           double* db,
                                           double* ddb) {
  const double t2 = t * t;
  const double t3 = t2 * t;
  const double t4 = t3 * t;
  const double t5 = t4 * t;

  b[0] = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5;
  b[1] = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5;
  b[2] = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5;
  b[3] = 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
  b[4] = -4.0 * t3 + 7.0 * t4 - 3.0 * t5;
  b[5] = 0.5 * t3 - t4 + 0.5 * t5;

  db[0] = -30.0 * t2 + 60.0 * t3 - 30.0 * t4;
  db[1] = 1.0 - 18.0 * t2 + 32.0 * t3 - 15.0 * t4;
  db[2] = t - 4.5 * t2 + 6.0 * t3 - 2.5 * t4;
  db[3] = 30.0 * t2 - 60.0 * t3 + 30.0 * t4;
  db[4] = -12.0 * t2 + 28.0 * t3 - 15.0 * t4;
  db[5] = 1.5 * t2 - 4.0 * t3 + 2.5 * t4;

  ddb[0] = -60.0 * t + 180.0 * t2 - 120.0 * t3;
  ddb[1] = -36.0 * t + 96.0 * t2 - 60.0 * t3;
  ddb[2] = 1.0 - 9.0 * t + 18.0 * t2 - 10.0 * t3;
  ddb[3] = 60.0 * t - 180.0 * t2 + 120.0 * t3;
  ddb[4] = -24.0 * t + 84.0 * t2 - 60.0 * t3;
  ddb[5] = 3.0 * t - 12.0 * t2 + 10.0 * t3;
}

__device__ inline HelmholtzJetDeviceEval device_helmholtz_jet_eval(
    const HelmholtzJetDeviceView& jet,
    const double log_rho_raw,
    const double log_T_raw) {
  HelmholtzJetDeviceEval out{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (jet.n_rho < 2 || jet.n_T < 2 || jet.cell_coeffs == nullptr) {
    return out;
  }

  const double log_rho = clamp_log_value(log_rho_raw, jet.log_rho_min, jet.log_rho_max);
  const double log_T = clamp_log_value(log_T_raw, jet.log_T_min, jet.log_T_max);
  const RhoBracket rb_x = find_bracket_from_log(jet.log_rho_grid,
                                                jet.n_rho,
                                                log_rho,
                                                jet.log_rho_min,
                                                jet.log_rho_max,
                                                jet.d_log_rho_inv);
  const RhoBracket rb_y = find_bracket_from_log(jet.log_T_grid,
                                                jet.n_T,
                                                log_T,
                                                jet.log_T_min,
                                                jet.log_T_max,
                                                jet.d_log_T_inv);

  const int i0 = clamp_int(rb_x.i0, 0, jet.n_rho - 2);
  const int j0 = clamp_int(rb_y.i0, 0, jet.n_T - 2);
  const double x0 = table_log_coord(jet.log_rho_grid,
                                    jet.n_rho,
                                    i0,
                                    jet.log_rho_min,
                                    jet.log_rho_max);
  const double x1 = table_log_coord(jet.log_rho_grid,
                                    jet.n_rho,
                                    i0 + 1,
                                    jet.log_rho_min,
                                    jet.log_rho_max);
  const double y0 = table_log_coord(jet.log_T_grid,
                                    jet.n_T,
                                    j0,
                                    jet.log_T_min,
                                    jet.log_T_max);
  const double y1 = table_log_coord(jet.log_T_grid,
                                    jet.n_T,
                                    j0 + 1,
                                    jet.log_T_min,
                                    jet.log_T_max);
  const double hx = fmax(x1 - x0, 1.0e-30);
  const double hy = fmax(y1 - y0, 1.0e-30);
  const double s = clamp01((log_rho - x0) / hx);
  const double t = clamp01((log_T - y0) / hy);

  double bx[6], dbx[6], ddbx[6];
  double by[6], dby[6], ddby[6];
  helmholtz_jet_basis(s, bx, dbx, ddbx);
  helmholtz_jet_basis(t, by, dby, ddby);

  const int cell = (j0 * (jet.n_rho - 1) + i0) * 36;
  for (int ax = 0; ax < 6; ++ax) {
    for (int ay = 0; ay < 6; ++ay) {
      const double c = jet.cell_coeffs[cell + ax * 6 + ay];
      out.phi += c * bx[ax] * by[ay];
      out.phi_x += c * dbx[ax] * by[ay] / hx;
      out.phi_y += c * bx[ax] * dby[ay] / hy;
      out.phi_xx += c * ddbx[ax] * by[ay] / (hx * hx);
      out.phi_yy += c * bx[ax] * ddby[ay] / (hy * hy);
      out.phi_xy += c * dbx[ax] * dby[ay] / (hx * hy);
    }
  }
  return out;
}

__device__ inline HelmholtzJetDeviceThermo device_helmholtz_jet_eval_thermo(
    const HelmholtzJetDeviceView& jet,
    const double rho,
    const double logT) {
  HelmholtzJetDeviceThermo out{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  if (!(rho > 0.0) || jet.n_rho < 2 || jet.n_T < 2) {
    return out;
  }
  const double T = exp(logT);
  const HelmholtzJetDeviceEval phi = device_helmholtz_jet_eval(jet, log(rho), logT);
  const double cv = -(phi.phi_y + phi.phi_yy);
  const double dP_drho_T = T * (phi.phi_x + phi.phi_xx);
  const double dP_dT_rho = rho * (phi.phi_x + phi.phi_xy);
  const double cs2 =
      dP_drho_T + T * dP_dT_rho * dP_dT_rho /
                      fmax(rho * rho * fmax(cv, 1.0e-30), 1.0e-30);
  out.pressure = rho * T * phi.phi_x;
  out.energy = -T * phi.phi_y;
  out.cv = cv;
  out.dP_drho_T = dP_drho_T;
  out.dP_dT_rho = dP_dT_rho;
  out.sound_speed = (cs2 > 0.0 && isfinite(cs2)) ? sqrt(cs2) : 0.0;
  return out;
}

__device__ inline double device_helmholtz_jet_T_from_e(const HelmholtzJetDeviceView& jet,
                                                       const double rho,
                                                       const double e_target) {
  if (!(rho > 0.0) || jet.n_rho < 2 || jet.n_T < 2) {
    return 0.0;
  }
  const double log_rho = clamp_log_value(log(rho), jet.log_rho_min, jet.log_rho_max);
  double y_prev = jet.log_T_grid[0];
  double e_prev = device_helmholtz_jet_eval_thermo(jet, rho, y_prev).energy;
  if (e_target <= e_prev) {
    return exp(y_prev);
  }
  double y_lo = y_prev;
  double y_hi = jet.log_T_grid[jet.n_T - 1];
  double e_lo = e_prev;
  double e_hi = device_helmholtz_jet_eval_thermo(jet, rho, y_hi).energy;
  bool found = false;
  for (int j = 1; j < jet.n_T; ++j) {
    const double y_curr = jet.log_T_grid[j];
    const double e_curr = device_helmholtz_jet_eval_thermo(jet, rho, y_curr).energy;
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
    return exp(y_hi);
  }
  if (e_hi < e_lo) {
    const double ty = y_lo;
    y_lo = y_hi;
    y_hi = ty;
    const double te = e_lo;
    e_lo = e_hi;
    e_hi = te;
  }
  double y = y_lo + (e_target - e_lo) * (y_hi - y_lo) / fmax(e_hi - e_lo, 1.0e-30);
  for (int iter = 0; iter < 32; ++iter) {
    const HelmholtzJetDeviceThermo thermo = device_helmholtz_jet_eval_thermo(jet, rho, y);
    const double g = thermo.energy - e_target;
    if (fabs(g) <= 1.0e-10 * fmax(fabs(e_target), 1.0)) {
      return exp(y);
    }
    if (g > 0.0) {
      y_hi = y;
    } else {
      y_lo = y;
    }
    const double gp = exp(y) * thermo.cv;
    double y_new = 0.5 * (y_lo + y_hi);
    if (fabs(gp) > 1.0e-30 && isfinite(gp)) {
      const double y_trial = y - g / gp;
      if (y_trial > y_lo && y_trial < y_hi) {
        y_new = y_trial;
      }
    }
    y = y_new;
  }
  return exp(0.5 * (y_lo + y_hi));
}

#endif

}  // namespace tenryu::materials
