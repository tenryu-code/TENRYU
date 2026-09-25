#pragma once

#include <cmath>

#include "materials/cold_equilibrium.hpp"
#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

#ifdef __CUDACC__

// Warp-cooperative forms of the table inverse T(rho, e) of eos_device_table.cuh
// (device_inverse_reclose, device_inverse_reclose_with_high_t_tail,
// device_eos_T_from_e_monotone) and of the node search of cold_inverse_Te
// (cold_equilibrium.hpp). Every lane of a full warp calls them with the same
// arguments. They perform the sequential form's floating-point operations on
// the same values, so they take the same decisions (rows, nodes, branches,
// clamps) and return the same results to rounding: the compiler may contract a
// multiply-add differently in another kernel (in the 1D closure the corrected
// electron pressure of cold cells differed by up to 1.2e-14 relative, the other
// outputs were bitwise equal). Only the bisections over table rows and
// temperature nodes change their schedule: the
// midpoints of the next five bisection levels -- every node of the bisection
// tree below the current bracket, each from its own bracket by the same
// integer halvings -- are evaluated in parallel, one per lane, and the bracket
// then descends the tree with the sequential loop's decisions, five levels per
// round instead of one. Pairs of evaluations of one expression at two points
// (the energies at the table's lowest and highest temperature, F at the two
// bracket ends, their logarithms, the two node searches) run on lanes 0 and 1
// at once and are broadcast. A single thread's chain of dependent table loads and
// FP64 exp/log per level set the time of the closure of a small mesh (NIF DS,
// 360 cells: the electron inverse of the cold-equilibrium table took about
// 80k cycles per cell, the ion inverse 21k). The sequential forms remain the
// reference (tests/materials/test_eos_inverse_warp.cu: the same flags, values
// within 1e-13 relative).

constexpr unsigned kEosWarpFullMask = 0xffffffffu;

// Bracket (a, b) of node `id` of the bisection tree below the bracket
// (lo, hi) (heap order: root 1, children 2 id and 2 id + 1, the right child
// taking a = m, the left b = m, with m = a + ((b - a) >> 1)); false when the
// sequential loop stops before the node (b - a <= 1 on the way to it or at it).
__device__ inline bool eos_warp_tree_node_bracket(const int id,
                                                  const int lo,
                                                  const int hi,
                                                  int* a_out,
                                                  int* b_out) {
  const int depth = 31 - __clz(id);
  int a = lo;
  int b = hi;
  for (int bit = depth - 1; bit >= 0; --bit) {
    if (b - a <= 1) {
      return false;
    }
    const int m = a + ((b - a) >> 1);
    if (((id >> bit) & 1) != 0) {
      a = m;
    } else {
      b = m;
    }
  }
  *a_out = a;
  *b_out = b;
  return b - a > 1;
}

// The row bisection of device_eos_T_from_e_monotone: the largest row lo in
// [lo, hi) with mixed_e_row(lo) <= e_target, by the sequential loop's
// decisions (mid = lo + ((hi - lo) >> 1); mixed_e_row(mid) <= e_target moves
// lo, else hi).
__device__ inline int eos_warp_row_bisection(const DeviceEOSTableView& tab,
                                             const int i0,
                                             const int i1,
                                             const double wr,
                                             const double e_target,
                                             int lo,
                                             int hi,
                                             const int lane) {
  while (hi - lo > 1) {
    double e_node = 0.0;
    int a = 0;
    int b = 0;
    if (lane < 31 && eos_warp_tree_node_bracket(lane + 1, lo, hi, &a, &b)) {
      e_node = mixed_e_row(tab, i0, i1, wr, a + ((b - a) >> 1));
    }
    int node = 1;
    for (int level = 0; level < 5 && hi - lo > 1; ++level) {
      const int mid = lo + ((hi - lo) >> 1);
      const double e_mid = __shfl_sync(kEosWarpFullMask, e_node, node - 1);
      if (e_mid <= e_target) {
        lo = mid;
        node = 2 * node + 1;
      } else {
        hi = mid;
        node = 2 * node;
      }
    }
  }
  return lo;
}

// cold_inverse_Te with the node bisection on the warp (the other paths --
// the bracket checks, the gate's Newton iteration in u, the iteration without
// nodes -- are the sequential code, run by every lane).
template <class EBase, class CvBase>
__device__ inline ColdInverseResult cold_inverse_Te_warp(const ColdEquilibriumView& view,
                                                         const double rho,
                                                         const double qe_target,
                                                         const double T_lo,
                                                         const double T_hi,
                                                         const double T_hint,
                                                         EBase&& ee_base,
                                                         CvBase&& cv_base,
                                                         const double* log_T_nodes,
                                                         const int n_nodes,
                                                         const double* C_given,
                                                         const int lane) {
  ColdInverseResult out{T_lo, 4, 0};
  if (!(T_hi > T_lo) || !(T_lo > 0.0) || !(qe_target == qe_target) || !(rho > 0.0)) {
    return out;
  }
  const double C = (C_given != nullptr) ? *C_given : cold_reference(view, rho).C;
  const auto G = [&](const double T) {
    const ColdGate g = cold_gate(view, T);
    return (g.w - T * g.dw - 1.0) * C;
  };
  auto F = [&](const double T) {
    const ColdGate g = cold_gate(view, T);
    return ee_base(T) + (g.w - T * g.dw - 1.0) * C - qe_target;
  };
  // F at both bracket ends on lanes 0 and 1 (the sequential code evaluates
  // F_hi only when F_lo < 0; the value is the same).
  const double F_pick = F((lane == 1) ? T_hi : T_lo);
  const double F_lo = __shfl_sync(kEosWarpFullMask, F_pick, 0);
  const double F_hi = __shfl_sync(kEosWarpFullMask, F_pick, 1);
  if (F_lo == 0.0) {
    out.status = 0;
    return out;
  }
  if (F_lo > 0.0) {
    out.status = 1;
    return out;
  }
  if (F_hi == 0.0) {
    out.T = T_hi;
    out.status = 0;
    return out;
  }
  if (F_hi < 0.0) {
    out.T = T_hi;
    out.status = 2;
    return out;
  }
  const int max_it = (view.max_iterations > 0) ? view.max_iterations : 80;
  if (log_T_nodes != nullptr && n_nodes >= 2) {
    // u_lo and u_hi on lanes 0 and 1; then lane 0 searches the first node
    // with u > u_lo (k0) and lane 1 the last node with u < u_hi (k1), each
    // with its sequential loop.
    const double u_pick = log((lane == 1) ? T_hi : T_lo);
    const double u_lo = __shfl_sync(kEosWarpFullMask, u_pick, 0);
    const double u_hi = __shfl_sync(kEosWarpFullMask, u_pick, 1);
    int k_pick = 0;
    {
      int lo = -1;
      int hi = n_nodes;
      while (hi - lo > 1) {
        const int mid = lo + ((hi - lo) >> 1);
        const double u_mid = log_T_nodes[mid];
        const bool upper = (lane == 1) ? !(u_mid < u_hi) : (u_mid > u_lo);
        if (upper) {
          hi = mid;
        } else {
          lo = mid;
        }
      }
      k_pick = (lane == 1) ? lo : hi;
    }
    const int k0 = __shfl_sync(kEosWarpFullMask, k_pick, 0);
    const int k1 = __shfl_sync(kEosWarpFullMask, k_pick, 1);
    double TL = T_lo;
    double FL = F_lo;
    double TR = T_hi;
    double FR = F_hi;
    int a = k0 - 1;
    int b = k1 + 1;
    int evaluations = 0;
    while (b - a > 1) {
      double T_node = 0.0;
      double f_node = 0.0;
      int na = 0;
      int nb = 0;
      if (lane < 31 && eos_warp_tree_node_bracket(lane + 1, a, b, &na, &nb)) {
        T_node = exp(log_T_nodes[na + ((nb - na) >> 1)]);
        f_node = F(T_node);
      }
      int node = 1;
      for (int level = 0; level < 5 && b - a > 1; ++level) {
        const int m = a + ((b - a) >> 1);
        const double Tm = __shfl_sync(kEosWarpFullMask, T_node, node - 1);
        const double fm = __shfl_sync(kEosWarpFullMask, f_node, node - 1);
        ++evaluations;
        if (fm < 0.0) {
          a = m;
          TL = Tm;
          FL = fm;
          node = 2 * node + 1;
        } else {
          b = m;
          TR = Tm;
          FR = fm;
          node = 2 * node;
        }
      }
    }
    out.iterations = evaluations;
    out.status = 0;
    if (FR == 0.0) {
      out.T = TR;
      return out;
    }
    const double u_end = log((lane == 1) ? TR : TL);
    double uL = __shfl_sync(kEosWarpFullMask, u_end, 0);
    double uR = __shfl_sync(kEosWarpFullMask, u_end, 1);
    const bool gate_constant =
        !(view.T_star > view.T_a) || TR <= view.T_a || TL >= view.T_star;
    if (gate_constant) {
      const double u = uL + (-FL) * (uR - uL) / (FR - FL);
      out.T = fmin(fmax(exp(u), TL), TR);
      return out;
    }
    const double s = ((FR - G(TR)) - (FL - G(TL))) / (uR - uL);
    double u = uL + (-FL) * (uR - uL) / (FR - FL);
    for (int it = 0; it < max_it; ++it) {
      const double T = exp(u);
      const double f = F(T);
      ++out.iterations;
      if (f == 0.0) {
        out.T = T;
        return out;
      }
      if (f < 0.0) {
        uL = u;
        TL = T;
      } else {
        uR = u;
        TR = T;
      }
      if ((TR - TL) <= 4.0e-15 * TR) {
        out.T = TR;
        return out;
      }
      const ColdGate g = cold_gate(view, T);
      const double D = s - T * T * g.d2w * C;
      double u_new = 0.5 * (uL + uR);
      if (D > 0.0) {
        const double u_newton = u - f / D;
        if (u_newton > uL && u_newton < uR) {
          u_new = u_newton;
        }
      }
      if (fabs(u_new - u) <= 4.0e-15) {
        out.T = exp(u_new);
        return out;
      }
      u = u_new;
    }
    out.T = TR;
    out.status = 3;
    return out;
  }
  double L = T_lo;
  double R = T_hi;
  double T = (T_hint == T_hint && T_hint > L && T_hint < R) ? T_hint : sqrt(L * R);
  int stall = 0;
  double sign_prev = 0.0;
  bool newton_prev = false;
  for (int it = 1; it <= max_it; ++it) {
    out.iterations = it;
    const double f = F(T);
    const double sign = (f < 0.0) ? -1.0 : 1.0;
    if (f < 0.0) {
      L = T;
    } else {
      R = T;
    }
    if ((R - L) <= 4.0e-15 * R) {
      out.T = R;
      out.status = 0;
      return out;
    }
    stall = (newton_prev && sign == sign_prev) ? stall + 1 : 0;
    sign_prev = sign;
    double T_new = sqrt(L * R);
    newton_prev = false;
    if (f != 0.0 && stall < 2) {
      const ColdGate g = cold_gate(view, T);
      const double cv = cv_base(T) - T * g.d2w * C;
      if (cv > 0.0) {
        const double T_newton = T - f / cv;
        if (T_newton > L && T_newton < R && T_newton != T) {
          T_new = T_newton;
          newton_prev = true;
        }
      }
    }
    T = T_new;
  }
  out.T = R;
  out.status = 3;
  return out;
}

// device_eos_T_from_e_monotone on the warp. rho_bracket: the density of the
// bracket (device_rho_from_bracket(tab, rb)), used by the cold branch.
__device__ inline double device_eos_T_from_e_monotone_warp(const DeviceEOSTableView& tab,
                                                           const RhoBracket& rb,
                                                           const double e_target,
                                                           const ColdReference* cref,
                                                           const double rho_bracket,
                                                           const int lane) {
  if (tab.n_rho <= 0 || tab.n_T <= 0 || tab.e_table == nullptr) {
    return 0.0;
  }

  if (cold_enabled(tab.cold)) {
    const double rho = rho_bracket;
    const double T_pick = exp((lane == 1) ? tab.log_T_max : tab.log_T_min);
    const double T_min = __shfl_sync(kEosWarpFullMask, T_pick, 0);
    const double T_max = __shfl_sync(kEosWarpFullMask, T_pick, 1);
    const ColdInverseResult inv = cold_inverse_Te_warp(
        tab.cold, rho, e_target, T_min, T_max, 0.0,
        [&](const double T) { return device_eos_energy_base(tab, rb, log(T)); },
        [&](const double T) { return device_eos_cv_base(tab, rb, log(T)); },
        tab.log_T_grid, tab.n_T, (cref != nullptr) ? &cref->C : nullptr, lane);
    return inv.T;
  }

  int i0 = 0;
  int i1 = 0;
  double wr = 0.0;
  if (tab.n_rho == 1) {
    i0 = 0;
    i1 = 0;
    wr = 0.0;
  } else {
    i0 = clamp_int(rb.i0, 0, tab.n_rho - 2);
    i1 = clamp_int(rb.i1, i0 + 1, tab.n_rho - 1);
    wr = clamp01(rb.w);
  }

  if (tab.n_T == 1) {
    const double logT0 = table_log_coord(
        tab.log_T_grid, tab.n_T, 0, tab.log_T_min, tab.log_T_max);
    return exp(logT0);
  }

  if (!isfinite(e_target)) {
    const double logT0 = table_log_coord(
        tab.log_T_grid, tab.n_T, 0, tab.log_T_min, tab.log_T_max);
    return exp(logT0);
  }

  // The first and last rows' energies on lanes 0 and 1.
  const int j_last = tab.n_T - 1;
  const double e_pick = mixed_e_row(tab, i0, i1, wr, (lane == 1) ? j_last : 0);
  const double e_min = __shfl_sync(kEosWarpFullMask, e_pick, 0);
  const double e_max = __shfl_sync(kEosWarpFullMask, e_pick, 1);
  if (e_target < e_min) {
    const double logT0 = table_log_coord(
        tab.log_T_grid, tab.n_T, 0, tab.log_T_min, tab.log_T_max);
    return exp(logT0);
  }

  if (e_target > e_max) {
    const double logT1 = table_log_coord(
        tab.log_T_grid, tab.n_T, j_last, tab.log_T_min, tab.log_T_max);
    return exp(logT1);
  }

  const int lo = eos_warp_row_bisection(tab, i0, i1, wr, e_target, 0, j_last, lane);

  const int j0 = lo;
  const int j1 = lo + 1;
  const double e0 = mixed_e_row(tab, i0, i1, wr, j0);
  const double e1 = mixed_e_row(tab, i0, i1, wr, j1);
  int idx = -1;
  if (e1 == e_target) {
    idx = j1;
  } else if (e0 == e_target) {
    idx = j0;
  }
  if (idx >= 0) {
    while (idx > 0 && mixed_e_row(tab, i0, i1, wr, idx - 1) == e_target) {
      --idx;
    }
    const double logT_p = table_log_coord(
        tab.log_T_grid, tab.n_T, idx, tab.log_T_min, tab.log_T_max);
    return exp(logT_p);
  }
  const double alpha = (e1 > e0) ? clamp01((e_target - e0) / (e1 - e0)) : 0.0;

  const double logT0 = table_log_coord(
      tab.log_T_grid, tab.n_T, j0, tab.log_T_min, tab.log_T_max);
  const double logT1 = table_log_coord(
      tab.log_T_grid, tab.n_T, j1, tab.log_T_min, tab.log_T_max);
  const double logT = logT0 + alpha * (logT1 - logT0);
  return exp(logT);
}

// device_inverse_reclose on the warp.
__device__ inline DeviceEOSInverseRecloseResult device_inverse_reclose_warp(
    const DeviceEOSTableView& tab,
    const double rho,
    const double e_target,
    const double T_floor,
    const int lane) {
  DeviceEOSInverseRecloseResult out{};
  out.T = fmax(T_floor, 1.0e-30);
  out.logT = log(out.T);
  out.pressure = 0.0;
  out.energy = 0.0;
  out.cv = 0.0;
  out.bracket_failure = 0;
  out.fallback_used = 0;
  out.lower_clamp = 0;
  out.upper_clamp = 0;
  out.floor_clamp = 0;

  if (tab.n_rho <= 0 || tab.n_T <= 0 || tab.e_table == nullptr ||
      !isfinite(rho) || !(rho > 0.0) || !isfinite(e_target)) {
    out.bracket_failure = 1;
    return out;
  }

  const RhoBracket rb = find_rho_bracket(tab, rho);
  int i0 = 0;
  int i1 = 0;
  double wr = 0.0;
  if (tab.n_rho == 1) {
    i0 = 0;
    i1 = 0;
    wr = 0.0;
  } else {
    i0 = clamp_int(rb.i0, 0, tab.n_rho - 2);
    i1 = clamp_int(rb.i1, i0 + 1, tab.n_rho - 1);
    wr = clamp01(rb.w);
  }

  const bool cold = cold_enabled(tab.cold);
  // device_cold_reference_at(tab, rb), with the bracket's density kept for
  // the inversion (which evaluates the same device_rho_from_bracket).
  const double rho_bracket = cold ? device_rho_from_bracket(tab, rb) : 0.0;
  const ColdReference cref = cold ? cold_reference(tab.cold, rho_bracket) : ColdReference{};
  // The energies at the lowest and highest table temperature on lanes 0 and 1.
  const double e_pick = cold ? device_eos_energy_with(tab, rb,
                                                      (lane == 1) ? tab.log_T_max : tab.log_T_min,
                                                      cref)
                             : mixed_e_row(tab, i0, i1, wr, (lane == 1) ? (tab.n_T - 1) : 0);
  const double e_min = __shfl_sync(kEosWarpFullMask, e_pick, 0);
  const double e_max = __shfl_sync(kEosWarpFullMask, e_pick, 1);
  const bool bad_bracket = !isfinite(e_min) || !isfinite(e_max) || !(e_max > e_min);
  if (bad_bracket) {
    out.bracket_failure = 1;
  } else if (e_target < e_min) {
    out.lower_clamp = 1;
  } else if (e_target > e_max) {
    out.upper_clamp = 1;
  }

  const double T_raw = device_eos_T_from_e_monotone_warp(tab, rb, e_target,
                                                        cold ? &cref : nullptr, rho_bracket, lane);
  const double T_floor_eff = fmax(T_floor, 1.0e-30);
  out.T = fmax(T_raw, T_floor_eff);
  out.floor_clamp = (isfinite(T_raw) && T_raw < T_floor_eff) ? 1 : 0;
  out.logT = log(fmax(out.T, 1.0e-30));
  const DeviceEOSThermo thermo = device_eos_thermo_with(tab, rb, out.logT, cref);
  out.pressure = thermo.pressure;
  out.energy = thermo.energy;
  out.cv = fmax(thermo.cv, 0.0);
  return out;
}

// device_inverse_reclose_with_high_t_tail on the warp.
__device__ inline DeviceEOSInverseRecloseResult device_inverse_reclose_with_high_t_tail_warp(
    const DeviceEOSTableView& tab,
    const double rho,
    const double e_target,
    const double T_floor,
    const int lane) {
  DeviceEOSInverseRecloseResult out = device_inverse_reclose_warp(tab, rho, e_target, T_floor, lane);
  if (out.bracket_failure != 0 || out.upper_clamp == 0) {
    return out;
  }
  const RhoBracket rb = find_rho_bracket(tab, rho);
  const DeviceEOSHighTTailAnchor a = device_eos_high_t_tail_anchor(tab, rb);
  if (a.valid == 0) {
    return out;
  }
  const double T_tail = a.T_top + (e_target - a.e_top) / a.cv_top;
  if (!isfinite(T_tail) || T_tail <= a.T_top) {
    return out;
  }
  out.T = T_tail;
  out.logT = log(T_tail);
  out.energy = e_target;
  out.pressure = a.P_top * (T_tail / a.T_top);
  out.cv = a.cv_top;
  return out;
}

#endif  // __CUDACC__

}  // namespace tenryu::materials
