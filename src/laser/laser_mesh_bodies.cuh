#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "laser/ib_absorption.cuh"
#include "laser/laser_phys_ext.cuh"

namespace tenryu::laser::laser_mesh_bodies {

constexpr double kProtonMass = tenryu::core::constants::proton_mass;

__device__ inline int locate_cell_1d_device(const double* __restrict__ edges,
                                            const int n_cells,
                                            const double r) {
  if (n_cells <= 0) {
    return 0;
  }
  if (r <= edges[0]) {
    return 0;
  }
  if (r >= edges[n_cells]) {
    return n_cells - 1;
  }

  int lo = 0;
  int hi = n_cells + 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (r < edges[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  const int idx = lo - 1;
  if (idx < 0) {
    return 0;
  }
  if (idx >= n_cells) {
    return n_cells - 1;
  }
  return idx;
}

__device__ inline double clamp_device(const double x, const double lo, const double hi) {
  return fmin(hi, fmax(lo, x));
}

// ---- Radial nodes of the 1D trace profile (2026-09-24) ----
//
// The 1D ray traces read n_hat, n_hat_raw, T_e, Zbar and the IB factor as
// linear functions of r between these nodes. Inside the hydro region (up to
// the outer surface cell) the nodes are every hydro face and cell centre and a
// node just below each upper face: n_e is linear between the cell centres
// (map_hydro_to_laser_1d_kernel_body), so these nodes carry it exactly, and
// the cell-sampled T_e and Zbar change only across the faces. Between the
// centres of the critical-adjacent pair n_hat_raw is log-linear, n_hi
// exp(-lambda (r - r_hi)); the clipped n_hat has its kink where n_hat_raw
// reaches the clip value, which is a node, and the subcritical side carries
// nodes whose spacing dx at distance x from the kink keeps the IB factor's
// linear-interpolation error of the exponential near 1e-3:
// lambda dx <= kTracePairMaxStep and dx <= sqrt(kTracePairKinkCoeff x /
// lambda) (dn/(2 eps) dominates near the kink, where eps = 1 - n_hat ~
// lambda x), from lambda dx = kTracePairFirstStep, at most
// kTracePairNodesMax nodes (the last interval then reaches the subcritical
// centre). The supercritical side needs none (n_hat is the clip value there).
// Beyond the outer surface the laser mesh's graded nodes carry the ghost
// corona and the vacuum. The profile is then a function of the hydro state
// alone inside the hydro region; it used to be the Z = 0 column of the 2D
// laser mesh, whose nodes follow the critical radius and the smallest cell near
// it: on GXII's early corona (steps 30-110) the absorbed power of one beam on
// that column differed by up to 47 % from the hydro-anchored profile refined
// 8 times, against up to 6 % for the hydro-anchored profile with the pair's
// former 15 uniform nodes and no kink node.
constexpr int kTracePairNodesMax = 160;
constexpr double kTraceProfileMinGap = 1.0e-9;
constexpr double kTracePairMaxStep = 0.06;
constexpr double kTracePairKinkCoeff = 0.016;
constexpr double kTracePairFirstStep = 1.0e-4;

// Most nodes build_trace_profile_nodes_1d writes (subdiv >= 1 parts per half
// cell).
TENRYU_HOST_DEVICE inline int trace_profile_node_capacity_1d(const int n_cells,
                                                            const int n_graded_nodes,
                                                            const int subdiv = 1) {
  const int parts = subdiv > 1 ? subdiv : 1;
  return (2 * parts + 1) * (n_cells > 0 ? n_cells : 0) + kTracePairNodesMax +
         (n_graded_nodes > 0 ? n_graded_nodes : 0) + 6;
}

// Rounded product and sum (no contraction into a fused multiply-add), so the
// host and device builds place the nodes identically.
TENRYU_HOST_DEVICE inline double trace_profile_mul(const double a, const double b) {
#ifdef __CUDA_ARCH__
  return __dmul_rn(a, b);
#else
  return a * b;
#endif
}
TENRYU_HOST_DEVICE inline double trace_profile_add(const double a, const double b) {
#ifdef __CUDA_ARCH__
  return __dadd_rn(a, b);
#else
  return a + b;
#endif
}

// n_hat_raw of hydro cell c at its centre, as map_hydro_to_laser_1d_kernel_body
// evaluates it.
TENRYU_HOST_DEVICE inline double trace_profile_cell_n_hat_raw(const double* rho,
                                                             const double* zbar,
                                                             const double* A_eff_cell,
                                                             const int c,
                                                             const double n_crit_safe) {
  return fmax(0.0, rho[c]) * fmax(0.0, zbar[c]) /
         (fmax(A_eff_cell[c], 1.0e-30) * kProtonMass) / n_crit_safe;
}

// Placement state of the profile nodes: the nodes written through this cursor
// (out[0 .. count); out == nullptr counts them without writing), the last two
// node radii held in registers (on the device the serial pass would otherwise
// read back its own global-memory stores for every node) and the rank of the
// last one. rule_count_base is the number of nodes placed before the cursor's
// first (the "fewer than two nodes" condition of the merge rule counts them):
// 0 for the whole profile, 2 for a cell or the tail placed on its own by
// build_trace_profile_nodes_1d_parallel (whose first node the whole profile
// always places as a new node).
struct TraceProfileCursor {
  double* out = nullptr;
  int capacity = 0;
  int count = 0;
  int rule_count_base = 0;
  int last_rank = -1;
  double last = 0.0;
  double before_last = 0.0;
  bool clip = false;
  double r_outer = 0.0;
};

// Consecutive nodes stay at least kTraceProfileMinGap * r apart (an interval of
// a few ulp makes its linear fields meaningless); of two closer nodes the one
// with the higher rank stays: hydro faces, cell centres and the hydro end (2)
// over the pair's kink (1) over the rest (0).
TENRYU_HOST_DEVICE inline void trace_profile_push_ranked(TraceProfileCursor& cur,
                                                        const double r,
                                                        const int rank) {
  if (!(r >= 0.0) || cur.count >= cur.capacity || (cur.clip && r > cur.r_outer)) {
    return;
  }
  const int placed = cur.rule_count_base + cur.count;
  if (placed > 0) {
    const double prev = cur.last;
    if (!(r > prev)) {
      return;
    }
    if (!(r - prev > kTraceProfileMinGap * r)) {
      if (rank > cur.last_rank &&
          (placed < 2 || r - cur.before_last > kTraceProfileMinGap * r)) {
        if (cur.out != nullptr) {
          cur.out[cur.count - 1] = r;
        }
        cur.last = r;
        cur.last_rank = rank;
      }
      return;
    }
  }
  if (cur.out != nullptr) {
    cur.out[cur.count] = r;
  }
  ++cur.count;
  cur.before_last = cur.last;
  cur.last = r;
  cur.last_rank = rank;
}

// The critical-adjacent pair [centre(fcrit - 1), centre(fcrit)] with the map's
// log-linear n_hat_raw (n_hi >= 1 > n_lo > 0): the kink and the subcritical
// nodes, ascending, into pair_r[0 .. n) (at most kTracePairNodesMax); returns n.
TENRYU_HOST_DEVICE inline int trace_profile_pair_nodes_1d(const double* r_edges,
                                                         const std::uint8_t* cell_is_void,
                                                         const int n_cells,
                                                         const int c_last,
                                                         const int fcrit_cell,
                                                         const double* rho,
                                                         const double* zbar,
                                                         const double* A_eff_cell,
                                                         const double n_crit_safe,
                                                         const int critical_clip,
                                                         const double n_hat_margin,
                                                         double* pair_r) {
  int n_pair = 0;
  if (fcrit_cell > 0 && fcrit_cell <= c_last && fcrit_cell < n_cells &&
      cell_is_void[fcrit_cell - 1] == 0U && cell_is_void[fcrit_cell] == 0U) {
    const int c_hi = fcrit_cell - 1;
    const int c_lo = fcrit_cell;
    const double n_hi = trace_profile_cell_n_hat_raw(rho, zbar, A_eff_cell, c_hi, n_crit_safe);
    const double n_lo = trace_profile_cell_n_hat_raw(rho, zbar, A_eff_cell, c_lo, n_crit_safe);
    const double r_hi = 0.5 * (r_edges[c_hi] + r_edges[c_hi + 1]);
    const double r_lo = 0.5 * (r_edges[c_lo] + r_edges[c_lo + 1]);
    const double n_kink = (critical_clip != 0) ? fmin(n_hat_margin, 1.0) : 1.0;
    if (n_hi >= 1.0 && n_lo > 0.0 && n_lo < 1.0 && r_lo > r_hi && n_hi > n_kink &&
        n_kink > n_lo) {
      const double lambda = log(n_hi / n_lo) / (r_lo - r_hi);
      const double r_kink = r_hi + (log(n_hi / n_kink) / lambda);
      const double inv_lambda = 1.0 / lambda;
      if (r_kink > r_hi && r_kink < r_lo && lambda > 0.0) {
        pair_r[n_pair++] = r_kink;
        double x = 0.0;
        const double x_end = r_lo - r_kink;
        while (n_pair < kTracePairNodesMax) {
          const double dx = fmin(kTracePairMaxStep * inv_lambda,
                                 fmax(kTracePairFirstStep * inv_lambda,
                                      sqrt(kTracePairKinkCoeff * x * inv_lambda)));
          x += dx;
          if (!(x < x_end)) {
            break;
          }
          pair_r[n_pair++] = r_kink + x;
        }
      }
    }
  }
  return n_pair;
}

// Interior nodes of (lo, hi): the pair nodes there and, with parts > 1, the
// nodes splitting [lo, hi] into `parts` equal parts, merged in order.
TENRYU_HOST_DEVICE inline void trace_profile_push_interior(TraceProfileCursor& cur,
                                                          const double lo,
                                                          const double hi,
                                                          const double* pair_r,
                                                          const int n_pair,
                                                          const int parts) {
  const double step = trace_profile_mul(trace_profile_add(hi, -lo),
                                        1.0 / static_cast<double>(parts));
  int k = 1;
  // The first pair node above lo (the pair's nodes are strictly increasing):
  // a binary search, where a scan from the start cost every cell beyond the
  // pair all of its nodes.
  int j = 0;
  {
    int hi_j = n_pair;
    while (j < hi_j) {
      const int mid = j + (hi_j - j) / 2;
      if (pair_r[mid] > lo) {
        hi_j = mid;
      } else {
        j = mid + 1;
      }
    }
  }
  while (true) {
    const double x_part = (k < parts)
                              ? trace_profile_add(lo, trace_profile_mul(static_cast<double>(k), step))
                              : hi;
    const double x_pair = (j < n_pair && pair_r[j] < hi) ? pair_r[j] : hi;
    if (!(x_part < hi) && !(x_pair < hi)) {
      break;
    }
    if (x_part <= x_pair) {
      trace_profile_push_ranked(cur, x_part, (x_part == x_pair && j == 0) ? 1 : 0);
      ++k;
      if (x_part == x_pair) {
        ++j;
      }
    } else {
      trace_profile_push_ranked(cur, x_pair, (j == 0) ? 1 : 0);
      ++j;
    }
  }
}

// The nodes of hydro cell c: the lower face, the lower half's interior nodes,
// the centre, the upper half's interior nodes and the node just below the upper
// face (1e-6 of the width, at least 4 minimum gaps below it: it carries the
// cell's T_e and Zbar up to the face). A cell without positive width has none.
TENRYU_HOST_DEVICE inline void trace_profile_push_cell(TraceProfileCursor& cur,
                                                      const double* r_edges,
                                                      const int c,
                                                      const double* pair_r,
                                                      const int n_pair,
                                                      const int parts) {
  const double a = r_edges[c];
  const double b = r_edges[c + 1];
  const double w = trace_profile_add(b, -a);
  if (!(w > 0.0)) {
    return;
  }
  const double m = trace_profile_mul(0.5, trace_profile_add(a, b));
  const double below = trace_profile_add(
      b, -fmax(trace_profile_mul(1.0e-6, w), trace_profile_mul(4.0 * kTraceProfileMinGap, b)));
  trace_profile_push_ranked(cur, a, 2);
  trace_profile_push_interior(cur, a, m, pair_r, n_pair, parts);
  trace_profile_push_ranked(cur, m, 2);
  trace_profile_push_interior(cur, m, below, pair_r, n_pair, parts);
  trace_profile_push_ranked(cur, below, 0);
}

// The nodes after the hydro region: its outer face, the laser mesh's graded
// nodes beyond it (the ghost corona and the vacuum) and the profile's outer
// radius.
TENRYU_HOST_DEVICE inline void trace_profile_push_tail(TraceProfileCursor& cur,
                                                      const double r_hydro_end,
                                                      const double* graded_r,
                                                      const int n_graded) {
  trace_profile_push_ranked(cur, r_hydro_end, 2);
  for (int k = 0; k < n_graded; ++k) {
    if (graded_r[k] > r_hydro_end) {
      trace_profile_push_ranked(cur, graded_r[k], 0);
    }
  }
  if (cur.clip) {
    trace_profile_push_ranked(cur, cur.r_outer, 0);
  }
}

// A cell whose nodes the serial pass decides from the cell's own nodes (see
// build_trace_profile_nodes_1d_parallel_kernel, laser_mesh.cu): finite faces
// and a width of at least 16 parts minimum gaps relative to the upper face.
// The node below the upper face then lies above the cell's other nodes (the
// gap it keeps below the face, max(1e-6 w, 4 kTraceProfileMinGap b), is under
// w / 4) and at least 4 minimum gaps below the next cell's lower face.
TENRYU_HOST_DEVICE inline bool trace_profile_cell_independent(const double a,
                                                             const double b,
                                                             const int parts) {
  constexpr double kLargest = 1.7976931348623157e308;  // finite: |x| <= DBL_MAX (false for NaN)
  const double w = trace_profile_add(b, -a);
  return fabs(a) <= kLargest && fabs(b) <= kLargest &&
         w >= 16.0 * static_cast<double>(parts) * kTraceProfileMinGap * b;
}

// The last hydro cell of the profile.
TENRYU_HOST_DEVICE inline int trace_profile_last_cell_1d(const int n_cells,
                                                        const int outer_surface_cell) {
  return (outer_surface_cell >= 0 && outer_surface_cell < n_cells) ? outer_surface_cell
                                                                   : n_cells - 1;
}

// Writes the strictly increasing profile nodes into out (at most capacity)
// and returns their count. r_edges: hydro faces [n_cells + 1];
// graded_r: the laser mesh's radial nodes [n_graded] (increasing, from 0).
// The profile starts at the laser mesh's first node (r = 0: the laser mesh's
// nodes below the innermost hydro face come first; a slab's (planar) at the
// innermost hydro face) and ends at its outer
// radius graded_r[n_graded - 1] (the launch radius of the rays), as the axis
// column it replaces. rho, zbar and A_eff_cell with n_crit_safe, critical_clip
// and n_hat_margin locate the pair's kink (the map's inputs). subdiv > 1
// splits every half cell into subdiv equal parts (a measure-only refinement;
// the n_e and T_e the nodes carry are the same functions of r).
TENRYU_HOST_DEVICE inline int build_trace_profile_nodes_1d(const double* r_edges,
                                                          const std::uint8_t* cell_is_void,
                                                          const int n_cells,
                                                          const int outer_surface_cell,
                                                          const int fcrit_cell,
                                                          const double* rho,
                                                          const double* zbar,
                                                          const double* A_eff_cell,
                                                          const double n_crit_safe,
                                                          const int critical_clip,
                                                          const double n_hat_margin,
                                                          const double* graded_r,
                                                          const int n_graded,
                                                          double* out,
                                                          const int capacity,
                                                          const int subdiv = 1,
                                                          const bool planar = false) {
  TraceProfileCursor cur;
  cur.out = out;
  cur.capacity = capacity;
  cur.clip = n_graded > 0;
  cur.r_outer = cur.clip ? graded_r[n_graded - 1] : 0.0;
  const int parts = subdiv > 1 ? subdiv : 1;
  if (n_cells <= 0) {
    for (int k = 0; k < n_graded; ++k) {
      trace_profile_push_ranked(cur, graded_r[k], 0);
    }
    return cur.count;
  }
  // A sphere's or cylinder's profile starts at the centre; a slab's at its
  // inner wall (the innermost hydro face).
  for (int k = 0; !planar && k < n_graded && graded_r[k] < r_edges[0]; ++k) {
    trace_profile_push_ranked(cur, graded_r[k], 0);
  }
  const int c_last = trace_profile_last_cell_1d(n_cells, outer_surface_cell);
  double pair_r[kTracePairNodesMax + 1];
  const int n_pair = trace_profile_pair_nodes_1d(r_edges, cell_is_void, n_cells, c_last,
                                                 fcrit_cell, rho, zbar, A_eff_cell, n_crit_safe,
                                                 critical_clip, n_hat_margin, pair_r);
  for (int c = 0; c <= c_last; ++c) {
    trace_profile_push_cell(cur, r_edges, c, pair_r, n_pair, parts);
  }
  trace_profile_push_tail(cur, r_edges[c_last + 1], graded_r, n_graded);
  return cur.count;
}

__device__ inline void map_hydro_to_laser_1d_kernel_body(
    const int idx,
    const double* __restrict__ node_R,
    const double* __restrict__ node_Z,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const double* __restrict__ A_eff_cell,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ r_edges,
    double* __restrict__ n_hat_out,
    double* __restrict__ n_hat_raw_out,
    double* __restrict__ Te_out,
    double* __restrict__ Z_out,
    const int n_nodes_total,
    const int n_nodes_z,
    const int n_cells,
    const double n_crit_safe,
    const int use_ghost_corona,
    const int outer_surface_cell,
    const double ghost_ne_inner,
    const double ghost_scale_length,
    const double ghost_ne_min,
    const double r_surface_outer,
    const double r_ghost_outer,
    const double Te_anchor,
    const double zbar_anchor,
    const double ghost_zbar_min,
    const double ghost_zbar_max,
    const double ghost_Te_min_eV,
    const int critical_clip,
    const double n_hat_margin,
    const int fcrit_cell) {
  const int i = idx / n_nodes_z;
  const int j = idx - i * n_nodes_z;
  const double R = node_R[i];
  const double Z = node_Z[j];
  const double r = sqrt(R * R + Z * Z);

  const int c = locate_cell_1d_device(r_edges, n_cells, r);
  const bool is_void = (cell_is_void[c] != 0U);
  const bool outside_surface = (outer_surface_cell >= 0 && r > r_surface_outer);
  const bool outer_hydro_clamp = outside_surface && !is_void;
  const double rho_c = fmax(0.0, rho[c]);
  double z_c = is_void ? 0.0 : fmax(0.0, zbar[c]);
  double Te_c = Te[c];
  double nh_raw = 0.0;

  if (!is_void && !outer_hydro_clamp) {
    const double A_eff = fmax(A_eff_cell[c], 1.0e-30);
    const double ne_cell_val = rho_c * z_c / (A_eff * kProtonMass);
    // Linear interpolation of n_e in r between adjacent hydro cell
    // centers (2026-07-31 audit): piecewise-constant sampling put a
    // slope*dr/2 staircase on nodal n_hat and a 0/double comb on the
    // segment gradient wherever laser nodes are finer than hydro cells,
    // scattering ray turning points. Linear-in-r interpolation of n_e
    // reproduces linear profiles exactly on any node layout. Te and
    // zbar outputs stay cell-sampled; only n_hat changes.
    double ne_interp = ne_cell_val;
    const double r_c = 0.5 * (r_edges[c] + r_edges[c + 1]);
    int c2 = (r < r_c) ? (c - 1) : (c + 1);
    if (c2 >= 0 && c2 < n_cells && c2 != c &&
        cell_is_void[c2] == 0U &&
        (outer_surface_cell < 0 || c2 <= outer_surface_cell)) {
      const double r_c2 = 0.5 * (r_edges[c2] + r_edges[c2 + 1]);
      const double dr_cc = r_c2 - r_c;
      if (fabs(dr_cc) > 1.0e-300) {
        const double A_eff2 = fmax(A_eff_cell[c2], 1.0e-30);
        const double ne2 =
            fmax(0.0, rho[c2]) * fmax(0.0, zbar[c2]) /
            (A_eff2 * kProtonMass);
        const double t = clamp_device((r - r_c) / dr_cc, 0.0, 1.0);
        ne_interp = ne_cell_val + t * (ne2 - ne_cell_val);
      }
    }
    nh_raw = fmax(0.0, ne_interp / n_crit_safe);

    // Critical-adjacent pair (supercritical centre of fcrit_cell-1 ->
    // subcritical centre of fcrit_cell): one log-linear profile through the
    // two centre values, n_hat = n_hi (n_lo/n_hi)^s. It is continuous at both
    // centres, monotone, and crosses 1 where the host critical-surface
    // estimate places r_crit (the same line). The former substitution only
    // where the linear value was < 1 held n_hat at 1 up to the linear
    // crossing and then dropped it discontinuously (10 | 0.5 neighbours:
    // 1 -> 0.586) (2026-09-23).
    if (fcrit_cell > 0 && fcrit_cell < n_cells &&
        ((c == fcrit_cell && r < r_c) || (c == fcrit_cell - 1 && r >= r_c))) {
      const int c_hi = fcrit_cell - 1;
      const int c_lo = fcrit_cell;
      if (cell_is_void[c_hi] == 0U && cell_is_void[c_lo] == 0U) {
        const double n_hi = fmax(0.0, rho[c_hi]) * fmax(0.0, zbar[c_hi]) /
                            (fmax(A_eff_cell[c_hi], 1.0e-30) * kProtonMass) /
                            n_crit_safe;
        const double n_lo = fmax(0.0, rho[c_lo]) * fmax(0.0, zbar[c_lo]) /
                            (fmax(A_eff_cell[c_lo], 1.0e-30) * kProtonMass) /
                            n_crit_safe;
        const double r_hi = 0.5 * (r_edges[c_hi] + r_edges[c_hi + 1]);
        const double r_lo = 0.5 * (r_edges[c_lo] + r_edges[c_lo + 1]);
        if (n_hi >= 1.0 && n_lo > 0.0 && n_lo < 1.0 && r_lo > r_hi) {
          const double s = clamp_device((r - r_hi) / (r_lo - r_hi), 0.0, 1.0);
          nh_raw = n_hi * exp(s * log(n_lo / n_hi));
        }
      }
    }
  } else if ((use_ghost_corona != 0 && is_void && r >= r_surface_outer &&
              r <= r_ghost_outer) ||
             (outer_hydro_clamp && use_ghost_corona != 0 && r <= r_ghost_outer)) {
    const double dr = fmax(r - r_surface_outer, 0.0);
    nh_raw = ghost_ne_inner * exp(-dr / fmax(ghost_scale_length, 1.0e-30));
    nh_raw = clamp_device(nh_raw, ghost_ne_min, ghost_ne_inner);
    z_c = clamp_device(zbar_anchor, ghost_zbar_min, fmax(ghost_zbar_max, ghost_zbar_min));
    Te_c = Te_anchor;
  } else if (outer_hydro_clamp) {
    z_c = 0.0;
    Te_c = ghost_Te_min_eV;
  }

  double n_hat = fmin(1.0, fmax(0.0, nh_raw));
  if (critical_clip != 0) {
    n_hat = fmin(n_hat, n_hat_margin);
  }

  n_hat_out[idx] = n_hat;
  n_hat_raw_out[idx] = nh_raw;
  Te_out[idx] = Te_c;
  Z_out[idx] = z_c;
}

__device__ inline void compute_gradient_kernel_body(
    const int idx,
    double* __restrict__ grad_R,
    double* __restrict__ grad_Z,
    const double* __restrict__ n_hat,
    const double* __restrict__ node_R,
    const double* __restrict__ node_Z,
    const int n_nodes_r,
    const int n_nodes_z) {
  const int i = idx / n_nodes_z;
  const int j = idx - i * n_nodes_z;

  auto node = [&](const int ii, const int jj) -> double {
    return n_hat[ii * n_nodes_z + jj];
  };

  double dndR = 0.0;
  if (i == 0) {
    dndR = 0.0;
  } else if (i == n_nodes_r - 1) {
    const double dR = node_R[i] - node_R[i - 1];
    dndR = (dR > 0.0) ? (node(i, j) - node(i - 1, j)) / dR : 0.0;
  } else {
    const double h_m = node_R[i] - node_R[i - 1];
    const double h_p = node_R[i + 1] - node_R[i];
    const double denom = h_m * h_p * (h_m + h_p);
    if (denom > 0.0) {
      dndR = (-h_p * h_p * node(i - 1, j) + (h_p * h_p - h_m * h_m) * node(i, j) +
              h_m * h_m * node(i + 1, j)) /
             denom;
    }
  }

  double dndZ = 0.0;
  if (j == 0) {
    const double dZ = node_Z[1] - node_Z[0];
    dndZ = (dZ > 0.0) ? (node(i, 1) - node(i, 0)) / dZ : 0.0;
  } else if (j == n_nodes_z - 1) {
    const double dZ = node_Z[j] - node_Z[j - 1];
    dndZ = (dZ > 0.0) ? (node(i, j) - node(i, j - 1)) / dZ : 0.0;
  } else {
    const double h_m = node_Z[j] - node_Z[j - 1];
    const double h_p = node_Z[j + 1] - node_Z[j];
    const double denom = h_m * h_p * (h_m + h_p);
    if (denom > 0.0) {
      dndZ = (-h_p * h_p * node(i, j - 1) + (h_p * h_p - h_m * h_m) * node(i, j) +
              h_m * h_m * node(i, j + 1)) /
             denom;
    }
  }

  if (!isfinite(dndR)) {
    dndR = 0.0;
  }
  if (!isfinite(dndZ)) {
    dndZ = 0.0;
  }

  grad_R[idx] = dndR;
  grad_Z[idx] = dndZ;
}

__device__ inline void compute_smooth_kappa_kernel_body(
    const int idx,
    double* __restrict__ smooth_kappa_factor,
    const double* __restrict__ n_hat,
    const double* __restrict__ T_e,
    const double* __restrict__ Zbar,
    const double lambda_cm,
    const double eps_n,
    const double coulomb_log_floor,
    const int n_nodes_total) {
  smooth_kappa_factor[idx] = compute_kappa_smooth_factor(
      n_hat[idx], T_e[idx], Zbar[idx], lambda_cm, eps_n, coulomb_log_floor);
}

__device__ inline void compute_smooth_kappa_ext_kernel_body(
    const int idx,
    double* __restrict__ smooth_kappa_factor,
    const double* __restrict__ n_hat,
    const double* __restrict__ T_e,
    const double* __restrict__ Zbar,
    const double lambda_cm,
    const double eps_n,
    const double coulomb_log_floor,
    const int n_nodes_total,
    const LaserPhysExtOptions opt,
    const int* __restrict__ node_material = nullptr) {
  if (idx >= n_nodes_total) {
    return;
  }
  // Multi-material decks: the node's material's collision-charge model.
  if (opt.zeff_materials != nullptr && node_material != nullptr) {
    smooth_kappa_factor[idx] = compute_kappa_smooth_factor_ext(
        with_material_zeff(opt, node_material[idx]), n_hat[idx], T_e[idx], Zbar[idx],
        lambda_cm, eps_n, coulomb_log_floor, /*I_wcm2=*/-1.0);
    return;
  }
  smooth_kappa_factor[idx] = compute_kappa_smooth_factor_ext(
      opt, n_hat[idx], T_e[idx], Zbar[idx], lambda_cm, eps_n,
      coulomb_log_floor, /*I_wcm2=*/-1.0);
}

// Material and Langdon collision charge of each node of the 1D laser mesh
// (multi-material decks, 2026-09-24): the material of the hydro cell whose
// values the node takes (map_hydro_to_laser_1d_kernel_body), the outer
// surface cell's for a ghost-corona node, -1 without material. The Langdon
// charge is the material's representative value, or the node's Zbar when
// it has none.
__device__ inline void map_node_material_1d_kernel_body(
    const int idx,
    const double* __restrict__ node_R,
    const double* __restrict__ node_Z,
    const double* __restrict__ r_edges,
    const std::uint8_t* __restrict__ cell_is_void,
    const int* __restrict__ cell_material_index,
    const double* __restrict__ node_Zbar,
    const int n_nodes_z,
    const int n_cells,
    const int use_ghost_corona,
    const int outer_surface_cell,
    const double r_surface_outer,
    const double r_ghost_outer,
    const LaserZeffMaterial* __restrict__ zeff_materials,
    const int n_materials,
    int* __restrict__ node_material_out,
    double* __restrict__ node_zcoll_out) {
  const int i = idx / n_nodes_z;
  const int j = idx - i * n_nodes_z;
  const double R = node_R[i];
  const double Z = node_Z[j];
  const double r = sqrt(R * R + Z * Z);
  const int c = locate_cell_1d_device(r_edges, n_cells, r);
  const bool is_void = (cell_is_void[c] != 0U);
  const bool outside_surface = (outer_surface_cell >= 0 && r > r_surface_outer);
  const bool outer_hydro_clamp = outside_surface && !is_void;
  int m = -1;
  if (!is_void && !outer_hydro_clamp) {
    m = cell_material_index[c];
  } else if (outer_surface_cell >= 0 &&
             ((use_ghost_corona != 0 && is_void && r >= r_surface_outer &&
               r <= r_ghost_outer) ||
              (outer_hydro_clamp && use_ghost_corona != 0 && r <= r_ghost_outer))) {
    m = cell_material_index[outer_surface_cell];
  }
  if (m >= n_materials) {
    m = n_materials - 1;
  }
  node_material_out[idx] = m;
  const double zc = (m >= 0 && zeff_materials != nullptr) ? zeff_materials[m].langdon_zcoll : 0.0;
  node_zcoll_out[idx] = (zc > 0.0) ? zc : fmax(node_Zbar[idx], 0.0);
}

__device__ inline void extract_radial_profile_kernel_body(
    const int i,
    double* __restrict__ radial_node_r,
    double* __restrict__ radial_n_hat,
    double* __restrict__ radial_n_hat_raw,
    double* __restrict__ radial_smooth_kappa,
    const double* __restrict__ node_R,
    const double* __restrict__ n_hat,
    const double* __restrict__ n_hat_raw,
    const double* __restrict__ smooth_kappa_factor,
    const int n_nodes_r,
    const int n_nodes_z,
    const int j_center,
    const int copy_smooth) {
  const int n = i * n_nodes_z + j_center;
  radial_node_r[i] = node_R[i];
  radial_n_hat[i] = n_hat[n];
  radial_n_hat_raw[i] = n_hat_raw[n];
  radial_smooth_kappa[i] = (copy_smooth != 0 && smooth_kappa_factor != nullptr)
                               ? smooth_kappa_factor[n]
                               : 0.0;
}

__device__ inline void extract_radial_te_kernel_body(
    const int i,
    double* __restrict__ radial_T_e,
    const double* __restrict__ T_e,
    const int n_nodes_z,
    const int j_center,
    const int n_nodes_r) {
  if (i >= n_nodes_r) {
    return;
  }
  radial_T_e[i] = T_e[i * n_nodes_z + j_center];
}

__device__ inline void compute_radial_gradient_kernel_body(
    const int i,
    double* __restrict__ radial_dn_dr,
    const double* __restrict__ radial_node_r,
    const double* __restrict__ radial_n_hat,
    const int n_nodes_r) {
  double grad = 0.0;
  if (n_nodes_r > 1) {
    // Consistency contract: the force is the exact derivative of the
    // piecewise-linear radial_n_hat interpolant, so linear profiles are exact
    // on any node layout. The last node copies the final segment slope.
    const int segment_i = (i < n_nodes_r - 1) ? i : n_nodes_r - 2;
    const double dr =
        radial_node_r[segment_i + 1] - radial_node_r[segment_i];
    if (dr > 0.0) {
      grad =
          (radial_n_hat[segment_i + 1] - radial_n_hat[segment_i]) / dr;
    }
  }

  radial_dn_dr[i] = isfinite(grad) ? grad : 0.0;
}

}  // namespace tenryu::laser::laser_mesh_bodies
