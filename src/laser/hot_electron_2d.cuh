#pragma once

#include "laser/hot_electron_1d.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::laser::hot_electron {

inline constexpr int kChiBins = 16;
inline constexpr double kPhiClassEdge = 0.3;

struct MeshView2D {
  int n_cells = 0;
  const int* cell_nodes = nullptr;
  const int* cell_neighbor = nullptr;
  const double* node_R = nullptr;
  const double* node_Z = nullptr;
};

struct MeshView2DStorage {
  std::vector<int> cell_nodes;
  std::vector<int> cell_neighbor;

  MeshView2D view(const double* node_R, const double* node_Z, int n_cells) const;
};

MeshView2DStorage build_single_block_view(int nr, int nz);

// Multiblock view: node slots normalized to ccw signed area in the (R,Z)
// plane; triangle cells stored as degenerate quads [n0,n1,n2,n0]; neighbors
// resolved from the topology's unique_internal_faces/boundary_faces tables
// (NOT node-id equality across seams). Fails hard on any face that resolves
// to neither an internal link nor a tagged boundary.
// cell_nverts: Mesh::cell_nverts channel (nullptr => all quads).
MeshView2DStorage build_multiblock_view(const tenryu::mesh::MeshTopology& topo,
                                        const double* node_R,
                                        const double* node_Z,
                                        const std::uint8_t* cell_nverts);

struct DirNode {
  double om[3] = {0.0, 0.0, 1.0};
  double weight = 0.0;
};

std::vector<DirNode> build_band_dirs_3d(const double axis[3], double mu_lo,
                                        double mu_hi, int n_mu, int n_phi);

namespace detail {
inline constexpr double kGeomTiny = 1.0e-300;

TENRYU_HOTE_HOST_DEVICE inline double absd(const double x) {
  return (x < 0.0) ? -x : x;
}

TENRYU_HOTE_HOST_DEVICE inline double maxd(const double a, const double b) {
  return (a > b) ? a : b;
}

TENRYU_HOTE_HOST_DEVICE inline double max3d(const double a, const double b,
                                            const double c) {
  return maxd(maxd(a, b), c);
}

// Deterministic contraction: both host (std::fma) and device (fma intrinsic)
// round identically, making crossing decisions bit-identical across
// compilers regardless of automatic FMA contraction settings.
TENRYU_HOTE_HOST_DEVICE inline double fma_d(const double a, const double b,
                                            const double c) {
  return fma(a, b, c);
}

TENRYU_HOTE_HOST_DEVICE inline double chord_q(const double qa, const double qb,
                                              const double qc, const double s) {
  const double q = fma_d(fma_d(qa, s, qb), s, qc);
  return (q > 0.0) ? q : 0.0;
}

TENRYU_HOTE_HOST_DEVICE inline double cross_face_2d(const MeshView2D& v,
                                                    const int c, const int f,
                                                    const double R,
                                                    const double Z) {
  const int ia = v.cell_nodes[c * 4 + f];
  const int ib = v.cell_nodes[c * 4 + ((f + 1) & 3)];
  const double RA = v.node_R[ia];
  const double ZA = v.node_Z[ia];
  const double RB = v.node_R[ib];
  const double ZB = v.node_Z[ib];
  return fma_d(RB - RA, Z - ZA, -((ZB - ZA) * (R - RA)));
}

TENRYU_HOTE_HOST_DEVICE inline int solve_quadratic_sorted(
    const double a, const double b, const double c, double roots[2]) {
  if (absd(a) < kGeomTiny) {
    if (absd(b) < kGeomTiny) {
      return 0;
    }
    roots[0] = -c / b;
    return 1;
  }
  const double four_ac = 4.0 * a * c;
  double disc = fma_d(b, b, -four_ac);
  if (disc < 0.0) {
    const double scale = absd(b * b) + absd(four_ac) + kGeomTiny;
    if (disc < -1.0e-14 * scale) {
      return 0;
    }
    disc = 0.0;
  }
  const double sd = sqrt(disc);
  if (sd == 0.0) {
    roots[0] = -0.5 * b / a;
    return 1;
  }
  const double q = -0.5 * (b + ((b >= 0.0) ? sd : -sd));
  if (q == 0.0) {
    roots[0] = (-b - sd) / (2.0 * a);
    roots[1] = (-b + sd) / (2.0 * a);
  } else {
    roots[0] = q / a;
    roots[1] = c / q;
  }
  if (roots[1] < roots[0]) {
    const double tmp = roots[0];
    roots[0] = roots[1];
    roots[1] = tmp;
  }
  return 2;
}

TENRYU_HOTE_HOST_DEVICE inline void append_crossing(const double s, int& n,
                                                    double s_out[2]) {
  if (n == 0) {
    s_out[0] = s;
    n = 1;
    return;
  }
  if (absd(s - s_out[n - 1]) <= 1.0e-14 * maxd(absd(s), 1.0)) {
    return;
  }
  if (n == 1) {
    if (s < s_out[0]) {
      s_out[1] = s_out[0];
      s_out[0] = s;
    } else {
      s_out[1] = s;
    }
    n = 2;
  }
}
}  // namespace detail

TENRYU_HOTE_HOST_DEVICE inline bool point_in_cell_2d(const MeshView2D& v,
                                                     const int c,
                                                     const double R,
                                                     const double Z) {
  for (int f = 0; f < 4; ++f) {
    const int ia = v.cell_nodes[c * 4 + f];
    const int ib = v.cell_nodes[c * 4 + ((f + 1) & 3)];
    const double RA = v.node_R[ia];
    const double ZA = v.node_Z[ia];
    const double RB = v.node_R[ib];
    const double ZB = v.node_Z[ib];
    const double s =
        detail::fma_d(RB - RA, Z - ZA, -((ZB - ZA) * (R - RA)));
    const double edge_scale =
        detail::max3d(detail::absd(RB - RA), detail::absd(ZB - ZA),
                      detail::kGeomTiny);
    if (s < -1.0e-12 * edge_scale) {
      return false;
    }
  }
  return true;
}

TENRYU_HOTE_HOST_DEVICE inline int locate_cell_2d(const MeshView2D& v,
                                                  const double R,
                                                  const double Z,
                                                  const int guess) {
  if (v.n_cells <= 0) {
    return -1;
  }
  int c = guess;
  if (c < 0) {
    c = 0;
  }
  if (c >= v.n_cells) {
    c = v.n_cells - 1;
  }
  for (int step = 0; step < v.n_cells; ++step) {
    if (point_in_cell_2d(v, c, R, Z)) {
      return c;
    }
    int best_f = -1;
    double best_s = 0.0;
    for (int f = 0; f < 4; ++f) {
      const double s = detail::cross_face_2d(v, c, f, R, Z);
      const int n = v.cell_neighbor[c * 4 + f];
      if (s < best_s && n != -1) {
        best_s = s;
        best_f = f;
      }
    }
    if (best_f < 0) {
      return -1;
    }
    c = v.cell_neighbor[c * 4 + best_f];
  }
  return -1;
}

struct Chord3D {
  double R0 = 0.0;
  double Z0 = 0.0;
  double om[3] = {0.0, 0.0, 1.0};
  double qa = 0.0;
  double qb = 0.0;
  double qc = 0.0;

  TENRYU_HOTE_HOST_DEVICE Chord3D() = default;

  TENRYU_HOTE_HOST_DEVICE Chord3D(const double R0_in, const double Z0_in,
                                  const double om_in[3])
      : R0(R0_in), Z0(Z0_in) {
    const double n2 =
        om_in[0] * om_in[0] + om_in[1] * om_in[1] + om_in[2] * om_in[2];
    if (n2 > 0.0) {
      const double inv_n = 1.0 / sqrt(n2);
      om[0] = om_in[0] * inv_n;
      om[1] = om_in[1] * inv_n;
      om[2] = om_in[2] * inv_n;
    }
    qa = detail::fma_d(om[0], om[0], om[1] * om[1]);
    qb = 2.0 * R0 * om[0];
    qc = R0 * R0;
  }
};

TENRYU_HOTE_HOST_DEVICE inline int face_crossings(
    const Chord3D& ch, const double RA, const double ZA, const double RB,
    const double ZB, const double s_min, const double eps_s, double s_out[2]) {
  const double dR = RB - RA;
  const double dZ = ZB - ZA;
  int n_out = 0;
  if (detail::absd(dR) < detail::kGeomTiny &&
      detail::absd(dZ) < detail::kGeomTiny) {
    return 0;
  }

  if (detail::absd(dZ) < detail::kGeomTiny) {
    if (detail::absd(dR) < detail::kGeomTiny ||
        detail::absd(ch.om[2]) < detail::kGeomTiny) {
      return 0;
    }
    const double s = (ZA - ch.Z0) / ch.om[2];
    if (s > s_min + eps_s) {
      const double r = sqrt(detail::chord_q(ch.qa, ch.qb, ch.qc, s));
      const double t = (r - RA) / dR;
      if (t >= -1.0e-12 && t <= 1.0 + 1.0e-12) {
        s_out[0] = s;
        return 1;
      }
    }
    return 0;
  }

  if (detail::absd(dR) < detail::kGeomTiny) {
    double roots[2] = {0.0, 0.0};
    const int nr = detail::solve_quadratic_sorted(
        ch.qa, ch.qb, detail::fma_d(-RA, RA, ch.qc), roots);
    for (int i = 0; i < nr; ++i) {
      const double s = roots[i];
      if (s <= s_min + eps_s) {
        continue;
      }
      const double z = detail::fma_d(s, ch.om[2], ch.Z0);
      const double t = (z - ZA) / dZ;
      if (t >= -1.0e-12 && t <= 1.0 + 1.0e-12) {
        detail::append_crossing(s, n_out, s_out);
      }
    }
    return n_out;
  }

  const double C0 = detail::fma_d(RA, dZ, (ch.Z0 - ZA) * dR);
  const double C1 = ch.om[2] * dR;
  const double dZ2 = dZ * dZ;
  const double a = detail::fma_d(ch.qa, dZ2, -(C1 * C1));
  const double b = detail::fma_d(ch.qb, dZ2, -(2.0 * C0 * C1));
  const double c = detail::fma_d(ch.qc, dZ2, -(C0 * C0));
  double roots[2] = {0.0, 0.0};
  const int nr = detail::solve_quadratic_sorted(a, b, c, roots);
  for (int i = 0; i < nr; ++i) {
    const double s = roots[i];
    if (s <= s_min + eps_s) {
      continue;
    }
    const double z = detail::fma_d(s, ch.om[2], ch.Z0);
    const double r = sqrt(detail::chord_q(ch.qa, ch.qb, ch.qc, s));
    const double L = detail::fma_d(C1, s, C0);
    if (L * dZ < 0.0) {
      continue;
    }
    const double residual = detail::absd(detail::fma_d(r, dZ, -L));
    const double scale =
        detail::absd(RA) + detail::absd(RB) + detail::absd(z - ZA) + 1.0e-30;
    if (residual > 1.0e-9 * scale) {
      continue;
    }
    const double denom = dZ * dZ + dR * dR;
    const double t = ((z - ZA) * dZ + (r - RA) * dR) / denom;
    if (t >= -1.0e-12 && t <= 1.0 + 1.0e-12) {
      detail::append_crossing(s, n_out, s_out);
    }
  }
  return n_out;
}

enum class WalkStatus : int { Segment = 0, Escaped = 1, Stuck = 2 };

struct ChordWalkerRZ {
  MeshView2D v;
  Chord3D ch;
  int cell = -1;
  double s_cur = 0.0;
  int segments = 0;
  int max_segments = 0;
  double eps_s = 0.0;
  int relocate_attempts = 0;
  bool done = false;
  bool exited_boundary = false;

  TENRYU_HOTE_HOST_DEVICE ChordWalkerRZ(const MeshView2D& view,
                                        const Chord3D& chord,
                                        const int start_cell,
                                        const double mesh_scale,
                                        const int max_segments_in)
      : v(view), ch(chord), cell(start_cell), max_segments(max_segments_in) {
    eps_s = 1.0e-12 * ((mesh_scale > 0.0) ? mesh_scale : 1.0);
    if (max_segments < 1) {
      max_segments = 1;
    }
    done = (cell < 0 || cell >= v.n_cells || v.n_cells <= 0);
  }

  TENRYU_HOTE_HOST_DEVICE WalkStatus next(double& ds, int& cell_out) {
    exited_boundary = false;
    if (done) {
      return WalkStatus::Escaped;
    }
    for (;;) {
      double best_s = 0.0;
      int best_f = -1;
      bool found = false;
      for (int f = 0; f < 4; ++f) {
        const int ia = v.cell_nodes[cell * 4 + f];
        const int ib = v.cell_nodes[cell * 4 + ((f + 1) & 3)];
        const double RA = v.node_R[ia];
        const double ZA = v.node_Z[ia];
        const double RB = v.node_R[ib];
        const double ZB = v.node_Z[ib];
        const int neighbor = v.cell_neighbor[cell * 4 + f];
        if (neighbor == -1 && detail::absd(RA) < detail::kGeomTiny &&
            detail::absd(RB) < detail::kGeomTiny) {
          continue;
        }
        double roots[2] = {0.0, 0.0};
        const int nr = face_crossings(ch, RA, ZA, RB, ZB, s_cur, eps_s, roots);
        for (int i = 0; i < nr; ++i) {
          const double s = roots[i];
          if (!found || s < best_s ||
              (s == best_s && (best_f < 0 || f < best_f))) {
            best_s = s;
            best_f = f;
            found = true;
          }
        }
      }
      if (found) {
        const int old_cell = cell;
        ds = best_s - s_cur;
        cell_out = old_cell;
        s_cur = best_s;
        ++segments;
        if (segments > max_segments) {
          done = true;
          return WalkStatus::Stuck;
        }
        const int neighbor = v.cell_neighbor[old_cell * 4 + best_f];
        cell = neighbor;
        if (neighbor == -1) {
          done = true;
          exited_boundary = true;
        }
        return WalkStatus::Segment;
      }
      if (relocate_attempts == 0) {
        ++relocate_attempts;
        s_cur += eps_s;
        const double R = sqrt(detail::chord_q(ch.qa, ch.qb, ch.qc, s_cur));
        const double Z = detail::fma_d(s_cur, ch.om[2], ch.Z0);
        const int relocated = locate_cell_2d(v, R, Z, cell);
        if (relocated >= 0) {
          cell = relocated;
          continue;
        }
      }
      cell_out = cell;
      done = true;
      return WalkStatus::Stuck;
    }
  }
};

struct RayCapture2D {
  double R_s = 0.0;
  double Z_s = 0.0;
  double kR = 0.0;
  double kphi = 0.0;
  double kZ = 1.0;
  double P_hot = 0.0;
};

struct HotESource2D {
  int cell = -1;
  double R_s = 0.0;
  double Z_s = 0.0;
  double k[3] = {0.0, 0.0, 1.0};
  double P_hot = 0.0;
};

struct ReduceResult2D {
  std::vector<HotESource2D> sources;
  double P_locate_failed = 0.0;
  int n_locate_failed = 0;
};

ReduceResult2D reduce_captures_2d(const std::vector<RayCapture2D>& caps,
                                  const MeshView2D& v);

struct DepositResult2D : DepositResult {
  int walker_stuck_terminations = 0;
  int walker_relocations = 0;
};

// max_segments <= 0 uses 8 * max(64, ceil(sqrt(view.n_cells))) walker steps.
DepositResult2D deposit_hot_electrons_cone_2d(
    const HotEChannelSpec& spec,
    const std::vector<HotESource2D>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const MeshView2D& view,
    double mesh_scale,
    std::vector<double>& dep_power_cell,
    int max_segments = 0);

}  // namespace tenryu::laser::hot_electron
